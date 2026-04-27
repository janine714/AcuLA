import os
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from audio_encoder import initialize_pretrained_model
from model import SelfSupervisedAudioAlignmentModel
from dataloader import AudioTextPairDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AcuLa: Audio–Clinical Understanding via Language Alignment"
    )

    # Data / checkpoint paths
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV containing audio_path and Gen_Report columns.")
    parser.add_argument("--audio_ckpt", type=str, required=True,
                        help="Path to pretrained audio encoder checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save AcuLa checkpoints.")

    # Model configuration
    parser.add_argument("--audio_backbone", type=str, default="operaGT",
                        choices=["operaGT", "operaCT", "operaCE", "ast", "clap"],
                        help="Pretrained audio backbone to initialize.")
    parser.add_argument("--llm_type", type=str, default="google/medgemma-4b-pt",
                        help="Language model used as the semantic teacher.")
    parser.add_argument("--alignment_layer", type=int, default=-1,
                        help="Language-model hidden layer used for alignment.")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--lambda_mam", type=float, default=1.0)
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--audio_input_sec", type=int, default=8)

    # Logging / saving
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AcuLa")
    parser.add_argument("--wandb_run_name", type=str, default="acula-operaGT-medgemma")

    return parser.parse_args()


def load_combined_dataset(csv_path):
    df = pd.read_csv(csv_path)

    required_columns = {"audio_path", "Gen_Report"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    paths, reports = [], []

    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        report = row["Gen_Report"]

        if not isinstance(report, str) or len(report.strip()) == 0:
            continue

        if os.path.exists(audio_path):
            paths.append(audio_path)
            reports.append(report.strip())

    if len(paths) == 0:
        raise RuntimeError("No valid audio-report pairs were found.")

    print(f"[Data] Loaded {len(paths)} valid audio-report pairs.")
    return paths, reports


def load_audio_encoder(backbone, checkpoint_path, device):
    print(f"[Audio] Initializing audio backbone: {backbone}")
    audio_model = initialize_pretrained_model(pretrain=backbone)

    if isinstance(audio_model, tuple):
        raise NotImplementedError(
            "This training script expects an audio encoder module. "
            "For CLAP-style models returning (model, processor), please adapt the wrapper first."
        )

    print(f"[Audio] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "audio_model_state_dict" in ckpt:
        state_dict = ckpt["audio_model_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = audio_model.load_state_dict(state_dict, strict=False)
    print(f"[Audio] Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    return audio_model


def load_language_teacher(llm_type):
    print(f"[Text] Loading language teacher: {llm_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        llm_type,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    language_model = AutoModelForCausalLM.from_pretrained(
        llm_type,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    language_model.eval()

    for param in language_model.parameters():
        param.requires_grad = False

    print("[Text] Language teacher loaded and frozen.")
    return tokenizer, language_model


def save_checkpoint(model, optimizer, scheduler, epoch, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "audio_model_state_dict": model.audio_model.state_dict(),
            "audio_projection_mlp_state_dict": model.audio_projection_mlp.state_dict(),
            "language_projection_mlp_state_dict": model.language_projection_mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        save_path,
    )

    print(f"[Checkpoint] Saved to {save_path}")


def train(model, loader, optimizer, scheduler, device, args):
    model.train()

    global_step = 0

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_align = 0.0
        total_mam = 0.0

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(progress):
            audio = batch["audio"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            losses = model(audio, input_ids, attention_mask)

            alignment_loss = losses["alignment_loss"]
            mam_loss = losses["mam_loss"]

            loss = args.lambda_align * alignment_loss + args.lambda_mam * mam_loss
            loss = loss / args.grad_accum_steps

            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            step_loss = loss.item() * args.grad_accum_steps
            total_loss += step_loss
            total_align += alignment_loss.item()
            total_mam += mam_loss.item()

            progress.set_postfix(
                {
                    "loss": f"{step_loss:.4f}",
                    "align": f"{alignment_loss.item():.4f}",
                    "mam": f"{mam_loss.item():.4f}",
                }
            )

            if args.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "train/loss_step": step_loss,
                        "train/alignment_loss_step": alignment_loss.item(),
                        "train/mam_loss_step": mam_loss.item(),
                        "train/global_step": global_step,
                    }
                )

        avg_loss = total_loss / len(loader)
        avg_align = total_align / len(loader)
        avg_mam = total_mam / len(loader)

        print(
            f"[Epoch {epoch + 1}] "
            f"loss={avg_loss:.6f} | "
            f"alignment={avg_align:.6f} | "
            f"mam={avg_mam:.6f}"
        )

        if args.use_wandb and wandb is not None:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/alignment_loss_epoch": avg_align,
                    "train/mam_loss_epoch": avg_mam,
                    "train/epoch": epoch + 1,
                }
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                output_dir=args.output_dir,
                filename=f"acula_epoch_{epoch + 1}.pt",
            )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        output_dir=args.output_dir,
        filename="acula_final.pt",
    )

    return model


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using device: {device}")

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or remove --use_wandb.")

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    audio_paths, reports = load_combined_dataset(args.csv_path)

    tokenizer, language_model = load_language_teacher(args.llm_type)

    dataset = AudioTextPairDataset(
        audio_paths=audio_paths,
        text_reports=reports,
        text_tokenizer=tokenizer,
        max_length=args.max_text_length,
        audio_input_sec=args.audio_input_sec,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    audio_model = load_audio_encoder(
        backbone=args.audio_backbone,
        checkpoint_path=args.audio_ckpt,
        device=device,
    )

    model = SelfSupervisedAudioAlignmentModel(
        audio_model=audio_model,
        language_model=language_model,
        alignment_layer=args.alignment_layer,
        compute_alignment=True,
    ).to(device)

    # Keep the language teacher frozen.
    for param in model.language_model.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = max(1, len(loader) // args.grad_accum_steps)
    total_training_steps = num_update_steps_per_epoch * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(f"[Train] Total optimization steps: {total_training_steps}")
    print(f"[Train] Warmup steps: {args.warmup_steps}")
    print(f"[Train] Gradient accumulation steps: {args.grad_accum_steps}")

    trained_model = train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
    )

    final_audio_path = Path(args.output_dir) / "acula_audio_encoder.pt"
    torch.save(trained_model.audio_model.state_dict(), final_audio_path)
    print(f"[Checkpoint] Saved final aligned audio encoder to {final_audio_path}")

    if args.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
