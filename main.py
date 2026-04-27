import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import wandb
from peft import LoraConfig, get_peft_model

from audio_encoder import initialize_pretrained_model
from model import SelfSupervisedAudioAlignmentModel
from dataloader import AudioTextPairDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_combined_dataset(csv_path):
    df = pd.read_csv(csv_path)
    paths, reports = [], []
    for _, row in df.iterrows():
        if os.path.exists(row["audio_path"]):
            paths.append(row["audio_path"])
            reports.append(row["Gen_Report"])
    return paths, reports


def train(model, loader, optimizer, epochs=10, λ_align=1.0, λ_mam=1.0):
    ckpt_root = "/cross_modal_alignment/checkpoints"
    os.makedirs(ckpt_root, exist_ok=True)

    model.train()
    for ep in range(epochs):
        total, comp = 0.0, {"align": 0.0, "mam": 0.0}

        for batch in tqdm(loader, desc=f"{ep+1}/{epochs}"):
            audio = batch["audio"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            optimizer.zero_grad()
            losses = model(audio, ids, mask)
            loss = λ_align * losses["alignment_loss"] + λ_mam * losses["mam_loss"]
            loss.backward()
            optimizer.step()

            total += loss.item()
            comp["align"] += losses["alignment_loss"].item()
            comp["mam"] += losses["mam_loss"].item()
            wandb.log({"train_loss_step": loss.item()})

        wandb.log(
            {
                "train_loss": total / len(loader),
                "alignment_loss": comp["align"] / len(loader),
                "mam_loss": comp["mam"] / len(loader),
                "epoch": ep + 1,
            }
        )

        # ── save a checkpoint every 10 epochs ────────────────────────────────
        if (ep + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": ep + 1,
                    "audio_model_state_dict": model.audio_model.state_dict(),
                    "language_model_state_dict": model.language_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(ckpt_root, f"checkpoint_epoch_{ep+1}.pt"),
            )
    return model


if __name__ == "__main__":
    wandb.init(project="my-awesome-project", name="opera-llama-lora", config={"epochs": 40, "batch": 24})

    audio_paths, reports = load_combined_dataset("/audiocraft/combined_dataset.csv")

    audio_model = initialize_pretrained_model(pretrain="operaGT")
    ckpt = torch.load("/cross_modal_alignment/encoder-operaGT.ckpt", map_location=DEVICE)
    audio_model.load_state_dict(ckpt["state_dict"], strict=False)

    llm_type = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(llm_type)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = AudioTextPairDataset(audio_paths, reports, tokenizer)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=4)

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_type, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    llm_model = get_peft_model(llm_model, lora_cfg)

    alignment_model = SelfSupervisedAudioAlignmentModel(
        audio_model=audio_model, language_model=llm_model, alignment_layer=-1, compute_alignment=True
    ).to(DEVICE)

    for n, p in alignment_model.language_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    lora_params = [p for n, p in alignment_model.named_parameters() if p.requires_grad and "lora_" in n]
    other_params = [p for n, p in alignment_model.named_parameters() if p.requires_grad and "lora_" not in n]

    optimizer = optim.AdamW(
        [{"params": other_params, "lr": 1e-5}, {"params": lora_params, "lr": 2e-4}]
    )

    trained = train(alignment_model, dataloader, optimizer, epochs=40, λ_align=1.0, λ_mam=1.0)

    torch.save(trained.audio_model.state_dict(), "/cross_modal_alignment/checkpoints/opera_lora_audio.pt")
    trained.language_model.save_pretrained("/cross_modal_alignment/checkpoints/llama_lora_adapters")

    wandb.finish()
