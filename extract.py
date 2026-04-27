import torch
import pandas as pd
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import os
import gc

csv_path = "/home/twang/cross_modal_alignment/datasets/combined.csv"
model_id = "google/medgemma-4b-pt"
out_file = "/home/twang/cross_modal_alignment/datasets/report_embeddings.pt"
batch_size = 8
max_length = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print(f"=== MedGemma Embedding Extraction ===")
print(f"Device: {device}")
print(f"Model: {model_id}")
print(f"CSV: {csv_path}")
print(f"Output: {out_file}")
print(f"Batch size: {batch_size}")

if os.path.exists(out_file):
    print(f"WARNING: Output file {out_file} already exists!")
    response = input("Do you want to overwrite it? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_file.replace('.pt', f'_{timestamp}.pt')
        print(f"Using new filename: {out_file}")

try:
    print("Loading model and processor...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model loaded successfully on {device}")

    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Total reports to process: {total_rows}")

    if "Gen_Report" not in df.columns:
        raise ValueError(f"'Gen_Report' column not found! Available columns: {list(df.columns)}")

    missing_reports = df["Gen_Report"].isna().sum()
    if missing_reports > 0:
        print(f"WARNING: {missing_reports} missing reports found. These will be skipped.")
        df = df.dropna(subset=["Gen_Report"])
        print(f"Processing {len(df)} valid reports.")

    reports = df["Gen_Report"].astype(str).tolist()

    print(f"\nSample reports:")
    for i, report in enumerate(reports[:3]):
        print(f"  {i+1}. {report[:100]}...")

    num_batches = (len(reports) + batch_size - 1) // batch_size
    print(f"\nProcessing in {num_batches} batches of size {batch_size}")

    all_embeddings = []
    failed_indices = []

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(reports))
            batch_reports = reports[start_idx:end_idx]

            batch_embeddings = []

            for local_idx, txt in enumerate(batch_reports):
                global_idx = start_idx + local_idx
                try:
                    tokens = proc(
                        text=str(txt),
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=False
                    ).to(device, dtype=dtype)

                    outputs = model(**tokens, output_hidden_states=True)
                    h = outputs.hidden_states[-1][:, -1]

                    h_norm = torch.nn.functional.normalize(h.float(), p=2, dim=-1)
                    h_cpu = h_norm.squeeze(0).cpu()

                    batch_embeddings.append(h_cpu)

                except Exception as e:
                    print(f"\nError processing report {global_idx}: {str(e)[:100]}")
                    failed_indices.append(global_idx)
                    dummy_emb = torch.zeros(model.config.hidden_size, dtype=torch.float32)
                    batch_embeddings.append(dummy_emb)

            all_embeddings.extend(batch_embeddings)

            if torch.cuda.is_available() and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    print("\nStacking embeddings...")
    embeddings_tensor = torch.stack(all_embeddings)

    print(f"Saving embeddings to {out_file}...")
    torch.save({
        'embeddings': embeddings_tensor,
        'model_id': model_id,
        'total_reports': len(reports),
        'failed_indices': failed_indices,
        'embedding_dim': embeddings_tensor.shape[1],
        'csv_path': csv_path
    }, out_file)

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE!")
    print("=" * 50)
    print(f" Total reports processed: {len(reports)}")
    print(f" Embeddings shape: {embeddings_tensor.shape}")
    print(f" Embedding dimension: {embeddings_tensor.shape[1]}")
    print(f" Failed extractions: {len(failed_indices)}")
    print(f" Success rate: {((len(reports) - len(failed_indices)) / len(reports) * 100):.1f}%")
    print(f" Output file: {out_file}")
    print(f" File size: {os.path.getsize(out_file) / (1024 ** 2):.1f} MB")

    if failed_indices:
        print(f"\n⚠ Failed indices: {failed_indices[:10]}")
        if len(failed_indices) > 10:
            print(f"   ... and {len(failed_indices) - 10} more")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'model' in locals():
        del model
    if 'proc' in locals():
        del proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\nGPU memory cleared.")

print("\nDone!")

