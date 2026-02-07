"""
Frame Delta Inference Script
Reproduces the exact test split from gold training, runs predictions,
and exports both discrete classifications and continuous frame measures.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForSequenceClassification, LongformerTokenizerFast

# ── A) Configuration & Constants ──────────────────────────────────────────────

MODEL_DIR = "final_model/best_model_gold_feb_26"
DATA_PATH = "data/gold_train_data.parquet"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_predictions.csv")

RANDOM_SEED = 42
TEST_SIZE = 0.10
MAX_LENGTH = 2048
BATCH_SIZE = 4

LABELS = [
    "Economic",                                        # 0
    "Capacity and resources",                          # 1
    "Morality",                                        # 2
    "Fairness and equality",                           # 3
    "Legality, constitutionality and jurisprudence",   # 4
    "Policy prescription and evaluation",              # 5
    "Crime and punishment",                            # 6
    "Security and defense",                            # 7
    "Health and safety",                               # 8
    "Quality of life",                                 # 9
    "Cultural identity",                               # 10
    "Public opinion",                                  # 11
    "Political",                                       # 12
    "External regulation and reputation",              # 13
    "Other",                                           # 14
]

NUM_LABELS = len(LABELS)

# Short names for CSV column headers
LABEL_SHORT = [
    "economic", "capacity", "morality", "fairness", "legality",
    "policy", "crime", "security", "health", "quality_of_life",
    "cultural", "public_opinion", "political", "external_reg", "other",
]


# ── B) Reproduce the exact test split ────────────────────────────────────────

def load_and_split():
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    return test_df.reset_index(drop=True)


# ── C) Dataset class ─────────────────────────────────────────────────────────

class GoldFramingDataset(Dataset):
    def __init__(self, texts, labels_list, tokenizer, max_length, num_labels=15):
        self.texts = texts
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels_list[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        labels = torch.zeros(self.num_labels)
        for li in label_indices:
            if li < self.num_labels:
                labels[li] = 1.0

        global_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        global_attention_mask[0] = 1  # CLS token
        global_attention_mask[3] = 1  # Topic token after "TOPIC:"

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "global_attention_mask": global_attention_mask,
            "labels": labels,
        }


# ── D) Run inference ─────────────────────────────────────────────────────────

def run_inference(test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_DIR)
    model = LongformerForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    )
    model.to(device)
    model.eval()

    # Load thresholds
    thresholds_path = os.path.join(MODEL_DIR, "optimized_thresholds.json")
    with open(thresholds_path) as f:
        thresholds_raw = json.load(f)
    thresholds = np.array([thresholds_raw[label]["threshold"] for label in LABELS])
    print(f"Thresholds loaded: {dict(zip(LABEL_SHORT, thresholds))}")

    # Build dataset and dataloader
    dataset = GoldFramingDataset(
        texts=test_df["text"].tolist(),
        labels_list=test_df["labels_idx"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        num_labels=NUM_LABELS,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            global_attention_mask = batch["global_attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"  Batch {i + 1}/{len(loader)}")

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_logits, all_labels


# ── E) Build output DataFrame ────────────────────────────────────────────────

def build_output(test_df, all_logits, all_labels, thresholds_path):
    # Load thresholds
    with open(thresholds_path) as f:
        thresholds_raw = json.load(f)
    thresholds = np.array([thresholds_raw[label]["threshold"] for label in LABELS])

    # Compute sigmoid and softmax
    sigmoid = 1 / (1 + np.exp(-all_logits))
    softmax = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)

    # Binary predictions using per-class thresholds
    preds = (sigmoid >= thresholds).astype(int)

    # Start building output
    out = pd.DataFrame({
        "id": test_df["id"].values,
        "source": test_df["source"].values,
        "article_id": test_df["article_id"].values,
        "text_preview": test_df["text"].str[:200].values,
    })

    for i, short in enumerate(LABEL_SHORT):
        out[f"{short}_label_true"] = all_labels[:, i].astype(int)
        out[f"{short}_label_pred"] = preds[:, i]
        out[f"{short}_sigmoid"] = np.round(sigmoid[:, i], 4)
        out[f"{short}_normalized"] = np.round(softmax[:, i], 4)

    return out


# ── F) Main ──────────────────────────────────────────────────────────────────

def main():
    print("=== Frame Delta Inference ===\n")

    test_df = load_and_split()

    all_logits, all_labels = run_inference(test_df)

    thresholds_path = os.path.join(MODEL_DIR, "optimized_thresholds.json")
    out = build_output(test_df, all_logits, all_labels, thresholds_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved predictions to {OUTPUT_FILE}")
    print(f"Shape: {out.shape}")

    # Quick sanity checks
    norm_cols = [c for c in out.columns if c.endswith("_normalized")]
    norm_sums = out[norm_cols].sum(axis=1)
    print(f"Normalized sums -- min: {norm_sums.min():.4f}, max: {norm_sums.max():.4f}, mean: {norm_sums.mean():.4f}")

    sig_cols = [c for c in out.columns if c.endswith("_sigmoid")]
    sig_vals = out[sig_cols].values
    print(f"Sigmoid range -- min: {sig_vals.min():.4f}, max: {sig_vals.max():.4f}")

    # Classification report (excluding Other)
    from sklearn.metrics import classification_report
    true_cols = [c for c in out.columns if c.endswith("_label_true") and not c.startswith("other")]
    pred_cols = [c for c in out.columns if c.endswith("_label_pred") and not c.startswith("other")]
    label_names = [s for s in LABEL_SHORT if s != "other"]

    y_true = out[true_cols].values
    y_pred = out[pred_cols].values
    print("\n=== Classification Report (excl. Other) ===")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    main()
