"""
Phase 1: Silver Fine-Tuning for Longformer - RunPod version optimized for A40 (48GB VRAM).

Before running:
1. Install dependencies: pip install wandb iterstrat
2. Run: wandb login (paste API key)
3. Upload data files: silver_articles_part1/2/3.parquet to data/
"""

import os
import gc
import json
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import wandb
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model": "allenai/longformer-base-4096",
    "max_length": 2048,
    "batch_size": 8,
    "grad_accum_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "lr_scheduler": "reduce_on_plateau",
    "lr_patience": 1,  # Reduce LR after 1 epoch without val loss improvement
    "lr_factor": 0.5,  # Multiply LR by 0.5 when reducing
    "loss": "weighted_bce",
    "global_attention": "cls_plus_topic",
    "seed": 42,
    # dataset_size will be added after data loading
}

# Paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
DATA_FILES = [
    "data/silver_articles_part1.parquet",
    "data/silver_articles_part2.parquet",
    "data/silver_articles_part3.parquet",
]
MODEL_SAVE_DIR = f"trained_models/silver-longformer/{TIMESTAMP}"
LOG_DIR = "training_logs/silver-longformer"

# Labels
OFFICIAL_LABELS = [
    "Economic", "Capacity and resources", "Morality", "Fairness and equality",
    "Legality, constitutionality and jurisprudence", "Policy prescription and evaluation",
    "Crime and punishment", "Security and defense", "Health and safety",
    "Quality of life", "Cultural identity", "Public opinion", "Political",
    "External regulation and reputation", "Other"
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and preprocess data from parquet files."""
    dfs = []
    for f in DATA_FILES:
        print(f"Loading {f}...")
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} total rows")

    # Filter: >100 words, drop nulls
    df['num_words'] = df['article_text'].str.split().str.len()
    df = df[df['num_words'] > 100]
    df = df.dropna()
    df = df.drop(columns=['num_words'])
    df = df.reset_index(drop=True)
    print(f"After filtering: {len(df):,} rows")

    # Topic injection: TOPIC:{topic}\n{title}\n{text}
    df['article_text'] = df['title'] + "\n" + df['article_text']
    df['article_text'] = "TOPIC:" + df['gpt_topic'] + "\n" + df['article_text']

    print(f"\nSample text (first 300 chars):")
    print("-" * 50)
    print(df.iloc[0]['article_text'][:300])

    return df


# =============================================================================
# DATASET CLASS
# =============================================================================

class NewsArticleDataset(Dataset):
    """Dataset with on-the-fly tokenization and global attention masks."""

    def __init__(self, df, tokenizer, labels_matrix, max_len=2048, global_attention_mode="cls_plus_topic"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels_matrix
        self.global_attention_mode = global_attention_mode
        print(f"Dataset created: {len(df):,} samples, global_attention_mode='{global_attention_mode}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['article_text'])

        # Tokenize on-the-fly (no padding - collator handles it)
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Global attention mask: 0 = local, 1 = global
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1  # CLS token always gets global attention

        if self.global_attention_mode == "cls_plus_topic":
            # Position 4 is the first token of the topic word
            # (after <s>, TOP, IC, :)
            if len(input_ids) > 4:
                global_attention_mask[4] = 1

        labels_vec = self.labels[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask,
            'labels': torch.tensor(labels_vec, dtype=torch.float)
        }


def longformer_collate_fn(batch, tokenizer):
    """Dynamic padding collator aligned to 512-token windows."""
    max_len = max(len(item['input_ids']) for item in batch)

    # Round up to nearest multiple of 512 (Longformer window size)
    window_size = 512
    padded_len = ((max_len + window_size - 1) // window_size) * window_size

    input_ids_batch = []
    attention_mask_batch = []
    global_attention_mask_batch = []
    labels_batch = []

    pad_token_id = tokenizer.pad_token_id

    for item in batch:
        curr_len = len(item['input_ids'])
        pad_len = padded_len - curr_len

        ids = item['input_ids'] + [pad_token_id] * pad_len
        mask = item['attention_mask'] + [0] * pad_len
        global_mask = item['global_attention_mask'] + [0] * pad_len

        input_ids_batch.append(ids)
        attention_mask_batch.append(mask)
        global_attention_mask_batch.append(global_mask)
        labels_batch.append(item['labels'])

    return {
        'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long),
        'global_attention_mask': torch.tensor(global_attention_mask_batch, dtype=torch.long),
        'labels': torch.stack(labels_batch)
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, scaler, criterion, device, config):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for step, batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast('cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            loss = criterion(outputs.logits, labels)
            loss = loss / config['grad_accum_steps']

        scaler.scale(loss).backward()
        total_loss += loss.item() * config['grad_accum_steps']

        if (step + 1) % config['grad_accum_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': f'{loss.item() * config["grad_accum_steps"]:.4f}'})

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs.logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_loss = total_loss / len(data_loader)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1_micro, f1_macro, all_preds, all_labels


def optimize_thresholds(model, data_loader, device, labels):
    """Grid search for optimal per-class thresholds."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Getting probabilities"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            batch_labels = batch['labels']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            probs = torch.sigmoid(outputs.logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    best_thresholds = {}
    print("\nOptimizing thresholds per class...")
    print("=" * 70)

    for i, label_name in enumerate(labels):
        best_score = 0
        best_thresh = 0.5

        y_true = all_labels[:, i]
        y_score = all_probs[:, i]

        for thresh in np.arange(0.10, 0.95, 0.05):
            y_pred = (y_score > thresh).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        best_thresholds[label_name] = float(best_thresh)
        print(f"{label_name:<50} Threshold: {best_thresh:.2f} (F1: {best_score:.3f})")

    return best_thresholds, all_probs, all_labels


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Run name
    run_name = f"silver-{TIMESTAMP}"
    print(f"\nRun name: {run_name}")
    print(f"Save directory: {MODEL_SAVE_DIR}")

    # Load data first (to get dataset_size for W&B config)
    df = load_data()
    dataset_size = len(df)

    # Add dataset_size to config and initialize W&B
    CONFIG['dataset_size'] = dataset_size
    run = wandb.init(
        entity="ryrousseau-london-school-of-economics-and-political-science",
        project="frame-delta-phase1",
        name=run_name,
        config=CONFIG
    )
    print(f"W&B run initialized: {run.name}")
    print(f"View at: {run.url}")

    # Encode labels
    mlb = MultiLabelBinarizer(classes=OFFICIAL_LABELS)
    labels_matrix = mlb.fit_transform(df['text_generic_frame'])
    print(f"\nLabels shape: {labels_matrix.shape}")

    # Class distribution
    print("\nClass distribution:")
    for i, label in enumerate(OFFICIAL_LABELS):
        count = labels_matrix[:, i].sum()
        print(f"  {label:<45} {count:>6} ({100*count/len(labels_matrix):.1f}%)")

    # Calculate class weights (normalized inverse frequency)
    num_positives = labels_matrix.sum(axis=0)
    inv_freq = 1.0 / (num_positives + 1e-5)
    alpha = inv_freq / inv_freq.sum()
    pos_weight = torch.tensor(alpha * len(OFFICIAL_LABELS), dtype=torch.float).to(device)

    print("\nClass weights (pos_weight):")
    for name, weight in zip(OFFICIAL_LABELS, pos_weight.cpu().numpy()):
        print(f"  {name:<45} {weight:.4f}")

    # Initialize tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained(CONFIG['model'])
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size:,}")

    # Train/Val/Test split (80/10/10)
    N = len(labels_matrix)
    X_indices = np.zeros(N)

    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=CONFIG['seed'])
    train_idx, temp_idx = next(iter(msss1.split(X_indices, labels_matrix)))

    temp_labels = labels_matrix[temp_idx]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=CONFIG['seed'])
    relative_val_idx, relative_test_idx = next(iter(msss2.split(np.zeros(len(temp_idx)), temp_labels)))

    val_idx = temp_idx[relative_val_idx]
    test_idx = temp_idx[relative_test_idx]

    print(f"\nData splits:")
    print(f"  Train: {len(train_idx):,} ({100*len(train_idx)/N:.0f}%)")
    print(f"  Val:   {len(val_idx):,} ({100*len(val_idx)/N:.0f}%)")
    print(f"  Test:  {len(test_idx):,} ({100*len(test_idx)/N:.0f}%)")

    # Create datasets
    full_dataset = NewsArticleDataset(
        df,
        tokenizer,
        labels_matrix,
        max_len=CONFIG['max_length'],
        global_attention_mode=CONFIG['global_attention']
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Create collate function with tokenizer
    def collate_fn(batch):
        return longformer_collate_fn(batch, tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f"\nDataLoader batches:")
    print(f"  Train: {len(train_loader):,}")
    print(f"  Val:   {len(val_loader):,}")
    print(f"  Test:  {len(test_loader):,}")

    # Initialize model
    gc.collect()
    torch.cuda.empty_cache()

    model = LongformerForSequenceClassification.from_pretrained(
        CONFIG['model'],
        num_labels=len(OFFICIAL_LABELS),
        problem_type="multi_label_classification"
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # ReduceLROnPlateau: reduces LR when val_loss stops improving
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize val_loss
        factor=CONFIG['lr_factor'],  # Multiply LR by this factor
        patience=CONFIG['lr_patience'],  # Wait this many epochs before reducing
        verbose=True
    )

    print(f"\nScheduler: ReduceLROnPlateau (patience={CONFIG['lr_patience']}, factor={CONFIG['lr_factor']})")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Save config
    with open(os.path.join(MODEL_SAVE_DIR, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print(f"\nConfig saved to {MODEL_SAVE_DIR}/config.json")

    # Training loop
    best_val_f1 = 0.0
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, CONFIG)

        # Validate
        val_loss, val_f1_micro, val_f1_macro, _, _ = evaluate(model, val_loader, criterion, device)

        # Step scheduler based on validation loss (ReduceLROnPlateau)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to W&B
        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1_micro": val_f1_micro,
            "val_f1_macro": val_f1_macro,
            "learning_rate": current_lr,
        })

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val F1 Micro: {val_f1_micro:.4f}")
        print(f"  Val F1 Macro: {val_f1_macro:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch + 1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Track best model
        if val_f1_micro > best_val_f1:
            best_val_f1 = val_f1_micro
            best_path = os.path.join(MODEL_SAVE_DIR, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            run.summary["best_val_f1_micro"] = best_val_f1
            run.summary["best_epoch"] = epoch + 1
            print(f"  -> New best model! (F1 Micro: {best_val_f1:.4f})")

    print(f"\nTraining complete. Best Val F1 Micro: {best_val_f1:.4f}")

    # Threshold optimization on validation set
    print("\n" + "=" * 70)
    print("Optimizing thresholds...")
    print("=" * 70)

    # Load best model for threshold optimization
    model = LongformerForSequenceClassification.from_pretrained(os.path.join(MODEL_SAVE_DIR, "best_model"))
    model.to(device)
    model.eval()

    best_thresholds, val_probs, val_labels = optimize_thresholds(model, val_loader, device, OFFICIAL_LABELS)

    # Save thresholds
    threshold_path = os.path.join(MODEL_SAVE_DIR, "class_thresholds_optimized.json")
    with open(threshold_path, 'w') as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"\nThresholds saved to {threshold_path}")

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final evaluation on test set...")
    print("=" * 70)

    test_probs = []
    test_labels_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            probs = torch.sigmoid(outputs.logits)

            test_probs.append(probs.cpu().numpy())
            test_labels_list.append(labels.numpy())

    test_probs = np.vstack(test_probs)
    test_labels_arr = np.vstack(test_labels_list)

    # Apply optimized thresholds
    test_preds = np.zeros_like(test_probs)
    for i, label_name in enumerate(OFFICIAL_LABELS):
        thresh = best_thresholds[label_name]
        test_preds[:, i] = (test_probs[:, i] > thresh).astype(int)

    # Final metrics
    test_f1_micro = f1_score(test_labels_arr, test_preds, average='micro')
    test_f1_macro = f1_score(test_labels_arr, test_preds, average='macro')

    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS (Optimized Thresholds)")
    print(f"{'='*70}")
    print(f"Micro F1: {test_f1_micro:.4f}")
    print(f"Macro F1: {test_f1_macro:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(test_labels_arr, test_preds, target_names=OFFICIAL_LABELS))

    # Log final metrics to W&B
    run.log({
        "test_f1_micro_optimized": test_f1_micro,
        "test_f1_macro_optimized": test_f1_macro,
    })
    run.summary["test_f1_micro_optimized"] = test_f1_micro
    run.summary["test_f1_macro_optimized"] = test_f1_macro

    # Save results summary
    results_summary = {
        "timestamp": TIMESTAMP,
        "config": CONFIG,
        "best_val_f1_micro": float(best_val_f1),
        "test_f1_micro_optimized": float(test_f1_micro),
        "test_f1_macro_optimized": float(test_f1_macro),
        "thresholds": best_thresholds
    }

    with open(os.path.join(MODEL_SAVE_DIR, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)

    print(f"\nResults saved to {MODEL_SAVE_DIR}/results_summary.json")

    # Finish W&B run
    run.finish()
    print("\nW&B run finished. View results at wandb.ai")


if __name__ == "__main__":
    main()
