"""
Phase 2: Gold Fine-Tuning for Longformer - RunPod-style script.

Before running:
1. Install dependencies: pip install wandb iterstrat
2. Run: wandb login
3. Ensure parquet data exists:
   - data/gold_train_data.parquet
   (Generate with: python scripts/db_loading/export_gold_data.py)
"""

import os
import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    MultilabelStratifiedKFold,
)
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model
    "model_checkpoint": "trained_models/checkpoint_epoch_4/checkpoint_epoch_4/",
    "num_labels": 15,
    "global_attention": "cls_plus_topic",
    "topic_token_index": 4,  # aligns with Phase 1 topic injection

    # Training
    "max_length": 2048,
    "batch_size": 2,
    "grad_accum_steps": 8,  # effective batch size = 16
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "scheduler": "reduce_on_plateau",  # "reduce_on_plateau" or "linear_warmup"
    "lr_patience": 1,
    "lr_factor": 0.5,
    "min_lr": 1e-6,

    # Loss
    "focal_gamma_values": [1, 2, 3],
    "run_bce_baseline": True,
    "use_class_weights": True,

    # CV / Splits
    "n_folds": 5,
    "test_size": 0.10,
    "seed": 42,

    # DataLoader
    "num_workers": 4,
    "pin_memory": True,
}

# Paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
RUN_NAME = f"gold-phase2-{TIMESTAMP}"
DATA_FILE = "data/gold_train_data.parquet"
MODEL_SAVE_DIR = f"trained_models/gold-phase2/{TIMESTAMP}"
LOG_DIR = "training_logs/gold-phase2"
FOLDS_PATH = "data/gold_cv_folds.json"

# Labels
OFFICIAL_LABELS = [
    "Economic", "Capacity and resources", "Morality", "Fairness and equality",
    "Legality, constitutionality and jurisprudence", "Policy prescription and evaluation",
    "Crime and punishment", "Security and defense", "Health and safety",
    "Quality of life", "Cultural identity", "Public opinion", "Political",
    "External regulation and reputation", "Other"
]


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_labels(value: Any) -> List[int]:
    """Parse labels from list/array/json string to list[int]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    return []


def load_gold_data(path: str) -> pd.DataFrame:
    """Load gold data from parquet and normalize columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_parquet(path)

    # Normalize column names
    if "formatted_text" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"formatted_text": "text"})

    if "labels_idx" not in df.columns and "labels_idx_json" in df.columns:
        df["labels_idx"] = df["labels_idx_json"].apply(parse_labels)
    else:
        df["labels_idx"] = df["labels_idx"].apply(parse_labels)

    # Basic cleaning
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Sanity check: TOPIC: prefix (no space)
    bad_prefix = df[~df["text"].astype(str).str.startswith("TOPIC:")]
    if len(bad_prefix) > 0:
        print(f"Warning: {len(bad_prefix)} rows do not start with 'TOPIC:'.")
        print("Example:")
        print(bad_prefix.iloc[0]["text"][:200])

    return df


def labels_to_matrix(labels_list: List[List[int]], num_labels: int) -> np.ndarray:
    matrix = np.zeros((len(labels_list), num_labels), dtype=np.float32)
    for i, indices in enumerate(labels_list):
        for idx in indices:
            if 0 <= idx < num_labels:
                matrix[i, idx] = 1.0
    return matrix


def compute_class_weights(labels_matrix: np.ndarray, num_labels: int) -> torch.Tensor:
    """Compute normalized inverse-frequency weights (same style as Phase 1)."""
    num_positives = labels_matrix.sum(axis=0)
    inv_freq = 1.0 / (num_positives + 1e-5)
    alpha = inv_freq / inv_freq.sum()
    pos_weight = torch.tensor(alpha * num_labels, dtype=torch.float)
    return pos_weight


# =============================================================================
# DATASET + COLLATE
# =============================================================================

class GoldArticleDataset(Dataset):
    """Gold dataset with on-the-fly tokenization and global attention masks."""

    def __init__(
        self,
        df: pd.DataFrame,
        labels_matrix: np.ndarray,
        tokenizer: LongformerTokenizerFast,
        max_len: int = 2048,
        global_attention_mode: str = "cls_plus_topic",
        topic_token_index: int = 4,
    ):
        self.df = df
        self.labels_matrix = labels_matrix
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.global_attention_mode = global_attention_mode
        self.topic_token_index = topic_token_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row["text"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1  # CLS token
        if self.global_attention_mode == "cls_plus_topic":
            if len(input_ids) > self.topic_token_index:
                global_attention_mask[self.topic_token_index] = 1

        labels = torch.tensor(self.labels_matrix[idx], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "labels": labels,
        }


def longformer_collate_fn(batch: List[Dict[str, Any]], tokenizer: LongformerTokenizerFast) -> Dict[str, torch.Tensor]:
    """Dynamic padding collator aligned to 512-token windows."""
    max_len = max(len(item["input_ids"]) for item in batch)
    window_size = 512
    padded_len = ((max_len + window_size - 1) // window_size) * window_size

    input_ids_batch = []
    attention_mask_batch = []
    global_attention_mask_batch = []
    labels_batch = []

    pad_token_id = tokenizer.pad_token_id

    for item in batch:
        curr_len = len(item["input_ids"])
        pad_len = padded_len - curr_len

        ids = item["input_ids"] + [pad_token_id] * pad_len
        mask = item["attention_mask"] + [0] * pad_len
        global_mask = item["global_attention_mask"] + [0] * pad_len

        input_ids_batch.append(ids)
        attention_mask_batch.append(mask)
        global_attention_mask_batch.append(global_mask)
        labels_batch.append(item["labels"])

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "global_attention_mask": torch.tensor(global_attention_mask_batch, dtype=torch.long),
        "labels": torch.stack(labels_batch),
    }


# =============================================================================
# LOSSES
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    scheduler=None,
    scheduler_type: str = "reduce_on_plateau",
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for step, batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        global_attention_mask = batch["global_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            loss = criterion(outputs.logits, labels)
            loss = loss / config["grad_accum_steps"]

        scaler.scale(loss).backward()
        total_loss += loss.item() * config["grad_accum_steps"]

        if (step + 1) % config["grad_accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and scheduler_type == "linear_warmup":
                scheduler.step()

        pbar.set_postfix({"loss": f"{loss.item() * config['grad_accum_steps']:.4f}"})

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            global_attention_mask = batch["global_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs.logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(outputs.logits.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)

    avg_loss = total_loss / len(data_loader)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "micro_f1": f1_micro,
        "macro_f1": f1_macro,
        "predictions": all_preds,
        "labels": all_labels,
        "logits": all_logits,
    }


# =============================================================================
# CV / TRAINING HELPERS
# =============================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    scheduler_type: str,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
):
    if scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=True,
        )
    if scheduler_type == "linear_warmup":
        warmup_steps = int(total_steps * warmup_ratio)
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return None


def train_fold(
    fold_idx: int,
    train_idx: List[int],
    val_idx: List[int],
    df: pd.DataFrame,
    labels_matrix: np.ndarray,
    tokenizer: LongformerTokenizerFast,
    config: Dict[str, Any],
    loss_type: str,
    gamma: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx} | Loss: {loss_type} | Gamma: {gamma if loss_type == 'focal' else 'N/A'}")
    print(f"{'='*60}")

    # Build datasets
    dataset = GoldArticleDataset(
        df,
        labels_matrix,
        tokenizer,
        max_len=config["max_length"],
        global_attention_mode=config["global_attention"],
        topic_token_index=config["topic_token_index"],
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    def collate_fn(batch):
        return longformer_collate_fn(batch, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Initialize model
    gc.collect()
    torch.cuda.empty_cache()

    model = LongformerForSequenceClassification.from_pretrained(
        config["model_checkpoint"],
        num_labels=config["num_labels"],
        problem_type="multi_label_classification",
        use_safetensors=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()

    # Class weights from TRAIN split only
    pos_weight = None
    alpha = None
    if config["use_class_weights"]:
        train_labels = labels_matrix[train_idx]
        pos_weight = compute_class_weights(train_labels, config["num_labels"]).to(device)
        alpha = pos_weight

    # Loss function
    if loss_type == "focal":
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = (len(train_loader) * config["epochs"]) // config["grad_accum_steps"]
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        config["warmup_ratio"],
        config["scheduler"],
        config["lr_patience"],
        config["lr_factor"],
        config["min_lr"],
    )

    scaler = GradScaler("cuda")

    # Training loop
    best_micro_f1 = 0.0
    best_metrics = None

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, config,
            scheduler=scheduler, scheduler_type=config["scheduler"]
        )

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Step ReduceLROnPlateau after validation
        if scheduler is not None and config["scheduler"] == "reduce_on_plateau":
            scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"LR: {current_lr:.2e}")

        # Log to W&B
        log_prefix = f"{loss_type}_gamma{gamma}" if loss_type == "focal" else "bce"
        wandb.log({
            f"{log_prefix}/fold{fold_idx}/epoch": epoch + 1,
            f"{log_prefix}/fold{fold_idx}/train_loss": train_loss,
            f"{log_prefix}/fold{fold_idx}/val_loss": val_metrics["loss"],
            f"{log_prefix}/fold{fold_idx}/val_micro_f1": val_metrics["micro_f1"],
            f"{log_prefix}/fold{fold_idx}/val_macro_f1": val_metrics["macro_f1"],
            f"{log_prefix}/fold{fold_idx}/lr": current_lr,
        })

        if val_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = val_metrics["micro_f1"]
            best_metrics = val_metrics.copy()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return best_metrics


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds_to_try=np.arange(0.1, 0.9, 0.05)
) -> List[Dict[str, Any]]:
    """Find optimal threshold for each class."""
    num_classes = probs.shape[1]
    optimal_thresholds = []

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = labels[:, class_idx]

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds_to_try:
            preds = (class_probs > threshold).astype(int)
            f1 = f1_score(class_labels, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds.append({
            "class_idx": class_idx,
            "label": OFFICIAL_LABELS[class_idx],
            "threshold": float(best_threshold),
            "f1": float(best_f1),
        })

    return optimal_thresholds


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\nRun name: {RUN_NAME}")
    print(f"Save directory: {MODEL_SAVE_DIR}")
    print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")

    # W&B init
    run = wandb.init(
        entity="ryrousseau-london-school-of-economics-and-political-science",
        project="frame-delta-gold",
        name=RUN_NAME,
        config=CONFIG
    )
    print(f"W&B run initialized: {run.name}")
    print(f"View at: {run.url}")

    # Load data
    df = load_gold_data(DATA_FILE)
    print(f"\nLoaded {len(df):,} gold rows")

    labels_list = df["labels_idx"].tolist()
    labels_matrix = labels_to_matrix(labels_list, CONFIG["num_labels"])

    # Label distribution
    print("\nClass distribution:")
    for i, label in enumerate(OFFICIAL_LABELS):
        count = labels_matrix[:, i].sum()
        print(f"  {label:<45} {int(count):>6} ({100*count/len(labels_matrix):.1f}%)")

    # Train/Test split (held-out test)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["seed"]
    )
    train_idx, test_idx = next(iter(msss.split(np.zeros(len(df)), labels_matrix)))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_labels_matrix = labels_matrix[train_idx]
    test_labels_matrix = labels_matrix[test_idx]

    print(f"\nTrain/Test split:")
    print(f"  Train: {len(train_df):,} ({100*len(train_df)/len(df):.0f}%)")
    print(f"  Test:  {len(test_df):,} ({100*len(test_df)/len(df):.0f}%) - held out")

    # CV folds on train set
    cv = MultilabelStratifiedKFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"]
    )
    folds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(train_df)), train_labels_matrix)):
        folds.append({
            "fold": fold_idx,
            "train_idx": tr_idx.tolist(),
            "val_idx": val_idx.tolist(),
        })
        print(f"Fold {fold_idx}: train={len(tr_idx)}, val={len(val_idx)}")

    # Save fold indices
    Path(FOLDS_PATH).parent.mkdir(exist_ok=True)
    with open(FOLDS_PATH, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Fold indices saved to {FOLDS_PATH}")

    # Tokenizer (once)
    tokenizer = LongformerTokenizerFast.from_pretrained(CONFIG["model_checkpoint"])

    # Experiments
    all_experiment_results = {}

    for gamma in CONFIG["focal_gamma_values"]:
        print("\n" + "#" * 70)
        print(f"# EXPERIMENT: Focal Loss (gamma={gamma})")
        print("#" * 70)

        fold_results = []
        oof_logits_list = []
        oof_labels_list = []

        for fold in folds:
            metrics = train_fold(
                fold["fold"],
                fold["train_idx"],
                fold["val_idx"],
                train_df,
                train_labels_matrix,
                tokenizer,
                CONFIG,
                loss_type="focal",
                gamma=gamma,
                device=device,
            )

            fold_results.append({
                "fold": fold["fold"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "loss": metrics["loss"],
            })

            oof_logits_list.append(metrics["logits"])
            oof_labels_list.append(metrics["labels"])

        results_df = pd.DataFrame(fold_results)
        mean_micro = results_df["micro_f1"].mean()
        std_micro = results_df["micro_f1"].std()
        mean_macro = results_df["macro_f1"].mean()
        std_macro = results_df["macro_f1"].std()

        print(f"\n{'='*60}")
        print(f"FOCAL LOSS (gamma={gamma}) RESULTS")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        print(f"\nMean Micro F1: {mean_micro:.4f} (+/- {std_micro:.4f})")
        print(f"Mean Macro F1: {mean_macro:.4f} (+/- {std_macro:.4f})")

        wandb.log({
            f"focal_gamma{gamma}/mean_micro_f1": mean_micro,
            f"focal_gamma{gamma}/std_micro_f1": std_micro,
            f"focal_gamma{gamma}/mean_macro_f1": mean_macro,
            f"focal_gamma{gamma}/std_macro_f1": std_macro,
        })

        all_experiment_results[f"focal_gamma{gamma}"] = {
            "fold_results": fold_results,
            "mean_micro_f1": mean_micro,
            "mean_macro_f1": mean_macro,
            "oof_logits": np.vstack(oof_logits_list),
            "oof_labels": np.vstack(oof_labels_list),
        }

    if CONFIG["run_bce_baseline"]:
        print("\n" + "#" * 70)
        print("# EXPERIMENT: Standard BCE Loss (Baseline)")
        print("#" * 70)

        fold_results = []
        oof_logits_list = []
        oof_labels_list = []

        for fold in folds:
            metrics = train_fold(
                fold["fold"],
                fold["train_idx"],
                fold["val_idx"],
                train_df,
                train_labels_matrix,
                tokenizer,
                CONFIG,
                loss_type="bce",
                gamma=0.0,
                device=device,
            )

            fold_results.append({
                "fold": fold["fold"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "loss": metrics["loss"],
            })

            oof_logits_list.append(metrics["logits"])
            oof_labels_list.append(metrics["labels"])

        results_df = pd.DataFrame(fold_results)
        mean_micro = results_df["micro_f1"].mean()
        mean_macro = results_df["macro_f1"].mean()

        print(f"\n{'='*60}")
        print("BCE LOSS RESULTS")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        print(f"\nMean Micro F1: {mean_micro:.4f} (+/- {results_df['micro_f1'].std():.4f})")
        print(f"Mean Macro F1: {mean_macro:.4f} (+/- {results_df['macro_f1'].std():.4f})")

        wandb.log({
            "bce/mean_micro_f1": mean_micro,
            "bce/mean_macro_f1": mean_macro,
        })

        all_experiment_results["bce"] = {
            "fold_results": fold_results,
            "mean_micro_f1": mean_micro,
            "mean_macro_f1": mean_macro,
            "oof_logits": np.vstack(oof_logits_list),
            "oof_labels": np.vstack(oof_labels_list),
        }

    # Compare experiments
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    comparison_data = []
    for exp_name, exp_data in all_experiment_results.items():
        comparison_data.append({
            "Experiment": exp_name,
            "Micro F1": f"{exp_data['mean_micro_f1']:.4f}",
            "Macro F1": f"{exp_data['mean_macro_f1']:.4f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    best_exp = max(all_experiment_results.items(), key=lambda x: x[1]["mean_micro_f1"])
    print(f"\nBest experiment: {best_exp[0]}")
    print(f"  Micro F1: {best_exp[1]['mean_micro_f1']:.4f}")
    print(f"  Macro F1: {best_exp[1]['mean_macro_f1']:.4f}")

    wandb.log({
        "best_experiment": best_exp[0],
        "best_micro_f1": best_exp[1]["mean_micro_f1"],
        "best_macro_f1": best_exp[1]["mean_macro_f1"],
    })

    # Threshold optimization (OOF)
    best_oof_logits = best_exp[1]["oof_logits"]
    best_oof_labels = best_exp[1]["oof_labels"]
    best_oof_probs = 1 / (1 + np.exp(-best_oof_logits))

    optimal_thresholds = optimize_thresholds(best_oof_probs, best_oof_labels)
    thresholds_dict = {t["label"]: t["threshold"] for t in optimal_thresholds}

    thresholds_path = os.path.join(MODEL_SAVE_DIR, "class_thresholds_optimized.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"\nThresholds saved to {thresholds_path}")

    # Final model training (100% train set)
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING (100% Train Set)")
    print("=" * 70)

    best_exp_name = best_exp[0]
    if "gamma" in best_exp_name:
        best_gamma = int(best_exp_name.split("gamma")[1])
        best_loss_type = "focal"
    else:
        best_gamma = 2
        best_loss_type = "bce"

    final_model = LongformerForSequenceClassification.from_pretrained(
        CONFIG["model_checkpoint"],
        num_labels=CONFIG["num_labels"],
        problem_type="multi_label_classification",
        use_safetensors=True,
    )
    final_model.to(device)
    final_model.gradient_checkpointing_enable()

    full_dataset = GoldArticleDataset(
        train_df,
        train_labels_matrix,
        tokenizer,
        max_len=CONFIG["max_length"],
        global_attention_mode=CONFIG["global_attention"],
        topic_token_index=CONFIG["topic_token_index"],
    )

    def collate_fn(batch):
        return longformer_collate_fn(batch, tokenizer)

    full_loader = DataLoader(
        full_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    # Loss & optimizer
    pos_weight = None
    alpha = None
    if CONFIG["use_class_weights"]:
        pos_weight = compute_class_weights(train_labels_matrix, CONFIG["num_labels"]).to(device)
        alpha = pos_weight

    if best_loss_type == "focal":
        loss_fn = FocalLoss(alpha=alpha, gamma=best_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(
        final_model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    total_steps = (len(full_loader) * CONFIG["epochs"]) // CONFIG["grad_accum_steps"]
    final_scheduler = build_scheduler(
        optimizer,
        total_steps,
        CONFIG["warmup_ratio"],
        CONFIG["scheduler"],
        CONFIG["lr_patience"],
        CONFIG["lr_factor"],
        CONFIG["min_lr"],
    )

    scaler = GradScaler("cuda")

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        train_loss = train_epoch(
            final_model, full_loader, optimizer, scaler, loss_fn, device, CONFIG,
            scheduler=final_scheduler, scheduler_type=CONFIG["scheduler"]
        )

        if final_scheduler is not None and CONFIG["scheduler"] == "reduce_on_plateau":
            # No validation set here; step on train loss to keep LR adaptive.
            final_scheduler.step(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
        wandb.log({
            "final_model/epoch": epoch + 1,
            "final_model/train_loss": train_loss,
            "final_model/lr": current_lr,
        })

    # Save final model
    final_model.save_pretrained(MODEL_SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    # Evaluate on held-out test set using optimized thresholds
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (Optimized Thresholds)")
    print("=" * 70)

    test_dataset = GoldArticleDataset(
        test_df,
        test_labels_matrix,
        tokenizer,
        max_len=CONFIG["max_length"],
        global_attention_mode=CONFIG["global_attention"],
        topic_token_index=CONFIG["topic_token_index"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    final_model.eval()
    test_probs = []
    test_labels_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            global_attention_mask = batch["global_attention_mask"].to(device)
            labels = batch["labels"]

            outputs = final_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            probs = torch.sigmoid(outputs.logits)
            test_probs.append(probs.cpu().numpy())
            test_labels_list.append(labels.numpy())

    test_probs = np.vstack(test_probs)
    test_labels_arr = np.vstack(test_labels_list)

    test_preds = np.zeros_like(test_probs)
    for i, label_name in enumerate(OFFICIAL_LABELS):
        thresh = thresholds_dict[label_name]
        test_preds[:, i] = (test_probs[:, i] > thresh).astype(int)

    test_f1_micro = f1_score(test_labels_arr, test_preds, average="micro")
    test_f1_macro = f1_score(test_labels_arr, test_preds, average="macro")

    print(f"Micro F1: {test_f1_micro:.4f}")
    print(f"Macro F1: {test_f1_macro:.4f}")
    print("\nPer-class report:")
    print(classification_report(test_labels_arr, test_preds, target_names=OFFICIAL_LABELS))

    wandb.log({
        "test_f1_micro_optimized": test_f1_micro,
        "test_f1_macro_optimized": test_f1_macro,
    })
    run.summary["test_f1_micro_optimized"] = test_f1_micro
    run.summary["test_f1_macro_optimized"] = test_f1_macro

    # Save results summary
    results_summary = {
        "timestamp": TIMESTAMP,
        "config": CONFIG,
        "best_experiment": best_exp[0],
        "cv_results": {k: {"mean_micro_f1": v["mean_micro_f1"], "mean_macro_f1": v["mean_macro_f1"]}
                       for k, v in all_experiment_results.items()},
        "optimized_thresholds": thresholds_dict,
        "test_f1_micro_optimized": float(test_f1_micro),
        "test_f1_macro_optimized": float(test_f1_macro),
    }

    with open(os.path.join(MODEL_SAVE_DIR, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nFinal model saved to {MODEL_SAVE_DIR}")
    print(f"Results saved to {MODEL_SAVE_DIR}/results_summary.json")

    run.finish()
    print("\nW&B run finished. View results at wandb.ai")


if __name__ == "__main__":
    main()
