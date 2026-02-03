# Phase 1: Silver Fine-Tuning Implementation Plan

**Goal:** Train Longformer on ~380k machine-labeled articles to learn general frame representations using Weighted BCE loss.

**Context:** TAPT was skipped due to prohibitive training time. This phase uses the base Longformer directly. Focal Loss is reserved for Phase 2 (Gold Fine-Tuning) where clean human labels make hard-example focus appropriate.

---

## Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Base model | `allenai/longformer-base-4096` | Proven 0.755 Micro F1 in v2 |
| Loss function | Weighted BCE | Mistral labels have 0.50 F1 vs gold; Focal Loss would amplify noise |
| Class weighting | Normalized inverse frequency | `alpha_i = (1/N_i) / sum(1/N_j)` |
| Global attention | Empirical test: [CLS] only vs [CLS] + first topic token | Avoid assuming benefit without evidence |
| Topic injection | `TOPIC: {topic}\n{title}\n{text}` | Proven approach from v2 |

---

## Part 1: Local Smoke Test (Notebook)

**Purpose:** Validate code, test global attention variants, catch issues before RunPod spend.

**Dataset:** ~10k stratified sample, 1-2 epochs

**Output:** `notebooks/silver_finetuning_smoke_test.ipynb`

### Task 1: Environment Setup & W&B Configuration

```python
# Imports
import os
import torch
import psycopg2
import pandas as pd
import numpy as np
import wandb
from dotenv import load_dotenv
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
)
from torch.utils.data import Dataset, DataLoader, Subset

load_dotenv()

# Verify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

#### W&B Setup (First Time)

```python
# One-time setup - run in terminal first:
# pip install wandb
# wandb login  (paste API key from wandb.ai/authorize)

# Initialize W&B for this run
wandb.init(
    project="frame-delta-silver",
    name="smoke-test-cls-only",  # Change for variant B: "smoke-test-cls-topic"
    config={
        "model": "allenai/longformer-base-4096",
        "dataset_size": 10000,
        "max_length": 2048,
        "batch_size": 2,
        "grad_accum_steps": 8,
        "learning_rate": 2e-5,
        "epochs": 2,
        "loss": "weighted_bce",
        "global_attention": "cls_only",  # or "cls_plus_topic"
    }
)
config = wandb.config
```

### Task 2: Load Data from Database

```python
# Connect and load stratified sample
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# Set seed for reproducibility
cur.execute("SELECT setseed(0.42)")

# Load sample with topic for injection
cur.execute("""
    SELECT a.text_generic_frame, a.gpt_topic, a.title, b.maintext
    FROM mm_framing_full a
    JOIN newsarticles b ON a.url = b.url
    WHERE b.maintext IS NOT NULL
    AND LENGTH(b.maintext) > 100
    ORDER BY RANDOM()
    LIMIT 10000
""")

result = cur.fetchall()
cur.close()
conn.close()

df = pd.DataFrame(result, columns=["text_generic_frame", "gpt_topic", "title", "article_text"])
print(f"Loaded {len(df):,} articles")
```

### Task 3: Preprocess Text with Topic Injection

```python
# Apply topic injection (proven approach from v2)
df['full_text'] = "TOPIC: " + df['gpt_topic'] + "\n" + df['title'] + "\n" + df['article_text']

# Preview
print(df.iloc[0]['full_text'][:500])
```

### Task 4: Encode Labels

```python
official_labels = [
    "Economic", "Capacity and resources", "Morality", "Fairness and equality",
    "Legality, constitutionality and jurisprudence", "Policy prescription and evaluation",
    "Crime and punishment", "Security and defense", "Health and safety",
    "Quality of life", "Cultural identity", "Public opinion", "Political",
    "External regulation and reputation", "Other"
]

mlb = MultiLabelBinarizer(classes=official_labels)
labels_matrix = mlb.fit_transform(df['text_generic_frame'])
print(f"Labels shape: {labels_matrix.shape}")

# Calculate normalized inverse frequency for pos_weight
num_positives = labels_matrix.sum(axis=0)
inv_freq = 1.0 / (num_positives + 1e-5)
alpha = inv_freq / inv_freq.sum()  # Normalized inverse frequency
pos_weight = torch.tensor(alpha * len(official_labels), dtype=torch.float).to(device)

print("Class weights (pos_weight):")
for name, weight in zip(official_labels, pos_weight.cpu().numpy()):
    print(f"  {name}: {weight:.3f}")
```

### Task 5: Tokenization with Longformer

```python
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Tokenize all texts
encodings = tokenizer(
    df['full_text'].tolist(),
    truncation=True,
    padding="max_length",
    max_length=config.max_length,
    return_tensors="pt"
)

print(f"Input IDs shape: {encodings['input_ids'].shape}")
```

### Task 6: Dataset Class with Global Attention Mask

```python
class FramingDataset(Dataset):
    def __init__(self, encodings, labels_matrix, tokenizer, global_attention_mode="cls_only"):
        """
        global_attention_mode: "cls_only" or "cls_plus_topic"
        """
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = labels_matrix
        self.tokenizer = tokenizer
        self.global_attention_mode = global_attention_mode

        # Token ID for "TOPIC" - we'll use this for cls_plus_topic mode
        # "TOPIC" tokenizes to specific IDs we can identify
        self.topic_token_ids = tokenizer.encode("TOPIC", add_special_tokens=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        # Build global attention mask
        # 0 = local attention, 1 = global attention
        global_attention_mask = torch.zeros_like(attention_mask)

        # Always apply global attention to [CLS] (position 0)
        global_attention_mask[0] = 1

        if self.global_attention_mode == "cls_plus_topic":
            # Find first token of the topic word (after "TOPIC: ")
            # "TOPIC:" is typically tokens 0-2, topic word starts around position 3-4
            # Apply global attention to position 3 (first token after "TOPIC: ")
            # This is a simplified heuristic - the exact position may vary slightly
            if len(input_ids) > 3:
                global_attention_mask[3] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
```

### Task 7: Train/Val/Test Split (Stratified)

```python
N = len(labels_matrix)
X_indices = np.zeros(N)

# Split 1: Train (80%) / Temp (20%)
msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, temp_idx = next(iter(msss1.split(X_indices, labels_matrix)))

# Split 2: Val (10%) / Test (10%)
temp_labels = labels_matrix[temp_idx]
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
relative_val_idx, relative_test_idx = next(iter(msss2.split(np.zeros(len(temp_idx)), temp_labels)))

val_idx = temp_idx[relative_val_idx]
test_idx = temp_idx[relative_test_idx]

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
```

### Task 8: Create DataLoaders

```python
def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'global_attention_mask': torch.stack([x['global_attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

# Create dataset with chosen global attention mode
# CHANGE THIS FOR VARIANT B: global_attention_mode="cls_plus_topic"
full_dataset = FramingDataset(
    encodings, labels_matrix, tokenizer,
    global_attention_mode=config.global_attention
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
```

### Task 9: Model Setup

```python
model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    num_labels=len(official_labels),
    problem_type="multi_label_classification"
)
model.to(device)

# Enable gradient checkpointing for VRAM efficiency
model.gradient_checkpointing_enable()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

# Loss function with class weights
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Task 10: Training Loop

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
best_val_f1 = 0.0
save_dir = "saved_models/silver_smoke_test"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(config.epochs):
    # ==================== TRAINING ====================
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for step, batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            loss = criterion(outputs.logits, labels)
            loss = loss / config.grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()
        train_loss += loss.item() * config.grad_accum_steps

        # Gradient accumulation step
        if (step + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': f'{loss.item() * config.grad_accum_steps:.4f}'})

    avg_train_loss = train_loss / len(train_loader)

    # ==================== VALIDATION ====================
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
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
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs.logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_val_loss = val_loss / len(val_loader)
    val_f1_micro = f1_score(all_labels, all_preds, average='micro')
    val_f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_f1_micro": val_f1_micro,
        "val_f1_macro": val_f1_macro,
    })

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"  Val F1 Micro: {val_f1_micro:.4f} | Val F1 Macro: {val_f1_macro:.4f}")

    # Checkpoint on best
    if val_f1_micro > best_val_f1:
        best_val_f1 = val_f1_micro
        model.save_pretrained(f"{save_dir}/best_model")
        tokenizer.save_pretrained(f"{save_dir}/best_model")
        print(f"  -> New best model saved! (F1: {best_val_f1:.4f})")

wandb.finish()
```

### Task 11: Post-Training Threshold Optimization

```python
# Load best model
model = LongformerForSequenceClassification.from_pretrained(f"{save_dir}/best_model")
model.to(device)
model.eval()

# Get raw probabilities on validation set
val_probs = []
val_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Getting probabilities"):
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

        val_probs.append(probs.cpu().numpy())
        val_labels.append(labels.numpy())

val_probs = np.vstack(val_probs)
val_labels = np.vstack(val_labels)

# Grid search for optimal thresholds per class
best_thresholds = {}
print("\nOptimizing thresholds per class...")

for i, label_name in enumerate(official_labels):
    best_score = 0
    best_thresh = 0.5

    y_true = val_labels[:, i]
    y_score = val_probs[:, i]

    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_score > thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    best_thresholds[label_name] = float(best_thresh)
    print(f"  {label_name:<45} Threshold: {best_thresh:.2f} (F1: {best_score:.3f})")

# Save thresholds
import json
with open(f"{save_dir}/class_thresholds_optimized.json", 'w') as f:
    json.dump(best_thresholds, f, indent=4)
```

### Task 12: Final Evaluation with Optimized Thresholds

```python
# Evaluate on TEST set with optimized thresholds
test_probs = []
test_labels = []

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
        test_labels.append(labels.numpy())

test_probs = np.vstack(test_probs)
test_labels = np.vstack(test_labels)

# Apply optimized thresholds
test_preds = np.zeros_like(test_probs)
for i, label_name in enumerate(official_labels):
    thresh = best_thresholds[label_name]
    test_preds[:, i] = (test_probs[:, i] > thresh).astype(int)

# Final metrics
test_f1_micro = f1_score(test_labels, test_preds, average='micro')
test_f1_macro = f1_score(test_labels, test_preds, average='macro')

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS (Optimized Thresholds)")
print(f"{'='*60}")
print(f"Micro F1: {test_f1_micro:.4f}")
print(f"Macro F1: {test_f1_macro:.4f}")
print(f"\nPer-class report:")
print(classification_report(test_labels, test_preds, target_names=official_labels))

# Log final metrics to W&B (re-init for summary)
wandb.init(project="frame-delta-silver", name=f"smoke-test-final-{config.global_attention}", reinit=True)
wandb.log({
    "test_f1_micro_optimized": test_f1_micro,
    "test_f1_macro_optimized": test_f1_macro,
})
wandb.finish()
```

---

## Part 2: Comparison & Decision

After running both variants:

| Variant | Global Attention | Test F1 Micro | Test F1 Macro |
|---------|------------------|---------------|---------------|
| A | [CLS] only | ___ | ___ |
| B | [CLS] + first topic token | ___ | ___ |

**Decision criteria:**
- If B improves both metrics by >1%: Use B for full run
- If difference is <1%: Use A (simpler, less compute)
- If B is worse: Definitely use A

---

## Part 3: Full RunPod Run (Script)

**To be created after smoke test completes.** Key differences from local:

| Aspect | Local Smoke Test | Full RunPod Run |
|--------|------------------|-----------------|
| Dataset | 10k samples | ~380k samples |
| Epochs | 2 | 5 (with early stopping) |
| Batch size | 2 | 4-8 (more VRAM) |
| Grad accum | 8 | 4-8 (adjust for effective batch 32) |
| Checkpointing | Best only | Per-epoch + best |
| Format | Notebook | Script with argparse |

Script will be created at: `scripts/train_silver_runpod.py`

---

## Output Artifacts

**Smoke test:**
- `notebooks/silver_finetuning_smoke_test.ipynb`
- `saved_models/silver_smoke_test/best_model/`
- `saved_models/silver_smoke_test/class_thresholds_optimized.json`

**Full run (after smoke test):**
- `scripts/train_silver_runpod.py`
- `saved_models/longformer-news-silver/`
- `saved_models/longformer-news-silver/class_thresholds_optimized.json`

---

## W&B Quick Reference

```bash
# First-time setup
pip install wandb
wandb login  # Get API key from wandb.ai/authorize

# View runs
# Go to wandb.ai -> Your projects -> frame-delta-silver
```

W&B will automatically track:
- Loss curves (train/val)
- F1 metrics per epoch
- Hyperparameters
- System metrics (GPU utilization, memory)

Compare runs in the W&B dashboard to pick the winning global attention strategy.
