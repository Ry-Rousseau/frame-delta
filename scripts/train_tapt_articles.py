"""Task-Adaptive Pre-Training (TAPT) for Longformer on news articles."""

import os
import json
import csv
from datetime import datetime

import torch
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    LongformerTokenizer,
    LongformerForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

load_dotenv()

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # effective batch = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 2048
WARMUP_RATIO = 0.1
VAL_SPLIT = 0.05
SEED = 42

# Paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_SAVE_DIR = f"trained_models/mlm-longformer/longformer-news-base/{TIMESTAMP}"
LOG_DIR = "training_logs/mlm-longformer/logs"


class ArticleMLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def load_articles():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT b.maintext
        FROM mm_framing_full a
        JOIN newsarticles b ON a.url = b.url
        WHERE b.maintext IS NOT NULL
        AND LENGTH(b.maintext) > 100
    """)
    articles = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return articles


def save_config(path):
    config = {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "warmup_ratio": WARMUP_RATIO,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "base_model": "allenai/longformer-base-4096",
        "timestamp": TIMESTAMP
    }
    with open(os.path.join(path, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load data
    print("Loading articles...")
    articles = load_articles()
    print(f"Loaded {len(articles):,} articles")

    train_texts, val_texts = train_test_split(
        articles, test_size=VAL_SPLIT, random_state=SEED
    )
    print(f"Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    # Setup
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    train_dataset = ArticleMLMDataset(train_texts, tokenizer, MAX_LENGTH)
    val_dataset = ArticleMLMDataset(val_texts, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=data_collator, num_workers=0, pin_memory=True
    )

    model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
    model.to(device)
    model.gradient_checkpointing_enable()

    num_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    save_config(MODEL_SAVE_DIR)

    # CSV logging
    log_file = os.path.join(LOG_DIR, f"training_log_{TIMESTAMP}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    print(f"Training steps: {num_training_steps:,} | Warmup: {num_warmup_steps:,}")
    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                avg_loss = total_loss / (step + 1)
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])

        # Checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = os.path.join(MODEL_SAVE_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
