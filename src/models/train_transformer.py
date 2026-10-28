"""
Step 5a — Fine-tune DistilBERT on the SAME task as the Step 4 baseline.

Task        : informative_label (informative vs not_informative)
Text column : text_clean  (same input as the baseline, for a fair comparison)
Splits      : data/processed/train.csv and validation.csv (unchanged from Step 2)

Design choices (all explainable in a viva):
  - distilbert-base-uncased: small, free, runs on a MacBook (Apple 'mps' GPU).
  - Plain PyTorch training loop (no Trainer) so every step is visible.
  - Class-weighted cross-entropy, mirroring class_weight="balanced" in the baseline.
  - The best epoch is chosen by VALIDATION macro-F1. Test sets are not touched here.

Run:
    python3 src/models/train_transformer.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "models" / "transformer"
BEST_DIR = OUT_DIR / "best"

TARGET_COL = "informative_label"
TEXT_COL = "text_clean"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64      # text_clean max is 48 words, mean 12 (Step 5 inspection)
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_label_maps(series: pd.Series):
    """Label ids come from the data itself (sorted), never hardcoded."""
    labels = sorted(series.dropna().unique())
    if "Unknown" in labels:
        raise ValueError("Unknown labels present in training data; fix Step 2 first.")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }


class TextDataset(Dataset):
    def __init__(self, texts, label_ids, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        golds.extend(batch["labels"].cpu().tolist())
    return np.array(preds), np.array(golds)


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    label2id, id2label = build_label_maps(train[TARGET_COL])
    print(f"Labels (from data): {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)

    def make_loader(df, shuffle):
        ds = TextDataset(df[TEXT_COL].fillna(""), df[TARGET_COL].map(label2id), tokenizer)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collator)

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)

    # "balanced" weights = n_samples / (n_classes * count), same idea as the baseline.
    counts = train[TARGET_COL].map(label2id).value_counts().sort_index().values
    weights = len(train) / (len(counts) * counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
    print(f"Class weights: {dict(zip(label2id, weights.round(3)))}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    history, best_f1, best_epoch = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        preds, golds = predict(model, val_loader, device)
        m = compute_metrics(golds, preds)
        m.update({"epoch": epoch, "train_loss": running / len(train_loader)})
        history.append(m)
        print(f"Epoch {epoch}: val acc {m['accuracy']:.4f}  val macro-F1 {m['macro_f1']:.4f}")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            BEST_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_DIR)
            tokenizer.save_pretrained(BEST_DIR)
            print(f"  -> saved best model (epoch {epoch})")

    report = {
        "task": TARGET_COL,
        "text_column": TEXT_COL,
        "model_name": MODEL_NAME,
        "config": {"max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                   "lr": LR, "seed": SEED, "device": str(device)},
        "label2id": label2id,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "step5_train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBest epoch {best_epoch}, val macro-F1 {best_f1:.4f}")
    print(f"Saved model to {BEST_DIR} and report to {OUT_DIR / 'step5_train_report.json'}")


if __name__ == "__main__":
    main()