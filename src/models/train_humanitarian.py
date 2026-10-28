"""
Step 7a — Fine-tune DistilBERT to predict humanitarian_category.

Reuses the Step 5 helpers (dataset class, metrics, seed, device) so the two
transformers are trained the same way. Differences from Step 5:
  - Target is humanitarian_category, with the two tiny classes merged
    (see humanitarian_labels.py).
  - Class weights are the square root of the "balanced" weights. Fully balanced
    weights would give the rarest class ~24x weight and destabilise training.
  - 4 epochs (7-way task is harder). Best epoch chosen by VALIDATION macro-F1.

Run:
    python src/models/train_humanitarian.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanitarian_labels import MERGED_NAME, merge_label  # noqa: E402
from train_transformer import (  # noqa: E402
    BATCH_SIZE, DATA_DIR, LR, MAX_LEN, MODEL_NAME, ROOT, SEED, TEXT_COL,
    TextDataset, build_label_maps, compute_metrics, get_device, predict, set_seed,
)

TARGET_COL = "humanitarian_category"
EPOCHS = 4
OUT_DIR = ROOT / "models" / "transformer" / "humanitarian"
BEST_DIR = OUT_DIR / "best"


def load_split(name):
    df = pd.read_csv(DATA_DIR / name)
    df["target"] = df[TARGET_COL].map(merge_label)
    return df


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    train, val = load_split("train.csv"), load_split("validation.csv")
    label2id, id2label = build_label_maps(train["target"])
    print(f"Labels (from data, after merging rare classes into '{MERGED_NAME}'):")
    print(train["target"].value_counts().to_string(), "\n")
    print(f"label2id: {label2id}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)

    def make_loader(df, shuffle):
        df = df[df["target"].isin(label2id)]
        ds = TextDataset(df[TEXT_COL].fillna(""), df["target"].map(label2id), tokenizer)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collator)

    train_loader, val_loader = make_loader(train, True), make_loader(val, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)

    counts = train["target"].map(label2id).value_counts().sort_index().values
    balanced = len(train) / (len(counts) * counts)
    weights = np.sqrt(balanced)
    print("Class weights (sqrt-balanced):",
          {id2label[i]: round(float(w), 2) for i, w in enumerate(weights)})
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))

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
        "merged_classes": {MERGED_NAME: ["vehicle_damage", "missing_or_found_people"]},
        "text_column": TEXT_COL,
        "model_name": MODEL_NAME,
        "config": {"max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                   "lr": LR, "seed": SEED, "device": str(device),
                   "class_weights": "sqrt of balanced"},
        "label2id": label2id,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "step7a_train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBest epoch {best_epoch}, val macro-F1 {best_f1:.4f}")
    print(f"Saved model to {BEST_DIR} and report to {OUT_DIR / 'step7a_train_report.json'}")


if __name__ == "__main__":
    main()