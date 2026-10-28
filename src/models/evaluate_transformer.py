"""
Step 5b — Evaluate the fine-tuned transformer and compare with the Step 4 baseline.

Evaluates on:
  - validation.csv          (in-distribution, used for epoch selection)
  - test.csv                (in-distribution, final)
  - event_holdout_test.csv  (disasters never seen in training: srilanka_floods,
                             iraq_iran_earthquake)

The baseline is re-scored on the same files using the models saved in Step 4,
so every number in the comparison is computed, none are copied by hand.

Run (after train_transformer.py):
    python3 src/models/evaluate_transformer.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_transformer import (  # noqa: E402
    BATCH_SIZE, BEST_DIR, DATA_DIR, OUT_DIR, ROOT, TARGET_COL, TEXT_COL,
    TextDataset, compute_metrics, get_device, predict,
)

BASELINE_DIR = ROOT / "models" / "baseline"
SPLITS = {
    "validation": "validation.csv",
    "test": "test.csv",
    "event_holdout_test": "event_holdout_test.csv",
}


def load_baseline():
    step4 = json.load(open(BASELINE_DIR / "step4_baseline_report.json"))
    name = step4["chosen_model"]
    vec = joblib.load(BASELINE_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(BASELINE_DIR / f"baseline_{name}.joblib")
    return name, vec, clf


def main():
    if not BEST_DIR.exists():
        raise SystemExit("No trained model found. Run train_transformer.py first.")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(device)
    label2id = model.config.label2id
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    class_names = [id2label[i] for i in range(len(id2label))]
    collator = DataCollatorWithPadding(tokenizer)
    base_name, vec, clf = load_baseline()
    print(f"Device: {device} | baseline model: {base_name}")

    results = {"task": TARGET_COL, "text_column": TEXT_COL, "baseline_model": base_name}
    for split, fname in SPLITS.items():
        df = pd.read_csv(DATA_DIR / fname)
        bad = ~df[TARGET_COL].isin(label2id)
        if bad.any():
            print(f"WARNING: dropping {bad.sum()} rows in {split} with labels not in training set")
            df = df[~bad].reset_index(drop=True)

        ds = TextDataset(df[TEXT_COL].fillna(""), df[TARGET_COL].map(label2id), tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
        t_pred, t_gold = predict(model, loader, device)

        y_true = [id2label[i] for i in t_gold]
        t_labels = [id2label[i] for i in t_pred]
        b_labels = clf.predict(vec.transform(df[TEXT_COL].fillna("")))

        entry = {
            "n": len(df),
            "transformer": compute_metrics(y_true, t_labels),
            "baseline": compute_metrics(y_true, b_labels),
            "transformer_per_class": classification_report(
                y_true, t_labels, labels=class_names, output_dict=True, zero_division=0),
            "transformer_confusion_matrix": {
                "labels": class_names,
                "matrix": confusion_matrix(y_true, t_labels, labels=class_names).tolist(),
            },
        }
        if split == "event_holdout_test":
            per_event = {}
            for ev, g in df.assign(t=t_labels, b=b_labels).groupby("disaster_event"):
                per_event[ev] = {
                    "n": len(g),
                    "transformer": compute_metrics(g[TARGET_COL], g["t"]),
                    "baseline": compute_metrics(g[TARGET_COL], g["b"]),
                }
            entry["per_event"] = per_event
        results[split] = entry

        print(f"\n=== {split} (n={len(df)}) ===")
        print(f"{'':14}{'acc':>8}{'macroF1':>10}")
        for who in ("baseline", "transformer"):
            m = entry[who]
            print(f"{who:14}{m['accuracy']:8.4f}{m['macro_f1']:10.4f}")
        print("\nTransformer per-class:")
        print(classification_report(y_true, t_labels, labels=class_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):", class_names)
        print(confusion_matrix(y_true, t_labels, labels=class_names))

    with open(OUT_DIR / "step5_comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'step5_comparison_report.json'}")


if __name__ == "__main__":
    main()