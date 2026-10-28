"""
Step 7a (evaluation) — humanitarian_category transformer vs a TF-IDF baseline.

Step 4 only built a baseline for informative_label, so for a fair comparison
this script fits a TF-IDF + Logistic Regression baseline on humanitarian_category
RIGHT HERE, on train.csv only, with the same vectorizer settings as Step 4.
Both models see the same text column and the same merged labels.

Evaluated on validation, test, and event_holdout_test (unseen disasters).

Run (after train_humanitarian.py):
    python src/models/evaluate_humanitarian.py
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanitarian_labels import merge_label  # noqa: E402
from train_transformer import (  # noqa: E402
    BATCH_SIZE, DATA_DIR, SEED, TEXT_COL, TextDataset, compute_metrics, get_device, predict,
)
from train_humanitarian import BEST_DIR, OUT_DIR, TARGET_COL  # noqa: E402

SPLITS = {"validation": "validation.csv", "test": "test.csv",
          "event_holdout_test": "event_holdout_test.csv"}


def main():
    if not BEST_DIR.exists():
        raise SystemExit("No trained model found. Run train_humanitarian.py first.")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(device)
    label2id = model.config.label2id
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    class_names = [id2label[i] for i in range(len(id2label))]
    collator = DataCollatorWithPadding(tokenizer)

    train = pd.read_csv(DATA_DIR / "train.csv")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(train[TEXT_COL].fillna(""))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X_train, train[TARGET_COL].map(merge_label))
    print(f"Device: {device} | baseline fit on train only ({len(train)} rows)")

    results = {"task": TARGET_COL, "classes": class_names}
    for split, fname in SPLITS.items():
        df = pd.read_csv(DATA_DIR / fname)
        df["target"] = df[TARGET_COL].map(merge_label)
        bad = ~df["target"].isin(label2id)
        if bad.any():
            print(f"WARNING: dropping {bad.sum()} rows in {split} with labels not in training set")
            df = df[~bad].reset_index(drop=True)

        ds = TextDataset(df[TEXT_COL].fillna(""), df["target"].map(label2id), tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
        t_pred, t_gold = predict(model, loader, device)
        y_true = [id2label[i] for i in t_gold]
        t_labels = [id2label[i] for i in t_pred]
        b_labels = clf.predict(vec.transform(df[TEXT_COL].fillna("")))

        results[split] = {
            "n": len(df),
            "transformer": compute_metrics(y_true, t_labels),
            "baseline": compute_metrics(y_true, b_labels),
            "transformer_per_class": classification_report(
                y_true, t_labels, labels=class_names, output_dict=True, zero_division=0),
            "transformer_confusion_matrix": {
                "labels": class_names,
                "matrix": confusion_matrix(y_true, t_labels, labels=class_names).tolist()},
        }
        print(f"\n=== {split} (n={len(df)}) ===")
        print(f"{'':14}{'acc':>8}{'macroF1':>10}")
        for who in ("baseline", "transformer"):
            m = results[split][who]
            print(f"{who:14}{m['accuracy']:8.4f}{m['macro_f1']:10.4f}")
        print("\nTransformer per-class:")
        print(classification_report(y_true, t_labels, labels=class_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):", class_names)
        print(confusion_matrix(y_true, t_labels, labels=class_names))

    with open(OUT_DIR / "step7a_comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'step7a_comparison_report.json'}")


if __name__ == "__main__":
    main()