"""
Step 4 — Baseline model: TF-IDF + Logistic Regression / Linear SVM.

Target task: informative_label (Informative vs Not informative).
This is the primary text classification task CrisisMMD supports directly
with a clean text-based ground truth (label_text, verified in Step 2 —
see data/processed/step2_build_report.json).

Why not the humanitarian/actionability task here too: "actionability" as a
label doesn't exist yet — it's explicitly defined from humanitarian
categories in Step 6, which hasn't been built. Running a baseline against
humanitarian_category directly is straightforward to add once Step 6's
mapping is settled, so it's deferred rather than guessed at now.

This script:
  1. Loads train/validation/test.csv (built in Step 2).
  2. Fits TF-IDF ONLY on the training set (no leakage from val/test into
     the vocabulary).
  3. Trains both Logistic Regression and Linear SVM, compares them on the
     validation set, and picks the better one.
  4. Reports final metrics (accuracy, precision, recall, F1, confusion
     matrix) on the held-out test set with the CHOSEN model only, and
     separately reports both models' validation numbers for comparison
     (required for the research paper's baseline-vs-transformer section).
  5. Saves the fitted vectorizer + chosen model under models/baseline/.

Run:
    python3 src/models/train_baseline.py
"""

import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "baseline"
TARGET_COL = "informative_label"
TEXT_COL = "text_clean"
SEED = 42


def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        n_unknown = (df[TARGET_COL] == "Unknown").sum()
        if n_unknown:
            print(f"WARNING: {n_unknown} rows in {name} have an Unknown "
                  f"{TARGET_COL} — check Step 2 label coverage before trusting these numbers.")
    return train, val, test


def evaluate(y_true, y_pred, label: str):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\n-- {label} --")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro P  : {precision:.4f}")
    print(f"Macro R  : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), classes:",
          sorted(set(y_true) | set(y_pred)))
    print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred))))
    return {"accuracy": acc, "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()

    print(f"Train: {len(train)}  Validation: {len(val)}  Test: {len(test)}")
    print(f"\nClass balance (train):")
    print(train[TARGET_COL].value_counts(normalize=True).round(3))

    X_train_text, y_train = train[TEXT_COL].fillna(""), train[TARGET_COL]
    X_val_text, y_val = val[TEXT_COL].fillna(""), val[TARGET_COL]
    X_test_text, y_test = test[TEXT_COL].fillna(""), test[TARGET_COL]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    # Fit ONLY on train — val/test are transform-only, no vocabulary leakage.
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"\nTF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    report = {"task": TARGET_COL, "text_column": TEXT_COL}

    # --- Logistic Regression ---
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    logreg.fit(X_train, y_train)
    val_preds_lr = logreg.predict(X_val)
    report["logreg_validation"] = evaluate(y_val, val_preds_lr, "Logistic Regression — validation")

    # --- Linear SVM ---
    svm = LinearSVC(class_weight="balanced", random_state=SEED)
    svm.fit(X_train, y_train)
    val_preds_svm = svm.predict(X_val)
    report["svm_validation"] = evaluate(y_val, val_preds_svm, "Linear SVM — validation")

    # Pick the better model by macro-F1 on validation (not test — test stays
    # untouched until the final chosen model is evaluated on it, below).
    if report["svm_validation"]["macro_f1"] >= report["logreg_validation"]["macro_f1"]:
        chosen_name, chosen_model = "linear_svm", svm
    else:
        chosen_name, chosen_model = "logistic_regression", logreg
    report["chosen_model"] = chosen_name
    print(f"\n>>> Chosen model (by validation macro-F1): {chosen_name}")

    test_preds = chosen_model.predict(X_test)
    report["test_metrics"] = evaluate(y_test, test_preds, f"{chosen_name} — TEST (final)")

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(chosen_model, MODEL_DIR / f"baseline_{chosen_name}.joblib")
    joblib.dump(logreg, MODEL_DIR / "baseline_logistic_regression.joblib")
    joblib.dump(svm, MODEL_DIR / "baseline_linear_svm.joblib")
    print(f"\nSaved vectorizer + both models to {MODEL_DIR}")

    with open(MODEL_DIR / "step4_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved metrics report to {MODEL_DIR / 'step4_baseline_report.json'}")


if __name__ == "__main__":
    main()