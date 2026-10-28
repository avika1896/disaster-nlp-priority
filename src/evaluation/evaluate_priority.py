"""
Step 14 (part) - Compare the system's priority level with human labels.

Needs the sheet from prepare_priority_review.py with `human_priority` filled
(HIGH / MEDIUM / LOW). Nothing is reported until real labels exist.

System side:
  default  : approximation from data/processed/step7b_scored_test.csv (top
             predicted category used as probability 1.0; see score_dataset_v2.py)
  --live   : run the real pipeline (needs torch + the trained models)

Run from the project root:
    python src/evaluation/evaluate_priority.py
    python src/evaluation/evaluate_priority.py --live
    python src/evaluation/evaluate_priority.py --sheet path/to/sheet.csv
Writes data/processed/step14_priority_eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "evaluation"), str(ROOT / "src" / "ranking"),
                str(ROOT / "src" / "information_extraction")]
from metrics import prf, spearman  # noqa: E402
import priority_score as ps  # noqa: E402
from score_dataset_v2 import raw_scores  # noqa: E402

DATA = ROOT / "data" / "processed"
LEVELS = ["LOW", "MEDIUM", "HIGH"]
ORD = {l: i for i, l in enumerate(LEVELS)}


def kappa(a, b):
    m = np.zeros((3, 3))
    for x, y in zip(a, b):
        m[ORD[x], ORD[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum() / n ** 2)
    return round(float((po - pe) / (1 - pe)), 4) if pe < 1 else 1.0


def clean_labels(series, col):
    s = series.fillna("").astype(str).str.strip().str.upper()
    bad = sorted(set(s[(s != "") & (~s.isin(LEVELS))]))
    if bad:
        sys.exit(f"Column {col} has invalid values {bad}. Use only HIGH, MEDIUM or LOW.")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(DATA / "priority_review_sheet.csv"))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", default=str(DATA / "step14_priority_eval_report.json"))
    a = ap.parse_args()

    sheet = pd.read_csv(a.sheet)
    sheet["h1"] = clean_labels(sheet["human_priority"], "human_priority")
    sheet["h2"] = clean_labels(sheet["human_priority_2"], "human_priority_2")
    sheet = sheet[sheet.h1 != ""]
    if sheet.empty:
        print("No labels yet. Fill human_priority in", a.sheet, "(see docs/priority_labelling_guide.md).")
        return

    scored = pd.read_csv(DATA / "step7b_scored_test.csv")
    df = sheet.merge(scored, on="id", how="left", suffixes=("", "_s"))
    if df.pred_category.isna().any():
        sys.exit("Some sheet ids are not in step7b_scored_test.csv.")
    if a.live:
        from src.pipeline import analyze_many
        res = analyze_many(None, list(zip(df.text_original, df.disaster_event)))
        df["system_score"] = [r["priority_score"] for r in res]
    else:
        df["system_score"] = raw_scores(df)
    df["system_level"] = [ps.level(x) for x in df.system_score]

    res = prf(list(df.h1), list(df.system_level), LEVELS)
    res.update({
        "n_labelled": int(len(df)),
        "system_scores": "live pipeline" if a.live else "approximation (top category as probability 1.0)",
        "human_label_distribution": df.h1.value_counts().to_dict(),
        "spearman_human_vs_score": spearman([ORD[x] for x in df.h1], df.system_score),
    })
    both = df[df.h2 != ""]
    if len(both) >= 10:
        res["reviewer_agreement_kappa"] = kappa(list(both.h1), list(both.h2))
        res["n_double_labelled"] = int(len(both))
    cols = ["text_original", "h1", "system_level", "system_score"]
    fn = df[(df.h1 == "HIGH") & (df.system_level == "LOW")]
    fp = df[(df.h1 == "LOW") & (df.system_level == "HIGH")]
    res["high_missed_as_low_count"], res["low_flagged_high_count"] = int(len(fn)), int(len(fp))
    res["error_examples_high_missed_as_low"] = fn[cols].head(8).to_dict("records")
    res["error_examples_low_flagged_high"] = fp[cols].head(8).to_dict("records")
    res["note"] = "Human labels are judgements on a small sample (see docs/priority_labelling_guide.md), not ground truth."
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"labelled posts: {res['n_labelled']}  accuracy: {res['accuracy']}  macro-F1: {res['macro']['f1']}  "
          f"Spearman: {res['spearman_human_vs_score']}")
    print("confusion [rows=human, cols=system]", LEVELS, res["confusion_matrix"])
    print("saved", a.out)


if __name__ == "__main__":
    main()
