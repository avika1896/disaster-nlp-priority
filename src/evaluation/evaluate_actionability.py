"""
Step 14 (part) - Evaluate actionability.

IMPORTANT: there is no human "actionable" label in CrisisMMD. Both sides use the
mapping in src/information_extraction/actionability.py:
    reference   = mapping applied to the DATASET humanitarian_category
    prediction  = mapping applied to the MODEL's predicted category
So this measures how well the 7-class model's mistakes cross the actionable /
not-actionable line. It is not a test against human actionability judgements.

Uses data/processed/step7b_scored_<split>.csv (it stores the top predicted
category only). Run from the project root:
    python src/evaluation/evaluate_actionability.py            # validation and test
Writes data/processed/step14_actionability_report.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "src" / "evaluation"), str(ROOT / "src" / "information_extraction")]
from actionability import actionable_from_label  # noqa: E402
from metrics import prf  # noqa: E402

DATA = ROOT / "data" / "processed"
LABELS = ["not_actionable", "actionable"]


def name(b):
    return "actionable" if b else "not_actionable"


def evaluate(split):
    df = pd.read_csv(DATA / f"step7b_scored_{split}.csv")
    ref = df.humanitarian_category.map(actionable_from_label)
    pred = df.pred_category.map(actionable_from_label)
    assert ref.notna().all() and pred.notna().all(), "unmapped category name in the data"
    res = prf([name(x) for x in ref], [name(x) for x in pred], LABELS)
    res["n"] = int(len(df))
    res["share_actionable_in_reference_pct"] = round(100 * float(ref.mean()), 2)
    fp = df[(~ref.astype(bool)) & (pred.astype(bool))]
    fn = df[(ref.astype(bool)) & (~pred.astype(bool))]
    cols = ["text_original", "humanitarian_category", "pred_category"]
    res["false_positive_count"], res["false_negative_count"] = int(len(fp)), int(len(fn))
    res["false_positive_examples"] = fp[cols].head(8).to_dict("records")
    res["false_negative_examples"] = fn[cols].head(8).to_dict("records")
    return res


def main():
    report = {"note": "Reference = mapping applied to dataset category, NOT a human actionability label.",
              "validation": evaluate("validation"), "test": evaluate("test")}
    (DATA / "step14_actionability_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for split in ("validation", "test"):
        r = report[split]
        print(f"{split}: n={r['n']} accuracy={r['accuracy']} macro-F1={r['macro']['f1']} "
              f"actionable: P={r['per_class']['actionable']['precision']} R={r['per_class']['actionable']['recall']} "
              f"F1={r['per_class']['actionable']['f1']}  FP={r['false_positive_count']} FN={r['false_negative_count']}")
        print("   confusion [rows=reference, cols=predicted] labels", LABELS, r["confusion_matrix"])
    print("saved", DATA / "step14_actionability_report.json")


if __name__ == "__main__":
    main()
