"""
Step 14 (part) - Evaluate the keyword disaster-type detector.

Reference label = type derived from the dataset's disaster_event column
(EVENT_TYPE in disaster_type.py). That is dataset METADATA, not a per-tweet human
label: a tweet from a hurricane event that only talks about flooding counts as
"wrong" here. The detector is run WITHOUT the event name, so there is no leakage.

Run from the project root:
    python src/information_extraction/evaluate_disaster_type.py
Writes data/processed/step8_disaster_type_report.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disaster_type import EVENT_TYPE, detect_disaster_type  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"


def evaluate(df):
    ref = df["disaster_event"].map(EVENT_TYPE)
    pred = [detect_disaster_type(t)["disaster_type"] for t in df["text_original"]]
    df = df.assign(ref=ref.values, pred=pred)
    covered = df[df.pred != "Unknown"]
    res = {
        "n": int(len(df)),
        "coverage_pct": round(100 * len(covered) / len(df), 2),
        "accuracy_on_covered_pct": round(100 * (covered.pred == covered.ref).mean(), 2) if len(covered) else None,
        "accuracy_overall_unknown_counts_as_wrong_pct": round(100 * (df.pred == df.ref).mean(), 2),
        "confusion": {r: dict(Counter(g.pred)) for r, g in df.groupby("ref")},
        "per_event": {e: {"n": int(len(g)),
                          "coverage_pct": round(100 * (g.pred != "Unknown").mean(), 2),
                          "accuracy_on_covered_pct": round(100 * (g[g.pred != "Unknown"].pred ==
                                                                  g[g.pred != "Unknown"].ref).mean(), 2)
                          if (g.pred != "Unknown").any() else None}
                      for e, g in df.groupby("disaster_event")},
    }
    wrong = covered[covered.pred != covered.ref].head(15)
    res["error_examples"] = [{"text": t[:200], "reference": r, "predicted": p, "event": e}
                             for t, r, p, e in zip(wrong.text_original, wrong.ref, wrong.pred, wrong.disaster_event)]
    return res


def main():
    df = pd.read_csv(DATA / "crisis_clean.csv")
    report = {"all_16k_tweets": evaluate(df),
              "note": "Reference labels come from the event name (dataset metadata), not per-tweet annotation."}
    out = DATA / "step8_disaster_type_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    r = report["all_16k_tweets"]
    print(f"tweets: {r['n']}  coverage: {r['coverage_pct']}%  "
          f"accuracy on covered: {r['accuracy_on_covered_pct']}%  "
          f"overall (Unknown = wrong): {r['accuracy_overall_unknown_counts_as_wrong_pct']}%")
    print("confusion (rows = reference from event name):")
    for k, v in r["confusion"].items():
        print(" ", k, v)
    print("per event:")
    for k, v in r["per_event"].items():
        print(" ", k, v)
    print("saved", out)


if __name__ == "__main__":
    main()
