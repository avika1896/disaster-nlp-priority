"""
Step 10/14 - Score a split with the new multi-factor score and report the tier mix.

APPROXIMATION: step7b_scored_*.csv stores only the TOP predicted category and the
informative probability, not the full category probabilities. This script treats
the top category as probability 1.0. The live pipeline uses the real
probabilities, so scores differ slightly. Use this only to choose tier cut-offs
and to sanity-check the score; re-run with real probabilities (score_dataset.py
style) on your machine to refine them.

Run from the project root:
    python src/ranking/score_dataset_v2.py validation
    python src/ranking/score_dataset_v2.py test
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "src" / "ranking"), str(ROOT / "src" / "information_extraction")]
import priority_score as ps  # noqa: E402
from extractor import extract  # noqa: E402
from location import extract_location  # noqa: E402
from urgency import assess_urgency  # noqa: E402

DATA = ROOT / "data" / "processed"


def raw_scores(df):
    out = []
    for t, ev, p, cat in zip(df.text_original, df.disaster_event, df.c_p_informative, df.pred_category):
        r = ps.priority_score_v2(p, {cat: 1.0}, extract(t, ev), assess_urgency(t), extract_location(t, ev), t)
        out.append(r["priority_score"])
    return np.array(out)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    df = pd.read_csv(DATA / f"step7b_scored_{split}.csv")
    s = raw_scores(df)
    df = df.assign(v2_score=s)
    qs = {q: round(float(np.percentile(s, q)), 1) for q in (50, 65, 70, 75, 80, 85, 90, 95, 99)}
    print(f"{split}: n={len(df)}  mean={s.mean():.1f}  percentiles={qs}")
    hi, med = ps.PRIORITY_V2_HIGH, ps.PRIORITY_V2_MEDIUM
    lvl = pd.Series([ps.level(x) for x in s])
    pct = (lvl.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"cut-offs in weights.py: HIGH>={hi} MEDIUM>={med} -> tier % {pct}")
    print("mean v2 score by TRUE humanitarian category (sanity check):")
    by = df.groupby("humanitarian_category").v2_score.mean().round(1).sort_values(ascending=False)
    print(by.to_string())
    rep = {"split": split, "n": int(len(df)), "approximation": "top category used as probability 1.0",
           "cutoffs": {"high": hi, "medium": med}, "tier_pct": pct, "percentiles": qs,
           "mean_score_by_true_category": by.to_dict()}
    (DATA / f"step10_priority_v2_report_{split}.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
