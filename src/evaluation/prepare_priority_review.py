"""
Step 14 (part) - Make the sheet for manual priority labelling.

Takes a sample of the TEST split, balanced across the dataset's humanitarian
categories (so serious posts are included, not only the many low ones). The
sheet contains the post text only. See docs/priority_labelling_guide.md.

Run from the project root:
    python src/evaluation/prepare_priority_review.py [n=100]
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"
SEED = 42


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    df = pd.read_csv(DATA / "step7b_scored_test.csv")
    cats = sorted(df.humanitarian_category.unique())
    per = max(1, n // len(cats))
    parts = [g.sample(min(per, len(g)), random_state=SEED) for _, g in df.groupby("humanitarian_category")]
    sample = pd.concat(parts)
    if len(sample) < n:
        rest = df[~df.id.isin(sample.id)]
        sample = pd.concat([sample, rest.sample(min(n - len(sample), len(rest)), random_state=SEED)])
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle

    sheet = pd.DataFrame({"id": sample.id, "text_original": sample.text_original,
                          "human_priority": "", "human_priority_2": "", "notes": ""})
    key = sample[["id", "disaster_event", "humanitarian_category"]]
    sheet.to_csv(DATA / "priority_review_sheet.csv", index=False)
    key.to_csv(DATA / "priority_review_key.csv", index=False)
    print(f"Wrote {len(sheet)} rows to data/processed/priority_review_sheet.csv (fill human_priority).")
    print("Guide: docs/priority_labelling_guide.md")


if __name__ == "__main__":
    main()
