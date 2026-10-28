"""
Step 6b — Run the extractor over a split and report COVERAGE.

Important: the dataset has no ground-truth extraction labels, so this script
reports how often each extractor fires, not how accurate it is. It also
cross-tabs coverage against the existing humanitarian_category labels as a
sanity check (e.g. injured_or_dead_people tweets should contain casualty
counts more often than other categories). For real accuracy, hand-label the
sample CSV this script writes.

Run:
    python src/information_extraction/run_extraction.py              # validation.csv
    python src/information_extraction/run_extraction.py train.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
SAMPLE_SIZE = 100
SEED = 42


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "validation.csv"
    df = pd.read_csv(DATA_DIR / fname)
    ex = [extract(t, e) for t, e in zip(df["text_original"], df["disaster_event"])]

    df = df.assign(
        has_casualties=[bool(e["casualties"]) for e in ex],
        has_needs=[bool(e["needs"]) for e in ex],
        has_location=[bool(e["locations"]) for e in ex],
        has_candidate=[bool(e["candidate_locations"]) for e in ex],
    )
    n = len(df)
    print(f"File: {fname}  rows: {n}\n")
    coverage = {}
    for col, label in [("has_casualties", "casualty count"),
                       ("has_needs", "need tag"),
                       ("has_location", "gazetteer location"),
                       ("has_candidate", "candidate location (not in gazetteer)")]:
        coverage[label] = {"rows": int(df[col].sum()), "pct": round(100 * df[col].mean(), 2)}
        print(f"{label:40} {df[col].sum():5d}  ({100 * df[col].mean():5.1f}%)")

    print("\nCoverage by humanitarian_category (% of rows with each extraction):")
    by_cat = (df.groupby("humanitarian_category")[["has_casualties", "has_needs", "has_location"]]
                .mean().mul(100).round(1)
                .join(df.humanitarian_category.value_counts().rename("n")))
    print(by_cat.sort_values("n", ascending=False).to_string())

    from collections import Counter
    need_counts = Counter(t for e in ex for t in e["needs"])
    print("\nNeed tag counts:", dict(need_counts.most_common()))
    cand = Counter(c for e in ex for c in e["candidate_locations"])
    print("\nTop 25 candidate locations NOT in gazetteer (review to grow it):")
    for name, c in cand.most_common(25):
        print(f"  {c:4d}  {name}")

    sample = df.sample(min(SAMPLE_SIZE, n), random_state=SEED)
    out = pd.DataFrame({
        "id": sample["id"],
        "text_original": sample["text_original"],
        "disaster_event": sample["disaster_event"],
        "extracted": [json.dumps(ex[i], ensure_ascii=False) for i in sample.index],
        "gold_locations": "", "gold_casualties": "", "gold_needs": "",
    })
    out_csv = DATA_DIR / "step6_extraction_sample_for_labelling.csv"
    out.to_csv(out_csv, index=False)

    report = {"file": fname, "rows": n, "coverage": coverage,
              "need_tag_counts": dict(need_counts),
              "top_candidate_locations": cand.most_common(25)}
    with open(DATA_DIR / "step6_extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {DATA_DIR / 'step6_extraction_report.json'}")
    print(f"Saved {out_csv}  (fill gold_* columns by hand to measure accuracy)")


if __name__ == "__main__":
    main()