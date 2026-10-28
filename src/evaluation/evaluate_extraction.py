"""
Step 14 (part) - Precision / recall / F1 for location, casualty and need extraction.

Needs the hand-labelled sample data/processed/step6_extraction_sample_for_labelling.csv
(written by src/information_extraction/run_extraction.py). Fill ALL THREE gold
columns for each post you label (write `none` if the post has nothing):
    gold_locations  : place names, separated by ;      e.g.  Houston; Texas
    gold_casualties : type:count, separated by ;       e.g.  deaths:217; injured:30
                      (types: deaths, injured, missing, casualties)
    gold_needs      : need tags separated by ;         e.g.  water; rescue
                      (tags: water food shelter medical rescue evacuation
                       power_utilities donations explicit_request; a tag means
                       "the post mentions this need", same as the extractor)
Rows with any gold column left empty are treated as not labelled and skipped.
Place names are compared in lower case, exact match.

Run from the project root:
    python src/evaluation/evaluate_extraction.py [--sample path.csv]
Writes data/processed/step14_extraction_eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from metrics import set_prf  # noqa: E402

DATA = ROOT / "data" / "processed"


def parse(cell):
    s = str(cell).strip().lower()
    if s in ("", "nan", "none"):
        return set()
    return {x.strip() for x in s.split(";") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(DATA / "step6_extraction_sample_for_labelling.csv"))
    ap.add_argument("--out", default=str(DATA / "step14_extraction_eval_report.json"))
    a = ap.parse_args()

    df = pd.read_csv(a.sample, dtype=str, keep_default_na=False)
    gold_cols = ["gold_locations", "gold_casualties", "gold_needs"]
    df = df[(df[gold_cols].apply(lambda c: c.str.strip()) != "").all(axis=1)].reset_index(drop=True)
    if df.empty:
        print("No labelled rows yet. Fill gold_locations, gold_casualties and gold_needs in", a.sample)
        print("(write `none` when a post has nothing). See the top of this file for the format.")
        return

    ext = [json.loads(x) for x in df.extracted]
    pred = {
        "location_gazetteer_only": [{p.lower() for p in e["locations"]} for e in ext],
        "location_with_candidates": [{p.lower() for p in e["locations"] + e["candidate_locations"]} for e in ext],
        "casualties": [{f"{c['type']}:{c['count']}" for c in e["casualties"]} for e in ext],
        "needs": [set(e["needs"]) for e in ext],
    }
    gold = {"location_gazetteer_only": [parse(x) for x in df.gold_locations],
            "location_with_candidates": [parse(x) for x in df.gold_locations],
            "casualties": [parse(x) for x in df.gold_casualties],
            "needs": [parse(x) for x in df.gold_needs]}
    report = {"n_labelled": int(len(df)),
              "note": "Gold labels were written by hand on a small sample; they are not ground truth.",
              "scores": {k: set_prf(gold[k], pred[k]) for k in pred}, "errors": {}}
    for k in pred:
        rows = [{"text": t[:200], "gold": sorted(g), "predicted": sorted(p)}
                for t, g, p in zip(df.text_original, gold[k], pred[k]) if g != p][:8]
        report["errors"][k] = rows
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("labelled posts:", report["n_labelled"])
    for k, v in report["scores"].items():
        print(f"  {k:26} P={v['precision']}  R={v['recall']}  F1={v['f1']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
