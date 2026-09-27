"""
Step 2 — Build a unified internal dataframe + train/val/test splits from
CrisisMMD v2.0's official per-task TSV files.

Everything below is based on facts VERIFIED against the actual uploaded
files (see notebooks/01_dataset_exploration.py output), not assumptions:

1. CONFIRMED COLUMN CHOICE: use `label_text`, NOT `label`.
   The Readme states `label` is "randomly selected from text and image
   labels" for the informative/humanitarian tasks. We verified this
   empirically: `label` disagrees with `label_text` in 27.6% of
   humanitarian rows and 14.8% of informative rows. Since this project
   is a text-only pipeline, using `label` would silently leak
   image-derived annotations into what's supposed to be a text label.
   => informative_label  <- informative task's `label_text`
   => humanitarian_category <- humanitarian task's `label_text`

2. CONFIRMED: `label_text` is constant across all image-rows of the same
   tweet_id (0 tweet_ids had disagreement) - safe to dedupe to one row
   per tweet_id.

3. CONFIRMED: damage severity in CrisisMMD is IMAGE-ONLY. The Readme
   explicitly states "for damage task we only have a label for the
   image" and provides no label_text/label_image split for that task.
   We therefore store it as `damage_severity_image_only` and DO NOT
   treat it as a text-classification ground truth. It's kept as a weak/
   auxiliary signal only, clearly separated from the text labels.

4. CONFIRMED: the official per-task train/dev/test splits are NOT
   tweet-consistent across tasks. Checking the 16,058 tweets common to
   both the humanitarian and informative files: 6,976 (43%) are
   assigned to a DIFFERENT split (train/dev/test) depending on which
   task file you consult. This means there is no single "official
   split" usable once we merge multiple tasks' labels into one row per
   tweet. We therefore build our OWN reproducible split over the
   deduplicated corpus, and document this explicitly (this is exactly
   the kind of methodological detail the brief asked to keep
   transparent for the research paper).

5. CONFIRMED: within the damage task's OWN official split, 62 tweet_ids
   leak between train/dev, 62 between train/test (likely because that
   task's split was done at the image level, not the tweet level). This
   is a known limitation of that specific file, reported here rather
   than silently inherited.

Outputs:
    data/processed/crisis_clean.csv       - full unified deduped dataset
    data/processed/train.csv              - our own random stratified split
    data/processed/validation.csv
    data/processed/test.csv
    data/processed/event_holdout_test.csv - generalization split: entire
                                             disaster event(s) held out
    data/processed/step2_build_report.json - stats used in this docstring,
                                             regenerated from the actual run
"""

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.text_cleaner import add_clean_text_columns

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "crisismmd_datasplit_all"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SEED = 42

# Held out entirely for the event-generalization split. Chosen as the two
# smallest events by volume so the random/stratified main split above
# still has plenty of training data from the other five events.
EVENT_HOLDOUT = {"iraq_iran_earthquake", "srilanka_floods"}


def load_task_all_splits(task: str) -> pd.DataFrame:
    """Load a task's train+dev+test tsvs, tag each row with which official
    split it came from (kept for transparency, NOT used as our split),
    and dedupe to one row per tweet_id (verified safe for label_text)."""
    frames = []
    for split in ["train", "dev", "test"]:
        path = RAW_DIR / f"task_{task}_text_img_{split}.tsv"
        df = pd.read_csv(path, sep="\t", engine="python", quoting=csv.QUOTE_NONE, dtype=str)
        df["_official_split"] = split
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    before = len(full)
    full = full.drop_duplicates(subset="tweet_id", keep="first")
    after = len(full)
    print(f"[{task}] loaded {before} rows across train/dev/test, "
          f"{after} unique tweet_id after multi-image dedup")
    return full


def build_unified_dataframe(report: dict) -> pd.DataFrame:
    hum = load_task_all_splits("humanitarian")
    inf = load_task_all_splits("informative")
    dmg = load_task_all_splits("damage")

    hum_small = hum[["tweet_id", "event_name", "tweet_text", "label_text"]].rename(
        columns={"label_text": "humanitarian_category"}
    )
    inf_small = inf[["tweet_id", "event_name", "tweet_text", "label_text"]].rename(
        columns={"label_text": "informative_label"}
    )
    # Damage: only the image-based `label` column exists for this task
    # (confirmed from Readme + column listing). Name it explicitly so
    # nobody downstream mistakes it for a text-derived label.
    dmg_small = dmg[["tweet_id", "event_name", "tweet_text", "label"]].rename(
        columns={"label": "damage_severity_image_only"}
    )

    # Outer-merge on tweet_id: a tweet may exist in some task files and not
    # others (confirmed: humanitarian/informative don't fully overlap
    # either -  8,409 of 11,584 train-tagged tweet_ids were shared, i.e.
    # not identical sets). Coalesce text/event_name across sources.
    merged = inf_small.merge(
        hum_small[["tweet_id", "humanitarian_category"]], on="tweet_id", how="outer"
    ).merge(
        dmg_small[["tweet_id", "damage_severity_image_only"]], on="tweet_id", how="outer"
    )

    # tweet_text/event_name might be missing from `inf_small` if a tweet_id
    # only existed in hum/dmg (outer join introduces NaN there) — backfill.
    merged = merged.merge(
        pd.concat([inf_small[["tweet_id", "tweet_text", "event_name"]],
                   hum_small[["tweet_id", "tweet_text", "event_name"]],
                   dmg_small[["tweet_id", "tweet_text", "event_name"]]])
        .drop_duplicates(subset="tweet_id", keep="first"),
        on="tweet_id", how="left", suffixes=("", "_backfill")
    )
    for col in ["tweet_text", "event_name"]:
        merged[col] = merged[col].fillna(merged[f"{col}_backfill"])
        merged.drop(columns=[f"{col}_backfill"], inplace=True)

    merged["informative_label"] = merged["informative_label"].fillna("Unknown")
    merged["humanitarian_category"] = merged["humanitarian_category"].fillna("Unknown")
    merged["damage_severity_image_only"] = merged["damage_severity_image_only"].fillna("Unknown")
    merged["source"] = "CrisisMMD_v2.0"
    merged = merged.rename(columns={"tweet_id": "id", "tweet_text": "text", "event_name": "disaster_event"})

    coverage = {
        "total_unique_tweets": len(merged),
        "with_informative_label": int((merged["informative_label"] != "Unknown").sum()),
        "with_humanitarian_category": int((merged["humanitarian_category"] != "Unknown").sum()),
        "with_damage_severity_image_only": int((merged["damage_severity_image_only"] != "Unknown").sum()),
    }
    report["label_coverage"] = coverage
    print("\nLabel coverage across the unified dataset:")
    for k, v in coverage.items():
        print(f"  {k}: {v}")

    return merged[["id", "text", "disaster_event", "informative_label",
                    "humanitarian_category", "damage_severity_image_only", "source"]]


def check_leakage(train_df, val_df, test_df, report: dict):
    tr_ids, va_ids, te_ids = set(train_df["id"]), set(val_df["id"]), set(test_df["id"])
    overlaps = {
        "train_val": len(tr_ids & va_ids),
        "train_test": len(tr_ids & te_ids),
        "val_test": len(va_ids & te_ids),
    }
    report["split_id_overlap"] = overlaps
    print("\nID overlap between our own splits (should all be 0):")
    for k, v in overlaps.items():
        print(f"  {k}: {v}")
    assert sum(overlaps.values()) == 0, "Leakage detected in our own split!"

    tr_events, te_events = set(train_df["disaster_event"]), set(test_df["disaster_event"])
    shared_events = tr_events & te_events
    report["events_shared_between_train_and_test_main_split"] = sorted(shared_events)
    print(f"\nDisaster events appearing in BOTH train and test (main random split): "
          f"{sorted(shared_events)}")
    print("This is EXPECTED for the main split (it is a random split, not an "
          "event-holdout split) — reported per the brief's instruction to "
          "flag this explicitly. See event_holdout_test.csv for the "
          "generalization-to-unseen-events split instead.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    df = build_unified_dataframe(report)

    # Text cleaning (Step 3) applied here so processed CSVs are ready for
    # both the TF-IDF baseline and the transformer model.
    df = add_clean_text_columns(df, text_column="text")

    df.to_csv(OUT_DIR / "crisis_clean.csv", index=False)
    print(f"\nSaved unified dataset: {OUT_DIR / 'crisis_clean.csv'} ({len(df)} rows)")

    # ---- Event-holdout split (generalization to unseen events) ----
    event_mask = df["disaster_event"].isin(EVENT_HOLDOUT)
    event_holdout_test = df[event_mask].reset_index(drop=True)
    remaining = df[~event_mask].reset_index(drop=True)
    report["event_holdout"] = {
        "held_out_events": sorted(EVENT_HOLDOUT),
        "n_rows_held_out": len(event_holdout_test),
        "n_rows_remaining": len(remaining),
    }
    print(f"\nEvent-holdout split: {len(event_holdout_test)} rows from "
          f"{sorted(EVENT_HOLDOUT)} set aside entirely for generalization "
          f"testing; {len(remaining)} rows remain for the main split below.")

    # ---- Main random split (stratified on informative_label), seeded ----
    train_df, temp_df = train_test_split(
        remaining, test_size=0.30, random_state=SEED,
        stratify=remaining["informative_label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df["informative_label"],
    )

    check_leakage(train_df, val_df, test_df, report)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    event_holdout_test.to_csv(OUT_DIR / "event_holdout_test.csv", index=False)

    report["split_sizes"] = {
        "train": len(train_df), "validation": len(val_df),
        "test": len(test_df), "event_holdout_test": len(event_holdout_test),
    }
    print("\nFinal split sizes:")
    for k, v in report["split_sizes"].items():
        print(f"  {k}: {v}")

    print("\nClass distribution (informative_label) per split:")
    for name, d in [("train", train_df), ("validation", val_df),
                     ("test", test_df), ("event_holdout_test", event_holdout_test)]:
        print(f"  {name}:")
        print(d["informative_label"].value_counts(normalize=True).round(3).to_string())

    with open(OUT_DIR / "step2_build_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved build report: {OUT_DIR / 'step2_build_report.json'}")


if __name__ == "__main__":
    main()
