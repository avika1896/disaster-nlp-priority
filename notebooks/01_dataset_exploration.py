"""
Step 1 — CrisisMMD Dataset Exploration
========================================
This script makes NO assumptions about column names or file layout.
It scans data/raw/ for CSV/TSV files, and for each one it:
  - reports filename, shape, columns, dtypes
  - guesses which column is the tweet text (longest avg string length)
  - guesses which columns are categorical labels (few unique values, object dtype)
  - prints class distributions for those guessed label columns
  - checks missing values per column
  - checks duplicate rows and duplicate text values
  - checks whether the same tweet_id / text appears across multiple files
    (a common source of leakage in CrisisMMD's per-task/per-split tsvs)

Run:
    python notebooks/01_dataset_exploration.py

This is meant to be run once you've downloaded and unzipped CrisisMMD's
annotation files into data/raw/ (any subfolder structure is fine — it
recurses). Nothing here is hardcoded to CrisisMMD's known schema on
purpose: the brief calls for inspecting the ACTUAL files, not assuming.
"""

import os
import csv
import json
import glob
from pathlib import Path
from collections import defaultdict

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
STATS_OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "step1_exploration_stats.json"

# .txt deliberately excluded: CrisisMMD ships a Readme.txt alongside the
# real .tsv data files, and it isn't tabular data. If any project's raw
# text data ever legitimately has a .txt extension, add it back here.
SUPPORTED_EXTS = {".csv", ".tsv"}


def sniff_separator(path: Path) -> str:
    """CrisisMMD's official splits are tab-separated; other exports may be
    comma-separated. Sniff instead of assuming."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    tab_count = first_line.count("\t")
    comma_count = first_line.count(",")
    if tab_count > comma_count:
        return "\t"
    return ","


def find_data_files():
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(str(RAW_DIR / "**" / f"*{ext}"), recursive=True))
    return sorted(set(files))


def guess_text_column(df: pd.DataFrame):
    """Text column = string-like column with the highest mean string length
    among columns that look like free text (not just short category codes).

    Note: uses pd.api.types.is_string_dtype rather than `dtype == object`.
    Depending on the pandas version/backend, string columns can come back
    as numpy object dtype OR as the newer pandas StringDtype (shown as
    'str' in df.dtypes) — comparing directly to `object` misses the
    latter and silently returns no candidates.
    """
    candidates = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                mean_len = df[col].astype(str).str.len().mean()
            except Exception:
                continue
            candidates[col] = mean_len
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def guess_label_columns(df: pd.DataFrame, text_col, max_unique=25):
    """Label-like columns = object/categorical dtype, low cardinality,
    and not the text column itself."""
    label_cols = []
    n = len(df)
    for col in df.columns:
        if col == text_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= max_unique and nunique < n * 0.5:
            label_cols.append(col)
    return label_cols


def guess_id_column(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if "id" in cl and df[col].nunique() > 0.9 * len(df):
            return col
    return None


def explore_file(path: str, all_text_index: defaultdict):
    path = Path(path)
    sep = sniff_separator(path)
    try:
        # quoting=csv.QUOTE_NONE: these TSV files are not quote-escaped —
        # tweet_text can contain stray literal double-quote characters
        # (e.g. quoting someone in a tweet), and pandas' default quoting
        # misinterprets those as the start of a quoted field, silently
        # swallowing the rest of the line into one field and desyncing
        # column counts (this was verified against the real files: it
        # was dropping dozens of rows per file before this fix).
        df = pd.read_csv(
            path, sep=sep, engine="python", on_bad_lines="warn",
            quoting=csv.QUOTE_NONE, dtype=str,
        )
    except Exception as e:
        print(f"  !! Failed to read {path.name}: {e}")
        return None

    text_col = guess_text_column(df)
    label_cols = guess_label_columns(df, text_col)
    id_col = guess_id_column(df)

    report = {
        "file": str(path.relative_to(RAW_DIR)),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "guessed_text_column": text_col,
        "guessed_label_columns": label_cols,
        "guessed_id_column": id_col,
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    print(f"\n{'=' * 80}")
    print(f"FILE: {report['file']}")
    print(f"{'=' * 80}")
    print(f"Rows: {report['rows']}  |  Columns: {report['columns']}")
    print(f"Guessed text column : {text_col}")
    print(f"Guessed id column   : {id_col}")
    print(f"Guessed label cols  : {label_cols}")

    print("\n-- Missing values per column --")
    miss = df.isna().sum()
    print(miss[miss > 0] if miss.any() else "  (none)")

    print(f"\n-- Duplicate full rows: {report['duplicate_rows']} --")

    if text_col:
        dup_text = df[text_col].astype(str).duplicated().sum()
        report["duplicate_text_values"] = int(dup_text)
        print(f"-- Duplicate text values (same {text_col} repeated): {dup_text} --")

        # track text -> files, to check cross-file overlap later
        for t in df[text_col].astype(str):
            all_text_index[t].append(report["file"])

    for col in label_cols:
        print(f"\n-- Class distribution: {col} --")
        vc = df[col].value_counts(dropna=False)
        print(vc)
        report.setdefault("class_distributions", {})[col] = vc.to_dict()

    return report


def main():
    if not RAW_DIR.exists() or not find_data_files():
        print(f"No data files found under {RAW_DIR}.")
        print("Download CrisisMMD's annotation split and unzip it into data/raw/, then re-run.")
        print("Official source: https://crisisnlp.qcri.org/crisismmd")
        return

    files = find_data_files()
    print(f"Found {len(files)} candidate data file(s) under {RAW_DIR}:")
    for f in files:
        print(f"  - {Path(f).relative_to(RAW_DIR)}")

    all_text_index = defaultdict(list)
    reports = []
    for f in files:
        r = explore_file(f, all_text_index)
        if r:
            reports.append(r)

    # Cross-file duplicate / overlap check
    print(f"\n{'=' * 80}")
    print("CROSS-FILE TEXT OVERLAP CHECK")
    print(f"{'=' * 80}")
    overlap = {t: fs for t, fs in all_text_index.items() if len(set(fs)) > 1}
    print(f"Number of distinct text values appearing in more than one file: {len(overlap)}")
    if overlap:
        sample = list(overlap.items())[:5]
        print("Sample overlaps (text -> files):")
        for t, fs in sample:
            print(f"  {t[:80]!r} -> {sorted(set(fs))}")

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUT, "w") as f:
        json.dump(
            {
                "files": reports,
                "num_files_with_cross_overlap_texts": len(overlap),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved full stats to {STATS_OUT}")


if __name__ == "__main__":
    main()
