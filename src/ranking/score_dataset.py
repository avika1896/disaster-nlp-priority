"""
Step 7b — Score a whole split and sanity-check the priority score.

Pipeline per tweet:
  text_clean    -> Step 5 transformer -> p_informative
  text_clean    -> Step 7a transformer -> category probabilities
  text_original -> Step 6 extractor   -> casualties, needs
  all of the above -> priority.priority_score (hand-set weights)

There is NO ground-truth priority in the dataset, so this script cannot report
"accuracy". It reports face-validity checks instead:
  - mean score per TRUE humanitarian_category (should follow the intended order)
  - share of tweets in each tier (to tune the thresholds in weights.py)
  - the top-ranked tweets, for you to read and judge

Run (after Steps 5 and 7a have been trained):
    python src/ranking/score_dataset.py                 # validation.csv
    python src/ranking/score_dataset.py test.csv
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ranking"))
sys.path.insert(0, str(ROOT / "src" / "information_extraction"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from extractor import extract  # noqa: E402
from priority import priority_score  # noqa: E402
from train_transformer import DATA_DIR, MAX_LEN, TEXT_COL, get_device  # noqa: E402

INFORMATIVE_DIR = ROOT / "models" / "transformer" / "best"
HUMANITARIAN_DIR = ROOT / "models" / "transformer" / "humanitarian" / "best"
OUT_DIR = ROOT / "data" / "processed"


@torch.no_grad()
def predict_proba(model_dir, texts, device, batch_size=64):
    """Return (probabilities [n, k], class names in model order)."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out), names


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "validation.csv"
    for d in (INFORMATIVE_DIR, HUMANITARIAN_DIR):
        if not d.exists():
            raise SystemExit(f"Missing trained model: {d}")
    device = get_device()
    df = pd.read_csv(DATA_DIR / fname)
    texts = df[TEXT_COL].fillna("").tolist()
    print(f"Scoring {fname}: {len(df)} tweets on {device}")

    p_inf_all, inf_names = predict_proba(INFORMATIVE_DIR, texts, device)
    p_cat_all, cat_names = predict_proba(HUMANITARIAN_DIR, texts, device)
    inf_idx = inf_names.index("informative")

    rows = []
    for i, (orig, event) in enumerate(zip(df["text_original"], df["disaster_event"])):
        ext = extract(orig, event)
        res = priority_score(p_inf_all[i, inf_idx], dict(zip(cat_names, p_cat_all[i])), ext)
        rows.append({
            "pred_category": cat_names[int(p_cat_all[i].argmax())],
            "priority_score": res["score"],
            "priority_tier": res["tier"],
            **{f"c_{k}": v for k, v in res["components"].items()},
            "max_deaths": ext["max_deaths"],
            "needs": ",".join(ext["needs"]),
            "locations": ",".join(ext["locations"]),
        })
    out = pd.concat([df[["id", "text_original", "disaster_event", "humanitarian_category"]],
                     pd.DataFrame(rows)], axis=1)
    out_csv = OUT_DIR / f"step7b_scored_{Path(fname).stem}.csv"
    out.to_csv(out_csv, index=False)

    tiers = out["priority_tier"].value_counts(normalize=True).mul(100).round(1)
    print("\nTier distribution (% of tweets):")
    print(tiers.to_string())

    by_cat = (out.groupby("humanitarian_category")["priority_score"]
                 .agg(["mean", "median", "count"]).round(1)
                 .sort_values("mean", ascending=False))
    print("\nPriority score by TRUE humanitarian_category (face-validity check):")
    print(by_cat.to_string())

    print("\nTop 15 tweets by priority score:")
    for _, r in out.sort_values("priority_score", ascending=False).head(15).iterrows():
        print(f"  {r.priority_score:5.1f} [{r.pred_category}] {str(r.text_original)[:110]}")

    report = {"file": fname, "n": len(out),
              "tier_pct": tiers.to_dict(),
              "mean_score_by_true_category": by_cat["mean"].to_dict()}
    with open(OUT_DIR / f"step7b_priority_report_{Path(fname).stem}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()