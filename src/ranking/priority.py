"""
Step 7b — Transparent priority scoring. No machine learning in this file:
it only combines model outputs and extraction results with the hand-set
weights in weights.py, and returns every component so a score can be explained.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import (  # noqa: E402
    CASUALTY_SATURATION, CATEGORY_WEIGHT, COMPONENT_WEIGHTS, HIGH_THRESHOLD,
    MEDIUM_THRESHOLD, NEED_WEIGHT, NOT_HUMANITARIAN,
)


def category_score(category_probs: dict) -> float:
    """Expected category weight over humanitarian classes (renormalised)."""
    unknown = set(category_probs) - set(CATEGORY_WEIGHT)
    if unknown:
        raise ValueError(f"No weight defined for categories: {sorted(unknown)}")
    hum = {k: v for k, v in category_probs.items() if k != NOT_HUMANITARIAN}
    total = sum(hum.values())
    if total <= 1e-9:
        return 0.0
    return sum(p * CATEGORY_WEIGHT[k] for k, p in hum.items()) / total


def casualty_score(casualties: list) -> float:
    """Log-scaled largest count (any type) found in the text; 0 if none."""
    counts = [c["count"] for c in casualties]
    if not counts:
        return 0.0
    return min(1.0, math.log10(1 + max(counts)) / math.log10(1 + CASUALTY_SATURATION))


def need_score(needs: list) -> float:
    return max((NEED_WEIGHT.get(n, 0.0) for n in needs), default=0.0)


def tier(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def priority_score(p_informative: float, category_probs: dict, extraction: dict) -> dict:
    p_inf = min(1.0, max(0.0, float(p_informative)))
    comp = {
        "category": category_score(category_probs),
        "casualty": casualty_score(extraction.get("casualties", [])),
        "need": need_score(extraction.get("needs", [])),
    }
    inner = sum(COMPONENT_WEIGHTS[k] * comp[k] for k in comp)
    score = 100.0 * p_inf * inner
    return {
        "score": round(score, 2),
        "tier": tier(score),
        "components": {"p_informative": p_inf, **{k: round(v, 4) for k, v in comp.items()}},
    }