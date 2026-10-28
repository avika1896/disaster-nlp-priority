"""
Step 10 - Explainable multi-factor priority score (0-100).

No machine learning here. It combines model outputs (informativeness, category
probabilities) with rule-based features (extraction, urgency, location) using
HAND-SET weights from weights.py. The weights are NOT scientifically
established and NOT learned (the dataset has no priority labels); they are
configurable so experiments can compare settings.

    priority = 100 * p_informative * sum_k( W[k] * component[k] )

Components, each 0..1:
  urgency       engineered indicator score / 100        (urgency.py; not ground truth)
  human_impact  max(casualty count score, category-weighted probability of
                injured/affected/missing people)
  assistance    strongest need tag found, or half of the rescue/donation
                category probability if that is higher
  damage        engineered from damage words in the text (severe 1.0, moderate
                0.5) or 0.7 x P(infrastructure damage), whichever is higher.
                NOTE: CrisisMMD's own damage-severity label is image-only, so
                it cannot be used for text.
  location      confidence of the location found (location.py)

p_informative multiplies everything, so posts that are probably not informative
sink to the bottom.

priority_score_v2() returns {"priority_score", "priority_level", "reason": [...],
"components", "damage_severity"}.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from priority import casualty_score, need_score  # noqa: E402
from urgency import prep_text  # noqa: E402
from weights import (  # noqa: E402
    HUMAN_IMPACT_CATEGORY_WEIGHT, PRIORITY_V2_HIGH, PRIORITY_V2_MEDIUM, PRIORITY_V2_WEIGHTS,
)

_SEVERE = re.compile(r"\bdestroy\w*|\bcollaps\w+|\bdevastat\w+|washed away|\bflatten\w*|wiped out|"
                     r"\bcatastroph\w+|\bobliterat\w+|\brazed\b|\blevell?ed\b", re.I)
_MODERATE = re.compile(r"\bdamag\w+|\bflooded\b|\bsubmerged\b|power outage|without power|\bno power\b|"
                       r"\bblackout|\bblocked\b|\bbroken\b|\bcracked\b|\bburn\w*|\bswept\b|\buprooted\b", re.I)


def damage_severity(text, category_probs):
    """Return (label, score 0..1). Label is Severe / Moderate / Unknown."""
    t = prep_text(text)
    keyword = 1.0 if _SEVERE.search(t) else 0.5 if _MODERATE.search(t) else 0.0
    infra = 0.7 * category_probs.get("infrastructure_and_utility_damage", 0.0)
    score = max(keyword, infra)
    if keyword == 1.0:
        label = "Severe"
    elif keyword == 0.5 or infra >= 0.35:
        label = "Moderate"
    else:
        label = "Unknown"
    return label, score


def level(score: float) -> str:
    if score >= PRIORITY_V2_HIGH:
        return "HIGH"
    if score >= PRIORITY_V2_MEDIUM:
        return "MEDIUM"
    return "LOW"


def priority_score_v2(p_informative, category_probs, extraction, urgency, location, text) -> dict:
    p_inf = min(1.0, max(0.0, float(p_informative)))
    cats = category_probs or {}

    human_prob = sum(w * cats.get(k, 0.0) for k, w in HUMAN_IMPACT_CATEGORY_WEIGHT.items())
    cas = casualty_score(extraction.get("casualties", []))
    needs = extraction.get("needs", [])
    dmg_label, dmg = damage_severity(text, cats)
    comp = {
        "urgency": min(1.0, urgency["urgency_score"] / 100.0),
        "human_impact": min(1.0, max(cas, human_prob)),
        "assistance": min(1.0, max(need_score(needs), 0.5 * cats.get("rescue_volunteering_or_donation_effort", 0.0))),
        "damage": min(1.0, dmg),
        "location": min(1.0, location["location_confidence"]),
    }
    score = 100.0 * p_inf * sum(PRIORITY_V2_WEIGHTS[k] * comp[k] for k in comp)

    reasons = []
    ind = urgency.get("indicators", {})
    firing = ind.get("high", []) + ind.get("medium", [])
    if firing:
        reasons.append("Urgency indicators: " + ", ".join(firing))
    if extraction.get("casualties"):
        reasons.append("Reported casualties: " + ", ".join(f"{c['count']} {c['type']}" for c in extraction["casualties"]))
    elif human_prob >= 0.5:
        reasons.append("Likely about injured or affected people")
    if needs:
        reasons.append("Assistance mentioned: " + ", ".join(needs))
    if dmg_label != "Unknown":
        reasons.append(f"{dmg_label} damage reported")
    if location["location"] != "Unknown":
        reasons.append("Location identified: " + location["location"])
    if p_inf < 0.5:
        reasons.append("Post is probably not informative, which lowers its priority")
    if not reasons:
        reasons.append("No strong urgency or impact signals found")

    return {
        "priority_score": round(score, 1),
        "priority_level": level(score),
        "reason": reasons,
        "components": {"p_informative": round(p_inf, 4), **{k: round(v, 4) for k, v in comp.items()}},
        "damage_severity": dmg_label,
    }
