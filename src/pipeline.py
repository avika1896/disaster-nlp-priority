"""
Step 11 - The complete inference pipeline.

    from src.pipeline import analyze_many, analyze
    result = analyze(predictor, "People trapped in Houston, rescue needed")

`predictor` is any object with
    predict(clean_texts) -> (list_of_p_informative, list_of_category_prob_dicts)
(backend.scorer.TransformerPredictor is the real one; tests use a fake). Pass
predictor=None to load the real one.

Per post:
  raw text -> clean_text -> two DistilBERT models (informative, humanitarian category)
  raw text -> rule-based: extraction, location, disaster type, urgency
  all of the above -> actionability and the explainable priority score

Fields that cannot be found are "Unknown" (or False). Nothing is invented.
How the descriptive fields are derived (all simple rules, not learned):
  incident             : casualty counts, then trapped/stranded words, then evacuation,
                         power outage, then the predicted category
  assistance_required  : the strongest need tag found by the extractor
  affected_people      : casualties found, or predicted category is injured/affected
  damage / severity    : predicted category or damage words (priority_score.py)
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
for p in (ROOT, SRC / "ranking", SRC / "information_extraction"):
    sys.path.insert(0, str(p))

from src.preprocessing.text_cleaner import clean_text  # noqa: E402
from actionability import actionability  # noqa: E402
from disaster_type import detect_disaster_type  # noqa: E402
from extractor import extract  # noqa: E402
from location import extract_location  # noqa: E402
from priority_score import priority_score_v2  # noqa: E402
from urgency import assess_urgency, prep_text  # noqa: E402
from weights import NEED_WEIGHT  # noqa: E402

INFORMATIVE_THRESHOLD = 0.5

NEED_LABEL = {
    "rescue": "Rescue", "medical": "Medical help", "water": "Water", "food": "Food",
    "shelter": "Shelter", "evacuation": "Evacuation", "power_utilities": "Power or utilities",
    "donations": "Donations", "explicit_request": "Help requested",
}


def _assistance(needs: list):
    ordered = sorted(needs, key=lambda n: -NEED_WEIGHT.get(n, 0.0))
    labels = [NEED_LABEL.get(n, n) for n in ordered]
    return (labels[0] if labels else "Unknown"), labels


def _incident(text: str, ext: dict, category) -> str:
    kinds = {c["type"] for c in ext["casualties"]}
    t = prep_text(text)
    if "deaths" in kinds:
        return "People killed"
    if "injured" in kinds:
        return "People injured"
    if "missing" in kinds:
        return "People missing"
    if re.search(r"\b(trapped|stranded)\b", t):
        return "People trapped or stranded"
    if "evacuation" in ext["needs"]:
        return "Evacuation"
    if "power_utilities" in ext["needs"]:
        return "Power outage"
    if category == "infrastructure_and_utility_damage":
        return "Infrastructure damage"
    if category == "rescue_volunteering_or_donation_effort":
        return "Rescue, relief or donation effort"
    return "Unknown"


def _empty(raw, clean, ext, event):
    dtype = detect_disaster_type(raw, event)
    loc = extract_location(raw, event)
    return {
        "text": raw, "is_informative": False, "is_actionable": False,
        "disaster_type": dtype["disaster_type"], "location": loc["location"],
        "incident": "Unknown", "assistance_required": "Unknown", "affected_people": False,
        "damage": "Unknown", "damage_severity": "Unknown", "urgency_score": 0.0,
        "priority_score": 0.0, "priority_level": "LOW",
        "priority_reasons": ["No usable text after cleaning"],
        "informative_probability": 0.0, "predicted_category": None, "actionable_probability": 0.0,
        "urgency_reason": [], "location_confidence": loc["location_confidence"],
        "location_source": loc["location_source"], "assistance_all": [],
        "components": {}, "extraction": ext,
        "warning": "No usable text after cleaning; scored 0.",
    }


def analyze_many(predictor, posts: List[Tuple[str, Optional[str]]]) -> List[dict]:
    """posts: list of (raw_text, disaster_event_or_None). One result dict per post."""
    if predictor is None:
        from backend.scorer import TransformerPredictor
        predictor = TransformerPredictor()

    cleaned = [clean_text(t) for t, _ in posts]
    usable = [i for i, c in enumerate(cleaned) if c.strip()]
    p_by_i, cat_by_i = {}, {}
    if usable:
        p_list, c_list = predictor.predict([cleaned[i] for i in usable])
        for i, p, c in zip(usable, p_list, c_list):
            p_by_i[i], cat_by_i[i] = float(p), c

    results = []
    for i, (raw, event) in enumerate(posts):
        ext = extract(raw, event)
        if i not in p_by_i:
            results.append(_empty(raw, cleaned[i], ext, event))
            continue
        probs = {k: round(float(v), 4) for k, v in cat_by_i[i].items()}
        category = max(probs, key=probs.get)
        act = actionability(probs)
        urg = assess_urgency(raw)
        loc = extract_location(raw, event)
        dtype = detect_disaster_type(raw, event)
        pr = priority_score_v2(p_by_i[i], probs, ext, urg, loc, raw)
        assist, assist_all = _assistance(ext["needs"])
        sev = pr["damage_severity"]
        damage = ("Infrastructure damage" if category == "infrastructure_and_utility_damage"
                  else "Damage reported" if sev != "Unknown" else "Unknown")
        results.append({
            "text": raw,
            "is_informative": p_by_i[i] >= INFORMATIVE_THRESHOLD,
            "is_actionable": act["is_actionable"],
            "disaster_type": dtype["disaster_type"],
            "location": loc["location"],
            "incident": _incident(raw, ext, category),
            "assistance_required": assist,
            "affected_people": bool(ext["casualties"]) or category in ("injured_or_dead_people", "affected_individuals"),
            "damage": damage,
            "damage_severity": sev,
            "urgency_score": urg["urgency_score"],
            "priority_score": pr["priority_score"],
            "priority_level": pr["priority_level"],
            "priority_reasons": pr["reason"],
            # extra detail for the dashboard and the report
            "informative_probability": round(p_by_i[i], 4),
            "predicted_category": category,
            "actionable_probability": act["actionable_probability"],
            "urgency_reason": urg["urgency_reason"],
            "location_confidence": loc["location_confidence"],
            "location_source": loc["location_source"],
            "assistance_all": assist_all,
            "components": pr["components"],
            "extraction": ext,
            "warning": None,
        })
    return results


def analyze(predictor, text: str, disaster_event: Optional[str] = None) -> dict:
    return analyze_many(predictor, [(text, disaster_event)])[0]
