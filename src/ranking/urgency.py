"""
Step 9 - Urgency engineering.

CrisisMMD has NO urgency label, so urgency here is an ENGINEERED, rule-based
feature. It is not ground truth and must not be presented as such.

How it works: look for indicator words/phrases (lists below, taken from the
project brief), then
    level = HIGH if any high indicator, else MEDIUM if any medium indicator,
            else LOW if only low indicators, else none
    score = base score of that level (weights.URGENCY_BASE)
            + URGENCY_EXTRA_PER_INDICATOR for each further high/medium indicator
            (capped at 100). Low indicators (awareness, forecast ...) never add.
Every score comes with `urgency_reason`, the list of indicators that fired.

Known limits: it matches words, not meaning. "Not urgent" still matches
"urgent"; "emergency shelter" matches "emergency". Say this in the report.
"""

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import (  # noqa: E402
    URGENCY_BASE, URGENCY_EXTRA_PER_INDICATOR, URGENCY_HIGH_CUTOFF, URGENCY_MEDIUM_CUTOFF,
)

_URL = re.compile(r"https?://\S+")


def prep_text(text) -> str:
    """Lower-case text with URLs removed; hashtag words stay as words."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text).replace("’", "'").replace("#", " ")
    t = _URL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def _rx(pattern):
    return re.compile(pattern, re.I)


HIGH_INDICATORS = [
    ("trapped", _rx(r"\btrapped\b")),
    ("injured", _rx(r"\binjur(?:ed|ies|y)\b")),
    ("dying", _rx(r"\bdying\b|\bwill die\b")),
    ("emergency", _rx(r"\bemergenc(?:y|ies)\b")),
    ("immediately", _rx(r"\bimmediate(?:ly)?\b")),
    ("urgent", _rx(r"\burgent(?:ly)?\b")),
    ("critical", _rx(r"\bcritical(?:ly)?\b")),
    ("life-threatening", _rx(r"life[- ]threatening")),
    ("rescue needed", _rx(r"rescue (?:is |are |team |teams )?needed|need(?:s|ed)?(?: to be)? rescu\w+|"
                          r"(?:please )?send rescue")),
    ("cannot escape", _rx(r"can(?:no|')t(?: \w+)? escape|unable to escape|no way out")),
]
MEDIUM_INDICATORS = [
    ("damage", _rx(r"\bdamag\w+")),
    ("stranded", _rx(r"\bstranded\b")),
    ("homeless", _rx(r"\bhomeless\b")),
    ("evacuation", _rx(r"\bevacuat\w+")),
    ("need supplies", _rx(r"need(?:s|ed)? (?:\w+ )?supplies|supplies (?:are )?needed|"
                          r"(?:lack|shortage) of (?:\w+ )?supplies")),
    ("road blocked", _rx(r"\broads? (?:\w+ )?(?:blocked|closed|impassable|cut off)|\bblocked roads?")),
]
LOW_INDICATORS = [
    ("awareness", _rx(r"\bawareness\b")),
    ("weather update", _rx(r"weather update|weather forecast|\bforecast\b")),
    ("general information", _rx(r"general information|for your information|\bfyi\b|\bpsa\b")),
    ("historical or general discussion", _rx(r"\banniversary\b|\bhistor(?:y|ical)\b|"
                                              r"\bremember(?:ing)?\b|\bremembrance\b")),
]
_LEVELS = [("high", HIGH_INDICATORS), ("medium", MEDIUM_INDICATORS), ("low", LOW_INDICATORS)]


def assess_urgency(text) -> dict:
    t = prep_text(text)
    hits = {lvl: [name for name, rx in pats if rx.search(t)] for lvl, pats in _LEVELS}
    if hits["high"]:
        level = "high"
    elif hits["medium"]:
        level = "medium"
    elif hits["low"]:
        level = "low"
    else:
        level = "none"

    score = URGENCY_BASE[level]
    if level in ("high", "medium"):
        extra = len(hits["high"]) + len(hits["medium"]) - 1
        score = min(100.0, score + URGENCY_EXTRA_PER_INDICATOR * extra)

    if score >= URGENCY_HIGH_CUTOFF:
        label = "HIGH"
    elif score >= URGENCY_MEDIUM_CUTOFF:
        label = "MEDIUM"
    else:
        label = "LOW"

    reasons = ([f"High urgency indicator: {n}" for n in hits["high"]]
               + [f"Medium urgency indicator: {n}" for n in hits["medium"]])
    if level == "low":
        reasons = [f"Low urgency indicator: {n}" for n in hits["low"]]
    return {"urgency_score": round(score, 1), "urgency_level": label,
            "urgency_reason": reasons, "indicators": hits}
