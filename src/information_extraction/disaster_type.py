"""
Step 8 - Disaster type.

Method: transparent keyword rules, NOT a trained classifier. Reason: the type
can only be derived from the event name (hurricane_*, *_wildfires, *_earthquake,
srilanka_floods), and the only flood event is held out of training, so a trained
text classifier would have no flood examples and its score would be misleading.

Each type has
  "name" words    (weight 3): the disaster's own name, e.g. "earthquake"
  "support" words (weight 1 each, at most 2 counted): weaker evidence
A type needs a score of at least MIN_SCORE. If two types tie, the order in
TIE_ORDER decides: hazards that are often CAUSED by another hazard (flood,
landslide) come last, because "Hurricane X flooding" is a hurricane tweet.
No keyword found -> the event name is used only if the caller passes it
(source = "event_name"), otherwise "Unknown".

The rules were written from general knowledge, not tuned on the labels. The
reference labels used in evaluate_disaster_type.py come from the dataset's
event name, not from per-tweet human annotation.
"""

import re

NAME = 3
SUPPORT = 1
MAX_SUPPORT = 2
MIN_SCORE = 2

TIE_ORDER = ["Hurricane", "Earthquake", "Tornado", "Tsunami", "Volcano",
             "Wildfire", "Landslide", "Flood"]


def _rx(p):
    return re.compile(p, re.I)


RULES = {
    "Flood": {
        "name": [_rx(r"\bflood\w*"), _rx(r"\bflash flood")],
        "support": [_rx(r"\binundat\w+"), _rx(r"\bsubmerged\b"), _rx(r"\brising water"),
                    _rx(r"\bwater levels?\b"), _rx(r"\bdam (?:burst|break|collapse)")],
    },
    "Earthquake": {
        "name": [_rx(r"\bearthquakes?\b"), _rx(r"\bquakes?\b"), _rx(r"\baftershocks?\b"),
                 _rx(r"\bseismic\b"), _rx(r"\btremors?\b")],
        "support": [_rx(r"\bmagnitude\b"), _rx(r"\brichter\b"), _rx(r"\brubble\b"),
                    _rx(r"\bepicent\w+"), _rx(r"\bcollapsed buildings?\b")],
    },
    "Hurricane": {
        "name": [_rx(r"\bhurricanes?\b"), _rx(r"\bcyclones?\b"), _rx(r"\btyphoons?\b"),
                 _rx(r"\btropical (?:storm|depression)\b"), _rx(r"\bstorm surge\b")],
        "support": [_rx(r"\bcategory [1-5]\b"), _rx(r"\blandfall\b"), _rx(r"\bstorm\b"),
                    _rx(r"\bstrong winds?\b"), _rx(r"\bevacuation orders?\b")],
    },
    "Wildfire": {
        "name": [_rx(r"\bwild ?fires?\b"), _rx(r"\bbush ?fires?\b"), _rx(r"\bforest fires?\b")],
        "support": [_rx(r"\bfires?\b"), _rx(r"\bburning\b"), _rx(r"\bflames?\b"),
                    _rx(r"\bsmoke\b"), _rx(r"\bblaze\b"), _rx(r"\bash\b")],
    },
    "Tornado": {
        "name": [_rx(r"\btornad(?:o|oes|os)\b"), _rx(r"\btwisters?\b")],
        "support": [_rx(r"\bfunnel cloud\b"), _rx(r"\btornado warning\b")],
    },
    "Tsunami": {
        "name": [_rx(r"\btsunamis?\b")],
        "support": [_rx(r"\btidal wave\b")],
    },
    "Landslide": {
        "name": [_rx(r"\blandslides?\b"), _rx(r"\bmudslides?\b")],
        "support": [_rx(r"\bmud flow\b")],
    },
    "Volcano": {
        "name": [_rx(r"\bvolcan\w+"), _rx(r"\beruption\b")],
        "support": [_rx(r"\blava\b")],
    },
}

# Derived from the event names that appear in the dataset's disaster_event column.
EVENT_TYPE = {
    "hurricane_harvey": "Hurricane",
    "hurricane_irma": "Hurricane",
    "hurricane_maria": "Hurricane",
    "california_wildfires": "Wildfire",
    "mexico_earthquake": "Earthquake",
    "iraq_iran_earthquake": "Earthquake",
    "srilanka_floods": "Flood",
}


def _prep(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"https?://\S+", " ", text).replace("#", " ")
    return re.sub(r"\s+", " ", t)


def detect_disaster_type(text, disaster_event=None) -> dict:
    t = _prep(text)
    scores, matched = {}, {}
    for dtype, rules in RULES.items():
        names = [rx.pattern for rx in rules["name"] if rx.search(t)]
        supports = [rx.pattern for rx in rules["support"] if rx.search(t)]
        score = (NAME if names else 0) + SUPPORT * min(MAX_SUPPORT, len(supports))
        if score:
            scores[dtype] = score
            matched[dtype] = names + supports
    best = [d for d, s in scores.items() if s >= MIN_SCORE]
    if best:
        top = max(scores[d] for d in best)
        tied = [d for d in best if scores[d] == top]
        winner = sorted(tied, key=TIE_ORDER.index)[0]
        return {"disaster_type": winner, "source": "text_keywords",
                "score": scores[winner], "matched": matched[winner]}
    if disaster_event in EVENT_TYPE:
        return {"disaster_type": EVENT_TYPE[disaster_event], "source": "event_name",
                "score": 0, "matched": []}
    return {"disaster_type": "Unknown", "source": "none", "score": 0, "matched": []}
