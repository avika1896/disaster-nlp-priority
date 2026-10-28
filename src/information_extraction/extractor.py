"""
Step 6 — Rule-based information extraction from disaster tweets.

Extracts, from raw tweet text (text_original, NOT text_clean: capitalisation
matters for place names):
  - casualties        : [{"type": deaths|injured|missing|casualties, "count": int}]
  - needs             : tags such as water, shelter, medical, rescue, ...
  - locations         : gazetteer matches found in the text
  - candidate_locations: Capitalised phrases after a preposition ("in Foo Bar")
                        that are NOT in the gazetteer. Lower confidence; for
                        review and gazetteer growth.
  - event_region      : fallback region from the dataset's disaster_event value

Fully rule-based, no external services, every output is traceable to a regex.

Known limits (state these in the report):
  - Need tags mean "the text mentions X", not "X is being requested". A tweet
    about a rescue team returning home still gets the `rescue` tag.
  - Counts like "thousands" or "dozens" are not extracted.
  - The gazetteer is small; places outside it only show up as candidates.
"""

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gazetteer import EVENT_REGION, PLACES  # noqa: E402

# ---------------------------------------------------------------- normalising
_URL = re.compile(r"https?://\S+")
_RT = re.compile(r"^RT\s+@\w+:\s*")
_MENTION = re.compile(r"@\w+")


def normalize_text(text) -> str:
    """Unescape HTML, drop URLs/RT/mentions and mojibake, keep capitalisation."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = (t.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"'))
    t = _URL.sub(" ", t)
    t = _RT.sub("", t)
    t = _MENTION.sub(" ", t)
    t = t.replace("#", "")
    t = t.encode("ascii", "ignore").decode("ascii")  # drops Â, â€“, emoji junk
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------ casualties
_WORD_NUM = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
     "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}
_NUM = r"(\d+(?:,\d{3})*|" + "|".join(_WORD_NUM) + r")"
_NOT_MORE_DIGITS = r"(?!\d)(?!\s*[xX%])"   # rejects "1000X", "50%", partial numbers

_COUNT_THEN_WORD = re.compile(
    r"\b" + _NUM + r"\+?" + _NOT_MORE_DIGITS +
    r"\s+(?:more\s+)?(?:people\s+|persons\s+|residents\s+|others\s+)?"
    r"(dead|killed|deaths?(?!\s+toll)|died|fatalities|injured|wounded|"
    r"missing|victims|casualties)\b",
    re.I,
)
_TOLL_THEN_COUNT = re.compile(
    r"\btoll\b[^\d.]{0,30}?" + _NUM + _NOT_MORE_DIGITS, re.I
)

_TYPE_OF_WORD = {
    "dead": "deaths", "killed": "deaths", "death": "deaths", "deaths": "deaths",
    "died": "deaths", "fatalities": "deaths",
    "injured": "injured", "wounded": "injured",
    "missing": "missing",
    "victims": "casualties", "casualties": "casualties",
}


def _to_int(s: str) -> int:
    s = s.lower()
    return _WORD_NUM[s] if s in _WORD_NUM else int(s.replace(",", ""))


def extract_casualties(text: str) -> list:
    found, seen = [], set()
    for m in _COUNT_THEN_WORD.finditer(text):
        typ = _TYPE_OF_WORD[m.group(2).lower()]
        key = (typ, _to_int(m.group(1)))
        if key not in seen:
            seen.add(key)
            found.append({"type": typ, "count": key[1]})
    for m in _TOLL_THEN_COUNT.finditer(text):
        key = ("deaths", _to_int(m.group(1)))
        if key not in seen:
            seen.add(key)
            found.append({"type": "deaths", "count": key[1]})
    return found


# ----------------------------------------------------------------------- needs
_NEED_PATTERNS = {
    "water": r"bottled water|drinking water|clean water|safe water|water supply|"
             r"need(?:s|ed)? water",
    "food": r"food|meals?|hungry|hunger|feed(?:ing)?",
    "shelter": r"shelters?|sheltering|homeless|displaced",
    "medical": r"medical|medicines?|hospitals?|doctors?|medics?|first aid",
    "rescue": r"rescue[ds]?|rescuing|trapped|stranded",
    "evacuation": r"evacuat\w+|flee(?:ing)?|fled",
    "power_utilities": r"power outages?|without power|no power|blackouts?|electricity",
    "donations": r"donat\w+|fundrais\w+|relief fund|supplies",
    "explicit_request": r"needs? (?:our |your )?(?:help|support|prayers|donations|"
                        r"water|food)|please help|help needed|urgent(?:ly)?|\bSOS\b",
}
_NEED_RE = {k: re.compile(r"\b(?:" + v + r")\b", re.I) for k, v in _NEED_PATTERNS.items()}


def extract_needs(text: str) -> list:
    return [k for k, rx in _NEED_RE.items() if rx.search(text)]


# ------------------------------------------------------------------- locations
_PLACES_LONGEST_FIRST = sorted(set(PLACES), key=len, reverse=True)
_PLACE_RE = [(p, re.compile(r"(?<![A-Za-z])" + re.escape(p) + r"(?![A-Za-z])", re.I))
             for p in _PLACES_LONGEST_FIRST]

_CANDIDATE = re.compile(
    r"\b(?:[Ii]n|near|across|from|to|at|outside|around)\s+"
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})"
)
_STOP_FIRST = {
    "hurricane", "hurricanes", "storm", "tropical", "category", "harvey", "irma",
    "maria", "jose", "the", "this", "that", "these", "those", "watch", "now",
    "via", "rt", "my", "our", "your", "his", "her", "its", "their", "all",
    "each", "every", "some", "any", "no", "new", "how", "why", "what", "when",
    "who", "here", "there", "today", "tonight", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "i", "we", "you", "it",
}


def extract_locations(text: str):
    """Return (gazetteer_locations, candidate_locations)."""
    taken, locs = [], []
    for place, rx in _PLACE_RE:                      # longest first
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue                              # overlaps a longer match
            taken.append((m.start(), m.end()))
            if place not in locs:
                locs.append(place)
    known = {p.lower() for p in locs} | {p.lower() for p in PLACES}
    cands = []
    for m in _CANDIDATE.finditer(text):
        phrase = m.group(1).strip()
        first = phrase.split()[0].lower()
        if first in _STOP_FIRST or phrase.lower() in known or phrase in cands:
            continue
        cands.append(phrase)
    return locs, cands


# ------------------------------------------------------------------- pipeline
def extract(text_original, disaster_event=None) -> dict:
    text = normalize_text(text_original)
    locs, cands = extract_locations(text)
    cas = extract_casualties(text)
    deaths = [c["count"] for c in cas if c["type"] == "deaths"]
    return {
        "casualties": cas,
        "max_deaths": max(deaths) if deaths else None,
        "needs": extract_needs(text),
        "locations": locs,
        "candidate_locations": cands,
        "event_region": EVENT_REGION.get(disaster_event, []),
    }