"""
Step 7 - Location extraction.

What this does: picks ONE location for a tweet from the rule-based extractor
(gazetteer match first, then capitalised "in Place" candidates). If nothing is
found it returns "Unknown". It never guesses.

IMPORTANT FOR THE REPORT: the project brief asked to use IDRISI to
investigate/train a location-mention model. IDRISI is NOT in data/raw and has
not been used. The gazetteer in gazetteer.py is small and hand-seeded from the
CrisisMMD events. So this component is rule-based, and its accuracy is unknown
until the sample in data/processed/step6_extraction_sample_for_labelling.csv
is labelled by hand and scored with evaluate_extraction.py.

location_source / location_confidence (fixed rule-of-thumb values, not
probabilities):
  gazetteer        0.9   place found in the gazetteer
  candidate        0.5   capitalised phrase after a preposition, not in gazetteer
  event_region     0.3   only if the caller gives disaster_event (dataset metadata)
  none             0.0   -> "Unknown"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import extract_locations, normalize_text  # noqa: E402
from gazetteer import EVENT_REGION  # noqa: E402

CONFIDENCE = {"gazetteer": 0.9, "candidate": 0.5, "event_region": 0.3, "none": 0.0}


def extract_location(text_original, disaster_event=None) -> dict:
    text = normalize_text(text_original)
    places, cands = extract_locations(text)
    if places:
        best, source = max(places, key=len), "gazetteer"     # most specific = longest name
    elif cands:
        best, source = cands[0], "candidate"
    elif disaster_event in EVENT_REGION and EVENT_REGION[disaster_event]:
        best, source = ", ".join(EVENT_REGION[disaster_event]), "event_region"
    else:
        best, source = "Unknown", "none"
    return {"location": best, "location_confidence": CONFIDENCE[source],
            "location_source": source, "all_locations": places, "candidate_locations": cands}
