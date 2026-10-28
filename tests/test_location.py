"""Step 7 - single best location, or Unknown."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "information_extraction"))
from location import extract_location  # noqa: E402


def test_gazetteer_place_found():
    r = extract_location("People are trapped after the flood in Houston.")
    assert r["location"] == "Houston" and r["location_source"] == "gazetteer"
    assert r["location_confidence"] == 0.9


def test_most_specific_place_wins():
    assert extract_location("Fires near Florida Keys and Florida")["location"] == "Florida Keys"


def test_candidate_is_lower_confidence():
    r = extract_location("Please help us in Foo Bar")
    assert r["location"] == "Foo Bar" and r["location_source"] == "candidate"
    assert r["location_confidence"] < 0.9


def test_no_location_means_unknown_not_a_guess():
    r = extract_location("Thoughts and prayers to everyone")
    assert r["location"] == "Unknown" and r["location_confidence"] == 0.0


def test_event_region_is_a_labelled_low_confidence_fallback():
    r = extract_location("Thoughts with everyone", "hurricane_maria")
    assert r["location"] == "Puerto Rico" and r["location_source"] == "event_region"
    assert r["location_confidence"] == 0.3


def test_text_location_beats_event_region():
    assert extract_location("Flooding in Houston", "hurricane_maria")["location"] == "Houston"
