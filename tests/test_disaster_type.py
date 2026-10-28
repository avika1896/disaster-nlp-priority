"""Step 8 - keyword disaster type."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "information_extraction"))
from disaster_type import EVENT_TYPE, detect_disaster_type  # noqa: E402


def t(text, event=None):
    return detect_disaster_type(text, event)["disaster_type"]


def test_basic_types():
    assert t("Massive earthquake hits Mexico City") == "Earthquake"
    assert t("Flash flood warning, streets underwater") == "Flood"
    assert t("Hurricane makes landfall tonight") == "Hurricane"
    assert t("Wildfire spreading near Santa Rosa") == "Wildfire"
    assert t("Tornado touched down near town") == "Tornado"


def test_flooding_caused_by_a_hurricane_is_a_hurricane():
    assert t("Hurricane flooding everywhere in Houston") == "Hurricane"


def test_weak_single_word_is_not_enough():
    assert t("There is smoke in the kitchen") == "Unknown"


def test_two_weak_signals_can_be_enough():
    assert t("Fire and smoke everywhere, flames visible") == "Wildfire"


def test_unknown_when_no_keyword_and_no_event():
    assert t("Thoughts and prayers to all") == "Unknown"


def test_event_name_fallback_only_when_given_and_labelled_as_such():
    r = detect_disaster_type("Thoughts and prayers to all", "srilanka_floods")
    assert r["disaster_type"] == "Flood" and r["source"] == "event_name"
    assert detect_disaster_type("Thoughts and prayers to all")["source"] == "none"


def test_text_beats_event_name():
    assert t("Earthquake felt downtown", "hurricane_maria") == "Earthquake"


def test_urls_and_non_text_are_safe():
    assert t("https://t.co/earthquake") == "Unknown"
    assert t(None) == "Unknown"


def test_every_dataset_event_has_a_type():
    assert set(EVENT_TYPE.values()) == {"Hurricane", "Wildfire", "Earthquake", "Flood"}
