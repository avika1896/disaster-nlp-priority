"""Step 11 - the full pipeline, with a fake predictor (no PyTorch needed)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import analyze, analyze_many  # noqa: E402

CATS = ["affected_individuals", "infrastructure_and_utility_damage", "injured_or_dead_people",
        "not_humanitarian", "other_relevant_information", "rare_humanitarian",
        "rescue_volunteering_or_donation_effort"]
SPEC_KEYS = {"text", "is_informative", "is_actionable", "disaster_type", "location", "incident",
             "assistance_required", "affected_people", "damage", "damage_severity", "urgency_score",
             "priority_score", "priority_level", "priority_reasons"}


class FakePredictor:
    def predict(self, clean_texts):
        p, c = [], []
        for t in clean_texts:
            zero = {k: 0.0 for k in CATS}
            if "dead" in t or "killed" in t or "trapped" in t:
                p.append(0.95)
                c.append({**zero, "injured_or_dead_people": 0.9, "other_relevant_information": 0.1})
            else:
                p.append(0.10)
                c.append({**zero, "not_humanitarian": 0.8, "other_relevant_information": 0.2})
        return p, c


def test_serious_post_has_every_field_filled_from_the_text():
    r = analyze(FakePredictor(), "People trapped after the flood in Houston, 12 dead, rescue needed immediately")
    assert SPEC_KEYS <= set(r)
    assert r["is_informative"] is True and r["is_actionable"] is True
    assert r["disaster_type"] == "Flood" and r["location"] == "Houston"
    assert r["incident"] == "People killed" and r["assistance_required"] == "Rescue"
    assert r["affected_people"] is True
    assert r["priority_level"] == "HIGH" and r["priority_score"] > 40
    assert r["urgency_score"] >= 70 and r["priority_reasons"]


def test_unknown_is_returned_instead_of_inventing():
    r = analyze(FakePredictor(), "Beautiful sunny day")
    assert r["is_informative"] is False and r["is_actionable"] is False
    assert r["disaster_type"] == "Unknown" and r["location"] == "Unknown"
    assert r["incident"] == "Unknown" and r["assistance_required"] == "Unknown"
    assert r["damage"] == "Unknown" and r["priority_level"] == "LOW"


def test_event_name_fills_type_and_region_only_when_given():
    r = analyze(FakePredictor(), "Thoughts with everyone", "hurricane_maria")
    assert r["disaster_type"] == "Hurricane" and r["location"] == "Puerto Rico"
    assert r["location_source"] == "event_region"


def test_text_that_is_only_a_url_scores_zero_with_a_warning():
    r = analyze(FakePredictor(), "https://t.co/abc123")
    assert r["priority_score"] == 0.0 and r["warning"] and r["predicted_category"] is None
    assert SPEC_KEYS <= set(r)


def test_batch_keeps_order_of_inputs():
    posts = [("nice day", None), ("20 dead after flood", None), ("", None)]
    out = analyze_many(FakePredictor(), posts)
    assert [o["text"] for o in out] == ["nice day", "20 dead after flood", ""]
    assert out[1]["priority_score"] > out[0]["priority_score"]
