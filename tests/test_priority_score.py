"""Step 10 - explainable multi-factor priority score."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "ranking"), str(ROOT / "src" / "information_extraction")]
import weights  # noqa: E402
from extractor import extract  # noqa: E402
from location import extract_location  # noqa: E402
from priority_score import damage_severity, level, priority_score_v2  # noqa: E402
from urgency import assess_urgency  # noqa: E402

DEAD = {"injured_or_dead_people": 0.8, "infrastructure_and_utility_damage": 0.1, "other_relevant_information": 0.1}
NOT_HUM = {"not_humanitarian": 0.9, "other_relevant_information": 0.1}


def score(text, p_inf, cats):
    return priority_score_v2(p_inf, cats, extract(text), assess_urgency(text), extract_location(text), text)


def test_weights_sum_to_one():
    assert abs(sum(weights.PRIORITY_V2_WEIGHTS.values()) - 1.0) < 1e-9
    assert weights.PRIORITY_V2_MEDIUM < weights.PRIORITY_V2_HIGH


def test_serious_post_is_high_and_explained():
    r = score("Building collapsed in Mexico City, 12 dead, people trapped, rescue needed immediately", 0.95, DEAD)
    assert r["priority_level"] == "HIGH" and r["priority_score"] > 60
    assert r["damage_severity"] == "Severe"
    text = " ".join(r["reason"])
    assert "trapped" in text and "12 deaths" in text and "Mexico City" in text


def test_non_informative_post_sinks_to_low_with_a_reason():
    r = score("Beautiful sunny day", 0.1, NOT_HUM)
    assert r["priority_level"] == "LOW" and r["priority_score"] < 5
    assert any("not informative" in x for x in r["reason"])


def test_informativeness_multiplies_the_score():
    text = "People trapped in Houston, rescue needed"
    assert score(text, 0.9, DEAD)["priority_score"] > 2 * score(text, 0.3, DEAD)["priority_score"] - 1


def test_score_bounds_and_components_present():
    r = score("urgent trapped dying 5000 dead destroyed Houston water food medical", 1.0, DEAD)
    assert 0 <= r["priority_score"] <= 100
    assert set(r["components"]) == {"p_informative", "urgency", "human_impact", "assistance", "damage", "location"}
    assert all(0 <= v <= 1 for v in r["components"].values())


def test_damage_severity_labels():
    assert damage_severity("Homes destroyed", {})[0] == "Severe"
    assert damage_severity("Roof damaged", {})[0] == "Moderate"
    assert damage_severity("Nice day", {})[0] == "Unknown"
    assert damage_severity("Nice day", {"infrastructure_and_utility_damage": 0.9})[0] == "Moderate"


def test_level_uses_configured_cutoffs():
    assert level(weights.PRIORITY_V2_HIGH) == "HIGH"
    assert level(weights.PRIORITY_V2_MEDIUM) == "MEDIUM"
    assert level(0.0) == "LOW"


def test_no_signals_still_gives_a_reason():
    r = score("hello there", 0.9, {"other_relevant_information": 1.0})
    assert r["reason"] == ["No strong urgency or impact signals found"]
