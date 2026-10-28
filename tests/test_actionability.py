"""Step 6 - actionability mapping (derived from humanitarian categories)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "information_extraction"))
from actionability import (  # noqa: E402
    ACTIONABLE_CATEGORIES, KNOWN_CATEGORIES, NOT_ACTIONABLE_CATEGORIES, actionability, actionable_from_label,
)


def test_sets_do_not_overlap():
    assert not (ACTIONABLE_CATEGORIES & NOT_ACTIONABLE_CATEGORIES)


def test_labels_from_dataset_categories():
    assert actionable_from_label("injured_or_dead_people") is True
    assert actionable_from_label("vehicle_damage") is True          # raw dataset name
    assert actionable_from_label("rare_humanitarian") is True       # merged name
    assert actionable_from_label("not_humanitarian") is False
    assert actionable_from_label("other_relevant_information") is False
    assert actionable_from_label("something_new") is None


def test_probability_mass_decides():
    yes = actionability({"injured_or_dead_people": 0.6, "not_humanitarian": 0.4})
    assert yes["is_actionable"] is True and yes["actionable_probability"] == 0.6
    no = actionability({"injured_or_dead_people": 0.2, "other_relevant_information": 0.8})
    assert no["is_actionable"] is False


def test_unknown_category_is_an_error_not_a_guess():
    try:
        actionability({"made_up": 1.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_all_seven_model_classes_are_known():
    seven = {"affected_individuals", "infrastructure_and_utility_damage", "injured_or_dead_people",
             "not_humanitarian", "other_relevant_information", "rare_humanitarian",
             "rescue_volunteering_or_donation_effort"}
    assert seven <= KNOWN_CATEGORIES
