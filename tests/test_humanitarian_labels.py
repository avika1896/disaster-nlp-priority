import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "models"))
from humanitarian_labels import MERGED_NAME, merge_label  # noqa: E402


def test_rare_classes_are_merged():
    assert merge_label("vehicle_damage") == MERGED_NAME
    assert merge_label("missing_or_found_people") == MERGED_NAME


def test_other_classes_unchanged():
    for lab in ["injured_or_dead_people", "not_humanitarian", "affected_individuals",
                "infrastructure_and_utility_damage", "other_relevant_information",
                "rescue_volunteering_or_donation_effort"]:
        assert merge_label(lab) == lab