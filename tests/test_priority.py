import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ranking"))
from priority import casualty_score, category_score, need_score, priority_score  # noqa: E402

NONE = {"casualties": [], "needs": []}


def probs(**kw):
    base = {"injured_or_dead_people": 0, "affected_individuals": 0, "rare_humanitarian": 0,
            "infrastructure_and_utility_damage": 0, "rescue_volunteering_or_donation_effort": 0,
            "other_relevant_information": 0, "not_humanitarian": 0}
    base.update(kw)
    return base


def test_non_informative_tweet_scores_near_zero():
    r = priority_score(0.02, probs(not_humanitarian=0.9, other_relevant_information=0.1), NONE)
    assert r["score"] < 2 and r["tier"] == "low"


def test_sure_not_humanitarian_gives_zero_category_score():
    assert category_score(probs(not_humanitarian=1.0)) == 0.0


def test_category_ordering():
    inj = priority_score(1.0, probs(injured_or_dead_people=1.0), NONE)["score"]
    res = priority_score(1.0, probs(rescue_volunteering_or_donation_effort=1.0), NONE)["score"]
    oth = priority_score(1.0, probs(other_relevant_information=1.0), NONE)["score"]
    assert inj > res > oth


def test_more_deaths_never_lowers_score():
    last = -1
    for n in [0, 3, 10, 217, 1000, 5000]:
        ext = {"casualties": [{"type": "deaths", "count": n}] if n else [], "needs": []}
        s = priority_score(0.9, probs(injured_or_dead_people=1.0), ext)["score"]
        assert s >= last
        last = s


def test_casualty_score_is_capped_and_zero_when_absent():
    assert casualty_score([]) == 0.0
    assert casualty_score([{"type": "deaths", "count": 10**6}]) == 1.0


def test_need_score_uses_strongest_tag():
    assert need_score(["donations", "medical"]) == 1.0
    assert need_score([]) == 0.0
    assert need_score(["unknown_tag"]) == 0.0


def test_score_bounds_and_components():
    ext = {"casualties": [{"type": "deaths", "count": 217}], "needs": ["medical"]}
    r = priority_score(1.0, probs(injured_or_dead_people=1.0), ext)
    assert 0 <= r["score"] <= 100
    assert set(r["components"]) == {"p_informative", "category", "casualty", "need"}
    expected = 100 * (0.60 * r["components"]["category"] + 0.25 * r["components"]["casualty"]
                      + 0.15 * r["components"]["need"])
    assert abs(r["score"] - expected) < 0.01


def test_unknown_category_label_raises():
    with pytest.raises(ValueError):
        category_score({"made_up_class": 1.0})


def test_p_informative_is_clamped():
    assert priority_score(1.7, probs(injured_or_dead_people=1.0), NONE)["components"]["p_informative"] == 1.0