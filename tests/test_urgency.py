"""Step 9 - engineered urgency score."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "ranking"))
from urgency import assess_urgency  # noqa: E402


def test_high_indicators_give_high_urgency_and_reasons():
    r = assess_urgency("People trapped on the roof, rescue needed immediately")
    assert r["urgency_level"] == "HIGH" and r["urgency_score"] >= 70
    assert set(r["indicators"]["high"]) == {"trapped", "immediately", "rescue needed"}
    assert any("trapped" in x for x in r["urgency_reason"])


def test_more_indicators_raise_the_score_but_cap_at_100():
    one = assess_urgency("trapped")["urgency_score"]
    many = assess_urgency("trapped injured dying emergency urgent critical immediately")["urgency_score"]
    assert many > one and many <= 100.0


def test_medium_indicators():
    r = assess_urgency("Roads blocked and homes damaged, many stranded")
    assert r["urgency_level"] == "MEDIUM"
    assert {"road blocked", "damage", "stranded"} <= set(r["indicators"]["medium"])


def test_low_indicators_stay_low():
    r = assess_urgency("Weather update for awareness")
    assert r["urgency_level"] == "LOW" and r["urgency_score"] == 10.0


def test_low_words_do_not_reduce_a_high_score():
    r = assess_urgency("Remembering the anniversary but people are trapped now")
    assert r["urgency_level"] == "HIGH"


def test_no_indicators_gives_zero_and_empty_reason():
    r = assess_urgency("Beautiful sunny day at the beach")
    assert r["urgency_score"] == 0.0 and r["urgency_reason"] == []


def test_urls_and_non_text_are_safe():
    assert assess_urgency("https://t.co/trapped")["urgency_score"] == 0.0
    assert assess_urgency(None)["urgency_score"] == 0.0


def test_cannot_escape_and_life_threatening():
    assert "cannot escape" in assess_urgency("we can't escape the water")["indicators"]["high"]
    assert "life-threatening" in assess_urgency("life threatening injuries")["indicators"]["high"]
