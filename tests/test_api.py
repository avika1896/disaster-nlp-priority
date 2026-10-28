"""API tests. A fake predictor stands in for the transformers, so these run fast
and need no trained models. They exercise cleaning, extraction, scoring,
ranking, validation and error handling end to end through the HTTP layer."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import create_app  # noqa: E402

CATS = ["affected_individuals", "infrastructure_and_utility_damage", "injured_or_dead_people",
        "not_humanitarian", "other_relevant_information", "rare_humanitarian",
        "rescue_volunteering_or_donation_effort"]


class FakePredictor:
    def predict(self, clean_texts):
        p, c = [], []
        for t in clean_texts:
            zero = {k: 0.0 for k in CATS}
            if "dead" in t or "killed" in t:
                p.append(0.95)
                c.append({**zero, "injured_or_dead_people": 0.9, "other_relevant_information": 0.1})
            else:
                p.append(0.10)
                c.append({**zero, "not_humanitarian": 0.8, "other_relevant_information": 0.2})
        return p, c


client = TestClient(create_app(FakePredictor()))


def test_health_reports_models_loaded():
    assert client.get("/health").json() == {"status": "ok", "models_loaded": True}


def test_score_high_priority_tweet_end_to_end():
    text = "At least 217 dead after powerful earthquake hits central Mexico - https://t.co/E53Quc2lz2"
    r = client.post("/score", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["priority_tier"] == "high" and body["priority_score"] > 60
    assert body["extraction"]["max_deaths"] == 217
    assert "Mexico" in body["extraction"]["locations"]
    assert "https" not in body["clean_text"] and body["clean_text"] == body["clean_text"].lower()
    assert body["predicted_category"] == "injured_or_dead_people"
    assert body["warning"] is None


def test_score_low_priority_tweet():
    r = client.post("/score", json={"text": "Beautiful sunny day at the beach"}).json()
    assert r["priority_tier"] == "low" and r["priority_score"] < 10


def test_event_region_fallback_used_when_event_given():
    r = client.post("/score", json={"text": "Thoughts with everyone", "disaster_event": "hurricane_maria"}).json()
    assert r["extraction"]["event_region"] == ["Puerto Rico"]


def test_text_that_is_only_a_url_scores_zero_with_warning():
    r = client.post("/score", json={"text": "https://t.co/abc123"}).json()
    assert r["priority_score"] == 0.0 and r["warning"] and r["predicted_category"] is None


def test_batch_is_sorted_and_keeps_input_index():
    tweets = [{"text": "nice day"}, {"text": "50 killed in quake"}, {"text": "20 dead after flood"}]
    body = client.post("/score/batch", json={"tweets": tweets}).json()
    assert body["count"] == 3
    assert [x["input_index"] for x in body["results"]] == [1, 2, 0]
    scores = [x["priority_score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_validation_errors():
    assert client.post("/score", json={"text": "   "}).status_code == 422
    assert client.post("/score", json={}).status_code == 422
    assert client.post("/score/batch", json={"tweets": []}).status_code == 422
    too_many = {"tweets": [{"text": "x"}] * 101}
    assert client.post("/score/batch", json=too_many).status_code == 422


def test_config_exposes_weights_that_sum_to_one():
    cfg = client.get("/config").json()
    assert abs(sum(cfg["component_weights"].values()) - 1.0) < 1e-9
    assert set(cfg["tier_thresholds"]) == {"high", "medium"}


def test_503_when_models_not_loaded():
    no_models = TestClient(create_app(None))   # lifespan not run, so no predictor
    assert no_models.post("/score", json={"text": "hello"}).status_code == 503
    assert no_models.get("/health").json()["models_loaded"] is False


# ---- Step 12: /analyze ----
def test_analyze_returns_full_structured_result():
    text = "People trapped after the flood in Houston, 12 dead, rescue needed immediately"
    r = client.post("/analyze", json={"text": text})
    assert r.status_code == 200
    b = r.json()
    assert b["is_informative"] is True and b["is_actionable"] is True
    assert b["disaster_type"] == "Flood" and b["location"] == "Houston"
    assert b["incident"] == "People killed" and b["assistance_required"] == "Rescue"
    assert b["priority_level"] == "HIGH" and b["priority_score"] > 40
    assert b["urgency_score"] >= 70 and b["priority_reasons"]


def test_analyze_unknown_instead_of_inventing():
    b = client.post("/analyze", json={"text": "Beautiful sunny day"}).json()
    assert b["disaster_type"] == "Unknown" and b["location"] == "Unknown"
    assert b["priority_level"] == "LOW" and b["is_informative"] is False


def test_analyze_event_fallback():
    b = client.post("/analyze", json={"text": "Thoughts with everyone", "disaster_event": "hurricane_maria"}).json()
    assert b["disaster_type"] == "Hurricane" and b["location"] == "Puerto Rico"


def test_analyze_url_only_text_gets_a_warning():
    b = client.post("/analyze", json={"text": "https://t.co/abc123"}).json()
    assert b["priority_score"] == 0.0 and b["warning"]


def test_analyze_batch_is_sorted_and_keeps_input_index():
    tweets = [{"text": "nice day"}, {"text": "50 killed in quake"}, {"text": "20 dead after flood"}]
    body = client.post("/analyze/batch", json={"tweets": tweets}).json()
    assert body["count"] == 3
    scores = [x["priority_score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert sorted(x["input_index"] for x in body["results"]) == [0, 1, 2]


def test_analyze_validation_and_503():
    assert client.post("/analyze", json={"text": "   "}).status_code == 422
    assert client.post("/analyze", json={}).status_code == 422
    assert TestClient(create_app(None)).post("/analyze", json={"text": "hello"}).status_code == 503
