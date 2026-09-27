import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "baseline"


def test_model_artifacts_exist():
    assert (MODEL_DIR / "tfidf_vectorizer.joblib").exists()
    assert (MODEL_DIR / "baseline_logistic_regression.joblib").exists()
    assert (MODEL_DIR / "baseline_linear_svm.joblib").exists()


def test_report_has_required_metrics():
    with open(MODEL_DIR / "step4_baseline_report.json") as f:
        report = json.load(f)
    assert report["task"] == "informative_label"
    assert "test_metrics" in report
    assert "accuracy" in report["test_metrics"]
    assert "macro_f1" in report["test_metrics"]
    # Sanity bound: must beat random/majority-class guessing on this
    # ~73/27 imbalanced task (majority-class accuracy would be ~0.73 but
    # macro-F1 near 0.5), without hardcoding an exact expected number
    # (that would break as soon as someone tweaks the model).
    assert report["test_metrics"]["macro_f1"] > 0.6


def test_model_predicts_known_classes():
    vec = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "baseline_logistic_regression.joblib")
    preds = model.predict(vec.transform(["urgent rescue needed now"]))
    assert preds[0] in {"informative", "not_informative"}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])