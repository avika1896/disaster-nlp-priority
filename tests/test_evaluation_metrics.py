"""Metric helpers used by the Step 14 evaluation scripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
from metrics import confusion, prf, set_prf, spearman  # noqa: E402


def test_confusion_and_prf_on_a_tiny_example():
    y_true = ["a", "a", "b", "b", "b"]
    y_pred = ["a", "b", "b", "b", "a"]
    assert confusion(y_true, y_pred, ["a", "b"]).tolist() == [[1, 1], [1, 2]]
    r = prf(y_true, y_pred, ["a", "b"])
    assert r["accuracy"] == 0.6
    assert r["per_class"]["a"]["precision"] == 0.5 and r["per_class"]["a"]["recall"] == 0.5
    assert r["per_class"]["b"]["precision"] == 0.6667 and r["per_class"]["b"]["recall"] == 0.6667


def test_class_never_predicted_gives_zero_not_a_crash():
    r = prf(["a", "b"], ["a", "a"], ["a", "b"])
    assert r["per_class"]["b"]["precision"] == 0.0 and r["per_class"]["b"]["f1"] == 0.0


def test_set_prf_counts_per_post_sets():
    r = set_prf([{"x", "y"}, {"z"}], [{"x"}, {"z", "w"}])
    assert (r["true_positives"], r["false_positives"], r["false_negatives"]) == (2, 1, 1)
    assert r["precision"] == 0.6667 and r["recall"] == 0.6667


def test_spearman_needs_no_scipy_and_handles_constants():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None
