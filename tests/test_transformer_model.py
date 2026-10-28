"""Fast tests for Step 5 helpers. No model download or training required."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "models"))
from train_transformer import build_label_maps, compute_metrics  # noqa: E402


def test_label_maps_are_sorted_and_consistent():
    s = pd.Series(["not_informative", "informative", "informative"])
    label2id, id2label = build_label_maps(s)
    assert label2id == {"informative": 0, "not_informative": 1}
    assert all(label2id[id2label[i]] == i for i in id2label)


def test_label_maps_reject_unknown():
    with pytest.raises(ValueError):
        build_label_maps(pd.Series(["informative", "Unknown"]))


def test_compute_metrics_perfect_and_wrong():
    perfect = compute_metrics(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0
    wrong = compute_metrics(["a", "b", "a", "b"], ["b", "a", "b", "a"])
    assert wrong["accuracy"] == 0.0 and wrong["macro_f1"] == 0.0