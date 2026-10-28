"""Small metric helpers (numpy/pandas only) so evaluation scripts run anywhere."""

import numpy as np
import pandas as pd


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def prf(y_true, y_pred, labels):
    """Per-class precision/recall/F1 plus accuracy and macro averages."""
    m = confusion(y_true, y_pred, labels)
    out = {}
    for i, l in enumerate(labels):
        tp = m[i, i]
        fp = m[:, i].sum() - tp
        fn = m[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": int(m[i, :].sum())}
    acc = float(np.trace(m) / m.sum()) if m.sum() else 0.0
    macro = {k: round(float(np.mean([out[l][k] for l in labels])), 4) for k in ("precision", "recall", "f1")}
    return {"accuracy": round(acc, 4), "macro": macro, "per_class": out,
            "labels": list(labels), "confusion_matrix": m.tolist()}


def set_prf(gold_sets, pred_sets):
    """Micro precision/recall/F1 over items (each argument: list of sets, one per post)."""
    tp = fp = fn = 0
    for g, p in zip(gold_sets, pred_sets):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def spearman(a, b):
    """Spearman correlation = Pearson correlation of average ranks (no scipy needed).
    Returns None if either side is constant (correlation undefined)."""
    ra, rb = pd.Series(list(a)).rank(), pd.Series(list(b)).rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return None
    return round(float(ra.corr(rb)), 4)
