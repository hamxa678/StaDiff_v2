"""
Detection metrics in plain numpy (no scikit-learn needed). Label 1 = anomalous, 0 = clean;
a higher score means more anomalous.
"""
from __future__ import annotations

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get their average rank."""
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def auroc(labels, scores) -> float:
    """Area under the ROC curve (Mann-Whitney U). nan if only one class is present."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(labels, scores) -> float:
    """Area under the precision-recall curve (step-wise, as in scikit-learn)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # evaluate only at the last index of each tied score group
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def binary_metrics(labels, preds) -> dict:
    labels = np.asarray(labels).astype(bool)
    preds = np.asarray(preds).astype(bool)
    tp = int((preds & labels).sum())
    fp = int((preds & ~labels).sum())
    tn = int((~preds & ~labels).sum())
    fn = int((~preds & labels).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(len(labels), 1),
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def best_f1(labels, scores) -> dict:
    """Best F1 over all thresholds. This uses the test labels, so report it only as an
    optimistic upper bound; the honest number uses the threshold calibrated on clean data."""
    scores = np.asarray(scores, dtype=np.float64)
    best = {"f1": -1.0}
    for thr in np.unique(scores):
        m = binary_metrics(labels, scores >= thr)
        if m["f1"] > best["f1"]:
            best = {**m, "threshold": float(thr)}
    return best


def summarize(labels, scores, threshold=None) -> dict:
    out = {"n": int(len(labels)), "n_anomalous": int(np.sum(labels)),
           "auroc": auroc(labels, scores), "average_precision": average_precision(labels, scores),
           "best_f1_oracle": best_f1(labels, scores)}
    if threshold is not None:
        out["at_calibrated_threshold"] = {**binary_metrics(labels, np.asarray(scores) >= threshold),
                                          "threshold": float(threshold)}
    return out
