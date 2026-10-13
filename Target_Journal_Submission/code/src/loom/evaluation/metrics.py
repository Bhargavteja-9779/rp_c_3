"""Evaluation metrics for probabilistic classifiers and prediction sets."""
from __future__ import annotations

import numpy as np

EPS = 1e-15


def align(P, classes_model, classes_all):
    """Map model probability columns onto the full label set (missing -> 0)."""
    if len(classes_model) == len(classes_all) and np.all(classes_model == classes_all):
        return P
    out = np.zeros((len(P), len(classes_all)))
    out[:, np.searchsorted(classes_all, classes_model)] = P
    return out


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(P, y):
    O = np.zeros_like(P); O[np.arange(len(y)), y] = 1.0
    return float(((P - O) ** 2).sum(1).mean())


def ece(P, y, bins=15):
    """Top-label expected calibration error with equal-width bins."""
    conf = P.max(1); acc = (P.argmax(1) == y)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        msk = idx == b
        if msk.any():
            e += msk.mean() * abs(acc[msk].mean() - conf[msk].mean())
    return float(e)


def classwise_ece(P, y, bins=15):
    K = P.shape[1]; tot = 0.0
    for k in range(K):
        p = P[:, k]; o = (y == k).astype(float)
        idx = np.minimum((p * bins).astype(int), bins - 1)
        for b in range(bins):
            msk = idx == b
            if msk.any():
                tot += msk.mean() * abs(o[msk].mean() - p[msk].mean())
    return float(tot / K)


def aurc(P, y):
    """Area under the risk-coverage curve (selective classification, lower is better)."""
    conf = P.max(1); err = (P.argmax(1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    risks = np.cumsum(err[order]) / np.arange(1, len(y) + 1)
    return float(risks.mean())


def balanced_accuracy(P, y):
    pred = P.argmax(1)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def all_metrics(P, y):
    P = np.clip(P, 0, 1)
    s = P.sum(1, keepdims=True); P = P / np.where(s > 0, s, 1)
    return dict(acc=float((P.argmax(1) == y).mean()), bacc=balanced_accuracy(P, y), nll=nll(P, y),
                brier=brier(P, y), ece=ece(P, y), cwece=classwise_ece(P, y), aurc=aurc(P, y),
                mean_conf=float(P.max(1).mean()))


def set_metrics(S, y):
    """S: boolean prediction-set matrix (m x K)."""
    cov = S[np.arange(len(y)), y]
    size = S.sum(1)
    return dict(coverage=float(cov.mean()), set_size=float(size.mean()), empty=float((size == 0).mean()),
                singleton_acc=float(cov[size == 1].mean()) if (size == 1).any() else float("nan"))


def reliability_bins(P, y, bins=15):
    """Per-bin [count, sum of confidence, sum of correct] for pooled reliability diagrams."""
    conf = P.max(1); acc = (P.argmax(1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return [np.bincount(idx, minlength=bins).tolist(), np.bincount(idx, conf, bins).round(6).tolist(),
            np.bincount(idx, acc, bins).tolist()]
