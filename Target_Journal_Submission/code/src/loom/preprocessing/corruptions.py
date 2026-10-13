"""Test-time corruption benchmark for univariate series (fixed before evaluation).

Six families x three severities, applied only to *test* series.  Amplitudes are
relative to each series' own standard deviation so that the benchmark is
comparable across datasets with different scales.  All randomness is driven by
an explicit seed so every method sees identical corrupted inputs.
"""
from __future__ import annotations

import numpy as np

SEVERITY = {
    "noise":   [0.1, 0.3, 0.5],      # Gaussian noise std (x series std)
    "wander":  [0.25, 0.5, 1.0],     # low-frequency sinusoidal baseline amplitude (x std)
    "missing": [0.1, 0.2, 0.3],      # fraction of a contiguous segment replaced by linear interpolation
    "spikes":  [0.01, 0.03, 0.05],   # fraction of points hit by +-3 std impulses
    "warp":    [0.05, 0.1, 0.2],     # max local time-stretch (slope in [1-s, 1+s]) of a smooth monotone warp
    "shift":   [0.05, 0.1, 0.2],     # circular time shift (fraction of length, random sign)
}
FAMILIES = list(SEVERITY)


def corrupt(X: np.ndarray, family: str, severity: int, seed: int) -> np.ndarray:
    """severity in {1, 2, 3}."""
    rng = np.random.default_rng(seed)
    X = np.array(X, dtype=np.float64, copy=True)
    n, L = X.shape
    s = SEVERITY[family][severity - 1]
    sd = X.std(1, keepdims=True) + 1e-8
    t = np.arange(L) / max(L - 1, 1)
    if family == "noise":
        return X + rng.normal(0.0, s, X.shape) * sd
    if family == "wander":
        f = rng.uniform(0.5, 2.0, (n, 1)); ph = rng.uniform(0, 2 * np.pi, (n, 1))
        return X + s * sd * np.sin(2 * np.pi * f * t + ph)
    if family == "missing":
        w = max(1, int(round(s * L)))
        for i in range(n):
            st = rng.integers(0, L - w + 1)
            a = X[i, st - 1] if st > 0 else X[i, min(st + w, L - 1)]
            b = X[i, st + w] if st + w < L else a
            X[i, st:st + w] = np.linspace(a, b, w)
        return X
    if family == "spikes":
        m = rng.random(X.shape) < s
        return X + m * rng.choice([-3.0, 3.0], X.shape) * sd
    if family == "warp":
        out = np.empty_like(X)
        for i in range(n):
            # smooth monotone warp t + c sin(pi k t), c = s u / (pi k), |u| <= 1, so slope in [1-s, 1+s]
            k = rng.integers(1, 3)
            d = s * np.sin(np.pi * k * t) * rng.uniform(-1, 1)
            tw = np.clip(t + d / (np.pi * k) , 0, 1)
            tw = np.maximum.accumulate(tw)
            out[i] = np.interp(tw, t, X[i])
        return out
    if family == "shift":
        k = int(round(s * L))
        sign = rng.choice([-1, 1], n)
        return np.stack([np.roll(X[i], sign[i] * k) for i in range(n)])
    raise ValueError(family)


def all_conditions():
    yield ("clean", 0)
    for f in FAMILIES:
        for sev in (1, 2, 3):
            yield (f, sev)
