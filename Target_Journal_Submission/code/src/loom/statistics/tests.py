"""Statistical comparison of methods over datasets.

Unit of analysis: dataset (metrics are first averaged over resamples, so the
paired observations are independent across datasets).  Following Demsar (2006)
and Benavoli et al. (2016) we use the two-sided Wilcoxon signed-rank test for
pairwise comparisons against the proposed method, Holm's step-down correction
for the family of comparisons, the matched-pairs rank-biserial correlation as
effect size, and a percentile bootstrap CI for the median paired difference.
Friedman's test with average ranks summarises all methods jointly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (Kerby 2014); zeros dropped.
    Positive -> the first method has the larger values."""
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_median_ci(d, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def pairwise_vs(table: pd.DataFrame, ref: str, lower_is_better=True, tol=0.0):
    """table: rows = datasets, columns = methods (one metric).
    Returns one row per competitor with W/D/L (from ref's perspective), Wilcoxon p,
    Holm-adjusted p, effect size and bootstrap CI of median(competitor - ref)."""
    out = []
    for m in table.columns:
        if m == ref:
            continue
        pair = table[[ref, m]].dropna()
        d = (pair[m] - pair[ref]).to_numpy()                 # >0 -> competitor larger
        better = d > tol if lower_is_better else d < -tol     # ref better
        worse = d < -tol if lower_is_better else d > tol
        try:
            p = stats.wilcoxon(pair[m], pair[ref], zero_method="wilcox", alternative="two-sided").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = bootstrap_median_ci(d)
        out.append(dict(competitor=m, n=len(d), ref_wins=int(better.sum()), draws=int(len(d) - better.sum() - worse.sum()),
                        ref_losses=int(worse.sum()), median_diff=float(np.median(d)), ci_low=lo, ci_high=hi,
                        rank_biserial=rank_biserial(d), p=float(p)))
    df = pd.DataFrame(out)
    if len(df):
        df["p_holm"] = holm(df["p"].to_numpy())
    return df


def friedman_ranks(table: pd.DataFrame, lower_is_better=True):
    t = table.dropna()
    R = t.rank(axis=1, ascending=lower_is_better, method="average")
    stat, p = stats.friedmanchisquare(*[t[c].to_numpy() for c in t.columns]) if t.shape[1] > 2 else (np.nan, np.nan)
    return R.mean().sort_values(), float(stat), float(p), len(t)


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference of the Nemenyi test (Demsar 2006)."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
