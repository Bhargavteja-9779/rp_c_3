"""E17 (post hoc, deviation D19): sanity checks of assumptions and of sampling noise.

1. Exchangeability of leverages (assumption A2 of the paper): on each dataset (official split,
   <= 3,000 training series, clean test data), compare the held-out leverages h_{-i}/h_ref with the
   test leverages h(x)/h_ref by a two-sample Kolmogorov-Smirnov test (Holm over datasets).
2. Coverage spread vs sampling noise: per-dataset jackknife+ coverage (E9, LOOM, LAC, clean) is
   compared with the spread expected from binomial sampling alone at the nominal level 0.9,
   by simulation (2,000 replicates).
3. Resample noise vs between-dataset differences: for LOOM - PreVal clean and faulty NLL, the share of
   datasets whose mean difference exceeds two standard errors estimated from the resamples.
Output: results/e17/sanity.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import dataset_info, dataset_names, load_split         # noqa: E402
from loom.models.loom import LOOM                                         # noqa: E402
from loom.statistics.tests import holm                                     # noqa: E402
from loom.training.runner import _feats, make_transform                    # noqa: E402

OUT = ROOT / "results" / "e17"


def leverage_exchangeability(name):
    Xtr, ytr, Xte, yte = load_split(name, 0)
    if len(Xte) > 2000:
        Xte = Xte[np.random.default_rng(0).choice(len(Xte), 2000, replace=False)]
    tf = make_transform("minirocket", 0).fit(Xtr[:, None, :])
    m = LOOM("standard").fit(_feats(tf, Xtr), ytr)
    r = m.ridge_
    h_loo = r.loo_leverage_                         # d_i / (1 - d_i) at the selected penalty
    assert abs(np.median(h_loo) / m.h_ref_ - 1) < 1e-6, "h_ref must be the median held-out leverage"
    h_te = r.leverage(_feats(tf, Xte))
    ks = ks_2samp(h_loo / m.h_ref_, h_te / m.h_ref_)
    return dict(dataset=name, ks=float(ks.statistic), p=float(ks.pvalue),
                median_ratio_test=float(np.median(h_te) / m.h_ref_), q90_ratio_test=float(np.quantile(h_te, 0.9) / m.h_ref_),
                q90_ratio_loo=float(np.quantile(h_loo, 0.9) / m.h_ref_))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = dataset_info(); res = {}
    cache = OUT / "leverage_exchangeability.json"
    rows = json.load(open(cache)) if cache.exists() else []
    done = {r["dataset"] for r in rows}
    for n in dataset_names():
        if info[n]["n_train"] > 3000 or n in done:
            continue
        try:
            rows.append(leverage_exchangeability(n)); print(n, flush=True)
        except Exception as e:  # noqa: BLE001
            print("FAILED", n, repr(e), flush=True)
        json.dump(rows, open(cache, "w"))
    df = pd.DataFrame(rows); df["p_holm"] = holm(df.p.to_numpy())
    res["exchangeability"] = dict(n=len(df), n_reject_holm=int((df.p_holm < 0.05).sum()),
                                  median_ks=float(df.ks.median()), median_test_ratio=float(df.median_ratio_test.median()),
                                  median_q90_test=float(df.q90_ratio_test.median()), median_q90_loo=float(df.q90_ratio_loo.median()),
                                  rejected=df[df.p_holm < 0.05].dataset.tolist())
    # 2. coverage spread vs binomial sampling noise
    raw = pd.read_csv(ROOT / "results" / "raw_results.csv", usecols=["dataset", "resample", "method", "cond", "coverage", "n_cp", "transform", "train_frac"])
    cp = raw[(raw.method == "CP-LOOM-jackknife+") & (raw.cond == "clean") & (raw["transform"] == "minirocket") & (raw.train_frac == 1.0)]
    cov = cp.groupby("dataset").coverage.mean(); ncp = cp.groupby("dataset").n_cp.sum()
    rng = np.random.default_rng(0)
    sim_sd = [np.std([rng.binomial(int(k), 0.9) / k for k in ncp.values]) for _ in range(2000)]
    res["coverage_spread"] = dict(n=int(len(cov)), observed_sd=float(cov.std(ddof=0)), sim_sd_median=float(np.median(sim_sd)),
                                  sim_sd_q975=float(np.quantile(sim_sd, 0.975)),
                                  share_outside_binomial_95=float(np.mean([abs(c - 0.9) > 1.96 * np.sqrt(0.09 / k) for c, k in zip(cov, ncp)])))
    # 3. resample noise vs dataset differences
    r2 = pd.read_csv(ROOT / "results" / "raw_results.csv", usecols=["dataset", "resample", "method", "cond", "nll", "transform", "train_frac"])
    r2 = r2[(r2["transform"] == "minirocket") & (r2.train_frac == 1.0) & r2.method.isin(["LOOM", "PreVal"])]
    out3 = {}
    for state, sel in (("clean", r2.cond == "clean"), ("faults", r2.cond != "clean")):
        t = r2[sel].groupby(["dataset", "resample", "method"]).nll.mean().unstack()
        d = (t["PreVal"] - t["LOOM"]).groupby("dataset")
        m, se, k = d.mean(), d.std(ddof=1) / np.sqrt(d.count()), d.count()
        ok = k > 1
        out3[state] = dict(n=int(ok.sum()), share_beyond_2se=float((m[ok].abs() > 2 * se[ok]).mean()),
                           share_loom_better_beyond_2se=float((m[ok] > 2 * se[ok]).mean()),
                           share_preval_better_beyond_2se=float((m[ok] < -2 * se[ok]).mean()))
    res["resample_noise"] = out3
    json.dump(res, open(OUT / "sanity.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
