"""E20: naturally occurring, expert-annotated faults in clinical 12-lead ECG (PTB-XL).

Protocol: Online Resource 2, section 12.1 (committed in 6b05cb1 before this script was written or run).
Five-class diagnostic superclass task, recommended folds (train 1-8, test 9-10). Primary: train on the
clean records of folds 1-8; secondary: train on all records of folds 1-8. Multivariate MiniRocket with
the LOOM, PreVal, Holdout-TS and Ridge-Platt-CV5 readouts, and QUANT on the raw 12-lead series; seeds 0-4.

Per-record outputs: results/e20/<train>_s<seed>.npz. Summary and hypothesis tests: results/e20/summary.json.
Usage: python experiments/e20_ptbxl.py [run|analyze]   (default: both)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import ptbxl                                                              # noqa: E402
from loom.evaluation.metrics import EPS, align, ece                       # noqa: E402
from loom.models.readout import ClosedFormRidge                          # noqa: E402
from loom.training.runner import MAIN_METHODS                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e20"
READOUTS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5"]
METHODS = READOUTS + ["QUANT"]
SEEDS, TRAINS, BUDGETS = range(5), ("clean", "all"), (0.1, 0.2, 0.3)
N_BOOT = 10_000


def run(train, seed):
    path = OUT / f"{train}_s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    from aeon.classification.interval_based import QUANTClassifier
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.ensemble import ExtraTreesClassifier
    X, y, noisy, fold, ids = ptbxl.load()
    tr = (fold <= 8) & (~noisy if train == "clean" else True)
    te = fold >= 9
    Xtr, ytr, Xte = X[tr].astype(np.float64), y[tr], X[te].astype(np.float64)
    classes = np.arange(len(ptbxl.CLASSES))
    out = {"y": y[te], "noisy": noisy[te], "ecg_id": ids[te]}
    t0 = time.time()
    tf = MiniRocket(random_state=seed, n_jobs=4).fit(Xtr)
    Ftr = np.asarray(tf.transform(Xtr), np.float64); Fte = np.asarray(tf.transform(Xte), np.float64)
    out["time_minirocket"] = time.time() - t0
    base = ClosedFormRidge().fit(Ftr, ytr)
    for m in READOUTS:
        t0 = time.time()
        mod = MAIN_METHODS[m]().fit(Ftr, ytr, ridge=base)
        out[f"P_{m}"] = align(mod.predict_proba(Fte), mod.classes_, classes)
        out[f"time_{m}"] = time.time() - t0
        print(train, seed, m, round(out[f"time_{m}"], 1), flush=True)
    del Ftr, Fte, base
    t0 = time.time()
    # QUANT's default estimator (aeon 1.6.0), with n_jobs added; results do not depend on n_jobs.
    et = ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy", n_jobs=4)
    q = QUANTClassifier(estimator=et, random_state=seed).fit(Xtr, ytr)
    out["P_QUANT"] = align(q.predict_proba(Xte), q.classes_, classes)
    out["time_QUANT"] = time.time() - t0
    print(train, seed, "QUANT", round(out["time_QUANT"], 1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# ---------------------------------------------------------------- analysis
def rec_nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))


def queue(conf, b):
    """Boolean mask of the round(b n) least confident records (stable order)."""
    k = int(round(b * len(conf))); q = np.zeros(len(conf), bool)
    q[np.argsort(conf, kind="stable")[:k]] = True
    return q


def auroc(score, lab):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(lab, score)) if 0 < lab.sum() < len(lab) else float("nan")


def describe(P, y, noisy, label_from=None):
    """Metrics of one method on one run. label_from: probabilities whose argmax gives the label (hybrid)."""
    from sklearn.metrics import roc_auc_score
    lab = (label_from if label_from is not None else P).argmax(1)
    corr = lab == y; conf = P.max(1); nl = rec_nll(P, y)
    d = dict(acc=float(corr.mean()), acc_clean=float(corr[~noisy].mean()), acc_noisy=float(corr[noisy].mean()),
             nll=float(nl.mean()), nll_clean=float(nl[~noisy].mean()), nll_noisy=float(nl[noisy].mean()),
             ece=ece(P, y), macro_auroc=float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
             detect_auroc=auroc(1 - conf, noisy), mean_conf_clean=float(conf[~noisy].mean()),
             mean_conf_noisy=float(conf[noisy].mean()))
    for b in BUDGETS:
        q = queue(conf, b); t = int(b * 100)
        d[f"caught@{t}"] = float(q[noisy].mean())
        d[f"auto_err@{t}"] = float((~corr[~q]).mean())
        d[f"auto_err_noisy@{t}"] = float((~corr[~q & noisy]).mean()) if (~q & noisy).any() else float("nan")
    return d


def hybrid(o, rule):
    """E21 rule on E20: label from QUANT; queue from LOOM confidence (primary) or min(LOOM, QUANT) (secondary)."""
    PL, PQ = o["P_LOOM"], o["P_QUANT"]
    conf = PL.max(1) if rule == "primary" else np.minimum(PL.max(1), PQ.max(1))
    y, noisy = o["y"], o["noisy"].astype(bool); corr = PQ.argmax(1) == y
    d = {}
    for b in BUDGETS:
        q = queue(conf, b); t = int(b * 100)
        d[f"caught@{t}"] = float(q[noisy].mean()); d[f"auto_err@{t}"] = float((~corr[~q]).mean())
    return d


def h8(o, b=0.2):
    """Exact McNemar test on the annotated-noisy records: in LOOM's queue vs in PreVal's queue."""
    from scipy.stats import binomtest
    noisy = o["noisy"].astype(bool)
    qL, qP = queue(o["P_LOOM"].max(1), b)[noisy], queue(o["P_PreVal"].max(1), b)[noisy]
    a, c = int((qL & ~qP).sum()), int((~qL & qP).sum())
    p = float(binomtest(a, a + c, 0.5).pvalue) if a + c else 1.0
    return dict(n_noisy=int(noisy.sum()), loom_only=a, preval_only=c, both=int((qL & qP).sum()),
                caught_loom=float(qL.mean()), caught_preval=float(qP.mean()), p=p, passed=bool(p < 0.05 and a > c))


def h9(o):
    """Paired bootstrap (records) of the mean NLL difference LOOM - PreVal on the annotated-noisy records."""
    noisy = o["noisy"].astype(bool); y = o["y"][noisy]
    dlt = rec_nll(o["P_LOOM"][noisy], y) - rec_nll(o["P_PreVal"][noisy], y)
    rng = np.random.default_rng(0)
    bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean_diff=float(dlt.mean()), ci_low=float(lo), ci_high=float(hi),
                nll_noisy_loom=float(rec_nll(o["P_LOOM"][noisy], y).mean()),
                nll_noisy_preval=float(rec_nll(o["P_PreVal"][noisy], y).mean()), passed=bool(hi < 0))


def agg(dicts):
    keys = dicts[0].keys()
    return {k: dict(mean=float(np.nanmean([d[k] for d in dicts])), min=float(np.nanmin([d[k] for d in dicts])),
                    max=float(np.nanmax([d[k] for d in dicts])), per_seed=[d[k] for d in dicts]) for k in keys}


def analyze():
    X, y, noisy, fold, _ = ptbxl.load()
    summ = {"experiment": "E20", "classes": ptbxl.CLASSES,
            "data": dict(n_records=int(len(y)), n_noisy=int(noisy.sum()),
                         n_train_all=int((fold <= 8).sum()), n_train_clean=int(((fold <= 8) & ~noisy).sum()),
                         n_test=int((fold >= 9).sum()), n_test_noisy=int(((fold >= 9) & noisy).sum()),
                         class_counts=np.bincount(y).tolist(),
                         noisy_share_train=float(noisy[fold <= 8].mean()), noisy_share_test=float(noisy[fold >= 9].mean()))}
    for train in TRAINS:
        runs = [run(train, s) for s in SEEDS if (OUT / f"{train}_s{s}.npz").exists()]
        if not runs:
            continue
        S = {"n_seeds": len(runs), "methods": {}, "hybrid": {}}
        for m in METHODS:
            S["methods"][m] = agg([describe(o[f"P_{m}"], o["y"], o["noisy"].astype(bool)) for o in runs])
            S["methods"][m]["time_s"] = float(np.mean([o[f"time_{m}"] for o in runs]))
        for rule in ("primary", "secondary"):
            S["hybrid"][rule] = agg([hybrid(o, rule) for o in runs])
        S["H8_seed0"], S["H9_seed0"] = h8(runs[0]), h9(runs[0])
        S["H8_all_seeds"] = [h8(o) for o in runs]; S["H9_all_seeds"] = [h9(o) for o in runs]
        summ[train] = S
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    for train in TRAINS:
        if train in summ:
            S = summ[train]
            print(train, "H8", S["H8_seed0"], "\n   H9", S["H9_seed0"])
            for m in METHODS:
                M = S["methods"][m]
                print(f"  {m:16s} acc {M['acc']['mean']:.3f} nll {M['nll']['mean']:.3f} nll_noisy {M['nll_noisy']['mean']:.3f}"
                      f" det {M['detect_auroc']['mean']:.3f} caught@20 {M['caught@20']['mean']:.3f} err@20 {M['auto_err@20']['mean']:.3f}")
            for rule, H in S["hybrid"].items():
                print(f"  hybrid-{rule:9s} caught@20 {H['caught@20']['mean']:.3f} err@20 {H['auto_err@20']['mean']:.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("run", "both"):
        for train in TRAINS:
            for s in SEEDS:
                run(train, s)
    if mode in ("analyze", "both"):
        analyze()
