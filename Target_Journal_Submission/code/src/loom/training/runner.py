"""One benchmark job = (dataset, resample, feature extractor). Results are cached
as JSON so interrupted runs resume where they stopped."""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from ..data.ucr import load_split
from ..evaluation.metrics import all_metrics, align, reliability_bins, set_metrics

REL_METHODS = {"LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "LogReg-CV", "PreVal-Ens5", "LOOM-Ens5", "Arsenal", "QUANT", "Ridge-softmax"}
from ..models.baselines import LogReg, RidgeCalibratedCV, RidgeOneHot, RidgeSoftmax
from ..models.loom import LOOM, _score
from ..models.readout import ClosedFormRidge
from ..preprocessing.corruptions import all_conditions, corrupt

ROOT = Path(__file__).resolve().parents[3]


def cond_seed(name, resample, fam, sev):
    h = hashlib.sha256(f"{name}|{resample}|{fam}|{sev}".encode()).hexdigest()
    return int(h[:8], 16)


def make_transform(kind, seed):
    from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket, Rocket
    if kind == "minirocket":
        return MiniRocket(random_state=seed)
    if kind == "multirocket":
        return MultiRocket(random_state=seed)
    if kind == "rocket":
        return Rocket(n_kernels=5000, random_state=seed)        # 10,000 features
    if kind == "hydra":
        from aeon.transformations.collection.convolution_based import HydraTransformer
        return HydraTransformer(random_state=seed)
    raise ValueError(kind)


def _feats(tf, X):
    F = tf.transform(X[:, None, :])
    return np.asarray(F, dtype=np.float64)


# methods evaluated on every condition; (name, factory)
MAIN_METHODS = {
    "LOOM": lambda: LOOM("standard"),
    "PreVal": lambda: LOOM("none", smooth=False),
    "Ridge-softmax": RidgeSoftmax,
    "Ridge-onehot": RidgeOneHot,
    "Holdout-TS": lambda: LOOM("none", pairs="holdout", smooth=False),
    "Ridge-Platt-CV5": lambda: RidgeCalibratedCV("sigmoid", 5),
    "Ridge-Isotonic-CV5": lambda: RidgeCalibratedCV("isotonic", 5),
    "LogReg-CV": LogReg,
}
SLOW_METHODS = ("LogReg-CV",)
ABLATIONS = {
    "PreVal-smooth": lambda: LOOM("none", smooth=True),
    "LOOM-nosmooth": lambda: LOOM("standard", smooth=False),
    "LOOM-free-b": lambda: LOOM("free"),
    "LOOM-finite-c1": lambda: LOOM("finite", c=1.0),
    "LOOM-kNN": lambda: LOOM("knn"),
    "LOOM-insample": lambda: LOOM("standard", pairs="insample"),
    "LOOM-holdout": lambda: LOOM("standard", pairs="holdout"),
    "LOOM-lambdaSSE": lambda: LOOM("standard", lambda_criterion="sse"),
}


def run_job(name, resample, transform="minirocket", methods="all", n_ens=5, alpha=0.1,
            outdir=None, skip_slow_n=None, train_frac=1.0):
    outdir = Path(outdir or ROOT / "results" / "raw")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{transform}__{name}__r{resample}__{methods}" + (f"__tf{train_frac}" if train_frac < 1 else "")
    path = outdir / f"{tag}.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_split(name, resample)
    if train_frac < 1:      # stratified training-set reduction (>= 2 per class when available)
        rng = np.random.default_rng(10_000 + resample)
        keep = []
        for c in np.unique(ytr):
            idx = np.flatnonzero(ytr == c)
            k = max(min(2, len(idx)), int(round(train_frac * len(idx))))
            keep.append(rng.choice(idx, k, replace=False))
        keep = np.sort(np.concatenate(keep)); Xtr, ytr = Xtr[keep], ytr[keep]
    classes = np.unique(np.concatenate([ytr, yte]))
    yi_te = np.searchsorted(classes, yte)
    conds = list(all_conditions())

    def X_of(c):   # corrupted inputs are regenerated deterministically (saves memory)
        return Xte if c[0] == "clean" else corrupt(Xte, c[0], c[1], cond_seed(name, resample, *c))

    rows, timing, extra = [], {}, {}
    seeds = [1000 * resample + j for j in range(n_ens if methods in ("all", "main") else 1)]
    probs = {}          # (method, cond) -> P   (needed for ensembles and shift AUROC)

    mset = {}
    if methods in ("all", "main"):
        mset.update(MAIN_METHODS)
    if methods in ("all", "ablation"):
        mset.update(ABLATIONS)
    if methods == "core":
        mset = {k: MAIN_METHODS[k] for k in ("LOOM", "PreVal", "Ridge-softmax", "Holdout-TS")}
    if skip_slow_n and len(ytr) > skip_slow_n:
        for k in SLOW_METHODS:
            mset.pop(k, None)
        extra["skipped_slow"] = list(SLOW_METHODS)

    def add_prob_row(mname, c, P):
        probs[(mname, c)] = P
        rows.append(dict(method=mname, cond=c[0], sev=c[1], **all_metrics(P, yi_te),
                         shift_auroc=shift_auroc(probs[(mname, ("clean", 0))], P) if c[0] != "clean" else float("nan"),
                         **({"rel": reliability_bins(P, yi_te)} if mname in REL_METHODS else {})))

    for j, seed in enumerate(seeds):
        tt = time.time()
        tf = make_transform(transform, seed).fit(Xtr[:, None, :])
        Ftr = _feats(tf, Xtr)
        t_feat_train = time.time() - tt
        ts = time.time(); base = ClosedFormRidge().fit(Ftr, ytr); t_svd = time.time() - ts
        models, fit_t = {}, {}
        names = list(mset) if j == 0 else ["PreVal", "LOOM"]
        for mname in names:
            fac = mset[mname] if j == 0 else MAIN_METHODS[mname]
            try:
                ts = time.time(); models[mname] = fac().fit(Ftr, ytr, ridge=base); fit_t[mname] = time.time() - ts
                if isinstance(models[mname], LOOM) and models[mname].pairs != "holdout":
                    fit_t[mname] += t_svd           # charge the shared decomposition to every user
            except Exception:
                extra.setdefault("errors", {})[f"{mname}#{j}"] = traceback.format_exc()[-1500:]
        if j == 0:
            extra["n_features"] = int(Ftr.shape[1]); extra["feature_time_train"] = t_feat_train; extra["svd_time"] = t_svd
            if "LOOM" in models:
                lm = models["LOOM"]; extra["loom"] = dict(lam=lm.lambda_, a=lm.a_, h_ref=lm.h_ref_, h_test_median={})
            ca = {}
            if "Holdout-TS" in models:
                for al in (alpha, 0.05, 0.2):
                    ca[al] = _split_conformal_threshold(models["Holdout-TS"], Ftr, ytr, al)
                ca_aps = _split_conformal_threshold(models["Holdout-TS"], Ftr, ytr, alpha, score="aps")
        del Ftr
        pred_t = {m: 0.0 for m in models}; cp_t = {}
        t_feat_test = 0.0
        for c in conds:
            tt = time.time(); F = _feats(tf, X_of(c)); t_feat_test += time.time() - tt
            for mname, model in models.items():
                tp = time.time()
                try:
                    P = align(model.predict_proba(F), model.classes_, classes)
                except Exception:
                    extra.setdefault("errors", {})[f"{mname}#{j}:{c}"] = traceback.format_exc()[-1500:]
                    continue
                pred_t[mname] += time.time() - tp
                if j == 0:
                    add_prob_row(mname, c, P)
                else:
                    probs[(f"{mname}#{j}", c)] = P
            if j == 0:
                if "LOOM" in models:
                    lm = models["LOOM"]
                    extra["loom"]["h_test_median"][f"{c[0]}{c[1]}"] = float(np.median(lm.ridge_.leverage(F)))
                    for meth, sc in (("jackknife+", "lac"), ("jackknife", "lac"), ("jackknife+", "aps")):
                        tp = time.time(); nm = f"CP-LOOM-{meth}" + ("-APS" if sc == "aps" else "")
                        S = align(lm.conformal_sets(F, alpha=alpha, method=meth, score=sc).astype(float), lm.classes_, classes) > 0.5
                        cp_t[nm] = cp_t.get(nm, 0.0) + time.time() - tp
                        rows.append(dict(method=nm, cond=c[0], sev=c[1], **set_metrics(S, yi_te)))
                    if c[0] == "clean":
                        for al in (0.05, 0.2):
                            S = align(lm.conformal_sets(F, alpha=al, method="jackknife+").astype(float), lm.classes_, classes) > 0.5
                            rows.append(dict(method=f"CP-LOOM-jackknife+@{al}", cond="clean", sev=0, **set_metrics(S, yi_te)))
                if "PreVal" in models:
                    pv = models["PreVal"]; tp = time.time()
                    for sc in ("lac", "aps"):
                        nm = "CP-PreVal-jackknife+" + ("-APS" if sc == "aps" else "")
                        S = align(pv.conformal_sets(F, alpha=alpha, method="jackknife+", score=sc).astype(float), pv.classes_, classes) > 0.5
                        cp_t[nm] = cp_t.get(nm, 0.0) + time.time() - tp
                        rows.append(dict(method=nm, cond=c[0], sev=c[1], **set_metrics(S, yi_te)))
                if "Holdout-TS" in models:
                    P = probs[("Holdout-TS", c)]
                    rows.append(dict(method="CP-Split-TS", cond=c[0], sev=c[1], **set_metrics((1 - P) <= ca[alpha], yi_te)))
                    rows.append(dict(method="CP-Split-TS-APS", cond=c[0], sev=c[1],
                                     **set_metrics(_score(P, "aps") <= ca_aps, yi_te)))
                    if c[0] == "clean":
                        for al in (0.05, 0.2):
                            rows.append(dict(method=f"CP-Split-TS@{al}", cond="clean", sev=0, **set_metrics((1 - P) <= ca[al], yi_te)))
            del F
        if j == 0:
            extra["feature_time_test_all_conditions"] = t_feat_test
            for m in models:
                timing[m] = dict(fit=fit_t[m], predict_all=pred_t[m])
            for m, t in cp_t.items():
                timing[m] = dict(fit=0.0, predict_all=t)
            timing["_features"] = dict(fit=t_feat_train, predict_all=t_feat_test)
        else:
            for m in models:
                e = timing.setdefault(f"{m}-Ens{len(seeds)}", dict(fit=timing[m]["fit"] + timing["_features"]["fit"],
                                                                  predict_all=timing[m]["predict_all"] + timing["_features"]["predict_all"]))
                e["fit"] += fit_t[m] + t_feat_train; e["predict_all"] += pred_t[m] + t_feat_test
        del models, base
    if len(seeds) > 1:
        for mname in ("PreVal", "LOOM"):
            for c in conds:
                if (mname, c) not in probs:
                    continue
                P = np.mean([probs[(mname, c)]] + [probs[(f"{mname}#{j}", c)] for j in range(1, len(seeds)) if (f"{mname}#{j}", c) in probs], 0)
                add_prob_row(f"{mname}-Ens{len(seeds)}", c, P)
    res = dict(dataset=name, resample=resample, transform=transform, methods=methods, train_frac=train_frac, n_train=len(ytr),
               n_test=len(yte), n_classes=len(classes), rows=rows, timing=timing, extra=extra,
               wall=time.time() - t0)
    tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); os.replace(tmp, path)
    return res


def shift_auroc(P_clean, P_shift):
    """AUROC for separating shifted from clean test inputs with 1 - max probability."""
    from sklearn.metrics import roc_auc_score
    u = np.r_[1 - P_clean.max(1), 1 - P_shift.max(1)]
    lab = np.r_[np.zeros(len(P_clean)), np.ones(len(P_shift))]
    return float(roc_auc_score(lab, u)) if np.ptp(u) > 0 else 0.5


def _split_conformal_threshold(model, Ftr, ytr, alpha, score="lac"):
    """Recover the 20 % calibration split used by the holdout model and compute the
    split-conformal LAC threshold on it."""
    rng = np.random.default_rng(model.seed)
    ica = []
    for c in model.classes_:
        idx = rng.permutation(np.flatnonzero(ytr == c))
        k = int(np.floor(model.holdout_frac * len(idx)))
        if len(idx) >= 2 and k == 0:
            k = 1
        ica.append(idx[:k])
    ica = np.concatenate(ica)
    P = model.predict_proba(Ftr[ica]); yi = np.searchsorted(model.classes_, ytr[ica])
    s = _score(P, score)[np.arange(len(ica)), yi]
    n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s)[k - 1]


def run_raw_baselines(name, resample, outdir=None):
    """Arsenal and QUANT operate on raw series and provide native probabilities."""
    from aeon.classification.convolution_based import Arsenal
    from aeon.classification.interval_based import QUANTClassifier
    outdir = Path(outdir or ROOT / "results" / "raw")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"raw__{name}__r{resample}__rawbaselines.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_split(name, resample)
    classes = np.unique(np.concatenate([ytr, yte])); yi_te = np.searchsorted(classes, yte)
    conds = list(all_conditions())
    rows, timing, errors = [], {}, {}
    # Arsenal: aeon's MiniRocket-kernel variant (25 x 2,000 kernels) instead of the default ROCKET
    # kernels, and on the official split only (resample 0) -- both for computational reasons.
    cands = {"QUANT": lambda: QUANTClassifier(random_state=resample)}
    if resample == 0:
        cands["Arsenal"] = lambda: Arsenal(rocket_transform="minirocket", random_state=resample)
    for mname, fac in cands.items():
        try:
            ts = time.time(); m = fac().fit(Xtr[:, None, :], ytr); ft = time.time() - ts
            tp = time.time(); Pc = None
            for c in conds:
                Xc = Xte if c[0] == "clean" else corrupt(Xte, c[0], c[1], cond_seed(name, resample, *c))
                P = align(m.predict_proba(Xc[:, None, :]), m.classes_, classes)
                if c[0] == "clean":
                    Pc = P
                rows.append(dict(method=mname, cond=c[0], sev=c[1], **all_metrics(P, yi_te),
                                 shift_auroc=shift_auroc(Pc, P) if c[0] != "clean" else float("nan"),
                                 rel=reliability_bins(P, yi_te)))
            timing[mname] = dict(fit=ft, predict_all=time.time() - tp)
        except Exception:
            errors[mname] = traceback.format_exc()[-1500:]
    res = dict(dataset=name, resample=resample, transform="raw", methods="rawbaselines", n_train=len(ytr),
               n_test=len(yte), n_classes=len(classes), rows=rows, timing=timing, extra=dict(errors=errors),
               wall=time.time() - t0)
    tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); os.replace(tmp, path)
    return res
