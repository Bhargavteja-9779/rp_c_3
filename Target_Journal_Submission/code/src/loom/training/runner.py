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
from ..evaluation.metrics import all_metrics, align, set_metrics
from ..models.baselines import LogReg, RidgeCalibratedCV, RidgeOneHot, RidgeSoftmax
from ..models.loom import LOOM
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
            outdir=None, skip_slow_n=None):
    outdir = Path(outdir or ROOT / "results" / "raw")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{transform}__{name}__r{resample}__{methods}"
    path = outdir / f"{tag}.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_split(name, resample)
    classes = np.unique(np.concatenate([ytr, yte]))
    yi_te = np.searchsorted(classes, yte)
    conds = list(all_conditions())
    Xs = {c: (Xte if c[0] == "clean" else corrupt(Xte, c[0], c[1], cond_seed(name, resample, *c))) for c in conds}

    rows, timing, extra = [], {}, {}
    seeds = [1000 * resample + j for j in range(n_ens if methods in ("all", "main") else 1)]
    probs = {}          # (method, cond) -> P   (for ensembles)

    def evaluate(mname, model, Fte_by_cond, fit_time):
        tp = time.time()
        for c in conds:
            P = align(model.predict_proba(Fte_by_cond[c]), model.classes_, classes)
            probs[(mname, c)] = P
            rows.append(dict(method=mname, cond=c[0], sev=c[1], **all_metrics(P, yi_te),
                             shift_auroc=shift_auroc(probs[(mname, ("clean", 0))], P) if c[0] != "clean" else float("nan")))
        timing[mname] = dict(fit=fit_time, predict_all=time.time() - tp)

    for j, seed in enumerate(seeds):
        tt = time.time()
        tf = make_transform(transform, seed).fit(Xtr[:, None, :])
        Ftr = _feats(tf, Xtr)
        Fte = {c: _feats(tf, Xs[c]) for c in conds}
        t_feat = time.time() - tt
        if j == 0:
            extra["n_features"] = int(Ftr.shape[1]); extra["feature_time"] = t_feat
            mset = {}
            if methods in ("all", "main"):
                mset.update(MAIN_METHODS)
            if methods in ("all", "ablation"):
                mset.update(ABLATIONS)
            if methods == "core":
                mset = {k: MAIN_METHODS[k] for k in ("LOOM", "PreVal", "Ridge-softmax", "Holdout-TS")}
            if skip_slow_n and len(ytr) > skip_slow_n:
                for k in ("LogReg-CV", "Ridge-Isotonic-CV5", "Ridge-Platt-CV5"):
                    mset.pop(k, None)
            for mname, fac in mset.items():
                try:
                    ts = time.time(); model = fac().fit(Ftr, ytr); ft = time.time() - ts
                    evaluate(mname, model, Fte, ft)
                    if mname == "LOOM":
                        extra["loom"] = dict(lam=model.lambda_, a=model.a_, h_ref=model.h_ref_,
                                             h_test_median={f"{c[0]}{c[1]}": float(np.median(model.ridge_.leverage(Fte[c]))) for c in conds})
                        # conformal prediction sets
                        for meth in ("jackknife+", "jackknife"):
                            ts = time.time()
                            for c in conds:
                                S = align(model.conformal_sets(Fte[c], alpha=alpha, method=meth).astype(float), model.classes_, classes) > 0.5
                                rows.append(dict(method=f"CP-LOOM-{meth}", cond=c[0], sev=c[1], **set_metrics(S, yi_te)))
                            timing[f"CP-LOOM-{meth}"] = dict(fit=0.0, predict_all=time.time() - ts)
                    if mname == "PreVal":
                        ts = time.time()
                        for c in conds:
                            S = align(model.conformal_sets(Fte[c], alpha=alpha, method="jackknife+").astype(float), model.classes_, classes) > 0.5
                            rows.append(dict(method="CP-PreVal-jackknife+", cond=c[0], sev=c[1], **set_metrics(S, yi_te)))
                        timing["CP-PreVal-jackknife+"] = dict(fit=0.0, predict_all=time.time() - ts)
                    if mname == "LOOM":
                        for al in (0.05, 0.2):     # sensitivity to the miscoverage level (clean only)
                            S = align(model.conformal_sets(Fte[("clean", 0)], alpha=al, method="jackknife+").astype(float), model.classes_, classes) > 0.5
                            rows.append(dict(method=f"CP-LOOM-jackknife+@{al}", cond="clean", sev=0, **set_metrics(S, yi_te)))
                    if mname == "Holdout-TS":
                        # split conformal with the 20 % calibration part of the holdout model
                        ca = _split_conformal_threshold(model, Ftr, ytr, alpha)
                        for c in conds:
                            P = align(model.predict_proba(Fte[c]), model.classes_, classes)
                            rows.append(dict(method="CP-Split-TS", cond=c[0], sev=c[1], **set_metrics((1 - P) <= ca, yi_te)))
                        for al in (0.05, 0.2):
                            ca2 = _split_conformal_threshold(model, Ftr, ytr, al)
                            P = align(model.predict_proba(Fte[("clean", 0)]), model.classes_, classes)
                            rows.append(dict(method=f"CP-Split-TS@{al}", cond="clean", sev=0, **set_metrics((1 - P) <= ca2, yi_te)))
                except Exception as e:  # report failures instead of hiding them
                    extra.setdefault("errors", {})[mname] = traceback.format_exc()[-1500:]
        else:
            # extra ensemble members: PreVal and LOOM refitted on another MiniRocket seed
            for mname in ("PreVal", "LOOM"):
                fac = MAIN_METHODS[mname]
                ts = time.time(); model = fac().fit(Ftr, ytr); ft = time.time() - ts
                for c in conds:
                    P = align(model.predict_proba(Fte[c]), model.classes_, classes)
                    probs[(f"{mname}#{j}", c)] = P
                timing.setdefault(f"{mname}-Ens{len(seeds)}", dict(fit=timing[mname]["fit"], predict_all=0.0))
                timing[f"{mname}-Ens{len(seeds)}"]["fit"] += ft + t_feat
    if len(seeds) > 1:
        for mname in ("PreVal", "LOOM"):
            for c in conds:
                P = np.mean([probs[(mname, c)]] + [probs[(f"{mname}#{j}", c)] for j in range(1, len(seeds))], 0)
                probs[(f"{mname}-Ens", c)] = P
                rows.append(dict(method=f"{mname}-Ens{len(seeds)}", cond=c[0], sev=c[1], **all_metrics(P, yi_te),
                                 shift_auroc=shift_auroc(probs[(f"{mname}-Ens", ("clean", 0))], P) if c[0] != "clean" else float("nan")))
    res = dict(dataset=name, resample=resample, transform=transform, methods=methods, n_train=len(ytr),
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


def _split_conformal_threshold(model, Ftr, ytr, alpha):
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
    s = 1 - P[np.arange(len(ica)), yi]
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
    Xs = {c: (Xte if c[0] == "clean" else corrupt(Xte, c[0], c[1], cond_seed(name, resample, *c))) for c in conds}
    rows, timing, errors = [], {}, {}
    for mname, fac in {"Arsenal": lambda: Arsenal(random_state=resample),
                       "QUANT": lambda: QUANTClassifier(random_state=resample)}.items():
        try:
            ts = time.time(); m = fac().fit(Xtr[:, None, :], ytr); ft = time.time() - ts
            tp = time.time(); Pc = None
            for c in conds:
                P = align(m.predict_proba(Xs[c][:, None, :]), m.classes_, classes)
                if c[0] == "clean":
                    Pc = P
                rows.append(dict(method=mname, cond=c[0], sev=c[1], **all_metrics(P, yi_te),
                                 shift_auroc=shift_auroc(Pc, P) if c[0] != "clean" else float("nan")))
            timing[mname] = dict(fit=ft, predict_all=time.time() - tp)
        except Exception:
            errors[mname] = traceback.format_exc()[-1500:]
    res = dict(dataset=name, resample=resample, transform="raw", methods="rawbaselines", n_train=len(ytr),
               n_test=len(yte), n_classes=len(classes), rows=rows, timing=timing, extra=dict(errors=errors),
               wall=time.time() - t0)
    tmp = path.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); os.replace(tmp, path)
    return res
