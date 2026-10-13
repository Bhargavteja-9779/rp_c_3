"""Feasibility pilot on a small, fixed development subset (disclosed in the paper).

Question: does the leverage term carry information about distribution shift that
a global temperature (PreVal) cannot use?  Only DEV datasets are used here.
"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aeon.transformations.collection.convolution_based import MiniRocket
from loom.models.readout import ClosedFormRidge
from loom.models.calibration import fit_moderation, moderated_log_proba

RAW = Path(__file__).resolve().parents[1] / "data/raw/Univariate_arff"


def load(name):
    tr = np.loadtxt(RAW / name / f"{name}_TRAIN.txt"); te = np.loadtxt(RAW / name / f"{name}_TEST.txt")
    return tr[:, 1:], tr[:, 0], te[:, 1:], te[:, 0]


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, bins + 1); e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def corrupt(X, kind, rng):
    X = X.copy(); n, L = X.shape; sd = X.std(1, keepdims=True) + 1e-8
    if kind == "noise":
        return X + rng.normal(0, 0.5, X.shape) * sd
    if kind == "missing":
        w = int(0.25 * L)
        for i in range(n):
            s = rng.integers(0, L - w + 1); a = X[i, s - 1] if s > 0 else X[i, min(s + w, L - 1)]
            b = X[i, s + w] if s + w < L else a
            X[i, s:s + w] = np.linspace(a, b, w)
        return X
    if kind == "wander":
        t = np.arange(L) / L
        ph = rng.uniform(0, 2 * np.pi, (n, 1)); f = rng.uniform(0.5, 2, (n, 1))
        return X + 1.0 * sd * np.sin(2 * np.pi * f * t + ph)
    return X


def run(name, seed=0):
    Xtr, ytr, Xte, yte = load(name)
    classes = np.unique(ytr); yi_tr = np.searchsorted(classes, ytr); yi_te = np.searchsorted(classes, yte)
    tf = MiniRocket(random_state=seed).fit(Xtr[:, None, :])
    Ftr = tf.transform(Xtr[:, None, :]); m = ClosedFormRidge().fit(Ftr, ytr)
    # joint selection of lambda with calibrator by LOO NLL (as in PreVal)
    best = {}
    for fit_b in (False, True):
        cand = []
        for lam in m.lambdas:
            m.set_lambda(lam)
            a, b, nll = fit_moderation(m.loo_logits_, m.loo_leverage_, yi_tr, fit_b=fit_b)
            cand.append((nll, lam, a, b))
        best[fit_b] = min(cand)
    rng = np.random.default_rng(1234)
    out = {}
    for kind in ["clean", "noise", "missing", "wander"]:
        Xc = Xte if kind == "clean" else corrupt(Xte, kind, rng)
        Fte = tf.transform(Xc[:, None, :])
        for fit_b in (False, True):
            nll_loo, lam, a, b = best[fit_b]
            m.set_lambda(lam)
            Z, h, _ = m.logits_and_leverage(Fte)
            LP = moderated_log_proba(Z, h, a, b); P = np.exp(LP)
            key = f"{kind}/{'LOOM' if fit_b else 'PreVal'}"
            out[key] = dict(acc=float((P.argmax(1) == yi_te).mean()), nll=float(-LP[np.arange(len(yi_te)), yi_te].mean()),
                            ece=float(ece(P, yi_te)), lam=float(lam), a=a, b=b,
                            h_test_med=float(np.median(h)), h_loo_med=float(np.median(m.loo_leverage_)))
    return out


if __name__ == "__main__":
    import csv
    names = [r["dataset"] for r in csv.DictReader(open(RAW.parent / "ucr_equal_length.csv"))]
    rng = np.random.default_rng(0); dev = sorted(rng.choice(names, 8, replace=False).tolist())
    print("DEV:", dev)
    res = {}
    for nm in dev:
        t = time.time(); res[nm] = run(nm); print(nm, f"{time.time()-t:.1f}s")
        for k, v in res[nm].items():
            print(f"   {k:16s} acc={v['acc']:.3f} nll={v['nll']:.3f} ece={v['ece']:.3f} lam={v['lam']:.3g} a={v['a']:.3g} b={v['b']:.3g} hte={v['h_test_med']:.3g} hloo={v['h_loo_med']:.3g}")
    out = Path(__file__).resolve().parents[1] / "results/dev_pilot.json"; out.parent.mkdir(exist_ok=True, parents=True)
    json.dump(dict(dev=dev, results=res), open(out, "w"), indent=1)
