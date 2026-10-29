"""Application case study: human-in-the-loop triage of sensor recordings with faults.

Protocol: Online Resource 2, section 10 (fixed before this script was run). A classifier labels
incoming recordings and sends the least confident share b of them to a human reviewer, while 30 %
of the recordings carry a domain-specific sensor fault. Per-sample outputs are saved to
results/case_study/<dataset>__r<resample>.npz; the summary goes to results/case_study.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import load_split                                    # noqa: E402
from loom.evaluation.metrics import align, nll                           # noqa: E402
from loom.models.readout import ClosedFormRidge                          # noqa: E402
from loom.preprocessing.corruptions import corrupt                       # noqa: E402
from loom.training.runner import MAIN_METHODS, _feats, cond_seed, make_transform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "case_study"
DOMAINS = {
    "cardiac": (["ECG200", "ECG5000", "TwoLeadECG", "NonInvasiveFetalECGThorax1"], ["noise", "wander", "spikes", "missing"]),
    "wearable": (["UWaveGestureLibraryX", "CricketX", "GunPoint", "SemgHandGenderCh2"], ["noise", "spikes", "missing", "warp"]),
    "industrial": (["Wafer", "FreezerRegularTrain", "PowerCons", "RefrigerationDevices"], ["noise", "wander", "spikes", "missing"]),
}
P_FAULT, BUDGETS, RESAMPLES = 0.3, (0.1, 0.2, 0.3), range(5)
READOUTS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5"]


def faulty_stream(X, name, r, faults):
    """Each recording is faulty with probability P_FAULT; fault family and severity drawn uniformly."""
    rng = np.random.default_rng(cond_seed(name, r, "casestudy", 0))
    is_f = rng.random(len(X)) < P_FAULT
    fam = rng.integers(0, len(faults), len(X)); sev = rng.integers(1, 4, len(X))
    Xs = X.copy()
    for k, f in enumerate(faults):
        for s in (1, 2, 3):
            idx = np.where(is_f & (fam == k) & (sev == s))[0]
            if len(idx):
                Xs[idx] = corrupt(X[idx], f, s, cond_seed(name, r, f"casestudy-{f}", s))
    return Xs, is_f


def triage(conf, correct, is_f, b):
    """Defer the b least confident recordings; return (auto-accepted error rate, share of faults deferred)."""
    n = len(conf); k = int(round(b * n))
    order = np.argsort(conf, kind="stable")          # ascending confidence
    deferred = np.zeros(n, bool); deferred[order[:k]] = True
    auto = ~deferred
    err = float((~correct[auto]).mean()) if auto.any() else float("nan")
    caught = float(deferred[is_f].mean()) if is_f.any() else float("nan")
    return err, caught


def run(name, r, faults):
    path = OUT / f"{name}__r{r}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    from aeon.classification.interval_based import QUANTClassifier
    Xtr, ytr, Xte, yte = load_split(name, r)
    classes = np.unique(np.r_[ytr, yte]); yi = np.searchsorted(classes, yte)
    Xs, is_f = faulty_stream(Xte, name, r, faults)
    tf = make_transform("minirocket", 1000 * r).fit(Xtr[:, None, :])
    Ftr = _feats(tf, Xtr); base = ClosedFormRidge().fit(Ftr, ytr)
    Fc, Fs = _feats(tf, Xte), _feats(tf, Xs)
    out = {"y": yi, "is_fault": is_f}
    for m in READOUTS:
        mod = MAIN_METHODS[m]().fit(Ftr, ytr, ridge=base)
        out[f"P_clean_{m}"] = align(mod.predict_proba(Fc), mod.classes_, classes)
        out[f"P_stream_{m}"] = align(mod.predict_proba(Fs), mod.classes_, classes)
    q = QUANTClassifier(random_state=r).fit(Xtr[:, None, :], ytr)
    out["P_clean_QUANT"] = align(q.predict_proba(Xte[:, None, :]), q.classes_, classes)
    out["P_stream_QUANT"] = align(q.predict_proba(Xs[:, None, :]), q.classes_, classes)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def main():
    rows = []
    for dom, (dsets, faults) in DOMAINS.items():
        for d in dsets:
            for r in RESAMPLES:
                o = run(d, r, faults)
                y, is_f = o["y"], o["is_fault"].astype(bool)
                for m in READOUTS + ["QUANT"]:
                    for stream in ("clean", "stream"):
                        P = o[f"P_{stream}_{m}"]; conf = P.max(1); corr = P.argmax(1) == y
                        rec = dict(domain=dom, dataset=d, resample=r, method=m, stream=stream,
                                   nll=nll(P, y), acc=float(corr.mean()))
                        f = is_f if stream == "stream" else np.zeros(len(y), bool)
                        for b in BUDGETS:
                            e, c = triage(conf, corr, f, b)
                            rec[f"auto_err@{int(b * 100)}"] = e; rec[f"fault_caught@{int(b * 100)}"] = c
                        rows.append(rec)
                print(d, r, flush=True)
    json.dump(rows, open(ROOT / "results" / "case_study.json", "w"), indent=0)


if __name__ == "__main__":
    main()
