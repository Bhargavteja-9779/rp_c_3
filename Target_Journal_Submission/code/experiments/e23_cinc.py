"""E23: real artefacts from another acquisition setting (PhysioNet/Computing in Cardiology Challenge 2011, set A).

Protocol: research_design/experimental_protocol.md, rows H11, H12 and E23 (committed in 3de341c before this
script was written or run). Models are trained exactly as in the E20 primary analysis (clean PTB-XL training
records, folds 1-8, seeds 0-4) and applied unchanged to the 998 graded records of CinC 2011 set A
(773 acceptable, 225 unacceptable; the 2 indeterminate records are excluded). No CinC record is used for
fitting, tuning or choosing anything. CinC has no diagnostic labels, so only quality triage is evaluated:
does low confidence single out the records that the challenge annotators judged unacceptable?

Per-record outputs: results/e23/s<seed>.npz. Summary and hypothesis tests: results/e23/summary.json.
Usage: python experiments/e23_cinc.py [run|analyze]   (default: both)
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
from e20_ptbxl import READOUTS, quant_fit_predict, queue, auroc         # noqa: E402
from loom.evaluation.metrics import align                                # noqa: E402
from loom.models.readout import ClosedFormRidge                          # noqa: E402
from loom.training.runner import MAIN_METHODS                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cinc2011" / "set-a"
CACHE = ROOT / "data" / "processed" / "cinc2011_seta_100.npz"
OUT = ROOT / "results" / "e23"
PTB_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SEEDS, BUDGETS, N_BOOT = range(5), (0.1, 0.2, 0.3), 10_000
METHODS = READOUTS + ["QUANT"]
DEEP = ["Deep-single", "Deep-ensemble", "MC-dropout"]


def read_record(rec):
    """(12, 5000) signal in mV in the PTB-XL lead order; checks initial values and checksums in the header."""
    lines = [l for l in (RAW / f"{rec}.hea").read_text().split("\n") if l.strip() and not l.startswith("#")]
    nsig, fs, nsamp = int(lines[0].split()[1]), int(lines[0].split()[2]), int(lines[0].split()[3])
    assert (nsig, fs, nsamp) == (12, 500, 5000), rec
    raw = np.fromfile(RAW / f"{rec}.dat", dtype="<i2")
    assert raw.size == nsig * nsamp, rec
    raw = raw.reshape(nsamp, nsig).T
    sig = {}
    for i, l in enumerate(lines[1:1 + nsig]):
        f = l.split()
        assert f[1] == "16", (rec, f[1])
        gain = float(f[2].split("(")[0].split("/")[0]); zero = int(f[4]); init, chk = int(f[5]), int(f[6])
        assert raw[i, 0] == init, (rec, i, "initial value")
        assert int(raw[i].astype(np.int64).sum()) % 65536 == chk % 65536, (rec, i, "checksum")
        sig[f[-1].upper()] = (raw[i] - zero) / gain
    assert sorted(sig) == sorted(PTB_LEADS), (rec, sorted(sig))
    return np.stack([sig[k] for k in PTB_LEADS]).astype(np.float64)


def load():
    """Signals (998, 12, 1000) float32 at 100 Hz, label (1 = unacceptable) and record ids."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["unacc"], z["rec"]
    from scipy.signal import resample_poly
    acc = (RAW / "RECORDS-acceptable").read_text().split()
    una = (RAW / "RECORDS-unacceptable").read_text().split()
    assert not set(acc) & set(una)
    recs = sorted(acc + una, key=int)
    X = np.stack([resample_poly(read_record(r), 1, 5, axis=-1) for r in recs]).astype(np.float32)
    unacc = np.array([r in set(una) for r in recs])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, unacc=unacc, rec=np.array(recs, dtype=np.int64))
    return load()


def run(seed):
    path = OUT / f"s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    part = OUT / f"s{seed}_readouts.npz"
    X, y, noisy, fold, ids = ptbxl.load()
    tr = (fold <= 8) & ~noisy                               # E20 primary analysis: clean training records
    Xc, unacc, rec = load()
    classes = np.arange(len(ptbxl.CLASSES))
    if part.exists():
        out = dict(np.load(part))
    else:
        from aeon.transformations.collection.convolution_based import MiniRocket
        Xtr, ytr = X[tr].astype(np.float64), y[tr]
        out = {"unacc": unacc, "rec": rec}
        t0 = time.time()
        tf = MiniRocket(random_state=seed, n_jobs=4).fit(Xtr)
        Ftr = np.asarray(tf.transform(Xtr), np.float64); Fc = np.asarray(tf.transform(Xc.astype(np.float64)), np.float64)
        out["time_minirocket"] = time.time() - t0
        del Xtr
        base = ClosedFormRidge().fit(Ftr, ytr)
        for m in READOUTS:
            mod = MAIN_METHODS[m]().fit(Ftr, ytr, ridge=base)
            out[f"P_{m}"] = align(mod.predict_proba(Fc), mod.classes_, classes)
            if m == "LOOM":
                out["lev_ratio"] = mod.ridge_.leverage(Fc) / mod.h_ref_
            print("cinc", seed, m, flush=True)
        del Ftr, Fc, base
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(part, **out)
    t0 = time.time()
    P, cls = quant_fit_predict(X[tr], y[tr], Xc, seed)
    out["P_QUANT"] = align(P, cls, classes)
    out["time_QUANT"] = time.time() - t0
    print("cinc", seed, "QUANT", round(out["time_QUANT"], 1), flush=True)
    np.savez_compressed(path, **out)
    part.unlink()
    return out


# ---------------------------------------------------------------- analysis
def scores(o):
    """Review score per method (higher = review first): 1 - max probability; leverage ratio for 'Leverage'."""
    s = {m: 1 - o[f"P_{m}"].max(1) for m in METHODS if f"P_{m}" in o}
    s["Leverage"] = o["lev_ratio"]
    return s


def describe(score, unacc):
    d = {"detect_auroc": auroc(score, unacc)}
    for b in BUDGETS:
        q = queue(-score, b)                                 # queue() takes confidence: lowest first
        d[f"caught@{int(b * 100)}"] = float(q[unacc].mean())
        d[f"precision@{int(b * 100)}"] = float(unacc[q].mean())
    return d


def h11(sL, sP, unacc, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); n = len(unacc); diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if 0 < unacc[i].sum() < n:
            diffs.append(roc_auc_score(unacc[i], sL[i]) - roc_auc_score(unacc[i], sP[i]))
    d = auroc(sL, unacc) - auroc(sP, unacc)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(auroc_loom=auroc(sL, unacc), auroc_preval=auroc(sP, unacc), diff=float(d), ci_low=float(lo),
                ci_high=float(hi), n_boot=len(diffs), passed=bool(lo > 0))


def h12(sL, sP, unacc, b=0.2):
    from scipy.stats import binomtest
    qL, qP = queue(-sL, b)[unacc], queue(-sP, b)[unacc]
    lo, po = int((qL & ~qP).sum()), int((qP & ~qL).sum())
    p = float(binomtest(lo, lo + po, 0.5).pvalue) if lo + po else 1.0
    return dict(n_unacc=int(unacc.sum()), loom_only=lo, preval_only=po, both=int((qL & qP).sum()),
                caught_loom=float(qL.mean()), caught_preval=float(qP.mean()), p=p, passed=bool(p < 0.05 and lo > po))


def deep_scores():
    """E24 deep baselines on CinC, if available: {name: 1 - max probability}."""
    p = ROOT / "results" / "e24" / "cinc.npz"
    if not p.exists():
        return {}
    z = dict(np.load(p))
    return {m: 1 - z[f"P_{m}"].max(1) for m in DEEP if f"P_{m}" in z}


def rules(X):
    """Post-hoc, hand-made quality rules (exploratory, not registered): added after seed 0 of the registered
    analysis was seen, to ask whether a trivial rule explains the result. Higher = review first."""
    return {"Rule-amplitude": np.abs(X).max(axis=(1, 2)).astype(np.float64),     # largest |value| in mV
            "Rule-flat": (X.std(-1) < 0.01).sum(1).astype(np.float64)}         # number of flat leads


def exploratory(S, X, unacc):
    """AUROC of every score on all records and on the records without extreme amplitude (max |x| < T mV)."""
    amp = np.abs(X).max(axis=(1, 2)); out = {}
    for T in (5, 10, 20):
        m = amp < T
        out[f"amp_lt_{T}"] = {"n": int(m.sum()), "n_unacc": int(unacc[m].sum()),
                              **{k: auroc(v[m], unacc[m]) for k, v in S.items()}}
    return out


def analyze():
    runs = {s: dict(np.load(OUT / f"s{s}.npz")) for s in SEEDS if (OUT / f"s{s}.npz").exists()}
    o0 = runs[0]; unacc = o0["unacc"].astype(bool)
    per = {s: {m: describe(v, unacc) for m, v in scores(o).items()} for s, o in runs.items()}
    names = list(per[0])
    mean = {m: {k: float(np.mean([per[s][m][k] for s in per])) for k in per[0][m]} for m in names}
    rng_ = {m: {k: [float(min(per[s][m][k] for s in per)), float(max(per[s][m][k] for s in per))] for k in per[0][m]}
            for m in names}
    S0 = scores(o0)
    summ = {"experiment": "E23 CinC 2011 set A (registered 3de341c)", "n_records": int(len(unacc)),
            "n_unacceptable": int(unacc.sum()), "n_seeds": len(runs), "seeds": sorted(runs),
            "H11_seed0": h11(S0["LOOM"], S0["PreVal"], unacc), "H12_seed0": h12(S0["LOOM"], S0["PreVal"], unacc),
            "H11_all_seeds": [h11(scores(o)["LOOM"], scores(o)["PreVal"], unacc) for o in runs.values()],
            "H12_all_seeds": [h12(scores(o)["LOOM"], scores(o)["PreVal"], unacc) for o in runs.values()],
            "per_seed": per, "mean_over_seeds": mean, "range_over_seeds": rng_}
    D = deep_scores()
    if D:
        summ["deep"] = {m: describe(v, unacc) for m, v in D.items()}
        summ["deep_vs_loom_seed0"] = {m: h11(S0["LOOM"], v, unacc) for m, v in D.items()}
    X = load()[0]; R = rules(X)
    summ["rules"] = {m: describe(v, unacc) for m, v in R.items()}
    summ["leverage_vs_loom_seed0"] = h11(S0["Leverage"], S0["LOOM"], unacc)
    summ["rule_vs_loom_seed0"] = h11(S0["LOOM"], R["Rule-amplitude"], unacc)
    allS = {k: v for k, v in S0.items()}; allS.update(D); allS.update(R)
    summ["exploratory_seed0"] = exploratory(allS, X, unacc)
    # same summary as the E22 realism forest plot (post hoc, D27/D29): LOOM - PreVal share caught at 20 %
    from e22_post_hoc_realism import boot_mean
    per_seed = [float(queue(-scores(o)["LOOM"], 0.2)[unacc].mean() - queue(-scores(o)["PreVal"], 0.2)[unacc].mean())
                for o in runs.values()]
    dl = queue(-S0["LOOM"], 0.2)[unacc].astype(float) - queue(-S0["PreVal"], 0.2)[unacc].astype(float)
    _, lo, hi = boot_mean(dl)
    summ["effect"] = dict(study="E23", unit="record (seed 0 CI)", n=int(unacc.sum()), diff=float(np.mean(per_seed)),
                          diff_seed0=float(dl.mean()), ci_low=lo, ci_high=hi, per_seed=per_seed, n_seeds=len(per_seed))
    summ["exploratory_all_seeds"] = {m: [exploratory({m: scores(o)[m]}, X, unacc)["amp_lt_10"][m] for o in runs.values()]
                                     for m in ("LOOM", "PreVal", "Leverage")}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print("H11", summ["H11_seed0"]); print("H12", summ["H12_seed0"])
    for m in names:
        print(f"  {m:16s} AUROC {mean[m]['detect_auroc']:.3f} caught@20 {mean[m]['caught@20']:.3f}")
    for m, d in {**summ.get("deep", {}), **summ["rules"]}.items():
        print(f"  {m:16s} AUROC {d['detect_auroc']:.3f} caught@20 {d['caught@20']:.3f}")
    print("amp<10:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summ["exploratory_seed0"]["amp_lt_10"].items()})
    return summ


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("run", "both"):
        Xc, unacc, rec = load()
        print("CinC set A:", Xc.shape, "unacceptable", int(unacc.sum()), flush=True)
        for s in SEEDS:
            run(s)
    if what in ("analyze", "both"):
        analyze()
