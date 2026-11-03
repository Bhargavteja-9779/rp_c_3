"""E12: real recorded sensor noise from the MIT-BIH Noise Stress Test Database (Online Resource 2, 11.1).

Noise records bw (baseline wander), em (electrode motion) and ma (muscle artefact) are added to
the test series of the seven UCR ECG datasets at 12, 6 and 0 dB SNR. Download with
scripts/download_nstdb.sh. Results: results/real_noise.json and results/real_noise/*.npz.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import load_split                                    # noqa: E402
from loom.evaluation.metrics import align, all_metrics                   # noqa: E402
from loom.models.readout import ClosedFormRidge                          # noqa: E402
from loom.training.runner import MAIN_METHODS, _feats, cond_seed, make_transform, shift_auroc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NSTDB = ROOT / "data" / "raw" / "nstdb"
OUT = ROOT / "results" / "real_noise"
DATASETS = ["ECG200", "ECG5000", "ECGFiveDays", "TwoLeadECG", "CinCECGTorso",
            "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2"]
NOISES, SNRS, RESAMPLES = ["bw", "em", "ma"], [12, 6, 0], range(5)
READOUTS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5"]
P_FAULT = 0.3


def read_212(name):
    """Read a two-channel WFDB format-212 record (12-bit two's complement, two samples per 3 bytes)."""
    hdr = (NSTDB / f"{name}.hea").read_text().split("\n")
    nsig, nsamp = int(hdr[0].split()[1]), int(hdr[0].split()[3])
    b = np.fromfile(NSTDB / f"{name}.dat", dtype=np.uint8)[: 3 * ((nsig * nsamp + 1) // 2)].reshape(-1, 3).astype(np.int32)
    s1 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s2 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    v = np.empty(2 * len(b), np.int32); v[0::2], v[1::2] = s1, s2
    v[v > 2047] -= 4096
    return v[: nsig * nsamp].reshape(nsamp, nsig).astype(np.float64).T     # (channels, samples)


def add_noise(X, noise, snr_db, seed):
    """Add a random window of recorded noise to each series, scaled to the given SNR (per series)."""
    rng = np.random.default_rng(seed)
    n, L = X.shape
    ch = rng.integers(0, noise.shape[0], n); off = rng.integers(0, noise.shape[1] - L, n)
    W = np.stack([noise[c, o:o + L] for c, o in zip(ch, off)])
    W = W - W.mean(1, keepdims=True)
    ps = X.var(1, keepdims=True); pn = W.var(1, keepdims=True) + 1e-12
    return X + W * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def build_stream(name, r, Xte, noise):
    """Noisy copies of the test set for every condition, and the mixed stream (30 % noisy recordings)."""
    conds = [(nz, s) for nz in NOISES for s in SNRS]
    Xc = {c: add_noise(Xte, noise[c[0]], c[1], cond_seed(name, r, f"nstdb-{c[0]}", c[1])) for c in conds}
    rng = np.random.default_rng(cond_seed(name, r, "nstdb-stream", 0))
    is_f = rng.random(len(Xte)) < P_FAULT; pick = rng.integers(0, len(conds), len(Xte))
    Xs = Xte.copy()
    for k, c in enumerate(conds):
        idx = is_f & (pick == k); Xs[idx] = Xc[c][idx]
    return conds, Xc, Xs, is_f, pick


def run(name, r, noise):
    path = OUT / f"{name}__r{r}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    from aeon.classification.interval_based import QUANTClassifier
    Xtr, ytr, Xte, yte = load_split(name, r)
    classes = np.unique(np.r_[ytr, yte]); yi = np.searchsorted(classes, yte)
    conds, Xc, Xs, is_f, _ = build_stream(name, r, Xte, noise)
    tf = make_transform("minirocket", 1000 * r).fit(Xtr[:, None, :])
    Ftr = _feats(tf, Xtr); base = ClosedFormRidge().fit(Ftr, ytr)
    out = {"y": yi, "is_fault": is_f}
    feats = {"clean": _feats(tf, Xte), "stream": _feats(tf, Xs), **{f"{c[0]}{c[1]}": _feats(tf, Xc[c]) for c in conds}}
    for m in READOUTS:
        try:
            mod = MAIN_METHODS[m]().fit(Ftr, ytr, ridge=base)
            for k, F in feats.items():
                out[f"P_{k}_{m}"] = align(mod.predict_proba(F), mod.classes_, classes)
        except Exception:                                   # e.g. CV calibration with singleton classes
            pass
    q = QUANTClassifier(random_state=r).fit(Xtr[:, None, :], ytr)
    raw = {"clean": Xte, "stream": Xs, **{f"{c[0]}{c[1]}": Xc[c] for c in conds}}
    for k, X in raw.items():
        out[f"P_{k}_QUANT"] = align(q.predict_proba(X[:, None, :]), q.classes_, classes)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def main():
    noise = {n: read_212(n) for n in NOISES}
    from case_study import triage
    rows = []
    for d in DATASETS:
        for r in RESAMPLES:
            o = run(d, r, noise); y = o["y"]; f = o["is_fault"].astype(bool)
            for m in READOUTS + ["QUANT"]:
                if f"P_clean_{m}" not in o:
                    continue
                Pc = o[f"P_clean_{m}"]
                rows.append(dict(dataset=d, resample=r, method=m, cond="clean", snr=None, **all_metrics(Pc, y)))
                for nz in NOISES:
                    for s in SNRS:
                        P = o[f"P_{nz}{s}_{m}"]
                        rows.append(dict(dataset=d, resample=r, method=m, cond=nz, snr=s, **all_metrics(P, y),
                                         shift_auroc=shift_auroc(Pc, P)))
                P = o[f"P_stream_{m}"]; e, c = triage(P.max(1), P.argmax(1) == y, f, 0.2)
                rows.append(dict(dataset=d, resample=r, method=m, cond="stream", snr=None, **all_metrics(P, y),
                                 **{"auto_err@20": e, "fault_caught@20": c}))
            print(d, r, flush=True)
    json.dump(rows, open(ROOT / "results" / "real_noise.json", "w"), indent=0, default=float)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
