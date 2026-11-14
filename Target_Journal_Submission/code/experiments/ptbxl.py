"""PTB-XL v1.0.3 loader (100 Hz, 12 leads) for E20 (Online Resource 2, section 12.1).

Download the records with scripts/download_ptbxl.sh. Signals are read with an own WFDB format-16
reader (16-bit little-endian, interleaved leads), checked against the initial value and checksum
stored in every header. Labels follow the registered rule: diagnostic superclass from
scp_statements.csv (likelihood > 0), records with exactly one superclass. A record is "noisy" if
any of baseline_drift, static_noise, burst_noise or electrodes_problems is annotated.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ptbxl"
CACHE = ROOT / "data" / "processed" / "ptbxl100.npz"
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]          # sorted; label index = position
NOISE_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def read_16(stem):
    """Return (12, 1000) signal in mV; raise if initial values or checksums disagree with the header."""
    lines = [l for l in (RAW / f"{stem}.hea").read_text().split("\n") if l.strip() and not l.startswith("#")]
    nsig, nsamp = int(lines[0].split()[1]), int(lines[0].split()[3])
    raw = np.fromfile(RAW / f"{stem}.dat", dtype="<i2")
    assert raw.size == nsig * nsamp, stem
    raw = raw.reshape(nsamp, nsig).T
    out = np.empty((nsig, nsamp), np.float32)
    for i, l in enumerate(lines[1:1 + nsig]):
        f = l.split()
        assert f[1] == "16", (stem, f[1])
        gain = float(f[2].split("(")[0].split("/")[0]); zero = int(f[4]); init, chk = int(f[5]), int(f[6])
        assert raw[i, 0] == init, (stem, i, "initial value")
        assert int(raw[i].astype(np.int64).sum()) % 65536 == chk % 65536, (stem, i, "checksum")
        out[i] = (raw[i] - zero) / gain
    return out


def metadata():
    db = pd.read_csv(RAW / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(RAW / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1]["diagnostic_class"]
    def supers(s):
        d = ast.literal_eval(s)
        return sorted({diag[k] for k, v in d.items() if k in diag.index and v > 0})
    db["supers"] = db["scp_codes"].map(supers)
    db = db[db["supers"].map(len) == 1].copy()
    db["label"] = db["supers"].map(lambda s: CLASSES.index(s[0]))
    db["noisy"] = db[NOISE_COLS].notna().any(axis=1) & (db[NOISE_COLS].fillna("").astype(str).apply(
        lambda c: c.str.strip() != "").any(axis=1))
    return db


def load():
    """Signals (n, 12, 1000) float32, labels, noisy flags, folds and ecg_ids; cached after the first call."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["X"], z["y"], z["noisy"], z["fold"], z["ecg_id"]
    db = metadata()
    X = np.stack([read_16(fn) for fn in db["filename_lr"]])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, y=db["label"].to_numpy(np.int64), noisy=db["noisy"].to_numpy(bool),
             fold=db["strat_fold"].to_numpy(np.int64), ecg_id=db.index.to_numpy(np.int64))
    return load()


if __name__ == "__main__":
    X, y, noisy, fold, ids = load()
    print(X.shape, np.bincount(y), "noisy", int(noisy.sum()), "folds", np.bincount(fold)[1:])
    for name, m in [("train", fold <= 8), ("test", fold >= 9)]:
        print(name, int(m.sum()), "noisy share", round(float(noisy[m].mean()), 3))
