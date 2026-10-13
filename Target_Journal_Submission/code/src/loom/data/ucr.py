"""UCR archive loading, deterministic resampling and leakage auditing."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "Univariate_arff"
LIST = ROOT / "data" / "raw" / "ucr_equal_length.csv"
# datasets used to design the method (chosen at random, seed 0, before any other
# dataset was evaluated); results are reported with and without them.
DEV_DATASETS = ["Adiac", "Beef", "Car", "EthanolLevel", "Fish", "Lightning2",
                "NonInvasiveFetalECGThorax2", "SonyAIBORobotSurface1"]


def dataset_names():
    return [r["dataset"] for r in csv.DictReader(open(LIST))]


def dataset_info():
    return {r["dataset"]: {k: int(v) for k, v in r.items() if k != "dataset"} for r in csv.DictReader(open(LIST))}


@lru_cache(maxsize=4)
def load_official(name: str):
    tr = np.loadtxt(RAW / name / f"{name}_TRAIN.txt")
    te = np.loadtxt(RAW / name / f"{name}_TEST.txt")
    return tr[:, 1:].astype(np.float64), tr[:, 0], te[:, 1:].astype(np.float64), te[:, 0]


def load_split(name: str, resample: int):
    """resample 0 = official split; r > 0 = stratified resample of the pooled data
    with the official train size and class proportions (seed r), the protocol of
    the UCR bake-offs."""
    Xtr, ytr, Xte, yte = load_official(name)
    if resample == 0:
        return Xtr, ytr, Xte, yte
    X = np.vstack([Xtr, Xte]); y = np.concatenate([ytr, yte])
    rng = np.random.default_rng(resample)
    tr_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = int(np.sum(ytr == c))                     # keep official per-class train counts
        tr_idx.append(rng.choice(idx, size=k, replace=False))
    tr_idx = np.sort(np.concatenate(tr_idx))
    mask = np.zeros(len(y), bool); mask[tr_idx] = True
    return X[mask], y[mask], X[~mask], y[~mask]


def duplicate_audit(name: str, decimals: int = 6):
    """Exact duplicates (after rounding) inside train, inside test and across the
    official train/test split; cross-split duplicates are a leakage risk."""
    Xtr, ytr, Xte, yte = load_official(name)
    key = lambda X: [hash(np.round(r, decimals).tobytes()) for r in X]
    ktr, kte = key(Xtr), key(Xte)
    str_, ste = set(ktr), set(kte)
    cross = sum(k in str_ for k in kte)
    lab_conflict = 0
    lab = {}
    for k, yy in zip(ktr, ytr):
        lab.setdefault(k, set()).add(yy)
    for k, yy in zip(kte, yte):
        if k in lab and yy not in lab[k]:
            lab_conflict += 1
    return dict(dataset=name, n_train=len(ktr), n_test=len(kte), dup_within_train=len(ktr) - len(str_),
                dup_within_test=len(kte) - len(ste), test_in_train=cross, test_in_train_label_conflict=lab_conflict)
