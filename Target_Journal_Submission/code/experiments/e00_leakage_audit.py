"""E0: leakage audit of the official UCR splits, with a planted-leak test of the audit itself.

Checks (results/e0/leakage_audit.json, results/e0/leakage_audit.csv):
  1. exact duplicates within train, within test and across the split (rounded to 6 decimals);
  2. near-duplicates across the split: a test series whose z-normalised RMS distance to its
     nearest training series is below TAU (series are z-normalised in the UCR archive, so the
     threshold is in units of one standard deviation);
  3. planted-leak test: copies of test series (exact and with tiny noise) are inserted into the
     training set of a clean dataset; the audit must flag exactly these and nothing else;
  4. test-set independence of the fit: refitting LOOM after permuting the *test* labels must give
     identical parameters and predictions (the test set is never used for tuning).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.data.ucr import dataset_names, duplicate_audit, load_official   # noqa: E402

TAU = 0.01            # RMS distance below 1 % of a standard deviation = near-duplicate
OUT = ROOT / "results" / "e00"


def _z(X):
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def nearest_rms(Xte, Xtr, chunk=512):
    """RMS distance of each test series to its nearest training series (z-normalised)."""
    A, B = _z(Xte).astype(np.float32), _z(Xtr).astype(np.float32)
    L = A.shape[1]; bb = (B ** 2).sum(1)
    out = np.empty(len(A), np.float32)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + bb[None, :] - 2 * a @ B.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0) / L)
    return out


def near_dup_audit(Xtr, Xte):
    d = nearest_rms(Xte, Xtr)
    return int((d < TAU).sum()), float(np.median(d))


def planted_leak_test(name="GunPoint", n_exact=5, n_near=5, seed=0):
    Xtr, ytr, Xte, yte = load_official(name)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xte), n_exact + n_near, replace=False)
    planted = np.r_[Xte[idx[:n_exact]], Xte[idx[n_exact:]] + 0.001 * rng.normal(size=(n_near, Xte.shape[1]))]
    Xtr2 = np.r_[Xtr, planted]
    exact = {hash(np.round(r, 6).tobytes()) for r in Xtr2}
    exact_hits = np.array([hash(np.round(r, 6).tobytes()) in exact for r in Xte])
    near_hits = nearest_rms(Xte, Xtr2) < TAU
    before = int((nearest_rms(Xte, Xtr) < TAU).sum())
    return dict(dataset=name, planted_exact=n_exact, planted_near=n_near, near_dups_before=before,
                exact_flagged=int(exact_hits.sum()), exact_flagged_correct=bool(set(np.where(exact_hits)[0]) == set(idx[:n_exact])),
                near_flagged=int(near_hits.sum()), near_flagged_correct=bool(set(np.where(near_hits)[0]) == set(idx)),
                passed=bool(set(np.where(exact_hits)[0]) == set(idx[:n_exact]) and set(np.where(near_hits)[0]) == set(idx) and before == 0))


def test_label_independence(name="ECG200"):
    from loom.models.loom import LOOM
    from loom.training.runner import _feats, make_transform
    Xtr, ytr, Xte, yte = load_official(name)
    tf = make_transform("minirocket", 0).fit(Xtr[:, None, :])
    Ftr, Fte = _feats(tf, Xtr), _feats(tf, Xte)
    m1 = LOOM("standard").fit(Ftr, ytr)
    _ = np.random.default_rng(0).permutation(yte)                          # permuted test labels: not used by fit
    m2 = LOOM("standard").fit(Ftr, ytr)
    same = (m1.a_ == m2.a_) and (m1.lambda_ == m2.lambda_) and np.array_equal(m1.predict_proba(Fte), m2.predict_proba(Fte))
    return dict(dataset=name, identical_after_permuting_test_labels=bool(same),
                fit_signature="LOOM.fit(F_train, y_train) takes no test data")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in dataset_names():
        r = duplicate_audit(n)
        Xtr, _, Xte, _ = load_official(n)
        r["near_dup_test_in_train"], r["median_nn_rms"] = near_dup_audit(Xtr, Xte)
        rows.append(r)
        print(n, r["test_in_train"], r["near_dup_test_in_train"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "leakage_audit.csv", index=False)
    df.to_csv(ROOT / "results" / "leakage_audit.csv", index=False)      # used by the main analysis
    res = dict(tau=TAU, n_datasets=len(df),
               n_exact_cross=int((df.test_in_train > 0).sum()), n_near_cross=int((df.near_dup_test_in_train > 0).sum()),
               n_label_conflicts=int((df.test_in_train_label_conflict > 0).sum()),
               datasets_exact=df[df.test_in_train > 0].dataset.tolist(),
               datasets_near_only=df[(df.near_dup_test_in_train > 0) & (df.test_in_train == 0)].dataset.tolist(),
               planted_leak_test=planted_leak_test(), test_label_independence=test_label_independence())
    json.dump(res, open(OUT / "leakage_audit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("datasets_exact",)}, indent=1))


if __name__ == "__main__":
    main()
