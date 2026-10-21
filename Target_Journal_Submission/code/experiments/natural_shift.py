"""Natural sensor shift (reviewer 5): train on one accelerometer axis of UWaveGestureLibrary and
test on another axis. X/Y/Z contain the same recordings in the same order with identical labels, so
the same resample seed gives the same recordings in train and test for every axis (no recording
leakage). 6 ordered axis pairs x 5 resamples; in-domain test = same axis."""
import sys, json, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import load_split
from loom.models.loom import LOOM
from loom.models.baselines import RidgeSoftmax, RidgeCalibratedCV, LogReg
from loom.models.readout import ClosedFormRidge
from loom.evaluation.metrics import all_metrics, align
from loom.training.runner import make_transform, _feats, shift_auroc
OUT = Path(__file__).resolve().parents[1] / "results" / "natural_shift.json"
done = json.load(open(OUT)) if OUT.exists() else []
key = lambda r: (r["source"], r["target"], r["resample"])
have = {key(r) for r in done}
M = {"LOOM": lambda: LOOM("standard"), "PreVal": lambda: LOOM("none", smooth=False),
     "Holdout-TS": lambda: LOOM("none", pairs="holdout", smooth=False), "Ridge-softmax": RidgeSoftmax,
     "Ridge-Platt-CV5": lambda: RidgeCalibratedCV("sigmoid", 5), "LogReg-CV": LogReg}
for r in range(5):
    for src in "XYZ":
        Xtr, ytr, Xte, yte = load_split("UWaveGestureLibrary" + src, r)
        tf = make_transform("minirocket", 1000 * r).fit(Xtr[:, None, :]); Ftr = _feats(tf, Xtr)
        base = ClosedFormRidge().fit(Ftr, ytr)
        models = {m: f().fit(Ftr, ytr, ridge=base) for m, f in M.items()}
        cl = np.unique(np.r_[ytr, yte]); Fin = _feats(tf, Xte)
        Pin = {m: align(md.predict_proba(Fin), md.classes_, cl) for m, md in models.items()}
        for tgt in "XYZ":
            if tgt == src or (src, tgt, r) in have:
                continue
            _, _, Xb, yb = load_split("UWaveGestureLibrary" + tgt, r)
            assert np.array_equal(yb, yte)
            Fb = _feats(tf, Xb); yi = np.searchsorted(cl, yb)
            hb = models["LOOM"].ridge_.leverage(Fb); hin = models["LOOM"].ridge_.leverage(Fin)
            for m, md in models.items():
                Pb = align(md.predict_proba(Fb), md.classes_, cl)
                done.append(dict(source=src, target=tgt, resample=r, method=m,
                                 indomain=all_metrics(Pin[m], np.searchsorted(cl, yte)), shifted=all_metrics(Pb, yi),
                                 shift_auroc=shift_auroc(Pin[m], Pb), lev_ratio=float(np.median(hb) / np.median(hin))))
            json.dump(done, open(OUT, "w"))
            print(src, "->", tgt, "r", r, {m: round(x["shifted"]["nll"], 3) for x in done[-len(M):] for m in [x["method"]]}, flush=True)
