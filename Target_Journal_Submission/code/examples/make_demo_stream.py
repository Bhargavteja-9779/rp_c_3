"""Build the demo files for ecg_triage_demo.sh and score the resulting review queue."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ECG5000"); ap.add_argument("--out"); ap.add_argument("--score")
a = ap.parse_args()
if a.score:
    out = Path(a.score)
    q = pd.read_csv(out / "queue.csv"); t = pd.read_csv(out / "truth.csv")
    rev = q.decision.eq("review").to_numpy(); noisy = t.noisy.to_numpy().astype(bool)
    ok = q.prediction.to_numpy() == t.label.to_numpy()
    print(f"noisy recordings caught by the review queue: {rev[noisy].mean():.0%} (random: {rev.mean():.0%})")
    print(f"error rate of automatic decisions: {(~ok[~rev]).mean():.1%}  (all decisions: {(~ok).mean():.1%})")
    sys.exit(0)
from loom.data.ucr import load_split            # noqa: E402
from real_noise import add_noise, read_212       # noqa: E402
Xtr, ytr, Xte, yte = load_split(a.dataset, 0)
rng = np.random.default_rng(0)
noisy = rng.random(len(Xte)) < 0.3
noise = read_212("em")                           # real electrode-motion artefact
Xs = Xte.copy(); Xs[noisy] = add_noise(Xte[noisy], noise, 6, seed=1)
out = Path(a.out)
np.savetxt(out / "train.csv", np.c_[ytr, Xtr], delimiter=",", fmt="%.6g")
np.savetxt(out / "stream.csv", Xs, delimiter=",", fmt="%.6g")
pd.DataFrame({"label": yte, "noisy": noisy.astype(int)}).to_csv(out / "truth.csv", index=False)
print(f"{a.dataset}: {len(ytr)} training recordings; stream of {len(Xs)} recordings, {noisy.mean():.0%} with electrode-motion noise at 6 dB")
