"""E18 (post hoc, deviation D21): validation of the loom-triage command-line tool on plain files.

For each dataset, the official UCR text files are converted to plain CSV files in a fresh folder (no use
of the internal loader or caches), the installed command `loom-triage` is run as a subprocess
(fit / run / evaluate), and its output is compared with the research pipeline (LOOM readout on the same
MiniRocket features, seed 0): identical predictions and probabilities are expected.
Output: results/e18/summary.json
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw" / "Univariate_arff"
DATASETS = ["ECG200", "GunPoint", "ItalyPowerDemand", "TwoLeadECG", "Wafer"]
OUT = ROOT / "results" / "e18"


def cli():
    exe = shutil.which("loom-triage")
    return [exe] if exe else [sys.executable, "-m", "loom.cli"]


def run(name, work):
    tr = np.loadtxt(RAW / name / f"{name}_TRAIN.txt"); te = np.loadtxt(RAW / name / f"{name}_TEST.txt")
    np.savetxt(work / "train.csv", tr, delimiter=","); np.savetxt(work / "test.csv", te, delimiter=",")
    np.savetxt(work / "new.csv", te[:, 1:], delimiter=",")
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    env = {**os.environ, **env}
    c = cli()
    subprocess.run(c + ["fit", "--train", str(work / "train.csv"), "--model", str(work / "m.pkl")], check=True, env=env, capture_output=True)
    subprocess.run(c + ["run", "--model", str(work / "m.pkl"), "--input", str(work / "new.csv"), "--budget", "0.2",
                        "--out", str(work / "q.csv"), "--report", str(work / "r.html")], check=True, env=env, capture_output=True)
    ev = subprocess.run(c + ["evaluate", "--model", str(work / "m.pkl"), "--test", str(work / "test.csv")], check=True, env=env,
                        capture_output=True, text=True).stdout.strip()
    q = pd.read_csv(work / "q.csv")
    # reference: research pipeline on the same data
    from loom.models.loom import LOOM
    from loom.training.runner import _feats, make_transform
    Xtr, ytr, Xte, yte = tr[:, 1:], tr[:, 0], te[:, 1:], te[:, 0]
    tf = make_transform("minirocket", 0).fit(Xtr[:, None, :])
    m = LOOM("standard").fit(_feats(tf, Xtr), ytr)
    P = m.predict_proba(_feats(tf, Xte))
    same_pred = bool(np.array_equal(m.classes_[P.argmax(1)], q.prediction.to_numpy()))
    max_conf_diff = float(np.abs(P.max(1) - q.confidence.to_numpy()).max())
    return dict(dataset=name, n_test=int(len(q)), review=int((q.decision == "review").sum()),
                identical_predictions=same_pred, max_confidence_difference=max_conf_diff,
                accuracy=float((q.prediction.to_numpy() == yte).mean()), evaluate_output=ev,
                report_written=(work / "r.html").exists())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DATASETS:
        with tempfile.TemporaryDirectory() as t:
            rows.append(run(d, Path(t))); print(rows[-1], flush=True)
    res = dict(cli=" ".join(cli()), datasets=rows, all_identical=all(r["identical_predictions"] for r in rows),
               max_confidence_difference=max(r["max_confidence_difference"] for r in rows))
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "datasets"}))


if __name__ == "__main__":
    main()
