"""Write data/raw/ucr_equal_length.csv: datasets whose official text files contain
equal-length series without missing values (114 datasets in the 2018 archive)."""
import csv, glob, os
from pathlib import Path
import numpy as np
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "Univariate_arff"
ok = []
for d in sorted(glob.glob(str(RAW / "*/"))):
    n = os.path.basename(d.rstrip("/")); tr = Path(d) / f"{n}_TRAIN.txt"; te = Path(d) / f"{n}_TEST.txt"
    if not tr.exists():
        continue
    try:
        A, B = np.loadtxt(tr), np.loadtxt(te)
    except ValueError:
        continue
    if not (np.isnan(A).any() or np.isnan(B).any()) and A.shape[1] == B.shape[1]:
        ok.append((n, A.shape[0], B.shape[0], A.shape[1] - 1, len(np.unique(A[:, 0]))))
with open(RAW.parent / "ucr_equal_length.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "n_train", "n_test", "length", "n_classes"]); w.writerows(ok)
print(len(ok), "datasets")
