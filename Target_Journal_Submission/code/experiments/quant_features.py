"""E14: LOOM and PreVal readouts on QUANT features vs the QUANT classifier (Online Resource 2, 11.3).

Official split, datasets with <= 3,000 training series, clean data and the six severity-3 faults.
Results: results/quant_features.json (one record per dataset, method and condition).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import dataset_info, dataset_names, load_split        # noqa: E402
from loom.evaluation.metrics import align, all_metrics                   # noqa: E402
from loom.models.readout import ClosedFormRidge                          # noqa: E402
from loom.preprocessing.corruptions import FAMILIES, corrupt              # noqa: E402
from loom.training.runner import MAIN_METHODS, cond_seed, shift_auroc     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "quant_features"


def run(name):
    path = OUT / f"{name}.json"
    if path.exists():
        return json.load(open(path))
    from aeon.classification.interval_based import QUANTClassifier
    from aeon.transformations.collection.interval_based import QUANTTransformer
    Xtr, ytr, Xte, yte = load_split(name, 0)
    classes = np.unique(np.r_[ytr, yte]); yi = np.searchsorted(classes, yte)
    conds = [("clean", 0)] + [(f, 3) for f in FAMILIES]
    Xc = {c: Xte if c[0] == "clean" else corrupt(Xte, c[0], 3, cond_seed(name, 0, *c)) for c in conds}
    tf = QUANTTransformer().fit(Xtr[:, None, :], ytr)
    feat = lambda X: np.nan_to_num(np.asarray(tf.transform(X[:, None, :]), dtype=np.float64))
    Ftr = feat(Xtr); base = ClosedFormRidge().fit(Ftr, ytr)
    probs = {}
    for m in ("LOOM", "PreVal"):
        mod = MAIN_METHODS[m]().fit(Ftr, ytr, ridge=base)
        probs[f"{m}-QF"] = {c: align(mod.predict_proba(feat(Xc[c])), mod.classes_, classes) for c in conds}
    q = QUANTClassifier(random_state=0).fit(Xtr[:, None, :], ytr)
    probs["QUANT"] = {c: align(q.predict_proba(Xc[c][:, None, :]), q.classes_, classes) for c in conds}
    rows = []
    for m, Pd in probs.items():
        for c in conds:
            rec = dict(dataset=name, method=m, cond=c[0], sev=c[1], n_features=int(Ftr.shape[1]), **all_metrics(Pd[c], yi))
            if c[0] != "clean":
                rec["shift_auroc"] = shift_auroc(Pd[("clean", 0)], Pd[c])
            rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(path, "w"), default=float)
    return rows


def main():
    info = dataset_info()
    rows = []
    for d in dataset_names():
        if info[d]["n_train"] > 3000:
            continue
        try:
            rows += run(d)
        except Exception as e:  # noqa: BLE001
            print("FAILED", d, repr(e), flush=True)
            continue
        print(d, flush=True)
    json.dump(rows, open(ROOT / "results" / "quant_features.json", "w"), default=float)


if __name__ == "__main__":
    main()
