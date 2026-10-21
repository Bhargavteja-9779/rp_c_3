"""Reviewer-4 check: is the LogReg-CV baseline under-converged with max_iter=300, tol=1e-3?
Compares it with max_iter=3000, tol=1e-5 on 12 randomly chosen datasets (seed 2026, official
split, clean test set, n_train <= 2000) using the same MiniRocket features."""
import sys, json, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from loom.data.ucr import load_split, dataset_info
from loom.models.baselines import LogReg
from loom.evaluation.metrics import nll, align
from loom.training.runner import make_transform, _feats
info = dataset_info()
cands = sorted(d for d, v in info.items() if v["n_train"] <= 2000)
dsets = sorted(np.random.default_rng(2026).choice(cands, 12, replace=False).tolist())
out = []
for d in dsets:
    Xtr, ytr, Xte, yte = load_split(d, 0); cl = np.unique(np.r_[ytr, yte]); yi = np.searchsorted(cl, yte)
    tf = make_transform("minirocket", 0).fit(Xtr[:, None, :]); Ftr, Fte = _feats(tf, Xtr), _feats(tf, Xte)
    r = {"dataset": d}
    for tag, kw in (("default", dict(max_iter=300)), ("long", dict(max_iter=3000))):
        m = LogReg(**kw)
        if tag == "long":
            from sklearn.linear_model import LogisticRegression  # tighter tolerance via the pipeline param
        m.fit(Ftr, ytr) if tag == "default" else None
        if tag == "long":
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from loom.models.baselines import _safe_cv
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, tol=1e-5))
            m.m = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                               cv=StratifiedKFold(_safe_cv(ytr, 3), shuffle=True, random_state=0), scoring="neg_log_loss").fit(Ftr, ytr)
            m.classes_ = m.m.classes_
        P = align(m.predict_proba(Fte), m.classes_, cl)
        r[f"nll_{tag}"] = nll(P, yi); r[f"acc_{tag}"] = float((P.argmax(1) == yi).mean())
        r[f"C_{tag}"] = m.m.best_params_["logisticregression__C"]
        r[f"iters_{tag}"] = int(np.max(m.m.best_estimator_[-1].n_iter_))
    out.append(r); print(r, flush=True)
res = Path(__file__).resolve().parents[1] / "results" / "check_logreg_convergence.json"
json.dump(out, open(res, "w"), indent=1)
d = np.array([[x["nll_default"], x["nll_long"]] for x in out])
print("median NLL default %.4f long %.4f; max |diff| %.4f" % (np.median(d[:, 0]), np.median(d[:, 1]), np.abs(d[:, 0] - d[:, 1]).max()))
