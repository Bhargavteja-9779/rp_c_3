"""Dev-set pilot 2: relative-leverage moderation b = c / median(h_loo); choose c on DEV only."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_dev import load, corrupt, ece, RAW
from aeon.transformations.collection.convolution_based import MiniRocket
from loom.models.readout import ClosedFormRidge
from loom.models.calibration import fit_moderation, moderated_log_proba
dev = json.load(open(Path(__file__).resolve().parents[1] / "results/dev_pilot.json"))["dev"]
CS = [0, 1, 10, 30, 100, 300, 1000, 10000, "free"]
rows = {}
for nm in dev:
    Xtr, ytr, Xte, yte = load(nm); cl = np.unique(ytr); yi = np.searchsorted(cl, ytr); yt = np.searchsorted(cl, yte)
    tf = MiniRocket(random_state=0).fit(Xtr[:, None, :]); m = ClosedFormRidge().fit(tf.transform(Xtr[:, None, :]), ytr)
    fits = {}
    for c in CS:
        cand = []
        for lam in m.lambdas:
            m.set_lambda(lam); href = np.median(m.loo_leverage_)
            if c == "free": a, b, nll = fit_moderation(m.loo_logits_, m.loo_leverage_, yi, fit_b=True)
            else: a, b, nll = fit_moderation(m.loo_logits_, m.loo_leverage_, yi, fit_b=False, b_fixed=c / href)
            cand.append((nll, lam, a, b))
        fits[c] = min(cand)
    rng = np.random.default_rng(1234)
    for kind in ["clean", "noise", "missing", "wander"]:
        F = tf.transform((Xte if kind == "clean" else corrupt(Xte, kind, rng))[:, None, :])
        for c in CS:
            _, lam, a, b = fits[c]; m.set_lambda(lam); Z, h, _ = m.logits_and_leverage(F)
            LP = moderated_log_proba(Z, h, a, b)
            rows[(nm, kind, str(c))] = (-LP[np.arange(len(yt)), yt].mean(), ece(np.exp(LP), yt), (LP.argmax(1) == yt).mean())
import pandas as pd
df = pd.DataFrame([dict(ds=k[0], cond=k[1], c=k[2], nll=v[0], ece=v[1], acc=v[2]) for k, v in rows.items()])
df["rank"] = df.groupby(["ds", "cond"])["nll"].rank()
print(df.pivot_table(index="c", columns="cond", values="nll", aggfunc="median").round(3))
print(df.pivot_table(index="c", columns="cond", values="ece", aggfunc="median").round(3))
print(df.pivot_table(index="c", columns="cond", values="acc", aggfunc="mean").round(3))
print("mean NLL rank over all dev conditions:\n", df.groupby("c")["rank"].mean().sort_values())
print("mean NLL rank clean only:\n", df[df.cond == "clean"].groupby("c")["rank"].mean().sort_values())
df.to_csv(Path(__file__).resolve().parents[1] / "results/dev_pilot_c2.csv", index=False)
