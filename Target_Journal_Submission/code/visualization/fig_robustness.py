"""Fig. 5: median NLL over datasets against fault severity for each fault type (E4), with 95 % bootstrap
confidence bands of the median (1,000 resamples over datasets)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.visualization.plots import FAMILY_LABEL, INK, PAGE_W, STYLE, _save, _style   # noqa: E402
import matplotlib.pyplot as plt                                                       # noqa: E402

METHODS = ("LOOM", "PreVal", "Ridge-Platt-CV5", "QUANT")


def med_ci(x, rng, n=1000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return np.median(x), np.quantile(b, 0.025), np.quantile(b, 0.975)


def main():
    _style(); rng = np.random.default_rng(0)
    raw = pd.read_csv(ROOT / "results" / "raw_results.csv", usecols=["dataset", "method", "cond", "sev", "resample", "transform", "train_frac", "nll"])
    raw = raw[raw["transform"].isin(["minirocket", "raw"]) & (raw.train_frac == 1.0) & raw.method.isin(METHODS)]
    per = raw.groupby(["dataset", "method", "cond", "sev"]).nll.mean().reset_index()
    fig, axs = plt.subplots(2, 3, figsize=(PAGE_W, 3.6), sharex=True)
    for ax, fam in zip(axs.ravel(), FAMILY_LABEL):
        for m in METHODS:
            pts = [med_ci(per[(per.method == m) & (per.cond == "clean")].nll, rng)]
            pts += [med_ci(per[(per.method == m) & (per.cond == fam) & (per.sev == s)].nll, rng) for s in (1, 2, 3)]
            md, lo, hi = map(np.array, zip(*pts)); c, mk, ls = STYLE[m]
            ax.fill_between(range(4), lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(range(4), md, color=c, marker=mk, ls=ls, label={"Ridge-Platt-CV5": "Platt (5-fold)"}.get(m, m))
        ax.set_title(FAMILY_LABEL[fam], color=INK); ax.set_xticks([0, 1, 2, 3])
    for ax in axs[1]:
        ax.set_xlabel("Severity (0 = clean)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Median NLL")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_robustness")


if __name__ == "__main__":
    main()
