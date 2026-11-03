"""Fig. 7: validity. (a) target vs achieved coverage of LOOM's closed-form jackknife+ sets on clean data;
(b) achieved coverage at target 0.9 per condition for LAC and APS scores; (c) pooled reliability of LOOM and
PreVal on clean data and under the strongest faults. Means over datasets with 95 % bootstrap CIs (E9)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.statistics.tests import bootstrap_mean_ci                         # noqa: E402
from loom.visualization.plots import FAMILY_LABEL, INK2, PAGE_W, STYLE, _save, _style  # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402


def main():
    _style()
    raw = pd.read_csv(ROOT / "results" / "raw_results.csv", usecols=["dataset", "resample", "method", "cond", "sev", "coverage", "transform", "train_frac"])
    raw = raw[(raw["transform"] == "minirocket") & (raw.train_frac == 1.0) & raw.method.str.startswith("CP-")]
    fig, axs = plt.subplots(1, 3, figsize=(PAGE_W, 2.75), gridspec_kw=dict(width_ratios=[1, 1.6, 1.1]))
    # (a) target vs achieved
    ax = axs[0]
    for meth, lab, col, mk in (("CP-LOOM-jackknife+", "LOOM jackknife+, LAC", STYLE["LOOM"][0], "o"),):
        xs, ys, lo, hi = [], [], [], []
        for a, suf in ((0.2, "@0.2"), (0.1, ""), (0.05, "@0.05")):
            sub = raw[(raw.method == meth + suf) & (raw.cond == "clean")].groupby("dataset").coverage.mean()
            d = bootstrap_mean_ci(sub.to_numpy()); xs.append(1 - a); ys.append(d["mean"]); lo.append(d["lo"]); hi.append(d["hi"])
            q = np.quantile(sub, [0.1, 0.9]); ax.plot([1 - a, 1 - a], q, color=col, lw=4, alpha=0.2, solid_capstyle="butt")
        ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt=mk, color=col, capsize=2, label=lab)
    ax.plot([0.75, 1], [0.75, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0.76, 0.98); ax.set_ylim(0.70, 1.0); ax.set_xlabel("Target coverage"); ax.set_ylabel("Achieved coverage")
    ax.set_title("(a) Clean data, LOOM jackknife+ (LAC)")
    # (b) per condition at target 0.9
    ax = axs[1]
    conds = [("clean", 0)] + [(f, s) for f in FAMILY_LABEL for s in (1, 2, 3)]
    xpos = np.arange(len(conds))
    for k, (meth, lab, col, mk) in enumerate((("CP-LOOM-jackknife+", "LOOM, LAC", STYLE["LOOM"][0], "o"),
                                              ("CP-LOOM-jackknife+-APS", "LOOM, APS", "#56B4E9", "s"),
                                              ("CP-Split-TS-APS", "Split conformal, APS", STYLE["Holdout-TS"][0], "^"))):
        m, l, h = [], [], []
        for c, s in conds:
            sub = raw[(raw.method == meth) & (raw.cond == c) & (raw.sev == s)].groupby("dataset").coverage.mean()
            d = bootstrap_mean_ci(sub.to_numpy()); m.append(d["mean"]); l.append(d["lo"]); h.append(d["hi"])
        ax.errorbar(xpos + (k - 1) * 0.22, m, yerr=[np.subtract(m, l), np.subtract(h, m)], fmt=mk, color=col, ms=2.8, capsize=1, lw=0.8, label=lab)
    ax.axhline(0.9, color=INK2, lw=0.8, ls="--")
    code = {"noise": "N", "wander": "W", "missing": "D", "spikes": "S", "warp": "V", "shift": "M"}
    ax.set_xticks(xpos); ax.set_xticklabels(["C"] + [f"{code[f]}{s}" for f in FAMILY_LABEL for s in (1, 2, 3)], rotation=90, fontsize=6.5)
    ax.set_ylabel("Achieved coverage (target 0.9)"); ax.set_title("(b) By fault type and severity")
    ax.legend(loc="lower right", ncol=1, borderaxespad=0.2); ax.set_ylim(0.15, 1.0)
    # (c) pooled reliability
    ax = axs[2]
    from loom.evaluation.aggregate import load_reliability
    pooled = load_reliability(ROOT / "results" / "raw")
    for m in ("LOOM", "PreVal"):
        for kind, ls in (("clean", "-"), ("shifted-sev3", ":")):
            if (m, kind) not in pooled:
                continue
            cnt, sc, sa = pooled[(m, kind)]; ok = cnt >= 200
            ax.plot(sc[ok] / cnt[ok], sa[ok] / cnt[ok], color=STYLE[m][0], ls=ls, marker=STYLE[m][1], ms=2.5,
                    label=f"{m}, {'clean' if kind == 'clean' else 'strongest faults'}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("(c) Reliability (pooled)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.8)
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_validity")


if __name__ == "__main__":
    main()
