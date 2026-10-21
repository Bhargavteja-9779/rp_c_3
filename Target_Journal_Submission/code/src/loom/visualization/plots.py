"""Publication figures, generated only from machine-readable result files.

Colour: fixed categorical slots per method (validated reference palette, light
mode), plus distinct markers/line styles as secondary encoding so that figures
remain legible in greyscale print.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {   # method -> (colour, marker, linestyle)
    "LOOM": ("#2a78d6", "o", "-"), "PreVal": ("#eb6834", "s", "--"), "Holdout-TS": ("#1baf7a", "^", "-."),
    "Ridge-Platt-CV5": ("#eda100", "D", ":"), "PreVal-Ens5": ("#e87ba4", "v", "--"), "LOOM-Ens5": ("#6250d6", "P", "-"),
    "Arsenal": ("#008300", "X", "-."), "QUANT": ("#e34948", "*", ":"), "LogReg-CV": ("#52514e", "h", "--"),
    "Ridge-softmax": ("#9b9a95", ">", ":"), "Ridge-Isotonic-CV5": ("#b8860b", "<", "-."),
}
FAMILY_LABEL = {"noise": "Gaussian noise", "wander": "Baseline wander", "missing": "Missing segment",
                "spikes": "Impulse spikes", "warp": "Time warp", "shift": "Circular shift"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 300, "figure.dpi": 150, "lines.linewidth": 1.6, "lines.markersize": 4.5})


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight"); fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_method(out):
    fig, ax = plt.subplots(figsize=(7.0, 2.5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1)
    texts = ["Time series\n$x$",
             "Random-kernel\ntransform\n(MiniRocket)",
             "Standardise,\nthin SVD\n$\\Phi_c = U S V^\\top$",
             "Exact LOO pairs\n$(z_{-i},\\, h_{-i})$ for\nevery $\\lambda$",
             "Fit $\\lambda$ and $a$\nby LOO NLL\n(smoothed targets)"]
    w, gap, x0, y0, hgt = 1.7, 0.3, 0.15, 1.75, 1.15
    for k, t in enumerate(texts):
        x = x0 + k * (w + gap)
        ax.add_patch(plt.Rectangle((x, y0), w, hgt, fc="#eef4fc", ec="#2a78d6", lw=1.0))
        ax.text(x + w / 2, y0 + hgt / 2, t, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.3)
        if k:
            ax.annotate("", (x, y0 + hgt / 2), (x - gap, y0 + hgt / 2), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    bx, bw = 1.5, 7.0
    ax.add_patch(plt.Rectangle((bx, 0.1), bw, 1.15, fc="#fdf0ea", ec="#eb6834", lw=1.0))
    ax.text(bx + bw / 2, 0.675, "Test series: logits $z(x)$ and leverage $h(x) = \\tilde x^\\top A^{-1} \\tilde x$ from the same SVD\n"
            "$\\hat p(y\\mid x) = \\mathrm{softmax}\\left(a\\, z(x) / \\sqrt{h(x)/h_{\\mathrm{ref}}}\\right)$;"
            "  jackknife+ sets from the exact LOO models $f_{-i}(x)$",
            ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.6)
    xl = x0 + 4 * (w + gap) + w / 2
    ax.annotate("", (bx + bw, 0.675), (xl, y0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1, connectionstyle="angle,angleA=-90,angleB=0"))
    _save(fig, out, "fig1_method")


def fig_dataset_distribution(root, out):
    d = pd.read_csv(root / "data/raw/ucr_equal_length.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for ax, col, lab in zip(axs, ["n_train", "length", "n_classes"], ["Training series", "Series length", "Classes"]):
        ax.hist(np.log10(d[col]), bins=20, color="#2a78d6", edgecolor="white", linewidth=0.8)
        ax.set_xlabel(f"{lab} (log$_{{10}}$)"); ax.set_ylabel("Datasets" if col == "n_train" else "")
    fig.suptitle(f"{len(d)} UCR datasets", fontsize=8, color=INK2, y=1.02)
    _save(fig, out, "fig_datasets")


def fig_leverage(root, out):
    p = root / "tables/table_leverage_ratios.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    cols = [c for c in r.columns if c.startswith("h_")]
    med = r.groupby("dataset")[cols].median()
    fams = list(FAMILY_LABEL)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    pos, labels = [], []
    data = [med["h_clean0"].dropna().values]; pos.append(0); labels.append("clean")
    x = 1
    for f in fams:
        for s in (1, 2, 3):
            c = f"h_{f}{s}"
            if c in med:
                data.append(med[c].dropna().values); pos.append(x); labels.append(f"{s}")
                x += 1
        x += 0.6
    bp = ax.boxplot([np.log10(v) for v in data], positions=pos, widths=0.6, showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor="#eef4fc", edgecolor="#2a78d6")
    for k in ("whiskers", "caps"):
        for l in bp[k]:
            l.set(color=INK2)
    for l in bp["medians"]:
        l.set(color="#2a78d6", lw=1.6)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=6.5)
    x = 1
    for f in fams:
        ax.text(x + 1, ax.get_ylim()[1], FAMILY_LABEL[f], ha="center", va="bottom", fontsize=6.5, color=INK2)
        x += 3.6
    ax.set_ylabel("log$_{10}$ median $h(x)/h_{ref}$")
    _save(fig, out, "fig_leverage_ratio")


def fig_robustness(root, out, methods=("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "PreVal-Ens5", "QUANT"), metric="nll"):
    p = root / "tables/table_robustness_by_family.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    raw = pd.read_csv(root / "results/raw_results.csv", usecols=["dataset", "method", "cond", "sev", "resample", "transform", "train_frac", metric])
    raw = raw[raw["transform"].isin(["minirocket", "raw"]) & (raw.train_frac == 1.0)]
    clean = raw[raw.cond == "clean"].groupby(["dataset", "method"])[metric].mean().groupby("method").median()
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True)
    for ax, f in zip(axs.ravel(), FAMILY_LABEL):
        for m in methods:
            g = t[(t.cond == f) & (t.method == m)].sort_values("sev")
            if g.empty or m not in clean:
                continue
            c, mk, ls = STYLE[m]
            ax.plot([0] + g.sev.tolist(), [clean[m]] + g[metric].tolist(), color=c, marker=mk, ls=ls, label=m)
        ax.set_title(FAMILY_LABEL[f], fontsize=8, color=INK)
        ax.set_xticks([0, 1, 2, 3])
    for ax in axs[1]:
        ax.set_xlabel("Severity (0 = clean)")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"Median test {metric.upper()}")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout()
    _save(fig, out, f"fig_robustness_{metric}")


def fig_scatter(root, out):
    raw = pd.read_csv(root / "results/raw_results.csv", usecols=["dataset", "method", "cond", "resample", "transform", "train_frac", "nll"])
    raw = raw[(raw["transform"] == "minirocket") & (raw.train_frac == 1.0) & raw.method.isin(["LOOM", "PreVal"])]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, (nm, sel) in zip(axs, [("Clean test", raw.cond == "clean"), ("Mean over 18 corruptions", raw.cond != "clean")]):
        T = raw[sel].groupby(["dataset", "method", "resample"]).nll.mean().groupby(["dataset", "method"]).mean().unstack().dropna()
        better = T.LOOM < T.PreVal
        T = T.clip(lower=1e-3)       # near-zero NLLs (perfectly separated datasets) are drawn at 1e-3
        ax.scatter(T.PreVal[better], T.LOOM[better], s=14, color="#2a78d6", marker="o", label=f"LOOM better ({better.sum()})", edgecolor="white", linewidth=0.4)
        ax.scatter(T.PreVal[~better], T.LOOM[~better], s=14, color="#eb6834", marker="s", label=f"PreVal better ({(~better).sum()})", edgecolor="white", linewidth=0.4)
        lo, hi = np.nanmin(T.values) * 0.8, np.nanmax(T.values) * 1.2
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("PreVal NLL"); ax.set_ylabel("LOOM NLL"); ax.set_title(nm, fontsize=8); ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); _save(fig, out, "fig_scatter_loom_vs_preval")


def fig_ranks(root, out):
    st = json.load(open(root / "results/statistics.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, key, title in zip(axs, ["E1_clean", "E4_shifted"], ["Clean test NLL", "Corrupted test NLL (mean of 18)"]):
        if key not in st or "nll" not in st[key]:
            continue
        r = pd.Series(st[key]["nll"]["mean_ranks"]).sort_values()
        y = np.arange(len(r))[::-1]
        ax.scatter(r.values, y, color=[STYLE.get(m, ("#52514e",))[0] for m in r.index], s=22, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(r.index, fontsize=7)
        cd = st[key]["nll"]["nemenyi_cd"]
        ax.plot([r.values[0], r.values[0] + cd], [len(r) - 0.3] * 2, color=INK, lw=2)
        ax.text(r.values[0] + cd / 2, len(r) - 0.15, f"CD = {cd:.2f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xlabel("Mean rank (lower is better)"); ax.set_title(f"{title}, n = {st[key]['nll']['n_datasets']}", fontsize=8)
    fig.tight_layout(); _save(fig, out, "fig_mean_ranks")


def fig_ablation(root, out):
    st = json.load(open(root / "results/statistics.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, key, title in zip(axs, ["E3_ablation_clean", "E3_ablation_shifted"], ["Clean", "Corrupted (mean of 18)"]):
        if key not in st or "nll" not in st[key]:
            continue
        pw = pd.DataFrame(st[key]["nll"]["pairwise"]).set_index("competitor")
        pw = pw.reindex(["PreVal", "PreVal-smooth", "LOOM-nosmooth", "LOOM-free-b", "LOOM-finite-c1", "LOOM-kNN",
                         "LOOM-insample", "LOOM-holdout", "LOOM-lambdaSSE"]).dropna(how="all")
        y = np.arange(len(pw))[::-1]
        ax.errorbar(pw.median_diff, y, xerr=[pw.median_diff - pw.ci_low, pw.ci_high - pw.median_diff], fmt="o",
                    color="#2a78d6", ecolor=INK2, elinewidth=1, capsize=2, ms=4)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(pw.index, fontsize=7)
        ax.set_xlabel("Median ΔNLL (variant − LOOM), 95% CI"); ax.set_title(title, fontsize=8)
        ax.set_xscale("symlog", linthresh=0.01)
        lo, hi = float(np.nanmin(pw.ci_low)), float(np.nanmax(pw.ci_high))
        ax.set_xlim(min(lo * 1.6, -0.02), max(hi * 1.6, 0.02))      # keep every interval inside the panel
    fig.tight_layout(); _save(fig, out, "fig_ablation")


def fig_conformal(root, out):
    raw = pd.read_csv(root / "results/raw_results.csv", usecols=["dataset", "method", "cond", "transform", "train_frac", "coverage", "set_size", "n_classes"])
    raw = raw[(raw["transform"] == "minirocket") & (raw.train_frac == 1.0) & raw.method.isin(["CP-LOOM-jackknife+", "CP-PreVal-jackknife+", "CP-Split-TS", "CP-LOOM-jackknife+-APS", "CP-Split-TS-APS"])]
    if raw.empty:
        return
    raw["kind"] = np.where(raw.cond == "clean", "Clean", "Corrupted")
    d = raw.groupby(["dataset", "method", "kind"]).agg(cov=("coverage", "mean"), size=("set_size", "mean"), K=("n_classes", "first")).reset_index()
    d["rel_size"] = d["size"] / d["K"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    meths = ["CP-LOOM-jackknife+", "CP-PreVal-jackknife+", "CP-Split-TS", "CP-LOOM-jackknife+-APS", "CP-Split-TS-APS"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]
    for ax, val, lab in zip(axs, ["cov", "rel_size"], ["Empirical coverage", "Mean set size / number of classes"]):
        pos = 0; ticks = []
        for kind in ["Clean", "Corrupted"]:
            for m, c in zip(meths, cols):
                v = d[(d.method == m) & (d.kind == kind)][val].dropna().values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, patch_artist=True)
                bp["boxes"][0].set(facecolor="white", edgecolor=c, lw=1.2); bp["medians"][0].set(color=c, lw=1.6)
                pos += 1
            ticks.append((pos - 3, kind)); pos += 0.8
        if val == "cov":
            ax.axhline(0.9, color=INK2, ls="--", lw=0.8)
        ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks]); ax.set_ylabel(lab)
    from matplotlib.patches import Patch
    fig.legend([Patch(fc="white", ec=c) for c in cols], ["LOOM jackknife+ (LAC)", "PreVal jackknife+ (LAC)", "Split TS (LAC)",
               "LOOM jackknife+ (APS)", "Split TS (APS)"], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout(); _save(fig, out, "fig_conformal")


def fig_efficiency(root, out):
    p = root / "results/timing.csv"
    if not p.exists():
        return
    tm = pd.read_csv(p)
    tm = tm[(tm["transform"].isin(["minirocket", "raw"])) & (tm.train_frac == 1.0)]
    g = tm.groupby(["dataset", "method"]).agg(fit=("fit", "mean"), n=("n_train", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for m in ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "LogReg-CV", "LOOM-Ens5", "Arsenal", "QUANT"]:
        v = g[g.method == m].sort_values("n")
        if v.empty:
            continue
        c, mk, _ = STYLE[m]
        ax.scatter(v.n, v.fit.clip(lower=1e-3), s=8, color=c, marker=mk, label=m, alpha=0.8, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Training series"); ax.set_ylabel("Fit time (s, single core)")
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, markerscale=1.5)
    fig.tight_layout(); _save(fig, out, "fig_efficiency")


MIN_BIN = 200   # minimum pooled test predictions for a reliability bin to be drawn


def fig_reliability(root, out):
    from ..evaluation.aggregate import load_reliability
    pooled = load_reliability(root / "results/raw")
    if not pooled:
        return
    meths = [m for m in ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5"] if (m, "clean") in pooled]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, kind, title in zip(axs, ["clean", "shifted-sev3"], ["Clean test (pooled)", "Severity-3 corruptions (pooled)"]):
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.8)
        for m in meths:
            if (m, kind) not in pooled:
                continue
            cnt, sc, sa = pooled[(m, kind)]
            c, mk, ls = STYLE[m]
            e = (np.abs(sa[cnt > 0] - sc[cnt > 0])).sum() / cnt.sum()     # ECE over all bins
            ok = cnt >= MIN_BIN                                           # plot only bins with enough support
            ax.plot(sc[ok] / cnt[ok], sa[ok] / cnt[ok], color=c, marker=mk, ls=ls, label=f"{m} (pooled ECE {e:.3f})")
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_title(title, fontsize=8); ax.legend(fontsize=6.5, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); _save(fig, out, "fig_reliability")


def fig_error(root, out):
    p = root / "tables/table_error_analysis_datasets.csv"
    if not p.exists():
        return
    e = pd.read_csv(p, index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, x, lab in zip(axs, ["n_train", "lev_ratio_shift", "acc_shift"], ["Training series", "Median shifted leverage ratio", "LOOM accuracy under corruption"]):
        if x not in e:
            continue
        ax.scatter(e[x], e.gain_shift, s=9, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
        if x != "acc_shift":
            ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel(lab)
    axs[0].set_ylabel("NLL gain under corruption\n(PreVal − LOOM)")
    fig.tight_layout(); _save(fig, out, "fig_error_analysis")


def make_all(root):
    root = Path(root); out = root / "figures"; out.mkdir(exist_ok=True)
    _style()
    for f in (lambda: fig_method(out), lambda: fig_dataset_distribution(root, out), lambda: fig_leverage(root, out),
              lambda: fig_robustness(root, out, metric="nll"), lambda: fig_robustness(root, out, metric="ece"),
              lambda: fig_scatter(root, out), lambda: fig_ranks(root, out), lambda: fig_ablation(root, out),
              lambda: fig_conformal(root, out), lambda: fig_efficiency(root, out), lambda: fig_reliability(root, out),
              lambda: fig_error(root, out)):
        try:
            f()
        except Exception as ex:   # a missing experiment must not stop the other figures
            import traceback
            print("figure failed:", traceback.format_exc()[-400:])
