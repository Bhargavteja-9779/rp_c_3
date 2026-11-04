"""Aggregate cached job results -> raw_results.csv, metrics.json, statistics.json,
tables (CSV + Markdown) and figures.  Every number in the manuscript comes from here."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.data.ucr import DEV_DATASETS, dataset_info  # noqa: E402
from loom.evaluation.aggregate import dataset_table, load_extra, load_rows, load_timing  # noqa: E402
from loom.statistics.tests import friedman_ranks, pairwise_vs, nemenyi_cd  # noqa: E402

SMOKE = ["GunPoint", "ECG200", "Coffee", "ItalyPowerDemand", "SonyAIBORobotSurface2", "TwoLeadECG"]   # smoke / quick-mode datasets
MAIN = ["LOOM", "PreVal", "Ridge-softmax", "Ridge-onehot", "Holdout-TS", "Ridge-Platt-CV5",
        "Ridge-Isotonic-CV5", "LogReg-CV", "QUANT"]          # ensembles: official split only (RAWCMP)
RAWCMP = ["LOOM", "LOOM-Ens5", "PreVal", "PreVal-Ens5", "QUANT", "Arsenal"]     # official split only
ABL = ["LOOM", "PreVal", "PreVal-smooth", "LOOM-nosmooth", "LOOM-free-b", "LOOM-finite-c1", "LOOM-kNN",
       "LOOM-insample", "LOOM-holdout", "LOOM-lambdaSSE"]
CP = ["CP-LOOM-jackknife+", "CP-LOOM-jackknife", "CP-PreVal-jackknife+", "CP-Split-TS",
      "CP-LOOM-jackknife+-APS", "CP-PreVal-jackknife+-APS", "CP-Split-TS-APS"]
LOWER = {"nll": True, "ece": True, "brier": True, "cwece": True, "aurc": True, "acc": False, "bacc": False,
         "shift_auroc": False, "mean_conf": None}


GROUP_RULES = [  # datasets derived from the same source are merged into one unit (sensitivity analysis)
    (r"^UWaveGestureLibrary", "UWaveGestureLibrary"), (r"Phalan", "PhalanxOutlines"), (r"^Pig", "Pig"),
    (r"^Freezer", "Freezer"), (r"^InsectEPG", "InsectEPG"), (r"^MixedShapes", "MixedShapes"), (r"^GunPoint", "GunPoint"),
    (r"^SemgHand", "SemgHand"), (r"^NonInvasiveFetalECG", "NonInvasiveFetalECG"), (r"^EOG", "EOG"), (r"^Worms", "Worms"),
    (r"^SonyAIBORobotSurface", "SonyAIBO"), (r"^ToeSegmentation", "ToeSegmentation"), (r"^Ford[AB]$", "Ford"),
    (r"^(SmallKitchenAppliances|LargeKitchenAppliances|RefrigerationDevices|Computers|ScreenType|ElectricDevices)$", "UKHouseholdAppliances"),
    (r"^Face(All|sUCR)$", "Faces"), (r"^(CricketX|CricketY|CricketZ)$", "Cricket"), (r"^Lightning", "Lightning"),
    (r"^(ECGFiveDays|TwoLeadECG)$", "ECG-TwoLead"), (r"^(Strawberry|Coffee|Beef|Meat|OliveOil|Ham|Wine)$", "FoodSpectrographs"),
]


def group_of(name):
    import re
    for pat, g in GROUP_RULES:
        if re.search(pat, name):
            return g
    return name


def grouped_test(df, methods, metric, cond, ref="LOOM"):
    """Wilcoxon/Holm on group means (one unit per data source)."""
    T = dataset_table(df, metric, cond=cond, methods=methods)
    T = T[T[ref].notna()]
    G = T.groupby([group_of(d) for d in T.index]).mean()
    return G, pairwise_vs(G, ref, lower_is_better=LOWER[metric])


def fmt(x, nd=3):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def pval(p):
    return "–" if pd.isna(p) else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def summary_table(df, methods, metrics, cond, ref="LOOM", subset=None, min_cover=0.5):
    """Median over datasets, mean rank, W/D/L and Holm-adjusted Wilcoxon p vs ref.
    Pairwise tests use every dataset on which both methods are available (n is reported);
    Friedman test and mean ranks use the methods available on all datasets."""
    out, stats_out = [], {}
    for met in metrics:
        T = dataset_table(df, met, cond=cond, methods=methods)
        if subset is not None:
            T = T.loc[T.index.intersection(subset)]
        if ref not in T.columns:
            continue
        T = T[T[ref].notna()]
        keep = [m for m in T.columns if T[m].notna().mean() >= min_cover]
        T = T[keep]
        if len(T) < 5:
            continue
        lower = LOWER[met]
        full = [m for m in T.columns if T[m].notna().all()]
        ranks, fstat, fp, n = friedman_ranks(T[full], lower_is_better=lower)
        pw = pairwise_vs(T, ref, lower_is_better=lower).set_index("competitor")
        stats_out[met] = dict(n_datasets=int(len(T)), n_ranked=int(n), ranked_methods=full, friedman_chi2=fstat, friedman_p=fp,
                              mean_ranks=ranks.to_dict(), nemenyi_cd=nemenyi_cd(len(full), n),
                              pairwise=pw.reset_index().to_dict(orient="records"),
                              medians=T.median().to_dict(), means=T.mean().to_dict(), n_available=T.notna().sum().to_dict())
        for m in T.columns:
            r = dict(metric=met, method=m, n=int(T[m].notna().sum()), median=T[m].median(), mean=T[m].mean(),
                     mean_rank=ranks.get(m, np.nan))
            if m != ref:
                rr = pw.loc[m]
                r.update(wdl=f"{rr.ref_wins}/{rr.draws}/{rr.ref_losses}", median_diff=rr.median_diff, ci=f"[{rr.ci_low:.3f}, {rr.ci_high:.3f}]",
                         r_rb=rr.rank_biserial, p_holm=rr.p_holm)
            out.append(r)
    return pd.DataFrame(out), stats_out


def to_markdown(tab, path, title, note=""):
    lines = [f"**{title}**", ""]
    if tab.empty:
        Path(path).write_text("\n".join(lines + ["(no data)"])); return
    for met, g in tab.groupby("metric", sort=False):
        lines += [f"*Metric: {met}* (n = {int(g.n.max())} datasets; methods with fewer datasets show their n)", "",
                  "| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["mean_rank", "median"], na_position="last").iterrows():
            lines.append(f"| {r.method} | {int(r.n)} | {fmt(r['median'])} | {fmt(r['mean'])} | {fmt(r.mean_rank, 2)} | {r.get('wdl', '–') if isinstance(r.get('wdl'), str) else '–'} | "
                         f"{fmt(r.get('median_diff'))} {r.get('ci') if isinstance(r.get('ci'), str) else ''} | {fmt(r.get('r_rb'), 2)} | {pval(r.get('p_holm'))} |")
        lines.append("")
    if note:
        lines += [note, ""]
    Path(path).write_text("\n".join(lines))


def _real_noise_figure(rows, out_fig):
    """(a) one ECG beat with each real noise type at 6 dB; (b) median NLL vs SNR (clean, 12, 6, 0 dB)."""
    import matplotlib.pyplot as plt
    import sys as _s
    _s.path.insert(0, str(ROOT / "experiments"))
    from loom.visualization.plots import STYLE, _save
    from loom.data.ucr import load_split
    from real_noise import add_noise, read_212
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.4, 1, 1, 1]))
    _, _, Xte, _ = load_split("ECG5000", 0)
    x = Xte[3:4]
    names = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artefact"}
    labels = ["clean", "baseline wander", "electrode motion", "muscle artefact"]
    traces = [x[0]] + [add_noise(x, read_212(nz), 6, seed=7)[0] for nz in ("bw", "em", "ma")]
    for k, (tr, lab) in enumerate(zip(traces, labels)):
        off = -4.5 * k
        if k:
            axs[0].plot(x[0] + off, color="#b0b0b8", lw=0.8)
        axs[0].plot(tr + off, color="#1d1d1f" if k == 0 else "#d9682b", lw=0.9)
        axs[0].text(len(tr) * 1.02, off, lab, fontsize=6, va="center")
    axs[0].set_xlim(0, len(x[0]) * 1.75)
    axs[0].set_title("ECG beat with real noise (6 dB)", fontsize=8); axs[0].axis("off")
    d = pd.DataFrame(rows)
    for ax, nz in zip(axs[1:], ["bw", "em", "ma"]):
        for m in ["LOOM", "PreVal", "Ridge-Platt-CV5", "QUANT"]:
            cl = d[(d.cond == "clean") & (d.method == m)].groupby("dataset").nll.mean().median()
            ys = [cl] + [d[(d.cond == nz) & (d.snr == sn) & (d.method == m)].groupby("dataset").nll.mean().median() for sn in (12, 6, 0)]
            c, mk, ls = STYLE[m]
            ax.plot(range(4), ys, color=c, marker=mk, ls=ls, label={"Ridge-Platt-CV5": "Platt (5-fold)"}.get(m, m), ms=3.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "12", "6", "0"], fontsize=7)
        ax.set_title(names[nz], fontsize=8); ax.set_xlabel("SNR (dB)", fontsize=7)
    axs[1].set_ylabel("Median NLL", fontsize=7); axs[3].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(); _save(fig, out_fig, "fig_real_noise")


def _case_study_figure(npz_dir, out_fig, datasets=None, name="fig_case_study"):
    """Triage curves: auto-accepted error and share of faulty recordings caught vs review budget."""
    import matplotlib.pyplot as plt
    from loom.visualization.plots import STYLE, _save
    budgets = np.linspace(0, 0.5, 26)
    curves = {}
    for f in sorted(Path(npz_dir).glob("*.npz")):
        if datasets is not None and f.name.split("__")[0] not in datasets:
            continue
        o = np.load(f); y, flt = o["y"], o["is_fault"].astype(bool)
        for m in ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"]:
            if f"P_stream_{m}" not in o.files:
                continue
            P = o[f"P_stream_{m}"]; conf = P.max(1); ok = P.argmax(1) == y
            order = np.argsort(conf, kind="stable"); n = len(y)
            err, caught = [], []
            for b in budgets:
                k = int(round(b * n)); d = np.zeros(n, bool); d[order[:k]] = True
                err.append((~ok[~d]).mean()); caught.append(d[flt].mean() if flt.any() else np.nan)
            ds = f.name.split("__")[0]
            curves.setdefault(m, {}).setdefault(ds, []).append((err, caught))
    if not curves:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    LAB = {"Holdout-TS": "Hold-out TS", "Ridge-Platt-CV5": "Platt (5-fold)"}
    rng = np.random.default_rng(0)
    for m, byds in curves.items():
        E = np.array([np.mean([e for e, _ in v], 0) for v in byds.values()])     # dataset means over resamples
        C = np.array([np.nanmean([c for _, c in v], 0) for v in byds.values()])
        col, mk, ls = STYLE[m]
        for ax, A in ((axs[0], E), (axs[1], C)):
            idx = rng.integers(0, len(A), (1000, len(A)))
            boot = np.nanmean(A[idx], 1)                                          # bootstrap over datasets
            ax.fill_between(100 * budgets, 100 * np.nanquantile(boot, 0.025, 0), 100 * np.nanquantile(boot, 0.975, 0), color=col, alpha=0.15, lw=0)
            ax.plot(100 * budgets, 100 * np.nanmean(A, 0), color=col, ls=ls, marker=mk, markevery=5, label=LAB.get(m, m))
    axs[0].set_xlabel("Recordings sent to human review (%)"); axs[0].set_ylabel("Errors among automatic decisions (%)")
    axs[1].set_xlabel("Recordings sent to human review (%)"); axs[1].set_ylabel("Faulty recordings caught (%)")
    axs[1].plot([0, 50], [0, 50], color="#888888", lw=0.8, ls=":", label="random selection")
    axs[0].set_title("Automatic decisions", fontsize=8); axs[1].set_title("Review queue", fontsize=8)
    axs[1].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout(); _save(fig, out_fig, name)


def main(mode="full"):
    raw = ROOT / "results" / "raw"
    out_res, out_tab, out_fig = ROOT / "results", ROOT / "tables", ROOT / "figures"
    for d in (out_tab, out_fig):
        d.mkdir(exist_ok=True, parents=True)
    df = load_rows(raw)
    if df.empty:
        print("no results"); return
    tm = load_timing(raw); ex = load_extra(raw)
    df.to_csv(out_res / "raw_results.csv", index=False)
    tm.to_csv(out_res / "timing.csv", index=False)
    ex.to_csv(out_res / "loom_fit_details.csv", index=False)
    info = dataset_info()

    mr = df[(df["transform"].isin(["minirocket", "raw"])) & (df.train_frac == 1.0)]
    all_ds = sorted(mr[mr.method == "LOOM"].dataset.unique())
    unseen = [d for d in all_ds if d not in DEV_DATASETS + SMOKE]
    leak = pd.read_csv(out_res / "leakage_audit.csv") if (out_res / "leakage_audit.csv").exists() else None
    nodup = [d for d in all_ds if leak is None or d not in set(leak[leak.test_in_train > 0].dataset)]
    near = leak[(leak.test_in_train > 0) | (leak.get("near_dup_test_in_train", 0) > 0)] if leak is not None else None
    noneardup = [d for d in all_ds if near is None or d not in set(near.dataset)]
    stats = {"n_datasets": len(all_ds), "n_unseen": len(unseen), "n_nodup": len(nodup), "n_noneardup": len(noneardup),
             "resamples": sorted(int(x) for x in mr["resample"].unique())}

    # ---------------- E1/E2 main: clean
    mets = ["nll", "ece", "brier", "cwece", "acc", "bacc", "aurc"]
    t1, s1 = summary_table(mr, MAIN, mets, "clean")
    t1.to_csv(out_tab / "table_main_clean.csv", index=False); stats["E1_clean"] = s1
    to_markdown(t1, out_tab / "table_main_clean.md", "Table: In-distribution (clean test) performance, MiniRocket features, mean over resamples")
    # ---------------- E4 robustness: mean over 18 corruption conditions
    t4, s4 = summary_table(mr, MAIN, ["nll", "ece", "brier", "acc", "aurc", "shift_auroc"], "shifted")
    t4.to_csv(out_tab / "table_robustness_shifted.csv", index=False); stats["E4_shifted"] = s4
    to_markdown(t4, out_tab / "table_robustness_shifted.md", "Table: Robustness — mean over the 18 corruption conditions")
    # per family x severity medians
    sh = mr[(mr.cond != "clean") & mr.method.isin(MAIN)]
    fam = sh.groupby(["dataset", "method", "cond", "sev"])[["nll", "ece", "acc", "shift_auroc"]].mean().reset_index()
    famtab = fam.groupby(["cond", "sev", "method"])[["nll", "ece", "acc", "shift_auroc"]].median().reset_index()
    famtab.to_csv(out_tab / "table_robustness_by_family.csv", index=False)
    # ---------------- E2b raw-series probabilistic classifiers, official split (resample 0)
    # the Arsenal was run on clean + severity-3 data only, so every method is compared on the same conditions
    r0 = mr[(mr["resample"] == 0) & (mr["sev"].isin([0, 3]))]
    for cond in ("clean", "shifted"):
        t, s_ = summary_table(r0, RAWCMP, ["nll", "brier", "ece", "acc"] + (["shift_auroc"] if cond == "shifted" else []), cond)
        t.to_csv(out_tab / f"table_rawbaselines_{cond}.csv", index=False); stats[f"E2b_raw_{cond}"] = s_
        to_markdown(t, out_tab / f"table_rawbaselines_{cond}.md", f"Table: comparison with Arsenal and QUANT on the official split — {cond}")
    # ---------------- E3 ablation (clean and shifted)
    t3c, s3c = summary_table(mr, ABL, ["nll", "ece", "acc"], "clean")
    t3s, s3s = summary_table(mr, ABL, ["nll", "ece", "shift_auroc"], "shifted")
    t3c.to_csv(out_tab / "table_ablation_clean.csv", index=False); t3s.to_csv(out_tab / "table_ablation_shifted.csv", index=False)
    stats["E3_ablation_clean"] = s3c; stats["E3_ablation_shifted"] = s3s
    to_markdown(t3c, out_tab / "table_ablation_clean.md", "Table: Ablation — clean test")
    to_markdown(t3s, out_tab / "table_ablation_shifted.md", "Table: Ablation — mean over corruption conditions")
    # ---------------- sensitivity subsets
    for nm, sub in (("unseen", unseen), ("nodup", nodup), ("noneardup", noneardup)):
        a, sa = summary_table(mr, MAIN, ["nll", "ece"], "clean", subset=sub)
        b, sb = summary_table(mr, MAIN, ["nll", "ece"], "shifted", subset=sub)
        stats[f"sens_{nm}_clean"] = sa; stats[f"sens_{nm}_shifted"] = sb
        pd.concat([a.assign(cond="clean"), b.assign(cond="shifted")]).to_csv(out_tab / f"table_sensitivity_{nm}.csv", index=False)
        to_markdown(a, out_tab / f"table_sensitivity_{nm}_clean.md", f"Table: clean, subset={nm} ({len(sub)} datasets)")
        to_markdown(b, out_tab / f"table_sensitivity_{nm}_shifted.md", f"Table: shifted, subset={nm} ({len(sub)} datasets)")
    # ---------------- sensitivity: related datasets merged into one unit per source
    for cond in ("clean", "shifted"):
        out_g = {}
        for met in (["nll", "ece"] + (["shift_auroc"] if cond == "shifted" else [])):
            G, pwg = grouped_test(mr, MAIN, met, cond)
            out_g[met] = dict(n_units=int(len(G)), pairwise=pwg.to_dict(orient="records"), medians=G.median().to_dict())
        stats[f"sens_grouped_{cond}"] = out_g
    # ---------------- E10 natural sensor shift (UWaveGestureLibrary cross-axis transfer)
    ns = out_res / "natural_shift.json"
    if ns.exists():
        d = json.load(open(ns))
        nsr = pd.DataFrame([dict(task=f"{r['source']}->{r['target']}", resample=r["resample"], method=r["method"],
                                 nll_in=r["indomain"]["nll"], nll_shift=r["shifted"]["nll"], ece_shift=r["shifted"]["ece"],
                                 acc_in=r["indomain"]["acc"], acc_shift=r["shifted"]["acc"], shift_auroc=r["shift_auroc"],
                                 lev_ratio=r["lev_ratio"]) for r in d])
        tk = nsr.groupby(["task", "method"]).mean(numeric_only=True).reset_index()
        tk.to_csv(out_tab / "table_natural_shift_tasks.csv", index=False)
        summ = tk.groupby("method")[["nll_in", "nll_shift", "ece_shift", "acc_in", "acc_shift", "shift_auroc"]].mean()
        summ.to_csv(out_tab / "table_natural_shift.csv")
        E10 = dict(n_tasks=int(tk.task.nunique()), n_resamples=int(nsr["resample"].nunique()), means=summ.to_dict(orient="index"),
                   lev_ratio=tk[tk.method == "LOOM"].set_index("task").lev_ratio.to_dict())
        for met in ("nll_shift", "nll_in", "shift_auroc"):
            P = tk.pivot(index="task", columns="method", values=met)
            E10[f"pairwise_{met}"] = pairwise_vs(P, "LOOM", lower_is_better=(met != "shift_auroc")).to_dict(orient="records")
        stats["E10_natural_shift"] = E10
    # ---------------- E11 application case study (triage of faulty sensor recordings; Online Resource 2, section 10)
    cs = out_res / "case_study.json"
    if cs.exists():
        c = pd.DataFrame(json.load(open(cs)))
        per = c.groupby(["domain", "dataset", "method", "stream"]).mean(numeric_only=True).reset_index()
        per.to_csv(out_tab / "table_case_study_datasets.csv", index=False)
        E11 = dict(n_datasets=int(per.dataset.nunique()), n_resamples=int(c["resample"].nunique()), metrics={}, domains={})
        for stream in ("stream", "clean"):
            sub = per[per.stream == stream]
            for met in ["nll", "acc"] + [k for k in c.columns if "@" in k]:
                if stream == "clean" and met.startswith("fault_caught"):
                    continue
                P = sub.pivot(index="dataset", columns="method", values=met)
                lower = not (met.startswith("fault_caught") or met == "acc")
                E11["metrics"][f"{stream}:{met}"] = dict(
                    medians=P.median().to_dict(), means=P.mean().to_dict(),
                    pairwise=pairwise_vs(P, "LOOM", lower_is_better=lower).to_dict(orient="records"))
        for dom, g in per[per.stream == "stream"].groupby("domain"):
            E11["domains"][dom] = g.groupby("method")[["auto_err@20", "fault_caught@20", "nll"]].median().to_dict(orient="index")
        stats["E11_case_study"] = E11
    # ---------------- E13 confirmatory triage replication (Online Resource 2, 11.2)
    from loom.statistics.tests import bayes_signed_rank, holm
    csc = out_res / "case_study_confirm.json"
    if csc.exists():
        c = pd.DataFrame(json.load(open(csc)))
        per = c.groupby(["domain", "dataset", "method", "stream"]).mean(numeric_only=True).reset_index()
        per.to_csv(out_tab / "table_case_study_confirm_datasets.csv", index=False)
        E13 = dict(n_datasets=int(per.dataset.nunique()), metrics={}, grouped={})
        prim = []
        for met in ["fault_caught@20", "auto_err@20", "fault_caught@10", "auto_err@10", "fault_caught@30", "auto_err@30", "nll"]:
            P = per[per.stream == "stream"].pivot(index="dataset", columns="method", values=met)
            lower = not met.startswith("fault_caught")
            pw_ = pairwise_vs(P, "LOOM", lower_is_better=lower)
            E13["metrics"][f"stream:{met}"] = dict(medians=P.median().to_dict(), means=P.mean().to_dict(),
                                                    pairwise=pw_.to_dict(orient="records"))
            if met in ("fault_caught@20", "auto_err@20"):
                prim += [(met, r["competitor"], r["p"]) for r in pw_.to_dict(orient="records")]
                G = P.groupby([group_of(d) for d in P.index]).mean()
                E13["grouped"][met] = dict(n_units=len(G), pairwise=pairwise_vs(G, "LOOM", lower_is_better=lower).to_dict(orient="records"))
        Pc = per[per.stream == "clean"].pivot(index="dataset", columns="method", values="auto_err@20")
        E13["metrics"]["clean:auto_err@20"] = dict(medians=Pc.median().to_dict(), pairwise=pairwise_vs(Pc, "LOOM").to_dict(orient="records"))
        ph = holm([p for _, _, p in prim])                     # Holm over 4 comparisons x 2 primary endpoints
        E13["primary_holm8"] = [dict(metric=m, competitor=cpt, p=p, p_holm8=float(q)) for (m, cpt, p), q in zip(prim, ph)]
        stats["E13_case_confirm"] = E13
    # ---------------- E12 real recorded noise (MIT-BIH NSTDB; Online Resource 2, 11.1)
    rn = out_res / "real_noise.json"
    if rn.exists():
        d = pd.DataFrame(json.load(open(rn)))
        noisy = d[~d.cond.isin(["clean", "stream"])].copy()
        cell = noisy.groupby(["dataset", "cond", "snr", "method"]).mean(numeric_only=True).reset_index()
        cell.to_csv(out_tab / "table_real_noise_cells.csv", index=False)
        E12 = dict(n_datasets=int(d.dataset.nunique()), metrics={}, by_noise={}, by_snr={})
        for met in ["nll", "ece", "acc", "shift_auroc"]:
            lower = met in ("nll", "ece")
            Pd = cell.groupby(["dataset", "method"])[met].mean().unstack()          # dataset level (mean of 9 cells)
            Pcell = cell.assign(k=cell.dataset + "|" + cell.cond + "|" + cell.snr.astype(str)).pivot(index="k", columns="method", values=met)
            E12["metrics"][met] = dict(medians=Pd.median().to_dict(), means=Pd.mean().to_dict(),
                                       pairwise=pairwise_vs(Pd, "LOOM", lower_is_better=lower).to_dict(orient="records"),
                                       pairwise_cells=pairwise_vs(Pcell, "LOOM", lower_is_better=lower).to_dict(orient="records"),
                                       n_cells=int(len(Pcell)))
        for nz, g in cell.groupby("cond"):
            E12["by_noise"][nz] = g.groupby("method")[["nll", "shift_auroc", "acc"]].median().to_dict(orient="index")
        for sn, g in cell.groupby("snr"):
            E12["by_snr"][str(int(sn))] = g.groupby("method")[["nll", "shift_auroc", "acc"]].median().to_dict(orient="index")
        cl = d[d.cond == "clean"].groupby(["dataset", "method"])["nll"].mean().unstack()
        E12["clean_nll_medians"] = cl.median().to_dict()
        st = d[d.cond == "stream"].groupby(["dataset", "method"])[["fault_caught@20", "auto_err@20"]].mean()
        E12["stream"] = {k: st[k].unstack().median().to_dict() for k in st.columns}
        E12["stream_pairwise"] = {k: pairwise_vs(st[k].unstack(), "LOOM", lower_is_better=(k != "fault_caught@20")).to_dict(orient="records") for k in st.columns}
        stats["E12_real_noise"] = E12
    # ---------------- E14 LOOM on QUANT features (Online Resource 2, 11.3)
    qf = out_res / "quant_features.json"
    if qf.exists():
        q = pd.DataFrame(json.load(open(qf)))
        q["state"] = np.where(q.cond == "clean", "clean", "shifted")
        E14 = dict(n_datasets=int(q.dataset.nunique()), n_features_median=float(q.n_features.median()), metrics={})
        for state in ("clean", "shifted"):
            for met in (["nll", "ece", "acc"] + (["shift_auroc"] if state == "shifted" else [])):
                P = q[q.state == state].groupby(["dataset", "method"])[met].mean().unstack()
                P = P.rename(columns={"LOOM-QF": "LOOM"})
                E14["metrics"][f"{state}:{met}"] = dict(medians=P.median().to_dict(),
                                                         pairwise=pairwise_vs(P, "LOOM", lower_is_better=met in ("nll", "ece")).to_dict(orient="records"))
        stats["E14_quant_features"] = E14
    # ---------------- E15 Bayesian signed-rank tests (Benavoli et al. 2017; ROPE fixed in Online Resource 2, 11.4)
    def _bayes(T, comp, rope, lower=True):
        T = T[["LOOM", comp]].dropna()
        dd = (T[comp] - T["LOOM"]) if lower else (T["LOOM"] - T[comp])    # > 0 means LOOM better
        return dict(n=int(len(T)), rope=rope, **bayes_signed_rank(dd.to_numpy(), rope))
    E15 = {}
    E15["clean_nll_vs_PreVal"] = _bayes(dataset_table(mr, "nll", "clean", methods=["LOOM", "PreVal"]), "PreVal", 0.01)
    Tsh = dataset_table(mr, "nll", "shifted", methods=["LOOM", "PreVal", "QUANT"])
    E15["fault_nll_vs_PreVal"] = _bayes(Tsh, "PreVal", 0.01)
    E15["fault_nll_vs_QUANT"] = _bayes(Tsh, "QUANT", 0.01)
    E15["fault_auroc_vs_PreVal"] = _bayes(dataset_table(mr, "shift_auroc", "shifted", methods=["LOOM", "PreVal"]), "PreVal", 0.01, lower=False)
    if csc.exists():
        P = per[per.stream == "stream"]
        for met, lower in (("fault_caught@20", False), ("auto_err@20", True)):
            T = P.pivot(index="dataset", columns="method", values=met)
            for comp in ("PreVal", "QUANT"):
                E15[f"confirm_{met}_vs_{comp}"] = _bayes(T, comp, 0.01, lower)
    if rn.exists():
        E15["real_noise_nll_vs_PreVal"] = _bayes(cell.groupby(["dataset", "method"])["nll"].mean().unstack(), "PreVal", 0.01)
    if qf.exists():
        T = q[q.state == "shifted"].groupby(["dataset", "method"])["nll"].mean().unstack().rename(columns={"LOOM-QF": "LOOM"})
        E15["qf_fault_nll_vs_QUANT"] = _bayes(T, "QUANT", 0.01)
        E15["qf_fault_nll_vs_PreValQF"] = _bayes(T, "PreVal-QF", 0.01)
    stats["E15_bayes"] = E15
    # ---------------- E9 conformal
    cp = mr[mr.method.isin(CP + [m for m in mr.method.unique() if "@" in str(m)])]
    if not cp.empty:
        cpt = cp.groupby(["dataset", "method", "cond", "sev"])[["coverage", "set_size", "empty"]].mean().reset_index()
        cpt["shifted"] = cpt.cond != "clean"
        cps = cpt.groupby(["method", "shifted"]).agg(cov_mean=("coverage", "mean"), cov_median=("coverage", "median"),
                                                    cov_q10=("coverage", lambda x: x.quantile(0.1)),
                                                    size_mean=("set_size", "mean"), size_median=("set_size", "median"),
                                                    empty_mean=("empty", "mean"), n=("dataset", "nunique")).reset_index()
        cps.to_csv(out_tab / "table_conformal.csv", index=False)
        stats["E9_conformal"] = cps.to_dict(orient="records")
        # paired set-size test on clean: jackknife+ vs split
        ss = cpt[cpt.cond == "clean"].pivot_table(index="dataset", columns="method", values="set_size")
        for ref, comps in (("CP-LOOM-jackknife+", ["CP-Split-TS", "CP-PreVal-jackknife+"]),
                           ("CP-LOOM-jackknife+-APS", ["CP-Split-TS-APS", "CP-PreVal-jackknife+-APS"])):
            if {ref, *comps} <= set(ss.columns):
                stats[f"E9_setsize_test_{ref}"] = pairwise_vs(ss[[ref] + comps].dropna(), ref, True).to_dict(orient="records")
        cv = cpt[cpt.cond == "clean"].pivot_table(index="dataset", columns="method", values="coverage")
        stats["E9_coverage_by_dataset_quantiles"] = cv.quantile([0.05, 0.1, 0.25, 0.5]).to_dict()
    # ---------------- E5 generalisation: other extractors
    gen = df[df["transform"].isin(["rocket", "multirocket", "hydra"])]
    if not gen.empty:
        rows = []
        for tfm, g in gen.groupby("transform"):
            for cond in ("clean", "shifted"):
                t, s = summary_table(g, ["LOOM", "PreVal", "Ridge-softmax", "Holdout-TS"], ["nll", "ece", "acc"], cond, min_cover=0.8)
                stats[f"E5_{tfm}_{cond}"] = s
                rows.append(t.assign(transform=tfm, cond=cond))
        gt = pd.concat(rows); gt.to_csv(out_tab / "table_generalisation_extractors.csv", index=False)
        to_markdown(gt[gt.cond == "clean"], out_tab / "table_generalisation_clean.md", "Table: other random-kernel extractors — clean")
        to_markdown(gt[gt.cond == "shifted"], out_tab / "table_generalisation_shifted.md", "Table: other random-kernel extractors — shifted")
    # dataset type breakdown (UCR 'Type' column)
    try:
        meta = pd.read_csv(ROOT / "data/raw/Univariate_arff/DataSummary.csv")
    except Exception:
        meta = None
    if meta is None:
        try:
            meta = pd.read_excel(ROOT / "data/raw/Univariate_arff/DataSummaryExpanded_NoMissingVals.xlsx")
        except Exception:
            meta = None
    if meta is not None:
        cols = {c.lower(): c for c in meta.columns}
        nc, tc = cols.get("name") or cols.get("problem") or cols.get("dataset"), cols.get("type")
        if nc and tc:
            tmap = dict(zip(meta[nc].astype(str).str.strip(), meta[tc].astype(str).str.strip()))
            Tc = dataset_table(mr, "nll", "clean", methods=["LOOM", "PreVal"]).dropna()
            Ts = dataset_table(mr, "nll", "shifted", methods=["LOOM", "PreVal"]).dropna()
            bt = pd.DataFrame({"type": [tmap.get(d, "unknown") for d in Tc.index],
                               "clean_delta": (Tc.PreVal - Tc.LOOM).values,
                               "shift_delta": (Ts.reindex(Tc.index).PreVal - Ts.reindex(Tc.index).LOOM).values}, index=Tc.index)
            bt.to_csv(out_tab / "table_by_type_detail.csv")
            agg = bt.groupby("type").agg(n=("clean_delta", "size"), clean_median_gain=("clean_delta", "median"),
                                         clean_frac_better=("clean_delta", lambda x: (x > 0).mean()),
                                         shift_median_gain=("shift_delta", "median"),
                                         shift_frac_better=("shift_delta", lambda x: (x > 0).mean())).reset_index()
            agg.to_csv(out_tab / "table_by_type.csv", index=False); stats["E5_by_type"] = agg.to_dict(orient="records")
    # ---------------- E7 training-size sensitivity
    ts_ = df[(df.train_frac < 1.0) | ((df.train_frac == 1.0) & (df["transform"] == "minirocket"))]
    ts_ = ts_[ts_.method.isin(["LOOM", "PreVal", "Ridge-softmax", "Holdout-TS"]) & (ts_["resample"] == 0)]
    ts_ = ts_[ts_.sev.isin([0, 3])]          # reduced-data runs use clean + severity-3 conditions only
    if (df.train_frac < 1).any():
        r = []
        for tfv, g in ts_.groupby("train_frac"):
            for cond in ("clean", "shifted"):
                T = dataset_table(g, "nll", cond, methods=["LOOM", "PreVal", "Ridge-softmax", "Holdout-TS"])
                dsets = sorted(df[df.train_frac < 1].dataset.unique())
                T = T.loc[T.index.intersection(dsets)].dropna()
                pw = pairwise_vs(T, "LOOM", True)
                for _, p in pw.iterrows():
                    r.append(dict(train_frac=tfv, cond=cond, competitor=p.competitor, n=p.n, median_loom=T.LOOM.median(),
                                  median_comp=T[p.competitor].median(), wdl=f"{p.ref_wins}/{p.draws}/{p.ref_losses}", p_holm=p.p_holm))
        tst = pd.DataFrame(r); tst.to_csv(out_tab / "table_train_size.csv", index=False); stats["E7_train_size"] = tst.to_dict(orient="records")
    # ---------------- E6 efficiency
    if not tm.empty:
        tmm = tm[(tm["transform"].isin(["minirocket", "raw"])) & (tm.train_frac == 1.0)].copy()
        svd = ex.dropna(subset=["lam"]) if "lam" in ex else None
        # Ridge-onehot / Ridge-softmax reuse the shared ridge fit: charge them its SVD as well
        uses_svd = tmm.method.isin(["Ridge-onehot", "Ridge-softmax"])
        tmm.loc[uses_svd, "fit"] = tmm.loc[uses_svd, "fit"] + tmm.loc[uses_svd, "svd_time"].fillna(0.0)
        # MiniRocket transform of the training set (shared by all readouts; not included in their fit times)
        tr = tmm[(tmm["transform"] == "minirocket") & (tmm["method"] == "LOOM")].copy()
        tr["method"] = "MiniRocket-transform"; tr["fit"] = tr["feature_time"]; tr["predict_all"] = np.nan
        tmm = pd.concat([tmm, tr.dropna(subset=["fit"])], ignore_index=True)
        eff = tmm.groupby(["dataset", "method"])[["fit", "predict_all"]].mean().reset_index()
        effs = eff.groupby("method").agg(fit_median=("fit", "median"), fit_mean=("fit", "mean"), fit_total=("fit", "sum"),
                                         pred_median=("predict_all", "median"), pred_total=("predict_all", "sum"),
                                         n=("dataset", "nunique")).reset_index()
        effs.to_csv(out_tab / "table_efficiency.csv", index=False); stats["E6_efficiency"] = effs.to_dict(orient="records")
    # ---------------- E8 error analysis (dataset-level drivers of LOOM vs PreVal)
    Tc = dataset_table(mr, "nll", "clean", methods=["LOOM", "PreVal", "Ridge-softmax"]).dropna()
    Ts = dataset_table(mr, "nll", "shifted", methods=["LOOM", "PreVal"]).dropna()
    acc_s = dataset_table(mr, "acc", "shifted", methods=["LOOM"])
    lev = ex.dropna(subset=["lam"]) if "lam" in ex.columns else pd.DataFrame()
    err = pd.DataFrame(index=Tc.index)
    err["gain_clean"] = Tc.PreVal - Tc.LOOM
    err["gain_shift"] = Ts.reindex(err.index).PreVal - Ts.reindex(err.index).LOOM
    err["n_train"] = [info[d]["n_train"] for d in err.index]; err["n_classes"] = [info[d]["n_classes"] for d in err.index]
    err["length"] = [info[d]["length"] for d in err.index]
    err["acc_shift"] = acc_s.reindex(err.index).LOOM
    if not lev.empty:
        hcols = [c for c in lev.columns if c.startswith("h_") and c != "h_ref" and c != "h_clean0"]
        lv = lev[lev["transform"] == "minirocket"].copy()
        lv["lev_ratio_clean"] = lv["h_clean0"] / lv["h_ref"]
        lv["lev_ratio_shift"] = lv[hcols].mean(1) / lv["h_ref"]
        g = lv.groupby("dataset")[["lev_ratio_clean", "lev_ratio_shift", "lam", "a"]].median()
        err = err.join(g)
    err.to_csv(out_tab / "table_error_analysis_datasets.csv")
    from scipy.stats import spearmanr
    corr = {}
    for col in ["n_train", "n_classes", "length", "acc_shift", "lev_ratio_clean", "lev_ratio_shift"]:
        if col in err:
            for gcol in ["gain_clean", "gain_shift"]:
                v = err[[col, gcol]].dropna()
                if len(v) > 5:
                    rho, p = spearmanr(v[col], v[gcol]); corr[f"{gcol}~{col}"] = dict(rho=float(rho), p=float(p), n=len(v))
    stats["E8_correlations"] = corr
    worst = err.sort_values("gain_clean").head(10); stats["E8_worst_clean"] = worst.reset_index().to_dict(orient="records")
    worst_s = err.sort_values("gain_shift").head(10); stats["E8_worst_shift"] = worst_s.reset_index().to_dict(orient="records")
    # per-condition: where does LOOM lose to PreVal under shift?
    pc = fam[fam.method.isin(["LOOM", "PreVal"])].pivot_table(index=["dataset", "cond", "sev"], columns="method", values="nll").dropna()
    pc["loom_better"] = pc.LOOM < pc.PreVal
    stats["E8_win_rate_by_condition"] = pc.groupby(["cond", "sev"]).loom_better.mean().reset_index().to_dict(orient="records")
    # ---------------- leverage diagnostics
    if not lev.empty:
        lv = lev[lev["transform"] == "minirocket"]
        hcols = [c for c in lv.columns if c.startswith("h_") and c != "h_ref"]
        ratio = lv[hcols].div(lv["h_ref"], axis=0)
        stats["leverage_ratio_median_by_condition"] = ratio.median().to_dict()
        ratio.assign(dataset=lv.dataset.values, resample=lv["resample"].values).to_csv(out_tab / "table_leverage_ratios.csv", index=False)
    # ---------------- metrics.json (headline medians)
    metrics = {}
    for cond in ("clean", "shifted"):
        for met in ["nll", "ece", "brier", "acc", "aurc", "shift_auroc"]:
            if cond == "clean" and met == "shift_auroc":
                continue
            T = dataset_table(mr, met, cond, methods=MAIN + ABL[2:])
            metrics[f"{cond}_{met}_median"] = T.median().to_dict()
            metrics[f"{cond}_{met}_mean"] = T.mean().to_dict()
    json.dump(metrics, open(out_res / "metrics.json", "w"), indent=1, default=float)
    # ---------------- means with 95 % bootstrap CIs (1,000 resamples; deviation D20) and per-experiment exports
    from loom.statistics.tests import bootstrap_mean_ci
    CI = {}
    for key, metric, cond, meths in (("clean_nll", "nll", "clean", MAIN), ("clean_ece", "ece", "clean", MAIN),
                                     ("faults_nll", "nll", "shifted", MAIN), ("faults_auroc", "shift_auroc", "shifted", MAIN),
                                     ("faults_ece", "ece", "shifted", MAIN)):
        T = dataset_table(mr, metric, cond, methods=meths)
        CI[key] = {m: bootstrap_mean_ci(T[m].to_numpy()) for m in T.columns}
        for m in T.columns:
            if m != "LOOM":
                pair = T[["LOOM", m]].dropna()
                CI[key][f"diff_{m}"] = bootstrap_mean_ci((pair[m] - pair["LOOM"]).to_numpy())   # competitor - LOOM
    if csc.exists():
        P = per[per.stream == "stream"]
        for met in ("fault_caught@20", "auto_err@20"):
            T = P.pivot(index="dataset", columns="method", values=met)
            CI[f"confirm_{met}"] = {m: bootstrap_mean_ci(T[m].to_numpy()) for m in T.columns}
    if rn.exists():
        Pc = cell.assign(k=cell.dataset + "|" + cell.cond + "|" + cell.snr.astype(str)).pivot(index="k", columns="method", values="nll")
        CI["realnoise_nll_cells"] = {m: bootstrap_mean_ci(Pc[m].to_numpy()) for m in Pc.columns}
    stats["CI"] = CI
    EXPORT = {"e01_clean": ["E1_clean", "CI"], "e02_raw_series": ["E2b_raw_clean", "E2b_raw_shifted"],
              "e03_ablation": ["E3_ablation_clean", "E3_ablation_shifted"], "e04_faults": ["E4_shifted"],
              "e05_extractors": ["E5_multirocket_clean", "E5_multirocket_shifted", "E5_hydra_clean", "E5_hydra_shifted", "E5_by_type"],
              "e06_cost": ["E6_efficiency"], "e07_train_size": ["E7_train_size"],
              "e08_error_analysis": ["E8_correlations", "E8_worst_clean", "E8_worst_shift", "E8_win_rate_by_condition", "leverage_ratio_median_by_condition"],
              "e09_conformal": ["E9_conformal", "E9_coverage_by_dataset_quantiles"], "e10_natural_shift": ["E10_natural_shift"],
              "e11_triage_exploratory": ["E11_case_study"], "e12_real_noise": ["E12_real_noise"], "e13_triage_confirmatory": ["E13_case_confirm"],
              "e14_quant_features": ["E14_quant_features"], "e15_bayes": ["E15_bayes"],
              "sensitivity": ["sens_unseen_clean", "sens_unseen_shifted", "sens_nodup_clean", "sens_nodup_shifted",
                              "sens_noneardup_clean", "sens_noneardup_shifted", "sens_grouped_clean", "sens_grouped_shifted"]}
    for name, keys in EXPORT.items():
        d = out_res / name.split("_")[0]
        d.mkdir(parents=True, exist_ok=True)
        json.dump({k: stats[k] for k in keys if k in stats} | {"experiment": name}, open(d / "summary.json", "w"), indent=1, default=float)
    if (out_res / "check_logreg_convergence.json").exists():
        (out_res / "e16").mkdir(exist_ok=True)
        json.dump({"experiment": "e16_logreg_convergence", "records": json.load(open(out_res / "check_logreg_convergence.json"))},
                  open(out_res / "e16" / "summary.json", "w"), indent=1)
    json.dump(stats, open(out_res / "statistics.json", "w"), indent=1, default=float)
    print(f"analysed {len(all_ds)} datasets; rows={len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="full"); a = ap.parse_args()
    main(a.mode)
