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


def fmt(x, nd=3):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def pval(p):
    return "–" if pd.isna(p) else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def summary_table(df, methods, metrics, cond, ref="LOOM", subset=None, min_cover=0.9):
    """Median over datasets, mean rank, W/D/L and Holm-adjusted Wilcoxon p vs ref."""
    out, stats_out = [], {}
    for met in metrics:
        T = dataset_table(df, met, cond=cond, methods=methods)
        if subset is not None:
            T = T.loc[T.index.intersection(subset)]
        # keep methods available on (almost) all datasets; compare on common datasets
        keep = [m for m in T.columns if T[m].notna().mean() >= min_cover]
        T = T[keep].dropna()
        if ref not in T.columns or len(T) < 5:
            continue
        lower = LOWER[met]
        ranks, fstat, fp, n = friedman_ranks(T, lower_is_better=lower)
        pw = pairwise_vs(T, ref, lower_is_better=lower).set_index("competitor")
        stats_out[met] = dict(n_datasets=int(n), friedman_chi2=fstat, friedman_p=fp, mean_ranks=ranks.to_dict(),
                              nemenyi_cd=nemenyi_cd(T.shape[1], n), pairwise=pw.reset_index().to_dict(orient="records"),
                              medians=T.median().to_dict(), means=T.mean().to_dict())
        for m in T.columns:
            r = dict(metric=met, method=m, n=len(T), median=T[m].median(), mean=T[m].mean(), mean_rank=ranks[m])
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
        lines += [f"*Metric: {met}* (n = {int(g.n.iloc[0])} datasets)", "",
                  "| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in g.sort_values("mean_rank").iterrows():
            lines.append(f"| {r.method} | {fmt(r['median'])} | {fmt(r['mean'])} | {r.mean_rank:.2f} | {r.get('wdl', '–') if isinstance(r.get('wdl'), str) else '–'} | "
                         f"{fmt(r.get('median_diff'))} {r.get('ci') if isinstance(r.get('ci'), str) else ''} | {fmt(r.get('r_rb'), 2)} | {pval(r.get('p_holm'))} |")
        lines.append("")
    if note:
        lines += [note, ""]
    Path(path).write_text("\n".join(lines))


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
    stats = {"n_datasets": len(all_ds), "n_unseen": len(unseen), "n_nodup": len(nodup),
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
    r0 = mr[mr["resample"] == 0]
    for cond in ("clean", "shifted"):
        t, s_ = summary_table(r0, RAWCMP, ["nll", "ece", "acc"] + (["shift_auroc"] if cond == "shifted" else []), cond)
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
    for nm, sub in (("unseen", unseen), ("nodup", nodup)):
        a, sa = summary_table(mr, MAIN, ["nll", "ece"], "clean", subset=sub)
        b, sb = summary_table(mr, MAIN, ["nll", "ece"], "shifted", subset=sub)
        stats[f"sens_{nm}_clean"] = sa; stats[f"sens_{nm}_shifted"] = sb
        pd.concat([a.assign(cond="clean"), b.assign(cond="shifted")]).to_csv(out_tab / f"table_sensitivity_{nm}.csv", index=False)
        to_markdown(a, out_tab / f"table_sensitivity_{nm}_clean.md", f"Table: clean, subset={nm} ({len(sub)} datasets)")
        to_markdown(b, out_tab / f"table_sensitivity_{nm}_shifted.md", f"Table: shifted, subset={nm} ({len(sub)} datasets)")
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
    json.dump(stats, open(out_res / "statistics.json", "w"), indent=1, default=float)
    print(f"analysed {len(all_ds)} datasets; rows={len(df)}")
    from loom.visualization.plots import make_all
    make_all(ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="full"); a = ap.parse_args()
    main(a.mode)
