"""Read all machine-generated results and write the LaTeX number macros and tables.

    tables/macros.tex   \\newcommand{\\Name}{value} for every number and data-dependent phrase in the paper
    tables/tab_*.tex    every table of the paper and the supplement (booktabs)

The manuscript uses only these macros and tables; nothing is typed by hand. Direction words
("better", "worse", "not significantly different") are computed from the tests.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
S = json.load(open(RES / "statistics.json"))
SAN = json.load(open(RES / "e17" / "sanity.json")) if (RES / "e17" / "sanity.json").exists() else None
LEAK = json.load(open(RES / "e00" / "leakage_audit.json"))
LR = json.load(open(RES / "check_logreg_convergence.json"))
NS = S["E10_natural_shift"]
MAC = {}
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def put(name, value):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"illegal macro name {name}")
    if name in MAC and MAC[name] != value:
        raise ValueError(f"macro {name} defined twice with different values")
    MAC[name] = value


def num(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("missing value")
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\,\\%"


def count(n):
    """Counts below ten as words (style rule)."""
    return WORDS.get(int(n), str(int(n)))


def pv(p):
    return "$p<0.001$" if p < 1e-3 else f"$p={p:.3f}$"


def pw(key, metric, comp):
    for r in S[key][metric]["pairwise"]:
        if r["competitor"] == comp:
            return r
    raise KeyError((key, metric, comp))


def find(recs, comp):
    for r in recs:
        if r["competitor"] == comp:
            return r
    raise KeyError(comp)


def cmp_phrase(r, lower=True, unit="datasets", p_key="p_holm"):
    """LOOM-perspective phrase: 'better on 108 of 114 datasets ($p<0.001$)'."""
    n = r["ref_wins"] + r["draws"] + r["ref_losses"]
    better = (r["median_diff"] > 0) if lower else (r["median_diff"] < 0)
    p = r[p_key]
    if p >= 0.05:
        return f"not significantly different ({r['ref_wins']} wins, {r['ref_losses']} losses of {n} {unit}, {pv(p)})"
    k = r["ref_wins"] if better else r["ref_losses"]
    return f"{'better' if better else 'worse'} on {k} of {n} {unit} ({pv(p)})"


def ci(d, nd=3, scale=1.0, unit=""):
    return f"{d['mean'] * scale:.{nd}f}{unit} [{d['lo'] * scale:.{nd}f}, {d['hi'] * scale:.{nd}f}]"


METHODS = {"LOOM": "Loom", "PreVal": "PreVal", "Holdout-TS": "HoldoutTS", "Ridge-Platt-CV5": "Platt", "Ridge-Isotonic-CV5": "Isotonic",
           "LogReg-CV": "LogReg", "QUANT": "Quant", "Ridge-softmax": "Softmax", "Ridge-onehot": "Onehot", "Arsenal": "Arsenal",
           "PreVal-Ens5": "PreValEns", "LOOM-Ens5": "LoomEns", "PreVal-QF": "PreValQF", "MiniRocket-transform": "Transform",
           "PreVal-smooth": "PreValSmooth", "LOOM-nosmooth": "NoSmooth", "LOOM-free-b": "FreeB", "LOOM-finite-c1": "FiniteC",
           "LOOM-kNN": "KNN", "LOOM-insample": "InSample", "LOOM-holdout": "HoldoutPairs", "LOOM-lambdaSSE": "LambdaSSE"}
NAMES = {"LOOM": "LOOM (ours)", "PreVal": "PreVal", "Holdout-TS": "Hold-out TS", "Ridge-Platt-CV5": "Ridge + Platt (5-fold)",
         "Ridge-Isotonic-CV5": "Ridge + isotonic (5-fold)", "LogReg-CV": "Logistic regression (CV)", "QUANT": "QUANT",
         "Ridge-softmax": "Ridge softmax ($T=1$)", "Ridge-onehot": "Ridge one-hot", "Arsenal": "Arsenal",
         "PreVal-Ens5": "PreVal, 5-seed ens.", "LOOM-Ens5": "LOOM, 5-seed ens.", "MiniRocket-transform": "MiniRocket transform (shared)"}
FAM = {"noise": "noise", "wander": "baseline wander", "missing": "dropouts", "spikes": "spikes", "warp": "speed variation", "shift": "misalignment"}
FAMS = list(FAM)


# ------------------------------------------------------------------ general
def general():
    put("NDatasets", str(S["n_datasets"])); put("NUnseen", str(S["n_unseen"])); put("NNoDup", str(S["n_nodup"]))
    put("NNoNearDup", str(S["n_noneardup"])); put("NDupExact", str(LEAK["n_exact_cross"])); put("NDupNear", str(LEAK["n_near_cross"]))
    put("NDupNearOnly", count(len(LEAK["datasets_near_only"])))
    put("PlantedPass", "passed" if LEAK["planted_leak_test"]["passed"] else "failed")
    put("NGrouped", str(S["sens_grouped_shifted"]["nll"]["n_units"]))


# ------------------------------------------------------------------ main comparisons (E1, E4)
def main_tables():
    for key, tag in (("E1_clean", "Clean"), ("E4_shifted", "Faults")):
        for metric, mt in (("nll", "NLL"), ("ece", "ECE"), ("acc", "Acc"), ("brier", "Brier"), ("shift_auroc", "AUROC")):
            if metric not in S[key]:
                continue
            for m, v in S[key][metric]["medians"].items():
                if m in METHODS:
                    put(f"{mt}{tag}{METHODS[m]}", num(v))
            for r in S[key][metric]["pairwise"]:
                if r["competitor"] in METHODS:
                    put(f"Cmp{mt}{tag}{METHODS[r['competitor']]}", cmp_phrase(r, lower=metric in ("nll", "ece", "brier")))
    for key, tag in (("clean_nll", "CINLLClean"), ("faults_nll", "CINLLFaults"), ("faults_auroc", "CIAUROCFaults"), ("clean_ece", "CIECEClean")):
        d = S["CI"][key]
        for m in ("LOOM", "PreVal", "QUANT", "Ridge-Platt-CV5", "Holdout-TS"):
            if m in d:
                put(f"{tag}{METHODS[m]}", ci(d[m]))
            if f"diff_{m}" in d:
                put(f"{tag}Diff{METHODS[m]}", ci(d[f"diff_{m}"]))
    lev = S["leverage_ratio_median_by_condition"]
    put("LevClean", num(lev["h_clean0"], 2))
    for fam in FAMS:
        put(f"Lev{fam.capitalize()}", num(lev[f"h_{fam}3"], 1 if lev[f"h_{fam}3"] > 2 else 2))
    # Bayesian
    for k, nm in (("clean_nll_vs_PreVal", "BayesCleanPreVal"), ("fault_nll_vs_PreVal", "BayesFaultsPreVal"),
                  ("fault_nll_vs_QUANT", "BayesFaultsQuant"), ("fault_auroc_vs_PreVal", "BayesAUROCPreVal"),
                  ("real_noise_nll_vs_PreVal", "BayesRealNoisePreVal"), ("qf_fault_nll_vs_QUANT", "BayesQFQuant"),
                  ("confirm_fault_caught@20_vs_PreVal", "BayesConfirmCaughtPreVal"), ("confirm_fault_caught@20_vs_QUANT", "BayesConfirmCaughtQuant"),
                  ("confirm_auto_err@20_vs_PreVal", "BayesConfirmErrPreVal"), ("confirm_auto_err@20_vs_QUANT", "BayesConfirmErrQuant")):
        b = S["E15_bayes"][k]
        put(f"{nm}Better", num(b["p_right"], 2)); put(f"{nm}Equiv", num(b["p_rope"], 2)); put(f"{nm}Worse", num(b["p_left"], 2))


# ------------------------------------------------------------------ E2 official split
def official():
    for key, tag in (("E2b_raw_clean", "OffClean"), ("E2b_raw_shifted", "OffSevThree")):
        for metric, mt in (("nll", "NLL"), ("brier", "Brier"), ("shift_auroc", "AUROC"), ("acc", "Acc")):
            if metric not in S[key]:
                continue
            for m, v in S[key][metric]["medians"].items():
                if m in METHODS:
                    put(f"{mt}{tag}{METHODS[m]}", num(v))
            for r in S[key][metric]["pairwise"]:
                if r["competitor"] in METHODS:
                    put(f"Cmp{mt}{tag}{METHODS[r['competitor']]}", cmp_phrase(r, lower=metric in ("nll", "brier")))
    t = pd.read_csv(TAB / "table_robustness_by_family.csv")
    t = t[t["sev"] == 3].pivot(index="cond", columns="method", values="nll")
    qb = [FAM[c] for c in FAMS if t.loc[c, "QUANT"] < t.loc[c, "LOOM"]]
    lb = [FAM[c] for c in FAMS if t.loc[c, "LOOM"] < t.loc[c, "QUANT"]]
    j = lambda L: L[0] if len(L) == 1 else ", ".join(L[:-1]) + " and " + L[-1]
    put("FamQuantBetter", j(qb)); put("FamLoomBetter", j(lb))


# ------------------------------------------------------------------ E3 ablation
def ablation():
    for key, tag in (("E3_ablation_clean", "AblClean"), ("E3_ablation_shifted", "AblFaults")):
        for metric, mt in (("nll", "NLL"), ("ece", "ECE"), ("shift_auroc", "AUROC")):
            if metric not in S[key]:
                continue
            for m, v in S[key][metric]["medians"].items():
                if m in METHODS:
                    put(f"{mt}{tag}{METHODS[m]}", num(v))
            for r in S[key][metric]["pairwise"]:
                if r["competitor"] in METHODS:
                    put(f"Cmp{mt}{tag}{METHODS[r['competitor']]}", cmp_phrase(r, lower=metric != "shift_auroc"))
    put("NWinsAUROCFiniteC", str(pw("E3_ablation_shifted", "shift_auroc", "LOOM-finite-c1")["ref_wins"]))


# ------------------------------------------------------------------ E5, E7, sensitivity
def generality():
    for ext, tag in (("multirocket", "MR"), ("hydra", "Hydra")):
        put(f"Cmp{tag}Faults", cmp_phrase(pw(f"E5_{ext}_shifted", "nll", "PreVal")))
        put(f"Cmp{tag}Clean", cmp_phrase(pw(f"E5_{ext}_clean", "nll", "PreVal")))
    for sub, tag in (("unseen", "Unseen"), ("nodup", "NoDup"), ("noneardup", "NoNearDup")):
        put(f"Cmp{tag}Faults", cmp_phrase(pw(f"sens_{sub}_shifted", "nll", "PreVal")))
        put(f"Cmp{tag}Clean", cmp_phrase(pw(f"sens_{sub}_clean", "nll", "PreVal")))
    g = find(S["sens_grouped_shifted"]["nll"]["pairwise"], "PreVal")
    put("CmpGroupedFaults", cmp_phrase(g, unit="source-level units"))
    ts = {(r["train_frac"], r["cond"], r["competitor"]): r for r in S["E7_train_size"]}
    for fr, tag in ((0.25, "TwentyFive"), (0.5, "Fifty")):
        r = ts[(fr, "shifted", "PreVal")]
        put(f"TS{tag}Loom", num(r["median_loom"])); put(f"TS{tag}PreVal", num(r["median_comp"]))
        put(f"TS{tag}WDL", r["wdl"].replace("/", "/")); put(f"TS{tag}P", pv(r["p_holm"]))
    put("TSN", str(ts[(0.25, "shifted", "PreVal")]["n"]))
    ok = all(ts[(fr, "shifted", "PreVal")]["p_holm"] < 0.05 and ts[(fr, "shifted", "PreVal")]["median_loom"] < ts[(fr, "shifted", "PreVal")]["median_comp"] for fr in (0.25, 0.5))
    put("TSClaim", "LOOM keeps its advantage under faults with smaller training sets" if ok else "LOOM's advantage under faults is not consistently kept with smaller training sets")
    m = NS["means"]
    put("NSAccIn", num(m["LOOM"]["acc_in"])); put("NSAccShift", num(m["LOOM"]["acc_shift"]))
    put("NSNLLLoom", num(m["LOOM"]["nll_shift"])); put("NSNLLPreVal", num(m["PreVal"]["nll_shift"]))
    put("NSAUROCLoom", num(m["LOOM"]["shift_auroc"])); put("NSAUROCPreVal", num(m["PreVal"]["shift_auroc"]))
    put("NSSoftmaxShift", num(m["Ridge-softmax"]["nll_shift"])); put("NSSoftmaxIn", num(m["Ridge-softmax"]["nll_in"])); put("NSLoomIn", num(m["LOOM"]["nll_in"]))
    lr = list(NS["lev_ratio"].values()); put("NSLevLo", num(min(lr), 2)); put("NSLevHi", num(max(lr), 2))
    r = find(NS["pairwise_nll_shift"], "PreVal"); put("NSWins", count(r["ref_wins"])); put("NSTasks", count(NS["n_tasks"]))


# ------------------------------------------------------------------ E8 error analysis
def errors():
    c = S["E8_correlations"]
    for k, nm in (("gain_shift~lev_ratio_shift", "RhoGainLev"), ("gain_shift~acc_shift", "RhoGainAcc"), ("gain_clean~n_classes", "RhoCleanClasses")):
        put(nm, num(c[k]["rho"], 2)); put(nm + "P", pv(c[k]["p"]))
    wr = {(r["cond"], int(r["sev"])): r["loom_better"] for r in S["E8_win_rate_by_condition"]}
    for fam in FAMS:
        put(f"Win{fam.capitalize()}", pct(wr[(fam, 3)], 0))
    s1 = [wr[(f, 1)] for f in FAMS if f != "warp"]
    put("WinSevOneLo", pct(min(s1), 0)); put("WinSevOneHi", pct(max(s1), 0))
    bad = [FAM[f] for f in FAMS if not (wr[(f, 1)] < wr[(f, 2)] < wr[(f, 3)])]
    put("TrendExcept", " and ".join(bad) if bad else "none")
    ws, wc = S["E8_worst_shift"], S["E8_worst_clean"]
    put("WorstShift", f"{ws[0]['dataset']}, {ws[1]['dataset']} and {ws[2]['dataset']}")
    put("WorstShiftEx", ws[2]["dataset"]); put("WorstShiftExAcc", num(ws[2]["acc_shift"], 2)); put("WorstShiftExLev", num(ws[2]["lev_ratio_shift"], 1))
    put("WorstClean", f"{wc[0]['dataset']}, {wc[1]['dataset']}, {wc[2]['dataset']} and {wc[3]['dataset']}")
    pc = [wc[i]["n_train"] / wc[i]["n_classes"] for i in range(4)]
    put("WorstCleanPerClassLo", count(round(min(pc)))); put("WorstCleanPerClassHi", str(round(max(pc))))


# ------------------------------------------------------------------ E9 conformal, E6 cost, E16
def conformal_cost():
    cp = {(r["method"], bool(r["shifted"])): r for r in S["E9_conformal"]}
    for m, nm in (("CP-LOOM-jackknife+", "CPLoom"), ("CP-LOOM-jackknife+-APS", "CPLoomAPS"), ("CP-Split-TS", "CPSplit"),
                  ("CP-PreVal-jackknife+", "CPPreVal")):
        for sh, t in ((False, "Clean"), (True, "Faults")):
            r = cp[(m, sh)]
            put(f"{nm}Cov{t}", pct(r["cov_mean"])); put(f"{nm}Size{t}", num(r["size_mean"], 2)); put(f"{nm}Empty{t}", pct(r["empty_mean"]))
    put("CPLoomCovAlphaFive", pct(cp[("CP-LOOM-jackknife+@0.05", False)]["cov_mean"]))
    put("CPLoomCovAlphaTwenty", pct(cp[("CP-LOOM-jackknife+@0.2", False)]["cov_mean"]))
    r = find(S["E9_setsize_test_CP-LOOM-jackknife+"], "CP-Split-TS")
    put("CmpSetSizeSplit", f"smaller on {r['ref_wins']} of {r['ref_wins'] + r['draws'] + r['ref_losses']} datasets ({pv(r['p_holm'])})")
    e = {r["method"]: r for r in S["E6_efficiency"]}
    for m in ("LOOM", "PreVal", "Ridge-Platt-CV5", "LogReg-CV", "LOOM-Ens5", "QUANT", "Arsenal", "MiniRocket-transform"):
        put(f"Fit{METHODS[m]}", num(e[m]["fit_median"], 2))
    put("FitEndToEndLoom", num(e["MiniRocket-transform"]["fit_median"] + e["LOOM"]["fit_median"], 2))
    put("QuantCostRatio", num(e["QUANT"]["fit_median"] / (e["MiniRocket-transform"]["fit_median"] + e["LOOM"]["fit_median"]), 1))
    g = sorted(LR, key=lambda r: r["nll_long"] - r["nll_default"])[0]
    put("LRMedDefault", num(float(np.median([r["nll_default"] for r in LR])), 4)); put("LRMedLong", num(float(np.median([r["nll_long"] for r in LR])), 4))
    put("LRMaxGain", num(g["nll_default"] - g["nll_long"])); put("LRMaxDataset", g["dataset"]); put("LRN", str(len(LR)))


# ------------------------------------------------------------------ E11, E13 triage; E12 real noise; E14 QUANT features
def application():
    E11 = S["E11_case_study"]["metrics"]
    for k, nm in (("stream:fault_caught@20", "XCaught"), ("stream:auto_err@20", "XErr")):
        for m in ("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"):
            put(f"{nm}{METHODS[m]}", pct(E11[k]["medians"][m], 0 if "Caught" in nm else 1))
    r = find(E11["stream:fault_caught@20"]["pairwise"], "PreVal"); put("XCaughtWinsPreVal", str(r["ref_wins"])); put("XCaughtPPreVal", pv(r["p_holm"]))
    r = find(E11["stream:auto_err@20"]["pairwise"], "PreVal"); put("XErrPPreVal", pv(r["p_holm"])); put("XErrWinsPreVal", count(r["ref_wins"]))
    E13 = S["E13_case_confirm"]
    put("CNDatasets", str(E13["n_datasets"]))
    for k, nm, nd in (("stream:fault_caught@20", "CCaught", 0), ("stream:auto_err@20", "CErr", 1), ("clean:auto_err@20", "CErrClean", 1)):
        for m in ("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"):
            put(f"{nm}{METHODS[m]}", pct(E13["metrics"][k]["medians"][m], nd))
    h8 = {(r["metric"], r["competitor"]): r["p_holm8"] for r in E13["primary_holm8"]}
    for met, nm, lower in (("fault_caught@20", "CmpCCaught", False), ("auto_err@20", "CmpCErr", True)):
        for c in ("PreVal", "QUANT", "Holdout-TS", "Ridge-Platt-CV5"):
            r = dict(find(E13["metrics"][f"stream:{met}"]["pairwise"], c)); r["p_holm"] = h8[(met, c)]
            put(f"{nm}{METHODS[c]}", cmp_phrase(r, lower=lower))
    g = find(E13["grouped"]["fault_caught@20"]["pairwise"], "PreVal"); put("CmpCCaughtGrouped", cmp_phrase(g, lower=False, unit="source-level units"))
    for m in ("LOOM", "PreVal", "QUANT"):
        d = S["CI"]["confirm_fault_caught@20"][m]; put(f"CICCaught{METHODS[m]}", ci(d, 1, 100, "\\,\\%"))
        d = S["CI"]["confirm_auto_err@20"][m]; put(f"CICErr{METHODS[m]}", ci(d, 1, 100, "\\,\\%"))
    E12 = S["E12_real_noise"]
    for m in ("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"):
        t = METHODS[m]
        put(f"RNNLL{t}", num(E12["metrics"]["nll"]["medians"][m])); put(f"RNAUROC{t}", num(E12["metrics"]["shift_auroc"]["medians"][m]))
        put(f"RNAcc{t}", num(E12["metrics"]["acc"]["medians"][m])); put(f"RNCleanNLL{t}", num(E12["clean_nll_medians"][m]))
        put(f"RNCaught{t}", pct(E12["stream"]["fault_caught@20"][m], 0)); put(f"RNErr{t}", pct(E12["stream"]["auto_err@20"][m]))
        put(f"RNNLLZero{t}", num(E12["by_snr"]["0"][m]["nll"], 2))
    for c in ("PreVal", "QUANT", "Ridge-Platt-CV5"):
        put(f"CmpRNCells{METHODS[c]}", cmp_phrase(find(E12["metrics"]["nll"]["pairwise_cells"], c), unit="dataset--noise combinations"))
        put(f"CmpRNAUROCCells{METHODS[c]}", cmp_phrase(find(E12["metrics"]["shift_auroc"]["pairwise_cells"], c), lower=False, unit="dataset--noise combinations"))
    put("RNNCells", str(E12["metrics"]["nll"]["n_cells"]))
    for nz, t in (("bw", "BW"), ("em", "EM"), ("ma", "MA")):
        put(f"RN{t}Loom", num(E12["by_noise"][nz]["LOOM"]["nll"], 2)); put(f"RN{t}PreVal", num(E12["by_noise"][nz]["PreVal"]["nll"], 2))
    put("CIRNLoom", ci(S["CI"]["realnoise_nll_cells"]["LOOM"])); put("CIRNPreVal", ci(S["CI"]["realnoise_nll_cells"]["PreVal"]))
    E14 = S["E14_quant_features"]
    put("QFN", str(E14["n_datasets"])); put("QFFeatures", f"{E14['n_features_median']:,.0f}".replace(",", "{,}"))
    for k, nm in (("shifted:nll", "QFNLLFaults"), ("clean:nll", "QFNLLClean"), ("shifted:shift_auroc", "QFAUROC"), ("shifted:acc", "QFAcc"), ("clean:acc", "QFAccClean")):
        for m in ("LOOM", "PreVal-QF", "QUANT"):
            put(f"{nm}{METHODS[m]}", num(E14["metrics"][k]["medians"][m], 1 if E14["metrics"][k]["medians"][m] > 10 else 3))
    put("CmpQFPreVal", cmp_phrase(find(E14["metrics"]["shifted:nll"]["pairwise"], "PreVal-QF")))
    put("CmpQFQuant", cmp_phrase(find(E14["metrics"]["shifted:nll"]["pairwise"], "QUANT")))
    put("CmpQFAUROCQuant", cmp_phrase(find(E14["metrics"]["shifted:shift_auroc"]["pairwise"], "QUANT"), lower=False))


# ------------------------------------------------------------------ E17 sanity
def sanity():
    if SAN is None:
        return
    e = SAN["exchangeability"]
    put("ExchN", str(e["n"])); put("ExchReject", str(e["n_reject_holm"])); put("ExchKS", num(e["median_ks"], 2))
    put("ExchQNineTest", num(e["median_q90_test"], 2)); put("ExchQNineLoo", num(e["median_q90_loo"], 2))
    c = SAN["coverage_spread"]
    put("CovSDObs", num(c["observed_sd"])); put("CovSDSim", num(c["sim_sd_median"])); put("CovSDSimHi", num(c["sim_sd_q975"]))
    put("CovOutside", pct(c["share_outside_binomial_95"], 0))
    for st, t in (("clean", "Clean"), ("faults", "Faults")):
        r = SAN["resample_noise"][st]
        put(f"Noise{t}Beyond", pct(r["share_beyond_2se"], 0)); put(f"Noise{t}Loom", pct(r["share_loom_better_beyond_2se"], 0))
        put(f"Noise{t}PreVal", pct(r["share_preval_better_beyond_2se"], 0))


# ------------------------------------------------------------------ tables
def latex_table(df, name, colspec=None):
    cols = list(df.columns)
    colspec = colspec or ("l" + "r" * (len(cols) - 1))
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(v) for v in r) + " \\\\" for r in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"tab_{name}.tex").write_text("\n".join(lines) + "\n")


def tables():
    def comp(key, metrics, methods, name):
        rows = []
        for m in methods:
            row = {"Method": NAMES[m]}
            for met, lab in metrics:
                v = S[key][met]["medians"].get(m)
                row[lab] = "--" if v is None else num(v)
            n = S[key]["nll"].get("n_per_method", {}).get(m) if isinstance(S[key]["nll"], dict) else None
            if m == "LOOM":
                row["W/D/L"] = "--"; row["$p$"] = "--"
            else:
                r = find(S[key]["nll"]["pairwise"], m)
                row["W/D/L"] = f"{r['ref_wins']}/{r['draws']}/{r['ref_losses']}"; row["$p$"] = pv(r["p_holm"])
            rows.append(row)
        latex_table(pd.DataFrame(rows), name)
    base = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "Ridge-Isotonic-CV5", "LogReg-CV", "QUANT", "Ridge-softmax", "Ridge-onehot"]
    comp("E1_clean", [("nll", "NLL"), ("ece", "ECE"), ("brier", "Brier"), ("acc", "Acc.")], base, "clean")
    comp("E4_shifted", [("nll", "NLL"), ("ece", "ECE"), ("acc", "Acc."), ("shift_auroc", "AUROC")], base, "faults")
    comp("E2b_raw_shifted", [("nll", "NLL"), ("brier", "Brier"), ("acc", "Acc."), ("shift_auroc", "AUROC")],
         ["LOOM", "LOOM-Ens5", "PreVal", "PreVal-Ens5", "QUANT", "Arsenal"], "official")
    E12 = S["E12_real_noise"]; rows = []
    for m in ("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"):
        rows.append({"Method": NAMES[m], "Clean NLL": num(E12["clean_nll_medians"][m]), "Noisy NLL": num(E12["metrics"]["nll"]["medians"][m]),
                     "AUROC": num(E12["metrics"]["shift_auroc"]["medians"][m]), "Acc.": num(E12["metrics"]["acc"]["medians"][m]),
                     "Caught": pct(E12["stream"]["fault_caught@20"][m], 0), "Auto err.": pct(E12["stream"]["auto_err@20"][m])})
    latex_table(pd.DataFrame(rows), "realnoise")
    E13 = S["E13_case_confirm"]["metrics"]; E11 = S["E11_case_study"]["metrics"]; rows = []
    for m in ("LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"):
        rows.append({"Method": NAMES[m], "Caught (expl.)": pct(E11["stream:fault_caught@20"]["medians"][m], 0),
                     "Auto err. (expl.)": pct(E11["stream:auto_err@20"]["medians"][m]),
                     "Caught (conf.)": pct(E13["stream:fault_caught@20"]["medians"][m], 0),
                     "Auto err. (conf.)": pct(E13["stream:auto_err@20"]["medians"][m]),
                     "Auto err., no faults": pct(E13["clean:auto_err@20"]["medians"][m])})
    latex_table(pd.DataFrame(rows), "triage")
    cp = {(r["method"], bool(r["shifted"])): r for r in S["E9_conformal"]}
    lab = {"CP-LOOM-jackknife+": "LOOM, jackknife+, LAC", "CP-LOOM-jackknife": "LOOM, jackknife, LAC", "CP-LOOM-jackknife+-APS": "LOOM, jackknife+, APS",
           "CP-PreVal-jackknife+": "PreVal, jackknife+, LAC", "CP-PreVal-jackknife+-APS": "PreVal, jackknife+, APS",
           "CP-Split-TS": "Split conformal, LAC", "CP-Split-TS-APS": "Split conformal, APS"}
    rows = []
    for m, l in lab.items():
        a, b = cp[(m, False)], cp[(m, True)]
        rows.append({"Method": l, "Cov. clean": num(a["cov_mean"]), "Size clean": num(a["size_mean"], 2), "Empty clean": pct(a["empty_mean"]),
                     "Cov. faults": num(b["cov_mean"]), "Size faults": num(b["size_mean"], 2), "Empty faults": pct(b["empty_mean"])})
    latex_table(pd.DataFrame(rows), "conformal")
    e = {r["method"]: r for r in S["E6_efficiency"]}; rows = []
    for m in ("MiniRocket-transform", "Ridge-onehot", "PreVal", "LOOM", "Holdout-TS", "Ridge-Platt-CV5", "LogReg-CV", "LOOM-Ens5", "QUANT", "Arsenal"):
        rows.append({"Method": NAMES[m], "Median fit (s)": num(e[m]["fit_median"]), "Mean fit (s)": num(e[m]["fit_mean"], 2),
                     "Median predict (s)": "--" if pd.isna(e[m]["pred_median"]) else num(e[m]["pred_median"], 2)})
    latex_table(pd.DataFrame(rows), "cost")
    # dataset summary table (supplement): per type
    info = pd.read_csv(ROOT / "tables" / "table_by_type.csv") if (ROOT / "tables" / "table_by_type.csv").exists() else None
    if info is not None:
        latex_table(info.round(3), "by_type")


def main():
    general(); main_tables(); official(); ablation(); generality(); errors(); conformal_cost(); application(); sanity(); tables()
    lines = ["% Generated by code/experiments/aggregate.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())]
    (TAB / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(MAC)} macros, {len(list(TAB.glob('tab_*.tex')))} tables")


if __name__ == "__main__":
    main()
