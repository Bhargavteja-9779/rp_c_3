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
        return f"not significantly different, with {r['ref_wins']} wins and {r['ref_losses']} losses of {n} {unit}, {pv(p)}"
    k = r["ref_wins"] if better else r["ref_losses"]
    return f"{'better' if better else 'worse'} on {k} of {n} {unit}, {pv(p)}"


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
    put("CmpSetSizeSplit", f"smaller on {r['ref_wins']} of {r['ref_wins'] + r['draws'] + r['ref_losses']} datasets, {pv(r['p_holm'])}")
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
def latex_table(df, name, colspec=None, header_top=None):
    cols = list(df.columns)
    colspec = colspec or ("l" + "r" * (len(cols) - 1))
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"] + ([header_top] if header_top else []) + [" & ".join(c.strip() for c in cols) + " \\\\", "\\midrule"]
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
        rows.append({"Method": NAMES[m], "Caught": pct(E11["stream:fault_caught@20"]["medians"][m], 0),
                     "Auto err.": pct(E11["stream:auto_err@20"]["medians"][m]),
                     "Caught ": pct(E13["stream:fault_caught@20"]["medians"][m], 0),
                     "Auto err. ": pct(E13["stream:auto_err@20"]["medians"][m]),
                     "No faults": pct(E13["clean:auto_err@20"]["medians"][m])})
    latex_table(pd.DataFrame(rows), "triage", header_top="& \\multicolumn{2}{c}{Exploratory (12)} & \\multicolumn{3}{c}{Confirmatory (\\CNDatasets)} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-6}")
    cp = {(r["method"], bool(r["shifted"])): r for r in S["E9_conformal"]}
    lab = {"CP-LOOM-jackknife+": "LOOM, jackknife+, LAC", "CP-LOOM-jackknife": "LOOM, jackknife, LAC", "CP-LOOM-jackknife+-APS": "LOOM, jackknife+, APS",
           "CP-PreVal-jackknife+": "PreVal, jackknife+, LAC", "CP-PreVal-jackknife+-APS": "PreVal, jackknife+, APS",
           "CP-Split-TS": "Split conformal, LAC", "CP-Split-TS-APS": "Split conformal, APS"}
    rows = []
    for m, l in lab.items():
        a, b = cp[(m, False)], cp[(m, True)]
        rows.append({"Method": l, "Coverage": num(a["cov_mean"]), "Size": num(a["size_mean"], 2), "Empty": pct(a["empty_mean"]),
                     "Coverage ": num(b["cov_mean"]), "Size ": num(b["size_mean"], 2), "Empty ": pct(b["empty_mean"])})
    latex_table(pd.DataFrame(rows), "conformal", header_top="& \\multicolumn{3}{c}{Clean} & \\multicolumn{3}{c}{Faults} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    e = {r["method"]: r for r in S["E6_efficiency"]}; rows = []
    for m in ("MiniRocket-transform", "Ridge-onehot", "PreVal", "LOOM", "Holdout-TS", "Ridge-Platt-CV5", "LogReg-CV", "LOOM-Ens5", "QUANT", "Arsenal"):
        rows.append({"Method": NAMES[m], "Median fit (s)": num(e[m]["fit_median"]), "Mean fit (s)": num(e[m]["fit_mean"], 2),
                     "Median predict (s)": "--" if pd.isna(e[m]["pred_median"]) else num(e[m]["pred_median"], 2)})
    latex_table(pd.DataFrame(rows), "cost")
    # dataset summary table (supplement): per type
    info = pd.read_csv(ROOT / "tables" / "table_by_type.csv") if (ROOT / "tables" / "table_by_type.csv").exists() else None
    if info is not None:
        latex_table(info.round(3), "by_type")


def tool():
    t = json.load(open(RES / "e18" / "summary.json"))
    put("ToolN", count(len(t["datasets"]))); put("ToolIdentical", count(sum(r["identical_predictions"] for r in t["datasets"])))
    put("ToolMaxDiff", f"{t['max_confidence_difference']:.1e}".replace("e-", "\\times10^{-").rstrip() + "}")
    put("ToolDatasets", ", ".join(r["dataset"] for r in t["datasets"]))


def prevalence():
    e = json.load(open(RES / "e19" / "summary.json"))
    for q, t in (("0.1", "Ten"), ("0.2", "Twenty")):
        for met, nm, lower in (("caught", "Caught", False), ("auto_err", "Err", True)):
            put(f"Prev{t}{nm}Loom", pct(e[q][met]["medians"]["LOOM"], 0 if nm == "Caught" else 1))
            put(f"Prev{t}{nm}PreVal", pct(e[q][met]["medians"]["PreVal"], 0 if nm == "Caught" else 1))
            put(f"CmpPrev{t}{nm}PreVal", cmp_phrase(find(e[q][met]["pairwise"], "PreVal"), lower=lower))


def hypotheses_table():
    """Main-text table of the pre-specified hypotheses with their outcome (computed from the tests)."""
    def res(r, lower=True, unit="datasets"):
        n = r["ref_wins"] + r["draws"] + r["ref_losses"]
        better = (r["median_diff"] > 0) if lower else (r["median_diff"] < 0)
        k = r["ref_wins"] if better else r["ref_losses"]
        return f"{'LOOM' if better else 'competitor'} better on {k}/{n}, {pv(r['p_holm'])}", (r["p_holm"] < 0.05), better
    rows = []
    def add(hid, hyp, metric, outcome, passed):
        rows.append({"ID": hid, "Hypothesis": hyp, "Metric": metric, "Result": outcome, "Outcome": "pass" if passed else "\\textbf{fail}"})
    t, sig, b = res(pw("E1_clean", "nll", "PreVal"))
    others = [pw("E1_clean", "nll", c) for c in ("PreVal", "Holdout-TS", "Ridge-Platt-CV5", "LogReg-CV")]
    ok = not any(r["p_holm"] < 0.05 and r["median_diff"] < 0 for r in others)
    add("H1a", "Clean NLL not worse than PreVal, hold-out TS, Platt, LogReg", "NLL", f"vs PreVal: {cmp_phrase(pw('E1_clean','nll','PreVal'))}", ok)
    r1, r2 = pw("E1_clean", "ece", "PreVal"), pw("E1_clean", "brier", "PreVal")
    add("H1b", "Clean ECE and Brier not worse than PreVal", "ECE, Brier", f"ECE: LOOM {cmp_phrase(r1)}", not ((r1["p_holm"] < 0.05 and r1["median_diff"] < 0) or (r2["p_holm"] < 0.05 and r2["median_diff"] < 0)))
    r = pw("E1_clean", "acc", "PreVal")
    add("H1c", "Accuracy unchanged vs PreVal", "accuracy", cmp_phrase(r, lower=False), r["p_holm"] >= 0.05)
    comps = ["PreVal", "Holdout-TS", "Ridge-Platt-CV5", "Ridge-Isotonic-CV5", "LogReg-CV", "Ridge-softmax", "Ridge-onehot"]
    ok = all(pw("E4_shifted", "nll", c)["p_holm"] < 0.05 and pw("E4_shifted", "nll", c)["median_diff"] > 0 for c in comps)
    add("H2a", "Faults: NLL lower than every closed-form / CV readout", "NLL, 18 conditions", f"vs PreVal: {res(pw('E4_shifted','nll','PreVal'))[0]}; all {len(comps)} readouts", ok)
    r = pw("E4_shifted", "shift_auroc", "PreVal")
    add("H2b", "Faults: better detection of faulty inputs than PreVal", "AUROC", res(r, lower=False)[0], r["p_holm"] < 0.05 and r["median_diff"] < 0)
    cpd = {(x["method"], bool(x["shifted"])): x for x in S["E9_conformal"]}
    cov = cpd[("CP-LOOM-jackknife+", False)]["cov_mean"]; r = find(S["E9_setsize_test_CP-LOOM-jackknife+"], "CP-Split-TS")
    add("H3", "Jackknife+: coverage 0.88--0.92, smaller sets than split", "coverage, size", f"coverage {cov:.3f}; smaller on {r['ref_wins']}/{r['ref_wins']+r['draws']+r['ref_losses']}, {pv(r['p_holm'])}", 0.88 <= cov <= 0.92 and r["p_holm"] < 0.05)
    r = find(S["E12_real_noise"]["metrics"]["nll"]["pairwise_cells"], "PreVal")
    add("H4", "Real ECG noise: NLL lower than PreVal", "NLL, 63 cells", res(r)[0], r["p_holm"] < 0.05 and r["median_diff"] > 0)
    h8 = {(x["metric"], x["competitor"]): x["p_holm8"] for x in S["E13_case_confirm"]["primary_holm8"]}
    for hid, met, hyp, lower in (("H5", "fault_caught@20", "Triage: more faulty recordings caught than PreVal", False),
                                 ("H6", "auto_err@20", "Triage: fewer automatic errors than PreVal", True)):
        r = dict(find(S["E13_case_confirm"]["metrics"][f"stream:{met}"]["pairwise"], "PreVal")); r["p_holm"] = h8[(met, "PreVal")]
        better = (r["median_diff"] > 0) if lower else (r["median_diff"] < 0)
        add(hid, hyp, "20\\,\\% budget", res(r, lower)[0], r["p_holm"] < 0.05 and better)
    r = find(S["E14_quant_features"]["metrics"]["shifted:nll"]["pairwise"], "PreVal-QF")
    add("H7", "QUANT features: NLL under faults lower than PreVal", "NLL, severity 3", res(r)[0], r["p_holm"] < 0.05 and r["median_diff"] > 0)
    if E20 is not None:
        h8, h9 = E20["clean"]["H8_seed0"], E20["clean"]["H9_seed0"]
        add("H8", "PTB-XL: LOOM's queue holds more annotated-noisy records than PreVal's", "20\\,\\% budget",
            f"{100*h8['caught_loom']:.1f}\\,\\% vs {100*h8['caught_preval']:.1f}\\,\\%; discordant {h8['loom_only']}/{h8['preval_only']}, McNemar {pv(h8['p'])}", h8["passed"])
        add("H9", "PTB-XL: NLL on annotated-noisy records lower than PreVal", "NLL",
            f"LOOM $-$ PreVal $={h9['mean_diff']:+.3f}$, 95\\,\\% CI [{h9['ci_low']:.3f}, {h9['ci_high']:.3f}] ({'LOOM worse' if h9['ci_low'] > 0 else 'LOOM better' if h9['ci_high'] < 0 else 'no difference'})", h9["passed"])
    if E21 is not None:
        rq, rl = find(E21["H10"]["pairwise"], "QUANT"), find(E21["H10"]["pairwise"], "LOOM")
        def hres(r):
            n = r["ref_wins"] + r["draws"] + r["ref_losses"]
            return f"{r['ref_wins']}/{n}, {pv(r['p_holm'])}"
        add("H10", "Hybrid: fewer automatic errors than QUANT and than LOOM", "20\\,\\% budget",
            f"better than QUANT on {hres(rq)}; than LOOM on {hres(rl)}", E21["H10"]["passed"])
    latex_table(pd.DataFrame(rows), "hypotheses", colspec="P{0.75cm}P{4.0cm}P{1.6cm}P{4.6cm}l")
    put("NHypPass", count(sum(1 for x in rows if x["Outcome"] == "pass"))); put("NHyp", count(len(rows)))
    put("HypFailed", ", ".join(x["ID"] for x in rows if x["Outcome"] != "pass") or "none")


def supplement_tables():
    for key, name, mets in (("E3_ablation_clean", "ablation_clean", [("nll", "NLL"), ("ece", "ECE")]),
                            ("E3_ablation_shifted", "ablation_faults", [("nll", "NLL"), ("shift_auroc", "AUROC")])):
        rows = []
        for m, v in sorted(S[key]["nll"]["medians"].items(), key=lambda kv: kv[1]):
            row = {"Variant": m}
            for met, lab in mets:
                row[lab] = num(S[key][met]["medians"][m])
            if m == "LOOM":
                row["W/D/L"], row["$p$"] = "--", "--"
            else:
                r = find(S[key]["nll"]["pairwise"], m); row["W/D/L"] = f"{r['ref_wins']}/{r['draws']}/{r['ref_losses']}"; row["$p$"] = pv(r["p_holm"])
            rows.append(row)
        latex_table(pd.DataFrame(rows), name)
    rows = [{"Fraction": f"{int(r['train_frac']*100)}\\,\\%", "Data": r["cond"], "Competitor": r["competitor"], "$n$": r["n"],
             "LOOM": num(r["median_loom"]), "Competitor NLL": num(r["median_comp"]), "W/D/L": r["wdl"], "$p$": pv(r["p_holm"])} for r in S["E7_train_size"]]
    latex_table(pd.DataFrame(rows), "trainsize")
    rows = []
    for ext in ("multirocket", "hydra"):
        for st in ("clean", "shifted"):
            for r in S[f"E5_{ext}_{st}"]["nll"]["pairwise"]:
                rows.append({"Features": ext, "Data": "clean" if st == "clean" else "severity 3", "Competitor": r["competitor"],
                             "W/D/L": f"{r['ref_wins']}/{r['draws']}/{r['ref_losses']}", "$p$": pv(r["p_holm"])})
    latex_table(pd.DataFrame(rows), "extractors")
    m = NS["means"]; rows = [{"Method": k, "NLL in-domain": num(v["nll_in"]), "NLL other axis": num(v["nll_shift"]), "Acc. other axis": num(v["acc_shift"]),
                              "AUROC": num(v["shift_auroc"])} for k, v in m.items()]
    latex_table(pd.DataFrame(rows), "naturalshift")
    rows = [{"Dataset": r["dataset"], "NLL default": num(r["nll_default"]), "NLL long": num(r["nll_long"]), "C default": r["C_default"], "C long": r["C_long"]} for r in LR]
    latex_table(pd.DataFrame(rows), "logreg")
    lk = pd.read_csv(RES / "e00" / "leakage_audit.csv")
    lk = lk[(lk.test_in_train > 0) | (lk.near_dup_test_in_train > 0)][["dataset", "n_train", "n_test", "test_in_train", "near_dup_test_in_train", "test_in_train_label_conflict"]]
    lk.columns = ["Dataset", "Train", "Test", "Exact dup.", "Near dup.", "Label conflicts"]
    latex_table(lk, "leakage")



# ------------------------------------------------------------------ E20 (PTB-XL) and E21 (hybrid); second revision
E20 = json.load(open(RES / "e20" / "summary.json")) if (RES / "e20" / "summary.json").exists() else None
E21 = json.load(open(RES / "e21" / "summary.json")) if (RES / "e21" / "summary.json").exists() else None
PTB_METHODS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"]


def ptbxl():
    if E20 is None:
        return
    d = E20["data"]
    put("PTBN", f"{d['n_records']:,}".replace(",", "{,}")); put("PTBNoisy", f"{d['n_noisy']:,}".replace(",", "{,}"))
    put("PTBTrainClean", f"{d['n_train_clean']:,}".replace(",", "{,}")); put("PTBTrainAll", f"{d['n_train_all']:,}".replace(",", "{,}"))
    put("PTBTest", f"{d['n_test']:,}".replace(",", "{,}")); put("PTBTestNoisy", str(d["n_test_noisy"]))
    put("PTBPatientsAcross", count(d["patients_in_more_than_one_fold"]))
    put("PTBNoisyShareTest", pct(d["noisy_share_test"])); put("PTBNoisyShareTrain", pct(d["noisy_share_train"]))
    for tr, T in (("clean", ""), ("all", "All")):
        if tr not in E20:
            continue
        S20 = E20[tr]
        for m in PTB_METHODS:
            M = S20["methods"][m]; k = METHODS[m]
            put(f"PTB{T}Acc{k}", pct(M["acc"]["mean"])); put(f"PTB{T}MacroAUROC{k}", num(M["macro_auroc"]["mean"]))
            put(f"PTB{T}NLLNoisy{k}", num(M["nll_noisy"]["mean"])); put(f"PTB{T}NLLClean{k}", num(M["nll_clean"]["mean"]))
            put(f"PTB{T}Det{k}", num(M["detect_auroc"]["mean"])); put(f"PTB{T}Caught{k}", pct(M["caught@20"]["mean"]))
            put(f"PTB{T}Err{k}", pct(M["auto_err@20"]["mean"])); put(f"PTB{T}ErrNoisy{k}", pct(M["auto_err_noisy@20"]["mean"]))
            put(f"PTB{T}AccNoisy{k}", pct(M["acc_noisy"]["mean"])); put(f"PTB{T}AccClean{k}", pct(M["acc_clean"]["mean"]))
            put(f"PTB{T}CaughtRange{k}", f"{100*M['caught@20']['min']:.1f}--{100*M['caught@20']['max']:.1f}\\,\\%")
        for rule, R in (("primary", "Hyb"), ("secondary", "HybMin")):
            H = S20["hybrid"][rule]
            put(f"PTB{T}{R}Caught", pct(H["caught@20"]["mean"])); put(f"PTB{T}{R}Err", pct(H["auto_err@20"]["mean"]))
        h8, h9 = S20["H8_seed0"], S20["H9_seed0"]
        put(f"PTB{T}HEightLoomOnly", str(h8["loom_only"])); put(f"PTB{T}HEightPreValOnly", str(h8["preval_only"]))
        put(f"PTB{T}HEightBoth", str(h8["both"])); put(f"PTB{T}HEightP", pv(h8["p"]))
        put(f"PTB{T}HEightCaughtLoom", pct(h8["caught_loom"])); put(f"PTB{T}HEightCaughtPreVal", pct(h8["caught_preval"]))
        put(f"PTB{T}HEightSeedsPass", count(sum(x["passed"] for x in S20["H8_all_seeds"])))
        dif = [100 * (x["caught_loom"] - x["caught_preval"]) for x in S20["H8_all_seeds"]]
        put(f"PTB{T}HEightDiffRange", f"{min(dif):.1f}--{max(dif):.1f}")
        put(f"PTB{T}HEightPRange", f"{min(x['p'] for x in S20['H8_all_seeds']):.3f}--{max(x['p'] for x in S20['H8_all_seeds']):.3f}")
        nd = [x["mean_diff"] for x in S20["H9_all_seeds"]]
        put(f"PTB{T}HNineDiffRange", f"{min(nd):.3f}--{max(nd):.3f}")
        put(f"PTB{T}HNineDiff", num(h9["mean_diff"])); put(f"PTB{T}HNineCI", f"[{h9['ci_low']:.3f}, {h9['ci_high']:.3f}]")
        put(f"PTB{T}HNineLoom", num(h9["nll_noisy_loom"])); put(f"PTB{T}HNinePreVal", num(h9["nll_noisy_preval"]))
        put(f"PTB{T}HNineSeedsPass", count(sum(x["passed"] for x in S20["H9_all_seeds"])))
        put(f"PTB{T}NSeeds", count(S20["n_seeds"]))
    put("PTBTimeQuant", f"{E20['clean']['methods']['QUANT']['time_s']/60:.0f}")
    put("PTBTimeLoom", f"{E20['clean']['methods']['LOOM']['time_s']:.0f}")
    S20 = E20["clean"]; rows = []
    for m in PTB_METHODS:
        M = S20["methods"][m]
        rows.append({"Method": NAMES[m], "Acc.": pct(M["acc"]["mean"]), "AUROC": num(M["macro_auroc"]["mean"]),
                     "Clean": num(M["nll_clean"]["mean"]), "Noisy": num(M["nll_noisy"]["mean"]),
                     "Det. AUROC": num(M["detect_auroc"]["mean"]), "Caught": pct(M["caught@20"]["mean"]),
                     "Auto err.": pct(M["auto_err@20"]["mean"])})
    for rule, nm in (("primary", "Hybrid"), ("secondary", "Hybrid-min")):
        H = S20["hybrid"][rule]
        rows.append({"Method": nm, "Acc.": "--", "AUROC": "--", "Clean": "--", "Noisy": "--", "Det. AUROC": "--",
                     "Caught": pct(H["caught@20"]["mean"]), "Auto err.": pct(H["auto_err@20"]["mean"])})
    latex_table(pd.DataFrame(rows), "ptbxl", header_top="& \\multicolumn{2}{c}{Diagnosis} & \\multicolumn{2}{c}{NLL} & & \\multicolumn{2}{c}{Triage, 20\\,\\% budget} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}")
    if "all" in E20:
        S20 = E20["all"]; rows = []
        for m in PTB_METHODS:
            M = S20["methods"][m]
            rows.append({"Method": NAMES[m], "Acc.": pct(M["acc"]["mean"]), "NLL noisy": num(M["nll_noisy"]["mean"]),
                         "Det. AUROC": num(M["detect_auroc"]["mean"]), "Caught": pct(M["caught@20"]["mean"]),
                         "Auto err.": pct(M["auto_err@20"]["mean"])})
        latex_table(pd.DataFrame(rows), "ptbxl_all")


def hybrid():
    if E21 is None:
        return
    H = E21["E13"]
    for met, nm, lower in (("auto_err@20", "Err", True), ("fault_caught@20", "Caught", False)):
        for m, k in (("Hybrid", "Hyb"), ("Hybrid-min", "HybMin"), ("LOOM", "Loom"), ("QUANT", "Quant")):
            put(f"HTen{nm}{k}", pct(H[met]["medians"][m]))
        for c, k in (("QUANT", "Quant"), ("LOOM", "Loom"), ("Hybrid-min", "HybMin")):
            put(f"CmpHTen{nm}{k}", cmp_phrase(find(H[met]["vs_hybrid"], c), lower=lower))
    for c, k in (("QUANT", "Quant"), ("LOOM", "Loom")):
        r = find(E21["H10"]["pairwise"], c)
        put(f"CmpHTenPrimary{k}", cmp_phrase(r))
    put("HTenPassed", "passed" if E21["H10"]["passed"] else "failed")
    put("HTenCleanErrHyb", pct(H["clean:auto_err@20"]["medians"]["Hybrid"])); put("HTenCleanErrQuant", pct(H["clean:auto_err@20"]["medians"]["QUANT"]))
    for b in ("10", "30"):
        put(f"HTenErr{'Ten' if b == '10' else 'Thirty'}Hyb", pct(H[f"auto_err@{b}"]["medians"]["Hybrid"]))
        put(f"HTenErr{'Ten' if b == '10' else 'Thirty'}Quant", pct(H[f"auto_err@{b}"]["medians"]["QUANT"]))
    E = E21["E12"]
    for met, nm in (("auto_err@20", "Err"), ("fault_caught@20", "Caught")):
        for m, k in (("Hybrid", "Hyb"), ("Hybrid-min", "HybMin"), ("LOOM", "Loom"), ("QUANT", "Quant")):
            put(f"HRN{nm}{k}", pct(E[met]["medians"][m]))
    rep = max(E21["E13_reproduction"]["max_abs_diff"].values()), max(E21["E12_reproduction"]["max_abs_diff"].values())
    put("ReproMaxDiff", f"{max(rep):.1g}" if max(rep) > 0 else "0")
    rows = []
    for m, nm in (("LOOM", "LOOM"), ("QUANT", "QUANT"), ("Hybrid", "Hybrid"), ("Hybrid-min", "Hybrid-min")):
        row = {"Method": nm, "Caught": pct(H["fault_caught@20"]["medians"][m]), "Auto err.": pct(H["auto_err@20"]["medians"][m]),
               "Caught ": pct(E["fault_caught@20"]["medians"][m]), "Auto err. ": pct(E["auto_err@20"]["medians"][m])}
        if E20 is not None:
            S20 = E20["clean"]
            if m in S20["methods"]:
                row["Caught  "], row["Auto err.  "] = pct(S20["methods"][m]["caught@20"]["mean"]), pct(S20["methods"][m]["auto_err@20"]["mean"])
            else:
                hh = S20["hybrid"]["primary" if m == "Hybrid" else "secondary"]
                row["Caught  "], row["Auto err.  "] = pct(hh["caught@20"]["mean"]), pct(hh["auto_err@20"]["mean"])
        rows.append(row)
    latex_table(pd.DataFrame(rows), "hybrid", header_top="& \\multicolumn{2}{c}{Confirmatory (E13)} & \\multicolumn{2}{c}{Real ECG noise (E12)} & \\multicolumn{2}{c}{PTB-XL (E20)} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")


E22 = json.load(open(RES / "e22" / "summary.json")) if (RES / "e22" / "summary.json").exists() else None
TYPE_MAC = {"baseline_drift": "Drift", "static_noise": "Static", "burst_noise": "Burst", "electrodes_problems": "Electrode"}
STUDY_MAC = {"E11": "Expl", "E13": "Conf", "E12": "RN", "E20": "PTB"}


def realism():
    if E22 is None:
        return
    T = E22["by_type"]["mean_over_seeds"]["clean"]
    for t, k in TYPE_MAC.items():
        put(f"PTBType{k}N", str(T[t]["n"]))
        for m in ("LOOM", "PreVal", "QUANT", "Ridge-Platt-CV5"):
            put(f"PTBType{k}AUROC{METHODS[m]}", num(T[t][m]["auroc"]))
    C = E22["error_capture"]
    put("CapDiff", f"{100*C['quant_err_in_quant_queue_minus_loom_queue_median']:.1f}"); put("CapOwnBetter", str(C["own_better"]))
    put("CapN", str(C["n_datasets"])); put("CapP", pv(C["p"]))
    K = E22["patient_cluster_bootstrap"]
    put("ClusterNLLCI", f"[{K['nll_diff']['ci_low']:.3f}, {K['nll_diff']['ci_high']:.3f}]")
    put("ClusterCaughtCI", f"[{100*K['caught_diff']['ci_low']:.1f}, {100*K['caught_diff']['ci_high']:.1f}]")
    put("ClusterPatients", str(K["n_patients"]))
    L = E22["lead_coverage"]
    put("LeadMedian", count(L["median_leads"])); put("LeadShareLeThree", pct(L["share_le3"], 0))
    put("LeadShareAll", pct(L["share_all12"], 0)); put("LeadUnparsed", str(L["n_unparsed"])); put("LeadParsed", str(L["n_parsed"]))
    for r in E22["effects"]:
        k = STUDY_MAC[r["study"]]
        put(f"Eff{k}Diff", f"{100*r['diff']:.1f}"); put(f"Eff{k}CI", f"[{100*r['ci_low']:.1f}, {100*r['ci_high']:.1f}]")

def main():
    general(); main_tables(); official(); ablation(); generality(); errors(); conformal_cost(); application(); sanity(); tables(); hypotheses_table(); supplement_tables(); tool(); prevalence(); ptbxl(); hybrid(); realism()
    lines = ["% Generated by code/experiments/aggregate.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())]
    (TAB / "macros.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(MAC)} macros, {len(list(TAB.glob('tab_*.tex')))} tables")


if __name__ == "__main__":
    main()
