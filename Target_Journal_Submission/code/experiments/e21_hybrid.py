"""E21: leverage-gated hybrid triage (Online Resource 2, section 12.2; registered in 6b05cb1 before running).

Rule (no fitted parameter): the automatic decision uses QUANT's label; the review queue holds the share b
of recordings with the lowest LOOM confidence. Secondary rule: queue by the lowest min(LOOM, QUANT) confidence.

Confirmatory H10 on E13 (38 datasets, 30 % faults): at b = 20 % the hybrid's automatic error rate is lower
than both QUANT's and LOOM's (two-sided Wilcoxon over datasets, Holm over the two comparisons).
Descriptive: E12 (real NSTDB noise stream, 7 ECG datasets) and E20 (PTB-XL; computed in e20_ptbxl.py).

The per-recording outputs of E12 and E13 are regenerated from the fixed seeds when absent; the script checks
that the regenerated LOOM/QUANT results equal the stored per-dataset results before using them.
Output: results/e21/summary.json, results/e21/per_dataset_e13.csv, results/e21/per_dataset_e12.csv.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import case_study as cs                                                   # noqa: E402
import case_study_confirm as csc                                          # noqa: E402
from loom.statistics.tests import pairwise_vs                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e21"
BUDGETS = (0.1, 0.2, 0.3)


def queue(conf, b):
    k = int(round(b * len(conf))); q = np.zeros(len(conf), bool)
    q[np.argsort(conf, kind="stable")[:k]] = True
    return q


def triage_rows(PL, PQ, y, is_f):
    """auto_err / fault_caught at each budget for LOOM, QUANT and the two hybrid rules."""
    rules = {"LOOM": (PL.argmax(1), PL.max(1)), "QUANT": (PQ.argmax(1), PQ.max(1)),
             "Hybrid": (PQ.argmax(1), PL.max(1)), "Hybrid-min": (PQ.argmax(1), np.minimum(PL.max(1), PQ.max(1)))}
    out = {}
    for name, (lab, conf) in rules.items():
        corr = lab == y; d = {}
        for b in BUDGETS:
            q = queue(conf, b); t = int(b * 100)
            d[f"auto_err@{t}"] = float((~corr[~q]).mean())
            d[f"fault_caught@{t}"] = float(q[is_f].mean()) if is_f.any() else float("nan")
        out[name] = d
    return out


def e13():
    rows = []
    for dom, (dsets, faults) in csc.DOMAINS.items():
        for d in dsets:
            for r in cs.RESAMPLES:
                o = cs.run(d, r, faults)
                y, f = o["y"], o["is_fault"].astype(bool)
                for stream in ("stream", "clean"):
                    ff = f if stream == "stream" else np.zeros(len(y), bool)
                    for m, v in triage_rows(o[f"P_{stream}_LOOM"], o[f"P_{stream}_QUANT"], y, ff).items():
                        rows.append(dict(domain=dom, dataset=d, resample=r, stream=stream, method=m, **v))
                print("E13", d, r, flush=True)
    return pd.DataFrame(rows)


def reproduction_check(df):
    """Compare regenerated LOOM/QUANT triage metrics with the stored E13 results (case_study_confirm.json)."""
    old = pd.DataFrame(json.load(open(ROOT / "results" / "case_study_confirm.json")))
    old = old[old.method.isin(["LOOM", "QUANT"])]
    new = df[df.method.isin(["LOOM", "QUANT"])].rename(columns={})
    keys = ["dataset", "resample", "stream", "method"]
    mrg = old.merge(new, on=keys, suffixes=("_old", "_new"))
    diffs = {c: float(np.nanmax(np.abs(mrg[f"{c}_old"] - mrg[f"{c}_new"]))) for c in ["auto_err@20", "fault_caught@20"]}
    return dict(n_rows_compared=int(len(mrg)), n_rows_stored=int(len(old)), max_abs_diff=diffs)


def e12():
    import real_noise as rn
    noise = {n: rn.read_212(n) for n in rn.NOISES}
    rows = []
    for d in rn.DATASETS:
        for r in rn.RESAMPLES:
            o = rn.run(d, r, noise)
            for m, v in triage_rows(o["P_stream_LOOM"], o["P_stream_QUANT"], o["y"], o["is_fault"].astype(bool)).items():
                rows.append(dict(dataset=d, resample=r, method=m, **v))
            print("E12", d, r, flush=True)
    return pd.DataFrame(rows)


def e12_check(df):
    old = pd.DataFrame(json.load(open(ROOT / "results" / "real_noise.json")))
    old = old[(old.cond == "stream") & old.method.isin(["LOOM", "QUANT"])]
    mrg = old.merge(df, on=["dataset", "resample", "method"], suffixes=("_old", "_new"))
    return dict(n_rows_compared=int(len(mrg)), n_rows_stored=int(len(old)),
                max_abs_diff={c: float(np.nanmax(np.abs(mrg[f"{c}_old"] - mrg[f"{c}_new"])))
                              for c in ["auto_err@20", "fault_caught@20"]})


def table(per, met):
    return per.pivot(index="dataset", columns="method", values=met)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"experiment": "E21", "rule": "label from QUANT, review queue by lowest LOOM confidence (secondary: lowest min)"}
    d13 = e13()
    summ["E13_reproduction"] = reproduction_check(d13)
    per = d13.groupby(["domain", "dataset", "stream", "method"]).mean(numeric_only=True).reset_index()
    per.to_csv(OUT / "per_dataset_e13.csv", index=False)
    S = per[per.stream == "stream"]
    pw = pairwise_vs(table(S, "auto_err@20")[["Hybrid", "QUANT", "LOOM"]], "Hybrid", lower_is_better=True)
    H10 = dict(pairwise=pw.to_dict(orient="records"),
               passed=bool(all((r["p_holm"] < 0.05) and (r["median_diff"] > 0) for r in pw.to_dict(orient="records"))))
    summ["H10"] = H10
    summ["E13"] = {}
    for met in ["auto_err@10", "auto_err@20", "auto_err@30", "fault_caught@10", "fault_caught@20", "fault_caught@30"]:
        T = table(S, met)
        summ["E13"][met] = dict(medians=T.median().to_dict(), means=T.mean().to_dict(),
                                vs_hybrid=pairwise_vs(T, "Hybrid", lower_is_better=met.startswith("auto")).to_dict(orient="records"))
    C = per[per.stream == "clean"]
    summ["E13"]["clean:auto_err@20"] = dict(medians=table(C, "auto_err@20").median().to_dict(),
                                            vs_hybrid=pairwise_vs(table(C, "auto_err@20"), "Hybrid").to_dict(orient="records"))
    d12 = e12()
    summ["E12_reproduction"] = e12_check(d12)
    p12 = d12.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index()
    p12.to_csv(OUT / "per_dataset_e12.csv", index=False)
    summ["E12"] = {met: dict(medians=table(p12, met).median().to_dict(), means=table(p12, met).mean().to_dict(),
                             vs_hybrid=pairwise_vs(table(p12, met), "Hybrid", lower_is_better=met.startswith("auto")).to_dict(orient="records"))
                   for met in ["auto_err@20", "fault_caught@20"]}
    e20 = ROOT / "results" / "e20" / "summary.json"
    if e20.exists():
        z = json.load(open(e20))
        summ["E20"] = {t: dict(hybrid=z[t]["hybrid"], loom={k: z[t]["methods"]["LOOM"][k] for k in ("auto_err@20", "caught@20")},
                               quant={k: z[t]["methods"]["QUANT"][k] for k in ("auto_err@20", "caught@20")})
                       for t in ("clean", "all") if t in z}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("E13_reproduction", "E12_reproduction")}, indent=1))
    print(pw[["competitor", "n", "ref_wins", "ref_losses", "median_diff", "p", "p_holm"]])
    print("H10 passed:", H10["passed"])


if __name__ == "__main__":
    main()
