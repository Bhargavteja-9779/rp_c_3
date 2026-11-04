"""E19 (post hoc, deviation D22): sensitivity of the confirmatory triage results to the fault prevalence.

The stored per-recording outputs of E13 (30 % faulty recordings) are re-used: for a target prevalence q,
faulty recordings are randomly dropped (seed 0) until they make up q of the stream; no model is refitted.
Faults caught and automatic error at a 20 % review budget are recomputed for q in {0.1, 0.2, 0.3}.
Output: results/e19/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.statistics.tests import holm, pairwise_vs   # noqa: E402
from case_study import triage                          # noqa: E402

METHODS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"]


def main():
    dsets = sorted({r["dataset"] for r in json.load(open(ROOT / "results" / "case_study_confirm.json"))})
    rows = []
    for q in (0.1, 0.2, 0.3):
        for d in dsets:
            for r in range(5):
                o = np.load(ROOT / "results" / "case_study" / f"{d}__r{r}.npz")
                y, f = o["y"], o["is_fault"].astype(bool)
                rng = np.random.default_rng(0)
                fi, ci = np.flatnonzero(f), np.flatnonzero(~f)
                k = int(round(q / (1 - q) * len(ci)))
                keep = np.sort(np.r_[ci, rng.choice(fi, min(k, len(fi)), replace=False)])
                for m in METHODS:
                    if f"P_stream_{m}" not in o.files:
                        continue
                    P = o[f"P_stream_{m}"][keep]
                    e, c = triage(P.max(1), P.argmax(1) == y[keep], f[keep], 0.2)
                    rows.append(dict(q=q, dataset=d, resample=r, method=m, auto_err=e, caught=c))
    df = pd.DataFrame(rows)
    per = df.groupby(["q", "dataset", "method"])[["auto_err", "caught"]].mean().reset_index()
    out = {}
    for q, g in per.groupby("q"):
        res = {}
        for met, lower in (("caught", False), ("auto_err", True)):
            T = g.pivot(index="dataset", columns="method", values=met)
            pw = pairwise_vs(T, "LOOM", lower_is_better=lower)
            res[met] = dict(medians=T.median().to_dict(), pairwise=pw.to_dict(orient="records"))
        out[str(q)] = res
    (ROOT / "results" / "e19").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "e19" / "summary.json", "w"), indent=1, default=float)
    for q, r in out.items():
        pc = [x for x in r["caught"]["pairwise"] if x["competitor"] == "PreVal"][0]
        pe = [x for x in r["auto_err"]["pairwise"] if x["competitor"] == "PreVal"][0]
        print(q, "caught LOOM", round(r["caught"]["medians"]["LOOM"], 3), "PreVal", round(r["caught"]["medians"]["PreVal"], 3),
              pc["ref_wins"], pc["p_holm"], "| err LOOM", round(r["auto_err"]["medians"]["LOOM"], 3), "PreVal", round(r["auto_err"]["medians"]["PreVal"], 3), pe["ref_wins"], pe["p_holm"])


if __name__ == "__main__":
    main()
