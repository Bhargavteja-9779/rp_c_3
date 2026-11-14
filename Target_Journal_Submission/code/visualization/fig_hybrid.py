"""Fig.: E21, leverage-gated hybrid triage on the confirmatory triage study (E13, 38 datasets, 30 % faults).

(a) Error rate of automatic decisions against the review budget: mean over datasets of the mean over the five
resamples, with 95 % bootstrap bands over datasets. (b) Per dataset at a 20 % budget: hybrid vs QUANT and
hybrid vs LOOM (points below the diagonal: hybrid better).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.visualization.plots import PAGE_W, STYLE, _save, _style   # noqa: E402
import case_study as cs                                              # noqa: E402
import case_study_confirm as csc                                     # noqa: E402

BUDGETS = np.round(np.arange(0.05, 0.501, 0.05), 2)
RULES = {"LOOM": STYLE["LOOM"], "QUANT": STYLE["QUANT"], "Hybrid": ("#000000", "P", "-"), "Hybrid-min": ("#999999", "x", "--")}
LABEL = {"LOOM": "LOOM", "QUANT": "QUANT", "Hybrid": "Hybrid (QUANT label, LOOM queue)", "Hybrid-min": "Hybrid-min"}


def queue(conf, b):
    k = int(round(b * len(conf))); q = np.zeros(len(conf), bool)
    q[np.argsort(conf, kind="stable")[:k]] = True
    return q


def main():
    _style()
    per = {r: [] for r in RULES}
    for dom, (dsets, faults) in csc.DOMAINS.items():
        for d in dsets:
            acc = {r: [] for r in RULES}
            for r in cs.RESAMPLES:
                o = dict(np.load(cs.OUT / f"{d}__r{r}.npz", allow_pickle=True)); y = o["y"]
                PL, PQ = o["P_stream_LOOM"], o["P_stream_QUANT"]
                spec = {"LOOM": (PL.argmax(1), PL.max(1)), "QUANT": (PQ.argmax(1), PQ.max(1)),
                        "Hybrid": (PQ.argmax(1), PL.max(1)), "Hybrid-min": (PQ.argmax(1), np.minimum(PL.max(1), PQ.max(1)))}
                for name, (lab, conf) in spec.items():
                    acc[name].append([(lab[~(q := queue(conf, b))] != y[~q]).mean() for b in BUDGETS])
            for name in RULES:
                per[name].append(np.mean(acc[name], axis=0))
    rng = np.random.default_rng(0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE_W, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for name, (col, mk, ls) in RULES.items():
        A = np.array(per[name]); m = A.mean(0)
        boot = np.array([A[rng.integers(0, len(A), len(A))].mean(0) for _ in range(1000)])
        ax.plot(100 * BUDGETS, 100 * m, color=col, marker=mk, ls=ls, label=LABEL[name])
        ax.fill_between(100 * BUDGETS, 100 * np.percentile(boot, 2.5, 0), 100 * np.percentile(boot, 97.5, 0), color=col, alpha=0.12, lw=0)
    ax.set_xlabel("review budget (%)"); ax.set_ylabel("error of automatic decisions (%)")
    ax.set_title(f"(a) {len(per['LOOM'])} sensor datasets, 30 % faulty recordings", loc="left"); ax.legend(loc="upper right")
    t = pd.read_csv(ROOT / "results" / "e21" / "per_dataset_e13.csv")
    T = t[t.stream == "stream"].pivot(index="dataset", columns="method", values="auto_err@20")
    for comp, (col, mk, _) in (("QUANT", RULES["QUANT"]), ("LOOM", RULES["LOOM"])):
        bx.scatter(100 * T[comp], 100 * T["Hybrid"], color=col, marker=mk, s=14, label=f"vs {comp}", zorder=3)
    lim = [0, 100 * max(T[["QUANT", "LOOM", "Hybrid"]].max()) * 1.05]
    bx.plot(lim, lim, color="#666666", lw=0.8, ls="--"); bx.set_xlim(lim); bx.set_ylim(lim)
    bx.set_xlabel("automatic error of QUANT or LOOM (%)"); bx.set_ylabel("automatic error of the hybrid (%)")
    bx.set_title("(b) Per dataset, 20 % budget (below diagonal: hybrid better)", loc="left"); bx.legend(loc="upper left")
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_hybrid")


if __name__ == "__main__":
    main()
