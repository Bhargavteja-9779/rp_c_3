"""Fig.: where leverage-aware confidence helps (post hoc summary, E22 / deviation D27).

(a) Difference in the share of faulty recordings caught at a 20 % review budget, LOOM minus PreVal, with 95 %
bootstrap CIs, for the triage studies ordered from simulated to expert-annotated real faults.
(b) PTB-XL: AUROC of low confidence for separating records with each type of annotated noise from clean records
(mean over seeds; 0.5 = chance), with the number of test records of each type.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.visualization.plots import PAGE_W, STYLE, _save, _style   # noqa: E402

LABEL = {"E11": "Simulated faults,\nexploratory (12 datasets)", "E13": "Simulated faults,\nconfirmatory (38 datasets)",
         "E12": "Recorded noise added\nto ECG (7 datasets)", "E20": "Expert-annotated noise,\n12-lead ECG (PTB-XL)"}
TYPES = {"baseline_drift": "baseline\ndrift", "static_noise": "static\nnoise", "burst_noise": "burst\nnoise",
         "electrodes_problems": "electrode\nproblems"}
METHODS = ["LOOM", "PreVal", "Ridge-Platt-CV5", "QUANT"]
MLAB = {"LOOM": "LOOM", "PreVal": "PreVal", "Ridge-Platt-CV5": "Platt (5-fold)", "QUANT": "QUANT"}


def main():
    _style()
    s = json.load(open(ROOT / "results" / "e22" / "summary.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
    eff = s["effects"]; ys = np.arange(len(eff))[::-1]
    for y, r in zip(ys, eff):
        ax.errorbar(100 * r["diff"], y, xerr=[[100 * (r["diff"] - r["ci_low"])], [100 * (r["ci_high"] - r["diff"])]],
                    fmt="o", color=STYLE["LOOM"][0], capsize=2.5, ms=4)
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([LABEL[r["study"]] for r in eff])
    ax.set_xlabel("extra faulty recordings caught by LOOM\nvs PreVal at 20 % budget (percentage points)")
    ax.set_title("(a) Gain shrinks as faults become realistic", loc="left")
    T = s["by_type"]["mean_over_seeds"]["clean"]; x = np.arange(len(TYPES)); w = 0.2
    for j, m in enumerate(METHODS):
        bx.bar(x + (j - 1.5) * w, [T[t][m]["auroc"] for t in TYPES], w, color=STYLE[m][0], label=MLAB[m])
    bx.axhline(0.5, color="#666666", lw=0.8, ls="--")
    bx.set_xticks(x); bx.set_xticklabels([f"{TYPES[t]}\n(n = {T[t]['n']})" for t in TYPES])
    bx.set_ylim(0.4, 0.8); bx.set_ylabel("AUROC, noisy vs clean records")
    bx.set_title(f"(b) PTB-XL by annotation type ({T['n_seeds']} seed{'s' if T['n_seeds'] > 1 else ''})", loc="left")
    bx.legend(loc="upper left", ncol=2)
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_realism")


if __name__ == "__main__":
    main()
