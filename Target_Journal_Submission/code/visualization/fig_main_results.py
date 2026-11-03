"""Fig. 2: main result. Mean over datasets with 95 % bootstrap CIs (1,000 resamples) of NLL on clean
data, NLL averaged over the 18 simulated fault conditions, and fault-detection AUROC (E1, E4)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.visualization.plots import INK2, PAGE_W, STYLE, _save, _style   # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402

ORDER = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "Ridge-Isotonic-CV5", "LogReg-CV", "QUANT", "Ridge-softmax"]
LABEL = {"LOOM": "LOOM (ours)", "PreVal": "PreVal", "Holdout-TS": "Hold-out TS", "Ridge-Platt-CV5": "Platt (5-fold)",
         "Ridge-Isotonic-CV5": "Isotonic (5-fold)", "LogReg-CV": "Logistic reg. (CV)", "QUANT": "QUANT", "Ridge-softmax": "Softmax (T = 1)"}


def main():
    _style()
    CI = json.load(open(ROOT / "results" / "statistics.json"))["CI"]
    panels = [("clean_nll", "NLL, clean data (lower is better)", True), ("faults_nll", "NLL under faults (lower is better)", True),
              ("faults_auroc", "Fault-detection AUROC (higher is better)", False)]
    fig, axs = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    y = list(range(len(ORDER)))[::-1]
    for ax, (key, title, logx) in zip(axs, panels):
        for yy, m in zip(y, ORDER):
            d = CI[key].get(m)
            if d is None:
                continue
            col = STYLE[m][0] if m in ("LOOM", "PreVal", "QUANT") else "#888888"
            ax.errorbar(d["mean"], yy, xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]], fmt=STYLE[m][1], color=col,
                        ecolor=col, elinewidth=1, capsize=2, ms=4)
        if logx:
            from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
            ax.set_xscale("log")
            ticks = [0.3, 0.5, 0.7, 1.0] if key == "clean_nll" else [1, 2, 3, 5]
            ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
        ax.set_title(title)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LABEL[m] for m in ORDER])
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_main_results")


if __name__ == "__main__":
    main()
