"""Fig.: E23, CinC 2011 set A. Records graded unacceptable by the challenge annotators, models trained on PTB-XL.

(a) One record chosen by a fixed rule (seed 0, 20 % budget): the unacceptable record with the smallest record id
whose largest absolute value is below 10 mV (so the amplitude rule misses it) and that LOOM sends to review while
PreVal does not. Leads II and V2 are shown.
(b) Share of the unacceptable records in the review queue against the budget, all 998 records.
(c) The same on the records without extreme amplitude (largest absolute value < 10 mV; post hoc, deviation D29).
Curves for the E23 models: mean over the seeds, band: min-max over seeds. Deep baselines (E24) and the
hand-made amplitude rule are single curves.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.visualization.plots import PAGE_W, STYLE, _save, _style   # noqa: E402
import e23_cinc                                                    # noqa: E402

BUDGETS = np.round(np.arange(0.05, 0.501, 0.05), 2)
SEEDED = {"LOOM": "LOOM", "PreVal": "PreVal", "QUANT": "QUANT", "Leverage": "leverage alone (inside LOOM)"}
EXTRA_STYLE = {"Leverage": ("#0072B2", "o", ":"), "Deep-ensemble": ("#555555", "v", "-."),
               "Rule-amplitude": ("#999999", "x", "--")}
EXTRA_LABEL = {"Deep-ensemble": "1-D ResNet ensemble", "Rule-amplitude": "largest |x| rule (hand-made)"}


def queue(score, b):
    k = int(round(b * len(score))); q = np.zeros(len(score), bool)
    q[np.argsort(-score, kind="stable")[:k]] = True
    return q


def curve(score, unacc, mask):
    s, u = score[mask], unacc[mask]
    return np.array([100 * queue(s, b)[u].mean() for b in BUDGETS])


def style(m):
    return STYLE.get(m, EXTRA_STYLE.get(m))


def panel(ax, runs, extra, unacc, mask, title):
    for m, lab in SEEDED.items():
        C = np.array([curve(e23_cinc.scores(o)[m], unacc, mask) for o in runs])
        c, mk, ls = style(m)
        ax.plot(100 * BUDGETS, C.mean(0), color=c, marker=mk, ls=ls, label=lab)
        if len(runs) > 1:
            ax.fill_between(100 * BUDGETS, C.min(0), C.max(0), color=c, alpha=0.15, lw=0)
    for m, sc in extra.items():
        c, mk, ls = style(m)
        ax.plot(100 * BUDGETS, curve(sc, unacc, mask), color=c, marker=mk, ls=ls, label=EXTRA_LABEL[m])
    ax.plot(100 * BUDGETS, 100 * BUDGETS, color="#666666", lw=0.8, ls=(0, (1, 2)), label="random order")
    ax.set_xlabel("review budget (% of records)"); ax.set_ylabel("unacceptable records caught (%)")
    ax.set_ylim(0, 100); ax.set_title(title, loc="left")


def main():
    _style()
    runs = [dict(np.load(p)) for p in sorted((ROOT / "results" / "e23").glob("s[0-9].npz"))]
    X, unacc, rec = e23_cinc.load(); unacc = unacc.astype(bool)
    amp = np.abs(X).max(axis=(1, 2)); sub = amp < 10
    extra = {"Rule-amplitude": e23_cinc.rules(X)["Rule-amplitude"]}
    D = e23_cinc.deep_scores()
    if "Deep-ensemble" in D:
        extra = {"Deep-ensemble": D["Deep-ensemble"], **extra}
    S0 = e23_cinc.scores(runs[0])
    qL, qP = queue(S0["LOOM"], 0.2), queue(S0["PreVal"], 0.2)
    pick = np.flatnonzero(unacc & sub & qL & ~qP)
    i = pick[np.argmin(rec[pick])]

    fig = plt.figure(figsize=(PAGE_W, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[0]); t = np.arange(X.shape[-1]) / 100
    for k, (lead, j) in enumerate((("II", 1), ("V2", 7))):
        ax.plot(t, X[i, j] - 3 * k, color="#333333", lw=0.6)
    ax.set_yticks([0, -3]); ax.set_yticklabels(["II", "V2"]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)"); ax.set_xlim(0, 10)
    ax.set_title(f"(a) Record {rec[i]}: unacceptable\nconf. LOOM {runs[0]['P_LOOM'][i].max():.2f}, "
                 f"PreVal {runs[0]['P_PreVal'][i].max():.2f}", loc="left")
    bx, cx = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    panel(bx, runs, extra, unacc, np.ones(len(unacc), bool), f"(b) All records (n = {len(unacc)})")
    panel(cx, runs, extra, unacc, sub, f"(c) Largest |x| < 10 mV (n = {sub.sum()})")
    cx.set_ylabel("")
    h, l = bx.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1)); _save(fig, ROOT / "figures", "fig_cinc")
    print("example record", rec[i])


if __name__ == "__main__":
    main()
