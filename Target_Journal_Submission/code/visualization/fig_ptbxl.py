"""Fig.: E20, clinical 12-lead ECG with expert-annotated noise (PTB-XL), primary analysis (training on clean records).

(a, b) Two noisy test records chosen by fixed rules (seed 0, 20 % budget): the noisy record with the smallest
ecg_id that LOOM sends to review but PreVal does not, and the noisy record with the smallest ecg_id that LOOM
does not send to review (a failure). Lead II is shown.
(c) Share of annotated-noisy records in the review queue against the budget; (d) error rate of the automatic
decisions, including the leverage-gated hybrid (E21). Curves: mean over the five seeds, band: min-max over seeds.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.visualization.plots import COL_W, PAGE_W, STYLE, _save, _style   # noqa: E402,F401
import ptbxl                                                              # noqa: E402

RES = ROOT / "results" / "e20"
BUDGETS = np.round(np.arange(0.05, 0.501, 0.05), 2)
METHODS = ["LOOM", "PreVal", "Holdout-TS", "Ridge-Platt-CV5", "QUANT"]
LABEL = {"LOOM": "LOOM", "PreVal": "PreVal", "Holdout-TS": "Hold-out TS", "Ridge-Platt-CV5": "Platt (5-fold)", "QUANT": "QUANT",
         "Hybrid": "Hybrid (QUANT label, LOOM queue)"}
HYB = ("#000000", "P", "-")


def queue(conf, b):
    k = int(round(b * len(conf))); q = np.zeros(len(conf), bool)
    q[np.argsort(conf, kind="stable")[:k]] = True
    return q


def curves(runs, label_key, conf_key):
    cau, err = [], []
    for o in runs:
        y, nz = o["y"], o["noisy"].astype(bool)
        lab, conf = o[f"P_{label_key}"].argmax(1), o[f"P_{conf_key}"].max(1)
        c, e = [], []
        for b in BUDGETS:
            q = queue(conf, b); c.append(q[nz].mean()); e.append((lab[~q] != y[~q]).mean())
        cau.append(c); err.append(e)
    return np.array(cau), np.array(err)


def main():
    _style()
    runs = [dict(np.load(p)) for p in sorted(RES.glob("clean_s*.npz"))]
    X, y, noisy, fold, ids = ptbxl.load()
    db = ptbxl.metadata()
    o = runs[0]; nz = o["noisy"].astype(bool)
    qL, qP = queue(o["P_LOOM"].max(1), 0.2), queue(o["P_PreVal"].max(1), 0.2)
    order = np.argsort(o["ecg_id"])
    pick_good = next(i for i in order if nz[i] and qL[i] and not qP[i])
    pick_fail = next(i for i in order if nz[i] and not qL[i])
    fig = plt.figure(figsize=(PAGE_W, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.25)
    for j, (i, tag) in enumerate(((pick_good, "a"), (pick_fail, "b"))):
        ax = fig.add_subplot(gs[0, j]); eid = int(o["ecg_id"][i])
        sig = X[np.where(ids == eid)[0][0], 1]
        ax.plot(np.arange(len(sig)) / 100, sig, color="#52514e", lw=0.6)
        ann = [c.replace("_", " ") for c in ptbxl.NOISE_COLS if isinstance(db.loc[eid, c], str) and db.loc[eid, c].strip()]
        outcome = "to review" if qL[i] else "decided automatically"
        ax.set_title(f"({tag}) ecg_id {eid}, lead II; annotated: {', '.join(ann)}\n"
                     f"LOOM conf. {o['P_LOOM'][i].max():.2f} ({outcome}), PreVal conf. {o['P_PreVal'][i].max():.2f}; "
                     f"true {ptbxl.CLASSES[o['y'][i]]}, LOOM {ptbxl.CLASSES[o['P_LOOM'][i].argmax()]}", loc="left", fontsize=6.5)
        ax.set_xlabel("time (s)"); ax.set_ylabel("mV")
    axc, axe = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for m in METHODS:
        c, e = curves(runs, m, m); col, mk, ls = STYLE[m]
        for ax, v in ((axc, c), (axe, e)):
            ax.plot(100 * BUDGETS, 100 * v.mean(0), color=col, marker=mk, ls=ls, label=LABEL[m])
            ax.fill_between(100 * BUDGETS, 100 * v.min(0), 100 * v.max(0), color=col, alpha=0.15, lw=0)
    c, e = curves(runs, "QUANT", "LOOM")
    axe.plot(100 * BUDGETS, 100 * e.mean(0), color=HYB[0], marker=HYB[1], ls=HYB[2], label=LABEL["Hybrid"])
    axe.fill_between(100 * BUDGETS, 100 * e.min(0), 100 * e.max(0), color=HYB[0], alpha=0.12, lw=0)
    axc.set_xlabel("review budget (%)"); axc.set_ylabel("annotated-noisy records in queue (%)")
    axe.set_xlabel("review budget (%)"); axe.set_ylabel("error of automatic decisions (%)")
    axc.set_title("(c) Noisy records sent to review", loc="left"); axe.set_title("(d) Errors among automatic decisions", loc="left")
    axc.legend(loc="lower right", ncol=1); axe.legend(loc="upper right", ncol=1)
    _save(fig, ROOT / "figures", "fig_ptbxl")
    print("examples:", int(o["ecg_id"][pick_good]), int(o["ecg_id"][pick_fail]))


if __name__ == "__main__":
    main()
