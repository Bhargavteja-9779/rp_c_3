"""Fig. 8: qualitative examples, chosen by a fixed rule, including failures.

Data: ECG5000, official split, the real-noise triage stream of E12 (30 % of recordings with MIT-BIH NSTDB
noise; LOOM; 20 % review budget). Rule: for each of the four outcomes below, the first recording in file
order of the test set that has this outcome.
  (a) clean, decided automatically, correct          (intended behaviour)
  (b) noisy, sent to review                           (intended behaviour)
  (c) noisy, decided automatically and wrong          (failure: missed fault)
  (d) clean, sent to review                           (failure: unnecessary review)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.data.ucr import load_split                                       # noqa: E402
from loom.visualization.plots import INK, INK2, PAGE_W, STYLE, _save, _style  # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402

NAME, R, BUDGET = "ECG5000", 0, 0.2


def main():
    _style()
    import real_noise as rn
    noise = {n: rn.read_212(n) for n in rn.NOISES}
    _, _, Xte, yte = load_split(NAME, R)
    conds, _, Xs, is_f, pick = rn.build_stream(NAME, R, Xte, noise)
    o = np.load(ROOT / "results" / "real_noise" / f"{NAME}__r{R}.npz")
    assert np.array_equal(o["is_fault"], is_f)
    P = o["P_stream_LOOM"]; y = o["y"]; conf = P.max(1); pred = P.argmax(1)
    k = int(round(BUDGET * len(y))); review = np.zeros(len(y), bool); review[np.argsort(conf, kind="stable")[:k]] = True
    rules = [("(a) clean, automatic,\ncorrect", ~is_f & ~review & (pred == y)),
             ("(b) noisy,\nsent to review", is_f & review),
             ("(c) failure: noisy,\nautomatic and wrong", is_f & ~review & (pred != y)),
             ("(d) failure: clean,\nsent to review", ~is_f & review)]
    fig, axs = plt.subplots(1, 4, figsize=(PAGE_W, 2.1), sharey=True)
    for ax, (title, mask) in zip(axs, rules):
        i = int(np.flatnonzero(mask)[0])
        if is_f[i]:
            nz, snr = conds[pick[i]]
            ax.plot(Xte[i], color="#BBBBBB", lw=0.8)
            ax.plot(Xs[i], color=STYLE["PreVal"][0], lw=0.9)
            sub = f"{ {'bw': 'baseline wander', 'em': 'electrode motion', 'ma': 'muscle'}[nz]}, {snr} dB"
        else:
            ax.plot(Xs[i], color=INK, lw=0.9)
            sub = "no noise added"
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(f"test #{i}: {sub}\ntrue {y[i] + 1}, predicted {pred[i] + 1}, confidence {conf[i]:.2f}", fontsize=6.5, color=INK2)
        ax.set_xticks([]); ax.grid(False)
    axs[0].set_ylabel("ECG (z-normalised)")
    fig.tight_layout(); _save(fig, ROOT / "figures", "fig_qualitative")


if __name__ == "__main__":
    main()
