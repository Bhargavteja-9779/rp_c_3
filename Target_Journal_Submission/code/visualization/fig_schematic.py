"""Fig. 1: how LOOM works, drawn with real ECG200 recordings and real electrode-motion noise.

Example selection rule (fixed): the first training series of each class in file order, and the first
test series in file order; its faulty version adds MIT-BIH NSTDB electrode-motion noise at 6 dB
(seed 7). The confidence and leverage values printed in the figure are computed, not typed.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from loom.data.ucr import load_split                                       # noqa: E402
from loom.models.loom import LOOM                                          # noqa: E402
from loom.training.runner import _feats, make_transform                     # noqa: E402
from loom.visualization.plots import INK, INK2, PAGE_W, STYLE, _save, _style  # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import FancyBboxPatch                              # noqa: E402

BLUE, ORANGE = STYLE["LOOM"][0], STYLE["PreVal"][0]


def main():
    _style()
    from real_noise import add_noise, read_212
    Xtr, ytr, Xte, yte = load_split("ECG200", 0)
    tf = make_transform("minirocket", 0).fit(Xtr[:, None, :])
    m = LOOM("standard").fit(_feats(tf, Xtr), ytr)
    x_clean = Xte[:1]; x_bad = add_noise(x_clean, read_212("em"), 6, seed=7)
    vals = []
    for x in (x_clean, x_bad):
        F = _feats(tf, x)
        vals.append((float(m.predict_proba(F).max()), float(m.ridge_.leverage(F)[0] / m.h_ref_)))

    fig = plt.figure(figsize=(PAGE_W, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, title, text, col=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=col, lw=1.0))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7, weight="bold", color=INK)
        if text:
            ax.text(x + w / 2, y + 1.2, text, ha="center", va="bottom", fontsize=6.5, color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))

    def inset(rect, series, colors, lw=0.8):
        a = fig.add_axes(rect)
        for sv, c in zip(series, colors):
            a.plot(sv, color=c, lw=lw)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(False)
        return a

    # top row: training pipeline (y 21-39)
    box(1, 21, 21, 18, "Training recordings", "ECG200, 100 series,\ntwo classes")
    cls = np.unique(ytr)
    inset([0.03, 0.69, 0.17, 0.15], [Xtr[np.where(ytr == c)[0][0]] for c in cls], [BLUE, "#009E73"])
    box(26, 21, 17, 18, "Random kernels", "MiniRocket:\n9,996 features\nper recording")
    box(47, 21, 24, 18, "One thin SVD", "exact leave-one-out\nscores $z_{-i}$ and held-out\nleverages $h_{-i}=d_i/(1-d_i)$")
    box(75, 21, 24, 18, "Calibrate on LOO pairs", "choose $\\lambda$ and temperature $a$\nby LOO likelihood;\nno held-out data")
    arrow(22.5, 30, 25.5, 30); arrow(43.5, 30, 46.5, 30); arrow(71.5, 30, 74.5, 30)

    # bottom row: deployment (y 1-18)
    box(1, 1, 30, 17, "New recordings", "", col=INK2)
    inset([0.02, 0.21, 0.13, 0.14], [x_clean[0]], [INK])
    inset([0.16, 0.21, 0.13, 0.14], [x_bad[0]], [ORANGE])
    ax.text(8.6, 3.0, "clean", ha="center", fontsize=6.5, color=INK)
    ax.text(22.5, 3.0, "electrode-motion\nnoise (NSTDB, 6 dB)", ha="center", fontsize=6.5, color=ORANGE)
    box(36, 1, 30, 17, "Leverage-scaled probabilities",
        "$\\hat p(y\\mid x)=\\mathrm{softmax}(a\\,z(x)/\\sqrt{h(x)/h_{\\mathrm{ref}}})$\n"
        f"clean:  $h/h_{{\\mathrm{{ref}}}}$ = {vals[0][1]:.1f},  confidence {vals[0][0]:.2f}\n"
        f"noisy:  $h/h_{{\\mathrm{{ref}}}}$ = {vals[1][1]:.1f},  confidence {vals[1][0]:.2f}")
    box(71, 1, 28, 17, "Triage", "most confident recordings:\ndecided automatically\n\nleast confident (budget):\nsent to a human reviewer", col=ORANGE)
    arrow(31.5, 9.5, 35.5, 9.5); arrow(66.5, 9.5, 70.5, 9.5)
    arrow(87, 20.5, 60, 18.5)
    _save(fig, ROOT / "figures", "fig_schematic")


if __name__ == "__main__":
    main()
