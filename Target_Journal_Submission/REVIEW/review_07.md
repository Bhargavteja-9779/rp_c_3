# Review round 7 — Reviewer #7 (figures, tables, visual integrity)

*Perspective: are figures legible at print size, honest (axes, clipping, support), consistent with
the text, and do tables report exactly what their captions say?*
Inspected: every figure used in the manuscript (Figs. 1–8, PNG at 300 dpi and PDF) and Tables 1–5.

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R7-1 | MAJOR | **Fig. 1** (method overview): text overflowed two boxes ("Exact LOO for every λ…", "Fit λ, a…"), the first two boxes touched, and the leverage was written h(x) = zᵀA⁻¹z. That reuses z, which denotes the logits elsewhere (the notation clash already fixed in the text). | **Fixed.** Redrawn with equal box widths and gaps and shorter labels; the leverage is now x̃ᵀA⁻¹x̃, matching Eq. (2), and an elbow arrow leads to the test-time box. |
| R7-2 | MAJOR | **Table 5** (cost): the caption said ridge-based readouts include the shared SVD, but Ridge-onehot/softmax times did not. The MiniRocket transform time was never loaded (key mismatch), so LOOM's readout-only time was compared with QUANT's end-to-end time. | **Fixed** (analysis code). The transform has its own row; the caption states what each time includes; the text compares end-to-end medians (LOOM ≈ 0.31 s, QUANT 1.23 s, five-seed ensemble 2.25 s, Arsenal 4.46 s). |
| R7-3 | MODERATE | **Fig. 5** (scatter): on clean data unsmoothed PreVal reaches NLL ≈ 10⁻¹⁶ on perfectly separated datasets. The log axes then spanned 16 decades and compressed all other points into a corner. | **Fixed.** Values below 10⁻³ are drawn at 10⁻³, which the caption states and explains. |
| R7-4 | MODERATE | **Fig. 6** (reliability): bins with very few pooled predictions produced spikes, and the legend covered part of the Platt curve. The pooled ECE in the legend differs from the per-dataset medians in Tables 1–2 without explanation. | **Fixed.** Bins with fewer than 200 pooled predictions are not drawn (the ECE still uses all bins); legend moved to lower right; the caption explains the pooled-vs-median difference. The remaining low-confidence point on clean data comes from many-class datasets, where a maximum probability of about 0.06 is still far above uniform. |
| R7-5 | MODERATE | **Fig. 7** (ablation): the right end of the clean panel clipped the LOOM-insample interval. | **Fixed.** The x-limits are derived from the CI extremes. |
| R7-6 | MINOR | **Fig. 8** (efficiency): the legend overlapped data points. LOOM and PreVal markers coincide, which is correct because their fit costs are identical. | **Fixed.** Legend placed outside the axes. |
| R7-7 | MINOR | Fig. 4 shows that QUANT is below LOOM for three corruption families at high severity, which the text did not mention. | **Fixed** in §6.3 (see R6-6). |
| R7-8 | MINOR | Colour-blind safety and greyscale printing. | Each method uses a fixed colour, marker *and* line style (`plots.STYLE`), so curves remain distinguishable in greyscale. |

## Verdict after fixes
Figures are legible and honest about clipping and support; table captions match the computed
quantities.
