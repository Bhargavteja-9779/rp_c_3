# Review round 15 (SIMULATED) — Reviewer B: statistics and design, trying to reject

*Simulated review, not an external review.* Version reviewed: elsarticle build of 2026-11-14.

**Recommendation of the simulated reviewer: minor/major revision.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| B1 | "H8/H9 rest on a single seed." | Moderate | The registration names seed 0 for the tests; the five-seed means and ranges are reported (Table 8, Fig. 7) and the per-seed test outcomes are counted (`results/e20/summary.json`). | As registered |
| B2 | "Records are not independent: patients have several test records." | Moderate | Post hoc patient-level bootstrap (757 patients, 810 noisy records): interval for H9 unchanged ([0.015, 0.051]); added to Sect. 6.6. Folds keep each patient in one fold (0 patients across folds; Suppl. S6). | Fixed |
| B3 | "The 'noisy' definition includes annotations that name no lead." | Minor | The registered definition (any annotation column filled) is used for H8/H9; 53 records whose text names no lead are excluded only from the post hoc lead count (Suppl. S6). | Disclosed |
| B4 | "The cross-study effect plot mixes units (datasets vs records)." | Minor | Stated in the caption of Fig. 9; the plot is descriptive and labelled post hoc. | Disclosed |
| B5 | "The explanation of the failures was built after seeing them." | Moderate | Labelled post hoc throughout (deviation D27); no test is claimed from it. | Disclosed |
| B6 | "QUANT was run differently in E20 (chunked)." | Minor | Unit test shows identical probabilities to QUANTClassifier (tolerance 1e-12); deviation D26. | Fixed |
| B7 | "Regenerated outputs for E21 might differ from the original runs." | Minor | E21 compares the regenerated LOOM/QUANT triage metrics with the stored results: 760 + 70 rows, maximum difference 0 (D25). | Fixed |

**Verdict after actions.** The statistics are sound and conservative; the remaining weakness is power on the rare
electrode-problem annotations (4 records), which the paper declines to interpret.
