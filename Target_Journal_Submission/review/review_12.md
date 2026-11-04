# Review round 12 (SIMULATED) — Reviewer B: statistics and methodology, trying to reject

*Simulated review, not an external review.* Version reviewed: LaTeX build of 2026-11-04.

**Recommendation of the simulated reviewer: major revision.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| B1 | "Dozens of Wilcoxon tests across many tables; family-wise error is not controlled across the paper." | Major | Rewording. Sect. 5.6 now states that only the hypotheses of Table 3 are confirmatory and that other comparisons are descriptive (Holm within each table). The confirmatory triage tests use Holm over all eight primary tests. | Fixed |
| B2 | "Pass/fail criteria for H1–H3 were set after the results." | Major | Already disclosed (deviation D18, Table 3 caption, Sect. 5.6). The criteria restate the registered tests, and H1b fails under them. | Disclosed; cannot be undone |
| B3 | "'Not significantly different' is presented as 'on a par'." | Major | Bayesian signed-rank tests added (E15). On clean data they show no practical equivalence and favour PreVal (0.62 vs 0.38); the text says so. A sampling-noise check (E17) shows PreVal better beyond noise on 38 % of datasets and LOOM on 13 %. | Fixed (reported) |
| B4 | "Datasets are not independent (shared sources)." | Moderate | A source-grouped analysis is reported for the benchmark (68/70 units) and the confirmatory triage (27/27 units). | Addressed |
| B5 | "Exchangeability, assumed by jackknife+ and by the calibration, is not checked." | Major | New sanity checks (E17). Held-out and test leverages have the same median scale, but their distributions differ on 19 of 110 datasets. Per-dataset coverage varies more than binomial noise allows (sd 0.035 vs 0.010). Both are reported in Sect. 6.10 and limitation 5. | Fixed (reported as weaknesses) |
| B6 | "Medians hide heavy tails." | Moderate | Means with 95 % bootstrap CIs are reported (Fig. 2, Sects. 6.3 and 6.5). | Fixed |
| B7 | "The near-duplicates found by the extended leakage audit may inflate results." | Moderate | Sensitivity analysis without all 24 datasets with exact or near duplicates: LOOM better under faults on 84/90 datasets. A planted-leak test validates the audit. | Fixed |
| B8 | "Post hoc changes (severity-3 restriction, exploratory triage) undermine confirmatory claims." | Moderate | All are labelled post hoc in the text and in the 22-entry deviations log. The triage claim rests on the confirmatory replication registered before running. | Disclosed |

**Verdict after actions.** The statistics are transparent. The remaining weaknesses
(approximate exchangeability, post hoc criteria for H1–H3) are stated in the paper.
