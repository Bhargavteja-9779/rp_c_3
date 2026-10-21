# Review round 6 — Reviewer #6 (clarity, writing, presentation of results)

*Perspective: can a non-specialist reader follow the paper; is the result text precise, readable
and free of statements that the numbers do not support?*
Reviewed: rendered manuscript (`manuscript/build/manuscript_rendered.md`) of 2026-10-21.

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R6-1 | **CRITICAL** | §6.4 (ablation) reversed the direction of several results. The template used the LOOM-perspective significance word with the *variant* as the grammatical subject, so the text said that removing leverage moderation "makes calibration under corruption significantly **better**" (median NLL 1.190 vs 0.738) and that in-sample pairs are "significantly **better**" (median 0.986 vs 0.295). | **Fixed.** New helper `varword()` returns the wording from the subject's perspective; the paragraph was rewritten into labelled parts (leverage moderation, exact LOO pairs, target smoothing, moderation strength, other choices). Every direction was checked against `tables/table_ablation_*.md`. |
| R6-2 | MAJOR | The ablation text hid a trade-off: on clean data LOOM-free-b and c = 1 are slightly but significantly *better* than LOOM (Δ ≈ −0.004 to −0.005), and the earlier text said c = 1 and c → ∞ "give similar results". | **Fixed.** Both results are now reported. The text explains that c → ∞ is justified by shift detection (AUROC 0.765 vs 0.729 for c = 1, LOOM higher on 111 of 114 datasets), not by NLL. §6.8 is changed to match. |
| R6-3 | MAJOR | Inline statistics were very long ("W/D/L …, median difference … [95 % CI …], r_rb = …, Holm-adjusted p …") and repeated in every clause, which made §6.1–6.5 hard to read. | **Fixed.** Compact form "(W/D/L a/b/c; Δ = x [l, u]; r_rb = r; p …)". §5.4 defines Δ, the CI, W/D/L and that p is Holm-adjusted, and says that the Spearman p-values in §6.9 are descriptive and unadjusted. |
| R6-4 | MAJOR | Error analysis (§6.9): the list of the largest clean losses silently skipped the second-worst dataset (Symbols). The severity-1 win rates were quoted for two hand-picked families ("64 %–56 % (noise, missing)"). "Strongly correlated" was used for ρ = 0.61. | **Fixed.** The four worst datasets are now listed by rank, with their per-class training counts computed from the data (2 to 10). The severity-1 range is computed over all non-warp families (48 %–64 %). The monotonicity claim ("grows with severity for every family except time warping") is now computed rather than hard-coded. "Strongly" is changed to "positively". |
| R6-5 | MAJOR | The claim that target smoothing causes the higher clean ECE was asserted without evidence. | **Fixed.** Evidence added: LOOM-nosmooth has median clean ECE 0.041, the same as PreVal (0.040), but worse NLL under corruption (§6.9, cross-referenced to §6.4). |
| R6-6 | MODERATE | The parity with QUANT under corruption appeared only in Table 2, not in the abstract or discussion, which overstates LOOM's position among all methods. | **Fixed.** The abstract now says LOOM "is on par with the tree-based QUANT classifier" (generated from the test result). The discussion reports QUANT's corrupted NLL and its higher clean NLL and cost. §6.3 adds the per-family pattern (QUANT better for wander, missing segments and spikes; LOOM for noise, warping and circular shifts). |
| R6-7 | MINOR | Inconsistent decimals for leverage ratios (30.3, 6.5, 8.9, 3.52). | **Fixed** (one decimal). |
| R6-8 | MINOR | Train-size results were referred only to "Table S4". | **Fixed.** Medians, W/D/L and p for 25 % and 50 % are now in the text. The concluding sentence is generated from the test results and is not asserted. |

## Verdict after fixes
The result text is now generated with the correct perspective throughout. Claims whose truth
depends on the data (trend, worst cases, QUANT parity, train-size conclusion) are computed rather
than written by hand.
