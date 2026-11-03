# Review round 8 — Reviewer #8 (time series classification domain expert)

*Perspective: would a TSC researcher (bake-off community) accept the protocol, the baselines and
the interpretation?*
Reviewed: manuscript, `tables/table_rawbaselines_*.md`, `results/raw_results.csv`, protocol in
Online Resource 2.

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R8-1 | **CRITICAL** | Table 3 compared methods on different conditions. The Arsenal was evaluated on severity-3 corruptions only, but LOOM, PreVal, the ensembles and QUANT were averaged over all 18 conditions (most of them milder). This favours LOOM against the Arsenal and hides how the methods behave under strong corruption. | **Fixed (analysis rerun).** The official-split table now uses the six severity-3 conditions for every method (`analyze.py`). **Consequence:** QUANT becomes significantly better than LOOM in NLL at severity 3 (median 0.917 vs 1.020; W/D/L 35/0/77; Holm p = 0.032) and in Brier score. This negative result is now reported in §6.3, the abstract ("on par with … QUANT on average but worse than it under the strongest corruptions", generated from the tests), the Introduction's list of where LOOM does not help, the discussion and the conclusion. It is documented as post-hoc change 7 in Online Resource 2 and disclosed in §6.3. |
| R8-2 | MAJOR | The Arsenal's clean NLL (median 1.66 vs 0.33) looks implausibly poor for a method with the same accuracy. | **Explained with evidence.** Its probabilities are accuracy-weighted vote fractions that often assign exactly 0 to the true class. NLL then hits the 10⁻¹⁵ floor. The Brier score was added: clean median 0.171 vs 0.163 for LOOM (much closer), and under severity-3 corruption LOOM still has the clearly lower Brier score. Both are reported in §6.3. |
| R8-3 | MAJOR | Five resamples is fewer than the 30 used in the bake-offs. | Pre-registered for compute reasons. The unit of analysis is the dataset (114 units). Results also hold on the 100 development-free datasets, without duplicate datasets, and when related datasets are grouped by source (70 units). Listed under compute-driven choices in the Limitations. |
| R8-4 | MAJOR | No comparison with HIVE-COTE 2.0, InceptionTime or other deep models. | Out of scope for a readout paper and computationally infeasible on 4 CPU cores. Stated in the Limitations. The Arsenal (HC2's probabilistic ROCKET component) and QUANT are included as raw-series probabilistic baselines. |
| R8-5 | MODERATE | Is the MiniRocket accuracy in line with the literature? | Mean clean accuracy 0.873 over 114 datasets and 5 resamples, consistent with published MiniRocket results on the UCR archive (≈0.87 mean). QUANT and the Arsenal are within 1 % of it, so there is no sign of a broken baseline pipeline. |
| R8-6 | MODERATE | Why does a circular shift increase leverage when MiniRocket's PPV pooling is nearly shift-invariant? | The wrap-around creates a discontinuity at the cut point, and the dilated kernels with padding see different boundary content. Leverage grows only moderately (median 3.5× at 20 %), and LOOM's win rate is the second lowest of the families. The text is consistent with this. |
| R8-7 | MODERATE | The corruptions are applied to z-normalised UCR series with amplitudes relative to each series' standard deviation. Is that realistic? | Relative amplitudes make the severities comparable across datasets. The paper calls the corruptions synthetic and adds a natural cross-axis shift (§6.5), where LOOM's gain is small. Limitations state that real deployment shifts remain to be studied. |
| R8-8 | MINOR | Equal-length datasets: 114 versus the commonly used 112. | The list is computed from the official files (equal length, no missing values) and committed (`data/ucr_equal_length.csv`). §5.1 states the criterion. |

## Verdict after fixes
The comparison is now like-for-like. The paper is more modest but more credible. LOOM is the best
*readout* for random-kernel features under corruption, but not the most robust probabilistic
time series classifier overall.
