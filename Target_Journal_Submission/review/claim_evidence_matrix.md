# Claim–evidence matrix (LaTeX version, 2026-11-04)

Type: **proven** (mathematical result, proof in Supplementary Sect. S1), **empirical-confirmatory**
(registered hypothesis), **empirical-descriptive** (not a registered hypothesis), **post hoc**
(designed after results). All numbers are macros from `code/tables/macros.tex`, written by
`code/experiments/aggregate.py` from the listed result files.

| # | Claim (location) | Type | Evidence: file → key | Macro(s) | Status |
|---|---|---|---|---|---|
| 1 | LOOM needs no held-out data, refitting or ensembles; one thin SVD (abstract, Sect. 4) | proven / by construction | Props. 1–2; `code/tests/test_core.py` | — | ✔ |
| 2 | Exact LOO scores and held-out leverages (Sect. 4.2) | proven | Supplementary Sect. S1; unit tests vs refits | — | ✔ |
| 3 | Exact jackknife+ sets without refitting (abstract, contribution 2) | proven (coverage approximate) | Prop. 2; `results/e09/summary.json` | `\CPLoomCovClean` | ✔ (approximation stated) |
| 4 | Under simulated faults, median NLL lowered from PreVal's to LOOM's (abstract, conclusion) | empirical-confirmatory (H2a) | `results/e04/summary.json` → E4_shifted.nll | `\NLLFaultsPreVal`, `\NLLFaultsLoom`, `\CmpNLLFaultsPreVal` | ✔ |
| 5 | Under real ECG noise, median NLL lowered (abstract, conclusion) | empirical-confirmatory (H4) | `results/e12/summary.json` | `\RNNLLPreVal`, `\RNNLLLoom`, `\CmpRNCellsPreVal` | ✔ |
| 6 | Review queue catches 41 % vs 24 % (PreVal) vs 34 % (QUANT) of faulty recordings (abstract, conclusion) | empirical-confirmatory (H5); QUANT comparison descriptive | `results/e13/summary.json` | `\CCaughtLoom`, `\CCaughtPreVal`, `\CCaughtQuant`, `\CmpCCaughtPreVal`, `\CmpCCaughtQuant` | ✔ |
| 7 | Fewer automatic errors than PreVal in triage (Sect. 6.5) | empirical-confirmatory (H6) | `results/e13/summary.json` | `\CmpCErrPreVal` | ✔ |
| 8 | On clean data LOOM is not better than PreVal (abstract, conclusion, limitations) | empirical-confirmatory (H1a pass, H1b fail) + Bayesian (E15) | `results/e01/summary.json`, `results/e15/summary.json` | `\CmpNLLCleanPreVal`, `\BayesCleanPreValWorse` | ✔ (negative result reported) |
| 9 | QUANT has a lower likelihood loss under the strongest faults (abstract, limitations) | empirical-descriptive | `results/e02/summary.json` → E2b_raw_shifted | `\CmpNLLOffSevThreeQuant` | ✔ |
| 10 | LOOM detects faulty recordings better than all competitors, including QUANT (Sects. 6.3, 6.4, discussion) | empirical-descriptive (vs PreVal: confirmatory H2b) | E4 shift_auroc; E12 shift_auroc cells | `\AUROCFaultsLoom`, `\CmpRNAUROCCellsQuant` | ✔ |
| 11 | LOOM works on non-convolutional (QUANT) features (contribution 3, Sect. 6.6) | empirical-confirmatory (H7) | `results/e14/summary.json` | `\CmpQFPreVal` | ✔ |
| 12 | QUANT's robustness comes mainly from its trees (Sect. 6.6) | interpretation of E14 (exploratory X1) | `results/e14/summary.json` | `\QFAccLoom`, `\QFAccQuant` | ◐ "mainly"; explanation about extrapolation stated as a plausible mechanism, not tested |
| 13 | Leverage reacts to faults except speed variation (Sects. 4.3, 6.3) | empirical-descriptive | E8 leverage_ratio_median_by_condition | `\LevNoise` … `\LevWarp` | ✔ |
| 14 | Results hold without development datasets, without (near-)duplicates, for source-grouped data (Sect. 6.10) | empirical-descriptive (near-duplicate analysis post hoc) | `results/sensitivity/summary.json` | `\CmpUnseenFaults`, `\CmpNoNearDupFaults`, `\CmpGroupedFaults` | ✔ |
| 15 | Results hold at 10 % fault prevalence (Sect. 6.5) | post hoc (D22) | `results/e19/summary.json` | `\PrevTenCaughtLoom`, `\CmpPrevTenCaughtPreVal` | ✔ labelled post hoc |
| 16 | Exchangeability of leverages holds only approximately; coverage varies across datasets more than chance (Sect. 6.10) | post hoc (D19) | `results/e17/sanity.json` | `\ExchReject`, `\CovSDObs` | ✔ (negative) |
| 17 | Fit cost equals PreVal's; end-to-end ~0.31 s vs QUANT 1.30 s (Sect. 6.11, intro) | empirical-descriptive | `results/e06/summary.json` | `\FitLoom`, `\FitEndToEndLoom`, `\FitQuant` | ✔ (single machine; stated) |
| 18 | The tool reproduces the pipeline on plain files (Sect. 6.12) | post hoc (D21) | `results/e18/summary.json` | `\ToolIdentical`, `\ToolMaxDiff` | ✔ |
| 19 | Only the natural cross-axis change was tested and LOOM gains little there (limitations) | post hoc (D11) | `results/e10/summary.json` | `\NSNLLLoom`, `\NSNLLPreVal` | ✔ (negative) |
| 20 | "Nine of ten hypotheses passed; H1b failed" (Sect. 6.1) | summary | `code/tables/tab_hypotheses.tex` | `\NHypPass`, `\HypFailed` | ✔ |

Claims removed or softened in this version: "reliable confidence" (title) → "fault-aware confidence";
"clearly better under simulated faults" (Fig. 2 caption) → "better than the other readouts, QUANT
close"; "much more likely" → "more likely" (discussion); an unmeasured causal claim about leverage of a
failure example (Sect. 6.9) was removed.
