# Claim–evidence matrix (ESWA version, third revision, 2026-11-15)

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
| 6 | Review queue catches 41 % vs 24 % (PreVal) of faulty recordings in the confirmatory study (abstract, conclusion, cover letter) | empirical-confirmatory (H5); QUANT comparison descriptive | `results/e13/summary.json` | `\CCaughtLoom`, `\CCaughtPreVal`, `\CCaughtQuant`, `\CmpCCaughtPreVal`, `\CmpCCaughtQuant` | ✔ |
| 7 | Fewer automatic errors than PreVal in triage (Sect. 6.5) | empirical-confirmatory (H6) | `results/e13/summary.json` | `\CmpCErrPreVal` | ✔ |
| 8 | On clean data LOOM is not better than PreVal (abstract, conclusion, limitations) | empirical-confirmatory (H1a pass, H1b fail) + Bayesian (E15) | `results/e01/summary.json`, `results/e15/summary.json` | `\CmpNLLCleanPreVal`, `\BayesCleanPreValWorse` | ✔ (negative result reported) |
| 9 | QUANT has a lower likelihood loss under the strongest faults (abstract, limitations) | empirical-descriptive | `results/e02/summary.json` → E2b_raw_shifted | `\CmpNLLOffSevThreeQuant` | ✔ |
| 10 | Under simulated faults and added recorded noise, LOOM detects faulty recordings better than all competitors, including QUANT (Sects. 6.3, 6.4); not on PTB-XL (Sect. 6.6) | empirical-descriptive (vs PreVal: confirmatory H2b) | E4 shift_auroc; E12 shift_auroc cells | `\AUROCFaultsLoom`, `\CmpRNAUROCCellsQuant` | ✔ |
| 11 | LOOM works on non-convolutional (QUANT) features (contribution 3, Sect. 6.9) | empirical-confirmatory (H7) | `results/e14/summary.json` | `\CmpQFPreVal` | ✔ |
| 12 | QUANT's robustness comes mainly from its trees (Sect. 6.9) | interpretation of E14 (exploratory X1) | `results/e14/summary.json` | `\QFAccLoom`, `\QFAccQuant` | ◐ "mainly"; explanation about extrapolation stated as a plausible mechanism, not tested |
| 13 | Leverage reacts to faults except speed variation (Sects. 4.3, 6.3) | empirical-descriptive | E8 leverage_ratio_median_by_condition | `\LevNoise` … `\LevWarp` | ✔ |
| 14 | Results hold without development datasets, without (near-)duplicates, for source-grouped data (Sect. 6.13) | empirical-descriptive (near-duplicate analysis post hoc) | `results/sensitivity/summary.json` | `\CmpUnseenFaults`, `\CmpNoNearDupFaults`, `\CmpGroupedFaults` | ✔ |
| 15 | Results hold at 10 % fault prevalence (Sect. 6.5) | post hoc (D22) | `results/e19/summary.json` | `\PrevTenCaughtLoom`, `\CmpPrevTenCaughtPreVal` | ✔ labelled post hoc |
| 16 | Exchangeability of leverages holds only approximately; coverage varies across datasets more than chance (Sect. 6.13) | post hoc (D19) | `results/e17/sanity.json` | `\ExchReject`, `\CovSDObs` | ✔ (negative) |
| 17 | Fit cost equals PreVal's; end-to-end ~0.31 s vs QUANT 1.30 s (Sect. 6.14, intro) | empirical-descriptive | `results/e06/summary.json` | `\FitLoom`, `\FitEndToEndLoom`, `\FitQuant` | ✔ (single machine; stated) |
| 18 | The tool reproduces the pipeline on plain files (Sect. 6.15) | post hoc (D21) | `results/e18/summary.json` | `\ToolIdentical`, `\ToolMaxDiff` | ✔ |
| 19 | Only the natural cross-axis change was tested and LOOM gains little there (limitations) | post hoc (D11) | `results/e10/summary.json` | `\NSNLLLoom`, `\NSNLLPreVal` | ✔ (negative) |
| 20 | "11 of 15 hypotheses passed; H1b, H8, H9, H10 failed" (Sect. 6.1, cover letter) | summary | `code/tables/tab_hypotheses.tex` | `\NHypPass`, `\HypFailed` | ✔ |

| 21 | On PTB-XL, LOOM helps at most marginally: not significant on the registered seed; +1.6 to +2.5 points in all five seeds, p < 0.05 in three (abstract, Sect. 6.6, conclusion) | empirical-confirmatory (H8, **fail**) + descriptive per-seed tests | `results/e20/summary.json` → clean.H8_seed0, clean.H8_all_seeds | `\PTBHEightCaughtLoom`, `\PTBHEightP`, `\PTBHEightDiffRange`, `\PTBHEightSeedsPass`, `\PTBHEightPRange` | ✔ (negative, nuance reported) |
| 22 | LOOM's NLL on annotated-noisy records is higher than PreVal's (abstract, Sect. 6.6) | empirical-confirmatory (H9, **fail**, opposite direction) + post hoc patient bootstrap | `results/e20/summary.json` → clean.H9_seed0; `results/e22/summary.json` → patient_cluster_bootstrap | `\PTBHNineLoom`, `\PTBHNineCI`, `\ClusterNLLCI` | ✔ (negative) |
| 23 | No tested confidence separates noisy from clean PTB-XL records much better than chance (abstract, Sect. 6.6, highlights) | empirical-descriptive | `results/e20/summary.json` → detect_auroc | `\PTBDetLoom`, `\PTBDetPreVal`, `\PTBDetPlatt`, `\PTBDetQuant` | ✔ ("much better" = AUROC ≤ about 0.55 on average) |
| 24 | First use of PTB-XL quality annotations to test whether a classifier's own confidence routes noisy records to review (contributions, Sect. 2, cover letter) | literature claim | `research_design/literature_review.md` rows 41–48 and the search note; prior uses (accuracy study, dedicated noise detectors) cited | — | ◐ narrow and hedged ("to our knowledge"); a web search is not exhaustive |
| 25 | The hybrid (QUANT label, LOOM queue) makes more automatic errors than QUANT and is not better than LOOM (abstract, Sect. 6.7, highlights) | empirical-confirmatory (H10, **fail**) | `results/e21/summary.json` → H10 | `\CmpHTenErrQuant`, `\CmpHTenErrLoom` | ✔ (negative) |
| 26 | A model's own queue catches more of its errors than another model's queue (Sect. 6.7, discussion) | post hoc (D27) | `results/e22/summary.json` → error_capture | `\CapOwnBetter`, `\CapN`, `\CapDiff`, `\CapP` | ✔ labelled post hoc |
| 27 | LOOM's gain is large for strong faults, simulated or real, and small on PTB-XL; annotated noise covers a median of five of 12 leads (Sect. 6.8, discussion) | post hoc (D27; CinC row added after E23) | `results/e22/summary.json` → effects, lead_coverage | `\EffConfDiff`, `\EffPTBDiff`, `\EffPTBCI`, `\LeadMedian` | ✔ labelled post hoc; "milder" rests on the small accuracy drop (`\PTBAccCleanLoom` → `\PTBAccNoisyLoom`) |
| 28 | Failure on PTB-XL not specific to MiniRocket features or ridge readout (Sect. 6.8) | empirical-descriptive | QUANT and Platt detection AUROCs | `\PTBDetQuant`, `\PTBDetPlatt` | ✔ |
| 29 | Regenerated E12/E13 outputs equal the stored results (Supplement S2, D25) | verification | `results/e21/summary.json` → E13_reproduction, E12_reproduction | `\ReproMaxDiff` | ✔ (max. difference 0) |
| 30 | On CinC 2011, LOOM separates unacceptable records better than PreVal (abstract, Sect. 6.7, conclusion, cover letter) | empirical-confirmatory (H11, pass) | `results/e23/summary.json` → H11_seed0, H11_all_seeds | `\CincHElevenLoom`, `\CincHElevenPreVal`, `\CincHElevenCI`, `\CincHElevenSeedsPass`, `\CincHElevenDiffRange` (all five seeds pass) | ✔ |
| 31 | LOOM's 20 % queue holds 59 % vs 13 % of unacceptable records (abstract, highlights, conclusion, cover letter) | empirical-confirmatory (H12, pass) | `results/e23/summary.json` → H12_seed0 | `\CincHTwelveCaughtLoom`, `\CincHTwelveCaughtPreVal`, `\CincHTwelveP` | ✔ (highlight rounds 59.1/13.3 to 59/13) |
| 32 | Deep ensemble catches fewer unacceptable records than LOOM; single network and MC dropout below chance (abstract, highlights, Sect. 6.7, discussion) | empirical-descriptive (E24 registered as comparator) | `results/e23/summary.json` → deep, deep_vs_loom_seed0 | `\CincCaughtDeepEns`, `\CincDetDeepEns`, `\CincDetDeepSingle`, `\CincDetMCDrop`, `\CincLoomMinusDeepEnsCI` | ✔ |
| 33 | Deep ensemble is the most accurate model on PTB-XL with fewest automatic errors, but detects noise near chance (Sect. 6.6, discussion) | empirical-descriptive | `results/e24/summary.json` | `\PTBAccDeepEns`, `\PTBErrDeepEns`, `\PTBDetDeepEns` | ✔ |
| 34 | An amplitude rule matches LOOM over all CinC records; LOOM is better on records below 10 mV (abstract, Sect. 6.7) | post hoc (D29) | `results/e23/summary.json` → rules, rule_vs_loom_seed0, exploratory_seed0 | `\CincDetRuleAmp`, `\CincLoomMinusRuleCI`, `\CincSubTenLoom`, `\CincSubTenRuleAmp` | ✔ labelled post hoc in text, table, figure; abstract says "including unusable recordings that a simple amplitude rule missed" |
| 35 | Leverage alone is a better quality score than LOOM's confidence (Sect. 6.7, discussion, cover letter) | post hoc comparison of registered secondary outputs | `results/e23/summary.json` → leverage_vs_loom_seed0 | `\CincDetLev`, `\CincLevMinusLoomCI` | ✔ (negative for the method, reported) |
| 36 | Deep ensemble costs tens of times LOOM's CPU time (Sect. 6.14, discussion) | empirical-descriptive | `results/logs/e24_run.log`, `results/e20/clean_s*.npz` | `\DeepTrainMin`, `\PTBTimeTransform`, `\PTBTimeLoom` | ◐ different thread counts and a shared CPU; stated in the text; ratio ≈ 70 (330 thread-min vs about 5) |

Claims removed or softened in the 2026-11-15 version: "no confidence beats chance" now explicitly includes deep
ensembles and MC dropout (tested); "gain shrinks steadily as faults become more realistic" replaced by "gain follows
fault strength" after CinC; "Platt gave the lowest NLL" restricted to MiniRocket readouts after E24.

Claims removed or softened in the 2026-11-14 version: "fault-aware confidence" in the title replaced by a question
that admits the limits; the planned claim that the hybrid is a better triage design was not made (H10 failed); "noise
confined to one or a few leads" corrected to "a median of five of 12 leads" after counting; "LOOM's low confidence does
not mark QUANT's errors" replaced by the paired per-dataset comparison after the medians turned out equal; "about twice
as many" removed from the highlights (41 % vs 24 %).

Claims removed or softened in this version: "reliable confidence" (title) → "fault-aware confidence";
"clearly better under simulated faults" (Fig. 2 caption) → "better than the other readouts, QUANT
close"; "much more likely" → "more likely" (discussion); an unmeasured causal claim about leverage of a
failure example (Sect. 6.9) was removed.
