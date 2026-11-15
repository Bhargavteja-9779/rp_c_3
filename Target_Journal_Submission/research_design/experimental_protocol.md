# Experimental protocol

This file consolidates the plan of the study. The original registration documents, with their
commit history, are in `supplementary/experimental_design_preregistration.md`; the git commits
cited below prove when each element was fixed. Changes to the plan are logged as D1, D2, … in
`deviations.md`.

## Research question
Can a closed-form readout make the confidence of fast random-kernel time series classifiers
drop for recordings that sensor faults have pushed away from the training data, without
held-out data, refitting or ensembles, and does this help a human-in-the-loop triage system?

**Gap (one sentence).** Existing probability outputs for random-kernel classifiers (one-hot
scores, global LOO scaling as in PreVal, cross-validated calibration, the Arsenal ensemble)
apply one global transformation, so none of them lowers its confidence specifically for
faulty or unfamiliar recordings, and none has been evaluated for that purpose.

## Hypotheses, metrics, tests and pass/fail criteria

"Registered" gives the commit that fixed the hypothesis before the relevant results existed.
For H1–H3 the registered wording was qualitative; the numeric decision rules in the
"criterion" column restate the registered statistics section (§6 of the registration:
two-sided Wilcoxon signed-rank, Holm within each table, α = 0.05) and were written down in this
form on 2026-11-03, after the results were known. They are marked **(formalised post hoc)**.

| ID | Hypothesis | Experiment | Metric | Test | Pass criterion | Registered | Outcome |
|---|---|---|---|---|---|---|---|
| H1a | On clean data LOOM's NLL is not worse than that of PreVal, hold-out TS, Platt-CV5 and LogReg-CV | E1 | test NLL, dataset = unit, mean over resamples | Wilcoxon, Holm over competitors | no competitor significantly better (formalised post hoc) | 3d20c3f, 2026-10-13 | **pass** (no competitor significantly better) |
| H1b | On clean data LOOM's ECE and Brier score are not worse than PreVal's | E1 | ECE (15 bins), Brier | as H1a | PreVal not significantly better (formalised post hoc) | 3d20c3f | **fail**: PreVal better in ECE and Brier |
| H1c | Accuracy unchanged vs PreVal | E1 | accuracy | as H1a | no significant difference (formalised post hoc) | 3d20c3f | **pass** |
| H2a | Under simulated faults LOOM's NLL is lower than that of every closed-form and cross-validated readout | E4 | NLL averaged over 18 conditions | Wilcoxon, Holm | significantly lower than each (formalised post hoc) | 3d20c3f | **pass** |
| H2b | Under simulated faults LOOM separates faulty from clean inputs better than PreVal | E4 | AUROC of 1 − max p | Wilcoxon, Holm | significantly higher | 3d20c3f | **pass** |
| H3 | Closed-form jackknife+ sets reach near-nominal coverage on clean data with smaller sets than split conformal | E9 | coverage, mean set size at α = 0.1 | mean coverage; Wilcoxon on set size | mean coverage within 0.88–0.92 and set size significantly smaller (formalised post hoc) | 3d20c3f | **pass** |
| H4 | Under real recorded ECG noise LOOM's NLL is lower than PreVal's | E12 | NLL per dataset × noise × SNR cell | Wilcoxon over the 63 cells, Holm over competitors | p < 0.05, LOOM lower | 972f1d9, 2026-11-02 | **pass** |
| H5 | In confirmatory triage, LOOM's review queue catches more faulty recordings than PreVal's | E13 | share of faulty recordings in the 20 % least confident | Wilcoxon, Holm over 8 primary tests | p < 0.05, LOOM higher | 972f1d9 | **pass** (38/38) |
| H6 | In confirmatory triage, LOOM's automatic decisions make fewer errors than PreVal's | E13 | error rate of the 80 % most confident | as H5 | p < 0.05, LOOM lower | 972f1d9 | **pass** (35/38) |
| H7 | On QUANT features, LOOM's NLL under faults is lower than PreVal's | E14 | NLL, severity-3 mean | Wilcoxon, Holm | p < 0.05, LOOM lower | 972f1d9 | **pass** (110/110) |
| H8 | On PTB-XL, LOOM's 20 % review queue contains a larger share of the expert-annotated noisy test records than PreVal's | E20 | share of annotated-noisy records in the 20 % least confident | exact McNemar test, seed 0 | p < 0.05, LOOM higher | 6b05cb1, 2026-11-14 | **fail** on the registered seed 0 (24.8 % vs 23.2 %, McNemar p = 0.12); same direction in all 5 seeds (+1.6 to +2.5 points), p < 0.05 in seeds 2–4 (descriptive). Secondary analysis (training on all records): −1.0 to +1.9 points, no seed significant |
| H9 | On PTB-XL, LOOM's mean NLL on the annotated-noisy test records is lower than PreVal's | E20 | NLL | paired bootstrap of records (10,000), seed 0 | 95 % CI of LOOM − PreVal below 0 | 6b05cb1 | **fail**, opposite direction (LOOM − PreVal = +0.034, CI [0.015, 0.051]); LOOM higher in every seed, primary and secondary analysis |
| H10 | The hybrid (QUANT label, LOOM review queue) has a lower automatic error rate than QUANT and than LOOM | E21 on E13 outputs | error rate of the 80 % most confident | Wilcoxon over 38 datasets, Holm over 2 | p < 0.05 against both, hybrid lower | 6b05cb1 | **fail** (hybrid worse than QUANT on 30/38, p < 0.001; vs LOOM p = 0.071) |
| H11 | Deployed without change on 12-lead ECGs recorded for a mobile-health project by recordists with varying training (PhysioNet/CinC Challenge 2011, set A; conventional ECG machines), LOOM's low confidence separates the records that the challenge annotators judged unacceptable for interpretation from the acceptable ones better than PreVal's | E23 | AUROC of 1 − max probability, unacceptable vs acceptable (the 2 indeterminate records excluded) | paired bootstrap over records (10,000 resamples), seed 0 | 95 % CI of AUROC(LOOM) − AUROC(PreVal) above 0 | registered before running (this commit), 2026-11-15 | **pass** (seed 0): AUROC 0.863 vs 0.344, difference 0.519, 95 % CI [0.475, 0.564] |
| H12 | In the same setting, LOOM's 20 % review queue contains more of the unacceptable records than PreVal's | E23 | share of unacceptable records among the 20 % least confident | exact McNemar test on the unacceptable records, seed 0 | p < 0.05, LOOM higher | registered before running (this commit) | **pass** (seed 0): 59.1 % vs 13.3 % caught; discordant 107 / 4; p < 0.001 |
| X1 | (exploratory) LOOM on QUANT features vs the QUANT classifier | E14 | NLL under faults | Wilcoxon | none (exploratory) | 972f1d9 | QUANT better (97/110) |

## Experiments

All experiments use univariate equal-length UCR datasets (114 in total), the official split
(resample 0) and stratified resamples 1–4 (numpy seed = resample index; MiniRocket seed
1000·r), unless stated otherwise. All seeds were fixed before results were seen.

| ID | Purpose | Data | Methods | Status / registration |
|---|---|---|---|---|
| E1 | Clean calibration | 114 datasets × 5 resamples (3 for the four largest) | LOOM, PreVal, Ridge-onehot, Ridge-softmax, Holdout-TS, Ridge-Platt-CV5, Ridge-Isotonic-CV5, LogReg-CV, QUANT | registered 3d20c3f |
| E2 | Ensembles and raw-series classifiers (official split) | 114 | PreVal-Ens5, LOOM-Ens5, QUANT, Arsenal | registered; compute amendments D3–D7 |
| E3 | Ablations | as E1 | 8 variants (Sec. 5.2 of the paper) | registered |
| E4 | Simulated faults: 6 types × 3 severities | as E1 | as E1 | registered |
| E5 | Other feature extractors | 110 datasets, official split, clean + severity 3 | MultiRocket, Hydra with LOOM / PreVal / Ridge-softmax / Holdout-TS | registered; Rocket dropped (D5) |
| E6 | Cost | as E1 | all | registered |
| E7 | Training-set size 25 %, 50 % | 110, official split | core readouts | registered |
| E8 | Error analysis (correlations, worst cases) | as E1 | LOOM vs PreVal | registered (descriptive) |
| E9 | Conformal prediction sets (LAC, APS) | as E1, ≤ 1,000 test series | jackknife(+) LOOM / PreVal, split conformal | registered; APS added before benchmark (D2) |
| E10 | Natural sensor change: UWaveGestureLibrary cross-axis | 6 axis pairs × 5 resamples | LOOM, PreVal, Ridge-softmax, Holdout-TS | **post hoc** (D11) |
| E11 | Exploratory triage study | 12 sensor datasets × 5 resamples | LOOM, PreVal, Holdout-TS, Ridge-Platt-CV5, QUANT | protocol committed before running (3d2c936), designed post hoc (D13) |
| E12 | Real recorded ECG noise (MIT-BIH NSTDB) | 7 ECG datasets × 5 resamples × 9 conditions | as E11 | registered 972f1d9 |
| E13 | Confirmatory triage replication | 38 further sensor datasets × 5 resamples | as E11 | registered 972f1d9 (+ count correction fa32f8e) |
| E14 | Readouts on QUANT features | 110 datasets, official split | LOOM-QF, PreVal-QF, QUANT | registered 972f1d9 |
| E15 | Bayesian signed-rank analysis (ROPE ±0.01) | results of E1, E4, E12–E14 | — | registered 972f1d9 |
| E16 | Logistic-regression convergence check | 12 random datasets | LogReg-CV default vs long training | **post hoc** (D12) |
| E17 | Sanity checks: sampling-noise simulation and exchangeability check | results of E1, E9 | — | **post hoc**, added 2026-11-03 (D19) |
| E18 | Validation of the command-line tool on plain CSV files | 5 datasets, official split | loom-triage vs pipeline | **post hoc**, added 2026-11-04 (D21) |
| E19 | Fault-prevalence sensitivity of the confirmatory triage study | E13 outputs, prevalence 10 / 20 / 30 % | as E13 | **post hoc**, added 2026-11-04 (D22) |
| E20 | Expert-annotated real faults in clinical 12-lead ECG (PTB-XL v1.0.3, 100 Hz) | 16,307 single-superclass records; train folds 1–8 (clean records: primary; all: secondary), test folds 9–10; seeds 0–4 | multivariate MiniRocket + LOOM / PreVal / Holdout-TS / Ridge-Platt-CV5; QUANT | registered 6b05cb1 |
| E21 | Leverage-gated hybrid triage (QUANT label, LOOM review queue) | outputs of E13 (confirmatory), E12 and E20 (descriptive) | Hybrid, Hybrid-min, LOOM, QUANT | registered 6b05cb1 |
| E22 | Post hoc explanation of E20/E21: detection by annotation type; LOOM − PreVal effect across studies | outputs of E11, E12, E13, E20 | LOOM, PreVal, Holdout-TS, Platt, QUANT | **post hoc** (D27) |
| E23 | Real faults from a different acquisition setting: models trained on the clean PTB-XL training records (exactly as in the E20 primary analysis, seeds 0–4) are applied unchanged to the 998 graded records of CinC 2011 set A (773 acceptable, 225 unacceptable; 500 Hz resampled to 100 Hz with an anti-aliasing polyphase filter; leads reordered by name to the PTB-XL order; gain from the header, values in mV). No CinC record is used for fitting, tuning or choosing anything. Diagnostic labels do not exist for these records, so only quality triage is evaluated. Secondary, descriptive: all seeds, budgets 10 / 30 %, Holdout-TS, Platt, QUANT, leverage alone (h/h_ref), and the deep baselines of E24 | CinC 2011 set A | LOOM, PreVal, Holdout-TS, Ridge-Platt-CV5, QUANT, leverage alone; E24 deep baselines | registered before running (this commit) |
| E24 | Deep-learning uncertainty baselines (comparators, no hypothesis): a 1-D residual CNN (three residual blocks with 32/64/64 channels, kernel sizes 8/5/3, global average pooling, dropout 0.2 before the output layer), trained on the clean PTB-XL training records (per-lead standardisation with training statistics; Adam, learning rate 10^-3, batch 64, 20 epochs, no early stopping), five networks with seeds 0–4. Reported: single network (seed 0), deep ensemble (mean of the five), MC dropout (network 0, 20 stochastic passes). Evaluated with the E20 metrics on the PTB-XL test folds and the E23 metrics on CinC 2011. Size chosen to fit the CPU budget; results are reported whatever their direction | PTB-XL folds 1–10, CinC 2011 set A | Deep-single, Deep-ensemble, MC-dropout | registered before running (this commit) |

## Compute budget and scope cuts
Hardware: one cloud container, 4 CPU cores, 14.3 GB memory limit, no GPU. Measured wall-clock
times: E1–E4 + E9 about 30 CPU-hours in total (largest single job: ElectricDevices, 108 min);
E2 raw-series classifiers about 6 CPU-hours; E5 and E7 about 8 CPU-hours; E11–E14 about
6 CPU-hours. The scope cuts made to fit this budget are logged as D3–D9.

## What would support or reject each hypothesis
* H1 is rejected if any baseline is significantly better than LOOM on clean data in the stated
  metric; H1b was rejected.
* H2 and H4 are supported only by a significant improvement over PreVal under faults. A gain
  under simulated faults that disappears under real noise would reject the practical value claim.
* H5 and H6 are supported only by significant results under the Holm correction over all eight
  primary tests of E13; a non-significant result would have been reported as a failed replication.
* H7 is supported by a significant improvement on QUANT features; a failure would have limited
  the claim to convolutional features.
* H8 and H9 were the test on real faults annotated by experts. They were supported only by a significant
  improvement over PreVal on PTB-XL; both failed, which limits the practical claim to strong artefacts.
* H10 was supported only by a significantly lower automatic error than both QUANT and LOOM; it failed.
* H11 and H12 test the claim that remains after H8/H9: that leverage-aware confidence catches strong real
  artefacts. They are supported only by a significant improvement over PreVal on CinC 2011; a failure would
  leave the practical claim resting on simulated faults and added recorded noise only, and will be reported as such.
