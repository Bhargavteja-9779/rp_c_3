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
