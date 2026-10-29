# Experimental design (fixed before the full benchmark was run)

Committed to the repository before any non-development dataset was evaluated with the
final protocol. Development exposure is disclosed in §7.

## 1. Hypothesis
For random-convolutional-kernel time-series classifiers (MiniRocket and relatives) with
a ridge readout, standardising the ridge logits by the square root of the test point's
leverage, z(x)/√h(x), and fitting a single temperature and the ridge penalty on the
*exact* leave-one-out pairs (z₋ᵢ, h₋ᵢ), yields

* H1 in-distribution probabilities at least as well calibrated (NLL, ECE, Brier) as the
  best existing closed-form readout (PreVal) and as cross-validated or hold-out calibration,
  with unchanged accuracy;
* H2 substantially better calibration and shift detection under test-time corruptions,
  because leverage grows for inputs off the training distribution;
* H3 closed-form jackknife+ prediction sets with near-nominal coverage on clean data
  and smaller sets than split conformal.

## 2. Research questions → experiments
| RQ | Question | Experiment |
|---|---|---|
| RQ1 | In-distribution calibration vs. baselines? | E1 main (clean), E2 strong baselines |
| RQ2 | Which components matter? | E3 ablation |
| RQ3 | Behaviour under test-time distribution shift? | E4 robustness (18 corruption conditions), shift-detection AUROC |
| RQ4 | Does the gain transfer to other random-kernel extractors and dataset groups? | E5 generalisation (Rocket, MultiRocket, Hydra; dataset types; dev vs. non-dev) |
| RQ5 | Cost? | E6 efficiency (fit/predict time, complexity) |
| RQ6 | Sensitivity to design choices? | E7 (moderation strength c, smoothing, λ criterion, miscoverage α, training-set size) |
| RQ7 | When does it fail? | E8 error analysis |
| RQ8 | Valid prediction sets? | E9 conformal |

## 3. Data
114 UCR univariate datasets with equal-length series and no missing values in the
official text distribution (UCR 2018 archive; list in `code/data/raw/ucr_equal_length.csv`).
Resamples r = 0 (official split) and r = 1…4 (stratified resamples of the pooled data that
keep the official per-class training counts; numpy seed = r). The MiniRocket seed for
resample r is 1000·r (+j for ensemble member j). No seed was chosen after seeing results.

## 4. Methods (all share identical MiniRocket features per resample)
Proposed: **LOOM** (standardised logits, Platt target smoothing, λ and a by LOO NLL).
Baselines: Ridge-onehot (default output of ROCKET-family implementations), Ridge-softmax
(T = 1), PreVal (Dempster et al.; global LOO scaling, faithful: no smoothing), Holdout-TS
(80/20 temperature scaling), Ridge-Platt-CV5 and Ridge-Isotonic-CV5 (sklearn
CalibratedClassifierCV), LogReg-CV (L2 logistic regression, C ∈ {1e-3,1e-2,1e-1,1} by 3-fold
CV log-loss), PreVal-Ens5 and LOOM-Ens5 (average of five MiniRocket seeds), Arsenal and QUANT
(aeon; raw-series classifiers with native probabilities).

Compute-driven protocol decisions (made after timing pilots, before any benchmark result):
* LogReg-CV is skipped for datasets with more than 2,000 training series (13 fits of an
  L2 logistic regression with ~10,000 features per job); its comparisons use the remaining datasets.
* Arsenal uses aeon's MiniRocket-kernel configuration (`rocket_transform="minirocket"`,
  25 estimators x 2,000 kernels) instead of the default ROCKET kernels: the default
  configuration needed 32 s to predict 1,029 series of length 24, which extrapolates to
  several CPU-days for 19 test conditions on the archive. Arsenal is run on the official
  split (resample 0) only and is compared with the other methods on that split in a
  separate table. QUANT is run on all five resamples.
Ablations: PreVal-smooth, LOOM-nosmooth, LOOM-free-b, LOOM-finite-c1, LOOM-kNN,
LOOM-insample, LOOM-holdout, LOOM-lambdaSSE.

## 5. Corruptions (test set only; severities 1–3)
noise σ ∈ {0.1, 0.3, 0.5}·sd; wander amplitude ∈ {0.25, 0.5, 1}·sd; missing segment ∈ {10, 20, 30} %
(linear interpolation); spikes ∈ {1, 3, 5} % of points at ±3 sd; smooth monotone time warp with
slope ∈ [1−s, 1+s], s ∈ {0.05, 0.1, 0.2}; circular shift ∈ {5, 10, 20} % of the length.
Seeds are a hash of (dataset, resample, family, severity): identical inputs for all methods.

## 6. Endpoints and statistics
* Primary endpoint: test NLL. Secondary: ECE (15 equal-width bins, top label), Brier,
  classwise ECE, accuracy, balanced accuracy, AURC, shift-detection AUROC, conformal coverage
  and set size (α = 0.1; 0.05 and 0.2 as sensitivity).
* Robustness endpoint: per-dataset mean over the 18 corruption conditions.
* Unit of analysis = dataset (metric averaged over resamples). Two-sided Wilcoxon
  signed-rank test LOOM vs. each competitor, Holm correction within each table,
  matched-pairs rank-biserial correlation, percentile-bootstrap 95 % CI (10,000 draws) for
  the median paired difference; Friedman test and average ranks over all methods.
* Significance level 0.05 after Holm correction. No claim of superiority without it.

Conformal scores: both LAC (1 − p) and deterministic APS (Romano et al. 2020) are reported
for jackknife+ (LOOM, PreVal) and split conformal (Holdout-TS). APS was added after the
quick-mode run on the six development-exposed datasets showed that LAC sets become empty
when moderated probabilities are flat under shift; it was added before the full benchmark.

## 7. Development exposure (disclosed)
Method design used 8 randomly chosen development datasets (Adiac, Beef, Car,
EthanolLevel, Fish, Lightning2, NonInvasiveFetalECGThorax2, SonyAIBORobotSurface1) with the
official split and three ad-hoc corruptions. GunPoint, ECG200 and Coffee were used for
pipeline smoke tests, and ItalyPowerDemand, SonyAIBORobotSurface2 and TwoLeadECG form the
`run_all.py --mode quick` set; the GunPoint smoke test revealed that separable LOO predictions drive
the temperature to its bound, which motivated Platt target smoothing (decided before the
benchmark). All results are reported for the 114 datasets and, as a sensitivity analysis,
for the 100 datasets never seen during development, and for the 98 datasets without
train/test duplicates (leakage audit, `code/results/leakage_audit.csv`).


## 8. Protocol amendments for compute (made before any benchmark result was inspected)
After the first full launch, one resample of the largest dataset (StarLightCurves) took
68 CPU-minutes with five-seed ensembles, and the container restarted, which terminated the
run. Before any result file was opened, the following amendments were made and the four
completed jobs were discarded:
1. PreVal-Ens5 and LOOM-Ens5 are computed on the official split (resample 0) only; resamples
   1–4 use a single MiniRocket seed. Ensembles are compared with the other methods on the
   official split, in the same table as Arsenal and QUANT.
2. Conformal prediction sets are evaluated on a seeded random subset of at most 1,000 test
   series per dataset (the same subset for every set method and condition).
3. Rocket, MultiRocket and Hydra (generalisation) and Arsenal are evaluated on the official
   split, on clean data and the six severity-3 corruptions.
4. (Second amendment, also before inspecting results; 26 main jobs had finished.) Because the
   cloud container restarts every one to two hours and does not compute while idle, the
   Rocket extractor was dropped from the generalisation experiment (its transform costs roughly
   ten times MiniRocket's), keeping MultiRocket and Hydra, and the training-size experiment uses
   clean data and the six severity-3 corruptions.
5. (Third amendment, before inspecting results for these datasets.) For the two datasets with more
   than 3,000 training and more than 2,000 test series (Crop, ElectricDevices) the leverage
   projection costs about 10^13 operations per readout and seed. Their test sets are therefore
   evaluated on a seeded random subsample of 2,000 test series (identical for all methods and
   conditions). The other 112 datasets use their full test sets.
6. (Fourth amendment, before inspecting results of the affected datasets.) The worker running the
   largest datasets was killed twice by the out-of-memory killer (9 GB peak) and the remaining serial
   work on them was estimated at ~16 h. For the four datasets with more than 3,000 training series
   (Crop, ElectricDevices, FordA, FordB): the main experiment and QUANT use resamples 0–2 (Crop:
   0–4, already completed); ElectricDevices is evaluated without the five-seed ensembles (single
   seed) and is therefore absent from the ensemble comparison; the extractor and training-size
   experiments exclude these four datasets. ElectricDevices and Crop jobs run alone at the end.

   *Note added 2026-10-21:* the official-split ElectricDevices job had already finished with the
   five-seed ensembles before this amendment took effect, so ElectricDevices **is** included in the
   ensemble comparison (114 datasets); the single-seed rule affects only its resamples 1–2, which
   carry no ensembles for any dataset.

## 9. Post-hoc analysis changes (made after results were inspected; reported as such)

7. (Review round 8.) The Arsenal was run on clean and severity-3 data only (amendment 3), but the
   corrupted-data comparison of the official-split table originally averaged LOOM, PreVal, the
   ensembles and QUANT over all 18 conditions. This mixed different condition sets and favoured
   LOOM against the Arsenal. The official-split comparison (manuscript Table 3) now uses the six
   severity-3 conditions **for every method**. As a result QUANT becomes significantly better than
   LOOM in NLL on this table, which is reported in the abstract, results, discussion and
   conclusion. The Brier score was added to this table because the Arsenal's vote-based
   probabilities contain exact zeros that dominate its NLL. The primary endpoint and the main
   analysis (Tables 1–2, all 18 conditions, five resamples) are unchanged.
8. (Review round 7.) Fit times of Ridge-onehot/softmax now include the shared SVD they use, and
   the MiniRocket transform time is reported separately (it was recorded but not loaded because
   of a key mismatch). No benchmark job was rerun.

## 10. Application case study (designed after the main benchmark; protocol fixed before it was run, 2026-10-29)

*Purpose.* Evaluate the methods as they would be used in a monitoring application: a classifier
labels incoming recordings automatically and passes the recordings it is least sure about to a
human reviewer, while a fraction of recordings is affected by sensor faults. This study is
**exploratory**: it was added after the main results were known, in response to the question of
practical value, and it is reported as such.

*Datasets (12, three application domains, chosen by domain before running; all UCR, official
split plus resamples 1–4):*
cardiac monitoring (ECG200, ECG5000, TwoLeadECG, NonInvasiveFetalECGThorax1);
wearable and motion sensing (UWaveGestureLibraryX, CricketX, GunPoint, SemgHandGenderCh2);
industrial and energy sensing (Wafer, FreezerRegularTrain, PowerCons, RefrigerationDevices).

*Fault model.* Each test recording is independently faulty with probability 0.3. A faulty recording
receives one fault drawn uniformly from the domain's plausible faults, at a severity drawn uniformly
from {1, 2, 3}, using the corruption functions of the main benchmark:
cardiac — broadband noise, baseline wander, electrode spikes, lead-off segment (missing);
wearable — noise, spikes, packet loss (missing), speed variation (warp);
industrial — noise, drift (wander), spikes, dropout (missing).
All draws are seeded by (dataset, resample).

*Methods.* LOOM, PreVal, Holdout-TS, Ridge-Platt-CV5 (MiniRocket features, as in the benchmark) and QUANT.

*Operating metrics (review budget b ∈ {10 %, 20 %, 30 %}; the b least confident recordings, by the
maximum predicted probability, go to review):* (1) error rate of the automatically accepted
decisions; (2) share of faulty recordings caught by the review queue; (3) for reference, the same
quantities on a fault-free stream. Primary operating point: b = 20 %. Secondary: NLL on the faulty
stream.

*Analysis.* Dataset is the unit (mean over the five resamples); LOOM vs each method by two-sided
Wilcoxon signed-rank tests with Holm correction over the four comparisons per metric; with 12
datasets these tests have low power, so per-domain medians are reported descriptively. Per-sample
outputs are saved for re-analysis.
