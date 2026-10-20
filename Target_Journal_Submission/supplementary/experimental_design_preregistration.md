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
