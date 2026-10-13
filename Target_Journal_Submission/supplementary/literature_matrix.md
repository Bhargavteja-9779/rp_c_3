# Literature matrix and gap analysis

All entries were verified (Crossref DOI metadata, official proceedings pages or arXiv; see
`manuscript/refs/*.json`). Results are quoted only where they are stated in the paper itself.
"–" means the paper does not address the aspect.

| Paper | Year / venue | Problem | Data | Method | Probabilistic output | Calibration evaluated | Shift / robustness | Uncertainty sets | Code | Limitation relevant here |
|---|---|---|---|---|---|---|---|---|---|---|
| ROCKET (Dempster, Petitjean, Webb) | 2020, DMKD 34 | Fast accurate TSC | UCR (85 + 43 datasets) | 10k random kernels, PPV+max, ridge/logistic | No (ridge) | – | – | – | Yes | Hard predictions only |
| MiniRocket (Dempster, Schmidt, Webb) | 2021, KDD | Faster ROCKET | UCR 109 | Almost deterministic kernels, PPV, ridge | No | – | – | – | Yes | Same |
| MultiRocket (Tan et al.) | 2022, DMKD 36 | Richer pooling | UCR 128 | 4 pooling operators + first-difference series, ridge | No | – | – | – | Yes | Same |
| Hydra (Dempster et al.) | 2023, DMKD 37 | Dictionary + kernels | UCR 112 | Competing kernels, counts, ridge | No | – | – | – | Yes | Same |
| QUANT (Dempster et al.) | 2024, DMKD 38 | Interval quantiles | UCR 142 | Quantiles of intervals + extra trees | Yes (tree votes) | Not as primary metric | – | – | Yes | Calibration under shift unknown |
| HIVE-COTE 2.0 (Middlehurst et al.) | 2021, ML 110 | Meta-ensemble SOTA | UCR 112, UEA 26 | Weighted ensemble of 4 modules incl. **Arsenal** (ROCKET ensemble built to obtain probabilities) | Yes | NLL reported in later bake-off | – | – | Yes (aeon) | Costly; Arsenal multiplies ROCKET cost by its ensemble size |
| Bake off redux (Middlehurst, Schäfer, Bagnall) | 2024, DMKD 38 | Benchmark of TSC | 142 datasets | 11 algorithm families | Evaluated | Yes: NLL, AUROC alongside accuracy | – | – | Yes | In-distribution only |
| PreVal (Dempster, Webb, Schmidt) | 2025, arXiv 2401.15610 | Cheap probabilistic ridge | 273 datasets incl. 106 UCR with MiniRocket | Scale ridge coefficients by κ minimising log-loss on exact LOO predictions | Yes | 0–1 loss and log-loss (no ECE/Brier) | **–** | **–** | Yes | One global scale factor: confidence cannot depend on how unusual a test input is |
| Guo et al. (temperature scaling) | 2017, ICML | Calibration of DNNs | Images, text | Global temperature on validation set | – | Yes (ECE) | – | – | Yes | Needs held-out data; global |
| Ovadia et al. | 2019, NeurIPS | Uncertainty under shift | Images, text, genomics | Benchmark of TS, ensembles, MC-dropout, SVI | – | Yes | **Yes**: TS degrades under shift; ensembles most robust | – | Yes | Deep models only |
| Lakshminarayanan et al. (deep ensembles) | 2017, NeurIPS | Predictive uncertainty | Images | Ensembles + proper scoring | – | Yes | OOD | – | – | M× cost |
| Liu et al. (SNGP) | 2020 NeurIPS / 2023 JMLR | Distance-aware uncertainty | Images, text | Spectral normalisation + GP output layer with random features | – | Yes | Yes | – | Yes | Requires network retraining; not closed form |
| van Amersfoort et al. (DUQ) | 2020, ICML | Single-model OOD | Images | RBF centroids + gradient penalty | – | – | OOD | – | Yes | Retraining |
| Kristiadi et al. | 2020, ICML | Overconfidence far from data | Images | Last-layer Laplace / moderated output | – | Yes | Far-away inputs | – | Yes | Needs Hessian; deep nets |
| MacKay | 1992, Neural Comput 4 | Bayesian classification | – | Moderated output σ(κ(s²)μ) | – | – | – | – | – | Theory; logistic models |
| Cawley & Talbot | 2004, Neural Netw 17 | Fast LOO for LS-SVM | UCI | Closed-form LOO for model selection | – | – | – | – | – | LOO used for model selection, not calibration under shift |
| Barber et al. (jackknife+) | 2021, Ann Stat 49 | Predictive intervals | Regression | LOO models, coverage ≥ 1−2α | – | – | – | Yes (regression) | – | Needs n refits in general |
| Romano et al. (APS) | 2020, NeurIPS | Classification sets | Images | Adaptive prediction sets, CV+/jackknife+ | – | – | – | Yes | Yes | Refits per fold |
| Sadinle et al. (LAC) | 2019, JASA 114 | Least ambiguous sets | – | Threshold on p(y\|x) | – | – | – | Yes | – | Empty sets when p is flat |
| Xu & Xie | 2023, TPAMI 45 | Conformal for time series | Forecasting | Ensemble-based, non-exchangeable | – | – | Temporal dependence | Yes (regression) | Yes | Forecasting, not series classification |
| Iwana & Uchida | 2021, PLOS ONE 16 | TS augmentation survey | UCR 128 | Jitter, scaling, warping, etc. | – | – | Augmentations (training) | – | Yes | Augmentation, not test-time evaluation |
| AdaTime (Ragab et al.) | 2023, TKDD 17 | Domain adaptation benchmark | HAR, sleep, fault | UDA methods | – | – | Domain shift (needs target data) | – | Yes | Requires unlabelled target data and deep models |

## Gaps (what exists → what is missing → why it matters → what we add → how verified)

1. **Methodological gap.** The ROCKET family outputs hard labels; probability estimates
   come from ensembles (Arsenal, M× cost), from k-fold or hold-out calibration
   (data and refits), or from PreVal's single global scale. *Missing:* a closed-form,
   per-input confidence that reflects how far a test input lies from the training data.
   *Matters:* a global temperature keeps the same confidence for off-distribution inputs
   (Ovadia et al. 2019). *We add:* leverage-standardised logits whose temperature is fitted
   on exact LOO pairs (z₋ᵢ, h₋ᵢ). *Verified by:* E1, E3, E4.
2. **Robustness / evaluation gap.** No study evaluates calibration of random-kernel
   classifiers under test-time corruptions. *We add:* an 18-condition corruption benchmark
   on 114 UCR datasets with shift-detection AUROC. *Verified by:* E4.
3. **Uncertainty-set gap.** Jackknife+ requires n refits in general; no closed-form
   jackknife+ exists for ridge readouts of time-series classifiers. *We add:* exact
   closed-form jackknife+ sets using the exact LOO models f₋ᵢ(x). *Verified by:* E9.
4. **Statistical-validation gap.** PreVal reports win counts on a single split. *We add:*
   5 resamples, Wilcoxon with Holm correction, effect sizes and bootstrap CIs.
5. **Efficiency.** *We quantify* the cost against CV calibration, logistic regression,
   ensembles and Arsenal. *Verified by:* E6.
