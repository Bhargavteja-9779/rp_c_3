# Literature review: closest work and remaining gap

40 papers closest to this study. Every entry is in `manuscript/build/references.json` and was checked against
Crossref metadata, the official proceedings or journal page, or arXiv (`manuscript/refs/*verified*.json`).
Descriptions summarise what each paper does; no result is quoted that is not stated in the paper.

| # | Paper | What it does | What remains open for our problem |
|---|---|---|---|
| 1 | ROCKET (Dempster et al. 2020) | 10,000 random dilated kernels, PPV and max pooling, ridge or logistic readout | Ridge readout gives scores, not probabilities |
| 2 | MiniRocket (Dempster et al. 2021) | Almost deterministic, much faster variant of ROCKET | Same readout limitation |
| 3 | MultiRocket (Tan et al. 2022) | Adds pooling operators and the first-difference series | Same |
| 4 | Hydra (Dempster et al. 2023) | Competing kernels linking convolution and dictionary methods | Same |
| 5 | QUANT (Dempster et al. 2024) | Interval quantiles with extremely randomised trees | Probabilities from trees; behaviour under sensor faults not studied |
| 6 | HIVE-COTE 2.0 (Middlehurst et al. 2021) | Meta-ensemble including the Arsenal (ensemble of ROCKET models) for probabilities | Expensive; global probabilities, no fault awareness |
| 7 | Bake off redux (Middlehurst et al. 2024) | Large benchmark of recent TSC methods, accuracy, log-loss, AUROC | No calibration under shift or faults |
| 8 | Great TSC bake off (Bagnall et al. 2017) | Benchmark protocol with resamples | Accuracy-centred |
| 9 | UCR archive (Dau et al. 2019) | 128 univariate datasets and evaluation practice | Data only |
| 10 | Deep learning for TSC review (Ismail Fawaz et al. 2019) | Review and benchmark of deep TSC models | No uncertainty evaluation |
| 11 | InceptionTime (Ismail Fawaz et al. 2020) | Ensemble of deep Inception networks | GPU-heavy; no fault-aware confidence |
| 12 | Multivariate TSC bake off (Ruiz et al. 2021) | Benchmark for multivariate TSC | Out of our univariate scope |
| 13 | PreVal (Dempster et al. 2025, arXiv) | Calibrates ridge by a global scale fitted on exact LOO predictions | **Closest work**; one global scale, no input-dependent uncertainty |
| 14 | Platt (2000) | Sigmoid calibration of SVM scores with target smoothing | Needs held-out scores; global |
| 15 | Zadrozny & Elkan (2002) | Isotonic regression / multiclass calibration | Needs held-out scores; global |
| 16 | Niculescu-Mizil & Caruana (2005) | Empirical comparison of calibration methods | In-distribution only |
| 17 | Guo et al. (2017) | Temperature scaling for neural networks | Held-out data; global |
| 18 | Kull et al. (2019) | Dirichlet calibration for many classes | Held-out data; global |
| 19 | Silva Filho et al. (2023) | Survey of classifier calibration | Survey |
| 20 | Tibshirani & Efron (2002) | Pre-validation: stacking on cross-validated predictions | No leverage, no shift |
| 21 | Allen (1974) | PRESS / closed-form LOO residuals for least squares | Regression diagnostics only |
| 22 | Golub et al. (1979) | Generalised cross-validation for ridge | Penalty selection only |
| 23 | Cawley & Talbot (2004) | Fast exact LOO for least-squares SVMs | Model selection, not calibration |
| 24 | Hoaglin & Welsch (1978) | Hat matrix and leverage as diagnostics | Not used for calibration |
| 25 | MacKay (1992) | Evidence framework; moderated outputs by predictive variance | Bayesian networks; no closed-form LOO fit of the moderation |
| 26 | Kristiadi et al. (2020) | Last-layer Laplace approximation reduces overconfidence away from data | Needs a Hessian; for ReLU networks |
| 27 | Liu et al. (2020) SNGP | Distance-aware single networks via spectral normalisation and GP output | Requires retraining networks |
| 28 | Liu et al. (2023) | Extended SNGP study | Same |
| 29 | van Amersfoort et al. (2020) DUQ | Distance-aware RBF output layer | Same |
| 30 | Lakshminarayanan et al. (2017) | Deep ensembles | Cost grows with ensemble size |
| 31 | Ovadia et al. (2019) | Calibration under dataset shift; ensembles more robust | Images/text; not random-kernel TSC |
| 32 | Hendrycks & Dietterich (2019) | Corruption benchmark (ImageNet-C) | Images only |
| 33 | Hendrycks & Gimpel (2017) | Max-softmax baseline for misclassification and OOD detection | Detection only, no calibration |
| 34 | Geifman & El-Yaniv (2017) | Selective classification with risk–coverage | Neural networks; no fault benchmark |
| 35 | Um et al. (2017) | Time series augmentation for wearable sensor data | Augmentation for training, not test-time confidence |
| 36 | Iwana & Uchida (2021) | Survey of time series augmentation | Same |
| 37 | Ragab et al. (2023) AdaTime | Domain adaptation benchmark for time series | Needs unlabelled target data; accuracy-centred |
| 38 | Barber et al. (2021) | Jackknife+ predictive inference | Requires n refits in general |
| 39 | Nouretdinov et al. (2001) / Burnaev & Vovk (2014) | Closed-form conformal ridge regression | Regression only |
| 40 | Moody et al. (1984) | Noise stress test for ECG algorithms with recorded noise (NSTDB) | Used here as real-noise test |
| 41 | PTB-XL (Wagner et al. 2020, Sci. Data) | 21,799 clinical 12-lead ECGs with diagnostic statements, recommended patient-separated folds and signal-quality annotations by a technical expert | Quality annotations not used to test whether classifier confidence identifies noisy recordings |
| 42 | PTB-XL benchmarks (Strodthoff et al. 2021, IEEE JBHI) | Benchmarks deep and classical ECG classifiers on PTB-XL tasks | Reports diagnostic accuracy; no evaluation of confidence on noisy records |
| 43 | Learning to defer (Madras et al. 2018, NeurIPS) | Trains a model jointly with a rule that defers cases to a human decision maker | Needs joint training; does not separate label choice from review choice |
| 44 | Consistent learning to defer (Mozannar & Sontag 2020, ICML) | Consistent surrogate loss for deferring to an expert | Same; deep-network setting, no sensor faults |
| 45 | McNemar (1947, Psychometrika) | Test for paired proportions | Used for H8 |
| 46 | Bender et al. 2023 (Stud. Health Technol. Inform.) | Uses the PTB-XL noise annotations to measure how four noise types affect a deep atrial-fibrillation classifier's accuracy | Accuracy only; no confidence or triage |
| 47 | Han et al. 2025 (arXiv 2506.11815) | Diffusion-based anomaly detector for ECG noise, evaluated with PTB-XL lead-level noise labels; notes annotation inconsistencies | Dedicated noise detector, not the classifier's own confidence |
| 48 | Ibrahim et al. 2025 (arXiv 2510.26501) | Lightweight unsupervised anomaly-detection filters in front of ECG classifiers; noise injected into PTB-XL | Injected, not annotated, noise; filters rather than classifier confidence |

Further references used for methods and statistics (not counted above): Romano et al. 2020 and
Angelopoulos et al. 2021 (APS), Sadinle et al. 2019 (LAC), Vovk et al. 2022, Angelopoulos & Bates
2023, Lei et al. 2018, Xu et al. 2023 (conformal prediction); Demšar 2006, Benavoli et al. 2016,
2017, Wilcoxon 1945, Holm 1979, Kerby 2014 (statistics); Hoerl & Kennard 1970, Suykens &
Vandewalle 1999; Abdar et al. 2021, Gawlikowski et al. 2023 (uncertainty surveys); aeon,
scikit-learn, PhysioNet.

**Gap.** Fast random-kernel classifiers obtain probabilities either by one global
transformation (PreVal, Platt, temperature scaling, isotonic) or by costly ensembles, none of
which lowers confidence specifically for recordings that lie far from the training data; the
closed-form ingredients for doing so (LOO identities, leverage, moderated outputs) exist
separately in classical statistics but have not been combined, fitted without held-out data, or
evaluated for fault-aware triage of sensor recordings.

**Second revision (2026-11-14).** Rows 41–48 were added for the PTB-XL study and the hybrid triage design. The gap
sentence is extended: it is also not known whether a classifier's own confidence catches the faults of real clinical
recordings. A web search on 2026-11-14 (queries combining "PTB-XL" with "static_noise", "baseline_drift", "signal
quality", "uncertainty", "out-of-distribution", "noisy ECG") found rows 46–48, which use the annotations (or injected
noise) for accuracy studies and dedicated noise detectors, but no study testing classifier confidence for triage. The
paper therefore states the novelty narrowly and hedged ("to our knowledge"), and cites rows 46–48. Full texts of rows
47–48 were not read; they are cited for what their abstracts state.
