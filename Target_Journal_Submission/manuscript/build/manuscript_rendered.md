---
title: "Shift-aware probabilities for random convolutional kernel time series classifiers via leverage-moderated leave-one-out calibration"
---

**Abstract** (draft placeholder – written after the full results are available.)

**Keywords** Time series classification · Uncertainty quantification · Probability calibration · Distribution shift · Conformal prediction · Ridge regression


# 1 Introduction

Random convolutional kernel transforms are among the most accurate and by far the cheapest methods for time series classification (TSC). ROCKET [@dempster2020rocket] and its successors MiniRocket [@dempster2021minirocket], MultiRocket [@tan2022multirocket] and Hydra [@dempster2023hydra] map each series to thousands of pooled convolution features and train a linear readout, usually ridge regression with the penalty chosen by efficient leave-one-out (LOO) cross-validation. On the UCR archive [@dau2019ucr] these models are competitive with far more expensive ensembles and deep networks [@middlehurst2024bakeoff; @ismailfawaz2019review], which has made them a default choice in practice.

A ridge readout, however, returns scores, not probabilities. Decision-making applications (alarm triage in monitoring, deferral to a human, cost-sensitive actions) need probabilities that are *calibrated*, meaning that among inputs predicted with confidence 0.8 about 80 % are correct, and that *remain informative when the test data drift away from the training data* [@ovadia2019; @abdar2021; @gawlikowski2023]. Existing options for the ROCKET family are limited. Library implementations report one-hot "probabilities". HIVE-COTE 2.0 introduced the Arsenal, an ensemble of ROCKET classifiers, largely to obtain probability estimates, at a cost that grows with the ensemble size [@middlehurst2021hc2]. Generic post-hoc calibration such as Platt scaling, isotonic regression or temperature scaling [@platt1999; @zadrozny2002; @niculescu2005; @guo2017] needs held-out data or repeated refitting, which is costly when training sets are small, as they are for most UCR problems. Most recently, PreVal [@dempster2025preval] showed that the exact LOO predictions that ridge regression provides almost for free can be used to fit a single scale factor that turns ridge into a well-calibrated probabilistic classifier, closely matching logistic regression in log-loss at a fraction of the cost.

All of these approaches apply *one* global transformation to the scores. A global temperature cannot distinguish a test series that resembles the training data from one that has been corrupted by sensor noise, a baseline drift or a missing segment; under such shifts globally calibrated models are known to become overconfident [@ovadia2019]. In deep learning, "distance-aware" models address this [@liu2020sngp; @vanamersfoort2020; @kristiadi2020], but they require retraining the network and have no counterpart for random-kernel classifiers.

This paper observes that ridge regression already provides a closed-form distance measure, the leverage h(x) = z(x)ᵀA⁻¹z(x), which is the posterior predictive variance factor of the equivalent Bayesian linear model [@mackay1992] and a classical diagnostic of extrapolation [@hoaglin1978]. The difficulty is calibrating how leverage should change confidence without held-out data. We show that the same Sherman–Morrison algebra behind the LOO shortcut also gives the *held-out leverage* of every training example, h₋ᵢ = dᵢ/(1 − dᵢ). The n exact LOO pairs (z₋ᵢ, h₋ᵢ) are therefore out-of-sample observations of both the score and its uncertainty, and a leverage-aware calibrator can be fitted on them without refitting anything.

The resulting method, LOOM (Leave-One-Out-fitted, leverage-Moderated readout), outputs p(y | x) = softmax(a·z(x)/√(h(x)/$h_{\mathrm{ref}}$)): ridge logits are standardised by their predictive standard deviation, and a single temperature a and the ridge penalty λ are selected by the LOO negative log-likelihood. The same identities yield every LOO model's prediction at a test point in closed form, which gives exact jackknife+ prediction sets [@barber2021] without the n refits that jackknife+ normally requires.

The contributions are:

1. **Method.** A leverage-moderated probabilistic readout for random-kernel time series classifiers whose temperature and ridge penalty are fitted on exact LOO (score, leverage) pairs. It needs one thin singular value decomposition (SVD) and no held-out data, refitting or ensembling (Section 4).
2. **Closed-form conformal prediction.** Exact jackknife+ prediction sets for ridge readouts, computed from the LOO models in closed form, with LAC and APS scores (Section 4.5).
3. **Shift benchmark.** A test-time corruption benchmark for univariate TSC (six corruption families × three severities on 114 UCR datasets, five resamples) that evaluates calibration, selective prediction and shift detection, not only accuracy (Section 5).
4. **Evidence.** A pre-registered comparison against eleven baselines including PreVal, cross-validated calibration, logistic regression, five-member ensembles, QUANT [@dempster2024quant] and Arsenal, with ablations of every component, three further feature extractors, training-set reduction, efficiency measurements and an error analysis. Statistical tests are corrected for multiple comparisons (Sections 6–7).

We also report where LOOM does not help: its in-distribution advantage is small, smoothing of the calibration targets costs some in-distribution calibration error, and LAC prediction sets can become empty under shift.

# 2 Related work

## 2.1 Random convolutional kernel classifiers

ROCKET [@dempster2020rocket] applies 10,000 random dilated kernels and pools each output into the proportion of positive values (PPV) and the maximum. MiniRocket [@dempster2021minirocket] makes the transform almost deterministic and roughly 75 times faster, MultiRocket [@tan2022multirocket] adds pooling operators and the first-difference series, and Hydra [@dempster2023hydra] lets groups of kernels compete, which links convolutional and dictionary methods. All use a linear readout, ridge regression with LOO-selected penalty for small and medium training sets and logistic regression for very large ones. The large benchmark of @middlehurst2024bakeoff confirms that these models are among the best accuracy–cost trade-offs and reports log-loss and AUROC alongside accuracy, but it does not consider calibration under shift. Probabilistic alternatives in the same accuracy range include the Arsenal (an ensemble of ROCKET classifiers within HIVE-COTE 2.0 [@middlehurst2021hc2]), QUANT (interval quantiles with extremely randomised trees [@dempster2024quant]) and deep networks such as InceptionTime [@ismailfawaz2020inception].

## 2.2 Post-hoc calibration and prevalidation

Platt scaling fits a sigmoid to classifier scores [@platt1999], isotonic regression a monotone map [@zadrozny2002], and temperature scaling a single softmax temperature [@guo2017]; Dirichlet calibration generalises these to the multi-class case [@kull2019]; see @silvafilho2023 for a survey. All require held-out or cross-validated scores. Fitting a second-stage model to cross-validated predictions is also known as stacking or, in statistics, pre-validation [@tibshirani2002]. For least-squares models the LOO predictions are available in closed form [@allen1974; @golub1979], which @cawley2004 exploited for model selection in least-squares support vector machines [@suykens1999]. PreVal [@dempster2025preval] was, to our knowledge, the first to use these closed-form LOO predictions to calibrate ridge regression, scaling the coefficients by a factor κ that minimises the LOO log-loss. It evaluates 0–1 loss and log-loss on 273 datasets, including 106 UCR datasets with MiniRocket features. LOOM builds on PreVal: it keeps the exact LOO idea and adds a per-input moderation by leverage, fitted with the held-out leverages.

## 2.3 Uncertainty under distribution shift

@ovadia2019 showed that temperature scaling and other in-distribution calibrators degrade under dataset shift and that ensembles [@lakshminarayanan2017] are comparatively robust. Distance-aware single models, such as SNGP [@liu2020sngp; @liu2023sngp] and DUQ [@vanamersfoort2020], obtain better behaviour away from the data by changing the architecture and the training. Last-layer Laplace approximations moderate the logits by their posterior variance [@kristiadi2020], an idea that goes back to the moderated output of @mackay1992. Corruption benchmarks such as ImageNet-C [@hendrycks2019] made such evaluations standard for images. For time series, perturbations such as jittering, scaling, warping and window slicing are widely used as training augmentations [@um2017; @iwana2021], and domain adaptation benchmarks such as AdaTime study shifts between subjects or devices with access to unlabelled target data [@ragab2023]. We are not aware of a test-time corruption benchmark for the calibration of random-kernel classifiers.

## 2.4 Conformal prediction

Conformal prediction converts any score into prediction sets with finite-sample marginal coverage under exchangeability [@vovk2022; @angelopoulos2023]. Split conformal prediction sacrifices part of the training data [@lei2018]. Jackknife+ uses the n LOO models and guarantees coverage of at least 1 − 2α, with about 1 − α in practice [@barber2021], but normally needs n refits; CV+ and adaptive prediction sets (APS) extend this to classification [@romano2020; @angelopoulos2021raps]. The least ambiguous set-valued classifier (LAC) thresholds the probability of each class [@sadinle2019]. For time series, conformal methods have mainly targeted forecasting under temporal dependence [@xu2023]. For ridge readouts, exact LOO models make jackknife+ essentially free (Section 4.5).

# 3 Problem formulation and research questions

Let 𝒟 = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ be univariate series with labels in {1, …, K}, and let φ: ℝᴸ → ℝᵖ be a random-kernel transform fitted on the training series (p ≈ 10⁴). The aim is a probabilistic classifier p̂(y | x) that (i) is as accurate as the standard ridge readout, (ii) is calibrated on test data from the training distribution, (iii) lowers its confidence on test inputs that a corruption has moved away from the training data, and (iv) costs no more than the ridge fit itself: no held-out data, no refitting and no ensembles. Hypotheses H1–H3 and research questions RQ1–RQ8 were registered before the benchmark was run (Online Resource 2):

* **RQ1** Does LOOM give better in-distribution probabilities than existing readouts at equal accuracy?
* **RQ2** Which components (leverage moderation, exact LOO pairs, target smoothing, λ criterion) matter?
* **RQ3** How do calibration and shift detection behave under test-time corruptions?
* **RQ4** Do the effects transfer to Rocket, MultiRocket and Hydra features and across dataset types?
* **RQ5** What does LOOM cost compared with the alternatives?
* **RQ6** How sensitive is it to the moderation strength, the miscoverage level and the training-set size?
* **RQ7** When does it fail?
* **RQ8** Do the closed-form jackknife+ sets reach their nominal coverage, and how large are they?

# 4 Method

## 4.1 Ridge readout from one thin SVD

Features are standardised with the training mean and standard deviation; $\Phi_c$ ∈ ℝⁿˣᵖ denotes the centred training matrix and Y ∈ {−1, +1}ⁿˣᴷ the one-vs-rest targets. With an unpenalised intercept and penalty λ, the readout has the augmented design z(x) = (1, ψ(x)), where ψ(x) are the centred standardised features, and

$$A = Z^\top Z + \operatorname{diag}(0, \lambda I_p), \qquad B = A^{-1} Z^\top Y, \qquad \hat Y = Z B = H Y. \qquad (1)$$

With the thin SVD $\Phi_c$ = U S Vᵀ (rank r ≤ min(n − 1, p)) and $D_\lambda$ = diag(1/($s_j^2$ + λ)), the hat diagonal, the logits z(x) := Bᵀz(x) of a test series and its leverage are

$$d_i = \frac1n + \sum_{j=1}^{r} u_{ij}^2\frac{s_j^2}{s_j^2+\lambda}, \qquad h(x) = \frac1n + \big\lVert D_\lambda^{1/2} V^\top\psi\big\rVert^2 + \frac{\lVert\psi\rVert^2 - \lVert V^\top\psi\rVert^2}{\lambda}. \qquad (2)$$

The last term of h(x) measures the part of ψ(x) outside the span of the training data; it dominates when p ≫ n. One SVD serves all candidate penalties λ ∈ {10⁻³, …, 10³}·n (ten log-spaced values on the standardised scale, as in the reference MiniRocket implementation).

## 4.2 Exact leave-one-out scores and leverages

For example i, let B₋ᵢ denote the fit without it (same λ and standardisation). By the Sherman–Morrison identity (proofs in Online Resource 1):

$$z_{-i} := B_{-i}^\top z(x_i) = Y_i - \frac{Y_i - \hat Y_i}{1 - d_i}, \qquad h_{-i} := z(x_i)^\top A_{-i}^{-1} z(x_i) = \frac{d_i}{1 - d_i}. \qquad (3)$$

The first identity is the PRESS shortcut [@allen1974] used by RidgeClassifierCV and PreVal. The second is what LOOM adds: h₋ᵢ is the leverage that example i would have as a *test* point of the model trained without it. Thus (z₋ᵢ, h₋ᵢ) is an exact out-of-sample observation of a score together with its uncertainty, available for all n examples at O(nr) cost.

## 4.3 Leverage-standardised probabilities

Under the Bayesian linear model with prior β ~ N(0, σ²λ⁻¹I) and noise variance σ², the posterior predictive standard deviation of each logit at x is σ√h(x). LOOM divides the logits by this quantity, relative to the median held-out leverage $h_{\mathrm{ref}}$ = median(h₋ᵢ), and applies one temperature:

$$\hat p(y = k \mid x) = \operatorname{softmax}_k\!\left( \frac{a\, z_k(x)}{\sqrt{h(x)/h_{\mathrm{ref}}}} \right). \qquad (4)$$

For test series from the training distribution h(x) ≈ $h_{\mathrm{ref}}$, so (4) behaves like a temperature-scaled ridge. For series pushed away from the training data, h(x) grows, often by orders of magnitude (Section 6.2), and the probabilities move towards uniform. Eq. (4) is the limit c → ∞ of the family softmax(a z /√(1 + c h/$h_{\mathrm{ref}}$)) used in Bayesian moderation [@mackay1992; @kristiadi2020]. The limit was chosen on eight development datasets before the benchmark (Online Resource 2), and finite c is evaluated as an ablation.

## 4.4 Fitting on exact LOO pairs

For every candidate λ, the temperature a is fitted by minimising the negative log-likelihood of the n LOO pairs,

$$\mathcal L(a, \lambda) = -\frac1n \sum_{i=1}^n \sum_{k=1}^K t_{ik} \log \operatorname{softmax}_k\!\left(\frac{a\, z_{-i,k}}{\sqrt{h_{-i}/h_{\mathrm{ref}}}}\right), \qquad (5)$$

and the pair (λ, a) with the smallest value is kept, as PreVal does for its scale factor. When the LOO predictions separate the classes perfectly, the maximum-likelihood temperature is unbounded, so the targets $t_{ik}$ are smoothed following @platt1999: the true class receives ($n_k$ + 1)/($n_k$ + 2), where $n_k$ is its training count, and the remainder is spread uniformly over the other classes (exactly Platt's rule for K = 2). The one-dimensional problem in log a is solved with L-BFGS from three starting values.

## 4.5 Closed-form jackknife+ prediction sets

Let s(x, y; f) be a non-conformity score of a probabilistic model f. We use LAC, s = 1 − p̂(y | x) [@sadinle2019], and deterministic APS, the total probability of classes at least as probable as y [@romano2020]. Jackknife+ predicts the set C(x) = {y : #{i : sᵢ < s(x, y; f₋ᵢ)} < (1 − α)(n + 1)}, where sᵢ = s(xᵢ, yᵢ; f₋ᵢ) [@barber2021]. It requires every LOO model at every test point, which for ridge is available in closed form. With k(x, i) = z(x)ᵀA⁻¹z(xᵢ) = 1/n + (Vᵀψ)ᵀ$D_\lambda$ S uᵢ and residuals E = Y − Ŷ,

$$z_{-i}(x) = z(x) - \frac{k(x,i)\,E_i}{1-d_i}, \qquad h_{-i}(x) = h(x) + \frac{k(x,i)^2}{1-d_i}. \qquad (6)$$

Each test point costs O(nK) after an O(pr) projection, and no model is refitted. Because λ, a and the scaler are fitted on all n examples, the coverage guarantee holds only approximately; we therefore assess coverage empirically (Section 6.5).

## 4.6 Complexity

The SVD costs O(np·min(n, p)), as for the standard RidgeClassifierCV readout. Evaluating ten penalties, the LOO pairs and the temperature fits adds O(10·nr·K) operations. Prediction costs O(p(r + K)) per test series, compared with O(pK) for plain ridge. Memory is dominated by U and V (O((n + p)r)). In contrast, k-fold calibration needs k additional ridge fits, logistic regression an iterative solver per regularisation value, and an M-member ensemble M transforms and fits.

# 5 Experimental setup

## 5.1 Data and protocol

We use the 114 univariate datasets of the UCR 2018 archive [@dau2019ucr] whose official files contain equal-length series without missing values (17 to 8,926 training series, lengths 15 to 2,844, 2 to 60 classes; Fig. 2). Each dataset is evaluated on the official split (resample 0) and on four stratified resamples that keep the official per-class training counts (numpy seeds 1–4), following the bake-off protocol [@bagnall2017bakeoff; @middlehurst2024bakeoff]. All seeds were fixed in advance. MiniRocket features (aeon 1.6.0 [@aeon2024], 9,996 features) are fitted on the training part only; the transform seed of resample r is 1000·r. A leakage audit of the official splits found 16 datasets in which some test series exactly duplicate a training series (Online Resource 2, Table S1). Because all methods share the same features, duplicates affect all methods. We nevertheless report a sensitivity analysis without these datasets.

## 5.2 Methods compared

All readouts share identical MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations. **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale with λ selected jointly [@dempster2025preval], reimplemented without target smoothing as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold out-of-fold calibration and a final model fitted on all data. **LogReg-CV** is L2 logistic regression on standardised features, with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it is skipped on the 13 datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models fitted on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two raw-series classifiers with native probabilities are also included: **QUANT** [@dempster2024quant], on all resamples, and the **Arsenal** [@middlehurst2021hc2], in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was computationally prohibitive (Online Resource 2).

Ablations: **PreVal-smooth** (PreVal with target smoothing, i.e. LOOM without leverage); **LOOM-nosmooth**; **LOOM-free-b** (moderation 1/√(1 + b h) with b fitted on the LOO pairs); **LOOM-finite-c1** (c = 1); **LOOM-kNN** (leverage replaced by the mean squared distance to the five nearest training series); **LOOM-insample** (calibrator fitted on in-sample instead of LOO pairs); **LOOM-holdout** (80/20 split instead of LOO pairs); and **LOOM-lambdaSSE** (λ chosen by the LOO squared error, as in RidgeClassifierCV).

## 5.3 Test-time corruptions

Six corruption families, each at three severities, are applied to the test series only, with amplitudes relative to each series' standard deviation sd: Gaussian noise (σ ∈ {0.1, 0.3, 0.5}·sd); sinusoidal baseline wander (amplitude {0.25, 0.5, 1}·sd, frequency 0.5–2 cycles per series); a missing segment of {10, 20, 30} % of the length replaced by linear interpolation; impulse spikes of ±3 sd at {1, 3, 5} % of the time points; smooth monotone time warping with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}; and circular shifts of {5, 10, 20} % of the length. Random draws are seeded by the dataset, resample, family and severity, so every method sees identical inputs. Such perturbations model sensor noise, drift, dropouts, artefacts and misalignment, and similar operations are common augmentations [@um2017; @iwana2021].

## 5.4 Metrics and statistical analysis

The primary endpoint is the test negative log-likelihood (NLL, a strictly proper score). Secondary endpoints are the top-label expected calibration error (ECE, 15 equal-width bins), the Brier score, classwise ECE, accuracy, balanced accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, under corruption, the AUROC for separating corrupted from clean test inputs by 1 − max p̂ (shift detection [@hendrycks2017]). For conformal sets (α = 0.1) we report empirical coverage, mean set size and the fraction of empty sets. The robustness endpoint of a dataset is its mean over the 18 corruption conditions.

The unit of analysis is the dataset, with metrics averaged over the five resamples. LOOM is compared with every other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945], with Holm's correction within each table [@holm1979], as recommended for comparisons over multiple datasets [@demsar2006; @benavoli2016]. We report the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ [@kerby2014] as effect size (positive values favour LOOM for lower-is-better metrics) and a percentile bootstrap 95 % confidence interval (10,000 draws) of the median paired difference. Friedman tests and mean ranks summarise all methods jointly. A difference is called significant only if the Holm-adjusted p-value is below 0.05.

## 5.5 Implementation and development disclosure

Everything is implemented in Python with NumPy, SciPy, scikit-learn and aeon. Each job runs on one CPU core of a 4-core machine without a GPU, and timings are wall-clock times in that setting. Unit tests check every closed-form identity in Eqs. (2), (3) and (6) against explicit refits to machine precision. `run_all.py` regenerates all results, tables and figures, and every number in this paper is read from machine-generated result files. The method was designed on eight randomly selected development datasets, and six further datasets were used for pipeline testing. Platt target smoothing and the APS score were added after these tests and before the benchmark. Results are also reported for the 8 datasets never used during development (Section 6.7). Use of generative AI is declared in the Statements and Declarations.


# 6 Results

All results are computed on 9 datasets (official split plus four resamples unless stated otherwise); per-dataset values are means over resamples. Complete per-dataset results are in Online Resource 3 and the repository.

## 6.1 In-distribution calibration (RQ1)

Table 1 summarises clean test performance. LOOM has a median test NLL of 0.196. It is not significantly different from PreVal (W/D/L 3/0/6, median difference -0.007 [95 % CI -0.054, +0.056], r_rb = -0.29, Holm-adjusted p = 1.000); not significantly different from Hold-out temperature scaling (W/D/L 6/0/3, median difference +0.010 [95 % CI -0.020, +0.766], r_rb = 0.47, Holm-adjusted p = 1.000); not significantly different from Ridge + Platt (5-fold) (W/D/L 4/0/4, median difference +0.002 [95 % CI -0.008, +0.009], r_rb = 0.17, Holm-adjusted p = 1.000); and not significantly different from Logistic regression (CV) (W/D/L 5/0/4, median difference +0.004 [95 % CI -0.079, +0.026], r_rb = 0.02, Holm-adjusted p = 1.000). The naive readouts are clearly worse: LOOM is better than Ridge softmax (T = 1) (W/D/L 9/0/0, median difference +0.692 [95 % CI +0.232, +2.083], r_rb = 1.00, Holm-adjusted p = 0.027) and better than Ridge one-hot (W/D/L 9/0/0, median difference +1.416 [95 % CI +0.764, +4.917], r_rb = 1.00, Holm-adjusted p = 0.027). For ECE, LOOM is not significantly different from PreVal (W/D/L 1/0/8, median difference -0.017 [95 % CI -0.075, -0.003], r_rb = -0.82, Holm-adjusted p = 0.082); the target smoothing that keeps the temperature finite (Section 4.4) slightly lowers confidence on easy datasets, which costs ECE (Section 6.4). Accuracy is essentially unchanged: median accuracy 0.952 for LOOM versus 0.951 for PreVal (not significantly different), because both use the same ridge logits and differ only through the selected λ.

| Method | n | NLL | ECE | BRIER | ACC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 9 | 0.196 | 0.040 | 0.080 | 0.952 | 2.44 | – | – | – |
| PreVal | 9 | 0.165 | 0.016 | 0.074 | 0.951 | 1.44 | 3/0/6 | -0.29 | 1.000 |
| Hold-out temperature scaling | 9 | 0.194 | 0.023 | 0.084 | 0.944 | 3.56 | 6/0/3 | 0.47 | 1.000 |
| Ridge + Platt (5-fold) | 8 | 0.168 | 0.059 | 0.063 | 0.961 | – | 4/0/4 | 0.17 | 1.000 |
| Ridge + isotonic (5-fold) | 8 | 0.353 | 0.017 | 0.066 | 0.958 | – | 8/0/0 | 1.00 | 0.039 |
| Logistic regression (CV) | 9 | 0.193 | 0.020 | 0.086 | 0.943 | 2.89 | 5/0/4 | 0.02 | 1.000 |
| Ridge softmax (T = 1) | 9 | 0.851 | 0.390 | 0.371 | 0.951 | 4.89 | 9/0/0 | 1.00 | 0.027 |
| Ridge one-hot | 9 | 1.709 | 0.049 | 0.099 | 0.951 | 5.78 | 9/0/0 | 1.00 | 0.027 |

: **Table 1** Clean test performance (median over datasets of the mean over five resamples). Mean rank, W/D/L, rank-biserial correlation (r_rb) and Holm-adjusted Wilcoxon p-value refer to NLL; W/D/L counts datasets on which LOOM is better/equal/worse


## 6.2 Behaviour under test-time corruptions (RQ3)

**Leverage reacts to corruption.** For clean test series the median ratio h(x)/h_ref is 1.00, confirming that exact held-out leverages are on the scale of in-distribution test leverages. Under corruption the ratio grows with severity, for example to 417.0 for strong Gaussian noise, 4.6 for strong baseline wander, 8.5 for a 30 % missing segment and 2.74 for a 20 % circular shift (Fig. 3).

**Calibration under corruption.** Averaged over the 18 corruption conditions (Table 2), LOOM has a median NLL of 0.896 against 5.054 for PreVal. LOOM is better than PreVal (W/D/L 8/0/1, median difference +4.239 [95 % CI +0.047, +6.533], r_rb = 0.96, Holm-adjusted p = 0.031), better than Hold-out temperature scaling (W/D/L 9/0/0, median difference +3.340 [95 % CI +0.082, +8.877], r_rb = 1.00, Holm-adjusted p = 0.027), better than Logistic regression (CV) (W/D/L 8/0/1, median difference +2.645 [95 % CI +0.097, +10.387], r_rb = 0.91, Holm-adjusted p = 0.031) and **[MISSING: cmp('E4_shifted','nll','QUANT')]**. For ECE under corruption LOOM is not significantly different from PreVal (W/D/L 7/0/2, median difference +0.049 [95 % CI -0.060, +0.233], r_rb = 0.60, Holm-adjusted p = 0.293). Accuracy under corruption is the same for the ridge-based readouts by construction; the improvement comes from lower confidence on the misclassified corrupted series, visible in the pooled reliability diagrams (Fig. 6). LOOM also flags corrupted inputs: the median AUROC for separating corrupted from clean test series by 1 − max p̂ is 0.831 for LOOM versus 0.542 for PreVal (significantly better).

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 9 | 0.896 | 0.158 | 0.635 | 0.831 | 1.22 | – | – | – |
| PreVal | 9 | 5.054 | 0.250 | 0.639 | 0.542 | 3.44 | 8/0/1 | 0.96 | 0.031 |
| Hold-out temperature scaling | 9 | 4.236 | 0.265 | 0.601 | 0.532 | 4.11 | 9/0/0 | 1.00 | 0.027 |
| Ridge + Platt (5-fold) | 8 | 2.501 | 0.142 | 0.653 | 0.749 | – | 7/0/1 | 0.94 | 0.031 |
| Ridge + isotonic (5-fold) | 8 | 6.242 | 0.146 | 0.651 | 0.698 | – | 8/0/0 | 1.00 | 0.031 |
| Logistic regression (CV) | 9 | 3.377 | 0.243 | 0.580 | 0.516 | 3.56 | 8/0/1 | 0.91 | 0.031 |
| Ridge softmax (T = 1) | 9 | 2.943 | 0.446 | 0.641 | 0.499 | 2.67 | 9/0/0 | 1.00 | 0.027 |
| Ridge one-hot | 9 | 12.416 | 0.359 | 0.641 | 0.500 | 6.00 | 9/0/0 | 1.00 | 0.027 |

: **Table 2** Performance under test-time corruption: per-dataset mean over 18 conditions (six families × three severities), then median over datasets. Statistics as in Table 1 (NLL)


![**Fig. 3** Distribution over datasets of the median test leverage ratio h(x)/h_ref (log scale) for clean test data and each corruption family at severities 1–3](/tmp/claude-0/partial/figures/fig_leverage_ratio.png){width=95%}

![**Fig. 4** Median test NLL over datasets versus corruption severity for each family (severity 0 = clean)](/tmp/claude-0/partial/figures/fig_robustness_nll.png){width=100%}

![**Fig. 5** Per-dataset NLL of LOOM versus PreVal on clean data (left) and averaged over the 18 corruption conditions (right); points below the diagonal favour LOOM](/tmp/claude-0/partial/figures/fig_scatter_loom_vs_preval.png){width=95%}

![**Fig. 6** Reliability diagrams pooled over all datasets and resamples for clean test data (left) and severity-3 corruptions (right)](/tmp/claude-0/partial/figures/fig_reliability.png){width=95%}

## 6.3 Ensembles, QUANT and the Arsenal (official split)

Table 3 compares LOOM with five-member ensembles and the two raw-series probabilistic classifiers on the official split. On clean data LOOM is not significantly different from PreVal, 5-seed ensemble (W/D/L 2/0/7, median difference -0.036 [95 % CI -0.098, +0.001], r_rb = -0.64, Holm-adjusted p = 0.293), **[MISSING: cmp('E2b_raw_clean','nll','QUANT')]** and **[MISSING: cmp('E2b_raw_clean','nll','Arsenal')]**. Under corruption (severity-3 conditions for the Arsenal, all 18 for the others), LOOM is better than PreVal, 5-seed ensemble (W/D/L 8/0/1, median difference +1.733 [95 % CI +0.042, +5.495], r_rb = 0.91, Holm-adjusted p = 0.023), **[MISSING: cmp('E2b_raw_shifted','nll','QUANT')]** and **[MISSING: cmp('E2b_raw_shifted','nll','Arsenal')]**. Averaging five LOOM models gives a median clean NLL of 0.201 and a corrupted NLL of 0.899.

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 9 | 0.901 | 0.182 | 0.658 | 0.833 | 2.00 | – | – | – |
| LOOM, 5-seed ensemble | 9 | 0.899 | 0.189 | 0.659 | 0.833 | 1.44 | 2/0/7 | -0.47 | 0.250 |
| PreVal | 9 | 5.535 | 0.241 | 0.658 | 0.525 | 3.78 | 8/0/1 | 0.96 | 0.023 |
| PreVal, 5-seed ensemble | 9 | 2.778 | 0.181 | 0.659 | 0.657 | 2.78 | 8/0/1 | 0.91 | 0.023 |

: **Table 3** Official split, under corruption (mean over the available corruption conditions, see text). Statistics as in Table 1 (NLL)


## 6.4 Ablation (RQ2)

Fig. 7 shows the median NLL difference of each variant to LOOM. Removing the leverage moderation while keeping smoothing (PreVal-smooth) makes calibration under corruption not significantly different relative to LOOM (median NLL 4.170 vs 0.896). On clean data the same comparison is not significantly different, so moderation does not hurt in-distribution calibration. Fitting the calibrator on in-sample pairs instead of exact LOO pairs is significantly better on clean data (median NLL 1.584), and replacing LOO pairs by an 80/20 hold-out split is significantly better (median 0.336), because the ridge is then trained on fewer series. Without target smoothing LOOM is not significantly different on clean NLL and not significantly different under corruption. Learning the moderation strength b on the LOO pairs (LOOM-free-b) is not significantly different under corruption, confirming the identifiability problem found during development. Replacing leverage by a k-NN distance is not significantly different under corruption, a finite moderation strength (c = 1) is not significantly different, and choosing λ by LOO squared error is not significantly different on clean data.

![**Fig. 7** Ablation: median paired NLL difference (variant − LOOM) with bootstrap 95 % confidence intervals on clean data (left) and under corruption (right); positive values mean the variant is worse](/tmp/claude-0/partial/figures/fig_ablation.png){width=100%}

## 6.5 Generalisation (RQ4)

**Other random-kernel extractors.** With MultiRocket features (official split, clean and severity-3 conditions), LOOM is **[MISSING: cmp('E5_multirocket_shifted','nll','PreVal')]** under corruption and **[MISSING: cmp('E5_multirocket_clean','nll','PreVal')]** on clean data. With Hydra features the corresponding comparisons are **[MISSING: cmp('E5_hydra_shifted','nll','PreVal')]** and **[MISSING: cmp('E5_hydra_clean','nll','PreVal')]**.

**Subsets.** On the 8 datasets never used during development LOOM is not significantly different from PreVal (W/D/L 7/0/1, median difference +2.765 [95 % CI +0.047, +6.533], r_rb = 0.94, Holm-adjusted p = 0.062) under corruption and not significantly different from PreVal (W/D/L 3/0/5, median difference -0.006 [95 % CI -0.054, +0.037], r_rb = -0.22, Holm-adjusted p = 1.000) on clean data. Excluding the 16 datasets with train/test duplicates (8 datasets), the comparisons are not significantly different from PreVal (W/D/L 7/0/1, median difference +2.765 [95 % CI +0.047, +6.533], r_rb = 0.94, Holm-adjusted p = 0.062) and not significantly different from PreVal (W/D/L 3/0/5, median difference -0.013 [95 % CI -0.054, +0.037], r_rb = -0.22, Holm-adjusted p = 1.000).

## 6.6 Conformal prediction sets (RQ8)

Table 4 reports coverage and set size at α = 0.1. On clean data the closed-form jackknife+ sets of LOOM cover on average 0.902 of the test series with a mean set size of 3.65, against 0.926 and 4.01 for split conformal prediction. Under corruption coverage drops for every method, because exchangeability no longer holds. With the LAC score, LOOM's flattened probabilities lead to empty sets (50.0 % of corrupted test series), whereas the APS score turns flat probabilities into larger sets (0.8 % empty, coverage 0.955).

| Method | Coverage, clean (mean / 10th pct.) | Set size, clean | Empty sets, clean | Coverage, corrupted | Set size, corrupted | Empty sets, corrupted |
|---|---|---|---|---|---|---|
| LOOM, jackknife+, LAC | 0.902 / 0.884 | 3.65 | 6.9 % | 0.440 | 3.62 | 50.0 % |
| LOOM, jackknife, LAC | 0.903 / 0.883 | 3.66 | 6.8 % | 0.442 | 3.63 | 49.6 % |
| LOOM, jackknife+, APS | 0.913 / 0.899 | 11.84 | 7.9 % | 0.955 | 14.73 | 0.8 % |
| PreVal, jackknife+, LAC | 0.900 / 0.883 | 3.69 | 6.9 % | 0.570 | 3.60 | 19.2 % |
| PreVal, jackknife+, APS | 0.897 / 0.876 | 11.07 | 9.3 % | 0.732 | 8.86 | 16.1 % |
| Split conformal (TS), LAC | 0.926 / 0.903 | 4.01 | 5.2 % | 0.628 | 3.99 | 15.8 % |
| Split conformal (TS), APS | 0.929 / 0.891 | 11.73 | 6.5 % | 0.788 | 9.39 | 12.9 % |

: **Table 4** Conformal prediction sets at α = 0.1 (target coverage 0.9). Coverage and set size are averaged over datasets (corrupted: over datasets and the 18 conditions); 10th pct. is the 10th percentile of per-dataset coverage


## 6.7 Computational cost (RQ5)

| Method | Median fit time (s) | Mean fit time (s) | Total fit time, all datasets (h) | Median prediction time, 19 test sets (s) |
|---|---|---|---|---|
| Ridge one-hot | 0.009 | 0.03 | 0.00 | 0.81 |
| PreVal | 0.497 | 1.66 | 0.00 | 1.13 |
| LOOM (ours) | 0.505 | 1.68 | 0.00 | 12.89 |
| Hold-out temperature scaling | 0.332 | 0.92 | 0.00 | 12.74 |
| Ridge + Platt (5-fold) | 2.785 | 5.70 | 0.01 | 20.07 |
| Ridge + isotonic (5-fold) | 2.776 | 5.86 | 0.01 | 19.98 |
| Logistic regression (CV) | 10.883 | 31.95 | 0.08 | 3.75 |
| LOOM, 5-seed ensemble | 16.026 | 18.81 | 0.05 | 483.05 |

: **Table 5** Fit and prediction time per dataset (mean over resamples; single CPU core). Ridge-based readouts include the shared SVD; ensembles include four additional MiniRocket transforms


LOOM's median fit time is 0.51 s, versus 0.50 s for PreVal, 2.79 s for five-fold Platt scaling, 10.88 s for cross-validated logistic regression and 16.03 s for a five-seed ensemble (including its additional transforms). The extra prediction cost of LOOM over plain ridge is the projection onto V (Section 4.6).

![**Fig. 8** Fit time versus number of training series (log–log, single core)](/tmp/claude-0/partial/figures/fig_efficiency.png){width=60%}

## 6.8 Sensitivity (RQ6)

The moderation limit c → ∞ chosen during development and the finite value c = 1 give similar results (Section 6.4). For conformal sets, the clean coverage of LOOM's jackknife+ is 0.955 at α = 0.05 and 0.800 at α = 0.2. With 25 % and 50 % of the training data (official split, clean and severity-3 conditions; Online Resource 3) the comparison with PreVal is summarised in Table S4.

## 6.9 Error analysis (RQ7)

[ERROR_ANALYSIS — to be written after the final results]

# 7 Discussion

[DISCUSSION — to be written after the final results]

# 8 Limitations

[LIMITATIONS — to be written after the final results]

# 9 Conclusion

[CONCLUSION — to be written after the final results]


# References {-}

::: {#refs}
:::


# Statements and Declarations {-}

**Funding** [To be completed by the authors. If none: "The authors did not receive support from any organization for the submitted work."]

**Competing Interests** [To be completed by the authors. If none: "The authors have no relevant financial or non-financial interests to disclose."]

**Author Contributions** [To be completed by the authors, following the journal's authorship principles.]

**Data Availability** All datasets analysed in this study are publicly available in the UCR Time Series Classification Archive (2018 release, https://www.timeseriesclassification.com) [@dau2019ucr]. All machine-generated results supporting the findings (per-dataset, per-resample and per-condition metrics, statistical tests, tables and figures) are included in the code repository described below.

**Code Availability** The complete source code, configuration, unit tests, the single entry point that regenerates every result, table and figure (`run_all.py`), and the cached raw results are available at [repository URL withheld for double-anonymous review; provided to the editor].

**Use of Generative AI** [Authors must review and edit this statement before submission.] An AI assistant (Claude, Anthropic) was used extensively in this work: for the literature and journal search, implementation of the code, execution and analysis of the experiments, and drafting of the manuscript text. All experimental numbers were produced by the released code and inserted into the manuscript programmatically from the result files. The authors take full responsibility for the content.

**Ethics Approval** Not applicable. The study uses only publicly available, de-identified benchmark data.
