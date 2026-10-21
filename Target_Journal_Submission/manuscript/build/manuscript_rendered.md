---
title: "Shift-aware probabilities for random convolutional kernel time series classifiers via leverage-moderated leave-one-out calibration"
---

**Abstract** Random convolutional kernel classifiers such as MiniRocket are accurate and fast, but their ridge readout returns scores rather than probabilities, and existing calibrations apply one global transformation that remains confident when test data drift. We propose LOOM, a closed-form probabilistic readout that divides each ridge logit by the square root of its leverage, the predictive variance factor of the equivalent Bayesian linear model, and fits a single temperature and the ridge penalty on exact leave-one-out pairs of scores and held-out leverages. LOOM needs one thin singular value decomposition, no held-out data, no refitting and no ensembles. The same identities yield exact jackknife+ prediction sets without refitting. On 111 UCR datasets with five resamples and a pre-registered benchmark of 18 test-time corruptions, LOOM matches the strongest closed-form baseline on clean data in negative log-likelihood (NLL), although target smoothing raises its expected calibration error. Under corruption it lowers the median NLL from 1.76 to 0.74, is better on 105 of 111 datasets (Holm-adjusted Wilcoxon p < 0.001), and raises the AUROC for detecting corrupted inputs from 0.56 to 0.77. Its clean jackknife+ sets reach 0.90 coverage at a nominal 0.90 with smaller sets than split conformal prediction. Code and all results are publicly available.

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

All readouts share identical MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations. **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale with λ selected jointly [@dempster2025preval], reimplemented without target smoothing as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold out-of-fold calibration and a final model fitted on all data. **LogReg-CV** is L2 logistic regression on standardised features, with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it is skipped on the four datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models fitted on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two raw-series classifiers with native probabilities are also included: **QUANT** [@dempster2024quant], on all resamples, and the **Arsenal** [@middlehurst2021hc2], in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was computationally prohibitive (Online Resource 2).

Ablations: **PreVal-smooth** (PreVal with target smoothing, i.e. LOOM without leverage); **LOOM-nosmooth**; **LOOM-free-b** (moderation 1/√(1 + b h) with b fitted on the LOO pairs); **LOOM-finite-c1** (c = 1); **LOOM-kNN** (leverage replaced by the mean squared distance to the five nearest training series); **LOOM-insample** (calibrator fitted on in-sample instead of LOO pairs); **LOOM-holdout** (80/20 split instead of LOO pairs); and **LOOM-lambdaSSE** (λ chosen by the LOO squared error, as in RidgeClassifierCV).

## 5.3 Test-time corruptions

Six corruption families, each at three severities, are applied to the test series only, with amplitudes relative to each series' standard deviation sd: Gaussian noise (σ ∈ {0.1, 0.3, 0.5}·sd); sinusoidal baseline wander (amplitude {0.25, 0.5, 1}·sd, frequency 0.5–2 cycles per series); a missing segment of {10, 20, 30} % of the length replaced by linear interpolation; impulse spikes of ±3 sd at {1, 3, 5} % of the time points; smooth monotone time warping with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}; and circular shifts of {5, 10, 20} % of the length. Random draws are seeded by the dataset, resample, family and severity, so every method sees identical inputs. Such perturbations model sensor noise, drift, dropouts, artefacts and misalignment, and similar operations are common augmentations [@um2017; @iwana2021].

## 5.4 Metrics and statistical analysis

The primary endpoint is the test negative log-likelihood (NLL, a strictly proper score). Secondary endpoints are the top-label expected calibration error (ECE, 15 equal-width bins), the Brier score, classwise ECE, accuracy, balanced accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, under corruption, the AUROC for separating corrupted from clean test inputs by 1 − max p̂ (shift detection [@hendrycks2017]). For conformal sets (α = 0.1) we report empirical coverage, mean set size and the fraction of empty sets. The robustness endpoint of a dataset is its mean over the 18 corruption conditions.

The unit of analysis is the dataset, with metrics averaged over the five resamples. LOOM is compared with every other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945], with Holm's correction within each table [@holm1979], as recommended for comparisons over multiple datasets [@demsar2006; @benavoli2016]. We report the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ [@kerby2014] as effect size (positive values favour LOOM for lower-is-better metrics) and a percentile bootstrap 95 % confidence interval (10,000 draws) of the median paired difference. Friedman tests and mean ranks summarise all methods jointly. A difference is called significant only if the Holm-adjusted p-value is below 0.05.

## 5.5 Implementation and development disclosure

Everything is implemented in Python with NumPy, SciPy, scikit-learn and aeon. Each job runs on one CPU core of a 4-core machine without a GPU, and timings are wall-clock times in that setting. Unit tests check every closed-form identity in Eqs. (2), (3) and (6) against explicit refits to machine precision. `run_all.py` regenerates all results, tables and figures, and every number in this paper is read from machine-generated result files. The method was designed on eight randomly selected development datasets, and six further datasets were used for pipeline testing. Platt target smoothing and the APS score were added after these tests and before the benchmark. Results are also reported for the 97 datasets never used during development (Section 6.7). Use of generative AI is declared in the Statements and Declarations.


# 6 Results

All results are computed on 111 datasets (official split plus four resamples unless stated otherwise); per-dataset values are means over resamples. Complete per-dataset results are in Online Resource 3 and the repository.

## 6.1 In-distribution calibration (RQ1)

Table 1 summarises clean test performance. LOOM has a median test NLL of 0.295. It is not significantly different from PreVal (W/D/L 49/0/62, median difference -0.004 [95 % CI -0.009, +0.001], r_rb = -0.05, Holm-adjusted p = 1.000); better than Hold-out temperature scaling (W/D/L 83/0/28, median difference +0.039 [95 % CI +0.020, +0.074], r_rb = 0.62, Holm-adjusted p < 0.001); not significantly different from Ridge + Platt (5-fold) (W/D/L 53/0/54, median difference -0.000 [95 % CI -0.004, +0.004], r_rb = 0.05, Holm-adjusted p = 1.000); and not significantly different from Logistic regression (CV) (W/D/L 53/0/56, median difference -0.001 [95 % CI -0.013, +0.010], r_rb = -0.01, Holm-adjusted p = 1.000). The naive readouts are clearly worse: LOOM is better than Ridge softmax (T = 1) (W/D/L 107/0/4, median difference +0.189 [95 % CI +0.120, +0.287], r_rb = 0.97, Holm-adjusted p < 0.001) and better than Ridge one-hot (W/D/L 100/0/11, median difference +2.673 [95 % CI +1.513, +3.951], r_rb = 0.95, Holm-adjusted p < 0.001). For ECE, LOOM is worse than PreVal (W/D/L 27/0/84, median difference -0.016 [95 % CI -0.029, -0.007], r_rb = -0.74, Holm-adjusted p < 0.001); the target smoothing that keeps the temperature finite (Section 4.4) slightly lowers confidence on easy datasets, which costs ECE (Section 6.4). Accuracy is essentially unchanged: median accuracy 0.915 for LOOM versus 0.915 for PreVal (not significantly different), because both use the same ridge logits and differ only through the selected λ.

| Method | n | NLL | ECE | BRIER | ACC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 111 | 0.295 | 0.077 | 0.142 | 0.915 | 1.95 | – | – | – |
| PreVal | 111 | 0.361 | 0.040 | 0.129 | 0.915 | 1.82 | 49/0/62 | -0.05 | 1.000 |
| Hold-out temperature scaling | 111 | 0.525 | 0.053 | 0.183 | 0.897 | 3.02 | 83/0/28 | 0.62 | < 0.001 |
| Ridge + Platt (5-fold) | 107 | 0.297 | 0.080 | 0.147 | 0.914 | – | 53/0/54 | 0.05 | 1.000 |
| Ridge + isotonic (5-fold) | 107 | 0.681 | 0.044 | 0.154 | 0.908 | – | 93/0/14 | 0.92 | < 0.001 |
| Logistic regression (CV) | 109 | 0.314 | 0.049 | 0.151 | 0.901 | – | 53/0/56 | -0.01 | 1.000 |
| Ridge softmax (T = 1) | 111 | 0.592 | 0.253 | 0.284 | 0.915 | 3.71 | 107/0/4 | 0.97 | < 0.001 |
| Ridge one-hot | 111 | 2.924 | 0.085 | 0.169 | 0.915 | 4.50 | 100/0/11 | 0.95 | < 0.001 |

: **Table 1** Clean test performance (median over datasets of the mean over five resamples). Mean rank, W/D/L, rank-biserial correlation (r_rb) and Holm-adjusted Wilcoxon p-value refer to NLL; W/D/L counts datasets on which LOOM is better/equal/worse


## 6.2 Behaviour under test-time corruptions (RQ3)

**Leverage reacts to corruption.** For clean test series the median ratio h(x)/h_ref is 1.01, confirming that exact held-out leverages are on the scale of in-distribution test leverages. Under corruption the ratio grows with severity, for example to 31.6 for strong Gaussian noise, 6.7 for strong baseline wander, 8.7 for a 30 % missing segment and 3.65 for a 20 % circular shift (Fig. 3).

**Calibration under corruption.** Averaged over the 18 corruption conditions (Table 2), LOOM has a median NLL of 0.739 against 1.757 for PreVal. LOOM is better than PreVal (W/D/L 105/0/6, median difference +0.842 [95 % CI +0.446, +1.470], r_rb = 0.97, Holm-adjusted p < 0.001), better than Hold-out temperature scaling (W/D/L 107/0/4, median difference +1.220 [95 % CI +0.932, +2.198], r_rb = 0.99, Holm-adjusted p < 0.001), better than Logistic regression (CV) (W/D/L 104/0/5, median difference +0.959 [95 % CI +0.663, +1.230], r_rb = 0.97, Holm-adjusted p < 0.001) and **[MISSING: cmp('E4_shifted','nll','QUANT')]**. For ECE under corruption LOOM is better than PreVal (W/D/L 78/0/33, median difference +0.025 [95 % CI +0.017, +0.037], r_rb = 0.43, Holm-adjusted p < 0.001). Accuracy under corruption is the same for the ridge-based readouts by construction; the improvement comes from lower confidence on the misclassified corrupted series, visible in the pooled reliability diagrams (Fig. 6). LOOM also flags corrupted inputs: the median AUROC for separating corrupted from clean test series by 1 − max p̂ is 0.769 for LOOM versus 0.564 for PreVal (significantly better).

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 111 | 0.739 | 0.154 | 0.692 | 0.769 | 1.17 | – | – | – |
| PreVal | 111 | 1.757 | 0.187 | 0.690 | 0.564 | 3.14 | 105/0/6 | 0.97 | < 0.001 |
| Hold-out temperature scaling | 111 | 2.385 | 0.204 | 0.661 | 0.559 | 3.41 | 107/0/4 | 0.99 | < 0.001 |
| Ridge + Platt (5-fold) | 107 | 1.132 | 0.148 | 0.691 | 0.612 | – | 83/0/24 | 0.77 | < 0.001 |
| Ridge + isotonic (5-fold) | 107 | 4.469 | 0.158 | 0.665 | 0.594 | – | 105/0/2 | 1.00 | < 0.001 |
| Logistic regression (CV) | 109 | 1.878 | 0.219 | 0.660 | 0.516 | – | 104/0/5 | 0.97 | < 0.001 |
| Ridge softmax (T = 1) | 111 | 1.088 | 0.257 | 0.687 | 0.530 | 2.29 | 102/0/9 | 0.95 | < 0.001 |
| Ridge one-hot | 111 | 10.804 | 0.313 | 0.687 | 0.500 | 4.99 | 111/0/0 | 1.00 | < 0.001 |

: **Table 2** Performance under test-time corruption: per-dataset mean over 18 conditions (six families × three severities), then median over datasets. Statistics as in Table 1 (NLL)


![**Fig. 3** Distribution over datasets of the median test leverage ratio h(x)/h_ref (log scale) for clean test data and each corruption family at severities 1–3](/tmp/claude-0/partial/figures/fig_leverage_ratio.png){width=95%}

![**Fig. 4** Median test NLL over datasets versus corruption severity for each family (severity 0 = clean)](/tmp/claude-0/partial/figures/fig_robustness_nll.png){width=100%}

![**Fig. 5** Per-dataset NLL of LOOM versus PreVal on clean data (left) and averaged over the 18 corruption conditions (right); points below the diagonal favour LOOM](/tmp/claude-0/partial/figures/fig_scatter_loom_vs_preval.png){width=95%}

![**Fig. 6** Reliability diagrams pooled over all datasets and resamples for clean test data (left) and severity-3 corruptions (right)](/tmp/claude-0/partial/figures/fig_reliability.png){width=95%}

## 6.3 Ensembles, QUANT and the Arsenal (official split)

Table 3 compares LOOM with five-member ensembles and the two raw-series probabilistic classifiers on the official split. On clean data LOOM is worse than PreVal, 5-seed ensemble (W/D/L 31/0/80, median difference -0.021 [95 % CI -0.031, -0.012], r_rb = -0.45, Holm-adjusted p < 0.001), **[MISSING: cmp('E2b_raw_clean','nll','QUANT')]** and **[MISSING: cmp('E2b_raw_clean','nll','Arsenal')]**. Under corruption (severity-3 conditions for the Arsenal, all 18 for the others), LOOM is better than PreVal, 5-seed ensemble (W/D/L 92/0/19, median difference +0.270 [95 % CI +0.186, +0.435], r_rb = 0.86, Holm-adjusted p < 0.001), **[MISSING: cmp('E2b_raw_shifted','nll','QUANT')]** and **[MISSING: cmp('E2b_raw_shifted','nll','Arsenal')]**. Averaging five LOOM models gives a median clean NLL of 0.308 and a corrupted NLL of 0.789.

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 111 | 0.798 | 0.157 | 0.668 | 0.765 | 1.96 | – | – | – |
| LOOM, 5-seed ensemble | 111 | 0.789 | 0.157 | 0.672 | 0.775 | 1.62 | 36/0/75 | -0.47 | < 0.001 |
| PreVal | 111 | 1.488 | 0.186 | 0.682 | 0.546 | 3.61 | 98/0/13 | 0.92 | < 0.001 |
| PreVal, 5-seed ensemble | 111 | 1.367 | 0.175 | 0.672 | 0.578 | 2.80 | 92/0/19 | 0.86 | < 0.001 |

: **Table 3** Official split, under corruption (mean over the available corruption conditions, see text). Statistics as in Table 1 (NLL)


## 6.4 Ablation (RQ2)

Fig. 7 shows the median NLL difference of each variant to LOOM. Removing the leverage moderation while keeping smoothing (PreVal-smooth) makes calibration under corruption significantly better relative to LOOM (median NLL 1.199 vs 0.739). On clean data the same comparison is not significantly different, so moderation does not hurt in-distribution calibration. Fitting the calibrator on in-sample pairs instead of exact LOO pairs is significantly better on clean data (median NLL 0.988), and replacing LOO pairs by an 80/20 hold-out split is significantly better (median 0.443), because the ridge is then trained on fewer series. Without target smoothing LOOM is not significantly different on clean NLL and significantly better under corruption. Learning the moderation strength b on the LOO pairs (LOOM-free-b) is significantly better under corruption, confirming the identifiability problem found during development. Replacing leverage by a k-NN distance is not significantly different under corruption, a finite moderation strength (c = 1) is not significantly different, and choosing λ by LOO squared error is not significantly different on clean data.

![**Fig. 7** Ablation: median paired NLL difference (variant − LOOM) with bootstrap 95 % confidence intervals on clean data (left) and under corruption (right); positive values mean the variant is worse](/tmp/claude-0/partial/figures/fig_ablation.png){width=100%}

## 6.5 Generalisation (RQ4)

**Other random-kernel extractors.** With MultiRocket features (official split, clean and severity-3 conditions), LOOM is **[MISSING: cmp('E5_multirocket_shifted','nll','PreVal')]** under corruption and **[MISSING: cmp('E5_multirocket_clean','nll','PreVal')]** on clean data. With Hydra features the corresponding comparisons are **[MISSING: cmp('E5_hydra_shifted','nll','PreVal')]** and **[MISSING: cmp('E5_hydra_clean','nll','PreVal')]**.

**Subsets.** On the 97 datasets never used during development LOOM is better than PreVal (W/D/L 91/0/6, median difference +0.664 [95 % CI +0.389, +1.085], r_rb = 0.96, Holm-adjusted p < 0.001) under corruption and not significantly different from PreVal (W/D/L 43/0/54, median difference -0.004 [95 % CI -0.009, +0.001], r_rb = -0.03, Holm-adjusted p = 1.000) on clean data. Excluding the 16 datasets with train/test duplicates (96 datasets), the comparisons are better than PreVal (W/D/L 90/0/6, median difference +0.853 [95 % CI +0.489, +1.470], r_rb = 0.97, Holm-adjusted p < 0.001) and not significantly different from PreVal (W/D/L 42/0/54, median difference -0.003 [95 % CI -0.010, +0.001], r_rb = -0.06, Holm-adjusted p = 1.000).

## 6.6 Conformal prediction sets (RQ8)

Table 4 reports coverage and set size at α = 0.1. On clean data the closed-form jackknife+ sets of LOOM cover on average 0.899 of the test series with a mean set size of 1.56, against 0.930 and 2.53 for split conformal prediction. Under corruption coverage drops for every method, because exchangeability no longer holds. With the LAC score, LOOM's flattened probabilities lead to empty sets (25.3 % of corrupted test series), whereas the APS score turns flat probabilities into larger sets (1.5 % empty, coverage 0.905).

| Method | Coverage, clean (mean / 10th pct.) | Set size, clean | Empty sets, clean | Coverage, corrupted | Set size, corrupted | Empty sets, corrupted |
|---|---|---|---|---|---|---|
| LOOM, jackknife+, LAC | 0.899 / 0.866 | 1.56 | 4.5 % | 0.636 | 1.68 | 25.3 % |
| LOOM, jackknife, LAC | 0.900 / 0.864 | 1.53 | 4.4 % | 0.637 | 1.64 | 25.1 % |
| LOOM, jackknife+, APS | 0.909 / 0.879 | 5.30 | 7.4 % | 0.905 | 6.73 | 1.5 % |
| PreVal, jackknife+, LAC | 0.909 / 0.875 | 1.60 | 3.3 % | 0.694 | 1.57 | 9.3 % |
| PreVal, jackknife+, APS | 0.924 / 0.878 | 5.32 | 5.8 % | 0.816 | 5.11 | 10.2 % |
| Split conformal (TS), LAC | 0.930 / 0.872 | 2.53 | 2.1 % | 0.763 | 2.42 | 6.8 % |
| Split conformal (TS), APS | 0.950 / 0.893 | 5.96 | 3.5 % | 0.878 | 5.50 | 5.8 % |

: **Table 4** Conformal prediction sets at α = 0.1 (target coverage 0.9). Coverage and set size are averaged over datasets (corrupted: over datasets and the 18 conditions); 10th pct. is the 10th percentile of per-dataset coverage


## 6.7 Computational cost (RQ5)

| Method | Median fit time (s) | Mean fit time (s) | Total fit time, all datasets (h) | Median prediction time, 19 test sets (s) |
|---|---|---|---|---|
| Ridge one-hot | 0.003 | 0.01 | 0.00 | 0.08 |
| PreVal | 0.122 | 1.44 | 0.04 | 0.10 |
| LOOM (ours) | 0.129 | 1.44 | 0.04 | 1.44 |
| Hold-out temperature scaling | 0.050 | 0.80 | 0.02 | 0.29 |
| Ridge + Platt (5-fold) | 0.241 | 4.91 | 0.15 | 0.35 |
| Ridge + isotonic (5-fold) | 0.248 | 4.92 | 0.15 | 0.35 |
| Logistic regression (CV) | 3.666 | 11.59 | 0.35 | 0.40 |
| LOOM, 5-seed ensemble | 2.204 | 9.71 | 0.30 | 25.61 |
| QUANT | 15.105 | 15.11 | 0.00 | 235.73 |

: **Table 5** Fit and prediction time per dataset (mean over resamples; single CPU core). Ridge-based readouts include the shared SVD; ensembles include four additional MiniRocket transforms


LOOM's median fit time is 0.13 s, versus 0.12 s for PreVal, 0.24 s for five-fold Platt scaling, 3.67 s for cross-validated logistic regression and 2.20 s for a five-seed ensemble (including its additional transforms). The extra prediction cost of LOOM over plain ridge is the projection onto V (Section 4.6).

![**Fig. 8** Fit time versus number of training series (log–log, single core)](/tmp/claude-0/partial/figures/fig_efficiency.png){width=60%}

## 6.8 Sensitivity (RQ6)

The moderation limit c → ∞ chosen during development and the finite value c = 1 give similar results (Section 6.4). For conformal sets, the clean coverage of LOOM's jackknife+ is 0.952 at α = 0.05 and 0.797 at α = 0.2. With 25 % and 50 % of the training data (official split, clean and severity-3 conditions; Online Resource 3) the comparison with PreVal is summarised in Table S4.

## 6.9 Error analysis (RQ7)

**Where the gain comes from.** Across datasets, the NLL improvement of LOOM over PreVal under corruption is strongly correlated with how much the corruptions increase leverage (Spearman ρ = 0.61, p < 0.001, n = 111), which supports the intended mechanism. The share of datasets on which LOOM has the lower NLL grows with severity for every family except time warping. At severity 3 it is 95 % for noise, 88 % for spikes, 89 % for missing segments, 87 % for wander and 75 % for circular shifts, but only 52 % for time warping, whose median leverage ratio stays at 1.25. Smooth warps of up to ±20 % leave the pooled MiniRocket features largely inside the span of the training features: leverage then carries no signal and LOOM behaves like a temperature-scaled ridge. At severity 1 the win rates are 63 %–56 % (noise, missing), i.e. mild corruptions are not reliably distinguished from in-distribution variation.

**Failure modes.** (i) *Over-moderation of shifted but still correct predictions.* The datasets on which LOOM loses most to PreVal under corruption (PigCVP, ECG5000, InsectEPGRegularTrain) keep comparatively high accuracy under corruption (e.g. 0.99 for InsectEPGRegularTrain) while their leverage grows (7.8×), so lowering confidence costs likelihood. Leverage measures distance from the training data, not the probability of error; across datasets the gain under corruption is negatively correlated with the accuracy retained under corruption (ρ = -0.22, p = 0.019). (ii) *Many classes with very few examples per class.* On clean data the gain over PreVal is negatively correlated with the number of classes (ρ = -0.39, p < 0.001). The largest clean losses occur on PigArtPressure, ShapesAll and PigAirwayPressure (52 or 60 classes with about two to ten training series per class), where Platt smoothing assigns each true class only (n_k + 1)/(n_k + 2) of the target mass and the fitted model becomes underconfident. This is also the main source of LOOM's higher in-distribution ECE (Section 6.1).

# 7 Discussion

**What LOOM adds to PreVal.** PreVal showed that exact LOO predictions turn ridge regression into a well-calibrated probabilistic classifier at almost no cost [@dempster2025preval], and our in-distribution results agree: the two are statistically indistinguishable in NLL on clean data. The difference appears when the test data move. A global scale factor keeps the confidence of a corrupted series at the level appropriate for clean data, so its NLL deteriorates quickly (median 1.757 averaged over the corruptions). LOOM uses the second exact LOO quantity, the held-out leverage, to put each prediction on the scale of its own predictive uncertainty. Its median NLL under corruption is 0.739, and its confidence separates corrupted from clean inputs far better than any other readout (median shift AUROC 0.769). The ablation attributes the gain to the leverage moderation: PreVal with the same target smoothing remains far behind under corruption. It also shows that the exact LOO pairs matter, since fitting the same calibrator on in-sample pairs or on a hold-out split is clearly worse.

**Relation to distance-aware deep models.** The effect parallels the distance-aware single models proposed for deep networks [@liu2020sngp; @vanamersfoort2020; @kristiadi2020], which aim to become uncertain away from the training data. For a linear readout on a fixed random-kernel transform, distance awareness is available in closed form through the ridge leverage, and its calibration can be fitted without held-out data. A k-NN distance used in the same role gives similar NLL but worse shift detection than leverage. Leverage measures distance in the regularised geometry that the classifier itself uses, and it is computed from the same decomposition without an additional search over the training set.

**Practical guidance.** LOOM is attractive when (i) training sets are small, so that holding out data for calibration is costly; (ii) test conditions may drift, as in sensor and monitoring applications; and (iii) a single fast model is preferred over an ensemble. Its in-distribution calibration is not better than PreVal's: practitioners who are confident that test data match the training distribution, particularly with many classes and few examples per class, may prefer PreVal (or LOOM without smoothing) for the lowest in-distribution ECE. For uncertainty sets, jackknife+ with LAC gives nominal coverage with the smallest sets on clean data. Under shift, the APS score should be used, because LAC sets become empty when LOOM flattens its probabilities; APS sets are larger but retain coverage close to the nominal level (Table 4).

**Threats to validity.** Conclusions are drawn over datasets (Wilcoxon tests with Holm correction) and hold on the datasets not used during development and on the datasets without train/test duplicates (Section 6.5). They are specific to random-kernel feature extractors with a ridge readout and to the synthetic corruptions studied; Section 8 lists further limitations.

# 8 Limitations

The study has the following limitations.

*Scope of data and shift.* All experiments use univariate, equal-length UCR datasets. The corruptions are synthetic and test-time only: they model sensor noise, drift, dropouts, artefacts and misalignment, but not real deployment shifts such as new subjects, devices or recording sites, and not label shift. Multivariate series and real domain-shift benchmarks [@ragab2023; @ruiz2021] remain to be studied.

*Approximations in the LOO quantities.* The identities in Section 4.2 are exact for fixed standardisation and fixed MiniRocket biases, but both are estimated on all n training series. In addition, λ and the temperature are selected on the same LOO pairs used to fit them. The LOO negative log-likelihood is therefore slightly optimistic, and the jackknife+ coverage guarantee [@barber2021] holds only approximately. We assessed coverage empirically instead.

*Behaviour of the moderation.* Leverage measures distance from the training data in the regularised feature geometry. It does not detect shifts that leave the features inside the span of the training data (for example, label noise or a change of class priors), and it lowers confidence even for shifted inputs that are still classified correctly (Section 6.9). Platt target smoothing, needed when LOO predictions are separable, makes the model underconfident on datasets with very few training series per class, which raised the clean ECE (Section 6.9).

*Compute-driven protocol choices.* All experiments ran on a 4-core CPU without a GPU in a cloud container that restarted several times. For this reason, and decided before results were inspected (Online Resource 2): ensembles, other extractors and the Arsenal were evaluated on the official split only; the Arsenal used aeon's MiniRocket-kernel configuration; Rocket features were not evaluated; logistic regression was skipped for the four datasets with more than 2,000 training series (Crop, ElectricDevices, FordA, FordB); conformal sets were evaluated on at most 1,000 test series per dataset; and the test sets of Crop and ElectricDevices were subsampled to 2,000 series. Deep-learning classifiers such as InceptionTime [@ismailfawaz2020inception] and the full HIVE-COTE 2.0 ensemble [@middlehurst2021hc2] were not run.

*Development exposure.* The method was designed on eight datasets and the pipeline was tested on six more. Results restricted to the remaining datasets are reported (Section 6.5), but some optimism from design choices made on development data cannot be fully excluded.

*Memory.* LOOM stores the factors U and V, i.e. O((n + p)r) numbers. For training sets much larger than those in the UCR archive, randomised or incremental SVDs would be needed, and the exactness of the LOO identities would then be lost.

# 9 Conclusion

We proposed LOOM, a closed-form probabilistic readout for random convolutional kernel time series classifiers. It divides ridge logits by the square root of their leverage and fits one temperature and the ridge penalty on exact leave-one-out score–leverage pairs, so it needs no held-out data, refitting or ensembles. The same identities give exact jackknife+ prediction sets. On 111 UCR datasets with five resamples, LOOM matches the best existing closed-form readout (PreVal) in-distribution in NLL, at the price of a higher ECE caused by target smoothing. Under 18 test-time corruptions it reduces the median NLL from 1.757 to 0.739, better than every other readout, ensemble and raw-series classifier evaluated, and its confidence detects corrupted inputs much better (median AUROC 0.769 vs 0.564). The fit costs essentially the same as the ridge readout itself. Future work includes multivariate series, real deployment shifts, adaptive smoothing for many-class few-shot problems, and moderation signals that also respond to shifts that leave the feature span unchanged.


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
