---
title: "Shift-aware probabilities for random convolutional kernel time series classifiers via leverage-moderated leave-one-out calibration"
---

**Abstract** Random convolutional kernel classifiers such as MiniRocket are accurate and fast, but their ridge readout returns scores rather than probabilities, and existing calibrations apply one global transformation that remains confident when test data drift. We propose LOOM, a closed-form probabilistic readout that divides each ridge logit by the square root of its leverage, the predictive variance factor of the equivalent Bayesian linear model, and fits a single temperature and the ridge penalty on exact leave-one-out pairs of scores and held-out leverages. LOOM needs one thin singular value decomposition, no held-out data, no refitting and no ensembles. The same identities yield exact jackknife+ prediction sets without refitting. On 114 UCR datasets with five resamples and a pre-registered benchmark of 18 test-time corruptions, LOOM matches the strongest closed-form baseline on clean data in negative log-likelihood (NLL), although target smoothing raises its expected calibration error. Under corruption it lowers the median NLL from 1.73 to 0.74, is better on 108 of 114 datasets (Holm-adjusted Wilcoxon p < 0.001), is on par with the tree-based QUANT classifier on average but worse than it under the strongest corruptions, and raises the AUROC for detecting corrupted inputs from 0.56 to 0.77. Under a natural cross-axis sensor shift that barely changes leverage, the gain is small. Its clean jackknife+ sets reach 0.90 coverage at a nominal 0.90 with smaller sets than split conformal prediction. Code and all results are publicly available.

**Keywords** Time series classification · Uncertainty quantification · Probability calibration · Distribution shift · Conformal prediction · Ridge regression


# 1 Introduction

Random convolutional kernel transforms are among the most accurate and by far the cheapest methods for time series classification (TSC). ROCKET [@dempster2020rocket] and its successors MiniRocket [@dempster2021minirocket], MultiRocket [@tan2022multirocket] and Hydra [@dempster2023hydra] map each series to thousands of pooled convolution features and train a linear readout, usually ridge regression with the penalty chosen by efficient leave-one-out (LOO) cross-validation. On the UCR archive [@dau2019ucr] these models are competitive with far more expensive ensembles and deep networks [@middlehurst2024bakeoff; @ismailfawaz2019review], which has made them a default choice in practice.

A ridge readout, however, returns scores, not probabilities. Decision-making applications (alarm triage in monitoring, deferral to a human, cost-sensitive actions) need probabilities that are *calibrated*, meaning that among inputs predicted with confidence 0.8 about 80 % are correct, and that *remain informative when the test data drift away from the training data* [@ovadia2019; @abdar2021; @gawlikowski2023]. Existing options for the ROCKET family are limited. Library implementations report one-hot "probabilities". HIVE-COTE 2.0 introduced the Arsenal, an ensemble of ROCKET classifiers, largely to obtain probability estimates, at a cost that grows with the ensemble size [@middlehurst2021hc2]. Generic post-hoc calibration such as Platt scaling, isotonic regression or temperature scaling [@platt1999; @zadrozny2002; @niculescu2005; @guo2017] needs held-out data or repeated refitting, which is costly when training sets are small, as they are for most UCR problems. Most recently, PreVal [@dempster2025preval] showed that the exact LOO predictions that ridge regression provides almost for free can be used to fit a single scale factor that turns ridge into a well-calibrated probabilistic classifier, closely matching logistic regression in log-loss at a fraction of the cost.

All of these approaches apply *one* global transformation to the scores. A global temperature cannot distinguish a test series that resembles the training data from one that has been corrupted by sensor noise, a baseline drift or a missing segment; under such shifts globally calibrated models are known to become overconfident [@ovadia2019]. In deep learning, "distance-aware" models address this [@liu2020sngp; @vanamersfoort2020; @kristiadi2020], but they require retraining the network and have no counterpart for random-kernel classifiers.

This paper observes that ridge regression already provides a closed-form distance measure, the leverage h(x) = x̃(x)ᵀA⁻¹x̃(x) (with x̃(x) the intercept-augmented, standardised features), which is the posterior predictive variance factor of the equivalent Bayesian linear model [@mackay1992] and a classical diagnostic of extrapolation [@hoaglin1978]. The difficulty is calibrating how leverage should change confidence without held-out data. We show that the same Sherman–Morrison algebra behind the LOO shortcut also gives the *held-out leverage* of every training example, h₋ᵢ = dᵢ/(1 − dᵢ). The n exact LOO pairs (z₋ᵢ, h₋ᵢ) are therefore out-of-sample observations of both the score and its uncertainty, and a leverage-aware calibrator can be fitted on them without refitting anything.

The resulting method, LOOM (Leave-One-Out-fitted, leverage-Moderated readout), outputs p(y | x) = softmax(a·z(x)/√(h(x)/$h_{\mathrm{ref}}$)): ridge logits are standardised by their predictive standard deviation, and a single temperature a and the ridge penalty λ are selected by the LOO negative log-likelihood. The same identities yield every LOO model's prediction at a test point in closed form, which gives exact jackknife+ prediction sets [@barber2021] without the n refits that jackknife+ normally requires.

The contributions are:

1. **Method.** A leverage-moderated probabilistic readout for random-kernel time series classifiers whose temperature and ridge penalty are fitted on exact LOO (score, leverage) pairs. It needs one thin singular value decomposition (SVD) and no held-out data, refitting or ensembling (Section 4).
2. **Closed-form conformal prediction.** Exact jackknife+ prediction sets for ridge readouts, computed from the LOO models in closed form, with LAC and APS scores (Section 4.5).
3. **Shift benchmark.** A test-time corruption benchmark for univariate TSC (six corruption families × three severities on 114 UCR datasets, five resamples) that evaluates calibration, selective prediction and shift detection, not only accuracy (Section 5).
4. **Evidence.** A pre-registered comparison against eleven baselines including PreVal, cross-validated calibration, logistic regression, five-member ensembles, QUANT [@dempster2024quant] and Arsenal, with ablations of every component, three further feature extractors, training-set reduction, efficiency measurements and an error analysis. Statistical tests are corrected for multiple comparisons (Sections 6–7).

We also report where LOOM does not help: its in-distribution advantage is small, smoothing of the calibration targets costs some in-distribution calibration error, a tree-based raw-series classifier (QUANT) is more robust to the strongest corruptions, shifts that leave the random-kernel features in the span of the training data are not detected, and LAC prediction sets can become empty under shift.

# 2 Related work

## 2.1 Random convolutional kernel classifiers

ROCKET [@dempster2020rocket] applies 10,000 random dilated kernels and pools each output into the proportion of positive values (PPV) and the maximum. MiniRocket [@dempster2021minirocket] makes the transform almost deterministic and roughly 75 times faster, MultiRocket [@tan2022multirocket] adds pooling operators and the first-difference series, and Hydra [@dempster2023hydra] lets groups of kernels compete, which links convolutional and dictionary methods. All use a linear readout, ridge regression with LOO-selected penalty for small and medium training sets and logistic regression for very large ones. The large benchmark of @middlehurst2024bakeoff confirms that these models are among the best accuracy–cost trade-offs and reports log-loss and AUROC alongside accuracy, but it does not consider calibration under shift. Probabilistic alternatives in the same accuracy range include the Arsenal (an ensemble of ROCKET classifiers within HIVE-COTE 2.0 [@middlehurst2021hc2]), QUANT (interval quantiles with extremely randomised trees [@dempster2024quant]) and deep networks such as InceptionTime [@ismailfawaz2020inception].

## 2.2 Post-hoc calibration and prevalidation

Platt scaling fits a sigmoid to classifier scores [@platt1999], isotonic regression a monotone map [@zadrozny2002], and temperature scaling a single softmax temperature [@guo2017]; Dirichlet calibration generalises these to the multi-class case [@kull2019]; see @silvafilho2023 for a survey. All require held-out or cross-validated scores. Fitting a second-stage model to cross-validated predictions is also known as stacking or, in statistics, pre-validation [@tibshirani2002]. For least-squares models the LOO predictions are available in closed form [@allen1974; @golub1979], which @cawley2004 exploited for model selection in least-squares support vector machines [@suykens1999]. PreVal [@dempster2025preval] was, to our knowledge, the first to use these closed-form LOO predictions to calibrate ridge regression, scaling the coefficients by a factor κ that minimises the LOO log-loss. It evaluates 0–1 loss and log-loss on 273 datasets, including 106 UCR datasets with MiniRocket features. LOOM builds on PreVal: it keeps the exact LOO idea and adds a per-input moderation by leverage, fitted with the held-out leverages.

## 2.3 Uncertainty under distribution shift

@ovadia2019 showed that temperature scaling and other in-distribution calibrators degrade under dataset shift and that ensembles [@lakshminarayanan2017] are comparatively robust. Distance-aware single models, such as SNGP [@liu2020sngp; @liu2023sngp] and DUQ [@vanamersfoort2020], obtain better behaviour away from the data by changing the architecture and the training. Last-layer Laplace approximations moderate the logits by their posterior variance [@kristiadi2020], an idea that goes back to the moderated output of @mackay1992. Corruption benchmarks such as ImageNet-C [@hendrycks2019] made such evaluations standard for images. For time series, perturbations such as jittering, scaling, warping and window slicing are widely used as training augmentations [@um2017; @iwana2021], and domain adaptation benchmarks such as AdaTime study shifts between subjects or devices with access to unlabelled target data [@ragab2023]. We are not aware of a test-time corruption benchmark for the calibration of random-kernel classifiers.

## 2.4 Conformal prediction

Conformal prediction converts any score into prediction sets with finite-sample marginal coverage under exchangeability [@vovk2022; @angelopoulos2023]. Split conformal prediction sacrifices part of the training data [@lei2018]. Jackknife+ uses the n LOO models and guarantees coverage of at least 1 − 2α, with about 1 − α in practice [@barber2021], but normally needs n refits; CV+ and adaptive prediction sets (APS) extend this to classification [@romano2020; @angelopoulos2021raps]. The least ambiguous set-valued classifier (LAC) thresholds the probability of each class [@sadinle2019]. For time series, conformal methods have mainly targeted forecasting under temporal dependence [@xu2023]. For ridge *regression*, full conformal prediction can be computed in closed form [@nouretdinov2001; @burnaev2014]. We use the analogous exact LOO identities to obtain jackknife+ sets for *classification* with leverage-moderated probability scores (Section 4.5).

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

Features are standardised with the training mean and standard deviation; $\Phi_c$ ∈ ℝⁿˣᵖ denotes the centred training matrix and Y ∈ {−1, +1}ⁿˣᴷ the one-vs-rest targets. With an unpenalised intercept and penalty λ, the readout uses the augmented feature vector x̃(x) = (1, ψ(x)), where ψ(x) are the centred standardised features; with X̃ stacking the training vectors,

$$A = \tilde X^\top \tilde X + \operatorname{diag}(0, \lambda I_p), \qquad B = A^{-1} \tilde X^\top Y, \qquad \hat Y = \tilde X B = H Y. \qquad (1)$$

With the thin SVD $\Phi_c$ = U S Vᵀ (rank r ≤ min(n − 1, p)) and $D_\lambda$ = diag(1/($s_j^2$ + λ)), the hat diagonal, the logits z(x) := Bᵀx̃(x) of a test series and its leverage h(x) := x̃(x)ᵀA⁻¹x̃(x) are

$$d_i = \frac1n + \sum_{j=1}^{r} u_{ij}^2\frac{s_j^2}{s_j^2+\lambda}, \qquad h(x) = \frac1n + \big\lVert D_\lambda^{1/2} V^\top\psi\big\rVert^2 + \frac{\lVert\psi\rVert^2 - \lVert V^\top\psi\rVert^2}{\lambda}. \qquad (2)$$

The last term of h(x) measures the part of ψ(x) outside the span of the training data; it dominates when p ≫ n. One SVD serves all candidate penalties λ ∈ {10⁻³, …, 10³}·n (ten log-spaced values on the standardised scale, as in the reference MiniRocket implementation).

## 4.2 Exact leave-one-out scores and leverages

For example i, let B₋ᵢ denote the fit without it (same λ and standardisation). By the Sherman–Morrison identity (proofs in Online Resource 1):

$$z_{-i} := B_{-i}^\top \tilde x_i = Y_i - \frac{Y_i - \hat Y_i}{1 - d_i}, \qquad h_{-i} := \tilde x_i^\top A_{-i}^{-1} \tilde x_i = \frac{d_i}{1 - d_i}. \qquad (3)$$

The first identity is the PRESS shortcut [@allen1974] used by RidgeClassifierCV and PreVal. The second is what LOOM adds: h₋ᵢ is the leverage that example i would have as a *test* point of the model trained without it. Thus (z₋ᵢ, h₋ᵢ) is an exact out-of-sample observation of a score together with its uncertainty, available for all n examples at O(nr) cost.

## 4.3 Leverage-standardised probabilities

Under the Bayesian linear model with prior β ~ N(0, σ²λ⁻¹I) and noise variance σ², the posterior predictive standard deviation of each logit at x is σ√h(x). LOOM divides the logits by this quantity, relative to the median held-out leverage $h_{\mathrm{ref}}$ = median(h₋ᵢ), and applies one temperature:

$$\hat p(y = k \mid x) = \operatorname{softmax}_k\!\left( \frac{a\, z_k(x)}{\sqrt{h(x)/h_{\mathrm{ref}}}} \right). \qquad (4)$$

For test series from the training distribution h(x) ≈ $h_{\mathrm{ref}}$, so (4) behaves like a temperature-scaled ridge. For series pushed away from the training data, h(x) grows, often by orders of magnitude (Section 6.2), and the probabilities move towards uniform. The scaling uses √h rather than the √(1 + h) of a regression prediction interval because only the epistemic (parameter) part of the uncertainty should be moderated; the aleatoric part of a classification decision is captured by the softmax and its temperature. Eq. (4) is the limit c → ∞ of the family softmax(a z /√(1 + c h/$h_{\mathrm{ref}}$)) used in Bayesian moderation [@mackay1992; @kristiadi2020]. The limit was chosen on eight development datasets before the benchmark (Online Resource 2), and finite c is evaluated as an ablation.

## 4.4 Fitting on exact LOO pairs

For every candidate λ, the temperature a is fitted by minimising the negative log-likelihood of the n LOO pairs,

$$\mathcal L(a, \lambda) = -\frac1n \sum_{i=1}^n \sum_{k=1}^K t_{ik} \log \operatorname{softmax}_k\!\left(\frac{a\, z_{-i,k}}{\sqrt{h_{-i}/h_{\mathrm{ref}}}}\right), \qquad (5)$$

and the pair (λ, a) with the smallest value is kept, as PreVal does for its scale factor. When the LOO predictions separate the classes perfectly, the maximum-likelihood temperature is unbounded, so the targets $t_{ik}$ are smoothed following @platt1999: the true class receives ($n_k$ + 1)/($n_k$ + 2), where $n_k$ is its training count, and the remainder is spread uniformly over the other classes (exactly Platt's rule for K = 2). The one-dimensional problem in log a is solved with L-BFGS from three starting values.

## 4.5 Closed-form jackknife+ prediction sets

Let s(x, y; f) be a non-conformity score of a probabilistic model f. We use LAC, s = 1 − p̂(y | x) [@sadinle2019], and deterministic APS, the total probability of classes at least as probable as y [@romano2020]. Jackknife+ predicts the set C(x) = {y : #{i : sᵢ < s(x, y; f₋ᵢ)} < (1 − α)(n + 1)}, where sᵢ = s(xᵢ, yᵢ; f₋ᵢ) [@barber2021]. It requires every LOO model at every test point, which for ridge is available in closed form. With k(x, i) = x̃(x)ᵀA⁻¹x̃ᵢ = 1/n + (Vᵀψ)ᵀ$D_\lambda$ S uᵢ and residuals E = Y − Ŷ,

$$z_{-i}(x) = z(x) - \frac{k(x,i)\,E_i}{1-d_i}, \qquad h_{-i}(x) = h(x) + \frac{k(x,i)^2}{1-d_i}. \qquad (6)$$

Each test point costs O(nK) after an O(pr) projection, and no model is refitted. This parallels the closed-form conformal ridge regression of @nouretdinov2001, here applied to the jackknife+ construction and to classification scores that depend on both the LOO logits and the LOO leverages. Because λ, a and the scaler are fitted on all n examples, the coverage guarantee holds only approximately; we therefore assess coverage empirically (Section 6.5).

![**Fig. 1** Overview of LOOM. One thin SVD of the standardised random-kernel features gives, for every candidate penalty λ, the exact leave-one-out logits and held-out leverages; λ and the temperature a are fitted on these pairs. At test time the logits are divided by the square root of the relative leverage, and exact leave-one-out models give jackknife+ prediction sets](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig1_method.png){width=100%}

## 4.6 Complexity

The SVD costs O(np·min(n, p)), as for the standard RidgeClassifierCV readout. Evaluating ten penalties, the LOO pairs and the temperature fits adds O(10·nr·K) operations. Prediction costs O(p(r + K)) per test series, compared with O(pK) for plain ridge. Memory is dominated by U and V (O((n + p)r)). In contrast, k-fold calibration needs k additional ridge fits, logistic regression an iterative solver per regularisation value, and an M-member ensemble M transforms and fits.

# 5 Experimental setup

## 5.1 Data and protocol

We use the 114 univariate datasets of the UCR 2018 archive [@dau2019ucr] whose official files contain equal-length series without missing values (17 to 8,926 training series, lengths 15 to 2,844, 2 to 60 classes; Fig. 2). Each dataset is evaluated on the official split (resample 0) and on four stratified resamples that keep the official per-class training counts (numpy seeds 1–4), following the bake-off protocol [@bagnall2017bakeoff; @middlehurst2024bakeoff]. All seeds were fixed in advance. MiniRocket features (aeon 1.6.0 [@aeon2024], 9,996 features) are fitted on the training part only; the transform seed of resample r is 1000·r. A leakage audit of the official splits found 16 datasets in which some test series exactly duplicate a training series (Online Resource 2, Table S1). Because all methods share the same features, duplicates affect all methods. We nevertheless report a sensitivity analysis without these datasets.

![**Fig. 2** Distribution of training-set size, series length and number of classes over the UCR datasets used](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_datasets.png){width=100%}

## 5.2 Methods compared

All readouts share identical MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations. **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale with λ selected jointly [@dempster2025preval], reimplemented without target smoothing as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split: the ridge (λ by LOO squared error) is fitted on 80 % and the temperature on the remaining 20 %. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold out-of-fold calibration and a final model fitted on all data. **LogReg-CV** is L2 logistic regression on standardised features, with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it is skipped on the four datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models fitted on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two raw-series classifiers with native probabilities are also included: **QUANT** [@dempster2024quant], on all resamples, and the **Arsenal** [@middlehurst2021hc2], in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was computationally prohibitive (Online Resource 2).

Ablations: **PreVal-smooth** (PreVal with target smoothing, i.e. LOOM without leverage); **LOOM-nosmooth**; **LOOM-free-b** (moderation 1/√(1 + b h) with b fitted on the LOO pairs); **LOOM-finite-c1** (c = 1); **LOOM-kNN** (leverage replaced by the mean squared Euclidean distance to the five nearest training series in the standardised feature space); **LOOM-insample** (calibrator fitted on in-sample instead of LOO pairs); **LOOM-holdout** (80/20 split instead of LOO pairs); and **LOOM-lambdaSSE** (λ chosen by the LOO squared error, as in RidgeClassifierCV).

## 5.3 Test-time corruptions

Six corruption families, each at three severities, are applied to the test series only, with amplitudes relative to each series' standard deviation sd: Gaussian noise (σ ∈ {0.1, 0.3, 0.5}·sd); sinusoidal baseline wander (amplitude {0.25, 0.5, 1}·sd, frequency 0.5–2 cycles per series); a missing segment of {10, 20, 30} % of the length replaced by linear interpolation; impulse spikes of ±3 sd at {1, 3, 5} % of the time points; smooth monotone time warping with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}; and circular shifts of {5, 10, 20} % of the length. Random draws are seeded by the dataset, resample, family and severity, so every method sees identical inputs. Such perturbations model sensor noise, drift, dropouts, artefacts and misalignment, and similar operations are common augmentations [@um2017; @iwana2021].

## 5.4 Metrics and statistical analysis

The primary endpoint is the test negative log-likelihood (NLL, a strictly proper score). Secondary endpoints are the top-label expected calibration error (ECE, 15 equal-width bins), the Brier score, classwise ECE, accuracy, balanced accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, under corruption, the AUROC for separating corrupted from clean test inputs by 1 − max p̂ (shift detection [@hendrycks2017]). For conformal sets (α = 0.1) we report empirical coverage, mean set size and the fraction of empty sets. The robustness endpoint of a dataset is its mean over the 18 corruption conditions.

The unit of analysis is the dataset, with metrics averaged over the five resamples. LOOM is compared with every other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945], with Holm's correction within each table [@holm1979], as recommended for comparisons over multiple datasets [@demsar2006; @benavoli2016]. Paired differences are always reported as competitor − LOOM, so positive values favour LOOM for lower-is-better metrics (NLL, ECE, Brier) and negative values favour LOOM for accuracy and AUROC. We report the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ [@kerby2014] as effect size, computed on the same differences (for NLL, ECE and Brier, positive values favour LOOM) and a percentile bootstrap 95 % confidence interval (10,000 draws) of the median paired difference. Friedman tests and mean ranks summarise all methods jointly. A difference is called significant only if the Holm-adjusted p-value is below 0.05. In the text, comparisons are reported as (W/D/L; Δ = median paired difference [bootstrap 95 % CI]; r_rb; p), where W/D/L counts the datasets on which LOOM is better/equal/worse and p is the Holm-adjusted Wilcoxon p-value. Spearman correlations in Section 6.9 are descriptive, and their p-values are not adjusted.

## 5.5 Implementation and development disclosure

Everything is implemented in Python with NumPy, SciPy, scikit-learn and aeon. Each job runs on one CPU core of a 4-core machine without a GPU, and timings are wall-clock times in that setting. Unit tests check every closed-form identity in Eqs. (2), (3) and (6) against explicit refits to machine precision. `run_all.py` regenerates all results, tables and figures, and every number in this paper is read from machine-generated result files. The method was designed on eight randomly selected development datasets, and six further datasets were used for pipeline testing. Platt target smoothing and the APS score were added after these tests and before the benchmark. Results are also reported for the 100 datasets never used during development (Section 6.7). **Use of a large language model.** As required by the journal, we document here that an AI assistant (Claude, Anthropic) was used for the literature search, for writing and running the code, and for drafting the text. All numerical results were produced by the released code and inserted into the manuscript automatically from the result files. The authors take full responsibility for the content (see also the Statements and Declarations).


# 6 Results

All results are computed on 114 datasets (official split plus four resamples unless stated otherwise); per-dataset values are means over resamples. Complete per-dataset results are in Online Resource 3 and the repository.

## 6.1 In-distribution calibration (RQ1)

Table 1 summarises clean test performance. LOOM has a median test NLL of 0.295. It is not significantly different from PreVal (W/D/L 50/0/64; Δ = -0.003 [-0.007, +0.001]; r_rb = -0.04; p = 1.000); better than Hold-out temperature scaling (W/D/L 86/0/28; Δ = +0.038 [+0.020, +0.070]; r_rb = 0.62; p < 0.001); not significantly different from Ridge + Platt (5-fold) (W/D/L 54/0/56; Δ = -0.000 [-0.003, +0.004]; r_rb = 0.05; p = 1.000); and not significantly different from Logistic regression (CV) (W/D/L 53/0/56; Δ = -0.001 [-0.013, +0.010]; r_rb = -0.01; p = 1.000). The naive readouts are clearly worse: LOOM is better than Ridge softmax (T = 1) (W/D/L 110/0/4; Δ = +0.184 [+0.119, +0.273]; r_rb = 0.97; p < 0.001) and better than Ridge one-hot (W/D/L 103/0/11; Δ = +2.696 [+1.578, +3.951]; r_rb = 0.96; p < 0.001). For ECE, LOOM is worse than PreVal (W/D/L 29/0/85; Δ = -0.015 [-0.026, -0.006]; r_rb = -0.71; p < 0.001); the target smoothing that keeps the temperature finite (Section 4.4) makes the model underconfident on datasets with few training series per class, which costs ECE (Sections 6.4 and 6.9). Accuracy is essentially unchanged: median accuracy 0.914 for LOOM versus 0.914 for PreVal (not significantly different), because both use the same ridge logits and differ only through the selected λ.

| Method | n | NLL | ECE | BRIER | ACC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 0.295 | 0.075 | 0.144 | 0.914 | 1.94 | – | – | – |
| PreVal | 114 | 0.360 | 0.040 | 0.134 | 0.914 | 1.81 | 50/0/64 | -0.04 | 1.000 |
| Hold-out temperature scaling | 114 | 0.524 | 0.053 | 0.177 | 0.896 | 3.02 | 86/0/28 | 0.62 | < 0.001 |
| Ridge + Platt (5-fold) | 110 | 0.294 | 0.080 | 0.148 | 0.912 | – | 54/0/56 | 0.05 | 1.000 |
| Ridge + isotonic (5-fold) | 110 | 0.670 | 0.044 | 0.157 | 0.904 | – | 96/0/14 | 0.92 | < 0.001 |
| Logistic regression (CV) | 109 | 0.314 | 0.049 | 0.151 | 0.901 | – | 53/0/56 | -0.01 | 1.000 |
| QUANT | 112 | 0.384 | 0.117 | 0.207 | 0.899 | – | 76/0/36 | 0.43 | < 0.001 |
| Ridge softmax (T = 1) | 114 | 0.588 | 0.249 | 0.284 | 0.914 | 3.71 | 110/0/4 | 0.97 | < 0.001 |
| Ridge one-hot | 114 | 2.959 | 0.086 | 0.171 | 0.914 | 4.52 | 103/0/11 | 0.96 | < 0.001 |

: **Table 1** Clean test performance (median over datasets of the mean over five resamples). Mean rank, W/D/L, rank-biserial correlation (r_rb) and Holm-adjusted Wilcoxon p-value refer to NLL; W/D/L counts datasets on which LOOM is better/equal/worse


## 6.2 Behaviour under test-time corruptions (RQ3)

**Leverage reacts to corruption.** For clean test series the median ratio h(x)/h_ref is 1.01, confirming that exact held-out leverages are on the scale of in-distribution test leverages. Under corruption the ratio grows with severity, for example to 30.3 for strong Gaussian noise, 6.5 for strong baseline wander, 8.9 for a 30 % missing segment and 3.5 for a 20 % circular shift (Fig. 3).

**Calibration under corruption.** Averaged over the 18 corruption conditions (Table 2), LOOM has a median NLL of 0.738 against 1.728 for PreVal. LOOM is better than PreVal (W/D/L 108/0/6; Δ = +0.704 [+0.389, +1.292]; r_rb = 0.97; p < 0.001), better than Hold-out temperature scaling (W/D/L 110/0/4; Δ = +1.119 [+0.805, +1.998]; r_rb = 0.99; p < 0.001), better than Logistic regression (CV) (W/D/L 104/0/5; Δ = +0.959 [+0.663, +1.230]; r_rb = 0.97; p < 0.001) and not significantly different from QUANT (W/D/L 43/0/69; Δ = -0.037 [-0.060, -0.006]; r_rb = -0.12; p = 0.253). For ECE under corruption LOOM is better than PreVal (W/D/L 80/0/34; Δ = +0.025 [+0.016, +0.037]; r_rb = 0.44; p < 0.001). Accuracy under corruption is almost identical for LOOM, PreVal and the uncalibrated ridge, which share the same logits up to the selected λ (median 0.694 vs 0.692); the improvement comes from lower confidence on the misclassified corrupted series, visible in the pooled reliability diagrams (Fig. 6). LOOM also flags corrupted inputs: the median AUROC for separating corrupted from clean test series by 1 − max p̂ is 0.765 for LOOM versus 0.563 for PreVal (significantly better).

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 0.738 | 0.151 | 0.694 | 0.765 | 1.17 | – | – | – |
| PreVal | 114 | 1.728 | 0.186 | 0.692 | 0.563 | 3.15 | 108/0/6 | 0.97 | < 0.001 |
| Hold-out temperature scaling | 114 | 2.221 | 0.197 | 0.663 | 0.558 | 3.41 | 110/0/4 | 0.99 | < 0.001 |
| Ridge + Platt (5-fold) | 110 | 1.125 | 0.145 | 0.696 | 0.610 | – | 86/0/24 | 0.78 | < 0.001 |
| Ridge + isotonic (5-fold) | 110 | 4.358 | 0.153 | 0.673 | 0.592 | – | 108/0/2 | 1.00 | < 0.001 |
| Logistic regression (CV) | 109 | 1.878 | 0.219 | 0.660 | 0.516 | – | 104/0/5 | 0.97 | < 0.001 |
| QUANT | 112 | 0.716 | 0.170 | 0.703 | 0.711 | – | 43/0/69 | -0.12 | 0.253 |
| Ridge softmax (T = 1) | 114 | 1.086 | 0.254 | 0.690 | 0.531 | 2.29 | 105/0/9 | 0.95 | < 0.001 |
| Ridge one-hot | 114 | 10.695 | 0.310 | 0.690 | 0.500 | 4.99 | 114/0/0 | 1.00 | < 0.001 |

: **Table 2** Performance under test-time corruption: per-dataset mean over 18 conditions (six families × three severities), then median over datasets. Statistics as in Table 1 (NLL)


![**Fig. 3** Distribution over datasets of the median test leverage ratio h(x)/h_ref (log scale) for clean test data and each corruption family at severities 1–3](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_leverage_ratio.png){width=95%}

![**Fig. 4** Median test NLL over datasets versus corruption severity for each family (severity 0 = clean)](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_robustness_nll.png){width=100%}

![**Fig. 5** Per-dataset NLL of LOOM versus PreVal on clean data (left) and averaged over the 18 corruption conditions (right); points below the diagonal favour LOOM. NLL values below 10⁻³ (datasets whose LOO predictions separate the classes perfectly, where unsmoothed PreVal becomes almost deterministic) are drawn at 10⁻³](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_scatter_loom_vs_preval.png){width=95%}

![**Fig. 6** Reliability diagrams pooled over all datasets and resamples for clean test data (left) and severity-3 corruptions (right). Bins with fewer than 200 pooled test predictions are not drawn; the pooled ECE in the legend uses all bins and differs from the per-dataset ECE medians of Tables 1–2](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_reliability.png){width=95%}

## 6.3 Ensembles, QUANT and the Arsenal (official split)

Table 3 compares LOOM with five-member ensembles and the two raw-series probabilistic classifiers on the official split. On clean data LOOM is worse than PreVal, 5-seed ensemble (W/D/L 31/0/83; Δ = -0.020 [-0.030, -0.012]; r_rb = -0.46; p < 0.001), better than QUANT (W/D/L 69/0/43; Δ = +0.028 [+0.004, +0.062]; r_rb = 0.31; p = 0.005) and better than Arsenal (W/D/L 92/0/19; Δ = +1.404 [+0.923, +1.981]; r_rb = 0.93; p < 0.001). Because the Arsenal was evaluated on clean and severity-3 data only, the corrupted-data comparison in this section uses the six severity-3 conditions for every method (a correction made after the results were first inspected, when an earlier version had averaged the other methods over all 18 conditions; Online Resource 2). Under these strongest corruptions, LOOM is better than PreVal, 5-seed ensemble (W/D/L 104/0/10; Δ = +0.654 [+0.512, +0.784]; r_rb = 0.94; p < 0.001), worse than QUANT (W/D/L 35/0/77; Δ = -0.050 [-0.093, -0.030]; r_rb = -0.23; p = 0.032) and better than Arsenal (W/D/L 108/0/3; Δ = +6.249 [+5.103, +6.941]; r_rb = 1.00; p < 0.001). The Arsenal's high NLL is largely caused by its vote-based probabilities, which often assign exactly zero probability to the true class: in the Brier score, which is bounded, the clean comparison is much closer (median 0.171 vs 0.163 for LOOM; LOOM is better than Arsenal (W/D/L 67/0/44; Δ = +0.007 [+0.002, +0.018]; r_rb = 0.43; p < 0.001)), while under severity-3 corruption LOOM is better than Arsenal (W/D/L 94/0/17; Δ = +0.129 [+0.107, +0.150]; r_rb = 0.83; p < 0.001) in Brier score. The comparison with QUANT depends on the corruption strength: averaged over all 18 conditions and five resamples (Table 2) the two are not significantly different, but at severity 3 LOOM is worse than QUANT (W/D/L 35/0/77; Δ = -0.050 [-0.093, -0.030]; r_rb = -0.23; p = 0.032) in NLL. The overall parity with QUANT hides a dependence on the corruption type (Fig. 4): at severity 3, QUANT has the lower median NLL for baseline wander, missing segments and spikes, and LOOM for Gaussian noise, time warping and circular shifts. We did not investigate the cause of this difference; it concerns the feature representation (raw-series quantiles versus random convolutions) rather than the readout, so LOOM-style moderation of a QUANT-like model is a possible extension. Averaging five LOOM models gives a median clean NLL of 0.329 and a corrupted NLL of 1.003.

| Method | n | NLL | BRIER | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 1.020 | 0.526 | 0.167 | 0.604 | 0.845 | 1.83 | – | – | – |
| LOOM, 5-seed ensemble | 114 | 1.003 | 0.519 | 0.165 | 0.614 | 0.847 | 1.49 | 38/0/76 | -0.48 | < 0.001 |
| PreVal | 114 | 2.260 | 0.673 | 0.257 | 0.613 | 0.531 | 3.70 | 105/0/9 | 0.97 | < 0.001 |
| PreVal, 5-seed ensemble | 114 | 2.040 | 0.636 | 0.240 | 0.616 | 0.584 | 2.97 | 104/0/10 | 0.94 | < 0.001 |
| QUANT | 112 | 0.917 | 0.487 | 0.179 | 0.630 | 0.764 | – | 35/0/77 | -0.23 | 0.032 |
| Arsenal | 111 | 6.933 | 0.669 | 0.283 | 0.611 | 0.675 | – | 108/0/3 | 1.00 | < 0.001 |

: **Table 3** Official split, severity-3 corruptions (per-dataset mean over the six severity-3 conditions, then median over datasets). Statistics as in Table 1 (NLL)


## 6.4 Ablation (RQ2)

Fig. 7 shows the median NLL difference of each variant to LOOM (variant − LOOM; positive values favour LOOM). **Leverage moderation.** Removing it while keeping target smoothing (PreVal-smooth) is significantly worse under corruption (median NLL 1.190 vs 0.738; Δ = +0.162, p < 0.001). On clean data it is not significantly different (Δ = -0.004, p = 0.084; PreVal-smooth is lower on 74 of 114 datasets), so the moderation costs at most a small amount of in-distribution NLL. **Exact LOO pairs.** Fitting the calibrator on in-sample pairs is significantly worse on clean data (median NLL 0.986), and an 80/20 hold-out split is significantly worse (median 0.441), because the ridge is then trained on fewer series; under corruption they are both significantly worse too. **Target smoothing.** Without it LOOM is not significantly different on clean NLL and significantly worse under corruption (Δ = +0.003, p = 0.003). **Moderation strength.** Learning a free strength b on the LOO pairs (LOOM-free-b) and the finite strength c = 1 are both significantly better on clean data (Δ = -0.005, p < 0.001 and Δ = -0.004, p < 0.001). Under corruption LOOM-free-b is significantly worse (median NLL 0.964; Δ = +0.025, p < 0.001), presumably because the held-out leverages vary little, so b is poorly identified (as observed during development), and c = 1 is not significantly different (Δ = -0.004, p = 1.000). For shift detection, however, LOOM's AUROC (0.765) exceeds that of c = 1 (0.729; LOOM higher on 111 of 114 datasets) and of LOOM-free-b (0.653), and the k-NN distance also detects shift worse (0.725). The pre-registered limit c → ∞ therefore trades a very small clean-data NLL loss for better shift detection. **Other choices.** Replacing leverage by a k-NN distance is not significantly different in NLL under corruption, and choosing λ by LOO squared error is not significantly different on clean data.

![**Fig. 7** Ablation: median paired NLL difference (variant − LOOM) with bootstrap 95 % confidence intervals on clean data (left) and under corruption (right); positive values mean the variant is worse](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_ablation.png){width=100%}

## 6.5 Generalisation (RQ4)

**Other random-kernel extractors.** With MultiRocket features (official split, clean and severity-3 conditions), LOOM is better than PreVal (W/D/L 49/0/2; Δ = +5.203 [+3.809, +7.982]; r_rb = 0.99; p < 0.001) under corruption and not significantly different from PreVal (W/D/L 21/0/30; Δ = -0.005 [-0.014, +0.004]; r_rb = -0.15; p = 0.349) on clean data. With Hydra features the corresponding comparisons are better than PreVal (W/D/L 51/0/0; Δ = +5.505 [+1.565, +9.010]; r_rb = 1.00; p < 0.001) and not significantly different from PreVal (W/D/L 23/0/28; Δ = -0.002 [-0.006, +0.007]; r_rb = -0.00; p = 0.985).

**Subsets.** On the 100 datasets never used during development LOOM is better than PreVal (W/D/L 94/0/6; Δ = +0.640 [+0.327, +1.021]; r_rb = 0.96; p < 0.001) under corruption and not significantly different from PreVal (W/D/L 44/0/56; Δ = -0.003 [-0.007, +0.001]; r_rb = -0.03; p = 1.000) on clean data. Excluding the 16 datasets with train/test duplicates (98 datasets), the comparisons are better than PreVal (W/D/L 92/0/6; Δ = +0.790 [+0.446, +1.355]; r_rb = 0.97; p < 0.001) and not significantly different from PreVal (W/D/L 42/0/56; Δ = -0.003 [-0.009, +0.001]; r_rb = -0.07; p = 1.000).

**Related datasets.** Several UCR datasets share a data source (e.g. the four UWaveGestureLibrary sets or the six phalanx-outline sets), so datasets are not fully independent units. Averaging related datasets into 70 source-level units, LOOM has the lower corrupted-data NLL than PreVal for 68 of 70 units (Holm-adjusted p < 0.001), while on clean data the comparison remains not significant (31/39 wins/losses, Holm-adjusted p = 1.000).

**A natural sensor shift.** To complement the synthetic corruptions we use the three accelerometer axes of UWaveGestureLibrary, which contain the same 4478 gesture recordings with identical labels. A model trained on one axis is tested on the same recordings measured on another axis (6 ordered axis pairs × 5 resamples; the same split seed keeps test recordings disjoint from training recordings). This shift is severe (mean accuracy drops from 0.810 to 0.323) but barely changes leverage: the median ratio h(x)/h_ref is only 1.13–1.29, because the other axes produce MiniRocket features inside the span of the training features. Correspondingly, LOOM's advantage is small. Its mean NLL on the shifted axis is 2.469 against 2.550 for PreVal (lower on 6 of 6 tasks), and its shift-detection AUROC is 0.727 against 0.712. The uncalibrated temperature-1 softmax, which is underconfident everywhere, attains the lowest NLL under this shift (1.894), but it is far worse in-domain (1.187 vs 0.579). Leverage moderation therefore protects against shifts that move inputs away from the training feature span, not against every distribution shift (Section 8).

## 6.6 Conformal prediction sets (RQ8)

Table 4 reports coverage and set size at α = 0.1. On clean data the closed-form jackknife+ sets of LOOM cover on average 0.897 of the test series with a mean set size of 1.55, against 0.927 and 2.49 for split conformal prediction. Under corruption coverage drops for every method, because exchangeability no longer holds. With the LAC score, LOOM's flattened probabilities lead to empty sets (25.0 % of corrupted test series), whereas the APS score turns flat probabilities into larger sets (1.5 % empty, coverage 0.905).

| Method | Coverage, clean (mean / 10th pct.) | Set size, clean | Empty sets, clean | Coverage, corrupted | Set size, corrupted | Empty sets, corrupted |
|---|---|---|---|---|---|---|
| LOOM, jackknife+, LAC | 0.897 / 0.861 | 1.55 | 4.6 % | 0.638 | 1.66 | 25.0 % |
| LOOM, jackknife, LAC | 0.899 / 0.861 | 1.51 | 4.4 % | 0.638 | 1.62 | 24.9 % |
| LOOM, jackknife+, APS | 0.908 / 0.878 | 5.23 | 7.3 % | 0.905 | 6.62 | 1.5 % |
| PreVal, jackknife+, LAC | 0.907 / 0.871 | 1.58 | 3.4 % | 0.695 | 1.55 | 9.3 % |
| PreVal, jackknife+, APS | 0.923 / 0.877 | 5.24 | 5.8 % | 0.816 | 5.03 | 10.1 % |
| Split conformal (TS), LAC | 0.927 / 0.871 | 2.49 | 2.2 % | 0.762 | 2.38 | 6.8 % |
| Split conformal (TS), APS | 0.948 / 0.892 | 5.86 | 3.5 % | 0.876 | 5.42 | 5.8 % |

: **Table 4** Conformal prediction sets at α = 0.1 (target coverage 0.9). Coverage and set size are averaged over datasets (corrupted: over datasets and the 18 conditions); 10th pct. is the 10th percentile of per-dataset coverage


## 6.7 Computational cost (RQ5)

| Method | Median fit time (s) | Mean fit time (s) | Total fit time, all datasets (h) | Median prediction time, 19 test sets (s) |
|---|---|---|---|---|
| MiniRocket transform of the training set (shared) | 0.168 | 1.36 | 0.04 | – |
| Ridge one-hot | 0.067 | 4.01 | 0.13 | 0.09 |
| PreVal | 0.144 | 4.34 | 0.14 | 0.10 |
| LOOM (ours) | 0.140 | 4.33 | 0.14 | 1.49 |
| Hold-out temperature scaling | 0.053 | 2.41 | 0.08 | 0.30 |
| Ridge + Platt (5-fold) | 0.256 | 15.14 | 0.46 | 0.36 |
| Ridge + isotonic (5-fold) | 0.255 | 15.08 | 0.46 | 0.35 |
| Logistic regression (CV) | 3.666 | 11.59 | 0.35 | 0.40 |
| LOOM, 5-seed ensemble | 2.247 | 24.93 | 0.79 | 26.26 |
| QUANT | 1.228 | 5.51 | 0.17 | 5.56 |
| Arsenal | 4.463 | 14.09 | 0.43 | 21.23 |

: **Table 5** Fit and prediction time per dataset (mean over resamples; single CPU core). Readouts on MiniRocket features include the ridge SVD they use but not the shared MiniRocket transform, which is listed separately; prediction times exclude the transform of the test sets. The ensembles include all five MiniRocket transforms, and QUANT and the Arsenal include their own feature extraction


LOOM's median fit time is 0.14 s, versus 0.14 s for PreVal, 0.26 s for five-fold Platt scaling and 3.67 s for cross-validated logistic regression, all on top of the shared MiniRocket transform (median 0.17 s). End to end, a single LOOM model therefore costs about 0.31 s (sum of medians), compared with 2.25 s for a five-seed ensemble, 1.23 s for QUANT and 4.46 s for the Arsenal. The extra prediction cost of LOOM over plain ridge is the projection onto V (Section 4.6).

![**Fig. 8** Fit time versus number of training series (log–log, single core)](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_efficiency.png){width=60%}

## 6.8 Sensitivity (RQ6)

Compared with the pre-registered limit c → ∞, the finite value c = 1 is significantly better on clean data (Δ = -0.004, p < 0.001) and not significantly different under corruption (Δ = -0.004, p = 1.000), but its shift-detection AUROC is lower (Section 6.4). For conformal sets, the clean coverage of LOOM's jackknife+ is 0.950 at α = 0.05 and 0.795 at α = 0.2. **Tuning of the logistic-regression baseline.** In a check on 12 randomly chosen datasets (official split; Online Resource 3), training LogReg-CV with 3,000 instead of 300 iterations and tolerance 10⁻⁵ instead of 10⁻³ changed its median NLL from 0.2707 to 0.2711, but improved it by 0.161 on ShapesAll, where the selected C moved to the upper end of the grid. The clean-data comparison with LogReg-CV should therefore be read as approximate for datasets with many classes. With 25 % and 50 % of the training data (official split, clean and severity-3 conditions; Online Resource 3) LOOM versus PreVal gives, on clean data, median NLL 0.522 vs 0.540, W/D/L 1/0/0, p = 1.000, n = 1 at 25 % and median NLL 0.499 vs 0.502, W/D/L 1/0/0, p = 1.000, n = 1 at 50 %; under severity-3 corruption, median NLL 0.642 vs 1.047, W/D/L 1/0/0, p = 1.000, n = 1 at 25 % and median NLL 0.625 vs 1.106, W/D/L 1/0/0, p = 1.000, n = 1 at 50 % (Table S4). The advantage under corruption is therefore not consistently retained with reduced training sets.

## 6.9 Error analysis (RQ7)

**Where the gain comes from.** Across datasets, the NLL improvement of LOOM over PreVal under corruption is positively correlated with how much the corruptions increase leverage (Spearman ρ = 0.61, p < 0.001, n = 114), which supports the intended mechanism. The share of datasets on which LOOM has the lower NLL grows with severity for every family except time warping. At severity 3 it is 95 % for noise, 89 % for spikes, 89 % for missing segments, 87 % for wander and 74 % for circular shifts, but only 53 % for time warping, whose median leverage ratio stays at 1.24. Smooth warps of up to ±20 % leave the pooled MiniRocket features largely inside the span of the training features: leverage then carries no signal and LOOM behaves like a temperature-scaled ridge. At severity 1 the win rates of the other families are only 48 %–64 %, i.e. mild corruptions are not reliably distinguished from in-distribution variation.

**Failure modes.** (i) *Over-moderation of shifted but still correct predictions.* The datasets on which LOOM loses most to PreVal under corruption (PigCVP, ECG5000, InsectEPGRegularTrain) keep comparatively high accuracy under corruption (e.g. 0.99 for InsectEPGRegularTrain) while their leverage grows (7.8×), so lowering confidence costs likelihood. Leverage measures distance from the training data, not the probability of error; across datasets the gain under corruption is negatively correlated with the accuracy retained under corruption (ρ = -0.24, p = 0.010). (ii) *Many classes with very few examples per class.* On clean data the gain over PreVal is negatively correlated with the number of classes (ρ = -0.37, p < 0.001). The four largest clean losses occur on PigArtPressure, Symbols, ShapesAll and PigAirwayPressure, which have only 2 to 10 training series per class on average. With so few series per class, Platt smoothing assigns each true class only (n_k + 1)/(n_k + 2) of the target mass, and the fitted model becomes underconfident. Smoothing is also what raises LOOM's in-distribution ECE (Section 6.1): without it, the median clean ECE is 0.041, at the level of PreVal (0.040), but the NLL under corruption is significantly worse (Section 6.4).

# 7 Discussion

**What LOOM adds to PreVal.** PreVal showed that exact LOO predictions turn ridge regression into a well-calibrated probabilistic classifier at almost no cost [@dempster2025preval], and our in-distribution results agree: the two are statistically indistinguishable in NLL on clean data. The difference appears when the test data move. A global scale factor keeps the confidence of a corrupted series at the level appropriate for clean data, so its NLL deteriorates quickly (median 1.728 averaged over the corruptions). LOOM uses the second exact LOO quantity, the held-out leverage, to put each prediction on the scale of its own predictive uncertainty. Its median NLL under corruption is 0.738, and its confidence separates corrupted from clean inputs better than any other readout and than QUANT (median shift AUROC 0.765 vs 0.711 for QUANT, the strongest competitor). Under corruption, QUANT, a tree ensemble on raw-series quantile features, reaches a similar NLL on average (0.716; not significantly different from LOOM) and a lower NLL under the strongest corruptions (Section 6.3), but at a higher clean NLL and about 4.0 times the median end-to-end fit time of LOOM including the MiniRocket transform (Tables 1 and 5). The ablation attributes the gain to the leverage moderation: PreVal with the same target smoothing remains far behind under corruption. It also shows that the exact LOO pairs matter, since fitting the same calibrator on in-sample pairs or on a hold-out split is clearly worse.

**Relation to distance-aware deep models.** The effect parallels the distance-aware single models proposed for deep networks [@liu2020sngp; @vanamersfoort2020; @kristiadi2020], which aim to become uncertain away from the training data. For a linear readout on a fixed random-kernel transform, distance awareness is available in closed form through the ridge leverage, and its calibration can be fitted without held-out data. A k-NN distance used in the same role gives similar NLL but worse shift detection than leverage. Leverage measures distance in the regularised geometry that the classifier itself uses, and it is computed from the same decomposition without an additional search over the training set.

**Which shifts are covered.** Leverage grows when a corruption pushes the random-kernel features outside the span of the training features, as noise, spikes, missing segments, wander and circular shifts do. Smooth time warps and the cross-axis transfer of Section 6.5 change the class-conditional distribution while keeping the features largely in-span. There the leverage ratio stays close to one and LOOM behaves like a globally calibrated ridge. Distance awareness is thus a complement to, not a substitute for, methods that address semantic or label shift.

**Practical guidance.** LOOM is attractive when (i) training sets are small, so that holding out data for calibration is costly; (ii) test conditions may drift, as in sensor and monitoring applications; and (iii) a single fast model is preferred over an ensemble. Its in-distribution calibration is not better than PreVal's: practitioners who are confident that test data match the training distribution, particularly with many classes and few examples per class, may prefer PreVal (or LOOM without smoothing) for the lowest in-distribution ECE. For uncertainty sets, jackknife+ with LAC gives nominal coverage with the smallest sets on clean data. Under shift, the APS score should be used, because LAC sets become empty when LOOM flattens its probabilities; APS sets are larger but retain coverage close to the nominal level (Table 4).

**Threats to validity.** Conclusions are drawn over datasets (Wilcoxon tests with Holm correction) and hold on the datasets not used during development and on the datasets without train/test duplicates (Section 6.5). They are specific to random-kernel feature extractors with a ridge readout and to the synthetic corruptions studied; Section 8 lists further limitations.

# 8 Limitations

The study has the following limitations.

*Scope of data and shift.* All experiments use univariate, equal-length UCR datasets. The corruptions are synthetic and test-time only: they model sensor noise, drift, dropouts, artefacts and misalignment, but not real deployment shifts such as new subjects, devices or recording sites, and not label shift. Multivariate series and real domain-shift benchmarks [@ragab2023; @ruiz2021] remain to be studied.

*Natural shift.* The only natural shift studied (UWaveGestureLibrary cross-axis transfer) barely changes leverage, and LOOM gains little there (Section 6.5). The benefit on real deployment shifts therefore depends on whether they move inputs off the training feature span.

*Approximations in the LOO quantities.* The identities in Section 4.2 are exact for fixed standardisation and fixed MiniRocket biases, but both are estimated on all n training series. In addition, λ and the temperature are selected on the same LOO pairs used to fit them. The LOO negative log-likelihood is therefore slightly optimistic, and the jackknife+ coverage guarantee [@barber2021] holds only approximately. We assessed coverage empirically instead.

*Behaviour of the moderation.* Leverage measures distance from the training data in the regularised feature geometry. It does not detect shifts that leave the features inside the span of the training data (for example, label noise or a change of class priors), and it lowers confidence even for shifted inputs that are still classified correctly (Section 6.9). Platt target smoothing, needed when LOO predictions are separable, makes the model underconfident on datasets with very few training series per class, which raised the clean ECE (Section 6.9).

*Compute-driven protocol choices.* All experiments ran on a 4-core CPU without a GPU in a cloud container that restarted several times. For this reason, and decided before results were inspected (Online Resource 2): ensembles, other extractors and the Arsenal were evaluated on the official split only; the Arsenal used aeon's MiniRocket-kernel configuration; Rocket features were not evaluated; logistic regression was skipped for the four datasets with more than 2,000 training series (Crop, ElectricDevices, FordA, FordB); conformal sets were evaluated on at most 1,000 test series per dataset; and the test sets of Crop and ElectricDevices were subsampled to 2,000 series. Deep-learning classifiers such as InceptionTime [@ismailfawaz2020inception], deep distance-aware models [@liu2020sngp; @vanamersfoort2020], a Laplace-approximated Bayesian logistic regression, and the full HIVE-COTE 2.0 ensemble [@middlehurst2021hc2] were not run.

*Baseline tuning.* The logistic-regression baseline used 300 iterations and a C grid ending at 1; a check on 12 datasets showed that longer training changes its median NLL negligibly but can improve it noticeably on datasets with many classes (Section 6.8).

*Development exposure.* The method was designed on eight datasets and the pipeline was tested on six more. Results restricted to the remaining datasets are reported (Section 6.5), but some optimism from design choices made on development data cannot be fully excluded.

*Memory.* LOOM stores the factors U and V, i.e. O((n + p)r) numbers. For training sets much larger than those in the UCR archive, randomised or incremental SVDs would be needed, and the exactness of the LOO identities would then be lost.

# 9 Conclusion

We proposed LOOM, a closed-form probabilistic readout for random convolutional kernel time series classifiers. It divides ridge logits by the square root of their leverage and fits one temperature and the ridge penalty on exact leave-one-out score–leverage pairs, so it needs no held-out data, refitting or ensembles. The same identities give exact jackknife+ prediction sets. On 114 UCR datasets with five resamples, LOOM matches the best existing closed-form readout (PreVal) in-distribution in NLL, at the price of a higher ECE caused by target smoothing. Under 18 synthetic test-time corruptions it reduces the median NLL from 1.728 to 0.738, better than the other closed-form and cross-validated readouts, and its confidence detects corrupted inputs much better (median AUROC 0.765 vs 0.563). LOOM is on par with the tree-based QUANT classifier on average but worse than it under the strongest corruptions. The fit costs essentially the same as the ridge readout itself. The protection is specific to shifts that move inputs away from the training feature span: under a natural cross-axis sensor shift that leaves leverage almost unchanged, its advantage over PreVal is small. Future work includes multivariate series, real deployment shifts, adaptive smoothing for many-class few-shot problems, and moderation signals that also respond to shifts that leave the feature span unchanged.


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
