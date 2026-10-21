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

We also report where LOOM does not help: its in-distribution advantage is small, smoothing of the calibration targets costs some in-distribution calibration error, and LAC prediction sets can become empty under shift.

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

![**Fig. 1** Overview of LOOM. One thin SVD of the standardised random-kernel features gives, for every candidate penalty λ, the exact leave-one-out logits and held-out leverages; λ and the temperature a are fitted on these pairs. At test time the logits are divided by the square root of the relative leverage, and exact leave-one-out models give jackknife+ prediction sets](figures/fig1_method.png){width=100%}

## 4.6 Complexity

The SVD costs O(np·min(n, p)), as for the standard RidgeClassifierCV readout. Evaluating ten penalties, the LOO pairs and the temperature fits adds O(10·nr·K) operations. Prediction costs O(p(r + K)) per test series, compared with O(pK) for plain ridge. Memory is dominated by U and V (O((n + p)r)). In contrast, k-fold calibration needs k additional ridge fits, logistic regression an iterative solver per regularisation value, and an M-member ensemble M transforms and fits.

# 5 Experimental setup

## 5.1 Data and protocol

We use the 114 univariate datasets of the UCR 2018 archive [@dau2019ucr] whose official files contain equal-length series without missing values (17 to 8,926 training series, lengths 15 to 2,844, 2 to 60 classes; Fig. 2). Each dataset is evaluated on the official split (resample 0) and on four stratified resamples that keep the official per-class training counts (numpy seeds 1–4), following the bake-off protocol [@bagnall2017bakeoff; @middlehurst2024bakeoff]. All seeds were fixed in advance. MiniRocket features (aeon 1.6.0 [@aeon2024], 9,996 features) are fitted on the training part only; the transform seed of resample r is 1000·r. A leakage audit of the official splits found 16 datasets in which some test series exactly duplicate a training series (Online Resource 2, Table S1). Because all methods share the same features, duplicates affect all methods. We nevertheless report a sensitivity analysis without these datasets.

![**Fig. 2** Distribution of training-set size, series length and number of classes over the UCR datasets used](figures/fig_datasets.png){width=100%}

## 5.2 Methods compared

All readouts share identical MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations. **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale with λ selected jointly [@dempster2025preval], reimplemented without target smoothing as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split: the ridge (λ by LOO squared error) is fitted on 80 % and the temperature on the remaining 20 %. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold out-of-fold calibration and a final model fitted on all data. **LogReg-CV** is L2 logistic regression on standardised features, with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it is skipped on the four datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models fitted on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two raw-series classifiers with native probabilities are also included: **QUANT** [@dempster2024quant], on all resamples, and the **Arsenal** [@middlehurst2021hc2], in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was computationally prohibitive (Online Resource 2).

Ablations: **PreVal-smooth** (PreVal with target smoothing, i.e. LOOM without leverage); **LOOM-nosmooth**; **LOOM-free-b** (moderation 1/√(1 + b h) with b fitted on the LOO pairs); **LOOM-finite-c1** (c = 1); **LOOM-kNN** (leverage replaced by the mean squared Euclidean distance to the five nearest training series in the standardised feature space); **LOOM-insample** (calibrator fitted on in-sample instead of LOO pairs); **LOOM-holdout** (80/20 split instead of LOO pairs); and **LOOM-lambdaSSE** (λ chosen by the LOO squared error, as in RidgeClassifierCV).

## 5.3 Test-time corruptions

Six corruption families, each at three severities, are applied to the test series only, with amplitudes relative to each series' standard deviation sd: Gaussian noise (σ ∈ {0.1, 0.3, 0.5}·sd); sinusoidal baseline wander (amplitude {0.25, 0.5, 1}·sd, frequency 0.5–2 cycles per series); a missing segment of {10, 20, 30} % of the length replaced by linear interpolation; impulse spikes of ±3 sd at {1, 3, 5} % of the time points; smooth monotone time warping with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}; and circular shifts of {5, 10, 20} % of the length. Random draws are seeded by the dataset, resample, family and severity, so every method sees identical inputs. Such perturbations model sensor noise, drift, dropouts, artefacts and misalignment, and similar operations are common augmentations [@um2017; @iwana2021].

## 5.4 Metrics and statistical analysis

The primary endpoint is the test negative log-likelihood (NLL, a strictly proper score). Secondary endpoints are the top-label expected calibration error (ECE, 15 equal-width bins), the Brier score, classwise ECE, accuracy, balanced accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, under corruption, the AUROC for separating corrupted from clean test inputs by 1 − max p̂ (shift detection [@hendrycks2017]). For conformal sets (α = 0.1) we report empirical coverage, mean set size and the fraction of empty sets. The robustness endpoint of a dataset is its mean over the 18 corruption conditions.

The unit of analysis is the dataset, with metrics averaged over the five resamples. LOOM is compared with every other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945], with Holm's correction within each table [@holm1979], as recommended for comparisons over multiple datasets [@demsar2006; @benavoli2016]. Paired differences are always reported as competitor − LOOM, so positive values favour LOOM for lower-is-better metrics (NLL, ECE, Brier) and negative values favour LOOM for accuracy and AUROC. We report the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ [@kerby2014] as effect size, computed on the same differences (for NLL, ECE and Brier, positive values favour LOOM) and a percentile bootstrap 95 % confidence interval (10,000 draws) of the median paired difference. Friedman tests and mean ranks summarise all methods jointly. A difference is called significant only if the Holm-adjusted p-value is below 0.05.

## 5.5 Implementation and development disclosure

Everything is implemented in Python with NumPy, SciPy, scikit-learn and aeon. Each job runs on one CPU core of a 4-core machine without a GPU, and timings are wall-clock times in that setting. Unit tests check every closed-form identity in Eqs. (2), (3) and (6) against explicit refits to machine precision. `run_all.py` regenerates all results, tables and figures, and every number in this paper is read from machine-generated result files. The method was designed on eight randomly selected development datasets, and six further datasets were used for pipeline testing. Platt target smoothing and the APS score were added after these tests and before the benchmark. Results are also reported for the {{S['n_unseen']}} datasets never used during development (Section 6.7). Use of generative AI is declared in the Statements and Declarations.
