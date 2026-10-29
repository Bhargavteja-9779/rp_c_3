---
title: "Knowing when to ask a human: reliable confidence for fast time series classifiers under sensor faults"
---

**Abstract** Monitoring systems, from heart monitors to industrial sensors, often let a time series classifier decide on its own and pass uncertain cases to a person. This only works if the classifier's confidence drops when a recording is damaged by noise, drift, spikes or dropouts. Fast random convolutional kernel classifiers such as MiniRocket are accurate and cheap, but their ridge readout gives scores rather than probabilities, and existing calibration methods rescale all scores in the same way, so faulty recordings still get confident answers. We propose LOOM, a closed-form readout that divides each ridge score by the square root of its leverage, which measures how far a recording lies from the training data, and fits one temperature and the ridge penalty on exact leave-one-out pairs of scores and leverages. LOOM needs no held-out data, refitting or ensemble, and also yields exact jackknife+ prediction sets. On 114 UCR datasets with 18 simulated sensor faults, LOOM matched the best existing readout on clean data and lowered the median negative log-likelihood under faults from 1.73 to 0.74. In a human-in-the-loop study on cardiac, wearable and industrial sensor data with a 20 % review budget, its review queue caught 45 % of the faulty recordings, against 25 % for the previous best readout and 38 % for the tree-based QUANT classifier. In likelihood, LOOM is on par with QUANT on average but worse than it under the strongest faults, and it cannot detect changes that keep the features within the training range.

**Keywords** Time series classification · Sensor faults · Uncertainty estimation · Probability calibration · Human-in-the-loop triage · Conformal prediction


# 1 Introduction

Consider a wearable heart monitor that classifies each short ECG recording as normal or abnormal, a factory system that labels the signal from a production sensor, or a smartwatch that recognises gestures from an accelerometer. All of these systems make many decisions per day, and nobody can check each one. The usual way to keep such a system safe is to let it decide on its own when it is sure and to pass the remaining cases to a person. For this to work, the classifier's confidence has to mean something. It must be high when the prediction is likely to be right and low when it is not, a requirement studied broadly under the name of uncertainty quantification [@abdar2021; @gawlikowski2023].

Real sensor data make this hard. Electrodes come loose, cables pick up interference, a patient moves, a sensor drifts or drops samples. A recording damaged in this way can look nothing like the training data, yet most classifiers still return a confident answer for it. A confident wrong answer is the worst outcome for a triage system, because it is exactly the case that never reaches the human reviewer.

In this paper we study the problem for the family of time series classifiers that is most attractive for such deployments: random convolutional kernel methods. ROCKET [@dempster2020rocket] and its successors MiniRocket [@dempster2021minirocket], MultiRocket [@tan2022multirocket] and Hydra [@dempster2023hydra] turn each series into thousands of features from random convolutions and then train a linear model on top, usually ridge regression. They are about as accurate as much more expensive ensembles and deep networks on the UCR benchmark archive [@dau2019ucr; @ismailfawaz2019review; @middlehurst2024bakeoff], and they train in seconds on an ordinary CPU, which makes them attractive for monitoring systems that run on modest hardware.

Their weak point is confidence. Ridge regression outputs scores, not probabilities, and library implementations report hard one-hot "probabilities". Standard calibration methods such as Platt scaling, isotonic regression or temperature scaling [@guo2017; @niculescu2005; @platt1999; @zadrozny2002] need held-out data or repeated refitting, and data are often scarce in exactly these applications. The Arsenal, an ensemble of ROCKET models inside HIVE-COTE 2.0 [@middlehurst2021hc2], gives probabilities at the price of many models. PreVal [@dempster2025preval] recently showed a much cheaper route: ridge regression gives its exact leave-one-out (LOO) predictions almost for free, and a single scale factor fitted on them turns the ridge scores into well-calibrated probabilities. However, PreVal, like the other methods, rescales all scores in the same way. A recording full of electrode noise gets the same treatment as a clean one, and globally calibrated models are known to become overconfident when the test data change [@ovadia2019].

Our starting point is a simple observation. Ridge regression already measures how far a new input is from the data it was trained on. The *leverage* h(x), a classical diagnostic of extrapolation in linear regression [@hoaglin1978], is also the predictive variance factor of the equivalent Bayesian linear model [@mackay1992]. For a clean recording it stays close to its typical training value. For a recording pushed into an unfamiliar region of feature space by a sensor fault it grows, often by orders of magnitude. If the ridge scores are divided by the square root of the leverage, confidence should drop automatically for such inputs. The open question is how strongly leverage should affect confidence, and how to set this without spending data on a separate calibration set. We show that the same algebra that gives the LOO predictions also gives the *held-out leverage* of each training example, h₋ᵢ = dᵢ/(1 − dᵢ). Every training example therefore provides an honest out-of-sample pair of a score and its uncertainty, and the calibration can be fitted on these n pairs without refitting anything.

We call the resulting readout LOOM (leave-one-out fitted, leverage-moderated). It computes p(y | x) = softmax(a·z(x)/√(h(x)/$h_{\mathrm{ref}}$)), where z are the ridge scores and a single temperature a and the ridge penalty λ are chosen by the LOO likelihood. The same identities give every LOO model's prediction for a new input, so LOOM also produces jackknife+ prediction sets [@barber2021] without the n refits that this method normally needs.

We evaluate LOOM in two ways. A large benchmark on 114 UCR datasets with six types of simulated sensor faults at three severities tests whether the probabilities stay reliable when the data are damaged. An application study then asks the practical question: with a fixed budget for human review, how many errors slip through, and how many faulty recordings end up in front of the reviewer? The application study covers heart monitoring, wearable motion sensing and industrial sensing.

Our contributions are:

1. LOOM, a closed-form probabilistic readout for random-kernel classifiers whose confidence drops for inputs that a sensor fault has moved away from the training data. It needs one singular value decomposition (SVD), no held-out data, no refitting and no ensemble (Section 4).
2. Exact jackknife+ prediction sets for ridge readouts, computed without refitting (Section 4.5).
3. A fault benchmark for univariate time series classification that measures calibration, selective prediction and fault detection, not only accuracy (Section 5), with a pre-registered comparison against eleven alternatives, ablations and a statistical analysis corrected for multiple comparisons (Section 6).
4. An application study of human-in-the-loop triage on twelve sensor datasets from three domains (Section 6.1).

LOOM does not solve every problem, and we report where it falls short. On clean data it is no better than PreVal, and the smoothing that keeps its fit stable slightly raises its calibration error. A tree-based classifier on raw series, QUANT [@dempster2024quant], copes better with the strongest faults. Faults that leave the features inside the range of the training data are not detected.

# 2 Related work

## 2.1 Random convolutional kernel classifiers

ROCKET [@dempster2020rocket] applies 10,000 random dilated kernels and summarises each output by its maximum and its proportion of positive values. MiniRocket [@dempster2021minirocket] makes the transform almost deterministic and about 75 times faster, MultiRocket [@tan2022multirocket] adds pooling operators and the first-difference series, and Hydra [@dempster2023hydra] lets groups of kernels compete, linking convolutional and dictionary methods. All of them use a linear readout: ridge regression with a penalty chosen by LOO cross-validation for small and medium training sets, and logistic regression for very large ones. The recent bake-off by @middlehurst2024bakeoff confirms that these models offer one of the best trade-offs between accuracy and cost and reports log-loss alongside accuracy, but it does not look at calibration when the data change. Probabilistic alternatives of similar accuracy include the Arsenal [@middlehurst2021hc2], QUANT, which uses interval quantiles and extremely randomised trees [@dempster2024quant], and deep networks such as InceptionTime [@ismailfawaz2020inception].

## 2.2 Calibration and prevalidation

Platt scaling fits a sigmoid to the classifier scores [@platt1999], isotonic regression fits a monotone map [@zadrozny2002], temperature scaling fits a single softmax temperature [@guo2017], and Dirichlet calibration extends these ideas to many classes [@kull2019]; @silvafilho2023 give a survey. All of them need held-out or cross-validated scores. Fitting a second model to cross-validated predictions is known as stacking or, in statistics, pre-validation [@tibshirani2002]. For least-squares models the LOO predictions have a closed form [@allen1974; @golub1979], which @cawley2004 used for model selection in least-squares support vector machines [@suykens1999]. To our knowledge, PreVal [@dempster2025preval] was the first method to use these closed-form LOO predictions to calibrate ridge regression. It scales the coefficients by a factor that minimises the LOO log-loss and was evaluated on 273 datasets, including 106 UCR datasets with MiniRocket features. LOOM builds directly on PreVal. It keeps the exact LOO idea and adds a per-input adjustment by leverage, fitted with the held-out leverages.

## 2.3 Uncertainty when the data change

@ovadia2019 showed that temperature scaling and similar calibrators degrade under dataset shift and that ensembles [@lakshminarayanan2017] hold up better. Distance-aware single networks such as SNGP [@liu2020sngp; @liu2023sngp] and DUQ [@vanamersfoort2020] become uncertain away from the training data, but they require changes to the architecture and the training. Last-layer Laplace approximations scale the logits by their posterior variance [@kristiadi2020], an idea that goes back to the moderated outputs of @mackay1992. Our moderation is the closed-form analogue of this idea for a linear readout on fixed random features. Corruption benchmarks such as ImageNet-C [@hendrycks2019] made robustness tests routine in computer vision. For time series, noise, scaling and warping are mostly used as training augmentations [@iwana2021; @um2017], and benchmarks such as AdaTime study adaptation to new subjects or devices when unlabelled target data are available [@ragab2023]. We are not aware of a benchmark that tests how the confidence of random-kernel classifiers behaves under sensor faults.

## 2.4 Conformal prediction

Conformal prediction turns any score into prediction sets that contain the true label with a chosen probability, provided training and test data are exchangeable [@angelopoulos2023; @vovk2022]. Split conformal prediction sets aside part of the training data [@lei2018]. Jackknife+ uses all n LOO models instead and guarantees coverage of at least 1 − 2α, with about 1 − α in practice [@barber2021], but it normally requires n refits. Adaptive prediction sets (APS) [@angelopoulos2021raps; @romano2020] and the least ambiguous set-valued classifier (LAC) [@sadinle2019] are common scores for classification. For time series, conformal methods have mostly addressed forecasting [@xu2023]. For ridge *regression*, conformal prediction can be computed in closed form [@burnaev2014; @nouretdinov2001]. We use the corresponding LOO identities to obtain jackknife+ sets for *classification*.

# 3 Problem setting

We are given n univariate training series xᵢ with labels yᵢ ∈ {1, …, K}, and a random-kernel transform φ: ℝᴸ → ℝᵖ fitted on the training series, with p ≈ 10⁴ features. We want a probabilistic classifier p̂(y | x) that is as accurate as the usual ridge readout, gives calibrated probabilities on clean test data, lowers its confidence when a fault has moved a recording away from the training data, and costs no more than the ridge fit itself. Before running the benchmark we registered three hypotheses and eight research questions (Online Resource 2). In short, they ask whether LOOM improves the probabilities on clean and on damaged data, which of its parts matter, whether the effect carries over to other feature extractors, what it costs, how sensitive it is to its settings, when it fails, and whether its prediction sets reach the intended coverage.

# 4 Method

## 4.1 Ridge readout from one thin SVD

The features are standardised with the training mean and standard deviation. $\Phi_c$ ∈ ℝⁿˣᵖ denotes the centred training matrix and Y ∈ {−1, +1}ⁿˣᴷ the one-vs-rest targets. The readout is ridge regression [@hoerl1970] with an unpenalised intercept and penalty λ, applied to the augmented feature vector x̃(x) = (1, ψ(x)), where ψ(x) are the centred standardised features. With X̃ stacking the training vectors,

$$A = \tilde X^\top \tilde X + \operatorname{diag}(0, \lambda I_p), \qquad B = A^{-1} \tilde X^\top Y, \qquad \hat Y = \tilde X B = H Y. \qquad (1)$$

Let $\Phi_c$ = U S Vᵀ be the thin SVD (rank r ≤ min(n − 1, p)) and $D_\lambda$ = diag(1/($s_j^2$ + λ)). The hat diagonal, the scores z(x) := Bᵀx̃(x) of a test series and its leverage h(x) := x̃(x)ᵀA⁻¹x̃(x) are then

$$d_i = \frac1n + \sum_{j=1}^{r} u_{ij}^2\frac{s_j^2}{s_j^2+\lambda}, \qquad h(x) = \frac1n + \big\lVert D_\lambda^{1/2} V^\top\psi\big\rVert^2 + \frac{\lVert\psi\rVert^2 - \lVert V^\top\psi\rVert^2}{\lambda}. \qquad (2)$$

The last term of h(x) measures the part of ψ(x) that lies outside the span of the training data. It dominates when p ≫ n, which is the usual situation with 10,000 random features and a few hundred training series. A single SVD serves all candidate penalties λ ∈ {10⁻³, …, 10³}·n (ten log-spaced values on the standardised scale, as in the reference MiniRocket implementation).

## 4.2 Exact leave-one-out scores and leverages

Let B₋ᵢ be the fit without example i, with the same λ and standardisation. The Sherman–Morrison identity gives (proofs in Online Resource 1)

$$z_{-i} := B_{-i}^\top \tilde x_i = Y_i - \frac{Y_i - \hat Y_i}{1 - d_i}, \qquad h_{-i} := \tilde x_i^\top A_{-i}^{-1} \tilde x_i = \frac{d_i}{1 - d_i}. \qquad (3)$$

The first identity is the well-known PRESS shortcut [@allen1974] used by RidgeClassifierCV and PreVal. The second is the one LOOM adds. h₋ᵢ is the leverage that example i would have as a test point for the model trained without it. The pair (z₋ᵢ, h₋ᵢ) is thus an exact out-of-sample observation of a score together with its uncertainty, and all n pairs cost O(nr) operations.

## 4.3 Leverage-scaled probabilities

Under the Bayesian linear model with prior β ~ N(0, σ²λ⁻¹I) and noise variance σ², the posterior predictive standard deviation of each score at x is σ√h(x). LOOM divides the scores by this quantity, relative to the median held-out leverage $h_{\mathrm{ref}}$ = median(h₋ᵢ), and applies one temperature:

$$\hat p(y = k \mid x) = \operatorname{softmax}_k\!\left( \frac{a\, z_k(x)}{\sqrt{h(x)/h_{\mathrm{ref}}}} \right). \qquad (4)$$

For a test series that resembles the training data, h(x) ≈ $h_{\mathrm{ref}}$, and Eq. (4) behaves like an ordinary temperature-scaled ridge. When a fault pushes a series away from the training data, h(x) grows and the probabilities move towards uniform, which is the behaviour a triage system needs. We use √h rather than the √(1 + h) of a regression prediction interval because only the parameter (epistemic) part of the uncertainty should be scaled; the inherent class overlap is handled by the softmax and its temperature. Eq. (4) is the limit c → ∞ of the family softmax(a z /√(1 + c h/$h_{\mathrm{ref}}$)) used in Bayesian moderation [@kristiadi2020; @mackay1992]. We chose this limit on eight development datasets before running the benchmark (Online Resource 2) and evaluate finite c as an ablation.

## 4.4 Fitting on exact LOO pairs

For each candidate λ, the temperature a minimises the negative log-likelihood of the n LOO pairs,

$$\mathcal L(a, \lambda) = -\frac1n \sum_{i=1}^n \sum_{k=1}^K t_{ik} \log \operatorname{softmax}_k\!\left(\frac{a\, z_{-i,k}}{\sqrt{h_{-i}/h_{\mathrm{ref}}}}\right), \qquad (5)$$

and the pair (λ, a) with the smallest value is kept, as PreVal does for its scale factor. On easy datasets the LOO predictions can separate the classes perfectly. The best temperature is then infinite and the model would claim certainty. To prevent this we smooth the targets $t_{ik}$ as @platt1999 suggested: the true class receives ($n_k$ + 1)/($n_k$ + 2), where $n_k$ is the number of training series of that class, and the rest is spread evenly over the other classes (for K = 2 this is exactly Platt's rule). The one-dimensional problem in log a is solved with L-BFGS from three starting values.

## 4.5 Closed-form jackknife+ prediction sets

Instead of a single label, a triage system can also return a set of plausible labels. Let s(x, y; f) be a non-conformity score of a probabilistic model f. We use LAC, s = 1 − p̂(y | x) [@sadinle2019], and deterministic APS, the total probability of all classes at least as probable as y [@romano2020]. Jackknife+ returns C(x) = {y : #{i : sᵢ < s(x, y; f₋ᵢ)} < (1 − α)(n + 1)}, where sᵢ = s(xᵢ, yᵢ; f₋ᵢ) [@barber2021]. It needs the prediction of every LOO model at every test point. For ridge regression these are available in closed form. With k(x, i) = x̃(x)ᵀA⁻¹x̃ᵢ = 1/n + (Vᵀψ)ᵀ$D_\lambda$ S uᵢ and residuals E = Y − Ŷ,

$$z_{-i}(x) = z(x) - \frac{k(x,i)\,E_i}{1-d_i}, \qquad h_{-i}(x) = h(x) + \frac{k(x,i)^2}{1-d_i}. \qquad (6)$$

After an O(pr) projection each test point costs O(nK), and no model is refitted. This parallels the closed-form conformal ridge regression of @nouretdinov2001, applied here to jackknife+ and to classification scores that depend on both the LOO scores and the LOO leverages. Because λ, a and the standardisation are fitted on all n examples, the coverage guarantee holds only approximately, so we check coverage empirically (Section 6.7).

![**Fig. 1** How LOOM works. One thin SVD of the standardised random-kernel features gives, for every candidate penalty λ, the exact leave-one-out scores and held-out leverages, on which λ and the temperature a are fitted. For a new recording the scores are divided by the square root of its relative leverage, and the exact leave-one-out models give jackknife+ prediction sets](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig1_method.png){width=100%}

## 4.6 Cost

The SVD costs O(np·min(n, p)), the same as the standard RidgeClassifierCV readout. Ten penalties, the LOO pairs and the temperature fits add O(10·nr·K) operations. Prediction costs O(p(r + K)) per series instead of O(pK) for plain ridge, and memory is dominated by U and V, O((n + p)r) numbers. For comparison, k-fold calibration needs k additional ridge fits, logistic regression an iterative solver for each regularisation value, and an M-member ensemble M transforms and fits.

# 5 Experimental setup

## 5.1 Data

We use the 114 univariate datasets of the UCR 2018 archive [@dau2019ucr] whose official files contain equal-length series without missing values. They have 17 to 8,926 training series, lengths from 15 to 2,844 and 2 to 60 classes (Fig. 2), and come from ECG and other medical signals, motion and gesture sensors, device and energy measurements, spectrographs, images converted to outlines, and simulations. Each dataset is evaluated on the official split and on four stratified resamples that keep the official number of training series per class, following the bake-off protocol [@bagnall2017bakeoff; @middlehurst2024bakeoff]. All seeds were fixed in advance. MiniRocket features (aeon 1.6.0 [@aeon2024], 9,996 features) are fitted on the training part only. A check of the official splits found 16 datasets in which some test series are exact copies of training series (Online Resource 2, Table S1). Since all methods share the same features, this affects every method equally, but we also report results without these datasets.

![**Fig. 2** Training-set size, series length and number of classes of the UCR datasets used](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_datasets.png){width=100%}

## 5.2 Methods compared

All readouts use the same MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations, and **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale factor with λ chosen jointly [@dempster2025preval]; we reimplemented it without target smoothing, as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold calibration. **LogReg-CV** is L2-regularised logistic regression with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it was skipped on the four datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models built on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two classifiers that work on the raw series and produce their own probabilities complete the comparison: **QUANT** [@dempster2024quant] on all resamples, and the **Arsenal** [@middlehurst2021hc2] in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was too slow for our hardware (Online Resource 2). Some methods cannot be applied to every dataset. The cross-validated calibrators need at least as many training series per class as folds, and the Arsenal returned undefined (NaN) probabilities on Fungi, which has a single training series per class. These methods are therefore compared on fewer datasets, as the n column of each table shows.

To see which parts of LOOM matter, we also ran eight variants: PreVal with target smoothing (that is, LOOM without leverage; **PreVal-smooth**), LOOM without smoothing (**LOOM-nosmooth**), a moderation strength b learned on the LOO pairs (**LOOM-free-b**), a finite strength c = 1 (**LOOM-finite-c1**), leverage replaced by the mean squared distance to the five nearest training series (**LOOM-kNN**), the calibration fitted on in-sample instead of LOO pairs (**LOOM-insample**) or on an 80/20 split (**LOOM-holdout**), and λ chosen by the LOO squared error as in RidgeClassifierCV (**LOOM-lambdaSSE**).

## 5.3 Simulated sensor faults

We simulate six common kinds of sensor faults, each at three severities, and apply them to the test series only. Amplitudes are relative to each series' standard deviation (sd). The six faults are broadband noise (Gaussian, σ ∈ {0.1, 0.3, 0.5}·sd), baseline wander or drift (a slow sinusoid of amplitude {0.25, 0.5, 1}·sd), signal dropout (a segment of {10, 20, 30} % of the series replaced by linear interpolation), impulse spikes such as electrode pops (±3 sd at {1, 3, 5} % of the time points), speed variation (a smooth time warp with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}), and misalignment (a circular shift of {5, 10, 20} % of the length). The random draws are seeded by dataset, resample, fault type and severity, so every method sees exactly the same damaged inputs.

## 5.4 Application study: triage with a human reviewer

The benchmark applies one fault type at a time to the whole test set. In practice, most recordings are fine and a few are damaged in different ways. The application study imitates this setting on twelve datasets from three domains: *heart monitoring* (ECG200, ECG5000, TwoLeadECG, NonInvasiveFetalECGThorax1), *wearable and motion sensing* (UWaveGestureLibraryX, CricketX, GunPoint, SemgHandGenderCh2) and *industrial and energy sensing* (Wafer, FreezerRegularTrain, PowerCons, RefrigerationDevices). Each test recording is independently faulty with probability 0.3. A faulty recording receives one fault that is plausible for its domain, at a random severity. For ECG, the faults are electrode noise, baseline wander from breathing or movement, electrode pops and lead-off segments. For wearables, they are noise, spikes, packet loss and changes in movement speed. For industrial sensors, they are noise, drift, spikes and dropouts. The classifier then works as a triage system: the share b of recordings with the lowest confidence (maximum predicted probability) goes to a human reviewer, and the rest is decided automatically. We report the error rate of the automatic decisions and the share of faulty recordings that end up in the review queue, at b = 20 % (with 10 % and 30 % in Online Resource 3). This study was designed after the benchmark results were known, but its protocol was written down before it was run (Online Resource 2, Section 10). It compares LOOM, PreVal, Holdout-TS, Ridge-Platt-CV5 and QUANT on the official split and four resamples.

## 5.5 Metrics and statistics

The primary measure is the test negative log-likelihood (NLL), a strictly proper score that rewards confident correct predictions and penalises confident mistakes. We also report the expected calibration error (ECE, 15 equal-width bins), the Brier score, accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, on damaged data, the AUROC for telling faulty from clean recordings by 1 − max p̂ [@hendrycks2017]. For prediction sets (α = 0.1) we report coverage, average set size and the share of empty sets. The robustness score of a dataset is its average over the 18 fault conditions.

Datasets are the unit of analysis, with each metric averaged over the five resamples. We compare LOOM with each other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945] with Holm's correction within each table [@holm1979], as recommended for comparisons across datasets [@benavoli2016; @demsar2006]. A difference is called significant only if the Holm-adjusted p-value is below 0.05. The tables also give the number of datasets on which LOOM is better, equal or worse (W/D/L), the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ as effect size [@kerby2014], and a bootstrap 95 % confidence interval of the median paired difference. In the text, "better on 108 of 114 datasets (p < 0.001)" refers to these tests, and all p-values of comparisons are Holm-adjusted. Spearman correlations in Section 6.9 are descriptive, and their p-values are not adjusted.

## 5.6 Implementation and disclosure

The code is written in Python with NumPy, SciPy, scikit-learn and aeon. Each job ran on one CPU core of a 4-core machine without a GPU, so the reported times are single-core wall-clock times. Unit tests compare every closed-form identity in Eqs. (2), (3) and (6) with explicit refits. A single script, `run_all.py`, regenerates all results, tables and figures, and every number in this paper is read from the machine-generated result files. The method was developed on eight randomly selected datasets, and six more were used to test the pipeline; target smoothing and the APS score were added at that stage, before the benchmark. Results restricted to the 100 datasets never used during development are reported in Section 6.6.

*Use of AI tools.* As the journal requires, we state here that an AI assistant (Claude, Anthropic) was used for the literature search, for writing and running the code, and for drafting the text. All numbers were produced by the released code and inserted into the manuscript automatically from the result files. The authors take full responsibility for the content (see also the Statements and Declarations).


# 6 Results

Unless stated otherwise, every result uses all 114 datasets with the official split plus four resamples, and per-dataset values are averages over the resamples. Complete per-dataset results are in Online Resource 3 and the code repository.

## 6.1 Triage of faulty recordings

We start with the practical question from the introduction. Fig. 3 shows what happens when each method decides on its own for the recordings it is most confident about and sends the rest to a human reviewer, on the twelve application datasets with 30 % faulty recordings.

![**Fig. 3** Human-in-the-loop triage on twelve sensor datasets (heart monitoring, wearables, industrial sensing) in which 30 % of the recordings carry a sensor fault. Left: error rate of the automatic decisions as more recordings are sent to review. Right: share of the faulty recordings that end up in the review queue; the dotted line is what random selection would achieve. Curves are averages over datasets of the mean over five resamples](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_case_study.png){width=100%}

With a review budget of 20 %, the clearest difference is in what reaches the reviewer. LOOM's review queue contained a median of 45 % of the faulty recordings, against 25 % for PreVal, 26 % for hold-out temperature scaling, 27 % for Platt scaling and 38 % for QUANT; random selection would catch 20 %. LOOM caught more faulty recordings than PreVal on every one of the twelve datasets (12 of 12 datasets, p = 0.002). Against QUANT the advantage was smaller and not significant (8 of 12 datasets, p = 0.052, not significant).

The errors that slip through automatically tell a more mixed story. LOOM's median error rate among the automatic decisions was 4.2 %, about half that of PreVal (9.5 %), hold-out temperature scaling (9.9 %) and Platt scaling (8.7 %). Across datasets, however, only the difference to hold-out temperature scaling was significant (11 of 12 datasets, p = 0.010; against PreVal: 9 of 12 datasets, p = 0.056, not significant). QUANT had the lowest automatic error rate (1.6 %). It catches fewer faulty recordings than LOOM, but it classifies damaged recordings more accurately in the first place, so fewer of the recordings it keeps are wrong. With only twelve datasets these tests have limited power, and the medians should be read alongside the p-values. Results for budgets of 10 % and 30 % follow the same pattern (Online Resource 3).

The advantage over PreVal in catching faults held in each domain. With a 20 % budget, LOOM's review queue caught 55 % of the faulty ECG recordings (PreVal 33 %), 33 % of the faulty wearable recordings (PreVal 19 %) and 47 % of the faulty industrial recordings (PreVal 25 %). On a fault-free stream the methods behaved much more alike: the median automatic error rate at the same budget was 1.2 % for LOOM and 1.0 % for PreVal. LOOM therefore does not change how a system behaves on good data. Its benefit appears when some recordings are damaged, because it routes many more of them to the reviewer.

## 6.2 Calibration on clean data

Table 1 summarises performance on the clean test sets. On clean data LOOM is as good as the best existing readouts, but not better. Its median NLL is 0.295. Compared with PreVal, LOOM is not significantly different (49 wins, 65 losses, p = 1.000), and the same holds against five-fold Platt scaling and cross-validated logistic regression (Table 1). Compared with hold-out temperature scaling, which gives up 20 % of the training data, LOOM is better on 85 of 114 datasets (p < 0.001), and it is far better than the raw ridge outputs. Accuracy is essentially the same as PreVal's (median 0.914 vs 0.914), since both use the same ridge scores.

There is one cost. LOOM's calibration error on clean data is higher than PreVal's (median ECE 0.075 vs 0.040; LOOM worse on 86 of 114 datasets; p < 0.001). The cause is the target smoothing of Section 4.4, which makes LOOM slightly underconfident when there are few training series per class (Section 6.9).

| Method | n | NLL | ECE | BRIER | ACC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 0.295 | 0.075 | 0.144 | 0.914 | 2.28 | – | – | – |
| PreVal | 114 | 0.360 | 0.040 | 0.134 | 0.914 | 2.13 | 49/0/65 | -0.06 | 1.000 |
| Hold-out temperature scaling | 114 | 0.524 | 0.052 | 0.177 | 0.896 | 3.56 | 85/0/29 | 0.61 | < 0.001 |
| Ridge + Platt (5-fold) | 110 | 0.294 | 0.080 | 0.148 | 0.912 | – | 53/0/57 | 0.03 | 1.000 |
| Ridge + isotonic (5-fold) | 110 | 0.670 | 0.044 | 0.157 | 0.904 | – | 96/0/14 | 0.92 | < 0.001 |
| Logistic regression (CV) | 109 | 0.314 | 0.049 | 0.151 | 0.901 | – | 53/0/56 | -0.01 | 1.000 |
| QUANT | 114 | 0.386 | 0.116 | 0.210 | 0.895 | 3.05 | 77/0/37 | 0.43 | < 0.001 |
| Ridge softmax (T = 1) | 114 | 0.588 | 0.249 | 0.284 | 0.914 | 4.57 | 110/0/4 | 0.97 | < 0.001 |
| Ridge one-hot | 114 | 2.959 | 0.086 | 0.171 | 0.914 | 5.40 | 103/0/11 | 0.96 | < 0.001 |

: **Table 1** Clean test data: median over datasets of the mean over five resamples. Mean rank, W/D/L (datasets on which LOOM is better/equal/worse), rank-biserial correlation r_rb and Holm-adjusted Wilcoxon p-value refer to NLL


## 6.3 Behaviour under sensor faults

**Leverage reacts to faults.** For clean test series the median ratio h(x)/$h_{\mathrm{ref}}$ is 1.01, so the held-out leverages computed during training are on the same scale as the leverages of new clean recordings. Under faults the ratio grows with severity: at the highest severity it reaches 30.3 for noise, 29.5 for spikes, 8.6 for a 30 % dropout, 6.2 for baseline wander and 3.5 for misalignment, but only 1.26 for speed variation (Fig. 4).

**Probabilities under faults.** Averaged over the 18 fault conditions (Table 2), LOOM's median NLL is 0.738, against 1.728 for PreVal. Compared with PreVal, LOOM is better on 108 of 114 datasets (p < 0.001), and it is also better than hold-out temperature scaling, Platt and isotonic calibration and logistic regression (Table 2). Fig. 5 shows how the gap opens as the faults become stronger, and Fig. 6 shows the per-dataset picture. Accuracy is unchanged, because LOOM and PreVal share the same ridge scores (median 0.695 vs 0.692). The gain comes entirely from lower confidence on the damaged recordings that are misclassified, which is visible in the reliability diagrams of Fig. 7. The same mechanism makes faulty recordings easier to spot. The median AUROC for telling faulty from clean recordings by 1 − max p̂ is 0.765 for LOOM and 0.563 for PreVal, where 0.5 means no separation at all.

| Method | n | NLL | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 0.738 | 0.151 | 0.695 | 0.765 | 1.78 | – | – | – |
| PreVal | 114 | 1.728 | 0.185 | 0.692 | 0.563 | 3.96 | 108/0/6 | 0.97 | < 0.001 |
| Hold-out temperature scaling | 114 | 2.221 | 0.197 | 0.676 | 0.558 | 4.28 | 110/0/4 | 0.98 | < 0.001 |
| Ridge + Platt (5-fold) | 110 | 1.125 | 0.144 | 0.697 | 0.610 | – | 86/0/24 | 0.78 | < 0.001 |
| Ridge + isotonic (5-fold) | 110 | 4.358 | 0.153 | 0.682 | 0.592 | – | 108/0/2 | 1.00 | < 0.001 |
| Logistic regression (CV) | 109 | 1.878 | 0.219 | 0.660 | 0.516 | – | 104/0/5 | 0.97 | < 0.001 |
| QUANT | 114 | 0.732 | 0.170 | 0.694 | 0.709 | 1.83 | 44/0/70 | -0.11 | 0.291 |
| Ridge softmax (T = 1) | 114 | 1.086 | 0.256 | 0.696 | 0.531 | 3.16 | 105/0/9 | 0.95 | < 0.001 |
| Ridge one-hot | 114 | 10.495 | 0.304 | 0.696 | 0.500 | 5.99 | 114/0/0 | 1.00 | < 0.001 |

: **Table 2** Damaged test data: per-dataset mean over 18 conditions (six fault types × three severities), then median over datasets. Statistics as in Table 1 (NLL)


![**Fig. 4** How far faulty recordings move away from the training data: distribution over datasets of the median leverage ratio h(x)/h_ref (log scale) for clean data and each fault type at severities 1–3](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_leverage_ratio.png){width=95%}

![**Fig. 5** Median test NLL over datasets against fault severity for each fault type (severity 0 = clean)](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_robustness_nll.png){width=100%}

![**Fig. 6** Per-dataset NLL of LOOM against PreVal on clean data (left) and averaged over the 18 fault conditions (right); points below the diagonal favour LOOM. NLL values below 10⁻³ (datasets whose LOO predictions separate the classes perfectly, where unsmoothed PreVal becomes almost deterministic) are drawn at 10⁻³](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_scatter_loom_vs_preval.png){width=95%}

![**Fig. 7** Reliability diagrams pooled over all datasets and resamples for clean data (left) and the strongest faults (right). Bins with fewer than 200 pooled predictions are not drawn; the pooled ECE in the legend uses all bins and differs from the per-dataset medians of Tables 1 and 2](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_reliability.png){width=95%}

## 6.4 Comparison with ensembles and raw-series classifiers

Table 3 compares LOOM with five-member ensembles and with the two classifiers that work on the raw series, on the official split. The Arsenal was only run on clean data and the strongest faults, so this comparison uses the six severity-3 conditions for every method. (An earlier version of the analysis averaged the other methods over all 18 conditions; we corrected this after seeing the results, and the correction works against LOOM. Online Resource 2 documents the change.)

On clean data a five-seed PreVal ensemble is slightly better than a single LOOM model (LOOM worse on 83 of 114 datasets; p < 0.001), while LOOM beats QUANT (LOOM better on 70 of 114 datasets; p = 0.004) and the Arsenal. Under the strongest faults LOOM beats the PreVal ensemble (LOOM better on 104 of 114 datasets; p < 0.001) and the Arsenal, but QUANT does better than LOOM (LOOM worse on 78 of 114 datasets; p = 0.042). Averaged over all 18 conditions and five resamples, LOOM and QUANT do not differ significantly (Table 2). Which one is better depends on the type of fault (Fig. 5): at the highest severity QUANT has the lower median NLL for baseline wander, dropouts and spikes, and LOOM for noise, speed variation and misalignment. We did not investigate the reason. It concerns the features (quantiles of the raw series versus random convolutions) rather than the readout, so applying a LOOM-style correction to a QUANT-like model is a natural next step.

The Arsenal's poor NLL deserves a comment. Its probabilities are weighted votes of its members, and they often give exactly zero probability to the true class, which the logarithm in the NLL punishes without limit. The bounded Brier score gives a fairer picture. On clean data the gap is small (median 0.173 vs 0.163 for LOOM), while under strong faults LOOM is still clearly ahead (LOOM better on 96 of 113 datasets; p < 0.001). Averaging five LOOM models helps only a little (median severity-3 NLL 1.003 vs 1.020 for a single model).

| Method | n | NLL | BRIER | ECE | ACC | Shift AUROC | Mean rank | W/D/L | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| LOOM (ours) | 114 | 1.020 | 0.526 | 0.167 | 0.604 | 0.845 | 2.52 | – | – | – |
| LOOM, 5-seed ensemble | 114 | 1.003 | 0.519 | 0.165 | 0.614 | 0.847 | 2.15 | 38/0/76 | -0.48 | < 0.001 |
| PreVal | 114 | 2.260 | 0.673 | 0.257 | 0.613 | 0.531 | 4.57 | 105/0/9 | 0.97 | < 0.001 |
| PreVal, 5-seed ensemble | 114 | 2.040 | 0.636 | 0.240 | 0.616 | 0.584 | 3.81 | 104/0/10 | 0.94 | < 0.001 |
| QUANT | 114 | 0.931 | 0.489 | 0.179 | 0.621 | 0.760 | 1.96 | 36/0/78 | -0.22 | 0.042 |
| Arsenal | 113 | 7.024 | 0.672 | 0.283 | 0.600 | 0.676 | – | 110/0/3 | 1.00 | < 0.001 |

: **Table 3** Official split under the strongest faults (per-dataset mean over the six severity-3 conditions, then median over datasets). Statistics as in Table 1 (NLL)


## 6.5 What makes LOOM work

Fig. 8 compares LOOM with eight variants. Three findings stand out.

*The leverage scaling is what helps under faults.* PreVal with the same target smoothing, which is LOOM without leverage, is much worse on damaged data (median NLL 1.172 vs 0.738; LOOM better on 94 of 114 datasets; p < 0.001). On clean data, LOOM compared with it is worse on 75 of 114 datasets (p = 0.038), so the scaling costs at most a small amount of clean-data NLL.

*The exact LOO pairs matter.* Fitting the same calibration on in-sample pairs instead of LOO pairs gives a median clean NLL of 0.978, because in-sample scores are overconfident. Using an 80/20 split gives 0.441, because the ridge is then trained on less data. Both variants are also worse under faults.

*The moderation strength is a trade-off.* A finite strength (c = 1) or a strength learned from the LOO pairs gives a slightly lower clean NLL than LOOM (LOOM worse on 89 of 114 datasets (p < 0.001) against c = 1). But both detect faults less well: the median detection AUROC is 0.729 for c = 1 and 0.653 for the learned strength, against 0.765 for LOOM. The learned strength is also worse in NLL under faults (median 0.964). This is probably because the held-out leverages of the training data vary too little to pin the strength down, as we had already seen during development. The full scaling chosen before the benchmark thus gives up a little clean-data NLL in exchange for better fault detection.

Removing the target smoothing gives the lowest clean ECE but is worse under faults (LOOM better on 71 of 114 datasets; p = 0.003). Replacing leverage by a nearest-neighbour distance gives a similar NLL under faults but detects them worse (median AUROC 0.725). Choosing λ by squared error instead of likelihood changes little.

![**Fig. 8** Ablation: median paired NLL difference (variant − LOOM) with bootstrap 95 % confidence intervals on clean data (left) and under faults (right); positive values mean that the variant is worse](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_ablation.png){width=100%}

## 6.6 Does it carry over?

**Other feature extractors.** With MultiRocket features (official split, clean data and the strongest faults), LOOM compared with PreVal is better on 105 of 110 datasets (p < 0.001) under faults and worse on 72 of 110 datasets (p = 0.008) on clean data. With Hydra features it is better on 106 of 110 datasets (p < 0.001) under faults and not significantly different (43 wins, 67 losses, p = 0.147) on clean data.

**Subsets of datasets.** The results do not depend on the datasets used during development: on the 100 datasets that were never used for development, LOOM compared with PreVal is better on 94 of 100 datasets (p < 0.001) under faults. They also hold without the 16 datasets with train/test duplicates (LOOM better on 92 of 98 datasets; p < 0.001). Some UCR datasets come from the same source, for example the four UWaveGestureLibrary sets. After averaging related datasets into 70 source-level units, LOOM still has the lower NLL under faults for 68 of 70 units (p < 0.001).

**Training-set size.** With only 25 % or 50 % of the training data (official split), LOOM's median NLL under the strongest faults is 1.058 and 1.049, against 2.155 and 2.543 for PreVal (111 datasets; Table S4). The advantage under faults is therefore kept with smaller training sets.

**A real change of sensor.** Simulated faults are only a model of reality. As a check with real data, we used the three accelerometer axes of UWaveGestureLibrary, which record the same 4478 gestures with identical labels. A model trained on one axis was tested on the same gestures recorded by another axis (6 axis pairs × 5 resamples, with test recordings disjoint from training recordings). This is a severe change: accuracy drops from 0.810 to 0.323. Yet leverage barely moves (median ratio 1.13–1.29), because the other axes produce features inside the range of the training features. LOOM's advantage is correspondingly small: mean NLL 2.469 against 2.550 for PreVal (lower on 6 of 6 axis pairs), and detection AUROC 0.727 against 0.712. Leverage scaling protects against changes that push recordings away from the training data, not against every kind of change (Section 8).

## 6.7 Prediction sets

Table 4 reports coverage and set size at α = 0.1, that is, for a target coverage of 90 %. On clean data LOOM's closed-form jackknife+ sets contain the true label for 89.8 % of the test series on average, with 1.55 labels per set, while split conformal prediction reaches 92.8 % with 2.49 labels. Under faults, coverage drops for every method, because damaged recordings are no longer exchangeable with the training data. Here the choice of score matters. With the LAC score, LOOM's flattened probabilities produce empty sets for 25.0 % of the damaged recordings. With the APS score, flat probabilities instead produce larger sets, which keeps coverage at 90.5 % with only 1.5 % empty sets. For a triage system, a large set is a useful signal in itself.

| Method | Coverage, clean (mean / 10th pct.) | Set size, clean | Empty sets, clean | Coverage, corrupted | Set size, corrupted | Empty sets, corrupted |
|---|---|---|---|---|---|---|
| LOOM, jackknife+, LAC | 0.898 / 0.861 | 1.55 | 4.6 % | 0.638 | 1.66 | 25.0 % |
| LOOM, jackknife, LAC | 0.899 / 0.861 | 1.51 | 4.4 % | 0.639 | 1.62 | 24.8 % |
| LOOM, jackknife+, APS | 0.908 / 0.878 | 5.22 | 7.3 % | 0.905 | 6.62 | 1.5 % |
| PreVal, jackknife+, LAC | 0.908 / 0.871 | 1.58 | 3.3 % | 0.696 | 1.55 | 9.3 % |
| PreVal, jackknife+, APS | 0.923 / 0.877 | 5.24 | 5.8 % | 0.817 | 5.03 | 10.1 % |
| Split conformal (TS), LAC | 0.928 / 0.871 | 2.49 | 2.2 % | 0.763 | 2.38 | 6.8 % |
| Split conformal (TS), APS | 0.948 / 0.892 | 5.86 | 3.5 % | 0.876 | 5.41 | 5.8 % |

: **Table 4** Prediction sets at α = 0.1 (target coverage 0.9). Coverage and set size are averaged over datasets (damaged data: over datasets and the 18 conditions); 10th pct. is the 10th percentile of per-dataset coverage


## 6.8 Cost and sensitivity

| Method | Median fit time (s) | Mean fit time (s) | Total fit time, all datasets (h) | Median prediction time, 19 test sets (s) |
|---|---|---|---|---|
| MiniRocket transform of the training set (shared) | 0.168 | 1.14 | 0.04 | – |
| Ridge one-hot | 0.067 | 3.33 | 0.11 | 0.09 |
| PreVal | 0.144 | 3.63 | 0.11 | 0.10 |
| LOOM (ours) | 0.140 | 3.61 | 0.11 | 1.49 |
| Hold-out temperature scaling | 0.053 | 2.03 | 0.06 | 0.30 |
| Ridge + Platt (5-fold) | 0.256 | 12.78 | 0.39 | 0.36 |
| Ridge + isotonic (5-fold) | 0.255 | 12.87 | 0.39 | 0.35 |
| Logistic regression (CV) | 3.666 | 11.59 | 0.35 | 0.40 |
| LOOM, 5-seed ensemble | 2.247 | 24.93 | 0.79 | 26.26 |
| QUANT | 1.298 | 5.72 | 0.18 | 5.56 |
| Arsenal | 4.475 | 22.98 | 0.72 | 20.45 |

: **Table 5** Fit and prediction time per dataset (mean over resamples, single CPU core). Readouts on MiniRocket features include the ridge SVD they use but not the shared MiniRocket transform, which is listed separately; prediction times exclude the transform of the test sets. The ensembles include all five MiniRocket transforms, and QUANT and the Arsenal include their own feature extraction


LOOM costs the same to fit as PreVal (median 0.14 s vs 0.14 s on top of the shared MiniRocket transform, which takes a median of 0.17 s; Table 5 and Fig. 9). From raw data to a fitted model, a single LOOM model takes about 0.31 s, compared with 1.30 s for QUANT, 2.25 s for a five-seed ensemble and 4.47 s for the Arsenal. Prediction is slower than for plain ridge because each new series is projected onto the singular vectors (Section 4.6), but it remains a fraction of a second per dataset.

The moderation strength was discussed in Section 6.5. For the prediction sets, the clean coverage of LOOM's jackknife+ is 95.1 % at α = 0.05 and 79.6 % at α = 0.2. We also checked whether the logistic-regression baseline was trained long enough. On 12 random datasets, ten times more iterations and a tighter tolerance changed its median NLL from 0.2707 to 0.2711, but improved it by 0.161 on ShapesAll, a dataset with many classes. The clean-data comparison with logistic regression should therefore be read as approximate for such datasets (Online Resource 3).

![**Fig. 9** Fit time against the number of training series (log–log, single core)](/home/user/rp_c_3/Target_Journal_Submission/code/figures/fig_efficiency.png){width=70%}

## 6.9 When LOOM fails

LOOM helps most where faults increase leverage the most. Across datasets, its gain over PreVal under faults rises with the increase in leverage that the faults cause (Spearman ρ = 0.61, p < 0.001, n = 114), which is the mechanism we designed it for. The share of datasets on which LOOM beats PreVal grows with severity for every fault type except speed variation. At the highest severity it is 95 % for noise, 89 % for spikes, 89 % for dropouts, 87 % for baseline wander and 73 % for misalignment, but only 53 % for speed variation, whose median leverage ratio stays at 1.26. A smooth change of speed of up to ±20 % leaves the pooled MiniRocket features largely where they were. Leverage then carries no signal, and LOOM behaves like an ordinary temperature-scaled ridge. Mild faults are also hard to tell apart from normal variation: at severity 1 the win rates of the other fault types are only 47 %–64 %.

We found two ways in which LOOM fails. The first is *over-caution for faulty recordings that are still classified correctly*. The datasets on which LOOM loses most to PreVal under faults (PigCVP, ECG5000 and InsectEPGRegularTrain) keep a high accuracy under faults (for example 0.99 for InsectEPGRegularTrain) while their leverage still grows (9.3× for InsectEPGRegularTrain). Lowering the confidence then costs likelihood. Leverage measures how unusual a recording is, not how likely the prediction is to be wrong, and across datasets the gain under faults is smaller where accuracy under faults stays high (ρ = -0.24, p = 0.010). In a triage system this failure is relatively harmless: it sends a few correct cases to the reviewer.

The second is *many classes with very few training series each*. On clean data, the gain over PreVal shrinks as the number of classes grows (ρ = -0.39, p < 0.001). The four largest clean losses occur on PigArtPressure, Symbols, ShapesAll and PigAirwayPressure, which have only 2 to 10 training series per class on average. With so few examples, the smoothed targets give the true class only (n_k + 1)/(n_k + 2) of the probability mass, and the fitted model ends up underconfident. Smoothing is also what raises LOOM's clean ECE (Section 6.2). Without it, the median clean ECE is 0.041, the same level as PreVal's 0.040, but the NLL under faults is significantly worse (Section 6.5).

# 7 Discussion

**What leverage adds.** PreVal showed that exact LOO predictions turn ridge regression into a well-calibrated classifier at almost no cost [@dempster2025preval], and our clean-data results agree: the two cannot be told apart in NLL. They part ways when the recordings are damaged. A single scale factor keeps the confidence of a faulty recording at the level that suits clean data, and its NLL grows quickly (median 1.728 over the fault conditions). LOOM uses a second quantity that the LOO algebra provides for free, the held-out leverage, to put each prediction on the scale of its own uncertainty. Its median NLL under faults is 0.738, and its confidence separates faulty from clean recordings better than any of the competing methods, including QUANT (median AUROC 0.765 vs 0.709). The ablation confirms that this comes from the leverage scaling and that the exact LOO pairs are needed to calibrate it.

**What this means in practice.** The application study translates these numbers into the terms that matter to an operator. With one recording in five sent to a human, LOOM's review queue contained a median of 45 % of the damaged recordings, against 25 % for PreVal. The error rate of its automatic decisions was also lower in the median, although not significantly so across twelve datasets, and QUANT, which is more robust to damage in the first place, made the fewest automatic errors. LOOM achieves this without extra models, extra data or a separate fault detector, with the same training cost as the ridge readout and on a standard CPU. This matters in settings where data are scarce and computing power is limited, which are common in wearable, clinical and industrial monitoring. LOOM is not a full replacement for a dedicated fault detector, however. It flags recordings that look unusual to the classifier, and it cannot tell a broken sensor from a genuinely new type of signal.

**How LOOM compares with other ways of being uncertain.** The behaviour resembles that of the distance-aware deep networks proposed to make neural networks uncertain away from their training data [@kristiadi2020; @liu2020sngp; @vanamersfoort2020]. For a linear readout on fixed random features, such distance awareness is available in closed form through the leverage, and it can be calibrated without held-out data. A nearest-neighbour distance used in the same role gave a similar NLL but detected faults worse. Leverage measures distance in the same regularised geometry that the classifier uses, and it comes from the decomposition that is computed anyway. QUANT, a tree ensemble on raw-series quantiles, reached a similar NLL under faults on average and a lower one under the strongest faults, at a higher clean NLL and about 4.2 times LOOM's end-to-end training time. The two approaches have different strengths, and combining them is a promising direction.

**Which faults are covered.** Leverage grows when a fault pushes the features outside the range of the training data, as noise, spikes, dropouts, drift and misalignment do. A change in speed, or the switch to another sensor axis in Section 6.6, changes the signal while keeping the features in range. Leverage then stays close to its training value, and LOOM behaves like a globally calibrated model. Leverage scaling therefore complements, but does not replace, methods aimed at other kinds of change, such as new classes or changing class frequencies.

**Recommendations for practitioners.** We would use LOOM when (i) training sets are small, so that setting data aside for calibration hurts; (ii) recordings may be damaged or conditions may drift, as in most sensor applications; and (iii) a single fast model is preferred over an ensemble. If the test data are known to match the training data, especially with many classes and few examples per class, PreVal or LOOM without smoothing gives a lower clean calibration error. Where prediction sets are needed, jackknife+ with the LAC score gives the intended coverage with the smallest sets on clean data. If damaged inputs are expected, the APS score is the better choice, because LAC sets can become empty when LOOM flattens its probabilities, whereas APS sets grow instead, which is itself a warning sign (Table 4).

**Validity of the conclusions.** All conclusions are drawn across datasets with Wilcoxon tests and Holm's correction, and they hold on the datasets not used during development, without the datasets with train/test duplicates, and when related datasets are grouped (Section 6.6). They apply to random-kernel features with a ridge readout and to the faults we simulated. Section 8 lists further limitations.

# 8 Limitations

Several limitations should be kept in mind.

*Data and faults.* All experiments use univariate, equal-length UCR datasets. The faults are simulated and applied at test time only. They imitate common sensor problems, but real faults can be more varied, can affect training data too, and can occur together. The only real change of sensor we studied (another accelerometer axis) barely moves the leverage, and LOOM gains little there. How much LOOM helps in a given deployment therefore depends on whether its faults push recordings away from the training data. Multivariate recordings and real deployment data collected over time remain to be studied [@ragab2023; @ruiz2021].

*The application study.* The triage study uses twelve datasets and simulated faults, and it was designed after the main benchmark results were known (although its protocol was fixed before it was run). With twelve datasets the statistical tests have little power. A study with real faulty recordings labelled by domain experts would be the natural next step.

*Approximations.* The leave-one-out identities are exact for a fixed standardisation and fixed MiniRocket biases, but both are estimated on all n training series. In addition, λ and the temperature are chosen on the same LOO pairs used to fit them. The LOO likelihood is therefore slightly optimistic, and the jackknife+ coverage guarantee [@barber2021] holds only approximately, which is why we checked coverage empirically.

*What leverage can and cannot see.* Leverage measures how unusual a recording is in the classifier's feature space. It does not respond to changes that leave the features in the range of the training data, such as label noise or a change of class frequencies, and it lowers the confidence of unusual recordings even when they are classified correctly. The target smoothing needed to keep the fit stable makes LOOM underconfident on datasets with very few training series per class, which raises its clean calibration error.

*Compute-driven choices.* All experiments ran on a 4-core CPU without a GPU. For this reason, and decided before the results were inspected (Online Resource 2), the ensembles, the other feature extractors and the Arsenal were evaluated on the official split only; the Arsenal used aeon's MiniRocket-kernel configuration; logistic regression was skipped for the four datasets with more than 2,000 training series; prediction sets were evaluated on at most 1,000 test series per dataset; and the test sets of Crop and ElectricDevices were subsampled to 2,000 series. Deep-learning classifiers such as InceptionTime [@ismailfawaz2020inception], distance-aware deep networks [@liu2020sngp; @vanamersfoort2020], a Laplace-approximated Bayesian logistic regression and the full HIVE-COTE 2.0 ensemble [@middlehurst2021hc2] were not run.

*Baseline tuning.* Logistic regression used 300 iterations and a regularisation grid ending at C = 1. Longer training changed its median NLL very little but helped noticeably on a dataset with many classes (Section 6.8).

*Development exposure.* The method was designed on eight datasets, and the pipeline was tested on six more. Results without these datasets are reported (Section 6.6), but some optimism from design choices cannot be ruled out completely.

*Memory.* LOOM stores the factors U and V, that is O((n + p)r) numbers. For training sets much larger than those in the UCR archive, a randomised or incremental SVD would be needed, and the LOO identities would then no longer be exact.

# 9 Conclusion

A classifier that works alongside a human reviewer is only useful if it knows when it might be wrong. We proposed LOOM, a simple change to the readout of fast random-kernel time series classifiers that makes their confidence drop when a recording looks unlike the training data. LOOM divides the ridge scores by the square root of their leverage and fits a single temperature and the ridge penalty on exact leave-one-out pairs of scores and leverages. It needs no held-out data, no refitting and no ensemble, and the same identities give exact jackknife+ prediction sets.

On 114 UCR datasets, LOOM matched the best existing readout on clean data, at the price of a slightly higher calibration error caused by target smoothing. Under 18 simulated sensor faults it lowered the median NLL from 1.728 to 0.738 and separated faulty from clean recordings much better (median AUROC 0.765 vs 0.563). In a triage study on heart, wearable and industrial sensor data, its review queue caught nearly twice as many of the faulty recordings as PreVal with the same review budget (45 % vs 25 %), and somewhat more than QUANT. LOOM is on par with QUANT on average but worse than it under the strongest faults, and it does not detect changes that leave the features within the range of the training data, such as the switch to another accelerometer axis.

Future work includes multivariate recordings, real deployment data collected over time, adaptive smoothing for problems with many classes and few examples, and combining leverage with signals that respond to changes it cannot see.


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
