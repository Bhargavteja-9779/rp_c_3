# 1 Introduction

Consider a wearable heart monitor that classifies each short ECG recording as normal or abnormal, a factory system that labels the signal from a production sensor, or a smartwatch that recognises gestures from an accelerometer. All of these systems make many decisions per day, and nobody can check each one. The usual way to keep such a system safe is to let it decide on its own when it is sure and to pass the remaining cases to a person. For this to work, the classifier's confidence has to mean something. It must be high when the prediction is likely to be right and low when it is not.

Real sensor data make this hard. Electrodes come loose, cables pick up interference, a patient moves, a sensor drifts or drops samples. A recording damaged in this way can look nothing like the training data, yet most classifiers still return a confident answer for it. A confident wrong answer is the worst outcome for a triage system, because it is exactly the case that never reaches the human reviewer.

In this paper we study the problem for the family of time series classifiers that is most attractive for such deployments: random convolutional kernel methods. ROCKET [@dempster2020rocket] and its successors MiniRocket [@dempster2021minirocket], MultiRocket [@tan2022multirocket] and Hydra [@dempster2023hydra] turn each series into thousands of features from random convolutions and then train a linear model on top, usually ridge regression. They are about as accurate as much more expensive ensembles and deep networks on the UCR benchmark archive [@dau2019ucr; @middlehurst2024bakeoff; @ismailfawaz2019review], and they train in seconds on an ordinary CPU, which makes them attractive for monitoring systems that run on modest hardware.

Their weak point is confidence. Ridge regression outputs scores, not probabilities, and library implementations report hard one-hot "probabilities". Standard calibration methods such as Platt scaling, isotonic regression or temperature scaling [@platt1999; @zadrozny2002; @niculescu2005; @guo2017] need held-out data or repeated refitting, and data are often scarce in exactly these applications. The Arsenal, an ensemble of ROCKET models inside HIVE-COTE 2.0 [@middlehurst2021hc2], gives probabilities at the price of many models. PreVal [@dempster2025preval] recently showed a much cheaper route: ridge regression gives its exact leave-one-out (LOO) predictions almost for free, and a single scale factor fitted on them turns the ridge scores into well-calibrated probabilities. However, PreVal, like the other methods, rescales all scores in the same way. A recording full of electrode noise gets the same treatment as a clean one, and globally calibrated models are known to become overconfident when the test data change [@ovadia2019].

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

@ovadia2019 showed that temperature scaling and similar calibrators degrade under dataset shift and that ensembles [@lakshminarayanan2017] hold up better. Distance-aware single networks such as SNGP [@liu2020sngp; @liu2023sngp] and DUQ [@vanamersfoort2020] become uncertain away from the training data, but they require changes to the architecture and the training. Last-layer Laplace approximations scale the logits by their posterior variance [@kristiadi2020], an idea that goes back to the moderated outputs of @mackay1992. Our moderation is the closed-form analogue of this idea for a linear readout on fixed random features. Corruption benchmarks such as ImageNet-C [@hendrycks2019] made robustness tests routine in computer vision. For time series, noise, scaling and warping are mostly used as training augmentations [@um2017; @iwana2021], and benchmarks such as AdaTime study adaptation to new subjects or devices when unlabelled target data are available [@ragab2023]. We are not aware of a benchmark that tests how the confidence of random-kernel classifiers behaves under sensor faults.

## 2.4 Conformal prediction

Conformal prediction turns any score into prediction sets that contain the true label with a chosen probability, provided training and test data are exchangeable [@vovk2022; @angelopoulos2023]. Split conformal prediction sets aside part of the training data [@lei2018]. Jackknife+ uses all n LOO models instead and guarantees coverage of at least 1 − 2α, with about 1 − α in practice [@barber2021], but it normally requires n refits. Adaptive prediction sets (APS) [@romano2020; @angelopoulos2021raps] and the least ambiguous set-valued classifier (LAC) [@sadinle2019] are common scores for classification. For time series, conformal methods have mostly addressed forecasting [@xu2023]. For ridge *regression*, conformal prediction can be computed in closed form [@nouretdinov2001; @burnaev2014]. We use the corresponding LOO identities to obtain jackknife+ sets for *classification*.

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

For a test series that resembles the training data, h(x) ≈ $h_{\mathrm{ref}}$, and Eq. (4) behaves like an ordinary temperature-scaled ridge. When a fault pushes a series away from the training data, h(x) grows and the probabilities move towards uniform, which is the behaviour a triage system needs. We use √h rather than the √(1 + h) of a regression prediction interval because only the parameter (epistemic) part of the uncertainty should be scaled; the inherent class overlap is handled by the softmax and its temperature. Eq. (4) is the limit c → ∞ of the family softmax(a z /√(1 + c h/$h_{\mathrm{ref}}$)) used in Bayesian moderation [@mackay1992; @kristiadi2020]. We chose this limit on eight development datasets before running the benchmark (Online Resource 2) and evaluate finite c as an ablation.

## 4.4 Fitting on exact LOO pairs

For each candidate λ, the temperature a minimises the negative log-likelihood of the n LOO pairs,

$$\mathcal L(a, \lambda) = -\frac1n \sum_{i=1}^n \sum_{k=1}^K t_{ik} \log \operatorname{softmax}_k\!\left(\frac{a\, z_{-i,k}}{\sqrt{h_{-i}/h_{\mathrm{ref}}}}\right), \qquad (5)$$

and the pair (λ, a) with the smallest value is kept, as PreVal does for its scale factor. On easy datasets the LOO predictions can separate the classes perfectly. The best temperature is then infinite and the model would claim certainty. To prevent this we smooth the targets $t_{ik}$ as @platt1999 suggested: the true class receives ($n_k$ + 1)/($n_k$ + 2), where $n_k$ is the number of training series of that class, and the rest is spread evenly over the other classes (for K = 2 this is exactly Platt's rule). The one-dimensional problem in log a is solved with L-BFGS from three starting values.

## 4.5 Closed-form jackknife+ prediction sets

Instead of a single label, a triage system can also return a set of plausible labels. Let s(x, y; f) be a non-conformity score of a probabilistic model f. We use LAC, s = 1 − p̂(y | x) [@sadinle2019], and deterministic APS, the total probability of all classes at least as probable as y [@romano2020]. Jackknife+ returns C(x) = {y : #{i : sᵢ < s(x, y; f₋ᵢ)} < (1 − α)(n + 1)}, where sᵢ = s(xᵢ, yᵢ; f₋ᵢ) [@barber2021]. It needs the prediction of every LOO model at every test point. For ridge regression these are available in closed form. With k(x, i) = x̃(x)ᵀA⁻¹x̃ᵢ = 1/n + (Vᵀψ)ᵀ$D_\lambda$ S uᵢ and residuals E = Y − Ŷ,

$$z_{-i}(x) = z(x) - \frac{k(x,i)\,E_i}{1-d_i}, \qquad h_{-i}(x) = h(x) + \frac{k(x,i)^2}{1-d_i}. \qquad (6)$$

After an O(pr) projection each test point costs O(nK), and no model is refitted. This parallels the closed-form conformal ridge regression of @nouretdinov2001, applied here to jackknife+ and to classification scores that depend on both the LOO scores and the LOO leverages. Because λ, a and the standardisation are fitted on all n examples, the coverage guarantee holds only approximately, so we check coverage empirically (Section 6.7).

![**Fig. 1** How LOOM works. One thin SVD of the standardised random-kernel features gives, for every candidate penalty λ, the exact leave-one-out scores and held-out leverages, on which λ and the temperature a are fitted. For a new recording the scores are divided by the square root of its relative leverage, and the exact leave-one-out models give jackknife+ prediction sets](figures/fig1_method.png){width=100%}

## 4.6 Cost

The SVD costs O(np·min(n, p)), the same as the standard RidgeClassifierCV readout. Ten penalties, the LOO pairs and the temperature fits add O(10·nr·K) operations. Prediction costs O(p(r + K)) per series instead of O(pK) for plain ridge, and memory is dominated by U and V, O((n + p)r) numbers. For comparison, k-fold calibration needs k additional ridge fits, logistic regression an iterative solver for each regularisation value, and an M-member ensemble M transforms and fits.

# 5 Experimental setup

## 5.1 Data

We use the 114 univariate datasets of the UCR 2018 archive [@dau2019ucr] whose official files contain equal-length series without missing values. They have 17 to 8,926 training series, lengths from 15 to 2,844 and 2 to 60 classes (Fig. 2), and come from ECG and other medical signals, motion and gesture sensors, device and energy measurements, spectrographs, images converted to outlines, and simulations. Each dataset is evaluated on the official split and on four stratified resamples that keep the official number of training series per class, following the bake-off protocol [@bagnall2017bakeoff; @middlehurst2024bakeoff]. All seeds were fixed in advance. MiniRocket features (aeon 1.6.0 [@aeon2024], 9,996 features) are fitted on the training part only. A check of the official splits found 16 datasets in which some test series are exact copies of training series (Online Resource 2, Table S1). Since all methods share the same features, this affects every method equally, but we also report results without these datasets.

![**Fig. 2** Training-set size, series length and number of classes of the UCR datasets used](figures/fig_datasets.png){width=100%}

## 5.2 Methods compared

All readouts use the same MiniRocket features within a resample. **Ridge-onehot** is the default output of ROCKET-family implementations, and **Ridge-softmax** applies a softmax with temperature 1 to the ridge scores. **PreVal** fits a global LOO scale factor with λ chosen jointly [@dempster2025preval]; we reimplemented it without target smoothing, as published. **Holdout-TS** is temperature scaling with a stratified 80/20 split. **Ridge-Platt-CV5** and **Ridge-Isotonic-CV5** use scikit-learn's CalibratedClassifierCV [@pedregosa2011] with five-fold calibration. **LogReg-CV** is L2-regularised logistic regression with C ∈ {10⁻³, 10⁻², 10⁻¹, 1} chosen by three-fold cross-validated log-loss; for cost reasons it was skipped on the four datasets with more than 2,000 training series. **PreVal-Ens5** and **LOOM-Ens5** average five models built on independently seeded MiniRocket transforms, the analogue of deep ensembles. Two classifiers that work on the raw series and produce their own probabilities complete the comparison: **QUANT** [@dempster2024quant] on all resamples, and the **Arsenal** [@middlehurst2021hc2] in aeon's MiniRocket-kernel configuration on the official split only, because the default configuration was too slow for our hardware (Online Resource 2).

To see which parts of LOOM matter, we also ran eight variants: PreVal with target smoothing (that is, LOOM without leverage; **PreVal-smooth**), LOOM without smoothing (**LOOM-nosmooth**), a moderation strength b learned on the LOO pairs (**LOOM-free-b**), a finite strength c = 1 (**LOOM-finite-c1**), leverage replaced by the mean squared distance to the five nearest training series (**LOOM-kNN**), the calibration fitted on in-sample instead of LOO pairs (**LOOM-insample**) or on an 80/20 split (**LOOM-holdout**), and λ chosen by the LOO squared error as in RidgeClassifierCV (**LOOM-lambdaSSE**).

## 5.3 Simulated sensor faults

We simulate six common kinds of sensor faults, each at three severities, and apply them to the test series only. Amplitudes are relative to each series' standard deviation (sd). The six faults are broadband noise (Gaussian, σ ∈ {0.1, 0.3, 0.5}·sd), baseline wander or drift (a slow sinusoid of amplitude {0.25, 0.5, 1}·sd), signal dropout (a segment of {10, 20, 30} % of the series replaced by linear interpolation), impulse spikes such as electrode pops (±3 sd at {1, 3, 5} % of the time points), speed variation (a smooth time warp with local slope in [1 − s, 1 + s], s ∈ {0.05, 0.1, 0.2}), and misalignment (a circular shift of {5, 10, 20} % of the length). The random draws are seeded by dataset, resample, fault type and severity, so every method sees exactly the same damaged inputs.

## 5.4 Application study: triage with a human reviewer

The benchmark applies one fault type at a time to the whole test set. In practice, most recordings are fine and a few are damaged in different ways. The application study imitates this setting on twelve datasets from three domains: *heart monitoring* (ECG200, ECG5000, TwoLeadECG, NonInvasiveFetalECGThorax1), *wearable and motion sensing* (UWaveGestureLibraryX, CricketX, GunPoint, SemgHandGenderCh2) and *industrial and energy sensing* (Wafer, FreezerRegularTrain, PowerCons, RefrigerationDevices). Each test recording is independently faulty with probability 0.3. A faulty recording receives one fault that is plausible for its domain, at a random severity. For ECG, the faults are electrode noise, baseline wander from breathing or movement, electrode pops and lead-off segments. For wearables, they are noise, spikes, packet loss and changes in movement speed. For industrial sensors, they are noise, drift, spikes and dropouts. The classifier then works as a triage system: the share b of recordings with the lowest confidence (maximum predicted probability) goes to a human reviewer, and the rest is decided automatically. We report the error rate of the automatic decisions and the share of faulty recordings that end up in the review queue, at b = 20 % (with 10 % and 30 % in Online Resource 3). This study was designed after the benchmark results were known, but its protocol was written down before it was run (Online Resource 2, Section 10). It compares LOOM, PreVal, Holdout-TS, Ridge-Platt-CV5 and QUANT on the official split and four resamples.

## 5.5 Metrics and statistics

The primary measure is the test negative log-likelihood (NLL), a strictly proper score that rewards confident correct predictions and penalises confident mistakes. We also report the expected calibration error (ECE, 15 equal-width bins), the Brier score, accuracy, the area under the risk–coverage curve (AURC) for selective classification [@geifman2017], and, on damaged data, the AUROC for telling faulty from clean recordings by 1 − max p̂ [@hendrycks2017]. For prediction sets (α = 0.1) we report coverage, average set size and the share of empty sets. The robustness score of a dataset is its average over the 18 fault conditions.

Datasets are the unit of analysis, with each metric averaged over the five resamples. We compare LOOM with each other method by the two-sided Wilcoxon signed-rank test [@wilcoxon1945] with Holm's correction within each table [@holm1979], as recommended for comparisons across datasets [@demsar2006; @benavoli2016]. A difference is called significant only if the Holm-adjusted p-value is below 0.05. The tables also give the number of datasets on which LOOM is better, equal or worse (W/D/L), the matched-pairs rank-biserial correlation $r_{\mathrm{rb}}$ as effect size [@kerby2014], and a bootstrap 95 % confidence interval of the median paired difference. In the text, "better on 108 of 114 datasets (p < 0.001)" refers to these tests, and all p-values of comparisons are Holm-adjusted. Spearman correlations in Section 6.9 are descriptive, and their p-values are not adjusted.

## 5.6 Implementation and disclosure

The code is written in Python with NumPy, SciPy, scikit-learn and aeon. Each job ran on one CPU core of a 4-core machine without a GPU, so the reported times are single-core wall-clock times. Unit tests compare every closed-form identity in Eqs. (2), (3) and (6) with explicit refits. A single script, `run_all.py`, regenerates all results, tables and figures, and every number in this paper is read from the machine-generated result files. The method was developed on eight randomly selected datasets, and six more were used to test the pipeline; target smoothing and the APS score were added at that stage, before the benchmark. Results restricted to the {{S['n_unseen']}} datasets never used during development are reported in Section 6.6.

*Use of AI tools.* As the journal requires, we state here that an AI assistant (Claude, Anthropic) was used for the literature search, for writing and running the code, and for drafting the text. All numbers were produced by the released code and inserted into the manuscript automatically from the result files. The authors take full responsibility for the content (see also the Statements and Declarations).
