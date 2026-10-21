# Review round 2 — Reviewer #2 (novelty, prior art, contribution differentiation)

*Perspective: is the contribution incremental, is the gap genuine, is relevant prior art missing?*
Reviewed: draft manuscript built from main-stage results on 111 datasets (2026-10-20).

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R2-1 | MAJOR | The core mechanism (exact LOO predictions to calibrate ridge) is PreVal (Dempster et al., arXiv 2401.15610). Without strong differentiation the work would be judged incremental. | Already addressed by design: PreVal is the primary baseline throughout; the Introduction, Related Work (§2.2) and the cover letter state that LOO calibration is prior art. The claimed contribution is restricted to (i) leverage moderation fitted on held-out leverages, (ii) closed-form jackknife+ for classification, (iii) the shift benchmark. The results support that the difference matters only under shift (clean NLL: no significant difference vs PreVal; corrupted NLL: LOOM better on 105/111 datasets). **Text checked for any "first"/"novel" overclaim — none.** |
| R2-2 | MAJOR | Closed-form conformal prediction for ridge *regression* exists (Ridge Regression Confidence Machine, Nouretdinov et al. 2001; Burnaev & Vovk 2014) but was not cited; the closed-form jackknife+ could be mistaken for a new idea in general. | **Fixed.** Both papers verified (ICML 2001 pp. 385–392; COLT 2014, PMLR 35:605–622), added to the bibliography, cited in §2.4 and §4.5, and the claim narrowed to jackknife+ sets for *classification* with scores depending on LOO logits and LOO leverages. |
| R2-3 | MODERATE | z/√h looks like a "studentised" prediction; why √h and not √(1 + h) as in regression prediction intervals? | **Fixed.** §4.3 now explains that only the epistemic (parameter) part is moderated; the aleatoric part of the classification decision is handled by the softmax temperature. |
| R2-4 | MODERATE | Distance-aware baselines (Mahalanobis/k-NN OOD scores, SNGP/DUQ) are not compared. | Partially addressed: SNGP/DUQ require network retraining and have no random-kernel counterpart (stated in §2.3); a k-NN distance moderation is evaluated as an ablation (LOOM-kNN): similar NLL, significantly worse shift detection. Listed as limitation for deep distance-aware models. |
| R2-5 | MODERATE | Bayesian logistic regression with a Laplace posterior and probit moderation would be the "textbook" alternative. | Not run (10⁴ features × many classes makes the Laplace Hessian costly); the closest closed-form analogues are evaluated (LOOM-free-b, LOOM-finite-c1 = moderation with finite strength). Added to limitations via the "Deep-learning classifiers … not run" sentence? → **Action:** add explicit sentence (done in Limitations: compute-driven choices). |
| R2-6 | MINOR | The research gap "no shift evaluation of random-kernel classifiers" must be stated carefully (bake-off redux evaluates NLL in-distribution only). | Wording in §2.3 "We are not aware of …" is appropriately hedged. |

## Verdict after fixes
The contribution is modest but genuine and clearly delimited from PreVal; the strongest
evidence is the shift experiment and the ablation isolating leverage moderation. Risk of an
"incremental" judgement remains (see final rejection-risk report).
