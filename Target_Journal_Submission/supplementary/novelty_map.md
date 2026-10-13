# Prior-art / novelty map (Phase 4)

Searches performed (2026-10-12/13): arXiv API (10 boolean queries on ROCKET/MiniRocket ×
calibration/uncertainty/conformal/leverage), web search (extended mode) for ROCKET
probability calibration, conformal time-series classification, leverage-based uncertainty
for ridge classifiers, LOO Platt scaling for LS-SVM / ridge, and HIVE-COTE Arsenal
probabilities. Patents were not searched systematically (no patent database access); this is
listed as a limitation.

| Existing work | Its contribution | Overlap with this work | Genuinely different element here |
|---|---|---|---|
| PreVal (Dempster et al., arXiv 2401.15610) | Exact LOO ("prevalidated") predictions of ridge used to fit one scale factor κ (and λ) by log-loss; evaluated on 106 UCR datasets with MiniRocket | **High**: closed-form LOO calibration of ridge on MiniRocket features is theirs | (i) per-input moderation by leverage, with the held-out leverage h₋ᵢ = dᵢ/(1−dᵢ) used to fit it on LOO pairs; (ii) evaluation under test-time shift; (iii) closed-form jackknife+ sets; (iv) ECE/Brier/AURC and resampled significance testing |
| Arsenal / HIVE-COTE 2.0 | Probabilities from an ensemble of ROCKET classifiers | Same goal (probabilities) | Single model, no ensemble; closed form |
| Temperature scaling (Guo et al. 2017) | Global temperature on hold-out data | LOOM also fits one temperature | Fitted on exact LOO pairs (no data held out) and applied to leverage-standardised logits |
| Bayesian moderated output (MacKay 1992); last-layer Laplace (Kristiadi et al. 2020) | Shrink logits by predictive variance | The z/√h form is a limiting case of moderation | Variance factor is the exact ridge leverage; scale fitted on exact LOO pairs; applied to random-kernel TSC with an empirical shift benchmark |
| SNGP / DUQ (Liu et al.; van Amersfoort et al.) | Distance-aware uncertainty in deep nets | Distance awareness | No retraining and no architecture change; closed form for a linear readout |
| LS-SVM closed-form LOO (Cawley & Talbot 2004) | Exact LOO for model selection | Exact LOO formulas | Used for leverage-moderated calibration and conformal sets |
| Jackknife+ (Barber et al. 2021); APS (Romano et al. 2020) | Coverage guarantees from LOO / CV models | Same set construction | Exact LOO models evaluated at test points in closed form (no refits) |
| Leverage / hat matrix in diagnostics (Hoaglin & Welsch 1978; Cook 1977) | Leverage flags influential / extrapolated points | Uses the same quantity | Turned into a calibrated predictive-confidence signal for classification |

**Decision.** The core of the contribution (exact LOO calibration of the MiniRocket ridge)
is prior art (PreVal) and is presented as such: PreVal is our primary baseline. The
contribution is the leverage-moderated extension, the closed-form jackknife+ sets, and
the shift evaluation. Whether these suffice for publication is assessed in the
reviewer rounds; the claimed novelty is limited to these elements.
