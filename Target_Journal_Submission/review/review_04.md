# Review round 4 — Reviewer #4 (ML methodology, optimisation, implementation, code–paper consistency)

*Perspective: does the code do what the paper says? Are training protocols and baselines sound?*
Inspected: `src/loom/models/{readout,calibration,loom,baselines}.py`, `src/loom/training/runner.py`,
`run_all.py`, unit tests, and manuscript §4–5.

## Code–paper consistency checks
| Item | Paper | Code | Status |
|---|---|---|---|
| Targets | one-vs-rest ±1 | `_encode` → {−1,+1} | ✔ |
| λ grid | {10⁻³…10³}·n, ten values, standardised features | `np.logspace(-3,3,10)`, `scale_by_n=True`, StandardScaler-equivalent | ✔ |
| Hat diagonal, LOO logits, held-out leverage, test leverage, cross-leverage | Eqs. (2), (3), (6) | `ClosedFormRidge` | ✔ verified against explicit refits in `tests/test_core.py` (machine precision) |
| Moderation | softmax(a z /√(h/h_ref)), h_ref = median h₋ᵢ | `LOOM._moderate`, `h_ref` per λ | ✔ |
| Selection of λ and a | jointly by LOO NLL | loop over λ, `fit_moderation` | ✔ |
| Platt smoothing | (n_k+1)/(n_k+2) for true class, rest uniform | `soft_targets` | ✔ (unit test for K=2) |
| Jackknife+ | Barber et al. with exact f₋ᵢ(x) | `conformal_sets` | ✔ unit test vs explicit refits |
| Holdout-TS | 80/20, λ by LOO-SSE on 80 % | `_fit_holdout` | ✔ (text clarified in this round) |
| LOOM-kNN | mean squared distance, 5-NN | `_knn_prepare/_knn_dist` (standardised space) | ✔ (text clarified) |
| CV calibrators | 5-fold OOF, final model on all data | `CalibratedClassifierCV(ensemble=False)` | ✔ |

## Findings
| # | Severity | Issue | Resolution |
|---|---|---|---|
| R4-1 | MAJOR | Logistic-regression baseline uses `max_iter=300`, `tol=1e-3` with warnings suppressed: possible under-convergence weakens a baseline. | **Experiment run** (`experiments/check_logreg_convergence.py`, 12 random datasets, official split): with `max_iter=3000, tol=1e-5` the median NLL changes from 0.2707 to 0.2711; 6/12 datasets improve, but on ShapesAll (60 classes) NLL improves by 0.161 and on SmoothSubspace by 0.031, where the selected C moves to the top of the grid. **Disclosed** in §6.8 and Limitations; the clean-data comparison LOOM vs LogReg-CV must be read with this caveat. Under corruption a larger C makes LR more confident, so the qualitative shift result is not expected to change (not re-run for all datasets: compute). |
| R4-2 | MAJOR | `CalibratedClassifierCV` fails when a class has a single training example (Phoneme, FiftyWords, DiatomSizeReduction, … resamples). | Not a bug: k-fold calibration is undefined there. These datasets are excluded only from the affected pairwise comparisons (n reported per method); stated in §5.2 and Table notes. LOOM and Holdout-TS handle singletons. |
| R4-3 | MODERATE | λ and the temperature are selected on the same LOO pairs (optimistic LOO NLL; approximate jackknife+). | Already stated (§4.5, Limitations); coverage checked empirically. |
| R4-4 | MODERATE | The analysis originally dropped any dataset on which some method was missing, which would have silently removed the largest datasets from all comparisons (LogReg skipped for n_train > 2000). | **Fixed** (`summary_table`): pairwise tests on per-pair available datasets, Friedman/ranks only for methods with complete coverage, n reported per method. |
| R4-5 | MODERATE | Number of datasets with n_train > 2000 was stated as 13; the data list shows 4 (Crop, ElectricDevices, FordA, FordB). | **Fixed** in §5.2 and Limitations. |
| R4-6 | MINOR | APS scores have exact ties at 1.0 for binary problems; tie-breaking depends on BLAS thread count in the last bit. | Documented in README (single-thread runs are bitwise reproducible). |
| R4-7 | MINOR | Prediction cost of LOOM includes the O(pr) projection; the efficiency table should not hide it. | Table 5 reports prediction time separately. |

## Verdict
Implementation matches the description; one baseline (LogReg-CV) may be under-tuned on a few
many-class datasets, which is now disclosed with quantitative evidence.
