# Data leakage audit

Script: `code/experiments/e0_leakage_audit.py` → `code/results/e0/leakage_audit.json` and
`leakage_audit.csv`. Run on all 114 datasets (official splits).

## Checks and results
| Check | Method | Result |
|---|---|---|
| Exact train/test duplicates | rounded to 6 decimals, hashed | 16 datasets with ≥ 1 test series identical to a training series: Crop, ElectricDevices, FaceAll, FreezerRegularTrain, FreezerSmallTrain, InlineSkate, InsectEPGSmallTrain, MedicalImages, MixedShapesRegularTrain, PowerCons, ShapesAll, Strawberry, SwedishLeaf, ToeSegmentation1, Worms, WormsTwoClass |
| Label conflicts among exact duplicates | same series, different label | 2 datasets: ElectricDevices (101 of its 208 duplicated test series carry a different label from the identical training series) and Strawberry (1). This is a data-quality issue of the archive, not leakage into our method, and affects all methods equally |
| Near-duplicates across the split | z-normalised RMS distance to the nearest training series < 0.01 sd | 24 datasets in total; 8 have near-duplicates but no exact duplicates: ACSF1, Adiac, ChlorineConcentration, Meat, OliveOil, SmallKitchenAppliances, Wafer, Wine |
| Test-set tuning | code inspection + check | λ, the temperature and the leverage reference are fitted only on training data via exact LOO; `LOOM.fit` receives no test data. Refitting after permuting the test labels gives bit-identical parameters and predictions (ECG200) |
| Development exposure | records | 8 development and 6 pipeline-test datasets (registration §7); results without them are reported |
| Label / prompt leakage | — | not applicable (no prompts; labels used only for training and scoring) |

## Planted-leak test (does the audit work?)
Five test series of GunPoint were copied exactly into its training set and five more with added
noise of 0.001 sd. Before planting, GunPoint had no near-duplicates. The audit flagged exactly
the 5 exact copies as exact duplicates and exactly the 10 planted series as near-duplicates,
with no false positives (`planted_leak_test.passed = true`).

## Consequences for the analysis
* Duplicates affect every method identically because all readouts share the same features.
* Sensitivity analyses exclude (a) the 16 datasets with exact duplicates (98 remain) and (b) the
  24 datasets with exact or near duplicates (90 remain). LOOM's advantage over PreVal under faults
  holds in both (b: better on 84 of 90 datasets, Holm p < 0.001).
* The near-duplicate sensitivity analysis (b) was added post hoc (deviation D19).
