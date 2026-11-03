# Review round 3 — Reviewer #3 (statistical validity, fairness, seeds, splitting, leakage)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R3-1 | MAJOR | Datasets are treated as independent units, but many UCR datasets share a source (UWaveGestureLibrary X/Y/Z/All, six phalanx-outline sets, Pig*, Freezer/InsectEPG Regular/Small, UK appliances, …); Wilcoxon p-values may be anti-conservative. | **New analysis** (`grouped_test`, 20 grouping rules): related datasets averaged into source-level units (69 units on the 111-dataset draft). Conclusions unchanged (shifted NLL: LOOM better than PreVal on 67/69 units, Holm p < 0.001; clean NLL: not significant). Reported in §6.5. |
| R3-2 | MAJOR | Several metrics are tested; Holm correction is applied only within each table/metric. | NLL is the pre-registered primary endpoint (Online Resource 2); other metrics are secondary and interpreted descriptively. Stated in §5.4 ("primary endpoint"). |
| R3-3 | MODERATE | Resamples within a dataset are not independent; averaging over resamples before testing is correct, but the number of resamples differs for four large datasets (amendment 6). | Per-dataset means use the available resamples; documented in Online Resource 2 and Limitations. |
| R3-4 | MODERATE | Seeds: are any chosen after results? | All seeds are functions of (dataset, resample, condition) fixed in code before the benchmark (pre-registration commit 3d20c3f precedes all full-run results). |
| R3-5 | MODERATE | Leakage: duplicate series across official train/test splits in 16 datasets. | Leakage audit table; sensitivity analysis excluding these datasets (§6.5). Duplicates affect all methods equally (shared features). |
| R3-6 | MODERATE | Hyper-parameter selection fairness: LOOM tunes λ and a on LOO NLL; PreVal identical procedure; Holdout-TS tunes on a split; CV calibrators on 5 folds; LogReg on 3-fold CV. No method sees test data. | ✔ Verified in code. LogReg tuning caveat (R4-1) disclosed. |
| R3-7 | MINOR | Friedman + Nemenyi CD are reported although Benavoli et al. advise against mean-rank post-hoc tests. | CD is shown only descriptively in the rank figure; inference uses Wilcoxon–Holm. |
| R3-8 | MINOR | Conformal coverage is an average over datasets; a mean of 0.90 could hide under-coverage. | 10th percentile of per-dataset coverage reported in Table 4. |
| R3-9 | MINOR | Wilcoxon with zero differences (accuracy ties). | `zero_method="wilcox"` (zeros dropped); W/D/L reports draws explicitly. |
