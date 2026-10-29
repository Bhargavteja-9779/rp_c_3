# 6 Results

Unless stated otherwise, every result uses all {{S['n_datasets']}} datasets with the official split plus four resamples, and per-dataset values are averages over the resamples. Complete per-dataset results are in Online Resource 3 and the code repository.

## 6.1 Triage of faulty recordings

We start with the practical question from the introduction. Fig. 3 shows what happens when each method decides on its own for the recordings it is most confident about and sends the rest to a human reviewer, on the twelve application datasets with 30 % faulty recordings.

![**Fig. 3** Human-in-the-loop triage on twelve sensor datasets (heart monitoring, wearables, industrial sensing) in which 30 % of the recordings carry a sensor fault. Left: error rate of the automatic decisions as more recordings are sent to review. Right: share of the faulty recordings that end up in the review queue; the dotted line is what random selection would achieve. Curves are averages over datasets of the mean over five resamples](figures/fig_case_study.png){width=100%}

With a review budget of 20 %, the clearest difference is in what reaches the reviewer. LOOM's review queue contained a median of {{pct(cs('stream:fault_caught@20','LOOM'),0)}} of the faulty recordings, against {{pct(cs('stream:fault_caught@20','PreVal'),0)}} for PreVal, {{pct(cs('stream:fault_caught@20','Holdout-TS'),0)}} for hold-out temperature scaling, {{pct(cs('stream:fault_caught@20','Ridge-Platt-CV5'),0)}} for Platt scaling and {{pct(cs('stream:fault_caught@20','QUANT'),0)}} for QUANT; random selection would catch 20 %. LOOM caught more faulty recordings than PreVal on every one of the twelve datasets ({{csbrief('stream:fault_caught@20','PreVal', lower=False)}}). Against QUANT the advantage was smaller and not significant ({{csbrief('stream:fault_caught@20','QUANT', lower=False)}}).

The errors that slip through automatically tell a more mixed story. LOOM's median error rate among the automatic decisions was {{pct(cs('stream:auto_err@20','LOOM'))}}, about half that of PreVal ({{pct(cs('stream:auto_err@20','PreVal'))}}), hold-out temperature scaling ({{pct(cs('stream:auto_err@20','Holdout-TS'))}}) and Platt scaling ({{pct(cs('stream:auto_err@20','Ridge-Platt-CV5'))}}). Across datasets, however, only the difference to hold-out temperature scaling was significant ({{csbrief('stream:auto_err@20','Holdout-TS')}}; against PreVal: {{csbrief('stream:auto_err@20','PreVal')}}). QUANT had the lowest automatic error rate ({{pct(cs('stream:auto_err@20','QUANT'))}}). It catches fewer faulty recordings than LOOM, but it classifies damaged recordings more accurately in the first place, so fewer of the recordings it keeps are wrong. With only twelve datasets these tests have limited power, and the medians should be read alongside the p-values. Results for budgets of 10 % and 30 % follow the same pattern (Online Resource 3).

The advantage over PreVal in catching faults held in each domain. With a 20 % budget, LOOM's review queue caught {{pct(csdom('cardiac','LOOM','fault_caught@20'),0)}} of the faulty ECG recordings (PreVal {{pct(csdom('cardiac','PreVal','fault_caught@20'),0)}}), {{pct(csdom('wearable','LOOM','fault_caught@20'),0)}} of the faulty wearable recordings (PreVal {{pct(csdom('wearable','PreVal','fault_caught@20'),0)}}) and {{pct(csdom('industrial','LOOM','fault_caught@20'),0)}} of the faulty industrial recordings (PreVal {{pct(csdom('industrial','PreVal','fault_caught@20'),0)}}). On a fault-free stream the methods behaved much more alike: the median automatic error rate at the same budget was {{pct(cs('clean:auto_err@20','LOOM'))}} for LOOM and {{pct(cs('clean:auto_err@20','PreVal'))}} for PreVal. LOOM therefore does not change how a system behaves on good data. Its benefit appears when some recordings are damaged, because it routes many more of them to the reviewer.

## 6.2 Calibration on clean data

Table 1 summarises performance on the clean test sets. On clean data LOOM is as good as the best existing readouts, but not better. Its median NLL is {{f(med('E1_clean','nll','LOOM'))}}. Compared with PreVal, LOOM is {{brief('E1_clean','nll','PreVal')}}, and the same holds against five-fold Platt scaling and cross-validated logistic regression (Table 1). Compared with hold-out temperature scaling, which gives up 20 % of the training data, LOOM is {{brief('E1_clean','nll','Holdout-TS')}}, and it is far better than the raw ridge outputs. Accuracy is essentially the same as PreVal's (median {{f(med('E1_clean','acc','LOOM'))}} vs {{f(med('E1_clean','acc','PreVal'))}}), since both use the same ridge scores.

There is one cost. LOOM's calibration error on clean data is higher than PreVal's (median ECE {{f(med('E1_clean','ece','LOOM'))}} vs {{f(med('E1_clean','ece','PreVal'))}}; LOOM {{brief('E1_clean','ece','PreVal')}}). The cause is the target smoothing of Section 4.4, which makes LOOM slightly underconfident when there are few training series per class (Section 6.9).

{{tab_compare('E1_clean', ['nll', 'ece', 'brier', 'acc'], ['LOOM', 'PreVal', 'Holdout-TS', 'Ridge-Platt-CV5', 'Ridge-Isotonic-CV5', 'LogReg-CV', 'QUANT', 'Ridge-softmax', 'Ridge-onehot'], '**Table 1** Clean test data: median over datasets of the mean over five resamples. Mean rank, W/D/L (datasets on which LOOM is better/equal/worse), rank-biserial correlation r_rb and Holm-adjusted Wilcoxon p-value refer to NLL')}}

## 6.3 Behaviour under sensor faults

**Leverage reacts to faults.** For clean test series the median ratio h(x)/$h_{\mathrm{ref}}$ is {{f2(lev('clean0'))}}, so the held-out leverages computed during training are on the same scale as the leverages of new clean recordings. Under faults the ratio grows with severity: at the highest severity it reaches {{f1(lev('noise3'))}} for noise, {{f1(lev('spikes3'))}} for spikes, {{f1(lev('missing3'))}} for a 30 % dropout, {{f1(lev('wander3'))}} for baseline wander and {{f1(lev('shift3'))}} for misalignment, but only {{f2(lev('warp3'))}} for speed variation (Fig. 4).

**Probabilities under faults.** Averaged over the 18 fault conditions (Table 2), LOOM's median NLL is {{f(med('E4_shifted','nll','LOOM'))}}, against {{f(med('E4_shifted','nll','PreVal'))}} for PreVal. Compared with PreVal, LOOM is {{brief('E4_shifted','nll','PreVal')}}, and it is also better than hold-out temperature scaling, Platt and isotonic calibration and logistic regression (Table 2). Fig. 5 shows how the gap opens as the faults become stronger, and Fig. 6 shows the per-dataset picture. Accuracy is unchanged, because LOOM and PreVal share the same ridge scores (median {{f(med('E4_shifted','acc','LOOM'))}} vs {{f(med('E4_shifted','acc','PreVal'))}}). The gain comes entirely from lower confidence on the damaged recordings that are misclassified, which is visible in the reliability diagrams of Fig. 7. The same mechanism makes faulty recordings easier to spot. The median AUROC for telling faulty from clean recordings by 1 − max p̂ is {{f(med('E4_shifted','shift_auroc','LOOM'))}} for LOOM and {{f(med('E4_shifted','shift_auroc','PreVal'))}} for PreVal, where 0.5 means no separation at all.

{{tab_compare('E4_shifted', ['nll', 'ece', 'acc', 'shift_auroc'], ['LOOM', 'PreVal', 'Holdout-TS', 'Ridge-Platt-CV5', 'Ridge-Isotonic-CV5', 'LogReg-CV', 'QUANT', 'Ridge-softmax', 'Ridge-onehot'], '**Table 2** Damaged test data: per-dataset mean over 18 conditions (six fault types × three severities), then median over datasets. Statistics as in Table 1 (NLL)')}}

![**Fig. 4** How far faulty recordings move away from the training data: distribution over datasets of the median leverage ratio h(x)/h_ref (log scale) for clean data and each fault type at severities 1–3](figures/fig_leverage_ratio.png){width=95%}

![**Fig. 5** Median test NLL over datasets against fault severity for each fault type (severity 0 = clean)](figures/fig_robustness_nll.png){width=100%}

![**Fig. 6** Per-dataset NLL of LOOM against PreVal on clean data (left) and averaged over the 18 fault conditions (right); points below the diagonal favour LOOM. NLL values below 10⁻³ (datasets whose LOO predictions separate the classes perfectly, where unsmoothed PreVal becomes almost deterministic) are drawn at 10⁻³](figures/fig_scatter_loom_vs_preval.png){width=95%}

![**Fig. 7** Reliability diagrams pooled over all datasets and resamples for clean data (left) and the strongest faults (right). Bins with fewer than 200 pooled predictions are not drawn; the pooled ECE in the legend uses all bins and differs from the per-dataset medians of Tables 1 and 2](figures/fig_reliability.png){width=95%}

## 6.4 Comparison with ensembles and raw-series classifiers

Table 3 compares LOOM with five-member ensembles and with the two classifiers that work on the raw series, on the official split. The Arsenal was only run on clean data and the strongest faults, so this comparison uses the six severity-3 conditions for every method. (An earlier version of the analysis averaged the other methods over all 18 conditions; we corrected this after seeing the results, and the correction works against LOOM. Online Resource 2 documents the change.)

On clean data a five-seed PreVal ensemble is slightly better than a single LOOM model (LOOM {{brief('E2b_raw_clean','nll','PreVal-Ens5')}}), while LOOM beats QUANT (LOOM {{brief('E2b_raw_clean','nll','QUANT')}}) and the Arsenal. Under the strongest faults LOOM beats the PreVal ensemble (LOOM {{brief('E2b_raw_shifted','nll','PreVal-Ens5')}}) and the Arsenal, but QUANT does better than LOOM (LOOM {{brief('E2b_raw_shifted','nll','QUANT')}}). Averaged over all 18 conditions and five resamples, LOOM and QUANT do not differ significantly (Table 2). Which one is better depends on the type of fault (Fig. 5): at the highest severity QUANT has the lower median NLL for {{fam_vs('QUANT')}}, and LOOM for {{fam_vs('QUANT', who='loom')}}. We did not investigate the reason. It concerns the features (quantiles of the raw series versus random convolutions) rather than the readout, so applying a LOOM-style correction to a QUANT-like model is a natural next step.

The Arsenal's poor NLL deserves a comment. Its probabilities are weighted votes of its members, and they often give exactly zero probability to the true class, which the logarithm in the NLL punishes without limit. The bounded Brier score gives a fairer picture. On clean data the gap is small (median {{f(med('E2b_raw_clean','brier','Arsenal'))}} vs {{f(med('E2b_raw_clean','brier','LOOM'))}} for LOOM), while under strong faults LOOM is still clearly ahead (LOOM {{brief('E2b_raw_shifted','brier','Arsenal')}}). Averaging five LOOM models helps only a little (median severity-3 NLL {{f(med('E2b_raw_shifted','nll','LOOM-Ens5'))}} vs {{f(med('E2b_raw_shifted','nll','LOOM'))}} for a single model).

{{tab_compare('E2b_raw_shifted', ['nll', 'brier', 'ece', 'acc', 'shift_auroc'], ['LOOM', 'LOOM-Ens5', 'PreVal', 'PreVal-Ens5', 'QUANT', 'Arsenal'], '**Table 3** Official split under the strongest faults (per-dataset mean over the six severity-3 conditions, then median over datasets). Statistics as in Table 1 (NLL)')}}

## 6.5 What makes LOOM work

Fig. 8 compares LOOM with eight variants. Three findings stand out.

*The leverage scaling is what helps under faults.* PreVal with the same target smoothing, which is LOOM without leverage, is much worse on damaged data (median NLL {{f(med('E3_ablation_shifted','nll','PreVal-smooth'))}} vs {{f(med('E3_ablation_shifted','nll','LOOM'))}}; LOOM {{brief('E3_ablation_shifted','nll','PreVal-smooth')}}). On clean data, LOOM compared with it is {{brief('E3_ablation_clean','nll','PreVal-smooth')}}, so the scaling costs at most a small amount of clean-data NLL.

*The exact LOO pairs matter.* Fitting the same calibration on in-sample pairs instead of LOO pairs gives a median clean NLL of {{f(med('E3_ablation_clean','nll','LOOM-insample'))}}, because in-sample scores are overconfident. Using an 80/20 split gives {{f(med('E3_ablation_clean','nll','LOOM-holdout'))}}, because the ridge is then trained on less data. Both variants are also worse under faults.

*The moderation strength is a trade-off.* A finite strength (c = 1) or a strength learned from the LOO pairs gives a slightly lower clean NLL than LOOM (LOOM {{brief('E3_ablation_clean','nll','LOOM-finite-c1')}} against c = 1). But both detect faults less well: the median detection AUROC is {{f(med('E3_ablation_shifted','shift_auroc','LOOM-finite-c1'))}} for c = 1 and {{f(med('E3_ablation_shifted','shift_auroc','LOOM-free-b'))}} for the learned strength, against {{f(med('E3_ablation_shifted','shift_auroc','LOOM'))}} for LOOM. The learned strength is also worse in NLL under faults (median {{f(med('E3_ablation_shifted','nll','LOOM-free-b'))}}). This is probably because the held-out leverages of the training data vary too little to pin the strength down, as we had already seen during development. The full scaling chosen before the benchmark thus gives up a little clean-data NLL in exchange for better fault detection.

Removing the target smoothing gives the lowest clean ECE but is worse under faults (LOOM {{brief('E3_ablation_shifted','nll','LOOM-nosmooth')}}). Replacing leverage by a nearest-neighbour distance gives a similar NLL under faults but detects them worse (median AUROC {{f(med('E3_ablation_shifted','shift_auroc','LOOM-kNN'))}}). Choosing λ by squared error instead of likelihood changes little.

![**Fig. 8** Ablation: median paired NLL difference (variant − LOOM) with bootstrap 95 % confidence intervals on clean data (left) and under faults (right); positive values mean that the variant is worse](figures/fig_ablation.png){width=100%}

## 6.6 Does it carry over?

**Other feature extractors.** With MultiRocket features (official split, clean data and the strongest faults), LOOM compared with PreVal is {{brief('E5_multirocket_shifted','nll','PreVal')}} under faults and {{brief('E5_multirocket_clean','nll','PreVal')}} on clean data. With Hydra features it is {{brief('E5_hydra_shifted','nll','PreVal')}} under faults and {{brief('E5_hydra_clean','nll','PreVal')}} on clean data.

**Subsets of datasets.** The results do not depend on the datasets used during development: on the {{S['n_unseen']}} datasets that were never used for development, LOOM compared with PreVal is {{brief('sens_unseen_shifted','nll','PreVal')}} under faults. They also hold without the 16 datasets with train/test duplicates (LOOM {{brief('sens_nodup_shifted','nll','PreVal')}}). Some UCR datasets come from the same source, for example the four UWaveGestureLibrary sets. After averaging related datasets into {{S['sens_grouped_shifted']['nll']['n_units']}} source-level units, LOOM still has the lower NLL under faults for {{pwg('shifted','nll','PreVal')['ref_wins']}} of {{pwg('shifted','nll','PreVal')['n']}} units (p {{pv(pwg('shifted','nll','PreVal')['p_holm'])}}).

**Training-set size.** With only 25 % or 50 % of the training data (official split), LOOM's median NLL under the strongest faults is {{f(ts(0.25,'shifted','PreVal','median_loom'))}} and {{f(ts(0.5,'shifted','PreVal','median_loom'))}}, against {{f(ts(0.25,'shifted','PreVal','median_comp'))}} and {{f(ts(0.5,'shifted','PreVal','median_comp'))}} for PreVal ({{ts(0.25,'shifted','PreVal','n')}} datasets; Table S4). {{tsclaim()}}

**A real change of sensor.** Simulated faults are only a model of reality. As a check with real data, we used the three accelerometer axes of UWaveGestureLibrary, which record the same {{ds_total('UWaveGestureLibraryX')}} gestures with identical labels. A model trained on one axis was tested on the same gestures recorded by another axis ({{S['E10_natural_shift']['n_tasks']}} axis pairs × {{S['E10_natural_shift']['n_resamples']}} resamples, with test recordings disjoint from training recordings). This is a severe change: accuracy drops from {{f(ns('LOOM','acc_in'))}} to {{f(ns('LOOM','acc_shift'))}}. Yet leverage barely moves (median ratio {{f2(nslev()[0])}}–{{f2(nslev()[1])}}), because the other axes produce features inside the range of the training features. LOOM's advantage is correspondingly small: mean NLL {{f(ns('LOOM','nll_shift'))}} against {{f(ns('PreVal','nll_shift'))}} for PreVal (lower on {{nsw('nll_shift','PreVal')['ref_wins']}} of {{nsw('nll_shift','PreVal')['n']}} axis pairs), and detection AUROC {{f(ns('LOOM','shift_auroc'))}} against {{f(ns('PreVal','shift_auroc'))}}. Leverage scaling protects against changes that push recordings away from the training data, not against every kind of change (Section 8).

## 6.7 Prediction sets

Table 4 reports coverage and set size at α = 0.1, that is, for a target coverage of 90 %. On clean data LOOM's closed-form jackknife+ sets contain the true label for {{pct(cp('CP-LOOM-jackknife+'))}} of the test series on average, with {{f2(cp('CP-LOOM-jackknife+', False, 'size_mean'))}} labels per set, while split conformal prediction reaches {{pct(cp('CP-Split-TS'))}} with {{f2(cp('CP-Split-TS', False, 'size_mean'))}} labels. Under faults, coverage drops for every method, because damaged recordings are no longer exchangeable with the training data. Here the choice of score matters. With the LAC score, LOOM's flattened probabilities produce empty sets for {{pct(cp('CP-LOOM-jackknife+', True, 'empty_mean'))}} of the damaged recordings. With the APS score, flat probabilities instead produce larger sets, which keeps coverage at {{pct(cp('CP-LOOM-jackknife+-APS', True))}} with only {{pct(cp('CP-LOOM-jackknife+-APS', True, 'empty_mean'))}} empty sets. For a triage system, a large set is a useful signal in itself.

{{tab_conformal('**Table 4** Prediction sets at α = 0.1 (target coverage 0.9). Coverage and set size are averaged over datasets (damaged data: over datasets and the 18 conditions); 10th pct. is the 10th percentile of per-dataset coverage')}}

## 6.8 Cost and sensitivity

{{tab_eff('**Table 5** Fit and prediction time per dataset (mean over resamples, single CPU core). Readouts on MiniRocket features include the ridge SVD they use but not the shared MiniRocket transform, which is listed separately; prediction times exclude the transform of the test sets. The ensembles include all five MiniRocket transforms, and QUANT and the Arsenal include their own feature extraction', ['MiniRocket-transform', 'Ridge-onehot', 'PreVal', 'LOOM', 'Holdout-TS', 'Ridge-Platt-CV5', 'Ridge-Isotonic-CV5', 'LogReg-CV', 'LOOM-Ens5', 'QUANT', 'Arsenal'])}}

LOOM costs the same to fit as PreVal (median {{f(eff('LOOM'), 2)}} s vs {{f(eff('PreVal'), 2)}} s on top of the shared MiniRocket transform, which takes a median of {{f(eff('MiniRocket-transform'), 2)}} s; Table 5 and Fig. 9). From raw data to a fitted model, a single LOOM model takes about {{f(eff('MiniRocket-transform') + eff('LOOM'), 2)}} s, compared with {{f(eff('QUANT'), 2)}} s for QUANT, {{f(eff('LOOM-Ens5'), 2)}} s for a five-seed ensemble and {{f(eff('Arsenal'), 2)}} s for the Arsenal. Prediction is slower than for plain ridge because each new series is projected onto the singular vectors (Section 4.6), but it remains a fraction of a second per dataset.

The moderation strength was discussed in Section 6.5. For the prediction sets, the clean coverage of LOOM's jackknife+ is {{pct(cp('CP-LOOM-jackknife+@0.05'))}} at α = 0.05 and {{pct(cp('CP-LOOM-jackknife+@0.2'))}} at α = 0.2. We also checked whether the logistic-regression baseline was trained long enough. On 12 random datasets, ten times more iterations and a tighter tolerance changed its median NLL from {{f(LR['med_def'],4)}} to {{f(LR['med_long'],4)}}, but improved it by {{f(LR['max_gain'])}} on {{LR['max_ds']}}, a dataset with many classes. The clean-data comparison with logistic regression should therefore be read as approximate for such datasets (Online Resource 3).

![**Fig. 9** Fit time against the number of training series (log–log, single core)](figures/fig_efficiency.png){width=70%}

## 6.9 When LOOM fails

{{ERROR_ANALYSIS}}

# 7 Discussion

{{DISCUSSION}}

# 8 Limitations

{{LIMITATIONS}}

# 9 Conclusion

{{CONCLUSION}}
