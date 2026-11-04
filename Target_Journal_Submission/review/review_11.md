# Review round 11 (SIMULATED) — Reviewer A: novelty and significance, trying to reject

*This is a simulated review written by the authors' assistant to stress-test the paper. It is not
an external review.* Version reviewed: LaTeX build of 2026-11-04 (main 25 pp., supplement 9 pp.).

**Recommendation of the simulated reviewer: major revision / reject in current form.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| A1 | "Dividing scores by the square root of leverage is textbook (MacKay 1992; regression prediction intervals). The paper repackages classical facts." | Major | Rewording. The contribution statement already says the ingredients are classical. We added that the leverage is a ridge-regularised Mahalanobis distance (Sect. 4.3), to make the link explicit. The claimed contribution is the LOO-fitted calibration without held-out data and the triage evaluation. | Addressed by wording; residual novelty risk recorded in the final audit |
| A2 | "The title promises *reliable* confidence, but clean-data ECE and Brier are worse than PreVal's (H1b fails)." | Major | Rewording. Title changed to "fault-aware confidence"; the abstract, Table 3 and limitations state the clean-data cost. | Fixed |
| A3 | "Under faults QUANT is as good (median NLL 0.732 vs 0.738) and better at severity 3. Why would anyone use LOOM?" | Major | Already reported (Table 5, Table 8, limitation 4). The answer is supported by experiments: LOOM detects faulty recordings better (AUROC 0.765 vs 0.709; real noise 0.973 vs 0.899), catches more faults in triage (34/38 datasets vs QUANT), and costs about a quarter of QUANT's end-to-end fit time. Fig. 2's caption was corrected from "clearly better under faults" to "better than the other readouts, with QUANT close to it". | Fixed (caption); trade-off stated |
| A4 | "Leverage is just one distance-based OOD score. Why not Mahalanobis or k-NN distances?" | Moderate | Existing evidence: the k-NN ablation gives similar NLL but worse detection (Sect. 6.7). The leverage *is* a regularised Mahalanobis distance (now stated). | Addressed |
| A5 | "The faults are simulated and favour leverage-based methods." | Major | New experiment: real recorded noise (MIT-BIH NSTDB, E12). H4 passes (50/63 combinations, p < 0.001). It is still ECG only and still added synthetically to clean beats; this is listed as limitation 1. | Partly addressed; limitation |
| A6 | "Significance: does a better review queue matter in practice?" | Moderate | The confirmatory triage study (38 datasets) gives an operator-level number: 41 % instead of 24 % of damaged recordings reach review at a 20 % budget. A new prevalence check (E19) gives 55 % vs 27 % at 10 % prevalence. | Addressed with new analysis |

**Verdict after actions.** The paper is honest about its limits. Novelty remains moderate; the
strongest arguments are the triage results and the detection advantage over QUANT.
