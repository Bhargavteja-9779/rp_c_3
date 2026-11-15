# Review round 17 (SIMULATED) — Reviewer D: ESWA reviewer trying to reject the third revision

*Simulated review, not an external review.* Version reviewed: elsarticle build of 2026-11-15 with E23 (CinC 2011)
and E24 (deep baselines). The reviewer compares the paper with the recent ESWA papers listed in
`journal_selection/eswa_benchmark.md` (titles and metadata only).

**Recommendation of the simulated reviewer before actions: major revision. After actions: minor revision.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| D1 | "CinC was picked after PTB-XL failed. Is this a search for a dataset that works?" | Major | The paper now says so in the methods (Sect. 5.2): the study was added after H8/H9 failed, to test the remaining claim, and H11/H12 were registered (3de341c) before any CinC result. The PTB-XL failures stay in the abstract, highlights and conclusion. | Fixed (disclosed); residual judgement for the editor |
| D2 | "A trivial amplitude threshold does as well as LOOM over all records." | Major | Reported in the results, table and discussion (AUROC 0.861 vs 0.863, difference CI includes 0; the rule catches more at 20 %). The advantage over the rule is shown on records below 10 mV, with 5 and 20 mV as sensitivity; labelled post hoc (D29) in text, table and figure. | Fixed (disclosed); the subset analysis is post hoc |
| D3 | "Leverage alone beats LOOM. Why propose LOOM?" | Major | Stated in results and discussion: for quality triage alone, use the leverage; LOOM's confidence is for when one number drives both the decision and the queue. | Fixed (disclosed) |
| D4 | "No deep-learning baseline; ESWA papers use deep models." | Major (earlier) | E24: 1-D ResNet (134,725 parameters), deep ensemble of 5 and MC dropout, registered before running. Deep ensemble is the most accurate model on PTB-XL and makes the fewest automatic errors; on CinC single networks and MC dropout are worse than chance, the ensemble below LOOM. Reported both ways. | Fixed |
| D5 | "The deep baselines are small and CPU-trained." | Moderate | Stated as a limitation; network size chosen for the CPU budget, registered in advance. | Residual |
| D6 | "One example record could be cherry-picked." | Minor | Chosen by a fixed rule stated in the caption (smallest record number satisfying the conditions). | Fixed |
| D7 | "Seed 0 only for the tests?" | Minor | The registered tests use seed 0; the same tests are reported for all seeds (macros `\CincHElevenSeedsPass`, `\CincHTwelveSeedsPass`) and the table gives seed ranges. | Fixed once all seeds finish |
| D8 | "The realism story changed between versions." | Moderate | The post hoc realism figure now includes CinC and its caption and text say the gain follows fault strength, which is an interpretation of five studies, not a test. | Fixed (wording hedged: "consistent with") |
| D9 | "Position against recent ESWA work?" | Moderate | Related work cites eight 2025–2027 ESWA papers on MiniRocket fault diagnosis, kernel-based TSC, risk-aware classification, deferral and routing, and wearable-ECG denoising, only for what their titles and (one) abstract say. | Fixed |
| D10 | "Dedicated signal-quality indices reach 97 % on CinC." | Moderate | Cited (Clifford et al. 2012) with the distinction that they are trained with quality labels; LOOM needs none. No claim that LOOM beats them. | Fixed |

**Verdict after actions.** The paper now has a positive pre-registered result on real, strong faults, a deep-learning
comparison, and honest caveats (amplitude rule, leverage alone, post hoc subset). The main residual risks are the
editor's view of novelty and that the CinC study was added after the PTB-XL failure.
