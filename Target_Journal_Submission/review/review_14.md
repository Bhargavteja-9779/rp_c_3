# Review round 14 (SIMULATED) — Reviewer A: novelty and method, trying to reject

*Simulated review, not an external review.* Version reviewed: elsarticle build of 2026-11-14 (scoped revision after the
JKSU-CIS desk rejection; ESWA target).

**Recommendation of the simulated reviewer: major revision / reject.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| A1 | "LOOM is PreVal plus a leverage factor; the ingredients are textbook." | Major | Contributions rewritten: novelty is claimed only for the combination (LOO-fitted leverage moderation without held-out data) and for the evaluation on real annotated faults. The ingredients are called classical in the introduction. | Residual risk (stated) |
| A2 | "The method fails on the only real, expert-annotated faults. Why publish it?" | Major | The failure is the main new finding. It is pre-registered (H8, H9), stated in the abstract and conclusion, and turned into a usable boundary: strong whole-recording artefacts vs mild annotated noise (Sect. 6.8, Fig. 9). Practical lessons are drawn in the discussion. | Residual risk; argued in cover letter |
| A3 | "H9 failed in the opposite direction: does LOOM harm clinical use?" | Moderate | Reported with its size (+0.034 nats on noisy records) in the abstract and Sect. 6.6; the conclusion now calls LOOM a safeguard against gross faults, not a signal-quality detector. | Fixed (wording) |
| A4 | "Is the PTB-XL failure about MiniRocket features rather than the readout?" | Moderate | QUANT (different features, tree ensemble) and Platt are also close to chance; sentence added to Sect. 6.8. | Fixed |
| A5 | "Multichannel recordings: LOOM pools all 12 leads into one leverage." | Moderate | Stated in the method (Sect. 3) and named as future work (per-lead leverage); not tested. | Limitation |
| A6 | "The hybrid was an obvious idea; its failure is uninformative." | Minor | A post hoc check explains the failure (QUANT's own queue catches more of QUANT's errors on 30/37 datasets); the practical rule (queue and label from the same model) is stated. | Addressed (post hoc, labelled) |

**Verdict after actions.** Novelty remains the main risk; the paper is now honest about its limits and adds real-fault evidence.
