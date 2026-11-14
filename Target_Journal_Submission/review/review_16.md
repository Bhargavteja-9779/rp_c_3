# Review round 16 (SIMULATED) — Reviewer C: application, journal fit, reproducibility and ethics, trying to reject

*Simulated review, not an external review.* Version reviewed: elsarticle build of 2026-11-14.

**Recommendation of the simulated reviewer: minor revision.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| C1 | "Is this an expert-systems paper?" | Moderate | Framed as a decision-support component (when to hand a case to a human expert) with practical lessons and a validated tool; cover letter argues the fit. | Residual risk (editor's judgement) |
| C2 | "The previous version had no real field data." | Major (earlier) | PTB-XL: 16,307 clinical 12-lead ECGs with expert noise annotations, recommended patient-separated folds. | Fixed |
| C3 | "Clinical implications?" | Moderate | No clinical claims. The conclusion advises a dedicated signal-quality model in front of the classifier for routine clinical noise. | Addressed |
| C4 | "Ethics of using clinical data." | Minor | Public, de-identified data (CC BY 4.0), secondary analysis; a placeholder asks the authors to confirm whether their institution needs an ethics statement. | Author action |
| C5 | "The paper is long (39 pages in the 12-pt preprint layout)." | Moderate | ESWA's length rules could not be retrieved automatically; the package README asks the authors to check them. Candidates to move to the supplement: Table 1, the qualitative figure, the ablation figure. | Author decision |
| C6 | "Generative AI use." | Moderate | Elsevier-style declaration in the back matter and title page; usage also described in Sect. 5.1. Authors must confirm. | Author action |
| C7 | "Reproducibility of the new experiments." | Minor | One script per experiment (E20–E22), checkpointing, memory-bounded QUANT with a unit test; all numbers from macros. | Fixed |

**Verdict after actions.** The application side is much stronger than before; fit and length are the editor's call.
