# Review round 13 (SIMULATED) — Reviewer C: application, reproducibility and ethics, trying to reject

*Simulated review, not an external review.* Version reviewed: LaTeX build of 2026-11-04.

**Recommendation of the simulated reviewer: minor/major revision.**

| # | Objection | Severity | Action | Status |
|---|---|---|---|---|
| C1 | "No clinical or field validation: the 'application' is benchmark data with added faults." | Major | Stated plainly in the conclusion and limitations. The next step named is a study with expert-labelled real faulty recordings. No clinical claims are made. | Limitation (cannot be fixed with available data) |
| C2 | "The 30 % fault prevalence in the triage studies is arbitrary." | Moderate | New post hoc check (E19, D22): at 10 % and 20 % prevalence LOOM still catches about twice as many faulty recordings as PreVal (10 %: 55 % vs 27 %, 38/38 datasets) and has fewer automatic errors (33/38). Added to Sect. 6.5. | Fixed |
| C3 | "NSTDB noise sampled at 360 Hz is added to UCR beats of unknown sampling rate." | Moderate | Disclosed in the registration (D17) and now in limitation 1. | Disclosed |
| C4 | "The software tool is claimed but not validated." | Moderate | New validation (E18): the CLI run on plain CSV files built from the raw UCR text files gives predictions identical to the pipeline on 5/5 datasets (confidence difference ≤ 1.1e-16). It has its own unit tests (API, CLI round-trip). | Fixed |
| C5 | "Data licences are not stated." | Minor | Data availability now gives the NSTDB licence (ODC-By 1.0) and the UCR citation requirement. | Fixed |
| C6 | "AI involvement is extensive. Is it disclosed as the publisher requires?" | Moderate | Disclosed in the Methods (Sect. 5.1, the journal's requirement) and in the Acknowledgements. Authors must review and approve it. | Disclosed; author action |
| C7 | "Code availability is a placeholder." | Moderate | A repository link must be inserted by the authors (anonymised for review). One script regenerates all results, tables, figures and macros; the quick mode was tested. | Author action |
| C8 | "Is the improvement visible to a reader? Show examples, including failures." | Minor | Fig. 9 shows four recordings chosen by a fixed rule, including two failures. | Addressed |

**Verdict after actions.** Reproducibility is strong. The main residual risk is the lack of real
field data, which the paper states openly.
