# Rejection-risk report (ESWA version, third revision, 2026-11-15)

Manuscript: *"Can a fast time series classifier tell when a sensor recording is faulty? Leverage-aware confidence and
its limits"*. Target: *Expert Systems with Applications* (Elsevier). Previous outcome: desk rejection at JKSU-CIS for
limited novelty.

**No acceptance probability is asserted.** The risks below are judgements, not measured probabilities.

## Main risks, ranked

| # | Risk | Likelihood (judgement) | What was done | Residual |
|---|---|---|---|---|
| 1 | **Novelty** (again): LOOM builds directly on PreVal; ingredients are classical. | High | Contribution reframed: the method plus a pre-registered evaluation along a gradient of realism, including the first (to our knowledge) test of classifier confidence on PTB-XL's expert noise annotations, and two pre-registered negative results that define the method's boundary. Prior uses of the annotations are cited. | High. A desk editor may still judge the method incremental. |
| 2 | **Mixed real-data evidence**: on PTB-XL LOOM does not beat PreVal (H8, H9 fail), the hybrid fails (H10); on CinC 2011 it passes (H11, H12). | Medium (was high) | CinC adds a pre-registered positive result on real, strong faults from another acquisition setting; failures stay in abstract, highlights, conclusion. | Medium. CinC was added after the PTB-XL failure (disclosed). |
| 2b | **A simple amplitude rule matches LOOM overall on CinC; leverage alone is better than LOOM.** | Medium | Both reported; LOOM's advantage over the rule shown on records < 10 mV (post hoc, D29). | Medium. A reviewer may see LOOM's confidence as an indirect leverage score. |
| 2c | **Deep baselines** (asked for by ESWA-style reviewers). | Low (was high) | E24: deep ensemble and MC dropout; more accurate on PTB-XL, worse on CinC; reported both ways. Small CPU-trained network stated as a limitation. | Low–medium. |
| 3 | **Journal fit**: ESWA expects systems and applications. | Medium | Framed as a decision-support component with a validated tool; cover letter argues the fit. | Medium. |
| 4 | **Length**: 48 pages in the 12-pt preprint layout, 12 figures, 10 tables. | Medium | ESWA length rules could not be retrieved; candidates to move to the supplement are listed in review 16. | Medium; author decision. |
| 5 | **Not the best probabilistic model overall**: QUANT better under the strongest faults; the deep ensemble most accurate on PTB-XL. | Medium | Reported throughout. | Medium. |
| 6 | **Post hoc elements** (exploratory triage study, E19, E22). | Low–medium | Labelled; deviations D1–D27 logged with timing. | Low. |
| 7 | **AI-assisted work** declared. | Low–medium | Elsevier-style declaration in back matter and title page; Methods description. Authors must review and own the content. | Depends on editor policy. |
| 8 | **Annotation quality of PTB-XL**: one expert, possibly incomplete labels. | Low–medium | Stated as a limitation with its likely effect. | Low. |
| 9 | **Formatting against the unverified guide for authors.** | Low | elsarticle template, highlights, declarations; README lists items to confirm. | Low once checked. |

## What would most improve the odds
1. A method change that works on PTB-XL, confirmed on a fresh split or dataset (the user chose not to pursue this now).
1b. A second real-fault dataset with strong artefacts from another sensor type, registered in advance.
2. Shortening the main text by moving secondary material to the supplement.
3. Real faults from a second sensor type (e.g., wearable accelerometers with annotated artefacts).
