# Review round 10 — Senior reviewer (overall assessment of the revised, application-oriented version)

*Perspective: a senior reviewer at an applied computer science journal reads the paper as a
whole. Is the problem real, is the contribution useful beyond "a benchmark number went down", is
the paper readable for a non-specialist, and are the claims honest?*
Reviewed: manuscript rebuilt on 2026-10-29 after the reframing (new title, rewritten text,
triage case study).

## What changed in this round (author request)
1. **Title** simplified: *"Knowing when to ask a human: reliable confidence for fast time series
   classifiers under sensor faults"* (previously a technical title naming leverage moderation and
   leave-one-out calibration).
2. **Problem framing**: the paper now opens with concrete monitoring settings (wearable heart
   monitor, factory sensor, gesture recognition) and the human-in-the-loop triage problem.
   "Corruptions" are described as the sensor faults they imitate (noise, drift, electrode pops,
   dropouts, speed variation, misalignment).
3. **New application study** (§5.4, §6.1, Fig. 3): triage on 12 sensor datasets from three
   domains with 30 % faulty recordings. Its protocol was written into Online Resource 2 §10 and
   committed (git 3d2c936) **before** the study was run; it is labelled exploratory because it was
   designed after the benchmark results were known.
4. **Writing**: prose rewritten in a plainer voice. Long inline statistic chains were replaced by
   "better on 108 of 114 datasets (p < 0.001)", with the full statistics in the tables; bulleted
   research questions were turned into a short paragraph; fewer bold run-in labels.

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R10-1 | MAJOR | The first drafts of the new case-study text claimed that "fewer errors slipped through automatically" and that the review queue caught "far more" faulty recordings "than the alternatives". The results do not support either claim in full. LOOM's lower median automatic error rate vs PreVal (4.2 % vs 9.5 %) is not significant (9/12 datasets, p = 0.056). QUANT has the lowest automatic error rate (1.6 %). LOOM's advantage over QUANT in faults caught (45 % vs 38 %) is not significant (8/12, p = 0.052). | **Fixed before release.** §6.1 now leads with what is significant: faults caught vs PreVal on 12 of 12 datasets (p = 0.002), and vs hold-out TS and Platt. It reports the automatic-error results as mixed, states QUANT's advantage and explains it (QUANT classifies damaged recordings more accurately). The discussion, conclusion, abstract, highlights and cover letter use the same wording ("nearly twice as many as PreVal", "somewhat more than QUANT"). |
| R10-2 | MAJOR | The abstract exceeded the 250-word limit after the reframing (273 words). | Shortened to 247 words. |
| R10-3 | MODERATE | The case study uses simulated faults, so the application claim could be read as stronger than it is. | Limitations now have a paragraph on the application study (12 datasets, simulated faults, designed post hoc but pre-specified, low power) and name a study with expert-labelled real faults as the next step. The only real change of sensor (cross-axis) is reported as showing a small gain. |
| R10-4 | MODERATE | With final extractor results, LOOM is significantly *worse* than PreVal on clean MultiRocket features (72 of 110 datasets, p = 0.008). | Reported as generated (§6.6); consistent with the clean-data cost of smoothing discussed in §6.9. |
| R10-5 | MINOR | Phrasing artefacts from generated statistics: nested parentheses, "on 12 of 12 datasets in favour of LOOM", the old fault names ("time warping", "corruption") in generated sentences, and "better than any other method" (LOOM's own ensemble and a variant have slightly higher detection AUROC). | **Fixed.** Nested parentheses are flattened automatically, the count phrase is rewritten, fault names are unified (`FAMNAMES`), and the claim now reads "better than any of the competing methods". |
| R10-6 | MINOR | The Limitations mentioned that the experiments ran in a cloud container that restarted, a process detail of no scientific relevance. | Removed from the manuscript (kept in the pre-registration amendments, where it explains the compute decisions). |
| R10-7 | — | **AI-use disclosure.** The author asked for the paper to read as more human-written. | Done for the *style* only. The disclosure of AI assistance in §5.6 and in the declarations is **kept unchanged**. JKSU-CIS requires it, and removing it would misrepresent how the work was produced. |

## Overall assessment
The revised paper reads as an applied contribution: a real operational problem (deciding which
sensor recordings a human must check), a cheap closed-form fix, a large controlled benchmark,
and an application study whose numbers an operator can interpret. The novelty relative to PreVal
remains moderate. The strongest practical evidence is the doubling of faulty recordings caught
at a fixed review budget, while QUANT stays a strong competitor and is reported as such.
**Recommendation (simulated): minor revision.** The remaining risks are recorded in
`REJECTION_RISK_REPORT.md`.
