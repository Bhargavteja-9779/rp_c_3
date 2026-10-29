# Rejection-risk report

Target: *Journal of King Saud University – Computer and Information Sciences* (Springer Nature;
JIF 6.4; verified median submission→acceptance 69 days, IQR 53–96, 54.6 % of 2026 papers accepted
within 72 days). Alternative: *Expert Systems with Applications* (JIF 9.4, publisher-reported
median 147 days, **exceeds the 72-day limit**; see `journal_selection/eswa_alternative.md`).

**No acceptance probability is asserted.** The journal does not publish an acceptance rate that we
could verify, and acceptance within 72 days cannot be guaranteed for any individual paper:
roughly half of JKSU-CIS papers take longer. The risks below are judgements, not measured
probabilities.

## Main risks, ranked

| # | Risk | Likelihood (judgement) | What was done | Residual |
|---|---|---|---|---|
| 1 | **Incremental relative to PreVal** (Dempster et al.): the LOO-calibration idea is prior art, and clean NLL does not improve. | High | PreVal is the primary baseline throughout. Contribution narrowed to leverage moderation fitted on held-out leverages, closed-form jackknife+ for classification, and the shift benchmark. Ablation isolates the effect of the leverage moderation (PreVal-smooth). Prior closed-form conformal ridge work (RRCM) cited. | Medium–high. Reviewers who care only about in-distribution performance will see no gain. |
| 2 | **Not the best probabilistic TSC model overall**: QUANT is on par on average and significantly better at severity 3; ensembles slightly better on clean data. | Medium | Reported in the abstract, results, discussion, conclusion and cover letter. Positioned as the best *readout* for random-kernel features, with an end-to-end cost advantage. | Medium. |
| 3 | **Synthetic shifts**: the corruptions are synthetic; the one natural shift shows only a small gain. | Medium–high | Natural cross-axis shift added and honestly reported; limitations and scope statement in the abstract. | Medium. A reviewer may ask for real domain-shift benchmarks (multivariate, new subjects), which the current scope cannot provide. |
| 4 | **Higher in-distribution ECE** caused by target smoothing. | Medium | Reported, explained with evidence (LOOM-nosmooth has PreVal-level ECE), and practical guidance given. | Low–medium. |
| 5 | **Limited protocol** (5 resamples, official split only for extractors/ensembles/Arsenal, LogReg skipped on 4 datasets, test subsampling on 2). | Medium | All pre-registered compute amendments are disclosed. Robustness shown on unseen, no-duplicate and source-grouped subsets. | Low–medium. |
| 6 | **Post-hoc analysis change** (severity-3 restriction in Table 3). | Low | Disclosed in §6.3 and Online Resource 2 §9; the change works *against* LOOM. | Low. |
| 7 | **AI-assisted authorship**: extensive use of an AI assistant is declared. Some editors may scrutinise this. | Low–medium | Declared in Methods (as JKSU-CIS requires) and in the declarations; all numbers machine-generated and reproducible. | Depends on editor policy; authors must review and own the content. |
| 8 | **Scope fit** for a broad CS/IS journal (desk rejection as "too specialised"). | Low–medium | Cover letter states the fit (ML, pattern recognition, data mining; monitoring applications). | Low–medium. |
| 9 | **Language / formatting screening.** | Low | Springer author–date style, decimal headings, Statements and Declarations, anonymised manuscript, separate title page, page numbers. | Low once the author placeholders are completed. |
| 10 | **Similarity check (iThenticate).** | Low | Text written from scratch; equations and standard identities cited. | Low. |

## Items the authors must complete before submission (blocking)
1. Author names, affiliations, ORCIDs and corresponding-author details (title page, cover letter).
2. Funding, competing interests and author contributions (declarations).
3. Review and, if needed, edit the generative-AI statement (Methods §5.5 and declarations). The
   authors must take responsibility for all content.
4. A public (or anonymised for review) repository link for the code package.
5. A final read-through of the manuscript by the authors.
