# Final audit and submission decision

Date: 2026-10-29. Manuscript: *"Knowing when to ask a human: reliable confidence for fast time
series classifiers under sensor faults"*. Primary target: JKSU-CIS (Springer Nature). Alternative
formatting: ESWA (Elsevier; exceeds the 72-day limit, see `journal_selection/eswa_alternative.md`).

## 1. Experiment completeness
* Benchmark: all **1,564** planned jobs finished. That is the main stage for 114 datasets (5 resamples;
  3 for FordA/FordB/ElectricDevices, 5 for Crop), QUANT/Arsenal, MultiRocket/Hydra, and 25 %/50 %
  training data. A script check found no missing result file. The "FAILED" lines in
  `results/logs/run_all_full.log` are from earlier runs killed by the out-of-memory killer; those
  jobs were rerun successfully by the self-healing scheduler.
* Natural cross-axis shift: 6 tasks × 5 resamples (`results/natural_shift.json`).
* Application case study: 12 datasets × 5 resamples, protocol committed before running (git
  3d2c936; Online Resource 2 §10).
* Known coverage gaps, all disclosed: LogReg-CV not run on 4 datasets with > 2,000 training series.
  The CV calibrators cannot be fitted on datasets with fewer series per class than folds (110
  datasets). The Arsenal returned NaN probabilities on Fungi (one training series per class),
  giving 113 datasets.

## 2. Automated consistency checks (`manuscript/build/final_checks.py`)
24/24 passed: no unresolved expressions; abstract 247 words (≤ 250); 6 keywords; 5 highlights
≤ 85 characters; figures 1–9 and tables 1–5 cited in order with captions; every cross-referenced
section exists; every citation is in the bibliography and every reference is cited; all output
files exist; anonymous DOCX metadata and no identifying strings; all experiments present in
`statistics.json`; 12/12 unit tests pass.

## 3. Consistency of numbers
Every number in the manuscript, cover letter and title page is generated at build time from
`results/statistics.json`, `tables/*.csv`, `results/natural_shift.json`,
`results/case_study.json` and `results/check_logreg_convergence.json`. A strict build aborts on
any missing value. Data-dependent wording (direction words, "significant", the QUANT comparison,
trend and train-size conclusions, worst-case datasets) is computed, not typed. The claim–evidence
audit (`supplementary/claim_evidence_audit.md`, 26 claims) was updated after the final run.

## 4. Reproducibility
`python run_all.py --mode quick` passed end to end (24 jobs, 0 failures, about 6.5 min).
`python run_all.py --mode full` contains every stage used in the paper (audit, main, raw, extract,
trainsize, natural, case, logreg, analyze). Exact package versions are pinned. Cached per-job
results are shipped as `code/results/raw_jobs.tar.gz`, and all per-condition metrics are in
`code/results/raw_results.csv`, so the analysis can be rerun without recomputation.

## 5. Review rounds
Ten review rounds (`REVIEW/review_01.md` … `review_10.md`) led to these substantive changes,
among others:
* correction of reversed ablation statements;
* a like-for-like severity-3 comparison that turned QUANT into a significant winner at the
  strongest faults, now reported throughout;
* timing accounting fixes;
* figure fixes;
* citation verification;
* the application-oriented reframing with a pre-specified triage study, whose wording was
  corrected to match its partly non-significant results.

## 6. Decision

**READY** with respect to scientific content, analysis, manuscript formatting and the code and
data package, **conditional on the following author-only items**. Until they are done the
submission itself is **NOT READY**, because no one but the authors can provide them:
1. author names, affiliations, ORCIDs and corresponding-author details (title page, cover letter);
2. funding, competing interests and author contributions (Statements and Declarations);
3. review of the generative-AI statements (§5.6 and declarations) and of the full text, since the
   authors must take responsibility for the content;
4. a repository link for the code package (anonymised for review);
5. for ESWA only: CRediT statement and the Elsevier generative-AI declaration section.

Acceptance, and acceptance within 72 days, cannot be guaranteed. The journal's measured median
is 69 days, and about 55 % of its 2026 papers were accepted within 72 days.
