# Final audit (LaTeX version, 2026-11-04)

Manuscript: *Knowing when to ask a human: fault-aware confidence for fast time series classifiers*.
Target: JKSU-CIS (Springer Nature, sn-jnl template), regular article. Package:
`submission/LOOM_Submission.zip`.

## Build and layout checks (`manuscript/latex/build_report.txt`)
| Check | Main | Supplement |
|---|---|---|
| Pages | 25 | 9 |
| Overfull boxes | 0 | 0 |
| Undefined references / citations | 0 / 0 | 0 / 0 |
| Macro names or "??" in PDF text | 0 | 0 |
| BibTeX warnings | 0 | -- |
| Standalone LaTeX zip compiles from fresh extraction | yes (25 pp.) | yes (9 pp.) |
| Word version: tables / figures / numbered captions / macro leaks | 8 / 9 / yes / 0 | -- |

JKSU-CIS publishes no strict page limit that we could verify, and as a fully open-access journal it
charges an APC rather than page charges (GBP 1590 / USD 2090 / EUR 1790). At 25 pages the paper is
long; the authors may move Table 1 or the qualitative figure to the supplement if the editor asks.

## Reproducibility
* One command: `python run_all.py --mode full` (stages audit … figures). Quick mode passed earlier
  (24 jobs, 0 failures).
* Unit tests: 16/16 pass (core identities, Bayesian test, API, CLI).
* Every number in the paper, cover letter and tables comes from `code/tables/macros.tex` (657 macros),
  written by `code/experiments/aggregate.py` from `code/results/eNN/` and the raw result files.
* Every figure has its own script in `code/visualization/`; examples chosen by fixed rules.
* The tool was validated on plain CSV files outside the pipeline (E18: identical predictions on 5/5 datasets).

## Integrity
* Hypotheses H1–H3 were registered on 2026-10-13 and H4–H7 on 2026-11-02, each before the
  corresponding results (git commits in `research_design/experimental_protocol.md`). Numeric criteria
  for H1–H3 were formalised afterwards (D18).
* 22 deviations are logged (`research_design/deviations.md`, Supplementary Sect. S2), and all post hoc
  elements are labelled in the paper.
* Negative and mixed results appear in the abstract, results and limitations: H1b fails; the
  Bayesian analysis favours PreVal on clean data; QUANT is better under the strongest faults; LOOM
  gains little under a natural change of sensor axis; coverage varies across datasets more than chance;
  leverage exchangeability holds only approximately.
* Citations: 61 references verified against Crossref, official proceedings pages or PhysioNet
  (`manuscript/refs/`). One truncated title (MiniRocket) was corrected.
* The AI disclosure is in the Methods and Acknowledgements. The authors must approve.

## Risk ratings (honest judgement; no probability of acceptance is claimed)
| Area | Risk | Reason |
|---|---|---|
| Novelty | **Medium–high** | Builds directly on PreVal; ingredients are classical. The contribution is the combination, the LOO-fitted calibration and the triage evaluation. |
| Scale | Low | 114 + 7 + 12 + 38 + 110 dataset evaluations, five resamples, 18 fault conditions, real noise. |
| Data realism | **Medium–high** | Faults simulated except NSTDB ECG noise, which is real noise added to clean beats; no field data with expert-labelled faults. |
| Statistics | Low–medium | Pre-registered hypotheses, Holm, Bayesian tests, bootstrap CIs, sanity checks. Residual: post hoc criteria for H1–H3, approximate exchangeability. |
| Competitive position | Medium | QUANT is as good under faults on average and better at severity 3; LOOM's advantages are detection, triage and cost. |
| Writing / presentation | Low–medium | Plain style, all numbers macro-driven; the paper is long (25 pp.). |
| Ethics | Low | Public de-identified data; AI use disclosed; no clinical claims. |
| Journal fit | Low–medium | ML for sensor data is within JKSU-CIS scope; the methods-heavy content may be judged specialised. |
| Timeline | Medium | Verified median submission-to-acceptance of 69 days; about 55 % of 2026 papers were accepted within 72 days. |

## Readiness decision
**READY** for submission as far as content, analysis, layout and package are concerned, **but NOT
READY to submit until the authors complete the author-only items**: author details, declarations,
repository link, a full read-through and approval by all authors (listed in the package README).
Acceptance cannot be promised.
