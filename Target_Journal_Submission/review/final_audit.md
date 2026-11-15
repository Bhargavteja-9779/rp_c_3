# Final audit (ESWA version, third revision, 2026-11-15)

Manuscript: *Can a fast time series classifier tell when a sensor recording is faulty? Leverage-aware
confidence and its limits*. Target: Expert Systems with Applications (`elsarticle`). Package:
`submission/LOOM_Submission.zip` (11 files). This revision adds E23 (CinC 2011, real recordings graded
unusable, H11/H12 registered at 3de341c before any CinC result) and E24 (1-D ResNet, deep ensemble,
MC dropout; registered before running), benchmarked against recent ESWA papers
(`journal_selection/eswa_benchmark.md`).

## Build and layout checks (`manuscript/latex/build_report.txt`)
| Check | Main | Supplement |
|---|---|---|
| Pages (12-pt preprint layout) | 48 | 10 |
| Figures / tables in main text | 13 / 11 | -- |
| Overfull boxes | 0 (plus the known elsarticle title-page footer box) | 0 |
| Undefined references / citations / multiply defined labels | 0 / 0 / 0 | 0 / 0 / 0 |
| Macro names or "??" in PDF text | 0 | 0 |
| BibTeX warnings | 11, all "empty pages" for article-number or proceedings entries | -- |
| Standalone LaTeX zip compiles from fresh extraction | yes (48 pp., 0 undefined refs) | yes (10 pp.) |
| Word version: tables / macro leaks | 11 / 0 | -- |
| Abstract | 239 words | -- |
| Highlights | 5 bullets, at most 77 characters each (ESWA limit 85) | -- |

## Reproducibility
* Unit tests: 17/17 pass.
* Every number comes from `code/tables/macros.tex` (1051 macros, 20 tables) written by
  `code/experiments/aggregate.py`; E23 and E24 outputs are committed (`results/e23/s0-s4.npz`,
  `results/e24/`), including the five trained networks.
* E23 ran five seeds (log `results/logs/e23_run.log`); E24 five networks of 134,725 parameters each.

## Integrity
* 15 registered hypotheses: 11 pass; H1b, H8, H9 and H10 fail and remain reported as failures in the
  abstract, highlights, results and conclusion.
* H11 and H12 pass on the registered seed 0 (AUROC 0.863 vs 0.344; 20 % queue 59.1 % vs 13.3 %,
  McNemar p < 0.001) and on each of the five seeds (AUROC difference 0.380-0.519).
* Disclosed caveats: CinC was added after H8/H9 failed (stated in the methods); a hand-made amplitude
  rule matches LOOM over all CinC records; the leverage alone beats LOOM; the < 10 mV subset analysis and
  the rules are post hoc (D29). Deep ensemble is the most accurate PTB-XL model.
* 29 deviations logged (`research_design/deviations.md`). 78 cited references; the ESWA papers are cited
  only for what their titles (and one abstract) say.
* AI use is disclosed. All authors must read and approve everything.

## Risk ratings (honest judgement; no probability of acceptance is claimed)
| Area | Risk | Reason |
|---|---|---|
| Novelty | **Medium-high** | Method unchanged since the desk rejection; the new evidence is a positive registered real-data result and a deep-learning comparison. |
| Data-dredging perception | Medium | CinC was chosen after the PTB-XL failure; disclosed, registered before results, PTB-XL failures kept prominent. |
| Simple baselines | Medium | Amplitude rule matches LOOM overall; leverage alone is better. Reported in text, table and figure. |
| Deep-learning comparison | Low-medium | Small CPU-trained ResNet; registered in advance and stated as a limitation. |
| Statistics | Low-medium | Registered tests, exact McNemar, paired bootstrap, all seeds reported; post hoc parts labelled. |
| Length | Medium | 48 pages in the preprint layout; ESWA's current limits must be checked by the authors. |
| Journal fit | Low-medium | Confidence-triggered review in an applied ECG and sensor setting matches recent ESWA topics. |

## Readiness decision
**READY** as far as content, analysis, layout and package are concerned. **NOT READY to submit until the
authors complete the author-only items:** names and affiliations, CRediT statement, funding, competing
interests, repository link, generative-AI declaration, ethics position, a check of ESWA's current guide
for authors (length, highlights, abstract), and a full read-through and approval by all authors.
Acceptance cannot be promised.

---
*Previous audit (ESWA version, second revision), kept for the record:*

# Final audit (ESWA version, 2026-11-15)

Manuscript: *Can a fast time series classifier tell when a sensor recording is faulty? Leverage-aware
confidence and its limits*. Target: Expert Systems with Applications (Elsevier, `elsarticle`), full-length
article. Package: `submission/LOOM_Submission.zip` (11 files). Previous submission: JKSU-CIS, desk-rejected
for limited novelty; this version adds three pre-registered studies (E20-E22) and reports their negative
outcomes.

## Build and layout checks (`manuscript/latex/build_report.txt`)
| Check | Main | Supplement |
|---|---|---|
| Pages (12-pt preprint layout) | 40 | 10 |
| Overfull boxes | 0 (plus 1 from the elsarticle title-page footer, which also appears in an empty elsarticle document) | 0 |
| Undefined references / citations / multiply defined labels | 0 / 0 / 0 | 0 / 0 / 0 |
| Macro names or "??" in PDF text | 0 | 0 |
| BibTeX warnings | 11, all "empty pages" for conference papers or article-number journals that have no page numbers | -- |
| Standalone LaTeX zip compiles from fresh extraction | yes (40 pp., 0 undefined refs) | yes (10 pp.) |
| Word version: tables / macro leaks | 10 / 0 | -- |
| Abstract | 249 words (limit 250, to be confirmed in the guide for authors) | -- |
| Highlights | 5 bullets, each at most 85 characters | -- |

## Reproducibility
* One command: `python run_all.py --mode full`; the new stages are `ptbxl` (E20), `hybrid` (E21) and
  `posthoc` (E22). PTB-XL is downloaded by `code/scripts/download_ptbxl.sh` and every record is checked
  against the checksums in its header.
* Unit tests: 17/17 pass, including a test that the chunked QUANT transform equals aeon's QUANTClassifier.
* E21 regenerated the E12 and E13 outputs from scratch; they equal the stored ones (0 difference over 830 rows).
* Every number in the paper, supplement, cover letter and tables comes from `code/tables/macros.tex`
  (890 macros, 19 tables), written by `code/experiments/aggregate.py`. Minus signs are now typeset as such.
* Every figure has its own script in `code/visualization/`; the PTB-XL example records are chosen by fixed rules.
* E20: 10 runs (primary and secondary analysis, seeds 0-4) with logs in `code/results/e20/run.log`;
  the first attempt ran out of memory and its log is kept (`results/logs/e20_run_oom_attempt1.log`, D26).

## Integrity
* H1-H7 were registered before their results; H8-H10 were registered at commit 6b05cb1 (2026-11-14),
  before E20 and E21 were run. Outcomes: 9 of 13 hypotheses pass; H1b, H8, H9 and H10 fail and are
  reported as failures in the abstract, results, discussion and conclusion.
* H8 fails on the registered seed 0 (p = 0.12). In the primary analysis LOOM caught 1.6-2.5 points
  more noisy records in all five seeds (p < 0.05 in three, reported as descriptive only). In the
  secondary analysis (training on all records) the difference ranged from -1.0 to +1.9 points with no seed
  significant. The paper says LOOM "helped at most marginally" and does not claim H8.
* H9 fails in the opposite direction in every seed of both analyses.
* 27 deviations are logged (`research_design/deviations.md`, Supplementary Sect. S2); E22 is post hoc
  (D27) and labelled so wherever it is used.
* Citations: 69 references, each checked against Crossref, an official proceedings page, arXiv or PhysioNet
  (`manuscript/refs/`, `refs/revision2_verified.json`). The literature search for the new related-work
  subsection was actually run (2026-11-14) and the novelty claim narrowed accordingly.
* Wording checked against the data source: the PTB-XL noise annotations are by "a technical expert"
  (PhysioNet), not cardiologists; annotations name a median of five of 12 leads.
* AI use is disclosed in the generative-AI declaration. The authors must read and approve everything.

## Risk ratings (honest judgement; no probability of acceptance is claimed)
| Area | Risk | Reason |
|---|---|---|
| Novelty | **High** | Same method as the desk-rejected version; the new content is mainly a careful negative result on real annotated noise. |
| Negative evidence | **High** | The main real-world test fails; some reviewers will read this as a reason to reject rather than a contribution. |
| Data realism | Medium | Now includes expert-annotated clinical noise (PTB-XL), but annotation quality is uncertain and only one such data set is used. |
| Statistics | Low-medium | Pre-registration, exact tests, Holm, bootstrap and patient-cluster bootstrap; post hoc parts are labelled. |
| Length | Medium | 40 pages, 12 figures, 10 tables in the 12-pt preprint layout; ESWA's limits could not be retrieved. |
| Journal fit | Medium | Decision-support and human-in-the-loop framing fits ESWA's scope; the methods content may be judged specialised. |
| Ethics | Low | Public, de-identified data; authors must confirm whether their institution needs an ethics statement. |

## Readiness decision
**READY** as far as content, analysis, layout and package are concerned. **NOT READY to submit until the
authors complete the author-only items:** names and affiliations, CRediT statement, funding, competing
interests, repository link, the generative-AI declaration, confirmation of the ethics position, a check
of ESWA's current guide for authors (abstract, highlights, length, anonymity), a full read-through and
approval by all authors. Acceptance cannot be promised.

---
*Previous audit (JKSU-CIS submission), kept for the record:*

## Final audit (LaTeX version, 2026-11-04)

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
