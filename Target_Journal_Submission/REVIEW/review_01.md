# Review round 1 — Handling editor (desk screening, scope, journal compliance)

*Perspective: would the editor send this out for review or desk-reject it? Is it compliant with
the JKSU-CIS author guidelines (`journal_selection/evidence/JKSU_guidelines.txt`)?*
Reviewed: manuscript built 2026-10-21 from main/raw results on all 114 datasets.

## Findings

| # | Severity | Issue | Resolution |
|---|---|---|---|
| R1-1 | MAJOR | Guidelines require that use of LLMs be documented **in the Methods section**; the draft only pointed to the Statements and Declarations. | **Fixed.** §5.5 now contains a paragraph "Use of a large language model" describing what the assistant was used for and that all numbers were inserted programmatically from result files. Statement in the declarations kept, bracketed for author review. |
| R1-2 | MAJOR | The text cites Online Resources 1–3 (derivations, pre-registration with amendments and leakage audit, per-dataset results). These must exist as files at submission. | **Action in packaging:** OR1 = `supplementary/derivations.md` → PDF, OR2 = pre-registration + amendments + leakage audit table → PDF, OR3 = per-dataset results (CSV/XLSX) + supplementary tables S1–S4. Tracked in the manifest. |
| R1-3 | MODERATE | Double-anonymous review: no author names may appear in the manuscript file or its metadata. | **Checked.** DOCX core properties author/last-modified-by are empty; no names, affiliations, or repository URL in the text (repository withheld, "provided to the editor"). Title page is a separate file. |
| R1-4 | MINOR | Abstract length 150–250 words, 4–6 keywords, ≤3 heading levels, "Fig. n" captions, figures cited in order. | **Checked.** Abstract 216 words; 6 keywords; two heading levels; Figs. 1–8 cited in order; Tables 1–5 numbered. |
| R1-5 | MODERATE | Scope: the journal is broad CS/IS. A pure-methods TSC paper might be judged "too specialised". | The Introduction motivates decision-making under sensor drift (monitoring, deferral); JKSU-CIS scope lists Machine Learning, Data Mining and Signal Processing/Pattern Recognition. Cover letter states the fit explicitly. Residual risk recorded in the rejection-risk report. |
| R1-6 | MODERATE | Statements and Declarations contain bracketed placeholders (Funding, Competing interests, Author contributions). | Cannot be completed without the authors — **author action item**, listed in the final decision. |
| R1-7 | MINOR | The editor will check that the claimed novelty vs. PreVal is visible in the first page. | Abstract and Introduction state that LOO calibration is prior art (PreVal) and the contribution is leverage moderation + closed-form jackknife+ + shift benchmark. |

## Verdict after fixes
Compliant with the formal requirements except for author-only items (declarations, title page
details, ORCID). Likely to pass desk screening if the scope fit is accepted.
