# Selected target journal

| Field | Value | Source / how verified |
|---|---|---|
| **TARGET JOURNAL** | *Journal of King Saud University – Computer and Information Sciences* (JKSU-CIS) | — |
| **PUBLISHER** | Springer Nature (on behalf of King Saud University), London | Clarivate Master Journal List entry (`evidence/MJL_JKSU.txt`, retrieved 2026-10-12) |
| **ISSN / eISSN** | 1319-1578 / 2213-1248 | Master Journal List |
| **JCR STATUS** | Indexed in Science Citation Index Expanded (SCIE); also Current Contents Engineering, Computing & Technology and Essential Science Indicators; not marked "on hold" | Master Journal List |
| **JOURNAL IMPACT FACTOR** | 6.4 (2025 JCR release); 5-year JIF 6.0 | Springer journal home page (`evidence/springer_homepages/JKSU.txt`) |
| **JCR QUARTILE** | Q1 (reported by secondary aggregator journalmetrics.org for the JCR release) | **Not verified from the JCR database itself** (subscription required). With JIF 6.4, Q1 status in *Computer Science, Information Systems* is very likely but the primary JCR record could not be inspected. |
| **JCR SUBJECT CATEGORY** | Computer Science, Information Systems (per secondary sources) | Not verified in JCR directly |
| **VERIFIED REVIEW / DECISION TIMELINE** | The journal home page publishes no "first decision" figure. | Springer home page |
| **VERIFIED ACCEPTANCE TIMELINE** | **Median submission→acceptance = 69 days** (IQR 53–96), computed from the `received` and `accepted` dates that Springer deposits in Crossref for **863 research articles published online in 2026**; 54.6 % accepted within 72 days. The monthly median has been 65–72 days for every month since February 2026 (it was 83–90 days in July 2025–January 2026). | Own measurement, `evidence/scripts/accept_time.py`, log `evidence/crossref_measurements.log` |
| **APC** | GBP 1590 / USD 2090 / EUR 1790 (fully open access, CC BY or CC BY-NC-ND); waivers for low-income countries | `evidence/JKSU_howto.txt` |
| **SCOPE FIT** | Aims & scope explicitly list Artificial Intelligence (Machine Learning, Data Mining), Data Analysis, and Signal Processing / Pattern Recognition | `evidence/JKSU_scope.txt` |
| **RESEARCH FIT** | Among 1,190 articles published since July 2025, titles include machine learning on time series, uncertainty, calibration, explainability, anomaly detection, sensor/medical signals. Methodological ML papers validated on public benchmarks are typical. | Crossref title scan |

## Interpretation of the acceptance timeline (important)
* The 69-day figure is a **median**; roughly half of the papers took longer than 72 days.
  Acceptance within 72 days therefore **cannot be guaranteed** for any individual paper.
* It is a *submission (received) → acceptance* interval, not first decision and not
  publication time. Acceptance → online publication is additionally ~2–3 weeks
  (not used for selection).

## Key author guidelines (from `evidence/JKSU_guidelines.txt`)
* Double-anonymous peer review: anonymised manuscript + separate title page.
* Abstract 150–250 words, no undefined abbreviations; 4–6 keywords.
* Word (.docx) preferred (LaTeX allowed); 10-pt Times-like font; automatic page numbering;
  decimal headings, at most three levels; equations with the equation editor.
* Springer *basic* author–year citations, e.g. "(Thompson 1990)", multiple citations
  alphabetical then chronological; reference list alphabetical with full DOI links.
* Mandatory *Statements and Declarations* after the references (Funding, Competing
  Interests, Author Contributions; plus Data Availability).
* Data availability statement mandatory; code/data in repositories encouraged.
* Figure captions "Fig. 1"; figures cited in order; tables with captions, Arabic numerals.
* Use of LLMs must be documented in the Methods section (copy editing need not be).

## Key rejection risks for this journal
1. Contribution judged incremental relative to the authors' closest prior art
   (PreVal, Dempster et al. 2025 arXiv 2401.15610).
2. English quality / technical screening before review.
3. Insufficient statistical support (single split, no significance tests).
4. Similarity-check (iThenticate) overlap with prior work.
