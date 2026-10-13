# Verification log for the journal selection

## 1. Legitimacy and indexing
* Clarivate **Master Journal List** (retrieved 2026-10-12, headless browser; text saved in
  `evidence/MJL_JKSU.txt`): "JOURNAL OF KING SAUD UNIVERSITY COMPUTER AND INFORMATION
  SCIENCES — Publisher SPRINGERNATURE … ISSN/eISSN 1319-1578 / 2213-1248 — Web of Science
  Core Collection: Science Citation Index Expanded — Additional indexes: Current Contents
  Engineering, Computing & Technology | Essential Science Indicators." No "on hold" flag.
* Publisher: Springer Nature (COPE member, established publisher). The journal moved
  from Elsevier to Springer Nature in 2025; the ISSNs are unchanged.
* A web search for retractions, delisting or paper-mill concerns specific to this
  journal returned no such report (search performed 2026-10-12). This is absence of
  evidence, not proof.

## 2. Impact factor and quartile
* JIF 6.4 (2025) and 5-year JIF 6.0 are shown on the publisher's journal page
  (`evidence/springer_homepages/JKSU.txt`).
* Q1: reported by a secondary aggregator; **the JCR database itself was not accessible**
  (subscription), so quartile and category are recorded as *secondary-source verified*.

## 3. Acceptance timeline — method
`evidence/scripts/accept_time.py` queries the Crossref REST API for all journal
articles of an ISSN with an online publication date ≥ 2026-01-01 and parses the
`received` and `accepted` assertions deposited by the publisher. The median, quartiles
and share ≤ 72 days are reported. `evidence/scripts/pubmed_accept.py` does the same with
PubMed `History` dates (received/accepted) for journals indexed in PubMed.
Measurements were run twice (2026-10-12) with identical results for JKSU-CIS
(median 69 days, n = 863).

Monthly medians for JKSU-CIS (published-online month: n, median days):
2025-07: 62, 84.5 · 2025-08: 61, 84 · 2025-09: 32, 83 · 2025-10: 49, 83 · 2025-11: 56, 89.5 ·
2025-12: 65, 88 · 2026-01: 52, 89.5 · 2026-02: 83, 74 · 2026-03: 104, 67 · 2026-04: 87, 67 ·
2026-05: 87, 72 · 2026-06: 99, 70 · 2026-07: 45, 65 · 2026-08: 225, 65 · 2026-09: 66, 67 ·
2026-10 (partial): 15, 65.

## 4. Requirements that could not be verified or satisfied
| Requirement | Status |
|---|---|
| Q1 in latest JCR | Secondary source only (JCR not accessible) |
| JCR subject category | Secondary source only |
| Acceptance < 60 days | **Not satisfied** (median 69 days) |
| Acceptance ≈ 30 days | **Not satisfied** |
| Acceptance ≤ 72 days (user's hard limit) | Satisfied **as a median**; ~45 % of papers exceed it |
| Acceptance rate | Not published; not reported here |
