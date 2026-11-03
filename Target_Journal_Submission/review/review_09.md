# Review round 9 — Integrity reviewer (claim–evidence audit, citation audit)

*Perspective: is every claim backed by a result, and does every reference exist with correct
bibliographic data?*

## Claim–evidence audit
All 22 claims in the title, abstract, highlights, contributions, discussion and conclusion are
mapped to their evidence in `supplementary/claim_evidence_audit.md`. Results:
* 2 claims were **corrected** in earlier rounds after this audit flagged them: the ablation
  directions and the moderation-strength statement (review 6), and the QUANT comparison (review 8).
* 1 conditional claim ("better than every … raw-series classifier") is generated only if all
  pairwise tests support it. They do not, so it is not printed.
* 2 claims (other extractors, train size) depend on jobs still running at the time of the audit.
  They are generated from the results and must be re-read after the final build (tracked in the
  final audit).
* 1 claim (the Arsenal's zero probabilities) is an inference. It is worded as "largely caused by",
  supported by the NLL/Brier divergence and by how the Arsenal forms its probabilities.

## Citation audit
| Check | Result |
|---|---|
| Every in-text citation key exists in `references.json` | ✔ (58 references) |
| Every reference is cited | ✘ → ✔: `hoerl1970` was uncited; it is now cited for ridge regression in §4.1. |
| Every reference verified against an authoritative source | 36 by Crossref DOI metadata (`refs/crossref_verified.json`), 18 by official landing pages (`refs/landing_pages_verified.json`, alias keys documented). The remaining 4 non-DOI items (Burnaev & Vovk 2014; Nouretdinov et al. 2001; Holm 1979; Platt 1999/2000) were verified in this round (`refs/manual_verified.json`): PMLR page, ML Anthology/Royal Holloway record, JSTOR/MR records, MIT Press/CiteSeerX records. |
| Fabricated or unverifiable references | None found. |
| Year conventions | Platt's chapter is listed as 2000 (MIT Press volume), although it is often cited as 1999. The key name `platt1999` is internal only. |
| Self-citation / citation padding | Not applicable (no author self-citations); all references are cited for a specific statement. |

## Findings
| # | Severity | Issue | Resolution |
|---|---|---|---|
| R9-1 | MODERATE | Four references had no recorded verification. | Verified and recorded (see above). |
| R9-2 | MINOR | Uncited reference `hoerl1970`. | Cited in §4.1. |
| R9-3 | MODERATE | Two generated claims depend on unfinished jobs. | Re-check after the final run (final audit checklist). |

## Verdict
No unsupported or fabricated claim or reference remains in the current build.
