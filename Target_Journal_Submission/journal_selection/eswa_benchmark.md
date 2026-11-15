# Recent ESWA papers close to this manuscript (benchmark for the third revision, 2026-11-15)

Purpose: the user asked to compare the paper with recently accepted *Expert Systems with Applications* papers and
raise it to that level. Method: Crossref API, journal ISSN 0957-4174, articles published from 2025-01-01, queried
with 12 topic phrases (query script and raw output:
`evidence/q.py`, `evidence/crossref_eswa.json`). Abstracts could not be read: Elsevier does not deposit them in Crossref, OpenAlex or Semantic Scholar for
these papers, and ScienceDirect refuses automated access (HTTP 403). **Only titles, authors, volumes and article
numbers are verified.** Statements about what these papers do are limited to what their titles say, except for
Şensoy et al. (2025), whose abstract is on OpenAlex.

| DOI | Year, vol. | Title (abridged) | Relevance |
|---|---|---|---|
| 10.1016/j.eswa.2025.128662 | 2025, 293 | Engine fault diagnosis framework based on machine learning and MiniRocket feature extraction | MiniRocket features in an ESWA fault-diagnosis application |
| 10.1016/j.eswa.2025.128693 | 2025, 293 | Kernel grouping for time series classification with multiple transformations and pooling operators | ROCKET-family classifier development in ESWA |
| 10.1016/j.eswa.2024.125906 | 2025, 265 | Risk-aware classification via uncertainty quantification (Şensoy et al.) | Evidential deep learning that lets a classifier exercise discretion under uncertainty (abstract read) |
| 10.1016/j.eswa.2025.126533 | 2025, 269 | Multi-sensor bearing fault diagnosis based on evidential neural network with sensor weights and reliability | Sensor reliability in fault diagnosis |
| 10.1016/j.eswa.2026.133628 | 2027, 332 | MMAP-Net: … retinal disease classification with score-calibrated deferral | Confidence-based deferral in a medical application |
| 10.1016/j.eswa.2026.134403 | 2027, 334 | CCA-ID: Confidence-calibrated adaptive intrusion detection with selective routing for IoT networks | Calibrated confidence used to route inputs |
| 10.1016/j.eswa.2026.132601 | 2026, 326 | Multi-task learning with signal denoising … for wearable ECG arrhythmia detection | Handling noise in wearable ECG |
| 10.1016/j.eswa.2024.125861 | 2025, 264 | LLM-TSFD: industrial time series human-in-the-loop fault diagnosis | Human-in-the-loop fault diagnosis on time series |

## What these titles say about the level expected (judgement)
1. Applied framing: each paper names an application (engine, bearing, retinal disease, IoT intrusion, wearable ECG).
2. Real data: the applications are evaluated on real recordings, not only on simulated corruption.
3. Confidence that triggers an action (deferral, routing, discretion) is an accepted topic in ESWA in 2025–2027.
4. Deep-learning methods are the norm among these papers; a paper on a linear readout will be asked how it compares
   with deep uncertainty methods (deep ensembles, MC dropout).

## Gap analysis of our manuscript against these points
| Point | Before this revision | Action in this revision |
|---|---|---|
| Real data with strong, expert-judged artefacts | Only PTB-XL, whose noise is mild; the main real-world test failed (H8, H9) | E23: CinC 2011, records graded unacceptable by 3–18 annotators, registered as H11/H12 before running (3de341c) |
| Comparison with deep uncertainty methods | None (only linear readouts, QUANT, Arsenal) | E24: 1-D ResNet, deep ensemble and MC dropout on both ECG studies, registered before running |
| Positioning against ESWA work | No ESWA citations | Cite the papers above where their titles support the statement |
| Applied framing | Present | Keep; frame around a deployable triage step |

Limitation of this benchmark: it cannot show what reviewers of these papers asked for or how the papers were
judged; it is not evidence about the chance of acceptance.

## Outcome of the actions (2026-11-15)
| Point | Result |
|---|---|
| Real data with strong, expert-judged artefacts | E23: H11 and H12 pass on the registered seed (AUROC 0.863 vs 0.344; queue 59.1 % vs 13.3 %). Caveats reported: an amplitude rule matches LOOM over all records; leverage alone is better than LOOM. |
| Comparison with deep uncertainty methods | E24: deep ensemble most accurate on PTB-XL (78.5 %) but near-chance noise detection like every method; on CinC single network and MC dropout below chance, ensemble AUROC 0.693 vs LOOM 0.863. |
| Positioning against ESWA work | Eight ESWA papers cited in the related work. |
