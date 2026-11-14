# LOOM — leverage-moderated leave-one-out calibration for random-kernel time series classifiers

Code, data instructions and machine-generated results for the manuscript
*"Can a fast time series classifier tell when a sensor recording is faulty? Leverage-aware confidence and its
limits"* (prepared for Expert Systems with Applications).

LOOM turns the ridge readout of MiniRocket/ROCKET-type classifiers into a probabilistic
classifier

    p(y | x) = softmax( a * z(x) / sqrt(h(x) / h_ref) )

where z are the ridge logits, h the leverage (posterior predictive variance factor), and the
temperature a and ridge penalty λ are fitted on the exact leave-one-out pairs
(z₋ᵢ, h₋ᵢ = dᵢ/(1 − dᵢ)). Everything comes from one thin SVD; no data are held out and nothing
is refitted. The same algebra gives exact closed-form jackknife+ prediction sets.

## 0. Quick start: triage of sensor recordings

```bash
pip install .                    # installs the `loom` package and the `loom-triage` command
loom-triage fit --train train.csv --model model.pkl                      # first column = label
loom-triage run --model model.pkl --input new.csv --budget 0.2 --alpha 0.1 \
                --out queue.csv --report report.html                     # review queue + HTML report
```

```python
from loom import LOOMClassifier
clf = LOOMClassifier().fit(X_train, y_train)      # X: (n_series, length)
P = clf.predict_proba(X_new)                      # probabilities that drop for unusual recordings
queue = clf.triage(X_new, budget=0.2)            # prediction, confidence, unusualness, auto/review
sets = clf.predict_sets(X_new, alpha=0.1)         # exact jackknife+ prediction sets
```

`examples/ecg_triage_demo.sh` runs the whole workflow on ECG5000 with real electrode-motion noise
from the MIT-BIH Noise Stress Test Database (`scripts/download_nstdb.sh`). On that stream (30 %
noisy recordings, 20 % review budget) the review queue caught 53 % of the noisy recordings and the
error rate of the automatic decisions fell from 6.1 % (all decisions) to 2.3 %.

## 1. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.13 used for the paper
pip install -r requirements.txt                          # exact versions
# or: conda env create -f environment.yml && conda activate loom
```

A CPU is sufficient. The reported results used one core per job on a 4-core Linux machine,
with no GPU. `torch` (CPU build) is needed only for aeon's Hydra transform.

## 2. Data

```bash
bash scripts/download_data.sh      # UCR 2018 univariate archive, ~550 MB zip, ~1.2 GB extracted
bash scripts/download_nstdb.sh     # MIT-BIH Noise Stress Test Database noise records (bw, em, ma)
bash scripts/download_ptbxl.sh     # PTB-XL v1.0.3: metadata + 16,307 100-Hz 12-lead records (~450 MB)
```

The UCR script writes the list of the 114 equal-length datasets without missing values
(`data/raw/ucr_equal_length.csv`; a copy is committed as `data/ucr_equal_length.csv`). Licences: UCR data
are free for research use with citation of Dau et al. (2019); NSTDB: Open Data Commons Attribution v1.0;
PTB-XL: CC BY 4.0. Raw data are not committed.

## 3. Reproduce everything

```bash
python run_all.py --mode quick      # ~10 min: 6 small datasets, 1 resample -- checks the whole pipeline
python run_all.py --mode full       # full study (several days on 4 cores; every stage resumes from its cache)
bash ../manuscript/latex/build.sh --word    # manuscript + supplement PDFs, layout checks, Word version
bash ../manuscript/latex/make_submission.sh # submission zip, with a compile test of the LaTeX source
```

Stages, in order (`--stages ...`; one script per experiment in `experiments/eNN_*.py`):

| Stage | Experiment | Output |
|---|---|---|
| `audit` | E0 leakage audit | `results/e00/` |
| `main` | E1, E3, E4, E9: MiniRocket readouts, ablations, 18 simulated faults, conformal sets | `results/raw/minirocket__*.json` |
| `raw` | E2: QUANT, Arsenal | `results/raw/raw__*.json` |
| `extract`, `trainsize` | E5 other extractors, E7 training-set size | `results/raw/` |
| `natural` | E10 cross-axis change | `results/natural_shift.json` |
| `case`, `confirm` | E11 exploratory and E13 confirmatory triage | `results/case_study*.json`, `results/case_study/*.npz` |
| `realnoise` | E12 recorded ECG noise (NSTDB) | `results/real_noise.json`, `results/real_noise/*.npz` |
| `quantfeat` | E14 readouts on QUANT features | `results/quant_features.json` |
| `logreg`, `sanity` | E16 LogReg convergence, E17 sanity checks | `results/check_logreg_convergence.json`, `results/e17/` |
| `tool`, `prevalence` | E18 tool validation, E19 prevalence sensitivity | `results/e18/`, `results/e19/` |
| `ptbxl` | E20 PTB-XL expert-annotated noise (H8, H9) | `results/e20/summary.json`, `results/e20/*.npz` |
| `hybrid` | E21 hybrid triage (H10) | `results/e21/` |
| `posthoc` | E22 post hoc explanation (D27) | `results/e22/summary.json` |
| `analyze`, `aggregate`, `figures` | statistics; `tables/macros.tex` and `tables/tab_*.tex`; `figures/fig_*.pdf` | |

`python -m pytest -q tests` checks every closed-form identity against explicit refits, the API and CLI, and
that the memory-bounded QUANT used in E20 equals aeon's QUANTClassifier.

## 4. Where each paper item comes from

Every number in the manuscript, supplement and cover letter is a LaTeX macro in `tables/macros.tex`, written by
`experiments/aggregate.py` from `results/eNN/summary.json` and the raw result files; no number is typed by hand.
Every figure has its own script `visualization/fig_*.py`. The claim-by-claim mapping to result files is in
`../review/claim_evidence_matrix.md`; the hypotheses, deviations and their timing are in `../research_design/`.

## 5. Code layout

```
src/loom/
  data/ucr.py                 loading, deterministic resampling, duplicate audit
  preprocessing/corruptions.py  6 corruption families x 3 severities (test-time only)
  models/readout.py           closed-form ridge: SVD, exact LOO logits / leverages, test & cross leverage
  models/calibration.py       temperature / moderation fitting, Platt target smoothing
  models/loom.py              LOOM (+ ablation variants), closed-form jackknife(+) sets (LAC, APS)
  models/baselines.py         ridge-onehot/softmax, CalibratedClassifierCV, logistic regression
  api.py                      LOOMClassifier: user-facing scikit-learn style estimator (raw series in)
  cli.py                      loom-triage command-line tool and HTML report
  training/runner.py          one benchmark job (cached JSON), Arsenal/QUANT job
  evaluation/metrics.py       NLL, ECE, Brier, classwise ECE, AURC, set metrics, reliability bins
  evaluation/aggregate.py     tidy tables from cached jobs
  statistics/tests.py         Wilcoxon + Holm, rank-biserial, bootstrap CI, Friedman, Nemenyi CD
  visualization/plots.py      all figures
experiments/                  one script per experiment (eNN_*.py), analysis, aggregation, development pilots
run_all.py                    single entry point (quick / full)
config/default.yaml           human-readable record of the configuration
```

## 6. Using LOOM on your own features

```python
from loom.models.loom import LOOM
model = LOOM().fit(F_train, y_train)          # F: (n, p) features, e.g. MiniRocket output
P = model.predict_proba(F_test)               # leverage-moderated probabilities
S = model.conformal_sets(F_test, alpha=0.1, method="jackknife+", score="aps")  # boolean sets
```

**Bitwise reproducibility.** `run_all.py` pins one BLAS/OpenMP thread per job. With a different
thread count, matrix products change in the last bits; this does not affect the probabilistic
metrics materially, but it can flip exact ties of the deterministic APS score (which equals
1.0 for many classes in binary problems), changing APS coverage/size in the third decimal.

## 7. Runtime of the full study

Measured runtimes are in `tables/table_efficiency.csv` and `results/timing.csv`. With 4 cores
the `main` stage dominates; the scheduler runs datasets with more than 3,000 training series
(n_train > 5,000) one at a time and the rest on `workers − 1` processes, to stay within ~14 GB of RAM. A worker killed by the out-of-memory killer is replaced and its jobs are retried; `scripts/resume_full.sh` restarts an interrupted run, and cached jobs are never recomputed. The full study took roughly two days of wall-clock time on 4 cores.

## 8. Development disclosure

`experiments/pilot_dev.py` and `experiments/pilot_dev_c.py` are the development pilots on
8 randomly selected datasets (results in `results/dev_pilot*.{json,csv}`), kept for
transparency. The pre-registered protocol is in
`../supplementary/experimental_design_preregistration.md`.

## Licence

Code: MIT. UCR data: see the archive's terms.
