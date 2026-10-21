# LOOM — leverage-moderated leave-one-out calibration for random-kernel time series classifiers

Code, data instructions and machine-generated results for the manuscript
*"Shift-aware probabilities for random convolutional kernel time series classifiers via
leverage-moderated leave-one-out calibration"* (submitted to the Journal of King Saud
University – Computer and Information Sciences).

LOOM turns the ridge readout of MiniRocket/ROCKET-type classifiers into a probabilistic
classifier

    p(y | x) = softmax( a * z(x) / sqrt(h(x) / h_ref) )

where z are the ridge logits, h the leverage (posterior predictive variance factor), and the
temperature a and ridge penalty λ are fitted on the exact leave-one-out pairs
(z₋ᵢ, h₋ᵢ = dᵢ/(1 − dᵢ)). Everything comes from one thin SVD; no data are held out and nothing
is refitted. The same algebra gives exact closed-form jackknife+ prediction sets.

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
```

The script downloads `Univariate2018_arff.zip` from timeseriesclassification.com, extracts
the `*_TRAIN.txt` / `*_TEST.txt` files into `data/raw/Univariate_arff/`, and writes the list of
the 114 equal-length datasets without missing values (`data/raw/ucr_equal_length.csv`; a
copy is committed as `data/ucr_equal_length.csv`). The data are the property of their
original creators; please cite Dau et al. (2019).

## 3. Reproduce everything

```bash
python run_all.py --mode quick      # ~10 min: 6 small datasets, 1 resample — checks the whole pipeline
python run_all.py --mode full       # full study: 114 datasets x 5 resamples (+ extractors, train-size, Arsenal/QUANT)
```

Stages (`--stages audit,main,raw,extract,trainsize,analyze`; each is resumable because
finished jobs are cached as `results/raw/*.json`):

| Stage | What it runs | Output |
|---|---|---|
| `audit` | duplicate / leakage audit of the official splits | `results/leakage_audit.csv` |
| `main` | MiniRocket features; LOOM, 7 baselines, 8 ablations, 5-seed ensembles (official split), conformal sets; clean + 18 corruption conditions; 5 resamples (3 for the four datasets with > 3,000 training series) | `results/raw/minirocket__*__all.json` |
| `raw` | QUANT (all resamples) and Arsenal (official split; clean + severity-3) on raw series | `results/raw/raw__*.json` |
| `extract` | MultiRocket and Hydra features with the core readouts (official split; clean + severity-3 conditions; datasets with ≤ 3,000 training series) | `results/raw/{multirocket,hydra}__*.json` |
| `trainsize` | 25 % and 50 % of the training data (official split; clean + severity-3; datasets with ≤ 3,000 training series) | `results/raw/minirocket__*__tf*.json` |
| `analyze` | statistics, tables, figures | `results/*.csv|json`, `tables/`, `figures/` |

Then rebuild the manuscript from the generated results:

```bash
python ../manuscript/build/build_manuscript.py     # -> ../manuscript/Final_Manuscript.docx (+ .pdf via LibreOffice)
```

`python -m pytest -q tests` checks every closed-form identity (LOO logits, held-out
leverage, test leverage, cross-leverage, jackknife+ sets) against explicit refits.

## 4. Where each paper item comes from

| Paper item | File |
|---|---|
| Clean results (Table 1) | `tables/table_main_clean.{md,csv}` |
| Robustness (Table 2, Figs. 3–6) | `tables/table_robustness_shifted.*`, `tables/table_robustness_by_family.csv`, `figures/fig_robustness_*.png` |
| Ensembles, Arsenal, QUANT on the official split (Table 3) | `tables/table_rawbaselines_*.md` |
| Ablation (Fig. 7) | `tables/table_ablation_*.md`, `figures/fig_ablation.png` |
| Extractors and dataset types | `tables/table_generalisation_*.md`, `tables/table_by_type.csv` |
| Conformal sets (Table 4) | `tables/table_conformal.csv`, `figures/fig_conformal.png` |
| Efficiency (Table 5, Fig. 8) | `tables/table_efficiency.csv`, `figures/fig_efficiency.png` |
| Training-set size (Table S4) | `tables/table_train_size.csv` |
| Natural cross-axis shift (§6.5) | `experiments/natural_shift.py` → `results/natural_shift.json`, `tables/table_natural_shift*.csv` |
| Grouped-by-source sensitivity | `results/statistics.json` (`sens_grouped_*`) |
| LogReg convergence check (§6.8) | `experiments/check_logreg_convergence.py` → `results/check_logreg_convergence.json` |
| Sensitivity subsets | `tables/table_sensitivity_{unseen,nodup}_*.md` |
| Error analysis | `tables/table_error_analysis_datasets.csv`, `figures/fig_error_analysis.png` |
| All statistics (Wilcoxon, Holm, effect sizes, CIs, Friedman) | `results/statistics.json` |
| Headline medians | `results/metrics.json` |
| Every per-dataset/resample/condition metric | `results/raw_results.csv` |

## 5. Code layout

```
src/loom/
  data/ucr.py                 loading, deterministic resampling, duplicate audit
  preprocessing/corruptions.py  6 corruption families x 3 severities (test-time only)
  models/readout.py           closed-form ridge: SVD, exact LOO logits / leverages, test & cross leverage
  models/calibration.py       temperature / moderation fitting, Platt target smoothing
  models/loom.py              LOOM (+ ablation variants), closed-form jackknife(+) sets (LAC, APS)
  models/baselines.py         ridge-onehot/softmax, CalibratedClassifierCV, logistic regression
  training/runner.py          one benchmark job (cached JSON), Arsenal/QUANT job
  evaluation/metrics.py       NLL, ECE, Brier, classwise ECE, AURC, set metrics, reliability bins
  evaluation/aggregate.py     tidy tables from cached jobs
  statistics/tests.py         Wilcoxon + Holm, rank-biserial, bootstrap CI, Friedman, Nemenyi CD
  visualization/plots.py      all figures
experiments/                  development pilots (disclosed), leakage audit, analysis
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
