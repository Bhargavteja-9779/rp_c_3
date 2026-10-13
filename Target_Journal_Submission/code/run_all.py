#!/usr/bin/env python3
"""Reproduce every experiment, statistic, table and figure.

    python run_all.py --mode quick     # pipeline check: 6 small datasets, 1 resample, reduced ensembles
    python run_all.py --mode full      # full study (114 datasets, 5 resamples; ~1-2 days on 4 CPU cores)

Stages (each is resumable: finished jobs are cached in results/raw/*.json):
  audit     leakage audit of the official splits
  main      MiniRocket features, all readouts + ablations + conformal, 5 resamples, 5-seed ensembles
  raw       Arsenal and QUANT (raw-series probabilistic baselines)
  extract   Rocket / MultiRocket / Hydra features, core readouts (generalisation)
  trainsize 25 % / 50 % training data, core readouts (sensitivity)
  analyze   statistics, tables, figures
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")          # one core per job; parallelism across jobs
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

QUICK_DATASETS = ["Coffee", "ECG200", "GunPoint", "ItalyPowerDemand", "SonyAIBORobotSurface2", "TwoLeadECG"]


def _job(kind, ds, r, kw, raw_dir):
    import warnings
    warnings.filterwarnings("ignore")
    from loom.training.runner import run_job, run_raw_baselines
    t = time.time()
    try:
        if kind == "raw":
            res = run_raw_baselines(ds, r, outdir=raw_dir)
        else:
            res = run_job(ds, r, outdir=raw_dir, **kw)
        err = res["extra"].get("errors")
        return f"{kind:9s} {ds:30s} r{r} {time.time()-t:7.1f}s" + (f" ERRORS: {list(err)}" if err else "")
    except Exception as e:  # noqa: BLE001
        import traceback
        return f"{kind:9s} {ds:30s} r{r} FAILED {e!r} {traceback.format_exc()[-300:]}"


def schedule(jobs, workers, raw_dir, heavy_n=3000, log=None):
    """Run light jobs on `workers-1` processes and heavy jobs (n_train > heavy_n) on one
    dedicated process, so at most one memory-hungry job runs at a time."""
    from loom.data.ucr import dataset_info
    info = dataset_info()
    cost = lambda j: (info[j[1]]["n_train"] + 19 * info[j[1]]["n_test"]) * info[j[1]]["length"]
    heavy = sorted([j for j in jobs if info[j[1]]["n_train"] > heavy_n], key=cost, reverse=True)
    light = sorted([j for j in jobs if info[j[1]]["n_train"] <= heavy_n], key=cost, reverse=True)
    n_light = max(1, workers - (1 if heavy else 0))
    done = 0
    with ProcessPoolExecutor(n_light) as pl, ProcessPoolExecutor(1) as ph:
        futs = [pl.submit(_job, *j, raw_dir) for j in light] + [ph.submit(_job, *j, raw_dir) for j in heavy]
        for f in as_completed(futs):
            done += 1
            msg = f"[{done}/{len(futs)}] {f.result()}"
            print(msg, flush=True)
            if log:
                with open(log, "a") as fh:
                    fh.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--stages", default="audit,main,raw,extract,trainsize,analyze")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--datasets", default=None, help="comma-separated subset (default: mode-dependent)")
    a = ap.parse_args()
    from loom.data.ucr import dataset_names
    stages = a.stages.split(",")
    if a.mode == "quick":
        dsets = QUICK_DATASETS
        raw_dir = ROOT / "results" / "quick" / "raw"
        R, R_gen, n_ens, extractors, fracs = [0], [0], 2, ["multirocket"], [0.5]
    else:
        dsets = dataset_names()
        raw_dir = ROOT / "results" / "raw"
        R, R_gen, n_ens, extractors, fracs = [0, 1, 2, 3, 4], [0], 5, ["rocket", "multirocket", "hydra"], [0.25, 0.5]
    if a.datasets:
        dsets = a.datasets.split(",")
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / "results" / "logs" / f"run_all_{a.mode}.log"; log.parent.mkdir(parents=True, exist_ok=True)

    if "audit" in stages:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "experiments" / "leakage_audit.py")], check=True)
    if "main" in stages:
        # 5-seed ensembles on the official split only (compute); other resamples use one seed
        schedule([("main", d, r, dict(methods="all", n_ens=n_ens if r == 0 else 1, skip_slow_n=2000)) for d in dsets for r in R], a.workers, raw_dir, log=log)
    if "raw" in stages:
        schedule([("raw", d, r, {}) for d in dsets for r in R], a.workers, raw_dir, log=log)
    if "extract" in stages:
        schedule([("extract", d, r, dict(methods="core", transform=t, n_ens=1, conds_subset="sev3")) for t in extractors for d in dsets for r in R_gen], a.workers, raw_dir, log=log)
    if "trainsize" in stages:
        schedule([("trainsize", d, 0, dict(methods="core", n_ens=1, train_frac=f)) for f in fracs for d in dsets], a.workers, raw_dir, log=log)
    if "analyze" in stages:
        import importlib.util
        spec = importlib.util.spec_from_file_location("analyze", ROOT / "experiments" / "analyze.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if a.mode == "quick":
            mod.ROOT = ROOT   # analysis of the quick run writes into results/quick
            _analyze_quick(mod, raw_dir)
        else:
            mod.main("full")


def _analyze_quick(mod, raw_dir):
    """Run the same analysis code on the quick results, writing to results/quick/."""
    import shutil
    q = ROOT / "results" / "quick"
    tmp_root = q / "_root"; (tmp_root / "results").mkdir(parents=True, exist_ok=True)
    for sub in ("data",):
        if not (tmp_root / sub).exists():
            os.symlink(ROOT / sub, tmp_root / sub)
    if (tmp_root / "results" / "raw").exists() or (tmp_root / "results" / "raw").is_symlink():
        os.remove(tmp_root / "results" / "raw") if (tmp_root / "results" / "raw").is_symlink() else shutil.rmtree(tmp_root / "results" / "raw")
    os.symlink(raw_dir, tmp_root / "results" / "raw")
    if (ROOT / "results" / "leakage_audit.csv").exists():
        shutil.copy(ROOT / "results" / "leakage_audit.csv", tmp_root / "results" / "leakage_audit.csv")
    mod.ROOT = tmp_root
    mod.main("quick")
    print(f"quick-mode outputs in {tmp_root}/(results|tables|figures)")


if __name__ == "__main__":
    main()
