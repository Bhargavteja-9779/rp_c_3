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


def schedule(jobs, workers, raw_dir, heavy_n=3000, log=None, max_retries=2):
    """Run jobs on a process pool, largest first. If a worker is killed (e.g. by the
    out-of-memory killer) the broken pool is replaced and the lost jobs are retried up to
    `max_retries` times; finished jobs are cached, so nothing is computed twice."""
    from concurrent.futures.process import BrokenProcessPool
    from loom.data.ucr import dataset_info
    info = dataset_info()
    cost = lambda j: (info[j[1]]["n_train"] + 19 * info[j[1]]["n_test"]) * info[j[1]]["length"]
    stage_rank = {"main": 0, "raw": 1, "extract": 2, "trainsize": 3}
    pending = sorted(jobs, key=lambda j: (stage_rank[j[0]], -cost(j)))
    tries = {}
    done = 0
    total = len(pending)

    def emit(msg):
        print(msg, flush=True)
        if log:
            with open(log, "a") as fh:
                fh.write(msg + "\n")
    while pending:
        retry = []
        with ProcessPoolExecutor(max(1, workers)) as pool:
            futs = {pool.submit(_job, *j, raw_dir): j for j in pending}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    res = f.result(); done += 1
                    emit(f"[{done}/{total}] {res}")
                except BrokenProcessPool as e:
                    k = (j[0], j[1], j[2], str(sorted(j[3].items())))
                    tries[k] = tries.get(k, 0) + 1
                    if tries[k] <= max_retries:
                        retry.append(j)
                    else:
                        done += 1
                        emit(f"[{done}/{total}] {j[0]:9s} {j[1]:30s} r{j[2]} FAILED after {max_retries} retries ({e!r})")
                except Exception as e:  # noqa: BLE001
                    done += 1
                    emit(f"[{done}/{total}] {j[0]:9s} {j[1]:30s} r{j[2]} FAILED ({e!r})")
        if retry:
            emit(f"pool broke; retrying {len(retry)} jobs in a fresh pool")
        pending = sorted(retry, key=lambda j: (stage_rank[j[0]], -cost(j)))


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
        R, R_gen, n_ens, extractors, fracs = [0, 1, 2, 3, 4], [0], 5, ["multirocket", "hydra"], [0.25, 0.5]
    if a.datasets:
        dsets = a.datasets.split(",")
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / "results" / "logs" / f"run_all_{a.mode}.log"; log.parent.mkdir(parents=True, exist_ok=True)

    if "audit" in stages:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "experiments" / "leakage_audit.py")], check=True)
    # all compute stages are queued together so that light jobs of later stages fill workers
    # that would otherwise idle while the few heavy jobs finish
    jobs = []
    LARGE = {"Crop", "ElectricDevices", "FordA", "FordB"}      # n_train > 3000 (amendment 6)
    def R_of(d):
        return [r for r in R if r <= 2] if (d in LARGE and a.mode == "full") else R
    def ens_of(d, r):
        return 1 if (d == "ElectricDevices" or r != 0) else n_ens
    if "main" in stages:
        jobs += [("main", d, r, dict(methods="all", n_ens=ens_of(d, r), skip_slow_n=2000)) for d in dsets for r in R_of(d)]
    if "raw" in stages:
        jobs += [("raw", d, r, {}) for d in dsets for r in R_of(d)]
    if "extract" in stages:
        jobs += [("extract", d, r, dict(methods="core", transform=t, n_ens=1, conds_subset="sev3")) for t in extractors for d in dsets for r in R_gen if d not in LARGE]
    if "trainsize" in stages:
        jobs += [("trainsize", d, 0, dict(methods="core", n_ens=1, train_frac=f, conds_subset="sev3")) for f in fracs for d in dsets if d not in LARGE]
    if jobs:
        from loom.data.ucr import dataset_info
        info = dataset_info()
        heavy = [j for j in jobs if info[j[1]]["n_train"] > 5000]       # Crop, ElectricDevices: run alone at the end
        light = [j for j in jobs if info[j[1]]["n_train"] <= 5000]
        schedule(light, max(1, a.workers - 1), raw_dir, log=log)
        schedule(heavy, 1, raw_dir, log=log)
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
