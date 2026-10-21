"""Run all jobs of the given datasets (every stage of the full protocol) in one process,
in reverse order; finished jobs are cached and skipped, so this can run next to run_all.py."""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
from loom.training.runner import run_job, run_raw_baselines
out = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
jobs = []
for d in sys.argv[1:]:
    jobs += [("main", d, r, dict(methods="all", n_ens=5 if r == 0 else 1, skip_slow_n=2000)) for r in range(5)]
    jobs += [("raw", d, r, {}) for r in range(5)]
    jobs += [("extract", d, 0, dict(methods="core", transform=t, n_ens=1, conds_subset="sev3")) for t in ("multirocket", "hydra")]
    jobs += [("trainsize", d, 0, dict(methods="core", n_ens=1, train_frac=f, conds_subset="sev3")) for f in (0.25, 0.5)]
for kind, d, r, kw in reversed(jobs):
    t = time.time()
    res = run_raw_baselines(d, r, outdir=out) if kind == "raw" else run_job(d, r, outdir=out, **kw)
    print(f"{kind:9s} {d:20s} r{r} {time.time()-t:7.1f}s errors={list((res['extra'].get('errors') or {}))}", flush=True)
