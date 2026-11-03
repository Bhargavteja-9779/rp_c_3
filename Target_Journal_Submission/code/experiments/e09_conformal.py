"""E9: closed-form jackknife+ prediction sets (Sec. 6.8).

Runs (or reuses the cached results of) the run_all.py stage(s) "main", then the aggregation,
which writes results/e09/summary.json, tables/macros.tex and tables/tab_*.tex.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = "main"

if __name__ == "__main__":
    if STAGES:
        subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--mode", "full", "--stages", STAGES], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--mode", "full", "--stages", "analyze,aggregate"], check=True, cwd=ROOT)
    print("summary:", ROOT / "results" / "e09" / "summary.json")
