"""Fig. 3: pre-registered confirmatory triage study (E13, 38 datasets): error of automatic decisions and share
of faulty recordings caught, against the review budget. Curves: mean over datasets of the mean over resamples."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.visualization.plots import _style   # noqa: E402

if __name__ == "__main__":
    _style()
    spec = importlib.util.spec_from_file_location("analyze", ROOT / "experiments" / "analyze.py")
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    ds = {r["dataset"] for r in json.load(open(ROOT / "results" / "case_study_confirm.json"))}
    an._case_study_figure(ROOT / "results" / "case_study", ROOT / "figures", datasets=ds, name="fig_triage")
    ds = {r["dataset"] for r in json.load(open(ROOT / "results" / "case_study.json"))}
    an._case_study_figure(ROOT / "results" / "case_study", ROOT / "figures", datasets=ds, name="fig_triage_exploratory")
