"""Fig. 4: real recorded ECG noise (E12): example beat with each MIT-BIH NSTDB noise type at 6 dB, and median
NLL against SNR per noise type."""
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
    an._real_noise_figure(json.load(open(ROOT / "results" / "real_noise.json")), ROOT / "figures")
