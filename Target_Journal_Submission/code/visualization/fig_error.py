"""Figure produced by loom.visualization.plots.fig_error from the machine-generated results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loom.visualization import plots   # noqa: E402

if __name__ == "__main__":
    plots._style()
    plots.fig_error(ROOT, ROOT / "figures")
