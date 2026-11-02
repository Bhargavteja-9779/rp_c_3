"""LOOM: fault-aware, closed-form probabilities for fast random-kernel time series classifiers."""
__version__ = "1.0.0"

from .api import LOOMClassifier  # noqa: E402,F401

__all__ = ["LOOMClassifier", "__version__"]
