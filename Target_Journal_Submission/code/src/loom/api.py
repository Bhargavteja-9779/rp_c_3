"""User-facing API: a scikit-learn style classifier for raw time series with fault-aware confidence.

    from loom import LOOMClassifier
    clf = LOOMClassifier().fit(X_train, y_train)        # X: (n_series, length) array
    P = clf.predict_proba(X_new)                        # calibrated, leverage-moderated probabilities
    queue = clf.triage(X_new, budget=0.2)              # which recordings a human should check

The classifier is MiniRocket features + the LOOM readout described in the paper. Everything is
fitted in closed form from one SVD; no validation data are needed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .models.loom import LOOM
from .models.readout import ClosedFormRidge


def _as_2d(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        if X.shape[1] != 1:
            raise ValueError("only univariate series are supported (shape (n, 1, length))")
        X = X[:, 0, :]
    if X.ndim != 2:
        raise ValueError("X must have shape (n_series, length)")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values; interpolate missing samples first")
    return X


class LOOMClassifier:
    """Time series classifier whose confidence drops for recordings unlike the training data.

    Parameters
    ----------
    transform : {"minirocket", "multirocket", "hydra"}
        Random-kernel feature extractor (aeon implementation).
    random_state : int
        Seed of the feature extractor.
    smooth : bool
        Platt target smoothing of the calibration (recommended; see the paper, Section 4.4).
    """

    def __init__(self, transform="minirocket", random_state=0, smooth=True):
        self.transform = transform
        self.random_state = random_state
        self.smooth = smooth

    # -- fitting -------------------------------------------------------------------------------
    def _features(self, X):
        return np.asarray(self.transformer_.transform(X[:, None, :]), dtype=np.float64)

    def fit(self, X, y):
        from .training.runner import make_transform
        X = _as_2d(X); y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("need at least two classes")
        self.transformer_ = make_transform(self.transform, self.random_state).fit(X[:, None, :])
        F = self._features(X)
        ridge = ClosedFormRidge().fit(F, y)
        self.model_ = LOOM("standard", smooth=self.smooth).fit(F, y, ridge=ridge)
        self.classes_ = self.model_.classes_
        self.length_ = X.shape[1]
        return self

    def _check(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("call fit() first")
        X = _as_2d(X)
        if X.shape[1] != self.length_:
            raise ValueError(f"series length {X.shape[1]} differs from the training length {self.length_}")
        return X

    # -- prediction ----------------------------------------------------------------------------
    def predict_proba(self, X):
        return self.model_.predict_proba(self._features(self._check(X)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def unusualness(self, X):
        """Leverage ratio h(x)/h_ref: about 1 for recordings like the training data, larger for unusual ones."""
        F = self._features(self._check(X))
        return self.model_.ridge_.leverage(F) / self.model_.h_ref_

    def predict_sets(self, X, alpha=0.1, score="aps"):
        """Exact jackknife+ prediction sets (boolean array, one column per class) at miscoverage alpha."""
        return self.model_.conformal_sets(self._features(self._check(X)), alpha=alpha, method="jackknife+", score=score)

    def triage(self, X, budget=0.2, threshold=None, alpha=None):
        """Split recordings into automatic decisions and a human review queue.

        Either the `budget` share of least confident recordings, or (if `threshold` is given) every
        recording with confidence below `threshold`, is marked for review. Returns a pandas
        DataFrame with prediction, confidence, unusualness, decision and, if `alpha` is given, the
        jackknife+ prediction set.
        """
        import pandas as pd
        X = self._check(X)
        F = self._features(X)
        P = self.model_.predict_proba(F)
        conf = P.max(1)
        if threshold is not None:
            review = conf < threshold
        else:
            k = int(round(budget * len(X)))
            review = np.zeros(len(X), bool)
            review[np.argsort(conf, kind="stable")[:k]] = True
        df = pd.DataFrame({"prediction": self.classes_[P.argmax(1)], "confidence": conf,
                           "unusualness": self.model_.ridge_.leverage(F) / self.model_.h_ref_,
                           "decision": np.where(review, "review", "auto")})
        if alpha is not None:
            S = self.model_.conformal_sets(F, alpha=alpha, method="jackknife+", score="aps")
            df["prediction_set"] = [" ".join(map(str, self.classes_[s])) for s in S]
        return df

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "LOOMClassifier":
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, LOOMClassifier):
            raise TypeError("file does not contain a LOOMClassifier")
        return obj
