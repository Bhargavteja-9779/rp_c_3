"""Probabilistic readouts compared against LOOM (all share the same features)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .readout import ClosedFormRidge


class RidgeEstimator(ClassifierMixin, BaseEstimator):
    """sklearn wrapper of the closed-form ridge (lambda chosen by LOO-SSE, as in
    RidgeClassifierCV), used inside CalibratedClassifierCV."""

    def __init__(self):
        pass

    def fit(self, X, y):
        self.m_ = ClosedFormRidge().fit(X, y)
        self.classes_ = self.m_.classes_
        return self

    def decision_function(self, X):
        Z = self.m_.decision_function(X)
        return Z[:, 1] - Z[:, 0] if len(self.classes_) == 2 else Z

    def predict(self, X):
        return self.m_.predict(X)


def _safe_cv(y, k):
    counts = np.bincount(np.unique(y, return_inverse=True)[1])
    return max(2, min(k, counts.min()))


class RidgeOneHot:
    """Default of the ROCKET-family implementations: hard one-hot 'probabilities'."""
    name = "Ridge-onehot"

    def fit(self, F, y):
        self.m = ClosedFormRidge().fit(F, y); self.classes_ = self.m.classes_; return self

    def predict_proba(self, F):
        Z = self.m.decision_function(F)
        P = np.zeros_like(Z); P[np.arange(len(Z)), Z.argmax(1)] = 1.0
        return P


class RidgeSoftmax:
    """Naive softmax of the raw ridge scores (temperature 1)."""
    name = "Ridge-softmax"

    def fit(self, F, y):
        self.m = ClosedFormRidge().fit(F, y); self.classes_ = self.m.classes_; return self

    def predict_proba(self, F):
        return np.exp(log_softmax(self.m.decision_function(F), axis=1))


class RidgeCalibratedCV:
    """CalibratedClassifierCV (k-fold, ensemble of fold models) around the ridge."""

    def __init__(self, method="sigmoid", cv=5):
        self.method, self.cv = method, cv
        self.name = f"Ridge-{'Platt' if method == 'sigmoid' else 'Isotonic'}-CV{cv}"

    def fit(self, F, y):
        k = _safe_cv(y, self.cv)
        self.m = CalibratedClassifierCV(RidgeEstimator(), method=self.method,
                                        cv=StratifiedKFold(k, shuffle=True, random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(F, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(F)


class LogReg:
    """L2 logistic regression on standardised features; C tuned by 3-fold CV log-loss."""
    name = "LogReg-CV"

    def __init__(self, Cs=(1e-3, 1e-2, 1e-1, 1.0), max_iter=300):
        self.Cs, self.max_iter = Cs, max_iter

    def fit(self, F, y):
        k = _safe_cv(y, 3)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=self.max_iter, tol=1e-3))
        self.m = GridSearchCV(pipe, {"logisticregression__C": list(self.Cs)}, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                              scoring="neg_log_loss", n_jobs=1, refit=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(F, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, F):
        return self.m.predict_proba(F)
