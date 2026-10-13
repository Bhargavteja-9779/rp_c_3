"""Closed-form ridge readout with exact leave-one-out (LOO) quantities.

Everything is derived from one thin SVD of the centred, standardised training
feature matrix.  For a ridge parameter ``lam`` and unpenalised intercept:

    fitted logits        Z      = 1 ybar^T + U diag(s^2/(s^2+lam)) U^T Yc
    hat diagonal         d_i    = 1/n + sum_j u_ij^2 s_j^2/(s_j^2+lam)
    exact LOO logits     Z_-i   = Y_i - (Y_i - Z_i) / (1 - d_i)
    held-out leverage    h_-i   = d_i / (1 - d_i)                 (Sherman-Morrison)
    test leverage        h(x)   = 1/n + sum_j (v_j^T psi)^2/(s_j^2+lam)
                                   + (||psi||^2 - ||V^T psi||^2) / lam
    cross leverage       k(x,i) = 1/n + (V^T psi)^T diag(1/(s^2+lam)) (V^T psi_i)

where psi = standardised(x) - training mean.  ``h`` is the quadratic form
z^T A^{-1} z with A = Z_aug^T Z_aug + blockdiag(0, lam I), i.e. the posterior
predictive variance factor of the equivalent Bayesian linear model.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LAMBDAS = np.logspace(-3, 3, 10)


def _encode(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One-vs-rest targets in {-1, +1} (as in sklearn's RidgeClassifier)."""
    Y = -np.ones((len(y), len(classes)))
    Y[np.arange(len(y)), np.searchsorted(classes, y)] = 1.0
    return Y


class ClosedFormRidge:
    """Multi-output ridge classifier exposing exact LOO logits and leverages.

    Parameters
    ----------
    lambdas : candidate ridge penalties (applied to standardised features,
        scaled by n so that the grid is comparable across dataset sizes, which
        matches sklearn's former ``normalize=True`` behaviour used by MiniRocket).
    standardize : standardise features with training mean / std.
    """

    def __init__(self, lambdas=DEFAULT_LAMBDAS, standardize: bool = True, scale_by_n: bool = True):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.standardize = standardize
        self.scale_by_n = scale_by_n

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClosedFormRidge":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        n, p = X.shape
        self.n_, self.p_ = n, p
        self.mu_ = X.mean(0)
        if self.standardize:
            sd = X.std(0)
            sd[sd < 1e-12] = 1.0
            self.sd_ = sd
        else:
            self.sd_ = np.ones(p)
        Xc = (X - self.mu_) / self.sd_
        Y = _encode(y, self.classes_)
        self.ybar_ = Y.mean(0)
        Yc = Y - self.ybar_
        # thin SVD via the smaller Gram matrix (exact up to round-off)
        if n <= p:
            G = Xc @ Xc.T
            w, U = np.linalg.eigh(G)
            keep = w > max(w.max(), 1e-300) * 1e-10
            w, U = w[keep], U[:, keep]
            s = np.sqrt(w)
            V = (Xc.T @ U) / s
        else:
            G = Xc.T @ Xc
            w, V = np.linalg.eigh(G)
            keep = w > max(w.max(), 1e-300) * 1e-10
            w, V = w[keep], V[:, keep]
            s = np.sqrt(w)
            U = (Xc @ V) / s
        self.U_, self.s_, self.V_ = U, s, V
        self.UtY_ = U.T @ Yc                     # r x K
        self.Y_ = Y
        self.y_ = np.asarray(y)
        self.lam_scale_ = n if self.scale_by_n else 1.0
        self._shared = {}          # projection cache shared by all views (see _project)
        # LOO squared error for each lambda (the criterion of RidgeClassifierCV)
        self.loo_sse_ = np.array([self._loo(l)[2] for l in self.lambdas])
        self.set_lambda(self.lambdas[int(np.argmin(self.loo_sse_))])
        return self

    def _loo(self, lam_raw: float):
        lam = lam_raw * self.lam_scale_
        s2 = self.s_ ** 2
        shrink = s2 / (s2 + lam)
        Zfit = self.ybar_ + self.U_ @ (shrink[:, None] * self.UtY_)
        d = 1.0 / self.n_ + (self.U_ ** 2) @ shrink
        d = np.minimum(d, 1 - 1e-10)
        E = self.Y_ - Zfit
        Zloo = self.Y_ - E / (1 - d)[:, None]
        return Zloo, d, float(((self.Y_ - Zloo) ** 2).sum())

    def set_lambda(self, lam_raw: float) -> "ClosedFormRidge":
        self.lambda_ = float(lam_raw)
        lam = lam_raw * self.lam_scale_
        self._lam = lam
        self._dinv = 1.0 / (self.s_ ** 2 + lam)
        # primal coefficients on the standardised scale: V diag(s/(s^2+lam)) U^T Yc
        self.coef_ = self.V_ @ ((self.s_ * self._dinv)[:, None] * self.UtY_)   # p x K
        self.loo_logits_, self.hat_diag_, _ = self._loo(lam_raw)
        self.loo_leverage_ = self.hat_diag_ / (1 - self.hat_diag_)
        return self

    def view(self, lam_raw=None):
        """Shallow copy sharing the SVD factors (U, s, V) but owning its own
        lambda-dependent state; lets many readouts share one decomposition."""
        import copy
        v = copy.copy(self)
        if lam_raw is not None:
            v.set_lambda(lam_raw)
        return v

    # -------------------------------------------------------------- predict
    def _psi(self, X):
        P = np.asarray(X, dtype=np.float64) - self.mu_
        P /= self.sd_
        return P

    def _project(self, X):
        """(psi, V^T psi, ||psi||^2) for X; cached for the most recent X object so that
        all views sharing this decomposition reuse it (identity check, not hashing)."""
        c = self._shared.get("last")
        if c is not None and c[0] is X:
            return c[1], c[2], c[3]
        P = self._psi(X)
        Q = P @ self.V_
        sq = np.einsum("ij,ij->i", P, P)
        self._shared["last"] = (X, P, Q, sq)
        return P, Q, sq

    def decision_function(self, X):
        P, _, _ = self._project(X)
        return self.ybar_ + P @ self.coef_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), 1)]

    def leverage(self, X):
        """h(x) = z^T A^{-1} z for new points (includes out-of-span component)."""
        P, Q, sq = self._project(X)                       # Q: m x r
        inspan = (Q ** 2) @ self._dinv
        resid = np.maximum(sq - (Q ** 2).sum(1), 0.0) / self._lam
        return 1.0 / self.n_ + inspan + resid

    def logits_and_leverage(self, X):
        P, Q, sq = self._project(X)
        Z = self.ybar_ + P @ self.coef_
        h = 1.0 / self.n_ + (Q ** 2) @ self._dinv + np.maximum(sq - (Q ** 2).sum(1), 0.0) / self._lam
        return Z, h, Q

    def cross_leverage(self, Q):
        """k(x, i) for test projections Q (m x r) against all training points (m x n).
        Training points lie in span(V) so the out-of-span term vanishes."""
        Qtr = self.U_ * self.s_                           # n x r  (= V^T psi_i)
        return 1.0 / self.n_ + (Q * self._dinv) @ Qtr.T
