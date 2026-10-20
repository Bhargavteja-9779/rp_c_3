"""LOOM: Leave-One-Out-fitted, leverage-Moderated probabilistic ridge readout.

    p(y | x) = softmax( a * z(x) / sqrt(h(x)) )

z(x): ridge logits, h(x): leverage (posterior predictive variance factor of the
equivalent Bayesian linear model).  z / sqrt(h) is therefore a standardised
logit.  The ridge penalty lambda and the inverse temperature a are chosen by
minimising the NLL of the n exact leave-one-out pairs (z_{-i}, h_{-i}); no data
are held out and no model is refitted.

Variants (used as ablations) are selected with ``moderation``:
  "none"      : p = softmax(a z)                     (PreVal; Dempster et al.)
  "standard"  : p = softmax(a z / sqrt(h))           (LOOM default, c -> inf)
  "finite"    : p = softmax(a z / sqrt(1 + c h/h_ref))
  "free"      : p = softmax(a z / sqrt(1 + b h)), b fitted on LOO pairs
  "knn"       : as "standard" but h replaced by a k-NN squared distance ratio
``smooth`` applies Platt target smoothing to the calibration NLL (default True;
PreVal as published uses smooth=False).
and ``pairs``:
  "loo"       : exact LOO pairs (default)
  "insample"  : in-sample fitted pairs (z_i, h_i)      -> tests the need for LOO
  "holdout"   : refit on 80 %, calibrate on 20 %        -> tests data efficiency
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax

from .calibration import fit_moderation
from .readout import ClosedFormRidge, DEFAULT_LAMBDAS


class LOOM:
    def __init__(self, moderation="standard", pairs="loo", c=1.0, lambdas=DEFAULT_LAMBDAS,
                 lambda_criterion="nll", knn_k=5, holdout_frac=0.2, seed=0, smooth=True):
        self.moderation, self.pairs, self.c, self.smooth = moderation, pairs, c, smooth
        self.lambdas, self.lambda_criterion = np.asarray(lambdas), lambda_criterion
        self.knn_k, self.holdout_frac, self.seed = knn_k, holdout_frac, seed

    # ---------------------------------------------------------- helpers
    def _signal_train(self, m):
        """uncertainty signal for the calibration pairs"""
        if self.moderation == "knn":
            return self._knn_loo
        return m.loo_leverage_ if self._pairs_kind == "loo" else m.hat_diag_

    def _signal_test(self, m, F, h):
        if self.moderation == "knn":
            return self._knn_dist(F)
        return h

    def _moderate(self, Z, h, a, b):
        if self.moderation == "none":
            S = Z
        elif self.moderation in ("standard", "knn"):
            S = Z / np.sqrt(h / self.h_ref_)[:, None]
        else:  # finite / free share the 1 + b h form
            S = Z / np.sqrt(1.0 + b * h)[:, None]
        return log_softmax(a * S, axis=1)

    def _knn_prepare(self, F):
        P = (F - self.ridge_.mu_) / self.ridge_.sd_
        self._Ptr = P.astype(np.float32)
        sq = (self._Ptr ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * self._Ptr @ self._Ptr.T
        np.fill_diagonal(D, np.inf)
        k = min(self.knn_k, len(P) - 1)
        self._knn_loo = np.sort(np.maximum(D, 0), 1)[:, :k].mean(1).astype(np.float64) + 1e-12

    def _knn_dist(self, F):
        P = ((F - self.ridge_.mu_) / self.ridge_.sd_).astype(np.float32)
        out = np.empty(len(P))
        sqt = (self._Ptr ** 2).sum(1)
        k = min(self.knn_k, len(self._Ptr))
        for s in range(0, len(P), 512):
            Q = P[s:s + 512]
            D = (Q ** 2).sum(1)[:, None] + sqt[None, :] - 2 * Q @ self._Ptr.T
            out[s:s + 512] = np.sort(np.maximum(D, 0), 1)[:, :k].mean(1)
        return out + 1e-12

    # -------------------------------------------------------------- fit
    def fit(self, F, y, ridge=None):
        """``ridge``: optional ClosedFormRidge already fitted on (F, y) whose SVD is reused."""
        self.classes_ = np.unique(y)
        self._pairs_kind = self.pairs
        if self.pairs == "holdout":
            return self._fit_holdout(F, y)
        m = ridge.view() if ridge is not None else ClosedFormRidge(lambdas=self.lambdas).fit(F, y)
        self.ridge_ = m
        yi = np.searchsorted(self.classes_, y)
        if self.moderation == "knn":
            self._knn_prepare(F)
        best = None
        for lam in self.lambdas:
            m.set_lambda(lam)
            Zc = m.loo_logits_ if self.pairs == "loo" else (m.Y_ - (m.Y_ - m.loo_logits_) * (1 - m.hat_diag_)[:, None])
            hs = self._signal_train(m)
            href = float(np.median(hs))
            if self.moderation == "none":
                a, b, nll = fit_moderation(Zc, hs, yi, fit_b=False, b_fixed=0.0, smooth=self.smooth)
            elif self.moderation in ("standard", "knn"):
                a, b, nll = fit_moderation(Zc / np.sqrt(hs / href)[:, None], hs, yi, fit_b=False, b_fixed=0.0, smooth=self.smooth)
            elif self.moderation == "finite":
                a, b, nll = fit_moderation(Zc, hs, yi, fit_b=False, b_fixed=self.c / href, smooth=self.smooth)
            elif self.moderation == "free":
                a, b, nll = fit_moderation(Zc, hs, yi, fit_b=True, smooth=self.smooth)
            else:
                raise ValueError(self.moderation)
            crit = nll if self.lambda_criterion == "nll" else ((m.Y_ - m.loo_logits_) ** 2).sum()
            if best is None or crit < best[0]:
                best = (crit, lam, a, b, href, nll)
        _, lam, self.a_, self.b_, self.h_ref_, self.loo_nll_ = best
        m.set_lambda(lam)
        self.lambda_ = lam
        return self

    def _fit_holdout(self, F, y):
        rng = np.random.default_rng(self.seed)
        idx_tr, idx_ca = [], []
        for c in self.classes_:
            idx = rng.permutation(np.flatnonzero(y == c))
            k = int(np.floor(self.holdout_frac * len(idx)))
            if len(idx) >= 2 and k == 0:
                k = 1
            idx_ca.append(idx[:k]); idx_tr.append(idx[k:])
        itr, ica = np.concatenate(idx_tr), np.concatenate(idx_ca)
        m = ClosedFormRidge(lambdas=self.lambdas).fit(F[itr], y[itr])     # lambda by LOO-SSE on 80 %
        # classes missing from the 80 % part keep a column of -1 logits
        self.ridge_ = m
        Z, h, _ = m.logits_and_leverage(F[ica])
        Z = self._expand(Z, m.classes_)
        yi = np.searchsorted(self.classes_, y[ica])
        self.h_ref_ = float(np.median(h))
        if self.moderation == "none":
            self.a_, self.b_, self.loo_nll_ = fit_moderation(Z, h, yi, fit_b=False, b_fixed=0.0, smooth=self.smooth)
        else:
            self.a_, self.b_, self.loo_nll_ = fit_moderation(Z / np.sqrt(h / self.h_ref_)[:, None], h, yi, fit_b=False, smooth=self.smooth)
        self.lambda_ = m.lambda_
        return self

    def _expand(self, Z, cls):
        if len(cls) == len(self.classes_):
            return Z
        out = -np.ones((len(Z), len(self.classes_)))
        out[:, np.searchsorted(self.classes_, cls)] = Z
        return out

    # ---------------------------------------------------------- predict
    def predict_log_proba(self, F, return_h=False):
        if self.moderation == "none" and not return_h:      # no leverage needed: skip the projection
            Z = self._expand(self.ridge_.decision_function(F), self.ridge_.classes_)
            return log_softmax(self.a_ * Z, axis=1)
        Z, h, Q = self.ridge_.logits_and_leverage(F)
        Z = self._expand(Z, self.ridge_.classes_)
        sig = self._signal_test(self.ridge_, F, h)
        LP = self._moderate(Z, sig, self.a_, self.b_)
        return (LP, h, Q, Z) if return_h else LP

    def predict_proba(self, F):
        return np.exp(self.predict_log_proba(F))

    def predict(self, F):
        return self.classes_[np.argmax(self.predict_log_proba(F), 1)]

    # --------------------------------------------------------- conformal
    def loo_log_proba(self):
        """LOO log-probabilities of the training points under the fitted calibrator."""
        m = self.ridge_
        sig = self._knn_loo if self.moderation == "knn" else m.loo_leverage_
        return self._moderate(m.loo_logits_, sig, self.a_, self.b_)

    def conformal_sets(self, F, alpha=0.1, method="jackknife+", chunk=None, score="lac"):
        """Prediction sets with the LAC score s(x, y) = 1 - p(y | x) (Sadinle et al.) or the
        deterministic APS score s(x, y) = sum of p_k over classes with p_k >= p_y
        (Romano et al. 2020), which yields larger rather than empty sets when p is flat.

        jackknife : threshold = ceil((n+1)(1-alpha))-th smallest LOO score.
        jackknife+: y in C(x) iff #{i : s_i < s_{-i}(x, y)} < (1-alpha)(n+1), where
                    s_{-i}(x, y) uses the exact LOO model f_{-i} evaluated at x:
                    z_{-i}(x) = z(x) - k(x,i) e_i / (1-d_i),
                    h_{-i}(x) = h(x) + k(x,i)^2 / (1-d_i).
        """
        assert self.pairs == "loo" and self.moderation in ("standard", "none")
        m = self.ridge_
        yi = np.searchsorted(self.classes_, m.y_)
        n = len(yi)
        s_loo = _score(np.exp(self.loo_log_proba()), score)[np.arange(n), yi]
        if method == "jackknife":
            P = self.predict_proba(F)
            k = int(np.ceil((n + 1) * (1 - alpha)))
            q = np.inf if k > n else np.sort(s_loo)[k - 1]
            return _score(P, score) <= q
        Z, h, Q = m.logits_and_leverage(F)
        E = m.Y_ - (m.ybar_ + m.U_ @ ((m.s_ ** 2 * m._dinv)[:, None] * m.UtY_))   # full-fit residuals (n x K)
        Ed = E / (1 - m.hat_diag_)[:, None]
        out = np.zeros((len(F), len(self.classes_)), bool)
        if chunk is None:   # bound the c x n x K working arrays to ~16M entries
            chunk = max(1, int(1.6e7 // (n * len(self.classes_))))
        thr = (1 - alpha) * (n + 1)
        for st in range(0, len(F), chunk):
            Kx = m.cross_leverage(Q[st:st + chunk])                    # c x n
            Zm = Z[st:st + chunk, None, :] - Kx[:, :, None] * Ed[None, :, :]   # c x n x K
            if self.moderation == "standard":
                hm = h[st:st + chunk, None] + Kx ** 2 / (1 - m.hat_diag_)[None, :]
                Zm = Zm / np.sqrt(hm / self.h_ref_)[:, :, None]
            LP = log_softmax(self.a_ * Zm, axis=2)
            S = _score(np.exp(LP), score)                             # c x n x K
            cnt = (s_loo[None, :, None] < S).sum(1)                   # c x K
            out[st:st + chunk] = cnt < thr
        return out


def _score(P, score):
    """Non-conformity scores for every class; P has classes on the last axis."""
    if score == "lac":
        return 1.0 - P
    if score == "aps":
        # deterministic APS: mass of all classes at least as probable as y
        order = np.argsort(-P, axis=-1, kind="stable")
        cum = np.cumsum(np.take_along_axis(P, order, -1), -1)
        out = np.empty_like(P)
        np.put_along_axis(out, order, cum, -1)
        return out
    raise ValueError(score)
