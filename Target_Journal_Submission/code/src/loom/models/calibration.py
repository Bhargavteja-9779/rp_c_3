"""Leverage-moderated softmax calibrators fitted on exact LOO pairs.

    p(y | x) = softmax( a * z(x) / sqrt(1 + b * h(x)) )

``a`` (> 0) is a global inverse temperature and ``b`` (>= 0) controls how much
the logits are shrunk towards uniform as the leverage ``h`` grows (the
"moderated output" of a Bayesian linear model, MacKay 1992).  ``b = 0`` is the
global scaling used by PreVal (Dempster et al.).  Both parameters are fitted by
minimising the negative log-likelihood on the n exact LOO pairs
(z_{-i}, h_{-i}), which are genuinely out-of-sample for the ridge fit.

Optional Platt (1999) target smoothing replaces the one-hot target of an example
of class k by (n_k + 1)/(n_k + 2) for the true class, spreading the remainder
uniformly over the other classes; for two classes this is exactly Platt's rule.
It keeps the fitted temperature finite when the LOO predictions are separable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax


def moderated_log_proba(Z, h, a, b):
    scale = a / np.sqrt(1.0 + b * np.asarray(h))
    return log_softmax(Z * scale[:, None], axis=1)


def soft_targets(yidx, K, smooth):
    n = len(yidx)
    T = np.zeros((n, K))
    if not smooth:
        T[np.arange(n), yidx] = 1.0
        return T
    nk = np.bincount(yidx, minlength=K)
    eps = 1.0 / (nk[yidx] + 2.0)
    T += (eps / max(K - 1, 1))[:, None]
    T[np.arange(n), yidx] = 1.0 - eps
    return T


def _nll_and_grad(theta, Z, h, T, fit_b):
    a = np.exp(theta[0])
    b = np.exp(theta[1]) if fit_b else 0.0
    root = np.sqrt(1.0 + b * h)
    S = Z * (a / root)[:, None]
    L = log_softmax(S, axis=1)
    n = len(T)
    nll = -(T * L).sum(1).mean()
    P = np.exp(L) - T                                 # dNLL/dS (times n); rows of T sum to 1
    dS_da = Z / root[:, None]                         # dS/da
    g_a = (P * dS_da).sum() / n * a                   # chain rule for log a
    grad = [g_a]
    if fit_b:
        dS_db = -0.5 * Z * (a * h / root ** 3)[:, None]
        grad.append((P * dS_db).sum() / n * b)
    else:
        grad.append(0.0)
    return nll, np.array(grad)


def fit_moderation(Z, h, yidx, fit_b=True, b_fixed=0.0, b_init=1e-2, smooth=False):
    """Return (a, b, nll) minimising LOO NLL.

    fit_b=True : a and b are both free (b >= 0).
    fit_b=False: b is held at ``b_fixed`` (0 -> PreVal-style global scaling) and
                 only the inverse temperature a is fitted.
    """
    best = None
    a_inits = [1.0, 5.0, 25.0]
    T = soft_targets(np.asarray(yidx), Z.shape[1], smooth)
    if fit_b:
        for a0 in a_inits:
            for b0 in [b_init, 1.0, 100.0]:
                r = minimize(_nll_and_grad, np.log([a0, b0]), args=(Z, h, T, True), jac=True,
                             method="L-BFGS-B", bounds=[(-10, 12), (-25, 25)])
                if best is None or r.fun < best.fun:
                    best = r
        return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.fun)
    # fixed b: fold the moderation into the logits and fit a only
    Zm = Z / np.sqrt(1.0 + b_fixed * np.asarray(h))[:, None]
    for a0 in a_inits:
        r = minimize(_nll_and_grad, np.log([a0, 1.0]), args=(Zm, np.zeros(len(h)), T, False), jac=True,
                     method="L-BFGS-B", bounds=[(-10, 12), (-25, 25)])
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(b_fixed), float(best.fun)
