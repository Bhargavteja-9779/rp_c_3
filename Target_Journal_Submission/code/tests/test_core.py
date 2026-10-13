"""Unit tests: closed-form LOO identities, calibration, conformal sets, corruptions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loom.models.readout import ClosedFormRidge, _encode  # noqa: E402
from loom.models.loom import LOOM  # noqa: E402
from loom.models.calibration import fit_moderation, soft_targets  # noqa: E402
from loom.evaluation.metrics import ece, nll, aurc, set_metrics  # noqa: E402
from loom.preprocessing.corruptions import corrupt, all_conditions  # noqa: E402
from loom.statistics.tests import holm, rank_biserial  # noqa: E402


def _brute(m, X, y, Xt):
    Xs = (X - m.mu_) / m.sd_; Pt = (Xt - m.mu_) / m.sd_
    n, p = Xs.shape; lam = m.lambda_ * n
    Y = _encode(y, m.classes_)
    Pen = np.eye(p + 1) * lam; Pen[0, 0] = 0
    loo, lev = [], []
    for i in range(n):
        idx = np.r_[0:i, i + 1:n]
        Za = np.c_[np.ones(n - 1), Xs[idx]]
        A = Za.T @ Za + Pen
        zi = np.r_[1, Xs[i]]
        loo.append(zi @ np.linalg.solve(A, Za.T @ Y[idx])); lev.append(zi @ np.linalg.solve(A, zi))
    Za = np.c_[np.ones(n), Xs]; A = Za.T @ Za + Pen; zt = np.c_[np.ones(len(Pt)), Pt]
    return np.array(loo), np.array(lev), np.einsum("ij,ij->i", zt, np.linalg.solve(A, zt.T).T), zt @ np.linalg.solve(A, Za.T @ Y), zt @ np.linalg.solve(A, Za.T)


@pytest.mark.parametrize("n,p,K", [(25, 120, 3), (60, 15, 2), (40, 40, 4)])
def test_closed_form_matches_refits(n, p, K):
    rng = np.random.default_rng(n + p)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3, p); y = rng.integers(0, K, n)
    Xt = rng.normal(size=(5, p))
    m = ClosedFormRidge().fit(X, y)
    for lam in (m.lambdas[0], m.lambdas[5]):
        m.set_lambda(lam)
        loo, lev, h, Z, Kx = _brute(m, X, y, Xt)
        Zt, ht, Q = m.logits_and_leverage(Xt)
        assert np.allclose(m.loo_logits_, loo, atol=1e-8)
        assert np.allclose(m.loo_leverage_, lev, atol=1e-8)
        assert np.allclose(ht, h, rtol=1e-8)
        assert np.allclose(Zt, Z, atol=1e-8)
        assert np.allclose(m.cross_leverage(Q), Kx, atol=1e-8)


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4000, 3)); a_true = 2.5
    P = np.exp(a_true * Z); P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=pp) for pp in P])
    a, b, _ = fit_moderation(Z, np.ones(len(y)), y, fit_b=False)
    assert abs(a - a_true) / a_true < 0.1 and b == 0.0


def test_platt_targets_binary():
    T = soft_targets(np.array([0, 0, 0, 1]), 2, True)
    assert np.allclose(T[0], [4 / 5, 1 / 5]) and np.allclose(T[3], [1 / 3, 2 / 3])
    assert np.allclose(T.sum(1), 1)


def test_loom_standard_logit_and_shift_shrinkage():
    rng = np.random.default_rng(1)
    X = np.r_[rng.normal(0, 1, (60, 50)), rng.normal(0.8, 1, (60, 50))]; y = np.r_[np.zeros(60), np.ones(60)]
    m = LOOM().fit(X, y)
    Xt = np.r_[rng.normal(0, 1, (20, 50)), rng.normal(0.8, 1, (20, 50))]
    P_in = m.predict_proba(Xt); P_out = m.predict_proba(Xt * 8.0)           # far-away points
    assert P_out.max(1).mean() < P_in.max(1).mean()
    assert np.allclose(P_in.sum(1), 1)


def test_jackknife_plus_matches_explicit_refits():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 40)); y = rng.integers(0, 3, 30); Xt = rng.normal(size=(4, 40))
    m = LOOM("standard").fit(X, y)
    S = m.conformal_sets(Xt, alpha=0.2, method="jackknife+")
    # explicit: refit ridge without i (same lambda, same standardisation), same a, h_ref
    r = m.ridge_; n = len(y); Y = r.Y_; Xs = (X - r.mu_) / r.sd_; Pt = (Xt - r.mu_) / r.sd_
    lam = r.lambda_ * n; Pen = np.eye(41) * lam; Pen[0, 0] = 0
    yi = np.searchsorted(m.classes_, y)
    s_loo = 1 - np.exp(m.loo_log_proba()[np.arange(n), yi])
    cnt = np.zeros((4, 3))
    for i in range(n):
        idx = np.r_[0:i, i + 1:n]; Za = np.c_[np.ones(n - 1), Xs[idx]]; A = Za.T @ Za + Pen
        B = np.linalg.solve(A, Za.T @ Y[idx]); zt = np.c_[np.ones(4), Pt]
        Zt = zt @ B; ht = np.einsum("ij,ij->i", zt, np.linalg.solve(A, zt.T).T)
        L = m.a_ * Zt / np.sqrt(ht / m.h_ref_)[:, None]; L -= L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        cnt += (s_loo[i] < 1 - P)
    assert np.array_equal(S, cnt < 0.8 * (n + 1))


def test_jackknife_coverage_gaussian_mixture():
    rng = np.random.default_rng(5); covs = []
    for rep in range(20):
        mu = rng.normal(0, 1.2, (3, 20))
        y = rng.integers(0, 3, 300); X = mu[y] + rng.normal(size=(300, 20))
        m = LOOM().fit(X[:100], y[:100])
        covs.append(set_metrics(m.conformal_sets(X[100:], alpha=0.1, method="jackknife+"), np.searchsorted(m.classes_, y[100:]))["coverage"])
    assert np.mean(covs) >= 0.85        # jackknife+ guarantees >= 1 - 2 alpha; in practice ~ 1 - alpha


def test_metrics_basic():
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]); y = np.array([0, 1, 1])
    assert abs(nll(P, y) - (-np.log([0.9, 0.8, 0.4]).mean())) < 1e-12
    assert 0 <= ece(P, y) <= 1 and 0 <= aurc(P, y) <= 1


def test_corruptions_deterministic_and_shape():
    X = np.sin(np.linspace(0, 6, 100))[None].repeat(5, 0)
    for fam, sev in all_conditions():
        if fam == "clean":
            continue
        A = corrupt(X, fam, sev, 7); B = corrupt(X, fam, sev, 7)
        assert A.shape == X.shape and np.array_equal(A, B) and np.isfinite(A).all()


def test_stats_helpers():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert rank_biserial([1, 2, 3]) == 1.0 and rank_biserial([-1, -2]) == -1.0


def test_aps_score_and_sets():
    from loom.models.loom import _score
    P = np.array([[0.5, 0.3, 0.2], [1 / 3, 1 / 3, 1 / 3]])
    S = _score(P, "aps")
    assert np.allclose(S[0], [0.5, 0.8, 1.0]) and np.allclose(np.sort(S[1]), [1 / 3, 2 / 3, 1.0])
    rng = np.random.default_rng(9)
    mu = rng.normal(0, 1.2, (3, 20)); y = rng.integers(0, 3, 300); X = mu[y] + rng.normal(size=(300, 20))
    m = LOOM().fit(X[:100], y[:100])
    for sc in ("lac", "aps"):
        Sj = m.conformal_sets(X[100:], alpha=0.1, method="jackknife+", score=sc)
        assert Sj.shape == (200, 3)
