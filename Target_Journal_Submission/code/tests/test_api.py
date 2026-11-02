"""Tests of the user-facing API and the loom-triage command-line tool."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loom import LOOMClassifier            # noqa: E402
from loom.cli import main as cli          # noqa: E402


def _data(n=60, L=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, L)
    y = np.repeat([0, 1], n // 2)
    X = np.where(y[:, None] == 0, np.sin(6 * np.pi * t), np.sign(np.sin(6 * np.pi * t))) + 0.1 * rng.normal(size=(n, L))
    return X, y


def test_fit_predict_and_triage_flags_faulty_recordings():
    X, y = _data()
    Xt, yt = _data(seed=1)
    clf = LOOMClassifier().fit(X, y)
    P = clf.predict_proba(Xt)
    assert np.allclose(P.sum(1), 1) and (clf.predict(Xt) == yt).mean() > 0.9
    bad = Xt + 3 * np.random.default_rng(2).normal(size=Xt.shape)            # heavy sensor noise
    assert np.median(clf.unusualness(bad)) > 3 * np.median(clf.unusualness(Xt))
    q = clf.triage(np.r_[Xt, bad], budget=0.5)
    assert (q.decision.to_numpy()[len(Xt):] == "review").mean() > 0.8          # faulty half goes to review
    S = clf.predict_sets(Xt, alpha=0.1)
    assert S.shape == (len(Xt), 2) and S[np.arange(len(yt)), yt].mean() >= 0.8     # coverage near 1 - alpha


def test_input_validation():
    X, y = _data()
    clf = LOOMClassifier().fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(X[:, :50])
    with pytest.raises(ValueError):
        LOOMClassifier().fit(X, np.zeros(len(y)))


def test_cli_roundtrip(tmp_path):
    X, y = _data(); Xt, _ = _data(seed=1)
    np.savetxt(tmp_path / "train.csv", np.c_[y, X], delimiter=",")
    np.savetxt(tmp_path / "new.csv", Xt, delimiter=",")
    assert cli(["fit", "--train", str(tmp_path / "train.csv"), "--model", str(tmp_path / "m.pkl")]) == 0
    assert cli(["run", "--model", str(tmp_path / "m.pkl"), "--input", str(tmp_path / "new.csv"), "--budget", "0.25",
                "--out", str(tmp_path / "q.csv"), "--report", str(tmp_path / "r.html"), "--alpha", "0.1"]) == 0
    import pandas as pd
    q = pd.read_csv(tmp_path / "q.csv")
    assert len(q) == len(Xt) and (q.decision == "review").sum() == round(0.25 * len(Xt))
    assert "<svg" in (tmp_path / "r.html").read_text()
