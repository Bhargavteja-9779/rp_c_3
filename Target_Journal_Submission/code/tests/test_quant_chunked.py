"""The chunked QUANT used in E20 must give the same probabilities as aeon's QUANTClassifier."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_chunked_quant_equals_quantclassifier():
    from aeon.classification.interval_based import QUANTClassifier
    from e20_ptbxl import quant_fit_predict
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 3, 200)).astype(np.float32); y = np.repeat(np.arange(3), 40)
    X[y == 1, :, 50:80] += 2.0; X[y == 2, 1] *= 2.5
    Xtr, ytr, Xte = X[::2], y[::2], X[1::2]
    ref = QUANTClassifier(random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    P, cls = quant_fit_predict(Xtr, ytr, Xte, seed=7, chunk=17)
    assert np.array_equal(cls, np.arange(3))
    np.testing.assert_allclose(P, ref, atol=1e-12)
