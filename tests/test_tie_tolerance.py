"""Model-pair differences of floating-point size are ties."""
import numpy as np
import pytest

from src import estimators as est
from src.intervals import Agreement


def test_tie_sign():
    d = np.array([0.0, 1e-13, -1e-13, 1e-9, 2e-9, -0.5])
    np.testing.assert_array_equal(est.tie_sign(d), [0, 0, 0, 0, 1, -1])


def test_floating_point_tie_counts_as_tie():
    # 100*a/n - 100*b/n can differ from 100*c/n - 100*d/n in the last bit when a - b = c - d
    n = 1319
    g1 = 100 * 2 / n - 100 * 1 / n
    g2 = 100 * 3 / n - 100 * 2 / n
    assert g1 != g2 and abs(g1 - g2) < 1e-12
    G = np.array([[g1, g2, 10.0], [g1, g2, 12.0], [g1, g2, 11.0]])
    r = est.pair_agreement(G)
    assert r["tie"] == pytest.approx(1 / 3)          # the first pair ties in every draw
    assert r["C"] == pytest.approx(r["same"] + 0.5 * r["tie"])
    a = Agreement(G)
    assert a.pos[0] == 0 and a.neg[0] == 0
