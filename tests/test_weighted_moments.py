"""The equal-family-weighted slope (src/instability.weighted_moments) on fabricated arrays."""
import numpy as np
import pytest

from src import instability as ins


def arrays(M=7, K=12, seed=3):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(20, 80, M)
    X = theta[None, :] + rng.normal(0, 2, (K, M))
    Y = 30 - 0.4 * theta[None, :] + rng.normal(0, 3, (K, M))
    return X, Y


@pytest.mark.parametrize("method", ["fieller", "jack_t"])
def test_equal_weights_equal_the_model_weighted_moments(method):
    X, Y = arrays()
    a, b = ins.scale_moments(X, Y, method), ins.weighted_moments(X, Y, np.ones(X.shape[1]), method)
    for k in ("V_x", "A", "V_y", "slope"):
        assert b[k] == pytest.approx(a[k], rel=1e-12)
    assert b["slope_set"]["kind"] == a["slope_set"]["kind"]
    assert b["slope_set"]["a"] == pytest.approx(a["slope_set"]["a"], rel=1e-10)
    assert b["slope_set"]["b"] == pytest.approx(a["slope_set"]["b"], rel=1e-10)


def test_family_weights_count_each_family_once():
    w = ins.family_weights(["a", "a", "b", "c", "c", "c"])
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(1 / 6) and w[2] == pytest.approx(1 / 3) and w[3] == pytest.approx(1 / 9)


def test_a_duplicated_model_in_its_own_family_counts_once():
    X, Y = arrays(M=4)
    Xd, Yd = np.column_stack([X, X[:, 3]]), np.column_stack([Y, Y[:, 3]])
    fam = ["a", "b", "c", "d", "d"]
    a, b = ins.scale_moments(X, Y), ins.weighted_moments(Xd, Yd, ins.family_weights(fam))
    assert b["slope"] == pytest.approx(a["slope"], rel=1e-12)
    assert b["slope_set"]["a"] == pytest.approx(a["slope_set"]["a"], rel=1e-10)
    assert b["slope_set"]["b"] == pytest.approx(a["slope_set"]["b"], rel=1e-10)


def test_bad_weights_refuse():
    X, Y = arrays(M=3)
    with pytest.raises(ValueError):
        ins.weighted_moments(X, Y, [1, 0, 1])
