"""Tests of src/transfer_regret.py."""
import numpy as np
import pytest

from src import decision as dc, transfer_regret as tr


def test_fit_apply_recovers_a_linear_rule():
    b = np.array([10.0, 20.0, 30.0, 40.0])
    g = 50 - b
    e = np.array([1.0, 3.0, 2.0, 5.0])
    p = tr.fit_apply(b, g, e, np.array([15.0, 50.0]), np.array([0.0, 0.0]), "baseline_linear")
    assert np.allclose(p, [35.0, 0.0])


def test_fit_apply_reports_too_few_training_models_as_nan():
    p = tr.fit_apply([1.0, 2.0], [0.0, 1.0], [0.0, 1.0], [1.5], [0.5], "baseline_quadratic_index")
    assert np.isnan(p).all()


def test_grouped_regret_pools_by_number_chosen_and_matches_paired():
    G = np.array([[5.0, 0.0, 1.0, 2.0, 3.0, 0.0], [4.0, 1.0, 0.0, 3.0, 2.0, 0.0], [6.0, 0.0, 0.0, 2.0, 3.0, 1.0]])
    groups = {"a": [0, 1, 2], "b": [3, 4, 5]}
    ns = {"a": 1, "b": 1}
    best = {"a": np.array([[1.0, 0, 0]]), "b": np.array([[0, 1.0, 0]])}
    worst = {"a": np.array([[0, 0, 1.0]]), "b": np.array([[0, 0, 1.0]])}
    rb_ = tr.grouped_regret(best, G, groups, ns)
    rw = tr.grouped_regret(worst, G, groups, ns)
    assert rb_["per_group"]["a"]["regret_plugin"] == pytest.approx(0)
    ga, gb = G[:, :3], G[:, 3:]
    want = 0.5 * (dc.regret_plugin(worst["a"], ga, 1)[0] + dc.regret_plugin(worst["b"], gb, 1)[0])
    assert rw["regret_plugin"] == pytest.approx(want)
    p = tr.paired_grouped(worst, best, G, groups)
    assert p["estimate"] == pytest.approx(rw["regret_plugin"] - rb_["regret_plugin"])


def test_third_weights_share_ties():
    W = tr.third_weights([[1.0, 1.0, 0.0]], 1)
    assert np.allclose(W, [[0.5, 0.5, 0.0]])
