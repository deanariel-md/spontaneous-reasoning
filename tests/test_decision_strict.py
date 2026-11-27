"""Tests of src/decision_strict.py (the strict decision comparison)."""
import numpy as np
import pytest

from src import decision as dc, decision_strict as ds, rule_budgets as rbud


def test_split_items_halves_each_stratum_and_depends_on_ids_only():
    ids = [f"q{i}" for i in range(10)]
    strata = [np.arange(0, 6), np.arange(6, 10)]
    m = ds.split_items(ids, "t", "lab", strata)
    assert m[:6].sum() == 3 and m[6:].sum() == 2
    # the same ids in another order give the same pilot pool
    perm = np.array([3, 1, 4, 0, 5, 2, 9, 7, 8, 6])
    m2 = ds.split_items([ids[i] for i in perm], "t", "lab", [np.arange(0, 6), np.arange(6, 10)])
    assert {ids[i] for i in perm[m2]} == {ids[i] for i in np.flatnonzero(m)}
    # another label gives another split, here at least
    assert not np.array_equal(ds.split_items(ids, "t", "other"), ds.split_items(ids, "t", "lab"))


def test_split_items_refuses_overlapping_strata():
    with pytest.raises(ValueError):
        ds.split_items(["a", "b", "c"], "t", "x", [np.array([0, 1]), np.array([1, 2])])


def test_eval_weights_keep_stratum_weight():
    mask = np.array([1, 0, 0, 0, 1, 1], bool)
    w = ds.eval_weights(6, mask, [np.array([0, 1, 2]), np.array([3, 4, 5])])
    assert w.sum() == pytest.approx(100)
    assert w[0] == pytest.approx(50) and w[4] == pytest.approx(25) and w[1] == 0
    w2 = ds.eval_weights(6, mask)
    assert np.allclose(w2[mask], 100 / 3)


def test_pilot_weights_match_rule_budgets():
    rng = np.random.default_rng(0)
    strata = [np.arange(0, 4), np.arange(4, 8)]
    pos, items = rbud.draw_pilots(5, 3, 1, rng, N=8, n=4, strata=strata)
    assert np.allclose(ds.pilot_weights(8, items, strata), rbud.subsample_weights(8, items, None, strata))
    pos, items = rbud.draw_pilots(5, 3, 1, rng, N=8, n=4)
    assert np.allclose(ds.pilot_weights(8, items), rbud.subsample_weights(8, items))


def test_draw_pilots_stay_in_pool():
    rng = np.random.default_rng(1)
    pool = np.array([1, 3, 5, 7, 9, 11])
    pos, items = ds.draw_pilots(20, 4, rng, pool, 3)
    assert np.isin(items, pool).all() and ((pos >= 0) & (pos < 4)).all()
    strata = [np.array([1, 3, 5]), np.array([7, 9, 11])]
    _, items = ds.draw_pilots(20, 4, rng, pool, 4, strata)
    assert all(np.isin(r[:2], strata[0]).all() and np.isin(r[2:], strata[1]).all() for r in items)


def test_fit_rules_own_gain_and_linear():
    b = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([30.0, 35.0, 40.0, 45.0])          # gain 20, 15, 10, 5: exactly linear
    e = np.array([1.0, 2.0, 3.0, 5.0])
    f = ds.fit_rules(b, p, e)
    assert np.allclose(f["own_gain"], [20, 15, 10, 5])
    assert np.allclose(f["linear"], [20, 15, 10, 5])
    assert f["quadratic_index"].shape == (4,)


def test_evaluate_perfect_rule_has_zero_plugin_regret_and_paired_sign():
    G = np.array([[5.0, 1.0, 0.0], [6.0, 0.0, 1.0], [4.0, 2.0, 0.0]])
    n = 1
    good = np.array([[1.0, 0, 0]])
    bad = np.array([[0, 0, 1.0]])
    out = ds.evaluate({"own_gain": good, "linear": bad}, G, n)
    assert out["rules"]["own_gain"]["regret_plugin"] == pytest.approx(0)
    assert out["rules"]["linear"]["regret_plugin"] == pytest.approx(5 - 1 / 3)
    # regret(own) - regret(linear) is negative: own gain costs less
    assert out["paired"]["own_gain_minus_linear"]["estimate"] == pytest.approx(-(5 - 1 / 3))
    assert out["rules"]["own_gain"]["plugin_interval"]["ci95"] is not None


def test_jackknife_regret_reports_too_few_configurations():
    G = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = ds.jackknife_regret(np.array([[1.0, 0.0]]), G, 1)
    assert r["ci95"] is None and r["n_configurations"] == 2
    assert r["estimate"] == pytest.approx(float(dc.regret_plugin(np.array([[1.0, 0.0]]), G, 1).mean()))
