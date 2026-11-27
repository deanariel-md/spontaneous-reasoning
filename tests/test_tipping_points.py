"""Tests for the tipping points (src/tipping_points.py) on fabricated cells and arrays."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from src import instability as ins
from src import tipping_points as tp
from src.estimators import cross_moment

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_tp_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def layout(models, configs, arms=("B", "P")):
    keys = [(m, a, c) for m in models for a in arms for c in sorted(set(configs))]
    return keys, {a: tp.cell_index(keys, models, a, configs) for a in arms}


def mean_problem(correct, n, keys, idx, bounds="unrestricted", here_only=None, there_only=None):
    down, up = tp.unrestricted(correct, n) if bounds == "unrestricted" else tp.restricted(here_only, there_only)
    return tp.at_or_below_zero_problem(correct, n, down, up, idx["B"], idx["P"], tp.mean_gain, "mean")


# ----------------------------------------------------------------------------- estimators


def test_cross_moment_grad_matches_appendix_a_and_finite_differences():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(50, 10, (7, 5)), rng.normal(0, 3, (7, 5))
    est, dX, dY = tp.cross_moment_grad(X, Y)
    assert est == pytest.approx(cross_moment(X, Y), abs=1e-12)
    h = 1e-6
    for (u, m) in ((0, 0), (3, 2), (6, 4)):
        Xh = X.copy()
        Xh[u, m] += h
        Yh = Y.copy()
        Yh[u, m] += h
        assert (cross_moment(Xh, Y) - est) / h == pytest.approx(dX[u, m], rel=1e-4, abs=1e-9)
        assert (cross_moment(X, Yh) - est) / h == pytest.approx(dY[u, m], rel=1e-4, abs=1e-9)


def test_slopes_reproduce_the_existing_estimators():
    """The baseline and index slopes equal src/instability.scale_moments and run_index_checks.slope_block (the path of
    scripts/run_outcomes.py) on fabricated arrays with a configuration drawn twice, and their gradients equal finite
    differences."""
    rrc = load_script("run_index_checks")
    rng = np.random.default_rng(1)
    K, M = 24, 9
    theta = rng.uniform(20, 70, M)
    B = theta + rng.normal(0, 2, (K, M))
    P = B + 0.2 * theta + rng.normal(0, 3, (K, M))
    B[5], P[5] = B[2], P[2]                                    # one configuration drawn twice
    e = rng.normal(130, 10, M)
    ref = rrc.slope_block(B, P, 1319, e, "fieller", 0.05)["scales"]["points"]
    s, dB, dP = tp.baseline_slope(B, P)
    assert s == pytest.approx(ins.scale_moments(B, P - B)["slope"], abs=1e-13)
    assert s == pytest.approx(ref["baseline"]["slope"], abs=1e-13)
    si, dBi, dPi = tp.index_slope(B, P, e)
    assert si == pytest.approx(ref["index"]["slope"], abs=1e-13)
    h = 1e-6
    for (u, m) in ((0, 0), (5, 3), (23, 8)):
        Bh, Ph = B.copy(), P.copy()
        Bh[u, m] += h
        Ph[u, m] += h
        assert (tp.baseline_slope(Bh, P)[0] - s) / h == pytest.approx(dB[u, m], rel=1e-4, abs=1e-9)
        assert (tp.baseline_slope(B, Ph)[0] - s) / h == pytest.approx(dP[u, m], rel=1e-4, abs=1e-9)
        assert (tp.index_slope(Bh, P, e)[0] - si) / h == pytest.approx(dBi[u, m], rel=1e-4, abs=1e-9)


def test_tercile_reading_reproduces_run_exemplar():
    rex = load_script("run_exemplar")
    rng = np.random.default_rng(2)
    K, M = 24, 29
    rank = rng.uniform(20, 95, M)
    Bv = rank + rng.normal(0, 2, (K, M))
    Pv = Bv + np.where(rank < 50, 5.0, 0.1) + rng.normal(0, 1, (K, M))
    panel = [f"m{i}" for i in range(M)]
    ref = rex.tercile_block(panel, rank, Bv, Pv)
    got = tp.tercile_reading(rank, Bv, Pv, rex.EQ_MARGIN)
    for g in ("weak", "strong"):
        assert [panel[i] for i in got["members"][g]] == ref[g]["models"]
        assert got[g]["mean"] == pytest.approx(ref[g]["mean"], abs=1e-12)
        assert got[g]["ci95"] == pytest.approx(ref[g]["ci95"], abs=1e-12)
        assert got[g]["ci90"] == pytest.approx(ref[g]["ci90"], abs=1e-12)
    for k in ("weak_benefits", "strong_equivalent", "claim_holds"):
        assert got[k] == ref[k]


def test_t_bound_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    d = rng.normal(1, 0.5, 24)
    q = tp.t_quantile(0.95, 24)
    for side in (-1, 1):
        b, g = tp.t_bound(d, q, side)
        assert b == pytest.approx(tp.t_interval(d, 0.95)[0 if side < 0 else 1], abs=1e-12)
        dh = d.copy()
        dh[4] += 1e-6
        assert (tp.t_bound(dh, q, side)[0] - b) / 1e-6 == pytest.approx(g[4], rel=1e-4)


# ----------------------------------------------------------------------------- the greedy search


def test_greedy_equals_the_exact_minimum_for_a_mean():
    """Brute force over every net-flip vector of a small panel: the greedy count is the minimum."""
    models, configs, n = ["a", "b"], ["c1", "c1", "c2"], 2
    keys, idx = layout(models, configs)
    correct = np.array([0, 1, 2, 1, 1, 0, 0, 2])            # cells in key order: (model, arm, config)
    p = mean_problem(correct, n, keys, idx)
    start = p.value(p.scores())["mean"]
    assert start > 0
    best = None
    ranges = [range(-int(c), n - int(c) + 1) for c in correct]
    for k in itertools.product(*ranges):
        k = np.array(k)
        if p.value(p.scores(k))["mean"] <= tp.TIE_TOL:
            cnt = int(np.abs(k).sum())
            best = cnt if best is None else min(best, cnt)
    r = tp.greedy(p)
    assert r["status"] == "reversed" and r["count"] == best


def test_greedy_equals_the_sorted_minimum_on_a_larger_panel():
    rng = np.random.default_rng(4)
    models = [f"m{i}" for i in range(12)]
    configs = [f"c{j}" for j in rng.integers(0, 15, 24)]
    n = 200
    keys, idx = layout(models, configs)
    acc_b = rng.uniform(0.2, 0.8, len(models))
    correct = np.array([rng.binomial(n, min(0.98, acc_b[models.index(m)] + (0.12 if a == "P" else 0)))
                        for m, a, c in keys])
    p = mean_problem(correct, n, keys, idx)
    d0, g = p.distance(p.scores())
    want = tp.exact_minimum_linear(-g * 100.0 / n, p.down, p.up, d0)
    r = tp.greedy(p)
    assert r["status"] == "reversed" and r["count"] == want
    assert r["value_end"]["mean"] <= tp.TIE_TOL
    k = r["k"].copy()
    k[np.flatnonzero(k)[0]] -= np.sign(k[np.flatnonzero(k)[0]])      # one flip fewer does not reverse
    assert p.value(p.scores(k))["mean"] > tp.TIE_TOL


def test_configuration_drawn_twice_doubles_the_effect_of_a_flip():
    models, configs, n = ["a", "b", "c"], ["c1", "c1", "c2"], 50
    keys, idx = layout(models, configs)
    correct = np.array([25 if a == "B" else 40 for m, a, c in keys])
    p = mean_problem(correct, n, keys, idx)
    _, g = p.distance(p.scores())
    e = {keys[i]: g[i] for i in range(len(keys))}
    assert e[("a", "P", "c1")] == pytest.approx(2 * e[("a", "P", "c2")])
    assert e[("b", "B", "c1")] == pytest.approx(2 * e[("b", "B", "c2")])
    assert e[("a", "P", "c1")] * 100.0 / n == pytest.approx(100.0 / n * 2 / (3 * 3))   # pp per flip x 2 rows / (K M)
    r = tp.greedy(p)
    # the six c1 cells hold 6 x (40 + 25) = 390 helpful flips worth 2 rows each against 2 x 3 x 15 = 90 pp-rows of gain
    hit = [keys[i] for i in np.flatnonzero(r["k"])]
    assert r["status"] == "reversed" and all(c == "c1" for _, _, c in hit)
    assert r["count"] == tp.exact_minimum_linear(-g * 100.0 / n, p.down, p.up, p.distance(p.scores())[0])


def test_availability_limits_the_flips_in_each_cell():
    models, configs, n = ["a", "b"], ["c1", "c2"], 10
    keys, idx = layout(models, configs)
    # intervention arm nearly all correct, baseline arm nearly all incorrect: gain 80 pp per cell
    correct = np.array([1 if a == "B" else 9 for m, a, c in keys])
    p = mean_problem(correct, n, keys, idx)
    r = tp.greedy(p)
    assert r["status"] == "reversed"
    assert np.all(r["k"] >= -p.down) and np.all(r["k"] <= p.up)
    # only one correct output per cell may go in each arm: a gain of 8 items per cell needs 4 + 4 flips per cell
    assert r["count"] == 4 * 8
    # with flips restricted to one output per cell the result cannot be reversed, and that is reported
    lim = tp.at_or_below_zero_problem(correct, n, np.minimum(p.down, 1), np.minimum(p.up, 1), idx["B"], idx["P"],
                                      tp.mean_gain, "mean")
    r2 = tp.greedy(lim)
    assert r2["status"] == "not_reversed" and r2["distance_end"] > 0
    assert np.all(np.abs(r2["k"]) <= 1)


def test_restricted_count_moves_only_disagreeing_outputs():
    """Per-output scores under two scorers: the restricted search flips only outputs where they disagree, each to the
    other scorer's reading, and needs at least as many flips as the unrestricted one."""
    rng = np.random.default_rng(5)
    models, configs, n = ["a", "b", "c", "d"], ["c1", "c2", "c2", "c3"], 60
    keys, idx = layout(models, configs)
    here = {key: rng.binomial(1, 0.7 if key[1] == "P" else 0.4, n) for key in keys}
    there = {key: np.where(rng.random(n) < 0.15, 1 - here[key], here[key]) for key in keys}
    correct = np.array([here[key].sum() for key in keys])
    here_only = np.array([int(((here[key] == 1) & (there[key] == 0)).sum()) for key in keys])
    there_only = np.array([int(((here[key] == 0) & (there[key] == 1)).sum()) for key in keys])
    pu = mean_problem(correct, n, keys, idx)
    pr = mean_problem(correct, n, keys, idx, "restricted", here_only, there_only)
    ru, rr = tp.greedy(pu), tp.greedy(pr)
    assert ru["status"] == "reversed"
    assert np.all(rr["k"] >= -here_only) and np.all(rr["k"] <= there_only)
    if rr["status"] == "reversed":
        assert rr["count"] >= ru["count"]
    # a down flip in the restricted search is an output correct here and incorrect there: moving all of them gives the
    # other scorer's count in every cell
    full = correct - here_only + there_only
    assert np.array_equal(full, np.array([there[key].sum() for key in keys]))
    comp = tp.composition(rr["k"], keys, here_only, there_only)
    assert comp["count"] == rr["count"]
    assert comp["disagreeing_in_touched_cells"] <= int(here_only.sum() + there_only.sum())


def test_index_slope_search_is_exact():
    rng = np.random.default_rng(6)
    models = [f"m{i}" for i in range(8)]
    configs = ["c1", "c2", "c2", "c3", "c4", "c5"]
    n = 100
    keys, idx = layout(models, configs)
    e = np.linspace(100, 150, 8)
    correct = np.array([rng.binomial(n, 0.4 if a == "B" else min(0.95, 0.3 + 0.004 * e[models.index(m)]))
                        for m, a, c in keys])
    p = tp.at_or_below_zero_problem(correct, n, *tp.unrestricted(correct, n), idx["B"], idx["P"],
                                    lambda B, P: tp.index_slope(B, P, e), "slope")
    d0, g = p.distance(p.scores())
    assert d0 > 0
    r = tp.greedy(p)
    assert r["status"] == "reversed"
    assert r["count"] == tp.exact_minimum_linear(-g * 100.0 / n, p.down, p.up, d0)


def test_baseline_slope_search_reverses_and_is_tight():
    rng = np.random.default_rng(7)
    models = [f"m{i}" for i in range(10)]
    configs = ["c1", "c2", "c2", "c3", "c4", "c5", "c6", "c7"]
    n = 150
    keys, idx = layout(models, configs)
    th = np.linspace(0.2, 0.8, 10)
    correct = np.array([rng.binomial(n, th[models.index(m)] * (1.0 if a == "B" else 1.15)) for m, a, c in keys])
    p = tp.at_or_below_zero_problem(correct, n, *tp.unrestricted(correct, n), idx["B"], idx["P"], tp.baseline_slope,
                                    "slope")
    assert p.value(p.scores())["slope"] > 0
    r = tp.greedy(p)
    assert r["status"] == "reversed" and r["value_end"]["slope"] <= tp.TIE_TOL
    assert np.all(r["k"] >= -p.down) and np.all(r["k"] <= p.up)


def test_tercile_routes_make_the_claim_fail_and_hold():
    rng = np.random.default_rng(8)
    M, n = 9, 120
    models = [f"m{i}" for i in range(M)]
    configs = [f"c{j}" for j in (1, 2, 2, 3, 4, 5, 6, 7, 8, 9)]
    keys, idx = layout(models, configs)
    th = np.linspace(0.3, 0.9, M)
    lift = np.where(th < 0.5, 0.15, 0.0)
    correct = np.array([rng.binomial(n, th[models.index(m)] + (lift[models.index(m)] if a == "P" else 0))
                        for m, a, c in keys])
    rank = th * 100
    fixed = lambda v: rank                                                       # noqa: E731
    v0 = tp.scores(correct, n)
    start = tp.tercile_reading(rank, v0[idx["B"]], v0[idx["P"]], 2.0)
    assert start["claim_holds"]
    counts = {}
    for route in ("weak_lower", "strong_upper", "strong_lower"):
        p = tp.tercile_problem(correct, n, *tp.unrestricted(correct, n), idx["B"], idx["P"], fixed, route, 2.0)
        r = tp.greedy(p)
        assert r["status"] == "reversed" and not r["value_end"]["claim_holds"]
        counts[route] = r["count"]
    assert counts["weak_lower"] > 0 and counts["strong_upper"] > 0
    # the failing reading: the weak tercile's benefit removed; route hold brings it back
    flat = np.array([c if a == "B" else correct[keys.index((m, "B", cc))] for (m, a, cc), c in zip(keys, correct)])
    p = tp.tercile_problem(flat, n, *tp.unrestricted(flat, n), idx["B"], idx["P"], fixed, "hold", 2.0)
    assert not p.value(p.scores())["claim_holds"]
    r = tp.greedy(p)
    assert r["status"] == "reversed" and r["value_end"]["claim_holds"]
    hit = {keys[i][0] for i in np.flatnonzero(r["k"])}
    weak = {models[i] for i in tp.tercile_members(rank)["weak"]}
    assert hit <= weak
