"""Tests of src/robustness.py."""
import math

import numpy as np
import pytest

from src import estimators as d1
from src import intervals as iv
from src import robustness as rb


def _arrays(K=12, M=8, slope=0.5, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(50, 15, M)
    B = theta[None, :] + rng.normal(0, 2, (K, M))
    G = 10 + slope * (theta - theta.mean())[None, :] + rng.normal(0, 2, (K, M))
    return B, B + G


def test_cross_slope_matches_cross_moments():
    B, P = _arrays()
    s, vx = rb.cross_slope(B, P - B)
    assert vx == pytest.approx(iv.CrossMoment(B, B).estimate())
    assert s == pytest.approx(iv.CrossMoment(P - B, B).estimate() / vx)
    assert s == pytest.approx(d1.cross_moment(P - B, B) / d1.cross_moment(B, B))


def test_cross_slope_edge_cases_reported_as_nan():
    B = np.tile(np.arange(4.0), (5, 1)) * 0 + 7.0          # no spread: V_x = 0
    s, vx = rb.cross_slope(B, B)
    assert math.isnan(s) and vx == 0
    s, vx = rb.cross_slope(np.ones((5, 1)), np.ones((5, 1)))
    assert math.isnan(s) and math.isnan(vx)


def test_resample_columns_cluster_keeps_whole_clusters():
    rng = np.random.default_rng(1)
    labels = ["a", "a", "b", "c", "c", "c"]
    for ix in rb.resample_columns(6, labels, 50, rng):
        drawn = [labels[i] for i in ix]
        for c in set(drawn):
            # every drawn copy of a cluster brings all of its models
            assert drawn.count(c) % labels.count(c) == 0
    with pytest.raises(ValueError):
        rb.resample_columns(5, labels, 1, rng)


def test_bootstrap_statistic_counts_undefined():
    rng = np.random.default_rng(2)
    calls = iter(range(10 ** 6))

    def stat(ix):
        return float("nan") if next(calls) % 4 == 0 else 1.0
    out = rb.bootstrap_statistic(stat, 5, None, 100, rng)
    assert out["n_undefined"] == 25
    assert out["ci95"] == [1.0, 1.0]
    assert out["share_below_zero"] == 0.0


def test_bootstrap_slope_covers_true_slope_in_simple_case():
    B, P = _arrays(K=24, M=40, slope=0.5, seed=3)
    rng = np.random.default_rng(4)
    out = rb.bootstrap_statistic(lambda ix: rb.cross_slope(B[:, ix], (P - B)[:, ix])[0], 40, None, 2000, rng)
    lo, hi = out["ci95"]
    assert lo < 0.5 < hi


def test_pair_reversals_counts_and_thresholds():
    g1 = np.array([0.0, 10.0, 20.0])
    g2 = np.array([0.0, 30.0, 25.0])        # pair (1, 2) reverses with gaps 10 and 5
    out = rb.pair_reversals(g1, g2, thresholds=(0.0, 5.0, 9.0))
    assert out["pairs"] == 3 and out["reversed"] == 1
    bt = {b["min_gap_pp"]: b for b in out["by_threshold"]}
    assert bt[0.0]["reversed"] == 1
    assert bt[5.0]["reversed"] == 0          # min gap 5 is not above 5
    assert bt[9.0]["pairs_both_above"] == 2


def test_pair_reversals_ties_are_neither():
    out = rb.pair_reversals(np.array([1.0, 1.0]), np.array([0.0, 2.0]))
    assert out["reversed"] == 0 and out["ties"] == 1


def test_heldout_predictions_constant_is_mean_of_others():
    b = np.array([10.0, 20.0, 30.0, 40.0])
    g = np.array([1.0, 2.0, 3.0, 6.0])
    e = np.array([100.0, 110.0, 120.0, 130.0])
    groups = {str(j): [j] for j in range(4)}
    p = rb.heldout_predictions(b, g, e, groups, "constant")
    assert p == pytest.approx([(2 + 3 + 6) / 3, (1 + 3 + 6) / 3, (1 + 2 + 6) / 3, 2.0])
    lin = rb.heldout_predictions(b, 0.1 * b, e, groups, "baseline_linear")
    assert lin == pytest.approx(0.1 * b)    # an exact line is recovered for every held-out model


def test_heldout_predictions_skip_underdetermined_fits():
    b, g, e = np.arange(4.0), np.arange(4.0), np.arange(4.0)
    p = rb.heldout_predictions(b, g, e, {"x": [0, 1]}, "baseline_quadratic_index")
    assert np.isnan(p).all()                # 4 columns, 2 training models: no fit, reported as nan


def test_per_model_cross_risk_matches_estimator():
    rng = np.random.default_rng(5)
    Z = rng.normal(0, 3, (10, 6))
    f = rng.normal(0, 1, 6)
    per = rb.per_model_cross_risk(Z, f)
    assert per.mean() * 6 == pytest.approx(d1.cross_risk(Z, f) * 6)
    # can be negative: never truncated
    Z2 = np.array([[1.0], [-1.0]])
    assert rb.per_model_cross_risk(Z2, np.array([0.0]))[0] == pytest.approx(-1.0)


def test_one_draw_c_and_reference():
    rng = np.random.default_rng(6)
    G = rng.normal(0, 1, (8, 5)) + np.arange(5) * 3
    assert rb.one_draw_c(G) == pytest.approx(d1.pair_agreement(G)["C"])
    assert rb.gaussian_reference(0.5, 1) == pytest.approx(0.5 + math.asin(0.5) / math.pi)
    assert rb.gaussian_reference(-0.1) is None and rb.gaussian_reference(float("nan")) is None


def test_k_draw_agreement_perfect_and_too_many():
    G = np.tile(np.arange(5.0), (8, 1))
    rng = np.random.default_rng(7)
    assert rb.k_draw_agreement(G, 2, rng, 10) == pytest.approx(1.0)
    assert rb.k_draw_agreement(G, 5, rng, 10) is None
    flat = np.zeros((8, 4))
    assert rb.k_draw_agreement(flat, 2, rng, 10) == pytest.approx(0.5)    # every pair tied


def test_gsm8k_steps_and_bins():
    assert rb.gsm8k_steps("a <<1+1=2>>2\nb <<2*3=6>>6\n#### 6") == 2
    assert rb.gsm8k_steps("first line\nsecond line\n#### 3") == 2
    assert rb.step_bins([1, 2, 3, 4, 5, 6, 9]) == ["<=2", "<=2", "3", "4", "5", ">=6", ">=6"]


def test_least_squares_trend():
    assert rb.least_squares_trend([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert math.isnan(rb.least_squares_trend([1.0, float("nan")]))


def test_generation_and_ladders():
    assert rb.generation("google/gemma-2b-it") == "Gemma-1"
    assert rb.generation("google/gemma-3-12b-it") == "Gemma-3"
    assert rb.generation("meta-llama/Meta-Llama-3-8B-Instruct") == "Llama-3"
    assert rb.generation("meta-llama/Llama-3.1-8B-Instruct") == "Llama-3.1"
    assert rb.generation("Qwen/Qwen3.5-35B-A3B") == "Qwen3.5"
    assert rb.generation("microsoft/Phi-3-mini-4k-instruct") == "Phi-3"
    with pytest.raises(ValueError):
        rb.generation("org/unknown-model")
    repos = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B", "google/gemma-3-4b-it", "microsoft/phi-4"]
    params = [8.2, 14.8, 30.5, 4.3, 14.7]
    keys = ["q8", "q14", "q30", "g4", "p4"]
    assert rb.ladders(repos, params, keys) == {"Qwen3": ["q8", "q14"]}
    assert rb.ladders(repos, params, keys, include_qwen3_mixtures=True) == {"Qwen3": ["q8", "q14", "q30"]}


def test_per_draw_ols_slope_and_jack_mean():
    x = np.array([1.0, 2.0, 3.0])
    G = np.array([[0.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    assert rb.per_draw_ols_slope(G, x) == pytest.approx([2.0, 1.0])
    j = rb.jack_mean([1.0, 2.0, 3.0, 4.0])
    assert j["estimate"] == pytest.approx(2.5)
    assert j["sign"] == "above zero"
    assert rb.interval_sign(None) == "undefined"


def test_fmt_edge_cases():
    assert rb.fmt(None) == "undefined"
    assert rb.fmt(float("nan")) == "undefined"
    assert rb.fmt(float("inf")) == "inf"
    assert rb.fmt(-0.5, 2) == "−0.50"


def test_ladders_dense_only_and_keep():
    repos = ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B", "google/gemma-3-4b-it", "google/gemma-3-12b-it",
             "google/gemma-3-27b-it"]
    params = [27.8, 36.0, 4.3, 12.2, 27.4]
    keys = ["q27", "q35", "g4", "g12", "g27"]
    assert set(rb.ladders(repos, params, keys)) == {"Qwen3.6", "Gemma-3"}
    assert set(rb.ladders(repos, params, keys, dense_only=True)) == {"Gemma-3"}
    assert rb.ladders(repos, params, keys, keep={"g4", "g27", "q27"}) == {"Gemma-3": ["g4", "g27"]}


def test_gaussian_null_shapes_and_reliability():
    rng = np.random.default_rng(8)
    out = rb.gaussian_null(100.0, 10.0, 24, 24, 20, rng, k_list=(2, 4), n_splits=20)
    assert set(out) == {"R_discovery", "C_one_draw", "C_ref_one_draw", "C_K2", "C_ref_K2", "C_K4", "C_ref_K4"}
    assert 0.7 < out["R_discovery"] <= 1.0
    zero = rb.gaussian_null(-5.0, 10.0, 24, 24, 10, np.random.default_rng(9), k_list=(2,), n_splits=10)
    assert abs(zero["C_one_draw"] - 0.5) < 0.1      # a nonpositive V becomes 1e-6: no true differences
