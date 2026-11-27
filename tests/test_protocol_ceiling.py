"""Tests of src/protocol_ceiling.py."""
import numpy as np
import pytest

from src import estimators, protocol_ceiling as pcl


def test_pair_c_counts_ties_half():
    assert pcl.pair_c([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
    assert pcl.pair_c([1, 2, 3], [3, 2, 1]) == pytest.approx(0)
    # pairs (0,1) tie in b -> 0.5; (0,2), (1,2) agree
    assert pcl.pair_c([1, 2, 3], [1, 1, 3]) == pytest.approx((0.5 + 1 + 1) / 3)


def test_within_curve_at_k1_matches_pair_agreement_in_expectation():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(12, 6)) + np.arange(6) * 0.3
    got = pcl.within_curve(Y, [1], np.random.default_rng(1), n_splits=4000)[1]
    want = estimators.pair_agreement(Y)["C"]
    assert got == pytest.approx(want, abs=0.02)
    c = pcl.within_curve(Y, [6, 7], rng, n_splits=5)
    assert 6 in c and 7 not in c                      # 2K must not exceed the 12 rows


def test_between_curve_identical_protocols_is_one_and_full_set_is_plugin():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(8, 5)) + np.arange(5)
    assert pcl.between_curve(Y, Y, [1, 3], rng)[3] == pytest.approx(1)
    Z = Y[:, ::-1]
    assert pcl.between_curve(Y, Z, [8], rng)[8] == pytest.approx(pcl.limit_plugin(Y, Z))


def test_gaussian_reference_limits():
    Y = np.array([[0.0, 1.0, 2.0], [0.2, 1.1, 2.3], [-0.1, 0.9, 1.8]])
    assert pcl.gaussian_within(Y, 10_000) == pytest.approx(1, abs=1e-6)
    assert pcl.gaussian_between(Y, -Y, 10_000) == pytest.approx(0, abs=1e-6)
    # a pair with zero difference everywhere is a tie: probability one half each way
    Z = np.zeros((4, 2))
    assert pcl.gaussian_within(Z, 3) == pytest.approx(0.5)


def test_bootstrap_limit_interval_contains_plugin_for_clear_signal():
    Y = np.tile(np.arange(5.0), (10, 1)) + np.random.default_rng(3).normal(scale=0.01, size=(10, 5))
    b = pcl.bootstrap_limit(Y, Y, np.random.default_rng(4), n_boot=200)
    assert b["ci95"][0] == pytest.approx(1) and b["ci95"][1] == pytest.approx(1)
