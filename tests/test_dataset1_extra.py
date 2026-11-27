"""Checks for Dataset 1 helpers not covered in test_dataset1_sim.py."""
import numpy as np
import pytest

from src import dataset1 as d1
from src import simulate


@pytest.fixture(scope="module")
def bank():
    rng = np.random.default_rng(7)
    nc0, nc1, n = simulate.synthetic_counts(rng, M=5, S=4, C=250)
    return simulate.bank_from_counts(nc0, nc1, n)


def test_classify_point():
    assert d1.classify_point(1.5, 1.0) == "outside margin"
    assert d1.classify_point(-1.5, 1.0) == "outside margin"
    assert d1.classify_point(0.5, 1.0) == "within margin"
    assert d1.classify_point(float("nan"), 1.0) == "undefined"
    assert d1.classify_point(None, 1.0) == "undefined"


def test_pairwise_differences_matches_direct(bank):
    B, P = bank
    G = P - B
    names = [f"m{i}" for i in range(G.shape[1])]
    rows = d1.pairwise_differences(G, names, margin=5.0)
    assert len(rows) == 10
    r = rows[0]
    d = G[:, 0] - G[:, 1]
    assert r["pair"] == "m0 - m1"
    assert r["expected_difference"] == pytest.approx(d.mean())
    assert r["median"] == pytest.approx(np.median(d))
    assert r["frac_abs_gt_margin"] == pytest.approx(np.mean(np.abs(d) > 5.0))
    assert r["expected_beyond_margin"] == (abs(d.mean()) > 5.0)


def test_capability_estimates_definitions(bank):
    B, P = bank
    idx = np.arange(6)
    est = d1.capability_estimates(B[idx], P[idx])
    G = P[idx] - B[idx]
    assert est["naive_single_unit"] == pytest.approx(-10 * np.polyfit(B[idx][0], G[0], 1)[0])
    assert est["averaged_ols"] == pytest.approx(-10 * np.polyfit(B[idx].mean(0), G.mean(0), 1)[0])
    q = est["_moments"]
    assert est["replicated"] == pytest.approx(-10 * q["A"] / q["V_theta"])
    one = d1.capability_estimates(B[:1], P[:1])
    assert np.isnan(one["replicated"])


def test_mc_capability_blocks_fields_and_target(bank):
    B, P = bank
    target = d1.exact_targets(B, P)["H"]
    out = d1.mc_capability_blocks(B, P, 12, 300, np.random.default_rng(3), target)
    assert set(out) == {"naive_single_unit", "averaged_ols", "replicated"}
    assert out["naive_single_unit"]["K"] == 1 and out["replicated"]["K"] == 12
    for m in out.values():
        assert 0 <= m["sign_agreement_between_blocks"] <= 1
        assert 0 <= m["undefined_fraction"] <= 1
    # With 12 units the replicated and averaged estimates concentrate nearer the target than one unit.
    assert out["replicated"]["sd_H"] < out["naive_single_unit"]["sd_H"]


def test_capability_blocks_count_undefined_denominators():
    # Identical baselines across models in every configuration: V_theta = 0, slope undefined.
    C, M = 50, 4
    B = np.tile(np.full(M, 40.0), (C, 1))
    P = B + np.random.default_rng(0).normal(0, 1, (C, M))
    out = d1.mc_capability_blocks(B, P, 4, 50, np.random.default_rng(1), float("nan"))
    assert out["replicated"]["undefined_fraction"] == 1.0
    assert out["naive_single_unit"]["undefined_fraction"] == 1.0
