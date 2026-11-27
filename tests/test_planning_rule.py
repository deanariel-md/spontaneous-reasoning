"""Tests for the conservative planning rule (scripts/run_planning_rule.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_planning_rule_t", ROOT / "scripts" / "run_planning_rule.py")
prl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prl)


def gains(K, M, signal, noise, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, signal, M)[None, :] + rng.normal(0, noise, (K, M)) + rng.normal(0, 3, (K, 1))


def test_r_through_mean_squares_equals_pilot_r():
    for seed in range(5):
        G = gains(4, 29, 2.0, 1.0, seed)
        assert prl.r_from_f(G) == pytest.approx(prl.pc.pilot_r(G), rel=1e-10, abs=1e-12)


def test_lower_bound_is_below_r_and_follows_the_f_quantile():
    G = gains(4, 5, 1.0, 1.0, 3)
    ms_a, ms_e, df1, df2 = prl.intervals.anova_mean_squares(G)
    F = ms_a / ms_e
    fl = F / stats.f.ppf(0.80, df1, df2)
    assert prl.r_lower(G) == pytest.approx((fl - 1) / (fl + 4 - 1))
    assert prl.r_lower(G) < prl.r_from_f(G)
    # more models narrow the bound toward R
    gap5 = prl.r_from_f(G) - prl.r_lower(G)
    G29 = gains(4, 29, 1.0, 1.0, 3)
    assert prl.r_from_f(G29) - prl.r_lower(G29) < gap5


def test_lower_bound_reported_negative_and_undefined_without_residual():
    G = gains(4, 5, 0.0, 1.0, 7)             # no model signal: R near zero, R_L negative
    assert prl.r_lower(G) < 0
    assert prl.pc.chosen_k(prl.r_lower(G)) is None
    flat = np.tile(np.arange(5.0), (4, 1))   # identical draws: no residual variation
    assert prl.r_lower(flat) is None


def test_lower_rule_plans_at_least_as_many_draws():
    for seed in range(10):
        G = gains(4, 29, 1.5, 1.0, seed)
        kc, kl = prl.pc.chosen_k(prl.pc.pilot_r(G)), prl.pc.chosen_k(prl.r_lower(G))
        if kc is not None and kl is not None:
            assert kl >= kc


def test_big_mean_k_prints_in_powers_of_ten():
    assert prl.big(26.94) == "26.9"
    assert prl.big(13543392628146.8) == "1.4 × 10^13"
    assert prl.big(13543392628146.8, tex=True) == r"$1.4\times 10^{13}$"
    assert prl.big(None) == "--"
