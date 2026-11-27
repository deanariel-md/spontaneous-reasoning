"""Helpers of scripts/run_late_generation.py."""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def lg():
    spec = importlib.util.spec_from_file_location("run_late_generation_under_test",
                                                  os.path.join(SCRIPTS, "run_late_generation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spearman_ranks_ties_and_constants(lg):
    assert lg.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert lg.spearman([1, 2, 3, 4], [4, 1, 3, 2]) == pytest.approx(-0.4)
    # ties take average ranks: ranks (1.5, 1.5, 3, 4) against (1, 2, 3, 4)
    r = np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1]
    assert lg.spearman([5, 5, 6, 7], [1, 2, 3, 4]) == pytest.approx(r)
    assert lg.spearman([1, 1, 1], [1, 2, 3]) is None


def test_partial_spearman_formula_and_edge_cases(lg):
    rng = np.random.default_rng(3)
    s, d, b = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
    r_sd, r_sb, r_db = lg.spearman(s, d), lg.spearman(s, b), lg.spearman(d, b)
    want = (r_sd - r_sb * r_db) / np.sqrt((1 - r_sb ** 2) * (1 - r_db ** 2))
    assert lg.partial_spearman(s, d, b) == pytest.approx(want)
    # the control identical to one variable leaves the denominator at zero: undefined, not clipped
    assert lg.partial_spearman(s, d, s) is None
    assert lg.partial_spearman([1, 1, 1], d[:3], b[:3]) is None


def test_statistics_pool_draws_and_drop_models(lg):
    # two draws, three models; shares are pooled counts over the kept draws
    C = np.array([[10.0, 10.0, 10.0], [10.0, 30.0, 10.0]])
    L = np.array([[1.0, 2.0, 9.0], [3.0, 6.0, 1.0]])
    B = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]])
    days = np.array([0.0, 100.0, 200.0])
    fam = ["A", "A", "A"]
    s, b, out = lg.statistics(C, L, B, days, fam, np.array([True, True]))
    assert np.allclose(s, [4 / 20, 8 / 40, 10 / 20])
    assert np.allclose(b, [11.0, 21.0, 31.0])
    assert out["family:A"] == pytest.approx(out["spearman"])
    s1, _, _ = lg.statistics(C, L, B, days, fam, np.array([True, False]))
    assert np.allclose(s1, [0.1, 0.2, 0.9])
    s2, _, o2 = lg.statistics(C, L, B, days, fam, np.array([True, True]), np.array([True, False, True]))
    assert np.allclose(s2, [0.2, 0.5]) and set(o2) == set(lg.ACROSS)


def test_jack_reports_undefined_replicates(lg):
    r = lg.jack(0.5, [0.4, None, 0.6])
    assert r["ci95"] is None and r["undefined_replicates"] == 1
    r = lg.jack(0.05, [0.0, 0.05, 0.1])
    assert r["ci95"][0] < 0 < r["ci95"][1] and r["sign"] == "includes zero"
    assert lg.jack(0.5, [0.4, 0.5, 0.6])["sign"] == "above zero"
