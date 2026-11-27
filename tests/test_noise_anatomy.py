"""Analysis N, noise anatomy: src/noise_anatomy.py, scripts/extract_dove_items.py, scripts/run_noise_anatomy.py.

Small fabricated arrays with answers worked by hand or by exhaustive enumeration. Arrays are (draws K, models M) in
percentage points, or (draws, models, items) of 0/1 item scores; panel moments use denominator M - 1.
"""
import importlib.util
import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest

from src import dataset1 as d1
from src import noise_anatomy as na
from src.estimators import reliability

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rn = _load("run_noise_anatomy")


# --------------------------------------------------------------------------- arm-wise split


def test_hand_worked_split_on_a_two_by_two_bank():
    """B = [[0, 0], [2, 0]], P = [[1, 1], [1, 3]]: each arm has bank noise 0.5, the arms' errors are perfectly
    anti-correlated (Cov = -0.5), so W_gain = 0.5 + 0.5 + 1 = 2; the Appendix A moments of the same two draws are
    K / (K - 1) = 2 times the bank moments."""
    B = np.array([[0.0, 0.0], [2.0, 0.0]])
    P = np.array([[1.0, 1.0], [1.0, 3.0]])
    s = na.arm_split(B, P, "bank")
    assert s["var_e_baseline"] == pytest.approx(0.5)
    assert s["var_e_intervention"] == pytest.approx(0.5)
    assert s["cov_e"] == pytest.approx(-0.5)
    assert s["corr_e"] == pytest.approx(-1.0)
    assert s["W_gain"] == pytest.approx(2.0) and s["parts_sum"] == pytest.approx(2.0)
    assert s["residual"] == pytest.approx(0.0, abs=1e-15)
    assert s["V_delta"] == pytest.approx(2.0)                     # gain means (0, 2)
    d = na.arm_split(B, P, "draws")
    for k in ("var_e_baseline", "var_e_intervention", "cov_e", "W_gain"):
        assert d[k] == pytest.approx(2 * s[k])


def test_draws_split_adds_up_and_matches_the_dataset2_noise():
    """The parts add up to W_gain, and W_gain and W_baseline equal reliability total - signal (src/dataset2.step2)."""
    rng = np.random.default_rng(3)
    theta = rng.normal(50, 10, 7)
    B = theta + rng.normal(0, 3, (9, 7))
    P = B + rng.normal(5, 2, 7) + 0.4 * (B - theta) + rng.normal(0, 1.5, (9, 7))
    s = na.arm_split(B, P, "draws")
    assert s["residual"] == pytest.approx(0.0, abs=1e-10)
    rg, rb = reliability(P - B), reliability(B)
    assert s["W_gain"] == pytest.approx(rg["total"] - rg["signal"], rel=1e-12)
    assert s["var_e_baseline"] == pytest.approx(rb["total"] - rb["signal"], rel=1e-12)
    assert s["V_delta"] == pytest.approx(rg["signal"], rel=1e-12)
    assert s["R_benefit"] == pytest.approx(rg["R"], rel=1e-12)


def test_bank_split_matches_dataset1_exact_targets():
    """On an enumerated bank the split reproduces noise_W_gain, noise_W_baseline and coupling.E_cov_eP_eB."""
    rng = np.random.default_rng(4)
    B = rng.normal(40, 8, (30, 5))
    P = B + rng.normal(6, 3, (30, 5))
    s = na.arm_split(B, P, "bank")
    t = d1.exact_targets(B, P)
    assert s["W_gain"] == pytest.approx(t["noise_W_gain"], rel=1e-12)
    assert s["var_e_baseline"] == pytest.approx(t["noise_W_baseline"], rel=1e-12)
    assert s["cov_e"] == pytest.approx(t["coupling"]["E_cov_eP_eB"], rel=1e-12)
    assert s["V_delta"] == pytest.approx(t["V_delta"], rel=1e-12)
    assert s["R_benefit"] == pytest.approx(t["R_benefit"], rel=1e-12)


def test_double_centred_form_of_the_appendix_a_noise():
    """W(X, Y) = sum of doubly centred products / ((K - 1)(M - 1)), the two-way interaction mean square."""
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    dc = lambda Z: Z - Z.mean(0) - Z.mean(1, keepdims=True) + Z.mean()
    assert na.noise_moment(X, Y, "draws") == pytest.approx(np.sum(dc(X) * dc(Y)) / (5 * 3), rel=1e-12)


def test_negative_noise_and_undefined_ratios_are_reported_as_they_are():
    """A cross-draw estimate can be negative; ratios with a nonpositive denominator are NaN, never clipped."""
    B = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    s = na.arm_split(B, P, "draws")
    assert s["var_e_baseline"] == 0.0
    assert math.isnan(s["ratio_W_gain_to_W_baseline"]) and math.isnan(s["corr_e"])
    assert s["V_delta"] < 0                                       # cross moment of two disagreeing draws
    assert s["W_gain"] == pytest.approx(s["var_e_intervention"])


def test_batched_moments_equal_the_single_array_moments():
    rng = np.random.default_rng(6)
    X, Y = rng.normal(size=(5, 4, 3)), rng.normal(size=(5, 4, 3))
    for kind in na.KINDS:
        nb, sb = na.noise_batch(X, Y, kind), na.signal_batch(X, Y, kind)
        for r in range(3):
            assert nb[r] == pytest.approx(na.noise_moment(X[:, :, r], Y[:, :, r], kind), rel=1e-12)
            assert sb[r] == pytest.approx(na.signal_moment(X[:, :, r], Y[:, :, r], kind), rel=1e-12)
    b = na.split_batch(X, Y, "draws")
    s = na.arm_split(X[:, :, 1], Y[:, :, 1], "draws")
    for k in na.PARTS + ("V_delta", "R_benefit"):
        assert b[k][1] == pytest.approx(s[k], rel=1e-12)


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        na.noise_moment(np.ones((2, 2)), np.ones((2, 2)), "pooled")


# --------------------------------------------------------------------------- item subsamples


def _items(K=3, M=3, N=6, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, (1, M, N))
    SB = (rng.uniform(size=(K, M, N)) < p).astype(np.int8)
    SP = (rng.uniform(size=(K, M, N)) < np.clip(p + 0.2, 0, 1)).astype(np.int8)
    return SB, SP


def test_full_set_subsample_reproduces_the_pooled_score():
    SB, _ = _items()
    v = np.array([1, 1, 1, 2, 2, 2], float) / 9
    U = na.subsample_matrix(6, na.draw_subsamples(6, 6, 5, np.random.default_rng(0)), v)
    assert U.shape == (6, 1)
    assert np.allclose(na.subsample_scores(SB, U)[:, :, 0], na.full_scores(SB, v))
    assert np.allclose(na.full_scores(SB, v), 100 * np.einsum("kmn,n->km", SB, v))


def test_subsample_scores_are_the_weighted_item_means():
    """Uniform weights: a subsample's score is the plain accuracy over its items, in pp."""
    SB, _ = _items(N=8)
    idx = np.array([[0, 2, 5], [1, 3, 7]])
    X = na.subsample_scores(SB, na.subsample_matrix(8, idx), chunk=2)
    for r in range(2):
        assert np.allclose(X[:, :, r], 100 * SB[:, :, idx[r]].mean(axis=2))


def test_draw_subsamples_is_without_replacement_and_reproducible():
    a = na.draw_subsamples(20, 7, 4, np.random.default_rng(11))
    b = na.draw_subsamples(20, 7, 4, np.random.default_rng(11))
    assert a.shape == (4, 7) and np.array_equal(a, b)
    assert all(len(set(r)) == 7 and list(r) == sorted(r) for r in a)
    assert np.array_equal(na.draw_subsamples(20, 20, 4, np.random.default_rng(0)), np.arange(20)[None])
    with pytest.raises(ValueError):
        na.draw_subsamples(20, 21, 4, np.random.default_rng(0))


@pytest.mark.parametrize("kind", na.KINDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_design_expectation_equals_the_exhaustive_mean_over_subsets(kind, weighted):
    """E over all C(N, n) item subsets of each noise part and of V_delta is full + (1/n - 1/N) W_item exactly."""
    SB, SP = _items()
    N = SB.shape[2]
    v = np.array([3, 1, 1, 2, 2, 1], float) / 10 if weighted else None
    de = na.design_expectation(SB, SP, v, kind, chunk=4)
    for n in (2, 3, 5):
        idx = np.array(list(itertools.combinations(range(N), n)))
        U = na.subsample_matrix(N, idx, v)
        b = na.split_batch(na.subsample_scores(SB, U), na.subsample_scores(SP, U), kind)
        for part in na.PARTS + ("V_delta",):
            assert b[part].mean() == pytest.approx(na.expected_at(de["parts"][part], n), rel=1e-10, abs=1e-10)
    full = na.split_batch(na.full_scores(SB, v)[:, :, None], na.full_scores(SP, v)[:, :, None], kind)
    for part in na.PARTS:
        assert de["parts"][part]["full"] == pytest.approx(full[part][0], rel=1e-12)
        assert na.expected_at(de["parts"][part], N) == pytest.approx(full[part][0], rel=1e-12)


# --------------------------------------------------------------------------- fit


def test_fit_recovers_an_exact_inverse_n_law():
    """Means on W(n) = 3 + 500/n with symmetric scatter: W_item = 500 and W_shift = 3 exactly, chi2 = 0."""
    N, ns = 1000, [50, 100, 200, 1000]
    vals = [np.array([3 + 500 / n - 0.4, 3 + 500 / n, 3 + 500 / n + 0.4]) for n in ns[:-1]] + [[3 + 500 / N]]
    f = na.fit_inverse_n(ns, vals, N)
    assert f["W_item"] == pytest.approx(500.0)
    assert f["W_shift"] == pytest.approx(3.0)
    assert f["W_full"] == pytest.approx(3.5)
    assert f["chi2"] == pytest.approx(0.0, abs=1e-12) and f["df"] == 2
    assert f["ols_sensitivity"]["W_item"] == pytest.approx(500.0) and f["ols_sensitivity"]["W_shift"] == pytest.approx(3.0)
    assert all(abs(r["fitted"] - (3 + 500 / r["n"])) < 1e-9 for r in f["per_n"])


def test_fit_weights_follow_the_monte_carlo_error():
    """Two points off the line; the tighter one dominates the weighted slope through W(N)."""
    N = 100
    vals = [np.array([11.0, 13.0]), np.array([4.9, 5.1]), [1.0]]      # n = 10 (mean 12), n = 20 (mean 5), N
    f = na.fit_inverse_n([10, 20, 100], vals, N)
    x1, x2 = 1 / 10 - 1 / N, 1 / 20 - 1 / N
    w1, w2 = 1 / (2.0 / 2), 1 / (0.02 / 2)
    want = (w1 * x1 * 11 + w2 * x2 * 4) / (w1 * x1 ** 2 + w2 * x2 ** 2)
    assert f["W_item"] == pytest.approx(want)
    assert f["W_item_se"] == pytest.approx(1 / math.sqrt(w1 * x1 ** 2 + w2 * x2 ** 2))
    assert abs(f["W_item"] - 4 / x2) < abs(f["W_item"] - 11 / x1)


def test_fit_falls_back_to_equal_weights_when_a_variance_is_zero():
    f = na.fit_inverse_n([10, 100], [np.array([5.0, 5.0]), [1.0]], 100)
    assert f["weights"] == "equal" and math.isnan(f["chi2"])
    assert f["W_item"] == pytest.approx(4 / (1 / 10 - 1 / 100))


def test_fit_needs_the_full_set_as_one_value():
    with pytest.raises(ValueError):
        na.fit_inverse_n([10, 50], [np.ones(3), np.ones(3)], 100)
    with pytest.raises(ValueError):
        na.fit_inverse_n([10, 100], [np.ones(3), np.ones(2)], 100)


# --------------------------------------------------------------------------- planning rule


def test_planning_rule_reproduces_the_appendix_b_prompt_counts():
    """R = 0.30 needs 10 draws for 80% and 46 for 90% (Appendix B); V_delta = 3, W = 7 gives R = 0.30."""
    r = na.planning_rule(3.0, 7.0, 0.0, [100])[0]
    assert r["R"] == pytest.approx(0.3)
    assert r["K_80"] == 10 and r["K_90"] == 46
    assert r["C_1"] == pytest.approx(0.5 + math.asin(0.3) / math.pi)


def test_planning_rule_uses_w_of_n():
    """W(n) = 2 + 800/n with V_delta = 10: R(100) = 0.5, R(800) = 10/13."""
    rows = na.planning_rule(10.0, 2.0, 800.0, [100, 800])
    assert rows[0]["W"] == pytest.approx(10.0) and rows[0]["R"] == pytest.approx(0.5)
    assert rows[1]["R"] == pytest.approx(10 / 13)
    assert rows[0]["K_80"] >= rows[1]["K_80"]


def test_planning_rule_reports_undefined_and_unbounded_cases():
    neg = na.planning_rule(-1.0, 0.5, 0.0, [10])[0]                 # V + W <= 0
    assert math.isnan(neg["R"]) and neg["K_80"] is None and "not positive" in neg["undefined"]
    over = na.planning_rule(5.0, -1.0, 0.0, [10])[0]                # negative W: R > 1
    assert over["R"] == pytest.approx(1.25) and over["K_90"] is None and over["undefined"] == "R outside [0, 1]"
    zero = na.planning_rule(0.0, 4.0, 0.0, [10])[0]                 # no true variation
    assert zero["R"] == 0.0 and zero["K_80"] == "unbounded"


# --------------------------------------------------------------------------- DOVE item extraction


def _raw(items, configs, scores, shots=0):
    rows = []
    for (i, c), s in zip(itertools.product(items, configs), scores):
        rows.append({"sample_index": i, "dimensions_1: enumerator": c[0], "dimensions_2: separator": c[1],
                     "dimensions_3: choices_order": c[2], "dimensions_4: instruction_phrasing_text": c[3],
                     "dimensions_5: shots": shots, "score": s})
    return pd.DataFrame(rows)


CFGS = [("capitals", "\n", "alphabetical", "Q: {question}"), ("numbers", "; ", "none", "Q: {question}"),
        ("roman", " ", "length", "Answer: {question}")]
KEY_TO_ID = {"|".join(c): k for k, c in zip((10, 11, 12), CFGS)}


# --------------------------------------------------------------------------- run script pieces


def test_dense_items_refuses_missing_or_duplicated_cells():
    rows = [{"model_key": m, "arm": a, "config_id": c, "item_id": i, "strict_correct": (i + len(m)) % 2}
            for m in ("a", "bb") for a in ("direct_answer", "zero_shot_cot") for c in ("c1", "c2") for i in (0, 1, 2)]
    df = pd.DataFrame(rows)
    S = rn.dense_items(df, ["a", "bb"], ["c1", "c2"], [0, 1, 2])
    assert S.shape == (2, 2, 2, 3) and S[0, 0, 1].tolist() == [0, 1, 0]
    with pytest.raises(rn.Refusal):
        rn.dense_items(df.iloc[1:], ["a", "bb"], ["c1", "c2"], [0, 1, 2])
    with pytest.raises(rn.Refusal):
        rn.dense_items(pd.concat([df, df.iloc[[0]]]), ["a", "bb"], ["c1", "c2"], [0, 1, 2])
    with pytest.raises(rn.Refusal):
        rn.dense_items(df, ["a", "bb"], ["c1", "c2"], [0, 1])                   # item 2 outside the set


def test_macros_are_letters_only_with_the_noise_prefix():
    SB, SP = _items(K=4, M=3, N=12)
    data = {"setting": "gsm8k", "B": SB, "P": SP, "weights": None, "panel": ["a", "b", "c"],
            "arms": {"baseline": "direct_answer", "intervention": "zero_shot_cot"},
            "variants": {"discovery": {"draws": np.arange(4), "kind": "draws", "label": "test"}},
            "reference": {}, "provenance": {"tables": [], "value": "strict_correct"}, "checks": {}}
    cfg = {"seed": 1, "subsamples_per_n": 5, "n_grid": {"gsm8k": [3, 6, 12]}, "agreement_targets": [0.8, 0.9],
           "planning_n": [6, 100]}
    out, rows = rn.run_setting(data, cfg)
    assert len(rows) == 5 + 5 + 1
    assert [r["n"] for r in out["variants"]["discovery"]["planning"]["from_fit"]] == [3, 6, 12]  # no n beyond N
    R = {"settings": {"gsm8k": out}, "config": cfg}
    m = rn.macros(R)
    assert m and all(k.startswith("noise") and k.isalpha() for k in m)
    assert all("−" not in v for v in m.values())
