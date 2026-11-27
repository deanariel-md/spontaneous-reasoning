"""The template SD's item-sampling term.

Every configuration has the same items, so item difficulty is common to every configuration mean and drops out of
their spread. The term to remove from the variance of the configuration means is the configuration x item interaction
mean square over the number of items (src/noise_anatomy.config_variance, the panel's design_expectation written for one
model), not the item variance within a configuration over n, which still holds item difficulty. Synthetic designs with
known components: y_ci = mu + a_c + b_i + e_ci, Var(a) = s2_a, Var(b) = s2_b, Var(e) = s2_e.
"""
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import noise_anatomy as na

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def two_way(y):
    """The interaction mean square by the classical sums of squares."""
    C, N = y.shape
    g = y.mean()
    ss_tot = ((y - g) ** 2).sum()
    ss_c = N * ((y.mean(1) - g) ** 2).sum()
    ss_i = C * ((y.mean(0) - g) ** 2).sum()
    return (ss_tot - ss_c - ss_i) / ((C - 1) * (N - 1))


def old_term(y):
    """The term in question: mean over configurations of the item variance within it, over n."""
    return float(np.mean(np.var(y, axis=1, ddof=1) / y.shape[1]))


def draw(rng, C, N, s2a, s2b, s2e):
    return (rng.normal(0, np.sqrt(s2a), (C, 1)) + rng.normal(0, np.sqrt(s2b), (1, N))
            + rng.normal(0, np.sqrt(s2e), (C, N)))


def test_w_item_is_the_interaction_mean_square():
    rng = np.random.default_rng(1)
    y = draw(rng, 12, 40, 3.0, 50.0, 20.0)
    out = na.config_variance(y, scale=1.0)
    assert out["W_item"] == pytest.approx(two_way(y), rel=1e-12)
    assert out["full"] == pytest.approx(np.var(y.mean(1), ddof=1), rel=1e-12)
    assert out["W_shift"] == pytest.approx(out["full"] - two_way(y) / 40, rel=1e-12)
    assert (out["C"], out["N"]) == (12, 40)


def test_item_difficulty_does_not_enter():
    """Adding any item effect (common to all configurations) changes nothing; the old term grows with it."""
    rng = np.random.default_rng(2)
    y = draw(rng, 10, 30, 2.0, 0.0, 10.0)
    b = rng.normal(0, 30, (1, 30))
    a, c = na.config_variance(y, 1.0), na.config_variance(y + b, 1.0)
    for k in ("full", "W_item", "W_shift"):
        assert c[k] == pytest.approx(a[k], rel=1e-10, abs=1e-10)
    assert old_term(y + b) > old_term(y) + 10


def test_unbiased_for_the_configuration_component_where_the_old_term_is_not():
    """Monte Carlo: E W_shift = s2_a; the old difference is low by about s2_b / N."""
    rng = np.random.default_rng(3)
    s2a, s2b, s2e, C, N = 4.0, 400.0, 25.0, 30, 100
    new, old = [], []
    for _ in range(1500):
        y = draw(rng, C, N, s2a, s2b, s2e)
        o = na.config_variance(y, 1.0)
        new.append(o["W_shift"])
        old.append(o["full"] - old_term(y))
    se = np.std(new, ddof=1) / np.sqrt(len(new))
    assert abs(np.mean(new) - s2a) < 4 * se
    assert np.mean(old) == pytest.approx(s2a - s2b / N, abs=4 * se + 0.05)
    assert np.mean(old) < s2a - 3.0


def test_expectations_of_the_parts():
    rng = np.random.default_rng(4)
    s2a, s2e, C, N = 1.5, 9.0, 20, 50
    fulls, items = [], []
    for _ in range(1500):
        o = na.config_variance(draw(rng, C, N, s2a, 100.0, s2e), 1.0)
        fulls.append(o["full"])
        items.append(o["W_item"])
    assert np.mean(fulls) == pytest.approx(s2a + s2e / N, rel=0.06)
    assert np.mean(items) == pytest.approx(s2e, rel=0.02)


def test_binary_scores_with_item_difficulty():
    """0/1 scores from a logistic model with widely spread item difficulty and no configuration effect: W_shift is
    near zero on average, the old difference clearly negative."""
    rng = np.random.default_rng(5)
    C, N = 25, 200
    new, old = [], []
    for _ in range(400):
        b = rng.normal(0, 2.0, N)
        p = 1 / (1 + np.exp(-(0.3 + b)))[None, :].repeat(C, 0)
        y = (rng.random((C, N)) < p).astype(float)
        o = na.config_variance(y)
        new.append(o["W_shift"])
        old.append(o["full"] - old_term(100 * y))
    assert abs(np.mean(new)) < 4 * np.std(new, ddof=1) / np.sqrt(len(new))
    assert np.mean(old) < -3.0


def test_input_checks():
    with pytest.raises(ValueError):
        na.config_variance(np.ones((1, 5)))
    with pytest.raises(ValueError):
        na.config_variance(np.array([[1.0, np.nan], [0.0, 1.0]]))


# --------------------------------------------------------------------------- the ladder's code paths


def _frame(rng, C=6, N=40, unbalanced=False):
    rows = []
    for c in range(C):
        for i in range(N):
            if unbalanced and c == 0 and i == 0:
                continue
            for arm, p in (("direct_answer", 0.4), ("forced_direct", 0.3), ("zero_shot_cot", 0.6)):
                rows.append({"config_id": f"c{c}", "item_id": f"i{i}", "arm": arm,
                             "strict_correct": int(rng.random() < p + 0.02 * c)})
    return pd.DataFrame(rows)


def test_nebius_template_block_uses_the_interaction_term():
    import nebius_panel_report as npr
    rng = np.random.default_rng(6)
    sc = _frame(rng)
    tb = npr.template_block(sc, "strict_correct")
    wide = sc.pivot_table(index=["config_id", "item_id"], columns="arm", values="strict_correct", aggfunc="first")
    for k, v in (("requested", wide["direct_answer"]), ("gain_over_forced", wide["zero_shot_cot"] - wide["forced_direct"])):
        o = na.config_variance(v.unstack("item_id").to_numpy(float))
        assert tb[k]["items"] == 40
        assert tb[k]["interaction_mean_square_pp2"] == pytest.approx(o["W_item"])
        assert tb[k]["item_sampling_variance_pp2"] == pytest.approx(o["W_item"] / 40)
        assert tb[k]["variance_minus_item_sampling_pp2"] == pytest.approx(o["W_shift"])
        assert tb[k]["variance_pp2"] == pytest.approx(o["full"])
    un = npr.template_block(_frame(rng, unbalanced=True), "strict_correct")
    assert un["requested"]["item_sampling_variance_pp2"] is None
    assert un["requested"]["variance_minus_item_sampling_pp2"] is None
    assert un["requested"]["sd_pp"] is not None


def test_ladder_exact_and_checker_agree_by_two_routes():
    import ladder_analysis as la
    import verify_ladder_analysis as vla
    rng = np.random.default_rng(7)
    sc = _frame(rng, C=5, N=30)
    ex = la.template_exact(sc, "strict_correct", "synthetic")
    chk = vla.template(sc, "strict_correct")
    for q, qa in (("requested", "requested"), ("forced", "forced"), ("cot", "cot"),
                  ("cot_minus_requested", "gain_over_requested"), ("cot_minus_forced", "gain_over_forced")):
        assert isinstance(chk[q]["ms_interaction"], Fraction)
        assert ex[qa]["interaction_mean_square_pp2"] == chk[q]["ms_interaction"]
        assert ex[qa]["variance_minus_item_sampling_pp2"] == chk[q]["excess"]
        assert ex[qa]["variance_pp2"] == chk[q]["V"]
        y = (100 * sc[sc["arm"] == {"requested": "direct_answer", "forced": "forced_direct",
                                    "cot": "zero_shot_cot"}.get(q, "zero_shot_cot")]
             .pivot(index="config_id", columns="item_id", values="strict_correct")).to_numpy(float)
        if q in ("requested", "forced", "cot"):
            assert float(chk[q]["ms_interaction"]) == pytest.approx(two_way(y), rel=1e-12)
