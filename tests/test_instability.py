"""Analysis B (src/instability.py): scales, cross-draw slopes, subpanels, single draws and the direct-answer variants,
on small fabricated data with known answers."""
import importlib.util
import itertools
import math
import os

import numpy as np
import pytest

from src import dataset1 as d1, instability as ins, intervals as iv, scoring
from src.estimators import cross_moment, pair_agreement

ROOT = os.path.join(os.path.dirname(__file__), "..")
BOXED = scoring.CANDIDATE_SPECS["boxed"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logit(p):
    return math.log(p / (1 - p))


# --------------------------------------------------------------------------- scales


def test_points_scale_is_identity():
    B = np.array([[20.0, 50.0], [30.0, 60.0]])
    P = B + np.array([[5.0, -2.0], [1.0, 3.0]])
    s = ins.scale_scores(B, P, "points", 100)
    assert np.array_equal(s["x"], B) and np.array_equal(s["y"], P - B) and s["n_replaced"] == 0


def test_logit_and_error_reduction_values():
    p1 = 100 / (1 + math.exp(-1))                  # logit 1
    B = np.array([[50.0, 80.0], [50.0, 80.0]])
    P = np.array([[p1, 90.0], [p1, 90.0]])
    lg = ins.scale_scores(B, P, "logit", 100)
    assert lg["x"][0, 0] == pytest.approx(0.0) and lg["y"][0, 0] == pytest.approx(1.0)
    assert lg["y"][0, 1] == pytest.approx(_logit(0.9) - _logit(0.8))
    er = ins.scale_scores(B, P, "error_reduction", 100)
    assert er["x"][0, 1] == 80.0 and er["y"][0, 1] == pytest.approx(50.0)     # half the 20-point error removed
    assert er["y"][0, 0] == pytest.approx(100 * (p1 - 50) / 50)


def test_undefined_cells_use_the_empirical_proportion_and_are_flagged():
    n = 9
    B = np.array([[0.0, 50.0], [100.0, 40.0]])
    P = np.array([[50.0, 60.0], [100.0, 40.0]])
    lg = ins.scale_scores(B, P, "logit", n)
    assert lg["n_replaced"] == 2 and lg["replaced"].tolist() == [[True, False], [True, False]]
    # (k + 1/2) / (n + 1): B = 0 of 9 -> 5%, P = 4.5 of 9 -> 50%, 100% -> 95%
    assert lg["x"][0, 0] == pytest.approx(_logit(0.05)) and lg["y"][0, 0] == pytest.approx(-_logit(0.05))
    assert lg["y"][1, 0] == pytest.approx(0.0)
    assert lg["y"][0, 1] == pytest.approx(_logit(0.6) - _logit(0.5))        # untouched cell
    er = ins.scale_scores(B, P, "error_reduction", n)
    assert er["n_replaced"] == 1 and er["replaced"][1, 0]
    assert er["x"][1, 0] == pytest.approx(95.0) and er["y"][1, 0] == pytest.approx(0.0)
    assert er["y"][0, 0] == pytest.approx(50.0)                              # B = 0% is defined here
    with pytest.raises(ValueError):
        ins.scale_scores(B, P, "odds", n)


# --------------------------------------------------------------------------- moments


def test_scale_moments_exact_line():
    """Draw shifts only: every draw orders the panel identically, so the slope is exact and R = 1."""
    x = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    shifts = np.array([0.0, 3.0, -2.0, 1.0])[:, None]
    X = x[None, :] + shifts
    Y = 4.0 - 0.25 * x[None, :] + 2 * shifts
    for method in ("fieller", "jack_t"):
        r = ins.scale_moments(X, Y, method)
        assert r["slope"] == pytest.approx(-0.25)
        assert r["V_x"] == pytest.approx(d1.pvar(x)) and r["A"] == pytest.approx(-0.25 * d1.pvar(x))
        assert r["R_benefit"] == pytest.approx(1.0) and r["W_y"] == pytest.approx(0.0, abs=1e-9)
        assert r["sign"] == "negative" and r["slope_set_sign"] == "below zero"
        assert r["slope_set"]["a"] == pytest.approx(-0.25) and r["slope_set"]["b"] == pytest.approx(-0.25)
        assert r["naive_model_mean_slope"] == pytest.approx(-0.25)


def test_scale_moments_match_appendix_a_and_jackknife():
    rng = np.random.default_rng(3)
    K, M = 8, 6
    th = np.linspace(20, 80, M)
    X = th + rng.normal(0, 3, (K, M))
    Y = 10 + 0.2 * th + rng.normal(0, 4, (K, M))
    r = ins.scale_moments(X, Y, "fieller")
    assert r["A"] == pytest.approx(cross_moment(Y, X)) and r["V_x"] == pytest.approx(cross_moment(X, X))
    assert r["slope"] == pytest.approx(cross_moment(Y, X) / cross_moment(X, X))
    kind, a, b = iv.fieller(cross_moment(Y, X), cross_moment(X, X), iv.CrossMoment(Y, X).loo(),
                            iv.CrossMoment(X, X).loo())
    assert r["slope_set"] == {"kind": kind, "a": a, "b": b}
    # R's jackknife: leave-one-draw-out values by deletion
    from src.estimators import reliability
    loo = np.array([reliability(np.delete(Y, u, 0))["R"] for u in range(K)])
    assert np.allclose(ins._reliability_loo(Y), loo)
    assert r["R_benefit_ci95"] == pytest.approx(list(iv.jack_t(reliability(Y)["R"], loo)))


def test_nonpositive_capability_variance_leaves_the_slope_undefined():
    X = np.tile(np.array([40.0, 40.0, 40.0]), (4, 1)) + np.arange(4)[:, None]
    Y = np.random.default_rng(0).normal(size=(4, 3))
    r = ins.scale_moments(X, Y, "jack_t")
    assert r["V_x_nonpositive"] and math.isnan(r["slope"]) and r["sign"] is None


def test_bank_exact_matches_dataset1_targets():
    rng = np.random.default_rng(5)
    B = rng.uniform(20, 70, (40, 5))
    P = B + rng.normal(5, 3, (40, 5))
    e = ins.bank_exact(B, P - B)
    t = d1.exact_targets(B, P)
    assert e["slope"] == pytest.approx(t["beta"]) and e["R_benefit"] == pytest.approx(t["R_benefit"])
    assert e["V_y"] == pytest.approx(t["V_delta"]) and e["W_y"] == pytest.approx(t["noise_W_gain"])


# --------------------------------------------------------------------------- single draws


def test_opposite_sign_share_counts():
    s = np.array([0.5, -0.2, 0.1, np.nan, 0.0, -0.3])
    r = ins.opposite_sign_share(s, reference=-1.0)
    assert r["n_defined"] == 5 and r["n_zero"] == 1 and r["n_opposite"] == 2
    assert r["share_opposite"] == pytest.approx(2 / 5)
    assert math.isnan(ins.opposite_sign_share(s, 0.0)["share_opposite"])


def test_single_draw_slopes():
    X = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    Y = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert ins.single_draw_slopes(X, Y).tolist() == [1.0, -1.0]


# --------------------------------------------------------------------------- subpanels


def test_subpanel_draw_enumerates_or_samples():
    all_, mode = ins.subpanel_draw(range(6), 4, 20, np.random.default_rng(0))
    assert mode == "all" and len(all_) == math.comb(6, 4)
    assert {tuple(r) for r in all_} == set(itertools.combinations(range(6), 4))
    s1, m1 = ins.subpanel_draw(range(6), 4, 10, np.random.default_rng(1))
    s2, _ = ins.subpanel_draw(range(6), 4, 10, np.random.default_rng(1))
    assert m1 == "sampled" and len({tuple(r) for r in s1}) == 10 and np.array_equal(s1, s2)
    assert ins.subpanel_draw(range(3), 4, 10, np.random.default_rng(0))[1] == "none"


def test_batch_moments_match_per_subset_computation():
    rng = np.random.default_rng(7)
    K, M = 6, 7
    X = np.linspace(20, 80, M) + rng.normal(0, 3, (K, M))
    Y = 0.3 * X + rng.normal(0, 2, (K, M))
    subsets = np.array(list(itertools.combinations(range(M), 4)))
    est, loo = ins.batch_cross_moment(Y, X, subsets, loo=True)
    for k, c in enumerate(subsets):
        c = list(c)
        assert est[k] == pytest.approx(cross_moment(Y[:, c], X[:, c]))
        assert np.allclose(loo[k], iv.CrossMoment(Y[:, c], X[:, c]).loo())
    sl = ins.batch_slopes(X, Y, subsets)
    assert sl[0] == pytest.approx(cross_moment(Y[:, :4], X[:, :4]) / cross_moment(X[:, :4], X[:, :4]))


def test_subset_agreement_matches_pair_agreement_and_agreement_between():
    rng = np.random.default_rng(11)
    Y = rng.integers(0, 4, size=(5, 6)).astype(float)       # ties included
    same, tie = ins.pair_matrices_draws(Y)
    subsets = np.array(list(itertools.combinations(range(6), 4)))
    C = ins.subset_agreement(same, tie, subsets)
    for k, c in enumerate(subsets):
        assert C[k] == pytest.approx(pair_agreement(Y[:, list(c)])["C"])
    a, b = rng.integers(0, 3, 6).astype(float), rng.integers(0, 3, 6).astype(float)
    sb, tb = ins.pair_matrices_between(a, b)
    Cb = ins.subset_agreement(sb, tb, subsets)
    for k, c in enumerate(subsets):
        assert Cb[k] == pytest.approx(d1.agreement_between(a[list(c)], b[list(c)])["C"])


def test_subpanel_statistics_on_an_inverted_u():
    """Gain rises with baseline in the lower half and falls in the upper half: lower subpanels have positive slopes,
    upper ones negative, and own-gain orderings repeat exactly because the draws differ only by shifts."""
    x = np.array([5.0, 15.0, 25.0, 35.0, 65.0, 75.0, 85.0, 95.0])
    g = np.array([2.0, 12.0, 22.0, 32.0, 33.0, 23.0, 13.0, 3.0])
    sh = np.array([0.0, 1.0, -1.0, 2.0, 0.5])[:, None]
    X, Y = x + sh, g + 0.5 * sh
    low, high = np.array([[0, 1, 2, 3]]), np.array([[4, 5, 6, 7]])
    for sub, sign in ((low, 1), (high, -1)):
        st = ins.subpanel_statistics(X, Y, X, Y, sub, "fieller")
        assert np.sign(st["slope_discovery"][0]) == sign and st["slope_validation"][0] == pytest.approx(
            st["slope_discovery"][0])
        assert st["C_own_gain_disc_val"][0] == 1.0 and st["C_benefit_one_draw"][0] == 1.0
        assert st["set_sign"][0] == ("above zero" if sign > 0 else "below zero")
    st = ins.subpanel_statistics(X, Y, X, Y, np.vstack([low, high]), "jack_t")
    s = ins.summarize_subpanels(st, "all")
    assert s["share_negative"] == 0.5 and s["share_positive"] == 0.5 and s["share_disc_val_same_sign"] == 1.0
    chk = ins.all_subset_sign_shares(X, Y, range(8), 4)
    assert chk["n_subpanels"] == math.comb(8, 4) and chk["negative"] + chk["positive"] + chk["zero"] + \
        chk["undefined"] == chk["n_subpanels"]


# --------------------------------------------------------------------------- variant A


@pytest.mark.parametrize("text,answer,end_after", [
    ("So the total is \\boxed{18}.", "18", "\\boxed{18}"),
    ("Step 1.\nStep 2.\n$$\n\\boxed{18}\n$$", "18", "\\boxed{18}"),
    ("The cost is \\boxed{\\$1,000}", "1000", "\\boxed{\\$1,000}"),
    ("Answer: \\boxed{$18$}", "18", "\\boxed{$18$}"),
    ("a\r\nb \\boxed{5}\r\n\r\n", "5", "\\boxed{5}"),
    ("First \\boxed{x} then \\boxed{-7}", "-7", "\\boxed{-7}"),
])
def test_boxed_answer_end_locates_the_scored_box(text, answer, end_after):
    assert scoring.strict_extract(text, BOXED) == answer            # the scorer reads this answer
    end, n = ins.boxed_answer_end(text, answer)
    assert n == 1 and end == text.rindex(end_after) + len(end_after)


def test_boxed_answer_end_edge_cases():
    assert ins.boxed_answer_end("\\boxed{5}\nThe end.", "5") == (None, 0)      # not on the answer line
    end, n = ins.boxed_answer_end("\\boxed{5} or \\boxed{5}", "5")
    assert n == 2 and end == len("\\boxed{5} or \\boxed{5}")
    assert ins.boxed_answer_end(None, "5") == (None, 0)
    assert ins.boxed_answer_end("\\boxed{5}", None) == (None, 0)


def test_token_position_and_cap():
    offsets = [(0, 0), (0, 3), (3, 8), (8, 9), (9, 12)]          # a zero-width prefix token counts
    assert ins.token_position(offsets, 8) == 3 and ins.token_position(offsets, 9) == 4
    assert ins.token_position(offsets, None) == -1
    assert ins.capped_correct([1, 1, 0, 1], [10, 70, -1, 64], 64).tolist() == [1, 0, 0, 1]
    with pytest.raises(ValueError):
        ins.capped_correct([1], [-1], 64)


def test_long_direct_models_rule():
    assert ins.long_direct_models({"a": 64.0, "b": 64.5, "c": 100.0, "d": 7.0}, 64) == ["b", "c"]


def test_quadratic_curve_recovers_an_inverted_u():
    th = np.linspace(0, 100, 11)
    q = ins.quadratic_curve(th, 50 - 0.02 * (th - 40) ** 2)
    assert q["c2"] == pytest.approx(-0.02) and q["vertex_baseline"] == pytest.approx(40.0)
    assert q["vertex_gain"] == pytest.approx(50.0) and q["inverted_u"] and q["r2_quadratic"] == pytest.approx(1.0)
    q = ins.quadratic_curve(th, 3 + 0.5 * th)
    assert not q["inverted_u"] and q["linear_slope"] == pytest.approx(0.5)
    assert "note" in ins.quadratic_curve([1, 2, 3], [1, 2, 3])


def test_tex_formatting():
    assert ins.tex_num(-2.068, 2) == "$-$2.07" and ins.tex_num(0.27, 2, sign=True) == "$+$0.27"
    assert ins.tex_pct(0.3122) == "31\\%" and ins.tex_num(float("nan")) == "n/a"


def test_macro_names_are_letters_only():
    run = _load("run_instability")
    assert run.macro_name("Gsm", "Slope", "Points") == "instGsmSlopePoints"
    with pytest.raises(ValueError):
        run.macro_name("Gsm", "K24")


# --------------------------------------------------------------------------- analysis R, three scales


def _cheng_scored(drop_one=False):
    import pandas as pd
    zero = {"a": 1, "b": 3, "c": 5, "d": 7}        # correct of 10 items, zero-shot
    eight = {"a": 3, "b": 4, "c": 5, "d": 6}       # eight-shot
    rows = []
    for cfg in ("c1", "c2", "c3"):
        for mk in zero:
            for arm, k in (("zero_shot_cot", zero[mk]), ("eight_shot_cot", eight[mk])):
                if drop_one and (cfg, mk, arm) == ("c2", "b", "eight_shot_cot"):
                    continue
                rows += [{"config_id": cfg, "model_key": mk, "arm": arm, "item_id": i, "correct": int(i < k)}
                         for i in range(10)]
    return pd.DataFrame(rows)


def test_cheng_arrays_and_slopes_on_three_scales():
    run = _load("run_cheng_scales")
    configs, models, B, P, n = run.arm_arrays(_cheng_scored(), "zero_shot_cot", "eight_shot_cot")
    assert configs == ["c1", "c2", "c3"] and models == ["a", "b", "c", "d"] and n == 10
    assert B[0].tolist() == [10.0, 30.0, 50.0, 70.0] and P[0].tolist() == [30.0, 40.0, 50.0, 60.0]
    r = run.scale_rows(B, P, n, "jack_t", 0.05)
    assert r["points"]["per_configuration"] == pytest.approx([-0.5] * 3)
    assert r["points"]["mean"] == pytest.approx(-0.5) and r["points"]["equal_budget"] == pytest.approx(-0.5)
    assert r["points"]["n_negative"] == 3 and r["points"]["n_positive"] == 0
    cd = r["points"]["cross_draw"]
    assert cd["slope"] == pytest.approx(-0.5) and cd["R_benefit"] == pytest.approx(1.0)
    x = np.array([10.0, 30.0, 50.0, 70.0])
    er = 100 * (np.array([30.0, 40.0, 50.0, 60.0]) - x) / (100 - x)
    assert r["error_reduction"]["mean"] == pytest.approx(d1.slope(er, x))
    lb = np.log(x / (100 - x))
    p = np.array([30.0, 40.0, 50.0, 60.0])
    assert r["logit"]["mean"] == pytest.approx(d1.slope(np.log(p / (100 - p)) - lb, lb))
    one = run.single_rows(B[:1], P[:1], n)
    assert one["points"]["slope"] == pytest.approx(-0.5)


def test_cheng_refuses_a_missing_cell():
    run = _load("run_cheng_scales")
    with pytest.raises(SystemExit):
        run.arm_arrays(_cheng_scored(drop_one=True), "zero_shot_cot", "eight_shot_cot")
