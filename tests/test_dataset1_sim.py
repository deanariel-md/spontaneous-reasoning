"""Checks of the Dataset 1 pipeline (src/dataset1.py, controls.py, sampling.py) on simulated data.

Every Monte Carlo comparison uses a fixed seed and a 4-SE tolerance, so the outcomes are
deterministic. Arrays are (bank configurations C, models M) in percentage points; B is the
baseline arm, P the intervention arm, G = P - B. Subject-level counts nc0/nc1 have shape
(subjects S, configurations C, models M) and n_items shape (S,).
"""
import math

import numpy as np
import pytest

from src import controls, dataset1, sampling, simulate
from src.estimators import unit_moments


# --------------------------------------------------------------------------- helpers


def make_counts(seed, M=5, S=4, C=300):
    rng = np.random.default_rng(seed)
    nc0, nc1, n = simulate.synthetic_counts(rng, M=M, S=S, C=C)
    B, P = simulate.bank_from_counts(nc0, nc1, n)
    return nc0, nc1, n, B, P


@pytest.fixture(scope="module")
def counts300():
    return make_counts(seed=101, M=5, S=4, C=300)


@pytest.fixture(scope="module")
def counts200():
    return make_counts(seed=202, M=5, S=4, C=200)


def within(mean, truth, se, z=4.0, what=""):
    assert se > 0, f"{what}: nonpositive Monte Carlo SE"
    assert abs(mean - truth) <= z * se, (
        f"{what}: |{mean:.5g} - {truth:.5g}| = {abs(mean - truth):.5g} exceeds {z} SE = {z * se:.5g}")


def brute_agreement(X):
    """Agreement over ALL ordered pairs of bank rows (c, c'), c == c' included, equal pair weights."""
    X = np.asarray(X, float)
    C, M = X.shape
    per_pair = []
    for i in range(M):
        for j in range(i + 1, M):
            sg = [int(np.sign(X[c, i] - X[c, j])) for c in range(C)]
            same = opp = tie = 0
            for c in range(C):
                for c2 in range(C):
                    if sg[c] != 0 and sg[c] == sg[c2]:
                        same += 1
                    elif sg[c] != 0 and sg[c2] != 0:
                        opp += 1
                    else:
                        tie += 1
            tot = C * C
            per_pair.append((i, j, same / tot, opp / tot, tie / tot))
    same = float(np.mean([p[2] for p in per_pair]))
    opp = float(np.mean([p[3] for p in per_pair]))
    tie = float(np.mean([p[4] for p in per_pair]))
    return per_pair, same, opp, tie


def panel_cov(x, y):
    return float(np.cov(x, y, ddof=1)[0, 1])


# --------------------------------------------------------------------------- 1. exact_agreement


@pytest.mark.parametrize("kind", ["integer", "continuous", "integer_with_constant_pair"])
def test_exact_agreement_matches_brute_force(kind):
    rng = np.random.default_rng(11)
    if kind == "integer":
        X = rng.integers(0, 4, size=(7, 4)).astype(float)
    elif kind == "continuous":
        X = rng.normal(size=(7, 4)) * 10
    else:
        X = rng.integers(0, 3, size=(7, 4)).astype(float)
        X[:, 3] = X[:, 2]            # one model pair tied on every row
    per_pair, same, opp, tie = brute_agreement(X)
    if kind != "continuous":
        assert tie > 0, "the integer-valued example must contain ties"
    res = dataset1.exact_agreement(X)
    assert res["same"] == pytest.approx(same, abs=1e-12)
    assert res["opposite"] == pytest.approx(opp, abs=1e-12)
    assert res["tie"] == pytest.approx(tie, abs=1e-12)
    assert res["C"] == pytest.approx(same + 0.5 * tie, abs=1e-12)
    assert res["decisive"] == pytest.approx(same + opp, abs=1e-12)
    assert res["conditional_agreement"] == pytest.approx(same / (same + opp), abs=1e-12)
    assert len(res["per_pair"]) == len(per_pair) == 6
    for got, (i, j, s, o, t) in zip(res["per_pair"], per_pair):
        assert (got["i"], got["j"]) == (i, j)
        assert got["same"] == pytest.approx(s, abs=1e-12)
        assert got["opposite"] == pytest.approx(o, abs=1e-12)
        assert got["tie"] == pytest.approx(t, abs=1e-12)
        assert got["same"] + got["opposite"] + got["tie"] == pytest.approx(1.0, abs=1e-12)


def test_exact_agreement_all_tied_gives_nan_conditional():
    res = dataset1.exact_agreement(np.ones((5, 3)))
    assert res["tie"] == 1.0 and res["same"] == 0.0 and res["opposite"] == 0.0
    assert res["C"] == 0.5
    assert math.isnan(res["conditional_agreement"])


# --------------------------------------------------------------------------- 2. mc_agreement_curve at K = 1


def test_mc_agreement_curve_k1_matches_exact(counts300):
    _, _, _, B, P = counts300
    G = P - B
    rows = dataset1.mc_agreement_curve(B, P, [1], 3000, np.random.default_rng(12))
    assert {r["quantity"] for r in rows} == {"benefit", "accuracy"}
    for r in rows:
        assert r["K"] == 1
        X = G if r["quantity"] == "benefit" else B
        ex = dataset1.exact_agreement(X)
        within(r["C"], ex["C"], r["C_mc_se"], what=f"K=1 agreement C for {r['quantity']}")
        assert r["same"] + r["opposite"] + r["tie"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- 3. agreement_between


def test_agreement_between_hand_cases():
    f = dataset1.agreement_between
    r = f([1, 2, 3], [1, 2, 3])                       # strict same
    assert (r["same"], r["opposite"], r["tie"], r["C"], r["conditional_agreement"]) == (1.0, 0.0, 0.0, 1.0, 1.0)
    r = f([1, 2, 3], [3, 2, 1])                       # strict opposite
    assert (r["same"], r["opposite"], r["tie"], r["C"], r["conditional_agreement"]) == (0.0, 1.0, 0.0, 0.0, 0.0)
    r = f([1, 2, 3], [1, 3, 2])                       # one reversed pair of three
    assert r["same"] == pytest.approx(2 / 3) and r["opposite"] == pytest.approx(1 / 3)
    assert r["tie"] == 0.0 and r["C"] == pytest.approx(2 / 3) and r["conditional_agreement"] == pytest.approx(2 / 3)
    r = f([1, 1, 3], [1, 2, 3])                       # tie in the first vector only
    assert r["same"] == pytest.approx(2 / 3) and r["opposite"] == 0.0 and r["tie"] == pytest.approx(1 / 3)
    assert r["C"] == pytest.approx(2 / 3 + 1 / 6) and r["conditional_agreement"] == 1.0
    r = f([1, 2, 3], [2, 2, 1])                       # tie in the second vector only, rest opposite
    assert r["same"] == 0.0 and r["opposite"] == pytest.approx(2 / 3) and r["tie"] == pytest.approx(1 / 3)
    assert r["C"] == pytest.approx(1 / 6) and r["conditional_agreement"] == 0.0
    r = f([1, 1, 1], [5, 5, 5])                       # everything tied
    assert r["tie"] == 1.0 and r["C"] == 0.5 and math.isnan(r["conditional_agreement"])
    r = f([1, 1, 2], [3, 3, 3])                       # tie wherever either vector ties
    assert r["tie"] == 1.0 and r["same"] == 0.0 and r["opposite"] == 0.0


# --------------------------------------------------------------------------- 4. exact_targets


def test_exact_targets_columns_and_panel_moments(counts300):
    _, _, _, B, P = counts300
    G = P - B
    t = dataset1.exact_targets(B, P)
    theta, delta, pi = B.mean(0), G.mean(0), P.mean(0)
    assert np.allclose(t["theta"], theta) and np.allclose(t["delta"], delta) and np.allclose(t["pi"], pi)
    assert np.allclose(np.array(t["pi"]) - np.array(t["theta"]), t["delta"])
    assert t["V_theta"] == pytest.approx(np.var(theta, ddof=1), rel=1e-12)
    assert t["V_delta"] == pytest.approx(np.var(delta, ddof=1), rel=1e-12)
    assert t["A"] == pytest.approx(panel_cov(delta, theta), rel=1e-12)
    assert t["mean_gain"] == pytest.approx(delta.mean()) and t["mean_baseline"] == pytest.approx(theta.mean())
    assert t["beta"] == pytest.approx(t["A"] / t["V_theta"], rel=1e-12)
    assert t["H"] == pytest.approx(-10 * t["beta"], rel=1e-12)
    assert t["gain_sd"] == pytest.approx(math.sqrt(t["V_delta"]))


def test_exact_targets_coupling_identity(counts300):
    _, _, _, B, P = counts300
    c = dataset1.exact_targets(B, P)["coupling"]
    assert np.isclose(c["E_cov_G_B_one_draw"], c["identity_rhs"], rtol=1e-10, atol=0)
    # also on a deliberately rough bank with strongly heterogeneous rows
    rng = np.random.default_rng(13)
    B2 = rng.uniform(0, 100, size=(40, 6))
    P2 = np.clip(B2 + rng.normal(5, 15, size=(40, 6)), 0, 100)
    c2 = dataset1.exact_targets(B2, P2)["coupling"]
    assert np.isclose(c2["E_cov_G_B_one_draw"], c2["identity_rhs"], rtol=1e-10, atol=0)


def test_exact_targets_identical_theta_gives_nan_slope(counts300):
    rng = np.random.default_rng(14)
    C, M = 9, 4
    base = rng.integers(0, 100, size=C).astype(float)
    B = np.stack([np.roll(base, m) for m in range(M)], axis=1)       # identical column means (exact integers)
    P = B + rng.integers(-3, 8, size=(C, M)) + np.arange(M)            # different gains per model
    assert np.all(B.mean(0) == B.mean(0)[0])
    t = dataset1.exact_targets(B, P)                                    # must not raise
    assert t["V_theta"] == 0.0
    assert math.isnan(t["beta"]) and math.isnan(t["H"])
    assert t["R_accuracy"] == 0.0
    assert math.isfinite(t["A"]) and math.isfinite(t["V_delta"])


# --------------------------------------------------------------------------- 5. unbiasedness of the replicated moments


def test_unit_moments_unbiased_over_sampler(counts300):
    _, _, _, B, P = counts300
    C = B.shape[0]
    ex = dataset1.exact_targets(B, P)
    n_rep, K = 3000, 6
    idx = np.random.default_rng(15).integers(0, C, size=(n_rep, K))
    vals = {"V_theta": [], "A": [], "V_delta": []}
    for r in range(n_rep):
        q = unit_moments(B[idx[r]], P[idx[r]])
        for key in vals:
            vals[key].append(q[key])
    for key, v in vals.items():
        v = np.array(v)
        within(v.mean(), ex[key], v.std(ddof=1) / math.sqrt(n_rep), what=f"E[{key}] over K={K} draws")


def test_mc_moment_distribution_centers_on_exact_targets(counts300):
    _, _, _, B, P = counts300
    ex = dataset1.exact_targets(B, P)
    res = dataset1.mc_moment_distribution(B, P, 6, 3000, np.random.default_rng(16))
    assert res["K"] == 6 and res["n_rep"] == 3000
    for key in ("V_theta", "A", "V_delta"):
        within(res[key]["mean"], ex[key], res[key]["mc_se"], what=f"mc_moment_distribution {key}")
        assert res[key]["n_undefined"] == 0
        assert res[key]["q025"] <= res[key]["q50"] <= res[key]["q975"]
    assert 0.0 <= res["V_delta"]["P_above_margin"] <= 1.0


# --------------------------------------------------------------------------- 6. evaluate_predictions


def _predictions(B, P, k=10, seed=17):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B.shape[0], k)
    return {n: v[0] for n, v in dataset1.fit_predictors(B[idx], P[idx]).items()}


def test_evaluate_predictions_full_bank_equals_exact_conditional(counts300):
    _, _, _, B, P = counts300
    G = P - B
    delta = G.mean(0)
    preds = _predictions(B, P)
    out = dataset1.evaluate_predictions(G, preds, delta=delta)
    assert len(out["comparisons"]) == len(dataset1.COMPARISONS)
    for row in out["comparisons"]:
        assert np.isclose(row["estimate"], row["exact_conditional"], rtol=1e-9, atol=1e-9), row
        lo, hi = row["provisional_ci"]
        assert lo <= row["estimate"] <= hi
    for name, f in preds.items():
        assert out["risks"][name]["exact_risk"] == pytest.approx(np.mean((delta - f) ** 2))
        assert out["risks"][name]["naive_mse"] == pytest.approx(np.mean((G - f) ** 2))
        assert np.isfinite(out["risks"][name]["cross_risk"])


def test_evaluate_predictions_positive_favors_second_predictor(counts300):
    _, _, _, B, P = counts300
    G = P - B
    delta = G.mean(0)
    rng = np.random.default_rng(18)
    Z = G[rng.integers(0, G.shape[0], 10)]
    # g (own_gain) equals delta, f (constant) is 20 pp away: g is better, difference positive
    preds = {"constant": delta + 20.0, "own_gain": delta.copy()}
    out = dataset1.evaluate_predictions(Z, preds, delta=delta)
    assert len(out["comparisons"]) == 1
    row = out["comparisons"][0]
    assert (row["f"], row["g"]) == ("constant", "own_gain")
    assert row["estimate"] > 0
    assert row["exact_conditional"] == pytest.approx(400.0)
    assert row["provisional_ci"][0] > 0
    # swap roles: f is now the accurate one, so the difference is negative
    preds = {"own_gain": delta.copy(), "replicated": delta + 20.0}
    row = dataset1.evaluate_predictions(Z, preds, delta=delta)["comparisons"][0]
    assert (row["f"], row["g"]) == ("own_gain", "replicated")
    assert row["estimate"] < 0 and row["exact_conditional"] == pytest.approx(-400.0)
    assert row["provisional_ci"][1] < 0


def test_evaluate_predictions_skips_missing_predictors_and_omits_exact_without_delta(counts300):
    _, _, _, B, P = counts300
    G = P - B
    Z = G[:8]
    preds = {"constant": np.full(5, 3.0), "own_gain": G.mean(0)}
    out = dataset1.evaluate_predictions(Z, preds)
    assert [(c["f"], c["g"]) for c in out["comparisons"]] == [("constant", "own_gain")]
    assert "exact_conditional" not in out["comparisons"][0]
    assert "exact_risk" not in out["risks"]["constant"]


# --------------------------------------------------------------------------- 7. fit_predictors


def _disc(seed=19, K=6, M=5):
    rng = np.random.default_rng(seed)
    ability = np.linspace(30, 70, M)
    Bd = ability + rng.normal(0, 6, size=(K, M)) + rng.normal(0, 4, size=(K, 1))
    Pd = Bd + np.linspace(8, 2, M) + rng.normal(0, 4, size=(K, M))
    return Bd, Pd


def test_fit_predictors_constant_and_own_gain():
    Bd, Pd = _disc()
    Gd = Pd - Bd
    out = dataset1.fit_predictors(Bd, Pd)
    assert set(out) == set(dataset1.PREDICTORS)
    Gb = Gd.mean(0)
    assert np.allclose(out["constant"][0], np.full(Bd.shape[1], Gb.mean()))
    assert np.allclose(out["own_gain"][0], Gb)


def test_fit_predictors_averaged_ols_matches_polyfit():
    Bd, Pd = _disc()
    Gb, Bb = (Pd - Bd).mean(0), Bd.mean(0)
    slope_, intercept = np.polyfit(Bb, Gb, 1)
    pred, info = dataset1.fit_predictors(Bd, Pd)["averaged_ols"]
    assert np.allclose(pred, slope_ * Bb + intercept)
    assert info["slope"] == pytest.approx(slope_) and info["fallback"] is None


def test_fit_predictors_single_prompt_uses_first_row_only():
    Bd, Pd = _disc()
    base = dataset1.fit_predictors(Bd, Pd)
    b0, g0 = Bd[0], (Pd - Bd)[0]
    slope_, intercept = np.polyfit(b0, g0, 1)
    assert np.allclose(base["single_prompt"][0], slope_ * b0 + intercept)
    assert base["single_prompt"][1]["units_used"] == 1

    rng = np.random.default_rng(20)
    Bd2, Pd2 = Bd.copy(), Pd.copy()
    Bd2[1:] = rng.normal(50, 20, size=Bd2[1:].shape)
    Pd2[1:] = Bd2[1:] + rng.normal(5, 9, size=Bd2[1:].shape)
    changed = dataset1.fit_predictors(Bd2, Pd2)
    assert np.array_equal(changed["single_prompt"][0], base["single_prompt"][0])
    assert not np.allclose(changed["own_gain"][0], base["own_gain"][0])      # the change is visible elsewhere
    assert not np.allclose(changed["constant"][0], base["constant"][0])


def test_fit_predictors_single_row_has_no_replicated():
    Bd, Pd = _disc()
    out = dataset1.fit_predictors(Bd[:1], Pd[:1])
    assert "replicated" not in out
    assert set(out) == {"constant", "own_gain", "averaged_ols", "single_prompt"}
    assert all(np.all(np.isfinite(v[0])) for v in out.values())
    assert np.allclose(out["averaged_ols"][0], out["single_prompt"][0])


@pytest.mark.parametrize("K", [1, 4])
def test_fit_predictors_constant_scores_are_finite(K):
    M = 5
    Bd = np.full((K, M), 50.0)
    Pd = np.full((K, M), 55.0)
    out = dataset1.fit_predictors(Bd, Pd)
    for name, (pred, info) in out.items():
        assert np.all(np.isfinite(pred)), name
        assert np.allclose(pred, 5.0), name
    assert out["averaged_ols"][1]["fallback"] == "constant"
    assert out["single_prompt"][1]["fallback"] == "constant"
    if K >= 2:
        assert out["replicated"][1]["fallback"] == "constant"


def test_fit_predictors_constant_across_draws_but_varying_across_models_is_finite():
    K, M = 4, 5
    Bd = np.tile(np.linspace(30, 70, M), (K, 1))
    Pd = Bd + 4.0                                  # constant gain: no benefit heterogeneity
    out = dataset1.fit_predictors(Bd, Pd)
    for name, (pred, _) in out.items():
        assert np.all(np.isfinite(pred)), name
        assert np.allclose(pred, 4.0, atol=1e-6), name


# --------------------------------------------------------------------------- 8. mc_prediction


def test_mc_prediction_shape_and_risk_decreases_with_k(counts200):
    _, _, _, B, P = counts200
    ks = [2, 4, 8]
    res = dataset1.mc_prediction(B, P, 8, 200, np.random.default_rng(21), ks=ks)
    seen = {(c["K"], c["f"], c["g"]) for c in res["comparisons"]}
    expected = {(k, f, g) for k in ks for f, g in dataset1.COMPARISONS}
    assert seen == expected
    for c in res["comparisons"]:
        assert np.isfinite(c["mean"]) and np.isfinite(c["mc_se"]) and c["mc_se"] >= 0
        assert c["q025"] <= c["q975"]
        assert 0.0 <= c["P_g_better_by_margin"] <= 1.0 and 0.0 <= c["P_f_better_by_margin"] <= 1.0
    risks = {(r["K"], r["predictor"]): r for r in res["risks"]}
    assert set(risks) == {(k, p) for k in ks for p in dataset1.PREDICTORS}
    assert all(np.isfinite(r["mean_exact_risk"]) for r in res["risks"])
    own = [risks[(k, "own_gain")] for k in ks]
    for a, b in zip(own[:-1], own[1:]):
        allowance = 3 * math.hypot(a["mc_se"], b["mc_se"])
        assert b["mean_exact_risk"] <= a["mean_exact_risk"] + allowance, (a, b)
    assert own[-1]["mean_exact_risk"] < own[0]["mean_exact_risk"]


def test_mc_prediction_defaults_to_single_budget(counts200):
    _, _, _, B, P = counts200
    res = dataset1.mc_prediction(B, P, 4, 30, np.random.default_rng(22))
    assert {c["K"] for c in res["comparisons"]} == {4}
    assert {r["K"] for r in res["risks"]} == {4}


# --------------------------------------------------------------------------- 9. controls.pooled


def test_pooled_equal_weight_matches_manual():
    rng = np.random.default_rng(23)
    S, C, M = 3, 6, 4
    n = np.array([50, 80, 120])
    nc = rng.integers(0, 51, size=(S, C, M)) * (n / 50).reshape(S, 1, 1)
    nc = np.floor(nc).astype(int)
    manual = 100 * sum(nc[s] / n[s] for s in range(S)) / S
    got = controls.pooled(nc, n)
    assert got.shape == (C, M)
    assert np.allclose(got, manual)


@pytest.mark.parametrize("weights", [[1, 2, 5], [0.2, 0.3, 0.5], [3, 0, 1]])
def test_pooled_unequal_weights_matches_manual(weights):
    rng = np.random.default_rng(24)
    S, C, M = 3, 6, 4
    n = np.array([50, 80, 120])
    nc = np.floor(rng.uniform(0, 1, size=(S, C, M)) * n.reshape(S, 1, 1)).astype(int)
    w = np.asarray(weights, float)
    w = w / w.sum()
    manual = 100 * sum(w[s] * nc[s] / n[s] for s in range(S))
    assert np.allclose(controls.pooled(nc, n, weights), manual)


def test_pooled_two_dimensional_counts():
    nc = np.array([[10, 20, 30], [40, 40, 10]])
    n = np.array([40, 80])
    manual = 100 * (nc[0] / 40 + nc[1] / 80) / 2
    assert np.allclose(controls.pooled(nc, n), manual)


# --------------------------------------------------------------------------- 10. null controls


@pytest.mark.parametrize("population", ["baseline", "intervention"])
def test_null_controls_moments_unbiased(counts200, population):
    nc0, nc1, n, B, P = counts200
    theta = B.mean(0)
    scenario = {"kind": "null", "population": population, "name": f"null_{population}"}
    res = controls.run_control(nc0, nc1, n, theta, scenario, 8, 400, np.random.default_rng(25))
    tr = res["truth"]
    assert tr["V_delta"] == 0 and tr["A"] == 0 and tr["H"] == 0
    # the truth follows whichever arm's bank the scenario samples from
    arm_bank = B if population == "baseline" else P
    assert tr["V_theta"] == pytest.approx(np.var(arm_bank.mean(0), ddof=1))
    assert res["d"] == [0.0] * len(theta)
    assert res["V_delta"]["n_undefined"] == 0
    within(res["V_delta"]["mean"], 0.0, res["V_delta"]["mc_se"], what=f"null/{population} V_delta")
    within(res["A"]["mean"], 0.0, res["A"]["mc_se"], what=f"null/{population} A")
    assert abs(res["V_theta"]["bias_z"]) < 4
    within(res["V_theta"]["mean"], np.var(arm_bank.mean(0), ddof=1), res["V_theta"]["mc_se"],
           what=f"null/{population} V_theta of the sampled arm")


@pytest.mark.parametrize("population", ["baseline", "intervention"])
def test_null_controls_naive_single_unit_h_is_pushed_positive(counts200, population):
    """The naive slope regresses G = e_P - e_B on B = theta + e_B, so Cov(G, B) contains -Var(e_B)
    and H_naive = -10 * beta is positive under the null. The effect is small next to the spread of
    the single-unit ratio in this bank (about 0.35 against an SD of 1.6), so this check needs
    several thousand replications; 400 give only about 50 percent power at 4 SE."""
    nc0, nc1, n, B, P = counts200
    theta = B.mean(0)
    scenario = {"kind": "null", "population": population, "name": f"null_{population}"}
    res = controls.run_control(nc0, nc1, n, theta, scenario, 8, 4000, np.random.default_rng(32))
    naive = res["H_naive"]
    assert naive["n_undefined"] == 0
    assert naive["mean"] > 0
    assert naive["bias"] > 4 * naive["mc_se"], naive
    # the replicated estimator is far closer to the truth than the naive one
    assert abs(res["H_replicated"]["bias"]) < naive["bias"]
    # the same replications leave V_delta and A centered on zero
    within(res["V_delta"]["mean"], 0.0, res["V_delta"]["mc_se"], what=f"null/{population} V_delta")
    within(res["A"]["mean"], 0.0, res["A"]["mc_se"], what=f"null/{population} A")


def test_draw_control_units_null_has_zero_truth_and_arm_shapes(counts200):
    nc0, nc1, n, B, P = counts200
    for pop, src in (("baseline", nc0), ("intervention", nc1)):
        Bu, Pu, d = controls.draw_control_units(nc0, nc1, n, 5, {"kind": "null", "population": pop},
                                                 B.mean(0), np.random.default_rng(26))
        assert Bu.shape == Pu.shape == (5, nc0.shape[2])
        assert np.all(d == 0)
        # every returned value is a possible pooled bank value of the selected arm
        bank = controls.pooled(src, n)
        for arr in (Bu, Pu):
            assert all(np.any(np.all(np.isclose(bank, row), axis=1)) for row in arr)


# --------------------------------------------------------------------------- 11. injected controls


def _injected_scenarios(theta):
    u = controls.residual_direction(theta, 3.0, np.random.default_rng(27))
    return {
        "capability": {"kind": "capability", "gain": 5.0, "beta": -0.2, "name": "capability"},
        "constant": {"kind": "constant", "gain": 5.0, "name": "constant"},
        "residual": {"kind": "residual", "gain": 3.0, "u": u, "name": "residual"},
    }


@pytest.mark.parametrize("name", ["capability", "constant", "residual"])
def test_injected_controls_moments_match_truth(counts200, name):
    nc0, nc1, n, B, P = counts200
    theta = B.mean(0)
    scenario = _injected_scenarios(theta)[name]
    d = controls.scenario_effects(theta, scenario)
    res = controls.run_control(nc0, nc1, n, theta, scenario, 8, 400, np.random.default_rng(28))
    assert np.allclose(res["d"], d)
    assert res["truth"]["V_delta"] == pytest.approx(np.var(d, ddof=1))
    assert res["truth"]["A"] == pytest.approx(panel_cov(d, theta), abs=1e-9)
    for key in ("V_delta", "A", "V_theta"):
        within(res[key]["mean"], res["truth"][key], res[key]["mc_se"], what=f"{name} {key}")
        assert res[key]["bias"] == pytest.approx(res[key]["mean"] - res["truth"][key])
    assert res["V_delta"]["n_undefined"] == 0


def test_capability_scenario_truth_slope():
    theta = np.array([30.0, 40.0, 50.0, 60.0, 70.0])
    d = controls.scenario_effects(theta, {"kind": "capability", "gain": 5.0, "beta": -0.2})
    assert d.mean() == pytest.approx(5.0)
    assert panel_cov(d, theta) / np.var(theta, ddof=1) == pytest.approx(-0.2)


def test_residual_scenario_is_orthogonal_with_requested_sd(counts200):
    _, _, _, B, _ = counts200
    theta = B.mean(0)
    u = controls.residual_direction(theta, 3.0, np.random.default_rng(29))
    assert np.std(u, ddof=1) == pytest.approx(3.0, rel=1e-12)
    assert u.mean() == pytest.approx(0.0, abs=1e-12)
    assert panel_cov(u, theta) == pytest.approx(0.0, abs=1e-9)
    d = controls.scenario_effects(theta, {"kind": "residual", "gain": 3.0, "u": u})
    assert panel_cov(d, theta) == pytest.approx(0.0, abs=1e-9)
    assert np.var(d, ddof=1) == pytest.approx(9.0, rel=1e-12)


# --------------------------------------------------------------------------- 12. draw_control_units for injected scenarios


@pytest.mark.parametrize("name", ["capability", "constant", "residual", "sparse_outlier"])
def test_draw_control_units_injected_gain_matches_d(counts200, name):
    nc0, nc1, n, B, P = counts200
    theta = B.mean(0)
    scenarios = _injected_scenarios(theta)
    scenarios["sparse_outlier"] = {"kind": "sparse_outlier", "gain": 2.0, "index": 1, "extra": 6.0}
    scenario = scenarios[name]
    k = 4000
    Bu, Pu, d = controls.draw_control_units(nc0, nc1, n, k, scenario, theta, np.random.default_rng(30))
    assert Bu.shape == Pu.shape == (k, nc0.shape[2])
    assert np.allclose(d, controls.scenario_effects(theta, scenario))
    G = Pu - Bu
    se = G.std(axis=0, ddof=1) / math.sqrt(k)
    for m in range(len(d)):
        within(G[:, m].mean(), d[m], se[m], what=f"{name} gain, model {m}")
    seB = Bu.std(axis=0, ddof=1) / math.sqrt(k)
    for m in range(len(d)):
        within(Bu[:, m].mean(), theta[m], seB[m], what=f"{name} baseline, model {m}")
    # outcomes stay valid accuracies
    assert Pu.min() >= 0 and Pu.max() <= 100


# --------------------------------------------------------------------------- 13. infeasible injections


@pytest.mark.parametrize("scenario", [
    {"kind": "constant", "gain": 99.0},
    {"kind": "constant", "gain": -99.0},
    {"kind": "capability", "gain": 5.0, "beta": -5.0},
    {"kind": "sparse_outlier", "gain": 0.0, "index": 0, "extra": 80.0},
])
def test_infeasible_injection_raises(counts200, scenario):
    nc0, nc1, n, B, P = counts200
    theta = B.mean(0)
    with pytest.raises(ValueError):
        controls.draw_control_units(nc0, nc1, n, 4, scenario, theta, np.random.default_rng(31))
    with pytest.raises(ValueError):
        controls.run_control(nc0, nc1, n, theta, scenario, 4, 3, np.random.default_rng(31))


def test_unknown_scenario_kind_raises():
    with pytest.raises(ValueError):
        controls.scenario_effects(np.array([40.0, 50.0]), {"kind": "bogus"})


# --------------------------------------------------------------------------- 14. sampling


def test_registered_split_reproducible_and_label_specific():
    a = sampling.registered_split(1000, 24, 40, "benchmark_a")
    a2 = sampling.registered_split(1000, 24, 40, "benchmark_a")
    b = sampling.registered_split(1000, 24, 40, "benchmark_b")
    assert a == a2
    assert a["discovery_bank_index"] != b["discovery_bank_index"]
    assert a["validation_bank_index"] != b["validation_bank_index"]
    assert len(a["discovery_bank_index"]) == 24 and len(a["validation_bank_index"]) == 40
    assert a["k_discovery"] == 24 and a["k_validation"] == 40 and a["n_bank"] == 1000
    assert a["master_seed"] == sampling.MASTER_SEED
    for s in (a, b):
        for key in ("discovery_bank_index", "validation_bank_index"):
            idx = np.array(s[key])
            assert idx.min() >= 0 and idx.max() < 1000
    # discovery and validation are separate draws, not a repeat of one stream
    assert a["discovery_bank_index"][:24] != a["validation_bank_index"][:24]


def test_registered_split_indices_in_range_for_small_bank():
    s = sampling.registered_split(3, 50, 50, "tiny")
    idx = np.array(s["discovery_bank_index"] + s["validation_bank_index"])
    assert set(idx.tolist()) <= {0, 1, 2}
    assert set(idx.tolist()) == {0, 1, 2}          # with replacement: all three appear in 100 draws


def test_rng_for_reproducible_and_independent():
    x1 = sampling.rng_for("a", "b").integers(0, 10 ** 9, 5)
    x2 = sampling.rng_for("a", "b").integers(0, 10 ** 9, 5)
    y = sampling.rng_for("a", "c").integers(0, 10 ** 9, 5)
    assert np.array_equal(x1, x2) and not np.array_equal(x1, y)


def test_budgets():
    assert sampling.budgets(24) == [1, 2, 4, 8, 12, 16, 24]
    assert sampling.budgets(12, start=2) == [2, 4, 8, 12]
    assert sampling.budgets(10) == [1, 2, 4, 8]
    assert sampling.budgets(1) == [1]
    assert sampling.budgets(0) == []


def test_sha256_json_is_order_independent():
    assert sampling.sha256_json({"a": 1, "b": [1, 2]}) == sampling.sha256_json({"b": [1, 2], "a": 1})
    assert sampling.sha256_json({"a": 1}) != sampling.sha256_json({"a": 2})
