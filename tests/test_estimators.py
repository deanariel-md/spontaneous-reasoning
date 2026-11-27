"""Verification record (Appendix B) for src/estimators.py.

Each test docstring quotes the record item it reproduces. Arrays are
(draws K, models M) in percentage points; panel moments use denominator M - 1.
"""
import math

import numpy as np
import pytest
from scipy.stats import norm

from src.estimators import (
    _array,
    cross_moment,
    cross_risk,
    fixed_task_components,
    gaussian_agreement,
    pair_agreement,
    paired_loss_difference,
    prompts_needed_gaussian,
    reliability,
    replicated_linear_predictor,
    unit_moments,
)

RTOL = 1e-12
ATOL = 1e-12


# ---------------------------------------------------------------------------
# B1. Cross-draw moments
# ---------------------------------------------------------------------------

def _enumerated_cross_moment(X, Y):
    """Mean over ordered draw pairs r != s of the panel cross-covariance."""
    K, M = X.shape
    vals = []
    for r in range(K):
        for s in range(K):
            if r == s:
                continue
            xm, ym = X[r].mean(), Y[s].mean()
            vals.append(sum((X[r, i] - xm) * (Y[s, i] - ym) for i in range(M)) / (M - 1))
    return float(np.mean(vals))


@pytest.mark.parametrize("K", [2, 3, 7])
@pytest.mark.parametrize("M", [2, 5, 11])
def test_b1_cross_moment_matches_enumeration(K, M):
    """B1: "Vectorized cross-draw moments agree with direct enumeration of
    distinct draw pairs." """
    rng = np.random.default_rng(1000 + 17 * K + M)
    X = rng.normal(50, 10, size=(K, M))
    Y = 0.5 * X + rng.normal(0, 8, size=(K, M))   # correlated with X, nonzero cross moment
    for A, B in [(X, Y), (X, X), (Y, Y), (Y, X)]:
        assert np.isclose(cross_moment(A, B), _enumerated_cross_moment(A, B),
                          rtol=RTOL, atol=ATOL)
    # The ordered-pair mean is symmetric in its arguments.
    assert np.isclose(cross_moment(X, Y), cross_moment(Y, X), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("K,M", [(2, 3), (4, 6), (7, 11)])
def test_b1_unit_moments_are_cross_moments(K, M):
    """B1: "Vectorized cross-draw moments agree with direct enumeration of
    distinct draw pairs." unit_moments(B, P) equals cross_moment of (B,B),
    (P-B,B) and (P-B,P-B)."""
    rng = np.random.default_rng(2000 + K * 100 + M)
    B = rng.normal(55, 12, size=(K, M))
    P = B + rng.normal(3, 4, size=(K, M))
    G = P - B
    q = unit_moments(B, P)
    assert set(q) == {"V_theta", "A", "V_delta"}
    assert np.isclose(q["V_theta"], cross_moment(B, B), rtol=RTOL, atol=ATOL)
    assert np.isclose(q["A"], cross_moment(G, B), rtol=RTOL, atol=ATOL)
    assert np.isclose(q["V_delta"], cross_moment(G, G), rtol=RTOL, atol=ATOL)
    # and against direct enumeration
    assert np.isclose(q["A"], _enumerated_cross_moment(G, B), rtol=RTOL, atol=ATOL)
    assert np.isclose(q["V_delta"], _enumerated_cross_moment(G, G), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("bad", [
    np.ones((1, 5)),                         # K < 2
    np.ones((5, 1)),                         # M < 2
    np.array([[1.0, 2.0], [np.nan, 4.0]]),   # NaN
    np.array([[1.0, 2.0], [np.inf, 4.0]]),   # +inf
    np.array([[1.0, 2.0], [-np.inf, 4.0]]),  # -inf
    np.ones(5),                              # 1-D
    np.ones((2, 3, 4)),                      # 3-D
])
def test_b1_array_rejects_invalid_input(bad):
    """B1: "Vectorized cross-draw moments agree with direct enumeration of
    distinct draw pairs." Input validation: _array rejects K<2, M<2,
    non-finite values and wrong rank."""
    with pytest.raises(ValueError):
        _array(bad)
    with pytest.raises(ValueError):
        cross_moment(bad, bad)
    with pytest.raises(ValueError):
        reliability(bad)
    with pytest.raises(ValueError):
        pair_agreement(bad)


def test_b1_array_accepts_minimal_valid_input():
    """B1: "Vectorized cross-draw moments agree with direct enumeration of
    distinct draw pairs." The smallest legal layout is (2, 2)."""
    out = _array([[1, 2], [3, 5]])
    assert out.shape == (2, 2) and out.dtype == float


def test_b1_shape_mismatch_raises():
    """B1: "Vectorized cross-draw moments agree with direct enumeration of
    distinct draw pairs." Shape mismatch raises ValueError."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4, 5))
    with pytest.raises(ValueError):
        cross_moment(X, rng.normal(size=(4, 6)))
    with pytest.raises(ValueError):
        cross_moment(X, rng.normal(size=(5, 5)))
    with pytest.raises(ValueError):
        unit_moments(X, rng.normal(size=(4, 6)))
    with pytest.raises(ValueError):
        unit_moments(X, rng.normal(size=(3, 5)))


# ---------------------------------------------------------------------------
# B2. Tie-aware agreement
# ---------------------------------------------------------------------------

def _enumerated_agreement(G):
    """Per model pair i<j, over ordered draw pairs r != s; equal weight per model pair."""
    K, M = G.shape
    same_l, opp_l, tie_l = [], [], []
    for i in range(M):
        for j in range(i + 1, M):
            s = o = t = n = 0
            for r in range(K):
                for q in range(K):
                    if r == q:
                        continue
                    a = np.sign(G[r, i] - G[r, j])
                    b = np.sign(G[q, i] - G[q, j])
                    n += 1
                    if a != 0 and b != 0 and a == b:
                        s += 1
                    elif a != 0 and b != 0:
                        o += 1
                    else:
                        t += 1
            same_l.append(s / n)
            opp_l.append(o / n)
            tie_l.append(t / n)
    same, opp, tie = float(np.mean(same_l)), float(np.mean(opp_l)), float(np.mean(tie_l))
    decisive = same + opp
    cond = same / decisive if decisive > 0 else float("nan")
    return {"same": same, "opposite": opp, "tie": tie,
            "C": same + 0.5 * tie, "conditional_agreement": cond}


@pytest.mark.parametrize("K,M,high", [
    (2, 4, 3), (3, 5, 3), (5, 6, 3), (8, 4, 3), (6, 7, 3),   # many ties
    (4, 6, 10), (7, 5, 10),                                  # few ties
])
def test_b2_pair_agreement_matches_enumeration_with_ties(K, M, high):
    """B2: "Tie-aware agreement agrees with enumeration; an all-tie tensor has
    C = 0.5, a tie fraction of one, and undefined strict agreement." """
    rng = np.random.default_rng(4000 + 31 * K + 7 * M + high)
    G = rng.integers(0, high, size=(K, M)).astype(float)
    got = pair_agreement(G)
    want = _enumerated_agreement(G)
    if high == 3:
        assert want["tie"] > 0, "construction must contain ties"
    for key in ("same", "opposite", "tie", "C"):
        assert np.isclose(got[key], want[key], rtol=RTOL, atol=ATOL), key
    assert np.isclose(got["conditional_agreement"], want["conditional_agreement"],
                      rtol=RTOL, atol=ATOL, equal_nan=True)
    assert np.isclose(got["same"] + got["opposite"] + got["tie"], 1.0, rtol=RTOL, atol=ATOL)


def test_b2_continuous_scores_have_no_ties():
    """B2: "Tie-aware agreement agrees with enumeration; an all-tie tensor has
    C = 0.5, a tie fraction of one, and undefined strict agreement." With
    continuous scores the tie fraction is exactly zero and C equals the strict
    same-sign fraction."""
    rng = np.random.default_rng(11)
    G = rng.normal(50, 5, size=(6, 8))
    got = pair_agreement(G)
    want = _enumerated_agreement(G)
    assert got["tie"] == 0.0
    assert np.isclose(got["C"], want["C"], rtol=RTOL, atol=ATOL)
    assert np.isclose(got["C"], got["same"], rtol=RTOL, atol=ATOL)
    assert np.isclose(got["conditional_agreement"], got["same"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("G", [
    np.full((4, 6), 50.0),
    np.arange(4, dtype=float)[:, None] * np.ones((4, 6)),   # constant within each draw
])
def test_b2_all_tie_tensor(G):
    """B2: "Tie-aware agreement agrees with enumeration; an all-tie tensor has
    C = 0.5, a tie fraction of one, and undefined strict agreement." """
    out = pair_agreement(G)
    assert out["C"] == 0.5
    assert out["tie"] == 1.0
    assert out["same"] == 0.0
    assert out["opposite"] == 0.0
    assert math.isnan(out["conditional_agreement"])


# ---------------------------------------------------------------------------
# B3. Fixed-task decomposition
# ---------------------------------------------------------------------------

M3, T3, K3 = 8, 3, 3


def _exact_construction(seed=5150):
    """Deterministic construction in which all cross-draw U-statistics are exact.

    Orthonormal basis Q of the subspace orthogonal to the ones vector. Columns
    0..K-1 are the draw-specific noise directions, the remaining columns carry
    the task signals. Noise in different draws is orthogonal to each other and
    to every signal.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(M3, M3 - 1))
    raw -= raw.mean(axis=0, keepdims=True)          # each column sums to zero
    Q, _ = np.linalg.qr(raw)                        # (M, M-1), orthonormal
    ones = np.ones(M3)
    assert np.allclose(Q.T @ Q, np.eye(M3 - 1), atol=1e-12)
    assert np.allclose(Q.T @ ones, 0.0, atol=1e-12)
    noise_dirs = Q[:, :K3]                          # u_k
    signal_basis = Q[:, K3:]                        # M x (M-1-K) = 8 x 4
    S = signal_basis @ (8.0 * rng.normal(size=(M3 - 1 - K3, T3)))   # column t = s_t
    a = rng.uniform(1.0, 4.0, size=(T3, K3)) * rng.choice([-1.0, 1.0], size=(T3, K3))
    c = rng.uniform(30.0, 70.0, size=T3)
    b = rng.normal(0.0, 5.0, size=(T3, K3))
    G = np.empty((T3, K3, M3))
    for t in range(T3):
        for k in range(K3):
            G[t, k] = c[t] + S[:, t] + a[t, k] * noise_dirs[:, k] + b[t, k]
    return G, S, a


@pytest.mark.parametrize("weights", [[0.5, 0.3, 0.2], [5, 3, 2]])
def test_b3_decomposition_matches_exact_construction(weights):
    """B3: "The fixed-task decomposition agrees exactly with a deterministic
    unequal-weight construction." Weights are also passed unnormalized."""
    G, S, a = _exact_construction()
    w = np.array([0.5, 0.3, 0.2])
    s2 = (S ** 2).sum(axis=0)                       # |s_t|^2
    m = S @ w                                       # sum_t w_t s_t

    true_within = float(np.sum(w * s2) / (M3 - 1))
    profile = float(m @ m / (M3 - 1))
    model_by_task = true_within - profile
    total = float(sum(w[t] * np.mean((s2[t] + a[t] ** 2) / (M3 - 1)) for t in range(T3)))
    prompt_noise = total - true_within

    out = fixed_task_components(G, weights)
    assert np.isclose(out["true_within_task"], true_within, rtol=1e-10, atol=1e-10)
    assert np.isclose(out["average_task_profile"], profile, rtol=1e-10, atol=1e-10)
    assert np.isclose(out["model_by_task"], model_by_task, rtol=1e-10, atol=1e-10)
    assert np.isclose(out["total"], total, rtol=1e-10, atol=1e-10)
    assert np.isclose(out["prompt_noise"], prompt_noise, rtol=1e-10, atol=1e-10)

    # The model-by-task component is the weighted variance of task profiles around the mean profile.
    direct = float(sum(w[t] * np.var(S[:, t] - m, ddof=1) for t in range(T3)))
    assert np.isclose(out["model_by_task"], direct, rtol=1e-10, atol=1e-10)

    # The prompt-noise component is the weighted mean squared noise amplitude over (M-1).
    noise_direct = float(sum(w[t] * np.mean(a[t] ** 2) for t in range(T3)) / (M3 - 1))
    assert np.isclose(out["prompt_noise"], noise_direct, rtol=1e-10, atol=1e-10)

    # additive identity of the returned components
    assert np.isclose(out["average_task_profile"] + out["model_by_task"] + out["prompt_noise"],
                      out["total"], rtol=1e-10, atol=1e-10)


def test_b3_default_weights_are_uniform():
    """B3: "The fixed-task decomposition agrees exactly with a deterministic
    unequal-weight construction." weights=None equals explicit equal weights."""
    G, _, _ = _exact_construction()
    a = fixed_task_components(G)
    b = fixed_task_components(G, [1, 1, 1])
    for key in a:
        assert np.isclose(a[key], b[key], rtol=1e-12, atol=1e-12), key


@pytest.mark.parametrize("weights", [
    [0.5, 0.5],                 # wrong length
    [1.0, -0.5, 0.5],           # negative
    [0.0, 0.0, 0.0],            # zero sum
    [np.nan, 1.0, 1.0],         # non-finite
    [np.inf, 1.0, 1.0],         # non-finite
    [[1.0, 1.0, 1.0]],          # wrong rank
])
def test_b3_invalid_weights_raise(weights):
    """B3: "The fixed-task decomposition agrees exactly with a deterministic
    unequal-weight construction." Invalid task weights raise ValueError."""
    G, _, _ = _exact_construction()
    with pytest.raises(ValueError):
        fixed_task_components(G, weights)


@pytest.mark.parametrize("shape", [(1, 3, 8), (3, 1, 8), (3, 3, 1)])
def test_b3_invalid_tensor_shape_raises(shape):
    """B3: "The fixed-task decomposition agrees exactly with a deterministic
    unequal-weight construction." Fewer than two tasks, joint draws or models
    raises ValueError, as do non-finite entries and the wrong rank."""
    with pytest.raises(ValueError):
        fixed_task_components(np.ones(shape))
    G, _, _ = _exact_construction()
    bad = G.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fixed_task_components(bad)
    with pytest.raises(ValueError):
        fixed_task_components(G[0])


# ---------------------------------------------------------------------------
# B4. Prediction in singular cases
# ---------------------------------------------------------------------------

def test_b4_constant_scores_give_finite_constant_prediction():
    """B4: "Prediction remains finite in a constant-score singular case; PSD
    projection is confined to prediction." (a) B = P = constant."""
    B = P = np.full((5, 6), 40.0)
    pred, info = replicated_linear_predictor(B, P)
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()
    assert info["fallback"] == "constant"
    assert np.allclose(pred, 0.0)                   # gain is identically zero
    assert isinstance(info["psd_projection"], bool)


def test_b4_constant_gain_varying_baseline_is_finite():
    """B4: "Prediction remains finite in a constant-score singular case; PSD
    projection is confined to prediction." (b) Gains identical across models
    and draws, baselines varying."""
    rng = np.random.default_rng(77)
    theta = np.array([20.0, 35.0, 50.0, 60.0, 72.0, 85.0])          # persistent model effects
    B = theta + rng.integers(-3, 4, size=(5, 6)).astype(float)      # integer-valued, varying baselines
    P = B + 3.0                                     # exact in floating point
    assert (P - B == 3.0).all()
    q = unit_moments(B, P)
    assert q["V_delta"] == 0.0 and q["A"] == 0.0 and q["V_theta"] > 0
    pred, info = replicated_linear_predictor(B, P)
    assert np.isfinite(pred).all()
    assert info["fallback"] == "constant"
    assert np.allclose(pred, 3.0)


def test_b4_negative_raw_variance_is_reported_and_prediction_finite():
    """B4: "Prediction remains finite in a constant-score singular case; PSD
    projection is confined to prediction." (c) Raw V_delta < 0 is reported
    unclipped; the predictor stays finite, flags the PSD projection, and leaves
    inputs and raw inference moments unchanged."""
    rng = np.random.default_rng(88)
    M = 6
    v = rng.normal(0, 3, size=M)
    v -= v.mean()
    G = np.stack([v, -v])                           # K = 2, perfectly negatively correlated gains
    B = rng.normal(50, 10, size=(2, M))
    P = B + G

    B0, P0 = B.copy(), P.copy()
    before = unit_moments(B, P)
    expected_v_delta = -float(v @ v) / (M - 1)      # cross-draw moment of (v, -v)
    assert before["V_delta"] < 0
    assert np.isclose(before["V_delta"], expected_v_delta, rtol=1e-9)

    pred, info = replicated_linear_predictor(B, P)
    assert np.isfinite(pred).all() and pred.shape == (M,)
    assert info["psd_projection"] is True

    # prediction mutates neither inputs nor the raw moments used for inference
    assert np.array_equal(B, B0) and np.array_equal(P, P0)
    after = unit_moments(B, P)
    assert after == before                          # exact equality of raw values
    assert after["V_delta"] < 0


def test_b4_heterogeneous_case_uses_regression_weights():
    """B4: "Prediction remains finite in a constant-score singular case; PSD
    projection is confined to prediction." Sanity check of the non-singular
    path: finite weights, no fallback, no PSD projection."""
    rng = np.random.default_rng(99)
    K, M = 6, 12
    theta = rng.normal(50, 10, size=M)
    delta = 0.3 * (theta - theta.mean()) + rng.normal(0, 4, size=M)
    B = theta + rng.normal(0, 1, size=(K, M))
    G = delta + rng.normal(0, 1, size=(K, M))
    P = B + G
    pred, info = replicated_linear_predictor(B, P)
    assert info["fallback"] is None
    assert info["psd_projection"] is False
    assert np.isfinite(info["weights"]).all() and np.isfinite(pred).all()
    # the prediction is centered on the mean observed gain
    assert np.isclose(pred.mean(), (P - B).mean(), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# B5. Validation-risk estimators under independent noise
# ---------------------------------------------------------------------------

def _se_band(values, target, k=4.0):
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, se, abs(mean - target) <= k * se


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(20261003)
    M, Kv, R = 10, 6, 4000
    delta = rng.normal(5.0, 3.0, size=M)
    f = np.full(M, 5.0)                             # frozen constant prediction
    g = delta + rng.normal(0.0, 1.0, size=M)        # frozen informative prediction
    sigma = np.linspace(2.0, 6.0, M)                # heterogeneous noise sd
    Z = delta + rng.normal(size=(R, Kv, M)) * sigma
    return {"delta": delta, "f": f, "g": g, "sigma": sigma, "Z": Z, "R": R, "M": M, "Kv": Kv}


def test_b5_paired_loss_difference_recovers_target(sim):
    """B5: "Under independent simulated validation noise, paired loss
    difference and cross-product risk recover their stated targets within
    Monte Carlo tolerance." Paired loss difference."""
    d, f, g, Z = sim["delta"], sim["f"], sim["g"], sim["Z"]
    target = float(np.mean((d - f) ** 2 - (d - g) ** 2))
    vals = np.array([paired_loss_difference(Zr, f, g).mean() for Zr in Z])
    mean, se, ok = _se_band(vals, target)
    assert ok, f"mean {mean:.4f} vs target {target:.4f}, SE {se:.4f}"
    # one value per independent validation draw
    assert paired_loss_difference(Z[0], f, g).shape == (sim["Kv"],)
    assert target > 0       # g is closer to delta than f, so the difference favors g


def test_b5_cross_risk_recovers_target(sim):
    """B5: "Under independent simulated validation noise, paired loss
    difference and cross-product risk recover their stated targets within
    Monte Carlo tolerance." Cross-product risk for frozen f."""
    d, f, Z = sim["delta"], sim["f"], sim["Z"]
    target = float(np.mean((d - f) ** 2))
    vals = np.array([cross_risk(Zr, f) for Zr in Z])
    mean, se, ok = _se_band(vals, target)
    assert ok, f"mean {mean:.4f} vs target {target:.4f}, SE {se:.4f}"
    # the same holds for the informative predictor g
    g = sim["g"]
    target_g = float(np.mean((d - g) ** 2))
    vals_g = np.array([cross_risk(Zr, g) for Zr in Z])
    mean_g, se_g, ok_g = _se_band(vals_g, target_g)
    assert ok_g, f"mean {mean_g:.4f} vs target {target_g:.4f}, SE {se_g:.4f}"


def test_b5_naive_mse_is_biased_upward_by_noise_variance(sim):
    """B5: "Under independent simulated validation noise, paired loss
    difference and cross-product risk recover their stated targets within
    Monte Carlo tolerance." Naive mean squared error is biased upward by about
    mean(sigma_i^2), which is why cross_risk is needed."""
    d, f, Z, sigma = sim["delta"], sim["f"], sim["Z"], sim["sigma"]
    target = float(np.mean((d - f) ** 2))
    naive = ((Z - f) ** 2).mean(axis=(1, 2))
    mean, se, ok = _se_band(naive, target)
    assert not ok                                   # clearly outside the tolerance band
    bias = mean - target
    assert bias > 10 * se
    assert abs(bias - float(np.mean(sigma ** 2))) <= 4 * se


def test_b5_cross_risk_can_be_negative_at_the_truth(sim):
    """B5: "Under independent simulated validation noise, paired loss
    difference and cross-product risk recover their stated targets within
    Monte Carlo tolerance." With f = delta the target is 0, the replicate mean
    is near 0, and some replicates are negative (no silent truncation)."""
    d, Z = sim["delta"], sim["Z"]
    vals = np.array([cross_risk(Zr, d) for Zr in Z])
    mean, se, ok = _se_band(vals, 0.0)
    assert ok, f"mean {mean:.4f}, SE {se:.4f}"
    assert vals.min() < 0
    assert (vals < 0).sum() > 0
    assert (vals > 0).sum() > 0


def test_b5_prediction_vector_shape_mismatch_raises(sim):
    """B5: "Under independent simulated validation noise, paired loss
    difference and cross-product risk recover their stated targets within
    Monte Carlo tolerance." Prediction vectors must match the model panel."""
    Z0, f, g = sim["Z"][0], sim["f"], sim["g"]
    with pytest.raises(ValueError):
        paired_loss_difference(Z0, f[:-1], g)
    with pytest.raises(ValueError):
        paired_loss_difference(Z0, f, g[:-1])
    with pytest.raises(ValueError):
        cross_risk(Z0, f[:-1])
    with pytest.raises(ValueError):
        cross_risk(Z0[:1], f)                       # only one validation draw


# ---------------------------------------------------------------------------
# B6. Gaussian-reference prompt counts
# ---------------------------------------------------------------------------

def test_b6_prompt_counts_at_r_030():
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." """
    assert prompts_needed_gaussian(0.30, 0.80) == 10
    assert prompts_needed_gaussian(0.30, 0.90) == 46


@pytest.mark.parametrize("target", [0.6, 0.75, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("R", [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
def test_b6_counts_are_minimal_and_sufficient(R, target):
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." Consistency: the returned K reaches the target and K-1 does not."""
    K = prompts_needed_gaussian(R, target)
    assert isinstance(K, int) and K >= 1
    assert gaussian_agreement(R, K) >= target
    if K > 1:
        assert gaussian_agreement(R, K - 1) < target


def test_b6_consistency_at_the_record_values():
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." Agreement at the record counts, and one fewer prompt falls short."""
    for target, K in [(0.80, 10), (0.90, 46)]:
        assert gaussian_agreement(0.30, K) >= target
        assert gaussian_agreement(0.30, K - 1) < target


def test_b6_edge_cases():
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." R = 0 needs unboundedly many prompts, R = 1 needs one."""
    assert prompts_needed_gaussian(0.0, 0.9) == math.inf
    assert prompts_needed_gaussian(1.0, 0.9) == 1
    assert prompts_needed_gaussian(1.0, 0.99) == 1
    assert gaussian_agreement(0.0, 1) == 0.5
    assert gaussian_agreement(1.0, 1) == 1.0
    assert gaussian_agreement(0.0, 50) == 0.5
    assert np.isclose(gaussian_agreement(0.7, 1), 0.5 + math.asin(0.7) / math.pi)


@pytest.mark.parametrize("R,target", [
    (-0.1, 0.9), (1.1, 0.9), (0.3, 0.5), (0.3, 0.4), (0.3, 1.0), (0.3, 1.2),
])
def test_b6_invalid_prompt_count_inputs_raise(R, target):
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." Invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        prompts_needed_gaussian(R, target)


@pytest.mark.parametrize("R,K", [(-0.1, 1), (1.1, 1), (0.3, 0), (0.3, -2), (0.3, 1.5)])
def test_b6_invalid_agreement_inputs_raise(R, K):
    """B6: "Gaussian-reference prompt counts at R = 0.30 are 10 for 80% and 46
    for 90%." Invalid inputs to gaussian_agreement raise ValueError."""
    with pytest.raises(ValueError):
        gaussian_agreement(R, K)


# ---------------------------------------------------------------------------
# B7. Fixed-panel R = 0.70 counterexample (plan Appendix C.1)
# ---------------------------------------------------------------------------

N7 = 20
NOISE_VAR7 = 3.0 / 7.0


def _counterexample_delta():
    """19 models with gain 0 and one outlier a, normalized so Var_M(delta) = 1 (ddof=1)."""
    a = math.sqrt(N7)
    delta = np.zeros(N7)
    delta[0] = a
    assert np.isclose(np.var(delta, ddof=1), 1.0, rtol=1e-12)
    return delta


def test_b7_analytic_reference_and_exact_fixed_panel_agreement():
    """B7: "The fixed-panel R = 0.70 counterexample gives 0.550 pairwise
    agreement rather than the reference formula's 0.747." Analytic values."""
    delta = _counterexample_delta()
    assert np.isclose(1.0 / (1.0 + NOISE_VAR7), 0.70, rtol=1e-12)

    ref = gaussian_agreement(0.70, 1)
    assert abs(ref - 0.747) < 5e-4
    assert round(ref, 3) == 0.747

    i, j = np.triu_indices(N7, 1)
    assert len(i) == 190
    diffs = np.abs(delta[i] - delta[j])
    p = norm.cdf(diffs / math.sqrt(2.0 * NOISE_VAR7))
    exact = float(np.mean(p ** 2 + (1.0 - p) ** 2))
    assert abs(exact - 0.550) < 5e-4
    assert round(exact, 3) == 0.550
    assert ref - exact > 0.15


def test_b7_simulation_through_the_estimators():
    """B7: "The fixed-panel R = 0.70 counterexample gives 0.550 pairwise
    agreement rather than the reference formula's 0.747." Simulation through
    pair_agreement and reliability with K = 4000 independent draws."""
    delta = _counterexample_delta()
    rng = np.random.default_rng(70)
    K = 4000
    G = delta + rng.normal(0.0, math.sqrt(NOISE_VAR7), size=(K, N7))

    agree = pair_agreement(G)
    rel = reliability(G)
    assert abs(agree["C"] - 0.550) < 0.005
    assert agree["tie"] == 0.0
    assert abs(rel["R"] - 0.70) < 0.02
    gap = gaussian_agreement(0.70) - agree["C"]
    assert gap > 0.15
    # the gap persists when the Gaussian reference is evaluated at the estimated R
    assert gaussian_agreement(min(max(rel["R"], 0.0), 1.0)) - agree["C"] > 0.15
