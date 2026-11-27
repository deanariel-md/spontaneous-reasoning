"""Known-effect controls on empirical scored outcomes .

Null controls: pseudo-baseline and pseudo-intervention units are independent draws from the
SAME arm's configuration bank (both the baseline and the intervention populations), so every
model's expected gain is exactly zero.

Injected controls: the pseudo-intervention unit is an independent baseline draw whose scored
outcomes are flipped: failures flip to successes with probability d_i/(1 - p_i) when d_i > 0,
successes to failures with probability -d_i/p_i when d_i < 0, where p_i is model i's
bank-mean baseline accuracy (a probability) and d_i the target change converted from
percentage points to probability. Averaged over the uniform sampler this gives an expected
gain of exactly d_i, so the post-injection truth is delta = d, theta unchanged; the noise
structure is whatever the flips produce and is not assumed.

Counts are per subject: nc has shape (subjects, bank configurations, models) and n_items
shape (subjects,). Pooled accuracy is the equal-weight mean of subject accuracies, in pp.
"""
from __future__ import annotations

import math

import numpy as np

from .dataset1 import PRACTICAL, capability_estimates, fit_predictors, pcov, pvar
from .estimators import pair_agreement, unit_moments


def pooled(nc, n_items, weights=None) -> np.ndarray:
    S = nc.shape[0]
    w = np.ones(S) / S if weights is None else np.asarray(weights, float) / np.sum(weights)
    return 100 * np.einsum("s,sc...->c...", w, nc / n_items.reshape((S,) + (1,) * (nc.ndim - 1)))


def scenario_effects(theta, scenario: dict) -> np.ndarray:
    """Target gains d_i in pp for a scenario."""
    theta = np.asarray(theta, float)
    kind = scenario["kind"]
    if kind == "null":
        return np.zeros_like(theta)
    if kind == "constant":
        return np.full_like(theta, scenario["gain"])
    if kind == "capability":
        return scenario["gain"] + scenario["beta"] * (theta - theta.mean())
    if kind == "residual":
        u = np.asarray(scenario["u"], float)
        return scenario["gain"] + u
    if kind == "sparse_outlier":
        d = np.full_like(theta, scenario["gain"])
        d[scenario["index"]] += scenario["extra"]
        return d
    raise ValueError(kind)


def residual_direction(theta, sd, rng) -> np.ndarray:
    """A fixed residual-heterogeneity vector orthogonal to capability, with panel SD = sd."""
    theta = np.asarray(theta, float)
    M = len(theta)
    u = rng.normal(size=M)
    tc = theta - theta.mean()
    u = u - u.mean()
    u = u - (u @ tc) / (tc @ tc) * tc
    return u * sd / math.sqrt(np.sum(u ** 2) / (M - 1))


def draw_control_units(nc0, nc1, n_items, k, scenario, theta, rng):
    """K pseudo-paired units (B, P), each (K, M) in pp, and the exact truth delta."""
    S, C, M = nc0.shape
    a, b = rng.integers(0, C, k), rng.integers(0, C, k)
    if scenario["kind"] == "null":
        src = nc0 if scenario.get("population", "baseline") == "baseline" else nc1
        return pooled(src[:, a, :], n_items), pooled(src[:, b, :], n_items), np.zeros(M)
    d = scenario_effects(theta, scenario)
    p = np.asarray(theta) / 100
    dp = d / 100
    if np.any(dp > 1 - p) or np.any(-dp > p):
        raise ValueError("Infeasible injected effect for some model.")
    base = nc0[:, b, :]
    n = np.broadcast_to(n_items.reshape(S, 1, 1), base.shape)
    q_up = np.where(dp > 0, dp / (1 - p), 0.0)
    q_dn = np.where(dp < 0, -dp / p, 0.0)
    up = rng.binomial((n - base).astype(np.int64), np.broadcast_to(q_up, base.shape))
    dn = rng.binomial(base.astype(np.int64), np.broadcast_to(q_dn, base.shape))
    return pooled(nc0[:, a, :], n_items), pooled(base + up - dn, n_items), d


def run_control(nc0, nc1, n_items, theta, scenario, k, n_rep, rng) -> dict:
    """Estimator behaviour against the known truth for one scenario."""
    theta = np.asarray(theta, float)
    d = scenario_effects(theta, scenario) if scenario["kind"] != "null" else np.zeros(len(theta))
    # Score against the population actually sampled: an intervention-population null draws
    # both pseudo-arms from the five-shot bank, so its theta is that bank's mean accuracy.
    theta_true = (pooled(nc1, n_items).mean(0) if scenario["kind"] == "null"
                  and scenario.get("population") == "intervention" else theta)
    truth = {"V_delta": pvar(d), "A": pcov(d, theta_true), "V_theta": pvar(theta_true)}
    truth["H"] = -10 * truth["A"] / truth["V_theta"]
    rec = {x: np.empty(n_rep) for x in ("V_delta", "A", "V_theta", "H_replicated", "H_naive",
                                        "H_averaged_ols", "C_benefit", "risk_constant",
                                        "risk_own_gain", "risk_averaged_ols", "risk_replicated")}
    for r in range(n_rep):
        B, P, delta = draw_control_units(nc0, nc1, n_items, k, scenario, theta, rng)
        q = unit_moments(B, P)
        rec["V_delta"][r], rec["A"][r], rec["V_theta"][r] = q["V_delta"], q["A"], q["V_theta"]
        cap = capability_estimates(B, P)
        rec["H_replicated"][r] = cap["replicated"]
        rec["H_naive"][r] = cap["naive_single_unit"]
        rec["H_averaged_ols"][r] = cap["averaged_ols"]
        rec["C_benefit"][r] = pair_agreement(P - B)["C"]
        preds = fit_predictors(B, P)
        for name in ("constant", "own_gain", "averaged_ols", "replicated"):
            rec[f"risk_{name}"][r] = float(np.mean((delta - preds[name][0]) ** 2))
    out = {"scenario": scenario.get("name"), "K": k, "n_rep": n_rep, "truth": truth,
           "d": d.tolist()}
    for x, v in rec.items():
        f = v[np.isfinite(v)]
        out[x] = {"mean": float(f.mean()), "mc_se": float(f.std(ddof=1) / math.sqrt(len(f))),
                  "sd": float(f.std(ddof=1)), "q025": float(np.quantile(f, 0.025)),
                  "q975": float(np.quantile(f, 0.975)), "frac_negative": float(np.mean(f < 0)),
                  "n_undefined": int((~np.isfinite(v)).sum())}
    for x in ("V_delta", "A", "V_theta"):
        out[x]["bias"] = out[x]["mean"] - truth[x]
        out[x]["bias_z"] = out[x]["bias"] / out[x]["mc_se"] if out[x]["mc_se"] > 0 else float("nan")
    for x in ("H_replicated", "H_naive", "H_averaged_ols"):
        out[x]["bias"] = out[x]["mean"] - truth["H"]
        f = rec[x][np.isfinite(rec[x])]
        out[x]["P_abs_above_margin"] = float(np.mean(np.abs(f) > PRACTICAL["H_pp_per_10pp"]))
    out["V_delta"]["P_above_margin"] = float(np.mean(rec["V_delta"] > PRACTICAL["V_delta_pp2"]))
    return out
