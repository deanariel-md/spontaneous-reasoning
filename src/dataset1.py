"""Dataset 1 analysis on an enumerated configuration bank .

Arrays are (bank configurations, models) in percentage points. B is the baseline arm
(zero-shot), P the intervention arm (five-shot), G = P - B. When the bank is fully
enumerated the conditional targets under the sampler (uniform, with
replacement) are computed exactly; quantities describing what a K-unit study would see are
computed by Monte Carlo over the bank and reported with their Monte Carlo error.

Edge cases are reported as they occur: negative variance estimates stay negative,
nonpositive denominators give NaN slopes and are counted, ties are counted.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .estimators import (cross_moment, cross_risk, pair_agreement, paired_loss_difference,
                         reliability, replicated_linear_predictor, tie_sign, unit_moments)

PRACTICAL = {"pairwise_pp": 5.0, "H_pp_per_10pp": 1.0, "gain_sd_pp": 2.0, "V_delta_pp2": 4.0,
             "risk_pp2": 1.0, "agreement_target": 0.80}


def pvar(x) -> float:
    return float(np.var(np.asarray(x, float), ddof=1))


def pcov(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def slope(g, b) -> float:
    """OLS slope of g on b across the panel; NaN when the panel variance of b is not positive."""
    v = pvar(b)
    return pcov(g, b) / v if v > 0 else float("nan")


# --------------------------------------------------------------------------- Step 1


def _pair_signs(X):
    M = X.shape[1]
    i, j = np.triu_indices(M, 1)
    return tie_sign(X[:, i] - X[:, j]), i, j


def exact_agreement(X) -> dict:
    """Ordering agreement between two independent draws from the uniform with-replacement
    sampler over the bank rows of X. Equal weights over model pairs."""
    X = np.asarray(X, float)
    C = X.shape[0]
    s, i, j = _pair_signs(X)
    pos, neg = (s > 0).sum(0), (s < 0).sum(0)
    same = (pos ** 2 + neg ** 2) / C ** 2
    opposite = 2 * pos * neg / C ** 2
    tie = 1 - same - opposite
    out = _summarize(same, opposite, tie)
    out["per_pair"] = [{"i": int(a), "j": int(b), "same": float(x), "opposite": float(y), "tie": float(z)}
                       for a, b, x, y, z in zip(i, j, same, opposite, tie)]
    return out


def _summarize(same, opposite, tie) -> dict:
    same, opposite, tie = float(np.mean(same)), float(np.mean(opposite)), float(np.mean(tie))
    tie = 0.0 if abs(tie) < 1e-12 else tie
    decisive = same + opposite
    return {"same": same, "opposite": opposite, "tie": tie, "decisive": decisive,
            "C": same + 0.5 * tie,
            "conditional_agreement": same / decisive if decisive > 0 else float("nan")}


def agreement_between(Xa, Xb) -> dict:
    """Agreement of model-pair orderings between two evaluations (vectors over models)."""
    s1, _, _ = _pair_signs(np.asarray(Xa, float)[None, :])
    s2, _, _ = _pair_signs(np.asarray(Xb, float)[None, :])
    s1, s2 = s1[0], s2[0]
    same = ((s1 == s2) & (s1 != 0)).astype(float)
    opposite = ((s1 == -s2) & (s1 != 0)).astype(float)
    tie = ((s1 == 0) | (s2 == 0)).astype(float)
    return _summarize(same, opposite, tie)


def pairwise_differences(X, names, margin=PRACTICAL["pairwise_pp"]) -> list[dict]:
    """Per model pair: distribution over bank configurations of the one-draw difference and the
    exact expected difference, against the practical margin."""
    X = np.asarray(X, float)
    s, i, j = _pair_signs(X)
    out = []
    for a, b in zip(i, j):
        d = X[:, a] - X[:, b]
        q = np.quantile(d, [0.025, 0.25, 0.5, 0.75, 0.975])
        out.append({"pair": f"{names[a]} - {names[b]}", "expected_difference": float(d.mean()),
                    "sd_one_draw": float(d.std(ddof=0)),
                    "q025": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q975": q[4],
                    "frac_abs_gt_margin": float(np.mean(np.abs(d) > margin)),
                    "expected_beyond_margin": bool(abs(d.mean()) > margin)})
    return out


def mc_agreement_curve(B, P, ks, n_rep, rng) -> list[dict]:
    """Expected agreement between two independent K-unit evaluations (each the mean over K
    units drawn from the bank), for benefit and baseline accuracy. Monte Carlo over the bank."""
    G = P - B
    C = B.shape[0]
    rows = []
    for k in ks:
        res = {"G": [], "B": []}
        for _ in range(n_rep):
            a, b = rng.integers(0, C, k), rng.integers(0, C, k)
            res["G"].append(agreement_between(G[a].mean(0), G[b].mean(0)))
            res["B"].append(agreement_between(B[a].mean(0), B[b].mean(0)))
        for lab, lst in res.items():
            Cs = np.array([r["C"] for r in lst])
            rows.append({"K": k, "quantity": "benefit" if lab == "G" else "accuracy",
                         "C": float(Cs.mean()), "C_mc_se": float(Cs.std(ddof=1) / math.sqrt(n_rep)),
                         "same": float(np.mean([r["same"] for r in lst])),
                         "opposite": float(np.mean([r["opposite"] for r in lst])),
                         "tie": float(np.mean([r["tie"] for r in lst]))})
    return rows


# --------------------------------------------------------------------------- Step 2


def exact_targets(B, P) -> dict:
    """Exact conditional targets over the enumerated bank (uniform sampler)."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    G = P - B
    theta, delta, pi = B.mean(0), G.mean(0), P.mean(0)
    eB, eG, eP = B - theta, G - delta, P - pi
    V_theta, A, V_delta = pvar(theta), pcov(delta, theta), pvar(delta)
    beta = A / V_theta if V_theta > 0 else float("nan")
    W_G = float(np.mean(np.var(eG, axis=1, ddof=1)))
    W_B = float(np.mean(np.var(eB, axis=1, ddof=1)))
    cov_eP_eB = float(np.mean([pcov(x, y) for x, y in zip(eP, eB)]))
    var_eB = W_B
    naive = np.array([slope(g, b) for g, b in zip(G, B)])
    return {
        "theta": theta.tolist(), "delta": delta.tolist(), "pi": pi.tolist(),
        "mean_gain": float(delta.mean()), "mean_baseline": float(theta.mean()),
        "V_theta": V_theta, "A": A, "V_delta": V_delta,
        "gain_sd": math.sqrt(V_delta) if V_delta >= 0 else float("nan"),
        "beta": beta, "H": -10 * beta,
        "noise_W_gain": W_G, "noise_W_baseline": W_B,
        "R_benefit": V_delta / (V_delta + W_G) if V_delta + W_G > 0 else float("nan"),
        "R_accuracy": V_theta / (V_theta + W_B) if V_theta + W_B > 0 else float("nan"),
        "coupling": {"E_cov_G_B_one_draw": float(np.mean([pcov(g, b) for g, b in zip(G, B)])),
                     "A": A, "E_cov_eP_eB": cov_eP_eB, "E_var_eB": var_eB,
                     "identity_rhs": A + cov_eP_eB - var_eB},
        "naive_single_unit": {"E_beta": float(np.mean(naive[~np.isnan(naive)])) if (~np.isnan(naive)).any() else float("nan"),
                              "E_H": float(-10 * np.mean(naive[~np.isnan(naive)])) if (~np.isnan(naive)).any() else float("nan"),
                              "n_nonpositive_denominator": int(np.isnan(naive).sum()),
                              "P_H_positive": float(np.mean(-10 * naive[~np.isnan(naive)] > 0))},
    }


def classify_point(value, margin) -> str:
    """Point classification of an exact conditional value against a margin (no interval)."""
    if value is None or not np.isfinite(value):
        return "undefined"
    return "outside margin" if abs(value) > margin else "within margin"


def capability_estimates(Bk, Pk) -> dict:
    """Three capability-claim estimators from one evaluation block of K units."""
    Gk = Pk - Bk
    out = {"naive_single_unit": -10 * slope(Gk[0], Bk[0]),
           "averaged_ols": -10 * slope(Gk.mean(0), Bk.mean(0))}
    if Bk.shape[0] >= 2:
        q = unit_moments(Bk, Pk)
        out["replicated"] = -10 * q["A"] / q["V_theta"] if q["V_theta"] > 0 else float("nan")
        out["_moments"] = q
    else:
        out["replicated"] = float("nan")
    return out


def mc_capability_blocks(B, P, k, n_rep, rng, target_H) -> dict:
    """Independent evaluation blocks of K units: distribution of H for each estimator, sign
    agreement between two independent blocks, and sign agreement with the exact target."""
    C = B.shape[0]
    H = {m: np.empty((n_rep, 2)) for m in ("naive_single_unit", "averaged_ols", "replicated")}
    for r in range(n_rep):
        for side in range(2):
            idx = rng.integers(0, C, k)
            est = capability_estimates(B[idx], P[idx])
            for m in H:
                H[m][r, side] = est[m]
    out = {}
    for m, h in H.items():
        ok = np.isfinite(h).all(axis=1)
        s1, s2 = np.sign(h[ok, 0]), np.sign(h[ok, 1])
        flat = h[np.isfinite(h)]
        out[m] = {"K": k if m != "naive_single_unit" else 1,
                  "mean_H": float(flat.mean()) if flat.size else float("nan"),
                  "sd_H": float(flat.std(ddof=1)) if flat.size > 1 else float("nan"),
                  "q025": float(np.quantile(flat, 0.025)) if flat.size else float("nan"),
                  "q975": float(np.quantile(flat, 0.975)) if flat.size else float("nan"),
                  "undefined_fraction": float(1 - np.isfinite(h).mean()),
                  "sign_agreement_between_blocks": float(np.mean(s1 == s2)) if ok.any() else float("nan"),
                  "sign_matches_target": float(np.mean(np.sign(flat) == np.sign(target_H))) if flat.size else float("nan"),
                  "P_abs_H_gt_margin": float(np.mean(np.abs(flat) > PRACTICAL["H_pp_per_10pp"])) if flat.size else float("nan")}
    return out


def mc_moment_distribution(B, P, k, n_rep, rng) -> dict:
    """Sampling distribution of the replicated moment estimators from K discovery units."""
    C = B.shape[0]
    keys = ("V_theta", "A", "V_delta", "H", "R_benefit")
    vals = {x: np.empty(n_rep) for x in keys}
    for r in range(n_rep):
        idx = rng.integers(0, C, k)
        q = unit_moments(B[idx], P[idx])
        vals["V_theta"][r], vals["A"][r], vals["V_delta"][r] = q["V_theta"], q["A"], q["V_delta"]
        vals["H"][r] = -10 * q["A"] / q["V_theta"] if q["V_theta"] > 0 else np.nan
        vals["R_benefit"][r] = reliability(P[idx] - B[idx])["R"]
    out = {"K": k, "n_rep": n_rep}
    for x, v in vals.items():
        f = v[np.isfinite(v)]
        out[x] = {"mean": float(f.mean()), "mc_se": float(f.std(ddof=1) / math.sqrt(len(f))),
                  "sd": float(f.std(ddof=1)), "q025": float(np.quantile(f, 0.025)),
                  "q50": float(np.quantile(f, 0.5)), "q975": float(np.quantile(f, 0.975)),
                  "frac_negative": float(np.mean(f < 0)), "n_undefined": int((~np.isfinite(v)).sum())}
    out["V_delta"]["P_above_margin"] = float(np.mean(vals["V_delta"] > PRACTICAL["V_delta_pp2"]))
    hv = vals["H"][np.isfinite(vals["H"])]
    out["H"]["P_abs_above_margin"] = float(np.mean(np.abs(hv) > PRACTICAL["H_pp_per_10pp"]))
    return out


# --------------------------------------------------------------------------- Step 3

PREDICTORS = ("constant", "own_gain", "averaged_ols", "replicated", "single_prompt")
COMPARISONS = (("constant", "own_gain"), ("constant", "replicated"), ("own_gain", "replicated"),
               ("constant", "averaged_ols"), ("own_gain", "averaged_ols"),
               ("averaged_ols", "replicated"), ("constant", "single_prompt"))


def fit_predictors(Bd, Pd) -> dict:
    """Predictions of each model's expected gain from discovery units only."""
    Bd, Pd = np.asarray(Bd, float), np.asarray(Pd, float)
    Gd = Pd - Bd
    Bb, Gb = Bd.mean(0), Gd.mean(0)
    M = Bd.shape[1]
    out = {"constant": (np.full(M, Gb.mean()), {}),
           "own_gain": (Gb.copy(), {})}
    b = slope(Gb, Bb)
    out["averaged_ols"] = ((Gb.mean() + b * (Bb - Bb.mean())) if np.isfinite(b) else np.full(M, Gb.mean()),
                           {"slope": b, "fallback": None if np.isfinite(b) else "constant"})
    b1 = slope(Gd[0], Bd[0])
    out["single_prompt"] = ((Gd[0].mean() + b1 * (Bd[0] - Bd[0].mean())) if np.isfinite(b1)
                            else np.full(M, Gd[0].mean()),
                            {"slope": b1, "fallback": None if np.isfinite(b1) else "constant",
                             "units_used": 1})
    if Bd.shape[0] >= 2:
        pred, info = replicated_linear_predictor(Bd, Pd)
        info = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
        out["replicated"] = (pred, info)
    return out


def evaluate_predictions(Z, preds: dict, delta=None, alpha=0.05) -> dict:
    """Validation of predictions on independent validation gains Z (K_v x M).
    Positive paired differences favor the second predictor. Intervals are provisional
    t-intervals over whole validation draws, pending the calibration gate."""
    Z = np.asarray(Z, float)
    kv = Z.shape[0]
    tq = stats.t.ppf(1 - alpha / 2, kv - 1)
    comps = []
    for f, g in COMPARISONS:
        if f not in preds or g not in preds:
            continue
        d = paired_loss_difference(Z, preds[f], preds[g])
        m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(kv))
        row = {"f": f, "g": g, "estimate": m, "se": se,
               "provisional_ci": [m - tq * se, m + tq * se]}
        if delta is not None:
            row["exact_conditional"] = float(np.mean((delta - preds[f]) ** 2 - (delta - preds[g]) ** 2))
        comps.append(row)
    risks = {}
    for name, f in preds.items():
        risks[name] = {"cross_risk": cross_risk(Z, f),
                       "naive_mse": float(np.mean((Z - f) ** 2))}
        if delta is not None:
            risks[name]["exact_risk"] = float(np.mean((delta - f) ** 2))
    return {"comparisons": comps, "risks": risks}


def mc_prediction(B, P, k_disc, n_rep, rng, ks=None) -> dict:
    """Procedure-level expected risk differences (exact, given the bank) over sampler
    realizations, at each nested discovery budget."""
    G = P - B
    delta = G.mean(0)
    C = B.shape[0]
    ks = ks or [k_disc]
    res = {k: {c: [] for c in COMPARISONS} for k in ks}
    risk = {k: {p: [] for p in PREDICTORS} for k in ks}
    for _ in range(n_rep):
        idx = rng.integers(0, C, max(ks))
        for k in ks:
            pr = {n: v[0] for n, v in fit_predictors(B[idx[:k]], P[idx[:k]]).items()}
            for n, f in pr.items():
                risk[k][n].append(float(np.mean((delta - f) ** 2)))
            for f, g in COMPARISONS:
                if f in pr and g in pr:
                    res[k][(f, g)].append(float(np.mean((delta - pr[f]) ** 2 - (delta - pr[g]) ** 2)))
    out = []
    for k in ks:
        for (f, g), v in res[k].items():
            if not v:
                continue
            v = np.array(v)
            out.append({"K": k, "f": f, "g": g, "mean": float(v.mean()),
                        "mc_se": float(v.std(ddof=1) / math.sqrt(len(v))),
                        "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975)),
                        "P_g_better_by_margin": float(np.mean(v > PRACTICAL["risk_pp2"])),
                        "P_f_better_by_margin": float(np.mean(v < -PRACTICAL["risk_pp2"]))})
    risks = [{"K": k, "predictor": p, "mean_exact_risk": float(np.mean(v)),
              "mc_se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}
             for k in ks for p, v in risk[k].items() if v]
    return {"comparisons": out, "risks": risks}
