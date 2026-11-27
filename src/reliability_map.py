"""Reliability map: ordering agreement C against reliability R.

Arrays are (draws, models) in percentage points: B the baseline arm, P the intervention arm, G = P - B. A setting is
one (benchmark or task, panel, intervention). For benefit (G) and accuracy (B) of each setting:

- R, the cross-draw panel moment over distinct draws divided by the mean within-draw panel variance
  (estimators.reliability; panel moments with denominator M - 1). On an enumerated DOVE bank R is the exact
  conditional V / (V + W) of dataset1.exact_targets, and the distinct-draw ratio is kept beside it as a check.
- C, the one-draw ordering agreement of model pairs between two draws, equal weights over model pairs, a tie counting
  one half, with the same, opposite, tie and decisive rates. Between distinct draws, or on
  an enumerated bank exactly under the uniform with-replacement sampler.
- The conditional Gaussian reference C_ref = 1/2 + arcsin(R)/pi and the fixed-panel
  pairwise references of pairwise_reference(), which locate a departure from C_ref.

Pilot test: R from the first n pilot draws predicts the one-draw agreement among validation draws by C_ref, the
agreement between two disjoint groups of K validation draws by C_ref at R_K = K R / (1 + (K - 1) R) (Spearman-Brown),
and the draws K a target agreement needs (estimators.prompts_needed_gaussian).

Ties. Scores are ratios of counts, and two equal count differences need not give bit-identical gains in floating
point (100 a / n - 100 b / n against 100 c / n - 100 d / n with a - b = c - d). estimators.pair_agreement
shares this module's rule (estimators.TIE_TOL). A difference of at most TIE_TOL pp is a tie:
the tolerance lies far above floating-point error (about 1e-13 pp) and far below the resolution of any score analysed
(the pooled MMLU block, whose subjects have 99 or 100 items, resolves about 1e-5 pp in a mean over 12 draws).

Edge cases are reported as they occur: an R outside [0, 1] has no reference value and gives no
prediction; it is never clipped. A zero within-draw variance leaves R undefined. R = 0 needs infinitely many draws.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .dataset1 import exact_targets
from .estimators import TIE_TOL, gaussian_agreement, prompts_needed_gaussian, reliability, tie_sign
from .sampling import draw_units, rng_for, sha256_json

QUANTITIES = ("benefit", "accuracy")


# --------------------------------------------------------------------------- reference curve


RESIDUE = 1e-12


def _unit(R):
    """R with floating-point residue at the ends of [0, 1] removed (identical draws give 1 + 2e-16); nothing else."""
    if R is not None and math.isfinite(R):
        if 1.0 < R <= 1.0 + RESIDUE:
            return 1.0
        if -RESIDUE <= R < 0.0:
            return 0.0
    return R


def r_defined(R) -> bool:
    """True when R can enter the Gaussian reference: finite and in [0, 1] up to floating-point residue."""
    R = _unit(R)
    return R is not None and math.isfinite(R) and 0.0 <= R <= 1.0


def spearman_brown(R, k) -> float:
    """Reliability of the mean of k independent draws, R_k = k R / (1 + (k - 1) R); NaN when R is not in [0, 1]."""
    if not r_defined(R):
        return float("nan")
    R = _unit(R)
    return k * R / (1 + (k - 1) * R)


def reference_c(R, k=1) -> float:
    """C_ref = 1/2 + arcsin(R_k)/pi for the mean of k draws; NaN (no reference value) when R is not in [0, 1]."""
    return gaussian_agreement(_unit(R), k) if r_defined(R) else float("nan")


def k_needed(R, target):
    """Draws per evaluation for the target agreement under the reference; None when R is not in [0, 1], inf at R = 0."""
    return prompts_needed_gaussian(_unit(R), target) if r_defined(R) else None


# --------------------------------------------------------------------------- agreement


def _signs(D, tol=TIE_TOL):
    return tie_sign(D, tol)


def _summary(same, opposite) -> dict:
    """C and its parts from per-pair same and opposite rates, equal weights over model pairs."""
    same, opposite = float(np.mean(same)), float(np.mean(opposite))
    tie = 1.0 - same - opposite
    tie = 0.0 if abs(tie) < 1e-12 else tie            # floating-point residue only
    dec = same + opposite
    return {"C": same + 0.5 * tie, "same": same, "opposite": opposite, "tie": tie, "decisive": dec,
            "conditional_agreement": same / dec if dec > 0 else float("nan")}


def agreement(X, exact=False, tol=TIE_TOL) -> dict:
    """One-draw ordering agreement over the rows of X (draws, models).

    exact=False: all ordered pairs of distinct draws, as estimators.pair_agreement.
    exact=True: two independent draws from the uniform with-replacement sampler over the rows of an enumerated bank,
    as dataset1.exact_agreement."""
    X = np.asarray(X, float)
    K, M = X.shape
    if M < 2 or K < (1 if exact else 2):
        raise ValueError("need two or more models and, between distinct draws, two or more draws")
    i, j = np.triu_indices(M, 1)
    s = _signs(X[:, i] - X[:, j], tol)
    pos, neg = (s > 0).sum(0).astype(float), (s < 0).sum(0).astype(float)
    if exact:
        same, opp = (pos ** 2 + neg ** 2) / K ** 2, 2 * pos * neg / K ** 2
    else:
        d = K * (K - 1)
        same, opp = (pos * (pos - 1) + neg * (neg - 1)) / d, 2 * pos * neg / d
    return _summary(same, opp)


def _row_parts(A, Bm, tol=TIE_TOL):
    """Per row r and model pair: same and opposite strict orderings of A[r] against Bm[r]."""
    A, Bm = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(Bm, float))
    i, j = np.triu_indices(A.shape[1], 1)
    s1, s2 = _signs(A[:, i] - A[:, j], tol), _signs(Bm[:, i] - Bm[:, j], tol)
    dec = (s1 != 0) & (s2 != 0)
    return (dec & (s1 == s2)).astype(float), (dec & (s1 != s2)).astype(float)


def agreement_between(A, Bm, tol=TIE_TOL) -> dict:
    """Agreement of model-pair orderings between paired evaluations (rows of A against the same rows of Bm), averaged
    over rows and model pairs. One row each is dataset1.agreement_between."""
    same, opp = _row_parts(A, Bm, tol)
    return _summary(same.mean(0), opp.mean(0))


def split_agreement(X, perms, k, tol=TIE_TOL) -> dict:
    """Mean agreement between two disjoint groups of k draws over the given splits: perms is (splits, draws) of
    permutations, the first k entries form one group and the next k the other."""
    X = np.asarray(X, float)
    a, b = X[perms[:, :k]].mean(1), X[perms[:, k:2 * k]].mean(1)
    return agreement_between(a, b, tol)


def bank_group_agreement(Xs: dict, ks, n_rep, rng, tol=TIE_TOL) -> dict:
    """Expected agreement between two independent k-draw evaluations from the uniform with-replacement sampler over
    the bank rows of each array in Xs (same draws for every array). k = 1 is exact; larger k by Monte Carlo, with the
    Monte Carlo standard error."""
    n_bank = next(iter(Xs.values())).shape[0]
    out = {q: {} for q in Xs}
    for k in ks:
        if k == 1:
            for q, X in Xs.items():
                out[q]["1"] = {"C": agreement(X, exact=True, tol=tol)["C"], "mc_se": 0.0, "n_rep": None}
            continue
        ia, ib = rng.integers(0, n_bank, (n_rep, k)), rng.integers(0, n_bank, (n_rep, k))
        for q, X in Xs.items():
            same, opp = _row_parts(X[ia].mean(1), X[ib].mean(1), tol)
            c = 0.5 * (1 + same.mean(1) - opp.mean(1))
            out[q][str(k)] = {"C": float(c.mean()), "mc_se": float(c.std(ddof=1) / math.sqrt(n_rep)), "n_rep": n_rep}
    return out


# --------------------------------------------------------------------------- fixed-panel references


def pairwise_reference(X, ddof=1, tol=TIE_TOL) -> dict:
    """Fixed-panel Gaussian references for the one-draw agreement.

    A model pair with mean contrast Delta and contrast SD sigma over the draws is ordered the same way by two
    independent draws with probability p^2 + (1 - p)^2, p = Phi(|Delta| / sigma), when its contrast is Gaussian.
    C_pairs averages this over the panel's pairs, each with its own sigma. C_common uses one sigma for every pair, the
    root mean square of the pairs' sigmas; it keeps R (mean Delta^2 is 2 V and mean sigma^2 is 2 W) and differs from
    C_ref only through the panel's realized differences. Averaged over Delta ~ N(0, 2 V) it equals C_ref exactly. The
    steps C_ref -> C_common -> C_pairs -> C therefore separate the realized true differences of this panel (their
    shape, which few models make lumpy), heterogeneous pair noise, and ties with non-Gaussian contrasts.

    Plug-in values. ddof=0 for an enumerated bank (exact moments); with sampled draws |Delta| carries its own noise,
    a small upward pull for pairs whose true difference is near zero."""
    X = np.asarray(X, float)
    i, j = np.triu_indices(X.shape[1], 1)
    D = X[:, i] - X[:, j]
    delta, var = D.mean(0), D.var(0, ddof=ddof)

    def c_of(v):
        sd = np.sqrt(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs(delta) / sd
        p = np.where(sd > tol, stats.norm.cdf(z), np.where(np.abs(delta) > tol, 1.0, 0.5))
        return float(np.mean(p ** 2 + (1 - p) ** 2))

    mv = float(var.mean())
    # where the ties fall: share of the one-draw ties on pairs with |Delta| < sigma / 2, where a Gaussian contrast
    # orders two draws the same way with probability at most 0.573 (a pair with sigma = 0 has no signal-to-noise)
    tie = (np.abs(D) <= tol).mean(0)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > tol, np.abs(delta) / sd, np.where(np.abs(delta) > tol, np.inf, 0.0))
    return {"C_pairs": c_of(var), "C_common": c_of(np.full_like(var, mv)), "n_pairs": int(len(var)),
            "mean_pair_noise": mv, "pair_noise_cv": float(var.std(ddof=0) / mv) if mv > 0 else float("nan"),
            "mean_sq_pair_difference": float(np.mean(delta ** 2)),
            "tie_share_low_signal": float(tie[z < 0.5].sum() / tie.sum()) if tie.sum() > 0 else float("nan")}


# --------------------------------------------------------------------------- one setting


def setting_point(B, P, exact=False) -> dict:
    """R, C, the reference and its departure for benefit and accuracy of one setting.

    exact=False: the rows are independent draws (distinct-draw moments and agreement).
    exact=True: the rows are an enumerated bank; R and C are the exact conditional values under the uniform
    with-replacement sampler, and the distinct-draw R is reported beside them."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    if B.shape != P.shape:
        raise ValueError("arms must have matching shapes")
    G = P - B
    out = {"n_draws": int(B.shape[0]), "M": int(B.shape[1]),
           "source": "enumerated bank, exact" if exact else "distinct draws"}
    tg = exact_targets(B, P) if exact else None
    for q, X in (("benefit", G), ("accuracy", B)):
        r = reliability(X)
        if exact:
            v, w = ((tg["V_delta"], tg["noise_W_gain"]) if q == "benefit" else (tg["V_theta"], tg["noise_W_baseline"]))
            blk = {"R": tg[f"R_{q}"], "signal": v, "noise": w, "within": v + w, "R_distinct_draws": r["R"]}
        else:
            blk = {"R": r["R"], "signal": r["signal"], "noise": r["total"] - r["signal"], "within": r["total"]}
        blk["R_status"] = r_status(blk["R"])
        ag = agreement(X, exact=exact)
        blk.update(ag)
        blk["C_ref"] = reference_c(blk["R"])
        blk["departure"] = blk["C"] - blk["C_ref"] if r_defined(blk["R"]) else float("nan")
        blk["pairwise"] = pairwise_reference(X, ddof=0 if exact else 1)
        out[q] = blk
    return out


def r_status(R) -> str:
    if R is None or not math.isfinite(R):
        return "undefined (zero within-draw variance)"
    R = _unit(R)
    if R < 0:
        return "negative estimate"
    if R > 1:
        return "above one"
    return "in [0, 1]"


# --------------------------------------------------------------------------- pilot test


def predict_from_r(R, ks, targets) -> dict:
    """The reference's predictions from one R: the one-draw agreement, the agreement between two disjoint groups of K
    draws and the draws K each target agreement needs. Nothing is predicted from an R outside [0, 1]."""
    return {"R": R, "R_status": r_status(R), "defined": r_defined(R), "C_one_draw": reference_c(R),
            "C_groups": {str(k): reference_c(R, k) for k in ks},
            "K_needed": {f"{t:.2f}": k_needed(R, t) for t in targets}}


def pilot_prediction(Bp, Pp, ks, targets) -> dict:
    """Predictions from the pilot draws alone (predict_from_r on the pilot R of benefit and of accuracy)."""
    Bp, Pp = np.asarray(Bp, float), np.asarray(Pp, float)
    out = {"n_pilot": int(Bp.shape[0])}
    for q, X in (("benefit", Pp - Bp), ("accuracy", Bp)):
        r = reliability(X)
        out[q] = {**predict_from_r(r["R"], ks, targets), "signal": r["signal"], "within": r["total"]}
    return out


def validation_agreement(Bv, Pv, ks, n_splits, rng) -> dict:
    """Observed agreement among the validation draws: one draw (all ordered pairs of distinct draws) and two disjoint
    groups of K draws (mean over random splits, the same splits for benefit and accuracy; the first K against the next
    K beside it). K = 1 over all pairs equals the one-draw value. K above half the draws is not supported."""
    Bv, Pv = np.asarray(Bv, float), np.asarray(Pv, float)
    kv = Bv.shape[0]
    Xs = {"benefit": Pv - Bv, "accuracy": Bv}
    out = {"K_validation": int(kv), "one_draw": {q: agreement(X) for q, X in Xs.items()},
           "groups": {q: {} for q in Xs}, "not_supported": [int(k) for k in ks if 2 * k > kv]}
    for k in ks:
        if 2 * k > kv:
            continue
        perms = None if k == 1 else np.stack([rng.permutation(kv) for _ in range(n_splits)])
        for q, X in Xs.items():
            mean = out["one_draw"][q] if k == 1 else split_agreement(X, perms, k)
            out["groups"][q][str(k)] = {"mean_over_splits": mean,
                                        "n_splits": kv * (kv - 1) if k == 1 else int(n_splits),
                                        "first_k_against_next_k": agreement_between(X[:k].mean(0),
                                                                                    X[k:2 * k].mean(0))}
    return out


def prediction_rows(setting: str, group: str, pred: dict, obs: dict, bank: dict | None = None) -> list[dict]:
    """One row per quantity and target (the one-draw agreement, then K = 1, 2, ... between groups): predicted,
    observed (mean over splits), the first split, and for an enumerated bank the bank's expected value."""
    rows = []
    for q in QUANTITIES:
        p = pred[q]
        targets = [("one_draw", 1, p["C_one_draw"], obs["one_draw"][q]["C"], None)]
        for k, g in obs["groups"][q].items():
            targets.append(("groups", int(k), p["C_groups"][k], g["mean_over_splits"]["C"],
                            g["first_k_against_next_k"]["C"]))
        for kind, k, c_pred, c_obs, c_first in targets:
            b = None if bank is None else bank[q][str(k)]["C"]
            ok = p["defined"]
            rows.append({"setting": setting, "group": group, "n_pilot": pred["n_pilot"], "quantity": q,
                         "target": kind, "K": k, "R_pilot": p["R"], "defined": ok,
                         "predicted": c_pred if ok else None, "observed": c_obs, "observed_first_split": c_first,
                         "error": c_pred - c_obs if ok else None, "bank": b,
                         "error_vs_bank": (c_pred - b) if ok and b is not None else None})
    return rows


def error_summary(errors) -> dict:
    """Spread of prediction errors (predicted minus observed) over settings with a defined prediction."""
    e = np.array([x for x in errors if x is not None and math.isfinite(x)], float)
    n_all = len(list(errors))
    if not e.size:
        return {"n": 0, "n_undefined": n_all}
    a = np.abs(e)
    return {"n": int(e.size), "n_undefined": int(n_all - e.size), "mean": float(e.mean()),
            "mae": float(a.mean()), "rmse": float(math.sqrt(np.mean(e ** 2))), "median_abs": float(np.median(a)),
            "q90_abs": float(np.quantile(a, 0.9)), "max_abs": float(a.max()),
            "sd": float(e.std(ddof=1)) if e.size > 1 else float("nan")}


def within_factor(k_pilot, k_ref, factor) -> bool | None:
    """Pilot K within the factor of the reference K: both infinite counts as within; None when either is undefined."""
    if k_pilot is None or k_ref is None:
        return None if k_ref is None else False
    if math.isinf(k_pilot) or math.isinf(k_ref):
        return math.isinf(k_pilot) and math.isinf(k_ref)
    return k_ref / factor <= k_pilot <= k_ref * factor


# --------------------------------------------------------------------------- pilot draws


def draw_pilot(bank_ids, n_pilot, n_validation, label, seed) -> dict:
    """Pilot and validation draws for an enumerated bank by the DOVE sampler's rule (uniform over the bank, with
    replacement, independent draws; src/sampling.py), the pilot draws first. Depends on the bank's ids, the label and
    the seed only."""
    ids = list(bank_ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("bank ids must be unique and nonempty")
    rng = rng_for("registered_split", label, seed=seed)
    p, v = draw_units(len(ids), n_pilot, rng), draw_units(len(ids), n_validation, rng)
    return {"label": label, "seed": int(seed), "n_bank": len(ids), "bank_sha256": sha256_json(ids),
            "pilot_index": p.tolist(), "validation_index": v.tolist(),
            "pilot_config_ids": [ids[k] for k in p], "validation_config_ids": [ids[k] for k in v]}
