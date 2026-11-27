"""Capability
measured on the external index, the share of V_delta a capability measure accounts for, the partial correlation of two
tasks' gains given the index, V_delta's interval over prompt draws, and a polynomial capability rule whose complexity
is chosen on discovery draws.

Layout as elsewhere: arrays are (draws, models) in percentage points, and the index is one value per model, the same in
every draw. Cross-draw moments are those of src/estimators.py, so with the index as capability A_x is the panel
covariance of the model mean benefits with the index and V_x is the index's panel variance (denominator M - 1).
Intervals are leave-one-draw-out jackknife t intervals (src/intervals.py), descriptive. Edge cases are reported as they
occur: a nonpositive V_delta leaves a share or a latent correlation undefined (None), never clipped.
"""
from __future__ import annotations

import numpy as np

from . import intervals as iv
from .estimators import cross_moment, cross_risk

# The candidate rules, simplest first: (name, degree in baseline, index term).
CANDIDATES = (("constant", 0, False), ("baseline_linear", 1, False), ("index_linear", 0, True),
              ("baseline_index_linear", 1, True), ("baseline_quadratic", 2, False),
              ("baseline_quadratic_index", 2, True))
FULL_RULE = "baseline_quadratic_index"


def tile(e, K):
    """The index as a (K, M) array, the same in every draw."""
    return np.tile(np.asarray(e, float), (int(K), 1))


def _jack(est, loo):
    if est is None or loo is None or any(x is None for x in loo):
        return {"estimate": est, "jackknife_se": None, "ci95": None}
    loo = np.asarray(loo, float)
    lo, hi = iv.jack_t(est, loo)
    return {"estimate": float(est), "jackknife_se": iv.jackknife_se(loo), "ci95": [float(lo), float(hi)]}


def vdelta_draws(G) -> dict:
    """V_delta = cross_moment(G, G) with its jackknife t interval over draws: the fixed panel's uncertainty from prompt
    draws alone."""
    cm = iv.CrossMoment(np.asarray(G, float), np.asarray(G, float))
    out = _jack(cm.estimate(), cm.loo())
    out["negative_estimate"] = bool(out["estimate"] < 0)
    return out


def _share(G, X):
    vx, vd, a = cross_moment(X, X), cross_moment(G, G), cross_moment(G, X)
    return None if vx <= 0 or vd <= 0 else a * a / (vx * vd)


def explained_share(G, X) -> dict:
    """A_x^2 / (V_x V_delta): the share of the true variance of benefit that a linear function of capability X
    accounts for (the squared latent correlation of benefit with X). Undefined when V_x or V_delta is not positive."""
    G, X = np.asarray(G, float), np.asarray(X, float)
    K = G.shape[0]
    est = _share(G, X)
    loo = [_share(np.delete(G, u, 0), np.delete(X, u, 0)) for u in range(K)] if K > 2 else None
    out = _jack(est, loo)
    out.update({"A_x": cross_moment(G, X), "V_x": cross_moment(X, X), "V_delta": cross_moment(G, G)})
    return out


def residual_projector(e):
    """I - H for least squares on an intercept and the index: rows of G @ P are the residuals of each draw's gains."""
    Z = np.column_stack([np.ones(len(e)), np.asarray(e, float)])
    return np.eye(len(e)) - Z @ np.linalg.pinv(Z)


def _partial_point(Ga, Gb, P):
    ra, rb = Ga.mean(0) @ P, Gb.mean(0) @ P
    Ra, Rb = Ga @ P, Gb @ P
    va, vb, c = cross_moment(Ra, Ra), cross_moment(Rb, Rb), cross_moment(Ra, Rb)
    sa, sb = float(np.std(ra, ddof=1)), float(np.std(rb, ddof=1))
    return {"pearson": float(np.corrcoef(ra, rb)[0, 1]) if sa > 0 and sb > 0 else None,
            "latent": c / float(np.sqrt(va * vb)) if va > 0 and vb > 0 else None}


def partial_correlation(Ga, Gb, e) -> dict:
    """Correlation of two tasks' gains after removing a linear function of the index from each, on joint bundles
    (bundle r pairs the r-th draw of each task): Pearson of the residual mean gains and the latent version, the
    cross-draw moment of the per-draw residuals over the root of their own moments. Leave-one-bundle-out jackknife t,
    as src/transfer_d2.cross_task_correlation."""
    Ga, Gb = np.asarray(Ga, float), np.asarray(Gb, float)
    if Ga.shape != Gb.shape:
        raise ValueError("both tasks need the same bundles and models")
    P = residual_projector(e)
    K = Ga.shape[0]
    est = _partial_point(Ga, Gb, P)
    loo = [_partial_point(np.delete(Ga, u, 0), np.delete(Gb, u, 0), P) for u in range(K)] if K > 2 else None
    return {k: _jack(v, None if loo is None else [r[k] for r in loo]) for k, v in est.items()}


def correlation_with_index(G, e) -> dict:
    """Pearson correlation of the model mean gains with the index, with the leave-one-draw-out jackknife t."""
    G, e = np.asarray(G, float), np.asarray(e, float)

    def r(X):
        g = X.mean(0)
        return float(np.corrcoef(g, e)[0, 1]) if np.std(g) > 0 and np.std(e) > 0 else None
    K = G.shape[0]
    return _jack(r(G), [r(np.delete(G, u, 0)) for u in range(K)] if K > 2 else None)


# --------------------------------------------------------------------------- capability rules


def design(b, e, degree, index):
    """Columns: intercept, baseline powers up to `degree` (baseline centred on its panel mean), and the index."""
    b = np.asarray(b, float) - np.mean(b)
    cols = [np.ones(len(b))] + [b ** p for p in range(1, degree + 1)]
    if index:
        cols.append(np.asarray(e, float) - np.mean(e))
    return np.column_stack(cols)


def fit_rule(Bd, Pd, e, degree, index) -> dict:
    """Least squares of the model mean gains on the model mean baselines and the index; the fitted values are the
    rule's predictions for the same models."""
    b, g = np.asarray(Bd, float).mean(0), (np.asarray(Pd, float) - np.asarray(Bd, float)).mean(0)
    X = design(b, e, degree, index)
    coef, *_ = np.linalg.lstsq(X, g, rcond=None)
    return {"prediction": X @ coef, "coefficients": coef.tolist(), "rank": int(np.linalg.matrix_rank(X)),
            "n_parameters": int(X.shape[1])}


def split_half_risk(Bd, Pd, e, degree, index) -> dict:
    """Complexity score: fit on the first half of the discovery draws, cross-product risk on the second half,
    and the reverse; the score is the average of the two risks."""
    K = Bd.shape[0]
    h = K // 2
    halves = (slice(0, h), slice(h, 2 * h))
    risks = []
    for fit, test in (halves, halves[::-1]):
        f = fit_rule(Bd[fit], Pd[fit], e, degree, index)["prediction"]
        risks.append(cross_risk(Pd[test] - Bd[test], f))
    return {"risks": risks, "score": float(np.mean(risks)), "halves": [h, h]}


def choose_rule(Bd, Pd, e) -> dict:
    """Every candidate's split-half score and the chosen rule: the lowest score, a tie going to the simpler rule."""
    scores = {n: split_half_risk(Bd, Pd, e, d, i) for n, d, i in CANDIDATES}
    best = min(scores, key=lambda n: (scores[n]["score"], [c[0] for c in CANDIDATES].index(n)))
    return {"scores": scores, "chosen": best}


def paired_difference(Zv, f, g) -> dict:
    """Step 3's paired loss difference over validation draws (positive favours g), with the jackknife t interval."""
    from .estimators import paired_loss_difference
    L = paired_loss_difference(np.asarray(Zv, float), np.asarray(f, float), np.asarray(g, float))
    kv = len(L)
    return _jack(float(L.mean()), (L.sum() - L) / (kv - 1) if kv > 2 else None)
