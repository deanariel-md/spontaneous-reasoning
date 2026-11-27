"""How stable is the capability slope of benefit?

The capability slope regresses each model's benefit on its baseline across the panel. It is computed here on three
scales of benefit (percentage points, logits, relative error reduction), over subpanels of the panel, over single
prompt draws, and under the GSM8K scoring variants; own-gain orderings and own-gain reliability are
computed on the same data beside it. The three-scale part of analysis R (reproduction Part B) uses the same
functions (scripts/run_cheng_scales.py).

Layout: arrays are (draws, models). B and P are the baseline and intervention arms in percentage points; the scale
transformation is applied to each per-draw score before any moment is taken (config/instability.yaml):

    points            capability x = B,                benefit y = P - B
    logit             capability x = logit(B / 100),   benefit y = logit(P / 100) - logit(B / 100)
    error_reduction   capability x = B,                benefit y = 100 (P - B) / (100 - B)

Cross-draw moments use distinct draws (cross_moment of src/estimators, panel denominator M - 1). The slope is A / V_x with
A = cross_moment(y, x) and V_x = cross_moment(x, x); in points H = -10 * slope. Intervals come from the
draw-level jackknife of src/intervals.py.

Edge cases are reported as they occur: a nonpositive V_x leaves the slope undefined, a slope of exactly
zero is a tie, negative variance estimates stay negative, Fieller sets may be unbounded, and cells where a scale is
undefined (a score of 0% or 100%; a baseline of 100% for error reduction) are counted and replaced by the empirical
proportion (k + 1/2) / (n + 1), with the replacement flagged.
"""
from __future__ import annotations

import itertools
import math
import re

import numpy as np

from . import dataset1 as d1, intervals as iv
from .estimators import reliability, tie_sign
from .scoring import normalize_number, normalize_text

SCALES = ("points", "logit", "error_reduction")
SLOPE_UNITS = {"points": "pp of gain per pp of baseline",
               "logit": "logits of gain per logit of baseline",
               "error_reduction": "pp of error reduction per pp of baseline"}
SCALE_LABELS = {"points": "Points", "logit": "Logits", "error_reduction": "Error reduction"}


# --------------------------------------------------------------------------- scales


def _logit_pct(p):
    q = np.asarray(p, float) / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(q) - np.log1p(-q)


def empirical_pct(p, n_items):
    """The empirical proportion (k + 1/2) / (n + 1) in pp, with k = p n / 100 correct of n items."""
    n = float(n_items)
    return 100.0 * (np.asarray(p, float) * n / 100.0 + 0.5) / (n + 1.0)


def scale_scores(B, P, scale, n_items) -> dict:
    """Capability x and benefit y of every (draw, model) cell on one scale.

    A cell whose transformation is undefined (logit: either arm at 0% or 100%; error reduction: a baseline of 100%)
    has both of its arms replaced by the empirical proportion (k + 1/2) / (n + 1) before transforming; the cells are
    counted and returned as a mask."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    if B.shape != P.shape:
        raise ValueError("arms must have the same (draws, models) layout")
    if scale == "points":
        bad = np.zeros(B.shape, bool)
        return {"scale": scale, "x": B.copy(), "y": P - B, "replaced": bad, "n_replaced": 0, "n_cells": int(B.size)}
    if scale == "logit":
        bad = (B <= 0) | (B >= 100) | (P <= 0) | (P >= 100)
    elif scale == "error_reduction":
        bad = B >= 100
    else:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")
    Bq = np.where(bad, empirical_pct(B, n_items), B)
    Pq = np.where(bad, empirical_pct(P, n_items), P)
    if scale == "logit":
        x = _logit_pct(Bq)
        y = _logit_pct(Pq) - x
    else:
        x = Bq
        y = 100.0 * (Pq - Bq) / (100.0 - Bq)
    return {"scale": scale, "x": x, "y": y, "replaced": bad, "n_replaced": int(bad.sum()), "n_cells": int(B.size)}


# --------------------------------------------------------------------------- moments and intervals


def interval_sign(kind, lo, hi) -> str:
    """Where a 95% set for a slope lies relative to zero."""
    if kind == "bounded":
        if lo > 0:
            return "above zero"
        if hi < 0:
            return "below zero"
        return "includes zero"
    if kind in ("exclusive", "whole"):
        return "unbounded"
    return "undefined"


def _reliability_loo(Y):
    """Leave-one-draw-out R = signal / total."""
    Y = np.asarray(Y, float)
    K = Y.shape[0]
    signal = iv.CrossMoment(Y, Y).loo()
    per_draw = np.var(Y, axis=1, ddof=1)
    total = (per_draw.sum() - per_draw) / (K - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, signal / np.where(total > 0, total, 1.0), np.nan)


def scale_moments(X, Y, slope_method="fieller", alpha=iv.ALPHA) -> dict:
    """Cross-draw moments of capability X and benefit Y (draws, models), the slope with its jackknife set, and
    own-gain reliability R_benefit = cross_moment(Y, Y) / mean within-draw panel variance, with a jackknife t
    interval. slope_method: "fieller" (Fieller set from the jackknife covariance of A and V_x) or "jack_t" (t
    interval on the leave-one-draw-out slopes)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    K, M = X.shape
    cx, ca, cy = iv.CrossMoment(X, X), iv.CrossMoment(Y, X), iv.CrossMoment(Y, Y)
    Vx, A, Vy = cx.estimate(), ca.estimate(), cy.estimate()
    lx, la = cx.loo(), ca.loo()
    slope = A / Vx if Vx > 0 else float("nan")
    out = {"K": int(K), "M": int(M), "V_x": Vx, "A": A, "V_y": Vy, "V_y_negative": bool(Vy < 0),
           "V_x_nonpositive": bool(Vx <= 0), "slope": slope,
           "sign": None if not np.isfinite(slope) else ("tie" if slope == 0 else ("negative" if slope < 0 else "positive")),
           "slope_method": slope_method}
    if slope_method == "fieller":
        kind, a, b = iv.fieller(A, Vx, la, lx, alpha)
        out["slope_set"] = {"kind": kind, "a": float(a), "b": float(b)}
    elif slope_method == "jack_t":
        with np.errstate(divide="ignore", invalid="ignore"):
            loo = np.where(lx > 0, la / np.where(lx > 0, lx, 1.0), np.nan)
        lo, hi = iv.jack_t(slope, loo, alpha)
        kind = "bounded" if np.isfinite(lo) and np.isfinite(hi) else "undefined"
        out["slope_set"] = {"kind": kind, "a": float(lo), "b": float(hi)}
    else:
        raise ValueError(f"unknown slope method {slope_method!r}")
    s = out["slope_set"]
    out["slope_set_sign"] = interval_sign(s["kind"], s["a"], s["b"])
    rel = reliability(Y)
    R_loo = _reliability_loo(Y)
    lo, hi = iv.jack_t(rel["R"], R_loo, alpha)
    out["R_benefit"] = rel["R"]
    out["R_benefit_ci95"] = [float(lo), float(hi)]
    out["W_y"] = rel["total"] - rel["signal"]
    out["total_y"] = rel["total"]
    out["mean_benefit"] = float(Y.mean())
    out["mean_capability"] = float(X.mean())
    out["naive_model_mean_slope"] = d1.slope(Y.mean(0), X.mean(0))
    out["model_means"] = {"capability": X.mean(0).tolist(), "benefit": Y.mean(0).tolist()}
    return out


def family_weights(families) -> np.ndarray:
    """Equal-family weights: 1 / (F n_f) for a model in a family of n_f models among F families."""
    fam = list(families)
    n = {f: fam.count(f) for f in fam}
    return np.array([1.0 / (len(n) * n[f]) for f in fam])


def weighted_moments(X, Y, w, slope_method="fieller", alpha=iv.ALPHA) -> dict:
    """Weighted cross-draw moments of capability X and benefit Y (draws, models) and the weighted slope with its
    jackknife set. Weights are normalized to sum 1; each draw is centred at its weighted mean, products are
    weighted and the denominator is 1 - sum w^2. With equal weights the moments, slope and set equal scale_moments'."""
    X, Y, w = np.asarray(X, float), np.asarray(Y, float), np.asarray(w, float)
    if w.shape != (X.shape[1],) or np.any(w <= 0):
        raise ValueError("one positive weight per model is needed")
    w = w / w.sum()
    K, M = X.shape
    r = np.sqrt(w)[None, :]
    Xc, Yc = (X - (X @ w)[:, None]) * r, (Y - (Y @ w)[:, None]) * r
    dm = 1.0 - float(w @ w)
    cx, ca, cy = (iv.CrossMoment(a, b, center=False, denom_models=dm) for a, b in ((Xc, Xc), (Yc, Xc), (Yc, Yc)))
    Vx, A, Vy = cx.estimate(), ca.estimate(), cy.estimate()
    lx, la = cx.loo(), ca.loo()
    slope = A / Vx if Vx > 0 else float("nan")
    out = {"K": int(K), "M": int(M), "V_x": Vx, "A": A, "V_y": Vy, "V_y_negative": bool(Vy < 0),
           "V_x_nonpositive": bool(Vx <= 0), "slope": slope,
           "sign": None if not np.isfinite(slope) else ("tie" if slope == 0 else ("negative" if slope < 0 else "positive")),
           "slope_method": slope_method, "weights": w.tolist()}
    if slope_method == "fieller":
        kind, a, b = iv.fieller(A, Vx, la, lx, alpha)
    elif slope_method == "jack_t":
        with np.errstate(divide="ignore", invalid="ignore"):
            loo = np.where(lx > 0, la / np.where(lx > 0, lx, 1.0), np.nan)
        a, b = iv.jack_t(slope, loo, alpha)
        kind = "bounded" if np.isfinite(a) and np.isfinite(b) else "undefined"
    else:
        raise ValueError(f"unknown slope method {slope_method!r}")
    out["slope_set"] = {"kind": kind, "a": float(a), "b": float(b)}
    out["slope_set_sign"] = interval_sign(kind, a, b)
    out["mean_benefit"], out["mean_capability"] = float((Y @ w).mean()), float((X @ w).mean())
    return out


def bank_exact(X, Y) -> dict:
    """Exact values over an enumerated bank (rows are configurations, uniform sampler), as dataset1.exact_targets:
    population moments of the model means, with W the mean within-configuration panel variance of the errors."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    theta, delta = X.mean(0), Y.mean(0)
    Vx, A, Vy = d1.pvar(theta), d1.pcov(delta, theta), d1.pvar(delta)
    W = float(np.mean(np.var(Y - delta, axis=1, ddof=1)))
    slope = A / Vx if Vx > 0 else float("nan")
    return {"n_configurations": int(X.shape[0]), "M": int(X.shape[1]), "V_x": Vx, "A": A, "V_y": Vy, "W_y": W,
            "slope": slope, "R_benefit": Vy / (Vy + W) if Vy + W > 0 else float("nan"),
            "sign": None if not np.isfinite(slope) else ("tie" if slope == 0 else ("negative" if slope < 0 else "positive"))}


# --------------------------------------------------------------------------- single draws


def single_draw_slopes(X, Y) -> np.ndarray:
    """OLS slope of benefit on capability across the panel within each draw (NaN for no spread in capability)."""
    return np.array([d1.slope(y, x) for x, y in zip(np.asarray(X, float), np.asarray(Y, float))])


def opposite_sign_share(slopes, reference) -> dict:
    """How many single-draw slopes have the sign opposite to a reference slope. Undefined slopes and exact zeros
    are counted apart; the share is over the defined slopes, zeros included in the denominator."""
    s = np.asarray(slopes, float)
    ok = np.isfinite(s)
    ref = float(reference)
    n_def = int(ok.sum())
    zero = int((s[ok] == 0).sum())
    if not np.isfinite(ref) or ref == 0:
        return {"n_draws": int(len(s)), "n_defined": n_def, "n_zero": zero, "reference": ref,
                "n_opposite": None, "share_opposite": float("nan"), "note": "reference slope undefined or zero"}
    opp = int((np.sign(s[ok]) == -np.sign(ref)).sum())
    return {"n_draws": int(len(s)), "n_defined": n_def, "n_zero": zero, "reference": ref,
            "n_opposite": opp, "share_opposite": opp / n_def if n_def else float("nan"),
            "n_positive": int((s[ok] > 0).sum()), "n_negative": int((s[ok] < 0).sum())}


# --------------------------------------------------------------------------- subpanels


def subpanel_draw(members, size, n_resamples, rng):
    """Subpanels of `size` models from `members`: every distinct one when there are at most n_resamples of them,
    otherwise n_resamples distinct ones drawn uniformly without replacement. Returns (array (n, size), mode)."""
    members = np.asarray(sorted(members), int)
    if size > len(members) or size < 2:
        return np.empty((0, size), int), "none"
    combos = np.array(list(itertools.combinations(members.tolist(), size)), int)
    if len(combos) <= n_resamples:
        return combos, "all"
    pick = np.sort(rng.choice(len(combos), size=n_resamples, replace=False))
    return combos[pick], "sampled"


def _gather(X, subsets):
    """(n, K, k) array of each subset's columns."""
    return np.moveaxis(np.asarray(X, float)[:, subsets], 0, 1)


def batch_cross_moment(X, Y, subsets, loo=False):
    """cross_moment(X[:, s], Y[:, s]) for every subset s (rows of `subsets`), with the
    leave-one-draw-out values when loo is set (as intervals.CrossMoment)."""
    Xs, Ys = _gather(X, subsets), _gather(Y, subsets)
    n, K, k = Xs.shape
    Xc = Xs - Xs.mean(2, keepdims=True)
    Yc = Ys - Ys.mean(2, keepdims=True)
    Sx, Sy = Xc.sum(1), Yc.sum(1)
    q = np.einsum("nui,nui->nu", Xc, Yc)
    est = (np.einsum("ni,ni->n", Sx, Sy) - q.sum(1)) / (K * (K - 1) * (k - 1))
    if not loo:
        return est
    num = np.einsum("nui,nui->nu", Sx[:, None, :] - Xc, Sy[:, None, :] - Yc) - (q.sum(1)[:, None] - q)
    return est, num / ((K - 1) * (K - 2) * (k - 1))


def _ratio(a, v):
    a, v = np.asarray(a, float), np.asarray(v, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, a / np.where(v > 0, v, 1.0), np.nan)


def batch_slopes(X, Y, subsets):
    return _ratio(batch_cross_moment(Y, X, subsets), batch_cross_moment(X, X, subsets))


def pair_matrices_between(a, b):
    """(same, tie) matrices over model pairs for the orderings of two evaluations a and b (vectors over models):
    same[i, j] = 1 when both order i and j strictly and the same way; tie[i, j] = 1 when either ties."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sa, sb = tie_sign(a[:, None] - a[None, :]), tie_sign(b[:, None] - b[None, :])
    same = ((sa == sb) & (sa != 0)).astype(float)
    tie = ((sa == 0) | (sb == 0)).astype(float)
    return same, tie


def pair_matrices_draws(Y):
    """(same, tie) matrices of one-draw ordering agreement between distinct draws (pair_agreement,
    pair by pair)."""
    Y = np.asarray(Y, float)
    K = Y.shape[0]
    s = tie_sign(Y[:, :, None] - Y[:, None, :])
    pos, neg = (s > 0).sum(0).astype(float), (s < 0).sum(0).astype(float)
    denom = K * (K - 1)
    same = (pos * (pos - 1) + neg * (neg - 1)) / denom
    opp = 2 * pos * neg / denom
    return same, 1.0 - same - opp


def subset_pair_mean(mat, subsets):
    """Mean of mat[i, j] over the model pairs i < j of each subset."""
    subsets = np.asarray(subsets, int)
    i, j = np.triu_indices(subsets.shape[1], 1)
    return np.asarray(mat, float)[subsets[:, i], subsets[:, j]].mean(1)


def subset_agreement(same, tie, subsets):
    """C = same + tie / 2, equal weights over the model pairs of each subset."""
    return subset_pair_mean(same, subsets) + 0.5 * subset_pair_mean(tie, subsets)


def subpanel_statistics(Xd, Yd, Xv, Yv, subsets, slope_method="fieller", alpha=iv.ALPHA) -> dict:
    """Per subpanel: the cross-draw slope on discovery and on validation draws, where the discovery slope's 95% set
    lies relative to zero, the agreement of own-gain orderings between the discovery and the validation means, and
    the one-draw agreement of benefit orderings between distinct discovery draws."""
    subsets = np.asarray(subsets, int)
    A, la = batch_cross_moment(Yd, Xd, subsets, loo=True)
    V, lv = batch_cross_moment(Xd, Xd, subsets, loo=True)
    slope_d = _ratio(A, V)
    slope_v = batch_slopes(Xv, Yv, subsets)
    sets = []
    for e_a, e_v, l_a, l_v in zip(A, V, la, lv):
        if slope_method == "fieller":
            kind, a, b = iv.fieller(e_a, e_v, l_a, l_v, alpha)
        else:
            s = e_a / e_v if e_v > 0 else float("nan")
            lo, hi = iv.jack_t(s, _ratio(l_a, l_v), alpha)
            kind, a, b = ("bounded" if np.isfinite(lo) and np.isfinite(hi) else "undefined"), lo, hi
        sets.append(interval_sign(kind, a, b))
    same_b, tie_b = pair_matrices_between(np.asarray(Yd).mean(0), np.asarray(Yv).mean(0))
    same_1, tie_1 = pair_matrices_draws(Yd)
    return {"subsets": subsets, "slope_discovery": slope_d, "slope_validation": slope_v, "set_sign": np.array(sets),
            "C_own_gain_disc_val": subset_agreement(same_b, tie_b, subsets),
            "C_benefit_one_draw": subset_agreement(same_1, tie_1, subsets)}


def _signs(s):
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    return {"negative": int((s[ok] < 0).sum()), "positive": int((s[ok] > 0).sum()),
            "zero": int((s[ok] == 0).sum()), "undefined": int((~ok).sum())}


def _dist(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    q = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
            "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6])}


def summarize_subpanels(st: dict, mode: str) -> dict:
    """Shares over the subpanels of one stratum and size."""
    n = len(st["subsets"])
    sd, sv = st["slope_discovery"], st["slope_validation"]
    signs = _signs(sd)
    both = np.isfinite(sd) & np.isfinite(sv) & (sd != 0) & (sv != 0)
    sets = st["set_sign"]
    out = {"n_subpanels": int(n), "mode": mode, "slope_signs": signs,
           "share_negative": signs["negative"] / n if n else float("nan"),
           "share_positive": signs["positive"] / n if n else float("nan"),
           "set_sign": {k: int((sets == k).sum()) for k in ("above zero", "below zero", "includes zero",
                                                            "unbounded", "undefined")},
           "share_disc_val_same_sign": float(np.mean(np.sign(sd[both]) == np.sign(sv[both]))) if both.any()
           else float("nan"),
           "n_disc_val_compared": int(both.sum()),
           "slope_discovery": _dist(sd),
           "C_own_gain_disc_val": _dist(st["C_own_gain_disc_val"]),
           "C_benefit_one_draw": _dist(st["C_benefit_one_draw"])}
    out["share_set_above_zero"] = out["set_sign"]["above zero"] / n if n else float("nan")
    out["share_set_below_zero"] = out["set_sign"]["below zero"] / n if n else float("nan")
    return out


def all_subset_sign_shares(X, Y, members, size, chunk=20000) -> dict:
    """Sign counts of the cross-draw slope over every subset of `size` models from `members` (a check on sampled
    strata)."""
    combos = np.array(list(itertools.combinations(sorted(members), size)), int)
    counts = {"negative": 0, "positive": 0, "zero": 0, "undefined": 0}
    for start in range(0, len(combos), chunk):
        c = _signs(batch_slopes(X, Y, combos[start:start + chunk]))
        for k in counts:
            counts[k] += c[k]
    n = len(combos)
    return {"n_subpanels": int(n), **counts, "share_negative": counts["negative"] / n,
            "share_positive": counts["positive"] / n}


# --------------------------------------------------------------------------- direct-answer variants


_LINE_BREAK = re.compile(r"\r\n|\r|\n|\u2028|\u2029")
_DISPLAY_CLOSE = ("$$", "\\]")
_BOXED_OPEN = re.compile(r"\\boxed\{")
_INLINE_MATH = re.compile(r"\$([^${}\n]*)\$")


def _answer_line(text):
    """(start, end) character offsets of the line strict extraction reads: the last line that is neither empty
    nor a closing display-math delimiter (src/scoring.strict_extract with strip_latex)."""
    bounds, pos = [], 0
    for m in _LINE_BREAK.finditer(text):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, len(text)))
    for s, e in reversed(bounds):
        line = normalize_text(text[s:e]).strip()
        if line and line not in _DISPLAY_CLOSE:
            return s, e
    return None


def _boxed_value(content):
    c = normalize_text(content).replace("\\$", "$").strip()
    m = _INLINE_MATH.fullmatch(c)
    if m:
        c = m.group(1).strip()
    return normalize_number(c)


def boxed_answer_end(text, answer):
    """Character offset just past the closing brace of the boxed answer that strict extraction scored.

    The answer line is located as strict extraction locates it; every "\\boxed{" on that line is brace-matched
    within the line, and the last box whose content reads as `answer` (after the "\\$" and inline-math tolerance of
    scorer 1.3.0) is taken. Returns (end or None, number of matching boxes on the line)."""
    if not isinstance(text, str) or answer is None:
        return None, 0
    span = _answer_line(text)
    if span is None:
        return None, 0
    s, e = span
    ends = []
    for m in _BOXED_OPEN.finditer(text, s, e):
        depth, close = 1, None
        for i in range(m.end(), e):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close is not None and _boxed_value(text[m.end():close]) == answer:
            ends.append(close + 1)
    return (ends[-1] if ends else None), len(ends)


def token_position(offsets, end) -> int:
    """Number of tokens whose character span starts before `end` (the token holding the closing brace included)."""
    if end is None:
        return -1
    starts = np.fromiter((o[0] for o in offsets), dtype=np.int64, count=len(offsets))
    return int((starts < end).sum())


def capped_correct(correct, position, cap) -> np.ndarray:
    """Variant A: a correct output stays correct only when its boxed answer ends within the first `cap` tokens."""
    correct = np.asarray(correct, int)
    position = np.asarray(position, float)
    if ((correct == 1) & ~(position >= 1)).any():
        raise ValueError("every correct output needs a located boxed answer (position >= 1)")
    return np.where((correct == 1) & (position <= cap), 1, 0).astype(np.int64)


def long_direct_models(median_tokens: dict, cap) -> list:
    """Variant B: models whose median direct-arm output length exceeds `cap` tokens."""
    return sorted(k for k, v in median_tokens.items() if v > cap)


def quadratic_curve(theta, delta) -> dict:
    """Descriptive OLS quadratic of model-mean gain on model-mean baseline: coefficients, the vertex, and R^2 of the
    quadratic and of the line. An inverted U has a negative quadratic coefficient and a vertex inside the range."""
    x, y = np.asarray(theta, float), np.asarray(delta, float)
    if len(x) < 4:
        return {"n_models": int(len(x)), "note": "fewer than four models; not fitted"}
    c2, c1, c0 = np.polyfit(x, y, 2)
    b1, b0 = np.polyfit(x, y, 1)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2q = 1 - float(np.sum((y - np.polyval([c2, c1, c0], x)) ** 2)) / sst if sst > 0 else float("nan")
    r2l = 1 - float(np.sum((y - np.polyval([b1, b0], x)) ** 2)) / sst if sst > 0 else float("nan")
    vertex = -c1 / (2 * c2) if c2 != 0 else float("nan")
    inside = bool(np.isfinite(vertex) and x.min() <= vertex <= x.max())
    return {"n_models": int(len(x)), "c0": float(c0), "c1": float(c1), "c2": float(c2),
            "vertex_baseline": float(vertex), "vertex_gain": float(np.polyval([c2, c1, c0], vertex))
            if np.isfinite(vertex) else float("nan"),
            "vertex_inside_range": inside, "inverted_u": bool(c2 < 0 and inside),
            "r2_quadratic": r2q, "r2_linear": r2l, "linear_slope": float(b1),
            "label": "descriptive; no interval"}


# --------------------------------------------------------------------------- formatting for paper macros


def tex_num(x, nd=2, sign=False) -> str:
    """A number for paper/numbers.tex, with the minus sign in math mode (as paper/make_numbers.py writes it)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    s = s.replace("-", "$-$")
    return s.replace("+", "$+$") if sign else s


def tex_pct(x, nd=0) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{nd}f}\\%"
