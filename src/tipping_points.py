"""Tipping points of the headline results: the smallest number of individual scored
outputs whose score would have to be wrong, placed where they do the most harm, to reverse a result.

Layout. A result reads cells: one cell is a (model, arm, configuration) with n items, and its score is 100 times its
correct count over n, in percentage points. A draw array (draws, models) takes each row from the configuration of its
draw (`cell_index`), so a configuration drawn twice fills two rows, and one flipped output changes its cell by 100 / n pp
in every row that uses the configuration. A `Problem` holds, per cell, the correct count under the reading and how many
outputs may flip each way:
    unrestricted   every correct output may become incorrect and every incorrect one correct;
    restricted     only the outputs where a second scorer disagrees, each moved to that scorer's reading
                   (`down` = correct here and incorrect there, `up` = the reverse).

Search (`greedy`). At the current state the distance to reversal is linearized: the first-order effect
of one flip in each cell, in the direction that shortens the distance (one flip moves a cell by 0.08 pp, so the
second-order term is negligible). Flips are taken in order of effect until the next one would exceed 1% of the remaining
distance, at least one per batch; then the statistic is recomputed exactly. When a batch reverses the result, the
shortest prefix of the batch that does is found by bisection. For a statistic linear in the cells (a mean, the slope on
the index) the effects never change, so the greedy count is the exact minimum; for the others (the slope on baseline, t
bounds) it is the greedy count. Ties in effect keep the order of the cell list.

The count is the number of outputs whose score differs from the reading: the sum over cells of |net flips|, so a flip
undone later is not counted. A value within TIE_TOL (1e-9) of zero counts as zero. Nothing is clipped:
a result that the available outputs cannot reverse is reported as not reversed, with the distance left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from .estimators import TIE_TOL

BATCH_SHARE = 0.01
MAX_BATCHES = 200_000


# --------------------------------------------------------------------------- cells and draw arrays


def cell_index(keys, models, arm, draws) -> np.ndarray:
    """(K, M) positions in `keys` (tuples (model, arm, configuration)) of the cell that each draw row and model reads; a
    configuration drawn twice gives two rows with the same positions."""
    pos = {tuple(k): i for i, k in enumerate(keys)}
    try:
        return np.array([[pos[(m, arm, str(c))] for m in models] for c in draws], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"no cell for {e.args[0]}") from None


def scores(correct, n) -> np.ndarray:
    """Cell scores in pp, computed as src/dataset2.cell_table computes them."""
    return 100.0 * np.asarray(correct, float) / n


def scatter(grad, idx, n_cells) -> np.ndarray:
    """The gradient with respect to the cells of a statistic whose gradient with respect to a draw array read through
    `idx` is `grad`: every row that reads a cell adds to it."""
    out = np.zeros(int(n_cells))
    np.add.at(out, np.asarray(idx), np.asarray(grad, float))
    return out


# --------------------------------------------------------------------------- statistics and their gradients


def cross_moment_grad(X, Y):
    """cross_moment(X, Y) (distinct draws, panel denominator M - 1) and its derivatives with respect to every
    entry of X and of Y. With row-centred Xc, Yc and column sums S, the estimate is sum over draws u != v of Xc_u . Yc_v
    over K (K - 1) (M - 1), so d/dX_u = (S_Y - Yc_u) / that denominator, and symmetrically."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    K, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    D = K * (K - 1) * (M - 1)
    Sx, Sy = Xc.sum(0), Yc.sum(0)
    est = float((Sx @ Sy - np.sum(Xc * Yc)) / D)
    return est, (Sy[None, :] - Yc) / D, (Sx[None, :] - Xc) / D


def mean_gain(B, P):
    """Mean of P - B over draws and models, with its gradients (dB, dP)."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    w = 1.0 / B.size
    return float((P - B).mean()), np.full(B.shape, -w), np.full(P.shape, w)


def baseline_slope(B, P):
    """Points-scale slope of gain on baseline, cross_moment(P - B, B) / cross_moment(B, B) as src/instability.py
    computes it (scale_scores, scale_moments), with its gradients (dB, dP); nan where cross_moment(B, B) <= 0."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    A, dA_g, dA_b = cross_moment_grad(P - B, B)
    V, dV1, dV2 = cross_moment_grad(B, B)
    if not V > 0:
        nan = np.full(B.shape, np.nan)
        return float("nan"), nan, nan
    s = A / V
    return float(s), (-dA_g + dA_b - s * (dV1 + dV2)) / V, dA_g / V


def index_slope(B, P, e):
    """Slope of gain on the external index (the same value in every draw), cross_moment(P - B, index) /
    cross_moment(index, index), with its gradients (dB, dP); linear in the cells. nan where the index does not vary."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    X = np.tile(np.asarray(e, float), (B.shape[0], 1))
    A, dA_g, _ = cross_moment_grad(P - B, X)
    V, _, _ = cross_moment_grad(X, X)
    if not V > 0:
        nan = np.full(B.shape, np.nan)
        return float("nan"), nan, nan
    return float(A / V), -dA_g / V, dA_g / V


def t_quantile(level, K) -> float:
    """Two-sided t quantile at `level` with K - 1 degrees of freedom."""
    return float(stats.t.ppf(0.5 + level / 2, K - 1))


def t_interval(d, level) -> list:
    """t interval over the values d (one per draw), K - 1 degrees of freedom, as scripts/run_exemplar.py computes it."""
    d = np.asarray(d, float)
    h = t_quantile(level, len(d)) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def t_bound(d, q, side):
    """mean(d) + side q sd(d) / sqrt(K) (side -1: lower bound, +1: upper bound) and its gradient with respect to d. Where
    sd(d) is zero the bound has no gradient in sd; the mean's part is returned alone."""
    d = np.asarray(d, float)
    K = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    b = float(m + side * q * sd / np.sqrt(K))
    g = np.full(K, 1.0 / K)
    if sd > 0:
        g = g + side * q / np.sqrt(K) * (d - m) / ((K - 1) * sd)
    return b, g


def tercile_members(rank) -> dict:
    """The terciles: the ceil(M/3) models with the lowest and with the highest rank (stable order), as
    scripts/run_exemplar.tercile_block forms them."""
    rank = np.asarray(rank, float)
    k = -(-len(rank) // 3)
    order = np.argsort(rank, kind="stable")
    return {"weak": order[:k], "strong": order[-k:]}


def tercile_reading(rank, Bv, Pv, margin):
    """The tercile reading with scripts/run_exemplar.tercile_block's numbers: each tercile's per-draw mean benefit over the
    validation draws, its 95% and 90% t intervals, and whether the claim holds (weak 95% lower bound above zero, strong
    90% interval inside (-margin, margin)). A bound within TIE_TOL of zero counts as zero."""
    G = np.asarray(Pv, float) - np.asarray(Bv, float)
    groups = tercile_members(rank)
    out = {"members": {g: [int(i) for i in ix] for g, ix in groups.items()}}
    for g, ix in groups.items():
        d = G[:, ix].mean(1)
        out[g] = {"mean": float(d.mean()), "ci95": t_interval(d, 0.95), "ci90": t_interval(d, 0.90)}
    out["weak_benefits"] = bool(out["weak"]["ci95"][0] > TIE_TOL)
    out["strong_equivalent"] = bool(-margin < out["strong"]["ci90"][0] and out["strong"]["ci90"][1] < margin)
    out["claim_holds"] = out["weak_benefits"] and out["strong_equivalent"]
    return out


def tercile_bound(Bv, Pv, ix, q, side):
    """A t bound of one tercile's per-draw mean benefit, with its gradients (dB, dP)."""
    G = np.asarray(Pv, float) - np.asarray(Bv, float)
    ix = np.asarray(ix)
    b, gd = t_bound(G[:, ix].mean(1), q, side)
    dP = np.zeros(G.shape)
    dP[:, ix] = gd[:, None] / len(ix)
    return b, -dP, dP


# --------------------------------------------------------------------------- problems


@dataclass
class Problem:
    """One result as a function of its cells (module doc). `distance` maps cell scores (pp) to the distance left to
    reversal (positive while not reversed) and its gradient with respect to the cell scores; `reversed` decides
    reversal exactly; `value` gives the statistic's values for the report."""
    correct: np.ndarray
    n: int
    down: np.ndarray
    up: np.ndarray
    distance: Callable
    reversed: Callable
    value: Callable

    def scores(self, k=None) -> np.ndarray:
        c = np.asarray(self.correct, float) if k is None else np.asarray(self.correct, float) + k
        return scores(c, self.n)


def unrestricted(correct, n):
    """(down, up): every correct output may become incorrect and every incorrect one correct."""
    c = np.asarray(correct, np.int64)
    return c.copy(), n - c


def restricted(here_only, there_only):
    """(down, up) for flips among disagreeing outputs only: `here_only` outputs are correct under the reading and
    incorrect under the second scorer (they may flip down), `there_only` the reverse (they may flip up)."""
    return np.asarray(here_only, np.int64).copy(), np.asarray(there_only, np.int64).copy()


def _gain_dist(stat_grad, idx_b, idx_p, n_cells, sign):
    """distance = sign x statistic (sign +1: reversed when the statistic reaches zero or below)."""
    def dist(v):
        s, dB, dP = stat_grad(v[idx_b], v[idx_p])
        return sign * s, sign * (scatter(dB, idx_b, n_cells) + scatter(dP, idx_p, n_cells))
    return dist


def at_or_below_zero_problem(correct, n, down, up, idx_b, idx_p, stat_grad, name):
    """A statistic of (B, P) that is positive and is reversed when it reaches zero or below (within TIE_TOL)."""
    C = len(correct)
    dist = _gain_dist(stat_grad, idx_b, idx_p, C, 1.0)

    def rev(v):
        s = stat_grad(v[idx_b], v[idx_p])[0]
        return bool(np.isfinite(s) and s <= TIE_TOL)

    def value(v):
        return {name: stat_grad(v[idx_b], v[idx_p])[0]}
    return Problem(np.asarray(correct, np.int64), n, np.asarray(down, np.int64), np.asarray(up, np.int64), dist, rev,
                   value)


TERCILE_ROUTES = {"weak_lower": ("weak", 0.95, -1), "strong_upper": ("strong", 0.90, 1),
                  "strong_lower": ("strong", 0.90, -1), "hold": ("weak", 0.95, -1)}


def tercile_problem(correct, n, down, up, idx_bv, idx_pv, rank_of, route, margin):
    """The tercile reading as a function of its cells. `rank_of(v)` gives the models' rank from the cell scores (the strict
    zero-shot mean over the discovery draws, recomputed after each batch; or fixed). Routes to make a holding claim fail:
    weak_lower (the weak tercile's 95% lower bound to zero or below), strong_upper (the strong tercile's 90% upper bound
    to +margin or above), strong_lower (its 90% lower bound to -margin or below); reversed when the claim fails. Route
    hold makes a failing claim hold through the weak tercile's 95% lower bound; reversed when the claim holds."""
    group, level, side = TERCILE_ROUTES[route]
    C = len(correct)
    K = np.asarray(idx_bv).shape[0]
    q = t_quantile(level, K)
    target = {"weak_lower": 0.0, "strong_upper": margin, "strong_lower": -margin, "hold": 0.0}[route]
    # distance > 0 while not reversed: bound - target for a bound that must fall, target - bound for one that must rise
    sign = -1.0 if route in ("strong_upper", "hold") else 1.0

    def dist(v):
        ix = tercile_members(rank_of(v))[group]
        b, dB, dP = tercile_bound(v[idx_bv], v[idx_pv], ix, q, side)
        return sign * (b - target), sign * (scatter(dB, idx_bv, C) + scatter(dP, idx_pv, C))

    def value(v):
        r = tercile_reading(rank_of(v), v[idx_bv], v[idx_pv], margin)
        return r

    def rev(v):
        holds = value(v)["claim_holds"]
        return holds if route == "hold" else not holds
    return Problem(np.asarray(correct, np.int64), n, np.asarray(down, np.int64), np.asarray(up, np.int64), dist, rev,
                   value)


# --------------------------------------------------------------------------- search


def _apply(k, segs, L):
    k = k.copy()
    for c, s, t in segs:
        u = min(t, L)
        k[c] += s * u
        L -= u
        if L <= 0:
            break
    return k


def greedy(p: Problem, share=BATCH_SHARE, max_batches=MAX_BATCHES) -> dict:
    """The greedy search of the module doc. Returns the status ("reversed", "already_reversed", "not_reversed" when no
    available flip shortens the distance, "undefined" when the statistic is not defined, "stalled" after max_batches),
    the count, the net flips per cell k (positive: incorrect to correct), the batches and the distance and values at the
    start and end."""
    C = len(p.correct)
    step = 100.0 / p.n
    k = np.zeros(C, np.int64)
    v = p.scores()
    d0, _ = p.distance(v)
    out = {"distance_start": float(d0), "value_start": p.value(v), "share_per_batch": share}

    def done(status, k, batches, d):
        vk = p.scores(k)
        return {**out, "status": status, "count": int(np.abs(k).sum()), "k": k, "batches": batches,
                "distance_end": float(d), "value_end": p.value(vk)}
    if p.reversed(v):
        return done("already_reversed", k, 0, d0)
    for it in range(max_batches):
        d, g = p.distance(v)
        if not (np.isfinite(d) and np.all(np.isfinite(g))):
            return done("undefined", k, it, d)
        e = g * step                       # change of the distance from one flip up (incorrect to correct)
        gain = np.abs(e)
        sgn = np.where(e < 0, 1, -1)       # the direction that shortens the distance
        avail = np.where(sgn > 0, p.up - k, p.down + k)
        ok = (gain > 0) & (avail > 0)
        if not ok.any():
            return done("not_reversed", k, it, d)
        cand = np.flatnonzero(ok)
        order = cand[np.argsort(-gain[cand], kind="stable")]
        budget, acc, segs = share * d, 0.0, []
        for c in order:
            room = (budget - acc) / gain[c]
            t = int(min(avail[c], math.floor(room))) if room > 0 else 0
            if t <= 0:
                if segs:
                    break
                t = 1
            segs.append((int(c), int(sgn[c]), t))
            acc += t * gain[c]
            if t < avail[c]:
                break
        T = sum(t for _, _, t in segs)
        k_new = _apply(k, segs, T)
        v_new = p.scores(k_new)
        if p.reversed(v_new):
            lo, hi = 0, T
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if p.reversed(p.scores(_apply(k, segs, mid))):
                    hi = mid
                else:
                    lo = mid
            k_end = _apply(k, segs, hi)
            return done("reversed", k_end, it + 1, p.distance(p.scores(k_end))[0])
        k, v = k_new, v_new
    return done("stalled", k, max_batches, p.distance(v)[0])


def exact_minimum_linear(effects_up, down, up, distance) -> int | None:
    """For a statistic linear in the cells: the smallest number of flips whose summed effects cover `distance`, where
    one flip up in cell c shortens the distance by effects_up[c] (a flip down by -effects_up[c]); None when all
    available flips do not cover it. Used to check `greedy` (it equals the greedy count)."""
    eff = []
    for c, x in enumerate(np.asarray(effects_up, float)):
        if x > 0:
            eff += [x] * int(up[c])
        elif x < 0:
            eff += [-x] * int(down[c])
    eff = np.sort(np.asarray(eff, float))[::-1]
    cum = np.cumsum(eff)
    hit = np.flatnonzero(cum >= distance - TIE_TOL)
    return int(hit[0]) + 1 if len(hit) else None


def composition(k, keys, here_only=None, there_only=None, arms=None) -> dict:
    """What a solution flips: outputs by arm and direction, the cells, models and configurations touched, and the
    outputs in the touched cells where the two scorers disagree."""
    k = np.asarray(k)
    hit = np.flatnonzero(k != 0)
    arms = arms or sorted({key[1] for key in keys})
    out = {"count": int(np.abs(k).sum()), "cells": int(len(hit)),
           "models": len({keys[i][0] for i in hit}), "configurations": len({keys[i][2] for i in hit}),
           "by_arm": {a: {"to_correct": int(sum(k[i] for i in hit if keys[i][1] == a and k[i] > 0)),
                          "to_incorrect": int(-sum(k[i] for i in hit if keys[i][1] == a and k[i] < 0))} for a in arms}}
    if here_only is not None:
        out["disagreeing_in_touched_cells"] = int(np.asarray(here_only)[hit].sum() + np.asarray(there_only)[hit].sum())
    return out
