"""Which part of the baseline shift makes the reversals.

A model's gain under the requested baseline is g_R = C - R and under the forced baseline g_F = C - F, so
g_F = g_R + D with D = R - F, the requested-minus-forced accuracy. src/baseline_mixture.decomposition splits D exactly
into a long part (rows whose requested reply is longer than the cut) and a short part (the other rows): D = L + S.
"Replace D by its long part" gives the forced-baseline gain g_R + L = g_F - S: the baseline switch then changes
accuracy only on the rows with a long requested reply. "Replace D by its short part" gives g_R + S = g_F - L: the
switch changes accuracy only on the rows with a short requested reply. A pair is reversed when g_R orders it one way
and the replaced forced-baseline gain the other way; ties (|gap| <= 1e-9 pp) under either are not reversed (D8).

Pure functions on model means (pp); scripts/run_attr_flips.py loads the data.
"""
from __future__ import annotations

import numpy as np

from src import baseline_mixture as bm
from src import robustness as rb

SUM_TOL_PP = 1e-8   # the two parts must add to D within this (they are sums of the same rows)

DEFINITION = (
    "Gain under the requested baseline: g_R = C - R. Gain under the forced baseline: g_F = C - F = g_R + D, with "
    "D = R - F split exactly into a long part L and a short part S (D = L + S; src/baseline_mixture.decomposition). "
    "Observed: pairs ordered oppositely by g_R and g_F. Long part alone: pairs ordered oppositely by g_R and g_R + L "
    "(= g_F - S). Short part alone: pairs ordered oppositely by g_R and g_R + S (= g_F - L). Both parts: pairs reversed "
    "under each part alone. Ties (|gap| <= 1e-9 pp) under either gain are not reversed; tied pairs stay in the "
    "denominator.")


def _pairs(g1, g2) -> set:
    return {tuple(p) for p in bm.reversed_pairs(g1, g2)}


def part_flips(g_req, g_fd, d_long, d_short, names=None, tol: float = SUM_TOL_PP) -> dict:
    """Reversed pairs between g_req and (a) g_fd, (b) g_req + d_long, (c) g_req + d_short; all (M,) model means in pp.

    Raises when d_long + d_short differs from g_fd - g_req by more than tol (the parts must come from the same rows and
    weights as the gains). Returns the counts, their overlaps with each other and with the observed pairs, the tie
    counts and the pair lists (by name when names are given, else by index)."""
    g_req, g_fd, L, S = (np.asarray(x, float) for x in (g_req, g_fd, d_long, d_short))
    if not (g_req.shape == g_fd.shape == L.shape == S.shape) or g_req.ndim != 1:
        raise ValueError("g_req, g_fd, d_long and d_short must be (M,) arrays of one length")
    if not np.all(np.isfinite(np.concatenate([g_req, g_fd, L, S]))):
        raise ValueError("a gain or part is not finite")
    D = g_fd - g_req
    resid = float(np.max(np.abs(L + S - D))) if len(D) else 0.0
    if resid > tol:
        raise ValueError(f"the long and short parts do not add to D (max |L + S - D| = {resid:.3g} pp)")
    alt = {"observed": g_fd, "long": g_req + L, "short": g_req + S}
    sets = {k: _pairs(g_req, v) for k, v in alt.items()}
    rev = {k: rb.pair_reversals(g_req, v) for k, v in alt.items()}
    for k in alt:
        if rev[k]["reversed"] != len(sets[k]):
            raise AssertionError(f"{k}: pair_reversals and reversed_pairs disagree")
    O, Lg, Sh = sets["observed"], sets["long"], sets["short"]
    both = Lg & Sh
    nm = list(names) if names is not None else list(range(len(g_req)))

    def named(s):
        return [[nm[a], nm[b]] for a, b in sorted(s)]
    return {
        "pairs": rev["observed"]["pairs"],
        "observed": len(O), "long": len(Lg), "short": len(Sh), "both": len(both),
        "observed_and_long": len(O & Lg), "observed_and_short": len(O & Sh),
        "observed_neither": len(O - Lg - Sh),
        "long_not_observed": len(Lg - O), "short_not_observed": len(Sh - O), "both_not_observed": len(both - O),
        "ties": {k: rev[k]["ties"] for k in alt},
        "sum_check_max_abs_pp": resid,
        "pair_lists": {"observed": named(O), "long": named(Lg), "short": named(Sh), "both": named(both),
                       "observed_neither": named(O - Lg - Sh)}}


def distribution(x) -> dict:
    """Median and range (min, max) over models, pp, with the count of models below, at and above zero."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": int(len(x)), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
            "below_zero": int((x < 0).sum()), "zero": int((x == 0).sum()), "above_zero": int((x > 0).sum())}
