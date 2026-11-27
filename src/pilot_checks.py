"""Pilot checks. The first asks whether a four-draw pilot delivers the agreement it promises: the pilot's reliability R picks, through the
Gaussian reference and Spearman-Brown (src/estimators.prompts_needed_gaussian), the number of draws K for 80% benefit
agreement, and the agreement actually achieved with K fresh draws is measured, either over an enumerated bank (DOVE)
or between disjoint groups of held-out draws. The second shrinks a pilot's centred gains by the reliability of a K-draw
mean, and charges rescaled transfer for the target draws it uses.

Arrays are (draws, models) in percentage points. Agreement is the tie-aware C of src/dataset1.agreement_between
(ties at estimators.TIE_TOL), vectorized over rows.
"""
from __future__ import annotations

import math

import numpy as np

from .estimators import cross_moment, prompts_needed_gaussian, reliability, tie_sign

TARGET = 0.80


def between_agreement(A, B) -> np.ndarray:
    """C between the paired rows of A and B (n, M): over model pairs, 1 for the same strict order, 1/2 when either
    evaluation ties, 0 otherwise; one value per row."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    i, j = np.triu_indices(A.shape[1], 1)
    sa, sb = tie_sign(A[:, i] - A[:, j]), tie_sign(B[:, i] - B[:, j])
    same = (sa == sb) & (sa != 0)
    tie = (sa == 0) | (sb == 0)
    return same.mean(1) + 0.5 * tie.mean(1)


def pilot_r(G) -> float:
    """Reliability of the pilot's benefits (nan when the within-draw variance is zero)."""
    return float(reliability(np.asarray(G, float))["R"])


def chosen_k(R, target=TARGET):
    """K for `target` agreement from a pilot's R; None (unresolved) when R is undefined or not positive."""
    if R is None or not np.isfinite(R) or R <= 0:
        return None
    return int(prompts_needed_gaussian(min(float(R), 1.0), target))


def _means_with_replacement(G, K, n, rng, chunk=256):
    """n means of K draws taken with replacement from the rows of G."""
    G = np.asarray(G, float)
    nb = G.shape[0]
    if K <= chunk:
        idx = rng.integers(0, nb, size=(n, K))
        return G[idx].mean(1)
    counts = rng.multinomial(K, np.full(nb, 1.0 / nb), size=n)
    return counts @ G / K


def achieved_enumerated(G, K, n_pairs, rng) -> float:
    """Expected C between two independent K-draw evaluations from the bank G, by Monte Carlo over n_pairs pairs."""
    a = _means_with_replacement(G, K, n_pairs, rng)
    b = _means_with_replacement(G, K, n_pairs, rng)
    return float(between_agreement(a, b).mean())


def achieved_heldout(Gv, K, n_splits, rng):
    """Mean C between two disjoint groups of K held-out draws over n_splits random splits; None when 2K exceeds the
    held-out draws."""
    Gv = np.asarray(Gv, float)
    kv = Gv.shape[0]
    if 2 * K > kv:
        return None
    perm = np.argsort(rng.random((n_splits, kv)), axis=1)
    a = Gv[perm[:, :K]].mean(1)
    b = Gv[perm[:, K:2 * K]].mean(1)
    return float(between_agreement(a, b).mean())


def classify(k, achieved, target=TARGET) -> str:
    if k is None:
        return "unresolved"
    if achieved is None:
        return "not measurable"
    return "meets" if achieved >= target else "short"


# --------------------------------------------------------------------------- 7.8


def shrink_factor(Gk):
    """Reliability of the K-draw mean, R_K = K R / (1 + (K - 1) R), with R from the same K draws, held to [0, 1]
    (a regularization inside the prediction rule); None when R is undefined or K < 2."""
    Gk = np.asarray(Gk, float)
    K = Gk.shape[0]
    if K < 2:
        return None
    R = pilot_r(Gk)
    if not np.isfinite(R):
        return None
    den = 1 + (K - 1) * R
    rk = K * R / den if den != 0 else math.inf
    return float(min(max(rk, 0.0), 1.0))


def charged_sd(Gt, k):
    """The target's SD of benefit across models from its first k draws: the SD of the one draw's gains for k = 1, the
    noise-corrected root of V_delta for k >= 2 (None when V_delta is not positive)."""
    Gt = np.asarray(Gt, float)[:k]
    if k == 1:
        return float(np.std(Gt[0], ddof=1))
    v = cross_moment(Gt, Gt)
    return float(np.sqrt(v)) if v > 0 else None
