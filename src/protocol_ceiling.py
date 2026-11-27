"""Protocol ceiling curve: how often two evaluations order a model pair's
gain the same way when each averages K prompt draws, within one protocol and between two protocols.

A protocol is one benefit definition: comparison arm x scoring x scale (src/instability.scale_scores, applied per
draw). Arrays are distinct configurations by models. Agreement over model pairs counts a tie under either evaluation as
one half (the agreement C).

    within(K)            two disjoint sets of K configurations, the same protocol; needs 2K <= D
    between, same(K)     one set of K configurations, two protocols (a study that changes only the protocol)
    between, disjoint(K) two disjoint sets of K configurations, two protocols; needs 2K <= D

Gaussian pair model (a reference, not a law): for each model pair the per-configuration gain difference has mean m and
variance s^2 (ddof 1) over the D configurations; with t = m sqrt(K) / s, two independent K-means order the pair the same
way with probability Phi(t)^2 + Phi(-t)^2 under one protocol, and Phi(t_p) Phi(t_q) + Phi(-t_p) Phi(-t_q) under two
protocols with independent errors. As K grows, within-protocol agreement tends to one (for m != 0) and between-protocol
agreement tends to the share of pairs whose true differences have the same sign under both protocols. That limit is
estimated two ways: plug-in, the agreement of the two protocols' means over all D configurations; and posterior, each
pair's Phi(a) Phi(b) + Phi(-a) Phi(-b) with a, b the pair's mean difference over its standard error at D configurations
(flat prior, independent errors; the same configurations make the two errors correlated, so this is approximate).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .estimators import tie_sign


def pair_c(a, b) -> float:
    """C between two evaluations (vectors over models): share of model pairs ordered the same way, ties one half."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    sa, sb = tie_sign(a[i] - a[j]), tie_sign(b[i] - b[j])
    tie = (sa == 0) | (sb == 0)
    same = (sa == sb) & ~tie
    return float(same.mean() + 0.5 * tie.mean())


def within_curve(Y, Ks, rng, n_splits: int = 200) -> dict:
    """{K: mean C} between disjoint K-sets of rows, same protocol; K with 2K > D are left out."""
    Y = np.asarray(Y, float)
    D = Y.shape[0]
    out = {}
    for K in Ks:
        if 2 * K > D:
            continue
        v = []
        for _ in range(n_splits):
            p = rng.permutation(D)
            v.append(pair_c(Y[p[:K]].mean(0), Y[p[K:2 * K]].mean(0)))
        out[K] = float(np.mean(v))
    return out


def between_curve(Ya, Yb, Ks, rng, n_splits: int = 200, disjoint: bool = False) -> dict:
    """{K: mean C} between two protocols' K-means: the same K rows, or (disjoint) two disjoint K-sets."""
    Ya, Yb = np.asarray(Ya, float), np.asarray(Yb, float)
    D = Ya.shape[0]
    out = {}
    for K in Ks:
        if K > D or (disjoint and 2 * K > D):
            continue
        if not disjoint and K == D:
            out[K] = pair_c(Ya.mean(0), Yb.mean(0))
            continue
        v = []
        for _ in range(n_splits):
            p = rng.permutation(D)
            s1, s2 = (p[:K], p[K:2 * K]) if disjoint else (p[:K], p[:K])
            v.append(pair_c(Ya[s1].mean(0), Yb[s2].mean(0)))
        out[K] = float(np.mean(v))
    return out


def pair_moments(Y):
    """(m, s) per model pair i < j: mean and standard deviation (ddof 1) of the per-row difference."""
    Y = np.asarray(Y, float)
    i, j = np.triu_indices(Y.shape[1], 1)
    d = Y[:, i] - Y[:, j]
    return d.mean(0), d.std(0, ddof=1)


def _t(m, s, K):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m * np.sqrt(K) / s
    return np.where(s > 0, t, np.where(m > 0, np.inf, np.where(m < 0, -np.inf, 0.0)))


def gaussian_within(Y, K) -> float:
    m, s = pair_moments(Y)
    t = _t(m, s, K)
    a = stats.norm.cdf(t)
    return float(np.mean(a * a + (1 - a) * (1 - a)))


def gaussian_between(Ya, Yb, K) -> float:
    ma, sa = pair_moments(Ya)
    mb, sb = pair_moments(Yb)
    a, b = stats.norm.cdf(_t(ma, sa, K)), stats.norm.cdf(_t(mb, sb, K))
    return float(np.mean(a * b + (1 - a) * (1 - b)))


def limit_posterior(Ya, Yb) -> float:
    """Posterior estimate of the share of pairs whose true differences share a sign under both protocols."""
    D = np.asarray(Ya).shape[0]
    return gaussian_between(Ya, Yb, D)


def limit_plugin(Ya, Yb) -> float:
    return pair_c(np.asarray(Ya, float).mean(0), np.asarray(Yb, float).mean(0))


def bootstrap_limit(Ya, Yb, rng, n_boot: int = 2000) -> dict:
    """Percentile interval of the plug-in limit, resampling configurations (rows) with replacement."""
    D = np.asarray(Ya).shape[0]
    v = []
    for _ in range(n_boot):
        ix = rng.integers(0, D, D)
        v.append(limit_plugin(np.asarray(Ya)[ix], np.asarray(Yb)[ix]))
    v = np.array(v)
    return {"ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))], "n_boot": n_boot}
