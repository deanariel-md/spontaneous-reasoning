"""Candidate interval procedures for the targets.

Every statistic here is a function of K independent evaluation units, each a whole model
vector in both arms. Resampling therefore acts on whole units, never on model pairs or draw
pairs: whole model vectors are preserved. The cross-draw moments and the agreement
estimates are U-statistics of order two in the units, so three resampling summaries are
available in closed form from per-unit quantities:

  * leave-one-unit-out values (jackknife);
  * bootstrap values over positions ("naive": resampled copies of one unit count as distinct
    draws, as a plain bootstrap would treat them);
  * bootstrap values over pairs of distinct original units ("distinct": the U-statistic
    evaluated on the resample, excluding pairs formed by two copies of the same unit).

A bootstrap resample is represented by its multiplicity matrix m (n_boot, K), m[b, u] = number
of copies of unit u. Intervals are candidates only: which one is used is decided by the
calibration suite (scripts/run_calibration.py). Nothing here clips, truncates or imputes.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .estimators import tie_sign

ALPHA = 0.05


# --------------------------------------------------------------------------- moments


def _center(X):
    X = np.asarray(X, float)
    return X - X.mean(axis=1, keepdims=True)


class CrossMoment:
    """cross_moment(X, Y) with its jackknife and bootstrap values.

    With row-centred Xc, Yc (K, M): estimate = (sum_i Sx_i Sy_i - sum_u q_u) / (K (K-1) (M-1)),
    S = column sums over units, q_u = sum_i Xc_ui Yc_ui."""

    def __init__(self, X, Y, center=True, denom_models=None):
        self.Xc = _center(X) if center else np.asarray(X, float)
        self.Yc = _center(Y) if center else np.asarray(Y, float)
        self.K, self.M = self.Xc.shape
        self.dm = (self.M - 1) if denom_models is None else denom_models
        self.Sx, self.Sy = self.Xc.sum(0), self.Yc.sum(0)
        self.q = np.einsum("ui,ui->u", self.Xc, self.Yc)

    def estimate(self) -> float:
        K = self.K
        return float((self.Sx @ self.Sy - self.q.sum()) / (K * (K - 1) * self.dm))

    def loo(self) -> np.ndarray:
        K = self.K
        Sx = self.Sx[None, :] - self.Xc
        Sy = self.Sy[None, :] - self.Yc
        num = np.einsum("ui,ui->u", Sx, Sy) - (self.q.sum() - self.q)
        return num / ((K - 1) * (K - 2) * self.dm)

    def boot(self, m, distinct=True) -> np.ndarray:
        Sx, Sy = m @ self.Xc, m @ self.Yc
        cross = np.einsum("bi,bi->b", Sx, Sy)
        if distinct:
            m2 = m * m
            return (cross - m2 @ self.q) / ((self.K ** 2 - m2.sum(1)) * self.dm)
        return (cross - m @ self.q) / (self.K * (self.K - 1) * self.dm)

    def boot_loo(self, m) -> np.ndarray:
        """For each resample b and unit u: the distinct-pair estimate after deleting one copy of
        u (only meaningful where m[b, u] >= 1). Used for a jackknife SE inside each resample."""
        Sx, Sy = m @ self.Xc, m @ self.Yc
        m2 = (m * m).sum(1)
        N = np.einsum("bi,bi->b", Sx, Sy) - (m * m) @ self.q
        Kp = m.sum(1)
        Np = N[:, None] - Sx @ self.Yc.T - Sy @ self.Xc.T + 2 * m * self.q[None, :]
        Dp = ((Kp[:, None] - 1) ** 2 - (m2[:, None] - 2 * m + 1)) * self.dm
        with np.errstate(divide="ignore", invalid="ignore"):
            return Np / Dp


# --------------------------------------------------------------------------- agreement


class Agreement:
    """Tie-aware pairwise ordering agreement between distinct units (pair_agreement):
    C = same + tie / 2 = (1 + same - opposite) / 2, equal weights over model pairs."""

    def __init__(self, X):
        X = np.asarray(X, float)
        self.K, M = X.shape
        i, j = np.triu_indices(M, 1)
        s = tie_sign(X[:, i] - X[:, j])
        self.pos_u, self.neg_u = (s > 0).astype(float), (s < 0).astype(float)
        self.dec_u = self.pos_u + self.neg_u
        self.pos, self.neg = self.pos_u.sum(0), self.neg_u.sum(0)

    @staticmethod
    def _c(pos, neg, dup, denom):
        same = (pos * (pos - 1) + neg * (neg - 1) - dup).mean(-1) / denom
        opp = (2 * pos * neg).mean(-1) / denom
        return same, opp, 0.5 * (1 + same - opp)

    def parts(self) -> dict:
        K = self.K
        same, opp, C = self._c(self.pos, self.neg, 0.0, K * (K - 1))
        tie = 1 - same - opp
        return {"same": float(same), "opposite": float(opp),
                "tie": 0.0 if abs(tie) < 1e-12 else float(tie), "C": float(C)}

    def estimate(self) -> float:
        return self.parts()["C"]

    def loo(self) -> np.ndarray:
        K = self.K
        pos = self.pos[None, :] - self.pos_u
        neg = self.neg[None, :] - self.neg_u
        return self._c(pos, neg, 0.0, (K - 1) * (K - 2))[2]

    def boot(self, m, distinct=True) -> np.ndarray:
        K = self.K
        pos, neg = m @ self.pos_u, m @ self.neg_u
        if distinct:
            dup = (m * (m - 1)) @ self.dec_u
            denom = (K ** 2 - (m * m).sum(1))[:, None]
            same = (pos * (pos - 1) + neg * (neg - 1) - dup) / denom
            opp = 2 * pos * neg / denom
            return 0.5 * (1 + same.mean(1) - opp.mean(1))
        return self._c(pos, neg, 0.0, K * (K - 1))[2]

    def boot_loo(self, m) -> np.ndarray:
        """Distinct-pair C after deleting one copy of unit u from resample b, for all (b, u).
        With a = pos - neg and s = pos + neg per model pair, C = 1/2 + mean_p(T_p) / (2 D), where
        T_p = a^2 - s - dup_p and D = K'^2 - sum m^2; deleting a copy of u gives
        T'_p = T_p - 2 a a_u + 2 m_u d_u."""
        au = self.pos_u - self.neg_u
        npairs = au.shape[1]
        a = m @ au
        s_ = m @ self.dec_u
        dup = (m * (m - 1)) @ self.dec_u
        T = (a * a - s_ - dup).mean(1)
        decm = self.dec_u.mean(1)
        Tp = T[:, None] - 2 * (a @ au.T) / npairs + 2 * m * decm[None, :]
        Kp = m.sum(1)
        m2 = (m * m).sum(1)
        Dp = (Kp[:, None] - 1) ** 2 - (m2[:, None] - 2 * m + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.5 + Tp / (2 * Dp)


# --------------------------------------------------------------------------- intervals


def jackknife_se(loo) -> float:
    loo = np.asarray(loo, float)
    K = len(loo)
    if not np.isfinite(loo).all():
        return float("nan")
    return float(math.sqrt((K - 1) / K * np.sum((loo - loo.mean()) ** 2)))


def jack_t(est, loo, alpha=ALPHA):
    """Two-sided t interval with jackknife SE and K - 1 degrees of freedom."""
    se = jackknife_se(loo)
    if not (np.isfinite(est) and np.isfinite(se)):
        return float("nan"), float("nan")
    t = stats.t.ppf(1 - alpha / 2, len(loo) - 1)
    return est - t * se, est + t * se


def jack_t_upper(est, loo, alpha=ALPHA) -> float:
    """One-sided upper bound at level 1 - alpha."""
    se = jackknife_se(loo)
    if not (np.isfinite(est) and np.isfinite(se)):
        return float("nan")
    return est + stats.t.ppf(1 - alpha, len(loo) - 1) * se


def percentile(boot, alpha=ALPHA):
    """Percentile interval over the finite bootstrap values, with the undefined fraction."""
    b = np.asarray(boot, float)
    f = b[np.isfinite(b)]
    frac_undef = 1 - len(f) / len(b)
    if len(f) == 0:
        return float("nan"), float("nan"), frac_undef
    lo, hi = np.quantile(f, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), frac_undef


def percentile_upper(boot, alpha=ALPHA) -> float:
    f = np.asarray(boot, float)
    f = f[np.isfinite(f)]
    return float(np.quantile(f, 1 - alpha)) if len(f) else float("nan")


def _bca_levels(est, boot, loo, probs):
    """BCa-adjusted quantile levels (Efron 1987): bias correction z0 from the bootstrap
    distribution, acceleration from the leave-one-unit-out values."""
    b = np.asarray(boot, float)
    f = b[np.isfinite(b)]
    loo = np.asarray(loo, float)
    if len(f) < 0.5 * len(b) or not np.isfinite(est) or not np.isfinite(loo).all():
        return None, f
    n = len(f)
    prop = (np.sum(f < est) + 0.5 * np.sum(f == est)) / n
    prop = min(max(prop, 0.5 / n), 1 - 0.5 / n)
    z0 = stats.norm.ppf(prop)
    d = loo.mean() - loo
    ss = np.sum(d ** 2)
    a = float(np.sum(d ** 3) / (6 * ss ** 1.5)) if ss > 0 else 0.0
    z = stats.norm.ppf(np.asarray(probs, float))
    return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))), f


def bca(est, boot, loo, alpha=ALPHA):
    """Two-sided BCa interval; undefined if most bootstrap values or any jackknife value is."""
    lv, f = _bca_levels(est, boot, loo, [alpha / 2, 1 - alpha / 2])
    if lv is None:
        return float("nan"), float("nan")
    lo, hi = np.quantile(f, lv)
    return float(lo), float(hi)


def bca_upper(est, boot, loo, alpha=ALPHA) -> float:
    lv, f = _bca_levels(est, boot, loo, [1 - alpha])
    return float("nan") if lv is None else float(np.quantile(f, lv[0]))


def weighted_jackknife_se(loo_bu, m) -> np.ndarray:
    """Jackknife SE over the K' positions of each resample: positions copying unit u share the
    leave-one-copy-out value loo_bu[b, u] and get weight m[b, u]."""
    Kp = m.sum(1)
    w = m / Kp[:, None]
    with np.errstate(invalid="ignore"):
        lz = np.where(m > 0, loo_bu, 0.0)
        mean = (w * lz).sum(1)
        var = (Kp - 1) * (w * (lz - mean[:, None]) ** 2).sum(1)
        bad = ((m > 0) & ~np.isfinite(loo_bu)).any(1)
    se = np.sqrt(var)
    se[bad] = np.nan
    return se


def boot_t(est, loo, boot, boot_loo, m, alpha=ALPHA):
    """Studentized bootstrap (bootstrap-t) with jackknife SEs in the sample and in each
    resample. Undefined if the sample SE is, or if fewer than half the resamples give a finite
    studentized value."""
    se = jackknife_se(loo)
    if not (np.isfinite(est) and np.isfinite(se) and se > 0):
        return float("nan"), float("nan")
    se_b = weighted_jackknife_se(boot_loo, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.asarray(boot, float) - est) / se_b
    t = t[np.isfinite(t)]
    if len(t) < 0.5 * len(m):
        return float("nan"), float("nan")
    qlo, qhi = np.quantile(t, [alpha / 2, 1 - alpha / 2])
    return float(est - qhi * se), float(est - qlo * se)


def boot_t_upper(est, loo, boot, boot_loo, m, alpha=ALPHA) -> float:
    se = jackknife_se(loo)
    if not (np.isfinite(est) and np.isfinite(se) and se > 0):
        return float("nan")
    se_b = weighted_jackknife_se(boot_loo, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.asarray(boot, float) - est) / se_b
    t = t[np.isfinite(t)]
    if len(t) < 0.5 * len(m):
        return float("nan")
    return float(est - np.quantile(t, alpha) * se)


def fieller(num, den, loo_num, loo_den, alpha=ALPHA):
    """Fieller set for num / den with the jackknife covariance of (num, den) and t_{K-1}.

    Returns (kind, a, b): kind 'bounded' gives [a, b]; 'exclusive' gives (-inf, a] U [b, inf);
    'whole' gives the real line; 'empty' cannot occur for a valid covariance but is reported."""
    ln, ld = np.asarray(loo_num, float), np.asarray(loo_den, float)
    K = len(ln)
    if not (np.isfinite(ln).all() and np.isfinite(ld).all()):
        return "undefined", float("nan"), float("nan")
    c = (K - 1) / K
    s_nn = c * np.sum((ln - ln.mean()) ** 2)
    s_dd = c * np.sum((ld - ld.mean()) ** 2)
    s_nd = c * np.sum((ln - ln.mean()) * (ld - ld.mean()))
    t2 = stats.t.ppf(1 - alpha / 2, K - 1) ** 2
    qa = den ** 2 - t2 * s_dd
    qb = -2 * (num * den - t2 * s_nd)
    qc = num ** 2 - t2 * s_nn
    disc = qb ** 2 - 4 * qa * qc
    if qa > 0:
        if disc < 0:
            return "empty", float("nan"), float("nan")
        r = math.sqrt(disc)
        return "bounded", (-qb - r) / (2 * qa), (-qb + r) / (2 * qa)
    if disc < 0:
        return "whole", -math.inf, math.inf
    if qa == 0:
        return "whole", -math.inf, math.inf
    r = math.sqrt(disc)
    lo, hi = sorted(((-qb - r) / (2 * qa), (-qb + r) / (2 * qa)))
    return "exclusive", lo, hi


def fieller_covers(kind, a, b, value) -> bool:
    if kind == "bounded":
        return a <= value <= b
    if kind == "exclusive":
        return value <= a or value >= b
    return kind == "whole"


def scale_fieller(kind, a, b, factor):
    """Fieller set for factor * ratio (factor may be negative)."""
    if kind in ("bounded", "exclusive"):
        x, y = sorted((factor * a, factor * b))
        return kind, x, y
    return kind, a, b


# --------------------------------------------------------------------------- variance components


def anova_mean_squares(G):
    """Model-by-draw layout of gains G (K, M): MS_models, MS_error and their df.
    (MS_models - MS_error) / K equals cross_moment(G, G)."""
    Gc = _center(G)
    K, M = Gc.shape
    gbar = Gc.mean(0)
    ms_a = K * np.sum(gbar ** 2) / (M - 1)
    ms_e = np.sum((Gc - gbar) ** 2) / ((K - 1) * (M - 1))
    return ms_a, ms_e, M - 1, (K - 1) * (M - 1)


def mls_difference(s1, s2, n1, n2, alpha):
    """Modified large-sample (Graybill-Wang / Ting et al.) one-sided bounds at level 1 - alpha
    for theta1 - theta2, where n_i s_i / theta_i ~ chi2(n_i) independently."""
    f_up = lambda g, a, b: stats.f.isf(g, a, b)        # F value with g in the upper tail
    finf = lambda g, a: stats.chi2.isf(g, a) / a
    G1 = 1 - 1 / finf(alpha, n1)
    H1 = 1 / finf(1 - alpha, n1) - 1
    G2 = 1 - 1 / finf(alpha, n2)
    H2 = 1 / finf(1 - alpha, n2) - 1
    F12 = f_up(alpha, n1, n2)
    F21 = f_up(1 - alpha, n1, n2)
    G12 = ((F12 - 1) ** 2 - G1 ** 2 * F12 ** 2 - H2 ** 2) / F12
    H12 = ((1 - F21) ** 2 - H1 ** 2 * F21 ** 2 - G2 ** 2) / F21
    d = s1 - s2
    lower = d - math.sqrt(max(G1 ** 2 * s1 ** 2 + H2 ** 2 * s2 ** 2 + G12 * s1 * s2, 0.0))
    upper = d + math.sqrt(max(H1 ** 2 * s1 ** 2 + G2 ** 2 * s2 ** 2 + H12 * s1 * s2, 0.0))
    return lower, upper


def mls_vdelta(G, alpha=ALPHA):
    """Normal-theory variance-component interval for V_delta = (E MS_models - E MS_error) / K.
    Assumes a random-effects Gaussian layout; it must earn its use in calibration."""
    ms_a, ms_e, n1, n2 = anova_mean_squares(G)
    K = np.asarray(G).shape[0]
    lo, _ = mls_difference(ms_a, ms_e, n1, n2, alpha / 2)
    _, hi = mls_difference(ms_a, ms_e, n1, n2, alpha / 2)
    return lo / K, hi / K


def mls_vdelta_upper(G, alpha=ALPHA) -> float:
    ms_a, ms_e, n1, n2 = anova_mean_squares(G)
    K = np.asarray(G).shape[0]
    return mls_difference(ms_a, ms_e, n1, n2, alpha)[1] / K


# --------------------------------------------------------------------------- binomial bound


def coverage_lower_bound(hits: int, n: int, level=0.95) -> float:
    """One-sided Clopper-Pearson lower confidence bound on a coverage probability."""
    if n == 0:
        return float("nan")
    if hits == 0:
        return 0.0
    return float(stats.beta.ppf(1 - level, hits, n - hits + 1))


def multiplicities(rng, K: int, n_boot: int) -> np.ndarray:
    """Bootstrap multiplicity matrix (n_boot, K): whole units resampled with replacement."""
    return rng.multinomial(K, np.full(K, 1 / K), size=n_boot).astype(float)


def coverage_upper_bound(hits: int, n: int, level=0.95) -> float:
    """One-sided Clopper-Pearson upper confidence bound on a coverage probability."""
    if n == 0:
        return float("nan")
    if hits == n:
        return 1.0
    return float(stats.beta.ppf(level, hits + 1, n - hits))
