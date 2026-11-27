"""Closed-form resampling summaries in src/intervals.py against brute force, and the
normal-theory variance-component interval under its own assumptions."""
import itertools
import math

import numpy as np
import pytest

from src import intervals as iv
from src.estimators import cross_moment, cross_risk, pair_agreement


@pytest.fixture(scope="module")
def units():
    rng = np.random.default_rng(11)
    K, M = 7, 6
    B = rng.normal(50, 10, (K, M)) + rng.normal(0, 3, (K, 1))
    P = B + rng.normal(5, 3, (K, M))
    return B, P


def _resample(X, counts):
    return np.repeat(X, counts.astype(int), axis=0)


def _distinct_cross(X, Y, counts, center=True, dm=None):
    """U-statistic over position pairs whose original units differ."""
    owner = np.repeat(np.arange(len(counts)), counts.astype(int))
    Xr, Yr = _resample(X, counts), _resample(Y, counts)
    if center:
        Xr = Xr - Xr.mean(1, keepdims=True)
        Yr = Yr - Yr.mean(1, keepdims=True)
    M = X.shape[1]
    dm = M - 1 if dm is None else dm
    num, n = 0.0, 0
    for r, s in itertools.permutations(range(len(owner)), 2):
        if owner[r] != owner[s]:
            num += Xr[r] @ Yr[s]
            n += 1
    return num / (n * dm)


def _distinct_agreement(X, counts):
    owner = np.repeat(np.arange(len(counts)), counts.astype(int))
    Xr = _resample(X, counts)
    M = X.shape[1]
    i, j = np.triu_indices(M, 1)
    s = np.sign(Xr[:, i] - Xr[:, j])
    same = opp = 0.0
    n = 0
    for r, t in itertools.permutations(range(len(owner)), 2):
        if owner[r] != owner[t]:
            same += np.mean((s[r] == s[t]) & (s[r] != 0))
            opp += np.mean((s[r] == -s[t]) & (s[r] != 0))
            n += 1
    same, opp = same / n, opp / n
    return 0.5 * (1 + same - opp)


def test_cross_moment_estimate_and_loo(units):
    B, P = units
    G = P - B
    for X, Y in ((B, B), (G, B), (G, G)):
        cm = iv.CrossMoment(X, Y)
        assert cm.estimate() == pytest.approx(cross_moment(X, Y))
        loo = cm.loo()
        for r in range(len(X)):
            keep = np.delete(np.arange(len(X)), r)
            assert loo[r] == pytest.approx(cross_moment(X[keep], Y[keep]))


def test_cross_moment_bootstrap_naive_and_distinct(units):
    B, P = units
    G = P - B
    rng = np.random.default_rng(3)
    m = iv.multiplicities(rng, len(B), 6)
    for X, Y in ((B, B), (G, B), (G, G)):
        cm = iv.CrossMoment(X, Y)
        naive, distinct = cm.boot(m, distinct=False), cm.boot(m, distinct=True)
        for b in range(len(m)):
            assert naive[b] == pytest.approx(cross_moment(_resample(X, m[b]), _resample(Y, m[b])))
            assert distinct[b] == pytest.approx(_distinct_cross(X, Y, m[b]))
    # No duplicates: both versions equal the estimate.
    ones = np.ones((1, len(B)))
    cm = iv.CrossMoment(G, B)
    assert cm.boot(ones, True)[0] == pytest.approx(cm.estimate())
    assert cm.boot(ones, False)[0] == pytest.approx(cm.estimate())


def test_cross_risk_form(units):
    B, P = units
    Z = P - B
    f = Z.mean(0) + 0.5
    E = Z - f
    cm = iv.CrossMoment(E, E, center=False, denom_models=Z.shape[1])
    assert cm.estimate() == pytest.approx(cross_risk(Z, f))
    rng = np.random.default_rng(5)
    m = iv.multiplicities(rng, len(Z), 4)
    d = cm.boot(m, distinct=True)
    for b in range(len(m)):
        assert d[b] == pytest.approx(_distinct_cross(E, E, m[b], center=False, dm=Z.shape[1]))


def test_agreement_estimate_loo_and_bootstrap(units):
    B, P = units
    G = np.round(P - B)          # rounding creates ties
    ag = iv.Agreement(G)
    ref = pair_agreement(G)
    parts = ag.parts()
    for k in ("same", "opposite", "tie", "C"):
        assert parts[k] == pytest.approx(ref[k])
    loo = ag.loo()
    for r in range(len(G)):
        assert loo[r] == pytest.approx(pair_agreement(np.delete(G, r, axis=0))["C"])
    rng = np.random.default_rng(9)
    m = iv.multiplicities(rng, len(G), 5)
    naive, distinct = ag.boot(m, False), ag.boot(m, True)
    for b in range(len(m)):
        assert naive[b] == pytest.approx(pair_agreement(_resample(G, m[b]))["C"])
        assert distinct[b] == pytest.approx(_distinct_agreement(G, m[b]))


def test_anova_matches_cross_moment(units):
    B, P = units
    G = P - B
    ms_a, ms_e, n1, n2 = iv.anova_mean_squares(G)
    assert (ms_a - ms_e) / len(G) == pytest.approx(cross_moment(G, G))
    assert (n1, n2) == (G.shape[1] - 1, (G.shape[0] - 1) * (G.shape[1] - 1))


def test_mls_coverage_under_random_effects_normal():
    """Under its own assumptions (random model effects, iid Gaussian errors) the MLS interval
    should cover the variance component near the nominal rate."""
    rng = np.random.default_rng(2024)
    K, M, sa2, se2, reps = 8, 12, 1.0, 4.0, 3000
    hit = upper_hit = 0
    for _ in range(reps):
        a = rng.normal(0, math.sqrt(sa2), M)
        G = a[None, :] + rng.normal(0, math.sqrt(se2), (K, M)) + rng.normal(0, 2, (K, 1))
        lo, hi = iv.mls_vdelta(G)
        hit += lo <= sa2 <= hi
        upper_hit += sa2 <= iv.mls_vdelta_upper(G)
    assert 0.93 <= hit / reps <= 0.97
    assert 0.93 <= upper_hit / reps <= 0.97


def test_fieller_cases():
    # Precise denominator: bounded set close to the delta-method interval.
    loo_n = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    loo_d = np.array([2.0, 2.01, 1.99, 2.0, 2.02, 1.98])
    kind, a, b = iv.fieller(1.0, 2.0, loo_n, loo_d)
    assert kind == "bounded" and a < 0.5 < b
    # Denominator indistinguishable from zero: unbounded set.
    loo_d0 = np.array([0.5, -0.4, 0.3, -0.2, 0.1, 0.0])
    kind, a, b = iv.fieller(1.0, 0.05, loo_n, loo_d0)
    assert kind in ("whole", "exclusive")
    assert iv.fieller_covers("exclusive", -1.0, 1.0, 5.0)
    assert not iv.fieller_covers("exclusive", -1.0, 1.0, 0.0)
    assert iv.scale_fieller("bounded", 0.2, 0.4, -10.0) == ("bounded", -4.0, -2.0)


def test_jackknife_t_and_percentile():
    loo = np.array([1.0, 2.0, 3.0, 4.0])
    se = iv.jackknife_se(loo)
    assert se == pytest.approx(math.sqrt(3 / 4 * 5.0))
    lo, hi = iv.jack_t(2.5, loo)
    assert lo < 2.5 < hi
    assert np.isnan(iv.jack_t(2.5, np.array([1.0, np.nan, 2.0]))[0])
    lo, hi, frac = iv.percentile(np.array([np.nan, 1.0, 2.0, 3.0]))
    assert frac == pytest.approx(0.25) and lo >= 1.0 and hi <= 3.0


def test_coverage_lower_bound():
    assert iv.coverage_lower_bound(4750, 5000) == pytest.approx(0.94446, abs=2e-4)
    assert iv.coverage_lower_bound(0, 10) == 0.0


def test_bca_basic_properties():
    rng = np.random.default_rng(4)
    x = rng.normal(size=40)
    loo = (x.sum() - x) / 39
    m = iv.multiplicities(rng, 40, 1999)
    boot = m @ x / 40
    lo, hi = iv.bca(x.mean(), boot, loo)
    plo, phi, _ = iv.percentile(boot)
    # Symmetric statistic: BCa is close to the percentile interval.
    assert lo < x.mean() < hi
    assert abs(lo - plo) < 0.05 and abs(hi - phi) < 0.05
    assert lo < iv.bca_upper(x.mean(), boot, loo) < hi
    assert np.isnan(iv.bca(x.mean(), boot, np.r_[loo[:-1], np.nan])[0])


def test_bca_improves_on_percentile_for_skewed_variance():
    """Sample variance of exponential data (n = 30) is hard for every method; BCa should
    still cover more often than the plain percentile interval on the same resamples."""
    rng = np.random.default_rng(8)
    n, reps, true = 30, 400, 1.0
    hit_bca = hit_pct = 0
    for _ in range(reps):
        x = rng.exponential(size=n)
        est = x.var(ddof=1)
        loo = np.array([np.delete(x, i).var(ddof=1) for i in range(n)])
        m = iv.multiplicities(rng, n, 999)
        mu = m @ x / n
        boot = (m @ x ** 2 - n * mu ** 2) / (n - 1)
        lo, hi = iv.bca(est, boot, loo)
        hit_bca += lo <= true <= hi
        lo, hi, _ = iv.percentile(boot)
        hit_pct += lo <= true <= hi
    assert hit_bca > hit_pct


def test_coverage_upper_bound():
    assert iv.coverage_upper_bound(930, 1000) == pytest.approx(0.9437, abs=1e-3)
    assert iv.coverage_upper_bound(10, 10) == 1.0


def _distinct_cross_positions(X, Y, owner, center=True):
    """Distinct-pair cross moment over positions listed by original unit ids."""
    Xr, Yr = X[owner], Y[owner]
    if center:
        Xr = Xr - Xr.mean(1, keepdims=True)
        Yr = Yr - Yr.mean(1, keepdims=True)
    num, n = 0.0, 0
    for r, s in itertools.permutations(range(len(owner)), 2):
        if owner[r] != owner[s]:
            num += Xr[r] @ Yr[s]
            n += 1
    return num / (n * (X.shape[1] - 1))


def test_boot_loo_cross_moment_matches_brute_force(units):
    B, P = units
    G = P - B
    rng = np.random.default_rng(21)
    m = iv.multiplicities(rng, len(B), 4)
    cm = iv.CrossMoment(G, B)
    bl = cm.boot_loo(m)
    for b in range(len(m)):
        owner = list(np.repeat(np.arange(len(B)), m[b].astype(int)))
        for u in range(len(B)):
            if m[b, u] >= 1:
                o = owner.copy()
                o.remove(u)
                assert bl[b, u] == pytest.approx(_distinct_cross_positions(G, B, np.array(o)))


def test_boot_loo_agreement_matches_brute_force(units):
    B, P = units
    G = np.round(P - B)
    rng = np.random.default_rng(22)
    m = iv.multiplicities(rng, len(G), 4)
    ag = iv.Agreement(G)
    bl = ag.boot_loo(m)
    for b in range(len(m)):
        for u in range(len(G)):
            if m[b, u] >= 1:
                mm = m[b].copy()
                mm[u] -= 1
                assert bl[b, u] == pytest.approx(_distinct_agreement(G, mm))


def test_weighted_jackknife_se_reduces_to_plain(units):
    B, P = units
    G = P - B
    cm = iv.CrossMoment(G, G)
    ones = np.ones((1, len(G)))
    se = iv.weighted_jackknife_se(cm.boot_loo(ones), ones)[0]
    assert se == pytest.approx(iv.jackknife_se(cm.loo()))


def test_boot_t_interval_contains_estimate(units):
    B, P = units
    G = P - B
    rng = np.random.default_rng(23)
    m = iv.multiplicities(rng, len(G), 499)
    cm = iv.CrossMoment(G, G)
    lo, hi = iv.boot_t(cm.estimate(), cm.loo(), cm.boot(m, True), cm.boot_loo(m), m)
    assert lo < cm.estimate() < hi
    assert iv.boot_t_upper(cm.estimate(), cm.loo(), cm.boot(m, True), cm.boot_loo(m), m) > cm.estimate()
