import numpy as np
import pytest

from src import capability_checks as cc
from src.estimators import cross_moment, paired_loss_difference


def _panel(K=12, M=9, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(100, 10, M)
    b = rng.uniform(20, 90, M)
    return rng, e, b


def test_vdelta_draws_matches_appendix_a():
    rng = np.random.default_rng(1)
    G = rng.normal(0, 1, (10, 6)) + rng.normal(0, 3, 6)
    r = cc.vdelta_draws(G)
    assert r["estimate"] == pytest.approx(cross_moment(G, G))
    lo, hi = r["ci95"]
    assert lo < r["estimate"] < hi


def test_index_moments_reduce_to_panel_covariance():
    rng, e, _ = _panel()
    G = rng.normal(0, 1, (12, 9)) + 0.3 * e
    X = cc.tile(e, 12)
    assert cross_moment(X, X) == pytest.approx(np.var(e, ddof=1))
    assert cross_moment(G, X) == pytest.approx(np.cov(G.mean(0), e)[0, 1])


def test_explained_share_without_draw_noise_is_squared_correlation():
    rng, e, _ = _panel()
    delta = 0.5 * e + rng.normal(0, 4, 9)
    G = cc.tile(delta, 8) + rng.normal(0, 1, (8, 1))        # a common shift per draw cancels in the moments
    r = cc.explained_share(G, cc.tile(e, 8))
    assert r["estimate"] == pytest.approx(np.corrcoef(delta, e)[0, 1] ** 2)


def test_explained_share_undefined_without_spread():
    G = np.ones((5, 4))
    r = cc.explained_share(G, cc.tile([1.0, 2.0, 3.0, 4.0], 5))
    assert r["estimate"] is None and r["ci95"] is None


def test_partial_correlation_removes_the_index():
    rng, e, _ = _panel()
    r = rng.normal(0, 3, 9)
    Ga = cc.tile(2 * e + r, 10) + rng.normal(0, 1e-6, (10, 9))
    Gb = cc.tile(-e + r, 10) + rng.normal(0, 1e-6, (10, 9))
    out = cc.partial_correlation(Ga, Gb, e)
    assert out["pearson"]["estimate"] == pytest.approx(1.0, abs=1e-6)
    assert out["latent"]["estimate"] == pytest.approx(1.0, abs=1e-4)
    raw = np.corrcoef(Ga.mean(0), Gb.mean(0))[0, 1]
    assert raw < 0.5


def test_correlation_with_index():
    rng, e, _ = _panel()
    G = cc.tile(e, 6) + rng.normal(0, 5, (6, 9))
    r = cc.correlation_with_index(G, e)
    assert r["estimate"] == pytest.approx(np.corrcoef(G.mean(0), e)[0, 1])


def test_fit_rule_recovers_quadratic():
    rng, e, b = _panel()
    g = 3 + 0.2 * (b - b.mean()) - 0.01 * (b - b.mean()) ** 2
    B = cc.tile(b, 4)
    P = B + cc.tile(g, 4)
    f = cc.fit_rule(B, P, e, 2, False)
    assert np.allclose(f["prediction"], g)
    assert f["coefficients"] == pytest.approx([3, 0.2, -0.01])
    assert f["n_parameters"] == 3


def test_choose_rule_prefers_the_shape_in_the_data():
    rng, e, b = _panel(M=20)
    e = rng.normal(100, 10, 20)
    b = rng.uniform(5, 95, 20)
    g = 30 - 0.02 * (b - 50) ** 2
    K = 24
    B = cc.tile(b, K) + rng.normal(0, 1, (K, 20))
    P = B + cc.tile(g, K) + rng.normal(0, 1, (K, 20))
    ch = cc.choose_rule(B, P, e)
    assert ch["chosen"] in ("baseline_quadratic", "baseline_quadratic_index")
    assert ch["scores"]["constant"]["score"] > ch["scores"][ch["chosen"]]["score"]
    assert ch["scores"][ch["chosen"]]["score"] == min(s["score"] for s in ch["scores"].values())


def test_choose_rule_tie_goes_to_simpler(monkeypatch):
    score = {"constant": 2.0, "baseline_linear": 1.0, "index_linear": 1.0}
    monkeypatch.setattr(cc, "split_half_risk",
                        lambda B, P, e, d, i: {"score": score.get(next(n for n, dd, ii in cc.CANDIDATES
                                                                       if (dd, ii) == (d, i)), 1.0)})
    assert cc.choose_rule(None, None, None)["chosen"] == "baseline_linear"


def test_paired_difference_is_step3_mean():
    rng = np.random.default_rng(3)
    Z = rng.normal(0, 2, (8, 5))
    f, g = np.zeros(5), Z.mean(0)
    d = cc.paired_difference(Z, f, g)
    assert d["estimate"] == pytest.approx(paired_loss_difference(Z, f, g).mean())
    assert d["ci95"][0] < d["estimate"] < d["ci95"][1]
