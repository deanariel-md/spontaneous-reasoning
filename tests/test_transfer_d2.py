"""Analysis T across tasks (src/transfer_d2.py)."""
import numpy as np
import pytest

from src import transfer_d2
from src.estimators import cross_risk
from src.tasks_d2 import centred


def test_ols_slope_and_no_spread():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert transfer_d2.ols_slope(x, 2 * x + 5) == pytest.approx(2.0)
    assert transfer_d2.ols_slope(np.ones(4), x) is None


def test_rules_shapes_and_definitions():
    rng = np.random.default_rng(0)
    K, M = 6, 5
    sg, sb, tg = rng.normal(size=(K, M)), rng.normal(size=(K, M)), rng.normal(size=(K, M))
    eci = [100, 110, 120, 130, 140]
    r = transfer_d2.rules(sg, sb, tg, eci)
    assert set(r) == {"constant", "transfer", "transfer_rescaled", "capability_eci", "capability_baseline",
                      "pilot_1", "pilot_2", "pilot_4", "own_24"}
    sd_s, sd_t = centred(sg.mean(0)).std(ddof=1), centred(tg.mean(0)).std(ddof=1)
    np.testing.assert_allclose(r["transfer_rescaled"]["prediction"], centred(sg.mean(0)) * sd_t / sd_s)
    assert np.std(r["transfer_rescaled"]["prediction"], ddof=1) == pytest.approx(sd_t)
    np.testing.assert_allclose(r["constant"]["prediction"], 0)
    np.testing.assert_allclose(r["transfer"]["prediction"], centred(sg.mean(0)))
    np.testing.assert_allclose(r["pilot_1"]["prediction"], centred(tg[0]))
    np.testing.assert_allclose(r["pilot_4"]["prediction"], centred(tg[:4].mean(0)))
    np.testing.assert_allclose(r["own_24"]["prediction"], centred(tg.mean(0)))
    for name in r:
        assert abs(np.sum(r[name]["prediction"])) < 1e-9          # every rule is centred
    b = sb.mean(0)
    s = transfer_d2.ols_slope(b, sg.mean(0))
    np.testing.assert_allclose(r["capability_baseline"]["prediction"], s * centred(b))


def test_score_signs_and_risks():
    rng = np.random.default_rng(1)
    M, Kv = 8, 20
    truth = centred(rng.normal(scale=10, size=M))
    Z = truth + rng.normal(scale=1, size=(Kv, M))
    preds = {"constant": np.zeros(M), "transfer": truth, "pilot_1": -truth}
    sc = transfer_d2.score(preds, Z, [("constant", "transfer"), ("transfer", "pilot_1")])
    d = {(x["f"], x["g"]): x for x in sc["differences"]}
    assert d[("constant", "transfer")]["estimate"] > 0           # the truth beats zero
    assert d[("transfer", "pilot_1")]["estimate"] < 0            # the reversed ordering is worse
    assert sc["risk_pp2"]["transfer"] == pytest.approx(cross_risk(centred(Z), truth))
    assert sc["differences"][0]["ci95"][0] < sc["differences"][0]["estimate"] < sc["differences"][0]["ci95"][1]


def test_rescaled_without_source_spread_is_reported():
    tg = np.random.default_rng(3).normal(size=(4, 5))
    r = transfer_d2.rules(np.ones((4, 5)), np.ones((4, 5)), tg, [1, 2, 3, 4, 5])
    assert r["transfer_rescaled"]["prediction"] is None


def test_cross_task_correlation_recovers_latent():
    rng = np.random.default_rng(4)
    K, M = 40, 30
    a = rng.normal(size=M)
    b = 0.6 * a + 0.8 * rng.normal(size=M)                       # true correlation about 0.6 at this M
    G = 25 * a + rng.normal(scale=3, size=(K, M))
    B = 10 * b + rng.normal(scale=3, size=(K, M))
    c = transfer_d2.cross_task_correlation(G, B)
    true = np.corrcoef(a, b)[0, 1]
    assert c["pearson"]["estimate"] == pytest.approx(np.corrcoef(G.mean(0), B.mean(0))[0, 1])
    assert c["latent"]["estimate"] == pytest.approx(true, abs=0.05)
    assert c["latent"]["ci95"][0] < c["latent"]["estimate"] < c["latent"]["ci95"][1]
    assert c["sd_mean_gain_a"]["estimate"] == pytest.approx(np.std(G.mean(0), ddof=1))
    with pytest.raises(ValueError):
        transfer_d2.cross_task_correlation(G, B[:-1])


def test_cross_task_latent_not_bounded():
    # two draws of pure noise give V_delta near zero; whatever the sign, the estimate is reported unclipped
    rng = np.random.default_rng(5)
    G, B = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
    c = transfer_d2.cross_task_correlation(G, B)
    assert set(c) >= {"pearson", "spearman", "latent", "V_delta_a", "V_delta_b", "cross_moment"}
    if c["V_delta_a"]["estimate"] <= 0 or c["V_delta_b"]["estimate"] <= 0:
        assert c["latent"]["estimate"] is None


def test_profile_share_limits():
    assert transfer_d2.profile_share(600, 130, 0.0) == pytest.approx(0.5)
    sg, sb = np.sqrt(600), np.sqrt(130)
    assert transfer_d2.profile_share(600, 130, 1.0) == pytest.approx((sg + sb) ** 2 / (2 * (600 + 130)))
    assert transfer_d2.profile_share(100, 100, 1.0) == pytest.approx(1.0)
    assert transfer_d2.profile_share(0, 100, 0.5) is None


def test_missing_prediction_is_reported():
    Z = np.random.default_rng(2).normal(size=(5, 4))
    sc = transfer_d2.score({"constant": np.zeros(4), "capability_eci": None}, Z, [("constant", "capability_eci")])
    assert sc["risk_pp2"]["capability_eci"] is None
    assert sc["differences"][0]["estimate"] is None
