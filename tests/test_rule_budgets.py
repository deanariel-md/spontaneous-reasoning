"""Tests for own gain against capability rules by budget (src/rule_budgets.py, scripts/run_rule_budgets.py).
Fabricated arrays only."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src import capability_checks as cc, dataset1 as d1, noise_anatomy as na, pilot_checks as pc, rule_budgets as rb
from src.estimators import cross_risk, paired_loss_difference

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_rule_budgets_t", ROOT / "scripts" / "run_rule_budgets.py")
rrb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrb)

BBH_STRATA = [np.arange(t * 54, (t + 1) * 54) for t in range(25)]


def test_draw_pilots_shapes():
    rng = np.random.default_rng(0)
    pos, items = rb.draw_pilots(50, 24, 4, rng, N=1319, n=100)
    assert pos.shape == (50, 4) and items.shape == (50, 100)
    assert all(len(set(r)) == 4 and r.min() >= 0 and r.max() < 24 and np.all(np.diff(r) > 0) for r in pos)
    assert all(len(set(r)) == 100 and r.min() >= 0 and r.max() < 1319 for r in items)
    pos, items = rb.draw_pilots(5, 4, 4, rng)
    assert items is None and (pos == np.arange(4)).all()
    with pytest.raises(ValueError):
        rb.draw_pilots(1, 4, 5, rng)
    with pytest.raises(ValueError):
        rb.draw_pilots(1, 24, 1, rng, N=1350, n=101, strata=BBH_STRATA)
    U = rb.subsample_weights(1319, rb.draw_pilots(7, 24, 2, rng, N=1319, n=300)[1])
    assert U.shape == (1319, 7) and rb.subsample_weights(1319, None).shape == (1319, 1)
    S = (rng.random((24, 5, 1319)) < 0.5).astype(np.int8)
    assert na.subsample_scores(S, U).shape == (24, 5, 7)


def test_stratified_bbh_sampler_keeps_equal_task_weights():
    rng = np.random.default_rng(1)
    task = np.repeat(np.arange(25), 54)
    for n, k in ((100, 4), (300, 12)):
        _, items = rb.draw_pilots(30, 24, 2, rng, N=1350, n=n, strata=BBH_STRATA)
        assert items.shape == (30, n)
        assert all((np.bincount(task[r], minlength=25) == k).all() for r in items)
        U = rb.subsample_weights(1350, items, strata=BBH_STRATA)
        per_task = np.stack([U[task == t].sum(0) for t in range(25)])
        assert per_task == pytest.approx(np.full((25, 30), 100 / 25))
        # the score is the mean over tasks of each task's subsample mean
        S = (rng.random((3, 4, 1350)) < 0.6).astype(np.int8)
        X = na.subsample_scores(S, U)
        want = np.stack([np.mean([100 * S[:, :, r[task[r] == t]].mean(2) for t in range(25)], axis=0)
                         for r in items], axis=2)
        assert X == pytest.approx(want)
    # equal task sizes: the full score's weights are uniform, and the simple-random-sample formula agrees
    assert rb.full_weights(1350, strata=BBH_STRATA) == pytest.approx(np.full(1350, 1 / 1350))
    assert rb.subsample_weights(1350, items, strata=BBH_STRATA) == pytest.approx(rb.subsample_weights(1350, items))
    # unequal strata still weigh equally, in subsamples and in the full score
    strata = [np.arange(0, 10), np.arange(10, 40), np.arange(40, 45)]
    items = np.stack([rb.stratified_items(strata, 3, rng) for _ in range(6)])
    for U in (rb.subsample_weights(45, items, strata=strata), rb.subsample_weights(45, None, strata=strata)):
        for s in strata:
            assert U[s].sum(0) == pytest.approx(100 / 3)
    with pytest.raises(ValueError):
        rb.subsample_weights(45, items[:, :8], strata=strata)


def test_simple_random_sample_uses_the_full_score_weights():
    rng = np.random.default_rng(2)
    v = rng.uniform(1, 3, 60)
    v /= v.sum()
    _, items = rb.draw_pilots(4, 24, 1, rng, N=60, n=20)
    assert rb.subsample_weights(60, items, v=v) == pytest.approx(na.subsample_matrix(60, items, v))
    S = (rng.random((2, 3, 60)) < 0.5).astype(np.int8)
    assert na.subsample_scores(S, rb.subsample_weights(60, None, v=v))[:, :, 0] == pytest.approx(na.full_scores(S, v))


def test_rules_and_shrinkage_not_defined_at_one_draw():
    rng = np.random.default_rng(3)
    e = rng.normal(1500, 100, 9)
    B = rng.normal(50, 15, (1, 9))
    P = B + rng.normal(5, 3, (1, 9))
    fit = rb.fit_pilot(B, P, e)
    assert fit["own_gain_shrunk"] is None and fit["factor"] is None and math.isnan(fit["R"])
    assert fit["own_gain"] == pytest.approx((P - B)[0])
    assert fit["averaged_ols"] == pytest.approx(d1.fit_predictors(B, P)["averaged_ols"][0])
    assert fit[cc.FULL_RULE] == pytest.approx(cc.fit_rule(B, P, e, 2, True)["prediction"])
    B = rng.normal(50, 15, 9) + rng.normal(0, 2, (4, 9))
    P = B + rng.normal(5, 3, 9) + rng.normal(0, 1, (4, 9))
    fit = rb.fit_pilot(B, P)
    g, f = (P - B).mean(0), pc.shrink_factor(P - B)
    assert fit["factor"] == pytest.approx(f) and 0 <= f <= 1
    assert fit["own_gain_shrunk"] == pytest.approx(g.mean() + f * (g - g.mean()))
    assert fit["own_gain_shrunk"].mean() == pytest.approx(g.mean())
    assert cc.FULL_RULE not in fit
    assert rb.unclipped_factor(fit["R"], 4) == pytest.approx(4 * fit["R"] / (1 + 3 * fit["R"]))
    assert math.isnan(rb.unclipped_factor(0.5, 1))


def test_risk_difference_sign_and_interval():
    rng = np.random.default_rng(4)
    delta = rng.normal(5, 6, 8)
    Z = delta + rng.normal(0, 2, (24, 8))
    own = delta + rng.normal(0, 0.5, (5, 8))
    cap = delta.mean() + rng.normal(0, 1, (5, 8))
    L = rb.loss_differences(Z, cap, own)
    assert L.shape == (5, 24)
    for p in range(5):
        assert L[p] == pytest.approx(paired_loss_difference(Z, cap[p], own[p]))
    d = rb.averaged_difference(L)
    assert d["estimate"] > 0                         # capability minus own gain: positive when own gain predicts better
    m = L.mean(0)
    se = m.std(ddof=1) / math.sqrt(24)
    q = stats.t.ppf(0.975, 23)
    assert d["df"] == 23 and d["se"] == pytest.approx(se)
    assert d["ci95"] == pytest.approx([m.mean() - q * se, m.mean() + q * se])
    assert rb.averaged_difference(rb.loss_differences(Z, own, cap))["estimate"] == pytest.approx(-d["estimate"])
    assert d["share_pilots_positive"] == pytest.approx(np.mean(L.mean(1) > 0))
    assert rb.cross_risks(Z, own) == pytest.approx([cross_risk(Z, f) for f in own])
    # the averaged difference is the difference of the averaged cross-product risks
    assert d["estimate"] == pytest.approx(rb.cross_risks(Z, cap).mean() - rb.cross_risks(Z, own).mean())


def test_freeze_keeps_identical_arrays_and_refuses_others(tmp_path):
    j, z = tmp_path / "p.json", tmp_path / "p.npz"
    arrays = {"a__own_gain": np.arange(6.0).reshape(2, 3), "a__factor": np.array([np.nan, 0.5])}
    rec = rb.freeze({"design": {"x": 1}}, arrays, j, z)
    assert rec["npz_sha256"] == rb.file_sha256(z) and rec["content_sha256"] == rb.arrays_sha256(arrays)
    again = rb.freeze({"design": {"x": 1}, "created": "later"}, {k: v.copy() for k, v in arrays.items()}, j, z)
    assert again["created"] == rec["created"] and again["npz_sha256"] == rec["npz_sha256"]
    with pytest.raises(FileExistsError):
        rb.freeze({"design": {"x": 1}}, {**arrays, "a__own_gain": arrays["a__own_gain"] + 1}, j, z)
    got, back = rb.load_frozen(j, z)
    assert np.array_equal(back["a__own_gain"], arrays["a__own_gain"])
    z.write_bytes(z.read_bytes() + b"x")
    with pytest.raises(ValueError):
        rb.load_frozen(j, z)


def fabricated(json_path, npz_path, calls, seed=5):
    """A Dataset 2-like setting (item scores, two strata, an index) and a DOVE benchmark-like setting (4 pilot draws
    of full-item scores). Each validation loader checks that the frozen pair exists and agrees when it is called."""
    rng = np.random.default_rng(seed)
    M, N, nd = 7, 40, 8
    ability = rng.uniform(0.2, 0.7, M)
    gain = rng.uniform(0.0, 0.25, M)
    SB = (rng.random((nd, M, N)) < ability[None, :, None]).astype(np.int8)
    SP = (rng.random((nd, M, N)) < (ability + gain)[None, :, None]).astype(np.int8)
    Zd = 100 * gain + rng.normal(0, 3, (24, M))
    Bt = rng.normal(50, 10, (4, 5))
    Pt = Bt + rng.normal(3, 2, 5) + rng.normal(0, 1, (4, 5))
    Zt = rng.normal(3, 2, (24, 5))

    def loader(name, Z):
        def validation():
            rec = json.load(open(json_path, encoding="utf-8"))
            assert rec["npz_sha256"] == rb.file_sha256(npz_path)
            assert any(c["setting"] == name for c in rec["design"]["cells"])
            calls.append(name)
            return Z
        return validation

    d2like = {"setting": "fab", "title": "Fabricated", "group": "d2", "models": [f"m{i}" for i in range(M)],
              "SB": SB, "SP": SP, "v": None, "strata": [np.arange(0, 20), np.arange(20, 40)], "n_items": N,
              "item_budgets": (10, None), "eci": rng.normal(1500, 100, M).tolist(),
              "configs": [f"c{i % 5}" for i in range(nd)], "validation": loader("fab", Zd),
              "item_sampling": "5 items from each of 2 strata", "draws": "fabricated"}
    dovelike = {"setting": "dove:fab", "title": "fab benchmark", "group": "dove_tasks",
                "models": [f"d{i}" for i in range(5)], "Bd": Bt, "Pd": Pt, "n_items": 100, "item_budgets": (None,),
                "eci": None, "configs": [3, 1, 1, 2], "validation": loader("dove:fab", Zt),
                "item_sampling": "all items only", "draws": "fabricated"}
    return [d2like, dovelike]


def test_predictions_frozen_with_npz_hash_before_validation_arrays(tmp_path):
    j, z = tmp_path / "predictions_rule_budgets.json", tmp_path / "predictions_rule_budgets.npz"
    calls = []
    settings = fabricated(j, z, calls)
    rec, Z, rows = rrb.compute(settings, json_path=j, npz_path=z, n_pilots=12)
    assert calls == ["fab", "dove:fab"]
    assert rec["npz_sha256"] == rb.file_sha256(z)
    cells = rec["design"]["cells"]
    assert len(cells) == 3 * 2 + 3 * 1
    assert [c["stream"] for c in cells][:2] == [["rule-budgets", "fab", 1, 10], ["rule-budgets", "fab", 1, "all"]]
    _, arrays = rb.load_frozen(j, z)
    key = rrb.cell_key("fab", 2, 10)
    assert arrays[key + "positions"].shape == (12, 2) and arrays[key + "items"].shape == (12, 10)
    assert arrays[key + cc.FULL_RULE].shape == (12, 7)
    assert rrb.cell_key("dove:fab", 1, None) + "items" not in arrays
    # shrinkage is not defined at K = 1 and reported so
    one = [r for r in rows if r["K"] == 1 and r["g"] == "own_gain_shrunk" and r["status"] != rrb.NOT_APPLICABLE]
    assert len(one) == 2 * 2 + 1 and all(r["status"] == rrb.NOT_DEFINED and r["estimate"] is None for r in one)
    assert np.isnan(arrays[rrb.cell_key("fab", 1, 10) + "own_gain_shrunk"]).all()
    # the DOVE benchmark has the full item budget only; the 100 and 300 rows are not applicable
    na_rows = [r for r in rows if r["setting"] == "dove:fab" and r["items"] in (100, 300)]
    assert len(na_rows) == 3 * 2 * 2 and all(r["status"] == rrb.NOT_APPLICABLE for r in na_rows)
    four = [r for r in rows if r["setting"] == "dove:fab" and r["K"] == 4 and r["items"] == "all"]
    assert all(r["distinct_pilots"] == 1 for r in four)
    two = next(r for r in rows if r["setting"] == "dove:fab" and r["K"] == 2 and r["items"] == "all")
    assert two["distinct_pilots"] <= 4                 # configurations 3, 1, 1, 2: four distinct pairs
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all(r["ci95"] is not None and r["df"] == 23 for r in ok)
    assert {(r["f"], r["g"]) for r in rows if r["setting"] == "fab"} == set(rb.COMPARISONS)
    # the scores come from the frozen file: a rerun with the same inputs keeps it, other inputs are refused
    calls.clear()
    rec2, _, rows2 = rrb.compute(fabricated(j, z, calls), json_path=j, npz_path=z, n_pilots=12)
    assert rec2["created"] == rec["created"] and rows2[0]["estimate"] == rows[0]["estimate"]
    with pytest.raises(FileExistsError):
        rrb.compute(fabricated(j, z, [], seed=6), json_path=j, npz_path=z, n_pilots=12)
