"""Reliability map: agreement, references, pilot test, draws and the order of
the Dataset 2 run, on small fabricated data with known answers."""
import importlib.util
import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import dataset1 as d1, reliability_map as rm
from src.estimators import gaussian_agreement, pair_agreement, reliability
from src.sampling import rng_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("run_reliability_map_cli",
                                                  os.path.join(ROOT, "scripts", "run_reliability_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- agreement


def test_agreement_known_values_between_distinct_draws_and_exact():
    X = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])        # pair signs -, -, +
    a = rm.agreement(X)
    assert a["same"] == pytest.approx(2 / 6) and a["opposite"] == pytest.approx(4 / 6) and a["tie"] == 0.0
    assert a["C"] == pytest.approx(1 / 3) and a["decisive"] == pytest.approx(1.0)
    e = rm.agreement(X, exact=True)                             # with replacement: (1 + 4) / 9 same
    assert e["same"] == pytest.approx(5 / 9) and e["opposite"] == pytest.approx(4 / 9)


def test_agreement_matches_appendix_a_and_dataset1_on_integer_scores():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 6, size=(9, 5)).astype(float)          # many exact ties
    a, ref = rm.agreement(X), pair_agreement(X)
    for k in ("same", "opposite", "tie", "C"):
        assert a[k] == pytest.approx(ref[k], abs=1e-12)
    e, ref = rm.agreement(X, exact=True), d1.exact_agreement(X)
    for k in ("same", "opposite", "tie", "C"):
        assert e[k] == pytest.approx(ref[k], abs=1e-12)


def test_exact_agreement_equals_enumeration_of_draw_pairs_with_replacement():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 4, size=(5, 4)).astype(float)
    i, j = np.triu_indices(4, 1)
    vals = []
    for r, s in itertools.product(range(5), repeat=2):
        s1, s2 = np.sign(X[r, i] - X[r, j]), np.sign(X[s, i] - X[s, j])
        vals.append(np.mean(np.where((s1 == 0) | (s2 == 0), 0.5, (s1 == s2).astype(float))))
    assert rm.agreement(X, exact=True)["C"] == pytest.approx(np.mean(vals), abs=1e-12)


def _hidden_tie():
    """Counts with equal gains whose float gains differ: 100 a/n - 100 b/n against 100 c/n - 100 d/n, a-b = c-d."""
    n = 1319
    for b, d, g in itertools.product(range(1, 60), range(1, 60), range(1, 30)):
        if b != d and (100 * (b + g) / n - 100 * b / n) != (100 * (d + g) / n - 100 * d / n):
            return n, b, d, g
    raise AssertionError("no floating-point example found")


def test_a_tie_hidden_by_floating_point_counts_as_a_tie():
    n, b, d, g = _hidden_tie()
    B = 100 * np.array([[b, d], [b, d]], float) / n
    P = 100 * np.array([[b + g, d + g], [b + g, d + g]], float) / n
    G = P - B
    assert G[0, 0] != G[0, 1]                                   # the raw difference is not zero
    assert pair_agreement(G)["tie"] == 1.0                      # the estimator counts it as a tie too
    a = rm.agreement(G)
    assert a["tie"] == 1.0 and a["C"] == 0.5 and math.isnan(a["conditional_agreement"])


def test_agreement_between_rows_matches_dataset1_per_row():
    rng = np.random.default_rng(3)
    A, Bm = rng.integers(0, 5, (6, 4)).astype(float), rng.integers(0, 5, (6, 4)).astype(float)
    got = rm.agreement_between(A, Bm)
    per = [d1.agreement_between(a, b) for a, b in zip(A, Bm)]
    for k in ("same", "opposite", "tie", "C"):
        assert got[k] == pytest.approx(np.mean([p[k] for p in per]), abs=1e-12)


# --------------------------------------------------------------------------- reference


def test_reference_curve_spearman_brown_and_undefined_r():
    assert rm.reference_c(0.0) == 0.5 and rm.reference_c(1.0) == 1.0
    assert rm.reference_c(0.5) == pytest.approx(0.5 + 1 / 6)
    assert rm.spearman_brown(0.5, 3) == pytest.approx(0.75) and rm.spearman_brown(0.4, 1) == pytest.approx(0.4)
    assert rm.reference_c(0.3, 4) == pytest.approx(gaussian_agreement(0.3, 4))
    for bad in (-0.2, float("nan"), None, 1.2, 1.0 + 1e-9):
        assert math.isnan(rm.reference_c(bad)) and math.isnan(rm.spearman_brown(bad, 2))
    assert rm.reference_c(1.0 + 2e-16) == 1.0 and rm.r_status(1.0 + 2e-16) == "in [0, 1]"   # residue only
    assert rm.r_status(1.2) == "above one" and rm.r_status(-0.2) == "negative estimate"


def test_draws_needed_reproduce_appendix_b_and_report_edges():
    assert rm.k_needed(0.30, 0.80) == 10 and rm.k_needed(0.30, 0.90) == 46
    assert rm.k_needed(0.0, 0.80) == math.inf and rm.k_needed(1.0, 0.90) == 1
    assert rm.k_needed(-0.05, 0.80) is None and rm.k_needed(float("nan"), 0.80) is None


def test_r_for_k_is_the_threshold_of_k_needed():
    rmap = _script()
    assert rmap.r_for_k(1, 0.8) == pytest.approx(math.sin(0.3 * math.pi))
    for k in (1, 2, 4, 8, 16):
        r = rmap.r_for_k(k, 0.8)
        assert rm.k_needed(r + 1e-9, 0.8) <= k
        assert rm.k_needed(r - 1e-4, 0.8) > k


def test_pairwise_reference_known_pairs():
    # contrast [0, 2]: Delta 1, sigma 1 (bank moments)
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    p = stats.norm.cdf(1.0)
    assert rm.pairwise_reference(X, ddof=0)["C_pairs"] == pytest.approx(p ** 2 + (1 - p) ** 2)
    assert rm.pairwise_reference(np.array([[-1.0, 0.0], [1.0, 0.0]]), ddof=0)["C_pairs"] == pytest.approx(0.5)
    assert rm.pairwise_reference(np.array([[3.0, 0.0], [3.0, 0.0]]), ddof=0)["C_pairs"] == 1.0   # no noise, Delta 3
    zero = rm.pairwise_reference(np.zeros((3, 2)), ddof=0)
    assert zero["C_pairs"] == 0.5 and math.isnan(zero["pair_noise_cv"])


def test_common_noise_reference_averages_to_the_arcsine_curve():
    """E over Delta ~ N(0, tau^2) of p^2 + (1 - p)^2 with p = Phi(|Delta| / sigma) is 1/2 + arcsin(R)/pi."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    for tau2, s2 in ((1.0, 3.0), (4.0, 1.0), (0.2, 5.0)):
        p = stats.norm.cdf(np.abs(x) * math.sqrt(tau2) / math.sqrt(s2))
        e = float(np.sum(w * (p ** 2 + (1 - p) ** 2)) / np.sum(w))
        assert e == pytest.approx(rm.reference_c(tau2 / (tau2 + s2)), abs=2e-4)


def test_common_noise_keeps_r_on_an_enumerated_bank():
    """Mean squared pair difference is 2 V and mean pair noise 2 W, so C_common sees the bank's R."""
    rng = np.random.default_rng(4)
    B = rng.normal(size=(40, 6)) + 2 * np.arange(6)
    G = rng.normal(size=(40, 6)) + np.arange(6)
    pw = rm.pairwise_reference(G, ddof=0)
    t = d1.exact_targets(B, B + G)
    assert pw["mean_sq_pair_difference"] == pytest.approx(2 * t["V_delta"])
    assert pw["mean_pair_noise"] == pytest.approx(2 * t["noise_W_gain"])


# --------------------------------------------------------------------------- one setting


def test_setting_point_exact_bank_matches_dataset1():
    rng = np.random.default_rng(5)
    B = rng.integers(20, 80, (30, 5)).astype(float)
    P = B + rng.integers(-5, 15, (30, 5))
    pt = rm.setting_point(B, P, exact=True)
    t = d1.exact_targets(B, P)
    assert pt["benefit"]["R"] == pytest.approx(t["R_benefit"]) and pt["accuracy"]["R"] == pytest.approx(t["R_accuracy"])
    assert pt["benefit"]["C"] == pytest.approx(d1.exact_agreement(P - B)["C"])
    assert pt["benefit"]["R_distinct_draws"] == pytest.approx(reliability(P - B)["R"])
    assert pt["benefit"]["departure"] == pytest.approx(pt["benefit"]["C"] - rm.reference_c(t["R_benefit"]))


def test_setting_point_distinct_draws_reports_negative_r_without_a_reference():
    B = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pt = rm.setting_point(B, B + 1.0)
    assert pt["accuracy"]["R"] == pytest.approx(-1.0) and pt["accuracy"]["R_status"] == "negative estimate"
    assert math.isnan(pt["accuracy"]["C_ref"]) and math.isnan(pt["accuracy"]["departure"])
    assert pt["benefit"]["R_status"].startswith("undefined")          # constant gains: zero within-draw variance


# --------------------------------------------------------------------------- pilot test


def test_pilot_prediction_from_a_negative_r_predicts_nothing():
    B = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    p = rm.pilot_prediction(B, 2 * B, [1, 2], [0.8, 0.9])
    acc = p["accuracy"]
    assert p["n_pilot"] == 2 and acc["R"] == pytest.approx(-1.0) and not acc["defined"]
    assert math.isnan(acc["C_one_draw"]) and all(math.isnan(v) for v in acc["C_groups"].values())
    assert acc["K_needed"] == {"0.80": None, "0.90": None}
    same = rm.pilot_prediction(np.tile([1.0, 5.0, 9.0], (2, 1)), np.tile([2.0, 9.0, 11.0], (2, 1)), [1, 4], [0.8])
    assert same["benefit"]["R"] == pytest.approx(1.0) and same["benefit"]["C_groups"]["4"] == 1.0
    assert same["benefit"]["K_needed"]["0.80"] == 1


def test_validation_agreement_one_draw_first_split_and_unsupported_k():
    rng = np.random.default_rng(6)
    B = rng.integers(0, 50, (8, 4)).astype(float)
    P = B + rng.integers(0, 10, (8, 4))
    v = rm.validation_agreement(B, P, [1, 2, 4, 8], 300, np.random.default_rng(0))
    assert v["not_supported"] == [8] and "8" not in v["groups"]["benefit"]
    assert v["groups"]["benefit"]["1"]["mean_over_splits"]["C"] == pytest.approx(pair_agreement(P - B)["C"])
    first = v["groups"]["accuracy"]["4"]["first_k_against_next_k"]
    assert first["C"] == pytest.approx(d1.agreement_between(B[:4].mean(0), B[4:8].mean(0))["C"])


def test_split_agreement_equals_the_mean_over_the_given_splits():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 6, (10, 4)).astype(float)
    perms = np.stack([rng.permutation(10) for _ in range(25)])
    got = rm.split_agreement(X, perms, 3)
    want = np.mean([d1.agreement_between(X[p[:3]].mean(0), X[p[3:6]].mean(0))["C"] for p in perms])
    assert got["C"] == pytest.approx(want, abs=1e-12)
    perms6 = np.stack([rng.permutation(6) for _ in range(5)])
    assert rm.split_agreement(np.tile([1.0, 2.0, 3.0], (6, 1)), perms6, 3)["C"] == 1.0


def test_bank_group_agreement_exact_at_one_and_close_to_enumeration_at_two():
    X = np.array([[0.0, 1.0, 3.0], [2.0, 1.0, 0.0], [1.0, 1.0, 2.0]])
    out = rm.bank_group_agreement({"x": X}, [1, 2], 20000, np.random.default_rng(8))
    assert out["x"]["1"]["C"] == pytest.approx(rm.agreement(X, exact=True)["C"]) and out["x"]["1"]["mc_se"] == 0.0
    vals = [rm.agreement_between(X[list(a)].mean(0), X[list(b)].mean(0))["C"]
            for a in itertools.product(range(3), repeat=2) for b in itertools.product(range(3), repeat=2)]
    exact = float(np.mean(vals))
    assert abs(out["x"]["2"]["C"] - exact) < 4 * out["x"]["2"]["mc_se"] + 1e-9


def test_prediction_rows_and_error_summary():
    pred = {"n_pilot": 2, "benefit": rm.predict_from_r(0.5, [1, 2], [0.8]),
            "accuracy": rm.predict_from_r(-0.1, [1, 2], [0.8])}
    obs = {"one_draw": {q: {"C": 0.6} for q in rm.QUANTITIES},
           "groups": {q: {"2": {"mean_over_splits": {"C": 0.7}, "first_k_against_next_k": {"C": 0.65}}}
                      for q in rm.QUANTITIES}}
    rows = rm.prediction_rows("s", "g", pred, obs)
    b = [r for r in rows if r["quantity"] == "benefit"]
    assert b[0]["error"] == pytest.approx(rm.reference_c(0.5) - 0.6)
    assert b[1]["K"] == 2 and b[1]["error"] == pytest.approx(rm.reference_c(0.5, 2) - 0.7)
    assert all(r["error"] is None and r["predicted"] is None for r in rows if r["quantity"] == "accuracy")
    s = rm.error_summary([0.1, -0.1, None, 0.3])
    assert s["n"] == 3 and s["n_undefined"] == 1 and s["mean"] == pytest.approx(0.1)
    assert s["mae"] == pytest.approx(0.5 / 3) and s["rmse"] == pytest.approx(math.sqrt(0.11 / 3)) and s["max_abs"] == 0.3
    assert rm.error_summary([None]) == {"n": 0, "n_undefined": 1}


def test_within_factor():
    assert rm.within_factor(10, 6, 2) and not rm.within_factor(13, 6, 2) and rm.within_factor(3, 6, 2)
    assert rm.within_factor(math.inf, math.inf, 2) and not rm.within_factor(math.inf, 5, 2)
    assert rm.within_factor(None, 5, 2) is False and rm.within_factor(5, None, 2) is None


def test_pilot_draws_follow_the_dove_rule_from_metadata_only():
    ids = list(range(100, 150))
    a = rm.draw_pilot(ids, 4, 24, "reliability_map/x", 20261101)
    assert a == rm.draw_pilot(ids, 4, 24, "reliability_map/x", 20261101)
    assert a["pilot_index"] != rm.draw_pilot(ids, 4, 24, "reliability_map/y", 20261101)["pilot_index"]
    rng = rng_for("registered_split", "reliability_map/x", seed=20261101)
    assert a["pilot_index"] == rng.integers(0, 50, 4).tolist() and a["validation_index"] == rng.integers(0, 50, 24).tolist()
    assert a["pilot_config_ids"] == [ids[k] for k in a["pilot_index"]] and len(a["validation_config_ids"]) == 24
    with pytest.raises(ValueError):
        rm.draw_pilot([1, 1, 2], 2, 2, "x", 1)


# --------------------------------------------------------------------------- Dataset 2 order


def _scored(panel, configs, items, rng):
    rows = []
    for m_i, mk in enumerate(panel):
        for arm, lift in (("direct_answer", 0.0), ("zero_shot_cot", 0.15 * (m_i + 1))):
            for c in configs:
                p = min(0.95, 0.2 + 0.2 * m_i + lift)
                for it in items:
                    rows.append({"model_key": mk, "arm": arm, "config_id": c, "item_id": it,
                                 "strict_correct": bool(rng.random() < p), "lenient_correct": False,
                                 "compliant": True})
    return pd.DataFrame(rows)


def test_dataset2_pilot_predictions_are_written_before_validation_draws_load(tmp_path):
    rmap = _script()
    rng = np.random.default_rng(9)
    panel, items = ["a", "b", "c"], list(range(20))
    disc, val = ["c1", "c2", "c3", "c4"], ["c5", "c6", "c7", "c8"]
    df = _scored(panel, disc + val, items, rng)
    pred_path = str(tmp_path / "predictions_reliability_map_sim.json")
    calls = []

    def load(ids):
        calls.append((list(ids), os.path.exists(pred_path)))
        return df[df["config_id"].isin(ids)]

    inp = {"panel": panel, "items": items, "load": load, "subsets": {"sim": ("Simulated", items)},
           "group_of": {"sim": "sim"},
           "sampler": {"discovery_config_ids": disc, "validation_config_ids": val, "sha256": "x", "n_bank": 8,
                       "label": "sim"}}
    cfg = {"group_sizes": [1, 2], "pilot_sizes": [2, 4], "agreement_targets": [0.8, 0.9], "seed": 1}
    rows, fz = rmap.run_d2(inp, "sim", cfg, 50, pred_path, False)
    assert calls[0] == (disc, False) and calls[1] == (val, True)      # discovery first; frozen before validation
    r = rows[0]
    assert r["validation"]["K_validation"] == 4 and r["M"] == 3 and r["n_items"] == 20
    assert set(r["pilot"]) == {"2", "4"} and r["point"]["n_draws"] == 4
    rows2, fz2 = rmap.run_d2(inp, "sim", cfg, 50, pred_path, False)  # same predictions: the file is kept
    assert fz2["sha256"] == fz["sha256"] and fz2["created"] == fz["created"]
    inp["sampler"]["discovery_config_ids"] = ["c1", "c3", "c2", "c4"]  # other pilot draws: refused
    with pytest.raises(FileExistsError):
        rmap.run_d2(inp, "sim", cfg, 50, pred_path, False)


def test_dataset2_task_subsets_give_one_setting_each(tmp_path):
    """The BIG-Bench Hard layout: the pooled item set and each task's items, from the same draws."""
    rmap = _script()
    rng = np.random.default_rng(10)
    panel, items = ["a", "b", "c", "d"], [f"t{k // 10}-{k}" for k in range(20)]
    disc, val = ["c1", "c2", "c3", "c4"], ["c5", "c6", "c7", "c8"]
    df = _scored(panel, disc + val, items, rng)
    subsets = {"bbh:pooled": ("pooled", items), "bbh:t0": ("t0", items[:10]), "bbh:t1": ("t1", items[10:])}
    inp = {"panel": panel, "items": items, "load": lambda ids: df[df["config_id"].isin(ids)], "subsets": subsets,
           "group_of": {"bbh:pooled": "bbh", "bbh:t0": "bbh_tasks", "bbh:t1": "bbh_tasks"},
           "sampler": {"discovery_config_ids": disc, "validation_config_ids": val, "sha256": "x", "n_bank": 8,
                       "label": "bbh"}}
    cfg = {"group_sizes": [1, 2], "pilot_sizes": [2, 4], "agreement_targets": [0.8], "seed": 1}
    rows, _ = rmap.run_d2(inp, "bbh", cfg, 20, str(tmp_path / "p.json"), False)
    by = {r["setting"]: r for r in rows}
    assert set(by) == set(subsets) and by["bbh:t0"]["n_items"] == 10 and by["bbh:pooled"]["group"] == "bbh"
    # equal task sizes: the pooled score is the mean of the task scores, draw by draw
    from src import dataset2 as d2
    cells = d2.cell_table(df[df["config_id"].isin(disc)], panel, disc, items)
    Bd, _ = d2.unit_arrays(cells, disc, panel)
    assert by["bbh:pooled"]["point"]["accuracy"]["R"] == pytest.approx(reliability(Bd)["R"])
