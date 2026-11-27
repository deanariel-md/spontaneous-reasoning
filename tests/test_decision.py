"""Decision analysis: src/decision.py and scripts/run_decision.py."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import decision as dc  # noqa: E402
import run_decision as rd  # noqa: E402


# --------------------------------------------------------------------------- choosing


def test_third_size():
    assert [dc.third_size(m) for m in (29, 28, 5, 3, 2)] == [10, 9, 2, 1, 1]
    with pytest.raises(ValueError):
        dc.third_size(1)


def test_choose_clean_cut():
    w, t = dc.choose([5.0, 1.0, 3.0, 4.0, 2.0], 2)
    assert w.tolist() == [1, 0, 0, 1, 0] and t == 0


def test_choose_ties_share_places():
    w, t = dc.choose([5.0, 3.0, 3.0, 3.0, 1.0], 2)
    assert np.isclose(w.sum(), 2) and w[0] == 1 and np.allclose(w[1:4], 1 / 3) and w[4] == 0 and t == 3
    w, t = dc.choose(np.full(6, 2.0), 2)                 # an undefined slope ties every model
    assert np.allclose(w, 1 / 3) and t == 6
    w, t = dc.choose([5.0, 3.0, 3.0 + 1e-12, 1.0], 3)    # tie within tolerance, both fit: no tie at the cut
    assert w.tolist() == [1, 1, 1, 0] and t == 0


def test_choose_refuses_nan():
    with pytest.raises(ValueError):
        dc.choose([1.0, np.nan, 2.0], 1)


def test_label_seed_matches_d20_convention():
    import hashlib
    assert dc.label_seed("decision/gsm8k/K2") == int.from_bytes(hashlib.sha256(b"decision/gsm8k/K2").digest()[:8], "big")


def test_pilot_sets_distinct_and_seeded():
    units = ["a", "b", "a", "c", "d", "b", "e"]
    s = dc.pilot_sets(units, "t", sizes=(1, 2, 4), n_sets=50)
    assert s[1] == [[0], [1], [3], [4], [6]]
    for k in (2, 4):
        assert len(s[k]) == 50
        for rows in s[k]:
            assert len(set(units[r] for r in rows)) == k and rows == sorted(rows)
    assert s == dc.pilot_sets(units, "t", sizes=(1, 2, 4), n_sets=50)
    assert s[2] != dc.pilot_sets(units, "u", sizes=(2,), n_sets=50)[2]
    with pytest.raises(ValueError):
        dc.pilot_sets(["a", "a", "b"], "t", sizes=(4,))


def test_capability_one_draw_direction():
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    g = 40 - 0.5 * b
    pred, s = dc.capability_one_draw(b, g)
    assert s < 0 and dc.choose(pred, 2)[0].tolist() == [1, 1, 0, 0, 0, 0]
    pred, s = dc.capability_one_draw(b, 0.5 * b)
    assert s > 0 and dc.choose(pred, 2)[0].tolist() == [0, 0, 0, 0, 1, 1]
    pred, s = dc.capability_one_draw(np.full(6, 3.0), g)
    assert not np.isfinite(s) and dc.choose(pred, 2)[1] == 6


def test_rule_predictions_shapes_and_own_gain():
    rng = np.random.default_rng(0)
    K, M = 8, 9
    B = rng.normal(50, 10, (K, M))
    P = B + rng.normal(10, 5, M) + rng.normal(0, 2, (K, M))
    sets = dc.pilot_sets([str(i) for i in range(K)], "x", n_sets=7)
    pr = dc.rule_predictions(B, P, sets)
    assert set(pr) == {"own_1", "capability_1", "own_2", "shrunk_2", "own_4", "shrunk_4"}
    assert len(pr["own_1"]) == K and len(pr["own_4"]) == 7
    rows = pr["own_4"][3][1]["rows"]
    assert np.allclose(pr["own_4"][3][0], (P - B)[rows].mean(0))
    ref = dc.reference_predictions(B, P, list(range(K)))
    assert np.allclose(ref["own_all"][0][0], (P - B).mean(0))


def test_shrunken_differs_from_scalar_shrinkage_only_through_capability():
    # The scalar shrinkage keeps the order of own gain; Appendix A's predictor can change it.
    rng = np.random.default_rng(3)
    B = rng.normal(50, 15, (4, 12))
    P = B + 20 - 0.3 * B.mean(0) + rng.normal(0, 4, (4, 12))
    G = P - B
    own = G.mean(0)
    factor = 0.6
    scalar = own.mean() + factor * (own - own.mean())
    assert np.array_equal(dc.choose(own, 4)[0], dc.choose(scalar, 4)[0])


# --------------------------------------------------------------------------- regret


def test_regret_plugin_and_values():
    Gv = np.array([[10.0, 0.0, 5.0], [12.0, 2.0, 5.0]])
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(dc.regret_plugin(W, Gv, 1), [10.0, 0.0])
    assert np.allclose(dc.values(W, Gv, 1), [[0, 2], [10, 12]])
    assert dc.same_as_best(W, Gv, 1) == 0.5


def test_regret_crossfit_can_be_negative_and_is_below_plugin_in_expectation():
    rng = np.random.default_rng(1)
    M, K, n = 12, 10, 4
    truth = np.linspace(0, 6, M)
    W = np.zeros((1, M))
    W[0, -n:] = 1                                         # the true best third
    rp, rc = [], []
    for _ in range(400):
        Gv = truth + rng.normal(0, 4, (K, M))
        rp.append(dc.regret_plugin(W, Gv, n)[0])
        rc.append(dc.regret_crossfit(W, Gv, n)[0])
    assert np.mean(rp) > 0 > np.mean(rc)                  # the bracket: plug-in above, cross-fitted below the truth (0)
    assert min(rc) < 0


def test_regret_against_known_truth():
    W = np.array([[1.0, 1.0, 0.0, 0.0]])
    assert np.isclose(dc.regret_against(W, [1.0, 2.0, 3.0, 4.0], 2)[0], 2.0)


def test_paired_difference_sign_and_cancel():
    rng = np.random.default_rng(2)
    Gv = rng.normal(0, 1, (6, 5)) + np.arange(5)
    good = np.array([[0, 0, 0, 1, 1.0]])
    bad = np.array([[1, 1, 0, 0, 0.0]])
    d = dc.paired_difference(bad, good, Gv, 2)
    assert np.isclose(d["estimate"], dc.regret_plugin(bad, Gv, 2)[0] - dc.regret_plugin(good, Gv, 2)[0])
    assert d["estimate"] > 0 and d["ci95"][0] < d["estimate"] < d["ci95"][1] and d["K"] == 6


def test_summarize_keys():
    Gv = np.random.default_rng(4).normal(0, 1, (6, 6))
    s = dc.summarize(np.eye(6)[:2], Gv, 1, g_true=np.arange(6.0))
    assert {"regret_plugin", "regret_crossfit", "regret_plugin_p10", "regret_plugin_p90", "share_best_third",
            "value", "regret_exact", "n_sets"} <= set(s)


# --------------------------------------------------------------------------- the script


def test_settings_and_cross_protocol_pairs():
    assert {"gsm8k_req", "bbh_req", "math_req", "exemplar", "dove_mmlu"} <= set(rd.GROUPS["main"])
    assert set(rd.GROUPS["fewshot"]) == {"fewshot_gsm8k", "fewshot_bbh", "fewshot_math"}
    for src, dst in rd.CROSS:
        assert src.endswith("_req") and rd.SETTINGS[src][0] == rd.SETTINGS[dst][0]
        assert rd.SETTINGS[src][1] in ("strict", "table") and rd.SETTINGS[src][2] == "requested"
    for t in ("gsm8k", "bbh", "math"):          # pilot seeds by task: both protocols use the same sets
        assert rd.SETTINGS[f"{t}_req"][0] == rd.SETTINGS[f"{t}_fd"][0] == rd.SETTINGS[f"{t}_con"][0]


def test_added_task_settings_come_from_the_configuration_and_run_last():
    """HotpotQA: three settings from config/added_contrasts.yaml, in a group of their own after the
    others, reusing the choices its own run saved (file label), with the requested -> forced pair added at the end of
    the cross-protocol list."""
    assert rd.GROUPS["added"] == ["hotpot_req", "hotpot_fd", "fewshot_hotpot"]
    assert not set(rd.GROUPS["added"]) & set(rd.GROUPS["main"] + rd.GROUPS["fewshot"])
    assert rd.SETTINGS["hotpot_req"][:4] == ("hotpot", "table", "requested", "cot")
    assert rd.SETTINGS["fewshot_hotpot"][:4] == ("hotpot", "table", "forced", "five_shot")
    assert rd.CHOICES == {"hotpot_req": "hotpot", "hotpot_fd": "hotpot_forced", "fewshot_hotpot": "hotpot_fewshot"}
    assert rd.choice_path("hotpot_fd", "D").replace("\\", "/").endswith("D/predictions_decision_hotpot_forced.json")
    assert rd.choice_path("gsm8k_req", "D").replace("\\", "/").endswith("D/predictions_decision_gsm8k_req.json")
    assert rd.CROSS[-1] == ("hotpot_req", "hotpot_fd")
    assert list(rd.SETTINGS)[-3:] == rd.GROUPS["added"]


def test_paired_table_puts_the_added_tasks_rows_last():
    blk = {"evaluation": {"independent": {"paired": [{"a": "capability_1", "b": "own_1", "estimate": 1.0,
                                                      "ci95": [0.0, 2.0], "K": 3}]}}}
    cross = lambda c, d: {"choice": c, "evaluation_setting": d, "evaluation": {"independent": {"rules": {"own_1": {
        "extra_regret": {"estimate": 0.5, "ci95": None, "K": 3}}}}}}
    R = {"settings": {"hotpot_req": {"result": blk}, "gsm8k_req": {"result": blk}, "gsm8k_fd": {"result": blk}},
         "cross_protocol": [cross("hotpot_req", "hotpot_fd"), cross("gsm8k_req", "gsm8k_fd")]}
    t = rd.paired_table(R)
    assert t["setting"].tolist() == ["gsm8k_req", "gsm8k_fd", "gsm8k_req->gsm8k_fd", "hotpot_req",
                                     "hotpot_req->hotpot_fd"]


def test_evaluation_rows():
    ev = rd.evaluation_rows(["a", "b", "a"], ["c", "a", "c", "d", "b", "e"])
    assert ev == {"independent": [0, 3, 5], "all_distinct": [0, 1, 3, 4, 5]}


def _synthetic(seed=5, M=9, K=12):
    rng = np.random.default_rng(seed)
    cap = rng.normal(50, 15, M)
    out = {}
    for split, units in (("discovery", [f"c{i % 10}" for i in range(K)]), ("validation", [f"c{i}" for i in range(5, 5 + K)])):
        B = cap + rng.normal(0, 3, (K, M))
        P = B + 10 + rng.normal(0, 6, M) * (split == "discovery") + rng.normal(0, 2, (K, M))
        out[split] = (units, [f"m{j}" for j in range(M)], B, P)
    return out


def test_run_freezes_before_reading_validation(tmp_path, monkeypatch):
    data = _synthetic()
    calls = []

    def fake_arrays(name, split):
        calls.append((name, split, sorted(os.listdir(tmp_path / "frozen")) if (tmp_path / "frozen").exists() else []))
        return data[split]
    monkeypatch.setattr(rd, "setting_arrays", fake_arrays)
    monkeypatch.setattr(rd, "available", lambda s: True)
    a = SimpleNamespace(settings=["gsm8k_req", "gsm8k_fd"], out=str(tmp_path / "out"), pred_dir=str(tmp_path / "frozen"),
                        overwrite_frozen=False, dry_run=False)
    R = rd.run(a)
    first_val = next(i for i, c in enumerate(calls) if c[1] == "validation")
    assert all(c[1] == "discovery" for c in calls[:first_val])
    assert {"predictions_decision_gsm8k_req.json", "predictions_decision_gsm8k_fd.json"} <= set(calls[first_val][2])
    assert set(R["settings"]) == {"gsm8k_req", "gsm8k_fd"} and len(R["cross_protocol"]) == 1
    for f in ("summary.md", "summary_clean.md", "out.json", "tables/regret.csv", "tables/paired.csv"):
        assert (tmp_path / "out" / f).exists()
    # a second run keeps the frozen choices; different choices are refused
    created = json.load(open(tmp_path / "frozen" / "predictions_decision_gsm8k_req.json"))["created"]
    rd.run(a)
    assert json.load(open(tmp_path / "frozen" / "predictions_decision_gsm8k_req.json"))["created"] == created
    data["discovery"] = _synthetic(seed=6)["discovery"]
    with pytest.raises(FileExistsError):
        rd.run(a)


def test_dry_run_refuses_main_settings(tmp_path):
    a = SimpleNamespace(settings=["gsm8k_req"], out=str(tmp_path), pred_dir=None, overwrite_frozen=False, dry_run=True)
    try:
        with pytest.raises(SystemExit):
            rd.run(a)
    finally:
        rd.DRY = False


FROZEN = [f for f in rd.GROUPS["main"] if os.path.exists(os.path.join(rd.MANIFESTS, f"predictions_decision_{f}.json"))]


@pytest.mark.parametrize("name", FROZEN)
def test_frozen_choices_load_and_match_their_record(name):
    rec, W = rd.load_choices(name)
    M, n = len(rec["models"]), rec["n_chosen"]
    assert n == dc.third_size(M)
    for r, w in W.items():
        assert w.shape[1] == M and np.allclose(w.sum(1), n), r
    assert len(W["own_1"]) == len(dc.distinct_rows(rec["discovery_units"]))
    assert len(W["own_4"]) == dc.N_PILOT_SETS


@pytest.mark.parametrize("name", [f for f in FROZEN if os.path.exists(os.path.join(rd.CACHE, f"{rd.SETTINGS[f][0]}_strict_discovery.npz"))
                                  and rd.SETTINGS[f][1] == "strict"])
def test_frozen_choices_reproduce_from_discovery_arrays(name):
    rec, arrays = rd.choice_record(name, *rd.setting_arrays(name, "discovery"))
    jp = os.path.join(rd.MANIFESTS, f"predictions_decision_{name}.json")
    old, old_arrays = rd.rbud.load_frozen(jp, jp[:-5] + ".npz")
    assert rd.rbud.arrays_sha256(arrays) == old["content_sha256"]
