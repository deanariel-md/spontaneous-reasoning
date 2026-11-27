"""HotpotQA in the cross-cutting analyses: the section `tasks` and
the pools of config/added_contrasts.yaml, the task layer that reads an added task's scored table, and the added-task
paths of the protocol ceiling, the decision analyses, the transfer analysis, model resampling, the arm effect, the
reliability map, the map by panel, the low-R map and the noise anatomy, with the macro readers and the value-by-value
comparison of results files. The checks that matter most: a later pool or task leaves every earlier number as it was,
and its own numbers come from the same estimators or from the files its own run saved."""
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import added_contrasts as ac  # noqa: E402
import macros_added as ma  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_added_tasks_test", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tac():
    spec = importlib.util.spec_from_file_location("tac_helpers", os.path.join(ROOT, "tests", "test_added_contrasts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    return yaml.safe_load(open(ac.CONFIG, encoding="utf-8"))


# --------------------------------------------------------------------------- configuration


def test_repository_configuration_adds_hotpotqa_as_task_and_pool():
    con = ac.load_config()
    pools = ac.pools(con)
    assert list(pools) == ["base", "hotpot"]
    assert list(pools["hotpot"]) == ["hotpot_cot_req", "hotpot_cot_forced", "hotpot_fewshot"]
    assert [ac.setting_name(c) for c in pools["hotpot"].values()] == ["hotpot:cot_req", "hotpot:cot_forced",
                                                                    "hotpot:fewshot"]
    assert [c["tag"] for c in pools["hotpot"].values()] == ["HqReq", "HqFd", "HqFs"]
    assert all(c["noise_base"] == "hotpot" for c in pools["hotpot"].values())
    tk = ac.load_tasks()
    assert list(tk) == ["hotpot"]
    h = tk["hotpot"]
    assert h["roles"] == {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot",
                          "five_shot": "fewshot_forced"}
    assert h["scorings"] == {"strict": "strict_correct", "lenient": "lenient_correct", "f1": "strict_f1"}
    assert h["forced_cap"] == 64 and h["fair"]["arms"] == {"cot_req": "requested", "cot_forced": "forced",
                                                         "fewshot": "fewshot"}
    assert {d["choices"] for d in h["decision"].values()} == {"hotpot", "hotpot_forced", "hotpot_fewshot"}
    # the noise grid covers HotpotQA's 1,000 items; the steps files are its own run's
    grid = yaml.safe_load(open(os.path.join(ROOT, "config", "noise_anatomy.yaml"), encoding="utf-8"))["n_grid"]
    assert grid["hotpot"][-1] == 1000
    assert [c["steps"] for c in pools["hotpot"].values()] == ["results/hotpot/hotpot.json",
                                                           "results/hotpot/hotpot_forced.json",
                                                           "results/hotpot/hotpot_fewshot.json"]


def test_pools_keys_and_info():
    con = {"a": {"pool": "hotpot"}, "b": {}, "c": {"pool": "hotpot"}}
    assert list(ac.pools(con)) == ["base", "hotpot"] and list(ac.pools(con)["hotpot"]) == ["a", "c"]
    assert ac.pool_key("added_settings", "base") == "added_settings"
    assert ac.pool_key("added_settings", "hotpot") == "added_settings_hotpot"
    assert ac.pool_info("base")["tag"] == "" and ac.pool_info("hotpot")["tag"] == "Hq"
    with pytest.raises(ac.ConfigError, match="not listed"):
        ac.pool_info("d99")
    assert ac.setting_name({"group": "g", "task": "t"}) == "g:t"
    assert ac.setting_name({"group": "g", "task": "t", "setting": "x:y"}) == "x:y"


def _with(cfg, path, value):
    cfg = copy.deepcopy(cfg)
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return cfg


@pytest.mark.parametrize("path, value, message", [
    (("tasks", "gsm8k"), {"title": "x"}, "already a main task"),
    (("tasks", "hotpot", "tag"), "Hq1", "letters only"),
    (("tasks", "hotpot", "roles"), {"requested": "direct_answer"}, "not named"),
    (("tasks", "hotpot", "decision", "hotpot_req", "baseline"), "elsewhere", "known roles"),
    (("tasks", "hotpot", "fair"), {"evaluation": "x.json"}, "evaluation file and arms"),
    (("contrasts", "hotpot_cot_req", "baseline"), {"kind": "table", "arm": "eight_shot"}, "roles"),
    (("contrasts", "hotpot_cot_req", "baseline"), {"kind": "collection", "arm": "direct_answer"}, "kind table"),
    (("contrasts", "fewshot_gsm8k", "baseline"), {"kind": "table", "arm": "direct_answer"}, "section tasks"),
    (("contrasts", "hotpot_cot_req", "pool"), "d-22", "letters and digits"),
    (("contrasts", "hotpot_cot_forced", "setting"), "hotpot:cot_req", "also"),
])
def test_task_and_pool_errors_are_refused(path, value, message):
    with pytest.raises(ac.ConfigError, match=message):
        ac.check_config(_with(_cfg(), path, value))


def test_a_task_entry_needs_its_fields():
    cfg = _cfg()
    del cfg["tasks"]["hotpot"]["scored"]
    with pytest.raises(ac.ConfigError, match="missing"):
        ac.check_config(cfg)


# --------------------------------------------------------------------------- the task layer


def _task_fixture(tmp_path, monkeypatch, rng=None):
    rng = rng or np.random.default_rng(1)
    panel, items, configs = ["m1", "m2", "m3"], ["10", "11", "12", "13"], ["c1", "c2", "c3", "c4"]
    rows = []
    for m in panel:
        for arm in ("direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced"):
            for c in configs:
                for it in items:
                    v = int(rng.random() < 0.5)
                    rows.append({"model_key": m, "arm": arm, "config_id": c, "item_id": it, "strict_correct": v,
                                 "lenient_correct": v, "compliant": True, "strict_f1": float(v), "lenient_f1": float(v),
                                 "n_output_tokens": int(rng.integers(1, 100))})
    df = pd.DataFrame(rows)
    scored = tmp_path / "scored.parquet"
    df.to_parquet(scored)
    sampler = tmp_path / "sampler.json"
    sampler.write_text(json.dumps({"discovery_config_ids": configs[:2], "validation_config_ids": configs[2:],
                                   "sha256": "s", "n_bank": 4, "label": "toy"}))
    mod = type(sys)("toy_task_module")
    mod.panel_info = lambda: (panel, ["F1", "F1", "F2"], None)
    mod.item_ids = lambda: [int(i) for i in items]
    monkeypatch.setitem(sys.modules, "toy_task_module", mod)
    tasks = {"toy": {"title": "Toy", "tag": "Toy", "scored": str(scored), "sampler": str(sampler), "scorer": "x",
                     "module": "toy_task_module",
                     "roles": {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot",
                               "five_shot": "fewshot_forced"}}}
    return tasks, df, panel, items, configs


def test_task_collection_and_forced_inputs_read_the_scored_table(tmp_path, monkeypatch):
    tasks, df, panel, items, configs = _task_fixture(tmp_path, monkeypatch)
    inp, s, its, info = ac.task_collection_inputs("toy", tasks)
    assert inp["panel"] == panel and its == items and info is None and s["discovery_config_ids"] == ["c1", "c2"]
    got = inp["load"](["c1"])
    assert set(got["arm"]) == {"direct_answer", "zero_shot_cot"} and set(got["config_id"]) == {"c1"}
    assert got["item_id"].map(type).eq(str).all()
    fl, prov, box = ac.task_forced_inputs("toy", panel, tasks)
    f = fl(["c2", "c3"])
    assert set(f["arm"]) == {"forced_direct"} and set(f["config_id"]) == {"c2", "c3"} and box is None
    want = df[(df["arm"] == "forced_direct") & df["config_id"].isin(["c2", "c3"])]
    assert len(f) == len(want) and f["strict_correct"].sum() == want["strict_correct"].sum()
    five = ac.role_loader("toy", "five_shot", "zero_shot_cot", tasks)(["c4"])
    assert set(five["arm"]) == {"zero_shot_cot"} and len(five) == len(panel) * len(items)


def test_the_task_layer_dispatches_an_added_task():
    rfd = _script("run_forced_direct")
    assert rfd.added_task("hotpot") and not rfd.added_task("gsm8k") and not rfd.added_task("math")


def test_robustness_inputs_registers_the_added_tasks_apart():
    ri = _script("robustness_inputs")
    assert ri.TASKS == ("gsm8k", "bbh", "math")
    assert ri.added_tasks() == ("hotpot",)
    assert ri.SCORED["hotpot"] == ["data/processed/hotpot/scored.parquet"] and ri.NAMES["hotpot"] == "HotpotQA"
    assert ri.TASKS == ("gsm8k", "bbh", "math")


def _fair(tmp_path):
    rule = lambda v: {"regret_plugin": v, "regret_crossfit": v - 0.01, "plugin_interval": {"ci95": [v - 1, v + 1]},
                      "regret_plugin_p10": v, "regret_plugin_p90": v, "share_best_third": 0.1, "n_sets": 2,
                      "value": 1.0}
    blk = lambda v: {"n_configurations": 13, "rules": {r: rule(v + i) for i, r in enumerate(
        ("linear", "quadratic_index", "own_gain", "random"))},
                     "paired": {"own_gain_minus_linear": {"estimate": -1.0, "ci95": [-2.0, 0.0], "K": 13}},
                     "best_third_value": 5.0, "panel_mean_gain": 1.0, "configurations": ["a"]}
    fam = lambda v: {"n_configurations": 13, "rules": {"target_own_1": {"regret_plugin": v, "regret_crossfit": v,
                                                                        "per_group": {}},
                                                       "random": {"regret_plugin": 3.0, "regret_crossfit": 3.0,
                                                                  "per_group": {}}},
                     "paired_minus_target_own_1": {"random": {"estimate": 2.9, "ci95": [2.5, 3.3], "K": 13}}}
    F = {"saved": {"predictions_hotpot_family": {"sha256": "f", "created": "2026-11-21 20:51:18"}},
         "strict": {d: {c: {"fresh": blk(1.0), "all_distinct": blk(2.0)} for c in ("cot_req", "cot_forced", "fewshot")}
                    for d in ("shared", "disjoint")},
         "family": {c: {"fresh": fam(0.1), "all_distinct": fam(0.2)} for c in ("cot_req", "cot_forced", "fewshot")},
         "cross_benchmark": {f"{s}/{c}": {"models": ["a"], "M": 29, "n_chosen": 10, "source_linear_slope": -0.2,
                                          "fresh": {**fam(0.5), "rules": {"source_linear": {"regret_plugin": 7.35,
                                                                                            "regret_crossfit": 7.3},
                                                                          "target_own_1": {"regret_plugin": 0.52,
                                                                                           "regret_crossfit": 0.5}}},
                                          "all_distinct": fam(0.6)}
                             for s in ("gsm8k", "bbh") for c in ("cot_req", "fewshot")}}
    (tmp_path / "results" / "hotpot").mkdir(parents=True)
    (tmp_path / "results" / "hotpot" / "fair_rules_evaluation.json").write_text(json.dumps(F))
    (tmp_path / "manifests").mkdir()
    rec = {"models": ["m"] * 29, "n_chosen": 10, "sha256": "abc", "created": "2026-11-21 20:51:17",
           "design": {"n_target_items": 1000, "item_split_sha256": "z",
                      "designs": {d: {"evaluation_items": n, "pilot_pool_items": n,
                                      "ties": {"cot_req/linear": 2, "fewshot/own_gain": 5}}
                                  for d, n in (("shared", 1000), ("disjoint", 500))}}}
    (tmp_path / "manifests" / "predictions_hotpot_decision_strict.json").write_text(json.dumps(rec))
    return F


def test_fair_evaluation_renames_the_contrasts_to_baselines(tmp_path):
    F = _fair(tmp_path)
    x = ac.fair_evaluation("hotpot", root=str(tmp_path))
    assert set(x["strict"]["disjoint"]) == {"requested", "forced", "fewshot"}
    assert x["strict"]["shared"]["forced"] == F["strict"]["shared"]["cot_forced"]
    assert set(x["family"]) == {"requested", "forced", "fewshot"}
    assert set(x["cross"]) == {("gsm8k", "requested"), ("gsm8k", "fewshot"), ("bbh", "requested"), ("bbh", "fewshot")}
    assert ac.fair_evaluation("hotpot", root=str(tmp_path / "nowhere")) is None


# --------------------------------------------------------------------------- decision variants


def test_decision_strict_reads_the_added_task_block(tmp_path):
    ds = _script("run_decision_strict")
    _fair(tmp_path)
    out = ds.added_task_blocks(root=str(tmp_path))
    h = out["hotpot"]
    assert h["added_task"] and h["M"] == 29 and h["n_chosen"] == 10 and h["n_items"] == 1000
    assert set(h["results"]) == {"shared", "disjoint"} and set(h["results"]["disjoint"]) == {"requested", "forced",
                                                                                            "fewshot"}
    assert h["design"]["disjoint"]["evaluation_items"] == 500
    assert h["design"]["shared"]["ties"] == {"requested/linear": 2, "fewshot/own_gain": 5}
    reg, par = ds.tables({"tasks": {"hotpot": h}})
    assert set(reg["baseline"]) == {"requested", "forced", "fewshot"} and len(reg) == 2 * 3 * 2 * 4
    m = ma.strict_macros({"tasks": {"hotpot": h}})
    assert m["dstrictHqDisjointReqLin"] == "1.00" and m["dstrictHqDisjointFsOwn"] == "3.00"
    assert m["dstrictHqDisjointFsOwnMinusLin"] == "$-$1.00"


def test_transfer_reads_the_added_cases_apart(tmp_path):
    tr = _script("run_transfer_regret")
    _fair(tmp_path)
    fam, tsk, src, rows = tr.added_cases(root=str(tmp_path))
    assert set(fam) == {"hotpot/requested", "hotpot/forced", "hotpot/fewshot"}
    assert set(tsk) == {"gsm8k->hotpot/requested", "gsm8k->hotpot/fewshot", "bbh->hotpot/requested",
                        "bbh->hotpot/fewshot"}
    assert tsk["gsm8k->hotpot/requested"]["M"] == 29 and tsk["gsm8k->hotpot/requested"]["n_configurations"] == 13
    assert len(rows) == 3 * 2 + 4 * 2
    R = {"family": {}, "task": {}, "added_family": fam, "added_task": tsk, "added_sources": src}
    text = "\n".join(tr.added_summary_lines(R))
    assert "## C. Added tasks" in text and "HotpotQA" in text
    assert tr.added_summary_lines({"family": {}}) == []
    m = ma.transfer_added_macros(R)
    assert m["tregretTaskGsmToHqReqSrcLin"] == "7.35" and m["tregretFamHqFsOwnOne"] == "0.10"
    assert m["tregretFamHqReqRandMinusOwn"] == "2.90" and m["tregretFamHqReqRandMinusOwnLo"] == "2.50"
    assert m["tregretAddedCases"] == "7"
    # the main readers never see the added cases: their ranges stay on the main cases
    base = {"family": {"gsm8k/requested": {"rules": {"random": {"regret_plugin": 9.0}}}},
            "task": {"gsm8k->bbh/requested": {"rules": {"random": {"regret_plugin": 9.0}}}}}
    assert ma.transfer_macros({**base, **R, "family": base["family"], "task": base["task"]}) == ma.transfer_macros(base)


# --------------------------------------------------------------------------- task analyses


def test_arm_effect_cap_per_task():
    ae = _script("run_arm_effect")
    assert ae.forced_cap("gsm8k") == 16 and ae.forced_cap("hotpot") == 64


def test_protocol_ceiling_scoring_axis_per_task():
    pc = _script("run_protocol_ceiling")
    assert pc.scorings("gsm8k") == pc.SCORINGS
    assert pc.scorings("hotpot") == {"strict": "strict_correct", "lenient": "lenient_correct", "f1": "strict_f1"}
    assert pc.task_title("bbh") == "BIG-Bench Hard-28" and pc.task_title("hotpot") == "HotpotQA"
    rng = np.random.default_rng(2)
    D = {sc: {a: rng.uniform(10, 90, (5, 4)) for a in ("requested", "forced", "cot")} for sc in ("strict", "f1")}
    Y, replaced = pc.benefits(D, 100)
    assert {k[0] for k in Y} == {"strict", "f1"} and len(Y) == 2 * 2 * 3
    assert np.allclose(Y[("f1", "forced", "points")], D["f1"]["cot"] - D["f1"]["forced"])


def test_model_resampling_added_task_blocks_match_a_fresh_seeded_block(monkeypatch):
    mr = _script("run_model_resampling")
    from src import instability as ins  # noqa: F401
    rng = np.random.default_rng(3)
    M = 9
    level = np.linspace(20, 80, M)                    # models differ, so the slope is defined
    d = {a: level + rng.normal(0, 2, (6, M)) + (5 if a == "cot" else 0) - (0.2 * level if a == "cot" else 0)
         for a in ("requested", "forced", "cot")}
    fams = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
    monkeypatch.setattr(mr.ri, "load", lambda t: {"facts": {"panel": [f"m{j}" for j in range(M)], "families": fams},
                                                  "discovery": d})
    monkeypatch.setitem(mr.ri.NAMES, "toy", "Toy")
    monkeypatch.setattr(mr, "N_BOOT", 50)
    T = mr.added_task_blocks("toy", "jack_t", 0.05)
    assert set(T["blocks"]) == {"CotMinusForced/full", "CotMinusReq/full"} and T["added_task"]
    ref = mr.block(d["forced"], d["cot"], fams, "jack_t", 0.05, np.random.default_rng(mr.SEED),
                   [f"m{j}" for j in range(M)])
    assert json.dumps(T["blocks"]["CotMinusForced/full"], default=str) == json.dumps(ref, default=str)
    row = mr.slope_row("toy", "CotMinusForced", "full", ref)
    assert row["M"] == M and np.isfinite(ref["slope"]) and row["slope"] == ref["slope"]


# --------------------------------------------------------------------------- reliability map and map by panel


def test_reliability_map_writes_a_later_pool_under_its_own_keys(tmp_path):
    rmap = _script("run_reliability_map")
    tac = _tac()
    assert rmap.pk("added_settings", "base") == "added_settings" and rmap.pk("x", "hotpot") == "x_hotpot"
    assert rmap.added_setting({"group": "hotpot", "task": "hotpot", "setting": "hotpot:cot_req"}) == "hotpot:cot_req"
    inp, _ = tac._d2_added_inputs(np.random.default_rng(4), label="hotpot:cot_req")
    inp["tag"] = "HqReq"
    rows, _ = rmap.run_d2(inp, "added_h", tac.CFG, 30, str(tmp_path / "p.json"), False)
    out = {"config": tac.CFG, "added_pools": ["base", "hotpot"], "added_settings_hotpot": rows,
           "added_checks_hotpot": [], "added_departures_hotpot": rmap.departure_table(rows),
           "added_pilot_test_hotpot": rmap.summaries(rows, tac.CFG)}
    m = rmap.added_macros(out, pool="hotpot")
    assert m["mapMHqReq"] == "4" and "mapPilotTwoErrBenefitHqReq" in m
    assert rmap.added_macros(out) == {}                    # pool base has no records here
    text = "\n".join(rmap.added_summary_lines(out, "hotpot"))
    assert "Added contrasts (HotpotQA)" in text and rmap.added_summary_lines(out) == []


def test_map_panels_pool_macros_and_styles():
    mp = _script("run_map_panels")
    tac = _tac()
    base = [tac._map_row("gsm8k30", "gsm8k30", "GSM8K (29)")]
    R = {"cheng": tac._cheng_block(), "pilot_split": {}, "pilot_panels": [], "panels": [], "map_rows": base,
         "n_points": 1, "n_panels": 1}
    m0 = mp.macros(R)
    rows_base = [tac._map_row("fewshot:gsm8k", "fewshot", "Five-shot forced")]
    rows_hotpot = [tac._map_row("hotpot:cot_req", "hotpot", "HotpotQA"), tac._map_row("hotpot:fewshot", "hotpot", "HotpotQA")]
    R1 = dict(R, map_rows=base + rows_base + rows_hotpot, added_pools=["base", "hotpot"], added_groups=["fewshot"],
              added_tags={"fewshot:gsm8k": "FsGsm"}, added_pilot_rows=[], added_groups_hotpot=["hotpot"],
              added_tags_hotpot={"hotpot:cot_req": "HqReq"},
              added_pilot_rows_hotpot=[{"setting": "hotpot:cot_req", "R_from_pilot_draws": 0.91, "share_pilot_1": 0.135,
                                     "share_capability": 0.926}])
    m1 = mp.macros(R1)
    assert {k: m1[k] for k in m0} == m0
    assert m1["mpPointsAdded"] == "1" and m1["mpPointsAllWithAdded"] == "2"
    assert m1["mpPointsAddedHq"] == "2" and m1["mpPanelsAddedHq"] == "1" and m1["mpPointsAllWithAddedHq"] == "3"
    assert m1["mpPilotHqReqOneShare"] == "14" and m1["mpPilotHqReqRdisc"] == "0.91"
    assert mp.all_added_groups(R1) == {"fewshot", "hotpot"}
    assert mp.added_pred_path("base") == mp.PRED_ADDED
    assert mp.added_pred_path("hotpot").endswith("predictions_map_panels_added_hotpot.json")


# --------------------------------------------------------------------------- low-R map and noise anatomy


def test_lowr_pools_keep_the_d20_counts_and_report_d22_apart():
    lr = _script("run_lowr_map")
    tac = _tac()
    pools = lr.added_settings()
    assert list(pools) == ["base", "hotpot"] and list(pools["hotpot"]) == ["hotpot_cot_req", "hotpot_cot_forced",
                                                                     "hotpot_fewshot"]
    rng = np.random.default_rng(13)
    df = pd.concat([tac._replicates(list(lr.SETTINGS), rng), tac._replicates(["fewshot_gsm8k"], rng),
                    tac._replicates(["hotpot_cot_req"], rng)], ignore_index=True)
    R = {"added_config": "config/added_contrasts.yaml", "added_pools": ["base", "hotpot"],
         "added_settings": {"fewshot_gsm8k": "five-shot"}, "n_replicates_added": 30,
         "R_outside_unit_interval_added": 1, "added_settings_hotpot": {"hotpot_cot_req": "HotpotQA"},
         "n_replicates_added_hotpot": 30, "R_outside_unit_interval_added_hotpot": 2}
    assert lr.all_added(R) == {"fewshot_gsm8k": "five-shot", "hotpot_cot_req": "HotpotQA"}
    bands = lr.band_table(df, lr.all_added(R))
    t20 = "\n".join(lr.added_lines(R, df, bands, bands))
    t22 = "\n".join(lr.added_lines(R, df, bands, bands, "hotpot"))
    assert "fewshot_gsm8k" in t20 and "hotpot_cot_req" not in t20
    assert "hotpot_cot_req" in t22 and "2 with R outside" in t22 and "earlier pools" in t22
    m = ma.lowr_settings_macros({"settings": list(lr.SETTINGS), "bands": [], **R})
    assert m["lowrAddedSettings"] == "1" and m["lowrAddedSettingsHq"] == "1"


def test_lowr_macros_take_the_pool_counts_and_config_tags():
    assert ma.SETTING_TAG["hotpot_cot_req"] == "HqReq" and ma.SETTING_TAG["fewshot_gsm8k"] == "FsGsm"
    band = {"setting": "hotpot_cot_req", "band": "R >= 0.8", "n": 5, "median_R": 0.9, "departure_one_draw": 0.01,
            "departure_K4": 0.0}
    R = {"n_replicates": 10, "R_outside_unit_interval": 0, "n_replicates_added": 4, "R_outside_unit_interval_added": 0,
         "added_pools": ["base", "hotpot"], "n_replicates_added_hotpot": 7, "R_outside_unit_interval_added_hotpot": 1,
         "bands": [band], "null_bands": []}
    m = ma.lowr_macros(R)
    assert m["lowrReplicatesAdded"] == "4" and m["lowrReplicatesAddedHq"] == "7" and m["lowrRoutsideAddedHq"] == "1"
    assert m["lowrHqReqBandEightN"] == "5"


def test_noise_arm_name_of_a_table_arm():
    na = _script("run_noise_anatomy")
    assert na.arm_name({"kind": "table", "arm": "forced_direct"}) == "forced_direct"
    assert na.arm_name({"kind": "processed", "dir": "data/processed/forced"}).startswith("forced (")
    assert "hotpot_cot_req" in na.ADDED and na.MACRO_SUFFIX["hotpot_fewshot"] == "HqFs"


def test_map_validation_reads_hotpotqa_from_its_pool():
    def s(name, group, C, Cref, tag=None):
        x = {"setting": name, "group": group, "validation": {"one_draw": {"benefit": {"C": C}}},
             "point": {"benefit": {"C_ref": Cref}}}
        if tag:
            x["tag"] = tag
        return x
    base = {"settings": [s("gsm8k30", "gsm8k30", 0.97, 0.95), s("math", "math", 0.96, 0.95)]}
    m0 = ma.map_validation_macros(base)
    R = {**base, "added_pools": ["base", "hotpot"],
         "added_settings_hotpot": [s("hotpot:cot_req", "hotpot", 0.884, 0.86, "HqReq"),
                                s("hotpot:fewshot", "hotpot", 0.78, 0.874, "HqFs")]}
    m = ma.map_validation_macros(R)
    assert {k: m[k] for k in m0} == m0
    assert m["mapDepartValidationHqReq"] == "0.024" and m["mapDepartValidationHq"] == "0.024"
    assert m["mapDepartValidationHqFs"] == "$-$0.094"


# --------------------------------------------------------------------------- comparison of results files
