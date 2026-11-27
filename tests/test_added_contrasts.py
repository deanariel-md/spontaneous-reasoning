"""Contrasts added to the cross-cutting analyses (config/added_contrasts.yaml): the configuration and its loader, and the added-contrast paths of the reliability map, the map by panel,
the noise anatomy and the low-R map, on small fabricated data. The checks that matter most: an added contrast leaves
every main-setting number as it was, and its own numbers come from the same estimators."""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import added_contrasts as ac  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_added_test", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _contrast(**kw):
    c = {"group": "fewshot", "map_panel": "Five-shot forced", "tag": "FsGsm", "task": "gsm8k", "title": "t",
         "short": "s", "baseline": {"kind": "processed", "dir": "data/processed/forced"},
         "intervention": {"kind": "processed", "dir": "data/processed/fewshot"}, "steps": "results/x.json",
         "noise_base": "gsm8k30"}
    c.update(kw)
    return c


def _scored(panel, configs, items, rng, lift=0.15):
    rows = []
    for m_i, mk in enumerate(panel):
        for arm, up in (("direct_answer", 0.0), ("zero_shot_cot", lift * (m_i + 1))):
            for c in configs:
                p = min(0.95, 0.2 + 0.2 * m_i + up)
                for it in items:
                    rows.append({"model_key": mk, "arm": arm, "config_id": c, "item_id": it,
                                 "strict_correct": bool(rng.random() < p), "lenient_correct": False,
                                 "compliant": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- configuration and loader


def test_repository_configuration_lists_the_four_contrasts():
    con = ac.pools(ac.load_config())["base"]        # pool base, the added contrasts (HotpotQA's pool follows)
    assert list(con) == ["fewshot_gsm8k", "fewshot_bbh", "fewshot_math", "constrained_bbh"]
    assert {c["group"] for c in con.values()} == {"fewshot", "constrained"}
    assert [c["tag"] for c in con.values()] == ["FsGsm", "FsBbh", "FsMath", "GcBbh"]
    assert con["constrained_bbh"]["intervention"] == {"kind": "collection", "arm": "zero_shot_cot"}
    for c in con.values():                         # the noise base has the contrast's items
        assert c["noise_base"] == {"gsm8k": "gsm8k30", "bbh": "bbh", "math": "math"}[c["task"]]


@pytest.mark.parametrize("change, message", [
    ({"task": "hotpot"}, "task"),
    ({"tag": "Fs1"}, "letters only"),
    ({"baseline": {"kind": "elsewhere"}}, "kind"),
    ({"intervention": {"kind": "collection", "arm": "five_shot"}}, "arm"),
    ({"baseline": {"kind": "processed"}}, "needs a dir"),
    ({"intervention": {"kind": "processed", "dir": "data/processed/forced"}}, "same"),
])
def test_configuration_errors_are_refused(change, message):
    with pytest.raises(ac.ConfigError, match=message):
        ac.check_config({"contrasts": {"a": _contrast(**change)}})


def test_missing_fields_duplicate_tags_and_bad_names_are_refused():
    c = _contrast()
    del c["steps"]
    with pytest.raises(ac.ConfigError, match="missing"):
        ac.check_config({"contrasts": {"a": c}})
    with pytest.raises(ac.ConfigError, match="also"):
        ac.check_config({"contrasts": {"a": _contrast(), "b": _contrast()}})
    with pytest.raises(ac.ConfigError, match="lower case"):
        ac.check_config({"contrasts": {"Fewshot": _contrast()}})
    with pytest.raises(ac.ConfigError, match="no contrasts"):
        ac.check_config({"contrasts": {}})


def test_relabelled_loader_marks_baseline_and_intervention_rows():
    b = lambda ids: pd.DataFrame({"config_id": ids, "arm": "forced_direct", "v": 0})
    p = lambda ids: pd.DataFrame({"config_id": ids, "arm": "forced_direct", "v": 1})
    df = ac.relabelled(b, p)(["c1", "c2"])
    assert df.groupby("arm")["v"].first().to_dict() == {"direct_answer": 0, "zero_shot_cot": 1}
    col = lambda ids: pd.DataFrame({"config_id": ["c1", "c1"], "arm": ["direct_answer", "zero_shot_cot"]})
    assert ac.collection_arm(col, "zero_shot_cot")(["c1"])["arm"].tolist() == ["zero_shot_cot"]


def test_arrays_are_the_cell_table_of_each_split():
    from src import dataset2 as d2
    rng = np.random.default_rng(1)
    panel, items, disc, val = ["a", "b", "c"], list(range(12)), ["c1", "c2"], ["c3", "c4", "c5"]
    df = _scored(panel, disc + val, items, rng)
    x = {"panel": panel, "items": items, "disc": disc, "val": val, "load": lambda ids: df[df["config_id"].isin(ids)]}
    B, P = ac.arrays(x, "validation")
    cells = d2.cell_table(df[df["config_id"].isin(val)], panel, val, items)
    Bv, Pv = d2.unit_arrays(cells, val, panel)
    assert B.shape == (3, 3) and np.array_equal(B, Bv) and np.array_equal(P, Pv)
    with pytest.raises(ValueError):
        ac.arrays(x, "test")


# --------------------------------------------------------------------------- reliability map


def _d2_added_inputs(rng, label="fewshot:gsm8k"):
    panel, items = ["a", "b", "c", "d"], list(range(15))
    disc, val = ["c1", "c2", "c3", "c4"], ["c5", "c6", "c7", "c8"]
    df = _scored(panel, disc + val, items, rng)
    return {"panel": panel, "items": items, "load": lambda ids: df[df["config_id"].isin(ids)],
            "subsets": {label: ("five-shot", items)}, "group_of": {label: label.split(":")[0]},
            "sampler": {"discovery_config_ids": disc, "validation_config_ids": val, "sha256": "x", "n_bank": 8,
                        "label": "gsm8k"},
            "family": "added", "intervention": "five-shot against zero-shot forced direct", "contrast": "fewshot_gsm8k",
            "tag": "FsGsm", "steps": "results/none.json", "map_panel": "Five-shot forced"}, df


CFG = {"group_sizes": [1, 2], "pilot_sizes": [2, 4], "agreement_targets": [0.8, 0.9], "seed": 1, "usable_factor": 2}


def test_an_added_contrast_goes_through_the_dataset2_route_with_its_labels(tmp_path):
    rmap = _script("run_reliability_map")
    inp, df = _d2_added_inputs(np.random.default_rng(3))
    rows, _ = rmap.run_d2(inp, "added_x", CFG, 30, str(tmp_path / "p.json"), False)
    r = rows[0]
    assert r["intervention"] == "five-shot against zero-shot forced direct" and r["family"] == "added"
    assert (r["tag"], r["contrast"], r["map_panel"], r["group"]) == ("FsGsm", "fewshot_gsm8k", "Five-shot forced",
                                                                      "fewshot")
    # the same inputs without the labels give the same numbers: the labels change nothing else
    plain = {k: v for k, v in inp.items() if k not in ("family", "intervention", "contrast", "tag", "steps",
                                                       "map_panel")}
    rows2, _ = rmap.run_d2(plain, "plain_x", CFG, 30, str(tmp_path / "q.json"), False)
    assert rows2[0]["intervention"] == rmap.INTERVENTION["d2"] and "tag" not in rows2[0]
    same = lambda u, v: json.dumps(u, default=str, sort_keys=True) == json.dumps(v, default=str, sort_keys=True)
    assert same(rows2[0]["point"], r["point"]) and same(rows2[0]["pilot"], r["pilot"])


def test_added_checks_macros_and_summary(tmp_path):
    rmap = _script("run_reliability_map")
    inp, _ = _d2_added_inputs(np.random.default_rng(4))
    rows, _ = rmap.run_d2(inp, "added_x", CFG, 30, str(tmp_path / "p.json"), False)
    r = rows[0]
    # a Steps 1-3 file holding the record's own values: every difference is zero
    steps = {"models": inp["panel"],
             "step1": {f"C_{q}": {"C": r["point"][q]["C"], "tie": r["point"][q]["tie"]} for q in ("benefit", "accuracy")},
             "step2": {"reliability": {f"R_{q}": r["point"][q]["R"] for q in ("benefit", "accuracy")}},
             "step3": {"validation_agreement_curve": [
                 {"K": 1, **{q: {"C": r["validation"]["groups"][q]["1"]["first_k_against_next_k"]["C"]}
                             for q in ("benefit", "accuracy")}}]}}
    p = tmp_path / "steps.json"
    p.write_text(json.dumps(steps))
    r["steps"] = str(p)
    ch = rmap.added_check_rows([r])
    assert ch and all(abs(c["difference"]) < 1e-12 for c in ch if c["quantity"] != "panel_matches")
    assert [c["difference"] for c in ch if c["quantity"] == "panel_matches"] == [0.0]
    r2 = dict(r, steps=str(tmp_path / "missing.json"), setting="x:y")
    assert rmap.added_check_rows([r2])[0]["quantity"] == "steps_file_present"
    out = {"config": CFG, "added_settings": [r], "added_checks": ch, "added_departures": rmap.departure_table([r]),
           "added_pilot_test": rmap.summaries([r], CFG)}
    m = rmap.added_macros(out)
    assert m["mapMFsGsm"] == "4" and all(k.isalpha() for k in m)
    assert m["mapRbenefitFsGsm"] == rmap._f(r["point"]["benefit"]["R"], 2)
    assert {"mapPilotTwoErrBenefitFsGsm", "mapCvalAccuracyFsGsm", "mapKninetyBenefitFsGsm"} <= set(m)
    with pytest.raises(ValueError, match="already set"):
        rmap.added_macros(out, taken={"mapMFsGsm": "4"})
    text = "\n".join(rmap.added_summary_lines(out))
    assert "kept apart" in text and "five-shot" in text and "Panel sizes: 1 of 1 match" in text
    assert rmap.added_summary_lines({"config": CFG}) == []


def test_added_setting_name_is_group_and_task():
    rmap = _script("run_reliability_map")
    assert rmap.added_setting(_contrast()) == "fewshot:gsm8k"


# --------------------------------------------------------------------------- map by panel


def _cheng_block():
    pt = {"signal": 1.0, "noise": 0.5, "R": 0.6, "C": 0.7, "C_ref": 0.7}
    ns = {"var_e_intervention": 0.2, "var_e_baseline": 0.3, "corr_e": 0.1}
    return {lab: {"M": 5, "point": {"benefit": pt, "accuracy": pt}, "noise_split": ns} for lab in ("bf16", "all")}


def _map_row(setting, group, panel, q="benefit"):
    return {"setting": setting, "group": group, "panel": panel, "quantity": q, "M": 5, "n_items": 10, "R": 0.5,
            "C": 0.6, "C_ref": 0.6, "departure": 0.0, "tie": 0.0, "pilot_error_2": None, "pilot_error_4": None}


def test_map_panels_counts_ignore_the_added_contrasts():
    mp = _script("run_map_panels")
    base = [_map_row("gsm8k30", "gsm8k30", "GSM8K (29)"), _map_row("forced:gsm8k", "forced", "Forced-direct"),
            _map_row("exemplar:gsm8k", "exemplar", "Eight exemplars")]
    R = {"cheng": _cheng_block(), "pilot_split": {}, "pilot_panels": [], "panels": [], "map_rows": base,
         "n_points": 3, "n_panels": 3}
    m0 = mp.macros(R)
    added = [_map_row("fewshot:gsm8k", "fewshot", "Five-shot forced"),
             _map_row("constrained:bbh", "constrained", "CoT vs constrained")]
    R1 = dict(R, map_rows=base + added, added_groups=["constrained", "fewshot"],
              added_tags={"fewshot:gsm8k": "FsGsm", "constrained:bbh": "GcBbh"},
              added_pilot_rows=[{"setting": "fewshot:gsm8k", "R_from_pilot_draws": 0.9, "share_pilot_1": 0.1,
                                 "share_pilot_2": 0.05, "share_pilot_4": None, "share_capability": 0.8,
                                 "share_own_24": 0.01}])
    m1 = mp.macros(R1)
    assert {k: m1[k] for k in m0} == m0                    # every main macro as it was
    assert m1["mpPointsAdded"] == "2" and m1["mpPanelsAdded"] == "2" and m1["mpPointsAllWithAdded"] == "5"
    assert m1["mpPilotFsGsmOneShare"] == "10" and m1["mpPilotFsGsmFourShare"] == "--"
    assert m1["mpPilotFsGsmOwnShare"] == "1" and m1["mpPilotFsGsmRdisc"] == "0.90"


def test_added_rows_take_their_points_and_pilot_errors_from_the_map():
    mp = _script("run_map_panels")
    pt = {"R": 0.9, "C": 0.85, "C_ref": 0.86, "departure": -0.01, "tie": 0.01}
    m = {"added_settings": [{"setting": "fewshot:bbh", "group": "fewshot", "map_panel": "Five-shot forced", "M": 28,
                             "n_items": 1350, "point": {"benefit": pt, "accuracy": pt}}],
         "added_pilot_test": {"prediction_rows": [
             {"setting": "fewshot:bbh", "n_pilot": 2, "quantity": "benefit", "target": "one_draw", "K": 1,
              "error": 0.03},
             {"setting": "fewshot:bbh", "n_pilot": 2, "quantity": "benefit", "target": "groups", "K": 2,
              "error": 0.5}]}}
    rows = mp.added_rows(m)
    assert len(rows) == 2 and rows[0]["panel"] == "Five-shot forced" and rows[0]["pilot_error_2"] == 0.03
    assert rows[0]["pilot_error_4"] is None and rows[1]["pilot_error_2"] is None
    assert mp.added_rows({}) == []


def test_risk_row_is_the_share_of_the_constant_rule():
    from src.estimators import cross_risk
    mp = _script("run_map_panels")
    rng = np.random.default_rng(5)
    Bp, Pp = rng.normal(50, 10, (6, 5)), rng.normal(60, 10, (6, 5))
    Bv, Pv = rng.normal(50, 10, (6, 5)), rng.normal(60, 10, (6, 5))
    rls = mp.rules(Bp, Pp)
    row = mp.risk_row("s", "fewshot", "Five-shot forced", Bp, Pp, Bv, Pv, rls)
    Z = mp.c(Pv - Bv)
    assert row["risk_constant"] == pytest.approx(cross_risk(Z, np.zeros(5)))
    if row["risk_constant"] > 0:
        assert row["share_pilot_1"] == pytest.approx(cross_risk(Z, rls["pilot_1"]) / row["risk_constant"])
    assert row["K_pilot_draws"] == 6 and row["M"] == 5


def test_added_pilot_test_freezes_rules_before_validation(tmp_path, monkeypatch):
    mp = _script("run_map_panels")
    rng = np.random.default_rng(6)
    calls = []

    def inputs(name, con):
        return {"name": name}

    def arrays(x, split):
        calls.append((split, os.path.exists(tmp_path / "frozen.json")))
        return rng.normal(50, 5, (4, 4)), rng.normal(55, 5, (4, 4))

    monkeypatch.setattr(ac, "inputs", inputs)
    monkeypatch.setattr(ac, "arrays", arrays)
    rows, fz = mp.added_pilot_test({"fewshot_gsm8k": _contrast()}, pred_path=str(tmp_path / "frozen.json"))
    assert calls == [("discovery", False), ("validation", True)]
    assert rows[0]["setting"] == "fewshot:gsm8k" and rows[0]["panel"] == "Five-shot forced"
    saved = json.load(open(tmp_path / "frozen.json"))
    assert saved["models"] == ["fewshot:gsm8k"] and "own_4" in saved["rules"]["fewshot:gsm8k"]


# --------------------------------------------------------------------------- noise anatomy


def _noise_data(setting, rng, K=6, M=4, N=20):
    B = (rng.random((K, M, N)) < 0.5).astype(np.int8)
    P = (rng.random((K, M, N)) < 0.6).astype(np.int8)
    return {"setting": setting, "B": B, "P": P, "weights": None, "panel": list("abcd"),
            "arms": {"baseline": "b", "intervention": "p"},
            "variants": {"discovery": {"draws": np.arange(K), "kind": "draws", "label": "all"}},
            "reference": {}, "provenance": {}, "checks": {}}


NCFG = {"n_grid": {"gsm8k30": [10, 20]}, "subsamples_per_n": 30, "seed": 7, "agreement_targets": [0.8, 0.9],
        "planning_n": [10, 20]}


def test_an_added_noise_setting_uses_its_base_grid_and_item_subsamples(monkeypatch):
    na_s = _script("run_noise_anatomy")
    monkeypatch.setitem(na_s.ADDED, "toy_added", _contrast(noise_base="gsm8k30"))
    for d in (na_s.TITLE, na_s.SHORT, na_s.PRIMARY):
        monkeypatch.setitem(d, "toy_added", "discovery" if d is na_s.PRIMARY else "toy")
    data = _noise_data("toy_added", np.random.default_rng(8))
    out, rows = na_s.run_setting(data, NCFG)
    assert out["subsamples_of"] == "gsm8k30" and out["n_grid"] == [10, 20]
    base, _ = na_s.run_setting(dict(data, setting="gsm8k30"), NCFG)   # the same items: the same numbers
    assert out["variants"]["discovery"]["fits"] == base["variants"]["discovery"]["fits"]
    assert "subsamples_of" not in base and {r["setting"] for r in rows} == {"toy_added"}


def test_added_noise_macros_take_the_contrast_tag(monkeypatch):
    na_s = _script("run_noise_anatomy")
    assert na_s.MACRO_SUFFIX["fewshot_gsm8k"] == "FsGsm" and na_s.MACRO_SUFFIX["constrained_bbh"] == "GcBbh"
    assert na_s.ALL_SETTINGS[:len(na_s.SETTINGS)] == na_s.SETTINGS
    assert set(na_s.ALL_SETTINGS[len(na_s.SETTINGS):]) == set(ac.load_config())


def test_load_added_builds_the_dense_table_and_refuses_a_wrong_reference(tmp_path, monkeypatch):
    na_s = _script("run_noise_anatomy")
    rng = np.random.default_rng(9)
    panel, items, disc, val = ["a", "b", "c"], list(range(10)), ["c1", "c2", "c3"], ["c4", "c5", "c6"]
    df = _scored(panel, disc + val, items, rng)
    sampler = {"discovery_config_ids": disc, "validation_config_ids": val, "sha256": "s"}
    x = {"task": "gsm8k", "sampler": sampler, "panel": panel, "items": items,
         "load": lambda ids: df[df["config_id"].isin(ids)],
         "provenance": {"collection": {"tables": [{"path": "col.parquet"}]}, "baseline": {"path": "b.parquet"},
                        "intervention": {"collection_arm": "zero_shot_cot"}, "arms": "relabelled"},
         "baseline": {"kind": "processed", "dir": "data/processed/forced_constrained"},
         "intervention": {"kind": "collection", "arm": "zero_shot_cot"}, "steps": str(tmp_path / "none.json")}
    monkeypatch.setattr(ac, "inputs", lambda name, con: x)
    data = na_s.load_added("toy", contrasts={"toy": {}})
    assert data["arms"] == {"baseline": "forced_constrained (data/processed/forced_constrained)",
                            "intervention": "zero_shot_cot"}
    assert [t["path"] for t in data["provenance"]["tables"]] == ["col.parquet", "b.parquet"]
    assert data["B"].shape == (6, 3, 10) and data["reference"] == {}
    steps = {"sampler_sha256": "s", "models": panel,
             "step2": {"noise": {"W_gain": 123.0, "W_baseline": 1.0}, "V_delta": {"estimate": 1.0}}}
    (tmp_path / "steps.json").write_text(json.dumps(steps))
    x["steps"] = str(tmp_path / "steps.json")
    data = na_s.load_added("toy", contrasts={"toy": {}})
    monkeypatch.setitem(na_s.ADDED, "toy", _contrast(noise_base="gsm8k30"))
    for d in (na_s.TITLE, na_s.SHORT, na_s.PRIMARY):
        monkeypatch.setitem(d, "toy", "discovery" if d is na_s.PRIMARY else "toy")
    with pytest.raises(na_s.Refusal, match="does not reproduce"):
        na_s.run_setting(data, {**NCFG, "n_grid": {"gsm8k30": [5, 10]}})


# --------------------------------------------------------------------------- low-R map


def _replicates(settings, rng, n=30):
    rows = []
    for s in settings:
        for _ in range(n):
            R = rng.uniform(-0.1, 1.0)
            row = {"setting": s, "n_items": 25, "n_models": 5, "R_discovery": R, "C_one_draw": rng.uniform(0.5, 1),
                   "C_ref_one_draw": 0.5 + np.arcsin(min(max(R, 0), 1)) / np.pi if 0 <= R <= 1 else np.nan}
            for K in (2, 4, 12):
                row[f"C_K{K}"], row[f"C_ref_K{K}"] = rng.uniform(0.5, 1), rng.uniform(0.5, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_lowr_pooled_bands_exclude_the_added_contrasts():
    lr = _script("run_lowr_map")
    rng = np.random.default_rng(10)
    main = _replicates(list(lr.SETTINGS), rng)
    added = {"fewshot_gsm8k": "five-shot"}
    both = pd.concat([main, _replicates(list(added), rng)], ignore_index=True)
    old = lr.band_table(main)
    new = lr.band_table(both, added)
    pd.testing.assert_frame_equal(new.iloc[:len(old)].reset_index(drop=True), old)
    assert set(new.iloc[len(old):]["setting"]) == {"fewshot_gsm8k"}
    assert lr.outside(both) == lr.outside(main) + lr.outside(both[both["setting"] == "fewshot_gsm8k"])


def test_lowr_added_section_reports_its_own_counts():
    lr = _script("run_lowr_map")
    rng = np.random.default_rng(11)
    added = {"fewshot_gsm8k": "five-shot"}
    df = pd.concat([_replicates(list(lr.SETTINGS), rng), _replicates(list(added), rng)], ignore_index=True)
    bands = lr.band_table(df, added)
    ad = df[df["setting"] == "fewshot_gsm8k"]
    R = {"added_settings": added, "added_config": "config/added_contrasts.yaml", "n_replicates_added": len(ad),
         "R_outside_unit_interval_added": lr.outside(ad)}
    text = "\n".join(lr.added_lines(R, df, bands, bands))
    assert f"{len(ad)} replicates" in text and "fewshot_gsm8k" in text and "kept apart" in text
    assert lr.added_lines({}, df, bands, bands) == []


def test_lowr_tensors_of_an_added_contrast_split_the_relabelled_arms(monkeypatch):
    lr = _script("run_lowr_map")
    rng = np.random.default_rng(12)
    panel, items, disc, val = ["a", "b"], ["i1", "i2", "i3"], ["c1", "c2"], ["c3", "c4"]
    df = _scored(panel, disc + val, items, rng)
    x = {"sampler": {"discovery_config_ids": disc, "validation_config_ids": val}, "panel": panel, "items": items,
         "load": lambda ids: df[df["config_id"].isin(ids)]}
    monkeypatch.setattr(ac, "inputs", lambda name: x)
    out = lr.setting_tensors("fewshot_gsm8k")
    Bd, Pd = out["discovery"]
    want = df[(df["config_id"] == "c1") & (df["model_key"] == "b") & (df["arm"] == "zero_shot_cot")]
    assert Pd.shape == (2, 2, 3) and Pd[0, 1].tolist() == want["strict_correct"].astype(float).tolist()
    assert out["n_items"] == 3
