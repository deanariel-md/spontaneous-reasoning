"""The macros of paper/macros_added.py, scripts/run_content_arm_ci.py,
scripts/compare_numbers_tex.py): fabricated results with known values, edge cases reported as they are, a HotpotQA task
taken under Hq, and each reader checked against the results file it reads where that file exists."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
ma = pytest.importorskip("macros_added")  # paper/ is not in the repository, so a clean checkout skips this file
import compare_numbers_tex as cmp  # noqa: E402
import run_content_arm_ci as cac  # noqa: E402


def result(name):
    p = os.path.join(ROOT, "results", *name.split("/"))
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def numbers():
    p = os.path.join(ROOT, "paper", "numbers.tex")
    if not os.path.exists(p):
        pytest.skip("paper/numbers.tex not built")
    return cmp.load(p)[0]


# ---------------------------------------------------------------- items 1 and 2: protocol ceiling


def ceiling_task(disjoint, plug=0.78):
    return {"headline": {"between_disjoint": disjoint}, "limits": {ma.CONTENT_ARM_PAIR: {"plugin": plug}}}


def test_disjoint_curve_and_content_arm_limit():
    R = {"tasks": {"gsm8k": ceiling_task({"1": 0.8002, "2": 0.79, "14": 0.7963}, 0.7833),
                   "math": ceiling_task({"1": 0.7903, "15": 0.7896}, 0.8004)}}
    c = {"tasks": {"gsm8k": {"ci95": [0.7761, 0.7934]}}}
    m = ma.ceiling_disjoint_macros(R, c)
    assert m["pceilGsmDisjOne"] == "0.800" and m["pceilGsmDisjMax"] == "0.796" and m["pceilGsmDisjMaxK"] == "14"
    assert m["pceilMathDisjOne"] == "0.790" and m["pceilMathDisjMax"] == "0.790" and m["pceilMathDisjMaxK"] == "15"
    assert m["pceilGsmArmContentLimitPlug"] == "0.783"
    assert m["pceilGsmArmContentLimitPlugLo"] == "0.776" and m["pceilGsmArmContentLimitPlugHi"] == "0.793"
    # no interval for a task the interval file does not hold: reported as absent
    assert m["pceilMathArmContentLimitPlug"] == "0.800"
    assert m["pceilMathArmContentLimitPlugLo"] == "--" and m["pceilMathArmContentLimitPlugHi"] == "--"
    assert all(k.isalpha() for k in m)


def test_disjoint_macros_without_the_interval_file_and_with_a_hotpot_task():
    R = {"tasks": {"hotpot": ceiling_task({"1": 0.7, "9": 0.72}), "bbh": ceiling_task({"1": 0.81})}}
    R["tasks"]["bbh"]["limits"] = {}                                   # no content pair: no content macros
    m = ma.ceiling_disjoint_macros(R)
    assert m["pceilHqDisjMax"] == "0.720" and m["pceilHqDisjMaxK"] == "9"
    assert m["pceilHqArmContentLimitPlugLo"] == "--"
    assert "pceilBbhArmContentLimitPlug" not in m and m["pceilBbhDisjMaxK"] == "1"


def test_content_arm_pair_result(monkeypatch):
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (20, 6))
    Ya, Yb = base + rng.normal(0, 0.5, base.shape), base + rng.normal(0, 0.5, base.shape)
    from src import protocol_ceiling as pcl
    plug = pcl.limit_plugin(Ya, Yb)
    r = cac.pair_result("gsm8k", Ya, Yb, plug, n_boot=50)
    assert r["plugin"] == plug and r["D"] == 20 and r["n_boot"] == 50 and r["ci95"][0] <= r["ci95"][1]
    assert cac.pair_result("gsm8k", Ya, Yb, plug, n_boot=50)["ci95"] == r["ci95"]          # seeded
    assert cac.pair_result("bbh", Ya, Yb, plug, n_boot=50)["ci95"] != r["ci95"]            # one stream per task
    with pytest.raises(ValueError):
        cac.pair_result("gsm8k", Ya, Yb, plug + 0.01, n_boot=50)                            # the saved plug-in must agree


def test_content_arm_limit_file_against_the_ceiling_file():
    c, pc = result("protocol_ceiling/content_arm_limit.json"), result("protocol_ceiling/protocol_ceiling.json")
    if c is None or pc is None:
        pytest.skip("results not present")
    for task, r in c["tasks"].items():
        saved = pc["tasks"][task]["limits"][ma.CONTENT_ARM_PAIR]["plugin"]
        assert abs(r["plugin"] - saved) < 1e-9 and r["D"] == pc["tasks"][task]["D"]
        assert r["ci95"][0] <= r["plugin"] + 0.02 and r["ci95"][1] >= r["plugin"] - 0.02


def test_requested_values_in_numbers_tex():
    n = numbers()
    for t, one, mx, k in (("Gsm", "0.800", "0.796", "14"), ("Bbh", "0.811", "0.815", "17"),
                          ("Math", "0.790", "0.790", "15")):
        assert (n[f"pceil{t}DisjOne"], n[f"pceil{t}DisjMax"], n[f"pceil{t}DisjMaxK"]) == (one, mx, k)
    # values under the current content reader (content/1.6.0)
    for t, v in (("Gsm", "0.783"), ("Bbh", "0.775"), ("Math", "0.803")):
        assert n[f"pceil{t}ArmContentLimitPlug"] == v


# ---------------------------------------------------------------- reliability map against validation draws


def setting(name, group, val, ref):
    return {"setting": name, "group": group, "validation": {"one_draw": {"benefit": {"C": val}}},
            "point": {"benefit": {"C_ref": ref}}}


def test_map_validation_macros():
    S = [setting("gsm8k30", "gsm8k30", 0.95, 0.94), setting("bbh:pooled", "bbh", 0.92, 0.90),
         setting("math", "math", 0.96, 0.95), setting("gsm8k19", "gsm8k", 0.957, 0.939),
         setting("d1", "dove_tasks", 0.50, 0.55), setting("d2", "dove_tasks", 0.60, 0.55),
         setting("d3", "dove_tasks", 0.70, 0.55), setting("d4", "dove_tasks", 0.90, 0.80),
         setting("b1", "bbh_tasks", 0.90, 0.92)]
    m = ma.map_validation_macros({"settings": S})
    dev = [0.01, 0.02, 0.01, 0.018, -0.05, 0.05, 0.15, 0.10, -0.02]
    assert m["mapDepartValidationRms"] == f"{np.sqrt(np.mean(np.square(dev))):.3f}" and m["mapDepartValidationN"] == "9"
    assert m["mapDepartValidationRmsPanels"] == f"{np.sqrt(np.mean(np.square(dev[:3]))):.3f}"
    assert m["mapDepartValidationRmsPanelsN"] == "3" and m["mapDepartValidationRmsDoveN"] == "4"
    assert m["mapDepartValidationRmsBbhTasks"] == "0.020" and m["mapDepartValidationRmsBbhTasksN"] == "1"
    assert m["mapDepartValidationGsmThirty"] == "0.010" and m["mapDepartValidationGsm"] == "0.018"
    assert m["mapDepartValidationBbh"] == "0.020" and m["mapDepartValidationMath"] == "0.010"
    assert m["mapValAboveNinety"] == "4" and m["mapValSettings"] == "9"       # 0.95, 0.92, 0.96, 0.957; 0.90 is not above
    assert m["mapDoveValMedian"] == "0.650" and m["mapDoveValN"] == "4"      # even count: mean of the middle two


def test_map_validation_keeps_a_hotpot_setting_apart():
    S = [setting("math", "math", 0.96, 0.95), setting("d1", "dove_tasks", 0.5, 0.5),
         setting("hotpot", "hotpot", 0.93, 0.90)]
    m = ma.map_validation_macros({"settings": S})
    assert m["mapValSettings"] == "2" and m["mapValAboveNinety"] == "1" and m["mapDepartValidationN"] == "2"
    assert m["mapDepartValidationHq"] == "0.030" and m["mapDepartValidationMath"] == "0.010"
    assert m["mapDoveValMedian"] == "0.500"


def test_map_validation_against_the_file():
    R = result("reliability_map/reliability_map.json")
    if R is None:
        pytest.skip("results not present")
    m = ma.map_validation_macros(R)
    assert len(R["settings"]) == 101 and m["mapDepartValidationN"] == "101"
    stored = {(e["scope"], e["against"]): e for e in R["pilot_test"]["prediction_errors"]
              if e["n_pilot"] == "map" and e["quantity"] == "benefit" and e["target"] == "one_draw"}
    assert m["mapDepartValidationRms"] == f"{stored[('all', 'validation')]['rmse']:.3f}"
    assert m["mapDepartValidationRmsDove"] == f"{stored[('dove_tasks', 'validation')]['rmse']:.3f}"
    assert m["mapDepartValidationRmsBbhTasks"] == f"{stored[('bbh_tasks', 'validation')]['rmse']:.3f}"
    panels = [stored[(s, "validation")]["rmse"] for s in ("gsm8k30", "bbh", "math")]
    assert m["mapDepartValidationRmsPanels"] == f"{np.sqrt(np.mean(np.square(panels))):.3f}"
    # the stored error is predicted minus observed; the macro is observed minus reference
    assert float(m["mapDepartValidationMath"]) == pytest.approx(-stored[("math", "validation")]["mean"], abs=6e-4)
    pred = {(r["setting"]): r for r in R["pilot_test"]["prediction_rows"]
            if r["n_pilot"] == "map" and r["quantity"] == "benefit" and r["target"] == "one_draw"}
    above = sum(r["observed"] > 0.9 for r in pred.values())
    assert m["mapValAboveNinety"] == str(above) == "5"
    assert m["mapDoveValMedian"] == "0.564"


# ---------------------------------------------------------------- model-bootstrap intervals that exclude zero


def block(lo, hi, flo=None, fhi=None, families=("A", "B")):
    return {"model_bootstrap": {"ci95": [lo, hi]}, "family_bootstrap": {"ci95": [flo, fhi]},
            "families": {f: 2 for f in families}}


def test_resampling_counts():
    R = {"tasks": {
        "gsm8k": {"blocks": {"a/full": block(0.1, 0.9, -1, 1), "a/readable": block(-0.9, -0.1, -0.5, -0.1),
                             "b/full": block(-0.2, 0.9, 0.1, 0.9)}},
        "bbh": {"blocks": {"a/full": block(0.2, 0.4, 0.2, 0.3), "a/readable": block(-3.0, 0.5, -0.04, -0.04, ("Q",)),
                           "b/readable": block(0.3, 0.9, 0.3, 0.9, ("Q",))}},
        "math": {"blocks": {"a/full": block(None, None, None, None), "b/full": block(0.0, 0.8, 0.0, 0.8)}}}}
    m = ma.resampling_count_macros(R)
    assert m["mresSlopes"] == "8" and m["mresExcludeZero"] == "4"
    # b/readable of BIG-Bench Hard also counts here (a readable panel, excluded in the count below)
    assert m["mresSlopesNoBbhReadable"] == "6" and m["mresExcludeZeroNoBbhReadable"] == "3"
    # an unbounded interval and one that touches zero do not exclude it; one-family blocks have no family interval
    assert m["mresFamilySlopes"] == "6" and m["mresFamilyExcludeZero"] == "3"
    assert "mresHqSlopes" not in m and all(k.isalpha() for k in m)


def test_resampling_counts_take_a_hotpot_task_apart():
    R = {"tasks": {"gsm8k": {"blocks": {"a/full": block(0.1, 0.9, 0.1, 0.9)}},
                   "hotpot": {"blocks": {"a/full": block(0.1, 0.9, 0.1, 0.9), "b/full": block(-1, 1, -1, 1)}}}}
    m = ma.resampling_count_macros(R)
    assert m["mresSlopes"] == "1" and m["mresExcludeZero"] == "1"          # the three tasks only
    assert m["mresHqSlopes"] == "2" and m["mresHqExcludeZero"] == "1"


def test_resampling_counts_against_the_summary_table():
    R, p = result("model_resampling/model_resampling.json"), os.path.join(ROOT, "results", "model_resampling",
                                                                          "summary.md")
    if R is None or not os.path.exists(p):
        pytest.skip("results not present")
    rows = [l for l in open(p, encoding="utf-8").read().splitlines() if re.match(r"^\| (GSM8K|BIG-Bench|MATH-500)", l)]
    assert len(rows) == 12
    excl, excl_no_bbh_readable = 0, 0
    for l in rows:
        cells = [c.strip() for c in l.strip("|").split("|")]
        lo, hi = [float(x.replace("−", "-")) for x in re.findall(r"[−-]?\d+\.\d+", cells[6])]
        e = lo > 0 or hi < 0
        excl += e
        if not (cells[0].startswith("BIG-Bench") and cells[2] == "readable"):
            excl_no_bbh_readable += e
    m = ma.resampling_count_macros(R)
    assert m["mresSlopes"] == "12" and m["mresExcludeZero"] == str(excl) == "6"
    assert m["mresSlopesNoBbhReadable"] == "10" and m["mresExcludeZeroNoBbhReadable"] == str(excl_no_bbh_readable)
    # the bootstrap's own sign label agrees with the interval, block by block
    for T in R["tasks"].values():
        for b in T["blocks"].values():
            assert (b["model_bootstrap_sign"] != "includes zero") == ma._excludes_zero(b["model_bootstrap"]["ci95"])


def test_excludes_zero_edge_cases():
    assert ma._excludes_zero([0.1, 0.2]) and ma._excludes_zero([-0.2, -0.1])
    assert not ma._excludes_zero([-0.1, 0.1]) and not ma._excludes_zero([0.0, 0.2]) and not ma._excludes_zero([-0.2, 0.0])
    assert not ma._excludes_zero(None) and not ma._excludes_zero([None, None]) and not ma._excludes_zero([float("nan"), 1.0])


# ---------------------------------------------------------------- the transfer table, already in numbers.tex


def test_transfer_own_gain_ranges_are_those_of_the_summary_table():
    """Section B of results/transfer_regret/summary.md: the plug-in value before the bracket, over the 12 rows."""
    p = os.path.join(ROOT, "results", "transfer_regret", "summary.md")
    if not os.path.exists(p):
        pytest.skip("summary not present")
    text = open(p, encoding="utf-8").read().split("## B. Another task")[1].split("Paired:")[0]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in text.splitlines() if l.startswith("| ") and (" | requested |" in l or " | forced |" in l)]
    assert len(rows) == 12
    first = lambda c: float(c.split("(")[0].replace("−", "-"))                                      # noqa: E731
    src, tgt = [first(r[5]) for r in rows], [first(r[8]) for r in rows]
    n = numbers()
    assert n["tregretTaskSrcOwnOneMin"] == f"{min(src):.2f}" and n["tregretTaskSrcOwnOneMax"] == f"{max(src):.2f}"
    assert n["tregretTaskTgtOwnOneMin"] == f"{min(tgt):.2f}" and n["tregretTaskTgtOwnOneMax"] == f"{max(tgt):.2f}"
    assert n["tregretTaskCases"] == "12"


# ---------------------------------------------------------------- the settings of the low-R map


def test_lowr_settings_macros():
    R = {"settings": {"exemplar": "x", "gsm8k": "x", "bbh": "x", "math": "x"}, "added_settings": {"fewshot_gsm8k": "x"},
         "bands": [{"setting": "all", "band": "0 <= R < 0.2", "n": 6},
                   {"setting": "exemplar", "band": "0 <= R < 0.2", "n": 6},
                   {"setting": "exemplar", "band": "0.4 <= R < 0.6", "n": 39},
                   {"setting": "exemplar", "band": "R >= 0.8", "n": 474},
                   {"setting": "gsm8k", "band": "0.4 <= R < 0.6", "n": 2},
                   {"setting": "bbh", "band": "R < 0", "n": 1}]}
    m = ma.lowr_settings_macros(R)
    assert m["lowrSettings"] == "4" and m["lowrAddedSettings"] == "1"
    assert (m["lowrBelowSixExemplar"], m["lowrBelowSixGsm"], m["lowrBelowSixBbh"], m["lowrBelowSixMath"]) == \
        ("45", "2", "1", "0")


def test_lowr_settings_against_the_file():
    R = result("lowr_map/lowr_map.json")
    if R is None:
        pytest.skip("results not present")
    assert sorted(R["settings"]) == ["bbh", "exemplar", "gsm8k", "math"]
    m = ma.lowr_settings_macros(R)
    assert m["lowrSettings"] == "4" and m["lowrAddedSettings"] == "4"
    assert sum(int(m[k]) for k in ("lowrBelowSixExemplar", "lowrBelowSixGsm", "lowrBelowSixBbh", "lowrBelowSixMath")) == 67
    n = numbers()
    assert n["lowrBelowSix"] == "67"


# ---------------------------------------------------------------- the check of existing macros


def test_compare_numbers_tex(tmp_path):
    a, b = tmp_path / "a.tex", tmp_path / "b.tex"
    head = "% header\n"
    a.write_text(head + "\\newcommand{\\one}{1}\n\\newcommand{\\two}{2}\n\\newcommand{\\three}{3}\n")
    b.write_text(head + "\\newcommand{\\one}{1}\n\\newcommand{\\two}{2.0}\n\\newcommand{\\four}{4}\n")
    na, _ = cmp.load(str(a))
    nb, _ = cmp.load(str(b))
    assert cmp.compare(na, nb) == (["two"], ["three"], ["four"])
    assert cmp.main([str(a), str(b), "--tsv", str(tmp_path / "t.tsv")]) == 1
    assert (tmp_path / "t.tsv").read_text() == "name\tvalue\nfour\t4\n"
    b.write_text(head + "\\newcommand{\\one}{1}\n\\newcommand{\\two}{2}\n\\newcommand{\\three}{3}\n\\newcommand{\\four}{4}\n")
    assert cmp.main([str(a), str(b)]) == 0
