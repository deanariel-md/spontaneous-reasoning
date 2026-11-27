"""Step 3 on validation draws that share no configuration with the discovery draws
(scripts/run_nonoverlap_validation.py)."""
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src import dataset1 as d1, dataset2 as d2
from src.simulate import GSM8KSim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NV = _module("run_nonoverlap_validation_cli", "run_nonoverlap_validation.py")
RD2 = _module("run_dataset2_for_nonoverlap", "run_dataset2.py")
SEL = d2.load_selection(NV.SELECTION)


def _sampler(disc, val, sha="sampler-sha"):
    return {"label": "t", "discovery_config_ids": list(disc), "validation_config_ids": list(val), "sha256": sha}


def _json(o):
    return json.loads(json.dumps(o, default=d2._jsonable))


# ------------------------------------------------------------------ the filter

def test_filter_keeps_exactly_the_validation_draws_without_a_discovery_configuration():
    s = NV.split_validation(_sampler(["a", "b", "a", "c"], ["b", "d", "e", "a", "d", "f"]))
    assert s["nonoverlap_positions"] == [1, 2, 4, 5]
    assert s["nonoverlap_config_ids"] == ["d", "e", "d", "f"]
    assert s["shared_configs"] == ["a", "b"] and s["n_shared_configs"] == 2
    assert s["overlapping_draws"] == 2 and s["K_nonoverlap"] == 4
    assert s["distinct_discovery_configs"] == 3 and s["nonoverlap_distinct_configs"] == 3
    assert s["nonoverlap_repeated_configs"] == {"d": 2}


@pytest.mark.parametrize("seed", range(20))
def test_filter_on_random_samplers(seed):
    rng = np.random.default_rng(seed)
    ids = [f"c{i}" for i in range(12)]
    disc, val = rng.choice(ids, 10).tolist(), rng.choice(ids, 10).tolist()
    s = NV.split_validation(_sampler(disc, val))
    kept = s["nonoverlap_positions"]
    assert kept == sorted(kept)                                         # sampler order kept
    assert all(val[i] not in disc for i in kept)                        # nothing kept shares a configuration
    assert all(val[i] in disc for i in range(len(val)) if i not in kept)  # nothing dropped is new
    assert s["nonoverlap_config_ids"] == [val[i] for i in kept]
    assert s["K_nonoverlap"] + s["overlapping_draws"] == len(val)


def test_gsm8k_sampler_has_twelve_non_overlapping_validation_draws():
    path = os.path.join(ROOT, "manifests", "sampler_gsm8k.json")
    if not os.path.exists(path):
        pytest.skip("manifests/sampler_gsm8k.json absent")
    s = NV.split_validation(json.load(open(path, encoding="utf-8")))
    assert (s["K_discovery"], s["K_validation"]) == (24, 24)
    assert s["distinct_discovery_configs"] == 19 and s["n_shared_configs"] == 11
    assert s["overlapping_draws"] == 12 and s["K_nonoverlap"] == 12


# ------------------------------------------------------------------ the analysis on fabricated data

@pytest.fixture(scope="module")
def no_overlap(tmp_path_factory):
    """The full analysis of scripts/run_dataset2.py on fabricated outputs whose validation draws share no
    configuration with the discovery draws."""
    tmp = tmp_path_factory.mktemp("nonov")
    sim = GSM8KSim(np.random.default_rng(3), M=8, C=40, n_items=40)
    ids = sim.config_ids
    sampler = _sampler([ids[0], ids[1], ids[2], ids[1], ids[3], ids[4], ids[5], ids[6]],
                       [ids[10], ids[11], ids[12], ids[12], ids[13], ids[14], ids[15], ids[16]])
    pred = str(tmp / "pred.json")
    out = RD2.run(sim.scored, sampler, sim.panel, sim.families, sim.tiers, list(range(sim.N)), str(tmp / "res"),
                  pred, "t", n_boot=199, n_ctrl=20)
    frozen = {"strict_correct": json.load(open(pred, encoding="utf-8")),
              "lenient_correct": json.load(open(str(tmp / "pred_lenient.json"), encoding="utf-8"))}
    split, raw = NV.step3_blocks(sim.scored, sampler, frozen, list(range(sim.N)), SEL)
    return {"sim": sim, "sampler": sampler, "out": out, "frozen": frozen, "split": split, "raw": raw, "tmp": tmp}


def test_without_overlap_the_result_equals_the_full_analysis(no_overlap):
    out, raw, split = no_overlap["out"], no_overlap["raw"], no_overlap["split"]
    assert split["K_nonoverlap"] == split["K_validation"] == 8 and split["overlapping_draws"] == 0
    for value, saved in (("strict_correct", out["step3"]),
                         ("lenient_correct", out["sensitivities"]["lenient_extraction"]["step3"])):
        want = _json({k: v for k, v in saved.items() if k != "validation_agreement_curve"})
        assert _json(raw[value]["nonoverlap"]) == want
        assert _json(raw[value]["full"]) == want
        chk = NV.difference_from_saved(raw[value]["nonoverlap"], saved)
        assert chk["matches"] and chk["max_abs_difference"] == 0.0


def test_with_overlap_the_result_is_the_full_estimator_on_the_kept_draws():
    sim = GSM8KSim(np.random.default_rng(5), M=6, C=30, n_items=30)
    ids, items = sim.config_ids, list(range(sim.N))
    disc = ids[:8]
    val = [ids[2], ids[9], ids[5], ids[10], ids[10], ids[0], ids[11], ids[12]]     # positions 1, 3, 4, 6, 7 kept
    sampler = _sampler(disc, val)
    cells_d = d2.cell_table(sim.scored(disc), sim.panel, disc, items)
    frozen = {}
    for v in ("strict_correct", "lenient_correct"):
        Bd, Pd = d2.unit_arrays(cells_d, disc, sim.panel, v)
        frozen[v] = _json(d2.freeze_predictions(Bd, Pd, "t", sim.panel, sampler["sha256"]))
    split, raw = NV.step3_blocks(sim.scored, sampler, frozen, items, SEL)
    assert split["nonoverlap_positions"] == [1, 3, 4, 6, 7]
    cells = d2.cell_table(sim.scored(val), sim.panel, val, items)
    for v in frozen:
        B, P = d2.unit_arrays(cells, val, sim.panel, v)
        Z = P - B
        assert _json(raw[v]["full"]) == _json(d2.evaluate_frozen(frozen[v], Z, SEL))
        assert _json(raw[v]["nonoverlap"]) == _json(d2.evaluate_frozen(frozen[v], Z[[1, 3, 4, 6, 7]], SEL))
        assert raw[v]["nonoverlap"]["K_validation"] == 5
        assert not NV.difference_from_saved(raw[v]["nonoverlap"], raw[v]["full"])["matches"]


def test_fewer_than_two_kept_draws_are_reported_not_estimated(tmp_path):
    sim = GSM8KSim(np.random.default_rng(7), M=5, C=20, n_items=20)
    ids, items = sim.config_ids, list(range(sim.N))
    disc, val = ids[:6], [ids[0], ids[1], ids[9], ids[2]]
    sampler = _sampler(disc, val)
    cells_d = d2.cell_table(sim.scored(disc), sim.panel, disc, items)
    frozen = {}
    for v in ("strict_correct", "lenient_correct"):
        Bd, Pd = d2.unit_arrays(cells_d, disc, sim.panel, v)
        frozen[v] = _json(d2.freeze_predictions(Bd, Pd, "t", sim.panel, sampler["sha256"]))
    split, raw = NV.step3_blocks(sim.scored, sampler, frozen, items, SEL)
    assert split["K_nonoverlap"] == 1
    assert raw["strict_correct"]["nonoverlap"]["status"].startswith("not computed: 1 non-overlapping")
    block = NV.build_block("gsm8k", split, raw, {}, SEL, {"path": "x.parquet"}, str(tmp_path / "absent.json"),
                           None, sim.panel, 0.0)
    block["frozen_predictions"] = {"strict": {"created": "now", "path": "p.json"},
                                   "lenient": {"created": "now", "path": "p_lenient.json"}}
    doc = NV.write_outputs(block, str(tmp_path / "out"))
    assert not any(k.startswith("nonovGsmStrictRD") for k in doc["macros"])     # nothing estimated, nothing printed
    assert "not computed" in open(tmp_path / "out" / "summary.md", encoding="utf-8").read()


# ------------------------------------------------------------------ checks of the inputs

def test_frozen_problems_catch_edits_and_mismatches(no_overlap):
    fz, sampler = no_overlap["frozen"]["strict_correct"], no_overlap["sampler"]
    panel = no_overlap["sim"].panel
    assert NV.frozen_problems(fz, sampler, panel) == []
    edited = json.loads(json.dumps(fz))
    edited["budgets"]["8"]["own_gain"]["prediction"][0] += 1.0
    assert any("hash" in p for p in NV.frozen_problems(edited, sampler, panel))
    assert any("sampler" in p for p in NV.frozen_problems(fz, {**sampler, "sha256": "other"}, panel))
    assert any("model list" in p for p in NV.frozen_problems(fz, sampler, panel[::-1]))


def test_difference_from_saved_flags_a_changed_estimate(no_overlap):
    res = no_overlap["raw"]["strict_correct"]["full"]
    other = json.loads(json.dumps(res, default=d2._jsonable))
    other["registered"][0]["estimate"] += 1e-3
    chk = NV.difference_from_saved(res, other)
    assert not chk["matches"] and chk["max_abs_difference"] == pytest.approx(1e-3)


# ------------------------------------------------------------------ outputs

BANNED = re.compile(r"(?i)\b(locked|registered|pre-registered|pre-specified)\b|rather than")


def _block(no_overlap, task):
    split, raw = no_overlap["split"], no_overlap["raw"]
    info = {key: {"path": f"manifests/predictions_t{suffix}.json", "sha256": no_overlap["frozen"][v]["sha256"],
                  "created": no_overlap["frozen"][v]["created"]} for v, key, _, suffix in NV.SCORINGS}
    return NV.build_block(task, split, raw, info, SEL, {"path": "data/x.parquet"}, os.path.join(ROOT, "full.json"),
                          {"max_abs_difference": 0.0, "by_scoring": {}}, no_overlap["sim"].panel, 1.0)


def test_outputs_merge_tasks_and_carry_macros(no_overlap, tmp_path):
    out = str(tmp_path / "nonov")
    doc = NV.write_outputs(_block(no_overlap, "gsm8k"), out)
    m = doc["macros"]
    assert all(k.isalpha() and k.startswith("nonov") for k in m)
    assert m["nonovGsmK"] == "8" and m["nonovGsmOverlap"] == "0"
    r = next(x for x in doc["tasks"]["gsm8k"]["strict"]["nonoverlap"]["comparisons"]
             if x["f"] == "constant" and x["g"] == "own_gain")
    assert m["nonovGsmStrictRDConstOwn"] == NV.fnum(r["estimate"]) == m["nonovGsmStrictFullRDConstOwn"]
    assert m["nonovGsmLenientRDConstOlsLo"] == NV.fnum(
        next(x for x in doc["tasks"]["gsm8k"]["lenient"]["nonoverlap"]["comparisons"]
             if x["f"] == "constant" and x["g"] == "averaged_ols")["ci95"][0])
    table = pd.read_csv(os.path.join(out, "nonoverlap_validation.csv"))
    n_rd = len(d1.COMPARISONS)
    n_risk = len(doc["tasks"]["gsm8k"]["strict"]["full"]["absolute_risk"])
    assert len(table) == 2 * (n_rd + n_risk)
    assert set(table["scoring"]) == {"strict", "lenient"}
    text = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert text == open(os.path.join(out, "summary_clean.md"), encoding="utf-8").read()
    assert not BANNED.search(text), BANNED.search(text)
    assert "Every classification is the same" in text
    # a second task adds its block and macros and keeps the first
    doc = NV.write_outputs(_block(no_overlap, "bbh"), out)
    assert set(doc["tasks"]) == {"gsm8k", "bbh"}
    assert "nonovBbhStrictRDConstOwn" in doc["macros"] and "nonovGsmStrictRDConstOwn" in doc["macros"]
    assert len(pd.read_csv(os.path.join(out, "nonoverlap_validation.csv"))) == 4 * (n_rd + n_risk)
    text = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "## GSM8K" in text and "## BIG-Bench Hard" in text


def test_bbh_flag_reads_its_own_files_and_refuses_without_predictions(tmp_path, monkeypatch):
    if not os.path.exists(NV.TASKS["bbh"]["sampler"]):
        pytest.skip("manifests/sampler_bbh.json absent")
    monkeypatch.setattr("sys.argv", ["run_nonoverlap_validation.py", "--task", "bbh", "--predictions",
                                     str(tmp_path / "predictions_bbh.json"), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="predictions_bbh.json does not exist"):
        NV.main()
    assert not (tmp_path / "out").exists()


def test_formatting_keeps_signs_and_edge_values():
    assert NV.fnum(-0.5123) == "$-$0.51" and NV.fnum(None) == "--" and NV.fnum(float("nan")) == "undefined"
    assert NV.fmd(-590.487) == "−590.5" and NV.fmd(7.9e-05) == "0.00" and NV.fmd(float("inf")) == "undefined"
    with pytest.raises(ValueError):
        NV.put({}, "nonovK2", "x")
    with pytest.raises(ValueError):
        NV.put({"nonovA": "1"}, "nonovA", "2")
