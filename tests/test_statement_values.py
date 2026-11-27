"""Tests of scripts/statement_values.py and paper/macros_statement_values.py.

The checks on results/statement_values/statement_values.json are skipped until the script has run."""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import macros_statement_values as mw  # noqa: E402

JSON = os.path.join(ROOT, mw.PATH)
needs_json = pytest.mark.skipif(not os.path.exists(JSON), reason="scripts/statement_values.py has not run")


def load():
    with open(JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------- the macros module


def test_results_absent_gives_none(tmp_path):
    assert mw.statement_values_results(str(tmp_path)) is None
    assert mw.statement_values(None) == {}


@pytest.mark.parametrize("name", ["hqReqMeanGain", "hq1Content", "ladBig", "aeGsmReversed", "ddirGsmFdToReq",
                                  "genThiGsmArmQwen2"])
def test_bad_names_refused(name):
    with pytest.raises(ValueError):
        mw.statement_values({"macros": {name: "1"}})


def test_good_names_pass():
    m = {"hqReqMeanGainContent": "0.9", "ladBigKnownRevPooled": "4", "aeRecentM": "14", "genThiGsmArmQwen": "7",
         "ddirGsmBestThirdReq": "68.93"}
    assert mw.statement_values({"macros": m}) == m


# ---------------------------------------------------------------------------------------------- the logic


def test_known_size_pairs_are_the_four_model_pairs_without_one():
    import ladder_orderings as lo
    rng = np.random.default_rng(0)
    keys = ["a", "b", "c", "d"]
    for _ in range(50):
        g1, g2 = rng.normal(size=4), rng.normal(size=4)
        four = lo.reversals(keys, g1, g2)
        three = lo.reversals(keys[:3], g1[:3], g2[:3])
        assert three["reversed_pairs"] == [p for p in four["reversed_pairs"] if "d" not in p]
        assert three["pairs"] == 3


def test_pooled_counts_sum():
    import ladder_orderings as lo
    rng = np.random.default_rng(1)
    blocks = {t: lo.reversals(["a", "b", "c"], rng.normal(size=3), rng.normal(size=3)) for t in lo.DISC3}
    p = lo.pooled_counts(blocks)
    assert p["pairs"] == 9
    assert p["reversed"] == sum(b["reversed"] for b in blocks.values())


# ---------------------------------------------------------------------------------------------- the results file


@needs_json
def test_macro_names_new_and_letters_only():
    R = load()
    m = mw.statement_values(mw.statement_values_results(ROOT))
    assert m == {k: str(v) for k, v in R["macros"].items()}
    tex = open(os.path.join(ROOT, "paper", "numbers.tex"), encoding="utf-8").read()
    # wired into make_numbers.py: every name is either absent (before a rebuild) or holds this value
    existing = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^\n]*)\}\s*$", tex, re.M))
    assert {k: (v, existing[k]) for k, v in m.items() if k in existing and existing[k] != v} == {}


@needs_json
def test_item1_strict_reproduces_and_content_formats():
    import macros_added as ma
    H = load()["hotpot"]
    for x in H["strict_reproduced"]:
        assert abs(x["here"] - x["stored"]) <= 1e-9 * max(1.0, abs(x["stored"]))
    C, m = H["content"], H["macros"]
    assert m["hqReqMeanGainContent"] == ma.num(C["step2"]["cot_req"]["mean_gain"], 1)
    assert m["hqFdRbenContent"] == ma.num(C["step2"]["cot_forced"]["R_benefit"], 2)
    assert m["hqReqTrPredCContent"] == ma.num(C["checks"]["cot_req"]["a"]["predicted"], 3)
    assert m["hqFdTrObsCContent"] == ma.num(C["checks"]["cot_forced"]["a"]["observed"], 3)
    o = C["checks"]["c_arm_switch"]["observed"]
    assert m["hqTrSwitchReversedContent"] == ma.num(o["switch"]["opposite"], 3)
    assert m["hqTrNewDrawReversedContent"] == ma.num(o["new_draw"]["opposite"], 3)
    assert len(m) == 10
    # the content prediction is a refit from the content-scored first four draws (exploratory), not the saved one
    saved = json.load(open(os.path.join(ROOT, "manifests", "predictions_hotpot_tranche.json"), encoding="utf-8"))
    assert H["strict"]["statements"]["cot_req"]["a_predicted_C_one_draw"] == pytest.approx(
        float(saved["statements"]["cot_req"]["a_predicted_C_one_draw"]), abs=1e-12)


@needs_json
def test_item4_same_sets():
    B = load()["best_third"]
    assert B["answer"] == "same"
    for x in B["tasks"].values():
        assert x["same"]
        assert x["n_configurations"] == x["best_third_configurations_stored"] == x["cost_configurations_stored"]
        assert len(x["configurations"]) == len(set(x["configurations"])) == x["n_configurations"]
    assert not any(k.startswith("ddir") for k in load()["macros"])


@needs_json
def test_item6_counts():
    K = load()["ladder_known"]
    assert "deepseek-v4.1-flash-1017" not in K["models"] and len(K["models"]) == 3
    for v, b in K["blocks"].items():
        for t in ("gsm8k", "math", "bbh", "hotpot"):
            assert b[t]["pairs"] == 3
            assert b[t]["reversed"] + b[t]["ties"] <= 3
        assert b["pooled"]["pairs"] == 9
        assert b["pooled"]["reversed"] == sum(b[t]["reversed"] for t in ("gsm8k", "math", "bbh"))
    assert all(c["equal"] for c in K["checks"])


@needs_json
def test_item7_recent_set():
    Rc = load()["recent"]
    for k in Rc["union_models"]:
        assert Rc["dates"][k]["created"] >= "2025-01-01"
        assert Rc["dates"][k]["agrees_with_hf_metadata"] is True
    for t, x in Rc["tasks"].items():
        assert x["pairs"] == x["M"] * (x["M"] - 1) // 2
        assert set(x["models"]) | set(x["older_models"]) and not (set(x["models"]) & set(x["older_models"]))
        assert all(Rc["dates"][k]["created"] < "2025-01-01" for k in x["older_models"] if k in Rc["dates"])
    assert Rc["tasks"]["hotpot"]["score"] == "content"
    assert {Rc["tasks"][t]["score"] for t in ("gsm8k", "math", "bbh")} == {"strict"}
    assert Rc["macros"]["aeRecentM"] == str(len(Rc["union_models"]))


@needs_json
def test_item8_qwen():
    Q = load()["thinking_qwen"]
    G = json.load(open(os.path.join(ROOT, "results", "generality", "generality.json"), encoding="utf-8"))
    assert Q["M"] == G["rows"]["thi_gsm8k_arm"]["M"]
    assert Q["macros"]["genThiGsmArmQwen"] == str(sum(v == "Qwen" for v in Q["families"].values()))
