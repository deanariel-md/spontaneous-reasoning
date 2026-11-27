"""The generality analysis: reversal counting, the one-draw decision cost in both
directions, and the macros, on synthetic arrays; the written results file when it exists."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "paper"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from src import decision as dc, generality as gen  # noqa: E402

RES = os.path.join(ROOT, "results", "generality", "generality.json")


def test_reversals_count_like_pair_reversals_and_zero_is_zero():
    g = np.array([1.0, 2.0, 3.0, 4.0])
    r = gen.reversals(g, g + 5, ["a", "a", "b", "b"], 199, np.random.default_rng(0))
    assert r["reversed"] == 0 and r["pairs"] == 6 and r["reversed_share"] == 0.0
    assert r["model_bootstrap"]["ci95"] == [0.0, 0.0]
    r = gen.reversals(g, g[::-1], ["a", "a", "b", "b"], 199, np.random.default_rng(0))
    assert r["reversed"] == 6 and r["reversed_share"] == 1.0


def test_ties_are_neither_reversed_nor_kept():
    r = gen.reversals([1.0, 1.0, 2.0], [2.0, 1.0, 0.0], ["a", "b", "c"], 99, np.random.default_rng(1))
    assert r["ties"] == 1 and r["reversed"] == 2 and r["pairs"] == 3


def test_evaluation_rows_are_fresh_distinct_configurations():
    assert gen.evaluation_rows(["c1", "c2", "c1"], ["c2", "c3", "c3", "c4", "c1"]) == [1, 3]


def test_one_draw_choices_take_each_distinct_configuration_once():
    G = np.array([[3.0, 2, 1, 0], [0, 1, 2, 3], [3.0, 2, 1, 0]])
    W = gen.one_draw_choices(G, ["a", "b", "a"], 2)
    assert W.shape == (2, 4)
    assert W[0].tolist() == [1, 1, 0, 0] and W[1].tolist() == [0, 0, 1, 1]


def test_decision_cost_is_zero_when_protocols_agree_and_positive_when_they_conflict():
    rng = np.random.default_rng(3)
    M, Kd, Kv = 9, 4, 6
    true = np.arange(M, dtype=float) * 3
    Gd = true + rng.normal(0, 0.1, (Kd, M))
    Gv = true + rng.normal(0, 0.1, (Kv, M))
    disc, val = [f"d{i}" for i in range(Kd)], [f"v{i}" for i in range(Kv)]
    c = gen.decision_cost({"a": Gd, "b": Gd.copy()}, {"a": Gv, "b": Gv.copy()}, disc, val)
    assert c["n_chosen"] == dc.third_size(M) == 3 and c["n_configurations"] == Kv
    assert abs(c["a_to_b"]["estimate"]) < 1e-12 and abs(c["b_to_a"]["estimate"]) < 1e-12
    assert c["share_same_choice"] == 1.0
    c = gen.decision_cost({"a": Gd, "b": -Gd}, {"a": Gv, "b": -Gv}, disc, val)
    assert c["a_to_b"]["estimate"] > 10 and c["b_to_a"]["estimate"] > 10
    assert c["a_to_b"]["K"] == Kv and c["a_to_b"]["ci95"][0] <= c["a_to_b"]["estimate"] <= c["a_to_b"]["ci95"][1]
    # the regret difference equals mismatched minus matched plug-in regret (the best third cancels)
    d = c["a_to_b"]
    assert d["estimate"] == pytest.approx(d["regret_mismatched"] - d["regret_matched"])


def test_decision_cost_with_too_few_fresh_configurations_says_so():
    G = np.zeros((2, 4))
    c = gen.decision_cost({"a": G, "b": G}, {"a": G, "b": G}, ["x", "y"], ["x", "z"])
    assert "note" in c and "a_to_b" not in c


def _fake_results():
    v = {"reversed": 0, "pairs": 6, "reversed_share": 0.0, "ties": 0,
         "model_bootstrap": {"ci95": [0.0, 0.0]}, "family_bootstrap": {"ci95": [0.0, 0.1]}}
    cost = {"n_configurations": 10, "n_chosen": 3,
            "strict_to_lenient": {"estimate": 0.5, "ci95": [0.1, 0.9]},
            "lenient_to_strict": {"estimate": -0.2, "ci95": [-0.5, 0.1]}}
    return {"rows": {
        "fs_gsm8k_content": {"protocols": ["strict", "content"], "computed": False, "reason": "none"},
        "fs_gsm8k_lenient": {"protocols": ["strict", "lenient"], "computed": True, "M": 4, "reversals": v,
                             "decision_cost": cost},
        "thi_gsm8k_arm": {"protocols": ["requested", "forced"], "computed": True, "M": 4,
                          "reversals": {**v, "reversed": 2, "reversed_share": 2 / 6},
                          "decision_cost": {"n_configurations": 10, "n_chosen": 3,
                                            "requested_to_forced": {"estimate": 1.0, "ci95": [0.5, 1.5]},
                                            "forced_to_requested": {"estimate": 4.0, "ci95": [3.0, 5.0]}}}},
        "thinking_vs_cot": {"requested": v, "forced": v},
        "scorer_reversal_pct": {"other_interventions": {"min": 0.0, "max": 12.34}, "with_cot": {"min": 0.0, "max": 20.0}}}


def test_macros_names_values_and_not_computed_rows():
    import macros_generality as mg
    m = mg.generality_macros(_fake_results())
    assert all(k.isalpha() and k.startswith("gen") for k in m)
    assert m["genFsGsmScorerLenRev"] == "0" and m["genFsGsmScorerLenRevPct"] == "0.0"
    assert m["genFsGsmScorerLenStrictToLen"] == "0.50" and m["genFsGsmScorerLenLenToStrict"] == "$-$0.20"
    assert m["genFsGsmScorerConRev"] == "--" and m["genFsGsmScorerConStrictToConLo"] == "--"
    assert m["genThiGsmArmFdToReq"] == "4.00" and m["genThiGsmArmRev"] == "2"
    assert m["genThiGsmVsCotReqRev"] == "0"
    assert m["genScorerRevPctMax"] == "12.3" and m["genScorerRevPctMin"] == "0.0"


@pytest.mark.skipif(not os.path.exists(RES), reason="results/generality not built")
def test_results_file_keeps_every_row_and_reproduces():
    R = json.load(open(RES, encoding="utf-8"))
    import run_generality as rg
    assert list(R["rows"]) == [r[0] for r in rg.ROWS]
    for k, r in R["rows"].items():
        assert r["computed"] or r["reason"]
        if r["computed"]:
            v = r["reversals"]
            assert v["reversed"] >= 0 and v["pairs"] == r["M"] * (r["M"] - 1) // 2
    assert R["checks"]["all_reproduced"] is True
    m = R["macros"]
    for k in ("genThiGsmArmRev", "genThiGsmArmFdToReq", "genExGsmScorerChengRev", "genFsGsmScorerLenRevPct",
              "genScorerRevPctMin", "genScorerRevPctMax"):
        assert k in m


# ---------------------------------------------------------------- content scoring added for the generality rows


def test_content_loader_merges_and_refuses_mismatches(tmp_path):
    import pandas as pd
    import run_generality as rg
    rows = pd.DataFrame({"model_key": ["m", "m"], "arm": ["forced_direct"] * 2, "config_id": ["c1", "c1"],
                         "item_id": [1, 2], "strict_correct": [1, 0]})
    tab = rows.assign(content_correct=pd.Series([1, 1], dtype="int8"))
    p = tmp_path / "t.parquet"
    tab.to_parquet(p, index=False)
    out = rg.content_loader(lambda ids: rows, str(p))(["c1"])
    assert out["content_correct"].tolist() == [1, 1] and out["strict_correct"].tolist() == [1, 0]
    tab.assign(strict_correct=[0, 0]).to_parquet(p, index=False)
    with pytest.raises(SystemExit):
        rg.content_loader(lambda ids: rows, str(p))(["c1"])
    tab.iloc[:1].to_parquet(p, index=False)
    with pytest.raises(SystemExit):
        rg.content_loader(lambda ids: rows, str(p))(["c1"])


def test_added_content_reading_follows_the_forced_and_thinking_rules():
    import score_content_added as sca
    import score_thinking
    gold = {1: ("42", None, None, None)}
    # forced-style output: the folder reader puts the prefill before the continuation
    assert sca.classify_rows(["\boxed{42}"], ["stop"], [1], gold, "gsm8k") == ["correct"]
    assert sca.classify_rows(["\boxed{4"], ["length"], [1], gold, "gsm8k") == ["truncated"]
    # native thinking: read after the end-of-reasoning marker; cut before it, no answer
    texts, closed = score_thinking.answer_bodies(["<think>so \boxed{7}</think>The answer is \boxed{42}",
                                                  "<think>still going \boxed{42}"], ["stop", "length"])
    assert closed.tolist() == [True, False]
    assert sca.classify_rows(texts, ["stop", "length"], [1, 1], gold, "gsm8k") == ["correct", "truncated"]


def test_added_content_scoring_refuses_to_overwrite(monkeypatch, tmp_path):
    import score_content_added as sca
    monkeypatch.setattr(sca, "CONTENT", tmp_path)
    (tmp_path / "fewshot_gsm8k_content.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="new files only"):
        sca.run(["fewshot_gsm8k"])


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results", "generality", "content_added.json")),
                    reason="added content scoring not run")
def test_added_content_record_and_rows_filled():
    rec = json.load(open(os.path.join(ROOT, "results", "generality", "content_added.json"), encoding="utf-8"))
    rd = rec["reader"]
    if rd["uncommitted_changes"]:
        # Rescored before the reader's change was committed; the record then pins the reader by its
        # bytes and its version, and both must be the ones here
        import hashlib
        from src import content_scoring as cs
        with open(os.path.join(ROOT, "src", "content_scoring.py"), "rb") as fh:
            assert rd["sha256"] == hashlib.sha256(fh.read()).hexdigest()
        assert rec["content_scorer_version"] == cs.CONTENT_SCORER_VERSION
    else:
        assert rd["uncommitted_changes"] is False
    for name, v in rec["settings"].items():
        assert v["strict_correct_not_content_correct"] == 0
        assert v["content_correct"] >= v["strict_correct"]
    R = json.load(open(RES, encoding="utf-8"))
    assert all(r["computed"] for r in R["rows"].values())
