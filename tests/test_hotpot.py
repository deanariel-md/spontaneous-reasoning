"""HotpotQA as a third task (src/hotpot_task.py, src/hotpot_d2.py, scripts/*hotpot*.py)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import dataset2 as d2, hotpot_d2, hotpot_task as ht, string_tasks  # noqa: E402

HAVE_ITEMS = os.path.exists(os.path.join(ROOT, "data", "processed", "hotpot", "items_target.parquet"))


# ----------------------------------------------------------------------------- normalisation, EM, F1


@pytest.mark.parametrize("s,want", [
    ("The Eiffel Tower!", "eiffel tower"), ("  an  Apple a day ", "apple day"), ("U.S.A.", "usa"),
    ("theatre", "theatre"), ("", ""), (None, ""), ("1952–1956", "1952–1956"), ("Rock-and-roll", "rockandroll")])
def test_normalize_answer(s, want):
    assert ht.normalize_answer(s) == want


def test_exact_match_and_f1():
    assert ht.exact_match("the Beatles", "Beatles") == 1
    assert ht.exact_match("Beatles band", "Beatles") == 0
    assert ht.f1("Paris France", "Paris") == pytest.approx(2 / 3)
    assert ht.f1("yes", "no") == 0.0
    assert ht.f1("no", "no") == 1.0
    assert ht.f1("yes it is", "yes") == 0.0          # gold yes/no: anything but the same string scores 0
    assert ht.f1("noanswer", "Paris") == 0.0
    assert ht.f1("London", "Paris") == 0.0
    assert ht.f1("x b b", "b b c") == pytest.approx(2 / 3)
    assert ht.f1("a b b", "b b c") == pytest.approx(0.8)      # "a" is an article and is removed


# ----------------------------------------------------------------------------- extraction


@pytest.mark.parametrize("text,strict,lenient", [
    ("So the answer is \\boxed{Paris}.", "Paris", "Paris"),
    ("\\boxed{\\text{Paris}}", "Paris", "Paris"),
    ("\\boxed{\\textbf{New \\emph{York}}}", "New York", "New York"),
    ("\\boxed{$1{,}000$}", "1{,}000", "1{,}000"),
    ("\\boxed{a} then \\boxed{b}", "b", "b"),
    ("\\boxed{Paris", None, "Paris"),
    ("\\boxed{a} and \\boxed{Lon", "a", "a"),
    ("Thinking...\nThe answer is: Marie Curie.\nDone", None, "Marie Curie."),
    ("Final Answer: Bob", None, "Bob"),
    ("first line\n\nlast line  \n", None, "last line"),
    ("", None, None),
    ("\\boxed{}", "", ""),
])
def test_extraction(text, strict, lenient):
    assert ht.strict_extract(text) == strict
    assert ht.lenient_extract(text) == lenient


def test_compliance_needs_non_empty_answer():
    assert ht.compliance("\\boxed{Paris}")
    assert not ht.compliance("\\boxed{}")
    assert not ht.compliance("\\boxed{the}")          # normalises to empty
    assert not ht.compliance("Paris")


def test_score_outputs_columns_and_values():
    items = pd.DataFrame({"item_id": ["a", "b"], "gold_answer": ["Paris", "Marie Curie"]})
    df = pd.DataFrame({"request_id": ["1", "2", "3"], "item_id": ["a", "b", "b"],
                       "output_text": ["\\boxed{paris}", "\\boxed{Curie}", "The answer is Marie Curie"]})
    out = ht.score_outputs(df, items)
    assert list(out["strict_correct"]) == [1, 0, 0]
    assert list(out["lenient_correct"]) == [1, 0, 1]
    assert list(out["compliant"]) == [True, True, False]
    assert out["strict_f1"].tolist() == pytest.approx([1.0, 2 / 3, 0.0])
    assert out["lenient_f1"].tolist() == pytest.approx([1.0, 2 / 3, 1.0])
    assert out["strict_correct"].dtype == np.int64
    with pytest.raises(KeyError):
        ht.score_outputs(pd.DataFrame({"request_id": ["x"], "item_id": ["zz"], "output_text": [""]}), items)


def test_scorer_manifest_matches_record():
    p = os.path.join(ROOT, "manifests", "hotpot_scorer.json")
    assert json.load(open(p, encoding="utf-8")) == ht.scorer_record()
    assert hotpot_d2.load_scorer()["scorer_version"] == "hotpot-boxed/1.0.0"


# ----------------------------------------------------------------------------- item text and sets


CTX = {"title": np.array(["A", "B", "C"], dtype=object),
       "sentences": np.array([np.array(["One.", " Two."], dtype=object), np.array(["Three."], dtype=object),
                              np.array([" Four.", " Five."], dtype=object)], dtype=object)}
SUP = {"title": np.array(["C", "A", "C"], dtype=object), "sent_id": np.array([0, 1, 1])}


def test_item_and_demo_text():
    assert ht.passage_block(CTX) == "Title: A\nOne. Two.\n\nTitle: B\nThree.\n\nTitle: C\nFour. Five."
    assert ht.item_text(CTX, "Q?").endswith("Four. Five.\n\nQuestion: Q?")
    assert ht.supporting_titles(SUP) == ["C", "A"]
    assert ht.demo_text(CTX, SUP, "Q?") == "Title: A\nOne. Two.\n\nTitle: C\nFour. Five.\n\nQuestion: Q?"


def test_draw_is_seeded_and_sorted():
    ids = [f"x{i:03d}" for i in range(50)]
    a = ht.draw(ids, 10, "lab")
    assert a == ht.draw(list(reversed(ids)), 10, "lab")
    assert a != ht.draw(ids, 10, "other") and len(set(a)) == 10
    assert ht.seed_from("d22/hotpot/target") == int.from_bytes(__import__("hashlib").sha256(b"d22/hotpot/target").digest()[:8], "big")
    with pytest.raises(ht.HotpotError):
        ht.draw(ids, 51, "lab")


def _toy_raw(n=40):
    rows = []
    for i in range(n):
        ans = "yes" if i % 10 == 0 else f"ans {i}"
        sup = {"title": np.array([f"T{i}", f"S{i % 7}"], dtype=object), "sent_id": np.array([0, 0])}
        ctx = {"title": np.array([f"T{i}", f"S{i % 7}"], dtype=object),
               "sentences": np.array([np.array([f"p{i}"], dtype=object), np.array(["q"], dtype=object)], dtype=object)}
        rows.append({"item_id": f"id{i:03d}", "question": f"q{i}?", "answer": ans, "qtype": "bridge", "level": "hard",
                     "supporting_facts": sup, "context": ctx})
    return pd.DataFrame(rows)


def test_split_items_rules(monkeypatch):
    monkeypatch.setattr(ht, "N_TARGET", 8)
    monkeypatch.setattr(ht, "N_SCREENING", 3)
    raw = _toy_raw()
    lengths = pd.DataFrame({"item_id": raw["item_id"], "item_tokens": [2000 if i == 5 else 100 for i in range(40)],
                            "demo_tokens": [300 if i == 7 else 50 for i in range(40)], "gold_tokens": 5})
    s = ht.split_items(raw, lengths)
    elig = {f"id{i:03d}" for i in range(40) if i % 10 and i != 5}
    assert set(s["target"]) <= elig and len(s["target"]) == 8
    assert not set(s["screening"]) & set(s["target"])
    assert not set(s["demo_pool"]) & (set(s["target"]) | set(s["screening"]))
    sup = {r.item_id: {f"T{int(r.item_id[2:])}", f"S{int(r.item_id[2:]) % 7}"} for r in raw.itertuples()}
    t_titles = set().union(*(sup[i] for i in s["target"]))
    assert all(not (sup[i] & t_titles) for i in s["screening"] + s["demo_pool"])
    assert "id007" not in s["demo_pool"]
    assert s["counts"]["gold_yes_no_or_empty"] == 4 and s["counts"]["over_length"] == 1
    tabs = ht.tables(raw, s)
    assert list(tabs["target"]["block"]) == ["A"] * 4 + ["B"] * 4
    assert tabs["demo_pool"]["question"].str.contains("Question:").all()


def test_forced_cap():
    g = pd.DataFrame({"item_id": ["a", "b", "c"], "gold_tokens": [10, 55, 70]})
    assert ht.forced_cap(["a", "b"], g) == {"longest_target_answer_tokens": 55, "cap": 64}
    assert ht.forced_cap(["a"], g)["cap"] == 16


@pytest.mark.skipif(not HAVE_ITEMS, reason="HotpotQA item tables are not on this machine (data/ is git-ignored)")
def test_registered_item_sets():
    m = json.load(open(ht.MANIFEST_PATH, encoding="utf-8"))
    t, info = hotpot_d2.load_items("target")
    s, _ = hotpot_d2.load_items("screening")
    p, _ = hotpot_d2.load_items("demo_pool")
    assert len(t) == 1000 and len(s) == 100 and info["matches_manifest"]
    assert (t["block"] == "A").sum() == 500 and list(t["block"][:500]) == ["A"] * 500
    assert not (set(t["item_id"]) & set(s["item_id"])) and not (set(p["item_id"]) & (set(t["item_id"]) | set(s["item_id"])))
    assert not t["gold_answer"].map(ht.normalize_answer).isin(["", "yes", "no"]).any()
    assert m["forced_cap"]["cap"] == 64 and m["rules"]["eligible"].startswith("normalised gold")
    assert hotpot_d2.forced_cap() == yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_hotpot.yaml")))["max_new_tokens"]["forced_direct"]


# ----------------------------------------------------------------------------- pipeline plumbing


def test_string_task_registered():
    assert "hotpot" in string_tasks.TASKS and string_tasks.module("hotpot") is hotpot_d2
    assert hotpot_d2.KIND_ITEM_SET == {"screening": "screening", "main": "target"} or \
        set(hotpot_d2.KIND_ITEM_SET.values()) == {"screening", "target"}


def test_decoding_file():
    dec = yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_hotpot.yaml"), encoding="utf-8"))
    assert dec["max_new_tokens"] == {"direct_answer": 1024, "zero_shot_cot": 1900, "forced_direct": 64,
                                     "fewshot_forced": 64}
    assert dec["prefill"] == {"forced_direct": "\\boxed{", "fewshot_forced": "\\boxed{"}
    assert dec["stop"] == [] and dec["temperature"] == 0.0
    from src import generation as gen
    d = gen.load_decoding(os.path.join(ROOT, "config", "decoding_hotpot.yaml"))
    assert d.params_for_arm("forced_direct")["prefill"] == "\\boxed{"


def test_collection_registry_lengths():
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models_hotpot.yaml"), encoding="utf-8"))["dataset2"]["models"]
    assert len(reg) == 29
    for k, e in reg.items():
        assert e["max_model_len"] == (4096 if k in hotpot_d2.NATIVE_4K else 8192)
    # the four 4,096 checkpoints: item at most 1,900 tokens, frame and markup, and the 1,900-token cap
    assert ht.MAX_ITEM_TOKENS + 1900 + 150 <= 4096


def test_screening_row_metrics_reads_no_gold():
    import re
    df = pd.DataFrame({"model_key": "m", "arm": "direct_answer", "config_id": "c", "item_id": ["a", "b", "c"],
                       "n_output_tokens": 3, "finish_reason": ["stop", "length", "stop"],
                       "output_text": ["\\boxed{Paris}", "", "the answer is x"]})
    r = hotpot_d2.row_metrics(df, None, re.compile("<think>"))
    assert r["registered"].tolist() == [True, False, False]
    assert r["lenient"].tolist() == [True, False, True]
    assert r["truncated"].tolist() == [False, True, False]


def test_cell_table_keeps_fractional_scores():
    df = pd.DataFrame({"model_key": "m", "arm": ["direct_answer"] * 2 + ["zero_shot_cot"] * 2, "config_id": "c",
                       "item_id": ["a", "b"] * 2, "strict_correct": [0.5, 0.25, 1.0, 0.0],
                       "lenient_correct": [1, 0, 1, 1], "compliant": [True] * 4})
    cells = d2.cell_table(df, ["m"], ["c"], ["a", "b"])
    assert cells["pct_strict_correct"].tolist() == pytest.approx([37.5, 50.0])
    assert cells["n_lenient_correct"].dtype == np.int64


# ----------------------------------------------------------------------------- request builder


def _coll():
    rows = []
    for c in ("c1", "c2"):
        for i in ("i1", "i2", "i3", "i4", "i5"):
            for a in ("direct_answer", "zero_shot_cot"):
                rows.append({"request_id": f"{c}|{a}|{i}", "model_key": "m", "arm": a, "config_id": c, "item_id": i,
                             "messages_json": json.dumps([{"role": "user", "content": f"{c} {a} {i}"}])})
    return pd.DataFrame(rows)


def test_four_arms_blocks_and_pilot():
    import build_hotpot_requests as b
    pre = {c: [{"role": "user", "content": "d"}, {"role": "assistant", "content": "\\boxed{x}"}] for c in ("c1", "c2")}
    allr = b.four_arms(_coll(), pre)
    assert allr["arm"].value_counts().to_dict() == {"direct_answer": 10, "zero_shot_cot": 10, "forced_direct": 10,
                                                     "fewshot_forced": 10}
    assert not allr["request_id"].duplicated().any()
    fd = allr[allr["arm"] == "forced_direct"]
    fs = allr[allr["arm"] == "fewshot_forced"]
    da = allr[allr["arm"] == "direct_answer"]
    assert fd["messages_json"].tolist() == da["messages_json"].tolist()
    assert fs["request_id"].str.endswith("|fs").all() and fd["request_id"].str.endswith("|fd").all()
    m = json.loads(fs["messages_json"].iloc[0])
    assert len(m) == 3 and m[-1] == json.loads(da["messages_json"].iloc[0])[0]
    target = pd.DataFrame({"item_id": ["i1", "i2", "i3", "i4", "i5"], "block": ["A", "A", "A", "A", "B"]})
    parts = b.split_blocks(allr, target)
    assert len(parts["A"]) == 32 and len(parts["B"]) == 8
    p = b.pilot_rows(allr, target, "c2")
    assert len(p) == 16 and set(p["config_id"]) == {"c2"} and set(p["arm"]) == set(hotpot_d2.ARMS_ALL)


def test_demo_manifest_rule():
    import build_hotpot_requests as b
    from build_fewshot_requests import demo_seed
    pool = pd.DataFrame({"item_id": [f"p{i}" for i in range(20)], "question": [f"q{i}" for i in range(20)],
                         "gold_answer": [f"g{i}" for i in range(20)]})
    s = {"discovery_config_ids": ["c2", "c1", "c2"], "validation_config_ids": ["c3"], "sha256": "x"}
    man = b.demo_manifest(s, pool, os.path.join(ROOT, "manifests", "sampler_hotpot.json"))
    assert man["configurations"] == ["c1", "c2", "c3"]
    ids = sorted(pool["item_id"])
    idx = np.random.default_rng(demo_seed("hotpot", "c1")).choice(20, size=5, replace=False)
    assert [d["item_id"] for d in man["demos"]["c1"]] == [ids[i] for i in idx]


# ----------------------------------------------------------------------------- scoring stages and statements


def test_tranche_configs_and_gate(tmp_path):
    import score_hotpot as s
    assert s.tranche_configs({"discovery_config_ids": ["a", "b", "a", "c", "d"]}) == ["a", "b", "c"]
    with pytest.raises(SystemExit, match="hard rule 6"):
        s.score_stage("all", tranche_pred=str(tmp_path / "absent.json"))


def test_decide_items(tmp_path, monkeypatch):
    import score_hotpot as s
    done = {"block_a": True, "block_b": False}
    monkeypatch.setattr(s, "complete", lambda d, t=False: done[os.path.basename(os.path.dirname(d))])
    with pytest.raises(SystemExit, match="--items A"):
        s.decide_items(["m"], None, str(tmp_path), str(tmp_path / "set.json"))
    rec = s.decide_items(["m"], "A", str(tmp_path), str(tmp_path / "set.json"))
    assert rec["items"] == "A"
    done["block_b"] = True       # once saved, the choice stands even when block B completes later
    assert s.decide_items(["m"], None, str(tmp_path), str(tmp_path / "set.json"))["items"] == "A"
    assert s.decide_items(["m"], None, str(tmp_path), str(tmp_path / "set2.json"))["items"] == "all"


def test_statements_on_synthetic_arrays():
    import run_hotpot as r
    rng = np.random.default_rng(0)
    M, K = 9, 4
    ab = np.linspace(20, 80, M)
    Breq = ab + rng.normal(0, 2, (K, M))
    Bfd = ab + 5 + rng.normal(0, 2, (K, M))
    P = ab + 0.3 * ab + rng.normal(0, 2, (K, M))
    mcfg = yaml.safe_load(open(os.path.join(ROOT, "config", "reliability_map.yaml"), encoding="utf-8"))
    T = {"pairs": {"cot_req": (Breq, P), "cot_forced": (Bfd, P), "fewshot": (Bfd, Bfd + 3 + rng.normal(0, 2, (K, M)))},
         "req4": Breq, "fd4": Bfd, "cot4": P}
    st = r.tranche_statements(T, mcfg)
    assert set(st) == {"cot_req", "cot_forced", "fewshot", "c_arm_switch", "c_arm_switch_split_half"}
    assert st["cot_req"]["d_own_gain_beats_capability"] in (True, False, None)
    assert "empirical forecast" in st["cot_req"]["d_note"]
    assert st["c_arm_switch"]["definition"] == r.C_DEFINITION
    assert st["c_arm_switch_split_half"]["definition"] == r.C_SPLIT_HALF
    sw = r.arm_switch(Breq, Bfd, P, [0, 1], [2, 3])
    assert 0 <= sw["switch"]["opposite"] <= 1 and isinstance(sw["switch_reorders_more"], bool)
    reg = r.one_draw_regrets(Breq, P, P - Breq)
    assert reg["n"] == 3 and reg["own_1"] >= 0 and reg["capability_1"] >= 0
    assert r.reordered([1, 2, 3], [3, 2, 1])["opposite"] == 1.0


# ----------------------------------------------------------------------------- further checks


def test_arm_switch_same_draw_known_values():
    import run_hotpot as r
    P = np.zeros((2, 3))
    Breq = -np.array([[1.0, 2, 3], [1, 2, 3]])        # gains g_req = [1, 2, 3] on both draws
    Bfd = -np.array([[3.0, 2, 1], [1, 2, 3]])         # g_fd reversed on draw 0, equal on draw 1
    x = r.arm_switch_same_draw(Breq, Bfd, P, units=["c1", "c1"])
    assert x["switch"]["opposite"] == pytest.approx(0.5)          # 3 of 3 pairs on draw 0, none on draw 1
    assert x["new_draw"]["requested"]["opposite"] == 0.0          # identical draws reverse nothing
    assert x["new_draw"]["forced"]["opposite"] == pytest.approx(1.0)
    assert x["new_draw"]["opposite"] == pytest.approx(0.5)        # mean over the two arms
    assert x["new_draw"]["requested"]["position_pairs_on_one_configuration"] == 1
    assert x["difference"] == pytest.approx(0.0)
    assert x["switch_reorders_more"] is False and x["equal"] is True     # equal shares: reported, not more


def test_arm_switch_same_draw_tie_convention():
    import run_hotpot as r
    P = np.zeros((1, 3))
    x = r.arm_switch_same_draw(-np.array([[1.0, 1, 3]]), -np.array([[2.0, 1, 0]]), P)
    # pair (0, 1) is tied under the requested arm: not reversed, counted as tied; the other two pairs reverse
    assert x["switch"]["opposite"] == pytest.approx(2 / 3) and x["switch"]["tie"] == pytest.approx(1 / 3)
    assert x["switch_reorders_more"] is None and x["new_draw"]["opposite"] is None     # one draw: no new-draw share
    # the within-arm share uses the same rule: a difference inside TIE_TOL is a tie, never a reversal
    w = r.within_arm_reversal(np.array([[1.0, 1.0 + 1e-12, 3], [2.0, 1, 3]]))
    assert w["opposite"] == 0.0 and w["tie"] == pytest.approx(1 / 3) and w["n_position_pairs"] == 1


def test_arm_switch_statistic_same_on_both_splits():
    import run_hotpot as r
    rng = np.random.default_rng(1)
    M = 9
    ab = np.linspace(20, 80, M)

    def draws(K):
        Breq = ab + rng.normal(0, 3, (K, M))
        Bfd = ab + 5 + rng.normal(0, 3, (K, M))
        P = 1.2 * ab + rng.normal(0, 3, (K, M))
        return Breq, Bfd, P

    mcfg = yaml.safe_load(open(os.path.join(ROOT, "config", "reliability_map.yaml"), encoding="utf-8"))
    B4, F4, P4 = draws(4)
    T = {"pairs": {"cot_req": (B4, P4), "cot_forced": (F4, P4), "fewshot": (F4, F4 + 3)},
         "req4": B4, "fd4": F4, "cot4": P4}
    st = r.tranche_statements(T, mcfg, ["a", "b", "c", "d"])
    assert st["c_arm_switch"] == r.arm_switch_same_draw(B4, F4, P4, ["a", "b", "c", "d"])
    assert st["c_arm_switch"]["new_draw"]["requested"]["n_position_pairs"] == 6
    assert st["c_arm_switch_split_half"] == r.arm_switch(B4, F4, P4, [0, 1], [2, 3])
    Bv, Fv, Pv = draws(24)
    V = {"cot_req": (Bv, Pv), "cot_forced": (Fv, Pv), "fewshot": (Fv, Fv + 3 + rng.normal(0, 1, (24, M)))}
    D4 = {c: T["pairs"][c] for c in r.CONTRASTS}
    res = r.evaluate_statements({"statements": st}, V, D4, mcfg, 20, [f"v{i}" for i in range(24)])
    obs = res["c_arm_switch"]["observed"]
    assert obs == r.arm_switch_same_draw(Bv, Fv, Pv, [f"v{i}" for i in range(24)])
    assert obs["switch"]["n_draws"] == 24 and obs["new_draw"]["forced"]["n_position_pairs"] == 276
    assert res["c_arm_switch"]["holds"] == (obs["switch_reorders_more"] == st["c_arm_switch"]["switch_reorders_more"])
    sh = res["c_arm_switch_split_half"]
    assert sh["secondary"] is True and sh["observed"] == r.arm_switch(Bv, Fv, Pv, list(range(12)), list(range(12, 24)))
    assert res["cot_req"]["d"]["forecast_own_better"] == st["cot_req"]["d_own_gain_beats_capability"]
    assert "empirical forecast" in res["cot_req"]["d"]["note"]


def test_report_filename_does_not_collide():
    import run_hotpot as r
    per_contrast = {f"{lab}{s}.json" for _, _, lab in r.CONTRASTS.values() for s in ("", "_f1", "_fresh_configurations")}
    assert r.REPORT == "hotpot_report.json" and r.REPORT not in per_contrast
    assert "hotpot.json" in per_contrast          # the requested-arm contrast keeps its own file


def test_fresh_rows_and_sensitivity():
    import run_hotpot as r
    assert r.fresh_rows(["a", "b", "a"], ["a", "c", "c", "b", "d"]) == [1, 2, 4]
    s = json.load(open(os.path.join(ROOT, "manifests", "sampler_hotpot.json"), encoding="utf-8"))
    rows = r.fresh_rows(s["discovery_config_ids"], s["validation_config_ids"])
    assert len(rows) == 15 and len({s["validation_config_ids"][i] for i in rows}) == 13
    rng = np.random.default_rng(2)
    M = 9
    ab = np.linspace(20, 80, M)
    Bd, Pd = ab + rng.normal(0, 2, (24, M)), 1.2 * ab + rng.normal(0, 2, (24, M))
    frozen = d2.freeze_predictions(Bd, Pd, "t", [f"m{i}" for i in range(M)], "x")
    sel = d2.load_selection(os.path.join(ROOT, "manifests", "calibration_selection_v2.json"))
    Zv = 0.2 * ab + rng.normal(0, 2, (24, M))
    one = r.fresh_sensitivity(frozen, Zv, [3], sel)
    assert "step3" not in one and "fewer than two" in one["note"]
    out = r.fresh_sensitivity(frozen, Zv, rows, sel)
    assert out["n_draws"] == 15 and out["step3"]["K_validation"] == 15
    assert out["step3"] == d2.evaluate_frozen(frozen, Zv[rows], sel)


def test_decision_choices_saved_for_hotpot(tmp_path):
    import run_hotpot as r
    from src import decision as dc
    rng = np.random.default_rng(3)
    M = 9
    ab = np.linspace(20, 80, M)
    B, P = ab + rng.normal(0, 2, (24, M)), 1.2 * ab + rng.normal(0, 2, (24, M))
    units = [f"c{i % 20}" for i in range(24)]
    panel = [f"m{i}" for i in range(M)]
    rec = r.freeze_decision("cot_forced", units, panel, B, P, str(tmp_path), False)
    jp = tmp_path / "predictions_decision_hotpot_forced.json"
    assert jp.exists() and (tmp_path / "predictions_decision_hotpot_forced.npz").exists()
    assert rec["pilot_seed_labels"] == {"2": "decision/hotpot/K2", "4": "decision/hotpot/K4"}
    assert rec["n_chosen"] == dc.third_size(M) and rec["setting"].startswith("HotpotQA")
    assert r.freeze_decision("cot_forced", units, panel, B, P, str(tmp_path), False)["sha256"] == rec["sha256"]
    with pytest.raises(FileExistsError):
        r.freeze_decision("cot_forced", units, panel, B + 1.0, P, str(tmp_path), False)


def _synthetic_scored(configs, panel, items, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for j, m in enumerate(panel):
        for c in configs:
            for a, shift in (("direct_answer", 0.0), ("zero_shot_cot", 0.15), ("forced_direct", 0.05),
                             ("fewshot_forced", 0.1)):
                p = min(0.95, 0.2 + 0.07 * j + shift)
                for q, i in enumerate(items):
                    x = int(rng.random() < p)
                    rows.append({"model_key": m, "arm": a, "config_id": c, "item_id": i, "strict_correct": x,
                                 "lenient_correct": x, "compliant": True, "strict_f1": float(x),
                                 "lenient_f1": float(x),
                                 # a token count for the mechanism forecast (no random draw, so scores are unchanged)
                                 "n_output_tokens": 10 + 12 * ((j + q) % 8) if a == "direct_answer" else 8})
    return pd.DataFrame(rows)


def _hotpot_setup(tmp_path, monkeypatch):
    import argparse
    import run_hotpot as r
    panel = [f"m{i}" for i in range(9)]
    items = [f"i{k}" for k in range(12)]
    rng = np.random.default_rng(5)
    cfgs = [f"c{k:02d}" for k in range(16)]
    sampler = {"discovery_config_ids": list(rng.choice(cfgs[:12], 24)), "validation_config_ids": list(rng.choice(cfgs, 24)),
               "sha256": "synthetic"}
    sp = tmp_path / "sampler.json"
    sp.write_text(json.dumps(sampler))
    monkeypatch.setattr(r, "item_ids", lambda processed=None: items)
    monkeypatch.setattr(r, "panel_info", lambda panel_=None: (panel, FAMS, None))
    a = argparse.Namespace(processed=str(tmp_path / "processed"), sampler=str(sp), pred_dir=str(tmp_path / "man"),
                           panel=None, overwrite_frozen=False, out=str(tmp_path / "out"), quick=True, n_boot=99,
                           n_control=10)
    os.makedirs(a.processed)
    os.makedirs(a.pred_dir)
    # the fair comparison: small pilots, a synthetic index, and the cross-benchmark file saved first
    import hotpot_fair as hf
    monkeypatch.setattr(hf, "N_PILOTS", 12)
    monkeypatch.setattr(hf, "N_ITEMS", 4)
    monkeypatch.setattr(hf, "panel_eci", lambda p: list(np.linspace(100.0, 140.0, len(p))))
    hf.save_cross(panel, hf.panel_eci(panel), synthetic_sources(panel, r.CONTRASTS), a.pred_dir)
    return a, sampler, panel, items


FAMS = ["A"] * 4 + ["B"] * 3 + ["C"] * 2


def synthetic_sources(panel, contrasts, seed=11):
    """{(source, contrast): (units, panel, B, P)} discovery arrays of the three source benchmarks; BIG-Bench Hard lacks
    the first model, as its real panel lacks one of HotpotQA's."""
    import hotpot_fair as hf
    rng = np.random.default_rng(seed)
    out = {}
    for t in hf.SOURCES:
        ps = panel[1:] if t == "bbh" else list(panel)
        ab = np.linspace(20, 80, len(ps))
        units = [f"{t}{k % 18}" for k in range(24)]
        for c in contrasts:
            B = ab + rng.normal(0, 3, (24, len(ps)))
            P = B + 30 - 0.3 * ab + rng.normal(0, 3, (24, len(ps)))
            out[(t, c)] = (units, ps, B, P)
    return out


def test_discovery_stage_order_and_index(tmp_path, monkeypatch):
    import run_hotpot as r
    import score_hotpot as s
    a, sampler, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="--stage predict"):           # no four-draw statements yet
        r.stage_discovery(a)
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    open(tranche, "w").write("{}")
    with pytest.raises(SystemExit, match="score_hotpot.py --stage discovery"):
        r.stage_discovery(a)
    disc = s.discovery_configs(sampler)
    assert disc == list(dict.fromkeys(sampler["discovery_config_ids"]))
    _synthetic_scored(disc, panel, items).to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    idx = r.stage_discovery(a)
    assert len(idx["files"]) == 27 and all(os.path.exists(p) for p in r.rule_files(a.pred_dir))   # 27: forecast, split and decomposition
    ip = os.path.join(a.pred_dir, r.DISCOVERY_INDEX)
    assert s.discovery_index_problems(ip) == []
    assert r.stage_discovery(a)["created"] == idx["created"]          # same rules: the index stands as saved
    # a rule file changed after the stage saved it: the validation draws stay unscored
    f = os.path.join(a.pred_dir, "predictions_hotpot_f1.json")
    keep = open(f, encoding="utf-8").read()
    open(f, "w", encoding="utf-8").write(keep + " ")
    assert any("changed" in p for p in s.discovery_index_problems(ip))
    with pytest.raises(SystemExit, match="hard rule 6"):
        s.score_stage("all", tranche_pred=tranche, discovery_pred=ip)
    open(f, "w", encoding="utf-8").write(keep)
    assert s.discovery_index_problems(ip) == []
    # the rules saved by the stage are the ones the full run's code path fits from the same discovery rows
    import run_dataset2
    fz, _, _ = run_dataset2.freeze_rules(r.loader(os.path.join(a.processed, "scored_discovery.parquet"), "cot_req"),
                                         sampler, panel, items, os.path.join(a.pred_dir, "predictions_hotpot.json"),
                                         "hotpot")
    assert fz["strict_correct"]["sha256"] == json.load(open(os.path.join(a.pred_dir, "predictions_hotpot.json")))["sha256"]


def test_scoring_and_full_run_refuse_wrong_order(tmp_path, monkeypatch):
    import run_hotpot as r
    import score_hotpot as s
    a, *_ = _hotpot_setup(tmp_path, monkeypatch)
    absent = str(tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="hard rule 6"):                # discovery before the four-draw statements
        s.score_stage("discovery", tranche_pred=absent)
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    open(tranche, "w").write("{}")
    ip = os.path.join(a.pred_dir, r.DISCOVERY_INDEX)
    with pytest.raises(SystemExit, match="hard rule 6"):                # all before the discovery rules
        s.score_stage("all", tranche_pred=tranche, discovery_pred=ip)
    with pytest.raises(SystemExit, match="--stage discovery"):
        r.run_full(a)
    open(ip, "w").write(json.dumps({"files": {}}))
    assert s.discovery_index_problems(ip) == [f"{ip} lists no rule file"]
