"""Rules of the arm-wording run (src/arm_wording.py), on synthetic data."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, d2_io, estimators as est, generation as gen, screening  # noqa: E402

CFG = aw.load_config()
CUR_D = "Give only the final answer, without any working."
CUR_T = "Let's think step by step."


# ------------------------------------------------------------------ wordings

def test_instruction_rule():
    assert aw.qualifies_as_instruction("Let's think about this logically.")
    assert aw.qualifies_as_instruction("The answer is after the proof.")
    for t in ("First,", "Let's think", "Before we dive into the answer,", "", "let's go."):
        assert not aw.qualifies_as_instruction(t)


def test_cot_candidates_refused_until_verified():
    c = copy.deepcopy(CFG)
    c["cot"]["verified_on"] = None
    with pytest.raises(aw.ArmWordingError, match="unverified"):
        aw.cot_candidates(c)


def test_cot_candidates_order_and_count():
    got = aw.cot_candidates(CFG, require_verified=False)
    assert [g["id"] for g in got] == ["cot0", "cot1", "cot2", "cot3", "cot4", "cot5"]
    assert got[0]["text"] == CUR_T and got[0]["current"]
    assert [g["text"] for g in got[1:]] == ["Let's think about this logically.",
                                           "Let's solve this problem by splitting it into steps.",
                                           "Let's be realistic and think step by step.",
                                           "Let's think like a detective step by step.",
                                           "The answer is after the proof."]
    c = copy.deepcopy(CFG)
    c["screen"]["n_candidates"] = 2
    assert len(aw.cot_candidates(c, require_verified=False)) == 3


def test_cot_first_row_must_be_current():
    c = copy.deepcopy(CFG)
    c["cot"]["table4_instructive"] = c["cot"]["table4_instructive"][1:]
    with pytest.raises(aw.ArmWordingError):
        aw.cot_candidates(c, require_verified=False)


def _record(tmp_path, variants, verdicts):
    rec = {"kind": "paraphrase_record", "request": {"text": CUR_D}, "variants": variants}
    raw = (json.dumps(rec) + "\n").encode()
    (tmp_path / "rec.json").write_bytes(raw)
    rev = {"record_sha256": hashlib.sha256(raw).hexdigest(), "verdicts": verdicts}
    (tmp_path / "rev.json").write_text(json.dumps(rev))
    return tmp_path / "rec.json", tmp_path / "rev.json"


def test_direct_candidates_keep_order_and_cap(tmp_path):
    vs = [CUR_D] + [f"Variant {i}." for i in range(1, 8)]
    verdicts = [{"text": t, "verdict": "drop" if i == 2 else "keep", "note": "adds a format" if i == 2 else ""}
                for i, t in enumerate(vs[1:], start=1)]
    rec, rev = _record(tmp_path, vs, verdicts)
    got = aw.direct_candidates(CFG, rec, rev)
    assert [g["text"] for g in got] == [CUR_D, "Variant 1.", "Variant 3.", "Variant 4.", "Variant 5.", "Variant 6."]
    assert [g["id"] for g in got] == ["dir0", "dir1", "dir2", "dir3", "dir4", "dir5"]


def test_direct_candidates_refusals(tmp_path):
    vs = [CUR_D, "A.", "B."]
    rec, rev = _record(tmp_path, vs, [{"text": "A.", "verdict": "keep", "note": ""}])
    with pytest.raises(aw.ArmWordingError, match="no keep/drop"):
        aw.direct_candidates(CFG, rec, rev)
    rec, rev = _record(tmp_path, vs, [{"text": "A.", "verdict": "keep", "note": ""},
                                      {"text": "B.", "verdict": "drop", "note": ""}])
    with pytest.raises(aw.ArmWordingError, match="note"):
        aw.direct_candidates(CFG, rec, rev)
    r = json.loads(rev.read_text())
    r["record_sha256"] = "0" * 64
    rev.write_text(json.dumps(r))
    with pytest.raises(aw.ArmWordingError, match="another paraphrase record"):
        aw.direct_candidates(CFG, rec, rev)


def test_paraphrase_config_is_read_by_paraphrase_script():
    import paraphrase_d2 as p
    block = p.confirmed_block(ROOT / "config" / "arm_wording_paraphrase.yaml")
    assert p.paraphrased_field(block) == "instruction"
    assert block["instruction"]["text"] == CFG["current"]["direct_answer"]
    assert block["llm"]["model"] == "llama-3.3-70b-instruct-fp8"
    assert block["variations_per_field"] == 8


# ------------------------------------------------------------------ configurations

def test_draw_is_seeded_disjoint_and_from_pools():
    s = {"discovery_config_ids": ["a", "b", "a", "c", "d", "e"], "validation_config_ids": ["a", "f", "g", "h", "g", "i"]}
    d1 = aw.draw_configurations("gsm8k", s, 3, 3)
    d2 = aw.draw_configurations("gsm8k", s, 3, 3)
    assert d1 == d2
    assert set(d1["discovery"]) <= {"a", "b", "c", "d", "e"} and len(set(d1["discovery"])) == 3
    assert set(d1["validation"]) <= {"f", "g", "h", "i"} and len(set(d1["validation"])) == 3
    with pytest.raises(aw.ArmWordingError):
        aw.draw_configurations("gsm8k", s, 6, 3)


def test_saved_configuration_draw_reproduces():
    import build_arm_wording as baw
    man = aw.configs_manifest(baw.SAMPLERS, baw.BANKS, CFG)
    saved = json.loads((ROOT / "manifests" / "arm_wording_configs.json").read_text(encoding="utf-8"))
    assert aw._substance(json.loads(json.dumps(man))) == aw._substance(saved)


# ------------------------------------------------------------------ messages and requests

@pytest.mark.parametrize("task", ["gsm8k", "math"])
def test_current_clause_once_in_every_configuration(task):
    bank = json.loads((ROOT / "manifests" / f"{task}_bank.json").read_text(encoding="utf-8"))
    for cid, c in bank["configurations"].items():
        for arm in aw.ARMS:
            text = CFG["current"][aw.BANK_ARM[arm]]
            assert aw.arm_messages(bank, cid, arm, text, CFG["current"]) == c["arms"][aw.BANK_ARM[arm]]["messages"]


def test_replace_clause_exactly_once():
    m = [{"role": "user", "content": "x A y"}]
    assert aw.replace_clause(m, "A", "B")[0]["content"] == "x B y"
    with pytest.raises(aw.ArmWordingError):
        aw.replace_clause([{"role": "user", "content": "A A"}], "A", "B")
    with pytest.raises(aw.ArmWordingError):
        aw.replace_clause(m, "Z", "B")


def _bank():
    def msgs(clause):
        return {"messages": [{"role": "user", "content": f"Solve.\n\nQ: {{question}}\n{clause} End with \\boxed{{}}."}]}
    return {"configurations": {c: {"arms": {"direct_answer": msgs(CUR_D), "zero_shot_cot": msgs(CUR_T)}}
                               for c in ("c1", "c2")}}


WORDS = {"cot": [{"id": "cot0", "text": CUR_T}, {"id": "cot2", "text": "Let's think about this logically."}],
         "dir": [{"id": "dir0", "text": CUR_D}, {"id": "dir1", "text": "Answer only."}, {"id": "dir3", "text": "Final only."}]}
ITEMS = pd.DataFrame({"item_id": ["1", "2", "3"], "question": ["q1", "q2", "q3"], "gold_answer": ["1", "2", "3"]})


def test_render_counts_ids_and_forced_messages():
    df = aw.render("m", "gsm8k", _bank(), ["c1", "c2"], WORDS, ITEMS, CFG["current"])
    assert len(df) == 2 * 3 * (2 + 3 + 3)
    assert df["request_id"].is_unique
    assert set(df["config_id"]) >= {"gsm8k/c1/cot2", "gsm8k/c2/dir3"}
    r = df[df["arm"] == aw.REQ].set_index(["config_id", "item_id"])["messages_json"]
    f = df[df["arm"] == aw.FD].set_index(["config_id", "item_id"])["messages_json"]
    assert r.sort_index().equals(f.sort_index())
    assert "Answer only." in df[(df["config_id"] == "gsm8k/c1/dir1")]["messages_json"].iloc[0]
    assert all("q" in m for m in df["messages_json"])
    s = aw.render("m", "gsm8k", _bank(), ["c1"], WORDS, ITEMS, CFG["current"], prefix=aw.SCREEN_PREFIX)
    assert s["request_id"].str.startswith("screen|").all()
    dec = gen.load_decoding(str(ROOT / "config" / "decoding_arm_wording.yaml"))
    d2_io.validate_rendered(df, "m", dec)


def test_check_current_matches():
    df = aw.render("m", "gsm8k", _bank(), ["c1"], WORDS, ITEMS, CFG["current"])
    cur = df[df["config_id"].str.endswith(("/cot0", "/dir0")) & (df["arm"] != aw.FD)].copy()
    col = cur.assign(config_id=cur["config_id"].map(lambda c: aw.parse_config_id(c)[1]))
    assert aw.check_current_matches(df, col, "gsm8k")["identical"] == len(cur)
    bad = col.copy()
    bad.loc[bad.index[0], "messages_json"] = "[]"
    with pytest.raises(aw.ArmWordingError):
        aw.check_current_matches(df, bad, "gsm8k")


def test_screen_items_seeded_sorted():
    pool = [str(i) for i in range(200)]
    a = aw.screen_item_ids("gsm8k", pool, 50)
    assert a == aw.screen_item_ids("gsm8k", list(reversed(pool)), 50)
    assert len(set(a)) == 50 and a == sorted(a)
    assert a != aw.screen_item_ids("math", pool, 50)
    with pytest.raises(aw.ArmWordingError):
        aw.screen_item_ids("gsm8k", pool[:10], 50)


def test_decoding_files():
    a = gen.load_decoding(str(ROOT / "config" / "decoding_arm_wording.yaml"))
    b = gen.load_decoding(str(ROOT / "config" / "decoding_arm_wording_4k.yaml"))
    assert a.confirmed and b.confirmed
    assert a.max_new_tokens == {"direct_answer": 1024, "zero_shot_cot": 3072, "forced_direct": 16}
    assert b.max_new_tokens == {"direct_answer": 1024, "zero_shot_cot": 2900, "forced_direct": 16}
    assert a.prefill == b.prefill == {"forced_direct": "\\boxed{"}
    assert a.stop == b.stop == () and a.temperature == 0.0
    raw_a = yaml.safe_load((ROOT / "config" / "decoding_arm_wording.yaml").read_text(encoding="utf-8"))
    raw_b = yaml.safe_load((ROOT / "config" / "decoding_arm_wording_4k.yaml").read_text(encoding="utf-8"))
    raw_b["max_new_tokens"]["zero_shot_cot"] = 3072
    assert raw_a == raw_b


def test_decoding_of_model():
    import build_arm_wording as baw
    d = baw.decoding_of_factory(CFG)
    assert d("llama-2-7b-chat-hf") == "config/decoding_arm_wording_4k.yaml"
    assert d("qwen3-8b") == "config/decoding_arm_wording.yaml"


def test_dry_options_refused_inside_repository():
    import build_arm_wording as baw

    class A:
        base = None
        allow_unverified = True
        items_dir = record = review = bank_dir = None
    with pytest.raises(aw.ArmWordingError, match="dry runs"):
        baw.dry_guard(A)


# ------------------------------------------------------------------ screen

def _screen_rows(task, comp, trunc=None, models=("m1", "m2", "m3"), n=40):
    rows = []
    for w, c in comp.items():
        arm = aw.COT if w.startswith("cot") else aw.REQ
        for m in models:
            for i in range(n):
                rows.append({"model_key": m, "arm": arm, "config_id": w, "wording": w, "item_id": str(i),
                             "n_output_tokens": 20, "registered": i < c * n / 100, "lenient": True,
                             "think_tag": False, "empty": False,
                             "truncated": trunc is not None and i < trunc.get(w, 0) * n / 100, "box_closed": True})
    return pd.DataFrame(rows)


def test_screen_rules_and_choice():
    rules, _ = screening.load_rules(ROOT / "config" / "admissibility.yaml")
    comp = {"cot0": 90, "cot1": 90, "cot2": 70, "cot3": 90, "cot4": 90, "dir0": 95, "dir1": 95, "dir2": 95, "dir3": 95}
    g = _screen_rows("gsm8k", comp, trunc={"cot3": 5})                 # cot3 truncates 5% > 2% (absolute)
    m = _screen_rows("math", {**comp, "cot2": 90, "cot3": 90}, trunc={"dir1": 5})  # relative: 5 - median 0 > 2
    res = aw.apply_screen({"gsm8k": g, "math": m}, rules, {"gsm8k": False, "math": True})
    assert set(res["failed"]) == {"cot2", "cot3", "dir1"}
    cands = {"cot": [{"id": f"cot{i}", "text": str(i), "source": "", "current": i == 0} for i in range(5)],
             "dir": [{"id": f"dir{i}", "text": str(i), "source": "", "current": i == 0} for i in range(4)]}
    ch = aw.choose_wordings(cands, res["failed"], 3)
    assert [w["id"] for w in ch["cot"]] == ["cot0", "cot1", "cot4"]     # fewer than three pass: run with those
    assert [w["id"] for w in ch["dir"]] == ["dir0", "dir2", "dir3"]


def test_current_wording_stays_when_it_fails():
    cands = {"cot": [{"id": f"cot{i}", "text": str(i), "source": "", "current": i == 0} for i in range(5)]}
    ch = aw.choose_wordings(cands, {"cot0": [{}]}, 3)
    assert [w["id"] for w in ch["cot"]] == ["cot0", "cot1", "cot2", "cot3"]
    assert ch["cot"][0]["screen_failed"]


def test_screen_metrics_read_no_gold():
    out = pd.DataFrame({"model_key": ["m"] * 2, "arm": [aw.REQ, aw.FD], "config_id": ["math/c/dir0", "math/c/dir0"],
                        "item_id": ["a", "b"], "n_output_tokens": [5, 3], "finish_reason": ["stop", "stop"],
                        "output_text": ["\\boxed{3}", "\\boxed{4"]})
    r = aw.screen_row_metrics(out, "math")
    screening.assert_no_outcome_keys({c: 1 for c in r.columns})
    assert list(r["box_closed"]) == [True, False]
    assert set(r["config_id"]) == {"dir0"}


# ------------------------------------------------------------------ scoring

def test_gsm8k_forced_cut_reproduces_the_stop_string():
    assert aw.cut_at_first_brace("\\boxed{42}\nThe answer is 42.") == "\\boxed{42}"
    assert aw.cut_at_first_brace("\\boxed{4") == "\\boxed{4"
    assert aw.cut_at_first_brace("\\boxed{}") == "\\boxed{}"
    assert aw.cut_at_first_brace("\\boxed{7} and \\boxed{8}") == "\\boxed{7}"


def test_score_both_tasks_on_prefilled_text():
    spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
    out = pd.DataFrame({"request_id": ["a", "b"], "model_key": ["m", "m"], "arm": [aw.FD, aw.COT],
                        "config_id": ["gsm8k/c/dir0", "gsm8k/c/cot0"], "item_id": ["1", "2"],
                        "output_text": ["\\boxed{42}\nThe answer", "so \\boxed{7}"]})
    items = pd.DataFrame({"item_id": ["1", "2"], "question": ["", ""], "gold_answer": ["42", "8"]})
    s = aw.score(out, "gsm8k", items, spec)
    assert list(s["strict_correct"].astype(bool)) == [True, False]
    assert list(s["box_closed"]) == [True, True]
    out2 = out.assign(config_id=["math/c/dir0", "math/c/cot0"], output_text=["\\boxed{\\frac{1}{2}}", "\\boxed{3"])
    items2 = pd.DataFrame({"item_id": ["1", "2"], "question": ["", ""], "gold_answer": ["1/2", "3"]})
    s2 = aw.score(out2, "math", items2)
    assert bool(s2["strict_correct"].iloc[0]) and not bool(s2["strict_correct"].iloc[1])
    assert list(s2["box_closed"]) == [True, False]


# ------------------------------------------------------------------ arrays and analyses

def _table(C=4, T=3, D=3, M=8, I=30, seed=0):
    rng = np.random.default_rng(seed)
    ab = rng.normal(0, 1, M)
    rows = []
    for c in range(C):
        for arm, ws in ((aw.COT, [f"cot{t}" for t in range(T)]), (aw.REQ, [f"dir{d}" for d in range(D)]),
                        (aw.FD, [f"dir{d}" for d in range(D)])):
            for w in ws:
                for m in range(M):
                    p = 1 / (1 + np.exp(-(ab[m] + (0.8 if arm == aw.COT else 0) + rng.normal(0, 0.2))))
                    for i in range(I):
                        rows.append({"task": "gsm8k", "config": f"k{c}", "wording": w, "arm": arm, "model_key": f"m{m}",
                                     "item_id": str(i), "strict_correct": float(rng.random() < p),
                                     "lenient_correct": 1.0, "n_output_tokens": 10, "box_closed": True,
                                     "prompt_sha256": ""})
    return pd.DataFrame(rows)


def _ta(tab, C=4, T=3, D=3, M=8):
    return aw.TaskArrays(tab, "gsm8k", [f"k{c}" for c in range(C)], [f"cot{t}" for t in range(T)],
                         [f"dir{d}" for d in range(D)], [f"m{m}" for m in range(M)])


def test_task_arrays_shape_and_missing_cells():
    tab = _table()
    ta = _ta(tab)
    acc = ta.acc()
    assert acc[aw.COT].shape == (4, 3, 8) and acc[aw.REQ].shape == (4, 3, 8)
    w = np.ones(len(ta.items))
    assert np.allclose(ta.acc(w)[aw.FD], acc[aw.FD])
    with pytest.raises(aw.ArmWordingError):
        _ta(tab.iloc[1:])


def test_fixed_scheme_equals_appendix_a_reliability():
    G = aw.gains(_ta(_table()).acc())["req"]
    r = aw.scheme_R(G, "fixed")
    ref = est.reliability(G[:, 0, 0, :])
    assert r["R"] == pytest.approx(ref["R"]) and r["signal"] == pytest.approx(ref["signal"])
    assert r["n_ordered_pairs"] == 4 * 3


def test_scheme_pairs_respect_drawn_factors():
    cells = aw.scheme_cells((3, 2, 2, 5), "both_drawn")
    mask = aw.scheme_pair_mask(cells, "both_drawn")
    a = np.array(cells)
    k, l = np.nonzero(mask)
    assert (a[k, 0] != a[l, 0]).all() and (a[k, 1] != a[l, 1]).all() and (a[k, 2] != a[l, 2]).all()
    assert mask.sum() == 3 * 2 * 2 * 2 * 1 * 1           # c' != c (2), t' != t (1), d' != d (1), per cell
    assert len(aw.scheme_cells((3, 2, 2, 5), "trigger_drawn")) == 6
    m2 = aw.scheme_pair_mask(aw.scheme_cells((3, 2, 2, 5), "trigger_drawn"), "trigger_drawn")
    assert m2.sum() == 6 * 2


def test_drawn_wording_with_no_wording_effect_matches_fixed_in_expectation():
    # gains identical across wordings: drawing the wording changes nothing
    rng = np.random.default_rng(1)
    base = rng.normal(0, 3, (6, 1, 1, 20)) + rng.normal(0, 5, (1, 1, 1, 20))
    G = np.broadcast_to(base, (6, 4, 4, 20)).copy()
    rs = {s: aw.scheme_R(G, s)["R"] for s in aw.SCHEMES}
    assert all(v == pytest.approx(rs["fixed"]) for v in rs.values())
    assert aw.scheme_C(G, "fixed") == pytest.approx(aw.scheme_C(G, "both_drawn"))


def test_wording_noise_lowers_drawn_R():
    rng = np.random.default_rng(2)
    true = rng.normal(0, 5, 20)
    G = true + rng.normal(0, 1, (6, 4, 4, 20)) + rng.normal(0, 4, (1, 4, 1, 20))   # model x trigger interaction
    assert aw.scheme_R(G, "trigger_drawn")["R"] < aw.scheme_R(G, "fixed")["R"]


def test_gaussian_reference_reports_edge_cases():
    assert aw.gaussian_ref(-0.1)["gaussian_C"] is None
    assert aw.gaussian_ref(float("nan"))["prompts_needed_0.9"] is None
    assert aw.gaussian_ref(0.96)["prompts_needed_0.9"] == 1
    assert aw.gaussian_ref(0.95)["prompts_needed_0.9"] == 2


def test_reversal_known_values_and_ties():
    r = aw.reversal([1, 2, 3], [3, 2, 1])
    assert r == {"one_minus_C": 1.0, "reversed": 1.0, "tied": 0.0}
    r = aw.reversal([1, 1, 3], [1, 2, 3])                 # pair (0,1) tied under a: one half
    assert r["tied"] == pytest.approx(1 / 3) and r["reversed"] == 0.0
    assert r["one_minus_C"] == pytest.approx(1 / 6)
    assert aw.reversal([1, 2], [1 + 1e-12, 2])["tied"] == 0.0


def test_reversal_table_zero_when_arms_agree():
    acc = _ta(_table()).acc()
    acc[aw.FD] = acc[aw.REQ].copy()
    rt = aw.reversal_table(aw.gains(acc))
    # identical evaluations reverse no pair; 1 - C keeps half the tied pairs (the convention for tied pairs)
    assert rt["arm_switch"]["reversed"] == 0.0
    assert rt["arm_switch"]["one_minus_C"] == pytest.approx(0.5 * rt["arm_switch"]["tied"])
    assert all(x["reversed"] == 0.0 for x in rt["arm_switch_by_clause"])
    assert rt["clause_switch_requested"]["n"] == 4 * 3 * 3


def test_direct_gap():
    acc = _ta(_table()).acc()
    g = aw.direct_gap(acc, ["dir0", "dir1", "dir2"], {"dir0": 0.1})
    assert g[0]["mean_gap_pp"] == pytest.approx(float((acc[aw.REQ][:, 0] - acc[aw.FD][:, 0]).mean()))
    assert g[1]["long_requested_share"] is None


def test_protocol_axes_and_ceiling():
    assert aw.axis_of(("req", 0, 0), ("req", 1, 0)) == "trigger"
    assert aw.axis_of(("req", 0, 0), ("fd", 0, 2)) == "arm_and_clause"
    G = aw.gains(_ta(_table()).acc())
    c = aw.ceiling_table(G, 0, n_splits=5, n_boot=5)
    P = 2 * 3 * 3
    assert c["random"]["pairs"] == P * (P - 1) // 2
    assert c["arm"]["pairs"] == 9
    assert sorted(c["within"]["curve"]) == [1, 2]
    for ax, v in c.items():
        if ax != "within":
            assert 0 <= v["limit_plugin"] <= 1


def test_decision_choices_and_extra_regret():
    acc = _ta(_table()).acc()
    disc = {a: v[:2] for a, v in acc.items()}
    val = {a: v[2:] for a, v in acc.items()}
    ch = aw.decision_choices(disc, 3)
    assert len(ch["pilots"]) == 2 and len(ch["protocols"]) == 18
    for pl in ch["pilots"]:
        assert sum(pl["current"]["weights"]) == pytest.approx(3)
        assert pl["own"][0]["weights"] == pl["current"]["weights"]       # protocol 0 is the current one
    ev = aw.decision_evaluate(json.loads(json.dumps(ch)), aw.gains(val))
    assert ev["grouped"]["same"]["estimate"] == 0.0
    assert {"trigger", "clause", "arm"} <= set(ev["grouped"])


def test_discovery_quantities_roundtrip_and_change_detected():
    acc = _ta(_table()).acc()
    disc = {a: v[:2] for a, v in acc.items()}
    q = aw.discovery_quantities(disc, 3)
    back = json.loads(json.dumps(q, default=aw._default))
    assert aw.same_quantities(aw.discovery_quantities(disc, 3), back)
    disc[aw.COT][0, 0, 0] += 5
    assert not aw.same_quantities(aw.discovery_quantities(disc, 3), back)


def test_verdicts_thresholds():
    def task(R, by, cl, arm, dec):
        return {"R": {"req": {"both_drawn": {"R": R}}},
                "reversals": {"arm_switch_by_clause": [{"one_minus_C": v} for v in by],
                              "clause_switch_requested": {"one_minus_C": cl}, "arm_switch": {"one_minus_C": arm}},
                "decision": {"grouped": {"trigger": {"estimate": dec}, "clause": {"estimate": dec}}}}
    thr = CFG["analysis"]["abstract"]
    v = aw.verdicts({"gsm8k": task(0.97, [0.2, 0.21, 0.19, 0.22], 0.05, 0.2, 0.2),
                     "math": task(0.96, [0.2, 0.2, 0.2, 0.2], 0.05, 0.2, 0.0)}, thr)
    assert not v["one_prompt"]["abstract_changes"] and not v["arm_switch"]["abstract_changes"]
    assert not v["clause_moves_as_much_as_arm"]["abstract_must_say"] and not v["decision"]["sentence_gains_wording"]
    v = aw.verdicts({"gsm8k": task(0.95, [0.2, 0.26, 0.19, 0.22], 0.25, 0.2, 1.0),
                     "math": task(0.96, [0.2, 0.2, 0.2, 0.2], 0.05, 0.2, 0.0)}, thr)
    assert v["one_prompt"]["abstract_changes"] and v["arm_switch"]["abstract_changes"]
    assert v["clause_moves_as_much_as_arm"]["abstract_must_say"] and v["decision"]["sentence_gains_wording"]


def test_write_once(tmp_path):
    p = tmp_path / "x.json"
    assert aw.write_once(p, {"a": 1}, "x") == "written"
    assert aw.write_once(p, {"a": 1}, "x") == "unchanged"
    with pytest.raises(aw.ArmWordingError):
        aw.write_once(p, {"a": 2}, "x")


def test_predictions_check_refuses(tmp_path):
    import score_arm_wording as saw
    paths = aw.Paths(tmp_path)
    with pytest.raises(aw.ArmWordingError, match="not saved"):
        saw.check_predictions(paths)
    aw.write_json(paths.predictions, {"a": 1})
    paths.predictions_sha.write_text(aw.sha256_file(paths.predictions) + "  x\n")
    assert saw.check_predictions(paths)
    aw.write_json(paths.predictions, {"a": 2})
    with pytest.raises(aw.ArmWordingError, match="changed"):
        saw.check_predictions(paths)


def test_synthetic_scores_depend_on_row_key_only():
    import score_arm_wording as saw
    t = pd.DataFrame({"model_key": ["m1", "m2"] * 50, "arm": [aw.COT, aw.REQ] * 50,
                      "config_id": [f"gsm8k/c{i % 3}/cot0" for i in range(100)], "item_id": [str(i) for i in range(100)],
                      "strict_correct": False, "lenient_correct": False})
    a = saw.synthetic(t, 7)
    b = saw.synthetic(t.iloc[::2].reset_index(drop=True), 7)
    assert list(a["strict_correct"].iloc[::2]) == list(b["strict_correct"])


def test_run_check_reads_logs_only(tmp_path):
    import build_arm_wording as baw
    dec = gen.load_decoding(str(ROOT / "config" / "decoding_arm_wording.yaml"))
    rec = {"models": {"a": {"requests_sha256": "s1", "decoding": "config/decoding_arm_wording.yaml"},
                      "b": {"requests_sha256": "s2", "decoding": "config/decoding_arm_wording.yaml"}}}

    def log(folder, sha, limit=None, prefill=None, mism=0):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run_log-1.json").write_text(json.dumps({
            "status": "complete", "inputs": {"requests_sha256": sha, "limit": limit}, "prompt_render_mismatches": mism,
            "decoding": {"prefill": dec.prefill if prefill is None else prefill, "stop": [],
                         "max_new_tokens": dec.max_new_tokens}, "gpu": None}))
    log(tmp_path / "a", "s1")
    log(tmp_path / "b", "s2", limit=16)
    rows = {r["model_key"]: r for r in baw.pilot_check({"a": tmp_path / "a", "b": tmp_path / "b"}, rec, CFG)}
    assert rows["a"]["passes"] and not rows["b"]["passes"]
    log(tmp_path / "c", "s1", prefill={})
    rows = baw.pilot_check({"a": tmp_path / "c"}, {"models": {"a": rec["models"]["a"]}}, CFG)
    assert not rows[0]["passes"] and "prefill" in rows[0]["problems"][0]
