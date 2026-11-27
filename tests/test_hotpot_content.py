"""HotpotQA's content reading (src/hotpot_task.content_read), the format counts, the
long-answer definitions, the decomposition by definition and the draw for a third rater (src/hotpot_content.py), the scripts
around them (scripts/score_hotpot_content.py, scripts/run_hotpot_content.py), the protocol ceiling's added parameters
and the macros (paper/macros_hotpot_content.py)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import hotpot_content as hc, hotpot_task as ht  # noqa: E402


# ----------------------------------------------------------------------------- the content reading


@pytest.mark.parametrize("text, answer, source", [
    ("Flavivirus \\boxed{<answer>}", "Flavivirus", "last_line"),
    ("\\boxed{<answer>Gillian Anderson}", "Gillian Anderson", "box"),
    ("\\boxed{<answer>: Gillian Anderson}", "Gillian Anderson", "box"),
    ("\\boxed{Paris} and then \\boxed{<answer>}", "Paris", "box"),
    ("\\boxed{\\text{<Paris>}}", "Paris", "box"),
    ("\\boxed{<answer>} done\n\\boxed{Rome", "\\boxed{Rome", "last_line"),     # an unclosed box after a complete one
    ("thinking\n\\boxed{Rome", "Rome", "unclosed_box"),
    ("The answer is: Lyon\nok", "Lyon", "answer_cue"),
    ("Answer: <answer> Lyon", "Lyon", "answer_cue"),
    ("blah\nboxed{Paris} done", "Paris", "bare_box"),
    ("The answer is boxed{X}", "boxed{X}", "answer_cue"),                   # the cue comes first
    ("line one\n  Oslo  \n", "Oslo", "last_line"),
    ("", None, "empty"),
    (None, None, "empty"),
    ("\\boxed{}\nMadrid", "Madrid", "last_line"),                           # an empty box is not an answer
])
def test_content_read_cases(text, answer, source):
    assert ht.content_read(text) == (answer, source)
    assert ht.content_extract(text) == answer


def test_strict_and_lenient_are_unchanged_by_the_content_reading():
    # the rules record of hotpot-boxed/1.0.0 is the pinned one, and the placeholder still reads as strict did
    pinned = json.load(open(os.path.join(ROOT, "manifests", "hotpot_scorer.json"), encoding="utf-8"))
    assert ht.scorer_record() == pinned
    assert ht.SCORER_VERSION == "hotpot-boxed/1.0.0"
    assert ht.strict_extract("Flavivirus \\boxed{<answer>}") == "<answer>"
    assert ht.lenient_extract("Flavivirus \\boxed{<answer>}") == "<answer>"
    assert ht.compliance("\\boxed{<answer>}") is True


def test_content_scorer_record_has_its_own_version_and_stable_hash():
    r = ht.content_scorer_record()
    assert r["scorer_version"] == ht.CONTENT_SCORER_VERSION == "hotpot-content/1.5.0"
    assert r["beside"] == ht.SCORER_VERSION and r["rules_sha256"] != ht.scorer_record()["rules_sha256"]
    assert ht.content_scorer_record() == r
    p = os.path.join(ROOT, "manifests", "hotpot_content_scorer.json")
    if os.path.exists(p):
        assert json.load(open(p, encoding="utf-8")) == r


def test_score_content_columns_and_edge_cases():
    df = pd.DataFrame({"item_id": ["a", "b", "a", "b"],
                       "output_text": ["Paris \\boxed{<answer>}", "\\boxed{The Beatles}", np.nan, "\\boxed{beatles}"]})
    items = pd.DataFrame({"item_id": ["a", "b"], "gold_answer": ["Paris", "the Beatles"]})
    out = ht.score_content(df, items)
    assert list(out["content_correct"]) == [1, 1, 0, 1]
    assert out["content_correct"].dtype == np.int64
    assert list(out["content_source"]) == ["last_line", "box", "empty", "box"]
    assert out["content_f1"].iloc[2] == 0.0 and out["content_answer"].iloc[2] is None
    with pytest.raises(KeyError):
        ht.score_content(df.assign(item_id=["a", "z", "a", "b"]), items)
    with pytest.raises(ValueError):
        ht.score_content(df.drop(columns="output_text"), items)


# ----------------------------------------------------------------------------- format flags


@pytest.mark.parametrize("text, ph, want", [
    ("\\boxed{Paris}", "<answer>", (False, False, "usable_box")),
    ("Paris \\boxed{<answer>}", "<answer>", (True, True, "unboxed")),
    ("\\boxed{<ANSWER>Paris}", "<answer>", (True, True, "usable_box")),
    ("so boxed{Paris}", "<answer>", (False, False, "bare_box")),
    ("\\boxed{Par", "<answer>", (False, False, "unclosed_box")),
    ("just Paris", "<answer>", (False, False, "unboxed")),
    ("42 \\boxed{<number>}", "<number>", (True, True, "unboxed")),
    ("\\boxed{42}", "<number>", (False, False, "usable_box")),
    (None, "<answer>", (False, False, "unboxed")),
])
def test_format_flags(text, ph, want):
    assert hc.format_flags(text, ph) == want


def test_flag_frame_and_counts():
    df = pd.DataFrame({"model_key": ["m1", "m1", "m2", "m2"], "arm": ["r", "r", "r", "c"],
                       "output_text": ["P \\boxed{<answer>}", "\\boxed{P}", "boxed{P}", "\\boxed{P"]})
    fl = hc.flag_frame(df["output_text"].tolist(), "<answer>")
    t = pd.concat([df, fl], axis=1)
    c = hc.flag_counts(t).set_index(["model_key", "arm"])
    assert c.loc[("m1", "r"), "n"] == 2 and c.loc[("m1", "r"), "placeholder_copy"] == 1
    assert c.loc[("m1", "r"), "unboxed"] == 1 and c.loc[("m1", "r"), "usable_box"] == 1
    assert c.loc[("m1", "r"), "placeholder_no_usable_box"] == 1
    assert c.loc[("m2", "r"), "bare_box"] == 1 and c.loc[("m2", "c"), "unclosed_box"] == 1
    tot = hc.flag_counts(t, by=("arm",)).set_index("arm")
    assert tot.loc["r", "n"] == 3 and tot[list(hc.ANSWER_PLACES)].sum(axis=1).equals(tot["n"])


# ----------------------------------------------------------------------------- long answers and the decomposition


def test_long_flags_definitions():
    r, f = np.array([10, 17, 65, 20, 30]), np.array([2, 9, 3, 12, 22])
    assert hc.long_flags("cap64", r).tolist() == [False, False, True, False, False]
    assert hc.long_flags("cap16", r).tolist() == [False, True, True, True, True]
    assert hc.long_flags("excess8", r, f).tolist() == [True, True, True, True, True]
    assert hc.long_flags("excess8_gt", r, f).tolist() == [False, False, True, False, False]
    with pytest.raises(ValueError):
        hc.long_flags("excess8", r)
    with pytest.raises(ValueError):
        hc.long_flags("cap32", r)


def _wide(seed=0, M=6, configs=("c1", "c2", "c3", "c4"), N=30):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(M):
        for c in configs:
            for i in range(N):
                long_ = rng.random() < 0.2 + 0.1 * j
                f = int(rng.random() < 0.5)
                rq = int(rng.random() < (0.4 if long_ else 0.3 + 0.05 * j))
                rows.append({"model_key": f"m{j}", "config_id": c, "item_id": str(i), "requested_strict": rq,
                             "forced_strict": f, "cot_strict": int(rng.random() < 0.6),
                             "requested_content": max(rq, int(rng.random() < 0.1)), "forced_content": f,
                             "cot_content": int(rng.random() < 0.62), "long_x": long_, "long_none": False})
    return pd.DataFrame(rows), [f"m{j}" for j in range(M)]


def test_decomposition_variant_parts_choices_and_fresh_configurations():
    from src import baseline_mixture as bm, decision as dc
    wide, panel = _wide()
    disc, val = ["c1", "c2", "c2"], ["c2", "c3", "c4", "c4"]
    v = hc.decomposition_variant(wide, panel, "long_x", disc, val, "content")
    for s in ("all_distinct", "discovery", "validation"):
        p = v["sets"][s]["per_model"]
        assert np.allclose(np.array(p["part_long"]) + np.array(p["part_short"]), np.array(p["shift"]))
    # validation weights: c4 counts twice, c2 once
    x = wide[wide["config_id"].isin(["c2", "c3", "c4"])]
    wt = hc.row_weights(x["config_id"], val)
    assert set(np.unique(wt)) == {1.0, 2.0}
    d = bm.decomposition(pd.Index(panel).get_indexer(x["model_key"]), x["long_x"].to_numpy(),
                         x["requested_content"], x["forced_content"], x["cot_content"], len(panel), wt)
    assert v["sets"]["validation"]["variance_share_long"] == pytest.approx(d["variance_share_long"])
    ch = v["choice"]
    n = dc.third_size(len(panel))
    assert ch["n_chosen"] == n and all(abs(sum(w) - n) < 1e-9 for w in ch["weights"].values())
    assert ch["fresh_configurations"]["n_configurations"] == 2          # c3 and c4 are in no discovery draw
    assert set(ch["validation_means"]["regret"]) == {"forced_gain", "corrected", "corrected_short"}
    assert min(ch["validation_means"]["regret"].values()) >= -1e-9
    assert "observed_vs_corrected" in v["mechanism"]["discovery"]
    assert 0 <= v["w_all_distinct_median"] <= 1


def test_decomposition_variant_with_no_long_rows_reports_edge_cases():
    wide, panel = _wide(1)
    v = hc.decomposition_variant(wide, panel, "long_none", ["c1", "c2"], ["c3", "c4"], "strict")
    assert v["w_discovery_median"] == 0.0
    # with no long row the long part is zero for every model: its share of the variance is 0, reported as such
    assert v["sets"]["all_distinct"]["variance_share_long"] == pytest.approx(0.0)
    # (1 - w) x forced gain equals the forced gain: both choices coincide
    assert v["choice"]["weights"]["corrected"] == v["choice"]["weights"]["forced_gain"]


def test_decomposition_variant_one_fresh_configuration_is_reported():
    wide, panel = _wide(2)
    v = hc.decomposition_variant(wide, panel, "long_x", ["c1", "c2", "c3"], ["c3", "c4"], "strict")
    assert v["choice"]["fresh_configurations"] == {"n_configurations": 1, "note": "fewer than two fresh configurations"}


# ----------------------------------------------------------------------------- the draw for a third rater


def test_allocate_equal_capped_and_complete():
    rng = np.random.default_rng(0)
    a = hc.allocate({"a": 100, "b": 100, "c": 2, "d": 0}, 20, rng)
    assert sum(a.values()) == 20 and a["c"] == 2 and "d" not in a and abs(a["a"] - a["b"]) <= 1
    a2 = hc.allocate({"a": 3, "b": 3, "c": 3}, 7, np.random.default_rng(1))
    assert sum(a2.values()) == 7 and max(a2.values()) == 3 and min(a2.values()) == 2
    with pytest.raises(ValueError):
        hc.allocate({"a": 1, "b": 2}, 4, rng)


def test_stratified_draw_seeded_without_replacement():
    df = pd.DataFrame({"m": ["x"] * 50 + ["y"] * 3 + ["z"] * 40, "v": range(93)})
    d1 = hc.stratified_draw(df, 20, "m", "lab")
    d2 = hc.stratified_draw(df, 20, "m", "lab")
    assert d1.equals(d2) and len(d1) == 20 and d1["v"].is_unique
    assert (d1["m"] == "y").sum() == 3
    assert not d1.equals(hc.stratified_draw(df, 20, "m", "other"))


# ----------------------------------------------------------------------------- scripts


def test_write_manifest_once_identical_kept_different_refused(tmp_path):
    import score_hotpot_content as shc
    p = str(tmp_path / "m.json")
    r = shc.write_manifest(p)
    assert json.load(open(p, encoding="utf-8")) == r
    assert shc.write_manifest(p) == r
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({**r, "scorer_version": "other"}, fh)
    with pytest.raises(SystemExit):
        shc.write_manifest(p)


def test_compare_stored_counts_leaves(tmp_path):
    import run_hotpot_content as rhc
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1.0, "b": [1, 2], "c": {"d": "s"}, "created": "t"}), encoding="utf-8")
    r = rhc.compare_stored({"a": 1.0 + 1e-12, "b": [1, 3], "created": "u"}, str(p))
    assert r["equal"] == 2 and r["different"] == 1 and r["missing"] == 1 and r["examples"] == ["/b/1"]


def _variant_stub():
    wide, panel = _wide(3)
    return {"cap64": {sc: hc.decomposition_variant(wide, panel, "long_x", ["c1", "c2"], ["c3", "c4", "c2"], sc)
                      for sc in ("strict", "content")}}


def test_threshold_rows_and_lines():
    import run_hotpot_content as rhc
    body = {"definitions": {"cap64": "x"}, "check_definitions": {}, "variants": _variant_stub(),
            "checks_cap64_strict": {"validation_variance_share_equals_saved_evaluation": {"here": 0.1, "stored": 0.1},
                                    "refit_choices_equal_saved_choices": {"corrected": True}}}
    rows = rhc.threshold_rows(body)
    assert len(rows) == 2 and {r["score"] for r in rows} == {"strict", "content"}
    assert all("/" in r["rev_forced_all"] for r in rows)
    L = rhc.threshold_lines(body)
    assert any(line.startswith("| cap64 | content |") for line in L) and "exploratory" in L[0]


def test_protocol_ceiling_task_result_with_loaded_arrays(monkeypatch):
    import run_protocol_ceiling as rpc
    from src import sampling
    monkeypatch.setattr(rpc, "KS", (1, 2))
    monkeypatch.setattr(rpc, "N_SPLITS", 5)
    monkeypatch.setattr(rpc, "N_BOOT", 10)
    rng = np.random.default_rng(0)
    D = {sc: {a: rng.uniform(20, 80, (6, 5)) for a in ("requested", "forced", "cot")} for sc in ("strict", "content")}
    r = rpc.task_result("hotpot", sampling.rng_for("t"), loaded=(D, [f"u{i}" for i in range(6)], list("abcde"), 50),
                        head_scoring="content", boot_rng=sampling.rng_for("t", "boot"),
                        scoring_columns={"strict": "s", "content": "c"}, title="T")
    assert r["headline"]["protocols"] == ["content/requested/points", "content/forced/points"]
    assert r["title"] == "T" and r["scorings"] == {"strict": "s", "content": "c"} and r["D"] == 6
    k = "content/requested/points | content/forced/points"
    assert r["headline"]["plugin"] == r["limits"][k]["plugin"]
    r2 = rpc.task_result("hotpot", sampling.rng_for("t"), loaded=(D, [f"u{i}" for i in range(6)], list("abcde"), 50),
                         boot_rng=sampling.rng_for("t", "boot"), scoring_columns={"strict": "s", "content": "c"})
    assert r2["headline"]["protocols"][0] == "strict/requested/points"


# ----------------------------------------------------------------------------- macros


def test_macros_from_a_synthetic_results_file():
    import macros_hotpot_content as mhc
    R = {"accuracy": {"by_arm": {"direct_answer": {"strict_em": 41.83, "lenient_em": 47.3, "content_em": 48.8},
                                 "forced_direct": {"strict_em": 51.5, "content_em": 51.5}},
                      "per_model_gaps": {"strict": {"requested_minus_forced_median": -7.0, "cot_minus_forced_median": 0.6},
                                         "content": {"requested_minus_forced_median": -1.1,
                                                     "cot_minus_forced_median": 0.7}},
                      "requested_rows_changed": {"wrong_to_right": 72719, "right_to_wrong": 0,
                                                 "wrong_to_right_placeholder_share": 0.162}},
         "formats": {"tasks": {"gsm8k": {"by_arm": {"direct_answer": {"n": 10, "placeholder_copy": 1,
                                                                       "last_box_placeholder": 1, "bare_box": 0,
                                                                       "unclosed_box": 0, "unboxed": 2}},
                                         "placeholder_last_box": {"rows": 1322, "would_change_to_correct": 54}}}},
         "thresholds_full": {"variants": {d: _variant_stub()["cap64"] for d in mhc.THR}}}
    m = mhc.hotpot_content_macros(R)
    assert m["hqArmAccReq"] == "41.8" and m["hqArmAccReqContent"] == "48.8" and m["hqArmAccReqLenient"] == "47.3"
    assert m["hqGapReqFdMedianContent"] == "$-$1.1" and m["hqContentUp"] == "72{,}719" and m["hqContentUpPhPct"] == "16"
    assert m["fmtGsmReqUnboxedPct"] == "20.0" and m["fmtGsmNumberBox"] == "1{,}322"
    for t in mhc.THR.values():
        assert "hqDecVarLongAllPct" + t in m and "hqDecVarLongAllPct" + t + "Content" in m
        assert m["hqDecChoiceHolds" + t] in ("yes", "no")
    assert all(k.isalpha() for k in m)


def test_suffixed_keeps_prefixes_and_drops():
    import macros_hotpot_content as mhc
    m = {}
    mhc._suffixed(m, {"hqDecVarLongPct": "1", "hqDecVarDiscPct": "2", "aeGsmM": "3"}, ("hqDec",), "Content",
                  drop=("VarDiscPct",))
    assert m == {"hqDecVarLongPctContent": "1"}


# ----------------------------------------------------------------------------- the stored outputs (skipped without data)

META = os.path.join(ROOT, "data", "processed", "hotpot", "scored_content.json")
RATER = os.path.join(ROOT, "results", "scoring_audit_hotpot", "ratings_rater_c_hotpot.csv")
KEYF = os.path.join(ROOT, "data", "scoring_audit_hotpot", "key_rater_c_hotpot.csv")


@pytest.mark.skipif(not os.path.exists(META), reason="data/processed/hotpot/scored_content.json is absent")
def test_content_table_reproduces_the_reference_values():
    m = json.load(open(META, encoding="utf-8"))
    assert m["n_rows"] == 4176000 and m["strict_correct_not_content_correct"] == 0
    assert m["content_scorer"] == ht.CONTENT_SCORER_VERSION
    b = m["by_arm"]
    # requested content exact match 49.3 under hotpot-content/1.5.0 (answer tags and typography read)
    assert round(b["direct_answer"]["strict_em_pct"], 1) == 41.8 and round(b["direct_answer"]["content_em_pct"], 1) == 49.3
    assert round(b["forced_direct"]["strict_em_pct"], 1) == 51.5 and round(b["forced_direct"]["content_em_pct"], 1) == 51.5


@pytest.mark.skipif(not os.path.exists(RATER), reason="the third rater's file is absent")
def test_rater_c_csv_shows_no_verdict_and_no_model():
    from run_human_audit import RATER3_COLUMNS
    d = pd.read_csv(RATER, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(d.columns) == list(RATER3_COLUMNS) and len(d) == 100
    assert d["audit_id"].is_unique and (d["verdict"] == "").all() and (d["note"] == "").all()
    assert d["order"].tolist() == [str(i) for i in range(1, 101)]
    if os.path.exists(KEYF):
        k = pd.read_csv(KEYF, dtype=str, keep_default_na=False)
        assert set(k["audit_id"]) == set(d["audit_id"])
        assert (k["strict_correct"] != k["content_correct"]).all() and (k["arm"] == "direct_answer").all()
        assert not d.drop(columns=["model_output", "question"]).apply(
            lambda c: c.isin(set(k["model_key"]))).any().any()
