"""Tests for the native-reasoning request builder (scripts/build_thinking_requests.py)."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_thinking_cli", ROOT / "scripts" / "build_thinking_requests.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def entry(method, kwargs=None):
    return {"hf_id": "x/y", "max_model_len": 4096, "kernels": "batch_invariant",
            "thinking_off": {"method": method, "chat_template_kwargs": kwargs, "system_prompt_token": None,
                             "rationale": None if method != "none_needed" else "no switch"}}


def test_thinking_models_takes_the_enable_thinking_switch_only():
    reg = {"a": entry("chat_template_kwargs", {"enable_thinking": False}), "b": entry("none_needed"),
           "c": entry("chat_template_kwargs", {"reasoning": "low"}), "d": entry("chat_template_kwargs", {"enable_thinking": False})}
    assert bt.thinking_models(reg, ["d", "a", "b", "c"]) == ["d", "a"]


def test_thinking_entry_turns_the_switch_on_and_keeps_the_rest():
    e = entry("chat_template_kwargs", {"enable_thinking": False, "other": 1})
    t = bt.thinking_entry(e, 32768)
    assert t["thinking_off"]["chat_template_kwargs"] == {"enable_thinking": True, "other": 1}
    assert t["max_model_len"] == 32768 and t["kernels"] == e["kernels"] and t["hf_id"] == e["hf_id"]
    assert e["thinking_off"]["chat_template_kwargs"]["enable_thinking"] is False      # the input is not changed
    assert "engine" not in t


def test_thinking_entry_merges_scheduler_limits_over_the_registered_ones():
    e = {**entry("chat_template_kwargs", {"enable_thinking": False}), "engine": {"max_num_seqs": 512}}
    t = bt.thinking_entry(e, 32768, {"max_num_batched_tokens": 8192})
    assert t["engine"] == {"max_num_seqs": 512, "max_num_batched_tokens": 8192}
    assert e["engine"] == {"max_num_seqs": 512}


def test_gemma_4_31b_gets_a_batch_limit_that_fits_one_long_request():
    reg = yaml.safe_load((ROOT / "config" / "models_thinking_gsm8k.yaml").read_text(encoding="utf-8"))["dataset2"]["models"]
    assert reg["gemma-4-31b-it"]["engine"]["max_num_batched_tokens"] == 8192
    assert all("max_num_batched_tokens" not in (v.get("engine") or {}) for k, v in reg.items() if k != "gemma-4-31b-it")


def test_the_panel_has_nine_thinking_checkpoints():
    keys = bt.thinking_models()
    assert len(keys) == 9 and "gemma-4-31b-it" in keys and all(k.startswith(("qwen3", "gemma-4")) for k in keys)


def test_answer_text_reads_after_the_last_end_of_reasoning_marker():
    spec2 = importlib.util.spec_from_file_location("analyze_thinking_cli", ROOT / "scripts" / "analyze_thinking_pilot.py")
    at = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(at)
    assert at.answer_text("<think>try \\boxed{3}</think>\n\nSo \\boxed{4}.") == "\n\nSo \\boxed{4}."
    assert at.answer_text("<|channel>thought\nmaybe 3<channel|>The answer is \\boxed{4}") == "The answer is \\boxed{4}"
    assert at.answer_text("<think>still thinking about \\boxed{3}") is None
    assert at.answer_text("a</think>b</think>c") == "c"
    assert at.ceil_to(4097, 1024) == 5120 and at.ceil_to(4096, 1024) == 4096


def test_run_requests_relabel_the_direct_arm_only(tmp_path, monkeypatch):
    df = pd.DataFrame({"request_id": ["r1", "r2", "r3", "r4"], "model_key": "m", "config_id": ["c1", "c1", "c2", "c2"],
                       "arm": ["direct_answer", "zero_shot_cot"] * 2, "item_id": [1, 1, 2, 2], "messages_json": "[]"})
    src = tmp_path / "m.parquet"
    df.to_parquet(src, index=False)
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "collection_file", lambda task, key: src)
    d, info = bt.run_requests("gsm8k", "m")
    assert d["arm"].tolist() == ["thinking_gsm8k"] * 2 and d["request_id"].tolist() == ["r1|th", "r3|th"]
    assert d["messages_json"].tolist() == ["[]", "[]"] and info["n"] == 2 and info["configs"] == 2
    df.iloc[:3].to_parquet(src, index=False)
    with pytest.raises(SystemExit, match="expected half"):
        bt.run_requests("gsm8k", "m")


def test_pilot_caps_need_the_complete_pilot_and_the_decoding_takes_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    p = tmp_path / "thinking_pilot.json"
    R = {"complete": False, "models_present": ["a"], "models_expected": ["a", "b"],
         "tasks": {"gsm8k": {"cap": 4096, "max_model_len": 5120}}}
    p.write_text(json.dumps(R), encoding="utf-8")
    with pytest.raises(SystemExit, match="1 of 2"):
        bt.pilot_caps("gsm8k", p)
    p.write_text(json.dumps({**R, "complete": True}), encoding="utf-8")
    c = bt.pilot_caps("gsm8k", p)
    assert c["cap"] == 4096 and c["max_model_len"] == 5120 and "decided_by" not in c
    # a partial pilot already at the rule's maximum decides the cap
    p.write_text(json.dumps({**R, "pilot_cap": 30720, "tasks": {"gsm8k": {"cap": 30720, "max_model_len": 31744}}}),
                 encoding="utf-8")
    c = bt.pilot_caps("gsm8k", p)
    assert c["cap"] == 30720 and c["max_model_len"] == 31744 and c["decided_by"] == ["a"]
    p.write_text(json.dumps({**R, "pilot_cap": 30720}), encoding="utf-8")
    with pytest.raises(SystemExit, match="1 of 2"):
        bt.pilot_caps("gsm8k", p)
    out = bt.write_decoding("gsm8k", 4096, tmp_path / "dec.yaml")
    dec = yaml.safe_load((tmp_path / "dec.yaml").read_text(encoding="utf-8"))
    pilot = yaml.safe_load(bt.PILOT_DECODING.read_text(encoding="utf-8"))
    assert dec["max_new_tokens"] == {"thinking_gsm8k": 4096} and out["cap"] == 4096
    assert {k: v for k, v in dec.items() if k != "max_new_tokens"} == {k: v for k, v in pilot.items() if k != "max_new_tokens"}


def load_script(name):
    s = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_answer_bodies_follow_d13():
    st = load_script("score_thinking")
    body, closed = st.answer_bodies(["<think>x</think>So \\boxed{4}", "<think>long and cut", "plain \\boxed{5}"],
                                    ["stop", "length", "stop"])
    assert body == ["So \\boxed{4}", "", "plain \\boxed{5}"] and closed.tolist() == [True, False, False]


def test_interventions_block_on_known_arrays():
    import numpy as np
    rt = load_script("run_thinking")
    rng = np.random.default_rng(0)
    B = rng.normal(50, 10, (12, 5))
    gain = np.array([1.0, 5.0, 9.0, 13.0, 17.0])
    P1 = B + gain + rng.normal(0, 0.5, (12, 5))
    P2 = B + gain[::-1] + rng.normal(0, 0.5, (12, 5))
    same = rt.interventions_block(B, P1, P1)
    assert same["latent_corr"]["estimate"] == pytest.approx(1.0)
    assert same["model_means"]["pairs_same_order"] == 10 == same["model_means"]["pairs"]
    rev = rt.interventions_block(B, P1, P2)
    assert rev["latent_corr"]["estimate"] < -0.9 and rev["model_means"]["pairs_same_order"] == 0
    assert rev["model_means"]["spearman"] == pytest.approx(-1.0)


@pytest.mark.skipif(not (ROOT / "data" / "requests" / "thinking" / "pilot" / "build_record.json").is_file(),
                    reason="pilot requests not built")
def test_pilot_files_match_their_record_and_registry():
    rec = json.loads((ROOT / "data" / "requests" / "thinking" / "pilot" / "build_record.json").read_text(encoding="utf-8"))
    reg = yaml.safe_load((ROOT / "config" / "models_thinking_pilot.yaml").read_text(encoding="utf-8"))["dataset2"]["models"]
    assert sorted(rec["models"]) == sorted(reg) == sorted(bt.thinking_models())
    for k, m in rec["models"].items():
        df = pd.read_parquet(ROOT / "data" / "requests" / "thinking" / "pilot" / m["requests_file"])
        assert len(df) == 800 and df["request_id"].is_unique and df["request_id"].str.endswith("|th").all()
        assert df.groupby("arm")["config_id"].nunique().to_dict() == {"thinking_gsm8k": 2, "thinking_math": 2}
        assert reg[k]["thinking_off"]["chat_template_kwargs"]["enable_thinking"] is True and reg[k]["max_model_len"] == 32768
    dec = yaml.safe_load((ROOT / "config" / "decoding_thinking_pilot.yaml").read_text(encoding="utf-8"))
    assert dec["status"] == "confirmed" and dec["skip_special_tokens"] is False and dec["temperature"] == 0.0
    assert set(dec["max_new_tokens"]) == {"thinking_gsm8k", "thinking_math"}


# ---------------------------------------------------------------- the pilot's exclusion rule
spec_ap = importlib.util.spec_from_file_location("analyze_thinking_pilot_cli", ROOT / "scripts" / "analyze_thinking_pilot.py")
ap = importlib.util.module_from_spec(spec_ap)
spec_ap.loader.exec_module(ap)


def pilot_model(key, cut_g, cut_m, p99=(1000.0, 2000.0), n=400):
    return {"model_key": key, "tasks": {"gsm8k": {"n": n, "cut_at_pilot_cap": cut_g, "p99": p99[0]},
                                        "math": {"n": n, "cut_at_pilot_cap": cut_m, "p99": p99[1]}}}


def test_exclusion_takes_more_than_half_of_all_pilot_outputs_at_the_cap():
    X = ap.exclusion([pilot_model("loops", 347, 320), pilot_model("half", 400, 0), pilot_model("short", 3, 2),
                      pilot_model("one_task", 401, 0)])
    assert X["loops"]["excluded"] and X["loops"]["share"] == pytest.approx(667 / 800)
    assert not X["half"]["excluded"]                       # exactly half is not more than half
    assert not X["short"]["excluded"] and X["short"]["at_cap"] == 5 and X["short"]["outputs"] == 800
    assert X["one_task"]["excluded"]                       # both tasks pooled


def test_caps_come_from_the_models_given():
    longest = {"gsm8k": 428, "math": 1274}
    kept = [pilot_model("a", 0, 0, (3754.0, 25702.0)), pilot_model("b", 0, 0, (8737.0, 16442.0))]
    caps = ap.task_caps(kept, longest)
    assert caps["gsm8k"]["cap"] == 9216 and caps["math"]["cap"] == 26624
    assert caps["gsm8k"]["max_model_len"] == 10240 and caps["math"]["max_model_len"] == 28672
    assert ap.task_caps(kept + [pilot_model("c", 0, 0, (30720.0, 30720.0))], longest)["math"]["cap"] == ap.PILOT_CAP


@pytest.mark.skipif(not (ROOT / "results" / "dataset2" / "gsm8k30.json").is_file(), reason="GSM8K-29 results missing")
def test_run_models_leave_out_the_excluded_checkpoints():
    every = bt.run_models("gsm8k", excluded={})
    assert "qwen3.5-9b" in every
    assert bt.run_models("gsm8k", excluded={"qwen3.5-9b": 0.83}) == [k for k in every if k != "qwen3.5-9b"]


def test_apply_exclusions_moves_the_entry_and_rewrites_the_registry(tmp_path, monkeypatch):
    rec = {"label": "thinking", "tasks": {
        "gsm8k": {"pilot": {"cap": 30720, "max_model_len": 31744}, "registry": {"path": "config/r.yaml", "sha256": "0"},
                  "models": {"a": {"requests_file": "a.parquet"}, "b": {"requests_file": "b.parquet"}}},
        "math": {"pilot": {"cap": 30720, "max_model_len": 32768}, "registry": {"path": "config/m.yaml", "sha256": "1"},
                 "models": {"a": {"requests_file": "a.parquet"}}}}}
    p = tmp_path / "build_record.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    calls = []
    monkeypatch.setattr(bt, "write_registry", lambda path, keys, mml, note: calls.append((path.name, keys, mml)) or
                        {"path": "config/r.yaml", "sha256": "new"})
    out = bt.apply_exclusions(p, excluded={"b": 0.83})
    g = out["tasks"]["gsm8k"]
    assert list(g["models"]) == ["a"] and g["excluded"]["b"] == {"requests_file": "b.parquet", "pilot_cap_share": 0.83}
    assert g["registry"]["sha256"] == "new" and calls == [("r.yaml", ["a"], 31744)]
    assert out["tasks"]["math"]["registry"]["sha256"] == "1"           # a task without the checkpoint is untouched
    assert json.loads(p.read_text(encoding="utf-8")) == out


def test_scoring_with_missing_folders_still_marks_the_scored_folders_complete(tmp_path, monkeypatch):
    """--allow-missing leaves out folders not yet copied back, but never scores a folder as incomplete:
    run_thinking.py --partial refuses a model scored with --allow-incomplete."""
    sp = importlib.util.spec_from_file_location("score_thinking_cli", ROOT / "scripts" / "score_thinking.py")
    st = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(st)
    import score_d2
    (tmp_path / "gsm8k" / "done-model").mkdir(parents=True)
    monkeypatch.setattr(st, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(st, "PROCESSED", str(tmp_path / "processed"))
    monkeypatch.setattr(st, "derived_record", lambda task: (str(tmp_path / "req" / "build_record.json"),
                                                            {"models": {"done-model": {}, "running-model": {}}}))
    monkeypatch.setattr(st, "scorer", lambda task: (None, {}, None, "test"))
    monkeypatch.setattr(st.d2_io, "load_build_records", lambda d: [])
    seen = {}

    def work_dirs(dirs, *a, allow_incomplete, **kw):
        seen["dirs"], seen["work"] = [Path(d).name for d in dirs], allow_incomplete
        return pd.DataFrame(), {}

    def write(args, *a, **kw):
        seen["meta"] = args.allow_incomplete
        return {"out": "x", "n_rows": 0, "models": ["done-model"], "scorer_version": "v"}
    monkeypatch.setattr(st.d2_io, "score_work_dirs", work_dirs)
    monkeypatch.setattr(score_d2, "_write", write)
    monkeypatch.setattr(pd, "concat", lambda *a, **kw: pd.DataFrame({"closed": []}))
    with pytest.raises(SystemExit, match="running-model"):
        st.score_task("gsm8k", allow_missing=False)
    st.score_task("gsm8k", allow_missing=True)
    assert seen == {"dirs": ["done-model"], "work": False, "meta": False}
