"""Size ladder: registry, decoding per arm with the raised caps and the
token-level cut, the -dk keys answering their base key's file, the DeepSeek renderer and text prompts (first build,
kept for the dropped entries' record), job script and loop arguments, the sharing of one-card jobs with a second account,
requests, pilot and equivalence helpers, and scoring of the cut and uncut readings."""
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from src import d2_io
from src import generation as g
from src import ladder as L

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "config" / "models_ladder.yaml"
MAIN_REGISTRIES = ["config/models.yaml", "config/models_math.yaml", "config/models_hotpot.yaml",
                   "config/models_extension.yaml", "config/models_reasoning_panel.yaml"]
NEW_FIELDS = {"pipeline_parallel_size", "renderer", "prefill_mode", "gpu_memory_utilization", "quantization_source",
              "requests_model_key", "text_only"}
BASH = next((p for p in (r"C:\Program Files\Git\bin\bash.exe", "/usr/bin/bash", "/bin/bash") if os.path.exists(p)), None)


def entry(**kw):
    e = {"family": "X", "hf_id": "org/m", "revision": "a" * 40, "dtype": "bfloat16", "quantization": "none",
         "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": 1,
         "chat_template": {"source": "tokenizer_default", "file": None, "sha256": "b" * 64},
         "thinking_off": {"method": "none_needed", "chat_template_kwargs": {}, "system_prompt_token": None,
                          "rationale": "none"},
         "kernels": "batch_invariant"}
    e.update(kw)
    return e


def echo_registry(tmp_path, keys):
    """The ladder registry with the given keys' template hashes cleared, for the echo backend's own template."""
    reg = yaml.safe_load(open(LADDER, encoding="utf-8"))
    for k in keys:
        reg["dataset2"]["models"][k]["chat_template"]["sha256"] = None
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(reg), encoding="utf-8")
    return p


# ---------------------------------------------------------------- registry

def test_active_and_dropped_keys_parse_with_their_gpu_counts():
    reg = g.load_registry(LADDER)
    want = {"llama-3.1-70b-instruct-fp8": 1, "qwen3.5-122b-a10b-fp8": 2, "ministral-3-8b-instruct-2512": 1,
            "ministral-3-14b-instruct-2512": 1}
    assert set(L.MODELS) == set(want)
    dropped = {"qwen3-235b-a22b-instruct-2507-fp8", "llama-3.3-70b-instruct-fp8", "gpt-oss-120b", "deepseek-v4-flash-0731"}
    assert set(L.DROPPED) == dropped and not (set(L.DROPPED) & set(L.MODELS))
    assert set(reg) == {k + s for k in set(want) | dropped for s in ("", L.FALLBACK_SUFFIX)}
    for k in dropped:
        for key in (k, k + "-dk"):
            assert reg[key]["ladder_status"] == "dropped" and "dropped from the ladder" in reg[key]["dropped_by"]
            g.parse_model_spec(key, reg)                  # kept as the record of the outputs they produced
    for k, n in want.items():
        assert "ladder_status" not in reg[k] and L.MODELS[k]["cards"] == n
        for key in (k, k + L.FALLBACK_SUFFIX):
            s = g.parse_model_spec(key, reg)
            assert s.gpus == n == g.model_gpu_count(key, LADDER)
            assert s.max_model_len == 10240 and g._SHA40.match(s.revision)
        base, dk = g.parse_model_spec(k, reg), g.parse_model_spec(k + "-dk", reg)
        assert base.kernels == "batch_invariant" and dk.kernels == "default"
        assert base.requests_model_key is None and dk.requests_model_key == k
        a, b = base.to_log(), dk.to_log()
        for d in (a, b):
            d.pop("model_key"), d.pop("kernels"), d.pop("requests_model_key", None)
        assert a == b                                     # the fallback differs only in its kernels
    q = g.parse_model_spec("qwen3.5-122b-a10b-fp8", reg)
    assert (q.tensor_parallel_size, q.pipeline_parallel_size, q.text_only) == (2, 1, True)
    assert q.thinking_off.method == "chat_template_kwargs" and q.thinking_off.chat_template_kwargs == {"enable_thinking": False}
    assert q.quantization == "fp8" and q.quantization_source == "checkpoint" and q.engine == {"max_num_seqs": 512}
    l70 = g.parse_model_spec("llama-3.1-70b-instruct-fp8", reg)
    assert l70.quantization == "compressed-tensors" and reg["llama-3.1-70b-instruct-fp8"]["base_model"].startswith(
        "meta-llama/Llama-3.1-70B-Instruct@")
    for k in ("ministral-3-8b-instruct-2512", "ministral-3-14b-instruct-2512"):
        s = g.parse_model_spec(k, reg)
        assert s.hf_id.endswith("-BF16") and s.quantization is None and s.dtype == "bfloat16"
        assert s.tokenizer_mode == "hf" and s.thinking_off.method == "none_needed"


def test_templates_and_thinking_switch_match_the_panel_entries():
    """Llama-3.1-70B's template is the panel's Llama-3.1-8B template; Qwen3.5-122B's is the panel's Qwen3.5-9B template,
    with its thinking switch copied from that entry."""
    reg = g.load_registry(LADDER)
    main = g.load_registry(ROOT / "config" / "models.yaml")
    ext = g.load_registry(ROOT / "config" / "models_extension.yaml")
    assert reg["llama-3.1-70b-instruct-fp8"]["chat_template"]["sha256"] == main["llama-3.1-8b-instruct"]["chat_template"]["sha256"]
    assert reg["qwen3.5-122b-a10b-fp8"]["chat_template"]["sha256"] == ext["qwen3.5-9b"]["chat_template"]["sha256"]
    assert reg["qwen3.5-122b-a10b-fp8"]["thinking_off"] == ext["qwen3.5-9b"]["thinking_off"]
    assert reg["qwen3.5-122b-a10b-fp8"]["engine"] == ext["qwen3.5-9b"]["engine"]
    assert ext["qwen3.5-9b"]["kernels"] == "default"     # the mode the -dk fallback reproduces


def test_pinned_gpt_oss_template_has_no_clock_and_its_registered_hash():
    p = ROOT / "config" / "chat_templates" / "gpt-oss-120b.jinja"
    text = p.read_text(encoding="utf-8")
    assert not g.template_reads_clock(text) and "Current date: 2026-11-22" in text
    reg = g.load_registry(LADDER)
    assert g.template_text_sha256(text) == reg["gpt-oss-120b"]["chat_template"]["sha256"]


@pytest.mark.parametrize("kw,match", [
    ({"pipeline_parallel_size": 0}, "pipeline_parallel_size"),
    ({"pipeline_parallel_size": True}, "pipeline_parallel_size"),
    ({"gpu_memory_utilization": 0.99}, "gpu_memory_utilization"),
    ({"gpu_memory_utilization": "0.9"}, "gpu_memory_utilization"),
    ({"quantization_source": "guess"}, "quantization_source"),
    ({"prefill_mode": "inline"}, "prefill_mode"),
    ({"renderer": {"kind": "deepseek_v4_encoding", "file": "e.py", "sha256": "b" * 64}}, "source is renderer"),
    ({"renderer": {"kind": "other", "file": "e.py", "sha256": "b" * 64}, "prefill_mode": "suffix",
      "chat_template": {"source": "renderer", "file": None, "sha256": "b" * 64}}, "renderer.kind"),
    ({"renderer": {"kind": "deepseek_v4_encoding", "file": "e.py", "sha256": "c" * 64}, "prefill_mode": "suffix",
      "chat_template": {"source": "renderer", "file": None, "sha256": "b" * 64}}, "must equal renderer.sha256"),
    ({"renderer": {"kind": "deepseek_v4_encoding", "file": "e.py", "sha256": "b" * 64},
      "chat_template": {"source": "renderer", "file": None, "sha256": "b" * 64}}, "prefill_mode must be suffix"),
    ({"chat_template": {"source": "renderer", "file": None, "sha256": "b" * 64}}, "source is renderer"),
    ({"requests_model_key": "m"}, "requests_model_key"),
    ({"requests_model_key": "absent"}, "requests_model_key"),
    ({"text_only": "yes"}, "text_only"),
])
def test_new_registry_fields_are_validated(kw, match):
    with pytest.raises(g.RunRefused, match=match):
        g.parse_model_spec("m", {"m": entry(**kw)})


def test_print_gpus_and_print_tp(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_d2
    assert generate_d2.main(["--print-gpus", "--model-key", "qwen3.5-122b-a10b-fp8", "--models-yaml", str(LADDER)]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert generate_d2.main(["--print-tp", "--model-key", "qwen3.5-122b-a10b-fp8", "--models-yaml", str(LADDER)]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_text_only_reaches_vllm(monkeypatch):
    captured = {}
    fake_vllm(monkeypatch, captured)
    reg = g.load_registry(LADDER)
    be = g.make_backend("vllm", g.parse_model_spec("qwen3.5-122b-a10b-fp8", reg), g.load_decoding(ROOT / L.DECODING["math"]))
    be.load()
    assert captured["limit_mm_per_prompt"] == {"image": 0, "video": 0} and captured["tensor_parallel_size"] == 2
    assert captured["max_num_seqs"] == 512 and "quantization" not in captured
    be2 = g.make_backend("vllm", g.parse_model_spec("ministral-3-8b-instruct-2512", reg), be.decoding)
    captured.clear()
    be2.load()
    assert "limit_mm_per_prompt" not in captured and captured["tokenizer_mode"] == "hf"


# ---------------------------------------------------------------- decoding

def test_each_ladder_arm_decodes_as_its_panel_collection_with_the_raised_cap():
    for t in L.TASKS:
        dec = g.load_decoding(ROOT / L.DECODING[t])
        assert dec.confirmed and set(dec.arms) == set(L.TASK_ARMS[t])
        for arm in L.TASK_ARMS[t]:
            f, src_arm = L.MAIN_DECODING[t][arm]
            main = g.load_decoding(ROOT / f)
            want = dict(main.params_for_arm(src_arm))
            if arm in L.RAISED_ARMS:
                assert dec.cut_caps[arm] == want["max_new_tokens"] == L.PANEL_CAPS[t][arm]
                want["max_new_tokens"] = L.RAISED_CAP
            else:
                assert arm not in dec.cut_caps
            assert dec.params_for_arm(arm) == want, (t, arm)
    assert g.load_decoding(ROOT / L.DECODING["hotpot"]).cut_caps == {"direct_answer": 1024, "zero_shot_cot": 1900}
    assert g.load_decoding(ROOT / L.DECODING["gsm8k"]).max_new_tokens["forced_direct"] == 16
    assert g.load_decoding(ROOT / L.DECODING["hotpot"]).max_new_tokens["fewshot_forced"] == 64
    pil = g.load_decoding(ROOT / L.PILOT_DECODING)
    eq = g.load_decoding(ROOT / L.EQUIV_DECODING)
    for t in L.TASKS:
        td = g.load_decoding(ROOT / L.DECODING[t])
        for arm in L.TASK_ARMS[t]:
            assert pil.params_for_arm(f"{t}.{arm}") == td.params_for_arm(arm)
            assert pil.cut_caps.get(f"{t}.{arm}") == td.cut_caps.get(arm)
            if arm in L.RAISED_ARMS:
                assert eq.params_for_arm(f"{t}.{arm}") == {**td.params_for_arm(arm), "max_new_tokens": td.cut_caps[arm]}
    assert not eq.cut_caps and set(eq.arms) == {f"{t}.{a}" for t in L.TASKS for a in L.TASK_ARMS[t] if a in L.RAISED_ARMS}
    assert yaml.safe_load(open(ROOT / L.PILOT_DECODING, encoding="utf-8")) == L.pilot_decoding_dict()
    assert yaml.safe_load(open(ROOT / L.EQUIV_DECODING, encoding="utf-8")) == L.pilot_decoding_dict(equiv=True)
    for t in L.TASKS:
        assert yaml.safe_load(open(ROOT / L.DECODING[t], encoding="utf-8")) == L.task_decoding_dict(t)


def test_cut_caps_are_validated_and_absent_from_old_logs(tmp_path):
    base = yaml.safe_load(open(ROOT / "config" / "decoding.yaml", encoding="utf-8"))
    for f in ("decoding.yaml", "decoding_forced.yaml", "decoding_hotpot.yaml", "decoding_ladder_gsm8k.yaml"):
        d = g.load_decoding(ROOT / "config" / f)
        assert d.cut_caps == {} and "cut_caps" not in d.to_log()
    forced = yaml.safe_load(open(ROOT / "config" / "decoding_forced.yaml", encoding="utf-8"))
    for raw, match in (({**base, "cut_caps": {"nope": 5}}, "without max_new_tokens"),
                       ({**base, "cut_caps": {"direct_answer": 1024}}, "below the arm's max_new_tokens"),
                       ({**base, "cut_caps": {"direct_answer": 0}}, "below the arm's max_new_tokens"),
                       ({**base, "cut_caps": [1]}, "cut_caps must map"),
                       ({**forced, "max_new_tokens": {"forced_direct": 64}, "cut_caps": {"forced_direct": 16}},
                        "without stop strings")):
        p = tmp_path / "d.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(g.RunRefused, match=match):
            g.load_decoding(p)
    d = g.load_decoding(ROOT / L.DECODING["gsm8k"])
    assert d.to_log()["cut_caps"] == {"direct_answer": 1024, "zero_shot_cot": 3072, "fewshot_requested": 1024}
    assert "cut_caps" not in d.params_for_arm("direct_answer")


# ---------------------------------------------------------------- the token-level cut in the runner

def small_run(tmp_path, cut=True, key="ministral-3-8b-instruct-2512", requests_key=None, n=30, out="out"):
    """An echo run of n requests of two arms under a decoding whose direct_answer cap is 300 and cut cap 20."""
    rows = [{"request_id": f"r{i}|{arm}", "model_key": requests_key or key, "arm": arm, "config_id": "c", "item_id": i,
             "messages_json": json.dumps([{"role": "user", "content": f"q{i} {arm}"}])}
            for i in range(n) for arm in ("direct_answer", "forced_direct")]
    req = tmp_path / "req.parquet"
    pd.DataFrame(rows).to_parquet(req)
    dec = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
           "max_new_tokens": {"direct_answer": 300, "forced_direct": 16}, "prefill": {"forced_direct": "\\boxed{"},
           "stop": [], "skip_special_tokens": True}
    if cut:
        dec["cut_caps"] = {"direct_answer": 20}
    dp = tmp_path / "dec.yaml"
    dp.write_text(yaml.safe_dump(dec), encoding="utf-8")
    reg = echo_registry(tmp_path, [key])
    log = g.run_generation(req, key, tmp_path / out, backend="echo", models_path=reg, decoding_path=dp, verbose=False,
                           shard_size=7)
    return log, g.read_outputs(tmp_path / out), req, reg, dp


def test_runner_keeps_a_token_level_cut_beside_the_uncut_output(tmp_path):
    log, df, *_ = small_run(tmp_path)
    assert log["status"] == "complete" and set(g.CUT_COLUMNS) <= set(df.columns)
    da = df[df["arm"] == "direct_answer"]
    long = da[da["n_output_tokens"] > 20]
    short = da[da["n_output_tokens"] <= 20]
    assert len(long) > 0
    assert long["cut_at_panel_cap"].all() and (long["n_output_tokens_cut"] == 20).all()
    assert (long["finish_reason_cut"] == "length").all()
    for _, r in long.iterrows():
        assert r["output_text_cut"] == " ".join(r["output_text"].split()[:20])
        assert len(r["output_token_ids_cut"]) == 20
    assert not short["cut_at_panel_cap"].any() and (short["output_text_cut"] == short["output_text"]).all()
    assert short["output_token_ids_cut"].isna().all()
    fd = df[df["arm"] == "forced_direct"]
    assert (fd["panel_cap"] == 16).all() and not fd["cut_at_panel_cap"].any() and (da["panel_cap"] == 20).all()
    assert log["counts"]["cut_at_panel_cap_all_requests"]["direct_answer"] == len(long)
    assert "cut_caps" in log["decoding"]


def test_cut_equals_a_run_at_the_panel_cap(tmp_path):
    """With the echo backend: the cut copy equals a run whose cap is the panel's."""
    _, cut, *_ = small_run(tmp_path, out="raised")
    reg = echo_registry(tmp_path, ["ministral-3-8b-instruct-2512"])
    dec = yaml.safe_load(open(tmp_path / "dec.yaml", encoding="utf-8"))
    dec.pop("cut_caps")
    dec["max_new_tokens"]["direct_answer"] = 20
    (tmp_path / "panel.yaml").write_text(yaml.safe_dump(dec), encoding="utf-8")
    g.run_generation(tmp_path / "req.parquet", "ministral-3-8b-instruct-2512", tmp_path / "panel", backend="echo",
                     models_path=reg, decoding_path=tmp_path / "panel.yaml", verbose=False)
    pan = g.read_outputs(tmp_path / "panel").set_index("request_id")
    assert not (set(g.CUT_COLUMNS) & set(pan.columns))
    c = cut.set_index("request_id").loc[pan.index]
    assert (c["output_text_cut"] == pan["output_text"]).all()
    assert (c["n_output_tokens_cut"] == pan["n_output_tokens"]).all()
    assert (c["finish_reason_cut"] == pan["finish_reason"]).all()


def test_resume_reads_the_cut_back_from_earlier_shards(tmp_path):
    log, df, req, reg, dp = small_run(tmp_path)
    shards = g.list_shards(tmp_path / "out")
    for sh in shards[len(shards) // 2:]:
        sh.unlink()
    for lp in (tmp_path / "out").glob("run_log-*.json"):
        lp.unlink()
    left = len(g.read_outputs(tmp_path / "out"))
    # a duplicate prompt in the second half is served from the first half's cache, with its cut
    log2 = g.run_generation(req, "ministral-3-8b-instruct-2512", tmp_path / "out", backend="echo", models_path=reg,
                            decoding_path=dp, verbose=False, shard_size=7)
    assert log2["status"] == "complete" and log2["requests_already_done"] == left
    again = g.read_outputs(tmp_path / "out").set_index("request_id").sort_index()
    first = df.set_index("request_id").sort_index()
    for c in ("output_text", "output_text_cut", "n_output_tokens_cut", "cut_at_panel_cap"):
        assert (again[c] == first[c]).all(), c


def test_cut_record_needs_token_ids_past_the_cap():
    r = g.GenResult("a b c", 3, 3, "stop")
    assert g.cut_record(r, 5, 7000, str)["cut_at_panel_cap"] is False
    assert g.cut_record(r, None, 16, str)["panel_cap"] == 16
    with pytest.raises(g.GenerationError, match="token ids"):
        g.cut_record(r, 2, 7000, str)
    c = g.cut_record(g.GenResult("x", 3, 3, "stop", token_ids=(7, 8, 9)), 2, 7000, lambda ids: f"{ids}")
    assert c == {"panel_cap": 2, "cut_at_panel_cap": True, "output_text_cut": "[7, 8]", "n_output_tokens_cut": 2,
                 "finish_reason_cut": "length", "output_token_ids_cut": [7, 8]}
    assert g.GenResult("x", 1, 1, "stop", token_ids=(1,)) == g.GenResult("x", 1, 1, "stop")


def test_fallback_key_answers_its_base_keys_file(tmp_path):
    log, df, *_ = small_run(tmp_path, key="ministral-3-8b-instruct-2512-dk", requests_key="ministral-3-8b-instruct-2512")
    assert log["status"] == "complete" and set(df["model_key"]) == {"ministral-3-8b-instruct-2512-dk"}
    out, info = d2_io.inspect_work_dir(tmp_path / "out", allow_test_backend=True)
    assert info["model_key"] == "ministral-3-8b-instruct-2512-dk"
    with pytest.raises(g.RunRefused, match="one model per run"):
        small_run(tmp_path, key="ministral-3-8b-instruct-2512", requests_key="llama-3.1-70b-instruct-fp8", out="o2")
def test_arm_overrides_are_validated_and_absent_from_old_logs(tmp_path):
    base = yaml.safe_load(open(ROOT / "config" / "decoding.yaml", encoding="utf-8"))
    assert "arm_overrides" not in g.load_decoding(ROOT / "config" / "decoding.yaml").to_log()
    for ov, match in (({"nope": {"stop": ["}"]}}, "without max_new_tokens"),
                      ({"direct_answer": {"stop": "}"}}, "list of non-empty strings"),
                      ({"direct_answer": {"top_p": 0.5}}, "may set only"),
                      ({"direct_answer": {"include_stop_str_in_output": "yes"}}, "true or false")):
        p = tmp_path / "d.yaml"
        p.write_text(yaml.safe_dump({**base, "arm_overrides": ov}), encoding="utf-8")
        with pytest.raises(g.RunRefused, match=match):
            g.load_decoding(p)


# ---------------------------------------------------------------- renderer

FAKE_ENCODER = '''
bos_token = "<B>"
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    assert thinking_mode in ("chat", "thinking")
    out = bos_token
    for m in messages:
        out += "<U>" + m["content"] + "<A>" + ("</think>" if thinking_mode == "chat" else "<think>")
    return out
'''


def fake_spec(tmp_path, **kw):
    f = tmp_path / "snap" / "encoding" / "enc.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(FAKE_ENCODER, encoding="utf-8")
    sha = hashlib.sha256(FAKE_ENCODER.encode("utf-8")).hexdigest()
    e = entry(renderer={"kind": "deepseek_v4_encoding", "file": "encoding/enc.py", "sha256": sha},
              prefill_mode="suffix", chat_template={"source": "renderer", "file": None, "sha256": sha},
              thinking_off={"method": "chat_template_kwargs", "chat_template_kwargs": {"thinking": False},
                            "system_prompt_token": None, "rationale": None}, **kw)
    return g.parse_model_spec("ds", {"ds": e}), tmp_path / "snap"


def test_deepseek_renderer_chat_mode_and_suffix_prefill(tmp_path):
    spec, snap = fake_spec(tmp_path)
    mod, sha = g.load_renderer(spec, snap)
    assert sha == spec.renderer["sha256"]
    r = lambda m: g.render_deepseek_v4(mod, m, g.template_kwargs(spec))  # noqa: E731
    assert r([{"role": "user", "content": "Q"}]) == "<B><U>Q<A></think>"
    assert r([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "\\boxed{"}]) == "<B><U>Q<A></think>\\boxed{"
    assert g.render_deepseek_v4(mod, [{"role": "user", "content": "Q"}], {"thinking": True}) == "<B><U>Q<A><think>"


def test_renderer_file_hash_and_presence_are_checked(tmp_path):
    spec, snap = fake_spec(tmp_path)
    (snap / "encoding" / "enc.py").write_text(FAKE_ENCODER + "\n# changed\n", encoding="utf-8")
    with pytest.raises(g.RunRefused, match="differs from the registered"):
        g.load_renderer(spec, snap)
    with pytest.raises(g.RunRefused, match="not in the snapshot"):
        g.load_renderer(spec, tmp_path / "empty")


def test_check_renderer_against_the_engine_tokenizer(tmp_path):
    spec, snap = fake_spec(tmp_path)
    mod, _ = g.load_renderer(spec, snap)
    render = lambda m: g.render_deepseek_v4(mod, m, g.template_kwargs(spec))  # noqa: E731

    class Tok:
        def __init__(self, text=None, boom=False):
            self.text, self.boom = text, boom

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert kw == {"thinking": False} and add_generation_prompt and not tokenize
            if self.boom:
                raise ValueError("no template")
            return self.text if self.text is not None else render(msgs)
    assert g.check_renderer(render, Tok(), spec)["equal"] is True
    assert g.check_renderer(render, Tok(boom=True), spec)["equal"] is None
    with pytest.raises(g.RunRefused, match="differs from the engine tokenizer"):
        g.check_renderer(render, Tok(text="<B>other"), spec)


REAL = ROOT / "data" / "ladder_ref" / "dsv4" / "encoding" / "encoding_dsv4.py"


@pytest.mark.skipif(not REAL.exists(), reason="the release's encoding_dsv4.py is not on this machine (data/ is not committed)")
def test_real_encoder_gives_the_d26_rendering():
    """The rendering a feasibility check saw for thinking_false and suffix_box."""
    reg = g.load_registry(LADDER)
    spec = g.parse_model_spec("deepseek-v4-flash-0731", reg)
    mod, sha = g.load_renderer(spec, REAL.parents[1])
    msgs = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "\\boxed{"}]
    assert g.render_deepseek_v4(mod, msgs, g.template_kwargs(spec)) == \
        "<｜begin▁of▁sentence｜><｜User｜>Q?<｜Assistant｜></think>\\boxed{"


# ---------------------------------------------------------------- vLLM backend with text prompts

def fake_vllm(monkeypatch, captured):
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: kw

    class Engine:
        def __init__(self):
            self.queue, self.model_config = [], types.SimpleNamespace(quantization="fp8")

        def add_request(self, rid, prompt, sp):
            ids = prompt["prompt_token_ids"]
            text = "out:" + "".join(chr(i) for i in ids[-3:])
            self.queue.append(types.SimpleNamespace(
                request_id=rid, finished=True, prompt=None, prompt_token_ids=ids,
                outputs=[types.SimpleNamespace(text=text, token_ids=[0, 0], finish_reason="stop")]))

        def step(self):
            return [self.queue.pop()] if self.queue else []

    class Tok:
        chat_template = "tmpl"

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": [ord(c) for c in text]}

        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, continue_final_message=False,
                                **kw):
            return "".join(f"[{m['role']}]{m['content']}" for m in msgs) + ("[assistant]" if add_generation_prompt else "")

    class LLM:
        def __init__(self, **kw):
            captured.update(kw)
            self.llm_engine, self.n = Engine(), 0

        def get_tokenizer(self):
            return Tok()

        def generate(self, prompts, sps, use_tqdm=True):
            captured.setdefault("generate_calls", []).append(len(prompts))
            for p, sp in zip(prompts, sps):
                self.llm_engine.add_request(str(self.n), p, sp)
                self.n += 1
            got = []
            while self.llm_engine.queue:
                got += self.llm_engine.step()
            return sorted(got, key=lambda o: int(o.request_id))

        def chat(self, *a, **kw):
            raise AssertionError("text prompts never go through LLM.chat")
    mod.LLM = LLM
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": list(prompt_token_ids)}
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def test_suffix_prefill_with_a_chat_template_goes_as_token_ids(monkeypatch):
    captured = {}
    fake_vllm(monkeypatch, captured)
    spec = g.parse_model_spec("m", {"m": entry(prefill_mode="suffix", pipeline_parallel_size=3,
                                               gpu_memory_utilization=0.95, quantization="fp8",
                                               quantization_source="checkpoint")})
    be = g.make_backend("vllm", spec, g.load_decoding(ROOT / "config" / "decoding_ladder_gsm8k.yaml"))
    be.load()
    assert captured["pipeline_parallel_size"] == 3 and captured["tensor_parallel_size"] == 1
    assert captured["gpu_memory_utilization"] == 0.95 and "quantization" not in captured
    assert be.describe()["quantization"] == {"registered": "fp8", "reported_by_vllm": "fp8", "equal": True}
    convs = [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "\\boxed{"}]]
    assert be.render(convs[1]) == "[user]q2[assistant]\\boxed{"
    params = [be.decoding.params_for_arm("direct_answer"), be.decoding.params_for_arm("forced_direct")]
    seen = {}
    res = be.generate(convs, params, on_result=lambda i, r: seen.__setitem__(i, r), on_tick=lambda: None)
    assert [r.text for r in res] == ["out:nt]", "out:ed{"] and captured["generate_calls"] == [2]
    assert all(r.prompt_echo is None for r in res) and seen == dict(enumerate(res))
    assert be.count_tokens(be.render(convs[1])) == len(be.render(convs[1]))


def test_deepseek_backend_renders_with_its_encoder(monkeypatch, tmp_path):
    captured = {}
    fake_vllm(monkeypatch, captured)
    spec, snap = fake_spec(tmp_path)
    orig = g.load_renderer
    monkeypatch.setattr(g, "load_renderer", lambda s: orig(s, snap))
    be = g.make_backend("vllm", spec, g.load_decoding(ROOT / "config" / "decoding_ladder_math.yaml"))
    with pytest.raises(g.RunRefused, match="differs from the engine tokenizer"):
        be.load()                   # the fake tokenizer renders "[user]...", the encoder "<B><U>...": refused
    assert be.chat_template_sha256() == spec.renderer["sha256"]
    conv = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "\boxed{"}]
    assert be.render(conv) == "<B><U>Q<A></think>\boxed{"
    be.llm, be.tokenizer = sys.modules["vllm"].LLM(), sys.modules["vllm"].LLM().get_tokenizer()
    res = be.generate([conv], [be.decoding.params_for_arm("forced_direct")])
    assert res[0].text == "out:ed{" and res[0].n_prompt_tokens == len(be.render(conv))




# ---------------------------------------------------------------- job and loop scripts

def read(p):
    return (ROOT / p).read_text(encoding="utf-8")


# ---------------------------------------------------------------- sharing the one-card jobs (ladder_share.py)

def share_env(tmp_path, monkeypatch):
    S = tmp_path / "S"
    (S / "out" / "ladder" / "chosen").mkdir(parents=True)
    (S / "claims.tsv").write_text("", encoding="utf-8")
    cat = "\n".join("\t".join(r) for r in [
        ["m-a", "ma", "public", "gsm8k", "1/2", "--gpus=1 --mem=96G", "0-12:00:00"],
        ["m-a", "ma", "public", "gsm8k", "2/2", "--gpus=1 --mem=96G", "0-12:00:00"],
        ["m-a", "ma", "public", "math", "", "--gpus=1 --mem=96G", "0-12:00:00"],
        ["m-b", "mb", "public", "gsm8k", "", "--gpus=1 --mem=64G", "0-12:00:00"],
        ["q", "qq", "reserved", "gsm8k", "", "--gpus=2 --mem=180G", "0-12:00:00"]]) + "\n"
    (S / "ladder_catalog.tsv").write_text(cat, encoding="utf-8")
    monkeypatch.setenv("LD_S", str(S))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ladder_share_t", ROOT / "ops" / "cluster" / "ladder_share.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    subs = []
    monkeypatch.setattr(mod, "submit", lambda r: (subs.append(r["stem"]), str(1000 + len(subs)))[1])
    return S, mod, subs


def run_share(mod, tmp_path, queue, mode="share", capsys=None):
    q = tmp_path / "q.txt"
    q.write_text("".join(" ".join(j) + "\n" for j in queue), encoding="utf-8")
    sys.argv = ["x", str(q), mode]
    mod.main()
    out = capsys.readouterr().out if capsys else ""
    return {ln.split()[1]: ln.split()[2] for ln in out.splitlines() if ln.startswith("ST ")}, out


def write_log(d, status, part=None):
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("run_log-*.json")))
    (d / f"run_log-{n}.json").write_text(json.dumps({"status": status, "inputs": {"part": part, "limit": None}}),
                                         encoding="utf-8")


def test_catalog_lists_every_one_card_job_once():
    p = ROOT / "data" / "requests" / "ladder" / "ladder_catalog.tsv"
    if not p.exists():
        pytest.skip("the ladder's request files are not built on this machine")
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_ladder_requests_t", ROOT / "scripts" / "build_ladder_requests.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    rows = [ln.split("\t") for ln in p.read_text(encoding="utf-8").splitlines()]
    want = [(k, j["task"], j["part"]) for k in L.MODELS for j in L.jobs(k)]
    assert sorted((r[0], r[3], r[4]) for r in rows) == sorted(want)
    for r in rows:
        m = L.MODELS[r[0]]
        assert r[1] == m["short"] and r[2] == m["pool"] and r[6] == "0-12:00:00"
        assert r[5] == B.card_flags(r[0])
    stems = [r[1] + "-" + r[3] + ("-p" + r[4].split("/")[0] if r[4] else "") for r in rows]
    assert len(set(stems)) == len(stems)


# ---------------------------------------------------------------- requests, pilot and equivalence helpers

def fake_sources(tmp_path, keys=("src", "chk")):
    rows = []
    for arm in ("direct_answer", "zero_shot_cot"):
        for c in ("c1", "c2"):
            for i in range(20):
                rows.append({"request_id": f"{c}|{arm}|{i}", "model_key": "src", "arm": arm, "config_id": c,
                             "item_id": i, "messages_json": json.dumps([{"role": "user", "content": f"{c}{i}"}])})
    a = pd.DataFrame(rows)
    d = a[a["arm"] == "direct_answer"]
    f = d.assign(arm="forced_direct", request_id=lambda x: x["request_id"] + "|fd")
    fs = d.assign(arm="forced_direct", request_id=lambda x: x["request_id"] + "|fd|fs",
                  messages_json=lambda x: x["messages_json"].str.replace("c", "five c"))
    fsr = pd.concat([d.assign(request_id=lambda x: "gsm8k/" + x["request_id"] + "|fsr", config_id=lambda x: x["config_id"] + "x"),
                     d.assign(request_id=lambda x: "bbh/" + x["request_id"] + "|fsr")])
    fsr["item_id"] = fsr["item_id"].astype(str)
    for k in keys:
        for sub, df in (("gsm8k", a), ("forced/gsm8k", f), ("fewshot/gsm8k", fs), ("fewshot_requested/collection", fsr)):
            p = tmp_path / "data/requests" / sub
            p.mkdir(parents=True, exist_ok=True)
            df.assign(model_key=k).to_parquet(p / f"{k}.parquet")
    return a


def test_task_requests_take_the_five_shot_arms_and_pilot_rows(tmp_path):
    a = fake_sources(tmp_path)
    df, rec = L.task_requests("gsm8k", "new", root=tmp_path, source_key="src", check_key="chk")
    assert len(df) == 80 + 40 + 40 + 40 and set(df["model_key"]) == {"new"}
    assert rec["by_arm"] == {"direct_answer": 40, "fewshot_forced": 40, "fewshot_requested": 40, "forced_direct": 40,
                             "zero_shot_cot": 40}
    assert df["request_id"].str.startswith("bbh/").sum() == 0           # the other benchmark's five-shot requested rows
    assert df["item_id"].dtype == "int64"
    pil = L.pilot_rows("gsm8k", df)
    assert len(pil) == 5 * 16 and pil["request_id"].str.startswith("gsm8k|").all()
    assert set(pil["arm"]) == {f"gsm8k.{x}" for x in L.TASK_ARMS["gsm8k"]}
    by = pil.groupby("arm")["config_id"].unique()
    assert list(by["gsm8k.direct_answer"]) == ["c1"] and list(by["gsm8k.fewshot_requested"]) == ["c1x"]
    a.assign(model_key="chk", messages_json="x").to_parquet(tmp_path / "data/requests/gsm8k/chk.parquet")
    with pytest.raises(ValueError, match="differ"):
        L.task_requests("gsm8k", "new", root=tmp_path, source_key="src", check_key="chk")


def test_box_helpers_and_verdict_with_five_shot_arms():
    assert L.boxed_answer("a \\boxed{1} b \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert L.boxed_answer("\\boxed{open") is None and L.boxed_answer(None) is None
    assert L.forced_closed("42}") and L.forced_closed("\\frac{1}{3}}") and not L.forced_closed("}") \
        and not L.forced_closed("42 and more")
    arms = ["gsm8k.forced_direct"] * 10 + ["gsm8k.fewshot_forced"] * 10 + ["gsm8k.direct_answer"] * 10 + \
        ["gsm8k.fewshot_requested"] * 10 + ["gsm8k.zero_shot_cot"] * 10
    texts = ["7}"] * 10 + ["7}"] * 8 + ["x"] * 2 + ["\\boxed{7}"] * 20 + ["so \\boxed{7}"] * 10
    df = pd.DataFrame({"arm": arms, "output_text": texts, "finish_reason": ["stop"] * 50, "n_output_tokens": [2] * 50,
                       "prompt_sha256": [str(i) for i in range(50)], "cut_at_panel_cap": [False] * 50,
                       "n_output_tokens_cut": [2] * 50})
    s = L.pilot_summary(df)
    assert s["gsm8k.fewshot_forced"]["rate"] == 0.8 and s["gsm8k.forced_direct"]["identical_outputs_to_different_prompts"] == 10
    assert s["gsm8k.direct_answer"]["past_panel_cap"] == 0 and L.arm_class("fewshot_forced") == "forced"
    assert L.pilot_verdict(s, 50, 50) == (True, [])                      # forced pooled 0.9
    df.loc[df["arm"] == "gsm8k.fewshot_requested", "output_text"] = "no box"
    ok, why = L.pilot_verdict(L.pilot_summary(df), 51, 50)
    assert not ok and "50 of 51" in why[0] and any("requested arms" in w for w in why)


def test_estimate_parts_fit_the_time_limit():
    for k in L.MODELS:
        e = L.estimate(k)
        for t, r in e["tasks"].items():
            assert r["parts"] >= 1 and (L.MODELS[k]["pool"] == "reserved" or L.wall_hours(k, t) / r["parts"] <= L.TARGET_PART_HOURS)
        assert e["gpu_hours"] == pytest.approx(e["wall_hours"] * L.MODELS[k]["cards"], abs=0.2)
        assert len(L.jobs(k)) == sum(r["parts"] for r in e["tasks"].values())
    assert all(j["part"] == "" for j in L.jobs("qwen3.5-122b-a10b-fp8"))


def test_equiv_rows_take_the_cut_ones_then_the_longest():
    req = pd.DataFrame({"request_id": [f"r{i}" for i in range(8)], "arm": ["gsm8k.direct_answer"] * 6 + ["gsm8k.forced_direct"] * 2})
    out = pd.DataFrame({"request_id": [f"r{i}" for i in range(8)], "arm": req["arm"],
                        "cut_at_panel_cap": [False, True, False, False, True, False, False, False],
                        "n_output_tokens": [5, 2000, 9, 7, 1500, 9, 50, 50]})
    rows = L.equiv_rows(req, out, n=4)
    assert list(rows["request_id"]) == ["r1", "r4", "r2", "r5"]


@pytest.mark.skipif(not (ROOT / "data/requests/ladder/gsm8k/ministral-3-8b-instruct-2512.parquet").exists()
                    or not (ROOT / "data/raw/gsm8k").exists(), reason="ladder requests or panel items not on this machine")
def test_scoring_reads_every_task_folder_cut_and_uncut(tmp_path):
    """score_ladder on echo outputs of 30 requests per arm and task: strict, lenient and content columns for all four
    tasks in both readings, the prefill put back in front of the forced outputs by d2_io."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import score_ladder as S
    key = "ministral-3-8b-instruct-2512"
    reg = echo_registry(tmp_path, [key])
    for t in L.TASKS:
        df = pd.read_parquet(ROOT / f"data/requests/ladder/{t}/{key}.parquet").groupby("arm").head(30)
        df.to_parquet(tmp_path / f"{t}.parquet")
        dec = yaml.safe_load(open(ROOT / L.DECODING[t], encoding="utf-8"))
        dec["cut_caps"] = {a: 40 for a in dec["cut_caps"]}
        (tmp_path / f"{t}.yaml").write_text(yaml.safe_dump(dec), encoding="utf-8")
        g.run_generation(tmp_path / f"{t}.parquet", key, tmp_path / "out" / key / t, backend="echo", models_path=reg,
                         decoding_path=tmp_path / f"{t}.yaml", verbose=False)
    summ = S.score_model(key, out_root=tmp_path / "out", proc=tmp_path / "proc", proc_uncut=tmp_path / "proc_u",
                         res_dir=tmp_path / "res", allow_test_backend=True)
    for t in L.TASKS:
        n = 30 * len(L.TASK_ARMS[t])
        for d in ("proc", "proc_u"):
            sc = pd.read_parquet(tmp_path / d / t / f"{key}.parquet")
            assert len(sc) == n and {"strict_correct", "lenient_correct", "content_correct"} <= set(sc.columns)
        cut = pd.read_parquet(tmp_path / "proc" / t / f"{key}.parquet")
        assert summ["tasks"][t]["by_arm"] == {a: 30 for a in sorted(L.TASK_ARMS[t])}
        assert sum(summ["tasks"][t]["cut_at_panel_cap"].values()) > 0
        assert (cut[cut["arm"].isin(L.RAISED_ARMS)]["n_output_tokens"] <= 40).all()

