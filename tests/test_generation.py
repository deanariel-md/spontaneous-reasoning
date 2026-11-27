"""Tests for the Dataset 2 generation runner (src/generation.py), EchoBackend only.

Nothing here needs vLLM, torch or a GPU; the module must import on Windows without them.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from src import generation as g

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
ARMS = ("direct_answer", "zero_shot_cot")
KEY = "m1"


# ---------------------------------------------------------------- helpers

def model_entry(**over):
    e = {"family": "test", "hf_id": "org/model", "revision": SHA, "dtype": "bfloat16",
         "quantization": "none", "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": 2,
         "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
         "thinking_off": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False},
                          "system_prompt_token": None, "rationale": None}, "kernels": "batch_invariant"}
    e.update(over)
    return e


def decoding_dict(**over):
    d = {"status": "confirmed", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
         "max_new_tokens": {"direct_answer": 400, "zero_shot_cot": 400}, "stop": [],
         "include_stop_str_in_output": False, "skip_special_tokens": True,
         "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                    "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}
    d.update(over)
    return d


class Cfg:
    def __init__(self, tmp_path, entry=None, decoding=None):
        self.models = tmp_path / "models.yaml"
        self.decoding = tmp_path / "decoding.yaml"
        self.budget = tmp_path / "budget.yaml"
        self.write(entry or model_entry(), decoding or decoding_dict())
        self.budget.write_text(yaml.safe_dump({"cap": {"money_usd": None, "gpu_hours": None},
                                               "dataset2_allocation": {"discovery_units_per_arm": None,
                                                                       "validation_units_per_arm": None}}))

    def write(self, entry, decoding):
        self.models.write_text(yaml.safe_dump({"dataset2": {"models": {KEY: entry}}}))
        self.decoding.write_text(yaml.safe_dump(decoding))

    def spec(self):
        return g.parse_model_spec(KEY, g.load_registry(self.models))

    def dec(self):
        return g.load_decoding(self.decoding)

    def echo(self, **kw):
        return g.EchoBackend(self.spec(), self.dec(), **kw)

    def run(self, requests, out_dir, backend_obj=None, **kw):
        kw.setdefault("shard_size", 1000)
        return g.run_generation(requests, KEY, out_dir, backend="echo", backend_obj=backend_obj,
                                models_path=self.models, decoding_path=self.decoding,
                                budget_path=self.budget, verbose=False, **kw)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 2, "devices": [{"name": "Fake A100", "memory_mib": 81920}] * 2, "driver_version": "580.1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})
    return Cfg(tmp_path)


ORIGINAL_FLASHINFER_IMPORTABLE = g._flashinfer_importable


@pytest.fixture(autouse=True)
def fi(monkeypatch):
    """Deterministic FlashInfer/CUDA discovery for every test: flashinfer not importable, no CUDA_HOME or
    CUDA_PATH, no nvcc on PATH, no /usr/local/cuda. Tests change `fi` to move the state."""
    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "CUDA_HOME", "CUDA_PATH"):
        monkeypatch.delenv(var, raising=False)
    state = {"importable": False, "nvcc": None, "usr_local_cuda": False}
    monkeypatch.setattr(g, "_flashinfer_importable", lambda: state["importable"])
    monkeypatch.setattr(g, "_which", lambda name: state["nvcc"] if name == "nvcc" else None)
    monkeypatch.setattr(g, "_path_exists", lambda path: state["usr_local_cuda"] and path == "/usr/local/cuda")
    return state


@pytest.fixture(autouse=True)
def kernel_switches(monkeypatch):
    """No kernel switch in the environment, and none left behind: the runner sets them in os.environ."""
    for name in g.KERNEL_SWITCHES:
        monkeypatch.setenv(name, "0")      # recorded, so teardown also removes what the runner set
        monkeypatch.delenv(name)


def make_requests(path, *, n_items=4, configs=("c0", "c1", "c2"), arms=ARMS, same_as=None, model_key=KEY,
                  item_ints=True, content=None):
    """Request file with one skeleton per config; configs in `same_as` reuse another config's text."""
    same_as = {"c2": "c0"} if same_as is None else same_as
    rows = []
    for c in configs:
        skel = same_as.get(c, c)
        for arm in arms:
            for it in range(n_items):
                text = content(skel, arm, it) if content else f"skeleton {skel} for {arm} question {it}"
                rows.append({"request_id": f"{c}|{arm}|{it}", "model_key": model_key, "arm": arm, "config_id": c,
                             "item_id": it if item_ints else f"gsm{it}",
                             "messages_json": json.dumps([{"role": "user", "content": text}])})
    pd.DataFrame(rows).to_parquet(path)
    return path


def read_log(work_dir):
    logs = sorted(Path(work_dir).glob("run_log-*.json")) or sorted(Path(work_dir).glob("throughput_log-*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in logs]


# ---------------------------------------------------------------- module hygiene

def test_module_imports_without_vllm_or_torch():
    code = "import sys, src.generation; assert 'vllm' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_vllm_backend_is_lazy(cfg):
    b = g.make_backend("vllm", cfg.spec(), cfg.dec())     # constructing must not import vllm
    assert isinstance(b, g.VLLMBackend)
    assert "vllm" not in sys.modules


# ---------------------------------------------------------------- registry and refusals

def test_committed_template_entry_is_refused():
    reg = g.load_registry()
    assert "_template" in reg
    with pytest.raises(g.RunRefused, match="template"):
        g.parse_model_spec("_template", reg)
    with pytest.raises(g.RunRefused, match="placeholder"):
        g.parse_model_spec("t", {"t": reg["_template"]})   # same content under a normal key


@pytest.mark.parametrize("revision", ["PLACEHOLDER_REVISION", "main", "abc123", "A" * 40, None])
def test_placeholder_or_inexact_revision_refuses_run(tmp_path, monkeypatch, revision):
    c = Cfg(tmp_path, entry=model_entry(revision=revision))
    req = make_requests(tmp_path / "req.parquet")
    out = tmp_path / "out"
    with pytest.raises(g.RunRefused, match="revision"):
        c.run(req, out)
    assert not out.exists()   # refused before anything was written


@pytest.mark.parametrize("thinking, text", [
    ({"method": "none_needed", "rationale": None}, "rationale"),
    ({"method": "chat_template_kwargs", "chat_template_kwargs": {}}, "chat_template_kwargs"),
    ({"method": "system_prompt_token", "system_prompt_token": "PLACEHOLDER"}, "system_prompt_token"),
    ({"method": "unknown"}, "method"),
])
def test_thinking_off_spec_is_validated(tmp_path, thinking, text):
    c = Cfg(tmp_path, entry=model_entry(thinking_off=thinking))
    with pytest.raises(g.RunRefused, match=text):
        c.spec()


def test_unknown_model_key_lists_known(cfg):
    with pytest.raises(g.RunRefused, match="m1"):
        g.parse_model_spec("nope", g.load_registry(cfg.models))


def test_decoding_must_be_greedy_and_capped_per_arm(tmp_path):
    with pytest.raises(g.RunRefused, match="temperature"):
        Cfg(tmp_path, decoding=decoding_dict(temperature=0.7)).dec()
    with pytest.raises(g.RunRefused, match="max_new_tokens"):
        Cfg(tmp_path, decoding=decoding_dict(max_new_tokens={"direct_answer": 0})).dec()


def test_committed_decoding_is_the_confirmed_choice_of_30_october():
    d = g.load_decoding()
    assert d.confirmed and set(d.arms) == set(ARMS)
    assert d.max_new_tokens == {"direct_answer": 1024, "zero_shot_cot": 3072}
    assert (d.temperature, d.top_p, d.top_k) == (0.0, 1.0, 0)


def test_vllm_run_gates(tmp_path):
    spec = Cfg(tmp_path).spec()
    pending = g.load_decoding(Cfg(tmp_path, decoding=decoding_dict(status="PENDING_REVIEW")).decoding)
    ok = g.load_decoding(Cfg(tmp_path).decoding)
    budget = tmp_path / "budget.yaml"
    with pytest.raises(g.RunRefused, match="PENDING_REVIEW"):
        g.preflight(spec, pending, backend="vllm", throughput=False, limit=None, budget_path=budget)
    with pytest.raises(g.RunRefused, match="sha256"):
        g.preflight(spec, ok, backend="vllm", throughput=False, limit=None, budget_path=budget)
    frozen = g.parse_model_spec(KEY, {KEY: model_entry(chat_template={"source": "tokenizer_default", "file": None,
                                                                      "sha256": "b" * 64})})
    with pytest.raises(g.RunRefused, match="budget gate"):
        g.preflight(frozen, ok, backend="vllm", throughput=False, limit=None, budget_path=budget)
    with pytest.raises(g.RunRefused, match="budget gate"):
        g.preflight(frozen, ok, backend="vllm", throughput=False, limit=g.PILOT_MAX_REQUESTS + 1, budget_path=budget)
    g.preflight(frozen, ok, backend="vllm", throughput=False, limit=100, budget_path=budget)    # pilot
    g.preflight(spec, pending, backend="vllm", throughput=True, limit=100, budget_path=budget)  # throughput tranche
    g.preflight(spec, pending, backend="echo", throughput=False, limit=None, budget_path=budget)
    budget.write_text(yaml.safe_dump({"cap": {"money_usd": 100, "gpu_hours": None},
                                      "dataset2_allocation": {"discovery_units_per_arm": 12,
                                                              "validation_units_per_arm": 12}}))
    g.preflight(frozen, ok, backend="vllm", throughput=False, limit=None, budget_path=budget)


def test_vllm_run_refused_before_any_work(cfg, tmp_path):
    cfg.write(model_entry(), decoding_dict(status="PENDING_REVIEW"))
    out = tmp_path / "out"
    with pytest.raises(g.RunRefused):
        g.run_generation(make_requests(tmp_path / "r.parquet"), KEY, out, backend="vllm", models_path=cfg.models,
                         decoding_path=cfg.decoding, budget_path=cfg.budget, verbose=False)
    assert not out.exists() and "vllm" not in sys.modules


def test_chat_template_hash_mismatch_is_refused_and_match_is_accepted(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")
    cfg.write(model_entry(chat_template={"source": "tokenizer_default", "file": None, "sha256": "0" * 64}),
              decoding_dict())
    with pytest.raises(g.RunRefused, match="chat template"):
        cfg.run(req, tmp_path / "o1")
    cfg.write(model_entry(chat_template={"source": "tokenizer_default", "file": None,
                                         "sha256": cfg.echo().chat_template_sha256()}), decoding_dict())
    log = cfg.run(req, tmp_path / "o2")
    assert log["chat_template"]["observed_sha256"] == log["chat_template"]["registered_sha256"]


def test_a_template_that_reads_the_clock_is_refused(cfg, tmp_path):
    class Clock(g.EchoBackend):
        def chat_template_text(self):
            return '{%- set today = strftime_now("%Y-%m-%d") %}{{ messages[0]["content"] }}'

    assert g.template_reads_clock(Clock(None, None).chat_template_text()) and not g.template_reads_clock(None)
    assert not g.template_reads_clock("{{ date_string }}")
    assert not g.template_reads_clock('{#- replaces `set today = strftime_now(...)` -#}\n{%- set today = "2026-10-30" %}')
    assert g.template_reads_clock('{# header #}{%- set today = strftime_now("%Y") %}{# end #}')
    pinned = Path(__file__).resolve().parents[1] / "config/chat_templates/mistral-small-3.1-24b-instruct-2503.jinja"
    assert "strftime_now" in pinned.read_text(encoding="utf-8") and not g.template_reads_clock(
        pinned.read_text(encoding="utf-8"))
    req = make_requests(tmp_path / "r.parquet")
    with pytest.raises(g.RunRefused, match="strftime_now"):
        g.run_generation(req, KEY, tmp_path / "o", backend="echo", models_path=cfg.models, decoding_path=cfg.decoding,
                         budget_path=cfg.budget, verbose=False, backend_obj=Clock(cfg.spec(), cfg.dec()))
    log = cfg.run(req, tmp_path / "o2")
    assert log["chat_template"]["reads_clock"] is False


# ---------------------------------------------------------------- request validation

def test_request_validation(cfg, tmp_path):
    good = make_requests(tmp_path / "good.parquet")
    df = pd.read_parquet(good)

    def refuses(frame, match):
        p = tmp_path / "bad.parquet"
        frame.to_parquet(p)
        with pytest.raises(g.RunRefused, match=match):
            cfg.run(p, tmp_path / "o")

    refuses(df.drop(columns=["messages_json"]), "lacks columns")
    refuses(pd.concat([df, df.iloc[:1]]), "not unique")
    refuses(df.assign(model_key="other"), "one model per run")
    refuses(df.assign(arm="mystery"), "no max_new_tokens")
    refuses(df.assign(messages_json="not json"), "invalid messages_json")
    refuses(df.assign(messages_json=json.dumps([{"role": "assistant", "content": "x"}])), "last message")
    refuses(df.assign(messages_json=json.dumps([{"role": "user", "content": 3}])), "invalid messages_json")


def test_prompt_exceeding_max_model_len_is_refused(tmp_path, monkeypatch):
    c = Cfg(tmp_path, entry=model_entry(max_model_len=450))
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    req = make_requests(tmp_path / "r.parquet", n_items=1, configs=("c0",), arms=("direct_answer",),
                        content=lambda s, a, i: " ".join(["w"] * 100))
    with pytest.raises(g.RunRefused, match="max_model_len"):
        c.run(req, tmp_path / "o")


# ---------------------------------------------------------------- dedupe, caching, schema

def test_dedupe_generates_identical_inputs_once(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")           # 3 configs x 2 arms x 4 items; c2 repeats c0
    be = cfg.echo()
    log = cfg.run(req, tmp_path / "out", backend_obj=be)
    assert log["status"] == "complete"
    assert log["requests_this_run"] == 24
    assert log["totals"]["unique_generations"] == 16
    assert log["totals"]["cached_calls"] == 8
    assert sum(be.generate_calls) == 16
    out = g.read_outputs(tmp_path / "out")
    assert len(out) == 24 and out["request_id"].is_unique
    assert int(out["cached"].sum()) == 8
    for (_, _), grp in out.groupby(["arm", "prompt_sha256"]):   # identical prompts share one output
        assert grp["output_text"].nunique() == 1
        assert (~grp["cached"]).sum() == 1
    assert log["counts"]["cached_calls"]["direct_answer"] == 4


def test_cache_key_separates_arms_with_different_caps(tmp_path, monkeypatch):
    c = Cfg(tmp_path, decoding=decoding_dict(max_new_tokens={"direct_answer": 350, "zero_shot_cot": 400}))
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    req = make_requests(tmp_path / "r.parquet", n_items=2, configs=("c0",),
                        content=lambda s, a, i: f"same text for both arms {i}")
    log = c.run(req, tmp_path / "o")
    assert log["totals"]["unique_generations"] == 4 and log["totals"]["cached_calls"] == 0


def test_cache_key_is_a_function_of_model_prompt_and_decoding(tmp_path):
    d = Cfg(tmp_path, decoding=decoding_dict(max_new_tokens={"direct_answer": 128, "zero_shot_cot": 400})).dec()
    cot = d.params_for_arm("zero_shot_cot")
    a = g.cache_key("org/m", SHA, g.sha256_text("p"), cot)
    assert a == g.cache_key("org/m", SHA, g.sha256_text("p"), dict(cot))
    assert a != g.cache_key("org/m", "b" * 40, g.sha256_text("p"), cot)
    assert a != g.cache_key("org/m2", SHA, g.sha256_text("p"), cot)
    assert a != g.cache_key("org/m", SHA, g.sha256_text("q"), cot)
    assert a != g.cache_key("org/m", SHA, g.sha256_text("p"), d.params_for_arm("direct_answer"))


@pytest.mark.parametrize("item_ints", [True, False])
def test_output_schema(cfg, tmp_path, item_ints):
    req = make_requests(tmp_path / "r.parquet", item_ints=item_ints)
    cfg.run(req, tmp_path / "out")
    shard = sorted((tmp_path / "out").glob("shard-*.parquet"))[0]
    table = pq.read_table(shard)
    assert tuple(table.column_names) == g.OUTPUT_COLUMNS
    s = {f.name: str(f.type) for f in table.schema}
    assert s["request_id"] == "string" and s["cached"] == "bool"
    assert s["n_prompt_tokens"] == "int64" and s["n_output_tokens"] == "int64"
    assert s["item_id"] == ("int64" if item_ints else "string")
    out = table.to_pandas()
    assert set(out["finish_reason"]) <= {"stop", "length"}
    assert (out["n_output_tokens"] >= 1).all() and (out["n_prompt_tokens"] >= 1).all()
    assert out["prompt_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    meta = table.schema.metadata
    assert meta[b"run_fingerprint"] and meta[b"model_key"] == KEY.encode()


def test_prompt_sha256_is_hash_of_rendered_prompt_with_thinking_off(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", n_items=1, configs=("c0",))
    cfg.run(req, tmp_path / "out")
    out = g.read_outputs(tmp_path / "out")
    be = cfg.echo()
    for _, r in out.iterrows():
        msgs = json.loads(pd.read_parquet(req).set_index("request_id").loc[r["request_id"], "messages_json"])
        assert r["prompt_sha256"] == g.sha256_text(be.render(msgs))
        assert '"enable_thinking":false' in be.render(msgs)       # the pinned kwargs reach both arms


def test_thinking_off_methods_change_the_rendered_prompt_in_both_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    req = make_requests(tmp_path / "r.parquet", n_items=2, configs=("c0",))
    shas = {}
    variants = {
        "kwargs": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False}},
        "token": {"method": "system_prompt_token", "system_prompt_token": "/no_think"},
        "none": {"method": "none_needed", "rationale": "The model has no reasoning mode."},
    }
    for name, to in variants.items():
        c = Cfg(tmp_path, entry=model_entry(thinking_off=to))
        c.run(req, tmp_path / name)
        out = g.read_outputs(tmp_path / name)
        shas[name] = set(out["prompt_sha256"])
        assert out["arm"].nunique() == 2
    assert not (shas["kwargs"] & shas["token"]) and not (shas["kwargs"] & shas["none"]) \
        and not (shas["token"] & shas["none"])
    spec = Cfg(tmp_path, entry=model_entry(thinking_off=variants["token"])).spec()
    assert g.effective_messages([{"role": "user", "content": "q"}], spec)[0] == {"role": "system", "content": "/no_think"}
    appended = g.effective_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], spec)
    assert appended[0]["content"] == "s\n/no_think" and len(appended) == 2


def test_truncation_is_recorded_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    c = Cfg(tmp_path, decoding=decoding_dict(max_new_tokens={"direct_answer": 3, "zero_shot_cot": 400}))
    req = make_requests(tmp_path / "r.parquet", same_as={})
    log = c.run(req, tmp_path / "o")
    out = g.read_outputs(tmp_path / "o")
    direct = out[out["arm"] == "direct_answer"]
    assert (direct["finish_reason"] == "length").all() and (direct["n_output_tokens"] == 3).all()
    assert log["counts"]["truncated_generated"]["direct_answer"] == len(direct)
    assert log["counts"]["truncated_generated"]["zero_shot_cot"] == 0
    assert log["totals"]["truncated_all_requests"] == int((out["finish_reason"] == "length").sum())


# ---------------------------------------------------------------- resume

def test_resume_skips_done_ids_and_reuses_outputs(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")
    out = tmp_path / "out"
    first = cfg.run(req, out, limit=8)            # config c0 rows only: sorted by config, arm, item
    assert first["requests_this_run"] == 8 and first["status"] == "complete"
    done = set(g.read_outputs(out)["request_id"])
    assert all(r.startswith("c0|") for r in done)
    be = cfg.echo()
    second = cfg.run(req, out, backend_obj=be)
    assert second["requests_already_done"] == 8 and second["requests_this_run"] == 16
    assert second["totals"]["unique_generations"] == 8          # c1 only; c2 repeats c0
    assert second["totals"]["cached_calls"] == 8
    assert second["totals"]["cached_from_prior_shards"] == 8
    assert sum(be.generate_calls) == 8
    final = g.read_outputs(out)
    assert len(final) == 24 and final["request_id"].is_unique
    assert set(final.loc[final["request_id"].isin(done), "cached"]) == {False}
    third = cfg.run(req, out, backend_obj=be)
    assert third["requests_this_run"] == 0 and sum(be.generate_calls) == 8   # nothing left; model not loaded
    assert third["status"] == "complete" and "model_load_s" not in third


def _runs(out):
    return [{"status": lg["status"], "limit": lg["inputs"]["limit"],
             "part": tuple(lg["inputs"]["part"]) if lg["inputs"].get("part") else None}
            for lg in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(out).glob("run_log-*.json")))]


def test_parts_cover_the_file_once_into_one_folder(cfg, tmp_path):
    from src import d2_io
    req = make_requests(tmp_path / "r.parquet")             # c2 repeats c0's text
    out = tmp_path / "out"
    logs = []
    for k in (1, 2, 3):
        logs.append(cfg.run(req, out, part=(k, 3)))
        assert logs[-1]["status"] == "part_complete" and logs[-1]["inputs"]["part"] == [k, 3]
        assert d2_io.runs_cover_file(_runs(out)) is (k == 3)
    final = g.read_outputs(out)
    assert final["request_id"].is_unique and set(final["request_id"]) == set(pd.read_parquet(req)["request_id"])
    assert sum(lg["totals"]["unique_generations"] for lg in logs) == 16    # a repeated text stays in one part
    again = cfg.run(req, out, part=(2, 3))
    assert again["requests_this_run"] == 0 and again["status"] == "part_complete"


def test_part_is_checked_and_does_not_combine_with_limit(cfg, tmp_path):
    assert g.parse_part("2/4") == (2, 4) and g.parse_part(None) is None
    for bad in ("0/3", "4/3", "1/1", "x", "1/2/3"):
        with pytest.raises(g.RunRefused):
            g.parse_part(bad)
    req = make_requests(tmp_path / "r.parquet")
    with pytest.raises(g.RunRefused, match="--part"):
        cfg.run(req, tmp_path / "out", limit=4, part=(1, 2))


def test_runs_cover_file_needs_a_whole_run_or_every_part_of_one_split():
    from src import d2_io
    whole = {"status": "complete", "limit": None, "part": None}
    p = lambda k, n, s="part_complete": {"status": s, "limit": None, "part": (k, n)}  # noqa: E731
    assert d2_io.runs_cover_file([whole])
    assert not d2_io.runs_cover_file([{**whole, "limit": 16}])
    assert not d2_io.runs_cover_file([p(1, 3), p(2, 3), p(3, 3, "interrupted")])
    assert not d2_io.runs_cover_file([p(1, 2), p(2, 3), p(3, 3)])                     # parts of two splits
    assert d2_io.runs_cover_file([p(1, 2), p(2, 3), p(2, 2)])
    assert not d2_io.runs_cover_file([{"status": "complete", "limit": None, "part": (1, 2)}])


def test_runs_cover_file_accepts_a_part_split_into_halves():
    """A waiting part K/N may be replaced by K/2N and (K+N)/2N."""
    from src import d2_io
    p = lambda k, n, s="part_complete": {"status": s, "limit": None, "part": (k, n)}  # noqa: E731
    assert d2_io.runs_cover_file([p(1, 4), p(2, 4), p(4, 4), p(3, 8), p(7, 8)])
    assert not d2_io.runs_cover_file([p(1, 4), p(2, 4), p(4, 4), p(3, 8)])
    assert not d2_io.runs_cover_file([p(1, 4), p(2, 4), p(4, 4), p(3, 8), p(7, 8, "interrupted")])
    assert not d2_io.runs_cover_file([p(1, 4), p(2, 4), p(4, 4), p(3, 8), p(5, 8)])  # 5/8 is half of 1/4, not of 3/4
    assert d2_io.runs_cover_file([p(1, 6), p(2, 6), p(3, 3), p(4, 6), p(5, 6)])
    assert d2_io.runs_cover_file([p(1, 3), p(2, 6), p(3, 6), p(5, 6), p(6, 6)])
    assert d2_io.runs_cover_file([p(2, 4), p(1, 4), p(4, 4), p(3, 4)])
    assert not d2_io.runs_cover_file([p(1, 997), p(2, 1000)])


def test_select_part_halves_are_the_part():
    """Part K/N is exactly the union of parts K/2N and (K+N)/2N, and the halves do not overlap."""
    msgs = [json.dumps([{"role": "user", "content": f"q{i}"}]) for i in range(600)]
    df = pd.DataFrame({"messages_json": msgs})
    for n in (2, 3, 4):
        for k in range(1, n + 1):
            whole = set(g.select_part(df, k, n).index)
            a, b = set(g.select_part(df, k, 2 * n).index), set(g.select_part(df, k + n, 2 * n).index)
            assert whole == a | b and not (a & b) and a and b


def test_interrupt_then_resume_completes_exactly_once(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", same_as={})
    out = tmp_path / "out"
    calls = {"n": 0}

    def stop_after_first_chunk():
        calls["n"] += 1
        return True

    first = cfg.run(req, out, shard_size=5, should_stop=stop_after_first_chunk)
    assert first["status"] == "interrupted"
    partial = g.read_outputs(out)
    assert 0 < len(partial) < 24
    second = cfg.run(req, out, shard_size=5)
    assert second["status"] == "complete" and second["requests_already_done"] == len(partial)
    final = g.read_outputs(out)
    assert len(final) == 24 and final["request_id"].is_unique and set(final["request_id"]) == set(
        pd.read_parquet(req)["request_id"])


_ROW_COLS = ["request_id", "config_id", "arm", "item_id", "prompt_sha256", "output_text", "n_prompt_tokens",
             "n_output_tokens", "finish_reason", "cached"]


def _rows(out):
    return g.read_outputs(out)[_ROW_COLS].sort_values("request_id").reset_index(drop=True)


def test_finished_outputs_are_written_inside_a_chunk(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")             # c2 repeats c0: cached rows in the same chunk
    ref = cfg.run(req, tmp_path / "ref", flush_seconds=None)
    inc = cfg.run(req, tmp_path / "inc", flush_seconds=0)
    assert len(ref["shards_written"]) == 1 and len(inc["shards_written"]) > 1
    assert inc["flush_seconds"] == 0 and inc["status"] == "complete"
    pd.testing.assert_frame_equal(_rows(tmp_path / "ref"), _rows(tmp_path / "inc"))
    assert g.read_outputs(tmp_path / "inc")["request_id"].is_unique
    assert inc["totals"] == ref["totals"]


def test_a_kill_inside_a_chunk_keeps_the_finished_outputs(cfg, tmp_path):
    class Killed(Exception):
        pass

    class DiesMidChunk(g.EchoBackend):
        def generate(self, conversations, params, on_result=None, on_tick=None):
            seen = {"n": 0}

            def counting(j, r):
                if seen["n"] == 7:
                    raise Killed("the job was killed")
                seen["n"] += 1
                on_result(j, r)
            return super().generate(conversations, params, on_result=counting, on_tick=on_tick)

    req = make_requests(tmp_path / "r.parquet", same_as={})
    cfg.run(req, tmp_path / "ref", flush_seconds=None)
    out = tmp_path / "out"
    with pytest.raises(Killed):
        cfg.run(req, out, backend_obj=DiesMidChunk(cfg.spec(), cfg.dec()), flush_seconds=0)
    kept = g.read_outputs(out)
    assert len(kept) == 7 and kept["request_id"].is_unique      # one chunk of 24, seven outputs finished
    second = cfg.run(req, out)
    assert second["requests_already_done"] == 7 and second["status"] == "complete"
    pd.testing.assert_frame_equal(_rows(tmp_path / "ref"), _rows(out))


def test_a_stop_signal_ends_the_chunk_after_writing_its_finished_outputs(cfg, tmp_path):
    """At a time limit USR1 comes 300 s ahead: the run must end well before the chunk does, so the job script can
    requeue the job itself (exit 75), and a resumed run must give the same rows as an uninterrupted one."""
    req = make_requests(tmp_path / "r.parquet", same_as={})
    cfg.run(req, tmp_path / "ref", flush_seconds=None)
    out = tmp_path / "out"
    state = {"ticks": 0}

    def stop_after_five_outputs():          # checked at each tick; one output finishes per tick in the echo backend
        state["ticks"] += 1
        return state["ticks"] > 5

    log = cfg.run(req, out, flush_seconds=10 ** 6, should_stop=stop_after_five_outputs)
    assert log["status"] == "interrupted"
    kept = g.read_outputs(out)
    assert 0 < len(kept) < 24 and kept["request_id"].is_unique   # one chunk of 24, ended inside
    second = cfg.run(req, out)
    assert second["status"] == "complete" and second["requests_already_done"] == len(kept)
    pd.testing.assert_frame_equal(_rows(tmp_path / "ref"), _rows(out))


def test_a_stop_signal_at_the_first_step_keeps_the_one_finished_output_and_resumes(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", same_as={})
    log = cfg.run(req, tmp_path / "out", should_stop=lambda: True)
    assert log["status"] == "interrupted" and len(g.read_outputs(tmp_path / "out")) == 1
    assert log["totals"]["unique_generations"] == 1
    again = cfg.run(req, tmp_path / "out")
    assert again["status"] == "complete" and len(g.read_outputs(tmp_path / "out")) == 24


def test_a_stop_signal_without_early_saving_still_waits_for_the_chunk(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", same_as={})
    log = cfg.run(req, tmp_path / "out", flush_seconds=None, should_stop=lambda: True)
    assert log["status"] == "complete" and len(g.read_outputs(tmp_path / "out")) == 24   # one chunk; stops between


def test_flush_seconds_is_checked(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")
    with pytest.raises(g.RunRefused, match="flush_seconds"):
        cfg.run(req, tmp_path / "out", flush_seconds=-1)


def test_vllm_backend_reports_outputs_as_they_finish(cfg, monkeypatch):
    """The step watch maps vLLM's request ids back to input positions, in both calls (plain and prefilled), and puts
    the engine's own step back afterwards."""
    import types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class Counter:
        def __init__(self):
            self.counter = 0

        def __next__(self):
            self.counter += 1
            return self.counter - 1

    class Engine:
        def __init__(self):
            self.queue = []

        def add_request(self, rid, conv, sp):    # vLLM 0.30: outputs carry rid; the return value is an internal id
            text = "out:" + conv[-1]["content"]
            self.queue.append(types.SimpleNamespace(
                request_id=rid, finished=True, prompt=None, prompt_token_ids=[1, 2],
                outputs=[types.SimpleNamespace(text=text, token_ids=[0] * len(text), finish_reason="stop")]))
            return f"{rid}-1a2b3c4d"

        def step(self):        # finish the newest request first, as continuous batching may
            return [self.queue.pop()] if self.queue else []

    class FakeLLM:      # vLLM 0.30's LLM.chat: _add_request per conversation in order, then step() until done
        def __init__(self):
            self.request_counter, self.llm_engine = Counter(), Engine()
            self.request_counter.counter = 40           # earlier calls used ids 0-39

        def _add_request(self, conv, sp):
            return self.llm_engine.add_request(str(next(self.request_counter)), conv, sp)

        def chat(self, convs, sps, **kw):
            for c, sp in zip(convs, sps):
                self._add_request(c, sp)
            got = []
            while self.llm_engine.queue:
                got += [o for o in self.llm_engine.step() if o.finished]
            return sorted(got, key=lambda o: int(o.request_id))

    be = g.make_backend("vllm", cfg.spec(), cfg.dec())
    be.llm = FakeLLM()
    convs = [[{"role": "user", "content": f"q{i}"}] for i in range(5)]
    convs[1] = convs[1] + [{"role": "assistant", "content": "p1"}]       # prefilled: the second call
    convs[3] = convs[3] + [{"role": "assistant", "content": "p3"}]
    params = [cfg.dec().params_for_arm(ARMS[0])] * 5
    seen, ticks = {}, []
    res = be.generate(convs, params, on_result=lambda i, r: seen.__setitem__(i, r), on_tick=lambda: ticks.append(1))
    assert [r.text for r in res] == ["out:q0", "out:p1", "out:q2", "out:p3", "out:q4"]
    assert seen == dict(enumerate(res)) and ticks
    assert "step" not in vars(be.llm.llm_engine) and "add_request" not in vars(be.llm.llm_engine)
    assert [r.text for r in be.generate(convs, params)] == [r.text for r in res]   # no watch without callbacks

    class Extra(FakeLLM):     # an output whose id this call did not add: early saving stops, nothing is guessed
        def chat(self, convs, sps, **kw):
            for c, sp in zip(convs, sps):
                self._add_request(c, sp)
            self.llm_engine.queue.append(types.SimpleNamespace(request_id="999", finished=True))   # finishes first
            got = []
            while self.llm_engine.queue:
                got += [o for o in self.llm_engine.step() if o.finished and o.request_id != "999"]
            return sorted(got, key=lambda o: int(o.request_id))
    be.llm, be.watch_notes, seen = Extra(), [], {}
    res2 = be.generate(convs, params, on_result=lambda i, r: seen.__setitem__(i, r))
    assert [r.text for r in res2] == [r.text for r in res] and seen == {}
    assert len(be.watch_notes) == 2 and all("was not added by this call" in m for m in be.watch_notes)
    assert "step" not in vars(be.llm.llm_engine) and "add_request" not in vars(be.llm.llm_engine)


def test_resume_refuses_a_changed_configuration(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")
    out = tmp_path / "out"
    cfg.run(req, out, limit=5)
    cfg.write(model_entry(), decoding_dict(max_new_tokens={"direct_answer": 399, "zero_shot_cot": 400}))
    with pytest.raises(g.RunRefused, match="different model/decoding"):
        cfg.run(req, out)
    cfg.write(model_entry(revision="c" * 40), decoding_dict())
    with pytest.raises(g.RunRefused, match="different model/decoding"):
        cfg.run(req, out)


def test_resume_refuses_another_gpu_model(cfg, tmp_path, monkeypatch):
    req = make_requests(tmp_path / "r.parquet")
    out = tmp_path / "out"
    cfg.run(req, out, limit=5)                                               # on "Fake A100"
    assert set(g.prior_gpu_types(out).values()) == {("Fake A100", "Fake A100")}
    other = {"count": 1, "devices": [{"name": "NVIDIA GeForce RTX 3090", "memory_mib": 24576}],
             "driver_version": "580.1", "cuda_version_driver": "13.0", "nvidia_smi": True}
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: other)
    n_before = len(g.read_outputs(out))
    with pytest.raises(g.RunRefused, match="one GPU model"):
        cfg.run(req, out)
    assert len(g.read_outputs(out)) == n_before                              # nothing generated
    # a node without visible GPUs (echo dry runs) does not trip the check
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0, "devices": [], "driver_version": None,
                                                              "cuda_version_driver": None, "nvidia_smi": False})
    assert cfg.run(req, out)["status"] == "complete"
    assert g.gpu_type(other) == ("NVIDIA GeForce RTX 3090",) and g.gpu_type(None) == ()


def test_family_label_does_not_break_resume(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet")
    cfg.run(req, tmp_path / "out", limit=5)
    cfg.write(model_entry(family="renamed"), decoding_dict())
    assert cfg.run(req, tmp_path / "out")["status"] == "complete"


# ---------------------------------------------------------------- run log

def test_run_log_fields(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_TOKEN_VALUE")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    req = make_requests(tmp_path / "r.parquet")
    cfg.run(req, tmp_path / "out")
    (log,) = read_log(tmp_path / "out")
    for k in ("started_utc", "ended_utc", "runtime_s", "hostname", "gpu", "versions", "model", "decoding",
              "counts", "totals", "run_fingerprint", "inputs", "scheduler", "chat_template", "status",
              "gpu_hours_billed_estimate", "generation_s", "output_tokens_per_s"):
        assert k in log, k
    assert log["gpu"]["count"] == 2 and log["gpu"]["devices"][0]["name"] == "Fake A100"
    assert log["gpu"]["driver_version"] == "580.1" and log["gpu"]["cuda_version_driver"] == "13.0"
    for pkg in ("torch", "vllm", "transformers", "python"):
        assert pkg in log["versions"]
    assert log["model"]["revision"] == SHA and log["model"]["hf_id"] == "org/model"
    assert log["model"]["thinking_off"]["method"] == "chat_template_kwargs"
    assert log["decoding"]["temperature"] == 0.0 and log["decoding"]["max_new_tokens"]["zero_shot_cot"] == 400
    t = log["totals"]
    for k in ("unique_generations", "cached_calls", "prompt_tokens_generated", "output_tokens_generated",
              "truncated_generated", "outputs_with_think_tag_generated"):
        assert k in t
    out = g.read_outputs(tmp_path / "out")
    gen_rows = out[~out["cached"]]
    assert t["unique_generations"] == len(gen_rows)
    assert t["output_tokens_generated"] == int(gen_rows["n_output_tokens"].sum())
    assert t["prompt_tokens_generated"] == int(gen_rows["n_prompt_tokens"].sum())
    assert t["output_tokens_all_requests"] == int(out["n_output_tokens"].sum())
    assert log["gpu_hours_billed_estimate"] == 0.0            # echo backend uses no GPU
    assert log["inputs"]["requests_sha256"] == g.sha256_file(req)
    assert log["scheduler"]["SLURM_JOB_ID"] == "12345" and "HF_TOKEN" not in log["scheduler"]
    text = (next((tmp_path / "out").glob("run_log-*.json"))).read_text(encoding="utf-8")
    assert "hf_SECRET_TOKEN_VALUE" not in text and "HF_TOKEN" not in text


def test_think_tags_are_counted(cfg, tmp_path):
    class Thinker(g.EchoBackend):
        def generate(self, conversations, params):
            res = super().generate(conversations, params)
            for r in res:
                r.text = "<think>x</think> " + r.text
            return res

    req = make_requests(tmp_path / "r.parquet", same_as={})
    log = cfg.run(req, tmp_path / "out", backend_obj=Thinker(cfg.spec(), cfg.dec()))
    assert log["totals"]["outputs_with_think_tag_generated"] == log["totals"]["unique_generations"]


def test_render_mismatch_is_reported(cfg, tmp_path):
    class Drifting(g.EchoBackend):
        def generate(self, conversations, params):
            res = super().generate(conversations, params)
            for r in res:
                r.prompt_echo = r.prompt_echo + " "
            return res

    req = make_requests(tmp_path / "r.parquet", same_as={}, n_items=1)
    log = cfg.run(req, tmp_path / "out", backend_obj=Drifting(cfg.spec(), cfg.dec()))
    assert log["prompt_render_mismatches"] == log["totals"]["unique_generations"] > 0
    assert log["warnings"]


def test_failed_run_still_writes_a_log(cfg, tmp_path):
    class Broken(g.EchoBackend):
        def generate(self, conversations, params):
            res = super().generate(conversations, params)
            res[0].finish_reason = "abort"
            return res

    req = make_requests(tmp_path / "r.parquet")
    with pytest.raises(g.GenerationError):
        cfg.run(req, tmp_path / "out", backend_obj=Broken(cfg.spec(), cfg.dec()))
    (log,) = read_log(tmp_path / "out")
    assert log["status"] == "failed" and "finish_reason" in log["error"]


def test_nvidia_smi_parser():
    q = "NVIDIA H100 80GB HBM3, 81559, 580.65.06\nNVIDIA H100 80GB HBM3, 81559, 580.65.06\n"
    h = "| NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 13.0 |"
    info = g.parse_nvidia_smi(q, h)
    assert info["count"] == 2 and info["devices"][0]["name"] == "NVIDIA H100 80GB HBM3"
    assert info["devices"][0]["memory_mib"] == 81559
    assert info["driver_version"] == "580.65.06" and info["cuda_version_driver"] == "13.0"
    newer = "| NVIDIA-SMI 610.88    KMD Version: 610.88    CUDA UMD Version: 13.3 |"
    assert g.parse_nvidia_smi(q, newer)["cuda_version_driver"] == "13.3"
    assert g.parse_nvidia_smi("", "no header")["count"] == 0


# ---------------------------------------------------------------- throughput mode

def test_throughput_subset_is_deterministic_and_stratified(tmp_path):
    p = make_requests(tmp_path / "r.parquet", n_items=10)
    df = g.order_requests(pd.read_parquet(p))
    a = g.select_throughput_subset(df, 9, seed=0)
    b = g.select_throughput_subset(df, 9, seed=0)
    assert a["request_id"].tolist() == b["request_id"].tolist()
    counts = a["arm"].value_counts()
    assert sorted(counts.tolist()) == [4, 5] and len(a) == 9
    assert a["request_id"].is_unique
    assert a["request_id"].tolist() != g.select_throughput_subset(df, 9, seed=1)["request_id"].tolist()


def test_throughput_mode_reports_tokens_per_second_and_lengths_per_arm(cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", n_items=10, same_as={})
    out = tmp_path / "out"
    log = cfg.run(req, out, throughput=True, limit=20, shard_size=7,
                  backend_obj=cfg.echo(delay_s=0.03))
    assert log["mode"] == "throughput" and log["excluded_from_main"]
    assert not list(out.glob("shard-*.parquet"))                       # main directory untouched
    tdir = out / "throughput"
    assert len(g.read_outputs(tdir)) == 20
    rep = json.loads(next(tdir.glob("throughput_report-*.json")).read_text(encoding="utf-8"))
    assert rep["output_tokens_per_s"] > 0 and rep["generation_s"] > 0
    assert set(rep["per_arm"]) == set(ARMS)
    for arm in ARMS:
        d = rep["per_arm"][arm]
        assert d["unique_generations"] == 10
        o = d["output_tokens"]
        assert o["n"] == 10 and o["min"] <= o["p50"] <= o["p95"] <= o["max"] and o["mean"] > 0
        assert d["share_truncated_at_cap"] == 0.0
    proj = rep["projection_full_request_file"]
    assert proj["requests_by_arm"] == {a: 30 for a in ARMS}
    assert proj["estimated_gpu_hours"] == pytest.approx(proj["estimated_generation_hours"] * 2)   # tp = 2
    assert "extrapolation" in proj["note"]
    assert len(list(tdir.glob("shard-*.parquet"))) == 3                  # 20 generations in chunks of 7
    # A later full run does not see or resume from the throughput outputs.
    full = cfg.run(req, out)
    assert full["requests_already_done"] == 0 and full["mode"] == "full"


def test_throughput_mode_needs_a_subset_size(cfg, tmp_path):
    with pytest.raises(g.RunRefused, match="--limit"):
        cfg.run(make_requests(tmp_path / "r.parquet"), tmp_path / "o", throughput=True)


# ---------------------------------------------------------------- FlashInfer sampler preflight

def _fi_preflight(cfg_obj, *, throughput=True, limit=100, backend="vllm"):
    return g.preflight(cfg_obj.spec(), cfg_obj.dec(), backend=backend, throughput=throughput, limit=limit,
                       budget_path=cfg_obj.budget)


def test_flashinfer_refused_when_sampler_used_and_no_cuda_toolkit(cfg, fi):
    fi["importable"] = True                      # env unset: the sampler is on by default
    with pytest.raises(g.RunRefused) as ei:
        _fi_preflight(cfg)
    msg = str(ei.value)
    for needle in ("FlashInfer sampler will be used", "container", "CUDA_HOME", "VLLM_USE_FLASHINFER_SAMPLER=0",
                   "greedy decoding", "kernels and speed differ", "recorded in the run log"):
        assert needle in msg, needle


@pytest.mark.parametrize("value", ["1", "2", " 1 "])
def test_flashinfer_refused_for_any_truthy_value(cfg, fi, monkeypatch, value):
    fi["importable"] = True
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", value)
    with pytest.raises(g.RunRefused, match="FlashInfer sampler"):
        _fi_preflight(cfg)


@pytest.mark.parametrize("throughput, limit", [(True, 512), (False, 100), (False, None)])
def test_flashinfer_check_applies_to_throughput_pilots_and_bulk_runs(cfg, fi, throughput, limit):
    fi["importable"] = True
    with pytest.raises(g.RunRefused, match="FlashInfer sampler"):
        _fi_preflight(cfg, throughput=throughput, limit=limit)


def test_flashinfer_message_joins_the_other_gate_problems(tmp_path, fi):
    c = Cfg(tmp_path, decoding=decoding_dict(status="PENDING_REVIEW"))
    fi["importable"] = True
    with pytest.raises(g.RunRefused) as ei:
        _fi_preflight(c, throughput=False, limit=None)
    assert "FlashInfer sampler" in str(ei.value) and "PENDING_REVIEW" in str(ei.value)


@pytest.mark.parametrize("value", ["0", " 0 ", "00"])
def test_flashinfer_passes_when_sampler_is_switched_off(cfg, fi, monkeypatch, value):
    fi["importable"] = True
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", value)
    rec = _fi_preflight(cfg)["flashinfer_sampler"]
    assert rec["checked"] and rec["env_enabled"] is False and rec["used"] is False
    assert rec["cuda_home_found"] is False and rec["env_value"] == value


def test_flashinfer_passes_when_flashinfer_is_not_installed(cfg, fi):
    fi["importable"] = False
    rec = _fi_preflight(cfg)["flashinfer_sampler"]
    assert rec["used"] is False and rec["flashinfer_importable"] is False and rec["env_enabled"] is True


@pytest.mark.parametrize("var", ["CUDA_HOME", "CUDA_PATH"])
def test_flashinfer_passes_with_cuda_home_or_path(cfg, fi, monkeypatch, var):
    fi["importable"] = True
    monkeypatch.setenv(var, "/opt/cuda-13.0")
    rec = _fi_preflight(cfg)["flashinfer_sampler"]
    assert rec["used"] is True and rec["cuda_home_found"] is True
    assert rec["cuda_home"] == "/opt/cuda-13.0" and rec["cuda_home_source"] == var


def test_cuda_lookup_order_follows_flashinfer(fi, monkeypatch):
    fi.update(importable=True, nvcc="/usr/bin/nvcc", usr_local_cuda=True)
    monkeypatch.setenv("CUDA_HOME", "")
    monkeypatch.setenv("CUDA_PATH", "/opt/from-path")            # empty CUDA_HOME counts as unset
    assert g.discover_cuda_home() == ("/opt/from-path", "CUDA_PATH")
    monkeypatch.setenv("CUDA_HOME", "/opt/from-home")
    assert g.discover_cuda_home() == ("/opt/from-home", "CUDA_HOME")
    monkeypatch.delenv("CUDA_HOME")
    monkeypatch.delenv("CUDA_PATH")
    assert g.discover_cuda_home() == ("/usr", "nvcc_on_path")     # the directory above bin/
    fi["nvcc"] = None
    assert g.discover_cuda_home() == ("/usr/local/cuda", "default_usr_local_cuda")
    fi["usr_local_cuda"] = False
    assert g.discover_cuda_home() == (None, None)
    monkeypatch.setenv("CUDA_HOME", "")
    assert g.discover_cuda_home() == (None, None)
    assert g.discover_cuda_home({"CUDA_PATH": "/x"}) == ("/x", "CUDA_PATH")   # an explicit environment mapping


def test_flashinfer_passes_with_nvcc_on_path(cfg, fi):
    fi.update(importable=True, nvcc="/opt/cuda-12.9/bin/nvcc")
    rec = _fi_preflight(cfg)["flashinfer_sampler"]
    assert rec["cuda_home"] == "/opt/cuda-12.9" and rec["cuda_home_source"] == "nvcc_on_path"


def test_flashinfer_passes_with_usr_local_cuda(cfg, fi):
    fi.update(importable=True, usr_local_cuda=True)
    rec = _fi_preflight(cfg)["flashinfer_sampler"]
    assert rec["cuda_home"] == "/usr/local/cuda" and rec["cuda_home_source"] == "default_usr_local_cuda"


def test_flashinfer_env_that_vllm_cannot_parse_is_refused(cfg, fi, monkeypatch):
    for bad in ("false", "True", "", "yes"):               # vLLM 0.30.0 reads bool(int(value)) and would raise
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", bad)
        with pytest.raises(g.RunRefused, match="not an integer"):
            _fi_preflight(cfg)
    assert g.flashinfer_sampler_enabled({}) is True          # unset: on
    assert g.flashinfer_sampler_enabled({"VLLM_USE_FLASHINFER_SAMPLER": "1"}) is True
    assert g.flashinfer_sampler_enabled({"VLLM_USE_FLASHINFER_SAMPLER": "0"}) is False


def test_flashinfer_check_does_not_run_for_the_echo_backend(cfg, fi, monkeypatch, tmp_path):
    fi["importable"] = True                                # would be refused on the vllm backend

    def boom(*a, **k):
        raise AssertionError("the FlashInfer check must not run for the echo backend")

    monkeypatch.setattr(g, "check_flashinfer_sampler", boom)
    assert _fi_preflight(cfg, backend="echo") == {"flashinfer_sampler": {"checked": False,
                                                                         "reason": "backend is not vllm"}}
    log = cfg.run(make_requests(tmp_path / "r.parquet"), tmp_path / "out")
    assert log["status"] == "complete" and log["preflight"]["flashinfer_sampler"]["checked"] is False


def test_flashinfer_lookup_does_not_import_flashinfer(monkeypatch):
    seen = []

    def fake_find_spec(name):
        seen.append(name)
        return object() if name == "flashinfer" else None

    monkeypatch.setattr(g.importlib.util, "find_spec", fake_find_spec)
    assert ORIGINAL_FLASHINFER_IMPORTABLE() is True and seen == ["flashinfer"]
    monkeypatch.setattr(g.importlib.util, "find_spec", lambda name: None)
    assert ORIGINAL_FLASHINFER_IMPORTABLE() is False

    def bad_spec(name):
        raise ValueError("flashinfer.__spec__ is None")

    monkeypatch.setattr(g.importlib.util, "find_spec", bad_spec)
    assert ORIGINAL_FLASHINFER_IMPORTABLE() is False
    assert "flashinfer" not in sys.modules


def test_vllm_run_refused_for_missing_cuda_before_anything_is_written(cfg, fi, tmp_path):
    fi["importable"] = True
    out = tmp_path / "out"
    with pytest.raises(g.RunRefused, match="FlashInfer sampler"):
        g.run_generation(make_requests(tmp_path / "r.parquet"), KEY, out, backend="vllm", throughput=True, limit=8,
                         models_path=cfg.models, decoding_path=cfg.decoding, budget_path=cfg.budget, verbose=False)
    assert not out.exists() and "vllm" not in sys.modules


def test_preflight_result_is_recorded_in_the_run_log(cfg, fi, monkeypatch, tmp_path):
    fi["importable"] = True
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-13.0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    req = make_requests(tmp_path / "r.parquet", n_items=4, same_as={})
    # backend="vllm" makes preflight run its vLLM checks; the engine itself is the echo stand-in
    log = g.run_generation(req, KEY, tmp_path / "out", backend="vllm", backend_obj=cfg.echo(), throughput=True,
                           limit=8, models_path=cfg.models, decoding_path=cfg.decoding, budget_path=cfg.budget,
                           verbose=False)
    rec = log["preflight"]["flashinfer_sampler"]
    assert rec == {"checked": True, "env_value": "1", "env_enabled": True, "flashinfer_importable": True,
                   "used": True, "cuda_home_found": True, "cuda_home": "/opt/cuda-13.0",
                   "cuda_home_source": "CUDA_HOME"}
    assert log["scheduler"]["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    assert log["scheduler"]["VLLM_BATCH_INVARIANT"] == "1"
    (on_disk,) = read_log(tmp_path / "out" / "throughput")
    assert on_disk["preflight"] == log["preflight"]


# ---------------------------------------------------------------- output-cap measurement (--measure-cap)

LEN_RE = re.compile(r"LEN=(\d+)")
DIRECT_LENS = [10, 50, 64, 65, 100, 128, 129, 200, 300, 500]
COT_LENS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


class LengthEcho(g.EchoBackend):
    """Echo backend whose output length is the number after LEN= in the prompt, cut at the arm's cap
    and marked finish_reason length like the real engine."""

    def generate(self, conversations, params):
        self.generate_calls.append(len(conversations))
        out = []
        for conv, p in zip(conversations, params):
            prompt = self.render(conv)
            n, finish = int(LEN_RE.search(prompt).group(1)), "stop"
            if n > p["max_new_tokens"]:
                n, finish = p["max_new_tokens"], "length"
            out.append(g.GenResult(" ".join(["w"] * n), self.count_tokens(prompt), n, finish, prompt))
        return out


def known_length_requests(path):
    lens = {"direct_answer": DIRECT_LENS, "zero_shot_cot": COT_LENS}
    return make_requests(path, n_items=10, configs=("c0",), same_as={},
                         content=lambda skel, arm, it: f"LEN={lens[arm][it]} item {it}")


@pytest.fixture
def cap_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 1, "devices": [{"name": "Fake", "memory_mib": 1}], "driver_version": "1",
        "cuda_version_driver": "13.0"})
    return Cfg(tmp_path, entry=model_entry(max_model_len=8192),
               decoding=decoding_dict(max_new_tokens={"direct_answer": 200, "zero_shot_cot": 1024}))


def run_known(cap_cfg, tmp_path, name="o", **kw):
    req = known_length_requests(tmp_path / "known.parquet")
    return cap_cfg.run(req, tmp_path / name, backend_obj=LengthEcho(cap_cfg.spec(), cap_cfg.dec()),
                       throughput=True, limit=20, **kw)


def test_parse_cap_overrides():
    assert g.parse_cap_overrides(None) == {} and g.parse_cap_overrides([]) == {}
    assert g.parse_cap_overrides(["direct_answer=512", " zero_shot_cot = 4096 "]) == \
        {"direct_answer": 512, "zero_shot_cot": 4096}
    for bad in ("direct_answer", "direct_answer=", "=5", "direct_answer=0", "direct_answer=-3",
                "direct_answer=1.5", "direct_answer=abc", "direct_answer=1e3"):
        with pytest.raises(g.RunRefused, match="ARM=N"):
            g.parse_cap_overrides([bad])
    with pytest.raises(g.RunRefused, match="more than once"):
        g.parse_cap_overrides(["direct_answer=5", "direct_answer=6"])


def test_cap_exceedance_is_strict_and_limited_to_candidates_below_the_cap():
    assert g.CAP_CANDIDATES == (64, 128, 256, 512, 1024, 2048, 4096)
    assert g.cap_exceedance([64, 65], 1024) == {"64": 0.5, "128": 0.0, "256": 0.0, "512": 0.0}
    assert g.cap_exceedance([10, 200], 65) == {"64": 0.5}
    assert g.cap_exceedance([10, 200], 64) == {} and g.cap_exceedance([10, 200], 1) == {}
    assert g.cap_exceedance([], 4096) == {} and g.cap_exceedance([5], None) == {}
    assert list(g.cap_exceedance([5], 5000)) == ["64", "128", "256", "512", "1024", "2048", "4096"]


def test_exceedance_shares_without_override_stop_at_the_configured_cap(cap_cfg, tmp_path):
    log = run_known(cap_cfg, tmp_path)
    assert log["cap_override"] is None and log["decoding"]["max_new_tokens"]["direct_answer"] == 200
    rep = json.loads(next((tmp_path / "o" / "throughput").glob("throughput_report-*.json")).read_text(encoding="utf-8"))
    assert rep["cap_override"] is None
    d, c = rep["per_arm"]["direct_answer"], rep["per_arm"]["zero_shot_cot"]
    # direct lengths 10 50 64 65 100 128 129 200 300 500 under cap 200: 300 and 500 are cut at 200
    assert d["max_new_tokens"] == 200 and d["share_truncated_at_cap"] == pytest.approx(0.2)
    assert d["share_exceeding_candidate_cap"] == {"64": pytest.approx(0.7), "128": pytest.approx(0.4)}
    # cot lengths 100..1000 under cap 1024: nothing is cut; candidates 64 128 256 512
    assert c["max_new_tokens"] == 1024 and c["share_truncated_at_cap"] == 0.0
    assert c["share_exceeding_candidate_cap"] == {"64": pytest.approx(1.0), "128": pytest.approx(0.9),
                                                  "256": pytest.approx(0.8), "512": pytest.approx(0.5)}
    assert log["throughput"]["per_arm"] == rep["per_arm"]       # the run log carries the same shares


def test_measure_cap_exposes_the_lengths_a_cap_hides(cap_cfg, tmp_path):
    log = run_known(cap_cfg, tmp_path, cap_override={"direct_answer": 1024})
    assert log["status"] == "complete"
    d, c = log["throughput"]["per_arm"]["direct_answer"], log["throughput"]["per_arm"]["zero_shot_cot"]
    assert d["max_new_tokens"] == 1024 and d["share_truncated_at_cap"] == 0.0
    assert d["output_tokens"]["max"] == 500                     # not cut at the configured 200
    assert d["share_exceeding_candidate_cap"] == {"64": pytest.approx(0.7), "128": pytest.approx(0.4),
                                                  "256": pytest.approx(0.2), "512": pytest.approx(0.0)}
    assert c["max_new_tokens"] == 1024                          # the other arm keeps its configured cap
    out = g.read_outputs(tmp_path / "o" / "throughput")
    direct = out[out["arm"] == "direct_answer"]
    assert sorted(direct["n_output_tokens"]) == sorted(DIRECT_LENS) and (direct["finish_reason"] == "stop").all()


def test_measure_cap_exceedance_matches_the_echo_outputs_in_the_shards(cap_cfg, tmp_path):
    req = make_requests(tmp_path / "r.parquet", n_items=15, configs=("c0", "c1"), same_as={})
    log = cap_cfg.run(req, tmp_path / "o", throughput=True, limit=40, cap_override={"direct_answer": 2048})
    out = g.read_outputs(tmp_path / "o" / "throughput")
    assert out["n_output_tokens"].max() <= 300 and (out["finish_reason"] == "stop").all()   # echo lengths are 1..300
    for arm, cap in (("direct_answer", 2048), ("zero_shot_cot", 1024)):
        lens = out.loc[(out["arm"] == arm) & ~out["cached"], "n_output_tokens"]
        shares = log["throughput"]["per_arm"][arm]["share_exceeding_candidate_cap"]
        assert list(shares) == [str(k) for k in g.CAP_CANDIDATES if k < cap]
        for k, v in shares.items():
            assert v == pytest.approx(float((lens > int(k)).mean()))
        assert shares["64"] > 0.5 and shares["256"] < 0.5       # natural echo lengths are spread over 1..300


def test_measure_cap_is_recorded_in_the_log_and_the_fingerprint(cap_cfg, tmp_path):
    plain = run_known(cap_cfg, tmp_path, "plain")
    over = run_known(cap_cfg, tmp_path, "over", cap_override={"direct_answer": 1024})
    same = run_known(cap_cfg, tmp_path, "same", cap_override={"direct_answer": 200})   # equals the configured cap
    assert plain["cap_override"] is None
    co = over["cap_override"]
    assert co["arms"] == {"direct_answer": 1024}
    assert co["configured_max_new_tokens"] == {"direct_answer": 200, "zero_shot_cot": 1024}
    assert co["effective_max_new_tokens"] == {"direct_answer": 1024, "zero_shot_cot": 1024}
    assert over["decoding"]["max_new_tokens"] == co["effective_max_new_tokens"]   # the decoding block is what ran
    assert over["mode"] == "throughput" and over["excluded_from_main"] is True
    assert len({plain["run_fingerprint"], over["run_fingerprint"], same["run_fingerprint"]}) == 3
    for name, log in (("plain", plain), ("over", over), ("same", same)):
        shards = list((tmp_path / name / "throughput").glob("shard-*.parquet"))
        assert shards
        for shard in shards:
            meta = pq.read_schema(shard).metadata
            assert meta[b"run_fingerprint"].decode() == log["run_fingerprint"] and meta[b"throughput"] == b"true"
            assert (b"cap_override" in meta) == (name != "plain")
    meta = pq.read_schema(next((tmp_path / "over" / "throughput").glob("shard-*.parquet"))).metadata
    assert json.loads(meta[b"cap_override"]) == {"direct_answer": 1024}
    spec, dec = cap_cfg.spec(), cap_cfg.dec()
    assert g.run_fingerprint(spec, dec, "echo") == g.run_fingerprint(spec, dec, "echo", None) \
        == g.run_fingerprint(spec, dec, "echo", {})             # no override: payload unchanged
    assert g.run_fingerprint(spec, dec, "echo") != g.run_fingerprint(spec, dec, "echo", {"direct_answer": 200})


def test_batch_invariant_kernels_enter_the_fingerprint_so_a_folder_never_mixes_kernel_modes(cap_cfg, monkeypatch):
    spec, dec = cap_cfg.spec(), cap_cfg.dec()
    plain = g.run_fingerprint(spec, dec, "vllm")
    assert g.run_fingerprint(spec, dec, "vllm", None, False) == plain          # default kernels: payload unchanged
    invariant = g.run_fingerprint(spec, dec, "vllm", None, True)
    assert invariant != plain
    assert invariant != g.run_fingerprint(spec, dec, "vllm", {"direct_answer": 200}, True)
    for raw, on in (("1", True), ("0", False), ("2", True), ("00", False)):    # vLLM's reading: int(...) != 0
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", raw)
        assert g.batch_invariant_env() is on
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    assert g.batch_invariant_env() is False
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "yes")
    with pytest.raises(g.RunRefused, match="not an integer"):
        g.batch_invariant_env()


def test_inductor_kernel_switches_enter_the_fingerprint_and_the_run_log(cap_cfg, monkeypatch):
    spec, dec = cap_cfg.spec(), cap_cfg.dec()
    invariant = g.run_fingerprint(spec, dec, "vllm", None, True)
    assert g.run_fingerprint(spec, dec, "vllm", None, True, []) == invariant      # no switch: payload unchanged
    both = g.run_fingerprint(spec, dec, "vllm", None, True, ["TORCHINDUCTOR_BATCH_INVARIANT"])
    assert both != invariant
    assert g.run_fingerprint(spec, dec, "vllm", None, False, ["TORCHINDUCTOR_BATCH_INVARIANT"]) not in (both, invariant)
    for name in g.INDUCTOR_KERNEL_ENV:
        monkeypatch.delenv(name, raising=False)
    assert g.inductor_switches_env() == []
    monkeypatch.setenv("TORCHINDUCTOR_BATCH_INVARIANT", "1")
    monkeypatch.setenv("TORCHINDUCTOR_DETERMINISTIC", "true")                       # torch reads == "1" only
    assert g.inductor_switches_env() == ["TORCHINDUCTOR_BATCH_INVARIANT"]
    assert {"TORCHINDUCTOR_BATCH_INVARIANT", "TORCHINDUCTOR_DETERMINISTIC", "TORCHINDUCTOR_CACHE_DIR"} <= set(g._LOGGED_ENV)


def _vllm_run(cfg, tmp_path, name="out"):
    """backend="vllm" runs the vLLM gates and the kernel switches; the engine is the echo stand-in."""
    req = make_requests(tmp_path / "r.parquet", n_items=4, same_as={})
    return g.run_generation(req, KEY, tmp_path / name, backend="vllm", backend_obj=cfg.echo(), throughput=True,
                            limit=8, models_path=cfg.models, decoding_path=cfg.decoding, budget_path=cfg.budget,
                            verbose=False)


def test_every_registry_entry_names_its_kernel_mode(cfg):
    assert cfg.spec().kernels == "batch_invariant"
    for bad in (None, "PLACEHOLDER", "batch-invariant", "vllm_default"):
        entry = model_entry(kernels=bad)
        if bad is None:
            entry.pop("kernels")
        cfg.write(entry, decoding_dict())
        with pytest.raises(g.RunRefused, match="kernels must be one of"):
            cfg.spec()
    cfg.write(model_entry(kernels="default"), decoding_dict())
    assert cfg.spec().kernels == "default"
    template = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    assert template["dataset2"]["models"]["_template"]["kernels"] in g.KERNEL_MODES


def test_the_runner_sets_the_registered_kernel_switches_before_vllm_loads(cfg, tmp_path):
    log = _vllm_run(cfg, tmp_path)
    assert log["kernels"] == {"mode": "batch_invariant",
                              "switches": {"VLLM_BATCH_INVARIANT": "1", "TORCHINDUCTOR_BATCH_INVARIANT": "1"},
                              "set_by_runner": ["VLLM_BATCH_INVARIANT", "TORCHINDUCTOR_BATCH_INVARIANT"]}
    assert log["batch_invariant"] is True and log["inductor_switches"] == ["TORCHINDUCTOR_BATCH_INVARIANT"]
    assert log["scheduler"]["VLLM_BATCH_INVARIANT"] == "1" == log["scheduler"]["TORCHINDUCTOR_BATCH_INVARIANT"]
    assert log["model"]["kernels"] == "batch_invariant"
    again = _vllm_run(cfg, tmp_path, "again")              # switches already set as the mode wants: nothing to set
    assert again["kernels"]["set_by_runner"] == [] and again["run_fingerprint"] == log["run_fingerprint"]


def test_default_kernels_set_no_switch_and_differ_in_the_fingerprint(cfg, tmp_path, monkeypatch):
    invariant = _vllm_run(cfg, tmp_path, "bi")
    for name in g.KERNEL_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    cfg.write(model_entry(kernels="default"), decoding_dict())
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")         # off, as vLLM reads it: no conflict
    log = _vllm_run(cfg, tmp_path, "default")
    assert log["kernels"] == {"mode": "default", "switches": {}, "set_by_runner": []}
    assert log["batch_invariant"] is False and log["inductor_switches"] == []
    assert "TORCHINDUCTOR_BATCH_INVARIANT" not in g.os.environ
    assert log["run_fingerprint"] != invariant["run_fingerprint"]


@pytest.mark.parametrize("kernels, env, match", [
    ("default", {"VLLM_BATCH_INVARIANT": "1"}, "VLLM_BATCH_INVARIANT='1' \\(the mode leaves it off\\)"),
    ("default", {"TORCHINDUCTOR_BATCH_INVARIANT": "1"}, "the mode leaves it off"),
    ("batch_invariant", {"TORCHINDUCTOR_BATCH_INVARIANT": "0"}, "TORCHINDUCTOR_BATCH_INVARIANT='0' \\(the mode sets '1'\\)"),
    ("batch_invariant", {"VLLM_BATCH_INVARIANT": "2"}, "the mode sets '1'"),
    ("batch_invariant", {"TORCHINDUCTOR_DETERMINISTIC": "1"}, "TORCHINDUCTOR_DETERMINISTIC='1' \\(the mode leaves it off\\)"),
])
def test_a_kernel_switch_set_against_the_registered_mode_refuses_the_run(cfg, tmp_path, monkeypatch, kernels, env, match):
    cfg.write(model_entry(kernels=kernels), decoding_dict())
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(g.RunRefused, match=match):
        _vllm_run(cfg, tmp_path)
    assert not (tmp_path / "out").exists()
    assert all(g.os.environ.get(n) == env.get(n) for n in g.KERNEL_SWITCHES)   # nothing was set


def test_kernel_switches_are_not_set_after_torch_is_imported(cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))
    with pytest.raises(g.RunRefused, match="torch is already imported"):
        _vllm_run(cfg, tmp_path)
    for name, value in g.KERNEL_MODES["batch_invariant"].items():        # exported before Python started: fine
        monkeypatch.setenv(name, value)
    assert _vllm_run(cfg, tmp_path)["kernels"]["set_by_runner"] == []


def test_the_echo_backend_ignores_the_batch_invariant_switch(cap_cfg, tmp_path, monkeypatch):
    plain = run_known(cap_cfg, tmp_path, "plain")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    flagged = run_known(cap_cfg, tmp_path, "flagged")
    assert plain["batch_invariant"] is False and flagged["batch_invariant"] is False
    assert plain["run_fingerprint"] == flagged["run_fingerprint"]
    assert flagged["kernels"]["set_by_runner"] == [] and "TORCHINDUCTOR_BATCH_INVARIANT" not in g.os.environ


def test_format_throughput_prints_the_exceedance_line_and_the_override(cap_cfg, tmp_path):
    text = g.format_throughput(run_known(cap_cfg, tmp_path, "p")["throughput"])
    assert "cap override" not in text
    assert "    share longer than N tokens (cap 200): 64 0.700, 128 0.400" in text
    assert "    share longer than N tokens (cap 1024): 64 1.000, 128 0.900, 256 0.800, 512 0.500" in text
    assert "truncated at cap 0.200" in text                      # the existing line is unchanged
    over = g.format_throughput(run_known(cap_cfg, tmp_path, "q", cap_override={"direct_answer": 4096})["throughput"])
    assert "[cap override: direct_answer=4096]" in over
    assert "(cap 4096): 64 0.700, 128 0.400, 256 0.200, 512 0.000, 1024 0.000, 2048 0.000" in over


def test_measure_cap_is_refused_outside_throughput_mode(cap_cfg, tmp_path):
    req = known_length_requests(tmp_path / "known.parquet")
    for kw in ({}, {"limit": 5}):
        with pytest.raises(g.RunRefused, match="--throughput"):
            cap_cfg.run(req, tmp_path / "o", cap_override={"direct_answer": 1024}, **kw)
    assert not (tmp_path / "o").exists()


def test_measure_cap_refuses_unknown_arm_and_caps_beyond_max_model_len(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    c = Cfg(tmp_path, entry=model_entry(max_model_len=2048),
            decoding=decoding_dict(max_new_tokens={"direct_answer": 200, "zero_shot_cot": 1024}))
    req = known_length_requests(tmp_path / "known.parquet")
    with pytest.raises(g.RunRefused, match="no max_new_tokens"):
        c.run(req, tmp_path / "o", throughput=True, limit=8, cap_override={"mystery": 512})
    for n in (2048, 4096):                                      # no prompt fits with n new tokens
        with pytest.raises(g.RunRefused, match="max_model_len"):
            c.run(req, tmp_path / "o", throughput=True, limit=8, cap_override={"direct_answer": n})
    assert not (tmp_path / "o").exists()
    assert c.run(req, tmp_path / "o", throughput=True, limit=8,
                 cap_override={"direct_answer": 1500})["status"] == "complete"


def test_measure_cap_refuses_when_the_longest_prompt_plus_cap_exceeds_max_model_len(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {"count": 0})
    c = Cfg(tmp_path, entry=model_entry(max_model_len=450),
            decoding=decoding_dict(max_new_tokens={"direct_answer": 100, "zero_shot_cot": 100}))
    req = make_requests(tmp_path / "r.parquet", n_items=2, configs=("c0",), same_as={},
                        content=lambda s, a, i: " ".join(["w"] * 100))
    assert c.run(req, tmp_path / "ok", throughput=True, limit=4)["status"] == "complete"   # fits as configured
    with pytest.raises(g.RunRefused, match="max_model_len=450") as ei:
        c.run(req, tmp_path / "o", throughput=True, limit=4, cap_override={"direct_answer": 400})
    m = re.search(r"longest prompt is (\d+) tokens, so a cap of at most (\d+) fits", str(ei.value))
    assert m and int(m.group(1)) + int(m.group(2)) == 450 and int(m.group(1)) > 100
    assert c.run(req, tmp_path / "fits", throughput=True, limit=4,
                 cap_override={"direct_answer": int(m.group(2))})["status"] == "complete"


# ---------------------------------------------------------------- command line

def _load_cli():
    spec = importlib.util.spec_from_file_location("generate_d2_cli", ROOT / "scripts" / "generate_d2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_exit_codes_and_print_tp(cfg, tmp_path, capsys):
    cli = _load_cli()
    req = make_requests(tmp_path / "r.parquet")
    base = ["--backend", "echo", "--model-key", KEY, "--models-yaml", str(cfg.models),
            "--decoding-yaml", str(cfg.decoding), "--budget-yaml", str(cfg.budget)]
    assert cli.main(base + ["--print-tp"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert cli.main(base + ["--requests", str(req), "--out-dir", str(tmp_path / "o")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert cli.main(["--backend", "echo", "--model-key", "nope", "--models-yaml", str(cfg.models),
                     "--requests", str(req), "--out-dir", str(tmp_path / "o2")]) == 2
    assert cli.main(base) == 2                                           # --requests and --out-dir missing
    assert cli.main(["--model-key", "_template", "--print-tp"]) == 2     # the schema template is not a model


def test_cli_measure_cap(cap_cfg, tmp_path, capsys):
    cli = _load_cli()
    req = known_length_requests(tmp_path / "known.parquet")
    base = ["--backend", "echo", "--model-key", KEY, "--models-yaml", str(cap_cfg.models),
            "--decoding-yaml", str(cap_cfg.decoding), "--budget-yaml", str(cap_cfg.budget),
            "--requests", str(req)]
    # refused without --throughput, with and without --limit
    for extra in ([], ["--limit", "5"]):
        assert cli.main(base + ["--out-dir", str(tmp_path / "o1"), "--measure-cap", "direct_answer=1024"] + extra) == 2
        assert "--measure-cap is allowed only" in capsys.readouterr().err
    assert not (tmp_path / "o1").exists()
    # malformed values, a repeated arm, an unknown arm and a cap beyond max_model_len (8192) are refused
    for bad in (["--measure-cap", "direct_answer"], ["--measure-cap", "direct_answer=0"],
                ["--measure-cap", "direct_answer=5", "--measure-cap", "direct_answer=6"],
                ["--measure-cap", "mystery=500"], ["--measure-cap", "direct_answer=8192"]):
        assert cli.main(base + ["--out-dir", str(tmp_path / "o2"), "--throughput", "--limit", "8"] + bad) == 2
        assert capsys.readouterr().err.startswith("REFUSED")
    assert not (tmp_path / "o2").exists()
    # accepted and repeated with --throughput; the echo lengths are 1..300 or the LEN= markers, below 1024 and 2048
    flags = ["--throughput", "--limit", "20", "--measure-cap", "direct_answer=1024",
             "--measure-cap", "zero_shot_cot=2048"]
    assert cli.main(base + ["--out-dir", str(tmp_path / "o3")] + flags) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    (log,) = read_log(tmp_path / "o3" / "throughput")
    assert log["cap_override"]["arms"] == {"direct_answer": 1024, "zero_shot_cot": 2048}
    assert log["decoding"]["max_new_tokens"] == {"direct_answer": 1024, "zero_shot_cot": 2048}
    assert log["throughput"]["per_arm"]["zero_shot_cot"]["max_new_tokens"] == 2048
    # without the flag the configured caps stay in force
    assert cli.main(base + ["--out-dir", str(tmp_path / "o4"), "--throughput", "--limit", "20"]) == 0
    capsys.readouterr()
    (plain,) = read_log(tmp_path / "o4" / "throughput")
    assert plain["cap_override"] is None
    assert plain["decoding"]["max_new_tokens"] == {"direct_answer": 200, "zero_shot_cot": 1024}
    assert plain["run_fingerprint"] != log["run_fingerprint"]


def test_cli_documents_measure_cap():
    help_text = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_d2.py"), "--help"],
                               capture_output=True, text=True, check=True).stdout
    assert "--measure-cap ARM=N" in help_text


# ---------------------------------------------------------------- container and job files

def _arg_defaults(text):
    return dict(re.findall(r"^ARG (\w+)=(\S+)\s*$", text, flags=re.M))


def _bash():
    found = shutil.which("bash")
    if found and "system32" not in found.lower():                          # WSL's bash cannot read a Windows path
        return found
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.exists():
        return str(git_bash)
    pytest.skip("no POSIX bash on this machine")


# ---------------------------------------------------------------- budget gate for bulk throughput runs

def _budget(path, money=None, gpu=None, disc=None, val=None):
    path.write_text(yaml.safe_dump({"cap": {"money_usd": money, "gpu_hours": gpu},
                                    "dataset2_allocation": {"discovery_units_per_arm": disc,
                                                            "validation_units_per_arm": val}}))
    return path


def test_budget_gate_is_split_into_cap_and_allocation_checks(tmp_path):
    b = _budget(tmp_path / "b.yaml")
    assert g.budget_cap_problems(b) == ["cap.money_usd and cap.gpu_hours are both unset"]
    assert g.budget_gate_problems(b) == ["cap.money_usd and cap.gpu_hours are both unset",
                                         "dataset2_allocation.discovery_units_per_arm is unset",
                                         "dataset2_allocation.validation_units_per_arm is unset"]
    b = _budget(tmp_path / "b2.yaml", money=50.0)
    assert g.budget_cap_problems(b) == []                                  # the cap alone satisfies the cap check
    assert g.budget_gate_problems(b) == ["dataset2_allocation.discovery_units_per_arm is unset",
                                         "dataset2_allocation.validation_units_per_arm is unset"]
    assert g.budget_gate_problems(_budget(tmp_path / "b3.yaml", gpu=10, disc=12, val=12)) == []
    missing = tmp_path / "absent.yaml"
    assert "cannot read" in g.budget_cap_problems(missing)[0] and "cannot read" in g.budget_gate_problems(missing)[0]


def test_bulk_throughput_run_needs_the_cap(cfg, tmp_path):
    spec, dec = cfg.spec(), cfg.dec()
    b = _budget(tmp_path / "b.yaml")                                       # nothing set
    for limit in (g.PILOT_MAX_REQUESTS + 1, 60000, None):
        with pytest.raises(g.RunRefused, match="budget gate") as e:
            g.preflight(spec, dec, backend="vllm", throughput=True, limit=limit, budget_path=b)
        assert "cap.money_usd and cap.gpu_hours are both unset" in str(e.value)
        assert "dataset2_allocation" not in str(e.value)                  # the allocation is not asked for
    # a pilot-sized throughput run is exempt, at the boundary and below it
    g.preflight(spec, dec, backend="vllm", throughput=True, limit=g.PILOT_MAX_REQUESTS, budget_path=b)
    g.preflight(spec, dec, backend="vllm", throughput=True, limit=512, budget_path=b)
    # the echo backend never runs the gate
    g.preflight(spec, dec, backend="echo", throughput=True, limit=60000, budget_path=b)


@pytest.mark.parametrize("money, gpu", [(100.0, None), (None, 24.0), (100.0, 24.0)])
def test_bulk_throughput_run_does_not_need_the_allocation(cfg, tmp_path, money, gpu):
    b = _budget(tmp_path / "b.yaml", money=money, gpu=gpu)                 # cap set, allocation unset
    rec = g.preflight(cfg.spec(), cfg.dec(), backend="vllm", throughput=True, limit=60000, budget_path=b)
    assert "flashinfer_sampler" in rec
    with pytest.raises(g.RunRefused, match="dataset2_allocation"):         # a full run still needs both
        g.preflight(cfg.spec(), cfg.dec(), backend="vllm", throughput=False, limit=60000, budget_path=b)


def test_non_throughput_budget_gate_is_unchanged(tmp_path):
    spec = g.parse_model_spec(KEY, {KEY: model_entry(chat_template={"source": "tokenizer_default", "file": None,
                                                                    "sha256": "b" * 64})})
    ok = g.load_decoding(Cfg(tmp_path).decoding)
    b = _budget(tmp_path / "b2.yaml", money=100)
    with pytest.raises(g.RunRefused, match="dataset2_allocation"):
        g.preflight(spec, ok, backend="vllm", throughput=False, limit=None, budget_path=b)
    g.preflight(spec, ok, backend="vllm", throughput=False, limit=100, budget_path=b)               # pilot
    g.preflight(spec, ok, backend="vllm", throughput=False, limit=None,
                budget_path=_budget(tmp_path / "b3.yaml", money=100, disc=12, val=12))


def _build_record(folder, kind="screening", split="train", excluded=True, model=KEY, sha=None):
    req = make_requests(folder / f"{KEY}.parquet")
    rec = {"kind": kind, "split": split, "purpose": "admissibility_screening", "excluded_from_main": excluded,
           "models": {model: {"requests_file": req.name, "requests_sha256": sha or g.sha256_file(req)}}}
    (folder / "build_record.json").write_text(json.dumps(rec), encoding="utf-8")
    return req


def test_screening_and_tranche_requests_need_the_cap_only(tmp_path):
    spec = g.parse_model_spec(KEY, {KEY: model_entry(chat_template={"source": "tokenizer_default", "file": None,
                                                                    "sha256": "b" * 64})})
    ok = g.load_decoding(Cfg(tmp_path).decoding)
    for kind in ("screening", "tranche"):
        d = tmp_path / kind
        d.mkdir()
        cal = g.calibration_build(_build_record(d, kind=kind), KEY)
        assert cal["kind"] == kind
        g.preflight(spec, ok, backend="vllm", throughput=False, limit=None, calibration=cal,
                    budget_path=_budget(tmp_path / "cap.yaml", gpu=40))              # cap set, allocation unset
        with pytest.raises(g.RunRefused, match="need the cap only") as e:
            g.preflight(spec, ok, backend="vllm", throughput=False, limit=None, calibration=cal,
                        budget_path=_budget(tmp_path / "none.yaml"))
        assert "dataset2_allocation" not in str(e.value)
    # every mismatch falls back to the full gate
    for i, kw in enumerate([{"kind": "main"}, {"split": "test"}, {"excluded": False}, {"model": "other"},
                            {"sha": "0" * 64}]):
        d = tmp_path / f"bad{i}"
        d.mkdir()
        assert g.calibration_build(_build_record(d, **kw), KEY) is None, kw
    d = tmp_path / "no_record"
    d.mkdir()
    assert g.calibration_build(make_requests(d / f"{KEY}.parquet"), KEY) is None
    with pytest.raises(g.RunRefused, match="dataset2_allocation"):
        g.preflight(spec, ok, backend="vllm", throughput=False, limit=None, calibration=None,
                    budget_path=_budget(tmp_path / "cap2.yaml", gpu=40))


def test_a_model_may_set_its_own_scheduler_limits_outside_the_fingerprint(cfg):
    base = g.parse_model_spec(KEY, {KEY: model_entry()})
    own = g.parse_model_spec(KEY, {KEY: model_entry(engine={"max_num_seqs": 512})})
    assert base.engine == {} and own.engine == {"max_num_seqs": 512} and own.to_log()["engine"] == {"max_num_seqs": 512}
    dec = cfg.dec()
    assert g.run_fingerprint(base, dec, "vllm") == g.run_fingerprint(own, dec, "vllm")
    for bad, match in (({"gpu_memory_utilization": 0.95}, "not a per-model setting"),
                       ({"max_num_seqs": 0}, "positive integer"), ({"max_num_seqs": True}, "positive integer"),
                       ([512], "must be a mapping")):
        with pytest.raises(g.RunRefused, match=match):
            g.parse_model_spec(KEY, {KEY: model_entry(engine=bad)})


def test_unreadable_budget_refuses_a_bulk_throughput_run(cfg, tmp_path):
    with pytest.raises(g.RunRefused, match="cannot read"):
        g.preflight(cfg.spec(), cfg.dec(), backend="vllm", throughput=True, limit=60000,
                    budget_path=tmp_path / "absent.yaml")


def test_bulk_throughput_run_is_refused_before_anything_is_written(cfg, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(g.RunRefused, match="budget gate"):
        g.run_generation(make_requests(tmp_path / "r.parquet"), KEY, out, backend="vllm", throughput=True,
                         limit=g.PILOT_MAX_REQUESTS + 1, models_path=cfg.models, decoding_path=cfg.decoding,
                         budget_path=cfg.budget, verbose=False)
    assert not out.exists() and "vllm" not in sys.modules


def test_generate_d2_docstring_points_throughput_runs_at_the_train_split_tranche_file():
    text = (ROOT / "scripts" / "generate_d2.py").read_text(encoding="utf-8")
    doc = text.split('"""')[1]
    assert "data/d2/requests.parquet" not in doc
    assert doc.count("data/requests/gsm8k/tranche/<key>.parquet") == 2        # both examples
    assert r"data\requests\gsm8k\tranche\<key>.parquet" in doc           # the build_requests output path
    assert "--tranche --split train --calibration-items N" in doc
    assert "never on the test-item" in doc
    throughput_examples = [i for i, ln in enumerate(doc.splitlines()) if "--throughput" in ln and "generate_d2.py" in ln]
    assert len(throughput_examples) == 2
    for i in throughput_examples:
        block = " ".join(doc.splitlines()[i:i + 4])
        assert "data/requests/gsm8k/tranche/<key>.parquet" in block
