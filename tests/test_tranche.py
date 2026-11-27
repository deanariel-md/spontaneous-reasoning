"""Tests for the throughput-tranche measurement and cost projection (src/tranche.py, scripts/analyze_tranche.py).

Fixtures come from the real runner (src/generation.py, EchoBackend, throughput mode) written into tmp_path,
and from fabricated minimal run logs. No file is written under results/.
"""
import importlib.util
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src import generation as g
from src import tranche as tr

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40
CAPS = {"direct_answer": 60, "zero_shot_cot": 400}


# ---------------------------------------------------------------- fixtures from the real runner

def _model_entry(tp):
    return {"family": "test", "hf_id": "org/model", "revision": SHA, "dtype": "bfloat16", "quantization": "none",
            "tokenizer_mode": "auto", "max_model_len": 4096, "tensor_parallel_size": tp,
            "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
            "thinking_off": {"method": "chat_template_kwargs", "chat_template_kwargs": {"enable_thinking": False},
                             "system_prompt_token": None, "rationale": None}, "kernels": "batch_invariant"}


def _decoding():
    return {"status": "PENDING_REVIEW", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
            "max_new_tokens": dict(CAPS), "stop": [], "include_stop_str_in_output": False,
            "skip_special_tokens": True,
            "engine": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enforce_eager": False,
                       "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto"}}


class Env:
    """models.yaml with m1 (tp 2) and m2 (tp 1), decoding.yaml, budget.yaml, request files, echo runs."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.models = tmp_path / "models.yaml"
        self.decoding = tmp_path / "decoding.yaml"
        self.budget = tmp_path / "budget.yaml"
        self.models.write_text(yaml.safe_dump({"dataset2": {"models": {"m1": _model_entry(2), "m2": _model_entry(1)}}}))
        self.decoding.write_text(yaml.safe_dump(_decoding()))
        self.write_budget(None, None)

    def write_budget(self, gpu_hours, money_usd):
        self.budget.write_text(yaml.safe_dump({"cap": {"money_usd": money_usd, "gpu_hours": gpu_hours},
                                               "dataset2_allocation": {"discovery_units_per_arm": None,
                                                                       "validation_units_per_arm": None}}))

    def requests(self, key, n_items=5, configs=("c0", "c1", "c2")):
        """Configs c0 and c2 render identical prompts, so every (arm, item) of c2 is a cached request."""
        rows = []
        for c in configs:
            skel = "c0" if c == "c2" else c
            for arm in ("direct_answer", "zero_shot_cot"):
                for it in range(n_items):
                    text = f"skeleton {skel} for {arm} question {it}"
                    rows.append({"request_id": f"{c}|{arm}|{it}", "model_key": key, "arm": arm, "config_id": c,
                                 "item_id": it, "messages_json": json.dumps([{"role": "user", "content": text}])})
        path = self.tmp / f"req_{key}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        return path

    def run(self, key, *, throughput=True, limit=None, name=None):
        out = self.tmp / (name or f"out_{key}_{'tp' if throughput else 'full'}")
        spec = g.parse_model_spec(key, g.load_registry(self.models))
        echo = g.EchoBackend(spec, g.load_decoding(self.decoding), delay_s=0.02)   # generation_s must be positive
        g.run_generation(self.requests(key), key, out, backend="echo", backend_obj=echo, throughput=throughput,
                         limit=limit if limit is not None else (30 if throughput else None), shard_size=1000,
                         models_path=self.models, decoding_path=self.decoding, budget_path=self.budget,
                         verbose=False)
        return out


@pytest.fixture(autouse=True)
def fake_gpu(monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 2, "devices": [{"name": "Fake A100", "memory_mib": 81920}] * 2, "driver_version": "580.1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ---------------------------------------------------------------- fabricated minimal run logs

FAB_LENGTHS = {"direct_answer": [10, 20, 30, 40], "zero_shot_cot": [100, 200, 300, 1024]}   # means 25 and 406
FAB_CAPS = {"direct_answer": 128, "zero_shot_cot": 1024}
FAB_PROMPT = {"direct_answer": 50, "zero_shot_cot": 70}


def fabricate(dir_, *, model_key="fake", run_id="20261010T000000Z-aaaaaa", lengths=None, caps=None, tp=2,
              tps=1000.0, generation_s=10.0, drop=(), drop_columns=(), log_over=None, shard_meta=None,
              backend="vllm", status="complete", mode="throughput", excluded=None, prompt=None):
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    lengths = lengths or FAB_LENGTHS
    caps = caps or FAB_CAPS
    prompt = prompt or FAB_PROMPT
    rows = []
    for arm, ls in lengths.items():
        for i, n in enumerate(ls):
            rows.append({"request_id": f"{arm}|{i}", "model_key": model_key, "arm": arm, "config_id": "c0",
                         "item_id": i, "prompt_sha256": "0" * 64, "output_text": "t", "n_prompt_tokens": prompt[arm],
                         "n_output_tokens": n, "finish_reason": "length" if n >= caps[arm] else "stop",
                         "cached": False})
    df = pd.DataFrame(rows).drop(columns=list(drop_columns))
    meta = {"throughput": "true", "run_id": run_id, "run_fingerprint": "fp"}
    meta.update(shard_meta or {})
    shard = f"shard-{run_id}-00000.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(meta), dir_ / shard)
    log = {"schema_version": "1", "run_id": run_id, "mode": mode,
           "excluded_from_main": (mode == "throughput") if excluded is None else excluded,
           "status": status, "started_utc": "2026-10-10T00:00:00Z", "run_fingerprint": "fp",
           "gpu": {"count": 2, "devices": [{"name": "Fake H100", "memory_mib": 81559}] * 2,
                   "driver_version": "580.1", "cuda_version_driver": "13.0", "nvidia_smi": True},
           "model": {"model_key": model_key, "hf_id": "org/model", "revision": SHA, "tensor_parallel_size": tp,
                     "kernels": "batch_invariant"},
           "decoding": {"status": "PENDING_REVIEW", "max_new_tokens": dict(caps)}, "decoding_confirmed": False,
           "backend": {"name": backend}, "requests_this_run": len(rows), "shards_written": [shard],
           "counts": {"unique_generations": {a: len(v) for a, v in lengths.items()}},
           "generation_s": generation_s, "model_load_s": 30.0, "output_tokens_per_s": tps}
    log.update(log_over or {})
    for path in drop:
        keys = path.split(".")
        cur = log
        for k in keys[:-1]:
            cur = cur[k]
        del cur[keys[-1]]
    (dir_ / f"throughput_log-{run_id}.json").write_text(json.dumps(log), encoding="utf-8")
    return dir_


NO_BANK = tr.Bank("none", None, "none")
NO_CAPS = {"gpu_hours": None, "money_usd": None, "source": "x"}


# ---------------------------------------------------------------- module hygiene

def test_module_imports_without_vllm_or_torch():
    code = "import sys, src.tranche; assert 'vllm' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_arms_match_the_task_config():
    tasks = yaml.safe_load((ROOT / "config" / "tasks.yaml").read_text(encoding="utf-8"))
    iv = tasks["dataset2"]["intervention"]
    assert tr.ARMS == (iv["baseline"], iv["intervention"])
    assert set(tr.ARMS) == set(g.load_decoding().arms)
    assert tr.K_GRID == (8, 12, 16, 24)


def test_committed_budget_yaml_loads():
    caps = tr.load_caps()
    assert set(tr.CAP_KEYS) <= set(caps)
    assert all(caps[k] is None or caps[k] >= 0 for k in tr.CAP_KEYS)


# ---------------------------------------------------------------- generation counts

def test_plan_generation_counts_without_duplication(tmp_path):
    assert tr.generations_before_caching(20, 1319, 12) == 1_266_240
    assert tr.generations_before_caching(20, 1319, 24) == 2_532_480
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    by_k = {r["K"]: r for r in rows}
    assert by_k[12]["generations_before_caching"] == 1_266_240 and by_k[24]["generations_before_caching"] == 2_532_480
    for k, r in by_k.items():
        assert r["expected_distinct_units"] == 2 * k
        assert r["unique_generations"] == r["generations_before_caching"] == 20 * 1319 * 2 * 2 * k
    assert [r["K"] for r in rows] == [8, 12, 16, 24]


# ---------------------------------------------------------------- distinct configurations

@pytest.mark.parametrize("probs", [(0.5, 0.3, 0.2), (0.25, 0.25, 0.25, 0.25), (0.7, 0.2, 0.1, 0.0)])
def test_expected_distinct_equals_exact_enumeration(probs):
    n = len(probs)
    for draws in range(1, 7):
        exact = 0.0
        for seq in itertools.product(range(n), repeat=draws):
            exact += math.prod(probs[i] for i in seq) * len(set(seq))
        assert tr.expected_distinct(probs, draws) == pytest.approx(exact, abs=1e-12)


def test_expected_distinct_matches_monte_carlo():
    w = np.array([1.0 / (i + 1) for i in range(30)])
    p = w / w.sum()
    rng = np.random.default_rng(12345)
    reps, draws = 40_000, 24                     # K = 12, so 2K draws
    s = np.sort(rng.choice(len(p), size=(reps, draws), p=p), axis=1)
    distinct = 1 + (np.diff(s, axis=1) != 0).sum(axis=1)
    se = distinct.std(ddof=1) / math.sqrt(reps)
    assert abs(distinct.mean() - tr.expected_distinct(p, draws)) < 4 * se


def test_expected_distinct_properties():
    n = 100
    for k in tr.K_GRID:
        e = tr.expected_distinct([1.0 / n] * n, 2 * k)
        assert e == pytest.approx(n * (1 - (1 - 1 / n) ** (2 * k)), rel=1e-12)
        assert 1 <= e <= min(2 * k, n)
    es = [tr.expected_distinct([1.0 / n] * n, 2 * k) for k in tr.K_GRID]
    assert es == sorted(es)
    assert tr.expected_distinct([1.0], 5) == 1.0
    assert tr.expected_distinct([0.5, 0.5], 0) == 0.0
    with pytest.raises(ValueError):
        tr.expected_distinct([1.0], -1)


def test_duplicates_lower_unique_generations_only(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    bank = tr.Bank("uniform", tuple([1 / 50] * 50), "uniform")
    rows = tr.project_rows([m], models=20, items=1319, bank=bank, usd_per_gpu_hour=None, caps=NO_CAPS)
    for r in rows:
        e = tr.expected_distinct([1 / 50] * 50, 2 * r["K"])
        assert r["expected_distinct_units"] == pytest.approx(e)
        assert r["unique_generations"] == pytest.approx(20 * 1319 * 2 * e)
        assert r["unique_generations"] < r["generations_before_caching"]
        assert r["generations_before_caching"] == 20 * 1319 * 2 * 2 * r["K"]


# ---------------------------------------------------------------- bank resolution

def _bank_file(path, probs, **extra):
    path.write_text(json.dumps({"sampling_probabilities": probs, **extra}), encoding="utf-8")
    return path


def test_bank_file_wins_then_bank_size_then_none(tmp_path, monkeypatch):
    f = _bank_file(tmp_path / "bank.json", {"a": 0.5, "b": 0.25, "c": 0.25})
    monkeypatch.setattr(tr, "BANK_JSON", tmp_path / "absent.json")
    assert tr.resolve_bank().kind == "none"
    none = tr.resolve_bank()
    assert none.expected_distinct(24) == 24.0 and "distinct" in none.source
    uni = tr.resolve_bank(bank_size=10)
    assert uni.kind == "uniform" and uni.n_configs == 10 and uni.probabilities[0] == pytest.approx(0.1)
    monkeypatch.setattr(tr, "BANK_JSON", f)
    default = tr.resolve_bank(bank_size=10)             # the default file exists and takes precedence
    assert default.kind == "file" and default.n_configs == 3 and "ignored" in default.note
    assert default.expected_distinct(4) == pytest.approx(tr.expected_distinct([0.5, 0.25, 0.25], 4))
    assert default.sha256 and tr.resolve_bank(f).kind == "file"
    with pytest.raises(tr.TrancheError, match="does not exist"):
        tr.resolve_bank(tmp_path / "nope.json")
    with pytest.raises(tr.TrancheError, match="positive"):
        tr.resolve_bank(bank_size=0)


@pytest.mark.parametrize("probs, text", [
    ({"a": 0.5, "b": 0.4}, "sum"),
    ({"a": 1.2, "b": -0.2}, "number in"),
    ({"a": "0.5", "b": 0.5}, "number in"),
    ({}, "non-empty"),
])
def test_bank_file_is_validated_and_never_rescaled(tmp_path, probs, text):
    with pytest.raises(tr.TrancheError, match=text):
        tr.load_bank_probabilities(_bank_file(tmp_path / "b.json", probs))


def test_bank_file_without_the_key_is_refused(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"configs": []}), encoding="utf-8")
    with pytest.raises(tr.TrancheError, match="sampling_probabilities"):
        tr.load_bank_probabilities(p)


# ---------------------------------------------------------------- cap comparison

@pytest.mark.parametrize("gpu, cost, caps, summary", [
    (100.0, 500.0, {"gpu_hours": None, "money_usd": None}, "cap not set"),
    (None, None, {"gpu_hours": None, "money_usd": None}, "cap not set"),
    (100.0, None, {"gpu_hours": 150.0, "money_usd": None}, "within"),
    (150.0, None, {"gpu_hours": 150.0, "money_usd": None}, "within"),          # equal to the cap
    (150.5, None, {"gpu_hours": 150.0, "money_usd": None}, "above"),
    (100.0, 900.0, {"gpu_hours": 150.0, "money_usd": 800.0}, "above"),        # money above, GPU-hours within
    (100.0, 700.0, {"gpu_hours": 150.0, "money_usd": 800.0}, "within"),
    (100.0, None, {"gpu_hours": 150.0, "money_usd": 800.0}, "not evaluable"),  # money cap set, no price
    (200.0, None, {"gpu_hours": 150.0, "money_usd": 800.0}, "above"),          # one violation is enough
    (None, None, {"gpu_hours": 150.0, "money_usd": None}, "not evaluable"),
    (None, 10.0, {"gpu_hours": None, "money_usd": 800.0}, "within"),
])
def test_compare_cap(gpu, cost, caps, summary):
    assert tr.compare_cap(gpu, cost, caps)[0] == summary


def test_compare_cap_detail_and_rows_use_budget_caps(tmp_path):
    s, d = tr.compare_cap(100.0, None, {"gpu_hours": 150.0, "money_usd": 800.0})
    assert d == {"gpu_hours": "within", "money_usd": "not evaluable"}
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    caps = {"gpu_hours": 150.0, "money_usd": None, "source": "x"}
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=caps)
    status = {r["K"]: r["within_cap"] for r in rows}
    # 20 x 1319 x 2 x 2K generations at 431 mean tokens per generation pair, 1000 tok/s, 2 GPUs
    assert status == {8: "within", 12: "above", 16: "above", 24: "above"}
    assert rows[0]["gpu_hours"] < 150.0 < rows[1]["gpu_hours"]


def test_load_caps_reads_budget_yaml(env):
    env.write_budget(300, 1200.5)
    caps = tr.load_caps(env.budget)
    assert caps["gpu_hours"] == 300.0 and caps["money_usd"] == 1200.5
    env.write_budget(None, None)
    assert tr.load_caps(env.budget)["gpu_hours"] is None
    env.budget.write_text(yaml.safe_dump({"cap": {"gpu_hours": -1, "money_usd": None}}))
    with pytest.raises(tr.TrancheError, match="gpu_hours"):
        tr.load_caps(env.budget)


# ---------------------------------------------------------------- measurement from the real runner

def test_measurement_matches_the_runners_own_report(env):
    out = env.run("m1", throughput=True, limit=30)           # all 3 configs x 2 arms x 5 items
    tdir = out / "throughput"
    m = tr.load_throughput_run(tdir, allow_non_vllm=True)
    rep = json.loads(next(tdir.glob("throughput_report-*.json")).read_text(encoding="utf-8"))
    log = json.loads(next(tdir.glob("throughput_log-*.json")).read_text(encoding="utf-8"))
    assert m["model_key"] == "m1" and m["backend"] == "echo" and m["run_id"] == log["run_id"]
    assert m["requests"] == 30 and m["unique_generations"] == 20       # c2 repeats c0
    assert m["gpu_name"] == "Fake A100" and m["gpu_count"] == 2 and m["tensor_parallel_size"] == 2
    assert m["kernels"] == log["model"]["kernels"] == "batch_invariant"
    assert m["output_tokens_per_s"] == log["output_tokens_per_s"] == rep["output_tokens_per_s"]
    assert m["output_tokens_per_s_source"] == "run log" and m["generation_s"] == log["generation_s"]
    assert m["requests_per_s"] == pytest.approx(30 / log["generation_s"])
    assert m["unique_generations_per_s"] == pytest.approx(20 / log["generation_s"])
    assert m["missing_fields"] == [] and m["decoding_status"] == "PENDING_REVIEW"
    shares = []
    for arm in tr.ARMS:
        b, r = m["per_arm"][arm], rep["per_arm"][arm]
        assert b["cap"] == CAPS[arm] and b["requests"] == 15 and b["unique_generations"] == 10
        assert b["cached_requests"] == 5 and b["unique_generations"] == r["unique_generations"]
        o, ro = b["output_tokens"], r["output_tokens"]
        assert o["n"] == ro["n"] and o["mean"] == pytest.approx(ro["mean"]) and o["p95"] == pytest.approx(ro["p95"])
        assert o["median"] == pytest.approx(ro["p50"]) and o["max"] == ro["max"]
        assert b["prompt_tokens_mean"] == pytest.approx(r["prompt_tokens"]["mean"])
        assert b["share_truncated"] == pytest.approx(r["share_truncated_at_cap"])
        shares.append(b["share_truncated"])
    assert shares[0] > 0.0 and shares[1] == 0.0                      # the direct arm cap (60) truncates some echoes
    # recompute from the shards
    df = g.read_outputs(tdir)
    uni = df[~df["cached"]]
    d = uni[uni["arm"] == "direct_answer"]
    assert m["per_arm"]["direct_answer"]["output_tokens"]["mean"] == pytest.approx(d["n_output_tokens"].mean())
    assert m["per_arm"]["direct_answer"]["share_truncated"] == pytest.approx((d["finish_reason"] == "length").mean())


def test_the_kernel_mode_of_each_measured_run_is_reported(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    assert m["kernels"] == "batch_invariant" and "model.kernels" not in m["missing_fields"]
    old = tr.load_throughput_run(fabricate(tmp_path / "y", drop=("model.kernels",)))
    assert old["kernels"] is None and "model.kernels" in old["missing_fields"]
    text = tr.render_markdown(tr.build_report([tmp_path / "x"], bank=NO_BANK, caps=NO_CAPS))
    assert "tensor_parallel_size 2; kernels batch_invariant." in text


def test_echo_backend_is_refused_unless_allowed(env):
    tdir = env.run("m1") / "throughput"
    with pytest.raises(tr.TrancheError, match="test backend"):
        tr.load_throughput_run(tdir)
    rep = tr.build_report([tdir], bank=NO_BANK, caps=NO_CAPS, allow_non_vllm=True)
    assert any("test backend" in w for w in rep["warnings"])


# ---------------------------------------------------------------- refusals

def test_full_mode_run_directory_is_refused(env):
    full = env.run("m1", throughput=False)
    assert list(full.glob("run_log-*.json")) and list(full.glob("shard-*.parquet"))
    with pytest.raises(tr.TrancheError, match="not a throughput-mode log"):
        tr.load_throughput_run(full, allow_non_vllm=True)


def test_parent_of_the_throughput_folder_is_refused_with_a_hint(env):
    out = env.run("m1")
    with pytest.raises(tr.TrancheError, match="throughput"):
        tr.load_throughput_run(out, allow_non_vllm=True)


def test_main_shards_next_to_throughput_logs_are_refused(env):
    tdir = env.run("m1") / "throughput"
    full = env.run("m1", throughput=False)
    shutil.copy(next(full.glob("shard-*.parquet")), tdir)
    with pytest.raises(tr.TrancheError, match="not marked as a throughput shard"):
        tr.load_throughput_run(tdir, allow_non_vllm=True)


@pytest.mark.parametrize("kw, text", [
    ({"mode": "full"}, "not a throughput-mode log"),
    ({"excluded": False}, "not a throughput-mode log"),
    ({"shard_meta": {"throughput": "false"}}, "not marked as a throughput shard"),
    ({"shard_meta": {"throughput": ""}}, "not marked as a throughput shard"),
    ({"status": "interrupted"}, "no complete throughput run"),
    ({"status": "failed"}, "no complete throughput run"),
    ({"log_over": {"requests_this_run": 99}}, "expects 99 requests"),
    ({"shard_meta": {"run_id": "other"}}, "belongs to run"),
    ({"shard_meta": {"run_fingerprint": "other"}}, "fingerprint"),
])
def test_fabricated_directories_that_do_not_qualify_are_refused(tmp_path, kw, text):
    d = fabricate(tmp_path / "x", **kw)
    with pytest.raises(tr.TrancheError, match=text):
        tr.load_throughput_run(d)


def test_missing_directory_and_empty_directory_are_refused(tmp_path):
    with pytest.raises(tr.TrancheError, match="not a directory"):
        tr.load_throughput_run(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(tr.TrancheError, match="no run log"):
        tr.load_throughput_run(tmp_path / "empty")


def test_two_complete_runs_in_one_folder_are_refused_a_failed_one_is_skipped(tmp_path):
    d = fabricate(tmp_path / "x", run_id="20261010T000000Z-aaaaaa")
    fabricate(d, run_id="20261010T010000Z-bbbbbb", status="failed")
    m = tr.load_throughput_run(d)
    assert m["run_id"] == "20261010T000000Z-aaaaaa"
    assert m["skipped_runs"] == [{"log_file": "throughput_log-20261010T010000Z-bbbbbb.json",
                                  "run_id": "20261010T010000Z-bbbbbb", "status": "failed"}]
    fabricate(d, run_id="20261010T020000Z-cccccc")
    with pytest.raises(tr.TrancheError, match="2 complete runs"):
        tr.load_throughput_run(d)


def test_same_model_twice_and_same_folder_twice_are_refused(tmp_path):
    a = fabricate(tmp_path / "a", model_key="same")
    b = fabricate(tmp_path / "b", model_key="same")
    with pytest.raises(tr.TrancheError, match="more than one folder"):
        tr.build_report([a, b], bank=NO_BANK, caps=NO_CAPS)
    with pytest.raises(tr.TrancheError, match="twice"):
        tr.build_report([a, a], bank=NO_BANK, caps=NO_CAPS)
    with pytest.raises(tr.TrancheError, match="at least one"):
        tr.build_report([], bank=NO_BANK, caps=NO_CAPS)


def test_one_bad_folder_refuses_the_whole_report(env, tmp_path):
    good = fabricate(tmp_path / "good", model_key="good")
    full = env.run("m1", throughput=False)
    with pytest.raises(tr.TrancheError):
        tr.build_report([good, full], bank=NO_BANK, caps=NO_CAPS, allow_non_vllm=True)


# ---------------------------------------------------------------- fabricated logs: exact numbers

def test_fabricated_statistics_are_exact(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    d, c = m["per_arm"]["direct_answer"], m["per_arm"]["zero_shot_cot"]
    assert (d["requests"], d["unique_generations"], d["cached_requests"]) == (4, 4, 0)
    assert d["output_tokens"] == {"n": 4, "mean": 25.0, "median": 25.0, "p95": pytest.approx(38.5), "max": 40}
    assert c["output_tokens"]["mean"] == 406.0 and c["output_tokens"]["median"] == 250.0
    assert c["output_tokens"]["p95"] == pytest.approx(300 + 0.85 * 724) and c["output_tokens"]["max"] == 1024
    assert (d["prompt_tokens_mean"], c["prompt_tokens_mean"]) == (50.0, 70.0)
    assert (d["share_truncated"], c["share_truncated"]) == (0.0, 0.25)
    assert (d["cap"], c["cap"]) == (128, 1024)
    assert m["gpu_name"] == "Fake H100" and m["gpu_count"] == 2 and m["tensor_parallel_size"] == 2
    assert m["output_tokens_per_s"] == 1000.0 and m["requests_per_s"] == pytest.approx(0.8)
    assert m["unique_generations"] == 8 and m["missing_fields"] == []


def test_projection_arithmetic(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x"))
    rows = {r["K"]: r for r in tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=3.0,
                                               caps=NO_CAPS)}
    r = rows[12]
    per_arm = 20 * 1319 * 24                                   # unique generations of one arm
    assert r["unique_generations"] == 2 * per_arm == 1_266_240
    assert r["output_tokens_by_arm"] == {"direct_answer": per_arm * 25.0, "zero_shot_cot": per_arm * 406.0}
    assert r["output_tokens"] == pytest.approx(per_arm * 431.0)
    assert r["prompt_tokens"] == pytest.approx(per_arm * 120.0)
    expected_hours = per_arm * 431.0 / 1000.0 / 3600.0 * 2
    assert r["gpu_hours"] == pytest.approx(expected_hours) and r["gpu_hours_by_model"] == {"fake": r["gpu_hours"]}
    assert r["cost_usd"] == pytest.approx(3.0 * expected_hours) and r["within_cap"] == "cap not set"
    assert rows[24]["gpu_hours"] == pytest.approx(2 * expected_hours)
    assert rows[8]["gpu_hours"] == pytest.approx(expected_hours * 16 / 24)
    none = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(x["cost_usd"] is None for x in none)


def test_several_models_use_per_arm_maximum_tokens_and_the_largest_gpu_hours(tmp_path):
    a = tr.load_throughput_run(fabricate(tmp_path / "a", model_key="a"))                  # tps 1000, tp 2
    b = tr.load_throughput_run(fabricate(tmp_path / "b", model_key="b", tp=1, tps=500.0, run_id="r2",
                                         lengths={"direct_answer": [50, 70], "zero_shot_cot": [300, 500]}))
    rows = {r["K"]: r for r in tr.project_rows([a, b], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None,
                                               caps=NO_CAPS)}
    r = rows[12]
    per_arm = 20 * 1319 * 24
    assert r["output_tokens_by_arm"] == {"direct_answer": per_arm * 60.0, "zero_shot_cot": per_arm * 406.0}
    ha = per_arm * (25 + 406) / 1000 / 3600 * 2
    hb = per_arm * (60 + 400) / 500 / 3600 * 1
    assert r["gpu_hours_by_model"] == {"a": pytest.approx(ha), "b": pytest.approx(hb)}
    assert r["gpu_hours"] == pytest.approx(max(ha, hb)) and r["gpu_hours_binding_model"] == ("b" if hb > ha else "a")


# ---------------------------------------------------------------- missing fields are reported, never filled in

def test_missing_run_log_fields_are_reported_and_projections_are_null(tmp_path):
    d = fabricate(tmp_path / "x", drop=("gpu", "model.tensor_parallel_size", "generation_s", "output_tokens_per_s"))
    m = tr.load_throughput_run(d)
    assert m["gpu_name"] is None and m["gpu_count"] is None and m["tensor_parallel_size"] is None
    assert m["generation_s"] is None and m["output_tokens_per_s"] is None and m["requests_per_s"] is None
    assert {"gpu.devices[].name", "gpu.count", "model.tensor_parallel_size", "generation_s",
            "output_tokens_per_s"} <= set(m["missing_fields"])
    # token statistics come from the shards and stay available
    assert m["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] == 406.0
    caps = {"gpu_hours": 100.0, "money_usd": None, "source": "x"}
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=3.0, caps=caps)
    for r in rows:
        assert r["gpu_hours"] is None and r["cost_usd"] is None and r["within_cap"] == "not evaluable"
        assert r["output_tokens"] is not None and r["gpu_hours_by_model"] == {"fake": None}
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=3.0, caps=NO_CAPS)
    assert all(r["within_cap"] == "cap not set" for r in rows)
    rep = tr.build_report([d], bank=NO_BANK, caps=NO_CAPS)
    assert any("lacks" in w and "generation_s" in w for w in rep["warnings"])
    json.dumps(rep, allow_nan=False)
    assert "missing" in tr.render_markdown(rep)


def test_throughput_is_derived_from_the_shards_when_only_the_rate_is_missing(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x", drop=("output_tokens_per_s",)))
    assert m["output_tokens_per_s"] == pytest.approx((100 + 1624) / 10.0)
    assert m["output_tokens_per_s_source"].startswith("derived") and m["missing_fields"] == []


def test_zero_generation_time_leaves_throughput_missing(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x", generation_s=0.0,
                                         log_over={"output_tokens_per_s": None}))
    assert m["output_tokens_per_s"] is None and m["requests_per_s"] is None
    assert "output_tokens_per_s" in m["missing_fields"]
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(r["gpu_hours"] is None for r in rows)


def test_missing_caps_and_missing_shard_column(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "a", drop=("decoding.max_new_tokens",)))
    assert m["per_arm"]["direct_answer"]["cap"] is None
    assert "decoding.max_new_tokens.direct_answer" in m["missing_fields"]
    assert m["per_arm"]["direct_answer"]["share_truncated"] == 0.0          # still measured from finish_reason
    m = tr.load_throughput_run(fabricate(tmp_path / "b", drop_columns=("cached",)))
    assert m["unique_generations"] is None and "shard column cached" in m["missing_fields"]
    assert m["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] is None
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(r["output_tokens"] is None and r["gpu_hours"] is None for r in rows)
    m = tr.load_throughput_run(fabricate(tmp_path / "c", drop_columns=("n_prompt_tokens",)))
    assert m["per_arm"]["direct_answer"]["prompt_tokens_mean"] is None
    assert m["per_arm"]["direct_answer"]["output_tokens"]["mean"] == 25.0


def test_arm_absent_from_the_tranche_gives_null_projection_not_zero(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x", lengths={"direct_answer": [10, 20]}, caps=FAB_CAPS))
    assert m["per_arm"]["zero_shot_cot"]["requests"] == 0
    assert m["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] is None
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(r["output_tokens"] is None and r["gpu_hours"] is None for r in rows)


# ---------------------------------------------------------------- outputs and command line

def _load_cli():
    spec = importlib.util.spec_from_file_location("analyze_tranche_cli", ROOT / "scripts" / "analyze_tranche.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_json_and_two_identical_markdown_files(env, tmp_path, capsys):
    cli = _load_cli()
    bank = _bank_file(tmp_path / "bank.json", {f"c{i}": 1 / 40 for i in range(40)})
    env.write_budget(500, None)
    d1, d2 = env.run("m1") / "throughput", env.run("m2") / "throughput"
    out = tmp_path / "out"
    rc = cli.main(["--run-dir", str(d1), "--run-dir", str(d2), "--allow-non-vllm", "--bank", str(bank),
                   "--budget-yaml", str(env.budget), "--usd-per-gpu-hour", "2.5", "--out-dir", str(out)])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["cost_projection.json", "cost_projection.md",
                                                     "cost_projection_clean.md"]
    md = (out / "cost_projection.md").read_text(encoding="utf-8")
    assert md == (out / "cost_projection_clean.md").read_text(encoding="utf-8")
    assert not re.search(r"\[\d+\]", md) and "rather than" not in md
    rep = json.loads((out / "cost_projection.json").read_text(encoding="utf-8"),
                     parse_constant=lambda c: pytest.fail(f"non-finite constant {c} in JSON"))
    assert {"measured", "assumptions", "inputs", "rows"} <= set(rep)
    assert [m["model_key"] for m in rep["measured"]["models"]] == ["m1", "m2"]
    assert rep["inputs"]["bank"]["kind"] == "file" and rep["inputs"]["bank"]["n_configs"] == 40
    assert rep["inputs"]["usd_per_gpu_hour"] == 2.5 and rep["inputs"]["caps"]["gpu_hours"] == 500.0
    assert rep["inputs"]["models"] == 20 and rep["inputs"]["items"] == 1319
    assert isinstance(rep["assumptions"], list) and rep["assumptions"]
    assert [r["K"] for r in rep["rows"]] == [8, 12, 16, 24]
    for r in rep["rows"]:
        assert {"K", "expected_distinct_units", "generations_before_caching", "unique_generations", "output_tokens",
                "gpu_hours", "cost_usd", "within_cap"} <= set(r)
        assert r["cost_usd"] == pytest.approx(2.5 * r["gpu_hours"])
        assert r["within_cap"] in ("within", "above")
        assert r["expected_distinct_units"] == pytest.approx(tr.expected_distinct([1 / 40] * 40, 2 * r["K"]))
        assert r["generations_before_caching"] == 20 * 1319 * 2 * 2 * r["K"]
    assert "test backend" in md and "Fake A100" in md and "assumptions" in md.lower()
    assert "paired" not in md and md.startswith("# Dataset 2 throughput tranche")


def test_cli_without_bank_says_so_and_reproduces_plan_counts(env, tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(tr, "BANK_JSON", tmp_path / "absent.json")
    d1 = env.run("m1") / "throughput"
    out = tmp_path / "out"
    assert cli.main(["--run-dir", str(d1), "--allow-non-vllm", "--budget-yaml", str(env.budget),
                     "--out-dir", str(out)]) == 0
    rep = json.loads((out / "cost_projection.json").read_text(encoding="utf-8"))
    rows = {r["K"]: r for r in rep["rows"]}
    assert rows[12]["generations_before_caching"] == rows[12]["unique_generations"] == 1_266_240
    assert rows[24]["generations_before_caching"] == rows[24]["unique_generations"] == 2_532_480
    assert rep["inputs"]["bank"]["kind"] == "none" and rows[12]["expected_distinct_units"] == 24
    assert rows[12]["cost_usd"] is None and rows[12]["within_cap"] == "cap not set"
    md = (out / "cost_projection.md").read_text(encoding="utf-8")
    assert "every draw is taken as distinct" in md and "1,266,240" in md and "2,532,480" in md
    assert "no price given" in md


def test_cli_refuses_and_writes_nothing_for_a_non_throughput_directory(env, tmp_path, capsys):
    cli = _load_cli()
    good = env.run("m1") / "throughput"
    full = env.run("m2", throughput=False)
    out = tmp_path / "out"
    base = ["--allow-non-vllm", "--budget-yaml", str(env.budget), "--bank-size", "10", "--out-dir", str(out)]
    assert cli.main(["--run-dir", str(good), "--run-dir", str(full)] + base) == 2
    assert not out.exists()
    assert "not a throughput-mode log" in capsys.readouterr().err
    assert cli.main(["--run-dir", str(full)] + base) == 2 and not out.exists()
    assert cli.main(["--run-dir", str(full.parent / "does_not_exist")] + base) == 2 and not out.exists()
    assert cli.main(base) == 2 and not out.exists()                               # no --run-dir
    assert cli.main(["--run-dir", str(env.run("m2") / "throughput")] + base[1:]) == 2   # echo without the flag
    assert not out.exists()
    assert cli.main(["--run-dir", str(good), "--bank", str(tmp_path / "nope.json")] + base) == 2
    assert not out.exists()


def test_cli_default_output_directory_is_results_tranche():
    assert tr.OUT_DIR == ROOT / "results" / "tranche"
    assert _load_cli().build_parser().parse_args(["--run-dir", "x"]).out_dir is None


def test_markdown_style_rules(env, tmp_path):
    rep = tr.build_report([fabricate(tmp_path / "x")], bank=tr.Bank("uniform", tuple([0.1] * 10), "a uniform bank"),
                          caps={"gpu_hours": 50.0, "money_usd": 10.0, "source": "budget.yaml"}, usd_per_gpu_hour=2.0)
    md = tr.render_markdown(rep)
    for banned in ("rather than", "instead of", "This section", "The table below", "In this section"):
        assert banned not in md
    assert not re.search(r"\[\d+\]", md)
    assert "| 12 |" in md and "above" in md
    assert "Missing from this run" not in md
    out = tr.write_outputs(rep, tmp_path / "w")
    assert out["md"].read_text(encoding="utf-8") == out["clean_md"].read_text(encoding="utf-8") == md


# ---------------------------------------------------------------- cap-measurement runs (--measure-cap) are projected at the configured caps

CONF_CAPS = {"direct_answer": 128, "zero_shot_cot": 256}


def _override_log(arms=None, configured=None):
    arms = {"zero_shot_cot": 1024} if arms is None else arms
    configured = CONF_CAPS if configured is None else configured
    return {"cap_override": {"arms": arms, "configured_max_new_tokens": configured,
                             "effective_max_new_tokens": {**configured, **arms}, "note": "throughput measurement only"}}


def test_cap_measurement_run_is_projected_at_the_configured_cap(tmp_path):
    m = tr.load_throughput_run(fabricate(tmp_path / "x", log_over=_override_log()))   # run cap 1024, configured 256
    d, c = m["per_arm"]["direct_answer"], m["per_arm"]["zero_shot_cot"]
    assert (c["cap"], c["run_cap"], c["cap_override_applied"]) == (256, 1024, True)
    assert c["output_tokens"]["mean"] == pytest.approx((100 + 200 + 256 + 256) / 4) and c["output_tokens"]["max"] == 256
    assert c["output_tokens_as_generated"]["mean"] == 406.0 and c["output_tokens_as_generated"]["max"] == 1024
    assert c["share_truncated"] == 0.5 and c["share_truncated_at_run_cap"] == 0.25        # lengths 300 and 1024 exceed 256
    assert (d["cap"], d["run_cap"], d["cap_override_applied"]) == (128, 128, False)       # the other arm is untouched
    assert d["output_tokens"]["mean"] == 25.0 and "output_tokens_as_generated" not in d
    assert m["cap_override"]["arms"] == {"zero_shot_cot": 1024} and m["cap_override"]["configured"] == CONF_CAPS
    assert any("CAP-MEASUREMENT RUN" in w and "configured caps" in w for w in m["warnings"])
    assert m["missing_fields"] == []
    # token counts and throughput of the run are the generated ones
    assert m["output_tokens_generated"] == 10 + 20 + 30 + 40 + 100 + 200 + 300 + 1024
    rows = tr.project_rows([m], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    per_arm = 20 * 1319 * 24
    r = {x["K"]: x for x in rows}[12]
    assert r["output_tokens_by_arm"] == {"direct_answer": per_arm * 25.0, "zero_shot_cot": per_arm * 203.0}
    assert r["gpu_hours"] == pytest.approx(per_arm * (25.0 + 203.0) / 1000.0 / 3600.0 * 2)


def test_cap_measurement_projection_is_below_the_projection_that_trusts_the_raised_cap(tmp_path):
    raised = tr.load_throughput_run(fabricate(tmp_path / "a", log_over={"cap_override": None}))   # no override recorded
    fixed = tr.load_throughput_run(fabricate(tmp_path / "b", log_over=_override_log()))
    assert raised["cap_override"] is None and raised["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] == 406.0
    a = tr.project_rows([raised], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    b = tr.project_rows([fixed], models=20, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(y["gpu_hours"] < x["gpu_hours"] and y["output_tokens"] < x["output_tokens"] for x, y in zip(a, b))


def test_override_equal_to_the_configured_cap_changes_nothing_but_is_named(tmp_path):
    log = _override_log(arms={"zero_shot_cot": 256})
    m = tr.load_throughput_run(fabricate(tmp_path / "x", caps={"direct_answer": 128, "zero_shot_cot": 256},
                                         lengths={"direct_answer": [10, 20], "zero_shot_cot": [100, 256]}, log_over=log))
    c = m["per_arm"]["zero_shot_cot"]
    assert c["cap_override_applied"] is False and c["cap"] == c["run_cap"] == 256
    assert c["output_tokens"]["mean"] == 178.0 and m["cap_override"] is not None


def test_report_names_the_override_and_says_it_uses_the_configured_caps(tmp_path):
    over = fabricate(tmp_path / "a", model_key="a", log_over=_override_log())
    plain = fabricate(tmp_path / "b", model_key="b", run_id="r2")
    rep = tr.build_report([over, plain], bank=NO_BANK, caps=NO_CAPS)
    co = rep["cap_override"]
    assert list(co["models"]) == ["a"] and co["models"]["a"]["arms"] == {"zero_shot_cot": 1024}
    assert "configured caps" in co["projection_uses"] and "min(length, configured cap)" in co["projection_uses"]
    assert any(w.startswith("a: CAP-MEASUREMENT RUN") for w in rep["warnings"])
    assert not any(w.startswith("b: CAP-MEASUREMENT RUN") for w in rep["warnings"])
    assert any("Cap-measurement run" in a and "configured output caps" in a for a in rep["assumptions"])
    json.dumps(rep, allow_nan=False)
    md = tr.render_markdown(rep)
    assert "Cap-measurement run (--measure-cap)" in md and "zero_shot_cot 1024" in md and "zero_shot_cot 256" in md
    for banned in ("rather than", "instead of"):
        assert banned not in md
    clean = tr.build_report([plain], bank=NO_BANK, caps=NO_CAPS)
    assert clean["cap_override"] is None and not any("Cap-measurement" in a for a in clean["assumptions"])
    assert "Cap-measurement run" not in tr.render_markdown(clean)


def test_unrecorded_configured_cap_is_reported_and_not_guessed(tmp_path):
    log = {"cap_override": {"arms": {"zero_shot_cot": 1024}}}
    m = tr.load_throughput_run(fabricate(tmp_path / "x", log_over=log))
    assert "cap_override.configured_max_new_tokens.zero_shot_cot" in m["missing_fields"]
    c = m["per_arm"]["zero_shot_cot"]
    assert c["cap"] == 1024 and c["cap_override_applied"] is False and c["output_tokens"]["mean"] == 406.0
    assert any("not recorded" in w and "overstates" in w for w in m["warnings"])
    assert any("CAP-MEASUREMENT RUN" in w for w in m["warnings"])


def test_unreadable_override_warns_that_the_projection_uses_the_runs_own_caps(tmp_path):
    for bad in ("raised", {"arms": {}}, {"configured_max_new_tokens": CONF_CAPS}):
        m = tr.load_throughput_run(fabricate(tmp_path / f"x{len(str(bad))}", log_over={"cap_override": bad}))
        assert "cap_override.arms" in m["missing_fields"]
        assert any("cannot read" in w and "not valid for the registered caps" in w for w in m["warnings"])
        assert m["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] == 406.0
        assert m["cap_override"]["readable"] is False


def test_override_that_lowers_a_cap_is_flagged_as_a_lower_bound(tmp_path):
    log = _override_log(arms={"zero_shot_cot": 300}, configured={"direct_answer": 128, "zero_shot_cot": 1024})
    m = tr.load_throughput_run(fabricate(tmp_path / "x", caps={"direct_answer": 128, "zero_shot_cot": 300},
                                         lengths={"direct_answer": [10, 20], "zero_shot_cot": [100, 300, 300]}, log_over=log))
    c = m["per_arm"]["zero_shot_cot"]
    assert (c["cap"], c["run_cap"], c["cap_override_applied"]) == (1024, 300, False)
    assert c["output_tokens"]["mean"] == pytest.approx(700 / 3)                   # lengths as generated
    assert any("lower bounds" in w and "zero_shot_cot" in w for w in m["warnings"])


def test_real_cap_measurement_run_is_projected_at_the_configured_cap(env):
    out = env.tmp / "capm"
    spec = g.parse_model_spec("m1", g.load_registry(env.models))
    echo = g.EchoBackend(spec, g.load_decoding(env.decoding), delay_s=0.02)
    log = g.run_generation(env.requests("m1"), "m1", out, backend="echo", backend_obj=echo, throughput=True, limit=30,
                           shard_size=1000, models_path=env.models, decoding_path=env.decoding,
                           budget_path=env.budget, verbose=False, cap_override={"direct_answer": 400})
    assert log["cap_override"]["configured_max_new_tokens"]["direct_answer"] == CAPS["direct_answer"] == 60
    tdir = out / "throughput"
    m = tr.load_throughput_run(tdir, allow_non_vllm=True)
    d = m["per_arm"]["direct_answer"]
    uniq = g.read_outputs(tdir)
    uniq = uniq[~uniq["cached"] & (uniq["arm"] == "direct_answer")]
    assert uniq["n_output_tokens"].max() > 60                                      # the raised cap let longer outputs through
    assert (d["cap"], d["run_cap"], d["cap_override_applied"]) == (60, 400, True)
    assert d["output_tokens"]["mean"] == pytest.approx(np.minimum(uniq["n_output_tokens"], 60).mean())
    assert d["output_tokens"]["max"] == 60
    assert d["output_tokens_as_generated"]["mean"] == pytest.approx(uniq["n_output_tokens"].mean())
    assert d["share_truncated"] == pytest.approx((uniq["n_output_tokens"] > 60).mean()) and d["share_truncated"] > 0
    assert m["per_arm"]["zero_shot_cot"]["cap_override_applied"] is False
    assert m["missing_fields"] == [] and any("CAP-MEASUREMENT RUN" in w for w in m["warnings"])


def test_cli_prints_the_cap_measurement_warning_and_writes_the_override_field(tmp_path, capsys):
    cli = _load_cli()
    d = fabricate(tmp_path / "x", log_over=_override_log())
    out = tmp_path / "out"
    budget = tmp_path / "budget.yaml"
    budget.write_text(yaml.safe_dump({"cap": {"money_usd": None, "gpu_hours": None}}))
    assert cli.main(["--run-dir", str(d), "--bank-size", "10", "--budget-yaml", str(budget), "--out-dir", str(out)]) == 0
    rep = json.loads((out / "cost_projection.json").read_text(encoding="utf-8"))
    assert rep["cap_override"]["models"]["fake"]["configured"] == CONF_CAPS
    assert "warning: fake: CAP-MEASUREMENT RUN" in capsys.readouterr().err
    assert "Cap-measurement run" in (out / "cost_projection.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- noise-tranche folders

def _tranche_record(env, keys, *, kind="tranche", split="train"):
    """A tranche build record under <tmp>/reqs/tranche/ listing each key's request file, as build_requests.py does."""
    d = env.tmp / "reqs" / "tranche"
    d.mkdir(parents=True, exist_ok=True)
    models = {}
    for k in keys:
        p = env.requests(k)
        models[k] = {"requests_file": p.name, "requests_sha256": g.sha256_file(p), "n_requests": len(pd.read_parquet(p))}
    (d / "build_record.json").write_text(json.dumps({"kind": kind, "split": split, "excluded_from_main": True,
                                                     "models": models}), encoding="utf-8")
    return env.tmp / "reqs"


def _pooled_rate(folder):
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("run_log-*.json"))]
    return (sum(lg["totals"]["output_tokens_generated"] for lg in logs) / sum(lg["generation_s"] for lg in logs),
            len(logs))


def test_a_noise_tranche_folder_is_measured_from_all_its_shards(env):
    full = env.run("m1", throughput=False)
    recs = tr.load_tranche_records(_tranche_record(env, ["m1"]))
    m = tr.load_tranche_run(full, recs, allow_non_vllm=True)
    out = g.read_outputs(full)
    assert m["source"] == "noise_tranche" and m["model_key"] == "m1" and m["requests"] == len(out) == 30
    assert m["unique_generations"] == int((~out["cached"]).sum()) == 20             # c2 repeats c0's prompts
    rate, n = _pooled_rate(full)
    assert n == 1 and m["output_tokens_per_s"] == pytest.approx(rate)
    assert m["per_arm"]["zero_shot_cot"]["output_tokens"]["mean"] == pytest.approx(
        out[~out["cached"] & (out["arm"] == "zero_shot_cot")]["n_output_tokens"].mean())
    assert m["missing_fields"] == [] and not [w for w in m["warnings"] if "unique generations" in w]


def test_a_resumed_noise_tranche_folder_pools_its_runs(env):
    env.run("m1", throughput=False, limit=8, name="t")
    full = env.run("m1", throughput=False, name="t")
    assert len(list(full.glob("run_log-*.json"))) == 2
    m = tr.load_tranche_run(full, tr.load_tranche_records(_tranche_record(env, ["m1"])), allow_non_vllm=True)
    rate, n = _pooled_rate(full)
    assert n == 2 and m["output_tokens_per_s"] == pytest.approx(rate) and len(m["pooled_runs"]) == 2
    assert m["requests"] == 30 and m["unique_generations"] == 20
    assert m["generation_s"] == pytest.approx(sum(r["generation_s"] for r in m["pooled_runs"]))


@pytest.mark.parametrize("case, text", [
    ("not_in_record", "not in the tranche build record"),
    ("limited_only", "no complete run without --limit"),
    ("throughput", "throughput folder"),
])
def test_folders_that_are_not_complete_noise_tranche_runs_are_refused(env, case, text):
    recs = tr.load_tranche_records(_tranche_record(env, ["m2"] if case == "not_in_record" else ["m1"]))
    if case == "throughput":
        folder = env.run("m1") / "throughput"
    else:
        folder = env.run("m1", throughput=False, limit=8 if case == "limited_only" else None)
    with pytest.raises(tr.TrancheError, match=text):
        tr.load_tranche_run(folder, recs, allow_non_vllm=True)


def test_a_record_of_another_kind_or_split_is_refused(env):
    with pytest.raises(tr.TrancheError, match="not a tranche build record"):
        tr.load_tranche_records(_tranche_record(env, ["m1"], kind="screening"))
    with pytest.raises(tr.TrancheError, match="not a tranche build record"):
        tr.load_tranche_records(_tranche_record(env, ["m1"], split="test"))
    with pytest.raises(tr.TrancheError, match="does not exist"):
        tr.load_tranche_records(env.tmp / "nowhere")


def test_with_the_whole_panel_measured_each_model_is_also_projected_at_its_own_figures(tmp_path):
    a = tr.load_throughput_run(fabricate(tmp_path / "a", model_key="a"))
    b = tr.load_throughput_run(fabricate(tmp_path / "b", model_key="b", tp=1, tps=500.0, run_id="r2",
                                         lengths={"direct_answer": [50, 70], "zero_shot_cot": [300, 500]}))
    full = {r["K"]: r for r in tr.project_rows([a, b], models=2, items=1319, bank=NO_BANK, usd_per_gpu_hour=None,
                                               caps=NO_CAPS)}
    r = full[24]
    per_arm = 2 * 1319 * 48
    ha = per_arm * (25 + 406) / 1000 / 3600 * 2
    hb = per_arm * (60 + 400) / 500 / 3600 * 1
    assert r["gpu_hours"] == pytest.approx(max(ha, hb))
    assert r["gpu_hours_sum_of_own"] == pytest.approx((ha + hb) / 2) and r["gpu_hours_sum_of_own"] <= r["gpu_hours"]
    part = tr.project_rows([a, b], models=3, items=1319, bank=NO_BANK, usd_per_gpu_hour=None, caps=NO_CAPS)
    assert all(row["gpu_hours_sum_of_own"] is None for row in part)


def test_cli_reads_noise_tranche_folders(env, tmp_path):
    cli = _load_cli()
    env.write_budget(500, 0)
    d1, d2 = env.run("m1", throughput=False), env.run("m2", throughput=False)
    reqs = _tranche_record(env, ["m1", "m2"])
    out = tmp_path / "out"
    rc = cli.main(["--tranche-dir", str(d1), "--tranche-dir", str(d2), "--requests-dir", str(reqs), "--models", "2",
                   "--allow-non-vllm", "--bank-size", "56", "--budget-yaml", str(env.budget),
                   "--usd-per-gpu-hour", "0", "--out-dir", str(out)])
    assert rc == 0
    rep = json.loads((out / "cost_projection.json").read_text(encoding="utf-8"))
    assert [m["source"] for m in rep["measured"]["models"]] == ["noise_tranche", "noise_tranche"]
    assert all(r["gpu_hours_sum_of_own"] is not None and r["cost_usd"] == 0 for r in rep["rows"])
    md = (out / "cost_projection.md").read_text(encoding="utf-8")
    assert "Noise-tranche folder" in md and "Every panel model was measured" in md
    assert md == (out / "cost_projection_clean.md").read_text(encoding="utf-8")
    assert cli.main(["--tranche-dir", str(d1), "--requests-dir", str(env.tmp / "nowhere"), "--allow-non-vllm",
                     "--out-dir", str(tmp_path / "out2")]) == 2
    assert not (tmp_path / "out2").exists()