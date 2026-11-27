r"""Generation runner for Dataset 2.

Input is an already-rendered request file (parquet) with columns request_id, model_key, arm,
config_id, item_id, messages_json (OpenAI-style chat messages as JSON). The runner embeds no
prompt content. One model per run.

What a run does
  1. Reads the model registry (config/models.yaml, dataset2.models) and decoding settings
     (config/decoding.yaml) and refuses to start on placeholders, an unfrozen chat template hash,
     unconfirmed decoding (vLLM, non-throughput) or missing budget caps (vLLM, bulk).
  2. Applies the configured thinking-off method identically to every request of both arms.
  3. Renders each prompt with the chat template, hashes it, and generates each distinct
     (model, revision, prompt, decoding params) once. Cache key: sha256 over canonical JSON of
     [hf_id, revision, sha256(rendered prompt text), decoding params of the request's arm].
  4. Writes parquet shards atomically after every chunk. A rerun skips request_ids already present
     and reuses their outputs for identical prompts (requeue-safe).
  5. Writes run_log-<run_id>.json (versions, hardware, pins, counts, tokens, runtime).

Output columns: request_id, model_key, arm, config_id, item_id, prompt_sha256 (sha256 of the
rendered prompt text), output_text, n_prompt_tokens, n_output_tokens, finish_reason (stop or
length), cached (True when no new generation was made for that request).

Greedy decoding with batched vLLM is not guaranteed to be bitwise reproducible across batch
compositions. Requests are processed in a fixed order (config_id, arm, item_id, request_id), so a
rerun submits the same chunks in the same order, but vLLM's continuous batching still forms batches
by timing: with its default kernels, 41-46% of greedy outputs differed between two identical runs on
a GPU machine. Each registry entry therefore names its kernel mode (`kernels`):
batch_invariant, under which the runner sets VLLM_BATCH_INVARIANT=1 and
TORCHINDUCTOR_BATCH_INVARIANT=1 before vLLM loads, or default, for architectures with Mamba or
linear-attention layers, for which vLLM 0.30.0 refuses batch-invariant mode. A switch set in the
environment against the configured mode refuses the run. The switches enter the run fingerprint, so one
output folder never mixes kernel modes.

The throughput mode runs a deterministic arm-stratified subset into <out_dir>/throughput/, ignores
and never touches the main shards, and reports tokens/s and the output-length distribution per arm,
with the share of unique generations longer than each candidate output cap (64 to 4096) that lies
below the run's cap. Its outputs are excluded from the main estimation. A vLLM throughput
run with --limit above PILOT_MAX_REQUESTS is a bulk GPU run: it needs cap.money_usd or cap.gpu_hours in
config/budget.yaml (the Dataset 2 allocation is not required, it is chosen after the tranche).

Cap measurement (throughput mode only): cap_override replaces max_new_tokens for the named arms in
that run only, so the length distribution is not hidden by the configured cap. The override is
recorded in the run log (cap_override, with the configured caps), enters the run fingerprint, and the
log's decoding block shows the caps actually applied. It is refused outside throughput mode.

vLLM preflight (before the model loads): vLLM's FlashInfer sampler JIT-compiles a CUDA kernel at
warm-up, and FlashInfer needs a CUDA toolkit for that. A vllm run is refused when the sampler will be
used and FlashInfer's CUDA lookup (CUDA_HOME or CUDA_PATH, then nvcc on PATH, then /usr/local/cuda)
finds nothing. The result is recorded in the run log as preflight.flashinfer_sampler.

vLLM is imported lazily; the module imports on Windows without vLLM or torch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MODELS_YAML = CONFIG_DIR / "models.yaml"
DECODING_YAML = CONFIG_DIR / "decoding.yaml"
BUDGET_YAML = CONFIG_DIR / "budget.yaml"

REQUEST_COLUMNS = ("request_id", "model_key", "arm", "config_id", "item_id", "messages_json")
# Optional column: per request, the constraint of guided decoding as JSON, {"choice": [...]}
# or {"regex": "..."}. Read only for arms whose decoding sets guided: request; it never enters the messages.
GUIDED_COLUMN = "guided_json"
GUIDED_KINDS = ("choice", "regex")
# Optional column (temperature spot check): per request, the seed of
# its sampled generation (an integer in [0, 2**31 - 1]). Read only when the decoding samples (Decoding.sampling); a
# greedy decoding refuses a file that carries one. It never enters the messages.
SEED_COLUMN = "sampling_seed"
SEED_MAX = 2 ** 31 - 1
OUTPUT_COLUMNS = ("request_id", "model_key", "arm", "config_id", "item_id", "prompt_sha256",
                  "output_text", "n_prompt_tokens", "n_output_tokens", "finish_reason", "cached")
FINISH_REASONS = ("stop", "length")
SCHEMA_VERSION = "1"

DTYPES = ("bfloat16", "float16", "float32")
THINKING_OFF_METHODS = ("chat_template_kwargs", "system_prompt_token", "none_needed")
TEMPLATE_SOURCES = ("tokenizer_default", "file", "renderer")
# Size ladder: optional registry fields, absent from the main-panel entries.
#   pipeline_parallel_size: GPUs in a pipeline (vLLM pipeline_parallel_size); a job holds tensor x pipeline GPUs.
#   renderer: a prompt renderer that is not a Jinja chat template, {kind, file, sha256}. deepseek_v4_encoding imports
#     the release's own encoding/encoding_dsv4.py from the pinned snapshot (chat_template.source: renderer, and
#     chat_template.sha256 is that file's sha256).
#   prefill_mode: continue_final_message (the default; LLM.chat continues the final assistant turn) or suffix (the
#     prompt is rendered with its generation prompt, the prefill text is appended, and the whole is tokenised without
#     added special tokens and passed as token ids, as scripts/rp_feasibility.py did for DeepSeek's suffix_box).
RENDERERS = ("deepseek_v4_encoding",)
PREFILL_MODES = ("continue_final_message", "suffix")
QUANTIZATION_SOURCES = ("argument", "checkpoint")
ENGINE_DEFAULTS: dict[str, Any] = {
    "gpu_memory_utilization": 0.90, "enable_prefix_caching": True, "enforce_eager": False,
    "max_num_seqs": None, "max_num_batched_tokens": None, "generation_config": "auto",
}
# Scheduler limits a registry entry may set for its own model. They bound how many sequences run at once and leave
# every output unchanged under batch-invariant kernels, so, like the global ones, they stay out of the fingerprint.
# Qwen3.6-27B needs one: its Mamba state cache holds 610 sequences on a 96 GB card, below vLLM's default of 1,024.
MODEL_ENGINE_KEYS = ("max_num_seqs", "max_num_batched_tokens")
DEFAULT_SHARD_SIZE = 4096     # distinct generations per chunk; the rest of a chunk is written when it ends
DEFAULT_FLUSH_S = 600.0       # finished outputs inside a chunk are written at least this often (seconds)
PILOT_MAX_REQUESTS = 2000     # a --limit up to this size is a pilot and skips the bulk-run budget gate (throughput included)
CAP_CANDIDATES = (64, 128, 256, 512, 1024, 2048, 4096)   # output caps reported in the throughput exceedance shares
FLASHINFER_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"
DEFAULT_CUDA_HOME = "/usr/local/cuda"                     # flashinfer's last-resort CUDA location

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 75

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
# The only environment variables copied into the run log. Tokens and everything else stay out.
_LOGGED_ENV = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT",
               "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_GPUS_PER_TASK",
               "APPTAINER_NAME", "SINGULARITY_NAME", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
               # vLLM engine switches that change kernels (sampler, attention backend, batch-invariant
               # kernels), not the decoding rule.
               "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND", "VLLM_FLASH_ATTN_VERSION",
               "VLLM_BATCH_INVARIANT", "VLLM_NO_USAGE_STATS",
               # Inductor switches that change compiled kernels, and where Inductor keeps them.
               "TORCHINDUCTOR_DETERMINISTIC", "TORCHINDUCTOR_BATCH_INVARIANT", "TORCHINDUCTOR_CACHE_DIR")
# torch 2.13 (torch/_inductor/config.py) reads each as == "1". DETERMINISTIC skips on-device benchmarking that
# affects numerics; BATCH_INVARIANT keeps a compiled kernel stable per sample across batch sizes and implies it.
# vLLM 0.30.0's batch-invariant mode sets neither, and without them a fresh compilation changed 39-45% of greedy
# outputs (results/batch_invariance/summary_20261029.md).
INDUCTOR_KERNEL_ENV = ("TORCHINDUCTOR_DETERMINISTIC", "TORCHINDUCTOR_BATCH_INVARIANT")
# Each registry entry names its kernel mode, and the runner sets the switches of that
# mode. batch_invariant sets vLLM's and Inductor's switches, with which runs that each compiled from scratch
# regenerated every greedy output exactly across nodes, drivers and batch composition. default sets none; it is for
# architectures with Mamba or linear-attention layers, for which vLLM 0.30.0 refuses batch-invariant mode.
KERNEL_MODES: dict[str, dict[str, str]] = {
    "batch_invariant": {"VLLM_BATCH_INVARIANT": "1", "TORCHINDUCTOR_BATCH_INVARIANT": "1"},
    "default": {},
}
KERNEL_SWITCHES = ("VLLM_BATCH_INVARIANT",) + INDUCTOR_KERNEL_ENV
_LOGGED_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors",
                    "numpy", "pandas", "pyarrow", "PyYAML", "flashinfer-python")


class RunRefused(Exception):
    """A configuration problem or gate stopped the run before any generation."""


class GenerationError(RuntimeError):
    """Generation returned something the runner cannot record faithfully."""


class _StopInChunk(Exception):
    """Raised from the tick callback once a stop signal has arrived and the chunk's finished outputs are written: the
    chunk's generation ends there, and the run ends as interrupted (exit 75)."""


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------

def _is_placeholder(v: Any) -> bool:
    return v is None or (isinstance(v, str) and (not v.strip() or v.strip().upper().startswith("PLACEHOLDER")))


@dataclasses.dataclass(frozen=True)
class ThinkingOff:
    method: str
    chat_template_kwargs: dict
    system_prompt_token: str | None
    rationale: str | None

    def to_log(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChatTemplate:
    source: str
    file: str | None
    sha256: str | None


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model_key: str
    hf_id: str
    revision: str
    dtype: str
    quantization: str | None
    tokenizer_mode: str
    max_model_len: int
    tensor_parallel_size: int
    chat_template: ChatTemplate
    thinking_off: ThinkingOff
    kernels: str
    family: str | None
    engine: dict = dataclasses.field(default_factory=dict)   # per-model MODEL_ENGINE_KEYS, over decoding.engine
    # Size-ladder fields (RENDERERS, PREFILL_MODES above). At their defaults they are left out of to_log, so the run
    # logs and fingerprints of every earlier model stay as they were.
    pipeline_parallel_size: int = 1
    renderer: dict = dataclasses.field(default_factory=dict)
    prefill_mode: str = "continue_final_message"
    # The model's own share of card memory in place of decoding.engine's (size ladder: 236 GB of weights on 3 x 96 GB
    # cards leaves little KV cache at 0.90). Like the scheduler limits it changes no output and stays out of the
    # fingerprint.
    gpu_memory_utilization: float | None = None
    # argument (the default): `quantization` is passed to vLLM. checkpoint: nothing is passed, vLLM reads the method from
    # the checkpoint's own config (as scripts/rp_feasibility.py loaded DeepSeek-V4 and gpt-oss), and the method vLLM
    # reports is recorded in the run log beside the configured one, with a warning when they differ.
    quantization_source: str = "argument"
    # Size ladder: the model_key of the request file this entry answers when it is not its own (a -dk fallback key runs
    # its base key's request file unchanged, so both keys' runs read one file with one sha256). The rows it writes carry
    # this entry's own key. None for every main-panel entry.
    requests_model_key: str | None = None
    # Size ladder: a natively multimodal checkpoint run on text alone (Qwen3.5-122B-A10B):
    # vLLM gets limit_mm_per_prompt 0 for images and videos, so no vision input is profiled or accepted. False for
    # every main-panel entry.
    text_only: bool = False

    @property
    def gpus(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def text_prompts(self) -> bool:
        """True when prompts reach the engine as token ids of the runner's own rendering (a renderer, or suffix
        prefills); False for LLM.chat."""
        return bool(self.renderer) or self.prefill_mode == "suffix"

    def to_log(self) -> dict:
        d = dataclasses.asdict(self)
        if d["pipeline_parallel_size"] == 1:
            d.pop("pipeline_parallel_size")
        if not d["renderer"]:
            d.pop("renderer")
        if d["prefill_mode"] == "continue_final_message":
            d.pop("prefill_mode")
        if d["gpu_memory_utilization"] is None:
            d.pop("gpu_memory_utilization")
        if d["requests_model_key"] is None:
            d.pop("requests_model_key")
        if not d["text_only"]:
            d.pop("text_only")
        if d["quantization_source"] == "argument":
            d.pop("quantization_source")
        return d


def load_registry(path: str | os.PathLike | None = None) -> dict[str, dict]:
    p = Path(path) if path else MODELS_YAML
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    models = (raw.get("dataset2") or {}).get("models") or {}
    if not isinstance(models, dict):
        raise RunRefused(f"{p}: dataset2.models must be a mapping of model_key to entry")
    return models


def parse_model_spec(model_key: str, registry: Mapping[str, dict]) -> ModelSpec:
    """Validate one registry entry. Every problem is reported at once; any problem refuses the run."""
    if model_key.startswith("_"):
        raise RunRefused(f"model_key {model_key!r} is a schema template, not a model")
    if model_key not in registry:
        known = sorted(k for k in registry if not k.startswith("_")) or ["(none registered)"]
        raise RunRefused(f"model_key {model_key!r} is not in the registry; known: {', '.join(known)}")
    e = registry[model_key] or {}
    problems: list[str] = []

    def req_str(key: str) -> str:
        v = e.get(key)
        if _is_placeholder(v) or not isinstance(v, str):
            problems.append(f"{key} is missing or a placeholder")
            return ""
        return v

    hf_id = req_str("hf_id")
    revision = req_str("revision")
    if revision and not _SHA40.match(revision):
        problems.append(f"revision {revision!r} is not an exact 40-hex commit sha")
    dtype = req_str("dtype")
    if dtype and dtype not in DTYPES:
        problems.append(f"dtype {dtype!r} not in {DTYPES}")
    quant_raw = e.get("quantization")
    if _is_placeholder(quant_raw) or not isinstance(quant_raw, str):
        problems.append("quantization is missing or a placeholder (write 'none' when unquantized)")
        quant_raw = "none"
    tokenizer_mode = e.get("tokenizer_mode", "auto")
    if not isinstance(tokenizer_mode, str) or _is_placeholder(tokenizer_mode):
        problems.append("tokenizer_mode must be a string such as 'auto'")
        tokenizer_mode = "auto"
    for key in ("max_model_len", "tensor_parallel_size"):
        v = e.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            problems.append(f"{key} must be a positive integer")
    ct = e.get("chat_template") or {}
    src = ct.get("source")
    if src not in TEMPLATE_SOURCES:
        problems.append(f"chat_template.source must be one of {TEMPLATE_SOURCES}")
    if src == "file" and _is_placeholder(ct.get("file")):
        problems.append("chat_template.file is required when source is file")
    t_sha = ct.get("sha256")
    if t_sha is not None and not (isinstance(t_sha, str) and _SHA64.match(t_sha)):
        problems.append("chat_template.sha256 must be null or a 64-hex sha256")
    to = e.get("thinking_off") or {}
    method = to.get("method")
    kwargs = to.get("chat_template_kwargs") or {}
    token = to.get("system_prompt_token")
    rationale = to.get("rationale")
    if method not in THINKING_OFF_METHODS:
        problems.append(f"thinking_off.method must be one of {THINKING_OFF_METHODS}")
    elif method == "chat_template_kwargs":
        if not isinstance(kwargs, dict) or not kwargs:
            problems.append("thinking_off.chat_template_kwargs must be a non-empty mapping")
        else:
            try:
                json.dumps(kwargs)
            except TypeError:
                problems.append("thinking_off.chat_template_kwargs must be JSON-serializable")
    elif method == "system_prompt_token":
        if _is_placeholder(token) or not isinstance(token, str):
            problems.append("thinking_off.system_prompt_token must be a non-empty string")
    elif method == "none_needed":
        if _is_placeholder(rationale) or not isinstance(rationale, str):
            problems.append("thinking_off.rationale is required when method is none_needed")
    kernels = e.get("kernels")
    if kernels not in KERNEL_MODES:
        problems.append(f"kernels must be one of {tuple(KERNEL_MODES)} (batch_invariant unless vLLM 0.30.0 "
                        "refuses batch-invariant mode for the architecture)")
    engine = e.get("engine") or {}
    if not isinstance(engine, dict):
        problems.append("engine must be a mapping")
        engine = {}
    for k, v in engine.items():
        if k not in MODEL_ENGINE_KEYS:
            problems.append(f"engine.{k} is not a per-model setting; allowed: {MODEL_ENGINE_KEYS}")
        elif isinstance(v, bool) or not isinstance(v, int) or v < 1:
            problems.append(f"engine.{k} must be a positive integer")
    gmu = e.get("gpu_memory_utilization")
    if gmu is not None and (isinstance(gmu, bool) or not isinstance(gmu, (int, float)) or not 0 < gmu <= 0.98):
        problems.append("gpu_memory_utilization must be a number above 0 and at most 0.98")
        gmu = None
    qsrc = e.get("quantization_source", "argument")
    if qsrc not in QUANTIZATION_SOURCES:
        problems.append(f"quantization_source must be one of {QUANTIZATION_SOURCES}")
        qsrc = "argument"
    pp = e.get("pipeline_parallel_size", 1)
    if isinstance(pp, bool) or not isinstance(pp, int) or pp < 1:
        problems.append("pipeline_parallel_size must be a positive integer")
        pp = 1
    renderer = e.get("renderer") or {}
    if not isinstance(renderer, dict):
        problems.append("renderer must be a mapping {kind, file, sha256}")
        renderer = {}
    elif renderer:
        if renderer.get("kind") not in RENDERERS:
            problems.append(f"renderer.kind must be one of {RENDERERS}")
        if _is_placeholder(renderer.get("file")) or not isinstance(renderer.get("file"), str):
            problems.append("renderer.file is required: the renderer's path inside the pinned snapshot")
        if not (isinstance(renderer.get("sha256"), str) and _SHA64.match(renderer["sha256"])):
            problems.append("renderer.sha256 must be the 64-hex sha256 of the renderer file")
        elif t_sha != renderer["sha256"]:
            problems.append("chat_template.sha256 must equal renderer.sha256 when a renderer renders the prompts")
        extra = sorted(set(renderer) - {"kind", "file", "sha256"})
        if extra:
            problems.append(f"renderer has unknown keys {extra}")
    if (src == "renderer") != bool(renderer):
        problems.append("chat_template.source is renderer exactly when a renderer is registered")
    prefill_mode = e.get("prefill_mode", "continue_final_message")
    if prefill_mode not in PREFILL_MODES:
        problems.append(f"prefill_mode must be one of {PREFILL_MODES}")
        prefill_mode = "continue_final_message"
    elif renderer and prefill_mode != "suffix":
        problems.append("a renderer has no final-message continuation: prefill_mode must be suffix")
    text_only = e.get("text_only", False)
    if not isinstance(text_only, bool):
        problems.append("text_only must be true or false")
    rmk = e.get("requests_model_key")
    if rmk is not None and (not isinstance(rmk, str) or not rmk or rmk == model_key or rmk not in registry):
        problems.append("requests_model_key must name another key of the same registry")
    if problems:
        raise RunRefused(f"model {model_key!r} cannot be run: " + "; ".join(problems))
    return ModelSpec(
        model_key=model_key, hf_id=hf_id, revision=revision, dtype=dtype,
        quantization=None if quant_raw.lower() == "none" else quant_raw,
        tokenizer_mode=tokenizer_mode, max_model_len=int(e["max_model_len"]),
        tensor_parallel_size=int(e["tensor_parallel_size"]),
        chat_template=ChatTemplate(source=src, file=ct.get("file"), sha256=t_sha),
        thinking_off=ThinkingOff(
            method=method, chat_template_kwargs=dict(kwargs) if method == "chat_template_kwargs" else {},
            system_prompt_token=token if method == "system_prompt_token" else None,
            rationale=rationale if method == "none_needed" else None),
        kernels=kernels, family=None if _is_placeholder(e.get("family")) else str(e["family"]), engine=dict(engine),
        pipeline_parallel_size=int(pp), renderer=dict(renderer), prefill_mode=str(prefill_mode),
        gpu_memory_utilization=None if gmu is None else float(gmu), quantization_source=str(qsrc),
        requests_model_key=rmk, text_only=bool(text_only))


def model_gpu_count(model_key: str, path: str | os.PathLike | None = None) -> int:
    """GPUs one job of the model holds: tensor_parallel_size x pipeline_parallel_size (the latter 1 when absent).
    Used to size the GPU allocation of a job; --print-tp prints the tensor-parallel size alone."""
    tp = model_tensor_parallel_size(model_key, path)
    pp = (load_registry(path)[model_key] or {}).get("pipeline_parallel_size", 1)
    if isinstance(pp, bool) or not isinstance(pp, int) or pp < 1:
        raise RunRefused(f"model {model_key!r} has no valid pipeline_parallel_size")
    return tp * pp


def model_tensor_parallel_size(model_key: str, path: str | os.PathLike | None = None) -> int:
    """GPUs per job. Reads the registry without the placeholder check."""
    reg = load_registry(path)
    if model_key.startswith("_") or model_key not in reg:
        raise RunRefused(f"model_key {model_key!r} is not a registered model")
    v = (reg[model_key] or {}).get("tensor_parallel_size")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise RunRefused(f"model {model_key!r} has no valid tensor_parallel_size")
    return v


@dataclasses.dataclass(frozen=True)
class Decoding:
    status: str
    temperature: float
    top_p: float
    top_k: int
    seed: int
    max_new_tokens: dict[str, int]
    stop: tuple[str, ...]
    include_stop_str_in_output: bool
    skip_special_tokens: bool
    engine: dict
    # {arm: text}: the assistant turn is opened with this text and the model continues it (forced-direct arm,
    # Empty for every arm of the collection's decoding.
    prefill: dict = dataclasses.field(default_factory=dict)
    # {arm: "request"}: the arm's outputs are constrained by guided decoding (vLLM structured outputs), with each
    # request's own constraint from the request file's GUIDED_COLUMN. Empty for every other decoding.
    guided: dict = dataclasses.field(default_factory=dict)
    # {arm: {stop: [...], include_stop_str_in_output: bool}}: the arm's own stop rule in place of the global one, so
    # that one run can hold arms whose collections were decoded under different files (size ladder: GSM8K's
    # requested and chain-of-thought arms under config/decoding.yaml, its forced arm under decoding_forced.yaml).
    # params_for_arm then equals that of the arm's own file. Empty for every earlier decoding.
    arm_overrides: dict = dataclasses.field(default_factory=dict)
    # {arm: N} (size ladder): the arm runs at its raised max_new_tokens, and every output is
    # also kept cut at the token level to the panel's cap N: the first N generated tokens, decoded as the engine
    # decodes them, marked cut when the uncut output ran past N (output_schema's cut columns). Not part of
    # params_for_arm (the generation itself is the raised-cap one) but part of the run fingerprint through to_log.
    # Empty for every earlier decoding.
    cut_caps: dict = dataclasses.field(default_factory=dict)
    # {"seed": "request"} (temperature spot check): the decoding samples at its
    # temperature (> 0), each request with its own seed from the request file's SEED_COLUMN, which joins the request's
    # params (so its cache key) as "sampling_seed". Empty for every greedy decoding, whose params are unchanged.
    sampling: dict = dataclasses.field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.max_new_tokens))

    def params_for_arm(self, arm: str) -> dict:
        """The decoding parameters that determine the output of one request. Part of the cache key. An arm without
        a prefill gives the same dict as before prefills existed, so earlier cache keys and fingerprints stand."""
        ov = self.arm_overrides.get(arm) or {}
        p = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
             "seed": self.seed, "max_new_tokens": self.max_new_tokens[arm],
             "stop": list(ov.get("stop", self.stop)),
             "include_stop_str_in_output": bool(ov.get("include_stop_str_in_output", self.include_stop_str_in_output)),
             "skip_special_tokens": self.skip_special_tokens}
        if self.prefill.get(arm):
            p["prefill"] = self.prefill[arm]
        if self.guided.get(arm):   # the marker; _execute adds each request's own constraint as p["guided"]
            p["guided_source"] = self.guided[arm]
        if self.sampling:   # the marker; _execute adds each request's own seed as p["sampling_seed"]
            p["sampling_seed_source"] = self.sampling["seed"]
        return p

    def to_log(self) -> dict:
        d = dataclasses.asdict(self)
        if not self.guided:   # the log of a decoding without guided arms reads as before the field existed
            d.pop("guided")
        if not self.arm_overrides:
            d.pop("arm_overrides")
        if not self.cut_caps:
            d.pop("cut_caps")
        if not self.sampling:
            d.pop("sampling")
        d["stop"] = list(self.stop)
        d["confirmed"] = self.confirmed
        return d


def load_decoding(path: str | os.PathLike | None = None) -> Decoding:
    p = Path(path) if path else DECODING_YAML
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    problems: list[str] = []
    sampling = raw.get("sampling") or {}
    temp = raw.get("temperature")
    if sampling:
        # A sampled decoding: each request carries its own seed (SEED_COLUMN), which enters its cache key, so two
        # requests share an output only when prompt, parameters and seed are all equal.
        if sampling != {"seed": "request"}:
            problems.append("sampling must be {seed: request} (each request carries its own seed in the request file)")
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0 < float(temp) <= 2:
            problems.append("a sampled decoding needs a temperature in (0, 2]")
    elif temp != 0:
        problems.append("temperature must be 0 (greedy); caching identical inputs is valid only for deterministic decoding")
    mnt = raw.get("max_new_tokens")
    if not isinstance(mnt, dict) or not mnt:
        problems.append("max_new_tokens must map each arm label to a positive integer")
        mnt = {}
    for arm, v in mnt.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            problems.append(f"max_new_tokens[{arm}] must be a positive integer")
    stop = raw.get("stop") or []
    if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop):
        problems.append("stop must be a list of non-empty strings")
        stop = []
    engine = dict(ENGINE_DEFAULTS)
    unknown = sorted(set(raw.get("engine") or {}) - set(ENGINE_DEFAULTS))
    if unknown:
        problems.append(f"unknown engine settings: {unknown}")
    engine.update({k: v for k, v in (raw.get("engine") or {}).items() if k in ENGINE_DEFAULTS})
    for key, typ in (("top_p", (int, float)), ("top_k", int), ("seed", int)):
        if isinstance(raw.get(key), bool) or not isinstance(raw.get(key), typ):
            problems.append(f"{key} is missing or has the wrong type")
    prefill = raw.get("prefill") or {}
    if not isinstance(prefill, dict) or not all(isinstance(k, str) and isinstance(v, str) and v
                                                for k, v in prefill.items()):
        problems.append("prefill must map arm labels to non-empty strings")
        prefill = {}
    elif sorted(set(prefill) - set(mnt)):
        problems.append(f"prefill names arms without max_new_tokens: {sorted(set(prefill) - set(mnt))}")
    guided = raw.get("guided") or {}
    if not isinstance(guided, dict) or not all(isinstance(k, str) and v == "request" for k, v in guided.items()):
        problems.append("guided must map arm labels to 'request' (each request carries its own constraint)")
        guided = {}
    elif sorted(set(guided) - set(mnt)):
        problems.append(f"guided names arms without max_new_tokens: {sorted(set(guided) - set(mnt))}")
    overrides = raw.get("arm_overrides") or {}
    if not isinstance(overrides, dict):
        problems.append("arm_overrides must map arm labels to {stop, include_stop_str_in_output}")
        overrides = {}
    for arm, ov in overrides.items():
        if arm not in mnt:
            problems.append(f"arm_overrides names arm {arm!r} without max_new_tokens")
        elif not isinstance(ov, dict) or not ov or sorted(set(ov) - {"stop", "include_stop_str_in_output"}):
            problems.append(f"arm_overrides[{arm}] may set only stop and include_stop_str_in_output")
        else:
            if "stop" in ov and not (isinstance(ov["stop"], list) and all(isinstance(x, str) and x for x in ov["stop"])):
                problems.append(f"arm_overrides[{arm}].stop must be a list of non-empty strings")
            if "include_stop_str_in_output" in ov and not isinstance(ov["include_stop_str_in_output"], bool):
                problems.append(f"arm_overrides[{arm}].include_stop_str_in_output must be true or false")
    cut_caps = raw.get("cut_caps") or {}
    if not isinstance(cut_caps, dict):
        problems.append("cut_caps must map arm labels to the panel's cap (a positive integer)")
        cut_caps = {}
    for arm, n in cut_caps.items():
        if arm not in mnt:
            problems.append(f"cut_caps names arm {arm!r} without max_new_tokens")
        elif isinstance(n, bool) or not isinstance(n, int) or n < 1 or n >= mnt[arm]:
            problems.append(f"cut_caps[{arm}] must be a positive integer below the arm's max_new_tokens")
        else:
            ov = overrides.get(arm) or {}
            if ov.get("stop", stop) or guided.get(arm):
                problems.append(f"cut_caps[{arm}]: a token-level cut equals a run at the lower cap only for an arm "
                                "without stop strings or guided decoding")
    if problems:
        raise RunRefused(f"{p}: " + "; ".join(problems))
    return Decoding(
        status=str(raw.get("status", "PENDING_REVIEW")), temperature=float(temp) if sampling else 0.0,
        top_p=float(raw["top_p"]),
        top_k=int(raw["top_k"]), seed=int(raw["seed"]), max_new_tokens={str(k): int(v) for k, v in mnt.items()},
        stop=tuple(stop), include_stop_str_in_output=bool(raw.get("include_stop_str_in_output", False)),
        skip_special_tokens=bool(raw.get("skip_special_tokens", True)), engine=engine,
        prefill={str(k): str(v) for k, v in prefill.items()}, guided={str(k): str(v) for k, v in guided.items()},
        arm_overrides={str(k): dict(v) for k, v in overrides.items()},
        cut_caps={str(k): int(v) for k, v in cut_caps.items()},
        sampling={str(k): str(v) for k, v in sampling.items()})


def _read_budget(path: str | os.PathLike | None) -> tuple[dict, list[str]]:
    p = Path(path) if path else BUDGET_YAML
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}, []
    except OSError as exc:
        return {}, [f"cannot read {p}: {exc}"]


def _cap_problems(raw: Mapping) -> list[str]:
    cap = raw.get("cap") or {}
    if cap.get("money_usd") is None and cap.get("gpu_hours") is None:
        return ["cap.money_usd and cap.gpu_hours are both unset"]
    return []


def budget_cap_problems(path: str | os.PathLike | None = None) -> list[str]:
    """Cap only: a bulk GPU run (also a bulk throughput run) needs a cap in config/budget.yaml.
    The Dataset 2 allocation is not checked here: it is chosen after the tranche."""
    raw, err = _read_budget(path)
    return err or _cap_problems(raw)


def calibration_build(requests_path: str | os.PathLike, model_key: str) -> dict | None:
    """The build record beside a request file (scripts/build_requests.py) when it marks the file as screening or
    tranche requests on the training split, with this model's entry naming the file and its sha256. Such runs come
    before the Dataset 2 allocation exists, since the tranche is what chooses it, so their budget gate needs the cap
    only. Any other file, or a record that does not match the file byte for byte, returns None and the full gate
    applies."""
    p = Path(requests_path)
    rec_path = p.parent / "build_record.json"
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        entry = (rec.get("models") or {}).get(model_key) or {}
        ok = (rec.get("kind") in ("screening", "tranche") and rec.get("split") == "train"
              and rec.get("excluded_from_main") is True and entry.get("requests_file") == p.name
              and entry.get("requests_sha256") == sha256_file(p))
    except (OSError, ValueError, AttributeError):
        return None
    return {"build_record": str(rec_path), "kind": rec["kind"], "purpose": rec.get("purpose")} if ok else None


def budget_gate_problems(path: str | os.PathLike | None = None) -> list[str]:
    """A bulk GPU run needs a cap and an allocation recorded in config/budget.yaml."""
    raw, err = _read_budget(path)
    if err:
        return err
    alloc = raw.get("dataset2_allocation") or {}
    out = _cap_problems(raw)
    for key in ("discovery_units_per_arm", "validation_units_per_arm"):
        if alloc.get(key) is None:
            out.append(f"dataset2_allocation.{key} is unset")
    return out


def _which(name: str) -> str | None:
    return shutil.which(name)


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


def _flashinfer_importable() -> bool:
    """True when a `flashinfer` package is installed. find_spec does not import it."""
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ValueError):
        return False


def flashinfer_sampler_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """vLLM's own reading of VLLM_USE_FLASHINFER_SAMPLER (vllm/envs.py, checked against 0.30.0): unset
    means on, otherwise bool(int(value)). A value that int() rejects (for example "false") makes vLLM
    raise at start-up, so it is refused here before the model loads."""
    raw = (os.environ if environ is None else environ).get(FLASHINFER_SAMPLER_ENV)
    if raw is None:
        return True
    try:
        return bool(int(raw))
    except ValueError:
        raise RunRefused(f"{FLASHINFER_SAMPLER_ENV}={raw!r} is not an integer; vLLM parses it with int() and would "
                         "fail at start-up. Use 0 (PyTorch sampler) or 1 (FlashInfer sampler), or unset it.") from None


def discover_cuda_home(environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """FlashInfer's CUDA lookup (flashinfer.jit.cpp_ext.get_cuda_path), replicated without importing it:
    CUDA_HOME or CUDA_PATH if set (an empty value counts as unset), else the directory above the
    directory of `nvcc` on PATH, else /usr/local/cuda when it exists. Returns (path, source) or
    (None, None). Like FlashInfer, it does not check that a set CUDA_HOME holds a working toolkit."""
    env = os.environ if environ is None else environ
    for name in ("CUDA_HOME", "CUDA_PATH"):
        if env.get(name):
            return env[name], name
    nvcc = _which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc)), "nvcc_on_path"
    if _path_exists(DEFAULT_CUDA_HOME):
        return DEFAULT_CUDA_HOME, "default_usr_local_cuda"
    return None, None


def check_flashinfer_sampler(environ: Mapping[str, str] | None = None) -> dict:
    """Will vLLM use its FlashInfer sampler (JIT-compiled at warm-up), and can FlashInfer find CUDA?
    The sampler counts as used when the variable is unset or truthy and flashinfer is importable. vLLM
    may still fall back on an unsupported GPU; that cannot be known before the engine starts, so the
    check is conservative."""
    env = os.environ if environ is None else environ
    enabled = flashinfer_sampler_enabled(env)
    importable = _flashinfer_importable()
    cuda_home, source = discover_cuda_home(env)
    return {"checked": True, "env_value": env.get(FLASHINFER_SAMPLER_ENV), "env_enabled": enabled,
            "flashinfer_importable": importable, "used": bool(enabled and importable),
            "cuda_home_found": cuda_home is not None, "cuda_home": cuda_home, "cuda_home_source": source}


def _flashinfer_problem(rec: Mapping) -> str:
    return (f"the FlashInfer sampler will be used ({FLASHINFER_SAMPLER_ENV} is "
            f"{'unset' if rec['env_value'] is None else repr(rec['env_value'])} and flashinfer is installed) and it "
            "JIT-compiles a CUDA kernel during engine warm-up, but FlashInfer finds no CUDA toolkit "
            "(CUDA_HOME and CUDA_PATH are unset, nvcc is not on PATH, /usr/local/cuda does not exist), so the run "
            "would crash after the model loads. Fix one of: (1) run in the container, which has nvcc "
            "(container/README.md); (2) set CUDA_HOME to a CUDA toolkit directory that contains bin/nvcc; "
            f"(3) set {FLASHINFER_SAMPLER_ENV}=0 to use vLLM's PyTorch sampler. Option 3 leaves greedy decoding "
            "unaffected (the argmax is the same) but kernels and speed differ; the variable is recorded in the "
            "run log (scheduler and preflight.flashinfer_sampler)")


def preflight(spec: ModelSpec, decoding: Decoding, *, backend: str, throughput: bool, limit: int | None,
              budget_path: str | os.PathLike | None = None, calibration: Mapping | None = None) -> dict:
    """Gates for a real (vLLM) run, before the model loads. The FlashInfer/CUDA check applies to every vLLM
    run including the throughput tranche. The decoding and chat-template gates are skipped for the tranche.
    The budget gate applies to every run whose --limit is absent or above PILOT_MAX_REQUESTS:
    a full run needs the cap and the Dataset 2 allocation, a bulk throughput run and a full run on screening or
    tranche requests (`calibration`, from calibration_build) need the cap only (the allocation is chosen after the
    tranche). Pilots are exempt. Nothing runs for the echo backend. Returns the record stored in the run log."""
    if backend != "vllm":
        return {"flashinfer_sampler": {"checked": False, "reason": "backend is not vllm"}}
    problems = []
    fi = check_flashinfer_sampler()
    if fi["used"] and not fi["cuda_home_found"]:
        problems.append(_flashinfer_problem(fi))
    if not throughput:
        if not decoding.confirmed:
            problems.append(f"decoding status is {decoding.status!r}; config/decoding.yaml must be confirmed (status: confirmed)")
        if spec.chat_template.sha256 is None:
            problems.append("chat_template.sha256 is not frozen in the registry; take it from the throughput run log")
        if limit is None or limit > PILOT_MAX_REQUESTS:
            bp = budget_cap_problems(budget_path) if calibration else budget_gate_problems(budget_path)
            if bp:
                problems.append("budget gate (approval needed first"
                                + (f"; {calibration['kind']} requests need the cap only" if calibration else "")
                                + "): " + "; ".join(bp))
    elif limit is None or limit > PILOT_MAX_REQUESTS:
        bp = budget_cap_problems(budget_path)
        if bp:
            problems.append(f"budget gate for a bulk throughput run of {'an unlimited' if limit is None else limit} "
                            "requests (approval needed first; the cap is required, the allocation is not): "
                            + "; ".join(bp))
    if problems:
        raise RunRefused("; ".join(problems))
    return {"flashinfer_sampler": fi, "calibration_build": dict(calibration) if calibration else None}


def parse_cap_overrides(items: Sequence[str] | None) -> dict[str, int]:
    """Parse repeated ARM=N strings (--measure-cap) into {arm: N}."""
    out: dict[str, int] = {}
    for item in items or ():
        arm, sep, num = str(item).partition("=")
        arm, num = arm.strip(), num.strip()
        if not sep or not arm or not re.fullmatch(r"[0-9]+", num) or int(num) < 1:
            raise RunRefused(f"--measure-cap {item!r} must look like ARM=N with a positive integer N")
        if arm in out:
            raise RunRefused(f"--measure-cap names arm {arm!r} more than once")
        out[arm] = int(num)
    return out


def validate_cap_override(override: Mapping[str, int] | None, decoding: Decoding, spec: ModelSpec) -> dict[str, int]:
    """Check an output-cap override against the decoding arms and the registry's max_model_len. The
    prompt-length part of that check (longest prompt plus N) runs after the tokenizer is available."""
    out: dict[str, int] = {}
    for arm, n in (override or {}).items():
        if arm not in decoding.arms:
            raise RunRefused(f"--measure-cap: arm {arm!r} has no max_new_tokens in the decoding config {list(decoding.arms)}")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise RunRefused(f"--measure-cap {arm}={n!r}: the cap must be a positive integer")
        if n >= spec.max_model_len:
            raise RunRefused(f"--measure-cap {arm}={n}: max_model_len for {spec.model_key!r} is {spec.max_model_len} in the "
                             "registry, so no prompt fits with that many new tokens; lower the cap or raise max_model_len")
        out[str(arm)] = int(n)
    return out


def apply_cap_override(decoding: Decoding, override: Mapping[str, int]) -> Decoding:
    """The decoding in force for one run: configured caps, with the override replacing the named arms."""
    if not override:
        return decoding
    return dataclasses.replace(decoding, max_new_tokens={**decoding.max_new_tokens, **dict(override)})


# --------------------------------------------------------------------------------------------
# Requests, prompts, keys
# --------------------------------------------------------------------------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(hf_id: str, revision: str, prompt_sha: str, params: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json([hf_id, revision, prompt_sha, dict(params)]))


def parse_messages(s: str) -> list[dict]:
    msgs = json.loads(s)
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be a non-empty list")
    for m in msgs:
        if not (isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                and isinstance(m.get("content"), str)):
            raise ValueError("each message needs a role (system/user/assistant) and string content")
    if msgs[-1]["role"] != "user":
        raise ValueError("the last message must have role user")
    return msgs


def read_requests(path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in REQUEST_COLUMNS if c not in df.columns]
    if missing:
        raise RunRefused(f"request file lacks columns: {missing}")
    extra = [c for c in (GUIDED_COLUMN, SEED_COLUMN) if c in df.columns]
    df = df[list(REQUEST_COLUMNS) + extra].copy()
    if df[list(REQUEST_COLUMNS)].isna().any().any():
        raise RunRefused("request file has null values in required columns")
    for c in ("request_id", "model_key", "arm", "config_id"):
        df[c] = df[c].astype(str)
    return df


def parse_guided(s: Any) -> dict:
    """A request's guided-decoding constraint (GUIDED_COLUMN): {"choice": [two or more distinct non-empty strings]} or
    {"regex": non-empty string}."""
    g = json.loads(s) if isinstance(s, str) else s
    if not isinstance(g, dict) or len(g) != 1 or next(iter(g)) not in GUIDED_KINDS:
        raise ValueError(f"a guided constraint is one of {{'choice': [...]}} or {{'regex': '...'}}, not {s!r}")
    kind, v = next(iter(g.items()))
    if kind == "choice":
        if not (isinstance(v, list) and len(v) >= 2 and all(isinstance(c, str) and c for c in v)
                and len(set(v)) == len(v)):
            raise ValueError("choice must list two or more distinct non-empty strings")
        return {"choice": list(v)}
    if not (isinstance(v, str) and v):
        raise ValueError("regex must be a non-empty string")
    re.compile(v)
    return {"regex": v}


def validate_requests(df: pd.DataFrame, model_key: str, decoding: Decoding, requests_model_key: str | None = None) -> None:
    if df["request_id"].duplicated().any():
        raise RunRefused("request_id is not unique")
    if requests_model_key is not None and set(df["model_key"]) == {requests_model_key}:
        model_key = requests_model_key   # a fallback key answering its base key's file (ModelSpec.requests_model_key)
    other = sorted(set(df["model_key"]) - {model_key})
    if other:
        raise RunRefused(f"one model per run: the file also contains model_key {other[:5]}")
    bad_arms = sorted(set(df["arm"]) - set(decoding.arms))
    if bad_arms:
        raise RunRefused(f"arms {bad_arms} have no max_new_tokens in the decoding config {list(decoding.arms)}")
    for rid, s in zip(df["request_id"], df["messages_json"]):
        try:
            parse_messages(s)
        except (ValueError, TypeError) as exc:
            raise RunRefused(f"request {rid}: invalid messages_json ({exc})") from exc
    guided_arms = set(decoding.guided)
    col = df[GUIDED_COLUMN] if GUIDED_COLUMN in df.columns else pd.Series([None] * len(df), index=df.index)
    for rid, arm, g in zip(df["request_id"], df["arm"], col):
        if arm in guided_arms:
            if g is None or (isinstance(g, float) and np.isnan(g)):
                raise RunRefused(f"request {rid}: arm {arm} is guided, and the request has no {GUIDED_COLUMN}")
            try:
                parse_guided(g)
            except (ValueError, TypeError, re.error) as exc:
                raise RunRefused(f"request {rid}: invalid {GUIDED_COLUMN} ({exc})") from exc
        elif g is not None and not (isinstance(g, float) and np.isnan(g)):
            raise RunRefused(f"request {rid}: it carries a {GUIDED_COLUMN}, and the decoding does not guide arm {arm}")
    seeds = df[SEED_COLUMN] if SEED_COLUMN in df.columns else None
    if decoding.sampling:
        if seeds is None:
            raise RunRefused(f"the decoding samples, and the request file has no {SEED_COLUMN} column")
        for rid, s in zip(df["request_id"], seeds):
            if not _is_seed(s):
                raise RunRefused(f"request {rid}: {SEED_COLUMN} must be an integer in [0, {SEED_MAX}], not {s!r}")
    elif seeds is not None and seeds.notna().any():
        raise RunRefused(f"the request file carries a {SEED_COLUMN}, and the decoding is greedy")


def _is_seed(s: Any) -> bool:
    if isinstance(s, (bool, np.bool_)) or s is None:
        return False
    if isinstance(s, (float, np.floating)):
        if not np.isfinite(s) or float(s) != int(s):
            return False
    elif not isinstance(s, (int, np.integer)):
        return False
    return 0 <= int(s) <= SEED_MAX


def order_requests(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["config_id", "arm", "item_id", "request_id"], kind="stable").reset_index(drop=True)


def select_throughput_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Arm-stratified random subset chosen from request keys only (never from outcomes)."""
    arms = sorted(df["arm"].unique())
    base, extra = divmod(n, len(arms))
    rng = np.random.default_rng(seed)
    parts = []
    for i, arm in enumerate(arms):
        pool = df[df["arm"] == arm]
        k = min(len(pool), base + (1 if i < extra else 0))
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), size=k, replace=False))])
    return order_requests(pd.concat(parts))


def parse_part(text: str | None) -> tuple[int, int] | None:
    """'K/N' -> (K, N) with 1 <= K <= N and N >= 2."""
    if text is None:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(text))
    if not m or not (1 <= int(m.group(1)) <= int(m.group(2))) or int(m.group(2)) < 2:
        raise RunRefused(f"--part must be K/N with 1 <= K <= N and N >= 2, not {text!r}")
    return int(m.group(1)), int(m.group(2))


def select_part(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """Part K of N of a request file: the rows whose messages hash to K - 1 modulo N. Requests with the same messages
    fall in one part, so the cache still gives a repeated input one generation; the N parts cover the file once."""
    h = df["messages_json"].astype(str).map(lambda s: int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16) % n)
    return df[h == k - 1]


def template_kwargs(spec: ModelSpec) -> dict:
    return dict(spec.thinking_off.chat_template_kwargs) if spec.thinking_off.method == "chat_template_kwargs" else {}


def effective_messages(messages: list[dict], spec: ModelSpec) -> list[dict]:
    """Apply the thinking-off system-prompt token, if that is the configured method."""
    if spec.thinking_off.method != "system_prompt_token":
        return messages
    token = spec.thinking_off.system_prompt_token
    if messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + "\n" + token}] + messages[1:]
    return [{"role": "system", "content": token}] + messages


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------

_NO_STEP = object()   # VLLMBackend.generate: no instance attribute of that name before the watch


@dataclasses.dataclass
class GenResult:
    text: str
    n_prompt_tokens: int
    n_output_tokens: int
    finish_reason: str
    prompt_echo: str | None = None   # the prompt the engine itself rendered, when it reports one
    # Size ladder (Decoding.cut_caps): the generated token ids, kept only until the runner has cut the output at the
    # panel's cap; and that cut (cut_record), computed once per result or read back from earlier shards. Neither takes
    # part in comparing two results.
    token_ids: tuple | None = dataclasses.field(default=None, compare=False, repr=False)
    cut: dict | None = dataclasses.field(default=None, compare=False, repr=False)


class Backend:
    name = "base"

    def __init__(self, spec: ModelSpec, decoding: Decoding):
        self.spec, self.decoding = spec, decoding

    def load(self) -> None:
        raise NotImplementedError

    def render(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def chat_template_sha256(self) -> str | None:
        raise NotImplementedError

    def chat_template_text(self) -> str | None:
        """The template string in use, for the clock check; None when the backend has no Jinja template."""
        return None

    def decode_ids(self, ids: Sequence[int], params: Mapping[str, Any]) -> str:
        """Text of a prefix of an output's generated token ids (Decoding.cut_caps)."""
        raise NotImplementedError

    def generate(self, conversations: list[list[dict]], params: list[dict],
                 on_result: Callable[[int, GenResult], None] | None = None,
                 on_tick: Callable[[], None] | None = None) -> list[GenResult]:
        """Results in input order. A backend that sees outputs finish one by one passes each to on_result (input
        index, result) as it finishes and calls on_tick between engine steps, so the runner can save finished
        outputs inside a long chunk; the returned list is still complete."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class EchoBackend(Backend):
    """Deterministic fake outputs for tests. Output length and words depend only on the rendered prompt."""
    name = "echo"
    TEMPLATE_ID = "echo-template-v1"

    def __init__(self, spec: ModelSpec, decoding: Decoding, delay_s: float = 0.0):
        super().__init__(spec, decoding)
        self.generate_calls: list[int] = []   # batch sizes, for tests
        self.delay_s = delay_s                # simulated latency per generate call, for timing tests

    def load(self) -> None:
        return None

    def render(self, messages: list[dict]) -> str:
        head = "<|kwargs|>" + canonical_json(template_kwargs(self.spec)) + "\n"
        if messages[-1]["role"] == "assistant":   # a prefilled assistant turn, continued as is
            body = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages[:-1])
            return head + body + "<|assistant|>\n" + messages[-1]["content"]
        body = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        return head + body + "<|assistant|>\n"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def chat_template_sha256(self) -> str:
        return sha256_text(self.TEMPLATE_ID)

    def generate(self, conversations: list[list[dict]], params: list[dict],
                 on_result: Callable[[int, GenResult], None] | None = None,
                 on_tick: Callable[[], None] | None = None) -> list[GenResult]:
        self.generate_calls.append(len(conversations))
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for conv, p in zip(conversations, params):
            prompt = self.render(conv)
            # a sampled request (Decoding.sampling): the fake output depends on its seed too, as a sampled one would
            salt = "" if p.get("sampling_seed") is None else f"|seed={int(p['sampling_seed'])}|t={p['temperature']}"
            h = hashlib.sha256((prompt + salt).encode("utf-8")).digest()
            if p.get("guided"):   # a constrained output: one of the choices, or a fixed string for a regex
                g = p["guided"]
                text = g["choice"][h[0] % len(g["choice"])] if "choice" in g else "7}"
                out.append(GenResult(text, self.count_tokens(prompt), 1, "stop", prompt))
                continue
            n = 1 + int.from_bytes(h[:2], "big") % 300
            ids = [(h[(3 * i) % 32] * 256 + h[(3 * i + 1) % 32]) % 1000 for i in range(n)]
            finish = "stop"
            if n > p["max_new_tokens"]:
                ids, finish = ids[: p["max_new_tokens"]], "length"
            out.append(GenResult(self.decode_ids(ids, p), self.count_tokens(prompt), len(ids), finish, prompt,
                                 token_ids=tuple(ids)))
        # finish in an order unrelated to the input order, as vLLM's continuous batching does
        for j in sorted(range(len(out)), key=lambda j: (out[j].n_output_tokens, j)):
            if on_result is not None:
                on_result(j, out[j])
            if on_tick is not None:
                on_tick()
        return out

    def decode_ids(self, ids: Sequence[int], params: Mapping[str, Any]) -> str:
        return " ".join(f"t{i}" for i in ids)

    def describe(self) -> dict:
        return {"name": self.name, "template_id": self.TEMPLATE_ID}


def template_reads_clock(text: str | None) -> bool:
    """True when a chat template calls strftime_now, the date function transformers gives Jinja templates. Such a
    template writes the date of the run into the prompt (Mistral-Small-3.1's default system message does), so the
    rendered prompt, its hash and the cache key change from one day to the next. Jinja comments are ignored, since
    a pinned copy may say in its header which vendor line it replaced."""
    return text is not None and "strftime_now" in re.sub(r"\{#.*?#\}", "", text, flags=re.DOTALL)


def template_text_sha256(template: Any) -> str | None:
    """sha256 of the chat template string in use. A dict of named templates hashes its default entry."""
    if template is None:
        return None
    if isinstance(template, Mapping):
        template = template.get("default", canonical_json(dict(template)))
    return sha256_text(str(template))


def structured_kwargs(guided: Mapping | None) -> dict:
    """SamplingParams arguments of a guided-decoding constraint ({"choice": [...]} or {"regex": "..."}).
    vLLM names the field structured_outputs (StructuredOutputsParams) in recent releases and guided_decoding
    (GuidedDecodingParams) in earlier ones; both take choice= and regex=. Nothing for an unguided request."""
    if not guided:
        return {}
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(**dict(guided))}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(**dict(guided))}


class VLLMBackend(Backend):
    """Offline vllm.LLM.chat with prefix caching. vLLM is imported lazily, inside load()."""
    name = "vllm"

    def __init__(self, spec: ModelSpec, decoding: Decoding):
        super().__init__(spec, decoding)
        self.llm = None
        self.tokenizer = None
        self.template_override: str | None = None
        self.engine_kwargs: dict[str, Any] = {}
        self._template_sha: str | None = None
        self._template_text: str | None = None
        self.watch_notes: list[str] = []   # generate(): why early saving stopped, for the run log
        self.renderer_module: Any = None           # a configured renderer (load_renderer), else None
        self.renderer_check: dict | None = None    # check_renderer's record, for the run log
        self.quantization_seen: dict | None = None  # quantization_source checkpoint: what vLLM reports

    def load(self) -> None:
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        from vllm import LLM  # lazy: absent on the Windows development machine
        s, eng = self.spec, self.decoding.engine
        kw: dict[str, Any] = dict(
            model=s.hf_id, revision=s.revision, tokenizer_revision=s.revision, tokenizer_mode=s.tokenizer_mode,
            dtype=s.dtype, tensor_parallel_size=s.tensor_parallel_size, max_model_len=s.max_model_len,
            seed=self.decoding.seed, trust_remote_code=False,
            gpu_memory_utilization=eng["gpu_memory_utilization"], enable_prefix_caching=eng["enable_prefix_caching"],
            enforce_eager=eng["enforce_eager"], generation_config=eng["generation_config"])
        if s.quantization and s.quantization_source == "argument":
            kw["quantization"] = s.quantization
        for k in ("max_num_seqs", "max_num_batched_tokens"):
            if eng[k] is not None:
                kw[k] = eng[k]
        kw.update(s.engine)
        if s.pipeline_parallel_size > 1:
            kw["pipeline_parallel_size"] = s.pipeline_parallel_size
        if s.gpu_memory_utilization is not None:
            kw["gpu_memory_utilization"] = s.gpu_memory_utilization
        if s.text_only:
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        if s.chat_template.source == "file":
            path = Path(s.chat_template.file)
            self.template_override = (path if path.is_absolute() else ROOT / path).read_text(encoding="utf-8")
            self._template_sha = template_text_sha256(self.template_override)
            self._template_text = self.template_override
        if s.renderer:   # the release's own renderer, from the pinned snapshot, checked against its recorded sha256
            self.renderer_module, self._template_sha = load_renderer(s)
        self.engine_kwargs = kw
        self.llm = LLM(**kw)
        self.tokenizer = self.llm.get_tokenizer()
        if s.quantization_source == "checkpoint":
            try:
                seen = getattr(self.llm.llm_engine.model_config, "quantization", None)
            except Exception as exc:  # noqa: BLE001
                seen = f"unreadable: {type(exc).__name__}"
            self.quantization_seen = {"registered": s.quantization, "reported_by_vllm": seen,
                                      "equal": seen == s.quantization}
            if seen != s.quantization:
                self.watch_notes.append(f"vLLM reports quantization {seen!r}; the registry says {s.quantization!r}")
        if s.renderer:
            self.renderer_check = check_renderer(self.render, self.tokenizer, s)
        elif self._template_sha is None:
            tmpl = getattr(self.tokenizer, "chat_template", None)
            if tmpl is None:  # vLLM's tokenizer wrapper may not expose it; read it from the pinned repo
                from transformers import AutoTokenizer
                tmpl = AutoTokenizer.from_pretrained(s.hf_id, revision=s.revision).chat_template
            self._template_sha = template_text_sha256(tmpl)
            self._template_text = None if tmpl is None else str(tmpl.get("default", tmpl) if isinstance(tmpl, Mapping) else tmpl)

    def render(self, messages: list[dict]) -> str:
        # A final assistant message is a prefill: no new generation prompt, and the template continues that turn.
        cont = messages[-1]["role"] == "assistant"
        if self.spec.renderer:
            return RENDER_FUNCS[self.spec.renderer["kind"]](self.renderer_module, messages, template_kwargs(self.spec))
        if cont and self.spec.prefill_mode == "suffix":   # the generation prompt, then the prefill text as it stands
            return self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
                **({"chat_template": self.template_override} if self.template_override else {}),
                **template_kwargs(self.spec)) + messages[-1]["content"]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not cont,
            **({"continue_final_message": True} if cont else {}),
            **({"chat_template": self.template_override} if self.template_override else {}),
            **template_kwargs(self.spec))

    def token_ids(self, text: str) -> list[int]:
        """Token ids of a rendered prompt, without added special tokens (the rendering carries its own), as
        scripts/rp_feasibility.py tokenised DeepSeek's suffix prompts."""
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_tokens(self, text: str) -> int:
        if self.spec.text_prompts:
            return len(self.token_ids(text))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chat_template_sha256(self) -> str | None:
        return self._template_sha

    def chat_template_text(self) -> str | None:
        return self._template_text

    def decode_ids(self, ids: Sequence[int], params: Mapping[str, Any]) -> str:
        """The engine tokenizer's decoding of the ids with the request's skip_special_tokens, as vLLM detokenizes a
        finished output. Whether this equals a run at the lower cap is checked by scripts/check_ladder_equiv.py (size ladder)."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=bool(params.get("skip_special_tokens", True)))

    @staticmethod
    def _result(ro: Any, echo: bool = True) -> GenResult:
        o = ro.outputs[0]
        return GenResult(o.text, len(ro.prompt_token_ids), len(o.token_ids), str(o.finish_reason),
                         getattr(ro, "prompt", None) if echo else None, token_ids=tuple(o.token_ids))

    def generate(self, conversations: list[list[dict]], params: list[dict],
                 on_result: Callable[[int, GenResult], None] | None = None,
                 on_tick: Callable[[], None] | None = None) -> list[GenResult]:
        from vllm import SamplingParams
        # temperature is 0.0 in every greedy decoding's params; a sampled one (Decoding.sampling) gives its temperature
        # and each request's own seed, so the request's output depends on its prompt, parameters and seed alone.
        sps = [SamplingParams(n=1, temperature=float(p["temperature"]), top_p=p["top_p"], top_k=p["top_k"],
                              max_tokens=p["max_new_tokens"], stop=p["stop"] or None,
                              include_stop_str_in_output=p["include_stop_str_in_output"],
                              skip_special_tokens=p["skip_special_tokens"], **structured_kwargs(p.get("guided")),
                              **({"seed": int(p["sampling_seed"])} if p.get("sampling_seed") is not None else {}))
               for p in params]
        # vLLM takes the continuation flags per call, so prefilled and plain conversations go in separate calls.
        # With text prompts (a renderer, or suffix prefills) every conversation is rendered here and all go in one
        # LLM.generate call as token ids (flag None); the engine then echoes no prompt text to compare.
        cont = [c[-1]["role"] == "assistant" for c in conversations]
        if self.spec.text_prompts:
            groups = [(None, list(range(len(conversations))))]
        else:
            groups = [(flag, [i for i, c in enumerate(cont) if c == flag]) for flag in (False, True)]
        echo = not self.spec.text_prompts
        outs: list[Any] = [None] * len(conversations)
        early: dict[int, GenResult] = {}
        eng = self.llm.llm_engine
        for flag, idx in groups:
            if not idx:
                continue
            # Outputs that finish inside a call go to on_result at once, so the runner can save them
            # before the call ends: a chunk of long thinking outputs runs for hours, and a preemption or a
            # time limit used to lose all of it. In vLLM 0.30, LLM.chat adds the conversations one by one, in input
            # order: LLM._add_request calls llm_engine.add_request(request_id, ...) once per conversation, with the
            # id that its finished RequestOutput carries (add_request itself returns an internal id with a random
            # suffix, so its return value cannot be used); LLM._run_engine then calls llm_engine.step() until no
            # request is left. Both engine methods are watched on this instance for the call: the ids map each
            # finished output back to its input. Saving earlier changes no output: the engine receives the same
            # requests in the same order.
            watch = (on_result is not None or on_tick is not None) and callable(getattr(eng, "add_request", None)) \
                and callable(getattr(eng, "step", None))
            # Fail safe: when the ids do not line up as described (vLLM changed), early saving stops for the rest of
            # the call, with a note in the run log, and the call's outputs are saved when it returns, as before.
            if watch:
                ids: dict[str, int] = {}
                off = {"why": None}
                own_add = getattr(eng, "__dict__", {}).get("add_request", _NO_STEP)
                own = getattr(eng, "__dict__", {}).get("step", _NO_STEP)

                def stop_early(why: str) -> None:
                    if off["why"] is None:
                        off["why"] = why
                        self.watch_notes.append(f"early saving stopped inside a call: {why}")

                def add(request_id, *a, _inner=eng.add_request, _idx=idx, **kw):
                    if str(request_id) in ids or len(ids) >= len(_idx):
                        stop_early(f"request id {request_id} repeats, or more requests than inputs")
                    else:
                        ids[str(request_id)] = _idx[len(ids)]
                    return _inner(request_id, *a, **kw)

                def step(_inner=eng.step):
                    got_now = _inner()
                    for ro in got_now:
                        if on_result is not None and off["why"] is None and getattr(ro, "finished", False):
                            j = ids.get(str(ro.request_id))
                            if j is None:
                                stop_early(f"request id {ro.request_id} was not added by this call")
                                continue
                            r = self._result(ro, echo)
                            early[j] = r
                            on_result(j, r)
                    if on_tick is not None:
                        on_tick()
                    return got_now
                eng.add_request = add
                eng.step = step
            try:
                # vLLM's progress bar ("Processed prompts: k/n") is the only sign of progress inside a chunk; the job
                # pane reads it.
                if flag is None:
                    from vllm.inputs import TokensPrompt
                    prompts = [TokensPrompt(prompt_token_ids=self.token_ids(self.render(conversations[i]))) for i in idx]
                    got = self.llm.generate(prompts, [sps[i] for i in idx], use_tqdm=True)
                else:
                    got = self.llm.chat([conversations[i] for i in idx], [sps[i] for i in idx], use_tqdm=True,
                                        chat_template=self.template_override,
                                        chat_template_kwargs=template_kwargs(self.spec) or None,
                                        add_generation_prompt=not flag, continue_final_message=flag)
            finally:
                if watch:   # remove the instance attributes, so the class's own methods show through again
                    if own is _NO_STEP:
                        del eng.step
                    else:
                        eng.step = own
                    if own_add is _NO_STEP:
                        del eng.add_request
                    else:
                        eng.add_request = own_add
            if watch and off["why"] is None and len(ids) != len(idx):
                self.watch_notes.append(f"vLLM added {len(ids)} requests for {len(idx)} inputs")
            for i, ro in zip(idx, got):
                outs[i] = ro
        res = [self._result(ro, echo) for ro in outs]
        for i, r in early.items():   # the early result of each index must be the one the call returned for it
            if r != res[i]:
                raise GenerationError(f"the output saved early for input {i} differs from the one vLLM returned")
        return res

    def describe(self) -> dict:
        d = {"name": self.name, "engine_kwargs": {k: v for k, v in self.engine_kwargs.items()}}
        if self.spec.renderer:
            d["renderer"] = {**self.spec.renderer, "check": getattr(self, "renderer_check", None)}
        if self.spec.prefill_mode != "continue_final_message":
            d["prefill_mode"] = self.spec.prefill_mode
        if self.quantization_seen is not None:
            d["quantization"] = self.quantization_seen
        return d


def load_renderer(spec: ModelSpec, snapshot_dir: str | os.PathLike | None = None) -> tuple[Any, str]:
    """Import the renderer file of a registry entry from the pinned snapshot (read offline), after checking its
    sha256 against the registry. Returns (module, sha256)."""
    import importlib.util
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(spec.hf_id, revision=spec.revision, local_files_only=True)
    path = Path(snapshot_dir) / spec.renderer["file"]
    if not path.is_file():
        raise RunRefused(f"renderer file {spec.renderer['file']} is not in the snapshot {snapshot_dir}")
    sha = sha256_text(path.read_text(encoding="utf-8"))
    if sha != spec.renderer["sha256"]:
        raise RunRefused(f"renderer file sha256 {sha} differs from the registered {spec.renderer['sha256']}")
    mod_spec = importlib.util.spec_from_file_location(f"_ladder_renderer_{abs(hash(sha))}", path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module, sha


def render_deepseek_v4(module: Any, messages: list[dict], kwargs: Mapping[str, Any]) -> str:
    """DeepSeek-V4's own encode_messages (encoding/encoding_dsv4.py). The thinking switch of the registry's
    chat_template_kwargs (thinking or enable_thinking, false by default) picks thinking_mode chat or thinking; a final
    assistant message is a prefill, appended to the generation prompt as text (the suffix_box prefill)."""
    prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else None
    msgs = [dict(m) for m in (messages[:-1] if prefill is not None else messages)]
    thinking = bool(kwargs.get("thinking", kwargs.get("enable_thinking", False)))
    extra = {"reasoning_effort": kwargs["reasoning_effort"]} if kwargs.get("reasoning_effort") else {}
    return module.encode_messages(msgs, thinking_mode="thinking" if thinking else "chat", **extra) + (prefill or "")


RENDER_FUNCS: dict[str, Callable[[Any, list[dict], Mapping[str, Any]], str]] = {
    "deepseek_v4_encoding": render_deepseek_v4}

RENDERER_PROBE = [{"role": "user", "content": "What is 2 + 2?\nEnd your reply with a final line that contains "
                                               "\\boxed{<number>}."}]


def check_renderer(render: Callable[[list[dict]], str], tokenizer: Any, spec: ModelSpec) -> dict:
    """The renderer's prompt for a probe conversation against the engine tokenizer's own apply_chat_template with the
    same switch (the rendering scripts/rp_feasibility.py used). A difference refuses the run; a tokenizer that cannot
    render is recorded and the run goes on with the renderer."""
    ours = render(RENDERER_PROBE)
    try:
        theirs = tokenizer.apply_chat_template(RENDERER_PROBE, tokenize=False, add_generation_prompt=True,
                                               **template_kwargs(spec))
    except Exception as exc:  # noqa: BLE001
        return {"equal": None, "renderer_sha256": sha256_text(ours), "tokenizer_error": f"{type(exc).__name__}: {exc}"[:500]}
    if ours != theirs:
        raise RunRefused(f"the renderer's prompt differs from the engine tokenizer's rendering of the same conversation "
                         f"(renderer ...{ours[-120:]!r}; tokenizer ...{str(theirs)[-120:]!r})")
    return {"equal": True, "sha256": sha256_text(ours)}


def make_backend(name: str, spec: ModelSpec, decoding: Decoding) -> Backend:
    if name == "vllm":
        return VLLMBackend(spec, decoding)
    if name == "echo":
        return EchoBackend(spec, decoding)
    raise RunRefused(f"unknown backend {name!r}")


# --------------------------------------------------------------------------------------------
# Shards and resume
# --------------------------------------------------------------------------------------------

def batch_invariant_env() -> bool:
    """VLLM_BATCH_INVARIANT as vLLM 0.30.0 reads it (vllm/envs.py: an integer, default 0)."""
    raw = os.environ.get("VLLM_BATCH_INVARIANT", "0")
    try:
        return bool(int(raw))
    except ValueError:
        raise RunRefused(f"VLLM_BATCH_INVARIANT={raw!r} is not an integer; vLLM would refuse it too") from None


def inductor_switches_env() -> list[str]:
    """The Inductor kernel switches that are on, in INDUCTOR_KERNEL_ENV order."""
    return [name for name in INDUCTOR_KERNEL_ENV if os.environ.get(name) == "1"]


def apply_kernel_env(spec: ModelSpec) -> dict:
    """Set the kernel switches of the model's configured mode in this process, before vLLM or torch is imported,
    so that the engine and the processes it starts read them. The registry is the one place where the mode is chosen: a
    switch already set to another value refuses the run, as does a switch that the mode leaves off but the environment
    turns on. Returns the record stored in the run log."""
    want = KERNEL_MODES[spec.kernels]
    conflicts, set_now = [], []
    for name in KERNEL_SWITCHES:
        raw = os.environ.get(name)
        if name in want:
            if raw is None:
                set_now.append(name)
            elif raw != want[name]:
                conflicts.append(f"{name}={raw!r} (the mode sets {want[name]!r})")
        elif raw is not None and (batch_invariant_env() if name == "VLLM_BATCH_INVARIANT" else raw == "1"):
            conflicts.append(f"{name}={raw!r} (the mode leaves it off)")
    if conflicts:
        raise RunRefused(f"model {spec.model_key!r} is configured with kernels: {spec.kernels}, but the environment "
                         f"has {', '.join(conflicts)}; unset them and let the runner set the mode")
    loaded = [m for m in ("torch", "vllm") if m in sys.modules]
    if set_now and loaded:
        raise RunRefused(f"{' and '.join(loaded)} is already imported, so {', '.join(set_now)} would not take effect; "
                         "export the switches before Python starts")
    for name in set_now:
        os.environ[name] = want[name]
    return {"mode": spec.kernels, "switches": dict(want), "set_by_runner": set_now}


def run_fingerprint(spec: ModelSpec, decoding: Decoding, backend: str,
                    cap_override: Mapping[str, int] | None = None, batch_invariant: bool = False,
                    inductor_switches: Sequence[str] = ()) -> str:
    """`decoding` is the decoding in force (caps after any override). A cap override is added to the
    payload explicitly, so a run with an override never shares a fingerprint with one without, even
    when the override equals the configured cap. Batch-invariant kernels and Inductor's kernel switches
    are added the same way: they change the outputs, so a folder started with them never resumes
    without them, and the reverse. Without any of these the payload is unchanged."""
    eng = decoding.engine
    model = spec.to_log()
    model.pop("family")   # descriptive only
    model.pop("engine")   # scheduler limits only (MODEL_ENGINE_KEYS)
    model.pop("gpu_memory_utilization", None)   # memory share only
    payload: dict[str, Any] = {
        "backend": backend, "model": model,
        "decoding": {a: decoding.params_for_arm(a) for a in decoding.arms},
        "engine": {"generation_config": eng["generation_config"], "enable_prefix_caching": eng["enable_prefix_caching"]},
    }
    if cap_override:
        payload["cap_override"] = {str(a): int(n) for a, n in sorted(cap_override.items())}
    if batch_invariant:
        payload["kernels"] = {"batch_invariant": True}
    if inductor_switches:
        payload.setdefault("kernels", {})["inductor"] = sorted(inductor_switches)
    return sha256_text(canonical_json(payload))


# Size ladder (Decoding.cut_caps): the output cut at the panel's cap, beside the uncut one. panel_cap is the arm's cut
# cap, or its max_new_tokens when the arm has none (its output is then never cut); output_token_ids_cut holds the kept
# ids of a cut output only.
CUT_COLUMNS = ("panel_cap", "cut_at_panel_cap", "output_text_cut", "n_output_tokens_cut", "finish_reason_cut",
               "output_token_ids_cut")


def output_schema(item_id_type: pa.DataType, cut: bool = False) -> pa.Schema:
    fields = [
        ("request_id", pa.string()), ("model_key", pa.string()), ("arm", pa.string()),
        ("config_id", pa.string()), ("item_id", item_id_type), ("prompt_sha256", pa.string()),
        ("output_text", pa.string()), ("n_prompt_tokens", pa.int64()), ("n_output_tokens", pa.int64()),
        ("finish_reason", pa.string()), ("cached", pa.bool_())]
    if cut:
        fields += [("panel_cap", pa.int64()), ("cut_at_panel_cap", pa.bool_()), ("output_text_cut", pa.string()),
                   ("n_output_tokens_cut", pa.int64()), ("finish_reason_cut", pa.string()),
                   ("output_token_ids_cut", pa.list_(pa.int64()))]
    return pa.schema(fields)


def cut_record(r: GenResult, cut_cap: int | None, arm_cap: int, decode: Callable[[list[int]], str]) -> dict:
    """An output cut at the token level to the panel's cap: the first cut_cap
    generated tokens, decoded by `decode`, finish_reason length, marked cut, whenever the output ran past cut_cap; else
    the output as it is. An arm without a cut cap keeps its output, with panel_cap its own cap."""
    same = {"panel_cap": int(cut_cap or arm_cap), "cut_at_panel_cap": False, "output_text_cut": r.text,
            "n_output_tokens_cut": int(r.n_output_tokens), "finish_reason_cut": r.finish_reason,
            "output_token_ids_cut": None}
    if cut_cap is None or r.n_output_tokens <= cut_cap:
        return same
    if r.token_ids is None or len(r.token_ids) != r.n_output_tokens:
        raise GenerationError("an output longer than the panel's cap came without its generated token ids, so it "
                              "cannot be cut at the token level")
    ids = [int(t) for t in r.token_ids[:cut_cap]]
    return {**same, "cut_at_panel_cap": True, "output_text_cut": decode(ids), "n_output_tokens_cut": int(cut_cap),
            "finish_reason_cut": "length", "output_token_ids_cut": ids}


def list_shards(work_dir: Path) -> list[Path]:
    return sorted(work_dir.glob("shard-*.parquet")) if work_dir.is_dir() else []


def gpu_type(gpu: Mapping | None) -> tuple[str, ...]:
    """The GPU model names of a gpu_info() record, sorted; empty when no GPU was seen."""
    return tuple(sorted(str(d.get("name")) for d in ((gpu or {}).get("devices") or []) if d.get("name")))


def prior_gpu_types(work_dir: Path) -> dict[str, tuple[str, ...]]:
    """GPU types of the earlier runs that wrote shards into this folder, by run log file name."""
    out = {}
    for lp in sorted(Path(work_dir).glob("run_log-*.json")):
        try:
            log = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if log.get("shards_written") and gpu_type(log.get("gpu")):
            out[lp.name] = gpu_type(log.get("gpu"))
    return out


def check_same_gpu_type(work_dir: Path, gpu: Mapping) -> None:
    """Refuse to resume a folder on another GPU model (hardware is pinned). Greedy outputs can
    differ between GPU models, and a model's outputs must come from one of them. The fingerprint does not cover
    the GPU, so a requeued job that lands on another card type would otherwise mix them."""
    now = gpu_type(gpu)
    before = set(prior_gpu_types(work_dir).values())
    if now and before and before != {now}:
        raise RunRefused(f"{work_dir} holds outputs generated on {sorted(before)} but this node has {list(now)}; a "
                         "model's outputs come from one GPU model. Pin the GPU type in the job's constraints or "
                         "generate into a fresh folder")


def check_shard_fingerprints(shards: Sequence[Path], fingerprint: str) -> None:
    for sh in shards:
        try:
            meta = pq.read_schema(sh).metadata or {}
        except Exception as exc:  # unreadable parquet: never guess
            raise RunRefused(f"{sh} is unreadable ({exc}); inspect or remove it before resuming") from exc
        if meta.get(b"run_fingerprint", b"").decode() != fingerprint:
            raise RunRefused(f"{sh.name} was written under a different model/decoding configuration; use a fresh --out-dir")


def write_shard(work_dir: Path, run_id: str, index: int, rows: list[dict], schema: pa.Schema,
                meta: Mapping[str, str]) -> Path:
    table = pa.Table.from_pylist(rows, schema=schema).replace_schema_metadata(dict(meta))
    final = work_dir / f"shard-{run_id}-{index:05d}.parquet"
    tmp = final.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, final)
    return final


def read_outputs(work_dir: str | os.PathLike) -> pd.DataFrame:
    shards = list_shards(Path(work_dir))
    if not shards:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def load_prior(shards: Sequence[Path], spec: ModelSpec, decoding: Decoding) -> tuple[set[str], dict[str, GenResult]]:
    done: set[str] = set()
    cache: dict[str, GenResult] = {}
    cols = ["request_id", "arm", "prompt_sha256", "output_text", "n_prompt_tokens", "n_output_tokens", "finish_reason"]
    if decoding.cut_caps:
        cols += list(CUT_COLUMNS)
    for sh in shards:
        t = pq.read_table(sh, columns=cols).to_pydict()
        for i, rid in enumerate(t["request_id"]):
            done.add(rid)
            if decoding.guided.get(t["arm"][i]) or decoding.sampling:
                continue   # the key needs the request's constraint or seed, which shards do not keep
            k = cache_key(spec.hf_id, spec.revision, t["prompt_sha256"][i], decoding.params_for_arm(t["arm"][i]))
            cut = {c: t[c][i] for c in CUT_COLUMNS} if decoding.cut_caps else None
            cache.setdefault(k, GenResult(t["output_text"][i], t["n_prompt_tokens"][i],
                                          t["n_output_tokens"][i], t["finish_reason"][i], cut=cut))
    return done, cache


# --------------------------------------------------------------------------------------------
# Environment, log, statistics
# --------------------------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def parse_nvidia_smi(query_csv: str, header: str) -> dict:
    devices = []
    driver = None
    for line in query_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            devices.append({"name": parts[0], "memory_mib": int(float(parts[1])) if parts[1] else None})
            driver = parts[2]
    m = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", header)   # newer drivers print "CUDA UMD Version"
    return {"count": len(devices), "devices": devices, "driver_version": driver,
            "cuda_version_driver": m.group(1) if m else None}


def gpu_info(timeout: float = 10.0) -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"count": 0, "devices": [], "driver_version": None, "cuda_version_driver": None, "nvidia_smi": False}
    try:
        q = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=timeout, check=True).stdout
        h = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, check=True).stdout
        return {**parse_nvidia_smi(q, h), "nvidia_smi": True}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"count": 0, "devices": [], "driver_version": None, "cuda_version_driver": None,
                "nvidia_smi": True, "error": type(exc).__name__}


def library_versions() -> dict:
    out: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in _LOGGED_PACKAGES:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = None
    torch = sys.modules.get("torch")   # only report CUDA when torch is already loaded
    out["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None) if torch else None
    return out


def scheduler_info() -> dict:
    return {k: os.environ[k] for k in _LOGGED_ENV if k in os.environ}


def length_distribution(values: Sequence[int]) -> dict:
    if not len(values):
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    q = np.percentile(a, [50, 90, 95, 99])
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": int(a.min()), "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
            "p99": float(q[3]), "max": int(a.max())}


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class StopFlag:
    """Set by SIGTERM or SIGUSR1; checked between chunks, and inside a chunk to write its finished outputs at once."""

    def __init__(self) -> None:
        self.requested = False
        self.signum: int | None = None

    def __call__(self) -> bool:
        return self.requested

    def install(self) -> "StopFlag":
        def handler(signum, frame):
            self.requested, self.signum = True, signum
        for name in ("SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), handler)
        return self


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class _Prep:
    row: dict
    messages: list[dict]
    prompt: str
    prompt_sha: str
    params: dict
    key: str


def _takes_callbacks(generate: Callable) -> bool:
    """True when a backend's generate accepts on_result and on_tick (a subclass may override it without them)."""
    try:
        names = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return False
    return "on_result" in names and "on_tick" in names


def _say(msg: str) -> None:
    print(f"[generate_d2 {utc_now()}] {msg}", file=sys.stderr, flush=True)


def run_generation(
    requests_path: str | os.PathLike, model_key: str, out_dir: str | os.PathLike, *,
    backend: str = "vllm", backend_obj: Backend | None = None, limit: int | None = None,
    throughput: bool = False, shard_size: int = DEFAULT_SHARD_SIZE, flush_seconds: float | None = DEFAULT_FLUSH_S,
    models_path: str | os.PathLike | None = None, decoding_path: str | os.PathLike | None = None,
    budget_path: str | os.PathLike | None = None, should_stop: Callable[[], bool] | None = None,
    verbose: bool = True, cap_override: Mapping[str, int] | None = None, part: tuple[int, int] | None = None,
) -> dict:
    """Run (or resume) generation for one model. Returns the run log dict; its `status` is
    complete or interrupted. Raises RunRefused before any generation when a gate fails.
    `cap_override` ({arm: N}) replaces max_new_tokens for those arms in this throughput run only.
    `part` (K, N) runs part K of N of the file (select_part) into the same folder as the other parts, so N jobs on
    N cards share one model's run; a finished part has status part_complete, and the folder is complete once every
    part of one N is (d2_io)."""
    say = _say if verbose else (lambda m: None)
    should_stop = should_stop or (lambda: False)
    if shard_size < 1:
        raise RunRefused("shard_size must be at least 1")
    if flush_seconds is not None and not flush_seconds >= 0:
        raise RunRefused("flush_seconds must be at least 0")
    if limit is not None and limit < 1:
        raise RunRefused("limit must be a positive integer")
    if throughput and limit is None:
        raise RunRefused("throughput mode needs --limit N (the subset size)")
    if part is not None and (throughput or limit is not None):
        raise RunRefused("--part splits a full run; it does not combine with --limit or --throughput")
    if cap_override and not throughput:
        raise RunRefused("--measure-cap is allowed only with --throughput: a changed output cap would give "
                         "main-run outputs that differ from the configured decoding")
    spec = parse_model_spec(model_key, load_registry(models_path))
    decoding_configured = load_decoding(decoding_path)
    override = validate_cap_override(cap_override, decoding_configured, spec)
    decoding = apply_cap_override(decoding_configured, override)   # the decoding in force for this run
    preflight_record = preflight(spec, decoding, backend=backend, throughput=throughput, limit=limit,
                                 budget_path=budget_path, calibration=calibration_build(requests_path, model_key))
    kernel_record = apply_kernel_env(spec) if backend == "vllm" else \
        {"mode": spec.kernels, "switches": None, "set_by_runner": [], "note": "not applied: backend is not vllm"}

    t_start, started = time.time(), utc_now()
    run_id = new_run_id()
    df_all = read_requests(requests_path)
    validate_requests(df_all, model_key, decoding, spec.requests_model_key)
    n_in_file = len(df_all)
    arm_totals = df_all["arm"].value_counts().to_dict()
    df = order_requests(df_all)
    if throughput:
        df = select_throughput_subset(df, int(limit), decoding.seed)
    elif limit is not None:
        df = df.head(int(limit))
    elif part is not None:
        df = select_part(df, *part)

    out_dir = Path(out_dir)
    work_dir = out_dir / "throughput" if throughput else out_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    batch_invariant = backend == "vllm" and batch_invariant_env()
    inductor = inductor_switches_env() if backend == "vllm" else []
    fingerprint = run_fingerprint(spec, decoding, backend, override, batch_invariant, inductor)
    shards = [] if throughput else list_shards(work_dir)
    check_shard_fingerprints(shards, fingerprint)
    gpu = gpu_info()
    if shards:
        check_same_gpu_type(work_dir, gpu)
    done, prior_cache = load_prior(shards, spec, decoding) if shards else (set(), {})
    todo = df[~df["request_id"].isin(done)]

    log: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "mode": "throughput" if throughput else "full",
        "excluded_from_main": bool(throughput), "status": "failed", "started_utc": started,
        "hostname": socket.gethostname(), "scheduler": scheduler_info(), "gpu": gpu,
        "versions": library_versions(), "preflight": preflight_record, "model": spec.to_log(),
        "decoding": decoding.to_log(), "decoding_confirmed": decoding.confirmed,
        "cap_override": None if not override else {
            "arms": dict(override), "configured_max_new_tokens": dict(decoding_configured.max_new_tokens),
            "effective_max_new_tokens": dict(decoding.max_new_tokens),
            "note": "throughput measurement only; the decoding block shows the caps applied in this run"},
        "run_fingerprint": fingerprint, "batch_invariant": batch_invariant, "inductor_switches": inductor,
        "kernels": kernel_record,
        "inputs": {"requests_path": str(requests_path), "requests_sha256": sha256_file(requests_path),
                   "requests_in_file": n_in_file, "limit": limit, "part": list(part) if part else None,
                   "generation_py_sha256": sha256_file(Path(__file__)),
                   "models_yaml_sha256": sha256_file(models_path or MODELS_YAML),
                   "decoding_yaml_sha256": sha256_file(decoding_path or DECODING_YAML)},
        "backend": {"name": backend}, "requests_selected": len(df), "requests_already_done": len(df) - len(todo),
        "requests_this_run": len(todo), "shard_size": shard_size, "flush_seconds": flush_seconds,
        "shards_written": [],
    }
    log_name = ("throughput_log-" if throughput else "run_log-") + run_id + ".json"
    try:
        _execute(log, todo, spec, decoding, backend, backend_obj, prior_cache, work_dir, run_id, fingerprint,
                 shard_size, throughput, arm_totals, should_stop, say, flush_seconds)
        if part is not None and log["status"] == "complete":
            log["status"] = "part_complete"   # never read as a complete run of the file
    except BaseException as exc:
        log["status"] = "failed"
        log["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log["ended_utc"] = utc_now()
        log["runtime_s"] = round(time.time() - t_start, 3)
        log["gpu_hours_billed_estimate"] = round(log["runtime_s"] / 3600.0 * spec.gpus, 4) \
            if backend == "vllm" else 0.0
        write_json(work_dir / log_name, log)
        say(f"run log: {work_dir / log_name}")
    return log


def _execute(log, todo, spec, decoding, backend_name, backend_obj, prior_cache, work_dir, run_id, fingerprint,
             shard_size, throughput, arm_totals, should_stop, say, flush_seconds=DEFAULT_FLUSH_S) -> None:
    arms = decoding.arms
    zero = lambda: {a: 0 for a in arms}  # noqa: E731
    counts = {"unique_generations": zero(), "cached_calls": zero(), "cached_from_prior_shards": zero(),
              "prompt_tokens_generated": zero(), "output_tokens_generated": zero(),
              "output_tokens_all_requests": zero(), "truncated_generated": zero(), "truncated_all_requests": zero(),
              "outputs_with_think_tag_generated": zero()}
    lengths: dict[str, list[int]] = {a: [] for a in arms}
    prompt_lens: dict[str, list[int]] = {a: [] for a in arms}
    log["counts"] = counts
    gen_seconds = 0.0
    mismatches = 0

    if todo.empty:
        log["status"] = "complete"
        log["note"] = "nothing to do: every selected request_id is already present"
        log["totals"] = _totals(counts)
        return

    be = backend_obj or make_backend(backend_name, spec, decoding)
    t0 = time.time()
    be.load()
    log["model_load_s"] = round(time.time() - t0, 3)
    log["backend"] = be.describe()
    observed = be.chat_template_sha256()
    log["chat_template"] = {"source": spec.chat_template.source, "registered_sha256": spec.chat_template.sha256,
                            "observed_sha256": observed}
    if spec.chat_template.sha256 is not None and observed != spec.chat_template.sha256:
        raise RunRefused(f"chat template sha256 {observed} differs from the registered {spec.chat_template.sha256}")
    if backend_name == "vllm" and observed is None and not throughput:
        raise RunRefused("could not read the chat template, so it cannot be pinned")
    log["chat_template"]["reads_clock"] = template_reads_clock(be.chat_template_text())
    if log["chat_template"]["reads_clock"]:
        raise RunRefused("the chat template calls strftime_now, so every rendered prompt carries the date of the run and "
                         "changes from one day to the next ; register a copy of the template with the date "
                         "fixed (chat_template.source: file)")

    say(f"rendering {len(todo)} prompts")
    preps: list[_Prep] = []
    for row in todo.to_dict("records"):
        msgs = effective_messages(parse_messages(row["messages_json"]), spec)
        params = decoding.params_for_arm(row["arm"])
        if params.get("guided_source"):   # the request's own constraint; part of the cache key, never of the prompt
            params = {**params, "guided": parse_guided(row[GUIDED_COLUMN])}
        if params.get("sampling_seed_source"):   # the request's own seed; part of the cache key, never of the prompt
            params = {**params, "sampling_seed": int(row[SEED_COLUMN])}
        if params.get("prefill"):   # open the assistant turn; the model continues it
            msgs = msgs + [{"role": "assistant", "content": params["prefill"]}]
        prompt = be.render(msgs)
        psha = sha256_text(prompt)
        preps.append(_Prep(row, msgs, prompt, psha, params, cache_key(spec.hf_id, spec.revision, psha, params)))

    cache: dict[str, GenResult] = {p.key: prior_cache[p.key] for p in preps if p.key in prior_cache}
    prior_keys = set(cache)
    too_long = []
    longest_prompt = 0
    for p in {p.key: p for p in preps if p.key not in cache}.values():
        n = be.count_tokens(p.prompt)
        longest_prompt = max(longest_prompt, n)
        if n + p.params["max_new_tokens"] > spec.max_model_len:
            too_long.append((p.row["request_id"], n))
    if too_long:
        hint = ""
        if log.get("cap_override"):   # --measure-cap: say what would fit
            hint = (f"; with --measure-cap the longest prompt is {longest_prompt} tokens, so a cap of at most "
                    f"{max(spec.max_model_len - longest_prompt, 0)} fits")
        raise RunRefused(f"{len(too_long)} prompts exceed max_model_len={spec.max_model_len} with their output cap "
                         f"(first: {too_long[:3]}); raise max_model_len in the registry or shorten the prompts{hint}")

    item_type = pa.int64() if pd.api.types.is_integer_dtype(todo["item_id"]) else pa.string()
    cutting = bool(decoding.cut_caps)
    schema = output_schema(item_type, cut=cutting)
    if cutting:
        counts["cut_at_panel_cap_all_requests"] = zero()
    meta = {"run_fingerprint": fingerprint, "model_key": spec.model_key, "run_id": run_id,
            "schema_version": SCHEMA_VERSION, "throughput": str(bool(throughput)).lower()}
    if log.get("cap_override"):
        meta["cap_override"] = canonical_json(log["cap_override"]["arms"])
    i, shard_idx, n = 0, 0, len(preps)
    status = "complete"
    # Inside a chunk, rows whose outputs have finished are written every flush_seconds, and at the next
    # engine step after a stop signal, so a kill loses at most that long of finished work; the rest of the chunk is
    # written when it ends, as before. Each row is written once. Throughput runs keep one shard per chunk.
    incremental = flush_seconds is not None and not throughput and _takes_callbacks(be.generate)
    while i < n:
        chunk: list[_Prep] = []
        fresh: dict[str, _Prep] = {}
        while i < n and len(fresh) < shard_size:
            p = preps[i]
            i += 1
            chunk.append(p)
            if p.key not in cache and p.key not in fresh:
                fresh[p.key] = p
        fresh_list = list(fresh.values())
        accepted: set[str] = set()
        written = [False] * len(chunk)
        last_flush = time.time()

        def accept(p: _Prep, r: GenResult) -> None:
            nonlocal mismatches
            if p.key in accepted:
                if cache[p.key] != r:
                    raise GenerationError(f"two different outputs for request {p.row['request_id']}")
                return
            if r.finish_reason not in FINISH_REASONS:
                raise GenerationError(f"unexpected finish_reason {r.finish_reason!r} for request {p.row['request_id']}")
            if r.prompt_echo is not None and r.prompt_echo != p.prompt:
                mismatches += 1
            if cutting:   # cut once, then drop the token ids (an output's ids are kept only while it is cut)
                arm = p.row["arm"]
                r = dataclasses.replace(r, token_ids=None, cut=cut_record(
                    r, decoding.cut_caps.get(arm), decoding.max_new_tokens[arm],
                    lambda ids, _params=p.params: be.decode_ids(ids, _params)))
            elif r.token_ids is not None:
                r = dataclasses.replace(r, token_ids=None)
            cache[p.key] = r
            accepted.add(p.key)
            a = p.row["arm"]
            counts["unique_generations"][a] += 1
            counts["prompt_tokens_generated"][a] += r.n_prompt_tokens
            counts["output_tokens_generated"][a] += r.n_output_tokens
            counts["truncated_generated"][a] += int(r.finish_reason == "length")
            counts["outputs_with_think_tag_generated"][a] += int(bool(_THINK_TAG.search(r.text)))
            lengths[a].append(r.n_output_tokens)
            prompt_lens[a].append(r.n_prompt_tokens)

        def flush() -> None:
            nonlocal shard_idx, last_flush
            last_flush = time.time()
            rows = []
            for q, p in enumerate(chunk):
                if written[q] or p.key not in cache:
                    continue
                r, a = cache[p.key], p.row["arm"]
                is_new = fresh.get(p.key) is p
                if not is_new:
                    counts["cached_calls"][a] += 1
                    counts["cached_from_prior_shards"][a] += int(p.key in prior_keys)
                counts["output_tokens_all_requests"][a] += r.n_output_tokens
                counts["truncated_all_requests"][a] += int(r.finish_reason == "length")
                rows.append({"request_id": p.row["request_id"], "model_key": spec.model_key, "arm": a,
                             "config_id": p.row["config_id"], "item_id": p.row["item_id"],
                             "prompt_sha256": p.prompt_sha, "output_text": r.text,
                             "n_prompt_tokens": int(r.n_prompt_tokens), "n_output_tokens": int(r.n_output_tokens),
                             "finish_reason": r.finish_reason, "cached": not is_new})
                if cutting:
                    want = int(decoding.cut_caps.get(a) or decoding.max_new_tokens[a])
                    if r.cut is None or int(r.cut["panel_cap"]) != want:
                        raise GenerationError(f"request {p.row['request_id']}: no cut at the panel's cap {want} for "
                                              "its output (a cached output of another arm, or an earlier shard)")
                    rows[-1].update({c: r.cut[c] for c in CUT_COLUMNS})
                    counts["cut_at_panel_cap_all_requests"][a] += int(bool(r.cut["cut_at_panel_cap"]))
                written[q] = True
            if rows:
                path = write_shard(work_dir, run_id, shard_idx, rows, schema, meta)
                log["shards_written"].append(path.name)
                shard_idx += 1

        def on_result(j: int, r: GenResult) -> None:
            accept(fresh_list[j], r)

        def on_tick() -> None:
            # A stop signal ends the chunk here, once its finished outputs are written: at a time limit
            # USR1 comes 300 s ahead, and the job script requeues the job itself when the runner exits 75 in time,
            # which it never did while the runner waited for the chunk to end (hours, for long thinking outputs). The outputs still
            # unfinished are generated again on resume; nothing of them is written.
            if should_stop():
                flush()
                raise _StopInChunk
            if time.time() - last_flush >= flush_seconds:
                flush()

        if fresh:
            t0 = time.time()
            try:
                if incremental:
                    res = be.generate([p.messages for p in fresh_list], [p.params for p in fresh_list],
                                      on_result=on_result, on_tick=on_tick)
                else:
                    res = be.generate([p.messages for p in fresh_list], [p.params for p in fresh_list])
            except _StopInChunk:
                gen_seconds += time.time() - t0
                flush()
                say(f"stopped inside a chunk: {sum(written)} of its {len(chunk)} rows written; the rest are "
                    "generated on resume")
                status = "interrupted"
                break
            gen_seconds += time.time() - t0
            if len(res) != len(fresh):
                raise GenerationError(f"backend returned {len(res)} results for {len(fresh)} prompts")
            for p, r in zip(fresh_list, res):
                accept(p, r)
        flush()
        if not all(written):
            raise GenerationError("a row of the chunk has no output")
        gen_tok = sum(counts["output_tokens_generated"].values())
        say(f"{i}/{n} requests, {sum(counts['unique_generations'].values())} generated, "
            f"{gen_tok / gen_seconds:.0f} output tok/s" if gen_seconds > 0 else f"{i}/{n} requests")
        if i < n and should_stop():
            status = "interrupted"
            break
    log["status"] = status

    log["generation_s"] = round(gen_seconds, 3)
    log["prompt_render_mismatches"] = mismatches
    log["early_saving"] = incremental
    notes = list(dict.fromkeys(getattr(be, "watch_notes", None) or []))
    if notes:
        log.setdefault("warnings", []).extend(notes)
    log["totals"] = _totals(counts)
    out_tok = log["totals"]["output_tokens_generated"]
    log["output_tokens_per_s"] = out_tok / gen_seconds if gen_seconds > 0 else None
    if mismatches:
        log.setdefault("warnings", []).append(
            f"{mismatches} prompts differ between the runner's rendering (cache key) and the engine's rendering")
    if throughput:
        report = throughput_report(log, lengths, prompt_lens, counts, arm_totals, spec, gen_seconds)
        log["throughput"] = report
        write_json(work_dir / f"throughput_report-{run_id}.json", report)
        say(format_throughput(report))


def _totals(counts: Mapping[str, Mapping[str, int]]) -> dict:
    return {k: int(sum(v.values())) for k, v in counts.items()}


def cap_exceedance(lengths: Sequence[int], cap: int | None, candidates: Sequence[int] = CAP_CANDIDATES) -> dict:
    """Share of unique generations whose output length exceeds each candidate cap, for the candidates
    strictly below the cap the run used. A generation cut at the run's cap is longer than every smaller
    candidate, so those shares are exact; candidates at or above the run's cap cannot be read from this
    run and are omitted (the share cut at the run's cap is reported separately). Keys are the candidate
    caps as strings. Length is the engine's output token count."""
    if cap is None or not len(lengths):
        return {}
    a = np.asarray(lengths)
    return {str(k): float((a > k).mean()) for k in candidates if k < cap}


def throughput_report(log: Mapping, lengths: Mapping[str, list[int]], prompt_lens: Mapping[str, list[int]],
                      counts: Mapping, arm_totals: Mapping[str, int], spec: ModelSpec, gen_seconds: float) -> dict:
    per_arm = {}
    caps = (log.get("decoding") or {}).get("max_new_tokens") or {}   # the caps in force for this run
    for a in lengths:
        g = counts["unique_generations"][a]
        per_arm[a] = {"unique_generations": g, "output_tokens": length_distribution(lengths[a]),
                      "prompt_tokens": length_distribution(prompt_lens[a]),
                      "max_new_tokens": caps.get(a),
                      "share_truncated_at_cap": (counts["truncated_generated"][a] / g) if g else None,
                      "share_exceeding_candidate_cap": cap_exceedance(lengths[a], caps.get(a))}
    out_tok = sum(counts["output_tokens_generated"].values())
    in_tok = sum(counts["prompt_tokens_generated"].values())
    tps = out_tok / gen_seconds if gen_seconds > 0 else None
    report: dict[str, Any] = {
        "model_key": spec.model_key, "generation_s": round(gen_seconds, 3), "model_load_s": log.get("model_load_s"),
        "output_tokens_per_s": tps, "total_tokens_per_s": (out_tok + in_tok) / gen_seconds if gen_seconds > 0 else None,
        "unique_generations": int(sum(counts["unique_generations"].values())), "per_arm": per_arm,
        "cap_override": log.get("cap_override"), "excluded_from_main": True}
    if tps:
        est_tokens = sum((per_arm[a]["output_tokens"].get("mean") or 0.0) * arm_totals.get(a, 0) for a in per_arm)
        hours = est_tokens / tps / 3600.0
        report["projection_full_request_file"] = {
            "note": "extrapolation from this subset, generation time only, before caching; not a measured quantity",
            "requests_by_arm": {a: int(arm_totals.get(a, 0)) for a in per_arm},
            "estimated_output_tokens": est_tokens, "estimated_generation_hours": hours,
            "estimated_gpu_hours": hours * spec.gpus}
    return report


def format_throughput(r: Mapping) -> str:
    head = (f"throughput {r['model_key']}: {r['unique_generations']} generations in {r['generation_s']} s, "
            f"{(r['output_tokens_per_s'] or 0):.0f} output tok/s")
    co = r.get("cap_override")
    if co:
        head += " [cap override: " + ", ".join(f"{a}={n}" for a, n in co["arms"].items()) + "]"
    lines = [head]
    for a, d in r["per_arm"].items():
        o = d["output_tokens"]
        if o.get("n"):
            lines.append(f"  {a}: n={o['n']} out tokens mean {o['mean']:.1f} p50 {o['p50']:.0f} p95 {o['p95']:.0f} "
                         f"max {o['max']}; truncated at cap {d['share_truncated_at_cap']:.3f}")
            ex, cap = d.get("share_exceeding_candidate_cap") or {}, d.get("max_new_tokens")
            if ex:
                lines.append(f"    share longer than N tokens (cap {cap}): "
                             + ", ".join(f"{k} {v:.3f}" for k, v in ex.items()))
            elif cap is not None:
                lines.append(f"    no candidate cap below the run cap {cap}")
    p = r.get("projection_full_request_file")
    if p:
        lines.append(f"  projection for the full request file (before caching): {p['estimated_generation_hours']:.2f} h, "
                     f"{p['estimated_gpu_hours']:.2f} GPU-h")
    return "\n".join(lines)
