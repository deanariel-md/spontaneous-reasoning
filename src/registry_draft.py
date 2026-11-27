r"""Draft Dataset 2 registry entries (config/models.yaml, dataset2.models) from a panel list.

Writing about 20 registry entries by hand means copying exact commit shas, tensor-parallel sizes and
thinking-off methods. This module drafts them for review. It never touches config/models.yaml: the
draft goes to its own file (default config/models_draft.yaml), together with a provenance JSON and a
markdown summary. The draft is reviewed and the entries copied across by hand.

Input: a YAML list of entries with `hf_id` (required) and optionally `revision` (branch, tag or 40-hex
sha; default the main branch), `family`, `tier`, `key` (default: the repository name). A plain string is
an entry with only an hf_id.

What is looked up (Hugging Face Hub metadata API, cached login, no weights; the one file fetched is
config.json, into a temporary cache that is deleted):
  revision      resolved to the exact 40-hex commit sha
  weights       safetensors parameter counts times bytes per dtype (decimal GB); the dtype breakdown
  facts         license, gated flag, config.json: architecture, torch_dtype, max_position_embeddings,
                attention heads, quantization_config

What is proposed (all of it is a proposal for review):
  tensor_parallel_size  smallest power of two t with weights * 1.2 / t <= gpu_mem_gb * 0.9 (decimal GB).
                        More GPUs than --max-tp gives PLACEHOLDER.
  max_model_len         (largest max_new_tokens cap in config/decoding.yaml + prompt budget) rounded up
                        to a multiple of 256, then capped at the model's max_position_embeddings. A model
                        maximum that is not a multiple of 256 is used as it is, so the proposal never
                        exceeds what the model supports.
  dtype                 the stored safetensors dtype when one plain dtype (BF16, F16 or F32) holds at least
                        99% of the parameters; otherwise PLACEHOLDER (quantized, FP8 or mixed storage).
  quantization          none for a plain stored dtype without a quantization_config; otherwise PLACEHOLDER.
  tokenizer_mode        auto; PLACEHOLDER when the repository carries Mistral-format files
                        (params.json or tekken.json), which the registry says to pin explicitly.
  thinking_off          from the table transcribed out of manifests/dataset2_inventory.md (INVENTORY_TABLE*
                        and METHOD_RULES below; tests compare the transcription with the document).
                        chat_template_kwargs and system_prompt_token only where the inventory documents
                        the off control, none_needed only for an inventory Table 2 row that states a
                        non-reasoning checkpoint (rationale quoted), PLACEHOLDER everywhere else.
  chat_template.sha256  null; frozen from the throughput run log.
  kernels               default when vLLM 0.30.0's model registry marks the
                        architecture as hybrid and config.json lists Mamba or linear-attention layers (or no layer
                        list), batch_invariant for any other architecture the registry supports. The registry facts
                        come from results/batch_invariance/support_20261028.json, and for architectures it lacks from
                        manifests/external/vllm030_arch_facts_20261030.json; an architecture missing in both or
                        unsupported by vLLM, a hybrid class whose config.json lists attention layers only, or an
                        unread config.json gives PLACEHOLDER.

Every drafted entry is parsed with generation.parse_model_spec against a registry loaded from the draft.
PLACEHOLDER entries are expected refusals; any other refusal is reported as a problem.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tempfile
import textwrap
import warnings
from datetime import datetime, timezone
from decimal import Decimal
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import errors as hub_errors

from src import d2_io
from src import generation as gen

ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = gen.MODELS_YAML                 # never written by this module
DECODING_YAML = gen.DECODING_YAML
DEFAULT_OUT = ROOT / "config" / "models_draft.yaml"
INVENTORY_PATH = ROOT / "manifests" / "dataset2_inventory.md"
INVENTORY_REF = "manifests/dataset2_inventory.md"
INVENTORY_DATE = "3 October 2026"
KERNEL_RECORD_PATH = ROOT / "results" / "batch_invariance" / "support_20261028.json"
KERNEL_RECORD_REF = "results/batch_invariance/support_20261028.json"
PANEL_ARCH_RECORD_PATH = ROOT / "manifests" / "external" / "vllm030_arch_facts_20261030.json"
PANEL_ARCH_RECORD_REF = "manifests/external/vllm030_arch_facts_20261030.json"
STATE_LAYERS = frozenset({"linear_attention", "mamba"})   # config.json layer_types with no batch-invariant backend

TOOL_VERSION = "1"
PLACEHOLDER = "PLACEHOLDER"
GB = 10 ** 9                                   # decimal gigabytes, as in the inventory's Weights GB column
WEIGHT_OVERHEAD = Decimal("1.2")               # weights * 1.2 must fit ...
GPU_MEM_FRACTION = Decimal("0.9")              # ... in gpu_mem_gb * 0.9 per GPU
DEFAULT_GPU_MEM_GB = 80
DEFAULT_PROMPT_BUDGET = 512
DEFAULT_MAX_TP = 8
LENGTH_MULTIPLE = 256
DTYPE_SHARE_MIN = 0.99
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_UNRESOLVED = 0, 1, 2, 3

TP_FORMULA = ("tensor_parallel_size = smallest power of two t such that weight_bytes * 1.2 / t <= "
              "gpu_mem_gb * 0.9 * 1e9 (decimal GB)")
LEN_FORMULA = ("max_model_len = min(ceil256(cot_cap + prompt_budget), max_position_embeddings), where cot_cap "
               "is the largest max_new_tokens cap in config/decoding.yaml")

_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOKEN_LIKE = re.compile(r"(hf_[A-Za-z0-9]{8,}|Bearer\s+\S+)")

# Bytes per element of each safetensors dtype name.
SAFETENSORS_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                     "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
                     "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
REGISTRY_DTYPE = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
_NESTED_CONFIG = ("text_config", "language_config", "llm_config", "lm_config")
_MISTRAL_FORMAT_FILES = ("params.json", "tekken.json")


class PanelError(ValueError):
    """The panel file is unusable."""


# ============================================================================================
# Panel file
# ============================================================================================

@dataclasses.dataclass(frozen=True)
class PanelEntry:
    hf_id: str
    revision: str | None
    family: str | None
    tier: str | None
    key: str
    key_given: bool


PANEL_FIELDS = ("hf_id", "revision", "family", "tier", "key")


def default_key(hf_id: str) -> str:
    """Repository name with characters outside the model-key alphabet replaced by '-'."""
    name = hf_id.split("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "-", name).lstrip("._-")


def parse_panel(raw: Any) -> list[PanelEntry]:
    """Validate the panel list. Every problem is reported at once."""
    if not isinstance(raw, list) or not raw:
        raise PanelError("the panel file must be a non-empty YAML list of entries")
    problems: list[str] = []
    out: list[PanelEntry] = []
    for i, item in enumerate(raw, 1):
        where = f"entry {i}"
        if isinstance(item, str):
            item = {"hf_id": item}
        if not isinstance(item, Mapping):
            problems.append(f"{where}: must be a mapping with hf_id (or a plain hf_id string)")
            continue
        extra = sorted(set(item) - set(PANEL_FIELDS))
        if extra:
            problems.append(f"{where}: unknown field(s) {extra}; allowed: {list(PANEL_FIELDS)}")
        hf_id = item.get("hf_id")
        if not isinstance(hf_id, str) or not _HF_ID.match(hf_id):
            problems.append(f"{where}: hf_id {hf_id!r} must be a string of the form org/name")
            continue
        where = f"entry {i} ({hf_id})"
        fields: dict[str, str | None] = {}
        for name in ("revision", "family", "tier", "key"):
            v = item.get(name)
            if v is None:
                fields[name] = None
            elif isinstance(v, str) and v.strip():
                fields[name] = v.strip()
            else:
                problems.append(f"{where}: {name} must be a non-empty string (quote values such as a numeric-looking sha); got {v!r}")
                fields[name] = None
        key = fields["key"] or default_key(hf_id)
        try:
            d2_io.check_model_key(key)
        except Exception:
            problems.append(f"{where}: key {key!r} must match {d2_io.MODEL_KEY_RE.pattern}; set `key` in the panel file")
            continue
        out.append(PanelEntry(hf_id=hf_id, revision=fields["revision"], family=fields["family"],
                              tier=fields["tier"], key=key, key_given=fields["key"] is not None))
    seen: dict[str, str] = {}
    for p in out:
        if p.key in seen:
            problems.append(f"key {p.key!r} is used by both {seen[p.key]} and {p.hf_id}; set `key` in the panel file")
        seen.setdefault(p.key, p.hf_id)
    if problems:
        raise PanelError("; ".join(problems))
    return out


def load_panel(path: str | Path) -> list[PanelEntry]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PanelError(f"cannot read the panel file {path}: {exc}") from exc
    return parse_panel(raw)


# ============================================================================================
# Thinking-off table, transcribed from manifests/dataset2_inventory.md
# ============================================================================================

# Table 1: (checkpoint, switch M-code, line, off-mode code, evidence class)
INVENTORY_TABLE1 = (
    ("Qwen/Qwen3-0.6B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-1.7B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-4B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-8B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-14B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-32B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3-30B-A3B", "M1", "Qwen3", "P", "C"),
    ("Qwen/Qwen3.5-0.8B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-2B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-4B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-9B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-27B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-35B-A3B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.5-122B-A10B", "M2", "Qwen3.5", "P", "C"),
    ("Qwen/Qwen3.6-27B", "M2", "Qwen3.6", "P", "C"),
    ("Qwen/Qwen3.6-35B-A3B", "M2", "Qwen3.6", "P", "C"),
    ("google/gemma-4-E2B-it", "M3", "Gemma 4", "C", "C"),
    ("google/gemma-4-E4B-it", "M3", "Gemma 4", "C", "C"),
    ("google/gemma-4-12B-it", "M3", "Gemma 4", "P", "C"),
    ("google/gemma-4-26B-A4B-it", "M3", "Gemma 4", "P", "C"),
    ("google/gemma-4-31B-it", "M3", "Gemma 4", "P", "C"),
    ("nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1", "M4", "Llama-Nemotron", "C", "C"),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "M4", "Llama-Nemotron", "C", "C"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1", "M4", "Llama-Nemotron", "C", "C"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "M5", "Llama-Nemotron", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "M5", "Nemotron Nano 2", "P", "C"),
    ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", "M5", "Nemotron Nano 2", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "M6", "Nemotron 3", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "M6", "Nemotron 3", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16", "M6", "Nemotron 3", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "M6", "Nemotron 3", "C", "C"),
    ("nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16", "M6", "Nemotron 3", "C", "C"),
    ("nvidia/Nemotron-Cascade-8B", "M5", "Nemotron-Cascade", "C", "C"),
    ("nvidia/Nemotron-Cascade-2-30B-A3B", "M6", "Nemotron-Cascade", "P", "C"),
    ("zai-org/GLM-4.5-Air", "M7", "GLM", "P", "C"),
    ("zai-org/GLM-4.7-Flash", "M7", "GLM", "P", "T"),
    ("mistralai/Mistral-Small-4-119B-2603", "M8", "Mistral Small 4", "C", "C"),
    ("HuggingFaceTB/SmolLM3-3B", "M9", "SmolLM3", "C", "C"),
    ("tencent/Hunyuan-0.5B-Instruct", "M10", "Hunyuan", "C", "C"),
    ("tencent/Hunyuan-1.8B-Instruct", "M10", "Hunyuan", "C", "C"),
    ("tencent/Hunyuan-4B-Instruct", "M10", "Hunyuan", "C", "C"),
    ("tencent/Hunyuan-7B-Instruct", "M10", "Hunyuan", "C", "C"),
    ("tencent/Hunyuan-A13B-Instruct", "M10", "Hunyuan", "C", "C"),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "M11", "Seed-OSS", "B", "C"),
    ("LGAI-EXAONE/EXAONE-4.0-1.2B", "M12", "EXAONE", "O", "C"),
    ("LGAI-EXAONE/EXAONE-4.0-32B", "M12", "EXAONE", "O", "C"),
    ("LGAI-EXAONE/EXAONE-4.0.1-32B", "M12", "EXAONE", "O", "C"),
    ("LGAI-EXAONE/EXAONE-4.5-33B", "M12", "EXAONE", "C", "C"),
    ("NousResearch/Hermes-4-14B", "M13", "Hermes 4", "O", "C"),
    ("NousResearch/Hermes-4-70B", "M13", "Hermes 4", "O", "C"),
    ("NousResearch/Hermes-4.3-36B", "M13", "Hermes 4", "O", "C"),
    ("deepcogito/cogito-v1-preview-llama-3B", "M14", "Cogito v1 preview", "C", "C"),
    ("deepcogito/cogito-v1-preview-llama-8B", "M14", "Cogito v1 preview", "C", "C"),
    ("deepcogito/cogito-v1-preview-qwen-14B", "M14", "Cogito v1 preview", "C", "C"),
    ("deepcogito/cogito-v1-preview-qwen-32B", "M14", "Cogito v1 preview", "C", "C"),
    ("deepcogito/cogito-v1-preview-llama-70B", "M14", "Cogito v1 preview", "C", "C"),
    ("deepcogito/cogito-v2-preview-llama-70B", "M14", "Cogito v2 preview", "C", "C"),
    ("deepcogito/cogito-v2-preview-llama-109B-MoE", "M14", "Cogito v2 preview", "C", "C"),
    ("openbmb/MiniCPM4.1-8B", "M15", "MiniCPM", "P", "C"),
    ("openbmb/MiniCPM5-1B", "M15", "MiniCPM", "C", "C"),
    ("internlm/internlm3-8b-instruct", "M17", "InternLM3 / Intern-S", "O", "C"),
    ("internlm/Intern-S1-mini", "M16", "InternLM3 / Intern-S", "C", "C"),
    ("internlm/Intern-S2-Preview", "M16", "InternLM3 / Intern-S", "C", "C"),
    ("ibm-granite/granite-3.2-2b-instruct", "M18", "Granite 3.x", "C", "C"),
    ("ibm-granite/granite-3.2-8b-instruct", "M18", "Granite 3.x", "C", "C"),
    ("ibm-granite/granite-3.3-2b-instruct", "M18", "Granite 3.x", "C", "C"),
    ("ibm-granite/granite-3.3-8b-instruct", "M18", "Granite 3.x", "C", "C"),
    ("deepseek-ai/DeepSeek-V4-Flash", "M19", "DeepSeek-V4", "C", "C"),
    ("sarvamai/sarvam-m", "M20", "Sarvam", "C", "C"),
    ("sarvamai/sarvam-30b", "M20", "Sarvam", "C", "T"),
    ("sarvamai/sarvam-105b", "M20", "Sarvam", "C", "T"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B", "M21", "HyperCLOVA X SEED Think", "C", "C"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", "M21", "HyperCLOVA X SEED Think", "C", "C"),
)
# Table 1b (outside the size envelope) rows whose switch cell reads "See M# in Table 1": (checkpoint, M-code)
INVENTORY_TABLE1B = (
    ("Qwen/Qwen3-235B-A22B", "M1"),
    ("Qwen/Qwen3.5-397B-A17B", "M2"),
    ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "M4"),
    ("zai-org/GLM-4.5", "M7"),
    ("mistralai/Mistral-Medium-3.5-128B", "M8"),
)
# Table 2 (instruct checkpoints without a switch): (checkpoint, direct-answer condition cell, verbatim)
INVENTORY_TABLE2 = (
    ("Qwen/Qwen2.5-0.5B-Instruct", "Yes. Non-reasoning instruct; the card has no reasoning mode."),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Yes. Same."),
    ("Qwen/Qwen2.5-3B-Instruct", "Yes. Same."),
    ("Qwen/Qwen2.5-7B-Instruct", "Yes. Same."),
    ("Qwen/Qwen2.5-14B-Instruct", "Yes. Same."),
    ("Qwen/Qwen2.5-32B-Instruct", "Yes. Same."),
    ("Qwen/Qwen2.5-72B-Instruct", "Yes. Same."),
    ("Qwen/Qwen3-4B-Instruct-2507", "Yes. Card: supports only non-thinking mode."),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Yes. Card: supports only non-thinking mode."),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "Yes. Card: supports only instruct (non-thinking) mode."),
    ("meta-llama/Llama-3.2-1B-Instruct", "Yes by model type. Gated; card unread."),
    ("meta-llama/Llama-3.2-3B-Instruct", "Yes by model type. Gated; card unread."),
    ("meta-llama/Llama-3.1-8B-Instruct", "Yes by model type. Gated; card unread."),
    ("meta-llama/Llama-3.3-70B-Instruct", "Yes by model type. Gated; card unread."),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Yes by model type. Gated; card unread."),
    ("google/gemma-3-1b-it", "Yes by model type. Gated; card unread."),
    ("google/gemma-3-4b-it", "Yes by model type. Gated; card unread."),
    ("google/gemma-3-12b-it", "Yes by model type. Gated; card unread."),
    ("google/gemma-3-27b-it", "Yes by model type. Gated; card unread."),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Yes. The card has no reasoning mode."),
    ("mistralai/Ministral-3-3B-Instruct-2512", "Yes. Separate `-Reasoning-2512` weights exist."),
    ("mistralai/Ministral-3-8B-Instruct-2512", "Yes. Separate `-Reasoning-2512` weights exist."),
    ("mistralai/Ministral-3-14B-Instruct-2512", "Yes. Separate `-Reasoning-2512` weights exist."),
    ("microsoft/phi-4", "Yes. The card has no reasoning mode."),
    ("microsoft/Phi-4-mini-instruct", "Yes. The card has no reasoning mode."),
    ("allenai/Olmo-3-7B-Instruct", "Yes. Separate `Olmo-3-7B-Think` weights exist."),
    ("allenai/Olmo-3.1-32B-Instruct", "Yes. Separate `Olmo-3.1-32B-Think` weights exist."),
    ("ibm-granite/granite-4.0-micro", "Yes. The card has no reasoning mode."),
    ("ibm-granite/granite-4.0-h-tiny", "Yes. The card has no reasoning mode."),
    ("ibm-granite/granite-4.0-h-small", "Yes. The card has no reasoning mode."),
    ("ibm-granite/granite-4.1-3b", "Yes. The card has no reasoning mode."),
    ("ibm-granite/granite-4.1-8b", "Yes. The card has no reasoning mode."),
    ("ibm-granite/granite-4.1-30b", "Yes. The card has no reasoning mode."),
    ("tiiuae/Falcon-H1-0.5B-Instruct", "Yes. The card has no reasoning mode."),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "Yes. The card has no reasoning mode."),
    ("tiiuae/Falcon-H1-3B-Instruct", "Yes. The card has no reasoning mode."),
    ("tiiuae/Falcon-H1-7B-Instruct", "Yes. The card has no reasoning mode."),
    ("tiiuae/Falcon-H1-34B-Instruct", "Yes. The card has no reasoning mode."),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "Yes. The card has no reasoning mode."),
    ("baidu/ERNIE-4.5-0.3B-PT", "Yes. The card has no reasoning mode."),
    ("baidu/ERNIE-4.5-21B-A3B-PT", "Yes. Separate `ERNIE-4.5-21B-A3B-Thinking` weights exist."),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "Yes. The card has no reasoning mode."),
    ("zai-org/GLM-4-9B-0414", "Yes. The card lists separate GLM-Z1 reasoning models."),
    ("zai-org/GLM-4-32B-0414", "Yes. The card lists separate GLM-Z1 reasoning models."),
    ("ai21labs/AI21-Jamba2-3B", "Yes. The card has no reasoning mode."),
    ("ai21labs/AI21-Jamba2-Mini", "Yes. The card describes a non-reasoning model."),
    ("swiss-ai/Apertus-8B-Instruct-2509", "Yes. The card has no reasoning mode."),
    ("swiss-ai/Apertus-70B-Instruct-2509", "Yes. The card has no reasoning mode."),
)
# Table 3 (reasoning-native), single-checkpoint rows: (checkpoint, documented route to disable reasoning)
INVENTORY_TABLE3 = (
    ("Qwen/Qwen3-4B-Thinking-2507", "Cannot be disabled by documented means. Separate Instruct-2507 weights exist."),
    ("Qwen/Qwen3-30B-A3B-Thinking-2507", "Same as above."),
    ("Qwen/Qwen3-Next-80B-A3B-Thinking", "Cannot be disabled by documented means. Separate Instruct weights exist."),
    ("Qwen/QwQ-32B", "No switch in the card."),
    ("openai/gpt-oss-20b", "No `none` level documented."),
    ("openai/gpt-oss-120b", "No `none` level documented."),
    ("tiiuae/Falcon-H1R-7B", "No switch in the card."),
    ("baidu/ERNIE-4.5-21B-A3B-Thinking", "No switch in the card; the non-thinking weights are `ERNIE-4.5-21B-A3B-PT`."),
    ("kakaocorp/kanana-2-30b-a3b-thinking-2601", "No switch found in the card."),
    ("ai21labs/AI21-Jamba-Reasoning-3B", "No switch in the card."),
    ("XiaomiMiMo/MiMo-7B-RL-0530", "Template-only flag; the card does not document it."),
)


@dataclasses.dataclass(frozen=True)
class MethodRule:
    """What the registry gets for one inventory switch (M-code), or why it gets PLACEHOLDER."""
    code: str
    mechanism: str                       # the inventory's "Exact control" text for the code, quoted
    method: str                          # chat_template_kwargs | system_prompt_token | PLACEHOLDER
    kwargs: Mapping[str, Any] | None = None
    token: str | None = None
    why_placeholder: str | None = None
    candidate: str | None = None         # suggestion for the reviewer when method is PLACEHOLDER
    alternative: str | None = None       # a second documented route, recorded for the reviewer


_OFF = {"enable_thinking": False}

METHOD_RULES: dict[str, MethodRule] = {r.code: r for r in (
    MethodRule("M1", "`enable_thinking=True` or `enable_thinking=False` passed to `apply_chat_template` "
               "(or `chat_template_kwargs` in vLLM/SGLang).", "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M2", "`enable_thinking` through `chat_template_kwargs` (or `apply_chat_template`).",
               "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M3", "Thinking is on when the `<|think|>` token starts the system prompt; the chat template exposes "
               "this as `enable_thinking` (template default false).", "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M4", "System prompt `detailed thinking on` or `detailed thinking off`; instructions go in the user turn.",
               "system_prompt_token", token="detailed thinking off"),
    MethodRule("M5", "`/think` or `/no_think` in the system prompt (Super-49B-v1.5, Nano-9B-v2, Nano-12B-v2; Nano v2 "
               "also accepts the tags in user messages).", "system_prompt_token", token="/no_think"),
    MethodRule("M6", "`enable_thinking=False` through `chat_template_kwargs` or `apply_chat_template` (default True).",
               "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M7", "`enable_thinking=False` through `chat_template_kwargs` (the GLM-4.5 card documents this for vLLM "
               "and SGLang; the template appends `/nothink` or an empty think block).",
               "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M8", "`reasoning_effort` set to `none` or `high` per request (the chat template raises an error for other "
               "values; undefined means `none`).", PLACEHOLDER,
               why_placeholder="the inventory documents reasoning_effort per request and gives no chat_template_kwargs form",
               candidate="reasoning_effort `none`; the Mistral-format tokenizer path must be settled first"),
    MethodRule("M9", "`/think` or `/no_think` in the system prompt, or the `enable_thinking` template kwarg.",
               "chat_template_kwargs", kwargs=_OFF,
               alternative="system_prompt_token '/no_think' is the other documented route"),
    MethodRule("M10", "`enable_thinking=False` in `apply_chat_template`, or `/no_think` and `/think` placed before the prompt.",
               "chat_template_kwargs", kwargs=_OFF,
               alternative="a '/no_think' prefix placed before the prompt is the other documented route"),
    MethodRule("M11", "`thinking_budget` in tokens. Default is unlimited thinking. The card states that the model is "
               "instructed to output a direct response when the budget is 0.", PLACEHOLDER,
               why_placeholder="the inventory documents a thinking budget and does not say how it reaches the template",
               candidate="chat_template_kwargs {thinking_budget: 0}"),
    MethodRule("M12", "`enable_thinking=True` in `apply_chat_template`. EXAONE 4.0 default is non-reasoning; EXAONE 4.5 "
               "default is reasoning (`enable_thinking=False` for non-reasoning).", PLACEHOLDER,
               why_placeholder="EXAONE 4.0 is non-reasoning by default and the inventory documents the flag only as "
                               "enable_thinking=True",
               candidate="chat_template_kwargs {enable_thinking: false}"),
    MethodRule("M13", "`thinking=True` chat-template flag, or a documented deep-thinking system prompt. Reasoning is opt-in.",
               PLACEHOLDER,
               why_placeholder="reasoning is opt-in and the inventory documents the flag only as thinking=True",
               candidate="chat_template_kwargs {thinking: false}"),
    MethodRule("M14", "`enable_thinking=True` in the chat template, or system prompt `Enable deep thinking subroutine.` "
               "Reasoning is opt-in.", PLACEHOLDER,
               why_placeholder="reasoning is opt-in and the inventory documents the flag only as enable_thinking=True",
               candidate="chat_template_kwargs {enable_thinking: false}"),
    MethodRule("M15", "`enable_thinking=True` or `enable_thinking=False` in `apply_chat_template`.",
               "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M16", "`enable_thinking=False` in `apply_chat_template` or `chat_template_kwargs` (thinking on by default).",
               "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M17", "A long documented thinking system prompt (math-competition persona) selects the thinking mode; the "
               "default is the normal response mode. No template flag.", PLACEHOLDER,
               why_placeholder="the documented switch is a long system prompt and the template has no flag",
               candidate="none_needed, since the default is the normal response mode (inventory off code O)"),
    MethodRule("M18", "Boolean `thinking=True` or `thinking=False` argument to `apply_chat_template` (the cards show "
               "outputs for both).", "chat_template_kwargs", kwargs={"thinking": False}),
    MethodRule("M19", "Messages are encoded with the repo's `encoding_dsv4.encode_messages` using `thinking_mode` set to "
               "`chat` or `thinking`; no Jinja template ships.", PLACEHOLDER,
               why_placeholder="no Jinja template ships, so the runner's chat-template path does not reach the thinking_mode switch",
               candidate="thinking_mode `chat` through the repository's encode_messages"),
    MethodRule("M20", "`enable_thinking` in `apply_chat_template`.", "chat_template_kwargs", kwargs=_OFF),
    MethodRule("M21", "Think-14B: assistant prefix `assistant/think` for reasoning and `assistant` for a direct answer in "
               "the ChatML prompt. Think-32B: `chat_template_kwargs` with `thinking` set to True, plus optional "
               "`thinking_token_budget`.", PLACEHOLDER,
               why_placeholder="Think-14B switches by assistant prefix, and Think-32B documents the flag only as thinking=True",
               candidate="Think-32B: chat_template_kwargs {thinking: false}; Think-14B: assistant prefix `assistant`"),
)}

# Rows of Table 1 whose method differs from their M-code's rule.
ROW_RULE_OVERRIDES: dict[str, MethodRule] = {
    "nvidia/nemotron-cascade-8b": MethodRule(
        "M5", METHOD_RULES["M5"].mechanism + " Cascade-8B appends the tag to the user input and also takes `enable_thinking`.",
        PLACEHOLDER, why_placeholder="Cascade-8B appends the tag to the user input and also takes enable_thinking, "
        "so two routes are documented and the system-prompt token does not apply",
        candidate="chat_template_kwargs {enable_thinking: false}"),
    "lgai-exaone/exaone-4.5-33b": MethodRule(
        "M12", METHOD_RULES["M12"].mechanism, "chat_template_kwargs", kwargs=_OFF),
}

NONE_NEEDED_RATIONALE = ("Inventory " + INVENTORY_REF + " (" + INVENTORY_DATE + "), Table 2, direct-answer condition: {cell}")


@dataclasses.dataclass(frozen=True)
class ThinkingOffProposal:
    method: str                                  # chat_template_kwargs | system_prompt_token | none_needed | PLACEHOLDER
    chat_template_kwargs: dict | None
    system_prompt_token: str | None
    rationale: str | None
    mcode: str | None
    table: str | None                            # "1", "1b", "2", "3" or None when the checkpoint is not an inventory row
    inventory_line: str | None
    evidence: str | None
    source: str
    reason: str | None = None                    # why the method is PLACEHOLDER
    candidate: str | None = None
    alternative: str | None = None

    def registry_block(self) -> dict:
        return {"method": self.method,
                "chat_template_kwargs": dict(self.chat_template_kwargs) if self.chat_template_kwargs else None,
                "system_prompt_token": self.system_prompt_token, "rationale": self.rationale}

    def label(self) -> str:
        if self.method == "chat_template_kwargs":
            return "chat_template_kwargs " + _flow(self.chat_template_kwargs)
        if self.method == "system_prompt_token":
            return f"system_prompt_token {self.system_prompt_token!r}"
        return self.method


def _resolve_same(rows: Sequence[tuple[str, str]], marker: str) -> dict[str, tuple[str, str]]:
    """checkpoint -> (cell as printed, resolved cell). A cell that reads 'Same ...' repeats the row above it."""
    out: dict[str, tuple[str, str]] = {}
    last = ""
    for ckpt, cell in rows:
        resolved = last if marker in cell else cell
        if marker not in cell:
            last = cell
        out[ckpt.lower()] = (cell, resolved)
    return out


_T1 = {r[0].lower(): r for r in INVENTORY_TABLE1}
_T1B = {r[0].lower(): r for r in INVENTORY_TABLE1B}
_T2 = _resolve_same(INVENTORY_TABLE2, "Same.")
_T3 = _resolve_same(INVENTORY_TABLE3, "Same as above.")


def _placeholder(source: str, reason: str, **kw) -> ThinkingOffProposal:
    return ThinkingOffProposal(PLACEHOLDER, None, None, None, source=source, reason=reason,
                               **{"mcode": None, "table": None, "inventory_line": None, "evidence": None, **kw})


def _from_rule(rule: MethodRule, table: str, hf_id: str, line: str | None, evidence: str | None) -> ThinkingOffProposal:
    source = f"{INVENTORY_REF} Table {table}, row `{hf_id}`, {rule.code}: {rule.mechanism}"
    common = dict(mcode=rule.code, table=table, inventory_line=line, evidence=evidence, source=source,
                  alternative=rule.alternative)
    if rule.method == "chat_template_kwargs":
        return ThinkingOffProposal("chat_template_kwargs", dict(rule.kwargs or {}), None, None, **common)
    if rule.method == "system_prompt_token":
        return ThinkingOffProposal("system_prompt_token", None, rule.token, None, **common)
    return ThinkingOffProposal(PLACEHOLDER, None, None, None, reason=rule.why_placeholder, candidate=rule.candidate, **common)


def thinking_off_for(hf_id: str) -> ThinkingOffProposal:
    """Thinking-off registry block for a checkpoint, from the inventory transcription."""
    k = hf_id.lower()
    if k in _T1:
        _, code, line, _off, evid = _T1[k]
        return _from_rule(ROW_RULE_OVERRIDES.get(k, METHOD_RULES[code]), "1", hf_id, line, evid)
    if k in _T1B:
        return _from_rule(ROW_RULE_OVERRIDES.get(k, METHOD_RULES[_T1B[k][1]]), "1b", hf_id, None, None)
    if k in _T2:
        cell, resolved = _T2[k]
        source = f"{INVENTORY_REF} Table 2, row `{hf_id}`: direct-answer condition \"{cell}\""
        if cell.startswith("Yes by model type"):
            return _placeholder(source, "the inventory lists this checkpoint as a non-reasoning instruct model by model "
                                        "type only (gated, card unread)",
                                candidate="none_needed once the card is read and confirms no reasoning mode",
                                table="2")
        quoted = f'"{cell}"' if cell == resolved else f'"{cell}" (the row above reads "{resolved}")'
        return ThinkingOffProposal("none_needed", None, None, NONE_NEEDED_RATIONALE.format(cell=quoted),
                                   mcode=None, table="2", inventory_line=None, evidence=None, source=source)
    if k in _T3:
        cell, resolved = _T3[k]
        return _placeholder(f"{INVENTORY_REF} Table 3, row `{hf_id}`: documented route to disable reasoning \"{resolved}\"",
                            "reasoning-native checkpoint; the inventory lists no documented route to disable reasoning, so "
                            "it has no direct-answer condition (excluded by rule)", table="3")
    return _placeholder(f"{INVENTORY_REF}: no row for `{hf_id}` in Tables 1, 1b, 2 or 3",
                        "the checkpoint is not an inventory row, so no thinking-off method is documented here")


_ROW_SPLIT = re.compile(r"(?<!\\)\|")


def parse_inventory_tables(text: str) -> dict[str, list[tuple]]:
    """Read the checkpoint rows out of manifests/dataset2_inventory.md (used by the tests to check the transcription).

    Returns {"1": [(checkpoint, M-code, line, off code, evidence)], "1b": [(checkpoint, M-code or None)],
    "2": [(checkpoint, direct-answer cell)], "3": [(checkpoint, route cell)]}.
    """
    def cells(line: str) -> list[str]:
        return [c.strip().replace("\\|", "|") for c in _ROW_SPLIT.split(line.strip())[1:-1]]

    def ident(cell: str) -> str | None:
        m = re.fullmatch(r"`([^`]+)`", cell)
        return m.group(1) if m else None

    def section(start: str, end: str | None) -> list[str]:
        i = text.index(start)
        j = text.index(end, i + 1) if end else len(text)
        return [ln for ln in text[i:j].splitlines() if ln.startswith("|")]

    out: dict[str, list[tuple]] = {"1": [], "1b": [], "2": [], "3": []}
    for ln in section("## Table 1. ", "## Table 1b"):
        c = cells(ln)
        if len(c) == 13 and ident(c[2]):
            out["1"].append((ident(c[2]), c[6], c[1], c[8], c[9]))
    for ln in section("## Table 1b", "## Table 2"):
        c = cells(ln)
        if len(c) == 8 and ident(c[0]):
            m = re.match(r"See (M\d+) in Table 1\.", c[4])
            out["1b"].append((ident(c[0]), m.group(1) if m else None))
    for ln in section("## Table 2", "## Table 3"):
        c = cells(ln)
        if len(c) == 8 and ident(c[1]):
            out["2"].append((ident(c[1]), c[5]))
    for ln in section("## Table 3", "## Count against"):
        c = cells(ln)
        if len(c) == 7 and ident(c[1]) and not re.search(r"[,( ]", ident(c[1])):
            out["3"].append((ident(c[1]), c[4]))
    return out


# ============================================================================================
# Arithmetic
# ============================================================================================

def propose_tensor_parallel(weight_bytes: int, gpu_mem_gb: float | int | str, max_tp: int = DEFAULT_MAX_TP) -> dict:
    """Smallest power of two t with weight_bytes * 1.2 / t <= gpu_mem_gb * 0.9 * 1e9. Equality fits.

    Exact decimal arithmetic. `within_max_tp` is False when t exceeds max_tp.
    """
    if isinstance(weight_bytes, bool) or not isinstance(weight_bytes, int) or weight_bytes <= 0:
        raise ValueError(f"weight_bytes must be a positive integer, got {weight_bytes!r}")
    mem = Decimal(str(gpu_mem_gb))
    if not mem.is_finite() or mem <= 0:
        raise ValueError(f"gpu_mem_gb must be positive, got {gpu_mem_gb!r}")
    required = Decimal(weight_bytes) * WEIGHT_OVERHEAD
    budget = mem * GPU_MEM_FRACTION * GB
    tp = 1
    while required > budget * tp:
        tp *= 2
    return {"tensor_parallel_size": tp, "weight_bytes": weight_bytes, "required_bytes_total": float(required),
            "per_gpu_budget_bytes": float(budget), "required_bytes_per_gpu": float(required / tp),
            "gpu_mem_gb": float(mem), "overhead": float(WEIGHT_OVERHEAD), "mem_fraction": float(GPU_MEM_FRACTION),
            "max_tp": max_tp, "within_max_tp": tp <= max_tp, "formula": TP_FORMULA}


def _ceil_multiple(n: int, m: int) -> int:
    return -(-n // m) * m


def propose_max_model_len(cot_cap: int, prompt_budget: int, model_max: int | None, multiple: int = LENGTH_MULTIPLE) -> dict:
    """(cot_cap + prompt_budget) rounded up to a multiple of 256, capped at the model's maximum.

    The cap is applied after the rounding so the value never exceeds what the model supports; a model
    maximum that is not a multiple of 256 is therefore used as it is.
    """
    for name, v in (("cot_cap", cot_cap), ("prompt_budget", prompt_budget)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v!r}")
    if cot_cap + prompt_budget < 1:
        raise ValueError("cot_cap + prompt_budget must be positive")
    raw = cot_cap + prompt_budget
    rounded = _ceil_multiple(raw, multiple)
    known = isinstance(model_max, int) and not isinstance(model_max, bool) and model_max > 0
    final = min(rounded, model_max) if known else rounded
    return {"max_model_len": final, "cot_cap": cot_cap, "prompt_budget": prompt_budget, "sum": raw,
            "rounded_to_multiple": rounded, "multiple": multiple, "model_max": model_max if known else None,
            "capped_at_model_max": bool(known and model_max < rounded),
            "model_max_below_sum": bool(known and model_max < raw), "formula": LEN_FORMULA}


def tp_shape_warnings(tp: int, heads: Any, kv_heads: Any) -> list[str]:
    """vLLM needs the attention heads split evenly across the tensor-parallel GPUs."""
    out: list[str] = []
    if tp > 1 and isinstance(heads, int) and heads % tp:
        out.append(f"num_attention_heads {heads} is not divisible by tensor_parallel_size {tp}")
    if tp > 1 and isinstance(kv_heads, int) and kv_heads > 0:
        ok = kv_heads % tp == 0 if kv_heads >= tp else tp % kv_heads == 0
        if not ok:
            out.append(f"num_key_value_heads {kv_heads} does not split across tensor_parallel_size {tp}")
    return out


def weight_bytes_from_parameters(parameters: Mapping[str, int]) -> tuple[int | None, list[str]]:
    """Bytes of the stored tensors from per-dtype parameter counts; (None, unknown dtypes) when a dtype is unknown."""
    total, unknown = 0, []
    for dt, n in parameters.items():
        width = SAFETENSORS_BYTES.get(str(dt).upper())
        if width is None:
            unknown.append(str(dt))
        else:
            total += int(n) * width
    return (None, unknown) if unknown else (total, [])


def propose_dtype(parameters: Mapping[str, int]) -> dict:
    """Registry dtype from the stored dtype shares; None when no single plain dtype holds 99% of the parameters."""
    total = sum(int(n) for n in parameters.values())
    shares = {str(dt): (int(n) / total if total else 0.0) for dt, n in parameters.items()}
    plain = [(s, dt) for dt, s in shares.items() if dt.upper() in REGISTRY_DTYPE and s >= DTYPE_SHARE_MIN]
    dtype = REGISTRY_DTYPE[max(plain)[1].upper()] if plain else None
    return {"dtype": dtype, "shares": shares, "rule": f"one of BF16, F16, F32 holding at least {DTYPE_SHARE_MIN:.0%} of the parameters"}


# ============================================================================================
# Hub access
# ============================================================================================

def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def clean_error(exc: BaseException | str, limit: int = 300) -> str:
    """One-line error text with anything token-shaped removed."""
    s = " ".join(str(exc).split())
    s = _TOKEN_LIKE.sub("[redacted]", s)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def classify_hub_error(exc: BaseException, hf_id: str, ref: str) -> str:
    detail = clean_error(exc)
    if isinstance(exc, hub_errors.GatedRepoError):
        return (f"gated repository {hf_id}: the metadata is not accessible with the cached login "
                f"(accept the license on the model page and check `hf auth whoami`, then rerun); {detail}")
    if isinstance(exc, hub_errors.RevisionNotFoundError):
        return f"revision {ref!r} of {hf_id} was not found; {detail}"
    if isinstance(exc, hub_errors.RepositoryNotFoundError):
        return f"repository {hf_id} was not found, or it is private and the cached login has no access; {detail}"
    if isinstance(exc, hub_errors.HfHubHTTPError):
        code = _attr(_attr(exc, "response"), "status_code")
        return f"Hub HTTP error{f' {code}' if code else ''} for {hf_id}: {detail}"
    return f"{type(exc).__name__} while reading the Hub metadata of {hf_id}: {detail}"


def find_in_config(cfg: Any, key: str) -> tuple[Any, str | None]:
    """Value of `key` at the top level of config.json, else inside a nested text/language config."""
    if isinstance(cfg, Mapping):
        if cfg.get(key) is not None:
            return cfg[key], key
        for sub in _NESTED_CONFIG:
            if isinstance(cfg.get(sub), Mapping) and cfg[sub].get(key) is not None:
                return cfg[sub][key], f"{sub}.{key}"
    return None, None


def config_facts(cfg: Mapping[str, Any]) -> dict:
    facts: dict[str, Any] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type")}
    for name, keys in (("torch_dtype", ("torch_dtype", "dtype")), ("max_position_embeddings", ("max_position_embeddings",)),
                       ("num_attention_heads", ("num_attention_heads",)), ("num_key_value_heads", ("num_key_value_heads",))):
        value, source = None, None
        for key in keys:
            value, source = find_in_config(cfg, key)
            if value is not None:
                break
        facts[name] = value
        facts[name + "_source"] = source
    qc = cfg.get("quantization_config")
    facts["quantization_config"] = (dict(qc) if isinstance(qc, Mapping) else qc) if qc else None
    layers, source = find_in_config(cfg, "layer_types")
    facts["layer_types"] = sorted({str(x) for x in layers}) if isinstance(layers, list) and layers else None
    facts["layer_types_source"] = source if facts["layer_types"] else None
    return facts


def load_kernel_record(path: str | Path | None = None) -> dict[str, dict]:
    """vLLM 0.30.0's model-registry facts per architecture (empty when unreadable). Without a path: the record in
    KERNEL_RECORD_PATH, with the architectures it lacks taken from the panel selection's record
    (PANEL_ARCH_RECORD_PATH, same vLLM version and fields); each entry names its record."""
    if path is not None:
        return _read_arch_record(Path(path), KERNEL_RECORD_REF if Path(path) == KERNEL_RECORD_PATH else str(path))
    out = _read_arch_record(PANEL_ARCH_RECORD_PATH, PANEL_ARCH_RECORD_REF)
    out.update(_read_arch_record(KERNEL_RECORD_PATH, KERNEL_RECORD_REF))
    return out


def _read_arch_record(path: Path, ref: str) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, Mapping) or data.get("vllm") not in (None, "0.30.0"):
        return {}
    archs = data.get("architectures", data.get("archs"))
    if not isinstance(archs, Mapping):
        return {}
    return {a: {**info, "record": ref} for a, info in archs.items() if isinstance(info, Mapping)}


def propose_kernels(cfg_facts: Mapping[str, Any], archs: Mapping[str, Mapping]) -> dict:
    """The kernel mode by the kernel rule, from config.json and the recorded vLLM registry facts. Returns
    {"kernels", "architecture", "rule" or "reason"}; reason is set when the mode is PLACEHOLDER."""
    names = cfg_facts.get("architectures")
    arch = names[0] if isinstance(names, list) and names else None
    info = archs.get(arch) if arch else None
    out: dict[str, Any] = {"kernels": PLACEHOLDER, "architecture": arch,
                           "record": (info or {}).get("record", KERNEL_RECORD_REF)}
    layers = set(cfg_facts.get("layer_types") or ())
    if not cfg_facts:
        out["reason"] = "config.json was not read, so the architecture is unknown"
    elif arch is None:
        out["reason"] = "config.json names no architecture"
    elif info is None:
        out["reason"] = (f"{arch} is not in {KERNEL_RECORD_REF} or {PANEL_ARCH_RECORD_REF}; record vLLM 0.30.0's "
                         "registry facts for it with "
                         "scripts/batch_invariance/inspect_arch.py")
    elif not info.get("supported"):
        out["reason"] = f"{arch} is not in vLLM 0.30.0's model registry"
    elif info.get("is_hybrid") and layers and not layers & STATE_LAYERS:
        out["reason"] = (f"vLLM 0.30.0 treats {arch} as hybrid, but config.json lists attention layers only "
                         f"({', '.join(sorted(layers))}); whether vLLM accepts batch-invariant mode shows at load")
    elif info.get("is_hybrid"):
        out.update(kernels="default", rule="vLLM 0.30.0 refuses batch-invariant mode for Mamba and linear-attention layers")
    else:
        out.update(kernels="batch_invariant", rule=f"vLLM 0.30.0 supports {arch} and does not mark it as hybrid")
    return out


def fetch_config(download: Callable[..., str], hf_id: str, sha: str) -> tuple[dict | None, str | None]:
    """config.json of the pinned revision, fetched into a temporary cache that is deleted afterwards."""
    with tempfile.TemporaryDirectory(prefix="registry_draft_", ignore_cleanup_errors=True) as tmp:
        try:
            with warnings.catch_warnings():               # Windows without symlink support warns once per cache directory
                warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*", category=UserWarning)
                path = download(repo_id=hf_id, filename="config.json", revision=sha, cache_dir=tmp)
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:                          # any failure leaves the config facts empty
            return None, clean_error(exc)
    if not isinstance(cfg, dict):
        return None, "config.json is not a JSON object"
    return cfg, None


def _license(info: Any) -> tuple[str | None, str | None]:
    card = _attr(info, "card_data")
    lic = _attr(card, "license")
    if isinstance(lic, (list, tuple)):
        lic = ", ".join(str(x) for x in lic)
    if not lic:
        lic = next((t.split(":", 1)[1] for t in (_attr(info, "tags") or []) if isinstance(t, str) and t.startswith("license:")), None)
    name = _attr(card, "license_name")
    return (str(lic) if lic else None), (str(name) if name else None)


def _iso(v: Any) -> str | None:
    return v.isoformat() if hasattr(v, "isoformat") else (str(v) if v else None)


# ============================================================================================
# Drafting one model
# ============================================================================================

@dataclasses.dataclass
class DraftContext:
    api: Any
    download: Callable[..., str]
    gpu_mem_gb: float | int
    max_tp: int
    prompt_budget: int
    cot_cap: int
    skip_config: bool
    now: Callable[[], str]
    kernel_archs: Mapping[str, Mapping] | None = None     # None: read KERNEL_RECORD_PATH


@dataclasses.dataclass
class ModelDraft:
    key: str
    hf_id: str
    panel: dict
    entry: dict | None                    # None: not drafted (errors)
    facts: dict
    errors: list[str]
    warnings: list[str]
    todo: list[str]                       # fields left as PLACEHOLDER, with the reason

    @property
    def drafted(self) -> bool:
        return self.entry is not None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_gb(n: int | None) -> str:
    return "unknown" if n is None else f"{n / GB:.1f}"


def draft_model(p: PanelEntry, ctx: DraftContext) -> ModelDraft:
    panel = {"hf_id": p.hf_id, "revision": p.revision, "family": p.family, "tier": p.tier, "key": p.key}
    ref = p.revision or "main"
    facts: dict[str, Any] = {"panel": panel, "ref_requested": p.revision, "ref_queried": ref, "resolved_at": ctx.now()}
    warnings: list[str] = []
    todo: list[str] = []

    def failed(msg: str) -> ModelDraft:
        return ModelDraft(p.key, p.hf_id, panel, None, facts, [msg], warnings, todo)

    # --- revision and metadata
    try:
        info = ctx.api.model_info(p.hf_id, revision=ref, files_metadata=True)
    except Exception as exc:                              # one model failing does not stop the others
        return failed(classify_hub_error(exc, p.hf_id, ref))
    sha = _attr(info, "sha")
    if not isinstance(sha, str) or not _SHA40.match(sha):
        return failed(f"the Hub returned no 40-hex commit sha for {p.hf_id}@{ref} (got {sha!r})")
    sha = sha.lower()
    if p.revision and _SHA40.match(p.revision) and sha != p.revision.lower():
        return failed(f"requested sha {p.revision} resolved to a different sha {sha}")
    repo_id = _attr(info, "id")
    if isinstance(repo_id, str) and repo_id.lower() != p.hf_id.lower():
        warnings.append(f"the Hub reports the repository id {repo_id!r} for {p.hf_id!r} (renamed or redirected)")
    if _attr(info, "disabled"):
        warnings.append("the repository is disabled on the Hub")
    gated = _attr(info, "gated")
    lic, lic_name = _license(info)
    siblings = list(_attr(info, "siblings") or [])
    sizes = {str(_attr(s, "rfilename")): _attr(s, "size") for s in siblings}
    st_files = {n: s for n, s in sizes.items() if n.endswith(".safetensors")}
    st_file_bytes = sum(s for s in st_files.values() if isinstance(s, int)) or None
    names = set(sizes)
    facts["revision"] = {"sha": sha, "short": sha[:12], "was_sha_given": bool(p.revision and _SHA40.match(p.revision))}
    facts["hub"] = {"repo_id": repo_id, "gated": gated if gated is not None else False, "private": bool(_attr(info, "private")),
                    "license": lic, "license_name": lic_name, "created_at": _iso(_attr(info, "created_at")),
                    "last_modified": _iso(_attr(info, "last_modified")), "n_files": len(siblings),
                    "n_safetensors_files": len(st_files), "safetensors_file_bytes": st_file_bytes}
    if gated:
        warnings.append(f"gated ({gated}): the license must be accepted on the Hub before the runner can download it")
    if not siblings:
        warnings.append("the Hub returned no file listing (gated or empty); file-based checks are skipped")

    # --- weights and dtype
    st = _attr(info, "safetensors")
    params = _attr(st, "parameters")
    params = {str(k): int(v) for k, v in params.items()} if isinstance(params, Mapping) and params else {}
    weight_bytes: int | None = None
    weight_source = None
    dtype_prop: dict[str, Any] = {"dtype": None, "shares": {}, "rule": None}
    if params:
        weight_bytes, unknown = weight_bytes_from_parameters(params)
        weight_source = "safetensors parameter counts x bytes per dtype"
        dtype_prop = propose_dtype(params)
        if weight_bytes is None:
            warnings.append(f"safetensors dtype(s) {unknown} have no known width; weight bytes taken from the file sizes")
    if weight_bytes is None and st_file_bytes:
        weight_bytes, weight_source = st_file_bytes, "sum of the .safetensors file sizes"
    elif weight_bytes is not None and st_file_bytes and abs(st_file_bytes - weight_bytes) > 0.1 * weight_bytes:
        warnings.append(f"safetensors file sizes ({_fmt_gb(st_file_bytes)} GB) differ from the parameter-count estimate "
                        f"({_fmt_gb(weight_bytes)} GB) by more than 10%")
    facts["weights"] = {"bytes": weight_bytes, "gb": None if weight_bytes is None else weight_bytes / GB, "source": weight_source,
                        "safetensors_parameters": params or None, "dtype_shares": dtype_prop["shares"] or None}

    # --- config.json
    cfg_facts: dict[str, Any] = {}
    if ctx.skip_config:
        warnings.append("config.json was not fetched (--skip-config); max_position_embeddings is unknown")
    else:
        cfg, err = fetch_config(ctx.download, p.hf_id, sha)
        if cfg is None:
            warnings.append(f"config.json was not readable ({err}); max_position_embeddings is unknown and max_model_len is not capped")
        else:
            cfg_facts = config_facts(cfg)
    facts["config"] = cfg_facts or None
    chat_cfg = _attr(_attr(info, "config"), "tokenizer_config")
    has_template = (bool(_attr(chat_cfg, "chat_template")) or bool(names & {"chat_template.jinja", "chat_template.json"})
                    if (chat_cfg is not None or siblings) else None)
    facts["hub"]["chat_template_present"] = has_template
    if has_template is False:
        warnings.append("no chat template found in the tokenizer config or the file listing")

    # --- dtype and quantization
    dtype = dtype_prop["dtype"]
    qc = cfg_facts.get("quantization_config")
    if not params:
        todo.append("dtype and quantization: the Hub reports no safetensors parameter counts")
    elif dtype is None:
        desc = ", ".join(f"{k} {v:.1%}" for k, v in sorted(dtype_prop["shares"].items(), key=lambda kv: -kv[1]))
        todo.append(f"dtype and quantization: stored dtypes are quantized or mixed ({desc}); no single plain dtype holds "
                    f"{DTYPE_SHARE_MIN:.0%} of the parameters")
    if qc:
        method = qc.get("quant_method") if isinstance(qc, Mapping) else None
        todo.append(f"quantization: config.json has a quantization_config (quant_method {method!r})")
    dtype_out = dtype or PLACEHOLDER
    quant_out = "none" if (dtype and not qc) else PLACEHOLDER
    if dtype == "float32":
        warnings.append("the weights are stored as float32; serving them as bfloat16 would halve the memory (a decision for the user)")
    cfg_dtype = cfg_facts.get("torch_dtype")
    if dtype and isinstance(cfg_dtype, str) and cfg_dtype.replace("torch.", "") != dtype:
        warnings.append(f"config.json torch_dtype {cfg_dtype!r} differs from the stored dtype {dtype!r}; the draft follows the stored tensors")

    # --- tensor parallel size
    tp_out: Any = PLACEHOLDER
    tp_facts: dict[str, Any] | None = None
    if weight_bytes:
        tp_facts = propose_tensor_parallel(weight_bytes, ctx.gpu_mem_gb, ctx.max_tp)
        tp = tp_facts["tensor_parallel_size"]
        if tp_facts["within_max_tp"]:
            tp_out = tp
            warnings.extend(tp_shape_warnings(tp, cfg_facts.get("num_attention_heads"), cfg_facts.get("num_key_value_heads")))
        else:
            todo.append(f"tensor_parallel_size: the arithmetic needs {tp} GPUs, above --max-tp {ctx.max_tp}")
    else:
        todo.append("tensor_parallel_size: the weight size is unknown")
    facts["tensor_parallel"] = tp_facts

    # --- context length
    len_facts = propose_max_model_len(ctx.cot_cap, ctx.prompt_budget, cfg_facts.get("max_position_embeddings"))
    if len_facts["model_max"] is None:
        warnings.append("the model's maximum context is unknown, so max_model_len is not capped at it")
    elif len_facts["model_max_below_sum"]:
        warnings.append(f"max_position_embeddings {len_facts['model_max']} is below cot cap + prompt budget "
                        f"({len_facts['sum']}); the output cap cannot be fully used")
    facts["max_model_len"] = len_facts

    # --- tokenizer mode
    tokenizer_mode = "auto"
    mistral_files = sorted(names & set(_MISTRAL_FORMAT_FILES))
    if mistral_files:
        tokenizer_mode = PLACEHOLDER
        todo.append(f"tokenizer_mode: Mistral-format file(s) {mistral_files} are present; pin it explicitly")
    facts["tokenizer_files"] = mistral_files

    # --- thinking-off
    off = thinking_off_for(p.hf_id)
    facts["thinking_off"] = {"method": off.method, "chat_template_kwargs": off.chat_template_kwargs,
                             "system_prompt_token": off.system_prompt_token, "rationale": off.rationale, "m_code": off.mcode,
                             "inventory_table": off.table, "inventory_line": off.inventory_line, "evidence_class": off.evidence,
                             "source": off.source, "reason_placeholder": off.reason, "candidate": off.candidate,
                             "alternative": off.alternative}
    if off.method == PLACEHOLDER:
        todo.append(f"thinking_off.method: {off.reason}" + (f". Candidate: {off.candidate}" if off.candidate else ""))
    if off.evidence == "T":
        warnings.append("inventory evidence class T: the off control is visible only in the chat template or sample code")
    if off.table == "1b":
        warnings.append("inventory Table 1b: outside the size envelope")
    if off.method == "system_prompt_token":
        warnings.append("system_prompt_token is appended to an existing system message by the runner (generation.effective_messages)")

    # --- kernel mode
    kern = propose_kernels(cfg_facts, ctx.kernel_archs if ctx.kernel_archs is not None else load_kernel_record())
    facts["kernels"] = kern
    if kern["kernels"] == PLACEHOLDER:
        todo.append(f"kernels: {kern['reason']}")
    elif kern["kernels"] == "batch_invariant" and qc:
        warnings.append("quantized weights: vLLM 0.30.0 refuses FP8 Marlin kernels in batch-invariant mode, and FP8 was "
                        "checked with Ministral-3-8B only; the throughput tranche shows whether the model loads")

    # --- family
    if p.family:
        family = p.family
    else:
        family = PLACEHOLDER
        hint = f" (the inventory line is {off.inventory_line!r})" if off.inventory_line else ""
        todo.append(f"family: not set in the panel file{hint}; scripts/run_dataset2.py groups models by this value")
    facts["family"] = {"value": family, "from_panel": bool(p.family), "inventory_line": off.inventory_line}

    entry: dict[str, Any] = {"family": family}
    if p.tier:
        entry["tier"] = p.tier
    entry.update({
        "hf_id": p.hf_id, "revision": sha, "dtype": dtype_out, "quantization": quant_out,
        "tokenizer_mode": tokenizer_mode, "max_model_len": len_facts["max_model_len"], "tensor_parallel_size": tp_out,
        "chat_template": {"source": "tokenizer_default", "file": None, "sha256": None},
        "thinking_off": off.registry_block(), "kernels": kern["kernels"]})
    return ModelDraft(p.key, p.hf_id, panel, entry, facts, [], warnings, todo)


def draft_registry(panel: Sequence[PanelEntry], ctx: DraftContext) -> list[ModelDraft]:
    return [draft_model(p, ctx) for p in panel]


# ============================================================================================
# Output
# ============================================================================================

def _flow(v: Any) -> str:
    """Inline YAML for a scalar or small mapping."""
    s = yaml.safe_dump(v, default_flow_style=True, width=10 ** 6, allow_unicode=True).strip()
    return s[:-4].strip() if s.endswith("\n...") else s


def placeholder_paths(entry: Mapping[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in entry.items():
        path = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.extend(placeholder_paths(v, path + "."))
        elif isinstance(v, str) and v.strip().upper().startswith(PLACEHOLDER):
            out.append(path)
    return out


def _comment(lines: Sequence[str], indent: int, width: int = 118) -> list[str]:
    pad = " " * indent
    out: list[str] = []
    for ln in lines:
        out.extend(textwrap.wrap(ln, width=width - indent - 2, subsequent_indent="  ", break_long_words=False,
                                 break_on_hyphens=False) or [""])
    return [f"{pad}# {ln}" if ln else f"{pad}#" for ln in out]


def entry_comment_lines(d: ModelDraft) -> list[str]:
    f = d.facts
    w, tp, ln = f["weights"], f["tensor_parallel"], f["max_model_len"]
    hub = f["hub"]
    lines = [f"{d.hf_id}: {f['ref_queried']} resolved to {f['revision']['sha']} at {f['resolved_at']}",
             f"license {hub['license'] or 'unknown'}; gated {hub['gated']}",
             f"weights {_fmt_gb(w['bytes'])} GB ({w['source'] or 'unknown'})"]
    if tp:
        lines.append(f"tensor_parallel_size {tp['tensor_parallel_size']}: {tp['weight_bytes']} B * 1.2 / TP = "
                     f"{tp['required_bytes_per_gpu'] / GB:.1f} GB per GPU against {tp['per_gpu_budget_bytes'] / GB:.1f} GB "
                     f"({tp['gpu_mem_gb']:g} GB * 0.9)")
    lines.append(f"max_model_len {ln['max_model_len']}: cot cap {ln['cot_cap']} + prompt budget {ln['prompt_budget']} = {ln['sum']}, "
                 f"rounded up to a multiple of {ln['multiple']} = {ln['rounded_to_multiple']}, "
                 f"model maximum {ln['model_max'] if ln['model_max'] else 'unknown'}")
    lines.append(f"thinking_off: {f['thinking_off']['source']}")
    if f.get("kernels", {}).get("rule"):
        lines.append(f"kernels {f['kernels']['kernels']}: {f['kernels']['rule']} ({f['kernels']['record']})")
    lines.extend(f"TODO {t}" for t in d.todo)
    lines.extend(f"NOTE {t}" for t in d.warnings)
    return lines


def render_entry_yaml(d: ModelDraft) -> str:
    e = d.entry
    assert e is not None
    ind = "      "
    out = _comment(entry_comment_lines(d), 4)
    out.append(f"    {_flow(d.key)}:")
    out.append(f"{ind}family: {_flow(e['family'])}")
    if "tier" in e:
        out.append(f"{ind}tier: {_flow(e['tier'])}")
    for k in ("hf_id", "revision", "dtype", "quantization", "tokenizer_mode", "max_model_len", "tensor_parallel_size"):
        out.append(f"{ind}{k}: {_flow(e[k])}")
    ct, to = e["chat_template"], e["thinking_off"]
    out.append(f"{ind}chat_template:")
    out.extend(f"{ind}  {k}: {_flow(ct[k])}" for k in ("source", "file", "sha256"))
    out.append(f"{ind}thinking_off:")
    out.extend(f"{ind}  {k}: {_flow(to[k])}" for k in ("method", "chat_template_kwargs", "system_prompt_token", "rationale"))
    out.append(f"{ind}kernels: {_flow(e['kernels'])}")
    return "\n".join(out)


def render_registry_yaml(drafts: Sequence[ModelDraft], generated_at: str) -> str:
    head = [f"# DRAFT registry entries written by scripts/draft_registry.py (src/registry_draft.py) at {generated_at}.",
            "# The runner does not read this file. Review every entry, resolve each PLACEHOLDER, then copy the entries",
            "# into config/models.yaml under dataset2.models. chat_template.sha256 stays null until the throughput",
            "# run log supplies the observed template hash. Facts behind each entry: the provenance JSON beside this file.",
            "dataset2:", "  registry_schema_version: 1"]
    done = [d for d in drafts if d.drafted]
    if not done:
        return "\n".join(head + ["  models: {}"]) + "\n"
    return "\n".join(head + ["  models:"] + [render_entry_yaml(d) for d in done]) + "\n"


_FILL_VALUES = {"revision": "0" * 40, "dtype": "bfloat16", "quantization": "none", "tokenizer_mode": "auto",
                "max_model_len": 1024, "tensor_parallel_size": 1, "kernels": "batch_invariant"}


def _is_placeholder_value(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper().startswith(PLACEHOLDER)


def _fill_placeholders(entry: Mapping[str, Any]) -> dict:
    """Copy of an entry with every PLACEHOLDER in a runner-required field replaced by a valid dummy value."""
    e = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in entry.items()}
    for k, v in _FILL_VALUES.items():
        if _is_placeholder_value(e.get(k)):
            e[k] = v
    to = e.get("thinking_off")
    if isinstance(to, dict) and _is_placeholder_value(to.get("method")):
        to.update(method="chat_template_kwargs", chat_template_kwargs={"enable_thinking": False})
    return e


def validate_registry_file(path: str | Path, drafts: Sequence[ModelDraft]) -> dict[str, dict]:
    """Parse every drafted entry with generation.parse_model_spec against the registry loaded from the draft file.

    status: accepted | refused. A refusal is expected when it comes only from PLACEHOLDER fields: the check
    refills every placeholder with a valid dummy value and parses the entry again, and any reason that remains
    is listed under `other_reasons` and reported as a defect in the draft (family is optional for the runner).
    """
    registry = gen.load_registry(path)
    out: dict[str, dict] = {}
    for d in drafts:
        if not d.drafted:
            continue
        entry = registry.get(d.key) or {}
        pending = [p for p in placeholder_paths(entry) if p != "family"]
        try:
            spec = gen.parse_model_spec(d.key, registry)
            out[d.key] = {"status": "accepted", "reasons": None, "expected": True, "other_reasons": None,
                          "placeholders": pending, "spec_tensor_parallel_size": spec.tensor_parallel_size,
                          "spec_max_model_len": spec.max_model_len}
        except gen.RunRefused as exc:
            other = None
            try:
                gen.parse_model_spec(d.key, {d.key: _fill_placeholders(entry)})
            except gen.RunRefused as exc2:
                other = str(exc2).split("cannot be run: ", 1)[-1]
            out[d.key] = {"status": "refused", "reasons": str(exc).split("cannot be run: ", 1)[-1],
                          "expected": other is None, "other_reasons": other, "placeholders": pending}
    return out


def _method_label(d: ModelDraft) -> str:
    to = d.entry["thinking_off"]
    if to["method"] == "chat_template_kwargs":
        return "chat_template_kwargs " + _flow(to["chat_template_kwargs"])
    if to["method"] == "system_prompt_token":
        return f"system_prompt_token {to['system_prompt_token']!r}"
    return to["method"]


def summary_rows(drafts: Sequence[ModelDraft]) -> list[list[str]]:
    rows = []
    for d in drafts:
        if not d.drafted:
            continue
        e, f = d.entry, d.facts
        rows.append([d.key, str(e["family"]), f["revision"]["short"], _fmt_gb(f["weights"]["bytes"]), str(e["tensor_parallel_size"]),
                     str(e["max_model_len"]), _method_label(d), str(e["kernels"])])
    return rows


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
           ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]


def build_problems(drafts: Sequence[ModelDraft], validation: Mapping[str, Mapping]) -> dict[str, Any]:
    unexpected = {k: v["other_reasons"] for k, v in validation.items() if v["status"] == "refused" and not v["expected"]}
    tiered = [d for d in drafts if d.drafted and "tier" in d.entry]
    n_drafted = sum(d.drafted for d in drafts)
    tier_note = None
    if tiered and len(tiered) != n_drafted:
        tier_note = (f"tier is set for {len(tiered)} of {n_drafted} drafted entries; scripts/run_dataset2.py uses tiers only "
                     "when every registered model has one")
    return {"not_drafted": {d.key: d.errors for d in drafts if not d.drafted},
            "unexpected_refusals": unexpected, "tier_note": tier_note}


def render_summary(drafts: Sequence[ModelDraft], validation: Mapping[str, Mapping], inputs: Mapping[str, Any],
                   generated_at: str, out_name: str, prov_name: str) -> str:
    problems = build_problems(drafts, validation)
    n_ok = sum(d.drafted for d in drafts)
    n_acc = sum(1 for v in validation.values() if v["status"] == "accepted")
    L = ["# Registry draft summary", "",
         f"Generated {generated_at} by `scripts/draft_registry.py`. Draft entries: `{out_name}`. Facts and arithmetic per model: "
         f"`{prov_name}`. The draft is a proposal for review; `config/models.yaml` is not touched.", "",
         "## Inputs", "",
         f"- Panel file: `{inputs['panel_file']}` (sha256 {inputs['panel_sha256'][:12]}), {len(drafts)} entries, {n_ok} drafted.",
         f"- GPU memory per GPU: {inputs['gpu_mem_gb']:g} GB, prompt budget {inputs['prompt_budget']} tokens, "
         f"chain-of-thought cap {inputs['cot_cap']} tokens (config/decoding.yaml, status {inputs['decoding_status']}; "
         f"caps {', '.join(f'{a} {c}' for a, c in inputs['max_new_tokens'].items())}), "
         f"maximum tensor-parallel size {inputs['max_tp']}.",
         f"- Thinking-off table: {INVENTORY_REF} ({INVENTORY_DATE}).",
         "", "## Formulas", "",
         f"- {TP_FORMULA}.", f"- {LEN_FORMULA}.",
         "- chat_template.sha256 is null in every entry; it is frozen from the throughput run log.",
         f"- kernels: default for a hybrid architecture with Mamba or linear-attention layers, batch_invariant "
         f"otherwise, from vLLM 0.30.0's registry facts in {KERNEL_RECORD_REF} (and {PANEL_ARCH_RECORD_REF} for "
         "architectures it lacks).", "",
         "## Drafted entries", ""]
    L += _md_table(["key", "family", "sha (12)", "weights GB", "TP", "max_model_len", "thinking-off method", "kernels"],
                   summary_rows(drafts))
    L += ["", "## Check with generation.parse_model_spec", "",
          f"{n_acc} of {n_ok} drafted entries would be accepted by the runner's registry parser; "
          f"{n_ok - n_acc} would be refused. Refusals caused by PLACEHOLDER fields are expected.", ""]
    ref_rows = [[k, v["reasons"]] for k, v in validation.items() if v["status"] == "refused"]
    if ref_rows:
        L += _md_table(["key", "reasons the runner would refuse"], ref_rows) + [""]
    L += ["## Problems", ""]
    if problems["not_drafted"]:
        L += ["Not drafted (the Hub metadata could not be read):", ""]
        L += [f"- `{k}`: {'; '.join(v)}" for k, v in problems["not_drafted"].items()] + [""]
    if problems["unexpected_refusals"]:
        L += ["Refused for reasons other than PLACEHOLDER (a defect in the draft):", ""]
        L += [f"- `{k}`: {v}" for k, v in problems["unexpected_refusals"].items()] + [""]
    todo = [(d.key, t) for d in drafts if d.drafted for t in d.todo]
    if todo:
        L += ["Left as PLACEHOLDER:", ""] + [f"- `{k}`: {t}" for k, t in todo] + [""]
    notes = [(d.key, t) for d in drafts if d.drafted for t in d.warnings]
    if notes:
        L += ["Notes:", ""] + [f"- `{k}`: {t}" for k, t in notes] + [""]
    if problems["tier_note"]:
        L += [f"- {problems['tier_note']}", ""]
    if not (problems["not_drafted"] or problems["unexpected_refusals"] or todo or notes or problems["tier_note"]):
        L += ["None.", ""]
    return "\n".join(L).rstrip() + "\n"


def build_provenance(drafts: Sequence[ModelDraft], validation: Mapping[str, Mapping], inputs: Mapping[str, Any],
                     generated_at: str) -> dict:
    try:
        hub_version = importlib_metadata.version("huggingface_hub")
    except importlib_metadata.PackageNotFoundError:
        hub_version = None
    return {"tool": {"script": "scripts/draft_registry.py", "module": "src/registry_draft.py", "version": TOOL_VERSION,
                     "huggingface_hub": hub_version, "authentication": "cached login if present; no token is read or stored"},
            "generated_at": generated_at, "inputs": dict(inputs),
            "formulas": {"tensor_parallel_size": TP_FORMULA, "max_model_len": LEN_FORMULA},
            "models": {d.key: {"hf_id": d.hf_id, "drafted": d.drafted, "errors": d.errors, "warnings": d.warnings,
                               "todo": d.todo, "entry": d.entry, **d.facts} for d in drafts},
            "validation": dict(validation)}


def draft_outputs(out: Path) -> tuple[Path, Path, Path, Path]:
    """YAML, provenance JSON, summary and clean summary paths for a draft path."""
    stem = out.with_suffix("")
    return (out, stem.parent / (stem.name + "_provenance.json"), stem.parent / (stem.name + "_summary.md"),
            stem.parent / (stem.name + "_summary_clean.md"))


def read_cot_cap(decoding_path: str | Path) -> tuple[int, dict, str]:
    """Largest max_new_tokens cap in the decoding file: (cap, caps by arm, status)."""
    raw = yaml.safe_load(Path(decoding_path).read_text(encoding="utf-8")) or {}
    caps = raw.get("max_new_tokens")
    if not isinstance(caps, Mapping) or not caps or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in caps.values()):
        raise PanelError(f"{decoding_path}: max_new_tokens must map each arm to a positive integer")
    return max(caps.values()), dict(caps), str(raw.get("status"))


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def run(panel_path: str | Path, out: str | Path = DEFAULT_OUT, *, gpu_mem_gb: float | int = DEFAULT_GPU_MEM_GB,
        prompt_budget: int = DEFAULT_PROMPT_BUDGET, max_tp: int = DEFAULT_MAX_TP, decoding_path: str | Path = DECODING_YAML,
        skip_config: bool = False, overwrite: bool = False, api: Any = None, download: Callable[..., str] | None = None,
        now: Callable[[], str] = utc_now) -> dict:
    """Draft, validate and write. Raises PanelError for unusable inputs and RuntimeError when an output would be overwritten."""
    out = Path(out)
    if _same_file(out, MODELS_YAML) or out.name == "models.yaml" and out.parent.name == "config":
        raise PanelError(f"{out} is the live registry; the draft goes to a separate file (default {DEFAULT_OUT.name})")
    if out.suffix.lower() not in (".yaml", ".yml"):
        raise PanelError(f"--out must be a .yaml file, got {out}")
    if isinstance(max_tp, bool) or not isinstance(max_tp, int) or max_tp < 1 or max_tp & (max_tp - 1):
        raise PanelError(f"--max-tp must be a power of two, got {max_tp!r}")
    if isinstance(prompt_budget, bool) or not isinstance(prompt_budget, int) or prompt_budget < 0:
        raise PanelError(f"--prompt-budget must be a non-negative integer, got {prompt_budget!r}")
    if not Decimal(str(gpu_mem_gb)) > 0:
        raise PanelError(f"--gpu-mem-gb must be positive, got {gpu_mem_gb!r}")
    paths = draft_outputs(out)
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise RuntimeError("not overwriting existing output(s) without --overwrite: " + ", ".join(existing))
    panel = load_panel(panel_path)
    cot_cap, caps, dec_status = read_cot_cap(decoding_path)
    panel_sha = hashlib.sha256(Path(panel_path).read_bytes()).hexdigest()
    ctx = DraftContext(api=api if api is not None else HfApi(), download=download or hf_hub_download, gpu_mem_gb=gpu_mem_gb,
                       max_tp=max_tp, prompt_budget=prompt_budget, cot_cap=cot_cap, skip_config=skip_config, now=now)
    generated_at = now()
    drafts = draft_registry(panel, ctx)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml_path, prov_path, md_path, clean_path = paths
    yaml_path.write_text(render_registry_yaml(drafts, generated_at), encoding="utf-8")
    validation = validate_registry_file(yaml_path, drafts)
    inputs = {"panel_file": str(panel_path).replace("\\", "/"), "panel_sha256": panel_sha, "gpu_mem_gb": float(Decimal(str(gpu_mem_gb))),
              "prompt_budget": prompt_budget, "max_tp": max_tp, "cot_cap": cot_cap, "max_new_tokens": caps,
              "decoding_file": str(decoding_path).replace("\\", "/"), "decoding_status": dec_status,
              "skip_config": skip_config, "inventory": f"{INVENTORY_REF} ({INVENTORY_DATE})"}
    prov_path.write_text(json.dumps(build_provenance(drafts, validation, inputs, generated_at), indent=2, sort_keys=False) + "\n",
                         encoding="utf-8")
    summary = render_summary(drafts, validation, inputs, generated_at, yaml_path.name, prov_path.name)
    md_path.write_text(summary, encoding="utf-8")
    clean_path.write_text(summary, encoding="utf-8")
    return {"drafts": drafts, "validation": validation, "summary": summary, "paths": paths,
            "n_unresolved": sum(not d.drafted for d in drafts)}


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description="Draft Dataset 2 registry entries from a panel list (output: a draft file, never "
                                             "config/models.yaml).")
    ap.add_argument("--panel", required=True, metavar="FILE",
                    help="YAML list of entries: hf_id (required); revision, family, tier, key (optional)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="draft registry YAML (default config/models_draft.yaml); the provenance "
                    "JSON and the markdown summaries are written beside it")
    ap.add_argument("--gpu-mem-gb", type=float, default=DEFAULT_GPU_MEM_GB, help="memory per GPU in decimal GB (default 80)")
    ap.add_argument("--prompt-budget", type=int, default=DEFAULT_PROMPT_BUDGET, help="prompt tokens added to the chain-of-thought cap (default 512)")
    ap.add_argument("--max-tp", type=int, default=DEFAULT_MAX_TP, help="largest tensor-parallel size accepted (default 8, one node)")
    ap.add_argument("--decoding", default=str(DECODING_YAML), help="decoding file giving the output caps (default config/decoding.yaml)")
    ap.add_argument("--skip-config", action="store_true", help="do not fetch config.json (metadata API only)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing draft outputs")
    return ap


def main(argv: Sequence[str] | None = None, *, api: Any = None, download: Callable[..., str] | None = None,
         now: Callable[[], str] = utc_now) -> int:
    import sys
    a = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        res = run(a.panel, a.out, gpu_mem_gb=a.gpu_mem_gb, prompt_budget=a.prompt_budget, max_tp=a.max_tp, decoding_path=a.decoding,
                  skip_config=a.skip_config, overwrite=a.overwrite, api=api, download=download, now=now)
    except PanelError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except RuntimeError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(res["summary"])
    print("written: " + ", ".join(str(p) for p in res["paths"]))
    return EXIT_UNRESOLVED if res["n_unresolved"] else EXIT_OK
