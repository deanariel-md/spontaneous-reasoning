"""Tests for the registry drafting tool (src/registry_draft.py, scripts/draft_registry.py).

No network: HfApi and hf_hub_download are replaced by fakes, and an autouse fixture makes any real
Hub call fail the test. Nothing here writes outside tmp_path, and config/models.yaml is only read.
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src import generation as gen
from src import registry_draft as rd

ROOT = Path(__file__).resolve().parent.parent
SHA_A = "c1899de289a04d12100db370d81485cdf75e47ca"
SHA_B = "b" * 40
SHA_C = "0123456789abcdef0123456789abcdef01234567"
NOW = "2026-10-15T12:00:00+00:00"
FAKE_TOKEN = "hf_FAKEFAKEFAKEFAKEFAKEFAKE123456"

MODELS_YAML_BYTES = (ROOT / "config" / "models.yaml").read_bytes()


# ---------------------------------------------------------------- fakes

def hub_error(cls, msg="403 Client Error", status=403):
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("GET", "https://huggingface.co/api/models/org/name")
    return cls(msg, response=httpx2.Response(status, request=req))


def make_info(sha=SHA_A, params=None, gated=False, license="apache-2.0", files=("config.json", "model.safetensors"),
              st_size=None, chat_template=True, repo_id=None, config=None, **over):
    params = {"BF16": 751_632_384} if params is None else params
    siblings = [SimpleNamespace(rfilename=n, size=(st_size if n.endswith(".safetensors") else 100)) for n in files]
    cfg = config if config is not None else {"architectures": ["Qwen3ForCausalLM"],
                                             "tokenizer_config": {"chat_template": "{{ x }}"} if chat_template else {}}
    d = dict(id=repo_id, sha=sha, gated=gated, private=False, disabled=False,
             card_data=SimpleNamespace(license=license), tags=[],
             safetensors=SimpleNamespace(parameters=params, total=sum(params.values())) if params else None,
             siblings=siblings, config=cfg, created_at=datetime(2025, 4, 27, tzinfo=timezone.utc),
             last_modified=datetime(2025, 7, 26, tzinfo=timezone.utc))
    d.update(over)
    return SimpleNamespace(**d)


class FakeApi:
    """model_info answers from a dict: hf_id -> info, an exception to raise, or a callable(revision)."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def model_info(self, repo_id, *, revision=None, files_metadata=False, token=None, **kw):
        self.calls.append({"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata})
        v = self.table[repo_id]
        if isinstance(v, BaseException):
            raise v
        return v(revision) if callable(v) else v


class FakeDownload:
    """hf_hub_download stand-in: writes config.json into cache_dir. hf_id -> dict, exception, or missing (raises)."""

    def __init__(self, configs):
        self.configs, self.calls = configs, []

    def __call__(self, *, repo_id, filename, revision=None, cache_dir=None, **kw):
        self.calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, "cache_dir": cache_dir})
        v = self.configs.get(repo_id, FileNotFoundError("no config in the fake"))
        if isinstance(v, BaseException):
            raise v
        path = Path(cache_dir) / f"{repo_id.replace('/', '__')}_config.json"
        path.write_text(json.dumps(v), encoding="utf-8")
        return str(path)


QWEN_CFG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "torch_dtype": "bfloat16",
            "max_position_embeddings": 40960, "num_attention_heads": 16, "num_key_value_heads": 8}


def ctx_for(api, download=None, **over):
    kw = dict(api=api, download=download or FakeDownload({}), gpu_mem_gb=80, max_tp=8, prompt_budget=512, cot_cap=1024,
              skip_config=False, now=lambda: NOW)
    kw.update(over)
    return rd.DraftContext(**kw)


def entry_of(hf_id, **kw):
    return rd.PanelEntry(hf_id=hf_id, revision=kw.get("revision"), family=kw.get("family"), tier=kw.get("tier"),
                         key=kw.get("key", rd.default_key(hf_id)), key_given="key" in kw)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the unit tests must not reach the Hugging Face Hub")
    monkeypatch.setattr(rd, "HfApi", boom)
    monkeypatch.setattr(rd, "hf_hub_download", boom)


@pytest.fixture(scope="module", autouse=True)
def live_registry_untouched():
    yield
    assert (ROOT / "config" / "models.yaml").read_bytes() == MODELS_YAML_BYTES, "config/models.yaml was modified"


@pytest.fixture
def decoding(tmp_path):
    p = tmp_path / "decoding.yaml"
    p.write_text(yaml.safe_dump({"status": "PENDING_REVIEW", "max_new_tokens": {"direct_answer": 128, "zero_shot_cot": 1024}}),
                 encoding="utf-8")
    return p


def write_panel(tmp_path, items):
    p = tmp_path / "panel.yaml"
    p.write_text(yaml.safe_dump(items), encoding="utf-8")
    return p


# ---------------------------------------------------------------- revision resolution

def test_branch_resolves_to_commit_sha_and_asks_for_file_metadata():
    api = FakeApi({"Qwen/Qwen3-0.6B": make_info(sha=SHA_A)})
    d = rd.draft_model(entry_of("Qwen/Qwen3-0.6B"), ctx_for(api, FakeDownload({"Qwen/Qwen3-0.6B": QWEN_CFG})))
    assert d.drafted and d.entry["revision"] == SHA_A
    assert api.calls == [{"repo_id": "Qwen/Qwen3-0.6B", "revision": "main", "files_metadata": True}]
    assert d.facts["resolved_at"] == NOW and d.facts["ref_requested"] is None and d.facts["ref_queried"] == "main"


def test_named_branch_and_tag_are_queried_as_given():
    api = FakeApi({"o/m": make_info(sha=SHA_B)})
    d = rd.draft_model(entry_of("o/m", revision="v1.2"), ctx_for(api))
    assert api.calls[0]["revision"] == "v1.2" and d.entry["revision"] == SHA_B
    assert d.facts["ref_requested"] == "v1.2"


def test_given_sha_is_verified_and_lowercased():
    api = FakeApi({"o/m": make_info(sha=SHA_C)})
    d = rd.draft_model(entry_of("o/m", revision=SHA_C.upper()), ctx_for(api))
    assert d.drafted and d.entry["revision"] == SHA_C and d.facts["revision"]["was_sha_given"]


def test_given_sha_that_resolves_elsewhere_is_an_error():
    api = FakeApi({"o/m": make_info(sha=SHA_B)})
    d = rd.draft_model(entry_of("o/m", revision=SHA_C), ctx_for(api))
    assert not d.drafted and "different sha" in d.errors[0]


@pytest.mark.parametrize("bad", [None, "", "main", "abc123", "g" * 40])
def test_hub_answer_without_a_40_hex_sha_is_an_error(bad):
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info(sha=bad)})))
    assert not d.drafted and "40-hex" in d.errors[0]


def test_renamed_repository_is_noted():
    d = rd.draft_model(entry_of("o/old"), ctx_for(FakeApi({"o/old": make_info(repo_id="o/new")})))
    assert any("o/new" in w for w in d.warnings)


# ---------------------------------------------------------------- failures

def test_gated_failure_is_reported_per_model_and_the_others_continue():
    err = hub_error(rd.hub_errors.GatedRepoError, "403 Client Error. Cannot access gated repo")
    api = FakeApi({"meta/gated": err, "Qwen/Qwen3-0.6B": make_info()})
    panel = [entry_of("meta/gated"), entry_of("Qwen/Qwen3-0.6B")]
    drafts = rd.draft_registry(panel, ctx_for(api, FakeDownload({"Qwen/Qwen3-0.6B": QWEN_CFG})))
    assert [d.drafted for d in drafts] == [False, True]
    assert "gated" in drafts[0].errors[0] and "meta/gated" in drafts[0].errors[0]
    assert drafts[0].entry is None and drafts[1].entry["revision"] == SHA_A


@pytest.mark.parametrize("cls,status,needle", [(None, None, "RuntimeError"),
                                                ("RepositoryNotFoundError", 404, "not found"),
                                                ("RevisionNotFoundError", 404, "revision 'nope'"),
                                                ("HfHubHTTPError", 503, "503")])
def test_other_hub_failures_name_the_cause(cls, status, needle):
    exc = RuntimeError("boom") if cls is None else hub_error(getattr(rd.hub_errors, cls), "oops", status)
    d = rd.draft_model(entry_of("o/m", revision="nope"), ctx_for(FakeApi({"o/m": exc})))
    assert not d.drafted and needle in d.errors[0]


def test_error_text_never_carries_a_token():
    exc = RuntimeError(f"auth failed for Bearer {FAKE_TOKEN} and {FAKE_TOKEN}")
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": exc})))
    assert FAKE_TOKEN not in d.errors[0] and "[redacted]" in d.errors[0]
    assert FAKE_TOKEN not in rd.clean_error(f"token={FAKE_TOKEN}")


def test_unreadable_config_json_keeps_the_model_with_a_warning():
    api = FakeApi({"o/m": make_info()})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": hub_error(rd.hub_errors.GatedRepoError)})))
    assert d.drafted and any("config.json was not readable" in w for w in d.warnings)
    assert d.facts["max_model_len"]["model_max"] is None and d.entry["max_model_len"] == 1536
    assert d.facts["config"] is None


def test_config_json_is_fetched_at_the_pinned_sha_into_a_temporary_cache():
    api, dl = FakeApi({"o/m": make_info(sha=SHA_B)}), FakeDownload({"o/m": QWEN_CFG})
    rd.draft_model(entry_of("o/m"), ctx_for(api, dl))
    assert dl.calls[0]["filename"] == "config.json" and dl.calls[0]["revision"] == SHA_B
    assert dl.calls[0]["cache_dir"] and not Path(dl.calls[0]["cache_dir"]).exists()


def test_the_windows_symlink_warning_from_the_temporary_cache_is_silenced(tmp_path):
    import warnings

    def noisy(*, repo_id, filename, revision=None, cache_dir=None, **kw):
        warnings.warn("`huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files", UserWarning)
        p = Path(cache_dir) / "config.json"
        p.write_text("{}", encoding="utf-8")
        return str(p)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg, err = rd.fetch_config(noisy, "o/m", SHA_A)
    assert cfg == {} and err is None and caught == []


def test_skip_config_makes_no_download_call():
    dl = FakeDownload({"o/m": QWEN_CFG})
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info()}), dl, skip_config=True))
    assert dl.calls == [] and d.drafted and any("--skip-config" in w for w in d.warnings)


# ---------------------------------------------------------------- tensor-parallel arithmetic

@pytest.mark.parametrize("weight_bytes,tp", [
    (1, 1), (1_503_264_768, 1),
    (60_000_000_000, 1),          # 60e9 * 1.2 = 72e9 = 80e9 * 0.9: equality fits
    (60_000_000_001, 2),
    (120_000_000_000, 2), (120_000_000_001, 4),
    (240_000_000_000, 4), (240_000_000_001, 8),
    (480_000_000_000, 8), (480_000_000_001, 16),
])
def test_tp_boundaries_at_80_gb(weight_bytes, tp):
    assert rd.propose_tensor_parallel(weight_bytes, 80)["tensor_parallel_size"] == tp


@pytest.mark.parametrize("gpu_gb,weight_bytes,tp", [(40, 30_000_000_000, 1), (40, 30_000_000_001, 2),
                                                    (24, 18_000_000_000, 1), (24, 18_000_000_001, 2),
                                                    (141, 105_750_000_000, 1), (141, 105_750_000_001, 2)])
def test_tp_boundaries_for_other_gpu_sizes(gpu_gb, weight_bytes, tp):
    assert rd.propose_tensor_parallel(weight_bytes, gpu_gb)["tensor_parallel_size"] == tp


def test_tp_result_records_the_arithmetic_and_the_cap():
    r = rd.propose_tensor_parallel(141_200_000_000, 80, max_tp=2)
    assert r["tensor_parallel_size"] == 4 and r["within_max_tp"] is False
    assert r["required_bytes_total"] == pytest.approx(169.44e9) and r["per_gpu_budget_bytes"] == pytest.approx(72e9)
    assert r["required_bytes_per_gpu"] == pytest.approx(42.36e9) and "1.2" in r["formula"] and "0.9" in r["formula"]


@pytest.mark.parametrize("w,mem", [(0, 80), (-5, 80), (True, 80), (1.5, 80), (10, 0), (10, -1), (10, "nan")])
def test_tp_rejects_nonsense_inputs(w, mem):
    with pytest.raises(ValueError):
        rd.propose_tensor_parallel(w, mem)


def test_entry_gets_the_tp_proposal_and_beyond_max_tp_it_is_a_placeholder():
    big = {"BF16": 70_600_000_000}                               # 141.2 GB
    api = FakeApi({"o/big": make_info(params=big, st_size=141_200_000_000)})
    d = rd.draft_model(entry_of("o/big"), ctx_for(api, FakeDownload({"o/big": QWEN_CFG})))
    assert d.entry["tensor_parallel_size"] == 4 and d.facts["tensor_parallel"]["weight_bytes"] == 141_200_000_000
    d2 = rd.draft_model(entry_of("o/big"), ctx_for(api, FakeDownload({"o/big": QWEN_CFG}), max_tp=2))
    assert d2.entry["tensor_parallel_size"] == rd.PLACEHOLDER
    assert any("above --max-tp 2" in t for t in d2.todo) and d2.facts["tensor_parallel"]["tensor_parallel_size"] == 4


def test_head_divisibility_warning_for_tp_above_one():
    assert rd.tp_shape_warnings(1, 14, 2) == []
    assert rd.tp_shape_warnings(4, 16, 8) == []
    assert rd.tp_shape_warnings(4, 16, 2) == []                  # tp is a multiple of the kv heads
    assert "num_attention_heads 14" in rd.tp_shape_warnings(4, 14, 8)[0]
    assert "num_key_value_heads 6" in rd.tp_shape_warnings(4, 24, 6)[0]


# ---------------------------------------------------------------- max_model_len arithmetic

def test_max_model_len_is_cap_plus_budget_rounded_up_to_256():
    r = rd.propose_max_model_len(1024, 512, 40960)
    assert r["max_model_len"] == 1536 and r["sum"] == 1536 and r["capped_at_model_max"] is False
    assert rd.propose_max_model_len(1000, 512, None)["max_model_len"] == 1536       # 1512 up
    assert rd.propose_max_model_len(1024, 513, None)["max_model_len"] == 1792       # 1537 up
    assert rd.propose_max_model_len(1024, 0, None)["max_model_len"] == 1024         # exact multiple stays
    assert rd.propose_max_model_len(1024, 1, None)["max_model_len"] == 1280
    assert rd.propose_max_model_len(1, 0, None)["max_model_len"] == 256


def test_max_model_len_is_capped_at_the_model_maximum_and_never_exceeds_it():
    assert rd.propose_max_model_len(1024, 512, 2048)["max_model_len"] == 1536
    assert rd.propose_max_model_len(1024, 512, 1536)["max_model_len"] == 1536
    r = rd.propose_max_model_len(1024, 512, 1500)               # a maximum that is not a multiple of 256 is used as it is
    assert r["max_model_len"] == 1500 and r["capped_at_model_max"] and r["model_max_below_sum"]
    r = rd.propose_max_model_len(1024, 512, 1024)
    assert r["max_model_len"] == 1024 and r["model_max_below_sum"]
    r = rd.propose_max_model_len(1000, 512, 1520)               # 1512 < 1520 < 1536: capped, but the sum still fits
    assert r["max_model_len"] == 1520 and r["capped_at_model_max"] and not r["model_max_below_sum"]


@pytest.mark.parametrize("model_max", [None, 0, -4, True])
def test_unknown_model_maximum_leaves_the_length_uncapped(model_max):
    r = rd.propose_max_model_len(1024, 512, model_max)
    assert r["max_model_len"] == 1536 and r["model_max"] is None


@pytest.mark.parametrize("cap,budget", [(-1, 512), (1024, -1), (True, 512), (1024.0, 512), (0, 0)])
def test_max_model_len_rejects_nonsense_inputs(cap, budget):
    with pytest.raises(ValueError):
        rd.propose_max_model_len(cap, budget, None)


def test_entry_length_uses_the_decoding_cap_and_the_config_maximum_including_nested_configs():
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info()}), FakeDownload({"o/m": QWEN_CFG})))
    assert d.entry["max_model_len"] == 1536 and d.facts["max_model_len"]["model_max"] == 40960
    nested = {"architectures": ["X"], "text_config": {"max_position_embeddings": 1024, "torch_dtype": "bfloat16",
                                                       "num_attention_heads": 8, "num_key_value_heads": 8}}
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info()}), FakeDownload({"o/m": nested})))
    assert d.entry["max_model_len"] == 1024 and d.facts["config"]["max_position_embeddings_source"] == "text_config.max_position_embeddings"
    assert any("below cot cap" in w for w in d.warnings)


def test_cot_cap_is_the_largest_arm_cap(decoding, tmp_path):
    assert rd.read_cot_cap(decoding) == (1024, {"direct_answer": 128, "zero_shot_cot": 1024}, "PENDING_REVIEW")
    bad = tmp_path / "bad.yaml"
    bad.write_text("max_new_tokens: {a: 0}\n", encoding="utf-8")
    with pytest.raises(rd.PanelError):
        rd.read_cot_cap(bad)
    real_cap = rd.read_cot_cap(rd.DECODING_YAML)[0]
    assert real_cap == max(yaml.safe_load(rd.DECODING_YAML.read_text(encoding="utf-8"))["max_new_tokens"].values())


# ---------------------------------------------------------------- dtype, quantization, weights

def test_plain_dtypes_map_to_registry_dtypes():
    for st, reg in (("BF16", "bfloat16"), ("F16", "float16"), ("F32", "float32")):
        assert rd.propose_dtype({st: 1000})["dtype"] == reg
    assert rd.propose_dtype({"BF16": 995, "F32": 5})["dtype"] == "bfloat16"          # 99.5%


@pytest.mark.parametrize("params", [{"BF16": 600, "F32": 400}, {"F8_E4M3": 900, "BF16": 100}, {"U8": 1000}, {"I32": 500, "F16": 500}])
def test_mixed_or_quantized_storage_has_no_dtype_proposal(params):
    assert rd.propose_dtype(params)["dtype"] is None


def test_weight_bytes_use_each_dtype_width():
    assert rd.weight_bytes_from_parameters({"BF16": 10, "F32": 5, "F8_E4M3": 3}) == (20 + 20 + 3, [])
    assert rd.weight_bytes_from_parameters({"BF16": 10, "MYSTERY": 5}) == (None, ["MYSTERY"])


def test_quantized_checkpoint_gets_placeholders_for_dtype_and_quantization():
    api = FakeApi({"o/fp8": make_info(params={"F8_E4M3": 100_000_000_000, "BF16": 1_000_000_000}, st_size=101_000_000_000)})
    d = rd.draft_model(entry_of("o/fp8"), ctx_for(api, FakeDownload({"o/fp8": QWEN_CFG})))
    assert d.entry["dtype"] == rd.PLACEHOLDER and d.entry["quantization"] == rd.PLACEHOLDER
    assert any(t.startswith("dtype and quantization") for t in d.todo)
    assert d.facts["weights"]["bytes"] == 102_000_000_000 and d.entry["tensor_parallel_size"] == 2


def test_quantization_config_in_config_json_leaves_quantization_a_placeholder():
    cfg = dict(QWEN_CFG, quantization_config={"quant_method": "bitsandbytes"})
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info()}), FakeDownload({"o/m": cfg})))
    assert d.entry["dtype"] == "bfloat16" and d.entry["quantization"] == rd.PLACEHOLDER
    assert any("bitsandbytes" in t for t in d.todo)


def test_float32_storage_and_config_dtype_mismatch_are_noted():
    api = FakeApi({"o/m": make_info(params={"F32": 8_000_000_000}, st_size=32_000_000_000)})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    assert d.entry["dtype"] == "float32" and d.entry["quantization"] == "none"
    assert any("stored as float32" in w for w in d.warnings) and any("torch_dtype" in w for w in d.warnings)


def test_missing_safetensors_metadata_falls_back_to_file_sizes_and_leaves_dtype_open():
    api = FakeApi({"o/m": make_info(params={}, st_size=4_000_000_000, safetensors=None)})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    assert d.facts["weights"]["source"] == "sum of the .safetensors file sizes" and d.entry["tensor_parallel_size"] == 1
    assert d.entry["dtype"] == rd.PLACEHOLDER and d.entry["quantization"] == rd.PLACEHOLDER


def test_no_weight_information_at_all_leaves_tp_open():
    api = FakeApi({"o/m": make_info(params={}, safetensors=None, files=("config.json",))})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    assert d.entry["tensor_parallel_size"] == rd.PLACEHOLDER and any("weight size is unknown" in t for t in d.todo)


def test_file_size_mismatch_is_noted():
    api = FakeApi({"o/m": make_info(params={"BF16": 1_000_000_000}, st_size=9_000_000_000)})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    assert any("differ" in w for w in d.warnings) and d.facts["weights"]["bytes"] == 2_000_000_000


def test_mistral_format_files_leave_tokenizer_mode_a_placeholder():
    api = FakeApi({"o/m": make_info(files=("config.json", "model.safetensors", "tekken.json"))})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    assert d.entry["tokenizer_mode"] == rd.PLACEHOLDER and any("tokenizer_mode" in t for t in d.todo)
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info()}), FakeDownload({"o/m": QWEN_CFG})))
    assert d.entry["tokenizer_mode"] == "auto"


def test_facts_record_license_gating_and_template_presence():
    api = FakeApi({"o/m": make_info(gated="manual", license="llama3.2", chat_template=False)})
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG})))
    h = d.facts["hub"]
    assert h["license"] == "llama3.2" and h["gated"] == "manual" and h["chat_template_present"] is False
    assert any("gated (manual)" in w for w in d.warnings) and any("no chat template" in w for w in d.warnings)
    assert d.entry["chat_template"] == {"source": "tokenizer_default", "file": None, "sha256": None}


def test_license_falls_back_to_the_license_tag():
    d = rd.draft_model(entry_of("o/m"), ctx_for(FakeApi({"o/m": make_info(license=None, tags=["license:mit"])}),
                                                FakeDownload({"o/m": QWEN_CFG})))
    assert d.facts["hub"]["license"] == "mit"


# ---------------------------------------------------------------- thinking-off table

KW_OFF = {"enable_thinking": False}


@pytest.mark.parametrize("hf_id,code", [
    ("Qwen/Qwen3-0.6B", "M1"), ("Qwen/Qwen3-30B-A3B", "M1"), ("Qwen/Qwen3.5-0.8B", "M2"), ("Qwen/Qwen3.6-35B-A3B", "M2"),
    ("google/gemma-4-31B-it", "M3"), ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "M6"), ("nvidia/Nemotron-Cascade-2-30B-A3B", "M6"),
    ("zai-org/GLM-4.5-Air", "M7"), ("zai-org/GLM-4.7-Flash", "M7"), ("HuggingFaceTB/SmolLM3-3B", "M9"),
    ("tencent/Hunyuan-7B-Instruct", "M10"), ("openbmb/MiniCPM4.1-8B", "M15"), ("openbmb/MiniCPM5-1B", "M15"),
    ("internlm/Intern-S1-mini", "M16"), ("sarvamai/sarvam-m", "M20"), ("LGAI-EXAONE/EXAONE-4.5-33B", "M12"),
    ("Qwen/Qwen3-235B-A22B", "M1"), ("Qwen/Qwen3.5-397B-A17B", "M2")])
def test_enable_thinking_false_where_the_inventory_documents_it(hf_id, code):
    t = rd.thinking_off_for(hf_id)
    assert (t.method, t.chat_template_kwargs, t.mcode) == ("chat_template_kwargs", KW_OFF, code)
    assert t.source.startswith("manifests/dataset2_inventory.md Table") and hf_id in t.source and code in t.source


@pytest.mark.parametrize("hf_id", ["ibm-granite/granite-3.2-2b-instruct", "ibm-granite/granite-3.3-8b-instruct"])
def test_granite_uses_the_thinking_boolean(hf_id):
    t = rd.thinking_off_for(hf_id)
    assert (t.method, t.chat_template_kwargs, t.mcode) == ("chat_template_kwargs", {"thinking": False}, "M18")


@pytest.mark.parametrize("hf_id,token", [
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "detailed thinking off"), ("nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1", "detailed thinking off"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1", "detailed thinking off"), ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "detailed thinking off"),
    ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "/no_think"), ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "/no_think"),
    ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", "/no_think")])
def test_system_prompt_token_where_that_is_the_documented_method(hf_id, token):
    t = rd.thinking_off_for(hf_id)
    assert (t.method, t.system_prompt_token, t.chat_template_kwargs) == ("system_prompt_token", token, None)
    assert t.registry_block() == {"method": "system_prompt_token", "chat_template_kwargs": None,
                                  "system_prompt_token": token, "rationale": None}


@pytest.mark.parametrize("hf_id,fragment", [
    ("nvidia/Nemotron-Cascade-8B", "appends the tag to the user input"),
    ("mistralai/Mistral-Small-4-119B-2603", "reasoning_effort"), ("mistralai/Mistral-Medium-3.5-128B", "reasoning_effort"),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "thinking budget"),
    ("LGAI-EXAONE/EXAONE-4.0-32B", "EXAONE 4.0"), ("LGAI-EXAONE/EXAONE-4.0-1.2B", "EXAONE 4.0"), ("LGAI-EXAONE/EXAONE-4.0.1-32B", "EXAONE 4.0"),
    ("NousResearch/Hermes-4-14B", "opt-in"), ("deepcogito/cogito-v1-preview-llama-8B", "opt-in"),
    ("internlm/internlm3-8b-instruct", "long system prompt"), ("deepseek-ai/DeepSeek-V4-Flash", "no Jinja template"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B", "assistant prefix")])
def test_ambiguous_or_undocumented_methods_are_placeholders_with_the_reason(hf_id, fragment):
    t = rd.thinking_off_for(hf_id)
    assert t.method == rd.PLACEHOLDER and fragment in t.reason and t.registry_block()["method"] == rd.PLACEHOLDER
    assert t.mcode is not None


def test_none_needed_quotes_the_inventory_and_resolves_same_rows():
    t = rd.thinking_off_for("Qwen/Qwen2.5-0.5B-Instruct")
    assert t.method == "none_needed" and t.table == "2" and t.chat_template_kwargs is None and t.system_prompt_token is None
    assert '"Yes. Non-reasoning instruct; the card has no reasoning mode."' in t.rationale and "Table 2" in t.rationale
    t = rd.thinking_off_for("Qwen/Qwen2.5-7B-Instruct")
    assert t.method == "none_needed"
    assert '"Yes. Same." (the row above reads "Yes. Non-reasoning instruct; the card has no reasoning mode.")' in t.rationale
    t = rd.thinking_off_for("microsoft/phi-4")
    assert t.rationale.endswith('"Yes. The card has no reasoning mode."')


def test_gated_model_type_rows_and_reasoning_native_rows_are_placeholders():
    t = rd.thinking_off_for("meta-llama/Llama-3.2-1B-Instruct")
    assert t.method == rd.PLACEHOLDER and "model type only" in t.reason and "none_needed" in t.candidate
    for hf_id in ("Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507", "openai/gpt-oss-20b"):
        t = rd.thinking_off_for(hf_id)
        assert t.method == rd.PLACEHOLDER and t.table == "3" and "excluded by rule" in t.reason
    assert "Cannot be disabled by documented means" in rd.thinking_off_for("Qwen/Qwen3-30B-A3B-Thinking-2507").source


def test_unknown_family_is_a_placeholder():
    t = rd.thinking_off_for("acme/Unheard-Of-7B-Chat")
    assert t.method == rd.PLACEHOLDER and t.table is None and t.mcode is None and "not an inventory row" in t.reason
    d = rd.draft_model(entry_of("acme/Unheard-Of-7B-Chat"), ctx_for(FakeApi({"acme/Unheard-Of-7B-Chat": make_info()}),
                                                                    FakeDownload({"acme/Unheard-Of-7B-Chat": QWEN_CFG})))
    assert d.entry["thinking_off"]["method"] == rd.PLACEHOLDER
    assert any(t.startswith("thinking_off.method") for t in d.todo)


def test_lookup_ignores_case_and_a_near_miss_is_unknown():
    assert rd.thinking_off_for("qwen/qwen3-0.6b").method == "chat_template_kwargs"
    assert rd.thinking_off_for("Qwen/Qwen3-0.6B-Base").method == rd.PLACEHOLDER        # a sibling, not an inventory row


def test_evidence_class_t_and_envelope_notes_reach_the_warnings():
    d = rd.draft_model(entry_of("zai-org/GLM-4.7-Flash"), ctx_for(FakeApi({"zai-org/GLM-4.7-Flash": make_info()}),
                                                                  FakeDownload({"zai-org/GLM-4.7-Flash": QWEN_CFG})))
    assert any("evidence class T" in w for w in d.warnings)
    d = rd.draft_model(entry_of("nvidia/Llama-3.1-Nemotron-Nano-8B-v1"), ctx_for(
        FakeApi({"nvidia/Llama-3.1-Nemotron-Nano-8B-v1": make_info()}), FakeDownload({"nvidia/Llama-3.1-Nemotron-Nano-8B-v1": QWEN_CFG})))
    assert any("system_prompt_token is appended" in w for w in d.warnings)
    d = rd.draft_model(entry_of("Qwen/Qwen3-235B-A22B"), ctx_for(FakeApi({"Qwen/Qwen3-235B-A22B": make_info()}),
                                                                 FakeDownload({"Qwen/Qwen3-235B-A22B": QWEN_CFG})))
    assert any("Table 1b" in w for w in d.warnings)


# ---------------------------------------------------------------- transcription against the inventory



def test_every_rule_is_well_formed_and_every_table_code_has_one():
    for code, r in rd.METHOD_RULES.items():
        assert r.code == code
        if r.method == "chat_template_kwargs":
            assert r.kwargs and r.token is None
        elif r.method == "system_prompt_token":
            assert r.token and r.kwargs is None
        else:
            assert r.method == rd.PLACEHOLDER and r.why_placeholder
    for row in rd.INVENTORY_TABLE1 + rd.INVENTORY_TABLE1B:
        assert row[1] in rd.METHOD_RULES
    ids = [r[0].lower() for r in rd.INVENTORY_TABLE1 + rd.INVENTORY_TABLE1B + rd.INVENTORY_TABLE2 + rd.INVENTORY_TABLE3]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- panel file

def test_panel_accepts_strings_and_mappings_and_derives_keys():
    p = rd.parse_panel(["Qwen/Qwen3-0.6B", {"hf_id": "o/m.v2", "tier": "low", "family": "F"},
                        {"hf_id": "o/other", "key": "mine", "revision": SHA_A}])
    assert [x.key for x in p] == ["Qwen3-0.6B", "m.v2", "mine"]
    assert p[1].tier == "low" and p[1].family == "F" and p[2].revision == SHA_A and p[2].key_given and not p[0].key_given


def test_default_key_replaces_characters_outside_the_key_alphabet():
    assert rd.default_key("org/Name_with+odd chars") == "Name_with-odd-chars"
    assert rd.default_key("org/_lead") == "lead"


@pytest.mark.parametrize("raw,needle", [
    ([], "non-empty"), ({"hf_id": "a/b"}, "non-empty"), (["noslash"], "org/name"), ([{"family": "x"}], "hf_id"),
    ([{"hf_id": "a/b", "famly": "x"}], "unknown field"), ([{"hf_id": "a/b", "revision": 1234567890}], "non-empty string"),
    ([{"hf_id": "a/b", "tier": ""}], "tier"), ([{"hf_id": "a/b", "key": "bad key"}], "key"),
    (["a/same", "b/same"], "used by both"), ([42], "mapping")])
def test_bad_panels_are_refused_with_the_reason(raw, needle):
    with pytest.raises(rd.PanelError, match=needle):
        rd.parse_panel(raw)


def test_all_panel_problems_are_reported_together():
    with pytest.raises(rd.PanelError) as e:
        rd.parse_panel(["noslash", {"hf_id": "a/b", "wat": 1}])
    assert "entry 1" in str(e.value) and "entry 2" in str(e.value)


def test_panel_file_errors(tmp_path):
    with pytest.raises(rd.PanelError):
        rd.load_panel(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- [unclosed\n", encoding="utf-8")
    with pytest.raises(rd.PanelError):
        rd.load_panel(bad)


# ---------------------------------------------------------------- schema validity through parse_model_spec

def template_entry():
    reg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["dataset2"]["models"]
    return reg.get("_template")


def drafts_for(tmp_path, decoding, items, api, dl, **kw):
    panel = write_panel(tmp_path, items)
    out = tmp_path / "out" / "draft.yaml"
    res = rd.run(panel, out, decoding_path=decoding, api=api, download=dl, now=lambda: NOW, **kw)
    return res, out


def test_drafted_entries_load_and_parse_through_parse_model_spec(tmp_path, decoding):
    api = FakeApi({"Qwen/Qwen3-0.6B": make_info(), "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": make_info(sha=SHA_B, params={"BF16": 8_030_261_248}, st_size=16_000_000_000),
                   "Qwen/Qwen2.5-7B-Instruct": make_info(sha=SHA_C, params={"BF16": 7_615_616_512}, st_size=15_000_000_000)})
    dl = FakeDownload({k: QWEN_CFG for k in api.table})
    items = [{"hf_id": "Qwen/Qwen3-0.6B", "family": "Qwen3", "tier": "low"}, {"hf_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "family": "Llama-Nemotron", "tier": "mid"},
             {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "family": "Qwen2.5", "tier": "mid", "key": "qwen25-7b"}]
    res, out = drafts_for(tmp_path, decoding, items, api, dl)
    reg = gen.load_registry(out)
    assert list(reg) == ["Qwen3-0.6B", "Llama-3.1-Nemotron-Nano-8B-v1", "qwen25-7b"]
    s = gen.parse_model_spec("Qwen3-0.6B", reg)
    assert (s.hf_id, s.revision, s.dtype, s.quantization, s.tokenizer_mode) == ("Qwen/Qwen3-0.6B", SHA_A, "bfloat16", None, "auto")
    assert (s.max_model_len, s.tensor_parallel_size, s.family) == (1536, 1, "Qwen3")
    assert s.thinking_off.method == "chat_template_kwargs" and s.thinking_off.chat_template_kwargs == KW_OFF
    assert s.chat_template == gen.ChatTemplate("tokenizer_default", None, None)
    n = gen.parse_model_spec("Llama-3.1-Nemotron-Nano-8B-v1", reg)
    assert (n.thinking_off.method, n.thinking_off.system_prompt_token, n.tensor_parallel_size) == ("system_prompt_token", "detailed thinking off", 1)
    q = gen.parse_model_spec("qwen25-7b", reg)
    assert q.thinking_off.method == "none_needed" and "Yes. Same." in q.thinking_off.rationale
    assert all(v["status"] == "accepted" and v["expected"] for v in res["validation"].values())
    assert reg["Qwen3-0.6B"]["tier"] == "low"


def test_entries_have_exactly_the_registry_schema_keys(tmp_path, decoding):
    tpl = template_entry()
    if tpl is None:
        pytest.skip("no _template entry in config/models.yaml")
    api = FakeApi({"o/m": make_info()})
    _, out = drafts_for(tmp_path, decoding, [{"hf_id": "o/m", "family": "F"}], api, FakeDownload({"o/m": QWEN_CFG}))
    e = gen.load_registry(out)["m"]
    assert set(e) == set(tpl)
    assert set(e["chat_template"]) == set(tpl["chat_template"]) and set(e["thinking_off"]) == set(tpl["thinking_off"])
    assert list(yaml.safe_load(out.read_text(encoding="utf-8"))["dataset2"]) == ["registry_schema_version", "models"]
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["dataset2"]["registry_schema_version"] == 1


def test_placeholder_entries_are_expected_refusals_and_a_missing_family_alone_is_not(tmp_path, decoding):
    api = FakeApi({"Qwen/Qwen3-0.6B": make_info(), "acme/Unknown-7B": make_info(sha=SHA_B)})
    dl = FakeDownload({k: QWEN_CFG for k in api.table})
    res, out = drafts_for(tmp_path, decoding, ["Qwen/Qwen3-0.6B", {"hf_id": "acme/Unknown-7B", "family": "Acme"}], api, dl)
    v = res["validation"]
    assert v["Qwen3-0.6B"]["status"] == "accepted" and v["Qwen3-0.6B"]["placeholders"] == []        # family PLACEHOLDER only
    assert gen.load_registry(out)["Qwen3-0.6B"]["family"] == rd.PLACEHOLDER
    assert v["Unknown-7B"]["status"] == "refused" and v["Unknown-7B"]["expected"] is True
    assert "thinking_off.method must be one of" in v["Unknown-7B"]["reasons"]
    assert v["Unknown-7B"]["placeholders"] == ["thinking_off.method"]
    assert not rd.build_problems(res["drafts"], v)["unexpected_refusals"]


def test_a_refusal_without_a_placeholder_is_reported_as_unexpected(tmp_path, decoding):
    api = FakeApi({"Qwen/Qwen3-0.6B": make_info(), "acme/Unknown-7B": make_info(sha=SHA_B)})
    res, out = drafts_for(tmp_path, decoding, [{"hf_id": "Qwen/Qwen3-0.6B", "family": "F"}, {"hf_id": "acme/Unknown-7B", "family": "F"}],
                          api, FakeDownload({k: QWEN_CFG for k in api.table}))
    text = out.read_text(encoding="utf-8").replace("dtype: bfloat16", "dtype: bf16")
    out.write_text(text, encoding="utf-8")
    v = rd.validate_registry_file(out, res["drafts"])
    q = v["Qwen3-0.6B"]
    assert q["status"] == "refused" and q["expected"] is False and "dtype 'bf16'" in q["reasons"] and "dtype 'bf16'" in q["other_reasons"]
    u = v["Unknown-7B"]                                        # a placeholder plus a second, non-placeholder reason
    assert u["status"] == "refused" and u["expected"] is False and u["placeholders"] == ["thinking_off.method"]
    assert "thinking_off.method" in u["reasons"] and "dtype 'bf16'" in u["other_reasons"] and "thinking_off" not in u["other_reasons"]
    assert set(rd.build_problems(res["drafts"], v)["unexpected_refusals"]) == {"Qwen3-0.6B", "Unknown-7B"}
    assert "defect in the draft" in rd.render_summary(res["drafts"], v, res_inputs(), NOW, "d.yaml", "p.json")


def res_inputs():
    return {"panel_file": "p.yaml", "panel_sha256": "0" * 64, "gpu_mem_gb": 80.0, "prompt_budget": 512, "cot_cap": 1024,
            "decoding_status": "PENDING_REVIEW", "max_new_tokens": {"a": 1}, "max_tp": 8}


def test_rendered_yaml_round_trips_to_the_entries_and_keys_are_quoted_when_ambiguous(tmp_path, decoding):
    api = FakeApi({"o/yes": make_info(), "o/m": make_info(sha=SHA_B)})
    items = [{"hf_id": "o/yes", "key": "123"}, {"hf_id": "o/m", "key": "null"}]
    res, out = drafts_for(tmp_path, decoding, items, api, FakeDownload({k: QWEN_CFG for k in api.table}))
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))["dataset2"]["models"]
    assert set(loaded) == {"123", "null"}
    for d in res["drafts"]:
        assert loaded[d.key] == d.entry


def test_the_yaml_carries_a_comment_block_with_sha_arithmetic_and_todos(tmp_path, decoding):
    api = FakeApi({"acme/Unknown-7B": make_info()})
    _, out = drafts_for(tmp_path, decoding, ["acme/Unknown-7B"], api, FakeDownload({"acme/Unknown-7B": QWEN_CFG}))
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# DRAFT registry entries") and f"resolved to {SHA_A}" in text
    assert "tensor_parallel_size 1: 1503264768 B * 1.2 / TP" in text and "(80 GB * 0.9)" in text
    assert "max_model_len 1536: cot cap 1024 + prompt budget 512 = 1536" in text
    assert "TODO thinking_off.method" in text and "TODO family" in text
    assert re.search(r"^    Unknown-7B:$", text, re.M)


# ---------------------------------------------------------------- never writing config/models.yaml

def test_default_output_is_the_draft_file_beside_the_provenance_and_summary():
    assert rd.DEFAULT_OUT == ROOT / "config" / "models_draft.yaml"
    names = [p.name for p in rd.draft_outputs(rd.DEFAULT_OUT)]
    assert names == ["models_draft.yaml", "models_draft_provenance.json", "models_draft_summary.md", "models_draft_summary_clean.md"]
    assert rd.build_parser().parse_args(["--panel", "x"]).out == str(rd.DEFAULT_OUT)


SENTINEL = "sentinel: true\n"


def test_the_live_registry_path_is_refused_as_an_output(tmp_path, monkeypatch, decoding):
    fake_live = tmp_path / "config" / "models.yaml"
    fake_live.parent.mkdir()
    fake_live.write_text(SENTINEL, encoding="utf-8")
    monkeypatch.setattr(rd, "MODELS_YAML", fake_live)
    panel = write_panel(tmp_path, ["o/m"])
    api, dl = FakeApi({"o/m": make_info()}), FakeDownload({})
    monkeypatch.chdir(tmp_path)
    for target in (fake_live, tmp_path / "config" / ".." / "config" / "models.yaml", Path("config") / "models.yaml"):
        with pytest.raises(rd.PanelError, match="live registry"):
            rd.run(panel, target, decoding_path=decoding, api=api, download=dl)
    assert fake_live.read_text(encoding="utf-8") == SENTINEL
    assert api.calls == []                                                   # refused before any Hub call
    argv = ["--panel", str(panel), "--out", str(fake_live), "--decoding", str(decoding)]
    assert rd.main(argv, api=api, download=dl) == rd.EXIT_REFUSED
    assert fake_live.read_text(encoding="utf-8") == SENTINEL and api.calls == []


def test_output_must_be_a_yaml_file(tmp_path, decoding):
    with pytest.raises(rd.PanelError, match=".yaml"):
        rd.run(write_panel(tmp_path, ["o/m"]), tmp_path / "x.json", decoding_path=decoding, api=FakeApi({}), download=FakeDownload({}))


def test_a_full_run_leaves_the_live_registry_byte_identical(tmp_path, decoding):
    api = FakeApi({"o/m": make_info()})
    drafts_for(tmp_path, decoding, ["o/m"], api, FakeDownload({"o/m": QWEN_CFG}))
    assert (ROOT / "config" / "models.yaml").read_bytes() == MODELS_YAML_BYTES


def test_existing_outputs_are_not_replaced_without_overwrite(tmp_path, decoding):
    api = FakeApi({"o/m": make_info()})
    res, out = drafts_for(tmp_path, decoding, ["o/m"], api, FakeDownload({"o/m": QWEN_CFG}))
    out.write_text("# hand edited\n", encoding="utf-8")
    panel = write_panel(tmp_path, ["o/m"])
    with pytest.raises(RuntimeError, match="--overwrite"):
        rd.run(panel, out, decoding_path=decoding, api=api, download=FakeDownload({"o/m": QWEN_CFG}))
    assert out.read_text(encoding="utf-8") == "# hand edited\n"
    rd.run(panel, out, decoding_path=decoding, api=api, download=FakeDownload({"o/m": QWEN_CFG}), overwrite=True, now=lambda: NOW)
    assert out.read_text(encoding="utf-8").startswith("# DRAFT")


@pytest.mark.parametrize("kw,needle", [({"max_tp": 3}, "power of two"), ({"max_tp": 0}, "power of two"), ({"prompt_budget": -1}, "prompt-budget"),
                                        ({"gpu_mem_gb": 0}, "gpu-mem-gb")])
def test_bad_options_are_refused(tmp_path, decoding, kw, needle):
    with pytest.raises(rd.PanelError, match=needle):
        rd.run(write_panel(tmp_path, ["o/m"]), tmp_path / "d.yaml", decoding_path=decoding, api=FakeApi({}), download=FakeDownload({}), **kw)


# ---------------------------------------------------------------- CLI end to end

def run_cli(tmp_path, decoding, items, api, dl, extra=()):
    panel = write_panel(tmp_path, items)
    out = tmp_path / "o" / "models_draft.yaml"
    code = rd.main(["--panel", str(panel), "--out", str(out), "--decoding", str(decoding), *extra], api=api, download=dl, now=lambda: NOW)
    return code, out


def test_cli_writes_all_outputs_and_prints_the_summary(tmp_path, decoding, capsys, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    api = FakeApi({"Qwen/Qwen3-0.6B": make_info(), "meta-llama/Llama-3.2-1B-Instruct": make_info(sha=SHA_B, gated="manual", license="llama3.2"),
                   "acme/Unknown-70B": make_info(sha=SHA_C, params={"BF16": 70_000_000_000}, st_size=140_000_000_000)})
    dl = FakeDownload({k: QWEN_CFG for k in api.table})
    code, out = run_cli(tmp_path, decoding, [{"hf_id": "Qwen/Qwen3-0.6B", "family": "Qwen3"}, "meta-llama/Llama-3.2-1B-Instruct", "acme/Unknown-70B"], api, dl)
    assert code == rd.EXIT_OK
    yml, prov, md, clean = rd.draft_outputs(out)
    assert all(p.exists() for p in (yml, prov, md, clean))
    assert md.read_text(encoding="utf-8") == clean.read_text(encoding="utf-8")
    text = md.read_text(encoding="utf-8")
    assert "| key | family | sha (12) | weights GB | TP | max_model_len | thinking-off method | kernels |" in text
    assert f"| Qwen3-0.6B | Qwen3 | {SHA_A[:12]} | 1.5 | 1 | 1536 | chat_template_kwargs {{enable_thinking: false}} | batch_invariant |" in text
    assert f"| Llama-3.2-1B-Instruct | PLACEHOLDER | {SHA_B[:12]} | 1.5 | 1 | 1536 | PLACEHOLDER | batch_invariant |" in text
    assert f"| Unknown-70B | PLACEHOLDER | {SHA_C[:12]} | 140.0 | 4 | 1536 | PLACEHOLDER | batch_invariant |" in text
    assert "1 of 3 drafted entries would be accepted" in text and "2 would be refused" in text
    assert "chat_template.sha256 is null in every entry" in text
    printed = capsys.readouterr().out
    assert "| Qwen3-0.6B |" in printed and "written:" in printed
    data = json.loads(prov.read_text(encoding="utf-8"))
    assert data["inputs"]["cot_cap"] == 1024 and data["inputs"]["gpu_mem_gb"] == 80.0 and data["inputs"]["panel_sha256"]
    m = data["models"]["Qwen3-0.6B"]
    assert m["revision"]["sha"] == SHA_A and m["tensor_parallel"]["tensor_parallel_size"] == 1 and m["hub"]["license"] == "apache-2.0"
    assert m["max_model_len"]["model_max"] == 40960 and m["thinking_off"]["m_code"] == "M1" and m["resolved_at"] == NOW
    assert data["validation"]["Qwen3-0.6B"]["status"] == "accepted"
    assert data["formulas"] == {"tensor_parallel_size": rd.TP_FORMULA, "max_model_len": rd.LEN_FORMULA}
    for p in (yml, prov, md, clean):
        assert FAKE_TOKEN not in p.read_text(encoding="utf-8") and FAKE_TOKEN not in printed


def test_cli_exit_code_and_summary_when_a_model_cannot_be_resolved(tmp_path, decoding):
    api = FakeApi({"meta/gated": hub_error(rd.hub_errors.GatedRepoError), "o/m": make_info()})
    code, out = run_cli(tmp_path, decoding, ["meta/gated", "o/m"], api, FakeDownload({"o/m": QWEN_CFG}))
    assert code == rd.EXIT_UNRESOLVED
    text = rd.draft_outputs(out)[2].read_text(encoding="utf-8")
    assert "Not drafted (the Hub metadata could not be read):" in text and "`gated`: gated repository meta/gated" in text
    assert set(gen.load_registry(out)) == {"m"}
    data = json.loads(rd.draft_outputs(out)[1].read_text(encoding="utf-8"))
    assert data["models"]["gated"]["drafted"] is False and data["models"]["gated"]["entry"] is None


def test_cli_with_every_model_unresolved_writes_an_empty_registry(tmp_path, decoding):
    code, out = run_cli(tmp_path, decoding, ["meta/gated"], FakeApi({"meta/gated": hub_error(rd.hub_errors.GatedRepoError)}), FakeDownload({}))
    assert code == rd.EXIT_UNRESOLVED and gen.load_registry(out) == {}
    assert "0 of 0 drafted entries" in rd.draft_outputs(out)[2].read_text(encoding="utf-8")


def test_cli_refuses_a_bad_panel_and_existing_outputs(tmp_path, decoding, capsys):
    api, dl = FakeApi({"o/m": make_info()}), FakeDownload({"o/m": QWEN_CFG})
    assert run_cli(tmp_path, decoding, ["noslash"], api, dl)[0] == rd.EXIT_REFUSED
    assert "org/name" in capsys.readouterr().err
    assert run_cli(tmp_path, decoding, ["o/m"], api, dl)[0] == rd.EXIT_OK
    assert run_cli(tmp_path, decoding, ["o/m"], api, dl)[0] == rd.EXIT_REFUSED
    assert run_cli(tmp_path, decoding, ["o/m"], api, dl, extra=["--overwrite"])[0] == rd.EXIT_OK


def test_cli_options_reach_the_arithmetic(tmp_path, decoding):
    api = FakeApi({"o/m": make_info(params={"BF16": 20_000_000_000}, st_size=40_000_000_000)})        # 40 GB
    code, out = run_cli(tmp_path, decoding, ["o/m"], api, FakeDownload({"o/m": QWEN_CFG}),
                        extra=["--gpu-mem-gb", "40", "--prompt-budget", "1024", "--max-tp", "1"])
    reg = gen.load_registry(out)["m"]
    assert reg["tensor_parallel_size"] == rd.PLACEHOLDER                    # 2 GPUs needed at 40 GB, above --max-tp 1
    assert reg["max_model_len"] == 2048                                     # 1024 + 1024
    d = rd.draft_model(entry_of("o/m"), ctx_for(api, FakeDownload({"o/m": QWEN_CFG}), gpu_mem_gb=40))
    assert d.entry["tensor_parallel_size"] == 2


def test_cli_tier_note_when_only_some_entries_have_one(tmp_path, decoding):
    api = FakeApi({"o/a": make_info(), "o/b": make_info(sha=SHA_B)})
    code, out = run_cli(tmp_path, decoding, [{"hf_id": "o/a", "tier": "low"}, "o/b"], api, FakeDownload({k: QWEN_CFG for k in api.table}))
    assert "tier is set for 1 of 2 drafted entries" in rd.draft_outputs(out)[2].read_text(encoding="utf-8")


def test_script_help_runs_without_network():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "draft_registry.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--panel" in r.stdout and "--gpu-mem-gb" in r.stdout and "--prompt-budget" in r.stdout


def test_script_refuses_to_run_without_a_panel():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "draft_registry.py")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "--panel" in r.stderr


# ---------------------------------------------------------------- kernel mode

HYBRID_CFG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                              "max_position_embeddings": 262144, "torch_dtype": "bfloat16"}}


def test_kernel_mode_follows_the_d4_rule():
    if not (ROOT / "results" / "batch_invariance" / "support_20261028.json").exists():
        pytest.skip("needs results/batch_invariance/support_20261028.json from the study's outputs")
    archs = rd.load_kernel_record()
    assert archs["Qwen3ForCausalLM"]["is_hybrid"] is False and archs["NemotronHForCausalLM"]["is_hybrid"] is True
    p = rd.propose_kernels(rd.config_facts(QWEN_CFG), archs)
    assert p["kernels"] == "batch_invariant" and "does not mark it as hybrid" in p["rule"]
    facts = rd.config_facts(HYBRID_CFG)
    assert facts["layer_types"] == ["full_attention", "linear_attention"] and facts["layer_types_source"] == "text_config.layer_types"
    assert rd.propose_kernels(facts, archs)["kernels"] == "default"
    nemotron_h = rd.config_facts({"architectures": ["NemotronHForCausalLM"], "hybrid_override_pattern": "M-M-M*-"})
    assert nemotron_h["layer_types"] is None and rd.propose_kernels(nemotron_h, archs)["kernels"] == "default"
    attention_only = rd.propose_kernels(rd.config_facts({"architectures": ["GraniteMoeHybridForCausalLM"],
                                                         "layer_types": ["attention"] * 4}), archs)
    assert attention_only["kernels"] == rd.PLACEHOLDER and "attention layers only" in attention_only["reason"]
    unknown = rd.propose_kernels(rd.config_facts({"architectures": ["AcmeForCausalLM"]}), archs)
    assert unknown["kernels"] == rd.PLACEHOLDER and "inspect_arch.py" in unknown["reason"]
    unsupported = rd.propose_kernels(rd.config_facts({"architectures": ["X"]}), {"X": {"supported": False}})
    assert unsupported["kernels"] == rd.PLACEHOLDER and "not in vLLM 0.30.0's model registry" in unsupported["reason"]
    assert rd.propose_kernels({}, archs)["reason"] == "config.json was not read, so the architecture is unknown"


def test_architectures_missing_on_28_october_come_from_the_panel_record():
    if not (ROOT / "results" / "batch_invariance" / "support_20261028.json").exists():
        pytest.skip("needs results/batch_invariance/support_20261028.json from the study's outputs")
    archs = rd.load_kernel_record()
    for a in ("GemmaForCausalLM", "Gemma3ForConditionalGeneration"):
        assert a not in rd.load_kernel_record(rd.KERNEL_RECORD_PATH)
        p = rd.propose_kernels(rd.config_facts({"architectures": [a]}), archs)
        assert p["kernels"] == "batch_invariant" and p["record"] == rd.PANEL_ARCH_RECORD_REF
    assert archs["Qwen3ForCausalLM"]["record"] == rd.KERNEL_RECORD_REF          # the earlier record wins
    assert rd.propose_kernels(rd.config_facts(QWEN_CFG), archs)["record"] == rd.KERNEL_RECORD_REF


def test_kernel_rule_reproduces_the_28_october_classification():
    record = json.loads((ROOT / "results" / "batch_invariance" / "support_20261028.json").read_text(encoding="utf-8"))
    expect = {"refused": "default", "no refusal found": "batch_invariant", "unread": rd.PLACEHOLDER,
              "unsupported": rd.PLACEHOLDER}
    for c in record["checkpoints"]:
        facts = {} if "error" in c else {"architectures": c["architectures"], "layer_types": c["layer_types"]}
        got = rd.propose_kernels(facts, record["architectures"])["kernels"]
        if c["status"] == "unverified":     # FP8 weights draft as batch_invariant; a hybrid class with attention only stays open
            assert got == (rd.PLACEHOLDER if c["why"].startswith("hybrid") else "batch_invariant"), c["hf_id"]
        else:
            assert got == expect[c["status"]], c["hf_id"]


def test_the_draft_carries_the_kernel_mode_its_todo_and_the_quantization_warning():
    api = FakeApi({"Qwen/Qwen3.5-4B": make_info(), "o/fp8": make_info(sha=SHA_B), "o/nocfg": make_info(sha=SHA_C)})
    fp8 = dict(QWEN_CFG, quantization_config={"quant_method": "fp8"})
    ctx = ctx_for(api, FakeDownload({"Qwen/Qwen3.5-4B": HYBRID_CFG, "o/fp8": fp8}))
    hybrid = rd.draft_model(entry_of("Qwen/Qwen3.5-4B"), ctx)
    assert hybrid.entry["kernels"] == "default" and hybrid.facts["kernels"]["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert any(ln.startswith("kernels default: vLLM 0.30.0 refuses") for ln in rd.entry_comment_lines(hybrid))
    assert "      kernels: default" in rd.render_entry_yaml(hybrid).splitlines()
    q = rd.draft_model(entry_of("o/fp8"), ctx)
    assert q.entry["kernels"] == "batch_invariant" and any("FP8 Marlin" in w for w in q.warnings)
    n = rd.draft_model(entry_of("o/nocfg"), ctx)
    assert n.entry["kernels"] == rd.PLACEHOLDER and "kernels: config.json was not read, so the architecture is unknown" in n.todo
