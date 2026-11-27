r"""MATH-500 in the Dataset 2 pipeline.

The scripts that take `--task bbh` also take `--task math` (src/string_tasks.py maps a task to its module); this
module holds what differs, with the names src/bbh_d2.py uses. Items, the numeric rule and the scorer are
src/math_task.py; the frozen inputs are manifests/math_items.json (the item sets and the item tables' sha256) and
manifests/math_scorer.json.

Item sets
---------
As for BIG-Bench Hard, each request kind renders one item set:
    screening     -> the 200 screening items      (from MATH training problems; build record split "train")
    tranche       -> the calibration items         (from MATH training problems, the size of the target set; "train")
    collection    -> the target items              (the numeric MATH-500 items; "test"; request kind name
                                                    d2_io.KIND_MAIN)
Items are read from data/processed/math/items_<set>.parquet and refused unless their file sha256 and canonical
sha256 equal the manifest's. Item ids are strings (MATH-500's unique_id, "test/<subject>/<n>.json", or
"train/<subject>/<row>"); request ids keep the GSM8K scheme.

Scoring
-------
math_task.score_outputs: strict (the last complete \boxed{} as one rational number), lenient (the last number
expression) and compliance (strict extraction succeeds). The scorer manifest must equal
math_task.scorer_record() exactly.

Screening metrics
-----------------
As for GSM8K (src/screening.py): compliance is math_task.compliance and the lenient-extractable share
counts outputs in which math_task.lenient_extract finds a number. No gold answer is read. The direct-answer model rule
is reported and excludes no model (DIAGNOSTIC_MODEL_RULES); the other rules of config/admissibility.yaml apply.

Panel and arms
--------------
The 29-model GSM8K panel (results/dataset2/gsm8k30.json), from config/models.yaml and config/models_extension.yaml.
Three arms: the request files hold requested direct and chain of thought (d2_io.ARMS); the forced-direct arm is
derived from the direct requests by scripts/build_forced_direct.py --task math and run with
config/decoding_forced.yaml.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import bbh_d2, d2_io, math_task

ROOT = d2_io.ROOT
TASK = "math"
TITLE = "MATH-500"

ITEMS_MANIFEST = Path(math_task.MANIFEST_PATH)
SCORER_MANIFEST = Path(math_task.SCORER_MANIFEST_PATH)
CANDIDATE_BANK = ROOT / "data" / "bank_candidates" / "math_bank_v1.json"
BANK = ROOT / "manifests" / "math_bank.json"
SAMPLER = ROOT / "manifests" / "sampler_math.json"
ADMISSIBILITY_LOG = ROOT / "manifests" / "math_admissibility.json"
TRANCHE_CONFIG = ROOT / "config" / "tranche_math.yaml"
TRANCHE_MANIFEST = ROOT / "manifests" / "math_tranche.json"
REQUESTS_DIR = ROOT / "data" / "requests" / "math"
PROCESSED = ROOT / "data" / "processed" / "math"
SCORED = {d2_io.KIND_MAIN: PROCESSED / "scored.parquet",
          d2_io.KIND_TRANCHE: PROCESSED / "tranche_scored.parquet",
          d2_io.KIND_SCREENING: PROCESSED / "screening_scored.parquet"}
SCREENING_REPORT_DIR = ROOT / "results" / "screening_math"
PREDICTIONS = ROOT / "manifests" / "predictions_math.json"
RESULTS_TABLES = ROOT / "results" / "dataset2" / "tables_math"
TRANCHE_RESULTS = ROOT / "results" / "tranche_math"
PANEL_EXTENSION = bbh_d2.PANEL_EXTENSION
REGISTRIES = bbh_d2.REGISTRIES
PANEL_SOURCE = ROOT / "results" / "dataset2" / "gsm8k30.json"

KIND_ITEM_SET = dict(bbh_d2.KIND_ITEM_SET)
KIND_SPLIT = dict(bbh_d2.KIND_SPLIT)
ITEM_COLUMNS = ("item_id", "question", "gold_answer", "subject", "level")
TARGET_LABEL = "the MATH-500 target items"
FIXES_ITEM_SETS = "the item sets are fixed in manifests/math_items.json"
FIXES_SCORER = "one scorer, fixed in manifests/math_scorer.json"
SCREEN_PURPOSE = ("Format and behaviour metrics of the prompt configurations and models on the 200 MATH screening "
                  "items (training problems). No correctness, accuracy or gain is computed.")
SCREEN_METRICS = {
    "registered_compliance_pct": "src/math_task.py strict extraction succeeds (the last complete \\boxed{} holds "
                                 "one rational number)",
    "strict_compliance_pct": "not reported for MATH: the GSM8K candidate formats are decimal formats",
    "lenient_extractable_pct": "src/math_task.py lenient extraction finds a number expression"}
# The panel is fixed at 29 models: the direct-answer model rule of config/admissibility.yaml (over half of the
# direct answers above 128 tokens) is computed and reported, and drops no model. The think-tag model rule and the
# configuration rules apply as for GSM8K (src/screening.apply_rules), the truncation rule relative to
# the arm's median configuration (below).
DIAGNOSTIC_MODEL_RULES = ("direct_answer_meaningful",)
DIAGNOSTIC_MODEL_RULES_SOURCE = ("the panel is fixed at 29 models, so the direct-answer rule is a diagnostic")
# MATH-500's reasoning runs past the caps of 1,024 and 3,072 tokens in 2 to 5% of the screen's outputs under every
# configuration, so the absolute bound of 2% excluded all 56. As for compliance, a
# configuration is excluded when its pooled share cut at the cap lies more than max_truncation_pct points above the
# arm's median configuration.
RELATIVE_TRUNCATION = True
RELATIVE_TRUNCATION_SOURCE = ("on MATH-500 the truncation rule compares a configuration with its arm's median "
                              "configuration, as the compliance rule does")


class MathPipelineError(d2_io.D2IOError):
    """A MATH input does not qualify; the scripts print it as REFUSED and exit 2."""


# --------------------------------------------------------------------------------------------
# Item sets
# --------------------------------------------------------------------------------------------

def _read_manifest(path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    p = Path(path) if path else ITEMS_MANIFEST
    try:
        raw = p.read_bytes()
        m = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise MathPipelineError(f"cannot read the MATH items manifest {d2_io.rel_path(p)} ({exc}); write it with "
                                "`.\\.venv\\Scripts\\python.exe -I scripts\\fetch_math.py`") from exc
    if (m.get("task") != TASK or m.get("target_revision") != math_task.REV_TARGET
            or m.get("train_revision") != math_task.REV_TRAIN):
        raise MathPipelineError(f"{d2_io.rel_path(p)} is not the items manifest of the pinned MATH revisions")
    return m, {"path": d2_io.rel_path(p), "sha256": d2_io.sha256_bytes(raw)}


def answer_types(manifest_path: str | os.PathLike | None = None) -> None:
    """MATH has one answer type (a number); screening.py's hook returns None."""
    _read_manifest(manifest_path)
    return None


def _check_items_frame(df: pd.DataFrame, where: str) -> pd.DataFrame:
    missing = [c for c in ITEM_COLUMNS if c not in df.columns]
    if missing:
        raise MathPipelineError(f"{where} lacks columns {missing}")
    df = df[list(ITEM_COLUMNS)].copy()
    for c in ITEM_COLUMNS:
        df[c] = df[c].astype(str)
    if df["item_id"].duplicated().any():
        raise MathPipelineError(f"{where}: item_id is not unique")
    for iid, gold in zip(df["item_id"], df["gold_answer"]):
        if math_task.normalize_answer(gold) != gold:
            raise MathPipelineError(f"{where}: item {iid}: gold {gold!r} is not a canonical number")
    return df.reset_index(drop=True)


def load_items(item_set: str, items_parquet: str | os.PathLike | None = None,
               manifest_path: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """One item set (screening, calibration or target) and its description for the records (as
    bbh_d2.load_items: the pinned table checked against the manifest, or a custom parquet labelled as a dry run)."""
    if item_set not in KIND_ITEM_SET.values():
        raise MathPipelineError(f"item set must be one of {sorted(KIND_ITEM_SET.values())}, got {item_set!r}")
    split = "test" if item_set == "target" else "train"
    if items_parquet:
        try:
            df = pd.read_parquet(items_parquet)
        except (OSError, ValueError) as exc:
            raise MathPipelineError(f"cannot read {items_parquet}: {exc}") from exc
        df = _check_items_frame(df, str(items_parquet))
        info = {"source": f"custom parquet (dry run, not the pinned MATH {item_set} set)", "task": TASK,
                "item_set": item_set, "split": split, "path": d2_io.rel_path(items_parquet), "n_items": int(len(df)),
                "canonical_sha256": math_task.canonical_sha256(df), "manifest_canonical_sha256": None,
                "matches_manifest": None}
        return df, info
    m, m_info = _read_manifest(manifest_path)
    t = m["tables"][item_set]
    path = Path(t["path"]) if Path(t["path"]).is_absolute() else ROOT / t["path"]
    if not path.is_file():
        raise MathPipelineError(f"{d2_io.rel_path(path)} is absent; write the item tables with "
                                "`.\\.venv\\Scripts\\python.exe -I scripts\\fetch_math.py`")
    got = d2_io.sha256_path(path)
    if got != t["sha256"]:
        raise MathPipelineError(f"{d2_io.rel_path(path)} differs from {m_info['path']} (file sha256 {got} != "
                                f"{t['sha256']})")
    df = _check_items_frame(pd.read_parquet(path), d2_io.rel_path(path))
    want = m["sets"][item_set]["canonical_sha256"]
    canon = math_task.canonical_sha256(df)
    if canon != want or len(df) != m["sets"][item_set]["n_items"]:
        raise MathPipelineError(f"{d2_io.rel_path(path)}: canonical sha256 {canon} differs from the manifest's {want}")
    src = (f"{math_task.REPO_TARGET} @ {math_task.REV_TARGET}" if item_set == "target"
           else f"{math_task.REPO_TRAIN} @ {math_task.REV_TRAIN}")
    info = {"source": f"{src}, {item_set} set", "task": TASK, "item_set": item_set, "split": split,
            "path": d2_io.rel_path(path), "n_items": int(len(df)), "canonical_sha256": canon,
            "manifest_canonical_sha256": want, "matches_manifest": True, "table_sha256": got,
            "items_manifest": m_info}
    return df, info


# --------------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------------

def load_scorer(path: str | os.PathLike | None = None) -> dict:
    """The MATH scorer record; refused unless the manifest equals math_task.scorer_record()."""
    p = Path(path) if path else SCORER_MANIFEST
    try:
        raw = p.read_bytes()
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise MathPipelineError(f"cannot read the MATH scorer manifest {d2_io.rel_path(p)} ({exc})") from exc
    want = math_task.scorer_record()
    if obj != want:
        raise MathPipelineError(f"{d2_io.rel_path(p)} differs from src/math_task.py's scorer record "
                                f"({want['scorer_version']}, rules sha256 {want['rules_sha256'][:12]}); the scorer "
                                "changed after it was written")
    return {**want, "status": d2_io.REGISTERED_STATUS, "manifest_path": d2_io.rel_path(p),
            "manifest_sha256": d2_io.sha256_bytes(raw)}


SCORER_VERSION = math_task.SCORER_VERSION


def score_fn(items: pd.DataFrame):
    """The scorer d2_io.score_work_dirs calls per folder: math_task.score_outputs against `items` (load_items)."""
    def _score(out: pd.DataFrame) -> pd.DataFrame:
        try:
            return math_task.score_outputs(out, items)
        except math_task.MathError as exc:
            raise ValueError(str(exc)) from exc
    return _score


# --------------------------------------------------------------------------------------------
# Screening metrics (format and behaviour only)
# --------------------------------------------------------------------------------------------

def row_metrics(df: pd.DataFrame, types: Any, think_re) -> pd.DataFrame:
    """The flags of screening.row_metrics for MATH outputs: registered (strict extraction succeeds), lenient (a
    number expression), think_tag, empty, truncated. No gold is read; `types` is unused (one answer type)."""
    key = ["model_key", "arm", "config_id", "item_id"]
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    out = df[key + ["n_output_tokens"]].copy().reset_index(drop=True)
    cache: dict = {}
    flags: dict[str, list] = {"registered": [], "lenient": [], "think_tag": [], "empty": []}
    for t in texts:
        r = cache.get(t)
        if r is None:
            r = {"registered": math_task.strict_extract(t) is not None,
                 "lenient": math_task.lenient_extract(t) is not None,
                 "think_tag": bool(think_re.search(t)), "empty": not t.strip()}
            cache[t] = r
        for name, v in r.items():
            flags[name].append(v)
    for name, v in flags.items():
        out[name] = np.asarray(v, dtype=bool)
    out["truncated"] = (df["finish_reason"].astype(str) == "length").to_numpy()
    return out


# --------------------------------------------------------------------------------------------
# Registries, panel and tiers (the GSM8K-29 panel; the registries of BIG-Bench Hard)
# --------------------------------------------------------------------------------------------

def load_registries(paths: Sequence[str | os.PathLike] | None = None) -> dict[str, dict]:
    try:
        return bbh_d2.load_registries(paths)
    except bbh_d2.BBHPipelineError as exc:
        raise MathPipelineError(str(exc)) from exc


def registry_for(model_key: str, paths: Sequence[str | os.PathLike] | None = None) -> str:
    return bbh_d2.registry_for(model_key, paths)


def extension_tiers(registry: Mapping[str, dict], panel: Sequence[str],
                    panel_manifest: str | os.PathLike | None = None) -> dict[str, str] | None:
    return bbh_d2.extension_tiers(registry, panel, panel_manifest)


def panel(path: str | os.PathLike | None = None) -> list[str]:
    """The 29 models of the GSM8K-29 analysis (results/dataset2/gsm8k30.json "models")."""
    p = Path(path) if path else PANEL_SOURCE
    try:
        models = json.loads(p.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError) as exc:
        raise MathPipelineError(f"cannot read the GSM8K-29 panel from {d2_io.rel_path(p)} ({exc})") from exc
    return list(models)


# --------------------------------------------------------------------------------------------
# Registry and decoding of the tranche, the collection and the forced arm
# --------------------------------------------------------------------------------------------
# The screen ran under the panel's registries (max_model_len 4,096). A tranche or target problem takes up to 1,031
# tokens and its frame at most 46, so with 100 tokens for the chat template's markup the longest prompt is 1,174
# tokens, and 3,072 new tokens do not fit under 4,096. The tranche, the collection and the forced arm therefore run
# under config/models_math.yaml: the panel's entries unchanged except max_model_len, 8,192. The four checkpoints whose
# native context is 4,096 keep it, with a chain-of-thought cap of 4,096 - 1,174 = 2,922 rounded down to 2,900
# (config/decoding_math_4k.yaml).
COLLECTION_REGISTRY = ROOT / "config" / "models_math.yaml"
DECODING = ROOT / "config" / "decoding.yaml"
DECODING_4K = ROOT / "config" / "decoding_math_4k.yaml"
DECODING_FORCED = ROOT / "config" / "decoding_forced_math.yaml"
# max_position_embeddings 4,096 in the pinned config.json (as scripts/build_exemplar_requests.py)
NATIVE_4K = frozenset({"llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "phi-3-mini-4k-instruct",
                       "phi-3-medium-4k-instruct"})
COLLECTION_MAX_MODEL_LEN = 8192


def collection_registry(models: Sequence[str] | None = None,
                        paths: Sequence[str | os.PathLike] | None = None) -> dict[str, dict]:
    """The entries of config/models_math.yaml: each model's entry of its own registry with max_model_len set to
    8,192, or 4,096 for NATIVE_4K."""
    reg = load_registries(paths)
    keys = list(models) if models is not None else panel()
    missing = [k for k in keys if k not in reg]
    if missing:
        raise MathPipelineError(f"not in the registries: {missing}")
    out = {}
    for k in keys:
        e = dict(reg[k])
        e["max_model_len"] = 4096 if k in NATIVE_4K else COLLECTION_MAX_MODEL_LEN
        out[k] = e
    return out


def write_collection_registry(path: str | os.PathLike | None = None, models: Sequence[str] | None = None,
                              paths: Sequence[str | os.PathLike] | None = None) -> Path:
    import yaml
    p = Path(path) if path else COLLECTION_REGISTRY
    head = ("# Generated by src/math_d2.write_collection_registry for MATH-500: the registry of the MATH\n"
            "# tranche, collection and forced arm. The GSM8K-29 panel's entries unchanged\n"
            "# except max_model_len (8,192; 4,096 for the checkpoints whose native context is 4,096 tokens, which run with\n"
            "# config/decoding_math_4k.yaml). Do not edit by hand.\n")
    p.write_text(head + yaml.safe_dump({"dataset2": {"models": collection_registry(models, paths)}}, sort_keys=False,
                                       allow_unicode=True), encoding="utf-8")
    return p


def decoding_for(model_key: str, forced: bool = False) -> Path:
    """The decoding file of a model's tranche or collection job (forced: the forced-direct arm)."""
    if forced:
        return DECODING_FORCED
    return DECODING_4K if model_key in NATIVE_4K else DECODING
