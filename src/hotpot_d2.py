r"""HotpotQA in the Dataset 2 pipeline.

The scripts that take `--task bbh` or `--task math` also take `--task hotpot` (src/string_tasks.py); this module holds
what differs, with the names src/math_d2.py uses. Items, the length rule and the scorer are src/hotpot_task.py; the
frozen inputs are manifests/hotpot_items.json and manifests/hotpot_scorer.json.

Item sets
---------
HotpotQA's training file is not available, so every set comes from the development set (BIG-Bench Hard's route):
the screen renders the 100 screening items, the collection the 1,000 target items. There is no
tranche: its predictions are made from the collection's first four discovery draws, scored alone before the rest
(scripts/score_hotpot.py, run_hotpot.py). The build record's split keeps the pipeline's meaning: "train"
marks items outside the target set.

Scoring
-------
hotpot_task.score_outputs: strict and lenient exact match (strict_correct, lenient_correct; the primary score is strict
exact match), compliance, and strict and lenient token F1 (strict_f1, lenient_f1, 0-1). The shared scoring path
(d2_io.score_work_dirs) keeps the exact-match columns; scripts/score_hotpot.py writes the F1 columns as well.

Screening metrics
-----------------
Compliance is hotpot_task.compliance (a complete box with a non-empty answer); the lenient share counts
outputs in which hotpot_task.lenient_extract finds a non-empty answer. No gold answer is read. The direct-answer model
rule is a diagnostic (the panel is fixed, as on MATH-500), and the truncation rule compares a configuration with its
arm's median configuration (MATH-500's rule, applied from the start).

Panel, registry, decoding and arms
----------------------------------
The 29 models of GSM8K-29 (results/dataset2/gsm8k30.json). Every stage runs under config/models_hotpot.yaml (each
model's registry entry with max_model_len 8,192, or 4,096 for the four checkpoints whose native context is 4,096) and
config/decoding_hotpot.yaml, which holds the four arms: requested direct (direct_answer, 1,024 tokens), chain of
thought (zero_shot_cot, 1,900 for every model), forced direct (forced_direct, prefill \boxed{, 64) and five-shot forced
direct (fewshot_forced, the same). One request file per model and block holds all four arms.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import bbh_d2, d2_io, hotpot_task, math_d2

ROOT = d2_io.ROOT
TASK = "hotpot"
TITLE = "HotpotQA"

ITEMS_MANIFEST = Path(hotpot_task.MANIFEST_PATH)
SCORER_MANIFEST = Path(hotpot_task.SCORER_MANIFEST_PATH)
CANDIDATE_BANK = ROOT / "data" / "bank_candidates" / "hotpot_bank_v1.json"
BANK = ROOT / "manifests" / "hotpot_bank.json"
SAMPLER = ROOT / "manifests" / "sampler_hotpot.json"
ADMISSIBILITY_LOG = ROOT / "manifests" / "hotpot_admissibility.json"
TRANCHE_CONFIG = None
TRANCHE_MANIFEST = None
REQUESTS_DIR = ROOT / "data" / "requests" / "hotpot"
PROCESSED = ROOT / "data" / "processed" / "hotpot"
SCORED = {d2_io.KIND_MAIN: PROCESSED / "scored.parquet",
          d2_io.KIND_SCREENING: PROCESSED / "screening_scored.parquet"}
SCREENING_REPORT_DIR = ROOT / "results" / "screening_hotpot"
PREDICTIONS = ROOT / "manifests" / "predictions_hotpot.json"
RESULTS_TABLES = ROOT / "results" / "hotpot" / "tables"
PANEL_EXTENSION = bbh_d2.PANEL_EXTENSION
REGISTRIES = bbh_d2.REGISTRIES
PANEL_SOURCE = ROOT / "results" / "dataset2" / "gsm8k30.json"

KIND_ITEM_SET = {d2_io.KIND_SCREENING: "screening", d2_io.KIND_MAIN: "target"}
KIND_SPLIT = {d2_io.KIND_SCREENING: "train", d2_io.KIND_MAIN: "test"}
ITEM_COLUMNS = ("item_id", "question", "gold_answer", "qtype", "block")
TARGET_LABEL = "the HotpotQA target items"
FIXES_ITEM_SETS = "the item sets are fixed in manifests/hotpot_items.json; there is no tranche"
FIXES_SCORER = "one scorer, fixed in manifests/hotpot_scorer.json"
SCREEN_PURPOSE = ("Format and behaviour metrics of the prompt configurations and models on the 100 HotpotQA screening "
                  "items (development items outside the target set). No correctness, accuracy "
                  "or gain is computed.")
SCREEN_METRICS = {
    "registered_compliance_pct": "src/hotpot_task.py strict extraction finds a complete \\boxed{} whose answer "
                                 "normalises to a non-empty string",
    "strict_compliance_pct": "not reported for HotpotQA: the GSM8K candidate formats are decimal formats",
    "lenient_extractable_pct": "src/hotpot_task.py lenient extraction finds a non-empty answer"}
DIAGNOSTIC_MODEL_RULES = ("direct_answer_meaningful",)
DIAGNOSTIC_MODEL_RULES_SOURCE = ("the 29-model panel is fixed, so the direct-answer rule is a "
                                 "diagnostic, as on MATH-500")
RELATIVE_TRUNCATION = True
RELATIVE_TRUNCATION_SOURCE = ("MATH-500's truncation rule relative to the arm's median "
                              "configuration applies from the start")
SCORER_VERSION = hotpot_task.SCORER_VERSION

COLLECTION_REGISTRY = ROOT / "config" / "models_hotpot.yaml"
DECODING = ROOT / "config" / "decoding_hotpot.yaml"
NATIVE_4K = math_d2.NATIVE_4K
COLLECTION_MAX_MODEL_LEN = 8192
ARMS_ALL = ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced")
FORCED_SUFFIX = "|fd"
FEWSHOT_SUFFIX = "|fs"


class HotpotPipelineError(d2_io.D2IOError):
    """A HotpotQA input does not qualify; the scripts print it as REFUSED and exit 2."""


# --------------------------------------------------------------------------------------------
# Item sets
# --------------------------------------------------------------------------------------------

def _read_manifest(path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    p = Path(path) if path else ITEMS_MANIFEST
    try:
        raw = p.read_bytes()
        m = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise HotpotPipelineError(f"cannot read the HotpotQA items manifest {d2_io.rel_path(p)} ({exc}); write it "
                                  "with `.\\.venv\\Scripts\\python.exe -I scripts\\fetch_hotpot.py`") from exc
    if m.get("task") != TASK or (m.get("source") or {}).get("sha256") != hotpot_task.FILE_SHA256:
        raise HotpotPipelineError(f"{d2_io.rel_path(p)} is not the items manifest of the pinned HotpotQA file")
    return m, {"path": d2_io.rel_path(p), "sha256": d2_io.sha256_bytes(raw)}


def answer_types(manifest_path: str | os.PathLike | None = None) -> None:
    """HotpotQA has one answer type (a short span); screening.py's hook returns None."""
    _read_manifest(manifest_path)
    return None


def _check_items_frame(df: pd.DataFrame, where: str) -> pd.DataFrame:
    missing = [c for c in ITEM_COLUMNS if c not in df.columns]
    if missing:
        raise HotpotPipelineError(f"{where} lacks columns {missing}")
    df = df[list(ITEM_COLUMNS)].copy()
    for c in ITEM_COLUMNS:
        df[c] = df[c].astype(str)
    if df["item_id"].duplicated().any():
        raise HotpotPipelineError(f"{where}: item_id is not unique")
    if (df["item_id"].str.contains("|", regex=False)).any():
        raise HotpotPipelineError(f"{where}: an item id holds '|'")
    return df.reset_index(drop=True)


def load_items(item_set: str, items_parquet: str | os.PathLike | None = None,
               manifest_path: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """One item set (screening, target or demo_pool) and its description for the records (as
    math_d2.load_items: the pinned table checked against the manifest, or a custom parquet labelled as a dry run)."""
    if item_set not in hotpot_task.SETS:
        raise HotpotPipelineError(f"item set must be one of {list(hotpot_task.SETS)}, got {item_set!r}")
    split = "test" if item_set == "target" else "train"
    if items_parquet:
        try:
            df = pd.read_parquet(items_parquet)
        except (OSError, ValueError) as exc:
            raise HotpotPipelineError(f"cannot read {items_parquet}: {exc}") from exc
        df = _check_items_frame(df, str(items_parquet))
        info = {"source": f"custom parquet (dry run, not the pinned HotpotQA {item_set} set)", "task": TASK,
                "item_set": item_set, "split": split, "path": d2_io.rel_path(items_parquet), "n_items": int(len(df)),
                "canonical_sha256": hotpot_task.canonical_sha256(df), "manifest_canonical_sha256": None,
                "matches_manifest": None}
        return df, info
    m, m_info = _read_manifest(manifest_path)
    t = m["tables"][item_set]
    path = Path(t["path"]) if Path(t["path"]).is_absolute() else ROOT / t["path"]
    if not path.is_file():
        raise HotpotPipelineError(f"{d2_io.rel_path(path)} is absent; write the item tables with "
                                  "`.\\.venv\\Scripts\\python.exe -I scripts\\fetch_hotpot.py`")
    got = d2_io.sha256_path(path)
    if got != t["sha256"]:
        raise HotpotPipelineError(f"{d2_io.rel_path(path)} differs from {m_info['path']} (file sha256 {got} != "
                                  f"{t['sha256']})")
    df = _check_items_frame(pd.read_parquet(path), d2_io.rel_path(path))
    want = m["sets"][item_set]["canonical_sha256"]
    canon = hotpot_task.canonical_sha256(df)
    if canon != want or len(df) != m["sets"][item_set]["n_items"]:
        raise HotpotPipelineError(f"{d2_io.rel_path(path)}: canonical sha256 {canon} differs from the manifest's {want}")
    info = {"source": f"{hotpot_task.REPO} @ {hotpot_task.REVISION}, {item_set} set", "task": TASK,
            "item_set": item_set, "split": split, "path": d2_io.rel_path(path), "n_items": int(len(df)),
            "canonical_sha256": canon, "manifest_canonical_sha256": want, "matches_manifest": True,
            "table_sha256": got, "items_manifest": m_info}
    return df, info


def forced_cap(manifest_path: str | os.PathLike | None = None) -> int:
    m, _ = _read_manifest(manifest_path)
    return int(m["forced_cap"]["cap"])


# --------------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------------

def load_scorer(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else SCORER_MANIFEST
    try:
        raw = p.read_bytes()
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise HotpotPipelineError(f"cannot read the HotpotQA scorer manifest {d2_io.rel_path(p)} ({exc})") from exc
    want = hotpot_task.scorer_record()
    if obj != want:
        raise HotpotPipelineError(f"{d2_io.rel_path(p)} differs from src/hotpot_task.py's scorer record "
                                  f"({want['scorer_version']}, rules sha256 {want['rules_sha256'][:12]})")
    return {**want, "status": d2_io.REGISTERED_STATUS, "manifest_path": d2_io.rel_path(p),
            "manifest_sha256": d2_io.sha256_bytes(raw)}


def score_fn(items: pd.DataFrame):
    """The scorer d2_io.score_work_dirs calls per folder: hotpot_task.score_outputs against `items`."""
    def _score(out: pd.DataFrame) -> pd.DataFrame:
        return hotpot_task.score_outputs(out, items)
    return _score


# --------------------------------------------------------------------------------------------
# Screening metrics (format and behaviour only)
# --------------------------------------------------------------------------------------------

def row_metrics(df: pd.DataFrame, types: Any, think_re) -> pd.DataFrame:
    """The flags of screening.row_metrics: registered (compliance), lenient (a non-empty lenient answer), think_tag,
    empty, truncated. No gold is read."""
    key = ["model_key", "arm", "config_id", "item_id"]
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    out = df[key + ["n_output_tokens"]].copy().reset_index(drop=True)
    cache: dict = {}
    flags: dict[str, list] = {"registered": [], "lenient": [], "think_tag": [], "empty": []}
    for t in texts:
        r = cache.get(t)
        if r is None:
            le = hotpot_task.lenient_extract(t)
            r = {"registered": hotpot_task.compliance(t),
                 "lenient": le is not None and hotpot_task.normalize_answer(le) != "",
                 "think_tag": bool(think_re.search(t)), "empty": not t.strip()}
            cache[t] = r
        for name, v in r.items():
            flags[name].append(v)
    for name, v in flags.items():
        out[name] = np.asarray(v, dtype=bool)
    out["truncated"] = (df["finish_reason"].astype(str) == "length").to_numpy()
    return out


# --------------------------------------------------------------------------------------------
# Registries, panel, collection registry and decoding
# --------------------------------------------------------------------------------------------

def load_registries(paths: Sequence[str | os.PathLike] | None = None) -> dict[str, dict]:
    try:
        return bbh_d2.load_registries(paths)
    except bbh_d2.BBHPipelineError as exc:
        raise HotpotPipelineError(str(exc)) from exc


def registry_for(model_key: str, paths: Sequence[str | os.PathLike] | None = None) -> str:
    return bbh_d2.registry_for(model_key, paths)


def extension_tiers(registry: Mapping[str, dict], panel: Sequence[str],
                    panel_manifest: str | os.PathLike | None = None) -> dict[str, str] | None:
    return bbh_d2.extension_tiers(registry, panel, panel_manifest)


def panel(path: str | os.PathLike | None = None) -> list[str]:
    """The 29 models of GSM8K-29 (results/dataset2/gsm8k30.json "models")."""
    p = Path(path) if path else PANEL_SOURCE
    try:
        models = json.loads(p.read_text(encoding="utf-8"))["models"]
    except (OSError, ValueError, KeyError) as exc:
        raise HotpotPipelineError(f"cannot read the GSM8K-29 panel from {d2_io.rel_path(p)} ({exc})") from exc
    return list(models)


def collection_registry(models: Sequence[str] | None = None,
                        paths: Sequence[str | os.PathLike] | None = None) -> dict[str, dict]:
    reg = load_registries(paths)
    keys = list(models) if models is not None else panel()
    missing = [k for k in keys if k not in reg]
    if missing:
        raise HotpotPipelineError(f"not in the registries: {missing}")
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
    head = ("# Generated by src/hotpot_d2.write_collection_registry for HotpotQA: the registry of\n"
            "# every HotpotQA stage (screen, collection, forced and five-shot arms). The GSM8K-29 panel's entries unchanged\n"
            "# except max_model_len (8,192; 4,096 for the checkpoints whose native context is 4,096 tokens). Do not edit by\n"
            "# hand.\n")
    p.write_text(head + yaml.safe_dump({"dataset2": {"models": collection_registry(models, paths)}}, sort_keys=False,
                                       allow_unicode=True), encoding="utf-8")
    return p


def decoding_for(model_key: str, forced: bool = False) -> Path:
    """One decoding file for every model and arm (one cap for all 29 models)."""
    return DECODING
