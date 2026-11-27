r"""BIG-Bench Hard in the Dataset 2 pipeline .

The GSM8K scripts take `--task bbh` (scripts/build_requests.py, score_d2.py, screen_d2.py, freeze_bank_d2.py,
draw_tranche_d2.py, freeze_sampler_d2.py, run_dataset2.py, precision_matched.py); this module holds what differs.
Items, answer types and the scorer are src/bbh.py; the frozen inputs are manifests/bbh_items.json (the item sets
and the item tables' sha256) and manifests/bbh_scorer.json.

Item sets
---------
BIG-Bench Hard has no training split. Three disjoint item sets are defined, and each request kind
renders one of them:
    screening     -> the 200 screening items      (build record split "train")
    tranche       -> the 1,350 calibration items  (split "train")
    main  -> the 1,350 target items       (split "test")
The build record's split keeps the pipeline's meaning: "train" marks items outside the target set, so the
runner's cap-only budget gate (generation.calibration_build), the scoring refusals (train outputs are never in a
target table) and the screening guard apply unchanged; "test" marks the target items. A BIG-Bench Hard
record also names the task and the item set. Items are read from the item tables
data/processed/bbh/items_<set>.parquet, refused unless their file sha256 and their canonical sha256 equal the
manifest's. Item ids are strings "<task>/<index>"; request ids keep the GSM8K scheme
("train|<config>|<arm>|<item>" for screening and tranche requests, "<config>|<arm>|<item>" for target items).

Scoring
-------
src/bbh.py score_outputs with the item's answer type: strict (the last complete \boxed{} normalised by the
task's answer type), lenient (the last answer-type token) and compliance (strict extraction succeeds). The
scorer manifest must equal bbh.scorer_record() exactly; its status is "registered".

Screening metrics
-----------------
As for GSM8K (src/screening.py), with two BIG-Bench Hard definitions: compliance is
bbh.compliance(output, answer type of the item's task), and the lenient-extractable share counts outputs in
which bbh.lenient_extract finds a token of that type. The GSM8K candidate formats are number formats and are not
reported. The answer type comes from the task name in the item id and the manifest's per-task type; no gold
answer is read.

Panel
-----
BIG-Bench Hard runs on all 30 checkpoints: config/models.yaml (the 19 main-panel models)
and config/models_extension.yaml (the 11 added). A key may appear in one registry only. Tiers for the extension
analyses are the ECI terciles of the 30 (manifests/d2_panel_extension.json), matched by Hugging Face repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import bbh, d2_io, generation as gen

ROOT = d2_io.ROOT
TASK = "bbh"
TASKS = ("gsm8k", TASK)

ITEMS_MANIFEST = Path(bbh.MANIFEST_PATH)
SCORER_MANIFEST = Path(bbh.SCORER_MANIFEST_PATH)
CANDIDATE_BANK = ROOT / "data" / "bank_candidates" / "bbh_bank_v1.json"
BANK = ROOT / "manifests" / "bbh_bank.json"
SAMPLER = ROOT / "manifests" / "sampler_bbh.json"
ADMISSIBILITY_LOG = ROOT / "manifests" / "bbh_admissibility.json"
TRANCHE_CONFIG = ROOT / "config" / "tranche_bbh.yaml"
TRANCHE_MANIFEST = ROOT / "manifests" / "bbh_tranche.json"
REQUESTS_DIR = ROOT / "data" / "requests" / "bbh"
PROCESSED = ROOT / "data" / "processed" / "bbh"
SCORED = {d2_io.KIND_MAIN: PROCESSED / "scored.parquet",
          d2_io.KIND_TRANCHE: PROCESSED / "tranche_scored.parquet",
          d2_io.KIND_SCREENING: PROCESSED / "screening_scored.parquet"}
SCREENING_REPORT_DIR = ROOT / "results" / "screening_bbh"
PREDICTIONS = ROOT / "manifests" / "predictions_bbh.json"
RESULTS_TABLES = ROOT / "results" / "dataset2" / "tables_bbh"
TRANCHE_RESULTS = ROOT / "results" / "tranche_bbh"
PANEL_EXTENSION = ROOT / "manifests" / "d2_panel_extension.json"
REGISTRIES = (ROOT / "config" / "models.yaml", ROOT / "config" / "models_extension.yaml")

KIND_ITEM_SET = {d2_io.KIND_SCREENING: "screening", d2_io.KIND_TRANCHE: "calibration",
                 d2_io.KIND_MAIN: "target"}
KIND_SPLIT = {d2_io.KIND_SCREENING: "train", d2_io.KIND_TRANCHE: "train", d2_io.KIND_MAIN: "test"}
ITEM_COLUMNS = ("item_id", "question", "gold_answer", "task", "answer_type")
TARGET_LABEL = "the BIG-Bench Hard target items"
# Texts the shared script branches print (src/string_tasks.py); src/math_d2.py has its own.
TITLE = "BIG-Bench Hard"
FIXES_ITEM_SETS = "the item sets are fixed in manifests/bbh_items.json"
FIXES_SCORER = "one scorer, fixed in manifests/bbh_scorer.json"
SCORER_VERSION = bbh.SCORER_VERSION
SCREEN_PURPOSE = ("Format and behaviour metrics of the prompt configurations and models on the 200 BIG-Bench Hard "
                  "screening items. No correctness, accuracy or gain is computed.")
SCREEN_METRICS = {
    "registered_compliance_pct": "src/bbh.py strict extraction succeeds (the last complete \\boxed{} "
                                 "normalises to the answer type of the item's task)",
    "strict_compliance_pct": "not reported for BIG-Bench Hard: the GSM8K candidate formats are number formats",
    "lenient_extractable_pct": "src/bbh.py lenient extraction finds a token of the item's answer type"}


class BBHPipelineError(d2_io.D2IOError):
    """A BIG-Bench Hard input does not qualify; the scripts print it as REFUSED and exit 2."""


def check_task(task: Any) -> str:
    if task not in TASKS:
        raise BBHPipelineError(f"task must be one of {TASKS}, got {task!r}")
    return task


# --------------------------------------------------------------------------------------------
# Item sets
# --------------------------------------------------------------------------------------------

def _read_manifest(path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    p = Path(path) if path else ITEMS_MANIFEST
    try:
        raw = p.read_bytes()
        m = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise BBHPipelineError(f"cannot read the BIG-Bench Hard items manifest {d2_io.rel_path(p)} ({exc}); write it "
                               "with `.\\.venv\\Scripts\\python.exe -I scripts\\fetch_bbh.py`") from exc
    if m.get("revision") != bbh.COMMIT or m.get("tasks") != list(bbh.TASKS):
        raise BBHPipelineError(f"{d2_io.rel_path(p)} is not the items manifest of the pinned commit and the 25 tasks")
    return m, {"path": d2_io.rel_path(p), "sha256": d2_io.sha256_bytes(raw)}


def answer_types(manifest_path: str | os.PathLike | None = None) -> dict[str, bbh.AnswerType]:
    """{task: AnswerType} from the manifest's per-task answer types (metadata; no gold answer is read)."""
    m, _ = _read_manifest(manifest_path)
    return {t: bbh.AnswerType.from_label(r["answer_type"]) for t, r in m["per_task"].items()}


def item_types(item_ids: Sequence[str], types: Mapping[str, bbh.AnswerType]) -> list[bbh.AnswerType]:
    """The answer type of each item id "<task>/<index>"."""
    out = []
    for iid in item_ids:
        task = str(iid).rsplit("/", 1)[0]
        if task not in types:
            raise BBHPipelineError(f"item id {iid!r} does not name one of the 25 BIG-Bench Hard tasks")
        out.append(types[task])
    return out


def _check_items_frame(df: pd.DataFrame, where: str) -> pd.DataFrame:
    missing = [c for c in ITEM_COLUMNS if c not in df.columns]
    if missing:
        raise BBHPipelineError(f"{where} lacks columns {missing}")
    df = df[list(ITEM_COLUMNS)].copy()
    for c in ITEM_COLUMNS:
        df[c] = df[c].astype(str)
    if df["item_id"].duplicated().any():
        raise BBHPipelineError(f"{where}: item_id is not unique")
    for iid, gold, label in zip(df["item_id"], df["gold_answer"], df["answer_type"]):
        try:
            atype = bbh.AnswerType.from_label(label)
        except bbh.BBHError as exc:
            raise BBHPipelineError(f"{where}: item {iid}: {exc}") from exc
        if bbh.normalize_answer(gold, atype) != gold:
            raise BBHPipelineError(f"{where}: item {iid}: gold {gold!r} is not a canonical {label} answer")
    return df.reset_index(drop=True)


def load_items(item_set: str, items_parquet: str | os.PathLike | None = None,
               manifest_path: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """One item set (screening, calibration or target) and its description for the records.

    Without `items_parquet` the set's item table under data/processed/bbh/ is read and refused unless its file
    sha256 and canonical sha256 equal the items manifest's. A custom parquet (the same columns) is for dry runs
    and is labelled as such."""
    if item_set not in KIND_ITEM_SET.values():
        raise BBHPipelineError(f"item set must be one of {sorted(KIND_ITEM_SET.values())}, got {item_set!r}")
    split = "test" if item_set == "target" else "train"
    if items_parquet:
        try:
            df = pd.read_parquet(items_parquet)
        except (OSError, ValueError) as exc:
            raise BBHPipelineError(f"cannot read {items_parquet}: {exc}") from exc
        df = _check_items_frame(df, str(items_parquet))
        info = {"source": f"custom parquet (dry run, not the pinned BIG-Bench Hard {item_set} set)", "task": TASK,
                "item_set": item_set, "split": split, "path": d2_io.rel_path(items_parquet), "n_items": int(len(df)),
                "canonical_sha256": bbh.canonical_sha256(df), "manifest_canonical_sha256": None,
                "matches_manifest": None}
        return df, info
    m, m_info = _read_manifest(manifest_path)
    t = m["tables"][item_set]
    path = Path(t["path"]) if Path(t["path"]).is_absolute() else ROOT / t["path"]
    if not path.is_file():
        raise BBHPipelineError(f"{d2_io.rel_path(path)} is absent; write the item tables with "
                               "`.\\.venv\\Scripts\\python.exe -I scripts\\fetch_bbh.py --no-download`")
    got = d2_io.sha256_path(path)
    if got != t["sha256"]:
        raise BBHPipelineError(f"{d2_io.rel_path(path)} differs from {m_info['path']} (file sha256 {got} != "
                               f"{t['sha256']})")
    df = _check_items_frame(pd.read_parquet(path), d2_io.rel_path(path))
    want = m["sets"][item_set]["canonical_sha256"]
    canon = bbh.canonical_sha256(df)
    if canon != want or len(df) != m["sets"][item_set]["n_items"]:
        raise BBHPipelineError(f"{d2_io.rel_path(path)}: canonical sha256 {canon} differs from the manifest's {want}")
    info = {"source": f"{bbh.REPO} @ {bbh.COMMIT}, {item_set} set", "task": TASK, "item_set": item_set,
            "split": split, "path": d2_io.rel_path(path), "n_items": int(len(df)), "canonical_sha256": canon,
            "manifest_canonical_sha256": want, "matches_manifest": True, "table_sha256": got,
            "items_manifest": m_info}
    return df, info


# --------------------------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------------------------

def load_scorer(path: str | os.PathLike | None = None) -> dict:
    """The BIG-Bench Hard scorer record; refused unless the manifest equals bbh.scorer_record()."""
    p = Path(path) if path else SCORER_MANIFEST
    try:
        raw = p.read_bytes()
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise BBHPipelineError(f"cannot read the BIG-Bench Hard scorer manifest {d2_io.rel_path(p)} ({exc})") from exc
    want = bbh.scorer_record()
    if obj != want:
        raise BBHPipelineError(f"{d2_io.rel_path(p)} differs from src/bbh.py's scorer record ({want['scorer_version']}, "
                               f"rules sha256 {want['rules_sha256'][:12]}); the scorer changed after registration")
    return {**want, "status": d2_io.REGISTERED_STATUS, "manifest_path": d2_io.rel_path(p),
            "manifest_sha256": d2_io.sha256_bytes(raw)}


def score_fn(items: pd.DataFrame):
    """The scorer d2_io.score_work_dirs calls per folder: bbh.score_outputs against `items` (load_items)."""
    def _score(out: pd.DataFrame) -> pd.DataFrame:
        try:
            return bbh.score_outputs(out, items)
        except bbh.BBHError as exc:
            raise ValueError(str(exc)) from exc
    return _score


# --------------------------------------------------------------------------------------------
# Screening metrics (format and behaviour only)
# --------------------------------------------------------------------------------------------

def row_metrics(df: pd.DataFrame, types: Mapping[str, bbh.AnswerType], think_re) -> pd.DataFrame:
    """The flags of screening.row_metrics for BIG-Bench Hard outputs: registered (strict extraction succeeds for the
    item's answer type), lenient (a token of that type), think_tag, empty, truncated. No gold is read."""
    key = ["model_key", "arm", "config_id", "item_id"]
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    atypes = item_types(df["item_id"].tolist(), types)
    out = df[key + ["n_output_tokens"]].copy().reset_index(drop=True)
    cache: dict = {}
    flags: dict[str, list] = {"registered": [], "lenient": [], "think_tag": [], "empty": []}
    for t, at in zip(texts, atypes):
        k = (t, at)
        r = cache.get(k)
        if r is None:
            r = {"registered": bbh.strict_extract(t, at) is not None,
                 "lenient": bbh.lenient_extract(t, at) is not None,
                 "think_tag": bool(think_re.search(t)), "empty": not t.strip()}
            cache[k] = r
        for name, v in r.items():
            flags[name].append(v)
    for name, v in flags.items():
        out[name] = np.asarray(v, dtype=bool)
    out["truncated"] = (df["finish_reason"].astype(str) == "length").to_numpy()
    return out


# --------------------------------------------------------------------------------------------
# Registries and tiers
# --------------------------------------------------------------------------------------------

def load_registries(paths: Sequence[str | os.PathLike] | None = None) -> dict[str, dict]:
    """The union of the registries' dataset2.models (templates left out); a key in two registries is refused."""
    out: dict[str, dict] = {}
    where: dict[str, str] = {}
    for p in (paths or REGISTRIES):
        try:
            reg = gen.load_registry(p)
        except Exception as exc:  # OSError, yaml.YAMLError, gen.RunRefused
            raise BBHPipelineError(f"cannot read the model registry {p}: {exc}") from exc
        for k, v in reg.items():
            if k.startswith("_"):
                continue
            if k in out:
                raise BBHPipelineError(f"model {k!r} is in two registries ({where[k]} and {d2_io.rel_path(p)})")
            out[k], where[k] = v, d2_io.rel_path(p)
    return out


def registry_for(model_key: str, paths: Sequence[str | os.PathLike] | None = None) -> str:
    """The registry file that holds a model (for the runner's MODELS_YAML)."""
    for p in (paths or REGISTRIES):
        if model_key in gen.load_registry(p):
            return d2_io.rel_path(p)
    raise BBHPipelineError(f"model {model_key!r} is in none of the registries")


def extension_tiers(registry: Mapping[str, dict], panel: Sequence[str],
                    panel_manifest: str | os.PathLike | None = None) -> dict[str, str] | None:
    """{model: tier} from the ECI terciles of the extended panel, matched by Hugging Face repository; None when a
    panel model has no tier there (the tier summaries are then left out, as for GSM8K)."""
    p = Path(panel_manifest) if panel_manifest else PANEL_EXTENSION
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BBHPipelineError(f"cannot read {d2_io.rel_path(p)} ({exc})") from exc
    by_repo = {r["repo"]: r.get("tier") for r in man.get("panel") or []}
    tiers = {k: by_repo.get((registry.get(k) or {}).get("hf_id")) for k in panel}
    return tiers if all(isinstance(t, str) and t for t in tiers.values()) else None
