r"""Dataset 2 request building and output scoring.

Data flow
    prompt bank -> sampler -> rendered request files (scripts/build_requests.py)
    -> src/generation.py runner, one model per run (scripts/generate_d2.py)
    -> scored table (scripts/score_d2.py) -> scripts/run_dataset2.py / src/dataset2.cell_table

This module holds the logic both scripts share; the scripts only parse arguments and write files.
Nothing here reads outcomes to choose models, items or configurations.

Bank schema (manifests/gsm8k_bank.json)
    {
      "generator": str,              # e.g. "PromptSuite", with the facts needed to regenerate it
      "generator_version": str,
      "seed": int,
      "sampling_probabilities": {config_id: p, ...},   # one per configuration, p >= 0, sum 1 +- 1e-9
      "configurations": {
        config_id: {
          "dimensions": {...},       # free metadata (which wording/format/separator variant, ...)
          "arms": {
            "direct_answer": {"messages": [{"role": ..., "content": ...}, ...]},
            "zero_shot_cot": {"messages": [...]}
          }
        }, ...
      }
    }
  * Messages are OpenAI-style dicts with exactly the keys role (system, user or assistant) and
    content (str); the last message has role user. Any other top-level or per-configuration key is
    carried along and ignored. Arms other than the two above are refused (the
    design contrasts these two).
  * The only item variable is the literal token `{question}`. It occurs exactly once across the
    messages of each arm of each configuration and is replaced by plain str.replace, never
    str.format, so any other brace in a prompt (JSON examples, \boxed{...}, {{x}}) is left alone.
    The replacement text is not scanned again.
  * config_id is a non-empty string without "|" and without surrounding whitespace. The request id
    is f"{config_id}|{arm}|{item_id}", unique by construction.
  * sampling_probabilities is a mapping so that scripts/freeze_sampler_d2.py, src/tranche.py and
    src/gate.py read the same file.
  * Within a configuration one fixed skeleton is applied to every item; only the question changes.
    All variation exists across configurations.
  * Warnings (not errors): a configuration whose two arms have identical messages, and
    configurations whose messages are identical to another configuration's (the runner generates
    identical prompts once).

Request files: parquet with the runner's columns (generation.REQUEST_COLUMNS). item_id is int64
(GSM8K test row position). messages_json is json.dumps(messages, ensure_ascii=False).

Splits. Main requests render the GSM8K test items (the target items). Admissibility
screening and noise or throughput tranches that must not touch a
target outcome render a deterministic sample of GSM8K TRAINING items, drawn from item ids only
(select_calibration_items, manifests/gsm8k_calibration_items.json). Train request ids carry the
split as a prefix, f"train|{config_id}|{arm}|{item_id}", so a request id never repeats across splits.
Train and test item ids are separate namespaces.

Build record (data/requests/gsm8k/build_record.json for main builds, .../tranche/ and
.../screening/ build_record.json otherwise): kind, split, purpose, excluded_from_main, bank
and sampler hashes, items canonical sha (and, for train, the calibration-sample manifest), configuration
ids, per-model request-file sha256 and counts per arm. scripts/score_d2.py matches each generation
folder to a record through the request-file sha256 in the run log, so main, tranche and
screening outputs cannot be mixed up.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import forced_stop, generation as gen, gsm8k, sampling, scoring

ROOT = Path(__file__).resolve().parent.parent

ARMS = ("direct_answer", "zero_shot_cot")        # config/tasks.yaml dataset2.intervention
QUESTION_TOKEN = "{question}"
ROLES = ("system", "user", "assistant")
MESSAGE_KEYS = frozenset({"role", "content"})
BANK_KEYS = ("generator", "generator_version", "seed", "sampling_probabilities", "configurations")
PROB_TOL = 1e-9
MODEL_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

BUILD_RECORD = "build_record.json"
TRANCHE_DIR = "tranche"
SCREENING_DIR = "screening"
KIND_MAIN, KIND_TRANCHE, KIND_SCREENING = "main", "tranche", "screening"
KINDS = (KIND_MAIN, KIND_TRANCHE, KIND_SCREENING)
SPLITS = ("test", "train")
PURPOSES = {KIND_MAIN: "main_collection", KIND_TRANCHE: "throughput_noise_tranche",
            KIND_SCREENING: "admissibility_screening"}
SCHEMA_VERSION = "1"
CALIBRATION_LABEL = "gsm8k-calibration-items"
CALIBRATION_MANIFEST = ROOT / "manifests" / "gsm8k_calibration_items.json"

SCORED_COLUMNS = ("model_key", "arm", "config_id", "item_id", "strict_correct", "lenient_correct", "compliant",
                  "finish_reason", "n_output_tokens", "cached", "prompt_sha256")
CORRECTNESS_COLUMNS = ("strict_correct", "lenient_correct")
# A screening table carries no correctness column: screening judges format and behaviour only.
SCREENING_COLUMNS = tuple(c for c in SCORED_COLUMNS if c not in CORRECTNESS_COLUMNS)
ADMISSIBILITY_LOG = ROOT / "manifests" / "gsm8k_admissibility.json"
TEST_BACKENDS = ("echo",)                        # same label as src/tranche.py
CANDIDATE_STATUS = "candidate, not registered"
REGISTERED_STATUS = "registered"
KEY_COLUMNS = ["model_key", "arm", "config_id", "item_id"]


class D2IOError(Exception):
    """An input does not qualify; the scripts print it as REFUSED and exit 2. Nothing is written."""


class BankError(D2IOError):
    """The prompt bank, the sampler or a request list is malformed."""


class ScoreInputError(D2IOError):
    """A generation folder, scorer or item set cannot be scored."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: str | os.PathLike) -> str:
    return gen.sha256_file(path)


def utc_now() -> str:
    return gen.utc_now()


def check_model_key(model_key: Any) -> str:
    """Model keys name request files, so they are restricted to a safe alphabet."""
    if not isinstance(model_key, str) or not MODEL_KEY_RE.match(model_key):
        raise BankError(f"model key {model_key!r} must match {MODEL_KEY_RE.pattern} (it names a file)")
    return model_key


def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"cannot serialize {type(o).__name__}")


def rel_path(path: str | os.PathLike) -> str:
    """Repository-relative path when the file lies under the repository, else the absolute path."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------------------------
# The prompt bank
# --------------------------------------------------------------------------------------------

def _no_duplicate_keys(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise BankError(f"duplicate key {k!r} in a JSON object (a later entry would silently replace the earlier one)")
        out[k] = v
    return out


def _message_problems(where: str, msgs: Any) -> list[str]:
    if not isinstance(msgs, list) or not msgs:
        return [f"{where}: messages must be a non-empty list"]
    out: list[str] = []
    n_token, all_ok = 0, True
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or set(m) != MESSAGE_KEYS:
            out.append(f"{where}: message {i} must be an object with exactly the keys role and content")
            all_ok = False
            continue
        if m["role"] not in ROLES:
            out.append(f"{where}: message {i} has role {m['role']!r}, expected one of {ROLES}")
            all_ok = False
        if not isinstance(m["content"], str):
            out.append(f"{where}: message {i} content must be a string")
            all_ok = False
            continue
        n_token += m["content"].count(QUESTION_TOKEN)
    if all_ok and msgs[-1]["role"] != "user":
        out.append(f"{where}: the last message must have role user")
    if all_ok and n_token != 1:
        out.append(f"{where}: {QUESTION_TOKEN} occurs {n_token} times across the messages (exactly once is required)")
    return out


def _probability_problems(sp: Any, ids: list[str]) -> list[str]:
    if not isinstance(sp, dict):
        return ["sampling_probabilities must be a mapping of config_id to probability"]
    out: list[str] = []
    missing, extra = sorted(set(ids) - set(sp)), sorted(set(sp) - set(ids))
    if missing:
        out.append(f"{len(missing)} configurations without a sampling probability, e.g. {missing[:3]}")
    if extra:
        out.append(f"{len(extra)} sampling probabilities without a configuration, e.g. {extra[:3]}")
    vals = [sp[i] for i in ids if i in sp]
    if not all(is_finite_number(v) for v in vals):
        out.append("a sampling probability is not a finite number")
        return out
    if any(v < 0 for v in vals):
        out.append("a sampling probability is negative")
    if not missing and not extra:
        total = math.fsum(vals)
        if abs(total - 1.0) > PROB_TOL:
            out.append(f"sampling probabilities sum to {total!r}, not 1 within {PROB_TOL:g}")
    return out


def validate_bank(bank: Any) -> list[str]:
    """Raise BankError listing every problem at once; return the non-fatal warnings."""
    if not isinstance(bank, dict):
        raise BankError("bank must be a JSON object")
    problems: list[str] = []
    warnings: list[str] = []
    for k in BANK_KEYS:
        if k not in bank or bank[k] is None or (isinstance(bank[k], str) and not bank[k].strip()):
            problems.append(f"missing or empty key {k!r}")
    configs = bank.get("configurations")
    if "configurations" in bank and bank["configurations"] is not None:
        if not isinstance(configs, dict) or not configs:
            problems.append("configurations must be a non-empty mapping of config_id to configuration")
            configs = None
    else:
        configs = None
    if configs is not None:
        for cid, cfg in configs.items():
            if not cid or cid != cid.strip() or "|" in cid:
                problems.append(f"config_id {cid!r} must be non-empty, without surrounding whitespace and without '|'")
            if not isinstance(cfg, dict):
                problems.append(f"configuration {cid!r} must be an object")
                continue
            if not isinstance(cfg.get("dimensions"), dict):
                problems.append(f"configuration {cid!r}: dimensions must be an object (it may be empty)")
            arms = cfg.get("arms")
            if not isinstance(arms, dict):
                problems.append(f"configuration {cid!r}: arms must be an object with {list(ARMS)}")
                continue
            for a in ARMS:
                if a not in arms:
                    problems.append(f"configuration {cid!r}: arm {a!r} is missing")
                else:
                    spec = arms[a]
                    where = f"configuration {cid!r}, arm {a!r}"
                    if not isinstance(spec, dict):
                        problems.append(f"{where}: must be an object with a messages list")
                    else:
                        problems += _message_problems(where, spec.get("messages"))
            for a in sorted(set(arms) - set(ARMS)):
                problems.append(f"configuration {cid!r}: unexpected arm {a!r} (the arms are {list(ARMS)})")
        if "sampling_probabilities" in bank and bank["sampling_probabilities"] is not None:
            problems += _probability_problems(bank["sampling_probabilities"], list(configs))
    if problems:
        raise BankError("prompt bank invalid: " + "; ".join(problems))

    same_arms = [cid for cid, c in configs.items()
                 if c["arms"]["direct_answer"]["messages"] == c["arms"]["zero_shot_cot"]["messages"]]
    if same_arms:
        warnings.append(f"{len(same_arms)} configurations have identical messages in both arms, e.g. {same_arms[:3]}")
    seen: dict[str, str] = {}
    dup = 0
    for cid, c in configs.items():
        key = gen.canonical_json([c["arms"][a]["messages"] for a in ARMS])
        if key in seen:
            dup += 1
        else:
            seen[key] = cid
    if dup:
        warnings.append(f"{dup} configurations repeat the messages of an earlier configuration; the runner generates "
                        "identical prompts once, but each remains a separately drawn unit")
    return warnings


def load_bank(path: str | os.PathLike) -> tuple[dict, str]:
    """Parsed and validated bank, and the sha256 of the file bytes."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise BankError(f"cannot read the bank {p}: {exc.strerror or exc}") from exc
    try:
        bank = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BankError(f"{p} is not valid UTF-8 JSON: {exc}") from exc
    validate_bank(bank)
    return bank, sha256_bytes(raw)


def bank_probabilities(bank: Mapping) -> dict[str, float]:
    ids = list(bank["configurations"])
    return {i: float(bank["sampling_probabilities"][i]) for i in ids}


# --------------------------------------------------------------------------------------------
# Sampler, items, registry
# --------------------------------------------------------------------------------------------

def sampler_config_ids(sampler: Mapping, bank: Mapping, bank_file_sha256: str | None = None) -> list[str]:
    """Distinct configuration ids of the discovery draws, then the validation draws, in draw order.

    Verifies that the sampler was drawn from this bank: the bank and probabilities hashes, the bank
    file hash when the sampler records one, and a re-derivation of both draw lists (draws depend on
    ids, probabilities and the label only)."""
    need = ("discovery_config_ids", "validation_config_ids", "sha256", "label", "k_discovery", "k_validation",
            "bank_sha256", "probabilities_sha256")
    missing = [k for k in need if k not in sampler]
    if missing:
        raise BankError(f"sampler manifest lacks keys: {missing}")
    disc, val = sampler["discovery_config_ids"], sampler["validation_config_ids"]
    if not (isinstance(disc, list) and isinstance(val, list) and disc and val
            and all(isinstance(c, str) for c in disc + val)):
        raise BankError("sampler manifest: discovery_config_ids and validation_config_ids must be non-empty lists of ids")
    probs = bank_probabilities(bank)
    ids = list(probs)
    if sampler["bank_sha256"] != sampling.sha256_json(ids):
        raise BankError("the sampler was not drawn from this bank: its bank_sha256 differs from the hash of the "
                        "bank's configuration ids")
    if sampler["probabilities_sha256"] != sampling.sha256_json(np.asarray(list(probs.values()), float).tolist()):
        raise BankError("the sampler was not drawn with this bank's sampling probabilities")
    rec_file = sampler.get("bank_file_sha256")
    if rec_file is not None and bank_file_sha256 is not None and rec_file != bank_file_sha256:
        raise BankError("the sampler records a different bank file hash (bank_file_sha256); the bank file changed "
                        "after the sampler was frozen")
    try:
        redo = sampling.registered_split_bank(ids, list(probs.values()), int(sampler["k_discovery"]),
                                              int(sampler["k_validation"]), str(sampler["label"]))
    except (ValueError, TypeError) as exc:
        raise BankError(f"cannot re-derive the sampler draws: {exc}") from exc
    if (redo["discovery_config_ids"] != disc or redo["validation_config_ids"] != val
            or redo["sha256"] != sampler["sha256"]):
        raise BankError("the sampler's draws differ from a re-derivation from the bank, its probabilities and the "
                        "label; the manifest was edited or drawn from other inputs")
    return list(dict.fromkeys(disc + val))


def check_split(split: Any) -> str:
    if split not in SPLITS:
        raise ScoreInputError(f"split must be one of {SPLITS}, got {split!r}")
    return split


def load_items(items_parquet: str | os.PathLike | None = None,
               manifest_path: str | os.PathLike | None = None,
               split: str = "test") -> tuple[pd.DataFrame, dict]:
    """GSM8K items of one split (item_id, question, gold_answer) and a description for the records.

    Without `items_parquet` the pinned split ("test", the default, or "train") is read from
    data/raw/gsm8k (never downloaded here) and its canonical sha256 must equal the one for that split
    in manifests/gsm8k_items.json. A custom parquet (columns item_id, question, gold_answer) is for
    dry runs and is labelled as such; `split` then only names what the parquet stands for."""
    check_split(split)
    if items_parquet:
        try:
            df = pd.read_parquet(items_parquet)
        except (OSError, ValueError) as exc:
            raise ScoreInputError(f"cannot read {items_parquet}: {exc}") from exc
        missing = [c for c in ("item_id", "question", "gold_answer") if c not in df.columns]
        if missing:
            raise ScoreInputError(f"{items_parquet} lacks columns {missing}")
        df = df[["item_id", "question", "gold_answer"]].copy()
        df["item_id"] = df["item_id"].astype("int64")
        df = df.sort_values("item_id", kind="stable").reset_index(drop=True)
        info = {"source": f"custom parquet (dry run, not the pinned GSM8K {split} split)", "split": split,
                "path": rel_path(items_parquet), "n_items": int(len(df)),
                "canonical_sha256": gsm8k.canonical_sha256(df),
                "manifest_canonical_sha256": None, "matches_manifest": None}
        return df, info
    try:
        df = gsm8k.load_gsm8k(split)
    except FileNotFoundError as exc:
        raise ScoreInputError(f"{exc}; run `.\\.venv\\Scripts\\python.exe -m src.gsm8k` (small public parquet files)") from exc
    mp = Path(manifest_path) if manifest_path else Path(gsm8k.MANIFEST_PATH)
    try:
        want = json.loads(mp.read_text(encoding="utf-8"))["splits"][split]["canonical_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ScoreInputError(f"cannot read the items manifest {mp} ({exc})") from exc
    got = gsm8k.canonical_sha256(df)
    if got != want:
        raise ScoreInputError(f"GSM8K {split} items differ from {mp.name}: canonical sha256 {got} != {want}")
    info = {"source": f"{gsm8k.DATASET} {split} @ {gsm8k.REVISION}", "split": split,
            "path": rel_path(gsm8k.local_path(split)), "n_items": int(len(df)), "canonical_sha256": got,
            "manifest_canonical_sha256": want, "matches_manifest": True}
    return df, info


# --------------------------------------------------------------------------------------------
# Calibration items (GSM8K training split), for admissibility screening and noise tranches
# --------------------------------------------------------------------------------------------

def calibration_sample(item_ids: Sequence[int], n: int) -> list[int]:
    """Sorted sample of n distinct item ids, drawn from the ids alone (never from questions, answers or
    outputs) with sampling.rng_for("gsm8k-calibration-items", n): Generator.choice without replacement
    over the sorted ids."""
    ids = sorted({int(i) for i in item_ids})
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= int(n) <= len(ids):
        raise BankError(f"--calibration-items must be an integer between 1 and {len(ids)}, got {n!r}")
    rng = sampling.rng_for(CALIBRATION_LABEL, int(n))
    pick = rng.choice(np.asarray(ids, dtype=np.int64), size=int(n), replace=False)
    return sorted(int(i) for i in pick)


def _calibration_body_sha(rec: Mapping) -> str:
    return sampling.sha256_json({k: v for k, v in rec.items() if k not in ("sha256", "created")})


def _new_calibration_record(items_df: pd.DataFrame, items_info: Mapping, n: int) -> dict:
    rec = {
        "split": "train", "dataset": gsm8k.DATASET, "revision": gsm8k.REVISION, "n_items": int(n),
        "seed_label": CALIBRATION_LABEL, "rng_labels": [CALIBRATION_LABEL, int(n)],
        "master_seed": sampling.MASTER_SEED,
        "sampling": "numpy Generator.choice without replacement over the sorted train item ids, then sorted; "
                    "metadata only (item ids), never outcomes",
        "population": {"n_items": int(len(items_df)), "canonical_sha256": items_info["canonical_sha256"],
                       "source": items_info["source"]},
        "item_ids": calibration_sample(items_df["item_id"], n)}
    rec["sha256"] = _calibration_body_sha(rec)
    rec["created"] = utc_now()
    return rec


def _calibration_problem(rec: Any, items_df: pd.DataFrame, items_info: Mapping, n: int) -> str | None:
    """Why an existing calibration manifest cannot be reused for this population and N, or None."""
    if not isinstance(rec, dict):
        return "it is not a JSON object"
    if rec.get("split") != "train":
        return f"its split is {rec.get('split')!r}, not 'train'"
    if rec.get("n_items") != n or not isinstance(rec.get("item_ids"), list) or len(rec["item_ids"]) != n:
        return f"it holds {rec.get('n_items')!r} items, not the requested {n}"
    if rec.get("revision") != gsm8k.REVISION:
        return f"its dataset revision {rec.get('revision')!r} differs from the pinned {gsm8k.REVISION!r}"
    pop = rec.get("population") or {}
    if pop.get("canonical_sha256") != items_info["canonical_sha256"]:
        return "it was drawn from different train items (population canonical sha256 differs)"
    if rec.get("sha256") != _calibration_body_sha(rec):
        return "its sha256 does not match its content (edited after it was written)"
    if rec["item_ids"] != calibration_sample(items_df["item_id"], n):
        return "its item ids differ from the deterministic draw (edited, or drawn under another seed)"
    return None


def select_calibration_items(items_df: pd.DataFrame, items_info: Mapping, n: int,
                             manifest_path: str | os.PathLike | None = None, overwrite: bool = False
                             ) -> tuple[pd.DataFrame, dict, dict]:
    """The calibration sample of N train items: (items, items info for the records, the manifest).

    The sample is written once to `manifest_path` (default manifests/gsm8k_calibration_items.json) and
    reused by every later run. A manifest for a different N, a different population or with edited
    content is refused unless `overwrite`. Returns the sampled items (sorted by item_id), an items
    description in the load_items schema (canonical sha256 of the sample, plus the population and the
    manifest), and the manifest with a "status" key ("written", "reused", "overwritten")."""
    if items_info.get("split") != "train":
        raise BankError("calibration items are drawn from the GSM8K train split only")
    path = Path(manifest_path) if manifest_path else CALIBRATION_MANIFEST
    status = "written"
    rec = None
    if path.exists():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        problem = "it is not readable JSON" if rec is None else _calibration_problem(rec, items_df, items_info, n)
        if problem is None:
            status = "reused"
        elif overwrite:
            rec, status = None, "overwritten"
        else:
            raise BankError(f"{rel_path(path)} cannot be reused: {problem}. The sample is fixed once drawn; pass "
                            "--overwrite only to replace it deliberately (earlier screening and tranche records "
                            "then describe other items)")
    if rec is None:
        rec = _new_calibration_record(items_df, items_info, n)
        _write_json(path, rec)
    ids = [int(i) for i in rec["item_ids"]]
    sub = items_df[items_df["item_id"].isin(ids)].sort_values("item_id", kind="stable").reset_index(drop=True)
    if len(sub) != n:
        raise BankError(f"the calibration sample has {len(sub)} items in the loaded train split, expected {n}")
    info = {**items_info, "n_items": int(n), "source": f"{items_info['source']}, calibration sample of {n} items",
            "canonical_sha256": gsm8k.canonical_sha256(sub),
            "population_n_items": int(len(items_df)), "population_canonical_sha256": items_info["canonical_sha256"],
            "calibration_items": {"path": rel_path(path), "sha256": rec["sha256"], "n_items": int(n),
                                  "seed_label": CALIBRATION_LABEL, "status": status}}
    return sub, info, {**rec, "status": status}


def unregistered_models(model_keys: Sequence[str], models_path: str | os.PathLike | None = None) -> list[str]:
    """Keys that are not entries of config/models.yaml dataset2.models (schema templates never count)."""
    try:
        reg = gen.load_registry(models_path)
    except Exception as exc:  # OSError, yaml.YAMLError, gen.RunRefused
        raise BankError(f"cannot read the model registry: {exc}") from exc
    return [k for k in model_keys if k.startswith("_") or k not in reg]


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def substitute_question(messages: Sequence[Mapping], question: str) -> list[dict]:
    """Replace the literal token in each message with str.replace; every other brace is untouched."""
    return [{"role": m["role"], "content": m["content"].replace(QUESTION_TOKEN, question)} for m in messages]


def validate_rendered(df: pd.DataFrame, model_key: str, decoding: gen.Decoding | None = None) -> None:
    """The runner's checks on a request frame. With a Decoding the runner's own validate_requests runs
    (unique ids, one model, arms known to the decoding config, parseable messages)."""
    missing = [c for c in gen.REQUEST_COLUMNS if c not in df.columns]
    if missing:
        raise BankError(f"request frame lacks columns {missing}")
    try:
        if decoding is not None:
            gen.validate_requests(df, model_key, decoding)
        else:
            if df["request_id"].duplicated().any():
                raise gen.RunRefused("request_id is not unique")
            other = sorted(set(df["model_key"]) - {model_key})
            if other:
                raise gen.RunRefused(f"one model per run: the frame also contains model_key {other[:5]}")
            for rid, s in zip(df["request_id"], df["messages_json"]):
                try:
                    gen.parse_messages(s)
                except (ValueError, TypeError) as exc:
                    raise gen.RunRefused(f"request {rid}: invalid messages_json ({exc})") from exc
    except gen.RunRefused as exc:
        raise BankError(f"rendered requests fail the runner's checks: {exc}") from exc


def request_id_for(config_id: str, arm: str, item_id: int, split: str = "test") -> str:
    """f"{config_id}|{arm}|{item_id}" for the test split; f"train|{config_id}|{arm}|{item_id}" for train,
    so that a request id never repeats across splits (config ids contain no '|')."""
    base = f"{config_id}|{arm}|{item_id}"
    return base if split == "test" else f"{split}|{base}"


def render_requests(bank: Mapping, config_ids: Sequence[str], model_key: str, items_df: pd.DataFrame,
                    arms: Sequence[str] = ARMS, decoding: gen.Decoding | None = None,
                    split: str = "test", string_item_ids: bool = False) -> pd.DataFrame:
    """One row per distinct configuration x arm x item, in that order, with the runner's columns.

    request_id is f"{config_id}|{arm}|{item_id}" for the test split and f"train|{config_id}|{arm}|{item_id}"
    for the train split. Repeated configuration ids (a unit drawn twice) are rendered once. The bank is
    validated first. item_id is int64 (GSM8K row positions) unless `string_item_ids`, which keeps string ids
    (BIG-Bench Hard "<task>/<index>", src/bbh_d2.py); a string id must not contain "|"."""
    validate_bank(bank)
    check_model_key(model_key)
    check_split(split)
    cids = list(dict.fromkeys(str(c) for c in config_ids))
    if not cids:
        raise BankError("no configuration ids to render")
    unknown = [c for c in cids if c not in bank["configurations"]]
    if unknown:
        raise BankError(f"{len(unknown)} configuration ids are not in the bank, e.g. {unknown[:3]}")
    arms = tuple(dict.fromkeys(arms))
    bad = [a for a in arms if a not in ARMS]
    if not arms or bad:
        raise BankError(f"arms must be a non-empty subset of {ARMS}, got {list(arms)}")
    for col in ("item_id", "question"):
        if col not in items_df.columns:
            raise BankError(f"items lack the column {col!r}")
    if len(items_df) == 0:
        raise BankError("no items to render")
    if items_df["item_id"].duplicated().any():
        raise BankError("item_id is not unique")
    if not all(isinstance(q, str) for q in items_df["question"]):
        raise BankError("every question must be a string")
    if string_item_ids:
        item_ids = [str(i) for i in items_df["item_id"]]
        bad_ids = [i for i in item_ids if not i or "|" in i or i != i.strip()]
        if bad_ids:
            raise BankError(f"string item ids must be non-empty, without '|' and surrounding whitespace, e.g. {bad_ids[:3]}")
    else:
        item_ids = [int(i) for i in items_df["item_id"]]
    questions = list(items_df["question"])

    rows: dict[str, list] = {c: [] for c in gen.REQUEST_COLUMNS}
    for cid in cids:
        for arm in arms:
            msgs = bank["configurations"][cid]["arms"][arm]["messages"]
            for iid, q in zip(item_ids, questions):
                rows["request_id"].append(request_id_for(cid, arm, iid, split))
                rows["model_key"].append(model_key)
                rows["arm"].append(arm)
                rows["config_id"].append(cid)
                rows["item_id"].append(iid)
                rows["messages_json"].append(json.dumps(substitute_question(msgs, q), ensure_ascii=False))
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype(str if string_item_ids else "int64")
    validate_rendered(df, model_key, decoding)
    return df


def describe_requests(df: pd.DataFrame) -> dict:
    return {"n_requests": int(len(df)), "n_configurations": int(df["config_id"].nunique()),
            "n_items": int(df["item_id"].nunique()),
            "counts_by_arm": {a: int(n) for a, n in sorted(df["arm"].value_counts().items())},
            "n_distinct_prompts_by_arm": {a: int(g["messages_json"].nunique()) for a, g in sorted(df.groupby("arm"))}}


# --------------------------------------------------------------------------------------------
# Request files and the build record
# --------------------------------------------------------------------------------------------

def write_requests_file(df: pd.DataFrame, path: Path, overwrite: bool = False) -> str:
    """Write the parquet. An existing file with identical content is left alone ("unchanged"); a different
    one is refused unless `overwrite`. Returns "written", "overwritten" or "unchanged"."""
    status = "written"
    if path.exists():
        try:
            old = pd.read_parquet(path)
        except Exception:
            old = None
        if old is not None and old.reset_index(drop=True).equals(df.reset_index(drop=True)):
            return "unchanged"
        if not overwrite:
            raise BankError(f"{path} exists with different content; pass --overwrite to replace it (generation "
                            "outputs keyed to the old request ids would no longer match)")
        status = "overwritten"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return status


def inputs_fingerprint(header: Mapping) -> str:
    """Hash of the inputs that determine the request ids and texts (not of paths or times)."""
    keep = {"kind": header["kind"], "bank_sha256": header["bank"]["sha256"],
            "sampler_registered_sha256": (header.get("sampler") or {}).get("registered_sha256"),
            "sampler_file_sha256": (header.get("sampler") or {}).get("file_sha256"),
            "items_canonical_sha256": header["items"]["canonical_sha256"],
            "configuration_ids": header["configuration_ids"], "arms": header["arms"]}
    if header.get("split", "test") != "test":      # test-split records keep their earlier fingerprint
        keep["split"] = header["split"]
        keep["calibration_items_sha256"] = ((header["items"].get("calibration_items") or {}).get("sha256"))
    return sampling.sha256_json(keep)


def merge_build_record(path: Path, header: dict, models: Mapping[str, dict], overwrite: bool = False) -> dict:
    """Write the build record. A record with the same input fingerprint gains the new model entries
    (the earlier entries and the original creation time are kept); different inputs are refused
    unless `overwrite`."""
    header = dict(header)
    header["inputs_fingerprint"] = inputs_fingerprint(header)
    record = None
    if path.exists():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if record is None and not overwrite:
            raise BankError(f"{path} exists but is not readable JSON; pass --overwrite to replace it")
        if record is not None and record.get("inputs_fingerprint") != header["inputs_fingerprint"]:
            if not overwrite:
                raise BankError(f"{path} describes different inputs (bank, sampler, items or configurations); "
                                "pass --overwrite to start a new record")
            record = None
    if record is None:
        record = {**header, "created": utc_now(), "models": {}}
    else:
        record.update({k: v for k, v in header.items() if k != "created"})
    for key, entry in models.items():
        old = record["models"].get(key)
        record["models"][key] = {**entry, "created": old["created"]} if old and old.get("requests_sha256") == entry["requests_sha256"] else entry
    record["contains_unregistered_models"] = any(not e.get("registered", True) for e in record["models"].values())
    record["updated"] = utc_now()
    _write_json(path, record)
    return record


def load_build_records(requests_dir: str | os.PathLike) -> list[dict]:
    """Records found in <dir>/build_record.json, <dir>/tranche/build_record.json and
    <dir>/screening/build_record.json, each with its path. Records written before the split field
    existed are test-split records."""
    d = Path(requests_dir)
    out = []
    for p in (d / BUILD_RECORD, d / TRANCHE_DIR / BUILD_RECORD, d / SCREENING_DIR / BUILD_RECORD):
        if p.is_file():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ScoreInputError(f"build record {p} cannot be read ({exc})") from exc
            rec["_path"] = str(p)
            out.append(rec)
    return out


# --------------------------------------------------------------------------------------------
# Generation folders
# --------------------------------------------------------------------------------------------

def runs_cover_file(runs: Sequence[Mapping]) -> bool:
    """True when one run without --limit or --part completed, or the finished parts (part_complete) cover the file.

    Part K of N holds the requests whose hash is K - 1 modulo N (gen.select_part). Parts 1..N of one N cover the file;
    so do parts of several N when every residue modulo the least common multiple of their N falls in one of them
    (for example, part K/N may be replaced by K/2N and (K+N)/2N, which hold exactly its requests)."""
    if any(r.get("status") == "complete" and r.get("limit") is None and not r.get("part") for r in runs):
        return True
    finished: set[tuple[int, int]] = set()
    for r in runs:
        if r.get("status") == "part_complete" and r.get("limit") is None and r.get("part"):
            k, n = r["part"]
            finished.add((int(k), int(n)))
    if not finished:
        return False
    by_n: dict[int, set[int]] = {}
    for k, n in finished:
        by_n.setdefault(n, set()).add(k)
    if any(ks == set(range(1, n + 1)) for n, ks in by_n.items()):
        return True
    m = math.lcm(*by_n)
    if m > 100_000:   # parts of splits that share no small multiple; only the whole-split rule above applies
        return False
    return all(any(res % n == k - 1 for k, n in finished) for res in range(m))


def _read_log(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScoreInputError(f"run log {path} cannot be read ({exc})") from exc


def inspect_work_dir(work_dir: str | os.PathLike, *, allow_test_backend: bool = False,
                     allow_incomplete: bool = False, string_item_ids: bool = False,
                     close_stopped: bool = True) -> tuple[pd.DataFrame, dict]:
    """Outputs of one model's generation folder, after the checks that keep tranche-throughput data out.

    A prefilled arm's outputs are returned as the assistant's whole message, the prefill followed by the output. Where
    the arm's logged decoding stops at the first "}" (the forced arms of GSM8K and BIG-Bench Hard), a reply that the
    stop left with its box open is read as the closed box it was meant to be (src/forced_stop.py); pass
    close_stopped=False for the text as generated (counts of the forms).

    The runner marks throughput output three ways: the folder is `<out-dir>/throughput`, its logs
    are throughput_log-*.json / throughput_report-*.json, and its shards carry the parquet metadata
    throughput=true (its run log also has mode throughput and excluded_from_main true). Any of
    these refuses the folder. The main out-dir may contain a throughput/ subfolder; it is not read.
    item_id is read as int64, or as a string with `string_item_ids` (BIG-Bench Hard)."""
    d = Path(work_dir)
    if not d.is_dir():
        raise ScoreInputError(f"{d} is not a directory")
    if d.name == "throughput":
        raise ScoreInputError(f"{d} is a throughput folder; throughput outputs are excluded from main "
                              "estimation. Pass the parent out-dir of a full run")
    stray = sorted(p.name for pat in ("throughput_log-*.json", "throughput_report-*.json") for p in d.glob(pat))
    if stray:
        raise ScoreInputError(f"{d} holds throughput logs {stray[:2]}; throughput outputs are excluded from "
                              "main estimation")
    shards = gen.list_shards(d)
    if not shards:
        raise ScoreInputError(f"{d} holds no shard-*.parquet")
    metas = []
    for sh in shards:
        try:
            md = pq.read_schema(sh).metadata or {}
        except Exception as exc:
            raise ScoreInputError(f"{sh} is unreadable ({exc})") from exc
        if md.get(b"throughput") == b"true":
            raise ScoreInputError(f"{sh.name} is marked as a throughput shard; throughput outputs are excluded from "
                                  "main estimation")
        metas.append(md)
    fingerprints = {m.get(b"run_fingerprint", b"").decode() for m in metas}
    if len(fingerprints) != 1:
        raise ScoreInputError(f"{d}: shards were written under {len(fingerprints)} different model/decoding "
                              "configurations")
    logs = sorted(d.glob("run_log-*.json"))
    if not logs:
        raise ScoreInputError(f"{d} holds no run_log-*.json; the log carries the pins")
    runs = []
    for lp in logs:
        log = _read_log(lp)
        if log.get("mode") == "throughput" or log.get("excluded_from_main") is True:
            raise ScoreInputError(f"{lp.name} is a throughput-mode log; throughput outputs are excluded from "
                                  "main estimation")
        inp, model = log.get("inputs") or {}, log.get("model") or {}
        runs.append({"log_file": lp.name, "run_id": log.get("run_id"), "status": log.get("status"),
                     "started_utc": log.get("started_utc"), "ended_utc": log.get("ended_utc"),
                     "model_key": model.get("model_key"), "hf_id": model.get("hf_id"),
                     "revision": model.get("revision"), "backend": (log.get("backend") or {}).get("name"),
                     "decoding_confirmed": log.get("decoding_confirmed"), "run_fingerprint": log.get("run_fingerprint"),
                     "chat_template": log.get("chat_template"), "limit": inp.get("limit"),
                     "requests_sha256": inp.get("requests_sha256"), "requests_path": inp.get("requests_path"),
                     "part": tuple(inp["part"]) if inp.get("part") else None,
                     "runtime_s": log.get("runtime_s"), "totals": log.get("totals"),
                     "gpu_type": list(gen.gpu_type(log.get("gpu"))), "wrote_shards": bool(log.get("shards_written")),
                     "prefill": dict((log.get("decoding") or {}).get("prefill") or {}),
                     "stops_at_brace": sorted(a for a in ((log.get("decoding") or {}).get("prefill") or {})
                                              if forced_stop.stops_at_brace(log.get("decoding") or {}, a))})
    runs.sort(key=lambda r: (r["started_utc"] or "", r["run_id"] or ""))
    shas = {r["requests_sha256"] for r in runs}
    if len(shas) != 1 or None in shas:
        raise ScoreInputError(f"{d}: its runs used {len(shas)} different request files (or the log lacks the hash); "
                              "a folder belongs to one request file. Generate into a fresh folder")
    # Run logs have one-second timestamps, so their order is not reliable. Completeness is read from the set
    # of runs: a complete run without --limit has generated every request id of the file, and so have
    # finished --part runs that together cover every residue (gen.select_part; runs_cover_file).
    if not runs_cover_file(runs) and not allow_incomplete:
        raise ScoreInputError(f"{d}: no complete run without --limit and no finished set of parts (runs: "
                              f"{[(r['status'], r['limit'], r['part']) for r in runs]}); resume it to completion "
                              "(or pass --allow-incomplete for a diagnostic table)")
    gpu_types = sorted({tuple(r["gpu_type"]) for r in runs if r["gpu_type"] and r["wrote_shards"]})
    if len(gpu_types) > 1:
        raise ScoreInputError(f"{d}: its outputs were generated on {len(gpu_types)} GPU models {gpu_types}; a model's "
                              "outputs come from one GPU model (hardware is pinned). Regenerate on one GPU type")
    test_backend = any(r["backend"] in TEST_BACKENDS for r in runs)
    if test_backend and not allow_test_backend:
        raise ScoreInputError(f"{d} was produced by a test backend (echo); pass --allow-test-backend for dry runs")

    out = gen.read_outputs(d)
    absent = [c for c in gen.OUTPUT_COLUMNS if c not in out.columns]
    if absent:
        raise ScoreInputError(f"{d}: outputs lack columns {absent}")
    keys = sorted(out["model_key"].astype(str).unique())
    if len(keys) != 1:
        raise ScoreInputError(f"{d} must hold exactly one model; found {keys[:5]}")
    model_key = keys[0]
    meta_keys = {m.get(b"model_key", b"").decode() for m in metas}
    if meta_keys != {model_key}:
        raise ScoreInputError(f"{d}: shard metadata model_key {sorted(meta_keys)} disagrees with the rows ({model_key!r})")
    if out["request_id"].duplicated().any():
        raise ScoreInputError(f"{d}: request_id is duplicated across shards")
    out = out.copy()
    out["config_id"] = out["config_id"].astype(str)
    out["model_key"] = out["model_key"].astype(str)
    out["item_id"] = out["item_id"].astype(str if string_item_ids else "int64")
    # A prefilled assistant turn (decoding prefill): the engine returns only the
    # continuation, so the assistant's message is the prefill followed by the output.
    prefills = {json.dumps(r["prefill"], sort_keys=True) for r in runs}
    if len(prefills) != 1:
        raise ScoreInputError(f"{d}: its runs used {len(prefills)} different prefills; a folder belongs to one decoding")
    prefill = json.loads(next(iter(prefills)))
    stop_sets = {tuple(r["stops_at_brace"]) for r in runs}
    if len(stop_sets) != 1:
        raise ScoreInputError(f"{d}: its runs stop different arms at '}}'; a folder belongs to one decoding")
    stop_arms = list(next(iter(stop_sets)))
    n_closed = {}
    for arm, text in prefill.items():
        m = out["arm"] == arm
        full = text + out.loc[m, "output_text"].astype(str)
        if close_stopped and arm in stop_arms:
            closed = full.map(lambda t, p=text: forced_stop.close(t, p))
            n_closed[arm] = int((closed != full).sum())
            full = closed
        out.loc[m, "output_text"] = full
    info = {"path": str(d), "model_key": model_key, "n_outputs": int(len(out)), "n_shards": len(shards),
            "prefill": prefill,
            "forced_stop": {"version": forced_stop.VERSION, "arms_stopped_at_brace": stop_arms,
                            "applied": bool(close_stopped), "replies_closed": n_closed},
            "run_fingerprint": next(iter(fingerprints)), "runs": runs, "test_backend": test_backend,
            "incomplete_allowed": bool(allow_incomplete),
            "gpu_type": list(gpu_types[0]) if gpu_types else [],
            "decoding_confirmed": all(r["decoding_confirmed"] is True for r in runs),
            "has_throughput_subfolder": (d / "throughput").is_dir()}
    return out, info


def match_requests(info: dict, records: Sequence[dict]) -> dict:
    """Link a folder to the build record that produced its request file (run-log requests_sha256).

    Returns {"kind", "split", "purpose", "model_key", "record_path", "bank_sha256", "sampler_registered_sha256",
    "requests_sha256", ...} or {"kind": None} when no record matches. A folder whose runs match records of
    different kinds, splits or models is refused."""
    by_sha: dict[str, dict] = {}
    for rec in records:
        for key, ent in (rec.get("models") or {}).items():
            by_sha[ent.get("requests_sha256")] = {
                "kind": rec.get("kind"), "split": rec.get("split", "test"), "purpose": rec.get("purpose"),
                "model_key": key, "record_path": rec.get("_path"),
                "bank_sha256": (rec.get("bank") or {}).get("sha256"),
                "sampler_registered_sha256": (rec.get("sampler") or {}).get("registered_sha256"),
                "sampler_file_sha256": (rec.get("sampler") or {}).get("file_sha256"),
                "items_canonical_sha256": (rec.get("items") or {}).get("canonical_sha256"),
                "requests_sha256": ent.get("requests_sha256"), "excluded_from_main": rec.get("excluded_from_main"),
                "calibration_items": (rec.get("items") or {}).get("calibration_items")}
    hits = [by_sha.get(r["requests_sha256"]) for r in info["runs"]]
    if any(h is None for h in hits):
        return {"kind": None}
    kinds = {(h["kind"], h["split"]) for h in hits}
    if len(kinds) != 1 or {h["model_key"] for h in hits} != {info["model_key"]}:
        raise ScoreInputError(f"{info['path']}: its runs match build records of different kinds, splits or models "
                              f"({sorted(map(str, kinds))})")
    return hits[-1]


def run_request_hashes(work_dir: str | os.PathLike) -> set[str]:
    """requests_sha256 values recorded in the run logs of a generation folder (run logs only; no shard is read)."""
    out: set[str] = set()
    for lp in sorted(Path(work_dir).glob("run_log-*.json")):
        sha = (_read_log(lp).get("inputs") or {}).get("requests_sha256")
        if sha:
            out.add(sha)
    return out


def resolve_split(work_dirs: Sequence[str | os.PathLike], records: Sequence[dict],
                  requested: str | None = None) -> str:
    """The GSM8K split a scoring run uses: the split of the build records that match the folders' request
    files (records without a split field are test-split records). `requested` (--split) must agree with
    them; folders matching records of two splits are refused. Without a matching record the requested
    split applies, else "test"."""
    if requested is not None:
        check_split(requested)
    by_sha = {ent.get("requests_sha256"): rec.get("split", "test")
              for rec in records for ent in (rec.get("models") or {}).values()}
    found: set[str] = set()
    for wd in work_dirs:
        if not Path(wd).is_dir():
            continue                                  # inspect_work_dir reports a missing folder
        found |= {by_sha[sha] for sha in run_request_hashes(wd) if sha in by_sha}
    if len(found) > 1:
        raise ScoreInputError(f"the folders match build records of different splits {sorted(found)}; train and test "
                              "outputs are never scored together")
    if found:
        split = next(iter(found))
        if requested is not None and requested != split:
            raise ScoreInputError(f"--split {requested} contradicts the build record of these folders (split {split})")
        return split
    return requested or "test"


def scored_metadata(path: str | os.PathLike) -> dict[str, str]:
    """The d2_* key-value metadata of a scored table (empty for a parquet without it)."""
    md = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode("utf-8") for k, v in md.items() if k.startswith(b"d2_")}


# --------------------------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------------------------

def load_registered_scorer(path: str | os.PathLike) -> tuple[scoring.FormatSpec, dict]:
    """The scorer manifest (src.scoring.scorer_record schema; `spec` is accepted as an alias of
    format_spec, as src/gate.py does). Raises ScoreInputError when absent or inconsistent."""
    p = Path(path)
    if not p.is_file():
        raise ScoreInputError(f"{p} does not exist: the scorer is not registered yet. For a dry "
                              "run use --candidate-spec NAME")
    raw = p.read_bytes()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ScoreInputError(f"{p} is not valid JSON: {exc}") from exc
    spec_dict = obj.get("format_spec", obj.get("spec")) if isinstance(obj, dict) else None
    if not isinstance(spec_dict, dict) or not spec_dict:
        raise ScoreInputError(f"{p} lacks format_spec (or spec)")
    if obj.get("scorer_version") != scoring.SCORER_VERSION:
        raise ScoreInputError(f"{p}: scorer_version {obj.get('scorer_version')!r} differs from src/scoring.py "
                              f"{scoring.SCORER_VERSION!r}")
    try:
        spec = scoring.spec_from_dict(spec_dict)
    except (ValueError, TypeError) as exc:
        raise ScoreInputError(f"{p}: format_spec cannot be loaded by src/scoring.py: {exc}") from exc
    fp = scoring.spec_fingerprint(spec)
    if obj.get("format_spec_sha256") is not None and obj["format_spec_sha256"] != fp:
        raise ScoreInputError(f"{p}: format_spec_sha256 does not match the recorded format_spec")
    record = scoring.scorer_record(spec)
    record.update({"status": REGISTERED_STATUS, "manifest_path": rel_path(p), "manifest_sha256": sha256_bytes(raw)})
    return spec, record


def candidate_scorer(name: str) -> tuple[scoring.FormatSpec, dict]:
    if name not in scoring.CANDIDATE_SPECS:
        raise ScoreInputError(f"unknown candidate spec {name!r}; candidates: {sorted(scoring.CANDIDATE_SPECS)}")
    spec = scoring.CANDIDATE_SPECS[name]
    record = scoring.scorer_record(spec)
    record.update({"status": CANDIDATE_STATUS})
    return spec, record


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def score_work_dirs(work_dirs: Sequence[str | os.PathLike], spec: scoring.FormatSpec, gold: Mapping, *,
                    records: Sequence[dict], tranche: bool, screening: bool = False, split: str = "test",
                    allow_test_backend: bool = False, allow_incomplete: bool = False,
                    allow_unmatched: bool = False, score_fn=None,
                    string_item_ids: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Score each folder separately (request ids repeat across models), then stack the rows.

    Returns the table with scored_columns(kind) (a screening table has no correctness column), sorted by
    (model_key, arm, config_id, item_id), and one info
    dict per folder. `gold` must be the gold answers of `split`. Refuses a folder whose request file
    belongs to another kind (main, tranche, screening) or another split, a folder without a
    matching build record (unless allow_unmatched), a train-split scoring run that is neither tranche nor
    screening, and duplicated (model_key, arm, config_id, item_id) rows. `score_fn(outputs)` replaces
    scoring.score_outputs(outputs, gold, spec) when given (BIG-Bench Hard: src/bbh_d2.py); it must add the
    columns of scoring.SCORE_COLUMNS. `string_item_ids` reads item ids as strings."""
    check_split(split)
    if tranche and screening:
        raise ScoreInputError("a scoring run is either a tranche or a screening run, not both")
    want = KIND_SCREENING if screening else KIND_TRANCHE if tranche else KIND_MAIN
    if split == "train" and want == KIND_MAIN:
        raise ScoreInputError("train-split outputs are calibration data and cannot be a main table; "
                              "score them with --screening or --tranche")
    parts, infos, seen = [], [], {}
    for wd in work_dirs:
        out, info = inspect_work_dir(wd, allow_test_backend=allow_test_backend, allow_incomplete=allow_incomplete,
                                     string_item_ids=string_item_ids)
        mk = info["model_key"]
        if mk in seen:
            raise ScoreInputError(f"model {mk!r} appears in two folders ({seen[mk]} and {wd}); the table would hold "
                                  "duplicated (model_key, arm, config_id, item_id) rows")
        seen[mk] = str(wd)
        m = match_requests(info, records)
        if m["kind"] is None:
            if not allow_unmatched:
                raise ScoreInputError(f"{wd}: no build record matches the request file of its run log (searched "
                                      f"{len(records)} records). Build the requests with scripts/build_requests.py and pass "
                                      "--requests-dir, or pass --allow-unmatched-requests for a diagnostic table")
        elif m["kind"] != want:
            flag = {KIND_TRANCHE: " (add --tranche)", KIND_SCREENING: " (add --screening)",
                    KIND_MAIN: " (drop --tranche/--screening)"}.get(m["kind"], "")
            raise ScoreInputError(f"{wd}: its request file was built as {m['kind']!r} but this scoring run is "
                                  f"{want!r}{flag}; main, tranche and screening outputs are never mixed "
                                  "")
        elif m["split"] != split:
            raise ScoreInputError(f"{wd}: its request file was built on the {m['split']} split but this scoring run "
                                  f"uses the {split} split gold answers")
        info["requests_match"] = m
        try:
            scored = scoring.score_outputs(out, gold, spec) if score_fn is None else score_fn(out)
        except KeyError as exc:
            raise ScoreInputError(f"{wd}: {exc.args[0] if exc.args else exc}") from exc
        except ValueError as exc:
            raise ScoreInputError(f"{wd}: {exc}") from exc
        parts.append(scored[list(scored_columns(want))])
        info["n_scored"] = int(len(scored))
        infos.append(info)
    table = pd.concat(parts, ignore_index=True)
    dup = table.duplicated(KEY_COLUMNS)
    if dup.any():
        ex = table.loc[dup, KEY_COLUMNS].head(3).values.tolist()
        raise ScoreInputError(f"{int(dup.sum())} duplicated (model_key, arm, config_id, item_id) rows, e.g. {ex}")
    table = table.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    return table, infos


def scored_columns(kind: str) -> tuple[str, ...]:
    """Columns of a scored table of this kind."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return SCREENING_COLUMNS if kind == KIND_SCREENING else SCORED_COLUMNS


def score_summary(table: pd.DataFrame, kind: str = KIND_MAIN) -> list[dict]:
    """Per model and arm: rows, compliance (percent), outputs cut at the cap (finish_reason length), total output
    tokens and cached rows. Strict and lenient accuracy are added for a tranche table only. A main table
    pools discovery and validation units, so its accuracy would show validation outcomes before the prediction
    rules are saved (scripts/run_dataset2.py reports them after the rules are saved). A screening table has
    no correctness. Descriptive only."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    out = []
    for (mk, arm), g in table.groupby(["model_key", "arm"], sort=True):
        n = len(g)
        row = {"model_key": mk, "arm": arm, "n_rows": int(n), "n_configurations": int(g["config_id"].nunique()),
               "n_items": int(g["item_id"].nunique())}
        if kind == KIND_TRANCHE:
            row.update({"strict_correct_pct": 100.0 * float(g["strict_correct"].mean()),
                        "lenient_correct_pct": 100.0 * float(g["lenient_correct"].mean())})
        row.update({"compliant_pct": 100.0 * float(g["compliant"].mean()),
                    "n_finish_length": int((g["finish_reason"] == "length").sum()),
                    "n_output_tokens_total": int(g["n_output_tokens"].sum()),
                    "n_cached_rows": int(g["cached"].sum())})
        out.append(row)
    return out


def write_scored(table: pd.DataFrame, path: str | os.PathLike, metadata: Mapping[str, str]) -> Path:
    """Parquet with the pandas schema plus d2_* key-value metadata (all string values); atomic."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(table, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta.update({k.encode(): str(v).encode("utf-8") for k, v in metadata.items()})
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(t.replace_schema_metadata(meta), tmp, compression="zstd")
    os.replace(tmp, p)
    return p


def sidecar_path(scored_path: str | os.PathLike) -> Path:
    p = Path(scored_path)
    return p.with_name(p.stem + ".meta.json")


def write_sidecar(path: Path, obj: Mapping) -> None:
    _write_json(path, obj)


def compact_input(info: dict) -> dict:
    """The part of a folder's info that goes into the parquet key-value metadata."""
    m = info.get("requests_match") or {}
    return {"model_key": info["model_key"], "work_dir": info["path"], "n_outputs": info["n_outputs"],
            "run_ids": [r["run_id"] for r in info["runs"]], "run_fingerprint": info["run_fingerprint"],
            "requests_sha256": sorted({r["requests_sha256"] for r in info["runs"] if r["requests_sha256"]}),
            "request_kind": m.get("kind"), "split": m.get("split"), "calibration_items": m.get("calibration_items"),
            "bank_sha256": m.get("bank_sha256"),
            "sampler_registered_sha256": m.get("sampler_registered_sha256"),
            "test_backend": info["test_backend"], "decoding_confirmed": info["decoding_confirmed"],
            "incomplete_allowed": info.get("incomplete_allowed"), "gpu_type": info.get("gpu_type")}


# --------------------------------------------------------------------------------------------
# Guards for the registered analysis and the frozen bank
# --------------------------------------------------------------------------------------------

def main_problems(md: Mapping[str, str], *, sampler: Mapping | None = None,
                          items_info: Mapping | None = None,
                          scorer_path: str | os.PathLike | None = None,
                          items_label: str = "the pinned GSM8K test items") -> list[str]:
    """Reasons a scored table cannot enter the analysis (scripts/run_dataset2.py); empty when it can.

    `md` is scored_metadata() of the table. Checked: the table labels (kind main, split test, registered
    scorer) and, from d2_inputs, every generation folder behind the table: a matching main build record on
    the test split, no test backend, a complete run (no --allow-incomplete) and confirmed decoding. With `sampler`
    every folder must come from requests built with that sampler. With `items_info` (load_items for the
    test split) the scored items must be the pinned test items. With `scorer_path` the scorer manifest must be the
    one the table was scored with."""
    problems = [f"{k} is {md.get(k)!r}, expected {want!r}" for k, want in (
        ("d2_kind", KIND_MAIN), ("d2_split", "test"), ("d2_scorer_status", REGISTERED_STATUS))
        if md.get(k) != want]
    try:
        inputs = json.loads(md["d2_inputs"])
        if not isinstance(inputs, list) or not inputs or not all(isinstance(i, dict) for i in inputs):
            raise ValueError("no folder listed")
    except (KeyError, ValueError, TypeError):
        return problems + ["d2_inputs is absent or unreadable, so the generation folders behind the table are unknown"]
    for i in inputs:
        name = i.get("model_key") or i.get("work_dir")
        if i.get("test_backend") is not False:
            problems.append(f"{name}: outputs from a test backend (echo)")
        if i.get("request_kind") != KIND_MAIN:
            problems.append(f"{name}: request kind {i.get('request_kind')!r}; a main table needs requests "
                            "matched to a main build record (it was scored with --allow-unmatched-requests "
                            "or from another kind)")
        if i.get("split") != "test":
            problems.append(f"{name}: requests built on the {i.get('split')!r} split, not test")
        if i.get("incomplete_allowed") is not False:
            problems.append(f"{name}: scored with --allow-incomplete (or the metadata predates the field)")
        if i.get("decoding_confirmed") is not True:
            problems.append(f"{name}: generated under unconfirmed decoding")
        if sampler is not None and i.get("sampler_registered_sha256") != sampler.get("sha256"):
            problems.append(f"{name}: requests built with sampler {i.get('sampler_registered_sha256')!r}, not the "
                            f"expected sampler {sampler.get('sha256')!r}")
    if items_info is not None:
        try:
            got = json.loads(md["d2_items"]).get("canonical_sha256")
        except (KeyError, ValueError, TypeError, AttributeError):
            got = None
        if got != items_info.get("canonical_sha256"):
            problems.append(f"the table was scored against items with canonical sha256 {got!r}, not {items_label} "
                            f"({items_info.get('canonical_sha256')!r})")
    if scorer_path is not None:
        try:
            rec_sha = json.loads(md["d2_scorer_record"]).get("manifest_sha256")
        except (KeyError, ValueError, TypeError, AttributeError):
            rec_sha = None
        p = Path(scorer_path)
        current = sha256_path(p) if p.is_file() else None
        if rec_sha is None or rec_sha != current:
            problems.append(f"the scorer manifest {rel_path(p)} (sha256 {current!r}) is not the one the table was "
                            f"scored with ({rec_sha!r})")
    return problems


def load_admissibility(path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    """The exclusion log that scripts/screen_d2.py writes under confirmed rules (manifests/gsm8k_admissibility.json)
    and its provenance {path, sha256}. BankError when it is absent or malformed."""
    p = Path(path) if path else ADMISSIBILITY_LOG
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise BankError(f"the admissibility exclusion log {rel_path(p)} does not exist ({exc.strerror or exc}); run "
                        "the screening with confirmed rules first (scripts/screen_d2.py)") from exc
    try:
        log = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise BankError(f"{rel_path(p)} is not valid JSON: {exc}") from exc
    need = ("models_screened", "excluded_models", "retained_models", "configurations_screened",
            "excluded_configurations", "retained_configuration_ids", "configuration_rules_evaluated")
    missing = [k for k in need if not isinstance(log, dict) or k not in log]
    if missing:
        raise BankError(f"{rel_path(p)} lacks {missing}; it is not an exclusion log written by scripts/screen_d2.py")
    return log, {"path": rel_path(p), "sha256": sha256_bytes(raw)}


def bank_admissibility_problems(bank: Mapping, log: Mapping) -> list[str]:
    """Reasons the bank is not the retained bank of the exclusion log (exclusions are logged and the
    target is scoped to the retained bank). Every configuration must have been screened and retained, and every retained
    configuration must be in the bank: dropping one would be a choice the rules did not make."""
    ids = set(map(str, bank.get("configurations") or {}))
    retained = set(map(str, log.get("retained_configuration_ids") or []))
    excluded = {str(c.get("config_id")) for c in log.get("excluded_configurations") or []}
    problems = []
    if log.get("configuration_rules_evaluated") is not True:
        problems.append("the exclusion log evaluated no configuration rule (no model was retained)")
    for label, bad in (("excluded by the admissibility rules", ids & excluded),
                       ("never screened", ids - retained - excluded),
                       ("retained by the screening but missing from the bank", retained - ids)):
        if bad:
            ex = sorted(bad)
            problems.append(f"{len(ex)} configurations {label}, e.g. {ex[:3]}")
    return problems


SAMPLING_LAWS = ("uniform",)          # config/promptsuite.yaml: the only law implemented


def frozen_bank(candidate: Mapping, candidate_sha: str, log: Mapping, *, candidate_path: str,
                log_info: Mapping) -> dict:
    """The bank to freeze (manifests/gsm8k_bank.json) from a screened candidate bank and its exclusion log.

    Keeps exactly the retained configurations, in the candidate's order, and sets their sampling probabilities
    under the law recorded in the candidate's generation record (only "uniform", 1/n over the retained bank, is
    implemented). Refuses (BankError) an example candidate, a candidate built from an unconfirmed configuration,
    a log that screened another bank or only part of this one, and a log without evaluated configuration rules.
    The result records where it came from: candidate path and sha256, the log's path and sha256, the excluded
    configuration ids and the law."""
    gen_rec = candidate.get("generation_record") or {}
    problems = []
    if "EXAMPLE" in str(candidate.get("generator", "")).upper():
        problems.append("the candidate is an example bank (its generator says so), not one built from the confirmed design")
    if gen_rec.get("config_status") != "confirmed":
        problems.append(f"the candidate was built from a configuration with status {gen_rec.get('config_status')!r}, "
                        "not 'confirmed'")
    law = gen_rec.get("sampling_law")
    if law not in SAMPLING_LAWS:
        problems.append(f"sampling law {law!r} is not one of {SAMPLING_LAWS}")
    screened_sha = (log.get("inputs") or {}).get("bank_sha256")
    if screened_sha != candidate_sha:
        problems.append(f"the exclusion log screened the bank {screened_sha!r}, not this candidate ({candidate_sha})")
    ids = [str(c) for c in candidate["configurations"]]
    screened = set(map(str, log.get("configurations_screened") or []))
    if screened != set(ids):
        problems.append(f"the screening covered {len(screened & set(ids))} of the candidate's {len(ids)} "
                        f"configurations (and {len(screened - set(ids))} others); the whole candidate must be screened")
    if log.get("configuration_rules_evaluated") is not True:
        problems.append("the exclusion log evaluated no configuration rule (no model was retained)")
    if problems:
        raise BankError("cannot freeze the bank: " + "; ".join(problems))
    retained = set(map(str, log["retained_configuration_ids"]))
    keep = [c for c in ids if c in retained]
    if not keep:
        raise BankError("cannot freeze the bank: no configuration was retained")
    bank = {k: v for k, v in candidate.items() if k not in ("configurations", "sampling_probabilities")}
    bank["configurations"] = {c: candidate["configurations"][c] for c in keep}
    bank["sampling_probabilities"] = {c: 1.0 / len(keep) for c in keep}
    bank["frozen_from"] = {
        "candidate_path": candidate_path, "candidate_sha256": candidate_sha,
        "admissibility": {"path": log_info.get("path"), "sha256": log_info.get("sha256")},
        "n_candidate_configurations": len(ids), "n_retained": len(keep),
        "excluded_configuration_ids": [c for c in ids if c not in retained],
        "sampling_law": law, "sampling_law_rule": "uniform over the retained configurations"}
    validate_bank(bank)
    assert not bank_admissibility_problems(bank, log)
    return bank


def panel_admissibility_problems(panel: Sequence[str], log: Mapping) -> list[str]:
    """Reasons a model list is not admissible under the exclusion log: a model the direct-answer rule excluded
    or a model that was never screened."""
    screened = set(log.get("models_screened") or [])
    excluded = {m.get("model_key") for m in log.get("excluded_models") or []}
    problems = []
    for k in panel:
        if k in excluded:
            problems.append(f"{k}: excluded by the admissibility rules (no meaningful direct-answer condition)")
        elif k not in screened:
            problems.append(f"{k}: never screened for admissibility")
    return problems
