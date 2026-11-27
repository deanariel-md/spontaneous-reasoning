r"""Admissibility screening of Dataset 2 prompt configurations and models.

Screening runs on calibration items from the GSM8K TRAINING split (scripts/build_requests.py
--screening --split train), never on target items. Configurations and models are not chosen on
outcomes, so this module reports FORMAT AND BEHAVIOUR metrics only. It never computes or
emits correctness, accuracy or gains: it does not call src.scoring.score_outputs, it reads none of the
strict_correct / lenient_correct columns of the scored table, and `assert_no_outcome_keys` stops any
output that carries such a key.

Metrics, per (model, configuration, arm) cell, per (configuration, arm) pooled over models and per
(model, arm) pooled over configurations (shares are percentages of the n outputs in the group):
  n                       outputs in the group
  strict_compliance_pct   strict extraction succeeds, under EVERY spec in scoring.CANDIDATE_SPECS (read at
                          run time) and under the chosen spec when manifests/gsm8k_scorer.json exists
  lenient_extractable_pct the output contains a number token (scoring.lenient_extract is not None)
  truncated_pct           finish_reason is "length" (cut at the output cap)
  think_tag_pct           the output contains a <think> or </think> tag
  empty_pct               the output is empty after stripping whitespace
  output_tokens_mean, _median, _p95   of n_output_tokens (p95: numpy percentile, linear interpolation)
Direct arm only, "meaningful direct answer" diagnostic: the share of outputs longer than 32, 64 and 128
tokens (n_output_tokens > L). A model that reasons at length in the direct arm has no meaningful
direct-answer condition.

Rules (config/admissibility.yaml; thresholds are null until confirmed):
  models           excluded when the direct-arm share of outputs above `length_tokens` exceeds
                   `max_share_above_pct` (no meaningful direct answer), or when the think-tag share of all the
                   model's screened outputs exceeds `thinking_off_holds.max_think_tag_pct` (native reasoning is
                   not off); one exclusion entry per model and rule
  configurations   evaluated per (configuration, arm), pooled over the models that were not excluded;
                   excluded when strict compliance falls more than `max_compliance_shortfall_pp`
                   percentage points below the median configuration of that arm, or the truncated share or the
                   think-tag share is above its maximum (a value equal to the threshold passes). The compliance
                   rule is relative: pooled over models, compliance mostly measures how
                   well the panel follows the format, which the strict outcome scores, so the
                   screen removes the configurations that follow it markedly worse than the rest of the bank
While the status is PENDING_REVIEW only the metrics report is written. When it is `confirmed` the
exclusion log (manifests/gsm8k_admissibility.json) is written as well: each excluded model or
configuration with the rule and the metric value, and the retained ids.

BIG-Bench Hard (task "bbh"; src/bbh_d2.py): the same rules and metrics on the 200 screening
items, with compliance meaning that src/bbh.py's strict extraction succeeds for the item's answer
type and the lenient-extractable share counting outputs with a token of that type. The GSM8K candidate formats
are number formats and are not reported for it. Item ids are strings; the report and the exclusion log name
the task.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from . import d2_io, generation as gen, scoring

ROOT = d2_io.ROOT
RULES_PATH = ROOT / "config" / "admissibility.yaml"
SCORER_PATH = ROOT / "manifests" / "gsm8k_scorer.json"
EXCLUSION_LOG = ROOT / "manifests" / "gsm8k_admissibility.json"
REPORT_DIR = ROOT / "results" / "screening"
SCORED_DEFAULT = ROOT / "data" / "processed" / "gsm8k" / "screening_scored.parquet"

SCHEMA_VERSION = "1"
DIRECT_ARM = "direct_answer"
DIRECT_LENGTHS = (32, 64, 128)
MODEL_RULES = ("direct_answer_meaningful", "thinking_off_holds")
STATUS_PENDING, STATUS_CONFIRMED = "PENDING_REVIEW", "confirmed"
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)      # the pattern of the runner's think-tag count
OUTCOME_KEY_RE = re.compile(r"correct|accura|gain", re.IGNORECASE)
KEY = ["model_key", "arm", "config_id", "item_id"]
# The only columns read from the scored table: no correctness column is ever loaded.
SCORED_READ_COLUMNS = KEY + ["finish_reason", "n_output_tokens", "prompt_sha256"]
RAW_COLUMNS = KEY + ["prompt_sha256", "n_output_tokens", "finish_reason", "output_text"]
CONFIG_ARMS = d2_io.ARMS


class ScreeningError(d2_io.D2IOError):
    """An input does not qualify for screening; the script prints it as REFUSED and exits 2."""


# --------------------------------------------------------------------------------------------
# Guard: no outcome in any output
# --------------------------------------------------------------------------------------------

def assert_no_outcome_keys(obj: Any, where: str = "report") -> None:
    """Raise when a mapping key (at any depth) or a DataFrame column mentions correctness, accuracy or gain."""
    if isinstance(obj, pd.DataFrame):
        keys: list = list(obj.columns)
        nested: list = []
    elif isinstance(obj, Mapping):
        keys, nested = list(obj.keys()), list(obj.values())
    elif isinstance(obj, (list, tuple)):
        keys, nested = [], list(obj)
    else:
        return
    for k in keys:
        if isinstance(k, str) and OUTCOME_KEY_RE.search(k):
            raise ScreeningError(f"{where}: key {k!r} names an outcome; screening reports format and behaviour "
                                 "metrics only (outcomes do not guide the choice)")
    for v in nested:
        assert_no_outcome_keys(v, where)


# --------------------------------------------------------------------------------------------
# Rules file
# --------------------------------------------------------------------------------------------

def _is_pct(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 100


def validate_rules(obj: Any) -> list[str]:
    """Problems with a parsed rules file (empty list when it is well formed). A `confirmed` file must carry
    every threshold: a null would silently disable a rule, so it is refused."""
    if not isinstance(obj, dict):
        return ["the rules file must be a mapping"]
    problems: list[str] = []
    extra = sorted(set(obj) - {"status", "configurations", "models"})
    if extra:
        problems.append(f"unknown top-level keys {extra}")
    status = obj.get("status")
    if status not in (STATUS_PENDING, STATUS_CONFIRMED):
        problems.append(f"status must be {STATUS_PENDING} or {STATUS_CONFIRMED}, got {status!r}")
    nulls: list[str] = []

    cfg = obj.get("configurations")
    names = ("max_compliance_shortfall_pp", "max_truncation_pct", "max_think_tag_pct")
    if not isinstance(cfg, dict):
        problems.append("configurations must be a mapping")
        cfg = {}
    for k in sorted(set(cfg) - set(names)):
        problems.append(f"configurations: unknown rule {k!r}")
    for name in names:
        per_arm = cfg.get(name)
        if not isinstance(per_arm, dict) or set(per_arm) != set(CONFIG_ARMS):
            problems.append(f"configurations.{name} must be a mapping with exactly the arms {list(CONFIG_ARMS)}")
            continue
        for arm in CONFIG_ARMS:
            v = per_arm[arm]
            if v is None:
                nulls.append(f"configurations.{name}.{arm}")
            elif not _is_pct(v):
                problems.append(f"configurations.{name}.{arm} must be a percentage between 0 and 100 or null, got {v!r}")

    mod = obj.get("models")
    if not isinstance(mod, dict) or set(mod) != set(MODEL_RULES):
        problems.append(f"models must be a mapping with exactly the rules {list(MODEL_RULES)}")
        mod = {}
    think = mod.get("thinking_off_holds")
    if mod and (not isinstance(think, dict) or set(think) != {"max_think_tag_pct"}):
        problems.append("models.thinking_off_holds must have exactly max_think_tag_pct")
    elif mod:
        if think["max_think_tag_pct"] is None:
            nulls.append("models.thinking_off_holds.max_think_tag_pct")
        elif not _is_pct(think["max_think_tag_pct"]):
            problems.append("models.thinking_off_holds.max_think_tag_pct must be a percentage between 0 and 100 "
                            f"or null, got {think['max_think_tag_pct']!r}")
    rule = mod.get("direct_answer_meaningful")
    if not isinstance(rule, dict) or set(rule) != {"length_tokens", "max_share_above_pct"}:
        problems.append("models.direct_answer_meaningful must have exactly length_tokens and max_share_above_pct")
        rule = {}
    if rule:
        if rule["length_tokens"] is None:
            nulls.append("models.direct_answer_meaningful.length_tokens")
        elif isinstance(rule["length_tokens"], bool) or rule["length_tokens"] not in DIRECT_LENGTHS:
            problems.append(f"models.direct_answer_meaningful.length_tokens must be one of {list(DIRECT_LENGTHS)} "
                            f"(the lengths the diagnostic reports) or null, got {rule['length_tokens']!r}")
        if rule["max_share_above_pct"] is None:
            nulls.append("models.direct_answer_meaningful.max_share_above_pct")
        elif not _is_pct(rule["max_share_above_pct"]):
            problems.append("models.direct_answer_meaningful.max_share_above_pct must be a percentage between 0 "
                            f"and 100 or null, got {rule['max_share_above_pct']!r}")
    if status == STATUS_CONFIRMED and nulls:
        problems.append(f"status is {STATUS_CONFIRMED} but these thresholds are null: {nulls}. Set each one (to "
                        "disable a rule use its trivial bound: 0 for a minimum, 100 for a maximum)")
    return problems


def load_rules(path: str | os.PathLike | None = None) -> tuple[dict, dict]:
    """(rules, provenance) from config/admissibility.yaml; ScreeningError when absent or malformed."""
    p = Path(path) if path else RULES_PATH
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ScreeningError(f"cannot read the rules file {p}: {exc.strerror or exc}") from exc
    try:
        obj = yaml.safe_load(raw.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ScreeningError(f"{p} is not valid YAML: {exc}") from exc
    problems = validate_rules(obj)
    if problems:
        raise ScreeningError(f"admissibility rules {p} invalid: " + "; ".join(problems))
    return obj, {"path": d2_io.rel_path(p), "sha256": d2_io.sha256_bytes(raw), "status": obj["status"]}


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

def load_scored(path: str | os.PathLike, task: str = "gsm8k") -> tuple[pd.DataFrame, dict]:
    """The screening scored table (key, finish_reason, n_output_tokens, prompt_sha256 only) and its d2_* metadata.
    Refused unless it is a train-split screening table of the task."""
    p = Path(path)
    if not p.is_file():
        raise ScreeningError(f"{p} does not exist; score the screening outputs with scripts/score_d2.py --screening")
    try:
        md = d2_io.scored_metadata(p)
    except Exception as exc:
        raise ScreeningError(f"{p} is not a readable parquet ({exc})") from exc
    if md.get("d2_kind") != d2_io.KIND_SCREENING:
        raise ScreeningError(f"{p} is a {md.get('d2_kind')!r} table (d2_kind), not a screening table; screening reads "
                             "only outputs scored with scripts/score_d2.py --screening")
    if md.get("d2_split") != "train":
        raise ScreeningError(f"{p} was scored on the {md.get('d2_split')!r} split; admissibility is screened on "
                             "training items only")
    if md.get("d2_task", "gsm8k") != task:
        raise ScreeningError(f"{p} is a {md.get('d2_task', 'gsm8k')!r} table (d2_task), not a {task!r} table")
    table = pd.read_parquet(p, columns=SCORED_READ_COLUMNS)
    table["config_id"] = table["config_id"].astype(str)
    table["item_id"] = table["item_id"].astype(str if task != "gsm8k" else "int64")
    if table.duplicated(KEY).any():
        raise ScreeningError(f"{p} has duplicated (model_key, arm, config_id, item_id) rows")
    if table.empty:
        raise ScreeningError(f"{p} has no rows")
    return table, md


def work_dirs_from_metadata(md: Mapping[str, str]) -> list[str]:
    try:
        inputs = json.loads(md["d2_inputs"])
        return [str(i["work_dir"]) for i in inputs]
    except (KeyError, ValueError, TypeError) as exc:
        raise ScreeningError("the scored table's metadata does not list its generation folders; pass --work-dir") from exc


def load_raw_outputs(work_dirs: Sequence[str | os.PathLike], string_item_ids: bool = False) -> pd.DataFrame:
    parts = []
    for wd in work_dirs:
        d = Path(wd)
        if not d.is_dir():
            raise ScreeningError(f"generation folder {d} does not exist; pass the folders with --work-dir")
        raw = gen.read_outputs(d)
        absent = [c for c in RAW_COLUMNS if c not in raw.columns]
        if absent:
            raise ScreeningError(f"{d}: outputs lack columns {absent}")
        raw = raw[RAW_COLUMNS].copy()
        raw["config_id"] = raw["config_id"].astype(str)
        raw["item_id"] = raw["item_id"].astype(str if string_item_ids else "int64")
        parts.append(raw)
    if not parts:
        raise ScreeningError("no generation folder")
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(KEY).any():
        raise ScreeningError("the generation folders hold duplicated (model_key, arm, config_id, item_id) rows "
                             "(one model per folder)")
    return out


def attach_outputs(table: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Join the raw outputs to the scored table on the key. Every scored row needs its output with the same prompt
    hash, token count and finish reason; the folders must not hold outputs the table lacks."""
    merged = table.merge(raw, on=KEY, how="left", suffixes=("", "_raw"), validate="one_to_one")
    missing = merged["prompt_sha256_raw"].isna()
    if missing.any():
        ex = merged.loc[missing, KEY].head(3).values.tolist()
        raise ScreeningError(f"{int(missing.sum())} scored rows have no output in the generation folders, e.g. {ex}")
    for col in ("prompt_sha256", "n_output_tokens", "finish_reason"):
        bad = merged[col] != merged[col + "_raw"]
        if bad.any():
            ex = merged.loc[bad, KEY].head(3).values.tolist()
            raise ScreeningError(f"{int(bad.sum())} rows disagree with the scored table on {col}, e.g. {ex}; the "
                                 "folders are not the ones that were scored")
    if len(raw) != len(table):
        raise ScreeningError(f"the generation folders hold {len(raw)} outputs but the scored table {len(table)} rows")
    return merged[KEY + ["finish_reason", "n_output_tokens", "output_text"]]


# --------------------------------------------------------------------------------------------
# Row metrics (format and behaviour only)
# --------------------------------------------------------------------------------------------

def row_metrics(df: pd.DataFrame, candidates: Mapping[str, scoring.FormatSpec],
                registered: scoring.FormatSpec | None = None) -> pd.DataFrame:
    """Boolean behaviour flags per output. `df` needs the key columns, finish_reason, n_output_tokens and output_text.
    Columns: cand__<name> for every candidate spec, registered (when a spec is given), lenient,
    truncated, think_tag, empty. Correctness is never computed."""
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    out = df[KEY + ["n_output_tokens"]].copy().reset_index(drop=True)
    cache: dict[str, dict] = {}
    specs = dict(candidates)
    flags = {f"cand__{n}": [] for n in specs}
    if registered is not None:
        flags["registered"] = []
    flags.update({"lenient": [], "think_tag": [], "empty": []})
    for t in texts:
        r = cache.get(t)
        if r is None:
            r = {f"cand__{n}": scoring.strict_extract(t, s) is not None for n, s in specs.items()}
            if registered is not None:
                r["registered"] = scoring.strict_extract(t, registered) is not None
            r["lenient"] = scoring.lenient_extract(t) is not None
            r["think_tag"] = bool(THINK_TAG_RE.search(t))
            r["empty"] = not t.strip()
            cache[t] = r
        for k, v in r.items():
            flags[k].append(v)
    for k, v in flags.items():
        out[k] = np.asarray(v, dtype=bool)
    out["truncated"] = (df["finish_reason"].astype(str) == "length").to_numpy()
    return out


def _pct(k: int, n: int) -> float:
    return 100.0 * float(k) / float(n)


def summarise(g: pd.DataFrame, arm: str, spec_names: Sequence[str], has_registered: bool) -> dict:
    """Metrics of one group of rows (all from one arm)."""
    n = int(len(g))
    tok = g["n_output_tokens"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "n": n,
        "strict_compliance_pct": {name: _pct(int(g[f"cand__{name}"].sum()), n) for name in spec_names},
        "registered_compliance_pct": _pct(int(g["registered"].sum()), n) if has_registered else None,
        "lenient_extractable_pct": _pct(int(g["lenient"].sum()), n),
        "truncated_pct": _pct(int(g["truncated"].sum()), n),
        "think_tag_pct": _pct(int(g["think_tag"].sum()), n),
        "empty_pct": _pct(int(g["empty"].sum()), n),
        "output_tokens_mean": float(tok.mean()),
        "output_tokens_median": float(np.median(tok)),
        "output_tokens_p95": float(np.percentile(tok, 95)),
        "direct_answer_diagnostic": None}
    if arm == DIRECT_ARM:
        out["direct_answer_diagnostic"] = {
            f"above_{L}_tokens_pct": _pct(int((tok > L).sum()), n) for L in DIRECT_LENGTHS}
    return out


def aggregate(rows: pd.DataFrame, by: Sequence[str], spec_names: Sequence[str], has_registered: bool) -> list[dict]:
    """One record per group of `by` (which must include "arm"), sorted by the group keys."""
    out = []
    for keys, g in rows.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(by, (k.item() if hasattr(k, "item") else k for k in keys)))
        rec.update(summarise(g, rec["arm"], spec_names, has_registered))
        out.append(rec)
    return out


# --------------------------------------------------------------------------------------------
# Rule application
# --------------------------------------------------------------------------------------------

def apply_rules(rows: pd.DataFrame, rules: Mapping, *, has_registered: bool,
                diagnostic_model_rules: Sequence[str] = (), relative_truncation: bool = False) -> dict:
    """Apply a confirmed rules mapping to the row metrics. Models first (direct-answer diagnostic, then think tags
    over all the model's outputs), then the configuration rules per (configuration, arm) pooled over the retained
    models. Returns the exclusion block.

    A model rule named in `diagnostic_model_rules` is computed and reported under model_diagnostics but excludes no
    model (MATH-500: math_d2.DIAGNOSTIC_MODEL_RULES); with none named the block is as before. With
    `relative_truncation` (MATH-500: math_d2.RELATIVE_TRUNCATION) a configuration's
    truncation counts by how far it lies above the arm's median configuration, as compliance does."""
    diagnostic = tuple(diagnostic_model_rules)
    unknown = sorted(set(diagnostic) - {"direct_answer_meaningful", "thinking_off_holds"})
    if unknown:
        raise ScreeningError(f"unknown model rules {unknown} named as diagnostics")
    cfg = rules["configurations"]
    rule = rules["models"]["direct_answer_meaningful"]
    max_think = float(rules["models"]["thinking_off_holds"]["max_think_tag_pct"])
    if not has_registered:
        raise ScreeningError("the rules are confirmed but the scorer manifest (manifests/gsm8k_scorer.json) "
                             "does not exist; strict compliance is judged under the recorded format")
    models = sorted(rows["model_key"].unique())
    configs = sorted(rows["config_id"].unique())
    length, max_share = int(rule["length_tokens"]), float(rule["max_share_above_pct"])

    excluded_models, retained_models, flagged = [], [], []
    for m in models:
        g = rows[(rows["model_key"] == m) & (rows["arm"] == DIRECT_ARM)]
        if g.empty:
            raise ScreeningError(f"model {m!r} has no {DIRECT_ARM} outputs; its direct-answer condition cannot be assessed")
        share = _pct(int((g["n_output_tokens"] > length).sum()), len(g))
        hits = []
        if share > max_share:
            hits.append({
                "model_key": m, "rule": "direct_answer_meaningful", "metric": f"direct_answer_above_{length}_tokens_pct",
                "value": share, "threshold": max_share, "comparison": "value > threshold", "n": int(len(g))})
        a = rows[rows["model_key"] == m]
        think = _pct(int(a["think_tag"].sum()), len(a))
        if think > max_think:
            hits.append({
                "model_key": m, "rule": "thinking_off_holds", "metric": "think_tag_pct_all_outputs",
                "value": think, "threshold": max_think, "comparison": "value > threshold", "n": int(len(a))})
        flagged += [h for h in hits if h["rule"] in diagnostic]
        hits = [h for h in hits if h["rule"] not in diagnostic]
        if hits:
            excluded_models += hits
        else:
            retained_models.append(m)

    violations: dict[str, list] = {}
    evaluated = bool(retained_models)
    if evaluated:
        pooled = rows[rows["model_key"].isin(retained_models)]
        compliance = {key: _pct(int(g["registered"].sum()), len(g))
                      for key, g in pooled.groupby(["config_id", "arm"], sort=True)}
        arm_median = {arm: float(np.median([v for (_, a), v in compliance.items() if a == arm]))
                      for arm in sorted({a for _, a in compliance})}
        truncation = {key: _pct(int(g["truncated"].sum()), len(g))
                      for key, g in pooled.groupby(["config_id", "arm"], sort=True)}
        trunc_median = {arm: float(np.median([v for (_, a), v in truncation.items() if a == arm]))
                        for arm in sorted({a for _, a in truncation})}
        checks = ((("max_think_tag_pct", "think_tag_pct", "think_tag"),) if relative_truncation else
                  (("max_truncation_pct", "truncated_pct", "truncated"),
                   ("max_think_tag_pct", "think_tag_pct", "think_tag")))
        for (cid, arm), g in pooled.groupby(["config_id", "arm"], sort=True):
            n = int(len(g))
            found = []
            thr = float(cfg["max_compliance_shortfall_pp"][arm])
            shortfall = arm_median[arm] - compliance[(cid, arm)]
            if shortfall > thr:
                found.append({"rule": "max_compliance_shortfall_pp", "metric": "registered_compliance_shortfall_pp",
                              "value": shortfall, "threshold": thr, "compliance_pct": compliance[(cid, arm)],
                              "arm_median_compliance_pct": arm_median[arm]})
            if relative_truncation:
                thr = float(cfg["max_truncation_pct"][arm])
                excess = truncation[(cid, arm)] - trunc_median[arm]
                if excess > thr:
                    found.append({"rule": "max_truncation_pct", "metric": "truncated_pct_above_arm_median",
                                  "value": excess, "threshold": thr, "truncated_pct": truncation[(cid, arm)],
                                  "arm_median_truncated_pct": trunc_median[arm]})
            for rule_name, metric, col in checks:
                thr = float(cfg[rule_name][arm])
                value = _pct(int(g[col].sum()), n)
                if value > thr:
                    found.append({"rule": rule_name, "metric": metric, "value": value, "threshold": thr})
            for v in found:
                violations.setdefault(cid, []).append({"arm": arm, **v, "comparison": "value > threshold", "n": n,
                                                       "pooled_over_models": retained_models})
    excluded_configs = [{"config_id": cid, "violations": violations[cid]} for cid in configs if cid in violations]
    extra = {}
    if diagnostic:
        extra = {"model_diagnostics": {
            "rules": list(diagnostic), "excludes_models": False, "flagged": flagged,
            "note": "these model rules are computed and reported only; a flagged model stays in the panel and in the "
                    "pool of the configuration rules"}}
    return {
        **extra,
        "models_screened": models, "excluded_models": excluded_models, "retained_models": retained_models,
        "configuration_rules_evaluated": evaluated,
        "configuration_rules_note": None if evaluated else "no model was retained, so no pooled configuration "
                                                           "metric exists and no configuration was assessed",
        "configurations_screened": configs, "excluded_configurations": excluded_configs,
        "retained_configuration_ids": [c for c in configs if c not in violations] if evaluated else [],
        "conventions": "a model is excluded when its direct-arm share above length_tokens is greater than the maximum, "
                       "or the think-tag share of all its outputs is greater than the thinking_off_holds maximum; "
                       "a configuration-arm is excluded when its compliance falls more than the maximum "
                       "shortfall below the median configuration of that arm (percentage points, pooled over the "
                       "retained models and the screened configurations) or its truncated or think-tag share is above "
                       "its maximum; a value equal to a threshold passes"}


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def _scorer_block(scorer_path: str | os.PathLike | None, explicit: bool) -> tuple[scoring.FormatSpec | None, dict]:
    p = Path(scorer_path) if scorer_path else SCORER_PATH
    if not p.is_file():
        if explicit:
            raise ScreeningError(f"{p} does not exist: the scorer is not recorded")
        return None, {"status": "absent", "path": d2_io.rel_path(p),
                      "note": "manifests/gsm8k_scorer.json does not exist yet; compliance is reported under every "
                              "candidate spec only"}
    try:
        spec, record = d2_io.load_registered_scorer(p)
    except d2_io.ScoreInputError as exc:
        raise ScreeningError(str(exc)) from exc
    return spec, record


def screen_bbh(scored_path: str | os.PathLike | None = None, work_dirs: Sequence[str | os.PathLike] | None = None,
               rules_path: str | os.PathLike | None = None, scorer_path: str | os.PathLike | None = None,
               items_manifest: str | os.PathLike | None = None) -> dict:
    """The BIG-Bench Hard screening report (module doc): the GSM8K report's structure, rules and metrics, with
    compliance and lenient extraction from src/bbh.py by each item's answer type."""
    from . import bbh_d2
    return screen_string_task(bbh_d2, scored_path, work_dirs, rules_path, scorer_path, items_manifest)


def screen_string_task(tm, scored_path: str | os.PathLike | None = None,
                       work_dirs: Sequence[str | os.PathLike] | None = None,
                       rules_path: str | os.PathLike | None = None, scorer_path: str | os.PathLike | None = None,
                       items_manifest: str | os.PathLike | None = None) -> dict:
    """The screening report of a task with fixed item sets (src/string_tasks.py: BIG-Bench Hard or MATH-500),
    with its module's scorer, row metrics and report texts."""
    sp = Path(scored_path) if scored_path else tm.SCORED[d2_io.KIND_SCREENING]
    rules, rules_prov = load_rules(rules_path)
    try:
        scorer_rec = tm.load_scorer(scorer_path)
        types = tm.answer_types(items_manifest)
    except d2_io.D2IOError as exc:
        raise ScreeningError(str(exc)) from exc
    table, md = load_scored(sp, task=tm.TASK)
    dirs = [str(d) for d in work_dirs] if work_dirs else work_dirs_from_metadata(md)
    data = attach_outputs(table, load_raw_outputs(dirs, string_item_ids=True))
    try:
        rows = tm.row_metrics(data, types, THINK_TAG_RE)
    except d2_io.D2IOError as exc:
        raise ScreeningError(str(exc)) from exc
    names: list[str] = []
    inputs_meta = json.loads(md.get("d2_inputs", "[]"))
    bank_shas = sorted({i.get("bank_sha256") for i in inputs_meta if i.get("bank_sha256")})
    try:
        items_md = json.loads(md.get("d2_items", "{}"))
    except ValueError:
        items_md = {}
    n_cell = rows.groupby(["model_key", "arm", "config_id"]).size()
    items_by_cell = rows.groupby(["model_key", "arm", "config_id"])["item_id"].apply(lambda s: tuple(sorted(s)))
    inputs = {
        "task": tm.TASK, "item_set": items_md.get("item_set"),
        "scored_path": d2_io.rel_path(sp), "scored_sha256": d2_io.sha256_path(sp), "split": md.get("d2_split"),
        "scored_scorer_status": md.get("d2_scorer_status"), "work_dirs": [d2_io.rel_path(d) for d in dirs],
        "models": sorted(rows["model_key"].unique()), "arms": sorted(rows["arm"].unique()),
        "n_rows": int(len(rows)), "n_configurations": int(rows["config_id"].nunique()),
        "n_items": int(rows["item_id"].nunique()),
        "n_per_cell_min": int(n_cell.min()), "n_per_cell_max": int(n_cell.max()),
        "item_set_identical_across_cells": bool(items_by_cell.nunique() == 1),
        "bank_sha256": bank_shas[0] if len(bank_shas) == 1 else bank_shas,
        "calibration_items": {"item_set": items_md.get("item_set"), "canonical_sha256": items_md.get("canonical_sha256"),
                              "n_items": items_md.get("n_items"), "table_sha256": items_md.get("table_sha256")},
        "test_backend": any(i.get("test_backend") for i in inputs_meta)}
    confirmed = rules["status"] == STATUS_CONFIRMED
    diagnostic = tuple(getattr(tm, "DIAGNOSTIC_MODEL_RULES", ()))   # MATH-500: reported, not applied
    relative = bool(getattr(tm, "RELATIVE_TRUNCATION", False))       # MATH-500: against the arm's median
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "created": d2_io.utc_now(), "kind": "admissibility_screening",
        "task": tm.TASK,
        "purpose": tm.SCREEN_PURPOSE,
        "inputs": inputs,
        "scorers": {"candidate_specs": names, "candidate_status": d2_io.CANDIDATE_STATUS, "registered": scorer_rec,
                    "scorer_version": tm.SCORER_VERSION},
        "metric_definitions": {
            "units": "shares are percentages of the n outputs in the group; token statistics are in tokens",
            **tm.SCREEN_METRICS,
            "truncated_pct": "finish_reason is length (cut at the output cap)",
            "think_tag_pct": "the output contains <think> or </think>",
            "empty_pct": "the output is empty after stripping whitespace",
            "output_tokens_p95": "numpy percentile, linear interpolation",
            "direct_answer_diagnostic": f"direct arm only: share of outputs longer than {list(DIRECT_LENGTHS)} tokens"},
        "rules": {**rules_prov, "applied": confirmed,
                  "thresholds": {k: rules[k] for k in ("configurations", "models")}},
        "by_model": aggregate(rows, ["model_key", "arm"], names, True),
        "by_configuration": aggregate(rows, ["config_id", "arm"], names, True),
        "cells": aggregate(rows, ["model_key", "config_id", "arm"], names, True),
        "exclusions": apply_rules(rows, rules, has_registered=True, diagnostic_model_rules=diagnostic,
                                  relative_truncation=relative)
                      if confirmed else None}
    if diagnostic:
        report["rules"]["diagnostic_model_rules"] = {"rules": list(diagnostic), "source": tm.DIAGNOSTIC_MODEL_RULES_SOURCE}
    if relative:
        report["rules"]["relative_truncation"] = {
            "rule": "max_truncation_pct is the most a configuration's pooled share cut at the cap may lie above the "
                    "arm's median configuration", "source": tm.RELATIVE_TRUNCATION_SOURCE}
    if not confirmed:
        report["rules"]["note"] = (f"status is {STATUS_PENDING}: no rule was applied and no exclusion log was written. "
                                   "Thresholds are set in config/admissibility.yaml")
    assert_no_outcome_keys(report)
    return report


def screen(scored_path: str | os.PathLike | None = None, work_dirs: Sequence[str | os.PathLike] | None = None,
           rules_path: str | os.PathLike | None = None, scorer_path: str | os.PathLike | None = None,
           task: str = "gsm8k") -> dict:
    """Run the screening and return the report mapping (JSON-serialisable; write it with write_report)."""
    if task == "bbh":
        return screen_bbh(scored_path, work_dirs, rules_path, scorer_path)
    if task in ("math", "hotpot"):         # MATH-500 and HotpotQA
        from . import string_tasks
        return screen_string_task(string_tasks.module(task), scored_path, work_dirs, rules_path, scorer_path)
    if task != "gsm8k":
        raise ScreeningError(f"task must be gsm8k, bbh, math or hotpot, got {task!r}")
    sp = Path(scored_path) if scored_path else SCORED_DEFAULT
    rules, rules_prov = load_rules(rules_path)
    registered, scorer_rec = _scorer_block(scorer_path, explicit=scorer_path is not None)
    if rules["status"] == STATUS_CONFIRMED and registered is None:
        raise ScreeningError("the rules are confirmed but the scorer manifest "
                             "(manifests/gsm8k_scorer.json) does not exist; strict compliance is judged under the "
                             "recorded format")
    table, md = load_scored(sp)
    dirs = [str(d) for d in work_dirs] if work_dirs else work_dirs_from_metadata(md)
    data = attach_outputs(table, load_raw_outputs(dirs))
    candidates = dict(scoring.CANDIDATE_SPECS)
    names = list(candidates)
    rows = row_metrics(data, candidates, registered)
    has_reg = registered is not None

    inputs_meta = json.loads(md.get("d2_inputs", "[]"))
    bank_shas = sorted({i.get("bank_sha256") for i in inputs_meta if i.get("bank_sha256")})
    cal = [i.get("calibration_items") for i in inputs_meta if i.get("calibration_items")]
    n_cell = rows.groupby(["model_key", "arm", "config_id"]).size()
    items_by_cell = rows.groupby(["model_key", "arm", "config_id"])["item_id"].apply(lambda s: tuple(sorted(s)))
    inputs = {
        "scored_path": d2_io.rel_path(sp), "scored_sha256": d2_io.sha256_path(sp), "split": md.get("d2_split"),
        "scored_scorer_status": md.get("d2_scorer_status"), "work_dirs": [d2_io.rel_path(d) for d in dirs],
        "models": sorted(rows["model_key"].unique()), "arms": sorted(rows["arm"].unique()),
        "n_rows": int(len(rows)), "n_configurations": int(rows["config_id"].nunique()),
        "n_items": int(rows["item_id"].nunique()),
        "n_per_cell_min": int(n_cell.min()), "n_per_cell_max": int(n_cell.max()),
        "item_set_identical_across_cells": bool(items_by_cell.nunique() == 1),
        "bank_sha256": bank_shas[0] if len(bank_shas) == 1 else bank_shas,
        "calibration_items": cal[0] if cal else None,
        "test_backend": any(i.get("test_backend") for i in inputs_meta)}

    confirmed = rules["status"] == STATUS_CONFIRMED
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "created": d2_io.utc_now(), "kind": "admissibility_screening",
        "purpose": "Format and behaviour metrics of the prompt configurations and models on GSM8K training "
                   "(calibration) items. No correctness, accuracy or gain is computed.",
        "inputs": inputs,
        "scorers": {"candidate_specs": names,
                    "candidate_status": d2_io.CANDIDATE_STATUS, "registered": scorer_rec,
                    "scorer_version": scoring.SCORER_VERSION},
        "metric_definitions": {
            "units": "shares are percentages of the n outputs in the group; token statistics are in tokens",
            "strict_compliance_pct": "strict extraction succeeds under the named FormatSpec",
            "lenient_extractable_pct": "scoring.lenient_extract finds a number token",
            "truncated_pct": "finish_reason is length (cut at the output cap)",
            "think_tag_pct": "the output contains <think> or </think>",
            "empty_pct": "the output is empty after stripping whitespace",
            "output_tokens_p95": "numpy percentile, linear interpolation",
            "direct_answer_diagnostic": f"direct arm only: share of outputs longer than {list(DIRECT_LENGTHS)} tokens"},
        "rules": {**rules_prov, "applied": confirmed,
                  "thresholds": {k: rules[k] for k in ("configurations", "models")}},
        "by_model": aggregate(rows, ["model_key", "arm"], names, has_reg),
        "by_configuration": aggregate(rows, ["config_id", "arm"], names, has_reg),
        "cells": aggregate(rows, ["model_key", "config_id", "arm"], names, has_reg),
        "exclusions": apply_rules(rows, rules, has_registered=has_reg) if confirmed else None}
    if not confirmed:
        report["rules"]["note"] = (f"status is {STATUS_PENDING}: no rule was applied and no exclusion log was written. "
                                   "Thresholds are set in config/admissibility.yaml")
    assert_no_outcome_keys(report)
    return report


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

def _f(x: Any, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(report: Mapping) -> str:
    inp, names = report["inputs"], report["scorers"]["candidate_specs"]
    reg = report["scorers"]["registered"]
    titles = {"bbh": "BIG-Bench Hard", "math": "MATH-500", "hotpot": "HotpotQA"}
    bbh_task = report.get("task") in titles
    lines = ["# Admissibility screening report" + (f" ({titles[report['task']]})" if bbh_task else ""), "",
             f"Created {report['created']}. Split {inp['split']}, {inp['n_items']} "
             f"{'screening' if bbh_task else 'calibration'} items, "
             f"{len(inp['models'])} models ({', '.join(inp['models'])}), {inp['n_configurations']} configurations, "
             f"{inp['n_rows']} outputs.", "",
             report["purpose"], "",
             f"Scorer: {reg.get('status')}"
             + (f" (format spec sha256 {reg['format_spec_sha256'][:12]}...)" if reg.get("format_spec_sha256") else "")
             + f". Candidate specs: {', '.join(names)}. Scorer version {report['scorers']['scorer_version']}.",
             f"Rules: {report['rules']['status']} ({report['rules']['path']}). "
             + ("Applied." if report["rules"]["applied"] else report["rules"].get("note", "Not applied.")), ""]
    if inp.get("test_backend"):
        lines += ["The outputs come from a test backend; the figures describe the echo backend only.", ""]

    def table(records, first):
        head = first + ["n", "registered %"] + [f"{n} %" for n in names] + [
            "lenient %", "truncated %", "think tag %", "empty %", "tok mean", "tok median", "tok p95",
            "direct >32 %", "direct >64 %", "direct >128 %"]
        out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        for r in records:
            d = r["direct_answer_diagnostic"] or {}
            cells = [str(r[k]) for k in first] + [str(r["n"]), _f(r["registered_compliance_pct"])]
            cells += [_f(r["strict_compliance_pct"][n]) for n in names]
            cells += [_f(r["lenient_extractable_pct"]), _f(r["truncated_pct"]), _f(r["think_tag_pct"]),
                      _f(r["empty_pct"]), _f(r["output_tokens_mean"]), _f(r["output_tokens_median"]),
                      _f(r["output_tokens_p95"])]
            cells += [_f(d.get(f"above_{L}_tokens_pct")) for L in DIRECT_LENGTHS]
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines += ["## By model and arm", ""] + table(report["by_model"], ["model_key", "arm"]) + [""]
    lines += ["## By configuration and arm, pooled over models", ""] + table(
        report["by_configuration"], ["config_id", "arm"]) + [""]
    lines += [f"The per model, configuration and arm table ({len(report['cells'])} rows) is in the JSON report.", ""]
    ex = report.get("exclusions")
    if ex:
        lines += ["## Exclusions", "",
                  f"Excluded models: {', '.join(dict.fromkeys(m['model_key'] for m in ex['excluded_models'])) or 'none'}.",
                  f"Excluded configurations: {', '.join(c['config_id'] for c in ex['excluded_configurations']) or 'none'}.",
                  f"Retained configurations: {len(ex['retained_configuration_ids'])} of {len(ex['configurations_screened'])}.", ""]
        diag = ex.get("model_diagnostics")
        if diag:
            flagged = ", ".join(f"{h['model_key']} ({h['rule']}, {_f(h['value'])} %)" for h in diag["flagged"]) or "none"
            lines += [f"Model rules reported as diagnostics only, excluding no model: {', '.join(diag['rules'])}. "
                      f"Flagged: {flagged}.", ""]
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_report(report: Mapping, out_dir: str | os.PathLike | None = None) -> dict[str, Path]:
    """screening_report.json, .md and the identical _clean.md (no inline citation markers exist in it)."""
    d = Path(out_dir) if out_dir else REPORT_DIR
    paths = {"json": d / "screening_report.json", "md": d / "screening_report.md",
             "clean_md": d / "screening_report_clean.md"}
    _write_text(paths["json"], json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md = render_markdown(report) + "\n"
    _write_text(paths["md"], md)
    _write_text(paths["clean_md"], md)
    return paths


def build_exclusion_log(report: Mapping, report_json: Path | None = None) -> dict:
    """The manifest written when the rules are confirmed (manifests/gsm8k_admissibility.json). `report_json`
    adds the path and sha256 of the written report; without it the log holds the substance only."""
    ex = report["exclusions"]
    if ex is None:
        raise ScreeningError("the rules are not confirmed; there is no exclusion log to write")
    log = {
        "schema_version": SCHEMA_VERSION, "created": report["created"], "kind": "admissibility_exclusion_log",
        "plan_reference": "admissibility screening of Dataset 2 prompt configurations", "split": report["inputs"]["split"],
        "rules": report["rules"], "scorer": report["scorers"]["registered"],
        "inputs": {k: report["inputs"][k] for k in ("scored_path", "scored_sha256", "bank_sha256", "calibration_items",
                                                    "models", "n_items", "n_per_cell_min", "n_per_cell_max",
                                                    "item_set_identical_across_cells", "test_backend")},
        **ex}
    if report.get("task"):
        log = {"task": report["task"], **log}
    if report_json is not None:
        log["report"] = {"path": d2_io.rel_path(report_json), "sha256": d2_io.sha256_path(report_json)}
    assert_no_outcome_keys(log, "exclusion log")
    return log


def _substance(obj: Any) -> Any:
    """A log without its time stamp and report pointer, normalised through JSON."""
    obj = json.loads(json.dumps(obj, default=d2_io._json_default))
    return None if not isinstance(obj, dict) else {k: v for k, v in obj.items() if k not in ("created", "report")}


def check_exclusion_log(report: Mapping, path: str | os.PathLike | None = None, overwrite: bool = False) -> Path:
    """Refuse, before anything is written, when an exclusion log with other content exists (unless `overwrite`)."""
    p = Path(path) if path else EXCLUSION_LOG
    log = build_exclusion_log(report)
    if p.exists() and not overwrite:
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if _substance(old) != _substance(log):
            raise ScreeningError(f"{d2_io.rel_path(p)} exists with different content; pass --overwrite to replace the "
                                 "exclusion log deliberately")
    return p


def write_exclusion_log(report: Mapping, report_json: Path, path: str | os.PathLike | None = None,
                        overwrite: bool = False) -> Path:
    """Write the exclusion log (with the path and sha256 of the written report). An existing log with other
    content is refused unless `overwrite`."""
    p = check_exclusion_log(report, path, overwrite)
    d2_io._write_json(p, build_exclusion_log(report, report_json))
    return p
