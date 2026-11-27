"""The GSM8K panel of the extension : the 19 main-panel
checkpoints and the 11 added ones, analysed together as GSM8K-30.

The 19 come from one scored table (data/processed/gsm8k/scored.parquet) and one exclusion log
(manifests/gsm8k_admissibility.json). Each added checkpoint has its own collection build, so its own scored table
(data/processed/gsm8k_ext/scored_<key>.parquet) and its own exclusion log, written by scripts/screen_d2.py under the
model rules (manifests/gsm8k_admissibility_extension/<key>.json). Every table must pass the target-table guards of
src/d2_io.py with the GSM8K sampler, the pinned test items and the scorer, and every
model must appear in exactly one table and one log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from . import bbh_d2, d2_io

ROOT = d2_io.ROOT
LABEL = "gsm8k30"
MAIN_SCORED = ROOT / "data" / "processed" / "gsm8k" / "scored.parquet"
EXTENSION_SCORED_DIR = ROOT / "data" / "processed" / "gsm8k_ext"
EXTENSION_LOG_DIR = ROOT / "manifests" / "gsm8k_admissibility_extension"
MAIN_LOG = ROOT / "manifests" / "gsm8k_admissibility.json"
REGISTRIES = bbh_d2.REGISTRIES                    # config/models.yaml, config/models_extension.yaml


class PanelError(Exception):
    """The GSM8K-30 inputs are incomplete or inconsistent; nothing was read past the check."""


def extension_tables(directory: str | os.PathLike | None = None) -> list[Path]:
    """The added checkpoints' scored tables, scored_<key>.parquet, in key order."""
    d = Path(directory) if directory else EXTENSION_SCORED_DIR
    return sorted(d.glob("scored_*.parquet"))


def table_models(md: Mapping[str, str]) -> list[str]:
    """The model keys behind a scored table, from its d2_inputs metadata."""
    try:
        inputs = json.loads(md["d2_inputs"])
        return [str(i["model_key"]) for i in inputs]
    except (KeyError, ValueError, TypeError) as exc:
        raise PanelError(f"d2_inputs is absent or unreadable ({exc})") from exc


def admissibility(registry: Mapping[str, dict], extension_keys: Sequence[str],
                  main_log: str | os.PathLike | None = None,
                  extension_dir: str | os.PathLike | None = None) -> tuple[set[str], list[dict]]:
    """The models the screens excluded, and the provenance of every log read. The main log covers the keys
    of config/models.yaml; each added key needs its own log that screened that key alone. A panel model that
    no log screened is refused."""
    log, info = d2_io.load_admissibility(main_log or MAIN_LOG)
    excluded = {m["model_key"] for m in log["excluded_models"]}
    infos = [info]
    problems = d2_io.panel_admissibility_problems([k for k in registry if k not in extension_keys], log)
    problems = [p for p in problems if "never screened" in p]
    d = Path(extension_dir) if extension_dir else EXTENSION_LOG_DIR
    for k in extension_keys:
        p = d / f"{k}.json"
        if not p.is_file():
            problems.append(f"{k}: never screened for admissibility ({d2_io.rel_path(p)} does not exist)")
            continue
        one, one_info = d2_io.load_admissibility(p)
        if list(one.get("models_screened") or []) != [k]:
            problems.append(f"{d2_io.rel_path(p)} screened {one.get('models_screened')}, not {k} alone")
            continue
        excluded |= {m["model_key"] for m in one["excluded_models"]}
        infos.append(one_info)
    if problems:
        raise PanelError("; ".join(problems))
    return excluded, infos


def check_tables(paths: Sequence[str | os.PathLike], sampler: Mapping, items_info: Mapping,
                 scorer_path: str | os.PathLike) -> dict[str, Path]:
    """{model: table} after the main guards of every table; a model in two tables is refused."""
    where: dict[str, Path] = {}
    problems = []
    for p in map(Path, paths):
        try:
            md = d2_io.scored_metadata(p)
        except (OSError, ValueError) as exc:
            raise PanelError(f"cannot read {p}: {exc}") from exc
        bad = d2_io.main_problems(md, sampler=sampler, items_info=items_info, scorer_path=scorer_path)
        if md.get("d2_task", "gsm8k") != "gsm8k":
            bad.append(f"d2_task is {md.get('d2_task')!r}, expected 'gsm8k'")
        if bad:
            problems.append(f"{d2_io.rel_path(p)}: " + "; ".join(bad))
            continue
        for k in table_models(md):
            if k in where:
                problems.append(f"{k} is in two tables ({d2_io.rel_path(where[k])} and {d2_io.rel_path(p)})")
            where[k] = p
    if problems:
        raise PanelError("; ".join(problems))
    return where


def bbh_inputs(scored: str | os.PathLike | None, sampler: Mapping, admissibility_log: str | os.PathLike | None = None,
               scorer_path: str | os.PathLike | None = None,
               registries: Sequence[str | os.PathLike] | None = None) -> dict:
    """The BIG-Bench Hard counterpart of inputs(), with the checks of scripts/run_dataset2.py main_bbh: a complete
    target table scored by the scorer with the sampler, the BIG-Bench Hard exclusion
    log, the panel of both registries less the excluded models. Adds `items_by_task`."""
    p = Path(scored or bbh_d2.SCORED[d2_io.KIND_MAIN])
    try:
        md = d2_io.scored_metadata(p)
    except (OSError, ValueError) as exc:
        raise PanelError(f"cannot read {p}: {exc}") from exc
    problems = d2_io.main_problems(md)
    if md.get("d2_task") != "bbh":
        problems.append(f"d2_task is {md.get('d2_task')!r}, expected 'bbh'")
    if problems:
        raise PanelError(f"{d2_io.rel_path(p)}: " + "; ".join(problems))
    try:
        items_df, items_info = bbh_d2.load_items("target")
        log, log_info = d2_io.load_admissibility(admissibility_log or bbh_d2.ADMISSIBILITY_LOG)
        reg = bbh_d2.load_registries(registries or REGISTRIES)
    except (OSError, ValueError, d2_io.D2IOError, bbh_d2.BBHPipelineError) as exc:
        raise PanelError(str(exc)) from exc
    if log.get("task") != "bbh":
        raise PanelError(f"{log_info['path']} is not the BIG-Bench Hard exclusion log")
    if sampler.get("label") != "bbh":
        raise PanelError(f"the sampler is not the BIG-Bench Hard sampler (label {sampler.get('label')!r})")
    problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info,
                                           scorer_path=scorer_path or bbh_d2.SCORER_MANIFEST,
                                           items_label=bbh_d2.TARGET_LABEL)
    unscreened = [q for q in d2_io.panel_admissibility_problems(list(reg), log) if "never screened" in q]
    if problems or unscreened:
        raise PanelError(f"{d2_io.rel_path(p)}: " + "; ".join(problems + unscreened))
    excluded = {m["model_key"] for m in log["excluded_models"]}
    panel = [k for k in reg if k not in excluded]
    if not panel:
        raise PanelError("every model was excluded by the BIG-Bench Hard admissibility rules")

    def load(ids):
        df = pd.read_parquet(p, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        df["item_id"] = df["item_id"].astype(str)
        return df[df["model_key"].isin(panel)]

    by_task = {t: g["item_id"].astype(str).tolist() for t, g in items_df.groupby("task", sort=True)}
    return {"panel": panel, "families": [reg[k].get("family") for k in panel],
            "tiers": bbh_d2.extension_tiers(reg, panel), "load": load, "excluded": sorted(excluded & set(reg)),
            "items": items_df["item_id"].astype(str).tolist(), "items_by_task": by_task,
            "provenance": {"tables": [{"path": d2_io.rel_path(p), "sha256": d2_io.sha256_path(p)}],
                           "admissibility": [log_info], "excluded_models": sorted(excluded & set(reg))}}


def math_inputs(scored: str | os.PathLike | None, sampler: Mapping, admissibility_log: str | os.PathLike | None = None,
                scorer_path: str | os.PathLike | None = None, panel_source: str | os.PathLike | None = None) -> dict:
    """The MATH-500 counterpart of bbh_inputs with the checks of scripts/run_dataset2.py
    main_bbh --task math: a complete target table scored by math-boxed with the MATH sampler,
    the MATH exclusion log, and the GSM8K-29 panel (math_d2.panel) less the models the screen excluded."""
    from . import math_d2
    p = Path(scored or math_d2.SCORED[d2_io.KIND_MAIN])
    try:
        md = d2_io.scored_metadata(p)
    except (OSError, ValueError) as exc:
        raise PanelError(f"cannot read {p}: {exc}") from exc
    problems = d2_io.main_problems(md)
    if md.get("d2_task") != math_d2.TASK:
        problems.append(f"d2_task is {md.get('d2_task')!r}, expected {math_d2.TASK!r}")
    if problems:
        raise PanelError(f"{d2_io.rel_path(p)}: " + "; ".join(problems))
    try:
        items_df, items_info = math_d2.load_items("target")
        log, log_info = d2_io.load_admissibility(admissibility_log or math_d2.ADMISSIBILITY_LOG)
        reg = math_d2.load_registries()
        registered = math_d2.panel(panel_source)
    except (OSError, ValueError, d2_io.D2IOError) as exc:
        raise PanelError(str(exc)) from exc
    if log.get("task") != math_d2.TASK:
        raise PanelError(f"{log_info['path']} is not the {math_d2.TITLE} exclusion log")
    if sampler.get("label") != math_d2.TASK:
        raise PanelError(f"the sampler is not the {math_d2.TITLE} sampler (label {sampler.get('label')!r})")
    if set(registered) - set(reg):
        raise PanelError(f"models {sorted(set(registered) - set(reg))} are in no registry")
    problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info,
                                           scorer_path=scorer_path or math_d2.SCORER_MANIFEST,
                                           items_label=math_d2.TARGET_LABEL)
    unscreened = [q for q in d2_io.panel_admissibility_problems(registered, log) if "never screened" in q]
    if problems or unscreened:
        raise PanelError(f"{d2_io.rel_path(p)}: " + "; ".join(problems + unscreened))
    excluded = {m["model_key"] for m in log["excluded_models"]}
    panel = [k for k in registered if k not in excluded]
    if not panel:
        raise PanelError(f"every panel model was excluded by the {math_d2.TITLE} admissibility rules")

    def load(ids):
        df = pd.read_parquet(p, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        df["item_id"] = df["item_id"].astype(str)
        return df[df["model_key"].isin(panel)]

    return {"panel": panel, "families": [reg[k].get("family") for k in panel],
            "tiers": math_d2.extension_tiers(reg, panel), "load": load, "excluded": sorted(excluded & set(registered)),
            "items": items_df["item_id"].astype(str).tolist(),
            "provenance": {"tables": [{"path": d2_io.rel_path(p), "sha256": d2_io.sha256_path(p)}],
                           "admissibility": [log_info], "excluded_models": sorted(excluded & set(registered))}}


def inputs(main_scored: str | os.PathLike | None, extension_scored: Sequence[str | os.PathLike] | None,
           sampler: Mapping, items_info: Mapping, scorer_path: str | os.PathLike,
           main_log: str | os.PathLike | None = None, extension_dir: str | os.PathLike | None = None,
           registries: Sequence[str | os.PathLike] | None = None) -> dict:
    """Everything the GSM8K-30 analysis needs before it reads an outcome: the panel (both registries less the
    excluded models, registry order), families, ECI tiers of the 30, a loader that reads only the units asked for
    from every table, and the provenance. Raises PanelError on any missing or inconsistent input."""
    try:
        reg = bbh_d2.load_registries(registries or REGISTRIES)
        ext_keys = list(bbh_d2.load_registries([(registries or REGISTRIES)[1]]))
    except bbh_d2.BBHPipelineError as exc:
        raise PanelError(str(exc)) from exc
    excluded, logs = admissibility(reg, ext_keys, main_log, extension_dir)
    tables = [Path(main_scored or MAIN_SCORED)] + [Path(p) for p in
                                                                    (extension_scored or extension_tables())]
    where = check_tables(tables, sampler, items_info, scorer_path)
    panel = [k for k in reg if k not in excluded]
    missing = [k for k in panel if k not in where]
    if missing:
        raise PanelError(f"no scored table holds {missing}")
    if not panel:
        raise PanelError("every model was excluded by the admissibility rules")
    used = sorted({where[k] for k in panel})

    def load(ids):
        keep = sorted(set(map(str, ids)))
        parts = [pd.read_parquet(p, filters=[("config_id", "in", keep)]) for p in used]
        df = pd.concat(parts, ignore_index=True)
        return df[df["model_key"].isin(panel)]

    return {"panel": panel, "families": [reg[k].get("family") for k in panel],
            "tiers": bbh_d2.extension_tiers(reg, panel), "load": load, "excluded": sorted(excluded & set(reg)),
            "provenance": {"tables": [{"path": d2_io.rel_path(p), "sha256": d2_io.sha256_path(p),
                                       "models": sorted(k for k in panel if where[k] == p)} for p in used],
                           "admissibility": logs, "excluded_models": sorted(excluded & set(reg)),
                           "tiers": "ECI terciles of the 30-model extension panel (manifests/d2_panel_extension.json)"}}
