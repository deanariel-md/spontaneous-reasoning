r"""Inputs of the contrasts added to the cross-cutting analyses (config/added_contrasts.yaml).

A contrast is a baseline arm B and an intervention arm P on one Dataset 2 task. load_config() reads and checks the
configuration; inputs(name) returns the task's panel, items and frozen sampler with a loader whose rows carry B as
direct_answer and P as zero_shot_cot, the labels every Dataset 2 analysis expects; arrays(x, split) gives (B, P) of the
discovery or validation draws through the cell table of the main analysis. Nothing here reads a validation draw unless
asked for it.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONFIG = os.path.join(ROOT, "config", "added_contrasts.yaml")
TASKS = ("gsm8k", "bbh", "math")
ARM_KINDS = ("collection", "processed", "table")
COLLECTION_ARMS = ("direct_answer", "zero_shot_cot")
REQUIRED = ("group", "map_panel", "tag", "task", "title", "short", "baseline", "intervention", "steps",
            "noise_base")
DEFAULT_POOL = "base"
TASK_REQUIRED = ("title", "tag", "scored", "sampler", "scorer", "module", "roles")
ROLES = ("requested", "forced", "cot")


class ConfigError(ValueError):
    """config/added_contrasts.yaml does not describe a usable contrast."""


def check_tasks(cfg: dict) -> dict:
    """The {task: entry} mapping of the section `tasks` (empty when absent) after the checks: a name not among the
    main tasks, every field present, a letters-only tag, the roles requested, forced and chain of thought named,
    decision settings on known roles, a fair-comparison entry with its file and arms."""
    tasks = (cfg or {}).get("tasks") or {}
    if not isinstance(tasks, dict):
        raise ConfigError("tasks must be a mapping")
    for t, e in tasks.items():
        if t in TASKS:
            raise ConfigError(f"task {t}: already a main task")
        miss = [k for k in TASK_REQUIRED if k not in e]
        if miss:
            raise ConfigError(f"task {t}: missing {miss}")
        if not str(e["tag"]).isalpha():
            raise ConfigError(f"task {t}: tag {e['tag']!r} must be letters only")
        lack = [r for r in ROLES if r not in (e["roles"] or {})]
        if lack:
            raise ConfigError(f"task {t}: roles {lack} are not named")
        for s, d in (e.get("decision") or {}).items():
            bad = [r for r in (d.get("baseline"), d.get("intervention")) if r not in e["roles"]]
            if bad or not d.get("choices") or not d.get("title"):
                raise ConfigError(f"task {t}: decision setting {s} needs known roles, choices and a title")
        fair = e.get("fair")
        if fair is not None and (not fair.get("evaluation") or not fair.get("arms")):
            raise ConfigError(f"task {t}: fair needs an evaluation file and arms")
    return tasks


def check_config(cfg: dict) -> dict:
    """The {name: contrast} mapping after the checks: every field present, a known task (main, or one of the section
    `tasks`), arms of a known kind, tags letters only and distinct, names lower case with underscores, settings
    distinct, pools letters and digits."""
    con = (cfg or {}).get("contrasts")
    if not isinstance(con, dict) or not con:
        raise ConfigError("no contrasts")
    added_tasks = check_tasks(cfg)
    tags, settings = {}, {}
    for name, c in con.items():
        if not name.replace("_", "").isalnum() or name != name.lower():
            raise ConfigError(f"{name}: a name is lower case letters, digits and underscores")
        miss = [k for k in REQUIRED if k not in c]
        if miss:
            raise ConfigError(f"{name}: missing {miss}")
        if c["task"] not in TASKS and c["task"] not in added_tasks:
            raise ConfigError(f"{name}: task {c['task']!r} is not one of {TASKS + tuple(added_tasks)}")
        pool = str(c.get("pool", DEFAULT_POOL))
        if not pool.isalnum():
            raise ConfigError(f"{name}: pool {pool!r} must be letters and digits")
        sn = setting_name(c)
        if sn in settings:
            raise ConfigError(f"{name}: setting {sn} is also {settings[sn]}'s")
        settings[sn] = name
        if not str(c["tag"]).isalpha():
            raise ConfigError(f"{name}: tag {c['tag']!r} must be letters only")
        if c["tag"] in tags:
            raise ConfigError(f"{name}: tag {c['tag']} is also {tags[c['tag']]}'s")
        tags[c["tag"]] = name
        for side in ("baseline", "intervention"):
            a = c[side]
            if a.get("kind") not in ARM_KINDS:
                raise ConfigError(f"{name}: {side} kind {a.get('kind')!r} is not one of {ARM_KINDS}")
            if a["kind"] == "collection" and a.get("arm") not in COLLECTION_ARMS:
                raise ConfigError(f"{name}: {side} arm {a.get('arm')!r} is not one of {COLLECTION_ARMS}")
            if a["kind"] == "processed" and not a.get("dir"):
                raise ConfigError(f"{name}: {side} needs a dir")
            if a["kind"] == "table":
                if c["task"] not in added_tasks:
                    raise ConfigError(f"{name}: a table arm needs a task of the section tasks")
                if a.get("arm") not in added_tasks[c["task"]]["roles"].values():
                    raise ConfigError(f"{name}: {side} arm {a.get('arm')!r} is not one of the task's roles")
            elif c["task"] in added_tasks:
                raise ConfigError(f"{name}: an added task's arms are of kind table")
        if c["baseline"] == c["intervention"]:
            raise ConfigError(f"{name}: the two arms are the same")
    return con


def load_config(path=CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        return check_config(yaml.safe_load(fh))


def load_tasks(path=CONFIG) -> dict:
    """{task: entry} of the section `tasks`; empty when the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    check_config(cfg)
    return check_tasks(cfg)


def setting_name(c) -> str:
    """The map point's name: the contrast's `setting`, else '<group>:<task>'."""
    return str(c.get("setting") or f"{c['group']}:{c['task']}")


def pool_of(c) -> str:
    return str(c.get("pool", DEFAULT_POOL))


def pools(con: dict) -> dict:
    """{pool: {name: contrast}} in order of first appearance, pool base (the default pool) first when present."""
    out = {}
    for name, c in con.items():
        out.setdefault(pool_of(c), {})[name] = c
    if DEFAULT_POOL in out:
        out = {DEFAULT_POOL: out[DEFAULT_POOL], **{k: v for k, v in out.items() if k != DEFAULT_POOL}}
    return out


def pool_info(pool: str, path=CONFIG) -> dict:
    """{"tag", "title"} of a pool from the section `pools` (tag letters only; the default pool base's tag is empty); a pool not listed
    gets an empty tag only for base, otherwise ConfigError."""
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    e = dict(((cfg.get("pools") or {}).get(pool)) or {})
    if not e:
        if pool != DEFAULT_POOL:
            raise ConfigError(f"pool {pool} is not listed in the section pools")
        e = {"tag": "", "title": "Added contrasts"}
    tag = str(e.get("tag") or "")
    if tag and not tag.isalpha():
        raise ConfigError(f"pool {pool}: tag {tag!r} must be letters only")
    if pool != DEFAULT_POOL and not tag:
        raise ConfigError(f"pool {pool}: a pool other than {DEFAULT_POOL} needs a tag")
    return {"tag": tag, "title": str(e.get("title") or pool)}


def pool_key(key: str, pool: str) -> str:
    """The results key of a pool: the key itself for the default pool base, key_<pool> otherwise."""
    return key if pool == DEFAULT_POOL else f"{key}_{pool}"


# ---------------------------------------------------------------- an added task (section `tasks`)


def task_module(task, tasks=None):
    import importlib
    e = (tasks if tasks is not None else load_tasks())[task]
    return importlib.import_module(e["module"])


def table_loader(task, arms, tasks=None):
    """load(ids) -> rows of the given arms of an added task's scored table (item and configuration ids as strings)."""
    e = (tasks if tasks is not None else load_tasks())[task]
    path = os.path.join(ROOT, e["scored"])

    def load(ids):
        df = pd.read_parquet(path, filters=[("config_id", "in", sorted(set(map(str, ids)))),
                                            ("arm", "in", list(arms))])
        df["item_id"] = df["item_id"].astype(str)
        df["config_id"] = df["config_id"].astype(str)
        return df
    return load


def role_loader(task, role, label=None, tasks=None):
    """load(ids) -> rows of the arm behind a role (requested, forced, cot, five_shot), relabelled `label` if given."""
    e = (tasks if tasks is not None else load_tasks())[task]
    base = table_loader(task, [e["roles"][role]], tasks)

    def load(ids):
        df = base(ids)
        return df.assign(arm=label) if label else df
    return load


def task_collection_inputs(task, tasks=None):
    """An added task in the form of scripts/run_forced_direct.collection_inputs: (inp, sampler, items, items_info),
    inp's loader giving the requested arm as direct_answer and chain of thought as zero_shot_cot."""
    import json
    tk = tasks if tasks is not None else load_tasks()
    e = tk[task]
    panel, fams, tiers = task_module(task, tk).panel_info()
    items = [str(i) for i in task_module(task, tk).item_ids()]
    s = json.load(open(os.path.join(ROOT, e["sampler"]), encoding="utf-8"))
    req, cot = e["roles"]["requested"], e["roles"]["cot"]
    base = table_loader(task, [req, cot], tk)

    def load(ids):
        df = base(ids)
        df["arm"] = df["arm"].map({req: COLLECTION_ARMS[0], cot: COLLECTION_ARMS[1]})
        return df[df["model_key"].isin(panel)]
    prov = {"tables": [{"path": e["scored"]}], "task": task, "config": "config/added_contrasts.yaml",
            "arms": f"{req} as direct_answer, {cot} as zero_shot_cot"}
    inp = {"panel": list(panel), "families": list(fams), "tiers": tiers, "load": load, "items": items,
           "provenance": prov}
    return inp, s, items, None


def task_forced_inputs(task, panel, tasks=None):
    """An added task's forced arm in the form of scripts/run_forced_direct.forced_inputs: (loader, provenance, None),
    rows labelled forced_direct."""
    e = (tasks if tasks is not None else load_tasks())[task]
    base = role_loader(task, "forced", "forced_direct", tasks)

    def load(ids):
        df = base(ids)
        return df[df["model_key"].isin(panel)]
    return load, {"path": e["scored"], "arm": e["roles"]["forced"]}, None


def relabelled(b_load, p_load):
    """A loader of both arms: B's rows as direct_answer, P's as zero_shot_cot."""
    def load(ids):
        return pd.concat([b_load(ids).assign(arm=COLLECTION_ARMS[0]), p_load(ids).assign(arm=COLLECTION_ARMS[1])],
                         ignore_index=True)
    return load


def collection_arm(col_load, arm):
    def load(ids):
        df = col_load(ids)
        return df[df["arm"] == arm]
    return load


def arm_loader(spec, task, panel, sampler, items_info, col_load):
    """(loader, provenance) of one arm. A processed table is read by scripts/run_forced_direct.forced_inputs with its
    checks (labels, sampler, items, scorer, panel), from the configured folder; a table arm from the added task's
    scored table."""
    if spec["kind"] == "collection":
        return collection_arm(col_load, spec["arm"]), {"collection_arm": spec["arm"]}
    if spec["kind"] == "table":
        tk = load_tasks()
        base = table_loader(task, [spec["arm"]], tk)

        def load(ids):
            df = base(ids)
            return df[df["model_key"].isin(panel)]
        return load, {"path": tk[task]["scored"], "arm": spec["arm"]}
    import run_forced_direct as rfd
    saved = rfd.PROCESSED
    rfd.PROCESSED = os.path.join(ROOT, spec["dir"])
    try:
        load, prov, _ = rfd.forced_inputs(task, panel, sampler, items_info, col_load, False)
    finally:
        rfd.PROCESSED = saved
    return load, prov


def inputs(name, contrasts=None) -> dict:
    """Everything an analysis needs for one contrast; the loader reads only the configurations it is given."""
    import run_forced_direct as rfd
    con = contrasts if contrasts is not None else load_config()
    if name not in con:
        raise ConfigError(f"unknown contrast {name!r}")
    c = con[name]
    inp, s, items, items_info = rfd.collection_inputs(c["task"])
    panel = list(inp["panel"])
    b_load, b_prov = arm_loader(c["baseline"], c["task"], panel, s, items_info, inp["load"])
    p_load, p_prov = arm_loader(c["intervention"], c["task"], panel, s, items_info, inp["load"])
    return {"name": name, **{k: c[k] for k in REQUIRED}, "setting": setting_name(c), "pool": pool_of(c),
            "panel": panel, "items": list(items), "sampler": s,
            "families": list(inp["families"]), "tiers": inp.get("tiers"), "load": relabelled(b_load, p_load),
            "provenance": {"collection": inp.get("provenance"), "baseline": b_prov, "intervention": p_prov,
                           "arms": "baseline relabelled direct_answer, intervention relabelled zero_shot_cot"},
            "disc": list(s["discovery_config_ids"]), "val": list(s["validation_config_ids"])}


def arrays(x, split, value="strict_correct"):
    """(B, P), each (K, M) in pp, of one split ("discovery" or "validation"), rows in the sampler's draw order."""
    from src import dataset2 as d2
    if split not in ("discovery", "validation"):
        raise ValueError(f"split {split!r}")
    units = x["disc"] if split == "discovery" else x["val"]
    values = tuple(dict.fromkeys(tuple(d2.VALUES) + (value,)))
    cells = d2.cell_table(x["load"](units), x["panel"], units, x["items"], values=values)
    return d2.unit_arrays(cells, units, x["panel"], value)


def fair_evaluation(task, tasks=None, root=ROOT):
    """The evaluation of an added task's fair comparison (section tasks, key fair; HotpotQA: scripts/hotpot_fair.py,
    from choices saved before its validation draws were scored), with its contrasts renamed to the baseline arms the
    strict decision design and the transfer analysis use: {"path", "saved", "arms", "strict": {design: {arm: {...}}},
    "family": {arm: {...}}, "cross": {(source, arm): {...}}}. None when the task has no such entry or the file is
    absent."""
    import json
    e = (tasks if tasks is not None else load_tasks())[task]
    fair = e.get("fair")
    if not fair:
        return None
    path = os.path.join(root, fair["evaluation"])
    if not os.path.exists(path):
        return None
    F = json.load(open(path, encoding="utf-8"))
    arms = dict(fair["arms"])
    out = {"path": fair["evaluation"], "saved": F.get("saved"), "arms": arms,
           "strict": {d: {arms[c]: v for c, v in by.items() if c in arms} for d, by in (F.get("strict") or {}).items()},
           "family": {arms[c]: v for c, v in (F.get("family") or {}).items() if c in arms}, "cross": {}}
    for key, v in (F.get("cross_benchmark") or {}).items():
        src, _, c = key.partition("/")
        if c in arms:
            out["cross"][(src, arms[c])] = v
    return out
