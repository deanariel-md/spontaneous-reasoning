r"""Reliability map.

PowerShell:
    .\.venv\Scripts\python.exe scripts\run_reliability_map.py              # DOVE block, 71 DOVE benchmarks, GSM8K-19
    .\.venv\Scripts\python.exe scripts\run_reliability_map.py --quick      # fewer splits and Monte Carlo draws
    # once collected and scored: BIG-Bench Hard (pooled and its 25 tasks) and GSM8K-30
    .\.venv\Scripts\python.exe scripts\run_reliability_map.py --bbh --gsm8k30
    # MATH-500, once its collection is scored: one more setting and its own frozen pilot file
    .\.venv\Scripts\python.exe scripts\run_reliability_map.py --bbh --gsm8k30 --math
    # the contrasts of config/added_contrasts.yaml, kept apart from the main settings
    .\.venv\Scripts\python.exe scripts\run_reliability_map.py --bbh --gsm8k30 --math --added

Settings. Each point is one (benchmark or task, panel, intervention), for benefit and accuracy:
- DOVE, five-shot against zero-shot, 5 models: the pooled MMLU block (scripts/run_dataset1.py) and every eligible
  English benchmark of the secondary analysis (scripts/run_dataset1_tasks.py: the same task list, exclusions and
  common-configuration rule). Each bank is enumerated, so R and C are exact conditional values over the bank.
- GSM8K, chain of thought against direct answer, strict scoring: the 19-model panel (scripts/run_dataset2.py), and
  with --gsm8k30 the 30-model panel (src/d2_panels.inputs). R and C come from the 24 discovery draws, as in the main
  analysis.
- With --bbh, BIG-Bench Hard: the pooled task set and each of its 25 tasks (src/d2_panels.bbh_inputs), discovery
  draws as for GSM8K.
- With --added, the contrasts of config/added_contrasts.yaml (five-shot against zero-shot forced direct on GSM8K, BIG-Bench
  Hard and MATH-500; chain of thought against constrained forced direct on BIG-Bench Hard), each through the Dataset 2
  route above with its own frozen pilot file (manifests/predictions_reliability_map_added_<name>.json). They are
  written to "added_settings" with their own departures, pilot test, checks and macros (prefix map, the contrast's tag),
  and do not enter the main settings' summaries, so every earlier number stays as it was. A later pool
  of the file (HotpotQA's three contrasts, pool hotpot) goes to the same keys with the suffix _<pool>
  ("added_settings_hotpot", ...), so the pooled summaries of pool base stay as they were.

Pilot test. R from the first 2 and the first 4 pilot draws predicts the agreement among the validation draws (see
src/reliability_map.py). Pilot and validation draws: the DOVE MMLU block and Dataset 2 use their samplers' discovery
order and validation draws (manifests/sampler_*.json); each DOVE benchmark uses 4 pilot and 24 validation
configurations drawn by config/reliability_map.yaml and written to manifests/sampler_reliability_map_dove_tasks.json.

Order: per group, the pilot predictions are computed from the pilot draws alone and written to
manifests/predictions_reliability_map_<group>.json before any validation draw is scored; Dataset 2 validation draws
are read from disk only after that.

Outputs: results/reliability_map/reliability_map.json (with a "macros" object for paper/make_numbers.py),
summary.md and summary_clean.md.
"""
import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import controls, dataset1 as d1, dataset2 as d2, reliability_map as rm  # noqa: E402
from src import dove_io as io  # noqa: E402
from src.estimators import pair_agreement  # noqa: E402
from src.sampling import rng_for, sha256_json  # noqa: E402

MAN = os.path.join(ROOT, "manifests")
CONFIG = os.path.join(ROOT, "config", "reliability_map.yaml")
OUT = os.path.join(ROOT, "results", "reliability_map")
LABEL = "reliability_map"
DOVE_TASK_DRAWS = os.path.join(MAN, "sampler_reliability_map_dove_tasks.json")
EXISTING = {"dove_mmlu": os.path.join(ROOT, "results", "dataset1", "dove_mmlu.json"),
            "dove_tasks": os.path.join(ROOT, "results", "dataset1", "dove_tasks.json"),
            "gsm8k": os.path.join(ROOT, "results", "dataset2", "gsm8k.json"),
            "gsm8k30": os.path.join(ROOT, "results", "dataset2", "gsm8k30.json"),
            "bbh": os.path.join(ROOT, "results", "dataset2", "bbh.json"),
            "math": os.path.join(ROOT, "results", "dataset2", "math.json")}
GROUP_TITLE = {"dove_mmlu": "DOVE, pooled MMLU block", "dove_tasks": "DOVE, single benchmarks",
               "gsm8k": "GSM8K, 19 models", "gsm8k30": "GSM8K, 30 models", "bbh": "BIG-Bench Hard, pooled",
               "bbh_tasks": "BIG-Bench Hard, single tasks", "math": "MATH-500, numeric items"}
INTERVENTION = {"dove": "five-shot against zero-shot", "d2": "chain of thought against direct answer"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_reliability_map", os.path.join(SCRIPT_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=jsonable)


def freeze(group, models, draws_sha, preds, path, overwrite=False):
    """Write the pilot predictions of a group before its validation draws are scored (d2.write_frozen keeps an
    identical existing file and refuses a different one)."""
    frozen = {"label": f"{LABEL}_{group}", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": list(models),
              "draws_sha256": draws_sha, "units": "agreement (probability); R dimensionless",
              "note": "pilot predictions from the pilot draws alone; written before any validation draw is scored",
              "settings": preds}
    frozen["sha256"] = sha256_json(json.loads(json.dumps(preds, default=jsonable)))
    return d2.write_frozen(frozen, path, overwrite=overwrite)


# --------------------------------------------------------------------------- DOVE inputs


def dove_mmlu_inputs():
    """The pooled MMLU block over its common bank and the frozen DOVE MMLU sampler."""
    rd1 = load_script("run_dataset1")
    _, subjects, common, _, nc, n_items = rd1.load_block()
    sampler = json.load(open(os.path.join(MAN, "sampler_dove_mmlu.json"), encoding="utf-8"))
    if [int(c) for c in sampler["bank_config_ids"]] != [int(c) for c in common]:
        raise ValueError("manifests/sampler_dove_mmlu.json was drawn over another bank")
    B, P = controls.pooled(nc["0_shot"], n_items), controls.pooled(nc["5_shot"], n_items)
    return {"setting": "dove_mmlu_block", "group": "dove_mmlu", "title": "MMLU block (56 subjects pooled)",
            "family": "dove", "M": len(rd1.PANEL), "models": rd1.PANEL, "n_items": int(n_items.sum()),
            "n_subjects": len(subjects), "n_bank": len(common), "B": B, "P": P,
            "pilot": list(sampler["discovery_bank_index"]), "validation": list(sampler["validation_bank_index"]),
            "draws": "manifests/sampler_dove_mmlu.json (discovery order, validation draws)",
            "draws_sha": sha256_json({k: sampler[k] for k in ("discovery_bank_index", "validation_bank_index")})}


def dove_benchmarks(rt):
    """Every eligible English DOVE benchmark with its common admissible bank, by the rules of
    scripts/run_dataset1_tasks.py (audit checks, full panel, one item count, two or more common configurations)."""
    bank = pd.read_csv(os.path.join(MAN, "dove_config_bank.csv"), keep_default_na=False)
    admissible = set(bank.loc[bank["admissible"].astype(str) == "True", "config_id"])
    counts = pd.read_parquet(os.path.join(io.PROCESSED, "counts.parquet"))
    failed = rt.failed_check_tasks()
    keep, excluded = [], []
    for task, x in sorted(counts.groupby("file")):
        if task in failed:
            excluded.append({"task": task, "reason": f"a file fails the audit checks {failed[task]}"})
            continue
        present = sorted(x["model"].unique())
        if set(present) != set(rt.PANEL):
            excluded.append({"task": task, "reason": f"panel models present: {present}"})
            continue
        n_items = x.groupby(["model", "arm"])["n_items"].max()
        if n_items.nunique() != 1:
            excluded.append({"task": task, "reason": "item count differs across model-arm cells"})
            continue
        full = x[x["n_items"] == n_items.iloc[0]]
        per_cfg = full.groupby("config_id").size()
        common = sorted(int(c) for c, n in per_cfg.items() if n == len(rt.PANEL) * len(rt.ARMS) and c in admissible)
        if len(common) < 2:
            excluded.append({"task": task, "reason": f"{len(common)} common admissible configurations"})
            continue
        n = int(n_items.iloc[0])
        B, P = (rt.scores_pp(full, a, common, n) for a in rt.ARMS)
        keep.append({"task": task, "family_name": rt.family_of(task), "n_items": n, "common": common, "B": B, "P": P})
    return keep, excluded


def dove_task_draws(tasks, cfg, path=DOVE_TASK_DRAWS):
    """Pilot and validation draws per benchmark from its common configurations (metadata only). Written once; an
    existing file must hold the same draws."""
    draws = {"rule": "config/reliability_map.yaml; DOVE sampler's rule (uniform over the benchmark's common "
                     "admissible configurations, with replacement, independent draws), pilot draws first",
             "seed": int(cfg["seed"]), "n_pilot": int(cfg["dove_tasks_pilot_draws"]),
             "n_validation": int(cfg["dove_tasks_validation_draws"]), "benchmarks": {}}
    for t in tasks:
        d = rm.draw_pilot(t["common"], draws["n_pilot"], draws["n_validation"], f"{LABEL}/{t['task']}", draws["seed"])
        draws["benchmarks"][t["task"]] = d
    draws["sha256"] = sha256_json(draws["benchmarks"])
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("sha256") != draws["sha256"]:
            raise ValueError(f"{os.path.relpath(path, ROOT)} holds other draws; refusing to replace them")
        return old
    draws["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump(draws, path)
    return draws


def dove_task_inputs(cfg):
    rt = load_script("run_dataset1_tasks")
    tasks, excluded = dove_benchmarks(rt)
    draws = dove_task_draws(tasks, cfg)
    out = []
    for t in tasks:
        d = draws["benchmarks"][t["task"]]
        out.append({"setting": f"dove:{t['task']}", "group": "dove_tasks", "title": t["task"], "family": "dove",
                    "task_family": t["family_name"], "M": len(rt.PANEL), "models": rt.PANEL,
                    "n_items": t["n_items"], "n_bank": len(t["common"]), "B": t["B"], "P": t["P"],
                    "pilot": d["pilot_index"], "validation": d["validation_index"],
                    "draws": os.path.relpath(DOVE_TASK_DRAWS, ROOT).replace("\\", "/")})
    return out, excluded, draws["sha256"]


# --------------------------------------------------------------------------- Dataset 2 inputs


def gsm8k19_inputs(scored):
    """The 19-model GSM8K inputs with the guards of scripts/run_dataset2.py (table labels, sampler, items, scorer,
    exclusion log); the panel is config/models.yaml less the excluded models."""
    from src import d2_io
    md = d2_io.scored_metadata(scored)
    sampler = json.load(open(os.path.join(MAN, "sampler_gsm8k.json"), encoding="utf-8"))
    items_df, items_info = d2_io.load_items(split="test")
    log, _ = d2_io.load_admissibility(os.path.join(MAN, "gsm8k_admissibility.json"))
    problems = d2_io.main_problems(md)
    if md.get("d2_task", "gsm8k") != "gsm8k":
        problems.append(f"d2_task is {md.get('d2_task')!r}")
    problems += d2_io.main_problems(md, sampler=sampler, items_info=items_info,
                                            scorer_path=os.path.join(MAN, "gsm8k_scorer.json"))
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models.yaml"), encoding="utf-8"))["dataset2"]["models"]
    listed = [k for k in reg if k != "_template"]
    problems += [p for p in d2_io.panel_admissibility_problems(listed, log) if "never screened" in p]
    if problems:
        raise ValueError("refused: " + "; ".join(problems))
    excluded = {m["model_key"] for m in log["excluded_models"]}
    panel = [k for k in listed if k not in excluded]

    def load(ids):
        return pd.read_parquet(scored, filters=[("config_id", "in", sorted(set(map(str, ids))))])

    return {"panel": panel, "items": items_df["item_id"].tolist(), "load": load, "sampler": sampler,
            "subsets": {"gsm8k19": ("GSM8K", items_df["item_id"].tolist())}, "group_of": {"gsm8k19": "gsm8k"}}


def gsm8k30_inputs(scored, extension):
    from src import d2_io, d2_panels
    sampler = json.load(open(os.path.join(MAN, "sampler_gsm8k.json"), encoding="utf-8"))
    items_df, items_info = d2_io.load_items(split="test")
    got = d2_panels.inputs(scored, extension, sampler, items_info, os.path.join(MAN, "gsm8k_scorer.json"))
    items = items_df["item_id"].tolist()
    return {"panel": got["panel"], "items": items, "load": got["load"], "sampler": sampler,
            "subsets": {"gsm8k30": ("GSM8K", items)}, "group_of": {"gsm8k30": "gsm8k30"}}


def bbh_inputs(scored):
    """BIG-Bench Hard: the pooled task set (equal item counts, so equal task weights) and each task."""
    from src import d2_panels
    sampler = json.load(open(os.path.join(MAN, "sampler_bbh.json"), encoding="utf-8"))
    got = d2_panels.bbh_inputs(scored, sampler)
    subsets = {"bbh:pooled": ("BIG-Bench Hard, 25 tasks pooled", got["items"])}
    group_of = {"bbh:pooled": "bbh"}
    for t, its in got["items_by_task"].items():
        subsets[f"bbh:{t}"] = (t, its)
        group_of[f"bbh:{t}"] = "bbh_tasks"
    return {"panel": got["panel"], "items": got["items"], "load": got["load"], "sampler": sampler,
            "subsets": subsets, "group_of": group_of}


def math_inputs(scored):
    """MATH-500: the numeric target items pooled, the collection's panel and frozen sampler."""
    from src import d2_panels
    sampler = json.load(open(os.path.join(MAN, "sampler_math.json"), encoding="utf-8"))
    got = d2_panels.math_inputs(scored, sampler)
    return {"panel": got["panel"], "items": got["items"], "load": got["load"], "sampler": sampler,
            "subsets": {"math": ("MATH-500, numeric items", got["items"])}, "group_of": {"math": "math"}}


def added_setting(c) -> str:
    """Setting name of an added contrast: its `setting` if given, else '<group>:<task>', as the contrasts' own map
    points name them."""
    return str(c.get("setting") or f"{c['group']}:{c['task']}")


def pk(key, pool):
    """Results key of an added pool (scripts/added_contrasts.pool_key): pool base keeps the unsuffixed keys."""
    return key if pool == "base" else f"{key}_{pool}"


def added_inputs(name, contrasts):
    """An added contrast (config/added_contrasts.yaml) in the form run_d2 takes."""
    import added_contrasts as ac
    x = ac.inputs(name, contrasts)
    s = added_setting(x)
    return {"panel": x["panel"], "items": x["items"], "load": x["load"], "sampler": x["sampler"],
            "subsets": {s: (x["title"], x["items"])}, "group_of": {s: x["group"]}, "family": "added",
            "intervention": x["title"], "contrast": name, "tag": x["tag"], "steps": x["steps"],
            "map_panel": x["map_panel"]}


# --------------------------------------------------------------------------- groups


def run_dove(settings, group, models, draws_sha, cfg, n_splits, n_mc, pred_path, overwrite):
    """Enumerated DOVE banks: pilot predictions frozen first, then the bank point, the validation agreement and the
    bank's expected agreement."""
    ks, sizes, targets = cfg["group_sizes"], cfg["pilot_sizes"], cfg["agreement_targets"]
    preds = {}
    for s in settings:
        pi = np.asarray(s["pilot"])
        preds[s["setting"]] = {str(n): rm.pilot_prediction(s["B"][pi[:n]], s["P"][pi[:n]], ks, targets)
                               for n in sizes}
    frozen = freeze(group, models, draws_sha, preds, pred_path, overwrite)
    rows = []
    for s in settings:
        B, P = s["B"], s["P"]
        vi = np.asarray(s["validation"])
        point = rm.setting_point(B, P, exact=True)
        obs = rm.validation_agreement(B[vi], P[vi], ks, n_splits, rng_for(LABEL, "splits", s["setting"],
                                                                           seed=cfg["seed"]))
        bank = rm.bank_group_agreement({"benefit": P - B, "accuracy": B}, ks, n_mc,
                                       rng_for(LABEL, "bank", s["setting"], seed=cfg["seed"]))
        no_tol = {"benefit": d1.exact_agreement(P - B), "accuracy": d1.exact_agreement(B)}
        rows.append(setting_record(s, point, frozen["settings"][s["setting"]], obs, bank, no_tol))
    return rows, {"path": os.path.relpath(pred_path, ROOT), "sha256": frozen["sha256"], "created": frozen["created"]}


def run_d2(inp, label, cfg, n_splits, pred_path, overwrite):
    """Dataset 2: discovery draws first (pilot predictions frozen, then the map point), validation draws after."""
    ks, sizes, targets = cfg["group_sizes"], cfg["pilot_sizes"], cfg["agreement_targets"]
    sampler, panel = inp["sampler"], inp["panel"]
    disc, val = sampler["discovery_config_ids"], sampler["validation_config_ids"]

    def arrays(df, units):
        out = {}
        for s, (_, its) in inp["subsets"].items():
            keep = pd.Index(its)
            cells = d2.cell_table(df[df["item_id"].isin(keep)], panel, units, its)
            out[s] = d2.unit_arrays(cells, units, panel, "strict_correct")
        return out

    ad = arrays(inp["load"](disc), disc)
    preds = {s: {str(n): rm.pilot_prediction(Bd[:n], Pd[:n], ks, targets) for n in sizes} for s, (Bd, Pd) in ad.items()}
    frozen = freeze(label, panel, sampler["sha256"], preds, pred_path, overwrite)
    points = {s: rm.setting_point(Bd, Pd) for s, (Bd, Pd) in ad.items()}
    no_tol = {s: {"benefit": pair_agreement(Pd - Bd), "accuracy": pair_agreement(Bd)} for s, (Bd, Pd) in ad.items()}
    av = arrays(inp["load"](val), val)                                   # validation draws, read only now
    rows = []
    for s, (Bv, Pv) in av.items():
        title, its = inp["subsets"][s]
        obs = rm.validation_agreement(Bv, Pv, ks, n_splits, rng_for(LABEL, "splits", s, seed=cfg["seed"]))
        meta = {"setting": s, "group": inp["group_of"][s], "title": title, "family": inp.get("family", "d2"),
                "M": len(panel), "models": panel, "n_items": len(its), "n_bank": int(sampler["n_bank"]),
                "draws": f"manifests/sampler_{sampler['label']}.json (discovery order, validation draws)"}
        for k in ("intervention", "contrast", "tag", "steps", "map_panel"):     # an added contrast's labels
            if k in inp:
                meta[k] = inp[k]
        rows.append(setting_record(meta, points[s], frozen["settings"][s], obs, None, no_tol[s]))
    return rows, {"path": os.path.relpath(pred_path, ROOT), "sha256": frozen["sha256"], "created": frozen["created"]}


def setting_record(s, point, preds, obs, bank, no_tol) -> dict:
    rec = {k: s[k] for k in ("setting", "group", "title", "family", "M", "n_items", "n_bank", "draws")}
    rec["task_family"] = s.get("task_family")
    rec["intervention"] = s["intervention"] if "intervention" in s else INTERVENTION[s["family"]]
    for k in ("contrast", "tag", "steps", "map_panel"):
        if k in s:
            rec[k] = s[k]
    rec["n_pairs"] = s["M"] * (s["M"] - 1) // 2
    rec["point"] = point
    rec["point_no_tie_tolerance"] = {q: {k: no_tol[q][k] for k in ("C", "same", "opposite", "tie")} for q in no_tol}
    rec["pilot"] = preds
    rec["validation"] = obs
    rec["bank_groups"] = bank
    return rec


# --------------------------------------------------------------------------- checks against existing results


def check_rows(records):
    """Overlap with results/dataset1 and results/dataset2: ours (tie tolerance), ours without it, existing."""
    rows = []

    def add(setting, quantity, ours, ours_raw, existing):
        rows.append({"setting": setting, "quantity": quantity, "ours": ours, "ours_no_tie_tolerance": ours_raw,
                     "existing": existing, "difference": None if existing is None else ours - existing,
                     "difference_no_tie_tolerance": None if existing is None or ours_raw is None
                     else ours_raw - existing})

    by = {r["setting"]: r for r in records}
    if "dove_mmlu_block" in by and os.path.exists(EXISTING["dove_mmlu"]):
        e = json.load(open(EXISTING["dove_mmlu"], encoding="utf-8"))
        r = by["dove_mmlu_block"]
        for q in rm.QUANTITIES:
            add(r["setting"], f"R_{q}", r["point"][q]["R"], None, e["step2"]["exact"][f"R_{q}"])
            for k in ("C", "tie"):
                add(r["setting"], f"{k}_{q}", r["point"][q][k], r["point_no_tie_tolerance"][q][k],
                    e["step1"][f"{q}_exact"][k])
            for x in e["step3"]["validation_agreement_registered"]:
                g = r["validation"]["groups"][q].get(str(x["K"]))
                if g is not None:
                    add(r["setting"], f"C_{q}_validation_first_split_K{x['K']}", g["first_k_against_next_k"]["C"],
                        None, x[q]["C"])
    if os.path.exists(EXISTING["dove_tasks"]):
        e = {t["task"]: t for t in json.load(open(EXISTING["dove_tasks"], encoding="utf-8"))["tasks"]}
        for r in records:
            if r["group"] != "dove_tasks":
                continue
            t = e.get(r["title"])
            if t is None:
                add(r["setting"], "present_in_existing", 0.0, None, 1.0)
                continue
            for q in rm.QUANTITIES:
                add(r["setting"], f"R_{q}", r["point"][q]["R"], None, t[f"R_{q}"])
                add(r["setting"], f"C_{q}", r["point"][q]["C"], r["point_no_tie_tolerance"][q]["C"], t[f"C_{q}"])
            add(r["setting"], "tie_benefit", r["point"]["benefit"]["tie"],
                r["point_no_tie_tolerance"]["benefit"]["tie"], t["tie_benefit"])
            add(r["setting"], "n_configurations", float(r["n_bank"]), None, float(t["n_configurations"]))
            add(r["setting"], "n_items", float(r["n_items"]), None, float(t["n_items"]))
    for setting, key in (("gsm8k19", "gsm8k"), ("gsm8k30", "gsm8k30"), ("bbh:pooled", "bbh"), ("math", "math")):
        if setting not in by or not os.path.exists(EXISTING[key]):
            continue
        e = json.load(open(EXISTING[key], encoding="utf-8"))
        r = by[setting]
        add(setting, "panel_matches", float(r["M"]), None, float(len(e["models"])))
        for q in rm.QUANTITIES:
            add(setting, f"R_{q}", r["point"][q]["R"], None, e["step2"]["reliability"][f"R_{q}"])
            for k in ("C", "tie"):
                add(setting, f"{k}_{q}", r["point"][q][k], r["point_no_tie_tolerance"][q][k], e["step1"][f"C_{q}"][k])
            for x in e["step3"]["validation_agreement_curve"]:
                g = r["validation"]["groups"][q].get(str(x["K"]))
                if g is not None:
                    add(setting, f"C_{q}_validation_first_split_K{x['K']}", g["first_k_against_next_k"]["C"], None,
                        x[q]["C"])
    return rows


def added_check_rows(records, root=ROOT):
    """Each added contrast against the Steps 1-3 file of its own analysis (the record's "steps"): panel size, R and C
    of benefit and accuracy with their tie rates, and the validation agreement of the first K against the next K. A
    missing file gives one row with existing None."""
    rows = []

    def add(setting, quantity, ours, ours_raw, existing):
        rows.append({"setting": setting, "quantity": quantity, "ours": ours, "ours_no_tie_tolerance": ours_raw,
                     "existing": existing, "difference": None if existing is None else ours - existing,
                     "difference_no_tie_tolerance": None if existing is None or ours_raw is None
                     else ours_raw - existing})

    for r in records:
        path = os.path.join(root, r["steps"])
        if not os.path.exists(path):
            add(r["setting"], "steps_file_present", 0.0, None, None)
            continue
        e = json.load(open(path, encoding="utf-8"))
        add(r["setting"], "panel_matches", float(r["M"]), None, float(len(e["models"])))
        for q in rm.QUANTITIES:
            add(r["setting"], f"R_{q}", r["point"][q]["R"], None, e["step2"]["reliability"][f"R_{q}"])
            for k in ("C", "tie"):
                add(r["setting"], f"{k}_{q}", r["point"][q][k], r["point_no_tie_tolerance"][q][k],
                    e["step1"][f"C_{q}"][k])
            for x in e["step3"]["validation_agreement_curve"]:
                g = r["validation"]["groups"][q].get(str(x["K"]))
                if g is not None:
                    add(r["setting"], f"C_{q}_validation_first_split_K{x['K']}", g["first_k_against_next_k"]["C"],
                        None, x[q]["C"])
    return rows


# --------------------------------------------------------------------------- summaries


MAP = "map"   # predictions from the setting's own map R, beside the pilots


def map_prediction(r, cfg) -> dict:
    """The reference's predictions from the map point's R (DOVE: the exact bank value; Dataset 2: the 24 discovery
    draws, disjoint from the validation draws). Not frozen: on DOVE the bank holds the validation configurations."""
    return {"n_pilot": MAP, **{q: rm.predict_from_r(r["point"][q]["R"], cfg["group_sizes"], cfg["agreement_targets"])
                               for q in rm.QUANTITIES}}


def summaries(records, cfg):
    """Prediction errors, R errors, K needed and usable pilots, overall and per group. Error summaries cover the
    settings with a defined prediction; the observed agreement of the others is summarized beside them."""
    ks, sizes, targets, factor = cfg["group_sizes"], cfg["pilot_sizes"], cfg["agreement_targets"], cfg["usable_factor"]
    sources = list(sizes) + [MAP]
    pred_rows = []
    for r in records:
        for n in sources:
            p = map_prediction(r, cfg) if n == MAP else r["pilot"][str(n)]
            pred_rows += rm.prediction_rows(r["setting"], r["group"], p, r["validation"], r["bank_groups"])
    scopes = ["all"] + sorted({r["group"] for r in records})
    err = []
    for n in sources:
        for q in rm.QUANTITIES:
            for kind, k in [("one_draw", 1)] + [("groups", k) for k in ks if k > 1]:
                for sc in scopes:
                    rows = [x for x in pred_rows if x["n_pilot"] == n and x["quantity"] == q and x["target"] == kind
                            and x["K"] == k and (sc == "all" or x["group"] == sc)]
                    if not rows:
                        continue
                    und = np.array([x["observed"] for x in rows if not x["defined"]], float)
                    base = {"n_pilot": n, "quantity": q, "target": kind, "K": k, "scope": sc}
                    err.append({**base, "against": "validation", **rm.error_summary([x["error"] for x in rows]),
                                "observed_where_undefined": None if not und.size else
                                {"n": int(und.size), "mean": float(und.mean()), "min": float(und.min()),
                                 "max": float(und.max())}})
                    if all(x["bank"] is not None for x in rows):
                        err.append({**base, "against": "bank",
                                    **rm.error_summary([x["error_vs_bank"] for x in rows])})
    r_err, k_rows = [], []
    for r in records:
        for q in rm.QUANTITIES:
            Rm = r["point"][q]["R"]
            kref = {f"{t:.2f}": rm.k_needed(Rm, t) for t in targets}
            for n in sizes:
                p = r["pilot"][str(n)][q]
                r_err.append({"setting": r["setting"], "group": r["group"], "quantity": q, "n_pilot": n,
                              "R_pilot": p["R"], "R_map": Rm, "difference": p["R"] - Rm,
                              "pilot_negative": bool(math.isfinite(p["R"]) and p["R"] < 0)})
            for n in sources:
                kp = kref if n == MAP else r["pilot"][str(n)][q]["K_needed"]
                for t in kref:
                    k_rows.append({"setting": r["setting"], "group": r["group"], "quantity": q, "n_pilot": n,
                                   "target": t, "K_pilot": kp[t], "K_map": kref[t],
                                   "within_factor": rm.within_factor(kp[t], kref[t], factor)})
    r_sum, k_sum = [], []
    for n in sources:
        for q in rm.QUANTITIES:
            for sc in scopes:
                rows = [x for x in r_err if x["n_pilot"] == n and x["quantity"] == q and (sc == "all" or x["group"] == sc)]
                if rows:
                    s = rm.error_summary([x["difference"] for x in rows])
                    r_sum.append({"n_pilot": n, "quantity": q, "scope": sc, "n_settings": len(rows),
                                  "n_negative": sum(x["pilot_negative"] for x in rows), **s})
                for t in (f"{t:.2f}" for t in targets):
                    rows = [x for x in k_rows if x["n_pilot"] == n and x["quantity"] == q and x["target"] == t
                            and (sc == "all" or x["group"] == sc)]
                    if rows:
                        fin = [x["K_pilot"] for x in rows if x["K_pilot"] is not None and math.isfinite(x["K_pilot"])]
                        k_sum.append({"n_pilot": n, "quantity": q, "target": t, "scope": sc, "n_settings": len(rows),
                                      "n_within_factor": sum(x["within_factor"] is True for x in rows),
                                      "n_pilot_undefined": sum(x["K_pilot"] is None for x in rows),
                                      "n_map_undefined": sum(x["K_map"] is None for x in rows),
                                      "median_K": float(np.median(fin)) if fin else None,
                                      "min_K": min(fin) if fin else None, "max_K": max(fin) if fin else None})
    return {"usable_factor": factor, "prediction_rows": pred_rows, "prediction_errors": err, "R_errors": r_err,
            "R_error_summary": r_sum, "K_needed": k_rows, "K_needed_summary": k_sum,
            "note": "error = predicted minus observed agreement; summaries over the settings whose R lies in [0, 1] "
                    "(the others have no prediction and their observed agreement is given apart). n_pilot 'map' "
                    "predicts from the map point's R: on DOVE the exact bank value, on Dataset 2 the 24 discovery "
                    "draws, whose first 2 and 4 are the pilots."}


def departure_table(records):
    """Per point: C - C_ref split into the panel's realized differences (C_common - C_ref), heterogeneous pair noise
    (C_pairs - C_common), and ties with non-Gaussian contrasts (C - C_pairs); src/reliability_map.pairwise_reference."""
    rows = []
    for r in records:
        for q in rm.QUANTITIES:
            b = r["point"][q]
            pw = b["pairwise"]
            rows.append({"setting": r["setting"], "group": r["group"], "quantity": q, "M": r["M"],
                         "n_items": r["n_items"], "R": b["R"], "C": b["C"], "C_ref": b["C_ref"],
                         "departure": b["departure"], "tie": b["tie"], "decisive": b["decisive"],
                         "conditional_agreement": b["conditional_agreement"],
                         "C_common": pw["C_common"], "C_pairs": pw["C_pairs"], "pair_noise_cv": pw["pair_noise_cv"],
                         "tie_share_low_signal": pw["tie_share_low_signal"],
                         "panel_differences": pw["C_common"] - b["C_ref"] if math.isfinite(b["C_ref"]) else float("nan"),
                         "pair_noise": pw["C_pairs"] - pw["C_common"], "ties_non_gaussian": b["C"] - pw["C_pairs"]})
    return rows


def group_stats(rows, key):
    out = {}
    for g in sorted({r["group"] for r in rows}):
        for q in rm.QUANTITIES:
            v = np.array([r[key] for r in rows if r["group"] == g and r["quantity"] == q], float)
            v = v[np.isfinite(v)]
            if v.size:
                out[f"{g}:{q}"] = {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                                   "min": float(v.min()), "max": float(v.max())}
    return out


def r_for_k(k, target):
    """Smallest R at which k draws reach the target agreement under the reference: s / (k (1 - s) + s)."""
    s = math.sin(math.pi * (target - 0.5))
    return s / (k * (1 - s) + s)


# --------------------------------------------------------------------------- macros


def _f(x, nd=2):
    if x is None or not math.isfinite(x):
        return "undefined" if x is None or math.isnan(x) else r"$\infty$"
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "$-$")


def _k(k):
    if k is None:
        return "undefined"
    return r"$\infty$" if math.isinf(k) else f"{int(k):,}".replace(",", "{,}")


def _pct(x, nd=1):
    return _f(100 * x, nd) + r"\%"


NAMES = {"dove_mmlu_block": "Block", "gsm8k19": "Gsm", "gsm8k30": "GsmThirty", "bbh:pooled": "Bbh", "math": "Math"}


def mmlu_subject_blocks(recs, targets):
    """Over the DOVE benchmarks named mmlu.<subject> (MMLU-Pro excluded; tag MmluSubj) and over all DOVE
    benchmarks (tag Bench), K for each agreement target from each point's R (median, min, max of the defined ones and
    the count undefined) and the mean absolute departure |C - C_ref| from the Gaussian curve."""
    out = {}
    for tag, keep in (("MmluSubj", lambda r: r["setting"].startswith("dove:mmlu.")), ("Bench", lambda r: True)):
        rows = [r for r in recs if r["group"] == "dove_tasks" and keep(r)]
        if not rows:
            continue
        b = {"n": len(rows), "settings": [r["setting"] for r in rows], "quantities": {}}
        for q in rm.QUANTITIES:
            K = {}
            for t in targets:
                ks = [rm.k_needed(r["point"][q]["R"], t) for r in rows]
                fin = [float(k) for k in ks if k is not None and math.isfinite(k)]
                K[t] = {"undefined": len(ks) - len(fin), "median": float(np.median(fin)) if fin else None,
                        "min": float(np.min(fin)) if fin else None, "max": float(np.max(fin)) if fin else None}
            d = np.array([r["point"][q]["departure"] for r in rows], float)
            d = d[np.isfinite(d)]
            b["quantities"][q] = {"K": K, "departure_mean_abs": float(np.abs(d).mean()) if d.size else None,
                                  "departure_n": int(d.size)}
        out[tag] = b
    return out


def macros(out):
    """LaTeX macros (letters only, prefix map) for paper/make_numbers.py, as formatted strings."""
    m = {}
    recs = out["settings"]
    by = {r["setting"]: r for r in recs}
    m["mapNsettings"] = str(len(recs))
    targets = out["config"]["agreement_targets"]
    tname = {0.8: "eighty", 0.9: "ninety"}
    for s, tag in NAMES.items():
        if s not in by:
            continue
        r = by[s]
        m[f"mapM{tag}"] = str(r["M"])
        for q in rm.QUANTITIES:
            b, Q = r["point"][q], q.capitalize()
            m[f"mapR{q}{tag}"] = _f(b["R"], 2)
            m[f"mapC{q}{tag}"] = _f(b["C"], 3)
            m[f"mapCref{Q}{tag}"] = _f(b["C_ref"], 3)
            m[f"mapTie{Q}{tag}"] = _pct(b["tie"], 2)
            for t in targets:
                m[f"mapK{tname[t]}{Q}{tag}"] = _k(rm.k_needed(b["R"], t))
    for grp, tag in (("dove_tasks", "Bench"), ("bbh_tasks", "BbhTask")):
        rows = [r for r in recs if r["group"] == grp]
        if not rows:
            continue
        m[f"mapN{tag}"] = str(len(rows))
        m[f"mapItems{tag}"] = "/".join(str(n) for n in sorted({r["n_items"] for r in rows}, reverse=True))
        for q in rm.QUANTITIES:
            Q = q.capitalize()
            R = np.array([r["point"][q]["R"] for r in rows])
            C = np.array([r["point"][q]["C"] for r in rows])
            T = np.array([r["point"][q]["tie"] for r in rows])
            for lab, fn in (("Median", np.median), ("Min", np.min), ("Max", np.max)):
                m[f"mapR{q}{tag}{lab}"] = _f(float(fn(R)), 2)
                m[f"mapC{q}{tag}{lab}"] = _f(float(fn(C)), 3)
            m[f"mapTie{Q}{tag}Mean"] = _pct(float(T.mean()), 1)
            for t in targets:
                ks = [rm.k_needed(r["point"][q]["R"], t) for r in rows]
                fin = [k for k in ks if k is not None and math.isfinite(k)]
                if fin:
                    m[f"mapK{tname[t]}{Q}{tag}Median"] = _k(float(np.median(fin)))
    for tag, rows in mmlu_subject_blocks(recs, targets).items():
        m[f"mapN{tag}"] = str(rows["n"])
        for q, b in rows["quantities"].items():
            Q = q.capitalize()
            for t, kb in b["K"].items():
                m[f"mapK{tname[t]}{Q}{tag}Undef"] = str(kb["undefined"])
                for lab in ("Median", "Min", "Max"):
                    if kb[lab.lower()] is not None:
                        m[f"mapK{tname[t]}{Q}{tag}{lab}"] = _k(kb[lab.lower()])
            m[f"mapDepart{Q}{tag}MeanAbs"] = _f(b["departure_mean_abs"], 3)
    dep = out["departures"]
    for q in rm.QUANTITIES:
        Q = q.capitalize()
        d = np.array([x["departure"] for x in dep if x["quantity"] == q], float)
        d = d[np.isfinite(d)]
        if d.size:
            m[f"mapDepart{Q}Mean"] = _f(float(d.mean()), 3)
            m[f"mapDepart{Q}MeanAbs"] = _f(float(np.abs(d).mean()), 3)
            m[f"mapDepart{Q}MaxAbs"] = _f(float(np.abs(d).max()), 3)
            m[f"mapDepart{Q}Rms"] = _f(float(np.sqrt(np.mean(d ** 2))), 3)
        for part, tag in (("panel_differences", "Panel"), ("pair_noise", "PairNoise"), ("ties_non_gaussian", "Ties")):
            v = np.array([x[part] for x in dep if x["quantity"] == q], float)
            v = v[np.isfinite(v)]
            if v.size:
                m[f"mapPart{tag}{Q}Mean"] = _f(float(v.mean()), 3)
    pt = out["pilot_test"]
    word = {2: "Two", 4: "Four", MAP: "Map"}
    for e in pt["prediction_errors"]:
        if e["scope"] != "all" or e["against"] != "validation" or e["n_pilot"] not in word or not e.get("n"):
            continue
        Q = e["quantity"].capitalize()
        K = "One" if e["target"] == "one_draw" else {2: "GroupsTwo", 4: "GroupsFour", 8: "GroupsEight",
                                                      12: "GroupsTwelve"}.get(e["K"])
        if K is None:
            continue
        m[f"mapPilot{word[e['n_pilot']]}Mae{Q}{K}"] = _f(e["mae"], 3)
        if K == "One":
            m[f"mapPilot{word[e['n_pilot']]}Bias{Q}{K}"] = _f(e["mean"], 3)
            m[f"mapPilot{word[e['n_pilot']]}N{Q}"] = str(e["n"])
            m[f"mapPilot{word[e['n_pilot']]}Undef{Q}"] = str(e["n_undefined"])
            u = e.get("observed_where_undefined")
            if u:
                m[f"mapPilot{word[e['n_pilot']]}UndefObs{Q}"] = _f(u["mean"], 3)
    for r in pt["R_error_summary"]:
        if r["scope"] == "all" and r["n_pilot"] in (2, 4):
            Q = r["quantity"].capitalize()
            m[f"mapPilot{word[r['n_pilot']]}Rmae{Q}"] = _f(r["mae"], 2)
            m[f"mapPilot{word[r['n_pilot']]}Neg{Q}"] = str(r["n_negative"])
    for r in pt["K_needed_summary"]:
        if r["scope"] == "all" and r["n_pilot"] in (2, 4):
            Q = r["quantity"].capitalize()
            m[f"mapPilot{word[r['n_pilot']]}Usable{Q}{tname[float(r['target'])].capitalize()}"] = str(
                r["n_within_factor"])
    m["mapUsableFactor"] = str(pt["usable_factor"])
    hidden = [c for c in out["checks"] if c["quantity"].startswith("tie_") and c["difference"] is not None
              and abs(c["difference"]) > 1e-12]
    m["mapNtieMismatch"] = str(len({c["setting"] for c in hidden}))
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    return m


def added_macros(out, taken=(), pool="base"):
    """Macros of the added contrasts, named as the main settings' with the contrast's tag (mapRbenefitFsGsm, ...):
    the point, K for each target from its R, the observed one-draw validation agreement and the one-draw pilot
    prediction errors at 2 and 4 draws. A name already in `taken` is refused. `pool` names the added pool whose keys are
    read (base: added_settings, ...; another: added_settings_<pool>, ...)."""
    m = {}
    targets = out["config"]["agreement_targets"]
    tname = {0.8: "eighty", 0.9: "ninety"}
    errs = {}
    for x in (out.get(pk("added_pilot_test", pool)) or {}).get("prediction_rows", []):
        if x["target"] == "one_draw" and x["K"] == 1 and x["n_pilot"] in (2, 4):
            errs[(x["setting"], x["n_pilot"], x["quantity"])] = x["error"] if x["defined"] else None
    for r in out.get(pk("added_settings", pool), []):
        tag = r["tag"]
        m[f"mapM{tag}"] = str(r["M"])
        for q in rm.QUANTITIES:
            b, Q = r["point"][q], q.capitalize()
            m[f"mapR{q}{tag}"] = _f(b["R"], 2)
            m[f"mapC{q}{tag}"] = _f(b["C"], 3)
            m[f"mapCref{Q}{tag}"] = _f(b["C_ref"], 3)
            m[f"mapTie{Q}{tag}"] = _pct(b["tie"], 2)
            m[f"mapDepart{Q}{tag}"] = _f(b["departure"], 3)
            m[f"mapCval{Q}{tag}"] = _f(r["validation"]["one_draw"][q]["C"], 3)
            for t in targets:
                m[f"mapK{tname[t]}{Q}{tag}"] = _k(rm.k_needed(b["R"], t))
            for n, w in ((2, "Two"), (4, "Four")):
                e = errs.get((r["setting"], n, q))
                m[f"mapPilot{w}Err{Q}{tag}"] = _f(e, 3)
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    clash = sorted(set(m) & set(taken))
    if clash:
        raise ValueError(f"added-contrast macros already set: {clash}")
    return m


# --------------------------------------------------------------------------- summary


POOLED = ("dove_mmlu", "gsm8k", "gsm8k30", "bbh", "math")
PER_TASK = ("dove_tasks", "bbh_tasks")


def _n(x, nd=3):
    """Plain-text number with a true minus sign; 'undefined' for NaN or None, 'inf' for infinity."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "−")


def _kk(k):
    return "undefined" if k is None else ("∞" if math.isinf(k) else f"{int(k):,}")


def _range(v, nd=3):
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], float)
    return "undefined" if not v.size else f"{_n(float(np.median(v)), nd)} [{_n(float(v.min()), nd)}, {_n(float(v.max()), nd)}]"


def _err(out, n, q, kind, k, scope="all", against="validation"):
    for e in out["pilot_test"]["prediction_errors"]:
        if (e["n_pilot"], e["quantity"], e["target"], e["K"], e["scope"], e["against"]) == (n, q, kind, k, scope, against):
            return e
    return None


def _ksum(out, n, q, t, scope="all"):
    for e in out["pilot_test"]["K_needed_summary"]:
        if (e["n_pilot"], e["quantity"], e["target"], e["scope"]) == (n, q, t, scope):
            return e
    return None


def _rsum(out, n, q, scope="all"):
    for e in out["pilot_test"]["R_error_summary"]:
        if (e["n_pilot"], e["quantity"], e["scope"]) == (n, q, scope):
            return e
    return None


def _items(recs):
    ns = sorted({r["n_items"] for r in recs}, reverse=True)
    if len(ns) == 1:
        return f"{ns[0]:,}"
    return f"{ns[0]:,} ({sum(r['n_items'] != ns[0] for r in recs)} with {'/'.join(f'{x:,}' for x in ns[1:])})"


def added_summary_lines(out, pool="base") -> list:
    """The section on the added contrasts (config/added_contrasts.yaml); empty when the run has none."""
    recs = out.get(pk("added_settings", pool)) or []
    if not recs:
        return []
    cfg = out["config"]
    targets = [f"{t:.2f}" for t in cfg["agreement_targets"]]
    import added_contrasts as ac
    L = ["", f"## Added contrasts ({ac.pool_info(pool)['title']}), kept apart", "",
         f"{len(recs)} contrasts from `config/added_contrasts.yaml`, each through the Dataset 2 route: R and C from the "
         "24 discovery draws, pilot predictions from the first 2 and 4 of them written to "
         "`manifests/predictions_reliability_map_added_<name>.json` before the validation draws were built, and the "
         "validation agreement from the 24 validation draws. Strict scoring. They enter none of the tables, pooled "
         "summaries or macros above. Earlier analyses of these contrasts had read their validation "
         "draws, so the pilot test here describes draws already seen.", "",
         "| Contrast | Models | Items | R benefit | C benefit | Reference | Ties | R accuracy | C accuracy | Reference | Ties |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        cells = [r["title"], str(r["M"]), f"{r['n_items']:,}"]
        for q in rm.QUANTITIES:
            b = r["point"][q]
            cells += [_n(b["R"], 2), _n(b["C"]), _n(b["C_ref"]), f"{100 * b['tie']:.2f}%"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "| Contrast | Benefit 80% | Benefit 90% | Accuracy 80% | Accuracy 90% | Mean C − C_ref, benefit | "
              "Realized differences | Pair noise | Ties and shape |", "|---|---|---|---|---|---|---|---|---|"]
    dep = {(d["setting"], d["quantity"]): d for d in out.get(pk("added_departures", pool), [])}
    for r in recs:
        cells = [_kk(rm.k_needed(r["point"][q]["R"], t)) for q in rm.QUANTITIES for t in cfg["agreement_targets"]]
        d = dep.get((r["setting"], "benefit"), {})
        cells += [_n(d.get("departure")), _n(d.get("panel_differences")), _n(d.get("pair_noise")),
                  _n(d.get("ties_non_gaussian"))]
        L.append(f"| {r['title']} | " + " | ".join(cells) + " |")
    L += ["", "Pilot test, one draw (predicted minus observed validation agreement; no prediction where the pilot R is "
              "below zero):", ""]
    errs = {}
    for x in out[pk("added_pilot_test", pool)]["prediction_rows"]:
        if x["target"] == "one_draw" and x["K"] == 1:
            errs[(x["setting"], x["n_pilot"], x["quantity"])] = x["error"]
    for r in recs:
        bits = []
        for q in rm.QUANTITIES:
            p2, p4 = r["pilot"]["2"][q], r["pilot"]["4"][q]
            km = {t: rm.k_needed(r["point"][q]["R"], float(t)) for t in targets}
            e2, e4 = errs.get((r["setting"], 2, q)), errs.get((r["setting"], 4, q))
            bits.append(f"{q.capitalize()}: pilot R {_n(p2['R'], 2)} (2 draws) and {_n(p4['R'], 2)} (4) against "
                        f"{_n(r['point'][q]['R'], 2)}; K for 80%/90% {_kk(p2['K_needed'][targets[0]])}/"
                        f"{_kk(p2['K_needed'][targets[1]])} and {_kk(p4['K_needed'][targets[0]])}/"
                        f"{_kk(p4['K_needed'][targets[1]])} against {_kk(km[targets[0]])}/{_kk(km[targets[1]])}; "
                        f"observed one-draw validation agreement {_n(r['validation']['one_draw'][q]['C'])}, pilot "
                        f"error {_n(e2)} (2) and {_n(e4)} (4)")
        L.append(f"- **{r['title']}.** " + ". ".join(bits) + ".")
    ch = out.get(pk("added_checks", pool)) or []
    vals = [c for c in ch if c["existing"] is not None and c["quantity"] != "panel_matches"]
    raw = [c["difference_no_tie_tolerance"] for c in vals if c["difference_no_tie_tolerance"] is not None]
    missing = sorted({c["setting"] for c in ch if c["quantity"] == "steps_file_present"})
    panel = [c for c in ch if c["quantity"] == "panel_matches"]
    tie_mm = [c for c in vals if c["quantity"].startswith("tie_") and abs(c["difference"]) > 1e-12]
    if vals:
        line = (f"Checks against each contrast's own Steps 1–3 file: {len(vals)} values (R, C and tie rates, validation "
                "agreement of the first K against the next K); largest absolute difference "
                f"{max(abs(c['difference']) for c in vals):.1e}")
        if raw:
            line += (f" ({max(abs(x) for x in raw):.1e} with ties taken by `np.sign`, as `estimators.pair_agreement` "
                     "does)")
        line += f". Panel sizes: {sum(c['difference'] == 0 for c in panel)} of {len(panel)} match."
        if tie_mm:
            line += f" Tie rates that differ through floating-point ties: {len(tie_mm)}."
        L += ["", line]
    if missing:
        L.append(f"No Steps 1–3 file for: {', '.join(missing)}.")
    return L


def write_summary(out, out_dir):
    """summary.md and summary_clean.md (identical: the text carries no inline citation markers)."""
    recs = out["settings"]
    cfg = out["config"]
    targets = [f"{t:.2f}" for t in cfg["agreement_targets"]]
    by_group = {g: [r for r in recs if r["group"] == g] for g in dict.fromkeys(r["group"] for r in recs)}
    dep = out["departures"]
    L = ["# Reliability map", ""]
    n_bench = len(by_group.get("dove_tasks", []))
    parts = []
    if "dove_mmlu" in by_group:
        parts.append("the pooled DOVE MMLU block")
    if n_bench:
        parts.append(f"{n_bench} single DOVE benchmarks (both five-shot against zero-shot, 5 models)")
    for g, txt in (("gsm8k", "GSM8K with 19 models"), ("gsm8k30", "GSM8K with 30 models"),
                   ("bbh", "BIG-Bench Hard pooled"), ("bbh_tasks", f"{len(by_group.get('bbh_tasks', []))} single "
                                                                   "BIG-Bench Hard tasks"),
                   ("math", "MATH-500 with its numeric items")):
        if g in by_group:
            parts.append(txt + " (chain of thought against direct answer, strict scoring)" if g == "gsm8k" else txt)
    L.append(f"{len(recs)} settings: " + ", ".join(parts[:-1]) + (", and " if len(parts) > 1 else "") + parts[-1]
             + ". " + ("BIG-Bench Hard and the 30-model GSM8K panel are still collecting; `--bbh` and `--gsm8k30` add "
                       "them." if "bbh" not in by_group and "gsm8k30" not in by_group else ""))
    L.append("")
    L.append("R is the cross-draw panel moment over distinct draws divided by the mean within-draw panel variance "
             "(denominator M − 1), for benefit (the gain) and for accuracy (the baseline arm). C is the one-draw "
             "agreement of model-pair orderings between two draws, with equal weights over pairs and a tie counting "
             "one half. Each DOVE configuration bank is enumerated, so there R and C are exact over the bank under "
             "the uniform with-replacement sampler. On Dataset 2 they come from the 24 discovery draws, as in the main "
             "analysis. The reference is C = 1/2 + arcsin(R)/π (the reference curve).")
    L += ["", "## The map", "",
          "| Setting | Models | Items | R benefit | C benefit | Reference | Ties | R accuracy | C accuracy | Reference | Ties |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in POOLED + PER_TASK:
        rows = by_group.get(g)
        if not rows:
            continue
        if g in POOLED:
            for r in rows:
                cells = [GROUP_TITLE[g], str(r["M"]), _items([r]) + (" (56 subjects)" if g == "dove_mmlu" else "")]
                for q in rm.QUANTITIES:
                    b = r["point"][q]
                    cells += [_n(b["R"], 2), _n(b["C"]), _n(b["C_ref"]), f"{100 * b['tie']:.2f}%"]
                L.append("| " + " | ".join(cells) + " |")
        else:
            cells = [f"{len(rows)} {GROUP_TITLE[g].split(',')[0]} {'benchmarks' if g == 'dove_tasks' else 'tasks'}, "
                     "median [min, max]", str(rows[0]["M"]), _items(rows)]
            for q in rm.QUANTITIES:
                cells += [_range([r["point"][q]["R"] for r in rows], 2), _range([r["point"][q]["C"] for r in rows]),
                          _range([r["point"][q]["C_ref"] for r in rows]),
                          f"{100 * float(np.mean([r['point'][q]['tie'] for r in rows])):.1f}% (mean)"]
            L.append("| " + " | ".join(cells) + " |")
    L += ["", "Draws per evaluation for 80% and 90% agreement under the reference (Spearman–Brown, "
          "R_K = K R / (1 + (K − 1) R)), from the map R:", "",
          "| Setting | Benefit 80% | Benefit 90% | Accuracy 80% | Accuracy 90% |", "|---|---|---|---|---|"]
    for g in POOLED + PER_TASK:
        rows = by_group.get(g)
        if not rows:
            continue
        cells = []
        for q in rm.QUANTITIES:
            for t in cfg["agreement_targets"]:
                ks = [rm.k_needed(r["point"][q]["R"], t) for r in rows]
                if len(rows) == 1:
                    cells.append(_kk(ks[0]))
                else:
                    fin = [k for k in ks if k is not None and math.isfinite(k)]
                    cells.append(f"{_kk(float(np.median(fin)))} [{_kk(min(fin))}, {_kk(max(fin))}]" if fin else "undefined")
        name = GROUP_TITLE[g] + (", median [min, max]" if len(rows) > 1 else "")
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- departures
    dq = {q: np.array([x["departure"] for x in dep if x["quantity"] == q], float) for q in rm.QUANTITIES}
    allabs = np.abs(np.concatenate([v[np.isfinite(v)] for v in dq.values()]))
    L += ["", "## Where points leave the curve", ""]
    bands, both = [], []
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0001)):
        cells, means = [f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}"], []
        for q in rm.QUANTITIES:
            v = np.array([x["departure"] for x in dep if x["quantity"] == q and lo <= x["R"] < hi], float)
            v = v[np.isfinite(v)]
            cells += [str(v.size), _n(float(v.mean())) if v.size else "none"]
            means.append(float(v.mean()) if v.size >= 3 else None)
        bands.append("| " + " | ".join(cells) + " |")
        if None not in means:
            both.append(abs(means[0] - means[1]))
    L.append(f"Every point lies within {_n(float(allabs.max()))} of the curve. The mean departure C − C_ref is "
             f"{_n(float(np.nanmean(dq['benefit'])))} for benefit and {_n(float(np.nanmean(dq['accuracy'])))} for "
             "accuracy. Benefit points sit lower on the curve because their R is lower. The table compares the two "
             "quantities at matched R" + (f"; in the bands where each has three or more points, their mean departures "
                                          f"differ by at most {_n(max(both))}." if both else "."))
    L += ["", "| R | Benefit: points | Benefit: mean C − C_ref | Accuracy: points | Accuracy: mean C − C_ref |",
          "|---|---|---|---|---|"] + bands
    L.append(f"\nThe mean absolute departure |C − C_ref| over all points is {_n(float(np.nanmean(np.abs(dq['benefit']))))} "
             f"for benefit and {_n(float(np.nanmean(np.abs(dq['accuracy']))))} for accuracy.")
    ms = out.get("mmlu_subjects", {})
    if ms:
        L += ["", "Over the DOVE MMLU subject benchmarks (`mmlu.<subject>`, MMLU-Pro excluded) and over all DOVE "
              "benchmarks, K for each agreement target from each point's R (median [min, max] of the defined "
              "values; undefined where R is zero or below) and the mean absolute departure:", "",
              "| Benchmarks | N | Quantity | K for 80% | Undefined | K for 90% | Undefined | Mean abs. departure |",
              "|---|---:|---|---|---:|---|---:|---:|"]
        for tag, name in (("MmluSubj", "MMLU subjects"), ("Bench", "All DOVE benchmarks")):
            if tag not in ms:
                continue
            for q, b in ms[tag]["quantities"].items():
                cells = []
                for t in sorted(b["K"], key=float):
                    k = b["K"][t]
                    cells += [("--" if k["median"] is None else f"{k['median']:g} [{k['min']:g}, {k['max']:g}]"),
                              str(k["undefined"])]
                L.append(f"| {name} | {ms[tag]['n']} | {q} | " + " | ".join(cells) + f" | {_n(b['departure_mean_abs'])} |")
    L += ["", "Three fixed-panel references locate each departure ("
              "`src/reliability_map.pairwise_reference`). For a model pair with mean contrast Δ and contrast SD σ over "
              "the draws, a Gaussian contrast gives same-order probability p² + (1 − p)², p = Φ(|Δ|/σ). C_common "
              "averages this over the panel's pairs with one σ for all, the root mean square of the pairs' σ; it has "
              "the same R and differs from C_ref only through the panel's realized differences. C_pairs gives each pair "
              "its own σ. So C − C_ref splits into the realized differences (C_common − C_ref), heterogeneous pair "
              "noise (C_pairs − C_common) and ties with non-Gaussian contrasts (C − C_pairs). Mean parts:", "",
          "| Group | Quantity | Points | Mean C − C_ref | Realized differences | Pair noise | Ties and shape | "
          "Mean tie rate | Pair-noise CV |", "|---|---|---|---|---|---|---|---|---|"]
    for g in POOLED + PER_TASK:
        for q in rm.QUANTITIES:
            x = [d for d in dep if d["group"] == g and d["quantity"] == q]
            if not x:
                continue
            mean = lambda k: _n(float(np.nanmean([d[k] for d in x])))
            L.append(f"| {GROUP_TITLE[g]} | {q} | {len(x)} | {mean('departure')} | {mean('panel_differences')} | "
                     f"{mean('pair_noise')} | {mean('ties_non_gaussian')} | "
                     f"{100 * float(np.mean([d['tie'] for d in x])):.2f}% | {_n(float(np.nanmean([d['pair_noise_cv'] for d in x])), 2)} |")
    L += [""]
    bench = [d for d in dep if d["group"] == "dove_tasks"]
    if bench:
        bt = float(np.mean([d["tie"] for d in bench if d["quantity"] == "benefit"]))
        tn = [d["ties_non_gaussian"] for d in bench]
        pdiff = [d["panel_differences"] for d in bench if d["quantity"] == "accuracy"]
        share = float(np.nanmean([d["tie_share_low_signal"] for d in bench if d["quantity"] == "benefit"]))
        L.append(f"- **Ties.** On the 100-item DOVE benchmarks {100 * bt:.1f}% of draw pairs tie on a benefit ordering. "
                 f"On average {100 * share:.0f}% of the ties fall on pairs whose mean difference is below half its "
                 "contrast SD, where a Gaussian contrast orders two draws the same way with probability at most 0.573. "
                 "Counting a tie as one half therefore moves C little: the ties-and-shape part stays within "
                 f"[{_n(min(tn))}, {_n(max(tn))}] there. Ties lower the decisive rate, which the JSON reports with the "
                 "conditional agreement.")
        mp = float(np.mean(pdiff))
        L.append(f"- **Few models.** Five models give ten pairs, and their true differences need not look like a "
                 "Gaussian sample. For accuracy on the DOVE benchmarks the realized differences move C by "
                 f"{_n(mp)} on average (range {_n(float(np.min(pdiff)))} to {_n(float(np.max(pdiff)))}), the largest of "
                 "the three parts. A positive value means the pairs' differences are more even than a Gaussian sample "
                 "of the same mean square, so more pairs are ordered reliably.")
    gsm = [d for d in dep if d["group"] == "gsm8k"]
    if gsm:
        gb = {d["quantity"]: d for d in gsm}
        pa = gb["accuracy"]["panel_differences"]
        L.append(f"- **Heterogeneous pair noise.** On GSM8K the noise of a pair contrast varies most (coefficient of "
                 f"variation {_n(gb['benefit']['pair_noise_cv'], 2)} for benefit, {_n(gb['accuracy']['pair_noise_cv'], 2)}"
                 " for accuracy), because models near the floor or the ceiling have little prompt noise. Giving each "
                 f"pair its own σ moves C by {_n(gb['benefit']['pair_noise'])} for benefit and "
                 f"{_n(gb['accuracy']['pair_noise'])} for accuracy. The realized differences part is "
                 f"{_n(gb['benefit']['panel_differences'])} for benefit and {_n(pa)} for accuracy"
                 + (": a negative value means a few large differences and many small ones, which lower agreement at a "
                    "given R, the direction of the fixed-panel counterexample (R = 0.70 with agreement 0.550 against "
                    "the curve's 0.747)." if pa < 0 else "."))
    L.append("- The curve holds in expectation under an exchangeable Gaussian random-effects model, where C_common "
             "averages to C_ref. A fixed panel is one realization, and the fixed-panel counterexample shows how far one "
             "can fall from it.")

    # ---------------------------------------------------------------- pilot test
    sizes = [n for n in cfg["pilot_sizes"]] + [MAP]
    lab = {MAP: "map R"}
    L += ["", "## Pilot test", ""]
    L.append("R is estimated from the first 2 and the first 4 pilot draws alone: the samplers' discovery order on the "
             "MMLU block and GSM8K, and on each DOVE benchmark 4 pilot draws of its own "
             f"(`{os.path.relpath(DOVE_TASK_DRAWS, ROOT).replace(chr(92), '/')}`, seed {cfg['seed']} in "
             "`config/reliability_map.yaml`). The predictions were written to "
             "`manifests/predictions_reliability_map_<group>.json` before any validation draw was scored. From the pilot R "
             "the reference predicts (i) the one-draw agreement among the 24 validation draws and (ii) the agreement "
             "between two disjoint groups of K validation draws, C_ref at R_K. The observed (ii) is the mean over "
             f"{out['n_validation_splits']:,} random disjoint splits of the validation draws. An R below zero gives no "
             "prediction. The row \"map R\" predicts from the setting's own map R, which separates the pilot's error "
             "from the curve's.")
    L += ["", "Prediction error, predicted minus observed agreement, over the settings with a prediction (all settings "
              "pooled):", "",
          "| Pilot | Quantity | No prediction (R < 0) | Mean error, one draw | MAE, one draw | MAE K = 2 | MAE K = 4 | "
          "MAE K = 8 | MAE K = 12 | MAE of R | K for 80% within ×" + f"{cfg['usable_factor']}" + " |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    N = len(recs)
    for q in rm.QUANTITIES:
        for n in sizes:
            one = _err(out, n, q, "one_draw", 1)
            cells = [lab.get(n, f"{n} draws"), q, f"{one['n_undefined']} of {N}", _n(one.get("mean")), _n(one.get("mae"))]
            for k in (2, 4, 8, 12):
                e = _err(out, n, q, "groups", k)
                cells.append(_n(e.get("mae")) if e else "not supported")
            rs = _rsum(out, n, q) if n != MAP else None
            cells.append(_n(rs["mae"], 2) if rs else "—")
            ks = _ksum(out, n, q, targets[0])
            cells.append(f"{ks['n_within_factor']} of {N}" if n != MAP else "—")
            L.append("| " + " | ".join(cells) + " |")
    L.append("")
    u2, u4 = (_err(out, n, "benefit", "one_draw", 1) for n in cfg["pilot_sizes"][:2])
    w = [(n, e["observed_where_undefined"]) for n, e in zip(cfg["pilot_sizes"][:2], (u2, u4))
         if e.get("observed_where_undefined")]
    if w:
        L.append("- R_benefit is below zero in " + " and ".join(f"{x['n']} of {N} settings with {n} pilot draws"
                                                               for n, x in w)
                 + ". The observed one-draw validation agreement of those settings averages "
                 + " and ".join(f"{_n(x['mean'])} (range {_n(x['min'])} to {_n(x['max'])}) at {n}" for n, x in w)
                 + ", close to one half: a negative pilot R flags a claim that one draw cannot settle.")
    m2, m4, mm = (_err(out, n, "benefit", "one_draw", 1) for n in sizes[:3])
    a2, a4, am = (_err(out, n, "accuracy", "one_draw", 1) for n in sizes[:3])
    L.append(f"- The one-draw benefit prediction misses by {_n(m2['mae'])} on average with 2 pilot draws and "
             f"{_n(m4['mae'])} with 4, against {_n(mm['mae'])} from the map R; for accuracy {_n(a2['mae'])}, "
             f"{_n(a4['mae'])} and {_n(am['mae'])}. The 4-draw error is {m4['mae'] / m2['mae']:.2f} times the 2-draw "
             f"error for benefit and {a4['mae'] / a2['mae']:.2f} times for accuracy, and "
             f"{m4['mae'] / mm['mae']:.1f} and {a4['mae'] / am['mae']:.1f} times the map-R error. The 2-draw and "
             "4-draw summaries cover different settings, since more settings have a prediction at 4.")
    bk = [_err(out, MAP, q, kind, k, "dove_tasks", "bank") for q in rm.QUANTITIES
          for kind, k in (("one_draw", 1), ("groups", 12))]
    if all(bk):
        L.append(f"- On the DOVE benchmarks the bank's own expected agreement removes the validation draws' sampling "
                 f"noise. Against it the map-R prediction misses by {_n(bk[0]['mae'])} (one draw) and "
                 f"{_n(bk[1]['mae'])} (K = 12) for benefit and by {_n(bk[2]['mae'])} and {_n(bk[3]['mae'])} for "
                 f"accuracy; the mean errors at K = 12 are {_n(bk[1]['mean'])} and {_n(bk[3]['mean'])} (negative: the "
                 "bank agrees more than the reference predicts).")
    for s, title in (("gsm8k19", "GSM8K"), ("dove_mmlu_block", "MMLU block"), ("gsm8k30", "GSM8K-30"),
                     ("bbh:pooled", "BIG-Bench Hard"), ("math", "MATH-500")):
        r = next((x for x in recs if x["setting"] == s), None)
        if r is None:
            continue
        bits = []
        for q in rm.QUANTITIES:
            p2, p4 = r["pilot"]["2"][q], r["pilot"]["4"][q]
            obs = r["validation"]["one_draw"][q]["C"]
            km = {t: rm.k_needed(r["point"][q]["R"], float(t)) for t in targets}
            bits.append(f"{q.capitalize()}: pilot R {_n(p2['R'], 2)} (2 draws) and {_n(p4['R'], 2)} (4) against "
                        f"{_n(r['point'][q]['R'], 2)}; K for 80%/90% {_kk(p2['K_needed'][targets[0]])}/"
                        f"{_kk(p2['K_needed'][targets[1]])} and {_kk(p4['K_needed'][targets[0]])}/"
                        f"{_kk(p4['K_needed'][targets[1]])} against {_kk(km[targets[0]])}/{_kk(km[targets[1]])}; "
                        f"observed one-draw validation agreement {_n(obs)}")
        L.append(f"- **{title}.** " + ". ".join(bits) + ".")
    ks2, ks4 = (_ksum(out, n, "benefit", targets[0]) for n in cfg["pilot_sizes"][:2])
    ka2, ka4 = (_ksum(out, n, "accuracy", targets[0]) for n in cfg["pilot_sizes"][:2])
    f = cfg["usable_factor"]
    miss = [x for x in out["pilot_test"]["K_needed"] if x["n_pilot"] == cfg["pilot_sizes"][-1]
            and x["quantity"] == "benefit" and x["target"] == targets[0] and x["within_factor"] is not True]
    hi = sum(1 for x in miss if x["K_pilot"] is not None and x["K_map"] is not None and x["K_pilot"] > f * x["K_map"])
    lo_ = sum(1 for x in miss if x["K_pilot"] is not None and x["K_map"] is not None and x["K_pilot"] < x["K_map"] / f)
    L.append(f"- A pilot counts as usable for planning when its K for 80% lies within a factor of {f} of the K from "
             f"the map R. For benefit that holds in {ks2['n_within_factor']} of {N} settings with 2 draws and "
             f"{ks4['n_within_factor']} with 4; for accuracy in {ka2['n_within_factor']} and {ka4['n_within_factor']}. "
             f"Of the {len(miss)} benefit settings that miss with {cfg['pilot_sizes'][-1]} draws, "
             f"{sum(1 for x in miss if x['K_pilot'] is None)} have no pilot prediction, {hi} ask for more than {f} "
             f"times the draws and {lo_} for less than 1/{f}.")

    # ---------------------------------------------------------------- checks
    ch = out["checks"]
    L += ["", "## Checks against the existing results", ""]
    rch = [c for c in ch if c["quantity"].startswith("R_")]
    L.append(f"- R: {len(rch)} values compared with `results/dataset1/dove_mmlu.json`, `results/dataset1/dove_tasks.json` "
             f"and `results/dataset2/gsm8k.json`; largest absolute difference "
             f"{max(abs(c['difference']) for c in rch):.1e}.")
    cch = [c for c in ch if c["quantity"].split("_")[0] in ("C", "tie") and "validation" not in c["quantity"]]
    raw = max(abs(c["difference_no_tie_tolerance"]) for c in cch if c["difference_no_tie_tolerance"] is not None)
    L.append(f"- C and tie rates: {len(cch)} values. With ties taken by `np.sign` of the raw difference, as "
             f"`estimators.pair_agreement` does, every value matches (largest difference {raw:.1e}).")
    vch = [c for c in ch if "validation_first_split" in c["quantity"]]
    if vch:
        L.append(f"- Validation agreement of the first K against the next K: {len(vch)} values; largest difference "
                 f"{max(abs(c['difference']) for c in vch):.1e}.")
    other = [c for c in ch if c["quantity"] in ("n_configurations", "n_items", "panel_matches", "present_in_existing")]
    if other:
        L.append(f"- Item counts, configuration counts and panel sizes: {len(other)} values; "
                 f"{sum(c['difference'] != 0 for c in other)} differ.")
    tie_mm = [c for c in ch if c["quantity"].startswith("tie_") and abs(c["difference"]) > 1e-12]
    if tie_mm:
        L.append(f"- **Mismatch, ties hidden by floating point.** Here a difference below {out['tie_tolerance_pp']:g} pp "
                 "counts as a tie (`src/reliability_map.py` explains why). Equal count differences can give gains that "
                 "differ in the last bit, and `np.sign` then counts a true tie as decisive. This changes "
                 f"{len({c['setting'] for c in tie_mm})} settings:")
        items = {r["setting"]: r["n_items"] for r in recs}
        same3 = 0
        for c in tie_mm:
            q = c["quantity"].split("_", 1)[1]
            cc = next(x for x in ch if x["setting"] == c["setting"] and x["quantity"] == f"C_{q}")
            same3 += round(cc["ours"], 3) == round(cc["existing"], 3)
            L.append(f"  - {c['setting']} ({items[c['setting']]:,} items), {q}: tie rate {100 * c['ours']:.3f}% against "
                     f"{100 * c['existing']:.3f}% reported; C {cc['ours']:.5f} against {cc['existing']:.5f}.")
        L.append(f"  At three decimals C is unchanged in {same3} of these {len(tie_mm)} cases. The existing results "
                 "files are not modified.")

    for pool in out.get("added_pools", ["base"]):
        L += added_summary_lines(out, pool)

    # ---------------------------------------------------------------- choices
    L += ["", "## Choices made here", "",
          "- Map points: exact bank values on DOVE (matching the existing Dataset 1 results), the 24 discovery draws on "
          "Dataset 2 (matching the main analysis, and disjoint from the validation draws of the pilot test).",
          f"- Ties: a difference of at most {out['tie_tolerance_pp']:g} pp is a tie (see the checks).",
          "- The 71 DOVE benchmarks are those of `scripts/run_dataset1_tasks.py`, with its exclusions "
          f"({len(out['excluded_dove_benchmarks'])} benchmarks, listed in the JSON) and its common-configuration rule.",
          "- Observed agreement between K-draw groups is the mean over random disjoint splits of the 24 validation "
          "draws. The split of the first K against the next K, as in the main analyses, is in the JSON. K = 12 uses "
          "all 24 draws, so its splits are strongly dependent.",
          "- No prediction is made from a pilot R below zero, and such settings leave the error summaries; their "
          "observed agreement is reported apart. Error summaries therefore describe the settings where a pilot "
          "gives an answer.",
          f"- \"Usable\" pilot: K for 80% within a factor of {cfg['usable_factor']} of the K from the map R, fixed in "
          "`config/reliability_map.yaml` before the run.",
          "- The fixed-panel references are plug-in values. With sampled draws (Dataset 2) |Δ| carries noise, a small "
          "upward pull for pairs with true difference near zero.",
          "- Strict scoring on Dataset 2. The DOVE intervention is five-shot against zero-shot; Dataset 2's is zero-shot "
          "chain of thought against direct answer."]
    L += ["", "## Files", "",
          "- `results/reliability_map/reliability_map.json`: every setting (map point, pilot predictions, validation "
          "agreement, bank agreement on DOVE), the departure table, the pilot-test summaries, the checks and a "
          "`macros` object for `paper/make_numbers.py`.",
          "- `manifests/predictions_reliability_map_<group>.json`: pilot predictions written before the validation draws "
          f"were scored; `{os.path.relpath(DOVE_TASK_DRAWS, ROOT).replace(chr(92), '/')}`: the DOVE benchmarks' pilot and "
          "validation draws.",
          "- Code: `src/reliability_map.py`, `scripts/run_reliability_map.py`, tests in `tests/test_reliability_map.py`; "
          "settings in `config/reliability_map.yaml`.", ""]
    text = "\n".join(L)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


# --------------------------------------------------------------------------- main


def compute(args, cfg):
    t0 = time.time()
    n_splits = 200 if args.quick else int(cfg["validation_splits"])
    n_mc = 200 if args.quick else int(cfg["bank_monte_carlo"])
    records, frozen, excluded = [], {}, []
    pred = lambda g: os.path.join(MAN, f"predictions_{LABEL}_{g}.json")
    blk = dove_mmlu_inputs()
    rows, frozen["dove_mmlu"] = run_dove([blk], "dove_mmlu", blk["models"], blk["draws_sha"], cfg, n_splits, n_mc,
                                         pred("dove_mmlu"), args.overwrite_frozen)
    records += rows
    print(f"DOVE MMLU block done ({time.time() - t0:.0f} s)")
    tasks, excluded, sha = dove_task_inputs(cfg)
    rows, frozen["dove_tasks"] = run_dove(tasks, "dove_tasks", tasks[0]["models"], sha, cfg, n_splits, n_mc,
                                          pred("dove_tasks"), args.overwrite_frozen)
    records += rows
    print(f"{len(tasks)} DOVE benchmarks done ({time.time() - t0:.0f} s)")
    d2_inputs = [("gsm8k", lambda: gsm8k19_inputs(args.gsm8k_scored))]
    if args.gsm8k30:
        d2_inputs.append(("gsm8k30", lambda: gsm8k30_inputs(args.gsm8k_scored, args.gsm8k_extension)))
    if args.bbh:
        d2_inputs.append(("bbh", lambda: bbh_inputs(args.bbh_scored)))
    if getattr(args, "math", False):
        d2_inputs.append(("math", lambda: math_inputs(args.math_scored)))
    for label, get in d2_inputs:
        rows, frozen[label] = run_d2(get(), label, cfg, n_splits, pred(label), args.overwrite_frozen)
        records += rows
        print(f"{label} done ({time.time() - t0:.0f} s)")
    out = {"label": LABEL, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "plan": "reliability map",
           "units": "scores in percentage points; R and C dimensionless", "config": cfg,
           "config_sha256": sha256_json(cfg), "quick": bool(args.quick),
           "n_validation_splits": n_splits, "n_bank_monte_carlo": n_mc, "tie_tolerance_pp": rm.TIE_TOL,
           "frozen_pilot_predictions": frozen, "excluded_dove_benchmarks": excluded, "settings": records}
    out["checks"] = check_rows(records)
    out["departures"] = departure_table(records)
    out["departure_stats"] = {k: group_stats(out["departures"], k)
                              for k in ("R", "C", "departure", "tie", "panel_differences", "pair_noise",
                                        "ties_non_gaussian", "pair_noise_cv")}
    out["pilot_test"] = summaries(records, cfg)
    out["mmlu_subjects"] = mmlu_subject_blocks(records, cfg["agreement_targets"])
    out["macros"] = macros(out)
    if getattr(args, "added", False):
        import added_contrasts as ac
        con = ac.load_config(getattr(args, "added_config", None) or ac.CONFIG)
        out["added_config"] = {"path": "config/added_contrasts.yaml", "sha256": sha256_json(con),
                               "note": "kept apart from the main settings: they enter none of the summaries above"}
        # one block of keys per pool: pool base under the unsuffixed keys, a later pool under
        # <key>_<pool>, so no pooled summary of an earlier pool changes when a pool is added
        out["added_pools"] = list(ac.pools(con))
        for pool, members in ac.pools(con).items():
            added = []
            for name in members:
                label = f"added_{name}"
                rows, frozen[label] = run_d2(added_inputs(name, con), label, cfg, n_splits, pred(label),
                                             args.overwrite_frozen)
                added += rows
                print(f"{label} done ({time.time() - t0:.0f} s)")
            out[pk("added_settings", pool)] = added
            out[pk("added_checks", pool)] = added_check_rows(added)
            out[pk("added_departures", pool)] = departure_table(added)
            out[pk("added_pilot_test", pool)] = summaries(added, cfg)
            out["macros"].update(added_macros(out, taken=out["macros"], pool=pool))
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "dove_revision": io.REVISION}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="200 splits and Monte Carlo draws (smoke test)")
    ap.add_argument("--gsm8k-scored", default=os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet"))
    ap.add_argument("--gsm8k30", action="store_true", help="add the 30-model GSM8K panel (src/d2_panels.inputs)")
    ap.add_argument("--gsm8k-extension", nargs="+", help="default data/processed/gsm8k_ext/scored_*.parquet")
    ap.add_argument("--bbh", action="store_true", help="add BIG-Bench Hard, pooled and per task")
    ap.add_argument("--bbh-scored", help="default data/processed/bbh/scored.parquet")
    ap.add_argument("--math", action="store_true", help="add MATH-500")
    ap.add_argument("--math-scored", help="default data/processed/math/scored.parquet")
    ap.add_argument("--added", action="store_true",
                    help="add the contrasts of config/added_contrasts.yaml, kept apart from the main settings")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    path = os.path.join(a.out, f"{LABEL}{'_quick' if a.quick else ''}.json")
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    out = compute(a, cfg)
    dump(out, path)
    if not out.get("quick"):
        write_summary(out, a.out)
    print(f"written {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
