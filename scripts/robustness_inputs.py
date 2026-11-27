r"""Inputs of the robustness checks of the full re-run: for GSM8K-29, BIG-Bench Hard-28 and
MATH-500 (29 models), the three arms' accuracies per prompt draw and model (strict scoring, pp) on the 24 discovery and
the 24 validation draws, with the panel facts. Cached in data/processed/robustness/<task>_arrays.npz and
<task>_facts.json (git-ignored); the cache is rebuilt when a scored table is newer than it.

Used by scripts/run_model_resampling.py, run_readable_sweep.py, run_heldout_capability.py, run_lowr_map.py and
run_arm_effect.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = os.path.join(ROOT, "data", "processed", "robustness")
TASKS = ("gsm8k", "bbh", "math")
NAMES = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 (29 models, 398 items)"}
SCORED = {"gsm8k": ["data/processed/gsm8k/scored.parquet", "data/processed/gsm8k_ext", "data/processed/forced/gsm8k_scored.parquet"],
          "bbh": ["data/processed/bbh/scored.parquet", "data/processed/forced/bbh_scored.parquet"],
          "math": ["data/processed/math/scored.parquet", "data/processed/forced/math_scored.parquet"]}
CLEAN_MIN = 95.0
ARMS = ("requested", "forced", "cot")


def added_tasks() -> tuple:
    """The tasks of the section `tasks` of config/added_contrasts.yaml (HotpotQA), registered here with their
    scored table and title. ri.TASKS keeps the three main tasks, so an analysis adds these only where it asks."""
    import added_contrasts as ac
    tk = ac.load_tasks()
    for t, e in tk.items():
        SCORED.setdefault(t, [e["scored"]])
        NAMES.setdefault(t, e["title"])
    return tuple(tk)


def _newest(paths):
    t = 0.0
    for p in paths:
        p = os.path.join(ROOT, p)
        if os.path.isdir(p):
            t = max([t] + [os.path.getmtime(os.path.join(p, f)) for f in os.listdir(p)])
        elif os.path.exists(p):
            t = max(t, os.path.getmtime(p))
    return t


def _build(task):
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    from run_transfer_tasks import eci_for
    from src import bbh_d2
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    params = {e["repo"]: e["params_b"] for e in man["panel"]}
    reg = bbh_d2.load_registries()
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    arrays = {}
    for split in ("discovery", "validation"):
        units = list(s[f"{split}_config_ids"])
        Breq, Bfd, Pc = three_arm_arrays(inp["load"], fd_load, units, panel, items)
        arrays.update({f"{split}_requested": Breq, f"{split}_forced": Bfd, f"{split}_cot": Pc})
    pm_path = os.path.join(ROOT, "results", "forced_direct", "tables", f"per_model_{task}.csv")
    # an added task (HotpotQA) has no forced-direct readability table: readable_pct is None, and the analyses that read
    # a readable panel leave it out for that task
    pm = pd.read_csv(pm_path).set_index("model_key") if task in TASKS else None
    facts = {"task": task, "panel": panel, "families": list(inp["families"]), "eci": [float(x) for x in eci_for(panel)],
             "repo": [reg[k]["hf_id"] for k in panel], "params_b": [float(params[reg[k]["hf_id"]]) for k in panel],
             "readable_pct": None if pm is None else [float(pm.loc[k, "closed_and_readable_pct"]) for k in panel],
             "n_items": len(items),
             "discovery_config_ids": [str(u) for u in s["discovery_config_ids"]],
             "validation_config_ids": [str(u) for u in s["validation_config_ids"]]}
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f"{task}_arrays.npz"), **arrays)
    json.dump(facts, open(os.path.join(CACHE, f"{task}_facts.json"), "w", encoding="utf-8"), indent=1)


def load(task):
    """{'facts': ..., 'discovery': {arm: (K, M)}, 'validation': {arm: (K, M)}} for a task, built if needed."""
    if task not in SCORED:
        added_tasks()
    npz, js = os.path.join(CACHE, f"{task}_arrays.npz"), os.path.join(CACHE, f"{task}_facts.json")
    if not (os.path.exists(npz) and os.path.exists(js)) or os.path.getmtime(npz) < _newest(SCORED[task]):
        print(f"{task}: building the arrays from the scored tables", flush=True)
        _build(task)
    z = np.load(npz)
    out = {"facts": json.load(open(js, encoding="utf-8"))}
    for split in ("discovery", "validation"):
        out[split] = {arm: z[f"{split}_{arm}"] for arm in ARMS}
    return out


def gsm8k_readable_panel():
    """The 23 models of the GSM8K-29 readable panel (results/outcomes/outcomes.json, clean_models), the MATH-500 panel."""
    o = json.load(open(os.path.join(ROOT, "results", "outcomes", "outcomes.json"), encoding="utf-8"))
    return list(o["tasks"]["gsm8k"]["clean_models"])


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else "nan")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_outputs(out_dir, R, summary_lines, macros=None):
    """results/<name>/<name>.json (with macros), summary.md and summary_clean.md (identical: no citation markers)."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(out_dir)
    if macros is not None:
        R["macros"] = macros
    json.dump(jsonable(R), open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8"), indent=1)
    text = "\n".join(summary_lines) + "\n"
    for f in ("summary.md", "summary_clean.md"):
        open(os.path.join(out_dir, f), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    for t in sys.argv[1:] or TASKS:
        x = load(t)
        print(t, {k: v.shape for k, v in x["discovery"].items()})
