r"""Analysis of five-shot against zero-shot forced direct on GSM8K-29 and BIG-Bench Hard-28.

PowerShell (from the repository root), after scripts\score_fewshot.py --task gsm8k and --task bbh:
    .\.venv\Scripts\python.exe scripts\run_fewshot.py
    .\.venv\Scripts\python.exe scripts\run_fewshot.py --dry-run --quick     # the requested arm stands in for P

B is the zero-shot forced-direct arm (data/processed/forced/<task>_scored.parquet) and P the five-shot arm
(data/processed/fewshot/<task>_scored.parquet). Per task:
1. map pilot predictions from the first 2 and 4 discovery draws, saved to manifests/predictions_fewshot_map.json before
   any validation draw is built;
2. Steps 1-3 of the main analysis (scripts/run_dataset2.py) with B as the baseline and P as the intervention, the rules
   saved to manifests/predictions_fewshot_<task>.json (and _lenient) before the validation draws are built;
3. then, on the validation draws: the map point (R and C of benefit and accuracy, observed one-draw agreement), and on
   the discovery draws the capability slope of the five-shot gain on zero-shot forced accuracy, with the model-level
   summaries (model and family bootstrap, leave-one-model-out and leave-one-family-out; 10,000 resamples,
   seed 20261117), and per model the two accuracies and the gain.
Outputs: results/fewshot/fewshot.json, fewshot_<task>.json, summary.md, summary_clean.md, tables/per_model_<task>.csv,
tables_fewshot_<task>/. A dry run writes under results/fewshot/dry_run/, its frozen files included.

MATH-500, after scripts\score_fewshot.py --task math; it runs alone, with its own folder and files:
    .\.venv\Scripts\python.exe scripts\run_fewshot.py --tasks math
B is the forced MATH-500 arm (data/processed/forced/math_scored.parquet), P the five-shot MATH-500 arm. Map pilot
predictions in manifests/predictions_fewshot_map_math.json, Steps 1-3 rules in manifests/predictions_fewshot_math.json
(and _lenient), outputs in results/fewshot/math/.

For other analyses: fewshot_arrays(task, split) returns the (B, P) arrays of one split; fewshot_available(task) says
whether the task's five-shot scores exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FS_PROCESSED = os.path.join(ROOT, "data", "processed", "fewshot")
OUT = os.path.join(ROOT, "results", "fewshot")
MANIFESTS = os.path.join(ROOT, "manifests")
TASKS = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 (29 models, 398 items)"}
RUN1_TASKS = ("gsm8k", "bbh")
SEED = 20261117


def pair_arrays(fd_load, fs_load, units, panel, items, value="strict_correct"):
    """(B zero-shot forced, P five-shot forced), each (K, M) pp, through the cell table of the main analysis."""
    from src import dataset2 as d2
    values = tuple(dict.fromkeys(tuple(d2.VALUES) + (value,)))
    comb = pd.concat([fd_load(units).assign(arm="direct_answer"), fs_load(units).assign(arm="zero_shot_cot")],
                     ignore_index=True)
    cells = d2.cell_table(comb, panel, units, items, values=values)
    return d2.unit_arrays(cells, units, panel, value)


def fewshot_available(task) -> bool:
    """Whether the five-shot scored table of a task exists (data/processed/fewshot/<task>_scored.parquet)."""
    return os.path.exists(os.path.join(FS_PROCESSED, f"{task}_scored.parquet"))


def fewshot_arrays(task, split, value="strict_correct", x=None):
    """(B, P, x): zero-shot forced (B) and five-shot forced (P) accuracies of one split ("discovery" or "validation"),
    each (K, M) in pp, rows in the sampler's draw order, columns in x["panel"] order; x is the inputs dict of
    inputs(task, False), built when not given and returned for reuse (load the discovery split first, the validation
    split after any rule is saved)."""
    if split not in ("discovery", "validation"):
        raise ValueError(f"split {split!r}")
    x = x if x is not None else inputs(task, False)
    units = x["disc"] if split == "discovery" else x["val"]
    B, P = pair_arrays(x["fd_load"], x["fs_load"], units, x["panel"], x["items"], value=value)
    return B, P, x


def combined_loader(fd_load, fs_load):
    def load(ids):
        return pd.concat([fd_load(ids).assign(arm="direct_answer"), fs_load(ids).assign(arm="zero_shot_cot")],
                         ignore_index=True)
    return load


def per_model_rows(panel, B, P):
    return [{"model_key": k, "zero_shot_forced": float(B[:, j].mean()), "five_shot_forced": float(P[:, j].mean()),
             "gain": float((P - B)[:, j].mean())} for j, k in enumerate(panel)]


def inputs(task, dry_run):
    import run_forced_direct as rfd
    inp, s, items, items_info = rfd.collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, fd_prov, _ = rfd.forced_inputs(task, panel, s, items_info, inp["load"], False)
    saved = rfd.PROCESSED
    rfd.PROCESSED = FS_PROCESSED
    try:
        fs_load, fs_prov, fs_box = rfd.forced_inputs(task, panel, s, items_info, inp["load"], dry_run)
    finally:
        rfd.PROCESSED = saved
    return {"inp": inp, "sampler": s, "items": items, "panel": panel, "fd_load": fd_load, "fd_prov": fd_prov,
            "fs_load": fs_load, "fs_prov": fs_prov, "fs_box": fs_box,
            "disc": list(s["discovery_config_ids"]), "val": list(s["validation_config_ids"])}


def run(a) -> dict:
    t0 = time.time()
    import run_dataset2
    import run_forced_direct as rfd
    from run_model_resampling import block
    from run_index_checks import jsonable
    from src import dataset2 as d2, reliability_map as rm
    from src.sampling import sha256_json
    tasks = list(dict.fromkeys(a.tasks))
    if "math" in tasks and tasks != ["math"]:
        raise SystemExit("--tasks math runs alone (its own folder and frozen files); run gsm8k bbh separately")
    sfx = "_math" if tasks == ["math"] else ""
    base = os.path.join(a.out, "math") if sfx else a.out
    out_dir = os.path.join(base, "dry_run") if a.dry_run else base
    pred_dir = os.path.join(out_dir, "frozen") if a.dry_run else a.pred_dir
    os.makedirs(pred_dir, exist_ok=True)
    mcfg = yaml.safe_load(open(rfd.MAP_CONFIG, encoding="utf-8"))
    icfg = yaml.safe_load(open(rfd.INST_CONFIG, encoding="utf-8"))
    method, alpha = d2.load_selection(rfd.SELECTION)["H"], float(icfg["alpha"])
    n_splits = 200 if a.quick else int(mcfg["validation_splits"])
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)
    S = {}
    for task in tasks:
        x = inputs(task, a.dry_run)
        x["D"] = pair_arrays(x["fd_load"], x["fs_load"], x["disc"], x["panel"], x["items"])
        S[task] = x
        print(f"{task}: discovery arrays ({time.time() - t0:.0f} s)", flush=True)

    # 1. map pilot predictions, discovery draws only
    preds = {t: {str(n): rm.pilot_prediction(x["D"][0][:n], x["D"][1][:n], mcfg["group_sizes"], mcfg["agreement_targets"])
                 for n in mcfg["pilot_sizes"]} for t, x in S.items()}
    fr = {"label": "fewshot_map", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
          "models": {t: x["panel"] for t, x in S.items()}, "samplers": {t: x["sampler"]["sha256"] for t, x in S.items()},
          "note": "pilot predictions of the five-shot map points (zero-shot forced as the baseline; R from the first 2 "
                  "and 4 discovery draws), written before the validation draws are built", "settings": jsonable(preds)}
    fr["sha256"] = sha256_json(fr["settings"])
    map_name = f"predictions_fewshot_map{sfx}.json"
    frozen_map = d2.write_frozen(fr, os.path.join(pred_dir, map_name), overwrite=a.overwrite_frozen)

    # 2. Steps 1-3 (rules frozen inside, then the validation draws)
    R = {"analysis": "five-shot against zero-shot forced direct ("
                     + ("MATH-500)" if sfx else "GSM8K-29 and BIG-Bench Hard-28)"),
         "dry_run": bool(a.dry_run), "created": time.strftime("%Y-%m-%d %H:%M"), "units": "percentage points",
         "slope_method": method, "alpha": alpha, "seed": SEED, "tasks": {}, "map_points": []}
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for task, x in S.items():
        label = f"fewshot_{task}"
        prov = {"collection": x["inp"]["provenance"], "zero_shot_forced": x["fd_prov"], "five_shot_forced": x["fs_prov"],
                "arms": "zero-shot forced relabelled direct_answer, five-shot forced relabelled zero_shot_cot"}
        steps = run_dataset2.run(combined_loader(x["fd_load"], x["fs_load"]), x["sampler"], x["panel"],
                                 x["inp"]["families"], x["inp"]["tiers"], x["items"], out_dir,
                                 os.path.join(pred_dir, f"predictions_{label}.json"), label, n_boot, n_ctrl,
                                 overwrite_frozen=a.overwrite_frozen, provenance=prov,
                                 tables_dir=os.path.join(out_dir, f"tables_{label}"))
        B, P = x["D"]
        Bv, Pv = pair_arrays(x["fd_load"], x["fs_load"], x["val"], x["panel"], x["items"])
        pts, point, obs = rfd.map_points(task, B, P, Bv, Pv, frozen_map, mcfg, n_splits, len(x["panel"]), len(x["items"]))
        for p in pts:
            p["setting"] = f"fewshot:{task}"
        fams = list(x["inp"]["families"])
        slope = block(B, P, fams, method, alpha, np.random.default_rng(SEED), x["panel"])
        rows = per_model_rows(x["panel"], B, P)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tables", f"per_model_{task}.csv"), index=False)
        R["tasks"][task] = {"title": TASKS[task], "models": x["panel"], "M": len(x["panel"]), "n_items": len(x["items"]),
                            "step2": steps.get("step2"), "map": {"point": point, "validation_one_draw": obs["one_draw"]},
                            "capability_slope": slope, "per_model": rows,
                            "means": {k: float(np.mean([r[k] for r in rows]))
                                      for k in ("zero_shot_forced", "five_shot_forced", "gain")},
                            "steps_file": os.path.relpath(os.path.join(out_dir, f"{label}.json"), ROOT).replace("\\", "/")}
        R["map_points"] += pts
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["frozen_map"] = {"path": os.path.relpath(os.path.join(pred_dir, map_name), ROOT).replace("\\", "/"),
                       "sha256": frozen_map["sha256"]}
    R["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "fewshot.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    txt = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return R


def est(b):
    if not isinstance(b, dict):
        return "--"
    def num(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else str(v)
    e = b.get("estimate")
    c = b.get("ci95")
    return "--" if e is None else num(e) + (f" [{num(c[0])}, {num(c[1])}]" if c else "")


def summary(R) -> str:
    which = "MATH-500" if "math" in R["tasks"] else "GSM8K-29 and BIG-Bench Hard-28"
    L = [f"# Five-shot against zero-shot forced direct ({which})" + (" -- dry run" if R["dry_run"] else ""), "",
         "B: zero-shot forced direct; P: five-shot forced direct (five demonstrations per configuration, "
         "manifests/fewshot_demos_<task>.json). Points in percent; R and C of benefit from the map rule.", "",
         "| Task | M | B mean | P mean | Gain | V_delta | H | R of benefit | C (validation, one draw) | Slope of gain on B | "
         "Model bootstrap | Family bootstrap |", "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|"]
    for t, T in R["tasks"].items():
        s2 = T.get("step2") or {}
        pb = T["map"]["point"]["benefit"]
        ob = T["map"]["validation_one_draw"]["benefit"]
        cs = T["capability_slope"]
        mb, fb = cs["model_bootstrap"]["ci95"], cs["family_bootstrap"]["ci95"]
        L.append(f"| {T['title']} | {T['M']} | {T['means']['zero_shot_forced']:.1f} | {T['means']['five_shot_forced']:.1f} | "
                 f"{T['means']['gain']:+.1f} | {est(s2.get('V_delta'))} | {est(s2.get('H'))} | "
                 f"{'--' if pb.get('R') is None else round(pb['R'], 3)} | {'--' if ob.get('C') is None else round(ob['C'], 3)} | "
                 f"{cs['slope']:+.3f} | {'--' if not mb else f'[{mb[0]:+.2f}, {mb[1]:+.2f}]'} | "
                 f"{'--' if not fb else f'[{fb[0]:+.2f}, {fb[1]:+.2f}]'} |")
    L += ["", f"Map pilot predictions: `{R['frozen_map']['path']}`; Steps 1-3 rules in "
              "`manifests/predictions_fewshot_<task>.json`, saved before the validation draws were built."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(RUN1_TASKS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=MANIFESTS)
    ap.add_argument("--dry-run", action="store_true", help="the requested-direct arm stands in for the five-shot arm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
