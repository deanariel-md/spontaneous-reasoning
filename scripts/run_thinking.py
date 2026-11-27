r"""Native reasoning (thinking on) as an intervention arm beside prompted chain of
thought, with forced direct as the baseline.

PowerShell (from the repository root), after scripts\score_thinking.py --task gsm8k and --task math (and, for MATH-500,
scripts\score_forced_direct.py --task math):
    cmd /c ".venv\Scripts\python.exe scripts\run_thinking.py 2>&1"
Dry run before the data exist (the thinking arm replaced by the collection's chain-of-thought arm; written only under
--out, never to manifests/):
    cmd /c ".venv\Scripts\python.exe scripts\run_thinking.py --dry-run --quick --tasks gsm8k --out <scratch dir> 2>&1"

Panel per task: the thinking checkpoints of the task's panel (data/requests/thinking/build_record.json).
Four arms per model on the collection's items and frozen sampler: forced direct (data/processed/forced/<task>_scored.
parquet), requested direct and prompted chain of thought (the collection's outputs, thinking off), and native reasoning
(data/processed/thinking/<task>_scored.parquet, the requested-direct prompts with the thinking switch on).

Order:
1. discovery draws of the arms;
2. the map's pilot predictions (R from the first 2 and 4 draws, config/reliability_map.yaml) saved to
   manifests/predictions_thinking_map.json;
3. Steps 1-3 by scripts/run_dataset2.py's run() with forced direct as the baseline and native reasoning as the
   intervention: its prediction rules are saved to manifests/predictions_thinking_<task>.json (and _lenient) before it
   builds the validation draws; its output is results/thinking/thinking_<task>.json;
4. validation draws; then the two interventions side by side.

Side by side, on the discovery draws: src/dataset2.step2 of prompted chain of thought against forced direct and of
native reasoning against prompted chain of thought (calibrated methods); the cross-draw moments of the two benefits over
forced direct (distinct draws) with their latent correlation and a leave-one-draw-out jackknife
(descriptive); the correlation of the models' mean benefits and the share of model pairs the two benefits order alike
(a difference of at most 1e-9 pp is a tie); and the slope of each benefit on forced-direct accuracy on three scales
(scripts/run_instability.py's scale_block). Per model: the four arms, the benefits, and the native-reasoning outputs'
length, closure of the reasoning block and share cut at the cap.

Outputs: results/thinking/thinking.json (with "macros", prefix thi, and "map_points"), thinking_<task>.json, summary.md,
summary_clean.md and tables/per_model_<task>.csv.
"""
import argparse
import json
import os
import platform
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
from src import dataset2 as d2, instability as ins, intervals as iv, reliability_map as rm, sampling  # noqa: E402
from src.estimators import cross_moment, tie_sign  # noqa: E402
from run_forced_direct import ci, collection_inputs, dump, fmt, forced_inputs, hset, jack, jsonable  # noqa: E402

OUT = os.path.join(ROOT, "results", "thinking")
MANIFESTS = os.path.join(ROOT, "manifests")
PROCESSED = os.path.join(ROOT, "data", "processed", "thinking")
RECORD = os.path.join(ROOT, "data", "requests", "thinking", "build_record.json")
EXCLUSIONS = os.path.join(ROOT, "manifests", "thinking_exclusions.json")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
MAP_CONFIG = os.path.join(ROOT, "config", "reliability_map.yaml")
INST_CONFIG = os.path.join(ROOT, "config", "instability.yaml")
TASKS = {"gsm8k": {"title": "GSM8K", "tag": "Gsm"}, "math": {"title": "MATH-500, numeric items", "tag": "Math"}}
ARMS = ("forced_direct", "requested_direct", "chain_of_thought", "native_reasoning")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def arm(task):
    return f"thinking_{task}"


# ============================================================================ inputs


def thinking_panel(task, panel, dry_run):
    """The task's thinking checkpoints in panel order (the build record; in a dry run the pilot rule on the registry)."""
    if dry_run and not os.path.exists(RECORD):
        from build_thinking_requests import thinking_models
        keys = set(thinking_models())
    else:
        keys = set(json.load(open(RECORD, encoding="utf-8"))["tasks"][task]["models"])
    return [k for k in panel if k in keys]


def thinking_inputs(task, panel, sampler, items_info, col_load, dry_run, partial):
    """Loader of the native-reasoning rows after the checks of the scored table; in a dry run the collection's
    chain-of-thought rows relabelled. With `partial` the panel shrinks to the models scored so far."""
    if dry_run:
        def load(ids):
            df = col_load(ids)
            return df[(df["arm"] == "zero_shot_cot") & df["model_key"].isin(panel)].assign(arm=arm(task))
        return load, {"dry_run": "native-reasoning arm replaced by the collection's chain-of-thought arm"}, None, panel
    from src import d2_io, string_tasks
    path = os.path.join(PROCESSED, f"{task}_scored.parquet")
    md = d2_io.scored_metadata(path)
    kw = {"sampler": sampler, "scorer_path": os.path.join(MANIFESTS, f"{task}_scorer.json")}
    tm = string_tasks.module(task)
    if tm is not None:
        _, items_info = tm.load_items("target")
        kw["items_label"] = tm.TARGET_LABEL
    problems = d2_io.main_problems(md, items_info=items_info, **kw)
    if md.get("d2_task", "gsm8k") != task:
        problems.append(f"d2_task is {md.get('d2_task')!r}, expected {task!r}")
    models = sorted({i["model_key"] for i in json.loads(md["d2_inputs"])})
    if partial:
        panel = [k for k in panel if k in models]
    elif sorted(panel) != models:
        problems.append(f"models {sorted(set(panel) ^ set(models))} are in only one of the panel and the table")
    if problems:
        raise SystemExit(f"refused: {rel(path)}: " + "; ".join(problems))

    def load(ids):
        df = pd.read_parquet(path, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        if tm is not None:
            df["item_id"] = df["item_id"].astype(str)
        if set(df["arm"].unique()) - {arm(task)}:
            raise SystemExit(f"refused: {rel(path)} holds arms other than {arm(task)}")
        return df[df["model_key"].isin(panel)]
    closure = pd.read_parquet(os.path.join(PROCESSED, f"{task}_closure.parquet"))
    prov = {"path": rel(path), "sha256": d2_io.sha256_path(path), "scorer_version": md.get("d2_scorer_version"),
            "gpu_types": {i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])}}
    return load, prov, closure, panel


def four_arm_arrays(col_load, fd_load, th_load, units, panel, items, value="strict_correct"):
    """(forced direct, requested direct, chain of thought, native reasoning), each (K, M) pp."""
    col = col_load(units)
    col = col[col["model_key"].isin(panel)]
    values = tuple(dict.fromkeys(tuple(d2.VALUES) + (value,)))

    def pair(base, interv):
        cells = d2.cell_table(pd.concat([base.assign(arm="direct_answer"), interv.assign(arm="zero_shot_cot")],
                                        ignore_index=True), panel, units, items, values=values)
        return d2.unit_arrays(cells, units, panel, value)
    fd = fd_load(units)
    fd = fd[fd["model_key"].isin(panel)]
    Breq, Pc = pair(col[col["arm"] == "direct_answer"], col[col["arm"] == "zero_shot_cot"])
    Bfd, Pth = pair(fd, th_load(units))
    return Bfd, Breq, Pc, Pth


def combined_loader(col_load, fd_load, th_load, panel):
    """Rows of the Steps 1-3 analysis: forced direct as the baseline (direct_answer), native reasoning as the
    intervention (zero_shot_cot)."""
    def load(ids):
        fd = fd_load(ids)
        return pd.concat([fd[fd["model_key"].isin(panel)].assign(arm="direct_answer"),
                          th_load(ids).assign(arm="zero_shot_cot")], ignore_index=True)
    return load


def gpu_check(col_prov, th_prov):
    from src import d2_io
    col = {}
    for t in col_prov.get("tables", []):
        md = d2_io.scored_metadata(os.path.join(ROOT, t["path"]))
        col.update({i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])})
    th = th_prov.get("gpu_types") or {}
    return {"mismatched": sorted(k for k in th if col.get(k) != th[k]),
            "by_model": {k: {"collection": col.get(k), "thinking": th[k]} for k in th}}


# ============================================================================ blocks


def interventions_block(Bfd, Pc, Pth):
    """The two benefits over forced direct: cross-draw moments, latent correlation, model-mean agreement."""
    X, Y = Pth - Bfd, Pc - Bfd

    def pt(X, Y):
        vx, vy, c = cross_moment(X, X), cross_moment(Y, Y), cross_moment(X, Y)
        return {"V_native": vx, "V_cot": vy, "cross_moment": c,
                "latent_corr": c / np.sqrt(vx * vy) if vx > 0 and vy > 0 else None}
    est = pt(X, Y)
    loo = [pt(np.delete(X, u, 0), np.delete(Y, u, 0)) for u in range(X.shape[0])]
    out = {}
    for k, v in est.items():
        vals = [r[k] for r in loo]
        out[k] = ({"estimate": v, "ci95": None, "note": "not defined"} if v is None or any(x is None for x in vals)
                  else jack(v, np.asarray(vals, float)))
    bx, by = X.mean(0), Y.mean(0)
    M = len(bx)
    iu = np.triu_indices(M, 1)
    sx = tie_sign(bx[:, None] - bx[None, :])[iu]
    sy = tie_sign(by[:, None] - by[None, :])[iu]
    out["model_means"] = {
        "pearson": float(np.corrcoef(bx, by)[0, 1]) if M > 2 else None,
        "spearman": float(pd.Series(bx).rank().corr(pd.Series(by).rank())) if M > 2 else None,
        "pairs": int(len(sx)), "pairs_same_order": int(np.sum((sx == sy) & (sx != 0))),
        "pairs_tied": int(np.sum((sx == 0) | (sy == 0))),
        "share_same_order": float(np.mean(sx == sy)) if len(sx) else None,
        "note": "discovery draws; a pair counts as ordered alike when both signs agree (ties included)"}
    # Pair orderings by the two benefits on the same draw, averaged over draws (a tie counts one half), beside each
    # benefit's own agreement between distinct draws (scripts/run_specification.py, c_between and c_within).
    def agree(same_tie):
        same, tie = same_tie
        return float(same[iu].mean() + 0.5 * tie[iu].mean())
    out["pair_agreement"] = {
        "C_between": float(np.mean([agree(ins.pair_matrices_between(X[r], Y[r])) for r in range(X.shape[0])])),
        "C_within_native": agree(ins.pair_matrices_draws(X)), "C_within_cot": agree(ins.pair_matrices_draws(Y)),
        "note": "discovery draws; C_between compares the native and the chain-of-thought benefit orderings draw by "
                "draw; C_within compares one benefit's orderings between distinct draws"}
    return out


def cot_cap():
    """The chain-of-thought arm's output cap in the collections of the thinking checkpoints (config/decoding.yaml;
    none of them is a native-4k checkpoint with the MATH 2,900-token cap)."""
    return int(yaml.safe_load(open(os.path.join(ROOT, "config", "decoding.yaml"), encoding="utf-8"))
               ["max_new_tokens"]["zero_shot_cot"])


def capped(th_load, cap):
    """Native-reasoning rows scored at the matched cap: an output longer than `cap` tokens counts as wrong."""
    def load(ids):
        df = th_load(ids)
        return df.assign(strict_correct=df["strict_correct"].astype(bool) & (df["n_output_tokens"] <= cap))
    return load


def per_model(panel, Bfd, Breq, Pc, Pth, th_df, closure, cot_df=None, cap=None, Pth_cap=None):
    rows = []
    for j, k in enumerate(panel):
        r = {"model_key": k, "forced_direct": Bfd[:, j].mean(), "requested_direct": Breq[:, j].mean(),
             "chain_of_thought": Pc[:, j].mean(), "native_reasoning": Pth[:, j].mean()}
        r["cot_benefit"] = r["chain_of_thought"] - r["forced_direct"]
        r["native_benefit"] = r["native_reasoning"] - r["forced_direct"]
        r["native_minus_cot"] = r["native_reasoning"] - r["chain_of_thought"]
        g = th_df[th_df["model_key"] == k]
        n = g["n_output_tokens"].astype(float)
        r.update({"native_tokens_mean": float(n.mean()), "native_tokens_median": float(n.median()),
                  "native_tokens_p99": float(np.percentile(n, 99)),
                  "native_cut_pct": 100 * float((g["finish_reason"] == "length").mean()),
                  "native_compliant_pct": 100 * float(g["compliant"].mean())})
        if cot_df is not None:          # token use and truncation of chain of thought, and the matched cap
            c = cot_df[cot_df["model_key"] == k]
            nc = c["n_output_tokens"].astype(float)
            r.update({"cot_tokens_mean": float(nc.mean()), "cot_tokens_median": float(nc.median()),
                      "cot_cut_pct": 100 * float((c["finish_reason"] == "length").mean()),
                      "native_over_cot_cap_pct": 100 * float((n > cap).mean()),
                      "native_reasoning_at_cot_cap": float(Pth_cap[:, j].mean())})
            r["native_at_cap_minus_cot"] = r["native_reasoning_at_cot_cap"] - r["chain_of_thought"]
        if closure is not None:
            c = closure[closure["model_key"] == k]
            r["native_closed_pct"] = 100 * float(c["closed"].mean())
        rows.append(r)
    return rows


def late_shares(path=os.path.join(ROOT, "results", "late_answers", "scatter.csv")):
    """Per model, the share (%) of strictly correct requested-direct answers whose boxed answer ends after output token
    64, over the 24 discovery draws of GSM8K-19 (results/late_answers); empty when that analysis has not run."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {str(k): 100 * float(v) for k, v in zip(df["model_key"], df["share_late"])}


def map_points(task, Bd, Pd, Bv, Pv, frozen, cfg, n_splits, M, n_items):
    obs = rm.validation_agreement(Bv, Pv, cfg["group_sizes"], n_splits,
                                  sampling.rng_for("thinking_map", "splits", task, seed=cfg["seed"]))
    point = rm.setting_point(Bd, Pd)
    rows = []
    for q in ("benefit", "accuracy"):
        p = point[q]
        err = {}
        for n in cfg["pilot_sizes"]:
            pr = frozen["settings"][task][str(n)][q]
            err[n] = pr["C_one_draw"] - obs["one_draw"][q]["C"] if pr["defined"] else None
        rows.append({"setting": f"thinking:{task}", "quantity": q, "M": M, "n_items": n_items, "R": p["R"],
                     "C": p["C"], "C_ref": p.get("C_ref"), "departure": p.get("departure"), "tie": p.get("tie"),
                     "pilot_error_2": err.get(2), "pilot_error_4": err.get(4),
                     "validation_C_one_draw": obs["one_draw"][q]["C"]})
    return rows, point, obs


# ============================================================================ run


def run(a):
    t0 = time.time()
    if (a.dry_run or a.partial) and os.path.abspath(a.pred_dir) == os.path.abspath(MANIFESTS):
        raise SystemExit("a dry run or a partial run writes its frozen predictions under --out, never to manifests/")
    import run_dataset2
    from run_instability import scale_block
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    icfg = yaml.safe_load(open(INST_CONFIG, encoding="utf-8"))
    sel = d2.load_selection(SELECTION)
    method, alpha = sel["H"], float(icfg["alpha"])
    n_splits = 200 if a.quick else int(mcfg["validation_splits"])
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)

    S = {}
    for task in a.tasks:
        inp, s, items, items_info = collection_inputs(task)
        full = list(inp["panel"])
        fd_load, fd_prov, _ = forced_inputs(task, full, s, items_info, inp["load"], False)
        panel = thinking_panel(task, full, a.dry_run)
        th_load, th_prov, closure, panel = thinking_inputs(task, panel, s, items_info, inp["load"], a.dry_run, a.partial)
        keep = [full.index(k) for k in panel]
        families = [inp["families"][i] for i in keep]
        tiers = {k: v for k, v in inp["tiers"].items() if k in panel} if inp["tiers"] else inp["tiers"]
        disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
        D = four_arm_arrays(inp["load"], fd_load, th_load, disc, panel, items)
        S[task] = {"inp": inp, "sampler": s, "items": items, "panel": panel, "families": families, "tiers": tiers,
                   "fd_load": fd_load, "fd_prov": fd_prov, "th_load": th_load, "th_prov": th_prov,
                   "closure": closure, "disc": disc, "val": val, "D": D}
        print(f"{task}: {len(panel)} models, discovery arrays ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ map pilot predictions, discovery only
    preds = {task: {str(n): rm.pilot_prediction(x["D"][0][:n], x["D"][3][:n], mcfg["group_sizes"],
                                                mcfg["agreement_targets"]) for n in mcfg["pilot_sizes"]}
             for task, x in S.items()}
    from src.sampling import sha256_json
    fr = {"label": "thinking_map", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
          "models": {t: x["panel"] for t, x in S.items()},
          "samplers": {t: x["sampler"]["sha256"] for t, x in S.items()},
          "note": "pilot predictions of the native-reasoning map points (forced direct as the baseline; R from the first "
                  "2 and 4 discovery draws), written before the validation draws are built", "settings": jsonable(preds)}
    fr["sha256"] = sha256_json(fr["settings"])
    frozen_map = d2.write_frozen(fr, os.path.join(a.pred_dir, "predictions_thinking_map.json"),
                                 overwrite=a.overwrite_frozen)

    # ------------------------------------------------------------ Steps 1-3 (freeze inside, then validation)
    steps = {}
    for task, x in S.items():
        label = f"thinking_{task}"
        prov = {"collection": x["inp"]["provenance"], "forced": x["fd_prov"], "thinking": x["th_prov"],
                "baseline_arm": "forced_direct (relabelled direct_answer inside the analysis)",
                "intervention_arm": f"{arm(task)} (relabelled zero_shot_cot inside the analysis)"}
        steps[task] = run_dataset2.run(combined_loader(x["inp"]["load"], x["fd_load"], x["th_load"], x["panel"]),
                                       x["sampler"], x["panel"], x["families"], x["tiers"], x["items"], a.out,
                                       os.path.join(a.pred_dir, f"predictions_{label}.json"), label, n_boot, n_ctrl,
                                       overwrite_frozen=a.overwrite_frozen, provenance=prov,
                                       tables_dir=os.path.join(a.out, f"tables_{label}"))

    # ------------------------------------------------------------ validation draws and the rest
    R = {"analysis": "native reasoning beside prompted chain of thought, forced direct as the baseline "
                     "", "dry_run": bool(a.dry_run), "partial": bool(a.partial),
         "created": time.strftime("%Y-%m-%d %H:%M"), "units": "percentage points; pp^2",
         "frozen_map": {"path": rel(os.path.join(a.pred_dir, "predictions_thinking_map.json")),
                        "sha256": frozen_map["sha256"], "created": frozen_map["created"]},
         "config": {"reliability_map": rel(MAP_CONFIG), "instability": rel(INST_CONFIG)},
         "build": None, "tasks": {}, "map_points": []}
    if os.path.exists(RECORD):
        rec = json.load(open(RECORD, encoding="utf-8"))
        R["build"] = {t: {"pilot": v.get("pilot"), "decoding": v.get("decoding"), "registry": v.get("registry"),
                          "excluded": sorted(v.get("excluded") or {})}
                      for t, v in rec.get("tasks", {}).items()}
    # the pilot rule's exclusions: each checkpoint's share of pilot outputs at the cap
    if os.path.exists(EXCLUSIONS):
        ex = json.load(open(EXCLUSIONS, encoding="utf-8"))
        R["pilot_exclusion"] = {"rule": ex["rule"], "excluded": ex["excluded"], "pending": ex["pending"],
                                "share": {k: v["share"] for k, v in ex["models"].items()}}
    for task, x in S.items():
        panel, items = x["panel"], x["items"]
        n = len(items)
        Bfd, Breq, Pc, Pth = x["D"]
        Vfd, Vreq, Vpc, Vth = four_arm_arrays(x["inp"]["load"], x["fd_load"], x["th_load"], x["val"], panel, items)
        kd = Bfd.shape[0]
        T = {"title": TASKS[task]["title"], "models": panel, "M": len(panel), "n_items": n, "K_discovery": kd,
             "K_validation": Vfd.shape[0], "steps_file": rel(os.path.join(a.out, f"thinking_{task}.json"))}
        m = {q: iv.multiplicities(sampling.rng_for(f"thinking_{task}_{q}", "bootstrap"), kd, n_boot)
             for q in ("cot", "native_minus_cot")}
        T["step2"] = {"native": steps[task]["step2"], "cot": d2.step2(Bfd, Pc, sel, m["cot"]),
                      "native_minus_cot": d2.step2(Pc, Pth, sel, m["native_minus_cot"])}
        T["interventions"] = interventions_block(Bfd, Pc, Pth)
        th_all = x["th_load"](list(x["disc"]) + list(x["val"]))
        # native reasoning at the chain-of-thought arm's cap, on the discovery draws
        cap = cot_cap()
        Pth_cap = four_arm_arrays(x["inp"]["load"], x["fd_load"], capped(x["th_load"], cap), x["disc"], panel, items)[3]
        col_all = x["inp"]["load"](list(x["disc"]) + list(x["val"]))
        cot_all = col_all[(col_all["arm"] == "zero_shot_cot") & col_all["model_key"].isin(panel)]
        T["per_model"] = per_model(panel, Bfd, Breq, Pc, Pth, th_all, x["closure"], cot_all, cap, Pth_cap)
        # beside native reasoning, the same checkpoint's
        # late-answer share in the requested arm with thinking off (results/late_answers); token positions exist for
        # the GSM8K-19 models only, so other models and MATH-500 get none
        if task == "gsm8k":
            late = late_shares()
            for r in T["per_model"]:
                r["requested_late_pct"] = late.get(r["model_key"])
        T["means"] = {k: float(np.mean([r[k] for r in T["per_model"]])) for k in
                      ARMS + ("cot_benefit", "native_benefit", "native_minus_cot", "native_tokens_mean", "native_cut_pct",
                              "native_tokens_median", "cot_tokens_mean", "cot_tokens_median", "cot_cut_pct",
                              "native_over_cot_cap_pct", "native_reasoning_at_cot_cap", "native_at_cap_minus_cot")}
        mc = iv.multiplicities(sampling.rng_for(f"thinking_{task}_native_cap", "bootstrap"), kd, n_boot)
        T["matched_cap"] = {"cap": cap, "rule": "a native-reasoning output longer than the chain-of-thought arm's cap "
                                                "counts as wrong; discovery draws",
                            "step2": d2.step2(Bfd, Pth_cap, sel, mc),
                            "slope_points": {k: v for k, v in ins.scale_moments(Bfd, Pth_cap - Bfd, method, alpha).items()
                                             if k != "model_means"}}
        T["scale"] = {"native": scale_block(Bfd, Pth, Vfd, Vth, n, method, alpha),
                      "cot": scale_block(Bfd, Pc, Vfd, Vpc, n, method, alpha)}
        pts, point, obs = map_points(task, Bfd, Pth, Vfd, Vth, frozen_map, mcfg, n_splits, len(panel), n)
        T["map"] = {"point": point, "validation_one_draw": obs["one_draw"]}
        R["map_points"] += pts
        if not a.dry_run:
            T["gpu_check"] = gpu_check(x["inp"]["provenance"], x["th_prov"])
        R["tasks"][task] = T
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    for task, T in R["tasks"].items():
        pd.DataFrame(T["per_model"]).to_csv(os.path.join(a.out, "tables", f"per_model_{task}.csv"), index=False)
    dump(R, os.path.join(a.out, "thinking.json"))
    write_summary(R, a.out)
    print(f"done in {R['runtime_seconds']} s -> {rel(a.out)}")
    return R


# ============================================================================ macros and summary


def kfmt(k):
    """Draws for an agreement target as the map prints them (scripts/run_reliability_map.py)."""
    if k is None:
        return "undefined"
    return r"$\infty$" if np.isinf(k) else f"{int(k):,}".replace(",", "{,}")


def macros(R):
    m = {}

    def put(k, v):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
        m[k] = str(v).replace("−", "$-$")
    for task, T in R["tasks"].items():
        t = "thi" + TASKS[task]["tag"]
        put(t + "M", T["M"])
        put(t + "KDisc", T["K_discovery"])
        put(t + "KVal", T["K_validation"])
        for k, s in (("forced_direct", "MeanForced"), ("requested_direct", "MeanReq"), ("chain_of_thought", "MeanCot"),
                     ("native_reasoning", "MeanNative"), ("cot_benefit", "BenCot"), ("native_benefit", "BenNative"),
                     ("native_minus_cot", "NativeMinusCot"), ("native_cut_pct", "CutPct")):
            put(t + s, fmt(T["means"][k], 1))
        put(t + "Tokens", f"{T['means']['native_tokens_mean']:,.0f}")
        for q, s in (("native", "Native"), ("cot", "Cot"), ("native_minus_cot", "NmC")):
            st = T["step2"][q]
            put(t + "Vdelta" + s, fmt(st["V_delta"]["estimate"], 1))
            if st["V_delta"].get("ci95"):
                put(t + "Vdelta" + s + "Lo", fmt(st["V_delta"]["ci95"][0], 1))
                put(t + "Vdelta" + s + "Hi", fmt(st["V_delta"]["ci95"][1], 1))
            put(t + "H" + s, fmt(st["H"]["estimate"], 2))
            put(t + "HSet" + s, hset(st["H"]))
            put(t + "R" + s, fmt(st["reliability"]["R_benefit"], 2))
            put(t + "Wgain" + s, fmt(st["noise"]["W_gain"], 2))
        put(t + "CutPctMax", fmt(max(r["native_cut_pct"] for r in T["per_model"]), 1))
        iv_ = T["interventions"]
        pa = iv_["pair_agreement"]
        put(t + "CBetween", fmt(pa["C_between"], 3))
        put(t + "CWithinNative", fmt(pa["C_within_native"], 3))
        put(t + "CWithinCot", fmt(pa["C_within_cot"], 3))
        if iv_["latent_corr"]["estimate"] is not None:
            put(t + "LatentCorr", fmt(iv_["latent_corr"]["estimate"], 2))
        mm = iv_["model_means"]
        if mm["pearson"] is not None:
            put(t + "Pearson", fmt(mm["pearson"], 2))
            put(t + "Spearman", fmt(mm["spearman"], 2))
        put(t + "PairsSame", mm["pairs_same_order"])
        put(t + "Pairs", mm["pairs"])
        for q, s in (("native", "Native"), ("cot", "Cot")):
            for sc, ss in (("points", "Points"), ("logit", "Logit"), ("error_reduction", "ErrRed")):
                d = T["scale"][q][sc]["discovery"]
                put(t + "Slope" + s + ss, fmt(d["slope"], 3))
                bounded = d["slope_set"]["kind"] == "bounded"
                put(t + "Slope" + s + ss + "Lo", fmt(d["slope_set"]["a"], 3) if bounded else d["slope_set"]["kind"])
                put(t + "Slope" + s + ss + "Hi", fmt(d["slope_set"]["b"], 3) if bounded else d["slope_set"]["kind"])
        mp = T["map"]["point"]
        put(t + "MapRBen", fmt(mp["benefit"]["R"], 2))
        put(t + "MapCBen", fmt(mp["benefit"]["C"], 3))
        for tg, ts in ((0.8, "Eighty"), (0.9, "Ninety")):
            put(t + "MapK" + ts + "Ben", kfmt(rm.k_needed(mp["benefit"]["R"], tg)))
        mn = T["means"]
        if "cot_tokens_mean" in mn:         # token use, truncation and the matched cap
            put(t + "TokensMedian", f"{mn['native_tokens_median']:,.0f}")
            put(t + "TokensCot", f"{mn['cot_tokens_mean']:,.0f}")
            put(t + "TokensCotMedian", f"{mn['cot_tokens_median']:,.0f}")
            put(t + "CutPctCot", fmt(mn["cot_cut_pct"], 2))
            put(t + "OverCapPct", fmt(mn["native_over_cot_cap_pct"], 1))
            put(t + "OverCapPctMax", fmt(max(r["native_over_cot_cap_pct"] for r in T["per_model"]), 1))
            put(t + "MeanNativeCap", fmt(mn["native_reasoning_at_cot_cap"], 1))
            put(t + "NativeCapMinusCot", fmt(mn["native_at_cap_minus_cot"], 1))
            mc = T["matched_cap"]
            put(t + "CotCap", f"{mc['cap']:,}")
            st = mc["step2"]
            put(t + "RNativeCap", fmt(st["reliability"]["R_benefit"], 2))
            put(t + "WgainNativeCap", fmt(st["noise"]["W_gain"], 2))
            put(t + "VdeltaNativeCap", fmt(st["V_delta"]["estimate"], 1))
            if st["V_delta"].get("ci95"):
                put(t + "VdeltaNativeCapLo", fmt(st["V_delta"]["ci95"][0], 1))
                put(t + "VdeltaNativeCapHi", fmt(st["V_delta"]["ci95"][1], 1))
            d = mc["slope_points"]
            put(t + "SlopeNativeCapPoints", fmt(d["slope"], 3))
            bounded = d["slope_set"]["kind"] == "bounded"
            put(t + "SlopeNativeCapPointsLo", fmt(d["slope_set"]["a"], 3) if bounded else d["slope_set"]["kind"])
            put(t + "SlopeNativeCapPointsHi", fmt(d["slope_set"]["b"], 3) if bounded else d["slope_set"]["kind"])
        if T.get("gpu_check") is not None:
            put(t + "GpuSame", T["M"] - len(T["gpu_check"]["mismatched"]))
    return m


def write_summary(R, out_dir):
    L = ["# Native reasoning beside prompted chain of thought", "",
         "The thinking checkpoints of each panel ran the collection's requested-direct prompts "
         "with the thinking switch on (greedy, special tokens kept, the cap set from the pilot). The answer is read "
         "after the last end-of-reasoning marker; an output cut at the cap before it counts as wrong. Forced direct is "
         "the baseline of both interventions: prompted chain of thought (switch off, the collection's) and native "
         "reasoning. Scores in percentage points, variances in pp². Steps 1–3 use native reasoning against forced "
         "direct with the methods and margins of the main analysis; the rules were saved to "
         "`manifests/predictions_thinking_<task>.json` before the validation draws were built. H = −10 × the slope of "
         "benefit on baseline accuracy: positive means weaker models gain more.", ""]
    if R["dry_run"]:
        L += ["**Dry run: the native-reasoning arm is the chain-of-thought arm relabelled. No number here is a "
              "result.**", ""]
    if R["partial"]:
        L += ["**Partial run on the models scored so far. No number here is a result.**", ""]
    px = R.get("pilot_exclusion")
    if px:
        L += [f"Pilot rule: a checkpoint is {px['rule']}. Excluded: "
              + (", ".join(f"{k} ({100 * px['share'][k]:.1f}% of pilot outputs at the cap)" for k in px["excluded"])
                 or "none") + ". The others' shares: "
              + ", ".join(f"{k} {100 * s:.1f}%" for k, s in sorted(px["share"].items()) if k not in px["excluded"])
              + "." + (f" Pilots still out: {', '.join(px['pending'])}." if px["pending"] else ""), ""]
    for task, T in R["tasks"].items():
        mn, iv_ = T["means"], T["interventions"]
        b = (R.get("build") or {}).get(task) or {}
        L += [f"## {T['title']} ({T['M']} models)", "",
              f"{T['n_items']:,} items, {T['K_discovery']} discovery and {T['K_validation']} validation draws"
              + (f"; cap {b['pilot']['cap']:,} new tokens, max_model_len {b['pilot']['max_model_len']:,}" if b.get("pilot")
                 else "") + ".", "",
              f"Mean accuracy over models: forced direct {fmt(mn['forced_direct'], 1)}, requested direct "
              f"{fmt(mn['requested_direct'], 1)}, chain of thought {fmt(mn['chain_of_thought'], 1)}, native reasoning "
              f"{fmt(mn['native_reasoning'], 1)}. Native reasoning used {mn['native_tokens_mean']:,.0f} output tokens on "
              f"average, and {fmt(mn['native_cut_pct'], 2)}% of its outputs reached the cap.", ""]
        if "matched_cap" in T:
            mc, st_ = T["matched_cap"], T["matched_cap"]["step2"]
            sp = mc["slope_points"]
            L += [f"Token use: native reasoning median {mn['native_tokens_median']:,.0f} output tokens, chain of "
                  f"thought mean {mn['cot_tokens_mean']:,.0f} and median {mn['cot_tokens_median']:,.0f}; "
                  f"{fmt(mn['cot_cut_pct'], 2)}% of chain-of-thought outputs reached its cap. At the chain-of-thought "
                  f"cap of {mc['cap']:,} tokens ({fmt(mn['native_over_cot_cap_pct'], 1)}% of native outputs are longer "
                  f"and count as wrong), native reasoning scores {fmt(mn['native_reasoning_at_cot_cap'], 1)}, "
                  f"{fmt(mn['native_at_cap_minus_cot'], 1)} against chain of thought; native − forced direct there has "
                  f"R {fmt(st_['reliability']['R_benefit'])}, V_delta {fmt(st_['V_delta']['estimate'], 1)} "
                  f"{ci(st_['V_delta'], 1)} pp² and a points slope on forced-direct accuracy of {fmt(sp['slope'], 3)} "
                  + (f"[{fmt(sp['slope_set']['a'], 3)}, {fmt(sp['slope_set']['b'], 3)}]."
                     if sp["slope_set"]["kind"] == "bounded" else f"(set {sp['slope_set']['kind']})."), ""]
        L += ["| Benefit | Mean | V_delta (pp²) | H | H set |", "|---|---:|---:|---:|---|"]
        for q, lab, key in (("cot", "chain of thought − forced direct", "cot_benefit"),
                            ("native", "native reasoning − forced direct", "native_benefit"),
                            ("native_minus_cot", "native reasoning − chain of thought", "native_minus_cot")):
            st = T["step2"][q]
            L.append(f"| {lab} | {fmt(mn[key], 2)} | {fmt(st['V_delta']['estimate'], 1)} {ci(st['V_delta'], 1)} | "
                     f"{fmt(st['H']['estimate'])} | {hset(st['H'])} |")
        mm = iv_["model_means"]
        L += ["", f"The two benefits over forced direct: latent correlation {fmt(iv_['latent_corr']['estimate'])} "
                  f"{ci(iv_['latent_corr'])} (cross-draw moments, jackknife, descriptive); over the models' mean "
                  f"benefits Pearson {fmt(mm['pearson'])} and Spearman {fmt(mm['spearman'])}; "
                  f"{mm['pairs_same_order']} of {mm['pairs']} model pairs ordered alike by both.", "",
              "Slope of benefit on forced-direct accuracy (discovery draws):", "",
              "| Scale | Chain of thought | Native reasoning |", "|---|---:|---:|"]
        for sc in ins.SCALES:
            L.append(f"| {sc.replace('_', ' ')} | {fmt(T['scale']['cot'][sc]['discovery']['slope'], 3)} | "
                     f"{fmt(T['scale']['native'][sc]['discovery']['slope'], 3)} |")
        mp = T["map"]["point"]
        L += ["", f"Map point (native reasoning against forced direct): R benefit {fmt(mp['benefit']['R'])}, C benefit "
                  f"{fmt(mp['benefit']['C'], 3)} against reference {fmt(mp['benefit']['C_ref'], 3)}.", ""]
        if T.get("gpu_check"):
            mm_ = T["gpu_check"]["mismatched"]
            L += [f"GPU type as in the collection: {T['M'] - len(mm_)} of {T['M']} models"
                  + (f" (differs: {', '.join(mm_)})." if mm_ else "."), ""]
        L += ["| Model | Forced | Requested | Chain of thought | Native | Native − forced | Chain of thought − forced | "
              "Tokens (mean) | Cut (%) | Closed (%) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(T["per_model"], key=lambda r: r["forced_direct"]):
            L.append(f"| {r['model_key']} | {fmt(r['forced_direct'], 1)} | {fmt(r['requested_direct'], 1)} | "
                     f"{fmt(r['chain_of_thought'], 1)} | {fmt(r['native_reasoning'], 1)} | {fmt(r['native_benefit'], 1)} | "
                     f"{fmt(r['cot_benefit'], 1)} | {r['native_tokens_mean']:,.0f} | {fmt(r['native_cut_pct'], 2)} | "
                     f"{fmt(r.get('native_closed_pct'), 1)} |")
        late = [r for r in sorted(T["per_model"], key=lambda r: r["forced_direct"]) if r.get("requested_late_pct") is not None]
        if late:
            L += ["", "Late direct answers of the same checkpoints in the requested arm with thinking off, as the share "
                      "of correct answers whose boxed answer ends after output token 64 (`results/late_answers`, GSM8K-19 "
                      "models only): "
                  + "; ".join(f"{r['model_key']} {fmt(r['requested_late_pct'], 1)}%" for r in late) + "."]
        L.append("")
    txt = "\n".join(L)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=None, help="default manifests/ (a dry or partial run: <out>/frozen)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partial", action="store_true", help="the models scored so far (a diagnostic)")
    ap.add_argument("--quick", action="store_true", help="999 bootstrap draws, 100 control replicates, 200 splits")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    if (a.dry_run or a.partial) and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("a dry or partial run needs --out outside results/thinking")
    a.pred_dir = a.pred_dir or (os.path.join(a.out, "frozen") if (a.dry_run or a.partial) else MANIFESTS)
    run(a)


if __name__ == "__main__":
    main()
