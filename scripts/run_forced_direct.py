r"""The forced-direct arm: who benefits from chain of thought when the
direct answer is forced to be direct?

PowerShell (from the repository root), after scripts\score_forced_direct.py --task gsm8k and --task bbh:
    cmd /c ".venv\Scripts\python.exe scripts\run_forced_direct.py 2>&1"
    cmd /c ".venv\Scripts\python.exe scripts\run_map_panels.py --forced 2>&1"        # adds the map points
Dry run of the whole pipeline before the data exist (the forced arm replaced by the requested-direct arm; written
only under --out, never to manifests/):
    cmd /c ".venv\Scripts\python.exe scripts\run_forced_direct.py --dry-run --quick --out <scratch dir> 2>&1"
MATH-500, after scripts\run_dataset2.py --task math and
scripts\score_forced_direct.py --task math; written to results/forced_direct/math/, with the frozen map file
manifests/predictions_forced_map_math.json and the rules manifests/predictions_forced_math*.json:
    cmd /c ".venv\Scripts\python.exe scripts\run_forced_direct.py --tasks math 2>&1"

Settings: GSM8K-29 and BIG-Bench Hard-28 (the 25 tasks pooled with equal weights), the collection's panels, items and
frozen samplers. Three arms per model: requested-direct (the collection's direct-answer arm), forced-direct (the
assistant turn prefilled with \boxed{, 16 new tokens) and chain of thought (the collection's).

Order:
1. discovery draws of the three arms;
2. the map's pilot predictions (R from the first 2 and 4 draws, config/reliability_map.yaml) saved to
   manifests/predictions_forced_map.json;
3. Steps 1-3 by scripts/run_dataset2.py's run() with forced-direct as the baseline arm, the same methods and margins:
   its prediction rules are saved to manifests/predictions_forced_<task>.json (and _lenient) before it builds the
   validation draws; its output is results/forced_direct/forced_<task>.json;
4. validation draws; then H on three scales, subpanel signs and the shape (scripts/run_instability.py's blocks,
   config/instability.yaml), the three-arm quantities and the map points.

Three-arm quantities on the discovery draws: adherence cost = requested-direct - forced-direct and reasoning benefit =
chain of thought - forced-direct, so the collection's gain = reasoning benefit - adherence cost. Each gets
src/dataset2.step2 with forced-direct as the baseline (V_delta and H with the calibrated methods), and the
cross-draw moment of the two parts (Appendix A, distinct draws) with a leave-one-draw-out jackknife (descriptive).
The collection's own step2 is recomputed and must equal results/dataset2/<gsm8k30|bbh>.json.

Outputs: results/forced_direct/forced_direct.json (with "macros", prefix fd, and "map_points"), forced_gsm8k.json,
forced_bbh.json, summary.md, summary_clean.md and tables/.
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
from src.estimators import cross_moment  # noqa: E402

OUT = os.path.join(ROOT, "results", "forced_direct")
MANIFESTS = os.path.join(ROOT, "manifests")
PROCESSED = os.path.join(ROOT, "data", "processed", "forced")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
MAP_CONFIG = os.path.join(ROOT, "config", "reliability_map.yaml")
INST_CONFIG = os.path.join(ROOT, "config", "instability.yaml")
ARM = "forced_direct"
TASKS = {"gsm8k": {"title": "GSM8K (29 models)", "collection": "gsm8k30", "tag": "Gsm"},
         "bbh": {"title": "BIG-Bench Hard (28 models, 25 tasks pooled)", "collection": "bbh", "tag": "Bbh"},
         "math": {"title": "MATH-500, numeric items (GSM8K-29 panel)", "collection": "math", "tag": "Math"}}
DEFAULT_TASKS = ("gsm8k", "bbh")
DECODING_NOTE = {"math": "config/decoding_forced_math.yaml: no stop string, since MATH answers may hold braces"}
QUANT = {"adherence": "adherence cost (requested-direct - forced-direct)",
         "reasoning": "reasoning benefit (chain of thought - forced-direct)",
         "collection": "collection gain (chain of thought - requested-direct)"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=1)


# ============================================================================ inputs


def added_task(task) -> bool:
    """True for a task of the section `tasks` of config/added_contrasts.yaml (HotpotQA)."""
    if task in ("gsm8k", "bbh", "math"):
        return False
    import added_contrasts as ac
    return task in ac.load_tasks()


def collection_inputs(task):
    from src import d2_io, d2_panels
    if added_task(task):                          # an added task: its scored table, by configuration
        import added_contrasts as ac
        return ac.task_collection_inputs(task)
    if task == "gsm8k":
        s = json.load(open(os.path.join(MANIFESTS, "sampler_gsm8k.json"), encoding="utf-8"))
        items_df, items_info = d2_io.load_items(split="test")
        inp = d2_panels.inputs(None, None, s, items_info, os.path.join(MANIFESTS, "gsm8k_scorer.json"))
        items = items_df["item_id"].tolist()
    elif task == "math":
        s = json.load(open(os.path.join(MANIFESTS, "sampler_math.json"), encoding="utf-8"))
        inp = d2_panels.math_inputs(None, s)
        items_info = None
        items = inp["items"]
    else:
        s = json.load(open(os.path.join(MANIFESTS, "sampler_bbh.json"), encoding="utf-8"))
        inp = d2_panels.bbh_inputs(None, s)
        items_info = None
        items = inp["items"]
    return inp, s, items, items_info


def forced_inputs(task, panel, sampler, items_info, col_load, dry_run):
    """Loader of the forced-direct rows (arm forced_direct), after the checks of the scored table; in a dry run the
    requested-direct rows relabelled."""
    if dry_run:
        def load(ids):
            df = col_load(ids)
            return df[df["arm"] == "direct_answer"].assign(arm=ARM)
        box = None
        return load, {"dry_run": "forced arm replaced by the requested-direct arm"}, box
    if added_task(task):                          # an added task: the forced role of its scored table
        import added_contrasts as ac
        return ac.task_forced_inputs(task, panel)
    from src import d2_io, string_tasks
    path = os.path.join(PROCESSED, f"{task}_scored.parquet")
    md = d2_io.scored_metadata(path)
    kw = {"sampler": sampler, "scorer_path": os.path.join(MANIFESTS, f"{task}_scorer.json")}
    tm = string_tasks.module(task)                 # BIG-Bench Hard and MATH-500: target items
    if tm is not None:
        _, items_info = tm.load_items("target")
        kw["items_label"] = tm.TARGET_LABEL
    problems = d2_io.main_problems(md, items_info=items_info, **kw)
    if (md.get("d2_task", "gsm8k")) != task:
        problems.append(f"d2_task is {md.get('d2_task')!r}, expected {task!r}")
    models = sorted({i["model_key"] for i in json.loads(md["d2_inputs"])})
    if sorted(panel) != models:
        problems.append(f"models {sorted(set(panel) ^ set(models))} are in only one of the panel and the table")
    if problems:
        raise SystemExit(f"refused: {rel(path)}: " + "; ".join(problems))

    def load(ids):
        df = pd.read_parquet(path, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        if tm is not None:
            df["item_id"] = df["item_id"].astype(str)
        if set(df["arm"].unique()) - {ARM}:
            raise SystemExit(f"refused: {rel(path)} holds arms other than {ARM}")
        return df[df["model_key"].isin(panel)]
    box = pd.read_parquet(os.path.join(PROCESSED, f"{task}_box.parquet"))
    prov = {"path": rel(path), "sha256": d2_io.sha256_path(path), "scorer_version": md.get("d2_scorer_version"),
            "gpu_types": {i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])}}
    return load, prov, box


def three_arm_arrays(col_load, fd_load, units, panel, items, value="strict_correct"):
    """(requested-direct, forced-direct, chain of thought), each (K, M) pp, and the combined table's cells."""
    col = col_load(units)
    fd = fd_load(units)
    values = tuple(dict.fromkeys(tuple(d2.VALUES) + (value,)))
    cells_c = d2.cell_table(col, panel, units, items, values=values)
    Breq, Pc = d2.unit_arrays(cells_c, units, panel, value)
    comb = pd.concat([fd.assign(arm="direct_answer"), col[col["arm"] == "zero_shot_cot"]], ignore_index=True)
    cells_f = d2.cell_table(comb, panel, units, items, values=values)
    Bfd, Pc2 = d2.unit_arrays(cells_f, units, panel, value)
    if not np.array_equal(Pc, Pc2):
        raise SystemExit("the chain-of-thought arm differs between the two tables")
    return Breq, Bfd, Pc


def gpu_check(col_prov, fd_prov):
    from src import d2_io
    col = {}
    for t in col_prov.get("tables", []):
        md = d2_io.scored_metadata(os.path.join(ROOT, t["path"]))
        col.update({i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])})
    fd = fd_prov.get("gpu_types") or {}
    return {"mismatched": sorted(k for k in fd if col.get(k) != fd[k]),
            "by_model": {k: {"collection": col.get(k), "forced": fd[k]} for k in fd}}


# ============================================================================ blocks


def jack(est, loo):
    lo, hi = iv.jack_t(est, loo)
    return {"estimate": float(est), "jackknife_se": iv.jackknife_se(loo), "ci95": [float(lo), float(hi)],
            "method": "jack_t", "note": "leave-one-draw-out jackknife; descriptive (outside the calibration)"}


def parts_block(Breq, Bfd, Pc):
    """Cross-draw moments of adherence cost and reasoning benefit; the identity V_gain = V_r + V_a - 2 C."""
    A, Rb, G = Breq - Bfd, Pc - Bfd, Pc - Breq

    def pt(A, Rb, G):
        va, vr, c, vg = cross_moment(A, A), cross_moment(Rb, Rb), cross_moment(A, Rb), cross_moment(G, G)
        return {"V_adherence": va, "V_reasoning": vr, "cross_moment": c, "V_collection": vg,
                "corr": c / np.sqrt(va * vr) if va > 0 and vr > 0 else None}
    est = pt(A, Rb, G)
    K = A.shape[0]
    loo = [pt(np.delete(A, u, 0), np.delete(Rb, u, 0), np.delete(G, u, 0)) for u in range(K)]
    out = {"identity_residual": est["V_collection"] - (est["V_reasoning"] + est["V_adherence"]
                                                       - 2 * est["cross_moment"])}
    for k, v in est.items():
        vals = [r[k] for r in loo]
        out[k] = ({"estimate": v, "ci95": None, "note": "not defined"} if v is None or any(x is None for x in vals)
                  else jack(v, np.asarray(vals, float)))
    return out


def per_model(panel, Breq, Bfd, Pc, fd_df, box):
    rows = []
    for j, k in enumerate(panel):
        r = {"model_key": k, "requested_direct": Breq[:, j].mean(), "forced_direct": Bfd[:, j].mean(),
             "chain_of_thought": Pc[:, j].mean()}
        r["adherence_cost"] = r["requested_direct"] - r["forced_direct"]
        r["reasoning_benefit"] = r["chain_of_thought"] - r["forced_direct"]
        r["collection_gain"] = r["chain_of_thought"] - r["requested_direct"]
        g = fd_df[fd_df["model_key"] == k]
        r["forced_compliant_pct"] = 100 * float(g["compliant"].mean())
        r["forced_finish_length_pct"] = 100 * float((g["finish_reason"] == "length").mean())
        if box is not None:
            b = box[box["model_key"] == k]
            m = g.merge(b[["config_id", "item_id", "box_closed"]].assign(item_id=b["item_id"].astype(g["item_id"].dtype)),
                        on=["config_id", "item_id"], how="left")
            r["box_closed_pct"] = 100 * float(m["box_closed"].mean())
            r["closed_and_readable_pct"] = 100 * float((m["box_closed"].astype(bool) & m["compliant"].astype(bool)).mean())
            r["fallback_no_prefill"] = bool((b["prefill"] == "").any())
        rows.append(r)
    return rows


def map_points(task, Bd, Pd, Bv, Pv, frozen, cfg, n_splits, M, n_items):
    obs = rm.validation_agreement(Bv, Pv, cfg["group_sizes"], n_splits,
                                  sampling.rng_for("forced_direct_map", "splits", task, seed=cfg["seed"]))
    point = rm.setting_point(Bd, Pd)
    rows = []
    for q in ("benefit", "accuracy"):
        p = point[q]
        err = {}
        for n in cfg["pilot_sizes"]:
            pr = frozen["settings"][task][str(n)][q]
            err[n] = pr["C_one_draw"] - obs["one_draw"][q]["C"] if pr["defined"] else None
        rows.append({"setting": f"forced:{task}", "quantity": q, "M": M, "n_items": n_items, "R": p["R"],
                     "C": p["C"], "C_ref": p.get("C_ref"), "departure": p.get("departure"), "tie": p.get("tie"),
                     "pilot_error_2": err.get(2), "pilot_error_4": err.get(4),
                     "validation_C_one_draw": obs["one_draw"][q]["C"]})
    return rows, point, obs


# ============================================================================ run


def run(a):
    t0 = time.time()
    if a.dry_run and os.path.abspath(a.pred_dir) == os.path.abspath(MANIFESTS):
        raise SystemExit("a dry run writes its frozen predictions under --out, never to manifests/")
    import run_dataset2
    from run_instability import panel_block, prompt_block, scale_block
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    icfg = yaml.safe_load(open(INST_CONFIG, encoding="utf-8"))
    if icfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    sel = d2.load_selection(SELECTION)
    method, alpha = sel["H"], float(icfg["alpha"])
    n_splits = 200 if a.quick else int(mcfg["validation_splits"])
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)
    pilots = None
    pc = os.path.join(OUT, "pilot_check.json")
    if os.path.exists(pc):
        pilots = json.load(open(pc, encoding="utf-8"))

    tasks, suffix = tuple(getattr(a, "tasks", DEFAULT_TASKS)), getattr(a, "suffix", "")
    S = {}
    for task in tasks:
        inp, s, items, items_info = collection_inputs(task)
        panel = list(inp["panel"])
        fd_load, fd_prov, box = forced_inputs(task, panel, s, items_info, inp["load"], a.dry_run)
        disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
        D = three_arm_arrays(inp["load"], fd_load, disc, panel, items)
        S[task] = {"inp": inp, "sampler": s, "items": items, "panel": panel, "fd_load": fd_load, "fd_prov": fd_prov,
                   "box": box, "disc": disc, "val": val, "D": D}
        print(f"{task}: discovery arrays ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ map pilot predictions, discovery only
    preds = {task: {str(n): rm.pilot_prediction(x["D"][1][:n], x["D"][2][:n], mcfg["group_sizes"],
                                                mcfg["agreement_targets"]) for n in mcfg["pilot_sizes"]}
             for task, x in S.items()}
    from src.sampling import sha256_json
    fr = {"label": "forced_map" + suffix, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
          "models": {t: x["panel"] for t, x in S.items()},
          "samplers": {t: x["sampler"]["sha256"] for t, x in S.items()},
          "note": "pilot predictions of the forced-direct map points (R from the first 2 and 4 discovery draws), "
                  "written before the validation draws are built", "settings": jsonable(preds)}
    fr["sha256"] = sha256_json(fr["settings"])
    frozen_map = d2.write_frozen(fr, os.path.join(a.pred_dir, f"predictions_forced_map{suffix}.json"),
                                 overwrite=a.overwrite_frozen)

    # ------------------------------------------------------------ Steps 1-3 (freeze inside, then validation)
    steps = {}
    for task, x in S.items():
        inp, fd_load = x["inp"], x["fd_load"]

        def combined(ids, inp=inp, fd_load=fd_load):
            col = inp["load"](ids)
            return pd.concat([fd_load(ids).assign(arm="direct_answer"), col[col["arm"] == "zero_shot_cot"]],
                             ignore_index=True)
        label = f"forced_{task}"
        prov = {"collection": inp["provenance"], "forced": x["fd_prov"],
                "baseline_arm": "forced_direct (relabelled direct_answer inside the analysis)"}
        steps[task] = run_dataset2.run(combined, x["sampler"], x["panel"], inp["families"], inp["tiers"], x["items"],
                                       a.out, os.path.join(a.pred_dir, f"predictions_{label}.json"), label, n_boot,
                                       n_ctrl, overwrite_frozen=a.overwrite_frozen, provenance=prov,
                                       tables_dir=os.path.join(a.out, f"tables_{label}"))

    # ------------------------------------------------------------ validation draws and the rest
    R = {"analysis": "forced-direct arm", "dry_run": bool(a.dry_run),
         "created": time.strftime("%Y-%m-%d %H:%M"), "units": "percentage points; pp^2",
         "frozen_map": {"path": rel(os.path.join(a.pred_dir, f"predictions_forced_map{suffix}.json")),
                        "sha256": frozen_map["sha256"], "created": frozen_map["created"]},
         "config": {"reliability_map": rel(MAP_CONFIG), "instability": rel(INST_CONFIG)},
         "pilots": None if pilots is None else {k: pilots[k] for k in ("n_pass", "n_models", "failed")},
         "suffix": suffix,
         "subpanel_sizes": icfg["subpanels"]["sizes"],
         "tasks": {}, "map_points": []}
    for task, x in S.items():
        panel, items = x["panel"], x["items"]
        n = len(items)
        Breq, Bfd, Pc = x["D"]
        Vreq, Vfd, Vpc = three_arm_arrays(x["inp"]["load"], x["fd_load"], x["val"], panel, items)
        kd = Breq.shape[0]
        sub = icfg["subpanels"]
        T = {"title": TASKS[task]["title"], "models": panel, "M": len(panel), "n_items": n,
             "K_discovery": kd, "K_validation": Vreq.shape[0], "steps_file": rel(os.path.join(a.out, f"forced_{task}.json"))}
        m = {q: iv.multiplicities(sampling.rng_for(f"forced_{task}_{q}", "bootstrap"), kd, n_boot)
             for q in ("adherence", "collection")}
        T["step2"] = {"reasoning": steps[task]["step2"],
                      "adherence": d2.step2(Bfd, Breq, sel, m["adherence"]),
                      "collection": d2.step2(Breq, Pc, sel, m["collection"])}
        ref = json.load(open(os.path.join(ROOT, "results", "dataset2", f"{TASKS[task]['collection']}.json"),
                             encoding="utf-8"))["step2"]
        dev = {k: abs(T["step2"]["collection"][k]["estimate"] - ref[k]["estimate"]) for k in ("H", "V_delta")}
        if max(dev.values()) > 1e-9:
            raise SystemExit(f"{task}: the collection's step2 does not reproduce results/dataset2 ({dev})")
        T["collection_check"] = {"reference": f"results/dataset2/{TASKS[task]['collection']}.json", "abs_dev": dev}
        T["parts"] = parts_block(Breq, Bfd, Pc)
        fd_all = x["fd_load"](list(x["disc"]) + list(x["val"]))
        T["per_model"] = per_model(panel, Breq, Bfd, Pc, fd_all, x["box"])
        T["means"] = {k: float(np.mean([r[k] for r in T["per_model"]])) for k in
                      ("requested_direct", "forced_direct", "chain_of_thought", "adherence_cost", "reasoning_benefit",
                       "collection_gain")}
        T["scale"] = scale_block(Bfd, Pc, Vfd, Vpc, n, method, alpha)
        T["scale_requested"] = scale_block(Breq, Pc, Vreq, Vpc, n, method, alpha)
        refs = {sc: T["scale"][sc]["discovery"]["slope"] for sc in ins.SCALES}
        T["prompt"] = prompt_block({"units": (np.vstack([Bfd, Vfd]), np.vstack([Pc, Vpc]), refs)}, n)["units"]
        tiers = x["inp"]["tiers"]
        idx = {mk: i for i, mk in enumerate(panel)}
        strata = {"random": list(range(len(panel)))}
        if tiers:
            for tier in ("low", "middle", "high"):
                strata[f"tercile:{tier}"] = [idx[mk] for mk in panel if tiers[mk] == tier]
        # The collection's stream (scripts/run_instability_tasks.py), so the subpanels are the same as with the
        # requested-direct baseline and the two shares compare like with like.
        summ, rows = panel_block(TASKS[task]["collection"], (Bfd, Pc, Vfd, Vpc), n, strata, sub["sizes"],
                                 sub["n_resamples"], sub["seed_label"], method, alpha)
        T["panel"] = summ
        T["_subpanel_rows"] = rows
        b, g = Bfd.mean(0), (Pc - Bfd).mean(0)
        q2 = np.polyfit(b, g, 2)
        T["shape"] = {"baseline": b, "gain": g, "quadratic_coefficients": q2.tolist(),
                      "vertex_baseline": float(-q2[1] / (2 * q2[0])) if q2[0] != 0 else None,
                      "curvature": "concave" if q2[0] < 0 else "convex" if q2[0] > 0 else "flat",
                      "note": "model means over discovery draws; the quadratic is descriptive"}
        pts, point, obs = map_points(task, Bfd, Pc, Vfd, Vpc, frozen_map, mcfg, n_splits, len(panel), n)
        T["map"] = {"point": point, "validation_one_draw": obs["one_draw"]}
        R["map_points"] += pts
        if not a.dry_run:
            T["gpu_check"] = gpu_check(x["inp"]["provenance"], x["fd_prov"])
        R["tasks"][task] = T
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    rows = []
    # A task run apart (MATH-500) also puts its per-model table beside the default run's, where the later analyses read results/forced_direct/tables/per_model_<task>.csv.
    shared = suffix and not a.dry_run and os.path.abspath(a.out) == os.path.abspath(os.path.join(OUT, suffix[1:]))
    for task, T in R["tasks"].items():
        pd.DataFrame(T["per_model"]).to_csv(os.path.join(a.out, "tables", f"per_model_{task}.csv"), index=False)
        if shared:
            os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
            pd.DataFrame(T["per_model"]).to_csv(os.path.join(OUT, "tables", f"per_model_{task}.csv"), index=False)
        rows += [{"task": task, **r} for r in T.pop("_subpanel_rows")]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tables", "subpanels.csv"), index=False)
    if suffix:
        R["decoding"] = {t: DECODING_NOTE.get(t, "config/decoding_forced.yaml") for t in tasks}
    dump(R, os.path.join(a.out, "forced_direct.json"))
    write_summary(R, a.out)
    print(f"done in {R['runtime_seconds']} s -> {rel(a.out)}")
    return R


# ============================================================================ macros and summary


def fmt(x, nd=2):
    if x is None:
        return "--"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "--"
    return f"{x:.{nd}f}".replace("-", "−")


def hset(h):
    s = h.get("set") or {}
    return f"[{fmt(s['a'])}, {fmt(s['b'])}]" if s.get("kind") == "bounded" else str(s.get("kind", "--")).replace("_", " ")


def ci(x, nd=2):
    c = x.get("ci95")
    return "--" if not c else f"[{fmt(c[0], nd)}, {fmt(c[1], nd)}]"


def macros(R):
    m = {}

    def put(k, v):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
        m[k] = v.replace("−", "$-$")
    if R.get("pilots") and not R.get("suffix"):   # the pilots belong to the default run's macros
        put("fdPilotPass", str(R["pilots"]["n_pass"]))
        put("fdPilotModels", str(R["pilots"]["n_models"]))
    for task, T in R["tasks"].items():
        t = "fd" + TASKS[task]["tag"]
        put(t + "M", str(T["M"]))
        for k, s in (("requested_direct", "MeanRequested"), ("forced_direct", "MeanForced"),
                     ("chain_of_thought", "MeanCot"), ("adherence_cost", "MeanAdherence"),
                     ("reasoning_benefit", "MeanReasoning"), ("collection_gain", "MeanGain")):
            put(t + s, fmt(T["means"][k], 1))
        for q, s in (("adherence", "Adherence"), ("reasoning", "Reasoning"), ("collection", "Collection")):
            st = T["step2"][q]
            put(t + "V" + s, fmt(st["V_delta"]["estimate"], 1))
            if st["V_delta"].get("ci95"):
                put(t + "V" + s + "Lo", fmt(st["V_delta"]["ci95"][0], 1))
                put(t + "V" + s + "Hi", fmt(st["V_delta"]["ci95"][1], 1))
            put(t + "H" + s, fmt(st["H"]["estimate"], 2))
            put(t + "H" + s + "Set", hset(st["H"]))
        p = T["parts"]
        put(t + "CovParts", fmt(p["cross_moment"]["estimate"], 1))
        put(t + "CorrParts", fmt(p["corr"]["estimate"], 2))
        for sc, s in (("points", "Points"), ("logit", "Logit"), ("error_reduction", "ErrRed")):
            put(t + "Slope" + s, fmt(T["scale"][sc]["discovery"]["slope"], 3))
            ss = T["scale"][sc]["discovery"]["slope_set"]
            bounded = ss["kind"] == "bounded"
            put(t + "Slope" + s + "Lo", fmt(ss["a"], 3) if bounded else ss["kind"])
            put(t + "Slope" + s + "Hi", fmt(ss["b"], 3) if bounded else ss["kind"])
            put(t + "SlopeRequested" + s, fmt(T["scale_requested"][sc]["discovery"]["slope"], 3))
        for st, s in (("random", "Random"), ("tercile:low", "Low"), ("tercile:middle", "Middle"),
                      ("tercile:high", "High")):
            if st in T["panel"]:
                put(t + "SubNeg" + s, fmt(100 * T["panel"][st]["pooled_sizes"]["points"]["share_negative"], 0))
        pm = T["per_model"]
        if "box_closed_pct" in pm[0]:
            put(t + "BoxClosedMin", fmt(min(r["box_closed_pct"] for r in pm), 1))
            put(t + "ClosedReadableMin", fmt(min(r["closed_and_readable_pct"] for r in pm), 1))
            put(t + "ClosedReadableMedian", fmt(float(np.median([r["closed_and_readable_pct"] for r in pm])), 1))
            put(t + "Fallback", str(sum(r["fallback_no_prefill"] for r in pm)))
        mp = T["map"]["point"]
        put(t + "MapRben", fmt(mp["benefit"]["R"], 2))
        put(t + "MapCben", fmt(mp["benefit"]["C"], 3))
        put(t + "MapCref", fmt(mp["benefit"]["C_ref"], 3))
    return m


def write_summary(R, out_dir):
    stop = "stop at `}`"
    files = "`forced_gsm8k.json` and `forced_bbh.json`"
    if "decoding" in R:                    # a run of other tasks than GSM8K and BIG-Bench Hard (MATH-500)
        stop = "; ".join(f"{TASKS[t]['title']}: {v}" for t, v in R["decoding"].items())
        files = " and ".join(f"`forced_{t}.json`" for t in R["tasks"])
    L = ["# The forced-direct arm", "",
         "The collection's direct-answer prompts, unchanged, with the assistant turn "
         f"prefilled with `\\boxed{{` and at most 16 new tokens ({stop}, greedy, each model's kernels). Scores in "
         "percentage points, variances in pp². Panel moments use denominator M − 1. Steps 1–3 use forced-direct as "
         "the baseline arm with the methods and margins of the main analysis; their rules were saved to "
         "`manifests/predictions_forced_<task>.json` before the validation draws were built (the files "
         f"{files} here). H = −10 × the slope of gain on baseline: positive means "
         "weaker models gain more.", ""]
    if R["dry_run"]:
        L += ["**Dry run: the forced arm is the requested-direct arm relabelled. No number here is a result.**", ""]
    if R.get("pilots"):
        p = R["pilots"]
        L += [f"Pilots: {p['n_pass']} of {p['n_models']} pass (`pilot_check.md`)" +
              (f"; failed: {', '.join(p['failed'])}." if p["failed"] else "."), ""]
    for task, T in R["tasks"].items():
        s2 = T["step2"]
        L += [f"## {T['title']}", "",
              f"{T['M']} models, {T['n_items']:,} items, {T['K_discovery']} discovery and {T['K_validation']} "
              "validation draws. Three arms per model (discovery means, `tables/per_model_" + task + ".csv`):", "",
              "| Quantity | Mean over models | V_delta (pp²) | H | H set |", "|---|---:|---:|---:|---|"]
        for q, mk in (("adherence", "adherence_cost"), ("reasoning", "reasoning_benefit"),
                      ("collection", "collection_gain")):
            st = s2[q]
            L.append(f"| {QUANT[q]} | {fmt(T['means'][mk], 1)} | {fmt(st['V_delta']['estimate'], 1)} "
                     f"{ci(st['V_delta'], 1)} | {fmt(st['H']['estimate'])} | {hset(st['H'])} |")
        p = T["parts"]
        L += ["", f"Mean accuracy: requested-direct {fmt(T['means']['requested_direct'], 1)}, forced-direct "
                  f"{fmt(T['means']['forced_direct'], 1)}, chain of thought {fmt(T['means']['chain_of_thought'], 1)}. "
                  f"The cross-draw moment of adherence cost and reasoning benefit is {fmt(p['cross_moment']['estimate'], 1)} "
                  f"{ci(p['cross_moment'], 1)} pp² (correlation {fmt(p['corr']['estimate'])}), and V_gain = V_reasoning + "
                  f"V_adherence − 2 × that moment holds to {abs(p['identity_residual']):.1e}. The collection's gain "
                  f"reproduces `{T['collection_check']['reference']}` (H and V_delta within 1e-9).", "",
              "Slope of gain on baseline (discovery draws), three scales:", "",
              "| Baseline | Points (pp/pp) | Logits | Relative error reduction |", "|---|---:|---:|---:|",
              "| forced-direct | " + " | ".join(fmt(T["scale"][sc]["discovery"]["slope"], 3) for sc in ins.SCALES) + " |",
              "| requested-direct | " + " | ".join(fmt(T["scale_requested"][sc]["discovery"]["slope"], 3)
                                               for sc in ins.SCALES) + " |", ""]
        neg = {st: T["panel"][st]["pooled_sizes"]["points"]["share_negative"] for st in T["panel"]}
        L += ["Subpanels of " + ", ".join(map(str, R["subpanel_sizes"])) + " models, share with a negative slope in "
              "points (forced-direct baseline): " +
              "; ".join(f"{st} {100 * v:.0f}%" for st, v in neg.items()) + ".",
              f"Single draws with the opposite slope sign to the 24-draw slope: "
              f"{T['prompt']['points']['n_opposite']} of {T['prompt']['points']['n_draws']}.", "",
              f"Shape: the quadratic of gain on forced-direct baseline is {T['shape']['curvature']}"
              + (f", vertex at {fmt(T['shape']['vertex_baseline'], 0)} pp" if T["shape"]["vertex_baseline"] is not None
                 else "") + " (descriptive).", ""]
        mp = T["map"]["point"]
        L += [f"Map point: R benefit {fmt(mp['benefit']['R'])}, C benefit {fmt(mp['benefit']['C'], 3)} against "
              f"reference {fmt(mp['benefit']['C_ref'], 3)}; R accuracy {fmt(mp['accuracy']['R'])}, C accuracy "
              f"{fmt(mp['accuracy']['C'], 3)}.", ""]
        pm = T["per_model"]
        if "box_closed_pct" in pm[0]:
            L += [f"Format: the box closes in {fmt(min(r['box_closed_pct'] for r in pm), 1)}–"
                  f"{fmt(max(r['box_closed_pct'] for r in pm), 1)}% of a model's outputs, with a readable answer in "
                  f"{fmt(min(r['closed_and_readable_pct'] for r in pm), 1)}–"
                  f"{fmt(max(r['closed_and_readable_pct'] for r in pm), 1)}%. Models without the prefill (fallback): "
                  f"{sum(r['fallback_no_prefill'] for r in pm)}.", ""]
        if T.get("gpu_check"):
            mm = T["gpu_check"]["mismatched"]
            L += [f"GPU type as in the collection: {T['M'] - len(mm)} of {T['M']} models"
                  + (f" (differs: {', '.join(mm)})." if mm else "."), ""]
        L += ["| Model | Requested-direct | Forced-direct | Chain of thought | Adherence cost | Reasoning benefit |"
              + (" Closed and readable |" if "box_closed_pct" in pm[0] else ""),
              "|---|---:|---:|---:|---:|---:|" + ("---:|" if "box_closed_pct" in pm[0] else "")]
        for r in sorted(pm, key=lambda r: r["forced_direct"]):
            L.append(f"| {r['model_key']} | {fmt(r['requested_direct'], 1)} | {fmt(r['forced_direct'], 1)} | "
                     f"{fmt(r['chain_of_thought'], 1)} | {fmt(r['adherence_cost'], 1)} | "
                     f"{fmt(r['reasoning_benefit'], 1)} |" +
                     (f" {fmt(r['closed_and_readable_pct'], 1)} |" if "box_closed_pct" in r else ""))
        L.append("")
    txt = "\n".join(L)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=None, help="default manifests/ (a dry run: <out>/frozen)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="999 bootstrap draws, 100 control replicates, 200 splits")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS),
                    help="default gsm8k bbh; any other set (MATH-500: --tasks math) writes under "
                         "results/forced_direct/<tasks>/ and names its frozen map file after the tasks")
    a = ap.parse_args(argv)
    a.tasks = list(dict.fromkeys(a.tasks))
    a.suffix = "" if tuple(a.tasks) == DEFAULT_TASKS else "_" + "_".join(a.tasks)
    if a.dry_run and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("a dry run needs --out outside results/forced_direct")
    if a.suffix and os.path.abspath(a.out) == os.path.abspath(OUT):
        a.out = os.path.join(OUT, a.suffix[1:])
    a.pred_dir = a.pred_dir or (os.path.join(a.out, "frozen") if a.dry_run else MANIFESTS)
    run(a)


if __name__ == "__main__":
    main()
