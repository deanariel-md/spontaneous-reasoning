r"""Reliability map by panel, the exemplar contrast as a third intervention, and pilots against R
.

Run from the repository root:
    python scripts/run_map_panels.py
    python scripts/run_map_panels.py --forced     # once results/forced_direct/ exists
    # MATH-500 joins through the reliability map (--math there), its forced-direct point through --forced
    # (results/forced_direct/math/), and the full-scale exemplar and native-reasoning points by flag
    python scripts/run_map_panels.py --forced --exemplar --thinking
    # the contrasts of config/added_contrasts.yaml, once the reliability map has them (--added there)
    python scripts/run_map_panels.py --forced --exemplar --thinking --added

Inputs: results/reliability_map/reliability_map.json (every existing map point and its pilot errors), the Cheng et
al. Part B table (data/processed/reproduction/transport_scored.parquet; 8 configurations, zero-shot against
8-exemplar chain of thought), and the loaders of scripts/run_reliability_map.py for the draws of every setting with
separate validation draws. With --forced, the forced-direct points of results/forced_direct/forced_direct.json.

A1. Cheng Part B is one setting of 13 BF16 models (the two FP8 70B-class models in a 15-model sensitivity): R, C and
the reference from the 8 configurations as distinct draws, and the arm-wise split of the gain noise
(src/noise_anatomy.arm_split). No pilot test: there is no validation split.

A3. The map points grouped by panel, with the in-sample departure from the reference and the pilot prediction errors
at 2 and 4 draws (from the map's pilot test). Then, in every setting with separate validation draws, the K-draw pilot
(K = 1, 2, 4), the capability rule and own gain over all pilot or discovery draws predict the centred benefit; each is
scored on the validation draws by cross-product risk and reported as a share of the constant rule's risk. Stated
before computing: a K-draw pilot's risk is about W_gain / K against V_delta for the constant, a share of
(1 - R) / (K R), so it beats the constant when R > 1 / (K + 1). The predictions are written to
manifests/predictions_map_panels.json before the validation draws are built. Those draws were read by earlier
analyses, so this is a description of draws already seen, not a fresh test.

Added contrasts (--added): the points of results/reliability_map/reliability_map.json "added_settings", one panel per
contrast group (config/added_contrasts.yaml, map_panel), with their pilot errors from its "added_pilot_test"; and the
pilots-against-R rules on each contrast's discovery draws, saved apart in manifests/predictions_map_panels_added.json
(a later pool of the file, such as HotpotQA's pool hotpot: ..._added_<pool>.json, keys with the suffix _<pool>)
before its validation draws are built. They are kept out of every count, panel summary row and pilot summary of the main
settings (added_* keys and macros), so those numbers are unaffected.

Outputs: results/map_panels/map_panels.json (with a "macros" object, prefix "mp"), summary.md, summary_clean.md,
tables/*.csv.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import dataset2 as d2, noise_anatomy as na, reliability_map as rm  # noqa: E402
from src.estimators import cross_risk  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

OUT = os.path.join(ROOT, "results", "map_panels")
MAP = os.path.join(ROOT, "results", "reliability_map", "reliability_map.json")
PRED = os.path.join(ROOT, "manifests", "predictions_map_panels.json")
PRED_ADDED = os.path.join(ROOT, "manifests", "predictions_map_panels_added.json")
FORCED = os.path.join(ROOT, "results", "forced_direct", "forced_direct.json")
FORCED_MATH = os.path.join(ROOT, "results", "forced_direct", "math", "forced_direct.json")
EXEMPLAR = os.path.join(ROOT, "results", "exemplar", "exemplar.json")
THINKING = os.path.join(ROOT, "results", "thinking", "thinking.json")
PANEL_OF = {"dove_mmlu": "DOVE (5)", "dove_tasks": "DOVE (5)", "gsm8k": "GSM8K (19)", "gsm8k30": "GSM8K (29)",
            "bbh": "BIG-Bench Hard (28)", "bbh_tasks": "BIG-Bench Hard (28)", "cheng": "Cheng et al. (13)",
            "forced": "Forced-direct", "math": "MATH-500", "exemplar": "Eight exemplars",
            "thinking": "Native reasoning"}
# Points counted apart from the settings the manuscript adds up (mpPoints): a second baseline or a second
# intervention on settings already counted.
APART = ("forced", "exemplar", "thinking")
PILOT_K = (1, 2, 4)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_map_panels", os.path.join(SCRIPT_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        return None
    return o


def c(x):
    x = np.asarray(x, float)
    return x - x.mean(axis=-1, keepdims=True)


# ============================================================================ A1: Cheng Part B


def cheng():
    cs = load_script("run_cheng_scales")
    from src import reproduction as rp
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "instability.yaml"), encoding="utf-8"))["cheng"]
    tr = pd.read_parquet(cs.TRANSPORT)
    configs, models, B, P, n_items = cs.arm_arrays(tr[tr["part"] == "transport"], cfg["baseline_arm"],
                                                   cfg["intervention_arm"])
    rcfg = rp.load_config()
    bf = [k for k, mk in enumerate(models) if rcfg["models"][mk].get("precision", "bf16") == "bf16"]
    out = {"configurations": configs, "n_items": n_items, "baseline_arm": cfg["baseline_arm"],
           "intervention_arm": cfg["intervention_arm"], "models_all": models, "models_bf16": [models[k] for k in bf]}
    for lab, idx in (("bf16", bf), ("all", list(range(len(models))))):
        b, p = B[:, idx], P[:, idx]
        out[lab] = {"M": len(idx), "point": rm.setting_point(b, p), "noise_split": na.arm_split(b, p, "draws"),
                    "mean_gain": float((p - b).mean()), "mean_baseline": float(b.mean())}
    return out


# ============================================================================ A3: settings with validation draws


def settings_with_validation():
    """(name, group, B_pilot, P_pilot, B_val, P_val, n_pilot_draws) for every setting with separate validation
    draws, from the loaders of scripts/run_reliability_map.py."""
    import yaml
    rmap = load_script("run_reliability_map")
    cfg = yaml.safe_load(open(rmap.CONFIG, encoding="utf-8"))
    out = []
    s = rmap.dove_mmlu_inputs()
    pi, vi = np.asarray(s["pilot"]), np.asarray(s["validation"])
    out.append((s["setting"], "dove_mmlu", s["B"][pi], s["P"][pi], s["B"][vi], s["P"][vi]))
    tasks, _, _ = rmap.dove_task_inputs(cfg)
    for s in tasks:
        pi, vi = np.asarray(s["pilot"]), np.asarray(s["validation"])
        out.append((s["setting"], "dove_tasks", s["B"][pi], s["P"][pi], s["B"][vi], s["P"][vi]))
    gs = os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet")
    for inp in (rmap.gsm8k19_inputs(gs), rmap.gsm8k30_inputs(gs, None), rmap.bbh_inputs(None)):
        out.extend(d2_settings(inp))
    return out


def d2_settings(inp):
    sampler, panel = inp["sampler"], inp["panel"]
    disc, val = sampler["discovery_config_ids"], sampler["validation_config_ids"]
    dd, dv = inp["load"](disc), None
    rows = []
    for name, (_, its) in inp["subsets"].items():
        cells = d2.cell_table(dd[dd["item_id"].isin(pd.Index(its))], panel, disc, its)
        Bd, Pd = d2.unit_arrays(cells, disc, panel, "strict_correct")
        rows.append([name, inp["group_of"][name], Bd, Pd, None, None])
    return [("d2", inp, rows)]


def rules(Bp, Pp):
    """Centred predictions from pilot or discovery draws (K, M)."""
    G = Pp - Bp
    r = {"constant": np.zeros(G.shape[1])}
    for k in PILOT_K:
        if G.shape[0] >= k:
            r[f"pilot_{k}"] = c(G[:k].mean(0))
    r[f"own_{G.shape[0]}"] = c(G.mean(0))
    b, g = c(Bp.mean(0)), c(G.mean(0))
    r["capability"] = (float(b @ g) / float(b @ b)) * b if float(b @ b) > 0 else None
    return r


def pilot_test(raw):
    """Freeze every rule from pilot or discovery draws, then score on validation draws."""
    settings, frozen = [], {}
    for item in raw:
        if item[0] == "d2":
            _, inp, rows = item
            for name, group, Bd, Pd, _, _ in rows:
                settings.append({"name": name, "group": group, "Bp": Bd, "Pp": Pd, "inp": inp})
        else:
            name, group, Bp, Pp, Bv, Pv = item
            settings.append({"name": name, "group": group, "Bp": Bp, "Pp": Pp, "Bv": Bv, "Pv": Pv})
    for s in settings:
        s["rules"] = rules(s["Bp"], s["Pp"])
        frozen[s["name"]] = {k: (None if v is None else v.tolist()) for k, v in s["rules"].items()}
    rec = {"label": "map_panels", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp, centred",
           "note": "rules from pilot or discovery draws; written before the validation draws are built", "rules": frozen}
    rec["models"] = sorted(frozen)          # here: the settings, which write_frozen compares
    rec["sha256"] = sha256_json(json.loads(json.dumps(frozen)))
    rec = d2.write_frozen(rec, PRED)
    # validation draws, built only now
    loaded = {}
    for s in settings:
        if "inp" in s:
            inp = s["inp"]
            key = id(inp)
            if key not in loaded:
                val = inp["sampler"]["validation_config_ids"]
                loaded[key] = (inp["load"](val), val)
            dv, val = loaded[key]
            its = inp["subsets"][s["name"]][1]
            cells = d2.cell_table(dv[dv["item_id"].isin(pd.Index(its))], inp["panel"], val, its)
            s["Bv"], s["Pv"] = d2.unit_arrays(cells, val, inp["panel"], "strict_correct")
    rows = []
    for s in settings:
        Z = c(s["Pv"] - s["Bv"])
        pt = rm.setting_point(s["Bp"], s["Pp"])["benefit"]
        risk = {k: (None if v is None else cross_risk(Z, v)) for k, v in s["rules"].items()}
        r0 = risk["constant"]
        row = {"setting": s["name"], "group": s["group"], "panel": PANEL_OF[s["group"]], "M": int(Z.shape[1]),
               "K_pilot_draws": int(s["Bp"].shape[0]), "K_validation": int(Z.shape[0]), "R_from_pilot_draws": pt["R"],
               "risk_constant": r0}
        for k, v in risk.items():
            if k != "constant":
                row[f"share_{k}"] = None if v is None or r0 is None or r0 <= 0 else v / r0
                row[f"risk_{k}"] = v
        rows.append(row)
    return rows, {"path": os.path.relpath(PRED, ROOT).replace("\\", "/"), "created": rec["created"],
                  "sha256": rec["sha256"]}


# ============================================================================ map by panel


def map_rows(m, ch, forced, extra=()):
    rows = []
    pe = {}
    for x in m["pilot_test"]["prediction_rows"]:
        if x["target"] == "one_draw" and x["K"] == 1:
            pe[(x["setting"], x["n_pilot"], x["quantity"])] = x["error"]
    for s in m["settings"]:
        for q in ("benefit", "accuracy"):
            p = s["point"][q]
            rows.append({"setting": s["setting"], "group": s["group"], "panel": PANEL_OF[s["group"]], "quantity": q,
                         "M": s["M"], "n_items": s["n_items"], "R": p["R"], "C": p["C"], "C_ref": p.get("C_ref"),
                         "departure": p.get("departure"), "tie": p.get("tie"),
                         "pilot_error_2": pe.get((s["setting"], 2, q)), "pilot_error_4": pe.get((s["setting"], 4, q))})
    for q in ("benefit", "accuracy"):
        p = ch["bf16"]["point"][q]
        rows.append({"setting": "cheng:part_b", "group": "cheng", "panel": PANEL_OF["cheng"], "quantity": q,
                     "M": ch["bf16"]["M"], "n_items": ch["n_items"], "R": p["R"], "C": p["C"], "C_ref": p.get("C_ref"),
                     "departure": p.get("departure"), "tie": p.get("tie"), "pilot_error_2": None, "pilot_error_4": None})
    for f in forced:
        rows.append({**f, "group": "forced", "panel": PANEL_OF["forced"]})
    for f in extra:                     # "exemplar:gsm8k", "thinking:gsm8k", "thinking:math"
        g = f["setting"].split(":")[0]
        rows.append({**f, "group": g, "panel": PANEL_OF[g]})
    return rows


def pk(key, pool):
    """Results key of an added pool: pool base keeps the plain keys, a later pool adds _<pool>."""
    import added_contrasts as ac
    return ac.pool_key(key, pool)


def all_added_groups(R):
    """The groups of every added pool (kept out of the main counts)."""
    return {g for pool in R.get("added_pools", ["base"]) for g in R.get(pk("added_groups", pool), [])}


def added_rows(m, pool="base"):
    """Map rows of the added contrasts of a pool (reliability_map.json "added_settings", or "added_settings_<pool>"),
    with their one-draw pilot errors at 2 and 4 draws from its pilot test; group and panel as
    config/added_contrasts.yaml names them."""
    recs = m.get(pk("added_settings", pool)) or []
    pe = {}
    for x in (m.get(pk("added_pilot_test", pool)) or {}).get("prediction_rows", []):
        if x["target"] == "one_draw" and x["K"] == 1:
            pe[(x["setting"], x["n_pilot"], x["quantity"])] = x["error"]
    rows = []
    for s in recs:
        for q in ("benefit", "accuracy"):
            p = s["point"][q]
            rows.append({"setting": s["setting"], "group": s["group"], "panel": s["map_panel"], "quantity": q,
                         "M": s["M"], "n_items": s["n_items"], "R": p["R"], "C": p["C"], "C_ref": p.get("C_ref"),
                         "departure": p.get("departure"), "tie": p.get("tie"),
                         "pilot_error_2": pe.get((s["setting"], 2, q)), "pilot_error_4": pe.get((s["setting"], 4, q))})
    return rows



def risk_row(name, group, panel, Bp, Pp, Bv, Pv, rls):
    """One setting of the pilots-against-R test: each rule's cross-product risk on the validation draws and its share
    of the constant rule's (none where that risk is not positive)."""
    Z = c(Pv - Bv)
    pt = rm.setting_point(Bp, Pp)["benefit"]
    risk = {k: (None if v is None else cross_risk(Z, v)) for k, v in rls.items()}
    r0 = risk["constant"]
    row = {"setting": name, "group": group, "panel": panel, "M": int(Z.shape[1]), "K_pilot_draws": int(Bp.shape[0]),
           "K_validation": int(Z.shape[0]), "R_from_pilot_draws": pt["R"], "risk_constant": r0}
    for k, v in risk.items():
        if k != "constant":
            row[f"share_{k}"] = None if v is None or r0 is None or r0 <= 0 else v / r0
            row[f"risk_{k}"] = v
    return row


def added_pred_path(pool):
    """The rules' file of a pool: manifests/predictions_map_panels_added.json for pool base, ..._added_<pool>.json."""
    return PRED_ADDED if pool == "base" else PRED_ADDED.replace(".json", f"_{pool}.json")


def added_pilot_test(con, pred_path=PRED_ADDED, label="map_panels_added"):
    """Pilots against R on the added contrasts: rules from the 24 discovery draws, frozen in their own file before the
    validation draws are built, then scored on the validation draws."""
    import added_contrasts as ac
    rmap = load_script("run_reliability_map")
    settings, frozen = [], {}
    for name, cn in con.items():
        x = ac.inputs(name, con)
        Bd, Pd = ac.arrays(x, "discovery")
        s = {"name": rmap.added_setting(cn), "group": cn["group"], "panel": cn["map_panel"], "x": x, "Bp": Bd,
             "Pp": Pd, "rules": rules(Bd, Pd)}
        frozen[s["name"]] = {k: (None if v is None else v.tolist()) for k, v in s["rules"].items()}
        settings.append(s)
    rec = {"label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp, centred",
           "note": "rules of the added contrasts (config/added_contrasts.yaml) from their discovery draws; written "
                   "before the validation draws are built", "rules": frozen}
    rec["models"] = sorted(frozen)
    rec["sha256"] = sha256_json(json.loads(json.dumps(frozen)))
    rec = d2.write_frozen(rec, pred_path)
    rows = []
    for s in settings:
        Bv, Pv = ac.arrays(s["x"], "validation")             # validation draws, built only now
        rows.append(risk_row(s["name"], s["group"], s["panel"], s["Bp"], s["Pp"], Bv, Pv, s["rules"]))
    return rows, {"path": os.path.relpath(pred_path, ROOT).replace("\\", "/"), "created": rec["created"],
                  "sha256": rec["sha256"]}


def full_points(path, what):
    """The map points of a full run of the exemplar or thinking collection (none from a dry or partial run)."""
    if not os.path.exists(path):
        raise SystemExit(f"--{what}: {os.path.relpath(path, ROOT)} is missing")
    r = json.load(open(path, encoding="utf-8"))
    if r.get("dry_run") or r.get("partial"):
        raise SystemExit(f"--{what}: {os.path.relpath(path, ROOT)} is a dry or partial run")
    return r["map_points"]


def panel_summary(rows):
    df = pd.DataFrame(rows)
    out = []
    for (panel, q), g in df.groupby(["panel", "quantity"], sort=False):
        dep = g["departure"].dropna()
        out.append({"panel": panel, "quantity": q, "points": int(len(g)), "R_median": float(g["R"].median()),
                    "R_min": float(g["R"].min()), "R_max": float(g["R"].max()),
                    "mean_departure": float(dep.mean()) if len(dep) else None,
                    "max_abs_departure": float(dep.abs().max()) if len(dep) else None,
                    "pilot2_mean_error": _m(g["pilot_error_2"]), "pilot2_mae": _m(g["pilot_error_2"].abs()),
                    "pilot4_mean_error": _m(g["pilot_error_4"]), "pilot4_mae": _m(g["pilot_error_4"].abs())})
    return out


def _m(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


# ============================================================================ summary and macros


def fmt(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def pilot_summary(prow):
    df = pd.DataFrame(prow)
    out = []
    for panel, g in df.groupby("panel", sort=False):
        r = {"panel": panel, "settings": int(len(g)), "R_median": float(g["R_from_pilot_draws"].median())}
        for k in ("pilot_1", "pilot_2", "pilot_4", "capability"):
            s = pd.to_numeric(g[f"share_{k}"], errors="coerce").dropna()
            r[f"{k}_median_share"] = float(s.median()) if len(s) else None
            r[f"{k}_beats_constant"] = int((s < 1).sum())
            r[f"{k}_n"] = int(len(s))
        out.append(r)
    lo = df[df["R_from_pilot_draws"] < 0.5]
    hi = df[df["R_from_pilot_draws"] >= 0.5]
    split = {}
    for lab, g in (("R_below_half", lo), ("R_at_least_half", hi)):
        s = pd.to_numeric(g["share_pilot_1"], errors="coerce").dropna()
        split[lab] = {"settings": int(len(s)), "pilot_1_beats_constant": int((s < 1).sum()),
                      "median_share": float(s.median()) if len(s) else None}
    return out, split


def macros(R):
    m = {}
    for lab, tag in (("bf16", ""), ("all", "All")):
        x = R["cheng"][lab]
        b, a, ns = x["point"]["benefit"], x["point"]["accuracy"], x["noise_split"]
        m[f"mpCheng{tag}M"] = str(x["M"])
        m[f"mpCheng{tag}Vdelta"] = fmt(b["signal"], 2)
        m[f"mpCheng{tag}Wgain"] = fmt(b["noise"], 2)
        m[f"mpCheng{tag}Rben"] = fmt(b["R"], 2)
        m[f"mpCheng{tag}Cben"] = fmt(b["C"], 3)
        m[f"mpCheng{tag}Cref"] = fmt(b["C_ref"], 3)
        m[f"mpCheng{tag}Cacc"] = fmt(a["C"], 3)
        m[f"mpCheng{tag}Racc"] = fmt(a["R"], 2)
        m[f"mpCheng{tag}VarP"] = fmt(ns["var_e_intervention"], 2)
        m[f"mpCheng{tag}VarB"] = fmt(ns["var_e_baseline"], 2)
        m[f"mpCheng{tag}Corr"] = fmt(ns["corr_e"], 2)
    for k, v in R["pilot_split"].items():
        t = "Low" if k == "R_below_half" else "High"
        m[f"mpPilot{t}N"] = str(v["settings"])
        m[f"mpPilot{t}Wins"] = str(v["pilot_1_beats_constant"])
        m[f"mpPilot{t}Share"] = fmt(100 * v["median_share"], 0) if v["median_share"] is not None else "--"
    tag = {"DOVE (5)": "Dove", "GSM8K (19)": "Gsm", "GSM8K (29)": "GsmThirty", "BIG-Bench Hard (28)": "Bbh",
           "MATH-500": "Math"}
    for row in R["pilot_panels"]:
        t = tag.get(row["panel"])
        if t is None:
            continue
        m[f"mpPilot{t}Settings"] = str(row["pilot_1_n"])
        for rule, rt in (("pilot_1", "One"), ("pilot_2", "Two"), ("pilot_4", "Four"), ("capability", "Cap")):
            m[f"mpPilot{t}{rt}Wins"] = str(row[f"{rule}_beats_constant"])
            sh = row[f"{rule}_median_share"]
            m[f"mpPilot{t}{rt}Share"] = "--" if sh is None else fmt(100 * sh, 0)
    for row in R["panels"]:
        t = tag.get(row["panel"])
        if t is not None and row["quantity"] == "benefit":
            m[f"mpMap{t}RMedian"] = fmt(row["R_median"], 2)
            for k, kt in (("pilot2_mae", "PilotTwoMae"), ("pilot4_mae", "PilotFourMae")):
                if row.get(k) is not None:
                    m[f"mpMap{t}{kt}"] = fmt(row[k], 3)
    # mpPoints and mpPanels count the settings with validation draws plus the exemplar contrast, the count the
    # manuscript adds up; the forced-direct points (a second baseline for settings already counted) and the points of
    # items H and I (a second intervention on GSM8K-29 and MATH-500) are separate
    # the added contrasts (config/added_contrasts.yaml) stay out of every count below; their own macros follow
    added = all_added_groups(R)
    rows = [r for r in R["map_rows"] if r["group"] not in APART and r["group"] not in added]
    m["mpPoints"] = str(len({r["setting"] for r in rows}))
    m["mpPanels"] = str(len({r["panel"] for r in rows}))
    m["mpPointsNoCheng"] = str(len({r["setting"] for r in rows if r["group"] != "cheng"}))
    m["mpPointsForced"] = str(len({r["setting"] for r in R["map_rows"] if r["group"] == "forced"}))
    wf = [r for r in R["map_rows"] if r["group"] not in ("exemplar", "thinking") and r["group"] not in added]
    m["mpPointsWithForced"] = str(len({r["setting"] for r in wf}))
    m["mpPanelsWithForced"] = str(len({r["panel"] for r in wf}))
    for g, gt in (("exemplar", "Exemplar"), ("thinking", "Thinking")):
        if any(r["group"] == g for r in R["map_rows"]):
            m[f"mpPoints{gt}"] = str(len({r["setting"] for r in R["map_rows"] if r["group"] == g}))
            m["mpPointsAll"], m["mpPanelsAll"] = str(R["n_points"]), str(R["n_panels"])
    for pool in R.get("added_pools", ["base"]):
        m.update(added_macros(R, taken=m, pool=pool))
    for k in m:
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    return m


RULE_TAG = (("pilot_1", "One"), ("pilot_2", "Two"), ("pilot_4", "Four"), ("capability", "Cap"))


def added_macros(R, taken=(), pool="base"):
    """Macros of the added contrasts of a pool: their counts beside the main ones (mpPointsAdded, mpPanelsAdded,
    mpPointsAllWithAdded; a later pool's with its tag, mpPointsAddedHq: the main points plus that pool's) and per
    contrast tag the pilot rules' risk as a percentage of the constant rule's (mpPilotFsGsmOneShare, ...; "--" where the
    constant's risk is not positive). A name in `taken` is refused."""
    import added_contrasts as ac
    m = {}
    if not R.get(pk("added_groups", pool)):
        return m
    pt = ac.pool_info(pool)["tag"]
    rows = [r for r in R["map_rows"] if r["group"] in R[pk("added_groups", pool)]]
    m["mpPointsAdded" + pt] = str(len({r["setting"] for r in rows}))
    m["mpPanelsAdded" + pt] = str(len({r["panel"] for r in rows}))
    m["mpPointsAllWithAdded" + pt] = str(R["n_points"] + len({r["setting"] for r in rows}))
    tags = R.get(pk("added_tags", pool), {})
    for row in R.get(pk("added_pilot_rows", pool), []):
        t = tags.get(row["setting"])
        if t is None:
            continue
        own = next((k for k in row if k.startswith("share_own_")), None)
        for rule, rt in RULE_TAG + (((own[len("share_"):], "Own"),) if own else ()):
            sh = row.get(f"share_{rule}")
            m[f"mpPilot{t}{rt}Share"] = "--" if sh is None else fmt(100 * sh, 0)
        m[f"mpPilot{t}Rdisc"] = fmt(row["R_from_pilot_draws"], 2)
    clash = sorted(set(m) & set(taken))
    if clash:
        raise ValueError(f"added-contrast macros already set: {clash}")
    return m


def added_summary_lines(R, pool="base") -> list:
    import added_contrasts as ac
    if not R.get(pk("added_groups", pool)):
        return []
    L = ["", f"## Added contrasts ({ac.pool_info(pool)['title']}), kept apart", "",
         f"{R[pk('added_n_points', pool)]} points of `config/added_contrasts.yaml` "
         f"(`results/reliability_map/reliability_map.json`, `{pk('added_settings', pool)}`), outside every count and "
         "summary row "
         "above.", "",
         "| Panel | Setting | Quantity | M | R | C | Reference | Departure | Pilot 2 error | Pilot 4 error |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["map_rows"]:
        if r["group"] in R[pk("added_groups", pool)]:
            L.append(f"| {r['panel']} | {r['setting']} | {r['quantity']} | {r['M']} | {fmt(r['R'], 2)} | {fmt(r['C'])} | "
                     f"{fmt(r['C_ref'])} | {fmt(r['departure'])} | {fmt(r['pilot_error_2'])} | "
                     f"{fmt(r['pilot_error_4'])} |")
    L += ["", f"Pilots against R on the same contrasts, rules saved to `{R[pk('added_frozen', pool)]['path']}` "
              f"({R[pk('added_frozen', pool)]['created']}) before their validation draws were built; earlier analyses "
              "had read "
              "those draws. Risk as a share of the constant rule's:", "",
          "| Setting | R (discovery) | Pilot 1 | Pilot 2 | Pilot 4 | Capability rule | Own gain, all discovery draws |",
          "|---|---|---|---|---|---|---|"]
    for r in R[pk("added_pilot_rows", pool)]:
        own = next((k for k in r if k.startswith("share_own_")), None)
        cell = lambda v: "--" if v is None else f"{fmt(100 * v, 1)}%"
        L.append(f"| {r['setting']} | {fmt(r['R_from_pilot_draws'], 2)} | {cell(r.get('share_pilot_1'))} | "
                 f"{cell(r.get('share_pilot_2'))} | {cell(r.get('share_pilot_4'))} | {cell(r.get('share_capability'))} | "
                 f"{cell(r.get(own)) if own else '--'} |")
    L.append("")
    return L


def write_summary(R):
    ch = R["cheng"]
    b, a, ns = ch["bf16"]["point"]["benefit"], ch["bf16"]["point"]["accuracy"], ch["bf16"]["noise_split"]
    ba, aa, nsa = ch["all"]["point"]["benefit"], ch["all"]["point"]["accuracy"], ch["all"]["noise_split"]
    L = ["# Reliability map by panel; the exemplar contrast; pilots against R", "",
         "Scores in percentage points, variances in pp². R is the cross-draw "
         "panel moment over the mean within-draw panel variance (denominator M − 1); C is one-draw agreement of "
         "model-pair orderings with a tie counting one half; the reference is C = 1/2 + arcsin(R)/π.", "",
         "## The exemplar contrast (Cheng et al. Part B)", "",
         f"Eight-exemplar against zero-shot chain of thought on GSM8K, the {len(ch['configurations'])} transported "
         f"configurations as distinct draws, {ch['n_items']} items.", "",
         "| Panel | M | V_delta | W_gain | R benefit | C benefit | Reference | R accuracy | C accuracy | Var(e_P) | Var(e_B) | Corr(e_P, e_B) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|",
         f"| BF16 models | {ch['bf16']['M']} | {fmt(b['signal'], 2)} | {fmt(b['noise'], 2)} | {fmt(b['R'], 2)} | "
         f"{fmt(b['C'])} | {fmt(b['C_ref'])} | {fmt(a['R'], 2)} | {fmt(a['C'])} | {fmt(ns['var_e_intervention'], 2)} | "
         f"{fmt(ns['var_e_baseline'], 2)} | {fmt(ns['corr_e'], 2)} |",
         f"| With the two FP8 models (sensitivity) | {ch['all']['M']} | {fmt(ba['signal'], 2)} | {fmt(ba['noise'], 2)} | "
         f"{fmt(ba['R'], 2)} | {fmt(ba['C'])} | {fmt(ba['C_ref'])} | {fmt(aa['R'], 2)} | {fmt(aa['C'])} | "
         f"{fmt(nsa['var_e_intervention'], 2)} | {fmt(nsa['var_e_baseline'], 2)} | {fmt(nsa['corr_e'], 2)} |", "",
         "Reference values: V_delta about 2.9, R 0.85, C benefit 0.825 against 0.824, C accuracy 0.987, "
         "gain noise about 0.5.", "",
         "## The map by panel", "",
         f"{R['n_points']} task-level points from {R['n_panels']} panels. Departure is C − C_ref in sample; the pilot "
         "errors are the one-draw prediction from 2 and 4 pilot draws minus the observed validation agreement "
         "(`results/reliability_map/`).", "",
         "| Panel | Quantity | Points | R median [min, max] | Mean departure | Max abs departure | Pilot 2: mean error, MAE | Pilot 4: mean error, MAE |",
         "|---|---|---|---|---|---|---|---|"]
    for r in R["panels"]:
        L.append(f"| {r['panel']} | {r['quantity']} | {r['points']} | {fmt(r['R_median'], 2)} [{fmt(r['R_min'], 2)}, "
                 f"{fmt(r['R_max'], 2)}] | {fmt(r['mean_departure'])} | {fmt(r['max_abs_departure'])} | "
                 f"{fmt(r['pilot2_mean_error'])}, {fmt(r['pilot2_mae'])} | {fmt(r['pilot4_mean_error'])}, "
                 f"{fmt(r['pilot4_mae'])} |")
    L += ["", "## Pilots against R", "",
          "Each rule predicts the centred benefit from the pilot draws (DOVE benchmarks: their 4 pilot draws) or the "
          "discovery draws (DOVE block and Dataset 2: 24), and is scored on the validation draws by cross-product "
          "risk, as a share of the constant rule's risk (below 100% beats the constant). Expected share of a K-draw "
          "pilot: (1 − R)/(K R), so a one-draw pilot needs R > 1/2. Rules saved to "
          f"`{R['frozen']['path']}` ({R['frozen']['created']}) before the validation draws were built; those draws "
          "were read by earlier analyses.", "",
          "| Panel | Settings | R median | Pilot 1: median share, beats constant | Pilot 2 | Pilot 4 | Capability rule |",
          "|---|---|---|---|---|---|---|"]
    for r in R["pilot_panels"]:
        cell = lambda k: (f"{fmt(100 * r[k + '_median_share'], 0)}%, {r[k + '_beats_constant']} of {r[k + '_n']}"
                          if r[k + "_median_share"] is not None else "--")
        L.append(f"| {r['panel']} | {r['settings']} | {fmt(r['R_median'], 2)} | {cell('pilot_1')} | {cell('pilot_2')} | "
                 f"{cell('pilot_4')} | {cell('capability')} |")
    sp = R["pilot_split"]
    L += ["", f"With R below 1/2 the one-draw pilot beats the constant in {sp['R_below_half']['pilot_1_beats_constant']} "
              f"of {sp['R_below_half']['settings']} settings (median share "
              f"{fmt(100 * sp['R_below_half']['median_share'], 0) if sp['R_below_half']['median_share'] is not None else '--'}%); "
              f"with R at least 1/2 in {sp['R_at_least_half']['pilot_1_beats_constant']} of "
              f"{sp['R_at_least_half']['settings']} (median "
              f"{fmt(100 * sp['R_at_least_half']['median_share'], 0) if sp['R_at_least_half']['median_share'] is not None else '--'}%).",
          f"In {sum(1 for r in R['pilot_rows'] if r['share_pilot_1'] is None)} settings the constant rule's "
          "cross-product risk is not positive, so no share is formed; their risks are in `tables/pilot_risks.csv`.",
          ""]
    for pool in R.get("added_pools", ["base"]):
        L += added_summary_lines(R, pool)
    txt = "\n".join(L)
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--forced", action="store_true", help="add the forced-direct points of results/forced_direct/ "
                                                          "(and of results/forced_direct/math/ once it exists)")
    ap.add_argument("--exemplar", action="store_true", help="add the exemplar point (results/exemplar/)")
    ap.add_argument("--thinking", action="store_true", help="add the native-reasoning points (results/thinking/)")
    ap.add_argument("--added", action="store_true",
                    help="add the contrasts of config/added_contrasts.yaml, kept apart (needs --added on the map)")
    a = ap.parse_args(argv)
    t0 = time.time()
    m = json.load(open(MAP, encoding="utf-8"))
    ch = cheng()
    forced = json.load(open(FORCED, encoding="utf-8"))["map_points"] if a.forced else []
    if a.forced and os.path.exists(FORCED_MATH):
        forced += json.load(open(FORCED_MATH, encoding="utf-8"))["map_points"]
    extra = (full_points(EXEMPLAR, "exemplar") if a.exemplar else []) + \
        (full_points(THINKING, "thinking") if a.thinking else [])
    rows = map_rows(m, ch, forced, extra)
    print(f"map rows {len(rows)} ({time.time() - t0:.0f} s)")
    prow, frozen = pilot_test(settings_with_validation())
    print(f"pilot test {len(prow)} settings ({time.time() - t0:.0f} s)")
    pilot_panels, split = pilot_summary(prow)
    n_points = len({r["setting"] for r in rows})
    R = {"analysis": "reliability map by panel, exemplar contrast, pilots against R", "cheng": ch, "map_rows": rows,
         "panels": panel_summary(rows), "pilot_rows": prow, "pilot_panels": pilot_panels, "pilot_split": split,
         "frozen": frozen, "n_points": n_points, "n_panels": len({r["panel"] for r in rows})}
    if a.added:
        import added_contrasts as ac
        con = ac.load_config()
        R["map_rows"] = list(rows)
        R["added_pools"] = list(ac.pools(con))
        for pool, members in ac.pools(con).items():     # one block of keys per pool (pool base: plain keys)
            arows = added_rows(m, pool)
            want = {ac.setting_name(c) for c in members.values()}
            if {r["setting"] for r in arows} != want:
                raise SystemExit("--added: results/reliability_map/reliability_map.json does not hold the contrasts of "
                                 "config/added_contrasts.yaml; run scripts/run_reliability_map.py with --added first")
            aprow, afrozen = added_pilot_test(members, added_pred_path(pool),
                                              "map_panels_added" + ("" if pool == "base" else f"_{pool}"))
            print(f"added contrasts, pool {pool}: {len(arows)} map rows, {len(aprow)} pilot settings "
                  f"({time.time() - t0:.0f} s)")
            R["map_rows"] += arows
            R.update({pk("added_groups", pool): sorted({r["group"] for r in arows}),
                      pk("added_panels", pool): panel_summary(arows), pk("added_pilot_rows", pool): aprow,
                      pk("added_pilot_panels", pool): pilot_summary(aprow)[0], pk("added_frozen", pool): afrozen,
                      pk("added_n_points", pool): len({r["setting"] for r in arows}),
                      pk("added_tags", pool): {ac.setting_name(c): c["tag"] for c in members.values()}})
        rows = R["map_rows"]
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "map_panels.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "map_by_panel.csv"), index=False)
    pd.DataFrame(prow).to_csv(os.path.join(OUT, "tables", "pilot_risks.csv"), index=False)
    for pool in R.get("added_pools", []):
        if R.get(pk("added_pilot_rows", pool)):
            pd.DataFrame(R[pk("added_pilot_rows", pool)]).to_csv(
                os.path.join(OUT, "tables", pk("added_pilot_risks", pool) + ".csv"), index=False)
    write_summary(R)
    print(f"done in {R['runtime_seconds']} s -> results/map_panels")


if __name__ == "__main__":
    main()
