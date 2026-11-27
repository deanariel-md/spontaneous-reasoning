r"""The specification analysis.

PowerShell (from the repository root), after scripts/score_content.py:
    cmd /c ".venv\Scripts\python.exe scripts\run_specification.py 2>&1"

A specification is one choice per axis: comparison arm (requested direct: CotMinusReq; forced direct: CotMinusForced),
scoring (strict, lenient, content), scale (points, logits, error reduction), weighting (models; family means, each
family one point; family-weighted, every model with each family the same total weight),
panel (full, clean, each family's size ladder) and capability (baseline, index, log2 parameters within families).
Per specification on the 24 discovery draws: the slope with its 95% set and the standardized slope
A_x / sqrt(V_x V_delta). Then the pair-ordering agreement within and between benefit definitions, and the
decomposition of the standardized slope and of the set's sign across the axes of the balanced core. Outputs:
results/specification/specification.json (macros, prefix spec), summary.md, summary_clean.md, and
results/figures/spec_curve.csv (one row per specification). Tasks: GSM8K-29 and BIG-Bench Hard, and MATH-500 once its
content scores exist (scripts/run_outcomes.content_tasks).
"""
import itertools
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
from src import capability_checks as cc, dataset2 as d2, instability as ins  # noqa: E402
from run_outcomes import (CLEAN_MIN, CONFIG, FAMILY_MIN, FD, OUTCOMES, SCORINGS, SELECTION,  # noqa: E402
                          content_tasks, family_means, pair, with_content)
from run_index_checks import fmt, write  # noqa: E402

ARM_OUTCOME = {"requested": "CotMinusReq", "forced": "CotMinusForced"}
CAPS = ("baseline", "index")
PANELS = ("full", "family_means", "family_weighted", "clean")
WEIGHTINGS = ("family_means", "family_weighted")
FIG = os.path.join(ROOT, "results", "figures", "spec_curve.csv")
SIGN = {"above zero": 1, "below zero": -1}


def load(task, method):
    """Discovery arrays of the three arms under each scoring, and the panel facts."""
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    from run_transfer_tasks import eci_for
    from src import bbh_d2
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    params = {e["repo"]: e["params_b"] for e in man["panel"]}
    reg = bbh_d2.load_registries()
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    units = list(s["discovery_config_ids"])
    D = {}
    for scoring, (value, _, _) in SCORINGS.items():
        cl, fl = (inp["load"], fd_load) if scoring != "content" else (with_content(inp["load"], task),
                                                                      with_content(fd_load, task))
        D[scoring] = three_arm_arrays(cl, fl, units, panel, items, value=value)
    pm = pd.read_csv(os.path.join(FD, "tables", f"per_model_{task}.csv")).set_index("model_key")
    return {"panel": panel, "n": len(items), "D": D, "eci": eci_for(panel), "families": list(inp["families"]),
            "clean": [bool(pm.loc[k, "closed_and_readable_pct"] >= CLEAN_MIN) for k in panel],
            "log2_params": [float(np.log2(params[reg[k]["hf_id"]])) for k in panel], "K": len(units)}


def spec_row(X, Y, method, alpha, w=None):
    m = ins.scale_moments(X, Y, method, alpha) if w is None else ins.weighted_moments(X, Y, w, method, alpha)
    vx, vy, a = m["V_x"], m["V_y"], m["A"]
    std = a / np.sqrt(vx * vy) if vx > 0 and vy > 0 else None
    s = m["slope_set"]
    return {"slope": m["slope"], "set_kind": s["kind"], "lo": s["a"], "hi": s["b"], "set_sign": m["slope_set_sign"],
            "std_slope": std, "V_x": vx, "V_y": vy, "A": a, "R_benefit": m.get("R_benefit"), "M": m["M"]}


def specifications(task, x, method, alpha):
    rows = []
    K, n = x["K"], x["n"]
    fam_order = sorted(set(x["families"]))
    ef = [float(np.mean([e for e, f in zip(x["eci"], x["families"]) if f == fam])) for fam in fam_order]
    clean = [i for i, c in enumerate(x["clean"]) if c]
    fw = ins.family_weights(x["families"])
    for scoring, arm in itertools.product(SCORINGS, ARM_OUTCOME):
        B, P = pair(x["D"][scoring], ARM_OUTCOME[arm])
        views = {"full": (B, P, x["eci"], None, None),
                 "clean": (B[:, clean], P[:, clean], [x["eci"][i] for i in clean], None, None),
                 "family_means": (family_means(B, x["families"], fam_order),
                                  family_means(P, x["families"], fam_order), ef, None, None),
                 "family_weighted": (B, P, x["eci"], None, fw)}
        for f in fam_order:
            ix = [i for i, y in enumerate(x["families"]) if y == f]
            if len(ix) >= FAMILY_MIN:
                views[f"family:{f}"] = (B[:, ix], P[:, ix], [x["eci"][i] for i in ix], [x["log2_params"][i] for i in ix],
                                        None)
        for view, (b, p, e, lp, w) in views.items():
            for scale in ins.SCALES:
                sc = ins.scale_scores(b, p, scale, n)
                caps = {"baseline": sc["x"], "index": cc.tile(e, K)}
                if lp is not None:
                    caps["log2_params"] = cc.tile(lp, K)
                for cap, X in caps.items():
                    r = spec_row(X, sc["y"], method, alpha, w)
                    rows.append({"task": task, "outcome": ARM_OUTCOME[arm], "arm": arm, "scoring": scoring,
                                 "scale": scale, "weighting": view if view in WEIGHTINGS else "model",
                                 "panel": view if view.startswith("family:") or view == "clean" else "full",
                                 "capability": cap, "replaced_cells": sc["n_replaced"], **r})
    return rows


def dove_rows():
    d = json.load(open(os.path.join(ROOT, "results", "instability", "instability.json"), encoding="utf-8"))["scale"]["dove"]
    rows = []
    for scale in ins.SCALES:
        m = d[scale]["discovery"]
        vx, vy, a = m["V_x"], m["V_y"], m["A"]
        s = m["slope_set"]
        rows.append({"task": "dove_mmlu", "outcome": "FiveShotMinusZeroShot", "arm": "zero_shot", "scoring": "strict",
                     "scale": scale, "weighting": "model", "panel": "full", "capability": "baseline",
                     "replaced_cells": None, "slope": m["slope"], "set_kind": s["kind"], "lo": s["a"], "hi": s["b"],
                     "set_sign": m["slope_set_sign"], "std_slope": a / np.sqrt(vx * vy) if vx > 0 and vy > 0 else None,
                     "V_x": vx, "V_y": vy, "A": a, "R_benefit": m["R_benefit"], "M": m["M"],
                     "slope_method": m["slope_method"]})
    return rows


# ---------------------------------------------------------------- pair-ordering agreement


def benefit(x, scoring, arm, scale):
    B, P = pair(x["D"][scoring], ARM_OUTCOME[arm])
    return ins.scale_scores(B, P, scale, x["n"])["y"]


def c_within(Y):
    same, tie = ins.pair_matrices_draws(Y)
    i, j = np.triu_indices(Y.shape[1], 1)
    return float(same[i, j].mean() + 0.5 * tie[i, j].mean())


def c_between(Y1, Y2):
    i, j = np.triu_indices(Y1.shape[1], 1)
    vals = []
    for r in range(Y1.shape[0]):
        same, tie = ins.pair_matrices_between(Y1[r], Y2[r])
        vals.append(same[i, j].mean() + 0.5 * tie[i, j].mean())
    return float(np.mean(vals))


def agreement(x):
    defs = list(itertools.product(SCORINGS, ARM_OUTCOME, ins.SCALES))
    Y = {d: benefit(x, *d) for d in defs}
    within = {"/".join(d): c_within(Y[d]) for d in defs}
    groups = {}
    for d1, d2_ in itertools.combinations(defs, 2):
        axes = tuple(a for a, u, v in zip(("scoring", "arm", "scale"), d1, d2_) if u != v)
        groups.setdefault("+".join(axes), []).append({"a": "/".join(d1), "b": "/".join(d2_),
                                                      "C_between": c_between(Y[d1], Y[d2_]),
                                                      "C_within_mean": 0.5 * (within["/".join(d1)]
                                                                              + within["/".join(d2_)])})
    summ = {g: {"pairs": len(v), "C_between_mean": float(np.mean([r["C_between"] for r in v])),
                "C_between_min": float(np.min([r["C_between"] for r in v])),
                "C_between_max": float(np.max([r["C_between"] for r in v])),
                "C_within_mean": float(np.mean([r["C_within_mean"] for r in v]))} for g, v in groups.items()}
    return {"definitions": len(defs), "within": within, "within_mean": float(np.mean(list(within.values()))),
            "within_min": float(np.min(list(within.values()))), "groups": summ, "pairs": groups}


# ---------------------------------------------------------------- decomposition


def decompose(core, value):
    """Main-effect shares of the sum of squares of `value` across the axes of a balanced core (DataFrame)."""
    axes = ("arm", "scoring", "scale", "capability", "panel_weighting")
    y = core[value].astype(float)
    if y.isna().any():
        return {"defined": False, "undefined_rows": int(y.isna().sum())}
    tot = float(((y - y.mean()) ** 2).sum())
    out = {"defined": True, "n": int(len(y)), "total_ss": tot, "mean": float(y.mean()), "main": {}}
    for a in axes:
        g = core.groupby(a)[value]
        ss = float((g.size() * (g.mean() - y.mean()) ** 2).sum())
        out["main"][a] = {"share": ss / tot if tot > 0 else None,
                          "level_means": {str(k): float(v) for k, v in g.mean().items()}}
    out["rest_share"] = (1 - sum(v["share"] for v in out["main"].values())) if tot > 0 else None
    return out


def core_table(rows, task):
    df = pd.DataFrame([r for r in rows if r["task"] == task and r["capability"] in CAPS
                       and (r["panel"] in ("full", "clean"))])
    df["panel_weighting"] = np.where(df["weighting"].isin(WEIGHTINGS), df["weighting"], df["panel"])
    df["sign"] = df["set_sign"].map(SIGN).fillna(0)
    n_expected = 2 * 3 * 3 * 2 * 4         # arm, scoring, scale, capability; full, clean, family means, family-weighted
    if len(df) != n_expected:
        raise SystemExit(f"{task}: the balanced core has {len(df)} rows, expected {n_expected}")
    return df


def run():
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    R = {"analysis": "specification analysis", "methods": "scripts/run_specification.py",
         "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method, "tasks": {}}
    rows = []
    for task in content_tasks():
        x = load(task, method)
        rows += [{**r, "slope_method": method} for r in specifications(task, x, method, alpha)]
        core = core_table(rows, task)
        R["tasks"][task] = {"M": len(x["panel"]), "clean_M": int(sum(x["clean"])),
                            "agreement": agreement(x),
                            "decomposition": {"std_slope": decompose(core, "std_slope"),
                                              "sign": decompose(core, "sign")},
                            "core_signs": core["set_sign"].value_counts().to_dict()}
        print(f"{task}: {sum(r['task'] == task for r in rows)} specifications ({time.time() - t0:.0f} s)", flush=True)
    rows += dove_rows()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    df.to_csv(FIG, index=False)
    R["n_specifications"] = int(len(df))
    R["counts"] = counts(df)
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write("specification", R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/specification, {os.path.relpath(FIG, ROOT)}", flush=True)


def counts(df):
    out = {}
    for task, g in df.groupby("task"):
        c = {"all": g["set_sign"].value_counts().to_dict()}
        for axis in ("arm", "scoring", "scale", "weighting", "capability"):
            c[axis] = {str(k): v["set_sign"].value_counts().to_dict() for k, v in g.groupby(axis)}
        c["panel"] = {k: v["set_sign"].value_counts().to_dict()
                      for k, v in g.assign(p=g["panel"].where(~g["panel"].str.startswith("family:"), "within_family"))
                      .groupby("p")}
        out[task] = c
    return out


TT = {"gsm8k": "Gsm", "bbh": "Bbh", "math": "Math", "dove_mmlu": "Dove"}
AX = {"arm": "Arm", "scoring": "Scoring", "scale": "Scale", "capability": "Capability", "panel_weighting": "Panel"}


def macros(R):
    m = {"specN": str(R["n_specifications"])}
    for task, c in R["counts"].items():
        t = TT[task]
        a = c["all"]
        m[f"spec{t}N"] = str(sum(a.values()))
        m[f"spec{t}Pos"], m[f"spec{t}Neg"] = str(a.get("above zero", 0)), str(a.get("below zero", 0))
        m[f"spec{t}Zero"] = str(a.get("includes zero", 0))
        m[f"spec{t}Other"] = str(sum(v for k, v in a.items() if k not in ("above zero", "below zero", "includes zero")))
        for axis, lv in c.items():
            if axis == "all":
                continue
            for lev, cnt in lv.items():
                tag = "".join(w.capitalize() for w in str(lev).replace(":", "_").replace("2", "").split("_"))
                m[f"spec{t}{axis.capitalize()}{tag}Pos"] = str(cnt.get("above zero", 0))
                m[f"spec{t}{axis.capitalize()}{tag}Neg"] = str(cnt.get("below zero", 0))
    for task, T in R["tasks"].items():
        t = TT[task]
        ag = T["agreement"]
        m[f"spec{t}CWithinMean"], m[f"spec{t}CWithinMin"] = fmt(ag["within_mean"], 3), fmt(ag["within_min"], 3)
        for g, v in ag["groups"].items():
            tag = "".join(w.capitalize() for w in g.split("+"))
            m[f"spec{t}CBetween{tag}"] = fmt(v["C_between_mean"], 3)
            m[f"spec{t}CBetween{tag}Min"] = fmt(v["C_between_min"], 3)
        # the plain specification: strict scoring in points, the direct arm switched
        x = next(r for r in ag["pairs"]["arm"] if r["a"] == "strict/requested/points" and r["b"] == "strict/forced/points")
        m[f"spec{t}CArmStrictPoints"] = fmt(x["C_between"], 3)
        m[f"spec{t}ArmStrictPointsReversed"] = fmt(100 * (1 - x["C_between"]), 0)
        for arm, at in (("requested", "Req"), ("forced", "Forced")):
            m[f"spec{t}CWithinStrictPoints{at}"] = fmt(ag["within"][f"strict/{arm}/points"], 3)
        # signs of the sets over the balanced core (full or readable panel; baseline or index capability)
        cs = T["core_signs"]
        m[f"spec{t}CorePos"], m[f"spec{t}CoreNeg"] = str(cs.get("above zero", 0)), str(cs.get("below zero", 0))
        m[f"spec{t}CoreZero"] = str(cs.get("includes zero", 0))
        m[f"spec{t}CoreOther"] = str(sum(v for k, v in cs.items() if k not in ("above zero", "below zero", "includes zero")))
        for q, qt in (("std_slope", "Std"), ("sign", "Sign")):
            dcm = T["decomposition"][q]
            if not dcm["defined"]:
                continue
            for axis, v in dcm["main"].items():
                m[f"spec{t}Share{qt}{AX[axis]}"] = fmt(100 * v["share"], 0)
            m[f"spec{t}Share{qt}Rest"] = fmt(100 * dcm["rest_share"], 0)
    return m


def summary(R):
    L = ["# Specification analysis", "",
         "Each specification is one choice of comparison arm, "
         "scoring, scale, weighting, panel and capability; slopes of gain on capability over the 24 discovery draws "
         f"with 95% sets ({R['slope_method']}); {R['n_specifications']} specifications in "
         "`results/figures/spec_curve.csv`.", "",
         "## Where the 95% sets lie", "", "| Task | Specifications | Above zero | Below zero | Include zero | Other |",
         "|---|---:|---:|---:|---:|---:|"]
    for task, c in R["counts"].items():
        a = c["all"]
        L.append(f"| {task} | {sum(a.values())} | {a.get('above zero', 0)} | {a.get('below zero', 0)} | "
                 f"{a.get('includes zero', 0)} | "
                 f"{sum(v for k, v in a.items() if k not in ('above zero', 'below zero', 'includes zero'))} |")
    for task, c in R["counts"].items():
        if task == "dove_mmlu":
            continue
        L += ["", f"### {task}: sets above / below zero by axis level", "", "| Axis | Level | Above | Below | Other |",
              "|---|---|---:|---:|---:|"]
        for axis, lv in c.items():
            if axis == "all":
                continue
            for lev, cnt in lv.items():
                L.append(f"| {axis} | {lev} | {cnt.get('above zero', 0)} | {cnt.get('below zero', 0)} | "
                         f"{sum(v for k, v in cnt.items() if k not in ('above zero', 'below zero'))} |")
    for task, T in R["tasks"].items():
        ag = T["agreement"]
        L += ["", f"## {task}: pair-ordering agreement of the benefit", "",
              f"{ag['definitions']} benefit definitions (arm × scoring × scale; full panel, model weighting). Within a "
              f"definition, across distinct draws: mean C {fmt(ag['within_mean'], 3)} (lowest "
              f"{fmt(ag['within_min'], 3)}). Between definitions on the same draw, grouped by the axes that differ:", "",
              "| Axes that differ | Pairs | C between (mean) | Lowest | Highest | C within (mean of the two) |",
              "|---|---:|---:|---:|---:|---:|"]
        for g, v in sorted(ag["groups"].items(), key=lambda kv: (kv[0].count("+"), kv[0])):
            L.append(f"| {g} | {v['pairs']} | {fmt(v['C_between_mean'], 3)} | {fmt(v['C_between_min'], 3)} | "
                     f"{fmt(v['C_between_max'], 3)} | {fmt(v['C_within_mean'], 3)} |")
        L += ["", f"## {task}: decomposition over the balanced core ({sum(T['core_signs'].values())} specifications)", "",
              "| Quantity | Mean | Arm | Scoring | Scale | Capability | Panel and weighting | Interactions |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for q, qn in (("std_slope", "standardized slope"), ("sign", "sign of the set")):
            dcm = T["decomposition"][q]
            if not dcm["defined"]:
                L.append(f"| {qn} | undefined in {dcm['undefined_rows']} rows | | | | | | |")
                continue
            L.append(f"| {qn} | {fmt(dcm['mean'], 3)} | " + " | ".join(
                f"{fmt(100 * dcm['main'][a]['share'], 0)}%" for a in AX) + f" | {fmt(100 * dcm['rest_share'], 0)}% |")
        L += ["", "Level means of the standardized slope:", ""]
        dcm = T["decomposition"]["std_slope"]
        if dcm["defined"]:
            for a, v in dcm["main"].items():
                L.append(f"- {a}: " + "; ".join(f"{k} {fmt(x, 3)}" for k, x in v["level_means"].items()))
    L.append("")
    return L


if __name__ == "__main__":
    run()
