r"""The test on MATH-500, with its ceiling check, difficulty levels and secondary analyses.

Run from the repository root, after scripts/score_forced_direct.py --task math (and, for the content
secondary, scripts/score_content.py --tasks math):
    python scripts/run_math_primary.py
A dry run before the MATH data exist (MATH replaced by GSM8K-29, levels by five bins of item order; written only under
--out); its primary and ceiling slopes must equal the GSM8K values of results/outcomes:
    python scripts/run_math_primary.py --dry-run --out <scratch dir>

Primary: chain of thought - forced direct, strict, points, model-weighted, the slope of the gain on forced-direct
accuracy over the 24 discovery draws with the 95% set of the main analysis's method, on the 23 models of the GSM8K-29
readable panel (results/outcomes/outcomes.json, clean_models; no reselection on MATH). Confirmed if the set lies above
zero; refuted if it lies below zero while the ceiling check passes; inconclusive otherwise. The ceiling check is the
slope of chain-of-thought accuracy on forced accuracy, off the ceiling when its set lies above 0.6. Within each MATH
level 1-5, the primary slope, its trend over level (least-squares slope of the level slopes on the level number, with
a delete-one-draw jackknife t interval), and the three accuracies. Secondary only: the slope on every MATH panel model,
the error-reduction and logit scales, content scoring, the external index, and the slope on the items each model's
forced arm gets wrong in every discovery draw.

Outputs: results/math_primary/math_primary.json (macros, prefix mathPrim), summary.md, summary_clean.md,
tables/levels.csv and tables/per_model_levels.csv.
"""
import argparse
import json
import math
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
from src import capability_checks as cc, dataset2 as d2, instability as ins, intervals as iv  # noqa: E402
from run_outcomes import CONFIG, SELECTION  # noqa: E402
from run_index_checks import fmt, jsonable  # noqa: E402

OUT = os.path.join(ROOT, "results", "math_primary")
OUTCOMES_JSON = os.path.join(ROOT, "results", "outcomes", "outcomes.json")
CEILING_MIN = 0.6
LEVELS = ("1", "2", "3", "4", "5")
WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
STAG = {"points": "Points", "logit": "Logit", "error_reduction": "ErrRed"}


# ============================================================================ the blocks (arrays in, no files)


def slope(X, Y, method, alpha):
    m = ins.scale_moments(X, Y, method, alpha)
    m.pop("model_means", None)
    return m


def verdict(gain, ceiling):
    """Confirmed, refuted or inconclusive from the gain slope's set and the ceiling check."""
    off = ceiling["slope_set"]["kind"] == "bounded" and ceiling["slope_set"]["a"] > CEILING_MIN
    s = gain["slope_set_sign"]
    if s == "above zero":
        return "confirmed", off
    if s == "below zero" and off:
        return "refuted", off
    return "inconclusive", off


def primary_block(Bfd, Pc, method, alpha):
    gain, ceil = slope(Bfd, Pc - Bfd, method, alpha), slope(Bfd, Pc, method, alpha)
    v, off = verdict(gain, ceil)
    return {"gain": gain, "ceiling": ceil, "verdict": v, "off_ceiling": bool(off), "ceiling_min": CEILING_MIN}


def trend(level_arrays, alpha):
    """Least-squares slope of the level slopes b_l on the level number l = 1..L, sum (l - mean) b_l / sum (l - mean)^2,
    with the delete-one-draw jackknife t interval; undefined when any V_x is zero or below in the estimate or in a
    replicate. level_arrays: list of (Bfd, Pc) per level, in level order."""
    L = len(level_arrays)
    c = np.arange(1, L + 1) - (L + 1) / 2
    c = c / float(c @ c)
    est, loo = [], []
    for B, P in level_arrays:
        cx, ca = iv.CrossMoment(B, B), iv.CrossMoment(P - B, B)
        vx, lx = cx.estimate(), cx.loo()
        if vx <= 0 or np.any(lx <= 0):
            return {"estimate": None, "ci95": None, "defined": False,
                    "note": "a level's V_x is zero or below in the estimate or a jackknife replicate"}
        est.append(ca.estimate() / vx)
        loo.append(ca.loo() / lx)
    t = float(c @ np.array(est))
    tl = c @ np.vstack(loo)
    lo, hi = iv.jack_t(t, tl, alpha)
    return {"estimate": t, "ci95": [float(lo), float(hi)], "defined": True, "level_slopes": est,
            "rising": bool(lo > 0), "sign": ins.interval_sign("bounded", lo, hi)}


def wrong_items_block(fd_rows, cot_rows, units, models, Bfd, method, alpha):
    """Contamination check: for each model, the items its forced-direct answer gets wrong in every listed
    draw; the gain is its chain-of-thought accuracy on those items (its forced accuracy there is zero), regressed on its
    forced-direct accuracy over all items (Bfd, columns in `models` order)."""
    units = [str(u) for u in units]
    f = fd_rows[fd_rows["config_id"].astype(str).isin(units)]
    wrong = {k: set(g.groupby("item_id")["strict_correct"].max().pipe(lambda s: s[s == 0]).index)
             for k, g in f.groupby("model_key")}
    Y = np.full((len(units), len(models)), np.nan)
    n_items = {}
    for j, k in enumerate(models):
        w = wrong.get(k, set())
        n_items[k] = len(w)
        if not w:
            continue
        g = cot_rows[(cot_rows["model_key"] == k) & cot_rows["item_id"].isin(w)]
        acc = 100.0 * g.groupby(g["config_id"].astype(str))["strict_correct"].mean()
        Y[:, j] = [acc.get(u, np.nan) for u in units]
    keep = [j for j, k in enumerate(models) if n_items[k] > 0 and np.isfinite(Y[:, j]).all()]
    out = {"items_per_model": n_items, "models_used": [models[j] for j in keep],
           "models_without_items": [k for k in models if n_items[k] == 0],
           "gain_model_means": {k: float(np.nanmean(Y[:, j])) if n_items[k] else None for j, k in enumerate(models)},
           "gain_by_draw": Y.tolist()}
    out["slope"] = slope(Bfd[:, keep], Y[:, keep], method, alpha) if len(keep) >= 3 else None
    return out


def level_table(levels, models):
    """Per level the mean accuracies over models and draws, and the per-model rows."""
    rows, per = [], []
    for lev, x in levels.items():
        Breq, Bfd, Pc = x["arrays"]
        rows.append({"level": lev, "n_items": x["n_items"], "forced_direct": float(Bfd.mean()),
                     "requested_direct": float(Breq.mean()), "chain_of_thought": float(Pc.mean()),
                     "gain": float((Pc - Bfd).mean()), "slope": x["slope"]["slope"],
                     "set_kind": x["slope"]["slope_set"]["kind"], "lo": x["slope"]["slope_set"]["a"],
                     "hi": x["slope"]["slope_set"]["b"], "set_sign": x["slope"]["slope_set_sign"]})
        for j, k in enumerate(models):
            per.append({"level": lev, "model_key": k, "forced_direct": float(Bfd[:, j].mean()),
                        "requested_direct": float(Breq[:, j].mean()), "chain_of_thought": float(Pc[:, j].mean())})
    return rows, per


# ============================================================================ inputs


def readable_models():
    """The 23 models of the GSM8K-29 readable panel (results/outcomes/outcomes.json, strict, clean_models)."""
    o = json.load(open(OUTCOMES_JSON, encoding="utf-8"))
    return list(o["tasks"]["gsm8k"]["clean_models"])


def inputs(dry_run):
    from run_forced_direct import collection_inputs, forced_inputs
    task = "gsm8k" if dry_run else "math"
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    if dry_run:
        # five bins of item order stand in for the levels
        lv = {str(it): str(1 + min(4, i * 5 // len(items))) for i, it in enumerate(items)}
    else:
        from src import math_d2
        t = math_d2.load_items("target")[0]
        lv = dict(zip(t["item_id"].astype(str), t["level"].astype(str)))
        if sorted(set(lv.values())) != list(LEVELS):
            raise SystemExit(f"MATH levels are {sorted(set(lv.values()))}, expected 1-5")
    return {"task": task, "inp": inp, "sampler": s, "items": list(items), "panel": panel, "fd_load": fd_load,
            "levels": lv}


def subset(load, keep):
    keep = set(keep)

    def f(ids):
        df = load(ids)
        return df[df["item_id"].isin(keep)]
    return f


# ============================================================================ run


def run(a):
    t0 = time.time()
    from run_forced_direct import three_arm_arrays
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    x = inputs(a.dry_run)
    inp, panel, items = x["inp"], x["panel"], x["items"]
    disc = list(x["sampler"]["discovery_config_ids"])
    want = readable_models()
    if len(want) != 23:
        raise SystemExit(f"results/outcomes lists {len(want)} readable GSM8K models, expected 23")
    models = [k for k in want if k in panel]
    missing = [k for k in want if k not in panel]
    ix = [panel.index(k) for k in models]
    Breq, Bfd, Pc = three_arm_arrays(inp["load"], x["fd_load"], disc, panel, items)
    n = len(items)
    print(f"{x['task']}: arrays ({time.time() - t0:.0f} s)", flush=True)
    R = {"analysis": "the test on MATH-500", "dry_run": bool(a.dry_run),
         "stand_in": "MATH replaced by GSM8K-29; levels by five bins of item order" if a.dry_run else None,
         "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method, "alpha": alpha,
         "units": "pp; slopes in pp per pp", "K_discovery": len(disc), "n_items": n,
         "models": models, "M": len(models), "missing_from_math": missing, "math_panel": panel}
    R["primary"] = primary_block(Bfd[:, ix], Pc[:, ix], method, alpha)
    ref = json.load(open(OUTCOMES_JSON, encoding="utf-8"))["tasks"]["gsm8k"]["outcomes"]["CotMinusForced"]
    R["gsm8k_reference"] = ref["clean"]["slopes"]["scales"]["points"]["baseline"]

    # ------------------------------------------------------------ difficulty levels
    lv = x["levels"]
    levels = {}
    for lev in LEVELS:
        its = [i for i in items if lv.get(str(i)) == lev]
        arr = three_arm_arrays(subset(inp["load"], its), subset(x["fd_load"], its), disc, panel, its)
        arr = tuple(A[:, ix] for A in arr)
        levels[lev] = {"n_items": len(its), "arrays": arr, "slope": slope(arr[1], arr[2] - arr[1], method, alpha)}
    R["levels"] = {lev: {"n_items": v["n_items"], "slope": v["slope"]} for lev, v in levels.items()}
    R["trend"] = trend([(v["arrays"][1], v["arrays"][2]) for v in levels.values()], alpha)
    lrows, per = level_table(levels, models)
    R["level_accuracies"] = lrows
    print(f"levels ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ secondary only
    sec = {"all_models": slope(Bfd, Pc - Bfd, method, alpha) | {"M": len(panel)}}
    for sc in ("error_reduction", "logit"):
        s = ins.scale_scores(Bfd[:, ix], Pc[:, ix], sc, n)
        sec[sc] = slope(s["x"], s["y"], method, alpha) | {"replaced_cells": s["n_replaced"]}
    from run_transfer_tasks import eci_for
    e = eci_for(panel)
    sec["index"] = slope(cc.tile([e[i] for i in ix], len(disc)), (Pc - Bfd)[:, ix], method, alpha)
    task = x["task"]
    cpath = os.path.join(ROOT, "data", "processed", "content", f"{task}_content.parquet")
    if os.path.exists(cpath):
        from run_outcomes import with_content
        _, Cfd, Cc = three_arm_arrays(with_content(inp["load"], task), with_content(x["fd_load"], task), disc, panel,
                                      items, value="content_correct")
        sec["content"] = slope(Cfd[:, ix], (Cc - Cfd)[:, ix], method, alpha)
    else:
        sec["content"] = None
        sec["content_note"] = f"{os.path.relpath(cpath, ROOT)} does not exist yet"
    col = inp["load"](disc)
    sec["forced_wrong_items"] = wrong_items_block(x["fd_load"](disc), col[col["arm"] == "zero_shot_cot"], disc, models,
                                                  Bfd[:, ix], method, alpha)
    R["secondary"] = sec
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    with open(os.path.join(a.out, "math_primary.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    txt = "\n".join(summary(R)) + "\n"
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)
    pd.DataFrame(lrows).to_csv(os.path.join(a.out, "tables", "levels.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(a.out, "tables", "per_model_levels.csv"), index=False)
    print(f"done in {R['runtime_seconds']} s -> {os.path.relpath(a.out, ROOT)}", flush=True)
    return R


# ============================================================================ macros and summary


def _set(m, stem, x, nd=3):
    m[stem] = fmt(x["slope"], nd)
    s = x["slope_set"]
    if s["kind"] == "bounded":
        m[stem + "Lo"], m[stem + "Hi"] = fmt(s["a"], nd), fmt(s["b"], nd)
    else:
        m[stem + "Lo"] = m[stem + "Hi"] = s["kind"]


def macros(R):
    t = "mathPrim"
    m = {t + "M": str(R["M"]), t + "Missing": str(len(R["missing_from_math"])), t + "K": str(R["K_discovery"])}
    p = R["primary"]
    _set(m, t + "Slope", p["gain"])
    _set(m, t + "CeilSlope", p["ceiling"])
    m[t + "Verdict"], m[t + "OffCeiling"] = p["verdict"], "yes" if p["off_ceiling"] else "no"
    _set(m, t + "GsmSlope", R["gsm8k_reference"])
    for lev, v in R["levels"].items():
        w = t + "Level" + WORDS[int(lev)]
        _set(m, w + "Slope", v["slope"])
        m[w + "N"] = str(v["n_items"])
    for r in R["level_accuracies"]:
        w = t + "Level" + WORDS[int(r["level"])]
        m[w + "Forced"], m[w + "Req"], m[w + "Cot"] = (fmt(r["forced_direct"], 1), fmt(r["requested_direct"], 1),
                                                     fmt(r["chain_of_thought"], 1))
    tr = R["trend"]
    if tr["defined"]:
        m[t + "Trend"], m[t + "TrendLo"], m[t + "TrendHi"] = (fmt(tr["estimate"], 3), fmt(tr["ci95"][0], 3),
                                                            fmt(tr["ci95"][1], 3))
        m[t + "TrendRising"] = "yes" if tr["rising"] else "no"
    else:
        m[t + "Trend"] = m[t + "TrendLo"] = m[t + "TrendHi"] = m[t + "TrendRising"] = "undefined"
    s = R["secondary"]
    _set(m, t + "AllSlope", s["all_models"])
    m[t + "AllM"] = str(s["all_models"]["M"])
    _set(m, t + "ErrRedSlope", s["error_reduction"])
    _set(m, t + "LogitSlope", s["logit"])
    _set(m, t + "IndexSlope", s["index"])
    if s["content"] is not None:
        _set(m, t + "ContentSlope", s["content"])
    w = s["forced_wrong_items"]
    if w["slope"] is not None:
        _set(m, t + "WrongSlope", w["slope"])
    m[t + "WrongM"] = str(len(w["models_used"]))
    cnt = [v for v in w["items_per_model"].values()]
    m[t + "WrongItemsMedian"] = fmt(float(np.median(cnt)), 0) if cnt else "--"
    for k in list(m):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    return {k: str(v).replace("−", "$-$") for k, v in m.items()}


def _st(x):
    s = x["slope_set"]
    if s["kind"] != "bounded":
        return f"{fmt(x['slope'], 3)} (set {s['kind']})"
    return f"{fmt(x['slope'], 3)} [{fmt(s['a'], 3)}, {fmt(s['b'], 3)}]"


def summary(R):
    p = R["primary"]
    g = R["gsm8k_reference"]
    L = ["# The test on MATH-500" + (" (dry run)" if R["dry_run"] else ""), ""]
    if R["dry_run"]:
        L += [f"Dry run: {R['stand_in']}.", ""]
    L += ["Chain of thought − forced direct, "
          f"strict scoring, points, model-weighted, on the 24 discovery draws; 95% sets by the {R['slope_method']} "
          f"method. Models: the {R['M']} of the GSM8K-29 readable panel present in the MATH panel"
          + (f" ({', '.join(R['missing_from_math'])} missing)" if R["missing_from_math"] else "") + ".", "",
          "## Primary", "",
          f"- Slope of the gain on forced-direct accuracy: {_st(p['gain'])} pp per pp (GSM8K: {_st(g)}).",
          f"- Ceiling check, slope of chain-of-thought accuracy on forced accuracy: {_st(p['ceiling'])}; off the "
          f"ceiling when the set lies above {CEILING_MIN}: {'yes' if p['off_ceiling'] else 'no'}.",
          f"- Reading: **{p['verdict']}**.", "",
          "## Difficulty levels", "",
          "| Level | Items | Forced direct | Requested direct | Chain of thought | Gain | Slope of gain on forced |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in R["level_accuracies"]:
        x = R["levels"][r["level"]]["slope"]
        L.append(f"| {r['level']} | {r['n_items']} | {fmt(r['forced_direct'], 1)} | {fmt(r['requested_direct'], 1)} | "
                 f"{fmt(r['chain_of_thought'], 1)} | {fmt(r['gain'], 1)} | {_st(x)} |")
    tr = R["trend"]
    L += ["", ("Trend of the level slopes over level: "
               + (f"{fmt(tr['estimate'], 3)} [{fmt(tr['ci95'][0], 3)}, {fmt(tr['ci95'][1], 3)}] per level (delete-one-"
                  f"draw jackknife); rising: {'yes' if tr['rising'] else 'no'}." if tr["defined"]
                  else f"undefined ({tr['note']}).")), ""]
    s = R["secondary"]
    w = s["forced_wrong_items"]
    L += ["## Secondary only", "",
          f"- Every MATH panel model ({s['all_models']['M']}): {_st(s['all_models'])}.",
          f"- Error-reduction scale: {_st(s['error_reduction'])}; logit scale: {_st(s['logit'])}.",
          f"- External capability index: {_st(s['index'])} pp per index point.",
          "- Content scoring: " + (_st(s["content"]) if s["content"] is not None else s.get("content_note", "--")) + ".",
          "- Items the forced arm gets wrong in every discovery draw: "
          + (_st(w["slope"]) if w["slope"] is not None else "fewer than three models with such items")
          + f" ({len(w['models_used'])} models; median {fmt(float(np.median(list(w['items_per_model'].values()))), 0)} "
            "items per model).", ""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("a dry run needs --out outside results/math_primary")
    run(a)


if __name__ == "__main__":
    main()
