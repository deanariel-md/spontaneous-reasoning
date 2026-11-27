r"""The prediction for MATH-500 from the GSM8K curve, written before any MATH output is read.

Run from the repository root, after scripts/run_forced_direct.py --tasks math and the MATH screen:
    python scripts/run_math_prediction.py
A dry run before the data exist (MATH replaced by GSM8K-29; written only under --out):
    python scripts/run_math_prediction.py --dry-run --out <scratch dir>

The prediction: if the floor-ceiling account is right, the strong models move from the descending limb of the GSM8K
curve to its rising part on MATH, so the capability slope turns positive among them, and requested and forced direct
still disagree for the models that reason when asked not to. For each outcome (chain of thought - forced direct, and
chain of thought - requested direct), on the 24 discovery draws of each task:
- the GSM8K curve is the least-squares quadratic of each model's mean gain on its mean baseline across GSM8K-29; when
  it is concave, the descending limb holds the models whose GSM8K baseline lies above the vertex, and otherwise the
  limb is undefined;
- the statistic is the slope of MATH gain on MATH baseline among the limb's models in MATH's readable panel (95% of
  forced outputs closed and readable, results/forced_direct/tables/per_model_math.csv), on three scales, with the 95%
  set of the main analysis's method; the prediction holds on a scale when the set lies above zero; fewer than three
  models give no slope;
- each model's shift is its MATH (baseline, gain) minus its GSM8K one, and its residual is its MATH gain minus the
  GSM8K curve at its MATH baseline.
The models that reason when asked not to are those the direct-answer model rule of config/admissibility.yaml flags
on the MATH screen (model_diagnostics of the exclusion log); for each, requested - forced direct on MATH with a
leave-one-draw-out jackknife t interval over the discovery draws, and the prediction holds when it excludes zero.

Outputs: results/math_prediction/math_prediction.json (macros, prefix mathPred), summary.md, summary_clean.md, and
results/figures/math_shift.csv (one row per model and outcome).
"""
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
from src import dataset2 as d2, instability as ins, intervals as iv  # noqa: E402
from run_outcomes import CLEAN_MIN, CONFIG, FD, OUTCOMES, SELECTION, pair  # noqa: E402
from run_index_checks import fmt, jsonable  # noqa: E402

OUT = os.path.join(ROOT, "results", "math_prediction")
FIG = os.path.join(ROOT, "results", "figures", "math_shift.csv")
CHECKED = ("CotMinusForced", "CotMinusReq")
MIN_MODELS = 3
TAG = {"CotMinusForced": "Forced", "CotMinusReq": "Req"}
STAG = {"points": "Points", "logit": "Logit", "error_reduction": "ErrRed"}


# ============================================================================ the blocks (arrays in, no files)


def gsm8k_curve(x, y):
    """Least-squares quadratic y = a + b x + c x^2 over the models; its vertex when concave."""
    c, b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 2)
    concave = bool(c < 0)
    return {"a": float(a), "b": float(b), "c": float(c), "concave": concave,
            "vertex": float(-b / (2 * c)) if concave else None}


def curve_at(fit, x):
    return fit["a"] + fit["b"] * np.asarray(x, float) + fit["c"] * np.asarray(x, float) ** 2


def slope_on(B, P, n, method, alpha):
    """The slope of gain on baseline on the three scales, with the 95% set (src/instability.scale_moments)."""
    out = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(B, P, sc, n)
        m = ins.scale_moments(s["x"], s["y"], method, alpha)
        out[sc] = {"slope": m["slope"], "slope_set": m["slope_set"], "set_sign": m["slope_set_sign"],
                   "replaced_cells": s["n_replaced"]}
    return out


def limb_block(gB, gP, mB, mP, gpanel, mpanel, mclean, n_g, n_m, method, alpha):
    """One outcome: the GSM8K curve, the limb, the MATH slope among the limb's readable models, the per-model rows."""
    gx, gy = gB.mean(0), (gP - gB).mean(0)
    fit = gsm8k_curve(gx, gy)
    limb = [k for k, x in zip(gpanel, gx) if fit["concave"] and x > fit["vertex"]]
    use = [k for k in limb if k in mpanel and mclean.get(k, False)]
    r = {"gsm8k_curve": fit, "limb_models": limb, "limb_readable_on_math": use,
         "limb_not_readable_on_math": [k for k in limb if k in mpanel and not mclean.get(k, False)],
         "limb_not_in_math_panel": [k for k in limb if k not in mpanel]}
    if not fit["concave"]:
        r["note"] = "the GSM8K curve is not concave, so the descending limb is undefined"
    if len(use) >= MIN_MODELS:
        im, ig = [mpanel.index(k) for k in use], [gpanel.index(k) for k in use]
        r["math_slopes"] = slope_on(mB[:, im], mP[:, im], n_m, method, alpha)
        r["gsm8k_slopes_same_models"] = slope_on(gB[:, ig], gP[:, ig], n_g, method, alpha)
        r["holds"] = {sc: v["set_sign"] == "above zero" for sc, v in r["math_slopes"].items()}
    else:
        r["math_slopes"] = r["gsm8k_slopes_same_models"] = r["holds"] = None
        r.setdefault("note", f"{len(use)} limb models are readable on MATH; a slope needs {MIN_MODELS}")
    mx, my = mB.mean(0), (mP - mB).mean(0)
    rows = []
    for j, k in enumerate(mpanel):
        g = gpanel.index(k) if k in gpanel else None
        row = {"model_key": k, "math_baseline": float(mx[j]), "math_gain": float(my[j]),
               "math_readable": bool(mclean.get(k, False)), "on_descending_limb": k in limb,
               "gsm8k_curve_at_math_baseline": float(curve_at(fit, mx[j])),
               "residual_from_gsm8k_curve": float(my[j] - curve_at(fit, mx[j]))}
        if g is not None:
            row.update({"gsm8k_baseline": float(gx[g]), "gsm8k_gain": float(gy[g]),
                        "shift_baseline": float(mx[j] - gx[g]), "shift_gain": float(my[j] - gy[g])})
        rows.append(row)
    return r, rows


def disagreement(Breq, Bfd, mpanel, flagged):
    """Requested - forced direct on MATH for each flagged model: mean over discovery draws with the jackknife t
    interval; holds when the interval excludes zero."""
    out = {}
    for k in flagged:
        if k not in mpanel:
            out[k] = {"in_panel": False}
            continue
        v = Breq[:, mpanel.index(k)] - Bfd[:, mpanel.index(k)]
        K = len(v)
        est = float(v.mean())
        lo, hi = iv.jack_t(est, (v.sum() - v) / (K - 1))
        out[k] = {"in_panel": True, "estimate": est, "ci95": [float(lo), float(hi)],
                  "holds": bool(lo > 0 or hi < 0)}
    return out


# ============================================================================ inputs


def task_arrays(task):
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    D = three_arm_arrays(inp["load"], fd_load, list(s["discovery_config_ids"]), panel, items)
    return {"panel": panel, "D": D, "n": len(items), "families": list(inp["families"]), "K": len(s["discovery_config_ids"])}


def readable(task, panel):
    pm = pd.read_csv(os.path.join(FD, "tables", f"per_model_{task}.csv")).set_index("model_key")
    return {k: bool(pm.loc[k, "closed_and_readable_pct"] >= CLEAN_MIN) for k in panel}


def flagged_on_math_screen():
    from src import math_d2
    log = json.load(open(math_d2.ADMISSIBILITY_LOG, encoding="utf-8"))
    diag = log.get("model_diagnostics") or (log.get("exclusions") or {}).get("model_diagnostics")
    if diag is None:
        raise SystemExit(f"{os.path.relpath(str(math_d2.ADMISSIBILITY_LOG), ROOT)} has no model_diagnostics block")
    return sorted({h["model_key"] for h in diag["flagged"] if h["rule"] == "direct_answer_meaningful"})


# ============================================================================ run


def run(a):
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    g = task_arrays("gsm8k")
    if a.dry_run:
        m = g
        mclean = readable("gsm8k", g["panel"])
        Breq, Bfd, _ = m["D"]
        gap = (Breq - Bfd).mean(0)
        flagged = [m["panel"][j] for j in np.argsort(-gap)[:3]]
        stand_in = "MATH replaced by GSM8K-29; flagged models: the three largest requested - forced gaps"
    else:
        m = task_arrays("math")
        mclean = readable("math", m["panel"])
        flagged = flagged_on_math_screen()
        stand_in = None
    print(f"arrays ({time.time() - t0:.0f} s)", flush=True)
    R = {"analysis": "the prediction for MATH-500", "dry_run": bool(a.dry_run),
         "stand_in": stand_in, "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method,
         "units": "pp; slopes in pp of gain per pp of baseline on points", "K_discovery": m["K"],
         "gsm8k_panel": g["panel"], "math_panel": m["panel"], "math_readable": [k for k in m["panel"] if mclean[k]],
         "clean_rule": f">= {CLEAN_MIN}% of forced outputs closed and readable", "min_models": MIN_MODELS,
         "outcomes": {}}
    rows = []
    for o in CHECKED:
        gB, gP = pair(g["D"], o)
        mB, mP = pair(m["D"], o)
        r, rr = limb_block(gB, gP, mB, mP, g["panel"], m["panel"], mclean, g["n"], m["n"], method, alpha)
        R["outcomes"][o] = {"label": OUTCOMES[o][2], **r}
        rows += [{"outcome": o, **x} for x in rr]
    Breq, Bfd, _ = m["D"]
    R["reason_when_asked_not_to"] = {"rule": "direct_answer_meaningful flagged on the MATH screen (model_diagnostics)",
                                     "flagged": flagged, "requested_minus_forced": disagreement(Breq, Bfd, m["panel"],
                                                                                                 flagged)}
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "math_prediction.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    txt = "\n".join(summary(R)) + "\n"
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)
    fig = a.fig or (FIG if not a.dry_run else os.path.join(a.out, "math_shift.csv"))
    os.makedirs(os.path.dirname(fig), exist_ok=True)
    pd.DataFrame(rows).to_csv(fig, index=False)
    print(f"done in {R['runtime_seconds']} s -> {os.path.relpath(a.out, ROOT)}", flush=True)
    return R


def macros(R):
    m = {}
    for o, r in R["outcomes"].items():
        s = "mathPred" + TAG[o]
        fit = r["gsm8k_curve"]
        m[s + "Concave"] = "yes" if fit["concave"] else "no"
        m[s + "Vertex"] = fmt(fit["vertex"], 1) if fit["concave"] else "--"
        m[s + "LimbM"] = str(len(r["limb_models"]))
        m[s + "LimbReadableM"] = str(len(r["limb_readable_on_math"]))
        if r["math_slopes"] is None:
            continue
        for sc, st in STAG.items():
            for key, kt in (("math_slopes", "Math"), ("gsm8k_slopes_same_models", "Gsm")):
                x = r[key][sc]
                m[f"{s}{kt}Slope{st}"] = fmt(x["slope"], 3)
                bounded = x["slope_set"]["kind"] == "bounded"
                m[f"{s}{kt}Slope{st}Lo"] = fmt(x["slope_set"]["a"], 3) if bounded else x["slope_set"]["kind"]
                m[f"{s}{kt}Slope{st}Hi"] = fmt(x["slope_set"]["b"], 3) if bounded else x["slope_set"]["kind"]
            m[f"{s}Holds{st}"] = "yes" if r["holds"][sc] else "no"
    d = R["reason_when_asked_not_to"]["requested_minus_forced"]
    m["mathPredReasonN"] = str(len(d))
    m["mathPredReasonDisagree"] = str(sum(bool(v.get("holds")) for v in d.values()))
    return m


def summary(R):
    L = ["# The prediction for MATH-500", ""]
    if R["dry_run"]:
        L += [f"**Dry run: {R['stand_in']}. No number here is a result.**", ""]
    L += ["Written before any MATH collection output was read. Discovery draws "
          f"({R['K_discovery']}); slopes of gain on baseline with 95% sets by the {R['slope_method']} method; MATH's "
          f"readable panel has {len(R['math_readable'])} of {len(R['math_panel'])} models ({R['clean_rule']}).", "",
          "| Outcome | GSM8K curve | Vertex | Limb models | Readable on MATH | MATH slope (points) | Logits | "
          "Error reduction | Same models on GSM8K (points) | Holds (points / logits / error reduction) |",
          "|---|---|---:|---:|---:|---|---|---|---|---|"]

    def st(x):
        s = x["slope_set"]
        return f"{fmt(x['slope'], 3)} " + (f"[{fmt(s['a'], 3)}, {fmt(s['b'], 3)}]" if s["kind"] == "bounded"
                                          else f"({s['kind']})")
    for o, r in R["outcomes"].items():
        fit = r["gsm8k_curve"]
        cells = [o, "concave" if fit["concave"] else "not concave", fmt(fit["vertex"], 1),
                 str(len(r["limb_models"])), str(len(r["limb_readable_on_math"]))]
        if r["math_slopes"] is None:
            cells += ["--"] * 4 + [r.get("note", "--")]
        else:
            cells += [st(r["math_slopes"][sc]) for sc in ins.SCALES] + [st(r["gsm8k_slopes_same_models"]["points"])]
            cells.append(" / ".join("yes" if r["holds"][sc] else "no" for sc in ins.SCALES))
        L.append("| " + " | ".join(cells) + " |")
    L += [""]
    for o, r in R["outcomes"].items():
        L.append(f"{o}: limb models {', '.join(r['limb_models']) or 'none'}; readable on MATH "
                 f"{', '.join(r['limb_readable_on_math']) or 'none'}" +
                 (f"; not readable {', '.join(r['limb_not_readable_on_math'])}" if r["limb_not_readable_on_math"] else "")
                 + ".")
    d = R["reason_when_asked_not_to"]
    L += ["", f"Models that reason when asked not to ({d['rule']}): {', '.join(d['flagged']) or 'none'}.", ""]
    if d["flagged"]:
        L += ["| Model | Requested − forced on MATH (pp) | Interval excludes zero |", "|---|---|---|"]
        for k, v in d["requested_minus_forced"].items():
            L.append(f"| {k} | " + ("not in the MATH panel | -- |" if not v["in_panel"] else
                                    f"{fmt(v['estimate'], 1)} [{fmt(v['ci95'][0], 1)}, {fmt(v['ci95'][1], 1)}] | "
                                    f"{'yes' if v['holds'] else 'no'} |"))
    L += ["", "Per-model shifts and residuals from the GSM8K curve: `results/figures/math_shift.csv`."]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig", default=None, help="default results/figures/math_shift.csv (a dry run: <out>)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("a dry run needs --out outside results/math_prediction")
    run(a)


if __name__ == "__main__":
    main()
