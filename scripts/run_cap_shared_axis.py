r"""Capability slopes of the chain-of-thought gain under both baselines on shared capability axes.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_cap_shared_axis.py

The earlier own-baseline slopes (results/model_resampling: chain of thought - forced on forced accuracy, chain of
thought - requested on requested accuracy) put the two gains on different x-axes. Here both gains, chain of thought -
requested (Req) and chain of thought - forced (Fd), are regressed on the same axis:
  Eci  the Epoch Capabilities Index (manifests/external/eci_scores_20261030.csv, through the panel manifest
       manifests/d2_panel_extension.json, scripts/run_transfer_tasks.eci_for), constant over draws;
  Fd   forced-baseline accuracy;
  Cot  chain-of-thought accuracy.
Every slope is src/robustness.cross_slope, the estimator of the existing capability slopes: cross-draw moments of
Appendix A (src/intervals.CrossMoment), so x and y come from different template draws wherever the axis is an arm.
Diff is Req - Fd on the same axis; both gains share the chain-of-thought arm, so Diff is also the slope of forced
minus requested accuracy on the axis. Intervals: 10,000 model resamples and 10,000 family resamples (whole
families, src/robustness.bootstrap_statistic); Req, Fd and Diff of a task use the same resamples (one generator per
task and kind, src/sampling.rng_for), so the difference is paired. Leave-one-family-out: src/robustness.leave_out.

Quadratic peak: the descriptive quadratic of src/instability.quadratic_curve (model-mean gain on model-mean x). On the
index the gain is the mean over the 24 draws. On an arm axis x and the gain come from disjoint halves of the draws
(even and odd positions of the sampler's discovery list), both assignments stacked into one fit, so x and y are again
from different draws; the fit on all-draw means is kept beside it as a check. A fit with no interior maximum is
labelled: 'trough' (convex, vertex inside the range), 'rises throughout' or 'falls throughout' (vertex outside the
range, or no curvature). The model bootstrap gives the share of resamples with an interior maximum.

Data: the 24 discovery draws and the panels of the existing capability slopes (scripts/robustness_inputs.py): GSM8K-29,
MATH-500 (29 models, 398 numeric items), BIG-Bench Hard-28. Strict scores from robustness_inputs; content scores from
data/processed/content/<task>_content.parquet through scripts/run_outcomes.with_content (cached in
data/processed/robustness/<task>_content_discovery.npz, rebuilt when a source table is newer).

Outputs: results/cap_shared_axis/cap_shared_axis.json (with macros, prefixes capSlope and capPeak), summary.md,
summary_clean.md, tables/slopes.csv and tables/per_model.csv.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import instability as ins, robustness as rb, sampling  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "cap_shared_axis")
CACHE = ri.CACHE
N_BOOT = 10000
SEED_LABEL = "cap-shared-axis"
TASKS = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"))
SCORINGS = (("strict", ""), ("content", "Content"))
AXES = (("eci", "Eci"), ("forced", "Fd"), ("cot", "Cot"))
GAINS = (("requested", "Req"), ("forced", "Fd"), ("diff", "Diff"))
AXIS_NAMES = {"eci": "Epoch Capabilities Index", "forced": "forced-baseline accuracy", "cot": "chain-of-thought accuracy"}
AXIS_UNITS = {"eci": "pp per index point", "forced": "pp per pp", "cot": "pp per pp"}
GAIN_NAMES = {"requested": "CoT - requested", "forced": "CoT - forced", "diff": "difference (requested - forced)"}
CONTENT = os.path.join(ROOT, "data", "processed", "content")


# ---------------------------------------------------------------------------- inputs

def content_discovery(task, panel):
    """{arm: (K, M) pp} of content scores on the discovery draws, the panel in the order of robustness_inputs, built
    with the loaders of scripts/run_outcomes.py --scoring content and cached. The strict arrays built on the same
    loaders are returned too, for the check against robustness_inputs."""
    npz = os.path.join(CACHE, f"{task}_content_discovery.npz")
    js = os.path.join(CACHE, f"{task}_content_discovery.json")
    sources = list(ri.SCORED[task]) + [os.path.join("data", "processed", "content", f"{task}_content.parquet")]
    if not (os.path.exists(npz) and os.path.exists(js)) or os.path.getmtime(npz) < ri._newest(sources):
        print(f"{task}: building the content arrays", flush=True)
        from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
        from run_outcomes import with_content
        inp, s, items, items_info = collection_inputs(task)
        pnl = list(inp["panel"])
        fd_load, _, _ = forced_inputs(task, pnl, s, items_info, inp["load"], False)
        load, fdl = with_content(inp["load"], task), with_content(fd_load, task)
        units = list(s["discovery_config_ids"])
        arr = {}
        for value, tag in (("content_correct", ""), ("strict_correct", "strict_")):
            Breq, Bfd, Pc = three_arm_arrays(load, fdl, units, pnl, items, value=value)
            arr.update({f"{tag}requested": Breq, f"{tag}forced": Bfd, f"{tag}cot": Pc})
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(npz, **arr)
        json.dump({"panel": pnl, "units": [str(u) for u in units]}, open(js, "w", encoding="utf-8"), indent=1)
    meta = json.load(open(js, encoding="utf-8"))
    if meta["panel"] != list(panel):
        raise SystemExit(f"{task}: the content panel differs from robustness_inputs")
    z = np.load(npz)
    return ({a: z[a] for a in ri.ARMS}, {a: z[f"strict_{a}"] for a in ri.ARMS}, meta["units"])


def task_inputs(task):
    """Panel facts and {scoring: {arm: (K, M)}} on the discovery draws."""
    x = ri.load(task)
    f = x["facts"]
    con, strict_check, units = content_discovery(task, f["panel"])
    if units != list(f["discovery_config_ids"]):
        raise SystemExit(f"{task}: the content draws differ from robustness_inputs")
    for a in ri.ARMS:
        if not np.allclose(strict_check[a], x["discovery"][a], atol=1e-9, rtol=0):
            raise SystemExit(f"{task}: strict {a} from the content loaders differs from robustness_inputs")
    return f, {"strict": x["discovery"], "content": con}


# ---------------------------------------------------------------------------- statistics

def axis_array(axis, arms, eci):
    """(K, M) array of an axis: the index tiled over draws, or an arm's accuracy."""
    if axis == "eci":
        return np.tile(np.asarray(eci, float), (arms["cot"].shape[0], 1))
    return arms[axis]


def gain_array(gain, arms):
    if gain == "requested":
        return arms["cot"] - arms["requested"]
    if gain == "forced":
        return arms["cot"] - arms["forced"]
    return (arms["cot"] - arms["requested"]) - (arms["cot"] - arms["forced"])


def slope_fn(X, G):
    def f(ix):
        return rb.cross_slope(X[:, ix], G[:, ix])[0]
    return f


def halves(K):
    """Draw positions of the two halves: even and odd positions of the discovery list."""
    return np.arange(0, K, 2), np.arange(1, K, 2)


def split_points(X, G, constant_x):
    """(x, y) of the quadratic fit, one point per model and assignment: x and the gain from disjoint halves of the
    draws, both assignments stacked; for a constant axis (the index) x and the all-draw mean gain."""
    if constant_x:
        return X[0], G.mean(0)
    a, b = halves(X.shape[0])
    return (np.concatenate([X[a].mean(0), X[b].mean(0)]), np.concatenate([G[b].mean(0), G[a].mean(0)]))


def shape(q, xmin, xmax):
    """What the fitted quadratic does over [xmin, xmax]: 'peak' (interior maximum), 'trough' (interior minimum),
    'rises throughout' or 'falls throughout' (no turning point inside the range); 'flat' if it does not move."""
    c2, c1 = q["c2"], q["c1"]
    v = q["vertex_baseline"]
    if np.isfinite(v) and xmin <= v <= xmax and c2 != 0:
        return "peak" if c2 < 0 else "trough"
    mid = 0.5 * (xmin + xmax)
    d = c1 + 2 * c2 * mid
    return "rises throughout" if d > 0 else "falls throughout" if d < 0 else "flat"


def quad(X, G, constant_x):
    x, y = split_points(X, G, constant_x)
    q = ins.quadratic_curve(x, y)
    s = shape(q, float(np.min(x)), float(np.max(x)))
    return {**q, "x_min": float(np.min(x)), "x_max": float(np.max(x)), "shape": s,
            "peak": q["vertex_baseline"] if s == "peak" else None,
            "n_points": int(len(x))}


def quad_boot(X, G, constant_x, idx):
    """Over model resamples: share with an interior maximum, percentiles of the peak among those."""
    peaks, n = [], 0
    for ix in idx:
        x, y = split_points(X[:, ix], G[:, ix], constant_x)
        if np.ptp(x) == 0:
            continue
        n += 1
        c2, c1, _ = np.polyfit(x, y, 2)
        if c2 < 0:
            v = -c1 / (2 * c2)
            if np.min(x) <= v <= np.max(x):
                peaks.append(v)
    out = {"n_boot": len(idx), "n_fitted": n, "share_peak": len(peaks) / n if n else None}
    if peaks:
        out["peak_ci95"] = [float(np.quantile(peaks, 0.025)), float(np.quantile(peaks, 0.975))]
        out["peak_median"] = float(np.median(peaks))
    return out


def boot_rng(task, kind):
    """A fresh generator per call: the same label gives the same resamples, so Req, Fd and Diff (and both scorings and
    all axes) of a task are resampled together."""
    return sampling.rng_for(SEED_LABEL, task, kind)


def cell(task, X, G, families, fams):
    f = slope_fn(X, G)
    M = X.shape[1]
    lofo = rb.leave_out(f, {fam: [k for k in range(M) if families[k] != fam] for fam in fams})
    mb = rb.bootstrap_statistic(f, M, None, N_BOOT, boot_rng(task, "model"))
    fb = rb.bootstrap_statistic(f, M, list(families), N_BOOT, boot_rng(task, "family"))
    vals = np.array(list(lofo.values()), float)
    return {"slope": f(np.arange(M)), "model_bootstrap": mb, "model_bootstrap_sign": rb.interval_sign(mb["ci95"]),
            "family_bootstrap": fb, "family_bootstrap_sign": rb.interval_sign(fb["ci95"]),
            "leave_one_family_out": lofo, "lofo_min": float(np.nanmin(vals)), "lofo_max": float(np.nanmax(vals)),
            "lofo_argmin": min(lofo, key=lambda k: lofo[k]), "lofo_argmax": max(lofo, key=lambda k: lofo[k])}


def analyse_task(task, f, S):
    panel, fams_list, eci = f["panel"], list(f["families"]), f["eci"]
    M, fams = len(panel), sorted(set(fams_list))
    idx_m = rb.resample_columns(M, None, N_BOOT, boot_rng(task, "model"))
    T = {"name": ri.NAMES[task], "M": M, "K": int(S["strict"]["cot"].shape[0]), "models": panel,
         "families": fams_list, "family_sizes": {k: fams_list.count(k) for k in fams}, "eci": eci,
         "discovery_config_ids": f["discovery_config_ids"], "scorings": {}}
    for sc, _ in SCORINGS:
        arms = S[sc]
        B = {}
        for ax, _ in AXES:
            X = axis_array(ax, arms, eci)
            A = {"gains": {}, "quadratic": {}}
            for g, _ in GAINS:
                A["gains"][g] = cell(task, X, gain_array(g, arms), fams_list, fams)
            for g in ("requested", "forced"):
                G = gain_array(g, arms)
                q = quad(X, G, ax == "eci")
                q["model_bootstrap"] = quad_boot(X, G, ax == "eci", idx_m)
                q["all_draw_fit"] = ins.quadratic_curve(X.mean(0), G.mean(0))
                A["quadratic"][g] = q
            B[ax] = A
        per_model = [{"model": k, "family": fams_list[j], "eci": eci[j],
                      **{f"{a}_pp": float(arms[a][:, j].mean()) for a in ri.ARMS},
                      "gain_requested_pp": float((arms["cot"] - arms["requested"])[:, j].mean()),
                      "gain_forced_pp": float((arms["cot"] - arms["forced"])[:, j].mean())}
                     for j, k in enumerate(panel)]
        T["scorings"][sc] = {"axes": B, "per_model": per_model}
    return T


# ---------------------------------------------------------------------------- checks against earlier results

def checks(R):
    """The strict forced-on-forced slope against results/model_resampling, and the index slopes against
    results/outcomes* (strict and content): the same estimator on the same arrays should give the same numbers."""
    out = []
    mr = json.load(open(os.path.join(ROOT, "results", "model_resampling", "model_resampling.json"), encoding="utf-8"))
    oc = {("strict", "gsm8k"): ("outcomes", "gsm8k"), ("strict", "bbh"): ("outcomes", "bbh"),
          ("strict", "math"): ("outcomes_math", "math"), ("content", "gsm8k"): ("outcomes_content", "gsm8k"),
          ("content", "bbh"): ("outcomes_content", "bbh"), ("content", "math"): ("outcomes_content_math", "math")}
    for task, T in R["tasks"].items():
        mine = T["scorings"]["strict"]["axes"]["forced"]["gains"]["forced"]["slope"]
        saved = mr["tasks"][task]["blocks"]["CotMinusForced/full"]["slope"]
        out.append({"check": f"{task} strict, CoT - forced on forced accuracy vs model_resampling", "value": mine,
                    "saved": saved, "abs_diff": abs(mine - saved)})
        for sc, _ in SCORINGS:
            name, t = oc[(sc, task)]
            p = os.path.join(ROOT, "results", name, f"{name}.json")
            if not os.path.exists(p):
                continue
            o = json.load(open(p, encoding="utf-8"))["tasks"].get(t)
            if o is None:
                continue
            for g, okey in (("requested", "CotMinusReq"), ("forced", "CotMinusForced")):
                mine = T["scorings"][sc]["axes"]["eci"]["gains"][g]["slope"]
                saved = o["outcomes"][okey]["slopes"]["scales"]["points"]["index"]["slope"]
                out.append({"check": f"{task} {sc}, {GAIN_NAMES[g]} on the index vs {name}", "value": mine,
                            "saved": saved, "abs_diff": abs(mine - saved)})
    return out


# ---------------------------------------------------------------------------- macros

def fx(x, nd):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}".replace("-", "$-$")


def macros(R):
    m = {"capSlopeNBoot": f"{N_BOOT:,}".replace(",", "{,}"), "capSlopeEciMissing": str(len(R["eci_missing"]))}
    for task, tt in TASKS:
        T = R["tasks"][task]
        m[f"capSlopeModels{tt}"] = str(T["M"])
        m[f"capSlopeDraws{tt}"] = str(T["K"])
        for sc, st in SCORINGS:
            for ax, at in AXES:
                A = T["scorings"][sc]["axes"][ax]
                for g, gt in GAINS:
                    c = A["gains"][g]
                    n = f"capSlope{tt}{at}{gt}{st}"
                    m[n] = fx(c["slope"], 3)
                    mb, fb = c["model_bootstrap"]["ci95"] or [None, None], c["family_bootstrap"]["ci95"] or [None, None]
                    m[n + "Lo"], m[n + "Hi"] = fx(mb[0], 3), fx(mb[1], 3)
                    m[n + "FamLo"], m[n + "FamHi"] = fx(fb[0], 3), fx(fb[1], 3)
                    m[n + "LofoLo"], m[n + "LofoHi"] = fx(c["lofo_min"], 3), fx(c["lofo_max"], 3)
                for g, gt in GAINS[:2]:
                    q = A["quadratic"][g]
                    n = f"capPeak{tt}{at}{gt}{st}"
                    m[n] = fx(q["peak"], 1) if q["peak"] is not None else "none"
                    m[n + "Shape"] = q["shape"]
                    m[n + "Vertex"] = fx(q["vertex_baseline"], 1)
    for k in m:
        if not k.isalpha():
            raise ValueError(f"macro name {k!r} is not letters only")
    return m


# ---------------------------------------------------------------------------- tables, summary

def slope_rows(R):
    rows = []
    for task, _ in TASKS:
        T = R["tasks"][task]
        for sc, _ in SCORINGS:
            for ax, _ in AXES:
                A = T["scorings"][sc]["axes"][ax]
                for g, _ in GAINS:
                    c = A["gains"][g]
                    mb, fb = c["model_bootstrap"], c["family_bootstrap"]
                    q = A["quadratic"].get(g, {})
                    rows.append({"task": task, "scoring": sc, "axis": ax, "gain": g, "M": T["M"], "slope": c["slope"],
                                 "model_lo": (mb["ci95"] or [None, None])[0], "model_hi": (mb["ci95"] or [None, None])[1],
                                 "model_share_below_zero": mb["share_below_zero"], "model_undefined": mb["n_undefined"],
                                 "family_lo": (fb["ci95"] or [None, None])[0], "family_hi": (fb["ci95"] or [None, None])[1],
                                 "family_share_below_zero": fb["share_below_zero"], "family_undefined": fb["n_undefined"],
                                 "lofo_min": c["lofo_min"], "lofo_max": c["lofo_max"],
                                 "quad_shape": q.get("shape"), "quad_vertex": q.get("vertex_baseline"),
                                 "quad_share_peak_model_boot": (q.get("model_bootstrap") or {}).get("share_peak")})
    return rows


def per_model_rows(R):
    rows = []
    for task, _ in TASKS:
        for sc, _ in SCORINGS:
            for r in R["tasks"][task]["scorings"][sc]["per_model"]:
                rows.append({"task": task, "scoring": sc, **r})
    return rows


def s3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}".replace("-", "−")


def ci(c):
    return "undefined" if c is None else f"[{s3(c[0])}, {s3(c[1])}]"


def excl(c):
    return c is not None and (c[0] > 0 or c[1] < 0)


def answer(R):
    """Cells where the two baselines' slopes have opposite signs, and how many of their intervals exclude zero."""
    rows = []
    for task, _ in TASKS:
        for sc, _ in SCORINGS:
            for ax, _ in AXES:
                G = R["tasks"][task]["scorings"][sc]["axes"][ax]["gains"]
                r, f_, d = G["requested"], G["forced"], G["diff"]
                rows.append({"task": task, "scoring": sc, "axis": ax, "req": r["slope"], "fd": f_["slope"],
                             "diff": d["slope"], "opposite": bool(np.sign(r["slope"]) * np.sign(f_["slope"]) < 0),
                             "req_excl_model": excl(r["model_bootstrap"]["ci95"]),
                             "req_excl_family": excl(r["family_bootstrap"]["ci95"]),
                             "fd_excl_model": excl(f_["model_bootstrap"]["ci95"]),
                             "fd_excl_family": excl(f_["family_bootstrap"]["ci95"]),
                             "diff_excl_model": excl(d["model_bootstrap"]["ci95"]),
                             "diff_excl_family": excl(d["family_bootstrap"]["ci95"]),
                             "lofo_req_sign_kept": bool(r["lofo_min"] * r["lofo_max"] > 0),
                             "lofo_fd_sign_kept": bool(f_["lofo_min"] * f_["lofo_max"] > 0)})
    return rows


BENCH = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard"}
AX_SHORT = {"eci": "index", "forced": "forced accuracy", "cot": "CoT accuracy"}


def lead(R):
    A = R["answer"]
    opp = [a for a in A if a["opposite"]]
    both = [a for a in opp if a["req_excl_model"] and a["fd_excl_model"] and a["req_excl_family"]
            and a["fd_excl_family"]]
    d_ex = [a for a in A if a["diff_excl_model"] and a["diff_excl_family"]]

    same_ex = [a for a in A if not a["opposite"] and a["req_excl_model"] and a["fd_excl_model"]
               and a["req_excl_family"] and a["fd_excl_family"]]
    neg = [a for a in A if a["diff"] < 0]

    def nm(a):
        return f"{BENCH[a['task']]} {a['scoring']} on {AX_SHORT[a['axis']]}"

    def cis(a, g):
        c = R["tasks"][a["task"]]["scorings"][a["scoring"]]["axes"][a["axis"]]["gains"][g]
        return f"model {ci(c['model_bootstrap']['ci95'])}, family {ci(c['family_bootstrap']['ci95'])}"
    L = [f"Opposite signs in {len(opp)} of {len(A)} cells (3 benchmarks × 2 scorings × 3 axes)"
         + (": " + "; ".join(f"{nm(a)} (requested {s3(a['req'])}, {cis(a, 'requested')}; forced {s3(a['fd'])}, "
                             f"{cis(a, 'forced')})" for a in opp) if opp else "") + "."]
    L.append(f"Of these, both slopes have model and family intervals that exclude zero in {len(both)}"
             + (": " + "; ".join(nm(a) for a in both) if both else "") + ".")
    L.append(f"Same sign with all four intervals (model and family, both baselines) excluding zero: {len(same_ex)} of "
             f"{len(A)} cells, all positive: " + "; ".join(nm(a) for a in same_ex) + "."
             if same_ex and all(a["req"] > 0 for a in same_ex) else
             f"Same sign with all four intervals excluding zero: {len(same_ex)} of {len(A)} cells.")
    L.append(f"The paired difference (requested − forced) is negative in {len(neg)} of {len(A)} cells (the gain over "
             f"the requested baseline rises less steeply with capability than the gain over the forced baseline); its "
             f"model and family intervals both exclude zero in {len(d_ex)}"
             + (": " + "; ".join(f"{nm(a)} {s3(a['diff'])}" for a in d_ex) if d_ex else "") + ".")
    ob = R.get("own_baseline", {}).get("gsm8k")
    if ob:
        g = R["tasks"]["gsm8k"]["scorings"]["strict"]["axes"]
        L.append(f"GSM8K strict, against the earlier own-baseline slopes (requested gain on requested accuracy "
                 f"{s3(ob['CotMinusReq'])}, forced gain on forced accuracy {s3(ob['CotMinusForced'])}): the requested "
                 f"gain's slope is {s3(g['eci']['gains']['requested']['slope'])} on the index, "
                 f"{s3(g['forced']['gains']['requested']['slope'])} on forced accuracy and "
                 f"{s3(g['cot']['gains']['requested']['slope'])} on CoT accuracy.")
    return L


def summary(R):
    L = ["# Capability slopes of the chain-of-thought gain under both baselines, on shared axes", "",
         "Code `scripts/run_cap_shared_axis.py`; "
         "macros `paper/macros_cap_shared_axis.py`; tests `tests/test_cap_shared_axis.py`.", "",
         "## Answer", ""] + ["- " + s for s in lead(R)] + [""]
    L += ["## Method", "",
          f"- Panels and draws: the panels of the existing capability slopes (`scripts/robustness_inputs.py`, "
          f"`results/model_resampling`): GSM8K-29 (29 models), MATH-500 (29 models, 398 numeric items), "
          f"BIG-Bench Hard-28 (28 models; {', '.join(R['bbh_missing']) or 'none'} not in it); the 24 discovery draws "
          f"of each sampler. Strict scores and content scores (`data/processed/content/<task>_content.parquet`, the "
          f"current readers).",
          "- Gains: Req = chain of thought − requested; Fd = chain of thought − forced; per draw and model, in pp.",
          "- Axes, shared by both gains: Eci, the Epoch Capabilities Index (`manifests/external/eci_scores_20261030.csv`, "
          "read through `manifests/d2_panel_extension.json`; the same in every draw); Fd, forced-baseline accuracy; "
          "Cot, chain-of-thought accuracy.",
          f"- Epoch index coverage: {'every panel model has an index value' if not R['eci_missing'] else 'missing: ' + ', '.join(R['eci_missing'])}"
          f"; the panel manifest's values equal the CSV's for all {R['eci_checked']} models.",
          "- Slope: `src/robustness.cross_slope`, the estimator of the existing capability slopes (cross-draw moments, "
          "Appendix A; x and y from different draws wherever the axis is an arm). Units: pp per index point on Eci, pp per "
          "pp on Fd and Cot.",
          "- Diff = Req − Fd on the same axis. Both gains share the chain-of-thought arm, so Diff is the slope of forced "
          "minus requested accuracy on that axis.",
          f"- Intervals: {N_BOOT:,} model resamples and {N_BOOT:,} family resamples (whole families drawn with "
          "replacement; five families), percentile 95%; Req, Fd and Diff of a task use the same resamples, and so do "
          "both scorings and all axes. Resamples where V_x is not positive are counted as undefined. Leave one family "
          "out: the range over the five panels with one family removed.",
          "- Quadratic peak: OLS quadratic of model-mean gain on model-mean x (`src/instability.quadratic_curve`). On "
          "Eci the gain is the 24-draw mean. On Fd and Cot, x and the gain come from disjoint halves of the draws (even "
          "and odd positions of the discovery list), both assignments stacked (2M points). Shape: peak (interior "
          "maximum), trough (interior minimum: the fit has no maximum), rises or falls throughout (turning point outside "
          "the range). The model bootstrap gives the share of resamples with an interior maximum. Descriptive.", ""]
    for sc, _ in SCORINGS:
        L += [f"## Slopes, {sc} scoring", "",
              "| Benchmark | Axis | Gain | Slope | Model 95% | Share < 0 | Family 95% | Leave one family out |",
              "|---|---|---|---:|---|---:|---|---|"]
        for task, _ in TASKS:
            for ax, _ in AXES:
                for g, _ in GAINS:
                    c = R["tasks"][task]["scorings"][sc]["axes"][ax]["gains"][g]
                    L.append(f"| {BENCH[task]} | {AX_SHORT[ax]} | {GAIN_NAMES[g]} | {s3(c['slope'])} | "
                             f"{ci(c['model_bootstrap']['ci95'])} | {c['model_bootstrap']['share_below_zero']:.3f} | "
                             f"{ci(c['family_bootstrap']['ci95'])} | {s3(c['lofo_min'])} to {s3(c['lofo_max'])} |")
        L.append("")
    L += ["## Quadratic peaks", "",
          "Vertex in the axis's units (index points on Eci, pp on Fd and Cot). Range: the x range of the fitted points.", "",
          "| Benchmark | Scoring | Axis | Gain | Shape | Vertex | Range | Share of model resamples with a peak | Peak 95% (resamples with a peak) | All-draw fit: shape, vertex |",
          "|---|---|---|---|---|---:|---|---:|---|---|"]
    for task, _ in TASKS:
        for sc, _ in SCORINGS:
            for ax, _ in AXES:
                for g in ("requested", "forced"):
                    q = R["tasks"][task]["scorings"][sc]["axes"][ax]["quadratic"][g]
                    b = q["model_bootstrap"]
                    a = q["all_draw_fit"]
                    a_shape = shape(a, q["x_min"], q["x_max"]) if "c2" in a else "n/a"
                    pk = b.get("peak_ci95")
                    L.append(f"| {BENCH[task]} | {sc} | {AX_SHORT[ax]} | {GAIN_NAMES[g]} | {q['shape']} | "
                             f"{q['vertex_baseline']:.1f} | {q['x_min']:.1f} to {q['x_max']:.1f} | "
                             f"{b['share_peak']:.3f} | "
                             + (f"[{pk[0]:.1f}, {pk[1]:.1f}]" if pk else "none") +
                             f" | {a_shape}, {a['vertex_baseline']:.1f} |")
    L += ["", "## Checks", ""]
    for c in R["checks"]:
        L.append(f"- {c['check']}: {c['value']:.6f} against {c['saved']:.6f} (difference {c['abs_diff']:.2e}).")
    L += ["", "## Undefined resamples", ""]
    und = [r for r in slope_rows(R) if r["model_undefined"] or r["family_undefined"]]
    if not und:
        L.append("None: V_x was positive in every resample.")
    for r in und:
        L.append(f"- {r['task']} {r['scoring']} {r['axis']} {r['gain']}: {r['model_undefined']} model, "
                 f"{r['family_undefined']} family.")
    L += ["", "## Files", "",
          "- `cap_shared_axis.json` (every number, per family leave-out values, macros), `tables/slopes.csv`, "
          "`tables/per_model.csv` (model means per arm and gain)."]
    return L


# ---------------------------------------------------------------------------- run

def eci_coverage(panels):
    """Every panel model's index value in the October CSV, through the panel manifest's eci_name; models missing."""
    from src import bbh_d2
    csv = pd.read_csv(os.path.join(ROOT, "manifests", "external", "eci_scores_20261030.csv"))
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    by = {e["repo"]: e for e in man["panel"]}
    reg = bbh_d2.load_registries()
    missing, checked = [], 0
    for k in sorted(set().union(*panels)):
        e = by.get(reg[k]["hf_id"])
        hit = csv[(csv["Model"] == e["eci_name"]) | (csv["Display name"] == e["eci_name"])] if e else csv.iloc[0:0]
        if e is None or len(hit) != 1 or not np.isfinite(hit["eci"].iloc[0]):
            missing.append(k)
            continue
        if abs(float(hit["eci"].iloc[0]) - float(e["eci"])) > 1e-9:
            raise SystemExit(f"{k}: the panel manifest's index {e['eci']} differs from the CSV's {hit['eci'].iloc[0]}")
        checked += 1
    return missing, checked


def run():
    t0 = time.time()
    R = {"analysis": "capability slopes of the chain-of-thought gain under both baselines on shared axes "
                     "",
         "created": time.strftime("%Y-%m-%d %H:%M"), "n_boot": N_BOOT, "seed_label": SEED_LABEL,
         "estimator": "src/robustness.cross_slope (cross-draw moments, Appendix A)",
         "units": AXIS_UNITS, "tasks": {}}
    for task, _ in TASKS:
        f, S = task_inputs(task)
        R["tasks"][task] = analyse_task(task, f, S)
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    panels = [R["tasks"][t]["models"] for t, _ in TASKS]
    R["bbh_missing"] = sorted(set(R["tasks"]["gsm8k"]["models"]) - set(R["tasks"]["bbh"]["models"]))
    R["eci_missing"], R["eci_checked"] = eci_coverage(panels)
    R["checks"] = checks(R)
    mr = json.load(open(os.path.join(ROOT, "results", "model_resampling", "model_resampling.json"), encoding="utf-8"))
    R["own_baseline"] = {t: {o: mr["tasks"][t]["blocks"][f"{o}/full"]["slope"] for o in ("CotMinusReq", "CotMinusForced")}
                         for t, _ in TASKS}
    R["answer"] = answer(R)
    R["lead"] = lead(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(slope_rows(R)).to_csv(os.path.join(OUT, "tables", "slopes.csv"), index=False)
    pd.DataFrame(per_model_rows(R)).to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    m = macros(R)
    ri.write_outputs(OUT, R, summary(R), macros=m)
    with open(os.path.join(OUT, "macros.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Macros of results/cap_shared_axis ({len(m)}), as paper/macros_cap_shared_axis.py returns them\n\n")
        fh.writelines(f"- `\\{k}` = {v}\n" for k, v in m.items())
    print(f"done in {R['runtime_seconds']} s -> results/cap_shared_axis ({len(m)} macros)", flush=True)
    return R


if __name__ == "__main__":
    run()
