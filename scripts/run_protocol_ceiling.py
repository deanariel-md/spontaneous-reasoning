r"""Protocol ceiling curve. Methods: src/protocol_ceiling.py.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_protocol_ceiling.py

Per task (GSM8K-29, BIG-Bench Hard-28, MATH-500 numeric; full panels), the 18 benefit definitions of the specification
analysis (arm: chain of thought against requested or forced direct; scoring: strict, lenient, content; scale: points,
logits, error reduction) on every distinct configuration of the discovery and validation draws (a repeated
configuration repeats its outputs, so it enters once). Agreement C of K-configuration mean gains, K = 1 to 24:
    within      the same protocol, two disjoint K-sets (empirical while 2K <= D; Gaussian pair model beyond);
    between     two protocols on the same K configurations, for pairs that differ in the arm only, the scoring only, the
                scale only, and any two distinct protocols (a randomly drawn second protocol);
    disjoint    two protocols on disjoint K-sets (sensitivity, 2K <= D).
The limit as K grows: within tends to one; between is estimated by the plug-in at all D configurations (with a
configuration bootstrap) and by the Gaussian posterior (src/protocol_ceiling.limit_posterior). One number per task: the
limit of between-arm agreement under the collection's scoring and scale (strict, points).

The tasks of the section `tasks` of config/added_contrasts.yaml (HotpotQA) follow the three,
each on its own random stream, with the scoring axis the configuration names (HotpotQA: strict and lenient exact
match, and token F1; it has no content scoring).

Outputs: results/protocol_ceiling/protocol_ceiling.json, summary.md, summary_clean.md, tables/curves.csv,
and tables/figure_curves.csv (scripts/write_curve_tables.py, every task, HotpotQA last).
"""
import itertools
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
from src import instability as ins, protocol_ceiling as pcl, sampling  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "protocol_ceiling")
CACHE = os.path.join(ROOT, "data", "processed", "protocol_ceiling")
TASKS = ("gsm8k", "bbh", "math")
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
SCORINGS = {"strict": "strict_correct", "lenient": "lenient_correct", "content": "content_correct"}
ARMS = ("requested", "forced")
KS = tuple(range(1, 25))
N_SPLITS = 200
N_BOOT = 2000
COLLECTION = ("strict", "requested", "points")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def scorings(task):
    """{name: column} of a task's scoring axis: strict, lenient and content on the main tasks; an added task's own."""
    if task in TASKS:
        return SCORINGS
    import added_contrasts as ac
    return dict(ac.load_tasks()[task]["scorings"])


def task_title(task):
    if task not in TITLE and task not in ri.NAMES:
        ri.added_tasks()
    return TITLE.get(task) or ri.NAMES[task]


def load(task):
    """{scoring: {'requested', 'forced', 'cot'} (D, M) pp} on the distinct configurations, the units, panel, n."""
    path = os.path.join(CACHE, f"{task}.npz")
    sc_map = scorings(task)
    if task not in ri.SCORED:
        ri.added_tasks()
    srcs = ri.SCORED[task] + ([f"data/processed/content/{task}_content.parquet"] if "content" in sc_map else [])
    if os.path.exists(path) and os.path.getmtime(path) >= ri._newest(srcs):
        z = np.load(path, allow_pickle=False)
        D = {s: {a: z[f"{s}_{a}"] for a in ("requested", "forced", "cot")} for s in sc_map}
        return D, [str(u) for u in z["units"]], [str(m) for m in z["panel"]], int(z["n_items"])
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    from run_outcomes import with_content
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    units = list(dict.fromkeys([str(u) for u in s["discovery_config_ids"]] + [str(u) for u in s["validation_config_ids"]]))
    D, flat = {}, {}
    for sc, value in sc_map.items():
        cl, fl = (inp["load"], fd_load) if sc != "content" else (with_content(inp["load"], task),
                                                                 with_content(fd_load, task))
        Breq, Bfd, Pc = three_arm_arrays(cl, fl, units, panel, items, value=value)
        D[sc] = {"requested": Breq, "forced": Bfd, "cot": Pc}
        flat.update({f"{sc}_requested": Breq, f"{sc}_forced": Bfd, f"{sc}_cot": Pc})
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, units=np.array(units), panel=np.array(panel), n_items=len(items), **flat)
    return D, units, panel, len(items)


def benefits(D, n):
    """{(scoring, arm, scale): (D, M) benefit}; and the count of cells a scale replaced."""
    Y, replaced = {}, {}
    for sc, arm, scale in itertools.product(D, ARMS, ins.SCALES):
        x = ins.scale_scores(D[sc][arm], D[sc]["cot"], scale, n)
        Y[(sc, arm, scale)] = x["y"]
        replaced["/".join((sc, arm, scale))] = x["n_replaced"]
    return Y, replaced


def axis_of(p, q):
    diff = [a for a, u, v in zip(("scoring", "arm", "scale"), p, q) if u != v]
    return diff[0] if len(diff) == 1 else None


def task_result(task, rng, loaded=None, head_scoring="strict", boot_rng=None, scoring_columns=None, title=None):
    """One task's curves. Defaults reproduce the main run; `loaded` (D, units, panel, n) replaces load(task),
    `head_scoring` names the scoring of the headline pair (the collection's, strict, by default), `boot_rng` the
    plug-in bootstrap's stream, `scoring_columns` the scoring axis recorded for an added task (HotpotQA under its
    content reading, scripts/run_hotpot_content.py)."""
    D, units, panel, n = loaded if loaded is not None else load(task)
    Y, replaced = benefits(D, n)
    nD = next(iter(Y.values())).shape[0]
    defs = list(Y)
    within = {"/".join(p): pcl.within_curve(Y[p], KS, rng, N_SPLITS) for p in defs}
    within_model = {"/".join(p): {K: pcl.gaussian_within(Y[p], K) for K in KS} for p in defs}
    pairs = list(itertools.combinations(defs, 2))
    between, disjoint, model_b, lim = {}, {}, {}, {}
    for p, q in pairs:
        k = "/".join(p) + " | " + "/".join(q)
        between[k] = pcl.between_curve(Y[p], Y[q], KS, rng, N_SPLITS)
        disjoint[k] = pcl.between_curve(Y[p], Y[q], KS, rng, N_SPLITS, disjoint=True)
        model_b[k] = {K: pcl.gaussian_between(Y[p], Y[q], K) for K in KS}
        lim[k] = {"axis": axis_of(p, q) or "several", "plugin": pcl.limit_plugin(Y[p], Y[q]),
                  "posterior": pcl.limit_posterior(Y[p], Y[q])}
    # axis means
    groups = {"arm": [], "scoring": [], "scale": [], "random": []}
    for p, q in pairs:
        k = "/".join(p) + " | " + "/".join(q)
        a = axis_of(p, q)
        if a:
            groups[a].append(k)
        groups["random"].append(k)

    def mean_curve(curves, keys):
        out = {}
        for K in KS:
            v = [curves[k][K] for k in keys if K in curves[k]]
            if len(v) == len(keys):
                out[K] = float(np.mean(v))
        return out
    axes = {}
    for g, keys in groups.items():
        axes[g] = {"pairs": len(keys), "same_draws": mean_curve(between, keys), "disjoint": mean_curve(disjoint, keys),
                   "model": mean_curve(model_b, keys),
                   "limit_plugin": float(np.mean([lim[k]["plugin"] for k in keys])),
                   "limit_posterior": float(np.mean([lim[k]["posterior"] for k in keys]))}
    axes["within"] = {"protocols": len(defs), "empirical": mean_curve(within, ["/".join(p) for p in defs]),
                      "model": mean_curve(within_model, ["/".join(p) for p in defs])}
    # the collection's protocol and its arm partner
    coll = (head_scoring, "requested", "points")
    cp = "/".join(coll)
    ap = "/".join((head_scoring, "forced", "points"))
    key = f"{cp} | {ap}"
    head = {"protocols": [cp, ap], "within_requested": within[cp], "within_forced": within[ap],
            "within_requested_model": within_model[cp], "between_same_draws": between[key],
            "between_disjoint": disjoint[key], "between_model": model_b[key], **lim[key],
            "plugin_bootstrap": pcl.bootstrap_limit(Y[coll], Y[(head_scoring, "forced", "points")],
                                                    boot_rng if boot_rng is not None
                                                    else sampling.rng_for("protocol-ceiling", task, "boot"), N_BOOT)}
    out = {"title": title or task_title(task), "M": len(panel), "D": nD, "n_items": n, "configurations": units,
            "replaced_cells": replaced, "headline": head, "axes": axes, "within": within, "between": between,
            "disjoint": disjoint, "limits": lim}
    if task not in TASKS:
        out["added_task"] = True
        out["scorings"] = scoring_columns or scorings(task)
    return out


def f3(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.3f}".replace("-", "−")


def summary(R):
    L = ["# Protocol ceiling curve", "",
         "Code `scripts/run_protocol_ceiling.py`, "
         "`src/protocol_ceiling.py`.", "",
         "Agreement C (share of model pairs whose K-draw mean gains are ordered the same way, ties one half) between "
         "two evaluations, each averaging K prompt configurations. Within a protocol the two evaluations use disjoint "
         "configurations; between protocols they use the same configurations, so only the protocol changes. Protocols "
         "are the 18 benefit definitions of the specification analysis (arm × scoring × scale), on every distinct "
         "configuration of the discovery and validation draws (D per task below). Averages over 200 random sets per K. "
         "Empirical within-protocol values exist while 2K ≤ D; beyond that the Gaussian pair model gives the curve "
         "(it ignores ties, so it runs above the empirical values at small K on discrete scores).", "",
         "## One number per task: the ceiling across the arm switch (strict, points)", "",
         "| Task | D | Within, K = 1 | Within, largest empirical K | Between arms, K = 1 | K = 12 | K = 24 | "
         "Limit, plug-in (all D) [bootstrap 95%] | Limit, posterior |", "|---|---:|---:|---|---:|---:|---:|---|---:|"]
    for t, x in R["tasks"].items():
        h = x["headline"]
        kmax = max(h["within_requested"]) if h["within_requested"] else None
        b = h["between_same_draws"]
        c = h["plugin_bootstrap"]["ci95"]
        L.append(f"| {x['title']} | {x['D']} | {f3(h['within_requested'].get(1))} | "
                 f"{f3(h['within_requested'].get(kmax))} (K = {kmax}) | {f3(b.get(1))} | {f3(b.get(12))} | "
                 f"{f3(b.get(24))} | {f3(h['plugin'])} [{f3(c[0])}, {f3(c[1])}] | {f3(h['posterior'])} |")
    L += ["", "Within: chain of thought − requested direct, strict, points (the collection's protocol). Between arms: "
          "the same draws scored against requested and against forced direct. The between-protocol curves are flat "
          "from K = 1: on the same draws the two protocols share their prompt noise, so what separates them is the "
          "protocol, and averaging more draws does not move it.", "",
          "When the draws change as well (disjoint K-sets, arm switch, strict, points): " + "; ".join(
              f"{x['title']} {f3(x['headline']['between_disjoint'].get(1))} at K = 1 and "
              f"{f3(x['headline']['between_disjoint'][max(x['headline']['between_disjoint'])])} at K = "
              f"{max(x['headline']['between_disjoint'])}" for x in R["tasks"].values()) + ".", "",
          "## Every axis (means over the protocol pairs that differ in that axis only; 'random' is any two distinct "
          "protocols)", "",
          "| Task | Comparison | Pairs | K = 1 | K = 4 | K = 12 | K = 24 | Limit, plug-in | Limit, posterior |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, x in R["tasks"].items():
        w = x["axes"]["within"]
        L.append(f"| {x['title']} | within (same protocol, disjoint draws) | {w['protocols']} protocols | "
                 f"{f3(w['empirical'].get(1))} | {f3(w['empirical'].get(4))} | {f3(w['empirical'].get(12))} | "
                 f"{f3(w['model'].get(24))} (model) | 1 | 1 |")
        for g in ("arm", "scoring", "scale", "random"):
            a = x["axes"][g]
            s = a["same_draws"]
            L.append(f"| {x['title']} | {g} | {a['pairs']} | {f3(s.get(1))} | {f3(s.get(4))} | {f3(s.get(12))} | "
                     f"{f3(s.get(24))} | {f3(a['limit_plugin'])} | {f3(a['limit_posterior'])} |")
    L += ["", "Disjoint-draw between-protocol curves (both the protocol and the draws change) are in the JSON and "
          "`tables/curves.csv`. Cells a scale replaced (a 0% or 100% score) are counted in the JSON (`replaced_cells`), "
          "as the specification analysis does.", ""] + [
          f"{x['title']} (added from `config/added_contrasts.yaml`): its scoring axis is "
          + ", ".join(f"{k} ({v})" for k, v in x["scorings"].items()) + "; it has no content scoring." for x in R["tasks"].values() if x.get("added_task")] + ([""] if any(
              x.get("added_task") for x in R["tasks"].values()) else []) + [
          "## Reading", "",
          "- Within a protocol, agreement rises toward one as K grows. Between protocols on the same draws it rises "
          "only to the limit: the share of model pairs whose true gains are ordered the same way under both "
          "protocols. No number of draws closes that gap.",
          "- The plug-in limit uses the D configurations' means, whose noise can cross a near-zero pair either way; "
          "the posterior weights each pair by its sign uncertainty. Two estimates of one limit under the Gaussian "
          "reference; neither is a law for a fixed panel. Here they differ by at most 0.01.", "",
          f"Runtime {R['runtime_s']:.0f} s."]
    return L


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    R = {"analysis": "protocol ceiling curve", "created": time.strftime("%Y-%m-%d %H:%M"),
         "Ks": list(KS), "n_splits": N_SPLITS, "tasks": {}}
    for t in TASKS + ri.added_tasks():
        R["tasks"][t] = task_result(t, sampling.rng_for("protocol-ceiling", t))
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_s"] = round(time.time() - t0, 1)
    rows = []
    for t, x in R["tasks"].items():
        for kind in ("within", "between", "disjoint"):
            for k, c in x[kind].items():
                for K, v in c.items():
                    rows.append({"task": t, "kind": kind, "protocols": k, "K": K, "C": v})
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "curves.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    import write_curve_tables                     # tables/figure_curves.csv (every task) and its summary line
    write_curve_tables.write_one("protocol_ceiling", ROOT)
    print("\n".join(summary(R)))


if __name__ == "__main__":
    main()
