r"""The ceiling test inside GSM8K: the slope by solution length and by cross-fitted difficulty.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_gsm8k_difficulty.py 2>&1"

MATH-500 changes the task, the bank and the answer type along with the ceiling. Here the same 23 models (and all 29)
are read on GSM8K items grouped by the number of calculator steps in the gold solution (metadata), and, secondary, by
chain-of-thought difficulty defined on the discovery draws and read on the validation draws. Outputs:
results/gsm8k_difficulty/gsm8k_difficulty.json, summary.md, summary_clean.md, tables/strata.csv.
"""
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
from src import dataset2 as d2, instability as ins, robustness as rb  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "gsm8k_difficulty")
N_BOOT = 10000
SEED = 20261117
STEP_ORDER = ("<=2", "3", "4", "5", ">=6")
N_QUANTILES = 5


def stratum_arrays(load, fd_load, units, panel, groups):
    """{label: (requested, forced, cot)} arrays (draws, models) on each group's items."""
    from run_forced_direct import three_arm_arrays
    from run_math_primary import subset
    return {lab: three_arm_arrays(subset(load, keep), subset(fd_load, keep), units, panel, keep)
            for lab, keep in groups.items()}


def stratum_block(arrs, ix, method, alpha, rng=None):
    out = []
    for lab, (Breq, Bfd, Pc) in arrs.items():
        B, P, Q = Bfd[:, ix], Pc[:, ix], Breq[:, ix]
        boot = (rb.bootstrap_statistic(lambda j, B=B, P=P: rb.cross_slope(B[:, j], (P - B)[:, j])[0], len(ix), None,
                                       N_BOOT, rng) if rng is not None else None)
        g = ins.scale_moments(B, P - B, method, alpha)
        c = ins.scale_moments(B, P, method, alpha)
        r = ins.scale_moments(Q, P - Q, method, alpha)
        out.append({"stratum": lab, "requested": float(Q.mean()), "forced": float(B.mean()), "cot": float(P.mean()),
                    "slope_cot_minus_forced": g["slope"], "set": g["slope_set"], "set_sign": g["slope_set_sign"],
                    "ceiling_slope": c["slope"], "ceiling_set": c["slope_set"],
                    "slope_cot_minus_requested": r["slope"], "set_requested": r["slope_set"],
                    "set_sign_requested": r["slope_set_sign"], "model_bootstrap": boot})
    return out


def trend_block(arrs, ix, alpha, rng, families):
    from run_math_primary import trend
    lv = [(Bfd[:, ix], Pc[:, ix]) for _, Bfd, Pc in arrs.values()]
    t = trend(lv, alpha)
    stat = lambda j: rb.least_squares_trend([rb.cross_slope(B[:, j], (P - B)[:, j])[0] for B, P in lv])  # noqa: E731
    t["model_bootstrap"] = rb.bootstrap_statistic(stat, len(ix), None, N_BOOT, rng)
    t["family_bootstrap"] = rb.bootstrap_statistic(stat, len(ix), families, N_BOOT, rng)
    return t


def run():
    t0 = time.time()
    from run_forced_direct import collection_inputs, forced_inputs
    from run_outcomes import CONFIG, SELECTION
    from src import d2_io
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    rng = np.random.default_rng(SEED)
    inp, s, items, items_info = collection_inputs("gsm8k")
    panel = list(inp["panel"])
    fams = list(inp["families"])
    fd_load, _, _ = forced_inputs("gsm8k", panel, s, items_info, inp["load"], False)
    disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
    readable = ri.gsm8k_readable_panel()
    panels = {"readable": [panel.index(k) for k in readable], "full": list(range(len(panel)))}

    # strata by solution steps (metadata)
    df, _ = d2_io.load_items(split="test")
    steps = dict(zip(df["item_id"], [rb.gsm8k_steps(t) for t in df["gold_text"]]))
    lab = dict(zip(df["item_id"], rb.step_bins([steps[i] for i in df["item_id"]])))
    groups = {b: [it for it in items if lab[it] == b] for b in STEP_ORDER}
    arrs = stratum_arrays(inp["load"], fd_load, disc, panel, groups)
    print(f"step strata built ({time.time() - t0:.0f} s)", flush=True)
    R = {"analysis": "slope inside GSM8K by solution steps and cross-fitted difficulty",
         "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method, "n_boot": N_BOOT, "seed": SEED,
         "steps": {"n_items": {b: len(v) for b, v in groups.items()}, "panels": {}}}
    for pname, ix in panels.items():
        R["steps"]["panels"][pname] = {"M": len(ix), "strata": stratum_block(arrs, ix, method, alpha, rng),
                                       "trend": trend_block(arrs, ix, alpha, rng, [fams[i] for i in ix])}

    # cross-fitted difficulty: defined on discovery chain of thought (29 models), read on validation draws
    col = inp["load"](disc)
    cot = col[col["arm"] == "zero_shot_cot"]
    diff = cot.groupby("item_id")["strict_correct"].mean().reindex(items)
    q = pd.qcut(diff.rank(method="first"), N_QUANTILES, labels=[f"Q{i + 1}" for i in range(N_QUANTILES)])
    # items whose solve rate also occurs in the neighbouring quintile (split by item order)
    edge_ties = 0
    for i in range(1, N_QUANTILES):
        a, b = diff[q == f"Q{i}"], diff[q == f"Q{i + 1}"]
        common = set(a) & set(b)
        edge_ties += int(a.isin(common).sum() + b.isin(common).sum())
    sel = {"validation_draws_repeating_a_discovery_configuration":
           int(sum(1 for u in val if str(u) in set(map(str, disc)))), "items_tied_across_a_quintile_edge": edge_ties}
    # Q1 is the hardest fifth (lowest chain-of-thought solve rate); order hard -> easy is reversed for the trend so
    # the trend runs from easy to hard, as the step strata do
    qgroups = {f"Q{i}": [it for it in items if q[it] == f"Q{i}"] for i in range(N_QUANTILES, 0, -1)}
    qarrs = stratum_arrays(inp["load"], fd_load, val, panel, qgroups)
    R["crossfit"] = {"definition": "quintiles of mean discovery-draw chain-of-thought accuracy over the 29 models; Q1 "
                                   "hardest; slopes on the 24 validation draws; strata listed easy to hard",
                     "solve_rate_range": {k: [float(diff[v].min()), float(diff[v].max())] for k, v in qgroups.items()},
                     "selection_notes": sel,
                     "panels": {}}
    for pname, ix in panels.items():
        R["crossfit"]["panels"][pname] = {"M": len(ix), "strata": stratum_block(qarrs, ix, method, alpha, rng),
                                          "trend": trend_block(qarrs, ix, alpha, rng, [fams[i] for i in ix])}
    R["runtime_seconds"] = round(time.time() - t0, 1)
    rows = []
    for kind in ("steps", "crossfit"):
        for pname, P in R[kind]["panels"].items():
            for r in P["strata"]:
                rows.append({"strata": kind, "panel": pname, **{k: v for k, v in r.items() if not isinstance(v, dict)},
                             "lo": r["set"]["a"], "hi": r["set"]["b"],
                             "boot_lo": (r["model_bootstrap"]["ci95"] or [None, None])[0],
                             "boot_hi": (r["model_bootstrap"]["ci95"] or [None, None])[1],
                             "boot_share_below_zero": r["model_bootstrap"]["share_below_zero"],
                             "boot_undefined": r["model_bootstrap"]["n_undefined"]})
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "strata.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/gsm8k_difficulty", flush=True)


def _set(s):
    return f"[{rb.fmt(s['a'])}, {rb.fmt(s['b'])}]" if s["kind"] == "bounded" else s["kind"]


def _ci(b):
    c = b["ci95"]
    return "undefined" if c is None else f"[{rb.fmt(c[0])}, {rb.fmt(c[1])}]"


def summary(R):
    L = ["# Slope inside GSM8K", "",
         "Slope of chain of thought − forced on forced accuracy, both on "
         "the stratum's items (cross-draw moments, pp per pp), with its draw set; the ceiling check is the slope of "
         "chain-of-thought accuracy on forced accuracy (the panel is called off the ceiling when its set lies above "
         "0.6). Trend: least-squares slope of the stratum slopes over stratum order (easy to hard), with the "
         "delete-one-draw jackknife and a model bootstrap.", ""]
    for kind, title in (("steps", "Strata by calculator steps in the gold solution (24 discovery draws)"),
                        ("crossfit", "Strata by cross-fitted difficulty (defined on discovery, read on 24 validation draws)")):
        L += [f"## {title}", ""]
        if kind == "steps":
            L += ["Items per stratum: " + ", ".join(f"{k}: {v}" for k, v in R["steps"]["n_items"].items()) + ".", ""]
        else:
            sn = R["crossfit"]["selection_notes"]
            L += ["These strata are selected on the same models' chain-of-thought outcome, so they keep part of the "
                  "selection they are meant to remove: item-by-model failures persist across prompts, and "
                  f"{sn['validation_draws_repeating_a_discovery_configuration']} of the 24 validation draws repeat a "
                  "discovery configuration (sampling with replacement). "
                  f"{sn['items_tied_across_a_quintile_edge']} items share a solve rate with an item across a quintile "
                  "edge and were split by item order. Read them as secondary to the step strata.", ""]
            L += ["Chain-of-thought solve rate on discovery draws by stratum: " +
                  ", ".join(f"{k}: {rb.fmt(a, 2)}–{rb.fmt(b, 2)}" for k, (a, b) in R["crossfit"]["solve_rate_range"].items()) + ".", ""]
        for pname, P in R[kind]["panels"].items():
            L += [f"**Panel: {pname} ({P['M']} models).**", "",
                  "| Stratum | Requested | Forced | Chain of thought | Slope (− forced) | Draw set | Model bootstrap | "
                  "Share < 0 | Ceiling slope | Slope (− requested) |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in P["strata"]:
                L.append(f"| {r['stratum']} | {rb.fmt(r['requested'], 1)} | {rb.fmt(r['forced'], 1)} | {rb.fmt(r['cot'], 1)} | "
                         f"{rb.fmt(r['slope_cot_minus_forced'])} | {_set(r['set'])} | {_ci(r['model_bootstrap'])} | "
                         f"{rb.fmt(r['model_bootstrap']['share_below_zero'])} | {rb.fmt(r['ceiling_slope'])} | "
                         f"{rb.fmt(r['slope_cot_minus_requested'])} |")
            t = P["trend"]
            tci = t.get("ci95")
            L += ["", f"Trend over strata: {rb.fmt(t.get('estimate'))} per stratum"
                  + (f", jackknife [{rb.fmt(tci[0])}, {rb.fmt(tci[1])}]" if tci else " (undefined: " + t.get("note", "") + ")")
                  + f"; model bootstrap {_ci(t['model_bootstrap'])}; family bootstrap {_ci(t['family_bootstrap'])}"
                  + f" (undefined replicates: {t['model_bootstrap']['n_undefined']} and {t['family_bootstrap']['n_undefined']}).", ""]
    L += ["Table: `results/gsm8k_difficulty/tables/strata.csv`."]
    return L


if __name__ == "__main__":
    run()
