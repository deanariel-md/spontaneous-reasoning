r"""Model-level intervals beside the prompt-draw intervals of the capability slopes.

Run from the repository root:
    python scripts/run_model_resampling.py

The paper's slope sets cover prompt draws for a fixed panel. Here the same cross-draw slopes (discovery draws, strict
scoring) also get intervals over the choice of models: a model bootstrap, a family cluster bootstrap, and leave-one-
model-out and leave-one-family-out ranges. Also the paired change of the slope from GSM8K to MATH-500 on the same
23 models, and the model bootstrap of the MATH level trend and of each level's slope. The tasks of the section `tasks` of
config/added_contrasts.yaml (HotpotQA) follow, full panel only, after every other draw.
Outputs: results/model_resampling/model_resampling.json, summary.md, summary_clean.md, tables/slopes.csv, tables/math_levels.csv.
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

OUT = os.path.join(ROOT, "results", "model_resampling")
N_BOOT = 10000
SEED = 20261117
OUTCOMES = {"CotMinusForced": ("forced", "chain of thought − forced, on forced accuracy"),
            "CotMinusReq": ("requested", "chain of thought − requested, on requested accuracy")}


def slope_fn(B, P):
    def f(ix):
        return rb.cross_slope(B[:, ix], (P - B)[:, ix])[0]
    return f


def block(B, P, families, method, alpha, rng, models):
    """The slope with its draw set and the four model-level summaries."""
    M = B.shape[1]
    m = ins.scale_moments(B, P - B, method, alpha)
    f = slope_fn(B, P)
    fams = sorted(set(families))
    lomo = rb.leave_out(f, {models[j]: [k for k in range(M) if k != j] for j in range(M)})
    lofo = rb.leave_out(f, {fam: [k for k in range(M) if families[k] != fam] for fam in fams})
    boot = rb.bootstrap_statistic(f, M, None, N_BOOT, rng)
    fboot = rb.bootstrap_statistic(f, M, families, N_BOOT, rng)
    return {"M": M, "slope": m["slope"], "draw_set": m["slope_set"], "draw_set_sign": m["slope_set_sign"],
            "model_bootstrap": boot, "model_bootstrap_sign": rb.interval_sign(boot["ci95"]),
            "family_bootstrap": fboot, "family_bootstrap_sign": rb.interval_sign(fboot["ci95"]),
            "leave_one_model_out": {"min": float(np.nanmin(list(lomo.values()))), "max": float(np.nanmax(list(lomo.values()))),
                                    "argmin": min(lomo, key=lambda k: lomo[k]), "argmax": max(lomo, key=lambda k: lomo[k])},
            "leave_one_family_out": lofo, "families": {fam: families.count(fam) for fam in fams}}


def math_levels(models):
    """Per MATH level, (forced, chain of thought) arrays (discovery draws) for the given models."""
    from run_forced_direct import three_arm_arrays
    from run_math_primary import inputs, subset
    x = inputs(False)
    panel, disc = x["panel"], list(x["sampler"]["discovery_config_ids"])
    out = {}
    for lev in ("1", "2", "3", "4", "5"):
        keep = [it for it in x["items"] if x["levels"][str(it)] == lev]
        _, Bfd, Pc = three_arm_arrays(subset(x["inp"]["load"], keep), subset(x["fd_load"], keep), disc, panel, keep)
        ix = [panel.index(k) for k in models]
        out[lev] = (Bfd[:, ix], Pc[:, ix])
    return out


def level_trend(levels, ix):
    return rb.least_squares_trend([rb.cross_slope(B[:, ix], (P - B)[:, ix])[0] for B, P in levels.values()])


def level_model_bootstrap(levels, n_models, rng, n_boot=N_BOOT):
    """Model bootstrap of each level's slope (one block per level, in level order). Called after every other draw of the
    run, so the existing statistics keep the random numbers they had before these blocks were added."""
    out = []
    for lev, (B, P) in levels.items():
        b = rb.bootstrap_statistic(slope_fn(B, P), n_models, None, n_boot, rng)
        out.append({"level": int(lev), **b})
    return out


def math_levels_table(R):
    """Rows of tables/math_levels.csv: the level slope with its model-bootstrap interval and share below zero."""
    t = R["math_level_trend"]
    rows = []
    for slope, b in zip(t["level_slopes"], t["level_model_bootstrap"]):
        ci = b["ci95"] or [None, None]
        rows.append({"level": b["level"], "slope": slope, "model_lo": ci[0], "model_hi": ci[1],
                     "model_share_below_zero": b["share_below_zero"]})
    return rows


def slope_row(task, oc, pname, b):
    """One row of tables/slopes.csv."""
    return {"task": task, "outcome": oc, "panel": pname, "M": b["M"], "slope": b["slope"],
            "draw_lo": b["draw_set"]["a"], "draw_hi": b["draw_set"]["b"],
            "model_lo": (b["model_bootstrap"]["ci95"] or [None, None])[0],
            "model_hi": (b["model_bootstrap"]["ci95"] or [None, None])[1],
            "model_share_below_zero": b["model_bootstrap"]["share_below_zero"],
            "family_lo": (b["family_bootstrap"]["ci95"] or [None, None])[0],
            "family_hi": (b["family_bootstrap"]["ci95"] or [None, None])[1],
            "lomo_min": b["leave_one_model_out"]["min"], "lomo_max": b["leave_one_model_out"]["max"],
            "lofo_min": float(np.nanmin(list(b["leave_one_family_out"].values()))),
            "lofo_max": float(np.nanmax(list(b["leave_one_family_out"].values())))}


def added_task_blocks(task, method, alpha):
    """An added task (config/added_contrasts.yaml, section tasks; HotpotQA): both outcomes on
    the full panel (no readable panel is defined for it), each block on a fresh generator seeded with SEED, as
    scripts/run_hotpot.py seeds its capability slopes, so the two agree and the main tasks' draws are untouched."""
    x = ri.load(task)
    f, d = x["facts"], x["discovery"]
    T = {"name": ri.NAMES[task], "added_task": True, "readable_rule": "none defined for an added task",
         "readable_models": None, "blocks": {}}
    for oc, (arm, _) in OUTCOMES.items():
        T["blocks"][f"{oc}/full"] = block(d[arm], d["cot"], list(f["families"]), method, alpha,
                                          np.random.default_rng(SEED), f["panel"])
    return T


def run():
    t0 = time.time()
    from run_outcomes import CONFIG, SELECTION
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    rng = np.random.default_rng(SEED)
    readable23 = ri.gsm8k_readable_panel()
    R = {"analysis": "model-level intervals of the capability slopes",
         "created": time.strftime("%Y-%m-%d %H:%M"), "n_boot": N_BOOT, "seed": SEED, "slope_method": method,
         "tasks": {}}
    rows, X = [], {}
    for task in ri.TASKS:
        x = ri.load(task)
        f, d = x["facts"], x["discovery"]
        X[task] = x
        panel, fams = f["panel"], f["families"]
        if task == "bbh":
            rd = [k for k, p in zip(panel, f["readable_pct"]) if p >= ri.CLEAN_MIN]
            rd_rule = "BIG-Bench Hard's own readable models (>= 95% closed and readable forced answers)"
        else:
            rd = [k for k in readable23 if k in panel]
            rd_rule = "the 23 models of the GSM8K-29 readable panel"
        T = {"name": ri.NAMES[task], "readable_rule": rd_rule, "readable_models": rd, "blocks": {}}
        for oc, (arm, label) in OUTCOMES.items():
            for pname, keep in (("full", panel), ("readable", rd)):
                ix = [panel.index(k) for k in keep]
                b = block(d[arm][:, ix], d["cot"][:, ix], [fams[i] for i in ix], method, alpha, rng, keep)
                T["blocks"][f"{oc}/{pname}"] = b
                rows.append(slope_row(task, oc, pname, b))
        R["tasks"][task] = T
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)

    # paired change of the slope, GSM8K -> MATH-500, same 23 models resampled together
    g, m = X["gsm8k"], X["math"]
    ig = [g["facts"]["panel"].index(k) for k in readable23]
    im = [m["facts"]["panel"].index(k) for k in readable23]
    Bg, Pg = g["discovery"]["forced"][:, ig], g["discovery"]["cot"][:, ig]
    Bm, Pm = m["discovery"]["forced"][:, im], m["discovery"]["cot"][:, im]
    fam23 = [g["facts"]["families"][i] for i in ig]
    change = lambda ix: rb.cross_slope(Bm[:, ix], (Pm - Bm)[:, ix])[0] - rb.cross_slope(Bg[:, ix], (Pg - Bg)[:, ix])[0]  # noqa: E731
    est = change(np.arange(23))
    R["item_g_change"] = {"estimate": est, "gsm8k": rb.cross_slope(Bg, Pg - Bg)[0], "math": rb.cross_slope(Bm, Pm - Bm)[0],
                          "model_bootstrap": rb.bootstrap_statistic(change, 23, None, N_BOOT, rng),
                          "family_bootstrap": rb.bootstrap_statistic(change, 23, fam23, N_BOOT, rng)}
    # MATH level trend under the model bootstrap
    lev = math_levels(readable23)
    tr = lambda ix: level_trend(lev, ix)  # noqa: E731
    R["math_level_trend"] = {"estimate": tr(np.arange(23)),
                             "level_slopes": [rb.cross_slope(B, P - B)[0] for B, P in lev.values()],
                             "model_bootstrap": rb.bootstrap_statistic(tr, 23, None, N_BOOT, rng),
                             "family_bootstrap": rb.bootstrap_statistic(tr, 23, fam23, N_BOOT, rng)}
    # per-level model bootstrap: last, after every other draw
    R["math_level_trend"]["level_model_bootstrap"] = level_model_bootstrap(lev, 23, rng)
    # added tasks: after every draw above, on their own generators
    for task in ri.added_tasks():
        R["tasks"][task] = added_task_blocks(task, method, alpha)
        rows += [slope_row(task, *k.split("/"), b) for k, b in R["tasks"][task]["blocks"].items()]
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "slopes.csv"), index=False)
    pd.DataFrame(math_levels_table(R)).to_csv(os.path.join(OUT, "tables", "math_levels.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R, rows))
    print(f"done in {R['runtime_seconds']} s -> results/model_resampling", flush=True)
    return R


def _dset(s):
    if s["kind"] == "bounded":
        return f"[{rb.fmt(s['a'])}, {rb.fmt(s['b'])}]"
    return f"{s['kind']} ({rb.fmt(s['a'])}, {rb.fmt(s['b'])})"


def _ci(c):
    return "undefined" if c is None else f"[{rb.fmt(c[0])}, {rb.fmt(c[1])}]"


def summary(R, rows):
    L = ["# Model-level intervals of the capability slopes", "",
         "Slope of the gain on the baseline arm's accuracy (cross-draw "
         "moments, 24 discovery draws, strict scoring, pp per pp). The draw set is the paper's interval: it covers "
         f"prompt draws for a fixed panel. The model and family bootstraps ({R['n_boot']:,} resamples, seed "
         f"{R['seed']}) also cover which models enter the panel; families are resampled whole. Leave-one-out columns "
         "give the range of the slope when one model or one family is removed.", "",
         "| Task | Outcome | Panel | M | Slope | Draw set | Model bootstrap | Share < 0 | Family bootstrap | "
         "Leave one model out | Leave one family out |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b = R["tasks"][r["task"]]["blocks"][f"{r['outcome']}/{r['panel']}"]
        L.append(f"| {ri.NAMES[r['task']]} | {r['outcome']} | {r['panel']} | {r['M']} | {rb.fmt(r['slope'])} | "
                 f"{_dset(b['draw_set'])} | {_ci(b['model_bootstrap']['ci95'])} | "
                 f"{rb.fmt(b['model_bootstrap']['share_below_zero'], 3)} | "
                 f"{_ci(b['family_bootstrap']['ci95']) if len(b['families']) > 1 else 'one family'} | "
                 f"{rb.fmt(r['lomo_min'])} to {rb.fmt(r['lomo_max'])} | {rb.fmt(r['lofo_min'])} to {rb.fmt(r['lofo_max'])} |")
    und = [(t, k, b["model_bootstrap"]["n_undefined"], b["family_bootstrap"]["n_undefined"])
           for t, T in R["tasks"].items() for k, b in T["blocks"].items()
           if b["model_bootstrap"]["n_undefined"] or b["family_bootstrap"]["n_undefined"]]
    L += ["", "Undefined bootstrap replicates (V_x not positive): " +
          ("none." if not und else "; ".join(f"{t} {k}: {a} model, {c} family" for t, k, a, c in und))]
    c = R["item_g_change"]
    L += ["", "## The change from GSM8K to MATH-500 on the same 23 models", "",
          f"Slope of chain of thought − forced on forced accuracy: GSM8K {rb.fmt(c['gsm8k'])}, MATH-500 {rb.fmt(c['math'])}, "
          f"change {rb.fmt(c['estimate'])}. Model bootstrap of the change {_ci(c['model_bootstrap']['ci95'])} (share "
          f"below zero {rb.fmt(c['model_bootstrap']['share_below_zero'])}); family bootstrap "
          f"{_ci(c['family_bootstrap']['ci95'])}. Undefined replicates: {c['model_bootstrap']['n_undefined']} model, "
          f"{c['family_bootstrap']['n_undefined']} family."]
    t = R["math_level_trend"]
    L += ["", "## MATH-500: trend of the slope over levels 1–5 (23 models)", "",
          "Level slopes: " + ", ".join(rb.fmt(v) for v in t["level_slopes"]) +
          f". Trend {rb.fmt(t['estimate'])} per level; model bootstrap {_ci(t['model_bootstrap']['ci95'])}, "
          f"family bootstrap {_ci(t['family_bootstrap']['ci95'])}. Undefined replicates: "
          f"{t['model_bootstrap']['n_undefined']} model, {t['family_bootstrap']['n_undefined']} family. The family "
          "bootstrap has five families, so read it as a rough range.",
          "", "Level slopes with their model-bootstrap intervals (23 models resampled; share of resamples below zero):", "",
          "| Level | Slope | Model bootstrap | Share < 0 | Undefined |", "|---|---|---|---|---|"]
    for v, b in zip(t["level_slopes"], t["level_model_bootstrap"]):
        L.append(f"| {b['level']} | {rb.fmt(v)} | {_ci(b['ci95'])} | {rb.fmt(b['share_below_zero'], 3)} | "
                 f"{b['n_undefined']} |")
    L += ["", "Table: `results/model_resampling/tables/slopes.csv`.",
          "Figure table: `results/model_resampling/tables/math_levels.csv` (level, slope, model_lo, model_hi, "
          "model_share_below_zero)."]
    return L


if __name__ == "__main__":
    run()
