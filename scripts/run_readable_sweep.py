r"""Readability cutoff sweep of the slope on the MATH-500 collection, and the Kojima and Wei patterns on
same-generation size ladders.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_readable_sweep.py 2>&1"

Outputs: results/readable_sweep/readable_sweep.json, summary.md, summary_clean.md, tables/cutoffs.csv, tables/ladders.csv.
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

OUT = os.path.join(ROOT, "results", "readable_sweep")
CUTOFFS = (80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5)
OUTCOMES = {"CotMinusReq": "requested", "CotMinusForced": "forced"}


def cutoff_rows(X, gsm_read, method, alpha):
    rows = []
    for task in ri.TASKS:
        f, d = X[task]["facts"], X[task]["discovery"]
        own = dict(zip(f["panel"], f["readable_pct"]))
        for basis in ("gsm8k", "own"):
            if task == "bbh" and basis == "gsm8k":
                continue
            if task == "gsm8k" and basis == "own":
                continue
            read = gsm_read if basis == "gsm8k" else own
            for c in CUTOFFS:
                keep = [k for k in f["panel"] if read.get(k, -1) >= c]
                ix = [f["panel"].index(k) for k in keep]
                row = {"task": task, "readability_basis": basis, "cutoff_pct": c, "M": len(ix), "models": ";".join(keep)}
                if len(ix) >= 3:
                    B, P = d["forced"][:, ix], d["cot"][:, ix]
                    m = ins.scale_moments(B, P - B, method, alpha)
                    row.update({"slope": m["slope"], "set_kind": m["slope_set"]["kind"], "lo": m["slope_set"]["a"],
                                "hi": m["slope_set"]["b"], "set_sign": m["slope_set_sign"]})
                rows.append(row)
    return rows


def ladder_rows(X):
    rows = []
    for task in ri.TASKS:
        f, d = X[task]["facts"], X[task]["discovery"]
        lp = dict(zip(f["panel"], np.log2(f["params_b"])))
        readable = {k for k, p in zip(f["panel"], f["readable_pct"]) if p >= ri.CLEAN_MIN}
        variants = [("primary", dict()), ("qwen3_mixtures", dict(include_qwen3_mixtures=True)),
                    ("dense", dict(dense_only=True)), ("dense_readable", dict(dense_only=True, keep=readable))]
        for variant, kw in variants:
            lad = rb.ladders(f["repo"], f["params_b"], f["panel"], **kw)
            for gen, keys in lad.items():
                if variant == "qwen3_mixtures" and gen != "Qwen3":
                    continue
                ix = [f["panel"].index(k) for k in keys]
                x = np.array([lp[k] for k in keys])
                for oc, arm in OUTCOMES.items():
                    if variant == "dense_readable" and oc != "CotMinusForced":
                        continue
                    G = d["cot"][:, ix] - d[arm][:, ix]
                    kj = rb.jack_mean(rb.per_draw_ols_slope(G, x))
                    small = rb.jack_mean(G[:, 0])
                    rows.append({"task": task, "variant": variant,
                                 "ladder": gen + (" with 30B-A3B mixtures" if variant == "qwen3_mixtures" else ""),
                                 "doublings_spanned": float(x.max() - x.min()),
                                 "models": ";".join(keys), "n_models": len(keys), "outcome": oc,
                                 "kojima_slope_pp_per_doubling": kj["estimate"], "kojima_lo": kj["ci95"][0],
                                 "kojima_hi": kj["ci95"][1], "kojima_sign": kj["sign"],
                                 "kojima_draws_same_sign": kj["draws_same_sign"],
                                 "smallest_model": keys[0], "smallest_gain": small["estimate"],
                                 "smallest_lo": small["ci95"][0], "smallest_hi": small["ci95"][1],
                                 "wei_smallest_loses": small["sign"] == "below zero"})
    return rows


def run():
    t0 = time.time()
    from run_outcomes import CONFIG, SELECTION
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    X = {t: ri.load(t) for t in ri.TASKS}
    g = X["gsm8k"]["facts"]
    gsm_read = dict(zip(g["panel"], g["readable_pct"]))
    cut = cutoff_rows(X, gsm_read, method, alpha)
    lad = ladder_rows(X)
    R = {"analysis": "readability cutoff sweep and same-generation ladders",
         "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method, "cutoffs": cut, "ladders": lad,
         "runtime_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(cut).to_csv(os.path.join(OUT, "tables", "cutoffs.csv"), index=False)
    pd.DataFrame(lad).to_csv(os.path.join(OUT, "tables", "ladders.csv"), index=False)
    ri.write_outputs(OUT, R, summary(cut, lad))
    print(f"done in {R['runtime_seconds']} s -> results/readable_sweep", flush=True)
    return R


def summary(cut, lad):
    L = ["# Readability cutoff sweep and same-generation ladders", "",
         "Strict scoring, 24 discovery draws.", "",
         "## The slope against the readability cutoff", "",
         "Slope of chain of thought − forced on forced accuracy (pp per pp) with its draw set, on the models whose "
         "forced answers are closed and readable at least as often as the cutoff. Basis `gsm8k` uses GSM8K-29 "
         "readability (the 95% rule); `own` uses the task's own forced arm.", "",
         "| Task | Basis | Cutoff (%) | M | Slope | Draw set |", "|---|---|---|---|---|---|"]
    for r in cut:
        s = (f"{rb.fmt(r['slope'])} | " + (f"[{rb.fmt(r['lo'])}, {rb.fmt(r['hi'])}]" if r["set_kind"] == "bounded"
                                           else f"{r['set_kind']} ({rb.fmt(r['lo'])}, {rb.fmt(r['hi'])})")
             if "slope" in r else "fewer than 3 models | ")
        L.append(f"| {ri.NAMES[r['task']]} | {r['readability_basis']} | {r['cutoff_pct']:g} | {r['M']} | {s} |")
    L += ["", "## Kojima et al. and Wei et al. on same-generation ladders", "",
          "Ladders: one family and one generation with at least two parameter counts (`src/robustness.ladders`). "
          "Kojima: least-squares slope of the gain on log2 parameters within the ladder, per draw, averaged over "
          "draws, with the delete-one-draw jackknife t interval (pp per doubling). Wei: the smallest model's gain with "
          "its jackknife interval; 'loses' when the interval lies below zero.", "",
          "| Task | Variant | Ladder | Models | Outcome | Slope per doubling | Interval | Smallest model | Its gain | Interval |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in lad:
        L.append(f"| {ri.NAMES[r['task']]} | {r['variant']} | {r['ladder']} | {r['n_models']} | {r['outcome']} | "
                 f"{rb.fmt(r['kojima_slope_pp_per_doubling'], 2)} | [{rb.fmt(r['kojima_lo'], 2)}, {rb.fmt(r['kojima_hi'], 2)}] | "
                 f"{r['smallest_model']} | {rb.fmt(r['smallest_gain'], 2)} | [{rb.fmt(r['smallest_lo'], 2)}, {rb.fmt(r['smallest_hi'], 2)}] |")
    df = pd.DataFrame(lad)
    names = {"primary": "Primary ladders (Qwen3 without its mixtures; Qwen3.5 and Qwen3.6 pair a dense model with a "
                        "mixture)", "dense": "Dense models only (every mixture dropped)",
             "dense_readable": "Dense models with readable forced answers (task's own forced arm >= 95%), "
                               "chain of thought − forced only"}
    for v, title in names.items():
        L += ["", f"Counts, {title}:", ""]
        for (t, oc), x in df[df["variant"] == v].groupby(["task", "outcome"]):
            L.append(f"- {ri.NAMES[t]}, {oc}: gain rises with size (interval above zero) in "
                     f"{int((x['kojima_sign'] == 'above zero').sum())} of {len(x)} ladders "
                     f"({', '.join(x.loc[x['kojima_sign'] == 'above zero', 'ladder']) or 'none'}), falls in "
                     f"{int((x['kojima_sign'] == 'below zero').sum())}; the smallest model loses in "
                     f"{int(x['wei_smallest_loses'].sum())}.")
    L += ["", "Five of the primary ladders have two models; a two-model ladder's slope is one pair difference divided "
              "by the doublings between them (column `doublings_spanned` in the table; Qwen3.6 spans 0.37)."]
    L += ["", "Tables: `results/readable_sweep/tables/cutoffs.csv`, `tables/ladders.csv`."]
    return L


if __name__ == "__main__":
    run()
