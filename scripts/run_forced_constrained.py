r"""Analysis of the constrained forced-direct arm on BIG-Bench Hard-28.

PowerShell (from the repository root), after scripts\score_forced_constrained.py --score:
    .\.venv\Scripts\python.exe scripts\run_forced_constrained.py [--quick]
    .\.venv\Scripts\python.exe scripts\run_forced_constrained.py --dry-run --quick   # the requested arm stands in

On the 24 discovery draws, the constrained arm beside the requested and forced (unconstrained) direct arms and chain of
thought:
- per model: the four accuracies, constrained - forced, and the share of closed and readable answers of each forced arm;
- for chain of thought minus each direct arm, the cross-draw slope of the gain on that arm's accuracy with its draw set,
  and the model-level summaries (model bootstrap, family bootstrap, leave-one-model-out,
  leave-one-family-out; 10,000 resamples, numpy.random.default_rng(20261117)), on all 28 models and, for the forced
  arms, on the models with at least 95% closed and readable answers;
- Steps 1-3 of the main analysis (scripts/run_dataset2.py) with the constrained arm as the baseline of chain of thought:
  the rules are saved to manifests/predictions_forced_constrained_bbh.json before the validation draws are built.
Outputs: results/forced_constrained/forced_constrained.json, summary.md, summary_clean.md, tables/per_model.csv and
tables_forced_constrained_bbh/. A dry run writes under results/forced_constrained/dry_run/, its frozen file included.
"""
from __future__ import annotations

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

GC_PROCESSED = os.path.join(ROOT, "data", "processed", "forced_constrained")
OUT = os.path.join(ROOT, "results", "forced_constrained")
MANIFESTS = os.path.join(ROOT, "manifests")
LABEL = "forced_constrained_bbh"
CLEAN_MIN = 95.0
SEED = 20261117
ARMS = {"requested": "requested direct", "forced": "forced direct", "constrained": "constrained forced direct"}


# ============================================================================ pure helpers (tested)


def readable_models(panel, rows, key="closed_and_readable_pct", cut=CLEAN_MIN):
    """The models whose share of closed and readable answers is at least cut (percent)."""
    by = {r["model_key"]: r.get(key) for r in rows}
    return [k for k in panel if by.get(k) is not None and by[k] >= cut]


def per_model_table(panel, D, fd_rows, gc_rows):
    """One row per model: the four accuracies (pp) over the discovery draws, constrained - forced, readability."""
    fdr, gcr = {r["model_key"]: r for r in fd_rows}, {r["model_key"]: r for r in gc_rows}
    out = []
    for j, k in enumerate(panel):
        r = {"model_key": k, **{a: float(D[a][:, j].mean()) for a in ("requested", "forced", "constrained", "cot")}}
        r["constrained_minus_forced"] = r["constrained"] - r["forced"]
        for name, src in (("forced", fdr), ("constrained", gcr)):
            for c in ("closed_and_readable_pct", "forced_compliant_pct", "forced_finish_length_pct"):
                r[f"{name}_{c.replace('forced_', '')}"] = src.get(k, {}).get(c)
        out.append(r)
    return out


def slope_blocks(D, panel, families, method, alpha, panels):
    """{outcome: {panel name: block}} of run_model_resampling.block; a fresh generator per block."""
    from run_model_resampling import block
    out = {}
    for arm in ARMS:
        out[f"cot_minus_{arm}"] = {}
        for name, models in panels[arm].items():
            ix = [panel.index(k) for k in models]
            if len(ix) < 3:
                out[f"cot_minus_{arm}"][name] = {"M": len(ix), "models": models, "note": "fewer than three models"}
                continue
            b = block(D[arm][:, ix], D["cot"][:, ix], [families[i] for i in ix], method, alpha,
                      np.random.default_rng(SEED), models)
            out[f"cot_minus_{arm}"][name] = {"models": models, **b}
    return out


# ============================================================================ inputs


def inputs(dry_run: bool):
    import run_forced_direct as rfd
    inp, s, items, items_info = rfd.collection_inputs("bbh")
    panel = list(inp["panel"])
    fd_load, fd_prov, fd_box = rfd.forced_inputs("bbh", panel, s, items_info, inp["load"], False)
    saved = rfd.PROCESSED
    rfd.PROCESSED = GC_PROCESSED
    try:
        gc_load, gc_prov, gc_box = rfd.forced_inputs("bbh", panel, s, items_info, inp["load"], dry_run)
    finally:
        rfd.PROCESSED = saved
    return {"inp": inp, "sampler": s, "items": items, "panel": panel, "fd": (fd_load, fd_prov, fd_box),
            "gc": (gc_load, gc_prov, gc_box)}


# ============================================================================ run


def run(a) -> dict:
    t0 = time.time()
    import run_dataset2
    import run_forced_direct as rfd
    from run_outcomes import CONFIG, SELECTION
    from src import dataset2 as d2
    out_dir = os.path.join(a.out, "dry_run") if a.dry_run else a.out
    pred_dir = os.path.join(out_dir, "frozen") if a.dry_run else a.pred_dir
    if not a.dry_run and os.path.abspath(pred_dir) != os.path.abspath(MANIFESTS) and not a.allow_other_pred_dir:
        raise SystemExit("the rules of a real run go to manifests/")
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)
    x = inputs(a.dry_run)
    inp, s, items, panel = x["inp"], x["sampler"], x["items"], x["panel"]
    families = list(inp["families"])
    disc = list(s["discovery_config_ids"])
    (fd_load, fd_prov, fd_box), (gc_load, gc_prov, gc_box) = x["fd"], x["gc"]
    Breq, Bfd, Pc = rfd.three_arm_arrays(inp["load"], fd_load, disc, panel, items)
    _, Bgc, Pc2 = rfd.three_arm_arrays(inp["load"], gc_load, disc, panel, items)
    if not np.array_equal(Pc, Pc2):
        raise SystemExit("the chain-of-thought arm differs between the two tables")
    D = {"requested": Breq, "forced": Bfd, "constrained": Bgc, "cot": Pc}
    fd_rows = rfd.per_model(panel, Breq, Bfd, Pc, fd_load(disc), fd_box)
    gc_rows = rfd.per_model(panel, Breq, Bgc, Pc, gc_load(disc), gc_box)
    rows = per_model_table(panel, D, fd_rows, gc_rows)
    panels = {"requested": {"all": panel},
              "forced": {"all": panel, "readable": readable_models(panel, fd_rows)},
              "constrained": {"all": panel, "readable": readable_models(panel, gc_rows)}}
    slopes = slope_blocks(D, panel, families, method, alpha, panels)
    print(f"discovery arrays and slopes ({time.time() - t0:.0f} s)", flush=True)

    def combined(ids):
        col = inp["load"](ids)
        return pd.concat([gc_load(ids).assign(arm="direct_answer"), col[col["arm"] == "zero_shot_cot"]],
                         ignore_index=True)
    prov = {"collection": inp["provenance"], "constrained": gc_prov, "forced": fd_prov,
            "baseline_arm": "constrained forced direct (relabelled direct_answer inside the analysis)"}
    os.makedirs(out_dir, exist_ok=True)
    steps = run_dataset2.run(combined, s, panel, families, inp["tiers"], items, out_dir,
                             os.path.join(pred_dir, f"predictions_{LABEL}.json"), LABEL, n_boot, n_ctrl,
                             overwrite_frozen=a.overwrite_frozen, provenance=prov,
                             tables_dir=os.path.join(out_dir, f"tables_{LABEL}"))
    from run_index_checks import jsonable
    R = {"analysis": "constrained forced-direct arm on BIG-Bench Hard-28",
         "dry_run": bool(a.dry_run), "created": time.strftime("%Y-%m-%d %H:%M"), "units": "percentage points",
         "panel": panel, "families": families, "slope_method": method, "alpha": alpha, "n_boot_models": 10000,
         "seed": SEED, "readable_rule": f"at least {CLEAN_MIN:g}% closed and readable forced answers",
         "panels": panels, "per_model": rows, "slopes": slopes, "steps": steps,
         "provenance": {"forced": fd_prov, "constrained": gc_prov}, "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tables", "per_model.csv"), index=False)
    with open(os.path.join(out_dir, "forced_constrained.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    txt = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return R


def f2(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{nd}f}"


def ci(c, nd=2):
    return "--" if not c else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def summary(R) -> str:
    rows = R["per_model"]
    m = {a: np.mean([r[a] for r in rows]) for a in ("requested", "forced", "constrained", "cot")}
    L = ["# Constrained forced-direct arm on BIG-Bench Hard-28" + (" -- dry run" if R["dry_run"] else ""), "",
         "Guided decoding over each item's answer set, otherwise the forced arm. 24 discovery draws; accuracies in "
         "percent, slopes in points of gain per point of the direct arm's accuracy.", "",
         f"Panel means: requested {m['requested']:.1f}, forced {m['forced']:.1f}, constrained {m['constrained']:.1f}, "
         f"chain of thought {m['cot']:.1f}.", "",
         f"Readable panels ({R['readable_rule']}): forced {len(R['panels']['forced']['readable'])} models, "
         f"constrained {len(R['panels']['constrained']['readable'])} models.", "",
         "| Outcome | Panel | M | Slope | Draw set | Model bootstrap | Family bootstrap | Leave one family out |",
         "|---|---|---:|---:|---|---|---|---|"]
    for o, blocks in R["slopes"].items():
        for name, b in blocks.items():
            if "slope" not in b:
                L.append(f"| {o} | {name} | {b['M']} | -- | {b.get('note', '')} | | | |")
                continue
            lofo = ", ".join(f"{k} {f2(v)}" for k, v in b["leave_one_family_out"].items())
            L.append(f"| {o} | {name} | {b['M']} | {f2(b['slope'])} | {b['draw_set_sign']} | "
                     f"{ci(b['model_bootstrap']['ci95'])} | {ci(b['family_bootstrap']['ci95'])} | {lofo} |")
    L += ["", "| Model | Requested | Forced | Constrained | Chain of thought | Constrained - forced | Forced readable | "
              "Constrained readable |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model_key']} | {r['requested']:.1f} | {r['forced']:.1f} | {r['constrained']:.1f} | "
                 f"{r['cot']:.1f} | {r['constrained_minus_forced']:+.1f} | "
                 f"{'--' if r['forced_closed_and_readable_pct'] is None else round(r['forced_closed_and_readable_pct'], 1)} | "
                 f"{'--' if r['constrained_closed_and_readable_pct'] is None else round(r['constrained_closed_and_readable_pct'], 1)} |")
    L += ["", f"Steps 1-3 with the constrained arm as baseline: `tables_{LABEL}/`; rules in "
              f"`manifests/predictions_{LABEL}.json`, saved before the validation draws were built."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=MANIFESTS)
    ap.add_argument("--allow-other-pred-dir", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="the requested arm stands in for the constrained arm")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
