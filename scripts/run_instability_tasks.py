r"""Analysis B on BIG-Bench Hard and the 30-model GSM8K panel: is "who
benefits more" stable across scales, panels and prompts? The GSM8K-19 and DOVE runs are in
scripts/run_instability.py; this script applies its scale, prompt and panel blocks, unchanged, to the two settings
the collections added, and fits the shape of gain against baseline.

Run from the repository root:
    python scripts/run_instability_tasks.py

Settings: BIG-Bench Hard, its 25 tasks pooled with equal weights (28 models), and GSM8K-30 (29 models). Draws: the
24 discovery and 24 validation units of manifests/sampler_bbh.json and manifests/sampler_gsm8k.json. Tiers are ECI
terciles of the 30-model extension panel (manifests/d2_panel_extension.json). Slope method, alpha, subpanel sizes,
resample counts and the seed label come from config/instability.yaml, as for GSM8K-19. The scoring variants of
run_instability.py (64-token direct answers, short direct answers, lenient extraction) are GSM8K-specific and are
not repeated here.

Outputs: results/instability_tasks/instability_tasks.json (with a "macros" object, prefix "it"), summary.md,
summary_clean.md and tables/subpanels.csv.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from src import dataset2 as d2, instability as ins  # noqa: E402
from run_instability import panel_block, prompt_block, scale_block  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LABEL = "instability_tasks"
OUT = os.path.join(ROOT, "results", LABEL)
CONFIG = os.path.join(ROOT, "config", "instability.yaml")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
NAMES = {"bbh": "BIG-Bench Hard, 25 tasks pooled", "gsm8k30": "GSM8K, 30-model panel"}
SCALE_NAME = {"points": "percentage points", "logit": "logits", "error_reduction": "relative error reduction"}


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def load_settings():
    from src import bbh_d2, d2_io, d2_panels
    s_g = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8"))
    s_b = json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
    items_g, items_info = d2_io.load_items(split="test")
    gsm = d2_panels.inputs(None, None, s_g, items_info, os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    bbh = d2_panels.bbh_inputs(None, s_b)
    reg = bbh_d2.load_registries()
    out = {}
    for name, inp, s, items_by_task in (("gsm8k30", gsm, s_g, {"gsm8k": items_g["item_id"].tolist()}),
                                        ("bbh", bbh, s_b, bbh["items_by_task"])):
        panel = list(inp["panel"])
        disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
        df = inp["load"](disc + val)
        arr = {"d": [], "v": []}
        for items in items_by_task.values():
            cells = d2.cell_table(df[df["item_id"].isin(items)], panel, disc + val, items)
            arr["d"].append(d2.unit_arrays(cells, disc, panel, "strict_correct"))
            arr["v"].append(d2.unit_arrays(cells, val, panel, "strict_correct"))
        Bd, Pd = (np.mean([a[i] for a in arr["d"]], axis=0) for i in (0, 1))
        Bv, Pv = (np.mean([a[i] for a in arr["v"]], axis=0) for i in (0, 1))
        tiers = bbh_d2.extension_tiers(reg, panel)
        n_items = sum(len(v) for v in items_by_task.values())
        out[name] = {"panel": panel, "arrays": (Bd, Pd, Bv, Pv), "tiers": tiers, "n_items": n_items,
                     "n_tasks": len(items_by_task), "families": [reg[k]["family"] for k in panel],
                     "K_discovery": len(disc), "K_validation": len(val)}
    return out


def run(a):
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    alpha, sub = float(cfg["alpha"]), cfg["subpanels"]
    method = d2.load_selection(SELECTION)["H"]
    S = load_settings()
    print(f"data loaded ({time.time() - t0:.0f} s)")
    R = {"analysis": "B on BIG-Bench Hard and GSM8K-30",
         "config": {"path": "config/instability.yaml", "sha256": sha256_file(CONFIG)}, "slope_method": method,
         "settings": {}, "scale": {}, "prompt": {}, "panel": {}, "shape": {}}
    rows = []
    for name, s in S.items():
        Bd, Pd, Bv, Pv = s["arrays"]
        n = s["n_items"]
        R["settings"][name] = {"name": NAMES[name], "models": s["panel"], "M": len(s["panel"]),
                               "families": s["families"], "tiers": s["tiers"], "n_items": n, "n_tasks": s["n_tasks"],
                               "K_discovery": s["K_discovery"], "K_validation": s["K_validation"],
                               "logit_note": "pooled accuracy over all items; a pooled score of 0 or 100 is replaced "
                                             "as in src/instability.scale_scores"}
        R["scale"][name] = scale_block(Bd, Pd, Bv, Pv, n, method, alpha)
        ref = {sc: R["scale"][name][sc]["discovery"]["slope"] for sc in ins.SCALES}
        R["prompt"][name] = prompt_block({"units": (np.vstack([Bd, Bv]), np.vstack([Pd, Pv]), ref)}, n)["units"]
        idx = {mk: i for i, mk in enumerate(s["panel"])}
        strata = {"random": list(range(len(s["panel"])))}
        if s["tiers"]:
            for tier in ("low", "middle", "high"):
                strata[f"tercile:{tier}"] = [idx[mk] for mk in s["panel"] if s["tiers"][mk] == tier]
        summ, r = panel_block(name, (Bd, Pd, Bv, Pv), n, strata, sub["sizes"], sub["n_resamples"],
                              sub["seed_label"], method, alpha)
        R["panel"][name] = summ
        rows += r
        b, g = Bd.mean(0), (Pd - Bd).mean(0)
        R["shape"][name] = {"baseline": b, "gain": g, "models": s["panel"],
                            "quadratic_coefficients": np.polyfit(b, g, 2).tolist(),
                            "note": "model means over discovery draws; the quadratic is descriptive"}
        print(f"{name} done ({time.time() - t0:.0f} s)")
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, f"{LABEL}.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "subpanels.csv"), index=False)
    write_summary(R)
    print(f"done in {R['runtime_seconds']} s -> {os.path.relpath(OUT, ROOT)}")


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def pct(x):
    return fmt(100 * x, 0)


def macros(R):
    m = {}
    tag = {"bbh": "Bbh", "gsm8k30": "GsmThirty"}
    stag = {"points": "Points", "logit": "Logit", "error_reduction": "ErrRed"}
    for name, t in tag.items():
        for sc, st in stag.items():
            d = R["scale"][name][sc]["discovery"]
            m[f"it{t}Slope{st}"] = fmt(d["slope"], 3)
            m[f"it{t}Slope{st}Lo"] = fmt(d["slope_set"]["a"], 3)
            m[f"it{t}Slope{st}Hi"] = fmt(d["slope_set"]["b"], 3)
            m[f"it{t}PromptOpp{st}"] = pct(R["prompt"][name][sc]["share_opposite"])
        rnd = R["panel"][name]["random"]["pooled_sizes"]["points"]
        m[f"it{t}PanelNeg"] = pct(rnd["share_negative"])
        m[f"it{t}PanelOwnGainMedian"] = fmt(rnd["C_own_gain_disc_val"]["median"], 2)
        m[f"it{t}H"] = fmt(R["scale"][name]["points"]["discovery"]["H"], 2)
    return m


def write_summary(R):
    L = ["# Analysis B on BIG-Bench Hard and GSM8K-30", "",
         "Chain of thought against direct answer, strict scoring. Capability is the model's direct-answer accuracy on "
         "the same draws; the slope is the cross-draw slope of benefit on capability over the 24 discovery draws, with "
         f"the {R['slope_method']} 95% set. The scale, prompt and panel blocks are those of "
         "`scripts/run_instability.py` (GSM8K-19 and DOVE in `results/instability/`).", ""]
    for name in ("gsm8k30", "bbh"):
        st = R["settings"][name]
        L += [f"## {st['name']}", "", f"{st['M']} models, {st['n_items']} items"
              + (f" ({st['n_tasks']} tasks of equal weight)" if st["n_tasks"] > 1 else "") + ".", "",
              "| Scale | Slope (discovery) | 95% set | Slope (validation) | Single draws with the opposite sign |",
              "|---|---:|---|---:|---:|"]
        for sc in ins.SCALES:
            x = R["scale"][name][sc]
            d, v, p = x["discovery"], x["validation"], R["prompt"][name][sc]
            L.append(f"| {SCALE_NAME[sc]} ({x['unit']}) | {fmt(d['slope'], 3)} | {d['slope_set']['kind']} "
                     f"[{fmt(d['slope_set']['a'], 3)}, {fmt(d['slope_set']['b'], 3)}] | {fmt(v['slope'], 3)} | "
                     f"{p['n_opposite']} of {p['n_defined']} |")
        L += ["", f"H in points (per 10 pp of baseline): {fmt(R['scale'][name]['points']['discovery']['H'])}.", "",
              "Subpanels (slope in points over the discovery draws):", "",
              "| Stratum | Size | Subpanels | Negative slope | Set below zero | Set above zero | Own-gain agreement (median) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for stratum, sm in R["panel"][name].items():
            for k, by in sm["sizes"].items():
                x = by["points"]
                L.append(f"| {stratum} | {k} | {x['n_subpanels']} ({x['mode']}) | {pct(x['share_negative'])}% | "
                         f"{pct(x['share_set_below_zero'])}% | {pct(x['share_set_above_zero'])}% | "
                         f"{fmt(x['C_own_gain_disc_val'].get('median'))} |")
        q = R["shape"][name]["quadratic_coefficients"]
        L += ["", f"Shape: the quadratic fit of model-mean gain on baseline has leading coefficient {fmt(q[0], 4)} "
                  "(descriptive).", ""]
    txt = "\n".join(L)
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
