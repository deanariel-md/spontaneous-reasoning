r"""Every Cheng et al. comparison on one panel, the 13 full-precision (BF16) models of the
reproduction, with all 15 (the two FP8 70B-class checkpoints added) as a sensitivity only.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_cheng_panel.py 2>&1"

Reads results/reproduction/benefit.csv (scorer cheng-fixed/1.0.0). Part A: the published frames; Part B: the eight
transported configurations of the frozen bank. Per panel: the published claim (no strong model's benefit interval
above zero), sign agreement with the published benefits, the capability slope (ours and the published values on the
same models), group means, the slope on three scales per configuration and across the configurations as draws, and
the map point. Outputs: results/cheng_panel/cheng_panel.json (macros, prefix cpn), summary.md, summary_clean.md.
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
from src import instability as ins, reliability_map as rm, reproduction as rp  # noqa: E402
from run_index_checks import fmt, write  # noqa: E402

BEN = os.path.join(ROOT, "results", "reproduction", "benefit.csv")
PANELS = {"bf16": ("BF16 models (main panel)", "Bf"), "all": ("all 15 models (sensitivity)", "All")}
SC_TAG = {"points": "Points", "logit": "Logit", "error_reduction": "ErrRed"}


def signs(g):
    return {"agree": int((g["sign_agrees"] == True).sum()), "disagree": int((g["sign_agrees"] == False).sum()),  # noqa: E712
            "tie_or_undefined": int(g["sign_agrees"].isna().sum())}


def group_means(g, col_b="benefit", col_z="zero_shot"):
    return {k: {"n": int(len(x)), "benefit": float(x[col_b].mean()), "zero_shot": float(x[col_z].mean())}
            for k, x in g.groupby("group")}


def part_a(a):
    strong = a.loc[a["group"] == "strong", "model_key"].tolist()
    return {"models": sorted(a["model_key"]), "claim": rp.claim_holds(a, strong), "signs": signs(a),
            "slope": rp.capability_slope(a["benefit"], a["zero_shot"]),
            "published_slope": rp.capability_slope(a["published_benefit"], a["published_zero_shot"]),
            "groups": group_means(a), "published_groups": group_means(a, "published_benefit", "published_zero_shot")}


def part_b(t, n_items):
    strong = t.loc[t["group"] == "strong", "model_key"].unique().tolist()
    configs = sorted(t["config_id"].unique())
    models = sorted(t["model_key"].unique())
    w = t.pivot(index="config_id", columns="model_key")
    B = w["zero_shot"].loc[configs, models].to_numpy(float)
    P = w["eight_shot"].loc[configs, models].to_numpy(float)
    per = {}
    for cid, g in t.groupby("config_id", sort=True):
        per[cid] = {"claim": rp.claim_holds(g, strong), "signs": signs(g),
                    "slope": rp.capability_slope(g["benefit"], g["zero_shot"])}
    scales = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(B, P, sc, n_items)
        per_cfg = [rp.capability_slope(s["y"][k], s["x"][k])["slope"] for k in range(len(configs))]
        m = ins.scale_moments(s["x"], s["y"], "jack_t")
        scales[sc] = {"per_configuration": dict(zip(configs, per_cfg)), "mean": float(np.mean(per_cfg)),
                      "positive": int(sum(x > 0 for x in per_cfg)), "negative": int(sum(x < 0 for x in per_cfg)),
                      "cross_draw": {k: m[k] for k in ("slope", "slope_set", "slope_set_sign", "R_benefit")},
                      "replaced_cells": s["n_replaced"]}
    pt = rm.setting_point(B, P)["benefit"]
    eq = rp.capability_slope((P - B).mean(0), B.mean(0))
    return {"configurations": configs, "models": models, "per_configuration": per,
            "claim_holds_in": int(sum(v["claim"]["holds"] for v in per.values())), "of": len(per),
            "signs_agree_mean": float(np.mean([v["signs"]["agree"] for v in per.values()])),
            "scales": scales, "equal_budget_slope": eq, "groups": group_means(t),
            "map_point": {k: pt[k] for k in ("R", "signal", "noise", "C", "C_ref", "departure")},
            "mean_gain": float((P - B).mean())}


def run():
    t0 = time.time()
    ben = pd.read_csv(BEN)
    ben["sign_agrees"] = ben["sign_agrees"].map({True: True, False: False, "True": True, "False": False})
    n_items = int(ben["n_items"].max())
    R = {"analysis": "the Cheng et al. comparisons on one panel", "methods": "Cheng et al. comparisons on one panel",
         "created": time.strftime("%Y-%m-%d %H:%M"), "source": "results/reproduction/benefit.csv", "n_items": n_items,
         "panels": {}}
    for p in PANELS:
        b = ben if p == "all" else ben[ben["precision"] == "bf16"]
        R["panels"][p] = {"M": int(b["model_key"].nunique()),
                          "fp8_models": sorted(ben.loc[ben["precision"] != "bf16", "model_key"].unique()),
                          "part_a": part_a(b[b["part"] == "numerical"]),
                          "part_b": part_b(b[b["part"] == "transport"], n_items)}
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write("cheng_panel", R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/cheng_panel", flush=True)


def macros(R):
    m = {}
    for p, (_, t) in PANELS.items():
        x = R["panels"][p]
        a, b = x["part_a"], x["part_b"]
        s = f"cpn{t}"
        m[s + "M"] = str(x["M"])
        m[s + "AClaim"] = "holds" if a["claim"]["holds"] else "fails"
        m[s + "ASignsAgree"], m[s + "ASignsDisagree"] = str(a["signs"]["agree"]), str(a["signs"]["disagree"])
        m[s + "ASignsTie"] = str(a["signs"]["tie_or_undefined"])
        m[s + "ASlope"], m[s + "APublishedSlope"] = fmt(a["slope"]["slope"], 3), fmt(a["published_slope"]["slope"], 3)
        for g, gt in (("weak", "Weak"), ("strong", "Strong")):
            if g in a["groups"]:
                m[f"{s}A{gt}N"] = str(a["groups"][g]["n"])
                m[f"{s}A{gt}Benefit"] = fmt(a["groups"][g]["benefit"], 2)
                m[f"{s}A{gt}PublishedBenefit"] = fmt(a["published_groups"][g]["benefit"], 2)
                m[f"{s}B{gt}Benefit"] = fmt(b["groups"][g]["benefit"], 2)
        m[s + "BClaimHolds"], m[s + "BConfigs"] = str(b["claim_holds_in"]), str(b["of"])
        m[s + "BSignsAgreeMean"] = fmt(b["signs_agree_mean"], 1)
        m[s + "BEqualBudgetSlope"] = fmt(b["equal_budget_slope"]["slope"], 3)
        for sc, st in SC_TAG.items():
            v = b["scales"][sc]
            m[f"{s}BSlope{st}Mean"] = fmt(v["mean"], 3)
            m[f"{s}BSlope{st}Pos"], m[f"{s}BSlope{st}Neg"] = str(v["positive"]), str(v["negative"])
            pc = list(v["per_configuration"].values())
            m[f"{s}BSlope{st}Min"], m[f"{s}BSlope{st}Max"] = fmt(min(pc), 3), fmt(max(pc), 3)
            cd = v["cross_draw"]
            m[f"{s}BSlope{st}Cross"] = fmt(cd["slope"], 3)
            ss = cd["slope_set"]
            if ss["kind"] == "bounded":
                m[f"{s}BSlope{st}CrossLo"], m[f"{s}BSlope{st}CrossHi"] = fmt(ss["a"], 3), fmt(ss["b"], 3)
        mp = b["map_point"]
        m[s + "BMapR"], m[s + "BMapC"], m[s + "BMapCref"] = fmt(mp["R"], 3), fmt(mp["C"], 3), fmt(mp["C_ref"], 3)
        m[s + "BVdelta"], m[s + "BW"] = fmt(mp["signal"], 2), fmt(mp["noise"], 2)
        m[s + "BMeanGain"] = fmt(b["mean_gain"], 2)
    return m


def summary(R):
    L = ["# The Cheng et al. comparisons on one panel", "",
         "The main panel is the 13 BF16 models of the reproduction; "
         f"the two FP8 checkpoints ({', '.join(R['panels']['bf16']['fp8_models'])}) enter only the sensitivity column. "
         "Benefit is eight-shot minus zero-shot chain-of-thought accuracy (pp); the claim is the protocol's rule (no "
         "strong model's benefit interval above zero). Cross-configuration slopes treat "
         "the eight configurations as draws (jackknife t set, descriptive).", "",
         "| Quantity | " + " | ".join(PANELS[p][0] for p in PANELS) + " |", "|---|---|---|"]

    def row(lab, f):
        L.append(f"| {lab} | " + " | ".join(f(R["panels"][p]) for p in PANELS) + " |")
    row("Models", lambda x: str(x["M"]))
    row("Part A: claim", lambda x: "holds" if x["part_a"]["claim"]["holds"] else "fails")
    row("Part A: signs agree / disagree / tie", lambda x: "{agree} / {disagree} / {tie_or_undefined}".format(**x["part_a"]["signs"]))
    row("Part A: capability slope, ours (published values)",
        lambda x: f"{fmt(x['part_a']['slope']['slope'], 3)} ({fmt(x['part_a']['published_slope']['slope'], 3)})")
    for g in ("weak", "strong"):
        row(f"Part A: mean benefit, {g} models, ours (published)",
            lambda x, g=g: f"{fmt(x['part_a']['groups'][g]['benefit'], 2)} ({fmt(x['part_a']['published_groups'][g]['benefit'], 2)})")
    row("Part B: claim holds in", lambda x: f"{x['part_b']['claim_holds_in']} of {x['part_b']['of']}")
    row("Part B: mean models with the published sign", lambda x: fmt(x["part_b"]["signs_agree_mean"], 1))
    for g in ("weak", "strong"):
        row(f"Part B: mean benefit, {g} models", lambda x, g=g: fmt(x["part_b"]["groups"][g]["benefit"], 2))
    for sc in ins.SCALES:
        row(f"Part B: slope ({sc}), mean of 8 [positive / negative]",
            lambda x, sc=sc: f"{fmt(x['part_b']['scales'][sc]['mean'], 3)} [{x['part_b']['scales'][sc]['positive']} / "
                             f"{x['part_b']['scales'][sc]['negative']}]")
        row(f"Part B: slope ({sc}), configurations as draws",
            lambda x, sc=sc: (lambda cd: f"{fmt(cd['slope'], 3)} [{fmt(cd['slope_set']['a'], 3)}, "
                                         f"{fmt(cd['slope_set']['b'], 3)}]")(x["part_b"]["scales"][sc]["cross_draw"]))
    row("Part B: equal-budget slope (points)", lambda x: fmt(x["part_b"]["equal_budget_slope"]["slope"], 3))
    row("Part B: R, C of benefit (C_ref)", lambda x: f"{fmt(x['part_b']['map_point']['R'], 3)}, "
                                                    f"{fmt(x['part_b']['map_point']['C'], 3)} "
                                                    f"({fmt(x['part_b']['map_point']['C_ref'], 3)})")
    row("Part B: V_delta, W_gain (pp²)", lambda x: f"{fmt(x['part_b']['map_point']['signal'], 2)}, "
                                                  f"{fmt(x['part_b']['map_point']['noise'], 2)}")
    L.append("")
    return L


if __name__ == "__main__":
    run()
