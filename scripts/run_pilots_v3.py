r"""Pilots averaged over every pilot draw, shrunken pilots over every block with unresolved pilots
counted, the four-draw budgeting test with unresolved pilots counted, and all of it on the forced arm
Estimators: src/pilot_checks.py, src/estimators.cross_risk.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_pilots_v3.py 2>&1"

Settings: those of the map's pilot test (scripts/run_map_panels.settings_with_validation: requested direct, strict)
and, on the forced arm (chain of thought - forced direct, strict), GSM8K-29, BIG-Bench Hard pooled and its 25 tasks.
Order: every one-draw and K-draw prediction is saved to manifests/predictions_pilots_v3.json before
this run builds the Dataset 2 validation draws (earlier analyses had read them, so this is not a fresh test).
Outputs: results/pilots_v3/pilots_v3.json (macros, prefix pv), summary.md, summary_clean.md, tables/*.csv.
"""
import argparse
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
from src import dataset2 as d2, pilot_checks as pc, reliability_map as rm  # noqa: E402
from src.estimators import cross_risk  # noqa: E402
from src.sampling import sha256_json  # noqa: E402
from run_pilot_checks import jsonable, load_script, write  # noqa: E402

PRED = os.path.join(ROOT, "manifests", "predictions_pilots_v3.json")
RES = os.path.join(ROOT, "results")
BLOCKS = (2, 4)
GROUPS = {"dove_mmlu": ("DOVE MMLU block", "DoveBlock"), "dove_tasks": ("DOVE benchmarks", "Dove"),
          "gsm8k": ("GSM8K (19)", "Gsm"), "gsm8k30": ("GSM8K (29)", "GsmThirty"),
          "bbh": ("BIG-Bench Hard (28), pooled", "Bbh"), "bbh_tasks": ("BIG-Bench Hard tasks", "BbhTasks"),
          "forced_gsm8k30": ("GSM8K (29), forced arm", "FdGsmThirty"),
          "forced_bbh": ("BIG-Bench Hard (28), pooled, forced arm", "FdBbh"),
          "forced_bbh_tasks": ("BIG-Bench Hard tasks, forced arm", "FdBbhTasks")}


def c(x):
    return x - x.mean()


def requested_settings():
    mp = load_script("run_map_panels")
    out = []
    for item in mp.settings_with_validation():
        if item[0] == "d2":
            _, inp, rows = item
            for name, group, Bd, Pd, _, _ in rows:
                out.append({"name": name, "group": group, "arm": "requested", "Bp": Bd, "Pp": Pd, "inp": inp})
        else:
            name, group, Bp, Pp, Bv, Pv = item
            out.append({"name": name, "group": group, "arm": "requested", "Bp": Bp, "Pp": Pp, "Bv": Bv, "Pv": Pv})
    return out


def forced_arrays(st, units):
    """(forced, chain of thought) of every forced-arm subset of one task on the given draws."""
    from run_forced_direct import three_arm_arrays
    col, fd = st["col_load"](units), st["fd_load"](units)
    out = {}
    for name, its in st["subsets"].items():
        ix = pd.Index(its)
        _, Bfd, Pc = three_arm_arrays(lambda ids: col[col["item_id"].isin(ix)], lambda ids: fd[fd["item_id"].isin(ix)],
                                      units, st["panel"], its)
        out[name] = (Bfd, Pc)
    return out


def forced_settings():
    from run_forced_direct import collection_inputs, forced_inputs
    out, states = [], []
    for task in ("gsm8k", "bbh"):
        inp, s, items, items_info = collection_inputs(task)
        panel = list(inp["panel"])
        fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
        if task == "gsm8k":
            subsets, groups = {"forced:gsm8k30": items}, {"forced:gsm8k30": "forced_gsm8k30"}
        else:
            subsets, groups = {"forced:bbh:pooled": items}, {"forced:bbh:pooled": "forced_bbh"}
            for t, its in inp["items_by_task"].items():
                subsets[f"forced:bbh:{t}"], groups[f"forced:bbh:{t}"] = its, "forced_bbh_tasks"
        st = {"col_load": inp["load"], "fd_load": fd_load, "panel": panel, "subsets": subsets, "sampler": s}
        arr = forced_arrays(st, list(s["discovery_config_ids"]))
        for name, (B, P) in arr.items():
            out.append({"name": name, "group": groups[name], "arm": "forced", "Bp": B, "Pp": P, "state": st})
        states.append(st)
    return out, states


def rules(Bp, Pp):
    G = Pp - Bp
    K = G.shape[0]
    r = {"constant": np.zeros(G.shape[1]).tolist(), "one": [c(G[k]).tolist() for k in range(K)]}
    for k in BLOCKS:
        blocks = []
        for b in range(K // k):
            Gb = G[b * k:(b + 1) * k]
            R = pc.pilot_r(Gb)
            f = pc.shrink_factor(Gb)
            unresolved = not np.isfinite(R) or R <= 0
            p = c(Gb.mean(0))
            blocks.append({"draws": [b * k, (b + 1) * k], "R": R if np.isfinite(R) else None, "factor": f,
                           "unresolved": bool(unresolved), "pilot": p.tolist(),
                           "shrunk": ((0.0 if f is None else f) * p).tolist()})
        r[f"blocks_{k}"] = blocks
    return r


def score(s, rule):
    Z = s["Pv"] - s["Bv"]
    Z = Z - Z.mean(1, keepdims=True)
    M = Z.shape[1]
    pt = rm.setting_point(s["Bp"], s["Pp"])["benefit"]
    r0 = cross_risk(Z, np.asarray(rule["constant"], float))
    one = np.array([cross_risk(Z, np.asarray(v, float)) for v in rule["one"]])
    W = pt["noise"]
    row = {"setting": s["name"], "group": s["group"], "arm": s["arm"], "M": int(M), "K_pilot": len(rule["one"]),
           "K_validation": int(Z.shape[0]), "R": pt["R"], "W_gain": W, "V_delta": pt["signal"], "risk_constant": r0,
           "one_mean": float(one.mean()), "one_first": float(one[0]), "one_min": float(one.min()),
           "one_max": float(one.max()), "one_expected": W * (M - 1) / M,
           "one_mean_over_expected": float(one.mean()) / (W * (M - 1) / M) if W > 0 else None,
           "one_share_draws_beat": float(np.mean(one < r0)),
           "one_mean_share": float(one.mean()) / r0 if r0 > 0 else None}
    for k in BLOCKS:
        bl = rule[f"blocks_{k}"]
        rp = np.array([cross_risk(Z, np.asarray(b["pilot"], float)) for b in bl])
        rs = np.array([cross_risk(Z, np.asarray(b["shrunk"], float)) for b in bl])
        row.update({f"blocks_{k}": len(bl), f"unresolved_{k}": float(np.mean([b["unresolved"] for b in bl])),
                    f"pilot_{k}_mean": float(rp.mean()), f"shrunk_{k}_mean": float(rs.mean()),
                    f"pilot_{k}_share": float(rp.mean()) / r0 if r0 > 0 else None,
                    f"shrunk_{k}_share": float(rs.mean()) / r0 if r0 > 0 else None,
                    f"pilot_{k}_blocks_beat": float(np.mean(rp < r0)), f"shrunk_{k}_blocks_beat": float(np.mean(rs < r0))})
    return row


def budget(forced, pp):
    """The four-draw budgeting test with unresolved pilots counted: per setting, the share of all pilots that meet."""
    rows = []
    req = pd.read_csv(os.path.join(RES, "pilot_promise", "tables", "settings.csv"))
    for _, r in req.iterrows():
        rows.append({"setting": r["setting"], "group": r["group"], "arm": "requested", "share_meets": r["share_meets"],
                     "share_unresolved": r["share_unresolved"], "share_short": r["share_short"],
                     "share_not_measurable": r["share_not_measurable"]})
    for s in forced:
        x = pp.promise_one(s["name"], "heldout", {"Gd": s["Pp"] - s["Bp"], "Gv": s["Pv"] - s["Bv"]})
        rows.append({"setting": s["name"], "group": s["group"], "arm": "forced", "share_meets": x["share_meets"],
                     "share_unresolved": x["share_unresolved"], "share_short": x["share_short"],
                     "share_not_measurable": x["share_not_measurable"]})
    df = pd.DataFrame(rows)
    df["passes"] = df["share_meets"] >= 0.5
    return df


def by_group(df, bud):
    out = {}
    for g, x in df.groupby("group", sort=False):
        pos = x[x["risk_constant"] > 0]
        b = bud[bud["group"] == g]
        r = {"settings": int(len(x)), "positive_constant": int(len(pos)),
             "one_mean_beats": int((pos["one_mean"] < pos["risk_constant"]).sum()),
             "one_first_beats": int((pos["one_first"] < pos["risk_constant"]).sum()),
             "one_share_draws_beat": float(pos["one_share_draws_beat"].mean()) if len(pos) else None,
             "one_mean_share_median": float(pos["one_mean_share"].median()) if len(pos) else None,
             "one_over_expected_median": float(pd.to_numeric(x["one_mean_over_expected"]).median()),
             "budget_pass": int(b["passes"].sum()), "budget_settings": int(len(b)),
             "budget_meets_mean": float(b["share_meets"].mean()) if len(b) else None,
             "budget_unresolved_mean": float(b["share_unresolved"].mean()) if len(b) else None}
        for k in BLOCKS:
            r[f"pilot_{k}_beats"] = int((pos[f"pilot_{k}_mean"] < pos["risk_constant"]).sum())
            r[f"shrunk_{k}_beats"] = int((pos[f"shrunk_{k}_mean"] < pos["risk_constant"]).sum())
            r[f"shrunk_{k}_beats_pilot"] = int((pos[f"shrunk_{k}_mean"] < pos[f"pilot_{k}_mean"]).sum())
            r[f"unresolved_{k}_mean"] = float(x[f"unresolved_{k}"].mean())
        out[g] = r
    for lab, x in (("R_at_least_half", df[df["R"] >= 0.5]), ("R_below_half", df[df["R"] < 0.5])):
        pos = x[x["risk_constant"] > 0]
        out[lab] = {"settings": int(len(x)), "positive_constant": int(len(pos)),
                    "one_mean_beats": int((pos["one_mean"] < pos["risk_constant"]).sum()),
                    "one_first_beats": int((pos["one_first"] < pos["risk_constant"]).sum()),
                    "one_share_draws_beat": float(pos["one_share_draws_beat"].mean()) if len(pos) else None,
                    # the same counts by setting group
                    "by_group": {g: {"settings": int(len(y)), "positive_constant": int((y["risk_constant"] > 0).sum()),
                                     "one_mean_beats": int(((y["one_mean"] < y["risk_constant"])
                                                            & (y["risk_constant"] > 0)).sum()),
                                     "one_first_beats": int(((y["one_first"] < y["risk_constant"])
                                                             & (y["risk_constant"] > 0)).sum())}
                                 for g, y in x.groupby("group", sort=False)}}
    return out


def run(a):
    t0 = time.time()
    pp = load_script("run_pilot_checks")
    settings = requested_settings()
    forced, states = forced_settings()
    settings += forced
    print(f"{len(settings)} settings ({time.time() - t0:.0f} s)", flush=True)
    frozen = {s["name"]: rules(s["Bp"], s["Pp"]) for s in settings}
    rec = {"label": "pilots_v3", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp, centred",
           "note": "from pilot or discovery draws only; written before this run builds the "
                   "Dataset 2 validation draws, which earlier analyses had read", "rules": frozen}
    rec["models"] = sorted(frozen)
    rec["sha256"] = sha256_json(json.loads(json.dumps(jsonable(frozen))))
    rec = d2.write_frozen(jsonable(rec), PRED, overwrite=a.overwrite_frozen)
    print(f"rules frozen -> manifests/predictions_pilots_v3.json ({time.time() - t0:.0f} s)", flush=True)
    # validation draws, built only now
    loaded = {}
    for s in settings:
        if "inp" in s:
            inp = s["inp"]
            if id(inp) not in loaded:
                val = inp["sampler"]["validation_config_ids"]
                loaded[id(inp)] = (inp["load"](val), val)
            dv, val = loaded[id(inp)]
            its = inp["subsets"][s["name"]][1]
            cells = d2.cell_table(dv[dv["item_id"].isin(pd.Index(its))], inp["panel"], val, its)
            s["Bv"], s["Pv"] = d2.unit_arrays(cells, val, inp["panel"], "strict_correct")
    for st in states:
        arr = forced_arrays(st, list(st["sampler"]["validation_config_ids"]))
        for s in forced:
            if s["name"] in arr:
                s["Bv"], s["Pv"] = arr[s["name"]]
    rows = [score(s, rec["rules"][s["name"]]) for s in settings]
    df = pd.DataFrame(rows)
    mp = {r["setting"]: r for r in json.load(open(os.path.join(RES, "map_panels", "map_panels.json"),
                                                  encoding="utf-8"))["pilot_rows"]}
    for r in rows:
        ref = mp.get(r["setting"], {}).get("risk_pilot_1")
        if r["arm"] == "requested" and ref is not None and abs(ref - r["one_first"]) > 1e-6 * max(1, abs(r["risk_constant"])):
            raise SystemExit(f"{r['setting']}: the first-draw pilot risk differs from results/map_panels")
    print(f"scored ({time.time() - t0:.0f} s)", flush=True)
    bud = budget(forced, pp)
    print(f"budgeting test ({time.time() - t0:.0f} s)", flush=True)
    out = {"analysis": "pilots", "methods": "scripts/run_pilots_v3.py",
           "created": time.strftime("%Y-%m-%d %H:%M"), "frozen": {"path": "manifests/predictions_pilots_v3.json",
                                                                 "created": rec["created"], "sha256": rec["sha256"]},
           "by_group": by_group(df, bud), "settings": rows}
    out["macros"] = macros(out)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    write("pilots_v3", out, summary(out), {"settings.csv": df, "budget.csv": bud})
    print(f"done in {out['runtime_seconds']} s -> results/pilots_v3", flush=True)


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def macros(out):
    m = {}
    for g, r in out["by_group"].items():
        t = GROUPS[g][1] if g in GROUPS else {"R_at_least_half": "High", "R_below_half": "Low"}[g]
        m[f"pv{t}N"], m[f"pv{t}PosN"] = str(r["settings"]), str(r["positive_constant"])
        m[f"pv{t}OneMeanWins"], m[f"pv{t}OneFirstWins"] = str(r["one_mean_beats"]), str(r["one_first_beats"])
        m[f"pv{t}OneDrawWinShare"] = fmt(100 * r["one_share_draws_beat"], 0) if r["one_share_draws_beat"] is not None else "--"
        for gg, y in r.get("by_group", {}).items():
            gt = t + GROUPS[gg][1]
            m[f"pv{gt}N"], m[f"pv{gt}PosN"] = str(y["settings"]), str(y["positive_constant"])
            m[f"pv{gt}OneMeanWins"], m[f"pv{gt}OneFirstWins"] = str(y["one_mean_beats"]), str(y["one_first_beats"])
        if g in GROUPS:
            m[f"pv{t}OneMeanShare"] = fmt(100 * r["one_mean_share_median"], 0) if r["one_mean_share_median"] is not None else "--"
            m[f"pv{t}OneOverExpected"] = fmt(r["one_over_expected_median"], 2)
            m[f"pv{t}BudgetPass"], m[f"pv{t}BudgetN"] = str(r["budget_pass"]), str(r["budget_settings"])
            m[f"pv{t}BudgetMeets"] = fmt(100 * r["budget_meets_mean"], 0) if r["budget_meets_mean"] is not None else "--"
            m[f"pv{t}BudgetUnresolved"] = (fmt(100 * r["budget_unresolved_mean"], 0)
                                           if r["budget_unresolved_mean"] is not None else "--")
            for k, kt in ((2, "Two"), (4, "Four")):
                m[f"pv{t}Pilot{kt}Wins"], m[f"pv{t}Shrunk{kt}Wins"] = str(r[f"pilot_{k}_beats"]), str(r[f"shrunk_{k}_beats"])
                m[f"pv{t}Shrunk{kt}BeatsPilot"] = str(r[f"shrunk_{k}_beats_pilot"])
                m[f"pv{t}Unresolved{kt}"] = fmt(100 * r[f"unresolved_{k}_mean"], 0)
    return m


def summary(out):
    L = ["# Pilots, averaged over draws, shrunken, budgeted, and on the forced arm", "",
         "Every prediction was saved to "
         f"`{out['frozen']['path']}` ({out['frozen']['created']}) before this run built the Dataset 2 validation draws, "
         "which earlier analyses had read. Cross-product risks in pp², the mean over models (may be negative); a one-draw "
         "pilot's expected risk is W_gain (M − 1) / M, the constant's about V_delta (M − 1) / M.", "",
         "## One-draw pilots averaged over every pilot draw", "",
         "| Settings | N | Constant risk positive | Mean one-draw pilot beats constant | First draw only | "
         "Single draws beating constant | Median mean risk, % of constant | Median mean risk / (W (M − 1) / M) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g, r in out["by_group"].items():
        lab = GROUPS[g][0] if g in GROUPS else {"R_at_least_half": "all, R ≥ ½", "R_below_half": "all, R < ½"}[g]
        L.append(f"| {lab} | {r['settings']} | {r['positive_constant']} | {r['one_mean_beats']} | {r['one_first_beats']} | "
                 f"{fmt(100 * r['one_share_draws_beat'], 0) if r['one_share_draws_beat'] is not None else '--'}% | "
                 + (f"{fmt(100 * r['one_mean_share_median'], 0)}% | {fmt(r['one_over_expected_median'], 2)} |"
                    if g in GROUPS else "-- | -- |"))
    L += ["", "The two R classes by setting group: settings, settings with a positive constant risk, and among "
          "those the settings where the mean one-draw pilot (and the first draw alone) beats the constant.", "",
          "| R class | Settings | N | Constant risk positive | Mean one-draw pilot beats constant | First draw only |",
          "|---|---|---:|---:|---:|---:|"]
    for lab, cls in (("R_at_least_half", "R ≥ ½"), ("R_below_half", "R < ½")):
        for g, y in out["by_group"][lab]["by_group"].items():
            L.append(f"| {cls} | {GROUPS[g][0]} | {y['settings']} | {y['positive_constant']} | {y['one_mean_beats']} | "
                     f"{y['one_first_beats']} |")
    L += ["", "## K-draw pilots over every block, unresolved blocks counted as the constant", "",
          "| Settings | 2-draw beats constant | shrunk | unresolved blocks | 4-draw beats constant | shrunk | unresolved "
          "blocks | Shrunk beats unshrunk (2 / 4) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for g, r in out["by_group"].items():
        if g not in GROUPS:
            continue
        L.append(f"| {GROUPS[g][0]} | {r['pilot_2_beats']} | {r['shrunk_2_beats']} | {fmt(100 * r['unresolved_2_mean'], 0)}% | "
                 f"{r['pilot_4_beats']} | {r['shrunk_4_beats']} | {fmt(100 * r['unresolved_4_mean'], 0)}% | "
                 f"{r['shrunk_2_beats_pilot']} / {r['shrunk_4_beats_pilot']} |")
    L += ["", "Counts are of settings whose constant risk is positive, comparing the mean risk over blocks.", "",
          "## Four-draw budgeting test, unresolved pilots counted", "",
          "| Settings | N | Pass (≥ half of all pilots meet 80%) | Mean share meeting | Mean share unresolved |",
          "|---|---:|---:|---:|---:|"]
    for g, r in out["by_group"].items():
        if g not in GROUPS:
            continue
        L.append(f"| {GROUPS[g][0]} | {r['budget_settings']} | {r['budget_pass']} | "
                 f"{fmt(100 * r['budget_meets_mean'], 0)}% | {fmt(100 * r['budget_unresolved_mean'], 0)}% |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
