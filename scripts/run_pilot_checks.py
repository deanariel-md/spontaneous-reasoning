r"""Pilot checks: does a
four-draw pilot deliver the agreement it promises; shrunken pilots; and rescaled transfer charged for the target
draws it uses. Estimators: src/pilot_checks.py.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_pilot_checks.py 2>&1"

Settings: those of the reliability map with draws to measure on (scripts/run_reliability_map.py loaders): the DOVE
MMLU block and the 71 DOVE benchmarks over their enumerated banks, GSM8K-19, GSM8K-29, BIG-Bench Hard pooled and
its 25 tasks. The shrunken-pilot and charged-transfer parts use the pilot test of scripts/run_map_panels.py (same settings, pilot and validation draws) and
the cross-task cases of scripts/run_transfer_tasks.py (28 shared models, the 21 held-out BIG-Bench Hard families).

Order: the shrunken pilots and the charged transfer rules are fitted from pilot or discovery draws and
written to manifests/predictions_shrunken_pilots.json and manifests/predictions_transfer_charged.json before this
run builds the Dataset 2 validation draws. Earlier analyses had read those draws, so neither is a fresh test.

Outputs: results/pilot_promise/pilot_promise.json (prefix "pp") and results/shrunken_pilots/shrunken_pilots.json
(prefix "sp"), each with summary.md, summary_clean.md and tables/*.csv.
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
from src import dataset2 as d2, pilot_checks as pc, sampling, transfer_d2  # noqa: E402
from src.estimators import cross_risk  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

RES = os.path.join(ROOT, "results")
SEED_LABEL = "pilot-promise-20261103"
N_PILOTS, N_PAIRS, N_SPLITS, PILOT_DRAWS = 500, 4000, 2000, 4
PRED_SHRUNK = os.path.join(ROOT, "manifests", "predictions_shrunken_pilots.json")
PRED_CHARGED = os.path.join(ROOT, "manifests", "predictions_transfer_charged.json")
GROUP_PANEL = {"dove_mmlu": "DOVE MMLU block", "dove_tasks": "DOVE benchmarks", "gsm8k": "GSM8K (19)",
               "gsm8k30": "GSM8K (29)", "bbh": "BIG-Bench Hard (28), pooled", "bbh_tasks": "BIG-Bench Hard tasks"}
CLASSES = ("meets", "short", "unresolved", "not measurable")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{name}_for_pilot_checks", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def write(name, out, lines, tables):
    d = os.path.join(RES, name)
    os.makedirs(os.path.join(d, "tables"), exist_ok=True)
    for k in out.get("macros", {}):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=1)
    for fn, df in tables.items():
        df.to_csv(os.path.join(d, "tables", fn), index=False)
    txt = "\n".join(lines) + "\n"
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)


# ============================================================================ pilot promise


def promise_settings():
    """(setting, group, kind, arrays): kind "bank" with the enumerated bank's gains, or "heldout" with the discovery
    and validation gains."""
    import yaml
    rmap = load_script("run_reliability_map")
    cfg = yaml.safe_load(open(rmap.CONFIG, encoding="utf-8"))
    out = []
    s = rmap.dove_mmlu_inputs()
    out.append((s["setting"], "dove_mmlu", "bank", {"G": s["P"] - s["B"]}))
    tasks, _, _ = rmap.dove_task_inputs(cfg)
    for s in tasks:
        out.append((s["setting"], "dove_tasks", "bank", {"G": s["P"] - s["B"]}))
    gs = os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet")
    for inp in (rmap.gsm8k19_inputs(gs), rmap.gsm8k30_inputs(gs, None), rmap.bbh_inputs(None)):
        disc, val = inp["sampler"]["discovery_config_ids"], inp["sampler"]["validation_config_ids"]
        dd, dv = inp["load"](disc), inp["load"](val)
        for name, (_, its) in inp["subsets"].items():
            arr = {}
            for lab, df, units in (("Gd", dd, disc), ("Gv", dv, val)):
                cells = d2.cell_table(df[df["item_id"].isin(pd.Index(its))], inp["panel"], units, its)
                B, P = d2.unit_arrays(cells, units, inp["panel"], "strict_correct")
                arr[lab] = P - B
            out.append((name, inp["group_of"][name], "heldout", arr))
    return out


def promise_one(setting, kind, arr):
    rng = sampling.rng_for(SEED_LABEL, setting)
    if kind == "bank":
        G = arr["G"]
        idx = rng.integers(0, G.shape[0], size=(N_PILOTS, PILOT_DRAWS))
        pilots = [G[i] for i in idx]
    else:
        G = arr["Gd"]
        pilots = [G[np.sort(rng.permutation(G.shape[0])[:PILOT_DRAWS])] for _ in range(N_PILOTS)]
    R = np.array([pc.pilot_r(p) for p in pilots])
    ks = [pc.chosen_k(r) for r in R]
    achieved = {}
    for k in sorted({k for k in ks if k is not None}):
        krng = sampling.rng_for(SEED_LABEL, setting, "achieved", k)
        achieved[k] = (pc.achieved_enumerated(arr["G"], k, N_PAIRS, krng) if kind == "bank"
                       else pc.achieved_heldout(arr["Gv"], k, N_SPLITS, krng))
    cls = [pc.classify(k, None if k is None else achieved[k]) for k in ks]
    n = len(cls)
    resolved = [c for c in cls if c != "unresolved"]
    kk = [k for k in ks if k is not None]
    return {"setting": setting, "kind": kind, "M": int(G.shape[1]), "n_pilots": n,
            **{f"share_{c.replace(' ', '_')}": cls.count(c) / n for c in CLASSES},
            "share_meets_of_resolved": (cls.count("meets") / len(resolved)) if resolved else None,
            "R_pilot_median": float(np.nanmedian(R)) if np.isfinite(R).any() else None,
            "R_pilot_negative_or_zero": int(np.sum(R <= 0)), "R_pilot_undefined": int(np.sum(~np.isfinite(R))),
            "K_median": float(np.median(kk)) if kk else None, "K_max": int(max(kk)) if kk else None,
            "achieved_at_median_K": (achieved.get(int(np.median(kk))) if kk and float(np.median(kk)).is_integer()
                                     else None),
            "achieved_by_K": {str(k): v for k, v in achieved.items()}}


def promise(rows_raw):
    rows = []
    t0 = time.time()
    for setting, group, kind, arr in rows_raw:
        r = promise_one(setting, kind, arr)
        r["group"], r["panel"] = group, GROUP_PANEL[group]
        rows.append(r)
    print(f"  pilot promise: {len(rows)} settings ({time.time() - t0:.0f} s)")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "achieved_by_K"} for r in rows])
    by = []
    for panel, g in df.groupby("panel", sort=False):
        x = {"panel": panel, "settings": int(len(g))}
        for c in CLASSES:
            x[f"mean_share_{c.replace(' ', '_')}"] = float(g[f"share_{c.replace(' ', '_')}"].mean())
        s = pd.to_numeric(g["share_meets_of_resolved"], errors="coerce")
        x["settings_half_resolved_meet"] = int((s >= 0.5).sum())
        x["settings_with_resolved"] = int(s.notna().sum())
        by.append(x)
    allx = {"panel": "all", "settings": int(len(df))}
    for c in CLASSES:
        allx[f"mean_share_{c.replace(' ', '_')}"] = float(df[f"share_{c.replace(' ', '_')}"].mean())
    s = pd.to_numeric(df["share_meets_of_resolved"], errors="coerce")
    allx["settings_half_resolved_meet"] = int((s >= 0.5).sum())
    allx["settings_with_resolved"] = int(s.notna().sum())
    by.append(allx)
    return rows, by, df


def macros_pp(by):
    m = {}
    tag = {"DOVE MMLU block": "DoveBlock", "DOVE benchmarks": "Dove", "GSM8K (19)": "Gsm", "GSM8K (29)": "GsmThirty",
           "BIG-Bench Hard (28), pooled": "Bbh", "BIG-Bench Hard tasks": "BbhTasks", "all": "All"}
    for x in by:
        t = tag[x["panel"]]
        m[f"pp{t}Settings"] = str(x["settings"])
        m[f"pp{t}Meets"] = fmt(100 * x["mean_share_meets"], 0)
        m[f"pp{t}Short"] = fmt(100 * x["mean_share_short"], 0)
        m[f"pp{t}Unresolved"] = fmt(100 * x["mean_share_unresolved"], 0)
        m[f"pp{t}NotMeasurable"] = fmt(100 * x["mean_share_not_measurable"], 0)
        m[f"pp{t}HalfMeet"] = str(x["settings_half_resolved_meet"])
        m[f"pp{t}WithResolved"] = str(x["settings_with_resolved"])
    m["ppPilots"] = str(N_PILOTS)
    m["ppDraws"] = str(PILOT_DRAWS)
    m["ppTarget"] = fmt(100 * pc.TARGET, 0)
    return m


def summary_pp(out):
    L = ["# Does a four-draw pilot deliver the agreement it promises?", "",
         "Per setting, 500 pilots of 4 draws. Each pilot's R ("
         "reliability of its benefits) picks K for 80% benefit agreement through the Gaussian reference and "
         "Spearman–Brown; R ≤ 0 or undefined leaves the pilot unresolved. The agreement achieved with K fresh draws "
         "is measured on DOVE over the enumerated bank (two independent K-draw evaluations, 4,000 Monte Carlo pairs) "
         "and elsewhere between two disjoint groups of K validation draws (2,000 random splits; not measurable when "
         "2K > 24). A pilot meets the target when the achieved agreement is at least 0.80.", "",
         "| Panel | Settings | Meets | Falls short | Unresolved | Not measurable | Settings where ≥ half of resolved pilots meet |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for x in out["by_panel"]:
        L.append(f"| {x['panel']} | {x['settings']} | {fmt(100 * x['mean_share_meets'], 0)}% | "
                 f"{fmt(100 * x['mean_share_short'], 0)}% | {fmt(100 * x['mean_share_unresolved'], 0)}% | "
                 f"{fmt(100 * x['mean_share_not_measurable'], 0)}% | {x['settings_half_resolved_meet']} of "
                 f"{x['settings_with_resolved']} |")
    L += ["", "Shares are the mean over the panel's settings of each setting's share of pilots.", "",
          "## Settings outside DOVE", "",
          "| Setting | M | Median pilot R | Median K | Achieved at the median K | Meets | Short | Unresolved | Not measurable |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out["settings"]:
        if r["kind"] != "heldout":
            continue
        L.append(f"| {r['setting']} | {r['M']} | {fmt(r['R_pilot_median'])} | {fmt(r['K_median'], 0)} | "
                 f"{fmt(r['achieved_at_median_K'], 3)} | {fmt(100 * r['share_meets'], 0)}% | "
                 f"{fmt(100 * r['share_short'], 0)}% | {fmt(100 * r['share_unresolved'], 0)}% | "
                 f"{fmt(100 * r['share_not_measurable'], 0)}% |")
    return L


# ============================================================================ shrunken pilots


def shrunk_rules(Bp, Pp):
    G = Pp - Bp
    c = lambda x: x - x.mean()  # noqa: E731
    out = {"constant": np.zeros(G.shape[1])}
    for k in (2, 4):
        if G.shape[0] < k:
            continue
        f = pc.shrink_factor(G[:k])
        out[f"pilot_{k}"] = c(G[:k].mean(0))
        out[f"shrunk_{k}"] = None if f is None else f * c(G[:k].mean(0))
        out[f"factor_{k}"] = f
    return out


def shrunken(overwrite):
    mp = load_script("run_map_panels")
    raw = mp.settings_with_validation()
    settings = []
    for item in raw:
        if item[0] == "d2":
            _, inp, rows = item
            for name, group, Bd, Pd, _, _ in rows:
                settings.append({"name": name, "group": group, "Bp": Bd, "Pp": Pd, "inp": inp})
        else:
            name, group, Bp, Pp, Bv, Pv = item
            settings.append({"name": name, "group": group, "Bp": Bp, "Pp": Pp, "Bv": Bv, "Pv": Pv})
    frozen = {}
    for s in settings:
        s["rules"] = shrunk_rules(s["Bp"], s["Pp"])
        frozen[s["name"]] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s["rules"].items()}
    rec = {"label": "shrunken_pilots", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp, centred",
           "note": "from pilot or discovery draws only; written before this run builds "
                   "the Dataset 2 validation draws, which earlier analyses had read", "rules": frozen}
    rec["models"] = sorted(frozen)
    rec["sha256"] = sha256_json(json.loads(json.dumps(jsonable(frozen))))
    rec = d2.write_frozen(jsonable(rec), PRED_SHRUNK, overwrite=overwrite)
    return settings, rec


def shrunken_score(settings, rec):
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
    mprows = {r["setting"]: r for r in json.load(open(os.path.join(RES, "map_panels", "map_panels.json"),
                                                      encoding="utf-8"))["pilot_rows"]}
    rows = []
    for s in settings:
        Z = s["Pv"] - s["Bv"]
        Z = Z - Z.mean(1, keepdims=True)
        rk = {k: (None if v is None else cross_risk(Z, np.asarray(v, float))) for k, v in rec["rules"][s["name"]].items()
              if not k.startswith("factor")}
        r0 = rk["constant"]
        m = mprows[s["name"]]
        for k in (2, 4):
            if m.get(f"risk_pilot_{k}") is not None and abs(m[f"risk_pilot_{k}"] - rk[f"pilot_{k}"]) > 1e-6 * max(1, abs(r0)):
                raise SystemExit(f"{s['name']}: pilot_{k} risk differs from results/map_panels")
        row = {"setting": s["name"], "group": s["group"], "R_from_pilot_draws": m["R_from_pilot_draws"],
               "risk_constant": r0}
        for k in (2, 4):
            row[f"factor_{k}"] = rec["rules"][s["name"]].get(f"factor_{k}")
            for lab in ("pilot", "shrunk"):
                v = rk.get(f"{lab}_{k}")
                row[f"risk_{lab}_{k}"] = v
                row[f"share_{lab}_{k}"] = None if v is None or r0 is None or r0 <= 0 else v / r0
        rows.append(row)
    return rows, {"path": rel(PRED_SHRUNK), "created": rec["created"], "sha256": rec["sha256"]}


def shrunk_summary(rows):
    df = pd.DataFrame(rows)
    df = df[df["risk_constant"] > 0]
    out = {"settings_positive_constant": int(len(df))}
    for split, g in (("all", df), ("R_at_least_half", df[df["R_from_pilot_draws"] >= 0.5]),
                     ("R_below_half", df[df["R_from_pilot_draws"] < 0.5]),
                     ("dove", df[df["group"].isin(["dove_mmlu", "dove_tasks"])])):
        x = {"settings": int(len(g))}
        for k in (2, 4):
            sp, ss = pd.to_numeric(g[f"share_pilot_{k}"]), pd.to_numeric(g[f"share_shrunk_{k}"])
            x[f"pilot_{k}_beats_constant"] = int((sp < 1).sum())
            x[f"shrunk_{k}_beats_constant"] = int((ss < 1).sum())
            x[f"shrunk_{k}_beats_pilot"] = int((g[f"risk_shrunk_{k}"] < g[f"risk_pilot_{k}"]).sum())
            x[f"pilot_{k}_median_share"] = float(sp.median()) if len(sp) else None
            x[f"shrunk_{k}_median_share"] = float(ss.median()) if len(ss) else None
        out[split] = x
    return out


# ============================================================================ charged transfer


def charged(overwrite):
    tt = load_script("run_transfer_tasks")
    from src import d2_io, d2_panels, tasks_d2
    s_g = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8"))
    s_b = json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
    items_g, items_info = d2_io.load_items(split="test")
    gsm = d2_panels.inputs(None, None, s_g, items_info, os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    gsm["items"] = items_g["item_id"].tolist()
    bbh = d2_panels.bbh_inputs(None, s_b)
    panel = [k for k in gsm["panel"] if k in bbh["panel"]]
    disc_g, val_g = s_g["discovery_config_ids"], s_g["validation_config_ids"]
    disc_b, val_b = s_b["discovery_config_ids"], s_b["validation_config_ids"]
    tasks = list(bbh["items_by_task"])
    fams = {t: tasks_d2.bbh_family(t) for t in tasks}
    Bg, Pg = tt.arms(gsm["load"](disc_g), panel, disc_g, gsm["items"])
    dfb = bbh["load"](disc_b)
    bt = {t: tt.arms(dfb, panel, disc_b, it) for t, it in bbh["items_by_task"].items()}
    Gb = np.mean([bt[t][1] - bt[t][0] for t in tasks], axis=0)
    src_tgt = {"gsm8k_to_bbh": (Pg - Bg, Gb), "bbh_to_gsm8k": (Gb, Pg - Bg)}
    held = {}
    for fam in sorted(set(fams.values())):
        train = [t for t in tasks if fams[t] != fam]
        test = [t for t in tasks if fams[t] == fam]
        w = np.asarray(tasks_d2.family_weights([fams[t] for t in train]), float)
        w = w / w.sum()
        sg = np.einsum("t,tkm->km", w, np.stack([bt[t][1] - bt[t][0] for t in train]))
        src_tgt[f"bbh_family:{fam}"] = (sg, np.mean([bt[t][1] - bt[t][0] for t in test], axis=0))
        held[fam] = test
    c = lambda x: x - x.mean()  # noqa: E731
    preds = {}
    for case, (Gs, Gt) in src_tgt.items():
        g = c(Gs.mean(0))
        sd_s = float(np.std(g, ddof=1))
        p = {}
        for k in (1, 2, 4):
            sd_t = pc.charged_sd(Gt, k)
            p[f"rescaled_{k}"] = None if sd_t is None or sd_s <= 0 else (g / sd_s * sd_t).tolist()
            p[f"pilot_{k}"] = c(Gt[:k].mean(0)).tolist()
            p[f"sd_target_{k}"] = sd_t
        p["sd_source"] = sd_s
        preds[case] = p
    rec = {"label": "transfer_charged", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": panel, "units": "pp",
           "samplers": {"gsm8k": s_g["sha256"], "bbh": s_b["sha256"]},
           "note": "rescaled transfer with the target SD from its first k discovery draws; "
                   "written before this run builds the validation draws, which earlier analyses had read",
           "predictions": preds}
    rec["sha256"] = sha256_json(json.loads(json.dumps(jsonable(preds))))
    rec = d2.write_frozen(jsonable(rec), PRED_CHARGED, overwrite=overwrite)
    return {"tt": tt, "gsm": gsm, "bbh": bbh, "panel": panel, "val_g": val_g, "val_b": val_b, "tasks": tasks,
            "held": held, "rec": rec}


def charged_score(st):
    tt, gsm, bbh, panel, val_g, val_b, tasks, held, rec = (st[k] for k in ("tt", "gsm", "bbh", "panel", "val_g",
                                                                           "val_b", "tasks", "held", "rec"))
    Vg = tt.arms(gsm["load"](val_g), panel, val_g, gsm["items"])
    dfv = bbh["load"](val_b)
    vt = {t: tt.arms(dfv, panel, val_b, it) for t, it in bbh["items_by_task"].items()}
    target = {"gsm8k_to_bbh": np.mean([vt[t][1] - vt[t][0] for t in tasks], axis=0), "bbh_to_gsm8k": Vg[1] - Vg[0]}
    for fam, test in held.items():
        target[f"bbh_family:{fam}"] = np.mean([vt[t][1] - vt[t][0] for t in test], axis=0)
    res = {}
    comps = [(f"pilot_{k}", f"rescaled_{k}") for k in (1, 2, 4)] + [("constant", f"rescaled_{k}") for k in (1, 2, 4)]
    for case, Z in target.items():
        p = rec["predictions"][case]
        rules = {"constant": [0.0] * len(panel), **{k: v for k, v in p.items() if k.startswith(("pilot", "rescaled"))}}
        sc = transfer_d2.score(rules, Z, comps)
        sc["sd"] = {k: v for k, v in p.items() if k.startswith("sd_")}
        res[case] = sc
    fam = [c_ for c_ in res if c_.startswith("bbh_family:")]
    summ = {}
    for f, g in comps:
        v = []
        for c_ in fam:
            d = next(x for x in res[c_]["differences"] if x["f"] == f and x["g"] == g)
            ci = d.get("ci95")
            v.append(0 if not ci else (1 if ci[0] > 0 else -1 if ci[1] < 0 else 0))
        summ[f"{f}_vs_{g}"] = {"families": len(fam), "g_better": v.count(1), "f_better": v.count(-1),
                               "undecided": v.count(0)}
    return {"models": panel, "M": len(panel), "cases": res, "families_summary": summ,
            "frozen": {"path": rel(PRED_CHARGED), "created": rec["created"], "sha256": rec["sha256"]}}


def macros_sp(out):
    m = {}
    s = out["shrunken"]["summary"]
    for split, t in (("all", "All"), ("R_at_least_half", "High"), ("R_below_half", "Low"), ("dove", "Dove")):
        x = s[split]
        m[f"sp{t}N"] = str(x["settings"])
        for k, kt in ((2, "Two"), (4, "Four")):
            m[f"sp{t}Pilot{kt}Wins"] = str(x[f"pilot_{k}_beats_constant"])
            m[f"sp{t}Shrunk{kt}Wins"] = str(x[f"shrunk_{k}_beats_constant"])
            m[f"sp{t}Shrunk{kt}BeatsPilot"] = str(x[f"shrunk_{k}_beats_pilot"])
            m[f"sp{t}Pilot{kt}Share"] = fmt(100 * x[f"pilot_{k}_median_share"], 0)
            m[f"sp{t}Shrunk{kt}Share"] = fmt(100 * x[f"shrunk_{k}_median_share"], 0)
    ch = out["charged"]
    for case, t in (("gsm8k_to_bbh", "GsmToBbh"), ("bbh_to_gsm8k", "BbhToGsm")):
        r = ch["cases"][case]
        for k, kt in ((1, "One"), (2, "Two"), (4, "Four")):
            m[f"spCh{t}RiskRescaled{kt}"] = fmt(r["risk_pp2"][f"rescaled_{k}"], 1)
            m[f"spCh{t}RiskPilot{kt}"] = fmt(r["risk_pp2"][f"pilot_{k}"], 1)
            d = next(x for x in r["differences"] if x["f"] == f"pilot_{k}" and x["g"] == f"rescaled_{k}")
            m[f"spCh{t}PilotVsRescaled{kt}"] = fmt(d["estimate"], 1)
            if d.get("ci95"):
                m[f"spCh{t}PilotVsRescaled{kt}Lo"], m[f"spCh{t}PilotVsRescaled{kt}Hi"] = (fmt(x, 1) for x in d["ci95"])
        m[f"spCh{t}RiskConstant"] = fmt(r["risk_pp2"]["constant"], 1)
    fs = ch["families_summary"]
    m["spChFamilies"] = str(fs["pilot_1_vs_rescaled_1"]["families"])
    for k, kt in ((1, "One"), (2, "Two"), (4, "Four")):
        x = fs[f"pilot_{k}_vs_rescaled_{k}"]
        m[f"spChFam{kt}RescaledBetter"], m[f"spChFam{kt}PilotBetter"] = str(x["g_better"]), str(x["f_better"])
    return m


def summary_sp(out):
    s = out["shrunken"]["summary"]
    L = ["# Shrunken pilots and rescaled transfer charged for its target draws", "",
         "Rules saved to "
         f"`{out['shrunken']['frozen']['path']}` ({out['shrunken']['frozen']['created']}) and "
         f"`{out['charged']['frozen']['path']}` ({out['charged']['frozen']['created']}) before this run built the "
         "Dataset 2 validation draws, which earlier analyses had read. Cross-product risks in pp² (may be negative).", "",
         "## Shrunken pilots", "",
         "The K-draw pilot's centred gains times R_K = K R / (1 + (K − 1) R), with R from the same K draws and the "
         "factor held to [0, 1]; K = 2 and 4 (one draw cannot estimate R). Settings of the map's pilot test whose "
         "constant-rule risk is positive; the R split uses the pilot test's R, as `\\mpPilot...`.", "",
         "| Settings | N | 2-draw pilot beats constant | shrunk | 4-draw pilot | shrunk | Shrunk beats unshrunk (2 / 4) | Median share of constant, 2: pilot / shrunk | 4: pilot / shrunk |",
         "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for split, lab in (("all", "all"), ("R_at_least_half", "R ≥ ½"), ("R_below_half", "R < ½"), ("dove", "DOVE")):
        x = s[split]
        L.append(f"| {lab} | {x['settings']} | {x['pilot_2_beats_constant']} | {x['shrunk_2_beats_constant']} | "
                 f"{x['pilot_4_beats_constant']} | {x['shrunk_4_beats_constant']} | {x['shrunk_2_beats_pilot']} / "
                 f"{x['shrunk_4_beats_pilot']} | {fmt(100 * x['pilot_2_median_share'], 0)}% / "
                 f"{fmt(100 * x['shrunk_2_median_share'], 0)}% | {fmt(100 * x['pilot_4_median_share'], 0)}% / "
                 f"{fmt(100 * x['shrunk_4_median_share'], 0)}% |")
    ch = out["charged"]
    L += ["", "## Rescaled transfer charged for its target draws", "",
          f"{ch['M']} shared models. The rescaled rule takes the target's SD across models from its first k discovery "
          "draws only (k = 1: the one draw's SD; k = 2, 4: the noise-corrected √V_delta), and is compared with a pilot "
          "of the same k draws. A positive difference favours rescaled transfer.", ""]
    for case, title in (("gsm8k_to_bbh", "GSM8K to BIG-Bench Hard"), ("bbh_to_gsm8k", "BIG-Bench Hard to GSM8K")):
        r = ch["cases"][case]
        L += [f"### {title}", "", f"Constant rule: {fmt(r['risk_pp2']['constant'], 1)} pp².", "",
              "| k | Target SD (pp) | Rescaled risk | Pilot risk | Rescaled better by | 95% interval |", "|---:|---:|---:|---:|---:|---|"]
        for k in (1, 2, 4):
            d = next(x for x in r["differences"] if x["f"] == f"pilot_{k}" and x["g"] == f"rescaled_{k}")
            ci = d.get("ci95")
            L.append(f"| {k} | {fmt(r['sd'][f'sd_target_{k}'])} | {fmt(r['risk_pp2'][f'rescaled_{k}'], 1)} | "
                     f"{fmt(r['risk_pp2'][f'pilot_{k}'], 1)} | {fmt(d['estimate'], 1)} | "
                     f"{'--' if not ci else f'[{fmt(ci[0], 1)}, {fmt(ci[1], 1)}]'} |")
        L.append("")
    fs = ch["families_summary"]
    L += ["### Held-out BIG-Bench Hard families", "", "Counts read each paired difference's interval against zero.", "",
          "| Comparison | Families | Rescaled better | Other better | Undecided |", "|---|---:|---:|---:|---:|"]
    for key, x in fs.items():
        L.append(f"| {key.replace('_vs_', ' against ').replace('_', ' ')} | {x['families']} | {x['g_better']} | "
                 f"{x['f_better']} | {x['undecided']} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    # Both rule files are written before any Dataset 2 validation draw is built.
    sh_settings, sh_rec = shrunken(a.overwrite_frozen)
    ch_state = charged(a.overwrite_frozen)
    print(f"shrunken and charged rules frozen ({time.time() - t0:.0f} s)")
    sh_rows, sh_frozen = shrunken_score(sh_settings, sh_rec)
    ch = charged_score(ch_state)
    print(f"shrunken and charged scored ({time.time() - t0:.0f} s)")
    raw = promise_settings()
    print(f"pilot promise inputs loaded ({time.time() - t0:.0f} s)")
    rows, by, df = promise(raw)
    out_pp = {"analysis": "does a four-draw pilot deliver its promised agreement",
              "methods": "scripts/run_pilot_checks.py",
              "seed_label": SEED_LABEL, "n_pilots": N_PILOTS,
              "pilot_draws": PILOT_DRAWS, "n_pairs": N_PAIRS, "n_splits": N_SPLITS, "target": pc.TARGET,
              "settings": rows, "by_panel": by}
    out_pp["macros"] = macros_pp(by)
    write("pilot_promise", out_pp, summary_pp(out_pp), {"settings.csv": df})
    out_sp = {"analysis": "shrunken pilots; rescaled transfer charged for its target draws",
              "methods": "scripts/run_pilot_checks.py",
              "shrunken": {"frozen": sh_frozen, "rows": sh_rows, "summary": shrunk_summary(sh_rows)}, "charged": ch}
    out_sp["macros"] = macros_sp(out_sp)
    diffs = [{"case": c_, "f": d["f"], "g": d["g"], "difference_pp2": d["estimate"],
              "ci95_lo": (d.get("ci95") or [None, None])[0], "ci95_hi": (d.get("ci95") or [None, None])[1]}
             for c_, sc in ch["cases"].items() for d in sc["differences"]]
    write("shrunken_pilots", out_sp, summary_sp(out_sp),
          {"shrunken.csv": pd.DataFrame(sh_rows), "charged_differences.csv": pd.DataFrame(diffs)})
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
