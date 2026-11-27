r"""Analysis T across tasks: on a task, which rule predicts each model's
benefit best? Transfer from another task, the capability rules (ECI; baseline accuracy), a short pilot of K = 1, 2 or
4 prompt draws on the task itself, and the constant. The rules are in src/transfer_d2.py.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_transfer_tasks.py 2>&1"

Cases: GSM8K to BIG-Bench Hard and back (the 25 tasks pooled with equal weights), and within BIG-Bench Hard each
held-out family predicted from the other families (task weights macro-averaged within family, as in
src/tasks_d2.within_bbh_transfer_predictions). The panel is the models both tasks retained (28), as in
scripts/run_dataset2_tasks.py, with the same inputs and samplers.

Order: discovery draws of both tasks; every rule fitted and written to
manifests/predictions_transfer_tasks.json; only then the validation draws. The transfer rule equals the one already
scored in results/dataset2/tasks.json, whose validation draws were read by earlier analyses, so this analysis adds rules
beside it; it is not a fresh test of transfer. The rescaled transfer rule goes to its own file,
manifests/predictions_transfer_rescaled.json, so the first file stays as it was, and it too is written before this
run builds the validation draws.

Also (A2): the correlation of model gains between GSM8K and BIG-Bench Hard on the 24 discovery bundles (bundle r
pairs the r-th discovery draw of each task), with the 48-draw version as a sensitivity, and the two-task profile
share of results/dataset2/tasks.json read against that correlation (src/transfer_d2.profile_share).

Outputs: results/transfer_tasks/transfer_tasks.json (with a "macros" object, prefix "tr", for paper/make_numbers.py),
summary.md, summary_clean.md and tables/*.csv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import dataset2 as d2, tasks_d2, transfer_d2  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LABEL = "transfer_tasks"
OUT = os.path.join(ROOT, "results", LABEL)
PRED = os.path.join(ROOT, "manifests", f"predictions_{LABEL}.json")
PRED_RESCALED = os.path.join(ROOT, "manifests", "predictions_transfer_rescaled.json")
ADDED_RULES = ("transfer_rescaled",)
RULE_NAMES = {"constant": "constant (zero centred benefit)", "transfer": "transfer from the other task",
              "transfer_rescaled": "rescaled transfer", "capability_eci": "capability slope on ECI", "capability_baseline": "capability slope on baseline",
              "pilot_1": "pilot, 1 draw", "pilot_2": "pilot, 2 draws", "pilot_4": "pilot, 4 draws",
              "own_24": "own gain, 24 draws"}


def arms(df, panel, units, items, value="strict_correct"):
    """(B, P), each (K, M) in pp, of the listed units over a fixed item set."""
    cells = d2.cell_table(df[df["item_id"].isin(items)], panel, units, items)
    return d2.unit_arrays(cells, units, panel, value)


def eci_for(panel):
    from src import bbh_d2
    reg = bbh_d2.load_registries()
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    by_repo = {e["repo"]: e for e in man["panel"]}
    out = []
    for k in panel:
        e = by_repo.get(reg[k]["hf_id"])
        if e is None or e["revision"] != reg[k]["revision"]:
            raise SystemExit(f"{k}: no matching entry in manifests/d2_panel_extension.json")
        out.append(float(e["eci"]))
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def run(gsm, bbh, s_g, s_b, overwrite_frozen=False):
    t0 = time.time()
    panel = [k for k in gsm["panel"] if k in bbh["panel"]]
    eci = eci_for(panel)
    disc_g, val_g = s_g["discovery_config_ids"], s_g["validation_config_ids"]
    disc_b, val_b = s_b["discovery_config_ids"], s_b["validation_config_ids"]
    tasks = list(bbh["items_by_task"])
    fams = {t: tasks_d2.bbh_family(t) for t in tasks}

    # ------------------------------------------------------------ discovery draws only
    Bg, Pg = arms(gsm["load"](disc_g), panel, disc_g, gsm["items"])
    dfb = bbh["load"](disc_b)
    bt = {t: arms(dfb, panel, disc_b, it) for t, it in bbh["items_by_task"].items()}
    Bb = np.mean([bt[t][0] for t in tasks], axis=0)
    Pb = np.mean([bt[t][1] for t in tasks], axis=0)
    cases = {"gsm8k_to_bbh": transfer_d2.rules(Pg - Bg, Bg, Pb - Bb, eci),
             "bbh_to_gsm8k": transfer_d2.rules(Pb - Bb, Bb, Pg - Bg, eci)}
    held = {}
    for fam in sorted(set(fams.values())):
        train = [t for t in tasks if fams[t] != fam]
        test = [t for t in tasks if fams[t] == fam]
        w = np.asarray(tasks_d2.family_weights([fams[t] for t in train]), float)
        w = w / w.sum()
        sg = np.einsum("t,tkm->km", w, np.stack([bt[t][1] - bt[t][0] for t in train]))
        sb = np.einsum("t,tkm->km", w, np.stack([bt[t][0] for t in train]))
        tg = np.mean([bt[t][1] - bt[t][0] for t in test], axis=0)
        cases[f"bbh_family:{fam}"] = transfer_d2.rules(sg, sb, tg, eci)
        held[fam] = test
    # the transfer rule must equal the one scripts/run_dataset2_tasks.py froze
    prev = json.load(open(os.path.join(ROOT, "manifests", "predictions_tasks.json"), encoding="utf-8"))
    if prev["models"] == panel:
        for c in ("gsm8k_to_bbh", "bbh_to_gsm8k"):
            dev = np.max(np.abs(np.asarray(prev["predictions"][c]) - cases[c]["transfer"]["prediction"]))
            if dev > 1e-9:
                raise SystemExit(f"{c}: transfer differs from manifests/predictions_tasks.json by {dev}")
    def freeze(path, keep, note):
        preds = jsonable({c: {r: v for r, v in rs.items() if keep(r)} for c, rs in cases.items()})
        fr = {"label": LABEL, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": panel, "units": "pp",
              "eci": eci, "samplers": {"gsm8k": s_g["sha256"], "bbh": s_b["sha256"]}, "note": note,
              "predictions": preds}
        fr["sha256"] = sha256_json(json.loads(json.dumps(preds)))
        return d2.write_frozen(fr, path, overwrite=overwrite_frozen)
    frozen = freeze(PRED, lambda r: r not in ADDED_RULES,
                    "model-centred predictions from discovery draws only; written before the validation draws are "
                    "built")
    added = freeze(PRED_RESCALED, lambda r: r in ADDED_RULES,
                   "rescaled transfer, from discovery draws only; written before this run "
                   "builds the validation draws, which earlier analyses had read")
    for c in frozen["predictions"]:
        frozen["predictions"][c].update(added["predictions"][c])
    print(f"predictions frozen: {len(cases)} cases x {len(next(iter(cases.values())))} rules")
    # cross-task correlation on the discovery bundles (pooled BIG-Bench Hard, equal task weights)
    corr = {"discovery_24": transfer_d2.cross_task_correlation(Pg - Bg, Pb - Bb)}

    # ------------------------------------------------------------ validation draws (built only now)
    Vg = arms(gsm["load"](val_g), panel, val_g, gsm["items"])
    dfv = bbh["load"](val_b)
    vt = {t: arms(dfv, panel, val_b, it) for t, it in bbh["items_by_task"].items()}
    Zg = Vg[1] - Vg[0]
    Zb = np.mean([vt[t][1] - vt[t][0] for t in tasks], axis=0)
    target = {"gsm8k_to_bbh": Zb, "bbh_to_gsm8k": Zg}
    for fam, test in held.items():
        target[f"bbh_family:{fam}"] = np.mean([vt[t][1] - vt[t][0] for t in test], axis=0)
    corr["all_48"] = transfer_d2.cross_task_correlation(np.vstack([Pg - Bg, Zg]), np.vstack([Pb - Bb, Zb]))
    corr["share"] = share_check(corr["discovery_24"])
    p = frozen["predictions"]
    res = {}
    for c, Z in target.items():
        sc = transfer_d2.score({r: v["prediction"] for r, v in p[c].items()}, Z, transfer_d2.COMPARISONS)
        sc["slopes"] = {"capability_eci_pp_per_eci": p[c]["capability_eci"]["slope_pp_per_eci"],
                        "capability_baseline_pp_per_pp": p[c]["capability_baseline"]["slope_pp_per_pp"]}
        sc["rescale"] = {k: p[c]["transfer_rescaled"][k] for k in ("sd_source_pp", "sd_target_discovery_pp")}
        res[c] = sc
    # the expectation from the discovery moments
    d24 = corr["discovery_24"]
    vg, vb, cgb = (d24[k]["estimate"] for k in ("V_delta_a", "V_delta_b", "cross_moment"))
    rho = d24["latent"]["estimate"]
    for c, vs, vt in (("gsm8k_to_bbh", vg, vb), ("bbh_to_gsm8k", vb, vg)):
        res[c]["expected_risk_pp2"] = {"constant": vt, "transfer": vs + vt - 2 * cgb,
                                       "transfer_rescaled": None if rho is None else 2 * (1 - rho) * vt}
    out = {"label": LABEL, "units": "percentage points; risks in pp^2", "models": panel, "M": len(panel),
           "K_discovery": len(disc_g), "K_validation": len(val_g), "held_out_families": held,
           "bbh_items_per_task": {t: len(v) for t, v in bbh["items_by_task"].items()},
           "frozen_predictions": {"path": os.path.relpath(PRED, ROOT).replace("\\", "/"), "sha256": frozen["sha256"],
                                  "created": frozen["created"]},
           "frozen_rescaled": {"path": os.path.relpath(PRED_RESCALED, ROOT).replace("\\", "/"),
                               "sha256": added["sha256"], "created": added["created"]},
           "inputs": {"gsm8k": gsm["provenance"], "bbh": bbh["provenance"]},
           "correlation": corr, "cases": res, "summary": summarise(res)}
    out["macros"] = macros(out)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{LABEL}.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=1)
    write_tables(out)
    write_summary(out)
    print(f"done in {out['runtime_seconds']} s -> {os.path.relpath(OUT, ROOT)}")
    return out


def share_check(c):
    """The two-task profile share of results/dataset2/tasks.json against the correlation."""
    dec = json.load(open(os.path.join(ROOT, "results", "dataset2", "tasks.json"), encoding="utf-8"))
    dec = dec["two_tasks"]["decomposition"]
    vg, vb, rho = c["V_delta_a"]["estimate"], c["V_delta_b"]["estimate"], c["latent"]["estimate"]
    per = {r["task"]: r["V_delta"] for r in dec["per_task"]}
    for t, v in (("gsm8k", vg), ("bbh", vb)):
        if abs(per[t] - v) > 1e-6 * max(1.0, abs(v)):
            raise SystemExit(f"{t}: V_delta {v} differs from results/dataset2/tasks.json ({per[t]})")
    observed = dec["share_profile_of_true_within"]
    at = transfer_d2.profile_share(vg, vb, rho) if rho is not None else None
    if at is not None and observed is not None and abs(at - observed) > 1e-6:
        raise SystemExit(f"profile share {observed} differs from the formula at the latent correlation ({at})")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {"observed": observed, "at_latent": at, "rho_latent": rho,
            "by_rho": {f"{r:.2f}": transfer_d2.profile_share(vg, vb, r) for r in grid},
            "minimum_for_nonnegative_rho": 0.5, "maximum_at_rho_one": transfer_d2.profile_share(vg, vb, 1.0),
            "gsm8k_part_of_within": vg / (vg + vb),
            "note": "share = (V_G + V_B + 2 rho sqrt(V_G V_B)) / (2 (V_G + V_B)), equal task weights; it is at least "
                    "1/2 for any rho >= 0, and its maximum at rho = 1 is below 1 when the tasks' spreads differ"}


def _d(sc, f, g):
    return next(d for d in sc["differences"] if d["f"] == f and d["g"] == g)


def _decided(d):
    """+1 when the interval lies above zero (g better), -1 below, 0 when it covers zero or is missing."""
    ci = d.get("ci95")
    if not ci:
        return 0
    return 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0


def summarise(res):
    fam = [c for c in res if c.startswith("bbh_family:")]
    s = {}
    for f, g in transfer_d2.COMPARISONS:
        v = [_decided(_d(res[c], f, g)) for c in fam]
        s[f"{f}_vs_{g}"] = {"families": len(fam), "g_better": v.count(1), "f_better": v.count(-1),
                            "undecided": v.count(0)}
    s["best_rule_by_risk"] = {c: min((r for r, x in sc["risk_pp2"].items() if x is not None),
                                     key=lambda r: sc["risk_pp2"][r]) for c, sc in res.items()}
    s["note"] = ("family counts read each paired difference's 95% jackknife interval against zero; the families are "
                 "observed cases, not a sample of future tasks")
    return s


def fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def macros(out):
    m = {}
    tag = {"gsm8k_to_bbh": "GsmToBbh", "bbh_to_gsm8k": "BbhToGsm"}
    for c, t in tag.items():
        sc = out["cases"][c]
        for r, x in sc["risk_pp2"].items():
            m["tr" + t + "Risk" + "".join(w.capitalize() for w in r.replace("24", "full").replace("1", "one")
                                          .replace("2", "two").replace("4", "four").split("_"))] = fmt(x)
        for f, g in (("constant", "transfer"), ("transfer", "pilot_1"), ("transfer", "pilot_4"),
                     ("constant", "capability_baseline"), ("constant", "transfer_rescaled"),
                     ("transfer", "transfer_rescaled"), ("transfer_rescaled", "pilot_1")):
            d = _d(sc, f, g)
            stem = "tr" + t + "".join(w.capitalize() for w in f"{f} vs {g}".replace("_1", " one")
                                      .replace("_4", " four").replace("_", " ").split())
            m[stem] = fmt(d["estimate"])
            if d.get("ci95"):
                m[stem + "Lo"], m[stem + "Hi"] = fmt(d["ci95"][0]), fmt(d["ci95"][1])
        m["tr" + t + "SlopeBaseline"] = fmt(sc["slopes"]["capability_baseline_pp_per_pp"], 3)
    s = out["summary"]
    m["trFamilies"] = str(s["transfer_vs_pilot_1"]["families"])
    for key, stem in (("transfer_vs_pilot_1", "PilotOne"), ("transfer_vs_pilot_4", "PilotFour"),
                      ("constant_vs_transfer", "Transfer"), ("constant_vs_capability_baseline", "CapBase"),
                      ("constant_vs_transfer_rescaled", "Rescaled"), ("transfer_vs_transfer_rescaled", "RescaledRaw"),
                      ("transfer_rescaled_vs_pilot_1", "RescaledPilotOne")):
        m["trFam" + stem + "Better"] = str(s[key]["g_better"])
        m["trFam" + stem + "Worse"] = str(s[key]["f_better"])
    for case, tg in (("discovery_24", ""), ("all_48", "All")):
        c = out["correlation"][case]
        for k, stem in (("pearson", "Pearson"), ("spearman", "Spearman"), ("latent", "Latent")):
            m["trCorr" + tg + stem] = fmt(c[k]["estimate"], 2)
            if c[k].get("ci95"):
                m["trCorr" + tg + stem + "Lo"], m["trCorr" + tg + stem + "Hi"] = (fmt(x, 2) for x in c[k]["ci95"])
    c = out["correlation"]["discovery_24"]
    m["trSdGsm"], m["trSdBbh"] = fmt(c["sd_mean_gain_a"]["estimate"]), fmt(c["sd_mean_gain_b"]["estimate"])
    m["trVdeltaGsm"], m["trVdeltaBbh"] = fmt(c["V_delta_a"]["estimate"], 0), fmt(c["V_delta_b"]["estimate"], 0)
    sh = out["correlation"]["share"]
    m["trShareObserved"] = fmt(100 * sh["observed"], 0)
    m["trShareMax"] = fmt(100 * sh["maximum_at_rho_one"], 0)
    m["trShareGsmPart"] = fmt(100 * sh["gsm8k_part_of_within"], 0)
    for k in m:
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    return m


def write_tables(out):
    td = os.path.join(OUT, "tables")
    os.makedirs(td, exist_ok=True)
    rows, diffs = [], []
    for c, sc in out["cases"].items():
        for r, x in sc["risk_pp2"].items():
            rows.append({"case": c, "rule": r, "risk_pp2": x})
        for d in sc["differences"]:
            ci = d.get("ci95") or [None, None]
            diffs.append({"case": c, "f": d["f"], "g": d["g"], "difference_pp2": d["estimate"],
                          "jackknife_se": d.get("jackknife_se"), "ci95_lo": ci[0], "ci95_hi": ci[1]})
    corr = [{"draws": case, "quantity": k, "estimate": v["estimate"], "jackknife_se": v.get("jackknife_se"),
             "ci95_lo": (v.get("ci95") or [None, None])[0], "ci95_hi": (v.get("ci95") or [None, None])[1]}
            for case in ("discovery_24", "all_48") for k, v in out["correlation"][case].items() if isinstance(v, dict)]
    pd.DataFrame(corr).to_csv(os.path.join(td, "correlation.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(td, "risks.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(td, "differences.csv"), index=False)


def write_summary(out):
    cs = out["cases"]
    L = ["# Analysis T across tasks: predicting a model's benefit on a task", "",
         f"Panel: the {out['M']} models that both GSM8K and BIG-Bench Hard retained. Chain of thought against direct "
         f"answer, strict scoring, percentage points. Rules are fitted on the {out['K_discovery']} discovery draws of "
         f"each task and saved to `{out['frozen_predictions']['path']}` ({out['frozen_predictions']['created']}) "
         f"before the validation draws are built; they are scored on the target's {out['K_validation']} validation "
         "draws. Every rule predicts the model-centred benefit, so the task's mean gain is not part of the score. "
         "The transfer rule is the one in `results/dataset2/tasks.json`; its validation draws were read there "
         "first, so the rules beside it are the new part. Intervals are leave-one-draw-out jackknife t intervals and "
         "are descriptive (the calibration did not cover these targets). Risks are cross-product risks in pp² and "
         f"may be negative. The rescaled transfer rule was added later and saved to "
         f"`{out['frozen_rescaled']['path']}` ({out['frozen_rescaled']['created']}) before this run built the "
         "validation draws, which earlier analyses had read.", ""]
    cr = out["correlation"]

    def ci(x, nd=2):
        return "--" if not x.get("ci95") else f"[{fmt(x['ci95'][0], nd)}, {fmt(x['ci95'][1], nd)}]"
    L += ["## Correlation of model gains across the two tasks", "",
          "Bundle r pairs the r-th draw of each task (independent draws from two banks). Pearson and Spearman are of "
          "the model mean gains; the latent correlation is the cross-draw moment of the two tasks' gains over "
          "√(V_delta,GSM8K · V_delta,BBH), which removes prompt noise from the mean gains. Leave-one-bundle-out "
          "jackknife t intervals, descriptive. They cover prompt draws for these 28 models only; a correlation "
          "across 28 models carries far more uncertainty about another panel of models (a Fisher-z interval at "
          f"M = {out['M']} would be about ±{1.96 / np.sqrt(out['M'] - 3):.2f} on the z scale), which the fixed-panel "
          "design does not address.", "",
          "| Draws | Pearson | Spearman | Latent | SD of mean gain, GSM8K | SD, BBH | V_delta GSM8K | V_delta BBH |",
          "|---|---|---|---|---:|---:|---:|---:|"]
    for case, title in (("discovery_24", "24 discovery bundles"), ("all_48", "all 48 (sensitivity)")):
        c = cr[case]
        L.append(f"| {title} | {fmt(c['pearson']['estimate'], 2)} {ci(c['pearson'])} | "
                 f"{fmt(c['spearman']['estimate'], 2)} {ci(c['spearman'])} | "
                 f"{fmt(c['latent']['estimate'], 2)} {ci(c['latent'])} | {fmt(c['sd_mean_gain_a']['estimate'])} | "
                 f"{fmt(c['sd_mean_gain_b']['estimate'])} | {fmt(c['V_delta_a']['estimate'], 0)} | "
                 f"{fmt(c['V_delta_b']['estimate'], 0)} |")
    sh = cr["share"]
    L += ["", "### The two-task profile share against the correlation", "",
          f"`results/dataset2/tasks.json` reports the average task profile as {100 * sh['observed']:.1f}% of the true "
          f"within-task variation. With equal task weights the share is (V_G + V_B + 2ρ s_G s_B) / (2(V_G + V_B)); "
          f"at the latent ρ = {fmt(sh['rho_latent'], 3)} it is {100 * sh['at_latent']:.1f}%, the same number. "
          "Against ρ:", "", "| ρ | " + " | ".join(sh["by_rho"]) + " |", "|---" * (len(sh["by_rho"]) + 1) + "|",
          "| share | " + " | ".join(f"{100 * v:.1f}%" for v in sh["by_rho"].values()) + " |", "",
          f"Limits: the share is at least 50% for any ρ ≥ 0, so 50% means no shared profile, and its maximum at ρ = 1 "
          f"is {100 * sh['maximum_at_rho_one']:.1f}%, because GSM8K carries {100 * sh['gsm8k_part_of_within']:.0f}% "
          "of the within-task variation. The share is a function of ρ and the two spreads; the correlation is the "
          "direct statement.", ""]
    for c, title in (("gsm8k_to_bbh", "GSM8K to BIG-Bench Hard"), ("bbh_to_gsm8k", "BIG-Bench Hard to GSM8K")):
        sc = cs[c]
        ex = sc["expected_risk_pp2"]
        L += [f"## {title}", "", "| Rule | Risk (pp²) | Expected from the discovery moments |", "|---|---:|---:|"]
        L += [f"| {RULE_NAMES[r]} | {fmt(x)} | {fmt(ex[r]) if r in ex else ''} |" for r, x in sc["risk_pp2"].items()]
        L += ["", "Expected: constant V_T, transfer V_S + V_T − 2 cov, rescaled 2(1 − ρ) V_T ("
                  "stated before computing). The rescaled rule beats the constant only when ρ > ½."]
        L += ["", f"Source slopes: {fmt(sc['slopes']['capability_baseline_pp_per_pp'], 3)} pp of gain per pp of "
                  f"baseline; {fmt(sc['slopes']['capability_eci_pp_per_eci'], 3)} pp per ECI point. Rescaling: "
                  f"source SD {fmt(sc['rescale']['sd_source_pp'])} pp, target discovery SD "
                  f"{fmt(sc['rescale']['sd_target_discovery_pp'])} pp.", "",
              "| f | g | g better by (pp²) | 95% interval |", "|---|---|---:|---|"]
        for d in sc["differences"]:
            ci = d.get("ci95")
            L.append(f"| {RULE_NAMES[d['f']]} | {RULE_NAMES[d['g']]} | {fmt(d['estimate'])} | "
                     f"{'--' if not ci else f'[{fmt(ci[0])}, {fmt(ci[1])}]'} |")
        L.append("")
    s = out["summary"]
    L += ["## Within BIG-Bench Hard: held-out families", "",
          f"Each of the {s['transfer_vs_pilot_1']['families']} families is predicted from the other families' tasks. "
          "Counts read each paired difference's interval against zero.", "",
          "| f | g | g better | f better | undecided |", "|---|---|---:|---:|---:|"]
    for f, g in transfer_d2.COMPARISONS:
        x = s[f"{f}_vs_{g}"]
        L.append(f"| {RULE_NAMES[f]} | {RULE_NAMES[g]} | {x['g_better']} | {x['f_better']} | {x['undecided']} |")
    best = sorted(set(s["best_rule_by_risk"].values()))
    L += ["", "Lowest risk per case: " + ("; ".join(f"{c} {RULE_NAMES[r]}" for c, r in s["best_rule_by_risk"].items())
                                          if len(best) > 1 else f"{RULE_NAMES[best[0]]} in all "
                                          f"{len(s['best_rule_by_risk'])} cases") + ".", "",
          "SDs here use denominator M − 1; with denominator M they are "
          f"{fmt(cr['discovery_24']['sd_mean_gain_a']['estimate'] * np.sqrt((out['M'] - 1) / out['M']))} and "
          f"{fmt(cr['discovery_24']['sd_mean_gain_b']['estimate'] * np.sqrt((out['M'] - 1) / out['M']))} pp.", ""]
    txt = "\n".join(L)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    from src import d2_io, d2_panels
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    try:
        s_g = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8"))
        s_b = json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
        items_g, items_info = d2_io.load_items(split="test")
        gsm = d2_panels.inputs(None, None, s_g, items_info, os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
        gsm["items"] = items_g["item_id"].tolist()
        bbh = d2_panels.bbh_inputs(None, s_b)
    except (OSError, ValueError, d2_io.D2IOError, d2_panels.PanelError) as exc:
        sys.exit(f"refused: {exc}")
    run(gsm, bbh, s_g, s_b, a.overwrite_frozen)


if __name__ == "__main__":
    main()
