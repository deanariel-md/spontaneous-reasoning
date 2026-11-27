r"""Content classes, format compliance as its own outcome, and BIG-Bench Hard slopes by family on
content scores. The per-outcome slopes, shares, V_delta, own-gain risks and map points on
content scores are scripts/run_outcomes.py --scoring content (results/outcomes_content/).

PowerShell (from the repository root), after scripts/score_content.py:
    cmd /c ".venv\Scripts\python.exe scripts\run_content.py 2>&1"

Reads data/processed/content/<task>_content.parquet for GSM8K-29 and BIG-Bench Hard, and for MATH-500 once
scripts/score_content.py --tasks math has written it (macros with the tag Math). Class shares are over every scored
output of the panel (both splits); everything else is on the 24 discovery draws. Outputs: results/content_scoring/content_scoring.json
(macros, prefix cs), summary.md, summary_clean.md and tables/*.csv.
"""
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
from src import capability_checks as cc, content_scoring as cs, dataset2 as d2, instability as ins  # noqa: E402
from src import reliability_map as rm, tasks_d2  # noqa: E402
from run_outcomes import CONFIG, OUTCOMES, SELECTION, TASKS, content_tasks, point_slopes  # noqa: E402
from run_index_checks import fmt, jsonable, set_txt, write  # noqa: E402

CONTENT = os.path.join(ROOT, "data", "processed", "content")
OUT = os.path.join(ROOT, "results", "content_scoring")
ARMS = {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot"}
ARM_TAG = {"requested": "Req", "forced": "Forced", "cot": "Cot"}
VALUES = ("strict_correct", "content_correct", "compliant")
CLASS_TAG = {"correct": "Correct", "wrong": "Wrong", "unreadable": "Unreadable", "malformed_box": "Malformed",
             "truncated": "Truncated"}


def arrays(tab, panel, units, value, key="config_id"):
    """{arm: (K, M) pp} of one value on the given units (one row per unit, repeats kept)."""
    g = tab.groupby(["arm", "model_key", key])[value].mean().mul(100)
    out = {}
    for a, arm in ARMS.items():
        s = g.loc[arm]
        out[a] = np.array([[s.loc[(m, u)] for m in panel] for u in units], float)
    return out


def mean_corr(X, Y):
    """Pearson correlation of the model means of two (K, M) arrays, with the leave-one-draw-out jackknife."""
    def r(x, y):
        return float(np.corrcoef(x.mean(0), y.mean(0))[0, 1])
    est = r(X, Y)
    loo = np.array([r(np.delete(X, u, 0), np.delete(Y, u, 0)) for u in range(X.shape[0])])
    return cc._jack(est, loo)


def ordering(X):
    rel = rm.reliability(X)
    ag = rm.agreement(X)
    return {"R": rel["R"], "C": ag["C"], "C_ref": rm.reference_c(rel["R"])}


def task_block(task, tab, panel, units, eci, method, alpha, n_items):
    sub = tab[tab["model_key"].isin(panel)]
    cls = (sub.groupby("arm")["content_class"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
           .reindex(columns=list(cs.CLASSES), fill_value=0))
    T = {"classes_pct": {a: {c: float(cls.loc[arm, c]) for c in cs.CLASSES} for a, arm in ARMS.items()},
         "n_outputs": int(len(sub))}
    disc = sub[sub["config_id"].isin(set(units))]
    X = {v: arrays(disc, panel, units, v) for v in VALUES}
    comp, per_model = {}, []
    for a in ARMS:
        C, S, Ct = X["compliant"][a], X["strict_correct"][a], X["content_correct"][a]
        mm = C.mean(0)
        comp[a] = {"mean": float(mm.mean()), "sd_models": float(mm.std(ddof=1)),
                   "min": {"model": panel[int(mm.argmin())], "pct": float(mm.min())},
                   "max": {"model": panel[int(mm.argmax())], "pct": float(mm.max())},
                   "ordering": ordering(C), "corr_index": cc.correlation_with_index(C, eci),
                   "format_gap": {"mean": float((Ct - S).mean()), "max": float((Ct - S).mean(0).max()),
                                  "max_model": panel[int((Ct - S).mean(0).argmax())],
                                  "corr_index": cc.correlation_with_index(Ct - S, eci)}}
    gains = {}
    for o, (b, p, _) in OUTCOMES.items():
        Gc = X["content_correct"][p] - X["content_correct"][b]
        Gs = X["strict_correct"][p] - X["strict_correct"][b]
        gains[o] = {"content_mean": float(Gc.mean()), "strict_mean": float(Gs.mean()),
                    "corr_baseline_compliance": mean_corr(X["compliant"][b], Gc),
                    "corr_compliance_difference": mean_corr(X["compliant"][p] - X["compliant"][b], Gc)}
    for j, m in enumerate(panel):
        r = {"model_key": m, "eci": eci[j]}
        for a in ARMS:
            for v in VALUES:
                r[f"{a}_{v}"] = float(X[v][a][:, j].mean())
        per_model.append(r)
    T.update({"compliance": comp, "outcomes": gains, "per_model": per_model})
    if task == "bbh":
        T["families"] = family_block(disc, panel, units, eci, method, alpha)
    return T


def family_block(disc, panel, units, eci, method, alpha):
    d = disc.assign(family=disc["item_id"].str.split("/").str[0].map(tasks_d2.bbh_family))
    out = {}
    K = len(units)
    for fam, g in d.groupby("family"):
        Xs = {v: arrays(g, panel, units, v) for v in ("strict_correct", "content_correct")}
        out[fam] = {"n_items": int(g["item_id"].nunique())}
        for v in Xs:
            out[fam][v] = {}
            for o, (b, p, _) in OUTCOMES.items():
                B, P = Xs[v][b], Xs[v][p]
                out[fam][v][o] = {"mean": float((P - B).mean()),
                                  **point_slopes(B, P, {"index": cc.tile(eci, K), "baseline": B}, method, alpha)}
    return out


def run():
    t0 = time.time()
    from run_transfer_tasks import eci_for
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    R = {"analysis": "content classes, format compliance and BIG-Bench Hard families",
         "methods": "content classes and format compliance", "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method,
         "classes": list(cs.CLASSES), "tasks": {}}
    for task in content_tasks():
        t = TASKS[task]
        panel = json.load(open(os.path.join(ROOT, "results", "dataset2", f"{t['collection']}.json"),
                               encoding="utf-8"))["models"]
        s = json.load(open(os.path.join(ROOT, "manifests", f"sampler_{task}.json"), encoding="utf-8"))
        tab = pd.read_parquet(os.path.join(CONTENT, f"{task}_content.parquet"))
        tab["config_id"] = tab["config_id"].astype(str)
        units = [str(u) for u in s["discovery_config_ids"]]
        n_items = int(tab["item_id"].nunique())
        R["tasks"][task] = {"models": panel, "M": len(panel), "K": len(units), "n_items": n_items,
                            **task_block(task, tab, panel, units, eci_for(panel), method, alpha, n_items)}
        print(f"{task} done ({time.time() - t0:.0f} s)", flush=True)
    R["macros"] = macros(R)
    tables(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write("content_scoring", R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/content_scoring", flush=True)


def _ci(m, stem, d, nd=2):
    m[stem] = fmt(d["estimate"], nd)
    if d.get("ci95"):
        m[stem + "Lo"], m[stem + "Hi"] = fmt(d["ci95"][0], nd), fmt(d["ci95"][1], nd)


def macros(R):
    m = {}
    for task, T in R["tasks"].items():
        t = TASKS[task]["tag"]
        for a, at in ARM_TAG.items():
            for c, ct in CLASS_TAG.items():
                m[f"cs{t}{at}{ct}Pct"] = fmt(T["classes_pct"][a][c], 1)
            x = T["compliance"][a]
            s = f"cs{t}{at}Compliant"
            m[s + "Mean"], m[s + "Sd"] = fmt(x["mean"], 1), fmt(x["sd_models"], 1)
            m[s + "Min"], m[s + "Max"] = fmt(x["min"]["pct"], 1), fmt(x["max"]["pct"], 1)
            m[s + "R"], m[s + "C"] = fmt(x["ordering"]["R"], 3), fmt(x["ordering"]["C"], 3)
            _ci(m, s + "CorrIndex", x["corr_index"])
            g = x["format_gap"]
            m[f"cs{t}{at}FormatGapMean"], m[f"cs{t}{at}FormatGapMax"] = fmt(g["mean"], 1), fmt(g["max"], 1)
            _ci(m, f"cs{t}{at}FormatGapCorrIndex", g["corr_index"])
        for o, x in T["outcomes"].items():
            s = f"cs{t}{o}"
            m[s + "ContentMean"], m[s + "StrictMean"] = fmt(x["content_mean"], 1), fmt(x["strict_mean"], 1)
            _ci(m, s + "CorrBaseCompliance", x["corr_baseline_compliance"])
            _ci(m, s + "CorrComplianceDiff", x["corr_compliance_difference"])
        if "families" in T:
            fams = T["families"]
            m[f"cs{t}Families"] = str(len(fams))
            for o in OUTCOMES:
                for cap, ct in (("index", "Index"), ("baseline", "Base")):
                    for v, vt in (("strict_correct", "Strict"), ("content_correct", "Content")):
                        signs = [f[v][o][cap]["set_sign"] for f in fams.values()]
                        m[f"cs{t}{o}{ct}Fam{vt}Pos"] = str(sum(x == 1 for x in signs))
                        m[f"cs{t}{o}{ct}Fam{vt}Neg"] = str(sum(x == -1 for x in signs))
                    flips = sum(np.sign(f["strict_correct"][o][cap]["slope"]) != np.sign(f["content_correct"][o][cap]["slope"])
                                for f in fams.values())
                    m[f"cs{t}{o}{ct}FamSignFlips"] = str(int(flips))
    return m


def tables(R):
    d = os.path.join(OUT, "tables")
    os.makedirs(d, exist_ok=True)
    for task, T in R["tasks"].items():
        pd.DataFrame(T["per_model"]).to_csv(os.path.join(d, f"compliance_{task}.csv"), index=False)
    rows = []
    for fam, f in R["tasks"]["bbh"]["families"].items():
        for v in ("strict_correct", "content_correct"):
            for o in OUTCOMES:
                for cap in ("index", "baseline"):
                    x = f[v][o][cap]
                    rows.append({"family": fam, "n_items": f["n_items"], "scoring": v.split("_")[0], "outcome": o,
                                 "capability": cap, "mean_gain": f[v][o]["mean"], "slope": x["slope"],
                                 "set_kind": x["slope_set"]["kind"], "lo": x["slope_set"]["a"],
                                 "hi": x["slope_set"]["b"], "set_sign": x["set_sign"]})
    pd.DataFrame(rows).to_csv(os.path.join(d, "bbh_families.csv"), index=False)


def summary(R):
    L = ["# Content-level scoring", "",
         "Reading rules are in `src/content_scoring.py`. Every output "
         "of the three arms is classified as correct, wrong, unreadable, malformed box or truncated; the content score "
         "is 1 for correct. Class shares are over all scored outputs of the panel; compliance, format gaps and "
         "correlations are on the 24 discovery draws, with leave-one-draw-out jackknife t intervals (descriptive). "
         "Per-outcome slopes, shares, V_delta, own-gain risks and map points on content scores are in "
         "`results/outcomes_content/summary.md`.", ""]
    for task, T in R["tasks"].items():
        L += [f"## {TASKS[task]['name']} ({T['M']} models, {T['n_outputs']:,} outputs)", "",
              "| Arm | Correct | Wrong | Unreadable | Malformed box | Truncated |", "|---|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            c = T["classes_pct"][a]
            L.append(f"| {a} | " + " | ".join(f"{fmt(c[k], 2)}" for k in cs.CLASSES) + " |")
        L += ["", "Format compliance (strict format) as an outcome, and the format gap (content − strict accuracy):",
              "", "| Arm | Compliance mean | SD across models | Lowest | R | C | Corr. with index | Format gap mean | "
              "Largest gap | Gap corr. with index |", "|---|---:|---:|---|---:|---:|---|---:|---|---|"]
        for a in ARMS:
            x = T["compliance"][a]
            g = x["format_gap"]
            ci = x["corr_index"]
            gi = g["corr_index"]
            L.append(f"| {a} | {fmt(x['mean'], 1)} | {fmt(x['sd_models'], 1)} | {x['min']['model']} "
                     f"{fmt(x['min']['pct'], 1)} | {fmt(x['ordering']['R'], 3)} | {fmt(x['ordering']['C'], 3)} | "
                     f"{fmt(ci['estimate'], 2)} [{fmt(ci['ci95'][0], 2)}, {fmt(ci['ci95'][1], 2)}] | "
                     f"{fmt(g['mean'], 1)} | {g['max_model']} {fmt(g['max'], 1)} | "
                     f"{fmt(gi['estimate'], 2)} [{fmt(gi['ci95'][0], 2)}, {fmt(gi['ci95'][1], 2)}] |")
        L += ["", "Gains (pp) and their correlation with compliance across models:", "",
              "| Outcome | Strict mean | Content mean | Corr. with baseline-arm compliance | "
              "Corr. with the compliance difference |", "|---|---:|---:|---|---|"]
        for o, x in T["outcomes"].items():
            a_, b_ = x["corr_baseline_compliance"], x["corr_compliance_difference"]
            L.append(f"| {o} | {fmt(x['strict_mean'], 1)} | {fmt(x['content_mean'], 1)} | "
                     f"{fmt(a_['estimate'], 2)} [{fmt(a_['ci95'][0], 2)}, {fmt(a_['ci95'][1], 2)}] | "
                     f"{fmt(b_['estimate'], 2)} [{fmt(b_['ci95'][0], 2)}, {fmt(b_['ci95'][1], 2)}] |")
        if "families" in T:
            L += ["", f"Slopes by family ({len(T['families'])} families; points, pp of gain per index point / per pp "
                  "of baseline), strict and content scores:", "",
                  "| Family | Items | Outcome | Strict: index | Content: index | Strict: baseline | Content: baseline |",
                  "|---|---:|---|---|---|---|---|"]
            for fam, f in T["families"].items():
                for o in OUTCOMES:
                    cells = [f"{fmt(f[v][o][cap]['slope'], 3)} {set_txt(f[v][o][cap]['slope_set'])}"
                             for cap in ("index", "baseline") for v in ("strict_correct", "content_correct")]
                    L.append(f"| {fam} | {f['n_items']} | {o} | " + " | ".join(cells) + " |")
        L.append("")
    return L


if __name__ == "__main__":
    run()
