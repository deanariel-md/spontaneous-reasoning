r"""Threshold decision (exploratory): each model gets
chain of thought when its pilot gain exceeds c points, c = 2, 5, 10, decided under the requested-direct baseline and
under the forced-direct one, and valued under each.

From the repository root:
    python scripts/threshold_decision.py

Inputs are the main decision analysis's (scripts/run_decision.py; both directions,
scripts/run_decision_directions.py): the saved one-draw pilot gains, rule own_1 of
manifests/predictions_decision_<task>_req and _fd (.json/.npz; one pilot per distinct discovery configuration, the same
configuration for both baselines), the 29-model panel (28 on BIG-Bench Hard), strict scores (MATH-500: the numeric
items), and the validation configurations no discovery draw uses, each once (run_decision.evaluation_rows,
"independent"). The decisions are a fixed function of the saved pilot gains and the three cuts; the validation draws had
been read by earlier analyses, so this values a rule set after those reads.

Decision under baseline a for pilot set s: d_a[s, m] = 1 when pilot gain_a[s, m] > c, else 0. A pilot gain within 1e-9
pp of c is a tie: it does not exceed c and gets no chain of thought; ties are counted and named, and the losses are
recomputed with ties given chain of thought.

Value of a decision on a validation configuration under deployment arm b, net of c (the primary reading):
(1/M) sum_m d[m] (CoT_m - b_m - c), pp of accuracy per panel model: a model given chain of thought is worth its gain
minus the c points the rule asks of it, so the rule "gain above c" is the deployment baseline's own plug-in best and a
decision made under the other baseline can only lose in expectation. Gross (the second reading): (1/M) sum_m d[m]
(CoT_m - b_m), the gain alone; then any decision that gives chain of thought to more models with positive gains is
worth more, whatever c. The two differ by c times the difference in the mean number of models given chain of thought,
over M. Both use src/decision.values with n = M. Added loss of deciding under a when deploying under b: value of d_b
minus value of d_a under b, averaged over pilot sets, with a 95% t interval over the validation configurations
(src/decision.paired_difference; positive: the wrong baseline costs points).

Outputs: results/threshold_decision/threshold_decision.json (with macros), summary.md, summary_clean.md,
tables/cuts.csv (one row per benchmark and cut), tables/decisions.csv (every pilot set, model, baseline and cut).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import decision as dc, decision_directions as dd, rule_budgets as rbud  # noqa: E402

OUT = os.path.join(ROOT, "results", "threshold_decision")
MAN = os.path.join(ROOT, "manifests")
TASKS = ("gsm8k", "math", "bbh")
TITLE = {"gsm8k": "GSM8K-29", "math": "MATH-500 numeric", "bbh": "BIG-Bench Hard-28"}
ARM = {"requested": "req", "forced": "fd"}
CUTS = (2.0, 5.0, 10.0)
TIE_TOL = dc.TIE_TOL
RULE = "own_1"
DIRECTIONS = {"forced_to_requested": ("forced", "requested"), "requested_to_forced": ("requested", "forced")}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ============================================================================ the rule and its valuation


def decide(pred, c, tol=TIE_TOL, tie_gets_cot=False):
    """(decisions, ties): 1.0 where the pilot gain exceeds c by more than tol; a gain within tol of c is a tie and gets
    chain of thought only when tie_gets_cot."""
    p = np.asarray(pred, float)
    if not np.all(np.isfinite(p)):
        raise ValueError("a pilot gain is not finite")
    tie = np.abs(p - c) <= tol
    d = (p > c + tol) | (tie & bool(tie_gets_cot))
    return d.astype(float), tie


def value_matrix(D, G):
    """(pilot sets, configurations) value of each decision: pp of accuracy per panel model."""
    D = np.atleast_2d(np.asarray(D, float))
    return dc.values(D, G, D.shape[1])


def losses(D: dict, G: dict) -> dict:
    """Added loss per panel model of deciding under one baseline and deploying under the other, both directions."""
    M = np.atleast_2d(D["requested"]).shape[1]
    return {k: dc.paired_difference(D[src], D[dst], G[dst], M) for k, (src, dst) in DIRECTIONS.items()}


def counts(D, M):
    k = np.atleast_2d(D).sum(1)
    return {"mean": float(k.mean()), "min": int(k.min()), "max": int(k.max()),
            "pilot_sets_every_model": int((k == M).sum()), "pilot_sets_no_model": int((k == 0).sum())}


def valuation(D: dict, G: dict) -> dict:
    """Both losses and the values behind them, for gains G (net of c or gross)."""
    M = np.atleast_2d(D["requested"]).shape[1]
    out = losses(D, G)
    vals = {}
    for b in ARM:
        a = "forced" if b == "requested" else "requested"
        gm = np.asarray(G[b], float).mean(0)
        vals[b] = {"matched": float(value_matrix(D[b], G[b]).mean()), "mismatched": float(value_matrix(D[a], G[b]).mean()),
                   "every_model": float(gm.mean()), "best_per_model_plugin": float(np.clip(gm, 0, None).sum() / M)}
    out["values"] = vals
    return out


def cut_block(pred: dict, G: dict, c: float, models, units_of_set) -> dict:
    """Everything reported for one cut: decisions under each baseline, the models whose decision differs, ties at c,
    both added losses and the values behind them, net of c (primary: a model given chain of thought is worth its gain
    minus c) and gross (its gain), and, if a tie exists, both readings with ties given chain of thought."""
    D, T = {}, {}
    for a in ARM:
        D[a], T[a] = decide(pred[a], c)
    S, M = D["requested"].shape
    diff = (D["requested"] != D["forced"]).sum(1)
    Gnet = {a: np.asarray(G[a], float) - c for a in ARM}
    out = {"cut": float(c), "n_pilot_sets": int(S), "M": int(M),
           "cot": {a: counts(D[a], M) for a in ARM},
           "differ": {"mean_count": float(diff.mean()), "share": float(diff.mean() / M), "min": int(diff.min()),
                      "max": int(diff.max()), "pilot_sets_with_none": int((diff == 0).sum()),
                      "cot_under_requested_only": float(((D["requested"] == 1) & (D["forced"] == 0)).sum(1).mean()),
                      "cot_under_forced_only": float(((D["forced"] == 1) & (D["requested"] == 0)).sum(1).mean())},
           "ties": {a: {"cells": int(T[a].sum()),
                        "where": [{"pilot_set": int(s), "configuration": units_of_set[s], "model": models[m],
                                   "pilot_gain": float(pred[a][s, m])} for s, m in zip(*np.nonzero(T[a]))]}
                    for a in ARM}}
    out.update(valuation(D, Gnet))
    out["gross"] = valuation(D, G)
    # gross = net + c * (mean count given CoT under the deployment baseline - under the deciding one) / M
    ident = {}
    for k, (src, dst) in DIRECTIONS.items():
        want = out[k]["estimate"] + c * (out["cot"][dst]["mean"] - out["cot"][src]["mean"]) / M
        ident[k] = abs(want - out["gross"][k]["estimate"])
    out["identity_gross_net_max_abs_diff"] = float(max(ident.values()))
    if any(out["ties"][a]["cells"] for a in ARM):
        Dt = {a: decide(pred[a], c, tie_gets_cot=True)[0] for a in ARM}
        dt = (Dt["requested"] != Dt["forced"]).sum(1)
        net = losses(Dt, Gnet)
        out["ties_get_cot"] = {"differ": {"mean_count": float(dt.mean()), "share": float(dt.mean() / M)},
                               **net, "gross": losses(Dt, G)}
    return out


# ============================================================================ inputs (the main decision analysis's)


def task_inputs(task):
    """Saved one-draw pilot gains {arm: (sets, M)}, validation gains {arm: (K, M)} on the independent configurations,
    and the facts that tie them to the main decision analysis."""
    import run_decision as rd
    rec, pred, rows = {}, {}, {}
    for a, s in ARM.items():
        name = f"{task}_{s}"
        jp = os.path.join(MAN, f"predictions_decision_{name}.json")
        rec[a], arr = rbud.load_frozen(jp, jp[:-5] + ".npz")
        pred[a], rows[a] = arr[f"{RULE}__pred"], arr[f"{RULE}__rows"]
    if rec["requested"]["models"] != rec["forced"]["models"] or \
            rec["requested"]["discovery_units"] != rec["forced"]["discovery_units"] or \
            not np.array_equal(rows["requested"], rows["forced"]):
        raise SystemExit(f"{task}: the two baselines' saved pilots differ in panel, discovery draws or pilot rows")
    models, disc = rec["requested"]["models"], rec["requested"]["discovery_units"]
    units_of_set = [str(disc[int(r[0])]) for r in rows["requested"]]
    # the saved pilot gains must equal the discovery gains of their configuration in the current strict arrays
    repro = {}
    for a, s in ARM.items():
        u, p, B, P = rd.setting_arrays(f"{task}_{s}", "discovery")
        if u != disc or p != models:
            raise SystemExit(f"{task}_{s}: discovery arrays differ from the saved pilots in draws or panel")
        G = P - B
        again = np.array([G[list(r)].mean(0) for r in rows[a]])
        repro[a] = float(np.abs(again - pred[a]).max())
    val = {a: rd.setting_arrays(f"{task}_{s}", "validation") for a, s in ARM.items()}
    vunits, vpanel = val["requested"][0], val["requested"][1]
    if val["forced"][0] != vunits or val["forced"][1] != vpanel or vpanel != models:
        raise SystemExit(f"{task}: validation draws or panel differ between the baselines")
    ev = rd.evaluation_rows(disc, vunits)["independent"]
    G = {a: (val[a][3] - val[a][2])[ev] for a in ARM}
    W_top = {}
    for a, s in ARM.items():
        _, W = rd.load_choices(f"{task}_{s}")
        W_top[a] = W[RULE]
    return {"models": models, "pred": pred, "G": G, "units_of_set": units_of_set,
            "validation_configurations": [vunits[r] for r in ev], "n_chosen_top_third": rec["requested"]["n_chosen"],
            "saved": {a: {"path": f"manifests/predictions_decision_{task}_{s}.json", "sha256": rec[a].get("sha256"),
                          "created": rec[a].get("created")} for a, s in ARM.items()},
            "pilot_gain_reproduced_max_abs_diff": repro, "W_top_third": W_top}


def check_top_third(task, x):
    """The top-third own_1 costs recomputed on these validation gains must equal results/decision_directions/."""
    n = x["n_chosen_top_third"]
    here = dd.both_directions(x["W_top_third"], x["G"], n)
    D = json.load(open(os.path.join(ROOT, "results", "decision_directions", "decision_directions.json"), encoding="utf-8"))
    printed = D["tasks"][task]["one_draw"]["rules"][RULE]
    M = len(x["models"])
    out = {}
    for k in DIRECTIONS:
        out[k] = {"printed": printed[k]["estimate"], "here": here[k]["estimate"],
                  "abs_difference": abs(printed[k]["estimate"] - here[k]["estimate"]),
                  # the same cost in this analysis's unit (pp per panel model): per chosen model x n / M
                  "per_panel_model": here[k]["estimate"] * n / M,
                  "per_panel_model_ci95": [v * n / M for v in here[k]["ci95"]]}
    out["ok"] = all(v["abs_difference"] < 1e-9 for v in out.values() if isinstance(v, dict))
    return out


# ============================================================================ report


def f2(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}".replace("-", "−")


def ci(d):
    c = d.get("ci95")
    return f"{f2(d['estimate'])} [{f2(c[0])}, {f2(c[1])}]" if c else f2(d["estimate"])


def summary(R):
    L = ["# Threshold decision: chain of thought for every model whose pilot gain exceeds c points", "",
         "Exploratory analysis. Code "
         "`scripts/threshold_decision.py`; numbers in `threshold_decision.json` and `tables/`; macros "
         "`paper/macros_threshold_decision.py` (prefix `thrDec`).", "",
         "**Rule.** Each model gets chain of thought when its one-draw pilot gain exceeds c points (c = 2, 5, 10), "
         "decided once under the requested-direct baseline and once under the forced-direct one. A gain equal to c does "
         "not exceed it.", "",
         "**Inputs, all the main decision analysis's** (`scripts/run_decision.py`; both directions, "
         "`results/decision_directions/`): the saved one-draw pilot gains (rule own_1 of "
         "`manifests/predictions_decision_<task>_req|_fd`; one pilot per distinct discovery configuration, the same "
         "configuration under both baselines), the panel (29 models; 28 on BIG-Bench Hard), and the validation "
         "configurations that no discovery draw uses, each once. **Scoring:** strict, as in the main decision analysis "
         "(MATH-500: its numeric items). The cuts were set after the validation draws had been read by earlier "
         "analyses; the decisions themselves use discovery draws only.", "",
         "**Loss.** Added loss of deciding under the forced baseline when deploying under the requested one (forced → "
         "requested): the value of the requested-baseline decision minus that of the forced-baseline decision, both "
         "valued on requested-baseline gains of the validation configurations, averaged over pilot sets; the reverse "
         "likewise. pp of accuracy per panel model; 95% t interval over validation configurations. Positive: the wrong "
         "baseline costs points.", "",
         "Two valuations of a decision. **Net of c (the macros):** a model given chain of thought is worth its gain "
         "minus c, the threshold read as what chain of thought must earn to be worth giving; the rule is then the "
         "deployment baseline's own plug-in best. **Gross:** worth its gain alone; any decision that gives chain of "
         "thought to more models with positive gains is then worth more, whatever c, so the decision made under the "
         "baseline with the larger gains wins whenever it gives chain of thought to more models. Gross = net + c × "
         "(models given chain of thought under the deployment baseline − under the deciding one) / M.", ""]
    for t, T in R["tasks"].items():
        L += [f"## {TITLE[t]}", "",
              f"{T['M']} models; {T['n_pilot_sets']} one-draw pilot sets; {T['n_validation_configurations']} validation "
              f"configurations. For reference, the top-third rule (one-draw pilots, own gain) gives chain of thought to "
              f"{T['n_chosen_top_third']}; its costs, `results/decision_directions/` (pp per chosen model) put in this "
              f"analysis's unit (× {T['n_chosen_top_third']}/{T['M']}, pp per panel model): forced → requested "
              f"{f2(T['check_top_third']['forced_to_requested']['per_panel_model'])}, requested → forced "
              f"{f2(T['check_top_third']['requested_to_forced']['per_panel_model'])}.", "",
              "| c (pp) | Given CoT, requested (min–max) | Given CoT, forced (min–max) | "
              "Decision differs: mean count (min–max) | Share | Net: forced → requested | Net: requested → forced | "
              "Gross: forced → requested | Gross: requested → forced |",
              "|---:|---|---|---|---:|---|---|---|---|"]
        for c in T["cuts"]:
            q, f, d, g = c["cot"]["requested"], c["cot"]["forced"], c["differ"], c["gross"]
            L.append(f"| {c['cut']:g} | {q['mean']:.1f} ({q['min']}–{q['max']}) | {f['mean']:.1f} ({f['min']}–{f['max']}) | "
                     f"{d['mean_count']:.2f} ({d['min']}–{d['max']}) | {100 * d['share']:.0f}% | "
                     f"{ci(c['forced_to_requested'])} | {ci(c['requested_to_forced'])} | "
                     f"{ci(g['forced_to_requested'])} | {ci(g['requested_to_forced'])} |")
        L += ["", "Where the decisions differ (mean models per pilot set): chain of thought under the requested baseline "
              "only; under the forced baseline only.", ""]
        L += [f"- c = {c['cut']:g}: {c['differ']['cot_under_requested_only']:.2f}; "
              f"{c['differ']['cot_under_forced_only']:.2f}." for c in T["cuts"]]
        L += ["", "Values behind the losses (pp per panel model; mean over pilot sets and validation configurations):", "",
              "| c (pp) | Valuation | Deployed under | Matched decision | Mismatched decision | Every model CoT | "
              "Best per model (plug-in) |", "|---:|---|---|---:|---:|---:|---:|"]
        for c in T["cuts"]:
            for name, V in (("net of c", c["values"]), ("gross", c["gross"]["values"])):
                for b, v in V.items():
                    L.append(f"| {c['cut']:g} | {name} | {b} | {f2(v['matched'])} | {f2(v['mismatched'])} | "
                             f"{f2(v['every_model'])} | {f2(v['best_per_model_plugin'])} |")
        L.append("")
    L += ["## Edge cases, as they are", ""]
    L += [f"- {e}" for e in R["edge_cases"]]
    L += ["", "## Checks", ""]
    L += [f"- {c}" for c in R["checks"]]
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def edge_cases(R):
    out = []
    for t, T in R["tasks"].items():
        for c in T["cuts"]:
            for a in ARM:
                k = c["cot"][a]
                if k["pilot_sets_every_model"] or k["pilot_sets_no_model"]:
                    out.append(f"{TITLE[t]}, c = {c['cut']:g}, {a}: {k['pilot_sets_every_model']} pilot sets give chain "
                               f"of thought to every model, {k['pilot_sets_no_model']} to no model.")
                for w in c["ties"][a]["where"]:
                    tg = c.get("ties_get_cot", {})
                    out.append(f"{TITLE[t]}, c = {c['cut']:g}, {a} baseline: tie at exactly c: {w['model']}, pilot "
                               f"configuration {w['configuration']}, pilot gain {w['pilot_gain']:.12g} pp; it gets no chain "
                               f"of thought. Given it instead: decision differs {tg['differ']['mean_count']:.2f} models "
                               f"({100 * tg['differ']['share']:.0f}%); net forced → requested "
                               f"{ci(tg['forced_to_requested'])}, requested → forced {ci(tg['requested_to_forced'])}; "
                               f"gross {ci(tg['gross']['forced_to_requested'])} and "
                               f"{ci(tg['gross']['requested_to_forced'])}.")
            if c["differ"]["mean_count"] == 0:
                out.append(f"{TITLE[t]}, c = {c['cut']:g}: no model's decision differs between the baselines in any "
                           f"pilot set.")
            if c["differ"]["pilot_sets_with_none"]:
                out.append(f"{TITLE[t]}, c = {c['cut']:g}: {c['differ']['pilot_sets_with_none']} pilot set(s) where no "
                           f"model's decision differs.")
            for name, blk in (("net", c), ("gross", c["gross"])):
                for k in DIRECTIONS:
                    lo, hi = blk[k]["ci95"]
                    neg, zero = blk[k]["estimate"] < 0, lo <= 0 <= hi
                    if neg or zero:
                        what = ", ".join(x for x, y in (("negative", neg), ("the interval includes 0", zero)) if y)
                        out.append(f"{TITLE[t]}, c = {c['cut']:g}, {name}, {k.replace('_', ' ')}: {ci(blk[k])} ({what}).")
    every = [(t, c["cut"], a) for t, T in R["tasks"].items() for c in T["cuts"] for a in ARM
             if c["cot"][a]["pilot_sets_every_model"] or c["cot"][a]["pilot_sets_no_model"]]
    if not every:
        out.insert(0, "At no cut, on no benchmark and under neither baseline does any pilot set give chain of thought to "
                      "every model or to no model.")
    ties = sum(c["ties"][a]["cells"] for T in R["tasks"].values() for c in T["cuts"] for a in ARM)
    out.insert(1 if not every else 0, f"Ties at exactly c (within {TIE_TOL:g} pp): {ties} pilot gain(s) in all.")
    return out


def checks(R):
    out = []
    for t, T in R["tasks"].items():
        r = T["pilot_gain_reproduced_max_abs_diff"]
        out.append(f"{TITLE[t]}: saved one-draw pilot gains recomputed from the current strict discovery arrays, largest "
                   f"absolute difference {max(r.values()):.3g} pp; {'reproduced' if max(r.values()) < 1e-9 else 'MISMATCH'}.")
        ck = T["check_top_third"]
        out.append(f"{TITLE[t]}: top-third own_1 costs on these validation gains, results/decision_directions/ "
                   f"{ck['requested_to_forced']['printed']:.4f} / {ck['forced_to_requested']['printed']:.4f}, here "
                   f"{ck['requested_to_forced']['here']:.4f} / {ck['forced_to_requested']['here']:.4f} "
                   f"(requested → forced / forced → requested); {'reproduced' if ck['ok'] else 'MISMATCH'}.")
        idm = max(c["identity_gross_net_max_abs_diff"] for c in T["cuts"])
        out.append(f"{TITLE[t]}: gross = net + c × (difference in models given chain of thought) / M, largest absolute "
                   f"difference {idm:.3g} pp; {'holds' if idm < 1e-9 else 'MISMATCH'}.")
    return out


def tables(R, decisions):
    rows = []
    for t, T in R["tasks"].items():
        for c in T["cuts"]:
            row = {"task": t, "cut": c["cut"], "M": T["M"], "n_pilot_sets": T["n_pilot_sets"],
                   "n_validation_configurations": T["n_validation_configurations"],
                   "cot_requested_mean": c["cot"]["requested"]["mean"], "cot_forced_mean": c["cot"]["forced"]["mean"],
                   "differ_mean_count": c["differ"]["mean_count"], "differ_share": c["differ"]["share"],
                   "differ_min": c["differ"]["min"], "differ_max": c["differ"]["max"],
                   "ties_requested": c["ties"]["requested"]["cells"], "ties_forced": c["ties"]["forced"]["cells"]}
            for name, blk in (("net", c), ("gross", c["gross"])):
                for k in DIRECTIONS:
                    row[f"{name}_{k}"] = blk[k]["estimate"]
                    row[f"{name}_{k}_lo"], row[f"{name}_{k}_hi"] = blk[k]["ci95"]
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def run_task(task):
    x = task_inputs(task)
    models, pred, G = x["models"], x["pred"], x["G"]
    S, M = pred["requested"].shape
    T = {"title": TITLE[task], "M": M, "n_pilot_sets": S, "n_validation_configurations": int(G["requested"].shape[0]),
         "validation_configurations": x["validation_configurations"], "pilot_configurations": x["units_of_set"],
         "n_chosen_top_third": x["n_chosen_top_third"], "saved_pilots": x["saved"],
         "pilot_gain_reproduced_max_abs_diff": x["pilot_gain_reproduced_max_abs_diff"],
         "check_top_third": check_top_third(task, x), "cuts": []}
    decisions = []
    for c in CUTS:
        T["cuts"].append(cut_block(pred, G, c, models, x["units_of_set"]))
        for a in ARM:
            D, tie = decide(pred[a], c)
            for s in range(S):
                for m in range(M):
                    decisions.append({"task": task, "cut": c, "baseline": a, "pilot_set": s,
                                      "configuration": x["units_of_set"][s], "model": models[m],
                                      "pilot_gain": float(pred[a][s, m]), "cot": int(D[s, m]), "tie": bool(tie[s, m])})
    return T, decisions


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import robustness_inputs as ri
    import macros_threshold_decision as mtd
    t0 = time.time()
    R = {"analysis": "threshold decision: chain of thought when the one-draw pilot gain exceeds c points "
                     "(exploratory)",
         "created": time.strftime("%Y-%m-%d %H:%M"), "scores": "strict (the main decision analysis's)",
         "units": "pp of accuracy per panel model", "cuts": list(CUTS), "tie_tolerance_pp": TIE_TOL,
         "rule": "cot when pilot gain > c; a pilot gain within the tolerance of c is a tie and gets no chain of thought",
         "pilots": "manifests/predictions_decision_<task>_req|_fd, rule own_1 (one draw per distinct discovery "
                   "configuration)",
         "evaluation": "validation configurations no discovery draw uses, each once (run_decision.evaluation_rows, "
                       "independent); 95% t interval over them",
         "tasks": {}}
    decisions = []
    for t in TASKS:
        R["tasks"][t], d = run_task(t)
        decisions += d
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["edge_cases"] = edge_cases(R)
    R["checks"] = checks(R)
    R["runtime_s"] = round(time.time() - t0, 1)
    cuts, dec = tables(R, decisions)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    cuts.to_csv(os.path.join(OUT, "tables", "cuts.csv"), index=False)
    dec.to_csv(os.path.join(OUT, "tables", "decisions.csv"), index=False)
    macros = mtd.threshold_decision_macros(R)
    lines = summary(R) + ["", "## Macros", "", "| Macro | Value |", "|---|---|"] + \
        [f"| `\\{k}` | {v} |" for k, v in macros.items()]
    ri.write_outputs(OUT, R, lines, macros=macros)
    print("\n".join(lines))
    return R


if __name__ == "__main__":
    main()
