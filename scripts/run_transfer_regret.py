r"""Transfer regret: held-out family and held-out task. Methods: src/transfer_regret.py.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_transfer_regret.py

A. Held-out family. For each task (GSM8K-29, BIG-Bench Hard-28, MATH-500 numeric), baseline (requested or forced
   direct) and family, choose the top third within the family (round(m / 3), at least one: Qwen 3 of 10, Gemma 3 of 8
   or 2 of 7, Llama, Mistral and Phi 1 each). Transported rules are fitted on the other families' discovery means
   (24 draws, all items) and applied to the family's models: linear in baseline and quadratic in baseline plus the
   index, at the baseline accuracy one pilot draw on the target measures; linear in the index, which needs no call.
   Against them: the family's own gain from the same one pilot draw, and a random third.
B. Another task. For each ordered pair of tasks (source, target) and baseline, on the models both panels share: the
   source's own gains (24 discovery draws, all items); the source's own gain from one pilot draw; the source's
   capability rules (linear; quadratic plus index) applied at the target pilot's baselines; against the target's own
   gain from one pilot draw and a random third.
Pilots: each distinct discovery configuration of the target once (as the one-draw pilots of the decision analysis), all items.
Evaluation: target validation configurations absent from every target discovery draw, each once. Regret: plug-in
and cross-fitted, in pp per chosen model; paired differences (rule minus target own gain) with t intervals over the
evaluation configurations. Held-out family results pool the families, weighted by the number chosen.

Order: every choice is fitted on discovery draws and written to results/transfer_regret/frozen/ before the validation
draws are read. The validation draws had been read by
earlier analyses, so this is not a fresh test.

Added tasks (config/added_contrasts.yaml, section tasks, key fair; HotpotQA): both cases were
fitted for HotpotQA as target, from GSM8K, BIG-Bench Hard and MATH-500 as sources, and saved before its validation
draws were scored (scripts/hotpot_fair.py); its own run evaluated them. Their evaluation is read here into
"added_family" and "added_task" (keys as above, baselines requested, forced and fewshot), kept apart from "family" and
"task", whose ranges and counts stay on the twelve cases of the three tasks; nothing is refitted.

Outputs: results/transfer_regret/transfer_regret.json, summary.md, summary_clean.md, tables/regret.csv.
"""
import itertools
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
from src import decision as dc, rule_budgets as rbud, transfer_regret as tr  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "transfer_regret")
FROZEN = os.path.join(OUT, "frozen")
TASKS = ("gsm8k", "bbh", "math")
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
BASELINES = ("requested", "forced")
FAMILY_RULES = ("transport_linear", "transport_quadratic_index", "transport_index", "target_own_1", "random")
TASK_RULES = ("source_own_all", "source_own_1", "source_linear", "source_quadratic_index", "target_own_1", "random")
NAMES = {"transport_linear": "other families' rule, linear in baseline",
         "transport_quadratic_index": "other families' rule, quadratic in baseline + index",
         "transport_index": "other families' rule, linear in index (no call)",
         "target_own_1": "target own gain, one pilot draw", "random": "random third",
         "source_own_all": "source task's own gain, 24 draws", "source_own_1": "source task's own gain, one pilot draw",
         "source_linear": "source task's rule, linear in baseline, at target baselines",
         "source_quadratic_index": "source task's rule, quadratic + index, at target baselines"}
ARR = ("baseline_linear", "baseline_quadratic_index", "index_linear")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def third(m):
    return max(1, int(round(m / 3)))


def pilot_rows(units):
    return dc.distinct_rows(units)


# ============================================================================ choices (discovery only)


def family_choices(x, base):
    """{family: {rule: (P, m) weights}} and the families' models, from discovery draws only."""
    f, d = x["facts"], x["discovery"]
    fams = f["families"]
    e = np.asarray(f["eci"], float)
    B, P = d[base], d["cot"]
    G = P - B
    b_mean, g_mean = B.mean(0), G.mean(0)
    rows = pilot_rows(f["discovery_config_ids"])
    out, groups, ns = {}, {}, {}
    for fam in sorted(set(fams)):
        ix = np.array([j for j, y in enumerate(fams) if y == fam])
        trn = np.setdiff1d(np.arange(len(fams)), ix)
        n = third(len(ix))
        groups[fam], ns[fam] = ix.tolist(), n
        W = {}
        for rule, key in zip(ARR, ("transport_linear", "transport_quadratic_index", "transport_index")):
            preds = [tr.fit_apply(b_mean[trn], g_mean[trn], e[trn], B[r, ix], e[ix], rule) for r in rows]
            W[key] = tr.third_weights(preds, n)
        W["target_own_1"] = tr.third_weights([G[r, ix] for r in rows], n)
        W["random"] = np.full((1, len(ix)), n / len(ix))
        out[fam] = W
    return out, groups, ns


def task_choices(xs, xt, base):
    """{rule: (P, m)} for the models both panels share, in the target's order, from discovery draws only."""
    pt, ps = xt["facts"]["panel"], xs["facts"]["panel"]
    shared = [k for k in pt if k in ps]
    jt = [pt.index(k) for k in shared]
    js = [ps.index(k) for k in shared]
    n = third(len(shared))
    Bs, Gs = xs["discovery"][base][:, js], (xs["discovery"]["cot"] - xs["discovery"][base])[:, js]
    Bt, Gt = xt["discovery"][base][:, jt], (xt["discovery"]["cot"] - xt["discovery"][base])[:, jt]
    e = np.asarray(xt["facts"]["eci"], float)[jt]
    rows_t = pilot_rows(xt["facts"]["discovery_config_ids"])
    rows_s = pilot_rows(xs["facts"]["discovery_config_ids"])
    W = {"source_own_all": tr.third_weights(Gs.mean(0), n),
         "source_own_1": tr.third_weights([Gs[r] for r in rows_s], n),
         "target_own_1": tr.third_weights([Gt[r] for r in rows_t], n),
         "random": np.full((1, len(shared)), n / len(shared))}
    for rule, key in (("baseline_linear", "source_linear"), ("baseline_quadratic_index", "source_quadratic_index")):
        W[key] = tr.third_weights([tr.fit_apply(Bs.mean(0), Gs.mean(0), e, Bt[r], e, rule) for r in rows_t], n)
    # the sign of the source's linear slope, for the report
    slope = float(np.polyfit(Bs.mean(0), Gs.mean(0), 1)[0])
    return W, shared, jt, n, slope


# ============================================================================ evaluation


def fresh_rows(x):
    units = x["facts"]["validation_config_ids"]
    disc = set(x["facts"]["discovery_config_ids"])
    return [r for r in dc.distinct_rows(units) if units[r] not in disc]


def added_cases(root=ROOT):
    """(added_family, added_task, sources, rows) of the added tasks' fair-comparison evaluations, the fresh evaluation
    set in the form of "family" and "task" (each entry also keeps its all_distinct block)."""
    import added_contrasts as ac
    fam, tsk, src, rows = {}, {}, {}, []
    for t, e in ac.load_tasks().items():
        F = ac.fair_evaluation(t, root=root)
        if F is None:
            continue
        src[t] = {"evaluation": F["path"], "saved": F["saved"], "title": e["title"]}
        for b, x in F["family"].items():
            fam[f"{t}/{b}"] = {**x["fresh"], "all_distinct": x.get("all_distinct")}
            for r, v in x["fresh"]["rules"].items():
                rows.append({"case": "held-out family (added task)", "source": "other families", "target": t,
                             "baseline": b, "rule": r, "n_configurations": x["fresh"]["n_configurations"],
                             "regret_plugin": v["regret_plugin"], "regret_crossfit": v["regret_crossfit"]})
        for (s, b), x in F["cross"].items():
            tsk[f"{s}->{t}/{b}"] = {**x["fresh"], "M": x["M"], "n_chosen": x["n_chosen"], "models": x.get("models"),
                                    "source_linear_slope": x.get("source_linear_slope"),
                                    "all_distinct": x.get("all_distinct")}
            for r, v in x["fresh"]["rules"].items():
                rows.append({"case": "another task (added task)", "source": s, "target": t, "baseline": b, "rule": r,
                             "n_configurations": x["fresh"]["n_configurations"], "regret_plugin": v["regret_plugin"],
                             "regret_crossfit": v["regret_crossfit"]})
    return fam, tsk, src, rows


def f2(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}".replace("-", "−")


def ci(d):
    c = d.get("ci95")
    return f"{f2(d['estimate'])} [{f2(c[0])}, {f2(c[1])}]" if c else f2(d["estimate"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    X = {t: ri.load(t) for t in TASKS}
    # 1. every choice from discovery draws
    arrays, meta = {}, {"family": {}, "task": {}}
    FAM, TSK = {}, {}
    for t in TASKS:
        for b in BASELINES:
            W, groups, ns = family_choices(X[t], b)
            FAM[(t, b)] = (W, groups, ns)
            meta["family"][f"{t}/{b}"] = {"groups": {g: [X[t]["facts"]["panel"][j] for j in ix] for g, ix in groups.items()},
                                          "n_chosen": ns}
            for g, w in W.items():
                for r, a in w.items():
                    arrays[f"family__{t}__{b}__{g}__{r}"] = a
    for s, t in itertools.permutations(TASKS, 2):
        for b in BASELINES:
            W, shared, jt, n, slope = task_choices(X[s], X[t], b)
            TSK[(s, t, b)] = (W, jt, n)
            meta["task"][f"{s}->{t}/{b}"] = {"models": shared, "n_chosen": n, "source_linear_slope": slope}
            for r, a in W.items():
                arrays[f"task__{s}__{t}__{b}__{r}"] = a
    record = {"label": "transfer_regret", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "weights per model",
              "design": meta, "note": "fitted on discovery draws only; written before the validation draws were read"}
    os.makedirs(FROZEN, exist_ok=True)
    jp = os.path.join(FROZEN, "predictions_transfer_regret.json")
    frozen = rbud.freeze(record, arrays, jp, jp[:-5] + ".npz")
    rbud.load_frozen(jp, jp[:-5] + ".npz")
    print(f"choices frozen ({time.time() - t0:.0f} s)", flush=True)
    # 2. validation
    R = {"analysis": "transfer regret", "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": "pp of accuracy gain per chosen model",
         "frozen": {"json": rel(jp), "npz_sha256": frozen["npz_sha256"], "created": frozen["created"]},
         "family": {}, "task": {}, "checks": []}
    rows = []
    for (t, b), (W, groups, ns) in FAM.items():
        v = X[t]["validation"]
        fr = fresh_rows(X[t])
        G = (v["cot"] - v[b])[fr]
        res = {"n_configurations": len(fr), "rules": {}, "paired_minus_target_own_1": {}}
        for r in FAMILY_RULES:
            Wg = {g: W[g][r] for g in groups}
            res["rules"][r] = tr.grouped_regret(Wg, G, groups, ns)
            if r != "target_own_1":
                res["paired_minus_target_own_1"][r] = tr.paired_grouped(
                    Wg, {g: W[g]["target_own_1"] for g in groups}, G, groups)
            rows.append({"case": "held-out family", "source": "other families", "target": t, "baseline": b, "rule": r,
                         "n_configurations": len(fr), "regret_plugin": res["rules"][r]["regret_plugin"],
                         "regret_crossfit": res["rules"][r]["regret_crossfit"]})
        R["family"][f"{t}/{b}"] = res
    for (s, t, b), (W, jt, n) in TSK.items():
        v = X[t]["validation"]
        fr = fresh_rows(X[t])
        G = (v["cot"] - v[b])[fr][:, jt]
        res = {"n_configurations": len(fr), "M": len(jt), "n_chosen": n, "rules": {}, "paired_minus_target_own_1": {},
               "source_linear_slope": meta["task"][f"{s}->{t}/{b}"]["source_linear_slope"]}
        for r in TASK_RULES:
            res["rules"][r] = dc.summarize(W[r], G, n)
            if r != "target_own_1":
                res["paired_minus_target_own_1"][r] = dc.paired_difference(W[r], W["target_own_1"], G, n)
            rows.append({"case": "another task", "source": s, "target": t, "baseline": b, "rule": r,
                         "n_configurations": len(fr), "regret_plugin": res["rules"][r]["regret_plugin"],
                         "regret_crossfit": res["rules"][r]["regret_crossfit"]})
        R["task"][f"{s}->{t}/{b}"] = res
    # check: on a full target panel, the target's one-draw own gain reproduces results/decision
    import json
    dec = json.load(open(os.path.join(ROOT, "results", "decision", "decision.json"), encoding="utf-8"))
    for (s, t, b) in (("gsm8k", "math", "requested"), ("gsm8k", "math", "forced"), ("gsm8k", "bbh", "requested"),
                      ("gsm8k", "bbh", "forced")):
        mine = R["task"][f"{s}->{t}/{b}"]["rules"]["target_own_1"]["regret_plugin"]
        ref = dec["settings"][f"{t}_{'req' if b == 'requested' else 'fd'}"]["result"]["evaluation"]["independent"][
            "rules"]["own_1"]["regret_plugin"]
        if abs(mine - ref) > 1e-9:
            raise SystemExit(f"{t}/{b}: target one-draw own gain {mine} differs from results/decision ({ref})")
        R["checks"].append(f"{TITLE[t]}, {b}: the target's one-draw own gain reproduces results/decision/decision.json "
                           f"(own_1, plug-in regret {ref:.4f} pp)")
    fam, tsk, src, arows = added_cases()          # added tasks: their own saved choices, read
    if src:
        R.update({"added_family": fam, "added_task": tsk, "added_sources": src,
                  "added_note": "kept apart from family and task: their ranges and counts cover the three main tasks"})
        rows += arows
    R["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "regret.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R, meta))
    print("\n".join(summary(R, meta)))


def summary(R, meta):
    L = ["# Transfer regret: choosing whom to prompt from a rule fitted elsewhere", "",
         "Code "
         "`scripts/run_transfer_regret.py`, `src/transfer_regret.py`. The strategy goes to the top third; regret is "
         "the best third's mean gain minus the chosen third's, in pp per chosen model, on target validation "
         "configurations absent from every target discovery draw (each once). Plug-in regret is too large in "
         "expectation, cross-fitted (in parentheses) too small. Pilots: each distinct target discovery configuration "
         "once, all items. Paired differences: the rule's regret minus the target's own one-draw gain's (positive: "
         "the transported rule costs more), t interval over evaluation configurations. Intervals cover draw noise for "
         "this panel only.", "",
         f"Choices frozen before the validation draws were read: `{R['frozen']['json']}` ({R['frozen']['created']}). "
         "The validation draws had been read by earlier analyses, so this is not a fresh test.", "",
         "## A. Held-out family (top third within each family, pooled over the five families)", "",
         "Transported rules are fitted on the other four families' discovery means and applied at the family's "
         "baseline from the same pilot draw (the index rule needs no call). Thirds: " + "; ".join(
             f"{g} {n}" for g, n in meta["family"]["gsm8k/requested"]["n_chosen"].items()) + " (GSM8K and MATH-500); "
         "BIG-Bench Hard's Gemma family has 7 models, third 2.", "",
         "| Task | Baseline | Configs | " + " | ".join(NAMES[r] for r in FAMILY_RULES) + " |",
         "|---|---|---:|" + "---|" * len(FAMILY_RULES)]
    for k, res in R["family"].items():
        t, b = k.split("/")
        L.append(f"| {TITLE[t]} | {b} | {res['n_configurations']} | " + " | ".join(
            f"{f2(res['rules'][r]['regret_plugin'])} ({f2(res['rules'][r]['regret_crossfit'])})" for r in FAMILY_RULES) + " |")
    L += ["", "Paired: rule minus the target's own one-draw gain (pp; positive: the rule costs more).", "",
          "| Task | Baseline | " + " | ".join(NAMES[r] for r in FAMILY_RULES if r != "target_own_1") + " |",
          "|---|---|" + "---|" * (len(FAMILY_RULES) - 1)]
    for k, res in R["family"].items():
        t, b = k.split("/")
        L.append(f"| {TITLE[t]} | {b} | " + " | ".join(ci(res["paired_minus_target_own_1"][r])
                                                      for r in FAMILY_RULES if r != "target_own_1") + " |")
    L += ["", "Per family (plug-in regret, pp per chosen model): " + "; ".join(
        f"{TITLE[k.split('/')[0]]} {k.split('/')[1]}: " + ", ".join(
            f"{g} own {f2(res['rules']['target_own_1']['per_group'][g]['regret_plugin'])} / quad "
            f"{f2(res['rules']['transport_quadratic_index']['per_group'][g]['regret_plugin'])} / random "
            f"{f2(res['rules']['random']['per_group'][g]['regret_plugin'])}"
            for g in res["rules"]["random"]["per_group"]) for k, res in R["family"].items()) + ".", "",
          "## B. Another task (models both panels share)", "",
          "| Source → target | Baseline | Models (third) | Source slope | " + " | ".join(NAMES[r] for r in TASK_RULES) + " |",
          "|---|---|---|---:|" + "---|" * len(TASK_RULES)]
    for k, res in R["task"].items():
        st, b = k.split("/")
        s, t = st.split("->")
        L.append(f"| {TITLE[s]} → {TITLE[t]} | {b} | {res['M']} ({res['n_chosen']}) | {f2(res['source_linear_slope'])} | "
                 + " | ".join(f"{f2(res['rules'][r]['regret_plugin'])} ({f2(res['rules'][r]['regret_crossfit'])})"
                              for r in TASK_RULES) + " |")
    L += ["", "Paired: rule minus the target's own one-draw gain (pp; positive: the rule costs more).", "",
          "| Source → target | Baseline | " + " | ".join(NAMES[r] for r in TASK_RULES if r != "target_own_1") + " |",
          "|---|---|" + "---|" * (len(TASK_RULES) - 1)]
    for k, res in R["task"].items():
        st, b = k.split("/")
        s, t = st.split("->")
        L.append(f"| {TITLE[s]} → {TITLE[t]} | {b} | " + " | ".join(ci(res["paired_minus_target_own_1"][r])
                                                                  for r in TASK_RULES if r != "target_own_1") + " |")
    L += ["", "Source slope: least-squares slope of the source's model mean gains on mean baseline accuracy (24 "
          "discovery draws); negative means the source says weaker models gain more.", "", "## Checks", ""]
    L += [f"- {c}." for c in R["checks"]]
    L += added_summary_lines(R)
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def added_summary_lines(R):
    """The section on the added tasks (plug-in regret on the fresh configurations; cross-fitted in parentheses)."""
    if not R.get("added_sources"):
        return []
    L = ["", "## C. Added tasks, kept apart from A and B", ""]
    for t, x in R["added_sources"].items():
        L.append(f"{x['title']}: read from `{x['evaluation']}`; choices saved before its validation draws were scored "
                 "(" + ", ".join(f"`{k}` {v.get('created')}" for k, v in (x["saved"] or {}).items()) + ").")
    L += ["", "| Case | Baseline | Configs | Rule: plug-in (cross-fitted) |", "|---|---|---:|---|"]
    for k, res in R["added_family"].items():
        t, b = k.split("/")
        L.append(f"| held-out family, {R['added_sources'][t]['title']} | {b} | {res['n_configurations']} | " + "; ".join(
            f"{r} {f2(v['regret_plugin'])} ({f2(v['regret_crossfit'])})" for r, v in res["rules"].items()) + " |")
    for k, res in R["added_task"].items():
        st, b = k.split("/")
        s, t = st.split("->")
        L.append(f"| {TITLE.get(s, s)} → {R['added_sources'][t]['title']}, {res['M']} models ({res['n_chosen']}) | {b} | "
                 f"{res['n_configurations']} | " + "; ".join(
                     f"{r} {f2(v['regret_plugin'])} ({f2(v['regret_crossfit'])})" for r, v in res["rules"].items()) + " |")
    L += ["", "Rule names as in `results/hotpot/summary.md`, fair comparison: the source rules marked at_source are applied "
          "at the source's own baselines, the others at the HotpotQA pilot baselines (as in B)."]
    return L


if __name__ == "__main__":
    main()
