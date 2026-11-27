r"""Decision comparison made strict. Equal-call comparison of top-third choices from a one-draw, 100-item pilot:
the capability rule linear in baseline, the rule quadratic in baseline plus the external index, and own measured gain;
a random third as the floor. Methods: src/decision_strict.py.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_decision_strict.py

Settings: GSM8K-29, BIG-Bench Hard-28 (25 tasks, equal weights) and MATH-500 numeric (29 models, 398 items); chain of
thought against requested direct and against forced direct; strict scoring.

Two pilot designs, each 200 pilots per task (shared by both baselines of the task):
    shared    the pilots of manifests/predictions_rule_budgets.npz (K = 1, 100 items from all items; stream
              sampling.rng_for("rule-budgets", setting, 1, 100)), rebuilt here and checked against that file for the
              requested arm; the forced-arm pilots use the same draws and items. Evaluation on all items.
    disjoint  items split by item id (src/decision_strict.split_items, label "decision-strict"); pilots draw 100 items
              from the pilot half (BIG-Bench Hard: 4 from the pilot half of each task) on stream
              sampling.rng_for("decision-strict", task, 1, 100); evaluation on the other half only.
Two evaluation sets: validation configurations absent from every discovery draw ("fresh", each once;
the main set) and all distinct validation configurations.

Order: discovery item scores are read, every pilot's predictions and choices are written to
manifests/predictions_decision_strict.{json,npz} and only then are the validation
configurations read. The validation
draws had been read by earlier analyses, so this is not a fresh test.

Added tasks (config/added_contrasts.yaml, section tasks, key fair; HotpotQA): the same
designs were fitted on HotpotQA's discovery rows and saved before its validation draws were scored
(scripts/hotpot_fair.py, manifests/predictions_hotpot_decision_strict.*), and evaluated by its own run. Their
evaluation is read here into the task's block, with its three baselines (requested, forced, and the zero-shot forced arm
against five-shot forced); nothing is refitted. They enter no other row of the outputs.

Outputs: results/decision_strict/decision_strict.json, summary.md, summary_clean.md, tables/regret.csv, tables/paired.csv.
"""
import importlib.util
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
from src import decision as dc, decision_strict as ds, rule_budgets as rbud, sampling  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "decision_strict")
FROZEN = os.path.join(ROOT, "manifests")   # the saved pilots
MAN = os.path.join(ROOT, "manifests")
CACHE = os.path.join(ROOT, "data", "processed", "decision_strict")
TASKS = ("gsm8k", "bbh", "math")
RB_SETTING = {"gsm8k": "gsm8k30", "bbh": "bbh", "math": "math"}
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
BASELINES = ("requested", "forced")
N_PILOTS = 200
N_ITEMS = 100
SPLIT_LABEL = "decision-strict"
REF_TABLE = {"gsm8k": (29.75, 7.74, 0.57), "bbh": (6.26, 4.24, 0.84), "math": (31.26, 3.47, 0.63)}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_decision_strict", os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================ inputs


def task_inputs(task):
    """Panel, items, strata, index, sampler units and loaders of a task's three arms."""
    from run_forced_direct import collection_inputs, forced_inputs
    from run_transfer_tasks import eci_for
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    items = [str(i) for i in items]
    strata = None
    if task == "bbh":
        pos = {it: i for i, it in enumerate(items)}
        strata = [np.array([pos[str(i)] for i in its]) for _, its in sorted(inp["items_by_task"].items())]
    return {"task": task, "panel": panel, "items": items, "strata": strata, "eci": np.asarray(eci_for(panel), float),
            "disc": [str(u) for u in s["discovery_config_ids"]], "val": [str(u) for u in s["validation_config_ids"]],
            "col_load": inp["load"], "fd_load": fd_load}


def dense(x, configs):
    """{'requested', 'forced', 'cot'}: (configs, models, items) int8 strict scores, cached under data/processed."""
    configs = sorted(set(map(str, configs)))
    key = f"{x['task']}_{sampling.sha256_json(configs)[:12]}"
    path = os.path.join(CACHE, f"{key}.npz")
    sources = ri.SCORED[x["task"]]
    if os.path.exists(path) and os.path.getmtime(path) >= ri._newest(sources):
        z = np.load(path, allow_pickle=False)
        if list(z["configs"]) == configs:
            return configs, {a: z[a] for a in ("requested", "forced", "cot")}
    rna = load_script("run_noise_anatomy")
    col = x["col_load"](configs)
    col = col.assign(item_id=col["item_id"].astype(str))
    S = rna.dense_items(col, x["panel"], configs, x["items"])
    fd = x["fd_load"](configs)
    fd = fd.assign(item_id=fd["item_id"].astype(str))
    F = rna.dense_items(fd, x["panel"], configs, x["items"], arms=("forced_direct",))
    out = {"requested": S[0], "cot": S[1], "forced": F[0]}
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, configs=np.array(configs), **out)
    return configs, out


# ============================================================================ pilots (discovery only)


def pilots(x, Sd, cfg_index):
    """Both designs' pilots for a task: draws, items, scores, predictions and choices for each baseline."""
    N, M = len(x["items"]), len(x["panel"])
    n = dc.third_size(M)
    mask = ds.split_items(x["items"], x["task"], SPLIT_LABEL, x["strata"])
    designs = {}
    # shared: rule_budgets' stream and draw call, so the pilots are the frozen ones
    rng = sampling.rng_for("rule-budgets", RB_SETTING[x["task"]], 1, N_ITEMS)
    pos, items = rbud.draw_pilots(N_PILOTS, len(x["disc"]), 1, rng, N=N, n=N_ITEMS, strata=x["strata"])
    designs["shared"] = (pos[:, 0], items, x["strata"], np.ones(N, bool))
    rng = sampling.rng_for(SPLIT_LABEL, x["task"], 1, N_ITEMS)
    sub = ds.restrict_strata(x["strata"], mask)
    pos2, items2 = ds.draw_pilots(N_PILOTS, len(x["disc"]), rng, np.flatnonzero(mask), N_ITEMS, sub)
    if not mask[items2].all():
        raise SystemExit("a disjoint pilot drew an item outside the pilot pool")
    designs["disjoint"] = (pos2, items2, sub, ~mask)
    arrays, rec = {}, {}
    for d, (pos, items, strata, emask) in designs.items():
        U = ds.pilot_weights(N, items, strata)
        rows = np.array([cfg_index[x["disc"][p]] for p in pos])
        sc = {a: ds.pilot_scores(Sd[a], rows, U) for a in ("requested", "forced", "cot")}
        arrays[f"{x['task']}__{d}__positions"] = pos.astype(np.int16)
        arrays[f"{x['task']}__{d}__items"] = items.astype(np.int16)
        rec[d] = {"evaluation_items": int(emask.sum()), "pilot_pool_items": int((~emask).sum()) if d == "disjoint"
                  else N, "ties": {}}
        for b in BASELINES:
            pr = ds.pilot_predictions(sc[b], sc["cot"], x["eci"])
            for r, P in pr.items():
                W, T = ds.choice_weights(P, n)
                arrays[f"{x['task']}__{d}__{b}__{r}__pred"] = P
                arrays[f"{x['task']}__{d}__{b}__{r}__weights"] = W
                rec[d]["ties"][f"{b}/{r}"] = T
    rec["item_split_sha256"] = sampling.sha256_json([x["items"][i] for i in np.flatnonzero(mask)])
    return arrays, rec, mask


def check_against_rule_budgets(x, arrays):
    """The shared design's requested-arm predictions equal manifests/predictions_rule_budgets.npz."""
    rec, ref = rbud.load_frozen(os.path.join(MAN, "predictions_rule_budgets.json"),
                                os.path.join(MAN, "predictions_rule_budgets.npz"))
    key = f"{RB_SETTING[x['task']]}__K1__n100__"
    pairs = {"own_gain": "own_gain", "linear": "averaged_ols", "quadratic_index": "baseline_quadratic_index"}
    worst = 0.0
    for mine, theirs in pairs.items():
        a = arrays[f"{x['task']}__shared__requested__{mine}__pred"]
        worst = max(worst, float(np.max(np.abs(a - ref[key + theirs]))))
    if worst > 1e-8:
        raise SystemExit(f"{x['task']}: the shared pilots differ from predictions_rule_budgets.npz by {worst:.3g} pp")
    if not np.array_equal(arrays[f"{x['task']}__shared__positions"], ref[key + "positions"][:, 0]):
        raise SystemExit(f"{x['task']}: the shared pilots' draws differ from predictions_rule_budgets.npz")
    return {"check": f"{TITLE[x['task']]}: the shared design's requested-arm predictions reproduce "
                     f"manifests/predictions_rule_budgets.npz ({key}*)", "max_abs_difference_pp": worst}


# ============================================================================ evaluation (after the freeze)


def evaluation_gains(Sv, configs, units, disc, b, w):
    """{'fresh', 'all_distinct'}: (configurations, models) gains on the weighted items, with the configuration ids."""
    rows = dc.distinct_rows(units)
    ids = {"all_distinct": [units[r] for r in rows],
           "fresh": [units[r] for r in rows if units[r] not in set(disc)]}
    ci = {c: i for i, c in enumerate(configs)}
    out = {}
    for k, cs in ids.items():
        ix = [ci[c] for c in cs]
        G = (Sv["cot"][ix].astype(float) - Sv[b][ix].astype(float)) @ w
        out[k] = (G, cs)
    return out


def ref_check(task, W, G):
    got = tuple(round(float(dc.regret_plugin(W[r], G, dc.third_size(G.shape[1])).mean()), 2)
                for r in ("linear", "quadratic_index", "own_gain"))
    return {"reference": list(REF_TABLE[task]), "here": list(got), "reproduced": got == REF_TABLE[task]}


# ============================================================================ report


def added_task_blocks(root=ROOT) -> dict:
    """{task: block} of the added tasks with a fair-comparison evaluation, in the form of the main tasks' blocks."""
    import added_contrasts as ac
    out = {}
    for t, e in ac.load_tasks().items():
        F = ac.fair_evaluation(t, root=root)
        if F is None or not F["strict"]:
            continue
        rec = json.load(open(os.path.join(root, "manifests", f"predictions_{t}_decision_strict.json"), encoding="utf-8"))
        rd = rec["design"]
        design = {d: {"evaluation_items": x["evaluation_items"], "pilot_pool_items": x["pilot_pool_items"],
                      "ties": {F["arms"][k.split("/")[0]] + "/" + k.split("/")[1]: v for k, v in x["ties"].items()}}
                  for d, x in rd["designs"].items()}
        design["item_split_sha256"] = rd.get("item_split_sha256")
        out[t] = {"title": e["title"], "M": len(rec["models"]), "n_chosen": rec["n_chosen"],
                  "n_items": rd["n_target_items"], "design": design, "results": F["strict"], "added_task": True,
                  "source": {"evaluation": F["path"], "saved": f"manifests/predictions_{t}_decision_strict.json",
                             "saved_sha256": rec["sha256"], "saved_created": rec["created"],
                             "note": "choices saved before the task's validation draws were scored; evaluated by the "
                                     "task's own run and read here"}}
    return out


def title(t, R=None):
    return TITLE.get(t) or (R or {}).get("tasks", {}).get(t, {}).get("title", t)


def f2(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "−")


def ci(d):
    c = d.get("ci95")
    return f"{f2(d['estimate'])} [{f2(c[0])}, {f2(c[1])}]" if c else f2(d["estimate"])


def summary(R):
    L = ["# Decision comparison made strict", "",
         "Code "
         "`scripts/run_decision_strict.py`, `src/decision_strict.py`; numbers in `decision_strict.json` and `tables/`.",
         "",
         "A practitioner runs one pilot (one discovery draw, 100 items, every model, both arms) and gives the strategy "
         "to the third of the panel predicted to gain most (10 of 29, 9 of 28). Three rules use the same pilot calls: "
         "the capability rule linear in baseline accuracy, the rule quadratic in baseline accuracy plus the external "
         "index, and each model's own measured gain. A random third is the floor. Regret is the mean gain of the best "
         "third minus the mean gain of the chosen third on the evaluation configurations, in pp per chosen model, "
         "averaged over 200 pilots. Plug-in regret (best third on the same evaluation draws) is too large in "
         "expectation; cross-fitted regret is too small and can be negative. The plug-in interval is a "
         "leave-one-configuration-out jackknife t interval; paired differences are linear in the draws and get a t "
         "interval over evaluation configurations. Intervals cover draw noise for this panel, not the choice of "
         "models or pilots.", "",
         "Designs. **Shared**: the 200 pilots of `manifests/predictions_rule_budgets.npz` (items from all items), "
         "valued on all items. **Disjoint**: items split in halves by a hash of the item id (within each BIG-Bench Hard "
         "task); pilots draw from one half and choices are valued on the other half only, with BIG-Bench Hard's task "
         "weights kept. Evaluation sets: **fresh**, validation configurations absent from every discovery draw, each "
         "once; **all**, every distinct validation configuration.", "",
         f"Frozen before the validation configurations were read: `{R['frozen']['json']}` (sha256 "
         f"`{R['frozen']['npz_sha256'][:16]}…`, {R['frozen']['created']}). The validation draws had been read by "
         "earlier analyses, so this is not a fresh test.", "",
         "## Reproduction of the reference table", ""]
    for t, a in R["ref_check"].items():
        L.append(f"- {TITLE[t]}: reference {', '.join(f2(v) for v in a['reference'])}; here {', '.join(f2(v) for v in a['here'])}"
                 f" (linear, quadratic + index, own gain) — {'reproduced' if a['reproduced'] else 'NOT reproduced'}.")
    L += ["", "## Strict: disjoint items, fresh configurations", "",
          "| Task | Baseline | Fresh configs | Linear | Quadratic + index | Own gain | Random | Own − quadratic + index | "
          "Own − linear |", "|---|---|---:|---|---|---|---|---|---|"]
    for t in R["tasks"]:
        for b in R["tasks"][t]["results"]["disjoint"]:
            e = R["tasks"][t]["results"]["disjoint"][b]["fresh"]
            r = e["rules"]
            L.append(f"| {title(t, R)} | {b} | {e['n_configurations']} | "
                     + " | ".join(ci(r[k]["plugin_interval"]) for k in ("linear", "quadratic_index", "own_gain", "random"))
                     + f" | {ci(e['paired']['own_gain_minus_quadratic_index'])} | {ci(e['paired']['own_gain_minus_linear'])} |")
    L += ["", "Cells: plug-in regret [95% jackknife interval]; the last two columns are paired differences of regret "
          "(negative: own gain costs less).", "",
          "## Every design and evaluation set (plug-in regret; cross-fitted in parentheses)", "",
          "| Task | Baseline | Design | Evaluation | Configs | Items | Linear | Quadratic + index | Own gain | Random | "
          "Own − quadratic + index |", "|---|---|---|---|---:|---:|---|---|---|---|---|"]
    for t in R["tasks"]:
        for b in R["tasks"][t]["results"]["disjoint"]:
            for d in ("shared", "disjoint"):
                for ev in ("fresh", "all_distinct"):
                    e = R["tasks"][t]["results"][d][b][ev]
                    r = e["rules"]
                    L.append(f"| {title(t, R)} | {b} | {d} | {ev} | {e['n_configurations']} | "
                             f"{R['tasks'][t]['design'][d]['evaluation_items']} | "
                             + " | ".join(f"{f2(r[k]['regret_plugin'])} ({f2(r[k]['regret_crossfit'])})"
                                          for k in ("linear", "quadratic_index", "own_gain", "random"))
                             + f" | {ci(e['paired']['own_gain_minus_quadratic_index'])} |")
    L += ["", "## Ties at the cut", "",
          "Pilots whose choice had models tied at the cut (they share the remaining places): " + "; ".join(
              f"{title(t, R)} {d}: " + ", ".join(f"{k} {v}" for k, v in R["tasks"][t]["design"][d]["ties"].items() if v)
              for t in R["tasks"] for d in ("shared", "disjoint")
              if any(R["tasks"][t]["design"][d]["ties"].values())) + ".", "",
          "## Checks", ""] + [f"- {c['check']} (largest difference {c['max_abs_difference_pp']:.2g} pp)." for c in R["checks"]]
    for t, x in R["tasks"].items():
        if x.get("added_task"):
            L += ["", f"{x['title']} (added task): read from `{x['source']['evaluation']}`; the choices were "
                      f"saved to `{x['source']['saved']}` ({x['source']['saved_created']}) before its validation draws "
                      "were scored. Its shared design draws from all its items on the stream rule-budgets/" + t
                  + "; its third baseline, fewshot, is zero-shot forced against five-shot forced."]
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def tables(R):
    reg, par = [], []
    for t, x in R["tasks"].items():
        for d, bx in x["results"].items():
            for b, ex in bx.items():
                for ev, e in ex.items():
                    for r, v in e["rules"].items():
                        c = v["plugin_interval"].get("ci95") or [None, None]
                        reg.append({"task": t, "baseline": b, "design": d, "evaluation": ev,
                                    "n_configurations": e["n_configurations"], "rule": r,
                                    "regret_plugin": v["regret_plugin"], "ci_lo": c[0], "ci_hi": c[1],
                                    "regret_crossfit": v["regret_crossfit"], "p10": v["regret_plugin_p10"],
                                    "p90": v["regret_plugin_p90"], "share_best_third": v["share_best_third"]})
                    for k, v in e["paired"].items():
                        c = v.get("ci95") or [None, None]
                        par.append({"task": t, "baseline": b, "design": d, "evaluation": ev, "comparison": k,
                                    "estimate": v["estimate"], "ci_lo": c[0], "ci_hi": c[1], "K": v["K"]})
    return pd.DataFrame(reg), pd.DataFrame(par)


# ============================================================================ main


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    X, arrays, design, checks, masks = {}, {}, {}, [], {}
    # 1. discovery only
    for t in TASKS:
        x = task_inputs(t)
        configs, Sd = dense(x, x["disc"])
        a, rec, mask = pilots(x, Sd, {c: i for i, c in enumerate(configs)})
        arrays.update(a)
        design[t] = rec
        masks[t] = mask
        checks.append(check_against_rule_budgets(x, a))
        X[t] = x
        print(f"{t}: pilots fitted ({time.time() - t0:.0f} s)", flush=True)
    record = {"label": "decision_strict", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "units": "pp; predictions uncentred; weights per model (ties share places)",
              "design": {"n_pilots": N_PILOTS, "n_items": N_ITEMS, "split_label": SPLIT_LABEL,
                         "streams": {"shared": "sampling.rng_for('rule-budgets', setting, 1, 100)",
                                     "disjoint": "sampling.rng_for('decision-strict', task, 1, 100)"},
                         "models": {t: X[t]["panel"] for t in TASKS},
                         "discovery_units": {t: X[t]["disc"] for t in TASKS}, "tasks": design},
              "note": "fitted on discovery draws only and written before the validation configurations were read"}
    os.makedirs(FROZEN, exist_ok=True)
    jp = os.path.join(FROZEN, "predictions_decision_strict.json")
    frozen = rbud.freeze(record, arrays, jp, jp[:-5] + ".npz")
    frozen, arrays = rbud.load_frozen(jp, jp[:-5] + ".npz")
    print(f"choices frozen ({time.time() - t0:.0f} s)", flush=True)
    # 2. validation
    R = {"analysis": "decision comparison made strict", "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": "pp of accuracy gain per chosen model",
         "frozen": {"json": rel(jp), "npz_sha256": frozen["npz_sha256"], "created": frozen["created"]},
         "tasks": {}, "checks": checks, "ref_check": {}}
    for t in TASKS:
        x = X[t]
        configs, Sv = dense(x, x["val"])
        N, M = len(x["items"]), len(x["panel"])
        n = dc.third_size(M)
        res = {}
        for d in ("shared", "disjoint"):
            emask = np.ones(N, bool) if d == "shared" else ~masks[t]
            w = ds.eval_weights(N, emask, x["strata"])
            res[d] = {}
            for b in BASELINES:
                Wd = {r: arrays[f"{t}__{d}__{b}__{r}__weights"] for r in ds.RULES}
                Wd["random"] = np.full((1, M), n / M)
                gains = evaluation_gains(Sv, configs, x["val"], x["disc"], b, w)
                res[d][b] = {ev: {**ds.evaluate(Wd, G, n), "configurations": cs} for ev, (G, cs) in gains.items()}
                if d == "shared" and b == "requested":
                    R["ref_check"][t] = ref_check(t, Wd, gains["fresh"][0])
        R["tasks"][t] = {"title": TITLE[t], "M": M, "n_chosen": n, "n_items": N, "design": design[t], "results": res}
        print(f"{t}: evaluated ({time.time() - t0:.0f} s)", flush=True)
    R["tasks"].update(added_task_blocks())        # added tasks: their own saved choices, read
    R["runtime_s"] = round(time.time() - t0, 1)
    reg, par = tables(R)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    reg.to_csv(os.path.join(OUT, "tables", "regret.csv"), index=False)
    par.to_csv(os.path.join(OUT, "tables", "paired.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print("\n".join(summary(R)))


if __name__ == "__main__":
    main()
