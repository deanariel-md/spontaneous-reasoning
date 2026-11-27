r"""Decision analysis: give the strategy to the third of the panel predicted to gain most,
and report the regret in pp against the best third, per rule, pilot size and setting, and across protocols.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_decision.py 2>&1"
    cmd /c ".venv\Scripts\python.exe scripts\run_decision.py --settings fewshot --dry-run 2>&1"   # test the slot

With no --settings every setting whose input exists is run, into results/decision/. The five-shot settings (five-shot
forced against zero-shot forced) join when the same command runs again once
data/processed/fewshot/<task>_scored.parquet exists; the choices already saved for the other settings are kept as they
are. A subset of the settings writes to results/decision/subset/; the dry run writes its choices to
results/decision/dry_run/frozen/, never to manifests/.

Order: for each setting the discovery draws are built, every choice is made and written to
manifests/predictions_decision_<setting>.json and .npz (an identical pair is kept, a different one refused); only then
are the validation draws built. Outputs: results/decision/decision.json, summary.md, summary_clean.md,
tables/regret.csv, tables/paired.csv. Array caches: data/processed/decision/ (git-ignored).
"""
import argparse
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
from src import decision as dc  # noqa: E402
from src import rule_budgets as rbud  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "decision")
MANIFESTS = os.path.join(ROOT, "manifests")
CACHE = os.path.join(ROOT, "data", "processed", "decision")
CONTENT = os.path.join(ROOT, "data", "processed", "content")
FEWSHOT = os.path.join(ROOT, "data", "processed", "fewshot")
EXEMPLAR = os.path.join(ROOT, "data", "processed", "exemplar")
TASK_NAMES = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500"}

# setting: (task for the pilot seed, source, baseline arm, intervention arm, title)
SETTINGS = {}
for _t in ("gsm8k", "bbh", "math"):
    SETTINGS[f"{_t}_req"] = (_t, "strict", "requested", "cot", f"{TASK_NAMES[_t]}: chain of thought − requested direct")
for _t in ("gsm8k", "bbh", "math"):
    SETTINGS[f"{_t}_fd"] = (_t, "strict", "forced", "cot", f"{TASK_NAMES[_t]}: chain of thought − forced direct")
for _t in ("gsm8k", "bbh", "math"):
    SETTINGS[f"{_t}_con"] = (_t, "content", "requested", "cot",
                             f"{TASK_NAMES[_t]}: chain of thought − requested direct, content scores")
SETTINGS["exemplar"] = ("gsm8k", "exemplar", "zero_shot_cot", "eight_shot_cot",
                        "GSM8K-29: eight-shot − zero-shot chain of thought")
SETTINGS["dove_mmlu"] = ("dove_mmlu", "dove", "0_shot", "5_shot", "DOVE MMLU block, 5 models: five-shot − zero-shot")
for _t in ("gsm8k", "bbh", "math"):
    SETTINGS[f"fewshot_{_t}"] = (_t, "fewshot", "forced", "five_shot",
                                 f"{TASK_NAMES[_t]}: five-shot − zero-shot forced direct")
GROUPS = {"main": [k for k in SETTINGS if not k.startswith("fewshot_")],          # before the added tasks' settings
          "fewshot": [k for k in SETTINGS if k.startswith("fewshot_")]}
# (choice made under, evaluated under)
CROSS = [(f"{t}_req", f"{t}_{x}") for t in ("gsm8k", "bbh", "math") for x in ("fd", "con")]
# Settings of the added tasks (config/added_contrasts.yaml, section tasks, key decision; HotpotQA):
# baseline and intervention roles of the task's scored table, and the label of the choices its own run
# saved before its validation draws were scored (manifests/predictions_decision_<label>.json), which this script refits
# from the discovery draws and keeps only when identical. They run after the other settings (group "added").
CHOICES = {}


def _added_settings():
    import added_contrasts as ac
    out = []
    for t, e in ac.load_tasks().items():
        for name, d in (e.get("decision") or {}).items():
            SETTINGS[name] = (t, "table", d["baseline"], d["intervention"], d["title"])
            CHOICES[name] = d["choices"]
            out.append(name)
        if f"{t}_req" in out and f"{t}_fd" in out:
            CROSS.append((f"{t}_req", f"{t}_fd"))
    return out


GROUPS["added"] = _added_settings()
RULE_ORDER = ["capability_1", "own_1", "own_2", "shrunk_2", "own_4", "shrunk_4", "own_all", "capability_all", "random"]
DRY = False            # set by run(): the five-shot settings with run_fewshot's stand-in arm
RULE_NAMES = {"capability_1":"capability slope, 1 draw", "own_1": "own gain, 1 draw", "own_2": "own gain, 2 draws",
              "own_4": "own gain, 4 draws", "shrunk_2": "shrunken own gain, 2 draws",
              "shrunk_4": "shrunken own gain, 4 draws", "own_all": "own gain, all discovery draws",
              "capability_all": "capability slope (cross-draw), all discovery draws", "random": "random third"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ============================================================================ inputs


def _cached(name, sources, build):
    """Arrays cached in data/processed/decision/<name>.npz, rebuilt when a source is newer."""
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path) and os.path.getmtime(path) >= ri._newest(sources):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    print(f"building {name}", flush=True)
    arrays = build()
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def task_split(task, scoring, split):
    """{'units', 'requested', 'forced', 'cot'} (K, M) pp for a split; strict or content scores."""
    sources = list(ri.SCORED[task]) + ([rel(os.path.join(CONTENT, f"{task}_content.parquet"))] if scoring == "content" else [])

    def build():
        from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
        inp, s, items, items_info = collection_inputs(task)
        panel = list(inp["panel"])
        fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
        col_load, value = inp["load"], "strict_correct"
        if scoring == "content":
            from run_outcomes import with_content
            col_load, fd_load, value = with_content(col_load, task), with_content(fd_load, task), "content_correct"
        units = [str(u) for u in s[f"{split}_config_ids"]]
        Breq, Bfd, Pc = three_arm_arrays(col_load, fd_load, units, panel, items, value=value)
        return {"units": np.array(units), "panel": np.array(panel), "requested": Breq, "forced": Bfd, "cot": Pc}
    return _cached(f"{task}_{scoring}_{split}", sources, build)


def exemplar_split(split):
    sources = list(ri.SCORED["gsm8k"]) + [rel(os.path.join(EXEMPLAR, "gsm8k_scored.parquet")),
                                          rel(os.path.join(EXEMPLAR, "zero_shot_4k.parquet"))]

    def build():
        from run_forced_direct import collection_inputs
        from run_exemplar import arrays, combined_loader, cut_table, exemplar_inputs
        inp, s, items, items_info = collection_inputs("gsm8k")
        ex_load, _, panel = exemplar_inputs(list(inp["panel"]), s, items_info, inp["load"], False, False)
        load = combined_loader(inp["load"], ex_load, cut_table())
        units = [str(u) for u in s[f"{split}_config_ids"]]
        B, P = arrays(load, units, panel, items)
        return {"units": np.array(units), "panel": np.array(panel), "zero_shot_cot": B, "eight_shot_cot": P}
    return _cached(f"exemplar_{split}", sources, build)


def dove_split(split):
    """DOVE's MMLU block: the frozen sampler's draws (bank indices) and, for split 'bank', every common configuration."""
    import json
    from run_dataset1 import PANEL, load_block
    from src import controls
    _, _, common, _, nc, n_items = load_block()
    s = json.load(open(os.path.join(MANIFESTS, "sampler_dove_mmlu.json"), encoding="utf-8"))
    idx = (np.arange(len(common)) if split == "bank" else np.array(s[f"{split}_bank_index"]))
    return {"units": np.array([str(i) for i in idx]), "panel": np.array(PANEL),
            "0_shot": controls.pooled(nc["0_shot"][:, idx, :], n_items),
            "5_shot": controls.pooled(nc["5_shot"][:, idx, :], n_items)}


def fewshot_available(task):
    return os.path.exists(os.path.join(FEWSHOT, f"{task}_scored.parquet"))


def fewshot_split(task, split):
    """Run 1's five-shot forced arm against the zero-shot forced arm. In a dry run (DRY) run_fewshot's stand-in arm
    replaces the five-shot arm, to test the path before the scores exist."""
    sources = list(ri.SCORED[task]) + ([] if DRY else [rel(os.path.join(FEWSHOT, f"{task}_scored.parquet"))])

    def build():
        import run_fewshot
        x = run_fewshot.inputs(task, DRY)
        units = [str(u) for u in (x["disc"] if split == "discovery" else x["val"])]
        B, P = run_fewshot.pair_arrays(x["fd_load"], x["fs_load"], units, x["panel"], x["items"])
        return {"units": np.array(units), "panel": np.array(list(x["panel"])), "forced": B, "five_shot": P}
    return _cached(f"fewshot_{task}_{split}" + ("_dry" if DRY else ""), sources, build)


def table_split(name, split):
    """An added task's setting: {'units', 'panel', <baseline role>, <intervention role>} (K, M) pp, strict scores."""
    task, _, b, p, _ = SETTINGS[name]

    def build():
        import added_contrasts as ac
        from src import dataset2 as d2
        from run_forced_direct import collection_inputs
        inp, s, items, _ = collection_inputs(task)
        panel = list(inp["panel"])
        units = [str(u) for u in s[f"{split}_config_ids"]]
        rows = pd.concat([ac.role_loader(task, b, "direct_answer")(units),
                          ac.role_loader(task, p, "zero_shot_cot")(units)], ignore_index=True)
        cells = d2.cell_table(rows[rows["model_key"].isin(panel)], panel, units, items)
        B, P = d2.unit_arrays(cells, units, panel, "strict_correct")
        return {"units": np.array(units), "panel": np.array(panel), b: B, p: P}
    if task not in ri.SCORED:
        ri.added_tasks()
    return _cached(f"{name}_{split}", list(ri.SCORED[task]), build)


def choice_path(name, pred_dir=MANIFESTS):
    """The choices' file: predictions_decision_<setting>.json; an added task's setting, the label its run saved."""
    return os.path.join(pred_dir, f"predictions_decision_{CHOICES.get(name, name)}.json")


def setting_arrays(name, split):
    """(units, panel, B, P) of a setting and split."""
    task, source, b, p, _ = SETTINGS[name]
    if source == "table":
        x = table_split(name, split)
    elif source in ("strict", "content"):
        x = task_split(task, source, split)
    elif source == "exemplar":
        x = exemplar_split(split)
    elif source == "dove":
        x = dove_split(split)
    else:
        x = fewshot_split(task, split)
    return [str(u) for u in x["units"]], [str(m) for m in x["panel"]], np.asarray(x[b], float), np.asarray(x[p], float)


def available(name):
    task, source = SETTINGS[name][:2]
    if source == "fewshot":
        return DRY or fewshot_available(task)
    if source == "content":
        return os.path.exists(os.path.join(CONTENT, f"{task}_content.parquet"))
    if source == "table":
        ri.added_tasks()
        return os.path.exists(os.path.join(ROOT, ri.SCORED[task][0]))
    return True


# ============================================================================ choices (discovery draws only)


def freeze_choices(name, pred_dir=MANIFESTS, overwrite=False):
    """Choices of every rule from the discovery draws, written to <pred_dir>/predictions_decision_<name>.json/.npz."""
    record, arrays = choice_record(name, *setting_arrays(name, "discovery"))
    jp = choice_path(name, pred_dir)
    return rbud.freeze(record, arrays, jp, jp[:-5] + ".npz", overwrite=overwrite)


def choice_record(name, units, panel, B, P, task=None, title=None):
    """The record and arrays of every rule's choices from a setting's discovery draws (units, panel, B, P). task and
    title default to the setting's entry in SETTINGS; HotpotQA (scripts/run_hotpot.py) passes its own."""
    task = task or SETTINGS[name][0]
    title = title or SETTINGS[name][4]
    M = len(panel)
    n = dc.third_size(M)
    sets = dc.pilot_sets(units, task)
    preds = dc.rule_predictions(B, P, sets)
    preds.update(dc.reference_predictions(B, P, dc.distinct_rows(units)))
    ch = dc.choices(preds, n)
    arrays, slopes, tied = {}, {}, {}
    for r, lst in preds.items():
        arrays[f"{r}__pred"] = np.array([x[0] for x in lst], float)
        arrays[f"{r}__weights"] = ch[r]["weights"]
        arrays[f"{r}__rows"] = np.array([x[1]["rows"] for x in lst], np.int64)
        tied[r] = {"choices_with_tie_at_cut": int(sum(t > 0 for t in ch[r]["n_tied"])), "n_choices": len(lst)}
        if r.startswith("capability"):
            sl = np.array([x[1]["slope"] for x in lst], float)
            slopes[r] = {"negative": int((sl < 0).sum()), "positive": int((sl > 0).sum()),
                         "zero_or_undefined": int((~np.isfinite(sl) | (sl == 0)).sum())}
        if r.startswith("shrunk"):
            slopes[r] = {"fallback_constant": int(sum(x[1].get("fallback") == "constant" for x in lst)),
                         "psd_projection": int(sum(bool(x[1].get("psd_projection")) for x in lst))}
    record = {"label": f"decision_{name}", "setting": title, "models": panel, "n_chosen": n,
              "discovery_units": units, "pilot_seed_labels": {str(k): f"decision/{task}/K{k}" for k in sets if k > 1},
              "units": "pp", "design": {"rules": sorted(preds), "pilot_sizes": sorted(sets), "n_sets": dc.N_PILOT_SETS,
                                        "rows": "row indices into discovery_units; distinct configurations only"},
              "slopes": slopes, "ties": tied,
              "note": "choices fitted on discovery draws only, written before the validation draws are built"}
    return record, arrays


def load_choices(name, pred_dir=MANIFESTS):
    jp = choice_path(name, pred_dir)
    rec, arrays = rbud.load_frozen(jp, jp[:-5] + ".npz")
    W = {k[:-9]: v for k, v in arrays.items() if k.endswith("__weights")}
    M = len(rec["models"])
    W["random"] = np.full((1, M), rec["n_chosen"] / M)
    return rec, W


# ============================================================================ evaluation


def evaluation_rows(disc_units, val_units):
    """Row indices of the evaluation draws: distinct validation configurations, all and those no discovery draw uses."""
    rows = dc.distinct_rows(val_units)
    d = set(map(str, disc_units))
    return {"independent": [r for r in rows if str(val_units[r]) not in d], "all_distinct": rows}


def evaluate(name, rec, W, val, g_bank=None):
    units, panel, Bv, Pv = val
    if panel != rec["models"]:
        raise SystemExit(f"{name}: the validation panel differs from the frozen choices")
    n = rec["n_chosen"]
    Gv = Pv - Bv
    ev = evaluation_rows(rec["discovery_units"], units)
    out = {"n_chosen": n, "M": len(panel), "evaluation": {}}
    for which, rows in ev.items():
        G = Gv[rows]
        blk = {"n_configurations": len(rows), "rules": {}, "paired": []}
        if len(rows) < 2:
            blk["note"] = "fewer than two evaluation configurations"
            out["evaluation"][which] = blk
            continue
        gm = G.mean(0)
        wb = dc.best_weights(G, n)
        blk["best_third_value"] = float(wb @ gm / n)
        blk["panel_mean_gain"] = float(gm.mean())
        for r in RULE_ORDER:
            if r in W:
                blk["rules"][r] = dc.summarize(W[r], G, n, g_true=g_bank)
        for r in RULE_ORDER:
            if r not in W:
                continue
            if r != "capability_1":
                blk["paired"].append({"a": "capability_1", "b": r, **dc.paired_difference(W["capability_1"], W[r], G, n)})
            if r != "random":
                blk["paired"].append({"a": "random", "b": r, **dc.paired_difference(W["random"], W[r], G, n)})
        for k in (2, 4):
            if f"own_{k}" in W and f"shrunk_{k}" in W:
                blk["paired"].append({"a": f"own_{k}", "b": f"shrunk_{k}",
                                      **dc.paired_difference(W[f"own_{k}"], W[f"shrunk_{k}"], G, n)})
        out["evaluation"][which] = blk
    return out


def cross_protocol(src, dst, W_src, W_dst, rec_src, rec_dst, val):
    """Choices made under src, evaluated under dst, against dst's own choices on the same pilot sets."""
    units, panel, Bv, Pv = val
    if rec_src["models"] != rec_dst["models"] or rec_src["discovery_units"] != rec_dst["discovery_units"]:
        raise SystemExit(f"{src} and {dst} differ in panel or discovery draws")
    n = rec_dst["n_chosen"]
    Gv = Pv - Bv
    out = {"choice": src, "evaluation_setting": dst, "evaluation": {}}
    for which, rows in evaluation_rows(rec_dst["discovery_units"], units).items():
        G = Gv[rows]
        blk = {"n_configurations": len(rows), "rules": {}}
        if len(rows) < 2:
            out["evaluation"][which] = blk
            continue
        for r in RULE_ORDER:
            if r in W_src and r in W_dst and r != "random":
                same = float(np.mean(np.all(np.abs(W_src[r] - W_dst[r]) <= 1e-9, axis=1)))
                blk["rules"][r] = {"mismatched": dc.summarize(W_src[r], G, n), "matched": dc.summarize(W_dst[r], G, n),
                                   "extra_regret": dc.paired_difference(W_src[r], W_dst[r], G, n),
                                   "share_same_choice": same}
        out["evaluation"][which] = blk
    return out


# ============================================================================ report


def f2(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}".replace("-", "−")


def ci(d):
    if d.get("ci95") is None:
        return f2(d["estimate"])
    return f"{f2(d['estimate'])} [{f2(d['ci95'][0])}, {f2(d['ci95'][1])}]"


def regret_table(R, which="independent"):
    rows = []
    for name, x in R["settings"].items():
        e = x["result"]["evaluation"].get(which, {})
        for r, v in e.get("rules", {}).items():
            rows.append({"setting": name, "evaluation": which, "rule": r, "n_configurations": e["n_configurations"],
                         "n_chosen": x["result"]["n_chosen"], "M": x["result"]["M"], **v})
    return pd.DataFrame(rows)


def paired_table(R):
    """Paired rows of the settings, then of the cross-protocol pairs; an added task's rows (settings, then pairs) after
    all others, so the rows written before they were added keep their places."""
    first = [n for n in R["settings"] if n not in CHOICES]
    return pd.concat([_paired_rows(R, first, lambda c: c["choice"] not in CHOICES),
                      _paired_rows(R, [n for n in R["settings"] if n in CHOICES], lambda c: c["choice"] in CHOICES)],
                     ignore_index=True)


def _paired_rows(R, names, cross_filter):
    rows = []
    for name in names:
        x = R["settings"][name]
        for which, e in x["result"]["evaluation"].items():
            for p in e.get("paired", []):
                rows.append({"setting": name, "evaluation": which, "a": p["a"], "b": p["b"], "extra_regret_of_a": p["estimate"],
                             "ci_lo": (p["ci95"] or [None, None])[0], "ci_hi": (p["ci95"] or [None, None])[1], "K": p["K"]})
    for c in R["cross_protocol"]:
        if not cross_filter(c):
            continue
        for which, e in c["evaluation"].items():
            for r, v in e.get("rules", {}).items():
                p = v["extra_regret"]
                rows.append({"setting": f"{c['choice']}->{c['evaluation_setting']}", "evaluation": which,
                             "a": f"{r} (chosen under {c['choice']})", "b": f"{r} (chosen under {c['evaluation_setting']})",
                             "extra_regret_of_a": p["estimate"], "ci_lo": (p["ci95"] or [None, None])[0],
                             "ci_hi": (p["ci95"] or [None, None])[1], "K": p["K"]})
    return pd.DataFrame(rows)


def summary(R):
    L = ["# Decision analysis: the cost of choosing who gets the strategy", ""]
    if R.get("dry_run"):
        L += ["**Dry run: run_fewshot's stand-in arm replaces the five-shot arm. These numbers test the code path and "
              "are not results.**", ""]
    L += [
         "Code `scripts/run_decision.py` and `src/decision.py`; numbers in "
         "`decision.json` and `tables/`. A practitioner gives the strategy to the third of the panel predicted to gain "
         "most. Regret is the mean gain of the best third minus the mean gain of the chosen third, in pp of accuracy "
         "gain per chosen model. The choices were saved to `manifests/predictions_decision_<setting>.json` and `.npz` "
         "before the validation draws were built in this run; earlier analyses had read the validation draws.", "",
         "Evaluation: validation configurations that no discovery draw uses, each once. Regret is bracketed: the "
         "plug-in value (best third on the same draws, too large in expectation) and the cross-fitted value (best third "
         "on one half valued on the other, too small in expectation, can be negative). Each cell gives the mean over "
         "pilot sets; pilot sets of one draw are the distinct discovery configurations, those of 2 and 4 are 200 seeded "
         "sets. \"Best\" is the share of pilot sets that choose the plug-in best third.", ""]
    for name, x in R["settings"].items():
        res = x["result"]
        e = res["evaluation"]["independent"]
        L += [f"## {x['title']}", "",
              f"{res['n_chosen']} of {res['M']} models chosen; {e['n_configurations']} evaluation configurations "
              f"({res['evaluation']['all_distinct']['n_configurations']} distinct validation configurations in all). "
              f"Best third's mean gain {f2(e.get('best_third_value'))} pp; panel mean gain {f2(e.get('panel_mean_gain'))} pp."]
        sl = x["choices"]["slopes"].get("capability_1")
        if sl:
            L.append(f"One-draw capability slopes: {sl['negative']} negative (weakest third chosen), {sl['positive']} "
                     f"positive, {sl['zero_or_undefined']} zero or undefined.")
        sa = x["choices"]["slopes"].get("capability_all")
        if sa:
            L.append(f"Cross-draw capability slope on all discovery draws: "
                     f"{'negative' if sa['negative'] else 'positive' if sa['positive'] else 'zero or undefined'}.")
        L += ["", "| Rule | Regret, plug-in | Regret, cross-fitted | Plug-in, 10th–90th percentile over pilots | Best |"
              + (" Regret, whole bank |" if any("regret_exact" in v for v in e["rules"].values()) else ""),
              "|---|---:|---:|---|---:|" + ("---:|" if any("regret_exact" in v for v in e["rules"].values()) else "")]
        for r, v in e["rules"].items():
            row = (f"| {RULE_NAMES[r]} | {f2(v['regret_plugin'])} | {f2(v['regret_crossfit'])} | "
                   f"{f2(v['regret_plugin_p10'])} to {f2(v['regret_plugin_p90'])} | {100 * v['share_best_third']:.0f}% |")
            if "regret_exact" in v:
                row += f" {f2(v['regret_exact'])} |"
            L.append(row)
        pr = [p for p in e["paired"] if p["a"] == "capability_1" and p["b"] in ("own_1", "own_4", "shrunk_4")]
        if pr:
            L += ["", "Regret of the one-draw capability slope minus the rule's (pp, 95% t interval over evaluation "
                  "configurations; positive: the capability slope costs more): "
                  + "; ".join(f"{RULE_NAMES[p['b']]} {ci(p)}" for p in pr) + "."]
        L.append("")
    if R["cross_protocol"]:
        L += ["## Across protocols", "",
              "Choices made under the requested arm with strict scoring, evaluated under the forced arm (strict) or "
              "under content scores (requested arm), against choices made under the evaluation's own protocol on the "
              "same pilot sets. Extra regret: the mismatched choice's regret minus the matched one's (pp, 95% t "
              "interval over evaluation configurations; positive: the wrong protocol costs points).", "",
              "| Choice under → evaluated under | Rule | Regret, matched | Regret, mismatched | Extra regret | Same choice |",
              "|---|---|---:|---:|---|---:|"]
        for c in R["cross_protocol"]:
            e = c["evaluation"]["independent"]
            for r, v in e["rules"].items():
                L.append(f"| {c['choice']} → {c['evaluation_setting']} | {RULE_NAMES[r]} | {f2(v['matched']['regret_plugin'])} | "
                         f"{f2(v['mismatched']['regret_plugin'])} | {ci(v['extra_regret'])} | {100 * v['share_same_choice']:.0f}% |")
        L.append("")
    L += ["## Notes", "",
          "- The shrunken pilot multiplies centred own gains by one factor, so it chooses the same third as own gain "
          "whenever the factor is positive; the shrunken rule here is Appendix A's replicate-informed predictor.",
          "- Intervals cover draw noise for this panel, not the choice of models or of pilot sets. The sensitivity on all "
          "distinct validation configurations is in `tables/regret_all_distinct.csv` and `decision.json`.",
          *([f"- Added tasks ({', '.join(x['title'] for k, x in R['settings'].items() if k in CHOICES)}): "
             "the choices are those the task's own run saved at its discovery stage, before its validation draws were "
             "scored (`config/added_contrasts.yaml`, section tasks); this run refitted them from the discovery draws "
             "and found them identical."] if any(k in CHOICES for k in R["settings"]) else []),
          f"- Settings not run: {', '.join(R['not_run']) if R['not_run'] else 'none'}."
          + (" The five-shot settings join when `scripts/run_decision.py` is run again, with no argument, once "
             "`data/processed/fewshot/<task>_scored.parquet` exists." if any(s.startswith('fewshot') for s in R['not_run']) else ""),
          f"- Runtime {R['runtime_s']:.0f} s."]
    return "\n".join(L)


# ============================================================================ main


def run(a):
    global DRY
    t0 = time.time()
    DRY = bool(getattr(a, "dry_run", False))
    pred_dir = getattr(a, "pred_dir", None) or (os.path.join(a.out, "frozen") if DRY else MANIFESTS)
    os.makedirs(pred_dir, exist_ok=True)
    names = []
    for s in a.settings:
        names += GROUPS.get(s, [s])
    names = list(dict.fromkeys(names))
    bad = [s for s in names if s not in SETTINGS]
    if bad:
        raise SystemExit(f"unknown settings {bad}")
    if DRY and any(SETTINGS[s][1] != "fewshot" for s in names):
        raise SystemExit("--dry-run is for the five-shot settings only (--settings fewshot)")
    not_run = [s for s in SETTINGS if s not in names or not available(s)]
    names = [s for s in names if available(s)]
    # 1. choices from discovery draws only, saved before any validation draw is built
    recs = {}
    for s in names:
        recs[s] = freeze_choices(s, pred_dir, overwrite=a.overwrite_frozen)
        print(f"{s}: choices saved ({recs[s]['created']}, {time.time() - t0:.0f} s)", flush=True)
    # 2. validation draws
    R = {"analysis": "decision analysis", "created": time.strftime("%Y-%m-%d %H:%M"),
         "dry_run": DRY, "units": "pp of accuracy gain per chosen model", "settings": {}, "cross_protocol": [],
         "not_run": not_run}
    W, V = {}, {}
    for s in names:
        rec, W[s] = load_choices(s, pred_dir)
        V[s] = setting_arrays(s, "validation")
        g_bank = None
        if SETTINGS[s][1] == "dove":
            _, _, Bb, Pb = setting_arrays(s, "bank")
            g_bank = (Pb - Bb).mean(0)
        R["settings"][s] = {"title": SETTINGS[s][4], "models": rec["models"],
                            "choices": {"path": rel(choice_path(s, pred_dir)),
                                        "created": rec["created"], "sha256": rec["sha256"], "slopes": rec["slopes"],
                                        "ties": rec["ties"]},
                            "result": evaluate(s, rec, W[s], V[s], g_bank)}
        print(f"{s}: evaluated ({time.time() - t0:.0f} s)", flush=True)
    for src, dst in CROSS:
        if src in W and dst in W:
            R["cross_protocol"].append(cross_protocol(src, dst, W[src], W[dst], recs[src], recs[dst], V[dst]))
    R["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    regret_table(R).to_csv(os.path.join(a.out, "tables", "regret.csv"), index=False)
    regret_table(R, "all_distinct").to_csv(os.path.join(a.out, "tables", "regret_all_distinct.csv"), index=False)
    paired_table(R).to_csv(os.path.join(a.out, "tables", "paired.csv"), index=False)
    ri.write_outputs(a.out, R, summary(R).split("\n"))
    print(summary(R))
    return R


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=["main", "fewshot", "added"],
                    help="setting names or groups (main, fewshot, added); default: every setting whose input exists")
    ap.add_argument("--out", default=None, help="default results/decision; a subset of the settings, or a dry run, "
                                                "writes to results/decision/subset or results/decision/dry_run")
    ap.add_argument("--dry-run", action="store_true",
                    help="five-shot settings only: run_fewshot's stand-in arm replaces the five-shot arm; choices go to "
                         "<out>/frozen, never to manifests/")
    ap.add_argument("--overwrite-frozen", action="store_true")
    ap.add_argument("--pred-dir", default=None,
                    help="write and read the choices in this folder instead of manifests/ (refits)")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = (os.path.join(OUT, "dry_run") if a.dry_run
                 else OUT if sorted(a.settings) == ["added", "fewshot", "main"] else os.path.join(OUT, "subset"))
    return run(a)


if __name__ == "__main__":
    main()
