r"""Step 3 on the validation draws whose configuration is not among the discovery draws' configurations.

The prediction rules are the ones
scripts/run_dataset2.py saved before any validation draw was read (manifests/predictions_<label>.json and
_lenient.json). They are read here and never refitted. One rule filters the validation draws: a draw is kept when
its configuration id is not among the configuration ids of the discovery draws. The contrasts on the kept draws come
from src.dataset2.evaluate_frozen, the function of the full analysis, with the interval methods of
manifests/calibration_selection_v2.json. The full-validation contrasts are recomputed by the same path and must equal
results/dataset2/<label>.json before anything is written.

Run from the repository root:
    # GSM8K, the 19 models (writes results/nonoverlap_validation/)
    python scripts/run_nonoverlap_validation.py
    # BIG-Bench Hard, once scripts/run_dataset2.py --task bbh has written manifests/predictions_bbh*.json
    python scripts/run_nonoverlap_validation.py --task bbh

Each run replaces its own task block in nonoverlap_validation.json; the macros, the CSV table and both summaries are
then rebuilt from every task block present. Units: percentage points; risks and risk differences in pp^2.
Negative estimates are reported as estimated.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import dataset1 as d1, dataset2 as d2  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
OUT = os.path.join(ROOT, "results", "nonoverlap_validation")
STEM = "nonoverlap_validation"

TASKS = {
    "gsm8k": {"label": "gsm8k", "name": "GSM8K", "tag": "Gsm",
              "scored": os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet"),
              "sampler": os.path.join(ROOT, "manifests", "sampler_gsm8k.json"),
              "scorer": os.path.join(ROOT, "manifests", "gsm8k_scorer.json"),
              "full": os.path.join(ROOT, "results", "dataset2", "gsm8k.json")},
    "bbh": {"label": "bbh", "name": "BIG-Bench Hard", "tag": "Bbh",
            "scored": os.path.join(ROOT, "data", "processed", "bbh", "scored.parquet"),
            "sampler": os.path.join(ROOT, "manifests", "sampler_bbh.json"),
            "scorer": os.path.join(ROOT, "manifests", "bbh_scorer.json"),
            "full": os.path.join(ROOT, "results", "dataset2", "bbh.json")},
}
TASK_ORDER = ("gsm8k", "bbh")
# (column of the scored table, key in this file's output, macro tag, suffix of the saved prediction file)
SCORINGS = (("strict_correct", "strict", "Strict", ""), ("lenient_correct", "lenient", "Lenient", "_lenient"))
SHORT = {"constant": "Const", "own_gain": "Own", "averaged_ols": "Ols", "replicated": "Rep",
         "single_prompt": "Single"}
WORDS = {"constant": "constant", "own_gain": "own gain", "averaged_ols": "averaged regression",
         "replicated": "replicate-informed", "single_prompt": "single-prompt regression (separate reference)"}
TOL = 1e-8          # pp^2; the recomputed full contrasts must equal the saved ones to this tolerance


def rel(path):
    """A repository-relative path with forward slashes, for the records (absolute on another drive)."""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


# --------------------------------------------------------------------------- the filter


def split_validation(sampler) -> dict:
    """Which validation draws share a configuration with the discovery draws. Draws keep their sampler order;
    a configuration drawn more than once stays as separate draws, as in the full analysis."""
    disc = [str(c) for c in sampler["discovery_config_ids"]]
    val = [str(c) for c in sampler["validation_config_ids"]]
    dset = set(disc)
    keep = [i for i, c in enumerate(val) if c not in dset]
    shared = sorted(set(val) & dset)
    kept = [val[i] for i in keep]
    return {"K_discovery": len(disc), "K_validation": len(val),
            "distinct_discovery_configs": len(dset), "distinct_validation_configs": len(set(val)),
            "shared_configs": shared, "n_shared_configs": len(shared),
            "overlapping_draws": len(val) - len(keep),
            "K_nonoverlap": len(keep), "nonoverlap_positions": keep, "nonoverlap_config_ids": kept,
            "nonoverlap_distinct_configs": len(set(kept)),
            "nonoverlap_repeated_configs": {c: kept.count(c) for c in sorted(set(kept)) if kept.count(c) > 1}}


# --------------------------------------------------------------------------- checks


def frozen_problems(frozen, sampler, panel=None) -> list:
    """Reasons the saved predictions cannot be used as they are; empty when they can."""
    p = []
    try:
        sha = sha256_json(json.loads(json.dumps(frozen["budgets"], default=d2._jsonable)))
    except (KeyError, TypeError) as exc:
        return [f"the prediction file has no readable budgets ({exc})"]
    if sha != frozen.get("sha256"):
        p.append(f"the budgets hash to {sha}, not to the saved sha256 {frozen.get('sha256')}")
    if frozen.get("sampler_sha256") != sampler.get("sha256"):
        p.append(f"the predictions were fitted under sampler {frozen.get('sampler_sha256')}, not {sampler.get('sha256')}")
    if panel is not None and list(frozen.get("models") or []) != list(panel):
        p.append("the predictions' model list differs from the panel")
    return p


def _rows_by_key(rows):
    return {(int(r["K"]), r["f"], r["g"]): r for r in rows}


def difference_from_saved(recomputed, saved) -> dict:
    """Largest absolute difference between two evaluate_frozen outputs (estimates, interval ends, absolute risks and
    their standard errors). Mismatched row sets or frozen hashes are reported as problems."""
    problems, diffs = [], []
    if recomputed["frozen_sha256"] != saved.get("frozen_sha256"):
        problems.append("frozen sha256 differs")
    if int(recomputed["K_validation"]) != int(saved.get("K_validation", -1)):
        problems.append("K_validation differs")
    for key in ("registered", "budget_curve"):
        a, b = _rows_by_key(recomputed[key]), _rows_by_key(saved.get(key, []))
        if set(a) != set(b):
            problems.append(f"{key}: different comparisons")
            continue
        for k in a:
            diffs.append(abs(a[k]["estimate"] - b[k]["estimate"]))
            ca, cb = a[k]["ci95"], b[k]["ci95"]
            if (ca is None) != (cb is None):
                problems.append(f"{key} {k}: interval present in one only")
            elif ca is not None:
                diffs += [abs(ca[0] - cb[0]), abs(ca[1] - cb[1])]
            if a[k]["classification"] != b[k]["classification"]:
                problems.append(f"{key} {k}: classification differs")
    a = {r["predictor"]: r for r in recomputed["absolute_risk"]}
    b = {r["predictor"]: r for r in saved.get("absolute_risk", [])}
    if set(a) != set(b):
        problems.append("absolute_risk: different predictors")
    else:
        for k in a:
            diffs.append(abs(a[k]["cross_risk"] - b[k]["cross_risk"]))
            sa, sb = a[k]["jackknife_se"], b[k]["jackknife_se"]
            if (sa is None) != (sb is None):
                problems.append(f"absolute_risk {k}: standard error present in one only")
            elif sa is not None:
                diffs.append(abs(sa - sb))
    d = max(diffs) if diffs else None
    if d is not None and not d <= TOL:
        problems.append(f"largest absolute difference {d:.3g} pp^2 exceeds {TOL:g}")
    return {"max_abs_difference": d, "problems": problems, "matches": not problems}


def saved_step3(full_results, value):
    if value == "strict_correct":
        return full_results["step3"]
    return full_results["sensitivities"]["lenient_extraction"]["step3"]


# --------------------------------------------------------------------------- the analysis


def _relabel(res):
    """evaluate_frozen's output with the rows at the full discovery budget under "comparisons"."""
    return {"frozen_sha256": res["frozen_sha256"], "K_validation": res["K_validation"],
            "K_discovery": res["K_discovery"], "comparisons": res["registered"],
            "budget_curve": res["budget_curve"], "absolute_risk": res["absolute_risk"], "note": res["note"]}


def step3_blocks(load, sampler, frozen_by_value, items, sel) -> tuple:
    """(split, raw) with raw[value] = {"full": evaluate_frozen on every validation draw, "nonoverlap": the same on
    the kept draws}. `load(config_ids)` returns scored rows; only validation configurations are read."""
    split = split_validation(sampler)
    panel = list(next(iter(frozen_by_value.values()))["models"])
    for fz in frozen_by_value.values():
        if list(fz["models"]) != panel:
            raise ValueError("the strict and lenient predictions cover different panels")
    val = [str(c) for c in sampler["validation_config_ids"]]
    kept = split["nonoverlap_config_ids"]
    cells = d2.cell_table(load(sorted(set(val))), panel, val, items)
    raw = {}
    for value, fz in frozen_by_value.items():
        Bv, Pv = d2.unit_arrays(cells, val, panel, value)
        res = {"full": d2.evaluate_frozen(fz, Pv - Bv, sel)}
        if len(kept) >= 2:
            Bn, Pn = d2.unit_arrays(cells, kept, panel, value)
            res["nonoverlap"] = d2.evaluate_frozen(fz, Pn - Bn, sel)
        else:
            res["nonoverlap"] = {"status": f"not computed: {len(kept)} non-overlapping validation draws; the "
                                           "cross-product risk and the t interval need at least two"}
        raw[value] = res
    return split, raw


# --------------------------------------------------------------------------- formatting


def fnum(x, nd=2):
    """A number for LaTeX, in the style of paper/make_numbers.py."""
    if x is None:
        return "--"
    if not math.isfinite(x):
        return "undefined"
    return f"{x:.{nd}f}".replace("-", "$-$")


def fmd(x):
    """A number for the Markdown summaries: one decimal, two below 1 in absolute value; Unicode minus."""
    if x is None:
        return "--"
    if not math.isfinite(x):
        return "undefined"
    s = f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"
    return s.replace("-", "−")


def fci(ci):
    return "no interval" if ci is None else f"[{fmd(ci[0])}, {fmd(ci[1])}]"


def put(macros, name, value):
    if not name.isalpha() or not name.startswith("nonov"):
        raise ValueError(f"macro name must be letters only with prefix nonov: {name}")
    if name in macros:
        raise ValueError(f"macro {name} set twice")
    macros[name] = value


def task_macros(block, macros):
    t = TASKS[block["task"]]["tag"]
    sp = block["split"]
    for name, key in (("K", "K_nonoverlap"), ("Kval", "K_validation"), ("Kdisc", "K_discovery"),
                      ("Overlap", "overlapping_draws"), ("Shared", "n_shared_configs"),
                      ("DiscConfigs", "distinct_discovery_configs"), ("KDistinct", "nonoverlap_distinct_configs")):
        put(macros, f"nonov{t}{name}", str(sp[key]))
    for _, key, tag, _ in SCORINGS:
        for which, settag in (("nonoverlap", ""), ("full", "Full")):
            res = block[key][which]
            if "comparisons" not in res:
                continue
            for r in res["comparisons"]:
                stem = f"nonov{t}{tag}{settag}RD{SHORT[r['f']]}{SHORT[r['g']]}"
                put(macros, stem, fnum(r["estimate"]))
                put(macros, stem + "Lo", fnum(None if r["ci95"] is None else r["ci95"][0]))
                put(macros, stem + "Hi", fnum(None if r["ci95"] is None else r["ci95"][1]))
            for r in res["absolute_risk"]:
                put(macros, f"nonov{t}{tag}{settag}Risk{SHORT[r['predictor']]}", fnum(r["cross_risk"]))


def table_rows(block):
    rows = []
    for _, key, _, _ in SCORINGS:
        full, non = block[key]["full"], block[key]["nonoverlap"]
        nrows = _rows_by_key(non["comparisons"]) if "comparisons" in non else {}
        for r in full["comparisons"]:
            n = nrows.get((int(r["K"]), r["f"], r["g"]))
            rows.append({"task": block["task"], "scoring": key, "quantity": "risk difference (pp2)",
                         "f": r["f"], "g": r["g"], "K_discovery": r["K"], "method": r["method"],
                         "K_full": full["K_validation"], "full_estimate": r["estimate"],
                         "full_ci95_lo": None if r["ci95"] is None else r["ci95"][0],
                         "full_ci95_hi": None if r["ci95"] is None else r["ci95"][1],
                         "full_jackknife_se": None, "full_classification": r["classification"],
                         "K_nonoverlap": block["split"]["K_nonoverlap"],
                         "nonoverlap_estimate": None if n is None else n["estimate"],
                         "nonoverlap_ci95_lo": None if n is None or n["ci95"] is None else n["ci95"][0],
                         "nonoverlap_ci95_hi": None if n is None or n["ci95"] is None else n["ci95"][1],
                         "nonoverlap_jackknife_se": None,
                         "nonoverlap_classification": None if n is None else n["classification"]})
        nrisk = {r["predictor"]: r for r in non.get("absolute_risk", [])}
        for r in full["absolute_risk"]:
            n = nrisk.get(r["predictor"])
            rows.append({"task": block["task"], "scoring": key, "quantity": "absolute cross-product risk (pp2)",
                         "f": r["predictor"], "g": None, "K_discovery": full["K_discovery"],
                         "method": "none calibrated (jackknife SE shown)", "K_full": full["K_validation"],
                         "full_estimate": r["cross_risk"], "full_ci95_lo": None, "full_ci95_hi": None,
                         "full_jackknife_se": r["jackknife_se"], "full_classification": None,
                         "K_nonoverlap": block["split"]["K_nonoverlap"],
                         "nonoverlap_estimate": None if n is None else n["cross_risk"],
                         "nonoverlap_ci95_lo": None, "nonoverlap_ci95_hi": None,
                         "nonoverlap_jackknife_se": None if n is None else n["jackknife_se"],
                         "nonoverlap_classification": None})
    return rows


def _lookup(res, f, g):
    return next(r for r in res["comparisons"] if r["f"] == f and r["g"] == g)


def task_summary(block) -> str:
    t, sp = TASKS[block["task"]], block["split"]
    M, kd, kv, kn = len(block["models"]), sp["K_discovery"], sp["K_validation"], sp["K_nonoverlap"]
    reps = sp["nonoverlap_repeated_configs"]
    rep_txt = ("; " + ", ".join(f"{c} is drawn {n} times" for c, n in reps.items())) if reps else ""
    fz = block["frozen_predictions"]
    lines = [f"## {t['name']}, {M} models", "",
             f"The sampler drew {kd} discovery and {kv} validation draws with replacement. The discovery draws use "
             f"{sp['distinct_discovery_configs']} distinct configurations. Of the {kv} validation draws, "
             f"{sp['overlapping_draws']} use one of {sp['n_shared_configs']} of those configurations. The other {kn} "
             f"draws use {sp['nonoverlap_distinct_configs']} configurations that no discovery draw uses{rep_txt}.", "",
             f"The prediction rules are the ones saved on {fz['strict']['created']} in `{fz['strict']['path']}` and "
             f"`{fz['lenient']['path']}`, fitted on the {kd} discovery draws. Nothing is refitted. Risk differences "
             f"are at the full discovery budget K = {kd}, in pp²; a positive value favours the second rule. "
             f"Classifications read the 95% interval against the {d1.PRACTICAL['risk_pp2']:g} pp² margin.", ""]
    if "comparisons" not in block["strict"]["nonoverlap"]:
        lines += [block["strict"]["nonoverlap"]["status"], ""]
        return "\n".join(lines)
    changed = []
    for _, key, _, _ in SCORINGS:
        full, non = block[key]["full"], block[key]["nonoverlap"]
        lines += [f"### {key.capitalize()} scoring", "",
                  f"| Comparison | All {kv} validation draws | {kn} non-overlapping draws | Classification on the {kn} |",
                  "|---|---|---|---|"]
        for r in full["comparisons"]:
            n = _lookup(non, r["f"], r["g"])
            same = n["classification"] == r["classification"]
            if not same:
                changed.append(f"{WORDS[r['f']]} vs {WORDS[r['g']]} ({key}): {r['classification']} on all "
                               f"{kv}, {n['classification']} on the {kn}")
            lines.append(f"| {WORDS[r['f']]} vs {WORDS[r['g']]} | {fmd(r['estimate'])} {fci(r['ci95'])} | "
                         f"{fmd(n['estimate'])} {fci(n['ci95'])} | {n['classification']}"
                         f"{'' if same else ' (all draws: ' + r['classification'] + ')'} |")
        nrisk = {r["predictor"]: r for r in non["absolute_risk"]}
        lines += ["", "Appendix C.4 cross-product risks (pp², jackknife SE in parentheses; no calibrated interval):",
                  "", f"| Rule | All {kv} draws | {kn} non-overlapping draws |", "|---|---|---|"]
        for r in full["absolute_risk"]:
            n = nrisk[r["predictor"]]
            lines.append(f"| {WORDS[r['predictor']]} | {fmd(r['cross_risk'])} ({fmd(r['jackknife_se'])}) | "
                         f"{fmd(n['cross_risk'])} ({fmd(n['jackknife_se'])}) |")
        lines.append("")
    s, l_ = block["strict"]["nonoverlap"], block["lenient"]["nonoverlap"]
    so, sc = _lookup(s, "constant", "own_gain"), _lookup(s, "constant", "averaged_ols")
    lo, lc = _lookup(l_, "constant", "own_gain"), _lookup(l_, "constant", "averaged_ols")
    neg = []
    for key in ("strict", "lenient"):
        xs = [f"{WORDS[r['predictor']]} {fmd(r['cross_risk'])}" for r in block[key]["nonoverlap"]["absolute_risk"]
              if r["cross_risk"] < 0]
        if xs:
            neg.append(" and ".join(xs) + f" ({key})")
    lines += [f"On the {kn} non-overlapping draws, own gain beats the constant by {fmd(so['estimate'])} "
              f"{fci(so['ci95'])} pp² under strict scoring and {fmd(lo['estimate'])} {fci(lo['ci95'])} under lenient "
              f"scoring. The averaged regression beats the constant by {fmd(sc['estimate'])} {fci(sc['ci95'])} "
              f"(strict) and {fmd(lc['estimate'])} {fci(lc['ci95'])} (lenient).",
              "", ("Every classification is the same as on all validation draws." if not changed else
                   "Classifications that change: " + "; ".join(changed) + "."), ""]
    if neg:
        lines += ["Negative cross-product risks on the non-overlapping draws are reported as estimated, in pp²: "
                  + "; ".join(neg) + ".", ""]
    chk = block["full_results"]
    lines += [f"The full-validation numbers above are recomputed from `{block['scored_table']['path']}` and equal "
              f"`{chk['path']}` (largest absolute difference {chk['check']['max_abs_difference']:.1g} pp² across "
              "estimates, interval ends, risks and standard errors)." if chk.get("check") else
              f"`{chk['path']}` was absent, so the full-validation numbers are only the recomputed ones.", ""]
    return "\n".join(lines)


CHOICES = """## Choices

- Filter. A validation draw is kept when its configuration id is not among the configuration ids of all discovery \
draws. The same kept draws serve every nested budget of the budget curve, which is in the \
JSON only; a rule fitted on the first K discovery draws may have used fewer configurations than the filter removes.
- Repeated configurations. The sampler draws with replacement, and the full analysis keeps a configuration drawn \
more than once as separate draws. So does this one. Such draws have identical outputs, because the runner generates a \
repeated input once.
- Intervals. Each risk difference uses the method selected for it in `manifests/calibration_selection_v2.json` \
(`jack_t`: a t interval over validation draws with K_v − 1 degrees of freedom). The calibration simulated validation \
draws independent of the discovery draws, with equal discovery and validation counts of 8, 12, 16 or 24. A subset \
chosen by its overlap with the discovery draws, with fewer validation than discovery draws, was not among its designs. \
The cross-product risks have no calibrated interval in the full analysis either; their jackknife standard errors are \
shown.
- Target. Selecting on the discovery draws changes the estimand. The kept draws estimate risks over the \
configurations outside the discovery set, weighted by their bank probabilities; all validation draws estimate risks \
over the whole bank.
- Panel and scoring. The panel is the model list saved with the predictions. Strict scoring is the main analysis and \
lenient scoring its extraction sensitivity, each with its own saved rules.
"""


def summary_text(blocks) -> str:
    head = ["# Step 3 on validation draws that share no configuration with the discovery draws", "",
            "Code: "
            "`scripts/run_nonoverlap_validation.py`; numbers: `nonoverlap_validation.json` (macros prefixed "
            "`nonov`) and `nonoverlap_validation.csv`.", ""]
    body = [task_summary(blocks[k]) for k in TASK_ORDER if k in blocks]
    for k in TASK_ORDER:
        if k in blocks:
            continue
        line = (f"{TASKS[k]['name']} is added by `--task {k}` once `manifests/predictions_{TASKS[k]['label']}.json` "
                "exists.")
        if os.path.exists(TASKS[k]["sampler"]):
            sp = split_validation(json.load(open(TASKS[k]["sampler"], encoding="utf-8")))
            line += (f" Its sampler keeps {sp['K_nonoverlap']} of its {sp['K_validation']} validation draws under "
                     "the same rule.")
        body.append(line + "\n")
    return "\n".join(head + body + [CHOICES])


def write_outputs(block, out_dir=OUT) -> dict:
    """Merge `block` into nonoverlap_validation.json and rebuild macros, the CSV and both summaries."""
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, f"{STEM}.json")
    doc = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else {}
    tasks = {k: v for k, v in doc.get("tasks", {}).items() if k in TASKS}
    tasks[block["task"]] = block
    macros = {}
    for k in TASK_ORDER:
        if k in tasks:
            task_macros(tasks[k], macros)
    doc = {"analysis": "Step 3 on validation draws whose configuration is not among the discovery draws' "
                       "configurations",
           "units": "percentage points; risks and risk differences in pp^2", "macros": macros,
           "tasks": {k: tasks[k] for k in TASK_ORDER if k in tasks}}
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=d2._jsonable)
    rows = [r for k in TASK_ORDER if k in tasks for r in table_rows(tasks[k])]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{STEM}.csv"), index=False)
    text = summary_text(doc["tasks"])
    # The summary carries no inline citation markers, so the clean copy has the same text.
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    return doc


def build_block(task, split, raw, frozen_info, sel, provenance, full_path, checks, models, runtime):
    block = {"task": task, "label": TASKS[task]["label"], "models": list(models), "split": split,
             "frozen_predictions": frozen_info,
             "selection": {"path": sel["_path"].replace("\\", "/"), "sha256": sel["_sha256"]},
             "scored_table": provenance, "full_results": {"path": rel(full_path), "check": checks},
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_seconds": runtime}
    for value, key, _, _ in SCORINGS:
        r = raw[value]
        block[key] = {"full": _relabel(r["full"]),
                      "nonoverlap": _relabel(r["nonoverlap"]) if "registered" in r["nonoverlap"] else r["nonoverlap"]}
    return block


# --------------------------------------------------------------------------- inputs


def _refuse(msg):
    sys.exit(f"refused: {msg}")


def load_inputs(a):
    """Checked inputs of one task: sampler, saved predictions, item ids, a loader for scored rows, provenance."""
    from src import d2_io
    t = TASKS[a.task]
    try:
        sampler = json.load(open(a.sampler, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _refuse(f"cannot read the sampler {a.sampler}: {exc}")
    if sampler.get("label") != t["label"]:
        _refuse(f"{a.sampler} has label {sampler.get('label')!r}, expected {t['label']!r}")
    frozen = {}
    for value, key, _, suffix in SCORINGS:
        path = a.predictions.replace(".json", f"{suffix}.json")
        if not os.path.exists(path):
            _refuse(f"{rel(path)} does not exist; scripts/run_dataset2.py writes it before it reads "
                    "the validation draws")
        frozen[value] = json.load(open(path, encoding="utf-8"))
        frozen[value]["_path"] = path
    panel = list(frozen["strict_correct"]["models"])
    for value, fz in frozen.items():
        probs = frozen_problems(fz, sampler, panel)
        if probs:
            _refuse(f"{rel(fz['_path'])}: " + "; ".join(probs))
    try:
        md = d2_io.scored_metadata(a.scored)
    except (OSError, ValueError) as exc:
        _refuse(f"cannot read {a.scored}: {exc}")
    try:
        if a.task == "bbh":
            from src import bbh_d2
            items_df, items_info = bbh_d2.load_items("target")
            items_label = bbh_d2.TARGET_LABEL
        else:
            items_df, items_info = d2_io.load_items(split="test")
            items_label = "the pinned GSM8K test items"
    except (OSError, ValueError, d2_io.D2IOError) as exc:
        _refuse(str(exc))
    problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info, scorer_path=a.scorer,
                                           items_label=items_label)
    if md.get("d2_task", "gsm8k") != a.task:
        problems.append(f"d2_task is {md.get('d2_task', 'gsm8k')!r}, expected {a.task!r}")
    if problems:
        _refuse(f"{rel(a.scored)} is not the scored table of the full analysis: " + "; ".join(problems))
    items = items_df["item_id"].astype(str).tolist() if a.task == "bbh" else items_df["item_id"].tolist()

    def load(ids):
        df = pd.read_parquet(a.scored, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        if a.task == "bbh":
            df["item_id"] = df["item_id"].astype(str)
        return df

    provenance = {"path": rel(a.scored), "sha256": d2_io.sha256_path(a.scored),
                  "scorer_version": md.get("d2_scorer_version"), "created": md.get("d2_created")}
    return sampler, frozen, items, load, provenance, panel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=tuple(TASKS), default="gsm8k")
    ap.add_argument("--scored", help="default data/processed/<task>/scored.parquet")
    ap.add_argument("--sampler", help="default manifests/sampler_<task>.json")
    ap.add_argument("--predictions", help="strict predictions, default manifests/predictions_<task>.json; the "
                                          "lenient file is the same name with _lenient")
    ap.add_argument("--scorer", help="default manifests/<task>_scorer.json")
    ap.add_argument("--full-results", help="default results/dataset2/<task>.json")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t = TASKS[a.task]
    a.scored = a.scored or t["scored"]
    a.sampler = a.sampler or t["sampler"]
    a.predictions = a.predictions or os.path.join(ROOT, "manifests", f"predictions_{t['label']}.json")
    a.scorer = a.scorer or t["scorer"]
    a.full_results = a.full_results or t["full"]

    t0 = time.time()
    sampler, frozen, items, load, provenance, panel = load_inputs(a)
    sel = d2.load_selection(SELECTION)
    full = json.load(open(a.full_results, encoding="utf-8")) if os.path.exists(a.full_results) else None
    if full is not None:
        if full.get("models") != panel:
            _refuse(f"{a.full_results} has a different model list from the saved predictions")
        if full.get("selection", {}).get("sha256") != sel["_sha256"]:
            _refuse(f"{a.full_results} used another interval selection file")
        if full.get("sampler_sha256") != sampler["sha256"]:
            _refuse(f"{a.full_results} used another sampler")
    split, raw = step3_blocks(load, sampler, {v: {k: x for k, x in fz.items() if k != "_path"}
                                              for v, fz in frozen.items()}, items, sel)
    checks = None
    if full is not None:
        checks = {}
        for value, key, _, _ in SCORINGS:
            checks[key] = difference_from_saved(raw[value]["full"], saved_step3(full, value))
        bad = {k: c["problems"] for k, c in checks.items() if not c["matches"]}
        if bad:
            _refuse(f"the recomputed full-validation contrasts differ from {a.full_results}: {bad}")
        checks = {"max_abs_difference": max(c["max_abs_difference"] for c in checks.values()), "by_scoring": checks}
    frozen_info = {key: {"path": rel(frozen[value]["_path"]), "sha256": frozen[value]["sha256"],
                         "created": frozen[value]["created"]} for value, key, _, _ in SCORINGS}
    block = build_block(a.task, split, raw, frozen_info, sel, provenance, a.full_results, checks, panel,
                        round(time.time() - t0, 1))
    write_outputs(block, a.out)
    print(f"{t['name']}: {split['K_nonoverlap']} of {split['K_validation']} validation draws share no configuration "
          f"with the discovery draws -> {rel(a.out)}")
    for value, key, _, _ in SCORINGS:
        if "comparisons" not in block[key]["nonoverlap"]:
            print(f"  {key}: {block[key]['nonoverlap']['status']}")
            continue
        for r in block[key]["nonoverlap"]["comparisons"]:
            f_ = _lookup(block[key]["full"], r["f"], r["g"])
            ci = "no interval" if r["ci95"] is None else f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            print(f"  {key:<8} {r['f']:>13} vs {r['g']:<13} full {f_['estimate']:9.3f}  "
                  f"non-overlap {r['estimate']:9.3f} {ci}  {r['classification']}")


if __name__ == "__main__":
    main()
