r"""Own gain against capability rules by pilot budget.
How much pilot evidence does a model's own measured gain need before it predicts the model's gain on fresh draws
better than a rule from capability? Estimators and the frozen-file helpers: src/rule_budgets.py.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_rule_budgets.py 2>&1"
    cmd /c ".venv\Scripts\python.exe scripts\run_rule_budgets.py --overwrite-frozen 2>&1"   # after a design change

Settings:
    gsm8k30    GSM8K-29: the 30 checkpoints less the one the screens excluded (src/d2_panels.inputs),
               requested direct answer as baseline and chain of thought as intervention, strict scoring, 1,319 items;
               the 24 discovery and 24 validation units of manifests/sampler_gsm8k.json.
    bbh        BIG-Bench Hard: 28 models (src/d2_panels.bbh_inputs), the same arms, 25 tasks of 54 items pooled with
               equal weight; manifests/sampler_bbh.json.
    math       MATH-500: the 29 models and 398 numeric target items of results/dataset2/math.json
               (src/d2_panels.math_inputs), the same arms, strict scoring; manifests/sampler_math.json.
    dove_mmlu  The DOVE MMLU block: 5 models, five-shot against zero-shot, 5,597 items scored with the block's subject
               weights 1 / (56 x subject size) (scripts/run_noise_anatomy.py load_dove); the 24 discovery and 24
               validation draws of manifests/sampler_dove_mmlu.json.
    dove:<b>   The 71 DOVE benchmarks of the reliability map (scripts/run_reliability_map.py dove_task_inputs), 99 or
               100 items each: the 4 recorded pilot draws and the 24 validation draws of
               manifests/sampler_reliability_map_dove_tasks.json.

Budgets: K = 1, 2 and 4 pilot draws, taken without replacement from the discovery draws (the 4 pilot draws for a DOVE
benchmark, so at most 4, 6 and 1 distinct pilots there); 100, 300 and all items. BIG-Bench Hard subsamples take 4 and
12 items from each task; GSM8K and the MMLU block take simple random samples. The DOVE benchmarks have the full budget
only; their 100 and 300 rows are reported as not applicable. Per (setting, K, items): 200 pilots, each with one item
subsample shared by every model, both arms and its K draws, from the stream sampling.rng_for("rule-budgets", setting,
K, n), n = 100, 300 or "all"; per pilot the draws are taken first, then the items.

Rules, refitted on every pilot: own gain; own gain shrunken toward the pilot's panel mean by R_K (not defined at K = 1);
Step 3's capability rule (averaged OLS of gain on baseline accuracy); on GSM8K-29, BIG-Bench Hard and MATH-500 also
the rule quadratic in baseline plus the external index (manifests/d2_panel_extension.json, "eci"). Risk: the paired loss
difference against each model's gain on the 24 validation draws with all items, averaged over the 200 pilots, with
Step 3's jack_t interval over the 24 validation draws (23 df). Reported as capability rule minus own gain: positive when
own gain predicts better.

Order: every pilot's predictions are computed from discovery (or pilot) draws only and written to
manifests/predictions_rule_budgets.npz, with the design, the streams and that file's sha256 in
manifests/predictions_rule_budgets.json, before any validation array is built. Dataset 2 validation units are read
from disk only after that. The DOVE loaders read each bank whole, as scripts/run_reliability_map.py does, and the
validation draws are taken from the banks only after that. Scoring reads the predictions back from the npz after
checking its sha256. Earlier analyses had read these validation draws, so this is not a fresh test.

Checks: with all items, the rules on the first K discovery draws reproduce the frozen own-gain and capability
predictions of manifests/predictions_<setting>.json at every budget (K = 1 to 24); the validation gains reproduce the
cross-product risk of the frozen 24-draw own-gain rule printed in results/dataset2/<setting>.json and
results/dataset1/dove_mmlu.json.

Outputs: results/rule_budgets/rule_budgets.json (with a "macros" object, prefix "rb", for paper/make_numbers.py),
summary.md, summary_clean.md, tables/risk_differences.csv and tables/dove_benchmarks.csv.
"""
import argparse
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import capability_checks as cc, dataset2 as d2, noise_anatomy as na, rule_budgets as rb, sampling  # noqa: E402
from src.estimators import cross_risk  # noqa: E402

MAN = os.path.join(ROOT, "manifests")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "rule_budgets")
PRED_JSON = os.path.join(MAN, "predictions_rule_budgets.json")
PRED_NPZ = os.path.join(MAN, "predictions_rule_budgets.npz")
LABEL = "rule-budgets"
N_PILOTS = 200
KS = (1, 2, 4)
ITEM_BUDGETS = (100, 300, None)          # None: all items
TITLE = {"gsm8k30": "GSM8K-29", "bbh": "BIG-Bench Hard", "math": "MATH-500", "dove_mmlu": "DOVE MMLU block"}
TAG = {"gsm8k30": "GsmThirty", "bbh": "Bbh", "math": "Math", "dove_mmlu": "DoveBlock"}
ND = {"gsm8k30": 1, "bbh": 1, "math": 1, "dove_mmlu": 2}
K_WORD = {1: "One", 2: "Two", 4: "Four"}
N_WORD = {100: "Hundred", 300: "ThreeHundred", "all": "All"}
RULE_WORDS = {"own_gain": "own gain", "own_gain_shrunk": "own gain, shrunken",
              "averaged_ols": "capability, linear in baseline",
              cc.FULL_RULE: "capability, quadratic in baseline plus the index"}
RULE_TAG = {"own_gain": "Own", "own_gain_shrunk": "Shrunk", "averaged_ols": "Ols", cc.FULL_RULE: "Quad"}
NOT_DEFINED = "not defined at K = 1 (one draw cannot estimate R)"
NOT_APPLICABLE = "not applicable (the benchmark has 99 or 100 items)"


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_rule_budgets", os.path.join(HERE, f"{name}.py"))
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


def nlabel(n):
    return "all" if n is None else int(n)


# ============================================================================ inputs (discovery draws only)


def d2_setting(name):
    """GSM8K-29, BIG-Bench Hard or MATH-500: the discovery draws as item scores (arms, draws, models, items) and a loader of the
    validation gains, which reads the validation units from disk only when called."""
    from run_transfer_tasks import eci_for
    from src import d2_io, d2_panels
    rna = load_script("run_noise_anatomy")
    task = name if name in ("bbh", "math") else "gsm8k"
    sampler = json.load(open(os.path.join(MAN, f"sampler_{task}.json"), encoding="utf-8"))
    scorer = os.path.join(MAN, f"{task}_scorer.json")
    if name == "gsm8k30":
        items_df, items_info = d2_io.load_items(split="test")
        got = d2_panels.inputs(None, None, sampler, items_info, scorer)
        items = items_df["item_id"].tolist()
        by_task, strata = {"gsm8k": items}, None
    elif name == "math":
        got = d2_panels.math_inputs(None, sampler, scorer_path=scorer)
        items = got["items"]
        by_task, strata = {"math": items}, None
    else:
        got = d2_panels.bbh_inputs(None, sampler, scorer_path=scorer)
        items, by_task = got["items"], got["items_by_task"]
        pos = {it: i for i, it in enumerate(items)}
        strata = [np.array([pos[i] for i in its]) for _, its in sorted(by_task.items())]
        if sorted(np.concatenate(strata).tolist()) != list(range(len(items))) or len({len(s) for s in strata}) != 1:
            raise SystemExit("BIG-Bench Hard: the tasks do not partition the items into equal sizes")
    panel = list(got["panel"])
    main = json.load(open(os.path.join(RES, "dataset2", f"{name}.json"), encoding="utf-8"))
    if main["models"] != panel or main.get("sampler_sha256") != sampler["sha256"]:
        raise SystemExit(f"{name}: the panel or sampler differs from results/dataset2/{name}.json")
    disc, val = list(sampler["discovery_config_ids"]), list(sampler["validation_config_ids"])
    configs = sorted(set(map(str, disc)))
    S = rna.dense_items(got["load"](configs), panel, configs, items)
    at = np.array([configs.index(str(u)) for u in disc])

    def validation():
        df = got["load"](val)
        parts = []
        for its in by_task.values():
            cells = d2.cell_table(df[df["item_id"].isin(pd.Index(its))], panel, val, its, values=("strict_correct",))
            parts.append(d2.unit_arrays(cells, val, panel, "strict_correct"))
        Bv, Pv = (np.mean([p[i] for p in parts], axis=0) for i in (0, 1))
        return Pv - Bv

    own = next(r for r in main["step3"]["absolute_risk"] if r["predictor"] == "own_gain")["cross_risk"]
    return {"setting": name, "title": TITLE[name], "group": "d2", "models": panel, "SB": S[0][at], "SP": S[1][at],
            "v": None, "strata": strata, "n_items": len(items), "item_budgets": ITEM_BUDGETS,
            "eci": eci_for(panel), "configs": [str(u) for u in disc], "validation": validation,
            "item_sampling": ("4 and 12 items from each of the 25 tasks; tasks weigh equally" if strata is not None
                              else "simple random sample; items weigh equally"),
            "draws": f"manifests/sampler_{task}.json (24 discovery units, 24 validation units)",
            "reference": {"predictions": os.path.join(MAN, f"predictions_{name}.json"),
                          "own_gain_risk": own, "source": f"results/dataset2/{name}.json (step3.absolute_risk)"}}


def dove_block_setting():
    """The DOVE MMLU block: discovery draws as item scores with the subject weights; validation gains from the bank."""
    rna = load_script("run_noise_anatomy")
    data = rna.load_dove()
    sampler = json.load(open(os.path.join(MAN, "sampler_dove_mmlu.json"), encoding="utf-8"))
    disc = np.asarray(sampler["discovery_bank_index"])
    val = np.asarray(sampler["validation_bank_index"])
    v = data["weights"]

    def validation():
        return na.full_scores(data["P"][val], v) - na.full_scores(data["B"][val], v)

    d1r = json.load(open(os.path.join(RES, "dataset1", "dove_mmlu.json"), encoding="utf-8"))
    return {"setting": "dove_mmlu", "title": TITLE["dove_mmlu"], "group": "dove_mmlu", "models": list(data["panel"]),
            "SB": data["B"][disc], "SP": data["P"][disc], "v": v, "strata": None, "n_items": int(len(v)),
            "item_budgets": ITEM_BUDGETS, "eci": None, "configs": [int(c) for c in disc], "validation": validation,
            "item_sampling": "simple random sample; scored with the block's subject weights 1 / (56 x subject size)",
            "draws": "manifests/sampler_dove_mmlu.json (24 discovery draws, 24 validation draws)",
            "reference": {"predictions": os.path.join(MAN, "predictions_dove_mmlu.json"),
                          "own_gain_risk": d1r["step3"]["registered"]["risks"]["own_gain"]["cross_risk"],
                          "source": "results/dataset1/dove_mmlu.json (step3.registered.risks)"}}


def dove_task_settings():
    """The 71 DOVE benchmarks: the 4 pilot draws as (draws, models) scores; validation gains from each bank."""
    import yaml
    rmap = load_script("run_reliability_map")
    cfg = yaml.safe_load(open(rmap.CONFIG, encoding="utf-8"))
    tasks, _, _ = rmap.dove_task_inputs(cfg)
    out = []
    for t in tasks:
        pi, vi = np.asarray(t["pilot"]), np.asarray(t["validation"])

        def validation(B=t["B"], P=t["P"], vi=vi):
            return P[vi] - B[vi]
        out.append({"setting": t["setting"], "title": t["title"], "group": "dove_tasks", "models": list(t["models"]),
                    "Bd": t["B"][pi], "Pd": t["P"][pi], "n_items": int(t["n_items"]), "item_budgets": (None,),
                    "eci": None, "configs": [int(c) for c in pi], "validation": validation,
                    "item_sampling": "all items only", "draws": t["draws"] + " (4 pilot draws, 24 validation draws)"})
    return out


# ============================================================================ discovery check


def discovery_scores(s, items=None):
    """(B, P) scores (draws, models, pilots) of the setting's discovery draws on the pilots' item subsamples (one
    column, all items, when items is None)."""
    if "SB" not in s:
        return s["Bd"][:, :, None], s["Pd"][:, :, None]
    U = rb.subsample_weights(s["n_items"], items, s.get("v"), s.get("strata"))
    return na.subsample_scores(s["SB"], U), na.subsample_scores(s["SP"], U)


def check_discovery(s):
    """With all items, the rules on the first K discovery draws reproduce the frozen Step 3 predictions."""
    ref = json.load(open(s["reference"]["predictions"], encoding="utf-8"))
    if ref["models"] != s["models"]:
        raise SystemExit(f"{s['setting']}: {rel(s['reference']['predictions'])} has another panel")
    XB, XP = discovery_scores(s)
    worst = 0.0
    for k, preds in ref["budgets"].items():
        fit = rb.fit_pilot(XB[:int(k), :, 0], XP[:int(k), :, 0], s.get("eci"))
        for r in ("own_gain", "averaged_ols"):
            worst = max(worst, float(np.max(np.abs(fit[r] - np.asarray(preds[r]["prediction"], float)))))
    if worst > 1e-9:
        raise SystemExit(f"{s['setting']}: the discovery rules differ from {rel(s['reference']['predictions'])} "
                         f"by up to {worst:.3g} pp")
    return {"check": f"{s['title']}: with all items, own gain and the capability rule on the first K discovery draws "
                     f"reproduce {rel(s['reference']['predictions'])} at K = {', '.join(ref['budgets'])}",
            "max_abs_difference_pp": worst, "passed": True}


def check_validation(s, Z):
    ref = json.load(open(s["reference"]["predictions"], encoding="utf-8"))
    own = np.asarray(ref["budgets"]["24"]["own_gain"]["prediction"], float)
    r = cross_risk(Z, own)
    want = s["reference"]["own_gain_risk"]
    if abs(r - want) > 1e-6 * max(1.0, abs(want)):
        raise SystemExit(f"{s['setting']}: validation risk {r} differs from {s['reference']['source']} ({want})")
    return {"check": f"{s['title']}: the validation gains reproduce the frozen 24-draw own-gain rule's cross-product "
                     f"risk in {s['reference']['source']}", "value": r, "reference": want, "passed": True}


# ============================================================================ pilots (before validation)


def cell_key(setting, K, n):
    return f"{setting.replace(':', '-')}__K{K}__n{nlabel(n)}__"


def fit_cells(settings, n_pilots=N_PILOTS):
    """Every pilot's rules, from discovery draws only: the arrays for the npz and the design for the manifest."""
    arrays, cells = {}, []
    for s in settings:
        n_draws = len(s["configs"])
        M = len(s["models"])
        rules = rb.OWN + (("averaged_ols", cc.FULL_RULE) if s.get("eci") is not None else ("averaged_ols",))
        for K in KS:
            for n in s["item_budgets"]:
                rng = sampling.rng_for(LABEL, s["setting"], K, nlabel(n))
                pos, items = rb.draw_pilots(n_pilots, n_draws, K, rng, N=s["n_items"], n=n, strata=s.get("strata"))
                XB, XP = discovery_scores(s, items)
                col = np.arange(n_pilots) if items is not None else np.zeros(n_pilots, dtype=int)
                P = {r: np.full((n_pilots, M), np.nan) for r in rules}
                factor, R, raw = (np.full(n_pilots, np.nan) for _ in range(3))
                for j in range(n_pilots):
                    fit = rb.fit_pilot(XB[:, :, col[j]][pos[j]], XP[:, :, col[j]][pos[j]], s.get("eci"))
                    for r in rules:
                        if fit[r] is not None:
                            P[r][j] = fit[r]
                    factor[j] = np.nan if fit["factor"] is None else fit["factor"]
                    R[j] = fit["R"]
                    raw[j] = rb.unclipped_factor(fit["R"], K)
                key = cell_key(s["setting"], K, n)
                arrays[key + "positions"] = pos.astype(np.int16)
                if items is not None:
                    arrays[key + "items"] = items.astype(np.int16)
                for r in rules:
                    arrays[key + r] = P[r]
                arrays[key + "factor"] = factor
                arrays[key + "R"] = R
                configs = np.asarray(s["configs"], dtype=object)[pos]
                cells.append({"setting": s["setting"], "K": K, "n": nlabel(n), "key": key, "rules": list(rules),
                              "stream": [LABEL, s["setting"], K, nlabel(n)], "pilots": n_pilots,
                              "distinct_pilots": rb.distinct_pilots(configs, items),
                              "factor": {"defined": int(np.isfinite(factor).sum()),
                                         "mean": rb.finite_or_none(np.nanmean(factor)) if np.isfinite(factor).any()
                                         else None,
                                         "held_at_zero": int(np.sum(raw <= 0)), "held_at_one": int(np.sum(raw >= 1)),
                                         "R_undefined": int(np.sum(~np.isfinite(R))) if K >= 2 else None}})
    return arrays, cells


def manifest_record(settings, cells):
    sets = {}
    for s in settings:
        sets[s["setting"]] = {"title": s["title"], "group": s["group"], "models": s["models"], "M": len(s["models"]),
                              "n_items": s["n_items"], "item_budgets": [nlabel(n) for n in s["item_budgets"]],
                              "item_sampling": s["item_sampling"], "draws": s["draws"],
                              "discovery_configs": s["configs"], "index": s.get("eci")}
    return {"label": "rule_budgets", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "units": "pp, uncentred predictions of each model's gain",
            "methods": "scripts/run_rule_budgets.py",
            "note": "every pilot's predictions from discovery (or pilot) draws only; written before any validation "
                    "array is built. Earlier analyses had read the validation draws, so this is not a fresh test",
            "design": {"n_pilots": N_PILOTS, "K": list(KS), "item_budgets": [nlabel(n) for n in ITEM_BUDGETS],
                       "stream": "sampling.rng_for('rule-budgets', setting, K, n), n = 100, 300 or 'all'; per pilot "
                                 "the K draws (without replacement) first, then the item subsample",
                       "master_seed": sampling.MASTER_SEED,
                       "rules": {r: RULE_WORDS[r] for r in rb.RULES},
                       "npz_arrays": "<key>positions (pilots, K) discovery positions; <key>items (pilots, n) item "
                                     "positions (absent for all items); <key><rule> (pilots, models) predictions, nan "
                                     "where the rule is not defined; <key>factor, <key>R (pilots,)",
                       "settings": sets, "cells": cells}}


# ============================================================================ scoring (after the freeze)


def score_cells(settings, record, arrays, Z):
    rows = []
    by = {s["setting"]: s for s in settings}
    for c in record["design"]["cells"]:
        s, z = by[c["setting"]], Z[c["setting"]]
        P = {r: arrays[c["key"] + r] for r in c["rules"]}
        ok = {r: np.isfinite(P[r]).all(1) for r in P}
        risk = {r: (float(rb.cross_risks(z, P[r][ok[r]]).mean()) if ok[r].any() else None) for r in P}
        for f, g in rb.COMPARISONS:
            if f not in P:
                continue
            use = ok[f] & ok[g]
            row = {"setting": c["setting"], "title": s["title"], "group": s["group"], "M": len(s["models"]),
                   "K": c["K"], "items": c["n"], "f": f, "g": g, "pilots": c["pilots"],
                   "distinct_pilots": c["distinct_pilots"], "pilots_used": int(use.sum()),
                   "factor_mean": c["factor"]["mean"],
                   "factor_held_at_zero": c["factor"]["held_at_zero"] if c["K"] >= 2 else None,
                   "factor_held_at_one": c["factor"]["held_at_one"] if c["K"] >= 2 else None,
                   "risk_f": risk[f], "risk_g": risk[g]}
            if not use.any():
                row.update({"status": NOT_DEFINED, "estimate": None, "ci95": None})
            else:
                d = rb.averaged_difference(rb.loss_differences(z, P[f][use], P[g][use]))
                row.update({"status": "ok" if use.all() else f"own gain shrunken not defined in "
                                                               f"{int((~use).sum())} of {c['pilots']} pilots",
                            **{k: d[k] for k in ("estimate", "ci95", "se", "df", "share_pilots_positive",
                                                 "pilot_q05", "pilot_median", "pilot_q95")}})
                if s["group"] != "dove_tasks":
                    row["per_draw"] = d["per_draw"]
            rows.append(row)
        if s["group"] == "dove_tasks" and c["n"] == "all":
            for n in (100, 300):
                for f, g in rb.COMPARISONS:
                    if f in P:
                        rows.append({"setting": c["setting"], "title": s["title"], "group": s["group"],
                                     "M": len(s["models"]), "K": c["K"], "items": n, "f": f, "g": g,
                                     "status": NOT_APPLICABLE, "estimate": None, "ci95": None})
    return rows


def dove_summary(rows):
    out = []
    for K in KS:
        for f, g in rb.COMPARISONS:
            x = [r for r in rows if r["group"] == "dove_tasks" and r["K"] == K and r["items"] == "all"
                 and r["f"] == f and r["g"] == g]
            if not x:
                continue
            est = [r["estimate"] for r in x if r["estimate"] is not None]
            ci = [r["ci95"] for r in x if r.get("ci95")]
            out.append({"K": K, "f": f, "g": g, "benchmarks": len(x), "defined": len(est),
                        "own_better": sum(c[0] > 0 for c in ci), "capability_better": sum(c[1] < 0 for c in ci),
                        "spans_zero": sum(c[0] <= 0 <= c[1] for c in ci), "no_interval": len(est) - len(ci),
                        "positive_estimate": sum(e > 0 for e in est),
                        "median": float(np.median(est)) if est else None, "mean": float(np.mean(est)) if est else None,
                        "distinct_pilots_max": max(r["distinct_pilots"] for r in x) if est else None,
                        "status": "ok" if est else x[0]["status"]})
    return out


def compute(settings, json_path=PRED_JSON, npz_path=PRED_NPZ, overwrite=False, n_pilots=N_PILOTS):
    """Pilots fitted and frozen first; each setting's validation loader is called only after the frozen pair is
    written, and scoring reads the predictions back from the checked npz."""
    arrays, cells = fit_cells(settings, n_pilots)
    record = rb.freeze(manifest_record(settings, cells), arrays, json_path, npz_path, overwrite)
    record, arrays = rb.load_frozen(json_path, npz_path)
    Z = {s["setting"]: np.asarray(s["validation"](), float) for s in settings}     # validation arrays, built only now
    return record, Z, score_cells(settings, record, arrays, Z)


# ============================================================================ macros and summary


def fmt(x, nd=2):
    if x is None:
        return "--"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "--"
    if round(x, nd) == 0:
        x = 0.0
    return f"{x:,.{nd}f}".replace("-", "−")


def ci_txt(r, nd):
    c = r.get("ci95")
    return "--" if not c else f"[{fmt(c[0], nd)}, {fmt(c[1], nd)}]"


def items_txt(s, n):
    if n == "all":
        return f"all ({s['n_items']:,})"
    if s.get("strata"):                    # the number of strata (tasks)
        return f"{n} ({n // s['strata']} per task)"
    return str(n)


def macros(out):
    m = {}

    def put(k, v):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
        m[k] = str(v).replace("−", "$-$")
    put("rbPilots", N_PILOTS)
    for name, s in out["settings"].items():
        if s["group"] == "dove_tasks":
            continue
        t = TAG[name]
        put(f"rb{t}M", s["M"])
        put(f"rb{t}Items", f"{s['n_items']:,}")
    for r in out["rows"]:
        if r["group"] == "dove_tasks":
            continue
        t, nd = TAG[r["setting"]], ND[r["setting"]]
        stem = f"rb{t}{RULE_TAG[r['f']]}{RULE_TAG[r['g']]}K{K_WORD[r['K']]}N{N_WORD[r['items']]}"
        put(stem, fmt(r["estimate"], nd))
        put(stem + "Lo", fmt(r["ci95"][0], nd) if r.get("ci95") else "--")
        put(stem + "Hi", fmt(r["ci95"][1], nd) if r.get("ci95") else "--")
        cell = f"K{K_WORD[r['K']]}N{N_WORD[r['items']]}"
        for rule, rk in ((r["f"], r.get("risk_f")), (r["g"], r.get("risk_g"))):
            put(f"rb{t}Risk{RULE_TAG[rule]}{cell}", fmt(rk, nd))
        if r["K"] >= 2:
            put(f"rb{t}Factor{cell}", fmt(r["factor_mean"], 2))
    d = out["dove_benchmarks"]
    put("rbDoveN", d["n_benchmarks"])
    put("rbDoveItemsMin", d["n_items_min"])
    put("rbDoveItemsMax", d["n_items_max"])
    for x in d["summary"]:
        stem = f"rbDove{RULE_TAG[x['f']]}{RULE_TAG[x['g']]}K{K_WORD[x['K']]}"
        if x["status"] != "ok":
            for k in ("OwnBetter", "CapBetter", "Spans", "Median"):
                put(stem + k, "--")
            continue
        put(stem + "OwnBetter", x["own_better"])
        put(stem + "CapBetter", x["capability_better"])
        put(stem + "Spans", x["spans_zero"])
        put(stem + "Median", fmt(x["median"], 2))
    return m


def summary(out):
    S = out["settings"]
    fr = out["frozen"]
    L = ["# Own gain against capability rules by budget", "",
         "Per setting, K = 1, 2 or 4 pilot draws taken without "
         "replacement from the discovery draws, and 100, 300 or all items; 200 pilots per (setting, K, items), each "
         "with one item subsample shared by every model, both arms and its draws (stream "
         "`sampling.rng_for(\"rule-budgets\", setting, K, n)`). On each pilot the rules are refitted: own gain (each "
         "model's mean gain over the pilot); own gain shrunken toward the pilot's panel mean by R_K = K R / (1 + (K − 1) "
         "R), with R from the same pilot and the factor held to [0, 1] (`src/pilot_checks.shrink_factor`; one draw "
         "cannot estimate R, so it is not defined at K = 1); Step 3's capability rule, linear in baseline accuracy; and "
         "on GSM8K-29 and BIG-Bench Hard the rule quadratic in baseline plus the external index. The target is each "
         "model's gain on the 24 validation draws with all items, uncentred as in Step 3. Each entry is the capability "
         "rule's paired loss minus own gain's, per validation draw, averaged over the 200 pilots, with Step 3's "
         "jack_t interval over the 24 validation draws (23 df): **positive when own gain predicts better**. The "
         "interval reflects the validation draws only; the pilots' spread is in the JSON and the CSV.", "",
         f"The pilots' predictions were written to `{fr['npz']}` (sha256 `{fr['npz_sha256'][:16]}…`) and "
         f"`{fr['path']}` ({fr['created']}) before the validation arrays were built, and scoring read them back after "
         "checking the hash. Earlier analyses had read these validation draws, so this is not a fresh test.", "",
         "Settings: " + "; ".join(
             f"{s['title']}, {s['M']} models, {s['n_items']:,} items ({s['item_sampling']})"
             for n, s in S.items() if s["group"] != "dove_tasks") +
         f"; and {out['dove_benchmarks']['n_benchmarks']} DOVE benchmarks of "
         f"{out['dove_benchmarks']['n_items_min']}–{out['dove_benchmarks']['n_items_max']} items, 5 models, with their "
         "4 recorded pilot draws and 24 validation draws (all items only).", "",
         "## Capability rule minus own gain (pp²)", "",
         "| Setting | K | Items | Capability rule | − own gain | 95% interval | − own gain, shrunken | 95% interval | "
         "Mean factor |", "|---|---:|---|---|---:|---|---:|---|---:|"]
    rows = [r for r in out["rows"] if r["group"] != "dove_tasks"]
    for name in (n for n in S if S[n]["group"] != "dove_tasks"):
        s, nd = S[name], ND[name]
        for K in KS:
            for n in [nlabel(x) for x in ITEM_BUDGETS]:
                for f in rb.CAPABILITY:
                    raw = next((r for r in rows if r["setting"] == name and r["K"] == K and r["items"] == n
                                and r["f"] == f and r["g"] == "own_gain"), None)
                    if raw is None:
                        continue
                    sh = next(r for r in rows if r["setting"] == name and r["K"] == K and r["items"] == n
                              and r["f"] == f and r["g"] == "own_gain_shrunk")
                    shv = ("not defined", "--") if sh["estimate"] is None else (fmt(sh["estimate"], nd), ci_txt(sh, nd))
                    L.append(f"| {s['title']} | {K} | {items_txt(s, n)} | {RULE_WORDS[f].replace('capability, ', '')} | "
                             f"{fmt(raw['estimate'], nd)} | {ci_txt(raw, nd)} | {shv[0]} | {shv[1]} | "
                             f"{fmt(raw['factor_mean'], 2) if K >= 2 else '--'} |")
    L += ["", "At K = 1 the shrunken rule is not defined: one draw cannot estimate R."]
    held = [r for r in rows if r["g"] == "own_gain" and r["f"] == "averaged_ols" and (r.get("factor_held_at_zero")
                                                                                       or r.get("factor_held_at_one"))]
    if held:
        L += ["", "Pilots whose factor was held at zero (R ≤ 0, so the shrunken rule predicts the pilot's panel mean "
                  "for every model) or at one, of 200: " + "; ".join(
                      f"{S[r['setting']]['title']}, K = {r['K']}, {r['items']} items: {r['factor_held_at_zero']} at "
                      f"zero, {r['factor_held_at_one']} at one" for r in held) + "."]
    L.append("")
    notes = [r for r in rows if r.get("status") not in ("ok", NOT_DEFINED)]
    for r in notes:
        L.append(f"- {r['title']}, K = {r['K']}, {r['items']} items, {r['f']} against {r['g']}: {r['status']}.")
    L += ["## Mean cross-product risk over the pilots (pp²)", "",
          "Cross-product risk of each rule against the validation gains, averaged over the 200 pilots "
          "(may be negative).", "",
          "| Setting | K | Items | Own gain | Own gain, shrunken | Linear in baseline | Quadratic plus index | "
          "Share of pilots where own gain beats the linear rule |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name in (n for n in S if S[n]["group"] != "dove_tasks"):
        s, nd = S[name], ND[name]
        for K in KS:
            for n in [nlabel(x) for x in ITEM_BUDGETS]:
                x = {(r["f"], r["g"]): r for r in rows if r["setting"] == name and r["K"] == K and r["items"] == n}
                if not x:
                    continue
                base = x[("averaged_ols", "own_gain")]
                quad = x.get((cc.FULL_RULE, "own_gain"))
                shr = x[("averaged_ols", "own_gain_shrunk")]
                L.append(f"| {s['title']} | {K} | {items_txt(s, n)} | {fmt(base['risk_g'], nd)} | "
                         f"{fmt(shr['risk_g'], nd) if shr['estimate'] is not None else 'not defined'} | "
                         f"{fmt(base['risk_f'], nd)} | {fmt(quad['risk_f'], nd) if quad else '--'} | "
                         f"{fmt(100 * base['share_pilots_positive'], 0)}% |")
    d = out["dove_benchmarks"]
    L += ["", f"## The {d['n_benchmarks']} DOVE benchmarks", "",
          "All items only (the 100- and 300-item rows are not applicable). Capability rule linear in baseline; each "
          "benchmark's interval is read against zero. With 4 recorded pilot draws there are at most 4, 6 and 1 "
          "distinct pilots at K = 1, 2 and 4.", "",
          "| K | Own gain | Benchmarks with an estimate | Own gain better | Capability better | Interval spans zero | "
          "Median difference (pp²) | Mean | Distinct pilots (max) |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in d["summary"]:
        if x["status"] != "ok":
            L.append(f"| {x['K']} | {RULE_WORDS[x['g']]} | not defined | -- | -- | -- | -- | -- | -- |")
            continue
        L.append(f"| {x['K']} | {RULE_WORDS[x['g']]} | {x['defined']} of {x['benchmarks']} | {x['own_better']} | "
                 f"{x['capability_better']} | {x['spans_zero']} | {fmt(x['median'], 2)} | {fmt(x['mean'], 2)} | "
                 f"{x['distinct_pilots_max']} |")
    if any(x.get("no_interval") for x in d["summary"]):
        L += ["", "Benchmarks with an estimate but no interval (the per-draw differences do not vary): " + "; ".join(
            f"K = {x['K']}, {RULE_WORDS[x['g']]}: {x['no_interval']}" for x in d["summary"] if x.get("no_interval"))
              + "."]
    L += ["", "## Checks", ""]
    L += [f"- {c['check']}." for c in out["checks"]]
    L += ["", "## Files", "",
          "- `results/rule_budgets/rule_budgets.json` (with `macros`, prefix `rb`), "
          "`results/rule_budgets/tables/risk_differences.csv` (every setting, K, item budget and comparison, with the "
          "pilots' spread), `results/rule_budgets/tables/dove_benchmarks.csv`",
          f"- `{fr['path']}`, `{fr['npz']}`",
          "- Code: `src/rule_budgets.py`, `scripts/run_rule_budgets.py`, `tests/test_rule_budgets.py`", ""]
    return L


def write(out, lines, tables):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    for k in out.get("macros", {}):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    with open(os.path.join(OUT, "rule_budgets.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=1)
    for fn, df in tables.items():
        df.to_csv(os.path.join(OUT, "tables", fn), index=False)
    txt = "\n".join(lines) + "\n"
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    settings = []
    for name in ("gsm8k30", "bbh", "math"):
        settings.append(d2_setting(name))
        print(f"{name}: discovery draws loaded ({time.time() - t0:.0f} s)", flush=True)
    settings.append(dove_block_setting())
    print(f"dove_mmlu: loaded ({time.time() - t0:.0f} s)", flush=True)
    tasks = dove_task_settings()
    if len(tasks) != 71:
        raise SystemExit(f"expected the 71 DOVE benchmarks of the reliability map, found {len(tasks)}")
    settings += tasks
    print(f"DOVE benchmarks: {len(tasks)} loaded ({time.time() - t0:.0f} s)", flush=True)
    checks = [check_discovery(s) for s in settings if s.get("reference")]
    print(f"discovery checks passed ({time.time() - t0:.0f} s)", flush=True)
    record, Z, rows = compute(settings, overwrite=a.overwrite_frozen)
    print(f"pilots frozen -> {rel(PRED_NPZ)}; validation scored ({time.time() - t0:.0f} s)", flush=True)
    checks += [check_validation(s, Z[s["setting"]]) for s in settings if s.get("reference")]
    sets = {s["setting"]: {"title": s["title"], "group": s["group"], "models": s["models"], "M": len(s["models"]),
                           "n_items": s["n_items"], "item_sampling": s["item_sampling"], "draws": s["draws"],
                           "strata": None if not s.get("strata") else len(s["strata"]),
                           "index": s.get("eci")} for s in settings}
    dv = [s for s in settings if s["group"] == "dove_tasks"]
    out = {"analysis": "own gain against capability rules by pilot budget",
           "methods": "scripts/run_rule_budgets.py",
           "sign": "capability rule minus own gain (paired loss difference, Step 3); positive when own gain predicts "
                   "better",
           "frozen": {"path": rel(PRED_JSON), "npz": rel(PRED_NPZ), "npz_sha256": record["npz_sha256"],
                      "content_sha256": record["content_sha256"], "created": record["created"],
                      "sha256": record["sha256"]},
           "design": {k: v for k, v in record["design"].items() if k not in ("settings", "cells")},
           "settings": {k: v for k, v in sets.items() if v["group"] != "dove_tasks"},
           "rows": [r for r in rows if r["group"] != "dove_tasks"],
           "dove_benchmarks": {"n_benchmarks": len(dv), "n_items_min": min(s["n_items"] for s in dv),
                               "n_items_max": max(s["n_items"] for s in dv), "summary": dove_summary(rows),
                               "rows": [r for r in rows if r["group"] == "dove_tasks"]},
           "checks": checks}
    out["macros"] = macros(out)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}

    def flat(r):
        x = {k: v for k, v in r.items() if k not in ("ci95", "per_draw")}
        x["ci95_lo"], x["ci95_hi"] = (r["ci95"] if r.get("ci95") else (None, None))
        return x
    write(out, summary(out), {"risk_differences.csv": pd.DataFrame([flat(r) for r in rows]).convert_dtypes(),
                              "dove_benchmarks.csv": pd.DataFrame([flat(r) for r in rows
                                                                   if r["group"] == "dove_tasks"]).convert_dtypes()})
    print(f"done in {out['runtime_seconds']} s -> {rel(os.path.join(OUT, 'rule_budgets.json'))}")


if __name__ == "__main__":
    main()
