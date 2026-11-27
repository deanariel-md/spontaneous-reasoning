r"""Published claims re-tested as patterns.

PowerShell (from the repository root), after scripts/score_content.py and scripts/run_cheng_panel.py:
    cmd /c ".venv\Scripts\python.exe scripts\run_patterns.py 2>&1"

Reads data/processed/content/<task>_content.parquet (strict and content scores of the three arms) on the 24 discovery
draws. Each pattern's statistic is computed per draw; its mean gets a leave-one-draw-out jackknife t interval, and
"repeats" is the share of draws with the mean's sign. Each statistic is recomputed with forced direct as the
comparison arm, on logits, on error reduction and on content scores. Outputs: results/patterns/patterns.json
(macros, prefix pat), summary.md, summary_clean.md, and results/figures/published_patterns.csv (one row per claim,
task and specification).
BIG-Bench Hard pooling: the Kojima and Wei rows pool accuracy over the target items,
so each of the 25 tasks (54 items each) weighs the same, as the specification rows of A do. The same rows with
the 21 task families weighted equally are reported as a sensitivity check (`bbh_family_equal`, macros `...FamEq...`).
"""
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
from src import instability as ins, intervals as iv, tasks_d2  # noqa: E402
from run_outcomes import TASKS, content_tasks  # noqa: E402
from run_index_checks import fmt, write  # noqa: E402

CONTENT = os.path.join(ROOT, "data", "processed", "content")
FIG = os.path.join(ROOT, "results", "figures", "published_patterns.csv")
ARMS = {"requested": "direct_answer", "forced": "forced_direct"}
COT = "zero_shot_cot"
SCORES = {"strict": "strict_correct", "content": "content_correct"}
SYMBOLIC = ("boolean_expressions", "geometric_shapes", "logical_deduction", "multistep_arithmetic_two", "navigate",
            "object_counting", "temporal_sequences", "tracking_shuffled_objects", "web_of_lies")
LANGUAGE = ("causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "hyperbaton",
            "movie_recommendation", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
            "salient_translation_error_detection", "snarks", "sports_understanding")
# Source of the split: Suzgun et al. (2023, arXiv 2210.09261v1), Table 3, marks the
# algorithmic tasks with a lambda; those count as symbolic, the unmarked ones as language. dyck_languages and
# word_sorting are also marked but are not in our set. Sprague et al. (arXiv 2409.12183v3) treat BIG-Bench Hard as
# one dataset ("Mixed datasets", their Table 1) and give no split of it.
SPLIT_SOURCE = ("Suzgun et al. (2023), Table 3: tasks marked as algorithmic count as symbolic, the rest as language; "
                "Sprague et al. (2024) do not split BIG-Bench Hard")
SPLIT_NOT_IN_SET = ("dyck_languages", "word_sorting")


def family_list(fams):
    """A family list for running text: underscores to spaces, comma separated with a final 'and'."""
    names = [f.replace("_", " ") for f in fams]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
VARIANTS = [("requested", "points", "strict"), ("forced", "points", "strict"), ("requested", "logit", "strict"),
            ("requested", "error_reduction", "strict"), ("requested", "points", "content")]
VTAG = {("requested", "points", "strict"): "Base", ("forced", "points", "strict"): "Forced",
        ("requested", "logit", "strict"): "Logit", ("requested", "error_reduction", "strict"): "ErrRed",
        ("requested", "points", "content"): "Content"}


def jack_stat(v):
    """Mean of per-draw values with the leave-one-draw-out jackknife t interval and the share of draws with its sign."""
    v = np.asarray(v, float)
    K = len(v)
    est = float(v.mean())
    loo = (v.sum() - v) / (K - 1)
    lo, hi = iv.jack_t(est, loo)
    sgn = np.sign(est)
    return {"estimate": est, "ci95": [float(lo), float(hi)], "repeats": float(np.mean(np.sign(v) == sgn)),
            "sign": "above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero"}


def load(task):
    from run_forced_direct import collection_inputs
    from src import bbh_d2
    inp, s, items, _ = collection_inputs(task)
    panel, families = list(inp["panel"]), list(inp["families"])
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    params = {e["repo"]: e["params_b"] for e in man["panel"]}
    reg = bbh_d2.load_registries()
    tab = pd.read_parquet(os.path.join(CONTENT, f"{task}_content.parquet"))
    tab["config_id"] = tab["config_id"].astype(str)
    units = [str(u) for u in s["discovery_config_ids"]]
    tab = tab[tab["model_key"].isin(panel) & tab["config_id"].isin(set(units))]
    if task == "bbh":
        tab["family"] = tab["item_id"].str.split("/").str[0].map(tasks_d2.bbh_family)
    else:
        tab["family"] = task
    n_items = tab.groupby("family")["item_id"].nunique().to_dict()
    arr = {}
    for sc, col in SCORES.items():
        g = tab.groupby(["arm", "family", "model_key", "config_id"])[col].mean().mul(100)
        for arm in (*ARMS.values(), COT):
            for fam in n_items:
                s_ = g.loc[(arm, fam)]
                arr[(sc, arm, fam)] = np.array([[s_.loc[(m, u)] for m in panel] for u in units], float)
    return {"panel": panel, "families": families, "units": units, "arr": arr, "n_items": n_items,
            "log2p": np.array([np.log2(params[reg[k]["hf_id"]]) for k in panel]),
            "params": [params[reg[k]["hf_id"]] for k in panel]}


def gain(x, fam, arm, scale, scoring):
    B = x["arr"][(scoring, ARMS[arm], fam)]
    P = x["arr"][(scoring, COT, fam)]
    return ins.scale_scores(B, P, scale, x["n_items"][fam])["y"]


def pooled_gain(x, arm, scale, scoring, weighting="task"):
    """BIG-Bench Hard: accuracy pooled over the task families, then the gain on the scale from the pooled accuracies.
    weighting "task": each family by its items, so each of the 25 tasks (54 items each) weighs the same, as in the
    specification rows of A; "family": each of the 21 families weighs the same (sensitivity)."""
    if weighting not in ("task", "family"):
        raise ValueError(f"weighting must be 'task' or 'family', not {weighting!r}")
    fams = list(x["n_items"])
    w = np.array([x["n_items"][f] if weighting == "task" else 1.0 for f in fams], float)
    w = w / w.sum()
    B = np.tensordot(w, np.stack([x["arr"][(scoring, ARMS[arm], f)] for f in fams]), axes=1)
    P = np.tensordot(w, np.stack([x["arr"][(scoring, COT, f)] for f in fams]), axes=1)
    return ins.scale_scores(B, P, scale, int(sum(x["n_items"].values())))["y"]


def draw_slopes(G, xcap):
    xc = xcap - xcap.mean()
    return (G - G.mean(1, keepdims=True)) @ xc / float(xc @ xc)


def kojima(x, task, weighting="task"):
    out = {}
    fams = sorted(set(x["families"]))
    for v in VARIANTS:
        G = gain(x, task, *v) if task != "bbh" else pooled_gain(x, *v, weighting=weighting)
        per = {}
        for f in fams:
            ix = [i for i, y in enumerate(x["families"]) if y == f]
            if len(ix) < 3:
                continue
            per[f] = {"M": len(ix), **jack_stat(draw_slopes(G[:, ix], x["log2p"][ix]))}
        out["/".join(v)] = {"families": per, "above": sum(p["sign"] == "above zero" for p in per.values()),
                            "below": sum(p["sign"] == "below zero" for p in per.values()),
                            "n": len(per), "repeats_mean": float(np.mean([p["repeats"] for p in per.values()])),
                            "positive_estimates": sum(p["estimate"] > 0 for p in per.values())}
    return out


def sprague(x, xg):
    out = {}
    sym = [f for f in SYMBOLIC if f in x["n_items"]]
    lang = [f for f in LANGUAGE if f in x["n_items"]]
    shared = [k for k in xg["panel"] if k in x["panel"]]
    ib, ig = [x["panel"].index(k) for k in shared], [xg["panel"].index(k) for k in shared]
    for v in VARIANTS:
        gs = np.mean([gain(x, f, *v).mean(1) for f in sym], axis=0)
        gl = np.mean([gain(x, f, *v).mean(1) for f in lang], axis=0)
        gl_sh = np.mean([gain(x, f, *v)[:, ib].mean(1) for f in lang], axis=0)
        gm = gain(xg, "gsm8k", *v)[:, ig].mean(1)
        per_model = (np.mean([gain(x, f, *v) for f in sym], axis=0) - np.mean([gain(x, f, *v) for f in lang], axis=0))
        out["/".join(v)] = {"symbolic_mean": float(gs.mean()), "language_mean": float(gl.mean()),
                            "symbolic_minus_language": jack_stat(gs - gl), "gsm8k_minus_language": jack_stat(gm - gl_sh),
                            "models_symbolic_above": int((per_model.mean(0) > 0).sum()), "M": len(x["panel"]),
                            "M_shared": len(shared)}
    out["families"] = {"symbolic": sym, "language": lang}
    return out


def wei(x, task, weighting="task"):
    fams = sorted(set(x["families"]))
    small = {}
    for f in fams:
        ix = [i for i, y in enumerate(x["families"]) if y == f]
        j = min(ix, key=lambda i: (x["params"][i], i))
        small[f] = j
    out = {"smallest": {f: x["panel"][j] for f, j in small.items()}}
    for v in VARIANTS:
        G = gain(x, task, *v) if task != "bbh" else pooled_gain(x, *v, weighting=weighting)
        per = {f: {"model": x["panel"][j], "params_b": x["params"][j], **jack_stat(G[:, j])} for f, j in small.items()}
        out["/".join(v)] = {"smallest": per, "negative": sum(p["sign"] == "below zero" for p in per.values()),
                            "positive": sum(p["sign"] == "above zero" for p in per.values()), "n": len(per),
                            "slope_log2_params": jack_stat(draw_slopes(G, x["log2p"]))}
    return out


def cheng_full():
    """The exemplar contrast at full scale (results/exemplar/exemplar.json from scripts/run_exemplar.py): per
    scale the slope of benefit on zero-shot accuracy over the 24 discovery draws with its set, the draws by sign, and
    the claim on the panel's models among the reproduction's BF16 models. None until the full run exists."""
    p = os.path.join(ROOT, "results", "exemplar", "exemplar.json")
    if not os.path.exists(p):
        return None
    ex = json.load(open(p, encoding="utf-8"))
    if ex.get("dry_run") or ex.get("partial"):
        return None
    c = ex["cheng"]["scorers"]["strict"]
    out = {"M": ex["M"], "K": ex["K_discovery"], "claim_models": len(ex["cheng"]["models"]),
           "claim_holds_in": c["claim_holds_in"], "of": c["n_configurations"], "scales": {}}
    for sc in ins.SCALES:
        d, ds = ex["scale"][sc]["discovery"], ex["draw_slopes"][sc]
        s = d["slope_set"]
        out["scales"][sc] = {"slope": d["slope"], "set_kind": s["kind"], "lo": s["a"], "hi": s["b"],
                             "set_sign": d.get("slope_set_sign"), "positive": ds["positive"],
                             "negative": ds["negative"], "repeats": ds["repeats"]}
    return out


def run():
    t0 = time.time()
    X = {t: load(t) for t in content_tasks()}
    print(f"arrays ({time.time() - t0:.0f} s)", flush=True)
    R = {"analysis": "published claims re-tested as patterns", "methods": "scripts/run_patterns.py",
         "created": time.strftime("%Y-%m-%d %H:%M"), "variants": ["/".join(v) for v in VARIANTS],
         "kojima": {t: kojima(X[t], t) for t in X}, "sprague": sprague(X["bbh"], X["gsm8k"]),
         "wei": {t: wei(X[t], t) for t in X},
         "bbh_pooling": "target items pooled: each of the 25 tasks weighs the same (54 items each), as the "
                        "specification rows of A",
         "bbh_family_equal": {"kojima": kojima(X["bbh"], "bbh", "family"), "wei": wei(X["bbh"], "bbh", "family"),
                              "note": "sensitivity: the 21 task families weighted equally"}}
    R["bbh_items_by_family"] = {f: int(n) for f, n in X["bbh"]["n_items"].items()}
    cp = json.load(open(os.path.join(ROOT, "results", "cheng_panel", "cheng_panel.json"), encoding="utf-8"))
    R["cheng"] = cp["panels"]["bf16"]
    R["cheng_full"] = cheng_full()
    rows = table(R)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    pd.DataFrame(rows).to_csv(FIG, index=False)
    R["table"] = rows
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write("patterns", R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/patterns, {os.path.relpath(FIG, ROOT)}", flush=True)


def table(R):
    rows = []
    for t in R["kojima"]:
        for v in R["variants"]:
            k = R["kojima"][t][v]
            rows.append({"claim": "kojima", "source": "Kojima et al. 2022", "pattern": "gain rises with scale within families",
                         "task": t, "specification": v, "statistic": "families with the slope on log2 parameters above zero",
                         "estimate": k["above"], "lo": None, "hi": None, "n": k["n"], "below": k["below"],
                         "positive_estimates": k["positive_estimates"], "repeats": k["repeats_mean"]})
            w = R["wei"][t][v]
            rows.append({"claim": "wei", "source": "Wei et al. 2022", "pattern": "chain of thought hurts small models",
                         "task": t, "specification": v, "statistic": "families whose smallest model's gain is below zero",
                         "estimate": w["negative"], "lo": None, "hi": None, "n": w["n"], "below": w["positive"],
                         "positive_estimates": None,
                         "repeats": float(np.mean([p["repeats"] for p in w["smallest"].values()]))})
    for v in R["variants"]:
        s = R["sprague"][v]
        for key, stat in (("symbolic_minus_language", "symbolic minus language families, BIG-Bench Hard"),
                          ("gsm8k_minus_language", "GSM8K minus BIG-Bench Hard language families")):
            x = s[key]
            rows.append({"claim": "sprague", "source": "Sprague et al. 2025", "pattern": "helps mainly math and symbolic tasks",
                         "task": "bbh" if key.startswith("symbolic") else "gsm8k+bbh", "specification": v,
                         "statistic": stat, "estimate": x["estimate"], "lo": x["ci95"][0], "hi": x["ci95"][1],
                         "n": s["M"] if key.startswith("symbolic") else s["M_shared"], "below": None,
                         "positive_estimates": s["models_symbolic_above"] if key.startswith("symbolic") else None,
                         "repeats": x["repeats"]})
    c = R["cheng"]["part_b"]
    for sc in ins.SCALES:
        v = c["scales"][sc]
        rows.append({"claim": "cheng", "source": "Cheng et al. 2025", "pattern": "exemplars help weak models, not strong",
                     "task": "gsm8k (13 BF16 models, 8 configurations)", "specification": f"exemplar/{sc}/cheng-fixed",
                     "statistic": "configurations where the claim holds; slope across configurations",
                     "estimate": v["cross_draw"]["slope"], "lo": v["cross_draw"]["slope_set"]["a"],
                     "hi": v["cross_draw"]["slope_set"]["b"], "n": c["of"], "below": v["negative"],
                     "positive_estimates": v["positive"], "repeats": None, "claim_holds_in": c["claim_holds_in"]})
    return rows + cheng_full_rows(R.get("cheng_full"))


def cheng_full_rows(h):
    rows = []
    if not h:
        return rows
    for sc in ins.SCALES:
        v = h["scales"][sc]
        bounded = v["set_kind"] == "bounded"
        rows.append({"claim": "cheng_full", "source": "Cheng et al. 2025", "pattern": "exemplars help weak models, not strong",
                     "task": f"gsm8k ({h['M']} models, {h['K']} discovery draws)", "specification": f"exemplar/{sc}/strict",
                     "statistic": "slope of benefit on zero-shot accuracy; draws by the sign of their slope",
                     "estimate": v["slope"], "lo": v["lo"] if bounded else None, "hi": v["hi"] if bounded else None,
                     "n": h["K"], "below": v["negative"], "positive_estimates": v["positive"], "repeats": v["repeats"],
                     "claim_holds_in": h["claim_holds_in"]})
    return rows


def cheng_full_macros(h):
    m = {}
    if not h:
        return m
    m["patChengFullM"], m["patChengFullK"] = str(h["M"]), str(h["K"])
    m["patChengFullClaim"], m["patChengFullClaimOf"] = str(h["claim_holds_in"]), str(h["of"])
    m["patChengFullClaimModels"] = str(h["claim_models"])
    for sc, st in (("points", "Points"), ("logit", "Logit"), ("error_reduction", "ErrRed")):
        v = h["scales"][sc]
        bounded = v["set_kind"] == "bounded"
        m[f"patChengFull{st}Slope"] = fmt(v["slope"], 3)
        m[f"patChengFull{st}SlopeLo"] = fmt(v["lo"], 3) if bounded else v["set_kind"]
        m[f"patChengFull{st}SlopeHi"] = fmt(v["hi"], 3) if bounded else v["set_kind"]
        m[f"patChengFull{st}Neg"], m[f"patChengFull{st}Pos"] = str(v["negative"]), str(v["positive"])
        m[f"patChengFull{st}Repeats"] = fmt(100 * v["repeats"], 0) if v["repeats"] is not None else "--"
    return m


def macros(R):
    m = {}
    for t in R["kojima"]:
        tt = TASKS[t]["tag"]
        for v, vt in ((("/".join(k)), VTAG[k]) for k in VARIANTS):
            k = R["kojima"][t][v]
            m[f"patKojima{tt}{vt}Above"], m[f"patKojima{tt}{vt}Below"] = str(k["above"]), str(k["below"])
            m[f"patKojima{tt}{vt}N"], m[f"patKojima{tt}{vt}PosEst"] = str(k["n"]), str(k["positive_estimates"])
            m[f"patKojima{tt}{vt}Repeats"] = fmt(100 * k["repeats_mean"], 0)
            w = R["wei"][t][v]
            m[f"patWei{tt}{vt}Negative"], m[f"patWei{tt}{vt}Positive"] = str(w["negative"]), str(w["positive"])
            m[f"patWei{tt}{vt}N"] = str(w["n"])
            sl = w["slope_log2_params"]
            m[f"patWei{tt}{vt}ParamSlope"] = fmt(sl["estimate"], 2)
            m[f"patWei{tt}{vt}ParamSlopeLo"], m[f"patWei{tt}{vt}ParamSlopeHi"] = fmt(sl["ci95"][0], 2), fmt(sl["ci95"][1], 2)
    fe = R.get("bbh_family_equal")
    if fe:
        tt = TASKS["bbh"]["tag"]
        for v, vt in ((("/".join(k)), VTAG[k]) for k in VARIANTS):
            k, w = fe["kojima"][v], fe["wei"][v]
            m[f"patKojima{tt}{vt}FamEqAbove"], m[f"patKojima{tt}{vt}FamEqBelow"] = str(k["above"]), str(k["below"])
            m[f"patKojima{tt}{vt}FamEqN"] = str(k["n"])
            m[f"patWei{tt}{vt}FamEqNegative"], m[f"patWei{tt}{vt}FamEqPositive"] = str(w["negative"]), str(w["positive"])
    for v, vt in ((("/".join(k)), VTAG[k]) for k in VARIANTS):
        s = R["sprague"][v]
        for key, kt in (("symbolic_minus_language", "SymMinusLang"), ("gsm8k_minus_language", "GsmMinusLang")):
            x = s[key]
            m[f"patSprague{vt}{kt}"] = fmt(x["estimate"], 1)
            m[f"patSprague{vt}{kt}Lo"], m[f"patSprague{vt}{kt}Hi"] = fmt(x["ci95"][0], 1), fmt(x["ci95"][1], 1)
            m[f"patSprague{vt}{kt}Repeats"] = fmt(100 * x["repeats"], 0)
        m[f"patSprague{vt}SymMean"], m[f"patSprague{vt}LangMean"] = fmt(s["symbolic_mean"], 1), fmt(s["language_mean"], 1)
        m[f"patSprague{vt}ModelsSymAbove"] = str(s["models_symbolic_above"])
    m["patSpragueSymFamilies"] = str(len(R["sprague"]["families"]["symbolic"]))
    m["patSpragueLangFamilies"] = str(len(R["sprague"]["families"]["language"]))
    m["patSpragueSymFamilyList"] = family_list(R["sprague"]["families"]["symbolic"])
    m["patSpragueLangFamilyList"] = family_list(R["sprague"]["families"]["language"])
    m["patSpragueSplitNotInSet"] = family_list(SPLIT_NOT_IN_SET)
    m["patSpragueSplitSource"] = SPLIT_SOURCE
    m.update(cheng_full_macros(R.get("cheng_full")))
    return m


def summary(R):
    L = ["# Published claims re-tested as patterns", "",
         "Discovery draws; each statistic's mean over the 24 draws "
         "with a leave-one-draw-out jackknife t interval (descriptive); repeats = share of draws with the mean's sign. "
         "Specifications: comparison arm / scale / scoring; `requested/points/strict` is the collection's.", ""]
    L += ["## Kojima et al.: gain rises with scale within families", "",
          "Slope of gain on log2 parameters within each family with at least three models: families above zero / "
          "below zero (of N), positive estimates, mean share of draws repeating the sign.", "",
          "| Task | Specification | Above | Below | N | Positive estimates | Repeats |", "|---|---|---:|---:|---:|---:|---:|"]
    for t in R["kojima"]:
        for v in R["variants"]:
            k = R["kojima"][t][v]
            L.append(f"| {t} | {v} | {k['above']} | {k['below']} | {k['n']} | {k['positive_estimates']} | "
                     f"{fmt(100 * k['repeats_mean'], 0)}% |")
        fam = R["kojima"][t][R["variants"][0]]["families"]
        L.append("")
        L.append(f"{t}, collection specification, by family: " + "; ".join(
            f"{f} (M = {p['M']}) {fmt(p['estimate'], 2)} [{fmt(p['ci95'][0], 2)}, {fmt(p['ci95'][1], 2)}]"
            for f, p in fam.items()) + ".")
        L.append("")
    L += ["On BIG-Bench Hard the gain pools the target items, so each of the 25 tasks (54 items each) weighs the same, "
          "as the specification rows of A do. The same rows with the 21 task families weighted equally are in "
          "the last section.", ""]
    sp = R["sprague"]
    L += ["## Sprague et al.: chain of thought helps mainly on math and symbolic tasks", "",
          f"Symbolic families ({len(sp['families']['symbolic'])}): {', '.join(sp['families']['symbolic'])}. Language "
          f"families ({len(sp['families']['language'])}): {', '.join(sp['families']['language'])}.", "",
          "Source of the split: Suzgun et al. (2023, arXiv 2210.09261v1), Table 3, marks the algorithmic tasks; "
          f"those count as symbolic and the unmarked ones as language. {' and '.join(SPLIT_NOT_IN_SET)} are also "
          "marked but are not in our set. Sprague et al. (arXiv 2409.12183v3) treat BIG-Bench Hard as one "
          "dataset (\"Mixed datasets\" in their Table 1) and do not split it, so the family comparison applies "
          "Suzgun et al.'s split to their claim.", "",
          "| Specification | Symbolic mean | Language mean | Symbolic − language | Repeats | Models with symbolic > language | "
          "GSM8K − language (shared models) | Repeats |", "|---|---:|---:|---|---:|---:|---|---:|"]
    for v in R["variants"]:
        s = sp[v]
        a, b = s["symbolic_minus_language"], s["gsm8k_minus_language"]
        L.append(f"| {v} | {fmt(s['symbolic_mean'], 1)} | {fmt(s['language_mean'], 1)} | {fmt(a['estimate'], 1)} "
                 f"[{fmt(a['ci95'][0], 1)}, {fmt(a['ci95'][1], 1)}] | {fmt(100 * a['repeats'], 0)}% | "
                 f"{s['models_symbolic_above']} of {s['M']} | {fmt(b['estimate'], 1)} [{fmt(b['ci95'][0], 1)}, "
                 f"{fmt(b['ci95'][1], 1)}] | {fmt(100 * b['repeats'], 0)}% |")
    L += ["", "## Wei et al.: chain of thought hurts small models", "",
          "Gain of each family's smallest model; families where it lies below / above zero; slope of gain on log2 "
          "parameters across the full panel.", "",
          "| Task | Specification | Below zero | Above zero | Families | Slope on log2 parameters |", "|---|---|---:|---:|---:|---|"]
    for t in R["kojima"]:
        for v in R["variants"]:
            w = R["wei"][t][v]
            sl = w["slope_log2_params"]
            L.append(f"| {t} | {v} | {w['negative']} | {w['positive']} | {w['n']} | {fmt(sl['estimate'], 2)} "
                     f"[{fmt(sl['ci95'][0], 2)}, {fmt(sl['ci95'][1], 2)}] |")
        w = R["wei"][t][R["variants"][0]]
        L.append("")
        L.append(f"{t}, collection specification: " + "; ".join(
            f"{f} {p['model']} ({fmt(p['params_b'], 1)}B) {fmt(p['estimate'], 1)} [{fmt(p['ci95'][0], 1)}, "
            f"{fmt(p['ci95'][1], 1)}]" for f, p in w["smallest"].items()) + ".")
        L.append("")
    fe = R.get("bbh_family_equal")
    if fe:
        L += ["## BIG-Bench Hard with the 21 task families weighted equally (sensitivity)", "",
              "Kojima et al. and Wei et al. rows as above, with each task family one equal share of the pooled "
              "accuracy (logical_deduction and tracking_shuffled_objects count once each, not three times).", "",
              "| Specification | Kojima above | Kojima below | Families | Wei below zero | Wei above zero |",
              "|---|---:|---:|---:|---:|---:|"]
        for v in R["variants"]:
            k, w = fe["kojima"][v], fe["wei"][v]
            L.append(f"| {v} | {k['above']} | {k['below']} | {k['n']} | {w['negative']} | {w['positive']} |")
        fam = fe["kojima"][R["variants"][0]]["families"]
        L += ["", "Kojima slopes, collection specification, by family: " + "; ".join(
            f"{f} (M = {p['M']}) {fmt(p['estimate'], 2)} [{fmt(p['ci95'][0], 2)}, {fmt(p['ci95'][1], 2)}]"
            for f, p in fam.items()) + ".", ""]
    c = R["cheng"]["part_b"]
    L += ["## Cheng et al.: exemplars help weak models and not strong ones", "",
          f"On the 13 BF16 models: the claim holds in {c['claim_holds_in']} of {c['of']} configurations; "
          "slope of benefit on zero-shot accuracy across the configurations: " + "; ".join(
              f"{sc} {fmt(c['scales'][sc]['cross_draw']['slope'], 3)} [{fmt(c['scales'][sc]['cross_draw']['slope_set']['a'], 3)}, "
              f"{fmt(c['scales'][sc]['cross_draw']['slope_set']['b'], 3)}] ({c['scales'][sc]['negative']} of 8 negative)"
              for sc in ins.SCALES) + f"; R of benefit {fmt(c['map_point']['R'], 2)}.", ""]
    h = R.get("cheng_full")
    if h:
        def st(v):
            return (f"{fmt(v['slope'], 3)} [{fmt(v['lo'], 3)}, {fmt(v['hi'], 3)}]" if v["set_kind"] == "bounded"
                    else f"{fmt(v['slope'], 3)} (set {v['set_kind']})")
        L += [f"At full scale (GSM8K, {h['M']} models, {h['K']} discovery draws, strict scorer): the claim holds "
              f"in {h['claim_holds_in']} of {h['of']} configurations on the {h['claim_models']} panel models among the "
              "13 BF16 models; slope of benefit on zero-shot accuracy across the panel: " + "; ".join(
                  f"{sc} {st(h['scales'][sc])} ({h['scales'][sc]['negative']} of {h['K']} draws negative)"
                  for sc in ins.SCALES) + ".", ""]
    return L


if __name__ == "__main__":
    run()
