r"""Every capability and cross-task number on each of the three outcomes.

Run from the repository root:
    python scripts/run_outcomes.py

Settings: GSM8K-29 and BIG-Bench Hard-28, the three arms of results/forced_direct (requested direct, forced direct,
chain of thought), strict scoring. Outcomes, each an intervention arm minus a baseline arm, named in every macro:
    CotMinusReq     chain of thought - requested direct   (baseline: requested direct; the collection's gain)
    CotMinusForced  chain of thought - forced direct      (baseline: forced direct)
    ReqMinusForced  requested direct - forced direct      (baseline: forced direct)

Order: discovery draws of the three arms; every discovery-only block; the capability rules, constant
and own gain of each outcome saved to manifests/predictions_outcomes.json; only then the validation draws, scored by
cross-product risk. Outputs: results/outcomes/outcomes.json (macros, prefix oc), summary.md, summary_clean.md and
tables/*.csv.

--scoring lenient | content runs the same analysis on the lenient reading of the scored
tables or on the content scores of scripts/score_content.py; outputs go to results/outcomes_<scoring>/, the rules to
manifests/predictions_outcomes_<scoring>.json, and the macros take the prefix ocLen or ocCon.

--tasks math runs MATH-500 alone, since the GSM8K and BIG-Bench Hard rules
are already saved: results/outcomes[_<scoring>]_math/ and manifests/predictions_outcomes[_<scoring>]_math.json, macros with
the task tag Math (ocMath, ocLenMath, ocConMath), no cross-task block.
"""
import argparse
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
from src import capability_checks as cc, dataset2 as d2, instability as ins, reliability_map as rm  # noqa: E402
from src import transfer_d2 as td  # noqa: E402
from src.sampling import sha256_json  # noqa: E402
from run_index_checks import (RULE_WORDS, fmt, jsonable, rel, set_txt, slope_block, weighted_slope_block,  # noqa: E402
                               write)

CONFIG = os.path.join(ROOT, "config", "instability.yaml")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
PRED = os.path.join(ROOT, "manifests", "predictions_outcomes.json")
FD = os.path.join(ROOT, "results", "forced_direct")
OUT = os.path.join(ROOT, "results", "outcomes")
ARMS = ("requested", "forced", "cot")  # the order of run_forced_direct.three_arm_arrays
OUTCOMES = {"CotMinusReq": ("requested", "cot", "chain of thought − requested direct"),
            "CotMinusForced": ("forced", "cot", "chain of thought − forced direct"),
            "ReqMinusForced": ("forced", "requested", "requested direct − forced direct")}
TASKS = {"gsm8k": {"tag": "Gsm", "name": "GSM8K-29", "collection": "gsm8k30"},
         "bbh": {"tag": "Bbh", "name": "BIG-Bench Hard", "collection": "bbh"},
         "math": {"tag": "Math", "name": "MATH-500, numeric items", "collection": "math"}}
DEFAULT_TASKS = ("gsm8k", "bbh")
SC = {"points": "", "logit": "Logit", "error_reduction": "ErrRed"}
CLEAN_MIN = 95.0
FAMILY_MIN = 3
SCORINGS = {"strict": ("strict_correct", "oc", ""), "lenient": ("lenient_correct", "ocLen", "_lenient"),
            "content": ("content_correct", "ocCon", "_content")}
CONTENT = os.path.join(ROOT, "data", "processed", "content")
KEY = ["model_key", "arm", "config_id", "item_id"]
PREFIX = "oc"


def paths(scoring, suffix=""):
    s = SCORINGS[scoring][2] + suffix
    return os.path.join(ROOT, "manifests", f"predictions_outcomes{s}.json"), os.path.join(ROOT, "results",
                                                                                        f"outcomes{s}")


def content_tasks():
    """The tasks of the analyses that read every task's content scores (scripts/run_content.py, run_specification.py,
    run_patterns.py): GSM8K-29 and BIG-Bench Hard, and MATH-500 once scripts/score_content.py --tasks math has written
    its table."""
    return [t for t in TASKS if t in DEFAULT_TASKS or os.path.exists(os.path.join(CONTENT, f"{t}_content.parquet"))]


def with_content(load, task):
    """A loader of scored rows with the content score of scripts/score_content.py beside them; refused unless every
    row has one and the strict scores agree."""
    tab = pd.read_parquet(os.path.join(CONTENT, f"{task}_content.parquet"),
                          columns=KEY + ["strict_correct", "content_correct"])

    def f(ids):
        df = load(ids)
        c = tab[tab["config_id"].isin(set(map(str, ids)))]
        j = df.merge(c.rename(columns={"strict_correct": "_strict"}), on=KEY, how="left", validate="one_to_one")
        if j["content_correct"].isna().any() or (j["_strict"] != j["strict_correct"]).any():
            raise SystemExit(f"{task}: the content table does not match the scored rows")
        return j.drop(columns="_strict")
    return f


def pair(D, outcome):
    """(baseline, intervention) arrays of an outcome from the three-arm tuple."""
    b, p, _ = OUTCOMES[outcome]
    return D[ARMS.index(b)], D[ARMS.index(p)]


def family_means(X, families, order):
    return np.column_stack([X[:, [i for i, f in enumerate(families) if f == fam]].mean(1) for fam in order])


def point_slopes(B, P, caps, method, alpha):
    """Points-scale slopes of P - B on each capability array (draws, models)."""
    out = {}
    for name, X in caps.items():
        m = ins.scale_moments(X, P - B, method, alpha)
        out[name] = {"slope": m["slope"], "slope_set": m["slope_set"], "set_sign": m["slope_set_sign"]}
    return out


def outcome_block(B, P, n, e, families, logp, clean, method, alpha):
    G = P - B
    K = B.shape[0]
    pt = rm.setting_point(B, P)
    r = {"slopes": slope_block(B, P, n, e, method, alpha),
         "share": {"index": cc.explained_share(G, cc.tile(e, K)), "baseline": cc.explained_share(G, B)},
         "V_delta": cc.vdelta_draws(G), "mean": float(G.mean()),
         "map_point": {q: {k: pt[q][k] for k in ("R", "signal", "noise", "C", "C_ref", "departure")}
                       for q in ("benefit", "accuracy")}}
    order = sorted(set(families))
    ef = np.array([np.mean([e[i] for i, f in enumerate(families) if f == fam]) for fam in order])
    # each family one point, and the equal-family-weighted regression
    r["family_means"] = {"families": order, "slopes": slope_block(family_means(B, families, order),
                                                                 family_means(P, families, order), n, ef, method, alpha)}
    r["family_weighted"] = {"families": order, "slopes": weighted_slope_block(B, P, n, e, ins.family_weights(families),
                                                                              method, alpha)}
    c = [i for i in range(len(e)) if clean[i]]
    ec = [e[i] for i in c]
    r["clean"] = {"M": len(c), "slopes": slope_block(B[:, c], P[:, c], n, ec, method, alpha),
                  "share": {"index": cc.explained_share(G[:, c], cc.tile(ec, K)),
                            "baseline": cc.explained_share(G[:, c], B[:, c])}}
    fam = {}
    for f in order:
        ix = [i for i, x in enumerate(families) if x == f]
        if len(ix) < FAMILY_MIN:
            continue
        caps = {"baseline": B[:, ix], "index": cc.tile([e[i] for i in ix], K),
                "log2_params": cc.tile([logp[i] for i in ix], K)}
        fam[f] = {"M": len(ix), **point_slopes(B[:, ix], P[:, ix], caps, method, alpha)}
    r["within_family"] = fam
    return r


def rules_block(B, P, e):
    ch = cc.choose_rule(B, P, e)
    preds = {n: cc.fit_rule(B, P, e, dg, ix)["prediction"].tolist() for n, dg, ix in cc.CANDIDATES}
    preds["own_gain"] = (P - B).mean(0).tolist()
    return {"chosen": ch["chosen"], "scores": {k: v["score"] for k, v in ch["scores"].items()}, "predictions": preds}


def check_frozen(task, outcome, rules, panel):
    """The constant and own-gain rules of CotMinusReq and CotMinusForced equal the frozen files of the main and
    forced-direct analyses; for CotMinusReq the capability rules equal predictions_capability_rule.json."""
    src = {"CotMinusReq": f"predictions_{TASKS[task]['collection']}.json",
           "CotMinusForced": f"predictions_forced_{task}.json"}.get(outcome)
    if src is None:
        return None
    mf = json.load(open(os.path.join(ROOT, "manifests", src), encoding="utf-8"))
    if mf["models"] != panel:
        raise SystemExit(f"manifests/{src} has another panel")
    dev = {}
    for k in ("own_gain", "constant"):
        ref = np.asarray(mf["budgets"]["24"][k]["prediction"], float)
        dev[k] = float(np.max(np.abs(ref - np.asarray(rules["predictions"][k], float))))
    if dev["own_gain"] > 1e-9:
        raise SystemExit(f"{task} {outcome}: own gain differs from manifests/{src}")
    out = {"file": f"manifests/{src}", "max_abs_dev": dev}
    cr = json.load(open(os.path.join(ROOT, "manifests", "predictions_capability_rule.json"), encoding="utf-8"))
    if outcome == "CotMinusReq" and TASKS[task]["collection"] not in cr["predictions"]:
        out["capability_rule"] = "no capability rule was frozen for this task before (MATH-500)"
    elif outcome == "CotMinusReq":
        fr = cr["predictions"][TASKS[task]["collection"]]
        if fr["chosen"] != rules["chosen"]:
            raise SystemExit(f"{task}: the chosen rule differs from predictions_capability_rule.json")
        dc = max(float(np.max(np.abs(np.asarray(v, float) - np.asarray(rules["predictions"][k], float))))
                 for k, v in fr["rules"].items())
        if dc > 1e-9:
            raise SystemExit(f"{task}: the capability rules differ from predictions_capability_rule.json")
        out["capability_rule_max_abs_dev"] = dc
    return out


def cross_task(S, scoring):
    gp, bp = S["gsm8k"]["panel"], S["bbh"]["panel"]
    shared = [k for k in gp if k in bp]
    ig, ib = [gp.index(k) for k in shared], [bp.index(k) for k in shared]
    e_sh = [S["gsm8k"]["eci"][i] for i in ig]
    if e_sh != [S["bbh"]["eci"][i] for i in ib]:
        raise SystemExit("the index differs between the two panels")
    cross = {"models": shared, "M": len(shared), "outcomes": {}}
    for o in OUTCOMES:
        Bg, Pg = pair(S["gsm8k"]["D"], o)
        Bb, Pb = pair(S["bbh"]["D"], o)
        Ga, Gb = (Pg - Bg)[:, ig], (Pb - Bb)[:, ib]
        cross["outcomes"][o] = {"raw": td.cross_task_correlation(Ga, Gb), "partial": cc.partial_correlation(Ga, Gb, e_sh),
                                "with_index": {"gsm8k": cc.correlation_with_index(Ga, e_sh),
                                               "bbh": cc.correlation_with_index(Gb, e_sh)}}
    tr = json.load(open(os.path.join(ROOT, "results", "transfer_tasks", "transfer_tasks.json"), encoding="utf-8"))
    dev = abs(cross["outcomes"]["CotMinusReq"]["raw"]["pearson"]["estimate"]
              - tr["correlation"]["discovery_24"]["pearson"]["estimate"])
    if dev > 1e-9 and scoring == "strict":
        raise SystemExit("CotMinusReq: the cross-task Pearson correlation differs from results/transfer_tasks")
    return cross


def run(a):
    global PRED, OUT, PREFIX
    t0 = time.time()
    scoring = a.scoring
    value, PREFIX = SCORINGS[scoring][:2]
    tasks = list(dict.fromkeys(getattr(a, "tasks", None) or DEFAULT_TASKS))
    if tuple(tasks) != DEFAULT_TASKS and tasks != ["math"]:
        raise SystemExit("--tasks is gsm8k bbh (the default) or math alone")
    PRED, OUT = paths(scoring, "" if tuple(tasks) == DEFAULT_TASKS else "_" + "_".join(tasks))
    if getattr(a, "frozen_dir", None):
        # rules refitted on corrected discovery draws are written to and read from this folder;
        # manifests/ keeps the earlier file (src/forced_stop.py)
        os.makedirs(a.frozen_dir, exist_ok=True)
        PRED = os.path.join(os.path.abspath(a.frozen_dir), os.path.basename(PRED))
    name = os.path.basename(OUT)
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays as _arrays

    def three_arm_arrays(col_load, fd_load, units, panel, items):
        return _arrays(col_load, fd_load, units, panel, items, value=value)
    from run_transfer_tasks import eci_for
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    params = {e["repo"]: e["params_b"] for e in man["panel"]}
    from src import bbh_d2
    reg = bbh_d2.load_registries()

    S = {}
    for task in tasks:
        inp, s, items, items_info = collection_inputs(task)
        panel = list(inp["panel"])
        fd_load, fd_prov, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
        if scoring == "content":
            inp = {**inp, "load": with_content(inp["load"], task)}
            fd_load = with_content(fd_load, task)
        pm = pd.read_csv(os.path.join(FD, "tables", f"per_model_{task}.csv")).set_index("model_key")
        clean = [bool(pm.loc[k, "closed_and_readable_pct"] >= CLEAN_MIN) for k in panel]
        D = three_arm_arrays(inp["load"], fd_load, list(s["discovery_config_ids"]), panel, items)
        S[task] = {"inp": inp, "sampler": s, "items": items, "panel": panel, "fd_load": fd_load, "D": D,
                   "eci": eci_for(panel), "families": list(inp["families"]), "clean": clean,
                   "log2_params": [float(np.log2(params[reg[k]["hf_id"]])) for k in panel]}
        print(f"{task}: discovery arrays, {sum(clean)} of {len(panel)} clean ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ discovery-only blocks
    R = {"analysis": "capability and cross-task numbers per outcome", "methods": "three outcomes on the same discovery and validation draws",
         "scoring": scoring, "score_column": value,
         "created": time.strftime("%Y-%m-%d %H:%M"), "slope_method": method, "clean_rule": f">= {CLEAN_MIN}% of forced "
         "outputs with a closed, readable box (results/forced_direct/tables/per_model_<task>.csv)",
         "outcomes": {k: v[2] for k, v in OUTCOMES.items()}, "tasks": {}}
    rules = {}
    for task, x in S.items():
        n = len(x["items"])
        T = {"models": x["panel"], "M": len(x["panel"]), "clean_models": [k for k, c in zip(x["panel"], x["clean"]) if c],
             "families": x["families"], "eci": x["eci"], "log2_params": x["log2_params"], "outcomes": {}}
        rules[task] = {}
        for o in OUTCOMES:
            B, P = pair(x["D"], o)
            T["outcomes"][o] = outcome_block(B, P, n, x["eci"], x["families"], x["log2_params"], x["clean"], method,
                                             alpha)
            rules[task][o] = rules_block(B, P, x["eci"])
            T["outcomes"][o]["frozen_check"] = (check_frozen(task, o, rules[task][o], x["panel"])
                                                if scoring == "strict" else None)
        R["tasks"][task] = T
        print(f"{task}: discovery blocks ({time.time() - t0:.0f} s)", flush=True)

    if "gsm8k" in S and "bbh" in S:
        R["cross_task"] = cross_task(S, scoring)

    # ------------------------------------------------------------ freeze, then validation
    preds = {t: {o: {"chosen": r["chosen"], "rules": r["predictions"]} for o, r in rs.items()} for t, rs in rules.items()}
    frozen = {"label": name, "scoring": scoring, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp",
              "models": {t: S[t]["panel"] for t in S}, "samplers": {t: S[t]["sampler"]["sha256"] for t in S},
              "outcomes": {k: v[2] for k, v in OUTCOMES.items()},
              "note": "fitted on the 24 discovery draws only; written before this run builds "
                      "the validation draws", "predictions": preds}
    frozen["sha256"] = sha256_json(json.loads(json.dumps(jsonable(preds))))
    frozen = d2.write_frozen(jsonable(frozen), PRED, overwrite=a.overwrite_frozen)
    print(f"rules frozen -> {rel(PRED)} ({time.time() - t0:.0f} s)", flush=True)

    for task, x in S.items():
        V = three_arm_arrays(x["inp"]["load"], x["fd_load"], list(x["sampler"]["validation_config_ids"]), x["panel"],
                             x["items"])
        for o in OUTCOMES:
            Bv, Pv = pair(V, o)
            Zv = Pv - Bv
            fr = frozen["predictions"][task][o]
            Pr = {k: np.asarray(v, float) for k, v in fr["rules"].items()}
            ch = fr["chosen"]
            comps = [(cc.FULL_RULE, "own_gain"), ("constant", cc.FULL_RULE), ("constant", "own_gain"),
                     ("baseline_linear", cc.FULL_RULE)]
            if ch not in (cc.FULL_RULE, "constant", "own_gain"):
                comps.append((ch, "own_gain"))
            R["tasks"][task]["outcomes"][o]["validation"] = {
                "K_validation": int(Zv.shape[0]), "chosen": ch,
                "risk_pp2": {k: float(cc.cross_risk(Zv, v)) for k, v in Pr.items()},
                "differences": [{"f": f, "g": g, **cc.paired_difference(Zv, Pr[f], Pr[g])} for f, g in comps]}
    print(f"validation scored ({time.time() - t0:.0f} s)", flush=True)
    R["frozen"] = {"path": rel(PRED), "sha256": frozen["sha256"], "created": frozen["created"]}
    R["macros"] = macros(R)
    tables(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write(name, R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/{name}", flush=True)


# ============================================================================ macros, tables, summary


def _set(m, stem, est, s, nd=3):
    m[stem] = fmt(est, nd)
    if s["kind"] == "bounded":
        m[stem + "Lo"], m[stem + "Hi"] = fmt(s["a"], nd), fmt(s["b"], nd)
    else:
        m[stem + "Lo"] = m[stem + "Hi"] = s["kind"]


def _ci(m, stem, d, nd=2):
    m[stem] = fmt(d["estimate"], nd)
    if d.get("ci95"):
        m[stem + "Lo"], m[stem + "Hi"] = fmt(d["ci95"][0], nd), fmt(d["ci95"][1], nd)


def _slopes(m, stem, sl):
    for sc, st in SC.items():
        for cap, ct in (("index", "Index"), ("baseline", "Base")):
            x = sl["scales"][sc][cap]
            _set(m, f"{stem}{ct}Slope{st}", x["slope"], x["slope_set"])
    x = sl["scales"]["points"]["index"]
    _set(m, f"{stem}IndexSlopePerSd", x["slope_per_sd"], x["slope_per_sd_set"], 2)


def macros(R):
    m = {}
    for task, T in R["tasks"].items():
        t = TASKS[task]["tag"]
        m[f"{PREFIX}{t}M"], m[f"{PREFIX}{t}CleanM"] = str(T["M"]), str(len(T["clean_models"]))
        for o, r in T["outcomes"].items():
            s = f"{PREFIX}{t}{o}"
            m[s + "Mean"] = fmt(r["mean"], 1)
            for q, qt in (("benefit", "Benefit"), ("accuracy", "Accuracy")):
                mp = r["map_point"][q]
                m[f"{s}MapR{qt}"], m[f"{s}MapC{qt}"] = fmt(mp["R"], 3), fmt(mp["C"], 3)
                m[f"{s}MapCref{qt}"] = fmt(mp["C_ref"], 3)
            _slopes(m, s, r["slopes"])
            for cap, ct in (("index", "Index"), ("baseline", "Base")):
                _ci(m, f"{s}Share{ct}", r["share"][cap])
                _ci(m, f"{s}CleanShare{ct}", r["clean"]["share"][cap])
            _ci(m, s + "Vdelta", r["V_delta"], 0)
            _slopes(m, s + "FamMeans", r["family_means"]["slopes"])
            _slopes(m, s + "FamWeighted", r["family_weighted"]["slopes"])
            _slopes(m, s + "Clean", r["clean"]["slopes"])
            for f, fr in r["within_family"].items():
                m[f"{s}Fam{f}M"] = str(fr["M"])
                for cap, ct in (("baseline", "Base"), ("index", "Index"), ("log2_params", "Param")):
                    _set(m, f"{s}Fam{f}{ct}Slope", fr[cap]["slope"], fr[cap]["slope_set"])
            v = r["validation"]
            rk = v["risk_pp2"]
            m[s + "Chosen"] = RULE_WORDS.get(v["chosen"], v["chosen"])
            for k, kt in (("constant", "Const"), ("own_gain", "Own"), (cc.FULL_RULE, "QuadIndex"),
                          ("baseline_linear", "Linear"), (v["chosen"], "ChosenRule")):
                m[f"{s}Risk{kt}"] = fmt(rk[k], 1)
            if rk["constant"] > 0:
                m[s + "QuadIndexShareRemoved"] = fmt(100 * (rk["constant"] - rk[cc.FULL_RULE]) / rk["constant"], 0)
                m[s + "LinearShareRemoved"] = fmt(100 * (rk["constant"] - rk["baseline_linear"]) / rk["constant"], 0)
            names = {cc.FULL_RULE: "QuadIndex", "own_gain": "Own", "constant": "Const", "baseline_linear": "Linear"}
            names.setdefault(v["chosen"], "ChosenRule")
            for d in v["differences"]:
                _ci(m, f"{s}{names[d['f']]}Vs{names[d['g']]}", d, 1)
    if "cross_task" not in R:
        return m
    for o, c in R["cross_task"]["outcomes"].items():
        s = f"{PREFIX}{o}"
        for k, kt in (("pearson", "CorrPearson"), ("spearman", "CorrSpearman"), ("latent", "CorrLatent")):
            _ci(m, s + kt, c["raw"][k])
        for k, kt in (("pearson", "PartialPearson"), ("latent", "PartialLatent")):
            _ci(m, s + kt, c["partial"][k])
        for task, t in (("gsm8k", "Gsm"), ("bbh", "Bbh")):
            _ci(m, f"{s}CorrIndex{t}", c["with_index"][task])
    m[f"{PREFIX}SharedM"] = str(R["cross_task"]["M"])
    return m


def tables(R):
    d = os.path.join(OUT, "tables")
    os.makedirs(d, exist_ok=True)
    rows, fam = [], []
    for task, T in R["tasks"].items():
        for o, r in T["outcomes"].items():
            for panel, sl, sh, M in (("full", r["slopes"], r["share"], T["M"]),
                                     ("clean", r["clean"]["slopes"], r["clean"]["share"], len(T["clean_models"])),
                                     ("family_means", r["family_means"]["slopes"], None,
                                      len(r["family_means"]["families"])),
                                     ("family_weighted", r["family_weighted"]["slopes"], None, T["M"])):
                for sc in ins.SCALES:
                    for cap in ("index", "baseline"):
                        x = sl["scales"][sc][cap]
                        rows.append({"task": task, "outcome": o, "panel": panel, "M": M, "scale": sc, "capability": cap,
                                     "slope": x["slope"], "set_kind": x["slope_set"]["kind"], "lo": x["slope_set"]["a"],
                                     "hi": x["slope_set"]["b"],
                                     "share": None if sh is None or sc != "points" else sh[cap]["estimate"]})
            for f, fr in r["within_family"].items():
                for cap in ("baseline", "index", "log2_params"):
                    x = fr[cap]
                    fam.append({"task": task, "outcome": o, "family": f, "M": fr["M"], "capability": cap,
                                "slope": x["slope"], "set_kind": x["slope_set"]["kind"], "lo": x["slope_set"]["a"],
                                "hi": x["slope_set"]["b"]})
    pd.DataFrame(rows).to_csv(os.path.join(d, "slopes.csv"), index=False)
    pd.DataFrame(fam).to_csv(os.path.join(d, "within_family.csv"), index=False)


def summary(R):
    sc_words = {"strict": "strict scoring", "lenient": "lenient scoring (the last answer token of the output)",
                "content": "content scores (scripts/score_content.py)"}[R["scoring"]]
    L = ["# Capability and cross-task numbers per outcome" + ("" if R["scoring"] == "strict"
                                                             else f", {R['scoring']} scoring"), "",
         f"Reading: {'strict' if R['scoring'] == 'strict' else 'lenient or content'}. Discovery draws, {sc_words}, Appendix A moments; "
         "slopes in pp of the outcome per pp of its baseline (or per index point), 95% sets from the "
         f"{R['slope_method']} method; shares and correlations with leave-one-draw-out jackknife t intervals "
         "(descriptive). Rules saved to " + f"`{R['frozen']['path']}` ({R['frozen']['created']}) before the "
         "validation draws were built.", "",
         "Outcomes: " + "; ".join(f"`{k}` = {v}" for k, v in R["outcomes"].items()) + ".", ""]
    for task, T in R["tasks"].items():
        L += [f"## {TASKS[task]['name']} ({T['M']} models; clean panel {len(T['clean_models'])})", "",
              "| Outcome | Mean | V_delta | Index slope | Per index SD | Share (index) | Baseline slope | Share (baseline) |",
              "|---|---:|---|---|---|---|---|---|"]
        for o, r in T["outcomes"].items():
            p = r["slopes"]["scales"]["points"]
            sh = r["share"]
            L.append(f"| {o} | {fmt(r['mean'], 1)} | {fmt(r['V_delta']['estimate'], 0)} "
                     f"[{fmt(r['V_delta']['ci95'][0], 0)}, {fmt(r['V_delta']['ci95'][1], 0)}] | "
                     f"{fmt(p['index']['slope'], 3)} {set_txt(p['index']['slope_set'])} | "
                     f"{fmt(p['index']['slope_per_sd'], 2)} | {fmt(sh['index']['estimate'], 2)} | "
                     f"{fmt(p['baseline']['slope'], 3)} {set_txt(p['baseline']['slope_set'])} | "
                     f"{fmt(sh['baseline']['estimate'], 2)} |")
        L += ["", "Map points on the discovery draws (R and C of the benefit and accuracy orderings; C_ref = 1/2 + "
              "arcsin(R)/π):", "", "| Outcome | R benefit | C benefit | C_ref | R accuracy | C accuracy | C_ref |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for o, r in T["outcomes"].items():
            b, a_ = r["map_point"]["benefit"], r["map_point"]["accuracy"]
            L.append(f"| {o} | {fmt(b['R'], 3)} | {fmt(b['C'], 3)} | {fmt(b['C_ref'], 3)} | {fmt(a_['R'], 3)} | "
                     f"{fmt(a_['C'], 3)} | {fmt(a_['C_ref'], 3)} |")
        L += ["", "Slopes on three scales (index / baseline): full panel, clean panel, family means (each family one "
              "point) and family-weighted (every model, each family the same total weight):", "",
              "| Outcome | Panel | Points | Logits | Error reduction |", "|---|---|---|---|---|"]
        for o, r in T["outcomes"].items():
            for panel, sl in (("full", r["slopes"]), (f"clean ({len(T['clean_models'])})", r["clean"]["slopes"]),
                              (f"family means ({len(r['family_means']['families'])})", r["family_means"]["slopes"]),
                              ("family-weighted", r["family_weighted"]["slopes"])):
                cells = []
                for sc in ins.SCALES:
                    x = sl["scales"][sc]
                    cells.append(f"{fmt(x['index']['slope'], 3)} / {fmt(x['baseline']['slope'], 3)} "
                                 f"{set_txt(x['baseline']['slope_set'])}")
                L.append(f"| {o} | {panel} | " + " | ".join(cells) + " |")
        L += ["", "Within families (points; slope on baseline, on the index, on log2 parameters):", "",
              "| Outcome | Family | M | Baseline | Index | log2 parameters |", "|---|---|---:|---|---|---|"]
        for o, r in T["outcomes"].items():
            for f, fr in r["within_family"].items():
                L.append(f"| {o} | {f} | {fr['M']} | " + " | ".join(
                    f"{fmt(fr[c]['slope'], 3)} {set_txt(fr[c]['slope_set'])}" for c in ("baseline", "index",
                                                                                         "log2_params")) + " |")
        L += ["", "Validation risks (pp², cross-product) and paired differences (positive favours the second rule):",
              "", "| Outcome | Chosen rule | Constant | Linear in baseline | Quadratic + index | Own gain | "
              "Quadratic + index vs own gain | Constant vs quadratic + index |", "|---|---|---:|---:|---:|---:|---|---|"]
        for o, r in T["outcomes"].items():
            v = r["validation"]
            rk = v["risk_pp2"]
            dd = {(d["f"], d["g"]): d for d in v["differences"]}

            def dtxt(f, g):
                d = dd[(f, g)]
                return f"{fmt(d['estimate'], 1)} " + (f"[{fmt(d['ci95'][0], 1)}, {fmt(d['ci95'][1], 1)}]"
                                                      if d.get("ci95") else "")
            L.append(f"| {o} | {RULE_WORDS.get(v['chosen'], v['chosen'])} | {fmt(rk['constant'], 1)} | "
                     f"{fmt(rk['baseline_linear'], 1)} | {fmt(rk[cc.FULL_RULE], 1)} | {fmt(rk['own_gain'], 1)} | "
                     f"{dtxt(cc.FULL_RULE, 'own_gain')} | {dtxt('constant', cc.FULL_RULE)} |")
        L.append("")
    if "cross_task" not in R:
        L += ["Checks: the constant and own-gain rules of `CotMinusReq` and `CotMinusForced` equal the collection's and "
              "the forced-direct analysis's frozen files of this task."] if R["scoring"] == "strict" else []
        return L
    c = R["cross_task"]
    L += [f"## Across the two tasks ({c['M']} shared models)", "",
          "| Outcome | Pearson | Spearman | Latent | Index-partial Pearson | Index-partial latent | "
          "With the index, GSM8K | With the index, BIG-Bench Hard |", "|---|---|---|---|---|---|---|---|"]

    def ct(d):
        return f"{fmt(d['estimate'], 2)}" + (f" [{fmt(d['ci95'][0], 2)}, {fmt(d['ci95'][1], 2)}]" if d.get("ci95") else "")
    for o, x in c["outcomes"].items():
        L.append(f"| {o} | {ct(x['raw']['pearson'])} | {ct(x['raw']['spearman'])} | {ct(x['raw']['latent'])} | "
                 f"{ct(x['partial']['pearson'])} | {ct(x['partial']['latent'])} | {ct(x['with_index']['gsm8k'])} | "
                 f"{ct(x['with_index']['bbh'])} |")
    if R["scoring"] == "strict":
        L += ["", "Checks: the constant and own-gain rules of `CotMinusReq` and `CotMinusForced` equal the main and "
              "forced-direct frozen files, the `CotMinusReq` capability rules equal "
              "`manifests/predictions_capability_rule.json`, and its cross-task Pearson correlation equals "
              "`results/transfer_tasks`."]
    elif R["scoring"] == "content":
        L += ["", "Check: every scored row has a content score, and the strict scores of the content table equal the "
              "scored tables row by row."]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    ap.add_argument("--frozen-dir", default=None,
                    help="write and read the frozen rules in this folder instead of manifests/ (refits)")
    ap.add_argument("--scoring", choices=sorted(SCORINGS), default="strict")
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS),
                    help="gsm8k bbh (default) or math alone")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
