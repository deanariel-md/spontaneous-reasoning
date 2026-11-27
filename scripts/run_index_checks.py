r"""Reanalyses of V_delta over prompt draws, capability measured on the external index, the cross-task correlation given the index,
the share of V_delta each capability measure accounts for, leave-one-family-out slopes and a polynomial capability
rule. Estimators: src/capability_checks.py and src/instability.py.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_index_checks.py 2>&1"

Settings: GSM8K-19 (the main panel, results/dataset2/gsm8k.json, with the scoring variants of analysis B), GSM8K-29
(the 30-model extension less its exclusion, results/dataset2/gsm8k30.json) and BIG-Bench Hard (28 models, 25 tasks
pooled with equal weights, results/dataset2/bbh.json). Draws: the 24 discovery and 24 validation units of
manifests/sampler_gsm8k.json and manifests/sampler_bbh.json. The index is each model's value in manifests/d2_panel.json
(GSM8K-19) or manifests/d2_panel_extension.json (the others).

Order: the discovery draws of every setting and every quantity that needs only them; the capability rules
fitted and written to manifests/predictions_capability_rule.json; only then the validation draws of GSM8K-29 and
BIG-Bench Hard. Their validation draws were read by earlier analyses, so the rule is not a fresh test.

Outputs, each with summary.md, summary_clean.md and a JSON with a "macros" object for paper/make_numbers.py:
    results/vdelta_draws/vdelta_draws.json           (7.1)
    results/capability_index/capability_index.json   (2, 7.2, 7.4, 7.5), tables/*.csv
    results/capability_rule/capability_rule.json     (7.3)
    results/late_answers/late_answers.json           (7.6), scatter.csv and results/figures/late_answers.csv
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
from src import capability_checks as cc, dataset2 as d2, instability as ins, sampling  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "instability.yaml")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
PRED = os.path.join(ROOT, "manifests", "predictions_capability_rule.json")
RES = os.path.join(ROOT, "results")
NAMES = {"gsm8k": "GSM8K-19", "gsm8k30": "GSM8K-29", "bbh": "BIG-Bench Hard"}
TAG = {"gsm8k": "Gsm", "gsm8k30": "GsmThirty", "bbh": "Bbh"}
PREFIX = {"gsm8k": "db", "gsm8k30": "gt", "bbh": "bh"}
SC = {"points": "Points", "logit": "Logit", "error_reduction": "ErrRed"}
VARIANTS = {"strict": "main scoring (strict)", "variant_a": "variant A (answer within 64 tokens)",
            "variant_b": "variant B (short direct answers only)", "lenient": "lenient extraction"}
VA = {"variant_a": "VarA", "variant_b": "VarB", "lenient": "Lenient"}
RULE_WORDS = {"constant": "constant", "baseline_linear": "linear in baseline", "index_linear": "linear in the index",
              "baseline_index_linear": "linear in baseline and index", "baseline_quadratic": "quadratic in baseline",
              "baseline_quadratic_index": "quadratic in baseline plus linear in the index"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}"


def ci_txt(d, nd=2):
    c = d.get("ci95") if d else None
    return "--" if not c else f"[{fmt(c[0], nd)}, {fmt(c[1], nd)}]"


def set_txt(s, nd=3):
    if s["kind"] == "bounded":
        return f"[{fmt(s['a'], nd)}, {fmt(s['b'], nd)}]"
    return s["kind"]


def write(name, out, lines):
    d = os.path.join(RES, name)
    os.makedirs(d, exist_ok=True)
    for k in out.get("macros", {}):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, indent=1)
    txt = "\n".join(lines) + "\n"
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)


# ============================================================================ inputs


def load_inputs():
    """Panels, loaders and indices of the three settings; nothing is read past the panel checks."""
    import run_instability as ri
    from run_transfer_tasks import eci_for
    from src import d2_io, d2_panels
    s_g = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8"))
    s_b = json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
    items_g, items_info = d2_io.load_items(split="test")
    gsm = d2_panels.inputs(None, None, s_g, items_info, os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    bbh = d2_panels.bbh_inputs(None, s_b)
    G19 = ri.load_gsm8k()
    S = {"gsm8k30": {"panel": list(gsm["panel"]), "families": gsm["families"], "load": gsm["load"], "sampler": s_g,
                     "items_by_task": {"gsm8k": items_g["item_id"].tolist()}, "eci": eci_for(gsm["panel"])},
         "bbh": {"panel": list(bbh["panel"]), "families": bbh["families"], "load": bbh["load"], "sampler": s_b,
                 "items_by_task": bbh["items_by_task"], "eci": eci_for(bbh["panel"])},
         "gsm8k": {"panel": list(G19["panel"]), "families": G19["families"], "sampler": G19["sampler"],
                   "items_by_task": {"gsm8k": G19["items"]["item_id"].tolist()},
                   "eci": [float(G19["eci"][k]) for k in G19["panel"]], "G": G19}}
    for s in S.values():
        s["n_items"] = sum(len(v) for v in s["items_by_task"].values())
    return S


def arrays(s, units, value="strict_correct", cells_fn=None, panel=None):
    """(B, P), each (K, M) in pp, the tasks pooled with equal weights."""
    panel = panel or s["panel"]
    if cells_fn is not None:
        return d2.unit_arrays(cells_fn(units, panel), units, panel, value)
    df = s["load"](units) if "load" in s else None
    if df is None:
        sc = s["G"]["scored"]
        df = sc[sc["model_key"].isin(panel) & sc["config_id"].isin(set(map(str, units)))]
    parts = []
    for items in s["items_by_task"].values():
        cells = d2.cell_table(df[df["item_id"].isin(items)], panel, units, items)
        parts.append(d2.unit_arrays(cells, units, panel, value))
    return tuple(np.mean([p[i] for p in parts], axis=0) for i in (0, 1))


def check_main(name, Bd, Pd):
    """The discovery arrays reproduce the main analysis's V_delta and mean gain (the same data)."""
    e = json.load(open(os.path.join(RES, "dataset2", f"{name}.json"), encoding="utf-8"))
    G = Pd - Bd
    v = cc.vdelta_draws(G)["estimate"]
    if e["models"] != SETTINGS[name]["panel"]:
        raise SystemExit(f"{name}: the panel differs from results/dataset2/{name}.json")
    if abs(v - e["step2"]["V_delta"]["estimate"]) > 1e-6 * max(1.0, abs(v)):
        raise SystemExit(f"{name}: V_delta {v} differs from results/dataset2/{name}.json")
    return e


# ============================================================================ discovery-only blocks


def slope_block(Bd, Pd, n_items, e, method, alpha):
    """Slopes of benefit on the index and on baseline accuracy, three scales, discovery draws."""
    out = {}
    K = Bd.shape[0]
    sd = float(np.std(e, ddof=1))
    for sc in ins.SCALES:
        s = ins.scale_scores(Bd, Pd, sc, n_items)
        r = {}
        for cap, X in (("index", cc.tile(e, K)), ("baseline", s["x"])):
            m = ins.scale_moments(X, s["y"], method, alpha)
            m.pop("model_means", None)
            r[cap] = m
        st = r["index"]["slope_set"]
        r["index"]["slope_per_sd"] = r["index"]["slope"] * sd
        r["index"]["slope_per_sd_set"] = {"kind": st["kind"], "a": st["a"] * sd, "b": st["b"] * sd}
        r["replaced_cells"] = s["n_replaced"]
        out[sc] = r
    return {"index_sd": sd, "index_mean": float(np.mean(e)), "scales": out}


def weighted_slope_block(Bd, Pd, n_items, e, w, method, alpha):
    """slope_block with model weights (the equal-family-weighted regression, src/instability.weighted_moments)."""
    out = {}
    K = Bd.shape[0]
    sd = float(np.std(e, ddof=1))
    for sc in ins.SCALES:
        s = ins.scale_scores(Bd, Pd, sc, n_items)
        r = {cap: ins.weighted_moments(X, s["y"], w, method, alpha)
             for cap, X in (("index", cc.tile(e, K)), ("baseline", s["x"]))}
        st = r["index"]["slope_set"]
        r["index"]["slope_per_sd"] = r["index"]["slope"] * sd
        r["index"]["slope_per_sd_set"] = {"kind": st["kind"], "a": st["a"] * sd, "b": st["b"] * sd}
        r["replaced_cells"] = s["n_replaced"]
        out[sc] = r
    return {"index_sd": sd, "index_mean": float(np.mean(e)), "weights": [float(x) for x in w], "scales": out}


def subpanel_block(label, Bd, Pd, n_items, e, sub):
    """Share of negative slopes over the random subpanels of analysis B (the same subpanels: stream
    sampling.rng_for(seed_label, setting, "random", size)), on the index and on baseline, discovery draws."""
    M, K = Bd.shape[1], Bd.shape[0]
    subsets = []
    for k in sub["sizes"]:
        rng = sampling.rng_for(sub["seed_label"], label, "random", k)
        ss, mode = ins.subpanel_draw(list(range(M)), k, sub["n_resamples"], rng)
        subsets.append((k, ss, mode))
    out = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(Bd, Pd, sc, n_items)
        r = {}
        for cap, X in (("index", cc.tile(e, K)), ("baseline", s["x"])):
            sl = np.concatenate([ins.batch_slopes(X, s["y"], ss) for _, ss, _ in subsets])
            sg = ins._signs(sl)
            r[cap] = {**sg, "n_subpanels": int(len(sl)), "share_negative": sg["negative"] / len(sl)}
        out[sc] = r
    return {"sizes": {str(k): {"n": int(len(ss)), "mode": mode} for k, ss, mode in subsets}, "scales": out}


def late_block(T, units, panel, Bd, Pd, cap):
    """Per model, the share of strictly correct direct answers whose boxed answer ends after token `cap`, over the
    listed draws (each draw's configuration counted once per draw), against the model's mean gain."""
    from scipy.stats import rankdata
    t = T.assign(late=((T["strict_correct"] == 1) & (T["answer_token_position"] > cap)).astype(int))
    agg = t.groupby(["model_key", "config_id"]).agg(correct=("strict_correct", "sum"), late=("late", "sum"))
    C = np.array([[agg.loc[(m, str(u)), "correct"] for m in panel] for u in units], float)
    L = np.array([[agg.loc[(m, str(u)), "late"] for m in panel] for u in units], float)
    G = Pd - Bd
    K = len(units)

    def point(keep):
        c, lt, g = C[keep].sum(0), L[keep].sum(0), G[keep].mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(c > 0, lt / np.where(c > 0, c, 1), np.nan)
        ok = np.isfinite(s)
        return s, g, {"pearson": float(np.corrcoef(s[ok], g[ok])[0, 1]),
                      "spearman": float(np.corrcoef(rankdata(s[ok]), rankdata(g[ok]))[0, 1])}
    s, g, est = point(np.ones(K, bool))
    loo = [point(np.arange(K) != u)[2] for u in range(K)]
    return {"cap_tokens": cap, "models": panel, "K": K, "share_late": s.tolist(), "gain": g.tolist(),
            "baseline": Bd.mean(0).tolist(), "n_correct": C.sum(0).tolist(), "n_late": L.sum(0).tolist(),
            "correlation": {k: cc._jack(v, [r[k] for r in loo]) for k, v in est.items()}}


def lofo_block(Bd, Pd, n_items, e, families, method, alpha):
    rows = []
    e = np.asarray(e, float)
    for fam in sorted(set(families)):
        keep = [i for i, f in enumerate(families) if f != fam]
        s = ins.scale_scores(Bd[:, keep], Pd[:, keep], "points", n_items)
        row = {"left_out": fam, "M": len(keep)}
        for cap, X in (("baseline", s["x"]), ("index", cc.tile(e[keep], Bd.shape[0]))):
            m = ins.scale_moments(X, s["y"], method, alpha)
            row[cap] = {"slope": m["slope"], "slope_set": m["slope_set"], "set_sign": m["slope_set_sign"]}
        rows.append(row)
    return rows


# ============================================================================ main


SETTINGS = {}


def run(a):
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    alpha, sub = float(cfg["alpha"]), cfg["subpanels"]
    method = d2.load_selection(SELECTION)["H"]
    S = load_inputs()
    SETTINGS.update(S)
    D, main = {}, {}
    for name, s in S.items():
        disc = list(s["sampler"]["discovery_config_ids"])
        D[name] = arrays(s, disc)
        main[name] = check_main(name, *D[name])
    print(f"discovery draws loaded ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ 7.1 V_delta over prompt draws
    vd = {}
    for name in ("gsm8k", "gsm8k30", "bbh"):
        Bd, Pd = D[name]
        r = cc.vdelta_draws(Pd - Bd)
        m = main[name]["step2"]["V_delta"]
        r["calibrated"] = {"method": m["method"], "ci95": m["ci95"], "upper_method": m["upper_method"],
                           "upper95": m["upper95"]}
        r["gain_sd"] = m["gain_sd"]
        r["gain_sd_ci95"] = None if not r["ci95"] else [float(np.sqrt(x)) if x >= 0 else None for x in r["ci95"]]
        r["M"], r["K"] = int(Bd.shape[1]), int(Bd.shape[0])
        vd[name] = r

    # ------------------------------------------------------------------ 2, 7.4 slopes on the index
    slopes, panels = {}, {}
    for name in ("gsm8k30", "bbh", "gsm8k"):
        s = S[name]
        slopes[name] = slope_block(*D[name], s["n_items"], s["eci"], method, alpha)
        panels[name] = subpanel_block(name, *D[name], s["n_items"], s["eci"], sub)
    # GSM8K-19 scoring variants (analysis B's arrays): lenient, variant A (answer within 64 tokens), variant B
    import run_instability as ri
    s19 = S["gsm8k"]
    G19, disc19 = s19["G"], list(s19["sampler"]["discovery_config_ids"])
    instab = json.load(open(os.path.join(RES, "instability", "instability.json"), encoding="utf-8"))
    dropped = list(instab["variant_b_rule"]["dropped"])
    var, late = {}, None
    tok_info = ri.local_tokenizers(G19["reg"], s19["panel"], cfg["variants"]["tokenizer_files"])
    for va in ("variant_a", "variant_b", "lenient"):
        if va == "variant_a" and tok_info is None:
            print("variant A skipped: tokenizer files missing")
            continue
        panel = [k for k in s19["panel"] if k not in dropped] if va == "variant_b" else s19["panel"]
        value = "lenient_correct" if va == "lenient" else "strict_correct"
        if va == "variant_a":
            from src import scoring
            spec = scoring.spec_from_dict(json.load(open(os.path.join(ROOT, "manifests", "gsm8k_scorer.json")))[
                "format_spec"])
            T = ri.token_table(G19, sorted(set(disc19)), tok_info, spec)
            cap = int(cfg["variants"]["direct_answer_cap_tokens"])
            fn = lambda units, pv: ri.variant_cells(G19, "variant_a", units, pv, T, cap)  # noqa: E731
        else:
            fn = lambda units, pv: ri.variant_cells(G19, va, units, pv)  # noqa: E731
        if va == "variant_a":
            late = late_block(T, disc19, s19["panel"], *D["gsm8k"], cap)
        Bv_, Pv_ = arrays(s19, disc19, value, cells_fn=fn, panel=panel)
        e = [s19["eci"][s19["panel"].index(k)] for k in panel]
        r = slope_block(Bv_, Pv_, s19["n_items"], e, method, alpha)
        r["M"] = len(panel)
        h = instab["variants"][va]["H"]["estimate"]
        if abs(-10 * r["scales"]["points"]["baseline"]["slope"] - h) > 1e-6:
            raise SystemExit(f"{va}: the baseline slope does not reproduce results/instability/instability.json")
        var[va] = r
    # Lenient extraction on the two main panels: the same slope blocks on the lenient scores of the discovery draws
    lenient = {}
    for name in ("gsm8k30", "bbh"):
        s = S[name]
        lenient[name] = slope_block(*arrays(s, list(s["sampler"]["discovery_config_ids"]), "lenient_correct"),
                                    s["n_items"], s["eci"], method, alpha)
        lenient[name]["M"] = len(s["panel"])
    print(f"slopes done ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ 7.2 correlations on the shared models
    gp, bp = S["gsm8k30"]["panel"], S["bbh"]["panel"]
    shared = [k for k in gp if k in bp]
    ig, ib = [gp.index(k) for k in shared], [bp.index(k) for k in shared]
    Gg = (D["gsm8k30"][1] - D["gsm8k30"][0])[:, ig]
    Gb = (D["bbh"][1] - D["bbh"][0])[:, ib]
    e_sh = [S["gsm8k30"]["eci"][i] for i in ig]
    if e_sh != [S["bbh"]["eci"][i] for i in ib]:
        raise SystemExit("the index differs between the two panels' manifests")
    raw = np.corrcoef(Gg.mean(0), Gb.mean(0))[0, 1]
    tr = json.load(open(os.path.join(RES, "transfer_tasks", "transfer_tasks.json"), encoding="utf-8"))
    if abs(raw - tr["correlation"]["discovery_24"]["pearson"]["estimate"]) > 1e-9:
        raise SystemExit("the cross-task Pearson correlation differs from results/transfer_tasks")
    corr = {"models": shared, "M": len(shared),
            "with_index": {"gsm8k": cc.correlation_with_index(Gg, e_sh), "bbh": cc.correlation_with_index(Gb, e_sh)},
            "raw_pearson": float(raw), "partial": cc.partial_correlation(Gg, Gb, e_sh)}
    shares = {}
    for name in ("gsm8k30", "bbh", "gsm8k"):
        Bd, Pd = D[name]
        shares[name] = {"index": cc.explained_share(Pd - Bd, cc.tile(S[name]["eci"], Bd.shape[0])),
                        "baseline": cc.explained_share(Pd - Bd, Bd)}

    # ------------------------------------------------------------------ 7.5 leave one family out
    lofo = {name: lofo_block(*D[name], S[name]["n_items"], S[name]["eci"], S[name]["families"], method, alpha)
            for name in ("gsm8k30", "bbh")}

    # ------------------------------------------------------------------ 7.3 capability rules, frozen first
    rules = {}
    for name in ("gsm8k30", "bbh"):
        Bd, Pd = D[name]
        e = S[name]["eci"]
        ch = cc.choose_rule(Bd, Pd, e)
        spec = {n: (dg, ix) for n, dg, ix in cc.CANDIDATES}
        fit = {}
        for n in sorted({ch["chosen"], cc.FULL_RULE}):
            f = cc.fit_rule(Bd, Pd, e, *spec[n])
            fit[n] = {"prediction": f["prediction"].tolist(), "coefficients": f["coefficients"],
                      "n_parameters": f["n_parameters"], "rank": f["rank"]}
        rules[name] = {"selection": ch, "fitted": fit, "models": S[name]["panel"], "eci": e}
    preds = {n: {"chosen": r["selection"]["chosen"], "rules": {k: v["prediction"] for k, v in r["fitted"].items()}}
             for n, r in rules.items()}
    frozen = {"label": "capability_rule", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "units": "pp",
              "models": {n: r["models"] for n, r in rules.items()},
              "samplers": {"gsm8k30": S["gsm8k30"]["sampler"]["sha256"], "bbh": S["bbh"]["sampler"]["sha256"]},
              "note": "fitted on the 24 discovery draws only; written before this run builds "
                      "the validation draws, which earlier analyses had read",
              "predictions": preds}
    frozen["sha256"] = sha256_json(json.loads(json.dumps(jsonable(preds))))
    frozen = d2.write_frozen(jsonable(frozen), PRED, overwrite=a.overwrite_frozen)
    print(f"capability rules frozen -> {rel(PRED)} ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ validation draws (built only now)
    scored = {}
    for name in ("gsm8k30", "bbh"):
        s = S[name]
        Bv, Pv = arrays(s, list(s["sampler"]["validation_config_ids"]))
        Zv = Pv - Bv
        mf = json.load(open(os.path.join(ROOT, "manifests", f"predictions_{name}.json"), encoding="utf-8"))
        if mf["models"] != s["panel"]:
            raise SystemExit(f"{name}: manifests/predictions_{name}.json has another panel")
        P = {k: np.asarray(mf["budgets"]["24"][k]["prediction"], float) for k in ("constant", "own_gain",
                                                                                  "averaged_ols")}
        if np.max(np.abs(P["own_gain"] - (D[name][1] - D[name][0]).mean(0))) > 1e-9:
            raise SystemExit(f"{name}: the frozen own-gain rule differs from the discovery means")
        fr = frozen["predictions"][name]
        P.update({k: np.asarray(v, float) for k, v in fr["rules"].items()})
        chosen = fr["chosen"]
        risks = {k: float(cc.cross_risk(Zv, v)) for k, v in P.items()}
        comps = [("constant", cc.FULL_RULE), ("averaged_ols", cc.FULL_RULE), (cc.FULL_RULE, "own_gain")]
        if chosen not in (cc.FULL_RULE, "constant"):
            comps += [("constant", chosen), (chosen, "own_gain")]
        diffs = [{"f": f, "g": g, **cc.paired_difference(Zv, P[f], P[g])} for f, g in comps]
        scored[name] = {"K_validation": int(Zv.shape[0]), "risk_pp2": risks, "differences": diffs,
                        "chosen": chosen}
    print(f"validation scored ({time.time() - t0:.0f} s)")

    common = {"config": {"path": rel(CONFIG)}, "slope_method": method, "methods": "scripts/run_index_checks.py"}
    out_vd = {"analysis": "V_delta over prompt draws", **common, "settings": vd}
    out_vd["macros"] = macros_vd(out_vd)
    write("vdelta_draws", out_vd, summary_vd(out_vd))
    out_ci = {"analysis": "capability measured on the external index",
              **common, "settings": {n: {"name": NAMES[n], "models": S[n]["panel"], "families": S[n]["families"],
                                         "eci": S[n]["eci"]} for n in S},
              "slopes": slopes, "subpanels": panels, "variants": var, "variant_b_dropped": dropped,
              "lenient_main_panels": lenient,
              "correlation": corr, "explained_share": shares, "leave_one_family_out": lofo}
    out_ci["macros"] = macros_ci(out_ci)
    write("capability_index", out_ci, summary_ci(out_ci))
    tdir = os.path.join(RES, "capability_index", "tables")
    os.makedirs(tdir, exist_ok=True)
    pd.DataFrame([{"setting": n, "left_out": r["left_out"], "M": r["M"],
                   **{f"{c}_{k}": (r[c]["slope"] if k == "slope" else r[c]["slope_set"][k]) for c in ("baseline", "index")
                      for k in ("slope", "a", "b")}, "baseline_set_kind": r["baseline"]["slope_set"]["kind"],
                   "index_set_kind": r["index"]["slope_set"]["kind"]}
                  for n, rows in lofo.items() for r in rows]).to_csv(os.path.join(tdir, "lofo.csv"), index=False)
    out_cr = {"analysis": "a polynomial capability rule", **common,
              "candidates": [{"name": n, "degree": dg, "index": ix} for n, dg, ix in cc.CANDIDATES],
              "frozen": {"path": rel(PRED), "sha256": frozen["sha256"], "created": frozen["created"]},
              "rules": rules, "scored": scored}
    out_cr["macros"] = macros_cr(out_cr)
    write("capability_rule", out_cr, summary_cr(out_cr))
    if late is not None:
        out_la = {"analysis": "late direct answers as a continuous variable",
                  "methods": "scripts/run_index_checks.py", "setting": "GSM8K-19, discovery draws", **late}
        c = late["correlation"]
        out_la["macros"] = {"laM": str(len(late["models"])), "laCap": str(late["cap_tokens"]),
                            "laShareMin": fmt(100 * np.nanmin(late["share_late"]), 0),
                            "laShareMax": fmt(100 * np.nanmax(late["share_late"]), 0),
                            "laShareMedian": fmt(100 * np.nanmedian(late["share_late"]), 0)}
        for k, stem in (("pearson", "laPearson"), ("spearman", "laSpearman")):
            out_la["macros"][stem] = fmt(c[k]["estimate"], 2)
            if c[k]["ci95"]:
                out_la["macros"][stem + "Lo"], out_la["macros"][stem + "Hi"] = (fmt(x, 2) for x in c[k]["ci95"])
        write("late_answers", out_la, summary_la(out_la))
        sc = pd.DataFrame({"model_key": late["models"], "share_late": late["share_late"], "gain": late["gain"],
                           "baseline": late["baseline"], "n_correct_direct": late["n_correct"],
                           "n_late": late["n_late"]})
        sc.to_csv(os.path.join(RES, "late_answers", "scatter.csv"), index=False)
        sc.to_csv(os.path.join(RES, "figures", "late_answers.csv"), index=False)
    print(f"done in {time.time() - t0:.0f} s")


# ============================================================================ macros


def macros_vd(out):
    m = {}
    for name, r in out["settings"].items():
        p = PREFIX[name]
        if r["ci95"]:
            m[p + "VdeltaDrawLo"], m[p + "VdeltaDrawHi"] = fmt(r["ci95"][0], 0), fmt(r["ci95"][1], 0)
        if r["gain_sd_ci95"]:
            m[p + "GainSDDrawLo"], m[p + "GainSDDrawHi"] = (fmt(x, 2) for x in r["gain_sd_ci95"])
    return m


def _put_set(m, stem, est, s, nd=3):
    m[stem] = fmt(est, nd)
    if s["kind"] == "bounded":
        m[stem + "Lo"], m[stem + "Hi"] = fmt(s["a"], nd), fmt(s["b"], nd)
    else:
        m[stem + "Lo"] = m[stem + "Hi"] = s["kind"]


def macros_ci(out):
    m = {}
    for name, r in out["slopes"].items():
        t = TAG[name]
        m[f"eci{t}Sd"] = fmt(r["index_sd"], 1)
        for sc, st in SC.items():
            x = r["scales"][sc]["index"]
            _put_set(m, f"eci{t}Slope{st}", x["slope"], x["slope_set"])
            _put_set(m, f"eci{t}SlopePerSd{st}", x["slope_per_sd"], x["slope_per_sd_set"], 2)
            p = out["subpanels"][name]["scales"][sc]
            m[f"eci{t}PanelNeg{st}"] = fmt(100 * p["index"]["share_negative"], 0)
            m[f"eci{t}PanelNegBase{st}"] = fmt(100 * p["baseline"]["share_negative"], 0)
        m[f"eci{t}PanelNeg"] = m[f"eci{t}PanelNegPoints"]
    for va, r in out["variants"].items():
        for sc, st in SC.items():
            x = r["scales"][sc]["index"]
            _put_set(m, f"eci{VA[va]}Slope{st}", x["slope"], x["slope_set"])
        m[f"eci{VA[va]}M"] = str(r["M"])
    for name, r in out["lenient_main_panels"].items():
        for sc, st in SC.items():
            for c_, ct in (("index", ""), ("baseline", "Base")):
                x = r["scales"][sc][c_]
                _put_set(m, f"eci{TAG[name]}Lenient{ct}Slope{st}", x["slope"], x["slope_set"])
    c = out["correlation"]
    m["eciSharedM"] = str(c["M"])
    for task, t in (("gsm8k", "Gsm"), ("bbh", "Bbh")):
        x = c["with_index"][task]
        m[f"eciCorr{t}"] = fmt(x["estimate"], 2)
        if x["ci95"]:
            m[f"eciCorr{t}Lo"], m[f"eciCorr{t}Hi"] = fmt(x["ci95"][0], 2), fmt(x["ci95"][1], 2)
    for k, stem in (("pearson", "eciPartialCorr"), ("latent", "eciPartialLatent")):
        x = c["partial"][k]
        m[stem] = fmt(x["estimate"], 2)
        if x["ci95"]:
            m[stem + "Lo"], m[stem + "Hi"] = fmt(x["ci95"][0], 2), fmt(x["ci95"][1], 2)
    for name, r in out["explained_share"].items():
        for cap, ct in (("index", "Index"), ("baseline", "Base")):
            x = r[cap]
            m[f"eciShare{ct}{TAG[name]}"] = fmt(x["estimate"], 2)
            if x["ci95"]:
                m[f"eciShare{ct}{TAG[name]}Lo"], m[f"eciShare{ct}{TAG[name]}Hi"] = (fmt(v, 2) for v in x["ci95"])
    for name, rows in out["leave_one_family_out"].items():
        t = TAG[name]
        for cap, ct in (("baseline", "Base"), ("index", "Index")):
            v = [r[cap]["slope"] for r in rows]
            m[f"eciLofo{t}{ct}Min"], m[f"eciLofo{t}{ct}Max"] = fmt(min(v), 3), fmt(max(v), 3)
            m[f"eciLofo{t}{ct}Neg"] = str(sum(x < 0 for x in v))
        m[f"eciLofo{t}N"] = str(len(rows))
    return m


def macros_cr(out):
    m = {}
    short = {"constant": "Const", "own_gain": "Own", "averaged_ols": "Ols", cc.FULL_RULE: "QuadIndex"}
    for name, s in out["scored"].items():
        t = TAG[name]
        ch = s["chosen"]
        m[f"cr{t}Chosen"] = RULE_WORDS[ch]
        sh = dict(short)
        sh[ch] = sh.get(ch, "Chosen")
        for k, v in s["risk_pp2"].items():
            m[f"cr{t}Risk{sh.get(k, 'Chosen')}"] = fmt(v, 1)
        rc = s["risk_pp2"]["constant"]
        m[f"cr{t}QuadIndexShare"] = fmt(100 * s["risk_pp2"][cc.FULL_RULE] / rc, 0) if rc > 0 else "--"
        for d in s["differences"]:
            stem = f"cr{t}{sh.get(d['f'], 'Chosen')}Vs{sh.get(d['g'], 'Chosen')}"
            m[stem] = fmt(d["estimate"], 1)
            if d["ci95"]:
                m[stem + "Lo"], m[stem + "Hi"] = fmt(d["ci95"][0], 1), fmt(d["ci95"][1], 1)
    return m


# ============================================================================ summaries


def summary_vd(out):
    L = ["# V_delta over prompt draws", "",
         "V_delta on the 24 discovery draws with a leave-one-draw-out jackknife t interval: the "
         "uncertainty of the fixed panel's V_delta from the prompt draws alone. Beside it, the interval printed so "
         "far, the calibrated one (modified large-sample variance-component interval on the model-by-draw mean "
         "squares). The calibration's targets were fixed-panel quantities, but its designs had gain SDs of a few "
         "points. At several hundred pp² the variance-component interval's width is set by the spread of the M "
         "model means (M − 1 degrees of freedom) and does not shrink with more draws, so it behaves as an interval "
         "for a panel sampled from a population of models. In the calibrated designs the jackknife covered at least "
         "92.2% and the variance-component interval 93.1% (`results/calibration/selection_run2.json`); here the "
         "jackknife is descriptive.", "",
         "| Setting | M | V_delta (pp²) | Over prompt draws (jackknife) | Variance-component (calibrated) | Gain SD (pp) | Gain SD over draws |",
         "|---|---:|---:|---|---|---:|---|"]
    for name, r in out["settings"].items():
        c = r["calibrated"]["ci95"]
        L.append(f"| {NAMES[name]} | {r['M']} | {fmt(r['estimate'], 1)} | {ci_txt(r, 1)} | "
                 f"{'--' if not c else f'[{fmt(c[0], 1)}, {fmt(c[1], 1)}]'} | {fmt(r['gain_sd'], 2)} | "
                 f"{'--' if not r['gain_sd_ci95'] else f'[{fmt(r['gain_sd_ci95'][0])}, {fmt(r['gain_sd_ci95'][1])}]'} |")
    return L


def summary_ci(out):
    L = ["# Capability measured on the external index", "",
         "Discovery draws, strict scoring. The index is each model's value in "
         "the panel manifest, the same in every draw, so the slope on the index is the panel covariance of mean "
         "benefit with the index over the index's panel variance. The 95% sets use the H method of "
         f"`manifests/calibration_selection_v2.json` ({out['slope_method']}); on the index it reduces to a t "
         "interval on the leave-one-draw-out slopes. The calibration covered the baseline slope in points only, so "
         "every other set is descriptive.", "",
         "## Slopes", "",
         "Benefit per index point (points: pp of gain; logits: logits of gain; error reduction: pp of the baseline "
         "error removed), with the baseline slope beside it. Per SD multiplies the index slope by the index's SD "
         "across the panel.", "",
         "| Setting | Scale | Index slope | 95% set | Per SD of the index | Baseline slope | 95% set |",
         "|---|---|---:|---|---:|---:|---|"]
    for name, r in out["slopes"].items():
        for sc in ins.SCALES:
            x, b = r["scales"][sc]["index"], r["scales"][sc]["baseline"]
            L.append(f"| {NAMES[name]} | {ins.SCALE_LABELS[sc]} | {fmt(x['slope'], 3)} | {set_txt(x['slope_set'])} | "
                     f"{fmt(x['slope_per_sd'], 2)} {set_txt(x['slope_per_sd_set'], 2)} | {fmt(b['slope'], 3)} | "
                     f"{set_txt(b['slope_set'])} |")
    L += ["", "Index SD across the panel (denominator M − 1): " + "; ".join(
        f"{NAMES[n]} {fmt(r['index_sd'], 2)} (mean {fmt(r['index_mean'], 1)})" for n, r in out["slopes"].items()) + ".",
          "", "### GSM8K-19 scoring variants", "",
          f"Variant B drops {', '.join(out['variant_b_dropped'])}.", "",
          "| Variant | M | Points | Logits | Error reduction |", "|---|---:|---|---|---|"]
    rows = {"strict": {"M": len(out["settings"]["gsm8k"]["models"]), "scales": out["slopes"]["gsm8k"]["scales"]}}
    rows.update(out["variants"])
    for va, r in rows.items():
        L.append(f"| {VARIANTS[va]} | {r['M']} | " + " | ".join(
            f"{fmt(r['scales'][sc]['index']['slope'], 3)} {set_txt(r['scales'][sc]['index']['slope_set'])}"
            for sc in ins.SCALES) + " |")
    L += ["", "### Lenient extraction on GSM8K-29 and BIG-Bench Hard", "",
          "The same slopes on the lenient scores of the discovery draws (strict values in the first table).", "",
          "| Setting | M | Scale | On the index | On baseline |", "|---|---:|---|---|---|"]
    for name, r in out["lenient_main_panels"].items():
        for sc in ins.SCALES:
            x, b = r["scales"][sc]["index"], r["scales"][sc]["baseline"]
            L.append(f"| {NAMES[name]} | {r['M']} | {ins.SCALE_LABELS[sc]} | {fmt(x['slope'], 3)} "
                     f"{set_txt(x['slope_set'])} | {fmt(b['slope'], 3)} {set_txt(b['slope_set'])} |")
    L += ["", "## Random subpanels of four to six models", "",
          "Share of subpanels whose discovery slope is negative, the subpanels of analysis B (same seeds and sizes).",
          "", "| Setting | Subpanels | Scale | On the index | On baseline |", "|---|---:|---|---:|---:|"]
    for name, p in out["subpanels"].items():
        for sc in ins.SCALES:
            x = p["scales"][sc]
            L.append(f"| {NAMES[name]} | {x['index']['n_subpanels']:,} | {ins.SCALE_LABELS[sc]} | "
                     f"{fmt(100 * x['index']['share_negative'], 1)}% | {fmt(100 * x['baseline']['share_negative'], 1)}% |")
    c = out["correlation"]
    L += ["", f"## Correlations over the {c['M']} models of both tasks", "",
          "Pearson correlations of discovery mean gains with the index, and of the two tasks' mean gains after "
          "removing a linear function of the index from each (partial). The latent partial correlation uses the "
          "per-draw residuals' cross-draw moments, which remove prompt noise. Leave-one-bundle-out jackknife t "
          "intervals over prompt draws for these models only.", "",
          "| Quantity | Estimate | 95% interval |", "|---|---:|---|",
          f"| GSM8K gain with the index | {fmt(c['with_index']['gsm8k']['estimate'])} | {ci_txt(c['with_index']['gsm8k'])} |",
          f"| BIG-Bench Hard gain with the index | {fmt(c['with_index']['bbh']['estimate'])} | {ci_txt(c['with_index']['bbh'])} |",
          f"| GSM8K with BIG-Bench Hard gain, raw | {fmt(c['raw_pearson'])} | see `results/transfer_tasks` |",
          f"| the same, partial on the index | {fmt(c['partial']['pearson']['estimate'])} | {ci_txt(c['partial']['pearson'])} |",
          f"| the same, partial, latent | {fmt(c['partial']['latent']['estimate'])} | {ci_txt(c['partial']['latent'])} |",
          "", "## Share of V_delta each capability measure accounts for", "",
          "A_x² / (V_x V_delta), the squared latent correlation of benefit with capability x, on each task's full "
          "panel; jackknife t over draws.", "", "| Setting | M | Index | Baseline accuracy |", "|---|---:|---|---|"]
    for name, r in out["explained_share"].items():
        L.append(f"| {NAMES[name]} | {len(out['settings'][name]['models'])} | {fmt(r['index']['estimate'])} "
                 f"{ci_txt(r['index'])} | {fmt(r['baseline']['estimate'])} {ci_txt(r['baseline'])} |")
    L += ["", "## Leave one family out", "", "Slope in points, discovery draws.", "",
          "| Setting | Left out | M | On baseline | 95% set | On the index | 95% set |", "|---|---|---:|---:|---|---:|---|"]
    for name, rows in out["leave_one_family_out"].items():
        for r in rows:
            L.append(f"| {NAMES[name]} | {r['left_out']} | {r['M']} | {fmt(r['baseline']['slope'], 3)} | "
                     f"{set_txt(r['baseline']['slope_set'])} | {fmt(r['index']['slope'], 3)} | "
                     f"{set_txt(r['index']['slope_set'])} |")
    return L


def summary_cr(out):
    L = ["# A polynomial capability rule", "",
         "Candidates, simplest first: " +
         "; ".join(RULE_WORDS[c["name"]] for c in out["candidates"]) +
         ". Each is least squares of the model mean gains on the model mean baselines (centred) and the index. "
         "Complexity is chosen on the discovery draws only: fit on draws 1–12, cross-product risk on draws 13–24, "
         "and the reverse; the lowest average wins, a tie going to the simpler rule. The chosen rule and the full "
         f"quadratic-plus-index rule were refitted on all 24 discovery draws and saved to `{out['frozen']['path']}` "
         f"({out['frozen']['created']}) before this run built the validation draws. Earlier analyses had read those "
         "draws, so this is not a fresh test. Scoring as Step 3: cross-product risks in pp² (may be negative) and "
         "paired loss differences with jackknife t intervals (positive favours the second rule), against the main "
         "analysis's frozen constant, own-gain and capability (averaged OLS on baseline) rules.", ""]
    for name, s in out["scored"].items():
        sel = out["rules"][name]["selection"]
        L += [f"## {NAMES[name]}", "", "Split-half scores on the discovery draws (pp²):", "",
              "| Candidate | Score | Halves |", "|---|---:|---|"]
        for c in out["candidates"]:
            x = sel["scores"][c["name"]]
            L.append(f"| {RULE_WORDS[c['name']]} | {fmt(x['score'], 1)} | {fmt(x['risks'][0], 1)}, "
                     f"{fmt(x['risks'][1], 1)} |")
        L += ["", f"Chosen: {RULE_WORDS[s['chosen']]}.", "", "Validation risks (pp²):", "",
              "| Rule | Risk |", "|---|---:|"]
        words = {"constant": "constant", "own_gain": "own gain", "averaged_ols": "capability, linear in baseline",
                 **RULE_WORDS}
        for k, v in s["risk_pp2"].items():
            L.append(f"| {words[k]} | {fmt(v, 1)} |")
        L += ["", "| f | g | g better by (pp²) | 95% interval |", "|---|---|---:|---|"]
        for d in s["differences"]:
            L.append(f"| {words[d['f']]} | {words[d['g']]} | {fmt(d['estimate'], 1)} | {ci_txt(d, 1)} |")
        L.append("")
    return L


def summary_la(out):
    c = out["correlation"]
    L = ["# Late direct answers as a continuous variable", "",
         f"GSM8K-19, the 24 discovery draws. Per model: the share of strictly correct direct answers whose boxed answer "
         f"ends after token {out['cap_tokens']} (token positions of analysis B's variant A; each draw's configuration "
         "counted once per draw), against the model's discovery mean gain. Leave-one-draw-out jackknife t intervals, "
         "descriptive. Token positions do not exist for the ten models GSM8K-29 adds, so it is not repeated there.", "",
         f"Pearson {fmt(c['pearson']['estimate'])} {ci_txt(c['pearson'])}; Spearman {fmt(c['spearman']['estimate'])} "
         f"{ci_txt(c['spearman'])}.", "",
         "| Model | Correct direct answers | Ending after the cap | Share | Gain (pp) | Baseline (pp) |",
         "|---|---:|---:|---:|---:|---:|"]
    rows = sorted(zip(out["models"], out["n_correct"], out["n_late"], out["share_late"], out["gain"], out["baseline"]),
                  key=lambda r: -r[3] if np.isfinite(r[3]) else 0)
    for m, n, lt, s, g, b in rows:
        L.append(f"| {m} | {int(n):,} | {int(lt):,} | {fmt(100 * s, 1)}% | {fmt(g, 1)} | {fmt(b, 1)} |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite-frozen", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
