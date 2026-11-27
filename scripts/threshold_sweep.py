r"""Threshold sweep for "long" (exploratory).

From the repository root:
    python scripts/threshold_sweep.py

A requested-direct output is long when its generated token count (n_output_tokens of the stored output, compared by
'>') exceeds the cutoff. The paper's cutoff is the forced arm's cap, 16 tokens on GSM8K, MATH-500 and BIG-Bench Hard.
This script recomputes three quantities at cutoffs of 8, 16, 32 and 64 tokens with the code of
scripts/run_baseline_mixture.py and src/baseline_mixture.py, the cutoff being the only parameter that changes:
  - VarLong: the long part's share of the between-model variance of the baseline shift (requested minus forced
    accuracy), cov(long part, shift) / var(shift) over models (ddof 1), on every distinct configuration of both splits
    once: the basis of bmixDec<T>VarLongPct (run_baseline_mixture.decomposition_block, set all_distinct). The value on
    the discovery draws (each configuration weighted by its draws) is kept beside it in the JSON.
  - RevPairs of RevOf: model pairs ordered one way by the observed gain over the requested arm (C - R) and the other
    way by the corrected gain (1 - w)(C - F), model means over the 24 discovery draws, w at the cutoff; ties
    (|difference| <= 1e-9 pp) are neither (src/baseline_mixture.mechanism, observed_vs_corrected): the basis of
    bmix<T>RevCorr of bmix<T>Pairs.
  - MeanW: the mean over models of w, each model's share of long requested outputs on the discovery draws, per
    configuration and averaged over the draws (run_baseline_mixture.w_of): the basis of bmix<T>WMeanPct. w needs no
    gold answer, so it is the same under strict and content scoring.
Strict and content scores (data/processed/content/<task>_content.parquet; the strict and content model means come from
the same arrays as run_baseline_mixture.run_task). The forced arm's outputs do not change with the cutoff: they were
generated with a 16-token cap, so above 16 tokens "long" no longer marks outputs the forced arm could not have produced.

HotpotQA (forced cap 64 tokens): cutoffs 16 and 64, from data/processed/hotpot/scored_content.parquet (one table with
the strict and the content exact match and the token counts of every arm; scripts/run_hotpot_content.wide_table). Same
code; the model means of C - R and C - F over the discovery draws come from the rows of the discovery draws, each
configuration weighted by its number of draws (every model has every item in every configuration, so this equals the
mean over draws of each draw's accuracy).

Checks: the 16-token column against results/baseline_mixture/baseline_mixture.json and the macros of paper/numbers.tex
(bmixDec<T>VarLongPct, bmix<T>RevCorr, bmix<T>Pairs, bmix<T>WMeanPct, each with Content); HotpotQA at 16 and 64 against
results/hotpot/decomposition_thresholds_post_hoc.json and hqDecVarLongAllPct<Sixteen|SixtyFour>, and at 64 strict
against the saved forecast manifests/predictions_hotpot_mechanism.json. A failed check is reported in the JSON and the
summary, and the script exits with status 1.

Outputs: results/threshold_sweep/threshold_sweep.json, summary.md, summary_clean.md, tables/threshold_sweep.csv;
paper/tables/threshold_sweep.tex (tabular only, no table float; paper/macros_threshold_sweep.threshold_sweep_table).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import baseline_mixture as bm  # noqa: E402

OUT = os.path.join(ROOT, "results", "threshold_sweep")
TABLE = os.path.join(ROOT, "paper", "tables", "threshold_sweep.tex")
CUTS = (8, 16, 32, 64)
HQ_CUTS = (16, 64)
TASKS = ("gsm8k", "math", "bbh")
SCORES = ("strict", "content")
TITLE = {"gsm8k": "GSM8K-29", "math": "MATH-500 numeric", "bbh": "BIG-Bench Hard-28", "hotpot": "HotpotQA"}
HQ_DEF = {16: "cap16", 64: "cap64"}   # src/hotpot_content.LONG_DEFINITIONS, the same cutoffs


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ============================================================================ the sweep (pure on the rows)


def sweep_cut(wide, panel, disc, val, G, cut) -> dict:
    """The three quantities at one cutoff. wide: rows of (model, configuration, item) with n_output_tokens (the requested
    output's) and requested_/forced_/cot_<score> 0/1; G: {score: {"requested": (M,), "forced": (M,)}} gains of chain of
    thought over each arm, model means over the discovery draws (pp)."""
    import run_baseline_mixture as rbm
    allc = sorted(set(map(str, disc)) | set(map(str, val)))
    w = rbm.w_of(wide, panel, cut, list(map(str, disc)))
    if np.isnan(w).any():
        raise SystemExit("a model lacks requested outputs in a discovery configuration")
    dec_all = rbm.decomposition_block(wide, panel, cut, allc, False)
    dec_disc = rbm.decomposition_block(wide, panel, cut, list(map(str, disc)), True)
    x = wide[wide["config_id"].isin(set(allc))]
    long_rows = int((x["n_output_tokens"].to_numpy() > cut).sum())
    wd = [100 * float(v) for v in w]
    out = {"cut": int(cut), "w_discovery": w.tolist(), "mean_w_pct": sum(wd) / len(wd),
           "median_w_pct": float(np.median(wd)), "min_w_pct": min(wd), "max_w_pct": max(wd),
           "models_w_zero": int(sum(v == 0 for v in wd)), "models_w_one": int(sum(v == 100 for v in wd)),
           "long_rows_all_distinct": long_rows, "rows_all_distinct": int(len(x))}
    for sc in SCORES:
        mech = bm.mechanism(G[sc]["requested"], G[sc]["forced"], w)
        da, dd = dec_all[sc], dec_disc[sc]
        out[sc] = {"variance_share_long": da["variance_share_long"],
                   "variance_share_short": da["variance_share_short"],
                   "variance_terms": da["variance_terms"], "shift_variance_pp2": da["shift_variance_pp2"],
                   "variance_share_long_discovery": dd["variance_share_long"],
                   "reversed_corrected": mech["observed_vs_corrected"]["reversed"],
                   "reversed_forced": mech["observed_vs_forced"]["reversed"],
                   "pairs": mech["observed_vs_corrected"]["pairs"],
                   "ties_corrected": mech["observed_vs_corrected"]["ties"],
                   "pearson_corrected": mech["corr_corrected"]["pearson"],
                   "rms_corrected_pp": mech["rms_corrected_pp"],
                   "discovery_rows_means": {k: dd["per_model"][k] for k in ("R", "F", "C")}}
    return out


def gains_from_rows(dec_disc_per_model) -> dict:
    """{"requested": C - R, "forced": C - F} from a decomposition's per-model accuracies (pp)."""
    p = dec_disc_per_model
    C, F, R = (np.array(p[k], float) for k in ("C", "F", "R"))
    return {"requested": C - R, "forced": C - F}


# ============================================================================ inputs


def task_inputs(task):
    """(wide rows, panel, discovery draws, validation draws, G) for GSM8K, MATH-500 or BIG-Bench Hard, as
    run_baseline_mixture.run_task builds them."""
    import run_baseline_mixture as rbm
    import robustness_inputs as ri
    import run_decision as rd
    wide, panel, disc, val = rbm.task_rows(task)
    x = ri.load(task)
    if x["facts"]["panel"] != panel or x["facts"]["discovery_config_ids"] != disc:
        raise SystemExit(f"{task}: the strict arrays differ in panel or draws")
    d = x["discovery"]
    G = {"strict": {a: (d["cot"] - d[a]).mean(0) for a in ("requested", "forced")}}
    c = rd.task_split(task, "content", "discovery")
    if [str(u) for u in c["units"]] != disc or [str(k) for k in c["panel"]] != panel:
        raise SystemExit(f"{task}: the content arrays differ in panel or draws")
    G["content"] = {a: (np.asarray(c["cot"], float) - np.asarray(c[a], float)).mean(0) for a in ("requested", "forced")}
    return wide, panel, disc, val, G


def hotpot_inputs():
    """(wide rows, panel, discovery draws, validation draws, G) for HotpotQA from the content table (strict and content
    exact match), G from the rows of the discovery draws."""
    import run_baseline_mixture as rbm
    import run_hotpot_content as rhc
    ctx = rhc.setup()
    wide = rhc.wide_table(ctx).rename(columns={"requested_tokens": "n_output_tokens"})
    panel, disc, val = list(ctx["panel"]), list(ctx["disc"]), list(ctx["val"])
    dec = rbm.decomposition_block(wide, panel, 64, disc, True)   # the accuracies do not depend on the cutoff
    G = {sc: gains_from_rows(dec[sc]["per_model"]) for sc in SCORES}
    return wide, panel, disc, val, G


# ============================================================================ checks against the stored values


def _numbers(root=ROOT) -> dict:
    p = os.path.join(root, "paper", "numbers.tex")
    txt = open(p, encoding="utf-8").read() if os.path.exists(p) else ""   # before the first make_numbers.py run
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}\n]*)\}", txt))


def _eq(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def check_task(task, res16, w16, R, N, macros) -> list:
    """The 16-token values against baseline_mixture.json and numbers.tex. Returns a list of check records."""
    import macros_threshold_sweep as mts
    T, D = R["tasks"][task], R["decomposition"][task]
    tt = mts.TASK_TAG[task]
    out = []
    w_old = np.array(T["w_discovery"], float)
    out.append({"what": f"{task}: w per model (discovery), max |difference|",
                "here": float(np.max(np.abs(w16 - w_old))), "stored": 0.0,
                "ok": bool(np.max(np.abs(w16 - w_old)) <= 1e-12)})
    for sc in SCORES:
        old_v = D["sets"]["all_distinct"][sc]["variance_share_long"]
        out.append({"what": f"{task} {sc}: long share of the variance (all distinct)", "here": res16[sc]["variance_share_long"],
                    "stored": old_v, "ok": _eq(res16[sc]["variance_share_long"], old_v)})
        m = T["mechanism"][sc]["observed_vs_corrected"]
        out.append({"what": f"{task} {sc}: reversed pairs under (1 - w)(C - F)", "here": res16[sc]["reversed_corrected"],
                    "stored": m["reversed"], "ok": res16[sc]["reversed_corrected"] == m["reversed"]})
        out.append({"what": f"{task} {sc}: pairs", "here": res16[sc]["pairs"], "stored": m["pairs"],
                    "ok": res16[sc]["pairs"] == m["pairs"]})
        suf = "" if sc == "strict" else "Content"
        for mine, theirs in ((f"thr{tt}SixteenVarLongPct{suf}", f"bmixDec{tt}VarLongPct{suf}"),
                             (f"thr{tt}SixteenRevPairs{suf}", f"bmix{tt}RevCorr{suf}"),
                             (f"thr{tt}SixteenRevOf{suf}", f"bmix{tt}Pairs{suf}"),
                             (f"thr{tt}SixteenMeanWPct{suf}", f"bmix{tt}WMeanPct")):
            out.append({"what": f"macro {mine} = \\{theirs}", "here": macros.get(mine), "stored": N.get(theirs),
                        "ok": macros.get(mine) is not None and (not N or macros.get(mine) == N.get(theirs))})
    return out


def check_hotpot(res, w_by_cut, G, H, F, N, macros) -> list:
    """HotpotQA at 16 and 64 against decomposition_thresholds_post_hoc.json (H), numbers.tex (N) and, at 64 strict, the
    saved forecast (F)."""
    out = []
    for cut in HQ_CUTS:
        d = HQ_DEF[cut]
        for sc in SCORES:
            v = H["variants"][d][sc]
            a = v["sets"]["all_distinct"]["variance_share_long"]
            out.append({"what": f"hotpot {cut} {sc}: long share of the variance (all distinct)",
                        "here": res[cut][sc]["variance_share_long"], "stored": a,
                        "ok": _eq(res[cut][sc]["variance_share_long"], a)})
            md = v["mechanism"]["discovery"]["observed_vs_corrected"]
            out.append({"what": f"hotpot {cut} {sc}: reversed pairs under (1 - w)(C - F), discovery means",
                        "here": res[cut][sc]["reversed_corrected"], "stored": md["reversed"],
                        "ok": res[cut][sc]["reversed_corrected"] == md["reversed"]})
            wd = np.array(v["sets"]["discovery"]["per_model"]["w"], float)
            dif = float(np.max(np.abs(w_by_cut[cut] - wd)))
            out.append({"what": f"hotpot {cut} {sc}: w per model against the row-weighted discovery w, max |difference|",
                        "here": dif, "stored": 0.0, "ok": dif <= 1e-12})
            suf = "" if sc == "strict" else "Content"
            word = {16: "Sixteen", 64: "SixtyFour"}[cut]
            mine, theirs = f"thrHq{word}VarLongPct{suf}", f"hqDecVarLongAllPct{word}{suf}"
            out.append({"what": f"macro {mine} = \\{theirs}", "here": macros.get(mine), "stored": N.get(theirs),
                        "ok": macros.get(mine) is not None and (not N or macros.get(mine) == N.get(theirs))})
    pm = F["per_model"]
    panel = list(F["models"])
    w_saved = np.array([pm[k]["w"] for k in panel], float)
    g_saved = np.array([pm[k]["gain_vs_forced"] for k in panel], float)
    out.append({"what": "hotpot 64 strict: w per model against the saved forecast, max |difference|",
                "here": float(np.max(np.abs(w_by_cut[64] - w_saved))), "stored": 0.0,
                "ok": bool(np.max(np.abs(w_by_cut[64] - w_saved)) <= 1e-12)})
    out.append({"what": "hotpot 64 strict: gain over the forced arm against the saved forecast, max |difference| (pp)",
                "here": float(np.max(np.abs(G["strict"]["forced"] - g_saved))), "stored": 0.0,
                "ok": bool(np.max(np.abs(G["strict"]["forced"] - g_saved)) <= 1e-9)})
    fit = F["discovery_context"]["post_hoc_fit_on_discovery"]["observed_vs_corrected"]
    out.append({"what": "hotpot 64 strict: reversed pairs under (1 - w)(C - F) against the saved forecast's discovery fit",
                "here": res[64]["strict"]["reversed_corrected"], "stored": fit["reversed"],
                "ok": res[64]["strict"]["reversed_corrected"] == fit["reversed"]})
    return out


# ============================================================================ outputs


def f_(x, nd):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}".replace("-", "−")


def table_rows(B) -> list:
    rows = []
    for task, T in B["tasks"].items():
        for sc in SCORES:
            for c in T["cuts"]:
                r = T["by_cut"][str(c)]
                s = r[sc]
                rows.append({"task": task, "score": sc, "cut": c, "mean_w_pct": r["mean_w_pct"],
                             "median_w_pct": r["median_w_pct"], "min_w_pct": r["min_w_pct"], "max_w_pct": r["max_w_pct"],
                             "variance_share_long": s["variance_share_long"],
                             "variance_share_long_discovery": s["variance_share_long_discovery"],
                             "var_long": (s["variance_terms"] or {}).get("long"),
                             "var_short": (s["variance_terms"] or {}).get("short"),
                             "two_cov": (s["variance_terms"] or {}).get("two_cov"),
                             "reversed_corrected": s["reversed_corrected"], "reversed_forced": s["reversed_forced"],
                             "pairs": s["pairs"], "ties_corrected": s["ties_corrected"],
                             "pearson_corrected": s["pearson_corrected"], "rms_corrected_pp": s["rms_corrected_pp"]})
    return rows


def summary(B) -> str:
    L = ["# Threshold sweep for \"long\" (exploratory)", "",
         "A requested-direct output is long when it has more generated tokens than the cutoff. The paper uses 16, the "
         "forced arm's cap on GSM8K, MATH-500 and BIG-Bench Hard; this sweep recomputes three quantities at 8, 16, 32 "
         "and 64 tokens with the same code (`scripts/run_baseline_mixture.py`, `src/baseline_mixture.py`), the cutoff "
         "the only thing that changes. HotpotQA's forced cap is 64 tokens; it is swept at 16 and 64.", "",
         "- Long share of the variance: the long part's share of the between-model variance of requested minus forced "
         "accuracy, cov(long part, shift) / var(shift) over models, every distinct configuration of both splits once "
         "(the basis of `\\bmixDec<T>VarLongPct`). It can fall below 0 or above 100%: it is a covariance share.",
         "- Reversed pairs: pairs ordered one way by the observed gain over the requested arm (C − R) and the other way "
         "by (1 − w)(C − F), model means over the 24 discovery draws, w at the cutoff (the basis of `\\bmix<T>RevCorr`). "
         "Ties are neither reversed nor not; the pair total counts them.",
         "- Mean w: the mean over models of each model's share of long requested outputs, discovery draws (the basis "
         "of `\\bmix<T>WMeanPct`). It needs no gold answer, so it is the same under both scorings.", "",
         "The forced arm's outputs do not change with the cutoff. On GSM8K, MATH-500 and BIG-Bench Hard they were "
         "generated with a 16-token cap, so at 32 and 64 tokens a short requested output can be longer than any forced "
         "output, and at 8 a long one can be as short as some forced ones.", "",
         "| Benchmark | Scores | " + " | ".join(f"{c} tokens: long share / reversed / mean w" for c in CUTS) + " |",
         "|---|---|" + "---|" * len(CUTS)]
    for task, T in B["tasks"].items():
        for sc in SCORES:
            cells = []
            for c in CUTS:
                r = T["by_cut"].get(str(c))
                if r is None:
                    cells.append("not computed")
                    continue
                s = r[sc]
                v = s["variance_share_long"]
                cells.append(f"{f_(None if v is None else 100 * v, 0)}% / {s['reversed_corrected']} of {s['pairs']} / "
                             f"{f_(r['mean_w_pct'], 0)}%")
            L.append(f"| {T['title']} | {sc} | " + " | ".join(cells) + " |")
    tie_notes = []
    for task, T in B["tasks"].items():
        for c in T["cuts"]:
            r = T["by_cut"][str(c)]
            for sc in SCORES:
                if r[sc]["ties_corrected"]:
                    tie_notes.append(f"{T['title']}, {c} tokens, {sc}: {r[sc]['ties_corrected']} tied pair(s) under the "
                                     f"corrected gain ({r['models_w_one']} model(s) with w = 1, corrected gain 0)")
    L += ["", "Ties under the corrected gain (neither reversed nor not; in the pair total): "
              + ("; ".join(tie_notes) if tie_notes else "none") + "."]
    L += ["", "Route check: the strict and content model means used for the reversals (the arrays of "
              "`run_baseline_mixture.run_task`) against the same means from the rows of the discovery draws, largest "
              "difference: " + ", ".join(f"{T['title']} {f_(max(T['route_check_max_abs_pp'].values()) if T.get('route_check_max_abs_pp') else None, 15)} pp"
                                         for T in B["tasks"].values() if T.get("route_check_max_abs_pp")) + "."]
    L += ["", "Beside them (JSON, `tables/threshold_sweep.csv`): the long share on the discovery draws, the variance "
              "terms var(long)/var(shift), var(short)/var(shift) and 2cov/var(shift), the reversed pairs between the "
              "observed gain and the uncorrected forced gain (they do not depend on the cutoff), ties, the Pearson "
              "correlation and the RMS error of the corrected gain, and the median, minimum and maximum of w.", "",
          "Per-model w at each cutoff, range over models (%):", "",
          "| Benchmark | " + " | ".join(f"{c} tokens: min / median / max" for c in CUTS) + " |", "|---|" + "---|" * len(CUTS)]
    for task, T in B["tasks"].items():
        cells = []
        for c in CUTS:
            r = T["by_cut"].get(str(c))
            cells.append("–" if r is None else f"{f_(r['min_w_pct'], 0)} / {f_(r['median_w_pct'], 0)} / {f_(r['max_w_pct'], 0)}")
        L.append(f"| {T['title']} | " + " | ".join(cells) + " |")
    ok = all(c["ok"] for c in B["checks"])
    L += ["", f"## Checks ({'all pass' if ok else 'FAILED'})", "",
          "The 16-token column against `results/baseline_mixture/baseline_mixture.json` and the macros of "
          "`paper/numbers.tex`; HotpotQA at 16 and 64 against `results/hotpot/decomposition_thresholds_post_hoc.json`, "
          "`\\hqDecVarLongAllPct...`, and at 64 (strict) the saved forecast `manifests/predictions_hotpot_mechanism.json`.",
          "", "| Check | Here | Stored | Same |", "|---|---|---|---|"]
    for c in B["checks"]:
        L.append(f"| {c['what']} | {c['here']} | {c['stored']} | {'yes' if c['ok'] else 'NO'} |")
    L += ["", f"HotpotQA's own reversal macros `\\hqDecRevCorr<Sixteen|SixtyFour>` count pairs on every distinct "
              f"configuration; the values here are on the discovery means, the basis of the other three benchmarks."]
    return "\n".join(L) + "\n"


def write_outputs(B):
    import macros_threshold_sweep as mts
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "threshold_sweep.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(B, fh, indent=1, ensure_ascii=False)
    pd.DataFrame(table_rows(B)).to_csv(os.path.join(OUT, "tables", "threshold_sweep.csv"), index=False)
    txt = summary(B)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    with open(TABLE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(mts.threshold_sweep_table(B))


# ============================================================================ main


def run(include_hotpot=True) -> dict:
    import macros_threshold_sweep as mts
    t0 = time.time()
    B = {"label": "exploratory", "created": time.strftime("%Y-%m-%d %H:%M"),
         "cuts": list(CUTS), "hotpot_cuts": list(HQ_CUTS),
         "definition": __doc__.split("\n\n")[1].strip(), "tasks": {}, "checks": []}
    raw = {}
    for task in TASKS:
        wide, panel, disc, val, G = task_inputs(task)
        T = {"title": TITLE[task], "models": panel, "M": len(panel), "cuts": list(CUTS), "by_cut": {}}
        for c in CUTS:
            T["by_cut"][str(c)] = sweep_cut(wide, panel, disc, val, G, c)
            print(f"{task}: cut {c} done ({time.time() - t0:.0f} s)", flush=True)
        # the strict and content means used here against the rows of the discovery draws (same quantities, another route)
        rows_g = {sc: gains_from_rows(T["by_cut"]["16"][sc]["discovery_rows_means"]) for sc in SCORES}
        T["route_check_max_abs_pp"] = {sc: max(float(np.max(np.abs(G[sc][a] - rows_g[sc][a]))) for a in G[sc])
                                       for sc in SCORES}
        B["tasks"][task] = T
        raw[task] = np.array(T["by_cut"]["16"]["w_discovery"], float)
    if include_hotpot:
        wide, panel, disc, val, G = hotpot_inputs()
        T = {"title": TITLE["hotpot"], "models": panel, "M": len(panel), "cuts": list(HQ_CUTS), "by_cut": {},
             "note": "HotpotQA's forced cap is 64 tokens; swept at 16 and 64. Model means of C - R and C - F from the "
                     "rows of the discovery draws, each configuration weighted by its draws."}
        for c in HQ_CUTS:
            T["by_cut"][str(c)] = sweep_cut(wide, panel, disc, val, G, c)
            print(f"hotpot: cut {c} done ({time.time() - t0:.0f} s)", flush=True)
        B["tasks"]["hotpot"] = T
        hq_w = {c: np.array(T["by_cut"][str(c)]["w_discovery"], float) for c in HQ_CUTS}
        hq_G = G
    macros = mts.threshold_sweep_macros(B)
    N = _numbers()
    R = json.load(open(os.path.join(ROOT, "results", "baseline_mixture", "baseline_mixture.json"), encoding="utf-8"))
    for task in TASKS:
        B["checks"] += check_task(task, B["tasks"][task]["by_cut"]["16"], raw[task], R, N, macros)
    if include_hotpot:
        H = json.load(open(os.path.join(ROOT, "results", "hotpot", "decomposition_thresholds_post_hoc.json"),
                           encoding="utf-8"))
        F = json.load(open(os.path.join(ROOT, "manifests", "predictions_hotpot_mechanism.json"), encoding="utf-8"))
        res = {c: B["tasks"]["hotpot"]["by_cut"][str(c)] for c in HQ_CUTS}
        B["checks"] += check_hotpot(res, hq_w, hq_G, H, F, N, macros)
    clash = sorted(k for k in macros if k in N and N[k] != macros[k])
    B["checks"].append({"what": "new macro names absent from paper/numbers.tex", "here": len(clash), "stored": 0,
                        "ok": not clash})
    B["n_macros"] = len(macros)
    B["runtime_s"] = round(time.time() - t0, 1)
    write_outputs(B)
    return B


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-hotpot", action="store_true", help="skip HotpotQA")
    a = ap.parse_args(argv)
    B = run(include_hotpot=not a.no_hotpot)
    bad = [c for c in B["checks"] if not c["ok"]]
    print(summary(B))
    print(f"{len(B['checks'])} checks, {len(bad)} failed; {B['n_macros']} macros; {B['runtime_s']} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
