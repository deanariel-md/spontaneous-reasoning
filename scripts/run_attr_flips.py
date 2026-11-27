r"""Which part of the baseline shift makes the reversals. Saved outputs only; local CPU.

PowerShell (repository root):
    .\.venv\Scripts\python.exe scripts\run_attr_flips.py

For GSM8K, MATH-500 and BIG-Bench Hard (long: requested reply longer than the forced cap of 16 tokens) and HotpotQA
(long: more than 64 tokens, its forced cap, and more than 16), under strict and content scoring, on the discovery means:
the pairs reversed between the requested-baseline gain and (a) the forced-baseline gain as published, (b) the
requested-baseline gain plus the long part of D = requested - forced accuracy, (c) the requested-baseline gain plus the
short part; the pairs both parts reverse; the per-model median and range of the short part. Definitions:
src/attr_flips.py.

Inputs, all read with the code that made the published numbers:
  gains        GSM8K, MATH-500, BIG-Bench Hard: scripts/robustness_inputs.load (strict) and
               scripts/run_decision.task_split(task, "content", "discovery"), as scripts/run_baseline_mixture.run_task;
               HotpotQA: scripts/run_hotpot_content._arrays3 on the strict table and on the content view, as its
               arm_effect_block.
  parts        src/baseline_mixture.decomposition on the discovery draws, each configuration weighted by its number of
               draws: scripts/run_baseline_mixture.decomposition_block (three benchmarks) and src/hotpot_content's
               block on scripts/run_hotpot_content.wide_table (HotpotQA, columns long_cap64 and long_cap16).
Checks: the observed count equals the published one (the results files and paper/numbers.tex); the parts add to D
within 1e-8 pp; the parts equal the stored ones (results/baseline_mixture/baseline_mixture.json,
results/hotpot/decomposition_thresholds_post_hoc.json).

Outputs: results/attr_flips/attr_flips.json, summary.md, summary_clean.md, tables/counts.csv, tables/per_model.csv.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import attr_flips as af  # noqa: E402

OUT = os.path.join(ROOT, "results", "attr_flips")
SCORES = ("strict", "content")
MAIN = ("gsm8k", "math", "bbh")
HQ_CUTS = (("hotpot_cap64", "cap64", 64), ("hotpot_cap16", "cap16", 16))
TITLE = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard",
         "hotpot_cap64": "HotpotQA, cut at 64 tokens", "hotpot_cap16": "HotpotQA, cut at 16 tokens"}
UNITS = MAIN + tuple(k for k, _, _ in HQ_CUTS)
# published observed counts in paper/numbers.tex (strict name; content adds the suffix Content)
PUBLISHED_MACRO = {"gsm8k": "bmixGsmRevObs", "math": "bmixMathRevObs", "bbh": "bmixBbhRevObs",
                   "hotpot_cap64": "aeHqReversed", "hotpot_cap16": "aeHqReversed"}
STORED_TOL_PP = 1e-9


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def numbers_tex(path=None) -> dict:
    path = path or os.path.join(ROOT, "paper", "numbers.tex")
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8").read()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^\n]*)\}", txt))


def _load_json(*parts):
    return json.load(open(os.path.join(ROOT, *parts), encoding="utf-8"))


# ============================================================================ GSM8K, MATH-500, BIG-Bench Hard


def main_task(task) -> dict:
    """{score: {'g_req', 'g_fd', 'w', 'part_long', 'part_short', 'models', 'cut'}} on the discovery draws."""
    import robustness_inputs as ri
    import run_baseline_mixture as rbm
    import run_decision as rd
    cap = rbm.cap_of(task)
    wide, panel, disc, _ = rbm.task_rows(task)
    dec = rbm.decomposition_block(wide, panel, cap, disc, True)
    x = ri.load(task)
    if x["facts"]["panel"] != panel or x["facts"]["discovery_config_ids"] != disc:
        raise SystemExit(f"{task}: the strict arrays differ in panel or draws")
    d = x["discovery"]
    G = {"strict": {a: (d["cot"] - d[a]).mean(0) for a in ("requested", "forced")}}
    c = rd.task_split(task, "content", "discovery")
    if [str(u) for u in c["units"]] != disc or [str(k) for k in c["panel"]] != panel:
        raise SystemExit(f"{task}: the content arrays differ in panel or draws")
    G["content"] = {a: (np.asarray(c["cot"], float) - np.asarray(c[a], float)).mean(0) for a in ("requested", "forced")}
    out = {}
    for sc in SCORES:
        p = dec[sc]["per_model"]
        out[sc] = {"g_req": G[sc]["requested"], "g_fd": G[sc]["forced"], "w": np.array(p["w"], float),
                   "part_long": np.array(p["part_long"], float), "part_short": np.array(p["part_short"], float),
                   "C_minus_R_rows": np.array(p["C"], float) - np.array(p["R"], float)}
    return {"models": panel, "cut": cap, "discovery_draws": len(disc), "scores": out}


def stored_main(task, sc) -> dict:
    B = _load_json("results", "baseline_mixture", "baseline_mixture.json")
    p = B["decomposition"][task]["sets"]["discovery"][sc]["per_model"]
    return {"published": B["tasks"][task]["mechanism"][sc]["observed_vs_forced"]["reversed"],
            "published_source": "results/baseline_mixture/baseline_mixture.json, tasks.<task>.mechanism.<score>."
                                "observed_vs_forced.reversed",
            "models": B["decomposition"][task]["models"],
            "part_long": np.array(p["part_long"], float), "part_short": np.array(p["part_short"], float)}


# ============================================================================ HotpotQA


def hotpot_units() -> dict:
    """{unit: {...}} for the two cuts, each {score: {...}} as main_task."""
    import run_hotpot_content as rhc
    from src import hotpot_content as hc
    ctx = rhc.setup()
    panel, disc = list(ctx["panel"]), list(ctx["disc"])
    wide = rhc.wide_table(ctx)
    G = {}
    for sc, path in (("strict", rhc.STRICT_TABLE), ("content", os.path.join(rhc.VIEW, "scored.parquet"))):
        breq, bfd, pc = rhc._arrays3(path, ctx, disc)
        G[sc] = {"requested": (pc - breq).mean(0), "forced": (pc - bfd).mean(0)}
    out = {}
    for unit, d, cut in HQ_CUTS:
        S = {}
        for sc in SCORES:
            p = hc._block(wide, panel, f"long_{d}", disc, True, sc)["per_model"]
            S[sc] = {"g_req": G[sc]["requested"], "g_fd": G[sc]["forced"], "w": np.array(p["w"], float),
                     "part_long": np.array(p["part_long"], float), "part_short": np.array(p["part_short"], float),
                     "C_minus_R_rows": np.array(p["C"], float) - np.array(p["R"], float)}
        out[unit] = {"models": panel, "cut": cut, "definition_key": d, "discovery_draws": len(disc), "scores": S}
    return out


def stored_hotpot(unit, sc) -> dict:
    d = dict((k, dk) for k, dk, _ in HQ_CUTS)[unit]
    T = _load_json("results", "hotpot", "decomposition_thresholds_post_hoc.json")
    p = T["variants"][d][sc]["sets"]["discovery"]["per_model"]
    if sc == "strict":
        pub = _load_json("results", "arm_effect", "arm_effect.json")["tasks"]["hotpot"]["reversals"]["reversed"]
        src = "results/arm_effect/arm_effect.json, tasks.hotpot.reversals.reversed"
    else:
        pub = _load_json("results", "hotpot_content", "hotpot_content.json")["arm_effect"]["content"]["reversals"]["reversed"]
        src = "results/hotpot_content/hotpot_content.json, arm_effect.content.reversals.reversed"
    return {"published": pub, "published_source": src, "models": None,
            "part_long": np.array(p["part_long"], float), "part_short": np.array(p["part_short"], float)}


# ============================================================================ the analysis


def analyse(unit, U, stored, tex) -> dict:
    out = {"title": TITLE[unit], "cut_tokens": U["cut"], "M": len(U["models"]), "models": U["models"],
           "discovery_draws": U["discovery_draws"], "scores": {}}
    for sc in SCORES:
        s = U["scores"][sc]
        st = stored(unit, sc)
        if st["models"] is not None and list(st["models"]) != list(U["models"]):
            raise SystemExit(f"{unit} {sc}: the stored decomposition has another panel")
        flips = af.part_flips(s["g_req"], s["g_fd"], s["part_long"], s["part_short"], U["models"])
        mac = PUBLISHED_MACRO[unit] + ("Content" if sc == "content" else "")
        tex_val = tex.get(mac)
        checks = {
            "observed_equals_published_file": {"here": flips["observed"], "published": st["published"],
                                               "source": st["published_source"],
                                               "equal": flips["observed"] == st["published"]},
            "observed_equals_numbers_tex": {"here": flips["observed"], "macro": mac, "value": tex_val,
                                            "equal": tex_val is not None and str(flips["observed"]) == tex_val.strip()},
            "parts_add_to_D_max_abs_pp": flips["sum_check_max_abs_pp"],
            "parts_equal_stored_max_abs_pp": float(max(np.max(np.abs(s["part_long"] - st["part_long"])),
                                                       np.max(np.abs(s["part_short"] - st["part_short"])))),
            "row_gain_equals_published_gain_max_abs_pp": float(np.max(np.abs(s["C_minus_R_rows"] - s["g_req"])))}
        if not checks["observed_equals_published_file"]["equal"]:
            raise SystemExit(f"{unit} {sc}: observed {flips['observed']} differs from the published {st['published']}")
        if tex and not checks["observed_equals_numbers_tex"]["equal"]:
            raise SystemExit(f"{unit} {sc}: observed {flips['observed']} differs from \\{mac} = {tex_val}")
        if checks["parts_equal_stored_max_abs_pp"] > STORED_TOL_PP:
            raise SystemExit(f"{unit} {sc}: the parts differ from the stored decomposition by "
                             f"{checks['parts_equal_stored_max_abs_pp']:.3g} pp")
        D = s["g_fd"] - s["g_req"]
        out["scores"][sc] = {
            "counts": flips, "checks": checks,
            "short_part": af.distribution(s["part_short"]), "long_part": af.distribution(s["part_long"]),
            "D": af.distribution(D), "w": af.distribution(s["w"]),
            "per_model": {k: {"w": float(s["w"][j]), "gain_requested": float(s["g_req"][j]),
                              "gain_forced": float(s["g_fd"][j]), "D": float(D[j]),
                              "part_long": float(s["part_long"][j]), "part_short": float(s["part_short"][j])}
                          for j, k in enumerate(U["models"])}}
    return out


def run(units=UNITS) -> dict:
    t0 = time.time()
    tex = numbers_tex()
    R = {"analysis": "which part of the baseline shift makes the reversals",
         "created": time.strftime("%Y-%m-%d %H:%M"), "definition": af.DEFINITION,
         "units_pp": "gains, D and its parts in percentage points; w as a share in [0, 1]",
         "means": "model means over the 24 discovery draws (a configuration drawn twice counts twice)",
         "choices": [
             "The parts are computed on the discovery draws with each configuration weighted by its number of draws, "
             "so that they add exactly to D on the discovery means. results/baseline_mixture/tables/"
             "decomposition_per_model.csv pools every distinct configuration once and does not add to the discovery D. "
             "This is the same function and the same cut; nothing is fitted.",
             "The long cut is the forced cap, 16 tokens, on GSM8K, MATH-500 and BIG-Bench Hard (as in "
             "results/baseline_mixture/), and 64 and 16 tokens on HotpotQA."],
         "tasks": {}}
    for t in [u for u in units if u in MAIN]:
        R["tasks"][t] = analyse(t, main_task(t), lambda u, sc: stored_main(u, sc), tex)
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    hq = [u for u in units if u.startswith("hotpot")]
    if hq:
        H = hotpot_units()
        for u in hq:
            R["tasks"][u] = analyse(u, H[u], stored_hotpot, tex)
            print(f"{u}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_s"] = round(time.time() - t0, 1)
    return R


# ============================================================================ outputs


def f1(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.1f}".replace("-", "−")


def tables(R):
    rows, pm = [], []
    for u, T in R["tasks"].items():
        for sc, B in T["scores"].items():
            c, s = B["counts"], B["short_part"]
            rows.append({"unit": u, "benchmark": T["title"], "cut_tokens": T["cut_tokens"], "scores": sc,
                         "models": T["M"], "pairs": c["pairs"], "observed": c["observed"], "long_part_alone": c["long"],
                         "short_part_alone": c["short"], "both_parts": c["both"],
                         "observed_and_long": c["observed_and_long"], "observed_and_short": c["observed_and_short"],
                         "observed_neither": c["observed_neither"], "ties_observed": c["ties"]["observed"],
                         "ties_long": c["ties"]["long"], "ties_short": c["ties"]["short"],
                         "short_part_median_pp": s["median"], "short_part_min_pp": s["min"],
                         "short_part_max_pp": s["max"], "long_part_median_pp": B["long_part"]["median"],
                         "long_part_min_pp": B["long_part"]["min"], "long_part_max_pp": B["long_part"]["max"],
                         "w_median": B["w"]["median"]})
            for k, v in B["per_model"].items():
                pm.append({"unit": u, "scores": sc, "model_key": k, **v})
    return pd.DataFrame(rows), pd.DataFrame(pm)


def summary(R) -> list:
    L = ["# Which part of the baseline shift makes the reversals", "",
         "Code: `scripts/run_attr_flips.py`, `src/attr_flips.py`.", "",
         "## Definition", "",
         "A model's gain under the requested baseline is chain-of-thought accuracy minus requested accuracy, C − R. "
         "Its gain under the forced baseline is C − F, which equals C − R plus D, where D = R − F. D splits exactly "
         "into a long part, from the rows whose requested reply is longer than the cut, and a short part, from the "
         "other rows. Replacing D by its long part gives a forced-baseline gain of C − R plus the long part, which is "
         "C − F minus the short part: the switch of baseline then changes accuracy only on the rows with a long "
         "requested reply. Replacing D by its short part gives C − R plus the short part, which is C − F minus the "
         "long part.", "",
         "A pair of models is reversed when the requested-baseline gain orders it one way and the forced-baseline gain "
         "(observed or replaced) orders it the other way. A pair tied within 1e-9 pp under either gain is not "
         "reversed and stays in the denominator. \"Both parts\" counts the pairs that each part reverses on its own. "
         "Such a pair is always reversed by the full D as well, because the two parts then push the gap the same way "
         "and each alone already crosses zero.", "",
         "All values are model means over the 24 discovery draws (a configuration drawn twice counts twice).", "",
         "## Choices not fixed by the request", ""] + [f"- {c}" for c in R["choices"]] + [
         "", "## Pairs reversed by the switch of baseline", "",
         "Observed, long part alone, short part alone and both parts are the four counts. The next columns split the "
         "observed pairs by the part that also reverses them on its own. A pair reversed by neither part alone needs "
         "the two parts together. The last column is the short part of D over models, in pp.", "",
         "| Benchmark | Scoring | Models | Pairs | Observed | Long part alone | Short part alone | Both parts | "
         "Observed, long also | Observed, short also | Observed, neither alone | Short part, median [min, max] |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for u, T in R["tasks"].items():
        for sc, B in T["scores"].items():
            c, s = B["counts"], B["short_part"]
            L.append(f"| {T['title']} | {sc} | {T['M']} | {c['pairs']} | {c['observed']} | {c['long']} | {c['short']} | "
                     f"{c['both']} | {c['observed_and_long']} | {c['observed_and_short']} | {c['observed_neither']} | "
                     f"{f1(s['median'])} [{f1(s['min'])}, {f1(s['max'])}] |")
    L += ["", "Tied pairs (under either gain) for observed, long part, short part: " + "; ".join(
        f"{T['title']} {sc} {B['counts']['ties']['observed']}, {B['counts']['ties']['long']}, "
        f"{B['counts']['ties']['short']}" for T in R["tasks"].values() for sc, B in T["scores"].items()) + ".", "",
        "## The long part and w, for reference", "",
        "| Benchmark | Scoring | Long part, median [min, max] (pp) | D, median [min, max] (pp) | w, median |",
        "|---|---|---|---|---:|"]
    for u, T in R["tasks"].items():
        for sc, B in T["scores"].items():
            lp, D = B["long_part"], B["D"]
            L.append(f"| {T['title']} | {sc} | {f1(lp['median'])} [{f1(lp['min'])}, {f1(lp['max'])}] | "
                     f"{f1(D['median'])} [{f1(D['min'])}, {f1(D['max'])}] | {B['w']['median']:.3f} |")
    L += ["", "## Checks", ""]
    for u, T in R["tasks"].items():
        for sc, B in T["scores"].items():
            k = B["checks"]
            a, b = k["observed_equals_published_file"], k["observed_equals_numbers_tex"]
            L.append(f"- {T['title']}, {sc}: observed {a['here']} equals the published {a['published']} "
                     f"({a['source']}) and `\\{b['macro']}` = {b['value']}. The parts add to D within "
                     f"{k['parts_add_to_D_max_abs_pp']:.1e} pp and equal the stored parts within "
                     f"{k['parts_equal_stored_max_abs_pp']:.1e} pp.")
    L += ["", "Tables: `tables/counts.csv`, `tables/per_model.csv`. Pair lists: `attr_flips.json`.", "",
          f"Runtime {R['runtime_s']:.0f} s."]
    return L


def write(R):
    import robustness_inputs as ri
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    a, b = tables(R)
    a.to_csv(os.path.join(OUT, "tables", "counts.csv"), index=False)
    b.to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    R = run()
    write(R)
    print("\n".join(summary(R)))
    return R


if __name__ == "__main__":
    main()
