r"""Fig. 2c on templates the counts were not fitted on.

Fig. 2c counts the model pairs reversed against the gain over the requested baseline, first by the gain over the forced
baseline and then by the corrected gain (1 - w) x gain over the forced baseline. The published counts
(results/baseline_mixture/baseline_mixture.json, key mechanism; macros bmix<T>RevObs and bmix<T>RevCorr) use the 24
discovery draws for the gains and for w. This script:
  1. recomputes the published discovery counts with the same code (scripts/run_baseline_mixture.py: task_rows, w_of;
     src/baseline_mixture.mechanism; strict arrays from robustness_inputs.load, content arrays from
     run_decision.task_split) and stops unless every count equals the saved one and the macro in paper/numbers.tex;
  2. finds, from the saved samplers (manifests/sampler_<task>.json), the validation draws whose configuration no
     discovery draw used, and counts them and their templates;
  3. recounts Fig. 2c on those draws: both gains are model means over those draws, w stays the discovery draws' w;
  4. gives, for reference, the same on all 24 validation draws.
GSM8K-29, MATH-500 and BIG-Bench Hard-28, strict and content scoring. Every pair is in the denominator; a pair tied
under either gain (|gap| <= 1e-9 pp) is neither reversed nor kept, and the ties are counted.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_bmix_val.py

Outputs: results/bmix_val/bmix_val.json, summary.md, summary_clean.md, tables/counts.csv, tables/per_model.csv.
Macros: paper/macros_bmix_val.py (prefix bmixVal).
"""
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
from src import bmix_val as bv  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "bmix_val")
SAVED = os.path.join(ROOT, "results", "baseline_mixture", "baseline_mixture.json")
NUMBERS = os.path.join(ROOT, "paper", "numbers.tex")
TASKS = ("gsm8k", "math", "bbh")          # Fig. 2c's order
TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
TITLE = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard"}
SCORES = (("strict", ""), ("content", "Content"))
COMPARISONS = (("observed_vs_forced", "Obs"), ("observed_vs_corrected", "Corr"))
SETS = ("discovery", "fresh_validation", "all_validation")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def numbers_tex(path=NUMBERS) -> dict:
    """{name: value} of every \\newcommand in paper/numbers.tex."""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def split_arrays(task, scoring, split, panel, ids):
    """{'requested', 'forced', 'cot'} (K, M) pp for a split, checked against the panel and the sampler's draws."""
    if scoring == "strict":
        x = ri.load(task)
        f = x["facts"]
        if f["panel"] != panel or f[f"{split}_config_ids"] != ids:
            raise SystemExit(f"{task}: the strict arrays differ in panel or {split} draws")
        A = x[split]
    else:
        import run_decision as rd
        c = rd.task_split(task, "content", split)
        if [str(u) for u in c["units"]] != ids or [str(k) for k in c["panel"]] != panel:
            raise SystemExit(f"{task}: the content arrays differ in panel or {split} draws")
        A = {a: np.asarray(c[a], float) for a in bv.ARMS}
    for a in bv.ARMS:
        if A[a].shape != (len(ids), len(panel)):
            raise SystemExit(f"{task} {scoring} {split} {a}: shape {A[a].shape}, expected {(len(ids), len(panel))}")
    return A


def reproduction(task, sc, suf, got, saved_mech, mac) -> dict:
    """The recomputed discovery counts against the saved json and the published macros; every field listed."""
    T = TAG[task]
    rows = []
    for k, tag in COMPARISONS:
        g, s = got[k], saved_mech[k]
        for f in ("pairs", "reversed", "ties"):
            rows.append({"what": f"{k}.{f}", "recomputed": g[f], "saved": s[f], "equal": g[f] == s[f]})
        rows.append({"what": f"{k}.reversed_share", "recomputed": g["reversed_share"], "saved": s["reversed_share"],
                     "equal": abs(g["reversed_share"] - s["reversed_share"]) < 1e-12})
        name = f"bmix{T}Rev{tag}{suf}"
        rows.append({"what": f"macro {name}", "recomputed": g["reversed"], "saved": mac.get(name),
                     "equal": mac.get(name) == str(g["reversed"])})
        name = f"bmix{T}Rev{tag}Pct{suf}"
        pct = f"{100 * g['reversed_share']:.0f}"
        rows.append({"what": f"macro {name}", "recomputed": pct, "saved": mac.get(name), "equal": mac.get(name) == pct})
    name = f"bmix{T}Pairs{suf}"
    rows.append({"what": f"macro {name}", "recomputed": got["observed_vs_forced"]["pairs"], "saved": mac.get(name),
                 "equal": mac.get(name) == str(got["observed_vs_forced"]["pairs"])})
    ok = all(r["equal"] for r in rows)
    if not ok:
        bad = [r for r in rows if not r["equal"]]
        raise SystemExit(f"{task} {sc}: the discovery counts do not reproduce: {bad}")
    return {"reproduced": ok, "checks": rows}


def run_task(task, saved, mac):
    import json

    import run_baseline_mixture as rbm
    t0 = time.time()
    cap = rbm.cap_of(task)
    wide, panel, disc, val = rbm.task_rows(task)
    s = json.load(open(os.path.join(ROOT, "manifests", f"sampler_{task}.json"), encoding="utf-8"))
    if [str(u) for u in s["discovery_config_ids"]] != disc or [str(u) for u in s["validation_config_ids"]] != val:
        raise SystemExit(f"{task}: the sampler's draws differ from those of the scored rows")
    w = rbm.w_of(wide, panel, cap, disc)
    del wide
    if np.isnan(w).any():
        raise SystemExit(f"{task}: a model lacks requested outputs in a discovery configuration")
    w_saved = np.asarray(saved["w_discovery"], float)
    if saved["models"] != panel or not np.allclose(w, w_saved, rtol=0, atol=1e-12):
        raise SystemExit(f"{task}: w from the discovery draws differs from the saved w")
    fd = bv.fresh_draws(disc, val)
    out = {"title": TITLE[task], "M": len(panel), "models": panel, "forced_cap_tokens": cap,
           "sampler": {"path": f"manifests/sampler_{task}.json", "sha256": s.get("sha256")},
           "draws": {k: v for k, v in fd.items() if k != "index"},
           "fresh_draw_positions": fd["index"], "w_discovery": w.tolist(), "scores": {}}
    for sc, suf in SCORES:
        A = {sp: split_arrays(task, sc, sp, panel, ids) for sp, ids in (("discovery", disc), ("validation", val))}
        d = bv.recount(A["discovery"], w)
        rep = reproduction(task, sc, suf, d, saved["mechanism"][sc], mac)
        f = bv.recount(A["validation"], w, fd["index"])
        a = bv.recount(A["validation"], w)
        for name, m in (("discovery", d), ("fresh_validation", f), ("all_validation", a)):
            if m["models_with_nan"]:
                raise SystemExit(f"{task} {sc} {name}: {m['models_with_nan']} models without a gain")
        out["scores"][sc] = {"discovery": d, "fresh_validation": f, "all_validation": a, "reproduction": rep,
                             "fresh_minus_discovery": bv.difference(f, d),
                             "all_minus_discovery": bv.difference(a, d)}
    print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    return out


# ============================================================================ outputs


def counts_table(R) -> pd.DataFrame:
    rows = []
    for t, T in R["tasks"].items():
        for sc, S in T["scores"].items():
            for st in SETS:
                m = S[st]
                n_draws = {"discovery": T["draws"]["discovery_draws"], "fresh_validation": T["draws"]["draws"],
                           "all_validation": T["draws"]["validation_draws"]}[st]
                n_tpl = {"discovery": T["draws"]["discovery_templates"], "fresh_validation": T["draws"]["templates"],
                         "all_validation": T["draws"]["validation_templates"]}[st]
                rows.append({"task": t, "scores": sc, "draws_set": st, "draws": n_draws, "templates": n_tpl,
                             "pairs": m["observed_vs_forced"]["pairs"],
                             "reversed_vs_forced_gain": m["observed_vs_forced"]["reversed"],
                             "share_vs_forced_gain": m["observed_vs_forced"]["reversed_share"],
                             "ties_vs_forced_gain": m["observed_vs_forced"]["ties"],
                             "reversed_vs_corrected_gain": m["observed_vs_corrected"]["reversed"],
                             "share_vs_corrected_gain": m["observed_vs_corrected"]["reversed_share"],
                             "ties_vs_corrected_gain": m["observed_vs_corrected"]["ties"],
                             "pearson_forced": m["corr_forced"]["pearson"],
                             "pearson_corrected": m["corr_corrected"]["pearson"]})
    return pd.DataFrame(rows)


def per_model_table(R) -> pd.DataFrame:
    rows = []
    for t, T in R["tasks"].items():
        for sc, S in T["scores"].items():
            for j, k in enumerate(T["models"]):
                r = {"task": t, "scores": sc, "model_key": k, "w_discovery": T["w_discovery"][j]}
                for st in SETS:
                    m = S[st]
                    r[f"{st}_gain_vs_requested"] = m["gain_vs_requested"][j]
                    r[f"{st}_gain_vs_forced"] = m["gain_vs_forced"][j]
                    r[f"{st}_corrected_gain"] = m["corrected_gain"][j]
                rows.append(r)
    return pd.DataFrame(rows)


def pct(x):
    return f"{100 * x:.0f}%"


def signed(n):
    return f"+{n}" if n > 0 else (f"−{-n}" if n < 0 else "0")


def moved(n):
    """'goes up by 2', 'goes down by 1', 'is unchanged'."""
    return f"goes up by {n}" if n > 0 else (f"goes down by {-n}" if n < 0 else "is unchanged")


RULE = (
    "Fig. 2c counts the model pairs that two gains order in opposite directions. The first gain is the gain over the "
    "requested baseline (chain of thought minus requested-direct accuracy). The second is the gain over the forced "
    "baseline (chain of thought minus forced-direct accuracy), or the corrected gain, (1 − w) times the gain over the "
    "forced baseline. w is the model's share of requested-direct outputs longer than the forced cap (16 tokens), from "
    "the discovery draws. Every pair of the panel is in the denominator. A pair whose gap is within 1e-9 pp of zero "
    "under either gain is a tie: it is neither reversed nor kept, and it stays in the denominator. Here both gains are "
    "model means over the validation draws whose configuration no discovery draw used. w is the discovery draws' w, "
    "unchanged; nothing is fitted again.")


def summary(R) -> list:
    L = ["# Fig. 2c on templates the counts were not fitted on", "",
         "Code: `scripts/run_bmix_val.py`, `src/bmix_val.py`; numbers: "
         "`bmix_val.json`, `tables/counts.csv`, `tables/per_model.csv`; macros: `paper/macros_bmix_val.py` (prefix "
         "`bmixVal`).", "", "## Definition", "", RULE, "",
         "Two choices the request leaves open. A template drawn more than once among the fresh validation draws counts "
         "once per draw, as the discovery means count repeats. The counts on all 24 validation draws are given beside "
         "them for reference. Neither is a refit.", "",
         "## Check: the published discovery counts", ""]
    for t, T in R["tasks"].items():
        for sc, S in T["scores"].items():
            d = S["discovery"]
            L.append(f"- {T['title']}, {sc}: {d['observed_vs_forced']['reversed']} and "
                     f"{d['observed_vs_corrected']['reversed']} of {d['observed_vs_forced']['pairs']} pairs, equal to "
                     f"the saved counts and to `\\bmix{TAG[t]}RevObs{'Content' if sc == 'content' else ''}` and "
                     f"`\\bmix{TAG[t]}RevCorr{'Content' if sc == 'content' else ''}` "
                     f"({len(S['reproduction']['checks'])} values checked, all equal).")
    L += ["", "## Draws and templates", "",
          "From the saved samplers (`manifests/sampler_<task>.json`): 24 discovery and 24 validation draws, with "
          "replacement.", "",
          "| Benchmark | Discovery templates | Validation templates | Validation draws on a discovery template | "
          "Fresh draws | Fresh templates | Fresh templates drawn more than once |", "|---|---:|---:|---:|---:|---:|---|"]
    for t, T in R["tasks"].items():
        D = T["draws"]
        rep = ", ".join(f"{u} ({D['template_draws'][u]})" for u in D["repeated_templates"]) or "none"
        L.append(f"| {T['title']} | {D['discovery_templates']} | {D['validation_templates']} | {D['overlap_draws']} | "
                 f"{D['draws']} | {D['templates']} | {rep} |")
    L += ["", "## Reversed pairs", "",
          "Pairs reversed against the gain over the requested baseline, of all pairs (share in brackets). Discovery: "
          "the published Fig. 2c counts. Fresh: the validation draws on templates no discovery draw used.", "",
          "| Benchmark | Scores | Pairs | By forced gain, discovery | By forced gain, fresh | Change | "
          "By corrected gain, discovery | By corrected gain, fresh | Change | By forced gain, all 24 validation | "
          "By corrected gain, all 24 validation |", "|---|---|---:|---|---|---:|---|---|---:|---|---|"]
    for t, T in R["tasks"].items():
        for sc, S in T["scores"].items():
            d, f, a = S["discovery"], S["fresh_validation"], S["all_validation"]
            df_ = S["fresh_minus_discovery"]

            def c(m, k):
                return f"{m[k]['reversed']} ({pct(m[k]['reversed_share'])})"
            L.append(f"| {T['title']} | {sc} | {d['observed_vs_forced']['pairs']} | {c(d, 'observed_vs_forced')} | "
                     f"{c(f, 'observed_vs_forced')} | {signed(df_['observed_vs_forced']['reversed'])} | "
                     f"{c(d, 'observed_vs_corrected')} | {c(f, 'observed_vs_corrected')} | "
                     f"{signed(df_['observed_vs_corrected']['reversed'])} | {c(a, 'observed_vs_forced')} | "
                     f"{c(a, 'observed_vs_corrected')} |")
    ties = [(T["title"], sc, st, k, S[st][k]["ties"]) for t, T in R["tasks"].items() for sc, S in T["scores"].items()
            for st in SETS for k, _ in COMPARISONS if S[st][k]["ties"]]
    L += ["", "Ties: " + ("none in any count." if not ties else
                          "; ".join(f"{a}, {b}, {c_}, {k}: {n}" for a, b, c_, k, n in ties) + ".")]
    L += ["", "## What changes", ""] + R["reading"]
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def reading(R) -> list:
    """Plain sentences on whether the fresh counts differ from the published ones, generated from the numbers."""
    out = []
    for t, T in R["tasks"].items():
        for sc, S in T["scores"].items():
            d, f = S["discovery"], S["fresh_validation"]
            o0, o1 = d["observed_vs_forced"]["reversed"], f["observed_vs_forced"]["reversed"]
            c0, c1 = d["observed_vs_corrected"]["reversed"], f["observed_vs_corrected"]["reversed"]
            n = d["observed_vs_forced"]["pairs"]
            if o1 == o0 and c1 == c0:
                ch = "Both counts are the same as on the discovery draws."
            else:
                ch = f"Against the forced gain the count {moved(o1 - o0)}. Against the corrected gain it {moved(c1 - c0)}."
            lower = ("The corrected gain leaves fewer reversed pairs than the forced gain"
                     if c1 < o1 else "The corrected gain leaves as many reversed pairs as the forced gain"
                     if c1 == o1 else "The corrected gain leaves more reversed pairs than the forced gain")
            line = (f"- {T['title']}, {sc}: {o1} of {n} pairs reversed by the forced gain and {c1} by the corrected "
                    f"gain on {T['draws']['draws']} fresh draws, against {o0} and {c0} on the discovery draws. {ch} "
                    f"{lower} on the fresh draws.")
            if np.sign(c1 - o1) != np.sign(c0 - o0):
                word = "fewer" if c0 < o0 else "as many" if c0 == o0 else "more"
                line += f" On the discovery draws it left {word} ({c0} against {o0})."
            out.append(line)
    return out


def run(tasks=TASKS):
    import json
    t0 = time.time()
    saved = json.load(open(SAVED, encoding="utf-8"))
    mac = numbers_tex()
    R = {"analysis": "Fig. 2c's reversal counts on validation draws whose templates no discovery draw used", "created": time.strftime("%Y-%m-%d %H:%M"), "rule": RULE,
         "choices": ["a fresh template drawn more than once counts once per draw, as the discovery means count repeats",
                     "both gains from the fresh validation draws; w from the discovery draws, unchanged",
                     "all 24 validation draws given for reference"],
         "sources": {"saved_counts": rel(SAVED), "saved_created": saved.get("created"), "numbers": rel(NUMBERS)},
         "tasks": {}}
    for t in tasks:
        R["tasks"][t] = run_task(t, saved["tasks"][t], mac)
    R["reading"] = reading(R)
    R["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    counts_table(R).to_csv(os.path.join(OUT, "tables", "counts.csv"), index=False)
    per_model_table(R).to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    lines = summary(R)
    ri.write_outputs(OUT, R, lines)
    print("\n".join(lines))
    return R


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run()
