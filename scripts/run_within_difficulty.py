r"""Does a model write long requested answers more often on harder questions, within itself? Existing outputs only; no generation.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_within_difficulty.py

Rows: scripts/run_baseline_mixture.task_rows (one row per model, configuration and item of GSM8K-29, MATH-500 numeric
and BIG-Bench Hard-28, with the requested output's n_output_tokens and the three arms' strict and content scores).
  long        n_output_tokens of the requested-direct output > the forced arm's cap (16 tokens on all three
              benchmarks; run_baseline_mixture.cap_of), the token counts and threshold behind w.
  draws       main: the 24 discovery draws, each configuration weighted by its number of draws (a configuration
              drawn twice counts twice), as w is averaged; so a model's weighted long share is its w (checked).
  difficulty  of item i for model m: 100 x the share of the other models' forced-direct answers on item i, over the
              same draws and weights, that are wrong (leave-this-model-out), in pp; larger is harder. Difficulty is
              100 minus the share correct, so an odds ratio above 1 means more long
              answers on harder items (on the share-correct scale the odds ratio is the reciprocal).
  fit         binomial logit of long on difficulty / 10 with one intercept per model and one common slope
              (src/within_difficulty.fe_logit); odds ratio per 10 points = exp(slope); 95% percentile interval from
              9,999 model resamples (models in the fit drawn with replacement, difficulty held as computed on the
              full panel). Models whose requested outputs are all long or all short have no finite intercept and are
              left out of the fit (listed). Within-model slopes: one logit per model (intercept and slope; in the
              subtask lines also one effect per subtask, leaving out the model's subtasks with no variation in long).
Lines: main (strict scoring of the forced answers that define difficulty); content (content scoring); all_distinct
(strict, every distinct configuration of both splits once); bbh_subtask and bbh_subtask_content (BIG-Bench Hard,
strict or content, discovery draws, with one effect per subtask added, so the slope compares items within a subtask).

Outputs: results/within_difficulty/within_difficulty.json, summary.md, summary_clean.md, tables/per_model.csv.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import within_difficulty as wd  # noqa: E402
from src.decision import label_seed  # noqa: E402

OUT = os.path.join(ROOT, "results", "within_difficulty")
TASKS = ("gsm8k", "math", "bbh")
TITLE = {"gsm8k": "GSM8K-29", "math": "MATH-500 numeric", "bbh": "BIG-Bench Hard-28"}
N_BOOT = 9999
LINES = {
    "main": "strict scoring of the forced answers; discovery draws, each configuration weighted by its draws",
    "content": "content scoring of the forced answers; discovery draws, each configuration weighted by its draws",
    "all_distinct": "strict scoring; every distinct configuration of both splits once",
    "bbh_subtask": "BIG-Bench Hard only; as main, with one effect per subtask",
    "bbh_subtask_content": "BIG-Bench Hard only; as content, with one effect per subtask",
}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def arrays(wide, panel, cap, weights: dict, score: str):
    """(K, N, F, items): (M, I) weighted counts of long requested outputs, outputs and correct forced answers.
    weights: {config_id: weight}; configurations not in it are left out."""
    x = wide[wide["config_id"].isin(set(weights))].copy()
    x["wt"] = x["config_id"].map(weights).astype(float)
    x["long_w"] = x["wt"] * (x["n_output_tokens"].to_numpy() > cap)
    x["fd_w"] = x["wt"] * x[f"forced_{score}"].astype(float)
    g = x.groupby(["model_key", "item_id"], sort=False)[["wt", "long_w", "fd_w"]].sum()
    items = sorted(x["item_id"].unique())
    idx = pd.MultiIndex.from_product([panel, items], names=["model_key", "item_id"])
    g = g.reindex(idx)
    if g.isna().any().any():
        raise SystemExit("a model lacks an item in the chosen configurations")
    sh = (len(panel), len(items))
    return (g["long_w"].to_numpy().reshape(sh), g["wt"].to_numpy().reshape(sh), g["fd_w"].to_numpy().reshape(sh),
            items)


def run_line(task, line, K, N, F, panel, items, group=None) -> dict:
    D = wd.loo_difficulty(F, N)
    if not np.all(np.isfinite(D)):
        raise SystemExit(f"{task}/{line}: a difficulty is not finite")
    X = D / 10.0
    var = wd.variation(K, N)
    fit_idx = np.flatnonzero(var == "varies")
    Kf, Nf, Xf = K[fit_idx], N[fit_idx], X[fit_idx]
    fit = wd.fe_logit(Kf, Nf, Xf, group)
    rng = np.random.default_rng(label_seed(f"within_difficulty/{task}/{line}"))
    boot = wd.model_bootstrap(Kf, Nf, Xf, N_BOOT, rng, group, fit=fit)
    pooled = wd.odds_ratio_summary(fit, boot)
    pm = wd.per_model_slopes(K, N, X, group)
    rows = []
    for j, m in enumerate(panel):
        s = pm[j]["slope"]
        rows.append({"model_key": m, "variation": pm[j]["variation"], "n_outputs": float(N[j].sum()),
                     "n_long": float(K[j].sum()), "long_share": float(K[j].sum() / N[j].sum()),
                     "mean_difficulty_pp": float(np.average(D[j], weights=N[j])),
                     "slope_per10": s, "odds_ratio_per10": (float(np.exp(s)) if s is not None else None),
                     "converged": pm[j]["converged"], "separation": pm[j]["separation"],
                     "iterations": pm[j]["iterations"], "groups_dropped": pm[j]["groups_dropped"]})
    with_slope = [r for r in rows if r["slope_per10"] is not None]
    full_acc = 100.0 * F.sum(0) / N.sum(0)
    return {
        "line": line, "description": LINES[line], "models_total": len(panel), "models_in_fit": int(len(fit_idx)),
        "items": len(items), "groups": (int(np.max(group)) + 1 if group is not None else None),
        "no_variation": [{"model_key": r["model_key"], "kind": r["variation"]} for r in rows if r["variation"] != "varies"],
        "pooled": pooled,
        "pooled_fit_converged": bool(fit["converged"]),
        "positive": sum(r["slope_per10"] > 0 for r in with_slope),
        "negative": sum(r["slope_per10"] < 0 for r in with_slope),
        "zero": sum(r["slope_per10"] == 0 for r in with_slope),
        "per_model_not_converged": [r["model_key"] for r in with_slope if not r["converged"]],
        "per_model_separation": [r["model_key"] for r in with_slope if r["separation"]],
        "difficulty": {"panel_forced_accuracy_mean_pp": float(full_acc.mean()),
                       "panel_forced_accuracy_sd_pp": float(full_acc.std(ddof=1)),
                       "items_all_panel_forced_wrong": int((F.sum(0) == 0).sum()),
                       "items_all_panel_forced_right": int((F.sum(0) == N.sum(0)).sum()),
                       "loo_difficulty_min_pp": float(D.min()), "loo_difficulty_max_pp": float(D.max())},
        "per_model": rows,
    }


def subtask_table(K, N, F, Fc, group, levels) -> dict:
    """Per BIG-Bench Hard subtask, over the whole panel and the main line's draws: forced accuracy under strict and
    content scoring and the long share of the requested answers (pp); Pearson correlations over subtasks."""
    n = np.bincount(group, N.sum(0))
    rows = [{"subtask": s, "forced_strict_pp": float(100 * np.bincount(group, F.sum(0))[j] / n[j]),
             "forced_content_pp": float(100 * np.bincount(group, Fc.sum(0))[j] / n[j]),
             "long_share_pp": float(100 * np.bincount(group, K.sum(0))[j] / n[j])} for j, s in enumerate(levels)]
    a = {k: np.array([r[k] for r in rows]) for k in ("forced_strict_pp", "forced_content_pp", "long_share_pp")}
    return {"rows": rows, "subtasks": len(rows),
            "corr_long_vs_forced_strict": float(np.corrcoef(a["long_share_pp"], a["forced_strict_pp"])[0, 1]),
            "corr_long_vs_forced_content": float(np.corrcoef(a["long_share_pp"], a["forced_content_pp"])[0, 1])}


def run_task(task) -> dict:
    import run_baseline_mixture as rbm
    cap = rbm.cap_of(task)
    wide, panel, disc, val = rbm.task_rows(task)
    w_disc = {c: float(n) for c, n in Counter(map(str, disc)).items()}
    w_all = {c: 1.0 for c in sorted(set(disc) | set(val))}
    out = {"title": TITLE[task], "forced_cap_tokens": cap, "models": panel,
           "discovery_draws": len(disc), "discovery_distinct_configurations": len(w_disc),
           "all_distinct_configurations": len(w_all), "lines": {}}
    K, N, F, items = arrays(wide, panel, cap, w_disc, "strict")
    w_ref = rbm.w_of(wide, panel, cap, disc)
    out["w_check_max_abs_diff"] = float(np.max(np.abs(K.sum(1) / N.sum(1) - w_ref)))
    if out["w_check_max_abs_diff"] > 1e-12:
        raise SystemExit(f"{task}: the weighted long share does not reproduce w ({out['w_check_max_abs_diff']})")
    out["lines"]["main"] = run_line(task, "main", K, N, F, panel, items)
    _, _, Fc, items_c = arrays(wide, panel, cap, w_disc, "content")
    assert items_c == items
    out["lines"]["content"] = run_line(task, "content", K, N, Fc, panel, items)
    Ka, Na, Fa, items_a = arrays(wide, panel, cap, w_all, "strict")
    out["lines"]["all_distinct"] = run_line(task, "all_distinct", Ka, Na, Fa, panel, items_a)
    if task == "bbh":
        sub = [i.split("/", 1)[0] for i in items]
        levels = sorted(set(sub))
        group = np.array([levels.index(s) for s in sub])
        out["lines"]["bbh_subtask"] = run_line(task, "bbh_subtask", K, N, F, panel, items, group)
        out["lines"]["bbh_subtask"]["subtasks"] = levels
        out["lines"]["bbh_subtask_content"] = run_line(task, "bbh_subtask_content", K, N, Fc, panel, items, group)
        out["subtask_table"] = subtask_table(K, N, F, Fc, group, levels)
    return out


# ============================================================================ report


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def group_line(L):
    return L.get("groups") is not None


def line_row(task, L):
    p = L["pooled"]
    nv = ", ".join(f"{r['model_key']} ({r['kind'].replace('_', ' ')})" for r in L["no_variation"]) or "none"
    nc = ", ".join(L["per_model_not_converged"]) or "none"
    return (f"| {TITLE[task]} | {f2(p['odds_ratio'])} | [{f2(p['odds_ratio_ci95'][0])}, {f2(p['odds_ratio_ci95'][1])}] "
            f"| {L['positive']} of {L['models_in_fit']} | {L['models_in_fit']} of {L['models_total']} | {nv} | {nc} |")


def summary_md(R) -> str:
    T = R["tasks"]
    s = ["# Long requested answers and item difficulty, within models", ""]
    m = {t: T[t]["lines"]["main"] for t in TASKS}
    s.append("Odds ratio of a long requested answer per 10 points of item difficulty, model fixed effects, strict "
             "scoring: " + "; ".join(f"{TITLE[t]} {f2(m[t]['pooled']['odds_ratio'])} "
                                     f"[{f2(m[t]['pooled']['odds_ratio_ci95'][0])}, "
                                     f"{f2(m[t]['pooled']['odds_ratio_ci95'][1])}], own slope positive in "
                                     f"{m[t]['positive']} of {m[t]['models_in_fit']} models" for t in TASKS) + ".")
    s += ["", "## Definitions", "",
          "- Long: a requested-direct output with more than 16 generated tokens (n_output_tokens > 16, the forced arm's "
          "cap), the token counts and threshold behind w. A model's long share on the main line's rows equals its w "
          "(largest difference " + ", ".join(f"{TITLE[t]} {T[t]['w_check_max_abs_diff']:.1e}" for t in TASKS) + ").",
          "- Difficulty of an item for a model: 100 x the share of the other models' forced-direct answers on that item "
          "that are wrong (leave-this-model-out), in points, over the same draws; larger is harder. It is 100 minus "
          "the share correct, so an odds ratio above 1 means more long answers on harder items; "
          "on the share-correct scale the odds ratio is the reciprocal and a positive slope here is a negative one there.",
          "- Draws: the 24 discovery draws, each configuration weighted by its number of draws, as w is averaged "
          f"(GSM8K {T['gsm8k']['discovery_distinct_configurations']}, MATH-500 "
          f"{T['math']['discovery_distinct_configurations']}, BIG-Bench Hard "
          f"{T['bbh']['discovery_distinct_configurations']} distinct configurations).",
          "- Fit: binomial logit of long on difficulty / 10, one intercept per model, one common slope; odds ratio = "
          f"exp(slope). Interval: 95% percentile over {N_BOOT:,} model resamples (the models in the fit drawn with "
          "replacement; difficulty held as computed on the full panel). A model whose requested outputs are all long "
          "or all short has no finite intercept and is left out of the fit. Own slope: one logit per model, intercept "
          "and slope (in the subtask lines also one effect per subtask, the model's subtasks with no variation in long "
          "left out of its fit).", ""]
    for line in LINES:
        rows = [(t, T[t]["lines"][line]) for t in TASKS if line in T[t]["lines"]]
        s += [f"## {line}: {LINES[line]}", "",
              "| Benchmark | Odds ratio per 10 points | 95% model-resampling interval | Own slope positive | Models in "
              "the fit | Left out (no variation in long) | Own fits not converged |",
              "|---|---:|---|---:|---:|---|---|"]
        s += [line_row(t, L) for t, L in rows]
        s.append("")
        notes, conv = [], True
        for t, L in rows:
            p = L["pooled"]
            if not L["pooled_fit_converged"] or p["boot_not_converged"] or p["boot_nonfinite"]:
                conv = False
                notes.append(f"{TITLE[t]}: pooled fit converged {L['pooled_fit_converged']}; of the {p['n_boot']:,} "
                             f"resamples, {p['boot_not_converged']} did not converge, "
                             f"{p['boot_not_converged_group_without_variation']} of them because a subtask has no long "
                             "answer (or only long answers) among the drawn models, so its effect has no finite "
                             "estimate; their slopes, at the last step, are kept in the interval; "
                             f"{p['boot_nonfinite']} not finite.")
            if group_line(L) and any(r["groups_dropped"] for r in L["per_model"] if r["groups_dropped"]):
                gd = [r["groups_dropped"] for r in L["per_model"] if r["groups_dropped"] is not None]
                notes.append(f"{TITLE[t]}: own fits with subtask effects leave out, per model, the subtasks whose "
                             f"requested outputs are all long or all short ({sum(x > 0 for x in gd)} of {len(gd)} "
                             f"models lose at least one; at most {max(gd)} of {L['groups']}).")
            if L["zero"]:
                notes.append(f"{TITLE[t]}: {L['zero']} own slopes exactly zero.")
        if conv:
            notes.insert(0, "Every pooled fit and all resampled fits converged.")
        s += ["Edge cases: " + " ".join(notes), ""]
    s += ["## Difficulty on each benchmark (main line)", "",
          "| Benchmark | Panel forced accuracy, item mean (pp) | SD over items (pp) | Items no model answers right "
          "(forced) | Items every model answers right |", "|---|---:|---:|---:|---:|"]
    for t in TASKS:
        d = T[t]["lines"]["main"]["difficulty"]
        s.append(f"| {TITLE[t]} | {d['panel_forced_accuracy_mean_pp']:.1f} | {d['panel_forced_accuracy_sd_pp']:.1f} | "
                 f"{d['items_all_panel_forced_wrong']} | {d['items_all_panel_forced_right']} |")
    st = T["bbh"]["subtask_table"]
    s += ["", "## BIG-Bench Hard subtasks (whole panel, main line's draws)", "",
          f"Over the {st['subtasks']} subtasks, the long share of the requested answers correlates "
          f"{st['corr_long_vs_forced_strict']:+.2f} with strict forced accuracy and {st['corr_long_vs_forced_content']:+.2f} "
          "with content forced accuracy (Pearson). Sorted by long share:", "",
          "| Subtask | Long share (%) | Forced accuracy, strict (%) | Forced accuracy, content (%) |",
          "|---|---:|---:|---:|"]
    s += [f"| {r['subtask']} | {r['long_share_pp']:.1f} | {r['forced_strict_pp']:.1f} | {r['forced_content_pp']:.1f} |"
          for r in sorted(st["rows"], key=lambda r: r["long_share_pp"])]
    s += ["", "Per model (long share, own odds ratio, convergence) in `tables/per_model.csv`, all lines.",
          f"Written by `scripts/run_within_difficulty.py` ({R['runtime_s']:.0f} s)."]
    return "\n".join(s) + "\n"


def main():
    t0 = time.time()
    R = {"request": "long requested answers and item difficulty", "n_boot": N_BOOT, "lines": LINES,
         "rule": __doc__.split("Outputs:")[0].strip(), "tasks": {}}
    for t in TASKS:
        print(f"[{t}] ...", flush=True)
        R["tasks"][t] = run_task(t)
        L = R["tasks"][t]["lines"]["main"]
        print(f"  OR {L['pooled']['odds_ratio']:.3f} {L['pooled']['odds_ratio_ci95']}, positive {L['positive']} of "
              f"{L['models_in_fit']}", flush=True)
    R["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "within_difficulty.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    rows = [{"task": t, "line": ln, **r} for t in TASKS for ln, L in R["tasks"][t]["lines"].items()
            for r in L["per_model"]]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    md = summary_md(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
