r"""Intervals for the flagship model's gains (Qwen4-Max).

Data: the 100 questions per benchmark of results/frontier_panel/closed_w.json, rebuilt from its row-level replies the
way `scripts/closed_w.py report` builds them (data/frontier_panel/closed_w/run_raw.jsonl, the first answered result
per key under the run's configuration hash, answered rows scored with the panel's scorers through
closed_w.score_rows). Per benchmark and scoring: the (templates, items) matrices of the three arms' scores and of the
long requested answers (the API's visible output count above the forced cap, closed_w.arm_mats on "n_api").

Quantities (accuracies as shares, pp when printed), each a weighted mean over templates with the discovery draws'
weights (manifests/sampler_<task>.json; a template drawn twice counts twice), as closed_w.model_quantities:
  requested-baseline gain   C - R
  forced-baseline gain      C - F
  difference                (C - F) - (C - R) = R - F, the forced-baseline gain minus the requested-baseline gain
  corrected gain            (1 - w) x (C - F), w and C - F each averaged over templates first
Intervals: 95% percentile intervals from 2,000 draws of the questions (config/closed_w.yaml, bootstrap), resampled
with replacement, the same draw for every arm and matrix (paired); on BIG-Bench Hard within task (the item id's prefix);
templates and their weights held fixed. The draws are made exactly as closed_w.boot_quantities makes them, with the
seed the report used (closed_w/bootstrap3/<model>/<task>/<score>), so the intervals of the two gains and the corrected
gain are the ones closed_w.json already holds; the difference is added from the same draws. The script checks both:
point values equal to closed_w.json's gains, the three intervals equal to its stored ones.

Benchmarks: GSM8K, MATH-500 and BIG-Bench Hard, strict scoring; BIG-Bench Hard also content scoring.

Writes results/frontier_panel/closed_w_intervals.json (with the matrices, so the intervals can be recomputed from the
file alone), closed_w_intervals.md and closed_w_intervals_clean.md.

  .venv\Scripts\python.exe scripts\closed_w_intervals.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import closed_w as cw  # noqa: E402

MODEL = "qwen4-max"
CELLS = (("gsm8k", "strict"), ("math", "strict"), ("bbh", "strict"), ("bbh", "content"))
BASIS = "discovery"
OUT = ROOT / "results" / "frontier_panel" / "closed_w_intervals"
STORED = ("cot_minus_requested", "cot_minus_forced", "corrected_gain")
QUANT = {"req_gain": "cot_minus_requested", "fd_gain": "cot_minus_forced", "gain_diff": None,
         "corrected_gain": "corrected_gain"}
LABEL = {"req_gain": "Requested-baseline gain (C − R)", "fd_gain": "Forced-baseline gain (C − F)",
         "gain_diff": "Difference (C − F) − (C − R) = R − F", "corrected_gain": "Corrected gain (1 − w)(C − F)",
         "w": "w (requested answers over the forced cap)"}


def seed_label(model: str, task: str, score: str) -> str:
    return f"closed_w/bootstrap3/{model}/{task}/{score}"


def strata_of(task: str, items) -> list[str]:
    return [i.split("/", 1)[0] if task == "bbh" else "all" for i in items]


# ---------------------------------------------------------------------------------------------------- rows

def load_scored(cfg: dict, paths, man: dict) -> tuple[pd.DataFrame, dict, str]:
    """The scored rows of closed_w.build_report (same steps, n_api only): returns (scored, scorer versions, hash)."""
    lines = cw.read_raw(paths.raw("run"))
    if not lines:
        raise SystemExit(f"{cw.rel(paths.raw('run'))} holds no replies")
    shas = {x.get("config_sha") for x in lines if x.get("event") == "result"}
    if len(shas) != 1:
        raise SystemExit(f"the run's raw file holds replies under {len(shas)} configurations")
    csha = shas.pop()
    fin = cw.final_by_key(lines, csha)
    acc = [dict(v, arm=cw.row_arm(v)) for v in fin.values() if v.get("outcome") in cw.ACCEPTED]
    df = cw.frame_of({i: v for i, v in enumerate(acc)})
    if "stop_sent" not in df:
        df["stop_sent"] = False
    df["n_api"] = [cw.api_count(x) for x in df.to_dict("records")]
    answered = df[df["outcome"].isin(("reply", "refusal"))].copy()
    scored, versions = cw.score_rows(answered, cfg)
    return scored, versions, csha


def cell_mats(scored: pd.DataFrame, cfg: dict, man: dict, task: str, score: str, model: str = MODEL) -> dict:
    t = man["tasks"][task]
    s = scored[(scored["model_key"] == model) & (scored["task"] == task)]
    return cw.arm_mats(s, t["templates"], t["items"], cw.SCORE_OF[score], "n_api", cw.forced_cap_of(cfg, task))


# ---------------------------------------------------------------------------------------------------- bootstrap

def draw_means(mats: dict, strata, draws: int, seed: int, chunk: int = 250) -> dict:
    """Per draw and template, the mean over resampled items of each matrix: {k: (draws, templates)}. The draws are
    made as in closed_w.boot_quantities (same generator, same order of strata, same chunks)."""
    C, N = next(iter(mats.values())).shape
    rng = np.random.default_rng(seed)
    groups = {}
    for j, s in enumerate(strata):
        groups.setdefault(s, []).append(j)
    idx = np.empty((draws, N), dtype=np.int64)
    for cols in groups.values():
        cols = np.asarray(cols)
        idx[:, cols] = rng.choice(cols, size=(draws, len(cols)), replace=True)
    per = {k: np.empty((draws, C)) for k in mats}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s0 in range(0, draws, chunk):
            sl = idx[s0:s0 + chunk]
            for k, M in mats.items():
                per[k][s0:s0 + chunk] = np.nanmean(M[:, sl], axis=2).T
    return per


def quantities(r, f, c, w) -> dict:
    """closed_w.quantities_from plus the difference of the two gains (forced-baseline minus requested-baseline)."""
    q = cw.quantities_from(r, f, c, w)
    q["gain_diff"] = q["cot_minus_forced"] - q["cot_minus_requested"]
    return q


def percentile(v: np.ndarray, level: float) -> dict:
    fin = v[np.isfinite(v)]
    a = (1 - level) / 2
    lo, hi = (float(x) for x in np.quantile(fin, [a, 1 - a])) if len(fin) else (None, None)
    return {"lo": lo, "hi": hi, "finite_draws": int(len(fin)), "draws": int(len(v))}


def intervals(mats: dict, strata, wv: np.ndarray, draws: int, level: float, seed: int) -> dict:
    """Point values (closed_w.model_quantities on the full matrices) and percentile intervals of every quantity."""
    tm = {k: cw.template_means(v) for k, v in mats.items()}
    pt = quantities(*(float(cw._wmean(tm[x], wv)) for x in ("R", "F", "C", "W")))
    per = draw_means(mats, strata, draws, seed)
    bq = quantities(*(cw._wmean(per[x], wv) for x in ("R", "F", "C", "W")))
    return {"point": {k: (float(v) if np.isfinite(v) else None) for k, v in pt.items()},
            "ci": {k: percentile(v, level) for k, v in bq.items()}}


# ---------------------------------------------------------------------------------------------------- build

def _mat_list(M: np.ndarray) -> list:
    return [[None if not np.isfinite(x) else float(x) for x in row] for row in M]


def mats_from_lists(d: dict) -> dict:
    return {k: np.array([[np.nan if x is None else x for x in row] for row in v], dtype=float) for k, v in d.items()}


def stored_gain(cwj: dict, task: str, score: str, model: str = MODEL) -> dict:
    hit = [g for g in cwj["gains"] if g["model"] == model and g["task"] == task and g["score"] == score
           and g["basis"] == BASIS]
    if len(hit) != 1:
        raise SystemExit(f"closed_w.json: {len(hit)} gains rows for {model} {task} {score} {BASIS}")
    return hit[0]


def compute_cell(mats: dict, task: str, score: str, items, templates, weights: dict, draws: int, level: float,
                 model: str = MODEL) -> dict:
    wv = np.array([weights.get(c, 0) for c in templates], dtype=float)
    lab = seed_label(model, task, score)
    res = intervals(mats, strata_of(task, items), wv, draws, level, cw.cwp.label_seed(lab))
    strata = strata_of(task, items)
    return {"task": task, "score": score, "model": model, "basis": BASIS, "seed_label": lab,
            "items": len(items), "strata": len(set(strata)), "templates": len(templates),
            "templates_weighted": int((wv > 0).sum()), "discovery_draws": int(wv.sum()),
            "missing_cells": {k: int((~np.isfinite(v)).sum()) for k, v in mats.items()},
            "weights": {c: int(weights.get(c, 0)) for c in templates},
            "item_ids": list(items), "template_ids": list(templates),
            **res, "mats": {k: _mat_list(v) for k, v in mats.items()}}


def check_against_stored(cell: dict, g: dict) -> dict:
    """Point values against closed_w.json's gains row (bit for bit), and the stored intervals of the two gains and the
    corrected gain against the ones recomputed here."""
    pts = {}
    for k in ("requested", "forced", "cot", "w", *STORED):
        a, b = cell["point"][k], g[k]
        pts[k] = {"here": a, "stored": b, "equal": a == b}
    pts["gain_diff"] = {"here": cell["point"]["gain_diff"], "stored": g["cot_minus_forced"] - g["cot_minus_requested"],
                        "equal": cell["point"]["gain_diff"] == g["cot_minus_forced"] - g["cot_minus_requested"]}
    cis = {}
    for k in STORED:
        here = [cell["ci"][k]["lo"], cell["ci"][k]["hi"]]
        cis[k] = {"here": here, "stored": g["ci"].get(k), "equal": here == g["ci"].get(k)}
    return {"points": pts, "intervals": cis,
            "all_equal": all(v["equal"] for v in pts.values()) and all(v["equal"] for v in cis.values())}


def build(root: Path = ROOT, scored: pd.DataFrame | None = None) -> dict:
    cfg = cw.load_config()
    paths = cw.Paths(cfg, root)
    man = cw.load_manifest(paths)
    cwj = json.loads((root / "results" / "frontier_panel" / "closed_w.json").read_text(encoding="utf-8"))
    versions, csha = None, None
    if scored is None:
        scored, versions, csha = load_scored(cfg, paths, man)
    b = cfg["bootstrap"]
    draws, level = int(b["draws"]), float(b["level"])
    out = {"made_on": date.today().isoformat(), "script": "scripts/closed_w_intervals.py", "model": MODEL,
           "basis": BASIS, "draws": draws, "level": level, "config_sha": csha, "closed_w_config_sha": cwj["config_sha"],
           "scorers_here": versions, "scorers_closed_w": cwj.get("scorers"),
           "definitions": {
               "req_gain": "C - R (closed_w.json cot_minus_requested)",
               "fd_gain": "C - F (closed_w.json cot_minus_forced)",
               "gain_diff": "(C - F) - (C - R) = R - F: the forced-baseline gain minus the requested-baseline gain",
               "corrected_gain": "(1 - w)(C - F), w and C - F each a weighted template mean first "
                                 "(closed_w.json corrected_gain)",
               "w": "share of requested answers whose API visible output count exceeds the forced cap (16 tokens)",
               "weights": "discovery draws of manifests/sampler_<task>.json (a template drawn twice counts twice)",
               "resampling": "questions with replacement, the same draw for every arm (paired); BIG-Bench Hard within "
                             "task; templates and weights fixed; percentile intervals"},
           "cells": []}
    if csha is not None and csha != cwj["config_sha"]:
        raise SystemExit(f"run hash {csha} differs from closed_w.json's {cwj['config_sha']}")
    for task, score in CELLS:
        t = man["tasks"][task]
        weights = cw.discovery_weights(task, t["templates"], root)
        if weights is None:
            raise SystemExit(f"{task}: no discovery weights")
        mats = cell_mats(scored, cfg, man, task, score)
        cell = compute_cell(mats, task, score, t["items"], t["templates"], weights, draws, level)
        cell["check"] = check_against_stored(cell, stored_gain(cwj, task, score))
        out["cells"].append(cell)
    out["all_equal_to_closed_w"] = all(c["check"]["all_equal"] for c in out["cells"])
    return out


# ---------------------------------------------------------------------------------------------------- render

TASK = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard"}


def _pp(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def render(res: dict) -> str:
    bbh = next(c for c in res["cells"] if c["task"] == "bbh")
    per_task = sorted({sum(1 for i in bbh["item_ids"] if i.split("/", 1)[0] == s)
                       for s in {i.split("/", 1)[0] for i in bbh["item_ids"]}})
    per_task_txt = (f"{per_task[0]} questions in each of {bbh['strata']} tasks" if len(per_task) == 1
                    else f"{bbh['strata']} tasks of {per_task[0]} to {per_task[-1]} questions")
    L = [f"# Qwen4-Max: intervals for the gains from chain-of-thought ({res['made_on']})", "",
         "Script `scripts/closed_w_intervals.py`; data: the 100 questions per "
         "benchmark of `results/frontier_panel/closed_w.json`, rebuilt from the run's row-level replies "
         "(`data/frontier_panel/closed_w/run_raw.jsonl`) with the panel's scorers, as `scripts/closed_w.py report` "
         "builds them.", "",
         f"Each value is a weighted mean over templates with the discovery draws' weights "
         f"(`manifests/sampler_<task>.json`). Intervals: {100 * res['level']:.0f}% percentile intervals from "
         f"{res['draws']:,} draws of the questions, resampled with replacement, the same draw for every arm; on "
         f"BIG-Bench Hard within task ({per_task_txt}); templates and weights held fixed. "
         "The draws and their seeds are those of `scripts/closed_w.py report`, so the intervals of the two gains and "
         "of the corrected gain are the ones `closed_w.json` already holds; the difference is added from the same "
         "draws.", "",
         "Definitions (pp): requested-baseline gain C − R; forced-baseline gain C − F; difference = forced-baseline "
         "gain − requested-baseline gain = R − F (the requested minus the forced accuracy); corrected gain "
         "(1 − w)(C − F), with w and C − F each averaged over templates first.", "",
         "| Benchmark | Scoring | Requested-baseline gain | Forced-baseline gain | Difference | Corrected gain | w |",
         "|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        p, ci = c["point"], c["ci"]

        def cell(k):
            return f"{_pp(p[k])} [{_pp(ci[k]['lo'])}, {_pp(ci[k]['hi'])}]"
        L.append(f"| {TASK[c['task']]} | {c['score']} | {cell('cot_minus_requested')} | {cell('cot_minus_forced')} | "
                 f"{cell('gain_diff')} | {cell('corrected_gain')} | {cell('w')} |")
    L += ["", "Accuracies (pp, discovery weights) and the data behind each row:", "",
          "| Benchmark | Scoring | Requested R | Forced F | CoT C | Questions | Strata | Templates (weighted) | "
          "Discovery draws | Missing cells R/F/C/W | Finite draws (fewest) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        p, mc = c["point"], c["missing_cells"]
        fd = min(v["finite_draws"] for v in c["ci"].values())
        L.append(f"| {TASK[c['task']]} | {c['score']} | {_pp(p['requested'])} | {_pp(p['forced'])} | {_pp(p['cot'])} | "
                 f"{c['items']} | {c['strata']} | {c['templates']} ({c['templates_weighted']}) | "
                 f"{c['discovery_draws']} | {mc['R']}/{mc['F']}/{mc['C']}/{mc['W']} | {fd} of {res['draws']:,} |")
    L += ["", "Checks against `closed_w.json` (gains rows, discovery basis):", ""]
    for c in res["cells"]:
        ch = c["check"]
        np_ = sum(v["equal"] for v in ch["points"].values())
        ni = sum(v["equal"] for v in ch["intervals"].values())
        L.append(f"- {TASK[c['task']]}, {c['score']}: point values equal {np_} of {len(ch['points'])} (bit for bit); "
                 f"stored intervals equal {ni} of {len(ch['intervals'])}.")
    L += ["", f"All equal: {'yes' if res['all_equal_to_closed_w'] else 'NO'}.", ""]
    zero = [f"{LABEL[k].split(' (')[0].lower()} on {TASK[c['task']]} ({c['score']})" for c in res["cells"]
            for k in ("req_gain", "fd_gain", "gain_diff", "corrected_gain")
            if c["ci"][QUANT[k] or k]["lo"] is not None and c["ci"][QUANT[k] or k]["lo"] <= 0 <= c["ci"][QUANT[k] or k]["hi"]]
    short = [f"{TASK[c['task']]} ({c['score']})" for c in res["cells"]
             if min(v["finite_draws"] for v in c["ci"].values()) < res["draws"]]
    L += ["Edge cases, as they are: intervals that include zero: " + ("; ".join(zero) if zero else "none") + ". "
          "Draws with a quantity that is not finite (left out of its interval): "
          + ("; ".join(short) if short else "none") + ".", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = build()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render(res)
    Path(str(a.out) + ".md").write_text(md, encoding="utf-8")
    Path(str(a.out) + "_clean.md").write_text(cw.clean_md(md), encoding="utf-8")
    print(md)
    if not res["all_equal_to_closed_w"]:
        print("NOT EQUAL to closed_w.json: see the checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
