r"""Analyses of the default- and firm-clause run with the arm-wording run (src/arm_wording_extra.py).

    .\.venv\Scripts\python.exe scripts\run_arm_wording_extra.py --stage discovery
        the arm-wording run's discovery quantities (src/arm_wording.discovery_quantities) with six clauses, from this run's discovery
        table and the arm-wording run's scored discovery rows -> manifests/predictions_arm_wording_extra_discovery.json and .sha256
    .\.venv\Scripts\python.exe scripts\run_arm_wording_extra.py
        refuses unless the forecast and the discovery predictions are saved and unchanged and the full table was scored
        after both; then, per task: the arm-wording run's analyses 1 (R when the wording is drawn) and 2 (pair reversals, the arm-switch
        share per clause) with six clauses, strict and lenient; the decision cost and the prediction checks; every
        statement of the forecast, misses included -> results/arm_wording_extra/arm_wording_extra.json,
        forecast_evaluation.json, summary.md and summary_clean.md

The arm-wording run's scored table (data/processed/arm_wording/all_scored.parquet) is read only here, after its own after-step has
scored it (its metadata must carry its predictions and mechanism hashes); its chain-of-thought rows (four triggers)
and its four clauses' direct rows are joined to this run's two clauses. --base DIR for dry runs; --n-boot for short
resampling. Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import arm_wording as aw, arm_wording_extra as ax, decision  # noqa: E402


def d24_design(paths):
    rec = json.loads((paths.d24_requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    cot = [w["id"] for w in rec["chosen"]["cot"]]
    dirs = [w["id"] for w in rec["chosen"]["dir"]]
    if dirs != list(ax.D24_CLAUSES):
        raise ax.ExtraError(f"the arm-wording run's clauses are {dirs}, not {list(ax.D24_CLAUSES)}")
    return rec, cot, list(rec["models"])


def d24_table(paths) -> pd.DataFrame:
    """The arm-wording run's full scored table, only once its after-step has scored it after its saved predictions and forecast."""
    src = paths.d24_processed / "all_scored.parquet"
    meta = src.with_suffix(".meta.json")
    if not src.is_file() or not meta.is_file():
        raise ax.ExtraError(f"{src} does not exist: the arm-wording run is not scored yet; the six-clause "
                            "analysis needs its chain-of-thought rows")
    md = json.loads(meta.read_text(encoding="utf-8"))
    if md.get("stage") != "all" or not md.get("predictions_sha256") or not md.get("mechanism_sha256"):
        raise ax.ExtraError(f"{src} is not the arm-wording run's full table scored after its saved predictions and forecast")
    return aw.tidy(pd.read_parquet(src))


def own_table(paths, stage: str, need: dict) -> tuple[pd.DataFrame, dict]:
    src = paths.processed / f"{stage}_scored.parquet"
    md = json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if md["stage"] != stage:
        raise ax.ExtraError(f"{src} is not the {stage} table")
    for k, v in need.items():
        if md.get(k) != v:
            raise ax.ExtraError(f"{src} was not scored after the saved {k.replace('_sha256', '')}")
    return aw.tidy(pd.read_parquet(src)), md


def arrays(table, task, configs, cot, models, score="strict_correct"):
    return aw.TaskArrays(table, task, configs, cot, list(ax.CLAUSES6), models, score)


def discovery_block(table, rec, cot, models) -> dict:
    return {t: aw.discovery_quantities(arrays(table, t, rec["configurations"][t]["discovery"], cot, models).acc(),
                                       decision.third_size(len(models))) for t in aw.TASKS}


def stage_discovery(args, paths) -> dict:
    fc = ax.check_forecast(paths)
    rec, cot, models = d24_design(paths)
    own, _ = own_table(paths, "discovery", {"forecast_sha256": fc})
    d24 = d24_table(paths)
    disc = {t: set(rec["configurations"][t]["discovery"]) for t in aw.TASKS}
    d24 = d24[[c in disc[t] for t, c in zip(d24["task"], d24["config"])]]
    table = pd.concat([d24, own], ignore_index=True)
    q = discovery_block(table, rec, cot, models)
    body = {"kind": "arm_wording_extra_discovery_predictions", "rule": "default and firm clauses (arm-wording step 2 "
            "with six clauses)", "clauses": list(ax.CLAUSES6), "triggers": cot, "models": models,
            "forecast_sha256": fc, "per_task": q, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if paths.predictions.exists():
        old = json.loads(paths.predictions.read_text(encoding="utf-8"))
        if not aw.same_quantities(old["per_task"], json.loads(json.dumps(q, default=aw._default))):
            raise ax.ExtraError(f"{paths.predictions} exists with other content; written once")
    else:
        sha = aw.write_json(paths.predictions, body)
        paths.predictions_sha.write_text(f"{sha}  {paths.predictions.name}\n", encoding="utf-8")
    return {"predictions": str(paths.predictions), "sha256": aw.sha256_file(paths.predictions),
            "arm_switch_share_by_clause": {t: v["arm_switch_share_by_clause"] for t, v in q.items()}}


def full(args, paths) -> dict:
    t0 = time.time()
    fc = ax.check_forecast(paths)
    pred_sha = ax.check_discovery_predictions(paths)
    pred = json.loads(paths.predictions.read_text(encoding="utf-8"))
    frec = json.loads(paths.forecast.read_text(encoding="utf-8"))
    rec, cot, models = d24_design(paths)
    own, md = own_table(paths, "all", {"forecast_sha256": fc, "predictions_sha256": pred_sha})
    table = pd.concat([d24_table(paths), own], ignore_index=True)
    refit = discovery_block(table, rec, cot, models)
    if not aw.same_quantities(refit, json.loads(json.dumps(pred["per_task"]))):
        raise ax.ExtraError("the discovery quantities refitted from the full tables differ from the saved predictions")
    n_boot = args.n_boot if args.n_boot is not None else 2000
    tol = ax.TOL
    per_task, E = {}, {"kind": "arm_wording_extra_forecast_evaluation", "sha256": frec["sha256"], "file_sha256": fc,
                       "created": frec.get("created"), "rule": frec["rule"], "tasks": {}}
    for task in aw.TASKS:
        disc, val = rec["configurations"][task]["discovery"], rec["configurations"][task]["validation"]
        confs = disc + val
        vi = list(range(len(disc), len(confs)))
        res = {"configurations": {"discovery": disc, "validation": val}}
        for score in ("strict_correct", "lenient_correct"):
            ta = arrays(table, task, confs, cot, models, score)
            acc = ta.acc()
            G = aw.gains(acc)
            block = {"R": aw.r_table(G), "reversals": aw.reversal_table(G)}
            if score == "strict_correct":
                block["R_item_bootstrap_ci95"] = aw.item_bootstrap_R(ta, n_boot, 20261121) if n_boot else None
                w_obs = ax.observed_w(table, task, list(ax.CLAUSES6), models)
                block["direct_gap"] = aw.direct_gap(acc, list(ax.CLAUSES6),
                                                    {d: float(np.nanmean(w_obs[k])) for k, d in enumerate(ax.CLAUSES6)})
                block["decision"] = aw.decision_evaluate(pred["per_task"][task]["decision"],
                                                         aw.gains({a: v[vi] for a, v in acc.items()}))
                pt = pred["per_task"][task]
                obs_c = aw.scheme_C(G["req"], "both_drawn", configs=vi)
                pc_ = pt["predicted_C_both_drawn_validation"]
                sw_val = [s["one_minus_C"] for s in aw.reversal_table(G, configs=vi)["arm_switch_by_clause"]]
                block["prediction_checks"] = {
                    "C_both_drawn": {"predicted": pc_, "observed_validation": obs_c,
                                     "within_tolerance": None if pc_ is None else bool(abs(obs_c - pc_) <= tol)},
                    "arm_switch_by_clause": [{"clause": w, "discovery": a, "validation": b,
                                              "within_tolerance": bool(abs(a - b) <= tol)}
                                             for w, a, b in zip(ax.CLAUSES6, pt["arm_switch_share_by_clause"], sw_val)],
                    "tolerance": tol}
                res["accuracy_mean_pp"] = {a: {w: float(v[:, k, :].mean()) for k, w in enumerate(ta.wids[a])}
                                           for a, v in acc.items()}
                E["tasks"][task] = ax.evaluate(frec, task, acc, block["reversals"], models, w_obs)
            res[score.replace("_correct", "")] = block
        per_task[task] = res
    E["holds"] = ax.holds_summary(E)
    out = {"kind": "arm_wording_extra_results", "rule": "default and firm clauses", "clauses": list(ax.CLAUSES6),
           "clause_texts": frec["clause_texts"], "triggers": cot, "models": models,
           "forecast": {"evaluation": "results/arm_wording_extra/forecast_evaluation.json", "sha256": frec["sha256"],
                        "holds": E["holds"]},
           "predictions_sha256": pred_sha, "per_task": per_task, "synthetic_seed": md.get("synthetic_seed"),
           "runtime_s": round(time.time() - t0, 1)}
    paths.results.mkdir(parents=True, exist_ok=True)
    aw.write_json(paths.results / "arm_wording_extra.json", out)
    aw.write_json(paths.results / "forecast_evaluation.json", E)
    text = summary_md(out, E)
    (paths.results / "summary.md").write_text(text, encoding="utf-8")
    (paths.results / "summary_clean.md").write_text(text, encoding="utf-8")
    return {"results": str(paths.results / "arm_wording_extra.json"), "holds": E["holds"], "runtime_s": out["runtime_s"]}


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summary_md(o, E) -> str:
    L = ["# Default and firm request clauses with the four drawn clauses", "",
         "Six direct clauses on the arm-wording run's six configurations, 29 models, strict scores unless noted. dir0 is the current "
         "clause, dir1–dir3 the paraphrases, dirdef the default request (no instruction about working), dirfirm the "
         f"firm request (\"{o['clause_texts']['dirfirm']}\"). Chain of thought from the arm-wording run (four triggers).", ""]
    for t, r in o["per_task"].items():
        s = r["strict"]
        L += [f"## {t}", "", "| Clause | Requested (pp) | Forced (pp) | Requested − forced (pp) | w observed | "
              "Arm-switch 1 − C |", "|---|---|---|---|---|---|"]
        acc = r["accuracy_mean_pp"]
        sw = s["reversals"]["arm_switch_by_clause"]
        for k, (d, g) in enumerate(zip(o["clauses"], s["direct_gap"])):
            L.append(f"| {d} | {f(acc['direct_answer'][d], 1)} | {f(acc['forced_direct'][d], 1)} | "
                     f"{f(g['mean_gap_pp'], 1)} | {f(g['long_requested_share'], 2)} | {f(sw[k]['one_minus_C'])} |")
        rv = s["reversals"]
        L += ["", f"Clause-only reversal (requested, six clauses) 1 − C {f(rv['clause_switch_requested']['one_minus_C'])}, "
                  f"configuration-only {f(rv['configuration_switch_requested']['one_minus_C'])}, arm switch "
                  f"{f(rv['arm_switch']['one_minus_C'])}. R with both wordings drawn (requested): "
                  f"{f(s['R']['req']['both_drawn']['R'])}.", ""]
        x = E["tasks"][t]
        a1 = x["A1_forced_meets_requested"]
        L += ["Forecast statements (saved before scoring; misses reported):", "",
              f"- A1 firm clause, {a1['n']} models with pilot w ≤ {a1['w_near0']}: mean requested − forced "
              f"{f(a1['mean_gap_pp'], 2)} pp (within ±{a1['tolerance_pp']}: {a1['holds_a']}); within tolerance in "
              f"{a1['within_tol']} of {a1['n']} ({a1['holds_b']}). {a1['note'] or ''}",
              f"- A2 default more working: median Δw {f(x['A2_default_more_working']['median_diff'])} "
              f"({x['A2_default_more_working']['holds_a']}), {x['A2_default_more_working']['models_in_direction']} of "
              f"{x['A2_default_more_working']['n']} ({x['A2_default_more_working']['holds_b']}).",
              f"- A3 firm less working: median Δw {f(x['A3_firm_less_working']['median_diff'])} "
              f"({x['A3_firm_less_working']['holds_a']}), {x['A3_firm_less_working']['models_in_direction']} of "
              f"{x['A3_firm_less_working']['n']} ({x['A3_firm_less_working']['holds_b']}).",
              f"- A4 pilot w against collection w: r = {f(x['A4_pilot_predicts_w']['pearson'])} "
              f"({x['A4_pilot_predicts_w']['holds']}).",
              f"- A5 median within-model intercept {f(x['A5_extrapolates_to_forced']['median_intercept_pp'], 2)} pp over "
              f"{x['A5_extrapolates_to_forced']['models']} models ({x['A5_extrapolates_to_forced']['holds']}).",
              f"- S1–S5 over six clauses: {E['holds'][t]}", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("discovery", "all"), default="all")
    ap.add_argument("--base")
    ap.add_argument("--n-boot", type=int)
    args = ap.parse_args(argv)
    paths = ax.Paths(args.base)
    try:
        res = stage_discovery(args, paths) if args.stage == "discovery" else full(args, paths)
    except (ax.ExtraError, aw.ArmWordingError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
