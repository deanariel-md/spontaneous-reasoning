r"""Analyses of the arm-wording run: R when the wording is drawn, pair reversals by arm,
wording and configuration, the protocol ceiling with wording as a factor, the decision cost, the repeat check against
the collection, the checks of the saved predictions and the fixed verdicts on the abstract's sentences.

    .\.venv\Scripts\python.exe scripts\run_arm_wording.py --stage discovery
        reads data/processed/arm_wording/discovery_scored.parquet (the three discovery configurations) and saves
        manifests/predictions_arm_wording.json with its sha256 in manifests/predictions_arm_wording.sha256
    .\.venv\Scripts\python.exe scripts\run_arm_wording.py
        refuses unless the predictions are saved and unchanged; reads data/processed/arm_wording/all_scored.parquet,
        refits the discovery quantities from its discovery rows and stops if they differ, then writes
        results/arm_wording/arm_wording.json, summary.md and summary_clean.md; it also refuses unless
        the mechanism forecast (manifests/predictions_arm_wording_mechanism.json) is saved, unchanged and recorded in the
        scored table's metadata, and evaluates its statements, misses included, into
        results/arm_wording/mechanism_forecast_evaluation.json
    .\.venv\Scripts\python.exe scripts\run_arm_wording.py --fig-w-table
        writes only results/arm_wording/tables/fig_w_vs_requested.csv (w, the
        share of requested outputs longer than 16 tokens, and the strict accuracies of the three arms per model, task
        and direct clause) and its line in the two summaries; a full run writes it too

--base DIR reads and writes under DIR (dry runs); --n-boot and --n-splits shorten the resampling (dry runs).
Exit codes: 0 written, 2 refused, 1 any other error.
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
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, arm_wording_mechanism as awm, decision  # noqa: E402
import score_arm_wording as saw  # noqa: E402


def design(paths):
    rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    chosen = rec["chosen"]
    return rec, [w["id"] for w in chosen["cot"]], [w["id"] for w in chosen["dir"]], list(rec["models"])


def arrays(table, task, configs, cot_ids, dir_ids, models, score="strict_correct"):
    return aw.TaskArrays(table, task, configs, cot_ids, dir_ids, models, score)


def discovery_block(table, rec, cot_ids, dir_ids, models) -> dict:
    out = {}
    for task in aw.TASKS:
        disc = rec["configurations"][task]["discovery"]
        ta = arrays(table, task, disc, cot_ids, dir_ids, models)
        out[task] = aw.discovery_quantities(ta.acc(), decision.third_size(len(models)))
    return out


def stage_discovery(args, paths) -> dict:
    rec, cot_ids, dir_ids, models = design(paths)
    src = paths.processed / "discovery_scored.parquet"
    md = json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if md["stage"] != "discovery":
        raise aw.ArmWordingError(f"{src} is not the discovery table")
    table = aw.tidy(pd.read_parquet(src))
    for task in aw.TASKS:
        extra = set(table.loc[table["task"] == task, "config"]) - set(rec["configurations"][task]["discovery"])
        if extra:
            raise aw.ArmWordingError(f"{src} holds validation configurations {sorted(extra)}")
    pred = {"kind": "arm_wording_predictions", "rule": "arm wording, order and analyses 1, 2, 4",
            "discovery_table_sha256": aw.sha256_file(src), "wordings": {"cot": cot_ids, "dir": dir_ids},
            "models": models, "per_task": discovery_block(table, rec, cot_ids, dir_ids, models)}
    status = aw.write_once(paths.predictions, pred, "the predictions file")
    sha = aw.sha256_file(paths.predictions)
    paths.predictions_sha.write_text(f"{sha}  {paths.predictions.name}\n", encoding="utf-8")
    return {"predictions": str(paths.predictions), "status": status, "sha256": sha}


def long_shares(table, task, long_tokens) -> dict:
    r = table[(table["task"] == task) & (table["arm"] == aw.REQ)]
    return {w: float((g["n_output_tokens"] > long_tokens).mean()) for w, g in r.groupby("wording")}


def repeat_check(table) -> dict:
    """Current-wording cells against the collection's scored tables (analysis 5). Not a gate."""
    srcs = {("gsm8k", "coll"): [ROOT / "data" / "processed" / "gsm8k" / "scored.parquet"]
            + sorted((ROOT / "data" / "processed" / "gsm8k_ext").glob("scored_*.parquet")),
            ("math", "coll"): [ROOT / "data" / "processed" / "math" / "scored.parquet"],
            ("gsm8k", "fd"): [ROOT / "data" / "processed" / "forced" / "gsm8k_scored.parquet"],
            ("math", "fd"): [ROOT / "data" / "processed" / "forced" / "math_scored.parquet"]}
    out = {}
    cur = table[((table["arm"] == aw.COT) & (table["wording"] == "cot0"))
                | ((table["arm"] != aw.COT) & (table["wording"] == "dir0"))]
    for task in aw.TASKS:
        mine = cur[cur["task"] == task]
        cols = ["model_key", "arm", "config_id", "item_id", "strict_correct", "prompt_sha256"]
        frames = []
        for kind in ("coll", "fd"):
            for p in srcs[(task, kind)]:
                if p.is_file():
                    d = pd.read_parquet(p, columns=cols)
                    d = d[d["model_key"].isin(set(mine["model_key"])) & d["config_id"].isin(set(mine["config"]))]
                    frames.append(d)
        if not frames:
            out[task] = {"note": "the collection's scored tables are absent"}
            continue
        col = pd.concat(frames, ignore_index=True)
        col["item_id"] = col["item_id"].astype(str)
        m = mine.merge(col, left_on=["model_key", "arm", "config", "item_id"],
                       right_on=["model_key", "arm", "config_id", "item_id"], suffixes=("", "_coll"))
        per_arm = {}
        for arm, g in m.groupby("arm"):
            acc = g.groupby("model_key").agg(a=("strict_correct", "mean"), b=("strict_correct_coll", "mean"))
            diff = 100 * (acc["a"] - acc["b"])
            per_arm[arm] = {"cells_matched": int(len(g)),
                            "same_strict_correctness": float((g["strict_correct"].astype(bool)
                                                              == g["strict_correct_coll"].astype(bool)).mean()),
                            "same_prompt_sha256": float((g["prompt_sha256"] == g["prompt_sha256_coll"]).mean()),
                            "accuracy_difference_pp_mean": float(diff.mean()),
                            "accuracy_difference_pp_range": [float(diff.min()), float(diff.max())],
                            "per_model_pp": {k: float(v) for k, v in diff.items()}}
        out[task] = {"cells_current": int(len(mine)), "per_arm": per_arm}
    return out


def full(args, paths, cfg) -> dict:
    t0 = time.time()
    pred_sha = saw.check_predictions(paths)
    pred = json.loads(paths.predictions.read_text(encoding="utf-8"))
    rec, cot_ids, dir_ids, models = design(paths)
    src = paths.processed / "all_scored.parquet"
    md = json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if md["stage"] != "all" or md["predictions_sha256"] != pred_sha:
        raise aw.ArmWordingError(f"{src} was not scored after the saved predictions")
    # the mechanism forecast must be saved, unchanged, and older than the scored table
    mech_sha = saw.check_mechanism(paths)
    if md.get("mechanism_sha256") != mech_sha:
        raise aw.ArmWordingError(f"{src} was not scored after the saved mechanism forecast")
    table = aw.tidy(pd.read_parquet(src))
    refit = discovery_block(table, rec, cot_ids, dir_ids, models)
    if not aw.same_quantities(refit, pred["per_task"]):
        raise aw.ArmWordingError("the discovery quantities refitted from the full table differ from the saved "
                                 "predictions; the discovery rows changed after the predictions were saved")
    A = cfg["analysis"]
    n_boot = args.n_boot if args.n_boot is not None else int(A["bootstrap_items"])
    n_boot_c = args.n_boot if args.n_boot is not None else int(A["bootstrap_configurations"])
    n_splits = args.n_splits or 200
    per_task, mech = {}, {}
    for task in aw.TASKS:
        disc, val = rec["configurations"][task]["discovery"], rec["configurations"][task]["validation"]
        confs = disc + val
        vi = list(range(len(disc), len(confs)))
        res = {"configurations": {"discovery": disc, "validation": val}}
        for score in ("strict_correct", "lenient_correct"):
            ta = arrays(table, task, confs, cot_ids, dir_ids, models, score)
            acc = ta.acc()
            G = aw.gains(acc)
            block = {"R": aw.r_table(G), "reversals": aw.reversal_table(G)}
            if score == "strict_correct":
                block["R_fixed_any_wording"] = {b: aw.fixed_any_wording(G[b]) for b in G}
                block["R_leave_one_configuration_out"] = {b: {s: aw.leave_one_config_out(G[b], s) for s in aw.SCHEMES}
                                                          for b in G}
                block["R_item_bootstrap_ci95"] = aw.item_bootstrap_R(ta, n_boot, int(A["seed"])) if n_boot else None
                block["direct_gap"] = aw.direct_gap(acc, dir_ids, long_shares(table, task, int(A["long_direct_tokens"])))
                block["ceiling"] = aw.ceiling_table(G, int(A["seed"]), n_splits=n_splits, n_boot=n_boot_c)
                Gv = aw.gains({a: v[vi] for a, v in acc.items()})
                block["decision"] = aw.decision_evaluate(pred["per_task"][task]["decision"], Gv)
                mech[task] = (acc, block["reversals"])
                pt = pred["per_task"][task]
                obs_c = aw.scheme_C(G["req"], "both_drawn", configs=vi)
                pc_ = pt["predicted_C_both_drawn_validation"]
                sw_val = [s["one_minus_C"] for s in aw.reversal_table(G, configs=vi)["arm_switch_by_clause"]]
                tol = float(A["prediction_tolerance"])
                block["prediction_checks"] = {
                    "C_both_drawn": {"predicted": pc_, "observed_validation": obs_c,
                                     "within_tolerance": None if pc_ is None else bool(abs(obs_c - pc_) <= tol)},
                    "arm_switch_by_clause": [{"clause": w, "discovery": a, "validation": b,
                                              "within_tolerance": bool(abs(a - b) <= tol)}
                                             for w, a, b in zip(dir_ids, pt["arm_switch_share_by_clause"], sw_val)],
                    "tolerance": tol}
                res["accuracy_mean_pp"] = {a: {w: float(v[:, k, :].mean()) for k, w in enumerate(ta.wids[a])}
                                           for a, v in acc.items()}
            res[score.replace("_correct", "")] = block
        per_task[task] = res
    strict = {t: {"R": v["strict"]["R"], "reversals": v["strict"]["reversals"], "decision": v["strict"]["decision"]}
              for t, v in per_task.items()}
    ME = mechanism_evaluation(paths, table, mech, models, dir_ids, mech_sha)
    out = {"kind": "arm_wording_results", "rule": "arm wording", "predictions_sha256": pred_sha,
           "mechanism_forecast": {"evaluation": "results/arm_wording/mechanism_forecast_evaluation.json",
                                  "sha256": ME["sha256"], "holds": ME["holds"]},
           "wordings": rec["chosen"], "models": models, "n_third": decision.third_size(len(models)), "per_task": per_task,
           "repeat_check": repeat_check(table), "verdicts": aw.verdicts(strict, A["abstract"]),
           "synthetic_seed": md.get("synthetic_seed"), "runtime_s": round(time.time() - t0, 1)}
    paths.results.mkdir(parents=True, exist_ok=True)
    aw.write_json(paths.results / "arm_wording.json", out)
    aw.write_json(paths.results / "mechanism_forecast_evaluation.json", ME)
    text = summary_md(out) + "\n".join(awm.summary_lines(ME)) + "\n"
    (paths.results / "summary.md").write_text(text, encoding="utf-8")
    (paths.results / "summary_clean.md").write_text(text, encoding="utf-8")
    write_fig_w_table(paths, table)
    return {"results": str(paths.results / "arm_wording.json"), "runtime_s": out["runtime_s"],
            "verdicts": {k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in out["verdicts"].items()}}


def observed_w(table, task, dir_ids, models, long_tokens=None):
    """(clauses, models) share of the arm-wording run's requested-direct outputs longer than the forced cap (no correctness read)."""
    from src import arm_wording_mechanism as awm
    lt = awm.LONG_TOKENS if long_tokens is None else long_tokens
    r = table[(table["task"] == task) & (table["arm"] == aw.REQ)]
    sh = (r["n_output_tokens"] > lt).groupby([r["wording"], r["model_key"]]).mean()
    return np.array([[float(sh.get((d, m), np.nan)) for m in models] for d in dir_ids])


FIG_W_COLUMNS = ["model_key", "task", "clause", "clause_text", "w", "n_requested", "n_requested_long",
                 "requested_accuracy_pp", "forced_accuracy_pp", "cot_accuracy_pp", "requested_minus_forced_pp"]
FIG_W_LINE = ("Figure table: `tables/fig_w_vs_requested.csv` (Fig. 2d) holds one "
              "row per model, task (gsm8k, math) and direct clause (dir0 the current clause, dir1-dir3 the drawn "
              "paraphrases; `clause_text` its words). `w` is the share of the model's requested-direct outputs under that "
              "clause in the arm-wording collection longer than 16 generated tokens (n_output_tokens > 16, the forced arm's cap; "
              "`n_requested` outputs, `n_requested_long` of them long), over the six configurations and every item; no "
              "correctness is read for it. Accuracies are strict, in pp, the mean over the six configurations: "
              "`requested_accuracy_pp` and `forced_accuracy_pp` under that clause (the direct arms carry no trigger), "
              "`cot_accuracy_pp` the chain-of-thought arm's mean over the six configurations and all four triggers "
              "(it carries no direct clause, so it repeats across a model's four clause rows); "
              "`requested_minus_forced_pp` the difference of the two direct arms. Written by "
              "`scripts/run_arm_wording.py --fig-w-table` (also at the end of a full run) from "
              "`data/processed/arm_wording/all_scored.parquet`; the threshold is `src/arm_wording_mechanism.LONG_TOKENS`, "
              "and the 128-token long share of `arm_wording.json` (`direct_gap`) is a different quantity.")


def fig_w_rows(table, rec, cot_ids, dir_ids, models, long_tokens=None) -> list:
    """Rows of tables/fig_w_vs_requested.csv: per task, clause and model, w (share of requested-direct outputs
    longer than long_tokens, default awm.LONG_TOKENS = 16) and the strict accuracies of the three arms in pp, each the
    mean over the task's six configurations (chain of thought also over all triggers)."""
    lt = awm.LONG_TOKENS if long_tokens is None else long_tokens
    text = {w["id"]: w["text"] for w in rec["chosen"]["dir"]}
    rows = []
    for task in aw.TASKS:
        confs = rec["configurations"][task]["discovery"] + rec["configurations"][task]["validation"]
        acc = arrays(table, task, confs, cot_ids, dir_ids, models).acc()
        cot = acc[aw.COT].mean(axis=(0, 1))                                       # (models,)
        r = table[(table["task"] == task) & (table["arm"] == aw.REQ) & table["config"].isin(confs)]
        long_ = r["n_output_tokens"] > lt
        n = long_.groupby([r["wording"], r["model_key"]]).size()
        k = long_.groupby([r["wording"], r["model_key"]]).sum()
        for d, clause in enumerate(dir_ids):
            req = acc[aw.REQ][:, d, :].mean(0)
            fd = acc[aw.FD][:, d, :].mean(0)
            for j, m in enumerate(models):
                nn, kk = int(n.get((clause, m), 0)), int(k.get((clause, m), 0))
                rows.append({"model_key": m, "task": task, "clause": clause, "clause_text": text.get(clause, ""),
                             "w": kk / nn if nn else float("nan"), "n_requested": nn, "n_requested_long": kk,
                             "requested_accuracy_pp": float(req[j]), "forced_accuracy_pp": float(fd[j]),
                             "cot_accuracy_pp": float(cot[j]), "requested_minus_forced_pp": float(req[j] - fd[j])})
    return rows


def write_fig_w_table(paths, table=None) -> dict:
    """Write results/arm_wording/tables/fig_w_vs_requested.csv and its line in summary.md and summary_clean.md; reads
    the scored table when none is given. Changes no value of arm_wording.json."""
    import csv
    import write_curve_tables as wfc
    rec, cot_ids, dir_ids, models = design(paths)
    if table is None:
        table = aw.tidy(pd.read_parquet(paths.processed / "all_scored.parquet"))
    rows = fig_w_rows(table, rec, cot_ids, dir_ids, models)
    out = paths.results / "tables" / "fig_w_vs_requested.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIG_W_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for s in ("summary.md", "summary_clean.md"):
        wfc.readme_line(str(paths.results / s), FIG_W_LINE)
    return {"table": str(out), "rows": len(rows)}


def mechanism_evaluation(paths, table, mech, models, dir_ids, mech_sha) -> dict:
    """Every statement of the saved mechanism forecast on the six configurations of the arm-wording run, misses included."""
    rec = json.loads(paths.mechanism.read_text(encoding="utf-8"))
    E = {"kind": "arm_wording_mechanism_evaluation", "sha256": rec["sha256"], "file_sha256": mech_sha,
         "created": rec.get("created"), "rule": rec["rule"], "tasks": {}}
    for task in aw.TASKS:
        acc, rev = mech[task]
        E["tasks"][task] = awm.evaluate(rec, task, acc, rev, models, dir_ids, observed_w(table, task, dir_ids, models))
    hold = {}
    for task, x in E["tasks"].items():
        hold[task] = {"S1": x["S1_clause_reversal"]["holds"], "S2": x["S2_clause_vs_template"]["holds"],
                      "S3_all": x["S3_arm_switch"]["holds_all"], "S3_count": x["S3_arm_switch"]["holds_count"],
                      "S4a": x["S4_requested_rises"]["holds_a"], "S4b": x["S4_requested_rises"]["holds_b"],
                      "S5a": x["S5_forced_steadier"]["holds_a"], "S5b": x["S5_forced_steadier"]["holds_b"]}
    E["holds"] = hold
    E["forecast_summary"] = {t: {"rho": x["rho"], "w_summary": x["w_summary"], "predicted": {
        k: v for k, v in x["predicted"].items() if k != "by_form"}} for t, x in rec["per_task"].items()}
    return E


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def summary_md(o) -> str:
    L = ["# Arm wording drawn", ""]
    if o.get("synthetic_seed") is not None:
        L += [f"**Dry run on synthetic scores (seed {o['synthetic_seed']}); no number here is a result.**", ""]
    L += ["Wordings: " + "; ".join(f"{w['id']} \"{w['text']}\"" for fam in ("cot", "dir") for w in o["wordings"][fam]), ""]
    L += ["## R when the wording is drawn (chain of thought − requested direct, strict, points)", "",
          "| Task | Scheme | R | One-draw C | Gaussian C | Prompts for C = 0.9 | Item bootstrap 95% | Leave one configuration out |",
          "|---|---|---:|---:|---:|---:|---|---|"]
    for t, v in o["per_task"].items():
        s = v["strict"]
        for sc in aw.SCHEMES:
            r = s["R"]["req"][sc]
            ci = (s["R_item_bootstrap_ci95"] or {}).get("req", {}).get(sc)
            lo = s["R_leave_one_configuration_out"]["req"][sc]
            L.append(f"| {t} | {sc} | {f(r['R'])} | {f(r['C_one_draw'])} | {f(r['gaussian_C'])} | "
                     f"{f(r['prompts_needed_0.9'])} | {'n/a' if ci is None else f'[{f(ci[0])}, {f(ci[1])}]'} | "
                     f"{f(lo['min'])}–{f(lo['max'])} |")
    L += ["", "Against forced direct, and lenient scoring, in `arm_wording.json`.", "",
          "## Pair reversals (1 − C, ties one half; chain of thought − direct, strict)", "",
          "| Task | Change | 1 − C | Reversed | Tied |", "|---|---|---:|---:|---:|"]
    for t, v in o["per_task"].items():
        rv = v["strict"]["reversals"]
        for k in ("arm_switch", "clause_switch_requested", "clause_switch_forced", "trigger_switch_requested",
                  "configuration_switch_requested", "arm_and_clause_switch"):
            L.append(f"| {t} | {k} | {f(rv[k]['one_minus_C'])} | {f(rv[k]['reversed'])} | {f(rv[k]['tied'])} |")
        for w, x in zip(o["wordings"]["dir"], rv["arm_switch_by_clause"]):
            L.append(f"| {t} | arm switch, clause {w['id']} | {f(x['one_minus_C'])} | {f(x['reversed'])} | {f(x['tied'])} |")
    L += ["", "## Requested − forced accuracy per clause (strict, points)", "",
          "| Task | Clause | Mean gap | Model range | Requested outputs > 128 tokens |", "|---|---|---:|---|---:|"]
    for t, v in o["per_task"].items():
        for g in v["strict"]["direct_gap"]:
            L.append(f"| {t} | {g['clause']} | {f(g['mean_gap_pp'], 2)} | {f(g['min_model_gap_pp'], 2)} to "
                     f"{f(g['max_model_gap_pp'], 2)} | {f(g['long_requested_share'])} |")
    L += ["", "## Protocol ceiling with wording as a factor (strict)", "",
          "| Task | Protocols differ in | Pairs | K = 1 | K = 6 | Limit, plug-in | Limit, posterior |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for t, v in o["per_task"].items():
        w = v["strict"]["ceiling"]["within"]
        wk = sorted(w["curve"], key=int)
        if wk:
            L.append(f"| {t} | nothing (same protocol, disjoint configurations) | {w['protocols']} protocols | "
                     f"{f(w['curve'][wk[0]])} | K = {wk[-1]}: {f(w['curve'][wk[-1]])} | n/a | n/a |")
        for ax, c in v["strict"]["ceiling"].items():
            if ax == "within":
                continue
            ks = sorted(c["curve"], key=int)
            L.append(f"| {t} | {ax} | {c['pairs']} | {f(c['curve'][ks[0]])} | {f(c['curve'][ks[-1]])} | "
                     f"{f(c['limit_plugin'])} | {f(c['limit_posterior'])} |")
    L += ["", f"## Decision cost (top third, {o['n_third']} of {len(o['models'])}; pilots: each discovery configuration; "
          "evaluation: the validation configurations)", "",
          "| Task | Evaluation protocol differs in | Protocols | Extra regret of choosing under the current wording (pp) | 95% t interval |",
          "|---|---|---:|---:|---|"]
    for t, v in o["per_task"].items():
        for ax, g in v["strict"]["decision"]["grouped"].items():
            ci = g.get("ci95")
            L.append(f"| {t} | {ax} | {g['protocols']} | {f(g['estimate'], 2)} | "
                     f"{'n/a' if ci is None else f'[{f(ci[0], 2)}, {f(ci[1], 2)}]'} |")
    L += ["", "## Checks of the saved predictions (validation configurations)", ""]
    for t, v in o["per_task"].items():
        c = v["strict"]["prediction_checks"]
        L.append(f"- {t}: one-draw C, both drawn: predicted {f(c['C_both_drawn']['predicted'])}, observed "
                 f"{f(c['C_both_drawn']['observed_validation'])}, within {c['tolerance']}: {c['C_both_drawn']['within_tolerance']}. "
                 "Arm switch by clause (discovery → validation): "
                 + "; ".join(f"{x['clause']} {f(x['discovery'])} → {f(x['validation'])}" for x in c["arm_switch_by_clause"]))
    L += ["", "## Repeat check against the collection (current wording)", ""]
    for t, v in o["repeat_check"].items():
        if "per_arm" not in v:
            L.append(f"- {t}: {v.get('note')}")
            continue
        for arm, a in v["per_arm"].items():
            L.append(f"- {t}, {arm}: {a['cells_matched']} cells; same strict correctness {f(a['same_strict_correctness'])}; "
                     f"same prompt {f(a['same_prompt_sha256'])}; accuracy difference {f(a['accuracy_difference_pp_mean'], 2)} pp "
                     f"(models {f(a['accuracy_difference_pp_range'][0], 2)} to {f(a['accuracy_difference_pp_range'][1], 2)})")
    v = o["verdicts"]
    L += ["", "## Fixed verdicts", "",
          f"- One prompt sufficed, with both wordings drawn: abstract changes = {v['one_prompt']['abstract_changes']} "
          + "(" + "; ".join(f"{t}: R {f(x['R_both_drawn'])}, prompts {x['prompts_needed_0.9']}" for t, x in v['one_prompt']['per_task'].items()) + ").",
          f"- Arm switch reverses one pair in five across clauses: abstract changes = {v['arm_switch']['abstract_changes']} "
          + "(" + "; ".join(f"{t}: mean {f(x['mean'])}, by clause {', '.join(f(y) for y in x['by_clause'])}" for t, x in v['arm_switch']['per_task'].items()) + ").",
          f"- The direct clause alone moves the ordering as much as the arm: {v['clause_moves_as_much_as_arm']['abstract_must_say']}.",
          f"- The decision sentence gains the wording: {v['decision']['sentence_gains_wording']}.",
          "", f"Runtime {o['runtime_s']} s."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("discovery",))
    ap.add_argument("--base")
    ap.add_argument("--config")
    ap.add_argument("--n-boot", type=int, help="dry runs: resamples for both bootstraps")
    ap.add_argument("--n-splits", type=int, help="dry runs: random sets per K in the ceiling curves")
    ap.add_argument("--fig-w-table", action="store_true",
                    help="write only results/arm_wording/tables/fig_w_vs_requested.csv (and its summary line)")
    args = ap.parse_args(argv)
    paths = aw.Paths(args.base)
    cfg = aw.load_config(args.config)
    try:
        if args.fig_w_table:
            res = write_fig_w_table(paths)
        else:
            res = stage_discovery(args, paths) if args.stage == "discovery" else full(args, paths, cfg)
    except aw.ArmWordingError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
