r"""Size of the arm effect: how large the reversed pairs are, and how much of requested − forced tracks reasoning in
the direct arm.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_arm_effect.py 2>&1"

Outputs: results/arm_effect/arm_effect.json, summary.md, summary_clean.md, tables/per_model.csv.

A task whose scoring axis (config/added_contrasts.yaml, section tasks) has token F1 (HotpotQA) also gets the key
`token_f1`: the same reversal count and the same short-answer stratum (the
models whose long share is below 10%, read from lengths only) with every arm scored by token F1 in place of strict exact
match. `--f1-only` adds that key to an existing arm_effect.json without recomputing anything else:
    cmd /c ".venv\Scripts\python.exe scripts\run_arm_effect.py --f1-only 2>&1"
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import robustness as rb  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "arm_effect")
FORCED_CAP = 16
SHORT_SHARE = 10.0


def forced_cap(task):
    """The forced arm's output cap: 16 tokens on the main tasks; an added task's own (config/added_contrasts.yaml)."""
    tk = {} if task in ri.TASKS else __import__("added_contrasts").load_tasks()
    return int(tk[task]["forced_cap"]) if task in tk else FORCED_CAP


def long_shares(task, panel):
    """Per model, the % of requested-direct outputs (both splits) longer than the forced arm's cap (16 tokens; an added
    task's own), and the median output length."""
    from run_forced_direct import collection_inputs
    inp, s, _, _ = collection_inputs(task)
    units = [str(u) for u in s["discovery_config_ids"]] + [str(u) for u in s["validation_config_ids"]]
    rows = inp["load"](units)
    d = rows[rows["arm"] == "direct_answer"]
    g = d.groupby("model_key")["n_output_tokens"]
    share = (d["n_output_tokens"] > forced_cap(task)).groupby(d["model_key"]).mean().mul(100)
    return {k: float(share[k]) for k in panel}, {k: float(g.median()[k]) for k in panel}


def f1_column(task):
    """The token-F1 column of an added task's scoring axis (HotpotQA: strict_f1), or None."""
    if task in ri.TASKS:
        return None
    return dict(__import__("added_contrasts").load_tasks()[task].get("scorings") or {}).get("f1")


def discovery_arrays(task, panel, value):
    """(requested, forced, cot), each (24, M) pp, on the discovery draws (repeats kept, as robustness_inputs builds
    them) under the scored column `value`."""
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    inp, s, items, items_info = collection_inputs(task)
    if list(inp["panel"]) != list(panel):
        raise SystemExit(f"{task}: the panel differs from robustness_inputs'")
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    return three_arm_arrays(inp["load"], fd_load, list(s["discovery_config_ids"]), panel, items, value=value)


def f1_block(task, panel, short_models, arrays=None):
    """The arm effect under token F1: model means over the discovery draws, pair reversals of chain of thought's gain
    over the requested and over the forced arm (same pairs and tie rule as the strict count), and requested - forced
    on the same short-answer stratum (chosen from lengths only, so it does not move with the scorer)."""
    col = f1_column(task)
    if col is None:
        return None
    breq, bfd, pc = arrays if arrays is not None else discovery_arrays(task, panel, col)
    req, fd, cot = breq.mean(0), bfd.mean(0), pc.mean(0)
    rmf = req - fd
    short = np.array([k in set(short_models) for k in panel])
    return {"scoring": f"token F1 ({col}), pp", "reversals": rb.pair_reversals(cot - req, cot - fd),
            "mean_req_minus_forced": float(rmf.mean()),
            "short_models_req_minus_forced_range": [float(rmf[short].min()), float(rmf[short].max())] if short.any() else None,
            "short_models_req_minus_forced_mean": float(rmf[short].mean()) if short.any() else None,
            "short_models_n": int(short.sum()),
            "short_models_req_minus_forced_below_zero": int((rmf[short] < 0).sum()),
            "per_model": {k: {"requested": float(req[j]), "forced": float(fd[j]), "cot": float(cot[j])}
                          for j, k in enumerate(panel)}}


def add_f1(path=None):
    """--f1-only: add `token_f1` to each added task of an existing arm_effect.json (no other value is recomputed) and
    rewrite the summaries."""
    import json
    path = path or os.path.join(OUT, "arm_effect.json")
    R = json.load(open(path, encoding="utf-8"))
    for task in ri.added_tasks():
        T = R["tasks"].get(task)
        if T is None:
            continue
        panel = ri.load(task)["facts"]["panel"]
        b = f1_block(task, panel, T["short_models"])
        if b is not None:
            T["token_f1"] = b
    ri.write_outputs(OUT if path is None else os.path.dirname(path), R, summary(R))
    return R


def run():
    t0 = time.time()
    R = {"analysis": "size of the arm effect", "created": time.strftime("%Y-%m-%d %H:%M"),
         "forced_cap_tokens": FORCED_CAP, "short_share_pct": SHORT_SHARE, "tasks": {}}
    per = []
    for task in ri.TASKS + ri.added_tasks():       # an added task (HotpotQA) after the main three
        x = ri.load(task)
        f, d = x["facts"], x["discovery"]
        panel = f["panel"]
        req, fd, cot = (d[a].mean(0) for a in ("requested", "forced", "cot"))
        rev = rb.pair_reversals(cot - req, cot - fd)
        # an added task has no readability table (readable_pct None): no readable panel
        rd = None if f["readable_pct"] is None else np.array([p >= ri.CLEAN_MIN for p in f["readable_pct"]])
        rev_rd = rb.pair_reversals((cot - req)[rd], (cot - fd)[rd]) if rd is not None and rd.sum() >= 2 else None
        share, med = long_shares(task, panel)
        sh = np.array([share[k] for k in panel])
        rmf = req - fd
        short = sh < SHORT_SHARE
        T = {"name": ri.NAMES[task], "M": len(panel), "reversals": rev, "readable_M": None if rd is None else int(rd.sum()),
             "reversals_readable": rev_rd,
             "pearson_long_share_vs_req_minus_forced": float(np.corrcoef(sh, rmf)[0, 1]),
             "spearman_long_share_vs_req_minus_forced": float(stats.spearmanr(sh, rmf).statistic),
             "short_models": [k for k, s_ in zip(panel, short) if s_],
             "short_models_req_minus_forced_range": [float(rmf[short].min()), float(rmf[short].max())] if short.any() else None,
             "short_models_req_minus_forced_mean": float(rmf[short].mean()) if short.any() else None,
             "mean_req_minus_forced": float(rmf.mean())}
        if task not in ri.TASKS:
            T["forced_cap_tokens"] = forced_cap(task)
            b = f1_block(task, panel, T["short_models"])
            if b is not None:
                T["token_f1"] = b
        R["tasks"][task] = T
        for j, k in enumerate(panel):
            per.append({"task": task, "model_key": k, "requested": req[j], "forced": fd[j], "cot": cot[j],
                        "req_minus_forced": rmf[j], "requested_long_share_pct": sh[j],
                        "requested_median_tokens": med[k]})
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/arm_effect", flush=True)


def summary(R):
    L = ["# Size of the arm effect", "",
         "Model means over the 24 discovery draws, strict scoring. A pair "
         "reverses when the chain-of-thought gain over the requested arm orders the two models one way and the gain "
         "over the forced arm the other way; a tie (|gap| ≤ 1e-9 pp) under either is neither. 'Gap above t' means both "
         "gaps exceed t pp. The long share is the share of requested-direct outputs (both splits) longer than the "
         f"forced arm's {R['forced_cap_tokens']}-token cap" + "".join(
             f"; on {T['name']}, its own forced cap of {T['forced_cap_tokens']} tokens" for T in R["tasks"].values()
             if "forced_cap_tokens" in T) + ". On BIG-Bench Hard a long requested output can be the option "
         "text written out, not reasoning.", "",
         "## Reversed pairs by gap size", "",
         "| Task | Pairs | Reversed | Share | Gap > 1 pp | > 2 pp | > 5 pp | > 10 pp |", "|---|---|---|---|---|---|---|---|"]
    for t, T in R["tasks"].items():
        v = T["reversals"]
        bt = {b["min_gap_pp"]: b for b in v["by_threshold"]}
        cells = " | ".join(f"{bt[g]['reversed']} of {bt[g]['pairs_both_above']}" for g in (1.0, 2.0, 5.0, 10.0))
        L.append(f"| {T['name']} | {v['pairs']} | {v['reversed']} | {rb.fmt(v['reversed_share'], 3)} | {cells} |")
    L += ["", "Each 'r of n' cell counts the reversed pairs among the n pairs whose two gaps both exceed the threshold.", "",
          "Among each task's readable models (>= 95% closed and readable forced answers), where unreadable answers do "
          "not lower the forced arm's accuracy:", ""]
    for t, T in R["tasks"].items():
        v = T["reversals_readable"]
        if v is None:
            L.append(f"- {T['name']}: " + ("no readable panel is defined (no readability table for this task)."
                                          if T["readable_M"] is None else "fewer than two readable models."))
            continue
        bt = {b["min_gap_pp"]: b for b in v["by_threshold"]}
        L.append(f"- {T['name']} ({T['readable_M']} models): {v['reversed']} of {v['pairs']} pairs reverse "
                 f"({rb.fmt(v['reversed_share'], 3)}); with both gaps above 5 pp, {bt[5.0]['reversed']} of "
                 f"{bt[5.0]['pairs_both_above']}.")
    L += ["",
          "## Requested − forced against reasoning in the requested arm", "",
          "| Task | Mean requested − forced (pp) | Pearson with long share | Spearman | Models with long share < 10% | "
          "Their requested − forced (range, pp) |", "|---|---|---|---|---|---|"]
    for t, T in R["tasks"].items():
        rg = T["short_models_req_minus_forced_range"]
        L.append(f"| {T['name']} | {rb.fmt(T['mean_req_minus_forced'], 1)} | {rb.fmt(T['pearson_long_share_vs_req_minus_forced'], 2)} | "
                 f"{rb.fmt(T['spearman_long_share_vs_req_minus_forced'], 2)} | {len(T['short_models'])} | "
                 + (f"{rb.fmt(rg[0], 1)} to {rb.fmt(rg[1], 1)}" if rg else "none") + " |")
    L += ["", "Per model: `results/arm_effect/tables/per_model.csv`."]
    f1 = [(T, T["token_f1"]) for T in R["tasks"].values() if T.get("token_f1")]
    if f1:
        L += ["", "## Under token F1", "",
              "The same pairs, tie rule and short-answer stratum, with every arm scored by token F1 (pp) in place of "
              "strict exact match.", "",
              "| Task | Reversed (of pairs) | Share | Ties | Mean requested − forced (pp) | Short-answer models | "
              "Their requested − forced: mean (range), pp | Below zero |", "|---|---|---|---|---|---|---|---|"]
        for T, b in f1:
            v, rg = b["reversals"], b["short_models_req_minus_forced_range"]
            L.append(f"| {T['name']} | {v['reversed']} of {v['pairs']} | {rb.fmt(v['reversed_share'], 3)} | "
                     f"{v['ties']} | {rb.fmt(b['mean_req_minus_forced'], 1)} | {b['short_models_n']} | "
                     + (f"{rb.fmt(b['short_models_req_minus_forced_mean'], 1)} ({rb.fmt(rg[0], 1)} to {rb.fmt(rg[1], 1)})"
                        if rg else "none") + f" | {b['short_models_req_minus_forced_below_zero']} |")
    return L


if __name__ == "__main__":
    if "--f1-only" in sys.argv[1:]:
        add_f1()
    else:
        run()
