"""Macros for the values of the closed model and the ladder that the manuscript prints.
Every value is read from a results file; nothing is typed here.

  cw*        the closed model (Qwen4-Max), results/frontier_panel/closed_w.json (scripts/closed_w.py report),
             discovery draws, strict unless the name ends in Content; w at the forced cap (16 tokens, HotpotQA 64);
             the repeat check from the pilot, results/frontier_panel/closed_w_pilot.json
  fsCheck*   the forced-stop completion rule against the API models' whole replies, results/forced_stop/
             full_replies.json, the current rule version, all replies: boxes compared, and of these, read with
             another verdict than the box the model closed (compared - agree)
  lad*       results/ladder/analysis/ladder.json (rows per model, sizes, placement against the panel's range),
             results/ladder/equiv_<model>.json (the agreement check of the token-level cut),
             results/frontier_panel/nebius_forced_check.json (Nebius repeat pairs; DeepSeek five-shot forced
             replies with no answer in their box)
  ladSize*   ladder.json's sizes block, which scripts/ladder_analysis.py reads from the records with their sources
             (B, as the source writes them). Published sizes from
             manifests/ladder_published_sizes.json (page, URL, read date, quoted sentences): ladSizeDeepSeekTotalB,
             ladSizeDeepSeekActiveInB, ladSizeDeepSeekActiveOutB (DeepSeek's release note of V4.1-Flash; the ladder
             ran the 1017 checkpoint, taken to share them), ladSizeMistralLargeActiveB (Mistral Large 3's model card).
             ladSizeDeepSeekB is V4-Flash's recorded size, the predecessor record ladder.json keeps marked not used

closed_ladder_results(root) reads the files, or returns None until closed_w.json exists; closed_ladder_macros(R) returns
{name: value}. Run as a script, it also prints where Qwen4-Max places among the models it is compared with.
"""
from __future__ import annotations

import json
import os
import re

TASKS = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"), ("hotpot", "Hq"))
LADDER6 = ("llama-3.1-70b-instruct-fp8", "qwen3.5-122b-a10b-fp8", "ministral-3-8b-instruct-2512",
           "ministral-3-14b-instruct-2512", "mistral-large-2512", "deepseek-v4.1-flash-1017")


def _load(root, *parts):
    p = os.path.join(root, *parts)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def closed_ladder_results(root: str):
    cw = _load(root, "results", "frontier_panel", "closed_w.json")
    if cw is None:
        return None
    return {
        "cw": cw,
        "pilot": _load(root, "results", "frontier_panel", "closed_w_pilot.json"),
        "full": _load(root, "results", "forced_stop", "full_replies.json"),
        "ladder": _load(root, "results", "ladder", "analysis", "ladder.json"),
        "eq_llama": _load(root, "results", "ladder", "equiv_llama-3.1-70b-instruct-fp8.json"),
        "eq_qwen": _load(root, "results", "ladder", "equiv_qwen3.5-122b-a10b-fp8.json"),
        "nebius": _load(root, "results", "frontier_panel", "nebius_forced_check.json"),
        "items": _items(root),
    }


def _items(root):
    """Items per benchmark of the closed-model run (config/closed_w.yaml, design.items_per_benchmark)."""
    import yaml
    with open(os.path.join(root, "config", "closed_w.yaml"), encoding="utf-8") as fh:
        return int(yaml.safe_load(fh)["design"]["items_per_benchmark"])


def f1(x, nd=1):
    """As make_numbers.f: a hyphen prints $-$ (a value that rounds to zero keeps its sign, as the manuscript prints it)."""
    return f"{x:.{nd}f}".replace("-", "$-$")


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def _one(rows, **kw):
    hit = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} rows for {kw}")
    return hit[0]


def _walk(x, key):
    """Every value of `key` anywhere in a JSON tree."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v, key)


def cw_macros(R) -> dict:
    cw = R["cw"]
    m = "qwen4-max"
    out = {}
    arms = cw["models"][m]["arms"]
    out["cwRequestsArm"] = comma(arms["requested"]["expected"])
    out["cwRequests"] = comma(sum(a["expected"] for a in arms.values()))
    cells = [r for r in cw["per_template"] if r["model"] == m and r["score"] == "strict"]
    out["cwCells"] = str(len(cells))
    # models ranked together, Qwen4-Max included: 36 on GSM8K, MATH-500 and HotpotQA; one fewer on BIG-Bench Hard,
    # where the panel has 28 models
    for t, T in TASKS:
        Ms = {p["panel_and_ladder"]["M"] for p in cw["placement"] if p["model"] == m and p["task"] == t}
        if len(Ms) != 1:
            raise ValueError(f"{t}: placement compares with different numbers of models: {Ms}")
        out[f"cwModels{T}"] = str(Ms.pop())
    out["cwModels"] = out["cwModelsGsm"]
    out["cwItems"] = str(R["items"])
    for t, T in TASKS:
        g = _one(cw["gains"], model=m, task=t, score="strict", basis="discovery")
        out[f"cwW{T}Pct"] = f"{100 * g['w']:.0f}"
        out[f"cwReqAcc{T}"] = f1(100 * g["requested"])
        out[f"cwFdAcc{T}"] = f1(100 * g["forced"])
        out[f"cwCotAcc{T}"] = f1(100 * g["cot"])
        out[f"cwReqGain{T}"] = f1(100 * g["cot_minus_requested"])
        out[f"cwFdGain{T}"] = f1(100 * g["cot_minus_forced"])
        out[f"cwCorr{T}"] = f1(100 * g["corrected_gain"])
        out[f"cwResid{T}"] = f1(100 * g["residual"])
    g = _one(cw["gains"], model=m, task="bbh", score="content", basis="discovery")
    out["cwFdGainBbhContent"] = f1(100 * g["cot_minus_forced"])
    out["cwReqGainBbhContent"] = f1(100 * g["cot_minus_requested"])
    out["cwCorrBbhContent"] = f1(100 * g["corrected_gain"])
    out["cwResidBbhContent"] = f1(100 * g["residual"])
    if R["pilot"] is not None:
        rep = [c for c in R["pilot"]["models"][m]["checks"] if c["name"] == "repeat_agreement"]
        if len(rep) != 1:
            raise ValueError("pilot: no single repeat_agreement check")
        q = rep[0]["detail"]["per_arm"]["requested"]
        out["cwRepeatPairs"] = str(q["pairs"])
        same = q["crossing"] * q["pairs"]
        if abs(same - round(same)) > 1e-9:
            raise ValueError(f"crossing share {q['crossing']} is not a count of {q['pairs']} pairs")
        out["cwRepeatSameSide"] = str(round(same))
    return out


def fs_macros(R) -> dict:
    full = R["full"]
    if full is None:
        return {}
    v = full["versions"][-1]
    a = full["totals"][v]["all"]
    return {"fsCheckBoxes": comma(a["compared"]), "fsCheckDiffer": comma(a["compared"] - a["agree"])}


def lad_macros(R) -> dict:
    lad, out = R["ladder"], {}
    if lad is None:
        return out
    comp = lad["completeness"]
    rows = {comp[k]["design_rows"] for k in LADDER6}
    if len(rows) != 1 or any(set(comp[k]["scored_rows"].values()) != rows for k in LADDER6):
        raise ValueError(f"ladder models differ in design rows: {rows}")
    out["ladRequests"] = comma(rows.pop())
    sz = lad["sizes"]
    totals = [float(sz[k]["total"]) for k in LADDER6 if sz[k]["total"] is not None]
    out["ladSizeMaxB"] = f"{max(totals):g}"
    out["ladSizeQwenB"] = f"{float(sz['qwen3.5-122b-a10b-fp8']['total']):.0f}"
    out["ladSizeQwenActiveB"] = f"{float(sz['qwen3.5-122b-a10b-fp8']['active']):g}"
    out["ladSizeMinistralEightB"] = f"{float(sz['ministral-3-8b-instruct-2512']['total']):.1f}"
    out["ladSizeMinistralFourteenB"] = f"{float(sz['ministral-3-14b-instruct-2512']['total']):.1f}"
    ds, ml = sz["deepseek-v4.1-flash-1017"], sz["mistral-large-2512"]
    for k, x in (("DeepSeek", ds), ("Mistral Large 3", ml)):
        if not x.get("published") or not x.get("url") or not x.get("read"):
            raise ValueError(f"{k}: ladder.json holds no published size with its page and read date")
    out["ladSizeDeepSeekTotalB"] = f"{float(ds['total']):g}"
    out["ladSizeDeepSeekActiveInB"] = f"{float(ds['active_input']):g}"
    out["ladSizeDeepSeekActiveOutB"] = f"{float(ds['active_output']):g}"
    out["ladSizeMistralLargeActiveB"] = f"{float(ml['active']):g}"
    pre = ds["predecessor"]                    # V4-Flash, kept for the record (used: false); no longer in the text
    if pre.get("used") is not False:
        raise ValueError("ladder.json: DeepSeek's predecessor record is not marked as not used")
    out["ladSizeDeepSeekB"] = f"{float(pre['total']):g}"
    cells = [lad["placement"][k][t]["cut"]["content"]["cot_minus_forced"]["outside_panel_range"]
             for k in LADDER6 for t, _ in TASKS]
    out["ladCells"] = str(len(cells))
    out["ladInRangeCells"] = str(sum(c is None for c in cells))
    ql, qq = R["eq_llama"], R["eq_qwen"]
    if ql is not None and qq is not None:
        if ql["rows"] != qq["rows"]:
            raise ValueError("the two agreement checks have different row counts")
        out["ladEquivRows"] = str(qq["rows"])
        out["ladEquivLlamaSame"] = str(ql["identical"])
        out["ladEquivQwenSame"] = str(qq["identical"])
        out["ladEquivQwenVerdict"] = str(qq["scored"]["content_verdict_same"])
    nb = R["nebius"]
    if nb is not None:
        s = nb["determinism"]["strict_answer"]["gsm8k"]
        out["ladNebiusPairs"] = str(s["pairs"])
        out["ladNebiusSameAnswer"] = str(s["pairs"] - s["strict_score_differs"])
        p = nb["placeholders"]["bbh|fewshot_forced"]
        out["ladDsFsNoAnswerPct"] = f"{100 * p['rows'] / p['n_arm']:.1f}"
    return out


def closed_ladder_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for block in (cw_macros(R), fs_macros(R), lad_macros(R)):
        for k, v in block.items():
            if k in out:
                raise ValueError(f"macro {k} set twice")
            out[k] = v
    for k in out:
        if not re.fullmatch(r"(cw|fsCheck|lad)[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} has no expected prefix")
    return out


def placement_report(R) -> list[str]:
    """Where Qwen4-Max ranks among the models it is compared with, discovery draws (rank 1 = highest)."""
    lines = []
    for p in R["cw"]["placement"]:
        if p["model"] != "qwen4-max" or p["basis"] != "discovery":
            continue
        pl = p["panel_and_ladder"]
        lines.append(f"{p['task']:7s} {p['score']:8s} {p['quantity']:22s} rank {pl['k']} of {pl['M']}"
                     f"{' ties ' + str(pl['ties']) if pl['ties'] else ''}")
    return lines


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    R = closed_ladder_results(root)
    for k, v in sorted(closed_ladder_macros(R).items()):
        print(f"{k} = {v}")
    print()
    for line in placement_report(R):
        print(line)
