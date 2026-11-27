"""Macros of the flagship model's gain intervals (Qwen4-Max), read from results/frontier_panel/closed_w_intervals.json
(scripts/closed_w_intervals.py). New names only; the point values of the gains stay those of
paper/macros_closed_ladder.py (cwReqGain<T>, cwFdGain<T>, cwCorr<T>), which this file's points equal bit for bit.

Discovery draws' template weights, 100 questions per benchmark, 95% percentile intervals from 2,000 draws of the
questions (paired across arms; BIG-Bench Hard within task; templates fixed). pp, one decimal. T is Gsm, Math, Bbh
(strict scoring); BbhContent is BIG-Bench Hard with content scoring.
  cwReqGain<T>Lo, ...Hi     requested-baseline gain C - R
  cwFdGain<T>Lo, ...Hi      forced-baseline gain C - F
  cwGainDiff<T>             the difference of the two gains, forced-baseline minus requested-baseline,
                            (C - F) - (C - R) = R - F (the requested minus the forced accuracy)
  cwGainDiff<T>Lo, ...Hi    its interval
  cwCorr<T>Lo, ...Hi        corrected gain (1 - w)(C - F)

cw_intervals_results(root) reads the JSON, or returns None until it exists; cw_intervals_macros(R) returns
{name: value}.
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "frontier_panel", "closed_w_intervals.json")
CELLS = (("gsm8k", "strict", "Gsm"), ("math", "strict", "Math"), ("bbh", "strict", "Bbh"),
         ("bbh", "content", "BbhContent"))
QUANT = (("cwReqGain", "cot_minus_requested"), ("cwFdGain", "cot_minus_forced"), ("cwGainDiff", "gain_diff"),
         ("cwCorr", "corrected_gain"))


def cw_intervals_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def f1(x):
    """As make_numbers.f(x, 1): a hyphen prints $-$."""
    return f"{x:.1f}".replace("-", "$-$")


def cw_intervals_macros(R) -> dict:
    if not R:
        return {}
    if not R.get("all_equal_to_closed_w"):
        raise ValueError("closed_w_intervals.json does not reproduce closed_w.json; rerun scripts/closed_w_intervals.py")
    out = {}
    for task, score, T in CELLS:
        hit = [c for c in R["cells"] if c["task"] == task and c["score"] == score]
        if len(hit) != 1:
            raise ValueError(f"{len(hit)} cells for {task} {score}")
        c = hit[0]
        for name, q in QUANT:
            ci = c["ci"][q]
            if ci["lo"] is None or ci["hi"] is None or ci["finite_draws"] != ci["draws"]:
                raise ValueError(f"{task} {score} {q}: interval from {ci['finite_draws']} of {ci['draws']} draws")
            if name == "cwGainDiff":
                out[f"{name}{T}"] = f1(100 * c["point"][q])
            out[f"{name}{T}Lo"] = f1(100 * ci["lo"])
            out[f"{name}{T}Hi"] = f1(100 * ci["hi"])
    for k in out:
        if not re.fullmatch(r"cw[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not cw followed by letters")
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for k, v in sorted(cw_intervals_macros(cw_intervals_results(root)).items()):
        print(f"{k} = {v}")
