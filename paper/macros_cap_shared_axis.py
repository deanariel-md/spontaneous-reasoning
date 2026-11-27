"""Macros of the capability slopes of the chain-of-thought gain under both baselines on shared axes, written by scripts/run_cap_shared_axis.py into
results/cap_shared_axis/cap_shared_axis.json, block "macros".

cap_shared_axis_results(root) reads the JSON, or returns None until it exists; cap_shared_axis_macros(R) returns
{name: value}, after checking every name (letters only, prefix capSlope or capPeak).

Names: <Task> in Gsm, Math, Bbh; <Axis> in Eci (Epoch Capabilities Index), Fd (forced-baseline accuracy), Cot
(chain-of-thought accuracy); <Gain> in Req (CoT - requested), Fd (CoT - forced), Diff (Req - Fd); <Sc> empty for strict
scoring, Content for content scoring. Discovery draws (24), panels GSM8K-29, MATH-500 29 models (398 items),
BIG-Bench Hard-28.
  capSlope<Task><Axis><Gain><Sc>            the cross-draw slope, three decimals; pp per index point on Eci, pp per
                                            pp on Fd and Cot
  capSlope<Task><Axis><Gain><Sc>Lo, ...Hi   95% interval over 10,000 model resamples
  capSlope<Task><Axis><Gain><Sc>FamLo, ...FamHi   95% interval over 10,000 family resamples (whole families)
  capSlope<Task><Axis><Gain><Sc>LofoLo, ...LofoHi range over the five leave-one-family-out panels
  capPeak<Task><Axis><Gain><Sc>             x of the interior maximum of the quadratic fit (one decimal), or "none"
  capPeak<Task><Axis><Gain><Sc>Shape        peak, trough, rises throughout or falls throughout (over the x range)
  capPeak<Task><Axis><Gain><Sc>Vertex       the vertex of the fit whatever its shape (one decimal)
  capSlopeModels<Task>, capSlopeDraws<Task>, capSlopeNBoot, capSlopeEciMissing (panel models with no index value)
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "cap_shared_axis", "cap_shared_axis.json")
NAME = re.compile(r"cap(Slope|Peak)[A-Za-z]+")


def cap_shared_axis_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cap_shared_axis_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for k, v in R["macros"].items():
        if not NAME.fullmatch(k):
            raise ValueError(f"macro name {k!r} is not capSlope or capPeak followed by letters")
        out[k] = str(v)
    return out


COUNT_NAMES = ("capCells", "capShallowerCells", "capDiffExclCells")


def cap_count_macros(R) -> dict:
    """Counts over the cells (benchmark x scoring x axis) of R["answer"]:
    capCells, all cells; capShallowerCells, cells whose paired difference (requested - forced slope) is negative,
    the requested slope the shallower; capDiffExclCells, cells whose difference has model and family 95%
    intervals that both exclude zero. The same definitions as the summary's lead (scripts/run_cap_shared_axis.py)."""
    if not R:
        return {}
    A = R["answer"]
    return {"capCells": str(len(A)),
            "capShallowerCells": str(sum(1 for a in A if a["diff"] < 0)),
            "capDiffExclCells": str(sum(1 for a in A if a["diff_excl_model"] and a["diff_excl_family"]))}
