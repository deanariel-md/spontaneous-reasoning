"""Macros of who gains more from chain of thought under each baseline (exploratory), written by scripts/run_cap_reversal.py into results/cap_reversal/cap_reversal.json, block "macros".

cap_reversal_results(root) reads the JSON, or returns None until it exists; cap_reversal_macros(R) returns
{name: value}, after checking every name (letters only, prefix crv).

Names: <Task> in Gsm, Math, Bbh; <Sc> empty for strict scoring, Content for content scoring. Discovery draws (24),
panels GSM8K-29, MATH-500 29 models (398 items), BIG-Bench Hard-28. Gains: Req = CoT - requested, Fd = CoT - forced,
Diff = Req - Fd, Corr = (1 - w) x Fd. Every value is exploratory.
  crv<Task>Own{Req,Fd,Diff,Corr}<Sc>      own-baseline axes: Req and Corr on requested accuracy, Fd on forced accuracy
                                          (three decimals, pp per pp)
  crv<Task>OwnShare<Sc>                   (Corr - Fd) / Diff in percent (no decimals): the share of the own-axis
                                          difference that the (1 - w) correction reproduces
  crv<Task>ReqAcc{Req,Fd,Diff}<Sc>        both gains on requested accuracy (three decimals)
  crv<Task>ReqAccOpp<Sc>, ...Fam          percent of model (family) resamples with Req < 0 < Fd on requested accuracy
  crv<Task>Size{Req,Fd,Diff}<Sc>          on log2 total parameters, pp per doubling (two decimals)
  crv<Task>Year{Req,Fd,Diff}<Sc>          on release year, pp per year (two decimals)
  crv<Task>W{Year,Size}                   Pearson r of w with release year and with log2 parameters (two decimals)
  each of the above (except Opp) with suffixes Lo, Hi (95% over 10,000 model resamples), FamLo, FamHi (10,000 family
  resamples), LofoLo, LofoHi (range over the five leave-one-family-out panels)
  crvOwnShareMin, crvOwnShareMax          range of OwnShare over the six benchmark and scoring cells (percent)
  crvOwnShareStrictMin, ...StrictMax      the same over the three strict cells
  crvItemSplitMax, crvItemSplitMaxDiff    largest change of a requested-axis slope (of the difference) when the
                                          questions are split between axis and gain (cited from the axis probe)
  crvModels<Task>, crvDraws<Task>, crvNBoot
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "cap_reversal", "cap_reversal.json")
NAME = re.compile(r"crv[A-Za-z]+")


def cap_reversal_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cap_reversal_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for k, v in R["macros"].items():
        if not NAME.fullmatch(k):
            raise ValueError(f"macro name {k!r} is not crv followed by letters")
        out[k] = str(v)
    return out
