"""Macros of the BIG-Bench Hard letter-text conflict boxes, prefix bbhConflict, written by a conflict-sensitivity script into
results/conflict_sensitivity/conflict_sensitivity.json, block "macros".

conflict_results(root) reads the JSON, or returns None until it exists; conflict_macros(R) returns {name: value},
after checking every name (letters only, prefix bbhConflict).
  bbhConflictBoxes                     the boxes by the letter-text definition, one reading
  bbhConflictBoxesRule                 the boxes the rule of content/1.6.0 treats as such, one reading
  bbhConflictGenuine                   of these, letter one of the item's options: two answers, scored wrong
  bbhConflictLetterNotOption           letter not one of the item's options (the box reads its text)
  bbhConflictLetterNotOptionTextRight  of these, the text is right
  bbhConflictLetterRight, bbhConflictTextRight   right by the letter, by the text (letter-text definition)
  bbhConflictMaxRevChange              the largest change of a BIG-Bench Hard reversal count (panel, six, 70B and
                                       above, combined) between the letter and the text readings
  bbhConflictMaxRevChangeAny           the same over any two of the three readings
  bbhConflictMacrosChanged             macros of paper/numbers.tex that differ between the three readings
  bbhConflict<Count><Reading>          each reversal count (CombPanelRevBbh, LadRevBbh, LadBigRevBbh, CombRevBbh) under
                                       each reading (Wrong, Letter, Text)
  bbhConflictDecMaxChange, ...SplitMaxChange, ...CorrMaxChange
                                       the largest change over any two readings of a BIG-Bench Hard macro of the
                                       decision costs (dec, ddir; pp), the split pilot (splitp) and the (1 - w)
                                       correction (bmix and lad macros holding "Corr"); ...MacrosChanged, how many of
                                       each group differ
  bbhConflictOtherRevMaxChange         the largest change of any other count of reversed pairs (threshold rows,
                                       scorer and check counts); ...OtherRevMacrosChanged, how many differ
"""
from __future__ import annotations

import json
import os
import re

PREFIX = "bbhConflict"
PATH = os.path.join("results", "conflict_sensitivity", "conflict_sensitivity.json")


def conflict_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def conflict_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for k, v in R["macros"].items():
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
        out[k] = str(v)
    return out
