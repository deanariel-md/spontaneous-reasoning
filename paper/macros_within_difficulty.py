"""Macros of within-model difficulty and rewording.

Prefix diff: does a model write long requested answers (n_output_tokens > 16, the definition behind w) more
often on harder items, within itself? From results/within_difficulty/within_difficulty.json
(scripts/run_within_difficulty.py): binomial logit of long on item difficulty with model fixed effects, difficulty
= 100 x the share of the other models' forced answers on the item that are wrong (leave-this-model-out; larger is
harder), discovery draws weighted as w is.
  diffOr<T>, diffOr<T>Lo, diffOr<T>Hi   odds ratio per 10 points of difficulty and its 95% model-resampling interval
                                        (two decimals); strict scoring of the forced answers
  diffPos<T>                            models whose own slope (one logit per model) is positive
  diffModels<T>                         models in the fit (those with both long and short requested answers)
  ...Content                            the same with difficulty from content scoring (diffOr<T>Content,
                                        diffOr<T>ContentLo, diffOr<T>ContentHi, diffPos<T>Content, diffModels<T>Content)
  <T> is Gsm (GSM8K-29), Math (MATH-500 numeric) or Bbh (BIG-Bench Hard-28).
  diffOrBbhSub, ...Lo, ...Hi, diffPosBbhSub, diffModelsBbhSub, and the same ending in Content (diffOrBbhSubContent,
                                        diffOrBbhSubContentLo, ...): BIG-Bench Hard with one effect per subtask added
                                        (pooled fit and each model's own fit), strict or content scoring
Prefix reword: rewording (results/arm_wording/within_difficulty.json, scripts/arm_wording_within_difficulty.py).
  rewordModelsGsm, rewordModelsMath     models whose w spans at least 0.2 across the four direct wordings

within_difficulty_results(root) reads both files, or returns None until within_difficulty.json exists;
within_difficulty_macros(R) returns {name: value}.
"""
from __future__ import annotations

import json
import os
import re

TASKS = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"))
LINES = (("main", ""), ("content", "Content"))
BBH_SUB = (("bbh_subtask", "Sub"), ("bbh_subtask_content", "SubContent"))
PATH = os.path.join("results", "within_difficulty", "within_difficulty.json")
FIG = os.path.join("results", "arm_wording", "within_difficulty.json")


def _load(root, rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def within_difficulty_results(root: str):
    wd = _load(root, PATH)
    if wd is None:
        return None
    return {"wd": wd, "fig": _load(root, FIG)}


def f2(x) -> str:
    """Two decimals, as make_numbers.num prints them ($-$ for a minus)."""
    x = float(x)
    if abs(x) < 0.005:
        x = 0.0
    return f"{x:.2f}".replace("-", "$-$")


def within_difficulty_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    T = R["wd"]["tasks"]
    cells = [(t, tag, line, suf) for t, tag in TASKS for line, suf in LINES]
    cells += [("bbh", "Bbh", line, suf) for line, suf in BBH_SUB if line in T["bbh"]["lines"]]
    for t, tag, line, suf in cells:
        L = T[t]["lines"][line]
        p = L["pooled"]
        out[f"diffOr{tag}{suf}"] = f2(p["odds_ratio"])
        out[f"diffOr{tag}{suf}Lo"] = f2(p["odds_ratio_ci95"][0])
        out[f"diffOr{tag}{suf}Hi"] = f2(p["odds_ratio_ci95"][1])
        out[f"diffPos{tag}{suf}"] = str(int(L["positive"]))
        out[f"diffModels{tag}{suf}"] = str(int(L["models_in_fit"]))
    if R.get("fig"):
        for t, tag in (("gsm8k", "Gsm"), ("math", "Math")):
            out[f"rewordModels{tag}"] = str(int(R["fig"]["counts"][t]["qualify"]))
    for k in out:
        if not re.fullmatch(r"(diff|reword)[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not letters only with prefix diff or reword")
    return out


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    for k, v in sorted(within_difficulty_macros(within_difficulty_results(ROOT)).items()):
        print(f"\\{k} = {v}")
