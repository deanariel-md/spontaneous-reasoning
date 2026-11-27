"""Macros of the threshold decision (exploratory), from
results/threshold_decision/threshold_decision.json (scripts/threshold_decision.py).

Each model gets chain of thought when its one-draw pilot gain exceeds c points (c = 2, 5, 10); the pilots are the main
decision analysis's saved one-draw pilots (manifests/predictions_decision_<task>_req|_fd, rule own_1), valued on the
validation configurations no discovery draw uses; strict scores.

threshold_decision_results(root) reads the JSON, or returns None until it exists; threshold_decision_macros(R) returns
{name: value}.
  thrDec<T><Cut>DiffPct     models whose decision differs between the two baselines, mean over pilot sets as a share
                            of the panel, whole percent
  thrDec<T><Cut>DiffCount   the same as a mean count of models, one decimal
  thrDec<T><Cut>FdToReq     added loss of deciding under the forced baseline when deploying under the requested one, pp
                            of accuracy per panel model, two decimals (positive: the wrong baseline costs points), each
                            model given chain of thought valued at its gain minus c (net of c, the primary reading);
                            ...Lo, ...Hi its 95% t interval over validation configurations
  thrDec<T><Cut>ReqToFd     the reverse: deciding under the requested baseline, deploying under the forced one
  thrDec<T><Cut>GrossFdToReq, ...GrossReqToFd (with Lo, Hi)
                            the same losses with each model given chain of thought valued at its gain alone (gross);
                            gross = net + c x (models given chain of thought under the deployment baseline - under the
                            deciding one) / M
<T>: Gsm, Math, Bbh. <Cut>: Two, Five, Ten.
The prefix thrDec does not collide with the threshold sweep's thr<T><Cut>... (paper/macros_threshold_sweep.py), whose
next letters are a benchmark (Gsm, Math, Bbh, Hq), never Dec.
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "threshold_decision", "threshold_decision.json")
PREFIX = "thrDec"
TASK_WORD = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
CUT_WORD = {2: "Two", 5: "Five", 10: "Ten"}
DIRECTION_WORD = {"forced_to_requested": "FdToReq", "requested_to_forced": "ReqToFd"}


def num(x, nd=2):
    """As paper/make_numbers.num: a value that rounds to zero prints unsigned; a minus prints $-$."""
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "$-$")


def threshold_decision_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def threshold_decision_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for t, T in R["tasks"].items():
        for cut in T["cuts"]:
            c = int(round(float(cut["cut"])))
            stem = PREFIX + TASK_WORD[t] + CUT_WORD[c]
            out[stem + "DiffPct"] = num(100 * cut["differ"]["share"], 0)
            out[stem + "DiffCount"] = num(cut["differ"]["mean_count"], 1)
            for pre, blk in (("", cut), ("Gross", cut["gross"])):
                for k, word in DIRECTION_WORD.items():
                    d = blk[k]
                    out[stem + pre + word] = num(d["estimate"], 2)
                    lo, hi = d["ci95"]
                    out[stem + pre + word + "Lo"] = num(lo, 2)
                    out[stem + pre + word + "Hi"] = num(hi, 2)
    for k in out:
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
    return out
