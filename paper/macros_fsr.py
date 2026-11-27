"""Macros of the five-shot requested-direct arm, prefix fsr.

fsr_results(root) reads results/fewshot_requested/fewshot_requested.json, or returns None until the full run has written
it; fsr_macros(R) returns {name: value}. To use it in paper/make_numbers.py:
    import macros_fsr
    _fsr = macros_fsr.fsr_results(ROOT)
    if _fsr is not None:
        _more.append(macros_fsr.fsr_macros(_fsr))

Names, per task T (Gsm, Bbh), strict scoring, validation configurations unless marked:
  fsr<T>GainReq, fsr<T>GainFd           mean five-shot gain under the requested and the forced procedure (pp)
  fsr<T>Rev, fsr<T>RevPairs, fsr<T>RevPct, fsr<T>RevTies
                                        model pairs reversed between the two procedures' gains, pairs, percent, ties
  fsr<T>RevBootLo/Hi, fsr<T>RevFamBootLo/Hi, fsr<T>RevLofoMin/Max   model and family bootstrap, one family left out
  fsr<T>RevDisc, fsr<T>RevDiscPct       the same count on the discovery configurations
  fsr<T>RevHolds                        "holds" or "does not hold": the saved share statement
  fsr<T>TplReq, fsr<T>TplFd             template-only 1 - C within each procedure
  fsr<T>ReqToFd, fsr<T>FdToReq (+Lo, Hi)  decision cost, one-draw choice, pp per chosen model, t interval
  fsr<T>MeanReqToFd, fsr<T>MeanFdToReq (+Lo, Hi)  the same for the four-configuration choice
  fsr<T>RevLenient                      reversed pairs under lenient scoring
  fsrValConfigs                         validation configurations per task (4)
Zero reversals print as 0; an undefined value prints as "--".
"""
from __future__ import annotations

import json
import math
import os

TASK = {"gsm8k": "Gsm", "bbh": "Bbh"}


def num(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    if round(v, nd) == 0:
        v = 0.0
    return f"{v:.{nd}f}".replace("-", "$-$")


def fsr_results(root):
    p = os.path.join(root, "results", "fewshot_requested", "fewshot_requested.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _lohi(m, stem, ci, nd=2):
    m[stem + "Lo"] = num(ci[0], nd) if ci else "--"
    m[stem + "Hi"] = num(ci[1], nd) if ci else "--"


def fsr_macros(R):
    m = {}
    for task, b in R["tasks"].items():
        t = "fsr" + TASK[task]
        s = b["strict"]
        m[t + "GainReq"] = num(s["mean_gain_validation"]["requested"])
        m[t + "GainFd"] = num(s["mean_gain_validation"]["forced"])
        rv, rd, rs = s["reversal_validation"], s["reversal_discovery"], s["reversal_validation_resampling"]
        m[t + "Rev"] = str(int(rv["reversed"]))
        m[t + "RevPairs"] = str(int(rv["pairs"]))
        m[t + "RevPct"] = num(100 * rv["reversed_share"], 1)
        m[t + "RevTies"] = str(int(rv["ties"]))
        _lohi(m, t + "RevBoot", [100 * x for x in rs["model_bootstrap"]["ci95"]] if rs["model_bootstrap"]["ci95"] else None, 1)
        _lohi(m, t + "RevFamBoot",
              [100 * x for x in rs["family_bootstrap"]["ci95"]] if rs["family_bootstrap"]["ci95"] else None, 1)
        lo, hi = rs["leave_one_family_out_range"]
        m[t + "RevLofoMin"], m[t + "RevLofoMax"] = num(100 * lo, 1), num(100 * hi, 1)
        m[t + "RevDisc"] = str(int(rd["reversed"]))
        m[t + "RevDiscPct"] = num(100 * rd["reversed_share"], 1)
        m[t + "RevHolds"] = "holds" if s["statement_share"]["holds"] else "does not hold"
        m[t + "TplReq"] = num(s["template_reversal_validation"]["requested"]["one_minus_C"], 3)
        m[t + "TplFd"] = num(s["template_reversal_validation"]["forced"]["one_minus_C"], 3)
        for name, pre in (("decision_one_draw", ""), ("decision_mean_choice", "Mean")):
            for k, tag in (("requested_to_forced", "ReqToFd"), ("forced_to_requested", "FdToReq")):
                d = s[name][k]
                m[t + pre + tag] = num(d["estimate"])
                _lohi(m, t + pre + tag, d.get("ci95"))
        m[t + "RevLenient"] = str(int(b["lenient"]["reversal_validation"]["reversed"]))
    m["fsrValConfigs"] = str(int(next(iter(R["tasks"].values()))["strict"]["K_validation"]))
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    return m
