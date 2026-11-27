"""Macros of the default- and firm-clause run on the arm-wording design, prefix awx.

awx_results(root) returns None until scripts/run_arm_wording_extra.py has written results/arm_wording_extra/
arm_wording_extra.json and forecast_evaluation.json; awx_macros(R) turns them into {name: value}. Task tags Gsm and
Math; clause tags Cur (dir0), Def (default), Firm. Values in pp, shares in [0, 1], verdicts as yes/no (n/a when not
testable).
"""
import json
import math
import os

TASK = {"gsm8k": "Gsm", "math": "Math"}
CLAUSE = {"dir0": "Cur", "dirdef": "Def", "dirfirm": "Firm"}


def _num(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def _yn(v):
    return "n/a" if v is None else ("yes" if v else "no")


def awx_results(root):
    d = os.path.join(root, "results", "arm_wording_extra")
    a, e = os.path.join(d, "arm_wording_extra.json"), os.path.join(d, "forecast_evaluation.json")
    if not (os.path.isfile(a) and os.path.isfile(e)):
        return None
    R = json.load(open(a, encoding="utf-8"))
    R["_evaluation"] = json.load(open(e, encoding="utf-8"))
    return R


def awx_macros(R):
    m = {}
    E = R["_evaluation"]
    clauses = list(R["clauses"])
    for task, T in TASK.items():
        r, x = R["per_task"][task], E["tasks"][task]
        s = r["strict"]
        acc = r["accuracy_mean_pp"]
        for d, D in CLAUSE.items():
            k = clauses.index(d)
            g = s["direct_gap"][k]
            m[f"awx{T}{D}Req"] = _num(acc["direct_answer"][d], 1)
            m[f"awx{T}{D}Fd"] = _num(acc["forced_direct"][d], 1)
            m[f"awx{T}{D}Gap"] = _num(g["mean_gap_pp"], 1)
            m[f"awx{T}{D}W"] = _num(g["long_requested_share"], 2)
            m[f"awx{T}{D}Switch"] = _num(s["reversals"]["arm_switch_by_clause"][k]["one_minus_C"], 3)
            m[f"awx{T}{D}SwitchPct"] = _num(100 * s["reversals"]["arm_switch_by_clause"][k]["one_minus_C"], 1)
        rv = s["reversals"]
        m[f"awx{T}ClauseRev"] = _num(rv["clause_switch_requested"]["one_minus_C"], 3)
        m[f"awx{T}ConfigRev"] = _num(rv["configuration_switch_requested"]["one_minus_C"], 3)
        m[f"awx{T}ArmSwitch"] = _num(rv["arm_switch"]["one_minus_C"], 3)
        sw = [v["one_minus_C"] for v in rv["arm_switch_by_clause"]]
        m[f"awx{T}SwitchMin"], m[f"awx{T}SwitchMax"] = _num(min(sw), 3), _num(max(sw), 3)
        m[f"awx{T}RBoth"] = _num(s["R"]["req"]["both_drawn"]["R"], 3)
        a1 = x["A1_forced_meets_requested"]
        m[f"awx{T}NearZeroN"] = str(a1["n"])
        m[f"awx{T}NearZeroGap"] = _num(a1["mean_gap_pp"], 2)
        m[f"awx{T}NearZeroWithin"] = str(a1["within_tol"])
        m[f"awx{T}AOneA"], m[f"awx{T}AOneB"] = _yn(a1["holds_a"]), _yn(a1["holds_b"])
        for key, stem in (("A2_default_more_working", "ATwo"), ("A3_firm_less_working", "AThree")):
            m[f"awx{T}{stem}Median"] = _num(x[key]["median_diff"], 3)
            m[f"awx{T}{stem}Count"] = str(x[key]["models_in_direction"])
            m[f"awx{T}{stem}A"], m[f"awx{T}{stem}B"] = _yn(x[key]["holds_a"]), _yn(x[key]["holds_b"])
        m[f"awx{T}AFourR"] = _num(x["A4_pilot_predicts_w"]["pearson"], 3)
        m[f"awx{T}AFour"] = _yn(x["A4_pilot_predicts_w"]["holds"])
        a5 = x["A5_extrapolates_to_forced"]
        m[f"awx{T}AFiveIntercept"] = _num(a5["median_intercept_pp"], 2)
        m[f"awx{T}AFiveN"] = str(a5["models"])
        m[f"awx{T}AFive"] = _yn(a5["holds"])
        h = E["holds"][task]
        for k in ("S1", "S2", "S4a", "S4b", "S5a", "S5b"):
            m[f"awx{T}{k.replace('1', 'One').replace('2', 'Two').replace('4', 'Four').replace('5', 'Five')}"] = _yn(h[k])
        m[f"awx{T}SThreeCount"] = str(h["S3_count"])
    m["awxModels"] = str(len(R["models"]))
    m["awxFirmText"] = R["clause_texts"]["dirfirm"]
    return m
