"""Macros of the correctness checks (scripts/run_output_checks.py,
results/output_checks/output_checks.json), prefix wck. Exploratory checks on GSM8K-29, BIG-Bench Hard-28 and MATH-500.

Constrained BIG-Bench Hard (chain-of-thought gain over the requested arm against gain over the constrained
forced arm, discovery means): wckBbhConRev of wckBbhConPairs (strict), wckBbhConRevPct; wckBbhConRevContent,
wckBbhConRevContentPct (content-scored chain of thought and requested, constrained strict: it has no content scores);
wckBbhConBelow (models whose chain of thought scores below the constrained baseline) of wckBbhConM,
wckBbhConMeanGain (pp); the same with a Content suffix.

Item 5, Qwen3.6-35B-A3B (tag QwenSixMoe): wckQwenSixMoeW<T> (w, discovery draws, 2 dp); on MATH-500
wckQwenSixMoeMath<Req|Fd|Cot> (discovery-mean accuracy, %, strict) and the same with Content; wckQwenSixMoeMathPooled
<Req|Fd>Content (all distinct configurations pooled); wckQwenSixMoeRank<T> and RankContent<T> (rank as a gainer over
the requested arm, 1 = largest gain). Without the model (prefix wckNoQ<T>): Rev, RevPairs, RevPct (arm-switch
reversals, strict), RevContent, RevContentPct; RevCorr and RevObs (corrected and observed reversals, strict) and
Content; VarLongPct and VarLongPctContent (long share of the between-model variance, %); ReqToFd, FdToReq (one-draw
pilots, own gain), PilotReqToFd, PilotFdToReq (100-item pilots), with a Content suffix; across tasks
wckNoQOwn<Dir>Min/Max and wckNoQPilot<Dir>Min/Max.

Item 6, equal traffic (cost per chosen model x n / M): wck<T><Dir>Panel, wck<T>Pilot<Dir>Panel (+Lo, Hi, the t
interval scaled the same way), Content suffix after Panel; wckOwn<Dir>PanelMin/Max, wckPilot<Dir>PanelMin/Max.

Item 7, tokens up to the first complete box: wck<T>LongPct (whole reply, %), LongFirstPct, LongNoPrefixPct;
wck<T>NoBoxLongPct (long outputs with no complete box, %); wck<T>RevCorrFirst, RevCorrNoPrefix (corrected
reversals, strict) and Content; wck<T>VarLongPctFirst, VarLongPctNoPrefix (long share, %) and Content.

Item 9 adds no macro: its table lists existing ddir macros.
"""
import macros_added as ma
from macros_added import _set, num

DIRS = (("requested_to_forced", "ReqToFd"), ("forced_to_requested", "FdToReq"))
ARM = {"requested": "Req", "forced": "Fd", "cot": "Cot"}


def output_checks_results(root):
    import json
    import os
    p = os.path.join(root, "results", "output_checks", "output_checks.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _pct(x, nd=1):
    return num(None if x is None else 100 * float(x), nd)


def output_checks_macros(R):
    m = {}
    x = R.get("item4")
    if x:
        for blk, suf in (("strict", ""), ("content_cot_and_requested", "Content")):
            b = x[blk]
            _set(m, "wckBbhConRev" + suf, str(b["reversed"]))
            _set(m, "wckBbhConRev" + suf + "Pct", _pct(b["share"], 0))
            _set(m, "wckBbhConBelow" + suf, str(b["models_cot_below_constrained"]))
            _set(m, "wckBbhConMeanGain" + suf, num(b["mean_gain_pp"], 1))
        _set(m, "wckBbhConPairs", str(x["strict"]["pairs"]))
        _set(m, "wckBbhConM", str(x["M"]))
    x = R.get("item5")
    if x:
        q = "wckQwenSixMoe"
        for t, b in x["tasks"].items():
            T = ma._task(t)
            _set(m, q + "W" + T, num(b["w_discovery"], 2))
            _set(m, q + "Rank" + T, str(b["rank_gain_vs_requested"]["strict"]))
            _set(m, q + "RankContent" + T, str(b["rank_gain_vs_requested"]["content"]))
            if t == "math":
                for sc, suf in (("strict", ""), ("content", "Content")):
                    for a, tag in ARM.items():
                        _set(m, q + "Math" + tag + suf, num(b["accuracy_discovery"][sc][a], 1))
            s = "wckNoQ" + T
            for sc, suf in (("strict", ""), ("content", "Content")):
                r = b["arm_switch_reversals"][sc]["without"]
                _set(m, s + "Rev" + suf, str(r["reversed"]))
                _set(m, s + "Rev" + suf + "Pct", _pct(r["share"], 0))
                mc = b["mechanism"][sc]["without"]
                _set(m, s + "RevCorr" + suf, str(mc["corrected"]))
                _set(m, s + "RevObs" + suf, str(mc["observed"]))
                v = (b.get("variance_share_long_without") or {}).get(sc)
                _set(m, s + "VarLongPct" + suf, _pct(v, 0))
                for design, pre in (("one_draw", ""), ("pilot100", "Pilot")):
                    d = b["decision_costs"][design].get(sc)
                    for k, tag in DIRS:
                        _set(m, s + pre + tag + suf, num(d["without"][k], 2) if d else "--")
            _set(m, s + "RevPairs", str(b["arm_switch_reversals"]["strict"]["without"]["pairs"]))
        pooled = x.get("math_all_distinct_pooled")
        if pooled:
            for a in ("requested", "forced"):
                _set(m, q + "MathPooled" + ARM[a] + "Content", num(pooled["content"][a], 1))
        for key, v in x["min_max_without"].items():
            design, k = key.split("/")
            tag = dict(DIRS)[k]
            stem = "wckNoQ" + ("Own" if design == "one_draw" else "Pilot") + tag
            _set(m, stem + "Min", num(v[0], 2))
            _set(m, stem + "Max", num(v[1], 2))
    x = R.get("item6")
    if x:
        for r in x["rows"]:
            stem = "wck" + ma._task(r["task"]) + ("Pilot" if r["design"] == "pilot100" else "") + \
                   dict(DIRS)[r["direction"]] + "Panel" + ("Content" if r["score"] == "content" else "")
            _set(m, stem, num(r["panel_equivalent"], 2))
            ma._lohi(m, stem, r.get("panel_ci95"), 2)
        for key, v in x["min_max"].items():
            design, k = key.split("/")
            stem = "wck" + ("Own" if design == "one_draw" else "Pilot") + dict(DIRS)[k] + "Panel"
            _set(m, stem + "Min", num(v[0], 2))
            _set(m, stem + "Max", num(v[1], 2))
    x = R.get("item7")
    if x:
        for t, b in x["tasks"].items():
            s = "wck" + ma._task(t)
            ls = b["long_share_pct"]
            _set(m, s + "LongPct", num(ls["whole_reply"], 1))
            _set(m, s + "LongFirstPct", num(ls["first_answer"], 1))
            _set(m, s + "LongNoPrefixPct", num(ls["first_answer_no_prefix"], 1))
            _set(m, s + "NoBoxLongPct", num(b["no_complete_box_among_long_pct"], 1))
            for name, tag in (("first_answer", "First"), ("first_answer_no_prefix", "NoPrefix")):
                for sc, suf in (("strict", ""), ("content", "Content")):
                    _set(m, s + "RevCorr" + tag + suf, str(b[name]["mechanism"][sc]["corrected"]))
                    _set(m, s + "VarLongPct" + tag + suf, _pct(b[name]["variance_share_long"][sc], 0))
    return ma._check(m)
