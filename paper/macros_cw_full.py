"""Macros of Qwen4-Max on all 398 numeric MATH-500 items under MATH-500's first 8 discovery draws, the three arms, prefix cwFull. Read from
results/frontier_panel/closed_w_full_math.json (scripts/closed_w_full.py report) and, for the pilot's call count,
closed_w_full_math_pilot.json. Every value is read from these files; nothing is typed here.

cwfull_results(root) reads the files, or returns None until closed_w_full_math.json exists; cwfull_macros(R) returns
{name: value}, after checking every name (letters only, prefix cwFull).

Gains in percentage points, one decimal, $-$ for a minus; Lo and Hi are the 95% question-bootstrap interval (2,000
draws, items resampled together across templates and arms, templates fixed, percentile). Accuracy and w are means over
the 8 templates of per-template means. Strict scoring; the same names with Content (before Lo or Hi) are content
scoring.
  cwFullReqGainMath{,Lo,Hi}         requested gain, CoT - requested accuracy
  cwFullFdGainMath{,Lo,Hi}          forced gain, CoT - forced accuracy
  cwFullCorrMath{,Lo,Hi}            corrected gain, (1 - w)(CoT - forced), w and the gain each averaged over templates
  cwFullGainDiffMath{,Lo,Hi}        requested gain - forced gain (= forced - requested accuracy), paired by item
  cwFullWMathPct{,Lo,Hi}            w in %, no decimal (as \\cwWMathPct): share of requested replies longer than 16
                                    tokens by the API's count
  cwFullReqAccMath, cwFullFdAccMath, cwFullCotAccMath   accuracies in %, one decimal
  cwFullPrevGainDiffMath{,Lo,Hi}    the 100-item run's paired difference on its published basis (100 items, the
                                    discovery draws: 24 draws over 18 templates), computed from its published gains;
                                    the interval from its raw replies with its own bootstrap seed. Its other values
                                    stay the cw macros (\\cwReqGainMath, \\cwFdGainMath, \\cwCorrMath, \\cwWMathPct)
  cwFullItems, cwFullTemplates      398 and 8
  cwFullCalls                       calls in the design (398 x 8 x 3 arms)
  cwFullAnswered                    calls with a reply (refusals included), provider blocks and missing calls not
  cwFullBlocked, cwFullRefusals, cwFullMissing, cwFullTruncated   provider blocks (Alibaba's content inspection),
                                    refusals, calls never answered, replies cut by the cap (finish reason length)
  cwFullPilotCalls                  the pilot's calls (32 cells x 3 arms x 2 sends and 2 thinking-on controls)
  cwFullCostUsd                     billed by OpenRouter (usage.cost), pilot and run, two decimals
"""
from __future__ import annotations

import json
import os
import re

PREFIX = "cwFull"
PATH = os.path.join("results", "frontier_panel", "closed_w_full_math.json")
PILOT = os.path.join("results", "frontier_panel", "closed_w_full_math_pilot.json")
QUANT = (("cot_minus_requested", "ReqGain"), ("cot_minus_forced", "FdGain"), ("corrected_gain", "Corr"),
         ("gain_diff", "GainDiff"))
ACC = (("requested", "ReqAcc"), ("forced", "FdAcc"), ("cot", "CotAcc"))


def _load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cwfull_results(root: str):
    rep = _load(os.path.join(root, PATH))
    if rep is None:
        return None
    return {"rep": rep, "pilot": _load(os.path.join(root, PILOT))}


def f1(x, nd=1):
    """As make_numbers.f: a hyphen prints $-$ (a value that rounds to zero keeps its sign)."""
    return f"{x:.{nd}f}".replace("-", "$-$")


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def _num(x, what):
    if x is None:
        raise ValueError(f"{what}: no finite value in the results file")
    return float(x)


def _one(rows, **kw):
    hit = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} rows for {kw}")
    return hit[0]


def cwfull_macros(R) -> dict:
    if not R:
        return {}
    rep, out = R["rep"], {}

    def put(k, v):
        if k in out:
            raise ValueError(f"macro {k} set twice")
        out[k] = v

    for score, S in (("strict", ""), ("content", "Content")):
        g = _one(rep["gains"], score=score)
        for q, N in QUANT:
            put(f"{PREFIX}{N}Math{S}", f1(100 * _num(g[q], f"{score} {q}")))
            lo, hi = g["ci"][q]
            put(f"{PREFIX}{N}Math{S}Lo", f1(100 * _num(lo, f"{score} {q} low")))
            put(f"{PREFIX}{N}Math{S}Hi", f1(100 * _num(hi, f"{score} {q} high")))
        put(f"{PREFIX}WMathPct{S}", f"{100 * _num(g['w'], score + ' w'):.0f}")
        lo, hi = g["ci"]["w"]
        put(f"{PREFIX}WMathPct{S}Lo", f"{100 * _num(lo, score + ' w low'):.0f}")
        put(f"{PREFIX}WMathPct{S}Hi", f"{100 * _num(hi, score + ' w high'):.0f}")
        for q, N in ACC:
            put(f"{PREFIX}{N}Math{S}", f1(100 * _num(g[q], f"{score} {q}")))
        prev = [x for x in rep.get("comparison") or [] if x["score"] == score and x["label"].startswith("(d)")]
        if len(prev) == 1:
            p = prev[0]
            put(f"{PREFIX}PrevGainDiffMath{S}", f1(100 * _num(p["gain_diff"], f"{score} 100-item gain_diff")))
            lo, hi = p["ci"]["gain_diff"]
            put(f"{PREFIX}PrevGainDiffMath{S}Lo", f1(100 * _num(lo, f"{score} 100-item gain_diff low")))
            put(f"{PREFIX}PrevGainDiffMath{S}Hi", f1(100 * _num(hi, f"{score} 100-item gain_diff high")))
    d = rep["design"]
    put(f"{PREFIX}Items", str(d["n_items"]))
    put(f"{PREFIX}Templates", str(d["n_templates"]))
    put(f"{PREFIX}Calls", comma(d["n_requests"]))
    t = rep["calls_total"]
    if t["expected"] != d["n_requests"]:
        raise ValueError(f"calls expected {t['expected']} differ from the design's {d['n_requests']}")
    put(f"{PREFIX}Answered", comma(t["answered"]))
    put(f"{PREFIX}Blocked", comma(t["blocked"]))
    put(f"{PREFIX}Refusals", comma(t["refusals"]))
    put(f"{PREFIX}Missing", comma(t["missing"]))
    put(f"{PREFIX}Truncated", comma(t["truncated"]))
    if R.get("pilot") is not None:
        put(f"{PREFIX}PilotCalls", comma(R["pilot"]["n_calls"]))
    put(f"{PREFIX}CostUsd", f"{float(rep['spend']['total_billed_usd']):.2f}")
    for k in out:
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for k, v in sorted(cwfull_macros(cwfull_results(root)).items()):
        print(f"\\newcommand{{\\{k}}}{{{v}}}")
