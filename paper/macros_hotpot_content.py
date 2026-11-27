"""Macros of HotpotQA under the content reading (scripts/run_hotpot_content.py). Every existing macro keeps its name and value; the new ones carry a Content suffix (or a
new name with a Content counterpart).

Under content scoring the saved choices (strict exact match, discovery draws, saved before validation was scored) are
valued on content-scored validation draws: the existing macro families with a Content suffix,
    aeHq...Content       arm effect (results/arm_effect/'s quantities), discovery means; ...FOneContent under the
                         content reading's token F1
    pceilHq...Content    protocol ceiling with the scoring axis strict, lenient and content and the headline pair under
                         content (the as-run HotpotQA axis was strict, lenient and token F1)
    hqDdir...Content     both directions of the wrong-baseline cost;  hqMech...Content  the mechanism forecast
    hqSplit...Content    split pilot;  hqDec...Content  the saved decomposition rule (hqDecVarDiscPct and
                         hqDecVarThresholdPct have no Content form: they are the saved discovery value and threshold)
    dstrictHq...Content  the fair comparison's strict decision design;  tregret...Hq...Content  its held-out-family and
                         cross-benchmark rules
Exploratory refits of the choices on content-scored discovery draws (after validation was scored): the same names with the
suffix ContentPostHoc (hqDdir, hqSplit, dstrictHq, tregret ...Hq...).
The decomposition and the (1 - w) choice by definition of long, suffix <Thr><Score>, Thr SixtyFour
(more than 64 tokens), Sixteen (more than 16), ExcessEight (at least 8 tokens beyond the same row's forced output),
Score empty for strict or Content: hqDecVarLongPct (validation draws, %), hqDecVarLongAllPct (all distinct
configurations, %), hqDecWMedPct (median w, all distinct, %), hqDecShortGap (median R_S - F_S, pp), hqDecShortPart
(median short part of the shift, pp), hqDecShiftSpearman (w against the shift), hqDecRevFd and hqDecRevCorr of
hqDecRevPairs (pairs reversed between the gain over the requested arm and the forced gain, and the corrected gain; all
distinct), hqDecRmsFd, hqDecRmsCorr (pp), hqDecRegretCorr, RegretFd, RegretShort (requested arm, validation means, pp;
choices refitted on discovery draws), hqDecChoiceHolds, hqDecFreshDiff, hqDecFreshDiffLo, hqDecFreshDiffHi
(fresh configurations, corrected minus forced-gain choice, pp, 95% t interval).
Accuracy and formats: hqArmAcc<Req|Fd|Cot|Fs> (strict), ...Lenient, ...Content (all rows, %); hqGapReqFdMedian and
hqGapCotFdMedian (median per-model gap, pp) and Content; hqContentUp (requested rows wrong under strict, right under
content), hqContentUpPhPct (% of them copying the placeholder), hqContentDown; hqContentUpOverLenient (wrong under
lenient, right under content). fmt<Hq|Gsm|Math|Bbh><Req|Cot|Fd|Fs><Placeholder|LastBox|Bare|Unclosed|Unboxed>Pct
(% of outputs); fmtGsmNumberBox (GSM8K rows ending in a copied <number> box), fmtGsmNumberBoxToCorrect.
"""
import json
import os

import macros_added as ma
from macros_added import _set, num

ARM = {"direct_answer": "Req", "forced_direct": "Fd", "zero_shot_cot": "Cot", "fewshot_forced": "Fs"}
TASKS = {"hotpot": "Hq", "gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
THR = {"cap64": "SixtyFour", "cap16": "Sixteen", "excess8": "ExcessEight"}
SCORE = {"strict": "", "content": "Content"}


def hotpot_content_results(root):
    p = os.path.join(root, "results", "hotpot_content", "hotpot_content.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf-8"))
    for key, rel in (("protocol_ceiling_full", ("results", "hotpot_content", "protocol_ceiling_content.json")),
                     ("thresholds_full", ("results", "hotpot", "decomposition_thresholds_post_hoc.json"))):
        q = os.path.join(root, *rel)
        if os.path.exists(q):
            R[key] = json.load(open(q, encoding="utf-8"))
    for key, name in (("fair_content", "fair_rules_evaluation_content.json"),
                      ("fair_content_post_hoc", "fair_rules_evaluation_content_post_hoc.json")):
        q = os.path.join(root, "results", "hotpot_content", name)
        if os.path.exists(q):
            R[key] = os.path.relpath(q, root).replace("\\", "/")
    R["_root"] = root
    return R


def _suffixed(m, block, keep, suffix, drop=()):
    """Copy the macros of `block` whose names start with one of `keep` (and contain none of `drop`), with suffix."""
    for k, v in block.items():
        if k.startswith(tuple(keep)) and not any(d in k for d in drop):
            _set(m, k + suffix, v)


def _fair(m, root, rel_path, suffix):
    import sys
    sys.path.insert(0, os.path.join(root, "scripts"))
    import added_contrasts as ac
    e = ac.load_tasks()["hotpot"]
    F = ac.fair_evaluation("hotpot", tasks={"hotpot": {**e, "fair": {**e["fair"], "evaluation": rel_path}}}, root=root)
    if F is None:
        return
    R = {"tasks": {"hotpot": {"results": F["strict"]}}}
    _suffixed(m, ma.strict_macros(R), ("dstrictHq",), suffix)
    _suffixed(m, ma.strict_more_macros(R), ("dstrictHq",), suffix)
    T = {"added_family": {f"hotpot/{b}": x["fresh"] for b, x in F["family"].items()},
         "added_task": {f"{s}->hotpot/{b}": x["fresh"] for (s, b), x in F["cross"].items()}}
    _suffixed(m, ma.transfer_added_macros(T), ("tregretFam", "tregretTask"), suffix)


def _thresholds(m, body):
    for d, tt in THR.items():
        for sc, ss in SCORE.items():
            v = body["variants"][d][sc]
            a, va = v["sets"]["all_distinct"], v["sets"]["validation"]
            s = tt + ss
            pct = lambda x: num(None if x is None else 100 * x, 0)  # noqa: E731
            _set(m, "hqDecVarLongPct" + s, pct(va["variance_share_long"]))
            _set(m, "hqDecVarLongAllPct" + s, pct(a["variance_share_long"]))
            _set(m, "hqDecWMedPct" + s, num(100 * v["w_all_distinct_median"], 1))
            _set(m, "hqDecShortGap" + s, num((a["short_gap_pp"] or {}).get("median"), 1))
            ps = sorted(a["per_model"]["part_short"])
            n = len(ps)
            _set(m, "hqDecShortPart" + s, num((ps[n // 2] if n % 2 else (ps[n // 2 - 1] + ps[n // 2]) / 2), 1))
            _set(m, "hqDecShiftSpearman" + s, num(a["closed_fraction"]["corr_shift_with_w_all_models"]["spearman"], 2))
            mech = v["mechanism"]["all_distinct"]
            _set(m, "hqDecRevFd" + s, str(mech["observed_vs_forced"]["reversed"]))
            _set(m, "hqDecRevCorr" + s, str(mech["observed_vs_corrected"]["reversed"]))
            _set(m, "hqDecRevPairs" + s, str(mech["observed_vs_forced"]["pairs"]))
            _set(m, "hqDecRmsFd" + s, num(mech["rms_forced_pp"], 1))
            _set(m, "hqDecRmsCorr" + s, num(mech["rms_corrected_pp"], 1))
            ch = v["choice"]
            r = ch["validation_means"]["regret"]
            _set(m, "hqDecRegretCorr" + s, num(r["corrected"], 1))
            _set(m, "hqDecRegretFd" + s, num(r["forced_gain"], 1))
            _set(m, "hqDecRegretShort" + s, num(r["corrected_short"], 1))
            _set(m, "hqDecChoiceHolds" + s, ma._yn(ch["validation_means"]["holds"]))
            p = (ch["fresh_configurations"] or {}).get("paired_corrected_minus_forced") or {}
            _set(m, "hqDecFreshDiff" + s, num(p.get("estimate"), 2))
            ci = p.get("ci95")
            ok = isinstance(ci, (list, tuple)) and len(ci) == 2
            _set(m, "hqDecFreshDiffLo" + s, num(ci[0], 2) if ok else "--")
            _set(m, "hqDecFreshDiffHi" + s, num(ci[1], 2) if ok else "--")


def hotpot_content_macros(R):
    m = {}
    root = R.get("_root")
    A = R.get("accuracy")
    if A:
        for arm, x in A["by_arm"].items():
            t = "hqArmAcc" + ARM[arm]
            _set(m, t, num(x["strict_em"], 1))
            if "lenient_em" in x:
                _set(m, t + "Lenient", num(x["lenient_em"], 1))
            _set(m, t + "Content", num(x["content_em"], 1))
        for sc, ss in SCORE.items():
            g = A["per_model_gaps"][sc]
            _set(m, "hqGapReqFdMedian" + ss, num(g["requested_minus_forced_median"], 1))
            _set(m, "hqGapCotFdMedian" + ss, num(g["cot_minus_forced_median"], 1))
        c = A["requested_rows_changed"]
        _set(m, "hqContentUp", f"{c['wrong_to_right']:,}".replace(",", "{,}"))
        _set(m, "hqContentUpPhPct", num(None if c["wrong_to_right_placeholder_share"] is None
                                        else 100 * c["wrong_to_right_placeholder_share"], 0))
        _set(m, "hqContentDown", str(c["right_to_wrong"]))
        lu = A.get("requested_lenient_wrong_content_right")
        if lu:
            _set(m, "hqContentUpOverLenient", f"{lu['n']:,}".replace(",", "{,}"))
    F = R.get("formats")
    if F:
        for task, b in F["tasks"].items():
            for arm, x in b["by_arm"].items():
                t = "fmt" + TASKS[task] + ARM[arm]
                for k, kt in (("placeholder_copy", "Placeholder"), ("last_box_placeholder", "LastBox"),
                              ("bare_box", "Bare"), ("unclosed_box", "Unclosed"), ("unboxed", "Unboxed")):
                    _set(m, t + kt + "Pct", num(100 * x[k] / x["n"], 1))
        g = F["tasks"].get("gsm8k", {}).get("placeholder_last_box")
        if g:
            _set(m, "fmtGsmNumberBox", f"{g['rows']:,}".replace(",", "{,}"))
            _set(m, "fmtGsmNumberBoxToCorrect", str(g["would_change_to_correct"]))
    E = R.get("arm_effect")
    if E:
        blk = ma.arm_effect_macros({"forced_cap_tokens": 16, "short_share_pct": 10, "tasks": {"hotpot": E["content"]}})
        _suffixed(m, blk, ("aeHq",), "Content")
        L = E.get("lenient_as_run")
        if L:
            _set(m, "aeHqReversedLenient", str(L["reversals"]["reversed"]))
            _set(m, "aeHqReversedPctLenient", num(100 * L["reversals"]["reversed_share"], 0))
            _set(m, "aeHqMeanReqMinusForcedLenient", num(L["mean_req_minus_forced"], 1))
        for sc, ss in SCORE.items():
            p = E["pooled_all_configurations"][sc]
            _set(m, "aeHqPooledReversed" + ss, str(p["reversed"]))
            _set(m, "aeHqPooledPairsNoTies" + ss, str(p["pairs_without_ties"]))
    P = R.get("protocol_ceiling_full")
    if P:
        for fn in (ma.ceiling_macros, ma.ceiling_more_macros, lambda r: ma.ceiling_disjoint_macros(r, None)):
            _suffixed(m, fn(P), ("pceilHq",), "Content", drop=("ArmContent",))
    S = R.get("saved_choices")
    if S:
        _suffixed(m, ma.hotpot_directions_macros({"decision_directions": S["decision_directions"],
                                                  "mechanism_forecast": S["mechanism_forecast"]}),
                  ("hqDdir", "hqMech"), "Content")
        _suffixed(m, ma.hotpot_d27_macros({"split_pilot": S["split_pilot"], "decomposition": S["decomposition"]}),
                  ("hqSplit", "hqDec"), "Content", drop=("VarDiscPct", "VarThresholdPct"))
    if root and R.get("fair_content"):
        _fair(m, root, R["fair_content"], "Content")
    Q = R.get("refit_post_hoc")
    if Q:
        _suffixed(m, ma.hotpot_directions_macros({"decision_directions": Q["decision_directions"]}), ("hqDdir",),
                  "ContentPostHoc")
        _suffixed(m, ma.hotpot_d27_macros({"split_pilot": Q["split_pilot"]}), ("hqSplit",), "ContentPostHoc")
        if root and R.get("fair_content_post_hoc"):
            _fair(m, root, R["fair_content_post_hoc"], "ContentPostHoc")
    T = R.get("thresholds_full")
    if T:
        _thresholds(m, T)
    return ma._check(m)
