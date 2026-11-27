"""Macros of the three-rater human audit, prefix hrt.

Read from results/human_audit/three_raters.json (scripts/run_human_audit.py --three-raters). The two-rater macros
(prefix hra, paper/macros_added.human_audit_macros and human_audit_rater_macros) are unchanged and come from
human_audit.json. Percentages to 1 dp, kappa and AC1 to 2 dp, "--" when undefined (the JSON says why). Rating times and
changed verdicts are not reported.

Pairs P in AB, AC (the 100 shared answers) and BC (the 242 answers of set B):
  hrtN<P>                 answers compared
  hrtAgree<P>, hrtAgreePct<P>   same code (correct / not correct / can't tell), count and percent
  hrtKappa<P>, hrtAcOne<P>      Cohen's kappa and Gwet's AC1 over the three codes (AC1 with q = 3)
  hrtBinN<P>              answers both raters decided
  hrtKappaBin<P>, hrtAcOneBin<P>  kappa and AC1 on correct / not correct over those (q = 2)
  hrtBothCorrect<P>, hrtFirstOnlyCorrect<P>, hrtSecondOnlyCorrect<P>, hrtBothNot<P>   the 2x2 table (first = the
                          pair's first letter)
Majority on the shared answers:
  hrtMajN, hrtMajUnanimous, hrtMajTwoOfThree, hrtMajNone, hrtMajCorrect, hrtMajNotCorrect, hrtMajCantTell
The A-B disagreements, two of three:
  hrtABDisagree, hrtABResolved, hrtABUnresolved, hrtABSideA, hrtABSideB, hrtABResolvedCorrect, hrtABResolvedNotCorrect
Headline (GSM8K and BIG-Bench Hard outputs that strict scoring marks wrong and content scoring marks correct; majority
on shared answers, single rating elsewhere). Stem hrt (C in place of B on set B, primary) or hrtWithB (B in place of C):
  <stem>DisagreeN         decided outputs; <stem>DisagreeExcluded no majority or can't tell
  <stem>DisagreeContentAgree, <stem>DisagreeContentAgreePct, ...Lo, ...Hi   read as correct (Wilson 95%)
  <stem>ContentAgreePct (Lo, Hi), <stem>StrictAgreePct, <stem>ContentN   over every GSM8K and BIG-Bench Hard answer
Rater C alone (as hraRater<A|B>...): hrtRaterCAnswers, CantTell, ContentAgree, ContentN, ContentAgreePct (Lo, Hi),
  StrictAgreePct, StrictAgreeAllPct, CtrlWrongRead, CtrlWrongN, CtrlRightRead, CtrlRightN, and ...All.
Counts: hrtAnswersC (answers C rated), hrtSharedThree (answers all three rated).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import macros_added as _ma  # noqa: E402

num, _set, _check, _pct = _ma.num, _ma._set, _ma._check, _ma._pct


def third_rater_results(root):
    """results/human_audit/three_raters.json, or None while it is absent."""
    p = os.path.join(root, "results", "human_audit", "three_raters.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _lohi(m, stem, ci):
    _set(m, stem + "Lo", _pct(ci[0]) if ci else "--")
    _set(m, stem + "Hi", _pct(ci[1]) if ci else "--")


def third_rater_macros(R):
    m = {}
    c = R["counts"]
    _set(m, "hrtAnswersC", str(c["rated_by_C"]))
    _set(m, "hrtSharedThree", str(c["shared"]))
    for p, b in R["pairwise"].items():
        co, bi, t = b["codes"], b["decided_both"], b["table_2x2"]
        _set(m, "hrtN" + p, str(b["n"]))
        _set(m, "hrtAgree" + p, str(co["agree"]))
        _set(m, "hrtAgreePct" + p, _pct(co["share"]))
        _set(m, "hrtKappa" + p, num(co["kappa"], 2))
        _set(m, "hrtAcOne" + p, num(co["ac1"], 2))
        _set(m, "hrtBinN" + p, str(bi["n"]))
        _set(m, "hrtKappaBin" + p, num(bi["kappa"], 2))
        _set(m, "hrtAcOneBin" + p, num(bi["ac1"], 2))
        for k, name in (("both_correct", "BothCorrect"), ("first_only_correct", "FirstOnlyCorrect"),
                        ("second_only_correct", "SecondOnlyCorrect"), ("both_not_correct", "BothNot")):
            _set(m, f"hrt{name}{p}", str(t[k]))
    mj = R["majority"]
    for k, name in (("n", "N"), ("unanimous", "Unanimous"), ("two_of_three", "TwoOfThree"), ("no_majority", "None"),
                    ("majority_correct", "Correct"), ("majority_not_correct", "NotCorrect"),
                    ("majority_cant_tell", "CantTell")):
        _set(m, "hrtMaj" + name, str(mj[k]))
    rs = R["ab_resolution"]
    for k, name in (("disagreements", "Disagree"), ("resolved", "Resolved"), ("unresolved", "Unresolved"),
                    ("sides_with_A", "SideA"), ("sides_with_B", "SideB"), ("resolved_correct", "ResolvedCorrect"),
                    ("resolved_not_correct", "ResolvedNotCorrect")):
        _set(m, "hrtAB" + name, str(rs[k]))
    for variant, stem in (("C_for_B", "hrt"), ("B_for_C", "hrtWithB")):
        h = R["headline"][variant]["strict_wrong_content_correct"]
        a = R["headline"][variant]["strict_and_content"]
        _set(m, stem + "DisagreeN", str(h["decided"]))
        _set(m, stem + "DisagreeExcluded", str(h["no_majority_excluded"] + h["cant_tell_excluded"]))
        _set(m, stem + "DisagreeContentAgree", str(h["read_correct"]))
        _set(m, stem + "DisagreeContentAgreePct", _pct(h["share_read_correct"]))
        _lohi(m, stem + "DisagreeContentAgreePct", h.get("ci95_wilson"))
        _set(m, stem + "ContentN", str(a["content"]["n"]))
        _set(m, stem + "ContentAgreePct", _pct(a["content"]["share"]))
        _lohi(m, stem + "ContentAgreePct", a["content"].get("ci95_wilson"))
        _set(m, stem + "StrictAgreePct", _pct(a["strict"]["share"]))
    for k, v in _ma.human_audit_rater_macros({"per_rater": {"C": R["per_rater"]["C"]}}).items():
        _set(m, "hrt" + k[len("hra"):], v)
    return _check(m)
