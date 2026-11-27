"""paper/macros_added.py: the macros of the added results, on small fabricated results
files with known values. Names are letters only, values keep their sign, undefined values print "--"."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
# paper/ is not in the repository, so a clean checkout skips this file
ma = pytest.importorskip("macros_added")


def test_number_format_and_undefined_values():
    assert ma.num(-1.234, 2) == "$-$1.23" and ma.num(0.0004, 3) == "0.000" and ma.num(-0.0004, 3) == "0.000"
    assert ma.num(None) == "--" and ma.num(float("nan")) == "--" and ma.num(float("inf")) == "--"
    assert ma.num("x") == "--"


@pytest.mark.parametrize("key, tag", [("gsm8k_req", "GsmReq"), ("fewshot_math", "FsMath"), ("dove_mmlu", "DoveMmlu"),
                                      ("gsm8k->bbh/forced", "GsmToBbhForced"), ("math/requested", "MathReq"),
                                      ("exemplar", "Exemplar")])
def test_tags_are_letters(key, tag):
    assert ma.tag_of(key) == tag


def _slope(s, lo, hi):
    return {"slope": s, "model_bootstrap": {"ci95": [lo, hi]}, "family_bootstrap": {"ci95": [lo - 0.1, hi + 0.1]}}


def test_fewshot_macros():
    T = {"M": 29, "n_items": 1319, "means": {"zero_shot_forced": 25.04, "five_shot_forced": 28.06, "gain": 3.024},
         "step2": {"V_delta": {"estimate": 6.661, "ci95": [4.19, 12.19]}},
         "map": {"point": {"benefit": {"R": 0.9439}}, "validation_one_draw": {"benefit": {"C": 0.8741}}},
         "capability_slope": _slope(0.06, -0.02, 0.13)}
    steps = {"gsm8k": {"step3": {"budget_curve": [
        {"K": 1, "f": "own_gain", "g": "averaged_ols", "estimate": -5.1304, "ci95": [-5.4, -4.87]},
        {"K": 2, "f": "own_gain", "g": "averaged_ols", "estimate": -9.0, "ci95": [-9.5, -8.5]}]}}}
    m = ma.fewshot_macros([{"tasks": {"gsm8k": T}}, {"dry_run": True, "tasks": {"bbh": T}}], steps)
    assert m["fewshotGsmItems"] == "1{,}319" and m["fewshotGsmMeanGainFine"] == "3.02"
    assert m["fewshotGsmVdeltaFine"] == "6.66" and m["fewshotGsmVdeltaFineHi"] == "12.19"
    assert m["fewshotGsmMapRbenFine"] == "0.944" and m["fewshotGsmMapCval"] == "0.874"
    assert m["fewshotGsmSlope"] == "0.060" and m["fewshotGsmSlopeModelLo"] == "$-$0.02"
    assert m["fewshotGsmSlopeIntervalsHoldZero"] == "yes"
    assert m["fewshotGsmBudgetOneOwnOls"] == "$-$5.13" and m["fewshotGsmBudgetOneOwnOlsLo"] == "$-$5.40"
    assert not any(k.startswith("fewshotBbh") for k in m)          # a dry run adds nothing
    assert "fewshotGsmM" not in m                                    # d2_macros sets it from the Steps 1-3 file
    T2 = dict(T, capability_slope=_slope(-0.3, -0.5, -0.1))
    assert ma.fewshot_macros([{"tasks": {"math": T2}}])["fewshotMathSlopeIntervalsHoldZero"] == "no"


def test_constrained_macros():
    rows = [{"requested": 40, "forced": 30, "constrained": 50, "cot": 45, "forced_closed_and_readable_pct": 60.0,
             "constrained_closed_and_readable_pct": 100.0},
            {"requested": 50, "forced": 40, "constrained": 60, "cot": 55, "forced_closed_and_readable_pct": None,
             "constrained_closed_and_readable_pct": 99.0}]
    R = {"panel": ["a", "b"], "per_model": rows,
         "panels": {"forced": {"readable": ["a"]}, "constrained": {"readable": ["a", "b"]}},
         "slopes": {"cot_minus_requested": {"all": {"M": 2, **_slope(0.5, 0.1, 0.9)}},
                    "cot_minus_forced": {"all": {"M": 2, **_slope(0.4, 0.0, 0.8)},
                                         "readable": {"M": 1, "note": "fewer than three models"}},
                    "cot_minus_constrained": {"all": {"M": 2, **_slope(1.31, 0.81, 1.79)}}}}
    steps = {"step2": {"mean_gain": {"estimate": -5.678}},
             "sensitivities": {"lenient_extraction": {"steps_1_2": {"mean_gain": 11.894, "C_benefit": 0.9638,
                                                                     "V_delta": 153.93, "H": -8.78}}}}
    m = ma.constrained_macros(R, steps)
    assert m["gcM"] == "2" and m["gcMeanConstrained"] == "55.0" and m["gcMeanCot"] == "50.0"
    assert m["gcReadableForced"] == "1" and m["gcReadablePctMinForced"] == "60.0"
    assert m["gcReadablePctMinConstrained"] == "99.0"
    assert m["gcSlopeConstrainedAll"] == "1.31" and m["gcSlopeConstrainedAllModelLo"] == "0.81"
    assert m["gcSlopeForcedReadable"] == "--" and m["gcSlopeForcedReadableM"] == "1"
    assert m["gcBbhMeanGainFine"] == "$-$5.68" and m["gcBbhLenientMeanGain"] == "11.89"
    assert ma.constrained_macros({"dry_run": True}) == {}


def _share(pct, lo, hi, k=1, n=10):
    return {"pct": pct, "lo": lo, "hi": hi, "k": k, "n": n}


def test_vendor_macros_and_ranges():
    def cell(cv, cg, d):
        return {"vendor": {"cut": _share(cv, 0, 1), "strict": _share(90.0, 0, 1), "mean_output_tokens": 8086.2},
                "greedy": {"cut": _share(cg, 0, 1), "strict": _share(50.0, 0, 1), "mean_output_tokens": 15578.0},
                "diff_vendor_minus_greedy": {"mean": d, "lo": d - 4, "hi": d + 4},
                "diff_vendor_minus_cot": {"mean": -d / 10, "lo": -1, "hi": 1}, "acc_cot": _share(95.0, 0, 1)}
    R = {"cells": {"gsm8k/qwen3.5-35b-a3b": cell(4.33, 44.0, 38.0), "math/qwen3.6-27b": cell(5.3, 30.5, 25.0)}}
    m = ma.vendor_macros(R)
    assert m["tvGsmQwenFiveMoeCutVendor"] == "4.3" and m["tvMathQwenSixDenseCutGreedy"] == "30.5"
    assert m["tvGsmQwenFiveMoeTokensVendor"] == "8{,}086" and m["tvGsmQwenFiveMoeAccCot"] == "95.0"
    assert m["tvCutVendorMin"] == "4.3" and m["tvCutVendorMax"] == "5.3"
    assert m["tvVendorMinusGreedyMin"] == "25.0" and m["tvGsmVendorMinusCotMin"] == "$-$3.8"
    assert all(k.isalpha() for k in m)


def _rule(p, c):
    return {"regret_plugin": p, "regret_crossfit": c}


def test_decision_macros():
    ev = {"n_configurations": 13, "best_third_value": 5.18, "panel_mean_gain": 2.48,
          "rules": {"capability_1": _rule(1.98, 1.97), "own_1": _rule(0.37, 0.36), "random": _rule(2.70, 2.69)},
          "paired": [{"a": "capability_1", "b": "own_1", "estimate": 1.6154, "ci95": [1.46, 1.77]}]}
    R = {"settings": {"fewshot_math": {"result": {"M": 29, "n_chosen": 10, "evaluation": {"independent": ev}}}}}
    m = ma.decision_macros(R)
    assert m["decFsMathCapOnePlug"] == "1.98" and m["decFsMathOwnOneCross"] == "0.36"
    assert m["decFsMathRandPlug"] == "2.70" and "decFsMathOwnFourPlug" not in m
    assert m["decFsMathCapMinusOwn"] == "1.62" and m["decFsMathCapMinusOwnHi"] == "1.77"
    assert m["decFsMathChosen"] == "10" and m["decFsMathConfigs"] == "13"


def test_strict_macros():
    fresh = {"n_configurations": 17, "rules": {"linear": {"regret_plugin": 2.14},
                                               "quadratic_index": {"regret_plugin": 0.1753},
                                               "own_gain": {"regret_plugin": 1.099}},
             "paired": {"own_gain_minus_quadratic_index": {"estimate": 0.924, "ci95": [0.78, 1.06]}}}
    R = {"tasks": {"bbh": {"results": {"disjoint": {"forced": {"fresh": fresh, "all_distinct": fresh}}}}},
         "ref_check": {"gsm8k": {"reproduced": True}, "bbh": {"reproduced": False}}}
    m = ma.strict_macros(R)
    assert m["dstrictBbhDisjointForcedQuad"] == "0.18" and m["dstrictBbhDisjointForcedOwn"] == "1.10"
    assert m["dstrictBbhDisjointForcedOwnMinusQuad"] == "0.92" and m["dstrictBbhDisjointForcedOwnMinusQuadLo"] == "0.78"
    assert "dstrictBbhDisjointForcedRand" not in m
    assert m["dstrictRefReproduced"] == "1" and m["dstrictRefTasks"] == "2"


def test_transfer_macros():
    fam = {"gsm8k/requested": {"rules": {"transport_quadratic_index": _rule(4.56, 4.56), "random": _rule(23.9, 23.9)}},
           "bbh/forced": {"rules": {"transport_quadratic_index": _rule(3.96, 3.96), "random": _rule(8.6, 8.6)}}}
    task = {"gsm8k->bbh/requested": {"rules": {"source_linear": _rule(39.74, 39.7), "random": _rule(13.35, 13.3)}},
            "bbh->math/forced": {"rules": {"source_linear": _rule(1.09, 1.1), "random": _rule(25.16, 25.1)}}}
    m = ma.transfer_macros({"family": fam, "task": task})
    assert m["tregretFamGsmReqQuad"] == "4.56" and m["tregretFamQuadMin"] == "3.96" and m["tregretFamQuadMax"] == "4.56"
    assert m["tregretTaskGsmToBbhReqSrcLin"] == "39.74" and m["tregretTaskSrcLinMin"] == "1.09"
    assert m["tregretTaskLinWorseThanRandom"] == "1" and m["tregretFamCases"] == "2"


def test_ceiling_macros():
    T = {"D": 29, "headline": {"within_requested": {"1": 0.9564, "14": 0.9865, "2": 0.97},
                               "between_same_draws": {"1": 0.8008, "12": 0.797, "24": 0.794},
                               "plugin": 0.7906, "plugin_bootstrap": {"ci95": [0.788, 0.805]}, "posterior": 0.7968},
         "axes": {"arm": {"same_draws": {"1": 0.809}, "limit_plugin": 0.8087, "limit_posterior": 0.8096},
                  "random": {"same_draws": {"1": 0.765}, "limit_plugin": 0.7664, "limit_posterior": 0.7669}}}
    m = ma.ceiling_macros({"tasks": {"gsm8k": T}})
    assert m["pceilGsmWithinMax"] == "0.987" and m["pceilGsmWithinMaxK"] == "14" and m["pceilGsmWithinOne"] == "0.956"
    assert m["pceilGsmBetweenTwentyFour"] == "0.794" and m["pceilGsmLimitPlugLo"] == "0.788"
    assert m["pceilGsmRandomLimitPost"] == "0.767" and "pceilGsmScaleOne" not in m


def test_published_macros():
    c = {"changes_direction_from_points": {"logits": True, "error_reduction": False},
         "pair_flips": {"logits": {"pairs": 105, "reversed": 6}, "error_reduction": {"pairs": 105, "reversed": 12}}}
    k = {"changes_direction_from_points": {"logits": False, "error_reduction": False},
         "pair_flips": {"logits": {"pairs": 1, "reversed": 0}, "error_reduction": {"pairs": 1, "reversed": 0}}}
    R = {"n_records": 45, "claims": {"cheng2025revisiting": c, "khan2025prompting": k}, "untraced": [1] * 6,
         "n_no_numbers": 36}
    m = ma.published_macros(R)
    assert m["pubreRecords"] == "45" and m["pubreBothArms"] == "2" and m["pubreUntraced"] == "6"
    assert m["pubreChangeDirection"] == "1" and m["pubreChengFlipsLogit"] == "6" and m["pubreChengPairsErrRed"] == "105"
    assert m["pubreChengChangesLogit"] == "yes" and m["pubreKhanChangesErrRed"] == "no"
