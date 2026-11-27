"""paper/macros_added.py: small fabricated results files with
known values, edge cases reported as they are, HotpotQA and the arm wording taken once scored and skipped
until then. The figure tables are tested in test_write_curve_tables.py."""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
# paper/ is not in the repository, so a clean checkout skips this file
ma = pytest.importorskip("macros_added")


def letters(m):
    return all(k.isalpha() for k in m)


def boot(lo, hi, below=0.0, undefined=0):
    return {"ci95": [lo, hi], "share_below_zero": below, "n_undefined": undefined}


def slope_block(s=0.5, fam=("A", "B")):
    return {"M": 6, "slope": s, "draw_set": {"kind": "bounded", "a": s - 0.01, "b": s + 0.01},
            "model_bootstrap": boot(s - 0.4, s + 0.4, 0.1), "family_bootstrap": boot(s - 0.6, s + 0.6, 0.2),
            "leave_one_model_out": {"min": s - 0.1, "max": s + 0.2},
            "leave_one_family_out": {f: s + 0.05 * k for k, f in enumerate(fam)}, "families": {f: 3 for f in fam}}


# ---------------------------------------------------------------- further values, existing prefixes


def test_strict_more_macros():
    rule = {"regret_plugin": 2.14, "regret_crossfit": 2.149, "plugin_interval": {"ci95": [1.894, 2.389]}}
    R = {"tasks": {"bbh": {"results": {"disjoint": {"forced": {"fresh": {
        "best_third_value": 25.98, "panel_mean_gain": 7.1,
        "rules": {"linear": rule, "quadratic_index": {"regret_plugin": 0.18, "regret_crossfit": 0.18,
                                                      "plugin_interval": {"ci95": [-0.007, 0.36]}}}}}}}}}}
    m = ma.strict_more_macros(R)
    t = "dstrictBbhDisjointForced"
    assert m[t + "BestThird"] == "25.98" and m[t + "PanelGain"] == "7.10"
    assert m[t + "LinCross"] == "2.15" and m[t + "LinLo"] == "1.89" and m[t + "LinHi"] == "2.39"
    assert m[t + "QuadLo"] == "$-$0.01"
    assert t + "Own" not in m and letters(m)


def test_ceiling_more_macros_and_hotpot_task_tag():
    T = {"axes": {"within": {"protocols": 18, "empirical": {"1": 0.9574, "4": 0.978, "12": 0.9861},
                             "model": {"24": 0.9912}},
                  "arm": {"pairs": 9}, "random": {"pairs": 153}}}
    m = ma.ceiling_more_macros({"tasks": {"gsm8k": T, "hotpot": T}})
    assert m["pceilGsmWithinAllOne"] == "0.957" and m["pceilGsmWithinAllTwentyFourModel"] == "0.991"
    assert m["pceilGsmWithinLimit"] == "1" and m["pceilGsmArmPairs"] == "9" and m["pceilGsmRandomPairs"] == "153"
    assert m["pceilHqWithinAllProtocols"] == "18"                        # a HotpotQA task comes in under Hq
    head = {"D": 20, "headline": {"within_requested": {"1": 0.95, "10": 0.98}, "between_same_draws": {"1": 0.8},
                                  "plugin": 0.79, "plugin_bootstrap": {"ci95": [0.78, 0.8]}, "posterior": 0.8},
            "axes": {}}
    assert ma.ceiling_macros({"tasks": {"hotpot": head}})["pceilHqLimitPlugLo"] == "0.780"


def test_fewshot_more_macros():
    pm = [{"model_key": "qwen3-8b", "zero_shot_forced": 44.648, "five_shot_forced": 19.386, "gain": -25.26},
          {"model_key": "m2", "zero_shot_forced": 10, "five_shot_forced": 13, "gain": 3.0},
          {"model_key": "m3", "zero_shot_forced": 10, "five_shot_forced": 9.6, "gain": -0.4}]
    comp = [{"model_key": "qwen3-8b", "arm": "direct_answer", "compliance_pct": "78.739"},
            {"model_key": "qwen3-8b", "arm": "zero_shot_cot", "compliance_pct": "32.884"},
            {"model_key": "m2", "arm": "zero_shot_cot", "compliance_pct": "99"}]
    m = ma.fewshot_more_macros([{"tasks": {"bbh": {"per_model": pm}}}, {"dry_run": True, "tasks": {"math": {}}}], comp)
    assert m["fewshotBbhModels"] == "3" and m["fewshotBbhLosers"] == "2"
    assert m["fewshotBbhGainMin"] == "$-$25.3" and m["fewshotBbhGainMax"] == "3.0"
    assert m["fewshotBbhQwenThreeEightAccZero"] == "44.6" and m["fewshotBbhQwenThreeEightAccFive"] == "19.4"
    assert m["fewshotBbhQwenThreeEightComplianceZero"] == "78.7"
    assert m["fewshotBbhQwenThreeEightComplianceFive"] == "32.9"
    assert not any(k.startswith("fewshotMath") for k in m) and letters(m)


def test_constrained_more_macros_one_family():
    one = dict(slope_block(-0.04, fam=("Qwen",)), leave_one_family_out={"Qwen": "nan"})
    R = {"slopes": {"cot_minus_requested": {"all": slope_block(0.32)},
                    "cot_minus_forced": {"readable": one, "all": {"M": 2, "note": "too few"}},
                    "cot_minus_constrained": {"all": slope_block(1.31, fam=("A", "B", "C"))}}}
    m = ma.constrained_more_macros(R)
    assert m["gcSlopeForcedReadableLofoMin"] == "--" and m["gcSlopeForcedReadableFamilies"] == "1"
    assert m["gcSlopeForcedReadableDrawLo"] == "$-$0.05"
    assert m["gcSlopeConstrainedAllLofoMin"] == "1.31" and m["gcSlopeConstrainedAllLofoMax"] == "1.41"
    assert "gcSlopeForcedAllDrawLo" not in m                            # no slope, nothing added
    assert ma.constrained_more_macros({"dry_run": True}) == {}


def test_vendor_more_macros():
    def cell(cv, cg, tv, tg):
        return {"vendor": {"cut": {"pct": cv}, "mean_output_tokens": tv},
                "greedy": {"cut": {"pct": cg}, "mean_output_tokens": tg}}
    R = {"cells": {"gsm8k/a": cell(1.33, 17.8, 4189.2, 7474), "gsm8k/b": cell(4.3, 44.0, 5000, 9000),
                   "math/a": cell(2.5, 30.5, 11443, 18934)}}
    m = ma.vendor_more_macros(R)
    assert m["tvGsmCutVendorMin"] == "1.3" and m["tvGsmCutGreedyMax"] == "44.0" and m["tvMathCutVendorMax"] == "2.5"
    assert m["tvTokensVendorMin"] == "4{,}189" and m["tvTokensGreedyMax"] == "18{,}934"


def test_validation_agreement_macros():
    def s(name, c):
        return {"setting": name, "validation": {"one_draw": {"benefit": {"C": c}}}}
    m = ma.validation_agreement_macros({"settings": [s("dove_mmlu_block", 0.63478), s("gsm8k30", 0.955999),
                                                     s("bbh:boolean_expressions", 0.7)]})
    assert m == {"daCbenVal": "0.635", "gtCbenVal": "0.956"}


# ---------------------------------------------------------------- new prefixes


def test_arm_effect_macros():
    def rev(pairs, r):
        return {"pairs": pairs, "reversed": r, "reversed_share": r / pairs, "ties": 0,
                "by_threshold": [{"min_gap_pp": 0.0, "pairs_both_above": pairs, "reversed": r},
                                 {"min_gap_pp": 5.0, "pairs_both_above": 294, "reversed": 35},
                                 {"min_gap_pp": 10.0, "pairs_both_above": 217, "reversed": 14},
                                 {"min_gap_pp": 7.0, "pairs_both_above": 1, "reversed": 0}]}
    R = {"forced_cap_tokens": 16, "short_share_pct": 10.0, "tasks": {"gsm8k": {
        "M": 29, "reversals": rev(406, 83), "readable_M": 23, "reversals_readable": rev(253, 69),
        "pearson_long_share_vs_req_minus_forced": 0.6437, "spearman_long_share_vs_req_minus_forced": 0.6901,
        "short_models": ["a"] * 8, "short_models_req_minus_forced_range": [-18.404, 7.262],
        "short_models_req_minus_forced_mean": -3.56, "mean_req_minus_forced": 8.15}}}
    m = ma.arm_effect_macros(R)
    assert m["aeGsmReversed"] == "83" and m["aeGsmPairs"] == "406" and m["aeGsmReversedPct"] == "20"
    assert m["aeGsmReversedShare"] == "0.204" and m["aeGsmGapFiveReversed"] == "35" and m["aeGsmGapTenPairs"] == "217"
    assert m["aeGsmReadableReversed"] == "69" and m["aeGsmReadableM"] == "23"
    assert m["aeGsmPearson"] == "0.64" and m["aeGsmSpearman"] == "0.69"
    assert m["aeGsmShortMin"] == "$-$18.4" and m["aeGsmShortMax"] == "7.3" and m["aeGsmShortM"] == "8"
    assert "aeGsmGapSevenPairs" not in m and letters(m)


def test_resampling_macros():
    one = dict(slope_block(-0.04, fam=("Qwen",)), leave_one_family_out={"Qwen": "nan"},
               family_bootstrap=boot(-0.0404, -0.0404, 1.0))
    R = {"tasks": {"bbh": {"blocks": {"CotMinusForced/readable": one}},
                   "math": {"blocks": {"CotMinusForced/readable": dict(slope_block(0.3712), draw_set={"kind": "whole_line"})}}},
         "item_g_change": {"estimate": 1.0573, "gsm8k": -0.686, "math": 0.371, "model_bootstrap": boot(0.698, 1.458),
                           "family_bootstrap": boot(0.63, 1.612)},
         "math_level_trend": {"estimate": 0.4719, "level_slopes": [-0.766, -0.306, 0.226, 0.415, 1.232],
                              "model_bootstrap": boot(0.309, 0.842), "family_bootstrap": boot(0.268, 1.028, 0.0003)}}
    m = ma.resampling_macros(R)
    assert m["mresBbhForcedReadableFamilies"] == "1" and m["mresBbhForcedReadableFamilyLo"] == "$-$0.040"
    assert m["mresBbhForcedReadableLofoMin"] == "--"                     # no family can be left out
    assert m["mresMathForcedReadable"] == "0.371" and m["mresMathForcedReadableDrawLo"] == "--"
    assert m["mresMathForcedReadableDrawKind"] == "whole line"           # an unbounded draw set, as it is
    assert m["mresChange"] == "1.057" and m["mresChangeModelLo"] == "0.698" and m["mresChangeFamilyHi"] == "1.612"
    assert m["mresLevelTrend"] == "0.472" and m["mresLevelFive"] == "1.232"
    assert m["mresLevelTrendFamilyShareBelow"] == "0.000" and letters(m)


def test_difficulty_macros():
    def stratum(name, s, cot):
        return {"stratum": name, "requested": 55.1, "forced": 53.0, "cot": cot, "slope_cot_minus_forced": s,
                "set": {"kind": "bounded", "a": s - 0.004, "b": s + 0.004}, "ceiling_slope": 1 + s,
                "slope_cot_minus_requested": -0.88 - 0.01 * len(name),
                "set_requested": {"kind": "bounded", "a": -0.9, "b": -0.87}, "model_bootstrap": boot(s - 0.1, s + 0.8, 0.19)}
    P = {"M": 23, "strata": [stratum("<=2", -0.8232, 94.68), stratum(">=6", 0.2442, 82.1)],
         "trend": {"estimate": 0.2373, "ci95": [0.2198, 0.2548], "model_bootstrap": boot(0.1357, 0.4748),
                   "family_bootstrap": boot(0.1632, 0.5858, 0.0004)}}
    m = ma.difficulty_macros({"steps": {"n_items": {"<=2": 427, ">=6": 91}, "panels": {"readable": P}}})
    t = "gdiffStepsReadable"
    assert m["gdiffStepsUpToTwoItems"] == "427" and m[t + "M"] == "23"
    assert m[t + "UpToTwoSlope"] == "$-$0.823" and m[t + "SixUpSlope"] == "0.244"
    assert m[t + "SixUpSlopeModelShareBelow"] == "0.190" and m[t + "SixUpCot"] == "82.1"
    assert m[t + "Trend"] == "0.237" and m[t + "TrendDrawLo"] == "0.220" and m[t + "TrendFamilyHi"] == "0.586"
    assert m[t + "CotMax"] == "94.7" and m[t + "CotMin"] == "82.1"
    assert m[t + "SlopeReqMin"] == "$-$0.910" and m[t + "SlopeReqMax"] == "$-$0.910" and letters(m)


def test_sweep_macros_counts_like_the_summary():
    def lad(task, var, out, name, sign, loses):
        return {"task": task, "variant": var, "outcome": out, "ladder": name, "kojima_sign": sign,
                "wei_smallest_loses": loses}
    R = {"cutoffs": [{"task": "gsm8k", "readability_basis": "gsm8k", "cutoff_pct": 92.5, "M": 23, "slope": -0.686,
                      "set_kind": "bounded", "lo": -0.691, "hi": -0.681},
                     {"task": "bbh", "readability_basis": "own", "cutoff_pct": 99.0, "M": 0, "models": ""}],
         "ladders": [lad("gsm8k", "primary", "CotMinusForced", "Gemma-1", "above zero", True),
                     lad("gsm8k", "primary", "CotMinusForced", "Gemma-2", "below zero", False),
                     lad("gsm8k", "primary", "CotMinusForced", "Qwen3.6", "above zero", False),
                     lad("gsm8k", "dense_readable", "CotMinusForced", "Gemma-2", "below zero", False)]}
    m = ma.sweep_macros(R)
    assert m["rswGsmGsmBasisNinetyTwoHalfSlope"] == "$-$0.686" and m["rswGsmGsmBasisNinetyTwoHalfSlopeLo"] == "$-$0.691"
    assert m["rswBbhOwnBasisNinetyNineM"] == "0" and m["rswBbhOwnBasisNinetyNineSlope"] == "--"
    assert m["rswGsmForcedPrimaryLadders"] == "3" and m["rswGsmForcedPrimaryRise"] == "2"
    assert m["rswGsmForcedPrimaryFall"] == "1" and m["rswGsmForcedPrimarySmallLoses"] == "1"
    assert m["rswGsmForcedDenseReadableRise"] == "0" and letters(m)


def test_heldout_macros():
    def rule(risk, r2, diff, lo, hi):
        return {"risk": risk, "heldout_r2": r2, "risk_constant_minus_rule": diff,
                "bootstrap_constant_minus_rule": {"ci95": [lo, hi]}}
    O = {"own_gain_risk": -0.3666, "model": {"constant": rule(674.1, 0.0, 0.0, 0, 0),
                                              "baseline_quadratic_index": rule(298.0, 0.558, 376.2, 181.9, 591.1),
                                              "index_linear": rule(693.3, -0.0285, -19.2, -158.0, 104.5)},
         "family": {"index_linear": rule(694.1, -0.081, -52.0, -160.4, 84.3)}}
    O2 = dict(O, own_gain_risk=0.29)
    m = ma.heldout_macros({"tasks": {"gsm8k": {"CotMinusReq": O, "CotMinusForced": O2}}})
    assert m["hcapGsmReqModelBaseQuadRTwo"] == "0.56" and m["hcapGsmReqModelBaseQuadDiffLo"] == "181.9"
    assert m["hcapGsmReqModelConstRisk"] == "674.1" and "hcapGsmReqModelConstRTwo" not in m
    assert m["hcapGsmIndexRTwoMin"] == "$-$0.08" and m["hcapGsmIndexRTwoMax"] == "$-$0.03"
    assert m["hcapOwnRiskMin"] == "$-$0.37" and m["hcapOwnRiskMax"] == "0.29"
    assert m["hcapGsmReqRTwoMax"] == "0.56" and letters(m)


def test_lowr_macros():
    def band(setting, name, n, dep1, dep4):
        return {"setting": setting, "band": name, "n": n, "median_R": 0.125, "departure_one_draw": dep1,
                "departure_K4": dep4}
    R = {"n_replicates": 2604, "R_outside_unit_interval": 0, "n_replicates_added": 2604,
         "R_outside_unit_interval_added": 1,
         "bands": [band("all", "0 <= R < 0.2", 6, 0.0027, -0.0008), band("all", "0.4 <= R < 0.6", 47, 0.018, 0.02),
                   band("all", "R >= 0.8", 2322, 0.006, 0.0), band("fewshot_gsm8k", "R < 0", 1, "nan", "nan")],
         "null_bands": [band("all", "0 <= R < 0.2", 13, 0.0218, 0.0515),
                        band("fewshot_gsm8k", "R < 0", 3, "nan", "nan")]}
    m = ma.lowr_macros(R)
    assert m["lowrBelowSix"] == "53" and m["lowrBelowSixPct"] == "2.0" and m["lowrRoutsideAdded"] == "1"
    assert m["lowrAllBandZeroN"] == "6" and m["lowrAllBandZeroNullN"] == "13"
    assert m["lowrAllBandZeroDataMinusNullFour"] == "$-$0.052" and m["lowrAllBandZeroDataMinusNullOne"] == "$-$0.019"
    assert m["lowrFsGsmBandNegDepOne"] == "--" and m["lowrFsGsmBandNegDataMinusNullOne"] == "--"   # no reference
    assert "lowrAllBandFourNullN" not in m and letters(m)


def test_new_functions_on_the_real_results_do_not_collide():
    """Every function of the request on the repository's results files (where present): letters only, and no name
    set by two of them."""
    def load(*parts):
        p = os.path.join(ROOT, "results", *parts)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    seen = {}
    for fn, parts in ((ma.strict_more_macros, ("decision_strict", "decision_strict.json")),
                      (ma.ceiling_more_macros, ("protocol_ceiling", "protocol_ceiling.json")),
                      (ma.constrained_more_macros, ("forced_constrained", "forced_constrained.json")),
                      (ma.vendor_more_macros, ("thinking_vendor", "thinking_vendor.json")),
                      (ma.arm_effect_macros, ("arm_effect", "arm_effect.json")),
                      (ma.resampling_macros, ("model_resampling", "model_resampling.json")),
                      (ma.difficulty_macros, ("gsm8k_difficulty", "gsm8k_difficulty.json")),
                      (ma.sweep_macros, ("readable_sweep", "readable_sweep.json")),
                      (ma.heldout_macros, ("heldout_capability", "heldout_capability.json")),
                      (ma.lowr_macros, ("lowr_map", "lowr_map.json")),
                      (ma.validation_agreement_macros, ("reliability_map", "reliability_map.json"))):
        R = load(*parts)
        if R is None:
            continue
        m = fn(R)
        assert m and letters(m)
        both = set(m) & set(seen)
        assert not both, (fn.__name__, sorted(both)[:5])
        seen.update(m)


# ---------------------------------------------------------------- HotpotQA and the arm wording


def hotpot_report_dict():
    ev = {"n_configurations": 10, "best_third_value": 60.0, "panel_mean_gain": 30.0,
          "rules": {"capability_1": {"regret_plugin": 12.345, "regret_crossfit": 12.0},
                    "own_1": {"regret_plugin": 0.3, "regret_crossfit": 0.1},
                    "random": {"regret_plugin": 9.0, "regret_crossfit": 8.9}},
          "paired": [{"a": "capability_1", "b": "own_1", "estimate": 12.0, "ci95": [11.5, 12.5]}]}
    contrast = {"means": {"baseline": 40.04, "intervention": 52.1, "gain": 12.06},
                "map": {"point": {"benefit": {"R": 0.9612}, "accuracy": {"R": 0.99}},
                        "validation_one_draw": {"benefit": {"C": 0.9011}, "accuracy": {"C": 0.97}}},
                "capability_slope": slope_block(0.25), "f1": {"step2": {"V_delta": {"estimate": 120.5, "ci95": [80, 200]}}},
                "decision": {"saved": "x", "evaluation": {"independent": ev}}}
    check = {"a": {"predicted": 0.91, "observed": 0.9011, "within_0.05": True},
             "b": {"0.8": {"K_needed": 1, "observed_C_at_K": 0.9011, "reached": True},
                   "0.9": {"K_needed": 13, "observed_C_at_K": None, "reached": None}},
             "d": {"R_benefit": 0.95, "forecast_own_better": True, "observed_own_better": True, "holds": True,
                   "regret": {"own_1": 0.3, "capability_1": 12.3}}}
    return {"M": 29, "n_items": 1000, "fresh_configurations": {"n_draws": 7, "n_distinct": 5},
            "contrasts": {"cot_req": contrast, "cot_forced": contrast, "fewshot": contrast},
            "tranche": {"checks": {"cot_req": check, "cot_forced": check, "fewshot": check,
                                   "c_arm_switch": {"predicted": True, "holds": False,
                                                    "observed": {"switch": {"opposite": 0.2},
                                                                 "new_draw": {"opposite": 0.25},
                                                                 "switch_reorders_more": False}}}}}


def test_hotpot_skipped_until_scored_then_taken(tmp_path):
    root = tmp_path
    d = root / "results" / "hotpot"
    d.mkdir(parents=True)
    (d / "pilot_check.json").write_text("{}")
    steps_src = os.path.join(ROOT, "results", "dataset2", "gsm8k30.json")
    if os.path.exists(steps_src):
        shutil.copy(steps_src, d / "hotpot.json")      # a Steps 1-3 file from an earlier stage is not read alone
    assert ma.hotpot_report(str(root)) is None and ma.hotpot_steps(str(root)) == []
    assert "not scored yet" in ma.pending(str(root))["HotpotQA"]
    (d / "hotpot_report.json").write_text(json.dumps(hotpot_report_dict()))
    steps = ma.hotpot_steps(str(root))
    if os.path.exists(steps_src):
        assert steps == [(str(d / "hotpot.json"), "hqReq")]
    assert ma.pending(str(root))["HotpotQA"].startswith("scored")
    m = ma.hotpot_macros(ma.hotpot_report(str(root)))
    assert letters(m) and m["hqM"] == "29" and m["hqItems"] == "1{,}000" and m["hqFreshDraws"] == "7"
    for c in ("Req", "Fd", "Fs"):
        assert m[f"hq{c}MeanGainFine"] == "12.06" and m[f"hq{c}MapRben"] == "0.961" and m[f"hq{c}MapCval"] == "0.901"
        assert m[f"hq{c}Slope"] == "0.250" and m[f"hq{c}SlopeModelLo"] == "$-$0.150"
        assert m[f"hq{c}DecCapOnePlug"] == "12.35" and m[f"hq{c}DecCapMinusOwnLo"] == "11.50"
        assert m[f"hq{c}FOneVdeltaHi"] == "200.00"
        assert m[f"hq{c}TrCWithin"] == "yes" and m[f"hq{c}TrKEighty"] == "1" and m[f"hq{c}TrReachedNinety"] == "--"
        assert m[f"hq{c}TrCAtKNinety"] == "--" and m[f"hq{c}TrOwnBetterHolds"] == "yes"
    assert m["hqTrSwitchMore"] == "no" and m["hqTrSwitchHolds"] == "no" and m["hqTrNewDrawReversed"] == "0.250"
    # the names d2_macros gives the Steps 1-3 files (prefixes hqReq, hqFd, hqFs) are not among these
    d2_names = ("M", "Fam", "Kdisc", "Kval", "Cben", "Cacc", "Gap", "Vdelta", "GainSD", "H", "Hset", "Rben", "Racc",
                "Wgain", "MeanGain", "MeanBase")
    for p in ("hqReq", "hqFd", "hqFs"):
        assert not any(p + n in m for n in d2_names)


def test_hotpot_task_tag_in_the_decision_reader():
    assert ma.tag_of("hotpot_req") == "HqReq" and ma.tag_of("hotpot_fd") == "HqFd"


def _aw_table(C=4, T=2, D=2, M=6, I=12, seed=0):
    from src import arm_wording as aw
    rng = np.random.default_rng(seed)
    ab = rng.normal(0, 1, M)
    rows = []
    for task in aw.TASKS:
        for c in range(C):
            for arm, ws in ((aw.COT, [f"cot{t}" for t in range(T)]), (aw.REQ, [f"dir{d}" for d in range(D)]),
                            (aw.FD, [f"dir{d}" for d in range(D)])):
                for w in ws:
                    for m in range(M):
                        p = 1 / (1 + np.exp(-(ab[m] + (0.8 if arm == aw.COT else 0) + rng.normal(0, 0.3))))
                        for i in range(I):
                            rows.append({"task": task, "config": f"k{c}", "wording": w, "arm": arm,
                                         "model_key": f"m{m}", "item_id": str(i),
                                         "strict_correct": float(rng.random() < p), "lenient_correct": 1.0,
                                         "n_output_tokens": 10, "box_closed": True, "prompt_sha256": ""})
    return pd.DataFrame(rows)


def aw_results(synthetic_seed=None):
    """An arm_wording.json built by src/arm_wording's own functions on synthetic scores, assembled as
    scripts/run_arm_wording.full assembles it (strict block; small resampling counts)."""
    from src import arm_wording as aw
    tab = _aw_table()
    confs, cot_ids, dir_ids, models = [f"k{c}" for c in range(4)], ["cot0", "cot1"], ["dir0", "dir1"], \
        [f"m{m}" for m in range(6)]
    per_task = {}
    for task in aw.TASKS:
        ta = aw.TaskArrays(tab, task, confs, cot_ids, dir_ids, models)
        acc = ta.acc()
        G = aw.gains(acc)
        block = {"R": aw.r_table(G), "reversals": aw.reversal_table(G),
                 "R_fixed_any_wording": {b: aw.fixed_any_wording(G[b]) for b in G},
                 "R_leave_one_configuration_out": {b: {s: aw.leave_one_config_out(G[b], s) for s in aw.SCHEMES}
                                                   for b in G},
                 "R_item_bootstrap_ci95": aw.item_bootstrap_R(ta, 5, 0),
                 "direct_gap": aw.direct_gap(acc, dir_ids, {"dir0": 0.25}),
                 "ceiling": aw.ceiling_table(G, 0, n_splits=3, n_boot=3)}
        ch = json.loads(json.dumps(aw.decision_choices({a: v[:2] for a, v in acc.items()}, 2), default=aw._default))
        block["decision"] = aw.decision_evaluate(ch, aw.gains({a: v[2:] for a, v in acc.items()}))
        block["prediction_checks"] = {"C_both_drawn": {"predicted": 0.9, "observed_validation": 0.84,
                                                       "within_tolerance": False}}
        per_task[task] = {"configurations": {"discovery": confs[:2], "validation": confs[2:]}, "strict": block}
    strict = {t: {"R": v["strict"]["R"], "reversals": v["strict"]["reversals"], "decision": v["strict"]["decision"]}
              for t, v in per_task.items()}
    out = {"kind": "arm_wording_results", "wordings": {"cot": [{"id": i} for i in cot_ids],
                                                       "dir": [{"id": i} for i in dir_ids]},
           "models": models, "n_third": 2, "per_task": per_task,
           "repeat_check": {"gsm8k": {"per_arm": {"forced_direct": {"same_strict_correctness": 0.991,
                                                                    "accuracy_difference_pp_mean": -0.12}}},
                            "math": {"note": "the collection's scored tables are absent"}},
           "verdicts": aw.verdicts(strict, aw.load_config()["analysis"]["abstract"]),
           "synthetic_seed": synthetic_seed}
    return out


def test_arm_wording_skipped_until_scored_then_taken(tmp_path):
    from src import arm_wording as aw
    d = tmp_path / "results" / "arm_wording"
    d.mkdir(parents=True)
    assert ma.arm_wording_results(str(tmp_path)) is None
    assert "not scored yet" in ma.pending(str(tmp_path))["arm wording"]
    aw.write_json(d / "arm_wording.json", aw_results(synthetic_seed=7))           # a dry run on synthetic scores
    assert ma.arm_wording_results(str(tmp_path)) is None
    assert "dry run" in ma.pending(str(tmp_path))["arm wording"]
    assert ma.arm_wording_macros(json.load(open(d / "arm_wording.json"))) == {}
    R0 = aw_results()
    aw.write_json(d / "arm_wording.json", R0)
    R = ma.arm_wording_results(str(tmp_path))
    assert R is not None and ma.pending(str(tmp_path))["arm wording"].startswith("scored")
    m = ma.arm_wording_macros(R)
    assert letters(m) and m["awM"] == "6" and m["awTriggers"] == "2" and m["awClauses"] == "2"
    g = R0["per_task"]["gsm8k"]["strict"]
    assert m["awGsmReqBothDrawnR"] == ma.num(g["R"]["req"]["both_drawn"]["R"], 3)
    assert m["awGsmFdFixedCone"] == ma.num(g["R"]["fd"]["fixed"]["C_one_draw"], 3)
    assert m["awGsmReqBothDrawnRLo"] == ma.num(g["R_item_bootstrap_ci95"]["req"]["both_drawn"][0], 3)
    assert m["awGsmArmSwitchOneMinusC"] == ma.num(g["reversals"]["arm_switch"]["one_minus_C"], 3)
    assert m["awGsmArmSwitchClauseTwo"] == ma.num(g["reversals"]["arm_switch_by_clause"][1]["one_minus_C"], 3)
    assert m["awGsmGapClauseOneLong"] == "0.250" and m["awGsmGapClauseTwoLong"] == "--"
    assert m["awGsmCeilArmPairs"] == str(g["ceiling"]["arm"]["pairs"])
    assert m["awGsmCeilWithinProtocols"] == str(g["ceiling"]["within"]["protocols"])
    assert m["awGsmDecSameExtra"] == "0.00" and "awGsmDecTriggerExtraLo" in m
    assert m["awGsmPredCWithin"] == "no" and m["awGsmRepeatForcedDirectSame"] == "0.991"
    assert not any(k.startswith("awMathRepeat") for k in m)
    assert m["awOnePromptAbstractChanges"] in ("yes", "no")
    assert all(k.startswith("awMath") for k in m if k.startswith("awMath"))
    assert any(k.startswith("awMathReq") for k in m)
