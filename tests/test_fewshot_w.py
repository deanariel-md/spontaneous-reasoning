"""Tests of w under five-shot answering: src/fewshot_w.py, scripts/run_fewshot_w.py,
paper/macros_fewshot_w.py. The formulas on small synthetic cases; then, where the saved outputs are present, the
reproduction of the published five-shot numbers, w and compliance recounted from the scored table, and the macro names
checked against paper/numbers.tex."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "paper"), os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import fewshot_w as fw  # noqa: E402
import macros_fewshot_w as mfw  # noqa: E402

RES = os.path.join(ROOT, "results", "fewshot_w", "fewshot_w.json")
SCORED = os.path.join(ROOT, "data", "processed", "fewshot_requested", "all_scored.parquet")
NUMBERS = os.path.join(ROOT, "paper", "numbers.tex")
have_results = pytest.mark.skipif(not os.path.exists(RES), reason="results/fewshot_w/fewshot_w.json not written yet")


# ---------------------------------------------------------------------------- w and compliance
def test_long_share_is_configuration_mean_with_strict_greater_than_cap():
    # model 0: config 0 has 4 rows (tokens 16, 17, 5, 40 -> 2 long), config 1 has 2 rows (16, 16 -> 0 long)
    # model 1: config 0 has 1 row (100 -> long), config 1 has 3 rows (1, 2, 17 -> 1 long)
    tok = [16, 17, 5, 40, 16, 16, 100, 1, 2, 17]
    m = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    c = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
    w = fw.long_share(tok, 16, m, c, 2, 2)
    assert w[0] == pytest.approx((2 / 4 + 0 / 2) / 2)          # 0.25, not the pooled 2/6
    assert w[1] == pytest.approx((1 / 1 + 1 / 3) / 2)          # 0.6667, not the pooled 2/4
    assert fw.long_share([16], 16, [0], [0], 1, 1)[0] == 0.0   # exactly the cap is not long


def test_missing_cell_gives_nan_not_imputed():
    w = fw.config_mean_share([True, False], [0, 0], [0, 0], 1, 2)
    assert np.isnan(w[0])


# ---------------------------------------------------------------------------- the sign-change variable
def test_sign_change_binary_and_zero_gain_is_no_change():
    g_r = np.array([-3.0, 2.0, 1.0, -1.0, 0.0, 5e-10, 4.0])
    g_f = np.array([2.0, -1.0, 1.0, -2.0, 3.0, -2.0, 0.0])
    assert fw.sign_change(g_r, g_f).tolist() == [1, 1, 0, 0, 0, 0, 0]
    d = fw.direction_counts(g_r, g_f)
    assert d == {"forced_up_requested_down": 1, "forced_down_requested_up": 1, "both_up": 1, "both_down": 1,
                 "zero_under_either": 3}


# ---------------------------------------------------------------------------- the rank association
def test_association_hand_case():
    fall = np.array([0.5, 0.4, 0.1, 0.0])
    s = np.array([1, 1, 0, 0])
    A = fw.association(fall, s, n_boot=0)
    # ranks of fall 4,3,2,1; ranks of S 3.5,3.5,1.5,1.5 -> rho = 4 / sqrt(5 * 4)
    assert A["spearman"] == pytest.approx(4 / np.sqrt(20))
    # 4 concordant pairs, 0 discordant, 2 pairs tied in S, none in fall -> tau_b = 4 / sqrt(6 * 4)
    assert A["kendall_tau_b"] == pytest.approx(4 / np.sqrt(24))
    assert A["mean_fall_difference"] == pytest.approx(0.45 - 0.05)
    assert (A["n_models"], A["n_sign_change"], A["n_no_change"], A["variable"]) == (4, 2, 2, "binary")


def test_association_constant_s_is_undefined_not_zero():
    A = fw.association(np.array([0.3, 0.2, 0.1]), np.array([0, 0, 0]), n_boot=0)
    assert np.isnan(A["spearman"]) and np.isnan(A["kendall_tau_b"]) and np.isnan(A["mean_fall_difference"])


def test_bootstrap_counts_undefined_resamples():
    fall = np.linspace(0, 1, 12)
    s = np.zeros(12, int)
    s[-1] = 1
    n_boot = 500
    A = fw.association(fall, s, n_boot=n_boot, rng=np.random.default_rng(7))
    rng = np.random.default_rng(7)
    expected = 0
    for _ in range(n_boot):
        ix = rng.integers(0, 12, 12)
        expected += int(s[ix].min() == s[ix].max())
    assert A["boot_undefined_rank"] == expected > 0
    lo, hi = A["spearman_ci95"]
    assert -1 <= lo <= hi <= 1


def test_continuous_companion_sign():
    fall = np.array([0.9, 0.6, 0.3, 0.1, 0.0])
    cont = -fall * 10 + np.array([0.1, -0.2, 0.0, 0.3, 0.0])
    A = fw.association(fall, np.array([1, 1, 0, 0, 0]), n_boot=0, cont=cont)
    assert A["continuous"]["spearman"] == pytest.approx(-1.0)


def test_distribution_counts_exact_zero_apart():
    d = fw.distribution([-0.2, 0.0, 0.1, -0.05])
    assert d == {"median": pytest.approx(-0.025), "min": -0.2, "max": 0.1, "n_below_zero": 2, "n_zero": 1,
                 "n_above_zero": 1}


# ---------------------------------------------------------------------------- the macro module
def test_model_token_letters_only_and_distinct():
    assert mfw.model_token("mistral-7b-instruct-v0.3") == "MistralSevenBInstructVZeroPtThree"
    assert mfw.model_token("qwen3.6-35b-a3b") == "QwenThreePtSixThreeFiveBAThreeB"
    assert mfw.model_token("qwen3-30b-a3b") != mfw.model_token("qwen3-30b-a3b-instruct-2507")
    assert mfw.model_token("qwen2.5-7b") != mfw.model_token("qwen25-7b")


def _synthetic_results():
    def assoc(n1):
        return {"n_models": 3, "n_sign_change": n1, "n_no_change": 3 - n1, "variable": "binary", "spearman": 0.5,
                "kendall_tau_b": -0.4, "mean_fall_sign_change": 0.3, "mean_fall_no_change": 0.1,
                "mean_fall_difference": 0.2, "n_boot": 10, "spearman_ci95": [-0.1, 0.9],
                "kendall_tau_b_ci95": [-0.8, 0.2], "mean_fall_difference_ci95": [-0.05, 0.4], "boot_undefined_rank": 2,
                "boot_undefined_difference": 2,
                "continuous": {"spearman": -0.7, "kendall_tau_b": -0.5, "spearman_ci95": [-0.9, -0.2],
                               "kendall_tau_b_ci95": [-0.8, -0.1], "boot_undefined": 0}}

    def split():
        d = {"median": -0.3, "min": -0.9, "max": 0.004, "n_below_zero": 2, "n_zero": 0, "n_above_zero": 1}
        return {"w": {"zero_shot_mean": 0.437, "five_shot_mean": 0.04, "change_mean": {"mean": -0.398, "ci95": [-0.5, -0.3]},
                      "change_distribution": d},
                "compliance": {"zero_shot_mean": 0.85, "five_shot_mean": 0.96,
                               "change_mean": {"mean": 0.11, "ci95": [0.05, 0.2]}, "change_distribution": d},
                "scores": {sc: {"mean_gain_requested": -5.456, "mean_gain_forced": 2.98,
                                "directions": {"forced_up_requested_down": 1, "forced_down_requested_up": 0, "both_up": 2,
                                               "both_down": 0, "zero_under_either": 0},
                                "association": assoc(1)} for sc in ("strict", "content")}}

    rows = [{"model_key": k, "w_zero_shot": 0.5, "w_five_shot": 0.0, "w_change": -0.5}
            for k in ("gemma-2-9b-it", "qwen3.6-27b", "mistral-7b-instruct-v0.3")]
    task = {"cap_tokens": 16, "M": 3, "splits": {sp: split() for sp in ("all", "discovery", "validation")},
            "per_model_all": rows}
    cnt = {"n_replies": 1234, "strict_correct": 10, "content_correct": 12, "unreadable": 1}
    return {"n_boot": 9999, "tasks": {"gsm8k": task, "bbh": json.loads(json.dumps(task))},
            "content_record": {"counts": {"gsm8k": cnt, "bbh": cnt}}}


def test_macros_synthetic_names_and_values():
    M = mfw.fewshot_w_macros(_synthetic_results())
    assert all(re.fullmatch(r"fsW[A-Za-z]+", k) for k in M)
    assert M["fsWGsmWZsPct"] == "43.7" and M["fsWGsmWFsPct"] == "4.0"
    assert M["fsWGsmWChange"] == "$-$39.8" and M["fsWGsmWChangeLo"] == "$-$50.0"
    assert M["fsWGsmGainReq"] == "$-$5.46" and M["fsWBbhValGainReqContent"] == "$-$5.46"
    assert M["fsWGsmTau"] == "$-$0.40" and M["fsWGsmDiscRhoContentHi"] == "0.90"
    assert M["fsWGsmFallDiff"] == "20.0" and M["fsWGsmBootUndef"] == "2"
    assert M["fsWGsmModelMistralSevenBInstructVZeroPtThreeWChange"] == "$-$50.0"
    assert M["fsWNBoot"] == "9{,}999" and M["fsWCap"] == "16" and M["fsWGsmFsReqReplies"] == "1{,}234"
    assert M["fsWGsmWChangeMax"] == "0.4"
    assert mfw.fewshot_w_macros(None) == {}


def test_macros_refuse_missing_value():
    R = _synthetic_results()
    R["tasks"]["gsm8k"]["splits"]["all"]["scores"]["strict"]["association"]["spearman_ci95"] = None
    with pytest.raises(ValueError):
        mfw.fewshot_w_macros(R)


# ---------------------------------------------------------------------------- the real results
@have_results
def test_published_five_shot_numbers_reproduce():
    R = json.load(open(RES, encoding="utf-8"))
    pub = json.load(open(os.path.join(ROOT, "results", "fewshot_requested", "fewshot_requested.json"), encoding="utf-8"))
    for t, b in R["tasks"].items():
        rp = b["reproduction"]
        assert rp["reproduced"] and rp["max_abs_diff_pp"] <= 1e-9
        assert rp["values_compared"] == 8 * b["M"] + 2
        for k in ("requested", "forced"):
            assert rp["mean_gain_validation"][k] == pytest.approx(pub["tasks"][t]["strict"]["mean_gain_validation"][k],
                                                                  abs=1e-9)
        val = b["splits"]["validation"]["scores"]["strict"]
        assert val["mean_gain_requested"] == pytest.approx(pub["tasks"][t]["strict"]["mean_gain_validation"]["requested"],
                                                           abs=1e-9)
        for chk in b["row_checks"].values():
            assert all(v == 0 for k, v in chk.items() if k.endswith("_mismatches"))
    # the published macros \fsrGsmGainReq and \fsrBbhGainReq (validation, strict) from this run's values
    if os.path.exists(NUMBERS):
        txt = open(NUMBERS, encoding="utf-8").read()
        for t, tag in (("gsm8k", "Gsm"), ("bbh", "Bbh")):
            m = re.search(r"\\newcommand\{\\fsr" + tag + r"GainReq\}\{([^}]*)\}", txt)
            v = R["tasks"][t]["splits"]["validation"]["scores"]["strict"]["mean_gain_requested"]
            assert m and m.group(1) == f"{v:.2f}".replace("-", "$-$")


@have_results
@pytest.mark.skipif(not os.path.exists(SCORED), reason="five-shot requested scored table absent")
def test_w_and_compliance_recounted_from_scored_table():
    R = json.load(open(RES, encoding="utf-8"))
    d = pd.read_parquet(SCORED, columns=["model_key", "task", "config_id", "n_output_tokens", "compliant"])
    d["long"] = d["n_output_tokens"] > 16
    per = d.groupby(["task", "model_key", "config_id"])[["long", "compliant"]].mean().groupby(["task", "model_key"]).mean()
    for t, b in R["tasks"].items():
        assert b["cap_tokens"] == 16
        for r in b["per_model_all"]:
            assert r["w_five_shot"] == pytest.approx(per.loc[(t, r["model_key"]), "long"], abs=1e-12)
            assert r["compliance_five_shot"] == pytest.approx(per.loc[(t, r["model_key"]), "compliant"], abs=1e-12)
        s = b["splits"]["all"]
        assert s["w"]["five_shot_mean"] == pytest.approx(np.mean([r["w_five_shot"] for r in b["per_model_all"]]))
        ch = np.array([r["w_change"] for r in b["per_model_all"]])
        assert s["w"]["change_distribution"]["n_below_zero"] == int((ch < 0).sum())
        st = s["scores"]["strict"]
        S = np.array([r["sign_change_strict"] for r in b["per_model_all"]])
        assert st["association"]["n_sign_change"] == int(S.sum())
        assert st["association"]["variable"] == "binary"
        exp = fw.sign_change([r["g_requested_strict"] for r in b["per_model_all"]],
                             [r["g_forced_strict"] for r in b["per_model_all"]])
        assert exp.tolist() == S.tolist()


@have_results
def test_macros_from_results_new_and_prefixed():
    M = mfw.fewshot_w_macros(mfw.fewshot_w_results(ROOT))
    R = json.load(open(RES, encoding="utf-8"))
    assert len(M) > 0 and all(k.startswith("fsW") and re.fullmatch(r"[A-Za-z]+", k) for k in M)
    a = R["tasks"]["gsm8k"]["splits"]["all"]
    assert M["fsWGsmRho"] == f"{a['scores']['strict']['association']['spearman']:.2f}".replace("-", "$-$")
    assert M["fsWGsmSignN"] == str(a["scores"]["strict"]["association"]["n_sign_change"])
    if os.path.exists(NUMBERS):
        names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(NUMBERS, encoding="utf-8").read())
        mk = open(os.path.join(ROOT, "paper", "make_numbers.py"), encoding="utf-8").read()
        if "macros_fewshot_w" in mk:            # once wired: each name defined exactly once
            for k in M:
                assert names.count(k) == 1, f"{k} defined {names.count(k)} times in numbers.tex"
            return
        assert not (set(M) & set(names)), "a macro name is already in numbers.tex"
