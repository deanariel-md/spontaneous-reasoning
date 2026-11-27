"""Tests of scripts/run_cap_reversal.py and paper/macros_cap_reversal.py (exploratory follow-up)."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_cap_reversal as crv  # noqa: E402
import macros_cap_reversal as mcr  # noqa: E402
from src import robustness as rb  # noqa: E402

RES = os.path.join(ROOT, "results", "cap_reversal", "cap_reversal.json")
N_MACROS = 6 * 100 + 3 * 2 * 7 + 1 + 3 + 3 + 4 + 2


# ---------------------------------------------------------------------------- unit tests on synthetic data

def _panel(K=24, M=14, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(20, 80, M)
    arms = {"requested": base[None] + rng.normal(0, 3, (K, M)),
            "forced": base[None] - 8 + rng.normal(0, 3, (K, M)),
            "cot": base[None] + 10 + 0.2 * base[None] + rng.normal(0, 3, (K, M))}
    return arms, rng.uniform(0.1, 0.7, M), np.log2(rng.uniform(2, 40, M)), rng.uniform(0, 3.5, M)


def test_slope_stats_definitions():
    arms, w, lp, yrs = _panel()
    s = crv.slope_stats(arms, w, lp, yrs)(np.arange(len(w)))
    R, F, C = arms["requested"], arms["forced"], arms["cot"]
    assert np.isclose(s[("own", "Req")], rb.cross_slope(R, C - R)[0])
    assert np.isclose(s[("own", "Fd")], rb.cross_slope(F, C - F)[0])
    assert np.isclose(s[("own", "Diff")], s[("own", "Req")] - s[("own", "Fd")])
    assert np.isclose(s[("own", "Corr")], rb.cross_slope(R, (1 - w)[None] * (C - F))[0])
    assert np.isclose(s[("own", "Share")], (s[("own", "Corr")] - s[("own", "Fd")]) / s[("own", "Diff")])
    assert s[("reqacc", "Req")] == s[("own", "Req")]
    # the difference on a shared axis is the slope of forced minus requested accuracy on it
    assert np.isclose(s[("reqacc", "Diff")], rb.cross_slope(R, F - R)[0])
    LP = np.tile(lp, (R.shape[0], 1))
    assert np.isclose(s[("size", "Diff")], rb.cross_slope(LP, F - R)[0])


def test_constant_axis_slope_is_ols_on_means():
    """On an axis constant over draws the cross-draw slope is the OLS slope of the cross-draw covariance; with no draw
    noise in the gain it is the OLS slope on model means."""
    K, M = 24, 10
    x = np.linspace(1, 5, M)
    G = np.tile(3 + 2 * x, (K, 1))
    assert np.isclose(rb.cross_slope(np.tile(x, (K, 1)), G)[0], 2.0)


def test_corrected_share_is_one_when_w_explains_everything():
    """If the requested gain is exactly (1 - w) x the forced gain, the correction reproduces the own-axis difference
    once Corr is measured on the same axis as Req."""
    arms, w, lp, yrs = _panel(seed=4)
    arms = dict(arms)
    arms["requested"] = arms["cot"] - (1 - w)[None] * (arms["cot"] - arms["forced"])
    s = crv.slope_stats(arms, w, lp, yrs)(np.arange(len(w)))
    assert np.isclose(s[("own", "Corr")], s[("own", "Req")]) and np.isclose(s[("own", "Share")], 1.0)


def test_ratio_and_summary_report_undefined():
    assert np.isnan(crv.ratio(1.0, 0.0)) and np.isnan(crv.ratio(float("nan"), 1.0))
    s = crv.summarise([1.0, float("nan"), -1.0, 2.0])
    assert s["n_undefined"] == 1 and s["n_boot"] == 4 and np.isclose(s["share_below_zero"], 1 / 3)
    assert crv.summarise([float("nan")])["ci95"] is None


def test_opposite_share():
    o = crv.opposite_share([-1, -1, 1, np.nan], [1, -1, -1, 1])
    assert o["n"] == 3 and np.isclose(o["req_neg_fd_pos"], 1 / 3) and np.isclose(o["req_pos_fd_neg"], 1 / 3)


def test_pearson_degenerate_is_nan():
    assert np.isnan(crv.pearson([1, 1, 1], [1, 2, 3]))
    assert np.isclose(crv.pearson([1, 2, 3], [2, 4, 7]), np.corrcoef([1, 2, 3], [2, 4, 7])[0, 1])


def test_resamples_repeat_for_the_same_label():
    a = rb.resample_columns(29, None, 20, crv.boot_rng("gsm8k", "model"))
    b = rb.resample_columns(29, None, 20, crv.boot_rng("gsm8k", "model"))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_formats():
    assert crv.fx(-0.4911, 3) == "$-$0.491" and crv.fx(float("nan"), 2) == "n/a"
    assert crv.pct(0.887) == "89" and crv.pct(-0.004) == "0" and crv.pct(-0.12) == "$-$12"


# ---------------------------------------------------------------------------- the results file

@pytest.fixture(scope="module")
def R():
    if not os.path.exists(RES):
        pytest.skip("results/cap_reversal/cap_reversal.json not written yet")
    return json.load(open(RES, encoding="utf-8"))


def test_post_hoc_and_panels(R):
    assert R["post_hoc"] is True
    assert {t: R["tasks"][t]["M"] for t in ("gsm8k", "math", "bbh")} == {"gsm8k": 29, "math": 29, "bbh": 28}
    assert all(R["tasks"][t]["K"] == 24 for t in R["tasks"])
    text = open(os.path.join(ROOT, "results", "cap_reversal", "summary.md"), encoding="utf-8").read()
    assert "exploratory" in text.lower()


def test_all_checks_pass(R):
    assert R["checks_failed"] == []
    names = [c["check"] for c in R["checks"]]
    assert sum("model_resampling" in n for n in names) == 6
    assert sum("req_axis_check" in n for n in names) == 18
    assert sum("reconcile pooled" in n for n in names) == 6 * 10
    assert sum("corrected share" in n for n in names) == 6


def test_reproduces_the_probe_values(R):
    g = R["tasks"]["gsm8k"]["scorings"]["strict"]["blocks"]
    assert f"{g['own']['Fd']['estimate']:.3f}" == "0.343"
    assert f"{g['own']['Req']['estimate']:.3f}" == "-0.320"
    assert f"{g['reqacc']['Fd']['estimate']:.3f}" == "0.171"
    assert f"{g['reqacc']['Diff']['estimate']:.3f}" == "-0.491"
    assert f"{g['own']['Diff']['estimate']:.3f}" == "-0.663"
    assert f"{g['own']['Share']['estimate']:.2f}" == "0.89"
    assert f"{R['item_split']['max_single']['abs_change']:.3f}" == "0.008"


def test_cells_consistent(R):
    for T in R["tasks"].values():
        for S in T["scorings"].values():
            B = S["blocks"]
            for b in ("own", "reqacc", "size", "year"):
                assert np.isclose(B[b]["Diff"]["estimate"], B[b]["Req"]["estimate"] - B[b]["Fd"]["estimate"])
                for c in (v for k, v in B[b].items() if k != "opposite"):
                    assert c["model_bootstrap"]["n_boot"] == crv.N_BOOT == c["family_bootstrap"]["n_boot"]
                    assert len(c["leave_one_family_out"]) == 5
                    lo, hi = c["model_bootstrap"]["ci95"]
                    assert lo <= hi
            assert B["own"]["Req"]["estimate"] == B["reqacc"]["Req"]["estimate"]
            assert 0 <= B["reqacc"]["opposite"]["model"]["req_neg_fd_pos"] <= 1


def test_per_model(R):
    for T in R["tasks"].values():
        for S in T["scorings"].values():
            for r in S["per_model"]:
                assert np.isclose(r["gain_requested_pp"], r["cot_pp"] - r["requested_pp"], atol=1e-9)
                assert np.isclose(r["gain_corrected_pp"], (1 - r["w_discovery"]) * r["gain_forced_pp"], atol=1e-9)
                assert np.isclose(r["log2_params"], np.log2(r["params_b"]))


def test_macros(R):
    m = mcr.cap_reversal_macros(R)
    assert len(m) == N_MACROS
    assert all(re.fullmatch(r"crv[A-Za-z]+", k) for k in m)
    c = R["tasks"]["math"]["scorings"]["content"]["blocks"]["reqacc"]["Diff"]
    assert m["crvMathReqAccDiffContent"] == crv.fx(c["estimate"], 3)
    assert m["crvMathReqAccDiffContentFamLo"] == crv.fx(c["family_bootstrap"]["ci95"][0], 3)
    assert m["crvMathReqAccDiffContentLofoHi"] == crv.fx(c["lofo_max"], 3)
    s = R["tasks"]["bbh"]["scorings"]["strict"]["blocks"]["size"]["Diff"]
    assert m["crvBbhSizeDiff"] == crv.fx(s["estimate"], 2)
    assert m["crvGsmOwnShare"] == crv.pct(R["tasks"]["gsm8k"]["scorings"]["strict"]["blocks"]["own"]["Share"]["estimate"])
    assert m["crvGsmWYear"] == crv.fx(R["tasks"]["gsm8k"]["w"]["Year"]["estimate"], 2)
    assert m["crvModelsBbh"] == "28" and m["crvNBoot"] == "10{,}000"
    assert mcr.cap_reversal_macros(None) == {}


def test_macro_names_are_new():
    p = os.path.join(ROOT, "paper", "numbers.tex")
    if not os.path.exists(RES) or not os.path.exists(p):
        pytest.skip("no results or numbers.tex")
    text = open(p, encoding="utf-8").read()
    if "\\crvNBoot" in text:
        pytest.skip("numbers.tex already wired with these macros")
    existing = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))
    assert not existing & set(mcr.cap_reversal_macros(json.load(open(RES, encoding="utf-8"))))


def test_outputs_exist(R):
    out = os.path.join(ROOT, "results", "cap_reversal")
    for f in ("summary.md", "summary_clean.md", "macros.md", "tables/slopes.csv", "tables/w_correlations.csv",
              "tables/per_model.csv"):
        assert os.path.exists(os.path.join(out, f)), f
