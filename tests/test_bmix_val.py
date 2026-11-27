"""Fig. 2c's reversal counts on validation draws whose templates no discovery draw
used (src/bmix_val.py, scripts/run_bmix_val.py, paper/macros_bmix_val.py). The formulas on small synthetic cases, the
published discovery counts reproduced, the fresh draws recounted from the samplers, and the macros."""
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
from src import baseline_mixture as bm, bmix_val as bv  # noqa: E402
import macros_bmix_val as mbv  # noqa: E402

RES = os.path.join(ROOT, "results", "bmix_val", "bmix_val.json")
NUMBERS = os.path.join(ROOT, "paper", "numbers.tex")
TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
need_res = pytest.mark.skipif(not os.path.exists(RES), reason="run scripts/run_bmix_val.py first")


def _numbers():
    txt = open(NUMBERS, encoding="utf-8").read()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", txt)), re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", txt)


# ---------------------------------------------------------------------------- synthetic cases


def test_fresh_draws_keep_repeats_and_count_templates():
    fd = bv.fresh_draws(["a", "a", "b"], ["b", "c", "c", "d", "a"])
    assert fd["index"] == [1, 2, 3]                       # positions in validation order; c drawn twice counts twice
    assert fd["draws"] == 3 and fd["templates"] == 2
    assert fd["template_draws"] == {"c": 2, "d": 1} and fd["repeated_templates"] == ["c"]
    assert fd["overlap_draws"] == 2 and fd["overlap_templates"] == 2
    assert (fd["discovery_draws"], fd["discovery_templates"]) == (3, 2)
    assert (fd["validation_draws"], fd["validation_templates"]) == (5, 4)
    assert bv.fresh_draws([1, 2], [1, 2])["draws"] == 0   # every validation draw on a discovery template


def test_gains_are_means_over_the_selected_draws_with_repeats():
    # K = 3 draws, M = 2 models
    req = np.array([[10., 50.], [20., 40.], [30., 30.]])
    fd = np.array([[0., 40.], [10., 40.], [0., 20.]])
    cot = np.array([[60., 70.], [60., 70.], [90., 70.]])
    A = {"requested": req, "forced": fd, "cot": cot}
    g_req, g_fd = bv.gains(A, [1, 2, 2])
    assert np.allclose(g_req, [((60 - 20) + 2 * (90 - 30)) / 3, ((70 - 40) + 2 * (70 - 30)) / 3])
    assert np.allclose(g_fd, [((60 - 10) + 2 * (90 - 0)) / 3, ((70 - 40) + 2 * (70 - 20)) / 3])
    g_all = bv.gains(A)
    assert np.allclose(g_all[0], (cot - req).mean(0)) and np.allclose(g_all[1], (cot - fd).mean(0))
    with pytest.raises(ValueError):
        bv.gains(A, [])
    with pytest.raises(ValueError):
        bv.gains({**A, "cot": cot[:, :1]})


def test_recount_by_hand_and_ties_stay_in_the_denominator():
    # four models, one draw: gain over requested g_req, gain over forced g_fd; w chosen so the correction
    # restores g_req's order for model a and leaves a tie for c and d
    g_fd = np.array([40.0, 30.0, 20.0, 10.0])
    w = np.array([0.9, 0.0, 0.5, 0.0])
    g_req = np.array([4.0, 30.0, 10.0, 12.0])
    cot = np.full((1, 4), 100.0)
    A = {"cot": cot, "requested": cot - g_req, "forced": cot - g_fd}
    m = bv.recount(A, w)
    # by forced gain: pairs (a,b) (a,c) (a,d) and (c,d) reversed; 6 pairs, no tie
    assert m["observed_vs_forced"]["pairs"] == 6
    assert m["observed_vs_forced"]["reversed"] == 4 and m["observed_vs_forced"]["ties"] == 0
    assert m["observed_vs_forced"]["reversed_share"] == pytest.approx(4 / 6)
    # corrected gain (1 - w) g_fd = 4, 30, 10, 10: (c, d) tie, so neither; (a, d) is 4 < 10 and 4 < 12, kept
    assert np.allclose(m["corrected_gain"], [4.0, 30.0, 10.0, 10.0])
    assert m["observed_vs_corrected"]["reversed"] == 0 and m["observed_vs_corrected"]["ties"] == 1
    assert m["observed_vs_corrected"]["reversed_share"] == 0.0      # the tie stays in the denominator of 6
    ref = bm.mechanism(g_req, g_fd, w)
    for k in ("observed_vs_forced", "observed_vs_corrected"):
        assert m[k] == ref[k]
    assert m["models_with_nan"] == 0
    with pytest.raises(ValueError):
        bv.recount(A, w[:3])


def test_missing_cell_is_reported():
    cot = np.array([[50.0, np.nan], [60.0, 70.0]])
    A = {"cot": cot, "requested": np.zeros((2, 2)), "forced": np.zeros((2, 2))}
    assert bv.recount(A, [0.1, 0.2])["models_with_nan"] == 1
    assert bv.recount(A, [0.1, 0.2], [1])["models_with_nan"] == 0


def test_difference_and_pair_totals():
    d = {"observed_vs_forced": {"pairs": 10, "reversed": 3, "reversed_share": 0.3},
         "observed_vs_corrected": {"pairs": 10, "reversed": 2, "reversed_share": 0.2}}
    v = {"observed_vs_forced": {"pairs": 10, "reversed": 1, "reversed_share": 0.1},
         "observed_vs_corrected": {"pairs": 10, "reversed": 4, "reversed_share": 0.4}}
    x = bv.difference(v, d)
    assert x["observed_vs_forced"]["reversed"] == -2 and x["observed_vs_corrected"]["reversed"] == 2
    assert x["observed_vs_corrected"]["reversed_share"] == pytest.approx(0.2)
    v["observed_vs_forced"]["pairs"] = 9
    with pytest.raises(ValueError):
        bv.difference(v, d)


def test_macro_formatting_on_a_synthetic_result():
    def mech(o, c, n=10, t=0):
        return {"observed_vs_forced": {"pairs": n, "reversed": o, "reversed_share": o / n, "ties": t},
                "observed_vs_corrected": {"pairs": n, "reversed": c, "reversed_share": c / n, "ties": 0}}
    draws = {"draws": 3, "templates": 2, "overlap_draws": 21, "discovery_templates": 18, "validation_templates": 20}
    tasks = {}
    for t in ("gsm8k", "math", "bbh"):
        sc = {}
        for s in ("strict", "content"):
            d, f, a = mech(4, 2), mech(3, 2, t=1), mech(5, 1)
            sc[s] = {"discovery": d, "fresh_validation": f, "all_validation": a, "reproduction": {"reproduced": True},
                     "fresh_minus_discovery": bv.difference(f, d)}
        tasks[t] = {"draws": draws, "scores": sc}
    M = mbv.bmix_val_macros({"tasks": tasks})
    assert M["bmixValGsmRevObs"] == "3" and M["bmixValGsmRevObsPct"] == "30"
    assert M["bmixValGsmDiffObs"] == "$-$1" and M["bmixValGsmDiffCorrContent"] == "0"
    assert M["bmixValBbhTiesObs"] == "1" and M["bmixValMathAllRevCorrPctContent"] == "10"
    assert M["bmixValMaxChange"] == "1"
    assert len(M) == 3 * (5 + 2 * 13) + 1
    tasks["math"]["scores"]["content"]["reproduction"]["reproduced"] = False
    with pytest.raises(ValueError):
        mbv.bmix_val_macros({"tasks": tasks})
    assert mbv.pct0(0.004) == "0" and mbv.pct0(0.204) == "20" and mbv.pct0(0.226) == "23"


# ---------------------------------------------------------------------------- the published counts and the real draws


@need_res
def test_discovery_counts_reproduce_the_published_fig2c():
    R = json.load(open(RES, encoding="utf-8"))
    saved = json.load(open(os.path.join(ROOT, "results", "baseline_mixture", "baseline_mixture.json"), encoding="utf-8"))
    mac, _ = _numbers()
    for t, T in TAG.items():
        assert R["tasks"][t]["models"] == saved["tasks"][t]["models"]
        assert np.allclose(R["tasks"][t]["w_discovery"], saved["tasks"][t]["w_discovery"], rtol=0, atol=1e-12)
        for sc, suf in (("strict", ""), ("content", "Content")):
            d = R["tasks"][t]["scores"][sc]["discovery"]
            s = saved["tasks"][t]["mechanism"][sc]
            assert R["tasks"][t]["scores"][sc]["reproduction"]["reproduced"]
            for k, tag in (("observed_vs_forced", "Obs"), ("observed_vs_corrected", "Corr")):
                assert d[k]["reversed"] == s[k]["reversed"] and d[k]["pairs"] == s[k]["pairs"]
                assert mac[f"bmix{T}Rev{tag}{suf}"] == str(d[k]["reversed"])
                assert mac[f"bmix{T}Rev{tag}Pct{suf}"] == f"{100 * d[k]['reversed_share']:.0f}"
    # Fig. 2c's printed values (25 Nov): GSM8K 83 -> 48, MATH-500 84 -> 24, BIG-Bench Hard 68 -> 70, content 85 -> 47
    pub = {("gsm8k", "strict"): (83, 48), ("math", "strict"): (84, 24), ("bbh", "strict"): (68, 70),
           ("bbh", "content"): (85, 47)}
    for (t, sc), (o, c) in pub.items():
        d = R["tasks"][t]["scores"][sc]["discovery"]
        assert (d["observed_vs_forced"]["reversed"], d["observed_vs_corrected"]["reversed"]) == (o, c)


@need_res
def test_fresh_draws_from_the_saved_samplers():
    R = json.load(open(RES, encoding="utf-8"))
    want = {"gsm8k": (12, 10), "math": (17, 13), "bbh": (20, 17)}
    for t in TAG:
        s = json.load(open(os.path.join(ROOT, "manifests", f"sampler_{t}.json"), encoding="utf-8"))
        disc = [str(u) for u in s["discovery_config_ids"]]
        val = [str(u) for u in s["validation_config_ids"]]
        fd = bv.fresh_draws(disc, val)
        assert (fd["draws"], fd["templates"]) == want[t]
        assert fd["index"] == R["tasks"][t]["fresh_draw_positions"]
        assert not set(val[k] for k in fd["index"]) & set(disc)
        assert fd["draws"] + fd["overlap_draws"] == 24 == len(val)
        assert R["tasks"][t]["draws"]["draws"] == fd["draws"]


@need_res
def test_fresh_counts_recomputed_from_the_cached_strict_arrays():
    import robustness_inputs as ri
    R = json.load(open(RES, encoding="utf-8"))
    for t in TAG:
        npz = os.path.join(ri.CACHE, f"{t}_arrays.npz")
        if not os.path.exists(npz):
            pytest.skip("the strict arrays are not cached here")
        x = ri.load(t)
        w = np.asarray(R["tasks"][t]["w_discovery"], float)
        idx = R["tasks"][t]["fresh_draw_positions"]
        m = bm.mechanism(*[(x["validation"]["cot"][idx] - x["validation"][a][idx]).mean(0)
                           for a in ("requested", "forced")], w)
        f = R["tasks"][t]["scores"]["strict"]["fresh_validation"]
        for k in ("observed_vs_forced", "observed_vs_corrected"):
            assert m[k]["reversed"] == f[k]["reversed"] and m[k]["ties"] == f[k]["ties"]


@need_res
def test_macros_from_results_and_no_clash():
    R = mbv.bmix_val_results(ROOT)
    M = mbv.bmix_val_macros(R)
    assert len(M) == 94
    assert all(re.fullmatch(r"bmixVal[A-Za-z]+", k) for k in M)
    for t, T in TAG.items():
        f = R["tasks"][t]["scores"]["strict"]["fresh_validation"]
        assert M[f"bmixVal{T}RevObs"] == str(f["observed_vs_forced"]["reversed"])
        assert M[f"bmixVal{T}RevCorr"] == str(f["observed_vs_corrected"]["reversed"])
        assert M[f"bmixVal{T}Draws"] == str(R["tasks"][t]["draws"]["draws"])
    if os.path.exists(NUMBERS):
        _, names = _numbers()
        mk = open(os.path.join(ROOT, "paper", "make_numbers.py"), encoding="utf-8").read()
        if "macros_bmix_val" in mk:
            for k in M:                     # wired: each name defined exactly once
                assert names.count(k) == 1, f"{k} defined {names.count(k)} times in numbers.tex"
            return
        assert not (set(M) & set(names)), "a macro name is already in numbers.tex"
