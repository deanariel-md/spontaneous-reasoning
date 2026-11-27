"""Tests of scripts/run_cap_shared_axis.py and paper/macros_cap_shared_axis.py."""
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

import run_cap_shared_axis as csa  # noqa: E402
import macros_cap_shared_axis as mca  # noqa: E402
from src import robustness as rb  # noqa: E402

RES = os.path.join(ROOT, "results", "cap_shared_axis", "cap_shared_axis.json")


# ---------------------------------------------------------------------------- unit tests on synthetic data

def _arms(K=24, M=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(20, 80, M)
    arms = {"requested": base[None] + rng.normal(0, 3, (K, M)),
            "forced": base[None] - 5 + rng.normal(0, 3, (K, M)),
            "cot": base[None] + 10 + 0.2 * base[None] + rng.normal(0, 3, (K, M))}
    return arms, rng.uniform(90, 150, M)


def test_diff_is_slope_of_forced_minus_requested():
    arms, eci = _arms()
    for ax in ("eci", "forced", "cot"):
        X = csa.axis_array(ax, arms, eci)
        d = rb.cross_slope(X, csa.gain_array("requested", arms))[0] - rb.cross_slope(X, csa.gain_array("forced", arms))[0]
        assert np.isclose(rb.cross_slope(X, csa.gain_array("diff", arms))[0], d, atol=1e-12)
        assert np.isclose(rb.cross_slope(X, arms["forced"] - arms["requested"])[0], d, atol=1e-12)


def test_index_axis_is_constant_over_draws():
    arms, eci = _arms()
    X = csa.axis_array("eci", arms, eci)
    assert X.shape == arms["cot"].shape and np.all(X == np.asarray(eci)[None])


def test_cross_draw_slope_removes_shared_draw_noise():
    """x = forced accuracy and y = CoT - forced share the forced arm's draw noise; the cross-draw slope does not carry
    it, the slope on same-draw model means does."""
    rng = np.random.default_rng(3)
    K, M = 24, 4000
    f_true = rng.uniform(20, 80, M)
    F = f_true[None] + rng.normal(0, 30, (K, M))
    C = 30 + 0.5 * f_true[None] + rng.normal(0, 30, (K, M))
    s = rb.cross_slope(F, C - F)[0]
    naive = np.polyfit(F.mean(0), (C - F).mean(0), 1)[0]     # about -0.56 in expectation
    assert abs(s - (-0.5)) < 0.02
    assert naive < -0.54


def test_split_points_use_disjoint_halves():
    K, M = 6, 3
    X = np.arange(K * M, dtype=float).reshape(K, M)
    G = 100 + X
    x, y = csa.split_points(X, G, constant_x=False)
    a, b = csa.halves(K)
    assert set(a) & set(b) == set() and len(a) + len(b) == K
    assert np.allclose(x[:M], X[a].mean(0)) and np.allclose(y[:M], G[b].mean(0))
    assert np.allclose(x[M:], X[b].mean(0)) and np.allclose(y[M:], G[a].mean(0))
    x2, y2 = csa.split_points(np.tile([1.0, 2.0, 3.0], (K, 1)), G, constant_x=True)
    assert np.allclose(x2, [1, 2, 3]) and np.allclose(y2, G.mean(0))


@pytest.mark.parametrize("c2,c1,lo,hi,expect", [
    (-1.0, 100.0, 0, 100, "peak"),            # vertex 50 inside
    (1.0, -100.0, 0, 100, "trough"),          # vertex 50 inside, convex
    (-1.0, 300.0, 0, 100, "rises throughout"),   # vertex 150 above the range
    (-1.0, -100.0, 0, 100, "falls throughout"),  # vertex -50 below the range
    (1.0, 100.0, 0, 100, "rises throughout"),    # convex, vertex -50 below the range
])
def test_shape(c2, c1, lo, hi, expect):
    q = {"c2": c2, "c1": c1, "vertex_baseline": -c1 / (2 * c2)}
    assert csa.shape(q, lo, hi) == expect


def test_quad_recovers_a_known_peak():
    K, M = 24, 15
    x = np.linspace(100, 150, M)
    X = np.tile(x, (K, 1))
    G = np.tile(10 - (x - 130) ** 2 / 50, (K, 1))
    q = csa.quad(X, G, constant_x=True)
    assert q["shape"] == "peak" and np.isclose(q["peak"], 130.0)
    q2 = csa.quad(X, G, constant_x=False)
    assert q2["shape"] == "peak" and np.isclose(q2["peak"], 130.0) and q2["n_points"] == 2 * M
    G3 = np.tile(10 - (x - 200) ** 2 / 50, (K, 1))
    q3 = csa.quad(X, G3, constant_x=True)
    assert q3["shape"] == "rises throughout" and q3["peak"] is None


def test_resamples_repeat_for_the_same_label():
    a = rb.resample_columns(29, None, 50, csa.boot_rng("gsm8k", "model"))
    b = rb.resample_columns(29, None, 50, csa.boot_rng("gsm8k", "model"))
    c = rb.resample_columns(29, None, 50, csa.boot_rng("gsm8k", "family"))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert not all(np.array_equal(u, v) for u, v in zip(a, c))


def test_fx_formats_like_make_numbers():
    assert csa.fx(-0.3204, 3) == "$-$0.320" and csa.fx(0.12351, 3) == "0.124" and csa.fx(float("nan"), 3) == "n/a"


# ---------------------------------------------------------------------------- the results file

@pytest.fixture(scope="module")
def R():
    if not os.path.exists(RES):
        pytest.skip("results/cap_shared_axis/cap_shared_axis.json not written yet")
    return json.load(open(RES, encoding="utf-8"))


def test_panels_and_draws(R):
    assert {t: R["tasks"][t]["M"] for t in ("gsm8k", "math", "bbh")} == {"gsm8k": 29, "math": 29, "bbh": 28}
    assert all(R["tasks"][t]["K"] == 24 for t in R["tasks"])
    assert len(R["bbh_missing"]) == 1 and R["eci_missing"] == [] and R["eci_checked"] == 29


def test_checks_reproduce_earlier_results(R):
    names = [c["check"] for c in R["checks"]]
    assert sum("model_resampling" in n for n in names) == 3
    assert sum("on the index" in n for n in names) == 12      # 3 tasks x 2 scorings x 2 gains
    for c in R["checks"]:
        assert c["abs_diff"] < 1e-9, c


def test_strict_gsm8k_forced_slope_is_the_earlier_value(R):
    c = R["tasks"]["gsm8k"]["scorings"]["strict"]["axes"]["forced"]["gains"]["forced"]
    assert f"{c['slope']:.3f}" == "0.343"


def test_diff_point_estimate_is_req_minus_fd(R):
    for T in R["tasks"].values():
        for S in T["scorings"].values():
            for A in S["axes"].values():
                g = A["gains"]
                assert np.isclose(g["diff"]["slope"], g["requested"]["slope"] - g["forced"]["slope"], atol=1e-10)
                for k in ("requested", "forced", "diff"):
                    lo, hi = g[k]["model_bootstrap"]["ci95"]
                    assert lo <= hi
                    assert len(g[k]["leave_one_family_out"]) == 5
                    assert g[k]["lofo_min"] <= g[k]["lofo_max"]


def test_per_model_gains_match_arms(R):
    for T in R["tasks"].values():
        for S in T["scorings"].values():
            for r in S["per_model"]:
                assert np.isclose(r["gain_requested_pp"], r["cot_pp"] - r["requested_pp"], atol=1e-9)
                assert np.isclose(r["gain_forced_pp"], r["cot_pp"] - r["forced_pp"], atol=1e-9)


def test_macros(R):
    m = mca.cap_shared_axis_macros(R)
    assert len(m) == 378 + 108 + 8
    assert all(re.fullmatch(r"cap(Slope|Peak)[A-Za-z]+", k) for k in m)
    c = R["tasks"]["math"]["scorings"]["content"]["axes"]["cot"]["gains"]["diff"]
    assert m["capSlopeMathCotDiffContent"] == csa.fx(c["slope"], 3)
    assert m["capSlopeMathCotDiffContentFamLo"] == csa.fx(c["family_bootstrap"]["ci95"][0], 3)
    assert m["capSlopeMathCotDiffContentLofoHi"] == csa.fx(c["lofo_max"], 3)
    for k, v in m.items():
        if k.startswith("capPeak") and not k.endswith(("Shape", "Vertex")):
            assert v == "none" or re.fullmatch(r"(\$-\$)?\d+\.\d", v)
            assert (v == "none") == (m[k + "Shape"] != "peak")
    assert m["capSlopeModelsBbh"] == "28" and m["capSlopeEciMissing"] == "0"


def test_count_macros(R):
    """The counts reported by hand (18 of 18 shallower, 5 excluding zero)."""
    m = mca.cap_count_macros(R)
    assert set(m) == set(mca.COUNT_NAMES)
    A = R["answer"]
    assert m["capCells"] == str(len(A)) == "18"
    assert m["capShallowerCells"] == str(sum(a["diff"] < 0 for a in A))
    assert m["capDiffExclCells"] == str(sum(a["diff_excl_model"] and a["diff_excl_family"] for a in A))
    assert not set(m) & set(mca.cap_shared_axis_macros(R))


def test_macro_names_are_new():
    p = os.path.join(ROOT, "paper", "numbers.tex")
    if not os.path.exists(RES) or not os.path.exists(p):
        pytest.skip("no results or numbers.tex")
    text = open(p, encoding="utf-8").read()
    if "capSlope" in text or "capPeak" in text:
        pytest.skip("numbers.tex already wired with these macros")
    existing = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))
    assert not existing & set(mca.cap_shared_axis_macros(json.load(open(RES, encoding="utf-8"))))


def test_outputs_exist(R):
    out = os.path.join(ROOT, "results", "cap_shared_axis")
    for f in ("summary.md", "summary_clean.md", "tables/slopes.csv", "tables/per_model.csv"):
        assert os.path.exists(os.path.join(out, f)), f
