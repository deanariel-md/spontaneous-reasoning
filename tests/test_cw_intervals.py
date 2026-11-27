"""Tests of scripts/closed_w_intervals.py and paper/macros_cw_intervals.py."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "paper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import closed_w as cw  # noqa: E402
import closed_w_intervals as cwi  # noqa: E402
import macros_cw_intervals as mci  # noqa: E402
import macros_closed_ladder as mw  # noqa: E402

RES = ROOT / "results" / "frontier_panel" / "closed_w_intervals.json"
CWJ = ROOT / "results" / "frontier_panel" / "closed_w.json"
RAW = ROOT / "data" / "frontier_panel" / "closed_w" / "run_raw.jsonl"
NUMBERS = ROOT / "paper" / "numbers.tex"


def _synthetic(C=5, N=12, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    mats = {k: rng.integers(0, 2, size=(C, N)).astype(float) for k in ("R", "F", "C", "W")}
    if nan:
        mats["F"][1, 3] = np.nan
    return mats


@pytest.mark.parametrize("bbh", [False, True])
def test_draws_match_closed_w_boot_quantities(bbh):
    """The draws are those of closed_w.boot_quantities: every quantity it reports has the same interval here."""
    mats = _synthetic(nan=True)
    strata = [f"t{j // 3}" for j in range(12)] if bbh else ["all"] * 12
    wv = np.array([2, 0, 1, 1, 3], dtype=float)
    ref = cw.boot_quantities(mats, strata, {"discovery": wv}, 400, 0.95, 12345)["benchmark"]["discovery"]
    got = cwi.intervals(mats, strata, wv, 400, 0.95, 12345)["ci"]
    for k, v in ref.items():
        assert [got[k]["lo"], got[k]["hi"]] == v, k


def test_point_values_are_closed_w_model_quantities():
    mats = _synthetic(nan=True)
    wv = np.array([1, 1, 0, 2, 1], dtype=float)
    ref = cw.model_quantities(mats, wv)
    got = cwi.intervals(mats, ["all"] * 12, wv, 50, 0.95, 1)["point"]
    for k, v in ref.items():
        assert got[k] == v, k
    assert got["gain_diff"] == pytest.approx(got["requested"] - got["forced"], abs=1e-12)


def test_gain_difference_is_requested_minus_forced_per_draw():
    mats = _synthetic()
    wv = np.ones(5)
    per = cwi.draw_means(mats, ["all"] * 12, 300, 7)
    q = cwi.quantities(*(cw._wmean(per[x], wv) for x in ("R", "F", "C", "W")))
    np.testing.assert_allclose(q["gain_diff"], q["requested"] - q["forced"], atol=1e-12)
    np.testing.assert_allclose(q["corrected_gain"], (1 - q["w"]) * (q["cot"] - q["forced"]), atol=1e-12)


def test_resampling_stays_within_task():
    """BIG-Bench Hard: questions are drawn within their task, so each task keeps its number of questions in every
    draw. A matrix holding 1 on task a's questions keeps its mean at task a's share in every draw."""
    strata = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    M = np.tile(np.array([1.0] * 4 + [0.0] * 6), (3, 1))
    per = cwi.draw_means({"R": M}, strata, 500, 99)
    np.testing.assert_allclose(per["R"], 0.4, atol=1e-12)
    per_all = cwi.draw_means({"R": M}, ["all"] * 10, 500, 99)
    assert per_all["R"].std() > 0


def test_templates_held_fixed():
    """Templates are not resampled: a matrix constant along items gives the template means in every draw."""
    M = np.repeat(np.array([[0.1], [0.5], [0.9]]), 8, axis=1)
    per = cwi.draw_means({"R": M}, ["all"] * 8, 200, 3)
    np.testing.assert_allclose(per["R"], np.tile([0.1, 0.5, 0.9], (200, 1)), atol=1e-12)


def test_percentile_reports_non_finite_draws():
    v = np.array([0.0, 1.0, np.nan, 2.0, np.inf])
    p = cwi.percentile(v, 0.95)
    assert p["finite_draws"] == 3 and p["draws"] == 5
    assert p["lo"] == pytest.approx(np.quantile([0.0, 1.0, 2.0], 0.025))
    assert cwi.percentile(np.array([np.nan]), 0.95)["lo"] is None


def test_matrix_round_trip():
    mats = _synthetic(nan=True)
    back = cwi.mats_from_lists({k: cwi._mat_list(v) for k, v in mats.items()})
    for k in mats:
        np.testing.assert_array_equal(np.isnan(back[k]), np.isnan(mats[k]))
        np.testing.assert_array_equal(np.nan_to_num(back[k]), np.nan_to_num(mats[k]))


# ------------------------------------------------------------------------------------------- the results file

needs_results = pytest.mark.skipif(not (RES.exists() and CWJ.exists()), reason="results files absent")


@pytest.fixture(scope="module")
def res():
    return json.loads(RES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cwj():
    return json.loads(CWJ.read_text(encoding="utf-8"))


@needs_results
def test_cells_and_design(res):
    assert [(c["task"], c["score"]) for c in res["cells"]] == list(cwi.CELLS)
    assert res["draws"] == 2000 and res["level"] == 0.95 and res["basis"] == "discovery"
    for c in res["cells"]:
        assert c["items"] == 100
        assert c["strata"] == (25 if c["task"] == "bbh" else 1)
        assert all(v["finite_draws"] == 2000 for v in c["ci"].values())


@needs_results
def test_points_reproduce_closed_w_exactly(res, cwj):
    """Point values equal closed_w.json's gains rows (discovery basis) bit for bit, and so do the intervals that
    closed_w.json already holds."""
    assert res["all_equal_to_closed_w"] is True
    for c in res["cells"]:
        g = cwi.stored_gain(cwj, c["task"], c["score"])
        for k in ("requested", "forced", "cot", "w", "cot_minus_requested", "cot_minus_forced", "corrected_gain",
                  "residual"):
            assert c["point"][k] == g[k], (c["task"], c["score"], k)
        assert c["point"]["gain_diff"] == g["cot_minus_forced"] - g["cot_minus_requested"]
        for k in cwi.STORED:
            assert [c["ci"][k]["lo"], c["ci"][k]["hi"]] == g["ci"][k], (c["task"], c["score"], k)


@needs_results
def test_intervals_recompute_from_stored_matrices(res):
    for c in res["cells"]:
        mats = cwi.mats_from_lists(c["mats"])
        wv = np.array([c["weights"][t] for t in c["template_ids"]], dtype=float)
        got = cwi.intervals(mats, cwi.strata_of(c["task"], c["item_ids"]), wv, res["draws"], res["level"],
                            cw.cwp.label_seed(c["seed_label"]))
        assert got["point"] == c["point"]
        for k, v in got["ci"].items():
            assert v == c["ci"][k], (c["task"], c["score"], k)


@needs_results
def test_point_inside_interval(res):
    for c in res["cells"]:
        for k in ("cot_minus_requested", "cot_minus_forced", "gain_diff", "corrected_gain"):
            assert c["ci"][k]["lo"] <= c["point"][k] <= c["ci"][k]["hi"], (c["task"], c["score"], k)


@pytest.mark.skipif(not (RAW.exists() and CWJ.exists() and RES.exists()), reason="row-level replies absent")
def test_rebuild_from_rows_equals_results(res):
    """The matrices rebuilt from the run's row-level replies are those stored, and every value is the same."""
    new = cwi.build()
    assert new["all_equal_to_closed_w"] is True
    for a, b in zip(new["cells"], res["cells"]):
        assert a["mats"] == b["mats"]
        assert a["point"] == b["point"] and a["ci"] == b["ci"]


# ------------------------------------------------------------------------------------------------- macros

TS = ("Gsm", "Math", "Bbh", "BbhContent")
EXPECTED = ({f"cw{q}{T}{s}" for q in ("ReqGain", "FdGain", "Corr") for T in TS for s in ("Lo", "Hi")}
            | {f"cwGainDiff{T}{s}" for T in TS for s in ("", "Lo", "Hi")})


@needs_results
def test_macro_names_values(res):
    m = mci.cw_intervals_macros(res)
    assert set(m) == EXPECTED and len(m) == 36
    for k, v in m.items():
        assert re.fullmatch(r"cw[A-Za-z]+", k)
        assert re.fullmatch(r"(\$-\$)?\d+\.\d", v), (k, v)
    c = {(x["task"], x["score"]): x for x in res["cells"]}
    assert m["cwGainDiffMath"] == mci.f1(100 * (c[("math", "strict")]["point"]["requested"]
                                                - c[("math", "strict")]["point"]["forced"]))
    assert m["cwCorrBbhContentHi"] == f"{100 * c[('bbh', 'content')]['ci']['corrected_gain']['hi']:.1f}"


@needs_results
def test_macros_are_new():
    """No name is defined in numbers.tex or by the closed-model module."""
    tex = NUMBERS.read_text(encoding="utf-8") if NUMBERS.exists() else ""
    old = set(mw.closed_ladder_macros(mw.closed_ladder_results(str(ROOT))))
    # once wired into make_numbers.py, each name is defined exactly once
    for k in EXPECTED:
        assert tex.count(f"\\newcommand{{\\{k}}}") <= 1, k
        assert k not in old, k


@needs_results
def test_points_match_existing_gain_macros(res):
    """The gains whose intervals are given here print the same point values as the existing cw macros."""
    old = mw.closed_ladder_macros(mw.closed_ladder_results(str(ROOT)))
    for (task, score), x in {(c["task"], c["score"]): c for c in res["cells"]}.items():
        T = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}[task] + ("Content" if score == "content" else "")
        assert old[f"cwReqGain{T}"] == mci.f1(100 * x["point"]["cot_minus_requested"])
        assert old[f"cwFdGain{T}"] == mci.f1(100 * x["point"]["cot_minus_forced"])
        assert old[f"cwCorr{T}"] == mci.f1(100 * x["point"]["corrected_gain"])


def test_macros_refuse_unreproduced_results():
    assert mci.cw_intervals_macros(None) == {}
    with pytest.raises(ValueError):
        mci.cw_intervals_macros({"all_equal_to_closed_w": False, "cells": []})
