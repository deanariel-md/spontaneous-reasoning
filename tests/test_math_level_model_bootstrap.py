"""Per-level model bootstrap of the MATH-500 level slopes: the function in
scripts/run_model_resampling.py, its table, the macros, and the saved results against each other."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from src import robustness as rb  # noqa: E402
import run_model_resampling as mr  # noqa: E402

RES = os.path.join(ROOT, "results", "model_resampling")


def fake_levels(seed=3, M=12, D=10):
    r = np.random.default_rng(seed)
    out = {}
    for k, lev in enumerate("12345"):
        base = r.normal(50, 15, (D, M)) + r.normal(0, 8, M)
        cot = base + (k - 2) * 0.3 * (base - 50) + r.normal(0, 3, (D, M))
        out[lev] = (base, cot)
    return out


def test_blocks_match_manual_bootstrap_with_same_stream():
    lev = fake_levels()
    got = mr.level_model_bootstrap(lev, 12, np.random.default_rng(7), n_boot=200)
    r = np.random.default_rng(7)
    assert [b["level"] for b in got] == [1, 2, 3, 4, 5]
    for b, (B, P) in zip(got, lev.values()):
        want = rb.bootstrap_statistic(mr.slope_fn(B, P), 12, None, 200, r)
        assert b["ci95"] == want["ci95"] and b["share_below_zero"] == want["share_below_zero"]
        assert b["n_undefined"] == want["n_undefined"] and b["n_boot"] == 200


def test_new_blocks_leave_earlier_draws_alone():
    """The generator is advanced only after the earlier statistics have drawn: drawing them with and without the
    level blocks afterwards gives identical earlier values."""
    lev = fake_levels()
    tr = lambda i: mr.level_trend(lev, i)  # noqa: E731
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    first = rb.bootstrap_statistic(tr, 12, None, 100, a)
    mr.level_model_bootstrap(lev, 12, a, n_boot=100)
    assert first == rb.bootstrap_statistic(tr, 12, None, 100, b)


def test_undefined_interval_is_kept_as_none():
    B = np.full((5, 6), 50.0)                     # no spread in the baseline: V_x is not positive
    out = mr.level_model_bootstrap({"1": (B, B + 1.0)}, 6, np.random.default_rng(0), n_boot=50)
    assert out[0]["ci95"] is None and out[0]["share_below_zero"] is None and out[0]["n_undefined"] == 50
    R = {"math_level_trend": {"level_slopes": [float("nan")], "level_model_bootstrap": out}}
    row = mr.math_levels_table(R)[0]
    assert row["model_lo"] is None and row["model_hi"] is None and row["model_share_below_zero"] is None


def test_table_rows():
    R = {"math_level_trend": {"level_slopes": [-0.5, 0.25],
                              "level_model_bootstrap": [{"level": 1, "ci95": [-0.7, -0.3], "share_below_zero": 1.0},
                                                        {"level": 2, "ci95": [0.0, 0.5], "share_below_zero": 0.02}]}}
    assert mr.math_levels_table(R) == [
        {"level": 1, "slope": -0.5, "model_lo": -0.7, "model_hi": -0.3, "model_share_below_zero": 1.0},
        {"level": 2, "slope": 0.25, "model_lo": 0.0, "model_hi": 0.5, "model_share_below_zero": 0.02}]


def test_summary_lines_name_the_figure_table():
    R = {"n_boot": 10, "seed": 1, "tasks": {}, "item_g_change": {
        "gsm8k": 0.0, "math": 0.0, "estimate": 0.0,
        "model_bootstrap": {"ci95": [0, 1], "share_below_zero": 0.0, "n_undefined": 0},
        "family_bootstrap": {"ci95": [0, 1], "n_undefined": 0}},
        "math_level_trend": {
            "estimate": 0.1, "level_slopes": [-0.5],
            "model_bootstrap": {"ci95": [0, 1], "n_undefined": 0}, "family_bootstrap": {"ci95": [0, 1], "n_undefined": 0},
            "level_model_bootstrap": [{"level": 1, "ci95": [-0.7, -0.3], "share_below_zero": 1.0, "n_undefined": 0}]}}
    text = "\n".join(mr.summary(R, []))
    assert "Figure table: `results/model_resampling/tables/math_levels.csv`" in text
    assert "| 1 | −0.500 | [−0.700, −0.300] | 1.000 | 0 |" in text


# ---- macros and saved results

def need_paper():
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    return pytest.importorskip("macros_added")


def test_level_macros_fabricated_and_absent():
    ma_ = need_paper()
    t = {"estimate": 0.47, "level_slopes": [-0.766, -0.306, 0.226, 0.415, 1.232]}
    R = {"tasks": {}, "math_level_trend": dict(t)}
    m = ma_.resampling_macros(R)
    assert "mresLevelOneModelLo" not in m and m["mresLevelOne"] == "$-$0.766"      # older file: no new macros
    t["level_model_bootstrap"] = [{"level": i, "ci95": [i - 1.5, i + 0.25], "share_below_zero": 0.5 / i} for i in range(1, 6)]
    t["level_model_bootstrap"][2]["ci95"] = None
    m = ma_.resampling_macros({"tasks": {}, "math_level_trend": t})
    assert m["mresLevelOneModelLo"] == "$-$0.500" and m["mresLevelOneModelHi"] == "1.250"
    assert m["mresLevelFiveModelLo"] == "3.500" and m["mresLevelTwoModelShareBelow"] == "0.250"
    assert m["mresLevelThreeModelLo"] == "--" and m["mresLevelThreeModelHi"] == "--"      # undefined, as it is


def test_saved_results_agree():
    path = os.path.join(RES, "model_resampling.json")
    if not os.path.exists(path):
        pytest.skip("results/model_resampling absent")
    R = json.load(open(path, encoding="utf-8"))
    t = R["math_level_trend"]
    assert "level_model_bootstrap" in t and len(t["level_model_bootstrap"]) == 5
    assert [b["level"] for b in t["level_model_bootstrap"]] == [1, 2, 3, 4, 5]
    for v, b in zip(t["level_slopes"], t["level_model_bootstrap"]):
        assert b["share_below_zero"] is not None and b["ci95"][0] <= b["ci95"][1]
    rows = mr.math_levels_table(R)
    csv = open(os.path.join(RES, "tables", "math_levels.csv"), encoding="utf-8").read().splitlines()
    assert csv[0] == "level,slope,model_lo,model_hi,model_share_below_zero" and len(csv) == 6
    for line, r in zip(csv[1:], rows):
        f = line.split(",")
        assert int(f[0]) == r["level"] and float(f[1]) == pytest.approx(r["slope"])
        assert float(f[2]) == pytest.approx(r["model_lo"]) and float(f[4]) == pytest.approx(r["model_share_below_zero"])
    for name in ("summary.md", "summary_clean.md"):
        text = open(os.path.join(RES, name), encoding="utf-8").read()
        assert "Figure table: `results/model_resampling/tables/math_levels.csv`" in text
    ma_ = need_paper()
    m = ma_.resampling_macros(R)
    b = t["level_model_bootstrap"][4]
    assert m["mresLevelFiveModelLo"] == ma_.num(b["ci95"][0], 3) and m["mresLevelOneModelShareBelow"] == "1.000"
