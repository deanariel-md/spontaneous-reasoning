"""The blocks of scripts/run_math_prediction.py on fabricated arrays."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_math_prediction as mp  # noqa: E402


def draws(base, gain, K=24, sd=0.5, seed=0):
    rng = np.random.default_rng(seed)
    B = np.asarray(base, float)[None, :] + rng.normal(0, sd, (K, len(base)))
    P = B + np.asarray(gain, float)[None, :] + rng.normal(0, sd, (K, len(base)))
    return B, P


def test_curve_vertex_and_limb():
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80], float)
    fit = mp.gsm8k_curve(x, -(x - 40) ** 2 / 20 + 30)
    assert fit["concave"] and fit["vertex"] == pytest.approx(40)
    assert not mp.gsm8k_curve(x, (x - 40) ** 2)["concave"]


def test_limb_block_reads_the_math_slope_among_readable_limb_models():
    panel = [f"m{i}" for i in range(8)]
    gb = [10, 20, 30, 40, 50, 60, 70, 80]
    gB, gP = draws(gb, [-(b - 40) ** 2 / 20 + 30 for b in gb])        # GSM8K: concave, vertex at 40
    mb = [5, 10, 15, 20, 25, 30, 35, 40]
    mB, mP = draws(mb, [0.8 * b for b in mb], seed=1)                  # MATH: rising everywhere
    clean = {k: k != "m7" for k in panel}
    r, rows = mp.limb_block(gB, gP, mB, mP, panel, panel, clean, 1319, 398, "fieller", 0.05)
    assert r["limb_models"] == ["m4", "m5", "m6", "m7"]
    assert r["limb_readable_on_math"] == ["m4", "m5", "m6"] and r["limb_not_readable_on_math"] == ["m7"]
    assert r["math_slopes"]["points"]["slope"] > 0 and r["holds"]["points"]
    assert r["gsm8k_slopes_same_models"]["points"]["slope"] < 0
    row = next(x for x in rows if x["model_key"] == "m5")
    assert row["on_descending_limb"] and row["shift_baseline"] == pytest.approx(row["math_baseline"] - row["gsm8k_baseline"])
    assert row["residual_from_gsm8k_curve"] == pytest.approx(row["math_gain"] - row["gsm8k_curve_at_math_baseline"])


def test_too_few_readable_limb_models_give_no_slope():
    panel = [f"m{i}" for i in range(6)]
    gb = [10, 20, 30, 40, 50, 60]
    gB, gP = draws(gb, [-(b - 45) ** 2 / 20 + 30 for b in gb])
    r, _ = mp.limb_block(gB, gP, gB, gP, panel, panel, {k: True for k in panel}, 100, 100, "fieller", 0.05)
    assert r["limb_models"] == ["m4", "m5"] and r["math_slopes"] is None and r["holds"] is None
    assert "needs 3" in r["note"]


def test_disagreement_interval():
    Breq = np.array([[50.0, 40], [52, 41], [51, 39], [53, 40]])
    Bfd = np.array([[40.0, 40], [41, 40.5], [40, 39.5], [42, 40]])
    d = mp.disagreement(Breq, Bfd, ["a", "b"], ["a", "b", "c"])
    assert d["a"]["holds"] and d["a"]["estimate"] == pytest.approx(10.75)
    assert not d["b"]["holds"] and d["c"] == {"in_panel": False}
