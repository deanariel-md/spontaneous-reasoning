"""The blocks of scripts/run_math_primary.py on fabricated arrays."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_math_primary as mp  # noqa: E402


def draws(base, slope, K=24, sd=0.5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float)
    B = base[None, :] + rng.normal(0, sd, (K, len(base)))
    P = B + (20 + slope * (base - base.mean()))[None, :] + rng.normal(0, sd, (K, len(base)))
    return B, P


def test_verdicts():
    base = np.linspace(10, 70, 12)
    B, P = draws(base, 0.3)
    assert mp.primary_block(B, P, "fieller", 0.05)["verdict"] == "confirmed"
    B, P = draws(base, -0.2)                    # falling gain, chain-of-thought slope 0.8: off the ceiling
    r = mp.primary_block(B, P, "fieller", 0.05)
    assert r["verdict"] == "refuted" and r["off_ceiling"]
    B, P = draws(base, -0.7)                    # chain-of-thought slope 0.3: on the ceiling
    r = mp.primary_block(B, P, "fieller", 0.05)
    assert r["verdict"] == "inconclusive" and not r["off_ceiling"]
    assert r["ceiling"]["slope"] == pytest.approx(1 + r["gain"]["slope"])


def test_trend_rises_with_level_slopes():
    base = np.linspace(10, 70, 12)
    lv = [draws(base, s, seed=i) for i, s in enumerate((-0.6, -0.3, 0.0, 0.3, 0.6))]
    t = mp.trend(lv, 0.05)
    assert t["defined"] and t["estimate"] == pytest.approx(0.3, abs=0.05) and t["rising"]
    # the estimate is the least-squares slope of the level slopes on the level number
    assert t["estimate"] == pytest.approx(np.polyfit(np.arange(1, 6), t["level_slopes"], 1)[0])


def test_trend_undefined_when_a_level_has_no_capability_variance():
    B = np.full((24, 6), 50.0)
    t = mp.trend([(B, B + 10), draws(np.linspace(10, 70, 6), 0.1)], 0.05)
    assert not t["defined"] and t["estimate"] is None


def test_wrong_items_block():
    units, models = ["c1", "c2", "c3"], ["a", "b", "c", "d"]
    rows_f, rows_c = [], []
    for k, n_right in zip(models, (1, 2, 3, 0)):
        for u in units:
            for i in range(5):
                rows_f.append({"model_key": k, "config_id": u, "item_id": i, "strict_correct": i < n_right})
                rows_c.append({"model_key": k, "config_id": u, "item_id": i, "strict_correct": i < n_right + 1 + (u == "c2")})
    fd, cot = pd.DataFrame(rows_f), pd.DataFrame(rows_c)
    Bfd = np.array([[20.0, 40, 60, 0]] * 3) + np.arange(3)[:, None]
    w = mp.wrong_items_block(fd, cot, units, models, Bfd, "fieller", 0.05)
    assert w["items_per_model"] == {"a": 4, "b": 3, "c": 2, "d": 5}
    assert w["models_used"] == models and w["slope"] is not None
    # model a: items 1-4 are its forced-wrong items; chain of thought gets item 1 of them (items 1-2 in draw c2)
    assert np.array(w["gain_by_draw"])[:, 0] == pytest.approx([25.0, 50.0, 25.0])
    assert w["gain_model_means"]["d"] == pytest.approx(100 * (1 + 1 / 3) / 5)
