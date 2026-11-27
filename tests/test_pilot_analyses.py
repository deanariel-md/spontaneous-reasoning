"""Helpers of the specification, pilot and pattern analyses (scripts/run_specification.py, run_pilots_v3.py, run_patterns.py)."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spec():
    return load("run_specification")


@pytest.fixture(scope="module")
def pilots():
    return load("run_pilots_v3")


@pytest.fixture(scope="module")
def patterns():
    return load("run_patterns")


def test_agreement_within_and_between(spec):
    rng = np.random.default_rng(0)
    truth = np.arange(8.0) * 3
    Y = truth + rng.normal(0, 0.1, (6, 8))
    assert spec.c_within(Y) == pytest.approx(1.0)
    assert spec.c_between(Y, Y) == pytest.approx(1.0)
    assert spec.c_between(Y, -Y) == pytest.approx(0.0)
    # ties count one half
    assert spec.c_between(np.zeros((2, 3)), Y[:2, :3]) == pytest.approx(0.5)


def test_decomposition_main_effects(spec):
    rows = []
    for arm in ("a", "b"):
        for sc in ("s1", "s2", "s3"):
            for scale in ("p", "l", "e"):
                for cap in ("x", "y"):
                    for pw in ("f", "fb", "c"):
                        rows.append({"arm": arm, "scoring": sc, "scale": scale, "capability": cap,
                                     "panel_weighting": pw, "v": (1.0 if arm == "a" else -1.0)})
    df = pd.DataFrame(rows)
    d = spec.decompose(df, "v")
    assert d["main"]["arm"]["share"] == pytest.approx(1.0)
    assert d["main"]["scale"]["share"] == pytest.approx(0.0)
    assert d["rest_share"] == pytest.approx(0.0, abs=1e-12)
    df.loc[0, "v"] = np.nan
    assert spec.decompose(df, "v") == {"defined": False, "undefined_rows": 1}


def test_pilot_blocks_count_unresolved_as_constant(pilots):
    rng = np.random.default_rng(1)
    B = rng.normal(50, 5, (8, 6))
    P = B + rng.normal(0, 3, (8, 6))          # no true variation in gain: many blocks unresolved
    r = pilots.rules(B, P)
    assert len(r["one"]) == 8 and len(r["blocks_2"]) == 4 and len(r["blocks_4"]) == 2
    for b in r["blocks_2"] + r["blocks_4"]:
        if b["unresolved"]:
            assert np.allclose(b["shrunk"], 0.0)
        assert abs(sum(b["pilot"])) < 1e-9     # centred


def test_jack_stat_and_draw_slopes(patterns):
    v = np.array([1.0, 2.0, 3.0, 2.0])
    s = patterns.jack_stat(v)
    assert s["estimate"] == pytest.approx(2.0) and s["repeats"] == 1.0 and s["sign"] == "above zero"
    x = np.array([0.0, 1.0, 2.0, 3.0])
    G = np.vstack([2 * x + 5, 2 * x - 1])
    assert np.allclose(patterns.draw_slopes(G, x), [2.0, 2.0])


def test_pooled_gain_weights_tasks_or_families(patterns):
    # Two families: one task of 54 items, one of three tasks (162 items). Task-equal pooling gives the second
    # family three quarters of the weight; family-equal gives each half.
    one = np.ones((2, 3))
    x = {"n_items": {"a": 54, "b": 162},
         "arr": {("strict", "direct_answer", "a"): 10 * one, ("strict", "direct_answer", "b"): 30 * one,
                 ("strict", "zero_shot_cot", "a"): 20 * one, ("strict", "zero_shot_cot", "b"): 60 * one}}
    assert np.allclose(patterns.pooled_gain(x, "requested", "points", "strict"), 25.0)
    assert np.allclose(patterns.pooled_gain(x, "requested", "points", "strict", weighting="family"), 20.0)
    with pytest.raises(ValueError):
        patterns.pooled_gain(x, "requested", "points", "strict", weighting="model")


def test_symbolic_and_language_families_cover_the_set(patterns):
    from src import bbh, tasks_d2
    fams = {tasks_d2.bbh_family(t) for t in bbh.TASKS}
    assert set(patterns.SYMBOLIC) | set(patterns.LANGUAGE) == fams
    assert not set(patterns.SYMBOLIC) & set(patterns.LANGUAGE)
    assert len(fams) == 21
