"""Checks on existing data: the matched cap of the native-reasoning run, the MMLU-subject block of the
reliability map and the pilot wins by setting group."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_pilots_v3 as pv3  # noqa: E402
import run_reliability_map as rmap  # noqa: E402
import run_thinking as rth  # noqa: E402


def test_matched_cap_counts_longer_outputs_as_wrong():
    rows = pd.DataFrame({"model_key": ["a"] * 4, "strict_correct": [True, True, False, True],
                         "n_output_tokens": [100, 3072, 50, 3073]})
    out = rth.capped(lambda ids: rows, 3072)(None)
    assert out["strict_correct"].tolist() == [True, True, False, False]
    assert rth.cot_cap() == 3072


def _rec(name, R, C, Cref):
    return {"setting": name, "group": "dove_tasks",
            "point": {q: {"R": R, "C": C, "departure": C - Cref} for q in ("benefit", "accuracy")}}


def test_mmlu_subject_block():
    recs = [_rec("dove:mmlu.anatomy", 0.5, 0.70, 0.66), _rec("dove:mmlu.virology", 0.2, 0.55, 0.60),
            _rec("dove:mmlu_pro.math", 0.9, 0.9, 0.85), _rec("dove:hellaswag", -0.1, 0.5, 0.47)]
    b = rmap.mmlu_subject_blocks(recs, [0.8, 0.9])
    assert b["MmluSubj"]["n"] == 2 and b["Bench"]["n"] == 4
    assert b["MmluSubj"]["settings"] == ["dove:mmlu.anatomy", "dove:mmlu.virology"]
    q = b["MmluSubj"]["quantities"]["benefit"]
    assert q["departure_mean_abs"] == pytest.approx(0.045)
    assert q["K"][0.8]["undefined"] == 0 and b["Bench"]["quantities"]["benefit"]["K"][0.8]["undefined"] == 1


def test_pilot_wins_split_by_group():
    df = pd.DataFrame({"group": ["dove_tasks"] * 3 + ["bbh_tasks"] * 2, "R": [0.6, 0.3, 0.2, 0.9, 0.8],
                       "risk_constant": [5.0, 4.0, -1.0, 6.0, 7.0], "one_mean": [1.0, 5.0, 0.0, 1.0, 1.0],
                       "one_first": [1.0, 3.0, 0.0, 1.0, 9.0], "one_share_draws_beat": [1.0, 0.0, 0.0, 1.0, 0.5],
                       "one_mean_share": [0.2] * 5, "one_mean_over_expected": [1.0] * 5})
    for k in pv3.BLOCKS:
        for c in (f"pilot_{k}_mean", f"shrunk_{k}_mean", f"unresolved_{k}"):
            df[c] = 0.0
    bud = pd.DataFrame({"group": [], "passes": [], "share_meets": [], "share_unresolved": []})
    out = pv3.by_group(df, bud)
    hi, lo = out["R_at_least_half"], out["R_below_half"]
    assert hi["by_group"]["dove_tasks"] == {"settings": 1, "positive_constant": 1, "one_mean_beats": 1, "one_first_beats": 1}
    assert hi["by_group"]["bbh_tasks"]["one_first_beats"] == 1
    assert lo["by_group"]["dove_tasks"] == {"settings": 2, "positive_constant": 1, "one_mean_beats": 0, "one_first_beats": 1}
    assert sum(g["one_mean_beats"] for g in hi["by_group"].values()) == hi["one_mean_beats"]
