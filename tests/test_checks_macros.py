"""Correctness checks (scripts/run_output_checks.py, paper/macros_output_checks.py)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_output_checks as wc  # noqa: E402
import macros_output_checks as mw  # noqa: E402

RESULT = os.path.join(ROOT, "results", "output_checks", "output_checks.json")


def test_first_box_end_balanced_and_first():
    t = "work 2+2 \\boxed{\\frac{1}{2}} then \\boxed{3}"
    e = wc.first_box_end(t)
    assert t[:e].endswith("\\boxed{\\frac{1}{2}}")
    assert wc.first_box_end("no box here") is None
    assert wc.first_box_end("\\boxed{12") is None          # never closes
    assert wc.first_box_end(None) is None
    assert wc.first_box_end("\\boxed{}") == len("\\boxed{}")


def test_panel_equivalent_matches_ref():
    assert round(wc.panel_equivalent(12.746487243911865, 10, 29), 2) == 4.40
    assert wc.panel_equivalent(None, 10, 29) is None
    assert wc.panel_equivalent(9.0, 9, 28) == pytest.approx(9.0 * 9 / 28)


def test_reversal_count_ties_and_order():
    g1 = np.array([1.0, 2.0, 3.0])
    g2 = np.array([3.0, 2.0, 1.0])
    r = wc.reversal_count(g1, g2)
    assert r["pairs"] == 3 and r["reversed"] == 3
    r = wc.reversal_count(g1, np.array([1.0, 1.0, 3.0]))   # tie under g2: not reversed
    assert r["reversed"] == 0 and r["ties"] == 1


def test_drop_family_labels():
    assert wc._drop_fams(["a", "b", "c"], "b") == ["keep", "drop", "keep"]


def _synthetic():
    rows6 = []
    for t, M, n in (("gsm8k", 29, 10), ("bbh", 28, 9), ("math", 29, 10)):
        for d in ("one_draw", "pilot100"):
            for sc in ("strict", "content"):
                for k in ("requested_to_forced", "forced_to_requested"):
                    rows6.append({"task": t, "design": d, "score": sc, "direction": k, "n_chosen": n, "M": M,
                                  "per_chosen_model": 3.0, "ci95": [2.0, 4.0],
                                  "panel_equivalent": 3.0 * n / M, "panel_ci95": [2.0 * n / M, 4.0 * n / M]})
    mm = {f"{d}/{k}": [1.0, 2.0] for d in ("one_draw", "pilot100") for k in ("requested_to_forced", "forced_to_requested")}
    it4 = {"M": 28, "strict": {"pairs": 378, "reversed": 81, "share": 81 / 378, "models_cot_below_constrained": 16,
                               "mean_gain_pp": -5.68},
           "content_cot_and_requested": {"pairs": 378, "reversed": 76, "share": 76 / 378,
                                         "models_cot_below_constrained": 3, "mean_gain_pp": 18.8}}
    return {"item4": it4, "item6": {"rows": rows6, "min_max": mm}}


def test_macros_synthetic_names_and_values():
    m = mw.output_checks_macros(_synthetic())
    assert m["wckBbhConRev"] == "81" and m["wckBbhConPairs"] == "378" and m["wckBbhConRevContent"] == "76"
    assert m["wckBbhConBelow"] == "16" and m["wckBbhConMeanGain"] == "$-$5.7"
    assert m["wckMathPilotFdToReqPanel"] == "1.03"
    assert m["wckBbhFdToReqPanelContent"] == f"{3.0 * 9 / 28:.2f}"
    assert m["wckOwnFdToReqPanelMin"] == "1.00" and m["wckPilotReqToFdPanelMax"] == "2.00"
    assert all(k.isalpha() and k.startswith("wck") for k in m)


@pytest.mark.skipif(not os.path.exists(RESULT), reason="results/output_checks not built")
def test_results_reproduce_printed_numbers():
    R = json.load(open(RESULT, encoding="utf-8"))
    if "item4" in R:
        assert R["item4"]["strict"]["reversed"] == 81 and R["item4"]["strict"]["pairs"] == 378
        assert R["item4"]["strict"]["models_cot_below_constrained"] == 16
        assert round(R["item4"]["strict"]["mean_gain_pp"], 1) == -5.7
    if "item6" in R:
        c = R["item6"]["check_math_pilot_fd_to_req"]
        assert round(c["per_chosen"], 2) == 12.75 and round(c["panel"], 2) == 4.40
    for k, v in R.get("checks", {}).items():
        assert v, k
    m = R.get("macros") or {}
    assert all(k.startswith("wck") for k in m)
