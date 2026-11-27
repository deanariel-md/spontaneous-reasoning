"""paper/macros_added.decision_cross_macros, the wrong-protocol cost:
fabricated results with known values, the real decision.json against the summary's own numbers."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
# paper/ is not in the repository, so a clean checkout skips this file
ma = pytest.importorskip("macros_added")


def entry(choice, ev, est, lo, hi):
    mk = lambda e, a, b: {"extra_regret": {"estimate": e, "ci95": [a, b], "K": 10}}  # noqa: E731
    return {"choice": choice, "evaluation_setting": ev, "evaluation": {
        "independent": {"rules": {"own_1": mk(est, lo, hi), "own_2": mk(9.0, 8.0, 10.0)}},
        "all_distinct": {"rules": {"own_1": mk(7.0, 6.0, 8.0)}}}}


def test_decision_cross_macros_values_and_selection():
    R = {"dry_run": False, "cross_protocol": [
        entry("gsm8k_req", "gsm8k_fd", 2.518, 2.357, 2.680),
        entry("gsm8k_req", "gsm8k_con", 0.106, 0.047, 0.165),
        entry("math_req", "math_fd", -0.004, -0.2, 0.15)]}
    m = ma.decision_cross_macros(R)
    assert m == {"decXGsmOwnOne": "2.52", "decXGsmOwnOneLo": "2.36", "decXGsmOwnOneHi": "2.68",
                 "decXMathOwnOne": "0.00",
                 "decXMathOwnOneLo": "$-$0.20", "decXMathOwnOneHi": "0.15"}
    assert all(k.isalpha() for k in m)


def test_decision_cross_macros_dry_run_and_missing_interval():
    assert ma.decision_cross_macros({"dry_run": True, "cross_protocol": [entry("gsm8k_req", "gsm8k_fd", 1, 0, 2)]}) == {}
    e = entry("bbh_req", "bbh_fd", 2.4, 2.2, 2.6)
    e["evaluation"]["independent"]["rules"]["own_1"]["extra_regret"]["ci95"] = None
    m = ma.decision_cross_macros({"cross_protocol": [e]})
    assert m == {"decXBbhOwnOne": "2.40", "decXBbhOwnOneLo": "--", "decXBbhOwnOneHi": "--"}


def test_decision_cross_macros_real_file_matches_summary():
    path = os.path.join(ROOT, "results", "decision", "decision.json")
    if not os.path.exists(path):
        pytest.skip("no decision.json")
    m = ma.decision_cross_macros(json.load(open(path, encoding="utf-8")))
    tasks = [("Gsm", "gsm8k"), ("Bbh", "bbh"), ("Math", "math")]
    if "decXHqOwnOne" in m:                     # HotpotQA's pair
        tasks.append(("Hq", "hotpot"))
    assert len(m) == 3 * len(tasks)
    lines = open(os.path.join(ROOT, "results", "decision", "summary.md"), encoding="utf-8").read().splitlines()
    for t, src in tasks:
        row = next(l for l in lines if l.startswith(f"| {src}_req → {src}_fd | own gain, 1 draw |"))
        cell = row.split("|")[5].strip()                   # "2.52 [2.36, 2.68]"
        est, rest = cell.split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        fix = lambda s: s.replace("\u2212", "$-$")  # noqa: E731
        assert (m[f"decX{t}OwnOne"], m[f"decX{t}OwnOneLo"], m[f"decX{t}OwnOneHi"]) == (fix(est), fix(lo), fix(hi))
