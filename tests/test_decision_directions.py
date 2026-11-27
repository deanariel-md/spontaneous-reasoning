"""Both directions of the wrong-baseline decision cost (src/decision_directions.py,
scripts/run_decision_directions.py), and the reproduction of the printed numbers from the results file."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import decision as dc, decision_directions as dd, robustness as rb  # noqa: E402

RESULT = os.path.join(ROOT, "results", "decision_directions", "decision_directions.json")


def _setup(seed=0, M=9, K=6, S=5):
    rng = np.random.default_rng(seed)
    G = {"requested": rng.normal(10, 5, (K, M)), "forced": rng.normal(20, 8, (K, M))}
    pred = {a: G[a].mean(0) + rng.normal(0, 2, (S, M)) for a in G}
    n = dc.third_size(M)
    W = {a: np.array([dc.choose(p, n)[0] for p in pred[a]]) for a in G}
    return G, pred, W, n


def test_both_directions_equal_the_paired_difference():
    G, pred, W, n = _setup()
    out = dd.both_directions(W, G, n)
    a = dc.paired_difference(W["requested"], W["forced"], G["forced"], n)
    b = dc.paired_difference(W["forced"], W["requested"], G["requested"], n)
    assert np.isclose(out["requested_to_forced"]["estimate"], a["estimate"])
    assert np.isclose(out["forced_to_requested"]["estimate"], b["estimate"])
    x = out["forced_to_requested"]
    assert np.isclose(x["estimate"], x["regret_mismatched"] - x["regret_matched"])   # the best third cancels
    same = dd.both_directions({"requested": W["requested"], "forced": W["requested"]}, G, n)
    assert same["requested_to_forced"]["estimate"] == 0.0


def test_leave_one_family_out_rechooses_from_saved_predictions():
    G, pred, W, n = _setup(M=9)
    fams = ["A"] * 4 + ["B"] * 3 + ["C"] * 2
    out = dd.leave_one_family_out(pred, G, fams)
    assert set(out["by_family"]) == {"A", "B", "C"}
    keep = np.array([j for j, f in enumerate(fams) if f != "B"])
    n2 = dc.third_size(len(keep))
    W2 = {a: np.array([dc.choose(p[keep], n2)[0] for p in pred[a]]) for a in G}
    want = dc.paired_difference(W2["forced"], W2["requested"], G["requested"][:, keep], n2)["estimate"]
    assert np.isclose(out["by_family"]["B"]["forced_to_requested"], want)
    assert out["by_family"]["B"]["n_chosen"] == n2 and out["by_family"]["B"]["models_left"] == 6
    v = [x["forced_to_requested"] for x in out["by_family"].values()]
    assert out["range"]["forced_to_requested"] == [min(v), max(v)]


def test_reversal_share_matches_pair_reversals_and_resampling():
    rng = np.random.default_rng(3)
    g1, g2 = rng.normal(0, 1, 12), rng.normal(0, 1, 12)
    g2[3] = g2[4]                                         # a tie under one gain: neither reversed
    fams = list("AAABBBBCCDDD")
    ref = rb.pair_reversals(g1, g2)
    out = dd.reversal_uncertainty(g1, g2, fams, 200, np.random.default_rng(1))
    assert out["reversed"] == ref["reversed"] and np.isclose(out["share"], ref["reversed_share"])
    lo, hi = out["model_bootstrap"]["ci95"]
    assert 0 <= lo <= hi <= 1
    keep = [j for j, f in enumerate(fams) if f != "B"]
    assert np.isclose(out["leave_one_family_out"]["B"], rb.pair_reversals(g1[keep], g2[keep])["reversed_share"])
    assert out["within_family"]["pairs"] + out["between_family"]["pairs"] == out["pairs"]
    # a model drawn twice is not a pair with itself
    Rm = dd.reversal_matrix(g1, g2)
    assert np.isclose(dd.share_on(Rm, np.array([0, 0, 1])), Rm[0, 1])
    assert np.isnan(dd.share_on(Rm, np.array([2, 2])))


@pytest.mark.skipif(not os.path.exists(RESULT), reason="results/decision_directions not built")
def test_results_reproduce_printed_and_reference_numbers():
    R = json.load(open(RESULT, encoding="utf-8"))
    printed = {"gsm8k": 2.52, "math": 3.00, "bbh": 2.42}
    reference = {"gsm8k": (2.06, 8.43), "math": (2.57, 12.75), "bbh": (2.65, 2.29)}
    for t, T in R["tasks"].items():
        assert T["one_draw"]["check_printed"]["abs_difference"] < 1e-9
        r = T["one_draw"]["rules"]["own_1"]
        assert round(r["requested_to_forced"]["estimate"], 2) == printed[t]
        s = T["strict"]["designs"]["disjoint"]["strict"]["rules"]["own_gain"]
        assert (round(s["requested_to_forced"]["estimate"], 2), round(s["forced_to_requested"]["estimate"], 2)) == reference[t]
        assert all(c["abs_difference"] < 1e-9 for c in T["strict"]["check_same_baseline"].values())
    assert round(R["tasks"]["math"]["one_draw"]["rules"]["own_1"]["forced_to_requested"]["estimate"], 2) == 13.08
    assert R["tasks"]["gsm8k"]["reversal"]["strict"]["reversed"] == 83
