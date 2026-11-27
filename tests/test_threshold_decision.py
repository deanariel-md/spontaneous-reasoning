"""Tests of the threshold decision (scripts/threshold_decision.py, paper/macros_threshold_decision.py)."""
import json
import math
import os
import re
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import macros_threshold_decision as mtd  # noqa: E402
import threshold_decision as td  # noqa: E402

RESULTS = os.path.join(ROOT, mtd.PATH)

# ------------------------------------------------------------------ a toy panel: 3 models, 2 pilot sets, 2 configurations
PRED = {"requested": np.array([[3.0, 1.0, 6.0], [2.0, 2.5, 7.0]]),
        "forced": np.array([[6.0, 1.0, 1.0], [6.0, 3.0, 2.0]])}
G = {"requested": np.array([[4.0, 0.0, 2.0], [2.0, 2.0, 4.0]]),
     "forced": np.array([[8.0, 1.0, 0.0], [6.0, 3.0, 2.0]])}
MODELS = ["a", "b", "c"]
UNITS = ["p0", "p1"]


def test_decide_strict_and_ties():
    d, tie = td.decide(PRED["requested"], 2.0)
    assert d.tolist() == [[1, 0, 1], [0, 1, 1]]
    assert tie.tolist() == [[False, False, False], [True, False, False]]
    d2, _ = td.decide(PRED["requested"], 2.0, tie_gets_cot=True)
    assert d2.tolist() == [[1, 0, 1], [1, 1, 1]]
    # a float a hair above c is a tie, not a gain above c
    d3, t3 = td.decide([2.0 + 1e-12, 2.0 + 1e-6], 2.0)
    assert d3.tolist() == [0, 1] and t3.tolist() == [True, False]
    with pytest.raises(ValueError):
        td.decide([np.nan, 1.0], 2.0)


def test_losses_by_hand():
    D = {a: td.decide(PRED[a], 2.0)[0] for a in td.ARM}
    gross = td.losses(D, G)
    # forced -> requested: mean decisions [.5,.5,1] vs [1,.5,0] on G_req: per configuration (4/3 - 4/3, 2 - 1)
    assert gross["forced_to_requested"]["estimate"] == pytest.approx(0.5)
    net = td.losses(D, {a: G[a] - 2.0 for a in td.ARM})
    d = np.array([-1 / 3, 2 / 3])
    assert net["forced_to_requested"]["estimate"] == pytest.approx(d.mean())
    se = d.std(ddof=1) / math.sqrt(2)
    q = stats.t.ppf(0.975, 1)
    assert net["forced_to_requested"]["ci95"] == pytest.approx([d.mean() - q * se, d.mean() + q * se])
    assert net["forced_to_requested"]["K"] == 2


def test_identical_decisions_cost_nothing():
    D = {a: td.decide(PRED["requested"], 2.0)[0] for a in td.ARM}
    for v in td.losses(D, G).values():
        assert v["estimate"] == pytest.approx(0.0)


def test_cut_block_counts_ties_and_identity():
    c = td.cut_block(PRED, G, 2.0, MODELS, UNITS)
    assert c["differ"]["mean_count"] == pytest.approx(1.5)
    assert c["differ"]["share"] == pytest.approx(0.5)
    assert (c["differ"]["min"], c["differ"]["max"]) == (1, 2)
    assert c["differ"]["cot_under_requested_only"] == pytest.approx(1.0)   # c in set 0, c in set 1
    assert c["differ"]["cot_under_forced_only"] == pytest.approx(0.5)      # a in set 1
    assert c["cot"]["requested"]["mean"] == pytest.approx(2.0)
    assert c["cot"]["forced"]["mean"] == pytest.approx(1.5)
    assert c["ties"]["requested"]["cells"] == 1
    assert c["ties"]["requested"]["where"] == [{"pilot_set": 1, "configuration": "p1", "model": "a", "pilot_gain": 2.0}]
    assert c["ties"]["forced"]["where"] == [{"pilot_set": 1, "configuration": "p1", "model": "c", "pilot_gain": 2.0}]
    assert "ties_get_cot" in c
    assert c["forced_to_requested"]["estimate"] == pytest.approx(1 / 6)
    assert c["gross"]["forced_to_requested"]["estimate"] == pytest.approx(0.5)
    assert c["identity_gross_net_max_abs_diff"] < 1e-12
    # with the ties given chain of thought, set 1's decisions agree on models a and c
    assert c["ties_get_cot"]["differ"]["mean_count"] == pytest.approx(0.5)


def test_every_and_no_model_counted():
    pred = {"requested": np.array([[9.0, 9.0, 9.0], [0.0, 0.0, 0.0]]), "forced": np.array([[9.0, 0.0, 9.0], [9.0, 0.0, 0.0]])}
    c = td.cut_block(pred, G, 5.0, MODELS, UNITS)
    assert c["cot"]["requested"]["pilot_sets_every_model"] == 1
    assert c["cot"]["requested"]["pilot_sets_no_model"] == 1
    assert c["cot"]["forced"]["pilot_sets_every_model"] == 0
    assert "ties_get_cot" not in c


# ------------------------------------------------------------------ macros


def toy_results():
    T = {"cuts": [td.cut_block(PRED, G, float(x), MODELS, UNITS) for x in td.CUTS]}
    return {"tasks": {"gsm8k": T, "math": T, "bbh": T}}


def test_macros_names_and_format():
    m = mtd.threshold_decision_macros(toy_results())
    assert len(m) == 3 * 3 * (2 + 2 * 2 * 3)
    for k in m:
        assert re.fullmatch(r"thrDec(Gsm|Math|Bbh)(Two|Five|Ten)(DiffPct|DiffCount|(Gross)?(FdToReq|ReqToFd)(Lo|Hi)?)", k), k
    assert m["thrDecGsmTwoDiffPct"] == "50"
    assert m["thrDecGsmTwoDiffCount"] == "1.5"
    assert m["thrDecGsmTwoFdToReq"] == "0.17"
    assert m["thrDecGsmTwoGrossFdToReq"] == "0.50"
    assert mtd.num(-0.004) == "0.00" and mtd.num(-0.25) == "$-$0.25" and mtd.num(12.5, 0) == "12"
    assert mtd.threshold_decision_macros(None) == {}


def test_prefix_does_not_collide_with_threshold_sweep():
    import macros_threshold_sweep as mts
    # the sweep's names are thr + a benchmark tag + a cutoff word; no tag begins with Dec
    assert not any(tag.startswith("Dec") for tag in mts.TASK_TAG.values())
    src = open(os.path.join(ROOT, "paper", "macros_threshold_sweep.py"), encoding="utf-8").read()
    assert "thrDec" not in src
    nt = os.path.join(ROOT, "paper", "numbers.tex")
    if os.path.exists(nt):
        names = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(nt, encoding="utf-8").read()))
        mine = set(mtd.threshold_decision_macros(toy_results()))
        # names already present are allowed only if numbers.tex was rebuilt with this module wired in
        clash = mine & {n for n in names if not n.startswith("thrDec")}
        assert not clash
        assert not any(n.startswith("thrDec") and n not in mine for n in names)


# ------------------------------------------------------------------ the real run (skipped until it exists)


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="results/threshold_decision/threshold_decision.json not built")
def test_results_file():
    R = json.load(open(RESULTS, encoding="utf-8"))
    assert set(R["tasks"]) == {"gsm8k", "math", "bbh"}
    assert R["scores"].startswith("strict")
    assert all(("MISMATCH" not in c) for c in R["checks"])
    for t, T in R["tasks"].items():
        assert T["check_top_third"]["ok"]
        assert max(T["pilot_gain_reproduced_max_abs_diff"].values()) < 1e-9
        assert [c["cut"] for c in T["cuts"]] == [2.0, 5.0, 10.0]
        for c in T["cuts"]:
            assert c["identity_gross_net_max_abs_diff"] < 1e-9
            assert 0 <= c["differ"]["share"] <= 1
            assert c["differ"]["mean_count"] == pytest.approx(c["differ"]["share"] * T["M"])
            for k in td.DIRECTIONS:
                assert c[k]["K"] == T["n_validation_configurations"]
    assert R["macros"] == mtd.threshold_decision_macros(R)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "processed", "decision", "math_strict_validation.npz")),
                    reason="the decision analysis's arrays are not built")
def test_rerun_matches_saved_results():
    if not os.path.exists(RESULTS):
        pytest.skip("results not built")
    R = json.load(open(RESULTS, encoding="utf-8"))
    T, dec = td.run_task("math")
    S = R["tasks"]["math"]
    assert T["n_validation_configurations"] == S["n_validation_configurations"]
    for a, b in zip(T["cuts"], S["cuts"]):
        for k in td.DIRECTIONS:
            assert a[k]["estimate"] == pytest.approx(b[k]["estimate"], abs=1e-12)
            assert a["gross"][k]["estimate"] == pytest.approx(b["gross"][k]["estimate"], abs=1e-12)
        assert a["differ"]["mean_count"] == pytest.approx(b["differ"]["mean_count"])
    assert len(dec) == 3 * 2 * T["n_pilot_sets"] * T["M"]
