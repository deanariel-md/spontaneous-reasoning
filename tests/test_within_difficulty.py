"""Tests of src/within_difficulty.py, scripts/run_within_difficulty.py,
scripts/arm_wording_within_difficulty.py and paper/macros_within_difficulty.py."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import within_difficulty as wd  # noqa: E402
import macros_within_difficulty as mwd  # noqa: E402

RES = os.path.join(ROOT, "results", "within_difficulty", "within_difficulty.json")
FIGJ = os.path.join(ROOT, "results", "arm_wording", "within_difficulty.json")
FIG_SRC = os.path.join(ROOT, "results", "arm_wording", "tables", "fig_w_vs_requested.csv")


def _sim(M=6, I=150, slope=0.15, n=24, seed=1, groups=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (M, I))
    al = rng.normal(-1, 1, M)
    eta = al[:, None] + slope * X
    g = None
    if groups:
        g = rng.integers(0, groups, I)
        eta = eta + np.concatenate([[0.0], rng.normal(0, 0.5, groups - 1)])[g][None, :]
    N = np.full((M, I), float(n))
    K = rng.binomial(n, expit(eta)).astype(float)
    return K, N, X, g


def _bfgs(K, N, X, g=None):
    M = K.shape[0]
    G = 1 if g is None else int(g.max()) + 1

    def nll(th):
        e = th[:M][:, None] + th[-1] * X
        if g is not None:
            e = e + np.concatenate([[0.0], th[M:M + G - 1]])[g][None, :]
        return -np.sum(K * e - N * np.logaddexp(0, e))
    return minimize(nll, np.zeros(M + G), method="BFGS", options={"gtol": 1e-9})


# ---------------------------------------------------------------------------- the estimator
@pytest.mark.parametrize("groups", [None, 4])
def test_fe_logit_matches_general_optimiser(groups):
    K, N, X, g = _sim(groups=groups)
    f = wd.fe_logit(K, N, X, g)
    r = _bfgs(K, N, X, g)
    assert f["converged"]
    assert f["slope"] == pytest.approx(r.x[-1], abs=1e-5)
    assert f["loglik"] == pytest.approx(-r.fun, rel=1e-10)


def test_fe_logit_recovers_slope():
    K, N, X, _ = _sim(M=20, I=2000, slope=-0.2, seed=3)
    f = wd.fe_logit(K, N, X)
    assert f["slope"] == pytest.approx(-0.2, abs=0.01)


def test_duplicated_model_equals_doubled_counts():
    """A model drawn twice in a resample (its own intercept each time) gives the fit of doubled counts."""
    K, N, X, _ = _sim(M=4, seed=5)
    ix = np.array([0, 0, 1, 2, 3])
    a = wd.fe_logit(K[ix], N[ix], X[ix])
    K2, N2 = K.copy(), N.copy()
    K2[0] *= 2
    N2[0] *= 2
    b = wd.fe_logit(K2, N2, X)
    assert a["slope"] == pytest.approx(b["slope"], abs=1e-9)


def test_loo_difficulty_hand_example():
    # 3 models, 2 items; F correct forced answers out of N
    F = np.array([[2, 0], [1, 1], [0, 2]], float)
    N = np.full((3, 2), 2.0)
    D = wd.loo_difficulty(F, N)
    # model 0, item 0: others 1 + 0 correct of 4 -> 25% correct -> difficulty 75
    assert D[0, 0] == pytest.approx(75.0)
    assert D[2, 0] == pytest.approx(25.0)   # others 2 + 1 of 4 correct
    assert D[1, 1] == pytest.approx(50.0)   # others 0 + 2 of 4 correct


def test_variation_and_per_model_edge_cases():
    K, N, X, _ = _sim(M=3, seed=7)
    K[1] = 0          # no long output
    K[2] = N[2]       # every output long
    assert list(wd.variation(K, N)) == ["varies", "none_long", "all_long"]
    pm = wd.per_model_slopes(K, N, X)
    assert pm[0]["slope"] is not None and pm[0]["converged"]
    assert pm[1]["slope"] is None and pm[1]["variation"] == "none_long"
    assert pm[2]["slope"] is None and pm[2]["variation"] == "all_long"


def test_per_model_with_groups_drops_groups_without_variation():
    K, N, X, g = _sim(M=2, I=300, seed=11, groups=4)
    K[0, g == 2] = 0                       # model 0 never long in group 2
    pm = wd.per_model_slopes(K, N, X, g)
    assert pm[0]["groups_dropped"] == 1 and pm[0]["converged"]
    assert pm[1]["groups_dropped"] == 0 and pm[1]["converged"]
    keep = g != 2
    ref = wd.fe_logit(K[:1, keep], N[:1, keep], X[:1, keep], np.searchsorted([0, 1, 3], g[keep]))
    assert pm[0]["slope"] == pytest.approx(ref["slope"], abs=1e-9)


def test_separation_is_flagged_not_hidden():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    N = np.full((1, 4), 10.0)
    K = np.array([[0.0, 0.0, 10.0, 10.0]])   # perfectly separated by X
    f = wd.fe_logit(K, N, X)
    assert not f["converged"]


def test_model_bootstrap_reproducible():
    K, N, X, _ = _sim(M=8, seed=9)
    fit = wd.fe_logit(K, N, X)
    b1 = wd.model_bootstrap(K, N, X, 200, np.random.default_rng(1), fit=fit)
    b2 = wd.model_bootstrap(K, N, X, 200, np.random.default_rng(1), fit=fit)
    assert np.array_equal(b1["slopes"], b2["slopes"])
    s = wd.odds_ratio_summary(fit, b1)
    assert s["odds_ratio_ci95"][0] <= s["odds_ratio"] <= s["odds_ratio_ci95"][1]
    assert s["boot_not_converged"] == 0


# ---------------------------------------------------------------------------- the results file
needs_res = pytest.mark.skipif(not os.path.exists(RES), reason="results/within_difficulty not written yet")


@needs_res
def test_results_consistent():
    R = json.load(open(RES, encoding="utf-8"))
    for t in ("gsm8k", "math", "bbh"):
        T = R["tasks"][t]
        assert T["forced_cap_tokens"] == 16
        assert T["w_check_max_abs_diff"] <= 1e-12
        assert T["discovery_draws"] == 24
        for ln, L in T["lines"].items():
            assert L["models_in_fit"] + len(L["no_variation"]) == L["models_total"] == len(T["models"])
            assert L["positive"] + L["negative"] + L["zero"] == L["models_in_fit"]
            assert L["pooled"]["n_boot"] == R["n_boot"]
            assert L["pooled_fit_converged"]
            lo, hi = L["pooled"]["odds_ratio_ci95"]
            assert lo <= L["pooled"]["odds_ratio"] <= hi
            pm = [r for r in L["per_model"] if r["slope_per10"] is not None]
            assert sum(r["slope_per10"] > 0 for r in pm) == L["positive"]
    assert "bbh_subtask" in R["tasks"]["bbh"]["lines"]
    st = R["tasks"]["bbh"]["subtask_table"]
    assert st["subtasks"] == len(st["rows"]) == R["tasks"]["bbh"]["lines"]["bbh_subtask"]["groups"]
    for ln in ("bbh_subtask", "bbh_subtask_content"):
        L = R["tasks"]["bbh"]["lines"][ln]
        assert all(r["groups_dropped"] is not None for r in L["per_model"] if r["variation"] == "varies")


@needs_res
def test_long_share_is_w_of_baseline_mixture():
    """The main line's long share per model is the w of results/baseline_mixture (discovery draws)."""
    bm = os.path.join(ROOT, "results", "baseline_mixture", "baseline_mixture.json")
    if not os.path.exists(bm):
        pytest.skip("baseline_mixture.json absent")
    B = json.load(open(bm, encoding="utf-8"))
    R = json.load(open(RES, encoding="utf-8"))
    for t in ("gsm8k", "math", "bbh"):
        tb = B["tasks"][t] if "tasks" in B else B[t]
        w = dict(zip(tb["models"], tb["w_discovery"]))
        for r in R["tasks"][t]["lines"]["main"]["per_model"]:
            assert r["long_share"] == pytest.approx(w[r["model_key"]], abs=1e-12)


# ---------------------------------------------------------------------------- macros
def test_macros_from_synthetic_results():
    line = lambda o, lo, hi, pos, m: {"pooled": {"odds_ratio": o, "odds_ratio_ci95": [lo, hi]},  # noqa: E731
                                      "positive": pos, "models_in_fit": m}
    R = {"wd": {"tasks": {t: {"lines": {"main": line(1.1049, 1.051, 1.16, 23, 28),
                                        "content": line(0.999, 0.9, 1.004, 10, 29)}}
                          for t in ("gsm8k", "math", "bbh")}},
         "fig": {"counts": {"gsm8k": {"qualify": 13}, "math": {"qualify": 15}}}}
    R["wd"]["tasks"]["bbh"]["lines"]["bbh_subtask"] = line(1.07, 1.05, 1.09, 24, 27)
    R["wd"]["tasks"]["bbh"]["lines"]["bbh_subtask_content"] = line(1.06, 1.045, 1.08, 24, 27)
    M = mwd.within_difficulty_macros(R)
    assert M["diffOrBbhSub"] == "1.07" and M["diffPosBbhSubContent"] == "24"
    assert M["diffOrBbhSubContentLo"] == f"{1.045:.2f}"
    assert M["diffOrGsm"] == "1.10" and M["diffOrGsmLo"] == "1.05" and M["diffOrGsmHi"] == "1.16"
    assert M["diffPosBbh"] == "23" and M["diffModelsMath"] == "28"
    assert M["diffOrMathContent"] == "1.00" and M["diffPosGsmContent"] == "10"
    assert M["rewordModelsGsm"] == "13" and M["rewordModelsMath"] == "15"
    assert len(M) == 3 * 2 * 5 + 2 * 5 + 2
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in M)
    assert mwd.within_difficulty_macros(None) == {}


@needs_res
def test_macros_from_results_and_no_clash():
    M = mwd.within_difficulty_macros(mwd.within_difficulty_results(ROOT))
    R = json.load(open(RES, encoding="utf-8"))
    for t, tag in mwd.TASKS:
        L = R["tasks"][t]["lines"]["main"]
        assert M[f"diffOr{tag}"] == f"{L['pooled']['odds_ratio']:.2f}"
        assert M[f"diffModels{tag}"] == str(L["models_in_fit"])
    nt = os.path.join(ROOT, "paper", "numbers.tex")
    if os.path.exists(nt):
        txt = open(nt, encoding="utf-8").read()
        names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", txt)
        mk = open(os.path.join(ROOT, "paper", "make_numbers.py"), encoding="utf-8").read()
        if "macros_within_difficulty" in mk:
            # wired: each name defined exactly once
            for k in M:
                assert names.count(k) == 1, f"{k} defined {names.count(k)} times in numbers.tex"
            return
        assert not (set(M) & set(names)), "a macro name is already in numbers.tex"


# ---------------------------------------------------------------------------- rewording
def test_fig_table_threshold_and_slopes(tmp_path):
    import arm_wording_within_difficulty as f2d
    rows = []
    for m, ws in (("a", [0.1, 0.2, 0.3, 0.3]), ("b", [0.5, 0.6, 0.6, 0.69])):   # spans 0.2 (qualifies) and 0.19
        for j, w in enumerate(ws):
            rows.append({"model_key": m, "task": "gsm8k", "clause": f"dir{j}", "clause_text": "x", "w": w,
                         "n_requested": 100, "n_requested_long": int(100 * w),
                         "requested_accuracy_pp": 10 + 50 * w, "forced_accuracy_pp": 20.0,
                         "cot_accuracy_pp": 80.0, "requested_minus_forced_pp": 0.0})
            rows.append({**rows[-1], "task": "math"})
    p = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    tab = f2d.table(str(p), fam={"a": "Qwen", "b": "Phi"})
    per = tab.drop_duplicates(["task", "model_key"]).set_index(["task", "model_key"])
    assert bool(per.loc[("gsm8k", "a"), "qualifies"]) is (0.3 - 0.1 >= 0.2)
    assert not per.loc[("gsm8k", "b"), "qualifies"]
    assert per.loc[("gsm8k", "a"), "requested_slope_pp_per_w"] == pytest.approx(50.0)
    assert per.loc[("gsm8k", "a"), "forced_slope_pp_per_w"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not os.path.exists(FIG_SRC), reason="rewording table absent")
def test_reword_counts():
    import arm_wording_within_difficulty as f2d
    tab = f2d.table()
    c = f2d.counts(tab)
    per = tab.drop_duplicates(["task", "model_key"])
    for t in f2d.TASKS:
        p = per[per["task"] == t]
        assert c[t]["models"] == 29
        assert c[t]["qualify"] == int((p["w_span"] >= 0.2).sum())
    if os.path.exists(FIGJ):
        J = json.load(open(FIGJ, encoding="utf-8"))
        assert {t: J["counts"][t]["qualify"] for t in f2d.TASKS} == {t: c[t]["qualify"] for t in f2d.TASKS}
