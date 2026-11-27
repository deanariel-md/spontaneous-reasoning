"""Tests of the threshold sweep (scripts/threshold_sweep.py, paper/macros_threshold_sweep.py)."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import macros_threshold_sweep as mts  # noqa: E402
import threshold_sweep as ts  # noqa: E402


# ------------------------------------------------------------------ a small synthetic panel


def toy_rows():
    """3 models x 2 configurations x 4 items. Tokens chosen so the long share moves with the cutoff."""
    rows = []
    tok = {"a": [4, 10, 20, 70], "b": [5, 6, 7, 40], "c": [30, 30, 30, 30]}
    req = {"a": [0, 1, 1, 1], "b": [0, 0, 1, 0], "c": [1, 1, 1, 0]}
    fd = {"a": [0, 0, 1, 0], "b": [0, 0, 0, 0], "c": [0, 1, 0, 0]}
    cot = {"a": [1, 1, 1, 1], "b": [1, 0, 1, 1], "c": [1, 1, 1, 1]}
    for m in "abc":
        for cfg in ("c1", "c2"):
            for i in range(4):
                r = {"model_key": m, "config_id": cfg, "item_id": str(i), "n_output_tokens": tok[m][i]}
                for sc in ("strict", "content"):
                    r[f"requested_{sc}"] = req[m][i]
                    r[f"forced_{sc}"] = fd[m][i]
                    r[f"cot_{sc}"] = cot[m][i]
                rows.append(r)
    return pd.DataFrame(rows)


def toy_gains(wide, panel):
    g = {}
    for sc in ("strict", "content"):
        acc = wide.groupby("model_key")[[f"requested_{sc}", f"forced_{sc}", f"cot_{sc}"]].mean().loc[panel] * 100
        C, R, F = (acc[f"{a}_{sc}"].to_numpy(float) for a in ("cot", "requested", "forced"))
        g[sc] = {"requested": C - R, "forced": C - F}
    return g


def test_sweep_cut_w_and_variance_share():
    wide, panel = toy_rows(), ["a", "b", "c"]
    G = toy_gains(wide, panel)
    r8 = ts.sweep_cut(wide, panel, ["c1", "c2"], [], G, 8)
    # w at 8: a has 3 of 4 above 8, b 1 of 4, c 4 of 4
    assert np.allclose(r8["w_discovery"], [0.75, 0.25, 1.0])
    assert r8["mean_w_pct"] == pytest.approx(100 * (0.75 + 0.25 + 1.0) / 3)
    assert r8["models_w_one"] == 1
    r64 = ts.sweep_cut(wide, panel, ["c1", "c2"], [], G, 64)
    assert np.allclose(r64["w_discovery"], [0.25, 0.0, 0.0])
    assert r64["models_w_zero"] == 2
    # the variance share by hand at 8: long part = sum over long rows of (r - f) / all rows, per model (pp)
    shift = np.array([50.0, 25.0, 50.0])
    long8 = np.array([100 * 2 / 4, 0.0, 100 * 2 / 4])   # a: items 1, 3 (1-0, 1-0; item 2 is 1-1); b: item 3 is 0-0
    share = np.cov(long8, shift, ddof=1)[0, 1] / np.var(shift, ddof=1)
    assert r8["strict"]["variance_share_long"] == pytest.approx(share)
    assert r8["content"]["variance_share_long"] == pytest.approx(share)
    # reversed pairs: observed gain over requested against (1 - w)(C - F)
    g_req, g_fd = G["strict"]["requested"], G["strict"]["forced"]
    pred = (1 - np.array(r8["w_discovery"])) * g_fd
    n = 0
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = g_req[i] - g_req[j], pred[i] - pred[j]
            if abs(a) > 1e-9 and abs(b) > 1e-9 and a * b < 0:
                n += 1
    assert r8["strict"]["reversed_corrected"] == n
    assert r8["strict"]["pairs"] == 3


def test_gains_from_rows():
    g = ts.gains_from_rows({"C": [80.0, 50.0], "F": [20.0, 40.0], "R": [30.0, 45.0]})
    assert np.allclose(g["requested"], [50.0, 5.0])
    assert np.allclose(g["forced"], [60.0, 10.0])


# ------------------------------------------------------------------ the macros and the table


def toy_results():
    def cut(c, var, rev, w):
        s = {"variance_share_long": var, "reversed_corrected": rev, "pairs": 406}
        return {"cut": c, "mean_w_pct": w, "strict": dict(s), "content": {**s, "variance_share_long": None, "reversed_corrected": rev + 1}}
    return {"tasks": {
        "gsm8k": {"cuts": [8, 16, 32, 64], "by_cut": {str(c): cut(c, -0.0123 if c == 8 else 0.92, 10 * c, 43.5)
                                                       for c in (8, 16, 32, 64)}},
        "hotpot": {"cuts": [16, 64], "by_cut": {str(c): cut(c, 0.2, 5, 2.4) for c in (16, 64)}}}}


def test_macros_names_and_format():
    m = mts.threshold_sweep_macros(toy_results())
    assert all(re.fullmatch(r"thr(Gsm|Hq)(Eight|Sixteen|ThirtyTwo|SixtyFour)(VarLongPct|RevPairs|RevOf|MeanWPct)(Content)?", k)
               for k in m)
    assert len(m) == 4 * 4 * 2 + 2 * 4 * 2
    assert m["thrGsmEightVarLongPct"] == "$-$1"
    assert m["thrGsmSixteenVarLongPct"] == "92"
    assert m["thrGsmSixteenVarLongPctContent"] == "--"   # a variance share that is undefined prints as such
    assert m["thrGsmThirtyTwoRevPairs"] == "320" and m["thrGsmThirtyTwoRevPairsContent"] == "321"
    assert m["thrGsmSixtyFourRevOf"] == "406"
    assert m["thrGsmSixteenMeanWPct"] == m["thrGsmSixteenMeanWPctContent"] == "44"
    assert "thrHqEightVarLongPct" not in m and m["thrHqSixtyFourMeanWPct"] == "2"
    assert mts.threshold_sweep_macros(None) == {}


def test_table_shape():
    tex = mts.threshold_sweep_table(toy_results())
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and ("strict" in ln or "content" in ln)
            and not ln.startswith("Benchmark")]
    assert len(body) == 4
    for ln in body:
        assert ln.count("&") == 14
    hq = [ln for ln in body if ln.startswith("HotpotQA")][0]
    assert hq.split(" & ")[3:6] == ["--", "--", "--"]
    assert r"\begin{tabular}" in tex and r"\begin{table}" not in tex


def test_table_refuses_mixed_pair_totals():
    R = toy_results()
    R["tasks"]["gsm8k"]["by_cut"]["32"]["strict"]["pairs"] = 400
    with pytest.raises(ValueError):
        mts.threshold_sweep_table(R)


# ------------------------------------------------------------------ the stored results


STORED = os.path.join(ROOT, "results", "threshold_sweep", "threshold_sweep.json")


@pytest.mark.skipif(not os.path.exists(STORED), reason="threshold sweep not run")
def test_stored_sixteen_column_reproduces_numbers():
    B = json.load(open(STORED, encoding="utf-8"))
    assert all(c["ok"] for c in B["checks"]), [c for c in B["checks"] if not c["ok"]]
    m = mts.threshold_sweep_macros(B)
    txt = open(os.path.join(ROOT, "paper", "numbers.tex"), encoding="utf-8").read()
    N = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}\n]*)\}", txt))
    for t in ("Gsm", "Math", "Bbh"):
        for suf in ("", "Content"):
            assert m[f"thr{t}SixteenVarLongPct{suf}"] == N[f"bmixDec{t}VarLongPct{suf}"]
            assert m[f"thr{t}SixteenRevPairs{suf}"] == N[f"bmix{t}RevCorr{suf}"]
            assert m[f"thr{t}SixteenRevOf{suf}"] == N[f"bmix{t}Pairs{suf}"]
            assert m[f"thr{t}SixteenMeanWPct{suf}"] == N[f"bmix{t}WMeanPct"]
    # wired into make_numbers.py: a name in numbers.tex must hold this value
    assert {k: (v, N[k]) for k, v in m.items() if k in N and N[k] != v} == {}
    with open(os.path.join(ROOT, "paper", "tables", "threshold_sweep.tex"), encoding="utf-8") as fh:
        assert fh.read() == mts.threshold_sweep_table(B)
