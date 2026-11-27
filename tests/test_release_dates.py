"""Tests of scripts/release_dates.py and paper/macros_release_dates.py.

The checks on the manifests and on results/release_dates/release_dates.json are skipped until the script has run."""
from __future__ import annotations

import datetime as dtm
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

import macros_release_dates as mr  # noqa: E402
import release_dates as rd  # noqa: E402

JSON = os.path.join(ROOT, mr.PATH)
needs_json = pytest.mark.skipif(not os.path.exists(JSON), reason="scripts/release_dates.py run has not run")
needs_manifests = pytest.mark.skipif(not (os.path.exists(rd.DATES) and os.path.exists(rd.PAIRS)),
                                     reason="scripts/release_dates.py dates and pairs have not run")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def numbers_tex() -> dict:
    p = os.path.join(ROOT, "paper", "numbers.tex")
    txt = open(p, encoding="utf-8").read()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^\n]*)\}\s*$", txt, flags=re.M))


# ---------------------------------------------------------------------------------------------- the pairing rule


def E(family, date, params, group="panel"):
    return {"family": family, "date": date, "params_b": params, "group": group}


def sizes_of(entries):
    return {k: rd.size_of(k, e, {}) for k, e in entries.items()}


def test_pair_rule_most_recent_similar_earlier():
    e = {"a-7b": E("F", "2023-01-01", 7.0), "b-8b": E("F", "2024-01-01", 8.0), "c-9b": E("F", "2025-01-01", 9.0),
         "d-30b": E("F", "2025-06-01", 30.0), "x-8b": E("G", "2022-01-01", 8.0)}
    P = rd.pair_rule(list(e), e, sizes_of(e), 1.5)
    got = {(p["older"], p["newer"]) for p in P}
    # c pairs with b (the most recent similar), not a; d has no similar earlier; x is another family
    assert got == {("a-7b", "b-8b"), ("b-8b", "c-9b")}


def test_pair_rule_size_band_and_same_date():
    e = {"a-7b": E("F", "2023-01-01", 7.0), "b-11b": E("F", "2024-01-01", 11.0),
         "c-8b": E("F", "2024-01-01", 8.0)}
    P = rd.pair_rule(list(e), e, sizes_of(e), 1.5)
    # b: 11/7 > 1.5, and c shares its date, so b has no pair; c pairs with a (8/7 <= 1.5)
    assert {(p["older"], p["newer"]) for p in P} == {("a-7b", "c-8b")}
    P2 = rd.pair_rule(list(e), e, sizes_of(e), 2.0)
    assert {(p["older"], p["newer"]) for p in P2} == {("a-7b", "b-11b"), ("a-7b", "c-8b")}


def test_pair_rule_moe_and_tie_break():
    e = {"m-30b-a3b": E("Q", "2025-04-27", 30.5), "d-32b": E("Q", "2025-04-27", 32.8),
         "m2-30b-a3b-2507": E("Q", "2025-07-28", 30.5), "d2-27b": E("Q", "2026-04-21", 27.8),
         "m3-35b-a3b": E("Q", "2026-02-24", 36.0)}
    s = sizes_of(e)
    assert s["m-30b-a3b"]["moe"] and s["m-30b-a3b"]["active"] == 3.0 and not s["d-32b"]["moe"]
    P = {(p["older"], p["newer"]) for p in rd.pair_rule(list(e), e, s, 1.5)}
    # a mixture of experts pairs only with a mixture of experts; the dense 27B goes to the dense 32B
    assert P == {("m-30b-a3b", "m2-30b-a3b-2507"), ("m2-30b-a3b-2507", "m3-35b-a3b"), ("d-32b", "d2-27b")}


def test_pair_rule_tie_in_date_goes_to_nearer_size():
    e = {"a-7b": E("F", "2023-01-01", 7.0), "b-9b": E("F", "2023-01-01", 9.0), "c-8b": E("F", "2024-01-01", 8.5)}
    P = rd.pair_rule(list(e), e, sizes_of(e), 1.5)
    assert [(p["older"], p["newer"]) for p in P] == [("b-9b", "c-8b")]


def test_no_size_no_pair():
    e = {"a-7b": E("F", "2023-01-01", 7.0), "closed": {"family": "F", "date": "2024-01-01", "group": "extension"}}
    s = {"a-7b": rd.size_of("a-7b", e["a-7b"], {}), "closed": rd.size_of("closed", e["closed"], {})}
    assert s["closed"] is None
    assert rd.pair_rule(list(e), e, s, 1.5) == []


# ---------------------------------------------------------------------------------------------- statistics


def ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman_by_hand(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def test_spearman_ties_and_constant():
    x = [1, 2, 2, 3, 5, 5, 5]
    y = [0.3, 0.1, 0.9, 0.2, 0.4, 0.4, 0.8]
    assert abs(rd.spearman(x, y) - spearman_by_hand(x, y)) < 1e-12
    assert np.isnan(rd.spearman([1, 1, 1], [1, 2, 3]))
    assert np.isnan(rd.spearman([1], [2]))


def test_family_bootstrap_reproducible_and_counts_undefined():
    x = [1, 2, 3, 4, 5, 6]
    y = [6, 5, 4, 3, 2, 1]
    f = ["A", "A", "B", "B", "C", "C"]
    a = rd.family_bootstrap_rho(x, y, f, 200, np.random.default_rng(1))
    b = rd.family_bootstrap_rho(x, y, f, 200, np.random.default_rng(1))
    assert a == b
    assert a["ci95"] == pytest.approx([-1.0, -1.0], abs=1e-12) and a["n_families"] == 3 and a["n_undefined"] == 0
    # one model per family: a draw of one family three times has a constant date, so rho is undefined
    c = rd.family_bootstrap_rho([1, 2, 3], [3, 1, 2], ["A", "B", "C"], 500, np.random.default_rng(2))
    assert c["n_undefined"] > 0


def test_pair_counts():
    pairs = [{"older": "a", "newer": "b"}, {"older": "b", "newer": "c"}, {"older": "c", "newer": "d"},
             {"older": "a", "newer": "z"}]
    W = {"a": 0.5, "b": 0.6, "c": 0.6 + 5e-10, "d": 0.1}
    x = rd.pair_counts(pairs, W)
    assert (x["pairs"], x["newer_higher"], x["newer_lower"], x["ties"]) == (3, 1, 1, 1)


def test_write_once(tmp_path):
    p = str(tmp_path / "m.json")
    assert rd.write_once(p, {"a": 1, "t": "x"}, ("a",), False) == "written"
    assert rd.write_once(p, {"a": 1, "t": "y"}, ("a",), False) == "kept (identical)"
    assert load(p)["t"] == "x"
    with pytest.raises(SystemExit):
        rd.write_once(p, {"a": 2}, ("a",), False)
    assert rd.write_once(p, {"a": 2}, ("a",), True) == "written"


# ---------------------------------------------------------------------------------------------- the macros module


def test_results_absent_gives_none(tmp_path):
    assert mr.release_dates_results(str(tmp_path)) is None
    assert mr.release_dates_macros(None) == {}


def test_names_are_letters_and_complete():
    assert len(mr.NAMES) == len(set(mr.NAMES)) == 30
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in mr.NAMES)
    with pytest.raises(ValueError):
        mr.release_dates_macros({"macros": {n: "1" for n in mr.NAMES[:-1]}})
    with pytest.raises(ValueError):
        mr.release_dates_macros({"macros": {**{n: "1" for n in mr.NAMES}, "wExtra": "1"}})


@needs_json
def test_no_clash_with_numbers_tex():
    """A name already in numbers.tex must hold the same value (it is ours, wired in); no other module sets these."""
    have = numbers_tex()
    m = mr.release_dates_macros(mr.release_dates_results(ROOT))
    for k, v in m.items():
        if k in have:
            assert have[k] == v, k
    import glob
    for f in glob.glob(os.path.join(ROOT, "paper", "macros_*.py")):
        if f.endswith("macros_release_dates.py"):
            continue
        s = open(f, encoding="utf-8").read()
        assert not [n for n in mr.NAMES if re.search(r"\b" + n + r"\b", s)], f


# ---------------------------------------------------------------------------------------------- the manifests


@needs_manifests
def test_dates_manifest_matches_the_fourteen():
    D = load(rd.DATES)
    W = load(os.path.join(ROOT, "results", "statement_values", "statement_values.json"))["recent"]
    panel = {k: e for k, e in D["entries"].items() if e["group"] == "panel"}
    assert len(panel) == 29
    assert sorted(k for k, e in panel.items() if e["date"] >= rd.RECENT_FROM) == sorted(W["union_models"])
    assert len(W["union_models"]) == 14
    for k, e in panel.items():
        assert e["date"] == W["dates"][k]["created"] and e["basis"] == "hf_repo_created"
    ext = {k: e for k, e in D["entries"].items() if e["group"] == "extension"}
    assert sorted(ext) == sorted(list(rd.LADDER6) + [rd.CLOSED])
    for k, e in D["entries"].items():
        dtm.date.fromisoformat(e["date"])
        assert e["source"] and e["basis"]


@needs_manifests
def test_dates_manifest_reproduces():
    D = load(rd.DATES)
    B = rd.build_dates()
    assert rd.jsonable(B["entries"]) == D["entries"]
    assert rd.jsonable(B["flags"]) == D["flags"]


@needs_manifests
def test_pairs_manifest_reproduces_and_was_written_after_the_dates():
    D, P = load(rd.DATES), load(rd.PAIRS)
    assert P["dates_written_at"] == D["written_at"]
    t = lambda s: dtm.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")  # noqa: E731
    assert t(D["written_at"]) <= t(P["written_at"])
    B = rd.jsonable(rd.build_pairs(D))
    for k in ("rule", "rule_alt", "pairs", "pairs_alt", "extension_pairs"):
        assert B[k] == P[k], k
    assert len(P["pairs"]) == 14
    for p in P["pairs"]:
        assert p["older_date"] < p["newer_date"]
        assert D["entries"][p["older"]]["family"] == D["entries"][p["newer"]]["family"]
        assert max(p["older_total_b"], p["newer_total_b"]) / min(p["older_total_b"], p["newer_total_b"]) <= 1.5


# ---------------------------------------------------------------------------------------------- the results


@needs_json
def test_results_written_after_the_manifests():
    R, P = load(JSON), load(rd.PAIRS)
    assert R["pairs_manifest"]["written_at"] == P["written_at"]
    assert dtm.datetime.strptime(R["created"], "%Y-%m-%d %H:%M") >= \
        dtm.datetime.strptime(P["written_at"][:16], "%Y-%m-%d %H:%M")


@needs_json
def test_item2_recomputed_from_per_model_csv():
    """rho, the means and the pair counts again, from results/baseline_mixture/tables/per_model.csv and the
    manifests, with ranks computed here."""
    R = load(JSON)
    D, P = load(rd.DATES), load(rd.PAIRS)
    pm = pd.read_csv(os.path.join(ROOT, "results", "baseline_mixture", "tables", "per_model.csv"))
    m = R["macros"]
    for t, tg in rd.T_TAG.items():
        x = pm[pm.task == t]
        W = dict(zip(x.model_key, x.w_discovery))
        keys = list(W)
        assert len(keys) == (28 if t == "bbh" else 29)
        days = [dtm.date.fromisoformat(D["entries"][k]["date"]).toordinal() for k in keys]
        rho = spearman_by_hand(days, [W[k] for k in keys])
        assert abs(rho - R["tasks"][t]["rho"]) < 1e-9
        assert m[f"wDateRho{tg}"] == rd.num(rho, 2)
        ci = R["tasks"][t]["family_bootstrap"]["ci95"]
        assert (m[f"wDateRho{tg}Lo"], m[f"wDateRho{tg}Hi"]) == (rd.num(ci[0], 2), rd.num(ci[1], 2))
        rec = [W[k] for k in keys if D["entries"][k]["date"] >= "2025-01-01"]
        old = [W[k] for k in keys if D["entries"][k]["date"] < "2025-01-01"]
        assert len(rec) == (13 if t == "bbh" else 14) and len(rec) + len(old) == len(keys)
        assert m[f"wRecentMean{tg}Pct"] == rd.num(100 * np.mean(rec), 1)
        assert m[f"wOlderMean{tg}Pct"] == rd.num(100 * np.mean(old), 1)
        hi = sum(1 for p in P["pairs"] if p["older"] in W and p["newer"] in W and W[p["newer"]] - W[p["older"]] > 1e-9)
        n = sum(1 for p in P["pairs"] if p["older"] in W and p["newer"] in W)
        assert hi == R["tasks"][t]["pairs"]["newer_higher"] and n == R["tasks"][t]["pairs"]["pairs"]
        if t != "bbh":
            assert m[f"wNewerHigher{tg}"] == str(hi) and m["wNewerPairs"] == str(n)


@needs_json
def test_item3_point_values_reproduce_from_the_compliance_tables():
    """The reversed shares among the recent checkpoints, from results/content_scoring/tables/compliance_<task>.csv
    (strict model means over the discovery draws), equal \\aeRecent<T>ReversedPct and lie in their intervals' span."""
    R = load(JSON)
    D = load(rd.DATES)
    tex = numbers_tex()
    for t, tg in rd.T_TAG.items():
        c = pd.read_csv(os.path.join(ROOT, "results", "content_scoring", "tables", f"compliance_{t}.csv"))
        c = c[c.model_key.map(lambda k: D["entries"][k]["date"] >= "2025-01-01")]
        g1 = (c.cot_strict_correct - c.requested_strict_correct).to_numpy()
        g2 = (c.cot_strict_correct - c.forced_strict_correct).to_numpy()
        i, j = np.triu_indices(len(c), 1)
        rev = int(((np.sign(g1[i] - g1[j]) * np.sign(g2[i] - g2[j])) < 0).sum())
        x = R["recent_reversals"][t]
        assert (rev, len(i)) == (x["reversed"], x["pairs"])
        assert tex[f"aeRecent{tg}ReversedPct"] == rd.num(100 * rev / len(i), 0) == x["stored_pct"]
        assert tex[f"aeRecent{tg}Reversed"] == str(rev) and tex[f"aeRecent{tg}Pairs"] == str(len(i))
        m = R["macros"]
        for s, b in (("", "model_bootstrap"), ("Fam", "family_bootstrap")):
            lo, hi = x[b]["ci95"]
            assert 0 <= lo <= hi <= 1
            assert m[f"aeRecent{tg}ReversedPct{s}Lo"] == rd.num(100 * lo, 0)
            assert m[f"aeRecent{tg}ReversedPct{s}Hi"] == rd.num(100 * hi, 0)
        assert x["model_bootstrap"]["n_boot"] == x["family_bootstrap"]["n_boot"] == 9999
        assert x["family_bootstrap"]["n_families"] == 3


@needs_json
def test_macro_module_reads_the_results():
    m = mr.release_dates_macros(mr.release_dates_results(ROOT))
    assert sorted(m) == sorted(mr.NAMES)
    for k, v in m.items():
        assert re.fullmatch(r"(\$-\$)?\d+(\.\d+)?|--", v), (k, v)
