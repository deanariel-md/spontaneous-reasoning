"""src/attr_flips.py, scripts/run_attr_flips.py and paper/macros_attr_flips.py."""
import itertools
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "paper")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import attr_flips as af  # noqa: E402
from src import baseline_mixture as bm  # noqa: E402
import macros_attr_flips as maf  # noqa: E402

RES = os.path.join(ROOT, "results", "attr_flips", "attr_flips.json")
NUMBERS = os.path.join(ROOT, "paper", "numbers.tex")
needs_results = pytest.mark.skipif(not os.path.exists(RES), reason="results/attr_flips/attr_flips.json not built")


def _tex():
    txt = open(NUMBERS, encoding="utf-8").read()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^\n]*)\}", txt))


# ---------------------------------------------------------------------------- the counts on a hand-worked case


def test_part_flips_hand_case():
    # models a..d plus e, f; g_req, long and short parts chosen so every count can be worked out by hand
    names = ["a", "b", "c", "d", "e", "f"]
    g_req = np.array([10.0, 8.0, 5.0, 0.0, 40.0, 36.0])
    L = np.array([0.0, 5.0, 0.0, 0.0, -5.0, 0.0])
    S = np.array([0.0, 0.0, 6.0, -1.0, -6.0, 0.0])
    g_fd = g_req + L + S                      # [10, 13, 11, -1, 29, 36]
    r = af.part_flips(g_req, g_fd, L, S, names)
    # observed: (a,b) 2 vs -3, (a,c) 5 vs -1, (e,f) 4 vs -7
    assert r["pair_lists"]["observed"] == [["a", "b"], ["a", "c"], ["e", "f"]]
    # long alone (g_req + L = [10, 13, 5, 0, 35, 36]): (a,b) 2 vs -3; (e,f) 4 vs -1
    assert r["pair_lists"]["long"] == [["a", "b"], ["e", "f"]]
    # short alone (g_req + S = [10, 8, 11, -1, 34, 36]): (a,c) 5 vs -1, (b,c) 3 vs -3, (e,f) 4 vs -2
    assert r["pair_lists"]["short"] == [["a", "c"], ["b", "c"], ["e", "f"]]
    assert r["pair_lists"]["both"] == [["e", "f"]]
    assert (r["pairs"], r["observed"], r["long"], r["short"], r["both"]) == (15, 3, 2, 3, 1)
    assert (r["observed_and_long"], r["observed_and_short"], r["observed_neither"]) == (2, 2, 0)
    assert (r["long_not_observed"], r["short_not_observed"], r["both_not_observed"]) == (0, 1, 0)
    assert r["ties"] == {"observed": 0, "long": 0, "short": 0}


def test_part_flips_neither_alone():
    # each part alone falls short of the gap; together they cross it
    g_req = np.array([3.0, 0.0])
    L, S = np.array([-2.0, 0.0]), np.array([-2.0, 0.0])
    r = af.part_flips(g_req, g_req + L + S, L, S)
    assert (r["observed"], r["long"], r["short"], r["both"], r["observed_neither"]) == (1, 0, 0, 0, 1)


def test_ties_are_not_reversed_and_stay_in_the_denominator():
    g_req = np.array([5.0, 5.0, 0.0])          # a and b tied under the requested baseline
    L = np.array([0.0, 3.0, 0.0])
    S = np.array([0.0, 0.0, 0.0])
    r = af.part_flips(g_req, g_req + L + S, L, S)
    assert r["pairs"] == 3
    assert r["observed"] == 0 and r["long"] == 0
    assert r["ties"]["observed"] == 1 and r["ties"]["short"] == 1
    # a gap within 1e-9 pp is a tie
    g2 = np.array([1.0, 1.0 + 5e-10])
    r2 = af.part_flips(g2, g2[::-1].copy(), np.array([2.5e-10, -2.5e-10]), np.array([2.5e-10, -2.5e-10]))
    assert r2["observed"] == 0 and r2["ties"]["observed"] == 1


def test_parts_must_add_to_D():
    g_req = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="do not add to D"):
        af.part_flips(g_req, g_req + np.array([1.0, 0.0]), np.array([0.5, 0.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="not finite"):
        af.part_flips(g_req, g_req, np.array([np.nan, 0.0]), np.array([0.0, 0.0]))


def test_both_parts_always_inside_observed_random():
    rng = np.random.default_rng(7)
    for _ in range(200):
        M = 8
        g = rng.normal(20, 10, M)
        L, S = rng.normal(0, 8, M), rng.normal(0, 8, M)
        r = af.part_flips(g, g + L + S, L, S)
        assert r["both_not_observed"] == 0
        assert r["observed_and_long"] + r["observed_and_short"] - r["both"] + r["observed_neither"] == r["observed"]


# ---------------------------------------------------------------------------- the definition on synthetic rows


def test_long_part_is_a_forced_arm_changed_only_on_long_rows():
    """g_R + L equals the gain over a forced arm that equals the requested arm on short rows (and g_R + S the gain over
    one that equals it on long rows); the parts come from src/baseline_mixture.decomposition, weights included."""
    rng = np.random.default_rng(3)
    M, n = 5, 400
    m = np.repeat(np.arange(M), n)
    lf = rng.random(M * n) < np.repeat(rng.uniform(0.1, 0.8, M), n)
    r = (rng.random(M * n) < 0.5).astype(float)
    f = (rng.random(M * n) < 0.4).astype(float)
    c = (rng.random(M * n) < 0.8).astype(float)
    wt = rng.integers(1, 3, M * n).astype(float)
    d = bm.decomposition(m, lf, r, f, c, M, wt)["per_model"]

    def mean(v):
        return np.array([100 * np.sum((v * wt)[m == j]) / np.sum(wt[m == j]) for j in range(M)])
    g_req, g_fd = mean(c - r), mean(c - f)
    L, S = np.array(d["part_long"]), np.array(d["part_short"])
    np.testing.assert_allclose(g_req + L, mean(c - np.where(lf, f, r)), atol=1e-10)
    np.testing.assert_allclose(g_req + S, mean(c - np.where(lf, r, f)), atol=1e-10)
    out = af.part_flips(g_req, g_fd, L, S)
    assert out["sum_check_max_abs_pp"] < 1e-10


def test_distribution():
    d = af.distribution([3.0, -1.0, 0.0, 2.0, np.nan])
    assert (d["n"], d["median"], d["min"], d["max"]) == (4, 1.0, -1.0, 3.0)
    assert (d["below_zero"], d["zero"], d["above_zero"]) == (1, 1, 2)
    assert af.distribution([np.nan])["median"] is None


# ---------------------------------------------------------------------------- the published numbers


@needs_results
def test_observed_reproduces_published_counts():
    R = json.load(open(RES, encoding="utf-8"))
    tex = _tex() if os.path.exists(NUMBERS) else {}
    pub = {"gsm8k": "bmixGsmRevObs", "math": "bmixMathRevObs", "bbh": "bmixBbhRevObs",
           "hotpot_cap64": "aeHqReversed", "hotpot_cap16": "aeHqReversed"}
    for u, mac in pub.items():
        for sc, S in (("strict", ""), ("content", "Content")):
            B = R["tasks"][u]["scores"][sc]
            assert B["checks"]["observed_equals_published_file"]["equal"]
            if tex:
                assert str(B["counts"]["observed"]) == tex[mac + S].strip(), (u, sc)
            assert B["checks"]["parts_add_to_D_max_abs_pp"] < 1e-8
            assert B["checks"]["parts_equal_stored_max_abs_pp"] < 1e-9
            assert B["counts"]["both_not_observed"] == 0


@needs_results
def test_pinned_counts():
    """The pinned counts (update when a scoring fix moves them)."""
    want = {("gsm8k", "strict"): (83, 71, 18, 3), ("gsm8k", "content"): (87, 78, 14, 3),
            ("math", "strict"): (84, 75, 18, 3), ("math", "content"): (80, 76, 9, 1),
            ("bbh", "strict"): (68, 32, 65, 15), ("bbh", "content"): (85, 79, 10, 5),
            ("hotpot_cap64", "strict"): (142, 2, 140, 1), ("hotpot_cap64", "content"): (84, 4, 78, 1),
            ("hotpot_cap16", "strict"): (142, 39, 119, 20), ("hotpot_cap16", "content"): (84, 50, 47, 23)}
    R = json.load(open(RES, encoding="utf-8"))
    for (u, sc), w in want.items():
        c = R["tasks"][u]["scores"][sc]["counts"]
        assert (c["observed"], c["long"], c["short"], c["both"]) == w, (u, sc)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "processed", "content", "gsm8k_content.parquet")),
                    reason="GSM8K content table not present")
@needs_results
def test_recompute_gsm8k_from_data():
    import run_attr_flips as raf
    R = raf.run(units=("gsm8k",))
    S = json.load(open(RES, encoding="utf-8"))
    for sc in ("strict", "content"):
        a, b = R["tasks"]["gsm8k"]["scores"][sc]["counts"], S["tasks"]["gsm8k"]["scores"][sc]["counts"]
        for k in ("pairs", "observed", "long", "short", "both", "observed_neither"):
            assert a[k] == b[k], (sc, k)
        assert a["pair_lists"] == b["pair_lists"]


# ---------------------------------------------------------------------------- the macro module


@needs_results
def test_macros_values_names_and_no_clash():
    R = maf.attr_flips_results(ROOT)
    M = maf.attr_flips_macros(R)
    assert len(M) == 155
    assert all(re.fullmatch(r"attr[A-Z][A-Za-z]*", k) for k in M)
    c = R["tasks"]["gsm8k"]["scores"]["strict"]["counts"]
    assert M["attrGsmObsRev"] == str(c["observed"]) and M["attrGsmLongRev"] == str(c["long"])
    assert M["attrGsmShortRev"] == str(c["short"]) and M["attrGsmBothRev"] == str(c["both"])
    assert M["attrHqSixtyFourObsRevContent"] == M["attrHqSixteenObsRevContent"]
    sp = R["tasks"]["bbh"]["scores"]["content"]["short_part"]
    assert M["attrBbhShortMedianContent"] == f"{sp['median']:.1f}".replace("-", "$-$")
    assert M["attrBbhObsRevPct"] == f"{100 * R['tasks']['bbh']['scores']['strict']['counts']['observed'] / 378:.0f}"
    if os.path.exists(NUMBERS):
        names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(NUMBERS, encoding="utf-8").read())
        mk = open(os.path.join(ROOT, "paper", "make_numbers.py"), encoding="utf-8").read()
        if "macros_attr_flips" in mk:
            for k in M:
                assert names.count(k) == 1, f"{k} defined {names.count(k)} times in numbers.tex"
            return
        assert not (set(M) & set(names)), "a macro name is already in numbers.tex"


def test_macros_empty_without_results():
    assert maf.attr_flips_macros(None) == {}


@needs_results
def test_summary_prose():
    for f in ("summary.md", "summary_clean.md"):
        txt = open(os.path.join(ROOT, "results", "attr_flips", f), encoding="utf-8").read()
        assert "\u2014" not in txt
        for w in ("pre-specified", "locked", "rather than"):
            assert w not in txt.lower(), w
