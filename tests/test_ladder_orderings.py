"""Tests of scripts/ladder_orderings.py and of its recount in
scripts/verify_ladder_analysis.py, on synthetic numbers: the reversal count with ties as the panel counts them, the
pooled count and pooled share, the bootstrap reproducible by its seed, the macro text, the note-file section, the
forms of unreadable replies, and the checker's own pair loop against the module's."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "paper"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ladder_orderings as lo  # noqa: E402
import macros_ladder_orderings as mlo  # noqa: E402
import verify_ladder_analysis as vla  # noqa: E402
from src import decision_directions as dd  # noqa: E402

KEYS4 = ["a", "b", "c", "d"]


def test_reversal_count_with_ties():
    """a > b under CoT - requested and b > a under CoT - forced: reversed. c and d tie exactly under the first gain and
    a and c differ by 1e-10 pp under the second: ties, neither reversed nor kept, and still among the pairs."""
    g1 = [10.0, 5.0, 3.0, 3.0]
    g2 = [1.0, 2.0, 1.0 + 1e-10, 0.5]
    r = lo.reversals(KEYS4, g1, g2)
    assert r["pairs"] == 6
    assert r["ties"] == 2
    assert sorted(map(tuple, r["tied_pairs"])) == [("a", "c"), ("c", "d")]
    # a-b reversed (10 > 5, 1 < 2); b-c: 5 > 3 and 2 > 1: kept; b-d: kept; a-d: 10 > 3, 1 > 0.5: kept
    assert [tuple(p) for p in r["reversed_pairs"]] == [("a", "b")]
    assert r["reversed"] == 1 and r["share"] == pytest.approx(1 / 6)


def test_a_tie_never_counts_as_a_reversal_even_against_a_large_gap():
    g1 = [0.0, 0.0]
    g2 = [50.0, -50.0]
    r = lo.reversals(["x", "y"], g1, g2)
    assert (r["reversed"], r["ties"], r["pairs"]) == (0, 1, 1)


def test_exact_ratios_count_like_floats():
    """The ladder's means are exact ratios; the count is the same as on their floats."""
    g1 = [Fraction(1, 3), Fraction(2, 3), Fraction(1, 3)]
    g2 = [Fraction(5, 7), Fraction(1, 7), Fraction(2, 7)]
    r = lo.reversals(["x", "y", "z"], g1, g2)
    r2 = lo.reversals(["x", "y", "z"], [float(v) for v in g1], [float(v) for v in g2])
    assert (r["reversed"], r["ties"]) == (r2["reversed"], r2["ties"]) == (2, 1)    # x-y and y-z reversed; x-z tied


def test_pooled_count_sums_reversed_and_pairs():
    def bt(n):        # threshold rows: n pairs above each threshold, one fewer reversed
        return [{"min_gap_pp": th, "pairs_both_above": n, "reversed": n - 1} for th in lo.GAP_THRESHOLDS]
    blocks = {"gsm8k": {"pairs": 15, "reversed": 2, "ties": 0, "by_threshold": bt(5)},
              "math": {"pairs": 15, "reversed": 4, "ties": 1, "by_threshold": bt(6)},
              "bbh": {"pairs": 10, "reversed": 3, "ties": 0, "by_threshold": bt(3)}}
    p = lo.pooled_counts(blocks)
    assert (p["pairs"], p["reversed"], p["ties"]) == (40, 9, 1)
    assert p["share"] == pytest.approx(9 / 40)
    assert all((x["pairs_both_above"], x["reversed"]) == (14, 11) for x in p["by_threshold"])


def _blocks(rng):
    keys = [f"m{i}" for i in range(7)]
    out = {}
    for t, drop in (("gsm8k", None), ("math", None), ("bbh", "m0")):        # one model absent on one benchmark
        ks = [k for k in keys if k != drop]
        out[t] = (ks, rng.normal(size=len(ks)), rng.normal(size=len(ks)))
    return out


def test_pooled_share_on_identity_equals_the_summed_counts():
    blocks = _blocks(np.random.default_rng(1))
    union = list(blocks["gsm8k"][0])
    parts, counts = [], {}
    for t, (ks, g1, g2) in blocks.items():
        Rm = dd.reversal_matrix(g1, g2)
        idx = {k: i for i, k in enumerate(ks)}
        parts.append((Rm, np.array([idx.get(k, -1) for k in union])))
        counts[t] = lo.reversals(ks, g1, g2)
    pc = lo.pooled_counts(counts)
    assert pc["pairs"] == 21 + 21 + 15
    assert lo.pooled_share_on(parts, np.arange(len(union))) == pytest.approx(pc["share"])
    b = lo.pooled_bootstrap(blocks, 0, np.random.default_rng(0))
    assert b["union"] == 7 and b["share"] == pytest.approx(pc["share"])


def test_pooled_share_on_leaves_out_a_model_with_its_own_copy():
    """Two copies of model 0 and one of model 1: two pairs (each copy with model 1), as share_on counts them."""
    Rm = np.array([[False, True], [True, False]])
    pos = np.array([0, 1])
    assert lo.pooled_share_on([(Rm, pos)], np.array([0, 0, 1])) == pytest.approx(1.0)
    # the single-benchmark case agrees with src/decision_directions.share_on
    assert lo.pooled_share_on([(Rm, pos)], np.array([0, 0, 1])) == dd.share_on(Rm, np.array([0, 0, 1]))
    assert np.isnan(lo.pooled_share_on([(Rm, pos)], np.array([1, 1])))


def test_single_benchmark_pooled_bootstrap_equals_reversal_uncertainty():
    """With one benchmark, the pooled resampling is src/decision_directions.reversal_uncertainty's model bootstrap,
    draw for draw, under the same seed."""
    rng = np.random.default_rng(5)
    ks = [f"m{i}" for i in range(9)]
    g1, g2 = rng.normal(size=9), rng.normal(size=9)
    a = lo.pooled_bootstrap({"gsm8k": (ks, g1, g2)}, 500, np.random.default_rng(42))
    b = dd.reversal_uncertainty(g1, g2, ["f"] * 9, 500, np.random.default_rng(42))
    assert a["model_bootstrap"]["ci95"] == b["model_bootstrap"]["ci95"]


def test_bootstrap_reproducible_by_seed():
    blocks = _blocks(np.random.default_rng(3))
    a = lo.pooled_bootstrap(blocks, 400, lo.seed("test/pooled"))
    b = lo.pooled_bootstrap(blocks, 400, lo.seed("test/pooled"))
    c = lo.pooled_bootstrap(blocks, 400, lo.seed("test/other"))
    assert a == b
    assert a["model_bootstrap"]["ci95"] != c["model_bootstrap"]["ci95"] or a["model_bootstrap"]["n_undefined"] != \
        c["model_bootstrap"]["n_undefined"]
    ks, g1, g2 = blocks["gsm8k"]
    u1 = dd.reversal_uncertainty(g1, g2, ["f"] * len(ks), 300, lo.seed("test/one"))
    u2 = dd.reversal_uncertainty(g1, g2, ["f"] * len(ks), 300, lo.seed("test/one"))
    assert u1["model_bootstrap"] == u2["model_bootstrap"]


def test_checker_pair_loop_agrees_with_the_module():
    """The checker's own pair loop (scripts/verify_ladder_analysis.own_pairs) and the module's count (src/robustness)
    agree on random gains with planted ties."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(3, 12))
        keys = [f"m{i}" for i in range(n)]
        g1 = list(np.round(rng.normal(size=n), 1))          # rounding plants ties
        g2 = list(np.round(rng.normal(size=n), 1))
        a = lo.reversals(keys, g1, g2)
        b = vla.own_pairs(keys, g1, g2)
        assert (a["reversed"], a["pairs"], a["ties"]) == (b["reversed"], b["pairs"], b["ties"])
        assert {tuple(p) for p in a["reversed_pairs"]} == b["rev_set"]


def test_checker_bootstrap_is_close_to_the_module_under_another_seed():
    blocks = _blocks(np.random.default_rng(4))
    a = lo.pooled_bootstrap(blocks, 4000, np.random.default_rng(1))["model_bootstrap"]["ci95"]
    b = vla.own_boot([(ks, list(g1), list(g2)) for ks, g1, g2 in blocks.values()], 4000, np.random.default_rng(2))
    assert all(abs(x - y) <= 0.03 for x, y in zip(a, b))


def test_macro_text():
    assert lo.mnum(Fraction(-1, 25), 1) == "0.0"                 # rounds to zero: no sign
    assert lo.mnum(Fraction(-5, 4), 1) == "$-$1.2"               # half-even on the exact value
    assert lo.mnum(Fraction(5, 4), 1) == "1.2"
    assert lo.mnum(Fraction(7, 4), 1) == "1.8"
    assert lo.mnum(1751, 0) == "1{,}751"
    assert lo.pct0(2, 15) == "13" and lo.pct0(4, 15) == "27" and lo.pct0(1, 8) == "12"
    for v, d in ((Fraction(-5, 4), 1), (Fraction(-1, 25), 1), (1751, 0), (0.125, 2), (Fraction(40, 3), 0)):
        assert lo.mnum(v, d) == vla.ord_fmt(v, d)


def test_macros_module_checks_names():
    good = {k: "1" for k in mlo.REQUIRED}
    assert mlo.ladder_orderings_macros({"macros": good}) == good
    with pytest.raises(ValueError):
        mlo.ladder_orderings_macros({"macros": {**good, "lad2x": "1"}})
    with pytest.raises(ValueError):
        mlo.ladder_orderings_macros({"macros": {**good, "otherName": "1"}})
    with pytest.raises(ValueError):
        mlo.ladder_orderings_macros({"macros": {k: v for k, v in good.items() if k != "ladRevGsm"}})


def test_reply_forms():
    f = lo.reply_form
    assert f("\\boxed{False}") == "box"
    assert f("1:00 pm\n\\boxed{1:00}") == "box"
    assert f("(A)") == "bare option" and f("(D) Named Entities") == "bare option"
    assert f("10") == "bare number" and f(" -126 ") == "bare number" and f("\\frac{15}{8}") == "bare number"
    assert f("-4, -1") == "bare number" and f("0.09") == "bare number"
    assert f("United States") == "bare words" and f("1:00 pm") == "bare words" and f("No") == "bare words"


def test_hand_count_section_is_replaced_in_place_and_keeps_line_endings(tmp_path):
    p = tmp_path / "ladder_note.md"
    p.write_bytes(b"# Note\r\n\r\n## Against the panel\r\n\r\n- x\r\n\r\n## Footnotes to carry\r\n\r\n- F1.\r\n")
    sec = [lo.SECTION, "", "- one"]
    lo.put_section(p, sec)
    first = p.read_bytes()
    assert b"\r\n" in first and b"\n" not in first.replace(b"\r\n", b"")
    txt = first.decode().replace("\r\n", "\n")
    assert txt.index(lo.SECTION) < txt.index("## Footnotes to carry")
    lo.put_section(p, sec)
    assert p.read_bytes() == first                                 # a second run changes nothing
    lo.put_section(p, [lo.SECTION, "", "- two"])
    txt = p.read_text(encoding="utf-8")
    assert "- two" in txt and "- one" not in txt and txt.count(lo.SECTION) == 1


def test_checker_hand_counts():
    assert vla.hand_counts_from_file(ROOT) == {"gsm8k": (2, 15), "math": (4, 15), "bbh": (2, 15), "hotpot": (12, 15)}


def test_split_parts_add_to_the_shift():
    """The checker's split at 16 tokens: long + short = requested - forced, per configuration and over the draws."""
    import pandas as pd
    rows = []
    for c in ("c1", "c2"):
        for i in range(4):
            rows.append((c, f"i{i}", "direct_answer", int(i % 2), 30 if i < 2 else 5))
            rows.append((c, f"i{i}", "forced_direct", int(i == 3), 3))
            rows.append((c, f"i{i}", "zero_shot_cot", 1, 200))
    z = pd.DataFrame(rows, columns=["config_id", "item_id", "arm", "strict_correct", "n_output_tokens"])
    z["content_correct"] = z["strict_correct"]
    out = vla.split_parts(z, ["c1", "c1", "c2"])
    # per configuration: long rows i0, i1 (requested 0, 1; forced 0, 0) -> +1; short rows i2, i3 (1, 0; 0, 1) -> 0
    assert out["strict"]["long"] == Fraction(100, 4) and out["strict"]["short"] == 0
    assert out["strict"]["shift"] == Fraction(100, 4)
    assert out["long_share"] == Fraction(1, 2)


needs_json = pytest.mark.skipif(not (lo.OUT / f"{lo.STEM}.json").exists(), reason="ladder_orderings.json not written")


@needs_json
def test_written_counts_hold_together():
    import json
    R = json.loads((lo.OUT / f"{lo.STEM}.json").read_text(encoding="utf-8"))
    for r in lo.READINGS:
        for s in lo.SCORES:
            six = R["reversals"]["six"][r][s]["discovery"]
            assert all(six[t]["pairs"] == 15 for t in lo.TASKS)
            assert six["pooled"]["pairs"] == 45
            assert six["pooled"]["reversed"] == sum(six[t]["reversed"] for t in ("gsm8k", "math", "bbh"))
            big = R["reversals"]["big"][r][s]
            assert all(big[t]["pairs"] == 6 for t in lo.TASKS)
            comb = R["reversals"]["combined"][r][s]
            assert comb["gsm8k"]["models"] == 35 and comb["bbh"]["models"] == 34
            assert comb["pooled"]["pairs"] == 595 + 595 + 561
    # the 29-model counts are the paper's (content/1.6.0 and hotpot-content/1.5.0)
    p = R["reversals"]["panel"]["content"]
    assert (p["gsm8k"]["reversed"], p["math"]["reversed"], p["bbh"]["reversed"], p["hotpot"]["reversed"]) == \
        (87, 80, 85, 84)


@needs_json
def test_threshold_rows_hold_together():
    """Each block's pairs with both gaps above 1, 2, 5 and 10 pp shrink as the threshold rises, the
    reversed among them never exceed them, and the pooled rows are the sums over the first three benchmarks."""
    import json
    R = json.loads((lo.OUT / f"{lo.STEM}.json").read_text(encoding="utf-8"))
    n_blocks = 0
    for grp in ("six", "big", "combined"):
        for r in lo.READINGS:
            for s in lo.SCORES:
                blk = R["reversals"][grp][r][s]
                blk = blk.get("discovery", blk)
                for t in list(lo.TASKS) + ["pooled"]:
                    bt = blk[t]["by_threshold"]
                    assert [x["min_gap_pp"] for x in bt] == list(lo.GAP_THRESHOLDS)
                    pairs = [x["pairs_both_above"] for x in bt]
                    rev = [x["reversed"] for x in bt]
                    assert pairs == sorted(pairs, reverse=True) and rev == sorted(rev, reverse=True)
                    assert all(0 <= v <= n <= blk[t]["pairs"] for v, n in zip(rev, pairs))
                    assert rev[0] <= blk[t]["reversed"]
                for i in range(len(lo.GAP_THRESHOLDS)):
                    for k in ("pairs_both_above", "reversed"):
                        assert blk["pooled"]["by_threshold"][i][k] == sum(
                            blk[t]["by_threshold"][i][k] for t in ("gsm8k", "math", "bbh"))
                n_blocks += 1
    assert n_blocks == 3 * len(lo.READINGS) * len(lo.SCORES)
