"""Tests for scripts/ladder_arms_table.py: the table reads the saved values, the HotpotQA content basis is the reversal
counts' (discovery draws), the panel medians are the saved ones, rounding is the ladder's, and the files on disk are
current."""
import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ladder_arms_table as lat  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / lat.LADDER).exists(), reason="ladder.json absent")


@pytest.fixture(scope="module")
def T():
    return lat.build()


@pytest.fixture(scope="module")
def lad():
    return json.loads((ROOT / lat.LADDER).read_text(encoding="utf-8"))


def test_one_row_per_model_and_benchmark(T, lad):
    assert len(T["rows"]) == len(lad["models"]) * len(lat.TASKS)
    assert {(r["task"], r["key"]) for r in T["rows"]} == {(t, k) for t in lat.TASKS for k in lad["models"]}
    for r in T["rows"]:
        assert set(r["cells"]) == {(s, m) for s in lat.SCORINGS for m in lat.METRICS}


def test_values_are_the_saved_discovery_values(T, lad):
    order = json.loads((ROOT / lat.ORDERINGS).read_text(encoding="utf-8"))
    for r in T["rows"]:
        t, k = r["task"], r["key"]
        for (s, m), q in r["cells"].items():
            disc = lad["models"][k]["tasks"][t]["cut"][s]["discovery"]
            assert abs(float(q) - lat._saved(disc, m)) < 1e-9
            assert abs(float(q) - order["values"]["cut"][t][k][s]["discovery"][m]) < 1e-9
            rec = lad["placement"][k][t]["cut"][s][m]
            if (t, s) == ("hotpot", "content"):
                assert rec["basis"] == "pooled"          # the placements' basis; the table keeps discovery
            else:
                assert rec["basis"] == "discovery" and abs(float(q) - rec["value"]) < 1e-9


def test_gains_are_differences_of_the_arms(T):
    for r in T["rows"]:
        for s in lat.SCORINGS:
            c = r["cells"]
            assert abs(float(c[(s, "cot")] - c[(s, "requested")] - c[(s, "cot_minus_requested")])) < 1e-9
            assert abs(float(c[(s, "cot")] - c[(s, "forced")] - c[(s, "cot_minus_forced")])) < 1e-9


def test_panel_medians(T, lad):
    first = next(iter(lad["placement"]))
    hq = json.loads((ROOT / lat.HOTPOT_CONTENT).read_text(encoding="utf-8"))["arm_effect"]["content"]["per_model"]
    for (t, s, m), d in T["medians"].items():
        if (t, s) == ("hotpot", "content"):
            keys = [x["model_key"] for x in lad["panel"]["hotpot"]]
            col = {"requested": [hq[k]["requested"] for k in keys], "forced": [hq[k]["forced"] for k in keys],
                   "cot": [hq[k]["cot"] for k in keys],
                   "cot_minus_requested": [hq[k]["cot"] - hq[k]["requested"] for k in keys],
                   "cot_minus_forced": [hq[k]["cot"] - hq[k]["forced"] for k in keys]}[m]
            assert abs(float(d["value"]) - statistics.median(col)) < 1e-9
            assert d["n"] == len(keys) == 29
        else:
            rec = lad["placement"][first][t]["cut"][s][m]
            assert abs(float(d["value"]) - rec["panel_median"]) < 1e-9
            assert d["n"] == rec["panel_n"]


def test_rounding_is_the_ladders():
    assert lat.fmt(Fraction(885, 100)) == "8.8"            # halfway: the even neighbour
    assert lat.fmt(Fraction(5875, 100)) == "58.8"
    assert lat.fmt(Fraction(-13, 10), "$-$") == "$-$1.3"
    assert lat.fmt(Fraction(-1, 100)) == "-0.0"            # a negative that rounds to zero keeps its sign
    assert lat.fmt(Fraction(1, 3)) == "0.3"


def test_exact_reads_the_grid_and_reports_off_grid_values():
    edges = []
    G = Fraction(240)
    assert lat.exact(55.0375, G, edges, "x") == Fraction(13209, 240)
    assert lat.exact(float(Fraction(1, 480)), G, edges, "x") == Fraction(1, 480)    # median of an even count
    v = lat.exact(0.1234567, G, edges, "y")
    assert isinstance(v, float) and len(edges) == 1 and edges[0].startswith("y:")


def test_tex_shape(T):
    text = lat.tex(T)
    body = [ln for ln in text.splitlines() if ln.endswith("\\\\") and "multicolumn" not in ln]
    ncols = len(lat.SCORINGS) * len(lat.METRICS) + 1
    assert all(ln.count(" & ") == ncols - 1 for ln in body)
    assert text.count("Panel median") == len(lat.TASKS)
    assert "\\begin{tabular}" in text and "\\end{tabular}" in text and "\\caption" not in text
    assert "$-$1.3" in text                                  # negative values printed as they are


def test_files_on_disk_are_current(T):
    assert (ROOT / lat.TEX).read_text(encoding="utf-8") == lat.tex(T)
    md = lat.md(T)
    assert (ROOT / lat.MD).read_text(encoding="utf-8") == md
    assert (ROOT / lat.MD_CLEAN).read_text(encoding="utf-8") == md
    assert "[@" not in md and "\\cite" not in md
