"""paper/macros_closed_ladder.py: the values the manuscript prints come out of the results files with the
manuscript's names and values."""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
import macros_closed_ladder as mw  # noqa: E402

PENDING = os.path.join(ROOT, "paper", "nmi", "pending_nmi.tex")


@pytest.fixture(scope="module")
def macros():
    R = mw.closed_ladder_results(ROOT)
    if R is None:
        pytest.skip("results/frontier_panel/closed_w.json absent")
    return mw.closed_ladder_macros(R)


def test_every_hand_value_is_a_macro_with_the_same_value(macros):
    """Each cw, fsCheck and lad value of a file of hand-read values, when present, is produced and equal to it
    (the file holds only the four published sizes; no minimum count)."""
    if not os.path.exists(PENDING):
        pytest.skip("pending_nmi.tex absent")
    text = open(PENDING, encoding="utf-8").read()
    hand = dict(re.findall(r"\\providecommand\{\\((?:cw|fsCheck|lad)[A-Za-z]+)\}\{\\nm\{(.*?)\}\}\s*$", text, re.M))
    missing = sorted(set(hand) - set(macros))
    assert not missing, missing
    differ = {k: (hand[k], macros[k]) for k in hand if hand[k] != macros[k]}
    assert not differ, differ


def test_names_and_formats(macros):
    for k, v in macros.items():
        assert re.fullmatch(r"(cw|fsCheck|lad)[A-Za-z]+", k)
        assert re.fullmatch(r"(\$-\$)?[0-9]+(\{,\}[0-9]{3})*(\.[0-9])?", v), (k, v)


def test_f1_keeps_the_sign_of_a_value_that_rounds_to_zero():
    assert mw.f1(-0.0003) == "$-$0.0"
    assert mw.f1(0.04) == "0.0"
    assert mw.f1(-1.94) == "$-$1.9"


def test_one_refuses_ambiguous_rows():
    with pytest.raises(ValueError):
        mw._one([{"a": 1}, {"a": 1}], a=1)
    with pytest.raises(ValueError):
        mw._one([{"a": 2}], a=1)


def test_published_sizes_from_ladder_json(macros):
    """DeepSeek's published size (V4.1-Flash's: 552B total, 8B active for input, 16B for output) and
    Mistral Large 3's 41B active, read from ladder.json's sizes, which carry the page and read date; V4-Flash's 284 stays
    as the predecessor record, marked not used."""
    assert (macros["ladSizeDeepSeekTotalB"], macros["ladSizeDeepSeekActiveInB"], macros["ladSizeDeepSeekActiveOutB"],
            macros["ladSizeMistralLargeActiveB"], macros["ladSizeDeepSeekB"]) == ("552", "8", "16", "41", "284")
    sz = mw.closed_ladder_results(ROOT)["ladder"]["sizes"]
    assert sz["deepseek-v4.1-flash-1017"]["url"] == "https://api-docs.deepseek.com/news/news260910"
    assert sz["mistral-large-2512"]["url"] == "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512"
    assert sz["deepseek-v4.1-flash-1017"]["predecessor"]["used"] is False


def test_sizes_refuse_a_record_without_its_page(macros):
    import copy
    R0 = mw.closed_ladder_results(ROOT)
    R = copy.deepcopy(R0)
    R["ladder"]["sizes"]["mistral-large-2512"]["published"] = False
    with pytest.raises(ValueError, match="no published size"):
        mw.lad_macros(R)
    R = copy.deepcopy(R0)
    R["ladder"]["sizes"]["deepseek-v4.1-flash-1017"]["predecessor"]["used"] = True
    with pytest.raises(ValueError, match="not marked as not used"):
        mw.lad_macros(R)
