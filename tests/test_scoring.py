"""Tests for src/scoring.py: normalization, strict and lenient extraction, compliance, scoring.

The format specs used here are the candidate specs or small specs built inside the tests; no
test asserts that any particular format is the selected one.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src import scoring as sc
from src.scoring import (CANDIDATE_SPECS, FormatSpec, compliance, lenient_extract, normalize_number,
                         normalize_text, score_outputs, strict_extract)

SENTENCE = CANDIDATE_SPECS["the_answer_is"]
HASH = CANDIDATE_SPECS["hash_n"]
COLON = CANDIDATE_SPECS["answer_colon"]
BOXED = CANDIDATE_SPECS["boxed"]


# ---------------------------------------------------------------------------------------
# normalize_number
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("18", "18"),
    ("18.0", "18"),
    ("18.00", "18"),
    ("18.50", "18.5"),
    ("0.50", "0.5"),
    ("0.0", "0"),
    ("007", "7"),
    ("1000", "1000"),
    ("1,000", "1000"),
    ("12,345,678", "12345678"),
    ("1,000.25", "1000.25"),
    ("-5", "-5"),
    ("+5", "5"),
    ("-0", "0"),
    ("-0.00", "0"),
    ("-0.50", "-0.5"),
    (".5", "0.5"),
    ("-.5", "-0.5"),
    ("$5", "5"),
    ("$ 5", "5"),
    ("$18.00", "18"),
    ("-$5", "-5"),
    ("$-5", "-5"),
    ("18.", "18"),
    ("$18.00.", "18"),
    ("  18  ", "18"),
    ("18\n", "18"),
    ("\u22125", "-5"),            # Unicode minus
    ("\u20135", "-5"),            # en dash glued to the digits
    ("\uff11\uff18", "18"),       # full-width digits
    ("123456789012345678901234567890", "123456789012345678901234567890"),
    ("0.1000000000000000000000000001", "0.1000000000000000000000000001"),  # no float rounding
])
def test_normalize_number_valid(raw, expected):
    assert normalize_number(raw) == expected


@pytest.mark.parametrize("raw", [
    "", " ", None, "abc", "--5", "-+5", "+-5", "1,00", "12,34", "1,0000", ",100", "1,,000",
    "1.2.3", "1..", "18%", "18 dollars", "3/4", "1e3", "one", "1 000", "$$5", "5$", "- 5",
    "\u20ac5", "1,000,00", ".", "-", "$",
])
def test_normalize_number_rejects(raw):
    assert normalize_number(raw) is None


def test_float_equivalence_by_canonical_string():
    assert normalize_number("18.0") == normalize_number("18") == normalize_number("18.00")
    assert normalize_number("1,000") == normalize_number("1000")
    assert normalize_number("18.0000001") != normalize_number("18")
    assert normalize_number("0.1") != normalize_number("0.10000000001")


def test_normalize_number_idempotent():
    for raw in ["1,000.50", "-0", "$7.", "007.100"]:
        once = normalize_number(raw)
        assert normalize_number(once) == once


# ---------------------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------------------
def test_normalize_text_basics():
    assert normalize_text(None) == ""
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    for dash in ["\u2212", "\u2012", "\u2013", "\u2010", "\u2011", "\ufe63", "\uff0d"]:
        assert normalize_text(f"x {dash}5") == "x -5"
    assert normalize_text("a\u2014b") == "a\u2014b"            # em dash untouched
    assert normalize_text("5\u00a0apples\u202f\u2009x") == "5 apples  x"
    assert normalize_text("1\u200b8\ufeff") == "18"            # zero-width removed
    assert normalize_text("\uff11\uff12\u0663") == "123"       # full-width and Arabic-Indic digits
    assert normalize_text("10\u00b3") == "10\u00b3"             # superscript untouched
    assert normalize_text("a\u2028b") == "a\nb"
    assert normalize_text(123) == "123"


# ---------------------------------------------------------------------------------------
# FormatSpec
# ---------------------------------------------------------------------------------------
def test_candidate_specs_are_labeled_as_candidates():
    assert set(CANDIDATE_SPECS) == {"the_answer_is", "hash_n", "answer_colon", "boxed",
                                    "answer_colon_markup", "the_answer_is_markup"}
    for name, spec in CANDIDATE_SPECS.items():
        assert spec.name == name
        assert spec.description.startswith("CANDIDATE, not a decision")


def test_spec_validation():
    with pytest.raises(ValueError):
        FormatSpec(name="x", template="no placeholder")
    with pytest.raises(ValueError):
        FormatSpec(name="x", template="<NUM> and <NUM>")
    with pytest.raises(ValueError):
        FormatSpec(name="x", template="a <NUM>", on_multiple="first")
    with pytest.raises(Exception):                    # invalid regex surfaces at construction
        FormatSpec(name="x", template="(unclosed <NUM>")


def test_spec_roundtrip_and_fingerprint():
    for spec in CANDIDATE_SPECS.values():
        assert sc.spec_from_dict(sc.spec_to_dict(spec)) == spec
    with pytest.raises(ValueError):
        sc.spec_from_dict({**sc.spec_to_dict(HASH), "bogus": 1})
    renamed = FormatSpec(**{**sc.spec_to_dict(HASH), "name": "other", "description": "other text"})
    assert sc.spec_fingerprint(renamed) == sc.spec_fingerprint(HASH)
    changed = FormatSpec(**{**sc.spec_to_dict(HASH), "allow_final_period": not HASH.allow_final_period})
    assert sc.spec_fingerprint(changed) != sc.spec_fingerprint(HASH)
    fps = {sc.spec_fingerprint(s) for s in CANDIDATE_SPECS.values()}
    assert len(fps) == len(CANDIDATE_SPECS)
    assert len(sc.spec_fingerprint(HASH)) == 64


def test_scorer_version_and_record():
    assert isinstance(sc.SCORER_VERSION, str) and sc.SCORER_VERSION.startswith("gsm8k-scorer/")
    rec = sc.scorer_record(HASH)
    assert rec["scorer_version"] == sc.SCORER_VERSION
    assert rec["format_spec"]["template"] == HASH.template
    assert rec["format_spec_sha256"] == sc.spec_fingerprint(HASH)


# ---------------------------------------------------------------------------------------
# strict_extract on the candidate specs
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("The answer is 18.", "18"),
    ("The answer is 18", "18"),
    ("Therefore, the answer is 18.", "18"),                     # leading words allowed
    ("THE ANSWER IS 18.", "18"),                                # case-insensitive
    ("reasoning\n\nThe answer is 1,000.\n\n  \n", "1000"),      # trailing blank lines ignored
    ("The answer is -3.", "-3"),
    ("The answer is 18.00.", "18"),
    ("The answer is 3.5.", "3.5"),
    ("The answer is\t18.", "18"),
    ("line1\r\nThe answer is 18.\r\n", "18"),
    ("The answer is \u22123.", "-3"),                           # Unicode minus
    ("The answer is \uff11\uff18.", "18"),                      # full-width digits
    ("The answer is 18 .", "18"),                               # whitespace before the final period is not text
    ("The answer is $18.", "18"),                               # 1.2.0: one "$", as normalize_number reads it
    ("The answer is $1,000.", "1000"),
])
def test_strict_sentence_compliant(text, expected):
    assert strict_extract(text, SENTENCE) == expected
    assert compliance(text, SENTENCE) is True


@pytest.mark.parametrize("text", [
    "",
    "   \n  ",
    "18",
    "The answer is 18.\nHope this helps!",       # not the last non-empty line
    "The answer is 18 apples.",                  # trailing text
    "The answer is €18.",                        # other currency signs are not numbers
    "The answer is 18 dollars.",                 # units stay non-compliant in 1.2.0
    "The answer is 18%",
    "The answer is: 18",
    "The answer is eighteen.",
    "The answer is 1,00.",
    "The answer is 12.3.4.",
    "The answer is 18..",
    "**The answer is 18**",                      # markdown wrapper counts as trailing text
    "The answer is 18, since 9 * 2 = 18.",
    "Answer: 18",
])
def test_strict_sentence_noncompliant(text):
    assert strict_extract(text, SENTENCE) is None
    assert compliance(text, SENTENCE) is False


@pytest.mark.parametrize("text, expected", [
    ("#### 18", "18"),
    ("Step 1.\nStep 2.\n#### 18", "18"),
    ("#### 1,000", "1000"),
    ("#### -3", "-3"),
    ("#### 3.50", "3.5"),
    ("#### 18\n\n", "18"),
    ("  #### 18  ", "18"),                  # surrounding whitespace is not text
    ("#### 18.", "18"),                     # 1.2.0: one final period
    ("#### $18", "18"),                     # 1.2.0: one "$"
])
def test_strict_hash(text, expected):
    assert strict_extract(text, HASH) == expected


@pytest.mark.parametrize("text", [
    "####18", "#### 18..", "#### 18 apples", "Therefore #### 18", "#### 18\nDone", "###  18",
    "#### $$18", "#### 18%", "#### 1,00", "## 18", "The answer is 18.",
])
def test_strict_hash_rejects(text):
    assert strict_extract(text, HASH) is None


def test_strict_answer_colon():
    assert strict_extract("blah\nAnswer: 42", COLON) == "42"
    assert strict_extract("Answer: -7", COLON) == "-7"
    assert strict_extract("answer: 42", COLON) is None            # case-sensitive spec
    assert strict_extract("Answer:42", COLON) is None
    assert strict_extract("Answer: 42.", COLON) == "42"            # 1.2.0: one final period
    assert strict_extract("Answer: $11", COLON) == "11"            # 1.2.0: one "$"
    assert strict_extract("Answer: $1,120.00.", COLON) == "1120"
    for text in ("Answer: 120 days.", "Answer: 17%", "Answer: 42..", "Answer: Mark has 12 tickets."):
        assert strict_extract(text, COLON) is None, text            # units, percent signs, sentences
    assert strict_extract("Final Answer: 42", COLON) is None      # leading text not allowed


def test_strict_boxed_only_when_the_spec_requests_it():
    text = "So the total is $\\boxed{18}$."
    assert strict_extract(text, BOXED) == "18"
    assert strict_extract(text, SENTENCE) is None
    assert strict_extract(text, HASH) is None
    assert lenient_extract(text) == "18"                         # lenient reads it as the last number
    assert strict_extract("\\boxed{1,000}", BOXED) == "1000"
    assert strict_extract("\\boxed{-4}", BOXED) == "-4"
    assert strict_extract("\\boxed{18} is wrong\nThe result", BOXED) is None   # not on last line
    assert strict_extract("\\boxed{\\$18}", BOXED) == "18"                      # 1.3.0: LaTeX-escaped dollar
    assert strict_extract("\\boxed{18 \\text{ apples}}", BOXED) is None


# --- strip_latex (scorer 1.3.0, after the audit of the boxed format screen) -------------------------
@pytest.mark.parametrize("text,expected", [
    ("So:\n$$\n\\boxed{18}\n$$", "18"),                          # Qwen3-8B's form
    ("So:\n\\[\n\\boxed{18}\n\\]", "18"),                        # Mistral-Small-3.1's form
    ("So:\n   \\[\n   \\boxed{18}\n   \\]\n", "18"),             # indented, trailing newline
    ("\\[\n\\boxed{\\$12.00}\n\\]", "12"),
    ("Total: \\boxed{$27$}", "27"),
    ("\\boxed{$8$} dollars in all.", "8"),
    ("$$\\boxed{18}$$", "18"),                                    # one line, as in 1.2.0
])
def test_strip_latex_accepts(text, expected):
    assert strict_extract(text, BOXED) == expected


@pytest.mark.parametrize("text", [
    "\\[\n\\boxed{80 \\text{ people}}\n\\]",                      # units stay non-compliant
    "Therefore, \\boxed{<17>} she will collect.",                 # copied placeholder brackets
    "\\boxed{18}\nI hope this helps!",                            # text after the boxed line
    "$$\n\\boxed{18}\n$$\nSo 18.",                                # text after the closing delimiter
    "\\boxed{25\\%}",
    "$$",
])
def test_strip_latex_rejects(text):
    assert strict_extract(text, BOXED) is None


def test_strip_latex_only_for_specs_that_set_it():
    plain = FormatSpec(**{**sc.spec_to_dict(BOXED), "strip_latex": False})
    assert strict_extract("$$\n\\boxed{18}\n$$", plain) is None
    assert strict_extract("\\boxed{\\$18}", plain) is None
    assert strict_extract("$$\n#### 18\n$$", HASH) is None        # no other candidate sets it
    assert not any(s.strip_latex for n, s in CANDIDATE_SPECS.items() if n != "boxed")
    assert sc.strip_latex("a\n$$\n\\]\n") == "a"


# ---------------------------------------------------------------------------------------
# Multiple answers and positions
# ---------------------------------------------------------------------------------------
def _spec(**kw):
    base = dict(name="t", template=r"result:[ \t]<NUM>", must_be_last_nonempty_line=False,
                leading_text_allowed=False, trailing_text_allowed=False)
    base.update(kw)
    return FormatSpec(**base)


def test_multiple_matches_policies():
    text = "result: 5\nsome words\nresult: 7"
    assert strict_extract(text, _spec(on_multiple="reject")) is None
    assert strict_extract(text, _spec(on_multiple="last")) == "7"
    assert strict_extract(text, _spec(on_multiple="unanimous")) is None
    same = "result: 7\nmore\nresult: 7.0"
    assert strict_extract(same, _spec(on_multiple="unanimous")) == "7"
    assert strict_extract(same, _spec(on_multiple="reject")) is None
    assert strict_extract("result: 5", _spec(on_multiple="reject")) == "5"


def test_last_line_spec_ignores_earlier_matches():
    text = "The answer is 5.\nwait, recompute\nThe answer is 7."
    assert strict_extract(text, SENTENCE) == "7"
    assert strict_extract("The answer is 7.\nbut maybe not", SENTENCE) is None


def test_two_matches_in_one_line_only_the_line_final_one_qualifies():
    text = "The answer is 5. The answer is 7."
    assert strict_extract(text, SENTENCE) == "7"
    assert strict_extract(text, HASH) is None


def test_leading_and_trailing_flags():
    base = dict(template=r"x=<NUM>", must_be_last_nonempty_line=True, on_multiple="reject")
    strict = FormatSpec(name="a", **base)
    assert strict_extract("x=5", strict) == "5"
    assert strict_extract("so x=5", strict) is None
    assert strict_extract("x=5 ok", strict) is None
    assert strict_extract("x=5.", strict) is None
    assert strict_extract("so x=5 ok", FormatSpec(name="b", leading_text_allowed=True,
                                                  trailing_text_allowed=True, **base)) == "5"
    assert strict_extract("x=5.", FormatSpec(name="c", allow_final_period=True, **base)) == "5"
    assert strict_extract("x=5..", FormatSpec(name="c", allow_final_period=True, **base)) is None
    assert strict_extract("x=5 .", FormatSpec(name="c", allow_final_period=True, **base)) == "5"


def test_trailing_text_with_number_boundaries():
    spec = FormatSpec(name="t", template=r"x=<NUM>", leading_text_allowed=True, trailing_text_allowed=True)
    assert strict_extract("x=18, because", spec) == "18"
    assert strict_extract("x=18.5 units", spec) == "18.5"
    assert strict_extract("x=1,00", spec) is None          # partial numbers are not captured
    assert strict_extract("x=12.3.4", spec) is None
    assert strict_extract("x=18abc", spec) == "18"
    assert strict_extract("x=1234,567", spec) is None


def test_currency_and_separator_options():
    cur = FormatSpec(name="c", template=r"total <NUM>", allow_currency=True)
    assert strict_extract("total $18", cur) == "18"
    assert strict_extract("total -$18", cur) == "-18"
    assert strict_extract("total $-18", cur) == "-18"
    assert strict_extract("total $1,000.50", cur) == "1000.5"
    nocur = FormatSpec(name="n", template=r"total <NUM>")
    assert strict_extract("total $18", nocur) is None
    nosep = FormatSpec(name="s", template=r"total <NUM>", allow_thousands_separator=False)
    assert strict_extract("total 1000", nosep) == "1000"
    assert strict_extract("total 1,000", nosep) is None


def test_placeholder_first_in_template():
    spec = FormatSpec(name="p", template=r"<NUM> is the answer", trailing_text_allowed=False)
    assert strict_extract("18 is the answer", spec) == "18"
    assert strict_extract("118 is the answer", spec) == "118"
    assert strict_extract("1,18 is the answer", spec) is None     # lookbehind refuses "1,18"
    assert strict_extract("x.5 is the answer", spec) is None


def test_match_may_not_span_lines():
    spec = FormatSpec(name="w", template=r"answer\s+<NUM>")       # \s can match a newline
    assert strict_extract("answer\n18", spec) is None
    assert strict_extract("answer 18", spec) == "18"


# ---------------------------------------------------------------------------------------
# Reasoning versus final line; strict versus lenient disagreement
# ---------------------------------------------------------------------------------------
def test_answer_in_reasoning_vs_final_line():
    reasoning = "She has 16 eggs. 16 - 3 - 4 = 9. 9 * 2 = 18."
    assert strict_extract(reasoning, SENTENCE) is None
    assert lenient_extract(reasoning) == "18"
    with_final = reasoning + "\nThe answer is 18."
    assert strict_extract(with_final, SENTENCE) == "18"
    assert lenient_extract(with_final) == "18"


def test_strict_lenient_disagreements():
    # Format violation, right number: lenient credits it, strict does not.
    assert strict_extract("It comes to 18 dollars.", SENTENCE) is None
    assert lenient_extract("It comes to 18 dollars.") == "18"
    # Trailing chatter after the answer: lenient reads the wrong number.
    t = "The answer is 18.\nI used 2 steps."
    assert strict_extract(t, SENTENCE) is None
    assert lenient_extract(t) == "2"
    # Correct format, wrong number of steps in the explanation after: strict wins when in position.
    t2 = "Step 1 uses 2 eggs, step 2 uses 3.\nThe answer is 18."
    assert strict_extract(t2, SENTENCE) == "18" and lenient_extract(t2) == "18"
    # Strict can be non-None while lenient differs: the number sits before a trailing marker.
    t3 = "The answer is 18. (see step 2)"
    assert strict_extract(t3, SENTENCE) is None
    assert lenient_extract(t3) == "2"


# ---------------------------------------------------------------------------------------
# lenient_extract
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("", None),
    (None, None),
    ("   \n", None),
    ("no digits at all", None),
    ("18", "18"),
    ("The answer is 18.", "18"),
    ("= 18.", "18"),
    ("#### 18", "18"),
    ("#### 1,000", "1000"),
    ("$1,000.50", "1000.5"),
    ("$18", "18"),
    ("It costs US$18 total", "18"),
    ("18 dollars", "18"),
    ("18%", "18"),
    ("18.00", "18"),
    ("18.", "18"),
    ("18.\n\nHope this helps!", "18"),
    ("x = -3", "-3"),
    ("x=-3", "-3"),
    ("(-3)", "-3"),
    ("-$5", "-5"),
    ("$-5", "-5"),
    ("5-3", "3"),                      # minus glued to a word character is not a sign
    ("5 - 3 = 2", "2"),
    ("a-5", "5"),
    ("\u22127 degrees", "-7"),        # Unicode minus
    ("temperature fell to \u22127.5.", "-7.5"),
    ("range 5\u201310", "10"),          # en dash between digits is not a sign
    ("5, 6, 7", "7"),
    ("1,000,000", "1000000"),
    ("1,2345", "2345"),                # invalid comma group ends the token
    ("12,34", "34"),
    ("3/4", "4"),                      # fractions are not interpreted
    ("1e3", "1"),                      # scientific notation is not interpreted
    ("H2O is 18", "18"),
    ("x2", None),                      # digits glued to a word are ignored
    ("Q1: five", None),
    ("\\boxed{18}", "18"),
    ("$\\boxed{1,000}$", "1000"),
    ("1{,}000", "1000"),
    ("1\\,000", "1000"),
    ("1\\,0000", "0"),                  # not a valid thousands group: last token is 0000 -> 0
    (".5", "0.5"),
    ("it is 0.50", "0.5"),
    ("first 7 then 8", "8"),
    ("\uff11\uff18", "18"),
    ("Answer: 18\n(Check: 9 * 2 = 18)", "18"),
    ("The answer is 5\u00a0000", "0"),   # NBSP-separated thousands are not merged
])
def test_lenient_extract(text, expected):
    assert lenient_extract(text) == expected


def test_lenient_picks_last_number_even_if_unrelated():
    assert lenient_extract("The answer is 18. Thanks for asking at 3 pm") == "3"


# ---------------------------------------------------------------------------------------
# Empty and odd outputs
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [None, "", " ", "\n\n", "\t"])
def test_empty_outputs(text):
    assert strict_extract(text, SENTENCE) is None
    assert lenient_extract(text) is None
    assert compliance(text, SENTENCE) is False


def test_very_long_output_is_handled():
    text = ("step " * 5000) + "\nThe answer is 18."
    assert strict_extract(text, SENTENCE) == "18"
    assert lenient_extract(text) == "18"


# ---------------------------------------------------------------------------------------
# score_outputs
# ---------------------------------------------------------------------------------------
def _gold():
    return pd.DataFrame({"item_id": [0, 1, 2, 3], "gold_answer": ["18", "3", "70000", "-10"]})


def _outputs():
    return pd.DataFrame({
        "request_id": ["r0", "r1", "r2", "r3", "r4", "r5", "r6"],
        "item_id": [0, 1, 2, 3, 0, 1, 2],
        "output_text": [
            "The answer is 18.",                 # correct and compliant
            "I think it is 3 apples.",           # lenient-correct only
            "The answer is 70,000.",             # correct with thousands separator
            "The answer is -10.",                # correct negative
            "The answer is 18.0.",               # decimal form of the same integer
            None,                                 # missing output
            "The answer is 71000.",              # wrong
        ],
    })


def test_score_outputs_columns_and_values():
    out = score_outputs(_outputs(), _gold(), SENTENCE)
    for col in sc.SCORE_COLUMNS:
        assert col in out.columns
    assert list(out["strict_correct"]) == [1, 0, 1, 1, 1, 0, 0]
    assert list(out["lenient_correct"]) == [1, 1, 1, 1, 1, 0, 0]
    assert list(out["compliant"]) == [True, False, True, True, True, False, True]
    assert list(out["strict_answer"]) == ["18", None, "70000", "-10", "18", None, "71000"]
    assert list(out["lenient_answer"]) == ["18", "3", "70000", "-10", "18", None, "71000"]
    assert out["strict_correct"].dtype.kind == "i" and out["lenient_correct"].dtype.kind == "i"
    assert out["compliant"].dtype == bool
    assert out["strict_answer"].dtype == object


def test_non_extractable_counts_as_incorrect_in_strict_mean():
    out = score_outputs(_outputs(), _gold(), SENTENCE)
    assert math.isclose(out["strict_correct"].mean(), 4 / 7)
    assert out["strict_correct"].sum() <= out["compliant"].sum()
    assert (out.loc[~out["compliant"], "strict_correct"] == 0).all()


def test_strict_correct_implies_compliant_and_answer_present():
    rng = np.random.default_rng(0)
    pieces = ["The answer is ", "#### ", "Answer: ", "18", "3", "1,000", "-4", ".", "\n", " ", "apples", "x", "$"]
    texts = ["".join(rng.choice(pieces, size=rng.integers(0, 8))) for _ in range(400)]
    df = pd.DataFrame({"request_id": [f"q{i}" for i in range(400)], "item_id": [0] * 400, "output_text": texts})
    for spec in CANDIDATE_SPECS.values():
        out = score_outputs(df, _gold(), spec)
        assert (out["strict_answer"].notna() == out["compliant"]).all()
        assert (out["strict_correct"] <= out["compliant"].astype(int)).all()


def test_score_outputs_gold_forms_and_gold_normalization():
    out_df = _outputs()
    a = score_outputs(out_df, _gold(), SENTENCE)
    b = score_outputs(out_df, dict(zip(_gold().item_id, _gold().gold_answer)), SENTENCE)
    c = score_outputs(out_df, pd.Series(_gold().gold_answer.values, index=_gold().item_id), SENTENCE)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    odd_gold = {0: "18.0", 1: "3", 2: "70,000", 3: "\u221210"}
    d = score_outputs(out_df, odd_gold, SENTENCE)
    pd.testing.assert_frame_equal(a, d)


def test_score_outputs_does_not_mutate_and_preserves_order_and_index():
    df = _outputs().iloc[::-1].copy()
    df.index = [10, 20, 30, 40, 50, 60, 70]
    before = df.copy(deep=True)
    out = score_outputs(df, _gold(), SENTENCE)
    pd.testing.assert_frame_equal(df, before)
    assert list(out.index) == [10, 20, 30, 40, 50, 60, 70]
    assert list(out["request_id"]) == list(df["request_id"])
    assert set(df.columns) <= set(out.columns)


def test_score_outputs_errors():
    with pytest.raises(ValueError, match="missing columns"):
        score_outputs(_outputs().drop(columns=["output_text"]), _gold(), SENTENCE)
    dup = _outputs()
    dup.loc[1, "request_id"] = "r0"
    with pytest.raises(ValueError, match="unique"):
        score_outputs(dup, _gold(), SENTENCE)
    unknown = _outputs()
    unknown.loc[0, "item_id"] = 99
    with pytest.raises(KeyError):
        score_outputs(unknown, _gold(), SENTENCE)
    with pytest.raises(ValueError):
        score_outputs(_outputs(), {0: "18", 1: "three", 2: "1", 3: "2"}, SENTENCE)
    with pytest.raises(TypeError):
        score_outputs(_outputs(), [18, 3, 7, 1], SENTENCE)
    nan_item = _outputs()
    nan_item["item_id"] = nan_item["item_id"].astype(float)
    nan_item.loc[0, "item_id"] = np.nan
    with pytest.raises(ValueError):
        score_outputs(nan_item, _gold(), SENTENCE)


def test_score_outputs_empty_frame():
    empty = pd.DataFrame({"request_id": [], "item_id": [], "output_text": []})
    out = score_outputs(empty, _gold(), SENTENCE)
    assert len(out) == 0 and set(sc.SCORE_COLUMNS) <= set(out.columns)


def test_score_outputs_non_string_outputs_become_empty():
    df = pd.DataFrame({"request_id": ["a", "b"], "item_id": [0, 1], "output_text": [np.nan, 3]})
    out = score_outputs(df, _gold(), SENTENCE)
    assert list(out["compliant"]) == [False, False]
    assert list(out["lenient_correct"]) == [0, 0]


def test_score_outputs_depends_on_spec_only_through_strict_columns():
    texts = ["#### 18", "The answer is 18.", "Answer: 18", "\\boxed{18}", "18"]
    df = pd.DataFrame({"request_id": list("abcde"), "item_id": [0] * 5, "output_text": texts})
    gold = {0: "18"}
    cols = {}
    for name, spec in CANDIDATE_SPECS.items():
        out = score_outputs(df, gold, spec)
        cols[name] = out
        assert list(out["lenient_correct"]) == [1] * 5
    assert list(cols["hash_n"]["strict_correct"]) == [1, 0, 0, 0, 0]
    assert list(cols["the_answer_is"]["strict_correct"]) == [0, 1, 0, 0, 0]
    assert list(cols["answer_colon"]["strict_correct"]) == [0, 0, 1, 0, 0]
    assert list(cols["boxed"]["strict_correct"]) == [0, 0, 0, 1, 0]
    for out in cols.values():
        pd.testing.assert_series_equal(out["lenient_answer"], cols["hash_n"]["lenient_answer"])


def test_duplicate_texts_are_scored_consistently():
    df = pd.DataFrame({"request_id": [f"r{i}" for i in range(50)], "item_id": [i % 4 for i in range(50)],
                       "output_text": ["The answer is 3."] * 50})
    out = score_outputs(df, _gold(), SENTENCE)
    assert out["strict_correct"].tolist() == [1 if (i % 4) == 1 else 0 for i in range(50)]


# --- strip_markup (scorer 1.1.0) ---------------------------

MARKUP = CANDIDATE_SPECS["answer_colon_markup"]


@pytest.mark.parametrize("text", ["**Answer: 260**", "work\n**Answer:** 260", "Answer: **260**",
                                  "*Answer: 260*", "`Answer: 260`", "***Answer: 260***",
                                  "__Answer: 260__", "  **Answer: 260**  \n\n"])
def test_markup_spec_accepts_emphasis(text):
    assert strict_extract(text, MARKUP) == "260"
    assert strict_extract(text, COLON) is None


@pytest.mark.parametrize("text, want", [("2**3 = 8\nAnswer: 8", "8"), ("Answer: 2**3", None),
                                        ("**Answer: 260** done", None), ("**Answer: 260**\nmore", None),
                                        ("**Answer: 1,000**", "1000")])
def test_markup_spec_keeps_other_rules(text, want):
    assert strict_extract(text, MARKUP) == want


def test_markup_leaves_exponents_and_unpaired_markers():
    assert sc.strip_markup("2**3 and 4**2") == "2**3 and 4**2"
    assert sc.strip_markup("a ** b") == "a ** b"
    assert sc.strip_markup("**bold** and **more**") == "bold and more"


def test_markup_sentence_candidate():
    spec = CANDIDATE_SPECS["the_answer_is_markup"]
    assert strict_extract("So the answer is **18**.", spec) == "18"
    assert strict_extract("So the answer is **18**.", SENTENCE) is None


def test_fingerprints_of_1_0_0_specs_are_reproducible():
    # Recorded with scorer 1.0.0 before strip_markup existed. Scorer 1.2.0 changed the candidates' number tolerance;
    # the 1.0.0 settings still hash to the recorded values, so the fingerprint itself is unchanged.
    want = {"the_answer_is": "688c70fe878f60698f659eb197b1d573893aa4e908a4112a35ae330603a8f82b",
            "hash_n": "dffdd5646ec9dd15ff11e298e7d7c2b34d5565f083934df4c901773ff5d21138",
            "answer_colon": "10e3459a544009854b8ec968713b7f0dc1e8cc0eff647e5ee025bd07a79a12c1",
            "boxed": "24f3a1d9c628aa9157f0e0758305f80f94d30695fce726eb604d026913b8d553"}
    v100 = {"the_answer_is": {"allow_currency": False}, "boxed": {"allow_currency": False, "strip_latex": False},
            "hash_n": {"allow_currency": False, "allow_final_period": False},
            "answer_colon": {"allow_currency": False, "allow_final_period": False}}
    for name, fp in want.items():
        assert sc.spec_fingerprint(FormatSpec(**{**sc.spec_to_dict(CANDIDATE_SPECS[name]), **v100[name]})) == fp
    old = {k: v for k, v in sc.spec_to_dict(COLON).items() if k != "strip_markup"}
    assert sc.spec_from_dict(old) == COLON
    assert sc.spec_fingerprint(MARKUP) != sc.spec_fingerprint(COLON)


def test_fingerprints_of_1_2_0_specs():
    # Scorer 1.3.0 set strip_latex for boxed only; its 1.2.0 settings still hash to the recorded value.
    want = {"the_answer_is": "dfd3ef6338306fd725dd9f5bfb2c2948071a7af035bb09bc4a1e46301ed54731",
            "hash_n": "dda68d01006ce4181914a914ff6b5e157970e35331bb2f5b55314b1f038891cd",
            "answer_colon": "7f3cd6b464190f0a0c54d6f98bfc342350ed408bd1499d1b8a5f5c78cfae93ed",
            "boxed": "b069d5b382643133da94f985428e1f62032ecca2c36cbb029cc755a4bc8021c6",
            "answer_colon_markup": "44155560675623f00dd0b4a9c2acdeef6c646491a8fb92c245dd7318d50aeb7e",
            "the_answer_is_markup": "38bf50b80350a317d87a274cb389c9cdc76e20abc49b7bfa606fa58095d48cb6"}
    v120 = {n: FormatSpec(**{**sc.spec_to_dict(s), "strip_latex": False}) for n, s in CANDIDATE_SPECS.items()}
    assert {n: sc.spec_fingerprint(s) for n, s in v120.items()} == want
    now = {n: sc.spec_fingerprint(s) for n, s in CANDIDATE_SPECS.items()}
    assert {n: fp for n, fp in now.items() if n != "boxed"} == {n: fp for n, fp in want.items() if n != "boxed"}
    assert now["boxed"] != want["boxed"]
