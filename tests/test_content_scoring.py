import pytest

from src import bbh
from src import content_scoring as cs


def test_gsm8k_classes():
    assert cs.classify("so \\boxed{18}", "stop", "18", "gsm8k") == ("correct", "18")
    assert cs.classify("\\boxed{18} dollars, so 36 in all", "stop", "18", "gsm8k") == ("correct", "18")
    assert cs.classify("\\boxed{\\$1,234.00}", "stop", "1234", "gsm8k")[0] == "correct"
    assert cs.classify("\\boxed{3 \\times 6 = 18}", "stop", "18", "gsm8k") == ("correct", "18")
    assert cs.classify("the answer is 17", "stop", "18", "gsm8k") == ("wrong", "17")
    assert cs.classify("\\boxed{eighteen}", "stop", "18", "gsm8k") == ("unreadable", None)
    assert cs.classify("no number here", "stop", "18", "gsm8k") == ("unreadable", None)
    assert cs.classify("\\boxed{18", "stop", "18", "gsm8k") == ("malformed_box", None)
    assert cs.classify("\\boxed{18", "length", "18", "gsm8k") == ("truncated", None)


def test_math_classes():
    assert cs.classify("so \\boxed{\\frac{14}{3}}", "stop", "14/3", "math") == ("correct", "14/3")
    assert cs.classify("\\boxed{\\dfrac{6}{4}}", "stop", "3/2", "math")[0] == "correct"
    assert cs.classify("\\boxed{x = 5}", "stop", "5", "math") == ("correct", "5")
    assert cs.classify("\\boxed{5\\text{ cm}}", "stop", "5", "math")[0] == "correct"      # strict reads it too
    assert cs.classify("\\boxed{\\text{5}}", "stop", "5", "math")[0] == "correct"
    assert cs.classify("\\boxed{0.5}", "stop", "1/2", "math")[0] == "correct"
    assert cs.classify("\\boxed{-\\frac{1}{2}}", "stop", "-1/2", "math")[0] == "correct"
    # no number is searched for inside a box: these hold the digit 2 without being 2
    assert cs.classify("\\boxed{\\sqrt{2}}", "stop", "2", "math") == ("unreadable", None)
    assert cs.classify("\\boxed{2\\pi}", "stop", "2", "math") == ("unreadable", None)
    # the placeholder is <number>; a copied placeholder box is skipped
    assert cs.classify("\\boxed{<number>} so 7", "stop", "7", "math") == ("correct", "7")
    assert cs.classify("\\boxed{<number>: 7}", "stop", "7", "math")[0] == "correct"
    assert cs.classify("7", "stop", "7", "math") == ("correct", "7")
    assert cs.classify("the answer is 3/4.", "stop", "3/4", "math") == ("correct", "3/4")
    assert cs.classify("\\boxed{12", "stop", "12", "math") == ("malformed_box", None)
    assert cs.classify("so far 12", "length", "12", "math") == ("truncated", None)
    assert cs.classify("\\boxed{11}", "stop", "12", "math") == ("wrong", "11")


def test_math_strict_correct_implies_content_correct():
    from src import math_task as mt
    for t in ("\\boxed{5^\\circ}", "\\boxed{\\$12}", "\\boxed{50\\%}", "\\boxed{1{,}000}", "\\boxed{\\frac43}",
              "\\boxed{14/3.}", "\\boxed{3} and \\boxed{-0}", "\\boxed{$7$}", "\\boxed{2\\text{ units}^2}"):
        s = mt.strict_extract(t)
        assert s is not None, t
        assert cs.classify(t, "stop", s, "math") == ("correct", s), t


def test_last_complete_box_wins():
    assert cs.classify("\\boxed{12} then \\boxed{18}", "stop", "18", "gsm8k")[0] == "correct"
    assert cs.classify("\\boxed{18} then \\boxed{12", "stop", "18", "gsm8k")[0] == "correct"


def test_bbh_letter_and_words():
    letter = bbh.AnswerType.from_label("letter")
    gold = bbh.normalize_answer("(B)", letter)
    assert cs.classify("\\boxed{(B)}", "stop", gold, "bbh", letter) == ("correct", gold)
    assert cs.classify("\\boxed{B}", "stop", gold, "bbh", letter)[0] == "correct"
    assert cs.classify("I pick (C).", "stop", gold, "bbh", letter)[0] == "wrong"
    assert cs.classify("\\boxed{none}", "stop", gold, "bbh", letter)[0] == "unreadable"
    with pytest.raises(ValueError):
        cs.classify("x", "stop", "B", "bbh")


def test_bbh_option_text_and_wrappers():
    letter = bbh.AnswerType.from_label("letter")
    q = "Which ball?\nOptions:\n(A) red ball\n(B) pink ball\n(C) The Fugitive"
    opts = cs.options(q)
    assert opts == {"red ball": "(A)", "pink ball": "(B)", "the fugitive": "(C)"}
    assert cs.classify("\\boxed{Pink ball.}", "stop", "(B)", "bbh", letter, opts)[0] == "correct"
    assert cs.classify("\\boxed{\\text{The Fugitive}}", "stop", "(B)", "bbh", letter, opts) == ("wrong", "(C)")
    assert cs.classify("\\boxed{blue ball}", "stop", "(B)", "bbh", letter, opts)[0] == "unreadable"
    assert cs.options("Options:\n(A) x\n(B) x\n(C) y") == {"y": "(C)"}
    counts = cs.options("How many?\nOptions:\n(A) zero\n(B) one\n(C) two\n(D) three")
    assert cs.classify("\\boxed{3}", "stop", "(D)", "bbh", letter, counts)[0] == "correct"
    yn_low = bbh.AnswerType.from_label("words:yes/no")
    for s in ("\\boxed{<not plausible>}", "\\boxed{<answer> - Not plausible}"):
        assert cs.classify(s, "stop", "no", "bbh", yn_low, task_name="sports_understanding")[0] == "correct"
    # a hedged answer is not one of the two
    assert cs.classify("\\boxed{Plausible, but unusual}", "stop", "yes", "bbh", yn_low,
                       task_name="sports_understanding")[0] == "unreadable"


def test_bbh_yes_no_wordings():
    yn = bbh.AnswerType.from_label("words:Yes/No")
    yn_low = bbh.AnswerType.from_label("words:yes/no")
    assert cs.classify("\\boxed{\\text{True}}", "stop", "Yes", "bbh", yn, task_name="web_of_lies")[0] == "correct"
    assert cs.classify("\\boxed{Osvaldo tells the truth.}", "stop", "Yes", "bbh", yn,
                       task_name="web_of_lies")[0] == "correct"
    assert cs.classify("\\boxed{Osvaldo does not tell the truth}", "stop", "No", "bbh", yn,
                       task_name="web_of_lies")[0] == "correct"
    assert cs.classify("\\boxed{Sal lies}", "stop", "Yes", "bbh", yn, task_name="web_of_lies")[0] == "wrong"
    assert cs.classify("Not plausible.\n\\boxed{Not plausible}", "stop", "no", "bbh", yn_low,
                       task_name="sports_understanding") == ("correct", "no")
    assert cs.classify("Plausible.", "stop", "yes", "bbh", yn_low, task_name="sports_understanding")[0] == "correct"
    # the plausibility wording belongs to its task only
    assert cs.classify("\\boxed{plausible}", "stop", "Yes", "bbh", yn, task_name="navigate")[0] == "unreadable"
    tf = bbh.AnswerType.from_label("words:True/False")
    assert cs.classify("\\boxed{Yes}", "stop", "True", "bbh", tf, task_name="boolean_expressions")[0] == "unreadable"


def test_placeholder_box_and_truncation_with_a_box():
    assert cs.classify("End with \\boxed{<answer>}. The answer is 18.", "stop", "18", "gsm8k")[0] == "correct"
    assert cs.classify("\\boxed{18} and then more text", "length", "18", "gsm8k")[0] == "correct"
    assert cs.classify("so far 12 and", "length", "18", "gsm8k")[0] == "truncated"
    assert cs.classify("\\boxed{<answer>} and 12 and", "length", "18", "gsm8k")[0] == "truncated"
