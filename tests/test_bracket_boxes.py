"""A box written with angle brackets in place of its braces ("\\boxed<Odysseus>") or as a
tag ("<boxed>Liguria</boxed>") is read as a box holding its content by every content reader (src/content_scoring.py
content/1.3.0, src/hotpot_task.content_read hotpot-content/1.2.0), unless the content is a placeholder; strict and
lenient do not change; the checker's own reader (scripts/verify_ladder_ground_truth.Reader) finds the same boxes with
its own scan."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import bbh, content_scoring as cs, forced_stop as fs, hotpot_content as hc, hotpot_task as ht  # noqa: E402
from src import math_task  # noqa: E402
import verify_ladder_ground_truth as gt  # noqa: E402

P = "\\boxed{"
LETTER = bbh.AnswerType.from_label("letter")
YESNO = bbh.AnswerType.from_label("words:yes/no")

# texts and the boxes of other brackets they hold: (start, end, content, form)
FORMS = [
    ("\\boxed<Odysseus>", [(0, 16, "Odysseus", "angle")]),
    ("so boxed <24>.", [(3, 13, "24", "angle")]),
    ("x \\BOXED<B> y", [(2, 11, "B", "angle")]),
    ("A\n\n<boxed>Liguria</boxed>", [(3, 25, "Liguria", "tag")]),
    ("<boxed>No</boxed", [(0, 16, "No", "tag")]),
    ("< boxed >Rock music< / boxed >.", [(0, 30, "Rock music", "tag")]),
    ("<boxed><answer></boxed>", [(0, 23, "<answer>", "tag")]),
    ("\\boxed<>", [(0, 8, "", "angle")]),
    ("\\boxed<Answer: (B)>}", [(0, 19, "Answer: (B)", "angle")]),
    ("Final answer: <boxed>", []),                               # a tag that never closes
    ("<boxed>Gaja\n</boxed>", []),                                # not on one line
    ("\\boxed<a\nb>", []),
    ("\\boxed{<B> x}</boxed>", []),                               # a complete box: read as before
    ("(<boxed answer>)", []),
    (P + "<boxed>x</boxed>} then \\boxed<y>", [(30, 39, "y", "angle")]),  # the tag inside the box is the box's
]


@pytest.mark.parametrize("text,want", FORMS)
def test_other_boxes_are_found(text, want):
    assert cs.other_boxes(text) == want


def test_versions_and_records():
    # content/1.4.0 and hotpot-content/1.3.0 add answer cues and clock times; the other-bracket rule is
    # unchanged
    # content/1.5.0 and hotpot-content/1.4.0: LaTeX spacing, options named with their letter, lead letters
    # content/1.6.0 and hotpot-content/1.5.0: conflicts, answer tags, ranges, typography
    assert cs.CONTENT_SCORER_VERSION == "content/1.7.0"
    assert ht.CONTENT_SCORER_VERSION == "hotpot-content/1.5.0"
    r = cs.other_box_rules()
    assert r["version"] == cs.CONTENT_SCORER_VERSION and r["pattern"] == cs.OTHER_BOX.pattern
    rules = ht.content_scorer_rules()
    assert rules["other_box_pattern"] == cs.OTHER_BOX.pattern and "angle brackets" in rules["other_box"]
    assert {"angle_box", "tag_box"} <= set(ht.CONTENT_SOURCES)


@pytest.mark.parametrize("c", ["answer", "<answer>", "", " ", "option", "<letter>", "number", "letter",
                               "final answer", "<your answer here>"])
def test_placeholders_in_other_brackets(c):
    assert cs.other_box_placeholder(c)
    assert ht.other_box_placeholder(c)


# "Option A" and "<Option X>" were placeholders until content/1.4.0; an option named with its letter is read since
# content/1.5.0
@pytest.mark.parametrize("c", ["Odysseus", "B", "(B)", "14", "not plausible", "Answer: (B)", "Option A B", "Option A",
                               "<Option X>"])
def test_filled_other_boxes_are_not_placeholders(c):
    assert not cs.other_box_placeholder(c)


def test_hotpot_reads_other_boxes():
    assert ht.content_read("The answer is Odysseus.\n\n\\boxed<Odysseus>") == ("Odysseus", "angle_box")
    assert ht.content_read("Liguria\n\n<boxed>Liguria</boxed>") == ("Liguria", "tag_box")
    assert ht.content_read("of <boxed>film director</boxed>.") == ("film director", "tag_box")
    # the box wins over the prose, as for a complete box
    assert ht.content_read("The answer is: American\n\n\\boxed<Robert Pastorelli>") == ("Robert Pastorelli",
                                                                                         "angle_box")
    # placeholders in either form are skipped and removed; the text is read as before
    assert ht.content_read("Holston River \\boxed<answer>") == ("Holston River", "last_line")
    assert ht.content_read("Holston River\n<boxed><answer></boxed>") == ("Holston River", "last_line")
    assert ht.content_read("Paris\n<boxed></boxed>") == ("Paris", "last_line")
    # order of openings with complete boxes: the last box is read
    assert ht.content_read(P + "Rome} then \\boxed<Paris>") == ("Paris", "angle_box")
    assert ht.content_read("\\boxed<Rome> then " + P + "Paris}") == ("Paris", "box")
    assert ht.content_read(P + "Paris} then \\boxed<answer>") == ("Paris", "box")
    # a reply without these forms reads as under 1.1.0
    assert ht.content_read(P + "Paris}") == ("Paris", "box")
    assert ht.content_read("Answer: Paris") == ("Paris", "answer_cue")


def test_hotpot_strict_lenient_and_format_count_do_not_change():
    for t in ("\\boxed<Odysseus>", "<boxed>Liguria</boxed>"):
        assert ht.strict_extract(t) is None and not ht.compliance(t)
    assert ht.lenient_extract("x\n\\boxed<Odysseus>") == "\\boxed<Odysseus>"
    assert hc.format_flags("\\boxed<Odysseus>")[2] == "unboxed"      # a count of format, not a reading


def test_gsm8k_math_bbh_read_other_boxes():
    assert cs.classify("So $720\n\n<boxed>$720</boxed>", "stop", "720", "gsm8k") == ("correct", "720")
    assert cs.classify("12 hours/4 weeks <boxed>12 hours</boxed> on TV", "stop", "36", "gsm8k") == ("wrong", "12")
    assert cs.classify("so boxed<256>", "stop", "256", "math") == ("correct", "256")
    # MATH-500 searches no number inside a box
    assert cs.classify("r = 3\n<boxed>r = √(A/3)</boxed>", "stop", "3", "math") == ("unreadable", None)
    assert cs.classify("The sentence is plausible.\n\n<boxed>Plausible</boxed>", "stop", "yes", "bbh", YESNO,
                       task_name="sports_understanding") == ("correct", "yes")
    opts = {"brave": "(A)", "named entities": "(D)"}
    assert cs.classify("Named Entities\n\n\\boxed<Named Entities>", "stop", "(D)", "bbh", LETTER, opts=opts) == \
        ("correct", "(D)")
    assert cs.classify("the answer is (C) Ambiguous.\n\n\\boxed<C>", "stop", "(C)", "bbh", LETTER) == \
        ("correct", "(C)")
    # a placeholder in other brackets is still no answer, and the text is read
    assert cs.classify("so 18 \\boxed<number>", "stop", "18", "gsm8k") == ("correct", "18")
    assert cs.classify("so 18 <boxed><number></boxed>", "stop", "18", "gsm8k") == ("correct", "18")
    assert cs.classify("<boxed><number></boxed>", "stop", "18", "gsm8k") == ("unreadable", None)
    # a reply that ran to its cap with a box of other brackets before the cap is read, as with a complete box
    assert cs.classify("... <boxed>17</boxed> and more", "length", "17", "gsm8k") == ("correct", "17")
    assert cs.classify("... and more", "length", "17", "gsm8k") == ("truncated", None)
    # order of openings with complete boxes
    assert cs.classify(P + "5} then \\boxed<7>", "stop", "7", "gsm8k") == ("correct", "7")
    assert cs.classify("\\boxed<5> then " + P + "7}", "stop", "7", "gsm8k") == ("correct", "7")


def test_strict_and_lenient_scorers_do_not_read_other_boxes():
    assert bbh.strict_extract("\\boxed<B>", LETTER) is None
    assert math_task.strict_extract("\\boxed<5>") is None


def test_drop_placeholders_removes_other_bracket_placeholders_only():
    assert cs.drop_placeholders("a <boxed><answer></boxed> b \\boxed<Paris> c \\boxed<> d") == \
        "a   b \\boxed<Paris> c   d"


@pytest.mark.skipif(not (ROOT / "data/raw/gsm8k/main/test-00000-of-00001.parquet").exists()
                    or not (ROOT / "data/processed/bbh/items_target.parquet").exists(), reason="item tables absent")
def test_checker_finds_the_same_boxes_and_agrees():
    r = gt.Reader(ROOT, set(fs.WRAPPER_COMMANDS))
    for text, want in FORMS:
        spans = r.box_spans(text)
        assert [(a, b, c) for a, b, c, _ in cs.other_boxes(text)] == r.other_boxes(text, spans), text
    import score_content as SC
    gs = SC.gold_table("gsm8k")[0][0]
    for t in ("<boxed>" + gs + "</boxed>", "\\boxed<" + gs + ">", "so " + gs + " \\boxed<number>",
              "<boxed><number></boxed>", P + "5} then \\boxed<" + gs + ">", "\\boxed<5> then " + P + gs + "}",
              "x <boxed>" + gs + "</boxed", "<boxed>1</boxed> <boxed>" + gs + "</boxed>"):
        mine = r.correct("gsm8k", "0", t, "stop")
        theirs = cs.classify(t, "stop", gs, "gsm8k")[0] == "correct"
        assert mine == theirs, (t, mine, theirs)
    item = next(i for i, v in r.bbh.items() if v[1] == "letter")
    g = SC.gold_table("bbh")[item]
    letter = g[0][1]
    for t in ("\\boxed<" + letter + ">", "<boxed>(" + letter + ")</boxed>", "<boxed><answer></boxed> (" + letter + ")",
              "\\boxed<Option " + letter + ">"):
        mine = r.correct("bbh", item, t, "stop")
        theirs = cs.classify(t, "stop", g[0], "bbh", g[1], opts=g[2], task_name=g[3])[0] == "correct"
        assert mine == theirs, (t, mine, theirs)


@pytest.mark.skipif(not (ROOT / "data/bracket_boxes/candidates.parquet").exists(), reason="candidate replies absent")
def test_checker_scan_equals_the_pattern_on_every_candidate_reply():
    import pandas as pd
    d = pd.read_parquet(ROOT / "data/bracket_boxes/candidates.parquet", columns=["text"])
    for t in d["text"].drop_duplicates():
        spans = gt.Reader.box_spans(t)
        assert [(a, b, c) for a, b, c, _ in cs.other_boxes(t, spans)] == gt.Reader.other_boxes(t, spans), t[-200:]


def test_hotpot_manifest_is_the_new_record_and_the_old_is_kept():
    p = ROOT / "manifests" / "hotpot_content_scorer.json"
    if p.exists():
        assert json.loads(p.read_text(encoding="utf-8")) == ht.content_scorer_record()
    old = ROOT / "manifests" / "hotpot_content_scorer_1.1.0.json"
    if old.exists():
        assert json.loads(old.read_text(encoding="utf-8"))["scorer_version"] == "hotpot-content/1.1.0"
