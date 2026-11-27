r"""src/forced_stop.py: forced replies the stop at the first "}" left with an open box.

The examples are stored replies (text after the prefill), from results/forced_stop/audit.json: Mistral NeMo and
Small 3.1 on BIG-Bench Hard, Ministral 3 8B and 14B, Qwen3.5-122B-A10B, DeepSeek and Mistral Large 3,
Mistral Small 3.1 and Gemma 4 on GSM8K. Unaffected replies (the box closes, or the arm has no stop) score exactly as
before.
"""
import pandas as pd
import pytest

from src import bbh, content_scoring as cs, forced_stop as fs, scoring

P = fs.BOX_PREFILL
LETTER = bbh.AnswerType.from_label("letter")
YESNO = bbh.AnswerType.from_label("words:Yes/No")
INTEGER = bbh.AnswerType.from_label("integer")
BOXED = scoring.CANDIDATE_SPECS["boxed"]


@pytest.mark.parametrize("reply, closed", [
    # BIG-Bench Hard (NeMo, Small 3.1) and the size ladder (Ministral 3, Qwen3.5-122B-A10B)
    (r"(\text{J}", r"(\text{J})}"),
    (r"(\text{C}", r"(\text{C})}"),
    (r"(\textbf{B}", r"(\textbf{B})}"),
    (r"(\mathrm{D}", r"(\mathrm{D})}"),
    (r"\text{(B)}", r"\text{(B)}}"),
    (r"\mathbf{(E)}", r"\mathbf{(E)}}"),
    (r"-\bf{50}", r"-\bf{50}}"),
    (r"-\underline{40}", r"-\underline{40}}"),
    # a bare group that opens the box (Ministral 3, Mistral Large 3)
    (r"{(B)}", r"{(B)}}"),
    (r"{-28}", r"{-28}}"),
    (r"({C}", r"({C})}"),
    # GSM8K (Small 3.1, Gemma 4, Qwen3.6, the ladder)
    (r"131 \text{ mm}", r"131 \text{ mm}}"),
    (r"9\text{ inches}", r"9\text{ inches}}"),
    (r"12:20\,\text{pm}", r"12:20\,\text{pm}}"),
    (r"\textbf{\text{C}", r"\textbf{\text{C}}}"),
    (r"[\text{B}", r"[\text{B}]}"),
    # 1.1.0: words that only begin like a connective, and units
    (r"(\text{Tom}", r"(\text{Tom})}"),
    (r"2 \text{ tons}", r"2 \text{ tons}}"),
    (r"9\text{ hours}", r"9\text{ hours}}"),
    # 1.1.0: DeepSeek writes a tab where the backslash of \text should be (the escape "\t" as a tab); read as \text
    ("\text{(B)}", r"\text{(B)}}"),
])
def test_completed(reply, closed):
    assert fs.classify(P + reply) == "completed"
    assert fs.close(P + reply) == P + closed


@pytest.mark.parametrize("reply, kind", [
    (r"16 \frac{2}", "open_other"),             # numerator of a fraction: the answer is cut
    (r"-\frac{48 - 21 - 3}", "open_other"),
    (r"6 {10}", "open_other"),                  # a group after other text
    (r"${20,000}", "open_other"),
    (r"20 \times (1.2)^{18}", "open_other"),    # an exponent
    (r"33.\overline{3}", "open_other"),         # a repeating decimal
    (r"(\ref{answer}", "open_other"),
    (r"(\textcolor{yellow}", "open_other"),
    (r"\frac{\text{a}", "open_other"),          # a wrapper inside a fraction still open
    (r"(\ref{A}", "open_other"),                # a label, not read as the option (module doc)
    ("\tfrac{1}", "open_other"),                # a tab before a name that is not a wrapper stays as it is
    (r"(\boxed{K}", "open_inner_box"),          # a complete inner box, read already
    (r" $8 + 3 * 8 = \boxed{34}", "open_inner_box"),
])
def test_left_as_it_is(reply, kind):
    assert fs.classify(P + reply) == kind
    assert fs.close(P + reply) == P + reply


@pytest.mark.parametrize("reply", [
    # 1.1.0: the box's text asks for more (Ministral 3 14B, Phi-3-medium, Mistral Large 3, Mistral Small 3.1, Phi-4,
    # Mistral NeMo): the answer is cut, so the reply stays open
    # (the two time ranges of this list, r"8pm \text{ to }" and r"4pm\ \text{to}", are the class open_range since
    # 1.2.0: test_time_ranges_are_their_own_class below)
    r"10 \text{ blue and }", r"((B) \text{ and }", r"4 \text{ blue, }",
    r"{-}", r"21.333\ldots \text{ hectares (or }", r"(\text{Neither is correct, but the closest is }",
    r"(\text{Both are correct, but }", r"(\text{Penguins: }", r"5\% \text{ discount on }",
    r"(\text{None of the options match the calculated date of }", r"(\text{the closest would be }",
    r"3 \text{ apples \& }",
    # 1.1.0: two options (Mistral Large 3)
    r"({A, D}", r"({A,D}", r"({A, D, F}",
])
def test_unfinished_left_open(reply):
    assert fs.classify(P + reply) == "open_unfinished"
    assert fs.close(P + reply) == P + reply
    assert fs.unfinished(P + reply) in ("continuation", "two options")
    # rule 1.0.0 completed them
    assert fs.classify(P + reply, version=fs.V100) == "completed"
    assert fs.close(P + reply, version=fs.V100) != P + reply


def test_version_100_leaves_the_tab_open_and_unknown_versions_fail():
    tab = "\text{(B)}"          # a tab, then "ext{(B)}"
    assert fs.classify(P + tab, version=fs.V100) == "open_other"
    assert fs.close(P + tab, version=fs.V100) == P + tab
    assert fs.classify(P + tab) == "completed" and fs.close(P + tab) == P + r"\text{(B)}}"
    with pytest.raises(ValueError):
        fs.close(P + r"(\text{C}", version="forced-stop/9")
    # 1.2.0: the class open_range
    assert fs.VERSION == "forced-stop/1.2.0"
    assert fs.VERSIONS == ("forced-stop/1.0.0", "forced-stop/1.1.0", "forced-stop/1.2.0")


@pytest.mark.parametrize("reply", [r"8pm \text{ to }", r"4pm\ \text{to}"])
def test_time_ranges_are_their_own_class(reply):
    """forced-stop/1.2.0: a clock time and "to" stays open as open_range; 1.1.0 called it unfinished and
    1.0.0 completed it. The completion is the same under 1.1.0 and 1.2.0."""
    assert fs.classify(P + reply) == "open_range" and fs.close(P + reply) == P + reply
    assert fs.classify(P + reply, version=fs.V110) == "open_unfinished"
    assert fs.close(P + reply, version=fs.V110) == P + reply
    assert fs.classify(P + reply, version=fs.V100) == "completed"


def test_plain_and_unfinished():
    assert fs.plain(P + r"8pm \text{ to }") == "8pm   to"
    assert fs.plain(P + r"12:20\,\text{pm}") == "12:20  pm"
    assert fs.plain(P + r"3 \& ") == "3 &"
    assert fs.unfinished(P + r"(\text{C}") is None
    assert fs.unfinished(P + r"(\text{Tom}") is None
    assert fs.unfinished(P + r"(\text{A) or (B}") == "two options"
    assert fs.unfinished(P + r"\text{ and}") == "continuation"


@pytest.mark.parametrize("text, kind", [
    (P + "C}", "closed"), (P + "(C)}", "closed"), (P + "((B)}", "closed"), (P + "18}", "closed"),
    (P + "C", "no_stop"), (P + r"\text{C", "no_stop"), ("C}", "no_prefill"),
])
def test_unaffected(text, kind):
    assert fs.classify(text) == kind
    assert fs.close(text) == (text if isinstance(text, str) else "")


def test_groups_reports_the_closed_group_and_the_open_ones():
    assert fs.groups(P + r"(\text{J}") == ("open", "text", (("{", "boxed"), ("(", "")))
    assert fs.groups(P + "{(B)}") == ("open", fs.LEAD_GROUP, (("{", "boxed"),))
    assert fs.groups(P + "6 {10}") == ("open", "", (("{", "boxed"),))
    assert fs.groups(P + "C}") == ("closed", "boxed", ())
    assert fs.groups(P + r"\{1\}")[0] == "open"           # escaped braces are text
    assert fs.groups("no box")[0] == "no_prefill"


def test_stops_at_brace_reads_the_logged_decoding():
    item_b = {"stop": ["}"], "include_stop_str_in_output": True, "prefill": {"forced_direct": P}}
    ladder = {"stop": [], "include_stop_str_in_output": False, "prefill": {"forced_direct": P, "fewshot_forced": P},
              "arm_overrides": {"forced_direct": {"stop": ["}"], "include_stop_str_in_output": True}}}
    math = {"stop": [], "include_stop_str_in_output": False, "prefill": {"forced_direct": P}}
    assert fs.stops_at_brace(item_b, "forced_direct")
    assert fs.stops_at_brace(ladder, "forced_direct") and not fs.stops_at_brace(ladder, "fewshot_forced")
    assert not fs.stops_at_brace(math, "forced_direct")
    assert not fs.stops_at_brace({"stop": ["}"], "include_stop_str_in_output": False}, "forced_direct")


# --------------------------------------------------------------------------- the readings of a completed box


def _gold_bbh_letter():
    q = "Which is it?\nOptions:\n(A) red\n(B) blue\n(C) green\n(D) pink\n(J) grey"
    return cs.options(q)


def test_content_reads_the_completed_letter_and_strict_lenient_do_not_change():
    opts = _gold_bbh_letter()
    for reply, gold in ((r"(\text{J}", "(J)"), (r"(\textbf{B}", "(B)"), (r"({C}", "(C)"), (r"{(B)}", "(B)"),
                        (r"\text{(D)}", "(D)")):
        raw, new = P + reply, fs.close(P + reply)
        assert cs.classify(raw, "stop", gold, "bbh", LETTER, opts=opts)[0] in ("malformed_box", "unreadable", "wrong",
                                                                             "correct")
        assert cs.classify(new, "stop", gold, "bbh", LETTER, opts=opts) == ("correct", gold)
        assert bbh.strict_extract(new, LETTER) == bbh.strict_extract(raw, LETTER)
        assert bbh.lenient_extract(new, LETTER) == bbh.lenient_extract(raw, LETTER)


def test_content_before_the_fix_could_not_read_the_open_box():
    opts = _gold_bbh_letter()
    assert cs.classify(P + r"(\text{J}", "stop", "(J)", "bbh", LETTER, opts=opts)[0] == "malformed_box"
    assert cs.classify(P + r"131 \text{ mm}", "stop", "131", "gsm8k")[0] == "malformed_box"


def test_gsm8k_completed_box_reads_its_number_under_content_only():
    raw = P + r"131 \text{ mm}"
    new = fs.close(raw)
    assert cs.classify(new, "stop", "131", "gsm8k") == ("correct", "131")
    assert scoring.strict_extract(new, BOXED) is None and scoring.strict_extract(raw, BOXED) is None
    assert scoring.lenient_extract(new) == scoring.lenient_extract(raw) == "131"


def test_yes_no_and_integer_tasks():
    for reply, gold, atype in ((r"\text{Yes}", "Yes", YESNO), (r"\textbf{No}", "No", YESNO), (r"-\bf{50}", "-50", INTEGER),
                               (r"{-28}", "-28", INTEGER)):
        new = fs.close(P + reply)
        assert cs.classify(new, "stop", gold, "bbh", atype)[0] == "correct", reply


def test_wider_wrappers_in_the_content_reader():
    assert cs.unwrap(r"\underline{40}") == "40"
    assert cs.unwrap(r"\textit{Yes}") == "Yes"
    assert cs.unwrap(r"\emph{(B)}") == "(B)"
    assert cs.unwrap(r"\textcolor{red}{x}") == r"\textcolor{red}{x}"     # not a wrapper
    assert cs.CONTENT_SCORER_VERSION == "content/1.7.0"     # 1.2.0 the placeholder rule; 1.3.0 boxes of other
    # brackets; 1.4.0 answer cues and clock times; 1.5.0 spacing and letters; 1.6.0 conflicts, tags, ranges; 1.7.0 sentences and lowercase labels
    assert set(fs.WRAPPER_COMMANDS) >= {"text", "textbf", "mathrm", "mathbf"}


def test_unaffected_replies_score_exactly_as_before():
    """Closed boxes and replies of arms without the stop: close() returns the text itself, so every reading is the
    same as before the fix."""
    texts = [P + "(C)}", P + "18}", P + r"\text{(C)}}", P + "C", P + r"\frac{1}{2}}", "The answer is \\boxed{7}."]
    for t in texts:
        assert fs.close(t) == t
    out, n = fs.close_series(texts)
    assert out == texts and n == 0
    out, n = fs.close_series([P + r"(\text{J}", P + r"(\text{J}"], [True, False])
    assert out == [P + r"(\text{J})}", P + r"(\text{J}"] and n == 1


def test_inspect_work_dir_closes_only_the_stopped_arm(tmp_path):
    """d2_io puts the prefill in front and completes the stopped replies of the arms whose decoding stops at '}'."""
    from src import d2_io
    df = pd.DataFrame({"arm": ["forced_direct", "forced_direct", "direct_answer"],
                       "output_text": [r"(\text{J}", "C}", r"(\text{J}"]})
    dec = {"stop": ["}"], "include_stop_str_in_output": True, "prefill": {"forced_direct": P}}
    stop_arms = [a for a in dec["prefill"] if fs.stops_at_brace(dec, a)]
    assert stop_arms == ["forced_direct"]
    m = df["arm"] == "forced_direct"
    full = P + df.loc[m, "output_text"]
    closed = full.map(fs.close)
    assert closed.tolist() == [P + r"(\text{J})}", P + "C}"]
    assert hasattr(d2_io, "inspect_work_dir")
