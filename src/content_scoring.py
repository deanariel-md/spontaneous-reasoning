"""Content-level reading of an output: the answer the output commits to,
read the same way in every arm, and the reason when there is none.

Classes, checked in this order:
    truncated      the generation hit its token cap (finish_reason "length") before any box (complete, or written
                   with other brackets, below) that is not a placeholder;
    correct, wrong the content of the last box that is not a placeholder (below), read for the task (below), the
                   complete \\boxed{...} and the boxes written with other brackets taken in the order of their
                   openings; with no such box, the output with its placeholders removed, read the same way when it is
                   nothing but an answer, else its last answer token (the lenient reading);
    unreadable     a complete box whose content holds no answer of the task's kind, or no box and no answer token
                   (a reply whose only box is a placeholder and which holds no answer elsewhere is unreadable);
    malformed_box  a box is opened ("\\boxed{") but never closed.

Placeholders (content/1.2.0; one rule for every benchmark, also used by src/hotpot_task.content_read). A box is a
placeholder, not an answer, when its content, with LaTeX text and font wrappers, Markdown bold and "$" removed,
whitespace collapsed, surrounding quotes and a final period dropped and case folded, is
    P1  empty;
    P2  the name of the answer's slot: "answer" or "number" (the two instruction placeholders), with optional words
        before it from SLOT_QUALIFIERS ("final answer", "your answer", "the correct answer") and an optional tail
        from SLOT_TAILS ("your answer here"); bare or wholly in angle brackets;
    P3  wholly in angle brackets, a slot noun of SLOT_NOUNS (answer, number, letter, option, choice, value, result,
        solution) with the same optional words ("<letter>", "<final result>", "<your answer here>").
A placeholder box is skipped when the reader picks the last box, and it is removed from the text before the reading
without a box, as is a copy written without braces ("\\boxed<answer>", "boxed<number>"; PLACEHOLDER_COPY). Angle
brackets around anything else are a filled slot and are read: "<B>", "<14>", "<not plausible>", "<Answer: (B)>". Across every collection scored for content,
angle-bracketed letters are right 58.5% of the time against 21.0% if they were unrelated to the items, numbers 19.9%
against 1.9%, words 53.9% against 20.0%, so they are answers.

Options named with their letter (content/1.5.0; OPTION_LETTER). "<Option A>", "<Option (B)>", "<choice
C>" name an option by its letter and are read as that letter on a letter task; slot names without a letter
("<option>", "<letter>") stay placeholders. From content/1.2.0 to 1.4.0 an option template with a letter was a
placeholder (P3), on the argument that DeepSeek V4.1-Flash-1017's zero-shot forced "<Option A>" (238 BIG-Bench Hard
replies on 56 items) was right exactly as often as (A) is the answer on its items, a copied template. That argument
was circular: if every reply says A, its share right is the share of (A) by construction. The test is whether the
letter tracks the items: on the arm's 113 two-option items the answer is
(A) on 40.7%; on the 56 items where DeepSeek writes "<Option A>" it is (A) on 37 (66.1%; one-sided binomial p = 0.0001
against 40.7%), and its 12 "<Option B>", "<Option (B)>", "<Option C>" and "<Option D>" replies (10 items) are all
right. The letter carries information about the item, so the box is the model's choice.

Boxes written with other brackets (content/1.3.0; OTHER_BOX). A box written with angle brackets in place
of its braces, the backslash optional ("\\boxed<Odysseus>", "boxed <24>"), or as a tag ("<boxed>Liguria</boxed>",
the closing ">" optional), its content on one line and opening outside every complete \\boxed{...}, is a box holding
its content X. It is a placeholder when X, or X in angle brackets, is one under the rule above ("\\boxed<answer>",
"\\boxed<>", "<boxed><number></boxed>"); such a placeholder is removed from the text as well. These are the model's
answer in a box with the wrong brackets, so reading them is part of the content reading; strict and lenient, the
panel's format rules, do not read them. On the main panel, 1,964, 27 and 16
replies of BIG-Bench Hard, GSM8K and MATH-500 hold a filled angle box, and 830, 129 and 124 a filled tag.

Reading a box's content: LaTeX text and font wrappers (\\text{...}, \\textbf{...}, \\mathrm{...} and the others of
src/forced_stop.WRAPPER_COMMANDS), Markdown bold, a leading copied instruction placeholder ("<answer>" or "<number>",
with an optional "-" or ":"), a leading answer cue (ANSWER_PREFIX, below) and angle brackets around the whole content
("<not plausible>") are removed, and LaTeX spacing commands are read as blanks (SPACING, below); then
    GSM8K          a number (src/scoring.normalize_number) or a clock time (CLOCK, below), else the last number inside
                   the box, a clock time counting as one number;
    MATH-500       one rational number (src/math_task.normalize_answer), the whole content or its part after the last
                   "="; no number is searched for inside the box; the instruction's placeholder is "<number>";
    BIG-Bench Hard the task's answer type (src/bbh.normalize_answer); for a letter task, an option named with its
                   letter ("<Option A>", above), or the text of exactly one of the item's options ("(A) pink ball"
                   read from "pink ball"; a number 0-20 also as its English word, for options spelled out), else an
                   option letter the box opens with, followed by more text (LEAD_LETTER, below); for a Yes/No task,
                   True and False as Yes and No, and the task's own wording of the two answers
                   (sports_understanding: plausible, not plausible, implausible, alone or as the unqualified sentence
                   "The sentence is not plausible." (below); web_of_lies: "... tells the truth", "... lies"); else
                   the last answer token inside the box.
An answer that is not one of the item's options is unreadable (it cannot be scored against the options).

LaTeX spacing (content/1.5.0; SPACING, spacing_off). A backslash before a blank, "\\,", "\\;", "\\:", "\\!",
"\\quad", "\\qquad" and "~" are read as blanks in a box's content and in the placeholder rule: "6am\\ \\text{to}\\ 7am"
reads "6am to 7am" and "None\\ of\\ the\\ above" reads "None of the above", which match their options; until 1.4.0 the
backslashes stayed in the text and no option matched (unreadable). A thin space between digit groups (a digit, an
optional comma, the command, three digits) is a thousands separator and reads as a comma, as src/scoring.lenient_extract
reads "1\\,000": "5,\\!700" reads 5,700, where 1.4.0 read its last group, 700. The HotpotQA reader takes the same rule
(hotpot-content/1.4.0).

A box that opens with an option letter (content/1.5.0; LEAD_LETTER, lead_letter). On a letter task, a box
whose content is not read as a whole answer and opens with an option letter X written "(X)" (blanks inside allowed),
"X)", "X.", "X:" or "X -", followed by more text, is read as X: "(C) Star Wars Episode IV - A New Hope" reads (C), where
the last-token reading took the "A" of "A New Hope", and "(B) Just like Hitler. A True demagogue" reads (B), where it
took (A). A bare "X" followed by the text of option X itself ("C 6am to 11am", "C (Ambiguous)") reads X too. A box
whose text after the letter names another option ("(A) or (B)", "A: Inception, D: Predator"; "(Y)", or "Y)", "Y:",
"Y - " not after a letter) commits to no single option by this rule and keeps the last-token reading.

A box whose letter and option text name different options (content/1.6.0). When
the letter X in the forms above is one of the item's option letters and the text after it is, read as a whole, the text
of exactly one option and that option is not X ("(B) two" where "two" is option (C); "C) 02/28/2015" where that date is option (A)), the box holds two answers
and is wrong, as a reply naming two options is (classify returns the answer "(X)|(Y)", CONFLICT). The prompts the
models received letter their options exactly as the benchmark and the scorer do (every prompt of every BIG-Bench Hard collection was checked against its request file), so the
disagreement is the model's own. From content/1.5.0 such a box was read by its letter. A bare "X" followed by another
option's text is not covered: the capital may be the article ("A hexagon"); it keeps the last-token reading.

Letters that are not options (content/1.6.0). On a letter task the answer must be one of the item's
option letters (Options.letters, every letter of the item's "Options:" block). A capital in the lead-letter forms that
is not one of them is no option label: when the text after it is, read as a whole, the text of exactly one option, the
box reads that option ("H) Amy finished fourth" on an item with options (A) to (G) reads the option "Amy finished
fourth"; it names one option, so it is not a box with two answers); otherwise ("E.T. the Extra-Terrestrial" on an item
with options (A) to (D)) the box goes on to the last-token reading. Any letter read that is not one of the item's
options ("(E)", "<Option E>", a last token "I") is unreadable, as the rule above states. Until 1.5.0 such letters
were read and scored wrong.

Answer tags (content/1.6.0; ANSWER_TAG). A pair "<answer>X</answer>" (blanks inside the tags allowed, any
case), X on one line and holding no "boxed", opening outside every complete \\boxed{...}, is a box holding X, taken
with the other boxes in the order of their openings; it is a placeholder when X is one (an empty pair, also over
several lines, "<answer>answer</answer>"), and such a pair is removed from the text before the reading without a box.
A box's content wholly in such a pair, or ending with "</answer>", loses the tags. Until 1.5.0 only a leading
"<answer>" was removed: "<answer>(B)</answer> because (A) is newer" read (A), "<answer>The truck is the oldest.</answer>"
matched no option. No ladder, five-shot or API reply uses the form; the main panel's weakest models do.

Time ranges cut at "to" (content/1.6.0; RANGE_CUT, range_option). On temporal_sequences, a box left open
(the forced arms' stop at the first "}" cut "\\boxed{6am \\text{ to }") whose content, cleaned, is a start and "to"
reads the option whose text begins with that start and " to ", when exactly one does (the 54 items' options all begin
at different times). The completion rule leaves such a box open (src/forced_stop.py, class open_range since
forced-stop/1.2.0); until 1.5.0 it was a malformed box.

Conditions and hedges in the wording rule (content/1.6.0; WORDING_HEDGES). On
web_of_lies the task's own wording is read at the end of the content ("... tells the truth" is Yes). A last sentence that
holds a condition or a hedge ("if", "whether", "unless", "otherwise", "cannot", "unable", "determine", "unclear", ...)
is not read by it: "Ka tells the truth if Shenna lies" and "we can't determine if Kristian tells the truth" state a
condition or no answer, where until 1.5.0 both read Yes. Such a content goes on to the last-token reading (a bare "Yes"
or "No" in it), else it is unreadable.

The plausibility sentence (content/1.7.0; SPORTS_SENTENCE). On sports_understanding the content read as a
whole may also be the task's answer as one unqualified sentence: "sentence" or "statement" (after "the" or "this", or
alone), optionally followed by the sentence judged in double quotes, then "is" and plausible, not plausible or
implausible, nothing else ("The sentence is not plausible."; The sentence "Mookie Betts skated behind the net." is
plausible). Until 1.6.0 only the bare words were read, and these boxes were unreadable. A sentence that adds anything
("Plausible, but unusual", "The sentence is implausible due to its phrasing, though the event is plausible", "The
corrected sentence is plausible") is not read by the rule, as before; on web_of_lies the hedge rule above stays as it is.

Lowercase option labels (content/1.7.0; LOWER_LETTER, LOWER_LEAD). On a letter task a lowercase letter in
the bracket forms of an option label, "(x)" or "x)", is read as the capital: a whole content "(b)" or "b)" reads (B), and
a box that opens with "(x)" or "x)" followed by more text is read under the rules for capitals above (one of the item's
letters, the conflict rule, a second option named in the rest, lowercase labels "(y)", "y)" included): "k) turquoise"
reads (K) and "a) Lolita" (A) when Lolita is option (A). A bare lowercase letter ("a", "b") and the forms "x.", "x:",
"x -" are not labels in lowercase: they open ordinary words and abbreviations ("a", "e.g."). The last-token reading is
unchanged (capitals only). Until 1.6.0 these boxes were unreadable.

Answer cues (content/1.4.0; ANSWER_PREFIX, strip_answer_prefix; also src/hotpot_task.content_clean). A
content that opens with an answer cue, case folded: optional words of SLOT_QUALIFIERS ("the", "final", "correct", ...),
"answer", then ":" or "is" (with an optional ":"), Markdown bold allowed around the word, loses the cue: "Answer: 12",
"The final answer is (B)", "**Answer:** Lisbon" read "12", "(B)", "Lisbon". When an opening bracket "(", "[" or "<"
stands before the cue, it goes too, with its partner at the very end when there is one: "(Answer: Mistress of the
Robes)", "<Answer: 2010". The cue is taken off once more after angle brackets around the whole content are removed
("<Answer: (B)>").

Clock times on GSM8K (content/1.4.0; CLOCK, clock_value). A clock time is H:MM (one or two digits, a colon,
minutes 00 to 59, not part of a longer run of digits and colons), with an optional "am" or "pm" (dots and blanks or LaTeX
spacing allowed). It is one number, its value in hours on the 12-hour clock: H + MM/60, an hour from 13 to 23 taken
less 12. GSM8K writes a time of day as its hour on the 12-hour clock (item 1001: "5-3 = 2:00 pm ... #### 2") and the
hour counts of its items (51, 363, 1000) in hours, so "2:00 pm", "2:00" and "14:00" read 2 and "1:30" reads 1.5, where
until 1.3.0 the last number token of "2:00 pm" was its minutes, 00. A box holding only a clock time reads its value; in
the search for the last number (a box holding more, or the reading without a box) a clock time is one token with that
value, never its minutes alone. "am" and "pm" do not change the value.

Fractions on GSM8K (content/1.6.0; FRACTION, fraction_value). A fraction, "\\frac{a}{b}" ("\\dfrac", "\\tfrac") or "a/b", with an optional sign and an optional whole number
before it ("16 \\frac{2}{15}" and "6\\frac{2}{3}", "3 1/2"; a "$" inside the braces allowed), is one number with its value: a box holding only a fraction reads its value,
and in the search for the last number it is one token with that value. Until 1.5.0 the last number token of
"\\boxed{16 \\frac{2}{15} liters}" was the denominator, 15, scored correct against the gold 15, and
"\\boxed{\\frac{30}{2}}" read 2, not 15. "a/b" inside a date ("12/25/2020") is not a fraction.

The content score is 1 for correct and 0 otherwise. Format compliance stays the strict format
(src/scoring.compliance, src/bbh.compliance) and is reported as its own outcome. Nothing is imputed; each class is
counted.

Versions (CONTENT_SCORER_VERSION). content/1.0.0: the first reading, with
the wrappers \\text, \\textbf, \\mathrm and \\mathbf. content/1.1.0: the wrappers are those of
src/forced_stop.WRAPPER_COMMANDS, the list by which a forced reply stopped at the first "}" is completed; the forced
replies of GSM8K and BIG-Bench Hard reach this reader completed by src/forced_stop.close. content/1.2.0: the placeholder rule above. Until 1.1.0 only the benchmark's own placeholder ("<answer>", or "<number>"
on MATH-500) was skipped, and only as the exact box content: GSM8K's "<number>", "<Option A>", "<final answer>",
"\\text{<answer>}" and empty boxes were read as the answer box. content/1.3.0: the boxes
written with other brackets above; until 1.2.0 they were text, read by the reading without a box. content/1.4.0:
answer cues and clock times above; until 1.3.0 "Answer: 5" in a MATH-500 box was
unreadable, "Answer: pink ball" on a letter task matched no option, and GSM8K's "2:00 pm" read 00. content/1.5.0:
LaTeX spacing read as blanks, options named with their letter read as that letter (no longer
placeholders), and boxes that open with an option letter followed by text read as that letter, all above; and on
GSM8K a braced colon between digits ("2{:}00", LaTeX's colon kept from spacing) is a colon, so "2{:}00\\ \\text{pm}"
reads the clock time 2, where 1.4.0 read 00. content/1.6.0: a box whose letter and option
text name different options is wrong; letters that are not one of the item's options are unreadable; answer tag pairs
are boxes; a temporal_sequences range cut at "to" reads the option it begins; the web_of_lies wording rule does not
read a condition or a hedge; a backslash before a blank and "~"
between digit groups are thousands separators ("1\\ 000" reads 1,000; 1.5.0 read 000); and a GSM8K fraction is one
number with its value, all above. content/1.7.0: the plausibility sentence and lowercase
option labels, above.
"""
from __future__ import annotations

import re

from . import bbh
from . import scoring as sc
from .forced_stop import WRAPPER_COMMANDS

CONTENT_SCORER_VERSION = "content/1.7.0"
CLASSES = ("correct", "wrong", "unreadable", "malformed_box", "truncated")
_BOX = "\\boxed{"
CONFLICT = "|"        # content/1.6.0: joins the two options a box names by its letter and by its text, "(B)|(C)"
PLACEHOLDER = "<answer>"
PLACEHOLDERS = {"gsm8k": "<number>", "math": "<number>", "bbh": "<answer>", "hotpot": "<answer>"}  # each answer format's
INSTRUCTION_PLACEHOLDERS = ("<answer>", "<number>")   # both are recognised on every benchmark
SLOT_WORDS = ("answer", "number")                     # P2: slot names placeholders bare or in angle brackets
SLOT_NOUNS = ("answer", "number", "letter", "option", "choice", "value", "result", "solution")   # P3, in brackets
SLOT_QUALIFIERS = ("the", "your", "my", "final", "correct", "chosen")
SLOT_TAILS = ("here", "goes here", "will be here", "letter", "number", "text")
_QUAL = r"(?:(?:" + "|".join(SLOT_QUALIFIERS) + r")\s+)*"
_TAIL = r"(?:\s+(?:" + "|".join(t.replace(" ", r"\s+") for t in SLOT_TAILS) + r"))?"
_SLOT_BARE = re.compile(r"^" + _QUAL + r"(?:" + "|".join(SLOT_WORDS) + r")" + _TAIL + r"$")
_SLOT_BRACKETED = re.compile(r"^" + _QUAL + r"(?:" + "|".join(SLOT_NOUNS) + r")" + _TAIL + r"$")
# content/1.5.0: an option named with its letter ("option a", "choice (b)", "letter c", the content case
# folded) is a choice of that option, read as the letter; until 1.4.0 it was a placeholder in angle brackets (P3)
OPTION_LETTER = re.compile(r"^(?:option|choice|letter)\s*(?:\(\s*([a-z])\s*\)|([a-z]))$")
# content/1.5.0: LaTeX spacing commands, read as blanks: "\ " (a backslash before a blank), \, \; \: \!,
# \quad, \qquad and "~"; a backslash before them is a line break "\\", not a space
SPACING = re.compile(r"(?<!\\)(?:\\[ ,;:!]|\\q?quad(?![A-Za-z]))|~")
# ... except between digit groups, where a thin space (with a comma before it or not) is a thousands separator, as
# src/scoring.lenient_extract reads "1\,000": "1\,000" and "10,\!000" read "1,000" and "10,000"; content/1.6.0: a backslash before a blank and "~" too ("1\ 000", "1~000"), which 1.5.0 read as a blank, so the last group,
# 000, was the number
_DIGIT_GROUP = re.compile(r"(?<=[0-9]),?(?:\\[ ,!;:]|~)(?=[0-9]{3}(?![0-9]))")
# content/1.6.0: an answer tag pair, "<answer>X</answer>" (blanks inside the tags allowed, any case), X on
# one line, holding no \boxed; a box holding X (ANSWER_TAG); the pair written around a whole box's content is removed
# by unwrap (ANSWER_TAG_WHOLE), as is a closing tag that ends a content (TRAILING_TAG); an empty pair over several
# lines is a placeholder (EMPTY_TAG)
ANSWER_TAG = re.compile(r"<[ \t]*answer[ \t]*>([^\n]*?)<[ \t]*/[ \t]*answer[ \t]*>", re.IGNORECASE)
ANSWER_TAG_WHOLE = re.compile(r"^<[ \t]*answer[ \t]*>(.*)<[ \t]*/[ \t]*answer[ \t]*>$", re.IGNORECASE | re.DOTALL)
TRAILING_TAG = re.compile(r"\s*<[ \t]*/[ \t]*answer[ \t]*>$", re.IGNORECASE)
EMPTY_TAG = re.compile(r"<[ \t]*answer[ \t]*>\s*<[ \t]*/[ \t]*answer[ \t]*>", re.IGNORECASE)
# content/1.6.0: a BIG-Bench Hard temporal_sequences box that the forced arms' stop at the first "}" cut
# after "to" ("\boxed{6am \text{ to }"): a start, then "to", read as the option whose range begins there (RANGE_CUT,
# on the cleaned content of the open box)
RANGE_CUT = re.compile(r"^\(?\s*(\S.*?)\s+to$")
# content/1.5.0: a box that opens with an option letter followed by more text ("(C) Star Wars ...",
# "C) Star Wars", "C. Star Wars", "C: Star Wars", "C - Star Wars"), group 1-4 the letter, group 5 the rest
LEAD_LETTER = re.compile(r"^(?:\(\s*([A-Z])\s*\)|([A-Z])\s*\)|([A-Z])\s*[.:]|([A-Z])\s+[-–—])\s*(\S.*)$",
                         re.DOTALL)
_BARE_LEAD_LETTER = re.compile(r"^([A-Z])\s+(\S.*)$", re.DOTALL)
# a second option named in the rest: "(B)", or "B)", "B:", "B - " standing alone ("(A) and (B)", "A: X, B: Y")
_ANOTHER_LETTER = re.compile(r"\(\s*([A-Z])\s*\)|(?<![A-Za-z(])([A-Z])\s*(?:\)|:|\s[-–—]\s)")
# content/1.7.0: lowercase option labels in the bracket forms. LOWER_LETTER, a whole content "(b)" or "b)"
# (a final period dropped); LOWER_LEAD, a box opening with "(b)" or "b)" followed by more text (group 1 or 2 the
# letter, group 3 the rest); _ANOTHER_LOWER, a second lowercase label "(c)" or "c)" in the rest of such a box
LOWER_LETTER = re.compile(r"^(?:\(([a-z])\)|([a-z])\))$")
LOWER_LEAD = re.compile(r"^(?:\(\s*([a-z])\s*\)|([a-z])\s*\))\s*(\S.*)$", re.DOTALL)
_ANOTHER_LOWER = re.compile(r"\(\s*([a-z])\s*\)|(?<![A-Za-z(])([a-z])\s*\)")
# a placeholder copied without the box's braces: "\boxed<answer>", "\boxed <number>}", "boxed<Option A>"
PLACEHOLDER_COPY = re.compile(r"\\?boxed\s*(<[^<>{}\n]{0,40}>)\}?")
# a box written with other brackets (content/1.3.0), on one line: angle brackets in place of the braces,
# the backslash optional ("\boxed<Odysseus>", "boxed <24>"; group 1), or a tag ("<boxed>Liguria</boxed>", the closing
# ">" optional; group 2)
OTHER_BOX = re.compile(r"\\?boxed\s*<([^<>\n]*)>|<[ \t]*boxed[ \t]*>([^\n]*?)<[ \t]*/[ \t]*boxed[ \t]*>?",
                       re.IGNORECASE)
_WRAPPER = re.compile(r"\\(?:" + "|".join(sorted(WRAPPER_COMMANDS, key=len, reverse=True)) + r")\s*\{([^{}]*)\}")
_OPTION = re.compile(r"^\(([A-Z])\)\s*(.*?)\s*$", re.MULTILINE)
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"
# content/1.7.0: the plausibility verdict as one unqualified sentence, on the cleaned content: "the
# sentence is not plausible", 'the sentence "mookie betts skated behind the net." is plausible'
SPORTS_SENTENCE = r"^(?:(?:the|this)\s+)?(?:sentence|statement)(?:\s+(?:\"[^\"]*\"|“[^“”]*”))?\s+is\s+"
TASK_WORDS = {
    "sports_understanding": ((re.compile(r"^plausible$"), 0),
                             (re.compile(r"^(?:not plausible|implausible)$"), 1),
                             (re.compile(SPORTS_SENTENCE + r"plausible$"), 0),
                             (re.compile(SPORTS_SENTENCE + r"(?:not\s+plausible|implausible)$"), 1)),
    "web_of_lies": ((re.compile(r"(?:\blies|\blying|\bnot tell the truth|n't tell the truth)$"), 1),
                    (re.compile(r"\btells? the truth$"), 0)),
}
# content/1.6.0: a last sentence holding one of these words states a condition or a hedge, not an answer;
# the wording rule of WORDING_TASKS_HEDGED does not read it ("ka tells the truth if shenna lies", "cannot be determined
# if dallas tells the truth")
WORDING_HEDGES = ("if", "whether", "unless", "otherwise", "cannot", "can't", "can not", "unable", "undetermined",
                  "determine", "determined", "unclear", "unknown", "impossible", "not sure", "depends", "depending")
WORDING_HEDGE = re.compile(r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in WORDING_HEDGES) + r")\b")
WORDING_TASKS_HEDGED = ("web_of_lies",)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n")


def hedged(k: str) -> bool:
    """content/1.6.0: whether the last sentence of a cleaned content holds a condition or a hedge (WORDING_HEDGES)."""
    return bool(WORDING_HEDGE.search(_SENTENCE_END.split(k)[-1]))


_ANGLE = re.compile(r"^<([^<>]*)>$")
_LEAD_PLACEHOLDER = re.compile(r"^<(?:answer|number)>\s*[-:]?\s*", re.IGNORECASE)
# content/1.4.0: a leading answer cue, an opening bracket before it optional ("(Answer: X)", "<Answer: X")
_BOLD = r"(?:\*\*|__)?"
ANSWER_PREFIX = re.compile(r"^(?P<open>[(\[<]?)\s*" + _BOLD + r"\s*" + _QUAL + r"answer" + _BOLD
                           + r"(?:\s*[:：]|\s+is\b\s*[:：]?)" + _BOLD + r"\s*", re.IGNORECASE)
_PARTNER = {"(": ")", "[": "]", "<": ">"}
# content/1.4.0: a clock time on GSM8K, H:MM with an optional am or pm (module doc)
CLOCK = re.compile(r"(?<![\d:.,])(\d{1,2}):([0-5]\d)(?![\d:])(?:(?:\s|\\[ ,;:!]|~)*([ap])\.?\s*m\b\.?)?",
                   re.IGNORECASE)
# content/1.5.0: LaTeX's braced colon between digits, "2{:}00"
_BRACED_COLON = re.compile(r"(?<=[0-9])\s*\{\s*:\s*\}\s*(?=[0-9])")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")


def _wrappers_off(s: str) -> str:
    prev = None
    while prev != s:
        prev, s = s, _WRAPPER.sub(r"\1", s)
    return s


def spacing_off(s: str) -> str:
    """The text with LaTeX spacing commands (SPACING) read as blanks (content/1.5.0): "6am\\ \\text{to}\\ 7am" reads
    "6am to 7am" once the wrapper is off; a thin space between digit groups stays a thousands separator ("1\\,000"
    reads "1,000")."""
    if not isinstance(s, str) or ("\\" not in s and "~" not in s):
        return s
    return SPACING.sub(" ", _DIGIT_GROUP.sub(",", s))


def strip_answer_prefix(s: str) -> str:
    """The text without a leading answer cue (ANSWER_PREFIX: "Answer:", "The final answer is", "**Answer:**"); an
    opening bracket before the cue goes with it, and its partner when it ends the text ("(Answer: X)" reads "X")."""
    m = ANSWER_PREFIX.match(s) if isinstance(s, str) else None
    if not m:
        return s
    rest = s[m.end():].strip()
    o = m.group("open")
    if o and rest.endswith(_PARTNER[o]):
        rest = rest[:-1].rstrip()
    return rest


def unwrap(s: str, placeholder: str = PLACEHOLDER) -> str:
    """Content without LaTeX text wrappers, LaTeX spacing commands (read as blanks, content/1.5.0), Markdown bold, a
    leading copied instruction placeholder ("<answer>" or "<number>", on every benchmark; `placeholder` is kept for the
    callers), a leading answer cue (content/1.4.0) or copied angle brackets; content/1.6.0: an answer tag pair around
    the whole content ("<answer>X</answer>" reads X) and a closing "</answer>" that ends it ("X</answer>" reads X)."""
    s = spacing_off(_wrappers_off(s)).replace("**", "").strip()
    m = ANSWER_TAG_WHOLE.match(s)
    if m:
        s = m.group(1).strip()
    s = strip_answer_prefix(_LEAD_PLACEHOLDER.sub("", TRAILING_TAG.sub("", s)))
    m = _ANGLE.match(s)
    return strip_answer_prefix(m.group(1).strip()) if m else s


def _canonical(v) -> str:
    """A Fraction as the scorers write numbers: a terminating decimal by src/scoring.normalize_number, else "a/b"."""
    from fractions import Fraction
    v = Fraction(v)
    d = v.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d != 1:
        return f"{v.numerator}/{v.denominator}"
    q, r = divmod(abs(v.numerator) * 10 ** 12 // v.denominator, 10 ** 12)
    s = ("-" if v < 0 else "") + f"{q}.{r:012d}"
    return sc.normalize_number(s)


def clock_value(m) -> str:
    """The value of a CLOCK match in hours on the 12-hour clock (module doc), as a canonical number."""
    from fractions import Fraction
    h, mm = int(m.group(1)), int(m.group(2))
    if 13 <= h <= 23:
        h -= 12
    return _canonical(h + Fraction(mm, 60))


# content/1.6.0: on GSM8K a fraction is one number with
# its value: "\frac{a}{b}" ("\dfrac", "\tfrac"; one-digit parts may go without braces, a "$" inside the braces is
# allowed) or "a/b", with an optional sign and an optional whole number before it, a mixed number ("16 \frac{2}{15}",
# "6\frac{2}{3}" with or without a blank; "3 1/2" with one). Not after
# a letter, digit, ".", ",", "/" or backslash, and "a/b" not followed by a digit or by "/" or "." and a digit, so a date
# "12/25/2020" and "1.5/2" are not fractions.
FRACTION = re.compile(r"(?<![\w.,/\\])(-[ \t]*)?"
                      r"(?:(?:([0-9]+)[ \t]*)?\\[dt]?frac[ \t]*(?:\{[ \t]*\$?[ \t]*([0-9]+)[ \t]*\}|([0-9]))"
                      r"[ \t]*(?:\{[ \t]*\$?[ \t]*([0-9]+)[ \t]*\}|([0-9]))"
                      r"|(?:([0-9]+)[ \t]+)?([0-9]+)[ \t]*/[ \t]*([0-9]+)(?![0-9]|[/.][0-9]))")


def fraction_value(m) -> str | None:
    """The value of a FRACTION match as a canonical number (a terminating decimal, else "a/b"); None for a zero
    denominator."""
    from fractions import Fraction
    a = m.group(3) or m.group(4) or m.group(8)
    b = m.group(5) or m.group(6) or m.group(9)
    if int(b) == 0:
        return None
    whole = m.group(2) or m.group(7)
    v = Fraction(int(a), int(b)) + (int(whole) if whole else 0)
    return _canonical(-v if m.group(1) else v)


def braced_colon(t: str) -> str:
    """content/1.5.0: a LaTeX braced colon between digits ("2{:}00") written as the colon it is."""
    return _BRACED_COLON.sub(":", t) if "{" in t else t


def gsm8k_whole(s: str) -> str | None:
    """A GSM8K content read as a whole: a number, or a clock time or a fraction (content/1.6.0) with nothing else (a
    final period and surrounding "$" allowed)."""
    v = sc.normalize_number(s)
    if v is not None:
        return v
    t = braced_colon(sc.normalize_text(s).strip()) if isinstance(s, str) else ""
    t = t[:-1].rstrip() if t.endswith(".") else t
    m = CLOCK.fullmatch(t)
    if m:
        return clock_value(m)
    m = FRACTION.fullmatch(t.strip("$ \t"))
    return fraction_value(m) if m else None


def gsm8k_last(text) -> str | None:
    """The last number of a GSM8K text (src/scoring.lenient_extract), a clock time (content/1.4.0, its value in hours)
    and a fraction (content/1.6.0, its value) each counting as one number. Without either, exactly lenient_extract."""
    t0 = sc.normalize_text(text)
    if not t0 or not (":" in t0 or "/" in t0 or "frac" in t0):
        return sc.lenient_extract(t0)
    t = sc._LATEX_THIN_COMMA.sub(",", sc._LATEX_BRACE_COMMA.sub(",", braced_colon(t0)))
    spans = [(m.start(), m.end(), clock_value(m)) for m in CLOCK.finditer(t)]
    masked = t
    for a, b, _ in spans:
        masked = masked[:a] + " " * (b - a) + masked[b:]
    fr = [(m.start(), m.end(), fraction_value(m)) for m in FRACTION.finditer(masked)]
    fr = [x for x in fr if x[2] is not None]
    if not spans and not fr:
        return sc.lenient_extract(t0)
    for a, b, _ in fr:
        masked = masked[:a] + " " * (b - a) + masked[b:]
    last = (-1, None)
    for m in sc._LENIENT_TOKEN.finditer(masked):
        last = (m.start(), m.group(0))
    special = max(spans + fr, key=lambda x: x[0])
    if special[0] > last[0]:
        return special[2]
    return sc.normalize_number(last[1])


def placeholder_key(content: str) -> str:
    """A box's content as the placeholder rule reads it: wrappers, Markdown bold and "$" removed, LaTeX spacing read as
    blanks (content/1.5.0), whitespace collapsed, surrounding quotes and a final period dropped, case folded."""
    t = " ".join(spacing_off(_wrappers_off(content if isinstance(content, str) else "")).replace("**", "")
                 .replace("$", "").split())
    t = t.strip(_QUOTES).strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    return t.strip(_QUOTES).strip().casefold()


def is_placeholder(content: str) -> bool:
    """Whether a box's content is a placeholder (module doc, P1-P3), not an answer."""
    k = placeholder_key(content)
    if not k or _SLOT_BARE.match(k):                       # P1, P2 bare
        return True
    t = ANSWER_TAG_WHOLE.match(k)                         # content/1.6.0: "<answer></answer>", "<answer>answer</answer>"
    if t:
        return is_placeholder(t.group(1))
    m = _ANGLE.match(k)
    if not m:
        return False
    inner = " ".join(m.group(1).split())
    return bool(_SLOT_BRACKETED.match(inner))   # P2, P3 in angle brackets (content/1.5.0: an option with its letter
    #                                             is no longer a placeholder; OPTION_LETTER reads it)


def placeholder_rules() -> dict:
    """The placeholder rule as a record (for the scorers' manifests)."""
    return {"version": CONTENT_SCORER_VERSION,
            "rule": "a box is a placeholder when its content (wrappers, bold and '$' removed, LaTeX spacing read as "
                    "blanks, whitespace collapsed, quotes and a final period dropped, case folded) is empty; or a slot "
                    f"name bare ({_SLOT_BARE.pattern}); or wholly in angle brackets a slot name "
                    f"({_SLOT_BRACKETED.pattern}). It is skipped as the answer box and removed from the text before the "
                    "reading without a box, as is a brace-less copy. An option named with its letter ('<Option A>', "
                    f"{OPTION_LETTER.pattern}) is not a placeholder (content/1.5.0): it is read as that letter.",
            "placeholder_copy": PLACEHOLDER_COPY.pattern}


def spacing_rules() -> dict:
    """The rule for LaTeX spacing commands as a record (for the scorers' manifests)."""
    return {"version": CONTENT_SCORER_VERSION, "spacing": SPACING.pattern, "digit_group": _DIGIT_GROUP.pattern,
            "rule": "LaTeX spacing commands (a backslash before a blank, \\, \\; \\: \\!, \\quad, \\qquad, '~') are read "
                    "as blanks in a box's content and in the placeholder rule; a thin space between digit groups "
                    "(a digit, an optional comma, the command, three digits; content/1.6.0: a backslash before a blank "
                    "and '~' too) is read as a thousands comma"}


def lead_letter_rules() -> dict:
    """The rule for a box that opens with an option letter followed by text, as a record (for the manifests)."""
    return {"version": CONTENT_SCORER_VERSION, "lead_letter": LEAD_LETTER.pattern,
            "bare_lead_letter": _BARE_LEAD_LETTER.pattern, "another_letter": _ANOTHER_LETTER.pattern,
            "lower_letter": LOWER_LETTER.pattern, "lower_lead": LOWER_LEAD.pattern,
            "another_lower": _ANOTHER_LOWER.pattern,
            "rule": "on a letter task, a box's content that is not read as a whole answer and opens with an option "
                    "letter X written '(X)', 'X)', 'X.', 'X:' or 'X -' followed by more text is read as X, unless the "
                    "text after it names another option ('(Y)', or 'Y)', 'Y:', 'Y - ' standing alone, Y not X); a bare "
                    "'X' followed by the text of option X itself is read as X too; an option named with its letter "
                    "('option a', 'choice (b)') is read as that letter; content/1.6.0: when the text after the letter "
                    "is, as a whole, the text of exactly one other option, the box holds two answers and is wrong; a "
                    "capital that is not one of the item's option letters is no option label, and any letter read "
                    "that is not one of the item's options is unreadable; content/1.7.0: a lowercase letter in the "
                    "bracket forms '(x)' and 'x)' is read as its capital under the same rules, as a whole content or "
                    "opening a box, a second lowercase label '(y)' or 'y)' in the rest counting as another option"}


def answer_tag_rules() -> dict:
    """The answer-tag and cut-range rules (content/1.6.0) as a record (for the scorers' manifests)."""
    return {"version": CONTENT_SCORER_VERSION, "answer_tag": ANSWER_TAG.pattern, "whole": ANSWER_TAG_WHOLE.pattern,
            "trailing": TRAILING_TAG.pattern, "empty": EMPTY_TAG.pattern, "range_cut": RANGE_CUT.pattern,
            "rule": "an answer tag pair '<answer>X</answer>' (X on one line, holding no 'boxed'), opening outside "
                    "every complete \\boxed{...}, is a box holding X, taken with the other boxes in the order of their "
                    "openings; it is a placeholder when X is one, and an empty pair over several lines is one; a box's "
                    "content wholly in such a pair, or ending with '</answer>', loses the tags. On temporal_sequences "
                    "a box left open whose cleaned content is a start and 'to' reads the option whose text begins "
                    "with that start and ' to ', when exactly one does"}


def answer_cue_rules() -> dict:
    """The answer-cue and clock-time rules as a record (for the scorers' manifests)."""
    return {"version": CONTENT_SCORER_VERSION, "answer_prefix": ANSWER_PREFIX.pattern, "clock": CLOCK.pattern,
            "braced_colon": _BRACED_COLON.pattern,
            "rule": "a content that opens with an answer cue (optional qualifier words, 'answer', then ':' or 'is') "
                    "loses it, with an opening bracket before it and that bracket's partner at the very end; on GSM8K "
                    "a clock time H:MM (optional am or pm; the colon also as LaTeX's braced colon '{:}', content/1.5.0) "
                    "is one number, H + MM/60 hours on the 12-hour clock"}


def other_box_rules() -> dict:
    """The rule for boxes written with other brackets as a record (for the scorers' manifests)."""
    return {"version": CONTENT_SCORER_VERSION, "pattern": OTHER_BOX.pattern,
            "rule": "a box written with angle brackets in place of its braces (the backslash optional) or as a tag "
                    "(<boxed>X</boxed>, the closing '>' optional), on one line and outside every complete \\boxed{...}, "
                    "is a box holding X; it is a placeholder when X, or X in angle brackets, is one; boxes of both "
                    "kinds are taken in the order of their openings, and the last that is not a placeholder is read; "
                    "a placeholder of either form is removed from the text before the reading without a box"}


def other_box_placeholder(content: str) -> bool:
    """Whether the content X of a box written with other brackets is a placeholder: X as it is, or X in angle brackets
    (so "\\boxed<answer>", "\\boxed<Option A>", "\\boxed<>" and "<boxed><number></boxed>" are placeholders)."""
    c = content if isinstance(content, str) else ""
    return is_placeholder(c) or is_placeholder("<" + c.strip() + ">")


def other_boxes(t: str, spans: list[tuple[int, int]] | None = None) -> list[tuple[int, int, str, str]]:
    """(start, end, content, form) of every box written with other brackets (OTHER_BOX; form "angle" or "tag") that
    does not open inside a complete \\boxed{...} (`spans`, from _box_spans by default)."""
    spans = _box_spans(t) if spans is None else spans
    out = []
    for m in OTHER_BOX.finditer(t):
        if any(a <= m.start() < b for a, b in spans):
            continue
        if m.group(1) is not None:
            out.append((m.start(), m.end(), m.group(1), "angle"))
        else:
            out.append((m.start(), m.end(), m.group(2), "tag"))
    return out


def answer_tags(t: str, spans: list[tuple[int, int]] | None = None) -> list[tuple[int, int, str]]:
    """(start, end, content) of every answer tag pair "<answer>X</answer>" (ANSWER_TAG, content/1.6.0), X on one line
    and holding no "boxed", that does not open inside a complete \\boxed{...} (`spans`, from _box_spans by default)."""
    if "answer" not in t.lower():
        return []
    found = [m for m in ANSWER_TAG.finditer(t) if "boxed" not in m.group(1).lower()]
    if not found:
        return []
    spans = _box_spans(t) if spans is None else spans
    return [(m.start(), m.end(), m.group(1)) for m in found if not any(a <= m.start() < b for a, b in spans)]


def answer_boxes(t: str) -> list[str]:
    """Contents of the boxes the reader may pick, in the order of their openings, placeholders left out: every complete
    \\boxed{...} (as bbh.boxed_contents reads them), every box written with other brackets (other_boxes) and every
    answer tag pair (answer_tags, content/1.6.0)."""
    proper = bbh.boxed_contents(t)
    others = other_boxes(t) if "boxed" in t.lower() else []
    tags = answer_tags(t)
    if not others and not tags:
        return [b for b in proper if not is_placeholder(b)]
    spans = _box_spans(t)
    if len(spans) != len(proper):
        raise ValueError("box spans and contents disagree")
    items = [(a, c) for (a, _), c in zip(spans, proper) if not is_placeholder(c)]
    items += [(a, c) for a, _, c, _ in others if not other_box_placeholder(c)]
    items += [(a, c) for a, _, c in tags if not is_placeholder(c)]
    return [c for _, c in sorted(items, key=lambda x: x[0])]


def _box_spans(t: str) -> list[tuple[int, int]]:
    """(start of "\\boxed{", index after its closing "}") of every complete box, matched as bbh.boxed_contents."""
    out = []
    for m in re.finditer(r"\\boxed\{", t):
        depth, i = 1, m.end()
        while i < len(t):
            c = t[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append((m.start(), i + 1))
                    break
            i += 1
    return out


def drop_placeholders(t: str) -> str:
    """The text with every complete placeholder box and every brace-less placeholder copy replaced by a space."""
    spans = [(a, b) for a, b in _box_spans(t) if is_placeholder(t[a + len(_BOX):b - 1])]
    if spans:
        parts, j = [], 0
        for a, b in spans:
            if a < j:          # a placeholder box inside one already removed
                continue
            parts.append(t[j:a])
            parts.append(" ")
            j = b
        parts.append(t[j:])
        t = "".join(parts)
    t = PLACEHOLDER_COPY.sub(lambda m: " " if is_placeholder(m.group(1)) else m.group(0), t)
    if "answer" in t.lower():      # content/1.6.0: an answer tag pair holding a placeholder, or empty over several lines
        t = EMPTY_TAG.sub(" ", ANSWER_TAG.sub(lambda m: " " if is_placeholder(m.group(1)) else m.group(0), t))
    if "boxed" not in t.lower():
        return t
    # content/1.3.0: a placeholder written with other brackets ("<boxed><answer></boxed>", "\boxed<>")
    return OTHER_BOX.sub(lambda m: " " if other_box_placeholder(m.group(1) if m.group(1) is not None else m.group(2))
                         else m.group(0), t)


def clean(s: str) -> str:
    """Case-folded text with whitespace collapsed and surrounding quotes and a final period removed."""
    t = " ".join(unwrap(s).split()).strip().strip(_QUOTES).strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    return t.strip(_QUOTES).strip().casefold()


class Options(dict):
    """Cleaned option text to its letter (a dict), with `letters`, the set of every option letter of the item, "(A)"
    and so on, those whose text is shared with another option included (content/1.6.0)."""

    def __init__(self, *args, letters=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.letters = frozenset(letters)


def options(question: str) -> Options:
    """Cleaned option text to its letter, from the "Options:" block of a BIG-Bench Hard question; texts shared by two
    options are dropped (they cannot be told apart). The result's `letters` holds every option letter."""
    q = sc.normalize_text(question)
    i = q.rfind("Options:")
    if i < 0:
        return Options()
    seen: dict[str, list[str]] = {}
    letters = []
    for letter, text in _OPTION.findall(q[i:]):
        seen.setdefault(clean(text), []).append(f"({letter})")
        letters.append(f"({letter})")
    return Options({t: ls[0] for t, ls in seen.items() if t and len(ls) == 1}, letters=letters)


def option_letters(opts: dict | None) -> frozenset:
    """The item's option letters: `opts.letters` (options()), else the letters of a plain dict's values."""
    if not opts:
        return frozenset()
    return getattr(opts, "letters", None) or frozenset(opts.values())


def option_of_text(s: str, opts: dict | None) -> str | None:
    """The option whose text a string is, read as the whole-content option reading of _bbh_exact reads it (cleaned; a
    number 0-20 also as its English word), or None."""
    if not opts:
        return None
    k = clean(s)
    if len(k) > 1 and k[0] == "(" and k[-1] == ")" and k[1:-1].strip() in opts:     # "C (Ambiguous)"
        k = k[1:-1].strip()
    if k not in opts and k.isdigit() and int(k) < len(NUMBER_WORDS):
        k = NUMBER_WORDS[int(k)]
    return opts.get(k)


def range_option(open_content: str, opts: dict | None) -> str | None:
    """content/1.6.0: the option a temporal_sequences box cut after "to" names ("6am \\text{ to }" reads the option
    "6am to 7am"): the cleaned content is a start and "to" (RANGE_CUT), and exactly one option's text begins with that
    start and " to "; None otherwise."""
    if not opts:
        return None
    m = RANGE_CUT.match(clean(open_content))
    if not m:
        return None
    hits = {v for t, v in opts.items() if t.startswith(m.group(1) + " to ")}
    return next(iter(hits)) if len(hits) == 1 else None


def _gsm8k_answer(content: str) -> str | None:
    c = unwrap(content).strip()
    v = gsm8k_whole(c)
    return v if v is not None else gsm8k_last(c)


def _math_answer(content: str) -> str | None:
    """A MATH box: the content, or its part after the last "=" ("x = 5"), read as one rational number. No search for
    a number inside it: a box holds expressions, and "\\sqrt{2}" or "2\\pi" hold the digit 2 without being 2."""
    from . import math_task
    for c in (content, unwrap(content, PLACEHOLDERS["math"])):   # as strict reads it first, then unwrapped
        v = math_task.normalize_answer(c.strip())
        if v is None and "=" in c:
            v = math_task.normalize_answer(c.rsplit("=", 1)[1])
        if v is not None:
            return v
    return None


def _bbh_exact(content: str, atype, opts: dict | None, task_name: str | None) -> str | None:
    """The content read as a whole answer (no token search), or None."""
    c = unwrap(content).strip()
    v = bbh.normalize_answer(c, atype)
    if v is not None:
        return v
    k = clean(c)
    if atype.kind == "letter":
        m = LOWER_LETTER.match(c[:-1] if c.endswith(".") else c)    # content/1.7.0: "(b)", "b)"
        if m:
            return f"({(m.group(1) or m.group(2)).upper()})"
        m = OPTION_LETTER.match(k)          # content/1.5.0: "<Option A>", "Choice (B)"
        if m:
            return f"({(m.group(1) or m.group(2)).upper()})"
    if atype.kind == "letter" and opts:
        if k not in opts and k.isdigit() and int(k) < len(NUMBER_WORDS):
            k = NUMBER_WORDS[int(k)]
        return opts.get(k)
    if atype.kind == "words" and atype.words[0].casefold() == "yes":
        if k in ("true", "false"):
            return atype.words[0 if k == "true" else 1]
        for rx, j in TASK_WORDS.get(task_name, ()):
            if rx.search(k):
                if task_name in WORDING_TASKS_HEDGED and hedged(k):     # content/1.6.0: a condition or a hedge
                    return None
                return atype.words[j]
    return None


def lead_letter(content: str, opts: dict | None) -> str | None:
    """content/1.5.0: the option letter a letter task's box opens with when more text follows (module doc), or None.
    "(C) Star Wars Episode IV - A New Hope" reads (C), where the last-token reading took the "A" of "A New Hope"; a
    box that names a second option ("(A) or (B)", "A: X, B: Y") is left to the last-token reading. content/1.7.0: a
    lowercase label in the bracket forms, "(k)" or "k)", is read as its capital under the same rules ("k) turquoise")."""
    c = unwrap(content).strip()
    m = LEAD_LETTER.match(c)
    low = None if m else LOWER_LEAD.match(c)
    if m or low:
        if m:
            x, rest = next(g for g in m.groups()[:4] if g), m.group(5)
        else:
            x, rest = (low.group(1) or low.group(2)).upper(), low.group(3)
        y = option_of_text(rest, opts)
        letters = option_letters(opts)
        if letters and f"({x})" not in letters:   # content/1.6.0: not one of the item's letters, no option label:
            return y                              # the text when it is one option's ("H) Amy finished fourth"), else None
        if y is not None and y != f"({x})":    # content/1.6.0: the letter and the text name different options
            return f"({x}){CONFLICT}{y}"
    else:
        b = _BARE_LEAD_LETTER.match(c)
        if not b or not opts:
            return None
        x, rest = b.group(1), b.group(2)
        k = clean(rest)
        if len(k) > 1 and k[0] == "(" and k[-1] == ")":      # "C (Ambiguous)"
            k = k[1:-1].strip()
        if opts.get(k) != f"({x})":
            return None
    for a, b in _ANOTHER_LETTER.findall(rest):
        if (a or b) != x:
            return None
    if low:                      # content/1.7.0: a second lowercase label, "a) Lolita or b) Bob"
        for a, b in _ANOTHER_LOWER.findall(rest):
            if (a or b).upper() != x:
                return None
    return f"({x})"


def conflict_parts(ans) -> tuple[str, str] | None:
    """(letter, text's option) of an answer read from a box whose letter and option text name different options
    (content/1.6.0), or None."""
    if isinstance(ans, str) and CONFLICT in ans:
        a, b = ans.split(CONFLICT, 1)
        return a, b
    return None


def classify(text, finish_reason, gold: str, task: str, atype=None, opts: dict | None = None,
             task_name: str | None = None) -> tuple[str, str | None]:
    """(class, answer read) for one output; `atype`, `opts` (from `options`) and `task_name` describe a BIG-Bench
    Hard item."""
    t = sc.normalize_text(text if isinstance(text, str) else "")
    ph = PLACEHOLDERS.get(task, PLACEHOLDER)
    boxes = answer_boxes(t)
    if finish_reason == "length" and not boxes:
        return "truncated", None
    if task == "gsm8k":
        read_box = _gsm8k_answer
        read_exact = lambda s: gsm8k_whole(unwrap(s).strip())  # noqa: E731
        read_all = gsm8k_last
    elif task == "math":
        from . import math_task
        read_box = _math_answer
        read_exact = lambda s: (math_task.normalize_answer(s.strip())  # noqa: E731
                                or math_task.normalize_answer(unwrap(s, ph).strip()))
        read_all = math_task.lenient_extract
    elif task == "bbh":
        if atype is None:
            raise ValueError("a BIG-Bench Hard item needs its answer type")
        read_exact = lambda s: _bbh_exact(s, atype, opts, task_name)  # noqa: E731
        lead = (lambda c: lead_letter(c, opts)) if atype.kind == "letter" else (lambda c: None)  # noqa: E731
        read_box = lambda c: read_exact(c) or lead(c) or bbh.lenient_extract(unwrap(c), atype)  # noqa: E731
        read_all = lambda s: bbh.lenient_extract(s, atype)  # noqa: E731
    else:
        raise ValueError(f"unknown task {task!r}")
    if boxes:
        ans = read_box(boxes[-1])
    else:
        rest = drop_placeholders(t)
        if _BOX in rest:
            if task == "bbh" and task_name == "temporal_sequences" and opts:     # content/1.6.0: a range cut at "to"
                ans = range_option(rest[rest.rfind(_BOX) + len(_BOX):], opts)
                if ans is not None:
                    return ("correct" if ans == gold else "wrong"), ans
            return "malformed_box", None
        ans = read_exact(rest) or read_all(rest)
    if ans is None:
        return "unreadable", None
    if task == "bbh" and atype.kind == "letter":
        if conflict_parts(ans):                  # content/1.6.0: two answers, the letter and another option's text
            return "wrong", ans
        letters = option_letters(opts)
        if letters and ans not in letters:       # content/1.6.0: not one of the item's options
            return "unreadable", None
    return ("correct" if ans == gold else "wrong"), ans
