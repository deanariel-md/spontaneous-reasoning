r"""Forced replies stopped at the first "}".

The forced arms open the assistant's turn with "\boxed{". On GSM8K and BIG-Bench Hard their generation stops at the
first "}" (vLLM stop string, kept in the output: config/decoding_forced.yaml and the ladder's arm_overrides); the API
runs (scripts/nebius_panel.py, scripts/mistral_ladder.py) and the arm-wording run (src/arm_wording.cut_at_first_brace) cut their
replies after the first "}" to the same text. MATH-500 and HotpotQA ran without the stop and are never cut.

When that first "}" closes a group inside the box, as in "(\text{C}", the stop leaves the box open, and the strict,
lenient and content readings see no box although the answer may be there. Such a reply is read as the closed box it was
meant to be (completed) when all of these hold:
  * the reply ended on the stop string (its text ends with "}") and its box is still open;
  * the group the stop closed is an answer wrapper: a text or font wrapper (WRAPPER_COMMANDS: \text{...},
    \textbf{...}, \mathrm{...} and the others), or a bare group that opens the box, with nothing but blanks and
    opening brackets before it ("\boxed{{(B)}", "\boxed{{-28}", "\boxed{({C}"); and every group still open inside
    the box is such a wrapper, "(" or "[";
  * (1.1.0) the text of the box so far does not end in a word or sign that asks for more (CONTINUATION_WORDS,
    CONTINUATION_SIGNS: "8pm \text{ to }", "10 \text{ blue and }", "4 \text{ blue, }", "{-}", "but the closest
    is }"), and it does not hold two options ("({A, D}");
then the groups left open are closed, innermost first: "\boxed{(\text{C}" reads as "\boxed{(\text{C})}",
"\boxed{131 \text{ mm}" as "\boxed{131 \text{ mm}}". Before that, (1.1.0) a tab followed by the rest of a wrapper whose
name starts with "t" ("<TAB>ext{(B)}", the escape "\t" of "\text" written as a tab, DeepSeek) is read as that wrapper.
Otherwise part of the answer may be missing, and the reply is left as it is (read as a malformed box): the numerator of
\frac{16}{..}, a group after other text ("6 {10}", "${20,000}"), an exponent "(1.2)^{18}", a repeating decimal
"33.\overline{3}", "\ref{answer}" (and "\ref{A}": below), the colour of \textcolor{..}{..}, a text that stops on a
connective or holds two options (above), or a box inside the prefilled one (class open_inner_box), which the readers
already read as a complete box: they read the last complete box of a reply, whatever stands before it in the open
prefilled box.

Inner boxes (counted over every forced reply the
audit reads). What stands between the prefill and the inner "\boxed{" is, in 9,392 replies, nothing but blanks and
opening brackets ("\boxed{(\boxed{K}"); in 32, other text ("\boxed{230 - 225 = \boxed{5}"); in 6, a closed reasoning
block: Qwen3.5-122B-A10B's five-shot BIG-Bench Hard forced replies "\boxed{()\n</think>\n\n\boxed{(C)}", where the model
wrote an empty "()", closed a reasoning block it never opened, and began a new box, which the stop closed. No other
model, panel or ladder, writes such a restart. The rule is the same for every model and every form: the inner box is
the reply's last complete box and is read (5 of the 6 restarts are right, about 0.01 pp of that arm); the reading is
stated with the ladder's results.

"(\ref{X}" is not read as the option X. \ref is LaTeX's cross-reference to a label, and the one model that writes it
(Mistral-7B-Instruct-v0.1, five-shot BIG-Bench Hard, 1,280 replies) uses it as a label: 1,080 of its arguments are not a
letter ("answer" 719, "eq:1" 313, "table:penguins" and others), and of the 200 single letters 149 are "A", written on
items whose answer is (A), (B) and (C) alike (58, 55 and 34 times; 2 more on (D)). 86 of the 200 letters equal the answer (0.18 pp of that
arm); they stay unread, a loss the rule does not recover.

Versions (VERSION). forced-stop/1.0.0: the wrapper and open-group conditions only. forced-stop/1.1.0:
adds the continuation and two-option conditions and the tab reading. close() and
classify() take version= to apply 1.0.0 (for comparisons only). forced-stop/1.2.0: the class
open_range (below); the completion is unchanged, so close() returns the same text under 1.1.0 and 1.2.0.

How often the completion misreads (measured against the API models' full replies, which keep the text after the first
"}"; scripts/forced_stop_full_replies.py, results/forced_stop/full_replies.md). The stop at the first "}" hides what
follows, so a completed box can read one answer where the model went on to write more:
  * 1.0.0: of 13,706 completed replies (Mistral Large 3 9,338, DeepSeek 4,368), 13,562 close their box in the full
    reply (144 never do: the token cap came first), and 97 of these read another answer or class than the full box:
    78 BIG-Bench Hard time ranges cut at "to" ("1pm \text{ to }", "6am\ \text{to}", read unreadable, where the model
    wrote "1pm to 5pm" or "6am\ \text{to}\ 7am", the option: false losses); 16 lead groups that hold, or go on to, a
    second option ("({A, D, F}", "({B}" where the model wrote "({B}, {F})"), 2 of them false gains; 3 GSM8K texts cut
    at "and" ("10 \text{ blue and }" where the model wrote "10 blue and 6 red"), 2 of them false gains. False gains 4,
    false losses 78.
  * 1.1.0: of 13,615 completed replies (Mistral Large 3 9,238, DeepSeek 4,377, the 9 tab replies included), 13,486
    close their box in the full reply (129 never do), and 12 read another answer or class than the full box: all are
    lead groups where the model went on to a second option ("({B}" where it wrote "({B}, {F})"), which the cut text
    cannot show; 1 is a false gain, none a false loss. The 78 time ranges are left open (open_unfinished in 1.1.0,
    open_range in 1.2.0). Over every compared reply, completed or left open, the 1.1.0 reading with content/1.5.0
    agreed with the full box on 15,394 of 15,474 (2 false gains, 78 false losses).
  * 1.2.0: the completion is that of 1.1.0; a box left open whose text so far
    is a clock time and "to" ("6am \text{ to }") is classed open_range. The 54 temporal_sequences items' options all begin at different
    times, so the start names one option, and src/content_scoring.py reads it (content/1.6.0, range_option): the
    end the stop cut is not needed. The 78 Mistral Large 3 ranges are then read as the option the full box holds
    (results/forced_stop/full_replies.md); the replies the stop leaves unread are open_unfinished, open_other, and
    open_range replies the content reader does not read (scripts/forced_stop_audit.py).
  Of the 78 full boxes, 47 are written with LaTeX
  spacing ("6am\ \text{to}\ 7am"), which content readers before content/1.5.0 left unreadable.
Strict and lenient readings are not touched by either version (a completed box is never in the strict format).

The readers then read the completed box by their own rules, which this module does not change: the strict
rules (src/scoring.py, gsm8k-scorer/1.3.0; src/bbh.py, bbh-boxed/1.0.0) read no wrapper inside a box, so a completed
box never becomes strict-compliant, and the lenient rules read the same tokens as before; the content reading
(src/content_scoring.py, since content/1.1.0) removes the wrappers. Replies that close their box and replies of arms without
the stop are never changed.

Callers: src/d2_io.inspect_work_dir (every collection; the arms whose logged decoding stops at "}"),
src/arm_wording.score (after its cut), scripts/nebius_panel_report.score_task (DeepSeek on Nebius and Mistral
Large 3, after their cut).
"""
from __future__ import annotations

import re

VERSION = "forced-stop/1.2.0"
V100 = "forced-stop/1.0.0"
V110 = "forced-stop/1.1.0"
VERSIONS = (V100, V110, VERSION)
BOX_PREFILL = "\\boxed{"
STOP_BRACE = "}"
WRAPPER_COMMANDS = ("text", "textbf", "textit", "textrm", "textsf", "texttt", "textup", "textnormal", "emph", "mbox",
                    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "boldsymbol", "bm", "bf", "underline")
LEAD_GROUP = "{"      # the command recorded for a bare group that opens the box's content
# 1.1.0: a box whose text so far ends in one of these asks for more (the list of endings seen, and the endings of completed
# wrappers in the stored replies that ask for a continuation: "but", "is", "be", "of", "on", ":"; every ending is
# listed by scripts/forced_stop_audit.py)
CONTINUATION_WORDS = ("and", "or", "to", "but", "is", "be", "of", "on")
CONTINUATION_SIGNS = (",", ":", "&", "-")
RULE = ("a forced reply that the stop at the first '}' ended (or that was cut after its first '}') with its \\boxed{ "
        "still open is completed when the group the stop closed is an answer wrapper (a text or font wrapper: "
        + ", ".join("\\" + c for c in WRAPPER_COMMANDS) + "; or a bare group that opens the box), every group "
        "still open inside the box is such a wrapper, '(' or '[', and the box's text so far neither ends in a word or "
        "sign that asks for more (" + ", ".join(CONTINUATION_WORDS) + "; " + " ".join(CONTINUATION_SIGNS) + ") nor "
        "holds two options: the open groups are closed innermost first (a tab followed by the rest of a wrapper "
        "whose name starts with 't' is read as that wrapper); otherwise the reply is read as it is")
CLASSES = ("no_prefill", "no_stop", "closed", "completed", "open_inner_box", "open_range", "open_unfinished",
           "open_other")

_CMD_BEFORE = re.compile(r"\\([A-Za-z]+)\s*$")
_LEAD_CHARS = set(" \t\n([")     # what may stand before a bare group that opens the box: blanks and opening brackets
_CLOSER = {"{": "}", "(": ")", "[": "]"}
_TAB_WRAPPER = re.compile("\t(" + "|".join(sorted((c[1:] for c in WRAPPER_COMMANDS if c.startswith("t")), key=len,
                                                  reverse=True)) + r")(?=\s*\{)")
_SPACING = re.compile(r"\\[ ,;:!]")         # LaTeX spacing commands: a blank
_COMMAND = re.compile(r"\\[A-Za-z]+\*?")     # a command name: removed
_ESCAPE = re.compile(r"\\(.)")               # an escaped character ("\&", "\%"): the character
_LAST_WORD = re.compile(r"(?<![A-Za-z])([A-Za-z]+)$")
# 1.2.0: a box whose text so far is a clock time and "to" ("6am \text{ to }", "11 a.m. to"), a range the stop cut
_RANGE_CUT = re.compile(r"^\(?\s*\d{1,2}(?::[0-5]\d)?\s*(?:[ap]\.?\s*m\.?)?\s+to$", re.IGNORECASE)
_TWO_OPTIONS = re.compile(r"(?<![A-Za-z])[A-R]\s*\)?\s*(?:,|/|&|\band\b|\bor\b)\s*\(?\s*[A-R](?![A-Za-z])"
                          r"|\(\s*[A-R]\s*\).*\(\s*[A-R]\s*\)", re.S)


def groups(text, prefill: str = BOX_PREFILL) -> tuple[str, str | None, tuple]:
    """How the groups stand at the end of a prefilled forced reply: (state, closed, open_groups).

    state is "no_prefill" (the text does not start with the prefill), "closed" (the box closes) or "open". `closed`
    names the group the last "}" closed: the command before its "{" ("text", "frac", "boxed"), LEAD_GROUP for a bare "{"
    that opens the box's content (only blanks and opening brackets before it, as in "({C}"), "" for another bare "{",
    None when no "}" closed a group. `open_groups` lists the groups still open, outermost first, the box included, each
    as (opener, command). A backslash escapes the next character; a "(" or "[" left open inside a group that a "}" closes
    is plain text, as in TeX."""
    t = text if isinstance(text, str) else ""
    if not t.startswith(prefill):
        return "no_prefill", None, ()
    stack: list[tuple[str, str]] = [("{", "boxed")]
    closed = None
    i, n = len(prefill), len(t)
    while i < n:
        c = t[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            m = _CMD_BEFORE.search(t, 0, i)
            lead = (all(o != "{" for o, _ in stack[1:])
                    and set(t[len(prefill):i]) <= _LEAD_CHARS)
            stack.append(("{", m.group(1) if m else (LEAD_GROUP if lead else "")))
        elif c == "}":
            while stack[-1][0] != "{":
                stack.pop()
            closed = stack.pop()[1]
            if not stack:
                return "closed", closed, ()
        elif c in "([":
            stack.append((c, ""))
        elif c in ")]":
            if stack[-1][0] == ("(" if c == ")" else "["):
                stack.pop()
        i += 1
    return "open", closed, tuple(stack)


def is_wrapper(cmd) -> bool:
    """A group that wraps the answer: a text or font wrapper, or a bare group that opens the box."""
    return cmd in WRAPPER_COMMANDS or cmd == LEAD_GROUP


def plain(text, prefill: str = BOX_PREFILL) -> str:
    """The box's text so far as words: command names and braces removed, spacing commands as blanks, escaped
    characters as themselves; trailing blanks removed."""
    t = text if isinstance(text, str) else ""
    t = t[len(prefill):] if t.startswith(prefill) else t
    t = _ESCAPE.sub(r"\1", _COMMAND.sub(" ", _SPACING.sub(" ", t)))
    return t.replace("{", "").replace("}", "").rstrip()


def unfinished(text, prefill: str = BOX_PREFILL) -> str | None:
    """Why the box's text so far cannot be the whole answer (1.1.0): 'continuation' (it ends in a word or sign that
    asks for more) or 'two options'; None otherwise."""
    p = plain(text, prefill)
    m = _LAST_WORD.search(p)
    if (p and p[-1] in CONTINUATION_SIGNS) or (m and m.group(1).casefold() in CONTINUATION_WORDS):
        return "continuation"
    if _TWO_OPTIONS.search(p):
        return "two options"
    return None


def range_cut(text, prefill: str = BOX_PREFILL) -> bool:
    """1.2.0: whether the box's text so far (plain) is a clock time and "to", a time range the stop cut before its end
    ("6am \\text{ to }", "6am\\ \\text{to}")."""
    return bool(_RANGE_CUT.match(plain(text, prefill)))


def _reading(text, prefill: str, version: str) -> tuple[str, str]:
    """(kind, text read) for a reply that ends on the stop: kind 'closed', 'completed', 'open_inner_box',
    'open_range', 'open_unfinished' or 'open_other' (classify), and the text the readers get."""
    if version not in VERSIONS:
        raise ValueError(f"unknown rule version {version!r}")
    t = text if isinstance(text, str) else ""
    state, closed, open_groups = groups(t, prefill)
    if state == "closed":
        return "closed", t
    if version != V100 and not is_wrapper(closed) and "\t" in t:
        r = _TAB_WRAPPER.sub(lambda m: "\\t" + m.group(1), t)
        if r != t:
            k, read = _reading(r, prefill, version)
            if k == "completed":
                return k, read
    if not is_wrapper(closed) or any(o == "{" and not is_wrapper(cmd) for o, cmd in open_groups[1:]):
        return ("open_inner_box" if closed == "boxed" else "open_other"), t
    if version != V100 and unfinished(t, prefill):
        if version not in (V100, V110) and range_cut(t, prefill):
            return "open_range", t
        return "open_unfinished", t
    return "completed", t + "".join(_CLOSER[o] for o, _ in reversed(open_groups))


def close(text, prefill: str = BOX_PREFILL, version: str = VERSION) -> str:
    """The prefilled text of a forced reply that the stop at the first "}" ended, read as the closed box it was meant
    to be (module doc). Unchanged when it does not end on the stop string, when its box closes, or when it is left
    open (the group the stop closed or a group still open is not an answer wrapper; 1.1.0: or its text asks for more
    or holds two options)."""
    t = text if isinstance(text, str) else ""
    if not t.endswith(STOP_BRACE) or not t.startswith(prefill):
        return t
    return _reading(t, prefill, version)[1]


def classify(text, prefill: str = BOX_PREFILL, version: str = VERSION) -> str:
    """For counts: 'no_prefill'; 'no_stop' (the reply does not end on "}"); 'closed' (its box closes); 'completed'
    (close() closes it); 'open_inner_box' (left open: the stop closed a nested \\boxed{...}, which the readers read as
    a complete inner box); 'open_range' (1.2.0; left open: a clock time and "to", a time range cut before its end,
    which the content reader reads on temporal_sequences); 'open_unfinished' (1.1.0; left open: the stop closed an
    answer wrapper, but the text asks
    for more or holds two options); 'open_other' (left open: the group the stop closed, or a group still open, is not
    an answer wrapper)."""
    t = text if isinstance(text, str) else ""
    if not t.startswith(prefill):
        return "no_prefill"
    if not t.endswith(STOP_BRACE):
        return "no_stop"
    return _reading(t, prefill, version)[0]


def closed_group(text, prefill: str = BOX_PREFILL) -> str | None:
    """The command of the group the stop closed (for counts of the forms)."""
    return groups(text, prefill)[1]


def stops_at_brace(decoding: dict, arm: str) -> bool:
    """Whether a logged decoding stops the arm at "}" with the stop string kept (the arm's override, else the file's
    stop list)."""
    d = decoding or {}
    ov = (d.get("arm_overrides") or {}).get(arm) or {}
    stop = ov.get("stop", d.get("stop")) or []
    keep = ov.get("include_stop_str_in_output", d.get("include_stop_str_in_output", False))
    return STOP_BRACE in list(stop) and bool(keep)


def close_series(texts, mask=None, prefill: str = BOX_PREFILL, version: str = VERSION) -> tuple[list, int]:
    """close() applied to texts (where mask is true); returns (texts, number changed)."""
    out, n = [], 0
    for k, t in enumerate(texts):
        if mask is None or mask[k]:
            c = close(t, prefill, version)
            n += c != t
            out.append(c)
        else:
            out.append(t)
    return out, n
