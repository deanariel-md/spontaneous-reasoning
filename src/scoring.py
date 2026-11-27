r"""Dataset 2 scoring: strict extraction, lenient extraction, format compliance.

The primary score is correctness under strict extraction,
interpreted as correct-and-extractable output. Lenient extraction and format compliance are
mandatory sensitivities. Everything that depends on the requested final-answer format is
carried by a `FormatSpec`; the format itself is an investigator choice that is NOT fixed
here. The specs in `CANDIDATE_SPECS` are candidates for that choice, not decisions.

Number normalization (shared by gold parsing, strict and lenient extraction)
----------------------------------------------------------------------------
`normalize_number` maps a numeric string to a canonical decimal string:
  * one optional sign (ASCII "-" or "+", or a Unicode minus normalized to "-") and one
    optional "$", in either order ("-$5", "$-5", "$5"); other currency symbols are not accepted;
  * digits, with comma thousands separators accepted only in valid groups of three
    ("1,000", "12,345,678"); "1,00" or "12,34" are rejected, not repaired;
  * optional decimal part; a leading-dot form (".5") is accepted;
  * one trailing sentence period ("18." -> "18");
  * canonical output: no sign for zero ("-0" -> "0"), no leading zeros ("007" -> "7"),
    no trailing fractional zeros ("18.00" -> "18", "0.50" -> "0.5"), no separators.
Equality of two numbers is equality of canonical strings. The arithmetic is purely
decimal-string arithmetic: no floating point is involved, so "18.0", "18.00" and "18" are
equal, "1,000" equals "1000", and "18.0000001" differs from "18". Percent signs, units,
fractions ("3/4"), number words and scientific notation are NOT normalized: a string that
contains them is rejected by `normalize_number`.

Text normalization (`normalize_text`, applied before any extraction)
---------------------------------------------------------------------
CRLF/CR to LF; Unicode minus/dash look-alikes U+2010-U+2013, U+2212, U+FE63, U+FF0D to "-"
(U+2014 em dash is left alone); Unicode space separators to " "; zero-width characters
removed; non-ASCII decimal digits (for example full-width) to ASCII. Superscripts and
vulgar fractions are untouched.

Strict extraction (`strict_extract`, deterministic)
---------------------------------------------------
1. Normalize the text and strip trailing whitespace. When the spec sets `strip_markup`,
   markdown emphasis is then removed (`strip_markup` below), and when it sets `strip_latex`,
   LaTeX math markup is (`strip_latex` below); otherwise the text is used as is.
2. Find every non-overlapping match of the spec template (regex, MULTILINE) in which the
   literal placeholder "<NUM>" has been replaced by the number sub-pattern. A match that
   spans a newline is ignored.
3. A match qualifies only if: it lies on the last non-empty line (when
   `must_be_last_nonempty_line`); there is no text before it on its line (unless
   `leading_text_allowed`); there is no text after it on its line (unless
   `trailing_text_allowed`), where one final "." is tolerated if `allow_final_period`;
   surrounding whitespace never counts as text.
4. The captured number must pass `normalize_number`.
5. Exactly one qualifying match gives its canonical value. With several, `on_multiple`
   decides: "reject" returns None, "last" takes the last, "unanimous" returns the common
   value if all agree and None otherwise. With none, the result is None.
The number sub-pattern accepts an optional "-" (and "$" if `allow_currency`), digits with
valid comma groups (if `allow_thousands_separator`), an optional decimal part, and refuses to
start inside or end inside a longer digit run ("1,00" and "12.3.4" do not match).

Markdown emphasis (`strip_markup`, only for specs that set it)
--------------------------------------------------------------
Added because a small model often bolded its final line
("**Answer: 260**"). Per line: a bold span "**...**" or "__...__" keeps its content when the
markers are not glued to a digit or to another marker on the outside, so "2**3 = 8" (an
exponent) is unchanged; then one or more enclosing single "*", "_" or backtick pairs around a
whole line are removed ("*Answer: 18*", "`Answer: 18`"). Headings ("#"), LaTeX and every other
character are untouched. Whether markup may surround the requested format is part of the
investigator's format choice; specs without `strip_markup` behave exactly as in version 1.0.0.

LaTeX math markup (`strip_latex`, only for specs that set it)
-------------------------------------------------------------
Added in 1.3.0 after an audit of the boxed format's rejected outputs in the format screen. A
reply that ends in a display-math block puts its closing delimiter on a line of its own
("$$\n\boxed{18}\n$$", "\[\n\boxed{18}\n\]"), so the requested form sat on the last line of
content and was rejected for the delimiter after it. Trailing lines that hold nothing but "$$"
or "\]" are removed; the LaTeX-escaped dollar "\$" is read as "$"; and inline-math delimiters
that enclose the whole content of a box are removed ("\boxed{$27$}" reads as "\boxed{27}").
Units ("\boxed{80 \text{ people}}"), percent signs, copied placeholder brackets
("\boxed{<17>}") and text after the boxed line stay non-compliant.

Lenient extraction (`lenient_extract`, independent of any spec)
----------------------------------------------------------------
Normalize the text and return the canonical value of the LAST number token in it. A token is
an optional sign/"$" prefix, digits with valid comma groups, and an optional decimal part (or
a leading-dot decimal). Rules that matter in practice: a token may not start directly after a
word character, so "x2" and "H2O" contribute nothing; "5-3" yields 5 and 3 (a minus is a sign
only when it is not glued to a preceding word character); a comma that does not form a valid
thousands group ends the token ("1,2345" yields 1 and 2345); units, percent signs, currency
signs and a trailing period are dropped; "\boxed{18}", "#### 18" and "= 18." all yield 18;
LaTeX thousands "1{,}000" and "1\,000" are read as 1000. Fractions, scientific notation and
number words are not interpreted ("3/4" yields 4, "1e3" yields 1). The result
is None when the text contains no number token.

Format compliance (`compliance`)
---------------------------------
True exactly when `strict_extract` returns a value, that is, when the output contains the
requested final-answer format in the required position with a well-formed number. Compliance
does not depend on correctness. `strict_correct` is compliance and equality with gold, so a
non-extractable output counts as incorrect.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import pandas as pd

SCORER_VERSION = "gsm8k-scorer/1.3.0"   # 1.1.0: optional strip_markup; 1.2.0: candidates read "$" and a final ".";
#                                         1.3.0: optional strip_latex, set for boxed

NUM_PLACEHOLDER = "<NUM>"
ON_MULTIPLE_CHOICES = ("reject", "last", "unanimous")

# ---------------------------------------------------------------------------------------
# Text and number normalization
# ---------------------------------------------------------------------------------------
_DASH_TO_MINUS = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d"), "-")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_TRANSLATE = {**_DASH_TO_MINUS, **_ZERO_WIDTH}


def normalize_text(text) -> str:
    """Normalize model output or a numeric string. None becomes the empty string."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n").translate(_TRANSLATE)
    if s.isascii():
        return s
    out = []
    for ch in s:
        if ch.isascii():
            out.append(ch)
        elif ch.isdecimal():
            out.append(str(unicodedata.decimal(ch)))
        else:
            cat = unicodedata.category(ch)
            if cat == "Zs":
                out.append(" ")
            elif cat in ("Zl", "Zp"):
                out.append("\n")
            else:
                out.append(ch)
    return "".join(out)


_INT = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
_PREFIX_RE = re.compile(r"(?P<s1>[-+])?(?:\$[ \t]?)?(?P<s2>[-+])?")
_BODY_RE = re.compile(rf"(?:(?P<int>{_INT})(?:\.(?P<frac>[0-9]+))?|\.(?P<frac2>[0-9]+))\.?")


def normalize_number(s) -> str | None:
    """Canonical decimal string for a numeric string, or None if it is not a plain number.

    Examples: "1,000" -> "1000"; "$18.00." -> "18"; "-0.50" -> "-0.5"; ".5" -> "0.5";
    "-0" -> "0"; "1,00" -> None; "18%" -> None; "" -> None.
    """
    if s is None:
        return None
    t = normalize_text(s).strip()
    if not t:
        return None
    pm = _PREFIX_RE.match(t)
    s1, s2 = pm.group("s1"), pm.group("s2")
    if s1 and s2:
        return None
    sign = s1 or s2
    bm = _BODY_RE.fullmatch(t[pm.end():])
    if bm is None:
        return None
    int_part = (bm.group("int") or "0").replace(",", "").lstrip("0") or "0"
    frac = (bm.group("frac") or bm.group("frac2") or "").rstrip("0")
    out = int_part + ("." + frac if frac else "")
    if sign == "-" and out != "0":
        out = "-" + out
    return out


# ---------------------------------------------------------------------------------------
# Format spec
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FormatSpec:
    """Description of the requested final-answer format.

    template: regex (MULTILINE, no DOTALL) containing the literal placeholder "<NUM>" exactly
        once; "<NUM>" is replaced by the number sub-pattern, a named group `num`. Use "[ \\t]",
        not "\\s", so that a match cannot span lines. Regex metacharacters in the
        literal part must be escaped.
    case_sensitive: match the template case-sensitively.
    must_be_last_nonempty_line: the match must lie on the last non-empty line of the output.
    leading_text_allowed: text may precede the match on its line.
    trailing_text_allowed: text may follow the match on its line.
    allow_final_period: when trailing text is not allowed, one final "." is tolerated.
    on_multiple: how several qualifying matches are resolved: "reject", "last" or "unanimous".
    allow_currency: the number may carry a "$" ("$18", "-$18", "$-18").
    allow_thousands_separator: the number may use valid comma groups ("1,000").
    strip_markup: remove markdown emphasis before matching (module doc); False keeps 1.0.0
        behaviour, and a False value is left out of the fingerprint so 1.0.0 hashes are stable.
    strip_latex: remove LaTeX math markup before matching (module doc); False keeps 1.2.0
        behaviour, and a False value is left out of the fingerprint as for strip_markup.
    description: free text, not part of the behavior fingerprint.
    """

    name: str
    template: str
    case_sensitive: bool = True
    must_be_last_nonempty_line: bool = True
    leading_text_allowed: bool = False
    trailing_text_allowed: bool = False
    allow_final_period: bool = False
    on_multiple: str = "reject"
    allow_currency: bool = False
    allow_thousands_separator: bool = True
    strip_markup: bool = False
    strip_latex: bool = False
    description: str = field(default="", compare=False)

    def __post_init__(self):
        if self.template.count(NUM_PLACEHOLDER) != 1:
            raise ValueError(f"template must contain {NUM_PLACEHOLDER!r} exactly once: {self.template!r}")
        if self.on_multiple not in ON_MULTIPLE_CHOICES:
            raise ValueError(f"on_multiple must be one of {ON_MULTIPLE_CHOICES}, got {self.on_multiple!r}")
        _compile(self)  # surfaces regex errors at registration time


def _number_subpattern(spec: FormatSpec) -> str:
    sign = r"(?:-\$?|\$-?)?" if spec.allow_currency else r"-?"
    digits = _INT if spec.allow_thousands_separator else r"[0-9]+"
    return rf"(?<![0-9.,])(?P<num>{sign}{digits}(?:\.[0-9]+)?)(?![0-9]|[.,][0-9])"


@functools.lru_cache(maxsize=None)
def _compile(spec: FormatSpec) -> re.Pattern:
    pattern = spec.template.replace(NUM_PLACEHOLDER, _number_subpattern(spec), 1)
    flags = re.MULTILINE | (0 if spec.case_sensitive else re.IGNORECASE)
    return re.compile(pattern, flags)


def spec_to_dict(spec: FormatSpec) -> dict:
    return asdict(spec)


def spec_from_dict(d: Mapping) -> FormatSpec:
    known = {f.name for f in fields(FormatSpec)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown FormatSpec fields: {sorted(unknown)}")
    return FormatSpec(**dict(d))


def spec_fingerprint(spec: FormatSpec) -> str:
    """SHA-256 over the behavior-defining fields (excludes `name` and `description`)."""
    d = spec_to_dict(spec)
    d.pop("name")
    d.pop("description")
    for flag in ("strip_markup", "strip_latex"):
        if not d[flag]:
            d.pop(flag)
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def scorer_record(spec: FormatSpec) -> dict:
    """Provenance record for manifests (pins the scorer version)."""
    return {"scorer_version": SCORER_VERSION, "format_spec": spec_to_dict(spec),
            "format_spec_sha256": spec_fingerprint(spec)}


# Candidates for the investigator's choice of requested final-answer format. The format chosen
# is what the prompt must request.
# Spec version 1.2.0: every candidate reads its number as normalize_number does, with one "$" and one final
# period accepted; units and percent signs stay non-compliant. In the format screen, 1,088 of 2,851 rejected
# "Answer:" outputs had nothing wrong but a "$" or a final period ("Answer: $11", "Answer: 95.").
# Spec version 1.3.0: boxed tolerates LaTeX math markup (strip_latex). In the format
# screen, 676 of 3,485 rejected boxed outputs ended in a display-math block whose closing delimiter followed the boxed
# line, among them 334 of Qwen3-8B's 400 chain-of-thought outputs and 258 of Mistral-Small-3.1's.
CANDIDATE_SPECS: dict[str, FormatSpec] = {
    "the_answer_is": FormatSpec(
        name="the_answer_is",
        template=r"the answer is[ \t]+<NUM>",
        case_sensitive=False, must_be_last_nonempty_line=True,
        leading_text_allowed=True, trailing_text_allowed=False, allow_final_period=True, allow_currency=True,
        description="CANDIDATE, not a decision. Sentence form on the last non-empty line, e.g. "
                    "'Therefore, the answer is 18.' Case-insensitive, preceding words allowed, "
                    "one final period allowed, nothing else after the number."),
    "hash_n": FormatSpec(
        name="hash_n",
        template=r"####[ \t]<NUM>",
        case_sensitive=True, must_be_last_nonempty_line=True,
        leading_text_allowed=False, trailing_text_allowed=False, allow_final_period=True, allow_currency=True,
        description="CANDIDATE, not a decision. GSM8K-native label: the last non-empty line is "
                    "exactly '#### 18'; '$18' and one final period are accepted."),
    "answer_colon": FormatSpec(
        name="answer_colon",
        template=r"Answer:[ \t]<NUM>",
        case_sensitive=True, must_be_last_nonempty_line=True,
        leading_text_allowed=False, trailing_text_allowed=False, allow_final_period=True, allow_currency=True,
        description="CANDIDATE, not a decision. The last non-empty line is exactly 'Answer: 18'; '$18' and "
                    "one final period are accepted."),
    "boxed": FormatSpec(
        name="boxed",
        template=r"\\boxed\{<NUM>\}",
        case_sensitive=True, must_be_last_nonempty_line=True,
        leading_text_allowed=True, trailing_text_allowed=True, allow_final_period=False, allow_currency=True,
        strip_latex=True,
        description="CANDIDATE, not a decision. LaTeX '\\boxed{18}' anywhere on the last "
                    "non-empty line, with surrounding text and '$' delimiters allowed; a closing "
                    "display-math line after it, '\\$' and '\\boxed{$18$}' are tolerated."),
    "answer_colon_markup": FormatSpec(
        name="answer_colon_markup",
        template=r"Answer:[ \t]<NUM>",
        case_sensitive=True, must_be_last_nonempty_line=True,
        leading_text_allowed=False, trailing_text_allowed=False, allow_final_period=True, allow_currency=True,
        strip_markup=True,
        description="CANDIDATE, not a decision. As answer_colon, with markdown emphasis around the "
                    "line or its parts tolerated ('**Answer: 18**', 'Answer: **18**')."),
    "the_answer_is_markup": FormatSpec(
        name="the_answer_is_markup",
        template=r"the answer is[ \t]+<NUM>",
        case_sensitive=False, must_be_last_nonempty_line=True,
        leading_text_allowed=True, trailing_text_allowed=False, allow_final_period=True, allow_currency=True,
        strip_markup=True,
        description="CANDIDATE, not a decision. As the_answer_is, with markdown emphasis tolerated "
                    "('Therefore, the answer is **18**.')."),
}


# ---------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------
_BOLD_SPAN = re.compile(r"(?<![0-9*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![0-9*])"
                        r"|(?<![0-9_])__(?=\S)(.+?)(?<=\S)__(?![0-9_])")
_WRAPPERS = ("*", "_", "`")


def strip_markup(text: str) -> str:
    """Remove markdown emphasis line by line (module doc); a normalized text is expected."""
    out = []
    for line in text.split("\n"):
        line = _BOLD_SPAN.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), line)
        s = line.strip()
        while len(s) >= 2 and s[0] == s[-1] and s[0] in _WRAPPERS:
            s = s[1:-1].strip()
        out.append(s if s != line.strip() else line)
    return "\n".join(out)


_DISPLAY_CLOSE = ("$$", "\\]")
_BOXED_INLINE_MATH = re.compile(r"\\boxed\{[ \t]*\$([^${}\n]*)\$[ \t]*\}")


def strip_latex(text: str) -> str:
    """Remove LaTeX math markup around the requested form (module doc); a normalized text is expected."""
    lines = text.split("\n")
    while lines and lines[-1].strip() in _DISPLAY_CLOSE + ("",):
        lines.pop()
    t = "\n".join(lines).replace("\\$", "$")
    return _BOXED_INLINE_MATH.sub(lambda m: "\\boxed{" + m.group(1) + "}", t)


def strict_extract(text, spec: FormatSpec) -> str | None:
    """Canonical number if the output complies exactly with `spec`, else None (see module doc)."""
    t = normalize_text(text).rstrip()
    if spec.strip_markup:
        t = strip_markup(t).rstrip()
    if spec.strip_latex:
        t = strip_latex(t).rstrip()
    if not t:
        return None
    rx = _compile(spec)
    last_line_start = t.rfind("\n") + 1
    values = []
    for m in rx.finditer(t):
        if "\n" in m.group(0):
            continue
        line_start = t.rfind("\n", 0, m.start()) + 1
        line_end = t.find("\n", m.end())
        line_end = len(t) if line_end == -1 else line_end
        if spec.must_be_last_nonempty_line and line_start != last_line_start:
            continue
        if not spec.leading_text_allowed and t[line_start:m.start()].strip():
            continue
        if not spec.trailing_text_allowed:
            after = t[m.end():line_end].strip()
            if after and not (spec.allow_final_period and after == "."):
                continue
        value = normalize_number(m.group("num"))
        if value is not None:
            values.append(value)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if spec.on_multiple == "last":
        return values[-1]
    if spec.on_multiple == "unanimous":
        return values[0] if len(set(values)) == 1 else None
    return None


_LENIENT_TOKEN = re.compile(
    r"(?<!\w)(?:-\$?|\$-?)?"
    r"(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
)
_LATEX_BRACE_COMMA = re.compile(r"(?<=[0-9])\{,\}(?=[0-9]{3}(?![0-9]))")
_LATEX_THIN_COMMA = re.compile(r"(?<=[0-9])\\[,!;:](?=[0-9]{3}(?![0-9]))")


def lenient_extract(text) -> str | None:
    """Canonical value of the last number token in the text, or None (see module doc)."""
    t = normalize_text(text)
    if not t:
        return None
    t = _LATEX_BRACE_COMMA.sub(",", t)
    t = _LATEX_THIN_COMMA.sub(",", t)
    last = None
    for m in _LENIENT_TOKEN.finditer(t):
        last = m.group(0)
    return normalize_number(last) if last is not None else None


def compliance(text, spec: FormatSpec) -> bool:
    """True exactly when the output satisfies the requested format (strict extraction succeeds)."""
    return strict_extract(text, spec) is not None


# ---------------------------------------------------------------------------------------
# Scoring a table of outputs
# ---------------------------------------------------------------------------------------
REQUIRED_COLUMNS = ("request_id", "item_id", "output_text")
SCORE_COLUMNS = ("strict_answer", "strict_correct", "lenient_answer", "lenient_correct", "compliant")


def _gold_lookup(gold) -> dict:
    if isinstance(gold, pd.DataFrame):
        if not {"item_id", "gold_answer"} <= set(gold.columns):
            raise ValueError("gold DataFrame needs columns item_id and gold_answer")
        raw = dict(zip(gold["item_id"].tolist(), gold["gold_answer"].tolist()))
    elif isinstance(gold, pd.Series):
        raw = gold.to_dict()
    elif isinstance(gold, Mapping):
        raw = dict(gold)
    else:
        raise TypeError("gold must be a DataFrame, Series or Mapping of item_id to gold answer")
    out = {}
    for k, v in raw.items():
        canon = normalize_number(v)
        if canon is None:
            raise ValueError(f"gold answer for item {k!r} is not a number: {v!r}")
        out[k] = canon
    return out


def _map_unique(func, values: list) -> list:
    cache: dict = {}
    out = []
    for v in values:
        if v not in cache:
            cache[v] = func(v)
        out.append(cache[v])
    return out


def score_outputs(df: pd.DataFrame, gold, spec: FormatSpec) -> pd.DataFrame:
    """Add strict_answer, strict_correct, lenient_answer, lenient_correct and compliant.

    df needs request_id (unique), item_id and output_text. Missing or non-string output text
    is treated as the empty string (not extractable, incorrect). `gold` maps item_id to the
    gold answer (a gsm8k DataFrame with item_id and gold_answer, a Series or a dict); gold
    values are passed through `normalize_number`. Correctness is equality of canonical
    strings. strict_correct is 0 for any non-extractable output. Returns a copy of df.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["request_id"].duplicated().any():
        dup = df.loc[df["request_id"].duplicated(), "request_id"].head(3).tolist()
        raise ValueError(f"request_id must be unique; duplicates such as {dup}")
    if df["item_id"].isna().any():
        raise ValueError("item_id contains missing values")
    lookup = _gold_lookup(gold)
    item_ids = df["item_id"].tolist()
    absent = sorted({i for i in item_ids if i not in lookup})
    if absent:
        raise KeyError(f"{len(absent)} item_id values have no gold answer, e.g. {absent[:5]}")
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    strict = _map_unique(lambda t: strict_extract(t, spec), texts)
    lenient = _map_unique(lenient_extract, texts)
    gold_list = [lookup[i] for i in item_ids]
    out = df.copy()
    out["strict_answer"] = pd.Series(strict, index=df.index, dtype=object)
    out["strict_correct"] = np.fromiter((int(s is not None and s == g) for s, g in zip(strict, gold_list)),
                                        dtype=np.int64, count=len(strict))
    out["lenient_answer"] = pd.Series(lenient, index=df.index, dtype=object)
    out["lenient_correct"] = np.fromiter((int(s is not None and s == g) for s, g in zip(lenient, gold_list)),
                                         dtype=np.int64, count=len(lenient))
    out["compliant"] = np.fromiter((s is not None for s in strict), dtype=bool, count=len(strict))
    return out
