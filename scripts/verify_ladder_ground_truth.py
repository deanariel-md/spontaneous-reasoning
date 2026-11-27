r"""Ground truth for the size ladder's checker (scripts/verify_ladder_analysis.py).

The checker recomputes from the scored rows, so on its own it cannot see an error in the scoring itself. This module goes back to the raw replies and reads them here, with its own code: it
imports nothing from src/ and nothing from the analysis. What it takes from the source files is the stated rules, read
as constants with ast (never imported): the completion rule's lists (src/forced_stop.py: VERSION, WRAPPER_COMMANDS,
CONTINUATION_WORDS, CONTINUATION_SIGNS) and the corrected gain's rule (src/baseline_mixture.py: RULE_TEXT).

  corrected_gain_rule  the parts of RULE_TEXT the checker's corrected gain follows (the comparison of a requested
                       output's length with the forced cap, the draws w and the gain are averaged over, the formula);
                       a part the text does not state is reported, so a change of the rule cannot pass unseen.
  StopRule             the forced arms' stop at the first "}": whether a reply ended on the stop with its box open,
                       and whether the stated rule completes it, written here from the rule's statement.
  raw_forced, raw_rows the raw replies: the GPU-run models' shards (data/d2_out/ladder2/<run>/<task>, the run named in
                       data/d2_out/ladder2/chosen/<model>), the panel's forced-arm shards (data/d2_out/forced/<task>/
                       <model>), Mistral Large 3's and DeepSeek's API records (cut after the first "}" for their GSM8K
                       and BIG-Bench Hard forced arms, as their collections' stop returns).
  stop_unread          per model, benchmark and forced arm: rows, replies the stop left open, completed, unfinished,
                       other, unread; per configuration for the zero-shot arm (the family steps' bound).
  Reader               a content reading of a reply written here: the last complete box that is not a placeholder
                       (the placeholder rule, written here from its statement; its word lists SLOT_WORDS,
                       SLOT_NOUNS, SLOT_QUALIFIERS and SLOT_TAILS read from src/content_scoring.py with ast: an empty
                       content, a slot name bare or in angle brackets; an option named with its
                       letter, "<Option A>", is no placeholder and reads as the letter), its text wrappers removed and
                       its LaTeX spacing commands read as blanks by a scan written here (a backslash before
                       a blank, "\,", "\;", "\:", "\!", "\quad", "\qquad", "~"; a thin space between digit groups is a
                       thousands comma); a box written with angle brackets in place of its braces ("\boxed<14>") or as
                       a tag ("<boxed>B</boxed>") counts as a box holding its content, found by a scan written here;
                       a leading answer cue ("Answer:", "The final answer is", "(Answer: X)") is taken off
                       the content by a scan written here;
                       GSM8K a number or a clock time H:MM read as hours on the 12-hour clock (LaTeX's
                       braced colon "2{:}00" a colon), else the last number in the box, a clock time counting
                       as one number; MATH-500 one rational number, the
                       content or its part after the last "=" (the numeric rule of src/math_task.py, written here from
                       its statement), no box: the whole reply as one number, else its last number expression;
                       BIG-Bench Hard the item's answer type (a letter, an option named with its letter, the option's
                       text, Yes/No and the task's own wording, an integer), else on a letter task the option letter
                       a box opens with when more text follows and that text names no other option, else the last answer token; no box: the reply, its placeholder boxes and brace-less placeholder
                       copies removed, read for its last answer token; an open box: not correct. HotpotQA: the content
                       reading of src/hotpot_task.py (hotpot-content), written here from its statement (the last box
                       that is not a placeholder, else an unclosed box, else the line after the last answer cue, else
                       the last line; LaTeX spacing read as blanks before cleaning), scored by exact match
                       after HotpotQA's normalisation, written here. Rules of content/1.6.0 and hotpot-content/1.5.0,
                       each written here: an answer tag pair "<answer>X</answer>" is a box (Reader.tags, a scan); a
                       BIG-Bench Hard box whose option letter and option text name different options is not correct,
                       and a capital that is not one of the item's letters reads the option its text names; a
                       temporal_sequences range cut at "to" reads the option it begins (Reader.range_option); the
                       web_of_lies wording does not read a last sentence holding a condition or hedge (Reader.hedge);
                       "\ " and "~" between digit groups are thousands commas; HotpotQA reads typographic quotes,
                       apostrophes and dashes as ASCII in the answer and the gold. Rules of content/1.7.0, each
                       written here: on sports_understanding the verdict as one unqualified sentence ("the sentence
                       is not plausible", the judged sentence in double quotes allowed) is read (Reader.sports_sentence,
                       a scan); a lowercase option label "(x)" or "x)" is read as its capital, whole or opening a box,
                       under the rules for capitals (Reader.bbh_exact, Reader.lead_letter).
  panel_sample_check   a seeded sample of the main panel's replies (PANEL_SAMPLE_N per model, benchmark and
                       arm, and per five-shot forced arm; seed PANEL_SEED), drawn from the scored content rows, each
                       read here from its generation folder; every disagreement is a mismatch.
  sample_check         a seeded random sample of replies per ladder model, benchmark (GSM8K, MATH-500, BIG-Bench Hard,
                       HotpotQA), reading and arm, read here and compared with the scored rows' content_correct of that
                       reading: uncut, the stored text as generated; cut, the copy cut at the panel's cap (the GPU-run
                       models' stored cut copy, output_text_cut; the API replies cut here, by their tokenizers, at the
                       panel's cap that their scored rows record), with the replies cut at the cap drawn apart as a
                       second stratum. Every disagreement is a mismatch (SAMPLE_MAX_DISAGREE); each cell reports a
                       one-sided 95% upper bound on its share of disagreements (exact binomial).
  full_reply_check     Mistral Large 3 and DeepSeek: every GSM8K and BIG-Bench Hard forced reply that the cut leaves
                       with its box open, the scored rows' content_correct against the box the model closed in its
                       whole reply, read here: agree, false gains, false losses; apart for the replies
                       the readers read (completed, inner box, ranges read; gated at FULL_MAX_DISAGREE, all together
                       and each class on its own, so a small class read wrongly throughout cannot pass inside the
                       large completed class) and those left unread (no false gain allowed; their false losses must
                       equal the stop's loss ladder.json states).
"""
from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PRE = "\\boxed{"
STOP_TASKS = ("gsm8k", "bbh")
FORCED_ARMS = ("forced_direct", "fewshot_forced")
SAMPLE_ARMS = ("direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced", "fewshot_requested")
FORCED_STOP_SRC = "src/forced_stop.py"
CONTENT_SRC = "src/content_scoring.py"
BASELINE_RULE_SRC = "src/baseline_mixture.py"
CLUSTER = ("ministral-3-8b-instruct-2512", "ministral-3-14b-instruct-2512", "llama-3.1-70b-instruct-fp8",
           "qwen3.5-122b-a10b-fp8")
MISTRAL, DEEPSEEK = "mistral-large-2512", "deepseek-v4.1-flash-1017"
API_RAW = {MISTRAL: "data/ladder_api/mistral-large-2512/raw.jsonl",
           DEEPSEEK: "data/frontier_panel/nebius_panel_raw.jsonl"}
SCORED_UNCUT = {MISTRAL: "data/ladder_api/mistral-large-2512/scored/{task}_uncut.parquet",
                DEEPSEEK: "data/frontier_panel/scored/{task}_scored_uncut.parquet",
                "cluster": "data/processed/ladder_uncut/{task}/{key}.parquet"}
SCORED_CUT = {MISTRAL: "data/ladder_api/mistral-large-2512/scored/{task}_cut.parquet",
              DEEPSEEK: "data/frontier_panel/scored/{task}_scored.parquet",
              "cluster": "data/processed/ladder/{task}/{key}.parquet"}
# the API replies' cut at the panel's cap, written here from the collections' statements: DeepSeek, the reply re-encoded
# by its tokenizer and kept up to the end of its first N tokens (scripts/nebius_panel_report.TokenCutter); Mistral
# Large 3, the reply's first N tokens of Large 3's tokenizer, decoded (scripts/mistral_ladder.Counter.cut)
API_TOKENIZER = {DEEPSEEK: ("offsets", "data/nebius_prefill/tokenizers/DeepSeek-V4.1-Flash-1017/tokenizer.json"),
                 MISTRAL: ("decode", ("mistralai/Mistral-Large-3-675B-Instruct-2512",
                                      "383ffea2c7d60dfd44ca960e8e691709d4fdb9cd"))}
SAMPLE_TASKS = ("gsm8k", "math", "bbh", "hotpot")
SAMPLE_READINGS = ("uncut", "cut")
SAMPLE_N = 150             # replies per model, benchmark, reading and arm
SAMPLE_CUT_N = 50          # and, in the cut reading, replies cut at the panel's cap, drawn apart
SAMPLE_SEED = 20261124
SAMPLE_MAX_DISAGREE = 0    # a cell with more disagreements than this is a mismatch (every one)
BOUND_LEVEL = 0.95         # the one-sided upper bound reported for each cell's share of disagreements
PANEL_SAMPLE_N = 100       # replies per main-panel model, benchmark and arm (and five-shot forced arm)
PANEL_SEED = 20261125
PANEL_TABLES = {"gsm8k": "data/processed/content/gsm8k_content.parquet",
                "bbh": "data/processed/content/bbh_content.parquet",
                "math": "data/processed/content/math_content.parquet",
                "fewshot_gsm8k": "data/processed/content/fewshot_gsm8k_content.parquet",
                "fewshot_bbh": "data/processed/content/fewshot_bbh_content.parquet",
                "fewshot_math": "data/processed/content/fewshot_math_content.parquet",
                "hotpot": "data/processed/hotpot/scored_content.parquet"}
FULL_MAX_DISAGREE = Fraction(1, 100)   # an API model whose scored forced replies disagree with more than this share of
                                       # the boxes closed in the whole replies is a mismatch
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
# a GSM8K last-number token as src/scoring.lenient_extract states it, written here: not after a digit or
# "_" (letters are allowed before it: this reader has removed the braces of "\overline{6}" by then); an optional sign
# "-", "-$", "$" or "$-" right before it; digits with comma groups, or with a decimal part, or a decimal part alone
GSM_TOKEN = re.compile(r"(?<![0-9_])(-\$?|\$-?)?([0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
# a clock time, written here: one or two digits, ":", minutes 00-59, not inside a longer run of digits
# and colons, then an optional am or pm (dots, blanks and LaTeX spacing allowed)
CLOCK_TIME = re.compile(r"(?<![0-9:.,])([0-9]{1,2}):([0-5][0-9])(?![0-9:])(?:(?:[ \t\n~]|\\[ ,;:!])*[AaPp]\.?[ \t]*"
                        r"[Mm](?![A-Za-z0-9_])\.?)?")
MATH_NUMBER = re.compile(r"(?<![\w\\])-?\s*\\[dt]?frac\s*(?:\{\s*-?[0-9]+\s*\}|[0-9])\s*(?:\{\s*-?[0-9]+\s*\}|[0-9])"
                         r"|(?<![\w./])-?[0-9]+\s*/\s*[0-9]+(?![0-9]|\.[0-9])"
                         r"|(?<!\w)-?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
PLACEHOLDER = "<answer>"


def upper_bound(k: int, n: int, level: float = BOUND_LEVEL) -> float | None:
    """The one-sided exact (Clopper-Pearson) upper bound on a share from k events in n trials: the p at which the
    chance of k or fewer is 1 - level."""
    import math
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    alpha = 1 - level

    def cdf(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------------------------- stated rules (ast)

def available(root: Path) -> bool:
    """Whether the raw replies are under root (the GPU runs' folders and both API records)."""
    r = Path(root)
    return (r / "data/d2_out/ladder2/chosen").is_dir() and all((r / p).exists() for p in API_RAW.values())


def source_constant(root: Path, rel: str, name: str):
    """The literal value of a module-level assignment `name = ...` in a source file, read with ast (not imported)."""
    tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{rel}: no literal assignment to {name}")


def corrected_gain_rule(root: Path) -> tuple[dict, list[str]]:
    """The corrected gain as src/baseline_mixture.RULE_TEXT states it: (parts, problems). Parts: 'compare' ('>' or
    '>='), 'w_draws' and 'gain_draws' ('discovery' or 'all'), 'form' ('(1 - w) x gain'). A part the text does not
    state in these words is a problem (reported as a mismatch by the checker)."""
    t = " ".join(source_constant(root, BASELINE_RULE_SRC, "RULE_TEXT").split())
    parts, problems = {}, []
    m = re.search(r"w is the share of its requested-direct outputs longer than the forced arm's output cap .*?"
                  r"compared with the cap by '(>=|>)'\), computed per configuration and averaged over the "
                  r"(discovery|validation|48) draws", t)
    if m:
        parts["compare"], parts["w_draws"] = m.group(1), m.group(2)
    else:
        problems.append("RULE_TEXT: the definition of w (cap comparison, draws) is not stated as the checker reads it")
    m = re.search(r"The gain over the forced arm is the chain-of-thought minus forced-direct accuracy \(strict scoring\), "
                  r"the model mean over the (discovery|validation|48) draws", t)
    if m:
        parts["gain_draws"] = m.group(1)
    else:
        problems.append("RULE_TEXT: the gain over the forced arm and its draws are not stated as the checker reads them")
    if re.search(r"The predicted gain over the requested arm is \(1 - w\) x the gain over the forced arm\.", t):
        parts["form"] = "(1 - w) x gain"
    else:
        problems.append("RULE_TEXT: the corrected gain is not stated as (1 - w) x the gain over the forced arm")
    return parts, problems


# --------------------------------------------------------------------------------------------- the stop's rule

class StopRule:
    """The completion of a forced reply that the stop at the first "}" ended with its box open, as the rule states it
    (src/forced_stop.py, its docstring and RULE), written here: complete when the group the stop closed is a text or
    font wrapper or a bare group that opens the box, every group still open is such a wrapper, "(" or "[", and the
    box's text so far neither ends in a word or sign that asks for more nor holds two options; a tab followed by the
    rest of a wrapper whose name starts with "t" is that wrapper."""

    def __init__(self, root: Path):
        c = lambda n: source_constant(root, FORCED_STOP_SRC, n)  # noqa: E731
        self.version = c("VERSION")
        self.wrappers = set(c("WRAPPER_COMMANDS"))
        self.words = {w.casefold() for w in c("CONTINUATION_WORDS")}
        self.signs = set(c("CONTINUATION_SIGNS"))
        tails = sorted((w[1:] for w in self.wrappers if w.startswith("t")), key=len, reverse=True)
        self.tab = re.compile("\t(" + "|".join(map(re.escape, tails)) + r")(?=\s*\{)")
        # forced-stop/1.2.0: a box whose text is a clock time and "to" is the class open_range
        self.ranges = self.version not in ("forced-stop/1.0.0", "forced-stop/1.1.0")

    @staticmethod
    def parse(text: str):
        """(closed_box, last_closed_name, open_entries) over the text after the prefill. A name is the command
        right before a "{" ('text', 'frac'), 'LEAD' for a bare "{" with only blanks and opening brackets before it
        in the box, '' for another bare "{". Entries are (char, name)."""
        body = text[len(PRE):]
        stack, last = [("{", "boxed")], None
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "{":
                j = i - 1
                while j >= 0 and body[j] in " \t":
                    j -= 1
                k = j
                while k >= 0 and body[k].isalpha():
                    k -= 1
                if k < j and k >= 0 and body[k] == "\\" and (k == 0 or body[k - 1] != "\\"):
                    name = body[k + 1: j + 1]
                elif all(e[0] != "{" for e in stack[1:]) and all(x in " \t\n([" for x in body[:i]):
                    name = "LEAD"
                else:
                    name = ""
                stack.append(("{", name))
            elif ch == "}":
                while stack[-1][0] != "{":
                    stack.pop()
                last = stack.pop()[1]
                if not stack:
                    return True, last, []
            elif ch in "([":
                stack.append((ch, ""))
            elif ch in ")]":
                want = "(" if ch == ")" else "["
                if stack[-1][0] == want:
                    stack.pop()
            i += 1
        return False, last, stack

    def words_of(self, text: str) -> str:
        b = text[len(PRE):]
        b = re.sub(r"\\[ ,;:!]", " ", b)
        b = re.sub(r"\\[A-Za-z]+\*?", " ", b)
        b = re.sub(r"\\(.)", r"\1", b)
        return b.replace("{", "").replace("}", "").rstrip()

    def asks_for_more(self, text: str) -> bool:
        w = self.words_of(text)
        if w and w[-1] in self.signs:
            return True
        m = re.search(r"([A-Za-z]+)$", w)
        if m and (m.start() == 0 or not w[m.start() - 1].isalpha()) and m.group(1).casefold() in self.words:
            return True
        return self.two_options(w)

    @staticmethod
    def two_options(w: str) -> bool:
        """Two option letters in a text read as words: "A, D", "(A) or (B"."""
        return bool(re.search(r"(?<![A-Za-z])[A-R]\s*\)?\s*(?:,|/|&|\band\b|\bor\b)\s*\(?\s*[A-R](?![A-Za-z])", w)
                    or re.search(r"\(\s*[A-R]\s*\).*\(\s*[A-R]\s*\)", w, re.S))

    def state(self, text: str, stops: bool) -> tuple[str, str]:
        """(state, text read): 'no_stop' (the arm has no stop, or the reply does not end on "}"), 'closed',
        'completed', 'inner' (the stop closed a nested box), 'range' (a clock time and "to"),
        'unfinished', 'other'."""
        if not stops or not text.endswith("}"):
            return "no_stop", text
        closed, last, open_ = self.parse(text)
        if closed:
            return "closed", text
        wrap = lambda n: n in self.wrappers or n == "LEAD"  # noqa: E731
        if not wrap(last) and "\t" in text:
            fixed = self.tab.sub(lambda m: "\\t" + m.group(1), text)
            if fixed != text:
                st, read = self.state(fixed, stops)
                if st == "completed":
                    return st, read
        if not wrap(last) or any(o == "{" and not wrap(n) for o, n in open_[1:]):
            return ("inner" if last == "boxed" else "other"), text
        if self.asks_for_more(text):
            if self.ranges and re.fullmatch(r"\(?[ \t\n]*[0-9]{1,2}(?::[0-5][0-9])?[ \t\n]*(?:[AaPp]\.?[ \t\n]*[Mm]\.?)?"
                                            r"[ \t\n]+to", self.words_of(text)):
                return "range", text
            return "unfinished", text
        closers = {"{": "}", "(": ")", "[": "]"}
        return "completed", text + "".join(closers[o] for o, _ in reversed(open_))


# --------------------------------------------------------------------------------------------- raw replies

def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def decoding_of(folder: Path) -> dict:
    decs = {json.dumps(json.loads(p.read_text(encoding="utf-8")).get("decoding") or {}, sort_keys=True)
            for p in sorted(folder.glob("run_log-*.json"))}
    if len(decs) != 1:
        raise ValueError(f"{folder}: {len(decs)} decodings in its run logs")
    return json.loads(next(iter(decs)))


def stops_at_brace(dec: dict, arm: str) -> bool:
    ov = ((dec.get("arm_overrides") or {}).get(arm)) or {}
    stop = ov.get("stop", dec.get("stop")) or []
    keep = ov.get("include_stop_str_in_output", dec.get("include_stop_str_in_output", False))
    return "}" in stop and bool(keep)


def shard_rows(folder: Path, arms, cols=("request_id", "arm", "config_id", "item_id", "output_text",
                                        "finish_reason")) -> pd.DataFrame:
    parts = []
    for sh in sorted(folder.glob("shard-*.parquet")):
        t = pq.read_table(sh, columns=list(cols)).to_pandas()
        parts.append(t[t["arm"].isin(arms)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cols))


def api_rows(root: Path, key: str, task: str, arms) -> pd.DataFrame:
    """An API model's answered records of one benchmark (the first record of a key), with the reply as the scorer
    reads it: forced GSM8K and BIG-Bench Hard replies cut after their first "}"; `full` keeps the whole reply."""
    seen, rows = set(), []
    for r in read_jsonl(root / API_RAW[key]):
        if r.get("status") != 200 or not isinstance(r.get("text"), str) or r.get("task") != task:
            continue
        if r.get("arm") not in arms or r["key"] in seen:
            continue
        seen.add(r["key"])
        t, fin = r["text"], r.get("finish_reason")
        cut = t
        if r["arm"] in FORCED_ARMS and task in STOP_TASKS and "}" in t:
            cut, fin = t[: t.index("}") + 1], "stop"
        rows.append({"request_id": r["request_id"], "arm": r["arm"], "config_id": str(r["config_id"]),
                     "item_id": str(r["item_id"]), "output_text": cut, "finish_reason": fin, "full": t})
    return pd.DataFrame(rows)


def raw_forced(root: Path, key: str, task: str, source: str) -> list[tuple]:
    """(arm, config_id, item_id, text with the prefill, stops) for the forced replies of one model and benchmark.
    source: 'ladder' (GPU run or API record) or 'panel' (forced arm)."""
    if key in API_RAW:
        df = api_rows(root, key, task, FORCED_ARMS)
        return [(a, c, i, PRE + t, task in STOP_TASKS) for a, c, i, t in
                zip(df["arm"], df["config_id"], df["item_id"], df["output_text"])]
    if source == "panel":
        folder = root / "data/d2_out/forced" / task / key
    else:
        rk = (root / "data/d2_out/ladder2/chosen" / key).read_text(encoding="utf-8").strip()
        folder = root / "data/d2_out/ladder2" / rk / task
    dec = decoding_of(folder)
    pre = dec.get("prefill") or {}
    df = shard_rows(folder, [a for a in FORCED_ARMS if a in pre])
    out = []
    for a, c, i, t in zip(df["arm"], df["config_id"], df["item_id"], df["output_text"]):
        if pre.get(a) != PRE:
            raise ValueError(f"{folder}: prefill {pre.get(a)!r}")
        out.append((a, str(c), str(i), PRE + (t if isinstance(t, str) else ""), stops_at_brace(dec, a)))
    return out


def stop_unread(root: Path, rule: StopRule, keys_sources: list[tuple[str, str]], reader=None) -> dict:
    """{key: {task: {arm: Counter(rows, open, completed, inner, range, range_read, range_unread, unfinished, other,
    unread), 'by_config': {config: (rows, unread)} for the zero-shot arm}}}. A range is unread unless the
    reader reads it (Reader.range_option, BIG-Bench Hard temporal_sequences)."""
    out = {}
    for key, src in keys_sources:
        for task in STOP_TASKS:
            cells = defaultdict(Counter)
            byc = defaultdict(lambda: [0, 0])
            for arm, cfg, item, text, stops in raw_forced(root, key, task, src):
                st, _ = rule.state(text, stops)
                c = cells[arm]
                c["rows"] += 1
                if arm == "forced_direct":
                    byc[cfg][0] += 1
                if st in ("no_stop", "closed"):
                    continue
                c["open"] += 1
                c[st] += 1
                if st == "inner" and "</think>" in text[len(PRE):text.find(PRE, len(PRE))]:
                    c["restart"] += 1          # a new box begun after a closed reasoning block
                if st == "range":
                    read = (reader is not None and task == "bbh"
                            and reader.range_option(str(item), text[text.rfind(PRE) + len(PRE):]) is not None)
                    c["range_read" if read else "range_unread"] += 1
                if st in ("unfinished", "other") or (st == "range" and not read):
                    c["unread"] += 1
                    if arm == "forced_direct":
                        byc[cfg][1] += 1
            out.setdefault(key, {})[task] = {"arms": dict(cells), "by_config": dict(byc)}
    return out


def discovery_pp(by_config: dict, disc: list[str]):
    """The unread share of the zero-shot forced arm, per configuration, averaged over the draws (pp, exact)."""
    if any(str(c) not in by_config for c in disc):
        return None
    return 100 * sum(Fraction(by_config[str(c)][1], by_config[str(c)][0]) for c in disc) / len(disc)


# --------------------------------------------------------------------------------------------- a reader of its own

class Reader:
    """Content reading of one reply, written here (module doc)."""

    # the task's own wording of the two answers, the whole content (sports_understanding) or its end (web_of_lies):
    # (pattern, 0 for the first answer word, 1 for the second), the first that matches is read (written
    # from src/content_scoring.py's statement; until then the checker searched the words anywhere in the content)
    YESNO_WORDS = {"sports_understanding": ((r"^plausible$", 0), (r"^(?:not plausible|implausible)$", 1)),
                   "web_of_lies": ((r"(?:\blies|\blying|\bnot tell the truth|n't tell the truth)$", 1),
                                   (r"\btells? the truth$", 0))}

    @staticmethod
    def norm_text(s: str) -> str:
        """The text as the scorers read it (src/scoring.normalize_text, written here): line ends as "\\n", dashes and
        the minus sign as "-", zero-width characters removed, other digits as ASCII digits, other spaces as " ", line
        and paragraph separators as "\\n"."""
        import unicodedata
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.translate({**dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d"), "-"),
                         **dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)})
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
                out.append(" " if cat == "Zs" else "\n" if cat in ("Zl", "Zp") else ch)
        return "".join(out)

    def __init__(self, root: Path, wrappers: set):
        self.wrap = re.compile(r"\\(?:" + "|".join(sorted(map(re.escape, wrappers), key=len, reverse=True))
                               + r")\s*\{([^{}]*)\}")
        c = lambda n: [" ".join(str(x).casefold().split()) for x in source_constant(root, CONTENT_SRC, n)]  # noqa: E731
        self.slot_words, self.slot_nouns = set(c("SLOT_WORDS")), set(c("SLOT_NOUNS"))
        self.qualifiers, self.tails = set(c("SLOT_QUALIFIERS")), set(c("SLOT_TAILS"))
        g = pd.read_parquet(root / "data/raw/gsm8k/main/test-00000-of-00001.parquet")
        self.gsm = {str(i): self.number(a.split("####")[-1]) for i, a in enumerate(g["answer"])}
        b = pd.read_parquet(root / "data/processed/bbh/items_target.parquet")
        self.bbh = {str(i): (str(gold), str(at), str(task), self.options(q))
                    for i, gold, at, task, q in zip(b["item_id"], b["gold_answer"], b["answer_type"], b["task"],
                                                    b["question"])}
        self.bbh_letters = {str(i): self.letters_of(q) for i, q in zip(b["item_id"], b["question"])}
        m = pd.read_parquet(root / "data/processed/math/items_target.parquet")
        self.math = {str(i): Fraction(str(gold)) for i, gold in zip(m["item_id"], m["gold_answer"])}
        h = pd.read_parquet(root / "data/processed/hotpot/items_target.parquet")
        self.hq = {str(i): str(gold) for i, gold in zip(h["item_id"], h["gold_answer"])}

    # ----------------------------------------------------------------------------------------- answer cues

    def cue(self, s: str) -> str:
        """The text without a leading answer cue, by a scan written here: an optional opening bracket, optional bold,
        qualifier words (SLOT_QUALIFIERS), "answer", optional bold, then ":" or "is" (an optional ":" after it),
        optional bold and blanks; the opening bracket's partner goes too when it ends the text."""
        if not isinstance(s, str):
            return s
        i, n = 0, len(s)
        while i < n and s[i].isspace():
            i += 1
        opener = ""
        if i < n and s[i] in "([<":
            opener = s[i]
            i += 1
            while i < n and s[i].isspace():
                i += 1

        def bold(j):
            return j + 2 if s.startswith(("**", "__"), j) else j
        i = bold(i)
        while i < n and s[i].isspace():
            i += 1
        while True:                                   # qualifier words, each followed by white space
            j = i
            while j < n and s[j].isalpha():
                j += 1
            w = s[i:j].casefold()
            if w in self.qualifiers and j < n and s[j].isspace():
                while j < n and s[j].isspace():
                    j += 1
                i = j
                continue
            break
        if s[i:i + 6].casefold() != "answer":
            return s
        i = bold(i + 6)
        k = i
        while k < n and s[k].isspace():
            k += 1
        if k < n and s[k] in ":：":
            i = k + 1
        elif k > i and s[k:k + 2].casefold() == "is" and (k + 2 >= n or not (s[k + 2].isalnum() or s[k + 2] == "_")):
            i = k + 2
            while i < n and s[i].isspace() and i + 1 < n and s[i + 1] in ":：":
                i += 1
            if i < n and s[i] in ":：":
                i += 1
        else:
            return s
        i = bold(i)
        rest = s[i:].strip()
        partner = {"(": ")", "[": "]", "<": ">"}.get(opener)
        if partner and rest.endswith(partner):
            rest = rest[:-1].rstrip()
        return rest

    # ----------------------------------------------------------------------------------------- clock times

    @staticmethod
    def clock(m) -> Fraction:
        """A clock time's value in hours on the 12-hour clock: H + MM/60, an hour from 13 to 23 less 12."""
        h, mm = int(m.group(1)), int(m.group(2))
        return (h - 12 if 13 <= h <= 23 else h) + Fraction(mm, 60)

    @staticmethod
    def colon(s: str) -> str:
        """LaTeX's braced colon between two digits, "2{:}00" (blanks allowed around and inside the braces), as ":"."""
        if "{" not in s:
            return s
        return re.sub(r"([0-9])[ \t\n]*\{[ \t\n]*:[ \t\n]*\}[ \t\n]*(?=[0-9])", r"\1:", s)

    @staticmethod
    def frac_slash(s: str) -> str:
        """every "\\frac", "\\dfrac" or "\\tfrac" (not after a letter) followed by two digit groups, each in
        braces (blanks allowed) or a single digit, written as "a/b"; anything else is left as it is. A scan."""
        if "frac" not in s:
            return s
        out, i, n = [], 0, len(s)
        while i < n:
            j = s.find("frac", i)
            if j < 0:
                out.append(s[i:])
                break
            k = j - 1
            if k >= 0 and s[k] in "dt":
                k -= 1
            if k < 0 or s[k] != "\\" or (k > 0 and s[k - 1].isalpha()):
                out.append(s[i:j + 4])
                i = j + 4
                continue
            parts, q = [], j + 4
            for _ in range(2):
                while q < n and s[q] in " \t":
                    q += 1
                if q < n and s[q] == "{":
                    r = q + 1
                    while r < n and s[r] in " \t$":
                        r += 1
                    d0 = r
                    while r < n and s[r].isdigit():
                        r += 1
                    d1 = r
                    while r < n and s[r] in " \t":
                        r += 1
                    if d1 > d0 and r < n and s[r] == "}":
                        parts.append(s[d0:d1])
                        q = r + 1
                        continue
                elif q < n and s[q].isdigit():
                    parts.append(s[q])
                    q += 1
                    continue
                break
            if len(parts) == 2:
                head = s[i:k]
                glue = " " if (head or "".join(out))[-1:].isdigit() else ""   # "6\\frac{2}{3}": a mixed number
                out.append(head + glue + parts[0] + "/" + parts[1])
                i = q
            else:
                out.append(s[i:j + 4])
                i = j + 4
        return "".join(out)

    def gsm_last(self, s: str):
        """The last number of a text, a clock time counting as one number (its value in hours), and a
        fraction "a/b", a whole number and blanks before it making a mixed number, counting as one number (its value)."""
        s = self.colon(s)
        spans = [(m.start(), m.end(), self.clock(m)) for m in CLOCK_TIME.finditer(s)]
        masked = s
        for a, b, _ in spans:
            masked = masked[:a] + " " * (b - a) + masked[b:]
        raw = [(m.start(), m.end(), ("-" if "-" in (m.group(1) or "") else "") + m.group(2))
               for m in GSM_TOKEN.finditer(masked)]
        toks, i = [], 0
        while i < len(raw):
            a0, a1, ta = raw[i]
            if i + 1 < len(raw):
                b0, b1, tb = raw[i + 1]
                before = masked[a0 - 1] if a0 > 0 else " "
                after = masked[b1:b1 + 2]
                if (masked[a1:b0].strip() == "/" and ta.lstrip("-").isdigit() and tb.isdigit() and int(tb) != 0
                        and not (before.isalnum() or before in "_/.,\\")
                        and not (after[:1].isdigit() or (after[:1] in ("/", ".") and after[1:2].isdigit()))):
                    v = Fraction(int(ta), int(tb))
                    start = a0
                    if toks and toks[-1][2] is not None and toks[-1][2].isdigit():
                        p0, p1 = toks[-1][0], toks[-1][1]
                        gap = masked[p1:a0]
                        pre = masked[p0 - 1] if p0 > 0 else " "
                        if gap and set(gap) <= {" ", "\t"} and not ta.startswith("-") and not (
                                pre.isalnum() or pre in "_/.,\\"):
                            v, start = int(toks[-1][2]) + v, p0
                            toks.pop()
                    toks.append((start, b1, None, v))
                    i += 2
                    continue
            toks.append((a0, a1, ta, self.number(ta)))
            i += 1
        allt = sorted([(a, v) for a, _, v in spans] + [(t[0], t[3]) for t in toks], key=lambda x: x[0])
        return allt[-1][1] if allt else None

    def gsm_whole(self, s: str):
        t0 = s.strip().strip("$").strip()
        t0 = t0[:-1].rstrip() if t0.endswith(".") else t0
        m = re.fullmatch(r"(-?)([0-9]+)[ \t]+([0-9]+)[ \t]*/[ \t]*([0-9]+)", t0)   # a mixed number
        if m and int(m.group(4)) != 0:
            v = int(m.group(2)) + Fraction(int(m.group(3)), int(m.group(4)))
            return -v if m.group(1) else v
        v = self.number(s)
        if v is not None:
            return v
        t = self.colon(s.strip())
        t = t[:-1].rstrip() if t.endswith(".") else t
        m = CLOCK_TIME.fullmatch(t)
        return self.clock(m) if m else None

    # ----------------------------------------------------------------------------------------- MATH-500

    @staticmethod
    def math_value(s):
        """One rational number, the numeric rule of src/math_task.py as its docstring states it, written here."""
        if s is None:
            return None
        t = s.strip()
        changed = True
        while changed:
            changed = False
            t = t.strip()
            for a, b in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
                if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
                    t, changed = t[len(a):len(t) - len(b)], True
                    break
        t = t.replace("\\displaystyle", "").replace("\\$", "$")
        t = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|\u00b0", "", t)
        t = re.sub(r"\\(?:text|textrm|mathrm|mbox)\s*\{[A-Za-z .]*\}(?:\s*\^\s*\{?\s*[23]\s*\}?)?\s*$", "", t.strip())
        t = re.sub(r"\\?%\s*$", "", t.strip())
        t = re.sub(r"\\[,!;:]|\\ |~", "", t).replace("{,}", ",").strip()
        m = re.match(r"^([-+]?)\s*\$\s*([-+]?)", t)
        if m and "$" in m.group(0):
            if m.group(1) and m.group(2):
                return None
            t = (m.group(1) or m.group(2)) + t[m.end():]
        t = t.strip()
        if t.endswith("."):
            t = t[:-1]
        t = re.sub(r"\s+", "", t)
        if not t:
            return None
        m = re.fullmatch(r"([-+]?)((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)", t)
        if m:
            v = Fraction(m.group(2).replace(",", ""))
            return -v if m.group(1) == "-" else v
        m = re.fullmatch(r"([-+]?)\\[dt]?frac(?:\{([-+]?)([0-9]+)\}|([0-9]))(?:\{([-+]?)([0-9]+)\}|([0-9]))", t)
        if m:
            signs = [x for x in (m.group(1), m.group(2), m.group(5)) if x]
            if len(signs) > 1:
                return None
            a, b = int(m.group(3) or m.group(4)), int(m.group(6) or m.group(7))
            if b == 0:
                return None
            return (-1 if signs and signs[0] == "-" else 1) * Fraction(a, b)
        m = re.fullmatch(r"([-+]?)([0-9]+)/([0-9]+)", t)
        if m and int(m.group(3)) != 0:
            return (-1 if m.group(1) == "-" else 1) * Fraction(int(m.group(2)), int(m.group(3)))
        return None

    def unwrap_math(self, s: str) -> str:
        """A MATH-500 box's content as the content reader unwraps it: wrappers, LaTeX spacing as blanks,
        bold, a copied placeholder, an answer cue and angle brackets around the whole (no bare-group removal: a MATH box
        holds \\frac{a}{b})."""
        prev = None
        while prev != s:
            prev, s = s, self.wrap.sub(r"\1", s)
        s = self.blanks(s).replace("**", "").strip()
        s = self.cue(re.sub(r"^<(?:answer|number)>\s*[-:]?\s*", "", s, flags=re.I))
        m = re.match(r"^<([^<>]*)>$", s)
        return self.cue(m.group(1).strip()) if m else s

    def math_item(self, item: str, content: str | None, whole: str | None) -> bool:
        gold = self.math[item]
        if content is not None:
            v = None
            for c in (content, self.unwrap_math(content)):
                v = self.math_value(c)
                if v is None and "=" in c:
                    v = self.math_value(c.rsplit("=", 1)[1])
                if v is not None:
                    break
        else:
            v = self.math_value(whole) or self.math_value(self.unwrap_math(whole))
            if v is None:
                t = whole.replace("{,}", ",")
                t = re.sub(r"(?<=[0-9]),?\\[,!;:]\s*(?=[0-9]{3}(?![0-9]))", ",", t)
                found = MATH_NUMBER.findall(t)
                v = self.math_value(found[-1]) if found else None
        return v is not None and v == gold

    # ----------------------------------------------------------------------------------------- HotpotQA

    @staticmethod
    def hq_norm(s: str) -> str:
        """HotpotQA's normalisation, written here: lower case, punctuation removed, the articles a, an, the removed,
        white space collapsed."""
        import string
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @staticmethod
    def hq_close(t: str, start: int):
        depth, i = 1, start
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
                    return i
            i += 1
        return None

    def hq_unwrap(self, s: str) -> str:
        """HotpotQA's unwrap: the wrappers text, textbf, textit, textrm, mathrm, emph replaced by their contents, every
        "$" removed, stripped."""
        t = s
        for _ in range(20):
            m = re.search(r"\\(?:text|textbf|textit|textrm|mathrm|emph)\s*\{", t)
            if not m:
                break
            j = self.hq_close(t, m.end())
            inner = t[m.end():] if j is None else t[m.end():j]
            t = t[:m.start()] + inner + ("" if j is None else t[j + 1:])
        return t.replace("$", "").strip()

    def hq_clean(self, u: str) -> str:
        u = self.tag_whole(self.blanks(self.typo(u)).strip())         # typographic marks, answer tags
        u = self.cue(re.sub(r"^\s*<answer>\s*[-:]?\s*", "", u, flags=re.I).strip())
        if len(u) > 1 and u[0] == "<" and u[-1] == ">":
            u = self.cue(u[1:-1].strip())
        return u

    def hq_placeholder(self, c: str) -> bool:
        return self.is_placeholder(c) or self.hq_norm(self.hq_clean(self.hq_unwrap(c))) in ("", "answer")

    def hq_boxes(self, t: str, bare: bool = False) -> list[tuple[int, int | None, int]]:
        """(start, index of the closing brace or None, index after the opening brace) of every box, "\\boxed" then
        white space then "{" (bare: "boxed" with no backslash before it)."""
        rx = r"(?<!\\)boxed\s*\{" if bare else r"\\boxed\s*\{"
        return [(m.start(), self.hq_close(t, m.end()), m.end()) for m in re.finditer(rx, t)]

    def hq_read(self, text: str):
        t = text if isinstance(text, str) else ""
        bx = self.hq_boxes(t)
        complete = [(a, t[o:j]) for a, j, o in bx if j is not None]
        unclosed = None
        for a, j, o in bx:
            if j is None:
                unclosed = t[o:]
        if unclosed is not None and complete and bx[-1][1] is not None:
            unclosed = None
        spans = [(a, j + 1) for a, j, o in bx if j is not None]
        others = self.other_boxes(t, spans) if "boxed" in t.lower() else []
        tags = [(a, c) for a, _, c in self.tags(t) if "boxed" not in c.lower()
                and not any(x <= a < y for x, y in spans)]
        if others or tags:
            items = [(a, c) for a, c in complete if not self.hq_placeholder(c)]
            items += [(a, c) for a, _, c in others if not (self.hq_placeholder(c) or self.hq_placeholder("<" + c.strip() + ">"))]
            items += [(a, c) for a, c in tags if not self.hq_placeholder(c)]
            if items:
                return self.hq_clean(self.hq_unwrap(max(items, key=lambda x: x[0])[1]))
        else:
            for _, c in reversed(complete):
                if not self.hq_placeholder(c):
                    return self.hq_clean(self.hq_unwrap(c))
        t2 = t
        for bare in (False, True):
            out, i = [], 0
            for a, j, o in self.hq_boxes(t2, bare):
                if a < i:
                    continue
                if j is not None and self.hq_placeholder(t2[o:j]):
                    out += [t2[i:a], " "]
                    i = j + 1
            t2 = "".join(out) + t2[i:]
        t2 = re.sub(r"\\?boxed\s*(<[^<>{}\n]{0,40}>)\}?", lambda m: " " if self.is_placeholder(m.group(1)) else m.group(0),
                    t2)
        t2 = self.drop_tags(t2, self.hq_placeholder)
        if "boxed" in t2.lower():
            drop = [(a, b) for a, b, c in self.other_boxes(t2, [])
                    if self.hq_placeholder(c) or self.hq_placeholder("<" + c.strip() + ">")]
            for a, b in reversed(drop):
                t2 = t2[:a] + " " + t2[b:]
        if unclosed is not None and not complete:
            return self.hq_clean(self.hq_unwrap(unclosed))
        cues = list(re.finditer(r"answer\s*(?:is\b\s*:?|:)", t2, re.I))
        if cues:
            return self.hq_clean(self.hq_unwrap(t2[cues[-1].end():].split("\n", 1)[0]))
        lines = [ln.strip() for ln in t2.split("\n") if ln.strip()]
        if not lines:
            return None
        last = lines[-1]
        m = re.search(r"(?<!\\)boxed\s*\{", last)
        if m:
            j = self.hq_close(last, m.end())
            if j is not None:
                return self.hq_clean(self.hq_unwrap(last[m.end():j]))
        return self.hq_clean(self.hq_unwrap(last))

    def hotpot_item(self, item: str, text: str) -> bool:
        a = self.hq_read(text)
        return a is not None and self.hq_norm(a) == self.hq_norm(self.typo(self.hq[item]))    # typography

    @staticmethod
    def number(s: str):
        if re.search(r"[0-9.,][ \t\n]+[0-9]", s.strip()):     # "180, 96, 120" is a list, not a number
            return None
        s = s.strip().replace(",", "").replace("$", "").replace(" ", "").replace("−", "-")
        try:
            v = Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
        return v

    @staticmethod
    def fold(s: str) -> str:
        s = " ".join(s.split()).strip().strip("\"'`“”‘’").strip()
        while s.endswith("."):
            s = s[:-1].rstrip()
        return s.strip("\"'`“”‘’").strip().casefold()     # quotes after a final period too

    def options(self, q: str) -> dict:
        i = q.rfind("Options:")
        if i < 0:
            return {}
        seen = defaultdict(list)
        for letter, text in re.findall(r"^\(([A-Z])\)\s*(.*?)\s*$", q[i:], re.M):
            seen[self.fold(text)].append(letter)
        return {t: ls[0] for t, ls in seen.items() if t and len(ls) == 1}

    HEDGE_WORDS = {"if", "whether", "unless", "otherwise", "cannot", "can't", "unable", "undetermined", "determine",
                   "determined", "unclear", "unknown", "impossible", "depends", "depending"}
    HEDGE_PAIRS = {("can", "not"), ("not", "sure")}

    def hedge(self, f: str) -> bool:
        """written from src/content_scoring.py's statement: the last sentence of a folded content (after the
        last ".", "!", "?" or ";" followed by white space, or the last line) holds a condition or hedge word."""
        last = f.split("\n")[-1]
        cut = -1
        for i in range(len(last) - 1):
            if last[i] in ".!?;" and last[i + 1].isspace():
                cut = i
        last = last[cut + 1:]
        words = re.findall(r"[a-z']+", last)
        return any(w in self.HEDGE_WORDS for w in words) or any(p in self.HEDGE_PAIRS for p in zip(words, words[1:]))

    @staticmethod
    def letters_of(q: str) -> set:
        """every option letter of the question's "Options:" block, letters of shared texts included."""
        i = q.rfind("Options:")
        if i < 0:
            return set()
        return set(re.findall(r"^\(([A-Z])\)", q[i:], re.M))

    # typographic quotes, apostrophes and dashes, as HotpotQA's content reading takes them
    TYPO = {**dict.fromkeys("\u201c\u201d\u201e\u201f\u2033", '"'), **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"),
            **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d", "-")}

    @classmethod
    def typo(cls, s):
        return "".join(cls.TYPO.get(ch, ch) for ch in s) if isinstance(s, str) else s

    @staticmethod
    def tag_end(low: str, i: int, closing: bool):
        """The index after an answer tag that starts at i ("<", blanks or tabs, "/" when closing, blanks or tabs,
        "answer", blanks or tabs, ">"), or None."""
        n = len(low)
        if i >= n or low[i] != "<":
            return None
        j = i + 1
        while j < n and low[j] in " \t":
            j += 1
        if closing:
            if j >= n or low[j] != "/":
                return None
            j += 1
            while j < n and low[j] in " \t":
                j += 1
        if not low.startswith("answer", j):
            return None
        j += 6
        while j < n and low[j] in " \t":
            j += 1
        return j + 1 if j < n and low[j] == ">" else None

    def tags(self, t: str) -> list[tuple[int, int, str]]:
        """(start, end, content) of every answer tag pair, by a scan written here: an opening tag, then the
        content up to the first closing tag on the same line; the scan goes on after a pair, or one character after an
        opening tag with no closing tag on its line."""
        low, n, out, i = t.lower(), len(t), [], 0
        while True:
            i = low.find("<", i)
            if i < 0:
                return out
            o = self.tag_end(low, i, False)
            if o is None:
                i += 1
                continue
            e = low.find("\n", o)
            line_end = n if e < 0 else e
            k, hit = o, None
            while True:
                k = low.find("<", k, line_end)
                if k < 0:
                    break
                c = self.tag_end(low, k, True)
                if c is not None:
                    hit = (i, c, t[o:k])
                    break
                k += 1
            if hit:
                out.append(hit)
                i = hit[1]
            else:
                i += 1

    @staticmethod
    def tag_whole(s: str) -> str:
        """A content wholly in an answer tag pair, or ending with a closing tag, without the tags."""
        m = re.match(r"^<[ \t]*answer[ \t]*>(.*)<[ \t]*/[ \t]*answer[ \t]*>$", s, re.I | re.S)
        if m:
            s = m.group(1).strip()
        return re.sub(r"\s*<[ \t]*/[ \t]*answer[ \t]*>$", "", s, flags=re.I)

    def drop_tags(self, t: str, placeholder) -> str:
        """The text with every answer tag pair whose content is a placeholder, and every empty pair over several lines,
        replaced by a space."""
        if "answer" not in t.lower():
            return t
        for a, b, c in reversed(self.tags(t)):
            if placeholder(c):
                t = t[:a] + " " + t[b:]
        return re.sub(r"<[ \t]*answer[ \t]*>\s*<[ \t]*/[ \t]*answer[ \t]*>", " ", t, flags=re.I)

    def range_option(self, item: str, content: str):
        """on temporal_sequences, the option an open box cut after "to" names: its content, unwrapped and
        folded, an optional "(", a start and " to"; the option whose text begins with that start and " to ", when
        exactly one does ("A", no parentheses), else None."""
        gold, at, task, opts = self.bbh[item]
        if task != "temporal_sequences" or not opts:
            return None
        f = self.fold(self.unwrap(content))
        if f.startswith("("):
            f = f[1:].lstrip()
        if not f.endswith(" to") or len(f) < 4:
            return None
        start = f[:-3].rstrip()
        hits = {v for k, v in opts.items() if k.startswith(start + " to ")}
        return next(iter(hits)) if len(hits) == 1 else None

    def option_text(self, rest: str, opts: dict):
        """The option ("A") whose text the rest of a box is, read as a whole (a number 0-20 also as its word), or None."""
        if not opts:
            return None
        f = self.fold(rest)
        if len(f) > 1 and f[0] == "(" and f[-1] == ")" and f[1:-1].strip() in opts:
            f = f[1:-1].strip()
        if f not in opts and f.isdigit() and int(f) < len(self.NUMBER_WORDS):
            f = self.NUMBER_WORDS[int(f)]
        return opts.get(f)

    @staticmethod
    def blanks(s: str) -> str:
        """LaTeX spacing commands as blanks, by a scan written here: a backslash followed by a blank, ",",
        ";", ":" or "!", or by "quad" or "qquad" with no letter after, is one blank, and so is "~"; a backslash right
        after another backslash (the line break "\\\\") starts no command. A ",", ";", ":" or "!" command after a digit
        (a comma between them allowed) and before exactly three digits is a thousands comma."""
        if not isinstance(s, str) or ("\\" not in s and "~" not in s):
            return s
        out, i, n = [], 0, len(s)
        while i < n:
            c = s[i]
            if c == "~":
                # "~" after a digit (a comma between them allowed) and before exactly three digits is a
                # thousands comma, as the thin space is
                prev = s[i - 1] if i > 0 else ""
                digits = "0123456789"
                three, after = s[i + 1:i + 4], (s[i + 4] if i + 4 < n else "")
                if ((prev != "" and prev in digits) or (prev == "," and i > 1 and s[i - 2] in digits)) \
                        and len(three) == 3 and all(ch in digits for ch in three) and (after == "" or after not in digits):
                    if prev != ",":
                        out.append(",")
                else:
                    out.append(" ")
                i += 1
                continue
            if c == "\\" and i + 1 < n:
                nxt = s[i + 1]
                prev = s[i - 1] if i > 0 else ""
                if nxt in ",;:! ":           # a backslash before a blank too
                    digits = "0123456789"
                    three = s[i + 2:i + 5]
                    after = s[i + 5] if i + 5 < n else ""
                    digit_before = (prev != "" and prev in digits) or (prev == "," and i > 1 and s[i - 2] in digits)
                    if digit_before and len(three) == 3 and all(ch in digits for ch in three) \
                            and (after == "" or after not in digits):
                        if prev != ",":
                            out.append(",")
                        i += 2
                        continue
                if prev != "\\":
                    if nxt in " ,;:!":
                        out.append(" ")
                        i += 2
                        continue
                    for word in ("qquad", "quad"):
                        if s.startswith(word, i + 1):
                            j = i + 1 + len(word)
                            if j >= n or not ("a" <= s[j].lower() <= "z"):
                                out.append(" ")
                                i = j
                                break
                    else:
                        out.append(c)
                        i += 1
                    continue
            out.append(c)
            i += 1
        return "".join(out)

    def unwrap(self, s: str) -> str:
        prev = None
        while prev != s:
            prev, s = s, self.wrap.sub(r"\1", s)
        s = self.blanks(s)
        # TeX: a bare group is its text ("({D})"); a backslash before a lone option letter is stray ("(\C)")
        s = re.sub(r"(?<!\\)[{}]", "", s)
        s = re.sub(r"\\([A-R])(?![A-Za-z])", r"\1", s)
        s = self.tag_whole(s.replace("**", "").strip())
        s = self.cue(re.sub(r"^<(?:answer|number)>\s*[-:]?\s*", "", s, flags=re.I))
        m = re.match(r"^<([^<>]*)>$", s)
        return self.cue(m.group(1).strip()) if m else s

    def slot_name(self, words: list[str], nouns: set) -> bool:
        """Qualifier words, one noun of `nouns`, then nothing or one tail."""
        i = 0
        while i < len(words) and words[i] in self.qualifiers:
            i += 1
        if i >= len(words) or words[i] not in nouns:
            return False
        return i == len(words) - 1 or " ".join(words[i + 1:]) in self.tails

    def is_placeholder(self, content: str) -> bool:
        """The placeholder rule: empty; a slot name bare; in angle brackets, a slot name (an option named
        with its letter, "<Option A>", is a choice, read as the letter)."""
        quotes = "\"'`\u201c\u201d\u2018\u2019"
        prev, s = None, content
        while prev != s:
            prev, s = s, self.wrap.sub(r"\1", s)
        s = " ".join(self.blanks(s).replace("**", "").replace("$", "").split()).strip(quotes).strip()
        while s.endswith("."):
            s = s[:-1].rstrip()
        s = s.strip(quotes).strip().casefold()
        if not s or self.slot_name(s.split(), self.slot_words):
            return True
        m = re.match(r"^<[ \t]*answer[ \t]*>(.*)<[ \t]*/[ \t]*answer[ \t]*>$", s, re.S)
        if m:
            return self.is_placeholder(m.group(1))
        if len(s) < 2 or s[0] != "<" or s[-1] != ">" or "<" in s[1:-1] or ">" in s[1:-1]:
            return False
        return self.slot_name(s[1:-1].split(), self.slot_nouns)

    @staticmethod
    def option_named(f: str):
        """a folded content that names an option by its letter ("option a", "choice (b)", "letter c"), read
        as "(X)"; None otherwise."""
        for word in ("option", "choice", "letter"):
            if f.startswith(word):
                rest = "".join(f[len(word):].split())
                if len(rest) == 1 and "a" <= rest <= "z":
                    return f"({rest.upper()})"
                if len(rest) == 3 and rest[0] == "(" and rest[2] == ")" and "a" <= rest[1] <= "z":
                    return f"({rest[1].upper()})"
                return None
        return None

    def lead_letter(self, s: str, opts: dict, letters: set | None = None):
        """written here from src/content_scoring.py's statement: a box that opens with an option letter X,
        written "(X)", "X)", "X.", "X:" or "X -", and goes on with more text reads X, unless the text after it names
        another option: "(Y)", or a capital Y not after a letter or "(" and followed by ")", ":" or a blank, a dash and a
        blank, Y not X. A bare "X", a blank, then option X's own text (in parentheses or not) reads X too.
        a lowercase letter in "(x)" or "x)" is read as its capital under the same rules, a lowercase "(y)" or "y)" in
        the rest (y not after a letter or "(") naming another option."""
        t, n = s.strip(), len(s.strip())
        x = rest = None
        labelled = True
        low = False             # a lowercase label in brackets, "(k)" or "k)", read as its capital
        if n and t[0] == "(":
            j = 1
            while j < n and t[j].isspace():
                j += 1
            if j < n and ("A" <= t[j] <= "Z" or "a" <= t[j] <= "z"):
                k = j + 1
                while k < n and t[k].isspace():
                    k += 1
                if k < n and t[k] == ")":
                    x, rest, low = t[j].upper(), t[k + 1:].strip(), "a" <= t[j] <= "z"
        elif n and "a" <= t[0] <= "z":
            k = 1
            while k < n and t[k].isspace():
                k += 1
            if k < n and t[k] == ")":
                x, rest, low = t[0].upper(), t[k + 1:].strip(), True
        elif n and "A" <= t[0] <= "Z":
            k = 1
            while k < n and t[k].isspace():
                k += 1
            if k < n and t[k] in ").:":
                x, rest = t[0], t[k + 1:].strip()
            elif k > 1 and k < n and t[k] in "-–—":
                x, rest = t[0], t[k + 1:].strip()
            elif k > 1:
                f = self.fold(t[k:])
                if len(f) > 1 and f[0] == "(" and f[-1] == ")":
                    f = f[1:-1].strip()
                if opts and opts.get(f) == t[0]:
                    x, rest = t[0], t[k:]
                    labelled = False
        if not x or not rest:
            return None
        if labelled:
            # the text after the letter is another option's text, read as a whole: two answers
            y = self.option_text(rest, opts)
            if letters and x not in letters:          # a capital that is not one of the item's letters: no label;
                return f"({y})" if y else None        # the option its text names, if any
            if y is not None and y != x:
                return "CONFLICT"
        m = len(rest)
        for i, c in enumerate(rest):
            if not ("A" <= c <= "Z") or c == x:
                continue
            if i > 0 and rest[i - 1] == "(" and i + 1 < m and rest[i + 1] == ")":
                return None
            # "( Y )" with blanks inside
            a = i - 1
            while a >= 0 and rest[a].isspace():
                a -= 1
            b = i + 1
            while b < m and rest[b].isspace():
                b += 1
            if a >= 0 and rest[a] == "(" and b < m and rest[b] == ")":
                return None
            before = rest[i - 1] if i > 0 else ""
            if before.isascii() and (before.isalpha() or before == "("):
                continue
            if b < m and rest[b] in "):":
                return None
            if b > i + 1 and b < m and rest[b] in "-–—" and b + 1 < m and rest[b + 1].isspace():
                return None
        if low:                 # a second lowercase label, "(y)" or "y)", y not x
            for i, c in enumerate(rest):
                if not ("a" <= c <= "z") or c.upper() == x:
                    continue
                a = i - 1
                while a >= 0 and rest[a].isspace():
                    a -= 1
                b = i + 1
                while b < m and rest[b].isspace():
                    b += 1
                if a >= 0 and rest[a] == "(" and b < m and rest[b] == ")":
                    return None
                before = rest[i - 1] if i > 0 else ""
                if before.isascii() and (before.isalpha() or before == "("):
                    continue
                if b < m and rest[b] == ")":
                    return None
        return f"({x})"

    def drop_placeholders(self, t: str) -> str:
        """The reply with its complete placeholder boxes and brace-less placeholder copies replaced by a space."""
        out, i = [], 0
        for a, b in self.box_spans(t):
            if a >= i and self.is_placeholder(t[a + len(PRE):b - 1]):
                out += [t[i:a], " "]
                i = b
        t = "".join(out) + t[i:]
        t = re.sub(r"\\?boxed\s*(<[^<>{}\n]{0,40}>)\}?",
                   lambda m: " " if self.is_placeholder(m.group(1)) else m.group(0), t)
        t = self.drop_tags(t, self.is_placeholder)
        # a placeholder in a box written with other brackets goes as well
        drop = [(a, b) for a, b, c in self.other_boxes(t, []) if self.other_placeholder(c)]
        for a, b in reversed(drop):
            t = t[:a] + " " + t[b:]
        return t

    def other_placeholder(self, c: str) -> bool:
        """A box written with other brackets holding X is a placeholder when X, or X in angle brackets, is one."""
        return self.is_placeholder(c) or self.is_placeholder("<" + c.strip() + ">")

    @staticmethod
    def other_boxes(t: str, spans: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        """(start, end, content) of the boxes written with other brackets, found here by a scan of the
        text: "boxed" (a backslash before it optional), white space, "<", then X up to the next ">" with no "<", ">"
        or line break inside; or a tag "<boxed>" (white space allowed inside), then X up to the first "</boxed" on the
        same line, the closing ">" optional. A box that opens inside a complete \\boxed{...} (`spans`) is left out.
        In the order of their openings; a box does not open inside an earlier one."""
        low, n, out, i = t.lower(), len(t), [], 0
        while True:
            j = low.find("boxed", i)
            if j < 0:
                break
            hit = None
            # the tag: "<", spaces or tabs, "boxed", spaces or tabs, ">"
            a = j - 1
            while a >= 0 and t[a] in " \t":
                a -= 1
            kt = j + 5
            while kt < n and t[kt] in " \t":
                kt += 1
            k = j + 5          # the angle form: any white space, then "<"
            while k < n and t[k].isspace():
                k += 1
            if a >= 0 and t[a] == "<" and kt < n and t[kt] == ">":
                k = kt
                e = low.find("\n", k + 1)
                line_end = n if e < 0 else e
                c = k + 1
                while c < line_end:
                    if t[c] == "<":
                        q = c + 1
                        while q < line_end and t[q] in " \t":
                            q += 1
                        if q < line_end and t[q] == "/":
                            q += 1
                            while q < line_end and t[q] in " \t":
                                q += 1
                            if low.startswith("boxed", q):
                                end = q + 5
                                while end < n and t[end] in " \t":
                                    end += 1
                                end = end + 1 if end < n and t[end] == ">" else end
                                hit = (a, end, t[k + 1:c])
                                break
                    c += 1
            elif k < n and t[k] == "<":
                c = k + 1
                while c < n and t[c] not in "<>\n":
                    c += 1
                if c < n and t[c] == ">":
                    start = j - 1 if j > 0 and t[j - 1] == "\\" else j
                    hit = (start, c + 1, t[k + 1:c])
            if hit:
                if not any(s <= hit[0] < e_ for s, e_ in spans):
                    out.append(hit)
                i = hit[1]
            else:
                i = j + 5
        return out

    @staticmethod
    def box_spans(t: str) -> list[tuple[int, int]]:
        """(start, end) of every complete box, end just after its closing brace."""
        out, i = [], 0
        while True:
            j = t.find(PRE, i)
            if j < 0:
                return out
            depth, k = 1, j + len(PRE)
            while k < len(t) and depth:
                if t[k] == "\\":
                    k += 2
                    continue
                depth += t[k] == "{"
                depth -= t[k] == "}"
                k += 1
            if depth == 0:
                out.append((j, k))
            i = j + len(PRE)

    @staticmethod
    def boxes(t: str) -> list[str]:
        out, i = [], 0
        while True:
            j = t.find(PRE, i)
            if j < 0:
                return out
            depth, k = 1, j + len(PRE)
            while k < len(t) and depth:
                if t[k] == "\\":
                    k += 2
                    continue
                depth += t[k] == "{"
                depth -= t[k] == "}"
                k += 1
            if depth == 0:
                out.append(t[j + len(PRE): k - 1])
            i = j + len(PRE)

    def gsm8k(self, item: str, content: str | None, whole: str | None):
        gold = self.gsm[item]
        # fractions written as "a/b" before the unwrap (which removes bare braces) reads them
        content = self.frac_slash(content) if content is not None else None
        whole = self.frac_slash(whole) if whole is not None else None
        s = self.unwrap(content) if content is not None else whole
        v = self.gsm_whole(s) if content is not None else (self.gsm_whole(self.unwrap(whole)) or self.gsm_whole(whole))
        if v is None:
            v = self.gsm_last(s)
        return v is not None and v == gold

    NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                    "sixteen seventeen eighteen nineteen twenty").split()

    def bbh_exact(self, s: str, at: str, task: str, opts: dict):
        """The content read as a whole answer of the item's type: a letter "(X)", "X" or "X)" (a final period dropped), an option named with its letter ("Option
        A"), the text of exactly one option (a number 0-20 also as its English word), Yes/No, True/False,
        the task's own wording; None when it is none of these."""
        t = s.strip()
        if at == "integer":
            return self.number(t)
        t1 = t[:-1] if t.endswith(".") else t
        f = self.fold(s)
        if at == "letter":
            m = re.fullmatch(r"\(([A-Z])\)|([A-Z])\)?", t1)
            if m:
                return f"({m.group(1) or m.group(2)})"
            # a lowercase letter in brackets, "(b)" or "b)", whole
            if len(t1) == 3 and t1[0] == "(" and t1[2] == ")" and "a" <= t1[1] <= "z":
                return f"({t1[1].upper()})"
            if len(t1) == 2 and t1[1] == ")" and "a" <= t1[0] <= "z":
                return f"({t1[0].upper()})"
            named = self.option_named(f)
            if named:
                return named
            if opts:
                if f not in opts and f.isdigit() and int(f) < len(self.NUMBER_WORDS):
                    f = self.NUMBER_WORDS[int(f)]
                return f"({opts[f]})" if f in opts else None
            return None
        yes, no = at.split(":", 1)[1].split("/")
        for w in (yes, no):
            if t1.casefold() == w.casefold():
                return w.casefold()
        if yes.casefold() == "yes":
            if f in ("true", "false"):
                return (yes if f == "true" else no).casefold()
            for rx, j in self.YESNO_WORDS.get(task, ()):
                if re.search(rx, f):
                    if task == "web_of_lies" and self.hedge(f):     # a condition or a hedge
                        return None
                    return (yes, no)[j].casefold()
            if task == "sports_understanding":
                j = self.sports_sentence(f)
                if j is not None:
                    return (yes, no)[j].casefold()
        return None

    @staticmethod
    def sports_sentence(f: str):
        """written from src/content_scoring.py's statement: a folded content that is the plausibility
        verdict as one sentence, "sentence" or "statement" (after "the" or "this", or alone), the judged sentence in
        double quotes optional, "is", then "plausible" (0) or "not plausible" or "implausible" (1); None otherwise."""
        w = f.split()
        if len(w) >= 3 and w[-2:] == ["not", "plausible"]:
            j, w = 1, w[:-2]
        elif w and w[-1] in ("plausible", "implausible"):
            j, w = (0 if w[-1] == "plausible" else 1), w[:-1]
        else:
            return None
        if not w or w[-1] != "is":
            return None
        w = w[:-1]
        if w and w[0] in ("the", "this"):
            w = w[1:]
        if not w or w[0] not in ("sentence", "statement"):
            return None
        q = " ".join(w[1:])
        if not q:
            return j
        for a, b in (('"', '"'), ("“", "”")):
            if len(q) >= 2 and q[0] == a and q[-1] == b and a not in q[1:-1] and b not in q[1:-1]:
                return j
        return None

    @staticmethod
    def bbh_last(s: str, at: str):
        """The last answer token of the item's type: a number; a letter "(X)", or a capital letter standing alone
        (not after "("), a ")" after it optional; one of the two answer words, whole words, any case."""
        if at == "integer":
            ns = NUMBER.findall(s)
            return Reader.number(ns[-1]) if ns else None
        if at == "letter":
            found = re.findall(r"(?<!\w)(?:\(([A-Z])\)|(?<!\()([A-Z])\)?)(?!\w)", s)
            return f"({found[-1][0] or found[-1][1]})" if found else None
        words = at.split(":", 1)[1].split("/")
        found = re.findall(r"(?<!\w)(" + "|".join(map(re.escape, words)) + r")(?!\w)", s, re.I)
        return found[-1].casefold() if found else None

    def bbh_item(self, item: str, content: str | None, whole: str | None):
        gold, at, task, opts = self.bbh[item]
        s = self.unwrap(content) if content is not None else self.unwrap(whole)
        v = self.bbh_exact(s, at, task, opts)
        if v is None and content is not None and at == "letter":
            v = self.lead_letter(s, opts, self.bbh_letters.get(item))   # "(C) Star Wars ..." reads (C)
            if v == "CONFLICT":                  # letter and text name different options
                return False
        if v is None:
            v = self.bbh_last(s if content is not None else whole, at)
        if v is None:
            return False
        if at == "integer":
            return v == self.number(gold)
        return v == (gold if at == "letter" else gold.casefold())

    def correct(self, task: str, item: str, text: str, finish) -> bool:
        t = text if isinstance(text, str) else ""
        if task == "hotpot":
            return self.hotpot_item(item, t)
        t = self.norm_text(t)
        spans = self.box_spans(t)
        found = [(a, t[a + len(PRE):b - 1]) for a, b in spans if not self.is_placeholder(t[a + len(PRE):b - 1])]
        found += [(a, c) for a, _, c in self.other_boxes(t, spans) if not self.other_placeholder(c)]
        found += [(a, c) for a, _, c in self.tags(t) if "boxed" not in c.lower()
                  and not any(x <= a < y for x, y in spans) and not self.is_placeholder(c)]
        bx = [c for _, c in sorted(found)]
        if bx:
            content, whole = bx[-1], None
        else:
            whole = self.drop_placeholders(t)
            if finish == "length":
                return False
            if PRE in whole:            # on temporal_sequences, a range cut at "to" names its option
                if task != "bbh":
                    return False
                y = self.range_option(item, whole[whole.rfind(PRE) + len(PRE):])
                return y is not None and f"({y})" == self.bbh[item][0]
            content = None
        return {"gsm8k": self.gsm8k, "math": self.math_item, "bbh": self.bbh_item}[task](item, content, whole)


# --------------------------------------------------------------------------------------------- the two checks

class ApiCutter:
    """The API replies' cut at the panel's cap, written here from the collections' statements (API_TOKENIZER): the
    tokenizers are loaded with the tokenizers library, never through the collections' code."""

    def __init__(self, root: Path):
        self.root, self.tok = Path(root), {}

    def _tok(self, key: str):
        if key not in self.tok:
            from tokenizers import Tokenizer
            how, where = API_TOKENIZER[key]
            if isinstance(where, tuple):
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(where[0], "tokenizer.json", revision=where[1], local_files_only=True)
            else:
                path = str(self.root / where)
            self.tok[key] = (how, Tokenizer.from_file(path))
        return self.tok[key]

    def cut(self, key: str, text: str, n: int) -> str:
        how, tk = self._tok(key)
        enc = tk.encode(text or "", add_special_tokens=False)
        if len(enc.ids) <= n:
            return text
        if how == "offsets":
            return text[: enc.offsets[n - 1][1]]
        return tk.decode(enc.ids[:n], skip_special_tokens=False)


def scored_path(root: Path, key: str, task: str, reading: str) -> Path:
    tbl = SCORED_UNCUT if reading == "uncut" else SCORED_CUT
    return root / (tbl[key] if key in API_RAW else tbl["cluster"]).format(task=task, key=key)


def reading_rows(root: Path, key: str, task: str, reading: str, cutter: ApiCutter | None) -> tuple[pd.DataFrame, dict]:
    """(rows, stops) of one model and benchmark in one reading: request_id, arm, item_id, text (as the reading scores
    it, before the forced arms' prefill and completion), finish, cut (cut at the panel's cap); stops: arm -> whether
    the arm stops at the first "}"."""
    if key in API_RAW:
        raw = api_rows(root, key, task, SAMPLE_ARMS)
        stops = {a: a in FORCED_ARMS and task in STOP_TASKS for a in SAMPLE_ARMS}
        raw["cut"] = False
        if reading == "cut" and len(raw):
            sc = pd.read_parquet(scored_path(root, key, task, "cut"),
                                 columns=["request_id", "cut_at_panel_cap", "panel_cap"])
            info = {str(r): (bool(c), int(p)) for r, c, p in zip(sc["request_id"], sc["cut_at_panel_cap"],
                                                                     sc["panel_cap"])}
            texts, fins, cuts = [], [], []
            for rid, t, f in zip(raw["request_id"], raw["output_text"], raw["finish_reason"]):
                c, n = info.get(str(rid), (False, 0))
                texts.append(cutter.cut(key, t, n) if c else t)
                fins.append("length" if c else f)
                cuts.append(c)
            raw["output_text"], raw["finish_reason"], raw["cut"] = texts, fins, cuts
        return raw.rename(columns={"output_text": "text", "finish_reason": "finish"}), stops
    rk = (root / "data/d2_out/ladder2/chosen" / key).read_text(encoding="utf-8").strip()
    folder = root / "data/d2_out/ladder2" / rk / task
    dec = decoding_of(folder)
    cols = ("request_id", "arm", "config_id", "item_id", "output_text", "finish_reason", "output_text_cut",
            "finish_reason_cut", "cut_at_panel_cap")
    raw = shard_rows(folder, SAMPLE_ARMS, cols)
    stops = {a: stops_at_brace(dec, a) for a in SAMPLE_ARMS}
    if reading == "cut":
        cut = raw["cut_at_panel_cap"].astype(bool)
        text = raw["output_text_cut"].where(cut, raw["output_text"])
        out = pd.DataFrame({"request_id": raw["request_id"], "arm": raw["arm"], "item_id": raw["item_id"],
                            "text": text, "finish": raw["finish_reason_cut"], "cut": cut})
    else:
        out = pd.DataFrame({"request_id": raw["request_id"], "arm": raw["arm"], "item_id": raw["item_id"],
                            "text": raw["output_text"], "finish": raw["finish_reason"], "cut": False})
    return out, stops


def sample_check(root: Path, rule: StopRule, reader: Reader, keys: list[str], tasks=SAMPLE_TASKS,
                 readings=SAMPLE_READINGS, n_sample: int = SAMPLE_N, n_cut: int = SAMPLE_CUT_N,
                 scored_hook=None) -> dict:
    """Seeded sample per (model, benchmark, reading, arm): the reply as the reading scores it, read here (forced arms
    with the prefill in front and completed by the stop's rule as stated), against content_correct of that reading's
    scored rows. In the cut reading the replies cut at the panel's cap are drawn apart (stratum "cut at the cap", up
    to n_cut per arm) beside the random draw (stratum "random"). scored_hook (tests): a function applied to each
    scored table before it is read, to plant an error. Cells: (model, benchmark, reading, arm, stratum) ->
    (agree, n)."""
    rng = random.Random(SAMPLE_SEED)
    cutter = ApiCutter(root)
    cells, disagree = {}, []
    for key in keys:
        for task in tasks:
            for reading in readings:
                rows, stops = reading_rows(root, key, task, reading, cutter)
                sc = pd.read_parquet(scored_path(root, key, task, reading), columns=["request_id", "content_correct"])
                if scored_hook is not None:
                    sc = scored_hook(key, task, reading, sc)
                truth = dict(zip(sc["request_id"].astype(str), sc["content_correct"].astype(int)))
                for arm, g in rows.groupby("arm"):
                    idx = sorted(g.index)
                    strata = [("random", rng.sample(idx, min(n_sample, len(idx))))]
                    if reading == "cut":
                        ci = sorted(g.index[g["cut"].astype(bool).to_numpy()])
                        if ci:
                            strata.append(("cut at the cap", rng.sample(ci, min(n_cut, len(ci)))))
                    for stratum, pick in strata:
                        n = agree = 0
                        for j in pick:
                            r = g.loc[j]
                            text = r["text"] if isinstance(r["text"], str) else ""
                            if arm in FORCED_ARMS:
                                text = rule.state(PRE + text, stops[arm])[1]
                            mine = reader.correct(task, str(r["item_id"]), text, r["finish"])
                            theirs = truth.get(str(r["request_id"]))
                            n += 1
                            where = {"model": key, "task": task, "reading": reading, "arm": arm, "stratum": stratum,
                                     "request_id": str(r["request_id"])}
                            if theirs is None:
                                disagree.append({**where, "why": "no scored row"})
                            elif bool(theirs) == mine:
                                agree += 1
                            else:
                                disagree.append({**where, "scored": int(theirs), "here": int(mine),
                                                 "text": text[-160:]})
                        cells[(key, task, reading, arm, stratum)] = (agree, n)
    return {"cells": cells, "disagree": disagree}


def panel_folders(root: Path, label: str, model: str) -> list[Path]:
    """The main panel's generation folders of one model for a scored table (PANEL_TABLES), found here from the
    collections' layout: GSM8K data/d2_out/collection/<m> (or collection_ext/<m>) and data/d2_out/forced/gsm8k/<m>;
    BIG-Bench Hard data/d2_out/bbh/collection/<m> and data/d2_out/forced/bbh/<m>; MATH-500 data/d2_out/math/collection/<m>
    and data/d2_out/math/forced/<m>; five-shot data/d2_out/fewshot/<task>/<m>; HotpotQA the folders that
    data/processed/hotpot/scored.json names for the model."""
    d = root / "data/d2_out"
    if label == "gsm8k":
        c = d / "collection" / model
        out = [c if c.is_dir() else d / "collection_ext" / model, d / "forced/gsm8k" / model]
    elif label == "bbh":
        out = [d / "bbh/collection" / model, d / "forced/bbh" / model]
    elif label == "math":
        out = [d / "math/collection" / model, d / "math/forced" / model]
    elif label.startswith("fewshot_"):
        out = [d / "fewshot" / label.split("_", 1)[1] / model]
    else:
        meta = json.loads((root / "data/processed/hotpot/scored.json").read_text(encoding="utf-8"))
        out = [root / f["path"] for k, f in meta["folders"].items() if k.split("/")[0] == model]
    return [f for f in out if f.is_dir()]


def panel_sample_check(root: Path, rule: StopRule, reader: Reader, n_sample: int = PANEL_SAMPLE_N, labels=None,
                       scored_hook=None) -> dict:
    """a seeded sample of main-panel replies (every model, benchmark and arm of PANEL_TABLES), drawn from
    the scored rows, each reply read here from its generation folder (the prefill of its arm in front; an arm that
    stops at the first "}" completed by the stop's rule as stated) and compared with the row's content_correct. Every
    disagreement is a mismatch, as in sample_check. Cells: ("panel/<model>", table label, "uncut", arm, "random") ->
    (agree, n)."""
    rng = random.Random(PANEL_SEED)
    cells, disagree = {}, []
    for label in (labels or PANEL_TABLES):
        task = label.split("_", 1)[1] if label.startswith("fewshot_") else label
        hq = task == "hotpot"
        cols = ["model_key", "arm", "content_correct"] + (["request_id"] if hq else ["config_id", "item_id"])
        sc = pd.read_parquet(root / PANEL_TABLES[label], columns=cols)
        if scored_hook is not None:
            sc = scored_hook(label, sc)
        for model in sorted(sc["model_key"].astype(str).unique()):
            mine_sc = sc[sc["model_key"].astype(str) == model]
            picks = {}
            for arm, g in mine_sc.groupby("arm"):
                idx = sorted(g.index)
                picks[arm] = [g.loc[j] for j in rng.sample(idx, min(n_sample, len(idx)))]
            raw = {}
            for folder in panel_folders(root, label, model):
                dec = decoding_of(folder)
                pre = dec.get("prefill") or {}
                rows = shard_rows(folder, list(picks))
                for r in rows.itertuples(index=False):
                    key = (str(r.request_id),) if hq else (str(r.arm), str(r.config_id), str(r.item_id))
                    raw[key] = (str(r.arm), str(r.item_id), r.output_text, r.finish_reason, pre.get(r.arm, ""),
                                stops_at_brace(dec, r.arm))
            for arm, pick in picks.items():
                n = agree = 0
                for r in pick:
                    key = (str(r["request_id"]),) if hq else (str(arm), str(r["config_id"]), str(r["item_id"]))
                    where = {"model": model, "table": label, "arm": arm, "key": "|".join(key)}
                    n += 1
                    if key not in raw:
                        disagree.append({**where, "why": "no raw reply"})
                        continue
                    _a, item, text, fin, pre, stops = raw[key]
                    text = pre + (text if isinstance(text, str) else "")
                    if stops and text.startswith(PRE):
                        text = rule.state(text, True)[1]
                    mine = reader.correct(task, item, text, fin)
                    if bool(int(r["content_correct"])) == mine:
                        agree += 1
                    else:
                        disagree.append({**where, "scored": int(r["content_correct"]), "here": int(mine),
                                         "text": text[-160:]})
                cells[(f"panel/{model}", label, "uncut", arm, "random")] = (agree, n)
    return {"cells": cells, "disagree": disagree}


def cell_failures(cells: dict) -> list[tuple]:
    """The cells whose disagreements exceed SAMPLE_MAX_DISAGREE, with their bound: (cell, agree, n, upper bound)."""
    return [(c, a, n, upper_bound(n - a, n)) for c, (a, n) in sorted(cells.items()) if n - a > SAMPLE_MAX_DISAGREE]


def full_reply_check(root: Path, rule: StopRule, reader: Reader, keys=(MISTRAL, DEEPSEEK), scored_hook=None) -> dict:
    """The API models' forced replies that the cut leaves open: scored content_correct against the box the model
    closed in its whole reply, read here. A whole box holding two options counts as not correct. scored_hook (tests):
    applied to each scored table before it is read."""
    res = {}
    for key in keys:
        c = Counter()
        ex = []
        for task in STOP_TASKS:
            raw = api_rows(root, key, task, FORCED_ARMS)
            sc = pd.read_parquet(root / SCORED_UNCUT[key].format(task=task),
                                 columns=["request_id", "content_correct"])
            if scored_hook is not None:
                sc = scored_hook(key, task, "uncut", sc)
            truth = dict(zip(sc["request_id"].astype(str), sc["content_correct"].astype(int)))
            for rid, item, cut, full in zip(raw["request_id"], raw["item_id"], raw["output_text"], raw["full"]):
                st, _ = rule.state(PRE + cut, True)
                if st in ("no_stop", "closed"):
                    continue
                c["replies"] += 1
                c[f"state_{st}"] += 1
                bx = Reader.boxes(PRE + full)
                if not (PRE + full).startswith(PRE) or not bx or not full_box_is_first(PRE + full, bx[0]):
                    c["full_not_closed"] += 1
                    continue
                # the rule's tab reading applies to the whole box as to the cut one: "<TAB>ext{(B)}" is "\text{(B)}"
                # with its escape "\t" written as a tab (without it, 9 DeepSeek boxes read as no letter)
                box = rule.tab.sub(lambda m: "\\t" + m.group(1), bx[0])
                two = rule.two_options(rule.words_of(PRE + box))
                here = False if two else reader.correct(task, str(item), PRE + box + "}", "stop")
                theirs = bool(truth.get(str(rid), 0))
                c["compared"] += 1
                c["agree"] += here == theirs
                c["false_gain"] += theirs and not here
                c["false_loss"] += here and not theirs
                # the replies the readers read (completed, inner box, a range the reader reads) and those
                # left unread (unfinished, other, a range it does not read) are counted and gated apart
                unread = st in ("unfinished", "other") or (
                    st == "range" and (task != "bbh" or reader.range_option(str(item), cut) is None))
                if unread:
                    c["unread_compared"] += 1
                    c["unread_false_loss"] += here and not theirs
                    c["unread_false_gain"] += theirs and not here
                else:
                    c["read_compared"] += 1
                    c["read_disagree"] += here != theirs
                    c[f"read_compared_{st}"] += 1              # per class too: each read class is gated on its own
                    c[f"read_disagree_{st}"] += here != theirs
                if here != theirs and len(ex) < 12:
                    ex.append({"task": task, "state": st, "cut": cut[-60:], "full_box": box[:80], "scored": theirs,
                               "here": here})
        res[key] = {"counts": dict(c), "examples": ex}
    return res


def full_box_is_first(full: str, box: str) -> bool:
    """Whether the first complete box found is the prefill's own box (it starts at the prefill)."""
    return full.startswith(PRE + box + "}")
