r"""HotpotQA (distractor setting): items, item sets, the length rule and the scorer.

Source (pinned)
---------------
The official development file of the distractor setting (Yang et al. 2018), `hotpot_dev_distractor_v1.json`, could
not be fetched: its host, curtis.ml.cmu.edu, did not resolve at collection time. The fallback is
the same set on Hugging Face: REPO, configuration "distractor", split validation, REVISION, FILE (27,452,575 bytes,
sha256 FILE_SHA256, equal to the Git LFS object id). 7,405 items, every one of level "hard"; columns id, question,
answer, type (bridge or comparison), level, supporting_facts {title, sent_id}, context {title, sentences}.

Item text (the bank's one item variable, {question})
----------------------------------------------------
`passage_block`: the item's paragraphs in the file's order, each as "Title: <title>", a newline, then its sentences
joined with the empty string (the sentences carry their own leading spaces) and stripped; one blank line between
paragraphs. `item_text`: the passage block, a blank line, then "Question: <question>". The bank's prompt format holds
the item text under its "Passages:" descriptor, so PromptSuite's format variation never touches the passages or the
question (the bank builder has one item placeholder). `demo_text`: the same with the two supporting paragraphs
only (the titles of supporting_facts, in the file's order), for the five-shot demonstrations.

Item sets (every choice reads metadata only, never a model output)
-----------------------------------------------------------------------
eligible     the gold answer, after the HotpotQA normalisation, is non-empty and neither "yes" nor "no", and the item
             text takes at most MAX_ITEM_TOKENS tokens under each of the pinned tokenizers of the four panel
             checkpoints whose native context is 4,096 (Llama-2-7B and 13B chat, Phi-3-mini-4k, Phi-3-medium-4k)
target       N_TARGET items drawn uniformly without replacement from the eligible ids in sorted order, with
             numpy.random.default_rng(seed_from("d22/hotpot/target")), kept in the order drawn; block A is the
             first N_TARGET / 2, block B the rest
pool         eligible items outside the target that share no supporting title with any target item, sorted
screening    N_SCREENING items drawn from the pool as the target is, with seed_from("d22/hotpot/screening"), in the
             order drawn (the admissibility screen's items)
demo pool    the rest of the pool whose demonstration (demo_text, a newline and the gold answer) takes at most
             MAX_DEMO_TOKENS tokens under the same four tokenizers and whose gold answer holds none of & % $ # _ ^ ~
             (characters a model may escape inside a box), sorted; the five-shot arm draws its demonstrations here
seed_from(label) is the first 8 bytes of sha256(label), big-endian, as run 1's demonstration seeds.

Scorer (SCORER_VERSION)
-----------------------
normalize_answer, exact_match and f1 port hotpot_evaluate_v1.py: lower case, punctuation (string.punctuation)
removed, the articles a, an and the removed, white space collapsed; F1 is the token F1, and a prediction or gold that
normalises to yes, no or noanswer scores 0 unless both normalise to the same string.
strict    the content of the last complete \boxed{...} (braces matched, a backslash escapes the next character),
          with the text wrappers \text, \textbf, \textit, \textrm, \mathrm and \emph replaced by their contents and
          every "$" removed; None without a complete box
lenient   the strict answer when there is one; else the content of a final \boxed{ that never closes; else the rest
          of the line after the last "answer is" or "answer:" (any case); else the last non-empty line
compliant the strict answer exists and normalises to a non-empty string
The primary score is strict exact match (binary, so every estimator runs unchanged); token F1 (0-1 per item, reported
as 100 x F1) and lenient scores sit beside it. An empty normalised answer is kept and scored as it is.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
FILE = "distractor/validation-00000-of-00001.parquet"
FILE_BYTES = 27452575
FILE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
LICENSE = "CC BY-SA 4.0 (HotpotQA; hotpotqa/hotpot_qa card: cc-by-sa-4.0)"
PLAN = "HotpotQA (third task)"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw", "hotpotqa")
RAW_PATH = os.path.join(RAW_DIR, "validation-00000-of-00001.parquet")
TABLE_DIR = os.path.join(ROOT, "data", "processed", "hotpot")
LENGTHS_PATH = os.path.join(TABLE_DIR, "lengths.parquet")
MANIFEST_PATH = os.path.join(ROOT, "manifests", "hotpot_items.json")
SCORER_MANIFEST_PATH = os.path.join(ROOT, "manifests", "hotpot_scorer.json")

N_TARGET = 1000
N_SCREENING = 100
MAX_ITEM_TOKENS = 1900
MAX_DEMO_TOKENS = 280
DEMO_EXCLUDED_CHARS = "&%$#_^~"
# The four panel checkpoints whose native context is 4,096 (src/math_d2.NATIVE_4K); their pinned tokenizers set the
# length rule.
LENGTH_MODELS = ("llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "phi-3-mini-4k-instruct", "phi-3-medium-4k-instruct")
SEED_TARGET = "d22/hotpot/target"
SEED_SCREENING = "d22/hotpot/screening"
SETS = ("target", "screening", "demo_pool")
TABLE_COLUMNS = ["item_id", "question", "gold_answer", "qtype", "block", "hotpot_question"]


class HotpotError(ValueError):
    """A HotpotQA input does not qualify."""


# ---------------------------------------------------------------------------------------
# Raw file and item text
# ---------------------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str = RAW_PATH, check: bool = True) -> pd.DataFrame:
    """The pinned file as a frame: item_id, question, answer, qtype, level, context, supporting_facts."""
    if check:
        if not os.path.isfile(path):
            raise HotpotError(f"{path} is absent (data/raw/hotpotqa/, see manifests/hotpot_source.json)")
        got = file_sha256(path)
        if got != FILE_SHA256:
            raise HotpotError(f"{path}: sha256 {got} is not the pinned {FILE_SHA256}")
    df = pd.read_parquet(path)
    need = {"id", "question", "answer", "type", "level", "supporting_facts", "context"}
    if not need <= set(df.columns):
        raise HotpotError(f"{path} lacks columns {sorted(need - set(df.columns))}")
    df = df.rename(columns={"id": "item_id", "type": "qtype"})
    if df["item_id"].duplicated().any():
        raise HotpotError("item ids are not unique")
    return df.reset_index(drop=True)


def _paragraphs(context) -> list[tuple[str, str]]:
    titles = list(context["title"])
    sents = list(context["sentences"])
    if len(titles) != len(sents):
        raise HotpotError("context titles and sentences differ in length")
    return [(str(t), "".join(str(s) for s in ss).strip()) for t, ss in zip(titles, sents)]


def passage_block(context, titles: Iterable[str] | None = None) -> str:
    """The paragraphs in the file's order (only those whose title is in `titles` when given)."""
    keep = None if titles is None else set(titles)
    parts = [f"Title: {t}\n{p}" for t, p in _paragraphs(context) if keep is None or t in keep]
    return "\n\n".join(parts)


def item_text(context, question: str) -> str:
    return f"{passage_block(context)}\n\nQuestion: {question}"


def supporting_titles(supporting_facts) -> list[str]:
    return list(dict.fromkeys(str(t) for t in supporting_facts["title"]))


def demo_text(context, supporting_facts, question: str) -> str:
    """The demonstration's item text: the supporting paragraphs only, then the question."""
    return f"{passage_block(context, supporting_titles(supporting_facts))}\n\nQuestion: {question}"


def seed_from(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def draw(ids: Sequence[str], n: int, label: str) -> list[str]:
    """n of the ids (sorted first), uniform without replacement, in the order drawn."""
    pool = sorted(ids)
    if len(pool) < n:
        raise HotpotError(f"{len(pool)} ids cannot fill {n} ({label})")
    idx = np.random.default_rng(seed_from(label)).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


# ---------------------------------------------------------------------------------------
# HotpotQA normalisation, exact match and F1 (port of hotpot_evaluate_v1.py)
# ---------------------------------------------------------------------------------------
_PUNCT = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction, gold) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction, gold) -> float:
    p, g = normalize_answer(prediction), normalize_answer(gold)
    special = ("yes", "no", "noanswer")
    if (p in special or g in special) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------------------
# Extraction from model outputs
# ---------------------------------------------------------------------------------------
SCORER_VERSION = "hotpot-boxed/1.0.0"
ANSWER_FORMAT = "End your reply with a final line that contains \\boxed{<answer>}."
_BOX_OPEN = re.compile(r"\\boxed\s*\{")
_WRAPPER = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|emph)\s*\{")
_ANSWER_CUE = re.compile(r"answer\s*(?:is\b\s*:?|:)", re.IGNORECASE)


def _close(t: str, start: int) -> int | None:
    """Index of the brace that closes the group opened just before `start`, or None."""
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


def boxes(text) -> tuple[list[str], str | None]:
    """Contents of the complete boxes in the order of their openings, and the content of a last box that never
    closes (None when every box closes)."""
    t = text if isinstance(text, str) else ""
    out, unclosed = [], None
    for m in _BOX_OPEN.finditer(t):
        j = _close(t, m.end())
        if j is None:
            unclosed = t[m.end():]
        else:
            out.append(t[m.end():j])
    # an unclosed box counts only when it opens after the last complete one
    if unclosed is not None and out:
        last_open = [m.end() for m in _BOX_OPEN.finditer(t)][-1]
        if _close(t, last_open) is not None:
            unclosed = None
    return out, unclosed


def unwrap(s: str) -> str:
    """Text wrappers replaced by their contents (repeatedly, innermost included); '$' removed; stripped."""
    t = s
    for _ in range(20):
        m = _WRAPPER.search(t)
        if not m:
            break
        j = _close(t, m.end())
        inner = t[m.end():] if j is None else t[m.end():j]
        rest = "" if j is None else t[j + 1:]
        t = t[:m.start()] + inner + rest
    return t.replace("$", "").strip()


def strict_extract(text) -> str | None:
    complete, _ = boxes(text)
    return unwrap(complete[-1]) if complete else None


def lenient_extract(text) -> str | None:
    s = strict_extract(text)
    if s is not None:
        return s
    t = text if isinstance(text, str) else ""
    _, unclosed = boxes(t)
    if unclosed is not None:
        return unwrap(unclosed)
    cues = list(_ANSWER_CUE.finditer(t))
    if cues:
        return unwrap(t[cues[-1].end():].split("\n", 1)[0])
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return unwrap(lines[-1]) if lines else None


def compliance(text) -> bool:
    s = strict_extract(text)
    return s is not None and normalize_answer(s) != ""


REQUIRED_COLUMNS = ("request_id", "item_id", "output_text")
SCORE_COLUMNS = ("strict_answer", "strict_correct", "lenient_answer", "lenient_correct", "compliant",
                 "strict_f1", "lenient_f1")


def score_outputs(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Add strict_answer, strict_correct (exact match), lenient_answer, lenient_correct, compliant, strict_f1 and
    lenient_f1 (0-1). Missing or non-string output text is the empty string; an absent answer scores 0."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["request_id"].duplicated().any():
        raise ValueError("request_id must be unique")
    if not {"item_id", "gold_answer"} <= set(items.columns) or items["item_id"].duplicated().any():
        raise ValueError("items need unique item_id and gold_answer")
    lookup = dict(zip(items["item_id"].astype(str), items["gold_answer"].astype(str)))
    ids = [str(i) for i in df["item_id"]]
    absent = sorted({i for i in ids if i not in lookup})
    if absent:
        raise KeyError(f"{len(absent)} item ids have no gold answer, e.g. {absent[:5]}")
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    cache: dict = {}
    rows = []
    for t, i in zip(texts, ids):
        key = (t, i)
        if key not in cache:
            g = lookup[i]
            s, le = strict_extract(t), lenient_extract(t)
            cache[key] = (s, 0 if s is None else exact_match(s, g), le, 0 if le is None else exact_match(le, g),
                          s is not None and normalize_answer(s) != "",
                          0.0 if s is None else f1(s, g), 0.0 if le is None else f1(le, g))
        rows.append(cache[key])
    out = df.copy()
    for k, name in enumerate(SCORE_COLUMNS):
        vals = [r[k] for r in rows]
        if name in ("strict_answer", "lenient_answer"):
            out[name] = pd.Series(vals, index=df.index, dtype=object)
        elif name in ("strict_correct", "lenient_correct"):
            out[name] = np.asarray(vals, dtype=np.int64)
        elif name == "compliant":
            out[name] = np.asarray(vals, dtype=bool)
        else:
            out[name] = np.asarray(vals, dtype=np.float64)
    return out


def scorer_rules() -> dict:
    return {
        "normalisation": "hotpot_evaluate_v1.py normalize_answer: lower case; characters of string.punctuation "
                         "removed; \\b(a|an|the)\\b replaced by a space; white space collapsed",
        "exact_match": "equal normalised strings",
        "f1": "token F1 of the normalised strings; 0 when either normalises to yes, no or noanswer and they differ; "
              "0 when no token is shared",
        "box_pattern": _BOX_OPEN.pattern, "wrapper_pattern": _WRAPPER.pattern, "answer_cue": _ANSWER_CUE.pattern,
        "strict": "content of the last complete box (braces matched, backslash escapes the next character), text "
                  "wrappers replaced by their contents, '$' removed, stripped",
        "lenient": "strict; else a final unclosed box's content; else the rest of the line after the last answer cue; "
                   "else the last non-empty line (each unwrapped as strict)",
        "compliance": "strict answer exists and normalises to a non-empty string",
        "primary": "strict exact match; strict F1 and lenient scores beside it",
    }


def scorer_record() -> dict:
    rules = scorer_rules()
    sha = hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"scorer_version": SCORER_VERSION, "plan": PLAN, "answer_format": ANSWER_FORMAT, "rules": rules,
            "rules_sha256": sha}


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def write_scorer_manifest(path: str = SCORER_MANIFEST_PATH) -> dict:
    rec = scorer_record()
    write_json(rec, path)
    return rec


# ---------------------------------------------------------------------------------------
# Item sets
# ---------------------------------------------------------------------------------------

def texts(raw: pd.DataFrame) -> pd.DataFrame:
    """item_id, item text, demonstration text (with its gold answer) and gold answer of every raw item."""
    return pd.DataFrame({
        "item_id": raw["item_id"].astype(str),
        "item_text": [item_text(c, q) for c, q in zip(raw["context"], raw["question"])],
        "demo_text": [demo_text(c, s, q) + "\n" + str(a)
                      for c, s, q, a in zip(raw["context"], raw["supporting_facts"], raw["question"], raw["answer"])],
        "gold_answer": raw["answer"].astype(str)})


def split_items(raw: pd.DataFrame, lengths: pd.DataFrame) -> dict:
    """The item sets of the module doc. `lengths` has item_id, item_tokens and demo_tokens (each the maximum over
    LENGTH_MODELS)."""
    L = lengths.set_index("item_id")
    ids = raw["item_id"].astype(str).tolist()
    if set(L.index) != set(ids):
        raise HotpotError("the lengths table does not cover exactly the raw items")
    norm = raw["answer"].map(normalize_answer)
    ok_gold = ~norm.isin(["", "yes", "no"])
    ok_len = raw["item_id"].map(L["item_tokens"]) <= MAX_ITEM_TOKENS
    elig = raw[ok_gold.to_numpy() & ok_len.to_numpy()]
    target = draw(elig["item_id"].tolist(), N_TARGET, SEED_TARGET)
    tset = set(target)
    sup = {str(i): set(supporting_titles(s)) for i, s in zip(raw["item_id"], raw["supporting_facts"])}
    t_titles = set().union(*(sup[i] for i in target))
    pool = sorted(i for i in elig["item_id"] if i not in tset and not (sup[i] & t_titles))
    screening = draw(pool, N_SCREENING, SEED_SCREENING)
    sset = set(screening)
    gold = dict(zip(raw["item_id"].astype(str), raw["answer"].astype(str)))
    demo_pool = [i for i in pool if i not in sset and int(L.loc[i, "demo_tokens"]) <= MAX_DEMO_TOKENS
                 and not any(ch in gold[i] for ch in DEMO_EXCLUDED_CHARS)]
    counts = {"raw": int(len(raw)), "gold_yes_no_or_empty": int((~ok_gold).sum()),
              "gold_ok": int(ok_gold.sum()), "over_length": int((ok_gold.to_numpy() & ~ok_len.to_numpy()).sum()),
              "eligible": int(len(elig)), "eligible_share_of_gold_ok": round(len(elig) / max(1, int(ok_gold.sum())), 4),
              "eligible_by_type": {k: int(v) for k, v in elig["qtype"].value_counts().sort_index().items()},
              "pool": len(pool), "eligible_sharing_a_target_title": int(len(elig) - N_TARGET - len(pool)),
              "screening": len(screening), "demo_pool": len(demo_pool)}
    return {"target": target, "screening": screening, "demo_pool": demo_pool, "counts": counts}


def tables(raw: pd.DataFrame, sets: dict) -> dict[str, pd.DataFrame]:
    """One table per set (TABLE_COLUMNS): question is the item variable (item_text; demo_text for the demo pool,
    without the gold answer), hotpot_question the dataset's question."""
    by = raw.set_index(raw["item_id"].astype(str))
    out = {}
    for name in SETS:
        ids = sets[name]
        rows = by.loc[ids]
        if name == "demo_pool":
            q = [demo_text(c, s, qq) for c, s, qq in zip(rows["context"], rows["supporting_facts"], rows["question"])]
        else:
            q = [item_text(c, qq) for c, qq in zip(rows["context"], rows["question"])]
        block = (["A"] * (N_TARGET // 2) + ["B"] * (N_TARGET - N_TARGET // 2)) if name == "target" else [""] * len(ids)
        out[name] = pd.DataFrame({"item_id": list(ids), "question": q, "gold_answer": rows["answer"].astype(str).tolist(),
                                  "qtype": rows["qtype"].astype(str).tolist(), "block": block,
                                  "hotpot_question": rows["question"].astype(str).tolist()})[TABLE_COLUMNS]
    return out


def canonical_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of [[item_id, question, gold_answer], ...] in row order (as the other tasks)."""
    rows = [[i, q, g] for i, q, g in zip(df["item_id"], df["question"], df["gold_answer"])]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def table_bytes(df: pd.DataFrame) -> bytes:
    t = pa.Table.from_pandas(df[TABLE_COLUMNS].reset_index(drop=True), preserve_index=False)
    buf = pa.BufferOutputStream()
    pq.write_table(t, buf, compression="zstd")
    return buf.getvalue().to_pybytes()


def forced_cap(target_ids: Sequence[str], gold_tokens: pd.DataFrame) -> dict:
    """The forced arms' cap: the smallest multiple of 16 that holds the longest target gold answer written as
    '\\text{<gold>}}' (wrapper, answer, the wrapper's and the box's closing braces) under every panel tokenizer.
    gold_tokens: item_id, gold_tokens (the maximum over the panel's tokenizers)."""
    g = gold_tokens.set_index("item_id")["gold_tokens"]
    longest = int(max(int(g[i]) for i in target_ids))
    return {"longest_target_answer_tokens": longest, "cap": int(-(-longest // 16) * 16)}


# ---------------------------------------------------------------------------------------
# Content reading
# ---------------------------------------------------------------------------------------
# strict_extract, lenient_extract, compliance and SCORER_VERSION above are unchanged: the content reading is a third
# reading beside them, with its own version and rules record (manifests/hotpot_content_scorer.json). It follows
# src/content_scoring.py: the instruction's placeholder "<answer>" copied into a box is not an answer.
#
# content   the content of the last box that is not a placeholder box, taking the complete \boxed{...} and the boxes
#           written with other brackets (src/content_scoring.OTHER_BOX: "\boxed<Odysseus>", "<boxed>Liguria</boxed>";
#           sources "angle_box" and "tag_box") in the order of their openings; with no such box, the lenient
#           reading of the text with every placeholder box and every brace-less placeholder copy ("\boxed<answer>")
#           removed, in lenient's order: a final unclosed box when there is no complete box at all; else the rest of
#           the line after the last "answer is" / "answer:" cue; else the last non-empty line, where a box written
#           without its backslash ("boxed{...}") is read as a box (its first complete one) unless it is a placeholder;
#           each cleaned (a leading copied "<answer>", with an optional "-" or ":", removed; then a leading answer cue,
#           src/content_scoring.strip_answer_prefix; then angle brackets around the whole content removed, and the cue
#           once more). None for an empty output.
# placeholder box  the one rule of src/content_scoring.is_placeholder (empty, a slot name bare or in angle
#           brackets, "<Option X>"), or a content that, unwrapped and cleaned, normalises to "" or "answer".
# Versions: 1.0.0 skipped boxes normalising to "" or "answer" and removed only "boxed{<answer>}" from the text.
#           1.1.0 removes every placeholder box and the brace-less copies, so "Holston River boxed<answer>" reads
#           "Holston River".
#           1.2.0 reads a box written with angle brackets in place of its braces, or as a tag, as a box holding its
#           content (a placeholder in that form is skipped and removed, as above): "boxed<Odysseus>" reads
#           "Odysseus".
#           1.3.0 removes a leading answer cue from what it reads, by the rule of src/content_scoring.strip_answer_prefix
#           (content/1.4.0) used on every benchmark: "boxed{Answer: Leon Uris}" reads "Leon Uris".
#           1.4.0 reads LaTeX spacing commands (a backslash before a blank, \, \; \: \!, \quad, \qquad, "~") as blanks
#           before cleaning, by src/content_scoring.spacing_off (content/1.5.0): "Leon\,Uris" reads "Leon Uris". With
#           content/1.5.0's placeholder rule, an option named with its letter ("<Option A>") is not a placeholder box.
#           1.5.0 reads an answer tag pair "<answer>X</answer>" (X on one line, holding no "boxed";
#           src/content_scoring.ANSWER_TAG, content/1.6.0) as a box holding X (source "answer_tag"), taken with the
#           other boxes in the order of their openings, a placeholder pair skipped and removed. It removes the pair
#           around a whole content or a closing "</answer>" that ends it: "<answer>Veyyil</answer>" reads "Veyyil". It
#           also reads typographic quotes, apostrophes and dashes as ASCII in the answer and in the gold it is compared
#           with (typographic_ascii): "“N.I.B.”" matches "N.I.B.", because the normalisation strips ASCII
#           punctuation only.
CONTENT_SCORER_VERSION = "hotpot-content/1.5.0"
PLACEHOLDER = "<answer>"
_LEAD_PLACEHOLDER = re.compile(r"^\s*<answer>\s*[-:]?\s*", re.IGNORECASE)
_BARE_BOX = re.compile(r"(?<!\\)boxed\s*\{")
CONTENT_SOURCES = ("box", "angle_box", "tag_box", "answer_tag", "unclosed_box", "answer_cue", "bare_box", "last_line",
                   "empty")
# hotpot-content/1.5.0: typographic quotes, apostrophes and dashes, read as their ASCII forms in the answer
# read and in the gold it is compared with (the normalisation strips ASCII punctuation only)
TYPOGRAPHIC = {**dict.fromkeys(map(ord, "“”„‟″"), '"'),
               **dict.fromkeys(map(ord, "‘’‚‛′"), "'"),
               **dict.fromkeys(map(ord, "‐‑‒–—―−﹣－"), "-")}
CONTENT_COLUMNS = ("content_answer", "content_correct", "content_f1", "content_source")


def typographic_ascii(s):
    """hotpot-content/1.5.0: typographic quotes, apostrophes and dashes as '"', "'" and "-" (TYPOGRAPHIC)."""
    return s.translate(TYPOGRAPHIC) if isinstance(s, str) else s


def content_clean(u: str) -> str:
    """LaTeX spacing commands read as blanks (src/content_scoring.spacing_off, hotpot-content/1.4.0: "Leon\\,Uris"
    reads "Leon Uris"), a leading copied "<answer>" (with an optional "-" or ":") removed, then a leading answer cue
    (src/content_scoring.strip_answer_prefix, hotpot-content/1.3.0: "Answer: Leon Uris", "(Answer: Mistress of the
    Robes)"), then angle brackets around the whole, then the cue once more ("<Answer: X>"). hotpot-content/1.5.0:
    typographic quotes, apostrophes and dashes read as ASCII first (typographic_ascii), and an answer tag pair around
    the whole ("<answer>X</answer>") or a closing "</answer>" at the end removed (src/content_scoring.ANSWER_TAG_WHOLE,
    TRAILING_TAG)."""
    from . import content_scoring as cs
    u = cs.spacing_off(typographic_ascii(u)).strip()
    m = cs.ANSWER_TAG_WHOLE.match(u)
    if m:
        u = m.group(1).strip()
    u = cs.strip_answer_prefix(_LEAD_PLACEHOLDER.sub("", cs.TRAILING_TAG.sub("", u)).strip())
    if len(u) > 1 and u[0] == "<" and u[-1] == ">":
        u = cs.strip_answer_prefix(u[1:-1].strip())
    return u


def is_placeholder_box(content: str) -> bool:
    """Whether a box's content is a placeholder (the rule of src/content_scoring.is_placeholder, or a content that
    normalises to "" or "answer")."""
    from . import content_scoring as cs
    return cs.is_placeholder(content) or normalize_answer(content_clean(unwrap(content))) in ("", "answer")


def drop_placeholder_boxes(t: str) -> str:
    """The text with every complete placeholder box and every brace-less placeholder copy replaced by a space."""
    from . import content_scoring as cs
    for rx in (_BOX_OPEN, _BARE_BOX):          # boxes, then boxes written without their backslash
        parts, j = [], 0
        for m in rx.finditer(t):
            if m.start() < j:
                continue
            k = _close(t, m.end())
            if k is not None and is_placeholder_box(t[m.end():k]):
                parts.extend([t[j:m.start()], " "])
                j = k + 1
        parts.append(t[j:])
        t = "".join(parts)
    t = cs.PLACEHOLDER_COPY.sub(lambda x: " " if cs.is_placeholder(x.group(1)) else x.group(0), t)
    if "answer" in t.lower():      # hotpot-content/1.5.0: an answer tag pair holding a placeholder, or empty
        t = cs.EMPTY_TAG.sub(" ", cs.ANSWER_TAG.sub(lambda x: " " if is_placeholder_box(x.group(1)) else x.group(0), t))
    if "boxed" not in t.lower():
        return t
    # hotpot-content/1.2.0: a placeholder written with other brackets ("<boxed><answer></boxed>", "\boxed<>")
    return cs.OTHER_BOX.sub(lambda x: " " if other_box_placeholder(x.group(1) if x.group(1) is not None else x.group(2))
                            else x.group(0), t)


def other_box_placeholder(content: str) -> bool:
    """A box written with other brackets is a placeholder when its content X, or X in angle brackets, is one
    (is_placeholder_box)."""
    c = content if isinstance(content, str) else ""
    return is_placeholder_box(c) or is_placeholder_box("<" + c.strip() + ">")


def box_spans(t: str) -> list[tuple[int, int, str]]:
    """(start, end, content) of every complete \\boxed{...}, in the order of their openings (the boxes of `boxes`)."""
    out = []
    for m in _BOX_OPEN.finditer(t):
        j = _close(t, m.end())
        if j is not None:
            out.append((m.start(), j + 1, t[m.end():j]))
    return out


def content_read(text) -> tuple[str | None, str]:
    """(answer, source) of the content reading; source is one of CONTENT_SOURCES."""
    from . import content_scoring as cs
    t = text if isinstance(text, str) else ""
    complete, unclosed = boxes(t)
    others = cs.other_boxes(t, [(a, b) for a, b, _ in box_spans(t)]) if "boxed" in t.lower() else []
    tags = cs.answer_tags(t, [(a, b) for a, b, _ in box_spans(t)]) if "answer" in t.lower() else []
    if others or tags:
        # hotpot-content/1.2.0: boxes written with other brackets join the complete boxes, in the order of openings;
        # hotpot-content/1.5.0: answer tag pairs too ("<answer>Veyyil</answer>", source "answer_tag")
        items = [(a, c, "box") for a, _, c in box_spans(t) if not is_placeholder_box(c)]
        items += [(a, c, f"{form}_box") for a, _, c, form in others if not other_box_placeholder(c)]
        items += [(a, c, "answer_tag") for a, _, c in tags if not is_placeholder_box(c)]
        if items:
            _, c, src = max(items, key=lambda x: x[0])
            return content_clean(unwrap(c)), src
    else:
        for c in reversed(complete):
            if not is_placeholder_box(c):
                return content_clean(unwrap(c)), "box"
    t2 = drop_placeholder_boxes(t)
    if unclosed is not None and not complete:
        return content_clean(unwrap(unclosed)), "unclosed_box"
    cues = list(_ANSWER_CUE.finditer(t2))
    if cues:
        return content_clean(unwrap(t2[cues[-1].end():].split("\n", 1)[0])), "answer_cue"
    lines = [ln.strip() for ln in t2.split("\n") if ln.strip()]
    if not lines:
        return None, "empty"
    last = lines[-1]
    m = _BARE_BOX.search(last)
    if m:
        j = _close(last, m.end())
        if j is not None:
            return content_clean(unwrap(last[m.end():j])), "bare_box"
    return content_clean(unwrap(last)), "last_line"


def content_extract(text) -> str | None:
    return content_read(text)[0]


def score_content(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Add content_answer, content_correct (exact match, int64), content_f1 (0-1) and content_source. Missing or
    non-string output text is the empty string; an absent answer scores 0 (as score_outputs)."""
    for c in ("item_id", "output_text"):
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")
    if not {"item_id", "gold_answer"} <= set(items.columns) or items["item_id"].duplicated().any():
        raise ValueError("items need unique item_id and gold_answer")
    lookup = dict(zip(items["item_id"].astype(str), items["gold_answer"].astype(str)))
    ids = [str(i) for i in df["item_id"]]
    absent = sorted({i for i in ids if i not in lookup})
    if absent:
        raise KeyError(f"{len(absent)} item ids have no gold answer, e.g. {absent[:5]}")
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    cache: dict = {}
    rows = []
    for t, i in zip(texts, ids):
        key = (t, i)
        if key not in cache:
            g = typographic_ascii(lookup[i])     # hotpot-content/1.5.0: as the answer read (content_clean)
            a, src = content_read(t)
            cache[key] = (a, 0 if a is None else exact_match(a, g), 0.0 if a is None else f1(a, g), src)
        rows.append(cache[key])
    out = df.copy()
    out["content_answer"] = pd.Series([r[0] for r in rows], index=df.index, dtype=object)
    out["content_correct"] = np.asarray([r[1] for r in rows], dtype=np.int64)
    out["content_f1"] = np.asarray([r[2] for r in rows], dtype=np.float64)
    out["content_source"] = pd.Series([r[3] for r in rows], index=df.index, dtype=object)
    return out


def content_scorer_rules() -> dict:
    return {
        "normalisation": scorer_rules()["normalisation"], "exact_match": "equal normalised strings",
        "f1": scorer_rules()["f1"], "box_pattern": _BOX_OPEN.pattern, "wrapper_pattern": _WRAPPER.pattern,
        "answer_cue": _ANSWER_CUE.pattern, "lead_placeholder": _LEAD_PLACEHOLDER.pattern,
        "bare_box_pattern": _BARE_BOX.pattern, "placeholder_copy_pattern": _cs_rules()["placeholder_copy"],
        "placeholder": "src/content_scoring.is_placeholder (" + _cs_rules()["version"] + "): " + _cs_rules()["rule"]
                       + "; or a content that, unwrapped and cleaned, normalises to '' or 'answer'",
        "spacing": _cs_spacing_rules()["rule"], "spacing_pattern": _cs_spacing_rules()["spacing"],
        "typographic": "typographic quotes, apostrophes and dashes read as '\"', \"'\" and '-' in the answer read and in "
                       "the gold it is compared with (hotpot-content/1.5.0)",
        "answer_tag": _cs_tag_rules()["rule"].split(" On temporal_sequences")[0],
        "answer_tag_pattern": _cs_tag_rules()["answer_tag"],
        "clean": "typographic punctuation as ASCII and LaTeX spacing read as blanks, then an answer tag pair around the "
                 "whole content or a closing '</answer>' at its end removed, then a leading copied '<answer>' "
                 "(optional '-' or ':') removed, then a leading answer cue "
                 "(src/content_scoring.strip_answer_prefix), then angle brackets around the whole content, then the "
                 "cue once more",
        "answer_prefix": _cs_cue_rules()["answer_prefix"],
        "other_box": _cs_other_rules()["rule"], "other_box_pattern": _cs_other_rules()["pattern"],
        "content": "the last box that is not a placeholder, complete boxes, boxes written with other brackets and answer "
                   "tag pairs taken in the order of their openings; else, on the text with every placeholder box "
                   "(with or without its backslash) and every brace-less placeholder copy removed: a final unclosed box "
                   "when there is no complete box at all; else the rest of the line after the last answer cue; else "
                   "the last non-empty line, read inside its first complete backslash-less box when it holds one; each "
                   "unwrapped and cleaned; None for an empty output",
        "unchanged": f"strict, lenient and compliance stay those of {SCORER_VERSION}",
    }


def _cs_rules() -> dict:
    from . import content_scoring as cs
    return cs.placeholder_rules()


def _cs_cue_rules() -> dict:
    from . import content_scoring as cs
    return cs.answer_cue_rules()


def _cs_tag_rules() -> dict:
    from . import content_scoring as cs
    return cs.answer_tag_rules()


def _cs_spacing_rules() -> dict:
    from . import content_scoring as cs
    return cs.spacing_rules()


def _cs_other_rules() -> dict:
    from . import content_scoring as cs
    return cs.other_box_rules()


def content_scorer_record() -> dict:
    rules = content_scorer_rules()
    sha = hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"scorer_version": CONTENT_SCORER_VERSION, "beside": SCORER_VERSION,
            "plan": "HotpotQA, content reading (22 November 2026)",
            "answer_format": ANSWER_FORMAT, "placeholder": PLACEHOLDER, "rules": rules, "rules_sha256": sha}
