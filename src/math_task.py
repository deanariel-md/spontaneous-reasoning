r"""MATH-500 with numeric answers: items, item sets and the scorer.

Sources (pinned)
----------------
Target items: HuggingFaceH4/MATH-500 at REV_TARGET, test.jsonl (the 500-problem test subset of Hendrycks et al.'s
MATH chosen in OpenAI's PRM800K "Let's Verify Step by Step" split). Training items for screening and the tranche:
EleutherAI/hendrycks_math at REV_TRAIN, the seven subjects' train-00000-of-00001.parquet files (the 7,500 MATH
training problems; MATH's own Hugging Face repository, hendrycks/competition_math, is disabled). MATH-500 is drawn
from MATH's test split, so no training problem is a target problem.

Numeric rule (gold strings only, never a model output)
------------------------------------------------------
An item is kept when its gold answer normalizes to one rational number (`normalize_answer`). The target gold is
MATH-500's `answer` field; a training item's gold is the content of the last complete \boxed{...} (or \fbox{...})
in its reference solution, and a training item without one is dropped. `normalize_answer`, used for gold and for
model outputs alike:
  1. scoring.normalize_text, surrounding whitespace removed; enclosing math delimiters ($...$, $$...$$, \(...\),
     \[...\]) removed; \displaystyle removed; the LaTeX dollar \$ read as $;
  2. decorations removed: degrees (^\circ, ^{\circ}, the degree sign), a trailing unit in \text{...}, \textrm{...},
     \mathrm{...} or \mbox{...} holding letters, spaces and periods only (optionally followed by ^2 or ^3), a trailing
     percent sign (% or \%), LaTeX spacing commands (\, \! \; \: "\ " and ~), the brace comma {,}, one leading "$"
     (after or before the sign), one trailing period; then all whitespace;
  3. the rest must fullmatch one of: an integer with optional sign and valid comma groups ("1,234"); a decimal
     ("-2.5", ".5", "1,234.5"); a fraction \frac, \dfrac or \tfrac with integer numerator and denominator in braces or
     as single digits ("\frac{14}{3}", "\frac43", "\frac9{19}", "-\frac{1}{2}", "\frac{-1}{2}"); a slash fraction
     ("14/3"). A zero denominator, two signs, mixed numbers, radicals, pi, variables, tuples, intervals and anything
     else return None;
  4. the canonical value is the reduced fraction as "p/q", or "p" when q = 1 ("0.5" -> "1/2", "\frac{6}{4}" -> "3/2",
     "1,000" -> "1000", "-0" -> "0"). Equality of answers is equality of canonical strings, so it is exact rational
     equality: "0.333" differs from "1/3".

Item sets
---------
target: every MATH-500 item whose gold is numeric, in file order; item id = MATH-500's unique_id
("test/<subject>/<n>.json"). Training: every numeric training item, item id "train/<subject>/<row>" (row = zero-based
position in the subject's pinned parquet file), subjects in SUBJECTS order. One permutation of the training items,
numpy.random.default_rng(SeedSequence([20261003, w("math-items"), w("train"), w(n)])) via src.sampling.rng_for
(n = number of numeric training items), fills the screening set (the first 200) and the calibration set (the next
n_target, the size of the target set), each stored in training order. The permutation reads ids only.

Scorer (SCORER_VERSION)
-----------------------
strict: normalize_answer of the content of the last complete \boxed{...} (bbh.boxed_contents: braces matched, an
escaped brace literal, boxes ordered by their opening); None without a box or when the content is not a number.
lenient: normalize_answer of the last number expression anywhere in the output (\frac forms, slash fractions,
decimals and integers with valid comma groups, LaTeX thin-space thousands read as commas); a token may not start
right after a word character. compliance: strict extraction succeeds.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import bbh, sampling, scoring as sc

REPO_TARGET = "HuggingFaceH4/MATH-500"
REV_TARGET = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
REPO_TRAIN = "EleutherAI/hendrycks_math"
REV_TRAIN = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus")
TARGET_FILE = "test.jsonl"
TRAIN_FILE = "{subject}/train-00000-of-00001.parquet"
LICENSE = {"MATH": "MIT (Hendrycks et al. 2021, github.com/hendrycks/math; EleutherAI/hendrycks_math card: mit)",
           "MATH-500": "no licence field on the HuggingFaceH4/MATH-500 card; the PRM800K split it copies is "
                       "released under MIT (github.com/openai/prm800k)"}
PLAN = "MATH-500 task"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw", "math")
RAW_TARGET = os.path.join(RAW, "MATH-500")
RAW_TRAIN = os.path.join(RAW, "hendrycks_math")
MANIFEST_PATH = os.path.join(ROOT, "manifests", "math_items.json")
SCORER_MANIFEST_PATH = os.path.join(ROOT, "manifests", "math_scorer.json")
TABLE_DIR = os.path.join(ROOT, "data", "processed", "math")

SEED_LABEL = "math-items"
N_SCREENING = 200
SETS = ("target", "calibration", "screening")
TABLE_COLUMNS = ["item_id", "question", "gold_answer", "subject", "level"]


class MathError(ValueError):
    """A MATH input does not qualify."""


# ---------------------------------------------------------------------------------------
# Pinned files
# ---------------------------------------------------------------------------------------
def target_path() -> str:
    return os.path.join(RAW_TARGET, TARGET_FILE)


def train_path(subject: str) -> str:
    return os.path.join(RAW_TRAIN, *TRAIN_FILE.format(subject=subject).split("/"))


def download(log=print) -> list[str]:
    """Fetch the pinned files that are absent (huggingface_hub; the CLI login supplies any token)."""
    from huggingface_hub import hf_hub_download
    got = []
    if not os.path.isfile(target_path()):
        got.append(hf_hub_download(REPO_TARGET, TARGET_FILE, repo_type="dataset", revision=REV_TARGET,
                                   local_dir=RAW_TARGET))
    for s in SUBJECTS:
        if not os.path.isfile(train_path(s)):
            got.append(hf_hub_download(REPO_TRAIN, TRAIN_FILE.format(subject=s), repo_type="dataset",
                                       revision=REV_TRAIN, local_dir=RAW_TRAIN))
    for g in got:
        log(f"downloaded {g}")
    return got


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------
_WRAPPERS = (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
_DEGREES = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\u00b0")
_UNIT = re.compile(r"\\(?:text|textrm|mathrm|mbox)\s*\{[A-Za-z .]*\}(?:\s*\^\s*\{?\s*[23]\s*\}?)?\s*$")
_PERCENT = re.compile(r"\\?%\s*$")
_SPACING = re.compile(r"\\[,!;:]|\\ |~")
_DOLLAR = re.compile(r"^([-+]?)\s*\$\s*([-+]?)")
_INTP = r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+"
_DEC = re.compile(rf"(?P<s>[-+]?)(?P<n>(?:{_INTP})(?:\.[0-9]+)?|\.[0-9]+)")
_FRAC = re.compile(r"(?P<s>[-+]?)\\[dt]?frac(?:\{(?P<s1>[-+]?)(?P<a>[0-9]+)\}|(?P<a1>[0-9]))"
                   r"(?:\{(?P<s2>[-+]?)(?P<b>[0-9]+)\}|(?P<b1>[0-9]))")
_SLASH = re.compile(r"(?P<s>[-+]?)(?P<a>[0-9]+)/(?P<b>[0-9]+)")


def _strip_wrappers(t: str) -> str:
    changed = True
    while changed:
        changed = False
        t = t.strip()
        for a, b in _WRAPPERS:
            if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
                t = t[len(a):len(t) - len(b)]
                changed = True
                break
    return t


def _canonical(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _sign(*signs) -> int | None:
    """-1 or 1 for the product of the signs; None when more than one sign is written."""
    written = [s for s in signs if s]
    if len(written) > 1:
        return None
    return -1 if written and written[0] == "-" else 1


def normalize_answer(s) -> str | None:
    """Canonical rational number of an answer string, or None (module doc, "Numeric rule"). A number of any length
    is read exactly: Python's limit on integer conversion (4,300 digits) is lifted for the call (scorer 1.0.2)."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        return _normalize_answer(s)
    finally:
        sys.set_int_max_str_digits(old)


def _normalize_answer(s) -> str | None:
    if s is None:
        return None
    t = sc.normalize_text(s).strip()
    t = _strip_wrappers(t).replace("\\displaystyle", "").replace("\\$", "$")
    t = _DEGREES.sub("", t)
    t = _UNIT.sub("", t.strip())
    t = _PERCENT.sub("", t.strip())
    t = _SPACING.sub("", t).replace("{,}", ",").strip()
    m = _DOLLAR.match(t)
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
    d = _DEC.fullmatch(t)
    if d:
        sign = -1 if d.group("s") == "-" else 1
        return _canonical(sign * Fraction(d.group("n").replace(",", "")))
    f = _FRAC.fullmatch(t)
    if f:
        sign = _sign(f.group("s"), f.group("s1"), f.group("s2"))
        a, b = int(f.group("a") or f.group("a1")), int(f.group("b") or f.group("b1"))
        if sign is None or b == 0:
            return None
        return _canonical(sign * Fraction(a, b))
    f = _SLASH.fullmatch(t)
    if f:
        b = int(f.group("b"))
        if b == 0:
            return None
        return _canonical((-1 if f.group("s") == "-" else 1) * Fraction(int(f.group("a")), b))
    return None


_SOLUTION_BOX = re.compile(r"\\(?:boxed|fbox)\{")


def solution_answer(solution) -> str | None:
    """Content of the last complete \\boxed{...} or \\fbox{...} of a reference solution, or None."""
    t = sc.normalize_text(solution)
    last = None
    for m in _SOLUTION_BOX.finditer(t):
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
                    last = t[m.end():i]
                    break
            i += 1
    return last


# ---------------------------------------------------------------------------------------
# Items and item sets
# ---------------------------------------------------------------------------------------
def load_target() -> pd.DataFrame:
    """MATH-500 in file order: item_id (unique_id), question (problem), gold_text, gold_answer (None when the gold
    is not numeric), subject, level."""
    p = target_path()
    if not os.path.isfile(p):
        raise MathError(f"{p} is absent; fetch it with `.\\.venv\\Scripts\\python.exe -I scripts\\fetch_math.py`")
    df = pd.read_json(p, lines=True, dtype=False)
    need = {"problem", "answer", "subject", "level", "unique_id"}
    if not need <= set(df.columns) or len(df) != 500:
        raise MathError(f"{p}: expected 500 rows with {sorted(need)}")
    out = pd.DataFrame({"item_id": df["unique_id"].astype(str), "question": df["problem"].astype(str),
                        "gold_text": df["answer"].astype(str), "subject": df["subject"].astype(str),
                        "level": df["level"].astype(str)})
    out["gold_answer"] = [normalize_answer(a) for a in out["gold_text"]]
    if out["item_id"].duplicated().any():
        raise MathError("MATH-500 unique_id is not unique")
    return out


def load_train() -> pd.DataFrame:
    """The 7,500 MATH training problems: item_id "train/<subject>/<row>", question, gold_text (the solution's last
    box, None without one), gold_answer, subject (the type field), level; subjects in SUBJECTS order."""
    parts = []
    for s in SUBJECTS:
        p = train_path(s)
        if not os.path.isfile(p):
            raise MathError(f"{p} is absent; fetch it with `.\\.venv\\Scripts\\python.exe -I scripts\\fetch_math.py`")
        df = pd.read_parquet(p)
        if not {"problem", "solution", "type", "level"} <= set(df.columns):
            raise MathError(f"{p}: unexpected columns {list(df.columns)}")
        gold = [solution_answer(x) for x in df["solution"]]
        parts.append(pd.DataFrame({"item_id": [f"train/{s}/{i}" for i in range(len(df))],
                                   "question": df["problem"].astype(str), "gold_text": gold,
                                   "subject": df["type"].astype(str), "level": df["level"].astype(str)}))
    out = pd.concat(parts, ignore_index=True)
    out["gold_answer"] = [normalize_answer(a) for a in out["gold_text"]]
    return out


def train_permutation(n: int) -> np.ndarray:
    return sampling.rng_for(SEED_LABEL, "train", n).permutation(n)


def split_items(target: pd.DataFrame, train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three item sets (module doc): numeric items only; screening and calibration from one permutation of the
    numeric training items, each stored in training order."""
    tgt = target[target["gold_answer"].notna()].reset_index(drop=True)
    tr = train[train["gold_answer"].notna()].reset_index(drop=True)
    n_cal = len(tgt)
    if len(tr) < N_SCREENING + n_cal:
        raise MathError(f"{len(tr)} numeric training items cannot fill {N_SCREENING} screening and {n_cal} "
                        "calibration items")
    perm = train_permutation(len(tr))
    scr = np.sort(perm[:N_SCREENING])
    cal = np.sort(perm[N_SCREENING:N_SCREENING + n_cal])
    return {"target": tgt, "calibration": tr.iloc[cal].reset_index(drop=True),
            "screening": tr.iloc[scr].reset_index(drop=True)}


def canonical_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of the canonical list [[item_id, question, gold_answer], ...] in row order (as src.bbh)."""
    rows = [[i, q, g] for i, q, g in zip(df["item_id"], df["question"], df["gold_answer"])]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def table_bytes(df: pd.DataFrame) -> bytes:
    """Parquet bytes of an item table (pyarrow, zstd, no index); the manifest's sha256 is over these bytes."""
    t = pa.Table.from_pandas(df[TABLE_COLUMNS].reset_index(drop=True), preserve_index=False)
    buf = pa.BufferOutputStream()
    pq.write_table(t, buf, compression="zstd")
    return buf.getvalue().to_pybytes()


def build_manifest() -> tuple[dict, dict[str, pd.DataFrame]]:
    target, train = load_target(), load_train()
    sets = split_items(target, train)
    tables = {k: v[TABLE_COLUMNS].copy() for k, v in sets.items()}
    dropped_target = target[target["gold_answer"].isna()]
    files = {f"{REPO_TARGET}/{TARGET_FILE}": {"revision": REV_TARGET, "sha256": file_sha256(target_path()),
                                              "bytes": os.path.getsize(target_path())}}
    for s in SUBJECTS:
        files[f"{REPO_TRAIN}/{TRAIN_FILE.format(subject=s)}"] = {
            "revision": REV_TRAIN, "sha256": file_sha256(train_path(s)), "bytes": os.path.getsize(train_path(s))}
    manifest = {
        "task": "math", "plan": PLAN, "target_dataset": REPO_TARGET, "target_revision": REV_TARGET,
        "train_dataset": REPO_TRAIN, "train_revision": REV_TRAIN, "license": LICENSE, "subjects": list(SUBJECTS),
        "numeric_rule": "src/math_task.py module doc, 'Numeric rule': normalize_answer of the gold string is not None",
        "gold_definition": {"target": "MATH-500 'answer' via normalize_answer",
                            "train": "content of the solution's last complete \\boxed{} or \\fbox{} via normalize_answer"},
        "item_id_definition": {"target": "MATH-500 unique_id", "train": "'train/<subject>/<row>', row = zero-based "
                               "position in the subject's pinned parquet file"},
        "counts": {"target_all": int(len(target)), "target_numeric": int(target["gold_answer"].notna().sum()),
                   "train_all": int(len(train)), "train_with_box": int(train["gold_text"].notna().sum()),
                   "train_numeric": int(train["gold_answer"].notna().sum())},
        "target_dropped_by_subject": {k: int(v) for k, v in dropped_target["subject"].value_counts().sort_index().items()},
        "target_kept_by_subject": {k: int(v) for k, v in sets["target"]["subject"].value_counts().sort_index().items()},
        "target_kept_by_level": {k: int(v) for k, v in sets["target"]["level"].value_counts().sort_index().items()},
        "seed_label": SEED_LABEL,
        "sampling": "src.sampling.rng_for('math-items', 'train', n).permutation(n) over the numeric training items "
                    "in training order (n = their count); first 200 screening, next n_target calibration, each "
                    "stored in training order; ids only",
        "sets": {k: {"n_items": int(len(v)), "canonical_sha256": canonical_sha256(v)} for k, v in sets.items()},
        "canonical_sha256_definition": "sha256 of UTF-8 JSON (ensure_ascii=False, separators ',' ':') of "
                                       "[[item_id, question, gold_answer], ...] in table order",
        "tables_definition": "one parquet per set, columns " + ", ".join(TABLE_COLUMNS) + "; gold_answer canonical; "
                             "math_task.table_bytes (pyarrow, zstd, no index)",
        "table_libraries": {"pyarrow": pa.__version__, "pandas": pd.__version__, "compression": "zstd"},
        "tables": {k: {"path": f"data/processed/math/items_{k}.parquet", "n_rows": int(len(t)),
                       "sha256": hashlib.sha256(table_bytes(t)).hexdigest()} for k, t in tables.items()},
        "files": files,
    }
    return manifest, tables


def write_manifest(path: str = MANIFEST_PATH, table_dir: str = TABLE_DIR) -> dict:
    manifest, tables = build_manifest()
    os.makedirs(table_dir, exist_ok=True)
    for k, t in tables.items():
        p = os.path.join(table_dir, f"items_{k}.parquet")
        with open(p + ".part", "wb") as f:
            f.write(table_bytes(t))
        os.replace(p + ".part", p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return manifest


# ---------------------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------------------
# Scorer versions: from 1.0.1, a slash fraction that ends a sentence ("3/4.") is read whole. From 1.0.2, a number of
# any length is read exactly, with no limit from Python's integer conversion.
SCORER_VERSION = "math-boxed/1.0.2"
ANSWER_FORMAT = "End your reply with a final line that contains \\boxed{<number>}."
_LENIENT = re.compile(
    r"(?<![\w\\])-?\s*\\[dt]?frac\s*(?:\{\s*-?[0-9]+\s*\}|[0-9])\s*(?:\{\s*-?[0-9]+\s*\}|[0-9])"
    r"|(?<![\w./])-?[0-9]+\s*/\s*[0-9]+(?![0-9]|\.[0-9])"
    r"|(?<!\w)-?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_THIN_THOUSANDS = re.compile(r"(?<=[0-9]),?\\[,!;:]\s*(?=[0-9]{3}(?![0-9]))")


def strict_extract(text) -> str | None:
    """Canonical number in the last complete \\boxed{...}, or None (module doc)."""
    boxes = bbh.boxed_contents(text)
    return normalize_answer(boxes[-1]) if boxes else None


def lenient_extract(text) -> str | None:
    """Canonical value of the last number expression in the output, or None (module doc)."""
    t = sc.normalize_text(text)
    if not t:
        return None
    t = t.replace("{,}", ",")
    t = _THIN_THOUSANDS.sub(",", t)
    last = None
    for m in _LENIENT.finditer(t):
        last = m.group(0)
    return normalize_answer(last) if last is not None else None


def compliance(text) -> bool:
    return strict_extract(text) is not None


REQUIRED_COLUMNS = sc.REQUIRED_COLUMNS
SCORE_COLUMNS = sc.SCORE_COLUMNS


def _gold_table(items: pd.DataFrame) -> dict:
    need = {"item_id", "gold_answer"}
    if not isinstance(items, pd.DataFrame) or not need <= set(items.columns):
        raise ValueError(f"items must be a DataFrame with columns {sorted(need)}")
    if items["item_id"].duplicated().any():
        raise ValueError("items hold duplicate item_id values")
    out = {}
    for iid, gold in zip(items["item_id"], items["gold_answer"]):
        if gold is None or normalize_answer(gold) != gold:
            raise ValueError(f"gold answer for item {iid!r} is not a canonical number: {gold!r}")
        out[str(iid)] = gold
    return out


def score_outputs(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Add strict_answer, strict_correct, lenient_answer, lenient_correct and compliant (as scoring.score_outputs).

    df needs request_id (unique), item_id and output_text; missing or non-string output text is the empty string.
    Correctness is equality of canonical numbers; strict_correct is 0 for any non-extractable output."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["request_id"].duplicated().any():
        dup = df.loc[df["request_id"].duplicated(), "request_id"].head(3).tolist()
        raise ValueError(f"request_id must be unique; duplicates such as {dup}")
    if df["item_id"].isna().any():
        raise ValueError("item_id contains missing values")
    lookup = _gold_table(items)
    item_ids = [str(i) for i in df["item_id"]]
    absent = sorted({i for i in item_ids if i not in lookup})
    if absent:
        raise KeyError(f"{len(absent)} item_id values have no gold answer, e.g. {absent[:5]}")
    texts = [t if isinstance(t, str) else "" for t in df["output_text"].tolist()]
    cache: dict = {}

    def both(t):
        if t not in cache:
            cache[t] = (strict_extract(t), lenient_extract(t))
        return cache[t]

    pairs = [both(t) for t in texts]
    gold = [lookup[i] for i in item_ids]
    strict = [p[0] for p in pairs]
    lenient = [p[1] for p in pairs]
    out = df.copy()
    out["strict_answer"] = pd.Series(strict, index=df.index, dtype=object)
    out["strict_correct"] = np.fromiter((int(s is not None and s == g) for s, g in zip(strict, gold)),
                                        dtype=np.int64, count=len(strict))
    out["lenient_answer"] = pd.Series(lenient, index=df.index, dtype=object)
    out["lenient_correct"] = np.fromiter((int(s is not None and s == g) for s, g in zip(lenient, gold)),
                                         dtype=np.int64, count=len(lenient))
    out["compliant"] = np.fromiter((s is not None for s in strict), dtype=bool, count=len(strict))
    return out


def scorer_rules() -> dict:
    """The behaviour-defining description of the scorer, hashed into the scorer manifest."""
    return {
        "text_normalization": "scoring.normalize_text (as GSM8K) before any extraction",
        "normalize_answer": "src/math_task.py module doc, 'Numeric rule' steps 1-4: decorations removed, then an "
                            "integer, decimal, \\frac/\\dfrac/\\tfrac or slash fraction; canonical reduced 'p/q' or 'p'",
        "patterns": {"degrees": _DEGREES.pattern, "unit": _UNIT.pattern, "percent": _PERCENT.pattern,
                     "spacing": _SPACING.pattern, "dollar": _DOLLAR.pattern, "decimal": _DEC.pattern,
                     "frac": _FRAC.pattern, "slash": _SLASH.pattern, "wrappers": [list(w) for w in _WRAPPERS]},
        "strict": "normalize_answer of the content of the last complete \\boxed{...} (bbh.boxed_contents); None "
                  "without a box or when the content is not a number",
        "lenient": "normalize_answer of the last match of the lenient pattern after '{,}' and LaTeX thin-space "
                   "thousands are read as commas",
        "lenient_pattern": _LENIENT.pattern, "thin_thousands": _THIN_THOUSANDS.pattern,
        "box_pattern": "\\\\boxed\\{ (bbh.boxed_contents)",
        "compliance": "strict extraction succeeds",
        "equality": "equal canonical strings (exact rational equality)",
        "length": "numbers of any length are read exactly (Python's 4,300-digit limit on integer conversion is lifted "
                  "inside normalize_answer)",
    }


def scorer_record() -> dict:
    """Provenance record for manifests/math_scorer.json (pins the scorer version)."""
    rules = scorer_rules()
    sha = hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"scorer_version": SCORER_VERSION, "plan": PLAN, "answer_format": ANSWER_FORMAT, "rules": rules,
            "rules_sha256": sha}


def write_scorer_manifest(path: str = SCORER_MANIFEST_PATH) -> dict:
    rec = scorer_record()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return rec


def latex_form(gold: str) -> str:
    """The LaTeX spelling of a canonical answer that the forced arm would write: 'p' or '\\frac{p}{q}' (sign first)."""
    if "/" not in gold:
        return gold
    p, q = gold.split("/")
    return f"-\\frac{{{p[1:]}}}{{{q}}}" if p.startswith("-") else f"\\frac{{{p}}}{{{q}}}"


def answer_token_lengths(answers: Sequence[str], tokenizer) -> list[int]:
    """Tokens of each answer string plus the closing brace, for a `tokenizers.Tokenizer`, without special tokens (the
    forced arm's cap check)."""
    return [len(tokenizer.encode(a + "}", add_special_tokens=False).ids) for a in answers]
