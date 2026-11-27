"""Tests for src/math_task.py: the numeric rule, the reference-solution box, the item sets, the scorer and its
manifest, and the forced arm's token check. Tests on the pinned MATH files and the local tokenizers skip when they
are absent."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import math_task as mt

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- the numeric rule

@pytest.mark.parametrize("s, want", [
    ("12", "12"), ("-3", "-3"), ("+3", "3"), ("-0", "0"), ("1,000", "1000"), ("1,234,567", "1234567"),
    ("2.50", "5/2"), (".5", "1/2"), ("-0.25", "-1/4"), ("1,234.5", "2469/2"), ("3.", "3"),
    ("\\frac{14}{3}", "14/3"), ("\\dfrac{6}{4}", "3/2"), ("\\tfrac{6}{4}", "3/2"), ("\\frac43", "4/3"),
    ("\\frac9{19}", "9/19"), ("-\\frac{1}{2}", "-1/2"), ("\\frac{-1}{2}", "-1/2"), ("\\frac{1}{-2}", "-1/2"),
    ("\\frac{4}{2}", "2"), ("14/3", "14/3"), ("-6/4", "-3/2"),
    ("$\\frac{14}{3}$", "14/3"), ("\\(5\\)", "5"), ("\\[5\\]", "5"), ("$$7$$", "7"), ("\\displaystyle\\frac{1}{2}", "1/2"),
    ("10^\\circ", "10"), ("10^{\\circ}", "10"), ("90°", "90"), ("50\\%", "50"), ("50%", "50"),
    ("5\\text{ inches}", "5"), ("12\\mbox{ cm}^2", "12"), ("8\\text{ cm}^{3}", "8"), ("\\$18.90", "189/10"),
    ("-\\$5", "-5"), ("$-5", "-5"), ("1{,}000", "1000"), ("1\\,000", "1000"), (" 42 ", "42"), ("\\!7", "7"),
])
def test_numeric_answers(s, want):
    assert mt.normalize_answer(s) == want


@pytest.mark.parametrize("s", [
    None, "", "x", "x=3", "2\\frac{1}{2}", "\\sqrt{2}", "2\\sqrt{3}", "\\pi", "3\\pi", "(1,2)", "[0,1)", "\\infty",
    "1e3", "2^{10}", "1,23", "12,34,567", "--3", "-\\frac{-1}{2}", "\\frac{1}{0}", "4/0", "\\text{5}", "\\text{(A)}",
    "\\frac{x}{2}", "3, 4", "-$-5", "\\$",
])
def test_non_numeric_answers(s):
    assert mt.normalize_answer(s) is None


def test_canonical_strings_are_fixed_points_and_latex_form_round_trips():
    for v in ("0", "7", "-7", "1/2", "-1/2", "1000", "355/113"):
        assert mt.normalize_answer(v) == v
        assert mt.normalize_answer(mt.latex_form(v)) == v
    assert mt.latex_form("-14/3") == "-\\frac{14}{3}" and mt.latex_form("5") == "5"
    assert mt.normalize_answer("0.333") != mt.normalize_answer("1/3")          # exact rational equality


def test_solution_answer_takes_the_last_complete_box():
    assert mt.solution_answer("so \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert mt.solution_answer("\\boxed{1} and \\fbox{3} then \\boxed{4") == "3"
    assert mt.solution_answer("\\boxed{\\text{\\{a\\}}}") == "\\text{\\{a\\}}"
    assert mt.solution_answer("no box") is None


# ---------------------------------------------------------------- the scorer

def test_strict_lenient_and_compliance():
    assert mt.strict_extract("work \\boxed{\\dfrac{3}{4}}") == "3/4"
    assert mt.strict_extract("\\boxed{3} then \\boxed{x}") is None              # the last box is not a number
    assert mt.strict_extract("\\boxed{3} then \\boxed{4}") == "4"
    assert mt.strict_extract("\\boxed{\\frac{14}") is None                     # a box cut at its first brace
    assert mt.strict_extract("The answer is 5.") is None
    assert mt.lenient_extract("so x = \\frac{3}{4}") == "3/4"
    assert mt.lenient_extract("3/4 of 12 is 9") == "9"
    assert mt.lenient_extract("the answer is 3/4.") == "3/4"                    # 1.0.1: a final period
    assert mt.lenient_extract("about 3/4.5 of it") == "9/2"                    # not a fraction of integers: 4.5
    assert mt.lenient_extract("The answer is 1,234.") == "1234"
    assert mt.lenient_extract("1\\,234 apples") == "1234"
    assert mt.lenient_extract("x2 and 7") == "7"
    assert mt.lenient_extract("it is -5") == "-5"
    assert mt.lenient_extract("no numbers") is None and mt.lenient_extract("") is None
    assert mt.compliance("\\boxed{7}") and not mt.compliance("\\boxed{\\sqrt{2}}") and not mt.compliance("7")


def items_frame():
    return pd.DataFrame({"item_id": ["test/algebra/1.json", "train/geometry/4", "x"], "gold_answer": ["12", "-1/2", "3"]})


def test_score_outputs():
    df = pd.DataFrame({"request_id": ["r1", "r2", "r3", "r4", "r5"],
                       "item_id": ["test/algebra/1.json", "train/geometry/4", "train/geometry/4", "x", "x"],
                       "output_text": ["\\boxed{12}", "So \\boxed{-\\frac{1}{2}}", "it is -0.5", None, "\\boxed{3.0}"]})
    out = mt.score_outputs(df, items_frame())
    assert out["strict_correct"].tolist() == [1, 1, 0, 0, 1]
    assert out["lenient_correct"].tolist() == [1, 1, 1, 0, 1]
    assert out["compliant"].tolist() == [True, True, False, False, True]
    assert out["strict_answer"].tolist() == ["12", "-1/2", None, None, "3"]
    assert list(out.columns[:3]) == ["request_id", "item_id", "output_text"]
    with pytest.raises(ValueError, match="unique"):
        mt.score_outputs(df.assign(request_id="r"), items_frame())
    with pytest.raises(KeyError, match="no gold"):
        mt.score_outputs(df.assign(item_id="y"), items_frame())
    with pytest.raises(ValueError, match="canonical"):
        mt.score_outputs(df, items_frame().assign(gold_answer=["12", "0.5", "3"]))
    with pytest.raises(ValueError, match="missing columns"):
        mt.score_outputs(df.drop(columns="output_text"), items_frame())


def test_numbers_of_any_length_are_read_exactly():
    limit = sys.get_int_max_str_digits()
    big = "7" * 7326
    assert mt.normalize_answer(big) == big
    assert mt.strict_extract("so \\boxed{" + big + "}") == big
    assert mt.lenient_extract("the answer is " + big + ".") == big
    assert mt.normalize_answer("\\frac{" + big + "}{7}") == "1" + "1" * 7325
    assert sys.get_int_max_str_digits() == limit                  # the limit is put back after each call
    assert mt.normalize_answer("12") == "12"


def test_scorer_record_and_manifest():
    rec = mt.scorer_record()
    assert rec["scorer_version"] == "math-boxed/1.0.2" and rec["answer_format"].endswith("\\boxed{<number>}.")
    sha = hashlib.sha256(json.dumps(rec["rules"], sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    assert rec["rules_sha256"] == sha
    assert json.loads((ROOT / "manifests" / "math_scorer.json").read_text(encoding="utf-8")) == rec


# ---------------------------------------------------------------- item sets on fabricated tables

def fake_frames(n_target=5, n_train=40):
    target = pd.DataFrame({"item_id": [f"test/algebra/{i}.json" for i in range(n_target + 2)],
                           "question": [f"T{i}" for i in range(n_target + 2)],
                           "gold_text": [str(i) for i in range(n_target)] + ["\\sqrt2", "(1,2)"],
                           "subject": "Algebra", "level": "Level 1"})
    train = pd.DataFrame({"item_id": [f"train/algebra/{i}" for i in range(n_train)],
                          "question": [f"Q{i}" for i in range(n_train)],
                          "gold_text": [str(i) if i % 4 else "x" for i in range(n_train)],
                          "subject": "Algebra", "level": "Level 2"})
    for df in (target, train):
        df["gold_answer"] = [mt.normalize_answer(g) for g in df["gold_text"]]
    return target, train


def test_split_items_is_seeded_disjoint_and_in_training_order(monkeypatch):
    monkeypatch.setattr(mt, "N_SCREENING", 8)
    target, train = fake_frames()
    a, b = mt.split_items(target, train), mt.split_items(target, train)
    assert len(a["target"]) == 5 and a["target"]["gold_answer"].notna().all()
    assert len(a["screening"]) == 8 and len(a["calibration"]) == 5
    for k in a:
        assert a[k].equals(b[k])
    scr, cal = set(a["screening"]["item_id"]), set(a["calibration"]["item_id"])
    assert not scr & cal and all(train.set_index("item_id").loc[list(scr | cal), "gold_answer"].notna())
    rows = {iid: i for i, iid in enumerate(train["item_id"])}
    for k in ("screening", "calibration"):
        pos = [rows[i] for i in a[k]["item_id"]]
        assert pos == sorted(pos)
    num = train[train["gold_answer"].notna()].reset_index(drop=True)
    perm = mt.train_permutation(len(num))
    assert list(a["screening"]["item_id"]) == list(num["item_id"].iloc[np.sort(perm[:8])])
    with pytest.raises(mt.MathError, match="cannot fill"):
        mt.split_items(target, train.iloc[:10])


def test_canonical_sha_and_table_bytes_are_deterministic():
    _, train = fake_frames()
    df = train[mt.TABLE_COLUMNS].dropna()
    assert mt.table_bytes(df) == mt.table_bytes(df.copy())
    rows = [[i, q, g] for i, q, g in zip(df["item_id"], df["question"], df["gold_answer"])]
    want = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert mt.canonical_sha256(df) == want


class CharTokenizer:
    """One token per character, for answer_token_lengths."""

    class _Enc:
        def __init__(self, n):
            self.ids = list(range(n))

    def encode(self, s, add_special_tokens=False):
        return self._Enc(len(s))


def test_answer_token_lengths_count_the_closing_brace():
    assert mt.answer_token_lengths(["12", "\\frac{1}{2}"], CharTokenizer()) == [3, 12]


# ---------------------------------------------------------------- the pinned files

needs_raw = pytest.mark.skipif(
    not Path(mt.target_path()).is_file() or not all(Path(mt.train_path(s)).is_file() for s in mt.SUBJECTS),
    reason="pinned MATH files absent (scripts/fetch_math.py)")


@needs_raw
def test_pinned_files_reproduce_the_manifest():
    man = json.loads(Path(mt.MANIFEST_PATH).read_text(encoding="utf-8"))
    built, tables = mt.build_manifest()
    for k in ("counts", "sets", "tables", "files", "target_kept_by_level", "target_dropped_by_subject"):
        assert built[k] == man[k], k
    assert man["counts"] == {"target_all": 500, "target_numeric": 398, "train_all": 7500, "train_with_box": 7498,
                             "train_numeric": 6091}
    assert {k: v["n_items"] for k, v in man["sets"].items()} == {"target": 398, "calibration": 398, "screening": 200}
    ids = {k: set(t["item_id"]) for k, t in tables.items()}
    assert not ids["screening"] & ids["calibration"]
    assert all(i.startswith("test/") for i in ids["target"])
    assert all(i.startswith("train/") for i in ids["screening"] | ids["calibration"])
    assert not set(tables["target"]["question"]) & (set(tables["screening"]["question"]) |
                                                    set(tables["calibration"]["question"]))
    assert all(mt.normalize_answer(g) == g for t in tables.values() for g in t["gold_answer"])
    assert all("|" not in i for t in tables.values() for i in t["item_id"])         # request ids use "|"


TOKENIZERS = sorted((ROOT / "data" / "tokenizers").glob("*/tokenizer.json"))


@needs_raw
@pytest.mark.skipif(not TOKENIZERS, reason="local tokenizers absent (data/tokenizers)")
def test_forced_cap_fits_every_target_answer():
    from tokenizers import Tokenizer
    man = json.loads(Path(mt.MANIFEST_PATH).read_text(encoding="utf-8"))
    gold = pd.read_parquet(ROOT / man["tables"]["target"]["path"])["gold_answer"]
    forms = [mt.latex_form(g) for g in gold]
    for p in TOKENIZERS:
        n = mt.answer_token_lengths(forms, Tokenizer.from_file(str(p)))
        assert max(n) <= 16, p.parent.name
