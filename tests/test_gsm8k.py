"""Tests for src/gsm8k.py. Data-dependent tests skip cleanly when the cached files are absent."""
import json
import os
import re

import pandas as pd
import pytest

from src import gsm8k
from src import scoring as sc

needs_test = pytest.mark.skipif(not gsm8k.is_cached("test"), reason="data/raw/gsm8k test split not downloaded")
needs_train = pytest.mark.skipif(not gsm8k.is_cached("train"), reason="data/raw/gsm8k train split not downloaded")
needs_manifest = pytest.mark.skipif(not os.path.exists(gsm8k.MANIFEST_PATH), reason="manifests/gsm8k_items.json absent")


# ---------------------------------------------------------------------------------------
# Always-run unit tests
# ---------------------------------------------------------------------------------------
def test_pinned_revision_is_a_full_commit_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", gsm8k.REVISION)
    assert gsm8k.REVISION_RESOLVED_ON == "2026-10-03"
    assert gsm8k.DATASET == "openai/gsm8k" and gsm8k.CONFIG == "main"
    assert gsm8k.EXPECTED_ROWS == {"test": 1319, "train": 7473}


@pytest.mark.parametrize("answer, expected", [
    ("She has 9 eggs <<16-3-4=9>>9.\n#### 18", "18"),
    ("x\n#### 70,000", "70000"),
    ("x\n#### 1,000,000", "1000000"),
    ("x\n#### -3", "-3"),
    ("x\n#### 18.00", "18"),
    ("x\n#### 3.50", "3.5"),
    ("x\n#### $18", "18"),
    ("x\n####18", "18"),
    ("x\n#### 0", "0"),
    ("x\n#### -0", "0"),
    ("x\n#### 007", "7"),
    ("x\n#### 18  ", "18"),
    ("x\n#### \u22125", "-5"),
])
def test_parse_gold(answer, expected):
    assert gsm8k.parse_gold(answer) == expected


@pytest.mark.parametrize("answer", [
    "no marker here 18",
    "#### 18\n#### 19",
    "x\n#### eighteen",
    "x\n#### 18 apples",
    "x\n#### ",
    "x\n#### 1,00",
    "text #### 18",            # marker must start a line
])
def test_parse_gold_errors(answer):
    with pytest.raises(ValueError):
        gsm8k.parse_gold(answer)


def test_gold_uses_the_scoring_normalizer():
    for raw in ["70,000", "-3", "18.0", "$5", "007"]:
        assert gsm8k.parse_gold(f"x\n#### {raw}") == sc.normalize_number(raw)


def test_local_path_and_split_validation():
    assert gsm8k.local_path("test").replace("\\", "/").endswith("data/raw/gsm8k/main/test-00000-of-00001.parquet")
    assert gsm8k.local_path("train").replace("\\", "/").endswith("data/raw/gsm8k/main/train-00000-of-00001.parquet")
    with pytest.raises(ValueError):
        gsm8k.local_path("validation")


def test_missing_data_raises_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(gsm8k, "RAW", str(tmp_path))
    assert not gsm8k.is_cached("test")
    with pytest.raises(FileNotFoundError):
        gsm8k.load_gsm8k("test")


def test_canonical_sha256_is_order_and_content_sensitive():
    df = pd.DataFrame({"item_id": [0, 1], "question": ["q0", "q1"], "gold_text": ["a", "b"], "gold_answer": ["1", "2"]})
    h = gsm8k.canonical_sha256(df)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert gsm8k.canonical_sha256(df.copy()) == h
    changed = df.copy()
    changed.loc[1, "gold_answer"] = "3"
    assert gsm8k.canonical_sha256(changed) != h
    swapped = df.iloc[::-1].reset_index(drop=True)
    assert gsm8k.canonical_sha256(swapped) != h
    # gold_text is not part of the canonical form
    other_text = df.copy()
    other_text["gold_text"] = ["x", "y"]
    assert gsm8k.canonical_sha256(other_text) == h
    # non-ASCII is hashed as UTF-8, not escaped
    uni = df.copy()
    uni.loc[0, "question"] = "Janet\u2019s ducks"
    assert gsm8k.canonical_sha256(uni) != h


# ---------------------------------------------------------------------------------------
# Tests on the cached data
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gsm_test():
    return gsm8k.load_gsm8k("test")


@pytest.fixture(scope="module")
def gsm_train():
    return gsm8k.load_gsm8k("train")


@needs_test
def test_test_split_shape_and_columns(gsm_test):
    assert len(gsm_test) == 1319
    assert list(gsm_test.columns) == ["item_id", "question", "gold_text", "gold_answer"]
    assert gsm_test["item_id"].tolist() == list(range(1319))
    assert str(gsm_test["item_id"].dtype) == "int64"
    assert gsm_test["question"].str.len().min() > 0
    assert not gsm_test.isna().any().any()


@needs_test
def test_gold_answers_are_canonical(gsm_test):
    pattern = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?")
    assert gsm_test["gold_answer"].map(lambda g: pattern.fullmatch(g) is not None).all()
    assert not gsm_test["gold_answer"].str.contains(",").any()
    assert (gsm_test["gold_answer"] == gsm_test["gold_answer"].map(sc.normalize_number)).all()
    assert gsm_test.loc[0, "gold_answer"] == "18"
    assert gsm_test.loc[0, "gold_text"].rstrip().endswith("#### 18")
    assert gsm_test["gold_text"].str.count("####").eq(1).all()


@needs_test
def test_gold_edge_cases_present_in_test_split(gsm_test):
    raw_tail = gsm_test["gold_text"].str.extract(r"####[ \t]*(.*?)[ \t]*$")[0]
    with_comma = raw_tail.str.contains(",", regex=False)
    assert with_comma.sum() == 14
    assert (gsm_test.loc[with_comma, "gold_answer"] == raw_tail[with_comma].str.replace(",", "", regex=False)).all()
    negatives = gsm_test[gsm_test["gold_answer"].str.startswith("-")]
    assert len(negatives) == 2
    assert not gsm_test["gold_answer"].str.contains(r"\.", regex=True).any()   # all test golds are integers


@needs_train
def test_train_split_is_separate_from_test(gsm_test, gsm_train):
    assert len(gsm_train) == 7473
    assert gsm_train["item_id"].tolist() == list(range(7473))
    assert set(gsm_train["question"]).isdisjoint(set(gsm_test["question"]))


@needs_test
def test_real_gold_text_scores_correct_under_hash_and_lenient(gsm_test):
    # The reference solution ends with "#### <gold>", so it must score correct under the
    # GSM8K-native candidate spec (strict) and under lenient extraction, including the
    # comma-formatted and negative golds.
    df = pd.DataFrame({"request_id": [f"g{i}" for i in gsm_test["item_id"]], "item_id": gsm_test["item_id"],
                       "output_text": gsm_test["gold_text"]})
    out = sc.score_outputs(df, gsm_test, sc.CANDIDATE_SPECS["hash_n"])
    assert out["strict_correct"].sum() == 1319
    assert out["lenient_correct"].sum() == 1319
    assert out["compliant"].all()


@needs_test
def test_gold_sentence_output_round_trip(gsm_test):
    texts = [f"Working...\nThe answer is {g}." for g in gsm_test["gold_answer"]]
    df = pd.DataFrame({"request_id": [f"s{i}" for i in gsm_test["item_id"]], "item_id": gsm_test["item_id"],
                       "output_text": texts})
    out = sc.score_outputs(df, gsm8k.gold_lookup(gsm_test), sc.CANDIDATE_SPECS["the_answer_is"])
    assert out["strict_correct"].sum() == 1319
    # A wrong-by-one answer is never credited by either extractor.
    wrong = [f"The answer is {int(float(g)) + 1}." for g in gsm_test["gold_answer"]]
    df2 = df.assign(output_text=wrong, request_id=[f"w{i}" for i in gsm_test["item_id"]])
    out2 = sc.score_outputs(df2, gsm_test, sc.CANDIDATE_SPECS["the_answer_is"])
    assert out2["strict_correct"].sum() == 0 and out2["lenient_correct"].sum() == 0


@needs_test
def test_gold_lookup(gsm_test):
    lookup = gsm8k.gold_lookup(gsm_test)
    assert len(lookup) == 1319 and lookup[0] == "18"


# ---------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------
@needs_manifest
def test_manifest_content():
    with open(gsm8k.MANIFEST_PATH, encoding="utf-8") as f:
        m = json.load(f)
    assert m["dataset"] == "openai/gsm8k" and m["config"] == "main"
    assert m["revision"] == gsm8k.REVISION
    assert m["revision_resolved_on"] == gsm8k.REVISION_RESOLVED_ON
    assert m["splits"]["test"]["n_items"] == 1319
    assert m["splits"]["train"]["n_items"] == 7473
    assert m["primary_evaluation_split"] == "test" and m["calibration_split"] == "train"
    for split in ("test", "train"):
        assert re.fullmatch(r"[0-9a-f]{64}", m["splits"][split]["canonical_sha256"])
    assert m["splits"]["test"]["canonical_sha256"] != m["splits"]["train"]["canonical_sha256"]


@needs_manifest
@needs_test
def test_manifest_hash_matches_cached_data(gsm_test):
    with open(gsm8k.MANIFEST_PATH, encoding="utf-8") as f:
        m = json.load(f)
    assert gsm8k.canonical_sha256(gsm_test) == m["splits"]["test"]["canonical_sha256"]
    rel = gsm8k.SPLIT_FILES["test"]
    assert gsm8k.file_sha256(gsm8k.local_path("test")) == m["files"][rel]["sha256"]
    assert os.path.getsize(gsm8k.local_path("test")) == m["files"][rel]["bytes"]


@needs_manifest
@needs_train
def test_manifest_hash_matches_cached_train(gsm_train):
    with open(gsm8k.MANIFEST_PATH, encoding="utf-8") as f:
        m = json.load(f)
    assert gsm8k.canonical_sha256(gsm_train) == m["splits"]["train"]["canonical_sha256"]
    rel = gsm8k.SPLIT_FILES["train"]
    assert gsm8k.file_sha256(gsm8k.local_path("train")) == m["files"][rel]["sha256"]


@needs_manifest
@needs_test
@needs_train
def test_manifest_is_reproducible(tmp_path):
    out = tmp_path / "m.json"
    built = gsm8k.write_manifest(str(out))
    with open(gsm8k.MANIFEST_PATH, encoding="utf-8") as f:
        committed = json.load(f)
    assert built == committed
    assert out.read_text(encoding="utf-8") == open(gsm8k.MANIFEST_PATH, encoding="utf-8", newline="").read()
