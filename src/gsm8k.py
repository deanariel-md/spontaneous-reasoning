r"""GSM8K access for Dataset 2: pinned revision, local cache, canonical gold answers, manifest.

Source: Hugging Face dataset `openai/gsm8k`, config `main`, parquet files for the `test`
(1,319 items; the Dataset 2 evaluation set) and `train` (7,473 items; used only for
calibration and admissibility screening on separate items, never as target items).
The dataset is public (license MIT per the dataset card) and not gated. Only the two small
parquet files are downloaded, always at the pinned revision, to data/raw/gsm8k/ (git-ignored).

Item ids are the zero-based row position within the split's parquet file. Test ids and
train ids are separate namespaces. The gold answer is the text after "####" in the reference
solution, passed through `scoring.normalize_number`, the same normalization that scoring
applies to model outputs.

PowerShell, to download (small public files) and write manifests/gsm8k_items.json:
    .\.venv\Scripts\python.exe -m src.gsm8k
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import pandas as pd

from .scoring import normalize_number

DATASET = "openai/gsm8k"
CONFIG = "main"
REVISION = "740312add88f781978c0658806c59bc2815b9866"  # repo last modified 2026-03-23
REVISION_RESOLVED_ON = "2026-10-03"
LICENSE = "mit"  # dataset card, as reported by the Hub API on the resolution date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw", "gsm8k")
MANIFEST_PATH = os.path.join(ROOT, "manifests", "gsm8k_items.json")

SPLIT_FILES = {
    "test": "main/test-00000-of-00001.parquet",
    "train": "main/train-00000-of-00001.parquet",
}
EXPECTED_ROWS = {"test": 1319, "train": 7473}
COLUMNS = ["item_id", "question", "gold_text", "gold_answer"]

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_GOLD_LINE = re.compile(r"^####[ \t]*(?P<num>.*?)[ \t]*$", re.MULTILINE)


def local_path(split: str) -> str:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    return os.path.join(RAW, *SPLIT_FILES[split].split("/"))


def is_cached(split: str) -> bool:
    return os.path.exists(local_path(split))


def download(splits=("test", "train"), log=print) -> dict:
    """Download the named parquet files at the pinned revision. Returns {split: path}."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for split in splits:
        rel = SPLIT_FILES[split]
        if not is_cached(split):
            log(f"downloading {DATASET}@{REVISION[:8]} {rel}")
            hf_hub_download(DATASET, rel, repo_type="dataset", revision=REVISION, local_dir=RAW)
        paths[split] = local_path(split)
    return paths


def parse_gold(answer_text: str) -> str:
    """Canonical gold number from a GSM8K reference solution (text after the single "####")."""
    matches = _GOLD_LINE.findall(answer_text)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '####' line, found {len(matches)}")
    canon = normalize_number(matches[0])
    if canon is None:
        raise ValueError(f"gold answer is not a plain number: {matches[0]!r}")
    return canon


def load_gsm8k(split: str = "test", download_if_missing: bool = False) -> pd.DataFrame:
    """DataFrame with item_id (split index), question, gold_text, gold_answer (canonical string)."""
    if not is_cached(split):
        if not download_if_missing:
            raise FileNotFoundError(
                f"{local_path(split)} is absent; call src.gsm8k.download() or "
                "load_gsm8k(split, download_if_missing=True)")
        download((split,))
    raw = pd.read_parquet(local_path(split))
    if len(raw) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, found {len(raw)}")
    df = pd.DataFrame({
        "item_id": pd.Series(range(len(raw)), dtype="int64"),
        "question": raw["question"].astype(str).to_numpy(),
        "gold_text": raw["answer"].astype(str).to_numpy(),
    })
    df["gold_answer"] = [parse_gold(a) for a in df["gold_text"]]
    return df[COLUMNS]


def gold_lookup(df: pd.DataFrame) -> dict:
    """item_id -> canonical gold answer."""
    return dict(zip(df["item_id"].tolist(), df["gold_answer"].tolist()))


def canonical_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of the canonical list [[item_id, question, gold_answer], ...].

    Serialization: JSON, ensure_ascii=False, separators (",", ":"), UTF-8, rows in item_id order.
    """
    rows = [[int(i), q, g] for i, q, g in zip(df["item_id"], df["question"], df["gold_answer"])]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict:
    splits = {}
    files = {}
    for split in SPLIT_FILES:
        df = load_gsm8k(split)
        splits[split] = {
            "n_items": int(len(df)),
            "canonical_sha256": canonical_sha256(df),
            "gold_has_negative": int(df["gold_answer"].str.startswith("-").sum()),
            "gold_has_decimal": int(df["gold_answer"].str.contains(r"\.", regex=True).sum()),
            "gold_raw_has_thousands_separator": int(
                df["gold_text"].str.extract(r"####[ \t]*(.*?)[ \t]*$")[0].str.contains(",", regex=False).sum()),
        }
        files[SPLIT_FILES[split]] = {"bytes": os.path.getsize(local_path(split)),
                                     "sha256": file_sha256(local_path(split))}
    return {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": REVISION,
        "revision_resolved_on": REVISION_RESOLVED_ON,
        "license": LICENSE,
        "item_id_definition": "zero-based row position within the split's parquet file; "
                              "test and train ids are separate namespaces",
        "gold_definition": "text after the single '####' line, via scoring.normalize_number "
                           "(commas removed, '-' kept, trailing zeros and leading zeros dropped)",
        "canonical_sha256_definition": "sha256 of UTF-8 JSON (ensure_ascii=False, separators ',' ':') "
                                       "of [[item_id, question, gold_answer], ...] in item_id order",
        "primary_evaluation_split": "test",
        "calibration_split": "train",
        "splits": splits,
        "files": files,
    }


def write_manifest(path: str = MANIFEST_PATH) -> dict:
    manifest = build_manifest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return manifest


if __name__ == "__main__":
    download()
    m = write_manifest()
    print(json.dumps({k: m[k] for k in ("revision", "splits")}, indent=2))
