"""DOVE_Lite access: pinned revision, parallel file download, compact score extraction.

Files go to data/raw/dove_lite/<model>/<lang>/<shots>/<file>.parquet (git-ignored).
Only files named by the caller are downloaded; nothing here lists or fetches the full
release. Authentication comes from the cached `hf auth login` token; no token is read,
written or logged by this module.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = "nlphuji/DOVE_Lite"
REVISION = "bc1129e223f3f48f0e4180694cde261e02da685e"  # pinned 3 Oct 2026; last modified 2025-08-14
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw", "dove_lite")
PROCESSED = os.path.join(ROOT, "data", "processed", "dove")

ITEM = "sample_index"
DIMS = ["dimensions_1: enumerator", "dimensions_2: separator",
        "dimensions_3: choices_order", "dimensions_4: instruction_phrasing_text"]
SHOTS = "dimensions_5: shots"
SCORE = "score"
DIM_NAMES = ["enumerator", "separator", "choices_order", "instruction"]

# Orderings defined by the position of the gold answer; unavailable for an unlabeled
# item, so excluded from the primary admissible population.
GOLD_DEPENDENT_ORDERS = ("correct_first", "correct_last")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def local_path(rel: str) -> str:
    return os.path.join(RAW, *rel.split("/"))


def download(rel_paths, threads: int = 8, log=print) -> dict:
    """Download the named repo files (skipping ones already present). Returns stats."""
    from huggingface_hub import hf_hub_download

    todo = [p for p in rel_paths if not os.path.exists(local_path(p))]
    stats = {"requested": len(rel_paths), "already_present": len(rel_paths) - len(todo),
             "downloaded": 0, "bytes": 0, "seconds": 0.0}
    t0 = time.time()

    def one(rel):
        hf_hub_download(REPO, rel, repo_type="dataset", revision=REVISION, local_dir=RAW)
        return rel, os.path.getsize(local_path(rel))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(one, p) for p in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rel, size = fut.result()
            stats["downloaded"] += 1
            stats["bytes"] += size
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t0
                log(f"  downloaded {k}/{len(todo)} files, {stats['bytes'] / 1e9:.1f} GB, "
                    f"{stats['bytes'] / 1e6 / max(el, 1e-9):.0f} MB/s")
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def read(rel: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(local_path(rel), columns=columns)


def config_key(df: pd.DataFrame) -> pd.Series:
    """Canonical configuration key over the non-intervention prompt dimensions."""
    return df[DIMS].astype(str).agg("|".join, axis=1)


def split_key(key: str) -> dict:
    parts = key.split("|", 3)
    return dict(zip(DIM_NAMES, parts))


def label_text(answer: str) -> tuple[str, str]:
    """Split a released answer string such as 'β. 4' into (label, text)."""
    if ". " in answer:
        lab, txt = answer.split(". ", 1)
        return lab, txt
    return "", answer
