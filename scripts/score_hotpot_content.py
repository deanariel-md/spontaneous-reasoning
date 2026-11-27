r"""HotpotQA's content reading on every stored output.

From the repository root:
    python scripts/score_hotpot_content.py

Reads every output folder of data/processed/hotpot/scored.json (the prefill prepended, as src/d2_io.inspect_work_dir
does), keeps the rows of data/processed/hotpot/scored.parquet, and adds the content reading (src/hotpot_task.score_content,
version CONTENT_SCORER_VERSION) and the format flags of src/hotpot_content.format_flags. The strict and lenient columns
are copied from scored.parquet and checked against the outputs' own strict reading on every row; scored.parquet itself is
not written.

Writes
    data/processed/hotpot/scored_content.parquet  scored.parquet's columns, content_answer, content_correct,
                                                  content_f1 (0-1), content_source, placeholder_copy,
                                                  last_box_placeholder, answer_place; and .json (metadata)
    data/processed/hotpot/content_view/           scored.parquet with strict_correct holding content_correct and
                                                  strict_f1 holding content_f1 (lenient columns as run), item_set.json
                                                  and scored.json: the input that lets the existing HotpotQA analysis
                                                  functions run unchanged under content scoring
    manifests/hotpot_content_scorer.json          the content reading's version and rules (written once; a different
                                                  existing file is refused, never overwritten)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROCESSED = os.path.join(ROOT, "data", "processed", "hotpot")
VIEW = os.path.join(PROCESSED, "content_view")
MANIFEST = os.path.join(ROOT, "manifests", "hotpot_content_scorer.json")
KEY = ["model_key", "arm", "config_id", "item_id"]


def write_manifest(path: str = MANIFEST) -> dict:
    """Write the content scorer's record once; an identical file is kept, a different one refused."""
    from src import hotpot_task as ht
    rec = ht.content_scorer_record()
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old != rec:
            raise SystemExit(f"{path} exists and differs from src/hotpot_task.content_scorer_record(); not overwritten")
        return rec
    ht.write_json(rec, path)
    return rec


def score_folder(d: str, target: pd.DataFrame, allow_test_backend: bool = False) -> pd.DataFrame:
    """One output folder: KEY, the strict reading (for the check), the content columns and the format flags."""
    from src import d2_io, hotpot_content as hc, hotpot_task as ht
    out, _ = d2_io.inspect_work_dir(d, string_item_ids=True, allow_test_backend=allow_test_backend)
    out = out[out["item_id"].astype(str).isin(set(target["item_id"]))].reset_index(drop=True)
    sc = ht.score_content(out[KEY + ["output_text"]].assign(item_id=out["item_id"].astype(str)), target)
    sc["strict_check"] = [0 if s is None else ht.exact_match(s, g) for s, g in zip(
        (ht.strict_extract(t if isinstance(t, str) else "") for t in sc["output_text"]),
        sc["item_id"].map(dict(zip(target["item_id"], target["gold_answer"]))))]
    fl = hc.flag_frame(sc["output_text"].tolist(), hc.PLACEHOLDERS["hotpot"])
    for c in fl.columns:
        sc[c] = fl[c].to_numpy()
    return sc.drop(columns=["output_text"])


def run(processed: str = PROCESSED, view: str = VIEW, manifest: str = MANIFEST, allow_test_backend: bool = False) -> dict:
    from src import hotpot_d2, hotpot_task as ht
    t0 = time.time()
    rec = write_manifest(manifest)
    meta0 = json.load(open(os.path.join(processed, "scored.json"), encoding="utf-8"))
    scored = pd.read_parquet(os.path.join(processed, "scored.parquet"))
    for c in ("model_key", "arm", "config_id", "item_id"):
        scored[c] = scored[c].astype(str)
    target, _ = hotpot_d2.load_items("target")
    target = target.assign(item_id=target["item_id"].astype(str))
    parts = []
    for key, fo in meta0["folders"].items():
        d = os.path.join(ROOT, fo["path"])
        parts.append(score_folder(d, target, allow_test_backend))
        print(f"{key}: {len(parts[-1])} outputs ({time.time() - t0:.0f} s)", flush=True)
    c = pd.concat(parts, ignore_index=True)
    for k in KEY:
        c[k] = c[k].astype(str)
    tab = scored.merge(c, on=KEY, how="left", validate="one_to_one")
    if tab["content_source"].isna().any():
        raise SystemExit(f"{int(tab['content_source'].isna().sum())} scored rows have no output")
    mism = int((tab["strict_check"] != tab["strict_correct"]).sum())
    if mism:
        raise SystemExit(f"{mism} rows: the outputs' strict reading differs from scored.parquet")
    tab = tab.drop(columns=["strict_check"])
    tab["content_correct"] = tab["content_correct"].astype(np.int64)
    path = os.path.join(processed, "scored_content.parquet")
    tab.to_parquet(path, index=False, compression="zstd")
    bad = int(((tab["strict_correct"] == 1) & (tab["content_correct"] == 0)).sum())
    by_arm = {a: {"n": int(len(g)), "strict_em_pct": float(100 * g["strict_correct"].mean()),
                  "content_em_pct": float(100 * g["content_correct"].mean()),
                  "strict_f1_pct": float(100 * g["strict_f1"].mean()),
                  "content_f1_pct": float(100 * g["content_f1"].mean()),
                  "placeholder_copy_pct": float(100 * g["placeholder_copy"].mean()),
                  "sources": {s: int(v) for s, v in g["content_source"].value_counts().sort_index().items()}}
              for a, g in tab.groupby("arm")}
    meta = {"source": "data/processed/hotpot/scored.parquet", "source_sha256": meta0.get("sha256"),
            "content_scorer": rec["scorer_version"], "content_rules_sha256": rec["rules_sha256"],
            "strict_scorer": meta0.get("scorer"), "n_rows": int(len(tab)), "strict_correct_not_content_correct": bad,
            "strict_rows_checked_against_outputs": int(len(tab)), "by_arm": by_arm,
            "sha256": ht.file_sha256(path), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    ht.write_json(meta, path[:-len(".parquet")] + ".json")
    # the view: the existing analysis functions read strict_correct (and strict_f1); here they hold the content reading
    os.makedirs(view, exist_ok=True)
    v = tab[list(scored.columns)].copy()
    v["strict_correct"] = tab["content_correct"].to_numpy()
    v["strict_f1"] = tab["content_f1"].to_numpy()
    vpath = os.path.join(view, "scored.parquet")
    v.to_parquet(vpath, index=False, compression="zstd")
    shutil.copyfile(os.path.join(processed, "item_set.json"), os.path.join(view, "item_set.json"))
    ht.write_json({**{k: meta0[k] for k in meta0 if k not in ("sha256", "created")},
                   "view": "strict_correct holds content_correct and strict_f1 holds content_f1 "
                           f"({rec['scorer_version']}); lenient columns as run",
                   "from": "data/processed/hotpot/scored_content.parquet", "sha256": ht.file_sha256(vpath),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(view, "scored.json"))
    print(json.dumps({k: meta[k] for k in ("n_rows", "strict_correct_not_content_correct")}, indent=1))
    for a, b in by_arm.items():
        print(f"{a}: strict {b['strict_em_pct']:.1f}, content {b['content_em_pct']:.1f}, placeholder "
              f"{b['placeholder_copy_pct']:.1f}%")
    print(f"done in {time.time() - t0:.0f} s")
    return meta


if __name__ == "__main__":
    run()
