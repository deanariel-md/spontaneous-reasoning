r"""Scoring of the thinking-on runs (native reasoning mode on).

From the repository root, after the folders data/d2_out/thinking/<task>/<key> are copied back:
    python scripts/score_thinking.py --task gsm8k
    python scripts/score_thinking.py --task math

The build record of the task's thinking requests is derived from the collection's records (the same bank, sampler and
items; each model's thinking request file and hash) and written to data/requests/thinking/<task>/build_record.json.
Each output is read after the last end-of-reasoning marker (`</think>` for Qwen, `<channel|>` for Gemma-4); an output
cut at the cap before the marker has no answer and counts as wrong; an output that closes no reasoning block and was
not cut is scored whole. The scorer of the task scores what is read: data/processed/thinking/
<task>_scored.parquet (strict, lenient, compliant; n_output_tokens and finish_reason are the whole output's). Beside
it, data/processed/thinking/<task>_closure.parquet holds closed per output (a marker is present). The sidecar holds
no accuracy.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import d2_io  # noqa: E402
from analyze_thinking_pilot import answer_text  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "d2_out", "thinking")
REQ = os.path.join(ROOT, "data", "requests", "thinking")
PROCESSED = os.path.join(ROOT, "data", "processed", "thinking")


def arm(task):
    return f"thinking_{task}"


def thinking_record():
    return json.load(open(os.path.join(REQ, "build_record.json"), encoding="utf-8"))


def derived_record(task):
    """The collection's build record with each model's thinking request file in place of its own (module doc)."""
    tr = thinking_record()["tasks"][task]["models"]
    sources = {}
    for k, ent in tr.items():
        sources.setdefault(os.path.join(ROOT, os.path.dirname(ent["source"]), "build_record.json"), []).append(k)
    base, base_sig, models = None, None, {}
    for p, keys in sorted(sources.items()):
        rec = json.load(open(p, encoding="utf-8"))
        sig = (rec["kind"], rec.get("split", "test"), rec["bank"]["sha256"], rec["sampler"]["registered_sha256"],
               rec["items"]["canonical_sha256"])
        if base is None:
            base, base_sig = rec, sig
        elif sig != base_sig:
            raise SystemExit(f"{d2_io.rel_path(p)}: its bank, sampler or items differ from the other collection records")
        for k in keys:
            src, ent = rec["models"][k], tr[k]
            if src["requests_file"] != os.path.basename(ent["source"]):
                raise SystemExit(f"{k}: the thinking requests were built from {ent['source']}, not from the request "
                                 f"file of {d2_io.rel_path(p)}")
            models[k] = {"n_requests": ent["n"], "n_configurations": ent["configs"], "counts_by_arm": {arm(task): ent["n"]},
                         "requests_file": f"{task}/{ent['requests_file']}", "requests_sha256": ent["requests_sha256"],
                         "registered": src.get("registered", True),
                         "derived_from": {"record": d2_io.rel_path(p), "requests_file": src["requests_file"],
                                          "requests_sha256": src["requests_sha256"]}}
    rec = {k: v for k, v in base.items() if k not in ("models", "_path")}
    rec.update({"arms": [arm(task)], "purpose": "thinking_collection", "models": models,
                "derived": ("the collection's direct-answer requests with the thinking switch on "
                            "(scripts/build_thinking_requests.py); bank, sampler and items as in "
                            + ", ".join(sorted(d2_io.rel_path(p) for p in sources)))})
    path = os.path.join(REQ, task, "build_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return path, rec


def answer_bodies(texts, finish):
    """What the scorer reads (module doc)."""
    out, closed = [], []
    for t, f in zip(texts, finish):
        a = answer_text(t)
        closed.append(a is not None)
        out.append(a if a is not None else ("" if f == "length" else t))
    return out, np.array(closed, bool)


def scorer(task):
    """(score_fn on outputs whose text is already the answer body, scorer record, items_info, split)."""
    if task == "math":
        from src import string_tasks
        tm = string_tasks.module("math")
        kind = d2_io.KIND_MAIN
        items, items_info = tm.load_items(tm.KIND_ITEM_SET[kind])
        return tm.score_fn(items), tm.load_scorer(None), items_info, tm.KIND_SPLIT[kind]
    from src import gsm8k, scoring
    spec, record = d2_io.load_registered_scorer(os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    items, items_info = d2_io.load_items(split="test")
    gold = gsm8k.gold_lookup(items)
    return (lambda out: scoring.score_outputs(out, gold, spec)), record, items_info, "test"


def score_task(task, allow_missing=False):
    import score_d2
    path, rec = derived_record(task)
    keys = sorted(rec["models"])
    dirs = [os.path.join(OUT_DIR, task, k) for k in keys]
    missing = [k for k, d in zip(keys, dirs) if not os.path.isdir(d)]
    if missing and not allow_missing:
        raise SystemExit(f"{task}: no thinking folder for {missing}")
    dirs = [d for d in dirs if os.path.isdir(d)]
    fn, record, items_info, split = scorer(task)
    closure = []

    def score_fn(out):
        out = out.copy()
        body, closed = answer_bodies(out["output_text"].astype(str).tolist(), out["finish_reason"].astype(str).tolist())
        closure.append(pd.DataFrame({"model_key": out["model_key"].astype(str), "arm": out["arm"].astype(str),
                                     "config_id": out["config_id"].astype(str), "item_id": out["item_id"],
                                     "closed": closed}))
        out["output_text"] = body
        return fn(out)
    table, infos = d2_io.score_work_dirs(dirs, None, None, records=d2_io.load_build_records(os.path.dirname(path)),
                                         tranche=False, split=split, allow_incomplete=False,
                                         score_fn=score_fn, string_item_ids=(task == "math"))
    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, f"{task}_scored.parquet")
    # --allow-missing leaves out the folders not yet copied back; each folder scored must still hold a
    # complete run. It is not passed on as --allow-incomplete, which would mark every model incomplete, so the
    # analysis's --partial diagnostic can read the table. The analysis without --partial still refuses a table that
    # lacks a panel model.
    args = Namespace(work_dir=dirs, allow_test_backend=False, allow_incomplete=False,
                     allow_unmatched_requests=False)
    res = score_d2._write(args, table, infos, record, items_info, d2_io.KIND_MAIN, split, out,
                          os.path.dirname(path), task=task)
    pd.concat(closure, ignore_index=True).to_parquet(os.path.join(PROCESSED, f"{task}_closure.parquet"), index=False)
    print(json.dumps({k: res[k] for k in ("out", "n_rows", "models", "scorer_version")}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=("gsm8k", "math"), required=True)
    ap.add_argument("--allow-missing", action="store_true",
                    help="score the folders present, each a complete run (a table for run_thinking.py --partial; "
                         "the analysis without --partial refuses an incomplete panel)")
    a = ap.parse_args(argv)
    score_task(a.task, a.allow_missing)


if __name__ == "__main__":
    main()
