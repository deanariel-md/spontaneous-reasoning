r"""Scoring of native reasoning under the vendors' sampling settings.

From the repository root, after the folders data/d2_out/thinking_vendor/<key> are copied back:
    python scripts/score_thinking_vendor.py [--allow-missing]

For each checkpoint of data/requests/thinking_vendor/build_record.json, the vendor outputs and the greedy thinking-on
outputs of the same 1,200 requests are read the same way: the text after the last end-of-reasoning marker; an output cut
at the cap before the marker has no answer and counts as wrong; one that closes no reasoning block and was not cut is
scored whole. The scorer of the task scores what is read (strict and lenient). Per output it also records
closed (a marker is present), cut_open (cut at the cap with no marker) and looping (the looping rule applied to the
output's text).
Writes data/processed/thinking_vendor/scored.parquet, one row per request and decoding (vendor, greedy).
A request without its vendor output refuses unless --allow-missing (then the row is left out and counted).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQ = os.path.join(ROOT, "data", "requests", "thinking_vendor")
VENDOR_OUT = os.path.join(ROOT, "data", "d2_out", "thinking_vendor")
GREEDY_OUT = os.path.join(ROOT, "data", "d2_out", "thinking")
PROCESSED = os.path.join(ROOT, "data", "processed", "thinking_vendor")
TASKS = ("gsm8k", "math")


def read_shards(d: str) -> pd.DataFrame:
    parts = [pd.read_parquet(os.path.join(d, p)) for p in sorted(os.listdir(d)) if p.startswith("shard-")] \
        if os.path.isdir(d) else []
    if not parts:
        return pd.DataFrame(columns=["request_id", "output_text", "n_output_tokens", "finish_reason"])
    df = pd.concat(parts, ignore_index=True)
    df["request_id"] = df["request_id"].astype(str)
    if df["request_id"].duplicated().any():
        raise SystemExit(f"{d}: a request id appears in two shards")
    return df


def flags(texts, finish) -> pd.DataFrame:
    """body (what the scorer reads), closed, cut_open and looping per output (module doc)."""
    from score_thinking import answer_bodies
    from run_capcheck import looping
    texts, finish = [str(t) for t in texts], [str(f) for f in finish]
    body, closed = answer_bodies(texts, finish)
    cut = [(f == "length") and not c for f, c in zip(finish, closed)]
    return pd.DataFrame({"body": body, "closed": closed, "cut_open": cut, "looping": [looping(t) for t in texts]})


def score_bodies(task: str, rows: pd.DataFrame) -> pd.DataFrame:
    """strict_correct and lenient_correct of rows whose output_text is already the body."""
    from run_capcheck import score_rows
    return score_rows(task, rows)


def assemble(req: pd.DataFrame, vendor: pd.DataFrame, greedy: pd.DataFrame, allow_missing: bool = False):
    """Rows of both decodings for one checkpoint (unscored), and the number of requests without a vendor output."""
    req = req.copy()
    req["request_id"] = req["request_id"].astype(str)
    v = req.merge(vendor[["request_id", "output_text", "n_output_tokens", "finish_reason"]], on="request_id", how="left")
    missing = int(v["output_text"].isna().sum())
    if missing and not allow_missing:
        raise SystemExit(f"{req['model_key'].iloc[0]}: {missing} requests without a vendor output")
    v = v[v["output_text"].notna()]
    g = req.drop(columns=["request_id"]).rename(columns={"item_i_request_id": "request_id"})
    g = g.merge(greedy[["request_id", "output_text", "n_output_tokens", "finish_reason"]], on="request_id", how="left")
    if g["output_text"].isna().any():
        raise SystemExit(f"{req['model_key'].iloc[0]}: greedy thinking outputs missing for {int(g['output_text'].isna().sum())} requests")
    cols = ["task", "model_key", "config_id", "item_id", "request_id", "output_text", "n_output_tokens", "finish_reason"]
    return pd.concat([v[cols].assign(decoding="vendor"), g[cols].assign(decoding="greedy")], ignore_index=True), missing


def run(allow_missing: bool = False) -> dict:
    from src import generation as gen
    rec = json.load(open(os.path.join(REQ, "build_record.json"), encoding="utf-8"))
    frames, missing = [], {}
    for k in rec["models"]:
        req = pd.read_parquet(os.path.join(REQ, f"{k}.parquet"))
        vendor = read_shards(os.path.join(VENDOR_OUT, k))
        g = []
        for t in TASKS:
            o = gen.read_outputs(os.path.join(GREEDY_OUT, t, k))
            o["request_id"] = o["request_id"].astype(str)
            g.append(o)
        rows, missing[k] = assemble(req, vendor, pd.concat(g, ignore_index=True), allow_missing)
        frames.append(rows)
    df = pd.concat(frames, ignore_index=True)
    fl = flags(df["output_text"].tolist(), df["finish_reason"].tolist())
    df = pd.concat([df.reset_index(drop=True), fl], axis=1)
    scored = []
    for t in TASKS:
        sub = df[df["task"] == t].reset_index(drop=True)
        if sub.empty:
            continue
        rows = sub.assign(output_text=sub["body"], arm="thinking", request_id=sub["decoding"] + "|" + sub["model_key"] + "|" + sub["request_id"])
        sc = score_bodies(t, rows[["request_id", "model_key", "arm", "config_id", "item_id", "output_text",
                                   "finish_reason", "n_output_tokens"]])
        scored.append(sub.assign(strict_correct=sc["strict_correct"].astype(int).to_numpy(),
                                 lenient_correct=sc["lenient_correct"].astype(int).to_numpy()))
    out = pd.concat(scored, ignore_index=True).drop(columns=["output_text", "body"])
    os.makedirs(PROCESSED, exist_ok=True)
    path = os.path.join(PROCESSED, "scored.parquet")
    out.to_parquet(path, index=False)
    res = {"out": os.path.relpath(path, ROOT).replace("\\", "/"), "n_rows": int(len(out)), "missing_vendor": missing}
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-missing", action="store_true",
                    help="leave out requests without a vendor output (a diagnostic; the analysis reports the count)")
    a = ap.parse_args(argv)
    run(a.allow_missing)


if __name__ == "__main__":
    main()
