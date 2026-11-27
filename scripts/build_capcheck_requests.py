r"""Sample and request files of the cap check of the native-reasoning runs.

From the repository root, after the native-reasoning outputs are in data/d2_out/thinking/<task>/<key>:
    .\.venv\Scripts\python.exe scripts\build_capcheck_requests.py

For each checkpoint with more than 5% of its native outputs cut on either task, and each task, the eligible
outputs are those that stopped at the cap (finish_reason length) with no end-of-reasoning marker. 300 of them are
drawn uniformly without replacement from the sorted request ids, with numpy.random.default_rng seeded by the first 8
bytes of sha256("d18/<task>/<model_key>"). Writes:
- manifests/thinking_capcheck_sample.json: the rule, the counts and the sampled request ids, with a hash of each list;
- data/requests/capcheck/<key>.parquet: one row per sampled output (task, request_id, config_id, item_id,
  messages_json, prompt_sha256, the cut text and its recorded token count), the input of scripts/capcheck_generate.py.
Nothing here reads a correctness score; refuses to overwrite a different sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import generation as gen  # noqa: E402
from analyze_thinking_pilot import answer_text  # noqa: E402

TASKS = ("gsm8k", "math")
OUT_DIR = os.path.join(ROOT, "data", "d2_out", "thinking")
REQ_DIR = os.path.join(ROOT, "data", "requests", "thinking")
CC_DIR = os.path.join(ROOT, "data", "requests", "capcheck")
MANIFEST = os.path.join(ROOT, "manifests", "thinking_capcheck_sample.json")
N_SAMPLE = 300
CUT_THRESHOLD_PCT = 5.0
EXTRA_TOKENS = 30720
MAX_MODEL_LEN = 65536


def seed_for(task: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"d18/{task}/{key}".encode("utf-8")).digest()[:8], "big")


def draw(ids: list[str], task: str, key: str, n: int = N_SAMPLE) -> list[str]:
    """n of the ids, uniform without replacement, from their sorted order; all of them when there are fewer."""
    ids = sorted(ids)
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed_for(task, key))
    pick = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def cut_open(out: pd.DataFrame) -> pd.Series:
    """Stopped at the cap with no end-of-reasoning marker (no answer, counted wrong)."""
    closed = out["output_text"].astype(str).map(lambda t: answer_text(t) is not None)
    return (out["finish_reason"].astype(str) == "length") & ~closed


def list_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    a = ap.parse_args(argv)
    keys = sorted(os.listdir(os.path.join(OUT_DIR, TASKS[0])))
    outs, shares = {}, {}
    for task in TASKS:
        for key in keys:
            d = os.path.join(OUT_DIR, task, key)
            if not os.path.isdir(d):
                raise SystemExit(f"no folder {d}")
            out = gen.read_outputs(d)
            if out["request_id"].duplicated().any():
                raise SystemExit(f"{d}: duplicated request ids")
            out = out.assign(cut_open=cut_open(out))
            outs[(task, key)] = out
            shares[(task, key)] = 100 * float((out["finish_reason"].astype(str) == "length").mean())
    chosen = sorted(k for k in keys if any(shares[(t, k)] > CUT_THRESHOLD_PCT for t in TASKS))
    man = {"label": "thinking_capcheck_sample", "study_part": "cap check",
           "rule": (f"checkpoints with more than {CUT_THRESHOLD_PCT:g}% of native outputs cut on either task; eligible: "
                    "finish_reason length with no end-of-reasoning marker; per checkpoint and task "
                    f"{a.n} drawn uniformly without replacement from the sorted request ids, numpy.random.default_rng "
                    "seeded by the first 8 bytes of sha256('d18/<task>/<model_key>')"),
           "cut_pct": {f"{t}/{k}": round(shares[(t, k)], 4) for t in TASKS for k in keys},
           "models": chosen, "extra_tokens": EXTRA_TOKENS, "max_model_len": MAX_MODEL_LEN, "samples": {}}
    os.makedirs(CC_DIR, exist_ok=True)
    for key in chosen:
        rows = []
        for task in TASKS:
            out = outs[(task, key)]
            elig = out.loc[out["cut_open"], "request_id"].astype(str).tolist()
            ids = draw(elig, task, key, a.n)
            man["samples"][f"{task}/{key}"] = {"n_outputs": int(len(out)), "n_eligible": len(elig),
                                               "n_sampled": len(ids), "seed": seed_for(task, key),
                                               "eligible_sha256": list_sha(sorted(elig)), "sampled_sha256": list_sha(ids),
                                               "request_ids": ids}
            req = pd.read_parquet(os.path.join(REQ_DIR, task, f"{key}.parquet"), columns=["request_id", "messages_json"])
            req["request_id"] = req["request_id"].astype(str)
            sel = out[out["request_id"].astype(str).isin(set(ids))].copy()
            sel["request_id"] = sel["request_id"].astype(str)
            sel = sel.merge(req, on="request_id", how="left", validate="one_to_one")
            if sel["messages_json"].isna().any() or len(sel) != len(ids):
                raise SystemExit(f"{task}/{key}: sampled outputs without their request")
            rows.append(pd.DataFrame({
                "task": task, "request_id": sel["request_id"], "model_key": key, "arm": sel["arm"].astype(str),
                "config_id": sel["config_id"].astype(str), "item_id": sel["item_id"].astype(str),
                "messages_json": sel["messages_json"].astype(str), "prompt_sha256": sel["prompt_sha256"].astype(str),
                "cut_text": sel["output_text"].astype(str), "cut_tokens": sel["n_output_tokens"].astype(int)}))
        df = pd.concat(rows, ignore_index=True).sort_values(["task", "request_id"], kind="stable").reset_index(drop=True)
        path = os.path.join(CC_DIR, f"{key}.parquet")
        df.to_parquet(path, index=False)
        man["samples"].setdefault("files", {})[key] = {"path": os.path.relpath(path, ROOT).replace("\\", "/"),
                                                       "n": int(len(df))}
    files = man["samples"].pop("files")
    man["request_files"] = files
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST, encoding="utf-8"))
        if old.get("samples") != json.loads(json.dumps(man["samples"])):
            raise SystemExit(f"{MANIFEST} holds a different sample; refusing to overwrite it")
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    for k, v in man["samples"].items():
        print(k, v["n_outputs"], v["n_eligible"], v["n_sampled"])
    print("models:", chosen)


if __name__ == "__main__":
    main()
