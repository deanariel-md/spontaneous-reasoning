r"""Sample and request files of native reasoning under the vendors' sampling settings.

From the repository root, with the native-reasoning outputs in data/d2_out/thinking/<task>/<key>:
    .\.venv\Scripts\python.exe scripts\build_thinking_vendor_requests.py

Rule:
- checkpoints: the three of the cap check (more than 5% of native outputs cut on either task), read from
  manifests/thinking_capcheck_sample.json;
- configurations: per task, the first two entries of discovery_config_ids of the task's sampler (manifests/
  sampler_gsm8k.json, manifests/sampler_math.json), which the native-reasoning runs used;
- items: per task, 300 of the item ids of the native-reasoning request files, drawn uniformly without replacement from their
  sorted order (GSM8K ids as integers, MATH-500 ids as strings) with numpy.random.default_rng seeded by the first 8
  bytes of sha256("d20/vendor/<task>"); the same items for the three checkpoints;
- requests: the native-reasoning request rows of each checkpoint for those (configuration, item) pairs, messages unchanged, the
  request id with the suffix "|tv", and the recorded prompt_sha256, which scripts/thinking_vendor_generate.py
  checks before it generates.

Writes manifests/thinking_vendor_sample.json (rule, items, configurations, hashes; written before any request file and
never overwritten with a different sample), data/requests/thinking_vendor/<key>.parquet and
data/requests/thinking_vendor/build_record.json (each file's sha256 and count, and a projection of GPU-hours from the
native-reasoning output lengths on the same requests and the measured rates). Nothing here reads a correctness score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASKS = ("gsm8k", "math")
N_ITEMS = 300
N_CONFIGS = 2
SUFFIX = "|tv"
REQ_IN = os.path.join(ROOT, "data", "requests", "thinking")
OUT_IN = os.path.join(ROOT, "data", "d2_out", "thinking")
REQ_OUT = os.path.join(ROOT, "data", "requests", "thinking_vendor")
MANIFEST = os.path.join(ROOT, "manifests", "thinking_vendor_sample.json")
CAPCHECK = os.path.join(ROOT, "manifests", "thinking_capcheck_sample.json")
# measured generation rates of the native-reasoning runs on the RTX PRO 6000 (output tokens per second per card, from the run logs of
# data/d2_out/thinking/<task>/<key>)
RATES = {("gsm8k", "qwen3.5-35b-a3b"): 1104, ("gsm8k", "qwen3.6-35b-a3b"): 1151, ("gsm8k", "qwen3.6-27b"): 376,
         ("math", "qwen3.5-35b-a3b"): 1120, ("math", "qwen3.6-35b-a3b"): 891, ("math", "qwen3.6-27b"): 389}


def seed_for(task: str) -> int:
    return int.from_bytes(hashlib.sha256(f"d20/vendor/{task}".encode("utf-8")).digest()[:8], "big")


def sort_ids(ids, task: str) -> list[str]:
    """Item ids in the rule's order, as strings: GSM8K by integer value, MATH-500 as strings."""
    ids = sorted(set(map(str, ids)))
    return sorted(ids, key=int) if task == "gsm8k" else ids


def draw_items(ids, task: str, n: int = N_ITEMS) -> list[str]:
    """n of the item ids, uniform without replacement from their sorted order; returned in that order."""
    ids = sort_ids(ids, task)
    if len(ids) < n:
        raise ValueError(f"{task}: {len(ids)} items, fewer than {n}")
    pick = np.random.default_rng(seed_for(task)).choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def first_configs(sampler: dict, n: int = N_CONFIGS) -> list[str]:
    """The first n distinct entries of discovery_config_ids (the sampler draws with replacement)."""
    out: list[str] = []
    for c in map(str, sampler["discovery_config_ids"]):
        if c not in out:
            out.append(c)
        if len(out) == n:
            return out
    raise ValueError(f"fewer than {n} distinct discovery configurations")


def list_sha(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def file_sha(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def models() -> list[str]:
    return list(json.load(open(CAPCHECK, encoding="utf-8"))["models"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    from src import generation as gen
    keys = models()
    samplers = {t: json.load(open(os.path.join(ROOT, "manifests", f"sampler_{t}.json"), encoding="utf-8"))
                for t in TASKS}
    man = {"label": "thinking_vendor_sample", "study_part": "vendor sampling settings", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": ("per task: the first two distinct discovery_config_ids of the task's sampler; 300 item ids of item "
                    "I's request files drawn uniformly without replacement from their sorted order (GSM8K as integers, "
                    "MATH-500 as strings), numpy.random.default_rng seeded by the first 8 bytes of "
                    "sha256('d20/vendor/<task>'); the same items for every checkpoint"),
           "models": keys, "tasks": {}}
    reqs = {}
    for t in TASKS:
        cfgs = first_configs(samplers[t])
        ids_per_model = []
        for k in keys:
            r = pd.read_parquet(os.path.join(REQ_IN, t, f"{k}.parquet"))
            reqs[(t, k)] = r
            ids_per_model.append(set(r["item_id"].astype(str)))
        if any(s != ids_per_model[0] for s in ids_per_model):
            raise SystemExit(f"{t}: the native-reasoning request files hold different items for the checkpoints")
        items = draw_items(ids_per_model[0], t)
        man["tasks"][t] = {"sampler": f"manifests/sampler_{t}.json", "sampler_sha256": samplers[t]["sha256"],
                           "configurations": cfgs, "n_items_pool": len(ids_per_model[0]), "seed": seed_for(t),
                           "items": items, "items_sha256": list_sha(items)}
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST, encoding="utf-8"))
        if old.get("tasks") != json.loads(json.dumps(man["tasks"])) or old.get("models") != keys:
            raise SystemExit(f"{MANIFEST} holds a different sample; refusing to overwrite it")
        man["created"] = old.get("created", man["created"])
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)

    os.makedirs(REQ_OUT, exist_ok=True)
    rec = {"label": "thinking_vendor", "study_part": "vendor sampling settings", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "sample": {"path": "manifests/thinking_vendor_sample.json", "sha256": file_sha(MANIFEST)},
           "decoding": {"path": "config/decoding_thinking_vendor.yaml",
                        "sha256": file_sha(os.path.join(ROOT, "config", "decoding_thinking_vendor.yaml"))},
           "models": {}, "projection": {}}
    for k in keys:
        rows = []
        for t in TASKS:
            m = man["tasks"][t]
            r = reqs[(t, k)].copy()
            r["item_id"] = r["item_id"].astype(str)
            sel = r[r["config_id"].astype(str).isin(m["configurations"]) & r["item_id"].isin(set(m["items"]))].copy()
            if len(sel) != N_CONFIGS * N_ITEMS or sel.duplicated(["config_id", "item_id"]).any():
                raise SystemExit(f"{t}/{k}: {len(sel)} native-reasoning requests for the sample, expected {N_CONFIGS * N_ITEMS}")
            out = gen.read_outputs(os.path.join(OUT_IN, t, k))[["request_id", "prompt_sha256", "n_prompt_tokens",
                                                                  "n_output_tokens"]]
            out["request_id"] = out["request_id"].astype(str)
            sel["request_id"] = sel["request_id"].astype(str)
            sel = sel.merge(out.drop_duplicates("request_id"), on="request_id", how="left", validate="one_to_one")
            if sel["prompt_sha256"].isna().any():
                raise SystemExit(f"{t}/{k}: sampled requests without a native-reasoning output")
            sel["greedy_tokens"] = sel["n_output_tokens"].astype(int)
            rows.append(pd.DataFrame({
                "task": t, "request_id": sel["request_id"] + SUFFIX, "item_i_request_id": sel["request_id"],
                "model_key": k, "arm": f"thinking_vendor_{t}", "config_id": sel["config_id"].astype(str),
                "item_id": sel["item_id"].astype(str), "messages_json": sel["messages_json"].astype(str),
                "prompt_sha256": sel["prompt_sha256"].astype(str),
                "n_prompt_tokens": sel["n_prompt_tokens"].astype(int)}))
            tok = int(sel["greedy_tokens"].sum())
            rec["projection"][f"{t}/{k}"] = {"greedy_output_tokens": tok, "rate_tokens_per_s": RATES[(t, k)],
                                             "gpu_hours": round(tok / RATES[(t, k)] / 3600, 2)}
        df = pd.concat(rows, ignore_index=True).sort_values(["task", "request_id"], kind="stable").reset_index(drop=True)
        p = os.path.join(REQ_OUT, f"{k}.parquet")
        df.to_parquet(p, index=False)
        rec["models"][k] = {"requests_file": f"{k}.parquet", "requests_sha256": file_sha(p), "n": int(len(df)),
                            "by_task": {t: int((df["task"] == t).sum()) for t in TASKS},
                            "longest_prompt_tokens": int(df["n_prompt_tokens"].max())}
    rec["projection_total_gpu_hours"] = round(sum(v["gpu_hours"] for v in rec["projection"].values()), 1)
    rec["projection_note"] = ("the native-reasoning runs' greedy output tokens on the same requests over their measured rates per card; "
                              "sampling may change lengths, and model loading (about 20 minutes per job) is extra")
    with open(os.path.join(REQ_OUT, "build_record.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    for k, v in rec["models"].items():
        print(k, v["n"], v["by_task"])
    for k, v in rec["projection"].items():
        print(k, v)
    print("projected GPU-hours:", rec["projection_total_gpu_hours"])


if __name__ == "__main__":
    main()
