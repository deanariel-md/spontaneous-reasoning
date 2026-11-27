r"""Continues the cut native-reasoning outputs to twice the cap. Runs on a GPU machine.

    python scripts/capcheck_generate.py --requests data/requests/capcheck/<key>.parquet --model-key <key> \
        --out-dir ~/d2_out/capcheck/<key> [--limit 30]

The input is the file of scripts/build_capcheck_requests.py: one row per sampled cut output (task, request_id,
messages_json, prompt_sha256, cut_text). For each row:
1. the native-reasoning prompt is rendered again by vLLM's chat path with the thinking switch on, as the runner did (one token is
   generated and dropped); its sha256 must equal the recorded prompt_sha256, else the job stops;
2. the prompt's token ids and the cut text's token ids (tokenized apart) are joined, and greedy decoding continues for
   up to --extra-tokens new tokens (30,720: 61,440 in all), with max_model_len --max-model-len (65,536).
Model, revision, dtype, tokenizer mode, engine settings and decoding are those of config/models_thinking_<task>.yaml and
config/decoding_thinking_<task>.yaml; the two tasks' entries must agree apart from max_model_len and the cap.
Outputs: <out-dir>/shard-<run>-<k>.parquet after every chunk (request_id, task, model_key, n_prompt_tokens,
n_cut_tokens_retok, continuation_text, n_cont_tokens, finish_reason) and run_log-<run>.json (versions, GPU, timings,
tokens). A restart skips the request ids already in a shard. --limit N takes the first N rows (the pilot). --part i/n
takes the rows whose position in the sorted file is i modulo n, so n jobs can share one checkpoint's rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gsm8k", "math")
STOP = {"flag": False}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_spec(key: str) -> tuple[dict, dict]:
    """(model entry, decoding) shared by the two tasks' native-reasoning configs; refuses when they differ beyond the cap."""
    ents, decs = {}, {}
    for t in TASKS:
        ents[t] = yaml.safe_load(open(ROOT / "config" / f"models_thinking_{t}.yaml", encoding="utf-8"))["dataset2"]["models"][key]
        decs[t] = yaml.safe_load(open(ROOT / "config" / f"decoding_thinking_{t}.yaml", encoding="utf-8"))
    a, b = ({k: v for k, v in e.items() if k != "max_model_len"} for e in ents.values())
    if a != b:
        raise SystemExit(f"{key}: the GSM8K and MATH entries differ beyond max_model_len")
    da, db = ({k: v for k, v in d.items() if k != "max_new_tokens"} for d in decs.values())
    if da != db:
        raise SystemExit("the two decoding files differ beyond the cap")
    if da.get("status") != "confirmed" or float(da["temperature"]) != 0.0:
        raise SystemExit("decoding is not the confirmed greedy setting")
    return a, da


def done_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in sorted(out_dir.glob("shard-*.parquet")):
        ids |= set(pd.read_parquet(p, columns=["request_id", "task"]).apply(lambda r: f"{r['task']}|{r['request_id']}", axis=1))
    return ids


def select_part(req: pd.DataFrame, part: str | None) -> pd.DataFrame:
    """Rows at positions i, i + n, i + 2n, ... of req for part "i/n"; all rows for None."""
    if part is None:
        return req
    i, n = (int(x) for x in part.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--part {part}: need 0 <= i < n")
    return req.iloc[i::n].reset_index(drop=True)


def gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-tokens", type=int, default=30720)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--part", default=None, help="i/n: every n-th row from row i")
    a = ap.parse_args(argv)

    signal.signal(signal.SIGUSR1, lambda *_: STOP.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    out_dir = Path(os.path.expanduser(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    req = pd.read_parquet(os.path.expanduser(a.requests))
    if set(req["model_key"]) != {a.model_key}:
        raise SystemExit(f"{a.requests} holds rows of {sorted(set(req['model_key']))}, not only {a.model_key}")
    req = req.sort_values(["task", "request_id"], kind="stable").reset_index(drop=True)
    if a.limit is not None:
        req = req.head(a.limit)
    req = select_part(req, a.part)
    have = done_ids(out_dir)
    todo = req[~(req["task"] + "|" + req["request_id"]).isin(have)].reset_index(drop=True)
    print(f"[capcheck] {a.model_key}: {len(req)} rows, {len(req) - len(todo)} done before, {len(todo)} to run", flush=True)
    if todo.empty:
        return

    ent, dec = load_spec(a.model_key)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if ent.get("kernels") not in (None, "default"):
        raise SystemExit(f"{a.model_key}: kernels {ent.get('kernels')!r}; this script runs the default kernels only")
    if ent["chat_template"]["source"] != "tokenizer_default":
        raise SystemExit(f"{a.model_key}: chat template from {ent['chat_template']['source']}; not handled here")
    import vllm
    from vllm import LLM, SamplingParams
    eng = dec["engine"]
    kw = dict(model=ent["hf_id"], revision=ent["revision"], tokenizer_revision=ent["revision"],
              tokenizer_mode=ent["tokenizer_mode"], dtype=ent["dtype"], tensor_parallel_size=ent["tensor_parallel_size"],
              max_model_len=a.max_model_len, seed=dec["seed"], trust_remote_code=False,
              gpu_memory_utilization=eng["gpu_memory_utilization"], enable_prefix_caching=eng["enable_prefix_caching"],
              enforce_eager=eng["enforce_eager"], generation_config=eng["generation_config"])
    if ent.get("quantization") not in (None, "none"):
        kw["quantization"] = ent["quantization"]
    for k in ("max_num_seqs", "max_num_batched_tokens"):
        if eng.get(k) is not None:
            kw[k] = eng[k]
    kw.update(ent.get("engine") or {})
    t_load = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    load_s = time.time() - t_load
    tmpl_kwargs = dict(ent["thinking_off"]["chat_template_kwargs"])

    # 1. the native-reasoning prompts again, through vLLM's chat path; their hashes must match the recorded ones
    one = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=0, max_tokens=1, skip_special_tokens=False)
    convs = [json.loads(s) for s in todo["messages_json"]]
    got = llm.chat(convs, [one] * len(convs), use_tqdm=False, chat_template_kwargs=tmpl_kwargs,
                   add_generation_prompt=True, continue_final_message=False)
    bad = [todo["request_id"][i] for i, ro in enumerate(got) if sha256_text(ro.prompt) != todo["prompt_sha256"][i]]
    if bad:
        raise SystemExit(f"[capcheck] {len(bad)} prompts render differently from the native-reasoning run, e.g. {bad[:3]}; stopping")
    prompt_ids = [list(ro.prompt_token_ids) for ro in got]
    cut_ids = [tok.encode(t, add_special_tokens=False) for t in todo["cut_text"]]
    print(f"[capcheck] prompts match the native-reasoning run ({len(got)}); re-tokenized cut lengths: min {min(map(len, cut_ids))}, "
          f"max {max(map(len, cut_ids))} (recorded {sorted(set(todo['cut_tokens']))})", flush=True)

    # 2. the continuations, chunk by chunk
    sp = SamplingParams(n=1, temperature=0.0, top_p=float(dec["top_p"]), top_k=int(dec["top_k"]),
                        max_tokens=a.extra_tokens, stop=dec["stop"] or None,
                        include_stop_str_in_output=bool(dec["include_stop_str_in_output"]),
                        skip_special_tokens=bool(dec["skip_special_tokens"]))
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + os.urandom(3).hex()
    log = {"run_id": run, "model_key": a.model_key, "hf_id": ent["hf_id"], "revision": ent["revision"],
           "requests": str(a.requests), "requests_sha256": hashlib.sha256(Path(os.path.expanduser(a.requests)).read_bytes()).hexdigest(),
           "limit": a.limit, "part": a.part, "extra_tokens": a.extra_tokens, "max_model_len": a.max_model_len, "engine": {k: str(v) for k, v in kw.items()},
           "sampling": {"temperature": 0.0, "top_p": sp.top_p, "top_k": sp.top_k, "max_tokens": sp.max_tokens,
                        "skip_special_tokens": sp.skip_special_tokens},
           "versions": {"vllm": vllm.__version__, "python": platform.python_version()},
           "gpu": gpu_name(), "host": platform.node(), "scheduler_job": os.environ.get("SLURM_JOB_ID"),
           "load_s": load_s, "chunks": []}
    try:
        import torch
        import transformers
        log["versions"].update({"torch": torch.__version__, "transformers": transformers.__version__})
    except Exception:  # noqa: BLE001
        pass
    for k0 in range(0, len(todo), a.chunk):
        if STOP["flag"]:
            print("[capcheck] signal received; stopping before the next chunk", flush=True)
            break
        idx = list(range(k0, min(k0 + a.chunk, len(todo))))
        t0 = time.time()
        res = llm.generate([{"prompt_token_ids": prompt_ids[i] + cut_ids[i]} for i in idx], [sp] * len(idx), use_tqdm=True)
        dt = time.time() - t0
        df = pd.DataFrame({
            "request_id": [todo["request_id"][i] for i in idx], "task": [todo["task"][i] for i in idx],
            "model_key": a.model_key, "n_prompt_tokens": [len(prompt_ids[i]) for i in idx],
            "n_cut_tokens_retok": [len(cut_ids[i]) for i in idx],
            "continuation_text": [r.outputs[0].text for r in res], "n_cont_tokens": [len(r.outputs[0].token_ids) for r in res],
            "finish_reason": [str(r.outputs[0].finish_reason) for r in res]})
        tmp = out_dir / f".shard-{run}-{k0 // a.chunk:05d}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_dir / f"shard-{run}-{k0 // a.chunk:05d}.parquet")
        log["chunks"].append({"n": len(idx), "seconds": dt, "new_tokens": int(df["n_cont_tokens"].sum()),
                              "prompt_tokens": int((df["n_prompt_tokens"] + df["n_cut_tokens_retok"]).sum()),
                              "finished_stop": int((df["finish_reason"] == "stop").sum())})
        print(f"[capcheck] chunk {k0 // a.chunk}: {len(idx)} outputs, {df['n_cont_tokens'].sum():,} new tokens in "
              f"{dt:.0f} s, {int((df['finish_reason'] == 'stop').sum())} stopped before the cap", flush=True)
        (out_dir / f"run_log-{run}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    log["generation_s"] = sum(c["seconds"] for c in log["chunks"])
    (out_dir / f"run_log-{run}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[capcheck] done: {sum(c['n'] for c in log['chunks'])} outputs this run", flush=True)


if __name__ == "__main__":
    sys.exit(main())
