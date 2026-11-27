r"""Native reasoning under the vendors' sampling settings. Runs on a GPU machine.

    python scripts/thinking_vendor_generate.py --requests data/requests/thinking_vendor/<key>.parquet \
        --model-key <key> --out-dir ~/d2_out/thinking_vendor/<key> [--limit 16] [--part i/n]

The input is the file of scripts/build_thinking_vendor_requests.py: the greedy thinking-on run's requests of the sample (task, request_id,
config_id, item_id, messages_json, prompt_sha256). For each row:
1. the prompt is rendered by vLLM's chat path with the thinking switch on, as the greedy thinking-on runner did (one token is
   generated and dropped); its sha256 must equal the recorded prompt_sha256, else the job stops;
2. those prompt token ids are sampled with the checkpoint's settings in config/decoding_thinking_vendor.yaml
   (temperature, top_p, top_k, min_p, presence and repetition penalty; seed 0 per request), up to 30,720 new tokens.
Model, revision, dtype, tokenizer mode, chat template and engine settings are the greedy thinking-on run's (config/models_thinking_<task>.yaml;
the two tasks' entries must agree apart from max_model_len), with the max_model_len of the vendor file.
Outputs: <out-dir>/shard-<run>-<k>.parquet after every chunk (request_id, task, model_key, arm, config_id, item_id,
prompt_sha256, output_text, n_prompt_tokens, n_output_tokens, finish_reason) and run_log-<run>.json (versions, GPU,
sampling, timings, tokens). A restart skips the request ids already in a shard. --limit N takes the first N rows (the
pilot); --part i/n the rows whose position in the sorted file is i modulo n, so n jobs can share one checkpoint.
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
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_vendor(path: Path | str = ROOT / "config" / "decoding_thinking_vendor.yaml") -> dict:
    v = yaml.safe_load(open(path, encoding="utf-8"))
    if v.get("status") != "confirmed":
        raise SystemExit(f"{path}: status is not confirmed")
    for k, m in (v.get("models") or {}).items():
        miss = [s for s in SAMPLING_KEYS if s not in m]
        if miss:
            raise SystemExit(f"{path}: {k} lacks {miss}")
        if not float(m["temperature"]) > 0:
            raise SystemExit(f"{path}: {k} has temperature {m['temperature']}; the vendors' settings sample")
    return v


def sampling_kwargs(vendor: dict, key: str) -> dict:
    """SamplingParams arguments of one checkpoint (module doc)."""
    if key not in vendor["models"]:
        raise SystemExit(f"{key}: no entry in config/decoding_thinking_vendor.yaml")
    m = vendor["models"][key]
    kw = {s: (int(m[s]) if s == "top_k" else float(m[s])) for s in SAMPLING_KEYS}
    kw.update(n=1, seed=int(vendor["seed"]), max_tokens=int(vendor["max_new_tokens"]), stop=list(vendor["stop"]) or None,
              skip_special_tokens=bool(vendor["skip_special_tokens"]))
    return kw


def load_entry(key: str) -> dict:
    """The greedy thinking-on run's registry entry of the checkpoint, shared by the two tasks apart from max_model_len."""
    ents = {t: yaml.safe_load(open(ROOT / "config" / f"models_thinking_{t}.yaml", encoding="utf-8"))["dataset2"]["models"][key]
            for t in TASKS}
    a, b = ({k: v for k, v in e.items() if k != "max_model_len"} for e in ents.values())
    if a != b:
        raise SystemExit(f"{key}: the GSM8K and MATH entries differ beyond max_model_len")
    return a


def engine_settings(key: str) -> dict:
    """The greedy thinking-on run's engine block (the two decoding files must agree on it)."""
    eng = [yaml.safe_load(open(ROOT / "config" / f"decoding_thinking_{t}.yaml", encoding="utf-8"))["engine"] for t in TASKS]
    if eng[0] != eng[1]:
        raise SystemExit("the two thinking decoding files differ in their engine settings")
    return eng[0]


def done_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in sorted(out_dir.glob("shard-*.parquet")):
        ids |= set(pd.read_parquet(p, columns=["request_id"])["request_id"].astype(str))
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
    ap.add_argument("--chunk", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--part", default=None, help="i/n: every n-th row from row i")
    a = ap.parse_args(argv)

    signal.signal(signal.SIGUSR1, lambda *_: STOP.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    out_dir = Path(os.path.expanduser(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = Path(os.path.expanduser(a.requests))
    req = pd.read_parquet(req_path)
    if set(req["model_key"]) != {a.model_key}:
        raise SystemExit(f"{a.requests} holds rows of {sorted(set(req['model_key']))}, not only {a.model_key}")
    req = req.sort_values(["task", "request_id"], kind="stable").reset_index(drop=True)
    if a.limit is not None:
        req = req.head(a.limit)
    req = select_part(req, a.part)
    have = done_ids(out_dir)
    todo = req[~req["request_id"].astype(str).isin(have)].reset_index(drop=True)
    print(f"[vendor] {a.model_key}: {len(req)} rows, {len(req) - len(todo)} done before, {len(todo)} to run", flush=True)
    if todo.empty:
        return 0

    vendor = load_vendor()
    skw = sampling_kwargs(vendor, a.model_key)
    ent, eng = load_entry(a.model_key), engine_settings(a.model_key)
    if ent.get("kernels") not in (None, "default"):
        raise SystemExit(f"{a.model_key}: kernels {ent.get('kernels')!r}; this script runs the default kernels only")
    if ent["chat_template"]["source"] != "tokenizer_default":
        raise SystemExit(f"{a.model_key}: chat template from {ent['chat_template']['source']}; not handled here")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import vllm
    from vllm import LLM, SamplingParams
    kw = dict(model=ent["hf_id"], revision=ent["revision"], tokenizer_revision=ent["revision"],
              tokenizer_mode=ent["tokenizer_mode"], dtype=ent["dtype"], tensor_parallel_size=ent["tensor_parallel_size"],
              max_model_len=int(vendor["max_model_len"]), seed=int(vendor["seed"]), trust_remote_code=False,
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
    load_s = time.time() - t_load
    tmpl_kwargs = dict(ent["thinking_off"]["chat_template_kwargs"])
    if tmpl_kwargs.get("enable_thinking") is not True:
        raise SystemExit(f"{a.model_key}: the thinking entry does not switch thinking on")

    # 1. the greedy run's prompts again, through vLLM's chat path; their hashes must match the recorded ones
    one = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=0, max_tokens=1, skip_special_tokens=False)
    convs = [json.loads(s) for s in todo["messages_json"]]
    got = llm.chat(convs, [one] * len(convs), use_tqdm=False, chat_template_kwargs=tmpl_kwargs,
                   add_generation_prompt=True, continue_final_message=False)
    bad = [todo["request_id"][i] for i, ro in enumerate(got) if sha256_text(ro.prompt) != todo["prompt_sha256"][i]]
    if bad:
        raise SystemExit(f"[vendor] {len(bad)} prompts render differently from the greedy thinking run, e.g. {bad[:3]}; stopping")
    prompt_ids = [list(ro.prompt_token_ids) for ro in got]
    print(f"[vendor] prompts match the greedy thinking run ({len(got)}); sampling {json.dumps({k: skw[k] for k in SAMPLING_KEYS})}",
          flush=True)

    # 2. the sampled outputs, chunk by chunk
    sp = SamplingParams(**skw)
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + os.urandom(3).hex()
    log = {"run_id": run, "model_key": a.model_key, "hf_id": ent["hf_id"], "revision": ent["revision"],
           "requests": str(a.requests), "requests_sha256": hashlib.sha256(req_path.read_bytes()).hexdigest(),
           "vendor_yaml_sha256": hashlib.sha256((ROOT / "config" / "decoding_thinking_vendor.yaml").read_bytes()).hexdigest(),
           "limit": a.limit, "part": a.part, "engine": {k: str(v) for k, v in kw.items()},
           "sampling": {k: (v if not isinstance(v, list) else list(v)) for k, v in skw.items()},
           "chat_template_kwargs": tmpl_kwargs,
           "versions": {"vllm": vllm.__version__, "python": platform.python_version()},
           "gpu": gpu_name(), "host": platform.node(), "scheduler_job": os.environ.get("SLURM_JOB_ID"),
           "load_s": load_s, "chunks": [], "status": "running"}
    try:
        import torch
        import transformers
        log["versions"].update({"torch": torch.__version__, "transformers": transformers.__version__})
    except Exception:  # noqa: BLE001
        pass
    for k0 in range(0, len(todo), a.chunk):
        if STOP["flag"]:
            print("[vendor] signal received; stopping before the next chunk", flush=True)
            log["status"] = "interrupted"
            break
        idx = list(range(k0, min(k0 + a.chunk, len(todo))))
        t0 = time.time()
        res = llm.generate([{"prompt_token_ids": prompt_ids[i]} for i in idx], [sp] * len(idx), use_tqdm=True)
        dt = time.time() - t0
        df = pd.DataFrame({
            "request_id": [todo["request_id"][i] for i in idx], "task": [todo["task"][i] for i in idx],
            "model_key": a.model_key, "arm": [todo["arm"][i] for i in idx],
            "config_id": [todo["config_id"][i] for i in idx], "item_id": [str(todo["item_id"][i]) for i in idx],
            "prompt_sha256": [todo["prompt_sha256"][i] for i in idx],
            "output_text": [r.outputs[0].text for r in res], "n_prompt_tokens": [len(prompt_ids[i]) for i in idx],
            "n_output_tokens": [len(r.outputs[0].token_ids) for r in res],
            "finish_reason": [str(r.outputs[0].finish_reason) for r in res]})
        tmp = out_dir / f".shard-{run}-{k0 // a.chunk:05d}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_dir / f"shard-{run}-{k0 // a.chunk:05d}.parquet")
        log["chunks"].append({"n": len(idx), "seconds": dt, "new_tokens": int(df["n_output_tokens"].sum()),
                              "prompt_tokens": int(df["n_prompt_tokens"].sum()),
                              "finished_length": int((df["finish_reason"] == "length").sum())})
        print(f"[vendor] chunk {k0 // a.chunk}: {len(idx)} outputs, {df['n_output_tokens'].sum():,} new tokens in "
              f"{dt:.0f} s, {int((df['finish_reason'] == 'length').sum())} at the cap", flush=True)
        (out_dir / f"run_log-{run}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    else:
        log["status"] = "complete" if a.limit is None and a.part is None else \
            ("pilot_complete" if a.limit is not None else "part_complete")
    log["generation_s"] = sum(c["seconds"] for c in log["chunks"])
    (out_dir / f"run_log-{run}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[vendor] {log['status']}: {sum(c['n'] for c in log['chunks'])} outputs this run", flush=True)
    return 0 if log["status"] != "interrupted" else 75


if __name__ == "__main__":
    sys.exit(main())
