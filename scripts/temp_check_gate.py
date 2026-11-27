r"""In-job gate of the temperature check's pilot (scripts/temp_check.py).

Runs on the GPU machine at the end of the pilot job, after the greedy and the sampled pilot of one model:
    python scripts/temp_check_gate.py --model-key <key> --work <work dir>
Exit 0 when the pilot passes, 3 when it fails; the full run of the model depends on the pilot job's success, so a failed gate cancels it.

Checks, against the reference built locally from the panel's greedy MATH-500 run (data/requests/temp_check/pilot_ref/
<key>.parquet: request_id, prompt_sha256, output_text of the greedy outputs; <key>.json: the greedy run's model block
and chat template sha256):
  1. both pilot runs complete, one output per request;
  2. every prompt sha256 of both runs equals the greedy run's (same messages, chat template and prefill);
  3. both run logs carry the greedy run's model block and chat template, no prompt-render mismatch;
  4. the greedy pilot's decoding has temperature 0; the sampled one temperature 0.7 and sampling {seed: request};
     their per-arm parameters (caps, prefill, stop rules) are equal;
  5. at least 90% of the greedy pilot's outputs equal the stored greedy outputs (batch-invariant kernels on the same
     card model reproduce them; every mismatch is listed in the gate file);
  6. every sampled output is non-empty.
Writes out/pilot_gate/<key>.json. pandas and pyarrow only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

MIN_GREEDY_IDENTICAL = 0.90
COLS = ["request_id", "arm", "prompt_sha256", "output_text", "finish_reason", "n_output_tokens"]


def read(folder: str):
    shards = sorted(glob.glob(os.path.join(folder, "shard-*.parquet")))
    logs = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(folder, "run_log-*.json")))]
    df = pd.concat([pd.read_parquet(s, columns=COLS) for s in shards], ignore_index=True) if shards else \
        pd.DataFrame(columns=COLS)
    return df, logs


def arm_params(dec: dict) -> dict:
    """Per arm: the run log's decoding parameters other than the sampling ones."""
    out = {}
    for arm, cap in dec["max_new_tokens"].items():
        ov = (dec.get("arm_overrides") or {}).get(arm) or {}
        out[arm] = {"max_new_tokens": cap, "prefill": (dec.get("prefill") or {}).get(arm),
                    "stop": ov.get("stop", dec.get("stop")), "top_p": dec.get("top_p"), "top_k": dec.get("top_k"),
                    "include_stop_str_in_output": ov.get("include_stop_str_in_output", dec.get("include_stop_str_in_output")),
                    "skip_special_tokens": dec.get("skip_special_tokens")}
    return out


def gate(key: str, work: str) -> dict:
    ref = pd.read_parquet(os.path.join(work, "data", "requests", "temp_check", "pilot_ref", f"{key}.parquet"))
    meta = json.load(open(os.path.join(work, "data", "requests", "temp_check", "pilot_ref", f"{key}.json"), encoding="utf-8"))
    checks = []

    def check(name, ok, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    runs = {}
    for kind in ("pilot_greedy", "pilot"):
        df, logs = read(os.path.join(work, "out", kind, key))
        done = [g for g in logs if g.get("status") == "complete"]
        runs[kind] = (df, done[-1] if done else None)
        check(f"{kind}: complete, one output per request", bool(done) and len(df) == len(ref)
              and set(df["request_id"]) == set(ref["request_id"]),
              {"rows": int(len(df)), "want": int(len(ref)), "statuses": [g.get("status") for g in logs]})
        if df.empty or not done:
            continue
        lg = done[-1]
        j = df.merge(ref, on="request_id", suffixes=("", "_ref"))
        check(f"{kind}: prompt sha256 equal to the greedy run's", (j["prompt_sha256"] == j["prompt_sha256_ref"]).all(),
              int((j["prompt_sha256"] != j["prompt_sha256_ref"]).sum()))
        check(f"{kind}: model block equal to the greedy run's", lg.get("model") == meta["model"],
              {k: [lg.get("model", {}).get(k), meta["model"].get(k)] for k in set(meta["model"]) | set(lg.get("model", {}))
               if lg.get("model", {}).get(k) != meta["model"].get(k)})
        check(f"{kind}: chat template equal to the greedy run's",
              (lg.get("chat_template") or {}).get("observed_sha256") == meta["chat_template_sha256"],
              (lg.get("chat_template") or {}).get("observed_sha256"))
        check(f"{kind}: no prompt-render mismatch", lg.get("prompt_render_mismatches") == 0, lg.get("prompt_render_mismatches"))
        check(f"{kind}: GPU model of the greedy run", lg["gpu"]["devices"][0]["name"] == meta["gpu"], lg["gpu"]["devices"][0]["name"])
        if kind == "pilot_greedy":
            same = j["output_text"] == j["output_text_ref"]
            mism = j.loc[~same, ["request_id", "arm"]].to_dict("records")
            check(f"pilot_greedy: at least {MIN_GREEDY_IDENTICAL:.0%} of outputs equal the stored greedy outputs",
                  same.mean() >= MIN_GREEDY_IDENTICAL, {"identical": int(same.sum()), "rows": int(len(j)), "mismatches": mism})
        else:
            check("pilot: every sampled output non-empty", (df["output_text"].str.len() > 0).all(),
                  int((df["output_text"].str.len() == 0).sum()))
    g, s = runs["pilot_greedy"][1], runs["pilot"][1]
    if g is not None and s is not None:
        dg, ds = g["decoding"], s["decoding"]
        check("greedy pilot: temperature 0, no sampling block", dg.get("temperature") == 0.0 and "sampling" not in dg,
              {k: dg.get(k) for k in ("temperature", "sampling")})
        check("sampled pilot: temperature 0.7, seed per request",
              ds.get("temperature") == 0.7 and ds.get("sampling") == {"seed": "request"},
              {k: ds.get(k) for k in ("temperature", "sampling")})
        check("per-arm parameters equal (caps, prefill, stop rules, top_p, top_k)", arm_params(dg) == arm_params(ds),
              [arm_params(dg), arm_params(ds)])
        check("versions equal to the greedy run's", {p: s["versions"].get(p) for p in ("vllm", "torch", "transformers")}
              == meta["versions"], [{p: s["versions"].get(p) for p in ("vllm", "torch", "transformers")}, meta["versions"]])
    res = {"model_key": key, "passed": all(c["ok"] for c in checks), "checks": checks}
    os.makedirs(os.path.join(work, "out", "pilot_gate"), exist_ok=True)
    with open(os.path.join(work, "out", "pilot_gate", f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--work", default=".")
    a = ap.parse_args(argv)
    r = gate(a.model_key, a.work)
    for c in r["checks"]:
        print(f"[gate] {'ok  ' if c['ok'] else 'FAIL'} {c['name']}")
    print(f"[gate] {a.model_key}: {'PASSED' if r['passed'] else 'FAILED'}")
    return 0 if r["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
