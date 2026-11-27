r"""Request files for the forced-direct arm.

From the repository root:
    .\.venv\Scripts\python.exe scripts\build_forced_direct.py

For each model of the GSM8K-29 panel (results/dataset2/gsm8k30.json) and the BIG-Bench Hard-28 panel
(results/dataset2/bbh.json), the collection's own request file is read and its direct-answer requests are kept
unchanged: the same configurations (all 48 draws of the frozen samplers), items and messages. Only the arm label
becomes forced_direct and the request id gains the suffix "|fd". The prefill and the 16-token cap live in
config/decoding_forced.yaml, so the rendered prompt differs from the requested-direct prompt only by the opened
assistant turn.

Outputs: data/requests/forced/<task>/<key>.parquet and data/requests/forced/build_record.json (each source file with
its sha256, each output with its sha256 and request count).

MATH-500: `--task math` derives the forced requests of the
GSM8K-29 panel's models that the MATH screen retained (manifests/math_admissibility.json) from
data/requests/math/<key>.parquet (scripts/build_requests.py --task math) by the same rule, and
adds them to the build record under tasks.math; the entries of the other tasks are kept. They run with
config/decoding_forced_math.yaml, which has no stop string because MATH answers may hold braces.
    .\.venv\Scripts\python.exe scripts\build_forced_direct.py --task math
"""
import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "data", "requests", "forced")
ARM = "forced_direct"


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def source(task, key):
    if task == "math":
        return os.path.join(ROOT, "data", "requests", "math", f"{key}.parquet")
    if task == "bbh":
        return os.path.join(ROOT, "data", "requests", "bbh", f"{key}.parquet")
    p = os.path.join(ROOT, "data", "requests", "gsm8k", f"{key}.parquet")
    return p if os.path.exists(p) else os.path.join(ROOT, "data", "requests", "gsm8k_ext", f"collection_{key}",
                                                    f"{key}.parquet")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=("math",), help="math only (default: GSM8K-29 and BIG-Bench Hard-28)")
    a = ap.parse_args(argv)
    rec = {"label": "forced_direct", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": ARM,
           "decoding": "config/decoding_forced.yaml (fallback config/decoding_forced_noprefill.yaml)",
           "rule": "the collection's direct_answer requests, unchanged except arm label and request id suffix '|fd'",
           "tasks": {}}
    rec_path = os.path.join(OUT, "build_record.json")
    if os.path.exists(rec_path):           # keep the entries of the tasks this run does not build
        rec["tasks"].update(json.load(open(rec_path, encoding="utf-8")).get("tasks", {}))
    tasks = ((("math", "gsm8k30.json"),) if a.task == "math" else (("gsm8k", "gsm8k30.json"), ("bbh", "bbh.json")))
    for task, res in tasks:
        panel = json.load(open(os.path.join(ROOT, "results", "dataset2", res), encoding="utf-8"))["models"]
        if task == "math":                 # the models the MATH screen retained, in panel order
            log = os.path.join(ROOT, "manifests", "math_admissibility.json")
            keep = set(json.load(open(log, encoding="utf-8"))["retained_models"])
            panel = [k for k in panel if k in keep]
        os.makedirs(os.path.join(OUT, task), exist_ok=True)
        rec["tasks"][task] = {}
        for k in panel:
            src = source(task, k)
            df = pd.read_parquet(src)
            d = df[df["arm"] == "direct_answer"].copy()
            if d.empty or len(d) * 2 != len(df):
                raise SystemExit(f"{task}/{k}: {len(d)} direct-answer requests of {len(df)}; expected half")
            d["arm"] = ARM
            d["request_id"] = d["request_id"].astype(str) + "|fd"
            dst = os.path.join(OUT, task, f"{k}.parquet")
            d.to_parquet(dst, index=False)
            rec["tasks"][task][k] = {"source": os.path.relpath(src, ROOT).replace("\\", "/"), "source_sha256": sha(src),
                                     "requests_file": f"{k}.parquet", "requests_sha256": sha(dst), "n": int(len(d)),
                                     "configs": int(d["config_id"].nunique())}
        print(f"{task}: {len(panel)} models, {sum(v['n'] for v in rec['tasks'][task].values())} requests")
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
