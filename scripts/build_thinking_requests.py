r"""Request files and registry for native reasoning (thinking on) as an intervention
arm, against prompted chain of thought and forced direct, on GSM8K and MATH-500.

From the repository root:
    .\.venv\Scripts\python.exe scripts\build_thinking_requests.py --pilot

Models: the GSM8K-29 panel's checkpoints whose thinking-off method is the chat template's enable_thinking
switch (config/models.yaml, config/models_extension.yaml), nine in all. The thinking-on arm is the collection's
direct-answer requests with the switch on, so the model reasons natively before the answer the prompt asks for and
every other part of the prompt is that of the requested-direct arm.

--pilot: data/requests/thinking/pilot/<key>.parquet holds, for each task (GSM8K, MATH-500), the direct-answer screening
requests (training items) of the two configurations with the smallest ids: arm thinking_<task>, request id suffix
"|th", item ids "<task>:<id>", 800 requests per model. It runs under config/models_thinking_pilot.yaml (the panel's
entries with the switch on and max_model_len 32,768) and config/decoding_thinking_pilot.yaml (greedy, 30,720 new
tokens), and measures output lengths and think-block closure for the cap of the runs; no correctness is read.
Writes data/requests/thinking/pilot/build_record.json. Nothing here reads an outcome.

--task gsm8k | math (the runs, after scripts/analyze_thinking_pilot.py on the complete pilot):
    .\.venv\Scripts\python.exe scripts\build_thinking_requests.py --task gsm8k
data/requests/thinking/<task>/<key>.parquet holds the collection's direct-answer requests of each model (GSM8K: the
collection's request file; MATH-500: data/requests/math/<key>.parquet of the models the MATH-500 screen retained), unchanged
except arm thinking_<task> and the request id suffix "|th". The cap and max_model_len are the task's values in
results/thinking_pilot/thinking_pilot.json, written to config/decoding_thinking_<task>.yaml and
config/models_thinking_<task>.yaml. data/requests/thinking/build_record.json lists each output with its source and
hashes, by task (the other task's entries are kept). Checkpoints excluded by the pilot rule
(manifests/thinking_exclusions.json) get no run.

--apply-exclusions: for a task built before a checkpoint was excluded, moves its entry in the build record from
"models" to "excluded_models" (its request file stays) and writes the task's registry without it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import math_d2  # noqa: E402

OUT = ROOT / "data" / "requests" / "thinking"
PILOT_REGISTRY = ROOT / "config" / "models_thinking_pilot.yaml"
PILOT_MAX_MODEL_LEN = 32768
SUFFIX = "|th"
SCREENING = {"gsm8k": (ROOT / "data" / "requests" / "gsm8k" / "screening", ROOT / "data" / "requests" / "gsm8k_ext" / "screening"),
             "math": (ROOT / "data" / "requests" / "math" / "screening",)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def thinking_models(registry: dict | None = None, panel: list[str] | None = None) -> list[str]:
    """Panel checkpoints whose thinking-off method is the template switch enable_thinking, in panel order."""
    reg = registry if registry is not None else math_d2.load_registries()
    keys = panel if panel is not None else math_d2.panel()
    out = []
    for k in keys:
        to = reg[k].get("thinking_off") or {}
        if to.get("method") == "chat_template_kwargs" and (to.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
            out.append(k)
    return out


# Scheduler limits for the long-context runs. vLLM 0.30.0 reserves a sliding-window layer's
# cache for the window plus twice max_num_batched_tokens (two batches in flight), so at its default of 16,384 the 50
# sliding layers of Gemma-4-31B hold the full 32,768 tokens: 27.5 GiB per request against 21.2 GiB free on a 96 GB
# card. At 8,192 one request needs about 16 GiB. Outputs are unchanged under batch-invariant kernels.
THINKING_ENGINE = {"gemma-4-31b-it": {"max_num_batched_tokens": 8192}}


def thinking_entry(e: dict, max_model_len: int, engine: dict | None = None) -> dict:
    """A registry entry with the switch on and any THINKING_ENGINE limits; everything else as in the registry."""
    e = dict(e)
    to = dict(e["thinking_off"])
    to["chat_template_kwargs"] = {**to["chat_template_kwargs"], "enable_thinking": True}
    to["rationale"] = "native reasoning: the thinking switch is on; the field keeps its name so the runner passes the kwargs"
    e["thinking_off"] = to
    e["max_model_len"] = max_model_len
    if engine:
        e["engine"] = {**(e.get("engine") or {}), **engine}
    return e


def write_registry(path: Path, keys: list[str], max_model_len: int, note: str) -> dict:
    reg = math_d2.load_registries()
    out = {k: thinking_entry(reg[k], max_model_len, THINKING_ENGINE.get(k)) for k in keys}
    head = (f"# Generated by scripts/build_thinking_requests.py for the native-reasoning runs.\n"
            f"# {note}\n# Do not edit by hand.\n")
    path.write_text(head + yaml.safe_dump({"dataset2": {"models": out}}, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def screening_file(task: str, key: str) -> Path:
    for d in SCREENING[task]:
        p = d / f"{key}.parquet"
        if p.exists():
            return p
    raise SystemExit(f"{task}/{key}: no screening request file in {[str(d) for d in SCREENING[task]]}")


def pilot_requests(key: str) -> tuple[pd.DataFrame, dict]:
    parts, src = [], {}
    for task in ("gsm8k", "math"):
        p = screening_file(task, key)
        df = pd.read_parquet(p)
        d = df[df["arm"] == "direct_answer"]
        configs = sorted(d["config_id"].astype(str).unique())[:2]
        d = d[d["config_id"].astype(str).isin(configs)].copy()
        d["arm"] = f"thinking_{task}"
        d["request_id"] = d["request_id"].astype(str) + SUFFIX
        d["item_id"] = task + ":" + d["item_id"].astype(str)
        parts.append(d)
        src[task] = {"source": p.relative_to(ROOT).as_posix(), "source_sha256": sha(p), "configs": configs,
                     "n": int(len(d))}
    return pd.concat(parts, ignore_index=True), src


PILOT_RESULT = ROOT / "results" / "thinking_pilot" / "thinking_pilot.json"
PILOT_DECODING = ROOT / "config" / "decoding_thinking_pilot.yaml"
EXCLUSIONS = ROOT / "manifests" / "thinking_exclusions.json"


def excluded_models(path: Path = EXCLUSIONS) -> dict[str, float]:
    """The checkpoints excluded from the native-reasoning runs by the pilot rule, with their share of pilot outputs at the cap."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing; run scripts/analyze_thinking_pilot.py first")
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {k: rec["models"][k]["share"] for k in rec["excluded"]}


def collection_file(task: str, key: str) -> Path:
    if task == "math":
        return ROOT / "data" / "requests" / "math" / f"{key}.parquet"
    p = ROOT / "data" / "requests" / "gsm8k" / f"{key}.parquet"
    return p if p.exists() else ROOT / "data" / "requests" / "gsm8k_ext" / f"collection_{key}" / f"{key}.parquet"


def run_models(task: str, keys: list[str] | None = None, excluded: dict | None = None) -> list[str]:
    """The thinking checkpoints of the task's panel: GSM8K-29 (results/dataset2/gsm8k30.json), or the models the MATH-500 screen
    retained (manifests/math_admissibility.json); without those the pilot rule excluded
    (manifests/thinking_exclusions.json)."""
    keys = thinking_models() if keys is None else keys
    excluded = excluded_models() if excluded is None else excluded
    if task == "math":
        keep = set(json.loads(math_d2.ADMISSIBILITY_LOG.read_text(encoding="utf-8"))["retained_models"])
    else:
        keep = set(json.loads((ROOT / "results" / "dataset2" / "gsm8k30.json").read_text(encoding="utf-8"))["models"])
    return [k for k in keys if k in keep and k not in excluded]


def run_requests(task: str, key: str) -> tuple[pd.DataFrame, dict]:
    src = collection_file(task, key)
    df = pd.read_parquet(src)
    d = df[df["arm"] == "direct_answer"].copy()
    if d.empty or len(d) * 2 != len(df):
        raise SystemExit(f"{task}/{key}: {len(d)} direct-answer requests of {len(df)}; expected half")
    d["arm"] = f"thinking_{task}"
    d["request_id"] = d["request_id"].astype(str) + SUFFIX
    return d, {"source": src.relative_to(ROOT).as_posix(), "source_sha256": sha(src), "n": int(len(d)),
               "configs": int(d["config_id"].nunique())}


def pilot_caps(task: str, path: Path = PILOT_RESULT) -> dict:
    """The task's cap and max_model_len from the complete pilot, or from a partial pilot whose models already put
    the cap at the pilot's own 30,720, the rule's maximum, which no later pilot can change."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing; run scripts/analyze_thinking_pilot.py first")
    R = json.loads(path.read_text(encoding="utf-8"))
    cap = int(R["tasks"][task]["cap"])
    early = not R.get("complete") and cap == int(R.get("pilot_cap", -1))
    if not R.get("complete") and not early:
        raise SystemExit(f"{path.relative_to(ROOT)} covers {len(R['models_present'])} of {len(R['models_expected'])} "
                         "pilot models; the cap needs all of them")
    out = {"cap": cap, "max_model_len": int(R["tasks"][task]["max_model_len"]), "pilot_sha256": sha(path)}
    if early:
        out["decided_by"] = sorted((R.get("exclusion") or {}).get("included", R["models_present"]))
    return out


def write_decoding(task: str, cap: int, path: Path | None = None) -> dict:
    """The pilot's decoding with the task's cap: greedy, special tokens kept."""
    path = path or ROOT / "config" / f"decoding_thinking_{task}.yaml"
    d = yaml.safe_load(PILOT_DECODING.read_text(encoding="utf-8"))
    d["max_new_tokens"] = {f"thinking_{task}": cap}
    head = (f"# Generated by scripts/build_thinking_requests.py --task {task}.\n"
            f"# config/decoding_thinking_pilot.yaml with the cap of the {task} runs, set by\n"
            f"# the cap rule from results/thinking_pilot/thinking_pilot.json. Do not edit by hand.\n")
    path.write_text(head + yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "cap": cap}


def registry_note(task: str, max_model_len: int) -> str:
    return (f"The {task} runs: the panel's entries with enable_thinking true and max_model_len {max_model_len:,}.")


def build_task(task: str) -> dict:
    caps = pilot_caps(task)
    excluded = excluded_models()
    keys = run_models(task, excluded=excluded)
    out = OUT / task
    out.mkdir(parents=True, exist_ok=True)
    rec_path = OUT / "build_record.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {"label": "thinking", "tasks": {}}
    ent = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": f"thinking_{task}", "pilot": caps,
           "rule": "the collection's direct_answer requests, unchanged except arm label and request id suffix '|th'",
           "models_rule": run_models.__doc__, "models": {},
           "excluded": {k: {"pilot_cap_share": excluded[k]} for k in run_models(task, excluded={}) if k in excluded}}
    for k in keys:
        df, src = run_requests(task, k)
        f = out / f"{k}.parquet"
        df.to_parquet(f, index=False)
        ent["models"][k] = {**src, "requests_file": f.name, "requests_sha256": sha(f)}
        print(f"{task}/{k:<24} {len(df)} requests")
    ent["decoding"] = write_decoding(task, caps["cap"])
    ent["registry"] = write_registry(ROOT / "config" / f"models_thinking_{task}.yaml", keys, caps["max_model_len"],
                                     registry_note(task, caps["max_model_len"]))
    rec["tasks"][task] = ent
    rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{task}: {len(keys)} models, cap {caps['cap']:,}, max_model_len {caps['max_model_len']:,}"
          + (f"; excluded {sorted(ent['excluded'])}" if ent["excluded"] else ""))
    return ent


def apply_exclusions(rec_path: Path = OUT / "build_record.json", excluded: dict | None = None) -> dict:
    """Moves the build record's entries of excluded checkpoints from "models" to "excluded" (request files stay) and
    writes each changed task's registry without them."""
    excluded = excluded_models() if excluded is None else excluded
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for task, ent in rec["tasks"].items():
        drop = [k for k in ent["models"] if k in excluded]
        if not drop:
            continue
        ent.setdefault("excluded", {})
        for k in drop:
            ent["excluded"][k] = {**ent["models"].pop(k), "pilot_cap_share": excluded[k]}
        ent["registry"] = write_registry(ROOT / ent["registry"]["path"], list(ent["models"]),
                                         int(ent["pilot"]["max_model_len"]),
                                         registry_note(task, int(ent["pilot"]["max_model_len"])))
        print(f"{task}: excluded {drop}; {len(ent['models'])} models left")
    rec_path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--task", choices=("gsm8k", "math"))
    g.add_argument("--apply-exclusions", action="store_true")
    a = ap.parse_args(argv)
    if a.task:
        build_task(a.task)
        return 0
    if a.apply_exclusions:
        apply_exclusions()
        return 0
    keys = thinking_models()
    out = OUT / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"label": "thinking_pilot", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "screening", "split": "train",
           "purpose": "native reasoning: output lengths and think-block closure with thinking on, for the cap; no correctness",
           "excluded_from_main": True, "rule": thinking_models.__doc__, "models": {}}
    for k in keys:
        df, src = pilot_requests(k)
        f = out / f"{k}.parquet"
        df.to_parquet(f, index=False)
        rec["models"][k] = {"requests_file": f.name, "requests_sha256": sha(f), "n_requests": int(len(df)), "sources": src}
        print(f"{k:<24} {len(df)} pilot requests")
    rec["registry"] = write_registry(PILOT_REGISTRY, keys, PILOT_MAX_MODEL_LEN,
                                     "The pilot registry: the panel's entries with enable_thinking true and max_model_len 32,768.")
    (out / "build_record.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{len(keys)} models; wrote {out / 'build_record.json'} and {rec['registry']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
