r"""Five-shot requested-direct requests on GSM8K-29 and BIG-Bench Hard-28 (design in
results/fewshot_requested/design.md).

From the repository root, in this order:
    .\.venv\Scripts\python.exe scripts\build_fewshot_requested_requests.py configs    # 1. the drawn configurations
    .\.venv\Scripts\python.exe scripts\build_fewshot_requested_requests.py context    # 2. the context-length record
    .\.venv\Scripts\python.exe scripts\build_fewshot_requested_requests.py requests   # 3. the request files

1. manifests/fewshot_requested_configs.json: four discovery and four fresh validation configurations per task, by
   src/fewshot_requested.draw_configurations from the frozen samplers' ids alone (written once; a different draw is
   refused).
2. results/fewshot_requested/context_check.json: per model, the five-shot forced arm's measured longest prompt
   (results/fewshot_context/context_check.json, rendered with the \boxed{ prefill, so no shorter than the requested
   prompt) plus the requested cap of config/decoding.yaml (1,024) against the registry's max_model_len. A
   model keeps 4,096 when everything fits; otherwise the run stops here and the model is reported.
3. data/requests/fewshot_requested/collection/<key>.parquet (zstd), one file per model with both tasks: the five-shot
   forced arm's rows of the eight drawn configurations (data/requests/fewshot/<task>/<key>.parquet, hash-checked against
   its build record), messages unchanged, arm direct_answer, configuration id "<task>/<config>", item id as a string,
   request id "<task>/<config>|direct_answer|<item>|fsr". The final user turn of every request is checked equal, byte
   for byte, to the main collection's requested-direct request of the same configuration and item (the source file of
   the forced arm's build record, hash-checked). data/requests/fewshot_requested/collection/build_record.json records
   each source and output with its sha256 and count. Nothing here reads a model output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import fewshot_requested as fr  # noqa: E402

FEWSHOT = ROOT / "data" / "requests" / "fewshot"
FORCED = ROOT / "data" / "requests" / "forced"
PANELS = {"gsm8k": ROOT / "results" / "dataset2" / "gsm8k30.json", "bbh": ROOT / "results" / "dataset2" / "bbh.json"}
CONTEXT_SRC = ROOT / "results" / "fewshot_context" / "context_check.json"
DECODING = "config/decoding.yaml"


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def panel(task: str) -> list[str]:
    return list(json.loads(PANELS[task].read_text(encoding="utf-8"))["models"])


def all_models() -> list[str]:
    """GSM8K-29 order, then any BIG-Bench Hard model not in it (none)."""
    return list(dict.fromkeys(panel("gsm8k") + panel("bbh")))


# ------------------------------------------------------------------------------------------ 1. configurations


def configs_manifest(base: Path | None = None) -> dict:
    root = base or ROOT
    tasks = {}
    for t in fr.TASKS:
        sp = root / "manifests" / f"sampler_{t}.json"
        raw = sp.read_bytes()
        d = fr.draw_configurations(t, json.loads(raw.decode("utf-8")))
        tasks[t] = {**d, "sampler": f"manifests/sampler_{t}.json", "sampler_sha256": fr.sha256_bytes(raw)}
    return {"kind": "fewshot_requested_configurations", "study_part": "five-shot requested-direct arm",
            "rule": "results/fewshot_requested/design.md, 'Configurations, by rule'", "n_discovery": fr.N_DISC,
            "n_validation": fr.N_VAL, "tasks": tasks}


def cmd_configs(paths: fr.Paths) -> dict:
    man = configs_manifest(paths.base)
    if paths.configs.exists():
        old = json.loads(paths.configs.read_text(encoding="utf-8"))
        if old.get("tasks") != json.loads(fr.dumps(man["tasks"])):
            raise fr.FewshotRequestedError(f"{rel(paths.configs)} holds a different draw; refusing to overwrite it")
        print(f"{rel(paths.configs)}: unchanged")
        return old
    man["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fr.write_json(paths.configs, man)
    print(f"wrote {rel(paths.configs)}: " + "; ".join(f"{t} {v['discovery']} | {v['validation']}"
                                                      for t, v in man["tasks"].items()))
    return man


# ------------------------------------------------------------------------------------------ 2. context


def cmd_context(paths: fr.Paths) -> dict:
    import yaml
    cap = int(yaml.safe_load((ROOT / DECODING).read_text(encoding="utf-8"))["max_new_tokens"]["direct_answer"])
    src = json.loads(CONTEXT_SRC.read_text(encoding="utf-8"))
    rows, bad = {}, []
    for k in all_models():
        m = src["models"].get(k)
        if m is None:
            bad.append(k)
            continue
        r = {"max_model_len": int(m["max_model_len"]), "tasks": {}}
        for t in fr.TASKS:
            if k in panel(t):
                lp = int(m["tasks"][t]["longest_prompt"])
                r["tasks"][t] = {"longest_prompt": lp, "plus_cap": lp + cap, "fits": lp + cap <= r["max_model_len"]}
                if not r["tasks"][t]["fits"]:
                    bad.append(f"{k}/{t}")
        rows[k] = r
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "cap": cap, "decoding": DECODING,
           "source": {"path": rel(CONTEXT_SRC), "sha256": fr.sha256_file(CONTEXT_SRC),
                      "note": "longest five-shot prompt per model and task over every demonstration prefix (three longest "
                              "items each), rendered with the \\boxed{ prefill; the requested prompt has the generation "
                              "prompt in its place and is no longer"},
           "rule": "a model keeps its registry max_model_len when every prompt plus the cap fits",
           "all_fit": not bad, "not_fitting_or_unchecked": bad, "models": rows}
    fr.write_json(paths.results / "context_check.json", res)
    if bad:
        raise fr.FewshotRequestedError(f"prompts plus the cap do not fit, or were not checked, for {bad}; apply the "
                                       "context rule and report it before any request is built")
    longest = max(v["plus_cap"] for r in rows.values() for v in r["tasks"].values())
    print(f"context: every model fits at 4,096 (longest prompt + {cap} = {longest})")
    return res


# ------------------------------------------------------------------------------------------ 3. requests


def final_turn(messages_json: str) -> str:
    m = json.loads(messages_json)
    if not m or m[-1]["role"] != "user":
        raise fr.FewshotRequestedError("a request does not end with a user turn")
    return json.dumps([m[-1]], ensure_ascii=False)


def task_rows(task: str, key: str, configs: list[str], fs_rec: dict, fd_rec: dict) -> tuple[pd.DataFrame, dict]:
    ent = fs_rec["tasks"][task][key]
    src = FEWSHOT / task / f"{key}.parquet"
    if fr.sha256_file(src) != ent["requests_sha256"]:
        raise fr.FewshotRequestedError(f"{rel(src)} differs from its build record")
    fs = pd.read_parquet(src)
    fs = fs[fs["config_id"].astype(str).isin(configs)].copy()
    col_p = ROOT / fd_rec["tasks"][task][key]["source"]
    if fr.sha256_file(col_p) != fd_rec["tasks"][task][key]["source_sha256"]:
        raise fr.FewshotRequestedError(f"{rel(col_p)} differs from the forced arm's build record")
    col = pd.read_parquet(col_p, columns=["arm", "config_id", "item_id", "messages_json"])
    col = col[(col["arm"] == "direct_answer") & col["config_id"].astype(str).isin(configs)]
    want = {(str(c), str(i)): m for c, i, m in zip(col["config_id"], col["item_id"], col["messages_json"])}
    if len(want) != len(fs):
        raise fr.FewshotRequestedError(f"{task}/{key}: {len(fs)} five-shot rows, {len(want)} requested-direct rows")
    mism = 0
    for c, i, m in zip(fs["config_id"].astype(str), fs["item_id"].astype(str), fs["messages_json"]):
        w = want.get((c, i))
        if w is None or json.loads(w) != json.loads(final_turn(m)):
            mism += 1
    if mism:
        raise fr.FewshotRequestedError(f"{task}/{key}: {mism} final turns differ from the requested-direct requests")
    out = pd.DataFrame({
        "request_id": [f"{task}/{c}|{fr.ARM}|{i}{fr.SUFFIX}" for c, i in zip(fs["config_id"].astype(str), fs["item_id"].astype(str))],
        "model_key": fs["model_key"].astype(str).to_numpy(),
        "arm": fr.ARM,
        "config_id": [fr.job_config_id(task, c) for c in fs["config_id"].astype(str)],
        "item_id": fs["item_id"].astype(str).to_numpy(),
        "messages_json": fs["messages_json"].to_numpy()})
    info = {"fewshot_forced_requests": rel(src), "fewshot_forced_sha256": ent["requests_sha256"],
            "requested_source": rel(col_p), "requested_source_sha256": fd_rec["tasks"][task][key]["source_sha256"],
            "n": int(len(out)), "configs": int(fs["config_id"].nunique()), "final_turn_matches": int(len(out))}
    return out, info


def cmd_requests(paths: fr.Paths, models: list[str] | None = None) -> dict:
    if not paths.configs.exists():
        raise fr.FewshotRequestedError("run `configs` first")
    ctx = paths.results / "context_check.json"
    if not ctx.exists() or not json.loads(ctx.read_text(encoding="utf-8"))["all_fit"]:
        raise fr.FewshotRequestedError("run `context` first; every model must fit")
    man = json.loads(paths.configs.read_text(encoding="utf-8"))
    fs_rec = json.loads((FEWSHOT / "build_record.json").read_text(encoding="utf-8"))
    fd_rec = json.loads((FORCED / "build_record.json").read_text(encoding="utf-8"))
    out_dir = paths.requests / "collection"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_p = out_dir / "build_record.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.exists() else {"models": {}}
    rec.update({"kind": "fewshot_requested_collection", "study_part": "five-shot requested-direct arm",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": fr.ARM, "decoding": DECODING,
                "configs_manifest": rel(paths.configs), "configs_manifest_sha256": fr.sha256_file(paths.configs),
                "configurations": {t: {"discovery": v["discovery"], "validation": v["validation"]}
                                   for t, v in man["tasks"].items()},
                "demos": fs_rec["demos"],
                "rule": "the five-shot forced arm's requests of the drawn configurations, messages unchanged, arm "
                        "direct_answer (no prefill), request id '<task>/<config>|direct_answer|<item>|fsr'"})
    for k in models or all_models():
        parts, info = [], {}
        for t in fr.TASKS:
            if k not in panel(t):
                continue
            cfgs = man["tasks"][t]["discovery"] + man["tasks"][t]["validation"]
            df, info[t] = task_rows(t, k, cfgs, fs_rec, fd_rec)
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        if df["request_id"].duplicated().any():
            raise fr.FewshotRequestedError(f"{k}: duplicate request ids")
        dst = out_dir / f"{k}.parquet"
        df.to_parquet(dst, index=False, compression="zstd")
        rec["models"][k] = {"requests_file": f"{k}.parquet", "requests_sha256": fr.sha256_file(dst),
                            "n_requests": int(len(df)), "decoding": DECODING, "tasks": info}
        print(f"{k:<40} {len(df):>7}", flush=True)
    rec["n_requests"] = int(sum(v["n_requests"] for v in rec["models"].values()))
    fr.write_json(rec_p, rec)
    print(f"total {rec['n_requests']:,} requests in {len(rec['models'])} files")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("configs", "context", "requests"))
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args(argv)
    paths = fr.Paths()
    try:
        {"configs": lambda: cmd_configs(paths), "context": lambda: cmd_context(paths),
         "requests": lambda: cmd_requests(paths, a.models)}[a.step]()
    except fr.FewshotRequestedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
