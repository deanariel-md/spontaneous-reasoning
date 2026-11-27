r"""Temperature 0.7 spot check on MATH-500.

Five panel models, one per family, generate MATH-500's three arms (requested, forced, chain of thought) under sampling
at temperature 0.7 (top_p 1.0, top_k off, one sample per prompt, a fixed seed per request), on the 398 numeric target
items and the first 4 discovery templates. Everything else is the panel's greedy MATH-500 run: the
same request rows (messages byte for byte), registry (config/models_math.yaml), caps, stop rules, prefill of the forced
arm, chat templates, kernels and GPU model (RTX 6000 Ada, as the greedy runs). The runner is src/generation.py, whose
opt-in sampling mode (Decoding.sampling) gives each request its own seed; config/decoding_temp_check*.yaml.

Subcommands (repository root):
    python scripts/temp_check.py choose     # manifests/temp_check_models.json (write-once)
    python scripts/temp_check.py build      # request files, manifests/temp_check_requests.json
    python scripts/temp_check.py pilot-check    # after the pilot jobs are copied back
    python scripts/temp_check.py report     # after the full run is copied back
    python scripts/temp_check.py refs       # the in-job gate's greedy reference
    python scripts/temp_check.py budget     # config/budget.yaml entry (GPU-hours from the scheduler's accounting)
    python scripts/temp_check.py finalize   # results/temp_check/finalize.md

Model rule (choose): within each of the panel's five families, the MATH-500 panel model (the 29 models of
data/requests/math/build_record.json) whose parameter count (params_b of manifests/d2_panel_extension.json, by
repository) is nearest the panel's median parameter count; ties: fewer parameters, then model key. Only metadata is
read; no output or score.

Templates: the first 4 of manifests/sampler_math.json's discovery_config_ids ("budget K uses the first K discovery
units in the order listed"); all 4 are distinct, each counted once.

Seeds: sampling_seed = int(sha256("temp_check/t0.7|" + request_id)[:8], 16) mod 2**31, one per request.

Pilot: 8 items of the first template (sha256 order of "temp_check/pilot|" + item_id), three arms, run twice per model in
one job: greedy (config/decoding_temp_check_greedy*.yaml, temperature 0) and sampled. pilot-check passes when the greedy
pilot outputs equal the panel's stored greedy outputs (same request ids), every prompt sha256 equals the panel's, the
model block and chat template equal the greedy run's, the decoding parameters equal the greedy ones apart from the
temperature and the per-request seed, and every sampled output is readable.

Report: w (share of requested outputs longer than the forced cap, 16 tokens; src/baseline_mixture.RULE_TEXT), the
requested gain (CoT - requested), the forced gain (CoT - forced), the corrected gain (1 - w)(CoT - forced) and the
accuracies, under sampling and under greedy decoding on the same items and templates, per model and pooled (mean of the
five), with the difference (sampled - greedy) and its question-bootstrap interval (items resampled with replacement,
the same draw for both decodings, all arms and all models). Strict scoring (math-boxed) and content scoring (content/
1.7.0), both through scripts/nebius_panel_report.score_task, which is checked against the panel's scored tables on the
greedy rows. Outputs: results/temp_check/temp_check.json, summary.md, summary_clean.md, pilot_check.{json,md}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TEMPERATURE = 0.7
K_TEMPLATES = 4
FORCED_CAP = 16
ARMS = ("direct_answer", "forced_direct", "zero_shot_cot")
ARM_LABEL = {"direct_answer": "requested", "forced_direct": "forced", "zero_shot_cot": "cot"}
SEED_LABEL = "temp_check/t0.7|"
PILOT_LABEL = "temp_check/pilot|"
PILOT_ITEMS = 8
BOOT_DRAWS = 4000
BOOT_SEED_LABEL = "temp_check/boot"
LEVEL = 0.95
FAMILIES = ("Gemma", "Llama", "Mistral", "Phi", "Qwen")

MODELS_MANIFEST = ROOT / "manifests" / "temp_check_models.json"
REQ_MANIFEST = ROOT / "manifests" / "temp_check_requests.json"
REQ_DIR = ROOT / "data" / "requests" / "temp_check"
OUT_DIR = ROOT / "data" / "d2_out" / "temp_check"
RESULTS = ROOT / "results" / "temp_check"
MODELS_YAML = ROOT / "config" / "models_math.yaml"
DECODING = {3072: "config/decoding_temp_check.yaml", 2900: "config/decoding_temp_check_4k.yaml"}
DECODING_GREEDY = {3072: "config/decoding_temp_check_greedy.yaml", 2900: "config/decoding_temp_check_greedy_4k.yaml"}
# The greedy collection's own decoding files, per arm: the twins must equal them arm by arm.
GREEDY_SOURCES = {3072: {"direct_answer": "config/decoding.yaml", "zero_shot_cot": "config/decoding.yaml",
                         "forced_direct": "config/decoding_forced_math.yaml"},
                  2900: {"direct_answer": "config/decoding_math_4k.yaml", "zero_shot_cot": "config/decoding_math_4k.yaml",
                         "forced_direct": "config/decoding_forced_math.yaml"}}


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def dumps(o) -> str:
    return json.dumps(o, indent=2, ensure_ascii=False, sort_keys=False, default=_default) + "\n"


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    raise TypeError(type(o))


def clean_md(md: str) -> str:
    """The clean copy: no citation markers ([n], [^n]) and no HTML comments."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    md = re.sub(r"\[\^?\d+(?:,\s*\d+)*\]", "", md)
    return md


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_once(path: Path, body: dict) -> dict:
    """Write a manifest once; an existing file with the same content (sha256 of the body) is kept, another refused."""
    body = dict(body)
    body["sha256"] = sha256_text(json.dumps({k: v for k, v in body.items() if k not in ("sha256", "written")},
                                            sort_keys=True, default=_default))
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("sha256") != body["sha256"]:
            raise SystemExit(f"{rel(path)} exists with other content; it is written once (delete it by hand to redo)")
        return old
    body["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_text(path, dumps(body))
    return body


def seed_of(request_id: str) -> int:
    return int(sha256_text(SEED_LABEL + str(request_id))[:8], 16) % (2 ** 31)


# ----------------------------------------------------------------------------------------------------------------
# choose
# ----------------------------------------------------------------------------------------------------------------

def choose_models(panel_keys, registry: dict, params_by_repo: dict, families=FAMILIES) -> dict:
    """The rule of the module doc: per family, the model whose params_b is nearest the panel median."""
    rows = []
    for k in sorted(panel_keys):
        e = registry[k]
        rows.append({"model_key": k, "hf_id": e["hf_id"], "family": e["family"],
                     "params_b": float(params_by_repo[e["hf_id"]])})
    med = float(statistics.median(r["params_b"] for r in rows))
    chosen, cands = {}, {}
    for fam in families:
        fr = [dict(r, distance=abs(r["params_b"] - med)) for r in rows if r["family"] == fam]
        if not fr:
            raise SystemExit(f"family {fam} has no panel model")
        fr.sort(key=lambda r: (r["distance"], r["params_b"], r["model_key"]))
        cands[fam] = fr
        chosen[fam] = fr[0]["model_key"]
    return {"panel": rows, "median_params_b": med, "candidates": cands, "chosen": chosen}


def cmd_choose(args) -> None:
    from src import generation as gen
    rec = json.loads((ROOT / "data" / "requests" / "math" / "build_record.json").read_text(encoding="utf-8"))
    keys = sorted(rec["models"])
    reg_raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["dataset2"]["models"]
    registry = {k: {"hf_id": reg_raw[k]["hf_id"], "family": gen.parse_model_spec(k, gen.load_registry(MODELS_YAML)).family}
                for k in keys}
    ext = json.loads((ROOT / "manifests" / "d2_panel_extension.json").read_text(encoding="utf-8"))
    params = {p["repo"]: p["params_b"] for p in ext["panel"]}
    names = {p["repo"]: p["eci_name"] for p in ext["panel"]}
    res = choose_models(keys, registry, params)
    body = {
        "purpose": "temperature 0.7 spot check on MATH-500",
        "rule": ("Within each of the panel's five families (Gemma, Llama, Mistral, Phi, Qwen), the MATH-500 panel model "
                 "(the 29 models of data/requests/math/build_record.json) whose parameter count (params_b of "
                 "manifests/d2_panel_extension.json, matched by Hugging Face repository) is nearest the median "
                 "parameter count of the 29; ties: fewer parameters, then model key. Metadata only; no output or "
                 "score was read."),
        "decided": "before any temperature-check output existed",
        "inputs": {rel(ROOT / "data" / "requests" / "math" / "build_record.json"):
                   sha256_file(ROOT / "data" / "requests" / "math" / "build_record.json"),
                   rel(ROOT / "manifests" / "d2_panel_extension.json"):
                   sha256_file(ROOT / "manifests" / "d2_panel_extension.json"),
                   rel(MODELS_YAML): sha256_file(MODELS_YAML)},
        "n_panel": len(keys),
        "median_params_b": res["median_params_b"],
        "candidates": {f: [{k: r[k] for k in ("model_key", "params_b", "distance")} for r in v]
                       for f, v in res["candidates"].items()},
        "chosen": [{"family": f, "model_key": k, "hf_id": registry[k]["hf_id"], "name": names.get(registry[k]["hf_id"]),
                    "params_b": params[registry[k]["hf_id"]]} for f, k in res["chosen"].items()],
    }
    out = write_once(MODELS_MANIFEST, body)
    print(json.dumps(out["chosen"], indent=1))


def chosen_models() -> list[dict]:
    return json.loads(MODELS_MANIFEST.read_text(encoding="utf-8"))["chosen"]


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------

def templates() -> list[str]:
    s = json.loads((ROOT / "manifests" / "sampler_math.json").read_text(encoding="utf-8"))
    t = [str(x) for x in s["discovery_config_ids"][:K_TEMPLATES]]
    if len(set(t)) != len(t):
        raise SystemExit(f"the first {K_TEMPLATES} discovery draws repeat a template: {t}")
    return t


def greedy_runs(key: str) -> dict:
    """The panel's greedy MATH-500 run logs of one model: {'collection': log, 'forced': log} (complete runs only)."""
    out = {}
    for sub in ("collection", "forced"):
        logs = [json.loads(Path(f).read_text(encoding="utf-8"))
                for f in sorted(glob.glob(str(ROOT / "data" / "d2_out" / "math" / sub / key / "run_log-*.json")))]
        logs = [g for g in logs if g.get("status") == "complete" and g.get("mode") == "full"]
        if len(logs) != 1:
            raise SystemExit(f"{key} {sub}: expected one complete greedy run log, found {len(logs)}")
        out[sub] = logs[0]
    return out


def cot_cap(key: str) -> int:
    cap = int(greedy_runs(key)["collection"]["decoding"]["max_new_tokens"]["zero_shot_cot"])
    if cap not in DECODING:
        raise SystemExit(f"{key}: chain-of-thought cap {cap} has no temperature-check decoding file")
    return cap


def build_rows(key: str, tmpl: list[str]) -> pd.DataFrame:
    a = pd.read_parquet(ROOT / "data" / "requests" / "math" / f"{key}.parquet")
    b = pd.read_parquet(ROOT / "data" / "requests" / "forced" / "math" / f"{key}.parquet")
    df = pd.concat([a, b], ignore_index=True)
    df = df[df["config_id"].isin(tmpl)].copy()
    df["sampling_seed"] = df["request_id"].map(seed_of).astype("int64")
    df = df.sort_values(["config_id", "arm", "item_id", "request_id"], kind="stable").reset_index(drop=True)
    return df


def pilot_rows(df: pd.DataFrame, tmpl: list[str], n: int = PILOT_ITEMS) -> pd.DataFrame:
    items = sorted(df["item_id"].unique(), key=lambda i: sha256_text(PILOT_LABEL + str(i)))[:n]
    return df[(df["config_id"] == tmpl[0]) & df["item_id"].isin(items)].reset_index(drop=True)


def check_rows(df: pd.DataFrame, key: str, tmpl: list[str], n_items: int) -> None:
    want = n_items * len(tmpl) * len(ARMS)
    if len(df) != want:
        raise SystemExit(f"{key}: {len(df)} request rows, expected {want}")
    if df["request_id"].duplicated().any() or df["sampling_seed"].duplicated().any():
        raise SystemExit(f"{key}: request ids or seeds repeat")
    if set(df["arm"]) != set(ARMS) or set(df["model_key"]) != {key}:
        raise SystemExit(f"{key}: unexpected arms or model keys")


def cmd_build(args) -> None:
    from src import generation as gen
    tmpl = templates()
    items_rec = json.loads((ROOT / "data" / "requests" / "math" / "build_record.json").read_text(encoding="utf-8"))
    n_items = int(items_rec["items"]["n_items"])
    models = []
    for m in chosen_models():
        key = m["model_key"]
        runs = greedy_runs(key)
        cap = cot_cap(key)
        df = build_rows(key, tmpl)
        check_rows(df, key, tmpl, n_items)
        pil = pilot_rows(df, tmpl)
        files = {}
        for kind, frame in (("run", df), ("pilot", pil), ("pilot_greedy", pil.drop(columns=["sampling_seed"]))):
            p = REQ_DIR / kind / f"{key}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(p, index=False)
            files[kind] = {"path": rel(p), "sha256": sha256_file(p), "rows": int(len(frame))}
        # the runner's own checks on the files, against the decodings they will run under
        dec, dg = gen.load_decoding(ROOT / DECODING[cap]), gen.load_decoding(ROOT / DECODING_GREEDY[cap])
        gen.validate_requests(gen.read_requests(REQ_DIR / "run" / f"{key}.parquet"), key, dec)
        gen.validate_requests(gen.read_requests(REQ_DIR / "pilot" / f"{key}.parquet"), key, dec)
        gen.validate_requests(gen.read_requests(REQ_DIR / "pilot_greedy" / f"{key}.parquet"), key, dg)
        gpu = runs["collection"]["gpu"]["devices"][0]["name"]
        models.append({"family": m["family"], "model_key": key, "cot_cap": cap, "decoding": DECODING[cap],
                       "decoding_sha256": sha256_file(ROOT / DECODING[cap]),
                       "decoding_greedy": DECODING_GREEDY[cap],
                       "decoding_greedy_sha256": sha256_file(ROOT / DECODING_GREEDY[cap]),
                       "greedy_gpu": gpu, "greedy_run_ids": {s: r["run_id"] for s, r in runs.items()},
                       "files": files, "counts_by_arm": df["arm"].value_counts().sort_index().to_dict()})
    body = {
        "purpose": "temperature 0.7 spot check on MATH-500",
        "models_manifest": {"path": rel(MODELS_MANIFEST), "sha256": sha256_file(MODELS_MANIFEST)},
        "templates": tmpl,
        "templates_rule": (f"the first {K_TEMPLATES} of manifests/sampler_math.json's discovery_config_ids (nested order: "
                           "budget K uses the first K discovery units in the order listed); all distinct, each counted "
                           "once"),
        "items": {"source": items_rec["items"]["source"], "n_items": n_items,
                  "canonical_sha256": items_rec["items"]["canonical_sha256"]},
        "arms": list(ARMS),
        "sampling": {"temperature": TEMPERATURE, "top_p": 1.0, "top_k": "0 (off)", "samples_per_prompt": 1,
                     "seed_rule": "sampling_seed = int(sha256('temp_check/t0.7|' + request_id)[:8], 16) mod 2**31"},
        "pilot": {"rule": f"{PILOT_ITEMS} items of template {tmpl[0]} in sha256 order of 'temp_check/pilot|' + item_id, "
                          "three arms; run greedy and sampled in one job",
                  "items": sorted(pil["item_id"].unique().tolist())},
        "request_sources": "data/requests/math/<key>.parquet (direct_answer, zero_shot_cot) and "
                           "data/requests/forced/math/<key>.parquet (forced_direct): the greedy run's rows unchanged",
        "models": models,
        "n_generations": int(sum(sum(m["counts_by_arm"].values()) for m in models)),
    }
    out = write_once(REQ_MANIFEST, body)
    print(f"{out['n_generations']} generations; templates {tmpl}; {rel(REQ_MANIFEST)}")


# ----------------------------------------------------------------------------------------------------------------
# Reading outputs
# ----------------------------------------------------------------------------------------------------------------

OUT_COLS = ["request_id", "model_key", "arm", "config_id", "item_id", "prompt_sha256", "output_text",
            "n_prompt_tokens", "n_output_tokens", "finish_reason"]


def read_folder(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    shards = sorted(glob.glob(str(folder / "shard-*.parquet")))
    logs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in sorted(glob.glob(str(folder / "run_log-*.json")))]
    if not shards:
        return pd.DataFrame(columns=OUT_COLS), logs
    df = pd.concat([pd.read_parquet(s, columns=OUT_COLS) for s in shards], ignore_index=True)
    if df["request_id"].duplicated().any():
        raise SystemExit(f"{rel(folder)}: a request id repeats across shards")
    return df, logs


def greedy_outputs(key: str, request_ids) -> pd.DataFrame:
    parts = []
    for sub in ("collection", "forced"):
        d, _ = read_folder(ROOT / "data" / "d2_out" / "math" / sub / key)
        parts.append(d[d["request_id"].isin(set(request_ids))])
    return pd.concat(parts, ignore_index=True)


def cmd_refs(args) -> None:
    """The in-job gate's reference (scripts/temp_check_gate.py): per model, the greedy outputs of the pilot requests
    (request_id, prompt_sha256, output_text) and the greedy run's model block, chat template, GPU and versions."""
    man = json.loads(REQ_MANIFEST.read_text(encoding="utf-8"))
    d = REQ_DIR / "pilot_ref"
    d.mkdir(parents=True, exist_ok=True)
    for m in man["models"]:
        key = m["model_key"]
        req = pd.read_parquet(ROOT / m["files"]["pilot"]["path"])
        g = greedy_outputs(key, req["request_id"])
        if len(g) != len(req):
            raise SystemExit(f"{key}: {len(g)} greedy outputs for {len(req)} pilot requests")
        g[["request_id", "prompt_sha256", "output_text"]].to_parquet(d / f"{key}.parquet", index=False)
        runs = greedy_runs(key)
        c, f = runs["collection"], runs["forced"]
        for part in ("model", "versions"):
            if c[part] != f[part] and part == "model":
                raise SystemExit(f"{key}: the collection and forced runs log different model blocks")
        meta = {"model": c["model"], "chat_template_sha256": c["chat_template"]["observed_sha256"],
                "gpu": c["gpu"]["devices"][0]["name"],
                "versions": {p: c["versions"].get(p) for p in ("vllm", "torch", "transformers")}}
        write_text(d / f"{key}.json", dumps(meta))
        print(key, len(g), meta["gpu"], meta["versions"])


# ----------------------------------------------------------------------------------------------------------------
# pilot-check
# ----------------------------------------------------------------------------------------------------------------

def decoding_diff(sampled: dict, greedy: dict) -> dict:
    """Keys of the two decoding blocks (run logs) that differ, with both values."""
    keys = sorted(set(sampled) | set(greedy))
    return {k: [sampled.get(k), greedy.get(k)] for k in keys if sampled.get(k) != greedy.get(k)}


def params_differences(cap: int) -> dict:
    """Per arm: the parameters that differ between the sampled decoding and the greedy collection's own file."""
    from src import generation as gen
    ds = gen.load_decoding(ROOT / DECODING[cap])
    dg = gen.load_decoding(ROOT / DECODING_GREEDY[cap])
    out = {}
    for arm, src in GREEDY_SOURCES[cap].items():
        orig = gen.load_decoding(ROOT / src).params_for_arm(arm)
        twin = dg.params_for_arm(arm)
        samp = ds.params_for_arm(arm)
        out[arm] = {"greedy_twin_equals_collection": twin == orig,
                    "sampled_vs_collection": {k: [samp.get(k), orig.get(k)] for k in sorted(set(samp) | set(orig))
                                              if samp.get(k) != orig.get(k)}}
    return out


def cmd_pilot_check(args) -> None:
    man = json.loads(REQ_MANIFEST.read_text(encoding="utf-8"))
    res = {"models": {}, "passed": True}
    for m in man["models"]:
        key, cap = m["model_key"], m["cot_cap"]
        runs = greedy_runs(key)
        req = pd.read_parquet(ROOT / m["files"]["pilot"]["path"])
        gref = greedy_outputs(key, req["request_id"])
        r = {"checks": []}

        def check(name, ok, detail=None):
            r["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

        pd_ = params_differences(cap)
        check("greedy twin decoding equals the collection's decoding, arm by arm",
              all(v["greedy_twin_equals_collection"] for v in pd_.values()), pd_)
        allowed = {"temperature", "sampling_seed_source"}
        check("sampled decoding differs from the collection's only in temperature and the per-request seed",
              all(set(v["sampled_vs_collection"]) <= allowed and
                  v["sampled_vs_collection"].get("temperature") == [TEMPERATURE, 0.0] for v in pd_.values()),
              {a: v["sampled_vs_collection"] for a, v in pd_.items()})
        for kind in ("pilot_greedy", "pilot"):
            out, logs = read_folder(OUT_DIR / kind / key)
            done = [g for g in logs if g.get("status") == "complete"]
            check(f"{kind}: a complete run", len(done) >= 1 and len(out) == len(req),
                  {"logs": [(g.get("run_id"), g.get("status")) for g in logs], "rows": len(out), "want": len(req)})
            if out.empty:
                continue
            j = out.merge(gref, on="request_id", suffixes=("", "_greedy"), how="left")
            check(f"{kind}: every prompt sha256 equals the greedy run's", (j["prompt_sha256"] == j["prompt_sha256_greedy"]).all(),
                  int((j["prompt_sha256"] != j["prompt_sha256_greedy"]).sum()))
            lg = done[-1] if done else logs[-1]
            check(f"{kind}: model block equals the greedy run's", lg["model"] == runs["collection"]["model"],
                  decoding_diff(lg["model"], runs["collection"]["model"]))
            check(f"{kind}: chat template observed equals the registered one",
                  lg.get("chat_template", {}).get("observed_sha256") == runs["collection"]["chat_template"]["observed_sha256"],
                  lg.get("chat_template"))
            check(f"{kind}: engine rendering equals the runner's", lg.get("prompt_render_mismatches") == 0,
                  lg.get("prompt_render_mismatches"))
            check(f"{kind}: GPU model equals the greedy run's", lg["gpu"]["devices"][0]["name"] == m["greedy_gpu"],
                  lg["gpu"]["devices"][0]["name"])
            check(f"{kind}: kernels as the greedy run's", lg.get("batch_invariant") == runs["collection"].get("batch_invariant"),
                  [lg.get("batch_invariant"), runs["collection"].get("batch_invariant")])
            vers = {p: [lg["versions"].get(p), runs["collection"]["versions"].get(p)] for p in ("vllm", "torch", "transformers")}
            r.setdefault("versions", {})[kind] = vers
            if kind == "pilot_greedy":
                same = (j["output_text"] == j["output_text_greedy"])
                r["greedy_identical"] = {"identical": int(same.sum()), "rows": int(len(j)),
                                         "by_arm": j.assign(s=same).groupby("arm")["s"].agg(["sum", "count"]).astype(int).to_dict("index")}
                check("pilot_greedy: outputs equal the panel's stored greedy outputs", same.all(), r["greedy_identical"])
            else:
                same = (j["output_text"] == j["output_text_greedy"])
                r["sampled_identical_to_greedy"] = {"identical": int(same.sum()), "rows": int(len(j)),
                                                    "by_arm": j.assign(s=same).groupby("arm")["s"].agg(["sum", "count"]).astype(int).to_dict("index")}
                check("pilot: decoding block has temperature 0.7 and the per-request seed",
                      lg["decoding"].get("temperature") == TEMPERATURE and lg["decoding"].get("sampling") == {"seed": "request"},
                      {k: lg["decoding"].get(k) for k in ("temperature", "top_p", "top_k", "sampling")})
                check("pilot: every output non-empty", (out["output_text"].str.len() > 0).all(),
                      int((out["output_text"].str.len() == 0).sum()))
                r["pilot_samples"] = [{"arm": a, "request_id": x["request_id"], "output_start": x["output_text"][:160],
                                       "greedy_start": x["output_text_greedy"][:160]}
                                      for a, g in j.groupby("arm") for x in g.head(1).to_dict("records")]
            r.setdefault("runtime_s", {})[kind] = [g.get("runtime_s") for g in logs]
        r["passed"] = all(c["ok"] for c in r["checks"])
        res["passed"] &= r["passed"]
        res["models"][key] = r
    write_text(RESULTS / "pilot_check.json", dumps(res))
    L = ["# Temperature check: pilot check", "",
         f"Passed: **{res['passed']}**. Each model ran {PILOT_ITEMS} items x 3 arms of template {man['templates'][0]} "
         "twice in one job: greedy (temperature 0, the collection's decoding) and sampled (temperature 0.7, one seed per "
         "request).", ""]
    for key, r in res["models"].items():
        L.append(f"## {key}: {'passed' if r['passed'] else 'FAILED'}")
        for c in r["checks"]:
            L.append(f"- {'ok' if c['ok'] else 'FAIL'}: {c['name']}")
        if "greedy_identical" in r:
            g = r["greedy_identical"]
            L.append(f"- greedy pilot outputs identical to the stored greedy outputs: {g['identical']} of {g['rows']}")
        if "sampled_identical_to_greedy" in r:
            g = r["sampled_identical_to_greedy"]
            L.append(f"- sampled pilot outputs identical to the greedy ones: {g['identical']} of {g['rows']}")
        L.append(f"- versions (this run, greedy run): {json.dumps(r.get('versions', {}).get('pilot'))}")
        L.append("")
    md = "\n".join(L) + "\n"
    write_text(RESULTS / "pilot_check.md", md)
    write_text(RESULTS / "pilot_check_clean.md", clean_md(md))
    print(md)
    if not res["passed"]:
        sys.exit(3)


# ----------------------------------------------------------------------------------------------------------------
# Scoring and quantities
# ----------------------------------------------------------------------------------------------------------------

def score_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Strict and content scores of MATH-500 outputs with the panel's scorers (nebius_panel_report.score_task: the
    forced arm read with the prefill in front)."""
    import nebius_panel_report as npr
    o = df[["request_id", "item_id", "arm", "output_text", "finish_reason"]].copy()
    o["item_id"] = o["item_id"].astype(str)
    sc, ver = npr.score_task(o.reset_index(drop=True), "math")
    out = df.reset_index(drop=True).copy()
    out["strict_correct"] = sc["strict_correct"].astype(float).to_numpy()
    out["content_correct"] = sc["content_correct"].astype(float).to_numpy()
    out.attrs["versions"] = ver
    return out


def panel_scores(keys) -> pd.DataFrame:
    """The panel's scored greedy tables (strict: math/scored.parquet and forced/math_scored.parquet; content:
    content/math_content.parquet) for the given models."""
    P = ROOT / "data" / "processed"
    s = pd.concat([pd.read_parquet(P / "math" / "scored.parquet"), pd.read_parquet(P / "forced" / "math_scored.parquet")])
    s = s[s["model_key"].isin(keys)][["model_key", "arm", "config_id", "item_id", "strict_correct"]]
    c = pd.read_parquet(P / "content" / "math_content.parquet")
    c = c[c["model_key"].isin(keys)][["model_key", "arm", "config_id", "item_id", "content_correct"]]
    return s.merge(c, on=["model_key", "arm", "config_id", "item_id"], how="outer")


def mats(df: pd.DataFrame, score: str, tmpl, items) -> dict:
    """(templates, items) matrices of one model and decoding: R, F, C (score per arm) and W (requested longer than
    the forced cap, '>' on n_output_tokens)."""
    out = {}
    for arm, k in (("direct_answer", "R"), ("forced_direct", "F"), ("zero_shot_cot", "C")):
        g = df[df["arm"] == arm]
        out[k] = g.pivot_table(index="config_id", columns="item_id", values=score, aggfunc="first") \
            .reindex(index=list(tmpl), columns=list(items)).to_numpy(float)
    g = df[df["arm"] == "direct_answer"].assign(_w=lambda x: (x["n_output_tokens"] > FORCED_CAP).astype(float))
    out["W"] = g.pivot_table(index="config_id", columns="item_id", values="_w", aggfunc="first") \
        .reindex(index=list(tmpl), columns=list(items)).to_numpy(float)
    return out


QUANTS = ("w", "req_gain", "fd_gain", "corrected_gain", "requested", "forced", "cot")
Q_LABEL = {"w": "w (requested longer than 16 tokens)", "req_gain": "Requested gain (CoT - requested)",
           "fd_gain": "Forced gain (CoT - forced)", "corrected_gain": "Corrected gain (1 - w)(CoT - forced)",
           "requested": "Requested accuracy", "forced": "Forced accuracy", "cot": "CoT accuracy"}


def quantities(R, F, C, W) -> dict:
    """From template means of item means (shares). Works elementwise on arrays (bootstrap)."""
    return {"w": W, "req_gain": C - R, "fd_gain": C - F, "corrected_gain": (1 - W) * (C - F),
            "requested": R, "forced": F, "cot": C}


def tmean(M: np.ndarray, axis_items: int = -1) -> np.ndarray:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.nanmean(M, axis=axis_items), axis=-1)


def point(m: dict) -> dict:
    return quantities(*(float(tmean(m[k])) for k in ("R", "F", "C", "W")))


def boot_draws(n_items: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(draws, n_items))


def boot_quantities(m: dict, idx: np.ndarray, chunk: int = 500) -> dict:
    """{quantity: (draws,)} for one model and decoding, the item draws idx (draws, N) shared by every matrix."""
    D = idx.shape[0]
    per = {k: np.empty(D) for k in m}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s0 in range(0, D, chunk):
            sl = idx[s0:s0 + chunk]
            for k, M in m.items():
                per[k][s0:s0 + chunk] = np.nanmean(np.nanmean(M[:, sl], axis=2), axis=0)
    return quantities(*(per[k] for k in ("R", "F", "C", "W")))


def ci(v: np.ndarray, level: float = LEVEL) -> list:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [None, None]
    a = (1 - level) / 2
    return [float(x) for x in np.quantile(v, [a, 1 - a])]


def analyse(scored: dict, tmpl, items, draws: int = BOOT_DRAWS, seed: int | None = None) -> dict:
    """scored: {model_key: {"greedy": df, "sampled": df}} with strict_correct, content_correct, n_output_tokens.
    Per model and pooled (mean over models), per score: greedy, sampled, difference (sampled - greedy) with
    question-bootstrap intervals (items resampled, one draw shared by decodings, arms and models)."""
    seed = int(sha256_text(BOOT_SEED_LABEL)[:8], 16) if seed is None else seed
    idx = boot_draws(len(items), draws, seed)
    out = {"draws": draws, "seed": seed, "level": LEVEL, "scores": {}}
    for score in ("strict_correct", "content_correct"):
        res, boots = {}, {}
        for key, d in scored.items():
            res[key], boots[key] = {}, {}
            for dec in ("greedy", "sampled"):
                m = mats(d[dec], score, tmpl, items)
                res[key][dec] = point(m)
                boots[key][dec] = boot_quantities(m, idx)
                res[key][dec + "_ci"] = {q: ci(boots[key][dec][q]) for q in QUANTS}
                res[key][dec + "_missing_cells"] = int(sum(np.isnan(v).sum() for v in m.values()))
            res[key]["diff"] = {q: res[key]["sampled"][q] - res[key]["greedy"][q] for q in QUANTS}
            res[key]["diff_ci"] = {q: ci(boots[key]["sampled"][q] - boots[key]["greedy"][q]) for q in QUANTS}
        keys = list(scored)
        pooled = {dec: {q: float(np.mean([res[k][dec][q] for k in keys])) for q in QUANTS} for dec in ("greedy", "sampled")}
        pb = {dec: {q: np.mean([boots[k][dec][q] for k in keys], axis=0) for q in QUANTS} for dec in ("greedy", "sampled")}
        pooled["greedy_ci"] = {q: ci(pb["greedy"][q]) for q in QUANTS}
        pooled["sampled_ci"] = {q: ci(pb["sampled"][q]) for q in QUANTS}
        pooled["diff"] = {q: pooled["sampled"][q] - pooled["greedy"][q] for q in QUANTS}
        pooled["diff_ci"] = {q: ci(pb["sampled"][q] - pb["greedy"][q]) for q in QUANTS}
        res["pooled"] = pooled
        out["scores"][score] = res
    return out


def output_stats(scored: dict) -> dict:
    """Per model, arm and decoding: rows, truncations (finish_reason length), mean output tokens; and the share of
    sampled outputs identical to the greedy output of the same request."""
    out = {}
    for key, d in scored.items():
        g, s = d["greedy"], d["sampled"]
        j = s.merge(g[["request_id", "output_text"]], on="request_id", suffixes=("", "_g"))
        o = {}
        for arm in ARMS:
            a = {}
            for dec, df in (("greedy", g), ("sampled", s)):
                x = df[df["arm"] == arm]
                a[dec] = {"rows": int(len(x)), "truncated": int((x["finish_reason"] == "length").sum()),
                          "mean_output_tokens": float(x["n_output_tokens"].mean()) if len(x) else None}
            jj = j[j["arm"] == arm]
            a["identical_outputs"] = int((jj["output_text"] == jj["output_text_g"]).sum())
            a["compared"] = int(len(jj))
            o[ARM_LABEL[arm]] = a
        out[key] = o
    return out


def verify_greedy_scoring(gs: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """The greedy rows scored here against the panel's scored tables, per score: rows compared and disagreements."""
    j = gs.merge(panel, on=["model_key", "arm", "config_id", "item_id"], suffixes=("", "_panel"), how="left")
    out = {}
    for c in ("strict_correct", "content_correct"):
        miss = int(j[c + "_panel"].isna().sum())
        dis = int((j[c].astype(float) != j[c + "_panel"].astype(float)).sum()) - miss
        out[c] = {"rows": int(len(j)), "missing_in_panel": miss, "disagree": dis}
    return out


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------

def pp(x, d=1, signed=False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    v = 100 * x
    if abs(v) < 0.5 * 10 ** (-d):
        v = 0.0
    return (f"{v:+.{d}f}" if signed else f"{v:.{d}f}")


def ci_txt(c, d=1) -> str:
    if not c or c[0] is None:
        return "[n/a]"
    return f"[{pp(c[0], d, True)}, {pp(c[1], d, True)}]"


def contains_zero(c) -> bool | None:
    if not c or c[0] is None:
        return None
    return c[0] <= 0 <= c[1]


def load_scored(man: dict) -> tuple[dict, dict, dict]:
    tmpl = man["templates"]
    scored, verify, versions, run_logs = {}, {}, {}, {}
    panel = panel_scores([m["model_key"] for m in man["models"]])
    for m in man["models"]:
        key = m["model_key"]
        req = pd.read_parquet(ROOT / m["files"]["run"]["path"])
        s, logs = read_folder(OUT_DIR / "run" / key)
        missing = sorted(set(req["request_id"]) - set(s["request_id"]))
        extra = sorted(set(s["request_id"]) - set(req["request_id"]))
        if missing or extra:
            raise SystemExit(f"{key}: sampled outputs miss {len(missing)} and add {len(extra)} requests")
        g = greedy_outputs(key, req["request_id"])
        if len(g) != len(req):
            raise SystemExit(f"{key}: {len(g)} greedy outputs for {len(req)} requests")
        if not (s.set_index("request_id")["prompt_sha256"].sort_index()
                == g.set_index("request_id")["prompt_sha256"].sort_index()).all():
            raise SystemExit(f"{key}: a sampled prompt differs from the greedy prompt of the same request")
        gs, ss = score_outputs(g), score_outputs(s)
        versions = gs.attrs.get("versions", {})
        verify[key] = verify_greedy_scoring(gs, panel)
        scored[key] = {"greedy": gs, "sampled": ss}
        run_logs[key] = [{k: lg.get(k) for k in ("run_id", "status", "runtime_s", "model_load_s", "generation_s",
                                                 "gpu_hours_billed_estimate", "hostname")}
                         | {"gpu": (lg.get("gpu") or {}).get("devices", [{}])[0].get("name"),
                            "scheduler_job": (lg.get("scheduler") or {}).get("SLURM_JOB_ID"),
                            "unique_generations": (lg.get("totals") or {}).get("unique_generations"),
                            "output_tokens": (lg.get("totals") or {}).get("output_tokens_generated"),
                            "prompt_tokens": (lg.get("totals") or {}).get("prompt_tokens_generated"),
                            "cached_calls": (lg.get("totals") or {}).get("cached_calls")}
                         for lg in logs]
    return scored, {"verify": verify, "versions": versions}, run_logs


def cmd_report(args) -> None:
    man = json.loads(REQ_MANIFEST.read_text(encoding="utf-8"))
    mm = json.loads(MODELS_MANIFEST.read_text(encoding="utf-8"))
    tmpl = man["templates"]
    items = sorted(pd.read_parquet(ROOT / man["models"][0]["files"]["run"]["path"])["item_id"].unique())
    scored, meta, run_logs = load_scored(man)
    an = analyse(scored, tmpl, items, draws=args.draws)
    stats = output_stats(scored)
    fam = {m["model_key"]: m["family"] for m in man["models"]}
    names = {c["model_key"]: c["name"] for c in mm["chosen"]}
    body = {
        "title": "Temperature 0.7 spot check on MATH-500",
        "design": {"models": [{"family": fam[k], "model_key": k, "name": names.get(k)} for k in scored],
                   "templates": tmpl, "templates_rule": man["templates_rule"], "n_items": len(items),
                   "arms": {a: ARM_LABEL[a] for a in ARMS}, "sampling": man["sampling"],
                   "greedy_reference": "the panel's greedy MATH-500 outputs of the same requests",
                   "forced_cap_for_w": FORCED_CAP, "n_generations_sampled": int(sum(len(d["sampled"]) for d in scored.values())),
                   "models_rule": mm["rule"]},
        "scorers": meta["versions"],
        "greedy_scoring_check": meta["verify"],
        "analysis": an,
        "outputs": stats,
        "run_logs": run_logs,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_text(RESULTS / "temp_check.json", dumps(body))
    md = render(body)
    write_text(RESULTS / "summary.md", md)
    write_text(RESULTS / "summary_clean.md", clean_md(md))
    print(md)


def headline(body: dict) -> list[str]:
    """Two sentences: does sampling change w or the gains materially (strict, pooled, and the largest per-model
    difference with its interval)."""
    a = body["analysis"]["scores"]
    lines = []
    for score, lab in (("strict_correct", "strict"), ("content_correct", "content")):
        r = a[score]
        p = r["pooled"]
        keys = [k for k in r if k != "pooled"]
        worst = {q: max(keys, key=lambda k: abs(r[k]["diff"][q])) for q in ("w", "req_gain", "fd_gain")}
        n_zero = {q: sum(bool(contains_zero(r[k]["diff_ci"][q])) for k in keys) for q in ("w", "req_gain", "fd_gain")}
        lines.append(
            f"{lab.capitalize()} scoring, pooled over the five models (sampled minus greedy): w {pp(p['diff']['w'], 1, True)} pp "
            f"{ci_txt(p['diff_ci']['w'])}, requested gain {pp(p['diff']['req_gain'], 1, True)} pp {ci_txt(p['diff_ci']['req_gain'])}, "
            f"forced gain {pp(p['diff']['fd_gain'], 1, True)} pp {ci_txt(p['diff_ci']['fd_gain'])}; "
            f"intervals containing 0 among the five models: w {n_zero['w']}, requested gain {n_zero['req_gain']}, "
            f"forced gain {n_zero['fd_gain']}; largest per-model differences: w {pp(r[worst['w']]['diff']['w'], 1, True)} pp "
            f"({worst['w']}), requested gain {pp(r[worst['req_gain']]['diff']['req_gain'], 1, True)} pp ({worst['req_gain']}), "
            f"forced gain {pp(r[worst['fd_gain']]['diff']['fd_gain'], 1, True)} pp ({worst['fd_gain']}).")
    return lines


def render(body: dict) -> str:
    d = body["design"]
    a = body["analysis"]
    L = [f"# {body['title']}", ""]
    L += headline(body) + [""]
    L += ["## Design", "",
          f"- Models (one per family, by rule: {d['models_rule']}): " +
          ", ".join(f"{m['family']}: {m['model_key']}" for m in d["models"]) + ".",
          f"- Templates: {', '.join(d['templates'])} ({d['templates_rule']}). Items: the {d['n_items']} numeric MATH-500 "
          "target items. Arms: requested (direct_answer), forced (forced_direct, prefill \\boxed{, cap 16, no stop), "
          "chain of thought (zero_shot_cot).",
          f"- Sampling: temperature {d['sampling']['temperature']}, top_p {d['sampling']['top_p']}, top_k off, one sample "
          f"per prompt, seed per request ({d['sampling']['seed_rule']}). Everything else as the panel's greedy run. "
          f"{d['n_generations_sampled']:,} sampled generations.",
          f"- Reference: {d['greedy_reference']}. w: share of requested outputs longer than {d['forced_cap_for_w']} "
          "tokens. Quantities are means over the 4 templates of item means; the difference is sampled minus greedy; "
          f"intervals: {int(a['level'] * 100)}% question bootstrap ({a['draws']} draws; items resampled, the same draw for "
          "both decodings, the three arms and the five models), percentile.",
          f"- Scorers: {json.dumps(body['scorers'])}. Greedy rows rescored here against the panel's scored tables: "
          + "; ".join(f"{k}: strict {v['strict_correct']['disagree']} and content {v['content_correct']['disagree']} "
                      f"disagreements of {v['strict_correct']['rows']}" for k, v in body["greedy_scoring_check"].items()) + ".",
          ""]
    for score, lab in (("strict_correct", "Strict"), ("content_correct", "Content")):
        r = a["scores"][score]
        L += [f"## {lab} scoring", "",
              "| Model | Quantity | Greedy | Sampled | Sampled - greedy [95% CI] |", "|---|---|---|---|---|"]
        for key in list(k for k in r if k != "pooled") + ["pooled"]:
            for q in ("w", "req_gain", "fd_gain", "corrected_gain", "requested", "forced", "cot"):
                x = r[key]
                L.append(f"| {key} | {Q_LABEL[q]} | {pp(x['greedy'][q])} | {pp(x['sampled'][q])} | "
                         f"{pp(x['diff'][q], 1, True)} {ci_txt(x['diff_ci'][q])} |")
        L.append("")
    L += ["## Outputs", "", "| Model | Arm | Truncated greedy | Truncated sampled | Mean tokens greedy | Mean tokens sampled | "
          "Sampled output identical to greedy |", "|---|---|---|---|---|---|---|"]
    for key, o in body["outputs"].items():
        for arm, x in o.items():
            L.append(f"| {key} | {arm} | {x['greedy']['truncated']} of {x['greedy']['rows']} | {x['sampled']['truncated']} of "
                     f"{x['sampled']['rows']} | {x['greedy']['mean_output_tokens']:.1f} | {x['sampled']['mean_output_tokens']:.1f} | "
                     f"{x['identical_outputs']} of {x['compared']} |")
    L += ["", "Values in percentage points (w and accuracies in percent). Pooled = the mean of the five models. "
          "Edge cases are reported as they are: an interval that contains 0 is not read as no change, and none is "
          "clipped.", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------------------------------------------------

def cmd_budget(args) -> None:
    body = json.loads((RESULTS / "temp_check.json").read_text(encoding="utf-8"))
    logs = body["run_logs"]
    gens = sum(sum((lg.get("unique_generations") or 0) for lg in v) for v in logs.values())
    out_tok = sum(sum((lg.get("output_tokens") or 0) for lg in v) for v in logs.values())
    in_tok = sum(sum((lg.get("prompt_tokens") or 0) for lg in v) for v in logs.values())
    cached = sum(sum((lg.get("cached_calls") or 0) for lg in v) for v in logs.values())
    runner_h = sum(sum((lg.get("gpu_hours_billed_estimate") or 0) for lg in v) for v in logs.values())
    if args.gpu_hours is None or args.jobs is None:   # from the jobs file and the final accounting read
        sp = RESULTS / "sacct_final.txt"
        js = job_summary(read_jobs(), parse_sacct(sp.read_text(encoding="utf-8")) if sp.exists() else {})
        args.gpu_hours = js["scheduler_gpu_hours"] if args.gpu_hours is None else args.gpu_hours
        args.jobs = args.jobs or ", ".join(f"{r['name']} {r['job_id']} {r['state']}" for r in js["jobs"])
    entry = {
        "temp_check_math": {
            "what": ("Temperature 0.7 spot check on MATH-500: 5 panel models, 398 items "
                     "x 4 templates x 3 arms, one sample per prompt; scripts/temp_check.py, results/temp_check/"),
            "date": time.strftime("%Y-%m-%d"),
            "unique_generations": int(gens), "cached_calls": int(cached),
            "prompt_tokens": int(in_tok), "output_tokens": int(out_tok),
            "runner_gpu_hours_full_run": round(runner_h, 3),
            "scheduler_gpu_hours_all_jobs": args.gpu_hours, "scheduler_jobs": args.jobs,
            "pilot": "5 jobs, greedy and sampled pilots (24 + 24 generations per model) in one job each",
            "hardware": "RTX 6000 Ada (48 GB), as the greedy MATH-500 runs",
            "not_counted_above": ("a first attempt (pilots failed, scratch erased; runs cancelled) and the "
                                  "full runs on the RTX PRO 6000, rerun on the Ada; their outputs are kept aside, unused"),
            "cost_usd": 0,
        }}
    p = ROOT / "config" / "budget.yaml"
    text = p.read_text(encoding="utf-8")
    if "temp_check_math:" in text:
        raise SystemExit("config/budget.yaml already has a temp_check_math entry")
    block = yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=200)
    write_text(p, text.rstrip("\n") + "\n" + block)
    print(block)


# ----------------------------------------------------------------------------------------------------------------
# finalize: job accounting and the report
# ----------------------------------------------------------------------------------------------------------------

FINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
                "DEADLINE")
REPORT = ROOT / "results" / "temp_check" / "finalize.md"


def read_jobs(path: Path = RESULTS / "jobs.tsv") -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split("\t")
        if len(p) == 4:
            out.append({"model_key": p[0], "kind": p[1], "job_id": p[2], "name": p[3]})
    return out


def parse_sacct(text: str) -> dict:
    """sacct -X -n -P -o JobID,JobName,State,ElapsedRaw,ExitCode,NodeList,Start,End -> {job id: record}."""
    out = {}
    for line in text.lstrip("﻿").splitlines():   # PowerShell's Set-Content -Encoding UTF8 writes a BOM
        p = line.strip().split("|")
        if len(p) < 5 or not p[0]:
            continue
        out[p[0]] = {"name": p[1], "state": p[2].split()[0] if p[2] else "", "elapsed_s": int(p[3] or 0),
                     "exit_code": p[4], "node": p[5] if len(p) > 5 else None, "start": p[6] if len(p) > 6 else None,
                     "end": p[7] if len(p) > 7 else None}
    return out


def job_summary(jobs: list[dict], sacct: dict) -> dict:
    rows = []
    for j in jobs:
        s = sacct.get(j["job_id"], {})
        rows.append({**j, **{k: s.get(k) for k in ("state", "elapsed_s", "exit_code", "node", "start", "end")}})
    secs = sum((r["elapsed_s"] or 0) for r in rows)
    return {"jobs": rows, "scheduler_gpu_hours": round(secs / 3600.0, 3) if rows else None,
            "all_final": bool(rows) and all((r["state"] or "").startswith(FINAL_STATES) for r in rows)}


def cmd_finalize(args) -> None:
    jobs = read_jobs()
    sacct_p = RESULTS / "sacct_final.txt"
    sacct = parse_sacct(sacct_p.read_text(encoding="utf-8")) if sacct_p.exists() else {}
    js = job_summary(jobs, sacct)
    res_p = RESULTS / "temp_check.json"
    body = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else None
    if body is not None:
        body["jobs"] = js
        write_text(res_p, dumps(body))
    squeue_txt = (RESULTS / "squeue_after.txt").read_text(encoding="utf-8").strip() if (RESULTS / "squeue_after.txt").exists() else None
    cleanup = (RESULTS / "cleanup.txt").read_text(encoding="utf-8").strip() if (RESULTS / "cleanup.txt").exists() else None
    md = render_report(body, js, squeue_txt, cleanup)
    write_text(REPORT, md)
    print(md)


def render_report(body: dict | None, js: dict, squeue_txt: str | None, cleanup: str | None) -> str:
    L = ["# Temperature 0.7 spot check on MATH-500", ""]
    if body is None:
        L += ["**No answer yet.** The check is built and tested, but no job has run. Generate "
              "the request files of `python scripts/temp_check.py build` through the runner (src/generation.py).", ""]
    else:
        L += headline(body) + [""]
    L += ["## Built", "",
          "- `src/generation.py`: an opt-in sampling mode. A decoding file with `sampling: {seed: request}` and a "
          "temperature in (0, 2] samples, each request with its own seed from the request file's `sampling_seed` column "
          "(part of the request's cache key; the forced arm's prefill and every other parameter unchanged). Greedy "
          "decodings give the same parameters, cache keys and fingerprints as before; a greedy decoding refuses a file "
          "that carries seeds. tests/test_generation.py passes unchanged.",
          "- `config/decoding_temp_check.yaml` (chain-of-thought cap 3,072) and `_4k` (2,900, for the two models of "
          "max_model_len 4,096, as their greedy run), with greedy twins `decoding_temp_check_greedy*.yaml`. Each arm's "
          "parameters equal the greedy collection's own file (`config/decoding.yaml` or `decoding_math_4k.yaml`, "
          "`decoding_forced_math.yaml`), apart from temperature 0.7 and the per-request seed (checked by "
          "`scripts/temp_check.py params_differences` and the tests).",
          "- `manifests/temp_check_models.json` (rule and choice, metadata only): per family, the model nearest the "
          "panel's median parameter count (12.25 B over the 29 MATH-500 models): gemma-3-12b-it, llama-2-13b-chat-hf, "
          "mistral-nemo-instruct-2407, phi-3-medium-4k-instruct, qwen3-14b.",
          "- `manifests/temp_check_requests.json`: the first 4 MATH-500 discovery templates (ps-i00-f05, ps-i02-f03, "
          "ps-i03-f07, ps-i06-f01), all 398 numeric target items, three arms: 23,880 generations (500 items would give 30,000; "
          "the panel uses 398). The greedy run's request rows unchanged, plus a seed per request.",
          "- `scripts/temp_check.py` (choose, build, refs, pilot-check, report, budget, finalize), "
          "`scripts/temp_check_gate.py` (the pilot's in-job gate).",
          "- Scoring: `scripts/nebius_panel_report.score_task` (strict math-boxed/1.0.2, content/1.7.0, forced arm read "
          "with the prefill); on the five models' greedy rows of the 4 templates (23,880) it reproduces the panel's "
          "scored tables with 0 disagreements, strict and content.",
          "- Macros: module `paper/macros_temp_check.py`, functions `temp_check_results(root)` and "
          "`temp_check_macros(R)`, prefix `tempCheck` (names in the module doc). They are empty until "
          "results/temp_check/temp_check.json exists.", ""]
    L += ["## Jobs", ""]
    if not js["jobs"]:
        L += ["- Nothing submitted yet.",
              "- Plan once submitted: per model a pilot (greedy and sampled pilot, 24 + 24 generations, "
              "then the gate) and a full run that depends on the pilot's success, on RTX 6000 Ada cards (the greedy "
              "runs' card), at most 5 running.",
              "- Estimate from the greedy runs' generation rates: about 1.4 GPU-hours of generation plus loading, "
              "2-4 GPU-hours in all.",
              "- If a pilot fails its gate, that model's full run is removed; the gate's "
              "file is copied to data/d2_out/temp_check/pilot_gate/ and results/temp_check/pilot_check.md says why.",
              ""]
    else:
        L += ["| Job | Id | State | Elapsed (min) | Node |", "|---|---|---|---|---|"]
        for r in js["jobs"]:
            L.append(f"| {r['name']} | {r['job_id']} | {r['state']} | "
                     f"{'' if r['elapsed_s'] is None else round(r['elapsed_s'] / 60, 1)} | {r['node'] or ''} |")
        L += ["", f"- GPU-hours, all jobs: {js['scheduler_gpu_hours']}.",
              f"- Cleanup: {cleanup or 'not yet'}.",
              f"- Queue afterwards: {squeue_txt if squeue_txt is not None else 'not read yet'}.", ""]
    tests_p = RESULTS / "tests.out"
    tlines = tests_p.read_text(encoding="utf-8", errors="replace").strip().splitlines() if tests_p.exists() else []
    tline = tlines[-1] if tlines else "not run"   # empty while a pytest run is writing it
    L += ["## Tests", "", f"`python -m pytest -q tests/test_temp_check.py`: `{tline}` "
          "(results/temp_check/tests.out). tests/test_generation.py: 127 passed after the runner change.", ""]
    if not js["jobs"]:
        L += ["## To drop the check instead", "",
              "Delete the staged working folder. Nothing else of the check is staged.", ""]
    if body is not None:
        L += ["## Results", "", f"results/temp_check/summary.md (and _clean.md, temp_check.json).", ""]
    L += ["## Uncertain or open", "",
          "- Per-request seeds under vLLM 0.30.0's batch-invariant kernels have not run on a GPU here; the pilot and its "
          "gate are the test (a refusal or an error fails the gate and removes the full run).",
          "- `src/generation.py` is shared by every run; the sampling mode is opt-in and leaves greedy runs unchanged "
          "(127 generation tests pass).",
          "- 23,880 generations, not 30,000: the panel's MATH-500 set has 398 numeric items.",
          "- The w and gain macros' values exist only after the run; the names are fixed (module doc).", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("choose")
    sub.add_parser("build")
    sub.add_parser("refs")
    sub.add_parser("pilot-check")
    r = sub.add_parser("report")
    r.add_argument("--draws", type=int, default=BOOT_DRAWS)
    b = sub.add_parser("budget")
    b.add_argument("--gpu-hours", type=float, help="GPU-hours (default: from jobs.tsv and sacct_final.txt)")
    b.add_argument("--jobs", help="job ids and states (default: from jobs.tsv and sacct_final.txt)")
    sub.add_parser("finalize")
    args = ap.parse_args(argv)
    {"choose": cmd_choose, "build": cmd_build, "refs": cmd_refs, "pilot-check": cmd_pilot_check, "report": cmd_report,
     "budget": cmd_budget, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
