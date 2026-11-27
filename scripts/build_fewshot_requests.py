r"""Five-shot forced-direct requests on GSM8K-29 and BIG-Bench Hard-28.

From the repository root, in this order:
    .\.venv\Scripts\python.exe scripts\build_fewshot_requests.py --demos           # 1. the demonstration manifests
    .\.venv\Scripts\python.exe scripts\build_fewshot_requests.py --context-check   # 2. prompt lengths per model
    .\.venv\Scripts\python.exe scripts\build_fewshot_requests.py                   # 3. the request files

1. Demonstrations (manifests/fewshot_demos_<task>.json, written before any request file; never overwritten with a
   different draw). Each distinct configuration of the task's frozen sampler (discovery and validation draws) gets one
   fixed set of five demonstrations, drawn without replacement, in the order drawn, by numpy.random.default_rng seeded
   with the first 8 bytes of sha256("fewshot/<task>/<config_id>") (GSM8K) or sha256("fewshot/bbh/<bbh task>/<config_id>")
   (BIG-Bench Hard, one set per configuration and BIG-Bench Hard task, since a demonstration must be of the item's own
   task). Pools: GSM8K, the training split (7,473 items) outside manifests/gsm8k_calibration_items.json and
   gsm8k_screening_items.json, in item-id order; BIG-Bench Hard, each task's items outside its target, calibration and
   screening sets and its malformed indices (manifests/bbh_items.json), in index order.
2. Context check (results/fewshot_context/): per model and task, the longest rendered five-shot prompt in tokens, with the
   model's pinned tokenizer and chat template, against the registry's max_model_len less the 16-token cap; the context rule
   then sets config/models_fewshot.yaml.
3. Requests (data/requests/fewshot/<task>/<key>.parquet, zstd): the forced arm's requests (data/requests/forced/<task>/
   <key>.parquet) with the five demonstrations before the item: for each, a user turn, the configuration's own
   direct-answer frame filled with the demonstration's question, and an assistant turn "\boxed{<gold>}"; then the forced
   request's own user turn, byte for byte. Arm forced_direct (config/decoding_forced.yaml applies unchanged), request id
   suffix "|fs". data/requests/fewshot/build_record.json records each source and output with its sha256 and count.
Nothing here reads a model output.

MATH-500, with --tasks math on each of the three steps:
    .\.venv\Scripts\python.exe scripts\build_fewshot_requests.py --demos --tasks math
    <tokenizer venv>\Scripts\python.exe scripts\build_fewshot_requests.py --context-check --tasks math
    .\.venv\Scripts\python.exe scripts\build_fewshot_requests.py --tasks math
   Demonstrations: one set of five per configuration, seeded by sha256("fewshot/math/<config_id>"), from the MATH
   training items with a numeric gold answer (src/math_task.py, "Numeric rule": the content of the reference
   solution's last \boxed{} normalizes to one rational number), outside MATH-500's screening and calibration sets
   (both drawn from those training items) and outside any training problem whose text equals a MATH-500 problem, in
   training order (subjects in src/math_task.SUBJECTS order, then row). The answer turn is "\boxed{<canonical gold>}",
   the canonical form of the scorer ("p" or "p/q"), which holds no brace. Context check against config/models_math.yaml
   (the registry of the MATH-500 collection and forced arm) into results/fewshot_context/context_check_math.json.
   Requests: the forced MATH-500 arm's requests (data/requests/forced/math/), decoded with
   config/decoding_forced_math.yaml, as that arm was. The default --tasks of every step stays gsm8k bbh (run 1).
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

TASKS = ("gsm8k", "bbh")               # the default of every step
ALL_TASKS = ("gsm8k", "bbh", "math")   # adds MATH-500
N_DEMOS = 5
SUFFIX = "|fs"
ARM = "forced_direct"
QUESTION = "{question}"
MANIFESTS = os.path.join(ROOT, "manifests")
OUT = os.path.join(ROOT, "data", "requests", "fewshot")
FORCED = os.path.join(ROOT, "data", "requests", "forced")
CONTEXT = os.path.join(ROOT, "results", "fewshot_context")
PANELS = {"gsm8k": "gsm8k30.json", "bbh": "bbh.json", "math": "math.json"}
DECODINGS = {"gsm8k": "config/decoding_forced.yaml", "bbh": "config/decoding_forced.yaml",
             "math": "config/decoding_forced_math.yaml"}
STUDY_PARTS = {"gsm8k": "five-shot forced arm", "bbh": "five-shot forced arm", "math": "five-shot forced arm, MATH-500"}
CONTEXT_FILES = {"run1": "context_check.json", "math": "context_check_math.json"}
CAP = 16


# ============================================================================ pure helpers (tested)


def demo_seed(task: str, config_id: str, bbh_task: str | None = None) -> int:
    label = f"fewshot/{task}/{config_id}" if bbh_task is None else f"fewshot/{task}/{bbh_task}/{config_id}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def draw_demos(pool: list, seed: int, n: int = N_DEMOS) -> list:
    """n members of the pool (in its given order), without replacement, in the order drawn."""
    if len(pool) < n:
        raise ValueError(f"pool of {len(pool)} for {n} demonstrations")
    return [pool[i] for i in np.random.default_rng(seed).choice(len(pool), size=n, replace=False)]


def fill(frame: list[dict], question: str) -> list[dict]:
    """The frame with {question} replaced; exactly one occurrence, in the last message."""
    n = sum(m["content"].count(QUESTION) for m in frame)
    if n != 1 or QUESTION not in frame[-1]["content"]:
        raise ValueError("the frame must hold {question} exactly once, in its last message")
    return [{"role": m["role"], "content": m["content"].replace(QUESTION, question)} for m in frame]


def demo_turns(frame: list[dict], demos: list[tuple[str, str]]) -> list[dict]:
    """User and assistant turns of the demonstrations: the frame filled with each question, then "\\boxed{<gold>}"."""
    if len(frame) != 1 or frame[0]["role"] != "user":
        raise ValueError("a five-shot arm needs a frame of exactly one user message")
    out: list[dict] = []
    for q, gold in demos:
        out += fill(frame, q)
        out.append({"role": "assistant", "content": f"\\boxed{{{gold}}}"})
    return out


def configurations(sampler: dict) -> list[str]:
    """The distinct configurations of the sampler's discovery and validation draws, sorted."""
    return sorted(set(map(str, sampler["discovery_config_ids"])) | set(map(str, sampler["validation_config_ids"])))


def file_sha(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def list_sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ============================================================================ pools and demonstrations


def gsm8k_pool() -> tuple[list[int], dict, dict]:
    """(pool item ids in order, {id: (question, gold)}, pool record)."""
    from src import gsm8k
    tr = gsm8k.load_gsm8k("train")
    excl = set()
    rec = {"split": "train", "n_train": int(len(tr)), "excluded": {}}
    for name in ("gsm8k_calibration_items.json", "gsm8k_screening_items.json"):
        m = json.load(open(os.path.join(MANIFESTS, name), encoding="utf-8"))
        if m["split"] != "train":
            raise SystemExit(f"{name} is not on the training split")
        excl |= set(map(int, m["item_ids"]))
        rec["excluded"][name] = {"n": len(m["item_ids"]), "sha256": m.get("sha256")}
    pool = [int(i) for i in tr["item_id"] if int(i) not in excl]
    qa = {int(i): (q, g) for i, q, g in zip(tr["item_id"], tr["question"], tr["gold_answer"])}
    rec.update({"n_pool": len(pool), "pool_sha256": list_sha(pool), "train_canonical_sha256": gsm8k.canonical_sha256(tr)})
    return pool, qa, rec


def bbh_pools() -> tuple[dict, dict, dict]:
    """({task: pool indices}, {item_id: (question, gold)}, pool record)."""
    from src import bbh
    man = json.load(open(os.path.join(MANIFESTS, "bbh_items.json"), encoding="utf-8"))
    items, _, _, _ = bbh.load_bbh(man["tasks"])
    pools, rec = {}, {"rule": "each task's indices outside target, calibration, screening and malformed", "tasks": {}}
    for t in man["tasks"]:
        pt = man["per_task"][t]
        out = set(pt["target"]) | set(pt["calibration"]) | set(pt["screening"]) | set(pt["malformed_indices"])
        pools[t] = [i for i in range(pt["n_items"]) if i not in out]
        rec["tasks"][t] = {"n_items": pt["n_items"], "n_pool": len(pools[t]), "pool_sha256": list_sha(pools[t])}
    qa = {iid: (q, g) for iid, q, g in zip(items["item_id"], items["question"], items["gold_answer"])}
    return pools, qa, rec


def math_pool_ids(train: pd.DataFrame, excluded_ids: set, target_questions: set) -> tuple[list[str], dict]:
    """Pool of the MATH-500 demonstrations (module doc): numeric training items, in training order, outside the
    excluded ids and outside any problem whose text (surrounding whitespace removed) equals a target problem."""
    num = train[train["gold_answer"].notna()]
    same_text = [i for i, q in zip(num["item_id"], num["question"]) if str(q).strip() in target_questions]
    pool = [str(i) for i in num["item_id"] if i not in excluded_ids and i not in set(same_text)]
    return pool, {"n_numeric": int(len(num)), "n_excluded_sets": int(len(excluded_ids & set(num["item_id"]))),
                  "n_same_text_as_target": len(same_text), "same_text_ids": same_text}


def math_pool() -> tuple[list[str], dict, dict]:
    """(pool item ids in order, {id: (question, canonical gold)}, pool record)."""
    from src import math_task as mt
    man = json.load(open(os.path.join(MANIFESTS, "math_items.json"), encoding="utf-8"))
    target, train = mt.load_target(), mt.load_train()
    sets = mt.split_items(target, train)
    rec = {"split": "train", "dataset": man["train_dataset"], "revision": man["train_revision"],
           "n_train": int(len(train)), "excluded": {}}
    excl = set()
    for name in ("screening", "calibration"):
        sha = mt.canonical_sha256(sets[name])
        if sha != man["sets"][name]["canonical_sha256"]:
            raise SystemExit(f"the {name} set rebuilt here differs from manifests/math_items.json")
        excl |= set(sets[name]["item_id"])
        rec["excluded"][name] = {"n": int(len(sets[name])), "canonical_sha256": sha}
    pool, info = math_pool_ids(train, excl, {str(q).strip() for q in target["question"]})
    qa = {i: (q, g) for i, q, g in zip(train["item_id"], train["question"], train["gold_answer"]) if g is not None}
    rec.update(info)
    rec.update({"n_pool": len(pool), "pool_sha256": list_sha(pool),
                "numeric_rule": man["numeric_rule"], "order": "training order: subjects in src/math_task.SUBJECTS order, "
                "then row in the subject's pinned parquet file"})
    return pool, qa, rec


def demo_manifest(task: str) -> dict:
    s = json.load(open(os.path.join(MANIFESTS, f"sampler_{task}.json"), encoding="utf-8"))
    cfgs = configurations(s)
    man = {"label": f"fewshot_demos_{task}", "study_part": STUDY_PARTS[task], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_demos": N_DEMOS, "sampler": f"manifests/sampler_{task}.json", "sampler_sha256": s["sha256"],
           "configurations": cfgs, "demos": {}}
    if task == "gsm8k":
        pool, qa, rec = gsm8k_pool()
        man["rule"] = ("per configuration, five training items drawn without replacement, in the order drawn, from the "
                       "pool in item-id order with numpy.random.default_rng seeded by the first 8 bytes of "
                       "sha256('fewshot/gsm8k/<config_id>')")
        man["pool"] = rec
        for c in cfgs:
            ids = draw_demos(pool, demo_seed("gsm8k", c))
            man["demos"][c] = [{"item_id": int(i), "gold": qa[int(i)][1]} for i in ids]
    elif task == "math":
        pool, qa, rec = math_pool()
        man["rule"] = ("per configuration, five MATH training items with a numeric gold answer drawn without "
                       "replacement, in the order drawn, from the pool in training order with numpy.random.default_rng "
                       "seeded by the first 8 bytes of sha256('fewshot/math/<config_id>'); the answer turn holds the "
                       "canonical gold ('p' or 'p/q')")
        man["pool"] = rec
        for c in cfgs:
            ids = draw_demos(pool, demo_seed("math", c))
            man["demos"][c] = [{"item_id": str(i), "gold": qa[str(i)][1]} for i in ids]
    else:
        pools, qa, rec = bbh_pools()
        man["rule"] = ("per configuration and BIG-Bench Hard task, five of the task's pool items drawn without replacement, "
                       "in the order drawn, from the pool in index order with numpy.random.default_rng seeded by the "
                       "first 8 bytes of sha256('fewshot/bbh/<bbh task>/<config_id>')")
        man["pool"] = rec
        for c in cfgs:
            man["demos"][c] = {}
            for t, pool in pools.items():
                ids = draw_demos(pool, demo_seed("bbh", c, t))
                man["demos"][c][t] = [{"item_id": f"{t}/{int(i)}", "gold": qa[f"{t}/{int(i)}"][1]} for i in ids]
    man["demos_sha256"] = list_sha(man["demos"])
    return man


def write_demos(tasks=TASKS) -> None:
    for task in tasks:
        p = os.path.join(MANIFESTS, f"fewshot_demos_{task}.json")
        man = demo_manifest(task)
        if os.path.exists(p):
            old = json.load(open(p, encoding="utf-8"))
            if old["demos"] != man["demos"]:
                raise SystemExit(f"{p} holds a different draw; refusing to overwrite it")
            print(f"{p}: unchanged")
            continue
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1, ensure_ascii=False)
        print(f"wrote {p} ({len(man['configurations'])} configurations)")


# ============================================================================ messages


def prefix_messages(task: str) -> dict:
    """{config_id or (config_id, bbh task): demonstration turns} from the saved manifest and the frozen bank."""
    man = json.load(open(os.path.join(MANIFESTS, f"fewshot_demos_{task}.json"), encoding="utf-8"))
    if man["demos_sha256"] != list_sha(man["demos"]):
        raise SystemExit(f"fewshot_demos_{task}.json: its demonstrations do not match their hash")
    bank = json.load(open(os.path.join(MANIFESTS, f"{task}_bank.json"), encoding="utf-8"))
    if task in ("gsm8k", "math"):
        _, qa, _ = gsm8k_pool() if task == "gsm8k" else math_pool()
        return {c: demo_turns(bank["configurations"][c]["arms"]["direct_answer"]["messages"],
                              [(qa[d["item_id"]][0], d["gold"]) for d in ds]) for c, ds in man["demos"].items()}
    _, qa, _ = bbh_pools()
    return {(c, t): demo_turns(bank["configurations"][c]["arms"]["direct_answer"]["messages"],
                               [(qa[d["item_id"]][0], d["gold"]) for d in ds])
            for c, by in man["demos"].items() for t, ds in by.items()}


def prefix_key(task: str, config_id: str, item_id) -> object:
    return config_id if task in ("gsm8k", "math") else (config_id, str(item_id).split("/")[0])


def fewshot_frame(forced: pd.DataFrame, task: str, prefixes: dict) -> pd.DataFrame:
    """The five-shot requests of one model's forced request file (module doc)."""
    out = forced.copy()
    msgs = []
    for c, i, m in zip(out["config_id"].astype(str), out["item_id"], out["messages_json"]):
        target = json.loads(m)
        if len(target) != 1 or target[0]["role"] != "user":
            raise SystemExit(f"{c}/{i}: the forced request is not one user turn")
        msgs.append(json.dumps(prefixes[prefix_key(task, c, i)] + target, ensure_ascii=False))
    out["messages_json"] = msgs
    out["request_id"] = out["request_id"].astype(str) + SUFFIX
    if (out["arm"] != ARM).any():
        raise SystemExit("the forced request file holds another arm")
    return out


# ============================================================================ context check


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json", "config.json", "generation_config.json", "vocab.json",
                   "merges.txt", "tokenizer.model.v3", "tekken.json")


def registry(names=("models.yaml", "models_extension.yaml")) -> dict:
    """Registry entries by model key; MATH-500 reads config/models_math.yaml, the registry of its collection and
    forced arm (max_model_len 8,192, or 4,096 for the checkpoints whose native context is 4,096)."""
    import yaml
    reg = {}
    for name in names:
        for k, v in yaml.safe_load(open(os.path.join(ROOT, "config", name), encoding="utf-8"))["dataset2"]["models"].items():
            reg.setdefault(k, {**v, "_registry": f"config/{name}"})
    return reg


def native_context(ent: dict) -> int | None:
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.load(open(hf_hub_download(ent["hf_id"], "config.json", revision=ent["revision"]), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for c in (cfg, cfg.get("text_config") or {}):
        if c.get("max_position_embeddings"):
            return int(c["max_position_embeddings"])
    return None


def load_tokenizer(ent: dict):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ent["hf_id"], revision=ent["revision"], trust_remote_code=False)
    tmpl = None
    if ent["chat_template"]["source"] == "file":
        tmpl = open(os.path.join(ROOT, ent["chat_template"]["file"]), encoding="utf-8").read()
    return tok, tmpl


def render_len(tok, tmpl, messages, thinking_kwargs) -> int:
    msgs = messages + [{"role": "assistant", "content": "\\boxed{"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True,
                                   **({"chat_template": tmpl} if tmpl else {}), **thinking_kwargs)
    return len(tok.encode(text, add_special_tokens=False))


def candidates(task: str, prefixes: dict, per_key: int = 3) -> list[list[dict]]:
    """For each demonstration prefix, the five-shot messages of its longest target items (by characters)."""
    rows = pd.read_parquet(os.path.join(FORCED, task, "qwen3-8b.parquet"), columns=["config_id", "item_id", "messages_json"])
    rows["k"] = [prefix_key(task, c, i) for c, i in zip(rows["config_id"].astype(str), rows["item_id"])]
    rows["n"] = rows["messages_json"].str.len()
    out = []
    for k, g in rows.groupby("k", sort=False):
        for m in g.nlargest(per_key, "n")["messages_json"]:
            out.append(prefixes[k] + json.loads(m))
    return out


def context_check(models: list[str] | None = None, tasks=TASKS) -> dict:
    tasks = tuple(tasks)
    if "math" in tasks and tasks != ("math",):
        raise SystemExit("--context-check runs math alone (its own registry and output file)")
    reg = registry(("models_math.yaml",) if tasks == ("math",) else ("models.yaml", "models_extension.yaml"))
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "cap": CAP, "models": {}}
    if tasks == ("math",):
        res.update({"registry": "config/models_math.yaml", "decoding": DECODINGS["math"], "study_part": "five-shot forced arm, MATH-500"})
    cands = {t: candidates(t, prefix_messages(t)) for t in tasks}
    panels = {t: json.load(open(os.path.join(ROOT, "results", "dataset2", PANELS[t]), encoding="utf-8"))["models"]
              for t in tasks}
    keys = models or panels[tasks[0]]
    for k in keys:
        ent = reg[k]
        tok, tmpl = load_tokenizer(ent)
        kw = dict(ent["thinking_off"].get("chat_template_kwargs") or {}) \
            if ent["thinking_off"]["method"] == "chat_template_kwargs" else {}
        if ent["thinking_off"]["method"] == "system_prompt_token":
            raise SystemExit(f"{k}: system-prompt thinking switch; not handled by the check")
        r = {"registry": ent["_registry"], "max_model_len": int(ent["max_model_len"]), "native_context": native_context(ent),
             "tasks": {}}
        for t in tasks:
            if k not in panels[t]:
                continue
            lens = [render_len(tok, tmpl, m, kw) for m in cands[t]]
            r["tasks"][t] = {"n_checked": len(lens), "longest_prompt": int(max(lens)),
                             "fits": bool(max(lens) + CAP <= int(ent["max_model_len"]))}
        res["models"][k] = r
        print(k, {t: (v["longest_prompt"], v["fits"]) for t, v in r["tasks"].items()}, r["native_context"], flush=True)
    os.makedirs(CONTEXT, exist_ok=True)
    with open(os.path.join(CONTEXT, CONTEXT_FILES["math" if tasks == ("math",) else "run1"]), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


# ============================================================================ requests


def check_context_math(panel) -> None:
    """MATH-500 requests are built only after the context check covered every panel model and each fits."""
    p = os.path.join(CONTEXT, CONTEXT_FILES["math"])
    if not os.path.exists(p):
        raise SystemExit(f"{p} is absent; run --context-check --tasks math first")
    res = json.load(open(p, encoding="utf-8"))
    bad = [k for k in panel if not ((res["models"].get(k) or {}).get("tasks", {}).get("math") or {}).get("fits")]
    if bad:
        raise SystemExit(f"MATH-500 five-shot prompts not checked or not fitting for {bad}")


def build_requests(tasks=TASKS) -> dict:
    rec_p = os.path.join(OUT, "build_record.json")
    rec = json.load(open(rec_p, encoding="utf-8")) if os.path.exists(rec_p) else {"tasks": {}}
    if set(tasks) & set(TASKS):
        rec.update({"label": "fewshot", "study_part": "five-shot forced arm", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "arm": ARM, "decoding": "config/decoding_forced.yaml",
                    "rule": "the forced arm's requests with five demonstration turns before the item's own turn; "
                            "request id suffix '|fs'"})
    # per task: its arm label, build time and decoding (MATH-500 decodes as its forced arm did)
    for task in tasks:
        rec.setdefault("per_task", {})[task] = {"study_part": STUDY_PARTS[task], "decoding": DECODINGS[task],
                                                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    fr = json.load(open(os.path.join(FORCED, "build_record.json"), encoding="utf-8"))
    for task in tasks:
        demos_p = os.path.join(MANIFESTS, f"fewshot_demos_{task}.json")
        prefixes = prefix_messages(task)
        panel = json.load(open(os.path.join(ROOT, "results", "dataset2", PANELS[task]), encoding="utf-8"))["models"]
        if task == "math":
            check_context_math(panel)
        os.makedirs(os.path.join(OUT, task), exist_ok=True)
        # tasks[task] holds model keys only (score_forced_direct reads every key there as a model)
        rec.setdefault("demos", {})[task] = {"path": f"manifests/fewshot_demos_{task}.json", "sha256": file_sha(demos_p)}
        rec["tasks"][task] = {}
        for k in panel:
            fent = fr["tasks"][task][k]
            src = os.path.join(FORCED, task, f"{k}.parquet")
            if file_sha(src) != fent["requests_sha256"]:
                raise SystemExit(f"{task}/{k}: the forced request file differs from its build record")
            df = fewshot_frame(pd.read_parquet(src), task, prefixes)
            dst = os.path.join(OUT, task, f"{k}.parquet")
            df.to_parquet(dst, index=False, compression="zstd")
            rec["tasks"][task][k] = {"source": fent["source"], "source_sha256": fent["source_sha256"],
                                     "forced_requests_file": f"data/requests/forced/{task}/{k}.parquet",
                                     "forced_requests_sha256": fent["requests_sha256"], "requests_file": f"{k}.parquet",
                                     "requests_sha256": file_sha(dst), "n": int(len(df)),
                                     "configs": int(df["config_id"].nunique())}
            print(f"{task} {k:<40} {len(df):>6}", flush=True)
    with open(rec_p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demos", action="store_true")
    g.add_argument("--context-check", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(ALL_TASKS))
    a = ap.parse_args(argv)
    if a.demos:
        write_demos(a.tasks)
    elif a.context_check:
        context_check(a.models, a.tasks)
    else:
        build_requests(a.tasks)


if __name__ == "__main__":
    main()
