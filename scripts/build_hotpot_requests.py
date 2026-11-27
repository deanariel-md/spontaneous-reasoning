r"""HotpotQA request files of the collection's four arms, its blocks and pilots.

PowerShell (repository root), after the bank and sampler are frozen and scripts\build_requests.py --task hotpot has
written the requested-direct and chain-of-thought requests (data/requests/hotpot/<key>.parquet):
    .\.venv\Scripts\python.exe scripts\build_hotpot_requests.py --demos                       # 1. demonstrations
    <tokenizer venv>\Scripts\python.exe scripts\build_hotpot_requests.py --context-check      # 2. prompt lengths
    .\.venv\Scripts\python.exe scripts\build_hotpot_requests.py                               # 3. request files

1. Demonstrations (manifests/fewshot_demos_hotpot.json, written before any five-shot request; never overwritten with a
   different draw). Each distinct configuration of the frozen sampler gets five demonstrations, drawn without
   replacement, in the order drawn, from the demonstration pool (manifests/hotpot_items.json, demo_pool: development
   items outside the target and screening sets that share no supporting title with a target item, at most 280 tokens
   with their gold answer), sorted by id, with numpy.random.default_rng seeded by the first 8 bytes of
   sha256("fewshot/hotpot/<config_id>") (run 1's rule).
2. Context check (results/hotpot/context_check.json): per panel model, with its pinned tokenizer and chat template,
   the three longest target items (by characters) under every configuration's requested-direct and chain-of-thought
   frames, the forced frame with the prefill, and the five-shot prompt with each configuration's demonstrations; each
   prompt plus its arm's cap (config/decoding_hotpot.yaml) must fit max_model_len of config/models_hotpot.yaml. The
   build refuses unless every panel model was checked and fits.
3. Requests. From each model's collection file (direct_answer and zero_shot_cot rows):
     forced_direct   the direct_answer rows, unchanged except arm and request id suffix "|fd" (the forced arm's rule)
     fewshot_forced  the forced rows with the five demonstrations before the item's own user turn: for each, a user
                     turn (the configuration's direct-answer frame filled with the demonstration text: its two
                     supporting paragraphs and its question) and an assistant turn "\boxed{<gold>}"; suffix "|fs"
   The four arms of an item go to one file per block: data/requests/hotpot/block_a/<key>.parquet (the first 500 target
   items drawn) and block_b/<key>.parquet (the other 500); data/requests/hotpot/pilot/<key>.parquet holds 16 requests,
   the first four block A items under the first discovery configuration in each arm. Every file decodes with
   config/decoding_hotpot.yaml under config/models_hotpot.yaml. data/requests/hotpot/arms_record.json records each
   source and output with its sha256 and count.
Nothing here reads a model output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_fewshot_requests import demo_seed, demo_turns, draw_demos, file_sha, list_sha  # noqa: E402

MANIFESTS = os.path.join(ROOT, "manifests")
REQ = os.path.join(ROOT, "data", "requests", "hotpot")
DEMOS = os.path.join(MANIFESTS, "fewshot_demos_hotpot.json")
CONTEXT = os.path.join(ROOT, "results", "hotpot", "context_check.json")
REGISTRY = os.path.join(ROOT, "config", "models_hotpot.yaml")
DECODING = os.path.join(ROOT, "config", "decoding_hotpot.yaml")
ARMS_RECORD = os.path.join(REQ, "arms_record.json")
N_DEMOS = 5
PILOT_ITEMS = 4
BLOCKS = {"A": "block_a", "B": "block_b"}


def paths(root: str = ROOT) -> dict:
    return {"sampler": os.path.join(root, "manifests", "sampler_hotpot.json"),
            "bank": os.path.join(root, "manifests", "hotpot_bank.json")}


def configurations(sampler: dict) -> list[str]:
    return sorted(set(map(str, sampler["discovery_config_ids"])) | set(map(str, sampler["validation_config_ids"])))


# ============================================================================ demonstrations


def demo_manifest(sampler: dict, pool: pd.DataFrame, sampler_path: str) -> dict:
    ids = sorted(pool["item_id"].astype(str))
    qa = {str(i): (q, g) for i, q, g in zip(pool["item_id"], pool["question"], pool["gold_answer"])}
    cfgs = configurations(sampler)
    man = {"label": "fewshot_demos_hotpot", "study_part": "HotpotQA", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_demos": N_DEMOS, "sampler": os.path.relpath(sampler_path, ROOT).replace("\\", "/"),
           "sampler_sha256": sampler["sha256"], "configurations": cfgs,
           "rule": "per configuration, five demonstration-pool items drawn without replacement, in the order drawn, from "
                   "the pool sorted by id with numpy.random.default_rng seeded by the first 8 bytes of "
                   "sha256('fewshot/hotpot/<config_id>'); each holds its two supporting paragraphs and its question, "
                   "and the answer turn is \\boxed{<gold as in the file>}",
           "pool": {"source": "manifests/hotpot_items.json, set demo_pool", "n_pool": len(ids), "pool_sha256": list_sha(ids)},
           "demos": {}}
    for c in cfgs:
        man["demos"][c] = [{"item_id": i, "gold": qa[i][1]} for i in draw_demos(ids, demo_seed("hotpot", c), N_DEMOS)]
    man["demos_sha256"] = list_sha(man["demos"])
    return man


def write_demos(sampler_path: str, out: str = DEMOS) -> dict:
    from src import hotpot_d2
    sampler = json.load(open(sampler_path, encoding="utf-8"))
    pool, _ = hotpot_d2.load_items("demo_pool")
    man = demo_manifest(sampler, pool, sampler_path)
    if os.path.exists(out):
        old = json.load(open(out, encoding="utf-8"))
        if old["demos"] != man["demos"]:
            raise SystemExit(f"{out} holds a different draw; refusing to overwrite it")
        print(f"{out}: unchanged")
        return old
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out} ({len(man['configurations'])} configurations)")
    return man


def prefixes(bank: dict, man: dict, pool: pd.DataFrame) -> dict:
    """{config_id: demonstration turns} from the saved manifest and the frozen bank."""
    if man["demos_sha256"] != list_sha(man["demos"]):
        raise SystemExit("fewshot_demos_hotpot.json: its demonstrations do not match their hash")
    text = dict(zip(pool["item_id"].astype(str), pool["question"]))
    return {c: demo_turns(bank["configurations"][c]["arms"]["direct_answer"]["messages"],
                          [(text[d["item_id"]], d["gold"]) for d in ds]) for c, ds in man["demos"].items()}


# ============================================================================ the four arms


def forced_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["arm"] == "direct_answer"].copy()
    if d.empty or len(d) * 2 != len(df):
        raise SystemExit(f"{len(d)} direct-answer requests of {len(df)}; expected half")
    d["arm"] = "forced_direct"
    d["request_id"] = d["request_id"].astype(str) + "|fd"
    return d


def fewshot_rows(forced: pd.DataFrame, pre: dict) -> pd.DataFrame:
    out = forced.copy()
    msgs = []
    for c, i, m in zip(out["config_id"].astype(str), out["item_id"], out["messages_json"]):
        target = json.loads(m)
        if len(target) != 1 or target[0]["role"] != "user":
            raise SystemExit(f"{c}/{i}: the forced request is not one user turn")
        msgs.append(json.dumps(pre[c] + target, ensure_ascii=False))
    out["messages_json"] = msgs
    out["arm"] = "fewshot_forced"
    out["request_id"] = out["request_id"].astype(str).str.replace(r"\|fd$", "|fs", regex=True)
    return out


def four_arms(collection: pd.DataFrame, pre: dict) -> pd.DataFrame:
    f = forced_rows(collection)
    return pd.concat([collection, f, fewshot_rows(f, pre)], ignore_index=True)


def split_blocks(df: pd.DataFrame, target: pd.DataFrame) -> dict[str, pd.DataFrame]:
    block = dict(zip(target["item_id"].astype(str), target["block"]))
    b = df["item_id"].astype(str).map(block)
    if b.isna().any():
        raise SystemExit("request items outside the target set")
    return {k: df[b == k].reset_index(drop=True) for k in BLOCKS}


def pilot_rows(df: pd.DataFrame, target: pd.DataFrame, first_config: str) -> pd.DataFrame:
    a_items = [str(i) for i in target.loc[target["block"] == "A", "item_id"]][:PILOT_ITEMS]
    p = df[(df["config_id"].astype(str) == first_config) & df["item_id"].astype(str).isin(a_items)]
    if len(p) != 4 * PILOT_ITEMS:
        raise SystemExit(f"pilot has {len(p)} rows, expected {4 * PILOT_ITEMS}")
    return p.reset_index(drop=True)


def check_context(panel: list[str], path: str = CONTEXT) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent; run --context-check first")
    res = json.load(open(path, encoding="utf-8"))
    bad = [k for k in panel if not (res["models"].get(k) or {}).get("fits")]
    if bad:
        raise SystemExit(f"HotpotQA prompts not checked or not fitting for {bad}")


def build(models: list[str] | None = None, skip_context: bool = False) -> dict:
    from src import hotpot_d2
    p = paths()
    sampler = json.load(open(p["sampler"], encoding="utf-8"))
    bank = json.load(open(p["bank"], encoding="utf-8"))
    man = json.load(open(DEMOS, encoding="utf-8"))
    if man["sampler_sha256"] != sampler["sha256"]:
        raise SystemExit("fewshot_demos_hotpot.json was drawn for another sampler")
    pool, _ = hotpot_d2.load_items("demo_pool")
    target, tinfo = hotpot_d2.load_items("target")
    pre = prefixes(bank, man, pool)
    panel = models or hotpot_d2.panel()
    if not skip_context:
        check_context(panel)
    first = str(sampler["discovery_config_ids"][0])
    rec = {"label": "hotpot_arms", "study_part": "HotpotQA", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "decoding": "config/decoding_hotpot.yaml", "registry": "config/models_hotpot.yaml",
           "decoding_sha256": file_sha(DECODING), "registry_sha256": file_sha(REGISTRY),
           "demos": {"path": "manifests/fewshot_demos_hotpot.json", "sha256": file_sha(DEMOS)},
           "items": {k: tinfo[k] for k in ("path", "canonical_sha256", "n_items")},
           "rule": "collection rows (direct_answer, zero_shot_cot); forced_direct = direct_answer rows with suffix '|fd'; "
                   "fewshot_forced = forced rows with five demonstration turns before the item's turn, suffix '|fs'; "
                   "blocks A and B by the target table's block; pilot = first discovery configuration, first four "
                   "block A items, four arms",
           "pilot_config": first, "models": {}}
    for d in list(BLOCKS.values()) + ["pilot"]:
        os.makedirs(os.path.join(REQ, d), exist_ok=True)
    for k in panel:
        src = os.path.join(REQ, f"{k}.parquet")
        coll = pd.read_parquet(src)
        if set(coll["arm"]) != {"direct_answer", "zero_shot_cot"}:
            raise SystemExit(f"{k}: {src} is not a collection file with the two arms")
        allr = four_arms(coll, pre)
        ent = {"source": os.path.relpath(src, ROOT).replace("\\", "/"), "source_sha256": file_sha(src),
               "configs": int(allr["config_id"].nunique()), "files": {}}
        parts = split_blocks(allr, target)
        parts["pilot"] = pilot_rows(allr, target, first)
        for name, df in parts.items():
            sub = BLOCKS.get(name, name)
            dst = os.path.join(REQ, sub, f"{k}.parquet")
            df.to_parquet(dst, index=False, compression="zstd")
            ent["files"][sub] = {"path": os.path.relpath(dst, ROOT).replace("\\", "/"), "sha256": file_sha(dst),
                                 "n": int(len(df)), "by_arm": {a: int(v) for a, v in df["arm"].value_counts().sort_index().items()}}
        rec["models"][k] = ent
        print(f"{k:<40} A {ent['files']['block_a']['n']:>7}  B {ent['files']['block_b']['n']:>7}", flush=True)
    rec["totals"] = {sub: int(sum(e["files"][sub]["n"] for e in rec["models"].values()))
                     for sub in list(BLOCKS.values()) + ["pilot"]}
    with open(ARMS_RECORD, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ============================================================================ context check (tokenizer venv)


def render_len(tok, tmpl, messages, kw, prefill: str | None) -> int:
    if prefill:
        msgs = messages + [{"role": "assistant", "content": prefill}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True,
                                       **({"chat_template": tmpl} if tmpl else {}), **kw)
    else:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       **({"chat_template": tmpl} if tmpl else {}), **kw)
    return len(tok.encode(text, add_special_tokens=False))


def fill(frame: list[dict], text: str) -> list[dict]:
    return [{"role": m["role"], "content": m["content"].replace("{question}", text)} for m in frame]


def context_check(models: list[str] | None = None, per_config: int = 3) -> dict:
    import yaml
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from build_fewshot_requests import load_tokenizer
    from src import hotpot_d2
    reg = yaml.safe_load(open(REGISTRY, encoding="utf-8"))["dataset2"]["models"]
    dec = yaml.safe_load(open(DECODING, encoding="utf-8"))
    caps, pre_fill = dec["max_new_tokens"], dec["prefill"]
    p = paths()
    sampler = json.load(open(p["sampler"], encoding="utf-8"))
    bank = json.load(open(p["bank"], encoding="utf-8"))
    man = json.load(open(DEMOS, encoding="utf-8"))
    pool, _ = hotpot_d2.load_items("demo_pool")
    target, _ = hotpot_d2.load_items("target")
    pre = prefixes(bank, man, pool)
    longest = target.assign(n=target["question"].str.len()).nlargest(per_config, "n")["question"].tolist()
    cfgs = configurations(sampler)
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "registry": "config/models_hotpot.yaml",
           "decoding": "config/decoding_hotpot.yaml", "caps": caps, "n_configs": len(cfgs),
           "rule": "each prompt plus its arm's cap fits max_model_len", "models": {}}
    for k in (models or hotpot_d2.panel()):
        ent = reg[k]
        tok, tmpl = load_tokenizer(ent)
        kw = dict(ent["thinking_off"].get("chat_template_kwargs") or {}) \
            if ent["thinking_off"]["method"] == "chat_template_kwargs" else {}
        if ent["thinking_off"]["method"] == "system_prompt_token":
            raise SystemExit(f"{k}: system-prompt thinking switch; not handled by the check")
        worst = {}
        for c in cfgs:
            arms = bank["configurations"][c]["arms"]
            for q in longest:
                for arm, msgs, pf in (("direct_answer", fill(arms["direct_answer"]["messages"], q), None),
                                      ("zero_shot_cot", fill(arms["zero_shot_cot"]["messages"], q), None),
                                      ("forced_direct", fill(arms["direct_answer"]["messages"], q), pre_fill["forced_direct"]),
                                      ("fewshot_forced", pre[c] + fill(arms["direct_answer"]["messages"], q),
                                       pre_fill["fewshot_forced"])):
                    n = render_len(tok, tmpl, msgs, kw, pf)
                    worst[arm] = max(worst.get(arm, 0), n)
        mml = int(ent["max_model_len"])
        per_arm = {a: {"longest_prompt": n, "cap": int(caps[a]), "total": n + int(caps[a]),
                       "fits": n + int(caps[a]) <= mml} for a, n in worst.items()}
        res["models"][k] = {"max_model_len": mml, "arms": per_arm, "fits": all(v["fits"] for v in per_arm.values())}
        print(k, mml, {a: (v["total"], v["fits"]) for a, v in per_arm.items()}, flush=True)
    os.makedirs(os.path.dirname(CONTEXT), exist_ok=True)
    with open(CONTEXT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demos", action="store_true")
    g.add_argument("--context-check", action="store_true")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args(argv)
    if a.demos:
        write_demos(paths()["sampler"])
    elif a.context_check:
        context_check(a.models)
    else:
        build(a.models)


if __name__ == "__main__":
    main()
