r"""Score the HotpotQA collection: pilots, then the first four discovery draws alone, then every discovery draw, then
all.

From the repository root, after copying the output folders back to data/d2_out/hotpot/{pilot,block_a,block_b}/<key>:
    python scripts/run_hotpot.py --stage cross           # 0. uses no HotpotQA data
    python scripts/score_hotpot.py --pilots              # 1. pilot check (no correctness read)
    python scripts/score_hotpot.py --stage discovery4    # 2. the first four discovery draws
    python scripts/run_hotpot.py --stage predict         # 3. four-draw statements saved from them
    python scripts/score_hotpot.py --stage discovery     # 4. every discovery draw (refused before 3)
    python scripts/run_hotpot.py --stage discovery       # 5. every other rule saved from them
    python scripts/score_hotpot.py --stage all           # 6. every draw (refused before 3 and 5)
    python scripts/run_hotpot.py                         # 7. the analyses

1. --pilots writes results/hotpot/pilot_check.md (and _clean, .json). A pilot passes on a complete 16-request run whose
   log has the prefills and the empty stop list of config/decoding_hotpot.yaml and no prompt-render mismatch, with all
   four arms present. It reports, per arm, the share of outputs with a readable box and the share cut at the cap; no
   gold answer is read.
2. --stage discovery4 scores only the rows of the configurations of the sampler's first four discovery draws into
   data/processed/hotpot/scored_discovery4.parquet. The item set is decided here from completeness alone: all 1,000
   target items when every model's block B folder holds a complete run, otherwise block A (500 items) only, which needs
   --items A. The choice is written to data/processed/hotpot/item_set.json and every
   later stage reads it.
4. --stage discovery refuses unless manifests/predictions_hotpot_tranche.json exists, then scores the rows of the
   configurations of all 24 discovery draws into data/processed/hotpot/scored_discovery.parquet. A validation draw
   that lands on a discovery configuration shares its outputs; that is the with-replacement design.
6. --stage all refuses unless the tranche file exists and manifests/predictions_hotpot_discovery.json (written by
   run_hotpot.py --stage discovery) lists every rule file of step 5 with an unchanged sha256, then scores every draw
   into data/processed/hotpot/scored.parquet. The required files include the fair comparison
   (scripts/hotpot_fair.py): manifests/predictions_hotpot_cross_benchmark.*, saved by run_hotpot.py --stage
   cross from the other benchmarks alone, and the discovery stage's predictions_hotpot_decision_strict.*,
   predictions_hotpot_family.* and predictions_hotpot_cross_discovery.*.
Scored tables: model_key, arm (direct_answer, zero_shot_cot, forced_direct, fewshot_forced), config_id, item_id, block,
strict_correct (exact match, the primary score), lenient_correct, compliant, strict_f1, lenient_f1 (0-1), finish_reason,
n_output_tokens, cached, prompt_sha256, request_id. Metadata in <table>.json: sources, folders, hashes, scorer, item set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_ROOT = os.path.join(ROOT, "data", "d2_out", "hotpot")
PROCESSED = os.path.join(ROOT, "data", "processed", "hotpot")
RESULTS = os.path.join(ROOT, "results", "hotpot")
DECODING = os.path.join(ROOT, "config", "decoding_hotpot.yaml")
SAMPLER = os.path.join(ROOT, "manifests", "sampler_hotpot.json")
TRANCHE_PRED = os.path.join(ROOT, "manifests", "predictions_hotpot_tranche.json")
DISCOVERY_PRED = os.path.join(ROOT, "manifests", "predictions_hotpot_discovery.json")
ITEM_SET = os.path.join(PROCESSED, "item_set.json")
ARMS = ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced")
COLUMNS = ["model_key", "arm", "config_id", "item_id", "block", "strict_correct", "lenient_correct", "compliant",
           "strict_f1", "lenient_f1", "finish_reason", "n_output_tokens", "cached", "prompt_sha256", "request_id"]
N_TRANCHE_DRAWS = 4


def tranche_configs(sampler: dict, n: int = N_TRANCHE_DRAWS) -> list[str]:
    """The distinct configurations of the first n discovery draws, in draw order."""
    return list(dict.fromkeys(map(str, sampler["discovery_config_ids"][:n])))


def discovery_configs(sampler: dict) -> list[str]:
    """The distinct configurations of all discovery draws, in draw order."""
    return list(dict.fromkeys(map(str, sampler["discovery_config_ids"])))


def required_rule_files() -> list[str]:
    """File names the discovery index must list: every rule file of run_hotpot.py --stage discovery, the fair
    comparison included (the cross-benchmark file, the strict decision design, held-out-family choices
    and the cross-benchmark rules at HotpotQA baselines; scripts/hotpot_fair.py)."""
    import run_hotpot
    return [os.path.basename(p) for p in run_hotpot.rule_files("")]


def discovery_index_problems(path: str = DISCOVERY_PRED, root: str = ROOT, required=None) -> list[str]:
    """Why the discovery-stage index does not license scoring the validation draws ([] when it does): it must exist,
    list every required file (required_rule_files() by default), and every file it lists must exist with the sha256
    recorded when it was saved."""
    from src import hotpot_task
    if not os.path.exists(path):
        return [f"{path} is absent"]
    rec = json.load(open(path, encoding="utf-8"))
    files = rec.get("files") or {}
    if not files:
        return [f"{path} lists no rule file"]
    listed = {os.path.basename(k.replace("\\", "/")) for k in files}
    out = [f"{name} is not listed in {path}" for name in (required_rule_files() if required is None else required)
           if name not in listed]
    for rel_path, sha in files.items():
        p = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
        if not os.path.exists(p):
            out.append(f"{rel_path} is absent")
        elif hotpot_task.file_sha256(p) != sha:
            out.append(f"{rel_path} changed after the discovery stage saved it")
    return out


def folders(sub: str, panel, root: str = OUT_ROOT) -> dict[str, str]:
    return {k: os.path.join(root, sub, k) for k in panel}


def complete(d: str, allow_test_backend: bool = False) -> bool:
    from src import d2_io
    try:
        d2_io.inspect_work_dir(d, string_item_ids=True, allow_test_backend=allow_test_backend)
        return True
    except d2_io.D2IOError:
        return False


def decide_items(panel, requested: str | None, root: str = OUT_ROOT, path: str = ITEM_SET,
                 allow_test_backend: bool = False) -> dict:
    """The analysed item set from completeness alone (never from scores); saved once and then fixed."""
    if os.path.exists(path):
        rec = json.load(open(path, encoding="utf-8"))
        if requested and requested != rec["items"]:
            raise SystemExit(f"{path} already fixes the item set to {rec['items']}")
        return rec
    b_done = {k: complete(d, allow_test_backend) for k, d in folders("block_b", panel, root).items()}
    a_done = {k: complete(d, allow_test_backend) for k, d in folders("block_a", panel, root).items()}
    if not all(a_done.values()):
        raise SystemExit(f"block A is incomplete for {[k for k, v in a_done.items() if not v]}")
    if all(b_done.values()):
        items = "all"
    elif requested == "A":
        items = "A"
    else:
        raise SystemExit(f"block B is incomplete for {[k for k, v in b_done.items() if not v]}; pass --items A to analyse "
                         "block A alone, or wait")
    rec = {"items": items, "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": "all 1,000 target items when every model's block B folder holds a complete run, otherwise block A",
           "block_b_complete": b_done}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def score_stage(stage: str, items_flag: str | None = None, panel=None, root: str = OUT_ROOT,
                processed: str = PROCESSED, sampler_path: str = SAMPLER, tranche_pred: str = TRANCHE_PRED,
                allow_test_backend: bool = False, discovery_pred: str = DISCOVERY_PRED, pred_root: str | None = None) -> str:
    from src import d2_io, hotpot_d2, hotpot_task
    if stage in ("discovery", "all") and not os.path.exists(tranche_pred):
        raise SystemExit(f"{tranche_pred} is absent: run run_hotpot.py --stage predict on the four-draw table first "
                         "(hard rule 6)")
    if stage == "all":
        problems = discovery_index_problems(discovery_pred, root=pred_root or ROOT)
        if problems:
            raise SystemExit("the validation draws stay unscored until run_hotpot.py --stage discovery has saved every "
                             "rule from the discovery table (hard rule 6): " + "; ".join(problems))
    panel = list(panel or hotpot_d2.panel())
    sampler = json.load(open(sampler_path, encoding="utf-8"))
    rec = decide_items(panel, items_flag, root, os.path.join(processed, "item_set.json"), allow_test_backend)
    target, tinfo = hotpot_d2.load_items("target")
    if rec["items"] == "A":
        target = target[target["block"] == "A"].reset_index(drop=True)
    blocks = ["A", "B"] if rec["items"] == "all" else ["A"]
    keep_cfg = (set(tranche_configs(sampler)) if stage == "discovery4"
                else set(discovery_configs(sampler)) if stage == "discovery" else None)
    scorer = hotpot_d2.load_scorer()
    parts, meta = [], {"stage": stage, "items": rec["items"], "n_items": int(len(target)), "scorer": scorer["scorer_version"],
                       "scorer_rules_sha256": scorer["rules_sha256"], "item_table": tinfo["path"],
                       "item_canonical_sha256": tinfo["canonical_sha256"], "sampler_sha256": sampler["sha256"],
                       "configs": sorted(keep_cfg) if keep_cfg else "all", "folders": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    for k in panel:
        for b in blocks:
            d = os.path.join(root, f"block_{b.lower()}", k)
            out, info = d2_io.inspect_work_dir(d, string_item_ids=True, allow_test_backend=allow_test_backend)
            if keep_cfg is not None:
                out = out[out["config_id"].astype(str).isin(keep_cfg)]
            out = out[out["item_id"].astype(str).isin(set(target["item_id"]))]
            sc = hotpot_task.score_outputs(out.reset_index(drop=True), target)
            sc["block"] = b
            parts.append(sc[[c for c in COLUMNS if c in sc.columns]])
            meta["folders"][f"{k}/{b}"] = {"path": os.path.relpath(d, ROOT).replace("\\", "/"),
                                           "run_fingerprint": info["run_fingerprint"], "gpu_type": info["gpu_type"],
                                           "prefill": info["prefill"], "n_outputs": info["n_outputs"],
                                           "test_backend": info["test_backend"]}
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["model_key", "arm", "config_id", "item_id"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicated (model, arm, configuration, item) rows")
    os.makedirs(processed, exist_ok=True)
    name = {"discovery4": "scored_discovery4", "discovery": "scored_discovery"}.get(stage, "scored")
    path = os.path.join(processed, f"{name}.parquet")
    df.to_parquet(path, index=False, compression="zstd")
    meta.update({"n_rows": int(len(df)), "by_arm": {a: int(v) for a, v in df["arm"].value_counts().sort_index().items()},
                 "empty_strict_answers": int((df["compliant"] == False).sum()),  # noqa: E712
                 "sha256": hotpot_task.file_sha256(path)})
    with open(os.path.join(processed, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{name}: {len(df)} rows, items {rec['items']} ({len(target)}), {meta['by_arm']}")
    return path


# ============================================================================ pilots


def check_pilots(panel=None, root: str = OUT_ROOT, results: str = RESULTS, allow_test_backend: bool = False) -> dict:
    from src import d2_io, generation as gen, hotpot_d2, hotpot_task
    dec = yaml.safe_load(open(DECODING, encoding="utf-8"))
    want_prefill, want_stop = dict(dec["prefill"]), list(dec["stop"])
    rows = []
    for k in (panel or hotpot_d2.panel()):
        d = os.path.join(root, "pilot", k)
        row = {"model_key": k, "folder": os.path.relpath(d, ROOT).replace("\\", "/")}
        logs = sorted(p for p in os.listdir(d) if p.startswith("run_log-")) if os.path.isdir(d) else []
        if not logs:
            rows.append({**row, "status": "not returned yet", "passes": False, "pending": True})
            continue
        L = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in logs]
        done = [x for x in L if x.get("status") == "complete"]
        ok_dec = all(dict((x.get("decoding") or {}).get("prefill") or {}) == want_prefill
                     and list((x.get("decoding") or {}).get("stop") or []) == want_stop for x in done)
        mism = all(x.get("prompt_render_mismatches") == 0 for x in done)
        row["gpu"] = [list(gen.gpu_type(x.get("gpu"))) for x in done]
        if done:
            out, _ = d2_io.inspect_work_dir(d, string_item_ids=True, allow_incomplete=True,
                                            allow_test_backend=allow_test_backend)
            arms_ok = set(out["arm"]) == set(ARMS) and len(out) == 16
            row["arms"] = {}
            for a, g in out.groupby("arm"):
                t = g["output_text"].astype(str).tolist()
                row["arms"][a] = {"n": int(len(g)), "readable_box": int(sum(hotpot_task.compliance(x) for x in t)),
                                  "cut_at_cap": int((g["finish_reason"] == "length").sum()),
                                  "example": t[0][:200]}
        else:
            arms_ok = False
        passes = bool(done) and ok_dec and mism and arms_ok
        row["status"] = ("passes" if passes else "no complete run" if not done
                         else "decoding differs from config/decoding_hotpot.yaml" if not ok_dec
                         else "rendered prompts differ between runner and engine" if not mism
                         else "not the 16 requests of the four arms")
        row["passes"] = passes
        rows.append(row)
    res = {"rule": "passes: a complete 16-request run with the prefills and empty stop list of config/decoding_hotpot.yaml, "
                   "prompt_render_mismatches 0 and all four arms; no gold answer is read",
           "models": rows, "n_pass": sum(r["passes"] for r in rows), "n_models": len(rows),
           "pending": [r["model_key"] for r in rows if r.get("pending")],
           "failed": [r["model_key"] for r in rows if not r["passes"] and not r.get("pending")]}
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "pilot_check.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = ["# HotpotQA pilots", "", res["rule"] + ".", "",
         f"{res['n_pass']} of {res['n_models']} pilots pass."
         + (f" Failed: {', '.join(res['failed'])}." if res["failed"] else "")
         + (f" Not returned yet: {len(res['pending'])}." if res["pending"] else ""), "",
         "| Model | Verdict | Readable box (direct, chain of thought, forced, five-shot) | Cut at the cap |", "|---|---|---|---|"]
    for r in rows:
        a = r.get("arms") or {}
        rb = ", ".join(f"{a[x]['readable_box']}/{a[x]['n']}" if x in a else "--" for x in ARMS)
        cut = ", ".join(str(a[x]["cut_at_cap"]) if x in a else "--" for x in ARMS)
        L.append(f"| {r['model_key']} | {r['status']} | {rb} | {cut} |")
    txt = "\n".join(L) + "\n"
    for name in ("pilot_check.md", "pilot_check_clean.md"):
        with open(os.path.join(results, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"pilots: {res['n_pass']} of {res['n_models']} pass" + (f"; failed {res['failed']}" if res["failed"] else ""))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--stage", choices=("discovery4", "discovery", "all"))
    ap.add_argument("--items", choices=("A",), help="analyse block A alone when block B is incomplete")
    a = ap.parse_args(argv)
    if a.pilots:
        check_pilots()
    else:
        score_stage(a.stage, a.items)


if __name__ == "__main__":
    main()
