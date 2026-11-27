r"""Pilot check and scoring of the forced-direct arm.

From the repository root:
    python scripts/score_forced_direct.py --pilots          # after the 16-request pilots
    python scripts/score_forced_direct.py --task gsm8k      # after a task's runs
    python scripts/score_forced_direct.py --task bbh
    python scripts/score_forced_direct.py --task math     # MATH-500

Generation folders: data/d2_out/forced/pilot/<key>, data/d2_out/forced/gsm8k/<key>, data/d2_out/forced/bbh/<key> and
data/d2_out/math/forced/<key> (copied back from the GPU machine). The runner's log records the decoding, prefill
included, and src/d2_io.py prepends the
prefill to each output, so every scorer reads the assistant's whole message: `\boxed{` followed by the output.

--pilots: for each model, the pilot passes when its run completed with --limit 16, the logged prefill is the one of
config/decoding_forced.yaml, and the engine's rendered prompts equal the runner's (prompt_render_mismatches 0). A
model whose run failed at rendering has no complete pilot and is listed as failed: it falls back to
config/decoding_forced_noprefill.yaml and is flagged. Beside the verdict: the share of
outputs that close the box and the share with a readable answer (strict extraction not empty). No correctness is
read. Output: results/forced_direct/pilot_check.json and pilot_check.md (with a clean copy).

--task: the build record of the task's forced requests is derived from the collection's records (the same bank,
sampler and items; each model's forced request file and hash), written to data/requests/forced/<task>/
build_record.json, and the folders are scored by scripts/score_d2.py with the scorer of the task into
data/processed/forced/<task>_scored.parquet (strict, lenient, compliant). Beside it,
data/processed/forced/<task>_box.parquet holds box_closed per output (the closing brace is present). The sidecar
holds no accuracy.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import d2_io, generation as gen  # noqa: E402

FORCED_OUT = os.path.join(ROOT, "data", "d2_out", "forced")
REQ = os.path.join(ROOT, "data", "requests", "forced")
PROCESSED = os.path.join(ROOT, "data", "processed", "forced")
RESULTS = os.path.join(ROOT, "results", "forced_direct")
DECODING = os.path.join(ROOT, "config", "decoding_forced.yaml")
ARM = "forced_direct"


def forced_record():
    return json.load(open(os.path.join(REQ, "build_record.json"), encoding="utf-8"))


def derived_record(task):
    """The collection's build record with each model's forced request file in place of its own (module doc)."""
    fr = forced_record()["tasks"][task]
    sources = {}
    for k, ent in fr.items():
        p = os.path.join(ROOT, os.path.dirname(ent["source"]), "build_record.json")
        sources.setdefault(p, []).append(k)
    base = None
    models = {}
    for p, keys in sorted(sources.items()):
        rec = json.load(open(p, encoding="utf-8"))
        sig = (rec["kind"], rec.get("split", "test"), rec["bank"]["sha256"], rec["sampler"]["registered_sha256"],
               rec["items"]["canonical_sha256"])
        if base is None:
            base, base_sig = rec, sig
        elif sig != base_sig:
            raise SystemExit(f"{d2_io.rel_path(p)}: its bank, sampler or items differ from the other collection records")
        for k in keys:
            src = rec["models"][k]
            ent = fr[k]
            if src["requests_file"] != os.path.basename(ent["source"]):
                raise SystemExit(f"{k}: the forced requests were built from {ent['source']}, not from the request "
                                 f"file of {d2_io.rel_path(p)}")
            models[k] = {"n_requests": ent["n"], "n_configurations": ent["configs"],
                         "counts_by_arm": {ARM: ent["n"]}, "requests_file": f"{task}/{ent['requests_file']}",
                         "requests_sha256": ent["requests_sha256"], "registered": src.get("registered", True),
                         "derived_from": {"record": d2_io.rel_path(p), "requests_file": src["requests_file"],
                                          "requests_sha256": src["requests_sha256"]}}
    rec = {k: v for k, v in base.items() if k not in ("models", "_path")}
    rec.update({"arms": [ARM], "purpose": "forced_direct_collection", "models": models,
                "derived": ("the collection's direct-answer requests with arm "
                            "forced_direct (scripts/build_forced_direct.py); bank, sampler and items as in "
                            + ", ".join(sorted(d2_io.rel_path(p) for p in sources)))})
    path = os.path.join(REQ, task, "build_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return path, rec


def box_closed(texts, task=None):
    """Whether the box closes: a closing brace after the prefill (the stop string is kept in the output). MATH runs
    with no stop string and its answers may hold braces (config/decoding_forced_math.yaml), so there a complete
    \\boxed{} must be present."""
    if task == "math":
        from src import bbh
        return np.array([bool(bbh.boxed_contents(t)) for t in texts])
    return np.array(["}" in t[len("\\boxed{"):] if t.startswith("\\boxed{") else "}" in t for t in texts])


def check_pilots():
    want = yaml.safe_load(open(DECODING, encoding="utf-8"))["prefill"]
    from src import gsm8k, scoring
    spec, record = d2_io.load_registered_scorer(os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    items, _ = d2_io.load_items(split="test")
    gold = gsm8k.gold_lookup(items)
    keys = sorted(forced_record()["tasks"]["gsm8k"])
    rows = []
    for k in keys:
        d = os.path.join(FORCED_OUT, "pilot", k)
        row = {"model_key": k, "folder": d2_io.rel_path(d)}
        logs = sorted(p for p in os.listdir(d) if p.startswith("run_log-")) if os.path.isdir(d) else []
        if not logs:
            rows.append({**row, "status": "not returned yet", "passes": False, "pending": True})
            continue
        L = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in logs]
        done = [x for x in L if x.get("status") == "complete" and (x.get("inputs") or {}).get("limit") == 16]
        prefill = [dict((x.get("decoding") or {}).get("prefill") or {}) for x in done]
        mism = [x.get("prompt_render_mismatches") for x in done]
        row.update({"runs": [{"status": x.get("status"), "limit": (x.get("inputs") or {}).get("limit"),
                              "prompt_render_mismatches": x.get("prompt_render_mismatches"),
                              "warnings": x.get("warnings", []), "gpu": list(gen.gpu_type(x.get("gpu")))} for x in L]})
        passes = bool(done) and all(p == want for p in prefill) and all(m == 0 for m in mism)
        row["status"] = ("passes" if passes else "no complete 16-request run" if not done
                         else "prefill differs from config/decoding_forced.yaml" if any(p != want for p in prefill)
                         else "rendered prompts differ between runner and engine")
        row["passes"] = passes
        if done:
            out, _ = d2_io.inspect_work_dir(d, allow_incomplete=True)
            sc = scoring.score_outputs(out, gold, spec)
            closed = box_closed(out["output_text"].astype(str).tolist())
            row.update({"n_outputs": int(len(out)), "box_closed": int(closed.sum()),
                        "readable": int(sc["compliant"].sum()),
                        "closed_and_readable": int((closed & sc["compliant"].to_numpy(bool)).sum()),
                        "finish_length": int((out["finish_reason"] == "length").sum()),
                        "examples": out["output_text"].astype(str).head(3).tolist()})
        rows.append(row)
    res = {"rule": "passes: a complete --limit 16 run, the prefill of config/decoding_forced.yaml in its log, and "
                   "prompt_render_mismatches 0; otherwise the model falls back to config/decoding_forced_noprefill.yaml "
                   "and is flagged. No correctness is read.",
           "prefill": want, "scorer": record["scorer_version"], "models": rows,
           "n_pass": sum(r["passes"] for r in rows), "n_models": len(rows),
           "pending": [r["model_key"] for r in rows if r.get("pending")],
           "failed": [r["model_key"] for r in rows if not r["passes"] and not r.get("pending")]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pilot_check.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Forced-direct pilots", "",
         f"Each model ran 16 GSM8K requests with the assistant turn prefilled with "
         f"`{want[ARM]}` before its runs. {res['rule']}", "",
         f"{res['n_pass']} of {res['n_models']} pilots pass." + (f" Failed: {', '.join(res['failed'])}." if res["failed"]
                                                                  else "")
         + (f" Not returned yet: {len(res['pending'])}." if res["pending"] else ""), "",
         "| Model | Verdict | Outputs | Box closed | Readable answer | Closed and readable | Cut at 16 tokens |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model_key']} | {r['status']} | {r.get('n_outputs', '--')} | {r.get('box_closed', '--')} | "
                 f"{r.get('readable', '--')} | {r.get('closed_and_readable', '--')} | {r.get('finish_length', '--')} |")
    txt = "\n".join(L) + "\n"
    for name in ("pilot_check.md", "pilot_check_clean.md"):
        with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"pilots: {res['n_pass']} of {res['n_models']} pass" + (f"; failed {res['failed']}" if res["failed"] else "")
          + (f"; not returned yet {len(res['pending'])}" if res["pending"] else ""))
    return res


def forced_dir(task, key):
    """A model's forced-direct folder; MATH-500's sits beside its tranche and collection (data/d2_out/math/forced/<key>,
    where G's rotation copies it back)."""
    if task == "math":
        return os.path.join(ROOT, "data", "d2_out", "math", "forced", key)
    return os.path.join(FORCED_OUT, task, key)


def score_task(task, allow_missing=False):
    import score_d2
    path, rec = derived_record(task)
    keys = sorted(rec["models"])
    dirs = [forced_dir(task, k) for k in keys]
    missing = [k for k, d in zip(keys, dirs) if not os.path.isdir(d)]
    if missing and not allow_missing:
        raise SystemExit(f"{task}: no forced-direct folder for {missing}")
    dirs = [d for d in dirs if os.path.isdir(d)]
    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, f"{task}_scored.parquet")
    args = Namespace(task=task, work_dir=dirs, scorer=None, candidate_spec=None, out=out, tranche=False,
                     screening=False, split=None, requests_dir=os.path.dirname(path), items_parquet=None,
                     allow_test_backend=False, allow_incomplete=allow_missing, allow_unmatched_requests=False)
    res = score_d2.run(args)
    box = []
    for d in dirs:
        o, info = d2_io.inspect_work_dir(d, allow_incomplete=allow_missing, string_item_ids=(task in ("bbh", "math")))
        if info["prefill"].get(ARM) is None:
            print(f"warning: {d2_io.rel_path(d)} ran without a prefill (the fallback decoding)", file=sys.stderr)
        box.append(pd.DataFrame({"model_key": o["model_key"].astype(str), "arm": o["arm"].astype(str),
                                 "config_id": o["config_id"].astype(str), "item_id": o["item_id"],
                                 "box_closed": box_closed(o["output_text"].astype(str).tolist(), task),
                                 "prefill": info["prefill"].get(ARM, "")}))
    pd.concat(box, ignore_index=True).to_parquet(os.path.join(PROCESSED, f"{task}_box.parquet"), index=False)
    print(json.dumps({k: res[k] for k in ("out", "n_rows", "models", "scorer_version")}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--task", choices=("gsm8k", "bbh", "math"))
    ap.add_argument("--allow-missing", action="store_true",
                    help="score the folders present (a diagnostic table; the analysis refuses an incomplete panel)")
    a = ap.parse_args(argv)
    if a.pilots:
        check_pilots()
    else:
        score_task(a.task, a.allow_missing)


if __name__ == "__main__":
    main()
