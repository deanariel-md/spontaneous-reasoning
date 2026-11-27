r"""Pilot check and scoring of the eight-exemplar arm.

From the repository root, after the folders are copied back from the GPU machine:
    python scripts/score_exemplar.py --pilots
    python scripts/score_exemplar.py --score
    python scripts/score_exemplar.py --score --allow-missing     # a diagnostic table

Generation folders: data/d2_out/exemplar/pilot/<key> (16 requests) and data/d2_out/exemplar/gsm8k/<key>. The
zero-shot arm is the collection's (data/d2_out/collection/<key> and data/d2_out/collection_ext/<key>, scored in
data/processed/gsm8k/scored.parquet and data/processed/gsm8k_ext/scored_<key>.parquet).

--pilots: a pilot passes when its run completed with --limit 16 and the engine's rendered prompts equal the runner's
(prompt_render_mismatches 0). Beside the verdict: outputs, readable answers (strict extraction) and outputs cut at the
cap. No correctness is read. Output: results/exemplar/pilot_check.json, pilot_check.md and its clean copy.

--score writes, in data/processed/exemplar/:
- gsm8k_scored.parquet: the eight-shot outputs scored by scripts/score_d2.py with the GSM8K scorer
  (manifests/gsm8k_scorer.json), matched to a build record derived from the collection's records (the same bank,
  sampler and items; each model's eight-shot request file and hash) in data/requests/exemplar/scoring/;
- zero_shot_4k.parquet: the collection's zero-shot outputs of the four checkpoints run at 2,048 tokens
  (config/decoding_exemplar_4k.yaml) that are longer than 2,048 tokens, cut to their first 2,048 tokens under the
  checkpoint's tokenizer (data/tokenizers/<key>/tokenizer.json) and scored again, so both arms of those models are
  read at the same cap;
- cheng_scored.parquet: both arms scored with Cheng et al.'s corrected extractor (src/reproduction.py, a
  sensitivity check), the zero-shot outputs of the four checkpoints cut as above.
The sidecars hold no accuracy.
"""
import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import d2_io, generation as gen  # noqa: E402

EX_OUT = os.path.join(ROOT, "data", "d2_out", "exemplar")
REQ = os.path.join(ROOT, "data", "requests", "exemplar")
PROCESSED = os.path.join(ROOT, "data", "processed", "exemplar")
RESULTS = os.path.join(ROOT, "results", "exemplar")
COLLECTION_OUT = (os.path.join(ROOT, "data", "d2_out", "collection"), os.path.join(ROOT, "data", "d2_out", "collection_ext"))
COLLECTION_REQ = (os.path.join(ROOT, "data", "requests", "gsm8k"), os.path.join(ROOT, "data", "requests", "gsm8k_ext"))
ARM = "eight_shot_cot"
CAP_4K = 2048
# the checkpoints run with config/decoding_exemplar_4k.yaml (scripts/build_exemplar_requests.NATIVE_4K)
NATIVE_4K = ("llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "phi-3-mini-4k-instruct", "phi-3-medium-4k-instruct")


def exemplar_record():
    return json.load(open(os.path.join(REQ, "gsm8k", "build_record.json"), encoding="utf-8"))


def collection_record_path(key):
    """The build record of the collection that holds the model (as scripts/build_exemplar_requests.py)."""
    for p in (os.path.join(COLLECTION_REQ[0], "build_record.json"),
              os.path.join(COLLECTION_REQ[1], f"collection_{key}", "build_record.json")):
        if os.path.isfile(p) and key in json.load(open(p, encoding="utf-8"))["models"]:
            return p
    raise SystemExit(f"{key}: no collection build record holds it")


def derived_record():
    """The collection's build record with each model's eight-shot request file in place of its own (module doc)."""
    ex = exemplar_record()
    base = None
    models = {}
    for k, ent in sorted(ex["models"].items()):
        p = collection_record_path(k)
        rec = json.load(open(p, encoding="utf-8"))
        sig = (rec["kind"], rec.get("split", "test"), rec["bank"]["sha256"], rec["sampler"]["registered_sha256"],
               rec["items"]["canonical_sha256"])
        if base is None:
            base, base_sig = rec, sig
        elif sig != base_sig:
            raise SystemExit(f"{d2_io.rel_path(p)}: its bank, sampler or items differ from the other collection records")
        if rec["bank"]["sha256"] != ex["bank"]["sha256"] or rec["items"]["canonical_sha256"] != ex["items"]["canonical_sha256"]:
            raise SystemExit(f"{k}: the eight-shot requests were built from another bank or item set than {d2_io.rel_path(p)}")
        models[k] = {"n_requests": ent["n_requests"], "n_configurations": len(ex["configurations"]),
                     "counts_by_arm": {ARM: ent["n_requests"]}, "requests_file": f"gsm8k/{ent['requests_file']}",
                     "requests_sha256": ent["requests_sha256"], "registered": True,
                     "derived_from": {"record": d2_io.rel_path(p), "exemplar_record": "data/requests/exemplar/gsm8k/build_record.json"}}
    rec = {k: v for k, v in base.items() if k not in ("models", "_path")}
    rec.update({"arms": [ARM], "purpose": "exemplar_collection", "models": models,
                "derived": ("Cheng et al.'s eight exemplars before each frame of the collection's "
                            "chain-of-thought requests (scripts/build_exemplar_requests.py); bank, sampler and items as "
                            "in the collection's records")})
    d = os.path.join(REQ, "scoring")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "build_record.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return path, rec


def gsm8k_scorer():
    from src import gsm8k
    spec, record = d2_io.load_registered_scorer(os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    items, _ = d2_io.load_items(split="test")
    return spec, record, gsm8k.gold_lookup(items)


def check_pilots():
    from src import scoring
    spec, record, gold = gsm8k_scorer()
    keys = sorted(exemplar_record()["models"])
    rows = []
    for k in keys:
        d = os.path.join(EX_OUT, "pilot", k)
        row = {"model_key": k, "folder": d2_io.rel_path(d)}
        logs = sorted(p for p in os.listdir(d) if p.startswith("run_log-")) if os.path.isdir(d) else []
        if not logs:
            rows.append({**row, "status": "not returned yet", "passes": False, "pending": True})
            continue
        L = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in logs]
        done = [x for x in L if x.get("status") == "complete" and (x.get("inputs") or {}).get("limit") == 16]
        passes = bool(done) and all(x.get("prompt_render_mismatches") == 0 for x in done)
        row.update({"status": "passes" if passes else "no complete 16-request run" if not done
                    else "rendered prompts differ between runner and engine", "passes": passes,
                    "gpu": [list(gen.gpu_type(x.get("gpu"))) for x in done]})
        if done:
            out, _ = d2_io.inspect_work_dir(d, allow_incomplete=True)
            sc = scoring.score_outputs(out, gold, spec)
            row.update({"n_outputs": int(len(out)), "readable": int(sc["compliant"].sum()),
                        "finish_length": int((out["finish_reason"] == "length").sum())})
        rows.append(row)
    res = {"rule": "passes: a complete --limit 16 run and prompt_render_mismatches 0. No correctness is read.",
           "scorer": record["scorer_version"], "models": rows, "n_pass": sum(r["passes"] for r in rows),
           "n_models": len(rows), "pending": [r["model_key"] for r in rows if r.get("pending")],
           "failed": [r["model_key"] for r in rows if not r["passes"] and not r.get("pending")]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pilot_check.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Eight-exemplar pilots", "",
         f"Each model ran 16 eight-shot requests before its run. {res['rule']}", "",
         f"{res['n_pass']} of {res['n_models']} pilots pass." + (f" Failed: {', '.join(res['failed'])}." if res["failed"]
                                                                  else "")
         + (f" Not returned yet: {len(res['pending'])}." if res["pending"] else ""), "",
         "| Model | Verdict | Outputs | Readable answer | Cut at the cap |", "|---|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model_key']} | {r['status']} | {r.get('n_outputs', '--')} | {r.get('readable', '--')} | "
                 f"{r.get('finish_length', '--')} |")
    txt = "\n".join(L) + "\n"
    for name in ("pilot_check.md", "pilot_check_clean.md"):
        with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"pilots: {res['n_pass']} of {res['n_models']} pass" + (f"; failed {res['failed']}" if res["failed"] else "")
          + (f"; not returned yet {len(res['pending'])}" if res["pending"] else ""))
    return res


def collection_dir(key):
    for root in COLLECTION_OUT:
        d = os.path.join(root, key)
        if os.path.isdir(d):
            return d
    raise SystemExit(f"{key}: no collection folder in {[d2_io.rel_path(r) for r in COLLECTION_OUT]}")


def zero_shot_outputs(key, configs):
    """The collection's zero-shot outputs of the model on the sampler's configurations, after its folder checks."""
    out, info = d2_io.inspect_work_dir(collection_dir(key))
    out = out[(out["arm"] == "zero_shot_cot") & out["config_id"].isin(configs)]
    return out.reset_index(drop=True), info


def cut_to_cap(texts, key, cap=CAP_4K):
    """Each text cut to its first `cap` tokens under the checkpoint's tokenizer (texts not longer stay as they are)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizers", key, "tokenizer.json"))
    cut, n = [], []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        n.append(len(ids))
        cut.append(t if len(ids) <= cap else tok.decode(ids[:cap], skip_special_tokens=True))
    return cut, n


def score_4k(configs):
    """zero_shot_4k.parquet: the long zero-shot outputs of NATIVE_4K, cut to 2,048 tokens and scored again."""
    from src import scoring
    spec, record, gold = gsm8k_scorer()
    rows = []
    for k in NATIVE_4K:
        out, _ = zero_shot_outputs(k, configs)
        long = out[out["n_output_tokens"] > CAP_4K].copy()
        if long.empty:
            continue
        before = scoring.score_outputs(long, gold, spec)
        long["output_text"], ntok = cut_to_cap(long["output_text"].astype(str).tolist(), k)
        after = scoring.score_outputs(long, gold, spec)
        rows.append(pd.DataFrame({"model_key": k, "arm": "zero_shot_cot", "config_id": long["config_id"].to_numpy(),
                                  "item_id": long["item_id"].to_numpy(),
                                  "n_output_tokens": long["n_output_tokens"].to_numpy(), "retokenized": ntok,
                                  "strict_before": before["strict_correct"].to_numpy(),
                                  "lenient_before": before["lenient_correct"].to_numpy(),
                                  "strict_correct": after["strict_correct"].to_numpy(),
                                  "lenient_correct": after["lenient_correct"].to_numpy(),
                                  "compliant": after["compliant"].to_numpy(), "finish_reason": "length",
                                  "output_text_cut": long["output_text"].to_numpy()}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df.to_parquet(os.path.join(PROCESSED, "zero_shot_4k.parquet"), index=False)
    summ = {k: {"n_cut": int(len(g)), "strict_changed": int((g["strict_before"] != g["strict_correct"]).sum()),
                "lenient_changed": int((g["lenient_before"] != g["lenient_correct"]).sum())}
            for k, g in df.groupby("model_key")} if len(df) else {}
    print("zero-shot outputs cut to 2,048 tokens: " + json.dumps(summ))
    return df, summ


def score_cheng(keys, configs, cut4k):
    """cheng_scored.parquet: both arms with the authors' corrected extractor (src/reproduction.py)."""
    from src import gsm8k, reproduction as rp
    items = gsm8k.load_gsm8k("test")
    parts = []
    for k in keys:
        d = os.path.join(EX_OUT, "gsm8k", k)
        if not os.path.isdir(d):
            continue
        ex, _ = d2_io.inspect_work_dir(d)
        zs, _ = zero_shot_outputs(k, configs)
        if k in NATIVE_4K and len(cut4k):
            c = cut4k[cut4k["model_key"] == k].set_index(["config_id", "item_id"])["output_text_cut"]
            idx = pd.MultiIndex.from_arrays([zs["config_id"], zs["item_id"]])
            hit = idx.isin(c.index)
            zs.loc[hit, "output_text"] = c.reindex(idx[hit]).to_numpy()
            zs.loc[hit, "finish_reason"] = "length"
        for o in (ex, zs):
            parts.append(rp.score_outputs(o, items, "exemplar_h"))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(os.path.join(PROCESSED, "cheng_scored.parquet"), index=False)
    print(f"cheng_scored.parquet: {len(df)} rows ({rp.SCORER_VERSION})")
    return df


def score(allow_missing=False):
    import score_d2
    path, rec = derived_record()
    keys = sorted(rec["models"])
    dirs = [os.path.join(EX_OUT, "gsm8k", k) for k in keys]
    missing = [k for k, d in zip(keys, dirs) if not os.path.isdir(d)]
    if missing and not allow_missing:
        raise SystemExit(f"no eight-shot folder for {missing}")
    dirs = [d for d in dirs if os.path.isdir(d)]
    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, "gsm8k_scored.parquet")
    args = Namespace(task="gsm8k", work_dir=dirs, scorer=None, candidate_spec=None, out=out, tranche=False,
                     screening=False, split=None, requests_dir=os.path.dirname(path), items_parquet=None,
                     allow_test_backend=False, allow_incomplete=False, allow_unmatched_requests=False)
    res = score_d2.run(args)
    configs = set(map(str, exemplar_record()["configurations"]))
    cut4k, summ = score_4k(configs)
    score_cheng(keys, configs, cut4k)
    with open(os.path.join(PROCESSED, "zero_shot_4k.json"), "w", encoding="utf-8") as fh:
        json.dump({"cap": CAP_4K, "models": list(NATIVE_4K), "by_model": summ,
                   "rule": "a collection zero-shot output longer than 2,048 tokens is cut to its first 2,048 tokens under "
                           "the checkpoint's tokenizer and scored again"}, fh, indent=1)
    print(json.dumps({k: res[k] for k in ("out", "n_rows", "models", "scorer_version")}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score the complete folders present (a diagnostic table; the analysis refuses a partial panel "
                         "unless run with --partial)")
    a = ap.parse_args(argv)
    if a.pilots:
        check_pilots()
    else:
        score(a.allow_missing)


if __name__ == "__main__":
    main()
