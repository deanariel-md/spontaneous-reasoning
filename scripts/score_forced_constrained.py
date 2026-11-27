r"""Pilot check and scoring of the constrained forced-direct arm on BIG-Bench Hard.

From the repository root:
    python scripts/score_forced_constrained.py --pilots      # after the 25-request pilots
    python scripts/score_forced_constrained.py --score       # after every model's run

Generation folders (copied back from the GPU machine): data/d2_out/forced_constrained/pilot/<key> and
data/d2_out/forced_constrained/bbh/<key>.

--pilots: per model, the pilot passes when its run completed, its log records the prefill "\boxed{" and guided
decoding of the arm, the engine's rendered prompts equal the runner's (prompt_render_mismatches 0), the prompt of every
pilot request equals the forced arm's prompt for the same configuration and item (prompt_sha256 of
data/d2_out/forced/bbh/<key>), every output is one of the request's allowed strings (or matches its pattern), and the
scorer reads an answer in every output. No correctness is read. Output:
results/forced_constrained/pilot_check.json and pilot_check.md (with a clean copy).

--score: the forced arm's scoring path (scripts/score_forced_direct.py) on the constrained folders: the derived build
record in data/requests/forced_constrained/bbh/build_record.json, the scorer bbh-boxed into
data/processed/forced_constrained/bbh_scored.parquet, and box closure in bbh_box.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQ = os.path.join(ROOT, "data", "requests", "forced_constrained")
OUT = os.path.join(ROOT, "data", "d2_out", "forced_constrained")
PROCESSED = os.path.join(ROOT, "data", "processed", "forced_constrained")
RESULTS = os.path.join(ROOT, "results", "forced_constrained")
FORCED_OUT = os.path.join(ROOT, "data", "d2_out", "forced", "bbh")
PREFILL = "\\boxed{"


def allowed(output: str, guided: dict) -> bool:
    """Whether an output (the text after the prefill, the stop string kept) satisfies the request's constraint."""
    if "choice" in guided:
        return output in guided["choice"]
    return re.fullmatch(guided["regex"], output) is not None


def check_pilot_folder(d: str, req: pd.DataFrame, forced_prompts: dict | None) -> dict:
    """The pilot verdict of one model's folder (module doc)."""
    from src import bbh_d2
    from src import generation as gen
    logs = sorted(p for p in os.listdir(d) if p.startswith("run_log-")) if os.path.isdir(d) else []
    if not logs:
        return {"status": "not returned yet", "passes": False, "pending": True}
    L = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in logs]
    done = [x for x in L if x.get("status") == "complete"]
    if not done:
        return {"status": "no complete run", "passes": False}
    dec = done[-1].get("decoding") or {}
    problems = []
    if (dec.get("prefill") or {}).get("forced_direct") != PREFILL:
        problems.append("prefill differs")
    if (dec.get("guided") or {}).get("forced_direct") != "request":
        problems.append("the log records no guided decoding")
    if any(x.get("prompt_render_mismatches") != 0 for x in done):
        problems.append("rendered prompts differ between runner and engine")
    out = gen.read_outputs(d)
    out["request_id"] = out["request_id"].astype(str)
    m = out.merge(req[["request_id", "guided_json"]].assign(request_id=req["request_id"].astype(str)), on="request_id",
                  how="left")
    if m["guided_json"].isna().any() or len(m) != len(req):
        problems.append(f"{len(m)} outputs for {len(req)} pilot requests")
    ok_c = [allowed(o, json.loads(g)) for o, g in zip(m["output_text"].astype(str), m["guided_json"].fillna("{}"))
            if g != "{}"]
    from src import bbh
    items = bbh_d2.load_items("target")[0]
    types = dict(zip(items["item_id"].astype(str), items["answer_type"]))
    readable = sum(bbh.strict_extract(PREFILL + t, bbh.AnswerType.from_label(types[str(i)])) is not None
                   for t, i in zip(m["output_text"].astype(str), m["item_id"].astype(str)))
    same_prompt = None
    if forced_prompts is not None:
        key = list(zip(m["config_id"].astype(str), m["item_id"].astype(str)))
        same_prompt = sum(forced_prompts.get(k) == p for k, p in zip(key, m["prompt_sha256"].astype(str)))
        if same_prompt != len(m):
            problems.append(f"{len(m) - same_prompt} prompts differ from the forced arm's")
    if sum(ok_c) != len(m):
        problems.append(f"{len(m) - sum(ok_c)} outputs outside their constraint")
    if readable != len(m):
        problems.append(f"{len(m) - readable} outputs the scorer cannot read")
    return {"status": "passes" if not problems else "; ".join(problems), "passes": not problems,
            "n_outputs": int(len(m)), "allowed": int(sum(ok_c)), "readable": readable, "same_prompt_as_forced": same_prompt,
            "gpu": list(gen.gpu_type(done[-1].get("gpu"))), "examples": m["output_text"].astype(str).head(5).tolist()}


def forced_prompt_index(key: str, pilot: pd.DataFrame) -> dict | None:
    """{(config_id, item_id): prompt_sha256} of the forced arm's outputs for the pilot's cells, or None without them."""
    from src import generation as gen
    d = os.path.join(FORCED_OUT, key)
    if not os.path.isdir(d):
        return None
    o = gen.read_outputs(d)
    o = o[o["config_id"].astype(str).isin(set(pilot["config_id"].astype(str)))]
    return dict(zip(zip(o["config_id"].astype(str), o["item_id"].astype(str)), o["prompt_sha256"].astype(str)))


def check_pilots() -> dict:
    rec = json.load(open(os.path.join(REQ, "build_record.json"), encoding="utf-8"))
    rows = []
    for k, ent in sorted(rec["tasks"]["bbh"].items()):
        req = pd.read_parquet(os.path.join(REQ, ent["pilot_file"]))
        r = check_pilot_folder(os.path.join(OUT, "pilot", k), req, forced_prompt_index(k, req))
        rows.append({"model_key": k, **r})
    res = {"rule": "passes: a complete run, the prefill and guided decoding in its log, prompt_render_mismatches 0, "
                   "prompts equal to the forced arm's, every output allowed by its constraint and readable by the "
                   "scorer. No correctness is read.",
           "models": rows, "n_pass": sum(r["passes"] for r in rows), "n_models": len(rows),
           "pending": [r["model_key"] for r in rows if r.get("pending")],
           "failed": [r["model_key"] for r in rows if not r["passes"] and not r.get("pending")]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pilot_check.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Constrained forced-direct pilots", "", res["rule"], "",
         f"{res['n_pass']} of {res['n_models']} pilots pass."
         + (f" Failed: {', '.join(res['failed'])}." if res["failed"] else "")
         + (f" Not returned yet: {len(res['pending'])}." if res["pending"] else ""), "",
         "| Model | Verdict | Outputs | Allowed | Readable | Prompt as forced |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model_key']} | {r['status']} | {r.get('n_outputs', '--')} | {r.get('allowed', '--')} | "
                 f"{r.get('readable', '--')} | {r.get('same_prompt_as_forced', '--')} |")
    txt = "\n".join(L) + "\n"
    for name in ("pilot_check.md", "pilot_check_clean.md"):
        with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"pilots: {res['n_pass']} of {res['n_models']} pass" + (f"; failed {res['failed']}" if res["failed"] else ""))
    return res


def score(allow_missing: bool = False) -> dict:
    """score_forced_direct.score_task on the constrained folders (module doc)."""
    import score_forced_direct as sfd
    saved = (sfd.REQ, sfd.FORCED_OUT, sfd.PROCESSED)
    sfd.REQ, sfd.FORCED_OUT, sfd.PROCESSED = REQ, OUT, PROCESSED
    try:
        return sfd.score_task("bbh", allow_missing)
    finally:
        sfd.REQ, sfd.FORCED_OUT, sfd.PROCESSED = saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    if a.pilots:
        check_pilots()
    else:
        score(a.allow_missing)


if __name__ == "__main__":
    main()
