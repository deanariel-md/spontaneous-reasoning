r"""Pilot check and scoring of the five-shot forced-direct arm (GSM8K, BIG-Bench Hard, MATH-500).

From the repository root:
    python scripts/score_fewshot.py --pilots          # after the 16-request GSM8K pilots
    python scripts/score_fewshot.py --task gsm8k      # after a task's runs
    python scripts/score_fewshot.py --task bbh
    python scripts/score_fewshot.py --pilots-math     # after the 16-request MATH-500 pilots
    python scripts/score_fewshot.py --task math

The forced arm's path (scripts/score_forced_direct.py), pointed at the five-shot folders: requests in
data/requests/fewshot/, outputs copied back to data/d2_out/fewshot/{pilot,gsm8k,bbh}/<key>, scored tables in
data/processed/fewshot/<task>_scored.parquet with box closure in <task>_box.parquet, and the pilot verdicts in
results/fewshot/pilot_check.json and pilot_check.md. The pilot rule is the forced arm's: a complete --limit 16 run, the
prefill of config/decoding_forced.yaml in its log, prompt_render_mismatches 0. No correctness is read by the pilots.

MATH-500: outputs in data/d2_out/fewshot/{pilot_math,math}/<key> (the forced MATH-500 arm's own folders sit elsewhere,
data/d2_out/math/forced/, so the folder lookup is pointed here too). Its pilots run 16 MATH-500 requests each with
config/decoding_forced_math.yaml; a pilot passes on a complete --limit 16 run whose logged decoding has that file's
prefill and no stop string, with prompt_render_mismatches 0. Verdicts in results/fewshot/pilot_check_math.json and .md
(with a clean copy). The scoring is the forced MATH-500 arm's (math scorer, box closure of
score_forced_direct.box_closed with task "math").
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REQ = os.path.join(ROOT, "data", "requests", "fewshot")
OUT = os.path.join(ROOT, "data", "d2_out", "fewshot")
PROCESSED = os.path.join(ROOT, "data", "processed", "fewshot")
RESULTS = os.path.join(ROOT, "results", "fewshot")
NAMES = ("REQ", "FORCED_OUT", "PROCESSED", "RESULTS")
DECODING_MATH = os.path.join(ROOT, "config", "decoding_forced_math.yaml")
PILOT_MATH = "pilot_math"
TASKS = ("gsm8k", "bbh", "math")


def fewshot_dir(task, key):
    """A model's five-shot folder for every task: data/d2_out/fewshot/<task>/<key>."""
    return os.path.join(OUT, task, key)


class pointed:
    """Context manager: score_forced_direct's folders replaced by the five-shot ones, restored on exit."""

    def __enter__(self):
        import score_forced_direct as sfd
        self.sfd = sfd
        self.saved = {n: getattr(sfd, n) for n in NAMES}
        self.saved_dir = sfd.forced_dir
        for n, v in zip(NAMES, (REQ, OUT, PROCESSED, RESULTS)):
            setattr(sfd, n, v)
        sfd.forced_dir = fewshot_dir
        return sfd

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            setattr(self.sfd, n, v)
        self.sfd.forced_dir = self.saved_dir
        return False


def pilot_verdict(logs, want_prefill, want_stop=()):
    """(passes, status, the complete 16-request runs) of one model's pilot run logs."""
    done = [x for x in logs if x.get("status") == "complete" and (x.get("inputs") or {}).get("limit") == 16]
    if not done:
        return False, "no complete 16-request run", done
    dec = [x.get("decoding") or {} for x in done]
    if any(dict(d.get("prefill") or {}) != dict(want_prefill) for d in dec):
        return False, "prefill differs from config/decoding_forced_math.yaml", done
    if any(list(d.get("stop") or []) != list(want_stop) for d in dec):
        return False, "stop strings differ from config/decoding_forced_math.yaml", done
    if any(x.get("prompt_render_mismatches") != 0 for x in done):
        return False, "rendered prompts differ between runner and engine", done
    return True, "passes", done


def check_pilots_math():
    """The MATH-500 pilots (module doc). Readability with the math scorer; no correctness is read."""
    import yaml
    import score_forced_direct as sfd
    from src import d2_io, generation as gen, math_task as mt
    dec = yaml.safe_load(open(DECODING_MATH, encoding="utf-8"))
    rec = json.load(open(os.path.join(REQ, "build_record.json"), encoding="utf-8"))
    keys = sorted(rec["tasks"]["math"])
    items = mt.load_target()
    items = items[items["gold_answer"].notna()]
    rows = []
    for k in keys:
        d = os.path.join(OUT, PILOT_MATH, k)
        row = {"model_key": k, "folder": d2_io.rel_path(d)}
        names = sorted(p for p in os.listdir(d) if p.startswith("run_log-")) if os.path.isdir(d) else []
        if not names:
            rows.append({**row, "status": "not returned yet", "passes": False, "pending": True})
            continue
        L = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in names]
        passes, status, done = pilot_verdict(L, dec["prefill"], dec.get("stop") or [])
        row.update({"status": status, "passes": passes,
                    "runs": [{"status": x.get("status"), "limit": (x.get("inputs") or {}).get("limit"),
                              "prompt_render_mismatches": x.get("prompt_render_mismatches"),
                              "gpu": list(gen.gpu_type(x.get("gpu")))} for x in L]})
        if done:
            out, _ = d2_io.inspect_work_dir(d, allow_incomplete=True, string_item_ids=True)
            sc = mt.score_outputs(out, items)
            closed = sfd.box_closed(out["output_text"].astype(str).tolist(), "math")
            row.update({"n_outputs": int(len(out)), "box_closed": int(closed.sum()),
                        "readable": int(sc["compliant"].sum()),
                        "closed_and_readable": int((closed & sc["compliant"].to_numpy(bool)).sum()),
                        "finish_length": int((out["finish_reason"] == "length").sum()),
                        "examples": out["output_text"].astype(str).head(3).tolist()})
        rows.append(row)
    res = {"rule": "passes: a complete --limit 16 run of the MATH-500 five-shot requests, the prefill and the (empty) stop "
                   "list of config/decoding_forced_math.yaml in its log, and prompt_render_mismatches 0. No correctness "
                   "is read.", "prefill": dec["prefill"], "scorer": mt.SCORER_VERSION, "models": rows,
           "n_pass": sum(r["passes"] for r in rows), "n_models": len(rows),
           "pending": [r["model_key"] for r in rows if r.get("pending")],
           "failed": [r["model_key"] for r in rows if not r["passes"] and not r.get("pending")]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "pilot_check_math.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Five-shot MATH-500 pilots", "", f"Rule: {res['rule']}", "",
         f"{res['n_pass']} of {res['n_models']} pilots pass." + (f" Failed: {', '.join(res['failed'])}." if res["failed"]
                                                                  else "")
         + (f" Not returned yet: {len(res['pending'])}." if res["pending"] else ""), "",
         "| Model | Verdict | Outputs | Box closed | Readable answer | Closed and readable | Cut at 16 tokens |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model_key']} | {r['status']} | {r.get('n_outputs', '--')} | {r.get('box_closed', '--')} | "
                 f"{r.get('readable', '--')} | {r.get('closed_and_readable', '--')} | {r.get('finish_length', '--')} |")
    txt = "\n".join(L) + "\n"
    for name in ("pilot_check_math.md", "pilot_check_math_clean.md"):
        with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"MATH-500 pilots: {res['n_pass']} of {res['n_models']} pass" + (f"; failed {res['failed']}" if res["failed"] else "")
          + (f"; not returned yet {len(res['pending'])}" if res["pending"] else ""))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--pilots-math", action="store_true")
    g.add_argument("--task", choices=TASKS)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    if a.pilots_math:
        check_pilots_math()
        return
    with pointed() as sfd:
        if a.pilots:
            sfd.check_pilots()
        else:
            sfd.score_task(a.task, a.allow_missing)


if __name__ == "__main__":
    main()
