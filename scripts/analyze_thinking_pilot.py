r"""The thinking-on pilot of the native-reasoning analysis: output lengths, closure of the
reasoning block, the cap of the runs by the cap rule below, and the projected cost.

PowerShell (from the repository root), after the folders data/d2_out/thinking/pilot/<key> are copied back:
    .\.venv\Scripts\python.exe scripts\analyze_thinking_pilot.py
    .\.venv\Scripts\python.exe scripts\analyze_thinking_pilot.py --allow-missing     # the models returned so far

No correctness is read: an answer counts as readable when the task's strict extraction finds one after the last
end-of-reasoning marker, and no gold answer is opened.

Exclusion: a checkpoint is excluded from the native-reasoning runs when more than half of its pilot
outputs (both tasks, 800 in all) reach the pilot's cap of 30,720 tokens. The rule reads output lengths of training
items only. Each returned checkpoint's status is final once its own pilot is back; manifests/thinking_exclusions.json
records it.

Cap rule: each task's cap is the smallest multiple of 1,024 at or above the largest 99th percentile of the pilot's
output lengths over the checkpoints not excluded (an output cut at the pilot's 30,720 counts at that length), and at
most 30,720. Its max_model_len is the cap plus the task's longest prompt with 100 tokens of template markup, rounded up
to a multiple of 1,024. The projection multiplies each model's request count by its mean output length under that cap
and divides by its pilot's output tokens per second of generation; its total covers the checkpoints not excluded.

Outputs: results/thinking_pilot/thinking_pilot.json (with "macros", prefix thp), summary.md, summary_clean.md and
tables/pilot_cap_share.csv; manifests/thinking_exclusions.json.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import generation as gen  # noqa: E402

PILOT = os.path.join(ROOT, "data", "d2_out", "thinking", "pilot")
RECORD = os.path.join(ROOT, "data", "requests", "thinking", "pilot", "build_record.json")
OUT = os.path.join(ROOT, "results", "thinking_pilot")
EXCLUSIONS = os.path.join(ROOT, "manifests", "thinking_exclusions.json")
PILOT_CAP = 30720
EXCLUDE_ABOVE = 0.5            # share of a checkpoint's pilot outputs at the cap above which it is excluded
END_MARKERS = ("</think>", "<channel|>")      # Qwen; Gemma-4
TASKS = ("gsm8k", "math")
# requests per model of the runs: the GSM8K collection's direct-answer requests; MATH-500 from the number of distinct
# configurations of the MATH-500 sampler (manifests/sampler_math.json) once it exists, else a projection
GSM8K_REQUESTS = 38251
MATH_ITEMS = 398
MATH_CONFIGS_PROJECTED = 32
LONGEST_PROMPT = {"math": 1174}                # longest MATH-500 prompt in tokens


def answer_text(text):
    """The text after the last end-of-reasoning marker, or None when no marker closes the reasoning."""
    cut = max((text.rfind(m) + len(m) if m in text else -1) for m in END_MARKERS)
    return text[cut:] if cut >= 0 else None


def readable(task, texts):
    """Whether the task's strict extraction finds an answer (no gold is read)."""
    if task == "math":
        from src import math_task
        return np.array([bool(math_task.compliance(t)) for t in texts])
    from src import d2_io, scoring
    spec, _ = d2_io.load_registered_scorer(os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
    return np.array([scoring.strict_extract(t, spec) is not None for t in texts])


def model_rows(key, d):
    out = gen.read_outputs(d)
    logs = [json.load(open(os.path.join(d, p), encoding="utf-8")) for p in sorted(os.listdir(d)) if p.startswith("run_log-")]
    gen_s = sum(float(x.get("generation_s") or 0) for x in logs)
    toks = sum(int((x.get("totals") or {}).get("output_tokens_generated") or 0) for x in logs)
    gpu = sorted({gen.gpu_type(x.get("gpu"))[0] for x in logs if x.get("gpu")})
    rows = {}
    for task in TASKS:
        g = out[out["arm"] == f"thinking_{task}"]
        n = g["n_output_tokens"].astype(int).to_numpy()
        cut = (g["finish_reason"] == "length").to_numpy()
        texts = g["output_text"].astype(str).tolist()
        ans = [answer_text(t) for t in texts]
        closed = np.array([a is not None for a in ans])
        body = [a if a is not None else ("" if c else t) for a, t, c in zip(ans, texts, cut)]
        rows[task] = {"n": int(len(g)), "mean": float(n.mean()), "median": float(np.median(n)),
                      "p90": float(np.percentile(n, 90)), "p99": float(np.percentile(n, 99)), "max": int(n.max()),
                      "cut_at_pilot_cap": int(cut.sum()), "closed": int(closed.sum()),
                      "unclosed_not_cut": int((~closed & ~cut).sum()), "readable_answer": int(readable(task, body).sum()),
                      "lengths": n.tolist()}
    return {"model_key": key, "gpu": gpu, "generation_s": gen_s, "output_tokens": toks,
            "output_tokens_per_s": toks / gen_s if gen_s else None, "tasks": rows,
            "complete": any(x.get("status") == "complete" for x in logs)}


def ceil_to(x, m):
    return int(math.ceil(x / m) * m)


def cap_share(m):
    """(outputs at the pilot's cap, outputs, share) over both tasks of one model's pilot."""
    cut = sum(m["tasks"][t]["cut_at_pilot_cap"] for t in TASKS)
    n = sum(m["tasks"][t]["n"] for t in TASKS)
    return cut, n, cut / n


def exclusion(M):
    """The exclusion rule over the returned models: {key: {"at_cap", "outputs", "share", "excluded"}}."""
    out = {}
    for m in M:
        cut, n, s = cap_share(m)
        out[m["model_key"]] = {"at_cap": cut, "outputs": n, "share": s, "excluded": s > EXCLUDE_ABOVE}
    return out


def task_caps(M, longest):
    """Each task's cap and max_model_len from the models given (the cap rule)."""
    caps = {}
    for task in TASKS:
        p99 = max(m["tasks"][task]["p99"] for m in M)
        cap = min(PILOT_CAP, ceil_to(p99, 1024))
        caps[task] = {"largest_p99": p99, "cap": cap, "max_model_len": ceil_to(cap + longest[task], 1024)}
    return caps


def gsm8k_longest_prompt():
    """The longest GSM8K collection prompt over the pilot's models: request text tokens plus 100 (as for MATH)."""
    from tokenizers import Tokenizer
    rec = json.load(open(RECORD, encoding="utf-8"))
    best = 0
    for k in rec["models"]:
        tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizers", k, "tokenizer.json"))
        for root in ("gsm8k", os.path.join("gsm8k_ext", f"collection_{k}")):
            p = os.path.join(ROOT, "data", "requests", root, f"{k}.parquet")
            if os.path.exists(p):
                rq = pd.read_parquet(p, columns=["arm", "messages_json"])
                rq = rq[rq["arm"] == "direct_answer"].drop_duplicates("messages_json")
                best = max(best, max(len(tok.encode("\n".join(m["content"] for m in json.loads(s)), add_special_tokens=False).ids)
                                     for s in rq["messages_json"]))
                break
    return best + 100


def math_requests():
    p = os.path.join(ROOT, "manifests", "sampler_math.json")
    if os.path.exists(p):
        s = json.load(open(p, encoding="utf-8"))
        n = len(set(map(str, s["discovery_config_ids"])) | set(map(str, s["validation_config_ids"])))
        return n * MATH_ITEMS, f"manifests/sampler_math.json ({n} distinct configurations)"
    return MATH_CONFIGS_PROJECTED * MATH_ITEMS, f"projected ({MATH_CONFIGS_PROJECTED} distinct configurations)"


def run(allow_missing):
    rec = json.load(open(RECORD, encoding="utf-8"))
    keys = sorted(rec["models"])
    present = [k for k in keys if os.path.isdir(os.path.join(PILOT, k))]
    if len(present) < len(keys) and not allow_missing:
        raise SystemExit(f"pilot folders missing for {sorted(set(keys) - set(present))}")
    M = [model_rows(k, os.path.join(PILOT, k)) for k in present]
    X = exclusion(M)
    kept = [m for m in M if not X[m["model_key"]]["excluded"]]
    longest = {"gsm8k": gsm8k_longest_prompt(), **LONGEST_PROMPT}
    nmath, nmath_src = math_requests()
    R = {"analysis": "thinking-on pilot (native reasoning)", "pilot_cap": PILOT_CAP,
         "models_present": present, "models_expected": keys, "complete": len(present) == len(keys),
         "requests_per_model": {"gsm8k": GSM8K_REQUESTS, "math": nmath}, "math_requests_source": nmath_src,
         "longest_prompt": longest,
         "exclusion": {"rule": f"excluded when more than {EXCLUDE_ABOVE:.0%} of the pilot outputs (both tasks) reach "
                               f"the pilot's cap of {PILOT_CAP:,} tokens (decided 4 November)",
                       "threshold": EXCLUDE_ABOVE, "excluded": sorted(k for k in X if X[k]["excluded"]),
                       "included": sorted(k for k in X if not X[k]["excluded"]),
                       "pending": sorted(set(keys) - set(present))},
         "tasks": task_caps(kept, longest), "models": []}
    total = excl_total = 0.0
    for m in M:
        row = {k: m[k] for k in ("model_key", "gpu", "generation_s", "output_tokens", "output_tokens_per_s", "complete")}
        row["pilot_cap_share"] = X[m["model_key"]]
        for task in TASKS:
            t = m["tasks"][task]
            cap = R["tasks"][task]["cap"]
            lens = np.minimum(np.asarray(t.pop("lengths")), cap)
            t["share_cut_at_pilot_cap"] = t["cut_at_pilot_cap"] / t["n"]
            t["share_over_cap"] = float(np.mean(np.asarray(lens) >= cap))
            t["mean_under_cap"] = float(lens.mean())
            n = R["requests_per_model"][task]
            t["projected_gpu_hours"] = (n * t["mean_under_cap"] / m["output_tokens_per_s"] / 3600
                                        if m["output_tokens_per_s"] else None)
            if X[m["model_key"]]["excluded"]:
                excl_total += t["projected_gpu_hours"] or 0
            else:
                total += t["projected_gpu_hours"] or 0
            row[task] = t
        R["models"].append(row)
    R["projected_gpu_hours_total"] = total
    R["projected_gpu_hours_excluded"] = excl_total
    R["projection_note"] = ("generation time only, at each model's pilot rate, over the checkpoints not excluded (a full "
                            "run batches more requests at once, so this may be an upper bound); model loading adds "
                            "about 0.1 GPU-hours per job")
    R["macros"] = macros(R)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "thinking_pilot.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    pd.DataFrame([{"model_key": r["model_key"],
                   **{f"{t}_at_cap_pct": 100 * r[t]["share_cut_at_pilot_cap"] for t in TASKS},
                   "both_at_cap_pct": 100 * r["pilot_cap_share"]["share"], "excluded": r["pilot_cap_share"]["excluded"]}
                  for r in R["models"]]).to_csv(os.path.join(OUT, "tables", "pilot_cap_share.csv"), index=False)
    write_exclusions(R)
    write_summary(R)
    print(json.dumps({t: R["tasks"][t] for t in TASKS}), f"excluded {R['exclusion']['excluded']};",
          f"projected {total:.1f} GPU-hours over {len(kept)} models")
    return R


def macros(R):
    """Ready-formatted values for paper/make_numbers.py, prefix thp."""
    ex, rows = R["exclusion"], R["models"]
    share = {r["model_key"]: 100 * r["pilot_cap_share"]["share"] for r in rows}
    m = {"thpModels": str(len(R["models_expected"])), "thpReturned": str(len(rows)),
         "thpThresholdPct": f"{100 * ex['threshold']:.0f}", "thpExcludedN": str(len(ex["excluded"])),
         "thpIncludedN": str(len(ex["included"])), "thpGsmCap": f"{R['tasks']['gsm8k']['cap']:,}".replace(",", "{,}"),
         "thpMathCap": f"{R['tasks']['math']['cap']:,}".replace(",", "{,}"),
         "thpGpuHours": f"{R['projected_gpu_hours_total']:,.0f}".replace(",", "{,}"),
         "thpGpuHoursExcluded": f"{R['projected_gpu_hours_excluded']:,.0f}".replace(",", "{,}")}
    if ex["included"]:
        m["thpCapSharePctMaxIncluded"] = f"{max(share[k] for k in ex['included']):.1f}"
    if ex["excluded"]:
        m["thpCapSharePctMinExcluded"] = f"{min(share[k] for k in ex['excluded']):.1f}"
    return m


def write_exclusions(R):
    """manifests/thinking_exclusions.json: each returned checkpoint's share at the cap and status under the rule."""
    ex = R["exclusion"]
    rec = {"rule": ex["rule"], "threshold": ex["threshold"], "decided": "the authors, 4 November 2026",
           "reads": "output lengths of the thinking pilot on training items; no correctness",
           "source": "results/thinking_pilot/thinking_pilot.json", "excluded": ex["excluded"],
           "included": ex["included"], "pending": ex["pending"],
           "models": {r["model_key"]: {**r["pilot_cap_share"],
                                       **{f"{t}_at_cap": r[t]["cut_at_pilot_cap"] for t in TASKS}}
                      for r in R["models"]}}
    with open(EXCLUSIONS, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")


def write_summary(R):
    L = ["# Thinking-on pilot (native reasoning)", "",
         "Each model ran 800 direct-answer screening requests (two configurations × 200 training "
         "items per task) with the thinking switch on, greedy, at most 30,720 new tokens. Lengths in tokens. No "
         "correctness is read; a readable answer is one the task's strict extraction finds after the last "
         "end-of-reasoning marker.", ""]
    if not R["complete"]:
        L += [f"**{len(R['models_present'])} of {len(R['models_expected'])} models returned so far.**", ""]
    ex = R["exclusion"]
    L += [f"Exclusion: a checkpoint is {ex['rule']}. Excluded: {', '.join(ex['excluded']) or 'none'}"
          + (f"; pilots still out: {', '.join(ex['pending'])}" if ex["pending"] else "") + ". The caps and the "
          "projection below use the checkpoints not excluded.", ""]
    for task in TASKS:
        T = R["tasks"][task]
        L += [f"{task}: largest per-model 99th percentile {T['largest_p99']:.0f}, so the cap is {T['cap']:,} and "
              f"max_model_len {T['max_model_len']:,} (longest prompt {R['longest_prompt'][task]:,}).", ""]
    L += [f"Requests per model of the runs: GSM8K {R['requests_per_model']['gsm8k']:,}; MATH-500 "
          f"{R['requests_per_model']['math']:,} ({R['math_requests_source']}). Projected generation: "
          f"{R['projected_gpu_hours_total']:.1f} GPU-hours ({R['projection_note']}); the excluded checkpoints would "
          f"have taken {R['projected_gpu_hours_excluded']:.1f}.", "",
          "| Model | Pilot outputs at the cap, both tasks (%) | Native reasoning |", "|---|---:|---|"]
    for m in R["models"]:
        s = m["pilot_cap_share"]
        L.append(f"| {m['model_key']} | {100 * s['share']:.1f} ({s['at_cap']} of {s['outputs']}) | "
                 f"{'excluded' if s['excluded'] else 'run'} |")
    L += ["", "| Model | Task | Mean | Median | p99 | Max | Cut at 30,720 | Closed | Readable | Over the cap (%) | GPU-hours |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in R["models"]:
        for task in TASKS:
            t = m[task]
            L.append(f"| {m['model_key']} | {task} | {t['mean']:.0f} | {t['median']:.0f} | {t['p99']:.0f} | {t['max']} | "
                     f"{t['cut_at_pilot_cap']} | {t['closed']} of {t['n']} | {t['readable_answer']} | "
                     f"{100 * t['share_over_cap']:.1f} | {t['projected_gpu_hours'] or 0:.1f} |")
    txt = "\n".join(L) + "\n"
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    run(a.allow_missing)


if __name__ == "__main__":
    main()
