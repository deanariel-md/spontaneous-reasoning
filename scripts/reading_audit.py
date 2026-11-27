r"""Reading audit of requested-direct replies (budget under $20). Is a long requested answer written-out reasoning? One LLM judge labels each sampled reply
R (intermediate reasoning), P (no reasoning but extra text) or A (answer only) from the prompt and the reply alone.

The judge: the model of JUDGE below, through the provider's Message Batches API, thinking disabled with
effort medium, max_tokens 256, at the model's default sampling: no temperature, top_p or top_k is sent (the model
refuses each with a 400). The key is ANTHROPIC_API_KEY, read inside Python only.

Subcommands (repository root):
    python scripts/reading_audit.py sample
    python scripts/reading_audit.py population
    python scripts/reading_audit.py human
    python scripts/reading_audit.py judge --backend anthropic --dry-run --tiktoken-path <dir> --tiktoken-cache <dir>
    python scripts/reading_audit.py judge --backend anthropic --count-tokens
    python scripts/reading_audit.py judge --backend anthropic --pilot --approved --limit-usd 20
    python scripts/reading_audit.py judge --backend anthropic --submit --approved --limit-usd 20
    python scripts/reading_audit.py judge --backend anthropic --collect
    python scripts/reading_audit.py analyse

sample   The panel's requested-direct replies (arm direct_answer) on the discovery draws of GSM8K-29, MATH-500, BIG-Bench
         Hard-28 and HotpotQA, stratified by benchmark x length bin x model. The length is n_output_tokens of the stored
         output, the count the panel's w uses (src/baseline_mixture.py via scripts/run_baseline_mixture.task_rows; for
         HotpotQA data/processed/hotpot/scored_content.parquet, the same counts as scored_discovery.parquet). Bins:
         1-16, 17-64, over 64 tokens. Population: every (model, configuration, item) of the distinct discovery
         configurations; a configuration drawn twice counts twice in every estimate (row weight = its number of draws),
         as in w. Allocation: 7 per model x bin (21 per model), a bin with fewer rows gives all it has and its shortfall
         goes round-robin to the model's other bins. Plus Qwen4-Max's requested replies (data/frontier_panel/closed_w/
         run_raw.jsonl; every template once, 100 items) on GSM8K and MATH-500, 55 per benchmark: 15, 20, 20 in the three
         bins; length by the API's count as scripts/closed_w.py's w (completion minus reasoning tokens). Draws without
         replacement inside each stratum, from numpy's default_rng([SEED, crc32(stratum)]). Writes sample.jsonl (the
         judge's input and the design weights), sample_key.jsonl (gold answers and scores, never sent), strata.csv,
         population.json (per model w, w16, w64, requested - forced and CoT - forced accuracy) and
         sample_manifest.json.
population  Recomputes population.json's per-model block from the scored tables (adds CoT - forced to a file written
         before it existed); the earlier fields must agree to 1e-9 and the sample is not redrawn.
human    Drawn before judging, never by the judge's class: 150 rows of the panel part of the sample, stratified by
         benchmark x length bin (12 cells, 12 or 13 each; the cells with 13 chosen by a seeded draw), seeded. Two
         blind sheets with the same rows in two seeded orders (human_rater1.csv, human_rater2.csv; ids H001..H150 in
         each sheet's own order; columns hid, benchmark, prompt, reply, label, unsure, note; UTF-8 with BOM), .xlsx
         copies if openpyxl is installed, human_key.csv (each sheet's hid -> sample_id, cell, design weight) and
         human_instructions.md (the rubric for the raters). A sheet with any label filled is never overwritten.
judge    --backend anthropic (the only backend). System = the rubric of results/reading_audit/rubric.md (verbatim),
         user message = its template filled with the prompt and the reply; no prefill, no sampling parameter.
         Every sample row once (custom_id m_<task>_<hash>) plus a repeat set of 150 panel rows (the human draw's rule,
         its own seeded stream; custom_id r_<task>_<hash>) for the judge's agreement with itself; the map is
         judge_custom_ids.csv. Append-only judge_raw.jsonl; every result line records the model, configuration hash,
         rubric and template sha256 and the batch id.
         --dry-run      sends nothing: request count, input by o200k and bytes, expected / high / worst cost at the
                        batch prices (judge_dryrun.json).
         --count-tokens free (POST /v1/messages/count_tokens): the exact input count of every request
                        (judge_token_counts.json), which makes the worst case exact.
         --pilot        20 rows (5 per benchmark) through the ordinary Messages API at standard prices; checks: HTTP
                        200, every reply read as one letter, usage present, no thinking block, the model echoed as
                        the judge's id (or a dated snapshot of it), no refusal (judge_pilot.json; exit 3 on a
                        failed check). The pilot's labels count as labels.
         --submit       needs a passed pilot under the same configuration; one batch with every row not yet labelled
                        and not in a batch still running; refuses (exit 4) if spent so far + the batch's worst case
                        exceeds --limit-usd, unless --split, which sends the part that fits (the rest with the next
                        --submit after --collect). Writes the batch id to judge_batch.json. Resumable.
         --collect      polls every --poll-seconds until each submitted batch has ended (--once: one look), downloads
                        its results, appends them, reports counts, cost from usage at the batch prices and unread
                        replies. Errored, canceled and expired requests are not billed and go again with --submit.
analyse  The check of the mixture assumption from the key file's scores (requested accuracy on
         replies labelled R and not R beside CoT and forced accuracy on the same items; across models requested -
         forced against r x (CoT - forced) and w x (CoT - forced)); judge vs each rater and rater vs rater (agreement,
         Cohen's kappa, 3x3 confusion, R vs not R), design-weighted and unweighted, with bootstrap intervals of each
         kappa and of the gaps kappa(judge, rater k) - kappa(rater 1, rater 2) (rows resampled within the check's
         benchmark x length cells; descriptive, no pass rule); the judge's self-agreement.
         Writes reading_audit.json (block "macros", prefix lab), summary.md, summary_clean.md, per_model.csv,
         paper/tables/reading_audit.tex. Shares are design-weighted (stratified ratio estimates); intervals: 2,000
         bootstrap draws within strata, 95% percentile.

The key is read from the environment variable ANTHROPIC_API_KEY (scripts/closed_w.read_key) and never
printed, logged or written; every saved line and error text passes through redact().
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "results" / "reading_audit"
SAMPLE = OUT / "sample.jsonl"
KEYFILE = OUT / "sample_key.jsonl"
STRATA = OUT / "strata.csv"
POPULATION = OUT / "population.json"
MANIFEST = OUT / "sample_manifest.json"
RUBRIC = OUT / "rubric.md"
JUDGE_RAW = OUT / "judge_raw.jsonl"
PILOT = OUT / "judge_pilot.json"
DRYRUN = OUT / "judge_dryrun.json"
BATCHES = OUT / "judge_batch.json"
COUNTS = OUT / "judge_token_counts.json"
IDMAP = OUT / "judge_custom_ids.csv"
REPEAT_SET = OUT / "judge_repeat_set.csv"
HUMAN_SHEETS = {1: OUT / "human_rater1.csv", 2: OUT / "human_rater2.csv"}
HUMAN_XLSX = {1: OUT / "human_rater1.xlsx", 2: OUT / "human_rater2.xlsx"}
HUMAN_KEY = OUT / "human_key.csv"
HUMAN_INSTRUCTIONS = OUT / "human_instructions.md"
RESULT = OUT / "reading_audit.json"
SUMMARY = OUT / "summary.md"
CLEAN = OUT / "summary_clean.md"
TABLE = ROOT / "paper" / "tables" / "reading_audit.tex"
QWEN_RAW = ROOT / "data" / "frontier_panel" / "closed_w" / "run_raw.jsonl"

TASKS = ("gsm8k", "math", "bbh", "hotpot")
TASK_NAME = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
TASK_TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh", "hotpot": "Hq"}
BINS = ("1-16", "17-64", "65+")
CAP = {"gsm8k": 16, "math": 16, "bbh": 16, "hotpot": 64}   # the forced arm's cap: the panel's w on each benchmark
SEED = 20261125
PER_BIN = 7
QWEN = "qwen4-max"
QWEN_TASKS = ("gsm8k", "math")
QWEN_ALLOC = {"1-16": 15, "17-64": 20, "65+": 20}
CLASSES = ("R", "P", "A")
HUMAN_N = 150
REPEAT_N = 150
N_BOOT = 2000
SCORINGS = ("strict", "content")
ARMS3 = ("requested", "cot", "forced")

# The judge model on the provider's Message Batches API. Prices in USD per token from the provider's pricing pages:
# standard $5 / $25 per million input / output tokens, batch half of that.
# Prompt caching is not requested (no cache_control); cache tokens, if any appear, are costed at 1.25x (write) and 0.1x
# (read) of the input price.
JUDGE = {
    "key": "claude-opus-5", "label": "Claude Opus 5", "id": "claude-opus-5", "provider": "Anthropic",
    "price_batch": {"input": 2.5e-06, "output": 1.25e-05},
    "price_standard": {"input": 5e-06, "output": 2.5e-05},
    "cache_write_factor": 1.25, "cache_read_factor": 0.1,
}
MODEL_ECHO = re.compile(r"claude-opus-5(?:-\d{8})?")   # the id or a dated snapshot of it
API_BASE = "https://api.anthropic.com"
# The API version used; on it a Messages or batch request that sends temperature, top_p or top_k is refused, which
# this script never sends.
ANTHROPIC_VERSION = "2026-10-21"
MAX_TOKENS = 256   # room for a reply that reasons aloud before its label
THINKING = {"type": "disabled"}             # accepted by the judge model at effort high or below
OUTPUT_CONFIG = {"effort": "medium"}
SAMPLING = ("the model's default sampling: no temperature, top_p or top_k is sent (the judge model refuses each of them "
            "with a 400)")
PROMPT_OVERHEAD = 64          # tokens per call beyond the message bytes, for the byte-based worst case
COUNT_MARGIN = 1.02           # exact counts (count_tokens) x 1.02 + COUNT_PAD tokens, for the counted worst case
COUNT_PAD = 16
EXPECTED_O200K = 1.25         # the judge's tokenizer against o200k (tiktoken undercounts it by ~15-20%)
HIGH_O200K = 1.6
EXPECTED_OUT = 2              # output tokens per call in the expected case (one letter and the end)
PILOT_PER_TASK = 5
THREADS = 8
MAX_TRIES = 5
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


# ================================================================================================ small helpers


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dumps(o) -> str:
    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    return json.dumps(o, ensure_ascii=False, default=default)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(json.loads(dumps(obj)), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def clean_md(md: str) -> str:
    """The summary without citation markers (none are written; kept for the rule of a clean copy)."""
    return re.sub(r"\[\^?\d+\]", "", md)


# ================================================================================================ the design


def bin_of(n) -> str:
    n = int(n)
    return BINS[0] if n <= 16 else (BINS[1] if n <= 64 else BINS[2])


def allocate(sizes: dict, targets: dict, order=BINS) -> dict:
    """Per bin min(target, size); the total shortfall goes one at a time, round-robin in `order`, to bins with rows
    left. Equal targets give an equal allocation with redistribution."""
    alloc = {b: min(int(targets.get(b, 0)), int(sizes.get(b, 0))) for b in order}
    left = sum(int(targets.get(b, 0)) for b in order) - sum(alloc.values())
    while left > 0:
        moved = False
        for b in order:
            if left == 0:
                break
            if alloc[b] < int(sizes.get(b, 0)):
                alloc[b] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return alloc


def stratum_rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])


def draw_rows(df: pd.DataFrame, n: int, seed: int, key: str) -> pd.DataFrame:
    """n rows without replacement, the frame first sorted by (config_id, item_id) so the draw is order-free."""
    df = df.sort_values(["config_id", "item_id"], kind="mergesort").reset_index(drop=True)
    if n >= len(df):
        return df
    idx = np.sort(stratum_rng(seed, key).choice(len(df), size=int(n), replace=False))
    return df.iloc[idx].reset_index(drop=True)


def stratify(pop: pd.DataFrame, targets_of, seed: int, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pop: rows with task, model_key, config_id, item_id, n_tokens, mult. targets_of(task, model) -> {bin: n}.
    Returns (sampled rows with the design columns, strata table)."""
    pop = pop.assign(bin=pop["n_tokens"].map(bin_of))
    rows, strata = [], []
    for (task, model), g in pop.groupby(["task", "model_key"], sort=True):
        sizes = g["bin"].value_counts().to_dict()
        alloc = allocate(sizes, targets_of(task, model))
        for b in BINS:
            gb = g[g["bin"] == b]
            key = f"{source}|{task}|{model}|{b}"
            N, M = len(gb), float(gb["mult"].sum())
            n = alloc[b]
            strata.append({"stratum": key, "source": source, "task": task, "model_key": model, "bin": b, "N": N,
                           "M": M, "n": n, "p_incl": (n / N) if N else None})
            if n == 0:
                continue
            s = draw_rows(gb, n, seed, key).assign(stratum=key, N_h=N, M_h=M, n_h=n, p_incl=n / N, source=source)
            rows.append(s)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(strata)


# ================================================================================================ populations


def draw_counts(disc) -> dict:
    out = {}
    for c in disc:
        out[str(c)] = out.get(str(c), 0) + 1
    return out


def panel_wide(task):
    """(wide, panel, discovery draws): one row per (model, discovery configuration, item) with the requested output's
    n_output_tokens and the strict and content scores of the requested, forced and chain-of-thought arms."""
    if task == "hotpot":
        return hotpot_wide()
    import run_baseline_mixture as rbm
    wide, panel, disc, _ = rbm.task_rows(task)
    wide = wide[wide["config_id"].isin(set(disc))].copy()
    return wide, panel, disc


def hotpot_wide():
    import run_hotpot as rh
    s = json.load(open(rh.SAMPLER, encoding="utf-8"))
    disc = [str(u) for u in s["discovery_config_ids"]]
    panel, _, _ = rh.panel_info()
    items = set(map(str, rh.item_ids()))
    path = ROOT / "data" / "processed" / "hotpot" / "scored_content.parquet"
    df = pd.read_parquet(path, columns=["model_key", "arm", "config_id", "item_id", "strict_correct", "content_correct",
                                        "n_output_tokens"],
                         filters=[("config_id", "in", sorted(set(disc))),
                                  ("arm", "in", ["direct_answer", "forced_direct", "zero_shot_cot"])])
    for c in ("model_key", "config_id", "item_id"):
        df[c] = df[c].astype(str)
    df = df[df["model_key"].isin(set(panel)) & df["item_id"].isin(items)]
    key = ["model_key", "config_id", "item_id"]
    wide = None
    for a, name in (("direct_answer", "requested"), ("forced_direct", "forced"), ("zero_shot_cot", "cot")):
        x = df[df["arm"] == a][key + ["strict_correct", "content_correct"] +
                               (["n_output_tokens"] if a == "direct_answer" else [])]
        x = x.rename(columns={"strict_correct": f"{name}_strict", "content_correct": f"{name}_content"})
        wide = x if wide is None else wide.merge(x, on=key, how="inner", validate="one_to_one")
    n_expect = len(panel) * len(set(disc)) * len(items)
    if len(wide) != n_expect:
        raise SystemExit(f"{rel(path)}: {len(wide)} aligned rows, expected {n_expect}")
    if wide["n_output_tokens"].isna().any():
        raise SystemExit(f"{rel(path)}: a requested output has no token count")
    return wide.reset_index(drop=True), panel, disc


def model_summaries(wide: pd.DataFrame, panel, disc, cap: int) -> dict:
    """Per model on the discovery draws (a configuration drawn twice counts twice): w at the cap, at 16 and at 64,
    requested - forced and CoT - forced accuracy (pp, strict and content)."""
    from src import baseline_mixture as bm
    cfg = sorted(set(disc))
    mi = pd.Index(panel).get_indexer(wide["model_key"])
    ci = pd.Index(cfg).get_indexer(wide["config_id"])
    di = pd.Index(cfg).get_indexer(disc)
    n = wide["n_output_tokens"].to_numpy()
    out = {"w": bm.draw_weighted_share(n > cap, mi, ci, len(panel), len(cfg), di),
           "w16": bm.draw_weighted_share(n > 16, mi, ci, len(panel), len(cfg), di),
           "w64": bm.draw_weighted_share(n > 64, mi, ci, len(panel), len(cfg), di)}
    for sc in ("strict", "content"):
        d = (wide[f"requested_{sc}"].astype(float) - wide[f"forced_{sc}"].astype(float)).to_numpy()
        out[f"req_minus_forced_{sc}"] = 100 * _draw_mean(d, mi, ci, len(panel), len(cfg), di)
        if f"cot_{sc}" in wide.columns:
            d = (wide[f"cot_{sc}"].astype(float) - wide[f"forced_{sc}"].astype(float)).to_numpy()
            out[f"cot_minus_forced_{sc}"] = 100 * _draw_mean(d, mi, ci, len(panel), len(cfg), di)
    return {k: dict(zip(panel, map(float, v))) for k, v in out.items()}


def _draw_mean(v, mi, ci, M, C, di) -> np.ndarray:
    """(M,) mean of v within each configuration, averaged over the draws (repeats included)."""
    s = np.zeros((M, C))
    k = np.zeros((M, C))
    np.add.at(s, (mi, ci), v)
    np.add.at(k, (mi, ci), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / k
    return m[:, list(di)].mean(1)


def saved_checks(task, summ: dict) -> dict:
    """The recomputed w and requested - forced against the values saved by run_baseline_mixture.py (GSM8K, MATH-500,
    BIG-Bench Hard) and the HotpotQA mechanism forecast."""
    out = {}
    if task == "hotpot":
        p = ROOT / "manifests" / "predictions_hotpot_mechanism.json"
        rec = json.load(open(p, encoding="utf-8"))
        diffs = [abs(summ["w"][m] - rec["per_model"][m]["w"]) for m in rec["per_model"]]
        out["w_vs_hotpot_forecast_max_abs"] = max(diffs)
        if "cot_minus_forced_strict" in summ:
            out["cot_minus_forced_strict_vs_hotpot_forecast_max_abs_pp"] = max(
                abs(summ["cot_minus_forced_strict"][m] - rec["per_model"][m]["gain_vs_forced"]) for m in rec["per_model"])
        out["source"] = rel(p)
        return out
    R = json.load(open(ROOT / "results" / "baseline_mixture" / "baseline_mixture.json", encoding="utf-8"))
    pm = {r["model_key"]: r for r in R["tasks"][task]["per_model"]}
    out["w_vs_baseline_mixture_max_abs"] = max(abs(summ["w"][m] - pm[m]["w_discovery"]) for m in pm)
    for sc in ("strict", "content"):
        out[f"req_minus_forced_{sc}_vs_baseline_mixture_max_abs_pp"] = max(
            abs(summ[f"req_minus_forced_{sc}"][m] - (pm[m][f"{sc}_gain_vs_forced"] - pm[m][f"{sc}_gain_vs_requested"]))
            for m in pm)
        if f"cot_minus_forced_{sc}" in summ:
            out[f"cot_minus_forced_{sc}_vs_baseline_mixture_max_abs_pp"] = max(
                abs(summ[f"cot_minus_forced_{sc}"][m] - pm[m][f"{sc}_gain_vs_forced"]) for m in pm)
    out["source"] = "results/baseline_mixture/baseline_mixture.json"
    return out


def qwen_population() -> pd.DataFrame:
    """Qwen4-Max's answered requested replies on GSM8K and MATH-500 (every template once); n_tokens = the API's count
    (completion minus reasoning tokens, scripts/closed_w.api_count)."""
    import closed_w as cw
    lines = cw.read_raw(QWEN_RAW)
    fin = cw.final_by_key(lines)
    rows = []
    for k, x in fin.items():
        if x.get("model_key") != QWEN or x.get("arm") != "requested" or x.get("task") not in QWEN_TASKS:
            continue
        if x.get("outcome") != "reply":
            continue
        n = cw.api_count(x)
        if n is None:
            raise SystemExit(f"{k}: no token count")
        rows.append({"task": x["task"], "model_key": QWEN, "config_id": str(x["config_id"]),
                     "item_id": str(x["item_id"]), "request_id": x["request_id"], "n_tokens": int(n), "mult": 1,
                     "reply": x.get("text") or "", "finish_reason": x.get("finish_reason"),
                     "text_sha256": _sent_sha(lines, k)})
    return pd.DataFrame(rows)


def _sent_sha(lines, key):
    for x in lines:
        if x.get("event") == "sent" and x.get("key") == key:
            return x.get("text_sha256")
    return None


# ================================================================================================ texts and prompts


def output_dirs(task: str, model: str) -> list[Path]:
    d = ROOT / "data" / "d2_out"
    if task == "gsm8k":
        p = d / "collection" / model
        return [p if p.is_dir() else d / "collection_ext" / model]
    if task == "math":
        return [d / "math" / "collection" / model]
    if task == "bbh":
        return [d / "bbh" / "collection" / model]
    return [d / "hotpot" / "block_a" / model, d / "hotpot" / "block_b" / model]


def read_outputs(task: str, model: str, configs) -> pd.DataFrame:
    """The model's requested-direct outputs on the given configurations, from its shards (top level of the output
    folder only; a throughput subfolder is never read). Duplicates must agree."""
    import pyarrow.dataset as ds
    files = [str(f) for d in output_dirs(task, model) for f in sorted(d.glob("shard-*.parquet"))]
    if not files:
        raise SystemExit(f"{task} {model}: no shards in {[rel(d) for d in output_dirs(task, model)]}")
    parts = []
    for f in files:   # one file at a time: item_id is int64 in some folders and a string in others
        t = ds.dataset(f, format="parquet").to_table(
            columns=["request_id", "model_key", "arm", "config_id", "item_id", "output_text", "n_output_tokens",
                     "finish_reason"],
            filter=(ds.field("arm") == "direct_answer") & ds.field("config_id").isin(sorted(set(configs))))
        if t.num_rows:
            parts.append(t.to_pandas())
    df = pd.concat(parts, ignore_index=True)
    for c in ("config_id", "item_id", "model_key"):
        df[c] = df[c].astype(str)
    if (df["model_key"] != model).any():
        raise SystemExit(f"{task} {model}: shards hold another model")
    dup = df.duplicated(["config_id", "item_id"], keep=False)
    if dup.any():
        g = df[dup].groupby(["config_id", "item_id"])
        if (g["output_text"].nunique() > 1).any() or (g["n_output_tokens"].nunique() > 1).any():
            raise SystemExit(f"{task} {model}: duplicated outputs that differ")
        df = df.drop_duplicates(["config_id", "item_id"])
    return df


def request_files(task: str, model: str) -> list[Path]:
    r = ROOT / "data" / "requests"
    c = {"gsm8k": [r / "gsm8k" / f"{model}.parquet", r / "gsm8k_ext" / f"collection_{model}" / f"{model}.parquet"],
         "math": [r / "math" / f"{model}.parquet"], "bbh": [r / "bbh" / f"{model}.parquet"],
         "hotpot": [r / "hotpot" / f"{model}.parquet"]}[task]
    return [p for p in c if p.exists()]


def read_prompts(task: str, model: str, request_ids) -> dict:
    """{request_id: the user message} from the model's own request file; every row must be one user message."""
    files = request_files(task, model)
    if not files:
        raise SystemExit(f"{task} {model}: no request file")
    df = pd.read_parquet(files[0], columns=["request_id", "arm", "messages_json"],
                         filters=[("request_id", "in", sorted(set(request_ids))), ("arm", "==", "direct_answer")])
    out = {}
    for rid, mj in zip(df["request_id"], df["messages_json"]):
        m = json.loads(mj)
        if not (isinstance(m, list) and len(m) == 1 and m[0].get("role") == "user"
                and isinstance(m[0].get("content"), str)):
            raise SystemExit(f"{rel(files[0])}: {rid} is not exactly one user message")
        out[str(rid)] = m[0]["content"]
    missing = set(map(str, request_ids)) - set(out)
    if missing:
        raise SystemExit(f"{rel(files[0])}: {len(missing)} sampled request ids absent, e.g. {sorted(missing)[:2]}")
    return out


def gold_answers(task: str) -> dict:
    try:
        if task == "gsm8k":
            from src import gsm8k
            it = gsm8k.load_gsm8k("test")
        elif task == "math":
            from src import math_d2
            it, _ = math_d2.load_items("target")
        elif task == "bbh":
            from src import bbh_d2
            it, _ = bbh_d2.load_items("target")
        else:
            from src import hotpot_d2
            it, _ = hotpot_d2.load_items("target")
        return {str(i): str(g) for i, g in zip(it["item_id"], it["gold_answer"])}
    except Exception as e:  # noqa: BLE001  (the key file is for later reading; a missing gold is recorded, not fatal)
        print(f"{task}: gold answers not loaded ({type(e).__name__}: {e})", flush=True)
        return {}


# ================================================================================================ sample


def cmd_sample(args) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, keys, strata, population, checks, sources = [], [], [], {}, {}, {}
    for task in TASKS:
        wide, panel, disc = panel_wide(task)
        mult = draw_counts(disc)
        summ = model_summaries(wide, panel, disc, CAP[task])
        checks[task] = saved_checks(task, summ)
        population[task] = {"models": list(panel), "discovery_draws": list(disc), "cap": CAP[task],
                            "rows_distinct": int(len(wide)), "per_model": summ}
        pop = wide.rename(columns={"n_output_tokens": "n_tokens"}).assign(task=task)
        pop["mult"] = pop["config_id"].map(mult).astype(int)
        s, st = stratify(pop, lambda t, m: {b: PER_BIN for b in BINS}, args.seed, "panel")
        strata.append(st)
        gold = gold_answers(task)
        for model, g in s.groupby("model_key", sort=True):
            outs = read_outputs(task, model, g["config_id"].unique())
            j = g.merge(outs, on=["model_key", "config_id", "item_id"], how="left", validate="one_to_one")
            if j["output_text"].isna().any():
                raise SystemExit(f"{task} {model}: {int(j['output_text'].isna().sum())} sampled rows have no output")
            bad = (j["n_output_tokens"].astype(int) != j["n_tokens"].astype(int))
            if bad.any():
                raise SystemExit(f"{task} {model}: {int(bad.sum())} token counts differ between the shards and the "
                                 "scored table")
            prompts = read_prompts(task, model, j["request_id"])
            for r in j.itertuples(index=False):
                sid = f"panel|{task}|{model}|{r.request_id}"
                rows.append({"sample_id": sid, "source": "panel", "task": task, "model_key": model,
                             "config_id": r.config_id, "item_id": r.item_id, "request_id": r.request_id,
                             "n_tokens": int(r.n_tokens), "bin": r.bin, "stratum": r.stratum, "N_h": int(r.N_h),
                             "M_h": float(r.M_h), "n_h": int(r.n_h), "p_incl": float(r.p_incl), "mult": int(r.mult),
                             "finish_reason": r.finish_reason, "question": prompts[str(r.request_id)],
                             "reply": r.output_text})
                keys.append({"sample_id": sid, "gold": gold.get(str(r.item_id)),
                             **{f"{a}_{sc}": int(getattr(r, f"{a}_{sc}")) for a in ("requested", "forced", "cot")
                                for sc in ("strict", "content")}})
        print(f"{task}: {len(s)} sampled of {len(wide)} rows, {len(panel)} models ({time.time() - t0:.0f} s)",
              flush=True)

    # Qwen4-Max (the commercial model), GSM8K and MATH-500
    qp = qwen_population()
    qs, qst = stratify(qp, lambda t, m: QWEN_ALLOC, args.seed, "qwen")
    strata.append(qst)
    import closed_w as cw
    cfg = cw.load_config()
    population["qwen"] = {}
    for task in QWEN_TASKS:
        gq = qp[qp["task"] == task]
        population["qwen"][task] = {"replies": int(len(gq)), "w16": float((gq["n_tokens"] > 16).mean()),
                                    "w64": float((gq["n_tokens"] > 64).mean()),
                                    "token_count": "API count (completion minus reasoning tokens)"}
        sub = qs[qs["task"] == task]
        reqs = cw.load_requests(cw.request_path(cfg, task), request_ids=list(sub["request_id"]))
        text = dict(zip(reqs["request_id"], reqs["text"]))
        gold = gold_answers(task)
        for r in sub.itertuples(index=False):
            q = text[r.request_id]
            if r.text_sha256 and sha256_text(q) != r.text_sha256:
                raise SystemExit(f"{r.request_id}: the request text differs from the one sent")
            sid = f"qwen|{task}|{QWEN}|{r.request_id}"
            rows.append({"sample_id": sid, "source": "qwen", "task": task, "model_key": QWEN,
                         "config_id": r.config_id, "item_id": r.item_id, "request_id": r.request_id,
                         "n_tokens": int(r.n_tokens), "bin": r.bin, "stratum": r.stratum, "N_h": int(r.N_h),
                         "M_h": float(r.M_h), "n_h": int(r.n_h), "p_incl": float(r.p_incl), "mult": 1,
                         "finish_reason": r.finish_reason, "question": q, "reply": r.reply})
            keys.append({"sample_id": sid, "gold": gold.get(str(r.item_id))})
    print(f"qwen4-max: {len(qs)} sampled of {len(qp)} replies", flush=True)

    if len({r["sample_id"] for r in rows}) != len(rows):
        raise SystemExit("duplicated sample ids")
    write_jsonl(SAMPLE, rows)
    write_jsonl(KEYFILE, keys)
    st = pd.concat(strata, ignore_index=True)
    st.to_csv(STRATA, index=False, encoding="utf-8", lineterminator="\n")
    write_json(POPULATION, {"population": population, "checks": checks})
    cells = (pd.DataFrame(rows).groupby(["source", "task", "bin"]).size().rename("n").reset_index())
    pop_cells = st.groupby(["source", "task", "bin"])[["N"]].sum().reset_index()
    cells = cells.merge(pop_cells, on=["source", "task", "bin"], how="right").fillna({"n": 0})
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": int(args.seed),
           "rng": "numpy default_rng([seed, zlib.crc32(stratum)]); rows sorted by (config_id, item_id) before the draw",
           "bins": list(BINS), "per_model_bin": PER_BIN, "qwen_alloc": QWEN_ALLOC,
           "allocation": "per model x bin min(7, rows); a model's shortfall round-robin to its other bins (1-16, "
                         "17-64, 65+ order); Qwen4-Max 15/20/20 per benchmark, the same rule",
           "population": "panel: every (model, configuration, item) of the distinct discovery configurations, arm "
                         "direct_answer; row weight = draws of its configuration (as w); Qwen4-Max: every answered "
                         "requested reply on GSM8K and MATH-500 (every template once, 100 items), weight 1",
           "token_count": {"panel": "n_output_tokens of the stored output (vLLM token ids; the count w uses)",
                           "qwen": "OpenRouter completion tokens minus reasoning tokens (scripts/closed_w.api_count)"},
           "judge_view": "question (the user message the model received; HotpotQA with its passages) and reply only",
           "n_rows": len(rows), "n_panel": sum(r["source"] == "panel" for r in rows),
           "n_qwen": sum(r["source"] == "qwen" for r in rows),
           "cells": cells.to_dict("records"),
           "cells_below_15": cells[cells["n"] < 15].to_dict("records"),
           "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
           "sources": {"qwen_raw": rel(QWEN_RAW), "hotpot": "data/processed/hotpot/scored_content.parquet",
                       "panel": "scripts/run_baseline_mixture.task_rows (GSM8K, MATH-500, BIG-Bench Hard); texts "
                                "data/d2_out/<collection>/<model>/shard-*.parquet; prompts data/requests/<task>/"},
           "checks": checks, "runtime_s": round(time.time() - t0, 1)}
    write_json(MANIFEST, man)
    write_design_md()
    print(cells.to_string(index=False))
    print(f"wrote {rel(SAMPLE)} ({len(rows)} rows), {rel(KEYFILE)}, {rel(STRATA)}, {rel(POPULATION)}, {rel(MANIFEST)}")


def write_design_md() -> str:
    """results/reading_audit/design.md and design_clean.md: the sample's design and counts (from the saved manifest,
    strata and population; the judge's dry run if it exists)."""
    man = json.load(open(MANIFEST, encoding="utf-8"))
    st = pd.read_csv(STRATA)
    pop = json.load(open(POPULATION, encoding="utf-8"))["population"]
    L = ["# Reading audit: the sample", "",
         f"Seed {man['seed']} ({man['rng']}). {man['n_rows']:,} replies: {man['n_panel']:,} from the panel's "
         f"requested-direct outputs on the discovery draws and {man['n_qwen']} from Qwen4-Max. Population: "
         f"{man['population']}. Token counts: panel, {man['token_count']['panel']}; Qwen4-Max, "
         f"{man['token_count']['qwen']}. Allocation: {man['allocation']}. The judge sees {man['judge_view']}.", "",
         "| Source | Benchmark | Bin (tokens) | Population rows | Share of the population (%) | Sampled |",
         "|---|---|---|---|---|---|"]
    for (src, task), g in st.groupby(["source", "task"], sort=False):
        tot = g["N"].sum()
        for b in BINS:
            x = g[g["bin"] == b]
            N, n = int(x["N"].sum()), int(x["n"].sum())
            L.append(f"| {'panel' if src == 'panel' else 'Qwen4-Max'} | {TASK_NAME[task]} | {b} | {N:,} | "
                     f"{100 * N / tot:.1f} | {n} |")
    small = st[(st["source"] == "panel") & (st["N"] < PER_BIN)]
    L += ["", f"Benchmark x bin cells with fewer than 15 sampled: {len(man['cells_below_15'])}. Model x bin strata "
              f"with fewer than {PER_BIN} rows (all taken; the shortfall went to the model's other bins): "
              f"{len(small)}, of them {int((small['N'] == 0).sum())} empty:"]
    for r in small.itertuples():
        L.append(f"- {TASK_NAME[r.task]}, {r.model_key}, {r.bin}: {r.N} rows")
    L += ["", "Models per benchmark: " + ", ".join(f"{TASK_NAME[t]} {len(pop[t]['models'])}" for t in TASKS) + ". "
          "Qwen4-Max's share of requested replies over 16 tokens (every template once): " +
          ", ".join(f"{TASK_NAME[t]} {100 * pop['qwen'][t]['w16']:.1f}%" for t in QWEN_TASKS) + ".",
          "", "Recomputed against the saved values (max absolute difference): " +
          "; ".join(f"{TASK_NAME[t]} " + ", ".join(f"{k} {v:.1e}" for k, v in c.items() if k != "source")
                    for t, c in man["checks"].items()) + "."]
    if HUMAN_KEY.exists():
        hk = pd.read_csv(HUMAN_KEY, dtype=str, keep_default_na=False)
        L += ["", f"Human check: {len(hk)} rows of the panel part, drawn before judging, stratified by benchmark x "
                  "length bin (12 cells, 12 or 13 each), the same rows for two raters in two seeded orders "
                  "(human_rater1.csv, human_rater2.csv; key human_key.csv)."]
    if DRYRUN.exists():
        d = json.load(open(DRYRUN, encoding="utf-8"))
        if d.get("judge") == JUDGE["id"]:
            pb = JUDGE["price_batch"]
            L += ["", f"Judge dry run ({JUDGE['label']}, `{JUDGE['id']}`, Message Batches API at "
                      f"${pb['input'] * 1e6:.2f} / ${pb['output'] * 1e6:.2f} per million input / output tokens; "
                      f"{SAMPLING}): {d['requests_to_send']:,} requests ({d['main_to_send']:,} sample rows and "
                      f"{d['repeat_to_send']} repeats); input {d['input_tokens_o200k']:,} tokens ({d['tokenizer']}), "
                      f"{d['input_bytes']:,} bytes; expected ${d['expected']['usd']:.2f}, high "
                      f"${d['high']['usd']:.2f}, worst ${d['worst']['usd']:.2f} ({d['worst']['note']})."]
    md = "\n".join(L) + "\n"
    (OUT / "design.md").write_text(md, encoding="utf-8")
    (OUT / "design_clean.md").write_text(clean_md(md), encoding="utf-8")
    return md


# ================================================================================================ the judge


def rubric_parts(path: Path = RUBRIC) -> tuple[str, str]:
    """(system message, user template) between the markers of rubric.md, byte for byte."""
    txt = path.read_text(encoding="utf-8")
    m1 = re.search(r"<!-- RUBRIC START -->\n(.*?)\n<!-- RUBRIC END -->", txt, re.S)
    m2 = re.search(r"<!-- TEMPLATE START -->\n(.*?)\n<!-- TEMPLATE END -->", txt, re.S)
    if not (m1 and m2):
        raise SystemExit(f"{rel(path)}: rubric or template markers missing")
    if "{question}" not in m2.group(1) or "{reply}" not in m2.group(1):
        raise SystemExit(f"{rel(path)}: the template lacks {{question}} or {{reply}}")
    return m1.group(1), m2.group(1)


def judge_user(template: str, question: str, reply: str) -> str:
    """The template with the prompt and the reply filled in; the reply goes in after the question, so a "{reply}"
    inside a question is never filled."""
    user = template.replace("{question}", question, 1)
    i = user.rfind("{reply}")
    return user[:i] + reply + user[i + len("{reply}"):]


def judge_params(system: str, user: str) -> dict:
    """The Messages API body of one judge call, the same for the pilot and the batch: the rubric in the top-level
    system field, one user message, no prefill, thinking disabled at effort medium, and no sampling parameter."""
    return {"model": JUDGE["id"], "max_tokens": MAX_TOKENS, "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": dict(THINKING), "output_config": dict(OUTPUT_CONFIG)}


def judge_config(system: str, template: str) -> dict:
    return {"model": JUDGE["id"], "max_tokens": MAX_TOKENS, "thinking": THINKING, "output_config": OUTPUT_CONFIG,
            "sampling": "default (no temperature, top_p or top_k sent)", "anthropic_version": ANTHROPIC_VERSION,
            "api_base": API_BASE, "rubric_sha256": sha256_text(system), "template_sha256": sha256_text(template)}


def judge_meta(system: str, template: str) -> dict:
    """What every judge line records besides its own fields."""
    return {"judge_model": JUDGE["id"], "config_sha": sha256_text(json.dumps(judge_config(system, template),
                                                                             sort_keys=True)),
            "rubric_sha256": sha256_text(system), "template_sha256": sha256_text(template)}


def judge_config_sha(system: str, template: str) -> str:
    return judge_meta(system, template)["config_sha"]


CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
KIND_PREFIX = {"main": "m", "repeat": "r"}


def custom_id(sample_id: str, task: str, kind: str = "main") -> str:
    """m_<task>_<20 hex of sha256(sample_id)> for a sample row, r_<...> for its repeat; matches ^[a-zA-Z0-9_-]{1,64}$."""
    cid = f"{KIND_PREFIX[kind]}_{task}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:20]}"
    if not CUSTOM_ID.fullmatch(cid):
        raise ValueError(f"custom_id {cid!r}")
    return cid


def parse_label(text) -> str | None:
    """One letter R, P or A. Tags such as <...> are removed first (with thinking disabled, the judge can write internal XML
    tags into its visible text); then surrounding blanks, quotes, asterisks, backticks and brackets and a final
    period are allowed. Anything else (two letters, a word, a lowercase letter, text left from inside tags) is not
    read: it stays unread and is counted, never guessed."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"<[^<>]*>", "", text)
    strip = "*`'\"()[] \t\r\n"

    def one(s):
        s = s.strip(strip).rstrip(".").strip(strip)
        return s if s in CLASSES else None
    # In a pilot, 1 of 20 replies began to reason aloud ("Wait — the answer is 2, but we don't jud") and hit the
    # 16-token cap. With room to finish (MAX_TOKENS 256), such a reply is read only if its last non-empty line is the
    # label alone; anything else stays unread.
    whole = one(t)
    if whole is not None:
        return whole
    lines = [x for x in t.splitlines() if x.strip(strip)]
    return one(lines[-1]) if len(lines) > 1 else None


def worst_usd(n_bytes: int, n_counted, price: dict) -> float:
    """The most one call can cost: input at its exact count x COUNT_MARGIN + COUNT_PAD when counted (count_tokens),
    else every message byte a token plus PROMPT_OVERHEAD; output at max_tokens."""
    if n_counted is not None:
        inp = math.ceil(int(n_counted) * COUNT_MARGIN) + COUNT_PAD
    else:
        inp = int(n_bytes) + PROMPT_OVERHEAD
    return inp * float(price["input"]) + MAX_TOKENS * float(price["output"])


def make_requests(rows: list[dict], system: str, template: str, kind: str = "main", counts: dict | None = None
                  ) -> list[dict]:
    """One judge request per sample row (kind main) or per repeat row (kind repeat)."""
    out = []
    for r in rows:
        user = judge_user(template, r["question"], r["reply"])
        tsha = sha256_text(system + "\x00" + user)
        c = (counts or {}).get(r["sample_id"])
        n_tok = int(c["input_tokens"]) if c and c.get("text_sha256") == tsha else None
        n_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8"))
        out.append({"custom_id": custom_id(r["sample_id"], r["task"], kind), "sample_id": r["sample_id"],
                    "kind": kind, "task": r["task"], "source": r["source"], "params": judge_params(system, user),
                    "n_bytes": n_bytes, "text_sha256": tsha, "input_tokens_counted": n_tok,
                    "worst_batch_usd": worst_usd(n_bytes, n_tok, JUDGE["price_batch"]),
                    "worst_standard_usd": worst_usd(n_bytes, n_tok, JUDGE["price_standard"])})
    return out


def id_frame(reqs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{k: r[k] for k in ("custom_id", "kind", "sample_id", "task", "source")} for r in reqs])
    if df["custom_id"].duplicated().any():
        raise SystemExit("custom_id collision")
    return df


class Files:
    """Where the judge's files live (the defaults under results/reading_audit/; tests pass a temporary folder)."""

    def __init__(self, folder: Path | None = None):
        d = Path(folder) if folder is not None else None
        self.raw = d / "judge_raw.jsonl" if d else JUDGE_RAW
        self.pilot = d / "judge_pilot.json" if d else PILOT
        self.batches = d / "judge_batch.json" if d else BATCHES
        self.counts = d / "judge_token_counts.json" if d else COUNTS
        self.idmap = d / "judge_custom_ids.csv" if d else IDMAP
        self.dryrun = d / "judge_dryrun.json" if d else DRYRUN
        self.repeat = d / "judge_repeat_set.csv" if d else REPEAT_SET


# ------------------------------------------------------------------------------------------------ raw file and money


_WRITE_LOCK = threading.Lock()


def redact(text, key: str | None = None) -> str:
    """The key, any Anthropic or OpenRouter key and any credential header removed (closed_w.redact plus sk-ant-)."""
    import closed_w as cw
    s = cw.redact(text, key)
    return re.sub(r"sk-ant-[A-Za-z0-9_\-]{8,}", "REDACTED", s)


def read_raw(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:       # a line cut by a crash mid-write
                continue
    return out


def append_raw(path: Path, rec: dict, key: str | None = None) -> None:
    line = redact(dumps(rec), key)
    with _WRITE_LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
            fh.flush()


def judge_state(lines: list[dict], csha: str) -> dict:
    """succeeded: {custom_id: its first succeeded result under the configuration}; in_flight: the custom_ids of
    batches submitted under it and not yet collected; submitted, collected: by batch id."""
    submitted, collected, succ = {}, set(), {}
    for x in lines:
        ev = x.get("event")
        if ev == "batch_submitted":
            submitted[x["batch_id"]] = x
        elif ev == "batch_collected":
            collected.add(x["batch_id"])
        elif ev == "result" and x.get("config_sha") == csha and x.get("result_type") == "succeeded":
            succ.setdefault(x["custom_id"], x)
    inflight = set()
    for b, x in submitted.items():
        if b not in collected and x.get("config_sha") == csha:
            inflight |= set(x.get("custom_ids") or [])
    return {"succeeded": succ, "in_flight": inflight, "submitted": submitted, "collected": collected}


def ledger(lines: list[dict]) -> dict:
    """What judge_raw.jsonl has cost or may still cost: pilot calls at their bill (a call with no HTTP reply, or one
    logged as sent and never answered, at its worst case), collected batches at the bill of their results, batches
    not yet collected at their worst case."""
    res = {x["attempt_id"]: x for x in lines if x.get("event") == "result" and x.get("attempt_id")}
    orphans = [x for x in lines if x.get("event") == "sent" and x.get("attempt_id") not in res]
    pilot_charged = sum(float(x.get("charged_usd") or 0.0) for x in res.values())
    pilot_billed = sum(float(x.get("cost_usd") or 0.0) for x in res.values())
    orphan_usd = sum(float(x.get("reserve_usd") or 0.0) for x in orphans)
    submitted = {x["batch_id"]: x for x in lines if x.get("event") == "batch_submitted"}
    collected = {x["batch_id"] for x in lines if x.get("event") == "batch_collected"}
    batch_billed = sum(float(x.get("cost_usd") or 0.0) for x in lines
                       if x.get("event") == "result" and x.get("route") == "batch" and x.get("batch_id") in collected)
    reserved = sum(float(x.get("reserve_usd") or 0.0) for b, x in submitted.items() if b not in collected)
    return {"counted_usd": pilot_charged + orphan_usd + batch_billed + reserved,
            "billed_usd": pilot_billed + batch_billed, "pilot_billed_usd": pilot_billed,
            "batch_billed_usd": batch_billed, "reserved_uncollected_usd": reserved, "orphans": len(orphans),
            "orphans_usd": orphan_usd, "batches_submitted": len(submitted), "batches_collected": len(collected)}


def message_fields(msg) -> dict:
    """One Messages API message read into flat fields (text = the text blocks joined; label read from it)."""
    msg = msg if isinstance(msg, dict) else {}
    content = msg.get("content") if isinstance(msg.get("content"), list) else []
    types = [str(b.get("type")) for b in content if isinstance(b, dict)]
    text = "".join(str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    u = msg.get("usage") if isinstance(msg.get("usage"), dict) else None
    g = (lambda k: u.get(k) if u else None)  # noqa: E731
    return {"message_id": msg.get("id"), "model": msg.get("model"), "stop_reason": msg.get("stop_reason"),
            "content_types": types, "thinking_blocks": sum(t in ("thinking", "redacted_thinking") for t in types),
            "text": text if msg else None, "label": parse_label(text) if msg else None, "usage": u,
            "input_tokens": g("input_tokens"), "output_tokens": g("output_tokens"),
            "cache_creation_input_tokens": g("cache_creation_input_tokens"),
            "cache_read_input_tokens": g("cache_read_input_tokens")}


def usage_cost(f: dict, price: dict) -> float | None:
    """USD from a message's usage at the given prices (cache writes 1.25x, cache reads 0.1x the input price)."""
    if f.get("input_tokens") is None and f.get("output_tokens") is None:
        return None
    pi, po = float(price["input"]), float(price["output"])
    return (int(f.get("input_tokens") or 0) * pi + int(f.get("output_tokens") or 0) * po
            + int(f.get("cache_creation_input_tokens") or 0) * pi * JUDGE["cache_write_factor"]
            + int(f.get("cache_read_input_tokens") or 0) * pi * JUDGE["cache_read_factor"])


# ------------------------------------------------------------------------------------------------ HTTP


class AnthropicHttp:
    """Anthropic's API over urllib: x-api-key, anthropic-version and content-type on every call. The key goes only to
    the API's own host (a results_url elsewhere is refused) and every error text is redacted."""

    def __init__(self, key: str | None, base: str = API_BASE, version: str = ANTHROPIC_VERSION,
                 timeout: float = 600.0, opener=None):
        import urllib.parse
        import urllib.request
        self._key, self.base, self.version, self.timeout = key, base.rstrip("/"), version, float(timeout)
        self.opener = opener or urllib.request.urlopen
        self.host = urllib.parse.urlparse(self.base).netloc

    def call(self, method: str, url: str, body: dict | None = None, text: bool = False) -> dict:
        import urllib.error
        import urllib.parse
        import urllib.request
        if not url.startswith("https://"):
            url = self.base + url
        if urllib.parse.urlparse(url).netloc != self.host:
            raise SystemExit(f"refusing to send the key to another host: {redact(url, self._key)}")
        data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("x-api-key", self._key or "")
        req.add_header("anthropic-version", self.version)
        req.add_header("content-type", "application/json")
        t0 = time.time()
        try:
            with self.opener(req, timeout=self.timeout) as r:
                raw = r.read()
                st = getattr(r, "status", None) or r.getcode()
            s = raw.decode("utf-8", "replace")
            if text:
                return {"status": st, "json": None, "text": s, "seconds": round(time.time() - t0, 3)}
            try:
                return {"status": st, "json": json.loads(s), "seconds": round(time.time() - t0, 3)}
            except ValueError:
                return {"status": st, "json": None, "error": redact(s[:2000], self._key),
                        "seconds": round(time.time() - t0, 3)}
        except urllib.error.HTTPError as e:
            try:
                txt = redact(e.read().decode("utf-8", "replace"), self._key)
            except Exception:  # noqa: BLE001
                txt = ""
            try:
                j = json.loads(txt)
            except ValueError:
                j = None
            return {"status": e.code, "json": j, "error": txt[:2000], "seconds": round(time.time() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            return {"status": None, "json": None, "error": redact(f"{type(e).__name__}: {e}", self._key),
                    "seconds": round(time.time() - t0, 3)}


def call_retry(http, method: str, url: str, body: dict | None = None, text: bool = False, tries: int = MAX_TRIES,
               retry_none: bool = True, sleep=time.sleep) -> dict:
    """Retries a status in RETRY_STATUS (and no reply at all, if retry_none) with backoff."""
    r = {"status": None, "json": None, "error": "not sent"}
    for k in range(int(tries)):
        r = http.call(method, url, body, text)
        st = r.get("status")
        if (st is not None and st not in RETRY_STATUS) or (st is None and not retry_none):
            return r
        if k < tries - 1:
            sleep(min(60.0, 2.0 * 2 ** k))
    return r


# ------------------------------------------------------------------------------------------------ dry run and counts


def token_counter(path=None, cache=None):
    """o200k_base through tiktoken (an approximation of the judge's tokenizer), or None."""
    if path and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    if cache:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:  # noqa: BLE001
        return None


def load_counts(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    d = json.load(open(path, encoding="utf-8"))
    if d.get("model") != JUDGE["id"]:
        return {}
    return d.get("counts") or {}


def todo_requests(main_reqs: list[dict], rep_reqs: list[dict], state: dict) -> list[dict]:
    """Rows not yet labelled under the configuration and not in a batch still running: sample rows first (sample
    order), then the repeat set."""
    skip = set(state["succeeded"]) | state["in_flight"]
    return [r for r in main_reqs + rep_reqs if r["custom_id"] not in skip]


def dry_run(main_reqs: list[dict], rep_reqs: list[dict], state: dict, count=None) -> dict:
    todo = todo_requests(main_reqs, rep_reqs, state)
    pb, ps = JUDGE["price_batch"], JUDGE["price_standard"]
    by, sys_tok = {}, {}
    for r in todo:
        b = by.setdefault(f"{r['kind']}|{r['source']}|{r['task']}", {"requests": 0, "bytes": 0, "o200k": 0})
        b["requests"] += 1
        b["bytes"] += r["n_bytes"]
        if count is not None:
            s = r["params"]["system"]
            if s not in sys_tok:
                sys_tok[s] = count(s)
            b["o200k"] += sys_tok[s] + count(r["params"]["messages"][0]["content"])
    n = len(todo)
    tot_bytes = sum(b["bytes"] for b in by.values())
    tot_tok = sum(b["o200k"] for b in by.values()) if count is not None else None
    base_in = (tot_tok + 8 * n) if tot_tok is not None else tot_bytes / 4
    exp_in, high_in = EXPECTED_O200K * base_in, HIGH_O200K * base_in
    counted = [r["input_tokens_counted"] for r in todo]
    out = {"judge": JUDGE["id"], "api": "Anthropic Message Batches API", "anthropic_version": ANTHROPIC_VERSION,
           "requests_to_send": n, "main_to_send": sum(r["kind"] == "main" for r in todo),
           "repeat_to_send": sum(r["kind"] == "repeat" for r in todo),
           "main_in_sample": len(main_reqs), "repeat_in_set": len(rep_reqs),
           "already_labelled": len(state["succeeded"]), "in_flight": len(state["in_flight"]),
           "price_per_token_batch": pb, "price_per_token_standard": ps, "max_tokens": MAX_TOKENS,
           "thinking": THINKING, "output_config": OUTPUT_CONFIG, "sampling": SAMPLING,
           "input_tokens_o200k": tot_tok, "input_bytes": tot_bytes, "by_cell": by,
           "tokenizer": "tiktoken o200k_base" if count is not None else "none (bytes / 4 used as the base)",
           "expected": {"note": f"input {EXPECTED_O200K} x the o200k count plus 8 per call (tiktoken "
                                f"undercounts the judge model's tokens by about 15-20%), {EXPECTED_OUT} output tokens per call",
                        "input_tokens": exp_in, "output_tokens": EXPECTED_OUT * n,
                        "usd": exp_in * pb["input"] + EXPECTED_OUT * n * pb["output"]},
           "high": {"note": f"input {HIGH_O200K} x the o200k count plus 8 per call, every call at max_tokens",
                    "input_tokens": high_in, "output_tokens": MAX_TOKENS * n,
                    "usd": high_in * pb["input"] + MAX_TOKENS * n * pb["output"]},
           "worst": {"note": f"every message byte a token plus {PROMPT_OVERHEAD} per call, every call at max_tokens "
                             "(a byte-level tokenizer never gives more tokens than bytes); errored, canceled and "
                             "expired requests are not billed",
                     "input_tokens": tot_bytes + PROMPT_OVERHEAD * n, "output_tokens": MAX_TOKENS * n,
                     "usd": sum(worst_usd(r["n_bytes"], None, pb) for r in todo)}}
    if n and all(c is not None for c in counted):
        cin = sum(counted)
        out["counted"] = {"note": f"exact input counts (count_tokens, {JUDGE['id']}) x {COUNT_MARGIN} plus "
                                  f"{COUNT_PAD} per call, every call at max_tokens: the worst case --submit uses",
                          "input_tokens_exact": cin, "usd": sum(r["worst_batch_usd"] for r in todo),
                          "expected_usd": cin * pb["input"] + EXPECTED_OUT * n * pb["output"]}
    else:
        out["counted"] = {"note": f"count_tokens covers {sum(c is not None for c in counted)} of {n} requests; "
                                  "--submit uses the byte bound for the rest", "usd": None}
    pil = [r for r in todo if r["kind"] == "main" and r["source"] == "panel"]
    pil = pilot_rows(pil, set())
    out["pilot"] = {"requests": len(pil), "price": "standard (ordinary Messages API)",
                    "worst_usd": sum(r["worst_standard_usd"] for r in pil),
                    "expected_usd": sum(EXPECTED_O200K * r["n_bytes"] / 4 * ps["input"] + EXPECTED_OUT * ps["output"]
                                        for r in pil)}
    return out


def count_tokens(http, reqs: list[dict], path: Path, threads: int = THREADS, sleep=time.sleep) -> dict:
    """Exact input counts from POST /v1/messages/count_tokens (free), one per distinct sample row (a repeat has the
    same text as its row), saved to path as they arrive; a row already counted for the same text is skipped."""
    counts = load_counts(path)
    todo, seen = [], set()
    for r in reqs:
        c = counts.get(r["sample_id"])
        if r["sample_id"] in seen or (c and c.get("text_sha256") == r["text_sha256"]):
            continue
        seen.add(r["sample_id"])
        todo.append(r)
    failed = []

    def one(r):
        p = r["params"]
        body = {"model": p["model"], "system": p["system"], "messages": p["messages"]}
        resp = call_retry(http, "POST", "/v1/messages/count_tokens", body, sleep=sleep)
        j = resp.get("json") if isinstance(resp.get("json"), dict) else {}
        if resp.get("status") == 200 and isinstance(j.get("input_tokens"), int):
            return r, {"input_tokens": int(j["input_tokens"]), "text_sha256": r["text_sha256"]}, None
        return r, None, {"status": resp.get("status"), "error": (resp.get("error") or "")[:300]}

    def save():
        write_json(path, {"model": JUDGE["id"], "anthropic_version": ANTHROPIC_VERSION,
                          "endpoint": "POST /v1/messages/count_tokens (free)", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "n": len(counts), "counts": counts})

    chunk = max(1, int(threads) * 25)
    for i in range(0, len(todo), chunk):
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
            for r, c, err in ex.map(one, todo[i:i + chunk]):
                if c is not None:
                    counts[r["sample_id"]] = c
                else:
                    failed.append({"sample_id": r["sample_id"], **err})
        save()
        print(f"  counted {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    if not todo:
        save()
    return {"counted_now": len(todo) - len(failed), "failed": failed, "total": len(counts)}


# ------------------------------------------------------------------------------------------------ pilot


def pilot_rows(reqs: list[dict], done: set, seed: int = SEED) -> list[dict]:
    """PILOT_PER_TASK pending panel rows per benchmark, a seeded draw (spread over models and bins)."""
    out = []
    for t in TASKS:
        pend = [r for r in reqs if r["task"] == t and r["source"] == "panel" and r["kind"] == "main"
                and r["custom_id"] not in done]
        if pend:
            idx = stratum_rng(seed, f"pilot|{t}").permutation(len(pend))[:PILOT_PER_TASK]
            out += [pend[i] for i in sorted(idx)]
    return out


def _line_base(req: dict, meta: dict) -> dict:
    return {"custom_id": req["custom_id"], "sample_id": req["sample_id"], "kind": req["kind"], "task": req["task"],
            "source": req["source"], **meta}


def pilot_call(http, req: dict, raw: Path, meta: dict, key: str | None, attempt: int) -> dict:
    """One ordinary Messages API call: a "sent" line, the call, a "result" line."""
    aid = uuid.uuid4().hex
    base = {**_line_base(req, meta), "route": "messages", "batch_id": None, "attempt_id": aid, "attempt": attempt}
    append_raw(raw, {"event": "sent", **base, "reserve_usd": req["worst_standard_usd"], "t": _now()}, key)
    resp = http.call("POST", "/v1/messages", req["params"])
    st, j = resp.get("status"), resp.get("json")
    ok = st == 200 and isinstance(j, dict) and j.get("type") == "message"
    f = message_fields(j if ok else None)
    cost = usage_cost(f, JUDGE["price_standard"]) if ok else 0.0
    if st is None or (ok and cost is None):      # no reply (it may have been billed) or a reply without usage
        charged = req["worst_standard_usd"]
    else:
        charged = cost if ok else 0.0             # an HTTP error is not billed
    rec = {"event": "result", **base, "result_type": "succeeded" if ok else "errored", "http_status": st, **f,
           "cost_usd": cost, "charged_usd": charged, "price_basis": "standard",
           "error": None if ok else (resp.get("error") or json.dumps(j, default=str))[:2000],
           "seconds": resp.get("seconds"), "t": _now()}
    append_raw(raw, rec, key)
    return rec


def pilot_checks(res: list[dict], n_expected: int) -> list[dict]:
    def chk(name, ok, detail):
        return {"name": name, "ok": bool(ok), "detail": detail}
    out = [chk("calls", len(res) == n_expected and n_expected > 0, {"calls": len(res), "expected": n_expected}),
           chk("http_200", all(r.get("http_status") == 200 and r.get("result_type") == "succeeded" for r in res),
               {"status": sorted({str(r.get("http_status")) for r in res}),
                "errors": [r.get("error") for r in res if r.get("result_type") != "succeeded"][:3]}),
           chk("one_letter", all(r.get("label") in CLASSES for r in res),
               {"unread": [r.get("text") for r in res if r.get("label") not in CLASSES][:5],
                "labels": {c: sum(r.get("label") == c for r in res) for c in CLASSES}}),
           chk("usage", all(isinstance(r.get("input_tokens"), int) and isinstance(r.get("output_tokens"), int)
                            for r in res), {"missing": sum(not isinstance(r.get("input_tokens"), int) for r in res)}),
           chk("no_thinking_block", all(not r.get("thinking_blocks") and set(r.get("content_types") or []) <= {"text"}
                                        for r in res),
               {"content_types": sorted({t for r in res for t in (r.get("content_types") or [])})}),
           chk("model", all(MODEL_ECHO.fullmatch(str(r.get("model") or "")) for r in res),
               {"echoed": sorted({str(r.get("model")) for r in res})}),
           chk("no_refusal", all(r.get("stop_reason") in ("end_turn", "max_tokens", "stop_sequence") for r in res),
               {"stop_reasons": sorted({str(r.get("stop_reason")) for r in res})})]
    return out


def run_pilot(http, rows: list[dict], files: Files, meta: dict, limit: float, key: str | None = None,
              sleep=time.sleep) -> dict:
    """The pilot: each row through the ordinary Messages API (retried on a retryable status or no reply), the spend
    checked against the limit before every attempt; writes files.pilot."""
    res, stopped = [], None
    for r in rows:
        rec = None
        for k in range(MAX_TRIES):
            counted = ledger(read_raw(files.raw))["counted_usd"]
            if counted + r["worst_standard_usd"] > limit:
                stopped = f"limit: counted ${counted:.4f} + this call's worst ${r['worst_standard_usd']:.4f} > ${limit:.2f}"
                break
            rec = pilot_call(http, r, files.raw, meta, key, k + 1)
            st = rec["http_status"]
            if st is not None and st not in RETRY_STATUS:
                break
            sleep(min(60.0, 2.0 * 2 ** k))
        if stopped:
            break
        res.append(rec)
    checks = pilot_checks(res, len(rows))
    passed = stopped is None and all(c["ok"] for c in checks)
    billed = sum(float(x.get("cost_usd") or 0.0) for x in res)
    ok = [x for x in res if isinstance(x.get("input_tokens"), int)]
    by_req = {r["custom_id"]: r for r in rows}
    ratio_bytes = (sum(x["input_tokens"] for x in ok) / sum(by_req[x["custom_id"]]["n_bytes"] for x in ok)) if ok else None
    cnt = [(x["input_tokens"], by_req[x["custom_id"]]["input_tokens_counted"]) for x in ok
           if by_req[x["custom_id"]]["input_tokens_counted"] is not None]
    out = {"config_sha": meta["config_sha"], "passed": passed, "checks": checks, "stopped": stopped,
           "calls": len(res), "billed_usd": billed, "price_basis": "standard (ordinary Messages API)",
           "billed_input_tokens_per_byte": ratio_bytes,
           "billed_over_counted_input_tokens": [round(a / b, 4) for a, b in cnt if b] or None,
           "rows": [{k: x.get(k) for k in ("custom_id", "task", "http_status", "model", "stop_reason", "content_types",
                                           "text", "label", "input_tokens", "output_tokens", "cost_usd")} for x in res],
           "judge": {"model": JUDGE["id"], "max_tokens": MAX_TOKENS, "thinking": THINKING,
                     "output_config": OUTPUT_CONFIG, "sampling": SAMPLING, "anthropic_version": ANTHROPIC_VERSION},
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(files.pilot, out)
    return out


# ------------------------------------------------------------------------------------------------ batch


def batches_summary(lines: list[dict]) -> dict:
    """judge_batch.json: every submitted batch with its state, rebuilt from the raw file."""
    col = {x["batch_id"]: x for x in lines if x.get("event") == "batch_collected"}
    out = []
    for x in lines:
        if x.get("event") != "batch_submitted":
            continue
        c = col.get(x["batch_id"])
        out.append({"batch_id": x["batch_id"], "submitted": x.get("t"), "n_requests": x.get("n_requests"),
                    "n_main": x.get("n_main"), "n_repeat": x.get("n_repeat"), "reserve_usd": x.get("reserve_usd"),
                    "worst_basis": x.get("worst_basis"), "config_sha": x.get("config_sha"),
                    "expires_at": x.get("expires_at"), "collected": c is not None,
                    "result_counts": c.get("counts") if c else None, "cost_usd": c.get("cost_usd") if c else None})
    return {"batches": out, "latest_batch_id": out[-1]["batch_id"] if out else None,
            "judge": JUDGE["id"], "api": "Anthropic Message Batches API", "anthropic_version": ANTHROPIC_VERSION}


def submit(http, todo: list[dict], files: Files, meta: dict, limit: float, split: bool = False,
           key: str | None = None, sleep=time.sleep) -> dict:
    """One batch with the given rows; refuses if spent so far + the batch's worst case exceeds the limit (with split:
    sends the part that fits, in order)."""
    lines = read_raw(files.raw)
    counted = ledger(lines)["counted_usd"]
    if not todo:
        return {"status": "nothing to send", "counted_usd": counted}
    worst = sum(r["worst_batch_usd"] for r in todo)
    send = list(todo)
    if counted + worst > limit:
        if not split:
            return {"status": "refused", "counted_usd": counted, "worst_usd": worst, "limit_usd": limit,
                    "requests": len(todo),
                    "message": f"spent so far ${counted:.4f} + the batch's worst case ${worst:.4f} > ${limit:.2f}"}
        room, send = limit - counted, []
        for r in todo:
            if r["worst_batch_usd"] > room:
                break
            send.append(r)
            room -= r["worst_batch_usd"]
        if not send:
            return {"status": "refused", "counted_usd": counted, "worst_usd": worst, "limit_usd": limit,
                    "requests": len(todo), "message": "not even one request fits under the limit"}
    worst_send = sum(r["worst_batch_usd"] for r in send)
    basis = ("count_tokens" if all(r["input_tokens_counted"] is not None for r in send) else
             "bytes" if all(r["input_tokens_counted"] is None for r in send) else "count_tokens where counted, else bytes")
    body = {"requests": [{"custom_id": r["custom_id"], "params": r["params"]} for r in send]}
    resp = {"status": None}
    for k in range(3):       # a 429 or 529 creates no batch; no reply or another error is not retried (the batch may
        resp = http.call("POST", "/v1/messages/batches", body)     # exist: check --list-batches before re-sending)
        if resp.get("status") not in (429, 529):
            break
        sleep(min(60.0, 5.0 * 2 ** k))
    j = resp.get("json") if isinstance(resp.get("json"), dict) else {}
    if resp.get("status") != 200 or not j.get("id"):
        rec = {"event": "submit_failed", **meta, "http_status": resp.get("status"), "n_requests": len(send),
               "error": redact((resp.get("error") or json.dumps(j, default=str))[:2000], key), "t": _now()}
        append_raw(files.raw, rec, key)
        return {"status": "failed", "http_status": resp.get("status"), "error": rec["error"],
                "message": "the batch may or may not exist: run --list-batches before submitting again"}
    rec = {"event": "batch_submitted", **meta, "batch_id": j["id"], "processing_status": j.get("processing_status"),
           "created_at": j.get("created_at"), "expires_at": j.get("expires_at"), "n_requests": len(send),
           "n_main": sum(r["kind"] == "main" for r in send), "n_repeat": sum(r["kind"] == "repeat" for r in send),
           "reserve_usd": worst_send, "worst_basis": basis, "custom_ids": [r["custom_id"] for r in send],
           "held_back": len(todo) - len(send), "t": _now()}
    append_raw(files.raw, rec, key)
    write_json(files.batches, batches_summary(read_raw(files.raw)))
    return {"status": "submitted", "batch_id": j["id"], "requests": len(send), "held_back": len(todo) - len(send),
            "reserve_usd": worst_send, "worst_basis": basis, "counted_before_usd": counted}


def read_idmap(path: Path) -> dict:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r["custom_id"]: r for r in df.to_dict("records")}


def ingest(http, batch_id: str, bobj: dict, files: Files, idmap: dict, meta_now: dict, key: str | None = None,
           sleep=time.sleep) -> dict:
    """Downloads an ended batch's results (JSONL, one line per request, any order), appends a result line per
    custom_id not yet recorded for this batch, then a batch_collected line."""
    url = bobj.get("results_url")
    if not url:
        return {"error": "ended without results_url"}
    r = call_retry(http, "GET", url, text=True, sleep=sleep)
    if r.get("status") != 200:
        return {"error": f"results download: HTTP {r.get('status')}: {(r.get('error') or '')[:300]}"}
    lines = read_raw(files.raw)
    sub = next((x for x in lines if x.get("event") == "batch_submitted" and x.get("batch_id") == batch_id), None)
    meta = ({k: sub[k] for k in ("judge_model", "config_sha", "rubric_sha256", "template_sha256")} if sub
            else dict(meta_now))
    have = {x["custom_id"] for x in lines if x.get("event") == "result" and x.get("batch_id") == batch_id}
    bad, unknown = 0, []
    for line in (r.get("text") or "").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            bad += 1
            continue
        cid = str(o.get("custom_id"))
        m = idmap.get(cid)
        if m is None:
            unknown.append(cid)
            continue
        if cid in have:
            continue
        res = o.get("result") if isinstance(o.get("result"), dict) else {}
        rt = res.get("type")
        f = message_fields(res.get("message") if rt == "succeeded" else None)
        cost = usage_cost(f, JUDGE["price_batch"]) if rt == "succeeded" else 0.0
        rec = {"event": "result", "custom_id": cid, "sample_id": m["sample_id"], "kind": m["kind"], "task": m["task"],
               "source": m["source"], **meta, "route": "batch", "batch_id": batch_id, "registered": sub is not None,
               "result_type": rt, "http_status": None, **f, "cost_usd": cost, "charged_usd": cost or 0.0,
               "price_basis": "batch",
               "error": None if rt == "succeeded" else json.dumps(res.get("error"), default=str)[:2000],
               "t": _now()}
        append_raw(files.raw, rec, key)
        have.add(cid)
    mine = [x for x in read_raw(files.raw) if x.get("event") == "result" and x.get("batch_id") == batch_id]
    counts = {t: sum(x.get("result_type") == t for x in mine) for t in ("succeeded", "errored", "canceled", "expired")}
    succ = [x for x in mine if x.get("result_type") == "succeeded"]
    labs = [parse_label(x.get("text")) for x in succ]
    missing = sorted(set(sub.get("custom_ids") or []) - have) if sub else []
    out = {"counts": counts, "results": len(mine), "missing": len(missing), "bad_lines": bad,
           "unknown_custom_ids": len(unknown), "unread": sum(lb is None for lb in labs),
           "labels": {c: sum(lb == c for lb in labs) for c in CLASSES},
           "thinking_blocks": sum(int(x.get("thinking_blocks") or 0) for x in succ),
           "models": sorted({str(x.get("model")) for x in succ}),
           "cost_usd": sum(float(x.get("cost_usd") or 0.0) for x in mine),
           "input_tokens": sum(int(x.get("input_tokens") or 0) for x in succ),
           "output_tokens": sum(int(x.get("output_tokens") or 0) for x in succ),
           "request_counts": bobj.get("request_counts"), "ended_at": bobj.get("ended_at")}
    append_raw(files.raw, {"event": "batch_collected", "batch_id": batch_id, **meta, **out, "t": _now()}, key)
    return out


def collect(http, files: Files, idmap: dict, meta_now: dict, poll_seconds: float = 60, once: bool = False,
            max_wait_s: float = 26 * 3600, batch_ids=None, key: str | None = None, sleep=time.sleep,
            clock=time.monotonic) -> dict:
    """Polls each batch (given, else every submitted one not yet collected) until it has ended, then ingests it."""
    lines = read_raw(files.raw)
    collected = {x["batch_id"] for x in lines if x.get("event") == "batch_collected"}
    pending = list(batch_ids) if batch_ids else [x["batch_id"] for x in lines if x.get("event") == "batch_submitted"
                                                  and x["batch_id"] not in collected]
    report, t0 = {}, clock()
    while pending:
        for b in list(pending):
            r = call_retry(http, "GET", f"/v1/messages/batches/{b}", sleep=sleep)
            j = r.get("json") if isinstance(r.get("json"), dict) else {}
            if r.get("status") != 200:
                report[b] = {"error": f"HTTP {r.get('status')}: {(r.get('error') or '')[:300]}"}
                if r.get("status") is not None:     # a refusal (401, 403, 404, ...) is not waited out; no reply is
                    pending.remove(b)
                continue
            if j.get("processing_status") == "ended":
                report[b] = {"processing_status": "ended", **ingest(http, b, j, files, idmap, meta_now, key, sleep)}
                pending.remove(b)
            else:
                report[b] = {"processing_status": j.get("processing_status"), "request_counts": j.get("request_counts")}
        if not pending or once or clock() - t0 > max_wait_s:
            break
        print(f"  {time.strftime('%H:%M:%S')} waiting: " +
              "; ".join(f"{b} {report[b].get('processing_status')} {report[b].get('request_counts')}" for b in pending),
              flush=True)
        sleep(poll_seconds)
    write_json(files.batches, batches_summary(read_raw(files.raw)))
    return {"batches": report, "pending": pending, "ledger": ledger(read_raw(files.raw))}


def _now() -> str:
    import closed_w as cw
    return cw.now_iso()


def judge_labels(csha: str | None = None, raw: Path = JUDGE_RAW, kind: str = "main") -> pd.DataFrame:
    """sample_id, label (R/P/A or None if unread), judge_text, route, batch_id: the first succeeded result of each row
    under the configuration (kind main: the sample rows; kind repeat: the repeat set)."""
    if csha is None:
        system, template = rubric_parts()
        csha = judge_config_sha(system, template)
    st = judge_state(read_raw(raw), csha)
    rows = [{"sample_id": x["sample_id"], "label": parse_label(x.get("text")), "judge_text": x.get("text"),
             "route": x.get("route"), "batch_id": x.get("batch_id")}
            for x in st["succeeded"].values() if x.get("kind") == kind]
    return pd.DataFrame(rows, columns=["sample_id", "label", "judge_text", "route", "batch_id"])


def judge_requests(sample: list[dict], system: str, template: str, counts: dict | None = None):
    """(main requests, repeat requests, repeat draw): every sample row, and the repeat set's rows again."""
    rep = cell_draw(panel_frame(sample), REPEAT_N, SEED, "repeat")
    by_id = {r["sample_id"]: r for r in sample}
    main_reqs = make_requests(sample, system, template, "main", counts)
    rep_reqs = make_requests([by_id[s] for s in rep["sample_id"]], system, template, "repeat", counts)
    return main_reqs, rep_reqs, rep


def cmd_judge(args) -> None:
    import closed_w as cw
    system, template = rubric_parts()
    meta = judge_meta(system, template)
    files = Files()
    sample = read_jsonl(SAMPLE)
    main_reqs, rep_reqs, rep = judge_requests(sample, system, template, load_counts(files.counts))
    idm = id_frame(main_reqs + rep_reqs)
    idm.to_csv(files.idmap, index=False, encoding="utf-8", lineterminator="\n")
    rep[["sample_id", "task", "bin", "cell", "stratum", "model_key", "n_tokens", "n_cell_sample", "n_cell_draw",
         "w_sample", "design_weight"]].assign(custom_id=[custom_id(s, t, "repeat") for s, t in
                                                          zip(rep["sample_id"], rep["task"])]).to_csv(
        files.repeat, index=False, encoding="utf-8", lineterminator="\n")
    state = judge_state(read_raw(files.raw), meta["config_sha"])
    if args.dry_run:
        out = dry_run(main_reqs, rep_reqs, state, token_counter(args.tiktoken_path, args.tiktoken_cache))
        out.update(config_sha=meta["config_sha"], rubric_sha256=meta["rubric_sha256"],
                   template_sha256=meta["template_sha256"], custom_id_map=rel(files.idmap),
                   repeat_set=rel(files.repeat), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(files.dryrun, out)
        print(json.dumps({k: out[k] for k in ("requests_to_send", "main_to_send", "repeat_to_send", "already_labelled",
                                              "input_tokens_o200k", "input_bytes", "tokenizer", "expected", "high",
                                              "worst", "counted", "pilot")}, indent=1))
        return
    key = cw.read_key(ROOT / ".env", "ANTHROPIC_API_KEY")
    http = AnthropicHttp(key)
    if args.list_batches:
        r = call_retry(http, "GET", "/v1/messages/batches?limit=20")
        for b in ((r.get("json") or {}).get("data") or []):
            print(b.get("id"), b.get("processing_status"), b.get("created_at"), b.get("request_counts"))
        if r.get("status") != 200:
            print(f"HTTP {r.get('status')}: {(r.get('error') or '')[:300]}")
        return
    if args.count_tokens:
        out = count_tokens(http, main_reqs, files.counts, args.threads)
        print(json.dumps({k: out[k] for k in ("counted_now", "total")}) + f"; failed {len(out['failed'])}"
              + (f", e.g. {out['failed'][:2]}" if out["failed"] else ""))
        return
    if args.collect:
        out = collect(http, files, read_idmap(files.idmap), meta, args.poll_seconds, args.once,
                      args.max_wait_hours * 3600, [args.batch_id] if args.batch_id else None, key)
        print(json.dumps(out, indent=1, default=str))
        lab = judge_labels(meta["config_sha"], files.raw)
        print(f"sample rows labelled {int(lab['label'].notna().sum())} of {len(sample)}; unread "
              f"{int(lab['label'].isna().sum())}; billed so far ${out['ledger']['billed_usd']:.4f}")
        if out["pending"]:
            raise SystemExit(5)
        return
    if not args.approved or args.limit_usd is None:
        raise SystemExit("paid calls need --approved and --limit-usd")
    if args.pilot:
        rows = pilot_rows(main_reqs, set(state["succeeded"]) | state["in_flight"])
        print(f"pilot: {len(rows)} calls through the Messages API", flush=True)
        out = run_pilot(http, rows, files, meta, args.limit_usd, key)
        for c in out["checks"]:
            print(("ok  " if c["ok"] else "FAIL") + f" {c['name']}: {c['detail']}", flush=True)
        print(f"billed ${out['billed_usd']:.4f}; input tokens per byte {out['billed_input_tokens_per_byte']}")
        if not out["passed"]:
            raise SystemExit(3)
        return
    if args.submit:
        pil = json.load(open(files.pilot, encoding="utf-8")) if files.pilot.exists() else None
        if not (pil and pil.get("config_sha") == meta["config_sha"] and pil.get("passed")):
            raise SystemExit("no passed pilot under this configuration: run --pilot first")
        todo = todo_requests(main_reqs, rep_reqs, state)
        out = submit(http, todo, files, meta, args.limit_usd, args.split, key)
        print(json.dumps(out, indent=1))
        if out["status"] == "refused":
            raise SystemExit(4)
        if out["status"] == "failed":
            raise SystemExit(6)


# ================================================================================================ human check and repeat set


def panel_frame(sample: list[dict]) -> pd.DataFrame:
    """The panel part of the sample with its design columns."""
    cols = ("sample_id", "source", "task", "bin", "stratum", "mult", "M_h", "model_key", "n_tokens")
    df = pd.DataFrame([{k: r[k] for k in cols} for r in sample if r["source"] == "panel"], columns=list(cols))
    return df


def cell_draw(panel: pd.DataFrame, n: int, seed: int, stream: str) -> pd.DataFrame:
    """n rows of the panel sample stratified by benchmark x length bin (12 cells), as even as possible: n // 12 per
    cell and one more in n % 12 cells chosen by a seeded draw (a cell with too few rows gives all; its shortfall goes
    round-robin to the others). Inside a cell a draw without replacement from the rows sorted by sample_id. Columns
    added: cell, w_sample (the row's weight in the sample: a stratified ratio weight, the draws of its configuration
    scaled so a stratum's weights sum to its draw-weighted size), n_cell_sample, n_cell_draw, p_phase2 (=
    n_cell_draw / n_cell_sample) and design_weight = w_sample / p_phase2."""
    p = panel.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    codes, mult, M = design(p)
    p["w_sample"] = weights(codes, mult, M)
    p["cell"] = p["task"] + "|" + p["bin"]
    cells = [f"{t}|{b}" for t in TASKS for b in BINS]
    base, extra_n = divmod(int(n), len(cells))
    extra = set(np.asarray(cells)[stratum_rng(seed, f"{stream}|extra").permutation(len(cells))[:extra_n]])
    alloc = allocate(p["cell"].value_counts().to_dict(), {c: base + (c in extra) for c in cells}, order=cells)
    parts = []
    for c in cells:
        g = p[p["cell"] == c].reset_index(drop=True)
        if alloc[c]:
            idx = np.sort(stratum_rng(seed, f"{stream}|{c}").choice(len(g), size=alloc[c], replace=False))
            parts.append(g.iloc[idx].assign(n_cell_sample=len(g), n_cell_draw=alloc[c]))
    d = pd.concat(parts, ignore_index=True)
    d["p_phase2"] = d["n_cell_draw"] / d["n_cell_sample"]
    d["design_weight"] = d["w_sample"] / d["p_phase2"]
    return d


SHEET_COLUMNS = ["hid", "benchmark", "prompt", "reply", "label", "unsure", "note"]
FORMULA_START = ("=", "+", "-", "@")
EXCEL_CELL_MAX = 32767


def human_draw(sample: list[dict], n: int = HUMAN_N, seed: int = SEED) -> pd.DataFrame:
    """The human check's rows (cell_draw, stream "human") with each rater's own order: hid_rater1, hid_rater2
    (H001.. in that sheet's order, two seeded permutations)."""
    d = cell_draw(panel_frame(sample), n, seed, "human")
    for k in (1, 2):
        perm = stratum_rng(seed, f"human|order|rater{k}").permutation(len(d))
        hid = np.empty(len(d), dtype=object)
        hid[perm] = [f"H{i + 1:03d}" for i in range(len(d))]
        d[f"hid_rater{k}"] = hid
    return d


def sheet_rows(d: pd.DataFrame, sample: list[dict], k: int) -> list[list]:
    smp = {r["sample_id"]: r for r in sample}
    out = []
    for r in d.sort_values(f"hid_rater{k}").itertuples(index=False):
        s = smp[r.sample_id]
        out.append([getattr(r, f"hid_rater{k}"), TASK_NAME[s["task"]], s["question"], s["reply"], "", "", ""])
    return out


def write_xlsx(path: Path, rows: list[list]) -> str | None:
    """The sheet as .xlsx (wrapped text, wide prompt and reply columns, a list R/P/A on the label column, every cell
    written as text); None if written, else the reason it was not."""
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return "openpyxl is not installed"
    try:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "labels"
        for j, h in enumerate(SHEET_COLUMNS, 1):
            ws.cell(row=1, column=j, value=h).font = Font(bold=True)
        top = Alignment(wrap_text=True, vertical="top")
        for i, row in enumerate(rows, 2):
            for j, v in enumerate(row, 1):
                c = ws.cell(row=i, column=j, value=v)
                c.data_type = "s"            # text, never a formula
                c.alignment = top
        for col, wd in zip("ABCDEFG", (7, 15, 90, 70, 7, 8, 30)):
            ws.column_dimensions[col].width = wd
        dv = DataValidation(type="list", formula1='"R,P,A"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"E2:E{len(rows) + 1}")
        ws.freeze_panes = "A2"
        wb.save(path)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def rater_instructions(system: str) -> str:
    body = system.split("\nOutput:")[0].rstrip()
    return "\n".join([
        "# Reading audit: instructions for the two raters", "",
        "Each rater gets the same 150 replies in a different order: rater 1 fills `human_rater1.csv` (or "
        "`human_rater1.xlsx`), rater 2 fills `human_rater2.csv` (or `human_rater2.xlsx`). Work alone and do not compare "
        "labels before both sheets are done. Do not open `human_key.csv`: it names the models and the sampling cells.",
        "",
        "For each row, read the prompt the model was given and the model's reply, then write one letter in the column "
        "`label`: R, P or A, by the rules below, which are the rules the LLM judge gets. The column `benchmark` names "
        "the task, because a few rules read differently by task. If you are unsure, still write your best label and "
        "put x in `unsure`. `note` is free text. Leave `hid`, `benchmark`, `prompt` and `reply` as they are.", "",
        "If a prompt or reply cell in the CSV opened in Excel starts with =, +, - or @ and shows an error or a number "
        "where text is expected, read that row in the .xlsx copy, or open the CSV through Data > From Text/CSV.", "",
        "The rules (the judge's system prompt, `rubric.md`, without its last line on the output format):", "",
        "```text", body, "```", ""])


def cmd_human(args) -> None:
    for k, p in HUMAN_SHEETS.items():
        if p.exists():
            h = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            if (h.get("label", pd.Series(dtype=str)).str.strip() != "").any():
                raise SystemExit(f"{rel(p)} holds labels; nothing is overwritten")
    sample = read_jsonl(SAMPLE)
    d = human_draw(sample, args.n, args.seed)
    notes = {}
    for k in (1, 2):
        rows = sheet_rows(d, sample, k)
        with open(HUMAN_SHEETS[k], "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(SHEET_COLUMNS)
            w.writerows(rows)
        why = write_xlsx(HUMAN_XLSX[k], rows)
        notes[k] = "written" if why is None else f"not written ({why})"
    d[["hid_rater1", "hid_rater2", "sample_id", "task", "bin", "cell", "stratum", "model_key", "n_tokens",
       "n_cell_sample", "n_cell_draw", "p_phase2", "w_sample", "design_weight"]].to_csv(
        HUMAN_KEY, index=False, encoding="utf-8", lineterminator="\n")
    system, _ = rubric_parts()
    HUMAN_INSTRUCTIONS.write_text(rater_instructions(system), encoding="utf-8")
    smp = {r["sample_id"]: r for r in sample}
    texts = [smp[s][f] for s in d["sample_id"] for f in ("question", "reply")]
    formula = sum(t.lstrip(" ").startswith(FORMULA_START) for t in texts)
    longest = max(len(t) for t in texts)
    out = {"rows": len(d), "cells": d.groupby("cell").size().to_dict(), "seed": args.seed,
           "xlsx": notes, "cells_starting_with_a_formula_character": formula, "longest_cell_chars": longest,
           "over_excel_cell_limit": sum(len(t) > EXCEL_CELL_MAX for t in texts)}
    print(json.dumps(out, indent=1))
    print(f"wrote {rel(HUMAN_SHEETS[1])}, {rel(HUMAN_SHEETS[2])}, {rel(HUMAN_KEY)}, {rel(HUMAN_INSTRUCTIONS)}")


def _sheet_frames(k: int, sheets: dict, xlsx: dict) -> list[tuple[str, pd.DataFrame]]:
    out = []
    p = Path(sheets[k])
    if p.exists():
        out.append((rel(p), pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")))
    px = Path(xlsx[k]) if xlsx and k in xlsx else None
    if px is not None and px.exists():
        try:
            import openpyxl
            ws = openpyxl.load_workbook(px, read_only=True).active
            vals = list(ws.iter_rows(values_only=True))
            df = pd.DataFrame(vals[1:], columns=list(vals[0])).fillna("").astype(str)
            out.append((rel(px), df))
        except ImportError:
            pass
    return out


def read_human(sheets: dict = HUMAN_SHEETS, key: Path = HUMAN_KEY, xlsx: dict = HUMAN_XLSX) -> pd.DataFrame | None:
    """The key's rows with rater1 and rater2 (R/P/A or None if empty) and unsure1, unsure2; None if no label yet. A
    sheet is read from its CSV or its .xlsx, whichever holds labels (both with labels that differ: an error)."""
    if not Path(key).exists():
        return None
    k_ = pd.read_csv(key, dtype=str, keep_default_na=False)
    k_["design_weight"] = k_["design_weight"].astype(float)
    any_label = False
    for k in (1, 2):
        got = {}
        for name, df in _sheet_frames(k, sheets, xlsx):
            lab = df["label"].astype(str).str.strip().str.upper()
            if (lab != "").any():
                got[name] = pd.DataFrame({"hid": df["hid"].astype(str), "lab": lab,
                                          "unsure": df.get("unsure", pd.Series([""] * len(df))).astype(str).str.strip()})
        if len(got) > 1:
            a, b = list(got.values())[:2]
            if not a.set_index("hid")["lab"].equals(b.set_index("hid")["lab"].reindex(a["hid"])):
                raise SystemExit(f"rater {k}: {list(got)} both hold labels and they differ")
        if not got:
            k_[f"rater{k}"], k_[f"unsure{k}"] = None, False
            continue
        any_label = True
        g = list(got.values())[0]
        bad = sorted(set(g.loc[g["lab"] != "", "lab"]) - set(CLASSES))
        if bad:
            raise SystemExit(f"rater {k}: labels other than R, P, A: {bad}")
        m = g.set_index("hid")
        hids = k_[f"hid_rater{k}"]
        if set(hids) - set(m.index):
            raise SystemExit(f"rater {k}: the sheet misses hids of the key")
        lab = m["lab"].reindex(hids).to_numpy()
        k_[f"rater{k}"] = [x if x in CLASSES else None for x in lab]
        k_[f"unsure{k}"] = (m["unsure"].reindex(hids).to_numpy() != "")
    return k_ if any_label else None


# ================================================================================================ analysis


def design(df: pd.DataFrame):
    """Arrays of the rows: stratum codes, multiplicities, stratum sizes M_h (draw-weighted)."""
    codes, uniq = pd.factorize(df["stratum"], sort=True)
    M = df.groupby("stratum")["M_h"].first().reindex(uniq).to_numpy(float)
    return codes, df["mult"].to_numpy(float), M


def weights(codes, mult, M, counts=None) -> np.ndarray:
    """Stratified ratio weights: inside each stratum the sampled rows' multiplicities scaled to sum to M_h."""
    c = np.ones(len(codes)) if counts is None else np.asarray(counts, float)
    s = np.bincount(codes, weights=c * mult, minlength=len(M))
    with np.errstate(invalid="ignore", divide="ignore"):
        W = c * mult * M[codes] / s[codes]
    W[~np.isfinite(W)] = 0.0
    return W


def boot_counts(codes, rng) -> np.ndarray:
    """Within-stratum resampling with replacement: how often each row is drawn."""
    order = np.argsort(codes, kind="mergesort")
    sc = codes[order]
    starts = np.searchsorted(sc, np.arange(codes.max() + 1))
    sizes = np.bincount(codes, minlength=codes.max() + 1)
    pick = starts[sc] + np.floor(rng.random(len(codes)) * sizes[sc]).astype(int)
    return np.bincount(order[pick], minlength=len(codes)).astype(float)


def _share(W, y, mask):
    d = float((W * mask).sum())
    return float((W * y * mask).sum() / d) if d > 0 else None


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _spearman(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def _rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else None


def _mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else None


def _pp_diff(a, b):
    return None if a is None or b is None else 100.0 * (a - b)


def _sub(a, b):
    return None if a is None or b is None else a - b


def task_estimates(df: pd.DataFrame, W: np.ndarray, models, per_model: dict | None) -> dict:
    """Point estimates of one benchmark's labelled panel rows (or Qwen4-Max's) under weights W."""
    lab = df["label"].to_numpy()
    y = (lab == "R").astype(float)
    n = df["n_tokens"].to_numpy()
    out = {}
    for cut in (16, 64):
        long_, short = (n > cut).astype(float), (n <= cut).astype(float)
        tot = float((W * y).sum())
        out[f"r_long_{cut}"] = _share(W, y, long_)
        out[f"r_short_{cut}"] = _share(W, y, short)
        out[f"precision_{cut}"] = out[f"r_long_{cut}"]
        out[f"recall_{cut}"] = float((W * y * long_).sum() / tot) if tot > 0 else None
        out[f"long_share_{cut}"] = _share(W, long_, np.ones(len(W)))
    out["r_all"] = _share(W, y, np.ones(len(W)))
    for b in BINS:
        mb = (df["bin"].to_numpy() == b).astype(float)
        for c in CLASSES:
            out[f"share_{c}_{b}"] = _share(W, (lab == c).astype(float), mb)
    # the mixture assumption on the same items: requested accuracy on replies labelled R and not R, beside the same
    # models' CoT and forced accuracy on the same (configuration, item)
    cols = [f"{a}_{s}" for a in ARMS3 for s in SCORINGS]
    if all(c in df.columns for c in cols) and np.isfinite(df[cols].to_numpy(float)).all():
        notR = 1.0 - y
        for s in SCORINGS:
            for a in ARMS3:
                v = df[f"{a}_{s}"].to_numpy(float)
                out[f"acc_{a}_{s}_R"] = _share(W, v, y)
                out[f"acc_{a}_{s}_notR"] = _share(W, v, notR)
            out[f"gap_req_cot_{s}_R_pp"] = _pp_diff(out[f"acc_requested_{s}_R"], out[f"acc_cot_{s}_R"])
            out[f"gap_req_forced_{s}_notR_pp"] = _pp_diff(out[f"acc_requested_{s}_notR"], out[f"acc_forced_{s}_notR"])
    if per_model is not None:
        mk = df["model_key"].to_numpy()
        rs = np.array([np.nan if v is None else v for v in (_share(W, y, (mk == m).astype(float)) for m in models)])
        w = np.array([per_model["w"][m] for m in models])
        w16 = np.array([per_model["w16"][m] for m in models])
        out["r_share_by_model"] = rs
        out["corr_w_pearson"] = _pearson(rs, w)
        out["corr_w_spearman"] = _spearman(rs, w)
        out["corr_w16_pearson"] = _pearson(rs, w16)
        for sc in SCORINGS:
            d = np.array([per_model[f"req_minus_forced_{sc}"][m] for m in models])
            a, b = _pearson(rs, d), _pearson(w, d)
            out[f"r2_rshare_{sc}"] = None if a is None else a * a
            out[f"r2_w_{sc}"] = None if b is None else b * b
            out[f"r_rshare_{sc}"], out[f"r_w_{sc}"] = a, b
            out[f"r2_diff_{sc}"] = None if (a is None or b is None) else a * a - b * b
            if f"cot_minus_forced_{sc}" in per_model:
                # across models: requested - forced against r x (CoT - forced) and w x (CoT - forced), in pp
                cmf = np.array([per_model[f"cot_minus_forced_{sc}"][m] for m in models])
                res_r, res_w = d - rs * cmf, d - w * cmf
                out[f"mix_res_r_{sc}_by_model"] = res_r
                out[f"mix_rms_r_{sc}"], out[f"mix_rms_w_{sc}"] = _rms(res_r), _rms(res_w)
                out[f"mix_rms_diff_{sc}"] = _sub(out[f"mix_rms_r_{sc}"], out[f"mix_rms_w_{sc}"])
                out[f"mix_mean_res_r_{sc}"], out[f"mix_mean_res_w_{sc}"] = _mean(res_r), _mean(res_w)
                out[f"mix_corr_r_{sc}"], out[f"mix_corr_w_{sc}"] = _pearson(d, rs * cmf), _pearson(d, w * cmf)
    return out


def _ci(vals):
    v = np.array([np.nan if x is None else x for x in vals], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def estimate_block(df: pd.DataFrame, models, per_model, n_boot: int, seed: int) -> dict:
    codes, mult, M = design(df)
    W = weights(codes, mult, M)
    est = task_estimates(df, W, models, per_model)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        c = boot_counts(codes, rng)
        boots.append(task_estimates(df, weights(codes, mult, M, c), models, per_model))
    ci = {}
    for k, v in est.items():
        if k.endswith("_by_model"):
            arr = np.array([b[k] for b in boots], float) if boots else np.zeros((0, len(v)))
            ci[k] = [_ci(arr[:, j]) for j in range(len(v))]
        else:
            ci[k] = _ci([b.get(k) for b in boots])
    est = {k: ([None if not np.isfinite(x) else float(x) for x in v] if isinstance(v, np.ndarray) else v)
           for k, v in est.items()}
    return {"estimate": est, "ci95": ci, "n_labelled": int(len(df)),
            "n_by_bin": df["bin"].value_counts().reindex(BINS, fill_value=0).astype(int).to_dict(),
            "label_counts": df["label"].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()}


def agreement(a, b, w=None) -> dict:
    """Agreement of two label lists over R, P, A: share agreeing, Cohen's kappa (None when the expected agreement is
    1), the 3x3 confusion (rows the first list, columns the second; counts, or weighted shares when w is given), and
    R against not R: precision P(second R | first R), recall P(first R | second R), agreement and kappa."""
    a, b = list(a), list(b)
    wt = np.ones(len(a)) if w is None else np.asarray(w, float)
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((3, 3))
    for x, y, ww in zip(a, b, wt):
        cm[idx[x], idx[y]] += ww
    tot = float(cm.sum())
    if tot <= 0:
        return {"n": len(a), "agreement": None, "expected_agreement": None, "kappa": None, "precision_R": None,
                "recall_R": None, "agreement_R": None, "kappa_R": None, "confusion": None}
    po = float(np.trace(cm) / tot)
    pe = float((cm.sum(1) / tot) @ (cm.sum(0) / tot))
    kappa = None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)
    aR, bR, both = float(cm[0].sum()), float(cm[:, 0].sum()), float(cm[0, 0])
    po2 = (both + (tot - aR - bR + both)) / tot
    pe2 = (aR / tot) * (bR / tot) + (1 - aR / tot) * (1 - bR / tot)
    conf = {CLASSES[i]: {CLASSES[j]: (int(round(cm[i, j])) if w is None else float(cm[i, j] / tot))
                         for j in range(3)} for i in range(3)}
    return {"n": len(a), "agreement": po, "expected_agreement": pe, "kappa": kappa,
            "precision_R": both / aR if aR > 0 else None, "recall_R": both / bR if bR > 0 else None,
            "agreement_R": po2, "kappa_R": None if pe2 >= 1 - 1e-12 else (po2 - pe2) / (1 - pe2),
            "confusion": conf, "confusion_kind": "counts" if w is None else "design-weighted shares"}


def cohen_kappa(a, b) -> dict:
    """agreement() unweighted (kept for callers of the earlier name)."""
    return agreement(a, b)


def pair_block(df: pd.DataFrame, first: str, second: str) -> dict:
    """agreement() of two label columns over the rows where both are R, P or A, design-weighted and unweighted."""
    ok = df[first].isin(CLASSES) & df[second].isin(CLASSES)
    x = df[ok]
    return {"first": first, "second": second, "n": int(ok.sum()), "n_left_out": int((~ok).sum()),
            "weighted": agreement(x[first], x[second], x["design_weight"].to_numpy(float)),
            "unweighted": agreement(x[first], x[second]),
            "by_task_unweighted": {t: agreement(g[first], g[second]) for t, g in x.groupby("task")}}


PAIRS = {"judge_rater1": ("judge", "rater1"), "judge_rater2": ("judge", "rater2"), "rater1_rater2": ("rater1", "rater2")}
PAIR_TAG = {"judge_rater1": "JudgeRaterOne", "judge_rater2": "JudgeRaterTwo", "rater1_rater2": "RaterOneRaterTwo"}


def _kappas(ia: np.ndarray, ib: np.ndarray, w: np.ndarray) -> tuple:
    """(kappa, kappa_R) of two label-index arrays (0, 1, 2 = R, P, A; -1 = no label) under row weights w; rows with
    -1 on either side drop out. None where the expected agreement is 1 or no weight is left."""
    ok = (ia >= 0) & (ib >= 0)
    cm = np.zeros((3, 3))
    np.add.at(cm, (ia[ok], ib[ok]), w[ok])
    tot = cm.sum()
    if tot <= 0:
        return None, None
    po, pe = np.trace(cm) / tot, (cm.sum(1) / tot) @ (cm.sum(0) / tot)
    aR, bR, both = cm[0].sum() / tot, cm[:, 0].sum() / tot, cm[0, 0] / tot
    po2, pe2 = 1 - aR - bR + 2 * both, aR * bR + (1 - aR) * (1 - bR)
    return (None if pe >= 1 - 1e-12 else float((po - pe) / (1 - pe)),
            None if pe2 >= 1 - 1e-12 else float((po2 - pe2) / (1 - pe2)))


GAP_TAG = {"judge_rater1": "JudgeRaterOne", "judge_rater2": "JudgeRaterTwo", "judge_mean": "JudgeMean"}


def human_boot(h: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap of the human check: the rows resampled with replacement inside each benchmark x length cell (the
    check's strata; 'task' when there is no 'cell' column), n_boot draws, 95% percentile intervals of each pair's
    kappa and kappa_R, design-weighted and unweighted, and of the gaps kappa(judge, rater k) - kappa(rater 1,
    rater 2) and their mean over k, on the same draws. Draws where a kappa is undefined are left out of that
    statistic and counted. Descriptive only: no pass rule is applied."""
    codes, _ = pd.factorize(h["cell"] if "cell" in h.columns else h["task"], sort=True)
    codes = np.asarray(codes)
    idx = {c: i for i, c in enumerate(CLASSES)}
    lab = {c: np.array([idx.get(x, -1) for x in h[c]]) for c in ("judge", "rater1", "rater2")}
    dw = h["design_weight"].to_numpy(float)
    rng = np.random.default_rng([seed, zlib.crc32(b"human_boot")])
    stats = {}
    for _ in range(n_boot):
        cnt = boot_counts(codes, rng)
        for wk, w in (("weighted", cnt * dw), ("unweighted", cnt)):
            k = {p: _kappas(lab[a], lab[b], w) for p, (a, b) in PAIRS.items()}
            for p, (kap, kapR) in k.items():
                stats.setdefault((wk, "kappa", p), []).append(kap)
                stats.setdefault((wk, "kappa_R", p), []).append(kapR)
            for kk, i in (("kappa", 0), ("kappa_R", 1)):
                rr = k["rater1_rater2"][i]
                g1 = None if rr is None or k["judge_rater1"][i] is None else k["judge_rater1"][i] - rr
                g2 = None if rr is None or k["judge_rater2"][i] is None else k["judge_rater2"][i] - rr
                gm = None if g1 is None or g2 is None else (g1 + g2) / 2
                for g, v in (("judge_rater1", g1), ("judge_rater2", g2), ("judge_mean", gm)):
                    stats.setdefault((wk, f"gap_{kk}", g), []).append(v)
    out = {"n_boot": n_boot, "seed": seed, "strata": "cell" if "cell" in h.columns else "task", "ci95": {}}
    for (wk, stat, p), vals in stats.items():
        v = np.array([x for x in vals if x is not None], float)
        ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
        out["ci95"].setdefault(wk, {}).setdefault(stat, {})[p] = {"ci": ci, "undefined_draws": len(vals) - len(v)}
    return out


def _gap_points(pairs: dict) -> dict:
    """Point gaps kappa(judge, rater k) - kappa(rater 1, rater 2), k = 1, 2, and their mean, for kappa and kappa_R,
    weighted and unweighted (None where a kappa is undefined)."""
    out = {}
    for wk in ("weighted", "unweighted"):
        for kk in ("kappa", "kappa_R"):
            rr, a, b = (pairs[p][wk][kk] for p in ("rater1_rater2", "judge_rater1", "judge_rater2"))
            g1 = None if rr is None or a is None else a - rr
            g2 = None if rr is None or b is None else b - rr
            out.setdefault(wk, {})[f"gap_{kk}"] = {"judge_rater1": g1, "judge_rater2": g2,
                                                   "judge_mean": None if g1 is None or g2 is None else (g1 + g2) / 2}
    return out


def human_block(human: pd.DataFrame, labels: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    h = human.merge(labels[["sample_id", "label"]].rename(columns={"label": "judge"}), on="sample_id", how="left")
    out = {"n_rows": int(len(h)), "labelled_rater1": int(h["rater1"].isin(CLASSES).sum()),
           "labelled_rater2": int(h["rater2"].isin(CLASSES).sum()), "judged": int(h["judge"].isin(CLASSES).sum()),
           "unsure_rater1": int(h["unsure1"].sum()), "unsure_rater2": int(h["unsure2"].sum()),
           "pairs": {k: pair_block(h, a, b) for k, (a, b) in PAIRS.items()}}
    sure = h[~h["unsure1"].astype(bool) & ~h["unsure2"].astype(bool)]
    out["pairs_without_unsure_unweighted"] = {k: pair_block(sure, a, b)["unweighted"] for k, (a, b) in PAIRS.items()}
    if all(out["pairs"][p]["n"] > 0 for p in PAIRS):
        out["gaps"] = _gap_points(out["pairs"])
        out["boot"] = human_boot(h, n_boot, seed)
    return out


def repeat_block(rep: pd.DataFrame, labels: pd.DataFrame, rep_labels: pd.DataFrame) -> dict:
    """The judge against itself on the repeat set: its label in the sample run (first) and in the repeat (second)."""
    d = rep[["sample_id", "task", "design_weight"]].merge(
        labels[["sample_id", "label"]].rename(columns={"label": "first"}), on="sample_id", how="left").merge(
        rep_labels[["sample_id", "label"]].rename(columns={"label": "repeat"}), on="sample_id", how="left")
    out = pair_block(d, "first", "repeat")
    out["n_set"] = int(len(d))
    out["repeat_judged"] = int(d["sample_id"].isin(set(rep_labels["sample_id"])).sum())
    return out


def analyse(sample: pd.DataFrame, labels: pd.DataFrame, population: dict, human: pd.DataFrame | None = None,
            n_boot: int = N_BOOT, seed: int = SEED, keys: pd.DataFrame | None = None,
            repeat: tuple | None = None) -> dict:
    """The mixture check, the human check and the repeat set. sample: the rows of
    sample.jsonl (design columns); labels: sample_id, label (the sample run); population: population.json's
    'population' block; keys: sample_key.jsonl's scores; repeat: (the repeat draw, the repeat labels)."""
    if keys is not None:
        sc = [c for c in keys.columns if c != "gold"]
        sample = sample.merge(keys[sc], on="sample_id", how="left")
    d = sample.merge(labels[["sample_id", "label"]], on="sample_id", how="left")
    judged = d["sample_id"].isin(set(labels["sample_id"]))
    unread = d[judged & ~d["label"].isin(CLASSES)]
    lab = d[d["label"].isin(CLASSES)].copy()
    R = {"n_sample": int(len(d)), "n_labelled": int(len(lab)), "n_unread": int(len(unread)),
         "n_not_judged": int((~judged).sum()),
         "unread_by_cell": unread.groupby(["source", "task", "bin"]).size().rename("n").reset_index()
                                 .to_dict("records"),
         "empty_strata": [], "panel": {}, "qwen": {}}
    for task in TASKS:
        g = lab[(lab["source"] == "panel") & (lab["task"] == task)].reset_index(drop=True)
        if g.empty:
            continue
        P = population[task]
        models = list(P["models"])
        pm = P["per_model"]
        all_strata = sample[(sample["source"] == "panel") & (sample["task"] == task)]["stratum"].unique()
        R["empty_strata"] += sorted(set(all_strata) - set(g["stratum"]))
        blk = estimate_block(g, models, pm, n_boot, seed + TASKS.index(task))
        blk["models"] = models
        e, c = blk["estimate"], blk["ci95"]
        rows = []
        for j, m in enumerate(models):
            row = {"model_key": m, "r_share": e["r_share_by_model"][j], "r_share_ci95": c["r_share_by_model"][j],
                   "w": pm["w"][m], "w16": pm["w16"][m], "w64": pm["w64"][m],
                   "n_labelled": int((g["model_key"] == m).sum())}
            for s in SCORINGS:
                row[f"req_minus_forced_{s}"] = pm[f"req_minus_forced_{s}"][m]
                if f"cot_minus_forced_{s}" in pm:
                    cmf, r_ = pm[f"cot_minus_forced_{s}"][m], e["r_share_by_model"][j]
                    row[f"cot_minus_forced_{s}"] = cmf
                    row[f"pred_r_{s}"] = None if r_ is None else r_ * cmf
                    row[f"pred_w_{s}"] = pm["w"][m] * cmf
                    row[f"res_r_{s}"] = e[f"mix_res_r_{s}_by_model"][j]
                    row[f"res_r_{s}_ci95"] = c[f"mix_res_r_{s}_by_model"][j]
                    row[f"res_w_{s}"] = row[f"req_minus_forced_{s}"] - row[f"pred_w_{s}"]
            rows.append(row)
        blk["per_model"] = rows
        blk["w_cap"] = P["cap"]
        R["panel"][task] = blk
    q = lab[lab["source"] == "qwen"].reset_index(drop=True)
    for task in QWEN_TASKS:
        g = q[q["task"] == task].reset_index(drop=True)
        if len(g):
            R["qwen"][task] = estimate_block(g, [QWEN], None, n_boot, seed + 10 + QWEN_TASKS.index(task))
    if len(q):
        R["qwen"]["pooled"] = estimate_block(q, [QWEN], None, n_boot, seed + 20)
    if human is not None:
        R["human"] = human_block(human, labels, n_boot, seed + 30)
    if repeat is not None:
        R["repeat"] = repeat_block(repeat[0], labels, repeat[1])
    return R


# ---------------------------------------------------------------------------------- macros, table, summary


def pct0(x):
    return "--" if x is None else f"{100 * x:.0f}"


def _num(x, d):
    if x is None:
        return "--"
    if abs(x) < 0.5 * 10 ** (-d):
        x = 0.0
    return f"{x:.{d}f}".replace("-", "$-$")


def f2(x):
    return _num(x, 2)


def f1(x):
    return _num(x, 1)


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def macros(R: dict, meta: dict) -> dict:
    """{name: value}, prefix lab, letters only. Shares and accuracies in whole percent; differences and residuals in
    pp to one decimal; correlations, R^2 and kappa to two decimals."""
    m = {"labJudge": meta["judge_label"], "labJudgeId": meta["judge_id"], "labJudgeMaxTokens": str(MAX_TOKENS),
         "labJudgeEffort": OUTPUT_CONFIG["effort"], "labSeed": str(meta["seed"]), "labN": comma(R["n_sample"]),
         "labNLabelled": comma(R["n_labelled"]), "labNUnread": comma(R["n_unread"]),
         "labNNotJudged": comma(R["n_not_judged"]),
         "labNPanel": comma(sum(b["n_labelled"] for b in R["panel"].values())),
         "labNQwen": comma(R["qwen"]["pooled"]["n_labelled"]) if "pooled" in R["qwen"] else "0"}
    if R.get("cost"):
        m["labCostUsd"] = f"{R['cost']['billed_usd']:.2f}"
    for task, b in R["panel"].items():
        T, e, c = TASK_TAG[task], b["estimate"], b["ci95"]
        m[f"labN{T}"] = comma(b["n_labelled"])
        m[f"labModels{T}"] = str(len(b["models"]))
        for name, k in (("RLong", "r_long_16"), ("RShort", "r_short_16"), ("Prec", "precision_16"),
                        ("Recall", "recall_16"), ("RAll", "r_all")):
            m[f"lab{name}{T}Pct"] = pct0(e[k])
            if c.get(k):
                m[f"lab{name}{T}Lo"], m[f"lab{name}{T}Hi"] = pct0(c[k][0]), pct0(c[k][1])
        if task == "hotpot":
            for name, k in (("RLong", "r_long_64"), ("RShort", "r_short_64"), ("Prec", "precision_64"),
                            ("Recall", "recall_64")):
                m[f"lab{name}{T}SixtyFourPct"] = pct0(e[k])
                if c.get(k):
                    m[f"lab{name}{T}SixtyFourLo"], m[f"lab{name}{T}SixtyFourHi"] = pct0(c[k][0]), pct0(c[k][1])
        m[f"labCorrW{T}"] = f2(e["corr_w_pearson"])
        m[f"labSpearW{T}"] = f2(e["corr_w_spearman"])
        if c.get("corr_w_pearson"):
            m[f"labCorrW{T}Lo"], m[f"labCorrW{T}Hi"] = f2(c["corr_w_pearson"][0]), f2(c["corr_w_pearson"][1])
        for sc, S in (("strict", ""), ("content", "Content")):
            m[f"labRsqR{T}{S}"] = f2(e[f"r2_rshare_{sc}"])
            m[f"labRsqW{T}{S}"] = f2(e[f"r2_w_{sc}"])
            if c.get(f"r2_diff_{sc}"):
                m[f"labRsqDiff{T}{S}Lo"] = f2(c[f"r2_diff_{sc}"][0])
                m[f"labRsqDiff{T}{S}Hi"] = f2(c[f"r2_diff_{sc}"][1])
            if f"acc_requested_{sc}_R" in e:
                for arm, A in (("requested", "Req"), ("cot", "Cot"), ("forced", "Forced")):
                    m[f"labAcc{A}R{T}{S}Pct"] = pct0(e[f"acc_{arm}_{sc}_R"])
                    m[f"labAcc{A}NotR{T}{S}Pct"] = pct0(e[f"acc_{arm}_{sc}_notR"])
                for name, k in (("GapReqCotR", f"gap_req_cot_{sc}_R_pp"),
                                ("GapReqForcedNotR", f"gap_req_forced_{sc}_notR_pp")):
                    m[f"lab{name}{T}{S}"] = f1(e[k])
                    if c.get(k):
                        m[f"lab{name}{T}{S}Lo"], m[f"lab{name}{T}{S}Hi"] = f1(c[k][0]), f1(c[k][1])
            if f"mix_rms_r_{sc}" in e:
                m[f"labMixRmsR{T}{S}"] = f1(e[f"mix_rms_r_{sc}"])
                m[f"labMixRmsW{T}{S}"] = f1(e[f"mix_rms_w_{sc}"])
                m[f"labMixMeanResR{T}{S}"] = f1(e[f"mix_mean_res_r_{sc}"])
                m[f"labMixMeanResW{T}{S}"] = f1(e[f"mix_mean_res_w_{sc}"])
                m[f"labMixCorrR{T}{S}"] = f2(e[f"mix_corr_r_{sc}"])
                m[f"labMixCorrW{T}{S}"] = f2(e[f"mix_corr_w_{sc}"])
                if c.get(f"mix_rms_diff_{sc}"):
                    m[f"labMixRmsDiff{T}{S}Lo"] = f1(c[f"mix_rms_diff_{sc}"][0])
                    m[f"labMixRmsDiff{T}{S}Hi"] = f1(c[f"mix_rms_diff_{sc}"][1])
    for task, b in R["qwen"].items():
        T = "" if task == "pooled" else TASK_TAG[task]
        e, c = b["estimate"], b["ci95"]
        m[f"labQwenRLong{T}Pct"] = pct0(e["r_long_16"])
        if c.get("r_long_16"):
            m[f"labQwenRLong{T}Lo"], m[f"labQwenRLong{T}Hi"] = pct0(c["r_long_16"][0]), pct0(c["r_long_16"][1])
        m[f"labQwenNLong{T}"] = str(b["n_by_bin"]["17-64"] + b["n_by_bin"]["65+"])
    if "human" in R:
        h = R["human"]
        m["labHumanN"] = str(h["n_rows"])
        m["labUnsureRaterOne"], m["labUnsureRaterTwo"] = str(h["unsure_rater1"]), str(h["unsure_rater2"])
        for k, P in PAIR_TAG.items():
            p = h["pairs"][k]
            if p["n"] == 0:                 # a rater with no label yet: no macros for the pair
                continue
            w, u = p["weighted"], p["unweighted"]
            m[f"labN{P}"] = str(p["n"])
            m[f"labAgree{P}Pct"], m[f"labAgree{P}UnwPct"] = pct0(w["agreement"]), pct0(u["agreement"])
            m[f"labKappa{P}"], m[f"labKappa{P}Unw"] = f2(w["kappa"]), f2(u["kappa"])
            m[f"labPrecR{P}Pct"], m[f"labRecallR{P}Pct"] = pct0(w["precision_R"]), pct0(w["recall_R"])
            m[f"labKappaR{P}"] = f2(w["kappa_R"])
        if "boot" in h:
            ci = h["boot"]["ci95"]
            for k, P in PAIR_TAG.items():
                for wk, stat, name in (("weighted", "kappa", f"labKappa{P}"), ("unweighted", "kappa", f"labKappa{P}Unw"),
                                       ("weighted", "kappa_R", f"labKappaR{P}")):
                    c = ci[wk][stat][k]["ci"]
                    if c:
                        m[f"{name}Lo"], m[f"{name}Hi"] = f2(c[0]), f2(c[1])
            for g, G in GAP_TAG.items():
                for wk, stat, name in (("weighted", "gap_kappa", f"labKappaGap{G}"),
                                       ("unweighted", "gap_kappa", f"labKappaGap{G}Unw"),
                                       ("weighted", "gap_kappa_R", f"labKappaRGap{G}")):
                    v = h["gaps"][wk][stat][g]
                    if v is not None:
                        m[name] = f2(v)
                    c = ci[wk][stat][g]["ci"]
                    if c:
                        m[f"{name}Lo"], m[f"{name}Hi"] = f2(c[0]), f2(c[1])
    if "repeat" in R and R["repeat"]["n"] > 0:
        p = R["repeat"]
        m["labSelfN"] = str(p["n"])
        m["labSelfAgreePct"], m["labSelfAgreeUnwPct"] = pct0(p["weighted"]["agreement"]), pct0(p["unweighted"]["agreement"])
        m["labSelfKappa"], m["labSelfKappaUnw"] = f2(p["weighted"]["kappa"]), f2(p["unweighted"]["kappa"])
    for k in m:
        if not re.fullmatch(r"lab[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r}")
    return m


def si_table(R: dict) -> str:
    def cell(e, c, k):
        v = pct0(e.get(k))
        return v if not c.get(k) else f"{v} [{pct0(c[k][0])}, {pct0(c[k][1])}]"

    def gap(e, c, k):
        if k not in e:
            return ""
        v = f1(e[k])
        return v if not c.get(k) else f"{v} [{f1(c[k][0])}, {f1(c[k][1])}]"
    L = ["% Generated by scripts/reading_audit.py analyse from results/reading_audit/reading_audit.json",
         "% R share: design-weighted share of requested replies labelled R, short (at most Cut tokens) and long (over",
         "% Cut), 95% interval; Recall: share of R replies that are long; r: Pearson across models of the R share with",
         "% w; R^2: requested - forced accuracy (strict) across models on the R share and on w; Req - CoT, R: requested",
         "% minus CoT accuracy (strict, pp) on the replies labelled R, same items; Req - forced, not R: requested minus",
         "% forced accuracy (strict, pp) on the replies not labelled R, same items.",
         "\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule",
         "Benchmark & $n$ & Cut & R, short (\\%) & R, long (\\%) & Recall (\\%) & $r(\\text{R}, w)$ & "
         "$R^2$ R & $R^2$ $w$ & Req.$-$CoT, R & Req.$-$forced, not R \\\\", "\\midrule"]
    for task, b in R["panel"].items():
        e, c = b["estimate"], b["ci95"]
        L.append(f"{TASK_NAME[task]} & {b['n_labelled']} & 16 & {cell(e, c, 'r_short_16')} & "
                 f"{cell(e, c, 'r_long_16')} & {pct0(e['recall_16'])} & {f2(e['corr_w_pearson'])} & "
                 f"{f2(e['r2_rshare_strict'])} & {f2(e['r2_w_strict'])} & {gap(e, c, 'gap_req_cot_strict_R_pp')} & "
                 f"{gap(e, c, 'gap_req_forced_strict_notR_pp')} \\\\")
        if task == "hotpot":
            L.append(f" & & 64 & {cell(e, c, 'r_short_64')} & {cell(e, c, 'r_long_64')} & "
                     f"{pct0(e['recall_64'])} & & & & & \\\\")
    for task in QWEN_TASKS:
        if task in R["qwen"]:
            b = R["qwen"][task]
            e, c = b["estimate"], b["ci95"]
            L.append(f"Qwen4-Max, {TASK_NAME[task]} & {b['n_labelled']} & 16 & {cell(e, c, 'r_short_16')} & "
                     f"{cell(e, c, 'r_long_16')} & {pct0(e['recall_16'])} & & & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def _agree_line(name: str, p: dict) -> str:
    if p["n"] == 0:
        return f"- {name}: no rows with both labels yet."
    w, u = p["weighted"], p["unweighted"]
    return (f"- {name}: n = {p['n']} ({p['n_left_out']} left out: a label missing or unread); agreement "
            f"{pct0(w['agreement'])}% weighted ({pct0(u['agreement'])}% unweighted), Cohen's kappa {f2(w['kappa'])} "
            f"({f2(u['kappa'])}); R against not R: precision {pct0(w['precision_R'])}%, recall {pct0(w['recall_R'])}%, "
            f"kappa {f2(w['kappa_R'])} (unweighted {pct0(u['precision_R'])}%, {pct0(u['recall_R'])}%, "
            f"{f2(u['kappa_R'])}). Confusion (rows {p['first']}, columns {p['second']}, counts): "
            f"{json.dumps(u['confusion'])}.").replace("$-$", "-")


def summary_md(R: dict, meta: dict) -> str:
    L = ["# Reading audit of requested answers", "",
         f"Judge {meta['judge_label']} (`{meta['judge_id']}`, Anthropic Message Batches API; thinking disabled, effort "
         f"{OUTPUT_CONFIG['effort']}, max_tokens {MAX_TOKENS}; {SAMPLING}), rubric `results/reading_audit/rubric.md`. "
         f"Sample: {R['n_sample']} replies, {R['n_labelled']} labelled, {R['n_unread']} replies not read as one letter "
         f"(left out, never guessed), {R['n_not_judged']} not judged. "
         "Shares are design-weighted estimates of the discovery-draw population (each configuration counted as often "
         "as it is drawn, as in w); 95% intervals from 2,000 bootstrap draws within strata. R2 is the squared Pearson "
         "correlation across models of requested minus forced accuracy (discovery means, pp) with the R share or with "
         "w; the R share is a sample estimate and w a census, so the comparison favours w by the R share's noise.", ""]
    if R.get("cost"):
        L += [f"Billed: ${R['cost']['billed_usd']:.2f} (pilot ${R['cost']['pilot_billed_usd']:.2f} at standard prices, "
              f"batches ${R['cost']['batch_billed_usd']:.2f} at batch prices, from usage).", ""]
    L.append("| Benchmark | n | R share, 1-16 tokens | R share, over 16 | Recall of the 16 cutoff | r(R share, w) | "
             "R2 R share (strict / content) | R2 w (strict / content) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for task, b in R["panel"].items():
        e, c = b["estimate"], b["ci95"]

        def s(k):
            return pct0(e[k]) + (f" [{pct0(c[k][0])}, {pct0(c[k][1])}]" if c.get(k) else "")
        L.append(f"| {TASK_NAME[task]} | {b['n_labelled']} | {s('r_short_16')} | {s('r_long_16')} | {s('recall_16')} | "
                 f"{f2(e['corr_w_pearson'])} | {f2(e['r2_rshare_strict'])} / {f2(e['r2_rshare_content'])} | "
                 f"{f2(e['r2_w_strict'])} / {f2(e['r2_w_content'])} |".replace("$-$", "-"))
        if task == "hotpot":
            L.append(f"| HotpotQA at 64 | | {s('r_short_64')} (1-64) | {s('r_long_64')} (over 64) | "
                     f"{s('recall_64')} | | | |")
    L += ["", "Class shares by length bin (%, R / P / A):", ""]
    for task, b in list(R["panel"].items()) + [(f"qwen {t}", b) for t, b in R["qwen"].items()]:
        e = b["estimate"]
        parts = [f"{bn}: " + " / ".join(pct0(e[f'share_{c}_{bn}']) for c in CLASSES) for bn in BINS]
        L.append(f"- {TASK_NAME.get(task, task)}: " + "; ".join(parts) + f" (labelled per bin {b['n_by_bin']})")
    mix = [(t, b) for t, b in R["panel"].items() if "acc_requested_strict_R" in b["estimate"]]
    if mix:
        L += ["", "The mixture assumption on the same items (design-weighted accuracy, %, strict / content): requested, "
                  "CoT and forced accuracy of the same models on the same (configuration, item), on the replies "
                  "labelled R and on those not labelled R; then requested minus CoT on R and requested minus forced on "
                  "not R (pp, 95% interval).", "",
              "| Benchmark | R: requested | R: CoT | R: forced | not R: requested | not R: CoT | not R: forced | "
              "Req. - CoT, R (pp) | Req. - forced, not R (pp) |", "|---|---|---|---|---|---|---|---|---|"]
        for task, b in mix:
            e, c = b["estimate"], b["ci95"]

            def a2(arm, part):
                return f"{pct0(e[f'acc_{arm}_strict_{part}'])} / {pct0(e[f'acc_{arm}_content_{part}'])}"

            def g2(k):
                return " / ".join(f1(e[k.format(s)]) + (f" [{f1(c[k.format(s)][0])}, {f1(c[k.format(s)][1])}]"
                                                         if c.get(k.format(s)) else "") for s in SCORINGS)
            L.append((f"| {TASK_NAME[task]} | {a2('requested', 'R')} | {a2('cot', 'R')} | {a2('forced', 'R')} | "
                      f"{a2('requested', 'notR')} | {a2('cot', 'notR')} | {a2('forced', 'notR')} | "
                      f"{g2('gap_req_cot_{}_R_pp')} | {g2('gap_req_forced_{}_notR_pp')} |").replace("$-$", "-"))
        L += ["", "Across models (discovery means, pp): requested minus forced against r x (CoT minus forced) and "
                  "w x (CoT minus forced), r the model's R share in the sample and w its census share over the cap. "
                  "Residual = observed minus predicted (per model in per_model.csv).", "",
              "| Benchmark | Scoring | RMS residual, r | RMS residual, w | RMS r - RMS w (95%) | mean residual r / w | "
              "Pearson obs. with pred. r / w |", "|---|---|---|---|---|---|---|"]
        for task, b in mix:
            e, c = b["estimate"], b["ci95"]
            for s in SCORINGS:
                if f"mix_rms_r_{s}" not in e:
                    continue
                ci = c.get(f"mix_rms_diff_{s}")
                L.append((f"| {TASK_NAME[task]} | {s} | {f1(e[f'mix_rms_r_{s}'])} | {f1(e[f'mix_rms_w_{s}'])} | "
                          f"{f1(e[f'mix_rms_diff_{s}'])}" + (f" [{f1(ci[0])}, {f1(ci[1])}]" if ci else "") +
                          f" | {f1(e[f'mix_mean_res_r_{s}'])} / {f1(e[f'mix_mean_res_w_{s}'])} | "
                          f"{f2(e[f'mix_corr_r_{s}'])} / {f2(e[f'mix_corr_w_{s}'])} |").replace("$-$", "-"))
    if R["qwen"]:
        L += ["", "Qwen4-Max (every template once): share of long replies (over 16 tokens) labelled R:"]
        for t, b in R["qwen"].items():
            e, c = b["estimate"], b["ci95"]
            L.append(f"- {TASK_NAME.get(t, t)}: {pct0(e['r_long_16'])}%"
                     + (f" [{pct0(c['r_long_16'][0])}, {pct0(c['r_long_16'][1])}]" if c.get("r_long_16") else "")
                     + f", n long = {b['n_by_bin']['17-64'] + b['n_by_bin']['65+']}")
    if "human" in R:
        h = R["human"]
        L += ["", f"Human check: {h['n_rows']} panel rows drawn before judging, stratified by benchmark x length bin "
                  f"(weights: the row's sample weight over its chance of entering the check); rater 1 labelled "
                  f"{h['labelled_rater1']}, rater 2 {h['labelled_rater2']}, the judge read {h['judged']}; marked "
                  f"unsure: rater 1 {h['unsure_rater1']}, rater 2 {h['unsure_rater2']}.", ""]
        names = {"judge_rater1": "Judge vs rater 1", "judge_rater2": "Judge vs rater 2",
                 "rater1_rater2": "Rater 1 vs rater 2"}
        for k, nm in names.items():
            L.append(_agree_line(nm, h["pairs"][k]))
        if "boot" in h:
            ci, b = h["boot"]["ci95"], h["boot"]

            def iv(wk, stat, p):
                c = ci[wk][stat][p]["ci"]
                return "[--]" if not c else f"[{f2(c[0])}, {f2(c[1])}]"
            L += ["", f"Bootstrap intervals (95%, {b['n_boot']} draws, rows resampled within each {b['strata']}; "
                      "draws with an undefined kappa left out): kappa weighted / unweighted; kappa R against not R, "
                      "weighted.", ""]
            for k, nm in names.items():
                p = h["pairs"][k]
                L.append(f"- {nm}: kappa {f2(p['weighted']['kappa'])} {iv('weighted', 'kappa', k)} / "
                         f"{f2(p['unweighted']['kappa'])} {iv('unweighted', 'kappa', k)}; kappa R "
                         f"{f2(p['weighted']['kappa_R'])} {iv('weighted', 'kappa_R', k)}".replace("$-$", "-"))
            L += ["", "Gap: kappa(judge, rater k) minus kappa(rater 1, rater 2), same bootstrap draws (negative: the "
                      "judge agrees with a person less than the two people agree with each other). No pass rule is "
                      "applied here.", ""]
            gn = {"judge_rater1": "Rater 1", "judge_rater2": "Rater 2", "judge_mean": "Mean of the two"}
            for g, nm in gn.items():
                gw, gu = h["gaps"]["weighted"], h["gaps"]["unweighted"]
                L.append(f"- {nm}: kappa {f2(gw['gap_kappa'][g])} {iv('weighted', 'gap_kappa', g)} / "
                         f"{f2(gu['gap_kappa'][g])} {iv('unweighted', 'gap_kappa', g)}; kappa R "
                         f"{f2(gw['gap_kappa_R'][g])} {iv('weighted', 'gap_kappa_R', g)}".replace("$-$", "-"))
    else:
        L += ["", "Human check: no human labels yet."]
    if "repeat" in R:
        p = R["repeat"]
        L += ["", f"The judge against itself: {p['n_set']} panel rows sent a second time in the batch (repeat judged "
                  f"{p['repeat_judged']}).", "", _agree_line("Sample run vs repeat", p)]
    if R["empty_strata"]:
        L += ["", f"Strata with no labelled row (their model's estimate misses that bin): {R['empty_strata']}"]
    return "\n".join(L) + "\n"


def cmd_analyse(args) -> None:
    system, template = rubric_parts()
    meta_j = judge_meta(system, template)
    full = read_jsonl(SAMPLE)
    sample = pd.DataFrame(full).drop(columns=["question", "reply"])
    labels = judge_labels(meta_j["config_sha"], JUDGE_RAW)
    if labels["label"].notna().sum() == 0:
        raise SystemExit("no judge labels yet: run `judge --backend anthropic --collect` first (nothing written)")
    pop = json.load(open(POPULATION, encoding="utf-8"))["population"]
    keys = pd.DataFrame(read_jsonl(KEYFILE))
    rep = cell_draw(panel_frame(full), REPEAT_N, SEED, "repeat")
    rep_labels = judge_labels(meta_j["config_sha"], JUDGE_RAW, kind="repeat")
    R = analyse(sample, labels, pop, read_human(), args.n_boot, args.seed, keys=keys,
                repeat=(rep, rep_labels) if len(rep_labels) else None)
    R["cost"] = ledger(read_raw(JUDGE_RAW))
    meta = {"judge_label": JUDGE["label"], "judge_id": JUDGE["id"],
            "seed": json.load(open(MANIFEST, encoding="utf-8"))["seed"]}
    R["macros"] = macros(R, meta)
    R["meta"] = {**meta, **meta_j, "api": "Anthropic Message Batches API", "anthropic_version": ANTHROPIC_VERSION,
                 "thinking": THINKING, "output_config": OUTPUT_CONFIG, "max_tokens": MAX_TOKENS, "sampling": SAMPLING,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": args.n_boot}
    write_json(RESULT, R)
    md = summary_md(R, meta)
    SUMMARY.write_text(md, encoding="utf-8")
    CLEAN.write_text(clean_md(md), encoding="utf-8")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(si_table(R), encoding="utf-8")
    pd.DataFrame([{"task": t, **{k: (v if not isinstance(v, list) else json.dumps(v)) for k, v in r.items()}}
                  for t, b in R["panel"].items() for r in b["per_model"]]).to_csv(
        OUT / "per_model.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(md)
    print(f"wrote {rel(RESULT)}, {rel(SUMMARY)}, {rel(CLEAN)}, {rel(TABLE)}; {len(R['macros'])} macros")


def cmd_population(args) -> None:
    """population.json's per-model block recomputed from the scored tables (adds CoT - forced); the fields already in
    the file must agree to 1e-9; the sample is not redrawn."""
    old = json.load(open(POPULATION, encoding="utf-8"))
    for task in TASKS:
        wide, panel, disc = panel_wide(task)
        summ = model_summaries(wide, panel, disc, CAP[task])
        prev = old["population"][task]["per_model"]
        for k, v in prev.items():
            dmax = max(abs(summ[k][m] - v[m]) for m in v)
            if dmax > 1e-9:
                raise SystemExit(f"{task} {k}: recomputed values differ from the saved ones by {dmax}")
        old["population"][task]["per_model"] = summ
        old["checks"][task] = saved_checks(task, summ)
        print(f"{task}: {sorted(summ)}; checks {old['checks'][task]}", flush=True)
    write_json(POPULATION, old)
    print(f"wrote {rel(POPULATION)}")


# ================================================================================================ main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("design", help="rewrite design.md from the saved sample files")
    sub.add_parser("population", help="recompute population.json's per-model block (adds CoT - forced)")
    j = sub.add_parser("judge")
    j.add_argument("--backend", choices=["anthropic"], default="anthropic")
    g = j.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--count-tokens", action="store_true", help="free: exact input counts (count_tokens)")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--collect", action="store_true")
    g.add_argument("--list-batches", action="store_true")
    j.add_argument("--approved", action="store_true")
    j.add_argument("--limit-usd", type=float)
    j.add_argument("--split", action="store_true", help="with --submit: send the part whose worst case fits")
    j.add_argument("--once", action="store_true", help="with --collect: look once, do not wait")
    j.add_argument("--poll-seconds", type=float, default=60.0)
    j.add_argument("--max-wait-hours", type=float, default=26.0)
    j.add_argument("--batch-id")
    j.add_argument("--threads", type=int, default=THREADS)
    j.add_argument("--tiktoken-path")
    j.add_argument("--tiktoken-cache")
    h = sub.add_parser("human")
    h.add_argument("--n", type=int, default=HUMAN_N)
    h.add_argument("--seed", type=int, default=SEED)
    a = sub.add_parser("analyse")
    a.add_argument("--n-boot", type=int, default=N_BOOT)
    a.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    {"sample": cmd_sample, "judge": cmd_judge, "human": cmd_human, "analyse": cmd_analyse,
     "population": cmd_population, "design": lambda a: print(write_design_md())}[args.cmd](args)


if __name__ == "__main__":
    main()
