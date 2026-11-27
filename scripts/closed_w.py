r"""Qwen4-Max on OpenRouter, the panel's three arms (requested, forced, chain of thought) on the same cells.
GPT-6 Luna is not in the defaults (OpenRouter's OpenAI endpoint does not list temperature:
results/frontier_panel/closed_w_pilot.json); its code path is kept.

Design (config/closed_w.yaml):
  Model     qwen/qwen4-max (reasoning off, presence penalty 0) on OpenRouter's chat completions; the provider pinned to
            Alibaba (provider.only, allow_fallbacks false, require_parameters true); temperature 0 and top_p 1.
  Arms      on every template of the four benchmarks (131 benchmark-template cells) and the same 100 items per
            benchmark (the B100 draws of scripts/closed_w_items.py; BIG-Bench Hard 4 per task), 13,100 calls per arm:
            requested  the user message of data/requests/<task>/qwen3-8b.parquet (direct_answer) as the only message,
                       byte for byte; cap 1,024; no stop string;
            forced     the panel's forced request (data/requests/forced/<task>/, HotpotQA data/requests/hotpot/block_a
                       and block_b; forced_direct: the requested arm's user message unchanged) and an assistant message
                       "\boxed{" with partial: true (the variant that passed results/frontier_panel/qwen_prefill_test.md,
                       40 of 40); caps 16, HotpotQA 64; stop "}" on GSM8K and BIG-Bench Hard only, where the panel's
                       generation stopped (config/decoding_forced.yaml; MATH-500 and HotpotQA ran without it);
            cot        the panel's zero-shot chain-of-thought request (zero_shot_cot of the same files as requested);
                       caps 3,072, HotpotQA 1,900; no stop string.
            Items, seeds, the pilot, prefill, control and stop-control rows and each arm's request-list hash are in
            manifests/closed_w_items.json, written by `plan` before any call.
  Recorded  per call: the raw text, refusal text, finish reason (OpenRouter's and the provider's), usage (prompt,
            completion, reasoning and cached tokens; the cost OpenRouter bills), served model and provider,
            OpenRouter's generation id, this script's attempt id, send and receive times. Lines are appended to
            data/frontier_panel/closed_w/<phase>_raw.jsonl keyed model|arm|benchmark|request id; a restart skips keys
            already answered under the same configuration hash.
  Scored    with the panel's scorers (scripts/nebius_panel_report.score_task: strict, lenient, compliance and the
            content reading, HotpotQA content as hotpot-content). A forced reply is stored as the panel's runner stores
            one (without the prefill): cut after its first "}" on GSM8K and BIG-Bench Hard, and, where the reply ended
            on the stop and the route did not return the stop string, the "}" put back (forced_reply); the scorer puts
            the prefill in front once and reads a box the stop left open through src/forced_stop.py, as for the panel.
            w is the share of requested replies longer than 16 tokens (and 64 on HotpotQA), from the API's completion
            count (which, like the panel's counts, includes the end-of-turn token) and from the visible text recounted
            by one tokenizer (Qwen3.6-27B's). Refusals, provider blocks and truncations are kept and counted.
  Report    per arm: accuracy (strict, lenient, content) and w; the gains CoT - forced and CoT - requested and the
            corrected gain of src/baseline_mixture.RULE_TEXT, (1 - w) x (CoT - forced), w at the forced cap; per
            template and per benchmark, with item-bootstrap intervals; and Qwen4-Max's placement k/M among the panel's
            and the ladder's models on the same 100 items and templates, from their scored tables.

Phases (subcommands):
  plan          items manifest and request lists of the three arms, counts and worst cases; offline
  quote         offline apart from a free read of the public listing: input tokens of every request (o200k, and
                Qwen3.6-27B with the template tokens measured on the prefill test), outputs at the panel's strongest
                models' lengths and at each arm's cap, list prices; results/frontier_panel/qwen3arm_quote.{json,md}
  pilot         5 prompts per benchmark per arm (the same five cells for the three arms), each sent twice; the forced
                prompts of GSM8K and BIG-Bench Hard once more without the stop (the stop control); 2 thinking-on control
                calls on a hard calibration item; before them a free read of the pinned endpoint's public listing.
                Checks: answered, reasoning tokens 0 with thinking off and above 0 on the controls, served provider and
                model as pinned, finish reasons, token accounting, repeat agreement per arm, billed cost per call, the
                forced replies continue the open box (as in the prefill test), and the stop control. A failed check
                stops (exit 3) and `run` refuses until the pilot passed under the same configuration hash.
  prefill-test  Qwen4-Max only (partial_true passed): 10 calibration items per benchmark, the
                assistant turn opened by "\boxed{", two variants (plain, partial: true).
  run           the three arms (or --arms), in chunks whose worst case must fit under the limit.
  report        results/frontier_panel/closed_w.{md,json}, closed_w_clean.md, closed_w_templates.csv.

Money: every paid subcommand needs --approved (an explicit approval flag) and --limit-usd. The pilot's and the prefill test's limits are
capped by the config; the run has no ceiling in the config (limits.run null): --limit-usd sets it and counts
everything this design has spent (pilot, prefill test, run). A chunk of calls is sent only when its worst case fits
under the limit beside what is already counted: each call's messages at one token per UTF-8 byte plus 64 (a byte-level
tokenizer never gives more tokens than bytes), the arm's whole output cap (32,768 tokens reserved for a thinking-on
control), list prices. An attempt with no HTTP reply, or one logged as sent and never answered (a crash), counts at its
worst case; HTTP errors count 0 (not billed); answered calls count what OpenRouter bills. Retries (with backoff) only
for 408, 429, 5xx and failed connections. Each paid phase appends an entry to the log of config/budget.yaml.

The key is read from the environment variable OPENROUTER_API_KEY and is never printed, logged or written;
every saved or printed line passes through redact().

Usage (repository root):
    python scripts/closed_w.py plan
    python scripts/closed_w.py quote --tiktoken-path <dir> --tiktoken-cache <dir>
    python scripts/closed_w.py pilot --approved --limit-usd 2
    python scripts/closed_w.py run --approved --limit-usd <limit>
    python scripts/closed_w.py report            # add --open-texts to recount the panel's texts
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import closed_w_items as cwp  # noqa: E402  (the B100 item draws and their seeds)

CONFIG = ROOT / "config" / "closed_w.yaml"
TASKS = ("gsm8k", "math", "bbh", "hotpot")
TASK_NAME = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
ARM = "direct_answer"
ARMS = ("requested", "forced", "cot")          # the run's arms (config run_arms gives the order sent)
ARM_SOURCE = {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot"}
ARM_NAME = {"requested": "requested", "forced": "forced", "cot": "chain of thought"}
SOURCE_ARM = {v: k for k, v in ARM_SOURCE.items()}
ACCEPTED = ("reply", "refusal", "blocked")     # outcomes that end a key; "failed" is retried, "error" is not
FINISH_OK = ("stop", "length", "content_filter")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
RESTART_MARKS = ["\\boxed", "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|start|>", "<|channel|>",
                 "<|end|>", "<｜User｜>", "<｜Assistant｜>"]
PHASE_FILES = {"pilot": "pilot_raw.jsonl", "prefill": "prefill_raw.jsonl", "run": "run_raw.jsonl"}
PHASE_CMD = {"pilot": "pilot", "prefill": "prefill-test", "run": "run"}


# ----------------------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hkey(*parts) -> str:
    return sha256_text("|".join(map(str, parts)))


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def dumps(o) -> str:
    return json.dumps(o, indent=1, ensure_ascii=False, default=_json_default)


def redact(text, secret: str | None = None) -> str:
    """The key and any bearer or OpenRouter credential removed from text before it is printed or saved."""
    s = "" if text is None else str(text)
    if secret:
        s = s.replace(secret, "REDACTED")
    s = re.sub(r"sk-or-[A-Za-z0-9_\-]{8,}", "REDACTED", s)
    s = re.sub(r"(?i)(bearer\s+)[^\s\"',}]+", r"\1REDACTED", s)
    s = re.sub(r"(?i)(\"?(?:authorization|api[_-]?key|x-api-key)\"?\s*[:=]\s*\"?)[^\s\"',}]+", r"\1REDACTED", s)
    return s


def clean_md(md: str) -> str:
    """The clean copy: links reduced to their text, nothing else changed (the reports carry no citation markers)."""
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", md)


def pct(x, d=1) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}"


def usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


# ----------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------

def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = [m["key"] for m in cfg["models"]]
    if len(set(keys)) != len(keys):
        raise SystemExit("config: duplicate model keys")
    for m in cfg["models"]:
        p = m.get("provider") or {}
        if not p.get("only") or p.get("allow_fallbacks") is not False or p.get("require_parameters") is not True:
            raise SystemExit(f"config: {m['key']} must pin its provider (only, allow_fallbacks false, "
                             "require_parameters true)")
        if not m.get("reasoning_off"):
            raise SystemExit(f"config: {m['key']} has no reasoning_off setting")
    for k in cfg.get("default_models") or []:
        if k not in keys:
            raise SystemExit(f"config: default model {k!r} is not among the models")
    bad = [a for a in run_arms(cfg) if a not in ARMS]
    if bad:
        raise SystemExit(f"config: unknown arms {bad}")
    return cfg


def run_arms(cfg: dict) -> list[str]:
    return list(cfg.get("run_arms") or ["requested"])


def default_model_keys(cfg: dict) -> list[str]:
    return list(cfg.get("default_models") or [m["key"] for m in cfg["models"]])


def config_sha(cfg: dict) -> str:
    """Hash of everything that shapes a request body or its reading."""
    part = {k: cfg.get(k) for k in ("base_url", "models", "decoding", "caps", "prefill", "forced_cut_at_first_brace",
                                    "forced", "arms", "run_arms", "requests")}
    return sha256_text(json.dumps(part, sort_keys=True, default=str))


def arm_cap(cfg: dict, arm: str, task: str) -> int:
    """The output cap of an arm on a benchmark (the panel's)."""
    if arm == "requested":
        return int(cfg["caps"]["requested"])
    if arm == "forced":
        return int(cfg["caps"]["forced"][task])
    if arm == "cot":
        return int(cfg["caps"]["cot"][task])
    raise ValueError(arm)


def forced_stop_sent(cfg: dict, task: str) -> bool:
    """Whether the forced arm sends the stop "}" on this benchmark: where the panel's generation stopped there."""
    return bool((cfg.get("forced") or {}).get("send_stop")) and task in cfg["forced_cut_at_first_brace"]


def model_of(cfg: dict, key: str) -> dict:
    for m in cfg["models"]:
        if m["key"] == key:
            return m
    raise SystemExit(f"no model {key!r} in the config")


def check_panel_decoding(cfg: dict, root: Path = ROOT) -> dict:
    """The caps, prefill and greedy settings of the config agree with the panel's decoding files."""
    c = Path(root) / "config"

    def rd(n):
        return yaml.safe_load((c / n).read_text(encoding="utf-8"))

    dec, hot = rd("decoding.yaml"), rd("decoding_hotpot.yaml")
    ff, fm = rd("decoding_forced.yaml"), rd("decoding_forced_math.yaml")
    forced = {"gsm8k": ff, "math": fm, "bbh": ff, "hotpot": hot}
    probs = []
    for name, d in (("decoding.yaml", dec), ("decoding_hotpot.yaml", hot)):
        if int(d["max_new_tokens"]["direct_answer"]) != int(cfg["caps"]["requested"]):
            probs.append(f"{name}: requested cap {d['max_new_tokens']['direct_answer']} != {cfg['caps']['requested']}")
        if d.get("stop"):
            probs.append(f"{name}: the requested and chain-of-thought arms have a stop list {d['stop']}")
        if float(d["temperature"]) != float(cfg["decoding"]["temperature"]) or \
                float(d["top_p"]) != float(cfg["decoding"]["top_p"]):
            probs.append(f"{name}: temperature/top_p differ from the config")
    cot = {"gsm8k": dec, "math": dec, "bbh": dec, "hotpot": hot}
    if "cot" in (cfg.get("caps") or {}):
        for t, d in cot.items():
            if int(d["max_new_tokens"]["zero_shot_cot"]) != int(cfg["caps"]["cot"][t]):
                probs.append(f"{t}: chain-of-thought cap {d['max_new_tokens']['zero_shot_cot']} != "
                             f"{cfg['caps']['cot'][t]}")
    cut = set()
    for t, d in forced.items():
        if int(d["max_new_tokens"]["forced_direct"]) != int(cfg["caps"]["forced"][t]):
            probs.append(f"{t}: forced cap {d['max_new_tokens']['forced_direct']} != {cfg['caps']['forced'][t]}")
        if d["prefill"]["forced_direct"] != cfg["prefill"]:
            probs.append(f"{t}: forced prefill {d['prefill']['forced_direct']!r} != {cfg['prefill']!r}")
        if "}" in (d.get("stop") or []):
            cut.add(t)
            if not d.get("include_stop_str_in_output"):
                probs.append(f"{t}: the forced stop '}}' is not kept in the panel's output")
    if cut != set(cfg["forced_cut_at_first_brace"]):
        probs.append(f"forced runs stopped at the first '}}' on {sorted(cut)}, config says "
                     f"{sorted(cfg['forced_cut_at_first_brace'])}")
    if probs:
        raise SystemExit("config/closed_w.yaml disagrees with the panel's decoding: " + "; ".join(probs))
    return {"requested_cap": int(cfg["caps"]["requested"]), "forced_caps": dict(cfg["caps"]["forced"]),
            "cot_caps": dict(cfg["caps"].get("cot") or {}),
            "prefill": cfg["prefill"], "cut_at_first_brace": sorted(cut),
            "stop_sent": sorted(t for t in TASKS if forced_stop_sent(cfg, t)),
            "sources": ["config/decoding.yaml", "config/decoding_hotpot.yaml", "config/decoding_forced.yaml",
                        "config/decoding_forced_math.yaml"]}


class Paths:
    def __init__(self, cfg: dict, root: Path = ROOT, **over):
        p = cfg["paths"]
        self.root = Path(root)
        self.manifest = Path(over.get("manifest") or self.root / p["manifest"])
        self.raw_dir = Path(over.get("raw_dir") or self.root / p["raw_dir"])
        self.results = Path(over.get("results") or self.root / p["results_dir"])
        self.budget = Path(over.get("budget") or self.root / p["budget"])
        self.env = Path(over.get("env") or self.root / ".env")
        self.prices = Path(over.get("prices") or self.root / p["prices"])
        self.price_tokens = Path(over.get("price_tokens") or self.root / p["price_tokens"])
        self.prefill_result = Path(over.get("prefill_result") or self.root / p.get(
            "prefill_result", "results/frontier_panel/qwen_prefill_test.json"))

    def raw(self, phase: str) -> Path:
        return self.raw_dir / PHASE_FILES[phase]


# ----------------------------------------------------------------------------------------------------------------
# Requests, items, manifest
# ----------------------------------------------------------------------------------------------------------------

def request_path(cfg: dict, task: str, root: Path = ROOT) -> Path:
    return Path(root) / "data" / "requests" / task / f"{cfg['design']['reference_requests']}.parquet"


def screening_path(cfg: dict, task: str, root: Path = ROOT) -> Path:
    return Path(root) / "data" / "requests" / task / "screening" / f"{cfg['design']['reference_requests']}.parquet"


def single_user(m) -> bool:
    return (isinstance(m, list) and len(m) == 1 and isinstance(m[0], dict) and set(m[0]) == {"role", "content"}
            and m[0]["role"] == "user" and isinstance(m[0]["content"], str))


def load_requests(path, arm: str = ARM, request_ids=None) -> pd.DataFrame:
    """The panel's rows of one arm (source label, e.g. direct_answer): request_id, config_id, item_id (str),
    messages_json, text (the user message). path is one file or a list (HotpotQA's forced blocks). Every row must be
    exactly one user message."""
    paths = [Path(x) for x in path] if isinstance(path, (list, tuple)) else [Path(path)]
    filters = [("arm", "==", arm)]
    if request_ids is not None:
        filters.append(("request_id", "in", list(request_ids)))
    parts = [pd.read_parquet(p, columns=["request_id", "arm", "config_id", "item_id", "messages_json"],
                             filters=filters) for p in paths]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    where = ", ".join(rel(p) for p in paths)
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    msgs = df["messages_json"].map(json.loads)
    bad = int((~msgs.map(single_user)).sum())
    if bad:
        raise SystemExit(f"{where}: {bad} {arm} rows are not exactly one user message")
    if df["request_id"].duplicated().any():
        raise SystemExit(f"{where}: duplicated request ids")
    df["text"] = msgs.map(lambda m: m[0]["content"])
    return df.drop(columns=["arm"]).reset_index(drop=True)


def arm_request_paths(cfg: dict, arm: str, task: str, root: Path = ROOT) -> list[Path]:
    """The request files of an arm on a benchmark (config `requests`)."""
    spec = (cfg.get("requests") or {}).get(arm)
    if spec is None and arm == "requested":
        return [request_path(cfg, task, root)]
    if isinstance(spec, dict):
        spec = spec[task]
    ref = cfg["design"]["reference_requests"]
    return [Path(root) / s.format(task=task, ref=ref) for s in spec]


def load_arm_requests(cfg: dict, arm: str, task: str, root: Path = ROOT, request_ids=None) -> pd.DataFrame:
    return load_requests(arm_request_paths(cfg, arm, task, root), ARM_SOURCE[arm], request_ids)


def list_digest(df: pd.DataFrame) -> str:
    """sha256 of the sorted lines 'request_id<TAB>sha256(user message)': pins the request list byte for byte."""
    lines = sorted(f"{r}\t{sha256_text(t)}" for r, t in zip(df["request_id"], df["text"]))
    return sha256_text("\n".join(lines))


def draw_items(task: str, item_ids, n: int = 100) -> list[str]:
    """The B100 draws of scripts/closed_w_items.py (nested seeded permutations; BIG-Bench Hard 4 per task)."""
    return [str(x) for x in cwp.sample_items(task, [str(i) for i in item_ids], n)]


def seed_labels(task: str, item_ids) -> dict:
    if task == "bbh":
        ks = sorted({str(i).split("/", 1)[0] for i in item_ids})
        return {f"closed_w/b/bbh/{k}": cwp.label_seed(f"closed_w/b/bbh/{k}") for k in ks}
    return {f"closed_w/b/{task}": cwp.label_seed(f"closed_w/b/{task}")}


def pick_spread(df: pd.DataFrame, n: int, label: str, task: str, longest_first: bool = False) -> pd.DataFrame:
    """n rows in sha256 order of (label, task, config, item), with distinct templates, items and (BIG-Bench Hard)
    tasks; with longest_first the longest message (in UTF-8 bytes) comes first."""
    d = df.copy()
    d["h"] = [hkey(label, task, c, i) for c, i in zip(d["config_id"], d["item_id"])]
    d["n_bytes"] = d["text"].map(lambda s: len(s.encode("utf-8")))
    d = d.sort_values("h", kind="stable")
    order = list(d.index)
    if longest_first and len(d):
        top = d.sort_values(["n_bytes", "h"], ascending=[False, True], kind="stable").index[0]
        order = [top] + [i for i in order if i != top]

    def group(i):
        return i.split("/", 1)[0] if task == "bbh" else i

    chosen, cs, its, gs = [], set(), set(), set()
    for ix in order:
        c, i = d.at[ix, "config_id"], d.at[ix, "item_id"]
        if c in cs or i in its or group(i) in gs:
            continue
        chosen.append(ix)
        cs.add(c), its.add(i), gs.add(group(i))
        if len(chosen) == n:
            break
    for ix in order:                      # fewer distinct ones than n: fill in sha256 order
        if len(chosen) >= n:
            break
        if ix not in chosen:
            chosen.append(ix)
    return d.loc[chosen]


def pick_control(screening_math: pd.DataFrame, levels: pd.DataFrame, templates) -> dict:
    """The thinking-on control: a MATH Level 5 calibration item (training split) under a run template, the first in
    sha256 order of ('closed_w/control', config, item)."""
    hard = set(levels.loc[levels["level"].astype(str) == "Level 5", "item_id"].astype(str))
    s = screening_math[screening_math["item_id"].isin(hard) & screening_math["config_id"].isin(set(templates))]
    if s.empty:
        raise SystemExit("no Level 5 MATH calibration item under a run template")
    s = s.assign(h=[hkey("closed_w/control", "math", c, i) for c, i in zip(s["config_id"], s["item_id"])])
    r = s.sort_values("h").iloc[0]
    return {"task": "math", "request_id": r["request_id"], "config_id": r["config_id"], "item_id": r["item_id"],
            "level": "Level 5", "text_sha256": sha256_text(r["text"]), "n_bytes": len(r["text"].encode("utf-8"))}


def build_manifest(cfg: dict, reqs: dict, screening: dict, control: dict, sources: dict,
                   made_on: str | None = None) -> dict:
    n = int(cfg["design"]["items_per_benchmark"])
    kp = int(cfg["design"]["pilot_prompts_per_benchmark"])
    kq = int(cfg["design"]["prefill_items_per_benchmark"])
    tasks, pilot, prefill, seeds = {}, {}, {}, {}
    for t in TASKS:
        df = reqs[t]
        items = draw_items(t, df["item_id"].unique().tolist(), n)
        sel = df[df["item_id"].isin(set(items))]
        templates = sorted(sel["config_id"].unique())
        if len(sel) != len(templates) * len(items):
            raise SystemExit(f"{t}: the B100 grid is incomplete ({len(sel)} rows for {len(templates)} templates x "
                             f"{len(items)} items)")
        seeds[t] = seed_labels(t, df["item_id"].unique())
        tasks[t] = {"templates": templates, "n_templates": len(templates), "items": items, "n_items": len(items),
                    "n_requests": int(len(sel)), "requests_sha256": list_digest(sel),
                    "request_id_example": sel["request_id"].iloc[0]}
        p = pick_spread(sel, kp, "closed_w/pilot", t, longest_first=(t == "hotpot"))
        pilot[t] = [{"request_id": r.request_id, "config_id": r.config_id, "item_id": r.item_id,
                     "n_bytes": int(r.n_bytes), "text_sha256": sha256_text(r.text)} for r in p.itertuples()]
        s = screening[t]
        s = s[s["config_id"].isin(set(templates))]
        q = pick_spread(s, kq, "closed_w/prefill", t)
        prefill[t] = [{"request_id": r.request_id, "config_id": r.config_id, "item_id": r.item_id,
                       "n_bytes": int(r.n_bytes), "text_sha256": sha256_text(r.text)} for r in q.itertuples()]
    return {
        "name": "closed_w_items", "made_on": made_on or today(), "script": "scripts/closed_w.py plan",
        "design": "requested arm (direct_answer) of the main panel, every template of the four benchmarks, the same "
                  f"{n} items per benchmark; one call per (template, item) per model",
        "item_rule": cwp_rule(),
        "seeds": seeds, "items_per_benchmark": n, "bbh_per_task": int(cwp.BBH_PER_TASK[n]) if n in cwp.BBH_PER_TASK
        else None,
        "sources": sources, "tasks": tasks,
        "n_requests_per_model": int(sum(v["n_requests"] for v in tasks.values())),
        "n_cells": int(sum(v["n_templates"] for v in tasks.values())),
        "request_list_rule": "per benchmark, the rows of the source file with arm direct_answer, an item of `items` "
                             "and a template of `templates` (every pair present once); requests_sha256 is the sha256 "
                             "of the sorted lines 'request_id<TAB>sha256(user message, UTF-8)' joined by newlines",
        "pilot": {"rule": f"per benchmark {kp} rows of the request list in sha256 order of ('closed_w/pilot', benchmark, "
                          "template, item), with distinct templates, items and BIG-Bench Hard tasks; on HotpotQA the "
                          "longest request (UTF-8 bytes) first", "rows": pilot},
        "prefill": {"rule": f"per benchmark {kq} calibration rows (training split; data/requests/<task>/screening/"
                            f"{cfg['design']['reference_requests']}.parquet, direct_answer) under a run template, in "
                            "sha256 order of ('closed_w/prefill', benchmark, template, item), with distinct templates, "
                            "items and BIG-Bench Hard tasks", "rows": prefill},
        "control": {"rule": "the MATH Level 5 calibration item (data/processed/math/items_screening.parquet) under a "
                            "run template, first in sha256 order of ('closed_w/control', 'math', template, item); its "
                            "direct_answer prompt is sent with thinking on", "row": control},
    }


def cwp_rule() -> str:
    return ("nested seeded permutations of the sorted item ids, numpy.random.default_rng(src.decision.label_seed("
            "'closed_w/b/<task>')), the first 100; BIG-Bench Hard per task, label 'closed_w/b/bbh/<task>', the first 4 "
            "of each of 25 tasks (scripts/closed_w_items.sample_items, Design B100)")


def manifest_body(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("made_on", "arms_added_on")}


def base_body(m: dict) -> dict:
    """The manifest of the requested arm as first written (items, templates, pilot, prefill, control rows)."""
    return {k: v for k, v in manifest_body(m).items() if k != "arms"}


def build_arms_section(cfg: dict, man: dict, reqs_by_arm: dict, root: Path = ROOT) -> dict:
    """The forced and chain-of-thought arms on the manifest's items and templates: per arm and benchmark the source
    files and their sha256, the request-list hash, and the pilot rows (the requested arm's pilot cells, so the three
    arms of the pilot share their cells); the forced user message checked equal to the requested one on every cell."""
    out = {"rule": "per arm and benchmark, the rows of the arm's request files (config `requests`) with the arm's "
                   "source label, an item of tasks.<benchmark>.items and a template of tasks.<benchmark>.templates "
                   "(every pair present once); requests_sha256 as tasks.<benchmark>.requests_sha256. The pilot rows "
                   "of every arm are the cells of pilot.rows (the requested arm's pilot); the stop control re-sends "
                   "the forced pilot rows of the benchmarks whose forced stop is sent, without the stop.",
           "arms": {}, "pilot": {}}
    req = reqs_by_arm["requested"]
    for arm in ARMS:
        out["arms"][arm], out["pilot"][arm] = {}, {}
        for t in TASKS:
            m = man["tasks"][t]
            df = reqs_by_arm[arm][t]
            sel = df[df["item_id"].isin(set(m["items"])) & df["config_id"].isin(set(m["templates"]))]
            if len(sel) != m["n_requests"] or sel[["config_id", "item_id"]].duplicated().any():
                raise SystemExit(f"{arm}/{t}: {len(sel)} rows on the manifest's cells, expected {m['n_requests']}")
            paths = arm_request_paths(cfg, arm, t, root)
            entry = {"source_arm": ARM_SOURCE[arm], "sources": [{"path": rel(p), "sha256": sha256_file(p)}
                                                                 for p in paths],
                     "n_requests": int(len(sel)), "requests_sha256": list_digest(sel),
                     "request_id_example": sel.sort_values("request_id")["request_id"].iloc[0]}
            if arm == "forced":
                r = req[t]
                r = r[r["item_id"].isin(set(m["items"])) & r["config_id"].isin(set(m["templates"]))]
                j = sel.merge(r[["config_id", "item_id", "text"]], on=["config_id", "item_id"], suffixes=("", "_req"))
                diff = int((j["text"] != j["text_req"]).sum())
                if len(j) != len(sel) or diff:
                    raise SystemExit(f"forced/{t}: {diff} user messages differ from the requested arm's "
                                     f"({len(j)} of {len(sel)} cells matched)")
                entry["user_message_equals_requested"] = True
            out["arms"][arm][t] = entry
            ix = sel.set_index(["config_id", "item_id"])
            rows = []
            for p in man["pilot"]["rows"][t]:
                r = ix.loc[(p["config_id"], p["item_id"])]
                rows.append({"request_id": r["request_id"], "config_id": p["config_id"], "item_id": p["item_id"],
                             "n_bytes": len(r["text"].encode("utf-8")), "text_sha256": sha256_text(r["text"])})
            out["pilot"][arm][t] = rows
    out["n_requests_per_arm"] = {a: int(sum(v["n_requests"] for v in out["arms"][a].values())) for a in ARMS}
    out["stop_control"] = {"benchmarks": sorted(t for t in TASKS if forced_stop_sent(cfg, t)),
                           "rows": "pilot.forced.<benchmark>, sent once more without the stop"}
    return out


def load_arms_all(cfg: dict, root: Path = ROOT, arms=ARMS) -> dict:
    """{arm: {benchmark: rows}}; the requested arm through load_all_requests."""
    out = {}
    for a in arms:
        out[a] = load_all_requests(cfg, root) if a == "requested" else \
            {t: load_arm_requests(cfg, a, t, root) for t in TASKS}
    return out


def selected_arm(man: dict, arm: str, reqs: dict, verify: bool = True) -> dict:
    """An arm's run rows per benchmark, checked against the manifest's request-list hash for that arm."""
    if arm == "requested":
        return selected(man, reqs, verify)
    if "arms" not in man:
        raise SystemExit("the manifest has no arms section: run `scripts/closed_w.py plan` first")
    out = {}
    for t in TASKS:
        m, a = man["tasks"][t], man["arms"]["arms"][arm][t]
        df = reqs[t]
        sel = df[df["item_id"].isin(set(m["items"])) & df["config_id"].isin(set(m["templates"]))].reset_index(drop=True)
        if verify and (len(sel) != a["n_requests"] or list_digest(sel) != a["requests_sha256"]):
            raise SystemExit(f"{arm}/{t}: the request list differs from the manifest ({len(sel)} rows; the source "
                             "file changed after `plan`)")
        out[t] = sel
    return out


def load_manifest(paths: Paths) -> dict:
    if not paths.manifest.exists():
        raise SystemExit(f"{rel(paths.manifest)} is absent: run `scripts/closed_w.py plan` first")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def load_all_requests(cfg: dict, root: Path = ROOT) -> dict:
    return {t: load_requests(request_path(cfg, t, root)) for t in TASKS}


def selected(man: dict, reqs: dict, verify: bool = True) -> dict:
    """The run's rows per benchmark, checked against the manifest's request-list hash."""
    out = {}
    for t in TASKS:
        m = man["tasks"][t]
        df = reqs[t]
        sel = df[df["item_id"].isin(set(m["items"])) & df["config_id"].isin(set(m["templates"]))].reset_index(drop=True)
        if verify:
            if len(sel) != m["n_requests"] or list_digest(sel) != m["requests_sha256"]:
                raise SystemExit(f"{t}: the request list differs from the manifest ({len(sel)} rows; the source file "
                                 "changed after `plan`)")
        out[t] = sel
    return out


# ----------------------------------------------------------------------------------------------------------------
# Request bodies and money
# ----------------------------------------------------------------------------------------------------------------

def sent_parameters(model: dict, stop: bool = False) -> list[str]:
    return ["max_tokens", "temperature", "top_p", "reasoning"] + (["stop"] if stop else []) + \
        sorted((model.get("extra") or {}).keys())


def design_parameters(cfg: dict, model: dict) -> list[str]:
    """Every parameter some call of the design sends (the stop only with the forced arm on its benchmarks)."""
    stop = "forced" in run_arms(cfg) and any(forced_stop_sent(cfg, t) for t in TASKS)
    return sent_parameters(model, stop)


def request_body(cfg: dict, model: dict, messages: list[dict], cap: int, reasoning: str = "off",
                 stop: list[str] | None = None) -> dict:
    """One chat-completions body: the messages as given, the cap, greedy decoding, reasoning off (or on for the
    control), the pinned provider. No system message, tools or response format is ever added; a stop list only when
    given (the forced arm of GSM8K and BIG-Bench Hard)."""
    body = {"model": model["id"], "messages": copy.deepcopy(messages), "max_tokens": int(cap),
            "temperature": float(cfg["decoding"]["temperature"]), "top_p": float(cfg["decoding"]["top_p"])}
    if stop:
        body["stop"] = list(stop)
    body.update(copy.deepcopy(model.get("extra") or {}))
    body["reasoning"] = copy.deepcopy(model["reasoning_off"] if reasoning == "off" else model["reasoning_on"])
    body["provider"] = copy.deepcopy(model["provider"])
    body["usage"] = {"include": True}
    return body


def forced_messages(cfg: dict, user_text: str, message_extra: dict | None = None) -> list[dict]:
    """The forced arm's messages: the user message and an assistant message holding the prefill (with the forced
    arm's extra fields, partial: true)."""
    extra = (cfg.get("forced") or {}).get("message_extra") if message_extra is None else message_extra
    return [{"role": "user", "content": user_text},
            {"role": "assistant", "content": cfg["prefill"], **copy.deepcopy(extra or {})}]


def encode_body(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def upper_cost(n_bytes: int, out_tokens: int, price: dict, overhead: int) -> float:
    """The most one call can cost: every message byte a prompt token plus the overhead, the whole output reserve."""
    return (int(n_bytes) + int(overhead)) * float(price["prompt"]) + int(out_tokens) * float(price["completion"])


REQUEST_KINDS = ("requested", "forced", "cot", "forced_nostop", "control", "prefill")


def make_request(cfg: dict, model: dict, task: str, row: dict, kind: str = "requested", rep: int | None = None,
                 variant: dict | None = None) -> dict:
    """One call. kind: an arm of the run (requested, forced, cot); forced_nostop (the pilot's stop control: the forced
    request without the stop); control (thinking on); prefill (the prefill test's variants). Keys:
    model|arm|benchmark|request id (|rep<k> in the pilot)."""
    text = row["text"]
    user = {"role": "user", "content": text}
    reasoning, stop, arm = "off", None, None
    sfx = f"|rep{rep}" if rep is not None else ""
    if kind in ARMS:
        arm = kind
        cap = reserve = arm_cap(cfg, arm, task)
        if kind == "forced":
            messages = forced_messages(cfg, text)
            stop = ["}"] if forced_stop_sent(cfg, task) else None
        else:
            messages = [user]
        key = f"{model['key']}|{arm}|{task}|{row['request_id']}" + sfx
    elif kind == "forced_nostop":
        arm = "forced"
        cap = reserve = arm_cap(cfg, "forced", task)
        messages = forced_messages(cfg, text)
        key = f"{model['key']}|forced_nostop|{task}|{row['request_id']}" + sfx
    elif kind == "control":
        messages = [user]
        reasoning, cap, reserve = "on", int(cfg["caps"]["control"]), int(cfg["caps"]["control_reserve"])
        key = f"{model['key']}|control|{task}|{row['request_id']}|rep{rep}"
    elif kind == "prefill":
        messages = [user, {"role": "assistant", "content": cfg["prefill"], **(variant.get("message_extra") or {})}]
        cap = reserve = int(cfg["caps"]["forced"][task])
        key = f"{model['key']}|{variant['name']}|{task}|{row['request_id']}"
    else:
        raise ValueError(kind)
    n_bytes = sum(len(m["content"].encode("utf-8")) for m in messages)
    return {"key": key, "model_key": model["key"], "task": task, "request_id": row["request_id"],
            "config_id": row["config_id"], "item_id": str(row["item_id"]), "kind": kind, "arm": arm, "rep": rep,
            "variant": variant["name"] if variant else None, "cap": cap, "reasoning": reasoning,
            "stop_sent": bool(stop), "body": request_body(cfg, model, messages, cap, reasoning, stop),
            "n_bytes": n_bytes,
            "reserve_usd": upper_cost(n_bytes, reserve, model["price"], cfg["run"]["prompt_overhead_tokens"]),
            "text_sha256": sha256_text(text), "price": dict(model["price"]),
            "expected_provider": model["expected_provider"], "served_model_prefix": model["served_model_prefix"]}


def expected_cost(p: dict, price: dict) -> float:
    pt, ct, ca = int(p.get("prompt_tokens") or 0), int(p.get("completion_tokens") or 0), int(p.get("cached_tokens") or 0)
    ca = min(max(ca, 0), pt)
    return (pt - ca) * float(price["prompt"]) + ca * float(price.get("cache_read", price["prompt"])) + \
        ct * float(price["completion"])


def billed_cost(p: dict, price: dict) -> tuple[float | None, str | None]:
    if p.get("cost_reported") is not None:
        return float(p["cost_reported"]), "openrouter usage.cost"
    if p.get("prompt_tokens") is not None or p.get("completion_tokens") is not None:
        return expected_cost(p, price), "tokens x config prices"
    return None, None


# ----------------------------------------------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------------------------------------------

def read_key(env_path: Path, name: str = "OPENROUTER_API_KEY") -> str:
    """The key from the environment variable `name` (OPENROUTER_API_KEY for OpenRouter, ANTHROPIC_API_KEY for the
    reading audit's judge). The value never leaves this process. `env_path` is not read; it stays in the signature
    for the callers."""
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"{name} is not set in the environment")
    return v


class Http:
    """OpenRouter over urllib. post_json sends the key; get_public reads a public page without it."""

    def __init__(self, base_url: str, key: str | None, timeout: float = 300.0, opener=None):
        self.base, self._key, self.timeout = base_url.rstrip("/"), key, float(timeout)
        self.opener = opener or urllib.request.urlopen

    def _send(self, method: str, url: str, data: bytes | None, auth: bool) -> dict:
        req = urllib.request.Request(url, data=data, method=method)
        if auth:
            req.add_header("Authorization", "Bearer " + (self._key or ""))
        req.add_header("Content-Type", "application/json")
        t0 = time.time()
        try:
            with self.opener(req, timeout=self.timeout) as r:
                raw = r.read()
                st = getattr(r, "status", None) or r.getcode()
            try:
                j = json.loads(raw.decode("utf-8"))
            except ValueError:
                return {"status": st, "json": None, "error": redact(raw[:2000].decode("utf-8", "replace"), self._key),
                        "seconds": round(time.time() - t0, 3)}
            return {"status": st, "json": j, "seconds": round(time.time() - t0, 3)}
        except urllib.error.HTTPError as e:
            try:
                txt = redact(e.read().decode("utf-8", "replace"), self._key)
            except Exception:  # noqa: BLE001
                txt = ""
            try:
                j = json.loads(txt)
            except ValueError:
                j = None
            return {"status": e.code, "json": j, "error": redact(txt[:2000], self._key),
                    "seconds": round(time.time() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            return {"status": None, "json": None, "error": redact(f"{type(e).__name__}: {e}", self._key),
                    "seconds": round(time.time() - t0, 3)}

    def post_json(self, path: str, data: bytes) -> dict:
        return self._send("POST", self.base + path, data, True)

    def get_public(self, url: str) -> dict:
        return self._send("GET", url, None, False)


def _flagged(err) -> bool:
    if isinstance(err, dict):
        md = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        if "reasons" in md or "flagged_input" in md:
            return True
        # Alibaba's own content inspection (input or output), passed through by OpenRouter as a 403 with
        # provider_error_code data_inspection_failed: no model output reaches us, so it is a provider block.
        if md.get("provider_error_code") == "data_inspection_failed":
            return True
        msg = str(err.get("message") or "")
    else:
        msg = str(err or "")
    return bool(re.search(r"(?i)moderation|flagged|data_inspection_failed|inappropriate content", msg))


def parse_reply(resp: dict) -> dict:
    """One OpenRouter reply read into flat fields; outcome reply, refusal, blocked (the route's moderation refused the
    input: no model output), failed (technical, retried) or error (not retried)."""
    st = resp.get("status")
    j = resp.get("json") if isinstance(resp.get("json"), dict) else None
    out = {"status": st, "outcome": None, "text": None, "refusal": None, "reasoning_text": None,
           "reasoning_details_n": 0, "finish_reason": None, "native_finish_reason": None, "prompt_tokens": None,
           "completion_tokens": None, "reasoning_tokens": None, "cached_tokens": None, "cost_reported": None,
           "usage": None, "served_model": None, "provider": None, "generation_id": None, "system_fingerprint": None,
           "created": None, "error": None}
    if j is not None:
        out.update(generation_id=j.get("id"), served_model=j.get("model"), provider=j.get("provider"),
                   system_fingerprint=j.get("system_fingerprint"), created=j.get("created"))
        u = j.get("usage")
        if isinstance(u, dict):
            ptd = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
            ctd = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
            out.update(usage=u, prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       reasoning_tokens=ctd.get("reasoning_tokens"), cached_tokens=ptd.get("cached_tokens"),
                       cost_reported=u.get("cost"))
    err = j.get("error") if j is not None else None
    if st == 200 and j is not None and not err and j.get("choices"):
        ch = j["choices"][0] or {}
        msg = ch.get("message") or {}
        text = msg.get("content")
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        out.update(text=text, refusal=msg.get("refusal"),
                   reasoning_text=msg.get("reasoning") or msg.get("reasoning_content"),
                   reasoning_details_n=len(msg.get("reasoning_details") or []),
                   finish_reason=ch.get("finish_reason"), native_finish_reason=ch.get("native_finish_reason"))
        if ch.get("error") or out["finish_reason"] == "error":
            out["outcome"] = "failed"
            out["error"] = json.dumps(ch.get("error"), default=str)[:2000]
        elif (out["refusal"] or "").strip() or out["finish_reason"] == "content_filter":
            out["outcome"] = "refusal"
        else:
            out["outcome"] = "reply"
        return out
    msg = err if err is not None else resp.get("error")
    out["error"] = (json.dumps(msg, default=str) if isinstance(msg, (dict, list)) else str(msg or ""))[:2000]
    if st == 403 and _flagged(err if err is not None else resp.get("error")):
        out["outcome"] = "blocked"
    elif st is None or st in RETRY_STATUS or st == 200:
        out["outcome"] = "failed"
    else:
        out["outcome"] = "error"
    return out


# ----------------------------------------------------------------------------------------------------------------
# Raw files
# ----------------------------------------------------------------------------------------------------------------

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
            except ValueError:       # a line cut by a crash mid-write; its attempt counts as an orphan
                continue
    return out


def ledger(lines: list[dict]) -> dict:
    """What a raw file has cost: answered attempts at their bill, attempts with no HTTP reply and attempts logged as
    sent but never answered (a crash) at their worst case."""
    res = {}
    for x in lines:
        if x.get("event") == "result":
            res[x["attempt_id"]] = x
    orphans = [x for x in lines if x.get("event") == "sent" and x.get("attempt_id") not in res]
    charged = sum(float(x.get("charged_usd") or 0.0) for x in res.values())
    unknown = sum(float(x.get("reserve_usd") or 0.0) for x in orphans)
    billed = sum(float(x.get("cost_usd") or 0.0) for x in res.values())
    return {"counted_usd": charged + unknown, "billed_usd": billed, "orphans": len(orphans),
            "orphans_usd": unknown, "attempts": len(res)}


def done_keys(lines: list[dict], csha: str) -> set:
    return {x["key"] for x in lines if x.get("event") == "result" and x.get("outcome") in ACCEPTED
            and x.get("config_sha") == csha}


def final_by_key(lines: list[dict], csha: str | None = None) -> dict:
    """Per key the first answered result (else the last result) under the configuration hash."""
    best = {}
    for x in lines:
        if x.get("event") != "result" or (csha is not None and x.get("config_sha") != csha):
            continue
        k = x["key"]
        if k not in best:
            best[k] = x
        elif best[k].get("outcome") not in ACCEPTED:
            best[k] = x
    return best


def counted_spend(paths: Paths, phases=("pilot", "prefill", "run")) -> dict:
    return {ph: ledger(read_raw(paths.raw(ph))) for ph in phases}


# ----------------------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------------------

def backoff(tries: int, cfg: dict) -> float:
    b = cfg["run"]["backoff_s"]
    return float(min(b["max"], b["base"] * (2 ** max(0, tries - 1))))


class Runner:
    """Sends requests in chunks. Before a chunk goes out, the worst case of its calls must fit under the limit beside
    what is already counted; the chunk is cut to the calls that fit, and with none the run stops. Failed attempts
    (408, 429, 5xx, no reply) go to the front of the next chunk after a backoff, at most max_tries per request."""

    def __init__(self, http, cfg: dict, raw_path: Path, phase: str, limit: float, counted: float, csha: str, *,
                 check=None, stop_on_error: bool = True, sleep=time.sleep, log=print, threads: int | None = None,
                 chunk: int | None = None, monitor=None):
        r = cfg["run"]
        self.http, self.cfg, self.raw_path, self.phase = http, cfg, Path(raw_path), phase
        self.limit, self.counted0, self.csha = float(limit), float(counted), csha
        self.check, self.stop_on_error, self.sleep, self.log = check, stop_on_error, sleep, log
        self.monitor = monitor          # called with every accepted record; a returned text stops the run
        self.threads = int(threads or r["threads"])
        self.chunk = int(chunk or r["chunk_calls"])
        self.max_tries = int(r["max_tries"])
        self.max_failed_chunks = int(r["max_failed_chunks"])
        self.secret = getattr(http, "_key", None)
        self.session = uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.counted_run = 0.0
        self.stop_reason = None
        self.stats = {"attempts": 0, "answered": 0, "refusals": 0, "blocked": 0, "failed_attempts": 0, "errors": 0,
                      "missing": 0, "billed_usd": 0.0, "counted_usd": 0.0, "chunks": 0}
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def counted(self) -> float:
        return self.counted0 + self.counted_run

    def write(self, rec: dict) -> None:
        line = redact(json.dumps(rec, ensure_ascii=False, default=_json_default), self.secret)
        with self.lock:
            with open(self.raw_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

    def fit(self, cand: list[dict]) -> list[dict]:
        room = self.limit - self.counted
        out, tot = [], 0.0
        for r in cand:
            if tot + r["reserve_usd"] > room + 1e-12:
                break
            out.append(r)
            tot += r["reserve_usd"]
        return out

    def attempt(self, req: dict) -> dict:
        aid = uuid.uuid4().hex
        data = encode_body(req["body"])
        base = {k: req.get(k) for k in ("key", "model_key", "task", "request_id", "config_id", "item_id", "kind", "arm",
                                        "rep", "variant", "cap", "reasoning", "stop_sent", "n_bytes", "text_sha256",
                                        "reserve_usd")}
        base.update(attempt_id=aid, attempt=req["tries"] + 1, phase=self.phase, config_sha=self.csha,
                    session=self.session)
        self.write({"event": "sent", **base, "t": now_iso()})
        t_sent, t0 = now_iso(), time.time()
        try:
            resp = self.http.post_json("/chat/completions", data)
        except Exception as e:  # noqa: BLE001
            resp = {"status": None, "json": None, "error": f"{type(e).__name__}: {e}"}
        t_recv = now_iso()
        p = parse_reply(resp)
        billed, source = billed_cost(p, req["price"])
        if p["status"] is None:
            charged = float(req["reserve_usd"])          # no reply: it may still have been billed
        elif billed is not None:
            charged = float(billed)
        else:
            charged = 0.0                                # an HTTP error is not billed
        rec = {"event": "result", **base, "body_sha256": hashlib.sha256(data).hexdigest(), **p,
               "cost_usd": billed, "cost_source": source, "charged_usd": charged,
               "expected_cost_usd": expected_cost(p, req["price"]) if p.get("completion_tokens") is not None else None,
               "t_sent": t_sent, "t_received": t_recv, "seconds": round(time.time() - t0, 3)}
        if self.check is not None and p["outcome"] in ACCEPTED:
            rec["problems"] = list(self.check(req, rec))
            if rec["problems"]:              # kept, but not an answer of the design: not done, and the run stops
                rec["outcome_as_returned"], rec["outcome"] = rec["outcome"], "flagged"
        self.write(rec)
        return rec

    def run(self, reqs: list[dict]) -> dict:
        queue = [dict(r, tries=0) for r in reqs]
        streak, t0 = 0, time.time()
        while queue and self.stop_reason is None:
            cand = queue[: self.chunk]
            fit = self.fit(cand)
            if not fit:
                self.stop_reason = (f"spend limit ${self.limit:.2f}: ${self.counted:.4f} counted, the next call's worst "
                                    f"case ${cand[0]['reserve_usd']:.4f} would pass it; {len(queue)} calls not sent")
                break
            queue = queue[len(fit):]
            wait = max(r["tries"] for r in fit)
            if wait:
                self.sleep(backoff(wait, self.cfg))
            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                recs = list(ex.map(self.attempt, fit))
            self.stats["chunks"] += 1
            retry, answered = [], 0
            for r, rec in zip(fit, recs):
                self.counted_run += rec["charged_usd"]
                self.stats["attempts"] += 1
                self.stats["billed_usd"] += float(rec.get("cost_usd") or 0.0)
                self.stats["counted_usd"] += rec["charged_usd"]
                o = rec["outcome"]
                if o in ACCEPTED:
                    answered += 1
                    self.stats["answered"] += 1
                    self.stats["refusals"] += o == "refusal"
                    self.stats["blocked"] += o == "blocked"
                    if self.monitor is not None and self.stop_reason is None:
                        why = self.monitor(rec)
                        if why:
                            self.stop_reason = why
                elif o == "flagged":
                    self.stats["flagged"] = self.stats.get("flagged", 0) + 1
                    if self.stop_reason is None:
                        self.stop_reason = f"{rec['key']}: " + "; ".join(rec["problems"])
                elif o == "failed":
                    self.stats["failed_attempts"] += 1
                    r["tries"] += 1
                    if r["tries"] < self.max_tries:
                        retry.append(r)
                    else:
                        self.stats["missing"] += 1
                else:
                    self.stats["errors"] += 1
                    if self.stop_on_error and self.stop_reason is None:
                        self.stop_reason = (f"HTTP {rec['status']} on {rec['key']} (not retried): "
                                            f"{redact(rec.get('error'), self.secret)[:400]}")
            queue = retry + queue
            streak = 0 if answered else streak + 1
            if streak >= self.max_failed_chunks and self.stop_reason is None:
                self.stop_reason = f"{streak} chunks in a row with no answered call"
            self.log(f"[{time.strftime('%H:%M:%S')}] {self.phase}: chunk {self.stats['chunks']}, answered "
                     f"{self.stats['answered']}, failed attempts {self.stats['failed_attempts']}, left {len(queue)}; "
                     f"billed ${self.stats['billed_usd']:.4f}, counted ${self.counted:.4f} of ${self.limit:.2f}")
        if self.stop_reason is not None:
            self.stats["not_sent"] = len(queue)
        return {**self.stats, "stop_reason": self.stop_reason, "seconds": round(time.time() - t0, 1),
                "session": self.session, "counted_total_usd": self.counted}


# ----------------------------------------------------------------------------------------------------------------
# Tokenizer and scoring
# ----------------------------------------------------------------------------------------------------------------

class Tok:
    """The one tokenizer that recounts every visible text (config `tokenizer`), checked by sha256."""

    def __init__(self, cfg: dict, root: Path = ROOT):
        t = cfg["tokenizer"]
        self.path, self.sha, self.name = Path(root) / t["path"], t["sha256"], t["name"]
        self._t = None

    def _load(self):
        if self._t is None:
            if not self.path.exists():
                raise SystemExit(f"tokenizer {rel(self.path)} is absent")
            if sha256_file(self.path) != self.sha:
                raise SystemExit(f"tokenizer {rel(self.path)}: sha256 differs from the config")
            from tokenizers import Tokenizer
            self._t = Tokenizer.from_file(str(self.path))
        return self._t

    def count(self, text) -> int:
        return len(self._load().encode(text if isinstance(text, str) else "", add_special_tokens=False).ids)

    def counts(self, texts) -> list[int]:
        t = self._load()
        xs = [x if isinstance(x, str) else "" for x in texts]
        return [len(e.ids) for e in t.encode_batch(xs, add_special_tokens=False)] if xs else []


SCORE_COLS = ("strict_correct", "lenient_correct", "compliant", "content_correct", "content_class", "content_f1",
              "strict_f1", "lenient_f1")


FORCED_READINGS = ("as_returned", "cut_after_first_brace", "stop_restored")


def forced_reply(text, task: str, finish_reason, cfg: dict, stop_sent: bool) -> tuple[str, str]:
    """A forced reply as the panel's runner would have stored it (output_text, without the prefill), and how it was
    read. On the benchmarks whose panel runs stopped at the first "}" (kept in the output): the reply cut after its
    first "}" (a route that returns the stop string, or a reply sent without the stop); else, when the stop was sent
    and the reply ended on it (finish reason "stop") and holds no "}", the route left the stop string out and the "}"
    is put back ("stop_restored"; a reply that ended its turn inside the open box reads the same, which the pilot's
    stop control bounds). A reply cut by the cap stays as it is. MATH-500 and HotpotQA replies are never changed."""
    t = text if isinstance(text, str) else ""
    if task not in cfg["forced_cut_at_first_brace"]:
        return t, "as_returned"
    j = t.find("}")
    if j >= 0:
        return t[: j + 1], ("cut_after_first_brace" if j + 1 < len(t) else "as_returned")
    if stop_sent and finish_reason == "stop":
        return t + "}", "stop_restored"
    return t, "as_returned"


def row_arm(x) -> str:
    a = x.get("arm") if isinstance(x, dict) else None
    return a if isinstance(a, str) and a in ARMS else "requested"


def score_rows(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The panel's scorers (scripts/nebius_panel_report.score_task) on replies of the three arms: df needs model_key,
    task, request_id, item_id, text, finish_reason and (else requested) arm; forced rows also stop_sent. A reply with
    no text (a refusal) is scored as an empty reply. A forced reply goes in as the panel's runner stores one
    (forced_reply: without the prefill); score_task puts the prefill in front once and reads a box the stop left open
    through src/forced_stop.py, as for the panel's API models. Adds forced_text and forced_reading."""
    import nebius_panel_report as npr
    cfg = cfg or load_config()
    versions, parts = {}, []
    ids = df["key"] if "key" in df.columns else df["request_id"]      # the scorers need unique ids (pilot repeats)
    arms = df["arm"].map(lambda a: a if isinstance(a, str) and a in ARMS else "requested") if "arm" in df.columns \
        else pd.Series("requested", index=df.index)
    texts = df["text"].map(lambda s: s if isinstance(s, str) else "")
    fins = df["finish_reason"].map(lambda s: s if isinstance(s, str) else "")
    stops = df["stop_sent"].fillna(False).astype(bool) if "stop_sent" in df.columns \
        else pd.Series(False, index=df.index)
    out_text, reading = texts.copy(), pd.Series(None, index=df.index, dtype=object)
    for ix in df.index[arms == "forced"]:
        out_text.at[ix], reading.at[ix] = forced_reply(texts.at[ix], df.at[ix, "task"], fins.at[ix], cfg,
                                                       bool(stops.at[ix]))
    for (mk, t), g in df.groupby(["model_key", "task"], sort=False):
        o = pd.DataFrame({"_row": g.index.to_numpy(), "request_id": ids.loc[g.index].to_numpy(),
                          "item_id": g["item_id"].astype(str).to_numpy(),
                          "arm": arms.loc[g.index].map(ARM_SOURCE).to_numpy(),
                          "output_text": out_text.loc[g.index].to_numpy(),
                          "finish_reason": fins.loc[g.index].to_numpy()})
        sc, ver = npr.score_task(o, t)
        versions[t] = ver
        parts.append(sc.set_index("_row")[[c for c in SCORE_COLS if c in sc.columns]])
    scores = pd.concat(parts) if parts else pd.DataFrame(index=df.index)
    out = df.drop(columns=[c for c in SCORE_COLS if c in df.columns]).join(scores)
    for c in SCORE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out["forced_text"] = np.where(arms == "forced", out_text, None)
    out["forced_reading"] = reading
    return out, versions


def api_count(rec: dict):
    """Visible output tokens by the API: completion minus reasoning tokens (reasoning is 0 with thinking off)."""
    ct = rec.get("completion_tokens")
    if ct is None or pd.isna(ct):
        return None
    rt = rec.get("reasoning_tokens")
    return int(ct) - (0 if rt is None or pd.isna(rt) else int(rt))


def thresholds(cfg: dict, task: str) -> list[int]:
    th = cfg["thresholds"]
    return [int(x) for x in th.get(task, th["default"])]


# ----------------------------------------------------------------------------------------------------------------
# Pilot
# ----------------------------------------------------------------------------------------------------------------

def rows_by_id(df: pd.DataFrame, wanted: list[dict]) -> list[dict]:
    d = df.set_index("request_id")
    out = []
    for w in wanted:
        if w["request_id"] not in d.index:
            raise SystemExit(f"{w['request_id']} is not in the source file")
        r = d.loc[w["request_id"]]
        if sha256_text(r["text"]) != w["text_sha256"]:
            raise SystemExit(f"{w['request_id']}: the message differs from the manifest's")
        out.append({"request_id": w["request_id"], "config_id": r["config_id"], "item_id": r["item_id"],
                    "text": r["text"]})
    return out


def pilot_requests(cfg: dict, man: dict, sources: dict, control_src: pd.DataFrame, models: list[dict],
                   arms=None) -> list[dict]:
    """Per repeat, per benchmark and per pilot cell the run's arms back to back, for every model (so the two sends of
    one prompt are apart in time); then the stop control (the forced pilot prompts of the benchmarks whose stop is
    sent, once more without it); then the thinking-on controls. sources: {arm: {benchmark: rows}}, or the requested
    arm's {benchmark: rows} alone."""
    arms = list(arms or run_arms(cfg))
    if set(sources) <= set(TASKS):
        sources = {"requested": sources}
    reps = int(cfg["design"]["pilot_repeats"])
    rows = {}
    for a in arms:
        wanted = man["pilot"]["rows"] if a == "requested" else man["arms"]["pilot"][a]
        rows[a] = {t: rows_by_id(sources[a][t], wanted[t]) for t in TASKS}
    out = []
    for rep in range(reps):
        for t in TASKS:
            for j in range(len(rows[arms[0]][t])):
                for a in arms:
                    for m in models:
                        out.append(make_request(cfg, m, t, rows[a][t][j], a, rep=rep))
    if "forced" in arms and cfg["design"].get("pilot_stop_controls"):
        for t in TASKS:
            if forced_stop_sent(cfg, t):
                for r in rows["forced"][t]:
                    for m in models:
                        out.append(make_request(cfg, m, t, r, "forced_nostop", rep=0))
    c = rows_by_id(control_src, [man["control"]["row"]])[0]
    for rep in range(int(cfg["design"]["pilot_controls"])):
        for m in models:
            out.append(make_request(cfg, m, "math", c, "control", rep=rep))
    return out


def preflight(http, cfg: dict, model: dict) -> dict:
    """Free read of the pinned endpoint's public listing: present, every parameter the design sends supported (the
    stop included when the forced arm sends it), list price as in the config (within 1%)."""
    url = cfg["base_url"].rstrip("/") + f"/models/{model['endpoints_slug']}/endpoints"
    resp = http.get_public(url)
    eps = (((resp.get("json") or {}).get("data") or {}).get("endpoints") or []) if resp.get("status") == 200 else []
    ep = next((e for e in eps if e.get("tag") == model["endpoint_tag"]), None)
    detail, ok = {"url": url, "status": resp.get("status")}, True
    if ep is None:
        return {"name": "preflight", "pass": False,
                "detail": {**detail, "problem": f"no endpoint tagged {model['endpoint_tag']!r} in the listing "
                                                f"(tags {[e.get('tag') for e in eps]})"}}
    sup = set(ep.get("supported_parameters") or [])
    sent = design_parameters(cfg, model)
    missing = [p for p in sent if p not in sup]
    pr = ep.get("pricing") or {}
    price_diff = {}
    for k in ("prompt", "completion"):
        try:
            v = float(pr.get(k))
        except (TypeError, ValueError):
            v = None
        want = float(model["price"][k])
        if v is None or abs(v - want) > 0.01 * want:
            price_diff[k] = {"listed": v, "config": want}
    detail.update(provider=ep.get("provider_name"), tag=ep.get("tag"), quantization=ep.get("quantization"),
                  max_completion_tokens=ep.get("max_completion_tokens"), pricing=pr,
                  supported_parameters=sorted(sup), parameters_sent=sent)
    if missing:
        ok = False
        detail["problem_parameters"] = (f"the endpoint does not list {missing}; with require_parameters the route "
                                        "would refuse: drop them from the config only deliberately")
    if price_diff:
        ok = False
        detail["problem_price"] = price_diff
    return {"name": "preflight", "pass": ok, "detail": detail}


def _check(name: str, ok: bool, detail) -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def forced_check(text, task: str, finish_reason, cfg: dict, stop_sent: bool, reasoning_tokens=None,
                 reasoning_text=None, completion_tokens=None) -> dict:
    """The prefill test's continuation rule on a forced reply read as the panel's (forced_reply)."""
    r, how = forced_reply(text, task, finish_reason, cfg, stop_sent)
    c = prefill_check(r, task, cfg, False, reasoning_tokens, reasoning_text, completion_tokens)
    c["reading"] = how
    return c


def _answered(x) -> bool:
    return x is not None and x.get("outcome") in ("reply", "refusal")


def forced_checks(cfg: dict, mine: list[dict], got: dict) -> list[dict]:
    """The forced arm's pilot checks: the replies continue the box the prefill opened (as in the prefill test), and
    the stop control: with the stop, read as the panel's, equal to the same request sent without it."""
    ck = cfg["checks"]
    fr = [r for r in mine if r["kind"] == "forced"]
    if not fr:
        return []
    per, ok = {}, True
    for t in TASKS:
        rows = []
        for r in fr:
            if r["task"] != t:
                continue
            x = got.get(r["key"])
            if x is None or x.get("outcome") != "reply":
                rows.append({"key": r["key"], "pass": False, "reasons": ["no reply"], "reading": None,
                             "starts_with_prefill": False})
                continue
            c = forced_check(x.get("text"), t, x.get("finish_reason"), cfg, r["stop_sent"], x.get("reasoning_tokens"),
                             x.get("reasoning_text"), x.get("completion_tokens"))
            rows.append({"key": r["key"], "text": x.get("text"), "finish_reason": x.get("finish_reason"), **c})
        share = float(np.mean([z["pass"] for z in rows])) if rows else 0.0
        p = (bool(rows) and all(z["reasons"] != ["no reply"] for z in rows)
             and share >= ck["prefill_min_continuation"])
        ok = ok and p
        per[t] = {"calls": len(rows), "continues": share, "pass": p,
                  "starts_with_prefill": sum(bool(z.get("starts_with_prefill")) for z in rows),
                  "readings": {k: sum(z.get("reading") == k for z in rows) for k in FORCED_READINGS},
                  "failures": [{"key": z["key"], "reasons": z["reasons"],
                                "text": (z.get("text") or "")[:120] if isinstance(z.get("text"), str) else None}
                               for z in rows if not z["pass"]][:3]}
    out = [_check("forced_continuation", ok,
                  {"per_benchmark": per, "rule": f"every forced call answered and, on every benchmark, at least "
                                                 f"{ck['prefill_min_continuation']:.0%} of the replies, read as the "
                                                 "panel's, continue the open box (the prefill test's rule: no new "
                                                 "\\boxed, think tag or turn token, no reasoning, something inside the "
                                                 "box, no line break before it closes, the prefix once)"})]
    nostop = {r["request_id"] + "|" + r["task"]: r for r in mine if r["kind"] == "forced_nostop"}
    if nostop:
        pairs, agree, returned, restored, after = [], [], 0, 0, 0
        for r in fr:
            if r["rep"] != 0 or not r["stop_sent"]:
                continue
            q = nostop.get(r["request_id"] + "|" + r["task"])
            a, b = got.get(r["key"]), got.get(q["key"]) if q else None
            if q is None or not _answered(a) or not _answered(b):
                pairs.append({"key": r["key"], "same": False, "why": "unanswered"})
                agree.append(False)
                continue
            ta = a.get("text") if isinstance(a.get("text"), str) else ""
            ra, how = forced_reply(ta, r["task"], a.get("finish_reason"), cfg, True)
            rb, _ = forced_reply(b.get("text"), r["task"], b.get("finish_reason"), cfg, False)
            returned += "}" in ta
            after += "}" in ta and not ta.endswith("}")
            restored += how == "stop_restored"
            same = ra == rb
            agree.append(same)
            pairs.append({"key": r["key"], "same": same, "with_stop": ta[:80], "read": ra[:80],
                          "without_stop": (b.get("text") or "")[:80], "read_without": (rb or "")[:80]})
        share = float(np.mean(agree)) if agree else 0.0
        out.append(_check("forced_stop_control", bool(agree) and share >= ck["stop_control_min_agreement"],
                          {"pairs": len(agree), "agreement": share, "stop_string_returned": returned,
                           "text_after_the_stop": after, "stop_restored": restored,
                           "disagreements": [p for p in pairs if not p["same"]][:5],
                           "rule": "forced pilot prompts of the benchmarks whose stop is sent: the reply sent with "
                                   "the stop, read as the panel's (cut after its first '}', or the '}' put back when "
                                   "the reply ended on the stop), equals the reply sent without the stop cut after its "
                                   f"first '}}' on at least {ck['stop_control_min_agreement']:.0%} of the pairs"}))
    return out


def pilot_model_checks(cfg: dict, model: dict, reqs: list[dict], fin: dict, scored: pd.DataFrame | None) -> list[dict]:
    ck = cfg["checks"]
    mine = [r for r in reqs if r["model_key"] == model["key"]]
    off = [r for r in mine if r["kind"] in ARMS or r["kind"] == "forced_nostop"]     # thinking off
    ctl = [r for r in mine if r["kind"] == "control"]
    got = {r["key"]: fin.get(r["key"]) for r in mine}
    out = []
    unanswered = [k for k, v in got.items() if not _answered(v)]
    out.append(_check("answered", not unanswered,
                      {"calls": len(mine), "unanswered": unanswered[:10], "n_unanswered": len(unanswered)}))
    ro = [got[r["key"]] for r in off if got[r["key"]] is not None and got[r["key"]].get("outcome") in ACCEPTED]
    bad = [x["key"] for x in ro if x.get("reasoning_tokens") is None or int(x["reasoning_tokens"]) != 0
           or (x.get("reasoning_text") or "").strip() or int(x.get("reasoning_details_n") or 0)]
    out.append(_check("reasoning_off", bool(ro) and not bad,
                      {"calls": len(ro), "with_reasoning_or_no_count": bad[:10],
                       "rule": "reasoning_tokens reported and 0, no reasoning text or details, on every thinking-off call"}))
    rc = [got[r["key"]] for r in ctl]
    cbad = [r["key"] for r, x in zip(ctl, rc) if x is None or not int(x.get("reasoning_tokens") or 0) > 0]
    out.append(_check("reasoning_on_control", bool(ctl) and not cbad,
                      {"calls": len(ctl), "reasoning_tokens": [None if x is None else x.get("reasoning_tokens")
                                                               for x in rc], "failed": cbad,
                       "rule": "reasoning_tokens above 0 on every thinking-on control: the route shows reasoning"}))
    allx = [x for x in got.values() if _answered(x)]
    pbad = [x["key"] for x in allx if x.get("provider") != model["expected_provider"]
            or not str(x.get("served_model") or "").startswith(model["served_model_prefix"])]
    out.append(_check("provider", bool(allx) and not pbad,
                      {"providers": sorted({str(x.get("provider")) for x in allx}),
                       "served_models": sorted({str(x.get("served_model")) for x in allx}), "off_route": pbad[:10]}))
    fr = {}
    for x in ro:
        fr[str(x.get("finish_reason"))] = fr.get(str(x.get("finish_reason")), 0) + 1
    out.append(_check("finish_reasons", bool(ro) and all(k in FINISH_OK for k in fr), {"counts": fr}))
    # token accounting, every thinking-off reply
    ov = int(cfg["run"]["prompt_overhead_tokens"])
    rows, abad = [], []
    for r in off:
        x = got[r["key"]]
        if not _answered(x):
            continue
        n_api = api_count(x)
        loc = x.get("n_local")
        if x.get("prompt_tokens") is None or n_api is None:
            abad.append(f"{r['key']}: usage absent")
            continue
        if int(x["completion_tokens"]) > r["cap"]:
            abad.append(f"{r['key']}: {x['completion_tokens']} completion tokens above the cap {r['cap']}")
        if int(x["prompt_tokens"]) > r["n_bytes"] + ov:
            abad.append(f"{r['key']}: {x['prompt_tokens']} prompt tokens above the worst-case bound "
                        f"{r['n_bytes'] + ov}")
        within = loc is not None and abs(n_api - loc) <= max(ck["accounting_abs_tol"], ck["accounting_rel_tol"] * loc)
        rows.append({"key": r["key"], "arm": r["arm"], "kind": r["kind"], "task": r["task"], "api": n_api,
                     "local": loc, "within": bool(within)})
    share = float(np.mean([z["within"] for z in rows])) if rows else 0.0
    by_arm = {}
    for z in rows:
        if z["local"] is not None and z["kind"] in ARMS:
            by_arm.setdefault(z["arm"], []).append(z["api"] - z["local"])
    cross = [all((z["api"] > k) == (z["local"] > k) for k in thresholds(cfg, z["task"]))
             for z in rows if z["local"] is not None and z["kind"] == "requested"]
    out.append(_check("token_accounting", bool(rows) and not abad and share >= ck["accounting_min_share"],
                      {"replies": len(rows), "problems": abad[:10], "share_within_tolerance": share,
                       "median_api_minus_recount": {a: float(np.median(v)) for a, v in by_arm.items()},
                       "crossing_agreement_api_vs_recount": float(np.mean(cross)) if cross else None,
                       "rule": f"usage on every reply; completion <= the arm's cap; prompt <= bytes + {ov}; "
                               f"|API - recount| <= max({ck['accounting_abs_tol']}, {ck['accounting_rel_tol']} x "
                               f"recount) on at least {ck['accounting_min_share']:.0%} of replies"}))
    # repeats, per arm
    sc = scored.set_index("key") if scored is not None and len(scored) else None
    per_arm, ok_rep, prow = {}, True, []
    for arm in [a for a in ARMS if any(r["kind"] == a for r in off)]:
        pairs = {}
        for r in off:
            if r["kind"] == arm:
                pairs.setdefault(r["key"].rsplit("|rep", 1)[0], {})[r["rep"]] = (r, got[r["key"]])
        text_eq, cross_eq, strict_eq, content_eq = [], [], [], []
        for base, d in pairs.items():
            if 0 not in d or 1 not in d:
                continue
            (ra, a), (_, b) = d[0], d[1]
            if a is None or b is None or a.get("outcome") not in ACCEPTED or b.get("outcome") not in ACCEPTED:
                continue
            na, nb = api_count(a), api_count(b)
            te = (a.get("text") or "") == (b.get("text") or "")
            ce = na is not None and nb is not None and all((na > k) == (nb > k) for k in thresholds(cfg, ra["task"]))
            se = co = None
            if sc is not None and a["key"] in sc.index and b["key"] in sc.index:
                se = bool(sc.at[a["key"], "strict_correct"] == sc.at[b["key"], "strict_correct"])
                co = bool(sc.at[a["key"], "content_correct"] == sc.at[b["key"], "content_correct"])
            text_eq.append(te)
            cross_eq.append(ce)
            if se is not None:
                strict_eq.append(se)
                content_eq.append(co)
            prow.append({"prompt": base, "arm": arm, "same_text": te, "same_crossing": ce, "same_strict": se,
                         "same_content": co, "tokens": [na, nb]})
        agree = {"pairs": len(text_eq), "text": float(np.mean(text_eq)) if text_eq else None,
                 "crossing": float(np.mean(cross_eq)) if cross_eq else None,
                 "strict": float(np.mean(strict_eq)) if strict_eq else None,
                 "content": float(np.mean(content_eq)) if content_eq else None}
        ok = (agree["pairs"] > 0 and agree["strict"] is not None and agree["strict"] >= ck["repeat_min_verdict"]
              and agree["content"] is not None and agree["content"] >= ck["repeat_min_verdict"])
        if arm == "requested":
            ok = ok and agree["crossing"] is not None and agree["crossing"] >= ck["repeat_min_crossing"]
        if ck.get("repeat_min_text") is not None:
            ok = ok and agree["text"] is not None and agree["text"] >= ck["repeat_min_text"]
        per_arm[arm] = {**agree, "pass": ok}
        ok_rep = ok_rep and ok
    out.append(_check("repeat_agreement", bool(per_arm) and ok_rep,
                      {"per_arm": per_arm, "pairs": sum(v["pairs"] for v in per_arm.values()),
                       "strict": (per_arm.get("requested") or {}).get("strict"),
                       "content": (per_arm.get("requested") or {}).get("content"), "rows": prow,
                       "rule": f"per arm, the strict and content verdicts agree on at least "
                               f"{ck['repeat_min_verdict']:.0%} of the pairs; on the requested arm also the crossing "
                               f"of {{16; 64 on HotpotQA}} tokens on at least {ck['repeat_min_crossing']:.0%}; exact "
                               "text reported" + ("" if ck.get("repeat_min_text") is None else
                                                  f" and required at {ck['repeat_min_text']:.0%}")}))
    out += forced_checks(cfg, mine, got)
    # cost per call
    cbad, ratios = [], []
    for x in allx:
        if x.get("cost_reported") is None:
            cbad.append(f"{x['key']}: no billed cost in the reply")
            continue
        e = expected_cost(x, model["price"])
        if e > 0:
            ratio = float(x["cost_reported"]) / e
            ratios.append(ratio)
            if abs(ratio - 1) > cfg["checks"]["cost_rel_tol"]:
                cbad.append(f"{x['key']}: billed ${float(x['cost_reported']):.6f}, tokens x prices ${e:.6f}")
    out.append(_check("cost_per_call", bool(allx) and not cbad,
                      {"calls": len(allx), "problems": cbad[:10],
                       "billed_over_expected": {"min": min(ratios) if ratios else None,
                                                "max": max(ratios) if ratios else None},
                       "mean_cost_per_call": float(np.mean([float(x.get("cost_reported") or 0) for x in allx]))
                       if allx else None}))
    return out


def projection(cfg: dict, man: dict, reqs: list[dict], fin: dict, models: list[dict] | None = None) -> dict:
    """Full run cost per model from the pilot's own replies: per arm and benchmark the mean billed cost of the
    arm's pilot calls times the arm's calls on the benchmark (high: the largest pilot cost there times its calls)."""
    out = {"models": {}, "total": 0.0, "total_high": 0.0}
    arms = sorted({r["kind"] for r in reqs if r["kind"] in ARMS}, key=ARMS.index)
    for m in (models or [model_of(cfg, k) for k in default_model_keys(cfg)]):
        per = {}
        for a in arms:
            for t in TASKS:
                cs = [float(fin[r["key"]]["cost_usd"]) for r in reqs
                      if r["model_key"] == m["key"] and r["task"] == t and r["kind"] == a
                      and fin.get(r["key"]) is not None and fin[r["key"]].get("cost_usd") is not None]
                n = man["tasks"][t]["n_requests"]
                per[f"{a}|{t}"] = {"arm": a, "task": t, "pilot_calls": len(cs), "mean": float(np.mean(cs)) if cs else None,
                                   "calls": n, "projected": float(np.mean(cs) * n) if cs else None,
                                   "high": float(np.max(cs) * n) if cs else None}
        if per and all(v["projected"] is not None for v in per.values()):
            tot, hi = sum(v["projected"] for v in per.values()), sum(v["high"] for v in per.values())
        else:
            tot = hi = None
        by_arm = {a: (sum(v["projected"] for v in per.values() if v["arm"] == a) if tot is not None else None)
                  for a in arms}
        out["models"][m["key"]] = {"per_task": per, "per_arm": by_arm, "projected": tot, "high": hi, "arms": arms}
        if tot is None:
            out["total"] = out["total_high"] = None
        elif out["total"] is not None:
            out["total"] += tot
            out["total_high"] += hi
    return out


def render_pilot(res: dict) -> str:
    L = []
    w = L.append
    w(f"# Qwen4-Max three arms: pilot ({res['made_on']})")
    w("")
    w(f"Written by `scripts/closed_w.py pilot`; configuration `config/closed_w.yaml` (hash `{res['config_sha'][:12]}`); "
      f"raw replies `{res['raw']}`. Thinking off on every call except the two controls. Verdict: "
      f"**{'PASS' if res['passed'] else 'FAIL'}**." + ("" if res["passed"] else
                                                     " `run` refuses until a pilot under this configuration passes."))
    w("")
    w(f"Spend: billed {usd(res['spend']['billed_usd'])} this pilot file in all, counted against the ${res['limit']:.2f} "
      f"limit {usd(res['spend']['counted_usd'])}.")
    w("")
    for mk, r in res["models"].items():
        w(f"## {r['label']} (`{r['id']}`): {'pass' if r['passed'] else 'FAIL'}")
        w("")
        w("| Check | Result | Detail |")
        w("|---|---|---|")
        for c in r["checks"]:
            d = {k: v for k, v in c["detail"].items() if k not in ("rows", "supported_parameters", "pricing")} \
                if isinstance(c["detail"], dict) else c["detail"]
            w(f"| {c['name']} | {'pass' if c['pass'] else '**FAIL**'} | "
              f"{json.dumps(d, ensure_ascii=False, default=str)[:700].replace('|', '/')} |")
        w("")
        w("| Benchmark | Arm | Prompt | Rep | Outcome | Finish | Prompt tok. | Output tok. (API) | Recount | "
          "Reasoning tok. | Billed | Provider |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in r["calls"]:
            w(f"| {x['task']} | {x['kind']} | `{x['request_id']}` | {x['rep']} | {x.get('outcome')} | "
              f"{x.get('finish_reason')} | {x.get('prompt_tokens')} | {x.get('n_api')} | {x.get('n_local')} | "
              f"{x.get('reasoning_tokens')} | {usd(x.get('cost_usd'))} | {x.get('provider')} |")
        w("")
    pj = res["projection"]
    w("## Projected cost of the full run (from the pilot's own replies)")
    w("")
    w("| Model | Arm | GSM8K | MATH-500 | BIG-Bench Hard | HotpotQA | Arm total |")
    w("|---|---|---|---|---|---|---|")
    for mk, v in pj["models"].items():
        for a in v["arms"]:
            w(f"| {mk} | {a} | " + " | ".join(usd(v["per_task"][f"{a}|{t}"]["projected"]) for t in TASKS) +
              f" | {usd(v['per_arm'][a])} |")
        w(f"| {mk} | **all arms** | | | | | **{usd(v['projected'])}** (high {usd(v['high'])}) |")
    w("")
    for c in res["global_checks"]:
        w(f"- {c['name']}: {'pass' if c['pass'] else '**FAIL**'} ({json.dumps(c['detail'], default=str)})")
    w("")
    if not res["passed"]:
        w("## Failed checks")
        w("")
        for f in res["failed"]:
            w(f"- {f}")
        w("")
    w("The pilot's w, accuracy and gains are not checks: they never decide whether the run proceeds.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Prefill test
# ----------------------------------------------------------------------------------------------------------------

def close_index(text: str):
    """Index of the '}' that closes the box opened by the prefill (brace depth from 1 back to 0), else None."""
    depth = 1
    for i, ch in enumerate(text or ""):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def prefill_check(text, task: str, cfg: dict, echo_mode: bool = False, reasoning_tokens=None, reasoning_text=None,
                  completion_tokens=None) -> dict:
    """Does a reply continue the box the prefill opened? A continuation holds no restart (a new \\boxed, a think tag or
    a turn token), no reasoning, is not empty and has no line break before the box closes (or before the cap). The
    box closes at the first '}' on the benchmarks whose forced runs stopped there (GSM8K, BIG-Bench Hard), elsewhere
    by brace depth. A reply that starts with the prefix fails: it is a restart, or the route returning the prefix with
    the continuation, and the text cannot tell them apart. echo_mode=True (never set by the test; for a re-reading
    once the route is known to echo) removes the echoed prefix first, so the scored text holds the prefix once."""
    p = cfg["prefill"]
    t = text if isinstance(text, str) else ""
    starts = t.startswith(p)
    cont = t[len(p):] if (echo_mode and starts) else t
    reasons = []
    if starts and not echo_mode:
        reasons.append("starts with the prefix itself (a restart, or an echo the text cannot tell apart)")
    if reasoning_tokens:
        reasons.append(f"{reasoning_tokens} reasoning tokens")
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        reasons.append("reasoning text returned")
    marks = [m for m in RESTART_MARKS if m in cont]
    if marks:
        reasons.append(f"restart: {marks}")
    cut_first = task in cfg["forced_cut_at_first_brace"]
    j = cont.find("}") if cut_first else close_index(cont)
    closed = j is not None and j >= 0
    before = cont[:j] if closed else cont
    if not before.strip():
        reasons.append("nothing inside the box")
    if "\n" in before.strip():
        reasons.append("a line break inside the box")
    cap = int(cfg["caps"]["forced"][task])
    if completion_tokens is not None and int(completion_tokens) > cap:
        reasons.append(f"{completion_tokens} tokens, above the cap {cap}")
    scoring = p + (cont[: j + 1] if (closed and cut_first) else cont)
    if scoring.count("\\boxed") != 1:
        reasons.append("the prefix would be counted more than once")
    return {"pass": not reasons, "reasons": reasons, "closed": bool(closed), "answer": before.strip() if closed else None,
            "starts_with_prefill": starts, "echo_removed": bool(echo_mode and starts), "scoring_text": scoring}


def prefill_requests(cfg: dict, man: dict, screening: dict, model: dict) -> list[dict]:
    out = []
    for v in cfg["design"]["prefill_variants"]:
        for t in TASKS:
            for r in rows_by_id(screening[t], man["prefill"]["rows"][t]):
                out.append(make_request(cfg, model, t, r, "prefill", variant=v))
    return out


def prefill_summary(cfg: dict, reqs: list[dict], fin: dict) -> dict:
    res = {"variants": {}, "passed": False, "passing_variants": []}
    for v in cfg["design"]["prefill_variants"]:
        mine = [r for r in reqs if r["variant"] == v["name"]]
        recs = [(r, fin.get(r["key"])) for r in mine]
        # A reply that starts with the prefix is never read as an echo: a route that drops the prefill gives the same
        # text, since the prompt itself asks for a box. Such replies fail (restart, or an echo the text cannot tell
        # apart from one) and are counted in the report.
        echo = False
        per, examples, failures = {}, {}, {}
        for t in TASKS:
            rows = []
            for r, x in recs:
                if r["task"] != t:
                    continue
                if x is None or x.get("outcome") != "reply":
                    rows.append({"request_id": r["request_id"], "outcome": None if x is None else x.get("outcome"),
                                 "status": None if x is None else x.get("status"),
                                 "error": None if x is None else (x.get("error") or "")[:300], "pass": False,
                                 "reasons": ["no reply"]})
                    continue
                c = prefill_check(x.get("text"), t, cfg, echo, x.get("reasoning_tokens"), x.get("reasoning_text"),
                                  x.get("completion_tokens"))
                rows.append({"request_id": r["request_id"], "config_id": r["config_id"], "outcome": "reply",
                             "text": x.get("text"), "finish_reason": x.get("finish_reason"),
                             "completion_tokens": x.get("completion_tokens"),
                             "reasoning_tokens": x.get("reasoning_tokens"), **c})
            n = len(rows)
            share = float(np.mean([z["pass"] for z in rows])) if rows else 0.0
            per[t] = {"n": n, "replies": sum(z["outcome"] == "reply" for z in rows), "continues": share,
                      "closed": float(np.mean([z.get("closed", False) for z in rows])) if rows else 0.0,
                      "starts_with_prefill": sum(bool(z.get("starts_with_prefill")) for z in rows),
                      "pass": n > 0 and all(z["outcome"] == "reply" for z in rows)
                      and share >= cfg["checks"]["prefill_min_continuation"]}
            examples[t] = [z for z in rows if z["outcome"] == "reply"][:3]
            failures[t] = [z for z in rows if not z["pass"]][:5]
        ok = all(per[t]["pass"] for t in TASKS)
        res["variants"][v["name"]] = {"message_extra": v.get("message_extra") or {}, "echo_mode": echo, "per_task": per,
                                      "pass": ok, "examples": examples, "failures": failures}
        if ok:
            res["passing_variants"].append(v["name"])
    res["passed"] = bool(res["passing_variants"])
    return res


def render_prefill(res: dict) -> str:
    L = []
    w = L.append
    s = res["summary"]
    w(f"# Qwen4-Max forced-arm prefill test ({res['made_on']})")
    w("")
    w(f"Written by `scripts/closed_w.py prefill-test`. Model `{res['model']}` on OpenRouter, provider pinned "
      f"({res['provider']}), thinking off, temperature 0. Each call sends a calibration item's requested-arm user message "
      f"(training split, a run template) and an assistant message holding the panel's prefill `{res['prefill']}`; the "
      f"output cap is the panel's forced cap (16 tokens; 64 on HotpotQA) and no stop string is sent (GSM8K and BIG-Bench "
      f"Hard replies are read up to their first `}}`, the panel's stop). Raw replies `{res['raw']}`.")
    w("")
    w(f"**Verdict: {'PASS' if s['passed'] else 'FAIL'}**" +
      (f" (passing variant{'s' if len(s['passing_variants']) > 1 else ''}: "
       f"{', '.join('`' + x + '`' for x in s['passing_variants'])})." if s["passed"] else ".") +
      " A variant passes when every call returns a reply and, on every benchmark, at least "
      f"{res['min_share']:.0%} of replies continue the open box: no new `\\boxed`, think tag or turn token, no "
      "reasoning, something written inside the box, no line break before it closes, the prefix counted once in the "
      "scored text. Whether the box closes within the cap is shown beside it and is not part of the verdict.")
    w("")
    w(f"Spend: billed {usd(res['spend']['billed_usd'])}, counted {usd(res['spend']['counted_usd'])} of the "
      f"${res['limit']:.2f} limit. No forced run follows without a separate quote.")
    w("")
    for name, v in s["variants"].items():
        w(f"## Variant `{name}`: {'pass' if v['pass'] else 'FAIL'}")
        w("")
        n_start = sum(v["per_task"][t]["starts_with_prefill"] for t in TASKS)
        w(f"Assistant message extra fields: `{json.dumps(v['message_extra'])}`. Replies that start with the prefix "
          f"itself: {n_start}. Such a reply is a restart, or the route returning the prefix with its continuation; "
          "the text cannot tell the two apart (the prompt asks for a box, so a route that drops the prefill gives the "
          "same text), so it counts as no continuation.")
        w("")
        w("| Benchmark | Calls | Replies | Continue the box | Box closed within the cap | Start with the prefix | Pass |")
        w("|---|---|---|---|---|---|---|")
        for t in TASKS:
            p = v["per_task"][t]
            w(f"| {TASK_NAME[t]} | {p['n']} | {p['replies']} | {pct(p['continues'])}% | {pct(p['closed'])}% | "
              f"{p['starts_with_prefill']} | {'yes' if p['pass'] else 'no'} |")
        w("")
        w("Examples (first three replies per benchmark; text as returned):")
        w("")
        for t in TASKS:
            for z in v["examples"][t]:
                w(f"- {TASK_NAME[t]} `{z['request_id']}`: `{json.dumps(z.get('text'), ensure_ascii=False)[:200]}` "
                  f"-> scored `{z.get('scoring_text', '')[:120]}`; {'continues' if z['pass'] else 'no: ' + '; '.join(z['reasons'])}")
        fails = [(t, z) for t in TASKS for z in v["failures"][t]]
        if fails:
            w("")
            w("Failures (up to five per benchmark):")
            w("")
            for t, z in fails:
                w(f"- {TASK_NAME[t]} `{z['request_id']}`: {'; '.join(z['reasons'])}; "
                  f"text `{json.dumps(z.get('text'), ensure_ascii=False)[:200]}`"
                  + (f"; HTTP {z.get('status')} {z.get('error')}" if z.get("outcome") != "reply" else ""))
        w("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Budget log
# ----------------------------------------------------------------------------------------------------------------

def append_budget(path: Path, entry: dict) -> bool:
    """Insert one entry at the end of config/budget.yaml's `log` list (before the `planned:` block and its comment),
    keeping every other line as it is; checked by parsing before and after. Returns False (file untouched) if the
    file's layout is not the expected one."""
    text = Path(path).read_bytes().decode("utf-8")      # bytes: keep the file's own line endings
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    before = yaml.safe_load(text)
    if not isinstance(before, dict) or not isinstance(before.get("log"), list):
        return False
    j = next((i for i, l in enumerate(lines) if l.startswith("planned:")), None)
    if j is None:
        keys = [l.split(":")[0] for l in lines if re.match(r"^[A-Za-z_]+:", l)]
        if not keys or keys[-1] != "log":
            return False
        j = len(lines)
    else:
        while j > 0 and (lines[j - 1].startswith("#") or not lines[j - 1].strip()):
            j -= 1
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=10 ** 6).splitlines()
    new = lines[:j] + ["  " + b for b in block] + lines[j:]
    new_text = nl.join(new) + (nl if text.endswith(("\n", "\r\n")) else "")
    after = yaml.safe_load(new_text)
    if len(after["log"]) != len(before["log"]) + 1 or after["log"][-1] != json.loads(json.dumps(entry, default=str)):
        return False
    if {k: v for k, v in after.items() if k != "log"} != {k: v for k, v in before.items() if k != "log"}:
        return False
    Path(path).write_bytes(new_text.encode("utf-8"))
    return True


def budget_entry(phase: str, cfg: dict, lines: list[dict], session: str, summary: dict, t_start: str, t_end: str,
                 notes: str) -> dict:
    res = [x for x in lines if x.get("event") == "result" and x.get("session") == session]
    ans = [x for x in res if x.get("outcome") in ("reply", "refusal")]
    pt = sum(int(x.get("prompt_tokens") or 0) for x in res)
    ct = sum(int(x.get("completion_tokens") or 0) for x in res)
    rt = sum(int(x.get("reasoning_tokens") or 0) for x in res)
    ca = sum(int(x.get("cached_tokens") or 0) for x in res)
    billed = sum(float(x.get("cost_usd") or 0) for x in res)
    by = {}
    for x in res:
        by.setdefault(x["model_key"], 0.0)
        by[x["model_key"]] += float(x.get("cost_usd") or 0)
    models = "; ".join(f"{m['id']} on OpenRouter, provider pinned to {m['provider']['only']}, reasoning "
                       f"{m['reasoning_off']}, temperature {cfg['decoding']['temperature']}, top_p "
                       f"{cfg['decoding']['top_p']}" + (f", {m['extra']}" if m.get("extra") else "")
                       for m in cfg["models"] if m["key"] in by) or "none answered"
    return {
        "date": today(),
        "run": f"Qwen4-Max three arms on OpenRouter, {phase} (scripts/closed_w.py {PHASE_CMD[phase]})",
        "model": models,
        "unique_generations": f"{len(ans)} answered calls of {len(res)} attempts (refusals "
                              f"{sum(x.get('outcome') == 'refusal' for x in res)}, provider blocks "
                              f"{sum(x.get('outcome') == 'blocked' for x in res)}); answered by kind: "
                              + ", ".join(f"{k} {sum(x.get('kind') == k for x in ans)}" for k in REQUEST_KINDS
                                          if any(x.get("kind") == k for x in ans)),
        "cached_calls": 0,
        "tokens": f"prompt {pt:,} (cached by the provider {ca:,}), completion {ct:,} (reasoning {rt:,})",
        "cost_usd": round(billed, 4),
        "cost_detail": "billed by OpenRouter (usage.cost): " + ", ".join(f"{k} ${v:.4f}" for k, v in by.items())
                       + f"; counted against the limit ${summary.get('counted_total_usd', 0):.4f}",
        "runtime": f"{summary.get('seconds', 0):.0f} s ({t_start} to {t_end}, UTC)",
        "notes": notes,
    }


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------

def frame_of(fin: dict) -> pd.DataFrame:
    rows = [x for x in fin.values()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def boot_ci(M: np.ndarray, strata, draws: int, level: float, seed: int):
    """Item bootstrap of the template mean of per-template means: items resampled together across templates,
    within strata (BIG-Bench Hard tasks). M is (templates, items) with NaN for missing cells."""
    C, N = M.shape
    if N == 0 or not np.isfinite(M).any():
        return None, None
    rng = np.random.default_rng(seed)
    idx = np.empty((draws, N), dtype=np.int64)
    groups = {}
    for j, s in enumerate(strata):
        groups.setdefault(s, []).append(j)
    for cols in groups.values():
        cols = np.asarray(cols)
        idx[:, cols] = rng.choice(cols, size=(draws, len(cols)), replace=True)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per = np.nanmean(M[:, idx], axis=2)
        stat = np.nanmean(per, axis=0)
    stat = stat[np.isfinite(stat)]
    if not len(stat):
        return None, None
    a = (1 - level) / 2
    lo, hi = np.quantile(stat, [a, 1 - a])
    return float(lo), float(hi)


def template_mean(M: np.ndarray) -> float | None:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(np.nanmean(M, axis=1))
    return float(v) if np.isfinite(v) else None


def matrix(df: pd.DataFrame, col: str, templates, items) -> np.ndarray:
    if df.empty:
        return np.full((len(templates), len(items)), np.nan)
    p = df.pivot_table(index="config_id", columns="item_id", values=col, aggfunc="first")
    return p.reindex(index=list(templates), columns=list(items)).to_numpy(dtype=float)


def discovery_weights(task: str, templates, root: Path = ROOT) -> dict | None:
    p = Path(root) / "manifests" / f"sampler_{task}.json"
    if not p.exists():
        return None
    disc = [str(x) for x in json.loads(p.read_text(encoding="utf-8"))["discovery_config_ids"]]
    w = {c: disc.count(c) for c in templates}
    return w if sum(w.values()) else None


def weighted_template_mean(M: np.ndarray, templates, weights: dict | None):
    if weights is None:
        return None
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(M, axis=1)
    ws = np.array([weights.get(c, 0) for c in templates], dtype=float)
    ok = np.isfinite(v) & (ws > 0)
    return float((v[ok] * ws[ok]).sum() / ws[ok].sum()) if ok.any() else None


def spread(M: np.ndarray) -> dict:
    """Spread of per-template values: SD over templates (denominator C - 1) and, with every cell present, the
    variance across templates with item sampling removed (src/noise_anatomy.config_variance; negative kept)."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(M, axis=1)
    v = v[np.isfinite(v)]
    out = {"templates": int(len(v)), "min": float(v.min()) if len(v) else None,
           "median": float(np.median(v)) if len(v) else None, "max": float(v.max()) if len(v) else None,
           "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "var_shift_pp2": None, "w_item_pp2": None}
    if np.isfinite(M).all() and M.shape[0] >= 2 and M.shape[1] >= 2:
        from src import noise_anatomy as na
        cv = na.config_variance(M, scale=100.0)
        out.update(var_shift_pp2=cv["W_shift"], w_item_pp2=cv["W_item"])
    return out


def cell_metrics(g: pd.DataFrame, task: str, cfg: dict, count_cols: dict) -> dict:
    """Per-template values of one model (closed or panel): w by each count, accuracy, compliance, refusals,
    truncations. g holds the rows that count for w and accuracy (answered by the model)."""
    out = {"n": int(len(g))}
    for name, col in count_cols.items():
        for k in thresholds(cfg, task):
            v = g[col].astype(float)
            out[f"w{k}_{name}"] = float((v[v.notna()] > k).mean()) if v.notna().any() else None
    for c in ("strict_correct", "content_correct", "compliant"):
        out[c] = float(pd.to_numeric(g[c], errors="coerce").mean()) if c in g and len(g) else None
    out["truncated"] = float((g["finish_reason"] == "length").mean()) if len(g) else None
    if "outcome" in g:
        out["refusal"] = float((g["outcome"] == "refusal").mean()) if len(g) else None
    return out


def summarize(df: pd.DataFrame, task: str, cfg: dict, templates, items, count_cols: dict, seed_label: str,
              root: Path = ROOT, with_ci: bool = True) -> dict:
    """Per model and benchmark: template means (equal weight; and the discovery draws' weights), item-bootstrap
    intervals of w, the spread across templates."""
    b = cfg["bootstrap"]
    strata = [i.split("/", 1)[0] if task == "bbh" else "all" for i in items]
    wts = discovery_weights(task, templates, root)
    out = {"n_rows": int(len(df))}
    for name, col in count_cols.items():
        for k in thresholds(cfg, task):
            d = df.assign(_long=(df[col].astype(float) > k).where(df[col].notna()))
            M = matrix(d, "_long", templates, items)
            key = f"w{k}_{name}"
            out[key] = template_mean(M)
            out[key + "_discovery"] = weighted_template_mean(M, templates, wts)
            if with_ci:
                lo, hi = boot_ci(M, strata, int(b["draws"]), float(b["level"]),
                                 cwp.label_seed(f"{seed_label}/{key}"))
                out[key + "_ci"] = [lo, hi]
            out[key + "_spread"] = spread(M)
            out[key + "_missing_cells"] = int((~np.isfinite(M)).sum())
    for c in ("strict_correct", "lenient_correct", "content_correct", "compliant"):
        if c in df:
            out[c] = template_mean(matrix(df.assign(**{c: pd.to_numeric(df[c], errors="coerce")}), c, templates,
                                          items))
    out["truncated"] = template_mean(matrix(df.assign(_t=(df["finish_reason"] == "length").astype(float)), "_t",
                                            templates, items))
    return out


def panel_table(task: str, items, templates, root: Path = ROOT) -> pd.DataFrame:
    """The panel's requested-arm rows on the same items and templates: n_output_tokens (each model's own tokenizer),
    finish_reason, strict_correct, compliant, content_correct."""
    p = Path(root) / "data" / "processed"
    f = [("arm", "==", ARM)]
    key = ["model_key", "config_id", "item_id"]
    if task == "hotpot":
        df = pd.read_parquet(p / "hotpot" / "scored_content.parquet", filters=f,
                             columns=key + ["arm", "n_output_tokens", "finish_reason", "strict_correct", "compliant",
                                            "content_correct"])
    else:
        c = pd.read_parquet(p / "content" / f"{task}_content.parquet", filters=f,
                            columns=key + ["arm", "finish_reason", "strict_correct", "compliant", "content_correct"])
        srcs = [p / task / "scored.parquet"]
        if task == "gsm8k":
            srcs += sorted((p / "gsm8k_ext").glob("scored_*.parquet"))
        s = pd.concat([pd.read_parquet(x, filters=f, columns=key + ["n_output_tokens"]) for x in srcs],
                      ignore_index=True)
        for d in (c, s):
            d["item_id"] = d["item_id"].astype(str)
            d["config_id"] = d["config_id"].astype(str)
        df = c.merge(s, on=key, how="left", validate="one_to_one")
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    df = df[df["item_id"].isin(set(items)) & df["config_id"].isin(set(templates))].reset_index(drop=True)
    return df.drop(columns=["arm"], errors="ignore")


def open_text_dirs(task: str, models, root: Path = ROOT) -> list[Path]:
    d = Path(root) / "data" / "d2_out"
    out = []
    for m in models:
        if task == "gsm8k":
            out.append(d / "collection" / m if (d / "collection" / m).is_dir() else d / "collection_ext" / m)
        elif task == "math":
            out.append(d / "math" / "collection" / m)
        elif task == "bbh":
            out.append(d / "bbh" / "collection" / m)
        else:
            out += [d / "hotpot" / "block_a" / m, d / "hotpot" / "block_b" / m]
    return out


def load_open_texts(task: str, items, templates, models, root: Path = ROOT, loader=None) -> tuple[pd.DataFrame, list]:
    """The panel's requested-arm texts on the same cells, read from their output folders (src/d2_io.inspect_work_dir)."""
    if loader is None:
        from src import d2_io
        loader = d2_io.inspect_work_dir
    parts, problems = [], []
    for d in open_text_dirs(task, models, root):
        if not d.is_dir():
            problems.append(f"{rel(d)} absent")
            continue
        try:
            out, _ = loader(d, string_item_ids=(task != "gsm8k"))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{rel(d)}: {type(e).__name__}: {e}"[:300])
            continue
        o = out[out["arm"] == ARM].copy()
        o["item_id"] = o["item_id"].astype(str)
        o["config_id"] = o["config_id"].astype(str)
        o = o[o["item_id"].isin(set(items)) & o["config_id"].isin(set(templates))]
        parts.append(o[["model_key", "config_id", "item_id", "output_text"]])
    if not parts:
        return pd.DataFrame(columns=["model_key", "config_id", "item_id", "output_text"]), problems
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["model_key", "config_id", "item_id"], keep="first")
    return df, problems


PANEL_COLS = ["model_key", "arm", "config_id", "item_id", "n_output_tokens", "finish_reason", "strict_correct",
              "lenient_correct", "compliant", "content_correct"]


def _cells(df: pd.DataFrame, items, templates) -> pd.DataFrame:
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    return df[df["item_id"].isin(set(items)) & df["config_id"].isin(set(templates))].reset_index(drop=True)


def panel_table3(task: str, items, templates, root: Path = ROOT) -> pd.DataFrame:
    """The panel's rows of the three arms on the same items and templates (arm as the run's label): the readings of
    the content tables (strict, lenient, compliance and content, as they stand), and n_output_tokens (each model's own
    tokenizer, at the panel's caps) of the scored tables (requested and chain of thought: data/processed/<task>/
    scored.parquet, GSM8K also gsm8k_ext; forced: data/processed/forced/<task>_scored.parquet). HotpotQA: its
    scored_content table."""
    p = Path(root) / "data" / "processed"
    srcs = list(ARM_SOURCE.values())
    key = ["model_key", "arm", "config_id", "item_id"]
    if task == "hotpot":
        df = pd.read_parquet(p / "hotpot" / "scored_content.parquet", filters=[("arm", "in", srcs)],
                             columns=PANEL_COLS)
        df = _cells(df, items, templates)
    else:
        c = _cells(pd.read_parquet(p / "content" / f"{task}_content.parquet", filters=[("arm", "in", srcs)],
                                   columns=[x for x in PANEL_COLS if x != "n_output_tokens"]), items, templates)
        files = [p / task / "scored.parquet"] + (sorted((p / "gsm8k_ext").glob("scored_*.parquet"))
                                                if task == "gsm8k" else [])
        parts = [_cells(pd.read_parquet(f, columns=key + ["n_output_tokens"],
                                        filters=[("arm", "in", ["direct_answer", "zero_shot_cot"])]), items, templates)
                 for f in files]
        parts.append(_cells(pd.read_parquet(p / "forced" / f"{task}_scored.parquet",
                                            columns=key + ["n_output_tokens"]), items, templates))
        s = pd.concat(parts, ignore_index=True).drop_duplicates(key, keep="first")
        df = c.merge(s, on=key, how="left", validate="one_to_one")
    df["arm"] = df["arm"].map(SOURCE_ARM)
    return df[PANEL_COLS].reset_index(drop=True)


def ladder_table(task: str, items, templates, root: Path = ROOT) -> pd.DataFrame:
    """The size ladder's six models (scripts/ladder_analysis.LADDER) on the same items and templates, the three arms,
    the reading cut at the panel's caps (the ladder's primary; ladder_analysis.model_paths)."""
    import ladder_analysis as la
    out = []
    for k in la.LADDER:
        f = la.model_paths(k, Path(root))["cut"][task]
        if not Path(f).exists():
            continue
        d = pd.read_parquet(f)
        d = _cells(d[d["arm"].isin(list(ARM_SOURCE.values()))].copy(), items, templates)
        if "n_output_tokens_cut" in d:
            d["n_output_tokens"] = d["n_output_tokens_cut"]
        if "finish_reason_cut" in d:
            d["finish_reason"] = d["finish_reason_cut"]
        d["model_key"] = k
        d["arm"] = d["arm"].map(SOURCE_ARM)
        for c in PANEL_COLS:
            if c not in d:
                d[c] = np.nan
        out.append(d[PANEL_COLS])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=PANEL_COLS)


QUANTS = ("requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced", "w", "corrected_gain", "residual")
Q_LABEL = {"requested": "Requested", "forced": "Forced", "cot": "CoT", "cot_minus_requested": "CoT − requested",
           "cot_minus_forced": "CoT − forced", "w": "w (forced cap)", "corrected_gain": "Corrected gain",
           "residual": "CoT − requested − corrected gain"}
SCORE_OF = {"strict": "strict_correct", "content": "content_correct", "lenient": "lenient_correct"}


def forced_cap_of(cfg: dict, task: str) -> int:
    return int(cfg["caps"]["forced"][task])


def arm_mats(df: pd.DataFrame, templates, items, score_col: str, count_col: str, cap: int) -> dict:
    """(templates, items) matrices of one model: R, F, C (the score per arm) and W (requested longer than the forced
    cap, by '>' on the count; NaN where there is no requested row)."""
    out = {}
    for a, k in (("requested", "R"), ("forced", "F"), ("cot", "C")):
        g = df[df["arm"] == a]
        out[k] = matrix(g.assign(_v=pd.to_numeric(g[score_col], errors="coerce")), "_v", templates, items)
    g = df[df["arm"] == "requested"]
    v = pd.to_numeric(g[count_col], errors="coerce")
    out["W"] = matrix(g.assign(_w=(v > cap).astype(float).where(v.notna())), "_w", templates, items)
    return out


def _wmean(V: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """Weighted mean over the last axis (templates), NaN templates left out (their weight with them)."""
    ok = np.isfinite(V) & (wv > 0)
    num = np.where(ok, V * wv, 0.0).sum(axis=-1)
    den = np.where(ok, wv, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def quantities_from(r, f, c, w) -> dict:
    """The quantities from per-basis means (accuracies as shares; pp later): the corrected gain of
    src/baseline_mixture.RULE_TEXT, (1 - w) x (CoT - forced), w and the gain each averaged first."""
    g_fd = c - f
    corr = (1 - w) * g_fd
    return {"requested": r, "forced": f, "cot": c, "cot_minus_requested": c - r, "cot_minus_forced": g_fd, "w": w,
            "corrected_gain": corr, "residual": (c - r) - corr}


def template_means(M: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(M, axis=1)


def model_quantities(mats: dict, wv: np.ndarray) -> dict:
    tm = {k: template_means(v) for k, v in mats.items()}
    return {k: (float(v) if np.isfinite(v) else None) for k, v in
            quantities_from(*(float(_wmean(tm[x], wv)) for x in ("R", "F", "C", "W"))).items()}


def boot_quantities(mats: dict, strata, bases: dict, draws: int, level: float, seed: int, chunk: int = 250) -> dict:
    """Item bootstrap of every quantity, per benchmark (for each basis, weights over templates) and per template:
    items resampled with replacement within strata (BIG-Bench Hard tasks), the same draw for every arm and matrix, so
    the gains are paired. Percentile intervals; [None, None] where no draw is finite."""
    any_m = next(iter(mats.values()))
    C, N = any_m.shape
    if N == 0:
        return {}
    rng = np.random.default_rng(seed)
    groups = {}
    for j, s in enumerate(strata):
        groups.setdefault(s, []).append(j)
    idx = np.empty((draws, N), dtype=np.int64)
    for cols in groups.values():
        cols = np.asarray(cols)
        idx[:, cols] = rng.choice(cols, size=(draws, len(cols)), replace=True)
    per = {k: np.empty((draws, C)) for k in mats}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s0 in range(0, draws, chunk):
            sl = idx[s0:s0 + chunk]
            for k, M in mats.items():
                per[k][s0:s0 + chunk] = np.nanmean(M[:, sl], axis=2).T
    a = (1 - level) / 2

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(x) for x in np.quantile(v, [a, 1 - a])] if len(v) else [None, None]
    out = {"benchmark": {}}
    for base, wv in bases.items():
        qb = quantities_from(*(_wmean(per[x], wv) for x in ("R", "F", "C", "W")))
        out["benchmark"][base] = {k: ci(v) for k, v in qb.items()}
    qt = quantities_from(*(per[x] for x in ("R", "F", "C", "W")))
    out["per_template"] = {k: [ci(v[:, j]) for j in range(C)] for k, v in qt.items()}
    return out


def placement(values: dict, me: str) -> dict:
    """Rank of `me` among the models with a finite value, from the largest: k = 1 + the number of others above it;
    ties (|difference| <= 1e-9) counted and named, not broken."""
    v = values.get(me)
    fin = {m: x for m, x in values.items() if x is not None and np.isfinite(x)}
    if v is None or me not in fin:
        return {"k": None, "M": len(fin), "ties": []}
    above = [m for m, x in fin.items() if m != me and x > v + 1e-9]
    ties = sorted(m for m, x in fin.items() if m != me and abs(x - v) <= 1e-9)
    return {"k": 1 + len(above), "M": len(fin), "ties": ties}


def basis_weights(task: str, templates, root: Path = ROOT) -> dict:
    """Template weights per basis: every template once; the discovery draws (a template drawn twice counts twice,
    src/baseline_mixture.RULE_TEXT's basis)."""
    out = {"templates": np.ones(len(templates))}
    dw = discovery_weights(task, templates, root)
    if dw is not None:
        out["discovery"] = np.array([dw.get(c, 0) for c in templates], dtype=float)
    return out


def build_report(cfg: dict, man: dict, paths: Paths, scorer=score_rows, tok: Tok | None = None,
                 open_texts: bool = False, panel_loader=None, open_loader=None, root: Path = ROOT,
                 ladder_loader=None) -> dict:
    panel_loader = panel_loader or panel_table3
    ladder_loader = ladder_table if ladder_loader is None else ladder_loader
    lines = read_raw(paths.raw("run"))
    if not lines:
        raise SystemExit(f"{rel(paths.raw('run'))} holds no replies: nothing to report")
    shas = {x.get("config_sha") for x in lines if x.get("event") == "result"}
    if len(shas) > 1:
        raise SystemExit(f"the run's raw file holds replies under {len(shas)} configurations")
    csha = shas.pop() if shas else None
    fin = final_by_key(lines, csha)
    tok = tok or Tok(cfg, root)
    acc = [dict(v, arm=row_arm(v)) for v in fin.values() if v.get("outcome") in ACCEPTED]
    df = frame_of({i: v for i, v in enumerate(acc)})
    if df.empty:
        raise SystemExit("no answered call in the run's raw file")
    if "stop_sent" not in df:
        df["stop_sent"] = False
    models = [m for m in cfg["models"] if m["key"] in set(df["model_key"])]
    arms = [a for a in ARMS if a in set(df["arm"])]
    expected = {(m["key"], a, t): man["tasks"][t]["n_requests"] for m in models for a in arms for t in TASKS}
    rep = {"made_on": today(), "config_sha": csha, "script": "scripts/closed_w.py report", "tokenizer": tok.name,
           "arms": arms, "models": {}, "per_template": [], "panel": {}, "per_benchmark": [], "token_accounting": {},
           "gains": [], "placement": [], "spend": {}, "open_texts": None, "scorers": {}, "cutoff_sample": {},
           "forced_reading": {}, "rule": None}
    try:
        from src import baseline_mixture as bm
        rep["rule"] = bm.RULE_TEXT
    except Exception:  # noqa: BLE001
        rep["rule"] = None
    for ph in ("pilot", "prefill", "run"):
        rep["spend"][ph] = ledger(read_raw(paths.raw(ph)))
    df["n_api"] = [api_count(x) for x in df.to_dict("records")]
    df["n_local"] = tok.counts(df["text"].tolist())
    answered = df[df["outcome"].isin(("reply", "refusal"))].copy()
    scored, versions = scorer(answered)
    rep["scorers"] = versions
    att_all = [x for x in lines if x.get("event") == "result"]
    for m in models:
        mk = m["key"]
        rep["models"][mk] = {"label": m["label"], "id": m["id"], "arms": {}}
        for a in arms:
            sub = df[(df["model_key"] == mk) & (df["arm"] == a)]
            att = [x for x in att_all if x.get("model_key") == mk and row_arm(x) == a]
            n_exp = sum(expected[(mk, a, t)] for t in TASKS)
            rep["models"][mk]["arms"][a] = {
                "expected": n_exp, "answered": int(sub["outcome"].isin(("reply", "refusal")).sum()),
                "refusals": int((sub["outcome"] == "refusal").sum()), "blocked": int((sub["outcome"] == "blocked").sum()),
                "missing": int(n_exp - len(sub)), "attempts": len(att),
                "failed_attempts": sum(x.get("outcome") in ("failed", "error") for x in att),
                "flagged": sum(x.get("outcome") == "flagged" for x in att),
                "truncated": int((sub["finish_reason"] == "length").sum()),
                "billed_usd": float(sum(float(x.get("cost_usd") or 0) for x in att)),
                "prompt_tokens": int(sum(int(x.get("prompt_tokens") or 0) for x in att)),
                "completion_tokens": int(sum(int(x.get("completion_tokens") or 0) for x in att)),
                "reasoning_tokens": int(sum(int(x.get("reasoning_tokens") or 0) for x in att)),
                "providers": sorted({str(x) for x in sub["provider"].dropna()}),
                "served_models": sorted({str(x) for x in sub["served_model"].dropna()})}
        if "forced" in arms and "forced_reading" in scored:
            f = scored[(scored["model_key"] == mk) & (scored["arm"] == "forced")]
            starts = f["text"].map(lambda s: isinstance(s, str) and s.startswith(cfg["prefill"]))
            rep["forced_reading"][mk] = {
                t: {"n": int((f["task"] == t).sum()),
                    **{k: int(((f["task"] == t) & (f["forced_reading"] == k)).sum()) for k in FORCED_READINGS},
                    "starts_with_prefill": int(((f["task"] == t) & starts).sum())} for t in TASKS}
    counts = {"api": "n_api", "recount": "n_local"}
    b = cfg["bootstrap"]
    for t in TASKS:
        templates, items = man["tasks"][t]["templates"], man["tasks"][t]["items"]
        strata = [i.split("/", 1)[0] if t == "bbh" else "all" for i in items]
        bw = basis_weights(t, templates, root)
        cap = forced_cap_of(cfg, t)
        # others: the panel and the ladder on the same cells
        others, problems = [], {}
        for src, loader in (("panel", panel_loader), ("ladder", ladder_loader)):
            if loader is None:
                continue
            try:
                d = loader(t, items, templates, root)
                d = d.assign(source=src)
                others.append(d)
            except Exception as e:  # noqa: BLE001
                problems[src] = f"{type(e).__name__}: {e}"[:300]
        oth = pd.concat(others, ignore_index=True) if others else pd.DataFrame(columns=PANEL_COLS + ["source"])
        if "arm" not in oth:
            oth["arm"] = "requested"
        oth["outcome"] = "reply"
        rep["panel"][t] = {"problems": problems, "models": {}}
        if open_texts:
            pm = sorted(oth.loc[oth["source"] == "panel", "model_key"].unique()) if len(oth) else []
            ot, probs = load_open_texts(t, items, templates, pm, root, open_loader)
            rec = {"problems": probs, "rows_recounted": int(len(ot))}
            if len(ot):
                ot = ot.assign(n_local=tok.counts(ot["output_text"].tolist()), arm="requested", source="panel")
                oth = oth.merge(ot[["model_key", "config_id", "item_id", "arm", "source", "n_local"]],
                                on=["model_key", "config_id", "item_id", "arm", "source"], how="left")
                ref = oth[(oth["model_key"] == "qwen3.6-27b") & (oth["arm"] == "requested")].dropna(subset=["n_local"])
                if len(ref):
                    d = ref["n_output_tokens"].astype(float) - ref["n_local"].astype(float)
                    rec["qwen3.6-27b_native_minus_recount"] = {"n": int(len(d)), "median": float(d.median()),
                                                               "share_zero": float((d == 0).mean()),
                                                               "share_one": float((d == 1).mean())}
            rep["open_texts"] = rep["open_texts"] or {}
            rep["open_texts"][t] = rec
        pcounts = {"native": "n_output_tokens", **({"recount": "n_local"} if "n_local" in oth else {})}
        # the requested arm's w side by side (as built), every closed model and panel/ladder model
        for m in models:
            mk = m["key"]
            s = scored[(scored["model_key"] == mk) & (scored["task"] == t)]
            for a in arms:
                sa = s[s["arm"] == a]
                summ = summarize(sa, t, cfg, templates, items, counts if a == "requested" else {},
                                 f"closed_w/bootstrap/{mk}/{a}/{t}", root)
                for c in ("strict_correct", "lenient_correct", "content_correct"):
                    if c in sa:
                        M = matrix(sa.assign(_v=pd.to_numeric(sa[c], errors="coerce")), "_v", templates, items)
                        summ[c + "_ci"] = list(boot_ci(M, strata, int(b["draws"]), float(b["level"]),
                                                       cwp.label_seed(f"closed_w/bootstrap/{mk}/{a}/{t}/{c}")))
                full = df[(df["model_key"] == mk) & (df["task"] == t) & (df["arm"] == a)]
                summ.update(source="closed", model=mk, task=t, arm=a,
                            refusal=float((sa["outcome"] == "refusal").mean()) if len(sa) else None,
                            missing=int(expected[(mk, a, t)] - len(full)),
                            blocked=int((full["outcome"] == "blocked").sum()))
                rep["per_benchmark"].append(summ)
            sreq = s[s["arm"] == "requested"]
            if len(sreq):
                aa = sreq.dropna(subset=["n_api"])
                if len(aa):
                    diff = aa["n_api"].astype(float) - aa["n_local"].astype(float)
                    agree = np.mean([all((x > k) == (y > k) for k in thresholds(cfg, t))
                                     for x, y in zip(aa["n_api"], aa["n_local"])])
                    rep["token_accounting"].setdefault(mk, {})[t] = {
                        "n": int(len(aa)), "median_api_minus_recount": float(diff.median()),
                        "p10": float(diff.quantile(0.1)), "p90": float(diff.quantile(0.9)),
                        "crossing_agreement": float(agree)}
        for (src, mk), g in oth.groupby(["source", "model_key"]):
            gr = g[g["arm"] == "requested"]
            sm = summarize(gr, t, cfg, templates, items, pcounts if src == "panel" else {"native": "n_output_tokens"},
                           f"closed_w/bootstrap/{src}/{mk}/{t}", root, with_ci=False) if len(gr) else {}
            sm.update(source=src, model=mk, task=t, cells_present={a: int((g["arm"] == a).sum()) for a in ARMS},
                      cells_expected=len(items) * len(templates))
            rep["panel"][t]["models"][mk] = sm
        # gains and placement
        if not {"requested", "forced", "cot"} <= set(arms):
            continue
        for score in ("strict", "content"):
            col = SCORE_OF[score]
            vals = {base: {} for base in bw}
            for m in models:
                mk = m["key"]
                s = scored[(scored["model_key"] == mk) & (scored["task"] == t)]
                mats = arm_mats(s, templates, items, col, "n_api", cap)
                mats_rc = arm_mats(s, templates, items, col, "n_local", cap)
                bq = boot_quantities(mats, strata, bw, int(b["draws"]), float(b["level"]),
                                     cwp.label_seed(f"closed_w/bootstrap3/{mk}/{t}/{score}"))
                for base, wv in bw.items():
                    q = model_quantities(mats, wv)
                    vals[base][mk] = q
                    rep["gains"].append({"model": mk, "task": t, "score": score, "basis": base, **q,
                                         "w_recount": model_quantities(mats_rc, wv)["w"],
                                         "ci": (bq.get("benchmark") or {}).get(base, {})})
                tm = {k: template_means(v) for k, v in mats.items()}
                qt = quantities_from(tm["R"], tm["F"], tm["C"], tm["W"])
                tci = bq.get("per_template", {})
                for j, c in enumerate(templates):
                    rep["per_template"].append({
                        "source": "closed", "model": mk, "task": t, "score": score, "config_id": c,
                        "discovery_weight": float(bw["discovery"][j]) if "discovery" in bw else None,
                        **{k: (float(v[j]) if np.isfinite(v[j]) else None) for k, v in qt.items()},
                        **{f"{k}_ci": tci.get(k, [[None, None]] * len(templates))[j] for k in QUANTS},
                        "cells": {a: int(np.isfinite(mats[k]).sum(axis=1)[j]) for a, k in
                                  (("requested", "R"), ("forced", "F"), ("cot", "C"))}})
            for (src, mk), g in oth.groupby(["source", "model_key"]):
                mats = arm_mats(g, templates, items, col, "n_output_tokens", cap)
                for base, wv in bw.items():
                    vals[base][f"{src}:{mk}"] = model_quantities(mats, wv)
                tm = {k: template_means(v) for k, v in mats.items()}
                qt = quantities_from(tm["R"], tm["F"], tm["C"], tm["W"])
                for j, c in enumerate(templates):
                    rep["per_template"].append({
                        "source": src, "model": mk, "task": t, "score": score, "config_id": c,
                        "discovery_weight": float(bw["discovery"][j]) if "discovery" in bw else None,
                        **{k: (float(v[j]) if np.isfinite(v[j]) else None) for k, v in qt.items()}})
            for m in models:
                mk = m["key"]
                for base in bw:
                    for qn in QUANTS:
                        allv = {k: v[qn] for k, v in vals[base].items() if k == mk or ":" in k}
                        pan = {k: v for k, v in allv.items() if k == mk or k.startswith("panel:")}
                        others_v = [v for k, v in allv.items() if k != mk and v is not None]
                        rep["placement"].append({
                            "model": mk, "task": t, "score": score, "basis": base, "quantity": qn,
                            "value": vals[base][mk][qn], "panel_and_ladder": placement(allv, mk),
                            "panel": placement(pan, mk),
                            "others_median": float(np.median(others_v)) if others_v else None,
                            "others_min": float(min(others_v)) if others_v else None,
                            "others_max": float(max(others_v)) if others_v else None,
                            "n_panel": sum(k.startswith("panel:") for k in allv),
                            "n_ladder": sum(k.startswith("ladder:") for k in allv)})
    # replies around the cutoff, for hand reading (requested arm)
    for m in models:
        mk = m["key"]
        s = scored[(scored["model_key"] == mk) & (scored["arm"] == "requested")
                   & pd.to_numeric(scored["n_api"], errors="coerce").between(12, 24)].copy()
        s["h"] = [hkey("closed_w/cutoff", k) for k in s["key"]]
        s = s.sort_values("h").head(12)
        rep["cutoff_sample"][mk] = [{"task": r.task, "request_id": r.request_id, "n_api": r.n_api,
                                     "n_recount": r.n_local, "strict": r.strict_correct, "content": r.content_correct,
                                     "text": (r.text or "")[:400] if isinstance(r.text, str) else None}
                                    for r in s.itertuples()]
    return rep


def _pp(x, d=1) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.{d}f}"


def _ci(c, sign=False) -> str:
    if not c or c[0] is None:
        return ""
    f = "{:+.1f}" if sign else "{:.1f}"
    return f" [{f.format(100 * c[0])}, {f.format(100 * c[1])}]"


def render_report(rep: dict, cfg: dict, man: dict) -> str:
    L = []
    w = L.append
    mlabel = {m["key"]: m["label"] for m in cfg["models"]}
    w(f"# Qwen4-Max, the panel's three arms on the same cells ({rep['made_on']})")
    w("")
    w("Written by `scripts/closed_w.py report` from `data/frontier_panel/closed_w/run_raw.jsonl`; design "
      "`config/closed_w.yaml`, items and request lists `manifests/closed_w_items.json`. Qwen4-Max (`qwen/qwen4-max`, "
      "reasoning off, presence penalty 0) on OpenRouter with the provider pinned to Alibaba, temperature 0, top_p 1. "
      f"Every template of the four benchmarks ({man['n_cells']} benchmark-template cells) on the same 100 items per "
      "benchmark (BIG-Bench Hard 4 per task). Requested arm: the panel's requested user message as the only message, "
      "cap 1,024. Forced arm: the same user message and an assistant message `\\boxed{` with `partial: true`, caps 16 "
      "(HotpotQA 64), the stop `}` on GSM8K and BIG-Bench Hard. Chain of thought: the panel's zero-shot request, caps "
      "3,072 (HotpotQA 1,900).")
    w("")
    w("Accuracy is the template mean (every template once) of per-template means, in %; intervals: item bootstrap "
      "(2,000 draws, items resampled together across templates and arms, BIG-Bench Hard within task), 95% percentile. "
      "w is the share of requested replies longer than 16 tokens (HotpotQA also 64) by the API's count (which, like "
      "the panel's counts, includes the end-of-turn token) and by the recount of the visible text. The gains and the "
      "corrected gain follow `src/baseline_mixture.RULE_TEXT`: w at the forced cap (16; 64 on HotpotQA), the corrected "
      "gain (1 − w) × (CoT − forced), w and the gain each averaged first; on the discovery draws (a template drawn "
      "twice counts twice) as the rule states, and on every template once beside it.")
    w("")
    w("## Calls")
    w("")
    w("| Model | Arm | Expected | Answered | Refusals | Provider blocks | Missing | Truncated (length) | Attempts | "
      "Billed | Prompt tok. | Output tok. | Reasoning tok. |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mk, r in rep["models"].items():
        for a, v in r["arms"].items():
            w(f"| {r['label']} | {a} | {v['expected']:,} | {v['answered']:,} | {v['refusals']} | {v['blocked']} | "
              f"{v['missing']} | {v['truncated']} | {v['attempts']:,} | {usd(v['billed_usd'])} | "
              f"{v['prompt_tokens']:,} | {v['completion_tokens']:,} | {v['reasoning_tokens']:,} |")
    sp = rep["spend"]
    tot = sum(v["billed_usd"] for v in sp.values())
    w("")
    w(f"Spend billed by OpenRouter: pilot {usd(sp['pilot']['billed_usd'])}, prefill test "
      f"{usd(sp['prefill']['billed_usd'])}, run {usd(sp['run']['billed_usd'])}; total {usd(tot)}.")
    w("")
    if rep.get("forced_reading"):
        w("Forced replies as read (`forced_reply`): kept as returned, cut after their first `}`, or with the `}` put "
          "back where the reply ended on the stop; and those that start with the prefill itself (a restart, scored "
          "as the panel's runner would have stored it, the prefill in front once).")
        w("")
        w("| Model | Benchmark | Forced replies | As returned | Cut after the first } | } put back | Start with \\boxed{ |")
        w("|---|---|---|---|---|---|---|")
        for mk, d in rep["forced_reading"].items():
            for t, v in d.items():
                w(f"| {mlabel.get(mk, mk)} | {TASK_NAME[t]} | {v['n']:,} | {v['as_returned']:,} | "
                  f"{v['cut_after_first_brace']:,} | {v['stop_restored']:,} | {v['starts_with_prefill']:,} |")
        w("")
    pb = {(r["model"], r.get("arm", "requested"), r["task"]): r for r in rep["per_benchmark"]}
    w("## Accuracy per arm (template mean, %, [95% CI])")
    w("")
    w("| Model | Arm | Reading | GSM8K | MATH-500 | BIG-Bench Hard | HotpotQA |")
    w("|---|---|---|---|---|---|---|")
    for mk in rep["models"]:
        for a in rep["arms"]:
            for c, lab in (("strict_correct", "strict"), ("lenient_correct", "lenient"),
                           ("content_correct", "content")):
                cells = []
                for t in TASKS:
                    r = pb.get((mk, a, t))
                    cells.append("-" if r is None or r.get(c) is None else f"{pct(r[c])}{_ci(r.get(c + '_ci'))}")
                w(f"| {mlabel[mk]} | {a} | {lab} | " + " | ".join(cells) + " |")
    w("")
    w("## w, requested arm (template mean, %, [95% CI])")
    w("")
    w("| Model | Count | GSM8K | MATH-500 | BIG-Bench Hard | HotpotQA (16) | HotpotQA (64) |")
    w("|---|---|---|---|---|---|---|")
    for mk in rep["models"]:
        for cnt, lab in (("api", "API count"), ("recount", "recount")):
            cells = []
            for t, k in (("gsm8k", 16), ("math", 16), ("bbh", 16), ("hotpot", 16), ("hotpot", 64)):
                r = pb.get((mk, "requested", t))
                if r is None or r.get(f"w{k}_{cnt}") is None:
                    cells.append("-")
                    continue
                cells.append(f"{pct(r[f'w{k}_{cnt}'])}{_ci(r.get(f'w{k}_{cnt}_ci'))}")
            w(f"| {mlabel[mk]} | {lab} | " + " | ".join(cells) + " |")
    for t in TASKS:
        pm = (rep["panel"].get(t) or {}).get("models") or {}
        vals = [r.get("w16_native") for r in pm.values() if r.get("w16_native") is not None]
        if vals:
            w(f"| Panel and ladder on {TASK_NAME[t]} ({len(vals)} models, own tokenizers) | median "
              f"{pct(float(np.median(vals)))} | range {pct(min(vals))} to {pct(max(vals))} | | | | |")
    w("")
    if rep["gains"]:
        gi = {(g["model"], g["task"], g["score"], g["basis"]): g for g in rep["gains"]}
        for basis, blab in (("discovery", "discovery draws (the rule's basis)"), ("templates", "every template once")):
            if not any(k[3] == basis for k in gi):
                continue
            w(f"## Gains and the corrected gain, {blab} (pp, [95% CI])")
            w("")
            w("| Model | Benchmark | Reading | Requested % | Forced % | CoT % | CoT − forced | CoT − requested | "
              "w (forced cap) % | Corrected gain | CoT − requested − corrected |")
            w("|---|---|---|---|---|---|---|---|---|---|---|")
            for mk in rep["models"]:
                for t in TASKS:
                    for score in ("strict", "content"):
                        g = gi.get((mk, t, score, basis))
                        if g is None:
                            continue
                        ci = g.get("ci") or {}
                        w(f"| {mlabel[mk]} | {TASK_NAME[t]} | {score} | {pct(g['requested'])}{_ci(ci.get('requested'))} | "
                          f"{pct(g['forced'])}{_ci(ci.get('forced'))} | {pct(g['cot'])}{_ci(ci.get('cot'))} | "
                          f"{_pp(g['cot_minus_forced'])}{_ci(ci.get('cot_minus_forced'), True)} | "
                          f"{_pp(g['cot_minus_requested'])}{_ci(ci.get('cot_minus_requested'), True)} | "
                          f"{pct(g['w'])}{_ci(ci.get('w'))} (recount {pct(g.get('w_recount'))}) | "
                          f"{_pp(g['corrected_gain'])}{_ci(ci.get('corrected_gain'), True)} | "
                          f"{_pp(g['residual'])}{_ci(ci.get('residual'), True)} |")
            w("")
    if rep["placement"]:
        w("## Where Qwen4-Max falls among the panel's and the ladder's models (same 100 items and templates)")
        w("")
        w("k/M: Qwen4-Max's rank from the largest value among the M models with a value (Qwen4-Max included); ties "
          "(|difference| ≤ 1e-9) are not broken and are counted. The panel's models from their scored tables (29; 28 "
          "on BIG-Bench Hard, where Gemma-4-31B-it has no rows); the ladder's six models in the reading cut at the "
          "panel's caps.")
        w("")
        for basis, blab in (("discovery", "discovery draws"), ("templates", "every template once")):
            rows = [p for p in rep["placement"] if p["basis"] == basis]
            if not rows:
                continue
            w(f"### Basis: {blab}")
            w("")
            w("| Benchmark | Reading | Quantity | Qwen4-Max | k/M, panel and ladder | k/M, panel | Others: median "
              "(min to max) |")
            w("|---|---|---|---|---|---|---|")
            for p in rows:
                isw = p["quantity"] == "w"
                fmt = (lambda x: pct(x)) if (isw or p["quantity"] in ("requested", "forced", "cot")) else _pp
                pl, pp_ = p["panel_and_ladder"], p["panel"]
                tie = f" (ties: {', '.join(pl['ties'])})" if pl["ties"] else ""
                w(f"| {TASK_NAME[p['task']]} | {p['score']} | {Q_LABEL[p['quantity']]} | {fmt(p['value'])} | "
                  f"{pl['k']}/{pl['M']}{tie} | {pp_['k']}/{pp_['M']} | {fmt(p['others_median'])} "
                  f"({fmt(p['others_min'])} to {fmt(p['others_max'])}) |")
            w("")
    w("## Token accounting, requested arm: API count against the recount")
    w("")
    w("| Model | Benchmark | Replies | Median API − recount | 10th / 90th pct | Same side of the threshold(s) % |")
    w("|---|---|---|---|---|---|")
    for mk, d in rep["token_accounting"].items():
        for t, v in d.items():
            w(f"| {mlabel[mk]} | {TASK_NAME[t]} | {v['n']:,} | {v['median_api_minus_recount']:+.0f} | "
              f"{v['p10']:+.0f} / {v['p90']:+.0f} | {pct(v['crossing_agreement'])} |")
    w("")
    if rep.get("open_texts"):
        w("With `--open-texts` the panel's requested texts were recounted by the same tokenizer. A panel count one "
          "above the recount is the end-of-sequence token, which the panel's runner (vLLM) counts among the generated "
          "tokens, as the API's count does.")
        w("")
        for t, v in rep["open_texts"].items():
            q = v.get("qwen3.6-27b_native_minus_recount")
            w(f"- {TASK_NAME[t]}: {v['rows_recounted']:,} panel texts recounted"
              + (f"; Qwen3.6-27B native minus recount: median {q['median']:+.0f}, zero on {pct(q['share_zero'])}%, "
                 f"one on {pct(q['share_one'])}%" if q else "")
              + (f"; not read: {len(v['problems'])} folders" if v["problems"] else ""))
        w("")
    w("## Spread of w across templates (requested arm, API count, threshold 16)")
    w("")
    w("| Model | Benchmark | Templates | Min % | Median % | Max % | SD (pp, C − 1) | Variance with item sampling "
      "removed (pp², negative kept) | Missing cells |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rep["per_benchmark"]:
        if r.get("arm", "requested") != "requested":
            continue
        s = r.get("w16_api_spread") or {}
        w(f"| {mlabel[r['model']]} | {TASK_NAME[r['task']]} | {s.get('templates')} | {pct(s.get('min'))} | "
          f"{pct(s.get('median'))} | {pct(s.get('max'))} | "
          f"{'-' if s.get('sd') is None else f'{100 * s['sd']:.1f}'} | "
          f"{'-' if s.get('var_shift_pp2') is None else f'{s['var_shift_pp2']:.1f}'} | "
          f"{r.get('w16_api_missing_cells')} |")
    w("")
    w("## Replies around the cutoff (requested arm, 12 to 24 API tokens), for hand reading")
    w("")
    for mk, rows in rep["cutoff_sample"].items():
        w(f"**{mlabel[mk]}**")
        w("")
        for r in rows:
            w(f"- {TASK_NAME[r['task']]} `{r['request_id']}`: {r['n_api']} tokens (recount {r['n_recount']}), strict "
              f"{r['strict']}, content {r['content']}: `{json.dumps(r['text'], ensure_ascii=False)[:300]}`")
        w("")
    w("## Reading")
    w("")
    w("Refusals are scored as the text they hold (no answer: wrong); provider blocks (the route's moderation refused "
      "the input, no model output) and missing calls are left out of w and accuracy and counted above. Truncated "
      "replies (finish reason `length`) are kept and scored as they are. A forced reply is scored as the panel's: the "
      "prefill once in front of the reply, cut after its first `}` on GSM8K and BIG-Bench Hard, a box the stop left "
      "open read through `src/forced_stop.py`.")
    w("")
    w(f"Scorers: {json.dumps(rep['scorers'], default=str)}. Per-template values of Qwen4-Max, the panel and the "
      "ladder: `results/frontier_panel/closed_w_templates.csv`; everything in `closed_w.json`.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Quote
# ----------------------------------------------------------------------------------------------------------------

def read_listing(http, cfg: dict, model: dict) -> dict:
    """Free read of the endpoint's public listing (no key): pricing, discount, parameters."""
    url = cfg["base_url"].rstrip("/") + f"/models/{model['endpoints_slug']}/endpoints"
    resp = http.get_public(url)
    eps = (((resp.get("json") or {}).get("data") or {}).get("endpoints") or []) if resp.get("status") == 200 else []
    ep = next((e for e in eps if e.get("tag") == model["endpoint_tag"]), None)
    out = {"url": url, "status": resp.get("status"), "read_at": now_iso(), "found": ep is not None}
    if ep is not None:
        pr = ep.get("pricing") or {}
        out.update(provider=ep.get("provider_name"), tag=ep.get("tag"), pricing=pr, discount=pr.get("discount"),
                   supported_parameters=sorted(ep.get("supported_parameters") or []),
                   max_completion_tokens=ep.get("max_completion_tokens"))
    return out


def o200k_counter(tiktoken_path: str | None = None, cache: str | None = None):
    """tiktoken's o200k_base (the package and its cache may sit outside the venv: the path is appended to sys.path,
    after the venv's packages)."""
    import os
    if cache:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    if tiktoken_path and str(tiktoken_path) not in sys.path:
        sys.path.append(str(tiktoken_path))
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")

    def counts(texts):
        return [len(x) for x in enc.encode_ordinary_batch([t if isinstance(t, str) else "" for t in texts])]
    counts.name = f"tiktoken {getattr(tiktoken, '__version__', '?')} o200k_base"
    return counts


def strongest_models(cfg: dict, oth: pd.DataFrame, task: str, n_cells: int, root: Path = ROOT) -> list[str]:
    """The panel's strongest open models by ECI (manifests/d2_panel*.json) with every cell of the three arms."""
    ranked = [m for m, _ in cwp.strongest_models()]
    have = oth.groupby(["model_key", "arm"]).size().unstack(fill_value=0)
    full = [m for m in have.index if all(have.at[m, a] == n_cells for a in ARMS if a in have.columns)
            and all(a in have.columns for a in ARMS)]
    return [m for m in ranked if m in full][: int(cfg.get("quote", {}).get("strongest_models", 5))]


def measure_template_tokens(paths: Paths, cfg: dict, tok, root: Path = ROOT) -> dict:
    """API prompt tokens minus the recount of the messages (user message and prefill) on the prefill test's calls,
    per variant: the template tokens the route adds."""
    lines = [x for x in read_raw(paths.raw("prefill")) if x.get("event") == "result" and x.get("prompt_tokens")]
    out = {}
    if not lines:
        return out
    for t in TASKS:
        ids = sorted({x["request_id"] for x in lines if x.get("task") == t})
        if not ids:
            continue
        s = load_requests(screening_path(cfg, t, root), request_ids=ids).set_index("request_id")
        for x in lines:
            if x.get("task") != t or x["request_id"] not in s.index:
                continue
            n = tok.count(s.at[x["request_id"], "text"]) + tok.count(cfg["prefill"])
            d = int(x["prompt_tokens"]) - n
            v = out.setdefault(x.get("variant") or "?", {})
            v[d] = v.get(d, 0) + 1
    return {k: {str(d): c for d, c in sorted(v.items())} for k, v in out.items()}


def billed_ratio(paths: Paths, price: dict) -> dict:
    """OpenRouter's billed cost over tokens x the listed prices, on the prefill test's calls."""
    rs = []
    for x in read_raw(paths.raw("prefill")):
        if x.get("event") == "result" and x.get("cost_reported") is not None and x.get("completion_tokens") is not None:
            e = expected_cost(x, price)
            if e > 0:
                rs.append(float(x["cost_reported"]) / e)
    return {"calls": len(rs), "min": min(rs) if rs else None, "max": max(rs) if rs else None}


def build_quote(cfg: dict, man: dict, paths: Paths, model: dict, sel: dict, o200k, tok, listing: dict | None,
                panel_loader=None, root: Path = ROOT, pilot_reqs: list[dict] | None = None) -> dict:
    """Per arm and benchmark: calls, input tokens (o200k, and Qwen3.6-27B), output tokens at the panel's strongest
    models' lengths on the same cells and at the arm's cap, their cost at the listed (billed) prices, and the chunk
    guard's bound. sel: {arm: {benchmark: rows}}."""
    panel_loader = panel_loader or panel_table3
    pin, pout = float(model["price"]["prompt"]), float(model["price"]["completion"])
    measured = measure_template_tokens(paths, cfg, tok, root)
    mv = measured.get((cfg.get("forced") or {}).get("prefill_variant") or "")
    if mv:
        tt, tt_source = int(max(mv.items(), key=lambda kv: kv[1])[0]), "measured on the prefill test (mode)"
    else:
        tt, tt_source = int(cfg.get("quote", {}).get("template_tokens", 0)), "config quote.template_tokens"
    ov = int(cfg["run"]["prompt_overhead_tokens"])
    rows, models_used, dists = [], {}, {}
    for t in TASKS:
        m = man["tasks"][t]
        n_cells = len(m["items"]) * len(m["templates"])
        oth = panel_loader(t, m["items"], m["templates"], root)
        top = strongest_models(cfg, oth, t, n_cells, root)
        models_used[t] = top
        for arm in ARMS:
            df = sel[arm][t]
            texts = df["text"].tolist()
            pre = cfg["prefill"] if arm == "forced" else ""
            n = len(df)
            nb = [len(x.encode("utf-8")) + len(pre.encode("utf-8")) for x in texts]
            in_o = int(sum(o200k(texts))) + (o200k([pre])[0] * n if pre else 0)
            in_q = int(sum(tok.counts(texts))) + (tok.count(pre) * n if pre else 0)
            cap = arm_cap(cfg, arm, t)
            g = oth[(oth["arm"] == arm) & oth["model_key"].isin(top)]
            per = g.groupby(["config_id", "item_id"])["n_output_tokens"].mean()
            per = per.reindex(pd.MultiIndex.from_frame(df[["config_id", "item_id"]]))
            out_obs = float(per.sum()) if per.notna().all() else None
            x = g["n_output_tokens"].astype(float).to_numpy()
            dists[f"{arm}|{t}"] = {"mean": float(x.mean()) if len(x) else None,
                                   "median": float(np.median(x)) if len(x) else None,
                                   "p90": float(np.quantile(x, 0.9)) if len(x) else None,
                                   "share_at_cap": float((g["finish_reason"] == "length").mean()) if len(g) else None}
            out_cap = cap * n
            guard = float(sum((b + ov) * pin + cap * pout for b in nb))
            r = {"arm": arm, "task": t, "calls": n, "cap": cap, "in_o200k": in_o, "in_qwen": in_q,
                 "template_tokens": tt * n, "out_observed": out_obs, "out_cap": out_cap, "guard_usd": guard}
            for lab, k in (("o200k", "in_o200k"), ("qwen", "in_qwen")):
                inp = (r[k] + r["template_tokens"]) * pin
                r[f"usd_in_{lab}"] = inp
                r[f"usd_observed_{lab}"] = None if out_obs is None else inp + out_obs * pout
                r[f"usd_worst_{lab}"] = inp + out_cap * pout
            rows.append(r)

    def tot(sub, k):
        v = [r[k] for r in sub]
        return None if any(x is None for x in v) else float(sum(v))
    keys = ["calls", "in_o200k", "in_qwen", "template_tokens", "out_observed", "out_cap", "guard_usd",
            "usd_in_o200k", "usd_in_qwen", "usd_observed_o200k", "usd_observed_qwen", "usd_worst_o200k",
            "usd_worst_qwen"]
    per_arm = {a: {k: tot([r for r in rows if r["arm"] == a], k) for k in keys} for a in ARMS}
    total = {k: tot(rows, k) for k in keys}
    spent = {ph: ledger(read_raw(paths.raw(ph)))["counted_usd"] for ph in ("pilot", "prefill", "run")}
    pilot = None
    if pilot_reqs is not None:
        obs = {(r["arm"], r["task"]): (r["usd_observed_qwen"] / r["calls"] if r["usd_observed_qwen"] else None)
               for r in rows}
        est = 0.0
        for p in pilot_reqs:
            if p["kind"] == "control":
                est += upper_cost(p["n_bytes"], int(cfg["caps"]["control"]), model["price"], tt)
            else:
                v = obs.get((p["arm"], p["task"]))
                est += v if v is not None else p["reserve_usd"]
        pilot = {"calls": len(pilot_reqs), "worst_usd": float(sum(p["reserve_usd"] for p in pilot_reqs)),
                 "at_observed_usd": est, "limit_usd": cfg["limits"]["pilot"]}
    cot_share = None
    if total["usd_observed_qwen"]:
        cot_share = per_arm["cot"]["usd_observed_qwen"] / total["usd_observed_qwen"]
    return {"made_on": today(), "model": model["id"], "provider": model["provider"]["only"],
            "price": {"prompt": pin, "completion": pout, "cache_read": model["price"].get("cache_read")},
            "listing": listing, "billed_vs_listed_on_prefill_calls": billed_ratio(paths, model["price"]),
            "template_tokens": {"used": tt, "source": tt_source, "measured_on_prefill_calls": measured},
            "o200k": getattr(o200k, "name", "o200k"), "qwen_tokenizer": tok.name,
            "strongest_models": models_used, "output_length_distribution": dists, "rows": rows, "per_arm": per_arm,
            "total": total, "spent_so_far_usd": spent, "spent_total_usd": float(sum(spent.values())),
            "pilot": pilot, "cot_share_of_total_observed": cot_share,
            "caps": {"requested": cfg["caps"]["requested"], "forced": cfg["caps"]["forced"], "cot": cfg["caps"]["cot"]},
            "stop_sent": sorted(t for t in TASKS if forced_stop_sent(cfg, t))}


def _int(x) -> str:
    return "-" if x is None else f"{int(round(x)):,}"


def render_quote(q: dict) -> str:
    L = []
    w = L.append
    p, li = q["price"], q.get("listing") or {}
    w(f"# Qwen4-Max, three arms on OpenRouter: quote ({q['made_on']})")
    w("")
    w(f"Written by `scripts/closed_w.py quote`. Nothing was sent to a model and no key was read: the only network read "
      f"is OpenRouter's public listing of `{q['model']}`"
      + (f" ({li.get('url')}, HTTP {li.get('status')}, read {li.get('read_at')})" if li else " (not read: --offline)")
      + ". The design: every template of the four benchmarks on the same 100 items per benchmark, the panel's three "
        "arms, 13,100 calls per arm, 39,300 in all.")
    w("")
    w("## Price")
    w("")
    if li.get("found"):
        pr = li.get("pricing") or {}
        w(f"Listed for the `{li.get('tag')}` endpoint ({li.get('provider')}): prompt {pr.get('prompt')}, completion "
          f"{pr.get('completion')}, cache read {pr.get('input_cache_read')} USD per token, `discount` "
          f"{li.get('discount')}. The listed prices already carry the 5% discount ($1.50 and $4.50 per million less "
          "5%).")
    w(f"Prices used: ${p['prompt'] * 1e6:.3f} per million input tokens, ${p['completion'] * 1e6:.3f} per million output "
      "tokens, no cache discount.")
    br = q["billed_vs_listed_on_prefill_calls"]
    if br["calls"]:
        w(f"The discount applies to the billed price: on the {br['calls']} prefill-test calls OpenRouter's billed "
          f"`usage.cost` over tokens × these prices lies between {br['min']:.4f} and {br['max']:.4f}.")
    w("")
    w("## Inputs and outputs")
    w("")
    tm = q["template_tokens"]
    w(f"Input tokens are counted on the exact request texts of the request files (`manifests/closed_w_items.json` pins "
      f"them by hash): the user message, and on the forced arm also the prefill `\\boxed{{`. Two counts: {q['o200k']}, "
      f"and the Qwen3.6-27B tokenizer ({q['qwen_tokenizer']}). Both add {tm['used']} template tokens per call "
      f"({tm['source']}): on the prefill test the API's prompt count minus the Qwen3.6-27B recount of the same "
      f"messages was {json.dumps(tm['measured_on_prefill_calls'])} (variant: {{difference: calls}}), so the Qwen count "
      f"plus {tm['used']} is the billed prompt count of a forced call. The requested and chain-of-thought arms send no "
      "assistant message and may carry a few template tokens fewer; that is not measured.")
    w("")
    w("Output tokens, two assumptions: (1) observed, the mean over the panel's five strongest open models (by ECI, "
      "those with every cell of the three arms) of their own output tokens on the same cell, summed over the design's "
      "cells, from the scored tables at the panel's caps; (2) worst case, every reply at the arm's cap (requested "
      "1,024; forced 16, HotpotQA 64; chain of thought 3,072, HotpotQA 1,900). Thinking is off on every call; a reply "
      "billed with reasoning tokens would stop the run.")
    w("")
    w("| Benchmark | Strongest models used |")
    w("|---|---|")
    for t in TASKS:
        w(f"| {TASK_NAME[t]} | {', '.join(q['strongest_models'][t])} |")
    w("")
    w("## Cost per arm and benchmark")
    w("")
    w("| Arm | Benchmark | Calls | Input tok. o200k | Input tok. Qwen | Output tok. observed | Output tok. at cap | "
      "Observed (o200k in) | Observed (Qwen in) | Worst case (o200k in) | Worst case (Qwen in) | Chunk-guard bound |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in q["rows"]:
        w(f"| {r['arm']} | {TASK_NAME[r['task']]} | {r['calls']:,} | {_int(r['in_o200k'] + r['template_tokens'])} | "
          f"{_int(r['in_qwen'] + r['template_tokens'])} | {_int(r['out_observed'])} | {_int(r['out_cap'])} | "
          f"{usd(r['usd_observed_o200k'])} | {usd(r['usd_observed_qwen'])} | {usd(r['usd_worst_o200k'])} | "
          f"{usd(r['usd_worst_qwen'])} | {usd(r['guard_usd'])} |")
    for a in ARMS:
        v = q["per_arm"][a]
        w(f"| **{a}** | all four | {_int(v['calls'])} | {_int(v['in_o200k'] + v['template_tokens'])} | "
          f"{_int(v['in_qwen'] + v['template_tokens'])} | {_int(v['out_observed'])} | {_int(v['out_cap'])} | "
          f"**{usd(v['usd_observed_o200k'])}** | **{usd(v['usd_observed_qwen'])}** | {usd(v['usd_worst_o200k'])} | "
          f"{usd(v['usd_worst_qwen'])} | {usd(v['guard_usd'])} |")
    v = q["total"]
    w(f"| **Total** | three arms | {_int(v['calls'])} | {_int(v['in_o200k'] + v['template_tokens'])} | "
      f"{_int(v['in_qwen'] + v['template_tokens'])} | {_int(v['out_observed'])} | {_int(v['out_cap'])} | "
      f"**{usd(v['usd_observed_o200k'])}** | **{usd(v['usd_observed_qwen'])}** | **{usd(v['usd_worst_o200k'])}** | "
      f"**{usd(v['usd_worst_qwen'])}** | **{usd(v['guard_usd'])}** |")
    w("")
    if q.get("cot_share_of_total_observed") is not None:
        c = q["per_arm"]["cot"]
        w(f"The chain-of-thought arm dominates: {100 * q['cot_share_of_total_observed']:.0f}% of the observed total "
          f"({usd(c['usd_observed_qwen'])} of {usd(v['usd_observed_qwen'])}, Qwen inputs), and "
          f"{usd(c['usd_worst_qwen'])} of the {usd(v['usd_worst_qwen'])} worst case.")
        w("")
    w("## Limits")
    w("")
    sp = q["spent_so_far_usd"]
    w(f"Spent on this design so far (counted as the run's limit counts it): prefill test {usd(sp['prefill'])}, pilot "
      f"{usd(sp['pilot'])}, run {usd(sp['run'])}; {usd(q['spent_total_usd'])} in all. The run's `--limit-usd` counts "
      "this too, so the room for the run is the limit minus it.")
    if q.get("pilot"):
        pl = q["pilot"]
        w("")
        w(f"Pilot: {pl['calls']} calls (5 prompts per benchmark per arm, each twice; the stop control; two thinking-on "
          f"controls): worst case {usd(pl['worst_usd'])} against its ${pl['limit_usd']:.2f} limit, about "
          f"{usd(pl['at_observed_usd'])} at the observed lengths (the controls at 4,096 output tokens).")
    w("")
    w(f"Chunk-guard bound: the guard reserves for each call its message bytes plus 64 as input tokens and the whole cap "
      f"as output tokens, so a run of the full design under a limit below {usd(v['guard_usd'])} plus what is already "
      "counted cannot send its last chunks until bills come back below their reserve (which they do: answered calls "
      "count what OpenRouter bills). It is the guard's bound, not a projection; the hard stop at the limit holds "
      "whatever the limit.")
    w("")
    w("Not included: OpenRouter's charge on credit purchases (not read), caching (reads billed at "
      f"${(p.get('cache_read') or 0) * 1e6:.4f} per million where the provider caches; some prefill-test calls had "
      "cached input), and retries of failed calls (not billed).")
    return "\n".join(L) + "\n"


def cmd_quote(a, cfg: dict, paths: Paths, http=None, root: Path = ROOT, log=print, o200k=None, panel_loader=None,
              tok=None) -> int:
    man = load_manifest(paths)
    if "arms" not in man:
        log("refused: the manifest has no arms section; run `scripts/closed_w.py plan` first")
        return 2
    model = model_of(cfg, (getattr(a, "models", None) or default_model_keys(cfg))[0])
    sel = {arm: selected_arm(man, arm, load_arms_all(cfg, root, (arm,))[arm]) for arm in ARMS}
    listing = None
    if not getattr(a, "offline", False):
        http = http or Http(cfg["base_url"], None, cfg["run"]["timeout_s"])
        listing = read_listing(http, cfg, model)
        pr = (listing.get("pricing") or {}) if listing.get("found") else {}
        diff = {k: (pr.get(k), model["price"][k]) for k in ("prompt", "completion")
                if pr.get(k) is None or abs(float(pr[k]) - float(model["price"][k])) > 0.01 * float(model["price"][k])}
        if diff:
            log(f"the listing's prices differ from the config's: {diff}; the quote uses the config's")
            listing["differs_from_config"] = diff
    o200k = o200k or o200k_counter(getattr(a, "tiktoken_path", None), getattr(a, "tiktoken_cache", None))
    tok = tok or Tok(cfg, root)
    ctl_src = load_requests(screening_path(cfg, "math", root), request_ids=[man["control"]["row"]["request_id"]])
    pil = pilot_requests(cfg, man, sel, ctl_src, [model])
    q = build_quote(cfg, man, paths, model, sel, o200k, tok, listing, panel_loader, root, pil)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "qwen3arm_quote.json").write_text(dumps(q) + "\n", encoding="utf-8")
    (paths.results / "qwen3arm_quote.md").write_text(render_quote(q), encoding="utf-8")
    v = q["total"]
    log(f"quote: {_int(v['calls'])} calls; observed {usd(v['usd_observed_o200k'])} (o200k inputs), "
        f"{usd(v['usd_observed_qwen'])} (Qwen inputs); worst case {usd(v['usd_worst_o200k'])} / "
        f"{usd(v['usd_worst_qwen'])}; chunk-guard bound {usd(v['guard_usd'])}; "
        f"{rel(paths.results / 'qwen3arm_quote.md')}")
    return 0


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------

def refuse_paid(a, cfg: dict, phase: str) -> str | None:
    if not getattr(a, "approved", False):
        return f"{PHASE_CMD[phase]} is a paid phase: it needs --approved and --limit-usd"
    if getattr(a, "limit_usd", None) is None:
        return f"{PHASE_CMD[phase]} needs --limit-usd"
    if not a.limit_usd > 0:
        return "--limit-usd must be positive"
    cap = (cfg.get("limits") or {}).get(phase)
    if cap is not None and a.limit_usd > float(cap) + 1e-12:
        return f"--limit-usd {a.limit_usd} is above the config's limit for {PHASE_CMD[phase]} (${float(cap):.2f})"
    return None


DECODING_KEYS_V1 = ("requested_cap", "forced_caps", "prefill", "cut_at_first_brace", "sources")


def cmd_plan(a, cfg: dict, paths: Paths, root: Path = ROOT, log=print) -> int:
    dec = check_panel_decoding(cfg, root)
    reqs = load_all_requests(cfg, root)
    screening = {t: load_requests(screening_path(cfg, t, root)) for t in TASKS}
    levels = pd.read_parquet(Path(root) / "data" / "processed" / "math" / "items_screening.parquet",
                             columns=["item_id", "level"])
    sources = {}
    for t in TASKS:
        p, q = request_path(cfg, t, root), screening_path(cfg, t, root)
        sources[t] = {"requests": rel(p), "requests_sha256": sha256_file(p), "screening": rel(q),
                      "screening_sha256": sha256_file(q), "arm": ARM}
    n = int(cfg["design"]["items_per_benchmark"])
    math_templates = sorted(reqs["math"].loc[reqs["math"]["item_id"].isin(set(draw_items(
        "math", reqs["math"]["item_id"].unique().tolist(), n))), "config_id"].unique())
    control = pick_control(screening["math"], levels, math_templates)
    man = build_manifest(cfg, reqs, screening, control, sources)
    man["panel_decoding"] = {k: dec[k] for k in DECODING_KEYS_V1}
    # the counts must be those of Design B100 in the price record
    if paths.price_tokens.exists():
        per = json.loads(paths.price_tokens.read_text(encoding="utf-8"))["designs"][f"B{n}"]["per_task"]
        diff = {t: (per[t]["calls"], man["tasks"][t]["n_requests"]) for t in TASKS
                if per[t]["calls"] != man["tasks"][t]["n_requests"] or per[t]["templates"] != man["tasks"][t]["n_templates"]}
        if diff:
            raise SystemExit(f"the request list differs from closed_w_price.json's B{n} counts: {diff}")
        man["matches_closed_w_price"] = f"Design B{n} of {rel(paths.price_tokens)}: calls and templates per benchmark"
    reqs_by_arm = {"requested": reqs, **{arm: {t: load_arm_requests(cfg, arm, t, root) for t in TASKS}
                                         for arm in ARMS if arm != "requested"}}
    arms = build_arms_section(cfg, man, reqs_by_arm, root)
    arms["panel_decoding"] = dec
    arms = json.loads(dumps(arms))
    if paths.manifest.exists():
        old = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if base_body(old) != base_body(man):
            raise SystemExit(f"{rel(paths.manifest)} exists and differs from a fresh plan; it fixes the items before any "
                             "call, so it is not overwritten (move it aside only if no call was made under it)")
        if "arms" in old:
            if old["arms"] != arms:
                raise SystemExit(f"{rel(paths.manifest)}: its arms section differs from a fresh plan (a request file "
                                 "changed after `plan`); it is not overwritten")
            man = old
            log(f"{rel(paths.manifest)} unchanged (made {old['made_on']}, arms added {old.get('arms_added_on')})")
        else:
            man = dict(old, arms=arms, arms_added_on=today())
            paths.manifest.write_text(dumps(man) + "\n", encoding="utf-8")
            log(f"{rel(paths.manifest)}: arms section added (items, templates and pilot rows unchanged)")
    else:
        man["arms"], man["arms_added_on"] = arms, today()
        paths.manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest.write_text(dumps(man) + "\n", encoding="utf-8")
        log(f"wrote {rel(paths.manifest)}")
    models = [model_of(cfg, k) for k in default_model_keys(cfg)]
    ra = run_arms(cfg)
    per_model = man["n_requests_per_model"] * len(ra)
    log("Request list (per arm and model): " + "; ".join(
        f"{TASK_NAME[t]} {man['tasks'][t]['n_templates']} templates x {man['tasks'][t]['n_items']} items = "
        f"{man['tasks'][t]['n_requests']:,}" for t in TASKS)
        + f"; arms {', '.join(ra)}: {per_model:,} per model, {len(models) * per_model:,} in all, "
          f"{man['n_cells']} cells")
    sel = {arm: selected_arm(man, arm, reqs_by_arm[arm]) for arm in ra}
    rlist = run_requests(cfg, man, sel, models, ra)
    for arm in ra:
        w_ = sum(r["reserve_usd"] for r in rlist if r["arm"] == arm)
        log(f"  {arm}: chunk-guard bound ${w_:.2f} (bytes + {cfg['run']['prompt_overhead_tokens']} as input tokens, "
            f"every reply at the cap)")
    screening_ctl = load_requests(screening_path(cfg, "math", root), request_ids=[man["control"]["row"]["request_id"]])
    src = {arm: ({t: v for t, v in sel[arm].items()}) for arm in ra}
    pilot_reqs = pilot_requests(cfg, man, src, screening_ctl, models, ra)
    log(f"Pilot: {len(pilot_reqs)} calls ({cfg['design']['pilot_prompts_per_benchmark']} prompts x 4 benchmarks x "
        f"{len(ra)} arms x {cfg['design']['pilot_repeats']} sends, the stop control, "
        f"{cfg['design']['pilot_controls']} thinking-on controls, per model); worst case "
        f"${sum(r['reserve_usd'] for r in pilot_reqs):.3f} (limit ${cfg['limits']['pilot']:.2f})")
    log(f"Run: chunk-guard bound ${sum(r['reserve_usd'] for r in rlist):.2f} over {len(rlist):,} calls; no run ceiling "
        "in the config (set --limit-usd). Costs: scripts/closed_w.py quote")
    return 0


def _models(cfg, a) -> list[dict]:
    keys = getattr(a, "models", None) or default_model_keys(cfg)
    return [model_of(cfg, k) for k in keys]


def pilot_sources(cfg: dict, man: dict, root: Path, arms) -> dict:
    """{arm: {benchmark: rows}}: the requested arm's full lists (checked against the manifest's hash), the other
    arms' pilot rows only."""
    out = {}
    for arm in arms:
        if arm == "requested":
            out[arm] = selected(man, load_all_requests(cfg, root))
        else:
            if "arms" not in man:
                raise SystemExit("the manifest has no arms section: run `scripts/closed_w.py plan` first")
            out[arm] = {t: load_arm_requests(cfg, arm, t, root,
                                             request_ids=[r["request_id"] for r in man["arms"]["pilot"][arm][t]])
                        for t in TASKS}
    return out


def cmd_pilot(a, cfg: dict, paths: Paths, http=None, scorer=score_rows, tok: Tok | None = None, sleep=time.sleep,
              root: Path = ROOT, log=print) -> int:
    why = refuse_paid(a, cfg, "pilot")
    if why:
        log("refused: " + why)
        return 2
    check_panel_decoding(cfg, root)
    man = load_manifest(paths)
    csha = config_sha(cfg)
    arms = run_arms(cfg)
    src = pilot_sources(cfg, man, root, arms)
    ctl_src = load_requests(screening_path(cfg, "math", root), request_ids=[man["control"]["row"]["request_id"]])
    models = _models(cfg, a)
    plist = pilot_requests(cfg, man, src, ctl_src, models, arms)
    if http is None:
        http = Http(cfg["base_url"], read_key(paths.env, cfg["key_env"]), cfg["run"]["timeout_s"])
    raw = paths.raw("pilot")
    lines = read_raw(raw)
    done = done_keys(lines, csha)
    pre = {m["key"]: preflight(http, cfg, m) for m in models}
    for mk, c in pre.items():
        log(f"preflight {mk}: {'pass' if c['pass'] else 'FAIL'}")
    if not all(c["pass"] for c in pre.values()):
        res = {"made_on": today(), "config_sha": csha, "passed": False, "preflight": pre,
               "failed": [f"{mk}: preflight: {json.dumps(c['detail'], default=str)[:500]}" for mk, c in pre.items()
                          if not c["pass"]]}
        paths.results.mkdir(parents=True, exist_ok=True)
        (paths.results / "closed_w_pilot.json").write_text(dumps(res) + "\n", encoding="utf-8")
        md = (f"# Qwen4-Max three arms: pilot ({res['made_on']})\n\nVerdict: **FAIL** before any paid call. The public "
              "listing of a pinned endpoint does not match the configuration:\n\n" +
              "".join(f"- {f}\n" for f in res["failed"]) +
              "\nNothing was sent with the key. `run` refuses until a pilot under this configuration passes.\n")
        (paths.results / "closed_w_pilot.md").write_text(md, encoding="utf-8")
        (paths.results / "closed_w_pilot_clean.md").write_text(clean_md(md), encoding="utf-8")
        log("pilot stopped before any paid call: " + "; ".join(res["failed"]))
        return 3
    t_start = now_iso()
    runner = Runner(http, cfg, raw, "pilot", a.limit_usd, ledger(lines)["counted_usd"], csha, sleep=sleep, log=log)
    summ = runner.run([r for r in plist if r["key"] not in done])
    t_end = now_iso()
    lines = read_raw(raw)
    fin = final_by_key(lines, csha)
    tok = tok or Tok(cfg, root)
    for r in plist:
        x = fin.get(r["key"])
        if x is not None and x.get("outcome") in ACCEPTED:
            x["n_local"] = tok.count(x.get("text"))
    off = [fin[r["key"]] for r in plist if r["kind"] in ARMS and _answered(fin.get(r["key"]))]
    scored = None
    score_problem = None
    if off:
        try:
            scored, _ = scorer(pd.DataFrame(off))
        except Exception as e:  # noqa: BLE001
            score_problem = f"{type(e).__name__}: {e}"[:300]
    res = {"made_on": today(), "config_sha": csha, "raw": rel(raw), "limit": a.limit_usd, "models": {}, "arms": arms,
           "preflight": pre, "runner": summ, "failed": []}
    for m in models:
        checks = [pre[m["key"]]] + pilot_model_checks(cfg, m, plist, fin, scored)
        if score_problem:
            checks.append(_check("scoring", False, score_problem))
        calls = []
        for r in plist:
            if r["model_key"] != m["key"]:
                continue
            x = fin.get(r["key"]) or {}
            calls.append({"task": r["task"], "request_id": r["request_id"], "kind": r["kind"], "rep": r["rep"],
                          "outcome": x.get("outcome"), "finish_reason": x.get("finish_reason"),
                          "prompt_tokens": x.get("prompt_tokens"), "n_api": api_count(x) if x else None,
                          "n_local": x.get("n_local"), "reasoning_tokens": x.get("reasoning_tokens"),
                          "cost_usd": x.get("cost_usd"), "provider": x.get("provider"),
                          "served_model": x.get("served_model")})
        ok = all(c["pass"] for c in checks)
        res["models"][m["key"]] = {"label": m["label"], "id": m["id"], "passed": ok, "checks": checks, "calls": calls}
        res["failed"] += [f"{m['key']}: {c['name']}: {json.dumps(c['detail'], default=str)[:400]}"
                          for c in checks if not c["pass"]]
    pj = projection(cfg, man, plist, fin, models)
    res["projection"] = pj
    lim = (cfg.get("limits") or {}).get("run")
    gchk = [_check("projected_full_cost", pj["total"] is not None and (lim is None or pj["total"] <= float(lim)),
                   {"projected_usd": pj["total"], "high_usd": pj["total_high"],
                    "run_limit_usd": lim if lim is not None else "set by --limit-usd"})]
    if summ.get("stop_reason"):
        gchk.append(_check("pilot_completed", False, summ["stop_reason"]))
    res["global_checks"] = gchk
    res["failed"] += [f"{c['name']}: {json.dumps(c['detail'], default=str)[:400]}" for c in gchk if not c["pass"]]
    res["passed"] = not res["failed"] and all(v["passed"] for v in res["models"].values())
    res["models_passed"] = [k for k, v in res["models"].items() if v["passed"]]
    led = ledger(lines)
    res["spend"] = {"billed_usd": led["billed_usd"], "counted_usd": led["counted_usd"]}
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "closed_w_pilot.json").write_text(dumps(res) + "\n", encoding="utf-8")
    md = render_pilot(res)
    (paths.results / "closed_w_pilot.md").write_text(md, encoding="utf-8")
    (paths.results / "closed_w_pilot_clean.md").write_text(clean_md(md), encoding="utf-8")
    entry = budget_entry("pilot", cfg, lines, runner.session, summ, t_start, t_end,
                         f"results/frontier_panel/closed_w_pilot.md; verdict {'pass' if res['passed'] else 'fail'}")
    if not append_budget(paths.budget, entry):
        log("config/budget.yaml not updated (layout not recognised); add this entry by hand:\n" +
            yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    log(f"pilot {'PASSED' if res['passed'] else 'FAILED'}; billed ${led['billed_usd']:.4f}; "
        f"{rel(paths.results / 'closed_w_pilot.md')}")
    if not res["passed"]:
        for f in res["failed"]:
            log("  failed: " + f)
        return 3
    return 0


def cmd_prefill(a, cfg: dict, paths: Paths, http=None, sleep=time.sleep, root: Path = ROOT, log=print) -> int:
    why = refuse_paid(a, cfg, "prefill")
    if why:
        log("refused: " + why)
        return 2
    check_panel_decoding(cfg, root)
    man = load_manifest(paths)
    csha = config_sha(cfg)
    model = model_of(cfg, cfg["design"]["prefill_model"])
    screening = {t: load_requests(screening_path(cfg, t, root),
                                  request_ids=[r["request_id"] for r in man["prefill"]["rows"][t]]) for t in TASKS}
    plist = prefill_requests(cfg, man, screening, model)
    if http is None:
        http = Http(cfg["base_url"], read_key(paths.env, cfg["key_env"]), cfg["run"]["timeout_s"])
    raw = paths.raw("prefill")
    lines = read_raw(raw)
    done = done_keys(lines, csha)
    t_start = now_iso()
    runner = Runner(http, cfg, raw, "prefill", a.limit_usd, ledger(lines)["counted_usd"], csha, stop_on_error=False,
                    sleep=sleep, log=log)
    summ = runner.run([r for r in plist if r["key"] not in done])
    t_end = now_iso()
    lines = read_raw(raw)
    fin = final_by_key(lines, csha)
    s = prefill_summary(cfg, plist, fin)
    led = ledger(lines)
    res = {"made_on": today(), "config_sha": csha, "model": model["id"], "provider": model["provider"]["only"],
           "prefill": cfg["prefill"], "raw": rel(raw), "limit": a.limit_usd,
           "min_share": cfg["checks"]["prefill_min_continuation"], "summary": s, "runner": summ,
           "spend": {"billed_usd": led["billed_usd"], "counted_usd": led["counted_usd"]}}
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "qwen_prefill_test.json").write_text(dumps(res) + "\n", encoding="utf-8")
    md = render_prefill(res)
    (paths.results / "qwen_prefill_test.md").write_text(md, encoding="utf-8")
    (paths.results / "qwen_prefill_test_clean.md").write_text(clean_md(md), encoding="utf-8")
    entry = budget_entry("prefill", cfg, lines, runner.session, summ, t_start, t_end,
                         f"results/frontier_panel/qwen_prefill_test.md; verdict {'pass' if s['passed'] else 'fail'}"
                         + (f" ({', '.join(s['passing_variants'])})" if s["passed"] else ""))
    if not append_budget(paths.budget, entry):
        log("config/budget.yaml not updated (layout not recognised); add this entry by hand:\n" +
            yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    log(f"prefill test {'PASSED' if s['passed'] else 'FAILED'}; billed ${led['billed_usd']:.4f}; "
        f"{rel(paths.results / 'qwen_prefill_test.md')}")
    return 0 if s["passed"] else 3


def run_check(req: dict, rec: dict) -> list[str]:
    """Problems that stop the run: the off switch failed, or the reply came from another route."""
    out = []
    if req["reasoning"] == "off" and (int(rec.get("reasoning_tokens") or 0) > 0 or
                                      (rec.get("reasoning_text") or "").strip()):
        out.append(f"reasoning with thinking off ({rec.get('reasoning_tokens')} tokens)")
    if rec.get("outcome") != "blocked":
        if rec.get("provider") != req["expected_provider"]:
            out.append(f"served by {rec.get('provider')!r}, pinned {req['expected_provider']!r}")
        if not str(rec.get("served_model") or "").startswith(req["served_model_prefix"]):
            out.append(f"served model {rec.get('served_model')!r}")
    return out


class ForcedMonitor:
    """Stops the run when the forced arm's replies start with the prefill more often than the pilot allowed: the
    route stopped honouring partial mode (it restarts the box, or echoes the prefill), and the prefix would be counted
    twice. Counts this session's forced replies."""

    def __init__(self, cfg: dict):
        ck = cfg["checks"]
        self.prefill = cfg["prefill"]
        self.max_share = float(ck.get("run_forced_restart_max_share", 0.1))
        self.min_n = int(ck.get("run_forced_restart_min_n", 40))
        self.n = self.starts = 0

    def __call__(self, rec: dict):
        if rec.get("kind") != "forced" or rec.get("outcome") != "reply":
            return None
        self.n += 1
        t = rec.get("text")
        self.starts += isinstance(t, str) and t.startswith(self.prefill)
        if self.n >= self.min_n and self.starts / self.n > self.max_share:
            return (f"{self.starts} of {self.n} forced replies start with the prefill "
                    f"(more than {self.max_share:.0%}): the route no longer continues the open box")
        return None


def run_requests(cfg: dict, man: dict, sel: dict, models: list[dict], arms=None) -> list[dict]:
    """Every (benchmark, template, item) unit in sha256 order (a run stopped early still covers every cell), the
    arms of a unit back to back (run_arms order) and the models' calls of an arm next to each other. sel:
    {arm: {benchmark: rows}}, or the requested arm's {benchmark: rows} alone."""
    if set(sel) <= set(TASKS):
        sel = {"requested": sel}
    arms = list(arms or [a for a in run_arms(cfg) if a in sel])
    units = []
    for t in TASKS:
        ix = {a: sel[a][t].set_index(["config_id", "item_id"]) for a in arms}
        for r in sel[arms[0]][t].itertuples():
            rows = {}
            for a in arms:
                x = ix[a].loc[(r.config_id, r.item_id)]
                rows[a] = {"request_id": x["request_id"], "config_id": r.config_id, "item_id": r.item_id,
                           "text": x["text"]}
            units.append((hkey("closed_w/run", t, r.config_id, r.item_id), t, rows))
    units.sort(key=lambda u: u[0])
    return [make_request(cfg, m, t, rows[a], a) for _, t, rows in units for a in arms for m in models]


def prefill_gate(cfg: dict, paths: Paths) -> str | None:
    """The forced arm runs only with the assistant-message variant that passed the prefill test."""
    f = cfg.get("forced") or {}
    p = paths.prefill_result
    if not p.exists():
        return f"no prefill test result ({rel(p)} absent)"
    s = json.loads(p.read_text(encoding="utf-8")).get("summary") or {}
    name = f.get("prefill_variant")
    v = (s.get("variants") or {}).get(name) or {}
    if name not in (s.get("passing_variants") or []):
        return f"the forced arm's variant {name!r} did not pass the prefill test ({rel(p)})"
    if (v.get("message_extra") or {}) != (f.get("message_extra") or {}):
        return (f"the forced arm's assistant fields {f.get('message_extra')} differ from those of the variant that "
                f"passed ({v.get('message_extra')})")
    return None


def cmd_run(a, cfg: dict, paths: Paths, http=None, sleep=time.sleep, root: Path = ROOT, log=print) -> int:
    why = refuse_paid(a, cfg, "run")
    if why:
        log("refused: " + why)
        return 2
    check_panel_decoding(cfg, root)
    man = load_manifest(paths)
    csha = config_sha(cfg)
    pj = paths.results / "closed_w_pilot.json"
    pilot = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else None
    models = _models(cfg, a)
    arms = list(getattr(a, "arms", None) or run_arms(cfg))
    bad = [x for x in arms if x not in run_arms(cfg)]
    if bad:
        log(f"refused: arms {bad} are not among the configuration's run arms {run_arms(cfg)}")
        return 2
    if pilot is None:
        log(f"refused: no pilot ({rel(pj)} absent); run `pilot` first")
        return 2
    if pilot.get("config_sha") != csha:
        log("refused: the pilot ran under another configuration (config/closed_w.yaml changed since); run the pilot again")
        return 2
    not_passed = [m["key"] for m in models if m["key"] not in (pilot.get("models_passed") or [])]
    if not pilot.get("passed") or not_passed:
        log(f"refused: the pilot did not pass (models not passed: {not_passed or 'none'}; see "
            f"{rel(paths.results / 'closed_w_pilot.md')})")
        return 2
    if "forced" in arms:
        why = prefill_gate(cfg, paths)
        if why:
            log("refused: " + why)
            return 2
    raw = paths.raw("run")
    lines = read_raw(raw)
    other = {x.get("config_sha") for x in lines if x.get("event") == "result"} - {csha}
    if other:
        log("refused: the run's raw file holds replies under another configuration")
        return 2
    allr = load_arms_all(cfg, root, arms)
    sel = {arm: selected_arm(man, arm, allr[arm]) for arm in arms}
    rlist = run_requests(cfg, man, sel, models, arms)
    done = done_keys(lines, csha)
    todo = [r for r in rlist if r["key"] not in done]
    spent = counted_spend(paths)
    counted = sum(v["counted_usd"] for v in spent.values())
    log(f"run: {len(rlist):,} calls in the design ({', '.join(arms)}), {len(done):,} answered before, {len(todo):,} to "
        f"send; counted so far ${counted:.4f} (pilot ${spent['pilot']['counted_usd']:.4f}, prefill "
        f"${spent['prefill']['counted_usd']:.4f}, run ${spent['run']['counted_usd']:.4f}) of ${a.limit_usd:.2f}")
    proj = (pilot.get("projection") or {}).get("total")
    if proj is not None and counted + proj > a.limit_usd:
        log(f"note: the pilot projects ${proj:.2f} for the full run; with ${counted:.4f} counted the limit "
            f"${a.limit_usd:.2f} stops it before the end (rerun with a new limit to resume)")
    if http is None:
        http = Http(cfg["base_url"], read_key(paths.env, cfg["key_env"]), cfg["run"]["timeout_s"])
    t_start = now_iso()
    runner = Runner(http, cfg, raw, "run", a.limit_usd, counted, csha, check=run_check, sleep=sleep, log=log,
                    threads=getattr(a, "threads", None), chunk=getattr(a, "chunk", None), monitor=ForcedMonitor(cfg))
    summ = runner.run(todo)
    t_end = now_iso()
    lines = read_raw(raw)
    done_after = done_keys(lines, csha)   # once: inside the comprehension it was recomputed per request (quadratic)
    left = len([r for r in rlist if r["key"] not in done_after])
    entry = budget_entry("run", cfg, lines, runner.session, summ, t_start, t_end,
                         f"{len(rlist) - left:,} of {len(rlist):,} calls answered after this session"
                         + (f"; stopped: {summ['stop_reason']}" if summ.get("stop_reason") else "")
                         + "; report: scripts/closed_w.py report -> results/frontier_panel/closed_w.md")
    if not append_budget(paths.budget, entry):
        log("config/budget.yaml not updated (layout not recognised); add this entry by hand:\n" +
            yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    if summ.get("stop_reason"):
        log(f"run stopped: {summ['stop_reason']}; {left:,} calls left (rerun the same command to resume)")
        return 3
    log(f"run done: {len(rlist) - left:,} of {len(rlist):,} answered, {left} left; billed ${summ['billed_usd']:.4f}")
    return 0 if left == 0 else 3


def cmd_report(a, cfg: dict, paths: Paths, scorer=score_rows, tok: Tok | None = None, panel_loader=None,
               open_loader=None, root: Path = ROOT, log=print, ladder_loader=None) -> int:
    man = load_manifest(paths)
    rep = build_report(cfg, man, paths, scorer=scorer, tok=tok, open_texts=getattr(a, "open_texts", False),
                       panel_loader=panel_loader, open_loader=open_loader, root=root, ladder_loader=ladder_loader)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "closed_w.json").write_text(dumps(rep) + "\n", encoding="utf-8")
    md = render_report(rep, cfg, man)
    (paths.results / "closed_w.md").write_text(md, encoding="utf-8")
    (paths.results / "closed_w_clean.md").write_text(clean_md(md), encoding="utf-8")
    pd.DataFrame(rep["per_template"]).to_csv(paths.results / "closed_w_templates.csv", index=False)
    log(f"wrote {rel(paths.results / 'closed_w.md')}, closed_w_clean.md, closed_w.json, closed_w_templates.csv")
    return 0


def main(argv=None, *, http=None, scorer=None, tok=None, paths: Paths | None = None, sleep=None, panel_loader=None,
         open_loader=None, root: Path = ROOT, log=print, ladder_loader=None, o200k=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    q = sub.add_parser("quote")
    q.add_argument("--offline", action="store_true", help="do not read the public listing; the config's prices")
    q.add_argument("--tiktoken-path", default=None, help="folder holding the tiktoken package, if not in the venv")
    q.add_argument("--tiktoken-cache", default=None, help="TIKTOKEN_CACHE_DIR holding o200k_base")
    for name in ("pilot", "prefill-test", "run"):
        p = sub.add_parser(name)
        p.add_argument("--approved", action="store_true", help="approval for this paid phase")
        p.add_argument("--limit-usd", type=float, default=None, help="hard spend limit (USD)")
        if name != "prefill-test":
            p.add_argument("--models", nargs="+", default=None, help="model keys (default: the config's default_models)")
        if name == "run":
            p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS),
                           help="arms to send (default: the config's run_arms)")
            p.add_argument("--threads", type=int, default=None)
            p.add_argument("--chunk", type=int, default=None)
    r = sub.add_parser("report")
    r.add_argument("--open-texts", action="store_true", help="recount the panel's texts with the same tokenizer")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    paths = paths or Paths(cfg, root)
    sleep = sleep or time.sleep
    if a.cmd == "plan":
        return cmd_plan(a, cfg, paths, root, log)
    if a.cmd == "quote":
        return cmd_quote(a, cfg, paths, http, root, log, o200k=o200k, panel_loader=panel_loader, tok=tok)
    if a.cmd == "pilot":
        return cmd_pilot(a, cfg, paths, http, scorer or score_rows, tok, sleep, root, log)
    if a.cmd == "prefill-test":
        return cmd_prefill(a, cfg, paths, http, sleep, root, log)
    if a.cmd == "run":
        return cmd_run(a, cfg, paths, http, sleep, root, log)
    return cmd_report(a, cfg, paths, scorer or score_rows, tok, panel_loader, open_loader, root, log, ladder_loader)


if __name__ == "__main__":
    raise SystemExit(main())
