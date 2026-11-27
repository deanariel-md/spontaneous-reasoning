r"""DeepSeek V4-Flash on Nebius at the main panel's design (thinking off).

Model: deepseek-ai/DeepSeek-V4.1-Flash-1017, the id Nebius names as the replacement for V4-Flash (served as
nvidia/DeepSeek-V4.1-Flash-1017-NVFP4, fp4). Every call goes to /v1/completions with the prompt rendered locally by the
model's own chat template (scripts/nebius_prefill_test.py, Renderer), thinking off. This is how the live prefill test
sent the forced arm; here the requested and chain-of-thought arms use the same route, so the rendering is the same text
for all three arms and thinking is off by the template itself.

Design: exactly the main panel's request rows, read from its request files (the text of every model's file is the same;
the reference file is qwen3-8b's):
  GSM8K        data/requests/gsm8k/<ref>.parquet (direct_answer, zero_shot_cot) + data/requests/forced/gsm8k/<ref>
  MATH-500     data/requests/math/<ref>.parquet + data/requests/forced/math/<ref>
  BIG-Bench H. data/requests/bbh/<ref>.parquet + data/requests/forced/bbh/<ref>
  HotpotQA     data/requests/hotpot/block_a and block_b/<ref>.parquet, arms direct_answer, forced_direct,
               zero_shot_cot (the five-shot forced arm is not one of the three arms and is not run)
Decoding as the main panel's run logs: greedy (temperature 0, top_p 1); caps requested 1,024, chain of thought 3,072
(HotpotQA 1,900), forced 16 (HotpotQA 64) after the prefill "\boxed{". The GSM8K and BIG-Bench Hard forced runs stopped
at the first "}" (kept in the output); here no stop string is sent and the reply is cut after its first "}" before
scoring, the rule of config/decoding_arm_wording.yaml: greedy decoding writes the same tokens up to that point.

Complete design: the DeepSeek rung of the size ladder runs the ladder's
design, the three arms above plus the five-shot arms (five-shot forced on all four benchmarks, five-shot requested on
GSM8K and BIG-Bench Hard), from data/requests/ladder/ (load_complete), with the ladder's caps (7,000 on the requested,
chain-of-thought and five-shot requested arms; the panel's cap kept for the cut reading). Five-shot prompts are
multi-turn and go through the same template and prefill (PanelRenderer.prompt_messages). Batch is refused for the
account, so the rest of the complete design runs in real time (hard limit $120 for the Nebius total): `run --design complete`, section "Real time, complete design" below, with a
rolling rate limit (600 calls and 1.2 M tokens per minute; the model allows 1,000 and 2 M) and the same spend meter.

Order: one unit is (benchmark, configuration, item); its three arms go back to back in one thread (requested, forced,
chain of thought), so the shared prompt opening can be billed at the cached rate. 16 threads by default.

Stops: (1) a hard spend limit, $10.00 by default, counting the live test's replies
(results/frontier_panel/nebius_prefill_raw.jsonl); before each call its largest possible cost (prompt at the full input
price, the whole cap at the output price) is reserved, so the limit holds with calls in flight. (2) The pilot: the first
5% of calls, spread across benchmarks, arms and templates (per benchmark and configuration, the first ceil(5%) of its
items in sha256 order). From the real bill the full cost is projected per benchmark and arm; `run --phase full` refuses
unless the projected total is at most the limit and the forced pass rate is at least 0.9 on every benchmark, or
--approved is given.

The key is read from the environment variable NEBIUS_API_KEY and never printed, logged or written; error
text is redacted. Replies are appended to data/frontier_panel/nebius_panel_raw.jsonl keyed by benchmark|request id; a
restart skips keys already answered.

Usage:
    python scripts/nebius_panel.py plan                 # counts, pilot size; no network
    python scripts/nebius_panel.py run --phase pilot    # the 5% pilot, then the projection
    python scripts/nebius_panel.py run --phase full     # only if the projection allows it
    python scripts/nebius_panel.py report               # score and write the reports
  complete design, real time (limit $120 for the Nebius total; --limit-usd above 120 is refused):
    python scripts/nebius_panel.py run --design complete --project   # cost; no network
    python scripts/nebius_panel.py run --design complete --test 20   # once, under $0.05
    python scripts/nebius_panel.py run --design complete             # the rest; resumable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_prefill_test as npt  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_KEY = "deepseek-v4.1-flash-1017"
SERVED_AS = "nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 (fp4, NVFP4), revision 3ae9c0d3fbbea5370bd06e679d22044db15883f1"
BASE_URL = npt.DEFAULT_BASE_URL
REF_MODEL = "qwen3-8b"
TASKS = ("gsm8k", "math", "bbh", "hotpot")
ARMS = ("direct_answer", "forced_direct", "zero_shot_cot")   # the order inside a unit: shared openings back to back
ARM_LABEL = {"direct_answer": "requested", "forced_direct": "forced", "zero_shot_cot": "cot"}
PREFILL = "\\boxed{"
CAPS = {"gsm8k": {"direct_answer": 1024, "zero_shot_cot": 3072, "forced_direct": 16},
        "math": {"direct_answer": 1024, "zero_shot_cot": 3072, "forced_direct": 16},
        "bbh": {"direct_answer": 1024, "zero_shot_cot": 3072, "forced_direct": 16},
        "hotpot": {"direct_answer": 1024, "zero_shot_cot": 1900, "forced_direct": 64}}
CUT_AT_FIRST_BRACE = {"gsm8k", "bbh"}        # their collection's forced runs stopped at the first "}"
# CAPS are the panel's caps, at which the real-time pilot ran. The complete design (load_complete, below) raises the
# requested and chain-of-thought caps to the ladder's 7,000 and keeps these as the cut reading's caps.
CAP_SOURCES = {"gsm8k": "run logs of data/d2_out/collection/<model> (config/decoding.yaml) and data/d2_out/forced/gsm8k "
                        "(config/decoding_forced.yaml)",
               "math": "data/d2_out/math/collection (config/decoding.yaml) and data/d2_out/math/forced "
                       "(config/decoding_forced_math.yaml)",
               "bbh": "data/d2_out/bbh/collection (config/decoding.yaml) and data/d2_out/forced/bbh "
                      "(config/decoding_forced.yaml)",
               "hotpot": "data/d2_out/hotpot/block_a, block_b (config/decoding_hotpot.yaml)"}
PRICE = {"input": 0.30, "cached_input": 0.006, "output": 1.20, "date": "2026-11-22",
         "source": "https://tokenfactory.nebius.com/api/public/models_info (flavor 'cheap', us-north1, quantization "
                   "fp4; input_price_per_million_tokens 0.3, input_cached_price_per_million_tokens 0.006, "
                   "output_price_per_million_tokens 1.2), read 2026-11-22"}
LIMIT_USD = 10.0
PILOT_SHARE = 0.05
PASS_MIN = 0.9
THREADS = 16
MAX_TRIES = 8
MAX_CONSECUTIVE_FAILURES = 40
SALT = "nebius-panel-pilot/1"

DATA = ROOT / "data" / "frontier_panel"
RAW = DATA / "nebius_panel_raw.jsonl"
LIVE_TEST_RAW = ROOT / "results" / "frontier_panel" / "nebius_prefill_raw.jsonl"
RES = ROOT / "results" / "frontier_panel"
PROJECTION = RES / "nebius_panel_projection.json"
SAMPLERS = {t: ROOT / "manifests" / f"sampler_{t}.json" for t in TASKS}


# ----------------------------------------------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------------------------------------------

def source_files(task: str, ref: str = REF_MODEL) -> dict:
    r = ROOT / "data" / "requests"
    if task == "hotpot":
        return {"blocks": [r / "hotpot" / "block_a" / f"{ref}.parquet", r / "hotpot" / "block_b" / f"{ref}.parquet"]}
    return {"collection": r / task / f"{ref}.parquet", "forced": r / "forced" / task / f"{ref}.parquet"}


def user_text(messages_json: str) -> str:
    m = json.loads(messages_json)
    if len(m) != 1 or m[0].get("role") != "user":
        raise ValueError(f"expected one user message, got {[x.get('role') for x in m]}")
    return m[0]["content"]


def base_id(request_id: str) -> str:
    return request_id[:-3] if request_id.endswith("|fd") else request_id


def load_requests(task: str, ref: str = REF_MODEL) -> pd.DataFrame:
    """The main panel's rows for one benchmark: request_id, arm, config_id, item_id (str), messages_json. Checks that
    each forced row's messages equal its requested row's (the forced build rule)."""
    cols = ["request_id", "arm", "config_id", "item_id", "messages_json"]
    src = source_files(task, ref)
    if task == "hotpot":
        df = pd.concat([pd.read_parquet(p, columns=cols, filters=[("arm", "in", list(ARMS))])
                        for p in src["blocks"]], ignore_index=True)
    else:
        df = pd.concat([pd.read_parquet(src["collection"], columns=cols), pd.read_parquet(src["forced"], columns=cols)],
                       ignore_index=True)
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    df = df[df["arm"].isin(ARMS)].reset_index(drop=True)
    if df["request_id"].duplicated().any():
        raise SystemExit(f"{task}: duplicated request ids in the source files")
    d = df[df["arm"] == "direct_answer"].set_index("request_id")["messages_json"]
    f = df[df["arm"] == "forced_direct"]
    mism = int((f["request_id"].map(base_id).map(d) != f["messages_json"]).sum())
    if mism:
        raise SystemExit(f"{task}: {mism} forced rows differ from their requested row")
    n = df.groupby("arm").size()
    if not (n.get("direct_answer") == n.get("forced_direct") == n.get("zero_shot_cot")):
        raise SystemExit(f"{task}: arms differ in size {n.to_dict()}")
    df["task"] = task
    df["key"] = task + "|" + df["request_id"]
    return df


def load_all(tasks=TASKS, ref: str = REF_MODEL) -> pd.DataFrame:
    return pd.concat([load_requests(t, ref) for t in tasks], ignore_index=True)


def unit_hash(task: str, config_id: str, item_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{task}|{config_id}|{item_id}".encode()).hexdigest()


def units_of(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, config_id, item_id) with its sha256 order key."""
    u = df[["task", "config_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    u["h"] = [unit_hash(t, c, i) for t, c, i in zip(u["task"], u["config_id"], u["item_id"])]
    return u


def pilot_units(u: pd.DataFrame, share: float = PILOT_SHARE) -> pd.DataFrame:
    """Per benchmark and configuration, the first ceil(share x items) items in sha256 order: every benchmark, arm and
    template is in the pilot in proportion."""
    parts = []
    for _, g in u.groupby(["task", "config_id"], sort=True):
        k = math.ceil(share * len(g))
        parts.append(g.sort_values("h").head(k))
    return pd.concat(parts, ignore_index=True)


def dispatch_order(u: pd.DataFrame) -> pd.DataFrame:
    """Units in sha256 order, so a run stopped early still covers every benchmark and template."""
    return u.sort_values("h").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------------------

class PanelRenderer:
    """The model's own chat template (pinned in nebius_prefill_test.RENDER), thinking off; the forced arm ends with
    the opened box inside the assistant turn."""

    def __init__(self):
        if npt.check_files(MODEL) is not None:
            raise SystemExit(f"template files: {npt.check_files(MODEL)} (run nebius_prefill_test.py --fetch-templates)")
        self.r = npt.Renderer(MODEL)

    def prompt(self, text: str, arm: str) -> str:
        msgs = npt.forced_messages(text) if arm == "forced_direct" else npt.chat_messages(text)
        p = self.r.render(msgs, thinking=False)
        ref = npt.reference_dsv4_chat(text, PREFILL if arm == "forced_direct" else None)
        if p != ref:
            raise SystemExit(f"rendering differs from DeepSeek-V4's chat-mode format for an {arm} prompt")
        return p

    def prompt_messages(self, messages: list[dict], arm: str) -> str:
        """Any user/assistant exchange ending on a user turn (the five-shot arms carry five worked pairs before the
        question), rendered by the same template, thinking off; the forced arms end with the opened box inside the
        assistant turn. Checked against DeepSeek-V4's documented multi-turn chat format (reference_chat)."""
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        forced = arm in FORCED_ARMS
        p = self.r.render(msgs + ([{"role": "assistant", "content": PREFILL}] if forced else []), thinking=False)
        if p != reference_chat(msgs, PREFILL if forced else None):
            raise SystemExit(f"rendering differs from DeepSeek-V4's chat-mode format for an {arm} prompt")
        return p

    def count(self, text: str) -> int:
        return self.r.count(text)

    def record(self) -> dict:
        return self.r.record()


def reference_chat(messages: list[dict], prefill: str | None) -> str:
    """DeepSeek-V4's chat-mode format for a multi-turn exchange (encoding/README.md, 'Basic Chat'):
    <bos><｜User｜>u1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜User｜>u2<｜Assistant｜></think>... ; the
    exchange must alternate user and assistant turns, start with a user turn and end on one."""
    roles = [m["role"] for m in messages]
    if not roles or roles[0] != "user" or roles[-1] != "user" or any(
            r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles)):
        raise ValueError(f"expected alternating user/assistant turns ending on a user turn, got {roles}")
    s = "<｜begin▁of▁sentence｜>"
    for m in messages:
        s += ("<｜User｜>" + m["content"] + "<｜Assistant｜></think>") if m["role"] == "user" \
            else (m["content"] + "<｜end▁of▁sentence｜>")
    return s + (prefill or "")


# ----------------------------------------------------------------------------------------------------------------
# The complete design (the size ladder's; src/ladder.py)
# ----------------------------------------------------------------------------------------------------------------
# The panel's three arms plus the five-shot arms the panel has on each benchmark (five-shot forced on all four;
# five-shot requested on GSM8K and BIG-Bench Hard, 8 configurations), read from the ladder's request files (every
# ladder key's file holds the same rows apart from model_key). Caps as the ladder: 7,000 new tokens on the requested,
# chain-of-thought and five-shot requested arms, each output also read cut to the panel's cap of its arm (the
# primary reading); the forced arms keep 16 (64 on HotpotQA).

LADDER_REQ_DIR = ROOT / "data" / "requests" / "ladder"
LADDER_REQ_KEY = "ministral-3-14b-instruct-2512"
COMPLETE_ARMS = ARMS + ("fewshot_forced", "fewshot_requested")     # the order inside a unit
FORCED_ARMS = ("forced_direct", "fewshot_forced")
ZERO_SHOT_OF = {"fewshot_forced": "forced_direct", "fewshot_requested": "direct_answer"}
ARM_LABEL.update({"fewshot_forced": "fs_forced", "fewshot_requested": "fs_requested"})
_LADDER_CAPS: dict = {}


def ladder_caps(task: str) -> tuple[dict, dict]:
    """(design cap, panel cap) per arm of one benchmark, from the ladder's decoding (src/ladder.task_decoding_dict)."""
    if task not in _LADDER_CAPS:
        from src import ladder as L
        d = L.task_decoding_dict(task)
        mx = {a: int(v) for a, v in d["max_new_tokens"].items()}
        _LADDER_CAPS[task] = (mx, {a: int(d["cut_caps"].get(a, v)) for a, v in mx.items()})
    return _LADDER_CAPS[task]


def design_cap(task: str, arm: str) -> int:
    return ladder_caps(task)[0][arm]


def panel_cap(task: str, arm: str) -> int:
    return ladder_caps(task)[1][arm]


def base_config(task: str, config_id: str) -> str:
    """The five-shot requested arm's configuration ids carry a '<task>/' prefix; the unit is the same
    configuration (scripts/mistral_ladder.base_config)."""
    p = f"{task}/"
    return config_id[len(p):] if config_id.startswith(p) else config_id


def load_complete(tasks=TASKS, req_dir: Path = LADDER_REQ_DIR, key: str = LADDER_REQ_KEY) -> pd.DataFrame:
    """The complete design: request_id, arm, config_id, item_id, messages_json, task, key, unit_config, cap (the
    design's), panel_cap. Checks unique keys and that each forced row's messages equal its requested twin's."""
    frames = []
    for t in tasks:
        df = pd.read_parquet(req_dir / t / f"{key}.parquet",
                             columns=["request_id", "arm", "config_id", "item_id", "messages_json"])
        df["item_id"] = df["item_id"].astype(str)
        df["config_id"] = df["config_id"].astype(str)
        df["task"] = t
        df["key"] = t + "|" + df["request_id"].astype(str)
        df["unit_config"] = [base_config(t, c) for c in df["config_id"]]
        mx, cut = ladder_caps(t)
        bad = sorted(set(df["arm"]) - set(mx))
        if bad:
            raise SystemExit(f"{t}: arms {bad} have no ladder cap")
        df["cap"] = df["arm"].map(mx).astype(int)
        df["panel_cap"] = df["arm"].map(cut).astype(int)
        d = df[df["arm"] == "direct_answer"].set_index("request_id")["messages_json"]
        f = df[df["arm"] == "forced_direct"]
        if int((f["request_id"].map(base_id).map(d) != f["messages_json"]).sum()):
            raise SystemExit(f"{t}: forced rows differ from their requested row")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["key"].duplicated().any():
        raise SystemExit("duplicated request keys in the ladder files")
    return out


def check_complete_against_panel(complete: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """The complete design's three zero-shot arms are the panel design the pilot ran: the same keys, arms,
    configurations, items and messages."""
    z = complete[complete["arm"].isin(ARMS)].set_index("key")
    p = panel.set_index("key")
    same_keys = set(z.index) == set(p.index)
    diff = 0
    if same_keys:
        q = z.loc[p.index]
        diff = int(((q["messages_json"] != p["messages_json"]) | (q["arm"] != p["arm"])
                    | (q["config_id"] != p["config_id"]) | (q["item_id"] != p["item_id"])).sum())
    return {"panel_calls": int(len(p)), "complete_zero_shot_calls": int(len(z)), "same_keys": same_keys,
            "rows_differing": diff, "ok": bool(same_keys and diff == 0)}


def possibly_cut(r: dict) -> bool:
    """A reply that may have been stopped by its cap: finish reason 'length', or no 'stop' with the whole cap used."""
    fr = r.get("finish_reason")
    ct = int(((r.get("usage") or {}).get("completion_tokens")) or 0)
    return fr == "length" or (fr != "stop" and r.get("cap") is not None and ct >= int(r["cap"]))


def answered_ok(r: dict) -> bool:
    return r.get("status") == 200 and isinstance(r.get("text"), str)


def valid_for_design(r: dict, cap_of=None) -> bool:
    """An answered reply stands for the complete design when it ran at the design's cap of its arm, or ended before
    its own cap: greedy decoding with the same prompt writes the same tokens, so a reply that stopped at the panel's
    cap is the reply at 7,000 too. A reply cut at a cap below the design's is rerun."""
    if not answered_ok(r):
        return False
    try:
        dc = (cap_of or design_cap)(r["task"], r["arm"])
    except KeyError:
        return True
    return int(r.get("cap") or 0) >= dc or not possibly_cut(r)


def complete_done_keys(rows: list[dict], cap_of=None) -> set:
    return {r["key"] for r in rows if valid_for_design(r, cap_of)}


def rerun_rows(rows: list[dict], cap_of=None) -> list[dict]:
    """Answered replies that ran below the design's cap and may have been cut there (they are sent again)."""
    good = complete_done_keys(rows, cap_of)
    out, seen = [], set()
    for r in rows:
        if answered_ok(r) and r["key"] not in good and r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Money
# ----------------------------------------------------------------------------------------------------------------

def call_cost(prompt_tokens, completion_tokens, cached_tokens=0, price: dict = PRICE) -> float:
    p, c, k = int(prompt_tokens or 0), int(completion_tokens or 0), int(cached_tokens or 0)
    k = min(max(k, 0), p)
    return ((p - k) * price["input"] + k * price["cached_input"] + c * price["output"]) / 1e6


def upper_cost(n_prompt_tokens: int, cap: int, price: dict = PRICE) -> float:
    """The most one call can cost: every prompt token at the full input price and the whole cap generated. The
    rendered prompt's local count plus a margin of 16 tokens covers a server that counts a token or two more."""
    return ((n_prompt_tokens + 16) * price["input"] + cap * price["output"]) / 1e6


def usage_of(j: dict) -> dict:
    u = (j or {}).get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    cached = det.get("cached_tokens") if isinstance(det, dict) else None
    if cached is None:
        cached = u.get("cached_tokens", u.get("prompt_cache_hit_tokens"))
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": cached}


def live_test_spend(path: Path = LIVE_TEST_RAW, price: dict = PRICE) -> dict:
    """The live prefill test's bill from its saved usage fields (no cached tokens were reported there)."""
    n, usd, pt, ct = 0, 0.0, 0, 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("usage") or {}
            if r.get("status") == 200:
                n += 1
                pt += int(u.get("prompt_tokens") or 0)
                ct += int(u.get("completion_tokens") or 0)
                usd += call_cost(u.get("prompt_tokens"), u.get("completion_tokens"), 0, price)
    return {"calls": n, "prompt_tokens": pt, "completion_tokens": ct, "usd": usd, "path": rel(path)}


class Budget(Exception):
    pass


class Meter:
    """Spent (billed replies) plus reserved (calls in flight at their largest cost) never passes the limit."""

    def __init__(self, limit: float, spent: float):
        self.limit, self.spent, self.reserved = float(limit), float(spent), 0.0
        self.lock = threading.Lock()

    def reserve(self, amount: float) -> None:
        with self.lock:
            if self.spent + self.reserved + amount > self.limit + 1e-12:
                raise Budget(f"spend limit ${self.limit:.2f}: spent ${self.spent:.4f}, in flight "
                             f"${self.reserved:.4f}, next call up to ${amount:.4f}")
            self.reserved += amount

    def settle(self, reserved: float, actual: float) -> None:
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


# ----------------------------------------------------------------------------------------------------------------
# Key and raw file
# ----------------------------------------------------------------------------------------------------------------

def read_key(env_path: Path = ROOT / ".env") -> str:
    """NEBIUS_API_KEY from the environment. The value never leaves this process. `env_path` is not read; it stays
    in the signature for the callers."""
    v = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not v:
        raise SystemExit("NEBIUS_API_KEY is not set in the environment")
    return v


def read_raw(path: Path = RAW) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:      # a line cut by a crash mid-write; its call is redone
                continue
    return out


def done_keys(rows: list[dict]) -> set:
    return {r["key"] for r in rows if r.get("status") == 200 and isinstance(r.get("text"), str)}


def spent_of(rows: list[dict]) -> float:
    return float(sum(r.get("cost_usd") or 0.0 for r in rows))


# ----------------------------------------------------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------------------------------------------------

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, None}


def backoff(attempt: int) -> float:
    return min(60.0, 2.0 * (2 ** attempt)) * (0.75 + 0.5 * ((hash((attempt, time.time())) % 1000) / 1000))


class Runner:
    def __init__(self, client, meter: Meter, raw_path: Path, phase: str, sleep=time.sleep, limiter=None):
        self.client, self.meter, self.raw_path, self.phase, self.sleep = client, meter, raw_path, phase, sleep
        self.limiter = limiter
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_reason = None
        self.stats = {"calls": 0, "replies": 0, "retries": 0, "failed": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0, "usd": 0.0}
        self.consecutive_failures = 0
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(raw_path, "a", encoding="utf-8")

    def close(self):
        self.fh.close()

    def write(self, rec: dict) -> None:
        line = npt.redact(json.dumps(rec, ensure_ascii=False), getattr(self.client, "_key", None))
        with self.lock:
            self.fh.write(line + "\n")
            self.fh.flush()

    def halt(self, reason: str) -> None:
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.stop.set()

    def call(self, req: dict) -> dict | None:
        """One request with retries; appends one raw line (the reply or the last failure)."""
        body = {"model": MODEL, "prompt": req["prompt"], "max_tokens": req["cap"], "temperature": 0.0, "top_p": 1.0}
        up = upper_cost(req["n_prompt_tokens_local"], req["cap"])
        tries, t0, resp = 0, time.time(), None
        while True:
            if self.stop.is_set():
                return None
            if self.limiter is not None:
                self.limiter.acquire(int(req["n_prompt_tokens_local"]))
            try:
                self.meter.reserve(up)
            except Budget as e:
                self.halt(str(e))
                return None
            tries += 1
            with self.lock:
                self.stats["calls"] += 1
            resp = self.client.post("/completions", body)
            ok = resp.get("status") == 200 and isinstance(resp.get("json"), dict)
            u = usage_of(resp.get("json")) if ok else {"prompt_tokens": None, "completion_tokens": None,
                                                         "cached_tokens": None}
            cost = call_cost(u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]) if ok else 0.0
            self.meter.settle(up, cost)
            if self.limiter is not None and ok:
                self.limiter.add(int(u["completion_tokens"] or 0))
            if ok or resp.get("status") not in RETRY_STATUS or tries >= MAX_TRIES:
                break
            with self.lock:
                self.stats["retries"] += 1
            self.sleep(backoff(tries - 1))
        ch = ((resp.get("json") or {}).get("choices") or [{}])[0] if ok else {}
        text = ch.get("text") if ok else None
        rec = {"key": req["key"], "task": req["task"], "request_id": req["request_id"], "arm": req["arm"],
               "config_id": req["config_id"], "item_id": req["item_id"], "phase": self.phase,
               "status": resp.get("status"), "text": text, "finish_reason": ch.get("finish_reason"),
               "stop_reason": ch.get("stop_reason"), "usage": u, "cost_usd": cost,
               "n_prompt_tokens_local": req["n_prompt_tokens_local"], "prompt_sha256": req["prompt_sha256"],
               "cap": req["cap"], "tries": tries, "seconds": round(time.time() - t0, 2),
               "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": None if ok else resp.get("error")}
        for k in ("panel_cap", "design"):          # the complete design's rows carry both
            if k in req:
                rec[k] = req[k]
        self.write(rec)
        with self.lock:
            if ok and isinstance(text, str):
                self.stats["replies"] += 1
                self.stats["prompt_tokens"] += int(u["prompt_tokens"] or 0)
                self.stats["cached_tokens"] += int(u["cached_tokens"] or 0)
                self.stats["completion_tokens"] += int(u["completion_tokens"] or 0)
                self.stats["usd"] += cost
                self.consecutive_failures = 0
            else:
                self.stats["failed"] += 1
                self.consecutive_failures += 1
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.stop_reason = self.stop_reason or (f"{MAX_CONSECUTIVE_FAILURES} failed calls in a row (last "
                                                            f"status {resp.get('status')})")
                    self.stop.set()
        return rec

    def unit(self, reqs: list[dict]) -> None:
        for r in reqs:            # requested, forced, chain of thought: back to back
            if self.stop.is_set():
                return
            self.call(r)


def build_unit_requests(rows: pd.DataFrame, rend: PanelRenderer, done: set) -> list[dict]:
    out = []
    for arm in ARMS:
        x = rows[rows["arm"] == arm]
        if len(x) != 1:
            raise SystemExit(f"unit {rows.iloc[0][['task', 'config_id', 'item_id']].tolist()}: {len(x)} {arm} rows")
        x = x.iloc[0]
        if x["key"] in done:
            continue
        p = rend.prompt(user_text(x["messages_json"]), arm)
        out.append({"key": x["key"], "task": x["task"], "request_id": x["request_id"], "arm": arm,
                    "config_id": x["config_id"], "item_id": x["item_id"], "prompt": p,
                    "prompt_sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(),
                    "n_prompt_tokens_local": rend.count(p), "cap": CAPS[x["task"]][arm]})
    return out


def run_units(runner: Runner, df: pd.DataFrame, units: pd.DataFrame, rend: PanelRenderer, done: set,
              threads: int = THREADS, progress_every: int = 500) -> dict:
    groups = df.groupby(["task", "config_id", "item_id"], sort=False).indices
    t0 = time.time()
    n_sub = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        pending = set()
        for t, c, i in zip(units["task"], units["config_id"], units["item_id"]):
            if runner.stop.is_set():
                break
            reqs = build_unit_requests(df.iloc[groups[(t, c, i)]], rend, done)
            if not reqs:
                continue
            while len(pending) >= 2 * threads:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    f.result()
            pending.add(ex.submit(runner.unit, reqs))
            n_sub += 1
            if n_sub % progress_every == 0:
                s = runner.stats
                print(f"[{time.strftime('%H:%M:%S')}] units {n_sub}/{len(units)}; replies {s['replies']}, "
                      f"failed {s['failed']}, retries {s['retries']}; this run ${s['usd']:.4f}, meter "
                      f"${runner.meter.spent:.4f} ({time.time() - t0:.0f} s)", flush=True)
        for f in pending:
            f.result()
    return {"units_submitted": n_sub, "seconds": round(time.time() - t0, 1), "stop_reason": runner.stop_reason,
            **runner.stats}


# ----------------------------------------------------------------------------------------------------------------
# Forced pass, projection
# ----------------------------------------------------------------------------------------------------------------

def box_closes(text: str) -> bool:
    """The box opened by the prefill closes: brace depth, counted from 1 after "\\boxed{", returns to 0."""
    depth = 1
    for ch in text or "":
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def forced_pass(text) -> dict:
    """Pass: the box closes and the reply holds no restart (a new \\boxed, a think tag, a role or turn token)."""
    t = text if isinstance(text, str) else ""
    reasons = [f"restart or new turn: {m!r}" for m in npt.RESTART_MARKS if m in t]
    if not box_closes(t):
        reasons.append("box not closed")
    return {"pass": not reasons, "reasons": reasons}


def cut_at_first_brace(text):
    """GSM8K and BIG-Bench Hard forced replies: what the collection's stop at the first "}" (kept) returned."""
    if not isinstance(text, str):
        return text, False
    j = text.find("}")
    return (text[: j + 1], True) if j >= 0 else (text, False)


def projection(rows: list[dict], df_counts: dict, live: dict, limit: float = LIMIT_USD) -> dict:
    """Full cost from the pilot's real bill: per benchmark and arm, the mean cost of a call x the design's calls."""
    ok = [r for r in rows if r.get("status") == 200 and isinstance(r.get("text"), str)]
    cells, total = [], live["usd"]
    for t in TASKS:
        for a in ARMS:
            xs = [r for r in ok if r["task"] == t and r["arm"] == a]
            n = df_counts.get((t, a), 0)
            if not xs:
                cells.append({"task": t, "arm": a, "calls_design": n, "pilot_calls": 0, "usd_per_call": None,
                              "projected_usd": None})
                continue
            c = float(np.mean([r["cost_usd"] for r in xs]))
            pt = float(np.mean([r["usage"]["prompt_tokens"] or 0 for r in xs]))
            kt = float(np.mean([r["usage"]["cached_tokens"] or 0 for r in xs]))
            ot = float(np.mean([r["usage"]["completion_tokens"] or 0 for r in xs]))
            cells.append({"task": t, "arm": a, "calls_design": n, "pilot_calls": len(xs), "usd_per_call": c,
                          "prompt_tokens_mean": pt, "cached_tokens_mean": kt, "completion_tokens_mean": ot,
                          "projected_usd": c * n})
            total += c * n
    missing = [f"{c['task']}/{c['arm']}" for c in cells if c["projected_usd"] is None and c["calls_design"]]
    passes = {}
    for t in TASKS:
        xs = [r for r in ok if r["task"] == t and r["arm"] == "forced_direct"]
        k = sum(forced_pass(r["text"])["pass"] for r in xs)
        passes[t] = {"n": len(xs), "pass": int(k), "rate": (k / len(xs)) if xs else None}
    n_all = sum(v["n"] for v in passes.values())
    pass_all = (sum(v["pass"] for v in passes.values()) / n_all) if n_all else None
    fits = (not missing) and total <= limit
    pass_ok = all(v["rate"] is not None and v["rate"] >= PASS_MIN for v in passes.values())
    return {"limit_usd": limit, "live_test_usd": live["usd"], "pilot_usd": float(sum(r["cost_usd"] for r in ok)),
            "projected_total_usd": total if not missing else None, "cells_missing": missing, "cells": cells,
            "forced_pass": passes, "forced_pass_all": pass_all, "pass_min": PASS_MIN,
            "fits": fits, "pass_ok": pass_ok, "go": bool(fits and pass_ok)}


# ----------------------------------------------------------------------------------------------------------------
# Real time, complete design (batch is refused for the account)
# ----------------------------------------------------------------------------------------------------------------
# The rows nebius_batch.py build writes (load_complete minus the keys answered at the design's cap, which includes the
# reruns of pilot replies cut at the panel's cap), sent to /v1/completions as the pilot sent its calls: same renderer
# (PanelRenderer.prompt_messages; for a one-message row it gives the pilot's prompt), forced prefill, greedy. Order: the
# zero-shot arms as the pilot's units (sha256 order, arms back to back); the five-shot arms in blocks of FS_BLOCK items
# of one benchmark, arm and configuration sent back to back by one thread, so the five worked examples they share can
# be billed at the cached rate after the first call. Units and blocks are interleaved by their position in their own
# order, so a run stopped early still covers every benchmark, arm and configuration in proportion.

DESIGN_COMPLETE = "complete/1"           # the tag nebius_batch.DESIGN_TAG gives the same design
COMPLETE_LIMIT_USD = 120.0               # hard limit for the Nebius total, live test and pilot in
RPM = 600                                # calls started per rolling minute (the model allows 1,000)
TPM = 1_200_000                          # prompt + generated tokens per rolling minute (the model allows 2 M)
FS_BLOCK = 40
TEST_LIMIT_USD = 0.05
TEST_PLAN = ((("hotpot", "fewshot_forced"), 5), (("bbh", "fewshot_forced"), 4), (("gsm8k", "fewshot_forced"), 3),
             (("math", "fewshot_forced"), 3), (("gsm8k", "fewshot_requested"), 2), (("bbh", "fewshot_requested"), 2))
REALTIME_PROJECTION = RES / "nebius_realtime_projection.json"
REALTIME_TEST = RES / "nebius_realtime_test.json"
PROMPT_COUNTS = DATA / "nebius_complete_prompt_counts.parquet"
FIVE_SHOT_ARMS = ("fewshot_forced", "fewshot_requested")
REQ_COLS = ["key", "task", "request_id", "arm", "config_id", "item_id", "messages_json", "cap", "panel_cap"]


class RateLimiter:
    """At most `rpm` calls started and `tpm` tokens (prompt counted at the start, generated tokens when the reply comes
    back) in any rolling 60 s. Thread-safe; clock and sleep can be replaced in tests."""

    def __init__(self, rpm: int = RPM, tpm: int = TPM, clock=time.monotonic, sleep=time.sleep):
        import collections
        self.rpm, self.tpm, self.clock, self.sleep = int(rpm), int(tpm), clock, sleep
        self.lock = threading.Lock()
        self.starts = collections.deque()
        self.toks = collections.deque()
        self.tok_sum = 0
        self.waited = 0.0

    def _trim(self, now: float) -> None:
        while self.starts and now - self.starts[0] >= 60.0:
            self.starts.popleft()
        while self.toks and now - self.toks[0][0] >= 60.0:
            self.tok_sum -= self.toks.popleft()[1]

    def acquire(self, prompt_tokens: int) -> None:
        while True:
            with self.lock:
                now = self.clock()
                self._trim(now)
                rpm_ok = len(self.starts) < self.rpm
                tpm_ok = self.tok_sum + prompt_tokens <= self.tpm or not self.toks
                if rpm_ok and tpm_ok:
                    self.starts.append(now)
                    self.toks.append((now, int(prompt_tokens)))
                    self.tok_sum += int(prompt_tokens)
                    return
                waits = []
                if not rpm_ok:
                    waits.append(60.0 - (now - self.starts[0]))
                if not tpm_ok:
                    waits.append(60.0 - (now - self.toks[0][0]))
                w = min(1.0, max(0.01, max(waits)))
                self.waited += w
            self.sleep(w)

    def add(self, tokens: int) -> None:
        with self.lock:
            self.toks.append((self.clock(), int(tokens)))
            self.tok_sum += int(tokens)


def complete_request(x: dict, rend) -> dict:
    """One design row as the runner's request: the prompt rendered by the pilot's renderer, the design's cap."""
    p = rend.prompt_messages(json.loads(x["messages_json"]), x["arm"])
    return {"key": x["key"], "task": x["task"], "request_id": x["request_id"], "arm": x["arm"],
            "config_id": x["config_id"], "item_id": x["item_id"], "prompt": p,
            "prompt_sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(), "n_prompt_tokens_local": rend.count(p),
            "cap": int(x["cap"]), "panel_cap": int(x["panel_cap"]), "design": DESIGN_COMPLETE}


def _frac(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16 ** 12


def shot_family(task: str, item_id: str) -> str:
    """The rows of a configuration that share their five worked examples: BIG-Bench Hard draws them per subtask (25
    sets per configuration, the item id's first part); GSM8K, MATH-500 and HotpotQA use one set per configuration."""
    return str(item_id).split("/", 1)[0] if task == "bbh" else ""


def complete_work(df: pd.DataFrame, done: set, fs_block: int = FS_BLOCK) -> list[np.ndarray]:
    """The rows still to send as work items (arrays of positions in df), in dispatch order. A zero-shot unit is its
    remaining arms in the pilot's order; a five-shot block is up to fs_block items of one (benchmark, arm,
    configuration, shot family) in sha256 item order. Each item's place: (its rank + an offset) / its stream's length, so units and
    the blocks of every configuration interleave evenly."""
    pos = np.flatnonzero(~df["key"].isin(done).to_numpy())
    if not len(pos):
        return []
    rem = df.iloc[pos].copy()
    rem["_pos"] = pos
    rem["_h"] = [unit_hash(t, c, i) for t, c, i in zip(rem["task"], rem["unit_config"], rem["item_id"])]
    rem["_arm_i"] = rem["arm"].map({a: i for i, a in enumerate(COMPLETE_ARMS)})
    keyed: list = []
    z = rem[rem["arm"].isin(ARMS)].sort_values(["_h", "_arm_i"])
    if len(z):
        groups = list(z.groupby("_h", sort=True))
        for r, (h, g) in enumerate(groups):
            keyed.append(((r + 0.5) / len(groups), h, g["_pos"].to_numpy()))
    f = rem[~rem["arm"].isin(ARMS)].copy()
    f["_fam"] = [shot_family(t, i) for t, i in zip(f["task"], f["item_id"])]
    for (t, a, c, fam), g in f.groupby(["task", "arm", "config_id", "_fam"], sort=True):
        g = g.sort_values("_h")
        nb = math.ceil(len(g) / fs_block)
        off = _frac(f"{SALT}|{t}|{a}|{c}|{fam}")
        p = g["_pos"].to_numpy()
        for b in range(nb):
            keyed.append(((b + off) / nb, f"{t}|{a}|{c}|{fam}|{b:05d}", p[b * fs_block:(b + 1) * fs_block]))
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [k[2] for k in keyed]


def run_work(runner: Runner, df: pd.DataFrame, work: list, rend, threads: int = THREADS,
             progress_every: int = 500) -> dict:
    """Sends the work items, each in one thread (its calls back to back); prompts are rendered as items are handed
    out."""
    t0 = time.time()
    n_sub = n_calls = 0
    recs = df[REQ_COLS].to_dict("records")
    total = int(sum(len(w) for w in work))
    with ThreadPoolExecutor(max_workers=threads) as ex:
        pending = set()
        for w in work:
            if runner.stop.is_set():
                break
            reqs = [complete_request(recs[int(i)], rend) for i in w]
            while len(pending) >= 2 * threads:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for x in fin:
                    x.result()
            pending.add(ex.submit(runner.unit, reqs))
            n_sub += 1
            n_calls += len(reqs)
            if n_sub % progress_every == 0:
                st = runner.stats
                el = time.time() - t0
                rate = 60 * st["replies"] / max(el, 1.0)
                print(f"[{time.strftime('%H:%M:%S')}] handed out {n_calls:,}/{total:,} calls; replies "
                      f"{st['replies']:,}, failed {st['failed']}, retries {st['retries']}; this run "
                      f"${st['usd']:.4f}, Nebius total ${runner.meter.spent:.4f} of ${runner.meter.limit:.2f}; "
                      f"{el:.0f} s, {rate:.0f} replies/min", flush=True)
        for x in pending:
            x.result()
    return {"work_items_submitted": n_sub, "calls_handed_out": n_calls, "seconds": round(time.time() - t0, 1),
            "stop_reason": runner.stop_reason, **runner.stats}


def pilot_means(rows: list[dict], cap_of=None) -> dict:
    """Per (task, arm) over the real-time replies (no batch rows): n, mean billed completion tokens, the same with every
    reply that may have been cut below the design's cap counted at the design's cap (high), and the share of prompt
    tokens billed at the cached rate."""
    cap_of = cap_of or design_cap
    acc: dict = {}
    for r in rows:
        if not answered_ok(r) or r.get("phase") == "batch":
            continue
        u = r.get("usage") or {}
        ct = int(u.get("completion_tokens") or 0)
        try:
            dc = cap_of(r["task"], r["arm"])
        except KeyError:
            dc = int(r.get("cap") or ct)
        cut = possibly_cut(r) and dc > int(r.get("cap") or 0)
        a = acc.setdefault((r["task"], r["arm"]), [0, 0, 0, 0, 0])
        a[0] += 1
        a[1] += ct
        a[2] += dc if cut else ct
        a[3] += int(u.get("prompt_tokens") or 0)
        a[4] += int(u.get("cached_tokens") or 0)
    return {k: {"n": v[0], "completion_tokens": v[1] / v[0], "completion_tokens_high": v[2] / v[0],
                "cached_share": (v[4] / v[3]) if v[3] else 0.0} for k, v in acc.items()}


def prompt_counts(df: pd.DataFrame, rend, path: Path = PROMPT_COUNTS, progress_every: int = 50_000) -> pd.Series:
    """key -> local prompt tokens of the rendered prompt, kept in a parquet (keyed also by the messages' sha256, so a
    changed row is counted again)."""
    msha = [hashlib.sha256(m.encode("utf-8")).hexdigest()[:16] for m in df["messages_json"]]
    have = {}
    if path.exists():
        c = pd.read_parquet(path)
        have = {(k, s): int(n) for k, s, n in zip(c["key"], c["msha"], c["n"])}
    out, t0 = [], time.time()
    for i, (k, s, m, a) in enumerate(zip(df["key"], msha, df["messages_json"], df["arm"])):
        n = have.get((k, s))
        if n is None:
            n = rend.count(rend.prompt_messages(json.loads(m), a))
        out.append(n)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[{time.strftime('%H:%M:%S')}] counted {i + 1:,} of {len(df):,} ({time.time() - t0:.0f} s)",
                  flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"key": df["key"].to_numpy(), "msha": msha, "n": out}).to_parquet(path, index=False)
    return pd.Series(out, index=df.index)


def complete_projection(rem: pd.DataFrame, n_prompt: pd.Series, means: dict, spent: float, limit: float,
                        fs_cached: dict | None = None, reruns: dict | None = None, price: dict = PRICE) -> dict:
    """Real-time cost of the rows left (rem), per benchmark and arm. Prompt: the local count; the share billed at the
    cached rate is the pilot's for its benchmark and arm (five-shot arms: fs_cached[(task, arm)], measured in the
    20-request test, else 0). Output: the pilot's mean billed completion tokens of the arm (five-shot arms: of the
    matching zero-shot arm), at most the cap; a rerun expects its own pilot completion tokens. High: possibly cut pilot
    replies and reruns at the design's cap. No cache: every prompt token at the full price. `go`: spent + projection
    <= limit."""
    fs_cached, reruns = fs_cached or {}, reruns or {}
    cells: dict = {}
    for x, n in zip(rem.itertuples(index=False), n_prompt.to_numpy()):
        m = means.get((x.task, x.arm)) or means.get((x.task, ZERO_SHOT_OF.get(x.arm, "")))
        if m is None:
            raise SystemExit(f"no pilot replies to project from for {x.task}/{x.arm}")
        share = m["cached_share"] if x.arm in ARMS else float(fs_cached.get((x.task, x.arm), 0.0))
        rr = reruns.get(x.key)
        lo = float(int((rr.get("usage") or {}).get("completion_tokens") or 0)) if rr else \
            min(float(x.cap), m["completion_tokens"])
        hi = float(x.cap) if rr else min(float(x.cap), m["completion_tokens_high"])
        pin = n * ((1 - share) * price["input"] + share * price["cached_input"]) / 1e6
        c = cells.setdefault((x.task, x.arm), {"calls": 0, "reruns": 0, "prompt_tokens": 0, "output_tokens": 0.0,
                                               "usd": 0.0, "usd_high": 0.0, "usd_nocache": 0.0, "usd_upper": 0.0,
                                               "cached_share": share, "cap": int(x.cap)})
        c["calls"] += 1
        c["reruns"] += rr is not None
        c["prompt_tokens"] += int(n)
        c["output_tokens"] += lo
        c["usd"] += pin + lo * price["output"] / 1e6
        c["usd_high"] += pin + hi * price["output"] / 1e6
        c["usd_nocache"] += (n * price["input"] + lo * price["output"]) / 1e6
        c["usd_upper"] += upper_cost(int(n), int(x.cap), price)
    order = {a: i for i, a in enumerate(COMPLETE_ARMS)}
    cl = [{"task": t, "arm": a, **c, "prompt_tokens_mean": c["prompt_tokens"] / c["calls"],
           "output_tokens_mean": c["output_tokens"] / c["calls"]}
          for (t, a), c in sorted(cells.items(), key=lambda kv: (TASKS.index(kv[0][0]), order[kv[0][1]]))]
    tot = {k: float(sum(c[k] for c in cl)) for k in ("usd", "usd_high", "usd_nocache", "usd_upper")}
    return {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": MODEL, "design": DESIGN_COMPLETE,
            "price": price, "to_send": int(len(rem)), "reruns": int(sum(c["reruns"] for c in cl)), "cells": cl,
            "projected_usd": tot["usd"], "projected_usd_high": tot["usd_high"],
            "projected_usd_nocache": tot["usd_nocache"], "projected_usd_upper": tot["usd_upper"],
            "spent_usd": spent, "limit_usd": limit, "total_usd": spent + tot["usd"],
            "total_high_usd": spent + tot["usd_high"], "total_nocache_usd": spent + tot["usd_nocache"],
            "go": spent + tot["usd"] <= limit + 1e-9,
            "five_shot_cached_share": {f"{t}/{a}": v for (t, a), v in fs_cached.items()},
            "rule": "prompt: local count x ((1 - cached share) x $0.30/M + cached share x $0.006/M); output: the pilot's "
                    "mean billed completion tokens of the arm (five-shot: the matching zero-shot arm's), capped, x "
                    "$1.20/M; a rerun: its pilot completion tokens. Cached share: the pilot's per benchmark and arm; "
                    "five-shot arms: the share measured on the 20-request test's calls after the first of each block "
                    "(0 before the test). High: possibly cut pilot replies and reruns at the design's cap."}


def test_selection(df: pd.DataFrame, done: set, plan=TEST_PLAN) -> list[np.ndarray]:
    """The 20-request test's five-shot part: for each (benchmark, five-shot arm) of the plan, the first k remaining
    items in sha256 order of its first configuration (sorted id), as one block (back to back, so caching shows).
    Positions in df."""
    out = []
    rem = df[~df["key"].isin(done)]
    for (t, a), k in plan:
        g = rem[(rem["task"] == t) & (rem["arm"] == a)]
        if g.empty:
            continue
        c = sorted(g["config_id"].unique())[0]
        g = g[g["config_id"] == c]
        h = [unit_hash(t, u, i) for u, i in zip(g["unit_config"], g["item_id"])]
        idx = g.assign(_h=h).sort_values("_h").index.to_numpy()[:k]
        out.append(np.array([df.index.get_loc(i) for i in idx]))
    return out


def check_test(recs: list[dict], pilot: dict) -> dict:
    """The test's checks, per reply: answered; the server's prompt count equals the local count; the prompt ends as
    its arm requires (the opened box inside the assistant turn for the forced arms); a five-shot prompt holds six user
    turns; a forced reply passes (the box closes, no restart). Reported beside them: whether a rerun's reply begins
    with the pilot's cut reply (a test of the server's determinism at temperature 0, not of the request), and cached
    tokens on the five-shot calls after the first of their block."""
    out, prev = [], None
    for r in recs:
        req, rep = r["req"], r["rec"] or {}
        p = req["prompt"]
        forced = req["arm"] in FORCED_ARMS
        u = rep.get("usage") or {}
        d = {"key": req["key"], "arm": req["arm"], "status": rep.get("status"),
             "n_prompt_local": req["n_prompt_tokens_local"], "n_prompt_server": u.get("prompt_tokens"),
             "cached": u.get("cached_tokens"), "completion_tokens": u.get("completion_tokens"),
             "finish_reason": rep.get("finish_reason"), "cost_usd": rep.get("cost_usd"),
             "text": (rep.get("text") or "")[:300], "user_turns": p.count("<｜User｜>"),
             "ends_right": p.endswith("<｜Assistant｜></think>" + (PREFILL if forced else "")),
             "first_of_block": (req["task"], req["arm"], req["config_id"]) != prev}
        prev = (req["task"], req["arm"], req["config_id"])
        checks = [answered_ok(rep), d["n_prompt_server"] == d["n_prompt_local"], d["ends_right"]]
        if req["arm"] in FIVE_SHOT_ARMS:
            checks.append(d["user_turns"] == 6)
        if forced:
            fp = forced_pass(rep.get("text"))
            d["forced_pass"], d["forced_reasons"] = fp["pass"], fp["reasons"]
            checks.append(fp["pass"])
        if req["key"] in pilot:     # reported, not a pass rule: it tests the server's determinism, and the rerun's
            old_t, new_t = pilot[req["key"]].get("text") or "", rep.get("text") or ""      # reply is the one kept
            d["rerun_starts_with_pilot_reply"] = new_t.startswith(old_t)
            d["rerun_common_prefix_chars"] = next((j for j in range(min(len(old_t), len(new_t)))
                                                   if old_t[j] != new_t[j]), min(len(old_t), len(new_t)))
        d["ok"] = bool(all(checks))
        out.append(d)
    fs_c: dict = {}
    for d in out:
        if d["arm"] in FIVE_SHOT_ARMS and not d["first_of_block"] and d["n_prompt_server"]:
            a = fs_c.setdefault((d["key"].split("|", 1)[0], d["arm"]), [0, 0])
            a[0] += int(d["cached"] or 0)
            a[1] += int(d["n_prompt_server"])
    return {"replies": out, "passed": bool(out) and all(d["ok"] for d in out),
            "fs_cached_share": {k: v[0] / v[1] for k, v in fs_c.items() if v[1]},
            "usd": float(sum(d["cost_usd"] or 0 for d in out))}


def cmd_complete(a, rend=None, client=None) -> int:
    """run --design complete: --project (no network), --test N, or the run itself."""
    if a.limit_usd > COMPLETE_LIMIT_USD + 1e-9:
        print(f"--limit-usd {a.limit_usd} is above the ${COMPLETE_LIMIT_USD:.0f} limit for the Nebius total; refused")
        return 2
    df = load_complete()
    chk = check_complete_against_panel(df, load_all())
    if not chk["ok"]:
        raise SystemExit(f"the complete design's zero-shot arms are not the pilot's panel design: {chk}")
    rows = read_raw(RAW)
    done = complete_done_keys(rows)
    keys = set(df["key"])
    reruns = {r["key"]: r for r in rerun_rows(rows) if r["key"] in keys}
    live = live_test_spend(LIVE_TEST_RAW)
    spent = live["usd"] + spent_of(rows)
    rend = rend or PanelRenderer()
    if a.project:
        rem = df[~df["key"].isin(done)]
        n = prompt_counts(rem, rend)
        tj = json.loads(REALTIME_TEST.read_text(encoding="utf-8")) if REALTIME_TEST.exists() else {}
        fs = {tuple(k.split("/")): v for k, v in (tj.get("fs_cached_share_by_arm") or {}).items()}
        pj = complete_projection(rem, n, pilot_means(rows), spent, a.limit_usd, fs, reruns)
        pj.update({"design_calls": int(len(df)), "answered_at_design_cap": int(len(done & keys)),
                   "check_against_panel": chk, "live_test_usd": live["usd"], "panel_usd": spent_of(rows),
                   "test": {"ran": bool(tj.get("ran")), "passed": tj.get("passed")}})
        RES.mkdir(parents=True, exist_ok=True)
        REALTIME_PROJECTION.write_text(json.dumps(pj, indent=1), encoding="utf-8")
        for c in pj["cells"]:
            print(f"  {c['task']:7s} {c['arm']:18s} {c['calls']:7,} calls  prompt {c['prompt_tokens_mean']:6.0f}  "
                  f"out {c['output_tokens_mean']:6.0f}  cached {c['cached_share']:.2f}  ${c['usd']:7.2f}  "
                  f"(no cache ${c['usd_nocache']:.2f}, high ${c['usd_high']:.2f})")
        print(f"to send {pj['to_send']:,} ({pj['reruns']} reruns): ${pj['projected_usd']:.2f} (high "
              f"${pj['projected_usd_high']:.2f}, no cache ${pj['projected_usd_nocache']:.2f}); with ${spent:.2f} spent: "
              f"${pj['total_usd']:.2f} of ${a.limit_usd:.2f}; go {pj['go']}")
        return 0
    if a.test:
        if REALTIME_TEST.exists() and json.loads(REALTIME_TEST.read_text(encoding="utf-8")).get("ran"):
            print(f"the test already ran ({rel(REALTIME_TEST)}); not sent again")
            return 6
        sel = test_selection(df, done)
        kpos = {k: i for i, k in enumerate(df["key"])}
        sel += [np.array([kpos[k]]) for k in sorted(reruns)[:1]]
        calls = [int(i) for s in sel for i in s][: a.test]
        limit = min(a.limit_usd, spent + TEST_LIMIT_USD)
        client = client or npt.Client(a.base_url, read_key(), max_calls=4 * a.test, timeout=a.timeout)
        meter = Meter(limit, spent)
        runner = Runner(client, meter, RAW, "complete-test", limiter=RateLimiter())
        recs = []
        try:
            for i in calls:
                q = complete_request(df.iloc[i][REQ_COLS].to_dict(), rend)
                recs.append({"req": q, "rec": runner.call(q)})
                if runner.stop.is_set():
                    break
        finally:
            runner.close()
        res = check_test(recs, reruns)
        res["fs_cached_share_by_arm"] = {f"{t}/{ar}": v for (t, ar), v in res.pop("fs_cached_share").items()}
        res.update({"ran": True, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "calls": len(recs),
                    "limit_usd": limit, "stop_reason": runner.stop_reason, "stats": runner.stats})
        RES.mkdir(parents=True, exist_ok=True)
        REALTIME_TEST.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        for d in res["replies"]:
            print(f"  {'ok ' if d['ok'] else 'BAD'} {d['arm']:18s} {d['key'][:58]:58s} prompt {d['n_prompt_local']}/"
                  f"{d['n_prompt_server']} cached {d['cached']} out {d['completion_tokens']} {d['finish_reason']} "
                  f"{d['text'][:40]!r}")
        print(f"test: {len(recs)} calls, ${res['usd']:.4f}; passed {res['passed']}; five-shot cached share "
              f"{res['fs_cached_share_by_arm']}")
        return 0 if res["passed"] else 3
    # the run
    if not REALTIME_PROJECTION.exists() or not REALTIME_TEST.exists():
        print("run --design complete --project and --test 20 first")
        return 2
    pj = json.loads(REALTIME_PROJECTION.read_text(encoding="utf-8"))
    tj = json.loads(REALTIME_TEST.read_text(encoding="utf-8"))
    if not tj.get("passed") and not a.approved:
        print(f"the 20-request test did not pass ({rel(REALTIME_TEST)}); not started")
        return 2
    if not pj.get("test", {}).get("ran"):
        print("the projection predates the test; run --project again")
        return 2
    answered_since = len(done & keys) - int(pj["answered_at_design_cap"])
    rem_usd = float(pj["projected_usd"]) * max(0.0, 1.0 - answered_since / max(int(pj["to_send"]), 1))
    if spent + rem_usd > a.limit_usd + 1e-9:
        print(f"projection: ${rem_usd:.2f} left + ${spent:.2f} spent passes ${a.limit_usd:.2f}; not started")
        return 2
    work = complete_work(df, done)
    if not work:
        print("every design key is answered at the design's cap; nothing to send")
        return 0
    client = client or npt.Client(a.base_url, read_key(), max_calls=10 ** 9, timeout=a.timeout)
    meter = Meter(a.limit_usd, spent)
    runner = Runner(client, meter, RAW, "complete", limiter=RateLimiter(a.rpm, a.tpm))
    print(f"complete: {int(sum(len(w) for w in work)):,} calls in {len(work):,} work items ({len(reruns)} reruns "
          f"among them); {len(done & keys):,} design keys answered; Nebius total starts at ${spent:.4f}, hard limit "
          f"${a.limit_usd:.2f}; projected ${rem_usd:.2f} left; {a.threads} threads, at most {a.rpm} calls and "
          f"{a.tpm:,} tokens per minute", flush=True)
    try:
        res = run_work(runner, df, work, rend, threads=a.threads)
    except KeyboardInterrupt:
        runner.halt("interrupted")
        res = {"stop_reason": "interrupted", **runner.stats}
    finally:
        runner.close()
    left = len(keys - complete_done_keys(read_raw(RAW)))
    log = {"phase": "complete", "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "meter_usd": meter.spent,
           "keys_left": left, "rate_wait_s": round(runner.limiter.waited, 1), **res}
    with open(DATA / "nebius_panel_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log) + "\n")
    print(json.dumps(log, indent=1))
    print(f"design keys left: {left}" + ("; done" if not left else "; run the same command again to resume"),
          flush=True)
    return 0 if not left else 1


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------

def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design_counts(df: pd.DataFrame) -> dict:
    return {(t, a): int(n) for (t, a), n in df.groupby(["task", "arm"]).size().items()}


def cmd_plan(a) -> int:
    df = load_all()
    u = units_of(df)
    pu = pilot_units(u)
    counts = design_counts(df)
    print(f"design: {len(df)} calls, {len(u)} units")
    for (t, ar), n in sorted(counts.items()):
        print(f"  {t:7s} {ar:14s} {n}")
    print(f"pilot: {len(pu)} units, {3 * len(pu)} calls ({3 * len(pu) / len(df):.2%})")
    print(pu.groupby("task").size().to_dict())
    print(f"live test: {live_test_spend()}")
    return 0


def cmd_run(a) -> int:
    df = load_all()
    u = units_of(df)
    pu = pilot_units(u)
    rows = read_raw()
    done = done_keys(rows)
    live = live_test_spend()
    spent = live["usd"] + spent_of(rows)
    if a.phase == "full":
        if not PROJECTION.exists():
            print("no projection yet: run --phase pilot first")
            return 2
        pj = json.loads(PROJECTION.read_text(encoding="utf-8"))
        if not pj.get("go") and not a.approved:
            print(f"the projection does not allow the full run (projected ${pj.get('projected_total_usd')}, fits "
                  f"{pj.get('fits')}, forced pass ok {pj.get('pass_ok')}); --approved is needed")
            return 2
        units = dispatch_order(pd.concat([dispatch_order(pu), dispatch_order(u)]).drop_duplicates(
            ["task", "config_id", "item_id"]))
    else:
        units = dispatch_order(pu)
    if spent >= a.limit_usd:
        print(f"already at the limit: ${spent:.4f} of ${a.limit_usd:.2f}")
        return 2
    rend = PanelRenderer()
    key = read_key()
    client = npt.Client(a.base_url, key, max_calls=10 ** 9, timeout=a.timeout)
    meter = Meter(a.limit_usd, spent)
    runner = Runner(client, meter, RAW, a.phase)
    print(f"{a.phase}: {len(units)} units; {len(done)} replies already saved; meter starts at ${spent:.4f} "
          f"(live test ${live['usd']:.4f}); limit ${a.limit_usd:.2f}", flush=True)
    try:
        res = run_units(runner, df, units, rend, done, threads=a.threads)
    except KeyboardInterrupt:
        runner.halt("interrupted")
        res = {"stop_reason": "interrupted", **runner.stats}
    finally:
        runner.close()
    log = {"phase": a.phase, "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "meter_usd": meter.spent, **res}
    with open(DATA / "nebius_panel_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log) + "\n")
    print(json.dumps(log, indent=1))
    rows = read_raw()
    pilot_keys = set(df.merge(pu[["task", "config_id", "item_id"]], on=["task", "config_id", "item_id"])["key"])
    prow = [r for r in rows if r["key"] in pilot_keys]
    pj = projection(prow, design_counts(df), live, a.limit_usd)
    pj["pilot"] = {"units": int(len(pu)), "calls": int(3 * len(pu)), "share_of_calls": 3 * len(pu) / len(df),
                   "replies": len(done_keys(prow)), "complete": len(done_keys(prow)) == 3 * len(pu),
                   "rule": f"per benchmark and configuration, the first ceil({PILOT_SHARE:.0%} x items) items in "
                           f"sha256('{SALT}|task|config|item') order; the three arms of each"}
    pj["spent_total_usd"] = live["usd"] + spent_of(rows)
    pj["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not pj["pilot"]["complete"]:
        pj["go"] = False
        pj["note"] = "the pilot is not complete; no full run"
    RES.mkdir(parents=True, exist_ok=True)
    PROJECTION.write_text(json.dumps(pj, indent=1), encoding="utf-8")
    print(f"projection: total ${pj['projected_total_usd']}, forced pass {pj['forced_pass_all']}, go {pj['go']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=["pilot", "full"])
    r.add_argument("--limit-usd", type=float, default=LIMIT_USD)
    r.add_argument("--threads", type=int, default=THREADS)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--base-url", default=BASE_URL)
    r.add_argument("--approved", action="store_true", help="run past a projection above the limit")
    r.add_argument("--design", choices=["panel", "complete"], default="panel",
                   help="complete: the ladder's design in real time (limit $120 for the Nebius total)")
    r.add_argument("--project", action="store_true", help="complete design: project the cost only (no network)")
    r.add_argument("--test", type=int, default=0, help="complete design: send the N-request test only")
    r.add_argument("--rpm", type=int, default=RPM)
    r.add_argument("--tpm", type=int, default=TPM)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "run":
        if a.design == "complete":
            if a.limit_usd == LIMIT_USD:
                a.limit_usd = COMPLETE_LIMIT_USD
            return cmd_complete(a)
        if a.phase is None:
            ap.error("--phase is required for the panel design")
        return cmd_run(a)
    import nebius_panel_report as rep
    return rep.main()


if __name__ == "__main__":
    raise SystemExit(main())
