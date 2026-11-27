r"""Size ladder: Mistral Large 3 (2512) on Mistral's own API, by batch only (no fallback to real time; pilot with
batching, hard limit $15, then project the full cost and stop; the full run is a separate subcommand).

Model: `mistral-large-2512` (docs.mistral.ai/models/mistral-large-3-25-12), the API id of Mistral Large 3; the open
weights are mistralai/Mistral-Large-3-675B-Instruct-2512 (revision pinned below), whose chat template and tokenizer
are used locally only to count tokens and to read the default system message.

Design: the ladder's complete design (src/ladder.py), read from the ladder's request
files (data/requests/ladder/<task>/ministral-3-14b-instruct-2512.parquet; every key's file holds the same rows apart from
model_key). Caps as the ladder: 7,000 new tokens on the requested, chain-of-thought and five-shot requested arms, kept
also cut at the panel's cap (1,024; 3,072; 1,900 on HotpotQA chain of thought); forced arms 16 (64 on HotpotQA).
Greedy (temperature 0, top_p 1, random_seed 0).

Prompt: the panel's Mistral models and the ladder's Ministral 3 run with their template's default system message;
Mistral's API applies its own template and sends no system message unless one is given, so each request carries
Large 3's default system message (the text of chat_template.jinja's default_system_message, equal to SYSTEM_PROMPT.txt,
with "{today}" and "{yesterday}" left literal as in the template) followed by the panel's messages. The forced test
sends one request without it to check that the server adds nothing.

Forced arms: the API's prefix mode, a final assistant message {"role": "assistant", "content": "\boxed{",
"prefix": true}; the reply continues that message. GSM8K and BIG-Bench Hard forced replies are cut after their first
"}" before scoring (the collection's stop string), as for the Nebius run. A reply that echoes the prefix
has it removed before the prefill is put back in front for scoring.

Pilot: per benchmark and configuration, the first ceil(5% x items) items in sha256('nebius-panel-pilot/1|task|config|
item') order (the Nebius pilot's rule and seed, scripts/nebius_panel.pilot_units), every arm of each such unit.

Subcommands (repository root):
    python scripts/mistral_ladder.py price          # model check (GET only) and prices
    python scripts/mistral_ladder.py forced-test    # 10 forced requests by batch
    python scripts/mistral_ladder.py forced-collect # read the forced test
    python scripts/mistral_ladder.py pilot-submit   # projection, then the pilot batches (<= $15)
    python scripts/mistral_ladder.py collect        # poll once; merge outputs; spend check
    python scripts/mistral_ladder.py report         # score cut and uncut; full projection

The key is read from the environment variable MISTRAL_API_KEY and is never printed, logged or written;
error text is redacted. Nothing here calls a real-time generation endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import ladder as L  # noqa: E402

MODEL = "mistral-large-2512"
MODEL_KEY = "mistral-large-2512"
BASE = "https://api.mistral.ai/v1"
ENDPOINT = "/v1/chat/completions"
HF_ID = "mistralai/Mistral-Large-3-675B-Instruct-2512"
HF_REV = "383ffea2c7d60dfd44ca960e8e691709d4fdb9cd"
REQ_KEY = "ministral-3-14b-instruct-2512"
PREFILL = "\\boxed{"
FORCED = ("forced_direct", "fewshot_forced")
CUT_AT_FIRST_BRACE = {"gsm8k", "bbh"}
SALT = "nebius-panel-pilot/1"
PILOT_SHARE = 0.05
LIMIT_USD = 15.0
PASS_MIN = 0.9
SEED = 0
PRICE = {"input": 0.5, "output": 1.5, "batch_factor": 0.5, "date": "2026-11-22",
         "source": "https://mistral.ai/pricing (read 2026-11-22: \"Mistral Large costs $0.5 /M tokens in and $1.5 /M "
                   "tokens out\"; \"Batch processing, for high-volume work, reduces the price by 50%\"); "
                   "https://docs.mistral.ai/capabilities/batch/ (\"at a 50% discount\")",
         "note": "the pricing page names 'Mistral Large' without a version; Mistral Large 3 (mistral-large-2512) was not in "
                 "this key's model list when checked (deprecated 31 Oct 2026, retirement 30 Apr 2027)"}
MODEL_DOC = {"api_id": MODEL, "page": "https://docs.mistral.ai/models/mistral-large-3-25-12",
             "deprecation": "2026-10-31", "retirement": "2027-04-30 (https://docs.mistral.ai/getting-started/models/)",
             "replacement": "Mistral Large 4", "context": "256k", "released": "2025-12-02"}
MAX_FILE_BYTES = 50 * 1024 * 1024
POLL_SEC = 300
TERMINAL = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}

DATA = ROOT / "data" / "ladder_api" / MODEL_KEY
RAW = DATA / "raw.jsonl"
STATE = DATA / "state.json"
STATE_FULL = DATA / "state_full.json"
FULL_META = DATA / "full_meta.parquet"
# Cumulative spend limit of the full run, pilot and forced test included.
FULL_LIMIT_USD = 250.0
FORCED_STATE = DATA / "forced_state.json"
RES = ROOT / "results" / "ladder"
PRICE_JSON = RES / "mistral_price.json"
FORCED_JSON = RES / "mistral_forced_test.json"
FORCED_MD = RES / "mistral_forced_test.md"
PROJ_JSON = RES / "mistral_pilot_projection.json"
REPORT_JSON = RES / "mistral_pilot.json"
REPORT_MD = RES / "mistral_pilot.md"


# ----------------------------------------------------------------------------------------------------------------
# Key, HTTP
# ----------------------------------------------------------------------------------------------------------------

def read_key(env_path: Path = ROOT / ".env") -> str:
    """MISTRAL_API_KEY from the environment. The value never leaves this process. `env_path` is not read; it stays
    in the signature for the callers."""
    v = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not v:
        raise SystemExit("MISTRAL_API_KEY is not set in the environment")
    return v


def redact(text, secret: str | None) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if secret:
        s = s.replace(secret, "REDACTED")
    return re.sub(r"(?i)(authorization\"?\s*[:=]\s*\"?bearer\s+)[^\s\"']+", r"\1REDACTED", s)


class Http:
    """urllib client for Mistral's files and batch endpoints. Only GET, file upload, job creation and job cancel."""

    def __init__(self, key: str, base: str = BASE, timeout: float = 600.0):
        self._key, self.base, self.timeout = key, base.rstrip("/"), timeout

    def _do(self, method: str, path: str, data: bytes | None, ctype: str | None) -> dict:
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("Authorization", "Bearer " + self._key)
        r.add_header("Accept", "application/json")
        if ctype:
            r.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                raw, code = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
            except Exception:  # noqa: BLE001
                raw = b""
            code = e.code
        except Exception as e:  # noqa: BLE001
            return {"status": None, "json": None, "text": redact(f"{type(e).__name__}: {e}", self._key)}
        txt = raw.decode("utf-8", "replace")
        try:
            return {"status": code, "json": json.loads(txt), "text": None}
        except ValueError:
            return {"status": code, "json": None, "text": redact(txt, self._key)}

    def get(self, path: str) -> dict:
        return self._do("GET", path, None, None)

    def post_json(self, path: str, body: dict) -> dict:
        return self._do("POST", path, json.dumps(body).encode("utf-8"), "application/json")

    def upload(self, data: bytes, filename: str) -> dict:
        b = "----ml" + uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
        return self._do("POST", "/files", body, f"multipart/form-data; boundary={b}")

    def content(self, file_id: str) -> dict:
        return self._do("GET", f"/files/{file_id}/content", None, None)


def err_text(resp: dict, key: str | None = None) -> str:
    body = resp.get("json") if resp.get("json") is not None else resp.get("text")
    return redact(body if body is not None else "", key)[:2000]


# ----------------------------------------------------------------------------------------------------------------
# Template, tokens
# ----------------------------------------------------------------------------------------------------------------

def hf_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_ID, name, revision=HF_REV))


def system_message(template_text: str | None = None, prompt_txt: str | None = None) -> str:
    """The template's default system message, checked against SYSTEM_PROMPT.txt."""
    t = template_text if template_text is not None else hf_file("chat_template.jinja").read_text(encoding="utf-8")
    m = re.search(r"set default_system_message = '(.*?)(?<!\\)' %\}", t, re.S)
    if not m:
        raise SystemExit("default_system_message not found in the chat template")
    s = m.group(1).replace("\\'", "'")
    ref = prompt_txt if prompt_txt is not None else hf_file("SYSTEM_PROMPT.txt").read_text(encoding="utf-8")
    if s.strip() != ref.strip():
        raise SystemExit("the template's default system message differs from SYSTEM_PROMPT.txt")
    return s


class Counter:
    """Local token counts with Large 3's tokenizer and chat template (projection and the cut at the panel's cap)."""

    def __init__(self):
        from transformers import PreTrainedTokenizerFast
        self.tok = PreTrainedTokenizerFast(tokenizer_file=str(hf_file("tokenizer.json")))
        self.tpl = hf_file("chat_template.jinja").read_text(encoding="utf-8")

    def prompt(self, messages: list[dict]) -> int:
        msgs = [{k: v for k, v in m.items() if k != "prefix"} for m in messages]
        cont = msgs[-1]["role"] == "assistant"
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=not cont,
                                         **({"continue_final_message": True} if cont else {}), chat_template=self.tpl)
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    def ids(self, text: str) -> list[int]:
        return self.tok(text or "", add_special_tokens=False)["input_ids"]

    def cut(self, text: str, n: int) -> str:
        return self.tok.decode(self.ids(text)[:n])

    def record(self) -> dict:
        return {"hf_id": HF_ID, "revision": HF_REV,
                "chat_template_sha256": hashlib.sha256(self.tpl.encode("utf-8")).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(hf_file("tokenizer.json").read_bytes()).hexdigest()}


# ----------------------------------------------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------------------------------------------

def caps(task: str) -> tuple[dict, dict]:
    d = L.task_decoding_dict(task)
    return {a: int(v) for a, v in d["max_new_tokens"].items()}, {a: int(v) for a, v in d["cut_caps"].items()}


def load_design(tasks=L.TASKS, req_dir: Path = ROOT / "data" / "requests" / "ladder") -> pd.DataFrame:
    frames = []
    for t in tasks:
        df = pd.read_parquet(req_dir / t / f"{REQ_KEY}.parquet",
                             columns=["request_id", "arm", "config_id", "item_id", "messages_json"])
        df["item_id"] = df["item_id"].astype(str)
        df["config_id"] = df["config_id"].astype(str)
        mx, cut = caps(t)
        df["task"] = t
        df["key"] = t + "|" + df["request_id"].astype(str)
        df["cap"] = df["arm"].map(mx).astype(int)
        df["panel_cap"] = df["arm"].map(lambda a: cut.get(a, mx[a])).astype(int)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["key"].duplicated().any():
        raise SystemExit("duplicated request keys in the ladder files")
    return out


def unit_hash(task: str, config_id: str, item_id: str, salt: str = SALT) -> str:
    return hashlib.sha256(f"{salt}|{task}|{config_id}|{item_id}".encode()).hexdigest()


def base_config(task: str, config_id: str) -> str:
    """The five-shot requested arm's configuration ids carry a '<task>/' prefix; the unit is the same
    configuration, so the prefix is dropped for the draw and the five-shot rows share their zero-shot units' items."""
    p = f"{task}/"
    return config_id[len(p):] if config_id.startswith(p) else config_id


def pilot_rows(df: pd.DataFrame, share: float = PILOT_SHARE) -> pd.DataFrame:
    """Per benchmark and configuration, the first ceil(share x items) items in sha256 order; every arm of each unit.
    The rule and salt of scripts/nebius_panel.pilot_units."""
    d = df.assign(cfg=[base_config(t, c) for t, c in zip(df["task"], df["config_id"])])
    u = d[["task", "cfg", "item_id"]].drop_duplicates().reset_index(drop=True)
    u["h"] = [unit_hash(t, c, i) for t, c, i in zip(u["task"], u["cfg"], u["item_id"])]
    parts = []
    for _, g in u.groupby(["task", "cfg"], sort=True):
        parts.append(g.sort_values("h").head(math.ceil(share * len(g))))
    pu = pd.concat(parts, ignore_index=True)
    out = d.merge(pu[["task", "cfg", "item_id", "h"]], on=["task", "cfg", "item_id"])
    return out.reset_index(drop=True)


def messages_of(row, system: str | None) -> list[dict]:
    msgs = json.loads(row["messages_json"])
    out = ([{"role": "system", "content": system}] if system is not None else []) + msgs
    if row["arm"] in FORCED:
        out.append({"role": "assistant", "content": PREFILL, "prefix": True})
    return out


def body_of(row, system: str | None) -> dict:
    return {"messages": messages_of(row, system), "max_tokens": int(row["cap"]), "temperature": 0.0, "top_p": 1.0,
            "random_seed": SEED}


def batch_line(row, system: str | None, custom_id: str | None = None) -> str:
    return json.dumps({"custom_id": custom_id or row["key"], "body": body_of(row, system)}, ensure_ascii=False)


def split_files(lines: list[str], max_bytes: int = MAX_FILE_BYTES) -> list[list[str]]:
    out, cur, size = [], [], 0
    for ln in lines:
        b = len(ln.encode("utf-8")) + 1
        if cur and size + b > max_bytes:
            out.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += b
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Money
# ----------------------------------------------------------------------------------------------------------------

def cost(prompt_tokens, completion_tokens, price: dict = PRICE) -> float:
    return (int(prompt_tokens or 0) * price["input"] + int(completion_tokens or 0) * price["output"]) \
        * price["batch_factor"] / 1e6


def expected_output(task: str, arm: str, cap: int, proxy=None) -> float:
    """Expected output tokens of one request: the ladder's prior length of Ministral 3 14B (src/ladder.arm_tokens, an
    upper bound since the panel's cut outputs count at the raised cap), never above the cap."""
    return float(min(cap, L.arm_tokens(REQ_KEY, task, arm, proxy)[0]))


def projection(rows: pd.DataFrame, n_prompt: list[int], proxy=None, price: dict = PRICE) -> dict:
    r = rows.assign(n_prompt=n_prompt)
    cells, tot_in, tot_out = [], 0, 0.0
    for (t, a), g in r.groupby(["task", "arm"], sort=True):
        out_per = expected_output(t, a, int(g["cap"].iloc[0]), proxy)
        n_in, n_out = int(g["n_prompt"].sum()), out_per * len(g)
        cells.append({"task": t, "arm": a, "requests": int(len(g)), "input_tokens": n_in,
                      "expected_output_tokens": round(n_out), "cap": int(g["cap"].iloc[0]),
                      "usd": cost(n_in, n_out, price), "usd_all_at_cap": cost(n_in, int(g["cap"].sum()), price)})
        tot_in += n_in
        tot_out += n_out
    return {"requests": int(len(r)), "input_tokens": tot_in, "expected_output_tokens": round(tot_out),
            "usd": cost(tot_in, tot_out, price), "usd_all_at_cap": float(sum(c["usd_all_at_cap"] for c in cells)),
            "cells": cells, "price": price}


def scale_to_design(pj: dict, df: pd.DataFrame, price: dict = PRICE) -> dict:
    """The pilot projection's per-request means (local prompt counts, expected outputs) times the design's requests."""
    n = df.groupby(["task", "arm"]).size()
    tin = tout = 0.0
    for c in pj["cells"]:
        k = int(n.get((c["task"], c["arm"]), 0))
        tin += c["input_tokens"] / c["requests"] * k
        tout += c["expected_output_tokens"] / c["requests"] * k
    return {"requests": int(n.sum()), "input_tokens": round(tin), "expected_output_tokens": round(tout),
            "usd": cost(tin, tout, price)}


# ----------------------------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------------------------

def content_text(msg: dict) -> str | None:
    c = (msg or {}).get("content")
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else None


def parse_output_line(o: dict) -> dict:
    """One batch output line -> status, text, finish reason, usage; the prefix removed if the reply echoes it."""
    resp = o.get("response") or {}
    body = resp.get("body") if isinstance(resp.get("body"), dict) else None
    status = resp.get("status_code")
    ok = status == 200 and body is not None and not o.get("error")
    ch = ((body or {}).get("choices") or [{}])[0] if ok else {}
    u = (body or {}).get("usage") or {}
    return {"status": status, "ok": bool(ok), "text": content_text(ch.get("message")) if ok else None,
            "finish_reason": ch.get("finish_reason"),
            "usage": {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                      "cached_tokens": ((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                                        if isinstance(u.get("prompt_tokens_details"), dict) else None)},
            "error": None if ok else json.dumps(o.get("error") or body or resp, ensure_ascii=False)[:1000]}


def strip_prefix(text: str | None, arm: str) -> tuple[str | None, bool]:
    if arm in FORCED and isinstance(text, str) and text.startswith(PREFILL):
        return text[len(PREFILL):], True
    return text, False


def parse_jsonl(text: str) -> list[dict]:
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def raw_record(meta: dict, o: dict, job_id: str, price: dict = PRICE) -> dict:
    p = parse_output_line(o)
    text, echoed = strip_prefix(p["text"], meta["arm"])
    return {"key": meta["key"], "task": meta["task"], "request_id": meta["request_id"], "arm": meta["arm"],
            "config_id": meta["config_id"], "item_id": meta["item_id"], "phase": meta.get("phase", "pilot"),
            "status": p["status"], "text": text, "prefix_echoed": echoed, "finish_reason": p["finish_reason"],
            "usage": p["usage"], "cost_usd": cost(p["usage"]["prompt_tokens"], p["usage"]["completion_tokens"], price)
            if p["ok"] else 0.0, "cap": int(meta["cap"]), "panel_cap": int(meta["panel_cap"]),
            "n_prompt_tokens_local": meta.get("n_prompt_tokens_local"), "job_id": job_id,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": p["error"]}


def read_raw(path: Path = RAW) -> list[dict]:
    if not path.exists():
        return []
    return parse_jsonl(path.read_text(encoding="utf-8"))


def load_state(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(p: Path, st: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# ----------------------------------------------------------------------------------------------------------------
# Forced pass rule (the Nebius run's, scripts/nebius_prefill_test.forced_pass)
# ----------------------------------------------------------------------------------------------------------------

RESTART_MARKS = ["\\boxed", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "<s>", "</s>", "<think>", "</think>"]


def forced_pass(text) -> dict:
    """Pass when the reply continues inside the opened box and closes it: the box closes (brace depth), the text
    before the closing brace is a short non-empty answer (no newline, at most a few words), and no restart."""
    t = text if isinstance(text, str) else ""
    reasons = [f"restart or new turn: {m!r}" for m in RESTART_MARKS if m in t]
    depth, close = 1, None
    for i, ch in enumerate(t):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            close = i
            break
    if close is None:
        reasons.append("box not closed")
    else:
        ans = t[:close].strip()
        if not ans:
            reasons.append("empty answer")
        elif "\n" in ans or len(ans.split()) > 8:
            reasons.append("long answer before the brace")
    return {"pass": not reasons, "reasons": reasons}


# ----------------------------------------------------------------------------------------------------------------
# Batch jobs
# ----------------------------------------------------------------------------------------------------------------

def create_job(http: Http, lines: list[str], name: str, meta: dict, key: str | None = None) -> dict:
    data = ("\n".join(lines) + "\n").encode("utf-8")
    up = http.upload(data, f"{name}.jsonl")
    rec = {"name": name, "lines": len(lines), "bytes": len(data), "upload_http": up.get("status")}
    if not (isinstance(up.get("json"), dict) and up["json"].get("id")):
        rec["error"] = "upload: " + err_text(up, key)
        return rec
    rec["input_file"] = up["json"]["id"]
    cr = http.post_json("/batch/jobs", {"input_files": [rec["input_file"]], "model": MODEL, "endpoint": ENDPOINT,
                                        "metadata": {k: str(v) for k, v in meta.items()}})
    rec["create_http"] = cr.get("status")
    if not (cr.get("status") and 200 <= cr["status"] < 300 and isinstance(cr.get("json"), dict)
            and cr["json"].get("id")):
        rec["error"] = "create: " + err_text(cr, key)
        return rec
    rec["job_id"] = cr["json"]["id"]
    rec["status"] = cr["json"].get("status")
    return rec


def job_status(http: Http, job_id: str) -> dict:
    r = http.get(f"/batch/jobs/{job_id}")
    j = r.get("json") if isinstance(r.get("json"), dict) else {}
    return {k: j.get(k) for k in ("status", "total_requests", "succeeded_requests", "failed_requests",
                                  "output_file", "error_file", "created_at", "started_at", "completed_at")} | \
        {"http": r.get("status")}


def fetch_lines(http: Http, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    c = http.content(file_id)
    if c.get("text") is not None:
        return parse_jsonl(c["text"])
    j = c.get("json")
    return [j] if isinstance(j, dict) else (j or [])


# ----------------------------------------------------------------------------------------------------------------
# price
# ----------------------------------------------------------------------------------------------------------------

def cmd_price(a, http: Http | None = None) -> int:
    key = read_key()
    http = http or Http(key)
    m = http.get(f"/models/{MODEL}")
    lst = http.get("/models")
    ids = sorted(x.get("id") for x in ((lst.get("json") or {}).get("data") or []) if isinstance(x, dict))
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL_DOC, "price_usd_per_million": PRICE,
           "models_endpoint": {"http": m.get("status"), "response": err_text(m, key)[:500]},
           "listed": MODEL in ids, "large_models_listed": [i for i in ids if "large" in i],
           "models_listed": len(ids)}
    RES.mkdir(parents=True, exist_ok=True)
    PRICE_JSON.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k != "price_usd_per_million"}, indent=1, ensure_ascii=False))
    return 0


# ----------------------------------------------------------------------------------------------------------------
# forced test
# ----------------------------------------------------------------------------------------------------------------

def forced_test_rows(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Three GSM8K questions in the forced and five-shot forced arms, one forced question each of MATH-500,
    BIG-Bench Hard and HotpotQA (the first items of the pilot draw), and the first GSM8K question again without the
    system message (custom id suffix |nosys)."""
    p = pilot_rows(df)
    g = p[(p["task"] == "gsm8k") & p["arm"].isin(FORCED)]
    cfg = sorted(g["config_id"].unique())[0]
    g = g[g["config_id"] == cfg].sort_values("h")
    items = list(dict.fromkeys(g["item_id"]))[:n]
    parts = [g[g["item_id"].isin(items) & g["arm"].isin(FORCED)]]
    for t in ("math", "bbh", "hotpot"):
        x = p[(p["task"] == t) & (p["arm"] == "forced_direct")].sort_values(["config_id", "h"])
        parts.append(x.head(1))
    out = pd.concat(parts, ignore_index=True)
    out["custom_id"] = out["key"]
    first = out.iloc[[0]].copy()
    first["custom_id"] = first["key"] + "|nosys"
    return pd.concat([out, first], ignore_index=True).drop(columns=["h", "cfg"])


def cmd_forced_test(a, http: Http | None = None) -> int:
    st = load_state(FORCED_STATE)
    if st.get("job_id") and not a.force:
        print(f"forced test already submitted: job {st['job_id']} (forced-collect reads it)")
        return 0
    key = read_key()
    http = http or Http(key)
    system = system_message()
    rows = forced_test_rows(load_design())
    cnt = Counter()
    lines, metas = [], []
    for _, r in rows.iterrows():
        sysm = None if r["custom_id"].endswith("|nosys") else system
        lines.append(batch_line(r, sysm, r["custom_id"]))
        metas.append({"custom_id": r["custom_id"], "key": r["key"], "task": r["task"], "arm": r["arm"],
                      "request_id": r["request_id"], "config_id": r["config_id"], "item_id": r["item_id"],
                      "cap": int(r["cap"]), "panel_cap": int(r["panel_cap"]), "system": sysm is not None,
                      "n_prompt_tokens_local": cnt.prompt(messages_of(r, sysm))})
    up_usd = sum(cost(m["n_prompt_tokens_local"], m["cap"]) for m in metas)
    rec = create_job(http, lines, "forced_test", {"run": "forced-test", "model": MODEL}, key)
    st = {"date": time.strftime("%Y-%m-%d %H:%M"), "requests": metas, "upper_usd": up_usd, **rec,
          "template": cnt.record()}
    save_state(FORCED_STATE, st)
    print(json.dumps({k: v for k, v in st.items() if k != "requests"}, indent=1, ensure_ascii=False))
    if rec.get("error"):
        write_forced_report(st, None, [])
        return 3
    return wait_job(http, rec["job_id"], a.wait_min)


def wait_job(http: Http, job_id: str, wait_min: float, poll_sec: int = POLL_SEC, sleep=time.sleep) -> int:
    t_end = time.time() + 60 * wait_min
    while True:
        s = job_status(http, job_id)
        print(f"[{time.strftime('%H:%M:%S')}] job {job_id}: {s.get('status')} ({s.get('succeeded_requests')}/"
              f"{s.get('total_requests')})", flush=True)
        if s.get("status") in TERMINAL or time.time() + poll_sec > t_end:
            return 0
        sleep(poll_sec)


def cmd_forced_collect(a, http: Http | None = None) -> int:
    st = load_state(FORCED_STATE)
    if not st.get("job_id"):
        print("no forced-test job")
        write_forced_report(st, None, [])
        return 2
    key = read_key()
    http = http or Http(key)
    s = job_status(http, st["job_id"])
    print(f"job {st['job_id']}: {s.get('status')}")
    if s.get("status") not in TERMINAL:
        return 1
    outs = {o.get("custom_id"): o for o in fetch_lines(http, s.get("output_file"))}
    errs = fetch_lines(http, s.get("error_file"))
    res = []
    for m in st["requests"]:
        o = outs.get(m["custom_id"])
        p = parse_output_line(o) if o else {"ok": False, "text": None, "status": None, "usage": {},
                                            "finish_reason": None, "error": "no output line"}
        text, echoed = strip_prefix(p["text"], m["arm"])
        fp = forced_pass(text)
        res.append({**m, "status": p["status"], "raw_text": p["text"], "text": text, "prefix_echoed": echoed,
                    "finish_reason": p["finish_reason"], "usage": p["usage"], "pass": fp["pass"],
                    "reasons": fp["reasons"], "error": p["error"],
                    "cost_usd": cost((p["usage"] or {}).get("prompt_tokens"), (p["usage"] or {}).get("completion_tokens"))
                    if p["ok"] else 0.0})
    write_forced_report(st, s, res, errs)
    print(FORCED_MD.read_text(encoding="utf-8"))
    return 0


def forced_verdict(res: list[dict]) -> dict:
    main = [r for r in res if r.get("system", True)]
    n_ok = sum(r["status"] == 200 for r in main)
    k = sum(r["pass"] for r in main)
    nosys = next((r for r in res if not r.get("system", True)), None)
    twin = next((r for r in res if nosys and r["key"] == nosys["key"] and r.get("system", True)), None)
    inj = None
    if nosys and twin and nosys.get("usage") and twin.get("usage") and nosys["usage"].get("prompt_tokens") \
            and twin["usage"].get("prompt_tokens"):
        inj = {"with_system_server": twin["usage"]["prompt_tokens"], "with_system_local": twin["n_prompt_tokens_local"],
               "without_system_server": nosys["usage"]["prompt_tokens"],
               "without_system_local": nosys["n_prompt_tokens_local"]}
    return {"requests": len(main), "answered": n_ok, "pass": k, "rate": (k / len(main)) if main else None,
            "faithful": bool(main) and n_ok == len(main) and k / len(main) >= PASS_MIN,
            "prefix_echoed": sum(bool(r.get("prefix_echoed")) for r in main), "system_check": inj,
            "same_text_without_system": (nosys["text"] == twin["text"]) if nosys and twin else None}


def write_forced_report(st: dict, s: dict | None, res: list[dict], errs: list | None = None) -> None:
    v = forced_verdict(res) if res else None
    rec = {"date": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL, "endpoint": ENDPOINT, "state": {
        k: st.get(k) for k in ("job_id", "input_file", "upload_http", "create_http", "error", "upper_usd", "template")},
        "job": s, "results": res, "verdict": v, "error_file_lines": (errs or [])[:20]}
    RES.mkdir(parents=True, exist_ok=True)
    FORCED_JSON.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    Lm = [f"# Mistral Large 3, forced-baseline test by batch ({rec['date']})", "",
          f"Model `{MODEL}` on Mistral's batch API (`/v1/batch/jobs`, endpoint `{ENDPOINT}`); script "
          "`scripts/mistral_ladder.py forced-test`. The forced arm is a final assistant message `\\boxed{` with "
          "`prefix: true`; each request carries Large 3's default system message.", ""]
    if st.get("error"):
        Lm += [f"Batch not run: {st['error']} (upload HTTP {st.get('upload_http')}, creation HTTP "
               f"{st.get('create_http')}; key redacted). No other call was made."]
    if s:
        Lm += [f"Job `{st.get('job_id')}`: {s.get('status')}, {s.get('succeeded_requests')} of "
               f"{s.get('total_requests')} requests succeeded.", ""]
    if res:
        Lm += ["| Benchmark | Arm | Item | System | Reply (after the prefix) | Finish | Prompt tok. (server/local) | "
               "Pass |", "|---|---|---|---|---|---|---|---|"]
        for r in res:
            u = r.get("usage") or {}
            Lm.append(f"| {r['task']} | {r['arm']} | {r['item_id']} | {'yes' if r['system'] else 'no'} | "
                      f"`{(r['text'] or '')[:60]!r}` | {r['finish_reason']} | {u.get('prompt_tokens')}/"
                      f"{r['n_prompt_tokens_local']} | {'yes' if r['pass'] else 'no: ' + '; '.join(r['reasons'])} |")
        Lm += ["", f"Verdict: {v['pass']} of {v['requests']} forced replies pass (rule: the box closes on a short "
                   f"answer with no restart; at least {PASS_MIN:.0%}); replies echoing the prefix: "
                   f"{v['prefix_echoed']}. Forced arm faithful: **{'yes' if v['faithful'] else 'no'}**.",
               f"System message check: {v['system_check']}; same reply without it: {v['same_text_without_system']}.",
               f"Spend: ${sum(r['cost_usd'] for r in res):.5f} (batch price)."]
    FORCED_MD.write_text("\n".join(Lm) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# pilot
# ----------------------------------------------------------------------------------------------------------------

def spent_total(raw: list[dict] | None = None) -> float:
    raw = read_raw() if raw is None else raw
    forced = 0.0
    if FORCED_JSON.exists():
        forced = sum(r.get("cost_usd") or 0.0 for r in json.loads(FORCED_JSON.read_text(encoding="utf-8"))
                     .get("results") or [])
    return forced + sum(r.get("cost_usd") or 0.0 for r in raw)


def cmd_pilot_submit(a, http: Http | None = None) -> int:
    st = load_state(STATE)
    if st.get("jobs") and not a.force:
        print(f"pilot already submitted: {len(st['jobs'])} jobs (collect reads them)")
        return 0
    fv = json.loads(FORCED_JSON.read_text(encoding="utf-8")).get("verdict") if FORCED_JSON.exists() else None
    if not fv and not a.dry_run:
        print("no forced-test verdict: run forced-test and forced-collect first")
        return 2
    skip_forced = bool(fv) and not fv.get("faithful")   # a dry run without a verdict projects every arm
    if skip_forced and not a.without_forced and not a.dry_run:
        print("the forced test did not pass; rerun with --without-forced to pilot the other arms (flagged)")
        return 2
    df = load_design()
    p = pilot_rows(df)
    if skip_forced:
        p = p[~p["arm"].isin(FORCED)].reset_index(drop=True)
    system = system_message()
    cnt = Counter()
    n_prompt = [cnt.prompt(messages_of(r, system)) for _, r in p.iterrows()]
    pj = projection(p, n_prompt)
    spent = spent_total()
    pj.update({"spent_before_usd": spent, "limit_usd": a.limit_usd, "fits": spent + pj["usd"] <= a.limit_usd,
               "forced_arms_included": not skip_forced, "template": cnt.record(),
               "pilot_rule": f"per benchmark and configuration, the first ceil({PILOT_SHARE:.0%} x items) items in "
                             f"sha256('{SALT}|task|config|item') order; every arm of each unit",
               "design_requests": int(len(df)), "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
    pj["full_design"] = scale_to_design(pj, df)
    RES.mkdir(parents=True, exist_ok=True)
    PROJ_JSON.write_text(json.dumps(pj, indent=1), encoding="utf-8")
    print(f"full design (same per-request means): ${pj['full_design']['usd']:.2f}")
    print(f"pilot: {pj['requests']} requests ({pj['requests'] / len(df):.2%} of {len(df)}); input "
          f"{pj['input_tokens']:,} tok; expected output {pj['expected_output_tokens']:,}; projected ${pj['usd']:.2f} "
          f"(all at cap ${pj['usd_all_at_cap']:.2f}); spent ${spent:.4f}; limit ${a.limit_usd:.2f}")
    if not pj["fits"]:
        print("refused: the projection passes the limit")
        return 2
    if a.dry_run:
        return 0
    key = read_key()
    http = http or Http(key)
    p = p.assign(n_prompt_tokens_local=n_prompt)
    jobs = []
    for t in L.TASKS:
        g = p[p["task"] == t]
        lines = [batch_line(r, system) for _, r in g.iterrows()]
        for j, chunk in enumerate(split_files(lines)):
            rec = create_job(http, chunk, f"pilot_{t}_{j}", {"run": "pilot", "task": t, "part": j}, key)
            rec["task"] = t
            rec["projected_usd"] = float(sum(c["usd"] for c in pj["cells"] if c["task"] == t)) * len(chunk) / len(g)
            jobs.append(rec)
            print(json.dumps(rec), flush=True)
            if rec.get("error"):
                break
    meta = p[["key", "task", "request_id", "arm", "config_id", "item_id", "cap", "panel_cap",
              "n_prompt_tokens_local"]].copy()
    DATA.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(DATA / "pilot_meta.parquet", index=False)
    save_state(STATE, {"submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "jobs": jobs, "limit_usd": a.limit_usd,
                       "forced_arms_included": not skip_forced})
    return 3 if any(j.get("error") for j in jobs) else 0


def cell_costs(raw: list[dict]) -> dict:
    """Mean cost per answered request of each (task, arm), from the rows already collected."""
    acc: dict = {}
    for r in raw:
        if r.get("status") == 200 or (r.get("usage") or {}).get("completion_tokens"):
            acc.setdefault((r["task"], r["arm"]), []).append(r.get("cost_usd") or 0.0)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def cmd_full_submit(a, http: Http | None = None) -> int:
    """The rest of the complete design for the named benchmarks, by batch, one task at a time; resumable (a task part
    already in state_full.json is not sent again). Refuses when the spend so far plus the projection (the pilot's real
    mean cost per request of each benchmark and arm) passes the limit."""
    tasks = [t for t in a.tasks.split(",") if t]
    bad = [t for t in tasks if t not in L.TASKS]
    if bad:
        print(f"unknown tasks {bad}")
        return 2
    raw = read_raw()
    have = {r["key"] for r in raw}
    cc = cell_costs(raw)
    df = load_design(tuple(tasks))
    rest = df[~df["key"].isin(have)].reset_index(drop=True)
    miss = sorted({(t, x) for t, x in zip(rest["task"], rest["arm"])} - set(cc))
    if miss:
        print(f"no pilot cost for {miss}: refused")
        return 2
    st = load_state(STATE_FULL)
    sent = {(j["task"], j["part"]) for j in st.get("jobs", []) if j.get("job_id")}
    per_task = {t: float(sum(cc[(t, x)] for x in rest.loc[rest["task"] == t, "arm"])) for t in tasks}
    spent = spent_total(raw)
    pend = sum(j.get("projected_usd") or 0.0 for j in st.get("jobs", []) if j.get("job_id") and not j.get("collected"))
    new_usd = sum(v for t, v in per_task.items() if not any(s[0] == t for s in sent))
    print(f"rest of {tasks}: {len(rest)} requests; projected " + ", ".join(f"{t} ${v:.2f}" for t, v in per_task.items())
          + f"; spent ${spent:.4f}; pending ${pend:.2f}; new ${new_usd:.2f}; limit ${a.limit_usd:.2f}")
    if spent + pend + new_usd > a.limit_usd:
        print("refused: the projection passes the limit")
        return 2
    if a.dry_run:
        return 0
    key = read_key()
    http = http or Http(key)
    system = system_message()
    cnt = Counter()
    jobs = st.get("jobs", [])
    metas = [pd.read_parquet(FULL_META)] if FULL_META.exists() else []
    for t in tasks:
        g = rest[rest["task"] == t].reset_index(drop=True)
        if any(s[0] == t for s in sent):
            print(f"{t}: already submitted")
            continue
        g = g.assign(n_prompt_tokens_local=[cnt.prompt(messages_of(r, system)) for _, r in g.iterrows()])
        metas.append(g[["key", "task", "request_id", "arm", "config_id", "item_id", "cap", "panel_cap",
                        "n_prompt_tokens_local"]].copy())
        DATA.mkdir(parents=True, exist_ok=True)
        pd.concat(metas, ignore_index=True).drop_duplicates("key").to_parquet(FULL_META, index=False)
        lines = [batch_line(r, system) for _, r in g.iterrows()]
        chunks = split_files(lines)
        for j, chunk in enumerate(chunks):
            rec = create_job(http, chunk, f"full_{t}_{j}", {"run": "full", "task": t, "part": j}, key)
            rec.update({"task": t, "part": j, "projected_usd": per_task[t] * len(chunk) / len(g)})
            jobs.append(rec)
            save_state(STATE_FULL, {"jobs": jobs, "limit_usd": a.limit_usd, "tasks": tasks,
                                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(json.dumps(rec), flush=True)
            if rec.get("error"):
                return 3
    return 0


def cmd_collect(a, http: Http | None = None) -> int:
    rc = 0
    key = read_key()
    http = http or Http(key)
    for state, meta_path, phase in ((STATE, DATA / "pilot_meta.parquet", "pilot"), (STATE_FULL, FULL_META, "full")):
        if load_state(state).get("jobs"):
            rc = max(rc, collect_one(a, http, key, state, meta_path, phase))
    return rc


def collect_one(a, http, key, state: Path, meta_path: Path, phase: str) -> int:
    st = load_state(state)
    meta = pd.read_parquet(meta_path).set_index("key")
    raw = read_raw()
    have = {r["key"] for r in raw}
    t_end = time.time() + 60 * a.wait_min
    while True:
        new = 0
        for j in st["jobs"]:
            if not j.get("job_id") or j.get("collected"):
                continue
            s = job_status(http, j["job_id"])
            j["status"] = s.get("status")
            j["counts"] = {k: s.get(k) for k in ("total_requests", "succeeded_requests", "failed_requests")}
            if s.get("status") not in TERMINAL:
                continue
            lines = fetch_lines(http, s.get("output_file")) + fetch_lines(http, s.get("error_file"))
            with open(RAW, "a", encoding="utf-8") as fh:
                for o in lines:
                    k = o.get("custom_id")
                    if k in have or k not in meta.index:
                        continue
                    m = {"key": k, **meta.loc[k].to_dict(), "phase": phase}
                    fh.write(redact(json.dumps(raw_record(m, o, j["job_id"]), ensure_ascii=False), key) + "\n")
                    have.add(k)
                    new += 1
            j["collected"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        spent = spent_total()
        pending = [j for j in st["jobs"] if j.get("job_id") and not j.get("collected")]
        pend_usd = sum(j.get("projected_usd") or 0.0 for j in pending)
        if pending and spent + pend_usd > st["limit_usd"]:
            for j in pending:
                c = http.post_json(f"/batch/jobs/{j['job_id']}/cancel", {})
                j["cancel_http"] = c.get("status")
            st["cancelled_for_limit"] = f"spent ${spent:.4f} + pending ${pend_usd:.4f} > ${st['limit_usd']:.2f}"
        save_state(state, st)
        print(f"[{time.strftime('%H:%M:%S')}] {phase}: new rows {new}; collected jobs "
              f"{sum(bool(j.get('collected')) for j in st['jobs'])}/{len(st['jobs'])}; spent ${spent:.4f}; "
              + "; ".join(f"{j['name']} {j.get('status')} {j.get('counts')}" for j in st["jobs"]), flush=True)
        if not pending or time.time() + POLL_SEC > t_end:
            return 0 if not pending else 1
        time.sleep(POLL_SEC)


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------

def outputs_frame(raw: list[dict], task: str, counter=None) -> pd.DataFrame:
    """Rows of one task with the uncut text and the copy cut at the panel's cap (token level, Large 3's tokenizer,
    only where the server's completion count passes the cap); forced GSM8K and BIG-Bench Hard replies cut after
    their first '}'."""
    rows = [r for r in raw if r["task"] == task and r.get("status") == 200 and isinstance(r.get("text"), str)]
    out = []
    for r in rows:
        txt, fin = r["text"], r["finish_reason"]
        n_out = int((r.get("usage") or {}).get("completion_tokens") or 0)
        cut_txt, cut_fin, cut = txt, fin, False
        if r["arm"] in FORCED and task in CUT_AT_FIRST_BRACE:
            j = txt.find("}")
            if j >= 0:
                txt = cut_txt = txt[: j + 1]
                fin = cut_fin = "stop"
        elif r["arm"] not in FORCED and n_out > int(r["panel_cap"]):
            cut = True
            cut_txt = counter.cut(txt, int(r["panel_cap"])) if counter is not None else txt
            cut_fin = "length"
        out.append({"request_id": r["request_id"], "key": r["key"], "model_key": MODEL_KEY, "arm": r["arm"],
                    "config_id": str(r["config_id"]), "item_id": str(r["item_id"]), "output_text": txt,
                    "output_text_cut": cut_txt, "finish_reason": fin, "finish_reason_cut": cut_fin,
                    "cut_at_panel_cap": cut, "n_output_tokens": n_out,
                    "n_prompt_tokens": int((r.get("usage") or {}).get("prompt_tokens") or 0),
                    "panel_cap": int(r["panel_cap"]), "cap": int(r["cap"])})
    return pd.DataFrame(out)


def score_frame(o: pd.DataFrame, task: str, reading: str) -> pd.DataFrame:
    import nebius_panel_report as npr
    s = o.copy()
    if reading == "cut":
        s["output_text"] = s["output_text_cut"]
        s["finish_reason"] = s["finish_reason_cut"]
    forced = s["arm"].isin(FORCED)
    s.loc[forced, "output_text"] = PREFILL + s.loc[forced, "output_text"].fillna("")
    arm = s["arm"].copy()
    s["arm"] = np.where(forced, "forced_prefilled", s["arm"])   # the prefill is already in front
    s["raw_text"] = s["output_text"]
    sc, ver = npr.score_task(s.drop(columns=["output_text_cut", "finish_reason_cut"]), task)
    sc["arm"] = arm.to_numpy()
    sc["reading"] = reading
    sc.attrs["versions"] = ver
    return sc


def cmd_report(a) -> int:
    raw = read_raw()
    if not raw:
        print("no pilot outputs yet")
        return 2
    df = load_design()
    cnt = Counter()
    rep = {"model": MODEL, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": {}, "price": PRICE}
    proc = DATA / "scored"
    proc.mkdir(parents=True, exist_ok=True)
    ok = [r for r in raw if r.get("status") == 200 and isinstance(r.get("text"), str)]
    for t in L.TASKS:
        o = outputs_frame(raw, t, cnt)
        if o.empty:
            continue
        tr = {"rows": int(len(o)), "arms": {}}
        for reading in ("cut", "uncut"):
            sc = score_frame(o, t, reading)
            sc.drop(columns=["output_text"], errors="ignore").to_parquet(proc / f"{t}_{reading}.parquet", index=False)
            tr["scorers"] = sc.attrs.get("versions")
            for arm, g in sc.groupby("arm"):
                d = tr["arms"].setdefault(arm, {"n": int(len(g))})
                d[f"strict_{reading}"] = float(100 * g["strict_correct"].mean())
                d[f"content_{reading}"] = float(100 * g["content_correct"].mean())
        for arm, g in o.groupby("arm"):
            d = tr["arms"][arm]
            d["cut_share"] = float(g["cut_at_panel_cap"].mean())
            d["at_raised_cap"] = int((g["finish_reason"] == "length").sum()) if arm not in FORCED else None
            d["mean_output_tokens"] = float(g["n_output_tokens"].mean())
            d["mean_prompt_tokens"] = float(g["n_prompt_tokens"].mean())
            if arm in FORCED:
                fp = [forced_pass(x)["pass"] for x in
                      [r["text"] for r in ok if r["task"] == t and r["arm"] == arm]]
                d["forced_pass"] = float(np.mean(fp)) if fp else None
        rep["tasks"][t] = tr
    # full projection from the pilot's real tokens: per task and arm, mean billed cost x the design's requests
    n_design = df.groupby(["task", "arm"]).size()
    cells, total = [], 0.0
    for (t, arm), n in n_design.items():
        xs = [r for r in ok if r["task"] == t and r["arm"] == arm]
        c = float(np.mean([r["cost_usd"] for r in xs])) if xs else None
        cells.append({"task": t, "arm": arm, "design_requests": int(n), "pilot_requests": len(xs),
                      "usd_per_request": c, "projected_usd": c * n if c is not None else None,
                      "prompt_tokens_mean": float(np.mean([r["usage"]["prompt_tokens"] or 0 for r in xs])) if xs else None,
                      "output_tokens_mean": float(np.mean([r["usage"]["completion_tokens"] or 0 for r in xs])) if xs else None})
        total += c * n if c is not None else 0.0
    rep["full_projection"] = {"cells": cells, "total_usd": total,
                              "missing": [f"{c['task']}/{c['arm']}" for c in cells if c["projected_usd"] is None]}
    rep["spend"] = {"pilot_usd": float(sum(r["cost_usd"] for r in ok)), "total_usd": spent_total(raw),
                    "requests": len(raw), "answered": len(ok),
                    "prompt_tokens": int(sum(r["usage"]["prompt_tokens"] or 0 for r in ok)),
                    "output_tokens": int(sum(r["usage"]["completion_tokens"] or 0 for r in ok)),
                    "cached_tokens": int(sum(r["usage"].get("cached_tokens") or 0 for r in ok)),
                    "prefix_echoed": int(sum(bool(r.get("prefix_echoed")) for r in ok))}
    REPORT_JSON.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    write_report_md(rep)
    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


def write_report_md(rep: dict) -> None:
    name = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
    design_n = sum(c["design_requests"] for c in rep["full_projection"]["cells"])
    full = rep["spend"]["answered"] >= design_n
    what = "complete design" if full else "5% pilot"
    rows = ("Values over every request of the complete design, simple means." if full else
            "Pilot values: about 5% of the items of each configuration, simple means over the pilot's rows.")
    Lm = [f"# Mistral Large 3 (2512), {what} by batch ({rep['written'][:10]})", "",
          f"Model `{MODEL}` on Mistral's batch API; prices ${PRICE['input']}/M input and ${PRICE['output']}/M output, "
          f"batch at {PRICE['batch_factor']:.0%} of that ({PRICE['source']}). {rows}", "",
          "| Benchmark | Arm | n | Strict cut | Strict uncut | Content cut | Content uncut | Cut at panel cap | "
          "At 7,000 | Mean out tok. | Forced pass |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, tr in rep["tasks"].items():
        for arm, d in tr["arms"].items():
            fp = d.get("forced_pass")
            Lm.append(f"| {name[t]} | {arm} | {d['n']} | {d['strict_cut']:.1f} | {d['strict_uncut']:.1f} | "
                      f"{d['content_cut']:.1f} | {d['content_uncut']:.1f} | {d['cut_share']:.1%} | "
                      f"{d['at_raised_cap'] if d['at_raised_cap'] is not None else '-'} | {d['mean_output_tokens']:.0f} | "
                      f"{f'{fp:.1%}' if fp is not None else '-'} |")
    s, fpj = rep["spend"], rep["full_projection"]
    Lm += ["", f"Spend: {'run' if full else 'pilot'} ${s['pilot_usd']:.2f} ({s['answered']:,} of {s['requests']:,} requests answered; "
               f"{s['prompt_tokens']:,} prompt and {s['output_tokens']:,} output tokens; cached {s['cached_tokens']:,}); "
               f"with the forced test ${s['total_usd']:.2f}. Replies echoing the prefix: {s['prefix_echoed']}.", "",
           (f"Cost by cell (complete design, batch price, real token counts): **${fpj['total_usd']:.2f}**." if full else
            f"Projected full run (complete design, batch price, from the pilot's real token counts): "
            f"**${fpj['total_usd']:.2f}**" + (f"; cells without pilot rows: {fpj['missing']}" if fpj["missing"] else "")
            + ". The full run is a separate step."), "",
           "| Benchmark | Arm | Design requests | Prompt tok. | Output tok. | $ per 1k | Projected |",
           "|---|---|---|---|---|---|---|"]
    for c in fpj["cells"]:
        if c["projected_usd"] is None:
            continue
        Lm.append(f"| {name[c['task']]} | {c['arm']} | {c['design_requests']:,} | {c['prompt_tokens_mean']:.0f} | "
                  f"{c['output_tokens_mean']:.0f} | {1000 * c['usd_per_request']:.3f} | ${c['projected_usd']:.2f} |")
    REPORT_MD.write_text("\n".join(Lm) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("price")
    f = sub.add_parser("forced-test")
    f.add_argument("--wait-min", type=float, default=9.0)
    f.add_argument("--force", action="store_true")
    sub.add_parser("forced-collect")
    p = sub.add_parser("pilot-submit")
    p.add_argument("--limit-usd", type=float, default=LIMIT_USD)
    p.add_argument("--without-forced", action="store_true", help="pilot the other arms if the forced test failed")
    p.add_argument("--dry-run", action="store_true", help="projection only, nothing sent")
    p.add_argument("--force", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--wait-min", type=float, default=0.0, help="keep polling (every 5 min) up to this many minutes")
    sub.add_parser("report")
    fs = sub.add_parser("full-submit", help="the rest of the design for --tasks, by batch (cumulative limit)")
    fs.add_argument("--tasks", required=True, help="comma-separated: gsm8k,math,bbh,hotpot")
    fs.add_argument("--limit-usd", type=float, default=FULL_LIMIT_USD)
    fs.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "full-submit":
        if a.limit_usd > FULL_LIMIT_USD:
            print(f"the limit is ${FULL_LIMIT_USD:.2f}.")
            return 2
        return cmd_full_submit(a)
    if getattr(a, "limit_usd", None) is not None and a.limit_usd > LIMIT_USD:
        print(f"the limit is ${LIMIT_USD:.2f}; a higher one needs a change in the code")
        return 2
    return {"price": cmd_price, "forced-test": cmd_forced_test, "forced-collect": cmd_forced_collect,
            "pilot-submit": cmd_pilot_submit, "collect": cmd_collect, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
