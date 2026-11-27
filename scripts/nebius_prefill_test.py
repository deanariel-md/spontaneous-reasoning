"""Nebius prefill test for the frontier panel.

Dry run by default: renders the forced-baseline prompts locally, counts prompt tokens, projects the cost with
placeholder output lengths, writes the report, and makes no network call at all. The live calls run only with
--live, and the key is read only then, from the environment variable NEBIUS_API_KEY. The key and request headers are never printed, logged or written; any error text is
passed through redact() before it is printed or saved.

Prompts
  * Questions: the first three GSM8K items, in stored order, of the probe set of manifests/rp_feasibility_items.json
    (screening items from the GSM8K training split, drawn by the rule recorded in that file). They are the items the
    reasoning panel's own prefill check used.
  * Template: configuration ps-i00-f00 of manifests/gsm8k_bank.json, the bank's first configuration. It is the seed
    template that manifests/rp_feasibility_items.json records for the reasoning panel (config/promptsuite.yaml); the
    script checks that the bank text filled with each question reproduces the stored probe message exactly.
  * Forced arm: as src/generation.py VllmBackend.render: the user message (direct_answer arm) and a final assistant
    message "\\boxed{", rendered with add_generation_prompt=False and continue_final_message=True, thinking off.
    For a Jinja template this is transformers' apply_chat_template, as in VllmBackend.render. DeepSeek-V4-Flash ships
    no Jinja template (tokenizer_config.json has none); its own renderer is encoding/encoding_dsv4.py, used here with
    thinking_mode "chat" and the final assistant message marked wo_eos (the renderer's way to leave a turn open).
  * Requested arm (b): the direct_answer message on the chat endpoint, thinking off, cap 1,024 new tokens.
  * Chain of thought (c): the zero_shot_cot message on the chat endpoint with thinking on, cap 8,192 new tokens:
    the reasoning panel's reasoning arm (scripts/rp_feasibility.py, REASONING_CAP). --cot-thinking off sends it with
    thinking off instead.

Calls (live): GET {base}/models first; then per served model and question: forced on POST {base}/completions
(max_tokens 16, temperature 0). If /completions is refused (HTTP 4xx), the chat endpoint with a final assistant
message "\\boxed{" and "partial": true, then with continue_final_message / add_generation_prompt false. Then (b) and
(c). Hard cap MAX_CALLS calls in all; the run stops if any reply reports more than ABORT_OUTPUT_TOKENS output tokens.

Model ids: Nebius's deprecation notice of November 2026 says deepseek-ai/DeepSeek-V4-Flash-0731 and
deepseek-ai/DeepSeek-V4.1-Flash left Token Factory Serverless on 9 November 2026, with
deepseek-ai/DeepSeek-V4.1-Flash-1017 the named replacement. The live run reads /models; a requested id that is not
listed is tried under its listed alias, then replaced by the named replacement, each served id once, and the report
says which id was called for which request.

    python scripts/nebius_prefill_test.py            # dry run
    python scripts/nebius_prefill_test.py --live     # live, with NEBIUS_API_KEY set in the environment
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "frontier_panel"
STEM = "nebius_prefill_test"
TOK_DIR = ROOT / "data" / "nebius_prefill" / "tokenizers"
ITEMS = ROOT / "manifests" / "rp_feasibility_items.json"
BANK = ROOT / "manifests" / "gsm8k_bank.json"
FEASIBILITY = ROOT / "results" / "reasoning_panel" / "feasibility.json"
BANK_CONFIG = "ps-i00-f00"
N_QUESTIONS = 3
PREFILL = "\\boxed{"
TODAY = "2026-11-22"

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
BASE_URL_SOURCE = ("https://docs.tokenfactory.nebius.com/api-reference/introduction (Python example: "
                   "base_url=\"https://api.tokenfactory.nebius.com/v1/\"); the API reference lists POST /v1/completions "
                   "(https://docs.tokenfactory.nebius.com/api-reference/inference/create-completion.md) and POST "
                   "/v1/chat/completions, server https://api.tokenfactory.nebius.com; read " + TODAY)

# Caps: the reasoning panel's own (scripts/rp_feasibility.py: FORCED_CAP 16, REQUESTED_CAP 1024, REASONING_CAP 8192).
CAPS = {"forced": 16, "requested": 1024, "cot": 8192}
MAX_QUESTIONS = 3
MAX_MODELS = 2
MAX_CALLS = MAX_QUESTIONS * MAX_MODELS * 5   # forced + up to two fallbacks + requested + cot, per question and model
ABORT_OUTPUT_TOKENS = 4000
BUDGET_STOP_USD = 10.0
BATCH_FACTOR = 0.5

THINK_OFF = {"thinking": False, "enable_thinking": False, "thinking_mode": "chat"}
THINK_ON = {"thinking": True, "enable_thinking": True, "thinking_mode": "thinking"}

# Templates and renderers, pinned. "panel_pin" marks the entry config/models_reasoning_panel.yaml pins.
RENDER = {
    "deepseek-ai/DeepSeek-V4-Flash-0731": {
        "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062", "kind": "encoding_dsv4",
        "file": "encoding/encoding_dsv4.py",
        "sha256": "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6",
        "panel_pin": "config/models_reasoning_panel.yaml, entry deepseek-v4-flash-0731 (hf_id and revision)"},
    "deepseek-ai/DeepSeek-V4-Flash": {
        "revision": "60d8d70770c6776ff598c94bb586a859a38244f1", "kind": "encoding_dsv4",
        "file": "encoding/encoding_dsv4.py",
        "sha256": "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0"},
    "deepseek-ai/DeepSeek-V4.1-Flash": {
        "revision": "314113f8a7dadb388215396ba5fbee2c610a04d6", "kind": "jinja", "file": "chat_template.jinja",
        "sha256": "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3"},
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {
        "revision": "08c8657f02291e11b6f6b63e0e16014f8e748518", "kind": "jinja", "file": "chat_template.jinja",
        "sha256": "95b7033a693b321ceaefb6f89ad07d1e0cf2b095305711f934a4ac649456739d",
        "note": "Nebius's catalog points this model at nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 (fp4, revision 3ae9c0d3); "
                "that repo ships no chat_template.jinja, and its encoding/encoding.py is identical (sha256 13231c59) "
                "to the one in deepseek-ai/DeepSeek-V4.1-Flash-1017, whose chat_template.jinja is used here"},
}

# The two models of the request, in its order, with the role each plays in the cost projection.
REQUESTED = [
    {"name": "deepseek-ai/DeepSeek-V4-Flash", "role": "full", "templates": 16,
     "render": "deepseek-ai/DeepSeek-V4-Flash-0731", "compare_render": "deepseek-ai/DeepSeek-V4-Flash",
     "aliases": ["deepseek-ai/DeepSeek-V4-Flash-0731"], "replacement": "deepseek-ai/DeepSeek-V4.1-Flash-1017"},
    {"name": "deepseek-ai/DeepSeek-V4.1-Flash", "role": "four_templates", "templates": 4,
     "render": "deepseek-ai/DeepSeek-V4.1-Flash", "compare_render": None,
     "aliases": [], "replacement": "deepseek-ai/DeepSeek-V4.1-Flash-1017"},
]
RENDER_FOR_SERVED = {"deepseek-ai/DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
                     "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-ai/DeepSeek-V4-Flash-0731",
                     "deepseek-ai/DeepSeek-V4.1-Flash": "deepseek-ai/DeepSeek-V4.1-Flash",
                     "deepseek-ai/DeepSeek-V4.1-Flash-1017": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}

# Per-million-token prices in USD (real time). Batch = BATCH_FACTOR of these.
PRICES = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {
        "input": 0.30, "output": 1.20, "date": TODAY, "official": True,
        "url": "https://tokenfactory.nebius.com/api/public/models_info",
        "note": "Nebius's public model catalog, flavor label 'cheap', region us-north1, quantization fp4"},
    "deepseek-ai/DeepSeek-V4.1-Flash": {
        "input": 0.30, "output": 1.20, "date": "2026-10-22", "official": False,
        "url": "https://futureagi.com/llm-cost-calculator/nebius/deepseek-ai-deepseek-v4-1-flash/",
        "note": "third-party listing ('last verified Oct 22, 2026'); the model is no longer in Nebius's public catalog "
                "(removed from Serverless 9 Nov 2026)"},
    "deepseek-ai/DeepSeek-V4-Flash-0731": {
        "input": 0.14, "output": 0.28, "date": TODAY, "official": False,
        "url": "https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/nebius/model/deepseek-v4-flash",
        "note": "third-party listings (getmaxim.ai, pricepertoken.com) citing Nebius's catalog; the model is no longer "
                "in Nebius's public catalog (removed from Serverless 9 Nov 2026)"},
}
PRICES["deepseek-ai/DeepSeek-V4-Flash"] = dict(PRICES["deepseek-ai/DeepSeek-V4-Flash-0731"])
PRICES["deepseek-ai/DeepSeek-V4-Flash"]["note"] = "price listed for deepseek-ai/DeepSeek-V4-Flash-0731; " \
    + PRICES["deepseek-ai/DeepSeek-V4-Flash-0731"]["note"]

DEPRECATION = {
    "url": "https://docs.tokenfactory.nebius.com/november-2026-deprecation-notice.md",
    "sentence": "These models will be removed from Token Factory Serverless on November 9, 2026. After that date, you "
                "will not be able to select them in the Playground or send API requests using their model IDs.",
    "rows": ["deepseek-ai/DeepSeek-V4-Flash-0731 -> deepseek-ai/DeepSeek-V4.1-Flash-1017",
             "deepseek-ai/DeepSeek-V4.1-Flash -> deepseek-ai/DeepSeek-V4.1-Flash-1017"],
    "read": TODAY,
}

BATCH = {
    "answer": "not confirmed; no official source says yes, and every documented example uses /v1/chat/completions",
    "accepts_completions": None,
    "official": {"url": "https://docs.tokenfactory.nebius.com/data-lab/batch-inference",
                 "status": f"HTTP 404 on {TODAY}; the documentation index (https://docs.tokenfactory.nebius.com/llms.txt) "
                           "has no batch-inference page"},
    "secondary": [
        {"url": "https://skills.lc/NeverSight/learn-skills.dev/neversight-learn-skills-dev-data-skills-md-arindam200-nebius-skills-nebius-batch-synthetic-skill-md",
         "sentence": "client.batches.create(input_file_id=..., endpoint=\"/v1/chat/completions\", completion_window=\"24h\")"},
        {"url": "https://makefun.ai/nebius-token-factory-batch-inference-cost-calculator/",
         "sentence": "async JSONL batch inference for chat completions and embeddings, normally within a 24-hour window, "
                     "billed at 50% of Base real-time model pricing"},
    ],
    "discount_source": "https://nebius.com/solutions/inference (batch at half the real-time price); "
                       "https://x.com/nebiusaistudio/status/1925586867002454070",
    "consequence": "the projection prices the forced arm in real time (no batch discount) and the other arms in batch",
}

GEMINI_NOTE = ("The Gemini Flash part of the request (w on the Google AI Studio free tier) was not attempted: it needs "
               "a Google AI Studio key, which is not available to this build.")


# ----------------------------------------------------------------------------------------------------------------
# Redaction and HTTP
# ----------------------------------------------------------------------------------------------------------------

def redact(text, secret: str | None = None) -> str:
    """Remove the key and any bearer credential from text before it is printed or saved."""
    s = "" if text is None else str(text)
    if secret:
        s = s.replace(secret, "REDACTED")
    s = re.sub(r"(?i)(bearer\s+)[^\s\"',}]+", r"\1REDACTED", s)
    s = re.sub(r"(?i)(\"?(?:authorization|api[_-]?key|x-api-key)\"?\s*[:=]\s*\"?)[^\s\"',}]+", r"\1REDACTED", s)
    return s


class Aborted(Exception):
    """Stops the run. `record` carries the call that tripped the stop, so it is still saved and counted."""

    def __init__(self, msg: str, record: dict | None = None):
        super().__init__(msg)
        self.record = record


class Client:
    """Minimal OpenAI-compatible client (urllib). Counts POST calls against the hard cap."""

    def __init__(self, base_url: str, key: str, max_calls: int = MAX_CALLS, timeout: float = 600.0):
        self.base, self._key, self.max_calls, self.timeout = base_url.rstrip("/"), key, max_calls, timeout
        self.calls = 0

    def _request(self, method: str, path: str, body: dict | None):
        url = self.base + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", "Bearer " + self._key)
        req.add_header("Content-Type", "application/json")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return {"status": r.status, "json": json.loads(r.read().decode("utf-8")),
                        "seconds": round(time.time() - t0, 2)}
        except urllib.error.HTTPError as e:
            try:
                txt = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                txt = ""
            return {"status": e.code, "error": redact(txt[:2000], self._key), "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            return {"status": None, "error": redact(f"{type(e).__name__}: {e}", self._key),
                    "seconds": round(time.time() - t0, 2)}

    def get(self, path: str):
        return self._request("GET", path, None)

    def post(self, path: str, body: dict):
        if self.calls >= self.max_calls:
            raise Aborted(f"hard cap of {self.max_calls} calls reached")
        self.calls += 1
        return self._request("POST", path, body)


# ----------------------------------------------------------------------------------------------------------------
# Pass rule
# ----------------------------------------------------------------------------------------------------------------

RESTART_MARKS = ["\\boxed", "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
                 "<｜begin▁of▁sentence｜>", "<|im_start|>", "<|start|>", "<|channel|>"]


def forced_pass(text: str | None, reasoning: str | None = None) -> dict:
    """Pass when the reply continues inside the opened box and closes it: a "}" is present, the text before the
    first "}" is a short non-empty answer (no newline, at most three words), and the reply holds no restart (a new
    \\boxed, a <think> or </think>, a role or turn token) and no separate reasoning text."""
    reasons = []
    t = text or ""
    if reasoning and reasoning.strip():
        reasons.append("reasoning text returned beside the reply")
    for m in RESTART_MARKS:
        if m in t:
            reasons.append(f"restart or new turn: {m!r} in the reply")
    if "}" not in t:
        reasons.append("box not closed: no '}' in the reply")
        before = t
    else:
        before = t.split("}", 1)[0]
        if not before.strip():
            reasons.append("nothing before the closing brace")
    if "\n" in before.strip():
        reasons.append("reasoning text before the brace: a line break")
    if len(before.split()) > 3:
        reasons.append(f"reasoning text before the brace: {len(before.split())} words")
    return {"pass": not reasons, "reasons": reasons, "answer": before.strip() if "}" in t else None}


# ----------------------------------------------------------------------------------------------------------------
# Prompts and rendering
# ----------------------------------------------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_questions(n: int = N_QUESTIONS) -> dict:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    cfg = bank["configurations"][BANK_CONFIG]["arms"]
    req_t = cfg["direct_answer"]["messages"][0]["content"]
    cot_t = cfg["zero_shot_cot"]["messages"][0]["content"]
    head, tail = req_t.split("{question}")
    probe = [r for r in items["sets"]["probe"] if r["task"] == "gsm8k"]
    out = []
    for r in probe[:n]:
        if not (r["requested"].startswith(head) and r["requested"].endswith(tail)):
            raise SystemExit(f"probe item {r['item_id']}: stored message does not match bank configuration {BANK_CONFIG}")
        q = r["requested"][len(head): len(r["requested"]) - len(tail)]
        if req_t.replace("{question}", q) != r["requested"] or cot_t.replace("{question}", q) != r["cot"]:
            raise SystemExit(f"probe item {r['item_id']}: bank configuration {BANK_CONFIG} does not reproduce it")
        out.append({"item_id": r["item_id"], "question": q, "requested": r["requested"], "cot": r["cot"]})
    other = {}
    for task in ("math", "bbh"):
        other[task] = [{"item_id": r["item_id"], "requested": r["requested"], "cot": r["cot"]}
                       for r in items["sets"]["probe"] if r["task"] == task]
    return {"rule": f"the first {n} GSM8K items, in stored order, of the probe set of "
                    f"{ITEMS.relative_to(ROOT).as_posix()} (screening items, GSM8K training split; draw rule: "
                    f"{items.get('rule')})",
            "template": {"bank": BANK.relative_to(ROOT).as_posix(), "configuration": BANK_CONFIG,
                         "why": "the bank's first configuration; it is also the seed template that the reasoning "
                                "panel's items record (config/promptsuite.yaml)",
                         "direct_answer": req_t, "zero_shot_cot": cot_t},
            "items_sha256": items.get("content_sha256"), "gsm8k": out, "other_probe": other}


def forced_messages(user_text: str) -> list[dict]:
    return [{"role": "user", "content": user_text}, {"role": "assistant", "content": PREFILL}]


def chat_messages(user_text: str) -> list[dict]:
    return [{"role": "user", "content": user_text}]


def template_path(hf_id: str) -> Path:
    return TOK_DIR / hf_id.split("/")[1] / RENDER[hf_id]["file"]


def tokenizer_path(hf_id: str) -> Path:
    return TOK_DIR / hf_id.split("/")[1] / "tokenizer.json"


def fetch_files(hf_id: str) -> None:
    """Download only the template/renderer, tokenizer_config.json and tokenizer.json (no weights) at the pin."""
    from huggingface_hub import hf_hub_download
    e = RENDER[hf_id]
    for f in (e["file"], "tokenizer_config.json", "tokenizer.json"):
        hf_hub_download(hf_id, f, revision=e["revision"], local_dir=str(TOK_DIR / hf_id.split("/")[1]))


def check_files(hf_id: str) -> str | None:
    """None when the pinned template file and tokenizer are present with the recorded digest, else the problem."""
    p, t = template_path(hf_id), tokenizer_path(hf_id)
    if not p.exists() or not t.exists():
        return f"missing {p.relative_to(ROOT).as_posix() if not p.exists() else t.relative_to(ROOT).as_posix()}"
    got = sha256_file(p)
    if got != RENDER[hf_id]["sha256"]:
        return f"{p.name}: sha256 {got} differs from the pinned {RENDER[hf_id]['sha256']}"
    return None


def render_jinja(template: str, messages: list[dict], kwargs: dict, tokenizer=None) -> str:
    """VllmBackend.render: a final assistant message is a prefill (no new generation prompt, continue that turn)."""
    if tokenizer is None:   # the template alone decides the text; a one-word tokenizer carries apply_chat_template
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")))
    cont = messages[-1]["role"] == "assistant"
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not cont,
                                         **({"continue_final_message": True} if cont else {}),
                                         chat_template=template, **kwargs)


_ENC_CACHE: dict = {}


def load_encoding(path: Path, sha: str):
    """Import DeepSeek's renderer from its pinned file, only after its sha256 matches the recorded digest."""
    if sha256_file(path) != sha:
        raise SystemExit(f"{path}: sha256 does not match the pinned digest")
    if sha not in _ENC_CACHE:
        spec = importlib.util.spec_from_file_location(f"dsv4_encoding_{sha[:8]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ENC_CACHE[sha] = mod
    return _ENC_CACHE[sha]


def render_dsv4(mod, messages: list[dict], thinking: bool) -> str:
    msgs = [dict(m) for m in messages]
    if msgs[-1]["role"] == "assistant":
        msgs[-1]["wo_eos"] = True
    return mod.encode_messages(msgs, thinking_mode="thinking" if thinking else "chat")


def reference_dsv4_chat(user_text: str, prefill: str | None) -> str:
    """Independent statement of DeepSeek-V4's chat-mode format (encoding/README.md, 'Basic Chat'), for the tests."""
    s = "<｜begin▁of▁sentence｜><｜User｜>" + user_text + "<｜Assistant｜></think>"
    return s + (prefill or "")


class Renderer:
    def __init__(self, hf_id: str):
        self.hf_id, self.entry = hf_id, RENDER[hf_id]
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(tokenizer_path(hf_id)))
        if self.entry["kind"] == "jinja":
            from transformers import PreTrainedTokenizerFast
            self.template = template_path(hf_id).read_text(encoding="utf-8")
            self.hf_tok = PreTrainedTokenizerFast(tokenizer_object=self.tok)
        else:
            self.mod = load_encoding(template_path(hf_id), self.entry["sha256"])

    def render(self, messages: list[dict], thinking: bool) -> str:
        if self.entry["kind"] == "jinja":
            return render_jinja(self.template, messages, THINK_ON if thinking else THINK_OFF, self.hf_tok)
        return render_dsv4(self.mod, messages, thinking)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def record(self) -> dict:
        return {"hf_id": self.hf_id, "revision": self.entry["revision"], "renderer": self.entry["kind"],
                "file": self.entry["file"], "sha256": self.entry["sha256"],
                "thinking_off": "thinking_mode 'chat'" + ("" if self.entry["kind"] == "encoding_dsv4" else
                                                          f" (template kwargs {THINK_OFF})"),
                **({"panel_pin": self.entry["panel_pin"]} if "panel_pin" in self.entry else {}),
                **({"note": self.entry["note"]} if "note" in self.entry else {})}


def check_forced_prompt(prompt: str) -> dict:
    """The forced prompt must end with the opened box inside the assistant turn, thinking off."""
    a = prompt.rfind("<｜Assistant｜>")
    tail = prompt[a:] if a >= 0 else ""
    return {"ends_with_prefill": prompt.endswith(PREFILL),
            "prefill_inside_assistant_turn": a >= 0 and tail == "<｜Assistant｜></think>" + PREFILL,
            "thinking_off": "<think>" not in prompt and tail.startswith("<｜Assistant｜></think>"),
            "no_end_of_turn_after_prefill": "<｜end▁of▁sentence｜>" not in tail}


def prompt_tokens_by_benchmark(r: Renderer, qs: dict) -> dict:
    """Mean prompt tokens per request, by arm and benchmark, from the local rendering of the probe items."""
    sets = {"gsm8k": qs["gsm8k"], "math": qs["other_probe"]["math"], "bbh": qs["other_probe"]["bbh"]}
    out = {}
    for b, rows in sets.items():
        f = [r.count(r.render(forced_messages(x["requested"]), False)) for x in rows]
        q = [r.count(r.render(chat_messages(x["requested"]), False)) for x in rows]
        c = [r.count(r.render(chat_messages(x["cot"]), True)) for x in rows]
        out[b] = {"n_items": len(rows), "forced": sum(f) / len(f), "requested": sum(q) / len(q), "cot": sum(c) / len(c)}
    return out


# ----------------------------------------------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------------------------------------------

def requests_per_arm(templates: int, items: int = 300, benchmarks: int = 3) -> int:
    return items * benchmarks * templates


def project_cost(prompt_tokens: dict, output_tokens: dict, price: dict, templates: int, items: int = 300,
                 batch_forced: bool = False) -> dict:
    """USD for one model: prompt_tokens[benchmark][arm] (mean per request), output_tokens[arm] (mean per request,
    the same on every benchmark). Batch price (BATCH_FACTOR) for the requested and chain-of-thought arms; the forced
    arm in batch only if batch_forced, else in real time."""
    per_arm = {}
    for arm in ("forced", "requested", "cot"):
        factor = BATCH_FACTOR if (arm != "forced" or batch_forced) else 1.0
        usd = 0.0
        for b, pt in prompt_tokens.items():
            n = items * templates
            usd += n * (pt[arm] * price["input"] + output_tokens[arm] * price["output"]) / 1e6 * factor
        per_arm[arm] = {"usd": usd, "price_factor": factor, "requests": items * templates * len(prompt_tokens)}
    return {"per_arm": per_arm, "total_usd": sum(v["usd"] for v in per_arm.values()), "templates": templates,
            "requests_per_arm": items * templates * len(prompt_tokens)}


def placeholder_output_tokens() -> dict:
    """Dry run: the reasoning panel's own counts for gpt-oss-120b (results/reasoning_panel/feasibility.json)."""
    d = json.loads(FEASIBILITY.read_text(encoding="utf-8"))["models"]["gpt-oss-120b"]["chosen"]
    return {"forced": d["prefill"]["methods"]["continue_final_message"]["mean_tokens"],
            "requested": d["requested"]["mean_tokens"], "cot": d["reasoning_probe"]["mean_tokens"],
            "source": "PLACEHOLDER: gpt-oss-120b in the reasoning panel's feasibility check "
                      "(results/reasoning_panel/feasibility.json): forced = continue_final_message prefill mean, "
                      "requested = lowest-effort switch mean, cot = reasoning probe mean (effort high, cap 8,192); "
                      "24 probe items over GSM8K, MATH-500 and BIG-Bench Hard"}


# ----------------------------------------------------------------------------------------------------------------
# Live calls
# ----------------------------------------------------------------------------------------------------------------

def usage_of(resp: dict) -> dict:
    u = (resp.get("json") or {}).get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": det.get("reasoning_tokens") if isinstance(det, dict) else None}


def reply_of(resp: dict, endpoint: str) -> dict:
    j = resp.get("json") or {}
    ch = (j.get("choices") or [{}])[0]
    if endpoint == "completions":
        return {"text": ch.get("text"), "reasoning": None, "finish_reason": ch.get("finish_reason")}
    m = ch.get("message") or {}
    return {"text": m.get("content"), "reasoning": m.get("reasoning_content") or m.get("reasoning"),
            "finish_reason": ch.get("finish_reason")}


def write_raw(fh, model: str, item_id: str, arm: str, rec: dict) -> None:
    """One line per reply, written right after its call (reply text, usage, status; no headers)."""
    for r in (rec.get("tries", []) if arm == "forced" else [rec]):
        fh.write(json.dumps({"model": model, "item_id": item_id, "arm": arm, "method": r.get("method"),
                             "status": r.get("status"), "text": r.get("text"), "reasoning": r.get("reasoning"),
                             "finish_reason": r.get("finish_reason"), "usage": r.get("usage"),
                             "error": r.get("error"), "stopped_the_run": r.get("stopped_the_run", False)},
                            ensure_ascii=False) + "\n")
    fh.flush()


def check_abort(u: dict) -> None:
    if (u.get("completion_tokens") or 0) > ABORT_OUTPUT_TOKENS:
        raise Aborted(f"a reply reported {u['completion_tokens']} output tokens (limit {ABORT_OUTPUT_TOKENS})")


def resolve_models(listed: set[str]) -> list[dict]:
    """Which Nebius id to call for each request: the id itself if listed, else a listed alias, else the named
    replacement; each served id once, at most MAX_MODELS."""
    plan, used = [], set()
    for r in REQUESTED:
        served, how = None, None
        if r["name"] in listed:
            served, how = r["name"], "listed under the requested id"
        else:
            for a in r["aliases"]:
                if a in listed:
                    served, how = a, f"requested id not listed; alias {a} is"
                    break
            if served is None and r["replacement"] in listed:
                served, how = r["replacement"], (f"requested id not listed; called the replacement Nebius names in "
                                                 f"its November 2026 deprecation notice")
        if served is None:
            plan.append({**r, "served": None, "how": "neither the requested id nor an alias or replacement is listed"})
        elif served in used:
            plan.append({**r, "served": served, "how": how + "; already called for an earlier request", "dup": True})
        else:
            used.add(served)
            plan.append({**r, "served": served, "how": how})
    return plan


def run_forced(cl: Client, model: str, prompt: str, user_text: str, state: dict) -> dict:
    """Forced call; state carries the method that worked for this model, so a refused method is not retried."""
    tries = []
    methods = [state["method"]] if state.get("method") else ["completions", "chat_partial", "chat_continue"]
    for meth in methods:
        if meth == "completions":
            body = {"model": model, "prompt": prompt, "max_tokens": CAPS["forced"], "temperature": 0}
            path, ep = "/completions", "completions"
        else:
            msgs = forced_messages(user_text)
            body = {"model": model, "messages": msgs, "max_tokens": CAPS["forced"], "temperature": 0,
                    "chat_template_kwargs": THINK_OFF}
            if meth == "chat_partial":
                body["messages"] = [msgs[0], {**msgs[1], "partial": True}]
            else:
                body.update({"continue_final_message": True, "add_generation_prompt": False})
            path, ep = "/chat/completions", "chat"
        resp = cl.post(path, body)
        rec = {"method": meth, "status": resp.get("status"), "seconds": resp.get("seconds")}
        if resp.get("status") == 200:
            u = usage_of(resp)
            rec.update({"usage": u, **reply_of(resp, ep)})
            rec["judgement"] = forced_pass(rec["text"], rec["reasoning"])
            tries.append(rec)
            state["method"] = meth
            try:
                check_abort(u)
            except Aborted as e:
                rec["stopped_the_run"] = True
                e.record = {"tries": tries, "worked": meth}
                raise
            break
        rec["error"] = resp.get("error")
        tries.append(rec)
        if not (resp.get("status") and 400 <= resp["status"] < 500):
            break   # 5xx or transport error: no fallback, the record shows it
    return {"tries": tries, "worked": state.get("method")}


def run_chat(cl: Client, model: str, user_text: str, arm: str, thinking: bool) -> dict:
    body = {"model": model, "messages": chat_messages(user_text), "max_tokens": CAPS[arm], "temperature": 0,
            "chat_template_kwargs": THINK_ON if thinking else THINK_OFF}
    resp = cl.post("/chat/completions", body)
    rec = {"arm": arm, "status": resp.get("status"), "seconds": resp.get("seconds"),
           "chat_template_kwargs": body["chat_template_kwargs"], "max_tokens": CAPS[arm]}
    if resp.get("status") == 200:
        u = usage_of(resp)
        rec.update({"usage": u, **reply_of(resp, "chat")})
        try:
            check_abort(u)
        except Aborted as e:
            rec["stopped_the_run"] = True
            e.record = rec
            raise
    else:
        rec["error"] = resp.get("error")
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------

def strip_links(md: str) -> str:
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", md)


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def write_report(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{STEM}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    live = res["mode"] == "live"
    L = [f"# Nebius prefill test ({'live' if live else 'dry run'}, {res['date']})", "",
         "Written by `scripts/nebius_prefill_test.py`; data in "
         f"`{STEM}.json`" + (", raw replies in `nebius_prefill_raw.jsonl`." if live else "."), ""]
    if not live:
        L += ["**Dry run: no call was sent.** Pass/fail and real token counts come from the live run "
              "(`--live`). Output lengths below are placeholders.", ""]
    L += ["## Model ids on Nebius", "",
          f"Nebius's [November 2026 deprecation notice]({DEPRECATION['url']}) (read {DEPRECATION['read']}): "
          f"\"{DEPRECATION['sentence']}\" Its rows include " + "; ".join(f"`{x}`" for x in DEPRECATION["rows"]) + ". "
          "The public catalog lists neither requested id today; it lists `deepseek-ai/DeepSeek-V4.1-Flash-1017`. "
          "The live run reads `/models` and, for an id that is not listed, calls the listed alias or the named "
          "replacement, and says so below.", ""]
    L += ["| Request | Nebius id called | How | Forced: pass | Endpoint that worked |", "|---|---|---|---|---|"]
    for m in res["models"]:
        if not live:
            L.append(f"| `{m['requested']}` | – | – | not run: dry run | – |")
            continue
        p = m.get("forced_summary") or {}
        L.append(f"| `{m['requested']}` | `{m.get('served') or '–'}` | {m.get('how', '')} | "
                 f"{p.get('verdict', 'not run')} | {p.get('endpoint') or '–'} |")
    L += [""]
    L += ["## Prompts", "",
          f"- Questions: {res['questions']['rule']}. Ids: " + ", ".join(res["questions"]["ids"]) + ".",
          f"- Template: `{res['questions']['template']['bank']}`, configuration "
          f"`{res['questions']['template']['configuration']}` ({res['questions']['template']['why']}).",
          "- Forced arm: user message (direct answer) and a final assistant message `\\boxed{`, rendered as "
          "`VllmBackend.render` does (`continue_final_message=True`, no generation prompt), thinking off; "
          f"max_tokens {CAPS['forced']}, temperature 0, sent to `/completions`.",
          f"- Requested arm: chat endpoint, thinking off, cap {CAPS['requested']}. Chain of thought: chat endpoint, "
          f"thinking {'on' if res['cot_thinking'] else 'off'} (the reasoning panel's reasoning arm), cap {CAPS['cot']}. "
          f"Thinking switch sent as `chat_template_kwargs` {THINK_OFF} / {THINK_ON}.", ""]
    for r in res["renders"]:
        L += [f"### Forced prompt, `{r['hf_id']}` (question {res['questions']['ids'][0]})", ""]
        if r.get("error"):
            L += [f"Not rendered: {r['error']}", ""]
            continue
        rec = r["record"]
        L += [f"Renderer: `{rec['file']}` at revision `{rec['revision'][:12]}` (sha256 `{rec['sha256'][:12]}`), "
              f"{rec['renderer']}; thinking off as {rec['thinking_off']}."
              + (f" Pin: {rec['panel_pin']}." if rec.get("panel_pin") else "")
              + (f" Note: {rec['note']}." if rec.get("note") else ""), "",
              "```", r["forced_prompt"], "```", "",
              f"Checks: " + ", ".join(f"{k} {v}" for k, v in r["checks"].items())
              + f"; {r['forced_prompt_tokens']} prompt tokens (local tokenizer)."
              + (f" Identical to the rendering with `{r['compare']['hf_id']}`: {r['compare']['identical']}."
                 if r.get("compare") else ""), ""]
    L += ["## Tokens per arm", ""]
    L += ["| Model | Arm | Calls | Prompt tokens (mean) | Output tokens (mean) | Source |", "|---|---|---|---|---|---|"]
    for t in res["tokens"]:
        L.append(f"| `{t['model']}` | {t['arm']} | {t['n']} | {t['prompt']:.1f} | {t['output']:.1f} | {t['source']} |")
    L += [""]
    c = res["cost"]
    L += ["## Projected cost", "",
          f"Design: 300 items on each of 3 benchmarks, 3 arms; V4-Flash at {REQUESTED[0]['templates']} templates "
          f"({requests_per_arm(REQUESTED[0]['templates']):,} requests per arm, "
          "`results/reasoning_panel/feasibility.md`), V4.1-Flash at "
          f"{REQUESTED[1]['templates']} templates ({requests_per_arm(REQUESTED[1]['templates']):,} per arm). Prompt "
          "tokens per benchmark from the local rendering of the reasoning panel's probe items (8 per benchmark); "
          f"output tokens per arm as in the table above, the same on every benchmark. Batch = {BATCH_FACTOR:.0%} of "
          "the real-time price; the forced arm is priced in real time, since batch is not known to accept "
          "`/v1/completions`.", ""]
    L += ["| Request | Priced as | Input / output per M | Templates | Forced | Requested | Chain of thought | Total |",
          "|---|---|---|---|---|---|---|---|"]
    for p in c["models"]:
        pa = p["projection"]["per_arm"]
        L.append(f"| `{p['requested']}` | `{p['priced_as']}` | ${p['price']['input']:.2f} / ${p['price']['output']:.2f}"
                 f"{'' if p['price']['official'] else ' (third-party)'} | {p['projection']['templates']} | "
                 f"{fmt_usd(pa['forced']['usd'])} | {fmt_usd(pa['requested']['usd'])} | {fmt_usd(pa['cot']['usd'])} | "
                 f"{fmt_usd(p['projection']['total_usd'])} |")
    L += ["", f"Together: **{fmt_usd(c['total_usd'])}**" + (" (placeholder output lengths)" if not live else "")
          + f". Above ${BUDGET_STOP_USD:.0f}: **{'yes, stop' if c['over_budget'] else 'no'}**.", ""]
    sc = res.get("cost_replacement_scenario") or {}
    if sc.get("models"):
        L += ["If both requests are served by `deepseek-ai/DeepSeek-V4.1-Flash-1017` (the replacement Nebius names), "
              "same design; output lengths measured on that model in a live run, else the placeholders:", "",
              "| Design role | Templates | Forced | Requested | Chain of thought | Total |", "|---|---|---|---|---|---|"]
        for p in sc["models"]:
            pa = p["projection"]["per_arm"]
            L.append(f"| `{p['requested']}` | {p['projection']['templates']} | {fmt_usd(pa['forced']['usd'])} | "
                     f"{fmt_usd(pa['requested']['usd'])} | {fmt_usd(pa['cot']['usd'])} | "
                     f"{fmt_usd(p['projection']['total_usd'])} |")
        L += ["", f"Together: **{fmt_usd(sc['total_usd'])}**. Above ${BUDGET_STOP_USD:.0f}: "
                  f"**{'yes, stop' if sc['over_budget'] else 'no'}**.", ""]
    L += ["Prices: " + "; ".join(f"`{k}` ${v['input']:.2f} / ${v['output']:.2f} per M ({v['date']}, "
                                 f"[source]({v['url']}); {v['note']})" for k, v in res["prices"].items()), ""]
    L += ["## Batch and `/v1/completions`", "",
          f"Answer: {BATCH['answer']}. Official page: [{BATCH['official']['url']}]({BATCH['official']['url']}), "
          f"{BATCH['official']['status']}. Secondary: "
          + "; ".join(f"\"{s['sentence']}\" ([source]({s['url']}))" for s in BATCH["secondary"])
          + f". Consequence: {BATCH['consequence']}. No batch job was submitted.", ""]
    L += ["## Base URL", "", f"`{res['base_url']}`. Source: {BASE_URL_SOURCE}.", ""]
    if live:
        L += ["## Calls", "", f"{res['calls']} POST calls (cap {MAX_CALLS}). " + (res.get("aborted") or "Not aborted."),
              ""]
        for m in res["models"]:
            for q in m.get("questions", []):
                for tr in q["forced"]["tries"]:
                    pt = (tr.get("usage") or {}).get("prompt_tokens")
                    L.append(f"- `{m.get('served')}` {q['item_id']} forced via {tr['method']}: HTTP {tr['status']}"
                             + (f", prompt tokens server {pt} / local {q['forced'].get('local_prompt_tokens')}"
                                if pt is not None else "")
                             + (f", reply {tr.get('text')!r}, {'pass' if tr['judgement']['pass'] else 'fail: ' + '; '.join(tr['judgement']['reasons'])}"
                                if tr.get("judgement") else f", {tr.get('error', '')[:300]!r}"))
                for arm in ("requested", "cot"):
                    a = q.get(arm)
                    if a:
                        L.append(f"- `{m.get('served')}` {q['item_id']} {arm}: HTTP {a['status']}, usage {a.get('usage')}"
                                 + (f", finish {a.get('finish_reason')}" if a.get("finish_reason") else "")
                                 + (f", reasoning returned {bool(a.get('reasoning'))}" if a["status"] == 200 else
                                    f", {a.get('error', '')[:300]!r}"))
        L += [""]
    L += ["## Not done", "", GEMINI_NOTE, ""]
    md = "\n".join(L)
    (out_dir / f"{STEM}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{STEM}_clean.md").write_text(strip_links(md), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

def build_renders(qs: dict, hf_ids: list[str], fetch: bool) -> dict:
    out = {}
    for h in hf_ids:
        if fetch and check_files(h):
            try:
                fetch_files(h)
            except Exception as e:  # noqa: BLE001
                out[h] = {"hf_id": h, "error": f"Hugging Face files not reachable or gated: {type(e).__name__}: {e}"}
                continue
        prob = check_files(h)
        if prob:
            out[h] = {"hf_id": h, "error": prob + " (run with --fetch-templates, or the live run fetches them)"}
            continue
        r = Renderer(h)
        q0 = qs["gsm8k"][0]
        fp = r.render(forced_messages(q0["requested"]), False)
        out[h] = {"hf_id": h, "renderer": r, "record": r.record(), "forced_prompt": fp,
                  "forced_prompts": {q["item_id"]: r.render(forced_messages(q["requested"]), False)
                                     for q in qs["gsm8k"]},
                  "checks": check_forced_prompt(fp), "forced_prompt_tokens": r.count(fp),
                  "prompt_tokens": prompt_tokens_by_benchmark(r, qs)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="send the calls (key from NEBIUS_API_KEY)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cot-thinking", choices=["on", "off"], default="on")
    ap.add_argument("--fetch-templates", action="store_true",
                    help="download missing template/tokenizer files from Hugging Face (no weights); network")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="let a dry run overwrite a live report")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    cot_thinking = a.cot_thinking == "on"

    prev = out_dir / f"{STEM}.json"
    if not a.live and prev.exists() and not a.force:
        try:
            if json.loads(prev.read_text(encoding="utf-8")).get("mode") == "live":
                print("a live report exists; a dry run would overwrite it (use --force)")
                return 2
        except Exception:  # noqa: BLE001
            pass

    qs = load_questions()
    render_ids = list(dict.fromkeys([r["render"] for r in REQUESTED] + [r["compare_render"] for r in REQUESTED
                                                                         if r["compare_render"]]
                                    + ["deepseek-ai/DeepSeek-V4.1-Flash-1017"]))
    renders = build_renders(qs, render_ids, fetch=a.fetch_templates or a.live)

    res = {"mode": "live" if a.live else "dry_run", "date": time.strftime("%Y-%m-%d"), "base_url": a.base_url,
           "base_url_source": BASE_URL_SOURCE, "cot_thinking": cot_thinking, "caps": CAPS, "max_calls": MAX_CALLS,
           "abort_output_tokens": ABORT_OUTPUT_TOKENS, "prices": PRICES, "batch": BATCH, "deprecation": DEPRECATION,
           "gemini": GEMINI_NOTE,
           "questions": {"rule": qs["rule"], "ids": [q["item_id"] for q in qs["gsm8k"]], "template": qs["template"],
                         "items_sha256": qs["items_sha256"]},
           "library_versions": {}, "models": [], "renders": [], "tokens": [], "calls": 0}
    for mod in ("transformers", "tokenizers", "huggingface_hub", "jinja2"):
        try:
            res["library_versions"][mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            res["library_versions"][mod] = f"missing: {e}"

    for r in REQUESTED:
        x = renders.get(r["render"], {})
        rr = {k: v for k, v in x.items() if k not in ("renderer", "forced_prompts")}
        if r["compare_render"] and "forced_prompts" in x and "forced_prompts" in renders.get(r["compare_render"], {}):
            rr["compare"] = {"hf_id": r["compare_render"],
                             "identical": x["forced_prompts"] == renders[r["compare_render"]]["forced_prompts"]}
        res["renders"].append(rr)
    x = renders.get("deepseek-ai/DeepSeek-V4.1-Flash-1017", {})
    res["renders"].append({k: v for k, v in x.items() if k not in ("renderer", "forced_prompts")})
    for x in renders.values():
        if "forced_prompts" in x:
            for iid, p in x["forced_prompts"].items():
                c = check_forced_prompt(p)
                if not all(c.values()):
                    print(f"WARNING {x['hf_id']} {iid}: forced prompt check {c}")

    measured: dict = {}
    if a.live:
        key = os.environ.get("NEBIUS_API_KEY")
        if not key:
            print("NEBIUS_API_KEY is not set")
            return 2
        cl = Client(a.base_url, key)
        raw_path = out_dir / "nebius_prefill_raw.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_path.open("w", encoding="utf-8")
        lm = cl.get("/models")
        if lm.get("status") != 200:
            print(f"GET /models failed: HTTP {lm.get('status')} {lm.get('error')}")
            res["models_list"] = {"status": lm.get("status"), "error": lm.get("error")}
            res["models"] = [{"requested": r["name"], "served": None, "how": "GET /models failed"} for r in REQUESTED]
            listed = set()
        else:
            listed = {m.get("id") for m in (lm["json"].get("data") or [])}
            res["models_list"] = {"status": 200, "deepseek_ids": sorted(i for i in listed if i and "eepSeek" in i)}
        plan = resolve_models(listed) if listed else []
        try:
            for p in plan:
                m = {"requested": p["name"], "served": p["served"], "how": p["how"], "role": p["role"]}
                res["models"].append(m)
                if not p["served"] or p.get("dup"):
                    continue
                rid = RENDER_FOR_SERVED.get(p["served"])
                rx = renders.get(rid, {})
                if "forced_prompts" not in rx:
                    m["how"] += f"; not run: no local rendering ({rx.get('error', 'no template')})"
                    continue
                m["render_hf_id"] = rid
                state, qrecs = {}, []
                m["questions"] = qrecs
                for q in qs["gsm8k"]:
                    qr = {"item_id": q["item_id"]}
                    qrecs.append(qr)
                    for arm in ("forced", "requested", "cot"):
                        try:
                            if arm == "forced":
                                qr[arm] = run_forced(cl, p["served"], rx["forced_prompts"][q["item_id"]],
                                                     q["requested"], state)
                            else:
                                qr[arm] = run_chat(cl, p["served"], q[arm], arm,
                                                   cot_thinking if arm == "cot" else False)
                        except Aborted as e:
                            if e.record is not None:   # the reply that tripped the stop is kept and counted
                                qr[arm] = e.record
                            raise
                        finally:   # each reply is saved right after its call, also when it stops the run
                            if arm in qr:
                                if arm == "forced":
                                    qr[arm]["local_prompt_tokens"] = rx["renderer"].count(
                                        rx["forced_prompts"][q["item_id"]])
                                write_raw(raw, p["served"], q["item_id"], arm, qr[arm])
        except Aborted as e:
            res["aborted"] = f"Aborted: {e}."
            print(res["aborted"])
        finally:
            raw.close()
        for p in plan[len(res["models"]):]:   # requests the run did not reach before it stopped
            res["models"].append({"requested": p["name"], "served": p["served"], "role": p["role"],
                                  "how": p["how"] + "; not run: the run stopped earlier"})
        res["calls"] = cl.calls
        for m in res["models"]:
            qrs = m.get("questions") or []
            for q in qrs:
                q.setdefault("forced", {"tries": [], "worked": None})
            fin = [q["forced"]["tries"][-1] for q in qrs if q["forced"]["tries"]]
            judged = [t for t in fin if t.get("judgement")]
            if qrs:
                ok = sum(t["judgement"]["pass"] for t in judged)
                meth = next((q["forced"]["worked"] for q in qrs if q["forced"].get("worked")), None)
                m["forced_summary"] = {"verdict": f"{'pass' if judged and ok == len(qrs) else 'fail'} ({ok}/{len(qrs)})",
                                       "endpoint": {"completions": "/v1/completions",
                                                    "chat_partial": "/v1/chat/completions, partial: true",
                                                    "chat_continue": "/v1/chat/completions, continue_final_message"
                                                    }.get(meth)}
                arms = {}
                for q in qrs:
                    f = q["forced"]["tries"][-1] if q["forced"]["tries"] else {}
                    for arm, rec in (("forced", f), ("requested", q.get("requested") or {}), ("cot", q.get("cot") or {})):
                        u = rec.get("usage")
                        if u and u.get("completion_tokens") is not None:
                            arms.setdefault(arm, []).append({**u, "stopped_the_run": rec.get("stopped_the_run", False)})
                measured[m["served"]] = arms
                for arm, us in arms.items():
                    res["tokens"].append({"model": m["served"], "arm": arm, "n": len(us),
                                          "prompt": sum(u["prompt_tokens"] or 0 for u in us) / len(us),
                                          "output": sum(u["completion_tokens"] for u in us) / len(us),
                                          "source": "measured (usage field)" + (
                                              "; includes the reply that stopped the run"
                                              if any(u["stopped_the_run"] for u in us) else "")})
    else:
        res["models"] = [{"requested": r["name"], "served": None, "how": "not run: dry run", "role": r["role"]}
                         for r in REQUESTED]

    # Local prompt tokens on GSM8K beside the measured ones; placeholders in a dry run.
    ph = placeholder_output_tokens()
    if not a.live:
        for h, x in renders.items():
            if "prompt_tokens" in x:
                for arm in ("forced", "requested", "cot"):
                    res["tokens"].append({"model": h, "arm": arm, "n": 0, "prompt": x["prompt_tokens"]["gsm8k"][arm],
                                          "output": ph[arm], "source": "prompt: local rendering (GSM8K probe items); "
                                                                       "output: PLACEHOLDER (gpt-oss-120b, local panel)"})
    res["placeholder_output_tokens"] = ph

    cost_rows = []
    for r, m in zip(REQUESTED, res["models"]):
        priced = m.get("served") or r["name"]
        rid = RENDER_FOR_SERVED.get(priced, r["render"])
        x = renders.get(rid, {})
        if "prompt_tokens" not in x:
            continue
        meas = measured.get(m.get("served")) or {}
        out_tok, src = {}, {}
        for arm in ("forced", "requested", "cot"):
            if meas.get(arm):
                out_tok[arm] = sum(u["completion_tokens"] for u in meas[arm]) / len(meas[arm])
                src[arm] = "measured"
            else:
                out_tok[arm] = ph[arm]
                src[arm] = "placeholder"
        price = PRICES[priced]
        cost_rows.append({"requested": r["name"], "priced_as": priced, "price": price, "render_hf_id": rid,
                          "output_tokens": out_tok, "output_source": src,
                          "prompt_tokens": x["prompt_tokens"],
                          "projection": project_cost(x["prompt_tokens"], out_tok, price, r["templates"]),
                          "projection_forced_in_batch": project_cost(x["prompt_tokens"], out_tok, price,
                                                                     r["templates"], batch_forced=True)})
    total = sum(c["projection"]["total_usd"] for c in cost_rows)
    # Scenario: both requests served by the replacement Nebius names (V4.1-Flash-1017), same design.
    rep, scen = "deepseek-ai/DeepSeek-V4.1-Flash-1017", []
    xr = renders.get(rep, {})
    if "prompt_tokens" in xr:
        meas = measured.get(rep) or {}
        out_tok = {arm: (sum(u["completion_tokens"] for u in meas[arm]) / len(meas[arm]) if meas.get(arm) else ph[arm])
                   for arm in ("forced", "requested", "cot")}
        for r in REQUESTED:
            scen.append({"requested": r["name"], "priced_as": rep, "price": PRICES[rep], "output_tokens": out_tok,
                         "projection": project_cost(xr["prompt_tokens"], out_tok, PRICES[rep], r["templates"])})
    scen_total = sum(c["projection"]["total_usd"] for c in scen)
    res["cost_replacement_scenario"] = {"models": scen, "total_usd": scen_total,
                                        "over_budget": scen_total > BUDGET_STOP_USD}
    res["cost"] = {"models": cost_rows, "total_usd": total, "over_budget": total > BUDGET_STOP_USD,
                   "budget_usd": BUDGET_STOP_USD, "batch_factor": BATCH_FACTOR,
                   "output_tokens_label": "measured where the live run has them, else PLACEHOLDER"}
    write_report(res, out_dir)

    print(f"mode {res['mode']}; report {(out_dir / (STEM + '.md')).as_posix()}")
    for x in res["renders"]:
        if x.get("forced_prompt"):
            print(f"{x['hf_id']}: forced prompt tail {x['forced_prompt'][-60:]!r}; checks {x['checks']}")
        else:
            print(f"{x.get('hf_id')}: {x.get('error')}")
    print(f"projected cost {fmt_usd(total)} (over ${BUDGET_STOP_USD:.0f}: {res['cost']['over_budget']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
