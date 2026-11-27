"""Tests for scripts/nebius_prefill_test.py: rendering, the pass rule, redaction, the call cap, the cost arithmetic,
and that a dry run sends nothing. No network: urllib is replaced wherever a call could happen."""
from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_prefill_test as npt  # noqa: E402

V4 = "deepseek-ai/DeepSeek-V4-Flash-0731"
V41 = "deepseek-ai/DeepSeek-V4.1-Flash"
V41R = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def have(h):
    return npt.check_files(h) is None


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    import socket
    monkeypatch.setattr(socket, "create_connection", boom)
    return boom


# ---------------------------------------------------------------- pass rule

@pytest.mark.parametrize("text", ["18}", "18}<｜end▁of▁sentence｜>", " 1,080 }", "\\frac{3}{4}}", "$42$}"])
def test_forced_pass_accepts_a_closed_short_answer(text):
    assert npt.forced_pass(text)["pass"], npt.forced_pass(text)


@pytest.mark.parametrize("text,why", [
    ("18", "not closed"),
    ("}", "nothing before"),
    ("Let me work this out step by}", "words"),
    ("90\n= 90}", "line break"),
    ("<think>so 18</think>18}", "restart"),
    ("18}\n\\boxed{18}", "restart"),
    ("18}<｜User｜>next", "restart"),
    (None, "not closed"),
])
def test_forced_pass_rejects(text, why):
    j = npt.forced_pass(text)
    assert not j["pass"] and any(why in r for r in j["reasons"]), j


def test_forced_pass_rejects_separate_reasoning():
    j = npt.forced_pass("18}", reasoning="First, 9 flights...")
    assert not j["pass"] and "reasoning text returned" in j["reasons"][0]
    assert npt.forced_pass("18}", reasoning="  ")["pass"]


# ---------------------------------------------------------------- redaction and client

def test_redact_removes_key_and_bearer():
    key = "nb-SECRET-123456"
    s = npt.redact(f"bad key {key}; header Authorization: Bearer {key}; {{\"api_key\": \"{key}\"}}", key)
    assert key not in s and "REDACTED" in s
    assert "abc.def" not in npt.redact("Authorization: Bearer abc.def")
    assert "xyz" not in npt.redact('{"api_key": "xyz"}')
    assert npt.redact(None) == ""


def test_client_redacts_error_body_and_caps_calls(monkeypatch, capsys):
    key = "nb-SECRET-abcdef"
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(f"invalid key {key}".encode()))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    cl = npt.Client("https://example.invalid/v1", key, max_calls=2)
    r = cl.post("/completions", {"model": "m", "prompt": "p"})
    assert r["status"] == 401 and key not in r["error"] and "REDACTED" in r["error"]
    assert seen[0].full_url == "https://example.invalid/v1/completions"
    cl.post("/completions", {})
    with pytest.raises(npt.Aborted):
        cl.post("/completions", {})
    assert len(seen) == 2
    assert key not in capsys.readouterr().out


def test_client_redacts_transport_error(monkeypatch):
    key = "nb-SECRET-xyz"

    def fake(req, timeout=None):
        raise OSError(f"proxy said Bearer {key}")
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    r = npt.Client("https://example.invalid/v1", key).get("/models")
    assert r["status"] is None and key not in r["error"]


class FakeClient:
    def __init__(self, replies):
        self.replies, self.sent, self.calls = list(replies), [], 0

    def post(self, path, body):
        self.calls += 1
        self.sent.append((path, body))
        return self.replies.pop(0)


def ok_completion(text, out=3):
    return {"status": 200, "seconds": 0.1,
            "json": {"choices": [{"text": text, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 78, "completion_tokens": out}}}


def ok_chat(text, out=3, reasoning=None):
    return {"status": 200, "seconds": 0.1,
            "json": {"choices": [{"message": {"content": text, "reasoning_content": reasoning}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 78, "completion_tokens": out}}}


def test_forced_falls_back_to_chat_partial_and_remembers():
    cl = FakeClient([{"status": 404, "error": "model does not support /v1/completions"}, ok_chat("18}"),
                     ok_chat("7}")])
    state = {}
    r = npt.run_forced(cl, "m", "<prompt>\\boxed{", "question", state)
    assert r["worked"] == "chat_partial" and [t["method"] for t in r["tries"]] == ["completions", "chat_partial"]
    assert r["tries"][-1]["judgement"]["pass"]
    path, body = cl.sent[1]
    assert path == "/chat/completions" and body["messages"][-1] == {"role": "assistant", "content": "\\boxed{",
                                                                    "partial": True}
    assert body["max_tokens"] == 16 and body["temperature"] == 0 and body["chat_template_kwargs"] == npt.THINK_OFF
    r2 = npt.run_forced(cl, "m", "<prompt>\\boxed{", "question", state)
    assert [t["method"] for t in r2["tries"]] == ["chat_partial"] and cl.calls == 3


def test_forced_completions_body():
    cl = FakeClient([ok_completion("18}")])
    r = npt.run_forced(cl, "m", "P\\boxed{", "q", {})
    assert cl.sent[0] == ("/completions", {"model": "m", "prompt": "P\\boxed{", "max_tokens": 16, "temperature": 0})
    assert r["worked"] == "completions"


def test_forced_no_fallback_on_server_error():
    cl = FakeClient([{"status": 503, "error": "busy"}])
    r = npt.run_forced(cl, "m", "P", "q", {})
    assert r["worked"] is None and cl.calls == 1


def test_abort_on_long_reply():
    cl = FakeClient([ok_chat("long", out=npt.ABORT_OUTPUT_TOKENS + 1)])
    with pytest.raises(npt.Aborted):
        npt.run_chat(cl, "m", "q", "cot", True)
    assert cl.sent[0][1]["max_tokens"] == 8192 and cl.sent[0][1]["chat_template_kwargs"] == npt.THINK_ON


def test_hard_cap_size():
    assert npt.MAX_CALLS == 30 and npt.CAPS == {"forced": 16, "requested": 1024, "cot": 8192}


# ---------------------------------------------------------------- model resolution

def test_resolve_models():
    plan = npt.resolve_models({V41R, "x/y"})
    assert [p["served"] for p in plan] == [V41R, V41R] and plan[1].get("dup")
    plan = npt.resolve_models({"deepseek-ai/DeepSeek-V4-Flash-0731", V41})
    assert [p["served"] for p in plan] == ["deepseek-ai/DeepSeek-V4-Flash-0731", V41] and "alias" in plan[0]["how"]
    plan = npt.resolve_models({"deepseek-ai/DeepSeek-V4-Flash", V41})
    assert [p["served"] for p in plan] == ["deepseek-ai/DeepSeek-V4-Flash", V41]
    assert [p["served"] for p in npt.resolve_models({"other"})] == [None, None]


# ---------------------------------------------------------------- cost

def test_cost_arithmetic():
    pt = {"gsm8k": {"forced": 100, "requested": 100, "cot": 100}}
    out = {"forced": 4, "requested": 200, "cot": 1000}
    c = npt.project_cost(pt, out, {"input": 1.0, "output": 2.0}, templates=1, items=10)
    pa = c["per_arm"]
    assert pa["forced"]["usd"] == pytest.approx(10 * (100 + 8) / 1e6)            # real time
    assert pa["requested"]["usd"] == pytest.approx(10 * (100 + 400) / 1e6 * 0.5)
    assert pa["cot"]["usd"] == pytest.approx(10 * (100 + 2000) / 1e6 * 0.5)
    assert c["total_usd"] == pytest.approx(sum(v["usd"] for v in pa.values()))
    cb = npt.project_cost(pt, out, {"input": 1.0, "output": 2.0}, templates=1, items=10, batch_forced=True)
    assert cb["per_arm"]["forced"]["usd"] == pytest.approx(pa["forced"]["usd"] / 2)


def test_cost_scales_with_design():
    pt = {b: {"forced": 90, "requested": 90, "cot": 90} for b in ("gsm8k", "math", "bbh")}
    out = {"forced": 3, "requested": 150, "cot": 1200}
    full = npt.project_cost(pt, out, {"input": 0.14, "output": 0.28}, templates=16)
    four = npt.project_cost(pt, out, {"input": 0.14, "output": 0.28}, templates=4)
    assert full["requests_per_arm"] == 14400 == npt.requests_per_arm(16) and four["requests_per_arm"] == 3600
    assert full["total_usd"] == pytest.approx(4 * four["total_usd"])
    want_cot = 14400 * (90 * 0.14 + 1200 * 0.28) / 1e6 * 0.5
    assert full["per_arm"]["cot"]["usd"] == pytest.approx(want_cot)


# ---------------------------------------------------------------- prompts and rendering

def test_render_jinja_continues_the_final_message():
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    tpl = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}</s>{% endfor %}"
           "{% if add_generation_prompt %}<assistant>{% endif %}")
    s = npt.render_jinja(tpl, npt.forced_messages("Q?"), {})
    assert s == "<user>Q?</s><assistant>\\boxed{"
    assert npt.render_jinja(tpl, npt.chat_messages("Q?"), {}) == "<user>Q?</s><assistant>"


def test_check_forced_prompt():
    good = npt.reference_dsv4_chat("Q", npt.PREFILL)
    assert all(npt.check_forced_prompt(good).values())
    bad = "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜><think>\\boxed{"
    c = npt.check_forced_prompt(bad)
    assert not c["thinking_off"] and not c["prefill_inside_assistant_turn"]
    closed = good + "<｜end▁of▁sentence｜>"
    assert not npt.check_forced_prompt(closed)["ends_with_prefill"]


@pytest.mark.parametrize("hf_id", [V4, "deepseek-ai/DeepSeek-V4-Flash", V41, V41R])
def test_real_templates_render_the_forced_prompt(hf_id):
    if not have(hf_id):
        pytest.skip(f"template files for {hf_id} not downloaded (data/ is git-ignored)")
    if npt.RENDER[hf_id]["kind"] == "jinja":
        pytest.importorskip("transformers")
    r = npt.Renderer(hf_id)
    q = npt.load_questions()["gsm8k"][0]
    forced = r.render(npt.forced_messages(q["requested"]), thinking=False)
    assert forced == npt.reference_dsv4_chat(q["requested"], npt.PREFILL)
    assert all(npt.check_forced_prompt(forced).values())
    assert r.render(npt.chat_messages(q["requested"]), thinking=False) == npt.reference_dsv4_chat(q["requested"], None)
    assert r.render(npt.chat_messages(q["cot"]), thinking=True).endswith("<｜Assistant｜><think>")
    assert r.count(forced) > 50


def test_encoding_refuses_a_changed_file(tmp_path):
    p = tmp_path / "encoding_dsv4.py"
    p.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        npt.load_encoding(p, "0" * 64)


# ---------------------------------------------------------------- dry run end to end

def test_dry_run_sends_nothing_and_never_reads_the_key(tmp_path, monkeypatch, no_network, capsys):
    if not all(have(h) for h in (V4, "deepseek-ai/DeepSeek-V4-Flash", V41, V41R)):
        pytest.skip("template files not downloaded")
    pytest.importorskip("transformers")
    key = "nb-SECRET-should-never-appear"
    monkeypatch.setenv("NEBIUS_API_KEY", key)
    read = []
    real_get = npt.os.environ.get

    def spy(k, *a):
        read.append(k)
        return real_get(k, *a)
    monkeypatch.setattr(npt.os.environ, "get", spy)
    assert npt.main(["--out-dir", str(tmp_path)]) == 0
    assert "NEBIUS_API_KEY" not in read
    res = json.loads((tmp_path / "nebius_prefill_test.json").read_text(encoding="utf-8"))
    assert res["mode"] == "dry_run" and res["calls"] == 0
    assert all(m["how"] == "not run: dry run" for m in res["models"])
    assert all(r["checks"] and all(r["checks"].values()) for r in res["renders"])
    md = (tmp_path / "nebius_prefill_test.md").read_text(encoding="utf-8")
    clean = (tmp_path / "nebius_prefill_test_clean.md").read_text(encoding="utf-8")
    assert "not run: dry run" in md and "PLACEHOLDER" in json.dumps(res["tokens"])
    assert "](http" not in clean
    for f in tmp_path.iterdir():
        assert key not in f.read_text(encoding="utf-8")
    assert key not in capsys.readouterr().out
    assert not (tmp_path / "nebius_prefill_raw.jsonl").exists()


def test_dry_run_does_not_overwrite_a_live_report(tmp_path):
    (tmp_path / "nebius_prefill_test.json").write_text(json.dumps({"mode": "live"}), encoding="utf-8")
    assert npt.main(["--out-dir", str(tmp_path)]) == 2


class _Resp:
    def __init__(self, payload):
        self.status, self._b = 200, json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_live_path_keeps_the_reply_that_stops_the_run(tmp_path, monkeypatch):
    """Fake server: /models lists only the replacement; the first chain-of-thought reply is too long. The run stops,
    and that reply is still in the raw file, the record and the token means."""
    if not have(V41R):
        pytest.skip("template files not downloaded")
    pytest.importorskip("transformers")
    key = "nb-SECRET-live-test"
    monkeypatch.setenv("NEBIUS_API_KEY", key)
    seen = []

    def fake(req, timeout=None):
        seen.append(req.full_url)
        assert req.get_header("Authorization") == "Bearer " + key
        if req.full_url.endswith("/models"):
            return _Resp({"data": [{"id": V41R}, {"id": "other/model"}]})
        body = json.loads(req.data.decode("utf-8"))
        if req.full_url.endswith("/completions") and "/chat/" not in req.full_url:
            return _Resp({"choices": [{"text": "18}", "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 78, "completion_tokens": 3}})
        n = npt.ABORT_OUTPUT_TOKENS + 500 if body["max_tokens"] == npt.CAPS["cot"] else 40
        return _Resp({"choices": [{"message": {"content": "x \boxed{18}"}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 80, "completion_tokens": n}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert npt.main(["--live", "--out-dir", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "nebius_prefill_test.json").read_text(encoding="utf-8"))
    assert res["mode"] == "live" and res["calls"] == 3 and "Aborted" in res["aborted"]
    assert [m["served"] for m in res["models"]] == [V41R, V41R]
    q = res["models"][0]["questions"][0]
    assert q["cot"]["stopped_the_run"] and q["cot"]["usage"]["completion_tokens"] == npt.ABORT_OUTPUT_TOKENS + 500
    assert q["forced"]["local_prompt_tokens"] == 78 and q["forced"]["tries"][0]["judgement"]["pass"]
    raw = [json.loads(x) for x in (tmp_path / "nebius_prefill_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["arm"] for r in raw] == ["forced", "requested", "cot"] and raw[-1]["stopped_the_run"]
    cot = [t for t in res["tokens"] if t["arm"] == "cot"][0]
    assert cot["output"] == npt.ABORT_OUTPUT_TOKENS + 500 and "stopped the run" in cot["source"]
    for f in tmp_path.iterdir():
        assert key not in f.read_text(encoding="utf-8")
