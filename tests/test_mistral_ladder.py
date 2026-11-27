"""Tests for scripts/mistral_ladder.py and scripts/nebius_batch.py: batch lines, the pilot draw, the forced pass rule,
output parsing, refusal handling, the spend limit, and that the key never reaches a file or a printed line. No network:
urlopen raises, and the HTTP layer is a fake."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mistral_ladder as M  # noqa: E402

SECRET = "ml-test-SECRET-key-0123456789abcdef"
HAVE_REQUESTS = all((ROOT / "data" / "requests" / "ladder" / t / f"{M.REQ_KEY}.parquet").exists() for t in M.L.TASKS)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def toy_design() -> pd.DataFrame:
    rows = []
    for t in ("gsm8k", "hotpot"):
        for c in ("ps-a", "ps-b"):
            for i in range(40):
                for arm in ("direct_answer", "forced_direct", "zero_shot_cot"):
                    rid = f"{c}|{arm}|{i}"
                    rows.append({"request_id": rid, "arm": arm, "config_id": c, "item_id": str(i),
                                 "messages_json": json.dumps([{"role": "user", "content": f"q{i}"}]), "task": t,
                                 "key": f"{t}|{rid}", "cap": 16 if arm == "forced_direct" else 7000,
                                 "panel_cap": 16 if arm == "forced_direct" else 1024})
        for i in range(40):   # five-shot requested rows: prefixed configuration id
            rid = f"{t}/ps-a|direct_answer|{i}|fsr"
            rows.append({"request_id": rid, "arm": "fewshot_requested", "config_id": f"{t}/ps-a", "item_id": str(i),
                         "messages_json": json.dumps([{"role": "user", "content": f"q{i}"}]), "task": t,
                         "key": f"{t}|{rid}", "cap": 7000, "panel_cap": 1024})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Mistral: design

def test_pilot_draw_rule_and_five_shot_alignment():
    df = toy_design()
    p = M.pilot_rows(df)
    # ceil(5% of 40) = 2 items per benchmark and configuration, every arm of each
    assert p.groupby(["task", "cfg"])["item_id"].nunique().eq(2).all()
    for t in ("gsm8k", "hotpot"):
        x = p[p["task"] == t]
        fsr = set(x[x["arm"] == "fewshot_requested"]["item_id"])
        za = set(x[(x["arm"] == "direct_answer") & (x["config_id"] == "ps-a")]["item_id"])
        assert fsr == za and len(fsr) == 2
    # same salt and order key as the Nebius pilot
    import nebius_panel as nps
    assert M.SALT == nps.SALT
    assert M.unit_hash("gsm8k", "ps-a", "3") == nps.unit_hash("gsm8k", "ps-a", "3")


def test_batch_line_forced_prefix_and_system():
    row = {"key": "gsm8k|c|direct_answer|1|fd", "arm": "forced_direct", "cap": 16,
           "messages_json": json.dumps([{"role": "user", "content": "2+2?"}])}
    j = json.loads(M.batch_line(row, "SYS"))
    assert j["custom_id"] == row["key"]
    b = j["body"]
    assert b["messages"][0] == {"role": "system", "content": "SYS"}
    assert b["messages"][-1] == {"role": "assistant", "content": "\\boxed{", "prefix": True}
    assert b["max_tokens"] == 16 and b["temperature"] == 0.0 and b["top_p"] == 1.0 and b["random_seed"] == 0
    assert "model" not in b    # the model is given at job creation
    row2 = dict(row, arm="zero_shot_cot", cap=7000)
    b2 = json.loads(M.batch_line(row2, None))["body"]
    assert b2["messages"] == [{"role": "user", "content": "2+2?"}] and b2["max_tokens"] == 7000


def test_system_message_extracted_and_checked():
    tpl = "{%- set default_system_message = 'You are X.\nIt\\'s {today}.' %}\n{{ bos_token }}"
    assert M.system_message(tpl, "You are X.\nIt's {today}.\n") == "You are X.\nIt's {today}."
    with pytest.raises(SystemExit):
        M.system_message(tpl, "something else")


def test_split_files_respects_bytes():
    lines = ["x" * 99] * 10
    parts = M.split_files(lines, max_bytes=250)
    assert sum(len(p) for p in parts) == 10 and all(len(p) <= 2 for p in parts)


@pytest.mark.skipif(not HAVE_REQUESTS, reason="ladder request files not present")
def test_real_design_counts_and_caps():
    df = M.load_design()
    assert len(df) == 556708
    assert set(df[df["arm"].isin(M.FORCED)]["cap"]) == {16, 64}
    assert set(df[df["arm"] == "zero_shot_cot"]["cap"]) == {7000}
    assert set(df[(df["task"] == "hotpot") & (df["arm"] == "zero_shot_cot")]["panel_cap"]) == {1900}
    p = M.pilot_rows(df)
    assert len(p) == 27928


# ------------------------------------------------------------------------------------------------ Mistral: money

def test_cost_batch_price():
    assert M.cost(1_000_000, 0) == pytest.approx(0.25)
    assert M.cost(0, 1_000_000) == pytest.approx(0.75)


def test_projection_uses_expected_output_capped(monkeypatch):
    monkeypatch.setattr(M.L, "arm_tokens", lambda key, t, a, proxy=None: (20.0 if a == "forced_direct" else 300.0, 0))
    rows = pd.DataFrame({"task": ["gsm8k"] * 2, "arm": ["forced_direct", "zero_shot_cot"], "cap": [16, 7000]})
    pj = M.projection(rows, [1000, 1000])
    assert pj["expected_output_tokens"] == 16 + 300
    assert pj["usd"] == pytest.approx(M.cost(2000, 316))
    assert pj["usd_all_at_cap"] == pytest.approx(M.cost(1000, 16) + M.cost(1000, 7000))


def test_limit_above_15_refused(capsys):
    assert M.main(["pilot-submit", "--limit-usd", "20"]) == 2


# ------------------------------------------------------------------------------------------------ Mistral: outputs

def out_line(cid, content, status=200, pt=100, ct=5, finish="stop"):
    return {"id": "x", "custom_id": cid, "response": {"status_code": status, "body": {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}}, "error": None}


def test_parse_output_and_prefix_echo():
    meta = {"key": "k|fd", "task": "gsm8k", "request_id": "r", "arm": "forced_direct", "config_id": "c",
            "item_id": "1", "cap": 16, "panel_cap": 16}
    r = M.raw_record(meta, out_line("k|fd", "\\boxed{42}"), "job1")
    assert r["text"] == "42}" and r["prefix_echoed"] and r["status"] == 200
    assert r["cost_usd"] == pytest.approx(M.cost(100, 5))
    r2 = M.raw_record(meta, out_line("k|fd", "42}"), "job1")
    assert r2["text"] == "42}" and not r2["prefix_echoed"]
    r3 = M.raw_record(dict(meta, arm="direct_answer"), out_line("k", "\\boxed{42}"), "job1")
    assert r3["text"] == "\\boxed{42}"
    bad = M.raw_record(meta, {"custom_id": "k|fd", "response": {"status_code": 400, "body": {"message": "no"}},
                              "error": None}, "job1")
    assert bad["status"] == 400 and bad["text"] is None and bad["cost_usd"] == 0.0
    chunks = M.parse_output_line(out_line("k", [{"type": "text", "text": "4"}, {"type": "text", "text": "2}"}]))
    assert chunks["text"] == "42}"


def test_forced_pass_rule():
    assert M.forced_pass("42}")["pass"]
    assert M.forced_pass("Paris} is the answer")["pass"]
    assert not M.forced_pass("42")["pass"]
    assert not M.forced_pass("}")["pass"]
    assert not M.forced_pass("\\frac{1}{2}")["pass"]        # box still open after the fraction
    assert M.forced_pass("\\frac{1}{2}}")["pass"]
    assert not M.forced_pass("42} \\boxed{43}")["pass"]
    assert not M.forced_pass("Let me think step by step\nabout this}")["pass"]


def test_outputs_frame_cut_rules():
    class C:
        def cut(self, text, n):
            return text[:n]
    raw = [{"task": "gsm8k", "status": 200, "text": "42} extra", "arm": "forced_direct", "finish_reason": "length",
            "usage": {"completion_tokens": 16, "prompt_tokens": 9}, "panel_cap": 16, "cap": 16, "request_id": "a",
            "key": "gsm8k|a", "config_id": "c", "item_id": "1"},
           {"task": "gsm8k", "status": 200, "text": "x" * 50, "arm": "zero_shot_cot", "finish_reason": "stop",
            "usage": {"completion_tokens": 2000, "prompt_tokens": 9}, "panel_cap": 10, "cap": 7000,
            "request_id": "b", "key": "gsm8k|b", "config_id": "c", "item_id": "1"},
           {"task": "gsm8k", "status": 200, "text": "short", "arm": "direct_answer", "finish_reason": "stop",
            "usage": {"completion_tokens": 3, "prompt_tokens": 9}, "panel_cap": 1024, "cap": 7000,
            "request_id": "c", "key": "gsm8k|c", "config_id": "c", "item_id": "1"}]
    o = M.outputs_frame(raw, "gsm8k", C()).set_index("request_id")
    assert o.loc["a", "output_text"] == "42}" and o.loc["a", "finish_reason"] == "stop"
    assert not o.loc["a", "cut_at_panel_cap"]
    assert o.loc["b", "cut_at_panel_cap"] and o.loc["b", "output_text_cut"] == "x" * 10
    assert o.loc["b", "finish_reason_cut"] == "length" and o.loc["b", "output_text"] == "x" * 50
    assert not o.loc["c", "cut_at_panel_cap"] and o.loc["c", "output_text_cut"] == "short"


# ------------------------------------------------------------------------------------------------ Mistral: HTTP

class FakeHttp:
    """Mistral's files and batch endpoints; creation answers `create_status`."""

    def __init__(self, create_status=200, job_status="SUCCESS", outputs=None):
        self._key = SECRET
        self.create_status, self.job_status, self.outputs = create_status, job_status, outputs or []
        self.calls = []

    def upload(self, data, filename):
        self.calls.append(("upload", filename, data))
        return {"status": 200, "json": {"id": "file-in"}}

    def post_json(self, path, body):
        self.calls.append(("post", path, body))
        if path == "/batch/jobs":
            if self.create_status != 200:
                return {"status": self.create_status, "json": {"detail": f"no access; Bearer {SECRET}"}}
            return {"status": 200, "json": {"id": "job-1", "status": "QUEUED"}}
        return {"status": 200, "json": {}}

    def get(self, path):
        self.calls.append(("get", path))
        return {"status": 200, "json": {"status": self.job_status, "total_requests": len(self.outputs),
                                        "succeeded_requests": len(self.outputs), "failed_requests": 0,
                                        "output_file": "file-out", "error_file": None}}

    def content(self, fid):
        self.calls.append(("content", fid))
        return {"status": 200, "json": None, "text": "\n".join(json.dumps(o) for o in self.outputs)}


def test_create_job_refused_is_redacted():
    h = FakeHttp(create_status=402)
    rec = M.create_job(h, ['{"custom_id": "a", "body": {}}'], "t", {"run": "x"}, SECRET)
    assert "job_id" not in rec and rec["create_http"] == 402
    assert SECRET not in json.dumps(rec) and "REDACTED" in rec["error"]
    post = [c for c in h.calls if c[0] == "post"][0]
    assert post[2]["model"] == M.MODEL and post[2]["endpoint"] == "/v1/chat/completions"
    assert post[2]["input_files"] == ["file-in"]


def test_create_job_ok():
    rec = M.create_job(FakeHttp(), ['{"custom_id": "a", "body": {}}'], "t", {"run": "x"})
    assert rec["job_id"] == "job-1" and rec["lines"] == 1


def test_wait_job_polls_at_most_every_five_minutes(monkeypatch):
    sleeps, clock = [], [1000.0]
    monkeypatch.setattr(M.time, "time", lambda: clock[0])

    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s
    h = FakeHttp(job_status="RUNNING")
    M.wait_job(h, "job-1", wait_min=11, sleep=fake_sleep)
    assert sleeps and all(s >= 300 for s in sleeps) and len(sleeps) <= 2


def test_redact_bearer():
    s = M.redact('{"Authorization": "Bearer abc123"} and ' + SECRET, SECRET)
    assert "abc123" not in s and SECRET not in s


def test_forced_verdict_and_system_check():
    base = {"task": "gsm8k", "arm": "forced_direct", "item_id": "1", "status": 200, "finish_reason": "stop",
            "reasons": [], "cost_usd": 0.0, "n_prompt_tokens_local": 600}
    res = [dict(base, key="a", system=True, text="42}", pass_=True, usage={"prompt_tokens": 601}),
           dict(base, key="b", system=True, text="7}", usage={"prompt_tokens": 650}),
           dict(base, key="a", system=False, text="42}", usage={"prompt_tokens": 80}, n_prompt_tokens_local=79)]
    for r in res:
        r["pass"] = True
        r["prefix_echoed"] = False
    v = M.forced_verdict(res)
    assert v["requests"] == 2 and v["faithful"] and v["same_text_without_system"]
    assert v["system_check"]["without_system_server"] == 80 and v["system_check"]["with_system_server"] == 601


# ------------------------------------------------------------------------------------------------ Nebius batch
