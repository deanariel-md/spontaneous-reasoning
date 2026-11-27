"""Tests for scripts/nebius_panel.py: the request text is the main panel's, the key never reaches a file or a
printed line, resume skips answered keys, the cost meter and both stops. No network: the client is a fake."""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_panel as npl  # noqa: E402

SECRET = "nb-test-SECRET-key-0123456789abcdef"
HAVE_TEMPLATE = npl.npt.check_files(npl.MODEL) is None
HAVE_REQUESTS = all(p.exists() for t in npl.TASKS for v in npl.source_files(t).values()
                    for p in (v if isinstance(v, list) else [v]))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


class FakeClient:
    """Answers like Nebius's /v1/completions; `script` maps a call number to an HTTP status (default 200)."""

    def __init__(self, script=None, prompt_tokens=100, completion_tokens=10, cached=0, text="42}"):
        self._key = SECRET
        self.script, self.n = dict(script or {}), 0
        self.pt, self.ct, self.cached, self.text = prompt_tokens, completion_tokens, cached, text
        self.bodies = []
        self.lock = threading.Lock()

    def post(self, path, body):
        with self.lock:
            self.n += 1
            n = self.n
            self.bodies.append((path, body))
        st = self.script.get(n, 200)
        if st != 200:
            return {"status": st, "error": f"Bearer {SECRET} rate limited", "seconds": 0.0}
        return {"status": 200, "seconds": 0.0, "json": {
            "choices": [{"text": self.text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": self.pt, "completion_tokens": self.ct,
                      "prompt_tokens_details": {"cached_tokens": self.cached}}}}


def req(key="gsm8k|a|direct_answer|0", arm="direct_answer", cap=1024, n=100):
    return {"key": key, "task": "gsm8k", "request_id": key.split("|", 1)[1], "arm": arm, "config_id": "c",
            "item_id": "0", "prompt": "P", "prompt_sha256": "x", "n_prompt_tokens_local": n, "cap": cap}


# ---------------------------------------------------------------- money

def test_call_cost_uses_cached_price():
    assert npl.call_cost(1_000_000, 0, 0) == pytest.approx(0.30)
    assert npl.call_cost(1_000_000, 0, 1_000_000) == pytest.approx(0.006)
    assert npl.call_cost(0, 1_000_000, 0) == pytest.approx(1.20)
    assert npl.call_cost(100, 10, 500) == pytest.approx(npl.call_cost(100, 10, 100))   # cached never above prompt


def test_upper_cost_bounds_any_reply():
    up = npl.upper_cost(100, 16)
    assert up >= npl.call_cost(116, 16, 0)
    assert up >= npl.call_cost(100, 16, 50)


def test_meter_refuses_past_the_limit():
    m = npl.Meter(1.0, 0.5)
    m.reserve(0.4)
    with pytest.raises(npl.Budget):
        m.reserve(0.2)
    m.settle(0.4, 0.1)
    assert m.spent == pytest.approx(0.6) and m.reserved == pytest.approx(0.0)
    m.reserve(0.39)


def test_live_test_spend_counts_the_prefill_test(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps({"status": 200, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
                           for _ in range(3)) + "\n" + json.dumps({"status": 500, "usage": {}}) + "\n",
                 encoding="utf-8")
    s = npl.live_test_spend(p)
    assert s["calls"] == 3 and s["usd"] == pytest.approx(3 * (1000 * 0.30 + 100 * 1.20) / 1e6)


def test_real_live_test_spend_is_cents():
    s = npl.live_test_spend()
    assert s["usd"] < 0.05


# ---------------------------------------------------------------- runner: stops, retries, resume, key

def test_budget_stop_halts_before_the_limit(tmp_path):
    cl = FakeClient(prompt_tokens=100, completion_tokens=10)
    # each call reserves its largest cost and is then billed its real cost; the limit lets exactly 3 in
    a, up = npl.call_cost(100, 10, 0), npl.upper_cost(100, 1024)
    m = npl.Meter(2 * a + up + a / 2, 0.0)
    r = npl.Runner(cl, m, tmp_path / "raw.jsonl", "pilot", sleep=lambda s: None)
    reqs = [req(key=f"gsm8k|a|direct_answer|{i}") for i in range(5)]
    r.unit(reqs)
    r.close()
    assert cl.n == 3 and r.stop.is_set() and "spend limit" in r.stop_reason
    assert m.spent <= m.limit and m.spent == pytest.approx(3 * npl.call_cost(100, 10, 0))


def test_retries_on_429_and_5xx_then_succeeds(tmp_path):
    cl = FakeClient(script={1: 429, 2: 503})
    r = npl.Runner(cl, npl.Meter(10, 0), tmp_path / "raw.jsonl", "pilot", sleep=lambda s: None)
    rec = r.call(req())
    r.close()
    assert rec["status"] == 200 and rec["tries"] == 3 and r.stats["retries"] == 2


def test_no_retry_on_400(tmp_path):
    cl = FakeClient(script={1: 400})
    r = npl.Runner(cl, npl.Meter(10, 0), tmp_path / "raw.jsonl", "pilot", sleep=lambda s: None)
    rec = r.call(req())
    r.close()
    assert rec["status"] == 400 and rec["tries"] == 1 and rec["text"] is None


def test_key_never_written(tmp_path):
    cl = FakeClient(script={1: 401}, text=f"leak {SECRET}")
    p = tmp_path / "raw.jsonl"
    r = npl.Runner(cl, npl.Meter(10, 0), p, "pilot", sleep=lambda s: None)
    r.call(req(key="gsm8k|a|direct_answer|0"))
    r.call(req(key="gsm8k|a|direct_answer|1"))
    r.close()
    txt = p.read_text(encoding="utf-8")
    assert SECRET not in txt and "REDACTED" in txt


def test_key_not_in_request_body(tmp_path):
    cl = FakeClient()
    r = npl.Runner(cl, npl.Meter(10, 0), tmp_path / "raw.jsonl", "pilot", sleep=lambda s: None)
    r.call(req())
    r.close()
    assert SECRET not in json.dumps(cl.bodies)
    body = cl.bodies[0][1]
    assert cl.bodies[0][0] == "/completions" and body["temperature"] == 0.0 and body["max_tokens"] == 1024


def test_read_key_from_environment(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", SECRET)
    assert npl.read_key() == SECRET
    monkeypatch.delenv("NEBIUS_API_KEY")
    with pytest.raises(SystemExit):
        npl.read_key()


def test_resume_skips_answered_keys(tmp_path):
    p = tmp_path / "raw.jsonl"
    rows = [{"key": "k1", "status": 200, "text": "a", "cost_usd": 0.001},
            {"key": "k2", "status": 429, "text": None, "cost_usd": 0.0},
            {"key": "k3", "status": 200, "text": "", "cost_usd": 0.002}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken", encoding="utf-8")
    got = npl.read_raw(p)
    assert npl.done_keys(got) == {"k1", "k3"}
    assert npl.spent_of(got) == pytest.approx(0.003)


def test_usage_reads_cached_tokens():
    assert npl.usage_of({"usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                   "prompt_tokens_details": {"cached_tokens": 7}}})["cached_tokens"] == 7
    assert npl.usage_of({"usage": {"prompt_tokens": 10, "completion_tokens": 2}})["cached_tokens"] is None


# ---------------------------------------------------------------- forced pass, cut, projection

@pytest.mark.parametrize("text,ok", [("42}", True), ("\\frac{3}{4}}", True), ("\\text{Paris}}", True),
                                     ("42", False), ("\\frac{3}{4}", False), ("42} \\boxed{43}", False),
                                     ("</think>42}", False)])
def test_forced_pass(text, ok):
    assert npl.forced_pass(text)["pass"] is ok


def test_cut_at_first_brace():
    assert npl.cut_at_first_brace("42} and more") == ("42}", True)
    assert npl.cut_at_first_brace("42") == ("42", False)


def _rows(task, arm, n, cost, text="1}"):
    return [{"key": f"{task}|{arm}|{i}", "task": task, "arm": arm, "status": 200, "text": text, "cost_usd": cost,
             "usage": {"prompt_tokens": 100, "cached_tokens": 0, "completion_tokens": 5}} for i in range(n)]


def test_projection_scales_by_design_and_stops_above_limit():
    rows = []
    counts = {}
    for t in npl.TASKS:
        for a in npl.ARMS:
            rows += _rows(t, a, 5, 0.0001)
            counts[(t, a)] = 1000
    live = {"usd": 0.001}
    pj = npl.projection(rows, counts, live, limit=10.0)
    assert pj["projected_total_usd"] == pytest.approx(0.001 + 12 * 1000 * 0.0001)
    assert pj["go"] is True
    pj = npl.projection(rows, {k: 10 ** 6 for k in counts}, live, limit=10.0)
    assert pj["fits"] is False and pj["go"] is False


def test_projection_stops_on_low_forced_pass():
    rows, counts = [], {}
    for t in npl.TASKS:
        for a in npl.ARMS:
            rows += _rows(t, a, 10, 1e-6, text="no box" if (t == "math" and a == "forced_direct") else "1}")
            counts[(t, a)] = 10
    pj = npl.projection(rows, counts, {"usd": 0.0})
    assert pj["forced_pass"]["math"]["rate"] == 0.0 and pj["pass_ok"] is False and pj["go"] is False


def test_projection_refuses_missing_cells():
    rows = _rows("gsm8k", "direct_answer", 3, 1e-6)
    pj = npl.projection(rows, {("gsm8k", "direct_answer"): 10, ("bbh", "direct_answer"): 10}, {"usd": 0.0})
    assert pj["go"] is False and "bbh/direct_answer" in pj["cells_missing"]


def test_pilot_units_spread_over_every_configuration():
    u = pd.DataFrame([{"task": t, "config_id": f"c{c}", "item_id": str(i)} for t in ("gsm8k", "bbh")
                      for c in range(3) for i in range(41)])
    u["h"] = [npl.unit_hash(*x) for x in zip(u["task"], u["config_id"], u["item_id"])]
    p = npl.pilot_units(u, 0.05)
    assert (p.groupby(["task", "config_id"]).size() == 3).all()      # ceil(0.05 x 41) = 3
    assert len(npl.pilot_units(u, 0.05)) == len(p) and p.equals(npl.pilot_units(u, 0.05))


# ---------------------------------------------------------------- rendering against the main panel's text

@pytest.mark.skipif(not (HAVE_TEMPLATE and HAVE_REQUESTS), reason="template or request files absent")
@pytest.mark.parametrize("task", npl.TASKS)
def test_rendering_matches_main_panel_text(task):
    df = npl.load_requests(task)
    rend = npl.PanelRenderer()
    n = df.groupby("arm").size()
    assert n["direct_answer"] == n["forced_direct"] == n["zero_shot_cot"]
    for arm in npl.ARMS:
        for _, x in df[df["arm"] == arm].sample(5, random_state=0).iterrows():
            text = json.loads(x["messages_json"])[0]["content"]
            p = rend.prompt(text, arm)
            head = "<｜begin▁of▁sentence｜><｜User｜>" + text + "<｜Assistant｜></think>"
            assert p == head + (npl.PREFILL if arm == "forced_direct" else "")
            assert "<think>" not in p


@pytest.mark.skipif(not HAVE_REQUESTS, reason="request files absent")
def test_design_counts_are_the_main_panels():
    df = npl.load_all()
    c = npl.design_counts(df)
    assert c[("gsm8k", "zero_shot_cot")] == 38251 and c[("math", "forced_direct")] == 12338
    assert c[("bbh", "direct_answer")] == 47250 and c[("hotpot", "zero_shot_cot")] == 36000
    assert len(df) == 401517
    other = pd.read_parquet(ROOT / "data" / "requests" / "gsm8k" / "gemma-2b-it.parquet",
                            columns=["request_id", "messages_json"])
    mine = df[df["task"] == "gsm8k"].set_index("request_id")["messages_json"]
    assert (other.set_index("request_id")["messages_json"] == mine.reindex(other["request_id"]).values).all()
