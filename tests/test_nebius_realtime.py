"""Tests for the real-time run of the complete design in scripts/nebius_panel.py (run --design complete): the rate
limiter, the dispatch order, the projection, the 20-request test's checks, the $120 limit, resume and the spend stop.
No network: the client is a fake, urlopen raises."""
from __future__ import annotations

import json
import sys
import threading
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_panel as npl  # noqa: E402

SECRET = "nb-test-SECRET-key-0123456789abcdef"
HAVE_TEMPLATE = npl.npt.check_files(npl.MODEL) is None
HAVE_LADDER = all((npl.LADDER_REQ_DIR / t / f"{npl.LADDER_REQ_KEY}.parquet").exists() for t in npl.TASKS)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


class FakeClient:
    def __init__(self, prompt_tokens=None, completion_tokens=10, cached=0, text="42}", finish="stop"):
        self._key = SECRET
        self.n, self.bodies = 0, []
        self.pt, self.ct, self.cached, self.text, self.finish = prompt_tokens, completion_tokens, cached, text, finish
        self.lock = threading.Lock()

    def post(self, path, body):
        with self.lock:
            self.n += 1
            self.bodies.append((path, body))
        pt = self.pt if self.pt is not None else len(body["prompt"])
        return {"status": 200, "seconds": 0.0, "json": {
            "choices": [{"text": self.text, "finish_reason": self.finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": self.ct,
                      "prompt_tokens_details": {"cached_tokens": self.cached}}}}


class FakeRend:
    """Renders a message list to a string whose length is its token count."""

    def prompt_messages(self, messages, arm):
        s = "<｜begin▁of▁sentence｜>"
        for m in messages:
            s += ("<｜User｜>" + m["content"] + "<｜Assistant｜></think>") if m["role"] == "user" \
                else (m["content"] + "<｜end▁of▁sentence｜>")
        return s + (npl.PREFILL if arm in npl.FORCED_ARMS else "")

    def count(self, p):
        return len(p)


def design(n_items=6, configs=("c1", "c2")):
    """A small complete design on GSM8K: three zero-shot arms and five-shot forced per (config, item), five-shot
    requested on c1 (config id with the 'gsm8k/' prefix)."""
    rows = []
    shots = [{"role": r, "content": f"{r}{k}"} for k in range(5) for r in ("user", "assistant")]
    for c in configs:
        for i in range(n_items):
            q = [{"role": "user", "content": f"q {c} {i}"}]
            for a, m in (("direct_answer", q), ("forced_direct", q), ("zero_shot_cot", q),
                         ("fewshot_forced", shots + q)):
                rid = f"{c}|{i}|{a}"
                rows.append({"request_id": rid, "arm": a, "config_id": c, "item_id": str(i),
                             "messages_json": json.dumps(m)})
            if c == "c1":
                rows.append({"request_id": f"gsm8k/{c}|{i}|fsr", "arm": "fewshot_requested", "config_id": f"gsm8k/{c}",
                             "item_id": str(i), "messages_json": json.dumps(shots + q)})
    df = pd.DataFrame(rows)
    df["task"] = "gsm8k"
    df["key"] = "gsm8k|" + df["request_id"]
    df["unit_config"] = [npl.base_config("gsm8k", c) for c in df["config_id"]]
    caps = {"direct_answer": 7000, "zero_shot_cot": 7000, "forced_direct": 16, "fewshot_forced": 16,
            "fewshot_requested": 7000}
    pcaps = {"direct_answer": 1024, "zero_shot_cot": 3072, "forced_direct": 16, "fewshot_forced": 16,
             "fewshot_requested": 1024}
    df["cap"] = df["arm"].map(caps)
    df["panel_cap"] = df["arm"].map(pcaps)
    return df


# ---------------------------------------------------------------- rate limiter

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_rate_limiter_holds_calls_per_minute():
    c = Clock()
    lim = npl.RateLimiter(rpm=3, tpm=10 ** 9, clock=c, sleep=c.sleep)
    for _ in range(3):
        lim.acquire(10)
    assert c.t == 0.0
    lim.acquire(10)
    assert c.t >= 60.0 and lim.waited > 0


def test_rate_limiter_holds_tokens_per_minute():
    c = Clock()
    lim = npl.RateLimiter(rpm=1000, tpm=1000, clock=c, sleep=c.sleep)
    lim.acquire(600)
    lim.add(300)
    assert c.t == 0.0
    lim.acquire(200)            # 600 + 300 + 200 > 1,000: waits for the first entry to leave the window
    assert c.t >= 60.0


def test_rate_limiter_lets_one_oversized_call_through():
    c = Clock()
    lim = npl.RateLimiter(rpm=10, tpm=100, clock=c, sleep=c.sleep)
    lim.acquire(5000)
    assert c.t == 0.0


# ---------------------------------------------------------------- order

def test_complete_work_covers_every_remaining_row_once_and_keeps_units_and_blocks():
    df = design(n_items=7)
    done = {"gsm8k|c1|0|direct_answer", "gsm8k|c2|3|fewshot_forced"}
    w = npl.complete_work(df, done, fs_block=3)
    pos = np.concatenate(w)
    assert len(pos) == len(set(pos)) == len(df) - len(done)
    assert not set(df.iloc[pos]["key"]) & done
    for item in w:
        g = df.iloc[item]
        if g["arm"].iloc[0] in npl.ARMS:
            assert g["arm"].isin(npl.ARMS).all() and g[["config_id", "item_id"]].drop_duplicates().shape[0] == 1
            assert list(g["arm"]) == [a for a in npl.ARMS if a in set(g["arm"])]
        else:
            assert len(g) <= 3 and g["arm"].nunique() == 1 and g["config_id"].nunique() == 1
    first_half = df.iloc[np.concatenate(w[: len(w) // 2])]
    assert set(first_half["arm"]) == set(df["arm"])            # the two streams interleave


def test_complete_work_is_empty_when_everything_is_answered():
    df = design(n_items=2)
    assert npl.complete_work(df, set(df["key"])) == []


# ---------------------------------------------------------------- projection

def test_complete_projection_uses_cached_share_reruns_and_limit():
    df = design(n_items=2, configs=("c1",))
    rem = df[df["arm"].isin(["direct_answer", "fewshot_forced"])]
    n = pd.Series(1000, index=rem.index)
    means = {("gsm8k", "direct_answer"): {"n": 5, "completion_tokens": 100.0, "completion_tokens_high": 200.0,
                                          "cached_share": 0.5},
             ("gsm8k", "forced_direct"): {"n": 5, "completion_tokens": 4.0, "completion_tokens_high": 4.0,
                                          "cached_share": 0.0}}
    rerun_key = rem[rem["arm"] == "direct_answer"]["key"].iloc[0]
    reruns = {rerun_key: {"usage": {"completion_tokens": 1024}}}
    pj = npl.complete_projection(rem, n, means, spent=3.0, limit=120.0, fs_cached={("gsm8k", "fewshot_forced"): 0.8},
                                 reruns=reruns)
    cells = {c["arm"]: c for c in pj["cells"]}
    da = cells["direct_answer"]
    pin_da = 1000 * (0.5 * 0.30 + 0.5 * 0.006) / 1e6
    assert da["usd"] == pytest.approx(2 * pin_da + (1024 + 100) * 1.20 / 1e6)
    assert da["usd_high"] == pytest.approx(2 * pin_da + (7000 + 200) * 1.20 / 1e6)
    ff = cells["fewshot_forced"]
    assert ff["cached_share"] == 0.8
    assert ff["usd"] == pytest.approx(2 * (1000 * (0.2 * 0.30 + 0.8 * 0.006) + 4 * 1.20) / 1e6)
    assert ff["usd_nocache"] == pytest.approx(2 * (1000 * 0.30 + 4 * 1.20) / 1e6)
    assert pj["go"] and pj["total_usd"] == pytest.approx(3.0 + pj["projected_usd"])
    assert not npl.complete_projection(rem, n, means, spent=119.9999, limit=120.0)["go"]


def test_complete_projection_refuses_an_arm_with_no_pilot_replies():
    df = design(n_items=1, configs=("c1",))
    with pytest.raises(SystemExit):
        npl.complete_projection(df, pd.Series(10, index=df.index), {}, 0.0, 120.0)


# ---------------------------------------------------------------- the test's checks

def test_check_test_reads_pass_rules_and_cached_share():
    rend = FakeRend()
    df = design(n_items=3, configs=("c1",))
    ff = df[df["arm"] == "fewshot_forced"].to_dict("records")
    recs = []
    for k, x in enumerate(ff):
        q = npl.complete_request(x, rend)
        recs.append({"req": q, "rec": {"status": 200, "text": "7}", "finish_reason": "stop", "cost_usd": 1e-5,
                                       "usage": {"prompt_tokens": len(q["prompt"]), "completion_tokens": 2,
                                                 "cached_tokens": 0 if k == 0 else len(q["prompt"]) // 2}}})
    r = npl.check_test(recs, {})
    assert r["passed"] and r["replies"][0]["user_turns"] == 6 and r["replies"][0]["first_of_block"]
    assert r["fs_cached_share"][("gsm8k", "fewshot_forced")] == pytest.approx(0.5, abs=0.01)
    recs[1]["rec"]["text"] = "7"                                         # the box never closes
    assert not npl.check_test(recs, {})["passed"]


def test_check_test_reports_whether_a_rerun_starts_with_the_pilot_reply():
    rend = FakeRend()
    x = design(n_items=1, configs=("c1",)).iloc[0].to_dict()
    q = npl.complete_request(x, rend)
    rec = {"status": 200, "text": "abc def ghi", "finish_reason": "stop", "cost_usd": 0.0,
           "usage": {"prompt_tokens": len(q["prompt"]), "completion_tokens": 3, "cached_tokens": 0}}
    same = npl.check_test([{"req": q, "rec": rec}], {x["key"]: {"text": "abc def"}})
    assert same["passed"] and same["replies"][0]["rerun_starts_with_pilot_reply"]
    other = npl.check_test([{"req": q, "rec": rec}], {x["key"]: {"text": "abx"}})
    # the server's determinism is reported, not a pass rule: the rerun's reply is the one kept
    assert other["passed"] and not other["replies"][0]["rerun_starts_with_pilot_reply"]
    assert other["replies"][0]["rerun_common_prefix_chars"] == 2


def test_bbh_five_shot_blocks_keep_one_subtask():
    df = design(n_items=8, configs=("c1",))
    df["task"] = "bbh"
    df["item_id"] = [f"{'sub_a' if int(i) % 2 else 'sub_b'}/{i}" for i in df["item_id"]]
    df["key"] = "bbh|" + df["request_id"] + "|" + df["item_id"]
    df["unit_config"] = [npl.base_config("bbh", c.replace("gsm8k/", "bbh/")) for c in df["config_id"]]
    w = npl.complete_work(df, set(), fs_block=3)
    for item in w:
        g = df.iloc[item]
        if g["arm"].iloc[0] in npl.FIVE_SHOT_ARMS:
            assert g["item_id"].str.split("/").str[0].nunique() == 1
    assert npl.shot_family("bbh", "sub_a/3") == "sub_a" and npl.shot_family("math", "test/algebra/1.json") == ""


# ---------------------------------------------------------------- end to end with the fake client

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    df = design(n_items=4)
    monkeypatch.setattr(npl, "load_complete", lambda *a, **k: df)
    monkeypatch.setattr(npl, "load_all", lambda *a, **k: df[df["arm"].isin(npl.ARMS)])
    monkeypatch.setattr(npl, "check_complete_against_panel", lambda c, p: {"ok": True})
    monkeypatch.setattr(npl, "RAW", tmp_path / "raw.jsonl")
    monkeypatch.setattr(npl, "DATA", tmp_path)
    monkeypatch.setattr(npl, "RES", tmp_path / "res")
    monkeypatch.setattr(npl, "LIVE_TEST_RAW", tmp_path / "none.jsonl")
    monkeypatch.setattr(npl, "REALTIME_PROJECTION", tmp_path / "res" / "proj.json")
    monkeypatch.setattr(npl, "REALTIME_TEST", tmp_path / "res" / "test.json")
    monkeypatch.setattr(npl, "PROMPT_COUNTS", tmp_path / "counts.parquet")
    monkeypatch.setattr(npl, "prompt_counts",
                        lambda d, rend, path=None, progress_every=0: pd.Series(
                            [rend.count(rend.prompt_messages(json.loads(m), a)) for m, a in
                             zip(d["messages_json"], d["arm"])], index=d.index))
    monkeypatch.setattr(npl, "read_key", lambda *a, **k: (_ for _ in ()).throw(AssertionError("key read")))
    # a pilot: the zero-shot arms of c1 item 0 answered at the panel's caps; its requested reply cut at 1,024
    rend = FakeRend()
    with open(tmp_path / "raw.jsonl", "w", encoding="utf-8") as fh:
        for x in df[(df["config_id"] == "c1") & (df["item_id"] == "0") & df["arm"].isin(npl.ARMS)].to_dict("records"):
            q = npl.complete_request(x, rend)
            cap = npl.CAPS["gsm8k"][x["arm"]]
            cut = x["arm"] == "direct_answer"
            fh.write(json.dumps({"key": x["key"], "task": "gsm8k", "request_id": x["request_id"], "arm": x["arm"],
                                 "config_id": x["config_id"], "item_id": x["item_id"], "phase": "pilot",
                                 "status": 200, "text": "42}", "finish_reason": "length" if cut else "stop",
                                 "usage": {"prompt_tokens": 50, "completion_tokens": cap if cut else 5,
                                           "cached_tokens": 10}, "cost_usd": 0.001, "cap": cap,
                                 "n_prompt_tokens_local": q["n_prompt_tokens_local"],
                                 "prompt_sha256": q["prompt_sha256"], "t": "2026-11-22T16:00:00"}) + "\n")
    return types.SimpleNamespace(df=df, rend=rend, tmp=tmp_path)


def args(**kw):
    a = dict(limit_usd=120.0, project=False, test=0, base_url="http://x", timeout=1.0, threads=3, rpm=10 ** 6,
             tpm=10 ** 9, approved=False)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_limit_above_120_is_refused(sandbox):
    assert npl.cmd_complete(args(limit_usd=120.01), rend=sandbox.rend, client=FakeClient()) == 2


def test_run_refuses_before_projection_and_test(sandbox):
    cl = FakeClient()
    assert npl.cmd_complete(args(), rend=sandbox.rend, client=cl) == 2 and cl.n == 0


def test_project_test_run_resume(sandbox):
    df, rend = sandbox.df, sandbox.rend
    assert npl.cmd_complete(args(project=True), rend=rend) == 0
    pj = json.loads(npl.REALTIME_PROJECTION.read_text(encoding="utf-8"))
    assert pj["to_send"] == len(df) - 2 and pj["reruns"] == 1 and not pj["test"]["ran"]
    cl = FakeClient()
    assert npl.cmd_complete(args(test=20), rend=rend, client=cl) == 0
    tj = json.loads(npl.REALTIME_TEST.read_text(encoding="utf-8"))
    assert tj["passed"] and tj["calls"] == cl.n <= 20
    assert npl.cmd_complete(args(test=20), rend=rend, client=cl) == 6               # never sent twice
    assert npl.cmd_complete(args(), rend=rend, client=FakeClient()) == 2            # projection predates the test
    assert npl.cmd_complete(args(project=True), rend=rend) == 0
    cl2 = FakeClient()
    assert npl.cmd_complete(args(), rend=rend, client=cl2) == 0
    rows = npl.read_raw(npl.RAW)
    assert npl.complete_done_keys(rows) >= set(df["key"])
    new = [r for r in rows if r["phase"] == "complete"]
    assert all(r["design"] == npl.DESIGN_COMPLETE and r["panel_cap"] for r in new)
    assert all(b["max_tokens"] in (7000, 16) and b["temperature"] == 0.0 for _, b in cl2.bodies)
    rerun = [r for r in rows if r["key"] == "gsm8k|c1|0|direct_answer"]
    assert len(rerun) == 2 and rerun[-1]["cap"] == 7000                              # the cut pilot reply, sent again
    assert cl.n + cl2.n == len(df) - 2
    cl3 = FakeClient()
    assert npl.cmd_complete(args(), rend=rend, client=cl3) == 0 and cl3.n == 0      # resume: nothing left
    assert SECRET not in npl.RAW.read_text(encoding="utf-8")


def test_run_stops_at_the_spend_limit(sandbox):
    rend = sandbox.rend
    assert npl.cmd_complete(args(project=True), rend=rend) == 0
    assert npl.cmd_complete(args(test=20), rend=rend, client=FakeClient()) == 0
    assert npl.cmd_complete(args(project=True), rend=rend) == 0
    spent = npl.spent_of(npl.read_raw(npl.RAW))
    pj = json.loads(npl.REALTIME_PROJECTION.read_text(encoding="utf-8"))
    limit = spent + pj["projected_usd"] + 0.02           # the projection fits; the replies cost more than projected
    cl = FakeClient(completion_tokens=6000)
    rc = npl.cmd_complete(args(limit_usd=limit), rend=rend, client=cl)
    rows = npl.read_raw(npl.RAW)
    assert rc == 1 and npl.spent_of(rows) <= limit + 1e-9
    log = [json.loads(x) for x in (sandbox.tmp / "nebius_panel_runs.jsonl").read_text().splitlines()]
    assert "spend limit" in log[-1]["stop_reason"]


# ---------------------------------------------------------------- real files

@pytest.mark.skipif(not (HAVE_TEMPLATE and HAVE_LADDER and npl.RAW.exists()), reason="ladder files or raw absent")
def test_complete_zero_shot_prompts_are_the_pilots():
    df = npl.load_complete()
    rows = {r["key"]: r for r in npl.read_raw(npl.RAW) if r.get("phase") == "pilot"}
    rend = npl.PanelRenderer()
    z = df[df["key"].isin(rows)].sample(60, random_state=3)
    for x in z.to_dict("records"):
        q = npl.complete_request(x, rend)
        assert q["prompt_sha256"] == rows[x["key"]]["prompt_sha256"]


@pytest.mark.skipif(not (HAVE_TEMPLATE and HAVE_LADDER), reason="ladder files absent")
@pytest.mark.parametrize("task", npl.TASKS)
def test_five_shot_prompts_render_as_multi_turn_chat(task):
    df = npl.load_complete(tasks=(task,))
    rend = npl.PanelRenderer()
    for arm in npl.FIVE_SHOT_ARMS:
        g = df[df["arm"] == arm]
        if g.empty:
            continue
        for x in g.sample(3, random_state=0).to_dict("records"):
            q = npl.complete_request(x, rend)
            assert q["prompt"].count("<｜User｜>") == 6 and "<think>" not in q["prompt"]
            assert q["prompt"].endswith("<｜Assistant｜></think>" + (npl.PREFILL if arm in npl.FORCED_ARMS else ""))
            assert q["cap"] == (npl.design_cap(task, arm))
