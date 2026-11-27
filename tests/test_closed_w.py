"""Tests for scripts/closed_w.py (Qwen4-Max's three arms on OpenRouter: requested, forced, chain of thought; the
prefill test; the quote). No network: urllib.request.urlopen raises in every test, and every reply comes from a fake
HTTP object or a fake opener."""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import closed_w as cw  # noqa: E402
import closed_w_items as cwp  # noqa: E402

SECRET = "sk-or-v1-TESTSECRET0123456789abcdefTESTSECRET"
CFG = cw.load_config()
BY_ID = {m["id"]: m for m in CFG["models"]}
QWEN = BY_ID["qwen/qwen4-max"]
HAVE_DATA = all(p.exists() for t in cw.TASKS for a in cw.ARMS for p in cw.arm_request_paths(CFG, a, t)) and \
    all(cw.screening_path(CFG, t).exists() for t in cw.TASKS)
HAVE_MANIFEST = (ROOT / CFG["paths"]["manifest"]).exists()
needs_data = pytest.mark.skipif(not (HAVE_DATA and HAVE_MANIFEST), reason="panel request files or manifest absent")
PREFIX = "\\boxed{"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


_REQ_CACHE = {}
_ARM_CACHE = {}


@pytest.fixture
def cached_requests(monkeypatch):
    """The panel's request files of the three arms are read once per test session."""
    if not _REQ_CACHE:
        _REQ_CACHE.update(cw.load_all_requests(CFG))
    real = cw.load_arm_requests

    def arm_requests(cfg, arm, task, root=ROOT, request_ids=None):
        if arm == "requested":
            df = _REQ_CACHE[task]
        else:
            if (arm, task) not in _ARM_CACHE:
                _ARM_CACHE[(arm, task)] = real(cfg, arm, task, root)
            df = _ARM_CACHE[(arm, task)]
        if request_ids is not None:
            df = df[df["request_id"].isin(set(request_ids))]
        return df.copy().reset_index(drop=True)
    monkeypatch.setattr(cw, "load_all_requests", lambda cfg, root=ROOT: {t: v.copy() for t, v in _REQ_CACHE.items()})
    monkeypatch.setattr(cw, "load_arm_requests", arm_requests)
    return _REQ_CACHE


# ------------------------------------------------------------------------------------------------ fakes

def body_arm(body) -> str:
    """Which arm a request body is: forced (an assistant message last), cot (the chain-of-thought cap), requested."""
    if body["messages"][-1]["role"] == "assistant":
        return "forced"
    return "cot" if body["max_tokens"] in (3072, 1900) else "requested"


def reply(body, text="\\boxed{18}", pt=100, ct=5, rt=0, provider=None, served=None, cost="auto", finish="stop",
          refusal=None, reasoning_text=None, cached=0):
    m = BY_ID[body["model"]]
    if cost == "auto":
        cost = cw.expected_cost({"prompt_tokens": pt, "completion_tokens": ct + rt, "cached_tokens": cached}, m["price"])
    return {"status": 200, "seconds": 0.01, "json": {
        "id": "gen-test", "model": served or (m["id"] + "-20261115"), "provider": provider or m["expected_provider"],
        "object": "chat.completion", "created": 1,
        "choices": [{"index": 0, "finish_reason": finish, "native_finish_reason": finish,
                     "message": {"role": "assistant", "content": text, "refusal": refusal,
                                 "reasoning": reasoning_text}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct + rt, "total_tokens": pt + ct + rt,
                  "prompt_tokens_details": {"cached_tokens": cached},
                  "completion_tokens_details": {"reasoning_tokens": rt}, "cost": cost}}}


def thinking_on(body) -> bool:
    return body["reasoning"] != BY_ID[body["model"]]["reasoning_off"]


def good(body, n):
    """A well-behaved route: the forced arm continues the open box (the stop string, when sent, left out, as an
    OpenAI-compatible API does), the chain of thought writes some working, the requested arm a bare box."""
    if thinking_on(body):
        return reply(body, rt=200)
    arm = body_arm(body)
    if arm == "forced":
        return reply(body, text="18" if body.get("stop") else "18}", ct=3)
    if arm == "cot":
        return reply(body, text="First add, then check. \\boxed{18}", ct=14)    # 12 Qwen tokens, 16 fake
    return reply(body)


def listing(url, params=None, price_factor=1.0):
    for m in CFG["models"]:
        if m["endpoints_slug"] in url:
            sup = params if params is not None else ["max_tokens", "temperature", "top_p", "reasoning",
                                                     "include_reasoning", "presence_penalty", "stop", "seed"]
            return {"status": 200, "json": {"data": {"endpoints": [
                {"tag": m["endpoint_tag"] + "/fast", "provider_name": m["expected_provider"],
                 "pricing": {"prompt": "1", "completion": "1"}, "supported_parameters": sup},
                {"tag": m["endpoint_tag"], "provider_name": m["expected_provider"],
                 "pricing": {"prompt": str(m["price"]["prompt"] * price_factor),
                             "completion": str(m["price"]["completion"] * price_factor), "discount": 0.05},
                 "supported_parameters": sup}]}}}
    return {"status": 404, "json": None, "error": "no such model"}


class FakeHttp:
    def __init__(self, responder=good, list_fn=listing):
        self._key = SECRET
        self.responder, self.list_fn = responder, list_fn
        self.posts, self.gets = [], []
        self.lock = threading.Lock()

    def post_json(self, path, data):
        with self.lock:
            self.posts.append((path, data))
            n = len(self.posts)
        return self.responder(json.loads(data.decode("utf-8")), n)

    def get_public(self, url):
        self.gets.append(url)
        return self.list_fn(url)

    def bodies(self):
        return [json.loads(d.decode("utf-8")) for _, d in self.posts]


class FakeTok:
    name = "fake: one token per two characters"

    def count(self, text):
        return max(1, len(text or "") // 2) if isinstance(text, str) else 0

    def counts(self, texts):
        return [self.count(t) for t in texts]


def fake_scorer(df):
    out = df.copy()
    ok = out["text"].map(lambda s: int(isinstance(s, str) and "18" in s))
    out["strict_correct"] = ok
    out["content_correct"] = ok
    out["lenient_correct"] = ok
    out["compliant"] = out["text"].map(lambda s: int(isinstance(s, str) and "\\boxed{" in s))
    return out, {"fake": "scorer"}


def tmp_paths(tmp_path, env=True, prefill=True) -> cw.Paths:
    shutil.copy(ROOT / "config" / "budget.yaml", tmp_path / "budget.yaml")
    if env:
        os.environ["OPENROUTER_API_KEY"] = SECRET
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)
    pr = tmp_path / "prefill_result.json"
    if prefill:
        pr.write_text(json.dumps({"summary": {"passed": True, "passing_variants": ["partial_true"], "variants": {
            "partial_true": {"message_extra": {"partial": True}, "pass": True},
            "assistant_prefix": {"message_extra": {}, "pass": False}}}}), encoding="utf-8")
    return cw.Paths(CFG, ROOT, raw_dir=tmp_path / "raw", results=tmp_path / "res", budget=tmp_path / "budget.yaml",
                    env=tmp_path / ".env", prefill_result=pr)


def row(rid="ps-x|direct_answer|0", text="Question?\nEnd with \\boxed{<number>}.", cfgid="ps-x", item="0"):
    return {"request_id": rid, "config_id": cfgid, "item_id": item, "text": text}


def files_text(*dirs) -> str:
    out = []
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.is_file():
                out.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def no_ladder(task, items, templates, root):
    return pd.DataFrame(columns=cw.PANEL_COLS)


# ------------------------------------------------------------------------------------------------ requests

@needs_data
def test_request_bytes_equal_the_panels():
    man = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    for t in cw.TASKS:
        ids = [r["request_id"] for r in man["pilot"]["rows"][t]]
        raw = pd.read_parquet(cw.request_path(CFG, t), columns=["request_id", "arm", "messages_json"],
                              filters=[("request_id", "in", ids)])
        raw = raw[raw["arm"] == "direct_answer"].set_index("request_id")
        df = cw.load_requests(cw.request_path(CFG, t), request_ids=ids)
        assert len(df) == len(ids)
        for r in df.to_dict("records"):
            panel = json.loads(raw.at[r["request_id"], "messages_json"])
            req = cw.make_request(CFG, QWEN, t, r)
            sent = json.loads(cw.encode_body(req["body"]).decode("utf-8"))
            assert sent["messages"] == panel                      # one user turn, the panel's, nothing added
            assert sent["messages"][0]["content"].encode("utf-8") == panel[0]["content"].encode("utf-8")
            assert not {"tools", "tool_choice", "response_format", "stop", "system"} & set(sent)
            assert req["n_bytes"] == len(panel[0]["content"].encode("utf-8"))


@needs_data
def test_forced_and_cot_requests_are_the_panels(cached_requests):
    """The forced arm sends the panel's forced user message (the requested one, unchanged) and the prefill with
    partial: true; the chain-of-thought arm sends the panel's zero-shot CoT message, alone."""
    man = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    for t in cw.TASKS:
        fr = man["arms"]["pilot"]["forced"][t]
        cr = man["arms"]["pilot"]["cot"][t]
        f = cw.load_arm_requests(CFG, "forced", t, request_ids=[r["request_id"] for r in fr]).set_index("request_id")
        c = cw.load_arm_requests(CFG, "cot", t, request_ids=[r["request_id"] for r in cr]).set_index("request_id")
        q = cw.load_requests(cw.request_path(CFG, t), request_ids=[r["request_id"] for r in man["pilot"]["rows"][t]])
        q = q.set_index(["config_id", "item_id"])
        for r in fr:
            x = f.loc[r["request_id"]]
            assert r["request_id"].endswith("|fd")
            assert x["text"] == q.loc[(r["config_id"], r["item_id"]), "text"]          # the requested message, unchanged
            b = cw.make_request(CFG, QWEN, t, {"request_id": r["request_id"], "config_id": r["config_id"],
                                               "item_id": r["item_id"], "text": x["text"]}, "forced")["body"]
            assert b["messages"] == [{"role": "user", "content": x["text"]},
                                     {"role": "assistant", "content": PREFIX, "partial": True}]
        for r in cr:
            x = c.loc[r["request_id"]]
            raw = pd.read_parquet(cw.request_path(CFG, t), columns=["request_id", "messages_json"],
                                  filters=[("request_id", "==", r["request_id"])])
            b = cw.make_request(CFG, QWEN, t, {"request_id": r["request_id"], "config_id": r["config_id"],
                                               "item_id": r["item_id"], "text": x["text"]}, "cot")["body"]
            assert b["messages"] == json.loads(raw["messages_json"].iloc[0]) and "stop" not in b


def test_request_bodies_per_arm():
    for t in cw.TASKS:
        req = cw.make_request(CFG, QWEN, t, row())
        f = cw.make_request(CFG, QWEN, t, row(rid="ps-x|direct_answer|0|fd"), "forced")
        c = cw.make_request(CFG, QWEN, t, row(rid="ps-x|zero_shot_cot|0"), "cot")
        n = cw.make_request(CFG, QWEN, t, row(rid="ps-x|direct_answer|0|fd"), "forced_nostop", rep=0)
        fb, cb, nb = f["body"], c["body"], n["body"]
        # forced: the prefill as an assistant message with partial: true, the panel's caps, the stop where it stopped
        assert fb["messages"][-1] == {"role": "assistant", "content": PREFIX, "partial": True}
        assert fb["messages"][0] == {"role": "user", "content": row()["text"]} and len(fb["messages"]) == 2
        assert fb["max_tokens"] == (64 if t == "hotpot" else 16)
        assert fb.get("stop") == (["}"] if t in ("gsm8k", "bbh") else None) and f["stop_sent"] == (t in ("gsm8k", "bbh"))
        assert "stop" not in nb and nb["messages"] == fb["messages"] and nb["max_tokens"] == fb["max_tokens"]
        # chain of thought: one user turn, the panel's caps, no stop
        assert cb["max_tokens"] == (1900 if t == "hotpot" else 3072) and len(cb["messages"]) == 1 and "stop" not in cb
        assert req["body"]["max_tokens"] == 1024 and "stop" not in req["body"]
        for b in (fb, cb, nb, req["body"]):
            assert b["reasoning"] == {"effort": "none"} and b["temperature"] == 0.0 and b["top_p"] == 1.0
            assert b["provider"] == {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
            assert b["presence_penalty"] == 0.0 and "seed" not in b
        assert len({req["key"], f["key"], c["key"], n["key"]}) == 4
        assert f["key"] == f"qwen4-max|forced|{t}|ps-x|direct_answer|0|fd" and f["arm"] == "forced"
        assert f["n_bytes"] == len(row()["text"].encode("utf-8")) + len(PREFIX)
        assert c["reserve_usd"] > req["reserve_usd"] > f["reserve_usd"]
        assert c["reserve_usd"] == pytest.approx((c["n_bytes"] + 64) * QWEN["price"]["prompt"]
                                                 + cb["max_tokens"] * QWEN["price"]["completion"])
    assert cw.design_parameters(CFG, QWEN) == ["max_tokens", "temperature", "top_p", "reasoning", "stop",
                                               "presence_penalty"]


def test_every_body_pins_the_provider_and_switches_reasoning_off():
    for m in CFG["models"]:
        b = cw.make_request(CFG, m, "gsm8k", row())["body"]
        assert b["provider"] == {"only": m["provider"]["only"], "allow_fallbacks": False, "require_parameters": True}
        assert b["reasoning"] == m["reasoning_off"]
        assert b["temperature"] == 0.0 and b["top_p"] == 1.0 and b["max_tokens"] == 1024
        assert "stop" not in b and "seed" not in b
        c = cw.make_request(CFG, m, "math", row(), "control", rep=0)["body"]
        assert c["reasoning"] == m["reasoning_on"] and c["reasoning"] != m["reasoning_off"]
    assert QWEN["reasoning_off"] == {"effort": "none"}
    assert "presence_penalty" not in cw.make_request(CFG, BY_ID["openai/gpt-6-luna"], "gsm8k", row())["body"]
    for t, cap in (("gsm8k", 16), ("hotpot", 64)):
        v = CFG["design"]["prefill_variants"][1]
        p = cw.make_request(CFG, QWEN, t, row(), "prefill", variant=v)
        assert p["body"]["messages"][-1] == {"role": "assistant", "content": PREFIX, "partial": True}
        assert p["body"]["max_tokens"] == cap and p["body"]["reasoning"] == {"effort": "none"}
        assert "stop" not in p["body"]                                  # the prefill test sent no stop


def test_luna_is_off_by_default():
    assert cw.default_model_keys(CFG) == ["qwen4-max"]
    assert [m["key"] for m in cw._models(CFG, type("A", (), {"models": None})())] == ["qwen4-max"]
    assert "gpt-6-luna" in [m["key"] for m in CFG["models"]]          # the code path stays
    assert cw.run_arms(CFG) == ["requested", "forced", "cot"] and CFG["limits"]["run"] is None


def test_config_agrees_with_the_panels_decoding(tmp_path):
    d = cw.check_panel_decoding(CFG)
    assert d["requested_cap"] == 1024 and d["forced_caps"] == {"gsm8k": 16, "math": 16, "bbh": 16, "hotpot": 64}
    assert d["cot_caps"] == {"gsm8k": 3072, "math": 3072, "bbh": 3072, "hotpot": 1900}
    assert d["prefill"] == PREFIX and d["cut_at_first_brace"] == ["bbh", "gsm8k"] and d["stop_sent"] == ["bbh", "gsm8k"]
    for k, v in (("requested", 512), ("cot", {"gsm8k": 3072, "math": 3072, "bbh": 3072, "hotpot": 3072})):
        bad = json.loads(json.dumps(CFG))
        bad["caps"][k] = v
        with pytest.raises(SystemExit):
            cw.check_panel_decoding(bad)
    bad = json.loads(json.dumps(CFG))
    bad["forced_cut_at_first_brace"] = ["gsm8k", "bbh", "math"]
    with pytest.raises(SystemExit):
        cw.check_panel_decoding(bad)
    worse = json.loads(json.dumps(CFG))
    worse["models"][0]["provider"]["allow_fallbacks"] = True
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(worse), encoding="utf-8")
    with pytest.raises(SystemExit):
        cw.load_config(p)


# ------------------------------------------------------------------------------------------------ items

def test_item_draws_are_stable_by_seed():
    ids = [str(i) for i in range(1319)]
    a = cw.draw_items("gsm8k", ids)
    b = cw.draw_items("gsm8k", list(reversed(ids)))
    assert a == b == [str(x) for x in cwp.sample_items("gsm8k", ids, 100)] and len(a) == 100
    bbh = [f"task{k}/{i}" for k in range(25) for i in range(54)]
    d = cw.draw_items("bbh", bbh)
    assert len(d) == 100 and pd.Series([x.split("/")[0] for x in d]).value_counts().eq(4).all()
    assert cw.seed_labels("gsm8k", ids) == {"closed_w/b/gsm8k": cwp.label_seed("closed_w/b/gsm8k")}
    df = pd.DataFrame({"request_id": [f"c{c}|{i}" for c in range(6) for i in range(10)],
                       "config_id": [f"c{c}" for c in range(6) for i in range(10)],
                       "item_id": [str(i) for c in range(6) for i in range(10)],
                       "text": ["x" * (c * 10 + i + 1) for c in range(6) for i in range(10)]})
    p1 = cw.pick_spread(df, 5, "lab", "gsm8k", longest_first=True)
    p2 = cw.pick_spread(df.sample(frac=1, random_state=3), 5, "lab", "gsm8k", longest_first=True)
    assert list(p1["request_id"]) == list(p2["request_id"])
    assert p1["config_id"].nunique() == 5 and p1["item_id"].nunique() == 5
    assert p1["request_id"].iloc[0] == "c5|9"                      # the longest first


@needs_data
def test_manifest_items_are_the_b100_draws():
    man = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    price = json.loads((ROOT / CFG["paths"]["price_tokens"]).read_text(encoding="utf-8"))["designs"]["B100"]["per_task"]
    for t in cw.TASKS:
        ids = pd.read_parquet(cw.request_path(CFG, t), columns=["item_id", "arm"],
                              filters=[("arm", "==", "direct_answer")])["item_id"].astype(str).unique().tolist()
        assert man["tasks"][t]["items"] == cw.draw_items(t, ids)
        assert man["tasks"][t]["n_requests"] == price[t]["calls"]
        assert man["tasks"][t]["n_templates"] == price[t]["templates"]
    assert man["n_requests_per_model"] == 13100 and man["n_cells"] == 131
    for t in cw.TASKS:
        assert all(r["request_id"].startswith("train|") for r in man["prefill"]["rows"][t])
        assert all(r["config_id"] in man["tasks"][t]["templates"] for r in man["prefill"]["rows"][t])
        assert len(man["pilot"]["rows"][t]) == 5 and len(man["prefill"]["rows"][t]) == 10
    assert man["control"]["row"]["request_id"].startswith("train|") and man["control"]["row"]["level"] == "Level 5"
    # the arms: the same cells, 13,100 calls each, the forced message equal to the requested one
    a = man["arms"]
    assert a["n_requests_per_arm"] == {"requested": 13100, "forced": 13100, "cot": 13100}
    for t in cw.TASKS:
        assert a["arms"]["forced"][t]["user_message_equals_requested"] is True
        for arm in ("forced", "cot"):
            cells = [(r["config_id"], r["item_id"]) for r in a["pilot"][arm][t]]
            assert cells == [(r["config_id"], r["item_id"]) for r in man["pilot"]["rows"][t]]
    assert [s["path"] for s in a["arms"]["forced"]["hotpot"]["sources"]] == [
        "data/requests/hotpot/block_a/qwen3-8b.parquet", "data/requests/hotpot/block_b/qwen3-8b.parquet"]
    assert a["stop_control"]["benchmarks"] == ["bbh", "gsm8k"]


@needs_data
def test_plan_rebuilds_the_same_manifest(tmp_path, cached_requests):
    paths = cw.Paths(CFG, ROOT, manifest=tmp_path / "m.json")
    logs = []
    assert cw.main(["plan"], paths=paths, log=logs.append) == 0
    new = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    assert cw.manifest_body(new) == cw.manifest_body(old)
    assert any("39,300 in all" in x for x in logs)
    # a manifest that differs is never overwritten
    bad = dict(old, tasks={**old["tasks"], "gsm8k": {**old["tasks"]["gsm8k"], "items": ["0"]}})
    (tmp_path / "m.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit):
        cw.main(["plan"], paths=paths, log=logs.append)


@needs_data
def test_plan_adds_the_arms_section_without_touching_the_items(tmp_path, cached_requests):
    old = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    first = {k: v for k, v in old.items() if k not in ("arms", "arms_added_on")}      # the requested-arm manifest
    (tmp_path / "m.json").write_text(json.dumps(first), encoding="utf-8")
    paths = cw.Paths(CFG, ROOT, manifest=tmp_path / "m.json")
    logs = []
    assert cw.main(["plan"], paths=paths, log=logs.append) == 0
    new = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert any("arms section added" in x for x in logs)
    assert cw.base_body(new) == cw.base_body(first) and new["arms"] == old["arms"]
    assert new["made_on"] == first["made_on"]
    n0 = len(logs)
    assert cw.main(["plan"], paths=paths, log=logs.append) == 0 and any("unchanged" in x for x in logs[n0:])
    changed = dict(new, arms={**new["arms"], "n_requests_per_arm": {}})
    (tmp_path / "m.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit):
        cw.main(["plan"], paths=paths, log=logs.append)


# ------------------------------------------------------------------------------------------------ replies, money

def test_parse_reply_outcomes():
    b = {"model": "qwen/qwen4-max", "reasoning": {"effort": "none"}}
    r = cw.parse_reply(reply(b, text="\\boxed{3}", ct=4))
    assert r["outcome"] == "reply" and r["completion_tokens"] == 4 and r["reasoning_tokens"] == 0
    assert r["provider"] == "Alibaba" and r["generation_id"] == "gen-test" and r["cost_reported"] > 0
    assert cw.parse_reply(reply(b, text=None, refusal="I can't help with that."))["outcome"] == "refusal"
    assert cw.parse_reply(reply(b, text="", finish="content_filter"))["outcome"] == "refusal"
    blocked = {"status": 403, "json": {"error": {"code": 403, "message": "Input was flagged",
                                                 "metadata": {"reasons": ["x"], "flagged_input": "..."}}}}
    assert cw.parse_reply(blocked)["outcome"] == "blocked"
    ali = {"status": 403, "json": {"error": {"message": "Upstream error from Alibaba: Output data may contain "
                                              "inappropriate content.", "code": 403,
                                              "metadata": {"provider_name": "Alibaba",
                                                           "provider_error_code": "data_inspection_failed"}}}}
    assert cw.parse_reply(ali)["outcome"] == "blocked"
    assert cw.parse_reply({"status": 403, "json": {"error": {"message": "forbidden"}}})["outcome"] == "error"
    assert cw.parse_reply({"status": 429, "json": {"error": {"code": 429}}})["outcome"] == "failed"
    assert cw.parse_reply({"status": 502, "json": None, "error": "bad gateway"})["outcome"] == "failed"
    assert cw.parse_reply({"status": None, "json": None, "error": "timeout"})["outcome"] == "failed"
    assert cw.parse_reply({"status": 400, "json": {"error": {"message": "bad"}}})["outcome"] == "error"
    assert cw.parse_reply({"status": 404, "json": {"error": {"message": "no endpoints"}}})["outcome"] == "error"
    mid = reply(b)
    mid["json"]["choices"][0]["finish_reason"] = "error"
    assert cw.parse_reply(mid)["outcome"] == "failed"
    p = cw.parse_reply(reply(b, pt=1000, ct=10, cached=400, cost=None))
    want = 600 * 1.425e-06 + 400 * 7.125e-08 + 10 * 4.275e-06
    assert cw.billed_cost(p, QWEN["price"]) == (pytest.approx(want), "tokens x config prices")
    p2 = cw.parse_reply(reply(b, cost=0.123))
    assert cw.billed_cost(p2, QWEN["price"]) == (0.123, "openrouter usage.cost")


def test_upper_cost_bounds_a_call():
    req = cw.make_request(CFG, QWEN, "hotpot", row(text="é" * 4000))
    assert req["n_bytes"] == 8000
    assert req["reserve_usd"] == pytest.approx((8000 + 64) * QWEN["price"]["prompt"] + 1024 * QWEN["price"]["completion"])
    c = cw.make_request(CFG, QWEN, "math", row(), "control", rep=0)
    assert c["reserve_usd"] > 32768 * QWEN["price"]["completion"]


def _runner(tmp_path, http, limit, counted=0.0, **kw):
    return cw.Runner(http, CFG, tmp_path / "raw.jsonl", "run", limit, counted, "sha", sleep=lambda s: None,
                     log=lambda *a: None, **kw)


def three_arm_requests(n=4, task="gsm8k"):
    out = []
    for i in range(n):
        for arm, rid in (("requested", f"c|direct_answer|{i}"), ("forced", f"c|direct_answer|{i}|fd"),
                         ("cot", f"c|zero_shot_cot|{i}")):
            out.append(cw.make_request(CFG, QWEN, task, row(rid=rid, item=str(i), cfgid="c"), arm))
    return out


def test_a_chunk_that_could_cross_the_limit_is_not_sent(tmp_path):
    reqs = [cw.make_request(CFG, QWEN, "gsm8k", row(rid=f"r{i}", item=str(i))) for i in range(10)]
    one = reqs[0]["reserve_usd"]
    http = FakeHttp()
    s = _runner(tmp_path, http, limit=one * 0.99).run(reqs)
    assert http.posts == [] and "spend limit" in s["stop_reason"]
    http = FakeHttp(responder=lambda b, n: {"status": None, "json": None, "error": "timeout"})
    r = _runner(tmp_path, http, limit=one * 3.5, counted=0.0, chunk=10)
    s = r.run(reqs)
    assert len(http.posts) == 3 and r.counted <= one * 3.5 and "spend limit" in s["stop_reason"]
    led = cw.ledger(cw.read_raw(tmp_path / "raw.jsonl"))
    assert led["counted_usd"] == pytest.approx(3 * one) and led["orphans"] == 0
    http = FakeHttp(responder=lambda b, n: {"status": None, "json": None, "error": "timeout"})
    r = _runner(tmp_path, http, limit=1.0, counted=1.0 - 2.5 * one, chunk=10)
    r.run(reqs)
    assert len(http.posts) == 2 and r.counted <= 1.0
    http = FakeHttp()
    r = _runner(tmp_path, http, limit=one * 2.5, chunk=10)
    r.run(reqs)
    assert len(http.posts) == 10 and r.counted <= one * 2.5


def test_the_limit_guard_over_three_arms(tmp_path):
    """Each arm reserves its own worst case (CoT 3,072 tokens, requested 1,024, forced 16); a chunk is cut at the
    first call whose worst case would pass the limit, whatever its arm, and nothing counted passes the limit."""
    reqs = three_arm_requests(4)
    res = [r["reserve_usd"] for r in reqs]
    assert res[2] > res[0] > res[1]                                    # cot > requested > forced
    # room for the first unit and the next requested and forced calls, not the next chain of thought
    limit = sum(res[:5]) + 0.5 * res[5]
    fit = _runner(tmp_path, FakeHttp(), limit=limit).fit([dict(x, tries=0) for x in reqs])
    assert [x["arm"] for x in fit] == ["requested", "forced", "cot", "requested", "forced"]
    once = json.loads(json.dumps(CFG))
    once["run"]["max_tries"] = 1                                       # no retry: each failed call counted once
    http = FakeHttp(responder=lambda b, n: {"status": None, "json": None, "error": "timeout"})
    r = cw.Runner(http, once, tmp_path / "raw.jsonl", "run", limit, 0.0, "sha", sleep=lambda s: None,
                  log=lambda *a: None, chunk=12)
    s = r.run(reqs)
    assert len(http.posts) == 5 and "spend limit" in s["stop_reason"] and r.counted <= limit
    assert sorted(body_arm(b) for b in http.bodies()) == sorted(["requested", "forced", "cot", "requested", "forced"])
    led = cw.ledger(cw.read_raw(tmp_path / "raw.jsonl"))
    assert led["counted_usd"] == pytest.approx(sum(res[:5]))           # no reply: each counts at its worst case
    # with what the pilot and the prefill test counted, less room: the same rule over the three arms
    http = FakeHttp(responder=lambda b, n: {"status": None, "json": None, "error": "timeout"})
    r = cw.Runner(http, once, tmp_path / "x" / "raw.jsonl", "run", 1.0, 1.0 - (res[0] + res[1] + 0.5 * res[2]), "sha",
                  sleep=lambda s: None, log=lambda *a: None, chunk=12)
    r.run(reqs)
    assert len(http.posts) == 2 and r.counted <= 1.0
    # real bills far below the worst case: every call of the three arms goes out, never past the limit
    http = FakeHttp()
    r = _runner(tmp_path / "y", http, limit=res[2] * 1.5, chunk=3)
    s = r.run(reqs)
    assert len(http.posts) == 12 and s["stop_reason"] is None and r.counted <= res[2] * 1.5


def test_retries_only_technical_failures_and_stop_on_errors(tmp_path):
    reqs = [cw.make_request(CFG, QWEN, "gsm8k", row(rid=f"r{i}", item=str(i))) for i in range(3)]
    seq = {1: 429, 2: 503}

    def flaky(body, n):
        st = seq.get(n)
        return {"status": st, "json": {"error": {"code": st}}} if st else good(body, n)
    http = FakeHttp(responder=flaky)
    s = _runner(tmp_path, http, limit=1.0, chunk=1, threads=1).run(reqs)
    assert s["answered"] == 3 and s["failed_attempts"] == 2 and s["stop_reason"] is None and len(http.posts) == 5
    http = FakeHttp(responder=lambda b, n: {"status": 400, "json": {"error": {"message": f"bad {SECRET}"}}})
    s = _runner(tmp_path, http, limit=1.0, chunk=1, threads=1).run(reqs)
    assert len(http.posts) == 1 and s["stop_reason"].startswith("HTTP 400") and SECRET not in s["stop_reason"]


def test_restart_skips_done_keys_and_counts_orphans(tmp_path):
    reqs = three_arm_requests(3)
    http = FakeHttp()
    _runner(tmp_path, http, limit=1.0).run(reqs[:4])
    # a crash after a "sent" line: the attempt is not answered and counts at its worst case
    with open(tmp_path / "raw.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "sent", "key": reqs[4]["key"], "attempt_id": "orphan",
                             "reserve_usd": reqs[4]["reserve_usd"]}) + "\n")
        fh.write('{"event": "result", "key": "cut-mid-wri')
    lines = cw.read_raw(tmp_path / "raw.jsonl")
    done = cw.done_keys(lines, "sha")
    assert done == {r["key"] for r in reqs[:4]} and cw.done_keys(lines, "other") == set()
    led = cw.ledger(lines)
    assert led["orphans"] == 1 and led["orphans_usd"] == pytest.approx(reqs[4]["reserve_usd"])
    http2 = FakeHttp()
    _runner(tmp_path, http2, limit=1.0, counted=led["counted_usd"]).run([r for r in reqs if r["key"] not in done])
    assert len(http2.posts) == 5
    lines = cw.read_raw(tmp_path / "raw.jsonl")
    keys = [x["key"] for x in lines if x["event"] == "result"]
    assert len(keys) == len(set(keys)) == 9 and set(keys) == {r["key"] for r in reqs}
    # the three arms of one cell are three keys; the arm is in every result line
    assert {x["arm"] for x in lines if x["event"] == "result"} == {"requested", "forced", "cot"}
    assert all(x["stop_sent"] == (x["arm"] == "forced") for x in lines if x["event"] == "result")


# ------------------------------------------------------------------------------------------------ forced reading

def test_forced_reply_reading():
    fr = cw.forced_reply
    # GSM8K and BIG-Bench Hard: stop sent; the route leaves the stop string out: the "}" is put back
    assert fr("18", "gsm8k", "stop", CFG, True) == ("18}", "stop_restored")
    assert fr("(B)", "bbh", "stop", CFG, True) == ("(B)}", "stop_restored")
    assert fr("", "gsm8k", "stop", CFG, True) == ("}", "stop_restored")
    # a route that returns the stop string, or a reply sent without the stop: cut after the first "}"
    assert fr("18}", "gsm8k", "stop", CFG, True) == ("18}", "as_returned")
    assert fr("18} because 6 x 3", "gsm8k", "length", CFG, False) == ("18}", "cut_after_first_brace")
    assert fr("\\text{C}", "bbh", "stop", CFG, True) == ("\\text{C}", "as_returned")     # forced_stop reads it later
    # cut by the cap: as it is
    assert fr("12345678901234", "gsm8k", "length", CFG, True) == ("12345678901234", "as_returned")
    # MATH-500 and HotpotQA ran without the stop: never cut, never completed
    assert fr("\\frac{1}{2}} so", "math", "length", CFG, False) == ("\\frac{1}{2}} so", "as_returned")
    assert fr("Paris", "hotpot", "stop", CFG, False) == ("Paris", "as_returned")


@needs_data
def test_forced_scoring_counts_the_prefix_once(monkeypatch):
    """A forced reply goes to the panel's scorer without the prefill (as the panel's runner stores it); the scorer
    puts the prefill in front once, and the real readers score it as the panel's."""
    from src import bbh_d2, gsm8k, math_d2
    import nebius_panel_report as npr
    g = gsm8k.load_gsm8k("test")
    g0 = str(g.loc[g["item_id"] == 0, "gold_answer"].iloc[0])
    mi, _ = math_d2.load_items("target")
    bi, _ = bbh_d2.load_items("target")
    m0, b0 = mi.iloc[0], bi.iloc[0]
    rows = [  # (arm, task, item, text, finish, stop_sent, expected content)
        ("forced", "gsm8k", "0", g0, "stop", True, 1),                       # stop left out by the route
        ("forced", "gsm8k", "0", g0 + "}", "stop", True, 1),                 # stop returned
        ("forced", "gsm8k", "0", g0 + "} and more", "length", False, 1),     # the stop control: no stop sent
        ("forced", "gsm8k", "0", "99999", "stop", True, 0),
        ("requested", "gsm8k", "0", f"\\boxed{{{g0}}}", "stop", False, 1),
        ("cot", "gsm8k", "0", f"Working.\n\\boxed{{{g0}}}", "stop", False, 1),
        ("forced", "math", str(m0["item_id"]), f"{m0['gold_answer']}}}", "stop", False, 1),
        ("forced", "bbh", str(b0["item_id"]), f"{b0['gold_answer']}", "stop", True, 1),
    ]
    df = pd.DataFrame([{"key": f"k{i}", "model_key": "qwen4-max", "arm": a, "task": t, "request_id": "same-id",
                        "item_id": it, "text": tx, "finish_reason": f, "stop_sent": s}
                       for i, (a, t, it, tx, f, s, _) in enumerate(rows)])
    seen = []
    real = npr.score_task

    def spy(o, task):
        seen.append(o.copy())
        return real(o, task)
    monkeypatch.setattr(npr, "score_task", spy)
    out, ver = cw.score_rows(df, CFG)
    assert list(out["content_correct"].astype(int)) == [r[6] for r in rows]
    assert out.loc[0, "strict_correct"] == out.loc[4, "strict_correct"] == 1
    assert list(out["forced_reading"].iloc[:4]) == ["stop_restored", "as_returned", "cut_after_first_brace",
                                                    "stop_restored"]
    allo = pd.concat(seen)
    fo = allo[allo["arm"] == "forced_direct"]
    assert not fo["output_text"].str.startswith(PREFIX).any()            # stored without the prefill
    texts = npr.scoring_text(fo)
    assert all(t.count(PREFIX) == 1 and t.startswith(PREFIX) for t in texts)   # the prefill once, in front
    assert set(allo["arm"]) == {"forced_direct", "direct_answer", "zero_shot_cot"}
    assert ver["gsm8k"]["forced_stop"].startswith("forced-stop/")
    # a reply that restarts the box ("\boxed{18}" after the prefill) would carry the prefix twice: the pilot fails it
    c = cw.forced_check("\\boxed{18}", "gsm8k", "stop", CFG, True)
    assert not c["pass"] and c["starts_with_prefill"]


def test_forced_check_follows_the_prefill_rule():
    assert cw.forced_check("18", "gsm8k", "stop", CFG, True)["pass"]                  # stop left out, put back
    assert cw.forced_check("18", "gsm8k", "stop", CFG, True)["scoring_text"] == "\\boxed{18}"
    assert cw.forced_check("Paris", "hotpot", "stop", CFG, False)["pass"]
    assert not cw.forced_check("Let me think.\nFirst", "math", "length", CFG, False)["pass"]
    assert not cw.forced_check("<think>x", "bbh", "length", CFG, True)["pass"]
    assert not cw.forced_check("18", "gsm8k", "stop", CFG, True, completion_tokens=17)["pass"]
    assert not cw.forced_check("18", "gsm8k", "stop", CFG, True, reasoning_tokens=5)["pass"]


# ------------------------------------------------------------------------------------------------ key handling

def test_http_sends_the_key_only_in_the_header_and_redacts_it():
    seen = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        seen.append(req)
        if req.get_method() == "GET":
            return Resp(b'{"data": {"endpoints": []}}')
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {},
                                     io.BytesIO(f'{{"error": "bad key {SECRET}"}}'.encode()))
    h = cw.Http(CFG["base_url"], SECRET, opener=opener)
    r = h.post_json("/chat/completions", b"{}")
    assert r["status"] == 401 and SECRET not in r["error"] and "REDACTED" in r["error"]
    assert seen[0].get_header("Authorization") == "Bearer " + SECRET
    h.get_public("https://openrouter.ai/api/v1/models/x/endpoints")
    assert seen[1].get_header("Authorization") is None
    assert SECRET not in cw.redact(f"Authorization: Bearer {SECRET} and {SECRET}", None)


@needs_data
def test_key_never_printed_or_written(tmp_path, capsys, cached_requests):
    paths = tmp_paths(tmp_path)
    key = cw.read_key(paths.env)
    assert key == SECRET

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = {"n": 0}

    def opener(req, timeout=None):
        if req.get_method() == "GET":
            return Resp(json.dumps(listing(req.full_url)["json"]).encode())
        assert req.get_header("Authorization") == "Bearer " + SECRET
        calls["n"] += 1
        body = json.loads(req.data.decode())
        if calls["n"] == 1:      # an error body that echoes the key: retried, redacted in the raw file
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(f'{{"k": "{SECRET}"}}'.encode()))
        return Resp(json.dumps(good(body, calls["n"])["json"]).encode())
    http = cw.Http(CFG["base_url"], key, opener=opener)
    logs = []

    def log(*a):
        logs.append(" ".join(map(str, a)))
        print(*a)
    rc = cw.main(["pilot", "--approved", "--limit-usd", "2"], http=http, scorer=fake_scorer, tok=FakeTok(),
                 paths=paths, sleep=lambda s: None, log=log)
    assert rc == 0, logs[-5:]
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err + "\n".join(logs)
    blob = files_text(tmp_path / "raw", tmp_path / "res") + (tmp_path / "budget.yaml").read_text(encoding="utf-8")
    assert SECRET not in blob and "REDACTED" in blob


# ------------------------------------------------------------------------------------------------ pilot

def _pilot(tmp_path, http, **kw):
    paths = kw.get("paths") or tmp_paths(tmp_path)
    logs = []
    rc = cw.main(["pilot", "--approved", "--limit-usd", "2"] + kw.get("extra", []), http=http,
                 scorer=kw.get("scorer", fake_scorer), tok=FakeTok(), paths=paths, sleep=lambda s: None,
                 log=logs.append)
    res = json.loads((paths.results / "closed_w_pilot.json").read_text(encoding="utf-8"))
    return rc, res, paths, logs


@needs_data
def test_pilot_passes_and_logs_the_budget(tmp_path, cached_requests):
    http = FakeHttp()
    rc, res, paths, logs = _pilot(tmp_path, http)
    assert rc == 0 and res["passed"], res.get("failed")
    # 5 prompts x 4 benchmarks x 3 arms x 2 sends, 10 stop controls, 2 thinking-on controls; one listing read
    assert len(http.posts) == 132 and len(http.gets) == 1
    bodies = http.bodies()
    assert all(b["model"] == "qwen/qwen4-max" for b in bodies)
    arms = pd.Series([body_arm(b) for b in bodies if not thinking_on(b)]).value_counts().to_dict()
    assert arms == {"requested": 40, "forced": 50, "cot": 40}
    nostop = [b for b in bodies if body_arm(b) == "forced" and "stop" not in b and b["max_tokens"] == 16]
    assert len(nostop) == 5 + 5 + 10                       # MATH-500 forced (2 x 5) and the 10 stop controls
    for b in bodies:
        m = BY_ID[b["model"]]
        assert b["provider"] == m["provider"] and b["temperature"] == 0.0
        if body_arm(b) == "forced":
            assert b["messages"][-1] == {"role": "assistant", "content": PREFIX, "partial": True}
        else:
            assert b["messages"][0]["role"] == "user" and len(b["messages"]) == 1
    assert sum(thinking_on(b) for b in bodies) == 2
    names = {c["name"] for c in res["models"]["qwen4-max"]["checks"]}
    assert {"preflight", "answered", "reasoning_off", "reasoning_on_control", "provider", "finish_reasons",
            "token_accounting", "repeat_agreement", "cost_per_call", "forced_continuation",
            "forced_stop_control"} <= names
    chk = {c["name"]: c for c in res["models"]["qwen4-max"]["checks"]}
    assert set(chk["repeat_agreement"]["detail"]["per_arm"]) == {"requested", "forced", "cot"}
    assert chk["forced_stop_control"]["detail"]["pairs"] == 10
    assert chk["forced_stop_control"]["detail"]["stop_restored"] == 10
    pj = res["projection"]["models"]["qwen4-max"]
    assert set(pj["per_arm"]) == {"requested", "forced", "cot"} and res["projection"]["total"] > 0
    md = (paths.results / "closed_w_pilot.md").read_text(encoding="utf-8")
    assert "PASS" in md and (paths.results / "closed_w_pilot_clean.md").exists()
    bud = yaml.safe_load((paths.budget).read_text(encoding="utf-8"))
    assert "Qwen4-Max three arms on OpenRouter, pilot" in bud["log"][-1]["run"]
    assert bud["planned"] == yaml.safe_load((ROOT / "config" / "budget.yaml").read_text(encoding="utf-8"))["planned"]
    http2 = FakeHttp()
    rc2, res2, _, _ = _pilot(tmp_path, http2, paths=paths)
    assert rc2 == 0 and http2.posts == []


@needs_data
def test_pilot_with_the_real_scorers_and_tokenizer(tmp_path, cached_requests):
    if not cw.Tok(CFG).path.exists():
        pytest.skip("tokenizer absent")
    paths = tmp_paths(tmp_path)
    http = FakeHttp()
    logs = []
    rc = cw.main(["pilot", "--approved", "--limit-usd", "2"], http=http, paths=paths, sleep=lambda s: None,
                 log=logs.append)
    res = json.loads((paths.results / "closed_w_pilot.json").read_text(encoding="utf-8"))
    assert rc == 0 and res["passed"], res.get("failed")
    rep = {c["name"]: c for c in res["models"]["qwen4-max"]["checks"]}["repeat_agreement"]["detail"]["per_arm"]
    assert all(rep[a]["pairs"] == 20 and rep[a]["strict"] == 1.0 and rep[a]["content"] == 1.0 for a in rep)


@needs_data
def test_panel_table_reads_the_same_cells():
    man = json.loads((ROOT / CFG["paths"]["manifest"]).read_text(encoding="utf-8"))
    t = man["tasks"]["math"]
    df = cw.panel_table("math", t["items"], t["templates"])
    n = df.groupby("model_key").size()
    assert len(n) >= 20 and (n == len(t["items"]) * len(t["templates"])).all()
    assert df["n_output_tokens"].notna().all() and set(df["content_correct"].unique()) <= {0, 1}
    d3 = cw.panel_table3("math", t["items"], t["templates"])
    k = d3.groupby(["model_key", "arm"]).size().unstack()
    assert set(k.columns) == {"requested", "forced", "cot"} and (k == 3100).all().all() and len(k) >= 20
    assert d3["n_output_tokens"].notna().all()
    assert d3.loc[d3["arm"] == "forced", "n_output_tokens"].max() <= 16


@needs_data
@pytest.mark.parametrize("fault,check", [
    ("reasoning", "reasoning_off"), ("control", "reasoning_on_control"), ("provider", "provider"),
    ("finish", "finish_reasons"), ("cost", "cost_per_call"), ("tokens", "token_accounting"),
    ("repeat", "repeat_agreement")])
def test_pilot_check_failures_stop_the_run(tmp_path, cached_requests, fault, check):
    seen, lock = {}, threading.Lock()

    def bad(body, n):
        on = thinking_on(body)
        if fault == "reasoning" and not on:
            return reply(body, rt=3)
        if fault == "control" and on:
            return reply(body, rt=0)
        if fault == "provider":
            return reply(body, provider="Azure", rt=200 if on else 0)
        if fault == "finish" and not on:
            return reply(body, finish="tool_calls")
        if fault == "cost":
            r = good(body, n)
            r["json"]["usage"]["cost"] *= 2                     # e.g. routed to a dearer endpoint
            return r
        if fault == "tokens" and not on:
            return reply(body, ct=400)                          # 400 tokens counted for a 5-token text
        if fault == "repeat" and not on and body_arm(body) == "cot":
            k = json.dumps(body["messages"])
            with lock:
                seen[k] = seen.get(k, 0) + 1
            if seen[k] == 2:
                return reply(body, text="I think the answer is \\boxed{17} " + "x " * 30, ct=40)
        return good(body, n)
    rc, res, paths, logs = _pilot(tmp_path, FakeHttp(responder=bad))
    assert rc == 3 and not res["passed"]
    failed = {c["name"] for mk in res["models"].values() for c in mk["checks"] if not c["pass"]}
    assert check in failed
    assert any("failed" in x for x in logs)
    http = FakeHttp()
    rc = cw.main(["run", "--approved", "--limit-usd", "25"], http=http, paths=paths, sleep=lambda s: None,
                 log=lambda *a: None)
    assert rc == 2 and http.posts == []


@needs_data
@pytest.mark.parametrize("fault,check", [
    ("no_partial", "forced_continuation"),       # the route drops partial mode: every reply restarts the box
    ("linebreak", "forced_continuation"),        # the model writes working inside the box
    ("echo_unstopped", "forced_continuation"),   # the prefill echoed where no stop is sent (MATH-500, controls)
    ("stop_mismatch", "forced_stop_control"),    # with the stop, another answer than without it
    ("forced_400", "answered"),                  # the forced body refused (e.g. partial or stop not accepted)
])
def test_pilot_forced_arm_failures_stop_the_run(tmp_path, cached_requests, fault, check):
    def bad(body, n):
        if thinking_on(body) or body_arm(body) != "forced":
            return good(body, n)
        if fault == "no_partial":
            return reply(body, text="\\boxed{18}", ct=6)
        if fault == "linebreak":
            return reply(body, text="Let me work\nthrough it", ct=8, finish="length")
        if fault == "echo_unstopped" and body["max_tokens"] == 16 and "stop" not in body:
            return reply(body, text="\\boxed{18}", ct=6)
        if fault == "stop_mismatch" and "stop" not in body and body["max_tokens"] == 16:
            return reply(body, text="17}", ct=3)
        if fault == "forced_400":
            return {"status": 400, "json": {"error": {"message": "partial is not supported"}}}
        return good(body, n)
    rc, res, paths, logs = _pilot(tmp_path, FakeHttp(responder=bad))
    assert rc == 3 and not res["passed"]
    failed = {c["name"] for mk in res["models"].values() for c in mk["checks"] if not c["pass"]}
    assert check in failed, failed
    http = FakeHttp()
    rc = cw.main(["run", "--approved", "--limit-usd", "100"], http=http, paths=paths, sleep=lambda s: None,
                 log=lambda *a: None)
    assert rc == 2 and http.posts == []


@needs_data
def test_preflight_failure_stops_before_any_paid_call(tmp_path, cached_requests):
    http = FakeHttp(list_fn=lambda u: listing(u, params=["max_tokens", "temperature", "top_p", "reasoning", "stop"]))
    rc, res, paths, logs = _pilot(tmp_path, http)
    assert rc == 3 and http.posts == [] and not res["passed"]
    assert "presence_penalty" in json.dumps(res["preflight"]["qwen4-max"])
    assert "gpt-6-luna" not in res["preflight"]                         # off by default
    # the forced arm sends the stop: an endpoint that does not list it fails before any paid call
    http = FakeHttp(list_fn=lambda u: listing(u, params=["max_tokens", "temperature", "top_p", "reasoning",
                                                         "presence_penalty"]))
    rc, res, _, _ = _pilot(tmp_path, http, paths=tmp_paths(tmp_path))
    assert rc == 3 and http.posts == [] and "'stop'" in json.dumps(res["preflight"]["qwen4-max"])
    http = FakeHttp(list_fn=lambda u: listing(u, price_factor=2.0))
    rc, res, _, _ = _pilot(tmp_path, http, paths=tmp_paths(tmp_path))
    assert rc == 3 and http.posts == []


def test_paid_phases_refuse_without_approval_and_limit(tmp_path):
    paths = tmp_paths(tmp_path, env=False)
    for argv in (["pilot"], ["pilot", "--limit-usd", "1"], ["pilot", "--approved"],
                 ["prefill-test", "--approved"], ["run", "--limit-usd", "25"],
                 ["run", "--approved", "--limit-usd", "0"], ["run", "--approved"],
                 ["pilot", "--approved", "--limit-usd", "2.5"],
                 ["prefill-test", "--approved", "--limit-usd", "0.6"]):
        http = FakeHttp()
        logs = []
        assert cw.main(argv, http=http, paths=paths, log=logs.append) == 2, argv
        assert http.posts == [] and http.gets == [] and logs[0].startswith("refused")
    # no run ceiling in the config: the given limit is taken as is (here it still refuses: no pilot)
    a = type("A", (), {"approved": True, "limit_usd": 500.0})()
    assert cw.refuse_paid(a, CFG, "run") is None
    assert cw.refuse_paid(a, CFG, "pilot") is not None


# ------------------------------------------------------------------------------------------------ run

@needs_data
def test_run_end_to_end_stops_at_the_limit_and_resumes(tmp_path, cached_requests):
    rc, res, paths, _ = _pilot(tmp_path, FakeHttp())
    assert rc == 0
    pilot_spend = cw.ledger(cw.read_raw(paths.raw("pilot")))["counted_usd"]

    def long(body, n):
        a = body_arm(body)
        if a == "forced":
            return good(body, n)
        return reply(body, text="Let me work it out. " * 60 + "\\boxed{18}", ct=900 if a == "requested" else 2500)
    http = FakeHttp(responder=long)
    logs = []
    limit = pilot_spend + 0.06
    rc = cw.main(["run", "--approved", "--limit-usd", f"{limit:.6f}", "--threads", "1"], http=http, paths=paths,
                 sleep=lambda s: None, log=logs.append)
    assert rc == 3 and any("spend limit" in x for x in logs)
    n1 = len(http.posts)
    assert n1 > 0
    spent = cw.counted_spend(paths)
    assert sum(v["counted_usd"] for v in spent.values()) <= limit + 1e-9
    keys1 = [x["key"] for x in cw.read_raw(paths.raw("run")) if x["event"] == "result"]
    # the arms of a unit go out back to back: requested, forced, chain of thought
    assert [body_arm(b) for b in http.bodies()[:6]] == ["requested", "forced", "cot"] * 2
    # resume with more room: nothing answered is sent again
    http2 = FakeHttp(responder=long)
    cw.main(["run", "--approved", "--limit-usd", f"{limit + 0.05:.6f}"], http=http2, paths=paths,
            sleep=lambda s: None, log=lambda *a: None)
    lines = cw.read_raw(paths.raw("run"))
    keys2 = [x["key"] for x in lines if x["event"] == "result"]
    assert len(keys2) > len(keys1) and len(set(keys2)) == len(keys2)
    for b in http.bodies() + http2.bodies():
        a = body_arm(b)
        if a == "forced":
            assert b["max_tokens"] in (16, 64) and b["messages"][-1]["partial"] is True
        else:
            assert len(b["messages"]) == 1 and b["max_tokens"] in ((1024,) if a == "requested" else (3072, 1900))
        assert b["reasoning"] == {"effort": "none"}
    bud = yaml.safe_load(paths.budget.read_text(encoding="utf-8"))
    assert [e["run"].split(", ")[1].split(" ")[0] for e in bud["log"][-3:]] == ["pilot", "run", "run"]
    # --arms sends only those arms
    http3 = FakeHttp(responder=long)
    cw.main(["run", "--approved", "--limit-usd", f"{limit + 0.08:.6f}", "--arms", "forced"], http=http3,
            paths=paths, sleep=lambda s: None, log=lambda *a: None)
    assert http3.posts and all(body_arm(b) == "forced" for b in http3.bodies())
    # a reply with reasoning or from another route stops the run
    http4 = FakeHttp(responder=lambda b, n: reply(b, rt=7))
    rc = cw.main(["run", "--approved", "--limit-usd", "25"], http=http4, paths=paths, sleep=lambda s: None,
                 log=lambda *a: None)
    assert rc == 3 and len(http4.posts) <= CFG["run"]["chunk_calls"]


@needs_data
def test_run_refuses_without_the_passing_prefill_variant(tmp_path, cached_requests):
    rc, res, paths, _ = _pilot(tmp_path, FakeHttp())
    assert rc == 0
    pilot_spend = cw.ledger(cw.read_raw(paths.raw("pilot")))["counted_usd"]
    for content in (None, {"summary": {"passed": True, "passing_variants": ["assistant_prefix"],
                                       "variants": {"assistant_prefix": {"message_extra": {}}}}}):
        if content is None:
            paths.prefill_result.unlink()
        else:
            paths.prefill_result.write_text(json.dumps(content), encoding="utf-8")
        http, logs = FakeHttp(), []
        rc = cw.main(["run", "--approved", "--limit-usd", "25"], http=http, paths=paths, sleep=lambda s: None,
                     log=logs.append)
        assert rc == 2 and http.posts == [] and "prefill" in logs[-1]
    # the requested and chain-of-thought arms alone do not need it
    http = FakeHttp()
    rc = cw.main(["run", "--approved", "--limit-usd", f"{pilot_spend + 0.03:.6f}", "--arms", "requested", "cot"],
                 http=http,
                 paths=paths, sleep=lambda s: None, log=lambda *a: None)
    assert http.posts and all(body_arm(b) != "forced" for b in http.bodies())


@needs_data
def test_run_stops_when_forced_replies_restart_the_box(tmp_path, cached_requests):
    rc, res, paths, _ = _pilot(tmp_path, FakeHttp())
    assert rc == 0

    def drift(body, n):                  # the route stops honouring partial mode after the pilot
        if body_arm(body) == "forced":
            return reply(body, text="\\boxed{18}", ct=6)
        return good(body, n)
    http, logs = FakeHttp(responder=drift), []
    rc = cw.main(["run", "--approved", "--limit-usd", "25"], http=http, paths=paths, sleep=lambda s: None,
                 log=logs.append)
    assert rc == 3 and any("start with the prefill" in x for x in logs)
    assert len(http.posts) <= 2 * CFG["run"]["chunk_calls"]


def test_forced_monitor():
    m = cw.ForcedMonitor(CFG)
    ok = {"kind": "forced", "outcome": "reply", "text": "18"}
    bad = {"kind": "forced", "outcome": "reply", "text": "\\boxed{18}"}
    for _ in range(36):
        assert m(ok) is None
    for _ in range(4):
        assert m(bad) is None                          # 4 of 40: within 10%
    assert m(bad) is not None                          # 5 of 41
    assert cw.ForcedMonitor(CFG)({"kind": "requested", "outcome": "reply", "text": "\\boxed{1}"}) is None


# ------------------------------------------------------------------------------------------------ prefill

def test_prefill_continuation_detection():
    ok = cw.prefill_check("42}", "gsm8k", CFG)
    assert ok["pass"] and ok["closed"] and ok["answer"] == "42" and ok["scoring_text"] == "\\boxed{42}"
    cut = cw.prefill_check("42} because 6 x 7", "bbh", CFG)
    assert cut["pass"] and cut["scoring_text"] == "\\boxed{42}"
    frac = cw.prefill_check("\\frac{1}{2}}", "math", CFG)
    assert frac["pass"] and frac["answer"] == "\\frac{1}{2}" and frac["scoring_text"] == "\\boxed{\\frac{1}{2}}"
    restart = cw.prefill_check("\\boxed{42}", "gsm8k", CFG)
    assert not restart["pass"] and any("restart" in r for r in restart["reasons"])
    echo = cw.prefill_check("\\boxed{42}", "gsm8k", CFG, echo_mode=True)
    assert echo["pass"] and echo["echo_removed"] and echo["scoring_text"].count("\\boxed") == 1
    assert not cw.prefill_check("<think>let me see", "gsm8k", CFG)["pass"]
    assert not cw.prefill_check("Let me think.\nFirst 3", "math", CFG)["pass"]
    assert not cw.prefill_check("}", "gsm8k", CFG)["pass"]
    assert not cw.prefill_check("42}", "gsm8k", CFG, reasoning_tokens=12)["pass"]
    assert not cw.prefill_check("42}", "gsm8k", CFG, completion_tokens=17)["pass"]
    long = cw.prefill_check("\\text{The Beatles' White Album", "hotpot", CFG, completion_tokens=64)
    assert long["pass"] and not long["closed"]


@needs_data
def test_prefill_test_end_to_end(tmp_path):
    def resp(body, n):
        last = body["messages"][-1]
        assert last["role"] == "assistant" and last["content"] == PREFIX and "stop" not in body
        if last.get("partial"):
            return {"status": 400, "json": {"error": {"message": "unknown field partial"}}}
        return reply(body, text="18}", ct=3)
    http = FakeHttp(responder=resp)
    paths = tmp_paths(tmp_path)
    logs = []
    rc = cw.main(["prefill-test", "--approved", "--limit-usd", "0.5"], http=http, paths=paths,
                 sleep=lambda s: None, log=logs.append)
    assert rc == 0, logs
    assert len(http.posts) == 80 and all(b["model"] == "qwen/qwen4-max" for b in http.bodies())
    assert sorted({b["max_tokens"] for b in http.bodies()}) == [16, 64]
    assert sum(b["max_tokens"] == 64 for b in http.bodies()) == 20
    res = json.loads((paths.results / "qwen_prefill_test.json").read_text(encoding="utf-8"))
    s = res["summary"]
    assert s["passed"] and s["passing_variants"] == ["assistant_prefix"]
    assert not s["variants"]["partial_true"]["pass"]
    md = (paths.results / "qwen_prefill_test.md").read_text(encoding="utf-8")
    assert "PASS" in md and "partial_true" in md and (paths.results / "qwen_prefill_test_clean.md").exists()
    bud = yaml.safe_load(paths.budget.read_text(encoding="utf-8"))
    assert "prefill" in bud["log"][-1]["run"]
    http = FakeHttp(responder=lambda b, n: reply(b, text="\\boxed{18}", ct=8))
    (tmp_path / "b").mkdir()
    paths2 = tmp_paths(tmp_path / "b")
    rc = cw.main(["prefill-test", "--approved", "--limit-usd", "0.5"], http=http, paths=paths2,
                 sleep=lambda s: None, log=lambda *a: None)
    assert rc == 3


# ------------------------------------------------------------------------------------------------ scoring, report

@needs_data
def test_scoring_hooks_on_synthetic_replies():
    from src import bbh_d2, gsm8k, hotpot_d2, math_d2
    g = gsm8k.load_gsm8k("test")
    g0 = str(g.loc[g["item_id"] == 0, "gold_answer"].iloc[0])
    mi, _ = math_d2.load_items("target")
    bi, _ = bbh_d2.load_items("target")
    hi, _ = hotpot_d2.load_items("target")
    m0, b0, h0 = mi.iloc[0], bi.iloc[0], hi.iloc[0]
    rows = [
        ("gpt-6-luna", "gsm8k", "0", f"\\boxed{{{g0}}}", 1),
        ("gpt-6-luna", "gsm8k", "0", "The answer is \\boxed{99999}", 0),
        ("qwen4-max", "gsm8k", "0", f"\\boxed{{{g0}}}", 1),
        ("qwen4-max", "math", str(m0["item_id"]), f"\\boxed{{{m0['gold_answer']}}}", 1),
        ("qwen4-max", "bbh", str(b0["item_id"]), f"\\boxed{{{b0['gold_answer']}}}", 1),
        ("qwen4-max", "hotpot", str(h0["item_id"]), f"\\boxed{{{h0['gold_answer']}}}", 1),
        ("qwen4-max", "hotpot", str(h0["item_id"]), None, 0),          # a refusal with no text
    ]
    df = pd.DataFrame([{"key": f"k{i}", "model_key": mk, "task": t, "request_id": "same-id", "item_id": it,
                        "text": tx, "finish_reason": "stop"} for i, (mk, t, it, tx, _) in enumerate(rows)])
    out, ver = cw.score_rows(df)
    assert list(out["strict_correct"].astype(int)) == [r[4] for r in rows]
    assert list(out["content_correct"].astype(int)) == [r[4] for r in rows]
    assert out.loc[0, "compliant"] == 1 and set(ver) == {"gsm8k", "math", "bbh", "hotpot"}
    assert ver["hotpot"]["content"] and ver["gsm8k"]["strict"]


def _small_manifest():
    tasks = {}
    for t in cw.TASKS:
        items = [f"t{k}/{i}" for k in range(2) for i in range(3)] if t == "bbh" else [str(i) for i in range(6)]
        tasks[t] = {"templates": ["a", "b", "c"], "items": items, "n_requests": 18}
    return {"tasks": tasks, "n_cells": 12, "n_requests_per_model": 72}


def test_report_on_synthetic_replies(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["bootstrap"]["draws"] = 300
    paths = tmp_paths(tmp_path)
    man = _small_manifest()
    raw = paths.raw("run")
    raw.parent.mkdir(parents=True)
    lines = []
    for m in cfg["models"]:
        for t in cw.TASKS:
            for c in man["tasks"][t]["templates"]:
                for j, i in enumerate(man["tasks"][t]["items"]):
                    if m["key"] == "qwen4-max" and t == "gsm8k" and c == "a" and j == 0:
                        continue                                   # one missing call
                    long_ = (j % 2 == 0)                           # half the items long: w16 = 0.5
                    out = "blocked" if (m["key"] == "gpt-6-luna" and t == "math" and c == "a" and j == 1) else "reply"
                    n = 40 if long_ else 4
                    if t == "hotpot" and long_ and j == 0:
                        n = 80                                     # w64 on HotpotQA: one item of six
                    lines.append({"event": "result", "key": f"{m['key']}|requested|{t}|{c}|{i}", "model_key": m["key"],
                                  "arm": "requested", "task": t, "request_id": f"{c}|direct_answer|{i}",
                                  "config_id": c, "item_id": i,
                                  "outcome": out, "status": 200 if out == "reply" else 403,
                                  "text": None if out == "blocked" else ("x" * (2 * n)) + "\\boxed{18}",
                                  "finish_reason": "stop" if out == "reply" else None,
                                  "completion_tokens": n if out == "reply" else None, "reasoning_tokens": 0,
                                  "prompt_tokens": 50, "cost_usd": 0.001, "config_sha": "s", "attempt_id": f"a{len(lines)}",
                                  "provider": m["expected_provider"], "served_model": m["id"]})
    raw.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    def panel_loader(task, items, templates, root):
        rows = [{"model_key": "open-a", "arm": "requested", "config_id": c, "item_id": i,
                 "n_output_tokens": 30 if k == 0 else 3, "finish_reason": "stop", "strict_correct": 1,
                 "lenient_correct": 1, "compliant": 1, "content_correct": 1}
                for c in templates for k, i in enumerate(items)]
        return pd.DataFrame(rows)
    rep = cw.build_report(cfg, man, paths, scorer=fake_scorer, tok=FakeTok(), panel_loader=panel_loader,
                          ladder_loader=no_ladder)
    pb = {(r["model"], r["task"]): r for r in rep["per_benchmark"] if r["arm"] == "requested"}
    assert pb[("gpt-6-luna", "gsm8k")]["w16_api"] == pytest.approx(0.5)
    lo, hi = pb[("gpt-6-luna", "gsm8k")]["w16_api_ci"]
    assert lo <= 0.5 <= hi
    assert pb[("gpt-6-luna", "hotpot")]["w64_api"] == pytest.approx(1 / 6)
    assert pb[("gpt-6-luna", "gsm8k")]["w16_api_spread"]["sd"] == pytest.approx(0.0)
    assert pb[("qwen4-max", "gsm8k")]["missing"] == 1 and pb[("gpt-6-luna", "math")]["blocked"] == 1
    assert pb[("gpt-6-luna", "math")]["w16_api_missing_cells"] == 1
    assert rep["models"]["qwen4-max"]["arms"]["requested"]["missing"] == 1
    assert rep["models"]["gpt-6-luna"]["arms"]["requested"]["blocked"] == 1
    assert pb[("gpt-6-luna", "gsm8k")]["strict_correct"] == pytest.approx(1.0)
    pm = rep["panel"]["gsm8k"]["models"]["open-a"]
    assert pm["w16_native"] == pytest.approx(1 / 6)
    assert rep["token_accounting"]["gpt-6-luna"]["gsm8k"]["n"] == 18
    assert rep["gains"] == [] and rep["placement"] == []                 # one arm: no gains
    md = cw.render_report(rep, cfg, man)
    assert "## w, requested arm" in md and "HotpotQA (64)" in md


def _three_arm_lines(man, cfg, model="qwen4-max"):
    """Synthetic replies of the three arms: requested right on items 0-2 of 6 (and long on items 0, 2, 4), forced
    right on item 0 only... per template a, b, c the same; CoT right everywhere."""
    lines = []
    m = cw.model_of(cfg, model)
    for t in cw.TASKS:
        for c in man["tasks"][t]["templates"]:
            for j, i in enumerate(man["tasks"][t]["items"]):
                for arm in cw.ARMS:
                    if arm == "requested":
                        ok, n = j < 3, (40 if j % 2 == 0 else 4)
                    elif arm == "forced":
                        ok, n = (j == 0 or (c == "a" and j == 1)), 3
                    else:
                        ok, n = True, 300
                    lines.append({"event": "result", "key": f"{model}|{arm}|{t}|{c}|{i}", "model_key": model,
                                  "arm": arm, "kind": arm, "task": t, "request_id": f"{c}|{arm}|{i}", "config_id": c,
                                  "item_id": i, "outcome": "reply", "status": 200,
                                  "text": ("RIGHT " if ok else "WRONG ") + "x" * (2 * n),
                                  "finish_reason": "stop", "completion_tokens": n, "reasoning_tokens": 0,
                                  "stop_sent": arm == "forced" and t in ("gsm8k", "bbh"),
                                  "prompt_tokens": 50, "cost_usd": 0.001, "config_sha": "s",
                                  "attempt_id": f"a{len(lines)}", "provider": m["expected_provider"],
                                  "served_model": m["id"]})
    return lines


def right_scorer(df):
    out = df.copy()
    ok = out["text"].map(lambda s: int(isinstance(s, str) and s.startswith("RIGHT")))
    for c in ("strict_correct", "content_correct", "lenient_correct", "compliant"):
        out[c] = ok
    out["forced_reading"] = np.where(out["arm"] == "forced", "as_returned", None)
    return out, {"right": "scorer"}


def test_report_three_arms_gains_and_placement(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["bootstrap"]["draws"] = 200
    paths = tmp_paths(tmp_path)
    man = _small_manifest()
    raw = paths.raw("run")
    raw.parent.mkdir(parents=True)
    raw.write_text("\n".join(json.dumps(x) for x in _three_arm_lines(man, cfg)) + "\n", encoding="utf-8")

    def others(task, items, templates, root, models=(("open-hi", 1.0, 0.0), ("open-lo", 0.5, 0.5))):
        rows = []
        for mk, cot, req in models:      # (model, CoT accuracy, requested accuracy); forced 0; w 0
            for c in templates:
                for k, i in enumerate(items):
                    for arm, acc in (("requested", req), ("forced", 0.0), ("cot", cot)):
                        rows.append({"model_key": mk, "arm": arm, "config_id": c, "item_id": i,
                                     "n_output_tokens": 3, "finish_reason": "stop",
                                     "strict_correct": int(k < acc * len(items)),
                                     "lenient_correct": 0, "compliant": 1,
                                     "content_correct": int(k < acc * len(items))})
        return pd.DataFrame(rows)

    def ladder(task, items, templates, root):
        return others(task, items, templates, root, models=(("big-model", 0.0, 0.0),))
    rep = cw.build_report(cfg, man, paths, scorer=right_scorer, tok=FakeTok(), panel_loader=others,
                          ladder_loader=ladder)
    g = {(x["task"], x["score"], x["basis"]): x for x in rep["gains"]}
    x = g[("gsm8k", "strict", "templates")]
    # requested 3/6; forced 1/6 on templates b and c, 2/6 on a; CoT 6/6; w (> 16 tokens) 3/6
    assert x["requested"] == pytest.approx(0.5) and x["cot"] == pytest.approx(1.0)
    assert x["forced"] == pytest.approx((2 / 6 + 1 / 6 + 1 / 6) / 3)
    assert x["cot_minus_requested"] == pytest.approx(0.5) and x["w"] == pytest.approx(0.5)
    assert x["corrected_gain"] == pytest.approx((1 - 0.5) * (1 - x["forced"]))
    assert x["residual"] == pytest.approx(0.5 - x["corrected_gain"])
    lo, hi = x["ci"]["cot_minus_forced"]
    assert lo <= x["cot_minus_forced"] <= hi
    assert x["ci"]["cot"] == [1.0, 1.0]
    assert x["w_recount"] is not None
    # placement among the panel (open-hi, open-lo) and the ladder (big-model)
    p = {(q["task"], q["score"], q["basis"], q["quantity"]): q for q in rep["placement"]}
    cot = p[("gsm8k", "strict", "templates", "cot")]
    assert cot["panel_and_ladder"]["M"] == 4 and cot["panel_and_ladder"]["k"] == 1
    assert cot["panel_and_ladder"]["ties"] == ["panel:open-hi"] and cot["panel"]["M"] == 3
    req = p[("gsm8k", "strict", "templates", "cot_minus_requested")]
    assert req["value"] == pytest.approx(0.5) and req["panel_and_ladder"]["k"] == 2        # open-hi has 1.0
    w = p[("gsm8k", "strict", "templates", "w")]
    assert w["panel_and_ladder"]["k"] == 1 and w["n_ladder"] == 1 and w["n_panel"] == 2
    # per template: corrected gain = (1 - w_c) x (CoT - forced)_c, with intervals
    pt = [r for r in rep["per_template"] if r["source"] == "closed" and r["task"] == "gsm8k" and r["score"] == "strict"]
    a = next(r for r in pt if r["config_id"] == "a")
    assert a["cot_minus_forced"] == pytest.approx(4 / 6) and a["corrected_gain"] == pytest.approx(0.5 * 4 / 6)
    assert a["corrected_gain_ci"][0] is not None
    # the discovery-draw basis exists when the sampler names the templates; here it does not, so it is absent
    assert all(q["basis"] == "templates" for q in rep["gains"]) or any(q["basis"] == "discovery" for q in rep["gains"])
    md = cw.render_report(rep, cfg, man)
    assert "Gains and the corrected gain" in md and "Where Qwen4-Max falls" in md and "1/4" in md


def test_placement_rule():
    assert cw.placement({"me": 0.5, "a": 0.7, "b": 0.5, "c": None, "d": 0.1}, "me") == \
        {"k": 2, "M": 4, "ties": ["b"]}
    assert cw.placement({"me": None, "a": 1.0}, "me") == {"k": None, "M": 1, "ties": []}


def test_report_writes_files(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["bootstrap"]["draws"] = 100
    paths = tmp_paths(tmp_path)
    man = _small_manifest()
    paths.manifest = tmp_path / "m.json"
    paths.manifest.write_text(json.dumps(man), encoding="utf-8")
    raw = paths.raw("run")
    raw.parent.mkdir(parents=True)
    raw.write_text("\n".join(json.dumps(x) for x in _three_arm_lines(man, cfg)) + "\n", encoding="utf-8")
    a = type("A", (), {"open_texts": True})()

    def open_loader(d, string_item_ids=True):
        raise FileNotFoundError("no folder")

    def panel_loader(task, items, templates, root):
        return pd.DataFrame([{"model_key": "open-a", "arm": arm, "config_id": c, "item_id": i, "n_output_tokens": 3,
                              "finish_reason": "stop", "strict_correct": 1, "lenient_correct": 1, "compliant": 1,
                              "content_correct": 1} for arm in cw.ARMS for c in templates for i in items])
    rc = cw.cmd_report(a, cfg, paths, scorer=right_scorer, tok=FakeTok(), panel_loader=panel_loader,
                       open_loader=open_loader, log=lambda *x: None, ladder_loader=no_ladder)
    assert rc == 0
    for f in ("closed_w.md", "closed_w_clean.md", "closed_w.json", "closed_w_templates.csv"):
        assert (paths.results / f).exists()
    csv = pd.read_csv(paths.results / "closed_w_templates.csv")
    assert set(csv["source"]) == {"closed", "panel"} and len(csv) == 2 * 2 * 4 * 3     # 2 models x 2 readings
    rep = json.loads((paths.results / "closed_w.json").read_text(encoding="utf-8"))
    assert rep["open_texts"]["gsm8k"]["rows_recounted"] == 0 and rep["open_texts"]["gsm8k"]["problems"]
    assert rep["forced_reading"]["qwen4-max"]["gsm8k"]["n"] == 18


def test_tokenizer_is_pinned():
    tok = cw.Tok(CFG)
    if not tok.path.exists():
        pytest.skip("tokenizer absent")
    assert tok.count("The answer is \\boxed{42}.") > 3
    assert tok.counts(["a b", None])[1] == 0
    bad = json.loads(json.dumps(CFG))
    bad["tokenizer"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        cw.Tok(bad).count("x")


# ------------------------------------------------------------------------------------------------ quote

def _quote_inputs():
    man = _small_manifest()
    sel = {}
    for arm in cw.ARMS:
        sel[arm] = {}
        for t in cw.TASKS:
            m = man["tasks"][t]
            sel[arm][t] = pd.DataFrame([{"request_id": f"{c}|{arm}|{i}", "config_id": c, "item_id": i,
                                         "text": "q" * (40 if arm != "cot" else 36)}
                                        for c in m["templates"] for i in m["items"]])

    def panel_loader(task, items, templates, root):
        rows = []
        for mk, scale in (("qwen3.6-27b", 1), ("qwen3.6-35b-a3b", 3), ("gemma-2b-it", 100)):
            for arm, n in (("requested", 10), ("forced", 4), ("cot", 200)):
                for c in templates:
                    for i in items:
                        rows.append({"model_key": mk, "arm": arm, "config_id": c, "item_id": i,
                                     "n_output_tokens": n * scale if mk != "gemma-2b-it" else 1,
                                     "finish_reason": "stop", "strict_correct": 1, "lenient_correct": 1,
                                     "compliant": 1, "content_correct": 1})
        return pd.DataFrame(rows)
    return man, sel, panel_loader


def test_quote_arithmetic(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["quote"]["strongest_models"] = 2
    paths = tmp_paths(tmp_path)
    man, sel, panel_loader = _quote_inputs()
    o200k = lambda texts: [len(t) // 4 for t in texts]           # noqa: E731
    q = cw.build_quote(cfg, man, paths, QWEN, sel, o200k, FakeTok(), None, panel_loader, ROOT)
    pin, pout = QWEN["price"]["prompt"], QWEN["price"]["completion"]
    tt = cfg["quote"]["template_tokens"]                       # no prefill raw file here: the config's value
    assert q["template_tokens"]["used"] == tt
    assert q["strongest_models"]["gsm8k"] == ["qwen3.6-27b", "qwen3.6-35b-a3b"]     # by ECI; gemma-2b-it is weaker
    r = {(x["arm"], x["task"]): x for x in q["rows"]}
    f = r[("forced", "gsm8k")]
    assert f["calls"] == 18 and f["in_o200k"] == 18 * (10 + len(PREFIX) // 4)        # the prefill counted once
    assert f["out_observed"] == 18 * (4 * 1 + 4 * 3) / 2 and f["out_cap"] == 18 * 16
    assert f["usd_observed_o200k"] == pytest.approx((f["in_o200k"] + 18 * tt) * pin + f["out_observed"] * pout)
    assert f["usd_worst_qwen"] == pytest.approx((f["in_qwen"] + 18 * tt) * pin + 18 * 16 * pout)
    c = r[("cot", "hotpot")]
    assert c["cap"] == 1900 and c["out_cap"] == 18 * 1900 and c["out_observed"] == 18 * (200 + 600) / 2
    # the chunk-guard bound is the sum of the run's reserves
    reqs = cw.run_requests(cfg, man, sel, [QWEN], list(cw.ARMS))
    assert q["total"]["guard_usd"] == pytest.approx(sum(x["reserve_usd"] for x in reqs))
    assert q["total"]["calls"] == 3 * 4 * 18
    assert q["total"]["usd_observed_qwen"] == pytest.approx(sum(x["usd_observed_qwen"] for x in q["rows"]))
    assert q["cot_share_of_total_observed"] == pytest.approx(q["per_arm"]["cot"]["usd_observed_qwen"]
                                                             / q["total"]["usd_observed_qwen"])
    md = cw.render_quote(q)
    assert "chain-of-thought arm dominates" in md and "Chunk-guard bound" in md


@needs_data
def test_quote_end_to_end_reads_only_the_public_listing(tmp_path, cached_requests):
    paths = tmp_paths(tmp_path, env=False)
    http = FakeHttp()
    logs = []
    o200k = lambda texts: [max(1, len(t or "") // 4) for t in texts]      # noqa: E731
    rc = cw.main(["quote"], http=http, paths=paths, tok=FakeTok(), o200k=o200k, log=logs.append)
    assert rc == 0, logs
    assert http.posts == [] and len(http.gets) == 1 and "qwen4-max" in http.gets[0]
    q = json.loads((paths.results / "qwen3arm_quote.json").read_text(encoding="utf-8"))
    assert q["total"]["calls"] == 39300 and {r["calls"] for r in q["rows"]} == {2900, 3100, 3500, 3600}
    assert q["listing"]["found"] and q["listing"]["discount"] == 0.05
    assert q["pilot"]["calls"] == 132 and q["pilot"]["worst_usd"] <= CFG["limits"]["pilot"]
    assert q["per_arm"]["cot"]["usd_observed_qwen"] > q["per_arm"]["requested"]["usd_observed_qwen"]
    assert (paths.results / "qwen3arm_quote.md").exists()
