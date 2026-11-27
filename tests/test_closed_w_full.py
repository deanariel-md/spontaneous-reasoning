"""Tests for scripts/closed_w_full.py (Qwen4-Max on all numeric MATH-500 items under MATH-500's first 8 discovery
draws). No network: urllib.request.urlopen raises in every test (the autouse fixture
below) and every reply comes from the fake HTTP object of tests/test_closed_w.py. The end-to-end tests run a synthetic
design (6 items x 2 templates) written to tmp_path; the 100-item run's files are never written."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import closed_w as cw  # noqa: E402
import closed_w_full as cwf  # noqa: E402
import test_closed_w as T  # noqa: E402  (fakes: FakeHttp, listing, reply, FakeTok, SECRET)

CFG = cw.load_config()
FCFG = cwf.load_full()
MODEL = cw.model_of(CFG, FCFG["model"])
ITEMS = [f"test/algebra/{i}.json" for i in range(6)]
SAMPLER_DRAWS = ["ta", "tb", "ta", "tc", "td"]
TEMPLATES = ["ta", "tb"]
PAD = "Find the value and show nothing else. " * 2
NOOP = lambda s: None  # noqa: E731

REAL_TABLE = ROOT / FCFG["items"]["table"]
REAL_MANIFEST = ROOT / FCFG["paths"]["manifest"]
needs_real = pytest.mark.skipif(not (REAL_TABLE.exists() and REAL_MANIFEST.exists()),
                                reason="MATH-500 item table or this design's manifest absent")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


# ------------------------------------------------------------------------------------------------ synthetic design

def item_index(body) -> int:
    m = re.search(r"Problem (\d+)", body["messages"][0]["content"])
    return int(m.group(1)) if m else 0


def graded(body, n):
    """Item i: requested right when i is even and long (40 tokens) when i is odd; forced right when i % 3 == 0;
    chain of thought always right. The output token count is the fake tokenizer's (one per two characters), so the
    pilot's token accounting holds."""
    if T.thinking_on(body):
        return T.reply(body, rt=200)
    i = item_index(body)
    arm = T.body_arm(body)
    if arm == "forced":
        t = ("18" if i % 3 == 0 else "7") + "}"
    elif arm == "cot":
        t = "Work it out step by step, then: \\boxed{18}"
    else:
        ans = "18" if i % 2 == 0 else "7"
        t = f"\\boxed{{{ans}}}" if i % 2 == 0 else ("Short reasoning first, " * 3 + f"so \\boxed{{{ans}}} it is.")
    return T.reply(body, text=t, ct=max(1, len(t) // 2))


def scorer(df):
    """The fake scorer of tests/test_closed_w.py with the columns scripts/closed_w.score_rows adds (forced_text,
    forced_reading: the forced reply as the panel's runner stores it)."""
    out, ver = T.fake_scorer(df)
    fr = [cw.forced_reply(t, k, f, CFG, bool(s)) if a == "forced" else (None, None) for t, k, f, s, a in
          zip(out["text"], out["task"], out["finish_reason"], out.get("stop_sent", [False] * len(out)), out["arm"])]
    out["forced_text"] = [x[0] for x in fr]
    out["forced_reading"] = [x[1] for x in fr]
    return out, ver


def blocked_403(body):
    return {"status": 403, "json": {"error": {
        "message": "Upstream error from Alibaba: Output data may contain inappropriate content.", "code": 403,
        "metadata": {"provider_name": "Alibaba", "provider_error_code": "data_inspection_failed"}}}}


def rows_by_arm() -> dict:
    out = {a: [] for a in cw.ARMS}
    for c in ["ta", "tb", "tc"]:
        for i, it in enumerate(ITEMS):
            q = f"Problem {i} under {c}. {PAD}End with \\boxed{{<number>}}."
            out["requested"].append({"request_id": f"{c}|direct_answer|{it}", "config_id": c, "item_id": it,
                                     "text": q})
            out["forced"].append({"request_id": f"{c}|direct_answer|{it}|fd", "config_id": c, "item_id": it,
                                  "text": q})
            out["cot"].append({"request_id": f"{c}|zero_shot_cot|{it}", "config_id": c, "item_id": it,
                               "text": f"Problem {i} under {c}. {PAD}Think step by step, then box the answer."})
    return {a: pd.DataFrame(v) for a, v in out.items()}


CTL = {"request_id": "ta|direct_answer|train/algebra/5", "config_id": "ta", "item_id": "train/algebra/5",
       "text": f"Problem 99 under ta. {PAD}End with \\boxed{{<number>}}."}
CTL_REC = {"task": "math", "request_id": CTL["request_id"], "config_id": "ta", "item_id": CTL["item_id"],
           "level": "Level 5", "text_sha256": cw.sha256_text(CTL["text"]), "n_bytes": len(CTL["text"].encode())}


def earlier_lines(csha="old-config", cost=None) -> list[dict]:
    """A synthetic 100-item raw file: every MATH cell of three templates and every item, billed per arm."""
    cost = cost or {"requested": 0.001, "forced": 0.0002, "cot": 0.002}
    out = []
    for a in cw.ARMS:
        for c in ["ta", "tb", "tc"]:
            for it in ITEMS:
                out.append({"event": "result", "config_sha": csha, "key": f"qwen4-max|{a}|math|{c}|{it}",
                            "outcome": "reply", "task": "math", "kind": a, "arm": a, "config_id": c, "item_id": it,
                            "cost_usd": cost[a] * (1.5 if c == "tc" else 1.0), "prompt_tokens": 120,
                            "completion_tokens": 10, "cached_tokens": 0})
    return out


def write_jsonl(p: Path, lines) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def setup(tmp_path, *, stop_usd=18.0, earlier=None, k=2):
    """A synthetic design in tmp_path: item table and its record, sampler, earlier raw file, .env with a fake key, the
    passing prefill result, budget copy. Returns (fcfg, paths)."""
    tbl = tmp_path / "items_target.parquet"
    pd.DataFrame({"item_id": ITEMS}).to_parquet(tbl, index=False)
    rec = tmp_path / "math_items.json"
    rec.write_text(json.dumps({"tables": {"target": {"sha256": cw.sha256_file(tbl)}},
                               "counts": {"target_numeric": len(ITEMS), "target_all": 8},
                               "numeric_rule": "test rule"}), encoding="utf-8")
    smp = tmp_path / "manifests" / "sampler_math.json"
    smp.parent.mkdir(parents=True, exist_ok=True)
    smp.write_text(json.dumps({"discovery_config_ids": SAMPLER_DRAWS, "k_discovery": len(SAMPLER_DRAWS),
                               "nested_order": "budget K uses the first K discovery units in the order listed",
                               "sha256": "x"}), encoding="utf-8")
    write_jsonl(tmp_path / "earlier_raw.jsonl", earlier if earlier is not None else earlier_lines())
    os.environ["OPENROUTER_API_KEY"] = T.SECRET
    pr = tmp_path / "prefill_result.json"
    pr.write_text(json.dumps({"summary": {"passed": True, "passing_variants": ["partial_true"], "variants": {
        "partial_true": {"message_extra": {"partial": True}, "pass": True}}}}), encoding="utf-8")
    (tmp_path / "budget.yaml").write_text(        # config/budget.yaml's layout, without this task's own entry
        "cap_usd: 0\nlog:\n  - date: '2026-11-24'\n    run: an earlier run\n    cost_usd: 0.1\n"
        "# Runs prepared but not yet submitted.\nplanned:\n  - date: 2026-11-21\n    run: a planned run\n",
        encoding="utf-8")
    fcfg = copy.deepcopy(FCFG)
    fcfg["templates"]["k"] = k
    fcfg["pilot"]["cells_per_template"] = 2
    fcfg["limits"]["projection_stop_usd"] = stop_usd
    fcfg["bootstrap"]["draws"] = 300
    paths = cwf.FPaths(fcfg, ROOT, manifest=tmp_path / "manifest.json", raw_dir=tmp_path / "raw",
                       results=tmp_path / "res", budget=tmp_path / "budget.yaml", env=tmp_path / ".env",
                       earlier_raw=tmp_path / "earlier_raw.jsonl", earlier_report=tmp_path / "earlier.json",
                       earlier_manifest=tmp_path / "earlier_manifest.json", items_table=tbl, items_record=rec,
                       sampler=smp, prefill_result=pr)
    return fcfg, paths


def approved(limit=20.0):
    return argparse.Namespace(approved=True, limit_usd=limit, threads=None, chunk=None)


def plan(tmp_path, **kw):
    fcfg, paths = setup(tmp_path, **kw)
    logs = []
    rc = cwf.cmd_plan(argparse.Namespace(), CFG, fcfg, paths, ROOT, logs.append, rows=rows_by_arm(), control=CTL_REC,
                      ctl=CTL)
    return rc, fcfg, paths, logs


def pilot_rows_of(paths) -> dict:
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    sel = cwf.selected(man, rows_by_arm())
    return {a: cw.rows_by_id(sel[a], man["pilot"]["rows"][a]) for a in sel}


def run_pilot(tmp_path, responder=graded, **kw):
    rc, fcfg, paths, _ = plan(tmp_path, **kw)
    assert rc == 0
    http = T.FakeHttp(responder)
    logs = []
    rc = cwf.cmd_pilot(approved(), CFG, fcfg, paths, http=http, scorer=scorer, tok=T.FakeTok(), sleep=NOOP,
                       root=ROOT, log=logs.append, prows=pilot_rows_of(paths), ctl=CTL)
    return rc, fcfg, paths, http, logs


def files_text(*dirs) -> str:
    out = []
    for d in dirs:
        d = Path(d)
        for p in ([d] if d.is_file() else d.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".yaml", ".csv", ".env") and p.name != ".env":
                out.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ design

def test_default_paths_are_new_files():
    paths = cwf.FPaths(FCFG)
    cwf.check_separate(paths, CFG)                      # passes
    old = cw.Paths(CFG)
    assert paths.raw_dir.resolve() != old.raw_dir.resolve()
    assert paths.manifest.resolve() != old.manifest.resolve()
    for suffix in (".json", ".md", "_clean.md", "_pilot.json", "_pilot.md", "_plan.json", "_templates.csv"):
        assert paths.out(suffix).name.startswith("closed_w_full_math")
    assert paths.earlier_raw.resolve() == (ROOT / "data/frontier_panel/closed_w/run_raw.jsonl").resolve()


@pytest.mark.parametrize("over,msg", [({"prefix": "closed_w"}, "collide"), ({"prefix": "closed_w_pilot"}, "collide"),
                                      ({"raw_dir": ROOT / "data/frontier_panel/closed_w"}, "raw directory"),
                                      ({"manifest": ROOT / "manifests/closed_w_items.json"}, "manifest")])
def test_check_separate_refuses_the_100_item_files(over, msg):
    with pytest.raises(SystemExit, match=msg):
        cwf.check_separate(cwf.FPaths(FCFG, **over), CFG)


def test_config_is_the_100_item_runs():
    assert FCFG["base_config"] == "config/closed_w.yaml"
    assert cwf.base_config(FCFG) == CFG
    assert FCFG["limits"]["total_usd"] == 20.0 and FCFG["limits"]["projection_stop_usd"] == 18.0
    assert FCFG["templates"]["k"] == 8 and FCFG["bootstrap"]["draws"] == 2000


def test_discovery_templates_take_the_first_k_and_refuse_a_repeat(tmp_path):
    _, paths = setup(tmp_path)
    t = cwf.discovery_templates(paths.sampler, 2)
    assert t["templates"] == ["ta", "tb"] and t["k_discovery"] == 5 and t["distinct_discovery_templates"] == 4
    with pytest.raises(SystemExit, match="repeat"):
        cwf.discovery_templates(paths.sampler, 3)       # ta, tb, ta
    with pytest.raises(SystemExit, match="fewer"):
        cwf.discovery_templates(paths.sampler, 6)


def test_target_items_check_the_table_against_its_record(tmp_path):
    _, paths = setup(tmp_path)
    got = cwf.target_items(paths.items_table, paths.items_record)
    assert got["items"] == ITEMS and got["n"] == 6
    pd.DataFrame({"item_id": ITEMS[:5]}).to_parquet(paths.items_table, index=False)
    with pytest.raises(SystemExit, match="sha256"):
        cwf.target_items(paths.items_table, paths.items_record)


@needs_real
def test_real_items_and_templates():
    items = cwf.target_items(REAL_TABLE, ROOT / FCFG["items"]["record"])
    assert items["n"] == 398 and len(items["items"]) == 398 and items["target_all"] == 500
    t = cwf.discovery_templates(ROOT / FCFG["templates"]["sampler"], 8)
    assert t["templates"] == ["ps-i00-f05", "ps-i02-f03", "ps-i03-f07", "ps-i06-f01", "ps-i01-f00", "ps-i04-f00",
                              "ps-i06-f05", "ps-i05-f02"]
    assert t["in_budget_grid"]
    man = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    assert man["items"] == items["items"] and man["templates"] == t["templates"]
    assert man["n_requests_per_arm"] == 3184 and man["n_requests"] == 9552
    assert man["config_sha"] == cw.config_sha(CFG)
    assert len(man["pilot"]["cells"]) == 32 and len({c["item_id"] for c in man["pilot"]["cells"]}) == 32
    # every template of the run was sent in the 100-item run (the projection's basis)
    old = json.loads((ROOT / "manifests" / "closed_w_items.json").read_text(encoding="utf-8"))
    assert set(man["templates"]) <= set(old["tasks"]["math"]["templates"])


# ------------------------------------------------------------------------------------------------ plan

def test_plan_writes_manifest_and_projection(tmp_path):
    rc, fcfg, paths, logs = plan(tmp_path)
    assert rc == 0
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert man["templates"] == TEMPLATES and man["items"] == ITEMS
    assert man["n_requests_per_arm"] == 12 and man["n_requests"] == 36
    assert man["arms"]["forced"]["user_message_equals_requested"] is True
    assert len(man["pilot"]["cells"]) == 4 and len({c["item_id"] for c in man["pilot"]["cells"]}) == 4
    res = json.loads(paths.out("_plan.json").read_text(encoding="utf-8"))
    pj = res["projection"]
    # earlier billed cost per call on the run's templates (ta, tb) times 12 calls per arm
    assert pj["per_arm"]["requested"]["projected_usd"] == pytest.approx(0.001 * 12)
    assert pj["per_arm"]["cot"]["projected_usd"] == pytest.approx(0.002 * 12)
    assert pj["projected_usd"] == pytest.approx(0.0032 * 12)
    assert pj["projected_all_templates_usd"] > pj["projected_usd"]          # template tc billed 1.5x
    assert pj["go"] is True and res["pilot_calls"] == 4 * 3 * 2 + 2
    assert paths.out("_plan_clean.md").exists()
    # a second plan leaves the manifest as it is
    rc = cwf.cmd_plan(argparse.Namespace(), CFG, fcfg, paths, ROOT, logs.append, rows=rows_by_arm(), control=CTL_REC,
                      ctl=CTL)
    assert rc == 0 and any("unchanged" in x for x in logs)


def test_plan_refuses_a_changed_manifest(tmp_path):
    rc, fcfg, paths, _ = plan(tmp_path)
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    man["templates"] = ["tb", "ta"]
    paths.manifest.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit, match="differs"):
        cwf.cmd_plan(argparse.Namespace(), CFG, fcfg, paths, ROOT, print, rows=rows_by_arm(), control=CTL_REC, ctl=CTL)


def test_run_list_order_and_bodies(tmp_path):
    rc, fcfg, paths, _ = plan(tmp_path)
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    sel = cwf.selected(man, rows_by_arm())
    rl = cwf.run_requests(CFG, sel, [MODEL])
    assert len(rl) == 36 and len({r["key"] for r in rl}) == 36
    # the arms of a unit back to back, units in sha256 order
    assert [r["kind"] for r in rl[:3]] == list(cw.run_arms(CFG))
    units = [(r["config_id"], r["item_id"]) for r in rl[::3]]
    assert units == sorted(units, key=lambda u: cw.hkey("closed_w_full/run", "math", *u))
    for r in rl:
        b = r["body"]
        assert b["provider"] == MODEL["provider"] and b["reasoning"] == MODEL["reasoning_off"]
        assert b["temperature"] == CFG["decoding"]["temperature"] == 0
        assert b["max_tokens"] == cw.arm_cap(CFG, r["kind"], "math")
        assert "stop" not in b                          # MATH-500 sends no stop on any arm
        if r["kind"] == "forced":
            assert b["messages"][-1] == {"role": "assistant", "content": CFG["prefill"], "partial": True}
        # the same request as scripts/closed_w.py would send for the same row
        row = {"request_id": r["request_id"], "config_id": r["config_id"], "item_id": r["item_id"],
               "text": b["messages"][0]["content"]}
        assert cw.make_request(CFG, MODEL, "math", row, r["kind"])["body"] == b


def test_projection_above_the_stop_blocks_every_paid_phase(tmp_path):
    rc, fcfg, paths, logs = plan(tmp_path, stop_usd=0.01)
    assert rc == 4
    assert json.loads(paths.out("_plan.json").read_text(encoding="utf-8"))["projection"]["go"] is False
    http = T.FakeHttp(graded)
    out = []
    assert cwf.cmd_pilot(approved(), CFG, fcfg, paths, http=http, log=out.append, prows={}, ctl=CTL) == 2
    assert cwf.cmd_run(approved(), CFG, fcfg, paths, http=http, log=out.append, rows=rows_by_arm()) == 2
    assert http.posts == [] and http.gets == []
    assert all("above the stop" in x for x in out)


@pytest.mark.parametrize("ns,msg", [
    (argparse.Namespace(approved=False, limit_usd=20.0), "approved"),
    (argparse.Namespace(approved=True, limit_usd=None), "limit-usd"),
    (argparse.Namespace(approved=True, limit_usd=20.01), "above the task's limit"),
    (argparse.Namespace(approved=True, limit_usd=0.0), "positive")])
def test_paid_phases_refuse_without_approval_or_within_the_limit(tmp_path, ns, msg):
    rc, fcfg, paths, _ = plan(tmp_path)
    http = T.FakeHttp(graded)
    out = []
    assert cwf.cmd_pilot(ns, CFG, fcfg, paths, http=http, log=out.append, prows={}, ctl=CTL) == 2
    assert msg in out[0] and http.posts == []


def test_paid_phases_refuse_without_a_dry_run(tmp_path):
    fcfg, paths = setup(tmp_path)
    out = []
    assert cwf.cmd_pilot(approved(), CFG, fcfg, paths, http=T.FakeHttp(), log=out.append, prows={}, ctl=CTL) == 2
    assert "no dry run" in out[0]


def test_run_refuses_without_a_passing_pilot(tmp_path):
    rc, fcfg, paths, _ = plan(tmp_path)
    http = T.FakeHttp(graded)
    out = []
    assert cwf.cmd_run(approved(), CFG, fcfg, paths, http=http, log=out.append, rows=rows_by_arm()) == 2
    assert "no pilot" in out[0] and http.posts == []


# ------------------------------------------------------------------------------------------------ pilot

def test_pilot_passes_and_projects(tmp_path):
    rc, fcfg, paths, http, logs = run_pilot(tmp_path)
    res = json.loads(paths.out("_pilot.json").read_text(encoding="utf-8"))
    bad = [c for c in res["checks"] if not c["pass"]]
    assert rc == 0 and res["passed"], bad
    names = [c["name"] for c in res["checks"]]
    for n in ("preflight", "answered", "reasoning_off", "reasoning_on_control", "provider", "finish_reasons",
              "token_accounting", "repeat_agreement", "cost_per_call", "forced_continuation", "projected_full_cost"):
        assert n in names
    assert "forced_stop_control" not in names           # MATH-500 sends no stop
    assert len(http.posts) == 4 * 3 * 2 + 2 == res["n_calls"]
    ctl = [b for b in http.bodies() if T.thinking_on(b)]
    assert len(ctl) == 2
    pj = res["projection"]
    assert pj["projected_usd"] == pytest.approx(sum(v["mean"] for v in pj["per_arm"].values()) * 12)
    assert res["spend"]["pilot"]["billed_usd"] > 0 and res["spend"]["run"]["billed_usd"] == 0
    assert T.SECRET not in files_text(tmp_path / "raw", tmp_path / "res") and not any(T.SECRET in x for x in logs)


def test_pilot_fails_when_a_forced_reply_restarts_the_box(tmp_path):
    def restart(body, n):
        if not T.thinking_on(body) and T.body_arm(body) == "forced":
            return T.reply(body, text="\\boxed{18}", ct=5)
        return graded(body, n)
    rc, fcfg, paths, http, logs = run_pilot(tmp_path, restart)
    res = json.loads(paths.out("_pilot.json").read_text(encoding="utf-8"))
    assert rc == 3 and not res["passed"]
    assert [c["name"] for c in res["checks"] if not c["pass"]] == ["forced_continuation"]
    out = []
    assert cwf.cmd_run(approved(), CFG, fcfg, paths, http=http, log=out.append, rows=rows_by_arm()) == 2
    assert "did not pass" in out[0]


def test_pilot_fails_when_the_route_reasons_with_thinking_off(tmp_path):
    def reasons(body, n):
        if not T.thinking_on(body) and T.body_arm(body) == "requested":
            return T.reply(body, rt=30)
        return graded(body, n)
    rc, _, paths, _, _ = run_pilot(tmp_path, reasons)
    res = json.loads(paths.out("_pilot.json").read_text(encoding="utf-8"))
    assert rc == 3 and "reasoning_off" in [c["name"] for c in res["checks"] if not c["pass"]]


def test_pilot_preflight_failure_sends_nothing(tmp_path):
    rc, fcfg, paths, _ = plan(tmp_path)
    http = T.FakeHttp(graded, list_fn=lambda url: T.listing(url, price_factor=2.0))
    assert cwf.cmd_pilot(approved(), CFG, fcfg, paths, http=http, scorer=scorer, tok=T.FakeTok(), sleep=NOOP,
                         log=print, prows=pilot_rows_of(paths), ctl=CTL) == 3
    assert http.posts == []


# ------------------------------------------------------------------------------------------------ run

def test_run_stops_at_the_limit_and_resumes_without_rebilling(tmp_path):
    rc, fcfg, paths, http, _ = run_pilot(tmp_path)
    assert rc == 0
    pilot_spend = cwf.spend(paths)["total_counted_usd"]
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    rl = cwf.run_requests(CFG, cwf.selected(man, rows_by_arm()), [MODEL])
    # room for the first unit's requested and forced calls, not for its chain of thought (the largest reserve)
    assert [r["kind"] for r in rl[:3]] == ["requested", "forced", "cot"]
    assert rl[2]["reserve_usd"] > rl[0]["reserve_usd"] + rl[1]["reserve_usd"]
    limit = pilot_spend + rl[0]["reserve_usd"] + rl[1]["reserve_usd"] + 1e-9
    http = T.FakeHttp(graded)
    out = []
    assert cwf.cmd_run(approved(limit), CFG, fcfg, paths, http=http, sleep=NOOP, log=out.append,
                       rows=rows_by_arm()) == 3
    first = [json.loads(d.decode())["messages"][0]["content"] for _, d in http.posts]
    assert len(first) == 2
    sp = cwf.spend(paths)
    assert sp["total_counted_usd"] <= limit + 1e-12      # the pilot counts against the same limit
    assert any("spend limit" in x for x in out)
    # resume under the task's limit: only the calls not answered are sent
    http2 = T.FakeHttp(graded)
    assert cwf.cmd_run(approved(20.0), CFG, fcfg, paths, http=http2, sleep=NOOP, log=out.append,
                       rows=rows_by_arm()) == 0
    assert len(http.posts) + len(http2.posts) == 36
    lines = cw.read_raw(paths.raw("run"))
    keys = [x["key"] for x in lines if x.get("event") == "result"]
    assert len(keys) == len(set(keys)) == 36
    assert cw.done_keys(lines, cw.config_sha(CFG)) == {r["key"] for r in rl}
    # a third session sends nothing
    http3 = T.FakeHttp(graded)
    assert cwf.cmd_run(approved(20.0), CFG, fcfg, paths, http=http3, sleep=NOOP, log=out.append,
                       rows=rows_by_arm()) == 0
    assert http3.posts == []
    assert T.SECRET not in files_text(tmp_path / "raw", tmp_path / "res") and not any(T.SECRET in x for x in out)


def test_run_retries_technical_failures(tmp_path):
    rc, fcfg, paths, _, _ = run_pilot(tmp_path)
    seen = set()

    def flaky(body, n):
        k = json.dumps(body["messages"], sort_keys=True) + str(body["max_tokens"])
        if k not in seen:
            seen.add(k)
            return {"status": 503, "json": None, "error": "busy"}
        return graded(body, n)
    http = T.FakeHttp(flaky)
    assert cwf.cmd_run(approved(), CFG, fcfg, paths, http=http, sleep=NOOP, log=print, rows=rows_by_arm()) == 0
    assert len(http.posts) == 72
    assert len(cw.done_keys(cw.read_raw(paths.raw("run")), cw.config_sha(CFG))) == 36


# ------------------------------------------------------------------------------------------------ report

def run_all(tmp_path, responder=graded, earlier=None):
    rc, fcfg, paths, _, _ = run_pilot(tmp_path, earlier=earlier)
    assert rc == 0
    http = T.FakeHttp(responder)
    assert cwf.cmd_run(approved(), CFG, fcfg, paths, http=http, sleep=NOOP, log=print, rows=rows_by_arm()) == 0
    return fcfg, paths, http


def g_of(rep, score="strict"):
    return next(x for x in rep["gains"] if x["score"] == score)


def test_report_gains_w_and_paired_difference(tmp_path):
    fcfg, paths, _ = run_all(tmp_path)
    rep = cwf.build_report(CFG, fcfg, paths, scorer=scorer, tok=T.FakeTok(), root=tmp_path)
    g = g_of(rep)
    # items 0..5: requested right on even items, forced on items 0 and 3, CoT on all; requested long on odd items
    assert g["requested"] == pytest.approx(0.5) and g["forced"] == pytest.approx(1 / 3) and g["cot"] == 1.0
    assert g["w"] == pytest.approx(0.5)
    assert g["cot_minus_requested"] == pytest.approx(0.5)
    assert g["cot_minus_forced"] == pytest.approx(2 / 3)
    assert g["corrected_gain"] == pytest.approx(0.5 * 2 / 3)
    assert g["gain_diff"] == pytest.approx(g["forced"] - g["requested"]) == pytest.approx(-1 / 6)
    for q in ("cot_minus_requested", "cot_minus_forced", "w", "corrected_gain", "gain_diff"):
        lo, hi = g["ci"][q]
        assert lo <= g[q] + 1e-12 and g[q] <= hi + 1e-12
    assert g["cells"] == {"requested": 12, "forced": 12, "cot": 12} and g["n_items"] == 6 and g["n_templates"] == 2
    c = rep["calls"]
    assert all(v["answered"] == 12 and v["blocked"] == 0 and v["missing"] == 0 for v in c.values())
    assert rep["calls_total"]["answered"] == 36
    assert rep["forced_reading"]["n"] == 12 and rep["forced_reading"]["starts_with_prefill"] == 0
    assert len([x for x in rep["per_template"] if x["score"] == "strict"]) == 2
    assert rep["comparison"] == []                      # no earlier manifest in this layout


def test_bootstrap_matches_closed_w_and_pairs_the_arms():
    rng = np.random.default_rng(1)
    mats = {k: (rng.random((3, 40)) < p).astype(float) for k, p in (("R", .5), ("F", .3), ("C", .8), ("W", .6))}
    mats["R"][0, 3] = np.nan
    wv = np.array([1.0, 2.0, 1.0])
    mine = cwf.boot(mats, ["all"] * 40, {"b": wv}, 500, 0.95, 7)["b"]
    ref = cw.boot_quantities(mats, ["all"] * 40, {"b": wv}, 500, 0.95, 7)["benchmark"]["b"]
    for k, v in ref.items():
        assert mine[k] == pytest.approx(v)
    # the paired difference's interval is the interval of (forced - requested) accuracy, item draws shared
    assert "gain_diff" in mine and mine["gain_diff"][0] < mine["gain_diff"][1]
    q = cwf.point(mats, wv)
    assert q["gain_diff"] == pytest.approx(q["forced"] - q["requested"])


def test_provider_block_is_counted_and_left_out(tmp_path):
    target = None

    def resp(body, n):
        nonlocal target
        if not T.thinking_on(body) and T.body_arm(body) == "requested" and item_index(body) == 1 \
                and "under tb" in body["messages"][0]["content"]:
            target = body
            return blocked_403(body)
        return graded(body, n)
    fcfg, paths, http = run_all(tmp_path, resp)
    assert target is not None
    lines = cw.read_raw(paths.raw("run"))
    blk = [x for x in lines if x.get("event") == "result" and x.get("outcome") == "blocked"]
    assert len(blk) == 1 and blk[0]["charged_usd"] == 0
    assert sum(1 for b in http.bodies() if b == target) == 1            # not retried
    rep = cwf.build_report(CFG, fcfg, paths, scorer=scorer, tok=T.FakeTok(), root=tmp_path)
    assert rep["calls"]["requested"]["blocked"] == 1 and rep["calls"]["requested"]["answered"] == 11
    assert rep["calls_total"]["blocked"] == 1
    g = g_of(rep)
    assert g["cells"]["requested"] == 11 and g["cells"]["forced"] == 12
    # template tb lost one long, wrong requested reply: per-template means, then the mean over templates
    assert g["w"] == pytest.approx((3 / 6 + 2 / 5) / 2)
    assert g["requested"] == pytest.approx((3 / 6 + 3 / 5) / 2)
    md = cwf.render_report(rep)
    assert "| requested | 12 | 11 | 0 | 1 |" in md


def test_report_sits_beside_the_100_item_run(tmp_path):
    # the earlier run: the same replies on items 0-3 under ta, tb and tc (3 templates)
    old_csha = "old-config"
    b100 = ITEMS[:4]
    old = []
    for a in cw.ARMS:
        for c in ["ta", "tb", "tc"]:
            for i, it in enumerate(b100):
                q = f"Problem {i} under {c}. {PAD}"
                body = {"model": MODEL["id"], "reasoning": MODEL["reasoning_off"], "max_tokens": cw.arm_cap(CFG, a, "math"),
                        "messages": [{"role": "user", "content": q}] +
                                    ([{"role": "assistant", "content": "\\boxed{"}] if a == "forced" else [])}
                p = cw.parse_reply(graded(body, 0))
                old.append({"event": "result", "config_sha": old_csha, "key": f"qwen4-max|{a}|math|{c}|{it}",
                            "model_key": "qwen4-max", "task": "math", "kind": a, "arm": a, "config_id": c,
                            "item_id": it, "stop_sent": False, **p, "cost_usd": 0.001})
    fcfg, paths, _ = run_all(tmp_path, earlier=old)
    paths.earlier_manifest.write_text(json.dumps({"tasks": {"math": {"items": b100, "templates": ["ta", "tb", "tc"]}}}),
                                      encoding="utf-8")
    pub = {"gains": [{"model": "qwen4-max", "task": "math", "score": s, "basis": "discovery", "requested": 0.25,
                      "forced": 0.1, "cot": 0.9, "cot_minus_requested": 0.65, "cot_minus_forced": 0.8, "w": 0.4,
                      "corrected_gain": 0.48, "residual": 0.17,
                      "ci": {k: [0.0, 1.0] for k in cw.QUANTS}} for s in ("strict", "content")]}
    paths.earlier_report.write_text(json.dumps(pub), encoding="utf-8")
    before = {p.name: cw.sha256_file(p) for p in (paths.earlier_raw, paths.earlier_report, paths.earlier_manifest)}
    rep = cwf.build_report(CFG, fcfg, paths, scorer=scorer, tok=T.FakeTok(), root=tmp_path)
    assert {p.name: cw.sha256_file(p) for p in (paths.earlier_raw, paths.earlier_report,
                                                paths.earlier_manifest)} == before
    rows = [x for x in rep["comparison"] if x["score"] == "strict"]
    assert [x["label"][:3] for x in rows] == ["(a)", "(b)", "(c)", "(d)"]
    a, b, c, d = rows
    assert a["n_items"] == 6 and b["n_items"] == 4 and c["n_items"] == 4 and d["n_items"] == 4
    assert b["cot_minus_forced"] == pytest.approx(c["cot_minus_forced"])     # same replies on the same cells
    # (d) shows the published values unchanged; its paired difference is computed from them
    assert d["cot_minus_requested"] == 0.65 and d["w"] == 0.4 and d["recomputed"] is False
    assert d["gain_diff"] == pytest.approx(0.65 - 0.8)
    assert rep["published_check"]["compared"] > 0
    ov = rep["overlap"]
    assert ov["requested"]["cells"] == 8 and ov["requested"]["same_text"] == 1.0
    assert ov["forced"]["strict_new_minus_old_pp"] == 0
    md = cwf.render_report(rep)
    assert "Beside the 100-item run" in md and "(d) the 100-item run as published" in md


def test_report_command_writes_new_files_only(tmp_path):
    fcfg, paths, _ = run_all(tmp_path)
    assert cwf.cmd_report(argparse.Namespace(), CFG, fcfg, paths, scorer=scorer, tok=T.FakeTok(),
                          root=tmp_path, log=print) == 0
    names = sorted(p.name for p in paths.results.iterdir())
    assert names == sorted(["closed_w_full_math.json", "closed_w_full_math.md", "closed_w_full_math_clean.md",
                            "closed_w_full_math_templates.csv", "closed_w_full_math_pilot.json",
                            "closed_w_full_math_pilot.md", "closed_w_full_math_pilot_clean.md",
                            "closed_w_full_math_plan.json", "closed_w_full_math_plan.md",
                            "closed_w_full_math_plan_clean.md"])
    assert T.SECRET not in files_text(tmp_path / "raw", tmp_path / "res")


def test_budget_entry_once(tmp_path):
    fcfg, paths, _ = run_all(tmp_path)
    before = yaml.safe_load(paths.budget.read_text(encoding="utf-8"))["log"]
    out = []
    assert cwf.cmd_budget(argparse.Namespace(), CFG, fcfg, paths, log=out.append) == 0
    after = yaml.safe_load(paths.budget.read_text(encoding="utf-8"))["log"]
    assert after[:-1] == before and len(after) == len(before) + 1
    e = after[-1]
    for k in ("date", "run", "model", "unique_generations", "cached_calls", "tokens", "cost_usd", "cost_detail",
              "runtime", "notes"):
        assert k in e
    sp = cwf.spend(paths)
    assert e["cost_usd"] == pytest.approx(round(sp["total_billed_usd"], 4))
    assert "alibaba" in e["model"] and "62 answered calls of 62 attempts" in e["unique_generations"]
    assert cwf.cmd_budget(argparse.Namespace(), CFG, fcfg, paths, log=out.append) == 2      # not twice
    assert T.SECRET not in paths.budget.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ macros

def test_macros_module(tmp_path):
    sys.path.insert(0, str(ROOT / "paper"))
    import macros_cw_full as M
    fcfg, paths, _ = run_all(tmp_path)
    rep = cwf.build_report(CFG, fcfg, paths, scorer=scorer, tok=T.FakeTok(), root=tmp_path)
    assert M.cwfull_results(str(tmp_path)) is None
    d = tmp_path / "results" / "frontier_panel"
    d.mkdir(parents=True)
    (d / "closed_w_full_math.json").write_text(cw.dumps(rep), encoding="utf-8")
    shutil.copy(paths.out("_pilot.json"), d / "closed_w_full_math_pilot.json")
    m = M.cwfull_macros(M.cwfull_results(str(tmp_path)))
    assert all(re.fullmatch(r"cwFull[A-Za-z]+", k) for k in m)
    assert m["cwFullReqGainMath"] == "50.0" and m["cwFullFdGainMath"] == "66.7" and m["cwFullCorrMath"] == "33.3"
    assert m["cwFullGainDiffMath"] == "$-$16.7" and m["cwFullWMathPct"] == "50"
    assert m["cwFullReqAccMath"] == "50.0" and m["cwFullFdAccMath"] == "33.3" and m["cwFullCotAccMath"] == "100.0"
    for base in ("cwFullReqGainMath", "cwFullFdGainMath", "cwFullCorrMath", "cwFullGainDiffMath", "cwFullWMathPct"):
        for s in ("", "Content"):
            assert {base + s, base + s + "Lo", base + s + "Hi"} <= set(m)
    assert m["cwFullCalls"] == "36" and m["cwFullAnswered"] == "36" and m["cwFullBlocked"] == "0"
    assert m["cwFullItems"] == "6" and m["cwFullTemplates"] == "2" and m["cwFullPilotCalls"] == "26"
    assert m["cwFullCostUsd"] == f"{rep['spend']['total_billed_usd']:.2f}"
    assert not any(k.startswith("cwFullPrev") for k in m)          # no 100-item comparison in this layout
    # no name shared with the 100-item run's cw macros (paper/macros_closed_ladder.py)
    import macros_closed_ladder as W
    old = W.closed_ladder_macros(W.closed_ladder_results(str(ROOT)))
    assert old and not set(m) & set(old)
    # an interval with no finite draw is reported, not printed as a number
    rep["gains"][0]["ci"]["gain_diff"] = [None, None]
    (d / "closed_w_full_math.json").write_text(cw.dumps(rep), encoding="utf-8")
    with pytest.raises(ValueError, match="no finite value"):
        M.cwfull_macros(M.cwfull_results(str(tmp_path)))
