"""Guided decoding of the constrained forced-direct arm (src/generation.py)."""
import json
import sys
import types

import pandas as pd
import pytest
import yaml

from src import generation as g
from tests.test_generation import Cfg, decoding_dict, fi, kernel_switches, make_requests  # noqa: F401

FORCED = decoding_dict(max_new_tokens={"forced_direct": 16}, prefill={"forced_direct": "\\boxed{"}, stop=["}"],
                       include_stop_str_in_output=True)
GUIDED = dict(FORCED, guided={"forced_direct": "request"})
CHOICE = {"choice": ["(A)}", "(B)}", "A}", "B}"]}


@pytest.fixture(autouse=True)
def fake_gpu(monkeypatch):
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 1, "devices": [{"name": "Fake A100", "memory_mib": 81920}], "driver_version": "580.1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})


def guided_requests(path, src, constraint=CHOICE):
    df = pd.read_parquet(src)
    df["guided_json"] = json.dumps(constraint)
    df.to_parquet(path)
    return path


def shards(d):
    return pd.concat(pd.read_parquet(f) for f in d.glob("shard-*.parquet")).sort_values("request_id").reset_index(drop=True)


def test_plain_decodings_have_no_guided_marker():
    for name in ("decoding.yaml", "decoding_forced.yaml"):
        d = g.load_decoding(g.ROOT / "config" / name)
        assert d.guided == {} and all("guided_source" not in d.params_for_arm(a) for a in d.arms)
        assert "guided" not in d.to_log()          # earlier run logs and fingerprints stand


def test_constrained_decoding_file_is_the_forced_file_plus_guided():
    f = yaml.safe_load(open(g.ROOT / "config" / "decoding_forced.yaml", encoding="utf-8"))
    c = yaml.safe_load(open(g.ROOT / "config" / "decoding_forced_constrained.yaml", encoding="utf-8"))
    assert c.pop("guided") == {"forced_direct": "request"}
    assert c == f
    d = g.load_decoding(g.ROOT / "config" / "decoding_forced_constrained.yaml")
    p = d.params_for_arm("forced_direct")
    assert p["guided_source"] == "request" and p["prefill"] == "\\boxed{"


def test_guided_for_an_unknown_arm_or_value_is_refused(tmp_path):
    for bad, msg in (({"other": "request"}, "guided names arms"), ({"forced_direct": "yes"}, "guided must map")):
        p = tmp_path / "d.yaml"
        p.write_text(yaml.safe_dump(dict(FORCED, guided=bad)))
        with pytest.raises(g.RunRefused, match=msg):
            g.load_decoding(p)


def test_parse_guided():
    assert g.parse_guided(json.dumps(CHOICE)) == CHOICE
    assert g.parse_guided('{"regex": "-?[0-9]{1,7}\\\\}"}') == {"regex": "-?[0-9]{1,7}\\}"}
    for bad in ('{"choice": ["A}"]}', '{"choice": ["A}", "A}"]}', '{"choice": ["A}", ""]}', '{"regex": ""}',
                '{"json": {}}', '{"choice": ["A}", "B}"], "regex": "x"}', '["A}"]'):
        with pytest.raises(ValueError):
            g.parse_guided(bad)


def test_rendered_prompt_is_unchanged_by_the_constraint(tmp_path):
    """The constrained arm renders exactly the forced arm's prompts: same prompt hashes for every request."""
    req = make_requests(tmp_path / "r.parquet", arms=("forced_direct",), n_items=3, configs=("c0", "c1"), same_as={})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = Cfg(tmp_path / "a", decoding=FORCED), Cfg(tmp_path / "b", decoding=GUIDED)
    assert a.run(req, tmp_path / "oa")["status"] == "complete"
    greq = guided_requests(tmp_path / "rg.parquet", req)
    assert b.run(greq, tmp_path / "ob")["status"] == "complete"
    oa, ob = shards(tmp_path / "oa"), shards(tmp_path / "ob")
    assert len(oa) == len(ob) == 6
    assert (oa["prompt_sha256"] == ob["prompt_sha256"]).all()
    assert (oa["n_prompt_tokens"] == ob["n_prompt_tokens"]).all()
    assert set(ob["output_text"]) <= set(CHOICE["choice"])     # the echo backend honours the constraint
    assert not set(oa["output_text"]) & set(CHOICE["choice"])


def test_constraint_reaches_the_backend_and_the_cache_key(tmp_path):
    (tmp_path / "c").mkdir()
    c = Cfg(tmp_path / "c", decoding=GUIDED)
    req = make_requests(tmp_path / "r.parquet", arms=("forced_direct",), n_items=2, configs=("c0",), same_as={})
    df = pd.read_parquet(req)
    df["guided_json"] = [json.dumps(CHOICE), json.dumps({"regex": "-?[0-9]{1,7}\\}"})]
    df.to_parquet(tmp_path / "rg.parquet")
    be = c.echo()
    seen = []
    orig = be.generate

    def spy(convs, params, **kw):
        seen.extend(params)
        return orig(convs, params, **kw)
    be.generate = spy
    assert c.run(tmp_path / "rg.parquet", tmp_path / "o", backend_obj=be)["status"] == "complete"
    assert sorted(json.dumps(p["guided"], sort_keys=True) for p in seen) == sorted(
        [json.dumps(CHOICE, sort_keys=True), json.dumps({"regex": "-?[0-9]{1,7}\\}"}, sort_keys=True)])
    k1 = g.cache_key("m", "r", "x", {**seen[0], "guided": CHOICE})
    k2 = g.cache_key("m", "r", "x", {**seen[0], "guided": {"choice": ["A}", "B}"]}})
    assert k1 != k2


def test_missing_or_unexpected_constraint_is_refused(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "f").mkdir()
    req = make_requests(tmp_path / "r.parquet", arms=("forced_direct",), n_items=2, configs=("c0",), same_as={})
    with pytest.raises(g.RunRefused, match="has no guided_json"):
        Cfg(tmp_path / "c", decoding=GUIDED).run(req, tmp_path / "o1")
    greq = guided_requests(tmp_path / "rg.parquet", req)
    with pytest.raises(g.RunRefused, match="does not guide"):
        Cfg(tmp_path / "f", decoding=FORCED).run(greq, tmp_path / "o2")


def test_fingerprint_differs_from_the_unconstrained_arm(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = Cfg(tmp_path / "a", decoding=FORCED), Cfg(tmp_path / "b", decoding=GUIDED)
    fa = g.run_fingerprint(a.spec(), a.dec(), "echo")
    fb = g.run_fingerprint(b.spec(), b.dec(), "echo")
    assert fa != fb


def test_resume_skips_done_guided_requests(tmp_path):
    (tmp_path / "c").mkdir()
    c = Cfg(tmp_path / "c", decoding=GUIDED)
    req = make_requests(tmp_path / "r.parquet", arms=("forced_direct",), n_items=4, configs=("c0",), same_as={})
    greq = guided_requests(tmp_path / "rg.parquet", req)
    assert c.run(greq, tmp_path / "o", limit=2)["status"] == "complete"
    log = c.run(greq, tmp_path / "o")
    assert log["status"] == "complete" and log["requests_already_done"] == 2 and log["requests_this_run"] == 2
    assert len(shards(tmp_path / "o")) == 4


@pytest.mark.parametrize("new_api", [True, False])
def test_structured_kwargs_for_either_vllm_api(monkeypatch, new_api):
    class P:
        def __init__(self, **kw):
            self.kw = kw
    mod = types.ModuleType("vllm.sampling_params")
    if new_api:
        mod.StructuredOutputsParams = P
    else:
        mod.GuidedDecodingParams = P
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", mod)
    kw = g.structured_kwargs(CHOICE)
    key = "structured_outputs" if new_api else "guided_decoding"
    assert list(kw) == [key] and kw[key].kw == CHOICE
    assert g.structured_kwargs(None) == {}
