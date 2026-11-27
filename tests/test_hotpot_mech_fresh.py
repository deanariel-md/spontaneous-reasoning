"""The HotpotQA mechanism forecast on validation configurations new to validation, on the synthetic
HotpotQA pipeline of tests/test_d22_hotpot.py (scripts/run_hotpot_mech_fresh.py, paper/macros_hotpot_mech_fresh.py)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "paper"))


def _full(tmp_path, monkeypatch):
    from test_hotpot_saved_rules import _stage
    r, s, a, sampler, panel, items, disc, table = _stage(tmp_path, monkeypatch, with_validation=True)
    four = s.tranche_configs(sampler)
    table[table["config_id"].isin(set(four))].to_parquet(os.path.join(a.processed, "scored_discovery4.parquet"))
    r.stage_predict(a)
    table[table["config_id"].isin(set(disc))].to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    r.stage_discovery(a)
    return r, a, sampler, panel, items, table


def test_refuses_before_the_full_run(tmp_path, monkeypatch):
    import run_hotpot_mech_fresh as f
    r, a, sampler, panel, items, table = _full(tmp_path, monkeypatch)
    table.to_parquet(os.path.join(a.processed, "scored.parquet"))
    with pytest.raises(SystemExit, match="full run"):
        f.run(a.processed, a.sampler, a.pred_dir, a.out, panel=panel, items=items)


def test_fresh_sets_after_the_full_run(tmp_path, monkeypatch):
    import run_hotpot_mech_fresh as f
    import macros_hotpot_mech_fresh as mb
    r, a, sampler, panel, items, table = _full(tmp_path, monkeypatch)
    table.to_parquet(os.path.join(a.processed, "scored.parquet"))
    r.run_full(a)
    written = json.load(open(os.path.join(a.out, "mechanism_forecast_evaluation.json")))
    R = f.run(a.processed, a.sampler, a.pred_dir, a.out, panel=panel, items=items)
    disc = set(map(str, sampler["discovery_config_ids"]))
    val = list(map(str, sampler["validation_config_ids"]))
    fresh = [u for u in val if u not in disc]
    assert R["fresh_draws"] == len(fresh) and R["fresh_distinct"] == len(set(fresh))
    assert R["reusing_discovery"] == len(val) - len(fresh)
    assert R["forecast_sha256"] == written["forecast_sha256"]
    for k in ("fresh_draws", "fresh_distinct"):
        E = R["sets"][k]
        assert E["pairs"] == len(panel) * (len(panel) - 1) // 2
        assert E["predicted_reversed"] == written["predicted_reversed"]   # the saved prediction is the same
    assert os.path.exists(os.path.join(a.out, "mechanism_forecast_fresh_clean.md"))
    # the written evaluation is untouched
    assert json.load(open(os.path.join(a.out, "mechanism_forecast_evaluation.json"))) == written
    m = mb.hotpot_mech_fresh_macros(R)
    assert {"hqMechFreshDraws", "hqMechFreshConfigs", "hqMechFreshDistinctShareHolds"} <= set(m)
    assert mb.hotpot_mech_fresh_macros(None) == {}


def test_refuses_a_changed_forecast(tmp_path, monkeypatch):
    import run_hotpot_mech_fresh as f
    r, a, sampler, panel, items, table = _full(tmp_path, monkeypatch)
    table.to_parquet(os.path.join(a.processed, "scored.parquet"))
    r.run_full(a)
    p = os.path.join(a.out, "mechanism_forecast_evaluation.json")
    E = json.load(open(p))
    E["forecast_sha256"] = "0" * 64
    json.dump(E, open(p, "w"))
    with pytest.raises(SystemExit, match="sha256"):
        f.run(a.processed, a.sampler, a.pred_dir, a.out, panel=panel, items=items)
