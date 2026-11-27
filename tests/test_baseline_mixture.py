"""The requested baseline as a mixture (src/baseline_mixture.py, scripts/run_baseline_mixture.py), and the
HotpotQA mechanism forecast on synthetic data (it cannot run on real data before HotpotQA's discovery table exists)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import baseline_mixture as bm, robustness as rb  # noqa: E402


def test_draw_weighted_share_counts_repeats_and_reports_missing():
    # model 0: config 0 all long, config 1 none long; config 0 drawn twice -> (1 + 1 + 0) / 3
    long_ = [1, 1, 0, 0, 0, 1]
    m = [0, 0, 0, 0, 1, 1]
    c = [0, 0, 1, 1, 0, 0]
    w = bm.draw_weighted_share(long_, m, c, 2, 2, [0, 0, 1])
    assert np.isclose(w[0], 2 / 3)
    assert np.isnan(w[1])                     # model 1 has no output in config 1: reported, not imputed


def test_correction_and_mechanism():
    g_fd = np.array([40.0, 30.0, 20.0, 10.0])
    w = np.array([0.9, 0.0, 0.5, 0.0])
    pred = bm.corrected_gain(g_fd, w)
    assert np.allclose(pred, [4.0, 30.0, 10.0, 10.0])
    # the observed gain over the requested arm equals the corrected one: no residual reversal
    m = bm.mechanism(pred, g_fd, w)
    assert m["observed_vs_corrected"]["reversed"] == 0
    assert m["observed_vs_forced"]["reversed"] == rb.pair_reversals(pred, g_fd)["reversed"] > 0
    assert m["observed_vs_corrected"]["ties"] == 1        # models 3 and 4 tie under the correction: neither
    assert np.isclose(m["corr_corrected"]["pearson"], 1.0)
    assert bm.reversed_pairs(g_fd, pred, list("abcd")) == [["a", "b"], ["a", "c"], ["a", "d"]]   # c, d tie


def test_item_level_pooled_with_bootstrap():
    rng = np.random.default_rng(0)
    model = np.repeat(np.arange(4), 10)
    lf = np.tile([0] * 6 + [1] * 4, 4).astype(bool)
    req = np.where(lf, 1, 0)
    out = bm.item_level(model, lf, {("requested", "strict"): req, ("cot", "strict"): np.ones(40)}, 4, 199, rng)
    assert out["short"]["n"] == 24 and np.isclose(out["short"]["share"], 0.6)
    assert out["short"]["acc"]["requested/strict"]["estimate"] == 0.0
    assert out["long"]["acc"]["requested/strict"]["estimate"] == 100.0
    assert out["long"]["acc"]["cot/strict"]["ci95"] == [100.0, 100.0]


def test_forecast_write_once_and_evaluation(tmp_path):
    panel = ["a", "b", "c", "d"]
    w = np.array([0.8, 0.0, 0.1, 0.0])
    g_fd = np.array([40.0, 30.0, 20.0, 10.0])
    g_rq = np.array([8.0, 29.0, 18.0, 11.0])
    rec = bm.forecast_record(panel, w, g_fd, g_rq, 64, ["c1", "c1", "c2"], {"x": 1})
    assert rec["predicted_reversal"]["reversed_pairs"] == [["a", "b"], ["a", "c"], ["a", "d"]]   # pred 8, 30, 18, 10
    assert np.isclose(rec["predicted_reversal"]["share"], 3 / 6)
    assert rec["rule"] == bm.RULE_TEXT and rec["forced_cap_tokens"] == 64
    p = str(tmp_path / "f.json")
    saved = bm.write_forecast(rec, p)
    assert "created" in saved and bm.load_forecast(p)["sha256"] == rec["sha256"]
    again = bm.write_forecast(bm.forecast_record(panel, w, g_fd, g_rq, 64, ["c1", "c1", "c2"], {"x": 1}), p)
    assert again["created"] == saved["created"]                 # identical: kept as saved
    with pytest.raises(FileExistsError):
        bm.write_forecast(bm.forecast_record(panel, w * 0.5, g_fd, g_rq, 64, ["c1"], {"x": 1}), p)
    # a file edited after it was saved is refused on loading
    txt = json.load(open(p))
    txt["per_model"]["a"]["w"] = 0.0
    json.dump(txt, open(p, "w"))
    with pytest.raises(ValueError, match="sha256"):
        bm.load_forecast(p)
    # evaluation: validation gains reverse (a, b), (a, d) and (c, d); hits (a, b), (a, d), false alarm (a, c), miss (c, d)
    E = bm.evaluate_forecast(rec, panel, np.array([5.0, 30.0, 4.0, 9.0]), np.array([40.0, 30.0, 20.0, 10.0]))
    assert E["hits"] == [["a", "b"], ["a", "d"]] and E["false_alarms"] == [["a", "c"]] and E["misses"] == [["c", "d"]]
    assert E["observed_reversed"] == len(E["hits"]) + len(E["misses"])
    st = E["statements"]
    assert np.isclose(st["pairs"]["rate_among_predicted"], 2 / 3) and np.isclose(st["pairs"]["rate_among_other"], 1 / 3)
    assert st["share"]["observed"] == E["observed_reversed"] / 6
    assert set(st) == {"share", "pairs", "gain"} and all("holds" in v for v in st.values())
    with pytest.raises(ValueError, match="another panel"):
        bm.evaluate_forecast(rec, panel[::-1], np.zeros(4), np.zeros(4))


def test_hotpot_forecast_from_the_discovery_table_only(tmp_path, monkeypatch):
    """The forecast path of run_hotpot.py --stage discovery on a synthetic table: discovery rows only, the forced cap of
    config/decoding_hotpot.yaml, the file listed in the discovery index; the full run evaluates it on validation."""
    import run_baseline_mixture as rbm
    import run_hotpot as r
    import score_hotpot as s
    from test_hotpot import _hotpot_setup, _synthetic_scored
    a, sampler, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    open(os.path.join(a.pred_dir, "predictions_hotpot_tranche.json"), "w").write("{}")
    disc = s.discovery_configs(sampler)
    table = _synthetic_scored(disc, panel, items)
    path = os.path.join(a.processed, "scored_discovery.parquet")
    # without token counts the forecast is not saved and the stage stops (the validation draws stay unscored)
    table.drop(columns="n_output_tokens").to_parquet(path)
    with pytest.raises(SystemExit, match="n_output_tokens"):
        r.stage_discovery(a)
    table.to_parquet(path)
    idx = r.stage_discovery(a)
    fp = os.path.join(a.pred_dir, rbm.HOTPOT_FORECAST)
    assert os.path.basename(fp) in {os.path.basename(k) for k in idx["files"]}
    rec = bm.load_forecast(fp)
    assert rec["forced_cap_tokens"] == 64 and rec["models"] == panel
    # w by hand: requested outputs of the discovery configurations longer than 64 tokens, averaged over the draws
    d = table[table["arm"] == "direct_answer"]
    for j, k in enumerate(panel):
        x = d[d["model_key"] == k]
        per = (x["n_output_tokens"] > 64).groupby(x["config_id"]).mean()
        assert np.isclose(rec["per_model"][k]["w"], np.mean([per[c] for c in sampler["discovery_config_ids"]]))
    # the gain over the forced arm is the discovery mean of chain of thought minus forced direct
    Bf, Pf = r.arrays(r.loader(path, "cot_forced"), sampler["discovery_config_ids"], panel, items)
    assert np.allclose([rec["per_model"][k]["gain_vs_forced"] for k in panel], (Pf - Bf).mean(0))
    # the hq_after.sh step: recomputing keeps the identical file
    monkeypatch.setattr(r, "PROCESSED", a.processed)
    monkeypatch.setattr(r, "SAMPLER", a.sampler)
    monkeypatch.setattr(r, "MANIFESTS", a.pred_dir)
    assert rbm.hotpot_forecast()["created"] == rec["created"]
    # a validation row in the table does not enter the forecast
    extra = _synthetic_scored(["zz-validation-only"], panel, items, seed=9)
    pd.concat([table, extra]).to_parquet(path)
    assert rbm.hotpot_forecast()["sha256"] == rec["sha256"]
    # evaluation against synthetic validation draws, every statement reported
    V = {c: (np.random.default_rng(1).normal(40, 5, (24, len(panel))),
             np.random.default_rng(2).normal(60, 5, (24, len(panel)))) for c in ("cot_req", "cot_forced")}
    E = rbm.evaluate_hotpot_forecast(a.pred_dir, panel, V)
    assert E["forecast_sha256"] == rec["sha256"] and E["pairs"] == len(panel) * (len(panel) - 1) // 2
    assert set(E["statements"]) == {"share", "pairs", "gain"}
    assert "Mechanism forecast" in "\n".join(rbm.hotpot_forecast_lines(E))

