"""The split pilot (src/split_pilot.py, scripts/run_split_pilot.py), the exact decomposition of the
baseline shift (src/baseline_mixture.decomposition), and the two HotpotQA rule files saved at its discovery stage
(manifests/predictions_hotpot_split.json, predictions_hotpot_decomposition.json) on synthetic data."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import baseline_mixture as bm, decision as dc, split_pilot as sp  # noqa: E402


# ----------------------------------------------------------------------------- the split pilot


def test_assignment_half_per_stratum_same_for_every_model_and_seeded():
    a = sp.assignment(200, 100, sp.SEED, 25)
    assert a.shape == (200, 100) and (a.reshape(200, 25, 4).sum(2) == 2).all()
    assert np.array_equal(a, sp.assignment(200, 100, sp.SEED, 25))
    assert not np.array_equal(a, sp.assignment(200, 100, 0, 25))
    b = sp.assignment(5, 100)
    assert (b.sum(1) == 50).all()
    c = sp.assignment(3, 5)            # odd: floor(5 / 2) requested
    assert (c.sum(1) == 2).all()
    with pytest.raises(ValueError):
        sp.assignment(2, 10, n_strata=3)


def test_split_gains_by_hand():
    rng = np.random.default_rng(0)
    S = {a: rng.integers(0, 2, (3, 4, 6)) for a in ("req", "fd", "cot")}   # configs, models, items
    rows, items = np.array([2, 0]), np.array([[0, 2, 3, 5], [1, 2, 4, 5]])
    rq = np.array([[True, False, True, False], [False, False, True, True]])
    g = sp.split_gains(S["req"], S["fd"], S["cot"], rows, items, rq)
    for p in range(2):
        for m in range(4):
            base = [S["req"][rows[p], m, i] if q else S["fd"][rows[p], m, i] for i, q in zip(items[p], rq[p])]
            want = 100 * np.mean([S["cot"][rows[p], m, i] for i in items[p]]) - 100 * np.mean(base)
            assert np.isclose(g[p, m], want)
    s = sp.single_gains(S["req"], S["cot"], rows, items)
    assert np.isclose(s[1, 3], 100 * (S["cot"][0, 3, items[1]] - S["req"][0, 3, items[1]]).mean())


def test_compare_takes_the_better_single_by_worst_regret_and_reports_components():
    M, n = 6, 2
    G = {"requested": np.tile(np.array([5, 4, 3, 2, 1, 0.0]), (4, 1)) + np.arange(4)[:, None] * 0.01,
         "forced": np.tile(np.array([0, 1, 2, 3, 4, 5.0]), (4, 1))}
    top = lambda j: np.eye(M)[list(j)].sum(0)[None]  # noqa: E731
    Wd = {"requested_only": top([0, 1]), "forced_only": top([4, 5]), "split": top([0, 5])}
    c = sp.compare(Wd, G, n)
    # requested-only: 0 / 4; forced-only: 4 / 0; split: 2 / 2 (pp per chosen model)
    assert np.isclose(c["regret"]["requested_only"]["forced"], 4) and np.isclose(c["regret"]["split"]["worst"], 2)
    assert c["better_single"] == "requested_only" and np.isclose(c["advantage"], 2) and c["split_better"]
    assert set(c["paired"]) == {f"split_minus_{p}/{d}" for p in ("requested_only", "forced_only")
                                for d in ("requested", "forced")}
    lo = sp.leave_one_family_out({p: np.tile(np.array([5, 4, 3, 2, 1, 0.0]) * (1 if p != "forced_only" else -1), (2, 1))
                                  for p in sp.POLICIES}, G, ["a", "a", "b", "b", "c", "c"])
    assert lo["families"] == 3 and len(lo["advantage_range"]) == 2


def test_retrospective_reproduces_ref_table():
    path = os.path.join(ROOT, "results", "split_pilot", "split_pilot.json")
    if not os.path.exists(path):
        pytest.skip("results/split_pilot/ not built")
    R = json.load(open(path, encoding="utf-8"))
    for t in ("gsm8k", "math", "bbh"):
        assert R["tasks"][t]["ref_check"]["reproduced"], R["tasks"][t]["ref_check"]["mismatches"]
        assert R["tasks"][t]["main"]["split_better"]


# ----------------------------------------------------------------------------- the decomposition


def _rows(seed=0, M=6, n=400):
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(M), n)
    w = np.linspace(0.05, 0.95, M)
    lf = rng.random(M * n) < w[m]
    c = (rng.random(M * n) < 0.8).astype(int)
    f = (rng.random(M * n) < 0.3).astype(int)
    r = np.where(lf, c * (rng.random(M * n) < 0.7), f).astype(int)
    return m, lf, r, f, c, M


def test_decomposition_adds_up_and_shares_sum_to_one():
    m, lf, r, f, c, M = _rows()
    d = bm.decomposition(m, lf, r, f, c, M)
    p = d["per_model"]
    assert np.allclose(np.array(p["part_long"]) + np.array(p["part_short"]), np.array(p["shift"]))
    assert np.isclose(d["variance_share_long"] + d["variance_share_short"], 1.0)
    t = d["variance_terms"]
    assert np.isclose(t["long"] + t["short"] + t["two_cov"], 1.0)
    # by hand for model 2
    j = m == 2
    assert np.isclose(p["w"][2], lf[j].mean())
    assert np.isclose(p["part_long"][2], 100 * ((r - f)[j] * lf[j]).sum() / j.sum())
    rl, fl, cl = (x[j & lf].mean() for x in (r, f, c))
    assert np.isclose(p["recovery_long"][2], (rl - fl) / (cl - fl))
    # the two forms
    C, F, R = (np.array(p[k]) for k in ("C", "F", "R"))
    assert np.allclose(p["predicted_pooled"], (1 - np.array(p["w"])) * (C - F))
    assert np.allclose(p["predicted_short"], (1 - np.array(p["w"])) * (np.array(p["C_short"]) - np.array(p["F_short"])))
    assert d["forms"]["pooled"]["pairs"] == M * (M - 1) // 2


def test_decomposition_row_weights_and_edge_cases():
    m, lf, r, f, c, M = _rows(1, M=4, n=50)
    wt = np.where(np.arange(len(m)) % 2, 2.0, 1.0)
    d = bm.decomposition(m, lf, r, f, c, M, wt)
    j = m == 1
    assert np.isclose(d["per_model"]["w"][1], (wt[j] * lf[j]).sum() / wt[j].sum())
    # a model whose requested outputs are all long: no short stratum, short form reported as 0 and listed
    lf2 = lf.copy()
    lf2[m == 0] = True
    d2 = bm.decomposition(m, lf2, r, f, c, M)
    assert d2["per_model"]["w"][0] == 1.0 and d2["per_model"]["predicted_short"][0] == 0.0
    assert 0 in d2["short_stratum_empty_models"]
    # a model with no long row: recovery undefined, listed among the exclusions
    lf3 = lf.copy()
    lf3[m == 3] = False
    d3 = bm.decomposition(m, lf3, r, f, c, M)
    assert any(x["model"] == 3 for x in d3["recovery_long_excluded"])
    # equal shifts across models: the variance share is not defined and is reported as None
    d4 = bm.decomposition(np.repeat(np.arange(3), 2), [True, False] * 3, [1, 0] * 3, [0, 0] * 3, [1, 1] * 3, 3)
    assert d4["variance_share_long"] is None


def test_corrected_choice_values_under_the_requested_arm():
    g_fd = np.array([50, 40, 30, 20, 10, 5.0])
    w = np.array([0.9, 0.9, 0.0, 0.0, 0.5, 0.0])
    g_req = (1 - w) * g_fd
    out = bm.corrected_choice(g_fd, w, g_req, 2)
    assert np.isclose(out["corrected"]["regret"], 0.0) and out["forced_gain"]["regret"] > 0


# ----------------------------------------------------------------------------- HotpotQA: saved at discovery, refused if absent


def _stage(tmp_path, monkeypatch, with_validation=False):
    import run_hotpot as r
    import score_hotpot as s
    from test_hotpot import _hotpot_setup, _synthetic_scored
    a, sampler, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    disc = s.discovery_configs(sampler)
    allc = sorted(set(sampler["discovery_config_ids"]) | set(sampler["validation_config_ids"]))
    table = _synthetic_scored(allc if with_validation else disc, panel, items)
    return r, s, a, sampler, panel, items, disc, table


def test_discovery_stage_writes_both_files_from_discovery_rows(tmp_path, monkeypatch):
    import hotpot_fair as hf
    import run_baseline_mixture as rbm
    import run_split_pilot as rsp
    from src import rule_budgets as rbud
    r, s, a, sampler, panel, items, disc, table = _stage(tmp_path, monkeypatch)
    open(os.path.join(a.pred_dir, "predictions_hotpot_tranche.json"), "w").write("{}")
    path = os.path.join(a.processed, "scored_discovery.parquet")
    table.to_parquet(path)
    idx = r.stage_discovery(a)
    names = {os.path.basename(k) for k in idx["files"]}
    assert {rsp.HOTPOT_SPLIT, rbm.HOTPOT_DECOMPOSITION} <= names
    assert {rsp.HOTPOT_SPLIT, rbm.HOTPOT_DECOMPOSITION} <= set(s.required_rule_files())
    # the split file: the strict design's disjoint pilots, its own-gain choices as the single-baseline pilots
    rec = bm.load_forecast(os.path.join(a.pred_dir, rsp.HOTPOT_SPLIT))
    srec, sarr = rbud.load_frozen(*hf.files(hf.STRICT, a.pred_dir))
    assert rec["pilots"]["strict_sha256"] == srec["sha256"] and rec["models"] == panel
    assert np.allclose(rec["weights"]["requested_only"], sarr["hotpot__disjoint__cot_req__own_gain__weights"])
    assert np.allclose(rec["weights"]["forced_only"], sarr["hotpot__disjoint__cot_forced__own_gain__weights"])
    n = dc.third_size(len(panel))
    assert np.allclose(np.asarray(rec["weights"]["split"]).sum(1), n)
    rq = np.asarray(rec["assignment_requested"], bool)
    assert (rq.sum(1) == rq.shape[1] // 2).all() and rec["assignment_seed"] == sp.SEED
    # split gains of one pilot by hand from the discovery rows
    p0, its = rec["pilots"]["positions"][0], [items[i] for i in rec["pilots"]["items"][0]]
    cfg = sampler["discovery_config_ids"]
    x = table[(table["config_id"] == cfg[p0]) & table["item_id"].isin(its)]
    for j, k in enumerate(panel):
        y = x[x["model_key"] == k].set_index(["arm", "item_id"])["strict_correct"]
        base = [y[("direct_answer" if q else "forced_direct", i)] for i, q in zip(its, rq[0])]
        want = 100 * np.mean([y[("zero_shot_cot", i)] for i in its]) - 100 * np.mean(base)
        assert np.isclose(rec["predictions"]["split"][0][j], want)
    # the decomposition file: w and forced gain from the discovery rows (each configuration weighted by its draws)
    dr = bm.load_forecast(os.path.join(a.pred_dir, rbm.HOTPOT_DECOMPOSITION))
    assert dr["threshold_variance_share_long"] == 0.5 and dr["forced_cap_tokens"] == 64
    req = table[table["arm"] == "direct_answer"]
    for k in panel:
        q = req[req["model_key"] == k]
        per = (q["n_output_tokens"] > 64).groupby(q["config_id"]).mean()
        assert np.isclose(dr["per_model"][k]["w"], np.mean([per[c] for c in sampler["discovery_config_ids"]]))
    assert np.isclose(sum(dr["choices"]["corrected"]["weights"]), n)
    # identical recomputation keeps both files; a different one is refused
    created = (rec["created"], dr["created"])
    r.stage_discovery(a)
    assert (bm.load_forecast(os.path.join(a.pred_dir, rsp.HOTPOT_SPLIT))["created"],
            bm.load_forecast(os.path.join(a.pred_dir, rbm.HOTPOT_DECOMPOSITION))["created"]) == created
    t2 = table.copy()
    t2.loc[t2["arm"] == "direct_answer", "n_output_tokens"] = 200
    t2.to_parquet(path)
    with pytest.raises(FileExistsError):
        rbm.hotpot_decomposition(a.processed, a.sampler, a.pred_dir, None)


def test_scoring_refused_while_either_file_is_absent_or_changed(tmp_path, monkeypatch):
    import run_baseline_mixture as rbm
    import run_split_pilot as rsp
    r, s, a, sampler, panel, items, disc, table = _stage(tmp_path, monkeypatch)
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    open(tranche, "w").write("{}")
    table.to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    r.stage_discovery(a)
    ip = os.path.join(a.pred_dir, r.DISCOVERY_INDEX)
    assert s.discovery_index_problems(ip) == []
    for name in (rsp.HOTPOT_SPLIT, rbm.HOTPOT_DECOMPOSITION):
        f = os.path.join(a.pred_dir, name)
        keep = open(f, "rb").read()
        os.remove(f)
        assert any("absent" in p and name in p for p in s.discovery_index_problems(ip))
        with pytest.raises(SystemExit, match="hard rule 6"):
            s.score_stage("all", tranche_pred=tranche, discovery_pred=ip)
        open(f, "wb").write(keep + b" ")
        assert any("changed" in p and name in p for p in s.discovery_index_problems(ip))
        with pytest.raises(SystemExit, match="hard rule 6"):
            s.score_stage("all", tranche_pred=tranche, discovery_pred=ip)
        open(f, "wb").write(keep)
    assert s.discovery_index_problems(ip) == []
    # an index that lists neither file is refused
    rec = json.load(open(ip))
    rec["files"] = {k: v for k, v in rec["files"].items() if "split" not in k and "decomposition" not in k}
    old = str(tmp_path / "old_index.json")
    json.dump(rec, open(old, "w"))
    probs = s.discovery_index_problems(old)
    assert len(probs) == 2 and all("is not listed" in p for p in probs)
    with pytest.raises(SystemExit, match="hard rule 6"):
        s.score_stage("all", tranche_pred=tranche, discovery_pred=old)


def test_full_run_evaluates_both_files(tmp_path, monkeypatch):
    r, s, a, sampler, panel, items, disc, table = _stage(tmp_path, monkeypatch, with_validation=True)
    four = s.tranche_configs(sampler)
    table[table["config_id"].isin(set(four))].to_parquet(os.path.join(a.processed, "scored_discovery4.parquet"))
    r.stage_predict(a)
    table[table["config_id"].isin(set(disc))].to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    r.stage_discovery(a)
    table.to_parquet(os.path.join(a.processed, "scored.parquet"))
    R = r.run_full(a)
    E = json.load(open(os.path.join(a.out, "split_pilot_evaluation.json")))
    assert set(E["sets"]) == {"fresh", "all_distinct"} and "holds" in E["primary"]
    for ev in E["sets"].values():
        if "regret" in ev:
            assert set(ev["regret"]) == set(sp.POLICIES)
            assert all(set(v) == {"requested", "forced", "worst"} for v in ev["regret"].values())
    D = json.load(open(os.path.join(a.out, "decomposition_evaluation.json")))
    assert set(D["statements"]) == {"variance_share", "choice"}
    assert D["statements"]["variance_share"]["threshold"] == 0.5
    assert isinstance(D["statements"]["choice"]["holds"], bool)
    txt = open(os.path.join(a.out, "summary.md"), encoding="utf-8").read()
    assert "Split pilot (saved before validation" in txt and "Decomposition rule (saved before validation" in txt
    assert R["split_pilot"]["saved"]["sha256"] == E["saved"]["sha256"]
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import macros_added as ma
    m = ma.hotpot_d27_macros(R)
    assert {"hqDecVarLongPct", "hqDecVarHolds", "hqDecChoiceHolds", "hqDecRegretCorr"} <= set(m)
    assert all(k.isalpha() for k in m)
    assert ma.hotpot_d27_macros({}) == {}


def test_post_hoc_macros_from_the_results_files():
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import macros_added as ma
    p = os.path.join(ROOT, "results", "split_pilot", "split_pilot.json")
    q = os.path.join(ROOT, "results", "baseline_mixture", "baseline_mixture.json")
    if not (os.path.exists(p) and os.path.exists(q)):
        pytest.skip("results not built")
    m = ma.split_pilot_macros(json.load(open(p, encoding="utf-8")))
    assert m["splitpGsmSplitReq"] == "1.21" and m["splitpGsmSplitFd"] == "1.83" and m["splitpLofoAdvMin"] == "0.40"
    d = ma.decomposition_macros(json.load(open(q, encoding="utf-8")))
    assert d["bmixDecMathVarLongPctContent"] == "95" and d["bmixDecGsmChoiceFd"] == "9.8"
    assert ma.decomposition_macros({}) == {}

