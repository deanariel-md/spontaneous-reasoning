"""The fair comparison on HotpotQA (scripts/hotpot_fair.py): choices
fitted on discovery rows only, the refusal when a saved file changed, and evaluation from the saved choices."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hotpot_fair as hf  # noqa: E402
import run_decision as r_dec  # noqa: E402
import run_hotpot as r  # noqa: E402
import score_hotpot as s  # noqa: E402
from src import decision as dc, decision_strict as ds, rule_budgets as rbud, transfer_regret as tr  # noqa: E402
from test_hotpot import FAMS, _hotpot_setup, synthetic_sources  # noqa: E402

PANEL = [f"m{i}" for i in range(9)]
ITEMS = [f"i{k}" for k in range(12)]
ARMS = r.contrast_arms()


def scored(configs, seed=4, shift=0.0):
    """Item-level strict scores for every model, arm, configuration and item (accuracy rises with the model index)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, m in enumerate(PANEL):
        for c in configs:
            for a, d in (("direct_answer", 0.0), ("zero_shot_cot", 0.15), ("forced_direct", 0.05),
                         ("fewshot_forced", 0.1)):
                p = min(0.95, max(0.02, 0.2 + 0.07 * j + d + shift))
                for q, i in enumerate(ITEMS):
                    x = int(rng.random() < p)
                    rows.append({"model_key": m, "arm": a, "config_id": c, "item_id": i, "strict_correct": x,
                                 "lenient_correct": x, "compliant": True, "strict_f1": float(x), "lenient_f1": float(x),
                                 # a token count for the mechanism forecast (no random draw, so scores are unchanged)
                                 "n_output_tokens": 10 + 12 * ((j + q) % 8) if a == "direct_answer" else 8})
    return pd.DataFrame(rows)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(hf, "N_PILOTS", 12)
    monkeypatch.setattr(hf, "N_ITEMS", 4)


def sampler(seed=5):
    rng = np.random.default_rng(seed)
    cfgs = [f"c{k:02d}" for k in range(16)]
    return {"discovery_config_ids": [str(x) for x in rng.choice(cfgs[:12], 24)],
            "validation_config_ids": [str(x) for x in rng.choice(cfgs, 24)], "sha256": "synthetic"}


ECI = list(np.linspace(100.0, 140.0, len(PANEL)))


# ----------------------------------------------------------------------------- no HotpotQA data


def test_cross_benchmark_reads_no_hotpot_data(tmp_path, monkeypatch):
    src = synthetic_sources(PANEL, r.CONTRASTS)

    def no_read(*a, **k):
        raise AssertionError("item 3a read a table")
    monkeypatch.setattr(pd, "read_parquet", no_read)
    rec = hf.save_cross(PANEL, ECI, src, str(tmp_path))
    rec2, arr = hf.load_cross(str(tmp_path), PANEL)
    assert rec2["sha256"] == rec["sha256"] and len(rec["design"]["pairs"]) == 9
    bbh = rec["design"]["pairs"]["bbh/cot_req"]
    assert bbh["models"] == PANEL[1:] and bbh["n_chosen"] == 3
    assert rec["design"]["pairs"]["gsm8k/fewshot"]["models"] == PANEL
    # source's own gain over every discovery draw: the top third of the source's mean gains
    units, ps, B, P = src[("math", "cot_forced")]
    w = arr["cross__math__cot_forced__source_own_all__weights"]
    assert np.array_equal(w[0], dc.choose((P - B).mean(0), 3)[0])
    # one-draw source choices: one per distinct source discovery configuration
    assert arr["cross__gsm8k__cot_req__source_own_1__weights"].shape == (len(dc.distinct_rows(units)), 9)
    # the index rule needs no call: its ranking follows the index, in the direction of the source's fit
    p = arr["cross__gsm8k__cot_req__source_index__pred"][0]
    assert np.all(np.diff(p) > 0) or np.all(np.diff(p) < 0)
    # identical rerun keeps the file; a different one is refused
    assert hf.save_cross(PANEL, ECI, src, str(tmp_path))["created"] == rec["created"]
    src2 = dict(src)
    u, ps, B, P = src2[("gsm8k", "cot_req")]
    src2[("gsm8k", "cot_req")] = (u, ps, B, P + np.linspace(0, 40, len(ps)))
    with pytest.raises(FileExistsError):
        hf.save_cross(PANEL, ECI, src2, str(tmp_path))
    with pytest.raises(SystemExit, match="another panel"):
        hf.load_cross(str(tmp_path), PANEL[::-1])


def test_cross_benchmark_absent_or_changed_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--stage cross"):
        hf.load_cross(str(tmp_path), PANEL)
    hf.save_cross(PANEL, ECI, synthetic_sources(PANEL, r.CONTRASTS), str(tmp_path))
    jp, npz = hf.files(hf.CROSS, str(tmp_path))
    with np.load(npz) as z:
        a = {k: z[k] for k in z.files}
    a["cross__gsm8k__cot_req__source_own_all__weights"] = a["cross__gsm8k__cot_req__source_own_all__weights"][:, ::-1]
    np.savez_compressed(npz, **a)
    with pytest.raises(SystemExit, match="does not match"):
        hf.load_cross(str(tmp_path), PANEL)


# ----------------------------------------------------------------------------- items 1, 2, 3b: discovery rows only


def _discovery(tmp_path, sp, table, name="scored.parquet"):
    path = str(tmp_path / name)
    table.to_parquet(path)
    return path


def test_discovery_choices_use_discovery_rows_only(tmp_path, small):
    sp = sampler()
    disc, val = sp["discovery_config_ids"], sp["validation_config_ids"]
    allc = sorted(set(disc) | set(val))
    src = synthetic_sources(PANEL, r.CONTRASTS)
    base = scored(allc)
    # two full tables that agree on the discovery configurations and differ everywhere else
    other = scored(allc, seed=99, shift=-0.15)
    keep = base["config_id"].isin(set(disc))
    mixed = pd.concat([base[keep], other[~other["config_id"].isin(set(disc))]], ignore_index=True)
    stems = {}
    for k, tab in (("a", base), ("b", mixed), ("c", base[keep])):
        d = tmp_path / k
        d.mkdir()
        hf.save_cross(PANEL, ECI, src, str(d))
        out = hf.save_discovery(_discovery(d, sp, tab), disc, PANEL, FAMS, ECI, ITEMS, ARMS, str(d))
        stems[k] = {st: out[st]["content_sha256"] for st in hf.DISCOVERY_FILES}
    assert stems["a"] == stems["b"] == stems["c"]
    # the choices change when the discovery rows change
    d = tmp_path / "d"
    d.mkdir()
    hf.save_cross(PANEL, ECI, src, str(d))
    flipped = base.copy()
    flipped.loc[flipped["config_id"].isin(set(disc)), "strict_correct"] ^= 1
    out = hf.save_discovery(_discovery(d, sp, flipped), disc, PANEL, FAMS, ECI, ITEMS, ARMS, str(d))
    assert out[hf.STRICT]["content_sha256"] != stems["a"][hf.STRICT]
    # refitting into a folder holding other choices is refused
    with pytest.raises(FileExistsError):
        hf.save_discovery(_discovery(tmp_path / "a", sp, flipped, "x.parquet"), disc, PANEL, FAMS, ECI, ITEMS, ARMS,
                          str(tmp_path / "a"))


def test_strict_design_matches_decision_strict(tmp_path, small):
    sp = sampler()
    disc = sp["discovery_config_ids"]
    path = _discovery(tmp_path, sp, scored(sorted(set(disc))))
    configs, Sd = hf.dense(path, PANEL, disc, ITEMS)
    rec, arr = hf.strict_record(ITEMS, disc, PANEL, ECI, Sd, {c: i for i, c in enumerate(configs)}, ARMS)
    mask = arr["item_split_pilot_pool"]
    assert mask.sum() == 6 and np.array_equal(mask, ds.split_items(ITEMS, "hotpot", "decision-strict"))
    assert mask[arr["hotpot__disjoint__items"]].all()                      # disjoint pilots: pilot half only
    assert rec["design"]["designs"]["disjoint"]["evaluation_items"] == 6
    assert rec["design"]["designs"]["shared"]["evaluation_items"] == 12
    # one pilot recomputed by hand: own gain is the 4-item gain on the drawn configuration
    p, it = arr["hotpot__disjoint__positions"][0], arr["hotpot__disjoint__items"][0]
    row = configs.index(disc[p])
    g = (Sd["zero_shot_cot"][row][:, it].mean(1) - Sd["forced_direct"][row][:, it].mean(1)) * 100
    assert np.allclose(arr["hotpot__disjoint__cot_forced__own_gain__pred"][0], g)
    for c in ARMS:
        for rule in ds.RULES:
            W = arr[f"hotpot__shared__{c}__{rule}__weights"]
            assert W.shape == (12, 9) and np.allclose(W.sum(1), dc.third_size(9))


def test_family_and_cross_discovery_reuse_transfer_regret(tmp_path, small):
    import run_transfer_regret as rtr
    sp = sampler()
    disc = sp["discovery_config_ids"]
    rng = np.random.default_rng(1)
    D = {c: (rng.normal(50, 10, (24, 9)), rng.normal(60, 10, (24, 9))) for c in ARMS}
    rec, arr = hf.family_record(disc, PANEL, FAMS, ECI, D)
    assert rec["design"]["contrasts"]["fewshot"]["n_chosen"] == {"A": 1, "B": 1, "C": 1}
    x = {"facts": {"families": FAMS, "eci": ECI, "discovery_config_ids": disc},
         "discovery": {"base": D["cot_req"][0], "cot": D["cot_req"][1]}}
    W, _, _ = rtr.family_choices(x, "base")
    assert np.array_equal(arr["family__cot_req__B__transport_quadratic_index"], W["B"]["transport_quadratic_index"])
    crec, carr = hf.cross_record(PANEL, ECI, synthetic_sources(PANEL, r.CONTRASTS))
    brec, barr = hf.cross_disc_record(disc, PANEL, D, crec, carr)
    k = "cross__bbh__cot_forced"
    jt = [PANEL.index(m) for m in crec["design"]["pairs"]["bbh/cot_forced"]["models"]]
    B = D["cot_forced"][0][:, jt]
    r0 = dc.distinct_rows(disc)[0]
    pred = tr.fit_apply(carr[f"{k}__source_baseline_mean"], carr[f"{k}__source_gain_mean"], carr[f"{k}__eci"],
                        B[r0], carr[f"{k}__eci"], "baseline_linear")
    assert np.allclose(barr[f"{k}__source_linear__pred"][0], pred)
    assert barr[f"{k}__target_own_1__weights"].shape == (len(dc.distinct_rows(disc)), 8)


# ----------------------------------------------------------------------------- refusal when a saved file changed


def test_index_must_list_new_files_unchanged(tmp_path, monkeypatch):
    a, sp, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    open(tranche, "w").write("{}")
    disc = s.discovery_configs(sp)
    scored(disc).to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    idx = r.stage_discovery(a)
    ip = os.path.join(a.pred_dir, r.DISCOVERY_INDEX)
    names = {os.path.basename(k) for k in idx["files"]}
    for f in hf.all_files(""):
        assert f in names
    assert s.discovery_index_problems(ip) == []
    # each new file, changed after the index was written, blocks the scoring of the validation draws
    for f in hf.all_files(a.pred_dir):
        keep = open(f, "rb").read()
        open(f, "wb").write(keep + b" ")
        probs = s.discovery_index_problems(ip)
        assert any("changed" in p and os.path.basename(f) in p for p in probs)
        with pytest.raises(SystemExit, match="hard rule 6"):
            s.score_stage("all", tranche_pred=tranche, discovery_pred=ip)
        open(f, "wb").write(keep)
    assert s.discovery_index_problems(ip) == []
    # an index written before the fair comparison existed (the new files not listed) is refused too
    rec = json.load(open(ip))
    rec["files"] = {k: v for k, v in rec["files"].items() if "fair" not in k and "family" not in k
                    and "cross" not in k and "decision_strict" not in k}
    old = str(tmp_path / "old_index.json")
    json.dump(rec, open(old, "w"))
    probs = s.discovery_index_problems(old)
    assert len(probs) == 8 and all("is not listed" in p for p in probs)
    with pytest.raises(SystemExit, match="hard rule 6"):
        s.score_stage("all", tranche_pred=tranche, discovery_pred=old)


def test_discovery_stage_refuses_without_or_after_cross(tmp_path, monkeypatch):
    a, sp, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    open(os.path.join(a.pred_dir, "predictions_hotpot_tranche.json"), "w").write("{}")
    scored(s.discovery_configs(sp)).to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    # the cross-benchmark file dated after the first HotpotQA scores
    json.dump({"created": "2000-01-01 00:00:00"}, open(os.path.join(a.processed, "scored_discovery4.json"), "w"))
    with pytest.raises(SystemExit, match="after the four-draw table"):
        r.stage_discovery(a)
    json.dump({"created": "2999-01-01 00:00:00"}, open(os.path.join(a.processed, "scored_discovery4.json"), "w"))
    for f in hf.files(hf.CROSS, a.pred_dir):
        os.remove(f)
    with pytest.raises(SystemExit, match="--stage cross"):
        r.stage_discovery(a)


# ----------------------------------------------------------------------------- evaluation from the saved choices


def _saved(tmp_path, sp, table):
    d = tmp_path / "man"
    d.mkdir()
    hf.save_cross(PANEL, ECI, synthetic_sources(PANEL, r.CONTRASTS), str(d))
    path = _discovery(tmp_path, sp, table)
    hf.save_discovery(path, sp["discovery_config_ids"], PANEL, FAMS, ECI, ITEMS, ARMS, str(d))
    return str(d), path


def _validation(path, sp):
    val = sp["validation_config_ids"]
    vconf, Sv = hf.dense(path, PANEL, val, ITEMS)
    ci = {c: i for i, c in enumerate(vconf)}
    rows = [ci[u] for u in val]
    V = {c: (Sv[b][rows].mean(2) * 100, Sv[q][rows].mean(2) * 100) for c, (b, q) in ARMS.items()}
    return vconf, Sv, V


def test_evaluation_reads_the_saved_choices(tmp_path, small):
    sp = sampler()
    disc, val = sp["discovery_config_ids"], sp["validation_config_ids"]
    d, path = _saved(tmp_path, sp, scored(sorted(set(disc) | set(val))))
    saved = hf.load_saved(d, PANEL)
    vconf, Sv, V = _validation(path, sp)
    F = hf.evaluate_all(saved, disc, val, V, Sv, vconf, len(ITEMS), ARMS)
    fresh = [u for u in dict.fromkeys(val) if u not in set(disc)]
    e = F["strict"]["disjoint"]["cot_req"]["fresh"]
    assert e["configurations"] == fresh and e["n_configurations"] == len(fresh)
    # the strict regret recomputed from the saved weights
    _, arr = saved[hf.STRICT]
    mask = arr["item_split_pilot_pool"]
    w = ds.eval_weights(len(ITEMS), ~mask)
    ix = [vconf.index(u) for u in fresh]
    G = (Sv["zero_shot_cot"][ix].astype(float) - Sv["direct_answer"][ix].astype(float)) @ w
    want = dc.regret_plugin(arr["hotpot__disjoint__cot_req__own_gain__weights"], G, dc.third_size(9)).mean()
    assert np.isclose(e["rules"]["own_gain"]["regret_plugin"], want)
    assert set(F["family"]["fewshot"]["fresh"]["rules"]) == {"transport_linear", "transport_quadratic_index",
                                                              "transport_index", "target_own_1", "random"}
    cb = F["cross_benchmark"]["bbh/cot_forced"]
    assert cb["M"] == 8 and set(cb["fresh"]["rules"]) == set(hf.CROSS_EVAL_ORDER)
    assert "target_own_1" not in cb["fresh"]["paired_minus_target_own_1"]
    assert {k: v["sha256"] for k, v in F["saved"].items()} == {k: v[0]["sha256"] for k, v in saved.items()}
    # replace one saved choice set (with its hashes): the evaluation follows the file
    jp, npz = hf.files(hf.CROSS, d)
    rec, a2 = rbud.load_frozen(jp, npz)
    k = "cross__gsm8k__cot_req__source_own_all__weights"
    a2[k] = np.roll(a2[k], 1, axis=1)
    rbud.freeze(rec, a2, jp, npz, overwrite=True)
    F2 = hf.evaluate_all(hf.load_saved(d, PANEL), disc, val, V, Sv, vconf, len(ITEMS), ARMS)
    G2 = (V["cot_req"][1] - V["cot_req"][0])[[val.index(u) for u in fresh]]
    assert np.isclose(F2["cross_benchmark"]["gsm8k/cot_req"]["fresh"]["rules"]["source_own_all"]["regret_plugin"],
                      dc.regret_plugin(a2[k], G2, 3).mean())
    # a saved file whose npz no longer matches its record stops the evaluation
    with np.load(npz) as z:
        a3 = {kk: z[kk] for kk in z.files}
    a3[k] = a3[k][:, ::-1]
    np.savez_compressed(npz, **a3)
    with pytest.raises(SystemExit, match="does not match"):
        hf.load_saved(d, PANEL)


def test_summary_lines(tmp_path, small):
    sp = sampler()
    disc, val = sp["discovery_config_ids"], sp["validation_config_ids"]
    d, path = _saved(tmp_path, sp, scored(sorted(set(disc) | set(val))))
    vconf, Sv, V = _validation(path, sp)
    F = hf.evaluate_all(hf.load_saved(d, PANEL), disc, val, V, Sv, vconf, len(ITEMS), ARMS)
    txt = "\n".join(hf.summary_lines(F, r.TITLES))
    assert "Fair comparison" in txt and "BIG-Bench Hard-28 -> HotpotQA" in txt and "Held-out family" in txt


# ----------------------------------------------------------------------------- the whole stage order, synthetic


def test_full_run_evaluates_the_saved_fair_choices(tmp_path, monkeypatch):
    a, sp, panel, items = _hotpot_setup(tmp_path, monkeypatch)
    allc = sorted(set(sp["discovery_config_ids"]) | set(sp["validation_config_ids"]))
    table = scored(allc)
    four = s.tranche_configs(sp)
    table[table["config_id"].isin(set(four))].to_parquet(os.path.join(a.processed, "scored_discovery4.parquet"))
    r.stage_predict(a)
    disc = s.discovery_configs(sp)
    table[table["config_id"].isin(set(disc))].to_parquet(os.path.join(a.processed, "scored_discovery.parquet"))
    r.stage_discovery(a)
    assert s.discovery_index_problems(os.path.join(a.pred_dir, r.DISCOVERY_INDEX)) == []
    table.to_parquet(os.path.join(a.processed, "scored.parquet"))
    before = {f: open(f, "rb").read() for f in hf.all_files(a.pred_dir)}
    R = r.run_full(a)
    assert {f: open(f, "rb").read() for f in hf.all_files(a.pred_dir)} == before      # refit kept the saved files
    F = json.load(open(os.path.join(a.out, r.FAIR_REPORT)))
    assert set(F) == {"saved", "strict", "family", "cross_benchmark"}
    assert R["fair_comparison"]["saved"][hf.CROSS]["sha256"] == json.load(open(hf.files(hf.CROSS, a.pred_dir)[0]))["sha256"]
    assert "Fair comparison" in open(os.path.join(a.out, "summary.md"), encoding="utf-8").read()
    # both directions from the two contrasts' saved choices, and the mechanism forecast against validation
    H = R["decision_directions"]
    assert set(H["one_draw"]["rules"]["own_1"]) == {"requested_to_forced", "forced_to_requested"}
    W_req = rbud.load_frozen(os.path.join(a.pred_dir, "predictions_decision_hotpot.json"),
                             os.path.join(a.pred_dir, "predictions_decision_hotpot.npz"))[1]["own_1__weights"]
    W_fd = rbud.load_frozen(os.path.join(a.pred_dir, "predictions_decision_hotpot_forced.json"),
                            os.path.join(a.pred_dir, "predictions_decision_hotpot_forced.npz"))[1]["own_1__weights"]
    val = sp["validation_config_ids"]
    rows = r_dec.evaluation_rows(sp["discovery_config_ids"], val)["independent"]
    Bv, Pv = r.arrays(r.loader(os.path.join(a.processed, "scored.parquet"), "cot_req"), val, panel, items)
    want = dc.paired_difference(W_fd, W_req, (Pv - Bv)[rows], dc.third_size(len(panel)))["estimate"]
    assert np.isclose(H["one_draw"]["rules"]["own_1"]["forced_to_requested"]["estimate"], want)
    assert "disjoint" in H["pilot100"]["designs"]
    E = json.load(open(os.path.join(a.out, "mechanism_forecast_evaluation.json")))
    assert set(E["statements"]) == {"share", "pairs", "gain"}
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import macros_added as ma
    m = ma.hotpot_directions_macros(R)
    assert {"hqDdirFdToReq", "hqDdirReqToFd", "hqDdirPilotFdToReq", "hqMechPredRev", "hqMechGainHolds"} <= set(m)
    assert all(k.isalpha() for k in m)


def test_non_finite_saved_statement_is_not_testable():
    """A four-draw R outside [0, 1] saves C as "nan" and K as None or "inf"; the full run reports them as saved."""
    assert r.finite("nan") is None and r.finite("inf") is None and r.finite(None) is None and r.finite(0.5) == 0.5
    rng = np.random.default_rng(2)
    Bv, Pv = rng.normal(50, 5, (24, 9)), rng.normal(55, 5, (24, 9))
    st = {"a_predicted_C_one_draw": "nan", "a_tolerance": 0.05, "b_K_needed": {"0.80": "inf", "0.90": None},
          "d_R_benefit": -0.1, "d_own_gain_beats_capability": None}
    saved = {"statements": {c: st for c in r.CONTRASTS}}
    saved["statements"]["c_arm_switch"] = {"switch_reorders_more": True}
    saved["statements"]["c_arm_switch_split_half"] = {"switch_reorders_more": True}
    V = {c: (Bv, Pv) for c in r.CONTRASTS}
    D4 = {c: (Bv[:4], Pv[:4]) for c in r.CONTRASTS}
    mcfg = {"group_sizes": [1, 2], "seed": 1}
    res = r.evaluate_statements(saved, V, D4, mcfg, 10)
    x = res["fewshot"]
    assert x["a"]["predicted"] == "nan" and x["a"]["within_0.05"] is None
    assert x["b"]["0.80"]["K_needed"] == "inf" and "not a finite number" in x["b"]["0.80"]["note"]
    assert x["b"]["0.90"]["reached"] is None and x["d"]["holds"] is None
