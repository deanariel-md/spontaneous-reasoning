"""The arm-wording mechanism forecast (src/arm_wording_mechanism.py,
scripts/build_arm_wording_mechanism.py), its refusal in scripts/score_arm_wording.py and its evaluation in
scripts/run_arm_wording.py, on synthetic data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from src import arm_wording as aw, arm_wording_mechanism as awm  # noqa: E402

M, D = 8, 4
MODELS = [f"m{m}" for m in range(M)]
CLAUSES = [f"dir{d}" for d in range(D)]


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in aw.TASKS:
        w = np.clip(rng.uniform(0, 1, (D, M)), 0, 1)
        w[:, 0] = 1.0                                   # a model at w = 1 under every clause
        out[t] = {"w": w, "g_fd": rng.uniform(5, 60, M), "g_short": np.r_[np.nan, rng.uniform(5, 60, M - 1)],
                  "rho": 0.5, "w_collection_current": w[0] + rng.normal(0, 0.02, M)}
    return out


def test_predicted_gains_and_shares_by_hand():
    w = np.array([[0.0, 0.5, 1.0], [0.5, 0.5, 0.0]])
    g = np.array([10.0, 20.0, 30.0])
    gs = np.array([12.0, np.nan, 28.0])
    P = awm.predicted_gains(w, g, gs, 0.5)
    assert np.allclose(P["pooled"], [[10, 10, 0], [5, 10, 30]])
    assert np.allclose(P["short_stratum"], [[12, 0, 0], [6, 0, 28]])
    assert np.allclose(P["partial"], [[10, 15, 15], [7.5, 15, 30]])
    sh = awm.predicted_shares(P, g)
    # pooled: clause 0 ties m0/m1 (not reversed) and reverses m0-m2 and m1-m2 against clause 1
    assert np.isclose(sh["pooled"]["clause_reversal"], awm.reversed_share(P["pooled"][0], P["pooled"][1]))
    assert np.isclose(awm.reversed_share([1, 2, 3], [3, 2, 1]), 1.0)
    assert np.isclose(awm.reversed_share([1, 1, 3], [3, 2, 1]), 2 / 3)      # the tied pair is not reversed


def test_record_write_once_sidecar_and_checks(tmp_path):
    rec = awm.forecast_record(MODELS, CLAUSES, _inputs(), {"x": 1})
    assert set(rec["per_task"]) == set(aw.TASKS) and rec["tolerance"] == 0.05 and rec["w_range"] == 0.2
    pm = rec["per_task"]["gsm8k"]["per_model"]["m0"]
    assert pm["g_forced_short"] is None and set(pm["by_clause"]) == set(CLAUSES)
    assert pm["by_clause"]["dir1"]["predicted_gain_vs_requested_pooled"] == 0.0     # w = 1
    for t in aw.TASKS:
        lo, hi = rec["per_task"][t]["predicted"]["clause_reversal_range"]
        assert lo <= hi and len(rec["per_task"][t]["predicted"]["arm_switch_by_clause_range"]) == D
    p, s = tmp_path / "m.json", tmp_path / "m.sha256"
    with pytest.raises(ValueError, match="not saved"):
        awm.check_forecast(p, s)
    saved = awm.write_forecast(rec, p, s)
    assert awm.check_forecast(p, s) and saved["created"]
    assert awm.write_forecast(rec, p, s)["created"] == saved["created"]            # identical: kept
    other = awm.forecast_record(MODELS, CLAUSES, _inputs(1), {"x": 1})
    with pytest.raises(FileExistsError):
        awm.write_forecast(other, p, s)
    raw = p.read_bytes()
    p.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="changed"):
        awm.check_forecast(p, s)
    p.write_bytes(raw)
    # an edit with the sidecar recomputed is still caught by the content hash
    j = json.loads(raw)
    j["per_task"]["gsm8k"]["rho"] = 0.9
    p.write_text(json.dumps(j))
    s.write_text(aw.sha256_file(p) + "  m.json\n")
    with pytest.raises(ValueError, match="content hash"):
        awm.check_forecast(p, s)


def test_scoring_refused_at_both_stages_without_the_forecast(tmp_path):
    import score_arm_wording as saw
    for stage in ("discovery", "all"):
        a = argparse.Namespace(base=str(tmp_path), synthetic=None, pilots=False, check_runs=False, stage=stage,
                               config=None, items_dir=None, allow_test_backend=True)
        with pytest.raises(aw.ArmWordingError, match="mechanism forecast"):
            saw.run(a)
    paths = aw.Paths(tmp_path)
    awm.write_forecast(awm.forecast_record(MODELS, CLAUSES, _inputs(), {}), paths.mechanism, paths.mechanism_sha)
    assert saw.check_mechanism(paths)
    paths.mechanism_sha.write_text("0" * 64 + "  x\n")
    with pytest.raises(aw.ArmWordingError, match="changed"):
        saw.check_mechanism(paths)


def _scored(C=6, T=2, seed=3):
    """A tidy arm-wording table with requested accuracy rising with w and forced accuracy fixed across clauses."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, (D, M))
    base = np.linspace(0.2, 0.6, M)
    rows = []
    for c in range(C):
        for arm, ws in ((aw.COT, [f"cot{t}" for t in range(T)]), (aw.REQ, CLAUSES), (aw.FD, CLAUSES)):
            for k, wd in enumerate(ws):
                for m in range(M):
                    if arm == aw.COT:
                        p = base[m] + 0.3
                    elif arm == aw.REQ:
                        p = base[m] + 0.3 * w[k, m] * 0.6
                    else:
                        p = base[m]
                    for i in range(60):
                        long_ = arm == aw.REQ and rng.random() < w[k, m]
                        rows.append({"task": "gsm8k", "config": f"k{c}", "wording": wd, "arm": arm,
                                     "model_key": MODELS[m], "item_id": str(i),
                                     "strict_correct": float(rng.random() < p), "lenient_correct": 1.0,
                                     "n_output_tokens": 40 if long_ else 8, "box_closed": True, "prompt_sha256": ""})
    return pd.DataFrame(rows), w


def test_evaluation_reports_every_statement():
    tab, w = _scored()
    ta = aw.TaskArrays(tab, "gsm8k", [f"k{c}" for c in range(6)], ["cot0", "cot1"], CLAUSES, MODELS)
    acc = ta.acc()
    rev = aw.reversal_table(aw.gains(acc))
    inp = _inputs()
    inp["gsm8k"]["w"] = w
    inp["gsm8k"]["g_fd"] = np.full(M, 30.0) + np.arange(M)
    rec = awm.forecast_record(MODELS, CLAUSES, inp, {})
    E = awm.evaluate(rec, "gsm8k", acc, rev, MODELS, CLAUSES, w_observed=w)
    assert {"S1_clause_reversal", "S2_clause_vs_template", "S3_arm_switch", "S4_requested_rises",
            "S5_forced_steadier", "diagnostics"} <= set(E)
    assert isinstance(E["S1_clause_reversal"]["holds"], bool) and E["S3_arm_switch"]["clauses"] == D
    assert E["S4_requested_rises"]["holds_a"] is True            # requested rises with w by construction
    assert E["S5_forced_steadier"]["holds_a"] is True            # forced does not move with the clause
    assert np.isclose(E["diagnostics"]["w_screen_vs_d24_pearson"], 1.0)
    with pytest.raises(ValueError, match="another panel"):
        awm.evaluate(rec, "gsm8k", acc, rev, MODELS[::-1], CLAUSES)
    # no model moves enough: S4 not testable, reported so
    inp["gsm8k"]["w"] = np.full((D, M), 0.3)
    E2 = awm.evaluate(awm.forecast_record(MODELS, CLAUSES, inp, {}), "gsm8k", acc, rev, MODELS, CLAUSES)
    assert E2["S4_requested_rises"]["holds_a"] is None and E2["S4_requested_rises"]["note"]
    txt = "\n".join(awm.summary_lines({"sha256": "a" * 64, "created": "now", "tasks": {"gsm8k": E}}))
    assert "S1 clause-only reversal" in txt


def test_run_arm_wording_evaluation_and_order_check(tmp_path):
    import run_arm_wording as raw
    tab, w = _scored()
    tab2 = tab.assign(task="math")
    full = pd.concat([tab, tab2], ignore_index=True)
    paths = aw.Paths(tmp_path)
    inp = _inputs()
    for t in aw.TASKS:
        inp[t]["w"] = w
    rec = awm.write_forecast(awm.forecast_record(MODELS, CLAUSES, inp, {}), paths.mechanism, paths.mechanism_sha)
    mech = {}
    for t in aw.TASKS:
        ta = aw.TaskArrays(full, t, [f"k{c}" for c in range(6)], ["cot0", "cot1"], CLAUSES, MODELS)
        acc = ta.acc()
        mech[t] = (acc, aw.reversal_table(aw.gains(acc)))
    E = raw.mechanism_evaluation(paths, full, mech, MODELS, CLAUSES, aw.sha256_file(paths.mechanism))
    assert E["sha256"] == rec["sha256"] and set(E["holds"]) == set(aw.TASKS)
    sys.path.insert(0, str(ROOT / "paper"))
    import macros_added as ma
    mm = ma.arm_wording_mechanism_macros(E)
    assert {"awMechGsmClauseObs", "awMechMathArmThreeHolds", "awMechGsmRho", "awMechMathSteadierHoldsA"} <= set(mm)
    assert all(k.isalpha() for k in mm) and ma.arm_wording_mechanism_macros(None) == {}
    assert ma.arm_wording_mechanism_results(str(tmp_path)) is None      # no scored results yet
    assert np.allclose(raw.observed_w(full, "gsm8k", CLAUSES, MODELS),
                       full[(full.task == "gsm8k") & (full.arm == aw.REQ)].assign(l=lambda x: x.n_output_tokens > 16)
                       .groupby(["wording", "model_key"])["l"].mean().unstack().loc[CLAUSES, MODELS].to_numpy())
    # the full run refuses a table scored without (or before) this forecast
    paths.manifests.mkdir(parents=True, exist_ok=True)
    aw.write_json(paths.predictions, {"per_task": {}})
    paths.predictions_sha.write_text(aw.sha256_file(paths.predictions) + "  x\n")
    (paths.requests / "collection").mkdir(parents=True, exist_ok=True)
    aw.write_json(paths.requests / "collection" / "build_record.json",
                  {"chosen": {"cot": [{"id": "cot0"}], "dir": [{"id": c} for c in CLAUSES]}, "models": {m: {} for m in MODELS}})
    paths.processed.mkdir(parents=True, exist_ok=True)
    aw.write_json(paths.processed / "all_scored.meta.json", {"stage": "all",
                                                              "predictions_sha256": aw.sha256_file(paths.predictions),
                                                              "mechanism_sha256": None})
    args = argparse.Namespace(n_boot=0, n_splits=2)
    with pytest.raises(aw.ArmWordingError, match="mechanism forecast"):
        raw.full(args, paths, aw.load_config())


def test_builder_end_to_end_on_a_synthetic_base(tmp_path):
    import build_arm_wording_mechanism as b
    paths = aw.Paths(tmp_path)
    (paths.requests / "collection").mkdir(parents=True)
    aw.write_json(paths.requests / "collection" / "build_record.json",
                  {"chosen": {"dir": [{"id": c, "text": f"clause {c}"} for c in CLAUSES]},
                   "models": {m: {} for m in MODELS}})
    rng = np.random.default_rng(0)
    for m in MODELS:
        d = paths.outputs / "screen" / m
        d.mkdir(parents=True)
        rows = [{"model_key": m, "arm": arm, "config_id": f"{t}/cfg{k}/{wid}", "n_output_tokens": int(rng.integers(1, 40))}
                for t in aw.TASKS for k in range(2) for wid in CLAUSES + ["dir4"] for arm in ("direct_answer", "forced_direct")
                for _ in range(50)]
        pd.DataFrame(rows).to_parquet(d / "shard-x-00000.parquet")
    mix = {"decomposition": {t: {"models": MODELS[::-1], "sets": {"discovery": {"strict": {
        "per_model": {"C": list(np.linspace(60, 90, M)), "F": list(np.linspace(20, 40, M)), "R": [30.0] * M,
                      "C_short": [70.0] * M, "F_short": [25.0] * M, "w": list(np.linspace(0, 1, M))},
        "recovery_long": {"median": 0.3, "n": M}, "recovery_long_w_at_least": {"median": 0.6, "n": 5, "w_min": 0.1}}}}}
        for t in aw.TASKS}}
    mp = tmp_path / "mixture.json"
    mp.write_text(json.dumps(mix))
    rec = b.build(str(tmp_path), str(mp))
    assert awm.check_forecast(paths.mechanism, paths.mechanism_sha)
    assert rec["per_task"]["gsm8k"]["rho"] == 0.6 and rec["clauses"] == CLAUSES and rec["models"] == MODELS
    # the main collection's gains are taken in the arm-wording run's model order (the mixture lists them reversed)
    assert np.isclose(rec["per_task"]["math"]["per_model"]["m0"]["g_forced"], 90 - 40)
    shares = pd.concat(pd.read_parquet(f) for f in (paths.outputs / "screen").glob("*/shard-*.parquet"))
    x = shares[(shares.arm == "direct_answer") & (shares.config_id.str.startswith("gsm8k/"))
               & (shares.config_id.str.endswith("/dir2")) & (shares.model_key == "m3")]
    assert np.isclose(rec["per_task"]["gsm8k"]["per_model"]["m3"]["by_clause"]["dir2"]["w"],
                      (x.n_output_tokens > 16).mean())
    assert b.build(str(tmp_path), str(mp))["created"] == rec["created"]          # identical: kept

