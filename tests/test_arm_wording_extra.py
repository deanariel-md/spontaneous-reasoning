"""Default and firm clauses on the arm-wording run (src/arm_wording_extra.py, scripts/*arm_wording_extra*.py).

Synthetic data only; no arm-wording collection output and no output of this run is read."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, arm_wording_extra as ax, arm_wording_mechanism as awm  # noqa: E402

MODELS = ["m0", "m1", "m2", "m3", "m4", "m5"]


def _bank():
    return json.loads((ROOT / "manifests" / "gsm8k_bank.json").read_text(encoding="utf-8")), \
        json.loads((ROOT / "manifests" / "math_bank.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ messages

def test_default_and_firm_messages_on_every_d24_configuration():
    banks = dict(zip(aw.TASKS, _bank()))
    man = json.loads((ROOT / "manifests" / "arm_wording_configs.json").read_text(encoding="utf-8"))
    for t in aw.TASKS:
        d, v = aw.task_configs(man, t)
        for cid in d + v:
            orig = banks[t]["configurations"][cid]["arms"]["direct_answer"]["messages"]
            cur = ax.clause_messages(banks[t], cid, "dir0")
            assert cur == [{"role": m["role"], "content": m["content"]} for m in orig]
            de = ax.clause_messages(banks[t], cid, "dirdef")
            fi = ax.clause_messages(banks[t], cid, "dirfirm")
            o = orig[-1]["content"]
            assert de[-1]["content"] == o.replace(ax.CURRENT + " ", "")
            assert fi[-1]["content"] == o.replace(ax.CURRENT, ax.FIRM)
            for msgs in (de, fi):
                c = msgs[-1]["content"]
                assert ax.CURRENT not in c and "  " not in c.replace(o.split("{question}")[0], "") and "\n\n\n" not in c
                assert "\\boxed{<number>}" in c
            assert "\nEnd your reply" in de[-1]["content"]


def test_default_refused_when_clause_not_followed_by_one_space():
    bank = {"configurations": {"c": {"arms": {"direct_answer": {"messages": [
        {"role": "user", "content": "Q {question}\n" + ax.CURRENT + "\nEnd your reply."}]}}}}}
    with pytest.raises(ax.ExtraError):
        ax.clause_messages(bank, "c", "dirdef")
    bank["configurations"]["c"]["arms"]["direct_answer"]["messages"][0]["content"] = \
        "Q {question} " + ax.CURRENT + " End."
    with pytest.raises(ax.ExtraError):      # not at the start of a line
        ax.clause_messages(bank, "c", "dirdef")


def test_render_layout_and_counts():
    g, _ = _bank()
    cid = next(iter(g["configurations"]))
    items = pd.DataFrame({"item_id": ["1", "2", "3"], "question": ["a", "b", "c"]})
    df = ax.render("k", "gsm8k", g, [cid], items, prefix=ax.PILOT_PREFIX)
    assert len(df) == 1 * 2 * 2 * 3
    assert set(df["arm"]) == {aw.REQ, aw.FD}
    assert df["request_id"].iloc[0].startswith("awxp|gsm8k/")
    # requested and forced rows of a clause carry the same messages (the prefill comes from the decoding)
    for w in ax.NEW:
        s = df[df["config_id"].str.endswith(w)]
        a = s[s.arm == aw.REQ].set_index("item_id")["messages_json"]
        b = s[s.arm == aw.FD].set_index("item_id")["messages_json"]
        assert (a == b).all()


def test_built_request_files_if_present():
    rec_p = ROOT / "data" / "requests" / "arm_wording_extra" / "collection" / "build_record.json"
    if not rec_p.is_file():
        pytest.skip("request files not built here")
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    assert len(rec["models"]) == 29
    assert {v["n_requests"] for v in rec["models"].values()} == {41208}
    assert all(c.get("d24_requested_dir0_identical") for c in rec["checks"])
    pil = json.loads((rec_p.parent.parent / "pilot" / "build_record.json").read_text(encoding="utf-8"))
    assert {v["n_requests"] for v in pil["models"].values()} == {800}


# ------------------------------------------------------------------ forecast

def _mech(rng, M=len(MODELS)):
    w4 = rng.uniform(0, 1, (4, M))
    w4[:, 0] = 0.0
    g = rng.uniform(5, 40, M)
    inputs = {t: {"w": w4, "g_fd": g, "g_short": g * 0.9, "rho": 0.5, "w_collection_current": w4[0] + 0.01}
              for t in aw.TASKS}
    return awm.forecast_record(MODELS, ax.D24_CLAUSES, inputs, {"test": True}), w4, g


def test_forecast_arithmetic_and_near_zero_sets():
    rng = np.random.default_rng(0)
    mech, w4, g = _mech(rng)
    wn = np.vstack([np.full(len(MODELS), 0.6), np.array([0.0, 0.02, 0.3, 0.05, 0.5, 0.06])])
    inp = ax.forecast_inputs(mech, MODELS, {t: wn for t in aw.TASKS})
    r = ax.forecast_record(MODELS, inp, {"test": True})
    T = r["per_task"]["gsm8k"]
    for j, m in enumerate(MODELS):
        bc = T["per_model"][m]["by_clause"]
        assert bc["dirfirm"]["predicted_gain_vs_requested_pooled"] == pytest.approx((1 - wn[1, j]) * g[j])
        assert bc["dirdef"]["predicted_gain_vs_requested_partial"] == pytest.approx((1 - 0.5 * wn[0, j]) * g[j])
        assert bc["dirfirm"]["predicted_requested_minus_forced_pooled"] == pytest.approx(wn[1, j] * g[j])
        assert bc["dir1"]["w"] == pytest.approx(w4[1, j])
    assert T["near0"]["dirfirm"] == ["m0", "m1", "m3"]          # w <= 0.05, the boundary included
    assert T["near0"]["dirdef"] == []
    assert r["clauses"] == list(ax.CLAUSES6)
    assert len(T["predicted"]["arm_switch_by_clause_range"]) == 6
    assert r["sha256"] == awm.sha256_json({k: v for k, v in r.items() if k not in awm.UNHASHED})
    with pytest.raises(ax.ExtraError):
        ax.forecast_inputs(mech, MODELS[::-1], {t: wn for t in aw.TASKS})
    with pytest.raises(ax.ExtraError):
        ax.forecast_inputs(mech, MODELS, {t: wn * np.nan for t in aw.TASKS})


def test_forecast_write_once_and_hash_refusal(tmp_path):
    rng = np.random.default_rng(1)
    mech, _, _ = _mech(rng)
    wn = rng.uniform(0, 1, (2, len(MODELS)))
    paths = ax.Paths(tmp_path)
    with pytest.raises(ax.ExtraError):
        ax.check_forecast(paths)
    r = ax.forecast_record(MODELS, ax.forecast_inputs(mech, MODELS, {t: wn for t in aw.TASKS}), {"a": 1})
    ax.write_forecast(r, paths)
    sha = ax.check_forecast(paths)
    ax.write_forecast(r, paths)                         # identical: kept
    assert ax.check_forecast(paths) == sha
    r2 = ax.forecast_record(MODELS, ax.forecast_inputs(mech, MODELS, {t: wn * 0.5 for t in aw.TASKS}), {"a": 1})
    with pytest.raises(FileExistsError):
        ax.write_forecast(r2, paths)
    txt = paths.forecast.read_text(encoding="utf-8")
    paths.forecast.write_text(txt.replace('"w_near0": 0.05', '"w_near0": 0.5'), encoding="utf-8")
    with pytest.raises(ax.ExtraError):
        ax.check_forecast(paths)


def test_scoring_refuses_without_forecast(tmp_path):
    import score_arm_wording_extra as sax
    p = ax.Paths(tmp_path)
    (p.requests / "collection").mkdir(parents=True)
    aw.write_json(p.requests / "collection" / "build_record.json",
                  {"models": {}, "configurations": {t: {"discovery": [], "validation": []} for t in aw.TASKS}})
    for stage in ("discovery", "all"):
        assert sax.main(["--stage", stage, "--base", str(tmp_path)]) == 2


# ------------------------------------------------------------------ evaluation

def _acc(rng, w6, near_eq):
    C, M = 6, len(MODELS)
    ability = rng.uniform(20, 70, M)
    cot = ability[None, None, :] + 25 + rng.normal(0, 0.5, (C, 4, M))
    F = ability[None, None, :] + rng.normal(0, 0.5, (C, 6, M))
    R = F + 20 * w6[None] + rng.normal(0, 0.3, (C, 6, M))
    for j in near_eq:
        R[:, 5, j] = F[:, 5, j] + 0.5
    return {aw.COT: cot, aw.REQ: R, aw.FD: F}


def test_evaluate_statements_on_synthetic_tables():
    rng = np.random.default_rng(2)
    mech, w4, g = _mech(rng)
    wn = np.vstack([np.clip(w4[0] + 0.3, 0, 1), np.array([0.0, 0.01, 0.2, 0.04, 0.5, 0.3])])
    rec = ax.forecast_record(MODELS, ax.forecast_inputs(mech, MODELS, {t: wn for t in aw.TASKS}), {})
    w6 = np.vstack([w4, wn])
    acc = _acc(rng, w6, near_eq=[0, 1, 3])
    rev = aw.reversal_table(aw.gains(acc))
    x = ax.evaluate(rec, "gsm8k", acc, rev, MODELS, w6)
    a1 = x["A1_forced_meets_requested"]
    assert a1["models"] == ["m0", "m1", "m3"] and a1["holds_a"] and a1["holds_b"]
    assert x["A2_default_more_working"]["holds_a"] and x["A2_default_more_working"]["holds_b"]
    assert x["A4_pilot_predicts_w"]["holds"]
    a5 = x["A5_extrapolates_to_forced"]
    assert a5["pooled_slope_pp_per_unit_w"] > 0 and a5["holds"]
    assert len(x["mechanism_six_clauses"]["S3_arm_switch"]["by_clause"]) == 6
    # a miss is reported as a miss: a large requested - forced gap at w near 0
    acc2 = _acc(rng, w6, near_eq=[])
    for j in (0, 1, 3):
        acc2[aw.REQ][:, 5, j] = acc2[aw.FD][:, 5, j] + 10
    x2 = ax.evaluate(rec, "gsm8k", acc2, aw.reversal_table(aw.gains(acc2)), MODELS, w6)
    assert x2["A1_forced_meets_requested"]["holds_a"] is False
    # empty near-zero set: not testable, reported as None
    rec3 = json.loads(json.dumps(rec))
    rec3["per_task"]["gsm8k"]["near0"]["dirfirm"] = []
    x3 = ax.evaluate(rec3, "gsm8k", acc, rev, MODELS, w6)
    assert x3["A1_forced_meets_requested"]["holds_a"] is None and x3["A1_forced_meets_requested"]["note"]


# ------------------------------------------------------------------ the order end to end on synthetic tables

def _scored(models, confs, wids, arms, n_items, rng, task):
    rows = []
    for m in models:
        for c in confs:
            for a in arms:
                for w in wids[a]:
                    for i in range(n_items):
                        p = 0.3 + 0.1 * models.index(m) / len(models) + (0.25 if a == aw.COT else 0)
                        ok = bool(rng.uniform() < p)
                        rows.append({"model_key": m, "arm": a, "config_id": f"{task}/{c}/{w}", "item_id": str(i),
                                     "strict_correct": ok, "lenient_correct": ok, "compliant": True,
                                     "box_closed": True, "finish_reason": "stop",
                                     "n_output_tokens": int(rng.integers(3, 40)), "prompt_sha256": "x"})
    return pd.DataFrame(rows)


def test_run_order_end_to_end(tmp_path):
    import run_arm_wording_extra as rax
    rng = np.random.default_rng(3)
    p = ax.Paths(tmp_path)
    confs = {t: {"discovery": ["a", "b", "c"], "validation": ["d", "e", "f"]} for t in aw.TASKS}
    cot = ["cot0", "cot1"]
    (p.d24_requests / "collection").mkdir(parents=True)
    aw.write_json(p.d24_requests / "collection" / "build_record.json",
                  {"models": {m: {} for m in MODELS}, "configurations": confs,
                   "chosen": {"cot": [{"id": c} for c in cot], "dir": [{"id": d} for d in ax.D24_CLAUSES]}})
    d24 = pd.concat([_scored(MODELS, confs[t]["discovery"] + confs[t]["validation"],
                             {aw.COT: cot, aw.REQ: list(ax.D24_CLAUSES), aw.FD: list(ax.D24_CLAUSES)},
                             (aw.COT, aw.REQ, aw.FD), 8, rng, t) for t in aw.TASKS], ignore_index=True)
    p.d24_processed.mkdir(parents=True)
    own = pd.concat([_scored(MODELS, confs[t]["discovery"] + confs[t]["validation"],
                             {aw.REQ: list(ax.NEW), aw.FD: list(ax.NEW)}, (aw.REQ, aw.FD), 8, rng, t)
                     for t in aw.TASKS], ignore_index=True)
    p.processed.mkdir(parents=True)
    # no forecast: refused
    assert rax.main(["--stage", "discovery", "--base", str(tmp_path)]) == 2
    mech, _, _ = _mech(rng)
    fc = ax.write_forecast(ax.forecast_record(MODELS, ax.forecast_inputs(
        mech, MODELS, {t: rng.uniform(0, 1, (2, len(MODELS))) for t in aw.TASKS}), {}), p)
    fsha = ax.check_forecast(p)
    is_disc = own["config_id"].map(lambda c: aw.parse_config_id(c)[1] in "abc")
    own[is_disc].to_parquet(p.processed / "discovery_scored.parquet", index=False)
    aw.write_json(p.processed / "discovery_scored.meta.json", {"stage": "discovery", "forecast_sha256": fsha})
    # the arm-wording run is not scored yet: refused
    assert rax.main(["--stage", "discovery", "--base", str(tmp_path)]) == 2
    d24.to_parquet(p.d24_processed / "all_scored.parquet", index=False)
    aw.write_json(p.d24_processed / "all_scored.meta.json",
                  {"stage": "all", "predictions_sha256": "p", "mechanism_sha256": "m"})
    assert rax.main(["--stage", "discovery", "--base", str(tmp_path)]) == 0
    psha = ax.check_discovery_predictions(p)
    own.to_parquet(p.processed / "all_scored.parquet", index=False)
    # scored before the discovery predictions: refused
    aw.write_json(p.processed / "all_scored.meta.json", {"stage": "all", "forecast_sha256": fsha,
                                                         "predictions_sha256": None})
    assert rax.main(["--base", str(tmp_path), "--n-boot", "0"]) == 2
    aw.write_json(p.processed / "all_scored.meta.json", {"stage": "all", "forecast_sha256": fsha,
                                                         "predictions_sha256": psha})
    assert rax.main(["--base", str(tmp_path), "--n-boot", "0"]) == 0
    out = json.loads((p.results / "arm_wording_extra.json").read_text(encoding="utf-8"))
    assert len(out["per_task"]["gsm8k"]["strict"]["reversals"]["arm_switch_by_clause"]) == 6
    E = json.loads((p.results / "forecast_evaluation.json").read_text(encoding="utf-8"))
    assert set(E["holds"]["math"]) >= {"A1a", "A2a", "A3a", "A4", "A5", "S1", "S3_count"}
    assert (p.results / "summary_clean.md").is_file()
    sys.path.insert(0, str(ROOT / "paper"))
    import macros_awx
    m = macros_awx.awx_macros(macros_awx.awx_results(str(tmp_path)))
    assert m["awxModels"] == "6" and "awxGsmFirmGap" in m and "awxMathAOneA" in m
    assert all(k.isalpha() for k in m)
    assert macros_awx.awx_results(str(tmp_path / "nothing")) is None
