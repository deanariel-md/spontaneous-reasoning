"""Dataset 2 analysis module (src/dataset2.py), sampler and pipeline order."""
import importlib.util
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import dataset1 as d1, dataset2 as d2, intervals as iv, sampling
from src.estimators import pair_agreement, paired_loss_difference, unit_moments
from src.simulate import GSM8KSim

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEL = {"gap": "jack_t", "C_benefit": "jack_t", "C_accuracy": "jack_t", "H": "fieller", "V_delta": "mls",
       "V_delta_upper": "boot_naive", "A": "jack_t", "V_theta": None,
       **{f"risk:{f}-{g}": "jack_t" for f, g in d1.COMPARISONS}}


def _scored(panel, configs, items, rng, arms=d2.ARMS):
    rows = []
    for mk, arm, c, it in itertools.product(panel, arms, configs, items):
        s = int(rng.random() < 0.6)
        rows.append({"model_key": mk, "arm": arm, "config_id": c, "item_id": it, "strict_correct": s,
                     "lenient_correct": max(s, int(rng.random() < 0.1)), "compliant": int(rng.random() < 0.9)})
    return pd.DataFrame(rows)


def _units(K, M, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    if ties:
        B = rng.integers(40, 44, size=(K, M)).astype(float)
        P = B + rng.integers(0, 3, size=(K, M))
    else:
        th = np.linspace(20, 80, M)
        B = th + rng.normal(0, 3, (K, M))
        P = B + 5 - 0.1 * (th - th.mean()) + rng.normal(0, 3, (K, M))
    return B, P


# ------------------------------------------------------------------ data adapter

def test_cell_table_and_unit_arrays():
    rng = np.random.default_rng(1)
    panel, configs, items = ["m1", "m2"], ["c1", "c2", "c3"], [0, 1, 2, 3]
    df = _scored(panel, configs, items, rng)
    cells = d2.cell_table(df, panel, configs, items)
    units = ["c2", "c1", "c2"]
    B, P = d2.unit_arrays(cells, units, panel)
    for r, c in enumerate(units):
        for i, mk in enumerate(panel):
            x = df[(df.model_key == mk) & (df.config_id == c)]
            assert B[r, i] == 100 * x[x.arm == d2.ARMS[0]].strict_correct.mean()
            assert P[r, i] == 100 * x[x.arm == d2.ARMS[1]].strict_correct.mean()
    nB, _ = d2.unit_arrays(cells, units, panel, counts=True)
    assert np.allclose(nB, B * len(items) / 100)
    assert np.array_equal(B[0], B[2])


def test_cell_table_refuses_incomplete_data():
    rng = np.random.default_rng(2)
    panel, configs, items = ["m1", "m2"], ["c1", "c2"], [0, 1, 2]
    df = _scored(panel, configs, items, rng)
    with pytest.raises(d2.IncompleteData, match="lack the full item set"):
        d2.cell_table(df.iloc[1:], panel, configs, items)
    with pytest.raises(d2.IncompleteData, match="lack the full item set"):
        d2.cell_table(df[df.model_key == "m1"], panel, configs, items)
    with pytest.raises(d2.IncompleteData, match="outside the fixed item set"):
        d2.cell_table(df, panel, configs, [0, 1])
    with pytest.raises(d2.IncompleteData, match="duplicated"):
        d2.cell_table(pd.concat([df, df.iloc[:1]]), panel, configs, items)


# ------------------------------------------------------------------ Steps 1 and 2

def test_step1_matches_appendix_a_and_jackknife():
    B, P = _units(12, 8, ties=True)
    m = iv.multiplicities(np.random.default_rng(0), 12, 199)
    s1 = d2.step1(B, P, SEL, m, [f"m{i}" for i in range(8)])
    ref_b, ref_a = pair_agreement(P - B), pair_agreement(B)
    for key in ("same", "opposite", "tie", "C"):
        assert s1["C_benefit"][key] == pytest.approx(ref_b[key])
        assert s1["C_accuracy"][key] == pytest.approx(ref_a[key])
    assert s1["C_benefit"]["tie"] > 0
    loo = iv.Agreement(B).loo() - iv.Agreement(P - B).loo()
    assert s1["gap"]["ci95"] == pytest.approx(list(iv.jack_t(s1["gap"]["estimate"], loo)))
    t = s1["gap"]["test_zero"]
    lo, hi = s1["gap"]["ci95"]
    assert (t["p_two_sided"] < 0.05) == (lo > 0 or hi < 0)


def test_step2_matches_moments_and_selected_methods():
    B, P = _units(12, 10)
    G = P - B
    m = iv.multiplicities(np.random.default_rng(3), 12, 499)
    s2 = d2.step2(B, P, SEL, m)
    q = unit_moments(B, P)
    for k in ("V_theta", "A", "V_delta"):
        assert s2[k]["estimate"] == pytest.approx(q[k])
    assert s2["V_delta"]["ci95"] == pytest.approx(list(iv.mls_vdelta(G)))
    assert s2["V_delta"]["upper95"] == pytest.approx(iv.percentile_upper(iv.CrossMoment(G, G).boot(m, distinct=False)))
    assert s2["V_theta"]["ci95"] is None
    cA, cV = iv.CrossMoment(G, B), iv.CrossMoment(B, B)
    kind, a, b = iv.scale_fieller(*iv.fieller(q["A"], q["V_theta"], cA.loo(), cV.loo()), -10.0)
    assert s2["H"]["set"] == {"kind": kind, "a": pytest.approx(a), "b": pytest.approx(b)}
    assert s2["H"]["estimate"] == pytest.approx(-10 * q["A"] / q["V_theta"])


def test_negative_variance_estimate_is_reported_not_clipped():
    rng = np.random.default_rng(5)
    B = 50 + rng.normal(0, 2, (8, 6))
    P = B + 5 + rng.normal(0, 2, (8, 6))           # equal true gains: V_delta estimate near zero
    found = False
    for seed in range(40):
        rng = np.random.default_rng(seed)
        P = B + 5 + rng.normal(0, 2, (8, 6))
        s2 = d2.step2(B, P, SEL, iv.multiplicities(rng, 8, 99))
        if s2["V_delta"]["estimate"] < 0:
            found = True
            assert s2["V_delta"]["negative_estimate"] and s2["V_delta"]["gain_sd"] is None
            break
    assert found


def test_classifications():
    c = d2.classify_interval
    assert c(1.2, 3.0, 1) == "outside margin (positive)"
    assert c(-3.0, -1.5, 1) == "outside margin (negative)"
    assert c(-0.5, 0.9, 1) == "within margin"
    assert c(-0.5, 1.5, 1) == "unresolved"
    assert c(float("nan"), 1.0, 1) == "undefined"
    assert d2.classify_set("exclusive", -2.0, 3.0, 1) == "outside margin (sign unresolved)"
    assert d2.classify_set("exclusive", -0.5, 3.0, 1) == "unresolved"
    assert d2.classify_set("whole", -np.inf, np.inf, 1) == "unresolved"
    assert d2.classify_set("undefined", np.nan, np.nan, 1) == "undefined"
    v = d2.classify_vdelta
    assert v(-1.0, 3.9) == "within margin"
    assert v(4.5, 30.0) == "outside margin (positive)"
    assert v(-1.0, 6.0) == "unresolved"
    assert v(None, None) == "undefined"


def test_per_model_draw_agreement_matches_enumeration():
    B, P = _units(6, 4, seed=7, ties=True)
    G = P - B
    K, M = G.shape
    same = opp = tie = n = 0
    for i, j in itertools.combinations(range(M), 2):
        for a, b, c, d in itertools.permutations(range(K), 4):
            s1, s2 = np.sign(G[a, i] - G[b, j]), np.sign(G[c, i] - G[d, j])
            n += 1
            if s1 == 0 or s2 == 0:
                tie += 1
            elif s1 == s2:
                same += 1
            else:
                opp += 1
    r = d2.per_model_draw_agreement(G)
    assert r["same"] == pytest.approx(same / n)
    assert r["opposite"] == pytest.approx(opp / n)
    assert r["tie"] == pytest.approx(tie / n)


# ------------------------------------------------------------------ Step 3

def test_freeze_is_idempotent_and_refuses_changes(tmp_path):
    B, P = _units(8, 6)
    path = str(tmp_path / "pred.json")
    f1 = d2.freeze_predictions(B, P, "t", [f"m{i}" for i in range(6)], "abc")
    w1 = d2.write_frozen(f1, path)
    f2 = d2.freeze_predictions(B, P, "t", [f"m{i}" for i in range(6)], "abc")
    assert d2.write_frozen(f2, path)["created"] == w1["created"]
    f3 = d2.freeze_predictions(B, P + 1e-3, "t", [f"m{i}" for i in range(6)], "abc")
    with pytest.raises(FileExistsError):
        d2.write_frozen(f3, path)
    assert d2.write_frozen(f3, path, overwrite=True)["sha256"] == f3["sha256"]
    assert sorted(map(int, f1["budgets"])) == [1, 2, 4, 8]


def test_evaluate_frozen_matches_paired_losses():
    B, P = _units(8, 6, seed=11)
    Bv, Pv = _units(10, 6, seed=12)
    Zv = Pv - Bv
    fz = json.loads(json.dumps(d2.freeze_predictions(B, P, "t", list("abcdef"), "x"), default=d2._jsonable))
    ev = d2.evaluate_frozen(fz, Zv, SEL)
    assert len(ev["registered"]) == len(d1.COMPARISONS)
    preds = {n: np.array(v["prediction"]) for n, v in fz["budgets"]["8"].items()}
    for row in ev["registered"]:
        L = paired_loss_difference(Zv, preds[row["f"]], preds[row["g"]])
        assert row["estimate"] == pytest.approx(L.mean())
        assert row["ci95"] == pytest.approx(list(iv.jack_t(L.mean(), (L.sum() - L) / 9)))
    with pytest.raises(NotImplementedError):
        d2.evaluate_frozen(fz, Zv, {**SEL, "risk:constant-own_gain": "boot"})
    # plan C.4: no square root of the absolute risk without a calibrated nonnegative interval
    for r in ev["absolute_risk"]:
        assert r["sqrt"] is None and "plan C.4" in r["sqrt_note"]
        assert r["naive_mse_includes_validation_noise"] == pytest.approx(np.mean((Zv - preds[r["predictor"]]) ** 2))


def test_capability_claims_reports_zero_slopes_as_ties():
    B, P = _units(4, 5, seed=3)
    Bv, Pv = _units(4, 5, seed=4)
    P[0] = B[0] + 2.0                                   # constant gain in one unit: slope 0, a tie
    P[1] = B[1] + 2.0
    Bv[0] = 50.0                                        # no capability variance: slope undefined
    out = d2.capability_claims(B, P, Bv, Pv, SEL)["naive_single_unit"]
    H = np.array(out["H_values"])
    assert out["n_zero_slope"] == 2 and out["n_undefined"] == 1 and out["n_undefined_or_zero"] == 3
    fin = H[np.isfinite(H)]
    pairs = list(itertools.combinations(fin, 2))
    pp = out["pairs_with_defined_slopes"]
    assert pp["n_pairs"] == len(pairs) == 21
    assert pp["at_least_one_tie"] == pytest.approx(sum(a == 0 or b == 0 for a, b in pairs) / 21)
    assert pp["same_sign"] == pytest.approx(sum(a * b > 0 for a, b in pairs) / 21)
    assert pp["same_sign"] + pp["opposite_sign"] + pp["at_least_one_tie"] == pytest.approx(1.0)


# ------------------------------------------------------------------ sampler

def test_registered_split_bank():
    ids = [f"c{i}" for i in range(50)]
    p = np.full(50, 1 / 50)
    a = sampling.registered_split_bank(ids, p, 12, 12, "x")
    assert a == sampling.registered_split_bank(ids, p, 12, 12, "x")
    assert a != sampling.registered_split_bank(ids, p, 12, 12, "y")
    assert len(a["discovery_config_ids"]) == 12 and set(a["validation_config_ids"]) <= set(ids)
    one = np.zeros(50)
    one[7] = 1
    b = sampling.registered_split_bank(ids, one, 4, 4, "x")
    assert set(b["discovery_config_ids"] + b["validation_config_ids"]) == {"c7"}
    with pytest.raises(ValueError):
        sampling.registered_split_bank(ids, p * 2, 4, 4, "x")
    with pytest.raises(ValueError):
        sampling.registered_split_bank(ids + ["c0"], np.full(51, 1 / 51), 4, 4, "x")


# ------------------------------------------------------------------ simulator and pipeline order

def test_gsm8k_sim_is_deterministic_and_matches_bank_counts():
    sim = GSM8KSim(np.random.default_rng(0), M=6, C=20, n_items=30)
    a, b = sim.scored(["cfg00003", "cfg00011"]), sim.scored(["cfg00011", "cfg00003"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    n = sim.bank_counts()
    x = a[(a.config_id == "cfg00011") & (a.arm == "zero_shot_cot")]
    got = x.groupby("model_key").strict_correct.sum().reindex(sim.panel).to_numpy()
    assert np.array_equal(got, n[0, 1, 11])


def _runner():
    spec = importlib.util.spec_from_file_location("run_dataset2", os.path.join(ROOT, "scripts", "run_dataset2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_freezes_before_reading_validation(tmp_path):
    rd2 = _runner()
    sim = GSM8KSim(np.random.default_rng(1), M=8, C=40, n_items=40)
    sampler = sampling.registered_split_bank(sim.config_ids, np.full(sim.C, 1 / sim.C), 8, 8, "t")
    pred = str(tmp_path / "pred.json")
    calls = []

    def load(ids):
        calls.append((sorted(set(ids)), os.path.exists(pred)))
        return sim.scored(ids)

    out = rd2.run(load, sampler, sim.panel, sim.families, sim.tiers, list(range(sim.N)), str(tmp_path / "res"),
                  pred, "t", n_boot=199, n_ctrl=20)
    assert calls[0] == (sorted(set(sampler["discovery_config_ids"])), False)
    assert all(exists for _, exists in calls[1:])
    assert set(calls[1][0]) == set(sampler["discovery_config_ids"]) | set(sampler["validation_config_ids"])
    saved = json.load(open(pred))
    assert out["step3"]["frozen_sha256"] == saved["sha256"]
    assert os.path.exists(str(tmp_path / "pred_lenient.json"))
    assert out["step1"]["gap"]["method"] == "jack_t" and out["step2"]["H"]["method"] == "fieller"
    for f in ("steps12_registered.csv", "step3_risk_differences.csv", "controls.csv", "sensitivities.csv"):
        assert os.path.exists(tmp_path / "res" / "tables" / f)


def test_freeze_sampler_never_overwrites(tmp_path):
    spec = importlib.util.spec_from_file_location("freeze_sampler_d2", os.path.join(ROOT, "scripts", "freeze_sampler_d2.py"))
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    bp, out, adm = tmp_path / "bank.json", str(tmp_path / "sampler.json"), tmp_path / "admissibility.json"
    bank = _valid_bank(30)
    bp.write_text(json.dumps(bank))
    adm.write_text(json.dumps(_admissibility(list(bank["configurations"]))))
    s = fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    assert len(s["discovery_config_ids"]) == 12 and s["bank_file_sha256"]
    assert s["admissibility"]["sha256"] and json.load(open(out))["sha256"] == s["sha256"]
    with pytest.raises(SystemExit, match="never overwritten"):
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    with pytest.raises(SystemExit, match="allocation"):
        fs.freeze(str(bp), str(tmp_path / "other.json"), 10, 12, admissibility_path=str(adm))


def _valid_bank(n):
    arm = {"messages": [{"role": "user", "content": "Q: {question}"}]}
    return {"generator": "test", "generator_version": "0", "seed": 0,
            "sampling_probabilities": {f"c{i}": 1 / n for i in range(n)},
            "configurations": {f"c{i}": {"dimensions": {}, "arms": {"direct_answer": arm, "zero_shot_cot": arm}}
                               for i in range(n)}}


def _admissibility(retained, excluded=(), models=("m1",)):
    return {"models_screened": list(models), "excluded_models": [], "retained_models": list(models),
            "configuration_rules_evaluated": True, "configurations_screened": [*retained, *excluded],
            "excluded_configurations": [{"config_id": c, "violations": []} for c in excluded],
            "retained_configuration_ids": list(retained)}


def test_freeze_sampler_refuses_a_bank_that_is_not_the_retained_bank(tmp_path):
    spec = importlib.util.spec_from_file_location("freeze_sampler_d2", os.path.join(ROOT, "scripts", "freeze_sampler_d2.py"))
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    bp, adm = tmp_path / "bank.json", tmp_path / "admissibility.json"
    bank = _valid_bank(30)
    bp.write_text(json.dumps(bank))
    ids = list(bank["configurations"])
    out = str(tmp_path / "s.json")
    with pytest.raises(SystemExit, match="does not exist"):                       # no exclusion log
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    adm.write_text(json.dumps(_admissibility(ids[:-1], excluded=ids[-1:])))
    with pytest.raises(SystemExit, match="excluded by the admissibility rules"):
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    adm.write_text(json.dumps(_admissibility(ids[:-1])))
    with pytest.raises(SystemExit, match="never screened"):
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    adm.write_text(json.dumps(_admissibility(ids + ["c99"])))
    with pytest.raises(SystemExit, match="missing from the bank"):
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    adm.write_text(json.dumps(_admissibility(ids)))
    bp.write_text(json.dumps({**bank, "generator": ""}))
    with pytest.raises(SystemExit, match="generator"):                            # validate_bank runs
        fs.freeze(str(bp), out, 12, 12, admissibility_path=str(adm))
    assert not os.path.exists(out)


def test_run_refuses_a_panel_without_families_before_freezing(tmp_path):
    spec = importlib.util.spec_from_file_location("run_dataset2", os.path.join(ROOT, "scripts", "run_dataset2.py"))
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    sampler = {"discovery_config_ids": ["a"], "validation_config_ids": ["b"], "sha256": "x"}
    pred = tmp_path / "pred.json"
    for fam in ([None, "f"], ["PLACEHOLDER", "f"], [" ", "f"]):
        with pytest.raises(ValueError, match="family"):
            rd.run(lambda ids: None, sampler, ["m1", "m2"], fam, None, [0], str(tmp_path), str(pred), "t", 9, 1)
    assert not pred.exists()


def test_registered_run_refuses_non_main_tables(tmp_path):
    import subprocess
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / "scored.parquet"
    t = pa.table({"model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": [0],
                  "strict_correct": [1], "lenient_correct": [1], "compliant": [1]})
    pq.write_table(t.replace_schema_metadata({"d2_kind": "screening", "d2_split": "train",
                                             "d2_scorer_status": "registered"}), p)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_dataset2.py"), "--scored", str(p)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in r.stderr and "d2_kind" in r.stderr and "d2_split" in r.stderr


def test_registered_run_refuses_echo_and_unmatched_tables_labelled_main(tmp_path):
    import subprocess
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / "scored.parquet"
    t = pa.table({"model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": [0],
                  "strict_correct": [1], "lenient_correct": [1], "compliant": [1]})
    inputs = [{"model_key": "m", "test_backend": True, "request_kind": None, "split": None,
               "incomplete_allowed": True, "decoding_confirmed": False}]
    pq.write_table(t.replace_schema_metadata({"d2_kind": "main", "d2_split": "test",
                                             "d2_scorer_status": "registered", "d2_inputs": json.dumps(inputs)}), p)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_dataset2.py"), "--scored", str(p)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in r.stderr
    for what in ("test backend", "request kind None", "--allow-incomplete", "unconfirmed decoding"):
        assert what in r.stderr
