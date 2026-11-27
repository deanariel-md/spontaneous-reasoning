"""Tests for the exemplar arm's scoring and analysis glue (scripts/score_exemplar.py, scripts/run_exemplar.py) on fabricated rows."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_ex_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rex = load_script("run_exemplar")
sex = load_script("score_exemplar")


def rows(model, arm, configs, items, strict, finish="stop"):
    out = []
    for c in configs:
        for i in items:
            out.append({"model_key": model, "arm": arm, "config_id": c, "item_id": i, "strict_correct": strict,
                        "lenient_correct": strict, "compliant": 1, "finish_reason": finish, "n_output_tokens": 100,
                        "cached": False, "prompt_sha256": "x"})
    return pd.DataFrame(out)


def test_zero_shot_cut_replaces_only_the_cut_rows():
    col = pd.concat([rows("m4k", "zero_shot_cot", ["c1", "c2"], [1, 2], 1), rows("m4k", "direct_answer", ["c1"], [1], 1),
                     rows("other", "zero_shot_cot", ["c1"], [1], 1)], ignore_index=True)
    cut = pd.DataFrame({"model_key": ["m4k"], "config_id": ["c2"], "item_id": [2], "strict_correct": [0],
                        "lenient_correct": [0], "compliant": [0], "finish_reason": ["length"]})
    z, n = rex.zero_shot(col, cut)
    assert n == 1 and len(z) == 5 and set(z["arm"]) == {"zero_shot_cot"}
    hit = (z["model_key"] == "m4k") & (z["config_id"] == "c2") & (z["item_id"] == 2)
    assert z.loc[hit, "strict_correct"].tolist() == [0] and z.loc[hit, "finish_reason"].tolist() == ["length"]
    assert z.loc[~hit, "strict_correct"].eq(1).all() and z["strict_correct"].dtype == col["strict_correct"].dtype
    z0, n0 = rex.zero_shot(col, None)
    assert n0 == 0 and z0["strict_correct"].eq(1).all()


def test_combined_loader_names_the_arms_for_the_analysis():
    col = pd.concat([rows("m", "zero_shot_cot", ["c1"], [1, 2], 0), rows("m", "direct_answer", ["c1"], [1, 2], 1)])
    ex = rows("m", "eight_shot_cot", ["c1"], [1, 2], 1)
    df = rex.combined_loader(lambda ids: col, lambda ids: ex, None)(["c1"])
    assert df.groupby("arm")["strict_correct"].mean().to_dict() == {"direct_answer": 0, "zero_shot_cot": 1}
    long = rex.strict_long(lambda ids: df, ["c1"], None)
    assert sorted(long["arm"].unique()) == ["eight_shot_cot", "zero_shot_cot"]
    assert long.groupby("arm")["correct"].mean().to_dict() == {"eight_shot_cot": 1, "zero_shot_cot": 0}


def _r_min():
    sset = {"kind": "bounded", "a": -0.2, "b": -0.1}
    disc = {"slope": -0.15, "slope_set": sset}
    return {"M": 2, "K_discovery": 24, "K_validation": 24, "means": {"zero_shot": 50.0, "eight_shot": 55.0, "benefit": 5.0},
            "step2": {"V_delta": {"estimate": 3.0, "ci95": [1.0, 9.0]}, "H": {"estimate": 1.5, "set": {"kind": "bounded", "a": 1, "b": 2}},
                      "noise": {"W_gain": 0.5}, "reliability": {"R_benefit": 0.85, "R_accuracy": 0.97}},
            "step3_registered": [], "scale": {sc: {"discovery": disc} for sc in ("points", "logit", "error_reduction")},
            "map": {"point": {q: {"R": 0.9, "C": 0.9, "C_ref": 0.9} for q in ("benefit", "accuracy")}}}


R_MIN = _r_min()


def test_macros_are_letters_only():
    sset = {"kind": "bounded", "a": -0.2, "b": -0.1}
    disc = {"slope": -0.15, "slope_set": sset}
    R = {"M": 2, "K_discovery": 24, "K_validation": 24, "means": {"zero_shot": 50.0, "eight_shot": 55.0, "benefit": 5.0},
         "step2": {"V_delta": {"estimate": 3.0, "ci95": [1.0, 9.0]}, "H": {"estimate": 1.5, "set": {"kind": "bounded", "a": 1, "b": 2}},
                   "noise": {"W_gain": 0.5}, "reliability": {"R_benefit": 0.85, "R_accuracy": 0.97}},
         "step3_registered": [{"f": "constant", "g": "own_gain", "estimate": 2.5, "ci95": [1.0, 4.0]},
                              {"f": "own_gain", "g": "single_prompt", "estimate": -0.4, "provisional_ci": [-1.0, 0.2]}],
         "scale": {sc: {"discovery": disc} for sc in ("points", "logit", "error_reduction")},
         "map": {"point": {q: {"R": 0.9, "C": 0.9, "C_ref": 0.9} for q in ("benefit", "accuracy")}}}
    m = rex.macros(R)
    assert all(k.isalpha() and k.startswith("exh") for k in m) and m["exhSlopePoints"] == "$-$0.150"
    assert m["exhRDConstOwn"] == "2.50" and m["exhRDOwnSingleLo"] == "$-$1.00" and m["exhWgain"] == "0.50"
    assert m["exhMapKEightyBen"] == "1"


def test_noise_block_splits_w_gain_per_arm_and_checks_step2():
    rng = np.random.default_rng(1)
    Bd = 60 + rng.normal(0, 3, (24, 1))[:, [0] * 6] + rng.normal(0, 1, (24, 6)) + np.arange(6)[None, :]
    Pd = Bd + 2 + rng.normal(0, 0.5, (24, 6))
    sp = rex.na.arm_split(Bd, Pd, "draws")
    x = rex.noise_block(Bd, Pd, {"W_gain": sp["W_gain"], "W_baseline": sp["var_e_baseline"]})
    assert all(x["check_against_step2"].values())
    assert x["split"]["W_gain"] == pytest.approx(x["split"]["var_e_intervention"] + x["split"]["var_e_baseline"]
                                                 - 2 * x["split"]["cov_e"])
    with pytest.raises(SystemExit, match="does not reproduce Step 2"):
        rex.noise_block(Bd, Pd, {"W_gain": sp["W_gain"], "W_baseline": sp["var_e_baseline"] + 0.01})
    m = rex.macros({**R_MIN, "noise_split": x})
    assert m["exhNoiseVarB"] == rex.fmt(x["split"]["var_e_baseline"], 2).replace("−", "$-$")
    assert m["exhNoiseVarP"] == rex.fmt(x["split"]["var_e_intervention"], 2).replace("−", "$-$")
    assert "exhNoiseVarB" not in rex.macros(R_MIN)


def test_tercile_block_reading():
    rng = np.random.default_rng(0)
    panel = [f"m{i}" for i in range(29)]
    rank = np.arange(29, dtype=float)                       # m0 weakest, m28 strongest
    gain = np.where(rank < 10, 6.0, np.where(rank >= 19, 0.2, 2.0))
    Bv = 50 + rng.normal(0, 1, (24, 29))
    Pv = Bv + gain[None, :] + rng.normal(0, 1, (24, 29))
    x = rex.tercile_block(panel, rank, Bv, Pv)
    assert x["k"] == 10 and x["weak"]["models"] == panel[:10] and x["strong"]["models"] == panel[19:]
    assert x["weak"]["mean"] == pytest.approx(6.0, abs=0.3) and x["weak_benefits"]
    assert x["strong_equivalent"] and x["claim_holds"]
    assert x["weak_minus_strong"]["mean"] == pytest.approx(x["weak"]["mean"] - x["strong"]["mean"])
    lo90, hi90 = x["strong"]["ci90"]
    lo95, hi95 = x["strong"]["ci95"]
    assert lo95 < lo90 < hi90 < hi95
    # a strong tercile that still gains 2.5 pp is not inside the 2 pp margin
    y = rex.tercile_block(panel, rank, Bv, Pv + np.where(rank >= 19, 2.3, 0.0)[None, :])
    assert not y["strong_equivalent"] and not y["claim_holds"]
    m = rex.macros({**R_MIN, "terciles": {"strict": x}})
    assert m["exhTerClaimStrict"] == "yes" and m["exhTerK"] == "10" and "exhTerStrongStrictLoNinety" in m


@pytest.mark.skipif(not (ROOT / "data" / "tokenizers" / "phi-3-mini-4k-instruct").is_dir(), reason="local tokenizers absent")
def test_cut_to_cap_keeps_short_texts_and_cuts_long_ones():
    short, long = "The answer is \\boxed{4}.", " ".join(["step"] * 3000)
    cut, n = sex.cut_to_cap([short, long], "phi-3-mini-4k-instruct", cap=50)
    assert cut[0] == short and n[0] < 50 and n[1] > 50
    assert len(cut[1]) < len(long) and long.startswith(cut[1].strip())


def test_native_4k_set_matches_the_request_builder():
    bx = load_script("build_exemplar_requests")
    assert set(sex.NATIVE_4K) == bx.NATIVE_4K
    assert np.isclose(sex.CAP_4K, 2048)
