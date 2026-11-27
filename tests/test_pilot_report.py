"""The released pilot report (release/pilot_report/pilot_report.py) against the repository's estimators.

Every output must equal src/estimators.py (reliability, pair_agreement, prompts_needed_gaussian),
src/reliability_map.py (agreement and setting_point with exact=False, reference_c, spearman_brown, k_needed),
src/instability.py (pair_matrices_draws) and src/pilot_checks.py (chosen_k), to 1e-12: on random arrays, on the
examples in release/pilot_report/examples/ and against the frozen pilot predictions of the reliability map.

1.1.0 (two baselines, who benefits, split pilot, decision): src/robustness.py (cross_slope, pair_reversals),
src/intervals.CrossMoment, src/decision.py (third_size, choose, capability_cross, regret_plugin) and
src/split_pilot.py (assignment, split_gains, single_gains, choices), to 1e-12, on random arrays and on Example 3 against
the saved 100-item pilots of the strict decision design (manifests/predictions_decision_strict.npz). The README's
printed outputs must equal the tool's.
"""
import ast
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import estimators as est
from src import instability as ins
from src import pilot_checks as pc
from src import reliability_map as rm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELEASE = os.path.join(ROOT, "pilot_report")
EXAMPLES = os.path.join(RELEASE, "examples")
sys.path.insert(0, RELEASE)
import pilot_report as pr  # noqa: E402

TOL = 1e-12
TARGETS = (0.80, 0.90)


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


def long_df(B, P, baseline="base", intervention="treat", models=None):
    K, M = B.shape
    models = models or [f"m{m}" for m in range(M)]
    rows = [{"model": models[m], "draw": k, "arm": arm, "score": float(X[k, m])}
            for k in range(K) for arm, X in ((baseline, B), (intervention, P)) for m in range(M)]
    return pd.DataFrame(rows)


def check_block(blk, X):
    """One quantity block of the report against the repository functions on the (K, M) array X."""
    K = X.shape[0]
    r = est.reliability(X)
    assert close(blk["R"], r["R"]) and close(blk["signal"], r["signal"]) and close(blk["within"], r["total"])
    assert close(blk["noise"], r["total"] - r["signal"])
    assert blk["R_status"] == rm.r_status(r["R"])
    ag = rm.agreement(X, exact=False)
    for k in ("C", "same", "opposite", "tie"):
        assert close(blk[k], ag[k]), k
    assert close(blk["C"], est.pair_agreement(X)["C"])
    same, tie = ins.pair_matrices_draws(X)
    i, j = np.triu_indices(X.shape[1], 1)
    assert close(blk["C"], float(np.mean(same[i, j] + 0.5 * tie[i, j])))
    assert close(blk["C_ref"], rm.reference_c(r["R"]))
    assert close(blk["R_K"], rm.spearman_brown(r["R"], K))
    assert close(blk["C_ref_K"], rm.reference_c(r["R"], K))
    for t in TARGETS:
        got, chosen = blk["draws_needed"][f"{t:.2f}"], pc.chosen_k(r["R"], t)
        assert got == chosen
        if chosen is not None:
            assert got == rm.k_needed(r["R"], t)
            assert got == est.prompts_needed_gaussian(min(r["R"], 1.0), t)
    assert blk["unresolved"] == (pc.chosen_k(r["R"], 0.8) is None)


def check_report(res, B, P):
    check_block(res["benefit"], P - B)
    check_block(res["accuracy_baseline"], B)
    check_block(res["accuracy_intervention"], P)
    pt = rm.setting_point(B, P, exact=False)            # the map's point: accuracy there is the baseline arm
    for q, mine in (("benefit", "benefit"), ("accuracy", "accuracy_baseline")):
        for k in ("R", "C", "C_ref"):
            assert close(res[mine][k], pt[q][k]), (q, k)
        assert close(res[mine]["signal"], pt[q]["signal"]) and close(res[mine]["within"], pt[q]["within"])
    assert res["unresolved"] == res["benefit"]["unresolved"]


# --------------------------------------------------------------------------- random arrays


@pytest.mark.parametrize("seed", range(40))
def test_random_arrays_match_repository(seed):
    rng = np.random.default_rng(seed)
    K, M = int(rng.integers(2, 9)), int(rng.integers(2, 12))
    if seed % 2:                                        # count-valued scores: many ties
        n = int(rng.integers(5, 40))
        B = 100.0 * rng.integers(0, n + 1, (K, M)) / n
        P = 100.0 * rng.integers(0, n + 1, (K, M)) / n
    else:                                               # model effects plus prompt noise
        theta, delta = rng.normal(50, 15, M), rng.normal(5, rng.uniform(0, 8), M)
        B = theta + rng.normal(0, rng.uniform(0.5, 10), (K, M))
        P = B + delta + rng.normal(0, rng.uniform(0.5, 10), (K, M))
    res = pr.report_arrays(B, P)
    check_report(res, B, P)
    full = pr.pilot_report(long_df(B, P), "base", "treat")
    for q in ("benefit", "accuracy_baseline", "accuracy_intervention"):
        for k in ("R", "C", "C_ref", "R_K", "C_ref_K", "signal", "within"):
            assert close(full[q][k], res[q][k])
        assert full[q]["draws_needed"] == res[q]["draws_needed"]


def test_negative_r_is_reported_as_estimated():
    # two draws whose centred benefits point in opposite directions: negative cross-draw moment
    G = np.array([[3.0, -3.0, 0.0, 1.0], [-3.0, 3.0, 1.0, 0.0]])
    B = np.full_like(G, 50.0) + np.arange(4)
    res = pr.report_arrays(B, B + G)
    r = est.reliability(G)
    assert r["R"] < 0
    blk = res["benefit"]
    assert blk["R"] == r["R"] and blk["R_status"] == "negative estimate"
    assert math.isnan(blk["C_ref"]) and math.isnan(blk["R_K"])
    assert blk["unresolved"] and "negative signal" in blk["reason"]
    assert all(v is None for v in blk["draws_needed"].values())
    assert pc.chosen_k(r["R"]) is None and rm.k_needed(r["R"], 0.8) is None
    check_report(res, B, B + G)
    assert f"{r['R']:.3f}" in pr.format_report(res)                 # printed as estimated, not as zero


# --------------------------------------------------------------------------- unresolved cases


def test_one_draw_is_unresolved():
    B, P = np.array([[40.0, 50.0, 60.0]]), np.array([[45.0, 52.0, 70.0]])
    with pytest.raises(ValueError):
        est.reliability(P - B)                          # the repository needs two draws
    res = pr.report_arrays(B, P)
    for q in ("benefit", "accuracy_baseline", "accuracy_intervention"):
        blk = res[q]
        assert blk["unresolved"] and blk["R"] is None and blk["C"] is None
        assert "fewer than two draws" in blk["reason"]
        assert all(v is None for v in blk["draws_needed"].values())
    assert res["unresolved"] and any("only 1 draw" in w for w in res["warnings"])
    assert res["per_model"][0]["gain_mean"] == 5.0 and math.isnan(res["per_model"][0]["gain_sd_across_draws"])
    text = pr.format_report(res)
    assert "benefit: unresolved, fewer than two draws" in text and "Warning: only 1 draw" in text


def test_constant_benefits_are_unresolved():
    rng = np.random.default_rng(3)
    B = rng.integers(20, 80, (5, 6)).astype(float)          # whole numbers, so P - B is exact
    P = B + np.array([[2.0], [4.0], [1.0], [3.0], [5.0]])   # every model gains the same within a draw
    res = pr.report_arrays(B, P)
    blk = res["benefit"]
    r = est.reliability(P - B)
    assert math.isnan(r["R"]) and pc.chosen_k(r["R"]) is None
    assert blk["R"] is not None and math.isnan(blk["R"]) and blk["within"] == r["total"]
    assert blk["unresolved"] and "zero within-draw variance" in blk["reason"]
    assert blk["R_status"] == rm.r_status(r["R"])
    assert blk["C"] == 0.5 and blk["tie"] == 1.0          # every comparison is a tie
    assert not res["accuracy_baseline"]["unresolved"]
    check_report(res, B, P)


def test_zero_signal_is_unresolved():
    G = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])   # centred draws orthogonal: S = 0 exactly
    B = np.tile([30.0, 40.0, 50.0, 60.0], (2, 1))
    res = pr.report_arrays(B, B + G)
    r = est.reliability(G)
    assert r["R"] == 0.0 and pc.chosen_k(0.0) is None and rm.k_needed(0.0, 0.8) == math.inf
    blk = res["benefit"]
    assert blk["R"] == 0.0 and blk["unresolved"] and "zero signal" in blk["reason"]
    assert blk["C_ref"] == 0.5
    check_report(res, B, B + G)


def test_identical_draws_need_one_draw():
    row = np.array([10.0, 25.0, 3.0, 40.0])
    B = np.tile([30.0, 40.0, 50.0, 60.0], (3, 1))
    res = pr.report_arrays(B, B + np.tile(row, (3, 1)))
    blk = res["benefit"]
    assert close(blk["R"], 1.0) and blk["draws_needed"] == {"0.80": 1, "0.90": 1}
    assert not blk["unresolved"] and "no prompt noise" in blk.get("note", "")
    assert blk["C"] == 1.0
    assert any("only 3 draws" in w for w in res["warnings"])


def test_no_warning_from_four_draws():
    rng = np.random.default_rng(0)
    B = rng.normal(50, 10, (4, 5))
    res = pr.report_arrays(B, B + rng.normal(5, 5, (4, 5)))
    assert res["warnings"] == []


# --------------------------------------------------------------------------- ties


def test_ties_follow_the_tolerance():
    # pair (0, 1): exact tie in draw 0, a difference of 5e-10 pp (a tie) in draw 1, 2e-9 pp (decisive) in draw 2
    B = np.array([[50.0, 50.0, 10.0], [50.0, 50.0 + 5e-10, 10.0], [50.0, 50.0 + 2e-9, 10.0]])
    blk = pr.quantity_block(B, "accuracy")
    ag = rm.agreement(B, exact=False)
    assert blk["C"] == ag["C"] and blk["tie"] == ag["tie"]
    # by hand: 3 model pairs; pairs (0, 2) and (1, 2) agree in all 6 ordered draw pairs; pair (0, 1) is tied in two
    # draws and decisive in one, so every ordered draw pair involves a tie: C = (1 + 1 + 0.5) / 3
    assert close(blk["C"], 2.5 / 3) and close(blk["tie"], 1 / 3)


def test_float_sum_ties():
    # 100 a / n - 100 b / n against 100 c / n - 100 d / n with a - b = c - d: equal gains, not bit-identical
    n = 7
    Bc = np.array([[0, 2], [0, 1]])
    Pc = np.array([[1, 3], [2, 3]])                     # both models gain 1 / 7 in draw 1 and 2 / 7 in draw 2
    B, P = 100.0 * Bc / n, 100.0 * Pc / n
    G = P - B
    assert G[0, 0] != G[0, 1] and G[1, 0] != G[1, 1]    # equal gains, different floating-point values
    assert np.all(np.abs(G[:, 0] - G[:, 1]) < 1e-13)
    res = pr.report_arrays(B, P)
    assert res["benefit"]["tie"] == 1.0 and res["benefit"]["C"] == 0.5
    check_block(res["benefit"], G)


def test_two_by_two_by_hand():
    G = np.array([[1.0, 1.0], [2.0, 1.0]])              # draw 1 ties the pair, draw 2 orders it
    blk = pr.quantity_block(G, "benefit")
    assert blk["C"] == 0.5 and blk["tie"] == 1.0 and blk["same"] == 0.0


# --------------------------------------------------------------------------- comparison-arm check


def toy_tokens():
    rows = []
    for m, lengths in (("a", [5] * 19 + [17]),         # 1 of 20 over 16: 5.0%, not above the maximum
                       ("b", [5] * 18 + [17, 300]),    # 2 of 20: 10%, flagged
                       ("c", [16] * 20)):              # exactly at the budget: not over
        rows += [{"model": m, "arm": "direct", "n_output_tokens": t} for t in lengths]
    rows += [{"model": "a", "arm": "cot", "n_output_tokens": 500}] * 30     # other arms are ignored
    rows += [{"model": "z", "arm": "direct", "n_output_tokens": 3}]          # not in the score table
    return pd.DataFrame(rows)


def test_token_check_toy_table():
    chk = pr.token_check(toy_tokens(), "direct", ["a", "b", "c", "d"], budget=16, max_share=0.05)
    by = {r["model"]: r for r in chk["models"]}
    assert by["a"]["n_outputs"] == 20 and by["a"]["n_over_budget"] == 1 and by["a"]["share_over_budget"] == 0.05
    assert not by["a"]["flagged"]
    assert by["b"]["flagged"] and by["b"]["share_over_budget"] == 0.1 and by["b"]["max_tokens"] == 300
    assert by["c"]["n_over_budget"] == 0 and not by["c"]["flagged"] and by["c"]["median_tokens"] == 16
    assert chk["flagged"] == ["b"] and chk["not_checked"] == ["d"] and chk["not_in_scores"] == ["z"]
    assert [r["model"] for r in chk["models"]] == ["a", "b", "c", "z"]
    tight = pr.token_check(toy_tokens(), "direct", ["a", "b", "c"], budget=4, max_share=0.5)
    assert set(tight["flagged"]) == {"a", "b", "c"}


def test_token_check_counts_equal_rows():
    df = toy_tokens()
    agg = df.groupby(["model", "arm", "n_output_tokens"]).size().rename("count").reset_index()
    a, b = pr.token_check(df, "direct", ["a", "b", "c"]), pr.token_check(agg, "direct", ["a", "b", "c"])
    assert a["models"] == b["models"] and a["flagged"] == b["flagged"]


@pytest.mark.parametrize("seed", range(10))
def test_weighted_median_equals_numpy(seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 50, int(rng.integers(1, 40)))
    vals, counts = np.unique(v, return_counts=True)
    assert pr.weighted_median(vals, counts) == float(np.median(v))


def test_token_check_errors():
    with pytest.raises(ValueError, match="no rows"):
        pr.token_check(toy_tokens(), "forced", ["a"])
    bad = toy_tokens().assign(count=1.5)
    with pytest.raises(ValueError, match="count"):
        pr.token_check(bad, "direct", ["a"])


# --------------------------------------------------------------------------- input checks


def test_input_errors():
    rng = np.random.default_rng(1)
    B = rng.normal(50, 10, (4, 3))
    df = long_df(B, B + 3)
    with pytest.raises(ValueError, match="lacks"):
        pr.pilot_report(df.iloc[1:], "base", "treat")
    with pytest.raises(ValueError, match="repeat"):
        pr.pilot_report(pd.concat([df, df.iloc[:1]]), "base", "treat")
    with pytest.raises(ValueError, match="not in the score table"):
        pr.pilot_report(df, "base", "other")
    with pytest.raises(ValueError, match="two or more models"):
        pr.pilot_report(df[df["model"] == "m0"], "base", "treat")
    with pytest.raises(ValueError, match="not finite"):
        pr.pilot_report(df.assign(score=df["score"].where(df.index != 3, np.nan)), "base", "treat")
    extra = pd.concat([df, df[df["arm"] == "base"].assign(arm="third")])
    res = pr.pilot_report(extra, "base", "treat")
    assert any("other arms are ignored" in w for w in res["warnings"])
    res = pr.pilot_report(df.assign(score=df["score"] / 100), "base", "treat")
    assert any("percentage points" in w for w in res["warnings"])


def test_self_contained():
    with open(os.path.join(RELEASE, "pilot_report.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "itertools", "json", "math", "sys", "textwrap", "numpy", "pandas"}


# --------------------------------------------------------------------------- the two examples


def example_arrays(name, baseline, intervention):
    df = pd.read_csv(os.path.join(EXAMPLES, name), float_precision="round_trip")
    models = list(pd.unique(df["model"]))
    draws = list(pd.unique(df["draw"]))
    out = []
    for a in (baseline, intervention):
        piv = df[df["arm"] == a].pivot(index="draw", columns="model", values="score").loc[draws, models]
        out.append(piv.to_numpy(float))
    return out[0], out[1], models, df


def frozen_pilot(path, setting):
    with open(os.path.join(ROOT, "manifests", path), encoding="utf-8") as fh:
        j = json.load(fh)
    return j, j["settings"][setting]["4"]


def check_frozen(res, pred):
    for mine, q in (("benefit", "benefit"), ("accuracy_baseline", "accuracy")):
        p = pred[q]
        assert close(res[mine]["R"], p["R"]) and close(res[mine]["signal"], p["signal"])
        assert close(res[mine]["within"], p["within"])
        assert close(res[mine]["C_ref"], p["C_one_draw"]) and close(res[mine]["C_ref_K"], p["C_groups"]["4"])
        assert res[mine]["draws_needed"] == p["K_needed"]


def test_dove_example_matches_repository_and_frozen_map():
    B, P, models, df = example_arrays("dove_arc_challenge_scores.csv", "zero_shot", "five_shot")
    res = pr.pilot_report(os.path.join(EXAMPLES, "dove_arc_challenge_scores.csv"), "zero_shot", "five_shot")
    assert res["M"] == 5 and res["K"] == 4 and res["models"] == models
    check_report(res, B, P)
    _, pred = frozen_pilot("predictions_reliability_map_dove_tasks.json", "dove:ai2_arc.arc_challenge")
    check_frozen(res, pred)
    with open(os.path.join(ROOT, "manifests", "sampler_reliability_map_dove_tasks.json"), encoding="utf-8") as fh:
        draws = json.load(fh)["benchmarks"]
    assert next(iter(draws)) == "ai2_arc.arc_challenge"             # the stated rule: first in the frozen order
    cfg = df.drop_duplicates("draw")["config_id"].tolist()
    assert cfg == draws["ai2_arc.arc_challenge"]["pilot_config_ids"]
    assert not res["unresolved"] and res["benefit"]["draws_needed"] == {"0.80": 13, "0.90": 58}


def test_gsm8k_example_matches_repository_and_frozen_map():
    path = os.path.join(EXAMPLES, "gsm8k29_scores.csv")
    B, P, models, df = example_arrays("gsm8k29_scores.csv", "direct_answer", "zero_shot_cot")
    res = pr.pilot_report(path, "direct_answer", "zero_shot_cot",
                          tokens=os.path.join(EXAMPLES, "gsm8k29_tokens.csv"))
    assert res["M"] == 29 and res["K"] == 4
    check_report(res, B, P)
    j, pred = frozen_pilot("predictions_reliability_map_gsm8k30.json", "gsm8k30")
    assert j["models"] == models
    check_frozen(res, pred)
    with open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8") as fh:
        units = json.load(fh)["discovery_config_ids"][:4]
    assert df.drop_duplicates("draw")["config_id"].tolist() == units
    chk = res["comparison_arm_check"]
    assert chk["not_checked"] == [] and chk["not_in_scores"] == []
    assert all(r["n_outputs"] == 4 * 1319 for r in chk["models"])
    tok = pd.read_csv(os.path.join(EXAMPLES, "gsm8k29_tokens.csv"))
    for r in chk["models"]:
        g = tok[tok["model"] == r["model"]]
        assert r["n_over_budget"] == int(g.loc[g["n_output_tokens"] > pr.TOKEN_BUDGET, "count"].sum())
        assert r["flagged"] == (r["n_over_budget"] / r["n_outputs"] > pr.MAX_OVER_BUDGET)
        assert r["median_tokens"] == float(np.median(np.repeat(g["n_output_tokens"], g["count"])))


def test_example_files_are_small():
    for name in os.listdir(EXAMPLES):
        assert os.path.getsize(os.path.join(EXAMPLES, name)) < 1_000_000, name


# --------------------------------------------------------------------------- the command line


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(7)
    B = rng.normal(50, 10, (5, 4))
    P = B + rng.normal(6, 6, 4) + rng.normal(0, 2, (5, 4))
    long_df(B, P, "direct", "cot").to_csv(tmp_path / "scores.csv", index=False)
    toy_tokens().assign(model=lambda d: d["model"].map({"a": "m0", "b": "m1", "c": "m2", "z": "zz"})).to_csv(
        tmp_path / "tokens.csv", index=False)
    out = tmp_path / "report.json"
    code = pr.main([str(tmp_path / "scores.csv"), "--baseline", "direct", "--intervention", "cot",
                    "--tokens", str(tmp_path / "tokens.csv"), "--json", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "R (reliability across draws)" in printed and pr.FLAG_MESSAGE in printed
    j = json.loads(out.read_text(encoding="utf-8"))
    ref = est.reliability(P - B)
    assert close(j["benefit"]["R"], ref["R"])
    assert j["benefit"]["draws_needed"]["0.80"] == pc.chosen_k(ref["R"], 0.8)
    assert j["comparison_arm_check"]["flagged"] == ["m1"]
    assert j["comparison_arm_check"]["not_checked"] == ["m3"]
    assert j["comparison_arm_check"]["not_in_scores"] == ["zz"]

    # as a script, in an isolated interpreter started outside the repository (nothing from src/ is reachable)
    proc = subprocess.run([sys.executable, "-I", os.path.join(RELEASE, "pilot_report.py"), "scores.csv",
                           "--baseline", "direct", "--intervention", "cot", "--json", "r2.json"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "draws for 80% agreement" in proc.stdout
    j2 = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))
    assert j2["benefit"] == j["benefit"]


def test_cli_json_has_no_nan(tmp_path):
    B = np.tile([30.0, 40.0, 50.0], (3, 1))
    P = B + np.array([[1.0], [2.0], [3.0]])             # constant benefits: R is NaN
    long_df(B, P).to_csv(tmp_path / "s.csv", index=False)
    assert pr.main([str(tmp_path / "s.csv"), "--baseline", "base", "--intervention", "treat",
                    "--json", str(tmp_path / "r.json")]) == 0
    j = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert j["benefit"]["R"] is None and j["benefit"]["unresolved"] and j["unresolved"]
    assert j["benefit"]["draws_needed"] == {"0.80": None, "0.90": None}


def test_cli_error_exit(tmp_path, capsys):
    pd.DataFrame({"model": ["a"], "draw": [1], "arm": ["x"], "score": [1.0]}).to_csv(tmp_path / "s.csv", index=False)
    assert pr.main([str(tmp_path / "s.csv"), "--baseline", "x", "--intervention", "y"]) == 2
    assert "error:" in capsys.readouterr().err


# =========================================================================== 1.1.0


from src import decision as dc  # noqa: E402
from src import intervals as iv  # noqa: E402
from src import robustness as rb  # noqa: E402
from src import split_pilot as sp  # noqa: E402

ARMS3 = ("req", "fd", "cot")


def three_df(B, F, P, models=None, arms=ARMS3):
    K, M = B.shape
    models = models or [f"m{m}" for m in range(M)]
    rows = [{"model": models[m], "draw": k, "arm": a, "score": float(X[k, m])}
            for k in range(K) for a, X in zip(arms, (B, F, P)) for m in range(M)]
    return pd.DataFrame(rows)


def random_three(rng, K=None, M=None):
    K = K or int(rng.integers(2, 9))
    M = M or int(rng.integers(3, 14))
    if rng.random() < 0.5:                                # count-valued: ties
        n = int(rng.integers(5, 30))
        B, F, P = (100.0 * rng.integers(0, n + 1, (K, M)) / n for _ in range(3))
    else:
        theta = rng.normal(50, 15, M)
        B = theta + rng.normal(0, rng.uniform(0.5, 8), (K, M))
        F = theta - rng.normal(5, 5, M) + rng.normal(0, rng.uniform(0.5, 8), (K, M))
        P = B + rng.normal(8, 6, M) + rng.normal(0, rng.uniform(0.5, 8), (K, M))
    return B, F, P


def same_float(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- estimators against the repository


@pytest.mark.parametrize("seed", range(30))
def test_cross_slope_matches_repository(seed):
    rng = np.random.default_rng(seed)
    K, M = int(rng.integers(2, 10)), int(rng.integers(2, 14))
    X = rng.normal(50, 15, (K, M)) if seed % 3 else 100.0 * rng.integers(0, 8, (K, M)) / 7
    Y = rng.normal(0, 10, (K, M)) + rng.normal(0, 1) * X
    mine = pr.cross_slope(X, Y)
    s, vx = rb.cross_slope(X, Y)
    assert same_float(mine["slope"], s) and same_float(mine["V_x"], vx)
    assert same_float(mine["A"], iv.CrossMoment(Y, X).estimate())
    E = np.tile(rng.normal(120, 15, M), (K, 1))           # a capability score, constant over draws
    mine, (s, vx) = pr.cross_slope(E, Y), rb.cross_slope(E, Y)
    assert same_float(mine["slope"], s) and same_float(mine["V_x"], vx)


def test_negative_cross_variance_leaves_the_slope_undefined():
    B = np.array([[50.0, 60.0, 40.0], [50.0, 40.0, 60.0]])  # opposite centred draws: V_x < 0
    P = B + np.array([[5.0, 9.0, 1.0], [4.0, 8.0, 2.0]])
    mine = pr.cross_slope(B, P - B)
    s, vx = rb.cross_slope(B, P - B)
    assert vx < 0 and mine["V_x"] == vx and math.isnan(mine["slope"]) and math.isnan(s)
    res = pr.report_arrays(B, P, decision=True)
    e = res["who_benefits"]["baselines"][0]["slope_on_accuracy"]
    assert not e["defined"] and e["V_x"] == vx and "not above zero" in e["reason"]
    cap = res["decision"]["baselines"][0]["capability"]
    assert not cap["slope_defined"] and cap["n_tied"] == 3
    pred, s2 = dc.capability_cross(B, P - B)
    assert np.allclose(cap["prediction"], pred, atol=TOL, rtol=0) and math.isnan(s2)
    assert np.allclose(cap["weights"], [1 / 3] * 3, atol=TOL, rtol=0)     # every model tied: 1 place shared
    text = pr.format_report(res)
    assert "undefined" in text and f"{vx:.2f}" in text


@pytest.mark.parametrize("seed", range(30))
def test_pair_reversals_match_repository(seed):
    rng = np.random.default_rng(seed)
    M = int(rng.integers(2, 25))
    if seed % 2:
        n = int(rng.integers(3, 12))                     # exact ratios: ties and gaps that equal a threshold
        g1, g2 = 100.0 * rng.integers(0, n, M) / n / 10, 100.0 * rng.integers(0, n, M) / n / 10
    else:
        g1, g2 = rng.normal(30, 15, M), rng.normal(30, 15, M)
    mine, ref = pr.pair_reversals(g1, g2), rb.pair_reversals(g1, g2)
    for k in ("pairs", "reversed", "ties", "by_threshold"):
        assert mine[k] == ref[k], k
    assert same_float(mine["reversed_share"], ref["reversed_share"])
    assert len(mine["reversed_index"]) == mine["reversed"]


def test_pair_reversals_threshold_tolerance():
    # a gap of 1 pp reached by different sums: within 1e-9 of the threshold, so not above it
    g1 = np.array([0.0, 100.0 * 3 / 7 - 100.0 * 2 / 7 + 1.0 - 100.0 / 7])
    g2 = np.array([1.0, 0.0])
    mine, ref = pr.pair_reversals(g1, g2), rb.pair_reversals(g1, g2)
    assert mine["by_threshold"] == ref["by_threshold"] and mine["by_threshold"][1]["pairs_both_above"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_choose_and_third_size_match_repository(seed):
    rng = np.random.default_rng(seed)
    M = int(rng.integers(2, 40))
    p = np.round(rng.normal(20, 10, M)) if seed % 2 else rng.normal(20, 10, M)    # rounded: ties at the cut
    n = dc.third_size(M)
    assert pr.third_size(M) == n
    w, t = pr.choose(p, n)
    w2, t2 = dc.choose(p, n)
    assert np.array_equal(w, w2) and t == t2
    q = rng.permutation(M)
    assert pr.shared(w, w[q]) == float(np.minimum(w, w[q]).sum())
    for m in range(2, 61):
        assert pr.third_size(m) == dc.third_size(m)


@pytest.mark.parametrize("seed", range(20))
def test_capability_prediction_and_regret_match_repository(seed):
    rng = np.random.default_rng(seed)
    B, F, P = random_three(rng)
    G = P - B
    pred, s = pr.capability_prediction(B, G)
    pred2, s2 = dc.capability_cross(B, G)
    assert same_float(s, s2) and np.allclose(pred, pred2, atol=TOL, rtol=0)
    n = dc.third_size(B.shape[1])
    W = np.array([pr.choose(G[k], n)[0] for k in range(G.shape[0])])
    assert np.allclose(pr.regret(W, P - F, n), dc.regret_plugin(W, P - F, n), atol=TOL, rtol=0)


def split_half_by_hand(B, G, n):
    """Every ordered (choice half, evaluation half), with the repository's functions."""
    import itertools
    K = G.shape[0]
    own, cap = [], []
    for h in sorted({K // 2, K - K // 2}):
        for a in itertools.combinations(range(K), h):
            a = list(a)
            b = [k for k in range(K) if k not in a]
            w_own = dc.choose(G[a].mean(0), n)[0]
            w_cap = dc.choose(dc.capability_cross(B[a], G[a])[0], n)[0]
            own.append(float(dc.regret_plugin(w_own, G[b], n)[0]))
            cap.append(float(dc.regret_plugin(w_cap, G[b], n)[0]))
    return own, cap


@pytest.mark.parametrize("seed", range(16))
def test_split_half_regret_matches_repository(seed):
    rng = np.random.default_rng(seed)
    B, F, P = random_three(rng, K=int(rng.integers(4, 9)))
    G = P - B
    n = dc.third_size(B.shape[1])
    out = pr.split_half_regret(B, G, n)
    own, cap = split_half_by_hand(B, G, n)
    assert out["exhaustive"] and out["n_choices"] == len(own)
    for key, v in (("own_gain", own), ("capability", cap)):
        assert same_float(out[key]["mean"], float(np.mean(v))) and out[key]["min"] == min(v)
        assert out[key]["max"] == max(v)
    assert min(own) >= -1e-12 and min(cap) >= -1e-12      # never negative beyond floating-point residue
    assert pr.split_half_regret(B[:3], G[:3], n) is None   # fewer than four draws


def test_split_half_regret_random_splits_beyond_the_limit():
    rng = np.random.default_rng(5)
    B, F, P = random_three(rng, K=18, M=6)                 # C(18, 9) = 48,620 choices: random splits
    out = pr.split_half_regret(B, P - B, 2, seed=11)
    assert not out["exhaustive"] and out["n_choices"] == 2 * pr.RANDOM_SPLITS
    assert out == pr.split_half_regret(B, P - B, 2, seed=11)
    pairs, exh = pr.half_splits(18, seed=11)
    assert not exh and all(sorted(a + b) == list(range(18)) and len(a) == 9 for a, b in pairs)
    pairs, exh = pr.half_splits(5)
    assert exh and len(pairs) == 20 and all(sorted(a + b) == list(range(5)) for a, b in pairs)


@pytest.mark.parametrize("args", [(1, 100, 20261121, 1), (200, 100, 20261121, 1), (7, 100, 3, 25), (4, 10, 0, 2),
                                  (3, 9, 1, 3), (5, 7, 9, 1)])
def test_split_assignment_matches_repository(args):
    assert np.array_equal(pr.split_assignment(*args), sp.assignment(*args))


# --------------------------------------------------------------------------- the split pilot


def item_long(S, rows, items, models, arms=ARMS3):
    """Long item table: one draw per pilot (rows, items); S: {arm: (configs, M, N)} 0/1."""
    recs = []
    for p, (r, its) in enumerate(zip(rows, items)):
        for m, name in enumerate(models):
            for q in its:
                for a in arms:
                    recs.append({"model": name, "draw": p, "item": f"i{q}", "arm": a, "score": int(S[a][r, m, q])})
    return pd.DataFrame(recs)


def to_wide(df):
    """One column per arm, rows in the long table's order of (model, draw, item)."""
    w = df.pivot(index=["model", "draw", "item"], columns="arm", values="score")
    order = pd.MultiIndex.from_frame(df[["model", "draw", "item"]].drop_duplicates())
    out = w.loc[order].reset_index()
    out.columns.name = None
    return out


def random_pilot(rng, P_=None, n=None, M=None):
    M = M or int(rng.integers(3, 12))
    N, C = 60, 5
    S = {a: rng.integers(0, 2, (C, M, N)).astype(np.int8) for a in ARMS3}
    S["cot"] = np.maximum(S["cot"], rng.integers(0, 2, (C, M, N)).astype(np.int8))
    P_ = P_ or int(rng.integers(1, 6))
    n = n or 2 * int(rng.integers(2, 11))
    rows = rng.integers(0, C, P_)
    items = np.array([np.sort(rng.choice(N, n, replace=False)) for _ in range(P_)])
    return S, rows, items, [f"m{m}" for m in range(M)]


@pytest.mark.parametrize("seed", range(16))
def test_split_pilot_matches_repository(seed):
    rng = np.random.default_rng(seed)
    S, rows, items, models = random_pilot(rng)
    M = len(models)
    df = item_long(S, rows, items, models)
    scores = three_df(*random_three(rng, K=3, M=M), models=models)
    sd = int(rng.integers(0, 1000))
    res = pr.pilot_report(scores, "req", "cot", forced="fd", items=df if seed % 2 else to_wide(df), seed=sd)
    spl = res["split_pilot"]
    assert spl["form"] == ("long" if seed % 2 else "wide")
    rq = sp.assignment(len(rows), items.shape[1], sd, 1)
    study = sp.split_gains(S["req"], S["fd"], S["cot"], rows, items, rq)
    assert np.allclose(spl["per_draw_split_gain"], study, atol=TOL, rtol=0)
    pooled = np.array([p["split_gain"] for p in spl["per_model"]])
    assert np.allclose(pooled, study.mean(0), atol=TOL, rtol=0)
    n = dc.third_size(M)
    w, t = dc.choose(pooled, n)
    assert np.array_equal(spl["weights"], w) and spl["n_tied"] == t
    for role, b in (("requested", "req"), ("forced", "fd")):
        single = sp.single_gains(S[b], S["cot"], rows, items).mean(0)
        assert np.allclose([p[f"gain_{role}_items"] for p in spl["per_model"]], single, atol=TOL, rtol=0)
        ws = np.array(spl["single_baseline_thirds"][role]["weights"])
        assert np.array_equal(ws, dc.choose(single, n)[0])
    # per-pilot choices, as the study takes them, from the tool's per-draw gains
    Wst, _ = sp.choices(study, n)
    Wt = np.array([pr.choose(r, n)[0] for r in np.array(spl["per_draw_split_gain"])])
    assert np.array_equal(Wst, Wt)
    assert spl["assignment"] == "seeded"
    assert all(v == items.shape[1] // 2 for v in spl["requested_items_per_draw"].values())
    assert sum(c["baseline"] == "requested" for c in spl["assignment_cells"]) == int(rq.sum())
    thirds = {b["role"]: np.array(b["top_third"]["weights"]) for b in res["who_benefits"]["baselines"]}
    for role, v in spl["shared_with"].items():
        assert v == float(np.minimum(w, thirds[role]).sum())


def test_split_pilot_long_and_wide_forms_agree():
    rng = np.random.default_rng(2)
    S, rows, items, models = random_pilot(rng, P_=3, n=8, M=5)
    df = item_long(S, rows, items, models)
    scores = three_df(*random_three(rng, K=3, M=5), models=models)
    a = pr.pilot_report(scores, "req", "cot", forced="fd", items=df)["split_pilot"]
    b = pr.pilot_report(scores, "req", "cot", forced="fd", items=to_wide(df))["split_pilot"]
    assert a["form"] == "long" and b["form"] == "wide"
    for k in ("per_draw_split_gain", "weights", "assignment_cells", "per_model", "shared_with"):
        assert a[k] == b[k], k


def test_split_pilot_strata_match_repository():
    rng = np.random.default_rng(8)
    M, n_strata, k, D = 4, 5, 4, 3
    models = [f"m{m}" for m in range(M)]
    recs, order = [], []
    for d in range(D):
        labs = [f"t{s}" for s in rng.permutation(n_strata)]            # stratum order differs between draws
        cells = [(labs[s], f"d{d}{labs[s]}i{i}") for s in range(n_strata) for i in range(k)]
        file_order = [cells[i] for i in rng.permutation(len(cells))]   # strata interleaved in the file
        # expected position order: strata by first appearance in the file, items in file order within a stratum
        first = list(dict.fromkeys(s for s, _ in file_order))
        order.append([it for s in first for (s2, it) in file_order if s2 == s])
        for m in models:
            for s, it in file_order:
                recs.append({"model": m, "draw": d, "item": it, "stratum": s, "req": int(rng.integers(0, 2)),
                             "fd": int(rng.integers(0, 2)), "cot": int(rng.integers(0, 2))})
    df = pd.DataFrame(recs)
    scores = three_df(*random_three(rng, K=2, M=M), models=models)
    spl = pr.pilot_report(scores, "req", "cot", forced="fd", items=df, seed=77)["split_pilot"]
    rq = sp.assignment(D, n_strata * k, 77, n_strata)
    got = {(c["draw"], c["item"]): c["baseline"] == "requested" for c in spl["assignment_cells"]}
    for d in range(D):
        assert [got[(str(d), it)] for it in order[d]] == rq[d].tolist()
    assert spl["n_strata"] == n_strata
    assert all(v == n_strata * (k // 2) for v in spl["requested_items_per_draw"].values())
    bad = df[df["item"] != df["item"].iloc[0]]                         # one stratum one item short in one draw
    with pytest.raises(ValueError, match="strata"):
        pr.pilot_report(scores, "req", "cot", forced="fd", items=bad)


def test_split_pilot_reads_an_assignment_already_run():
    rng = np.random.default_rng(4)
    S, rows, items, models = random_pilot(rng, P_=2, n=10, M=4)
    df = to_wide(item_long(S, rows, items, models))
    on_req = (((df["draw"] == 0) & df["item"].isin({f"i{q}" for q in items[0][:3]}))
              | ((df["draw"] == 1) & df["item"].isin({f"i{q}" for q in items[1][:6]})))
    df["fd"] = df["fd"].astype(float).where(~on_req)
    df["req"] = df["req"].astype(float).where(on_req)
    scores = three_df(*random_three(rng, K=2, M=4), models=models)
    res = pr.pilot_report(scores, "req", "cot", forced="fd", items=df)
    spl = res["split_pilot"]
    assert spl["assignment"] == "observed" and spl["seed"] is None
    base = np.where(on_req, df["req"], df["fd"])
    expect = (100.0 * (df["cot"] - base)).groupby(df["model"]).mean().loc[models].to_numpy()
    assert np.allclose([p["split_gain"] for p in spl["per_model"]], expect, atol=1e-10, rtol=0)
    assert all(p["gain_requested_items"] is None for p in spl["per_model"]) and "single_baseline_thirds" not in spl
    assert spl["requested_items_per_draw"] == {"0": 3, "1": 6}
    assert "read from the item table" in pr.format_report(res)


def test_item_input_errors():
    rng = np.random.default_rng(6)
    S, rows, items, models = random_pilot(rng, P_=2, n=6, M=3)
    df = to_wide(item_long(S, rows, items, models))
    scores = three_df(*random_three(rng, K=2, M=3), models=models)

    def run(x, **kw):
        return pr.pilot_report(scores, "req", "cot", forced="fd", items=x, **kw)

    mixed = df.astype({"fd": float})
    mixed.loc[0, "fd"] = np.nan                           # one model only: not the same for every model
    with pytest.raises(ValueError, match="exactly one baseline"):
        run(mixed)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run(df.assign(cot=df["cot"] * 100))
    with pytest.raises(ValueError, match="intervention scores"):
        run(df.assign(cot=df["cot"].astype(float).where(df.index != 2)))
    with pytest.raises(ValueError, match="models differ"):
        run(df[df["model"] != "m0"])
    first_item = df.loc[df["draw"] == 1, "item"].iloc[0]
    with pytest.raises(ValueError, match="same number of items"):
        run(df[~((df["draw"] == 1) & (df["item"] == first_item))])
    with pytest.raises(ValueError, match="repeat"):
        run(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError, match="non-numeric"):
        run(df.astype({"req": object}).assign(req=lambda d: d["req"].where(d.index != 1, "x")))
    with pytest.raises(ValueError, match="neither"):
        run(df.drop(columns=["fd"]))
    with pytest.raises(ValueError, match="needs the forced baseline"):
        pr.pilot_report(scores, "req", "cot", items=df)
    with pytest.raises(ValueError, match="three arms"):
        pr.pilot_report(scores, "req", "cot", forced="req")


# --------------------------------------------------------------------------- the two-baseline report


@pytest.mark.parametrize("seed", range(20))
def test_two_baseline_report_matches_repository(seed):
    rng = np.random.default_rng(seed)
    B, F, P = random_three(rng, K=int(rng.integers(2, 8)))
    K, M = B.shape
    models = [f"m{m}" for m in range(M)]
    cap = pd.DataFrame({"model": models, "capability": rng.normal(120, 20, M)})
    res = pr.pilot_report(three_df(B, F, P, models), "req", "cot", forced="fd", capability=cap, decision=True)
    check_report(res, B, P)                               # the requested baseline's blocks as in 1.0.0
    check_block(res["benefit_forced"], P - F)
    check_block(res["accuracy_forced"], F)
    for m, p in enumerate(res["per_model"]):
        assert close(p["forced_mean"], float(F[:, m].mean()))
        assert close(p["gain_forced_mean"], float((P - F)[:, m].mean()))
        assert close(p["gain_difference"], float((P - B)[:, m].mean() - (P - F)[:, m].mean()), 1e-11)
    wb = res["who_benefits"]
    n = dc.third_size(M)
    E = np.tile(cap["capability"].to_numpy(float), (K, 1))
    W = {}
    for e, X in zip(wb["baselines"], (B, F)):
        G = P - X
        s, vx = rb.cross_slope(X, G)
        assert same_float(e["slope_on_accuracy"]["slope"], s) and same_float(e["slope_on_accuracy"]["V_x"], vx)
        s, vx = rb.cross_slope(E, G)
        assert same_float(e["slope_on_capability"]["slope"], s) and same_float(e["slope_on_capability"]["V_x"], vx)
        W[e["role"]] = dc.choose(G.mean(0), n)[0]
        assert np.array_equal(e["top_third"]["weights"], W[e["role"]])
    assert wb["shared_top_third"] == float(np.minimum(W["requested"], W["forced"]).sum())
    ref = rb.pair_reversals((P - B).mean(0), (P - F).mean(0))
    for k in ("pairs", "reversed", "ties", "by_threshold"):
        assert wb["opposite_orderings"][k] == ref[k]
    for e, X in zip(res["decision"]["baselines"], (B, F)):
        G = P - X
        assert np.array_equal(e["own_gain"]["weights"], dc.choose(G.mean(0), n)[0])
        pred, s = dc.capability_cross(X, G)
        assert np.allclose(e["capability"]["prediction"], pred, atol=TOL, rtol=0)
        assert np.array_equal(e["capability"]["weights"], dc.choose(pred, n)[0])
        assert e["shared"] == float(np.minimum(e["own_gain"]["weights"], e["capability"]["weights"]).sum())
        if K >= 4:
            own, capr = split_half_by_hand(X, G, n)
            assert same_float(e["split_half_regret"]["own_gain"]["mean"], float(np.mean(own)))
            assert same_float(e["split_half_regret"]["capability"]["mean"], float(np.mean(capr)))
        else:
            assert e["split_half_regret"] is None
    text = pr.format_report(res)
    assert "Who benefits" in text and "Decision:" in text and "benefit/forced" in text
    json.dumps(pr.jsonable(res), allow_nan=False)


def test_one_draw_two_baselines():
    B, F, P = np.array([[40.0, 50.0, 60.0]]), np.array([[38.0, 55.0, 61.0]]), np.array([[70.0, 72.0, 90.0]])
    res = pr.report_arrays(B, P, F=F, forced="fd", decision=True)
    for e in res["who_benefits"]["baselines"]:
        assert e["slope_on_accuracy"]["slope"] is None and "fewer than two draws" in e["slope_on_accuracy"]["reason"]
    for e in res["decision"]["baselines"]:
        assert e["capability"] is None and e["shared"] is None and e["split_half_regret"] is None
    assert res["benefit_forced"]["unresolved"]
    assert "needs at least 4 draws" in pr.format_report(res)


def test_capability_missing_and_constant():
    rng = np.random.default_rng(9)
    B, F, P = random_three(rng, K=4, M=6)
    models = [f"m{m}" for m in range(6)]
    df = three_df(B, F, P, models)
    cap = pd.DataFrame({"model": models[1:] + ["extra"], "capability": [1.0, 3.0, 2.0, 7.0, 5.0, 0.0]})
    res = pr.pilot_report(df, "req", "cot", forced="fd", capability=cap)
    wb = res["who_benefits"]
    assert wb["capability_missing"] == ["m0"] and wb["capability_not_in_scores"] == ["extra"]
    assert any("no capability score for ['m0']" in w for w in res["warnings"])
    E = np.tile([1.0, 3.0, 2.0, 7.0, 5.0], (4, 1))
    s, _ = rb.cross_slope(E, (P - B)[:, 1:])
    assert same_float(wb["baselines"][0]["slope_on_capability"]["slope"], s)
    assert wb["baselines"][0]["slope_on_capability"]["models"] == 5
    flat = pd.DataFrame({"model": models, "capability": 3.0})
    e = pr.pilot_report(df, "req", "cot", forced="fd", capability=flat)["who_benefits"]["baselines"][0]
    assert not e["slope_on_capability"]["defined"] and e["slope_on_capability"]["V_x"] == 0.0
    one = pd.DataFrame({"model": ["m0"], "capability": [1.0]})
    e = pr.pilot_report(df, "req", "cot", forced="fd", capability=one)["who_benefits"]["baselines"][0]
    assert "fewer than two models" in e["slope_on_capability"]["reason"]
    with pytest.raises(ValueError, match="repeats"):
        pr.pilot_report(df, "req", "cot", capability=pd.concat([cap, cap.iloc[:1]]))
    # with one baseline, the capability score alone adds the block for that baseline
    r1 = pr.pilot_report(df, "req", "cot", capability=cap)
    assert [e["role"] for e in r1["who_benefits"]["baselines"]] == ["baseline"]
    assert "opposite_orderings" not in r1["who_benefits"] and "forced" not in r1


def test_default_report_has_the_keys_of_1_0_0():
    rng = np.random.default_rng(1)
    B = rng.normal(50, 10, (4, 3))
    res = pr.pilot_report(long_df(B, B + 3), "base", "treat")
    keys_100 = {"tool", "version", "baseline", "intervention", "M", "K", "models", "draws", "targets", "benefit",
                "accuracy_baseline", "accuracy_intervention", "unresolved", "unresolved_reason", "per_model",
                "comparison_arm_check", "warnings", "inputs"}
    assert set(res) == keys_100 and res["version"] == "1.1.0"
    assert set(res["per_model"][0]) == {"model", "baseline_mean", "intervention_mean", "gain_mean",
                                        "gain_sd_across_draws"}
    assert set(res["inputs"]) == {"scores", "tokens", "token_budget", "max_over_budget"}


# --------------------------------------------------------------------------- Example 3


def example3(decision=True):
    ex = EXAMPLES
    return pr.pilot_report(os.path.join(ex, "gsm8k29_two_baselines_scores.csv"), "direct_answer", "zero_shot_cot",
                           tokens=os.path.join(ex, "gsm8k29_tokens.csv"), forced="forced_direct",
                           capability=os.path.join(ex, "gsm8k29_capability.csv"),
                           items=os.path.join(ex, "gsm8k29_pilot_items.csv"), decision=decision)


def test_example3_matches_repository_and_saved_pilots():
    res = example3()
    df = pd.read_csv(os.path.join(EXAMPLES, "gsm8k29_two_baselines_scores.csv"), float_precision="round_trip")
    models = list(pd.unique(df["model"]))
    A = {a: df[df["arm"] == a].pivot(index="draw", columns="model", values="score").loc[[1, 2, 3, 4], models]
         .to_numpy(float) for a in ("direct_answer", "forced_direct", "zero_shot_cot")}
    B, F, P = A["direct_answer"], A["forced_direct"], A["zero_shot_cot"]
    assert res["M"] == 29 and res["K"] == 4 and res["models"] == models
    check_report(res, B, P)
    check_block(res["benefit_forced"], P - F)
    check_block(res["accuracy_forced"], F)
    # Example 2's file holds the same requested and chain-of-thought scores
    B2, P2, models2, _ = example_arrays("gsm8k29_scores.csv", "direct_answer", "zero_shot_cot")
    assert models2 == models and np.array_equal(B2, B) and np.array_equal(P2, P)
    with open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8") as fh:
        units = json.load(fh)["discovery_config_ids"][:4]
    assert df.drop_duplicates("draw")["config_id"].tolist() == units
    # who benefits
    cap = pr.read_csv(os.path.join(EXAMPLES, "gsm8k29_capability.csv"))
    assert cap["model"].tolist() == models
    E = np.tile(cap["capability"].to_numpy(float), (4, 1))
    n = dc.third_size(29)
    for e, X in zip(res["who_benefits"]["baselines"], (B, F)):
        s, vx = rb.cross_slope(X, P - X)
        assert same_float(e["slope_on_accuracy"]["slope"], s) and same_float(e["slope_on_accuracy"]["V_x"], vx)
        assert same_float(e["slope_on_capability"]["slope"], rb.cross_slope(E, P - X)[0])
        assert np.array_equal(e["top_third"]["weights"], dc.choose((P - X).mean(0), n)[0])
    ref = rb.pair_reversals((P - B).mean(0), (P - F).mean(0))
    for k in ("pairs", "reversed", "ties", "by_threshold"):
        assert res["who_benefits"]["opposite_orderings"][k] == ref[k]
    # w: the comparison-arm check's share of direct_answer outputs over 16 tokens
    share = {r["model"]: r["share_over_budget"] for r in res["comparison_arm_check"]["models"]}
    assert all(p["w"] == share[p["model"]] for p in res["per_model"])
    # split pilot: the items are those of the first saved 100-item pilot on each draw, and the single-baseline gains
    # reproduce that file's saved own-gain predictions
    z = np.load(os.path.join(ROOT, "manifests", "predictions_decision_strict.npz"))
    pos, its = z["gsm8k__disjoint__positions"].astype(int), z["gsm8k__disjoint__items"].astype(int)
    first = [int(np.flatnonzero(pos == k)[0]) for k in range(4)]
    it = pd.read_csv(os.path.join(EXAMPLES, "gsm8k29_pilot_items.csv"))
    S = {}
    for k, p in enumerate(first):
        sub = it[it["draw"] == k + 1]
        assert sub[sub["model"] == models[0]]["item"].astype(int).tolist() == its[p].tolist()
        for a in ("direct_answer", "forced_direct", "zero_shot_cot"):
            S.setdefault(a, []).append(sub.pivot(index="model", columns="item", values=a)
                                       .loc[models, its[p]].to_numpy())
    S = {a: np.stack(v) for a, v in S.items()}               # (4 draws, M, 100 items)
    rows, items = np.arange(4), np.tile(np.arange(100), (4, 1))
    for role, a, key in (("requested", "direct_answer", "requested"), ("forced", "forced_direct", "forced")):
        single = sp.single_gains(S[a], S["zero_shot_cot"], rows, items)
        assert np.allclose(single, z[f"gsm8k__disjoint__{key}__own_gain__pred"][first], atol=1e-12, rtol=0)
        assert np.allclose([p[f"gain_{role}_items"] for p in res["split_pilot"]["per_model"]], single.mean(0),
                           atol=TOL, rtol=0)
    rq = sp.assignment(4, 100, sp.SEED, 1)
    study = sp.split_gains(S["direct_answer"], S["forced_direct"], S["zero_shot_cot"], rows, items, rq)
    spl = res["split_pilot"]
    assert np.allclose(spl["per_draw_split_gain"], study, atol=TOL, rtol=0)
    assert np.array_equal(spl["weights"], dc.choose(study.mean(0), n)[0])
    assert spl["seed"] == sp.SEED == pr.SPLIT_SEED
    assert spl["requested_items_per_draw"] == {str(d): 50 for d in range(1, 5)}
    # decision
    for e, X in zip(res["decision"]["baselines"], (B, F)):
        own, capr = split_half_by_hand(X, P - X, n)
        sh = e["split_half_regret"]
        assert sh["n_choices"] == 6 and same_float(sh["own_gain"]["mean"], float(np.mean(own)))
        assert same_float(sh["capability"]["mean"], float(np.mean(capr)))
        assert np.array_equal(e["capability"]["weights"], dc.choose(dc.capability_cross(X, P - X)[0], n)[0])


# --------------------------------------------------------------------------- the README


def readme_outputs(name):
    """{command: printed output} of the README's examples: a command block followed by an output block."""
    with open(os.path.join(RELEASE, name), encoding="utf-8") as fh:
        text = fh.read()
    blocks = text.split("```")[1::2]
    out = {}
    for cmd, printed in zip(blocks, blocks[1:]):
        cmd = cmd.strip()
        if cmd.startswith("python pilot_report.py examples/") and printed.lstrip().startswith("Pilot report"):
            out[cmd] = printed.strip("\n")
    return out


@pytest.mark.parametrize("name", ["README.md"])
def test_readme_outputs_equal_the_tool(name, capsys):
    outs = readme_outputs(name)
    assert len(outs) == 3
    for cmd, printed in outs.items():
        argv = [os.path.join(RELEASE, a) if a.startswith("examples/") else a for a in cmd.split()[2:]]
        assert pr.main(argv) == 0
        got = capsys.readouterr().out.rstrip("\n")
        assert got == printed, cmd


def test_capability_example_carries_the_epoch_credit():
    with open(os.path.join(EXAMPLES, "gsm8k29_capability.csv"), encoding="utf-8") as fh:
        first = fh.readline().rstrip("\n")
    assert first == "# Epoch Capabilities Index, Epoch AI, CC BY 4.0, retrieved 30 October 2026, https://epoch.ai/eci"
    cap = pr.read_csv(os.path.join(EXAMPLES, "gsm8k29_capability.csv"))
    assert list(cap.columns) == ["model", "capability"] and len(cap) == 29


# --------------------------------------------------------------------------- the command line, 1.1.0


def test_cli_new_options(tmp_path, capsys):
    rng = np.random.default_rng(12)
    B, F, P = random_three(rng, K=4, M=6)
    models = [f"m{m}" for m in range(6)]
    three_df(B, F, P, models).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"model": models, "capability": rng.normal(0, 1, 6)}).to_csv(tmp_path / "c.csv", index=False)
    S, rows, items, _ = random_pilot(rng, P_=4, n=10, M=6)
    to_wide(item_long(S, rows, items, models)).to_csv(tmp_path / "i.csv", index=False)
    pd.DataFrame({"model": models * 3, "arm": "req", "n_output_tokens": [5] * 6 + [20] * 6 + [3] * 6}).to_csv(
        tmp_path / "t.csv", index=False)
    out = tmp_path / "r.json"
    code = pr.main([str(tmp_path / "s.csv"), "--baseline", "req", "--forced", "fd", "--intervention", "cot",
                    "--tokens", str(tmp_path / "t.csv"), "--capability", str(tmp_path / "c.csv"),
                    "--items", str(tmp_path / "i.csv"), "--seed", "5", "--decision", "--json", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    for s in ("Who benefits", "Split pilot", "Decision:", "w, mean over the 6 models"):
        assert s in printed
    j = json.loads(out.read_text(encoding="utf-8"))
    assert j["forced"] == "fd" and j["split_pilot"]["seed"] == 5 and j["inputs"]["decision"] is True
    assert all(abs(p["w"] - 1 / 3) < 1e-15 for p in j["per_model"]) and abs(j["w"]["mean"] - 1 / 3) < 1e-15
    rq = sp.assignment(4, 10, 5, 1)
    assert sum(c["baseline"] == "requested" for c in j["split_pilot"]["assignment_cells"]) == int(rq.sum())
    assert pr.main([str(tmp_path / "s.csv"), "--baseline", "req", "--intervention", "cot",
                    "--items", str(tmp_path / "i.csv")]) == 2
    assert "needs the forced baseline" in capsys.readouterr().err
    # as a script, in an isolated interpreter started outside the repository
    proc = subprocess.run([sys.executable, "-I", os.path.join(RELEASE, "pilot_report.py"), "s.csv", "--baseline",
                           "req", "--forced", "fd", "--intervention", "cot", "--items", "i.csv", "--decision"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Split pilot" in proc.stdout
