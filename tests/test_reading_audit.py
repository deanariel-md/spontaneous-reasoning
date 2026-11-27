"""Tests of scripts/reading_audit.py: the design, the judge on a provider's message batches
API (fake HTTP, no network), the two-rater human sheets, the analysis on synthetic labels (with the mixture check, the
pairwise agreement and the repeat set), and the macros module."""
import csv
import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reading_audit as ra  # noqa: E402
from paper import macros_reading_audit as mra  # noqa: E402

KEY = "sk-ant-api03-SECRETSECRETSECRET-abc"


# ------------------------------------------------------------------------------------------------ design


def test_bins():
    assert [ra.bin_of(n) for n in (1, 16, 17, 64, 65, 1024)] == ["1-16", "1-16", "17-64", "17-64", "65+", "65+"]


def test_allocate_equal_and_redistributed():
    t = {b: 7 for b in ra.BINS}
    assert ra.allocate({"1-16": 100, "17-64": 100, "65+": 100}, t) == {"1-16": 7, "17-64": 7, "65+": 7}
    a = ra.allocate({"1-16": 2, "17-64": 100, "65+": 0}, t)
    assert a == {"1-16": 2, "17-64": 19, "65+": 0}
    a = ra.allocate({"1-16": 2, "17-64": 9, "65+": 100}, t)
    assert a == {"1-16": 2, "17-64": 9, "65+": 10} and sum(a.values()) == 21
    assert ra.allocate({"1-16": 2, "17-64": 3}, t) == {"1-16": 2, "17-64": 3, "65+": 0}


def _population(n_models=5, items=60, seed=0):
    """Synthetic panel rows: four configurations (c1 drawn twice), model-specific lengths, scores of three arms."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_models):
        for c in ("c1", "c2", "c3", "c4"):
            for i in range(items):
                p_long = 0.1 + 0.18 * j
                n = int(rng.integers(17, 300)) if rng.random() < p_long else int(rng.integers(1, 17))
                rows.append({"model_key": f"m{j}", "config_id": c, "item_id": str(i), "n_tokens": n,
                             "requested_strict": int(rng.random() < 0.3 + 0.1 * j),
                             "forced_strict": int(rng.random() < 0.5),
                             "cot_strict": int(rng.random() < 0.6 + 0.05 * j),
                             "requested_content": int(rng.random() < 0.35 + 0.1 * j),
                             "forced_content": int(rng.random() < 0.5),
                             "cot_content": int(rng.random() < 0.65 + 0.05 * j)})
    df = pd.DataFrame(rows)
    disc = ["c1", "c1", "c2", "c3", "c4"]
    df["mult"] = df["config_id"].map(ra.draw_counts(disc))
    return df, [f"m{j}" for j in range(n_models)], disc


def test_stratify_is_seeded_and_order_free():
    pop, _, _ = _population()
    pop = pop.assign(task="gsm8k")
    a, st = ra.stratify(pop, lambda t, m: {b: 7 for b in ra.BINS}, 123, "panel")
    b, _ = ra.stratify(pop.sample(frac=1, random_state=4), lambda t, m: {b: 7 for b in ra.BINS}, 123, "panel")
    key = ["model_key", "config_id", "item_id"]
    assert a[key].sort_values(key).values.tolist() == b[key].sort_values(key).values.tolist()
    c, _ = ra.stratify(pop, lambda t, m: {b: 7 for b in ra.BINS}, 124, "panel")
    assert a[key].sort_values(key).values.tolist() != c[key].sort_values(key).values.tolist()
    for r in st.itertuples():
        g = pop[(pop["model_key"] == r.model_key) & (pop["n_tokens"].map(ra.bin_of) == r.bin)]
        assert r.N == len(g) and r.M == g["mult"].sum()
    assert (a.groupby("model_key").size() == 21).all()
    assert np.allclose(a["p_incl"], a["n_h"] / a["N_h"])


def test_weights_sum_to_stratum_size():
    codes = np.array([0, 0, 1, 1, 1])
    mult = np.array([2.0, 1.0, 1.0, 1.0, 2.0])
    M = np.array([30.0, 50.0])
    W = ra.weights(codes, mult, M)
    assert np.isclose(W[:2].sum(), 30) and np.isclose(W[2:].sum(), 50)
    assert np.isclose(W[0] / W[1], 2.0)
    Wb = ra.weights(codes, mult, M, counts=np.array([0, 2, 1, 0, 2]))
    assert Wb[0] == 0 and np.isclose(Wb[:2].sum(), 30) and np.isclose(Wb[2:].sum(), 50)


def test_boot_counts_stay_in_strata():
    codes = np.array([0, 1, 0, 2, 1, 0])
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = ra.boot_counts(codes, rng)
        assert [c[codes == k].sum() for k in range(3)] == [3, 2, 1]


def test_model_summaries_cot_minus_forced():
    pop, models, disc = _population()
    s = ra.model_summaries(pop.rename(columns={"n_tokens": "n_output_tokens"}), models, disc, 16)
    g = pop[pop["model_key"] == "m1"]
    per_cfg = g.assign(d=g["cot_strict"] - g["forced_strict"]).groupby("config_id")["d"].mean()
    assert s["cot_minus_forced_strict"]["m1"] == pytest.approx(100 * np.mean([per_cfg[c] for c in disc]))


# ------------------------------------------------------------------------------------------------ analysis


def _synthetic(label_rule, targets=7, tasks=("gsm8k", "hotpot")):
    """A sample (with scores), labels and population block for the given tasks, labels from label_rule(row)."""
    sample, popblk = [], {}
    for k, task in enumerate(tasks):
        pop, models, disc = _population(seed=k)
        wide = pop.rename(columns={"n_tokens": "n_output_tokens"})
        summ = ra.model_summaries(wide, models, disc, ra.CAP[task])
        popblk[task] = {"models": models, "cap": ra.CAP[task], "per_model": summ}
        s, _ = ra.stratify(pop.assign(task=task), lambda t, m: {b: targets for b in ra.BINS}, 7, "panel")
        s["sample_id"] = [f"{task}|{m}|{c}|{i}" for m, c, i in zip(s["model_key"], s["config_id"], s["item_id"])]
        sample.append(s)
    q, _, _ = _population(n_models=1, seed=9)
    q = q.assign(task="gsm8k", model_key=ra.QWEN, mult=1).drop(columns=[f"{a}_{s}" for a in ra.ARMS3
                                                                         for s in ra.SCORINGS])
    qs, _ = ra.stratify(q, lambda t, m: ra.QWEN_ALLOC, 7, "qwen")
    qs["sample_id"] = [f"q|{c}|{i}" for c, i in zip(qs["config_id"], qs["item_id"])]
    sample.append(qs)
    s = pd.concat(sample, ignore_index=True)
    labels = pd.DataFrame({"sample_id": s["sample_id"], "label": [label_rule(r) for r in s.itertuples()]})
    return s, labels, popblk


def test_labels_equal_to_length_give_exact_w():
    s, labels, pop = _synthetic(lambda r: "R" if r.n_tokens > 16 else "A")
    R = ra.analyse(s, labels, pop, n_boot=30)
    g = R["panel"]["gsm8k"]["estimate"]
    assert g["precision_16"] == 1.0 and g["recall_16"] == 1.0 and g["r_short_16"] == 0.0
    for row in R["panel"]["gsm8k"]["per_model"]:
        assert abs(row["r_share"] - row["w16"]) < 1e-12
    assert abs(g["corr_w_pearson"] - 1.0) < 1e-12              # GSM8K's w is at 16 tokens
    assert abs(g["r2_rshare_strict"] - g["r2_w_strict"]) < 1e-12
    assert abs(g["mix_rms_r_strict"] - g["mix_rms_w_strict"]) < 1e-9   # r = w for every model: the same predictions
    h = R["panel"]["hotpot"]["estimate"]
    assert abs(h["corr_w16_pearson"] - 1.0) < 1e-12            # HotpotQA's w is at 64 tokens
    assert h["r_short_64"] > 0 and h["precision_64"] == 1.0
    qq = R["qwen"]["gsm8k"]["estimate"]
    assert qq["r_long_16"] == 1.0 and R["qwen"]["pooled"]["n_labelled"] == 55
    assert "acc_requested_strict_R" not in qq                   # no scores for Qwen4-Max in the key


def test_draw_multiplicity_counts_twice():
    # every row sampled (a census inside each stratum): R exactly on configuration c1, drawn twice of five draws
    s, labels, pop = _synthetic(lambda r: "R" if r.config_id == "c1" else "P", targets=10 ** 6, tasks=("gsm8k",))
    R = ra.analyse(s, labels, pop, n_boot=5)
    for row in R["panel"]["gsm8k"]["per_model"]:
        assert abs(row["r_share"] - 2 / 5) < 1e-12
    assert abs(R["panel"]["gsm8k"]["estimate"]["r_all"] - 0.4) < 1e-12


def test_hand_computed_stratified_ratio():
    df = pd.DataFrame({"stratum": ["a", "a", "b"], "M_h": [10.0, 10.0, 30.0], "mult": [1, 2, 1],
                       "label": ["R", "A", "R"], "n_tokens": [5, 5, 80], "bin": ["1-16", "1-16", "65+"],
                       "model_key": ["m", "m", "m"]})
    codes, mult, M = ra.design(df)
    W = ra.weights(codes, mult, M)
    est = ra.task_estimates(df, W, ["m"], None)
    # stratum a: mult-weighted R share 1/3 of 10; stratum b: 1 of 30 -> (10/3 + 30) / 40
    assert abs(est["r_all"] - (10 / 3 + 30) / 40) < 1e-12
    assert abs(est["r_short_16"] - 1 / 3) < 1e-12 and est["r_long_16"] == 1.0


def test_mixture_on_the_same_items_hand_computed():
    # stratum a (M 10): rows R (w 10/3) and A (w 20/3); stratum b (M 30): one R row
    df = pd.DataFrame({"stratum": ["a", "a", "b"], "M_h": [10.0, 10.0, 30.0], "mult": [1, 2, 1],
                       "label": ["R", "A", "R"], "n_tokens": [5, 5, 80], "bin": ["1-16", "1-16", "65+"],
                       "model_key": ["m", "m", "m"],
                       "requested_strict": [1, 0, 0], "cot_strict": [1, 1, 1], "forced_strict": [0, 0, 1],
                       "requested_content": [1, 1, 0], "cot_content": [1, 1, 0], "forced_content": [0, 1, 0]})
    codes, mult, M = ra.design(df)
    e = ra.task_estimates(df, ra.weights(codes, mult, M), ["m"], None)
    wR = np.array([10 / 3, 30.0])                                # the two R rows
    assert e["acc_requested_strict_R"] == pytest.approx((10 / 3) / wR.sum())
    assert e["acc_cot_strict_R"] == pytest.approx(1.0)
    assert e["acc_forced_strict_R"] == pytest.approx(30 / wR.sum())
    assert e["gap_req_cot_strict_R_pp"] == pytest.approx(100 * ((10 / 3) / wR.sum() - 1.0))
    assert e["acc_requested_strict_notR"] == 0.0 and e["acc_forced_strict_notR"] == 0.0
    assert e["gap_req_forced_strict_notR_pp"] == 0.0
    assert e["acc_requested_content_notR"] == 1.0 and e["gap_req_forced_content_notR_pp"] == 0.0


def test_mixture_across_models_residuals():
    # a census of one stratum per model: model a labelled R on 1 of 4 rows (r = 0.25), model b on 3 of 4 (r = 0.75)
    rows = []
    for m, nr in (("a", 1), ("b", 3), ("c", 2)):
        for i in range(4):
            rows.append({"stratum": m, "M_h": 4.0, "mult": 1, "model_key": m, "label": "R" if i < nr else "P",
                         "n_tokens": 5, "bin": "1-16"})
    df = pd.DataFrame(rows)
    r = {"a": 0.25, "b": 0.75, "c": 0.5}
    w = {"a": 0.5, "b": 0.5, "c": 0.5}
    cmf = {"a": 20.0, "b": 10.0, "c": 4.0}
    obs = {m: r[m] * cmf[m] for m in r}                         # exactly r x (CoT - forced)
    pm = {"w": w, "w16": w, "req_minus_forced_strict": obs, "req_minus_forced_content": obs,
          "cot_minus_forced_strict": cmf, "cot_minus_forced_content": cmf}
    codes, mult, M = ra.design(df)
    e = ra.task_estimates(df, ra.weights(codes, mult, M), ["a", "b", "c"], pm)
    assert e["mix_rms_r_strict"] == pytest.approx(0.0, abs=1e-12)
    res_w = np.array([obs[m] - w[m] * cmf[m] for m in ("a", "b", "c")])     # -5, 2.5, 0
    assert e["mix_rms_w_strict"] == pytest.approx(np.sqrt(np.mean(res_w ** 2)))
    assert e["mix_mean_res_w_strict"] == pytest.approx(res_w.mean())
    assert e["mix_rms_diff_strict"] == pytest.approx(-np.sqrt(np.mean(res_w ** 2)))
    assert np.allclose(e["mix_res_r_strict_by_model"], 0.0)


def test_unread_and_not_judged_are_counted_apart():
    s, labels, pop = _synthetic(lambda r: "R" if r.n_tokens > 64 else "P", tasks=("gsm8k",))
    labels.loc[labels.index[:3], "label"] = None          # judged, not read as one letter
    labels = labels.iloc[5:].reset_index(drop=True)       # five rows never judged (two of them were unread)
    R = ra.analyse(s, labels, pop, n_boot=5)
    assert R["n_not_judged"] == 5 and R["n_unread"] == 0
    labels.loc[labels.index[:2], "label"] = None
    R = ra.analyse(s, labels, pop, n_boot=5)
    assert R["n_unread"] == 2 and R["n_labelled"] == len(s) - 7


def test_agreement_unweighted_and_weighted():
    k = ra.agreement(list("RRPPAA"), list("RRPPAA"))
    assert k["agreement"] == 1.0 and k["kappa"] == 1.0 and k["precision_R"] == 1.0 and k["recall_R"] == 1.0
    k = ra.agreement(list("RRRRPPPP"), list("RRPPRRPP"))
    assert k["agreement"] == 0.5 and abs(k["kappa"]) < 1e-12
    k = ra.agreement(list("RRR"), list("RRR"))
    assert k["kappa"] is None                                   # expected agreement 1: kappa undefined, reported so
    k = ra.agreement(list("RPAR"), list("RPPR"))
    po, pe = 0.75, (2 / 4) * (2 / 4) + (1 / 4) * (2 / 4) + (1 / 4) * 0
    assert abs(k["kappa"] - (po - pe) / (1 - pe)) < 1e-12
    assert k["confusion"]["A"]["P"] == 1 and k["confusion"]["R"]["R"] == 2
    # weighted: the disagreeing row (A vs P) weighs 3 of 6
    kw = ra.agreement(list("RPAR"), list("RPPR"), [1, 1, 3, 1])
    assert kw["agreement"] == pytest.approx(0.5)
    pe_w = (2 / 6) * (2 / 6) + (1 / 6) * (4 / 6) + (3 / 6) * 0
    assert kw["kappa"] == pytest.approx((0.5 - pe_w) / (1 - pe_w))
    assert sum(v for row in kw["confusion"].values() for v in row.values()) == pytest.approx(1.0)
    # R against not R: first list R on 3 rows, second R on 2 of them and on one more
    kb = ra.agreement(list("RRRPA"), list("RRPRA"))
    assert kb["precision_R"] == pytest.approx(2 / 3) and kb["recall_R"] == pytest.approx(2 / 3)
    assert kb["agreement_R"] == pytest.approx(3 / 5)


def _human_frame(n=12):
    rng = np.random.default_rng(1)
    lab = list(rng.choice(list(ra.CLASSES), n))
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "task": ["gsm8k"] * n,
                         "design_weight": rng.uniform(1, 5, n), "rater1": lab,
                         "rater2": [x if i % 4 else "R" for i, x in enumerate(lab)],
                         "unsure1": [i == 0 for i in range(n)], "unsure2": [False] * n}), lab


def test_human_and_repeat_blocks():
    h, lab = _human_frame()
    h.loc[3, "rater2"] = None
    judge = pd.DataFrame({"sample_id": h["sample_id"], "label": lab})
    judge.loc[5, "label"] = None                                # unread by the judge
    b = ra.human_block(h, judge)
    assert b["pairs"]["judge_rater1"]["n"] == 11 and b["pairs"]["judge_rater1"]["unweighted"]["agreement"] == 1.0
    assert b["pairs"]["rater1_rater2"]["n"] == 11 and b["pairs"]["judge_rater2"]["n"] == 10
    assert b["unsure_rater1"] == 1 and b["labelled_rater2"] == 11
    rep = h[["sample_id", "task", "design_weight"]]
    rb = ra.repeat_block(rep, judge, pd.DataFrame({"sample_id": h["sample_id"][:6], "label": lab[:6]}))
    assert rb["n"] == 5 and rb["n_set"] == 12 and rb["repeat_judged"] == 6
    assert rb["unweighted"]["agreement"] == 1.0


def test_kappas_match_agreement():
    rng = np.random.default_rng(3)
    a, b = list(rng.choice(list(ra.CLASSES), 40)), list(rng.choice(list(ra.CLASSES), 40))
    w = rng.uniform(0.5, 3, 40)
    idx = {c: i for i, c in enumerate(ra.CLASSES)}
    ia, ib = np.array([idx[x] for x in a]), np.array([idx[x] for x in b])
    for ww in (None, w):
        ref = ra.agreement(a, b, ww)
        k, kR = ra._kappas(ia, ib, np.ones(40) if ww is None else ww)
        assert k == pytest.approx(ref["kappa"]) and kR == pytest.approx(ref["kappa_R"])
    ia2 = ia.copy()
    ia2[:5] = -1                                                  # unlabelled rows drop out
    k, _ = ra._kappas(ia2, ib, np.ones(40))
    assert k == pytest.approx(ra.agreement(a[5:], b[5:])["kappa"])
    assert ra._kappas(np.zeros(4, int), np.zeros(4, int), np.ones(4)) == (None, None)   # one class: undefined


def test_human_boot_intervals_and_gaps():
    rng = np.random.default_rng(7)
    n = 150
    truth = rng.choice(list(ra.CLASSES), n)

    def noisy(p):
        return [x if rng.random() > p else rng.choice(list(ra.CLASSES)) for x in truth]
    h = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "task": ["gsm8k"] * n,
                      "cell": [f"c{i % 12}" for i in range(n)], "design_weight": rng.uniform(1, 5, n),
                      "rater1": noisy(0.2), "rater2": noisy(0.2), "unsure1": [False] * n, "unsure2": [False] * n})
    judge = pd.DataFrame({"sample_id": h["sample_id"], "label": noisy(0.6)})
    b = ra.human_block(h, judge, n_boot=300, seed=1)
    ci = b["boot"]["ci95"]
    for p in ra.PAIRS:
        lo, hi = ci["weighted"]["kappa"][p]["ci"]
        assert lo <= b["pairs"][p]["weighted"]["kappa"] <= hi and hi - lo < 0.5
    g = b["gaps"]["unweighted"]["gap_kappa"]
    assert g["judge_rater1"] == pytest.approx(b["pairs"]["judge_rater1"]["unweighted"]["kappa"]
                                              - b["pairs"]["rater1_rater2"]["unweighted"]["kappa"])
    assert g["judge_mean"] == pytest.approx((g["judge_rater1"] + g["judge_rater2"]) / 2)
    assert g["judge_mean"] < 0 and ci["unweighted"]["gap_kappa"]["judge_mean"]["ci"][1] < 0   # noisier judge shows
    assert b == ra.human_block(h, judge, n_boot=300, seed=1)                                  # seeded
    assert b["boot"]["strata"] == "cell"
    h1 = h.assign(rater2=None)                                    # one rater still out: no gaps, no bootstrap
    assert "boot" not in ra.human_block(h1, judge, n_boot=50)


def test_macros_table_summary_and_module(tmp_path):
    s, labels, pop = _synthetic(lambda r: "R" if r.n_tokens > 40 else ("P" if r.n_tokens > 10 else "A"))
    h, lab = _human_frame()
    h["sample_id"] = list(s["sample_id"][:12])
    rep = s[["sample_id", "task"]].iloc[:20].assign(design_weight=1.0)
    rep_labels = labels.iloc[:20].copy()
    flip = {"R": "A", "P": "A", "A": "P"}
    rep_labels.loc[rep_labels.index[:2], "label"] = rep_labels["label"][:2].map(flip)
    R = ra.analyse(s, labels, pop, human=h, n_boot=40, repeat=(rep, rep_labels))
    R["cost"] = {"billed_usd": 7.123, "pilot_billed_usd": 0.15, "batch_billed_usd": 6.973}
    m = ra.macros(R, {"judge_label": "Judge model", "judge_id": "claude-opus-5", "seed": 1})
    assert all(k.startswith("lab") and k.isalpha() for k in m)
    for k in ("labRLongGsmPct", "labRShortGsmPct", "labPrecGsmPct", "labRecallGsmPct", "labCorrWGsm", "labRsqRGsm",
              "labRsqWGsm", "labRsqRGsmContent", "labRLongHqSixtyFourPct", "labQwenRLongGsmPct", "labQwenRLongPct",
              "labAccReqRGsmPct", "labAccCotRGsmPct", "labAccForcedNotRGsmContentPct", "labGapReqCotRGsm",
              "labGapReqCotRGsmLo", "labGapReqForcedNotRHqContent", "labMixRmsRGsm", "labMixRmsWHqContent",
              "labMixRmsDiffGsmLo", "labMixCorrRGsm", "labHumanN", "labNJudgeRaterOne", "labAgreeJudgeRaterOnePct",
              "labKappaJudgeRaterTwo", "labKappaRaterOneRaterTwoUnw", "labPrecRJudgeRaterOnePct",
              "labRecallRJudgeRaterTwoPct", "labSelfN", "labSelfAgreePct", "labSelfKappa", "labCostUsd",
              "labJudge", "labJudgeEffort", "labNNotJudged", "labKappaJudgeRaterOneLo", "labKappaJudgeRaterOneHi",
              "labKappaRaterOneRaterTwoUnwLo", "labKappaRJudgeRaterTwoHi", "labKappaGapJudgeMean",
              "labKappaGapJudgeRaterOneUnwLo", "labKappaRGapJudgeRaterTwo"):
        assert k in m, k
    assert m["labHumanN"] == "12" and m["labCostUsd"] == "7.12" and m["labJudge"] == "Judge model"
    assert m["labSelfN"] == "20" and m["labSelfAgreePct"] == "90"
    assert not any(re.search(r"(?<!\$)-", v) for v in m.values() if v not in ("--",) and "claude" not in v)
    tex = ra.si_table(R)
    assert tex.count("\\\\") >= 5 and "HotpotQA" in tex and "Qwen4-Max" in tex and "not R" in tex
    assert all(line.count("&") == 10 for line in tex.splitlines() if line.endswith("\\\\"))
    md = ra.summary_md(R, {"judge_label": "Judge model", "judge_id": "claude-opus-5", "seed": 1})
    assert "Cohen's kappa" in md and "temperature 0" not in md and "default sampling" in md
    assert "Judge vs rater 2" in md and "Sample run vs repeat" in md and "r x (CoT minus forced)" in md
    assert ra.clean_md(md) == md
    assert mra.reading_audit_results(str(tmp_path)) is None
    p = tmp_path / "results" / "reading_audit"
    p.mkdir(parents=True)
    (p / "reading_audit.json").write_text(json.dumps({"macros": m}), encoding="utf-8")
    got = mra.reading_audit_macros(mra.reading_audit_results(str(tmp_path)))
    assert got == {k: str(v) for k, v in m.items()}
    with pytest.raises(ValueError):
        mra.reading_audit_macros({"macros": {"labBad1": "x"}})
    assert mra.reading_audit_macros(None) == {}


def test_pairs_without_labels_give_no_macros():
    s, labels, pop = _synthetic(lambda r: "R" if r.n_tokens > 16 else "A", tasks=("gsm8k",))
    h, lab = _human_frame()
    h["sample_id"] = list(s["sample_id"][:12])
    h["rater2"] = None
    R = ra.analyse(s, labels, pop, human=h, n_boot=5)
    m = ra.macros(R, {"judge_label": "Judge model", "judge_id": "claude-opus-5", "seed": 1})
    assert "labAgreeJudgeRaterOnePct" in m and not any("RaterTwo" in k for k in m if k != "labUnsureRaterTwo")
    md = ra.summary_md(R, {"judge_label": "Judge model", "judge_id": "claude-opus-5", "seed": 1})
    assert "Judge vs rater 2: no rows with both labels yet." in md and "--%" not in md


def test_no_temperature_wording_in_outputs():
    src = (ROOT / "scripts" / "reading_audit.py").read_text(encoding="utf-8")
    assert "temperature 0" not in src and '"temperature"' not in src and "TEMPERATURE" not in src


# ------------------------------------------------------------------------------------------------ the judge


def test_rubric_holds_the_request_definitions():
    system, template = ra.rubric_parts()
    for s in ("at least one step toward the answer that the question does not state, such as a calculation, a "
              "deduction, or facts combined toward the answer",
              "a restatement of the question, a preamble or politeness, a format remark or an explanation after the "
              "answer with no step", "A (answer only)", "R (intermediate reasoning)",
              "P (no reasoning but extra text)", "exactly one letter"):
        assert s in system, s
    assert "{question}" in template and "{reply}" in template


def test_parse_label_robust():
    ok = {"R": "R", " P\n": "P", "**A**": "A", "A.": "A", "R.": "R", "(R)": "R", "`P`": "P", "**P**": "P",
          "<thinking></thinking>R": "R", "<label>P</label>": "P", "<answer>\nA\n</answer>": "A", "**R**.": "R",
          "</s>R": "R"}
    for t, want in ok.items():
        assert ra.parse_label(t) == want, t
    for t in ("RP", "Answer: R", "r", "", None, "R because", "<thinking>the reply has a step</thinking>R",
              "R or P", "Class: A", "PA."):
        assert ra.parse_label(t) is None, t
    # a reply that reasons aloud first is read only from a last line holding the label alone
    assert ra.parse_label("Wait, the reply computes 4 x 3 = 12.\n\nR") == "R"
    assert ra.parse_label("The reply only restates the question.\n**P**") == "P"
    for t in ("Wait — the answer is 2, but we don't jud", "Looks like R.\nor maybe P", "A\nbut also R?"):
        assert ra.parse_label(t) is None, t


def test_params_blind_no_sampling_no_prefill():
    system, template = ra.rubric_parts()
    row = {"sample_id": "panel|gsm8k|secret-model|x", "task": "gsm8k", "source": "panel",
           "question": "What is {reply} + 1?", "reply": "\\boxed{2} {question}", "model_key": "secret-model"}
    req = ra.make_requests([row], system, template)[0]
    p = req["params"]
    assert set(p) == {"model", "max_tokens", "system", "messages", "thinking", "output_config"}
    assert p["model"] == "claude-opus-5" and p["max_tokens"] == ra.MAX_TOKENS == 256 and p["system"] == system
    assert p["thinking"] == {"type": "disabled"} and p["output_config"] == {"effort": "medium"}
    assert not ({"temperature", "top_p", "top_k"} & set(p))
    assert [m["role"] for m in p["messages"]] == ["user"]           # no prefilled assistant turn
    u = p["messages"][0]["content"]
    assert "What is {reply} + 1?" in u and "\\boxed{2} {question}" in u and "secret-model" not in u
    assert req["worst_batch_usd"] == pytest.approx(ra.worst_usd(req["n_bytes"], None, ra.JUDGE["price_batch"]))
    assert req["worst_standard_usd"] == pytest.approx(2 * req["worst_batch_usd"])


def test_custom_ids_match_the_pattern_and_map_back():
    rows = [{"sample_id": f"panel|{t}|some/model.v1|ps-i01-f02|direct_answer|{i}", "task": t} for t in ra.TASKS
            for i in range(300)]
    ids = [ra.custom_id(r["sample_id"], r["task"], k) for r in rows for k in ("main", "repeat")]
    assert all(re.fullmatch(r"^[a-zA-Z0-9_-]{1,64}$", c) for c in ids) and len(set(ids)) == len(ids)
    assert ra.custom_id("x", "gsm8k") == ra.custom_id("x", "gsm8k") and ra.custom_id("x", "gsm8k").startswith("m_")
    system, template = ra.rubric_parts()
    full = [dict(r, source="panel", question="q", reply="r") for r in rows[:10]]
    df = ra.id_frame(ra.make_requests(full, system, template) + ra.make_requests(full[:3], system, template, "repeat"))
    assert len(df) == 13 and set(df["kind"]) == {"main", "repeat"}
    assert (df.groupby("sample_id")["custom_id"].nunique() <= 2).all()


def test_model_echo():
    for m in ("claude-opus-5", "claude-opus-5-20260815"):
        assert ra.MODEL_ECHO.fullmatch(m)
    for m in ("claude-opus-5-5", "claude-opus-4-8", "claude-opus-5x", None):
        assert not ra.MODEL_ECHO.fullmatch(str(m))


# ---------------------------------------------------------------- a fake Anthropic API


def _msg(text, model="claude-opus-5", content=None, inp=1000, out=1, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": content if content is not None else [{"type": "text", "text": text}],
            "stop_reason": stop, "usage": {"input_tokens": inp, "output_tokens": out,
                                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}


class FakeAnthropic:
    """Answers the calls the script makes: messages, count_tokens, batch create / retrieve / results."""

    def __init__(self, messages=None, polls_before_end=1, results=None, key=KEY, batch_status=200):
        self._key = key
        self.messages = list(messages or [])
        self.polls_before_end = polls_before_end
        self.results = results          # custom_id -> result object; default: every request succeeds with "R"
        self.batch_status = batch_status
        self.calls, self.batches, self.polls = [], {}, {}

    def call(self, method, url, body=None, text=False):
        self.calls.append((method, url, body))
        if method == "POST" and url == "/v1/messages":
            m = self.messages.pop(0) if self.messages else _msg("R")
            return m if "status" in m and "json" in m else {"status": 200, "json": m}
        if method == "POST" and url == "/v1/messages/count_tokens":
            return {"status": 200, "json": {"input_tokens": len(body["messages"][0]["content"]) // 3 + 900}}
        if method == "POST" and url == "/v1/messages/batches":
            if self.batch_status != 200:
                return {"status": self.batch_status, "json": {"type": "error", "error": {"message": f"bad {KEY}"}},
                        "error": f"bad {KEY}"}
            bid = f"msgbatch_{len(self.batches) + 1:03d}"
            self.batches[bid] = [r["custom_id"] for r in body["requests"]]
            assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", r["custom_id"]) for r in body["requests"])
            return {"status": 200, "json": {"id": bid, "type": "message_batch", "processing_status": "in_progress",
                                            "created_at": "t0", "expires_at": "t1"}}
        if method == "GET" and url.startswith("/v1/messages/batches/"):
            bid = url.rsplit("/", 1)[1]
            self.polls[bid] = self.polls.get(bid, 0) + 1
            ended = self.polls[bid] > self.polls_before_end
            return {"status": 200, "json": {
                "id": bid, "processing_status": "ended" if ended else "in_progress",
                "request_counts": {"processing": 0 if ended else len(self.batches[bid])},
                "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results" if ended else None}}
        if method == "GET" and url.endswith("/results"):
            bid = url.split("/")[-2]
            lines = []
            for cid in reversed(self.batches[bid]):                 # any order
                res = (self.results or {}).get(cid, {"type": "succeeded", "message": _msg("R", inp=1200)})
                lines.append(json.dumps({"custom_id": cid, "result": res}))
            return {"status": 200, "text": "\n".join(lines) + "\n", "json": None}
        raise AssertionError(f"unexpected call {method} {url}")


def _rows(n, task_cycle=True):
    return [{"sample_id": f"panel|{ra.TASKS[i % 4] if task_cycle else 'gsm8k'}|m{i % 3}|cfg|{i}",
             "task": ra.TASKS[i % 4] if task_cycle else "gsm8k", "source": "panel", "question": f"question {i}",
             "reply": f"reply {i}"} for i in range(n)]


def _setup(tmp_path, n=24, n_rep=4):
    system, template = ra.rubric_parts()
    meta = ra.judge_meta(system, template)
    rows = _rows(n)
    main = ra.make_requests(rows, system, template)
    rep = ra.make_requests(rows[:n_rep], system, template, "repeat")
    files = ra.Files(tmp_path)
    ra.id_frame(main + rep).to_csv(files.idmap, index=False)
    return meta, main, rep, files


def test_pilot_round_trip_and_checks(tmp_path):
    meta, main, rep, files = _setup(tmp_path)
    rows = ra.pilot_rows(main, set())
    assert len(rows) == 4 * ra.PILOT_PER_TASK and {r["task"] for r in rows} == set(ra.TASKS)
    replies = [{"status": 529, "json": {"type": "error"}, "error": "overloaded"}] + \
              [_msg(x) for x in ("R", "<thinking></thinking>P", "**A**", "R.")]
    http = FakeAnthropic(messages=replies)
    out = ra.run_pilot(http, rows, files, meta, limit=5.0, key=KEY, sleep=lambda s: None)
    assert out["passed"], out["checks"]
    assert out["calls"] == 20 and len([c for c in http.calls if c[1] == "/v1/messages"]) == 21    # one retry
    sent = [c[2] for c in http.calls if c[1] == "/v1/messages"]
    assert all(not ({"temperature", "top_p", "top_k"} & set(b)) for b in sent)
    lines = ra.read_raw(files.raw)
    assert all(x.get("config_sha") == meta["config_sha"] for x in lines)
    st = ra.judge_state(lines, meta["config_sha"])
    assert len(st["succeeded"]) == 20                            # the pilot's labels count as labels
    led = ra.ledger(lines)
    assert led["pilot_billed_usd"] == pytest.approx(20 * (1000 * 5e-06 + 1 * 2.5e-05))
    assert led["counted_usd"] == pytest.approx(led["pilot_billed_usd"])
    for bad in (_msg("R", model="claude-opus-5-5"), _msg("RP"), _msg("", content=[{"type": "thinking", "thinking": ""},
                                                                                  {"type": "text", "text": "R"}]),
                _msg("R", stop="refusal"), {"status": 400, "json": {"type": "error"}, "error": "invalid"}):
        f2 = ra.Files(tmp_path / f"bad{len(list(tmp_path.iterdir()))}")
        f2.raw.parent.mkdir(parents=True, exist_ok=True)
        o = ra.run_pilot(FakeAnthropic(messages=[bad]), rows[:1], f2, meta, limit=5.0, sleep=lambda s: None)
        assert not o["passed"], bad


def test_submit_poll_collect_and_resubmit(tmp_path):
    meta, main, rep, files = _setup(tmp_path)
    idmap = ra.read_idmap(files.idmap)
    todo = ra.todo_requests(main, rep, ra.judge_state([], meta["config_sha"]))
    assert len(todo) == 28 and todo[-1]["kind"] == "repeat"
    fail_cid, err_cid = todo[2]["custom_id"], todo[-1]["custom_id"]
    http = FakeAnthropic(polls_before_end=2, results={
        err_cid: {"type": "errored", "error": {"type": "error", "error": {"type": "api_error", "message": "x"}}},
        fail_cid: {"type": "succeeded", "message": _msg("<thinking>it computes</thinking>R")}})
    out = ra.submit(http, todo, files, meta, limit=20.0, key=KEY)
    assert out["status"] == "submitted" and out["requests"] == 28 and out["worst_basis"] == "bytes"
    batch = json.load(open(files.batches, encoding="utf-8"))
    assert batch["latest_batch_id"] == out["batch_id"] and batch["batches"][0]["n_repeat"] == 4
    st = ra.judge_state(ra.read_raw(files.raw), meta["config_sha"])
    assert len(st["in_flight"]) == 28
    assert ra.ledger(ra.read_raw(files.raw))["reserved_uncollected_usd"] == pytest.approx(out["reserve_usd"])
    # nothing new is sent while the batch runs
    assert ra.todo_requests(main, rep, st) == []
    once = ra.collect(http, files, idmap, meta, once=True, key=KEY, sleep=lambda s: None)
    assert once["pending"] == [out["batch_id"]] and once["batches"][out["batch_id"]]["processing_status"] == "in_progress"
    res = ra.collect(http, files, idmap, meta, poll_seconds=0, key=KEY, sleep=lambda s: None)
    b = res["batches"][out["batch_id"]]
    assert res["pending"] == [] and b["counts"] == {"succeeded": 27, "errored": 1, "canceled": 0, "expired": 0}
    assert b["unread"] == 1 and b["labels"]["R"] == 26
    assert b["cost_usd"] == pytest.approx(26 * (1200 * 2.5e-06 + 1 * 1.25e-05) + (1000 * 2.5e-06 + 1 * 1.25e-05))
    lines = ra.read_raw(files.raw)
    assert all(x["batch_id"] == out["batch_id"] and x["config_sha"] == meta["config_sha"] and x["judge_model"] ==
               "claude-opus-5" and x["rubric_sha256"] == meta["rubric_sha256"] for x in lines if x["event"] == "result")
    led = ra.ledger(lines)
    assert led["reserved_uncollected_usd"] == 0 and led["counted_usd"] == pytest.approx(b["cost_usd"])
    # a second collect adds nothing
    n_lines = len(lines)
    ra.collect(http, files, idmap, meta, key=KEY, sleep=lambda s: None)
    assert len(ra.read_raw(files.raw)) == n_lines
    # labels: the unread reply stays unread (never guessed); the errored repeat goes again with the next submit
    lab = ra.judge_labels(meta["config_sha"], files.raw)
    assert len(lab) == 24 and lab["label"].isna().sum() == 1
    st = ra.judge_state(ra.read_raw(files.raw), meta["config_sha"])
    again = ra.todo_requests(main, rep, st)
    assert [r["custom_id"] for r in again] == [err_cid]
    http.results = None                                          # this time it succeeds
    out2 = ra.submit(http, again, files, meta, limit=20.0, key=KEY)
    assert out2["requests"] == 1 and http.batches[out2["batch_id"]] == [err_cid]
    ra.collect(http, files, idmap, meta, poll_seconds=0, key=KEY, sleep=lambda s: None)
    rl = ra.judge_labels(meta["config_sha"], files.raw, kind="repeat")
    assert len(rl) == 4 and set(rl["label"]) == {"R"}
    assert "SECRET" not in files.raw.read_text(encoding="utf-8")


def test_submit_refuses_over_the_limit_and_split(tmp_path):
    meta, main, rep, files = _setup(tmp_path)
    todo = ra.todo_requests(main, rep, ra.judge_state([], meta["config_sha"]))
    worst = sum(r["worst_batch_usd"] for r in todo)
    http = FakeAnthropic()
    out = ra.submit(http, todo, files, meta, limit=worst * 0.5)
    assert out["status"] == "refused" and http.batches == {} and not files.raw.exists()
    out = ra.submit(http, todo, files, meta, limit=worst * 0.5, split=True)
    assert out["status"] == "submitted" and 0 < out["requests"] < len(todo) and out["reserve_usd"] <= worst * 0.5
    assert out["held_back"] == len(todo) - out["requests"]
    # the reserve of the batch still running counts against the limit
    st = ra.judge_state(ra.read_raw(files.raw), meta["config_sha"])
    rest = ra.todo_requests(main, rep, st)
    assert len(rest) == out["held_back"]
    assert ra.submit(http, rest, files, meta, limit=worst * 0.5)["status"] == "refused"


def test_submit_failure_is_recorded_and_redacted(tmp_path):
    meta, main, rep, files = _setup(tmp_path, n=4, n_rep=0)
    out = ra.submit(FakeAnthropic(batch_status=500), main, files, meta, limit=20.0, key=KEY)
    assert out["status"] == "failed" and "SECRET" not in json.dumps(out)
    text = files.raw.read_text(encoding="utf-8")
    assert "submit_failed" in text and "SECRET" not in text and "batch_submitted" not in text


def test_collect_stops_on_a_refusal(tmp_path):
    meta, main, rep, files = _setup(tmp_path, n=4, n_rep=0)
    http = FakeAnthropic()
    out = ra.submit(http, main, files, meta, limit=20.0)

    class Refusing(FakeAnthropic):
        def call(self, method, url, body=None, text=False):
            return {"status": 401, "json": {"type": "error"}, "error": "invalid x-api-key"}
    res = ra.collect(Refusing(), files, ra.read_idmap(files.idmap), meta, poll_seconds=0, sleep=lambda s: None)
    assert res["pending"] == [] and "401" in res["batches"][out["batch_id"]]["error"]
    assert ra.ledger(ra.read_raw(files.raw))["reserved_uncollected_usd"] == pytest.approx(out["reserve_usd"])


def test_counted_worst_case_from_count_tokens(tmp_path):
    meta, main, rep, files = _setup(tmp_path, n=8, n_rep=2)
    http = FakeAnthropic()
    out = ra.count_tokens(http, main + rep, files.counts, threads=2, sleep=lambda s: None)
    assert out["counted_now"] == 8 and out["failed"] == []          # a repeat shares its row's count
    bodies = [c[2] for c in http.calls if c[1] == "/v1/messages/count_tokens"]
    assert all(set(b) == {"model", "system", "messages"} for b in bodies)
    counts = ra.load_counts(files.counts)
    system, template = ra.rubric_parts()
    rows = _rows(8)
    m2 = ra.make_requests(rows, system, template, "main", counts)
    assert all(r["input_tokens_counted"] is not None for r in m2)
    r0 = m2[0]
    assert r0["worst_batch_usd"] == pytest.approx((np.ceil(r0["input_tokens_counted"] * ra.COUNT_MARGIN) + ra.COUNT_PAD)
                                                  * 2.5e-06 + ra.MAX_TOKENS * 1.25e-05)
    # a changed text is not covered by an old count
    rows[0]["reply"] = "changed"
    assert ra.make_requests(rows[:1], system, template, "main", counts)[0]["input_tokens_counted"] is None
    assert ra.count_tokens(http, m2, files.counts, sleep=lambda s: None)["counted_now"] == 0


def test_dry_run_bounds(tmp_path):
    meta, main, rep, files = _setup(tmp_path, n=10, n_rep=2)
    st = ra.judge_state([], meta["config_sha"])
    out = ra.dry_run(main, rep, st, count=lambda s: max(1, len(s) // 4))
    assert out["requests_to_send"] == 12 and out["main_to_send"] == 10 and out["repeat_to_send"] == 2
    assert out["expected"]["usd"] < out["high"]["usd"] < out["worst"]["usd"]
    assert out["worst"]["input_tokens"] == sum(r["n_bytes"] for r in main + rep) + 12 * ra.PROMPT_OVERHEAD
    assert out["worst"]["usd"] == pytest.approx(sum(r["worst_batch_usd"] for r in main + rep))
    assert out["counted"]["usd"] is None and out["pilot"]["requests"] == 10
    assert out["price_per_token_batch"] == {"input": 2.5e-06, "output": 1.25e-05}


def test_http_headers_host_and_redaction():
    seen = {}

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["url"] = req.full_url
        if "fail" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(
                json.dumps({"error": {"message": f"invalid x-api-key {KEY}"}}).encode()))
        return Resp(b'{"ok": true}')
    h = ra.AnthropicHttp(KEY, opener=opener)
    r = h.call("POST", "/v1/messages/batches", {"requests": []})
    assert r["status"] == 200 and r["json"] == {"ok": True}
    assert seen["url"] == "https://api.anthropic.com/v1/messages/batches"
    assert seen["headers"]["x-api-key"] == KEY and seen["headers"]["anthropic-version"] == ra.ANTHROPIC_VERSION
    assert seen["headers"]["content-type"] == "application/json"
    r = h.call("GET", "/v1/fail")
    assert r["status"] == 401 and "SECRET" not in r["error"] and "SECRET" not in json.dumps(r["json"])
    with pytest.raises(SystemExit):
        h.call("GET", "https://elsewhere.example.com/results")
    assert "SECRET" not in ra.redact(f"key {KEY} and sk-ant-admin01-OTHERKEY123456")


# ------------------------------------------------------------------------------------------------ human sheets


def _fake_sample(n_per_cell=20):
    rows, k = [], 0
    for t in ra.TASKS:
        for b, n_tok in zip(ra.BINS, (5, 40, 200)):
            for i in range(n_per_cell):
                m = f"model{i % 4}"
                rows.append({"sample_id": f"panel|{t}|{m}|cfg|{k}", "source": "panel", "task": t, "bin": b,
                             "stratum": f"panel|{t}|{m}|{b}", "mult": 1 + (i % 2), "M_h": 100.0, "model_key": m,
                             "n_tokens": n_tok, "question": f"Q{k}, \"quoted\"\nline 2", "reply": f"= reply {k}"})
                k += 1
    rows.append({"sample_id": "qwen|gsm8k|qwen4-max|x", "source": "qwen", "task": "gsm8k", "bin": "1-16",
                 "stratum": "qwen|x", "mult": 1, "M_h": 5.0, "model_key": "qwen4-max", "n_tokens": 3,
                 "question": "q", "reply": "r"})
    return rows


def test_cell_draw_even_seeded_and_weighted():
    smp = _fake_sample()
    d = ra.cell_draw(ra.panel_frame(smp), 150, 5, "human")
    sizes = d.groupby("cell").size()
    assert len(d) == 150 and len(sizes) == 12 and set(sizes) == {12, 13} and (sizes == 13).sum() == 6
    assert d["sample_id"].is_unique and not d["sample_id"].str.startswith("qwen").any()
    assert d.equals(ra.cell_draw(ra.panel_frame(list(reversed(smp))), 150, 5, "human"))
    assert not set(d["sample_id"]) == set(ra.cell_draw(ra.panel_frame(smp), 150, 5, "repeat")["sample_id"])
    # design weight = sample weight / (n drawn / n in the cell); the sample weights of a stratum sum to M_h
    p = ra.panel_frame(smp)
    codes, mult, M = ra.design(p.sort_values("sample_id").reset_index(drop=True))
    assert ra.weights(codes, mult, M).sum() == pytest.approx(100.0 * p["stratum"].nunique())
    r = d.iloc[0]
    assert r["design_weight"] == pytest.approx(r["w_sample"] * r["n_cell_sample"] / r["n_cell_draw"])
    assert r["n_cell_sample"] == 20


def test_human_sheets_two_orders_and_read_back(tmp_path, monkeypatch):
    smp = _fake_sample()
    sample = tmp_path / "sample.jsonl"
    ra.write_jsonl(sample, smp)
    sheets = {1: tmp_path / "h1.csv", 2: tmp_path / "h2.csv"}
    xlsx = {1: tmp_path / "h1.xlsx", 2: tmp_path / "h2.xlsx"}
    for k, v in (("SAMPLE", sample), ("HUMAN_SHEETS", sheets), ("HUMAN_XLSX", xlsx), ("HUMAN_KEY", tmp_path / "hk.csv"),
                 ("HUMAN_INSTRUCTIONS", tmp_path / "hi.md")):
        monkeypatch.setattr(ra, k, v)
    ra.cmd_human(SimpleNamespace(n=150, seed=5))
    assert sheets[1].read_bytes()[:3] == b"\xef\xbb\xbf"              # UTF-8 with BOM
    h1 = pd.read_csv(sheets[1], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    h2 = pd.read_csv(sheets[2], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(h1.columns) == ra.SHEET_COLUMNS == ["hid", "benchmark", "prompt", "reply", "label", "unsure", "note"]
    assert list(h1["hid"]) == [f"H{i:03d}" for i in range(1, 151)] == list(h2["hid"])
    assert set(h1["reply"]) == set(h2["reply"]) and list(h1["reply"]) != list(h2["reply"])
    assert (h1[["label", "unsure", "note"]] == "").all().all() and h1["prompt"].str.contains("line 2").all()
    assert set(h1["benchmark"]) == set(ra.TASK_NAME.values())
    text = sheets[1].read_text(encoding="utf-8-sig")
    assert "model0" not in text and "panel|" not in text              # no model name, no ids
    key = pd.read_csv(tmp_path / "hk.csv", dtype=str)
    assert {"hid_rater1", "hid_rater2", "sample_id", "cell", "design_weight"} <= set(key.columns) and len(key) == 150
    assert "R (intermediate reasoning)" in (tmp_path / "hi.md").read_text(encoding="utf-8")
    try:
        import openpyxl  # noqa: F401
        assert xlsx[1].exists()
    except ImportError:
        assert not xlsx[1].exists()
    assert ra.read_human(sheets, tmp_path / "hk.csv", xlsx) is None   # no labels yet
    # both raters fill their sheets; the same row gets the same label through each sheet's own hid
    smp_by = {r["sample_id"]: r for r in smp}
    truth = {s: "RPA"[int(s.rsplit("|", 1)[1]) % 3] for s in key["sample_id"]}
    for k in (1, 2):
        rows = list(csv.reader(open(sheets[k], encoding="utf-8-sig")))
        hid2sid = dict(zip(key[f"hid_rater{k}"], key["sample_id"]))
        for r in rows[1:]:
            r[4] = truth[hid2sid[r[0]]].lower() if k == 1 else truth[hid2sid[r[0]]]
            r[5] = "x" if r[0] == "H001" else ""
        with open(sheets[k], "w", encoding="utf-8-sig", newline="") as fh:
            csv.writer(fh).writerows(rows)
    hum = ra.read_human(sheets, tmp_path / "hk.csv", xlsx)
    assert (hum["rater1"] == hum["sample_id"].map(truth)).all() and (hum["rater2"] == hum["rater1"]).all()
    assert hum["unsure1"].sum() == 1 and hum["unsure2"].sum() == 1
    assert smp_by[hum["sample_id"][0]]["source"] == "panel"
    with pytest.raises(SystemExit):                                   # labelled sheets are never overwritten
        ra.cmd_human(SimpleNamespace(n=150, seed=5))
    rows = list(csv.reader(open(sheets[2], encoding="utf-8-sig")))
    rows[1][4] = "B"
    with open(sheets[2], "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with pytest.raises(SystemExit):
        ra.read_human(sheets, tmp_path / "hk.csv", xlsx)


def test_xlsx_writer_reports_why_not_written(tmp_path):
    why = ra.write_xlsx(tmp_path / "x.xlsx", [["H001", "GSM8K", "=1+1", "r", "", "", ""]])
    try:
        import openpyxl
        assert why is None
        ws = openpyxl.load_workbook(tmp_path / "x.xlsx").active
        assert ws["C2"].value == "=1+1" and ws["C2"].data_type == "s"
    except ImportError:
        assert why == "openpyxl is not installed"


# ------------------------------------------------------------------------------------------------ the saved files


@pytest.mark.skipif(not ra.SAMPLE.exists(), reason="sample not drawn")
def test_saved_sample_invariants():
    rows = ra.read_jsonl(ra.SAMPLE)
    keys = {r["sample_id"]: r for r in ra.read_jsonl(ra.KEYFILE)}
    assert len({r["sample_id"] for r in rows}) == len(rows) == len(keys)
    for r in rows:
        assert not ({"gold", "requested_strict", "content_correct", "score"} & set(r))
        assert r["bin"] == ra.bin_of(r["n_tokens"]) and abs(r["p_incl"] - r["n_h"] / r["N_h"]) < 1e-12
        assert r["question"] and isinstance(r["reply"], str)
    df = pd.DataFrame(rows)
    panel = df[df["source"] == "panel"]
    assert (panel.groupby(["task", "model_key"]).size() == 3 * ra.PER_BIN).all()
    cells = panel.groupby(["task", "bin"]).size()
    assert len(cells) == 12 and (cells >= 15).all()
    q = df[df["source"] == "qwen"]
    assert set(q["task"]) == set(ra.QWEN_TASKS) and len(q) == 110
    man = json.load(open(ra.MANIFEST, encoding="utf-8"))
    assert man["seed"] == ra.SEED and man["n_rows"] == len(rows)
    import hashlib
    assert man["sample_sha256"] == hashlib.sha256(ra.SAMPLE.read_bytes()).hexdigest()
    chk = json.load(open(ra.POPULATION, encoding="utf-8"))["checks"]
    assert chk["hotpot"]["w_vs_hotpot_forecast_max_abs"] < 1e-12
    for t in ("gsm8k", "math", "bbh"):
        assert chk[t]["w_vs_baseline_mixture_max_abs"] < 1e-12
        assert chk[t]["req_minus_forced_strict_vs_baseline_mixture_max_abs_pp"] < 1e-9
        assert chk[t].get("cot_minus_forced_strict_vs_baseline_mixture_max_abs_pp", 0) < 1e-9


@pytest.mark.skipif(not ra.HUMAN_KEY.exists(), reason="human sheets not drawn")
def test_saved_human_sheets():
    key = pd.read_csv(ra.HUMAN_KEY, dtype=str)
    smp = ra.read_jsonl(ra.SAMPLE)
    d = ra.human_draw(smp)
    assert list(key["sample_id"]) == list(d["sample_id"])           # the saved draw is the seeded one
    assert set(key.groupby("cell").size()) == {12, 13} and len(key) == ra.HUMAN_N
    for k in (1, 2):
        h = pd.read_csv(ra.HUMAN_SHEETS[k], dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert list(h.columns) == ra.SHEET_COLUMNS and len(h) == ra.HUMAN_N
