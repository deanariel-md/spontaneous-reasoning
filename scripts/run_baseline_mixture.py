r"""The requested baseline as a mixture of short answers and written-out working. Exploratory on GSM8K-29,
BIG-Bench Hard-28 and MATH-500; its forward test is the HotpotQA forecast saved before HotpotQA validation is scored.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_baseline_mixture.py                    # the three completed benchmarks
    .\.venv\Scripts\python.exe scripts\run_baseline_mixture.py --hotpot-forecast  # HotpotQA discovery stage (see below)

Three benchmarks (exploratory). Token count: n_output_tokens of each stored requested-direct output, the number of token
ids the model generated (vLLM's len(output.token_ids), under the model's own tokenizer; src/generation.py), carried
unchanged into the scored tables. An output is long when n_output_tokens > the forced arm's max_new_tokens
(config/decoding_forced.yaml: 16 for GSM8K and BIG-Bench Hard; config/decoding_forced_math.yaml: 16 for MATH-500).
  1. Item level: every (model, configuration, item) of the distinct configurations (both splits; the discovery
     configurations alone in tables/), split by the requested output's length; accuracy of the requested, forced and
     chain-of-thought arms on the same model, configuration and item, under strict, content and lenient scores
     (data/processed/content/<task>_content.parquet), pooled over the rows, with a percentile model bootstrap (models
     drawn with replacement, all rows of a drawn model kept).
  2. Per model w (discovery draws, src/baseline_mixture.RULE_TEXT) and the correction: predicted gain over the
     requested arm = (1 - w) x gain over the forced arm (model means over the 24 discovery draws). Reversed pairs
     (results/arm_effect/'s definition) between the observed gain over the requested arm and the gain over the forced
     arm, against those between the observed gain and the corrected one; correlations with the observed gain. Strict
     and content scores; w is the same for both (it needs no gold answer).
HotpotQA (--hotpot-forecast): reads data/processed/hotpot/scored_discovery.parquet only (discovery draws), computes each
model's w at HotpotQA's forced cap (64 tokens, config/decoding_hotpot.yaml) and its gain over the forced arm (strict
exact match), and writes manifests/predictions_hotpot_mechanism.json (write-once: an identical forecast is kept, a
different one refused). run_hotpot.py --stage discovery calls the same function and lists the file in its index, so
score_hotpot.py --stage all refuses if it is absent or changed; this step can be run again after the
discovery stage (it keeps the identical file). run_hotpot.py's full run evaluates the forecast on the validation draws
(evaluate_hotpot_forecast), misses included.

Outputs: results/baseline_mixture/baseline_mixture.json, summary.md, summary_clean.md, tables/item_level.csv,
tables/per_model.csv, tables/mechanism.csv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import baseline_mixture as bm, decision as dc  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "baseline_mixture")
CONTENT = os.path.join(ROOT, "data", "processed", "content")
TASKS = ("gsm8k", "bbh", "math")
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
CAP_FILE = {"gsm8k": "decoding_forced.yaml", "bbh": "decoding_forced.yaml", "math": "decoding_forced_math.yaml"}
KEY = ["model_key", "config_id", "item_id"]
ARMS = {"direct_answer": "requested", "forced_direct": "forced", "zero_shot_cot": "cot"}
SCORES = {"strict_correct": "strict", "content_correct": "content", "lenient_correct": "lenient"}
N_BOOT = 9999
HOTPOT_FORECAST = "predictions_hotpot_mechanism.json"
# Reference figures (lenient scores, split at 16 tokens): share, requested, forced, chain of thought
REF_ITEM = {("math", "short"): (58, 27.3, 28.1, 71.8), ("math", "long"): (42, 40.0, 18.3, 48.5),
              ("gsm8k", "short"): (57, 25.7, 29.1, 88.6), ("gsm8k", "long"): (43, 47.1, 20.9, 68.9)}
# Reference reversed-pair shares, observed -> corrected (score)
REF_REV = {("math", "strict"): (0.20, 0.06), ("gsm8k", "strict"): (0.20, 0.12), ("bbh", "content"): (0.25, 0.13)}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def cap_of(task):
    return int(yaml.safe_load(open(os.path.join(ROOT, "config", CAP_FILE[task]), encoding="utf-8"))
               ["max_new_tokens"]["forced_direct"])


# ============================================================================ the three benchmarks


def task_rows(task):
    """(wide rows, panel, discovery draws, validation draws): one row per (model, configuration, item) with the
    requested output's token count and the three arms' strict, content and lenient scores."""
    from run_forced_direct import collection_inputs
    inp, s, items, _ = collection_inputs(task)
    panel = [str(m) for m in inp["panel"]]
    disc, val = [str(u) for u in s["discovery_config_ids"]], [str(u) for u in s["validation_config_ids"]]
    allc = sorted(set(disc) | set(val))
    col = inp["load"](allc)
    req = col[col["arm"] == "direct_answer"][KEY + ["n_output_tokens"]].copy()
    con = pd.read_parquet(os.path.join(CONTENT, f"{task}_content.parquet"),
                          columns=KEY + ["arm"] + list(SCORES), filters=[("config_id", "in", allc)])
    for d in (req, con):
        d["model_key"] = d["model_key"].astype(str)
        d["config_id"] = d["config_id"].astype(str)
        d["item_id"] = d["item_id"].astype(str)
    req = req[req["model_key"].isin(panel)]
    con = con[con["model_key"].isin(panel)]
    wide = None
    for a, name in ARMS.items():
        x = con[con["arm"] == a].drop(columns="arm").rename(columns={c: f"{name}_{SCORES[c]}" for c in SCORES})
        wide = x if wide is None else wide.merge(x, on=KEY, how="inner", validate="one_to_one")
    n0 = len(wide)
    wide = wide.merge(req, on=KEY, how="inner", validate="one_to_one")
    per_arm = {a: int((con["arm"] == a).sum()) for a in ARMS}
    if not (len(wide) == n0 == len(req) == per_arm["direct_answer"] == per_arm["forced_direct"] == per_arm["zero_shot_cot"]):
        raise SystemExit(f"{task}: the content table and the requested outputs do not align one to one "
                         f"(joined {len(wide)}, content per arm {per_arm}, requested rows {len(req)})")
    if wide["n_output_tokens"].isna().any():
        raise SystemExit(f"{task}: a requested output has no token count")
    return wide, panel, disc, val


def item_block(wide, panel, cap, configs, rng):
    w = wide[wide["config_id"].isin(set(configs))]
    m = pd.Index(panel).get_indexer(w["model_key"])
    lf = w["n_output_tokens"].to_numpy() > cap
    scores = {(arm, sc): w[f"{arm}_{sc}"].to_numpy().astype(float)
              for arm in ("requested", "forced", "cot") for sc in ("strict", "content", "lenient")}
    return bm.item_level(m, lf, scores, len(panel), N_BOOT, rng)


def w_of(wide, panel, cap, draws):
    """(M,) draw-weighted long share on the given draws (repeats included)."""
    cfg = sorted(set(draws))
    w = wide[wide["config_id"].isin(set(cfg))]
    return bm.draw_weighted_share(w["n_output_tokens"].to_numpy() > cap, pd.Index(panel).get_indexer(w["model_key"]),
                                  pd.Index(cfg).get_indexer(w["config_id"]), len(panel), len(cfg),
                                  pd.Index(cfg).get_indexer(draws))


def run_task(task):
    import run_decision as rd
    cap = cap_of(task)
    wide, panel, disc, val = task_rows(task)
    rng = np.random.default_rng(dc.label_seed(f"baseline_mixture/{task}"))
    out = {"title": TITLE[task], "forced_cap_tokens": cap, "M": len(panel), "models": panel,
           "rows": int(len(wide)), "item_level": {"all_distinct": item_block(wide, panel, cap, set(disc) | set(val), rng),
                                                  "discovery": item_block(wide, panel, cap, set(disc), rng)}}
    w_disc = w_of(wide, panel, cap, disc)
    w_all = w_of(wide, panel, cap, disc + val)
    if np.isnan(w_disc).any():
        raise SystemExit(f"{task}: a model lacks requested outputs in a discovery configuration")
    x = ri.load(task)
    if x["facts"]["panel"] != panel or x["facts"]["discovery_config_ids"] != disc:
        raise SystemExit(f"{task}: the strict arrays differ in panel or draws")
    d = x["discovery"]
    G = {"strict": {a: (d["cot"] - d[a]).mean(0) for a in ("requested", "forced")}}
    c = rd.task_split(task, "content", "discovery")
    if [str(u) for u in c["units"]] != disc or [str(k) for k in c["panel"]] != panel:
        raise SystemExit(f"{task}: the content arrays differ in panel or draws")
    G["content"] = {a: (np.asarray(c["cot"], float) - np.asarray(c[a], float)).mean(0) for a in ("requested", "forced")}
    out["w_discovery"] = w_disc.tolist()
    out["w_all_draws"] = w_all.tolist()
    out["mechanism"] = {sc: bm.mechanism(g["requested"], g["forced"], w_disc) for sc, g in G.items()}
    out["mechanism_w_all_draws"] = {sc: bm.mechanism(g["requested"], g["forced"], w_all) for sc, g in G.items()}
    out["per_model"] = [{"model_key": k, "w_discovery": w_disc[j], "w_all_draws": w_all[j],
                         **{f"{sc}_gain_vs_{a}": float(G[sc][a][j]) for sc in G for a in ("requested", "forced")},
                         **{f"{sc}_predicted_gain_vs_requested": float((1 - w_disc[j]) * G[sc]["forced"][j]) for sc in G}}
                        for j, k in enumerate(panel)]
    return out


# ============================================================================ the exact decomposition (exploratory)


DECOMP_SETS = ("all_distinct", "discovery", "validation")
RECOVERY_W_MIN = 0.1    # sensitivity: within-model recovery among models with at least this share of long answers


def row_weights(config_ids, draws):
    """Per row, the number of draws of its configuration (repeats count; 0 for a configuration in no draw)."""
    from collections import Counter
    cnt = Counter(map(str, draws))
    return np.array([cnt.get(str(c), 0) for c in config_ids], float)


def decomposition_block(wide, panel, cap, draws, weighted: bool, scores=("strict", "content")) -> dict:
    """{score: bm.decomposition(...)} on the rows of the given draws: every distinct configuration once (weighted False)
    or each configuration weighted by its number of draws (weighted True)."""
    x = wide[wide["config_id"].isin(set(map(str, draws)))]
    m = pd.Index(panel).get_indexer(x["model_key"])
    if (m < 0).any():
        raise SystemExit("a row's model is not in the panel")
    lf = x["n_output_tokens"].to_numpy() > cap
    wt = row_weights(x["config_id"], draws) if weighted else None
    out = {}
    for sc in scores:
        d = bm.decomposition(m, lf, x[f"requested_{sc}"].to_numpy(), x[f"forced_{sc}"].to_numpy(),
                             x[f"cot_{sc}"].to_numpy(), len(panel), wt)
        p = d["per_model"]
        rec, w = np.array(p["recovery_long"], float), np.array(p["w"], float)
        den = np.array(p["C_long"], float) - np.array(p["F_long"], float)
        sel = np.isfinite(rec) & (den > 0) & (w >= RECOVERY_W_MIN)
        d["recovery_long_w_at_least"] = {"w_min": RECOVERY_W_MIN, **(bm._quart(rec[sel]) or {"n": 0})}
        out[sc] = d
    return out


def decomposition_task(task, rows=None) -> dict:
    """The decomposition (all distinct configurations once; discovery and validation draws, each configuration weighted
    by its draws), and the choice by (1 - w) x forced gain against the choice by forced gain, both valued under the
    requested arm: in-sample on the discovery means and on the validation means (exploratory)."""
    from src import decision as dc
    cap = cap_of(task)
    wide, panel, disc, val = rows or task_rows(task)
    sets = {"all_distinct": (sorted(set(disc) | set(val)), False), "discovery": (disc, True), "validation": (val, True)}
    out = {"title": TITLE[task], "forced_cap_tokens": cap, "models": panel,
           "sets": {k: decomposition_block(wide, panel, cap, d, wt) for k, (d, wt) in sets.items()}}
    n = dc.third_size(len(panel))
    out["corrected_choice"] = {}
    for sc in ("strict", "content"):
        pd_ = out["sets"]["discovery"][sc]["per_model"]
        pv = out["sets"]["validation"][sc]["per_model"]
        C, F, Rq = (np.array(pd_[k], float) for k in ("C", "F", "R"))
        g_fd, w = C - F, np.array(pd_["w"], float)
        gS = np.array(pd_["C_short"], float) - np.array(pd_["F_short"], float)
        g_req_val = np.array(pv["C"], float) - np.array(pv["R"], float)
        out["corrected_choice"][sc] = {
            "n_chosen": n,
            "in_sample_discovery": bm.corrected_choice(g_fd, w, C - Rq, n, gS),
            "validation_means": bm.corrected_choice(g_fd, w, g_req_val, n, gS),
            "note": "choices from the discovery means (w, gain over the forced arm, short-stratum gain); regret under "
                    "the requested arm, plug-in on model means; exploratory"}
    return out


DECOMP_RULE = (
    "Exact decomposition: a model's requested-minus-forced accuracy (its baseline shift, pp) is the sum of a part "
    "from rows whose requested output is longer than the forced cap and a part from the other rows, each the sum of "
    "(requested - forced) over its rows divided by all the model's rows. Between-model variance share of the long part: "
    "cov(long part, shift) / var(shift) over models (ddof 1). Within-model recovery on long rows: (R_L - F_L) / "
    "(C_L - F_L), models with a positive denominator (the others listed). Closed fraction: (R - F) / (C - F) among "
    f"models with C - F >= {bm.DECOMP_ROOM_PP:g} pp, correlated with w. Two forms of the (1 - w) rule against the "
    "observed gain over the requested arm C - R: (1 - w) x (C - F) and (1 - w) x (C_S - F_S), the latter needing the "
    "short stratum.")


def decomposition_tables(D: dict):
    rows, pm = [], []
    for t, T in D.items():
        for s, blk in T["sets"].items():
            for sc, d in blk.items():
                cf, fo = d["closed_fraction"], d["forms"]
                rows.append({"task": t, "set": s, "scores": sc, "variance_share_long": d["variance_share_long"],
                             "variance_share_short": d["variance_share_short"],
                             "var_long": (d["variance_terms"] or {}).get("long"),
                             "var_short": (d["variance_terms"] or {}).get("short"),
                             "two_cov": (d["variance_terms"] or {}).get("two_cov"),
                             "recovery_long_median": (d["recovery_long"] or {}).get("median"),
                             "recovery_long_q25": (d["recovery_long"] or {}).get("q25"),
                             "recovery_long_q75": (d["recovery_long"] or {}).get("q75"),
                             "recovery_long_n": (d["recovery_long"] or {}).get("n"),
                             "recovery_long_w01_median": d["recovery_long_w_at_least"].get("median"),
                             "short_gap_median_pp": (d["short_gap_pp"] or {}).get("median"),
                             "closed_models": cf["models"], "closed_corr_w": cf["corr_with_w"]["pearson"],
                             "shift_corr_w_room_models": cf["corr_shift_with_w"]["pearson"],
                             "shift_corr_w_all_models": cf["corr_shift_with_w_all_models"]["pearson"],
                             "rms_pooled_form": fo["pooled"]["rms_pp"], "rms_short_form": fo["short_stratum"]["rms_pp"],
                             "pearson_pooled_form": fo["pooled"]["corr"]["pearson"],
                             "pearson_short_form": fo["short_stratum"]["corr"]["pearson"],
                             "reversed_pooled_form": fo["pooled"]["reversed"],
                             "reversed_short_form": fo["short_stratum"]["reversed"], "pairs": fo["pooled"]["pairs"]})
                if s == "all_distinct":
                    p = d["per_model"]
                    for j, k in enumerate(T["models"]):
                        pm.append({"task": t, "scores": sc, "model_key": k, **{q: p[q][j] for q in p}})
    return pd.DataFrame(rows), pd.DataFrame(pm)


def decomposition_lines(D: dict) -> list:
    L = ["", "## The exact decomposition of the baseline shift (exploratory)", "",
         "Requested minus forced accuracy (the baseline shift) splits exactly "
         "into a part from rows with a long requested output (more than the forced cap) and a part from short ones. "
         "Rows: every distinct configuration of both splits once. Variance share: cov(part, shift) / var(shift) over "
         "models. Recovery: (R_L − F_L) / (C_L − F_L) on long rows, median [quartiles] over models with a positive "
         f"denominator. Closed fraction (R − F) / (C − F) among models with at least {bm.DECOMP_ROOM_PP:g} pp of room, "
         "correlated with w.", "",
         "| Benchmark | Scores | Long share of variance | Recovery on long rows | Recovery, w ≥ 0.1 | Short-row gap R − F "
         "(median, pp) | Closed fraction vs w (models) | Shift vs w, all models |", "|---|---|---:|---|---:|---:|---|---:|"]
    for t, T in D.items():
        for sc, d in T["sets"]["all_distinct"].items():
            r, cf = d["recovery_long"] or {}, d["closed_fraction"]
            L.append(f"| {T['title']} | {sc} | {f3(d['variance_share_long'])} | {f3(r.get('median'))} "
                     f"[{f3(r.get('q25'))}, {f3(r.get('q75'))}] (n {r.get('n')}) | "
                     f"{f3(d['recovery_long_w_at_least'].get('median'))} | {f1(d['short_gap_pp']['median'])} | "
                     f"{f3(cf['corr_with_w']['pearson'])} ({cf['models']}) | "
                     f"{f3(cf['corr_shift_with_w_all_models']['pearson'])} |")
    L += ["", "Both forms of the (1 − w) rule against the observed gain over the requested arm (rows of the discovery "
          "draws, each configuration weighted by its draws):", "",
          "| Benchmark | Scores | RMS, (1 − w)(C − F) | RMS, (1 − w)(C_S − F_S) | Pearson, pooled form | Pearson, short form | "
          "Reversed pairs, pooled / short form |", "|---|---|---:|---:|---:|---:|---|"]
    for t, T in D.items():
        for sc, d in T["sets"]["discovery"].items():
            fo = d["forms"]
            L.append(f"| {T['title']} | {sc} | {f1(fo['pooled']['rms_pp'])} | {f1(fo['short_stratum']['rms_pp'])} | "
                     f"{f3(fo['pooled']['corr']['pearson'])} | {f3(fo['short_stratum']['corr']['pearson'])} | "
                     f"{fo['pooled']['reversed']} / {fo['short_stratum']['reversed']} of {fo['pooled']['pairs']} |")
    L += ["", "Choosing the top third under the requested arm (regret in pp per chosen model; choices from discovery "
          "means; exploratory):", "",
          "| Benchmark | Scores | By forced gain, in sample | By (1 − w) × forced gain, in sample | By short form, in sample | "
          "By forced gain, validation | By (1 − w) × forced gain, validation | By short form, validation |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t, T in D.items():
        for sc, c in T["corrected_choice"].items():
            a, b = c["in_sample_discovery"], c["validation_means"]
            L.append(f"| {T['title']} | {sc} | " + " | ".join(f1(x[k]["regret"]) for x in (a, b)
                                                             for k in ("forced_gain", "corrected", "corrected_short")) + " |")
    L += ["", "Tables: `tables/decomposition.csv` (every set and score), `tables/decomposition_per_model.csv` (all "
          "distinct configurations)."]
    return L


def run_decomposition():
    """Add the decomposition to results/baseline_mixture/ without recomputing the existing outputs: new keys in
    baseline_mixture.json, new tables, and new sections appended to the summary."""
    import robustness_inputs as rinp
    path = os.path.join(OUT, "baseline_mixture.json")
    R = json.load(open(path, encoding="utf-8"))
    D = {}
    for t in TASKS:
        D[t] = decomposition_task(t)
        print(f"{t}: decomposition done", flush=True)
    R["decomposition_rule"] = DECOMP_RULE
    R["decomposition"] = D
    R["decomposition_created"] = time.strftime("%Y-%m-%d %H:%M")
    a, b = decomposition_tables(D)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    a.to_csv(os.path.join(OUT, "tables", "decomposition.csv"), index=False)
    b.to_csv(os.path.join(OUT, "tables", "decomposition_per_model.csv"), index=False)
    rinp.write_outputs(OUT, R, summary(R))
    print("\n".join(decomposition_lines(D)))
    return R


# ============================================================================ HotpotQA: the decomposition rule (discovery only)

HOTPOT_DECOMPOSITION = "predictions_hotpot_decomposition.json"
HQ_VAR_SHARE_THRESHOLD = 0.5
HQ_DECOMP_RULE = (
    "Saved from HotpotQA discovery draws only, before any validation draw is scored. Long: a requested-direct "
    "output of more than 64 generated tokens (n_output_tokens > HotpotQA's forced cap, config/decoding_hotpot.yaml). "
    "Statement 1: on the 24 validation draws (rows of each configuration weighted by its number of validation draws; "
    "strict exact match), the long part's share of the between-model variance of the baseline shift (requested minus "
    "forced accuracy), cov(long part, shift) / var(shift) over models (ddof 1), exceeds 0.5. The threshold was written "
    "before any HotpotQA output was scored: 0.5 is where long answers carry more of the between-model variance than "
    "short ones; the three completed benchmarks give 0.53 (BIG-Bench Hard, strict) to 0.92 (GSM8K, strict), and "
    "HotpotQA's short extractive answers make a format-driven shift, as on BIG-Bench Hard, the plausible failure. "
    "Statement 2: the top third (decision.third_size, ties at the cut sharing places) by (1 - w) x the gain over the "
    "forced arm (w the share of long requested outputs, both from the discovery draws, each configuration weighted by "
    "its draws) has lower plug-in regret under the requested arm (model means of chain of thought minus requested "
    "direct over the 24 validation draws) than the top third by the gain over the forced arm. Beside it: the same on "
    "the validation configurations absent from every discovery draw (each once), with the paired difference and its t "
    "interval over those configurations, and the choice by the short-stratum form (1 - w) x (C_S - F_S) (an empty "
    "short stratum counts 0). Every statement is reported, misses included; a variance of the shift that is not "
    "positive leaves statement 1 untestable and is reported so.")


def hotpot_wide(path, panel, items, configs):
    """One row per (model, configuration, item) of HotpotQA's scored table at path: the requested output's token count
    and the strict exact match of the requested, forced and chain-of-thought arms (scored as requested_strict etc.)."""
    df = pd.read_parquet(path, columns=["model_key", "arm", "config_id", "item_id", "strict_correct", "n_output_tokens"],
                         filters=[("config_id", "in", sorted(set(map(str, configs)))),
                                  ("arm", "in", ["direct_answer", "forced_direct", "zero_shot_cot"])])
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    df = df[df["model_key"].isin(set(panel)) & df["item_id"].isin(set(map(str, items)))]
    wide = None
    for a, name in (("direct_answer", "requested"), ("forced_direct", "forced"), ("zero_shot_cot", "cot")):
        x = df[df["arm"] == a][KEY + ["strict_correct"] + (["n_output_tokens"] if a == "direct_answer" else [])]
        x = x.rename(columns={"strict_correct": f"{name}_strict"})
        wide = x if wide is None else wide.merge(x, on=KEY, how="inner", validate="one_to_one")
    n_expect = len(panel) * len(set(map(str, configs))) * len(set(map(str, items)))
    if len(wide) != n_expect:
        raise SystemExit(f"{path}: {len(wide)} aligned rows, expected {n_expect} (models x configurations x items)")
    if wide["n_output_tokens"].isna().any():
        raise SystemExit(f"{path}: a requested output has no token count")
    return wide


def hotpot_cap():
    return int(yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_hotpot.yaml"), encoding="utf-8"))
               ["max_new_tokens"]["forced_direct"])


def hotpot_decomposition(processed=None, sampler_path=None, pred_dir=None, panel=None) -> dict:
    """Compute and save manifests/predictions_hotpot_decomposition.json from HotpotQA's discovery table alone
    (write-once)."""
    import run_hotpot as rh
    from src import decision as dc
    processed = processed or rh.PROCESSED
    sampler_path = sampler_path or rh.SAMPLER
    pred_dir = pred_dir or rh.MANIFESTS
    path = os.path.join(processed, "scored_discovery.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent; run score_hotpot.py --stage discovery first")
    sampler = json.load(open(sampler_path, encoding="utf-8"))
    disc = [str(u) for u in sampler["discovery_config_ids"]]
    panel, _, _ = rh.panel_info(panel)
    items = rh.item_ids(processed)
    cap = hotpot_cap()
    wide = hotpot_wide(path, panel, items, disc)
    d = decomposition_block(wide, panel, cap, disc, True, scores=("strict",))["strict"]
    p = d["per_model"]
    w, C, F, Rq = (np.array(p[k], float) for k in ("w", "C", "F", "R"))
    gS = np.array(p["C_short"], float) - np.array(p["F_short"], float)
    n = dc.third_size(len(panel))
    picks = {}
    for k, pred in (("forced_gain", C - F), ("corrected", bm.corrected_gain(C - F, w)),
                    ("corrected_short", (1 - w) * np.nan_to_num(gS, nan=0.0))):
        W, t = dc.choose(pred, n)
        picks[k] = {"prediction": pred.tolist(), "weights": W.tolist(), "ties_at_cut": int(t)}
    meta = path[:-len(".parquet")] + ".json"
    body = {"label": "hotpot_decomposition", "units": "pp; w as a share in [0, 1]", "rule": HQ_DECOMP_RULE,
            "forced_cap_tokens": cap, "threshold_variance_share_long": HQ_VAR_SHARE_THRESHOLD, "n_chosen": n,
            "models": panel, "discovery_units": disc,
            "per_model": {k: {"w": float(w[j]), "gain_vs_forced": float((C - F)[j]),
                              "gain_vs_forced_short": None if not np.isfinite(gS[j]) else float(gS[j]),
                              "predicted_gain_vs_requested": float(bm.corrected_gain(C - F, w)[j]),
                              "predicted_gain_vs_requested_short": float(((1 - w) * np.nan_to_num(gS, nan=0.0))[j])}
                          for j, k in enumerate(panel)},
            "choices": picks,
            "statements": {
                "variance_share": f"on the 24 validation draws, the long part's share of the between-model variance of "
                                  f"the baseline shift exceeds {HQ_VAR_SHARE_THRESHOLD}",
                "choice": "under the requested arm (validation means over the 24 draws), the top third by (1 - w) x "
                          "forced gain has lower plug-in regret than the top third by forced gain"},
            "discovery_context": {"variance_share_long": d["variance_share_long"],
                                  "note": "the discovery draws' own value, kept for reference; with reliable model "
                                          "means it largely anticipates the validation value, so the test is that the "
                                          "rule and threshold were fixed before any HotpotQA output was scored"},
            "source": {"scored_discovery": rel(path) if os.path.abspath(path).startswith(ROOT) else path,
                       "scored_discovery_sha256": (json.load(open(meta, encoding="utf-8")).get("sha256")
                                                   if os.path.exists(meta) else None),
                       "sampler_sha256": sampler.get("sha256"), "score": "strict exact match"},
            "note": "read only HotpotQA discovery-draw outputs; written before any validation draw was scored "
                    "The decomposition was found on GSM8K, BIG-Bench Hard and "
                    "MATH-500 (results/baseline_mixture/); this is its forward test."}
    body["sha256"] = bm.sha256_json({k: v for k, v in body.items() if k not in bm.UNHASHED})
    return bm.write_forecast(body, os.path.join(pred_dir, HOTPOT_DECOMPOSITION))


def evaluate_hotpot_decomposition(pred_dir, processed, panel, disc, val) -> dict:
    """The saved decomposition statements against HotpotQA's validation draws (data/processed/hotpot/scored.parquet)."""
    import run_hotpot as rh
    from src import decision as dc
    rec = bm.load_forecast(os.path.join(pred_dir, HOTPOT_DECOMPOSITION))
    panel = list(panel)
    if list(rec["models"]) != panel:
        raise SystemExit(f"{HOTPOT_DECOMPOSITION} was saved for another panel")
    path = os.path.join(processed, "scored.parquet")
    items = rh.item_ids(processed)
    cap = int(rec["forced_cap_tokens"])
    val = list(map(str, val))
    disc_set = set(map(str, disc))
    fresh = sorted(set(u for u in val if u not in disc_set))
    wide = hotpot_wide(path, panel, items, sorted(set(val)))
    dv = decomposition_block(wide, panel, cap, val, True, scores=("strict",))["strict"]
    th = float(rec["threshold_variance_share_long"])
    vs = dv["variance_share_long"]
    n = int(rec["n_chosen"])
    pv = dv["per_model"]
    g_req = np.array(pv["C"], float) - np.array(pv["R"], float)
    W = {k: np.asarray(v["weights"], float) for k, v in rec["choices"].items()}
    reg = {k: float(dc.regret_plugin(Wk, g_req[None], n)[0]) for k, Wk in W.items()}
    out = {"saved": {"path": f"manifests/{HOTPOT_DECOMPOSITION}", "sha256": rec["sha256"], "created": rec.get("created")},
           "statements": {
               "variance_share": {"threshold": th, "observed": vs, "discovery_value": rec["discovery_context"]["variance_share_long"],
                                  "holds": None if vs is None else bool(vs > th),
                                  "note": None if vs is not None else "the shift's variance over models is not positive"},
               "choice": {"regret_corrected": reg["corrected"], "regret_forced_gain": reg["forced_gain"],
                          "regret_corrected_short": reg["corrected_short"],
                          "difference_corrected_minus_forced": reg["corrected"] - reg["forced_gain"],
                          "holds": bool(reg["corrected"] < reg["forced_gain"]),
                          "equal": bool(reg["corrected"] == reg["forced_gain"])}},
           "validation_decomposition": {k: dv[k] for k in ("variance_share_long", "variance_share_short", "variance_terms",
                                                           "recovery_long", "short_gap_pp", "closed_fraction", "forms")}}
    # fresh configurations (each once): the same statements, with the paired difference over configurations
    if len(fresh) >= 2:
        df_ = decomposition_block(wide, panel, cap, fresh, False, scores=("strict",))["strict"]
        x = wide[wide["config_id"].isin(set(fresh))]
        cm = x.groupby(["config_id", "model_key"])[["cot_strict", "requested_strict"]].mean() * 100
        G = (cm["cot_strict"] - cm["requested_strict"]).unstack("model_key").loc[fresh, panel].to_numpy(float)
        out["fresh_configurations"] = {
            "n_configurations": len(fresh), "variance_share_long": df_["variance_share_long"],
            "regret": {k: float(dc.regret_plugin(Wk, G, n)[0]) for k, Wk in W.items()},
            "paired_corrected_minus_forced": dc.paired_difference(W["corrected"], W["forced_gain"], G, n)}
    else:
        out["fresh_configurations"] = {"n_configurations": len(fresh), "note": "fewer than two fresh configurations"}
    return out


def hotpot_decomposition_lines(E: dict) -> list:
    s = E["statements"]
    v, c = s["variance_share"], s["choice"]
    L = ["", "## Decomposition rule (saved before validation was scored)", "",
         f"Saved `{E['saved']['path']}` (sha256 `{E['saved']['sha256'][:16]}…`, {E['saved']['created']}).",
         f"- Long-answer share of the between-model variance of the baseline shift, 24 validation draws: "
         f"{f3(v['observed'])} (threshold {v['threshold']}; discovery {f3(v['discovery_value'])}); holds: {v['holds']}.",
         f"- Requested-arm regret of the top third: by (1 − w) × forced gain {f1(c['regret_corrected'])} pp, by forced "
         f"gain {f1(c['regret_forced_gain'])} pp, by the short form {f1(c['regret_corrected_short'])} pp; holds: "
         f"{c['holds']}."]
    fc = E.get("fresh_configurations") or {}
    if "regret" in fc:
        pdiff = fc["paired_corrected_minus_forced"]
        ci_ = pdiff.get("ci95")
        L.append(f"- Fresh configurations ({fc['n_configurations']}): variance share {f3(fc['variance_share_long'])}; "
                 f"regret corrected {f1(fc['regret']['corrected'])}, forced gain {f1(fc['regret']['forced_gain'])}; "
                 f"paired difference {f2_(pdiff['estimate'])}" + (f" [{f2_(ci_[0])}, {f2_(ci_[1])}]" if ci_ else "") + ".")
    return L


def f2_(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}".replace("-", "−")


# ============================================================================ HotpotQA: the forecast (discovery only)


def hotpot_forecast(processed=None, sampler_path=None, pred_dir=None, panel=None) -> dict:
    """Compute and save manifests/predictions_hotpot_mechanism.json from HotpotQA's discovery table alone (write-once)."""
    import run_hotpot as rh
    processed = processed or rh.PROCESSED
    sampler_path = sampler_path or rh.SAMPLER
    pred_dir = pred_dir or rh.MANIFESTS
    path = os.path.join(processed, "scored_discovery.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent; run score_hotpot.py --stage discovery first")
    sampler = json.load(open(sampler_path, encoding="utf-8"))
    disc = [str(u) for u in sampler["discovery_config_ids"]]
    panel, _, _ = rh.panel_info(panel)
    items = rh.item_ids(processed)
    cap = int(yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_hotpot.yaml"), encoding="utf-8"))
              ["max_new_tokens"]["forced_direct"])
    Bf, Pf = rh.arrays(rh.loader(path, "cot_forced"), disc, panel, items)
    Br, Pr = rh.arrays(rh.loader(path, "cot_req"), disc, panel, items)
    import pyarrow.parquet as pq
    if "n_output_tokens" not in pq.ParquetFile(path).schema.names:
        raise SystemExit(f"{path} has no n_output_tokens column: the forecast's w cannot be computed from it")
    df = pd.read_parquet(path, columns=["model_key", "arm", "config_id", "item_id", "n_output_tokens"],
                         filters=[("arm", "in", ["direct_answer"])])
    df["item_id"] = df["item_id"].astype(str)
    df["config_id"] = df["config_id"].astype(str)
    df = df[df["model_key"].isin(set(panel)) & df["item_id"].isin(set(items)) & df["config_id"].isin(set(disc))]
    if df["n_output_tokens"].isna().any():
        raise SystemExit(f"{path}: a requested output has no token count")
    cfg = sorted(set(disc))
    w = bm.draw_weighted_share(df["n_output_tokens"].to_numpy() > cap, pd.Index(panel).get_indexer(df["model_key"]),
                               pd.Index(cfg).get_indexer(df["config_id"]), len(panel), len(cfg),
                               pd.Index(cfg).get_indexer(disc))
    if np.isnan(w).any():
        raise SystemExit("a model lacks requested outputs in a discovery configuration: the forecast is not saved")
    meta = path[:-len(".parquet")] + ".json"
    source = {"scored_discovery": rel(path) if path.startswith(ROOT) else path,
              "scored_discovery_sha256": (json.load(open(meta, encoding="utf-8")).get("sha256")
                                          if os.path.exists(meta) else None),
              "sampler_sha256": sampler.get("sha256"), "decoding": "config/decoding_hotpot.yaml",
              "score": "strict exact match", "n_requested_outputs": int(len(df))}
    rec = bm.forecast_record(panel, w, (Pf - Bf).mean(0), (Pr - Br).mean(0), cap, disc, source)
    return bm.write_forecast(rec, os.path.join(pred_dir, HOTPOT_FORECAST))


def evaluate_hotpot_forecast(pred_dir, panel, V: dict) -> dict:
    """The saved forecast against the validation draws: V {contrast: (Bv, Pv)} (24 validation draws, strict exact
    match), as run_hotpot.run_full builds it."""
    rec = bm.load_forecast(os.path.join(pred_dir, HOTPOT_FORECAST))
    g_req = (np.asarray(V["cot_req"][1], float) - np.asarray(V["cot_req"][0], float)).mean(0)
    g_fd = (np.asarray(V["cot_forced"][1], float) - np.asarray(V["cot_forced"][0], float)).mean(0)
    return bm.evaluate_forecast(rec, panel, g_req, g_fd)


def hotpot_forecast_lines(E: dict) -> list:
    st = E["statements"]

    def f3(v):
        return "--" if v is None else f"{v:.3f}"
    return ["", "## Mechanism forecast (saved before validation was scored)", "",
            f"Forecast sha256 `{E['forecast_sha256'][:16]}…` ({E['forecast_created']}). Predicted reversed pairs "
            f"{E['predicted_reversed']} of {E['pairs']}; observed on the validation draws {E['observed_reversed']}; "
            f"hits {len(E['hits'])}, false alarms {len(E['false_alarms'])}, misses {len(E['misses'])}.",
            f"- Share: predicted {f3(st['share']['predicted'])}, observed {f3(st['share']['observed'])}, within "
            f"{st['share']['tolerance']}: {st['share']['holds']}.",
            f"- Pairs: reversal rate among predicted pairs {f3(st['pairs']['rate_among_predicted'])}, among the others "
            f"{f3(st['pairs']['rate_among_other'])}; holds: {st['pairs']['holds']}"
            + (f" ({st['pairs']['note']})" if st["pairs"].get("note") else "") + ".",
            f"- Gain: Pearson of the predicted gain with the validation gain over the requested arm "
            f"{f3(st['gain']['pearson_predicted'])}, of the discovery gain over the forced arm "
            f"{f3(st['gain']['pearson_discovery_forced_gain'])}; holds: {st['gain']['holds']}.",
            f"- The correction on the validation draws (saved w): reversed pairs "
            f"{E['correction_on_validation']['observed_vs_forced']['reversed']} against the forced gain, "
            f"{E['correction_on_validation']['observed_vs_corrected']['reversed']} against the corrected gain."]


# ============================================================================ report


def f1(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.1f}".replace("-", "−")


def f3(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.3f}".replace("-", "−")


def cell(x):
    c = x.get("ci95")
    return f"{f1(x['estimate'])} [{f1(c[0])}, {f1(c[1])}]" if c else f1(x["estimate"])


def checks(R):
    out = []
    for (t, g), want in REF_ITEM.items():
        b = R["tasks"][t]["item_level"]["all_distinct"][g]
        got = (100 * b["share"], *(b["acc"][f"{a}/lenient"]["estimate"] for a in ("requested", "forced", "cot")))
        ok = round(got[0]) == want[0] and all(round(x, 1) == y for x, y in zip(got[1:], want[1:]))
        out.append(f"{TITLE[t]}, {g} requested outputs (lenient, all distinct configurations): reference "
                   f"{want[0]}% / {want[1]} / {want[2]} / {want[3]}; here {got[0]:.0f}% / {got[1]:.1f} / {got[2]:.1f} / "
                   f"{got[3]:.1f} (share / requested / forced / chain of thought); {'reproduced' if ok else 'differs'}.")
    for (t, sc), (a, b) in REF_REV.items():
        m = R["tasks"][t]["mechanism"][sc]
        x, y = m["observed_vs_forced"]["reversed_share"], m["observed_vs_corrected"]["reversed_share"]
        ok = abs(x - a) < 0.01 and abs(y - b) < 0.01
        out.append(f"{TITLE[t]}, {sc} scores: reversed pairs reference {a:.0%} → {b:.0%}; here {x:.1%} → {y:.1%} (w from "
                   f"the discovery draws); {'reproduced to the whole percent of the reference' if ok else 'differs'}; with w "
                   f"from both splits "
                   f"{R['tasks'][t]['mechanism_w_all_draws'][sc]['observed_vs_corrected']['reversed_share']:.1%}.")
    cs = [(t, sc, R["tasks"][t]["mechanism"][sc]) for t in R["tasks"] for sc in ("strict", "content")]
    out.append("Correlation with the observed gain over the requested arm, forced gain → corrected gain (reference: about "
               "0.75 → 0.92–0.98): " + "; ".join(f"{TITLE[t]} {sc} {f3(m['corr_forced']['pearson'])} → "
                                                f"{f3(m['corr_corrected']['pearson'])}" for t, sc, m in cs) + ".")
    return out


def summary(R):
    L = ["# The requested baseline as a mixture of short answers and written-out working", "",
         "**Exploratory on these three benchmarks**: the pattern was found in these data. Its forward test is the "
         "HotpotQA forecast, saved before HotpotQA's validation draws are scored "
         "(`manifests/predictions_hotpot_mechanism.json`). Code "
         "`scripts/run_baseline_mixture.py`, `src/baseline_mixture.py`.", "",
         "Token count: `n_output_tokens` of each stored requested-direct output, the number of tokens the model "
         "generated (under its own tokenizer). A requested output is long when it exceeds the forced arm's cap of "
         f"{R['tasks']['gsm8k']['forced_cap_tokens']} tokens (`config/decoding_forced.yaml`, "
         "`config/decoding_forced_math.yaml`); a long output is one the forced arm could not have produced.", "",
         "## Item level: accuracy by the length of the requested output", "",
         "Every (model, configuration, item) of the distinct configurations of both splits; the three arms on the same "
         "model, configuration and item. Accuracy in percent, pooled over the rows; 95% percentile model bootstrap "
         f"({N_BOOT} replicates; models drawn with replacement). Discovery configurations alone: `tables/item_level.csv`.", "",
         "| Benchmark | Requested output | Share of outputs | Scores | Requested | Forced | Chain of thought |",
         "|---|---|---:|---|---|---|---|"]
    for t, T in R["tasks"].items():
        for g in ("short", "long"):
            b = T["item_level"]["all_distinct"][g]
            for sc in ("strict", "content", "lenient"):
                L.append(f"| {T['title']} | {'at most' if g == 'short' else 'more than'} {T['forced_cap_tokens']} tokens | "
                         f"{100 * b['share']:.0f}% | {sc} | " + " | ".join(cell(b["acc"][f"{a}/{sc}"])
                                                                       for a in ("requested", "forced", "cot")) + " |")
    L += ["", "## The correction (1 − w) × gain over the forced arm", "",
          "w: each model's share of requested outputs longer than the cap, discovery draws (no gold answers). Gains: "
          "model means over the 24 discovery draws. Reversed pairs as in `results/arm_effect/` (ties are neither).", "",
          "| Benchmark | Scores | Reversed, observed vs forced gain | Reversed, observed vs corrected | Pearson, forced | "
          "Pearson, corrected | RMS, forced (pp) | RMS, corrected (pp) |", "|---|---|---|---|---:|---:|---:|---:|"]
    for t, T in R["tasks"].items():
        for sc, m in T["mechanism"].items():
            a, b = m["observed_vs_forced"], m["observed_vs_corrected"]
            L.append(f"| {T['title']} | {sc} | {a['reversed']} of {a['pairs']} ({a['reversed_share']:.3f}) | "
                     f"{b['reversed']} of {b['pairs']} ({b['reversed_share']:.3f}) | {f3(m['corr_forced']['pearson'])} | "
                     f"{f3(m['corr_corrected']['pearson'])} | {f1(m['rms_forced_pp'])} | {f1(m['rms_corrected_pp'])} |")
    L += ["", "Per model (w, gains, corrected gains): `tables/per_model.csv`. On BIG-Bench Hard a long requested "
          "output can be the option text written out, not reasoning.", "",
          "## Checks against the reference figures", ""] + [f"- {c}" for c in R["checks"]]
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    if R.get("decomposition"):
        L += decomposition_lines(R["decomposition"])
    return L


def tables(R):
    il, pm, mc = [], [], []
    for t, T in R["tasks"].items():
        for cs, blk in T["item_level"].items():
            for g, b in blk.items():
                for k, x in b["acc"].items():
                    arm, sc = k.split("/")
                    c = x["ci95"] or [None, None]
                    il.append({"task": t, "configurations": cs, "requested_output": g, "n_rows": b["n"],
                               "share": b["share"], "arm": arm, "scores": sc, "accuracy": x["estimate"],
                               "ci_lo": c[0], "ci_hi": c[1]})
        for r in T["per_model"]:
            pm.append({"task": t, **r})
        for w_src, MM in (("discovery", T["mechanism"]), ("all_draws", T["mechanism_w_all_draws"])):
            for sc, m in MM.items():
                mc.append({"task": t, "w_from": w_src, "scores": sc, "pairs": m["observed_vs_forced"]["pairs"],
                           "reversed_vs_forced": m["observed_vs_forced"]["reversed"],
                           "reversed_vs_corrected": m["observed_vs_corrected"]["reversed"],
                           "pearson_forced": m["corr_forced"]["pearson"], "pearson_corrected": m["corr_corrected"]["pearson"],
                           "spearman_forced": m["corr_forced"]["spearman"],
                           "spearman_corrected": m["corr_corrected"]["spearman"],
                           "rms_forced": m["rms_forced_pp"], "rms_corrected": m["rms_corrected_pp"]})
    return pd.DataFrame(il), pd.DataFrame(pm), pd.DataFrame(mc)


def run_three():
    t0 = time.time()
    R = {"analysis": "the requested baseline as a mixture; exploratory on these three "
                     "benchmarks", "created": time.strftime("%Y-%m-%d %H:%M"), "rule": bm.RULE_TEXT,
         "n_boot": N_BOOT, "tasks": {}}
    for t in TASKS:
        R["tasks"][t] = run_task(t)
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["checks"] = checks(R)
    R["decomposition_rule"] = DECOMP_RULE   # the exact decomposition, exploratory
    R["decomposition"] = {t: decomposition_task(t) for t in TASKS}
    R["runtime_s"] = round(time.time() - t0, 1)
    da, db = decomposition_tables(R["decomposition"])
    il, pm, mc = tables(R)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    il.to_csv(os.path.join(OUT, "tables", "item_level.csv"), index=False)
    pm.to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    mc.to_csv(os.path.join(OUT, "tables", "mechanism.csv"), index=False)
    da.to_csv(os.path.join(OUT, "tables", "decomposition.csv"), index=False)
    db.to_csv(os.path.join(OUT, "tables", "decomposition_per_model.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print("\n".join(summary(R)))
    return R


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hotpot-forecast", action="store_true",
                    help="save (or keep, when identical) the HotpotQA forecast from the discovery table only")
    ap.add_argument("--hotpot-decomposition", action="store_true",
                    help="save (or keep, when identical) the HotpotQA decomposition rule from the discovery table only")
    ap.add_argument("--decomposition", action="store_true",
                    help="add the decomposition to results/baseline_mixture/ without recomputing the rest")
    a = ap.parse_args(argv)
    if a.decomposition:
        return run_decomposition()
    if a.hotpot_decomposition:
        rec = hotpot_decomposition()
        print(f"{HOTPOT_DECOMPOSITION}: sha256 {rec['sha256']} created {rec['created']}")
        return rec
    if a.hotpot_forecast:
        rec = hotpot_forecast()
        print(f"{HOTPOT_FORECAST}: sha256 {rec['sha256']} created {rec['created']}; predicted reversed pairs "
              f"{rec['predicted_reversal']['reversed']} of {rec['predicted_reversal']['pairs']}")
        return rec
    return run_three()


if __name__ == "__main__":
    main()
