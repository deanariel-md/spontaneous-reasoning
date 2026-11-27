r"""HotpotQA's analyses under the content reading, beside strict exact match.

Run from the repository root, after scripts/score_hotpot_content.py:
    python scripts/run_hotpot_content.py               # everything
    python scripts/run_hotpot_content.py --only formats rater_c

Every choice evaluated here is one HotpotQA's discovery stage saved before its validation draws were scored (strict
exact match); under content scoring the same saved choices are valued on content-scored validation draws, through the
same functions as the full run (scripts/run_hotpot.py) fed data/processed/hotpot/content_view/ (strict_correct holding
the content reading). Nothing is refitted on validation draws. Refits on discovery draws under content scoring, and every
analysis of the thresholds part, are exploratory (made after the saved prediction was checked) and labelled so.

Parts (--only):
  formats       where outputs put their answer (src/hotpot_content.format_flags), per model and arm, on HotpotQA
                (four arms) and on the requested and chain-of-thought arms of GSM8K-29, MATH-500 and BIG-Bench Hard-28
                (the rows of their content tables); and on GSM8K, the rows whose last box is the copied "<number>"
                placeholder, which src/content_scoring.py skips
  accuracy      arm accuracies and per-model gaps, strict and content, and the rows the content reading changes
  arm_effect    results/arm_effect/'s quantities for HotpotQA (macros aeHq...Content)
  ceiling       the protocol ceiling with the scoring axis strict, lenient and content and the headline pair
                under content (macros pceilHq...Content)
  saved         from the saved choices: the fair comparison (dstrictHq...Content, tregret...Hq...Content), both
                directions of the wrong-baseline cost (hqDdir...Content), the mechanism forecast (hqMech...Content), the
                split pilot (hqSplit...Content), the decomposition rule (hqDec...Content), the decision analysis's
                one-draw choices (JSON only); and the same functions on strict scores, checked against the stored files
  thresholds    exploratory: the decomposition and the (1 - w) choice with long defined as more than 64 tokens
                (as saved), more than 16 tokens, and at least 8 tokens beyond the same row's forced output, under strict
                and content (results/hotpot/decomposition_thresholds_post_hoc.json, beside the saved rule's files)
  rater_c       100 requested outputs on which strict and content exact match disagree, stratified by model,
                for rater C (results/scoring_audit_hotpot/ratings_rater_c_hotpot.csv; key in data/scoring_audit_hotpot/)

Outputs: results/hotpot_content/hotpot_content.json, fair_rules_evaluation_content.json, protocol_ceiling_content.json,
summary.md, summary_clean.md, tables/; results/hotpot/decomposition_evaluation_content.json and
decomposition_thresholds_post_hoc.json (+ .md, _clean.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROCESSED = os.path.join(ROOT, "data", "processed", "hotpot")
CONTENT_TABLE = os.path.join(PROCESSED, "scored_content.parquet")
VIEW = os.path.join(PROCESSED, "content_view")
STRICT_TABLE = os.path.join(PROCESSED, "scored.parquet")
MANIFESTS = os.path.join(ROOT, "manifests")
OUT = os.path.join(ROOT, "results", "hotpot_content")
HQ_OUT = os.path.join(ROOT, "results", "hotpot")
AUDIT_OUT = os.path.join(ROOT, "results", "scoring_audit_hotpot")
AUDIT_KEY = os.path.join(ROOT, "data", "scoring_audit_hotpot")
PARTS = ("formats", "accuracy", "arm_effect", "ceiling", "saved", "refit", "thresholds", "rater_c")
KEY = ["model_key", "arm", "config_id", "item_id"]
ARM_TAG = {"direct_answer": "Req", "forced_direct": "Fd", "zero_shot_cot": "Cot", "fewshot_forced": "Fs"}
RATER_C_N = 100
RATER_C_SEED = 16763816804538338054        # seed of rater C's draw
POST_HOC = ("exploratory: made after HotpotQA's saved decomposition prediction (manifests/predictions_hotpot_decomposition."
            "json) was checked against its validation draws (results/hotpot/decomposition_evaluation.json)")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def write_json(obj, path):
    from run_index_checks import jsonable
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(obj), fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def setup():
    import run_hotpot as rh
    sampler = json.load(open(rh.SAMPLER, encoding="utf-8"))
    panel, fams, _ = rh.panel_info()
    items = rh.item_ids(PROCESSED)
    disc, val = list(map(str, sampler["discovery_config_ids"])), list(map(str, sampler["validation_config_ids"]))
    return {"panel": panel, "fams": fams, "items": items, "disc": disc, "val": val}


# ============================================================================ formats


def other_task_flags(task: str) -> pd.DataFrame:
    """Format flags of the requested and chain-of-thought rows of a main task's content table (its analysed rows)."""
    import score_content as scn
    from src import d2_io, hotpot_content as hc
    con = pd.read_parquet(os.path.join(ROOT, "data", "processed", "content", f"{task}_content.parquet"),
                          columns=KEY + ["content_class"], filters=[("arm", "in", ["direct_answer", "zero_shot_cot"])])
    for k in KEY:
        con[k] = con[k].astype(str)
    panel = sorted(set(con["model_key"]))
    parts = []
    folders = [f for f in scn.folders(task, panel) if os.path.basename(os.path.dirname(f)) != "forced"
               and os.path.basename(os.path.dirname(os.path.dirname(f))) != "forced"]
    for wd in folders:
        if not os.path.isdir(wd):
            continue
        out, info = d2_io.inspect_work_dir(wd, string_item_ids=(task in scn.STRING_IDS))
        out = out[out["arm"].isin(["direct_answer", "zero_shot_cot"])]
        if out.empty:
            continue
        for k in KEY:
            out[k] = out[k].astype(str)
        j = con[con["model_key"] == info["model_key"]].merge(out[KEY + ["output_text", "finish_reason"]], on=KEY,
                                                              how="inner",
                                                              validate="one_to_one")
        fl = hc.flag_frame(j["output_text"].tolist(), hc.PLACEHOLDERS[task])
        for c in fl.columns:
            j[c] = fl[c].to_numpy()
        if task == "gsm8k":
            j["gsm8k_placeholder_reread"] = gsm8k_placeholder_reread(j)
        parts.append(j.drop(columns=["output_text", "finish_reason"]))
    tab = pd.concat(parts, ignore_index=True)
    tab = tab.drop_duplicates(KEY)
    if len(tab) != len(con):
        raise SystemExit(f"{task}: {len(tab)} outputs found for {len(con)} content rows")
    return tab


def gsm8k_placeholder_reread(j: pd.DataFrame) -> list:
    """GSM8K rows whose last complete box is the copied "<number>": the content class src/content_scoring.py gives
    when that box is skipped as it skips "<answer>" (the text re-read with "\\boxed{<number>}" written as
    "\\boxed{<answer>}", with the row's finish reason); None on the other rows. A check: the reader skips "<number>"
    itself, so the re-read equals the scored class on every row."""
    from src import content_scoring as cs, gsm8k, hotpot_task as ht, scoring
    gold = {str(int(i)): scoring.normalize_number(g) for i, g in zip(*[gsm8k.load_gsm8k("test")[c]
                                                                         for c in ("item_id", "gold_answer")])}
    out = []
    for t, i, last, fin in zip(j["output_text"], j["item_id"], j["last_box_placeholder"], j["finish_reason"]):
        t = t if isinstance(t, str) else ""
        comp, _ = ht.boxes(t)
        if not (last and comp and comp[-1].strip().lower() == "<number>"):
            out.append(None)
            continue
        t2 = t.replace("\\boxed{<number>}", "\\boxed{<answer>}")
        out.append(cs.classify(t2, fin, gold[str(i)], "gsm8k")[0])
    return out


def formats_part(R):
    from src import hotpot_content as hc
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    blocks = {}
    hq = pd.read_parquet(CONTENT_TABLE, columns=KEY + list(hc.FLAG_COLUMNS))
    tabs = {"hotpot": hq}
    for task in ("gsm8k", "math", "bbh"):
        tabs[task] = other_task_flags(task)
        print(f"formats: {task} ({len(tabs[task])} rows)", flush=True)
    for task, t in tabs.items():
        cnt = hc.flag_counts(t)
        cnt.insert(0, "task", task)
        cnt.to_csv(os.path.join(OUT, "tables", f"format_counts_{task}.csv"), index=False)
        tot = hc.flag_counts(t, by=("arm",))
        blocks[task] = {"placeholder": hc.PLACEHOLDERS[task],
                        "by_arm": {r["arm"]: {k: int(v) for k, v in r.items() if k != "arm"}
                                   for r in tot.to_dict("records")},
                        "per_model_table": rel(os.path.join(OUT, "tables", f"format_counts_{task}.csv"))}
    g = tabs["gsm8k"]
    x = g[g["gsm8k_placeholder_reread"].notna()]
    blocks["gsm8k"]["placeholder_last_box"] = {
        "rows": int(len(x)),
        "content_class_as_scored": {k: int(v) for k, v in x["content_class"].value_counts().sort_index().items()},
        "content_class_if_skipped": {k: int(v) for k, v in x["gsm8k_placeholder_reread"].value_counts().sort_index().items()},
        "would_change_to_correct": int(((x["gsm8k_placeholder_reread"] == "correct") & (x["content_class"] != "correct")).sum()),
        "by_arm": {a: int(v) for a, v in x["arm"].value_counts().sort_index().items()},
        "note": "GSM8K's answer format writes \\boxed{<number>}; "
                "src/content_scoring.py skips it as it skips '<answer>', so the re-read equals the scored class "
                "(would_change_to_correct 0)."}
    R["formats"] = {"definitions": {"placeholder_copy": "the output holds the placeholder anywhere (case-insensitive)",
                                    "last_box_placeholder": "the last complete box begins with the placeholder",
                                    "answer_place": "exclusive: usable_box, bare_box, unclosed_box, unboxed "
                                                    "(src/hotpot_content.py)",
                                    "placeholder_no_usable_box": "placeholder copied and no usable box"},
                    "tasks": blocks}


# ============================================================================ accuracy and the rows that change


def accuracy_part(R):
    t = pd.read_parquet(CONTENT_TABLE, columns=KEY + ["strict_correct", "lenient_correct", "content_correct", "strict_f1",
                                                      "content_f1", "content_source", "placeholder_copy", "answer_place",
                                                      "content_answer"])
    acc = {a: {"strict_em": float(100 * g["strict_correct"].mean()), "content_em": float(100 * g["content_correct"].mean()),
               "lenient_em": float(100 * g["lenient_correct"].mean()),
               "strict_f1": float(100 * g["strict_f1"].mean()), "content_f1": float(100 * g["content_f1"].mean()),
               "n": int(len(g))} for a, g in t.groupby("arm")}
    pm = t.groupby(["model_key", "arm"])[["strict_correct", "content_correct", "strict_f1", "content_f1"]].mean() * 100
    pm.reset_index().to_csv(os.path.join(OUT, "tables", "per_model_arm.csv"), index=False)
    gaps = {}
    for sc, col in (("strict", "strict_correct"), ("content", "content_correct")):
        w = pm[col].unstack("arm")
        gaps[sc] = {"requested_minus_forced_median": float((w["direct_answer"] - w["forced_direct"]).median()),
                    "cot_minus_forced_median": float((w["zero_shot_cot"] - w["forced_direct"]).median()),
                    "cot_minus_requested_median": float((w["zero_shot_cot"] - w["direct_answer"]).median())}
    req = t[t["arm"] == "direct_answer"]
    up = req[(req["strict_correct"] == 0) & (req["content_correct"] == 1)]
    down = req[(req["strict_correct"] == 1) & (req["content_correct"] == 0)]
    lup = req[(req["lenient_correct"] == 0) & (req["content_correct"] == 1)]
    R["accuracy"] = {"requested_lenient_wrong_content_right": {
                         "n": int(len(lup)), "lenient_right_content_wrong": int(((req["lenient_correct"] == 1)
                                                                                  & (req["content_correct"] == 0)).sum()),
                         "placeholder_copy": int(lup["placeholder_copy"].sum()),
                         "by_answer_place": {k: int(v) for k, v in lup["answer_place"].value_counts().sort_index().items()}},
                     "by_arm": acc, "per_model_gaps": gaps, "per_model_table": rel(os.path.join(OUT, "tables",
                                                                                               "per_model_arm.csv")),
                     "requested_rows_changed": {
                         "wrong_to_right": int(len(up)), "right_to_wrong": int(len(down)),
                         "wrong_to_right_placeholder_copy": int(up["placeholder_copy"].sum()),
                         "wrong_to_right_placeholder_share": float(up["placeholder_copy"].mean()) if len(up) else None,
                         "wrong_to_right_by_answer_place": {k: int(v) for k, v in up["answer_place"].value_counts().sort_index().items()},
                         "wrong_to_right_by_content_source": {k: int(v) for k, v in up["content_source"].value_counts().sort_index().items()}},
                     "content_source_by_arm": {a: {k: int(v) for k, v in g["content_source"].value_counts().sort_index().items()}
                                               for a, g in t.groupby("arm")},
                     "answers_read_as_boxed_text": int(t["content_source"].eq("answer_cue").sum())}
    # requested answers read that still hold "boxed"
    holds = req["content_answer"].fillna("").astype(str).str.contains("boxed", regex=False)
    R["accuracy"]["requested_answer_holds_boxed"] = {
        "n": int(holds.sum()), "correct": int(req.loc[holds, "content_correct"].sum()),
        "by_source": {k: int(v) for k, v in req.loc[holds, "content_source"].value_counts().sort_index().items()}}
    R["accuracy"]["requested_read_in_other_box"] = {
        k: int(req["content_source"].eq(k).sum()) for k in ("angle_box", "tag_box")}


# ============================================================================ arm effect


def _arrays3(path, ctx, units, score="em"):
    import run_hotpot as rh
    Bq, Pq = rh.arrays(rh.loader(path, "cot_req", score), units, ctx["panel"], ctx["items"])
    Bf, Pf = rh.arrays(rh.loader(path, "cot_forced", score), units, ctx["panel"], ctx["items"])
    if not np.array_equal(Pq, Pf):
        raise SystemExit("chain-of-thought arrays differ between the two contrasts")
    return Bq, Bf, Pq


def arm_effect_block(path, ctx, AE, per_model):
    """results/arm_effect/'s HotpotQA block from the table at `path` (strict_correct and strict_f1 as scored there)."""
    from scipy import stats
    from src import robustness as rb
    panel = ctx["panel"]
    breq, bfd, pc = _arrays3(path, ctx, ctx["disc"])
    req, fd, cot = breq.mean(0), bfd.mean(0), pc.mean(0)
    rmf = req - fd
    sh = np.array([per_model[k] for k in panel])
    short = np.array([k in set(AE["short_models"]) for k in panel])
    T = {"name": "HotpotQA", "M": len(panel), "reversals": rb.pair_reversals(cot - req, cot - fd),
         "readable_M": None, "reversals_readable": None,
         "pearson_long_share_vs_req_minus_forced": float(np.corrcoef(sh, rmf)[0, 1]),
         "spearman_long_share_vs_req_minus_forced": float(stats.spearmanr(sh, rmf).statistic),
         "short_models": list(AE["short_models"]),
         "short_models_req_minus_forced_range": [float(rmf[short].min()), float(rmf[short].max())],
         "short_models_req_minus_forced_mean": float(rmf[short].mean()), "mean_req_minus_forced": float(rmf.mean()),
         "forced_cap_tokens": AE["forced_cap_tokens"],
         "per_model": {k: {"requested": float(req[j]), "forced": float(fd[j]), "cot": float(cot[j])}
                       for j, k in enumerate(panel)}}
    f = _arrays3(path, ctx, ctx["disc"], "f1")
    fr, ff, fc = (x.mean(0) for x in f)
    frmf = fr - ff
    T["token_f1"] = {"scoring": "token F1 of the same reading, pp", "reversals": rb.pair_reversals(fc - fr, fc - ff),
                     "mean_req_minus_forced": float(frmf.mean()),
                     "short_models_req_minus_forced_range": [float(frmf[short].min()), float(frmf[short].max())],
                     "short_models_req_minus_forced_mean": float(frmf[short].mean()), "short_models_n": int(short.sum()),
                     "short_models_req_minus_forced_below_zero": int((frmf[short] < 0).sum())}
    return T


def rh_arrays_value(path, ctx, contrast, value):
    import run_hotpot as rh
    return rh.arrays(rh.loader(path, contrast), ctx["disc"], ctx["panel"], ctx["items"], value)


def pooled_reversals(path, ctx):
    """Pooled reversals: model means over every distinct configuration
    of both splits, each once; reversed pairs between chain of thought's gain over the forced and over the requested
    arm, with ties left out of the denominator."""
    from src import robustness as rb
    units = sorted(set(ctx["disc"]) | set(ctx["val"]))
    breq, bfd, pc = _arrays3(path, ctx, units)
    v = rb.pair_reversals((pc - breq).mean(0), (pc - bfd).mean(0))
    return {"reversed": v["reversed"], "pairs": v["pairs"], "ties": v["ties"], "pairs_without_ties": v["pairs"] - v["ties"],
            "configurations": len(units)}


def arm_effect_part(R, ctx):
    AE = json.load(open(os.path.join(ROOT, "results", "arm_effect", "arm_effect.json"), encoding="utf-8"))["tasks"]["hotpot"]
    pm = pd.read_csv(os.path.join(ROOT, "results", "arm_effect", "tables", "per_model.csv"))
    pm = pm[pm["task"] == "hotpot"].set_index("model_key")["requested_long_share_pct"].to_dict()
    vpath = os.path.join(VIEW, "scored.parquet")
    Ts = arm_effect_block(STRICT_TABLE, ctx, AE, pm)
    Tc = arm_effect_block(vpath, ctx, AE, pm)
    check = {"reversed": (Ts["reversals"]["reversed"], AE["reversals"]["reversed"]),
             "mean_req_minus_forced": (Ts["mean_req_minus_forced"], AE["mean_req_minus_forced"]),
             "f1_reversed": (Ts["token_f1"]["reversals"]["reversed"], AE["token_f1"]["reversals"]["reversed"])}
    from src import robustness as rb
    lq, lf = (rh_arrays_value(STRICT_TABLE, ctx, c, "lenient_correct") for c in ("cot_req", "cot_forced"))
    lreq, lfd, lcot = lq[0].mean(0), lf[0].mean(0), lq[1].mean(0)
    lenient = {"reversals": rb.pair_reversals(lcot - lreq, lcot - lfd), "mean_req_minus_forced": float((lreq - lfd).mean()),
               "note": "lenient exact match as run (hotpot-boxed/1.0.0), for context"}
    R["arm_effect"] = {"content": Tc, "lenient_as_run": lenient, "strict_recomputed": {k: v for k, v in Ts.items() if k != "per_model"},
                       "check_against_results_arm_effect": {k: {"here": a, "stored": b, "equal": bool(abs(a - b) < 1e-9)}
                                                           for k, (a, b) in check.items()},
                       "pooled_all_configurations": {"strict": pooled_reversals(STRICT_TABLE, ctx),
                                                     "content": pooled_reversals(vpath, ctx)},
                       "note": "model means over the 24 discovery draws (results/arm_effect/'s definition); the "
                               "short-answer stratum and the long shares are those of results/arm_effect/ (lengths "
                               "only, so they do not move with the scorer)"}


# ============================================================================ protocol ceiling


def ceiling_part(R, ctx):
    import run_hotpot as rh
    import run_protocol_ceiling as rpc
    from src import sampling
    units = list(dict.fromkeys(ctx["disc"] + ctx["val"]))
    cols = {"strict": "strict_correct", "lenient": "lenient_correct", "content": "content_correct"}
    # content scores come in through the view (strict_correct holding content_correct), as every other part reads them
    src = {"strict": (STRICT_TABLE, "strict_correct"), "lenient": (STRICT_TABLE, "lenient_correct"),
           "content": (os.path.join(VIEW, "scored.parquet"), "strict_correct")}
    D = {}
    for sc, (path, col) in src.items():
        Bq, Pq = rh.arrays(rh.loader(path, "cot_req"), units, ctx["panel"], ctx["items"], col)
        Bf, Pf = rh.arrays(rh.loader(path, "cot_forced"), units, ctx["panel"], ctx["items"], col)
        if not np.array_equal(Pq, Pf):
            raise SystemExit("chain-of-thought arrays differ between the two contrasts")
        D[sc] = {"requested": Bq, "forced": Bf, "cot": Pq}
    res = rpc.task_result("hotpot", sampling.rng_for("protocol-ceiling", "hotpot_content"),
                          loaded=(D, units, list(ctx["panel"]), len(ctx["items"])), head_scoring="content",
                          boot_rng=sampling.rng_for("protocol-ceiling", "hotpot_content", "boot"),
                          scoring_columns=cols, title="HotpotQA (content reading)")
    full = {"analysis": "protocol ceiling, HotpotQA under the content reading (scripts/run_hotpot_content.py)",
            "note": "scoring axis strict, lenient and content (the main tasks' axis); headline pair content, points, "
                    "requested against forced; streams rng_for('protocol-ceiling', 'hotpot_content')",
            "Ks": list(rpc.KS), "n_splits": rpc.N_SPLITS, "tasks": {"hotpot": res}}
    write_json(full, os.path.join(OUT, "protocol_ceiling_content.json"))
    R["protocol_ceiling"] = {"file": rel(os.path.join(OUT, "protocol_ceiling_content.json")), "note": full["note"],
                             "headline": {k: res["headline"][k] for k in ("protocols", "plugin", "posterior",
                                                                          "plugin_bootstrap")},
                             "within_requested": res["headline"]["within_requested"],
                             "between_same_draws": res["headline"]["between_same_draws"],
                             "axes": {a: {k: v for k, v in x.items() if k in ("pairs", "protocols", "limit_plugin",
                                                                               "limit_posterior")}
                                      for a, x in res["axes"].items()}}


# ============================================================================ the saved choices


def _leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{p}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _leaves(v, f"{p}/{i}")
    else:
        yield p, o


def compare_stored(new, path, skip=("created",)):
    """Leaves of `new` (made jsonable) against a stored JSON file: counts of equal, different (beyond 1e-9 relative),
    and missing leaves, with up to five differing paths."""
    from run_index_checks import jsonable
    old = dict(_leaves(json.load(open(path, encoding="utf-8"))))
    nw = dict(_leaves(json.loads(json.dumps(jsonable(new)))))
    diff, eq, miss = [], 0, 0
    for k, v in old.items():
        if any(s in k.split("/") for s in skip):
            continue
        if k not in nw:
            miss += 1
            continue
        w = nw[k]
        same = (v == w) or (isinstance(v, (int, float)) and isinstance(w, (int, float)) and not isinstance(v, bool)
                            and (abs(v - w) <= 1e-9 * max(1.0, abs(v)) or (v != v and w != w)))
        eq += same
        if not same:
            diff.append(k)
    return {"file": rel(path), "equal": eq, "different": len(diff), "missing": miss, "examples": diff[:5]}


def saved_part(R, ctx, check_strict=True):
    import hotpot_fair as hf
    import run_baseline_mixture as rbm
    import run_decision as rd
    import run_decision_directions as rdd
    import run_hotpot as rh
    import run_split_pilot as rsp
    panel, fams, items, disc, val = (ctx[k] for k in ("panel", "fams", "items", "disc", "val"))
    out = {}
    sources = {"content": (os.path.join(VIEW, "scored.parquet"), VIEW)}
    if check_strict:
        sources["strict"] = (STRICT_TABLE, PROCESSED)
    for sc, (path, proc) in sources.items():
        t0 = time.time()
        V = {c: rh.arrays(rh.loader(path, c), val, panel, items) for c in rh.CONTRASTS}
        vconfigs, Sv = hf.dense(path, panel, val, items)
        F = hf.evaluate_all(hf.load_saved(MANIFESTS, panel), disc, val, V, Sv, vconfigs, len(items), rh.contrast_arms())
        blk = {"fair": F,
               "decision_directions": rdd.hotpot_directions(MANIFESTS, panel, fams, disc, val, V, Sv, vconfigs, len(items)),
               "mechanism_forecast": rbm.evaluate_hotpot_forecast(MANIFESTS, panel, V),
               "split_pilot": rsp.evaluate_hotpot(MANIFESTS, panel, fams, disc, val, Sv, vconfigs, len(items)),
               "decomposition": rbm.evaluate_hotpot_decomposition(MANIFESTS, proc, panel, disc, val),
               "decision": {}}
        for c, (_, _, label) in rh.CONTRASTS.items():
            rec, W = rd.load_choices(label, MANIFESTS)
            blk["decision"][c] = rd.evaluate(label, rec, W, (val, panel, *V[c]))
        out[sc] = blk
        print(f"saved choices, {sc}: done ({time.time() - t0:.0f} s)", flush=True)
    c = out["content"]
    note = ("saved choices (strict exact match, discovery draws, saved before validation was scored) valued on "
            "content-scored validation draws")
    from src import hotpot_task as ht
    write_json({**c["fair"], "scoring": f"content exact match ({ht.CONTENT_SCORER_VERSION})", "note": note},
               os.path.join(OUT, "fair_rules_evaluation_content.json"))
    write_json({**c["decomposition"], "scoring": f"content exact match ({ht.CONTENT_SCORER_VERSION})",
                "label": "exploratory: the saved rule (strict exact match) evaluated under content scoring after the "
                         "saved prediction was checked; results/hotpot/decomposition_evaluation.json is unchanged"},
               os.path.join(HQ_OUT, "decomposition_evaluation_content.json"))
    R["saved_choices"] = {"note": note, "fair_file": rel(os.path.join(OUT, "fair_rules_evaluation_content.json")),
                          "decomposition_file": rel(os.path.join(HQ_OUT, "decomposition_evaluation_content.json")),
                          **{k: v for k, v in c.items() if k != "fair"}}
    if check_strict:
        s = out["strict"]
        R["saved_choices"]["strict_reproduction"] = [
            compare_stored(s["fair"], os.path.join(HQ_OUT, "fair_rules_evaluation.json")),
            compare_stored(s["decision_directions"], os.path.join(HQ_OUT, "decision_directions.json")),
            compare_stored(s["mechanism_forecast"], os.path.join(HQ_OUT, "mechanism_forecast_evaluation.json")),
            compare_stored(s["split_pilot"], os.path.join(HQ_OUT, "split_pilot_evaluation.json")),
            compare_stored(s["decomposition"], os.path.join(HQ_OUT, "decomposition_evaluation.json"))]


# ============================================================================ exploratory: choices refitted under content


def refit_part(R, ctx):
    """Exploratory: the discovery stage's choices refitted on content-scored discovery draws with the discovery stage's
    own functions (one-draw decision choices, the strict decision design's 100-item pilots, held-out-family and
    cross-benchmark rules, the split pilot), written to results/hotpot_content/post_hoc_choices/ (never manifests/),
    and valued on content-scored validation draws. The cross-benchmark file (saved from the other
    benchmarks alone) is copied in unchanged, as save_discovery needs it."""
    import shutil
    import hotpot_fair as hf
    import run_decision as rd
    import run_decision_directions as rdd
    import run_hotpot as rh
    import run_split_pilot as rsp
    panel, fams, items, disc, val = (ctx[k] for k in ("panel", "fams", "items", "disc", "val"))
    pdir = os.path.join(OUT, "post_hoc_choices")
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
    os.makedirs(pdir)
    for f in hf.files(hf.CROSS, MANIFESTS):
        shutil.copyfile(f, os.path.join(pdir, os.path.basename(f)))
    vpath = os.path.join(VIEW, "scored.parquet")
    D = {c: rh.arrays(rh.loader(vpath, c), disc, panel, items) for c in rh.CONTRASTS}
    for c in rh.CONTRASTS:
        rh.freeze_decision(c, disc, panel, D[c][0], D[c][1], pdir, overwrite=True)
    hf.save_discovery(vpath, disc, panel, fams, hf.panel_eci(panel), items, rh.contrast_arms(), pdir, True, D=D)
    rsp.save_hotpot(vpath, disc, panel, items, pdir)
    V = {c: rh.arrays(rh.loader(vpath, c), val, panel, items) for c in rh.CONTRASTS}
    vconfigs, Sv = hf.dense(vpath, panel, val, items)
    F = hf.evaluate_all(hf.load_saved(pdir, panel), disc, val, V, Sv, vconfigs, len(items), rh.contrast_arms())
    label = ("exploratory: choices refitted on content-scored discovery draws (after validation had been scored and the "
             "saved choices checked), valued on content-scored validation draws")
    write_json({**F, "label": label}, os.path.join(OUT, "fair_rules_evaluation_content_post_hoc.json"))
    blk = {"label": label, "choices_dir": rel(pdir),
           "decision_directions": rdd.hotpot_directions(pdir, panel, fams, disc, val, V, Sv, vconfigs, len(items)),
           "split_pilot": rsp.evaluate_hotpot(pdir, panel, fams, disc, val, Sv, vconfigs, len(items)),
           "fair_file": rel(os.path.join(OUT, "fair_rules_evaluation_content_post_hoc.json")), "decision": {}}
    for c, (_, _, lab) in rh.CONTRASTS.items():
        rec, W = rd.load_choices(lab, pdir)
        blk["decision"][c] = rd.evaluate(lab, rec, W, (val, panel, *V[c]))
    R["refit_post_hoc"] = blk


# ============================================================================ thresholds (exploratory)


def wide_table(ctx) -> pd.DataFrame:
    """One row per (model, configuration, item): token counts of the requested and forced outputs and the strict and
    content exact match of the requested, forced and chain-of-thought arms; long flags of every definition."""
    from src import hotpot_content as hc
    t = pd.read_parquet(CONTENT_TABLE, columns=KEY + ["strict_correct", "content_correct", "n_output_tokens"],
                        filters=[("arm", "in", ["direct_answer", "forced_direct", "zero_shot_cot"])])
    for k in KEY:
        t[k] = t[k].astype(str)
    t = t[t["model_key"].isin(set(ctx["panel"])) & t["item_id"].isin(set(ctx["items"]))]
    k3 = ["model_key", "config_id", "item_id"]
    wide = None
    for a, name in (("direct_answer", "requested"), ("forced_direct", "forced"), ("zero_shot_cot", "cot")):
        x = t[t["arm"] == a][k3 + ["strict_correct", "content_correct", "n_output_tokens"]].rename(
            columns={"strict_correct": f"{name}_strict", "content_correct": f"{name}_content",
                     "n_output_tokens": f"{name}_tokens"})
        wide = x if wide is None else wide.merge(x, on=k3, how="inner", validate="one_to_one")
    n_expect = len(ctx["panel"]) * len(set(ctx["disc"]) | set(ctx["val"])) * len(ctx["items"])
    if len(wide) != n_expect:
        raise SystemExit(f"{len(wide)} aligned rows, expected {n_expect}")
    for d in list(hc.LONG_DEFINITIONS) + list(hc.CHECK_DEFINITIONS):
        wide[f"long_{d}"] = hc.long_flags(d, wide["requested_tokens"], wide["forced_tokens"])
    return wide


def thresholds_part(R, ctx):
    from src import baseline_mixture as bm, hotpot_content as hc
    wide = wide_table(ctx)
    V = {}
    for d in list(hc.LONG_DEFINITIONS) + list(hc.CHECK_DEFINITIONS):
        V[d] = {sc: hc.decomposition_variant(wide, ctx["panel"], f"long_{d}", ctx["disc"], ctx["val"], sc)
                for sc in ("strict", "content")}
        print(f"thresholds: {d} done", flush=True)
    saved = bm.load_forecast(os.path.join(MANIFESTS, "predictions_hotpot_decomposition.json"))
    ev = json.load(open(os.path.join(HQ_OUT, "decomposition_evaluation.json"), encoding="utf-8"))
    v64 = V["cap64"]["strict"]
    checks = {"validation_variance_share_equals_saved_evaluation": {
                  "here": v64["sets"]["validation"]["variance_share_long"],
                  "stored": ev["statements"]["variance_share"]["observed"]},
              "refit_choices_equal_saved_choices": {k: bool(np.allclose(v64["choice"]["weights"][k],
                                                                        saved["choices"][k]["weights"]))
                                                    for k in saved["choices"]},
              "regret_equals_saved_evaluation": {
                  "corrected": (v64["choice"]["validation_means"]["regret"]["corrected"],
                                ev["statements"]["choice"]["regret_corrected"]),
                  "forced_gain": (v64["choice"]["validation_means"]["regret"]["forced_gain"],
                                  ev["statements"]["choice"]["regret_forced_gain"])}}
    body = {"label": POST_HOC, "analysis": "decomposition of the baseline shift and the (1 - w) choice under three "
                                           "definitions of a long requested output, strict and content exact match",
            "definitions": hc.LONG_DEFINITIONS, "check_definitions": hc.CHECK_DEFINITIONS,
            "saved_rule": {"path": "manifests/predictions_hotpot_decomposition.json", "sha256": saved["sha256"],
                           "evaluation": "results/hotpot/decomposition_evaluation.json (unchanged)"},
            "sets": "all_distinct: every distinct configuration of both splits once; discovery and validation: each "
                    "configuration weighted by its draws",
            "choices": "refitted on the discovery draws for each definition and score (exploratory), valued under the "
                       "requested arm on the 24 validation draws' means and on the fresh validation configurations",
            "checks_cap64_strict": checks, "variants": V}
    write_json(body, os.path.join(HQ_OUT, "decomposition_thresholds_post_hoc.json"))
    txt = "\n".join(threshold_lines(body)) + "\n"
    for name in ("decomposition_thresholds_post_hoc.md", "decomposition_thresholds_post_hoc_clean.md"):
        with open(os.path.join(HQ_OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    R["thresholds"] = {"file": rel(os.path.join(HQ_OUT, "decomposition_thresholds_post_hoc.json")), "label": POST_HOC,
                       "checks_cap64_strict": checks, "table": threshold_rows(body)}


def threshold_rows(body) -> list:
    rows = []
    for d, by in body["variants"].items():
        for sc, v in by.items():
            a, va = v["sets"]["all_distinct"], v["sets"]["validation"]
            ch, fr = v["choice"], v["choice"]["fresh_configurations"]
            ma, md = v["mechanism"]["all_distinct"], v["mechanism"]["discovery"]
            rows.append({"definition": d, "score": sc, "w_median_discovery": v["w_discovery_median"],
                         "w_median_all": v["w_all_distinct_median"],
                         "var_share_long_all": a["variance_share_long"], "var_share_long_validation": va["variance_share_long"],
                         "var_share_long_discovery": v["sets"]["discovery"]["variance_share_long"],
                         "short_gap_median_all": (a["short_gap_pp"] or {}).get("median"),
                         "short_part_median_all": float(np.median(a["per_model"]["part_short"])),
                         "shift_w_spearman_all": a["closed_fraction"]["corr_shift_with_w_all_models"]["spearman"],
                         "rev_forced_all": f"{ma['observed_vs_forced']['reversed']}/{ma['observed_vs_forced']['pairs']}",
                         "rev_corrected_all": f"{ma['observed_vs_corrected']['reversed']}/{ma['observed_vs_corrected']['pairs']}",
                         "rms_forced_all": ma["rms_forced_pp"], "rms_corrected_all": ma["rms_corrected_pp"],
                         "rev_forced_disc": md["observed_vs_forced"]["reversed"],
                         "rev_corrected_disc": md["observed_vs_corrected"]["reversed"],
                         "regret_corrected": ch["validation_means"]["regret"]["corrected"],
                         "regret_forced_gain": ch["validation_means"]["regret"]["forced_gain"],
                         "regret_short": ch["validation_means"]["regret"]["corrected_short"],
                         "fresh_diff": (fr.get("paired_corrected_minus_forced") or {}).get("estimate")})
    return rows


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}".replace("-", "−")


def threshold_lines(body) -> list:
    L = ["# HotpotQA: the decomposition and the (1 − w) choice under three definitions of a long answer (exploratory)", "",
         "These analyses were made after HotpotQA's saved decomposition prediction "
         "(`manifests/predictions_hotpot_decomposition.json`) had been checked against its validation draws "
         "(`results/hotpot/decomposition_evaluation.json`); both files are unchanged. Choices are refitted on the "
         "discovery draws for each definition and score.", "",
         "Definitions:"] + [f"- `{k}`: {v}." for k, v in body["definitions"].items()] + [
         f"- `{k}` (check only): {v}." for k, v in body["check_definitions"].items()] + ["",
         "| Long | Score | Median w | Long share, all configs | Long share, validation | Short gap (pp) | "
         "Short part of the shift (pp) | "
         "Reversed, forced vs requested gain | Reversed after (1 − w) | RMS forced / corrected (pp) | "
         "Regret: (1 − w) / forced gain / short form (pp) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in threshold_rows(body):
        L.append(f"| {r['definition']} | {r['score']} | {_f(r['w_median_all'], 3)} | {_f(r['var_share_long_all'])} | "
                 f"{_f(r['var_share_long_validation'])} | {_f(r['short_gap_median_all'])} | {_f(r['short_part_median_all'])} | "
                 f"{r['rev_forced_all']} | "
                 f"{r['rev_corrected_all']} | {_f(r['rms_forced_all'], 1)} / {_f(r['rms_corrected_all'], 1)} | "
                 f"{_f(r['regret_corrected'], 1)} / {_f(r['regret_forced_gain'], 1)} / {_f(r['regret_short'], 1)} |")
    L += ["", "Median w, the long share, the short gap, the reversals and the RMS are on every distinct configuration "
              "once; reversals count tied pairs in the denominator. The short gap is the median "
              "over models of requested minus forced accuracy within the short rows (R_S − F_S, as "
              "results/baseline_mixture/ reports it); the short part of the shift is the median of the short rows' "
              "share of the shift, sum over short rows of (requested − forced) divided by all the model's rows (short_gap_med). Regret is under the requested arm on the means of the 24 validation draws."]
    c = body["checks_cap64_strict"]
    L += ["", f"Check, `cap64` strict against the saved rule: validation long share "
              f"{_f(c['validation_variance_share_equals_saved_evaluation']['here'], 4)} against the stored "
              f"{_f(c['validation_variance_share_equals_saved_evaluation']['stored'], 4)}; refitted choices equal to the "
              f"saved ones: {c['refit_choices_equal_saved_choices']}."]
    return L


# ============================================================================ rater C


def rater_c_part(R, ctx):
    from run_human_audit import RATER3_COLUMNS
    from src import d2_io, hotpot_content as hc, hotpot_d2
    t = pd.read_parquet(CONTENT_TABLE, columns=KEY + ["request_id", "strict_correct", "content_correct", "content_answer",
                                                      "content_source", "placeholder_copy", "answer_place"],
                        filters=[("arm", "==", "direct_answer")])
    dis = t[t["strict_correct"] != t["content_correct"]].sort_values(KEY).reset_index(drop=True)
    pick = hc.stratified_draw(dis, RATER_C_N, "model_key", RATER_C_SEED)
    meta = json.load(open(os.path.join(PROCESSED, "scored.json"), encoding="utf-8"))["folders"]
    texts = {}
    for m in sorted(set(pick["model_key"])):
        want = set(pick.loc[pick["model_key"] == m, "request_id"])
        for b in ("A", "B"):
            fo = meta.get(f"{m}/{b}")
            if not fo:
                continue
            out, _ = d2_io.inspect_work_dir(os.path.join(ROOT, fo["path"]), string_item_ids=True)
            out = out[out["request_id"].isin(want)]
            texts.update(dict(zip(out["request_id"], out["output_text"])))
    if set(pick["request_id"]) - set(texts):
        raise SystemExit("some drawn outputs were not found")
    hotpot_d2.load_items("target")   # checks the pinned table against its manifest
    items = pd.read_parquet(os.path.join(PROCESSED, "items_target.parquet"))
    q = dict(zip(items["item_id"].astype(str), items["hotpot_question"].astype(str)))
    g = dict(zip(items["item_id"].astype(str), items["gold_answer"].astype(str)))
    ids = [f"H{j + 1:03d}" for j in range(len(pick))]
    csv = pd.DataFrame({"order": range(1, len(pick) + 1), "audit_id": ids,
                        "question": [q[i] for i in pick["item_id"]], "reference_answer": [g[i] for i in pick["item_id"]],
                        "model_output": [texts[r] for r in pick["request_id"]], "verdict": "", "note": ""})
    csv = csv[list(RATER3_COLUMNS)]
    key = pick.assign(audit_id=ids)[["audit_id"] + KEY + ["request_id", "strict_correct", "content_correct",
                                                           "content_answer", "content_source", "placeholder_copy",
                                                           "answer_place"]]
    os.makedirs(AUDIT_OUT, exist_ok=True)
    os.makedirs(AUDIT_KEY, exist_ok=True)
    cpath = os.path.join(AUDIT_OUT, "ratings_rater_c_hotpot.csv")
    kpath = os.path.join(AUDIT_KEY, "key_rater_c_hotpot.csv")
    csv.to_csv(cpath, index=False, encoding="utf-8-sig")
    key.to_csv(kpath, index=False, encoding="utf-8")
    alloc = pick["model_key"].value_counts().sort_index()
    info = {"csv": rel(cpath), "key": rel(kpath) + " (git-ignored: data/)", "n": int(len(pick)),
            "population": {"requested_rows_disagreeing": int(len(dis)),
                           "strict_right_content_wrong": int(((dis["strict_correct"] == 1) & (dis["content_correct"] == 0)).sum()),
                           "models_with_disagreements": int(dis["model_key"].nunique())},
            "rule": "requested-direct outputs (both splits, every configuration) on which strict and content exact match "
                    "disagree; equal allocation over the models that have any, each capped at what it holds, places a "
                    "model cannot fill passed on one at a time in a seeded order (src/hotpot_content.allocate); rows "
                    f"drawn without replacement within each model, numpy default_rng({RATER_C_SEED}); "
                    "listed in a seeded random order",
            "columns": list(RATER3_COLUMNS),
            "shown": "the HotpotQA question (without the passages), the gold answer and the full output; no scorer "
                     "verdict, no model name, no AI reading; verdict to fill: gold, other, none or unclear "
                     "(scripts/run_human_audit.py's vocabulary)",
            "models_drawn": int(len(alloc)), "per_model_min": int(alloc.min()), "per_model_max": int(alloc.max())}
    write_json(info, os.path.join(AUDIT_OUT, "draw_rater_c_hotpot.json"))
    R["rater_c"] = info


# ============================================================================ summary


def summary(R) -> str:
    from src import hotpot_task as ht
    hb = R.get("accuracy", {}).get("requested_answer_holds_boxed")
    L = ["# HotpotQA under the content reading", "",
         f"Content reading `{ht.CONTENT_SCORER_VERSION}` (`src/hotpot_task.content_read`, rules in "
         "`manifests/hotpot_content_scorer.json`); strict and lenient exact match are those of `hotpot-boxed/1.0.0`, "
         "unchanged. Every choice valued here was saved from strict scores on the discovery draws before validation "
         "was scored; under content scoring the same choices are valued on content-scored validation draws. "
         "Refits under content scoring and the threshold analyses are exploratory.", ""]
    A = R.get("accuracy")
    if A:
        L += ["## Arm accuracy (all rows)", "", "| Arm | Strict EM | Lenient EM | Content EM | Strict F1 | Content F1 |",
              "|---|---|---|---|---|---|"]
        for a, x in A["by_arm"].items():
            L.append(f"| {a} | {_f(x['strict_em'], 1)} | {_f(x.get('lenient_em'), 1)} | {_f(x['content_em'], 1)} | "
                     f"{_f(x['strict_f1'], 1)} | {_f(x['content_f1'], 1)} |")
        g, c = A["per_model_gaps"], A["requested_rows_changed"]
        L += ["", f"Median per-model requested − forced: strict {_f(g['strict']['requested_minus_forced_median'], 1)} pp, "
                  f"content {_f(g['content']['requested_minus_forced_median'], 1)} pp. Chain of thought − forced: strict "
                  f"{_f(g['strict']['cot_minus_forced_median'], 1)}, content {_f(g['content']['cot_minus_forced_median'], 1)} pp.",
              f"Requested rows wrong under strict and right under content: {c['wrong_to_right']:,} "
              f"({_f(100 * (c['wrong_to_right_placeholder_share'] or 0), 1)}% copy the placeholder); right under strict "
              f"and wrong under content: {c['right_to_wrong']}.", ""]
    F = R.get("formats")
    if F:
        L += ["## Where outputs put their answer", "",
              "| Task | Arm | Outputs | Placeholder copied | Last box is the placeholder | Usable box | Box without "
              "backslash | Unclosed box | Unboxed |", "|---|---|---|---|---|---|---|---|---|"]
        for t, b in F["tasks"].items():
            for a, x in b["by_arm"].items():
                n = x["n"]
                pc = lambda k: f"{x[k]:,} ({_f(100 * x[k] / n, 1)}%)"  # noqa: E731
                L.append(f"| {t} | {a} | {n:,} | {pc('placeholder_copy')} | {pc('last_box_placeholder')} | "
                         f"{pc('usable_box')} | {pc('bare_box')} | {pc('unclosed_box')} | {pc('unboxed')} |")
        g = F["tasks"]["gsm8k"].get("placeholder_last_box")
        if g:
            L += ["", f"GSM8K's answer format writes `\\boxed{{<number>}}`: {g['rows']:,} analysed GSM8K rows end with a "
                      "copied `<number>` box. `src/content_scoring.py` skips it as it "
                      "skips `<answer>`. Re-read with the box written as `<answer>`, "
                      f"{g['would_change_to_correct']:,} of them read differently from the scored class."]
        L += ["", "Per model: `results/hotpot_content/tables/format_counts_<task>.csv`.", ""]
    E = R.get("arm_effect")
    if E:
        s, c = E["strict_recomputed"], E["content"]
        L += ["## Arm effect (discovery means)", "",
              f"- Reversed pairs, chain of thought's gain over the requested against over the forced arm: strict "
              f"{s['reversals']['reversed']} of {s['reversals']['pairs']}, content {c['reversals']['reversed']} of "
              f"{c['reversals']['pairs']} ({c['reversals']['ties']} tied).",
              f"- Mean requested − forced: strict {_f(s['mean_req_minus_forced'], 1)}, content "
              f"{_f(c['mean_req_minus_forced'], 1)} pp; short-answer models ({len(c['short_models'])}): content mean "
              f"{_f(c['short_models_req_minus_forced_mean'], 1)} pp.",
              f"- Lenient exact match as run: {E['lenient_as_run']['reversals']['reversed']} of "
              f"{E['lenient_as_run']['reversals']['pairs']} reversed; mean requested − forced "
              f"{_f(E['lenient_as_run']['mean_req_minus_forced'], 1)} pp." if E.get("lenient_as_run") else "",
              f"- Pooled over every configuration: strict "
              f"{E['pooled_all_configurations']['strict']['reversed']}/{E['pooled_all_configurations']['strict']['pairs_without_ties']}, "
              f"content {E['pooled_all_configurations']['content']['reversed']}/"
              f"{E['pooled_all_configurations']['content']['pairs_without_ties']} (ties left out).", ""]
    P = R.get("protocol_ceiling")
    if P:
        h = P["headline"]
        L += ["## Protocol ceiling, content headline", "",
              f"Arm-switch limit (content, points): plug-in {_f(h['plugin'], 3)}, posterior {_f(h['posterior'], 3)}; "
              f"within the requested protocol at K = 1 "
              f"{_f(P['within_requested'].get('1', P['within_requested'].get(1)), 3)}.", ""]
    S = R.get("saved_choices")
    if S:
        d = S["decision_directions"]["one_draw"]["rules"].get("own_1") or {}
        sp = S["split_pilot"]["primary"]
        dc_ = S["decomposition"]["statements"]
        L += ["## Saved choices valued under content", "",
              f"- Wrong-baseline cost, one-draw own gain: requested → forced "
              f"{_f((d.get('requested_to_forced') or {}).get('estimate'))}, forced → requested "
              f"{_f((d.get('forced_to_requested') or {}).get('estimate'))} pp per chosen model.",
              f"- Split pilot (fresh configurations): advantage {_f(sp['advantage'])} pp; holds: {sp['holds']}.",
              f"- Decomposition rule: validation long share {_f(dc_['variance_share']['observed'], 3)} (threshold "
              f"{dc_['variance_share']['threshold']}), holds {dc_['variance_share']['holds']}; regret by (1 − w) "
              f"{_f(dc_['choice']['regret_corrected'], 1)} against forced gain {_f(dc_['choice']['regret_forced_gain'], 1)} pp, "
              f"holds {dc_['choice']['holds']}.", ""]
        if S.get("strict_reproduction"):
            L += ["Strict scores through the same code, against the stored files:", ""]
            for x in S["strict_reproduction"]:
                L.append(f"- `{x['file']}`: {x['equal']} leaves equal, {x['different']} different, {x['missing']} missing.")
            L.append("")
    Q = R.get("refit_post_hoc")
    if Q:
        d = Q["decision_directions"]["one_draw"]["rules"].get("own_1") or {}
        sp = Q["split_pilot"]["primary"]
        L += ["## Exploratory: choices refitted on content-scored discovery draws", "",
              f"Choices in `{Q['choices_dir']}` (never `manifests/`), valued on content-scored validation draws.", "",
              f"- Wrong-baseline cost, one-draw own gain: requested → forced "
              f"{_f((d.get('requested_to_forced') or {}).get('estimate'))}, forced → requested "
              f"{_f((d.get('forced_to_requested') or {}).get('estimate'))} pp per chosen model.",
              f"- Split pilot (fresh configurations): advantage {_f(sp['advantage'])} pp; holds: {sp['holds']}.",
              f"- Fair comparison: `{Q['fair_file']}`.", ""]
    T = R.get("thresholds")
    if T:
        L += ["## Thresholds (exploratory)", "", f"`{T['file']}` and its summary beside it.", ""]
    C = R.get("rater_c")
    if C:
        L += ["## Rater C", "", f"{C['n']} outputs in `{C['csv']}` (key `{C['key']}`), from "
                                f"{C['population']['requested_rows_disagreeing']:,} disagreeing requested outputs over "
                                f"{C['population']['models_with_disagreements']} models; {C['per_model_min']} to "
                                f"{C['per_model_max']} per model.", ""]
    L += ["## Known residual of the content reading", "",
          "Every placeholder (a `boxed<answer>` without braces, an empty box, a box holding the word Answer) is "
          "removed by one rule (`src/content_scoring.is_placeholder`). A box written with angle brackets in place of "
          "braces (`\\boxed<Odysseus>`) or as a tag (`<boxed>Liguria</boxed>`) is read as a box holding its content, "
          "unless the content is a placeholder."
          + ("" if not hb else
             f" Requested outputs whose answer read still holds `boxed`: {hb['n']:,} ({hb['correct']:,} correct; "
             + ", ".join(f"{k} {v:,}" for k, v in hb["by_source"].items()) + "); read from a box of other brackets: "
             + ", ".join(f"{k} {v:,}" for k, v in R.get("accuracy", {}).get("requested_read_in_other_box", {}).items())
             + ".")
          + residual_words(R.get("residual_tags")), ""]
    return "\n".join(L) + "\n"


RESIDUAL_TAGS = os.path.join(ROOT, "results", "hotpot_content", "hotpot_residual.json")  # counted under hotpot-content/1.5.0


def residual_words(Z) -> str:
    """The two residuals the reader leaves as they are, with their definitions, from
    results/hotpot_content/hotpot_residual.json (counted under the current reader on every main-panel reply). The
    lone tags are counted first by the reader's own tag pattern, the count with paired full tags beside it."""
    if not Z:
        return ""
    lo, ph, rd = Z["lone_any"], Z["paren_placeholder"], Z.get("lone_reader")
    arms = "; ".join(f"{k} {v:,}" for k, v in lo["by_model_arm"].items())
    if rd:
        rarms = "; ".join(f"{k} {v:,}" for k, v in rd["by_model_arm"].items())
        s = (f" Left as they are under `{Z['scorer']}`: by the reader's own tag pattern (an opening `<boxed>`, a "
             f"closing `</boxed` whose `>` is optional, a pair only on one line), {rd['replies']:,} replies hold a "
             f"tag the reader does not take into a box ({rarms}): {rd['correct']:,} correct, "
             f"{rd['would_match_lone_dropped']:,} would match if the tag were dropped. Counted with both tags written "
             f"in full and paired across lines (an opening tag with no closing tag after it before the next opening "
             f"one, or a closing tag with no opening tag since the last closing one), the count is "
             f"{lo['replies']:,} ({arms}); the count turns on the definition.")
    else:
        s = (f" Left as they are under `{Z['scorer']}`: {lo['replies']:,} replies hold a lone `<boxed>` or "
             f"`</boxed>` tag, one the reader cannot pair: an opening tag with no closing tag after it before the next "
             f"opening one, or a closing tag with no opening tag since the last closing one ({arms}). A box needs both "
             f"tags, so these are read without one: {lo['correct']:,} correct, {lo['would_match_lone_dropped']:,} "
             f"would match if the lone tag were dropped.")
    if ph["replies"]:
        rows = ph["would_match_rows"]
        cfg = ", ".join(sorted({r["config_id"] for r in rows}))
        its = ", ".join(sorted({r["item_id"] for r in rows}))
        s += (f" {ph['replies']:,} replies end their answer with `(<boxed answer>)`, which is not a slot name of "
              f"the placeholder rule, so it stays in the answer read: "
              + "; ".join(f"{k} {v:,}" for k, v in ph["by_model_arm"].items())
              + f", {ph['correct']:,} correct; {ph['would_match_without']:,} would match without it (item {its}, "
                f"configurations {cfg}: \"{rows[0]['answer']}\" against gold \"{rows[0]['gold']}\")." if rows else ".")
    return s + " (`results/hotpot_content/hotpot_residual.json`.)"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=PARTS)
    ap.add_argument("--no-strict-check", action="store_true")
    a = ap.parse_args(argv)
    parts = a.only or PARTS
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    path = os.path.join(OUT, "hotpot_content.json")
    R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    from src import hotpot_task as ht
    R.update({"analysis": "HotpotQA under the content reading",
              "content_scorer": ht.CONTENT_SCORER_VERSION, "strict_scorer": ht.SCORER_VERSION,
              "content_table": rel(CONTENT_TABLE), "created": time.strftime("%Y-%m-%d %H:%M")})
    if os.path.exists(RESIDUAL_TAGS):
        Z = json.load(open(RESIDUAL_TAGS, encoding="utf-8"))
        if Z.get("scorer") != ht.CONTENT_SCORER_VERSION:
            raise SystemExit(f"{RESIDUAL_TAGS} was counted under {Z.get('scorer')}; rerun it under "
                             f"{ht.CONTENT_SCORER_VERSION}")
        R["residual_tags"] = Z
    ctx = setup()
    for p in parts:
        t1 = time.time()
        if p == "formats":
            formats_part(R)
        elif p == "accuracy":
            accuracy_part(R)
        elif p == "arm_effect":
            arm_effect_part(R, ctx)
        elif p == "ceiling":
            ceiling_part(R, ctx)
        elif p == "saved":
            saved_part(R, ctx, not a.no_strict_check)
        elif p == "refit":
            refit_part(R, ctx)
        elif p == "thresholds":
            thresholds_part(R, ctx)
        elif p == "rater_c":
            rater_c_part(R, ctx)
        print(f"{p}: done ({time.time() - t1:.0f} s)", flush=True)
        write_json(R, path)
    txt = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    print(txt)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
