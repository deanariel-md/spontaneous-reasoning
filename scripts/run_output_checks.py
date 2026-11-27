r"""Checks on stored outputs of GSM8K-29, BIG-Bench Hard-28 and MATH-500 (items 4-7 and 9);
nothing here chooses anew, and no HotpotQA or arm-wording output is read.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_output_checks.py                 # every item
    .\.venv\Scripts\python.exe scripts\run_output_checks.py --items 4 6 9   # some items

Items:
  4. Constrained BIG-Bench Hard. Reversed pairs between chain-of-thought gain over the requested arm and over the
     constrained forced arm (model means over the 24 discovery draws, src/robustness.pair_reversals), strict (the
     printed 81 of 378) and with content-scored chain of thought and requested arms. The constrained arm has no content
     scores (data/processed/content/ holds the three collection arms only); its strict scores stand in, and since every
     constrained answer is a member of the item's answer set inside a closed box, strict and content readings of it
     differ only where the strict scorer would reject an allowed string, which a test excludes.
     Beside it, the models whose chain-of-thought accuracy falls below the constrained baseline.
  5. Qwen3.6-35B-A3B. Its w (results/baseline_mixture, discovery draws) on each task, its requested and forced accuracy
     on MATH-500 (strict and content; discovery means and all distinct configurations), its rank as a gainer over the
     requested arm; and the headline results on the panel without it: the arm-switch reversal shares (strict, content),
     both directions of the wrong-baseline cost (one-draw and 100-item designs, the top third re-taken from the saved
     discovery predictions of the models that remain, as the leave-one-family-out ranges do), the corrected
     reversals and the long share of the between-model variance.
  6. Equal traffic. Every per-chosen-model cost of results/decision_directions as a whole-panel accuracy difference
     under equal traffic over the panel: cost x n / M (n = round(M / 3) chosen models).
  7. Length up to the first complete answer. For each stored requested output: tokens up to and including the closing
     brace of the first complete \boxed{...} (brace-balanced), under the checkpoint's own tokenizer
     (data/tokenizers/<key>/tokenizer.json), capped at the generated count; an output with no complete box keeps its
     generated count. Sensitivity: the same minus the tokens of the string "\boxed{" (the forced arm writes that prefix
     for the model). w, the corrected reversals and the long share are recomputed with each length beside the
     saved ones.
  9. Intervals. The template-only t intervals of the abstract's cost numbers beside their leave-one-family-out ranges
     (existing macros, listed by name).

Outputs: results/output_checks/output_checks.json, summary.md, summary_clean.md, tables/*.csv. Cache of item 7's
lengths: data/processed/output_checks/<task>_first_answer.parquet (git-ignored).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import baseline_mixture as bm, decision as dc, robustness as rb  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "output_checks")
CACHE = os.path.join(ROOT, "data", "processed", "output_checks")
TASKS = ("gsm8k", "bbh", "math")
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
TARGET = "qwen3.6-35b-a3b"
SHARDS = {"gsm8k": ["data/d2_out/collection", "data/d2_out/collection_ext"], "bbh": ["data/d2_out/bbh/collection"],
          "math": ["data/d2_out/math/collection"]}
BOX = "\\boxed{"
DIRS = (("requested_to_forced", "ReqToFd"), ("forced_to_requested", "FdToReq"))


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_json(*parts):
    return json.load(open(os.path.join(ROOT, *parts), encoding="utf-8"))


def fnum(v, nd=2):
    if v is None:
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "–" if not np.isfinite(v) else f"{v:.{nd}f}".replace("-", "−")


# ============================================================================ pure helpers (tested)


def first_box_end(text: str):
    """Index just past the closing brace of the first complete \\boxed{...} (braces balanced), or None."""
    if not isinstance(text, str):
        return None
    i = text.find(BOX)
    while i >= 0:
        depth, j = 1, i + len(BOX)
        while j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return None   # the first box never closes: no complete answer later either counts as first
    return None


def panel_equivalent(cost, n_chosen: int, M: int):
    """Whole-panel accuracy difference (pp) under equal traffic over the M models when n_chosen of them are chosen:
    the per-chosen-model cost x n / M (the unchosen models get the baseline under both choices)."""
    if cost is None:
        return None
    return float(cost) * n_chosen / M


def reversal_count(g1, g2):
    r = rb.pair_reversals(g1, g2)
    return {"pairs": r["pairs"], "reversed": r["reversed"], "share": r["reversed_share"], "ties": r["ties"]}


# ============================================================================ item 4


def item4():
    import run_decision as rd
    pm = pd.read_csv(os.path.join(ROOT, "results", "forced_constrained", "tables", "per_model.csv"))
    panel = list(pm["model_key"])
    strict = reversal_count(pm["cot"] - pm["requested"], pm["cot"] - pm["constrained"])
    c = rd.task_split("bbh", "content", "discovery")
    if [str(k) for k in c["panel"]] != panel:
        raise SystemExit("bbh: the content arrays' panel differs from the constrained per-model table")
    x = ri.load("bbh")
    if x["facts"]["panel"] != panel or [str(u) for u in c["units"]] != x["facts"]["discovery_config_ids"]:
        raise SystemExit("bbh: draws or panel differ")
    cot_s, req_s = x["discovery"]["cot"].mean(0), x["discovery"]["requested"].mean(0)
    chk = {"cot_max_abs_diff": float(np.abs(cot_s - pm["cot"]).max()),
           "requested_max_abs_diff": float(np.abs(req_s - pm["requested"]).max())}
    cq, cc = (np.asarray(c[a], float).mean(0) for a in ("requested", "cot"))
    con = pm["constrained"].to_numpy()
    content = reversal_count(cc - cq, cc - con)
    gs, gc = pm["cot"].to_numpy() - con, cc - con
    return {"note": "the constrained arm has no content scores; its strict scores stand in (constrained answers are "
                    "always members of the answer set inside a closed box)",
            "M": len(panel), "check_per_model_table": chk,
            "strict": {**strict, "models_cot_below_constrained": int((gs < 0).sum()), "mean_gain_pp": float(gs.mean())},
            "content_cot_and_requested": {**content, "models_cot_below_constrained": int((gc < 0).sum()),
                                          "mean_gain_pp": float(gc.mean())},
            "earlier_counts": {"printed_strict": 81, "cross_content": 79, "earlier_content": 76}}


# ============================================================================ item 5


def _drop_fams(panel, target):
    return ["drop" if k == target else "keep" for k in panel]


def ddir_without(task, target, srec, sarr):
    """Both directions with the target model left out (top third re-taken from the saved discovery predictions of the
    models that remain: the leave-one-family-out code with the target as its own family), and the full-panel
    estimates for the reproduction check."""
    import run_decision_directions as rdd
    facts = ri.load(task)["facts"]
    fams = _drop_fams(facts["panel"], target)
    o = rdd.one_draw_design(task, fams)
    s = rdd.strict_design(task, srec, sarr, fams)
    out = {"one_draw": {}, "pilot100": {}}
    for sc, blk in (("strict", o), ("content", o.get("content"))):
        if not blk:
            continue
        lofo = blk["leave_one_family_out_own_1"]["by_family"]["drop"]
        out["one_draw"][sc] = {"full": {k: blk["rules"]["own_1"][k]["estimate"] for k, _ in DIRS},
                               "without": {k: lofo[k] for k, _ in DIRS}, "n_chosen": lofo["n_chosen"],
                               "models_left": lofo["models_left"]}
    for sc, blk in s["designs"]["disjoint"].items():
        lofo = blk["leave_one_family_out_own_gain"]["by_family"]["drop"]
        out["pilot100"][sc] = {"full": {k: blk["rules"]["own_gain"][k]["estimate"] for k, _ in DIRS},
                               "without": {k: lofo[k] for k, _ in DIRS}, "n_chosen": lofo["n_chosen"],
                               "models_left": lofo["models_left"]}
    return out


def item5(rows_by_task=None):
    import run_decision as rd
    import run_decision_strict  # noqa: F401  (imported by run_decision_directions' strict design)
    from src import rule_budgets as rbud
    import run_decision_directions as rdd
    BM = load_json("results", "baseline_mixture", "baseline_mixture.json")
    DD = load_json("results", "decision_directions", "decision_directions.json")
    AE = load_json("results", "arm_effect", "arm_effect.json")
    srec, sarr = rbud.load_frozen(rdd.STRICT_FROZEN, rdd.STRICT_FROZEN[:-5] + ".npz")
    out = {"model": TARGET, "tasks": {}}
    for task in TASKS:
        T = BM["tasks"][task]
        panel = T["models"]
        j = panel.index(TARGET)
        keep = np.array([k != TARGET for k in panel])
        pm = pd.DataFrame(T["per_model"]).set_index("model_key").loc[panel]
        g = {sc: {a: pm[f"{sc}_gain_vs_{a}"].to_numpy() for a in ("requested", "forced")} for sc in ("strict", "content")}
        w = pm["w_discovery"].to_numpy()
        blk = {"w_discovery": float(w[j]), "w_all_draws": float(pm["w_all_draws"].iloc[j])}
        # accuracies, discovery means
        x = ri.load(task)
        c = rd.task_split(task, "content", "discovery")
        acc = {"strict": {a: float(x["discovery"][a].mean(0)[j]) for a in ("requested", "forced", "cot")},
               "content": {a: float(np.asarray(c[a], float).mean(0)[j]) for a in ("requested", "forced", "cot")}}
        blk["accuracy_discovery"] = acc
        for sc in ("strict", "content"):
            gr = g[sc]["requested"]
            order = np.argsort(-gr, kind="stable")
            blk.setdefault("rank_gain_vs_requested", {})[sc] = int(np.where(order == j)[0][0]) + 1
        blk["M"] = len(panel)
        # reversals: full (check) and without
        rev = {}
        for sc in ("strict", "content"):
            full = reversal_count(g[sc]["requested"], g[sc]["forced"])
            wo = reversal_count(g[sc]["requested"][keep], g[sc]["forced"][keep])
            rev[sc] = {"full": full, "without": wo}
        blk["arm_switch_reversals"] = rev
        blk["check_reversals"] = {
            "strict_vs_arm_effect": [rev["strict"]["full"]["reversed"], AE["tasks"][task]["reversals"]["reversed"]],
            "content_vs_decision_directions": [rev["content"]["full"]["reversed"],
                                               DD["tasks"][task]["reversal"]["content"]["reversed"]]}
        # mechanism
        mech = {}
        for sc in ("strict", "content"):
            full = bm.mechanism(g[sc]["requested"], g[sc]["forced"], w)
            wo = bm.mechanism(g[sc]["requested"][keep], g[sc]["forced"][keep], w[keep])
            mech[sc] = {"full": {"observed": full["observed_vs_forced"]["reversed"],
                                 "corrected": full["observed_vs_corrected"]["reversed"],
                                 "pairs": full["observed_vs_forced"]["pairs"]},
                        "without": {"observed": wo["observed_vs_forced"]["reversed"],
                                    "corrected": wo["observed_vs_corrected"]["reversed"],
                                    "pairs": wo["observed_vs_forced"]["pairs"]},
                        "saved": {"observed": T["mechanism"][sc]["observed_vs_forced"]["reversed"],
                                  "corrected": T["mechanism"][sc]["observed_vs_corrected"]["reversed"]}}
        blk["mechanism"] = mech
        # long share without the model
        if rows_by_task and task in rows_by_task:
            import run_baseline_mixture as rbm
            wide, pnl, disc, val = rows_by_task[task]
            wk = wide[wide["model_key"] != TARGET]
            pk = [k for k in pnl if k != TARGET]
            blk["variance_share_long_without"] = {
                sc: v["variance_share_long"] for sc, v in
                rbm.decomposition_block(wk, pk, rbm.cap_of(task), sorted(set(disc) | set(val)), False).items()}
            blk["variance_share_long_saved"] = {
                sc: BM["decomposition"][task]["sets"]["all_distinct"][sc]["variance_share_long"]
                for sc in ("strict", "content")}
        # decision costs
        d = ddir_without(task, TARGET, srec, sarr)
        saved = DD["tasks"][task]
        d["check"] = {
            "one_draw_strict": {k: [d["one_draw"]["strict"]["full"][k], saved["one_draw"]["rules"]["own_1"][k]["estimate"]]
                                for k, _ in DIRS},
            "pilot100_strict": {k: [d["pilot100"]["strict"]["full"][k],
                                    saved["strict"]["designs"]["disjoint"]["strict"]["rules"]["own_gain"][k]["estimate"]]
                                for k, _ in DIRS}}
        blk["decision_costs"] = d
        out["tasks"][task] = blk
        print(f"item 5, {task}: done", flush=True)
    # MATH-500 pooled accuracies over all distinct configurations (an earlier count gave 36.2 against 50.7 under content)
    if rows_by_task and "math" in rows_by_task:
        wide = rows_by_task["math"][0]
        r = wide[wide["model_key"] == TARGET]
        out["math_all_distinct_pooled"] = {sc: {a: float(100 * r[f"{a}_{sc}"].mean()) for a in ("requested", "forced", "cot")}
                                           for sc in ("strict", "content")}
    # abstract min-max without the model
    mm = {}
    for design in ("one_draw", "pilot100"):
        for k, _ in DIRS:
            v = [out["tasks"][t]["decision_costs"][design]["strict"]["without"][k] for t in TASKS]
            mm[f"{design}/{k}"] = [float(min(v)), float(max(v))]
    out["min_max_without"] = mm
    return out


# ============================================================================ item 6


def item6():
    DD = load_json("results", "decision_directions", "decision_directions.json")
    rows = []
    for task in TASKS:
        T = DD["tasks"][task]
        o = T["one_draw"]
        blocks = [("one_draw", "strict", o["rules"]["own_1"], o["n_chosen"], o["M"])]
        if "content" in o:
            blocks.append(("one_draw", "content", o["content"]["rules"]["own_1"], o["n_chosen"], o["M"]))
        S = T["strict"]
        for sc, b in S["designs"]["disjoint"].items():
            blocks.append(("pilot100", sc, b["rules"]["own_gain"], S["n_chosen"], S["M"]))
        for design, sc, r, n, M in blocks:
            for k, _ in DIRS:
                e = r[k]
                ci = e.get("ci95")
                rows.append({"task": task, "design": design, "score": sc, "direction": k, "n_chosen": n, "M": M,
                             "per_chosen_model": e["estimate"], "ci95": ci,
                             "panel_equivalent": panel_equivalent(e["estimate"], n, M),
                             "panel_ci95": [panel_equivalent(ci[0], n, M), panel_equivalent(ci[1], n, M)] if ci else None})
    mm = {}
    for design in ("one_draw", "pilot100"):
        for k, _ in DIRS:
            v = [r["panel_equivalent"] for r in rows if r["design"] == design and r["score"] == "strict" and r["direction"] == k]
            mm[f"{design}/{k}"] = [float(min(v)), float(max(v))]
    chk = next(r for r in rows if r["task"] == "math" and r["design"] == "pilot100" and r["score"] == "strict"
               and r["direction"] == "forced_to_requested")
    return {"rule": "whole-panel accuracy difference under equal traffic over the panel = per-chosen-model cost x n / M; "
                    "the leave-one-family-out ranges change n and M and are not scaled",
            "rows": rows, "min_max": mm,
            "check_math_pilot_fd_to_req": {"per_chosen": chk["per_chosen_model"], "panel": chk["panel_equivalent"],
                                           "reference": [12.75, 4.40]}}


# ============================================================================ item 7


def requested_texts(task, panel, configs):
    """(model_key, config_id, item_id, output_text, n_output_tokens) of the requested-direct outputs of the given
    configurations, deduplicated on the key (a requeued shard repeats rows; their token counts must agree)."""
    cols = ["model_key", "arm", "config_id", "item_id", "output_text", "n_output_tokens"]
    parts = []
    for d in SHARDS[task]:
        for m in panel:
            for f in sorted(glob.glob(os.path.join(ROOT, d, m, "shard-*.parquet"))):
                x = pd.read_parquet(f, columns=cols)
                x = x[(x["arm"] == "direct_answer") & x["config_id"].astype(str).isin(configs)]
                if len(x):
                    parts.append(x.drop(columns="arm"))
    df = pd.concat(parts, ignore_index=True)
    for c in ("model_key", "config_id", "item_id"):
        df[c] = df[c].astype(str)
    key = ["model_key", "config_id", "item_id"]
    g = df.groupby(key)["n_output_tokens"].nunique()
    if (g > 1).any():
        raise SystemExit(f"{task}: repeated outputs disagree in token count ({int((g > 1).sum())} keys)")
    return df.drop_duplicates(key, keep="first")


def first_answer_lengths(task, panel, configs):
    """Per requested output: tokens up to the first complete box (capped at the generated count), the same minus the
    tokens of the box opening, whether a complete box exists. Cached."""
    path = os.path.join(CACHE, f"{task}_first_answer.parquet")
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if set(c["config_id"].unique()) >= set(configs) and set(c["model_key"].unique()) >= set(panel):
            return c
    from tokenizers import Tokenizer
    df = requested_texts(task, panel, set(configs))
    out = []
    for m in panel:
        x = df[df["model_key"] == m].copy()
        tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizers", m, "tokenizer.json"))
        k_box = len(tok.encode(BOX, add_special_tokens=False).ids)
        ends = [first_box_end(t) for t in x["output_text"]]
        has = np.array([e is not None for e in ends])
        n = x["n_output_tokens"].to_numpy()
        first = n.copy()
        ix = np.where(has)[0]
        if len(ix):
            texts = [x["output_text"].iloc[i][:ends[i]] for i in ix]
            enc = tok.encode_batch(texts, add_special_tokens=False)
            first[ix] = np.minimum(np.array([len(e.ids) for e in enc]), n[ix])
        nopre = first.copy()
        nopre[has] = np.maximum(first[has] - k_box, 0)
        out.append(pd.DataFrame({"model_key": m, "config_id": x["config_id"].to_numpy(), "item_id": x["item_id"].to_numpy(),
                                 "n_output_tokens": n, "n_first_answer": first, "n_first_answer_no_prefix": nopre,
                                 "has_complete_box": has, "box_prefix_tokens": k_box}))
        print(f"  {task} {m}: {len(x)} outputs, {has.mean():.3f} with a complete box", flush=True)
    res = pd.concat(out, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    res.to_parquet(path, index=False)
    return res


def item7(rows_by_task):
    import run_baseline_mixture as rbm
    BM = load_json("results", "baseline_mixture", "baseline_mixture.json")
    out = {"tasks": {}}
    tabs = []
    for task in TASKS:
        wide, panel, disc, val = rows_by_task[task]
        cap = rbm.cap_of(task)
        allc = sorted(set(disc) | set(val))
        L = first_answer_lengths(task, panel, allc)
        key = ["model_key", "config_id", "item_id"]
        j = wide.merge(L.drop(columns="n_output_tokens"), on=key, how="left", validate="one_to_one")
        if j["n_first_answer"].isna().any():
            raise SystemExit(f"{task}: {int(j['n_first_answer'].isna().sum())} rows lack a stored output text")
        if not (L.set_index(key).loc[pd.MultiIndex.from_frame(wide[key]), "n_output_tokens"].to_numpy()
                == wide["n_output_tokens"].to_numpy()).all():
            raise SystemExit(f"{task}: shard token counts differ from the scored tables'")
        T = BM["tasks"][task]
        pm = pd.DataFrame(T["per_model"]).set_index("model_key").loc[panel]
        g = {sc: {a: pm[f"{sc}_gain_vs_{a}"].to_numpy() for a in ("requested", "forced")} for sc in ("strict", "content")}
        blk = {"cap": cap, "rows": int(len(j))}
        w_saved_re = rbm.w_of(wide, panel, cap, disc)
        blk["check_w_reproduces"] = float(np.abs(w_saved_re - np.array(T["w_discovery"])).max())
        long_whole = j["n_output_tokens"].to_numpy() > cap
        blk["long_share_pct"] = {"whole_reply": float(100 * long_whole.mean())}
        blk["no_complete_box_among_long_pct"] = float(100 * (~j["has_complete_box"].to_numpy()[long_whole]).mean())
        for name, col in (("first_answer", "n_first_answer"), ("first_answer_no_prefix", "n_first_answer_no_prefix")):
            w2 = j.copy()
            w2["n_output_tokens"] = w2[col].astype(int)
            lf = w2["n_output_tokens"].to_numpy() > cap
            blk["long_share_pct"][name] = float(100 * lf.mean())
            wn = rbm.w_of(w2, panel, cap, disc)
            res = {"w_discovery": wn.tolist(),
                   "w_median_change": float(np.median(wn - np.array(T["w_discovery"]))),
                   "w_max_drop": float(np.max(np.array(T["w_discovery"]) - wn)),
                   "mechanism": {}, "variance_share_long": {}}
            for sc in ("strict", "content"):
                mch = bm.mechanism(g[sc]["requested"], g[sc]["forced"], wn)
                res["mechanism"][sc] = {"observed": mch["observed_vs_forced"]["reversed"],
                                        "corrected": mch["observed_vs_corrected"]["reversed"],
                                        "pairs": mch["observed_vs_forced"]["pairs"],
                                        "rms_corrected_pp": mch["rms_corrected_pp"]}
            dec = rbm.decomposition_block(w2, panel, cap, allc, False)
            res["variance_share_long"] = {sc: dec[sc]["variance_share_long"] for sc in dec}
            blk[name] = res
            for k, m_ in enumerate(panel):
                tabs.append({"task": task, "measure": name, "model_key": m_, "w_saved": T["w_discovery"][k],
                             "w_new": wn[k]})
        blk["saved"] = {"mechanism": {sc: {"observed": T["mechanism"][sc]["observed_vs_forced"]["reversed"],
                                           "corrected": T["mechanism"][sc]["observed_vs_corrected"]["reversed"]}
                                      for sc in ("strict", "content")},
                        "variance_share_long": {sc: BM["decomposition"][task]["sets"]["all_distinct"][sc]["variance_share_long"]
                                                for sc in ("strict", "content")}}
        out["tasks"][task] = blk
        print(f"item 7, {task}: done", flush=True)
    out["_table"] = tabs
    return out


# ============================================================================ item 9


ABSTRACT_COSTS = [("ddirOwnFdToReqMin", "ddirOwnFdToReqMax"), ("ddirOwnReqToFdMin", "ddirOwnReqToFdMax")]


def item9():
    import macros_added as ma
    DD = load_json("results", "decision_directions", "decision_directions.json")
    m = ma.decision_directions_macros(DD)
    rows = []
    for task in TASKS:
        t = "ddir" + ma._task(task)
        for design, stem in (("one_draw", t), ("pilot100", t + "Pilot")):
            for _, tag in DIRS:
                s = stem + tag
                rows.append({"task": task, "design": design, "direction": tag, "estimate_macro": s,
                             "estimate": m.get(s), "template_ci_macros": [s + "Lo", s + "Hi"],
                             "template_ci": [m.get(s + "Lo"), m.get(s + "Hi")],
                             "lofo_macros": [s + "LofoMin", s + "LofoMax"],
                             "lofo_range": [m.get(s + "LofoMin"), m.get(s + "LofoMax")]})
    return {"note": "template-only t intervals (over validation configurations) beside the leave-one-family-out "
                    "ranges; all values are existing ddir macros", "rows": rows,
            "abstract_macros": {a: m.get(a) for pair in ABSTRACT_COSTS for a in pair}}


# ============================================================================ report


def summary(R):
    L = ["# Checks on stored outputs", "",
         "Code `scripts/run_output_checks.py`; numbers in `output_checks.json` and `tables/`; macros prefix `wck` "
         "(`paper/macros_output_checks.py`). Checks on stored outputs of GSM8K-29, BIG-Bench Hard-28 and "
         "MATH-500 (items 4-7 and 9). No HotpotQA or arm-wording output was read.", ""]
    if "item4" in R:
        x = R["item4"]
        s, c = x["strict"], x["content_cot_and_requested"]
        L += ["## Item 4. Constrained BIG-Bench Hard", "",
              f"- Strict (chain of thought, requested, constrained all strict): {s['reversed']} of {s['pairs']} pairs "
              f"reverse; {s['models_cot_below_constrained']} of {x['M']} models score below the constrained baseline "
              f"with chain of thought (mean gain {fnum(s['mean_gain_pp'], 2)} pp).",
              f"- Content-scored chain of thought and requested arms, constrained arm strict: {c['reversed']} of "
              f"{c['pairs']} pairs reverse; {c['models_cot_below_constrained']} of {x['M']} models below the constrained "
              f"baseline (mean gain {fnum(c['mean_gain_pp'], 2)} pp).",
              f"- {x['note']}. Earlier counts: printed 81 (strict), 79 (a reviewer's count), an earlier count 76.",
              f"- Check against the per-model table: largest difference {fnum(x['check_per_model_table']['cot_max_abs_diff'], 6)} pp "
              "(chain of thought).", ""]
    if "item5" in R:
        x = R["item5"]
        L += ["## Item 5. Qwen3.6-35B-A3B", "",
              "| Task | w (discovery) | Requested / forced / CoT, strict (%) | Same, content (%) | Rank as gainer over requested (strict; content) |",
              "|---|---|---|---|---|"]
        for t, b in x["tasks"].items():
            a = b["accuracy_discovery"]
            L.append(f"| {TITLE[t]} | {fnum(b['w_discovery'], 3)} | "
                     f"{fnum(a['strict']['requested'], 1)} / {fnum(a['strict']['forced'], 1)} / {fnum(a['strict']['cot'], 1)} | "
                     f"{fnum(a['content']['requested'], 1)} / {fnum(a['content']['forced'], 1)} / {fnum(a['content']['cot'], 1)} | "
                     f"{b['rank_gain_vs_requested']['strict']}; {b['rank_gain_vs_requested']['content']} of {b['M']} |")
        if "math_all_distinct_pooled" in x:
            p = x["math_all_distinct_pooled"]["content"]
            L += ["", f"MATH-500, all distinct configurations pooled, content: requested {fnum(p['requested'], 1)}%, forced "
                      f"{fnum(p['forced'], 1)}% (an earlier count: 36.2 against 50.7)."]
        L += ["", "Without the model (the top third re-taken from the saved discovery predictions of the models that "
                  "remain; costs in pp per chosen model):", "",
              "| Task | Reversed pairs strict (full → without) | Content | Corrected reversals strict (full → without) | "
              "One-draw cost req→fd; fd→req (full → without) | 100-item cost req→fd; fd→req (full → without) |",
              "|---|---|---|---|---|---|"]
        for t, b in x["tasks"].items():
            r, mc, d = b["arm_switch_reversals"], b["mechanism"]["strict"], b["decision_costs"]
            od, pi = d["one_draw"]["strict"], d["pilot100"]["strict"]

            def two(z):
                return "; ".join(f"{fnum(z['full'][k])} → {fnum(z['without'][k])}" for k, _ in DIRS)
            L.append(f"| {TITLE[t]} | {r['strict']['full']['reversed']}/{r['strict']['full']['pairs']} → "
                     f"{r['strict']['without']['reversed']}/{r['strict']['without']['pairs']} | "
                     f"{r['content']['full']['reversed']}/{r['content']['full']['pairs']} → "
                     f"{r['content']['without']['reversed']}/{r['content']['without']['pairs']} | "
                     f"{mc['full']['corrected']} → {mc['without']['corrected']} | {two(od)} | {two(pi)} |")
        L += ["", "Full-panel recomputations reproduce the saved values (`checks` in the json)."]
        for k, v in x["min_max_without"].items():
            L.append(f"- Without the model, {k}: {fnum(v[0])} to {fnum(v[1])} pp.")
        L.append("")
    if "item6" in R:
        x = R["item6"]
        c = x["check_math_pilot_fd_to_req"]
        L += ["## Item 6. Equal traffic over the panel", "", f"Rule: {x['rule']}.",
              f"Check: MATH-500, 100-item pilots, forced → requested: {fnum(c['per_chosen'])} pp per chosen model is "
              f"{fnum(c['panel'])} pp over the panel (an earlier count: 12.75 and 4.40).", "",
              "| Task | Design | Score | Direction | Per chosen model | Whole panel |", "|---|---|---|---|---|---|"]
        for r in x["rows"]:
            L.append(f"| {TITLE[r['task']]} | {r['design']} | {r['score']} | {r['direction']} | "
                     f"{fnum(r['per_chosen_model'])} | {fnum(r['panel_equivalent'])} |")
        L.append("")
    if "item7" in R:
        x = R["item7"]
        L += ["## Item 7. Length up to the first complete answer", "",
              "Tokens up to the closing brace of the first complete box, under each checkpoint's tokenizer; the "
              "sensitivity also removes the tokens of the box opening. An output with no complete box keeps its "
              "generated count.", "",
              "| Task | Long share: whole / first answer / no prefix (%) | Long outputs with no complete box (%) | "
              "Corrected reversals strict: saved / first / no prefix | Long share of variance strict: saved / first / no prefix |",
              "|---|---|---|---|---|"]
        for t, b in x["tasks"].items():
            ls = b["long_share_pct"]
            L.append(f"| {TITLE[t]} | {fnum(ls['whole_reply'], 1)} / {fnum(ls['first_answer'], 1)} / "
                     f"{fnum(ls['first_answer_no_prefix'], 1)} | {fnum(b['no_complete_box_among_long_pct'], 1)} | "
                     f"{b['saved']['mechanism']['strict']['corrected']} / {b['first_answer']['mechanism']['strict']['corrected']} / "
                     f"{b['first_answer_no_prefix']['mechanism']['strict']['corrected']} | "
                     f"{fnum(100 * b['saved']['variance_share_long']['strict'], 1)} / "
                     f"{fnum(100 * b['first_answer']['variance_share_long']['strict'], 1)} / "
                     f"{fnum(100 * b['first_answer_no_prefix']['variance_share_long']['strict'], 1)} |")
        L.append("")
    if "item9" in R:
        x = R["item9"]
        L += ["## Item 9. Template-only intervals beside leave-one-family-out ranges", "",
              "| Task | Design | Direction | Estimate | Template-only 95% t interval | Leaving one family out | Macros |",
              "|---|---|---|---|---|---|---|"]
        for r in x["rows"]:
            L.append(f"| {TITLE[r['task']]} | {r['design']} | {r['direction']} | {r['estimate']} | "
                     f"[{r['template_ci'][0]}, {r['template_ci'][1]}] | {r['lofo_range'][0]} to {r['lofo_range'][1]} | "
                     f"`{r['estimate_macro']}` (+Lo/Hi, LofoMin/Max) |")
        L.append("")
    return [s.replace("$-$", "−") for s in L]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="*", type=int, default=[4, 5, 6, 7, 9])
    a = ap.parse_args(argv)
    t0 = time.time()
    path = os.path.join(OUT, "output_checks.json")
    R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    R.update({"analysis": "checks on stored outputs, items 4-7 and 9 (exploratory)",
              "created": time.strftime("%Y-%m-%d %H:%M")})
    rows = None
    if 5 in a.items or 7 in a.items:
        import run_baseline_mixture as rbm
        rows = {t: rbm.task_rows(t) for t in TASKS}
    if 4 in a.items:
        R["item4"] = item4()
    if 6 in a.items:
        R["item6"] = item6()
    if 9 in a.items:
        R["item9"] = item9()
    if 7 in a.items:
        r7 = item7(rows)
        tab = r7.pop("_table")
        os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
        pd.DataFrame(tab).to_csv(os.path.join(OUT, "tables", "w_first_answer.csv"), index=False)
        R["item7"] = r7
    if 5 in a.items:
        R["item5"] = item5(rows)
    R["checks"] = checks(R)
    R["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    if "item6" in R:
        pd.DataFrame(R["item6"]["rows"]).to_csv(os.path.join(OUT, "tables", "equal_traffic.csv"), index=False)
    if "item9" in R:
        pd.DataFrame(R["item9"]["rows"]).to_csv(os.path.join(OUT, "tables", "intervals.csv"), index=False)
    import macros_output_checks as mw
    R["macros"] = mw.output_checks_macros(R)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ri.jsonable(R), f, indent=1)
    L = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return R


def checks(R):
    c = {}
    if "item4" in R:
        c["item4_strict_81"] = R["item4"]["strict"]["reversed"] == 81
        c["item4_strict_16_below"] = R["item4"]["strict"]["models_cot_below_constrained"] == 16
    if "item5" in R:
        ok = True
        for t, b in R["item5"]["tasks"].items():
            cr = b["check_reversals"]
            ok &= cr["strict_vs_arm_effect"][0] == cr["strict_vs_arm_effect"][1]
            ok &= cr["content_vs_decision_directions"][0] == cr["content_vs_decision_directions"][1]
            for sc in ("strict", "content"):
                mc = b["mechanism"][sc]
                ok &= mc["full"]["observed"] == mc["saved"]["observed"] and mc["full"]["corrected"] == mc["saved"]["corrected"]
            for blk in b["decision_costs"]["check"].values():
                for v in blk.values():
                    ok &= abs(v[0] - v[1]) < 1e-9
        c["item5_full_panel_reproduces"] = bool(ok)
    if "item6" in R:
        v = R["item6"]["check_math_pilot_fd_to_req"]
        c["item6_math_pilot_4_40"] = round(v["panel"], 2) == 4.40
    if "item7" in R:
        c["item7_w_reproduces"] = all(b["check_w_reproduces"] < 1e-12 for b in R["item7"]["tasks"].values())
    return c


if __name__ == "__main__":
    main()
