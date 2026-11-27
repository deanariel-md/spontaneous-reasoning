r"""Supporting quantities for five statements of the paper (the parts below): task statement_values.

From the repository root:
    export PYTHONIOENCODING=utf-8
    python scripts/statement_values.py                  # every part
    python scripts/statement_values.py --only best_third thinking_qwen

Parts:
  hotpot        HotpotQA's strict Steps 1-3 quantities and four-draw statements on content exact match
                (data/processed/hotpot/content_view/, strict_correct holding the content reading), through the full
                run's own functions (scripts/run_hotpot.py: arrays, tranche_statements, evaluate_statements;
                src/dataset2.step2). The same code on the strict table must reproduce the stored strict values
                (results/hotpot/hotpot.json, hotpot_forced.json, hotpot_report.json), or the script stops.
                Mean gain and R of benefit are discovery-draw estimates. The four-draw prediction of one-draw C under
                content is refitted on the content-scored first four discovery draws (exploratory: the saved prediction,
                manifests/predictions_hotpot_tranche.json, is strict and was saved before validation was scored). The
                arm-switch shares and the observed C are validation-draw observations on content scores.
  best_third    whether \dec<T>ReqBestThird (results/decision/decision.json) is valued on the validation
                templates and items of \ddir<T>FdToReq (results/decision_directions/decision_directions.json): both
                recomputed from one load of the validation arrays (scripts/run_decision.setting_arrays) on the rows
                of scripts/run_decision.evaluation_rows (independent), each checked against its stored value.
  ladder_known  the 70B-and-above reversal count without DeepSeek V4.1-Flash-1017 (Llama-3.1-70B,
                Qwen3.5-122B-A10B, Mistral Large 3), with scripts/ladder_orderings.reversals and pooled_counts on the
                model means stored in results/ladder/analysis/ladder_orderings.json (discovery draws); the same code
                on the four models must reproduce the stored 70B-and-above block, or the script stops.
  recent        the reversal count (CoT - requested against CoT - forced, src/robustness.pair_reversals, as
                \ae<T>Reversed) among the panel models whose Hugging Face repository was created in 2025 or later
                ("created" in manifests/d2_panel_extension.json, from manifests/external/hf_metadata_20261030.json;
                model key to repository by the model registries). Panel means through
                scripts/ladder_orderings.panel_means (discovery draws; strict, HotpotQA content); the full panel must
                reproduce the stored counts.
  thinking_qwen the Qwen checkpoints among the models of results/generality/generality.json, row
                thi_gsm8k_arm (\genThiGsmArmM), family from the model registries.

Outputs: results/statement_values/statement_values.json (block "macros"), statement_values.md, statement_values_clean.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE, os.path.join(ROOT, "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(ROOT, "results", "statement_values")
STEM = "statement_values"
PARTS = ("hotpot", "best_third", "ladder_known", "recent", "thinking_qwen")
TOL = 1e-9
REQUEST = "supporting quantities for the paper's text"
T_TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh", "hotpot": "Hq"}
RECENT_FROM = "2025-01-01"
DEEPSEEK = "deepseek-v4.1-flash-1017"
PANEL_MANIFEST = os.path.join(ROOT, "manifests", "d2_panel_extension.json")
HF_METADATA = os.path.join(ROOT, "manifests", "external", "hf_metadata_20261030.json")


def rel(p) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def load_json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def jsonable(o):
    from run_index_checks import jsonable as j
    return json.loads(json.dumps(j(o)))


def num(x, nd):
    import macros_added as ma
    return ma.num(x, nd)


def close(a, b, tol=TOL) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ============================================================================ HotpotQA on content scores


def hotpot_block(path: str, ctx: dict, mcfg: dict, units4: list, sel) -> dict:
    """Steps 2 (discovery draws) and the four-draw statements (predicted from the first four discovery draws, checked
    on the validation draws) of the requested and forced contrasts, from the table at `path`."""
    import run_hotpot as rh
    from src import dataset2 as d2, intervals as iv, sampling
    panel, items, disc, val = ctx["panel"], ctx["items"], ctx["disc"], ctx["val"]
    out = {"step2": {}}
    for c in ("cot_req", "cot_forced"):
        label = rh.CONTRASTS[c][2]
        Bd, Pd = rh.arrays(rh.loader(path, c), disc, panel, items)
        m = iv.multiplicities(sampling.rng_for(label, "bootstrap"), len(disc), 999)
        s2 = d2.step2(Bd, Pd, sel, m)
        out["step2"][c] = {"mean_gain": s2["mean_gain"]["estimate"], "R_benefit": s2["reliability"]["R_benefit"],
                           "R_accuracy": s2["reliability"]["R_accuracy"], "K": s2["K"]}
    pairs = {c: rh.arrays(rh.loader(path, c), units4, panel, items) for c in rh.CONTRASTS}
    T = {"pairs": pairs, "req4": pairs["cot_req"][0], "fd4": pairs["cot_forced"][0], "cot4": pairs["cot_req"][1]}
    st = jsonable(rh.tranche_statements(T, mcfg, units4))
    V = {c: rh.arrays(rh.loader(path, c), val, panel, items) for c in rh.CONTRASTS}
    checks = rh.evaluate_statements({"statements": st}, V, pairs, mcfg, int(mcfg["validation_splits"]), val)
    out["statements"] = {c: {k: st[c][k] for k in ("a_predicted_C_one_draw", "b_K_needed", "d_R_benefit")}
                         for c in rh.CONTRASTS}
    out["statements"]["c_arm_switch"] = {"switch_reorders_more": st["c_arm_switch"]["switch_reorders_more"],
                                         "switch_opposite": st["c_arm_switch"]["switch"]["opposite"],
                                         "new_draw_opposite": st["c_arm_switch"]["new_draw"]["opposite"]}
    out["checks"] = jsonable(checks)
    return out


def hotpot_part(R: dict) -> None:
    import yaml
    import run_hotpot as rh
    import run_hotpot_content as rhc
    from src import dataset2 as d2, hotpot_task as ht
    ctx = rhc.setup()
    mcfg = yaml.safe_load(open(rh.MAP_CONFIG, encoding="utf-8"))
    saved = load_json("manifests", "predictions_hotpot_tranche.json")
    units4 = list(map(str, saved["units"]))
    sel = d2.load_selection(rh.SELECTION)
    t0 = time.time()
    S = hotpot_block(rhc.STRICT_TABLE, ctx, mcfg, units4, sel)
    print(f"hotpot strict: {time.time() - t0:.0f} s", flush=True)
    # the strict values must reproduce the stored ones
    rep = load_json("results", "hotpot", "hotpot_report.json")["tranche"]["checks"]
    pairs = []
    for c, f in (("cot_req", "hotpot.json"), ("cot_forced", "hotpot_forced.json")):
        st2 = load_json("results", "hotpot", f)["step2"]
        pairs += [(f"{f} mean gain", S["step2"][c]["mean_gain"], st2["mean_gain"]["estimate"]),
                  (f"{f} R of benefit", S["step2"][c]["R_benefit"], st2["reliability"]["R_benefit"]),
                  (f"tranche {c} predicted C", S["checks"][c]["a"]["predicted"], rep[c]["a"]["predicted"]),
                  (f"tranche {c} observed C", S["checks"][c]["a"]["observed"], rep[c]["a"]["observed"]),
                  (f"saved {c} predicted C", S["statements"][c]["a_predicted_C_one_draw"],
                   saved["statements"][c]["a_predicted_C_one_draw"])]
    o, so = S["checks"]["c_arm_switch"]["observed"], rep["c_arm_switch"]["observed"]
    pairs += [("switch, opposite", o["switch"]["opposite"], so["switch"]["opposite"]),
              ("new draw, opposite", o["new_draw"]["opposite"], so["new_draw"]["opposite"])]
    bad = [(n, a, b) for n, a, b in pairs if not close(a, b)]
    if bad:
        raise SystemExit(f"strict values do not reproduce the stored ones: {bad}")
    t0 = time.time()
    C = hotpot_block(os.path.join(rhc.VIEW, "scored.parquet"), ctx, mcfg, units4, sel)
    print(f"hotpot content: {time.time() - t0:.0f} s", flush=True)
    view = load_json("data", "processed", "hotpot", "content_view", "scored.json")
    if ht.CONTENT_SCORER_VERSION not in str(view.get("view")):
        raise SystemExit(f"the content view was not built under {ht.CONTENT_SCORER_VERSION}: {view.get('view')}; "
                         "rerun scripts/score_hotpot_content.py")
    m = {}
    for c, t in (("cot_req", "hqReq"), ("cot_forced", "hqFd")):
        m[t + "MeanGainContent"] = num(C["step2"][c]["mean_gain"], 1)
        m[t + "RbenContent"] = num(C["step2"][c]["R_benefit"], 2)
        m[t + "TrPredCContent"] = num(C["checks"][c]["a"]["predicted"], 3)
        m[t + "TrObsCContent"] = num(C["checks"][c]["a"]["observed"], 3)
    o = C["checks"]["c_arm_switch"]["observed"]
    m["hqTrSwitchReversedContent"] = num(o["switch"]["opposite"], 3)
    m["hqTrNewDrawReversedContent"] = num(o["new_draw"]["opposite"], 3)
    R["hotpot"] = {
        "content_scorer": ht.CONTENT_SCORER_VERSION, "content_view": rel(os.path.join(rhc.VIEW, "scored.parquet")),
        "content_view_note": view.get("view"),
        "strict_reproduced": [{"what": n, "here": a, "stored": b} for n, a, b in pairs],
        "content": C, "strict": S,
        "saved_strict_prediction_against_content": {
            c: {"saved_predicted": saved["statements"][c]["a_predicted_C_one_draw"],
                "content_observed": C["checks"][c]["a"]["observed"],
                "within_0.05": bool(abs(float(saved["statements"][c]["a_predicted_C_one_draw"])
                                        - C["checks"][c]["a"]["observed"]) <= 0.05)}
            for c in ("cot_req", "cot_forced")},
        "labels": {
            "MeanGain, Rben": "discovery-draw estimates on content scores (src/dataset2.step2), as the strict ones",
            "TrPredC": "exploratory: refitted on the content-scored first four discovery draws (the saved prediction is "
                       "strict, manifests/predictions_hotpot_tranche.json, saved before validation was scored)",
            "TrObsC, TrSwitchReversed, TrNewDrawReversed": "observed on the content-scored validation draws (24), "
                                                           "the full run's statistic"},
        "macros": m}


# ============================================================================ the best third's denominator


def best_third_part(R: dict) -> None:
    import run_decision as rd
    from src import decision as dc, decision_directions as dd
    D = load_json("results", "decision", "decision.json")
    DD = load_json("results", "decision_directions", "decision_directions.json")["tasks"]
    out, m = {}, {}
    for task in ("gsm8k", "math", "bbh"):
        units, panel, B, P = rd.setting_arrays(f"{task}_req", "validation")
        rec, W = {}, {}
        for a, s in (("requested", "req"), ("forced", "fd")):
            rec[a], W[a] = rd.load_choices(f"{task}_{s}")
        rows = rd.evaluation_rows(rec["requested"]["discovery_units"], units)["independent"]
        n = rec["requested"]["n_chosen"]
        G = (np.asarray(P, float) - np.asarray(B, float))[rows]
        gm = G.mean(0)
        best = float(dc.best_weights(G, n) @ gm / n)
        # the cost on the same arrays (the forced arm's validation gains on the same rows, as run_decision_directions)
        uf, pf, Bf, Pf = rd.setting_arrays(f"{task}_fd", "validation")
        if list(map(str, uf)) != list(map(str, units)) or list(pf) != list(panel):
            raise SystemExit(f"{task}: the two arms' validation draws or panel differ")
        Gf = (np.asarray(Pf, float) - np.asarray(Bf, float))[rows]
        cost = dd.both_directions({a: W[a]["own_1"] for a in W}, {"requested": G, "forced": Gf}, n)
        ev = D["settings"][f"{task}_req"]["result"]["evaluation"]["independent"]
        o = DD[task]["one_draw"]
        stored_cost = o["rules"]["own_1"]["forced_to_requested"]["estimate"]
        r = {"configurations": [str(units[i]) for i in rows], "n_configurations": len(rows), "M": len(panel),
             "n_chosen": n, "best_third_here": best, "best_third_stored": ev["best_third_value"],
             "best_third_configurations_stored": ev["n_configurations"],
             "cost_fd_to_req_here": cost["forced_to_requested"]["estimate"], "cost_fd_to_req_stored": stored_cost,
             "cost_configurations_stored": o["n_configurations"],
             "regret_matched_requested": cost["forced_to_requested"]["regret_matched"],
             "regret_mismatched_requested": cost["forced_to_requested"]["regret_mismatched"]}
        r["same"] = bool(close(best, ev["best_third_value"]) and close(r["cost_fd_to_req_here"], stored_cost)
                         and ev["n_configurations"] == len(rows) == o["n_configurations"]
                         and D["settings"][f"{task}_req"]["result"]["M"] == len(panel) == o["M"]
                         and D["settings"][f"{task}_req"]["result"]["n_chosen"] == n == o["n_chosen"])
        out[task] = r
    R["best_third"] = {
        "answer": ("same" if all(x["same"] for x in out.values()) else "differs"),
        "note": ("Both values recomputed from one load of the requested contrast's validation arrays "
                 "(scripts/run_decision.setting_arrays('<task>_req', 'validation'): every validation configuration's "
                 "items as scored), on the validation configurations no discovery draw used "
                 "(run_decision.evaluation_rows, 'independent'), the same panel and third"),
        "tasks": out, "macros": m}


# ============================================================================ 70B and above, size known
# (built while DeepSeek V4.1-Flash-1017's own size was not recorded; its published size puts it in
# the group; the text does not use these macros)


def ladder_known_part(R: dict) -> None:
    import ladder_orderings as lo
    J = load_json("results", "ladder", "analysis", "ladder_orderings.json")
    if J["big"] != lo.BIG:
        raise SystemExit("ladder_orderings.json's 70B-and-above group differs from scripts/ladder_orderings.BIG")
    known = [k for k in lo.BIG if k != DEEPSEEK]
    V = J["values"]

    def g(keys, r, t, s):
        g1 = [V[r][t][k][s]["discovery"]["cot_minus_requested"] for k in keys]
        g2 = [V[r][t][k][s]["discovery"]["cot_minus_forced"] for k in keys]
        return g1, g2

    blocks, checks, m = {}, [], {}
    for (s, r), v_ in lo.VARIANT.items():
        big = {t: lo.reversals(lo.BIG, *g(lo.BIG, r, t, s)) for t in lo.TASKS}
        st = J["reversals"]["big"][r][s]
        for t in lo.TASKS:
            a, b = big[t], st[t]
            ok = (a["reversed"], a["pairs"], a["ties"], a["reversed_pairs"]) == \
                 (b["reversed"], b["pairs"], b["ties"], b["reversed_pairs"])
            checks.append({"variant": f"{s}/{r}", "task": t, "equal": ok})
            if not ok:
                raise SystemExit(f"{s} {r} {t}: the four-model count does not reproduce ladder_orderings.json")
        kb = {t: lo.reversals(known, *g(known, r, t, s)) for t in lo.TASKS}
        for t in lo.TASKS:
            exp = [p for p in st[t]["reversed_pairs"] if DEEPSEEK not in p]
            if kb[t]["reversed_pairs"] != exp:
                raise SystemExit(f"{s} {r} {t}: the three-model pairs are not the four-model pairs without DeepSeek")
        kb["pooled"] = lo.pooled_counts({t: kb[t] for t in lo.DISC3})
        blocks[f"{s}/{r}"] = {t: {k: x[k] for k in ("reversed", "pairs", "ties") if k in x}
                              | ({"reversed_pairs": x["reversed_pairs"], "tied_pairs": x["tied_pairs"],
                                  "reversed_gaps": x["reversed_gaps"]} if t != "pooled" else {})
                              for t, x in kb.items()}
        for t in list(lo.TASKS) + ["pooled"]:
            tag = "Pooled" if t == "pooled" else lo.T_TAG[t]
            m[f"ladBigKnownRev{tag}{v_}"] = lo.mnum(kb[t]["reversed"], 0)
            m[f"ladBigKnownPairs{tag}{v_}"] = lo.mnum(kb[t]["pairs"], 0)
            m[f"ladBigKnownTies{tag}{v_}"] = lo.mnum(kb[t]["ties"], 0)
    R["ladder_known"] = {"models": known, "left_out": DEEPSEEK, "source": "results/ladder/analysis/ladder_orderings.json "
                         f"(created {J['created']}, readers {J['readers']})", "basis": "model means over the 24 "
                         "discovery draws; cut reading primary (no suffix), Strict, Uncut, StrictUncut as the lad macros",
                         "checks": checks, "blocks": blocks, "macros": m}


# ============================================================================ recent checkpoints


def release_dates(keys) -> tuple[dict, list]:
    """{model key: {'repo', 'created'}} from the registries and the panel manifest; keys without a recorded date."""
    from src import hotpot_d2
    reg = hotpot_d2.load_registries()
    man = {p["repo"]: p for p in load_json("manifests", "d2_panel_extension.json")["panel"]}
    meta = load_json("manifests", "external", "hf_metadata_20261030.json")
    meta_created = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, dict) and v.get("created"):
                meta_created[v.get("repo", k)] = v["created"]
    elif isinstance(meta, list):
        for v in meta:
            if isinstance(v, dict) and v.get("created"):
                meta_created[v.get("repo") or v.get("id")] = v["created"]
    out, missing = {}, []
    for k in keys:
        repo = (reg.get(k) or {}).get("hf_id")
        p = man.get(repo)
        if not repo or not p or not p.get("created"):
            missing.append({"model_key": k, "repo": repo})
            continue
        mc = meta_created.get(repo)
        out[k] = {"repo": repo, "created": p["created"], "hf_metadata_created": mc,
                  "agrees_with_hf_metadata": (None if mc is None else str(mc)[:10] == p["created"])}
    return out, missing


def recent_part(R: dict) -> None:
    import ladder_orderings as lo
    from src import robustness as rb
    P, notes = lo.panel_means()
    AE = load_json("results", "arm_effect", "arm_effect.json")["tasks"]
    HC = load_json("results", "hotpot_content", "hotpot_content.json")["arm_effect"]["content"]["reversals"]
    score_of = {"gsm8k": "strict", "math": "strict", "bbh": "strict", "hotpot": "content"}
    allkeys = sorted({k for t in P for k in P[t]["strict"]["keys"]})
    dates, missing = release_dates(allkeys)
    out, m, recent_union = {}, {}, set()
    for t, s in score_of.items():
        p = P[t][s]
        keys = list(p["keys"])
        g1, g2 = p["cot"] - p["requested"], p["cot"] - p["forced"]
        full = rb.pair_reversals(g1, g2)
        stored = AE[t]["reversals"] if s == "strict" else HC
        if (full["reversed"], full["pairs"]) != (stored["reversed"], stored["pairs"]):
            raise SystemExit(f"{t} {s}: the full panel's count {full['reversed']} of {full['pairs']} differs from the "
                             f"stored {stored['reversed']} of {stored['pairs']}")
        sel = [j for j, k in enumerate(keys) if k in dates and dates[k]["created"] >= RECENT_FROM]
        rk = [keys[j] for j in sel]
        recent_union |= set(rk)
        x = rb.pair_reversals(g1[sel], g2[sel])
        # sensitivity: the panel model whose recorded date is closest before the cut, added
        before = sorted((dates[k]["created"], k) for k in keys if k in dates and dates[k]["created"] < RECENT_FROM)
        near = before[-1][1] if before else None
        sel2 = sorted(sel + ([keys.index(near)] if near else []))
        x2 = rb.pair_reversals(g1[sel2], g2[sel2])
        out[t] = {"score": s, "models": rk, "M": len(rk), "reversed": x["reversed"], "pairs": x["pairs"],
                  "ties": x["ties"], "share": x["reversed_share"],
                  "full_panel": {"M": len(keys), "reversed": full["reversed"], "pairs": full["pairs"]},
                  "older_models": [k for k in keys if k not in rk],
                  "with_latest_before_cut": {"added": near, "added_created": dates[near]["created"] if near else None,
                                             "M": len(sel2), "reversed": x2["reversed"], "pairs": x2["pairs"]}}
        tg = T_TAG[t]
        m[f"aeRecent{tg}Reversed"] = str(x["reversed"])
        m[f"aeRecent{tg}Pairs"] = str(x["pairs"])
        m[f"aeRecent{tg}Ties"] = str(x["ties"])
        m[f"aeRecent{tg}ReversedPct"] = num(None if x["reversed_share"] is None else 100 * x["reversed_share"], 0)
        m[f"aeRecent{tg}M"] = str(len(rk))
    m["aeRecentM"] = str(len(recent_union))
    R["recent"] = {"cut": f"Hugging Face repository created on or after {RECENT_FROM}",
                   "date_source": f"{rel(PANEL_MANIFEST)} ('created', from {rel(HF_METADATA)}); model key to "
                                  "repository by the model registries (src/hotpot_d2.load_registries)",
                   "basis": "model means over the 24 discovery draws (scripts/ladder_orderings.panel_means); strict on "
                            "GSM8K, MATH-500 and BIG-Bench Hard, content on HotpotQA, as \\ae<T>Reversed and "
                            "\\aeHqReversedContent",
                   "dates": dates, "no_recorded_date": missing, "union_M": len(recent_union),
                   "union_models": sorted(recent_union), "panel_checks": notes, "tasks": out, "macros": m}


# ============================================================================ Qwen among the thinking models


def thinking_qwen_part(R: dict) -> None:
    from src import hotpot_d2
    G = load_json("results", "generality", "generality.json")
    row = G["rows"]["thi_gsm8k_arm"]
    panel = list(row["panel"])
    if len(panel) != int(row["M"]):
        raise SystemExit("thi_gsm8k_arm: panel length differs from M")
    reg = hotpot_d2.load_registries()
    fam = {k: (reg.get(k) or {}).get("family") for k in panel}
    if any(v is None for v in fam.values()):
        raise SystemExit(f"no registry family for {[k for k, v in fam.items() if v is None]}")
    q = [k for k in panel if fam[k] == "Qwen"]
    R["thinking_qwen"] = {"row": "thi_gsm8k_arm", "M": len(panel), "families": fam, "qwen": q,
                          "macros": {"genThiGsmArmQwen": str(len(q))}}


# ============================================================================ output


def summary(R: dict) -> str:
    L = [f"# Supporting quantities ({REQUEST})", "",
         f"Script: scripts/statement_values.py; created {R['created']}.", ""]
    H = R.get("hotpot")
    if H:
        L += ["## HotpotQA on content exact match", "",
              f"Content reader {H['content_scorer']}, through {H['content_view']}. The same code on the strict table "
              f"reproduces the {len(H['strict_reproduced'])} stored strict values checked.", "",
              "| Quantity | Strict (stored) | Content | Basis |", "|---|---:|---:|---|"]
        S, C = H["strict"], H["content"]
        for c, nm in (("cot_req", "requested"), ("cot_forced", "forced")):
            L.append(f"| Mean CoT gain over {nm} (pp) | {num(S['step2'][c]['mean_gain'], 2)} | "
                     f"{num(C['step2'][c]['mean_gain'], 2)} | discovery draws |")
            L.append(f"| R of benefit, {nm} | {num(S['step2'][c]['R_benefit'], 3)} | "
                     f"{num(C['step2'][c]['R_benefit'], 3)} | discovery draws |")
            L.append(f"| Predicted one-draw C, {nm} | {num(S['checks'][c]['a']['predicted'], 3)} | "
                     f"{num(C['checks'][c]['a']['predicted'], 3)} | first four discovery draws; content refit exploratory |")
            L.append(f"| Observed one-draw C, {nm} | {num(S['checks'][c]['a']['observed'], 3)} | "
                     f"{num(C['checks'][c]['a']['observed'], 3)} | validation draws |")
        so, co = S["checks"]["c_arm_switch"]["observed"], C["checks"]["c_arm_switch"]["observed"]
        L.append(f"| Pairs reversed by the arm switch | {num(so['switch']['opposite'], 3)} | "
                 f"{num(co['switch']['opposite'], 3)} | validation draws |")
        L.append(f"| Pairs reversed by a new draw | {num(so['new_draw']['opposite'], 3)} | "
                 f"{num(co['new_draw']['opposite'], 3)} | validation draws |")
        L += ["", f"Switch reorders more than a new draw under content: {co['switch_reorders_more']} "
                  f"(tied pairs: switch {num(co['switch']['tie'], 3)}, new draw {num(co['new_draw']['tie'], 3)}).",
              "The saved strict predictions of C against the content-scored validation draws: "
              + "; ".join(f"{c} saved {num(v['saved_predicted'], 3)}, observed {num(v['content_observed'], 3)}, within "
                          f"0.05 {v['within_0.05']}" for c, v in H["saved_strict_prediction_against_content"].items())
              + ".", ""]
    B = R.get("best_third")
    if B:
        L += ["## Best third and the decision costs", "", f"Answer: {B['answer']}. {B['note']}.", "",
              "| Benchmark | Configurations | M | Third | Best third, here | stored | FdToReq, here | stored |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for t, x in B["tasks"].items():
            L.append(f"| {t} | {x['n_configurations']} | {x['M']} | {x['n_chosen']} | {num(x['best_third_here'], 2)} | "
                     f"{num(x['best_third_stored'], 2)} | {num(x['cost_fd_to_req_here'], 2)} | "
                     f"{num(x['cost_fd_to_req_stored'], 2)} |")
        L.append("")
    K = R.get("ladder_known")
    if K:
        L += ["## 70B and above, size known", "", f"Models: {', '.join(K['models'])} ({K['left_out']} left "
              f"out). Source: {K['source']}. The four-model counts reproduce the stored ones in every variant.", "",
              "| Variant | GSM8K | MATH-500 | BIG-Bench Hard | Pooled (first three) | HotpotQA |",
              "|---|---|---|---|---|---|"]
        for v, b in K["blocks"].items():
            L.append(f"| {v} | " + " | ".join(f"{b[t]['reversed']} of {b[t]['pairs']}" + (f" ({b[t]['ties']} tied)"
                                                                                       if b[t]["ties"] else "")
                                              for t in ("gsm8k", "math", "bbh", "pooled", "hotpot")) + " |")
        L += ["", "Reversed pairs, with the two gaps (pp; CoT - requested, CoT - forced):", ""]
        for v in ("content/cut", "strict/cut"):
            for t in ("gsm8k", "math", "bbh", "hotpot"):
                gs = K["blocks"][v][t]["reversed_gaps"]
                L.append(f"- {v}, {t}: " + ("; ".join(f"{g['pair'][0]} and {g['pair'][1]} ({g['printed'][0]}, "
                                                       f"{g['printed'][1]})" for g in gs) if gs else "none"))
        L.append("")
    Rc = R.get("recent")
    if Rc:
        L += ["## Recent checkpoints", "", f"Cut: {Rc['cut']}. Dates: {Rc['date_source']}. Models without a "
              f"recorded date: {len(Rc['no_recorded_date'])}"
              + (f" ({', '.join(x['model_key'] for x in Rc['no_recorded_date'])})" if Rc["no_recorded_date"] else "")
              + f". Union over the four benchmarks: {Rc['union_M']} models.", "",
              "| Benchmark | Score | Recent M | Reversed | Pairs | Ties | Full panel | With the latest model before the cut |",
              "|---|---|---:|---:|---:|---:|---|---|"]
        for t, x in Rc["tasks"].items():
            w = x["with_latest_before_cut"]
            L.append(f"| {t} | {x['score']} | {x['M']} | {x['reversed']} | {x['pairs']} | {x['ties']} | "
                     f"{x['full_panel']['reversed']} of {x['full_panel']['pairs']} | {w['added']} ({w['added_created']}): "
                     f"{w['reversed']} of {w['pairs']} |")
        L += ["", "Recent models: " + ", ".join(f"{k} ({Rc['dates'][k]['created']})" for k in Rc["union_models"]) + ".",
              ""]
    Q = R.get("thinking_qwen")
    if Q:
        L += ["## Qwen among the native-thinking checkpoints", "",
              f"{len(Q['qwen'])} of {Q['M']} ({', '.join(Q['qwen'])}); the other: "
              + ", ".join(f"{k} ({v})" for k, v in Q["families"].items() if v != "Qwen") + ".", ""]
    L += ["## Macros", "", "| Name | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(R["macros"].items())]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=PARTS)
    a = ap.parse_args(argv)
    path = os.path.join(OUT, f"{STEM}.json")
    R = load_json("results", "statement_values", f"{STEM}.json") if os.path.exists(path) else {}
    R.update({"request": REQUEST, "script": "scripts/statement_values.py", "created": time.strftime("%Y-%m-%d %H:%M")})
    fn = {"hotpot": hotpot_part, "best_third": best_third_part, "ladder_known": ladder_known_part,
          "recent": recent_part, "thinking_qwen": thinking_qwen_part}
    for p in a.only or PARTS:
        t0 = time.time()
        fn[p](R)
        print(f"{p}: done ({time.time() - t0:.0f} s)", flush=True)
    R["macros"] = {}
    for p in PARTS:
        for k, v in ((R.get(p) or {}).get("macros") or {}).items():
            if k in R["macros"]:
                raise SystemExit(f"macro {k} set twice")
            R["macros"][k] = v
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(R), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    txt = summary(R)
    for name in (f"{STEM}.md", f"{STEM}_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
