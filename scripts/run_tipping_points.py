r"""Tipping points of the headline results: the smallest number of individual scored
outputs whose score would have to be wrong, placed where they do the most harm, to reverse each result; and the t
interval for the mean of B.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_tipping_points.py 2>&1"

Results, each read from the draws and outputs of its source analysis:
    A  GSM8K-29, content scores, chain of thought - forced direct (scripts/run_outcomes.py --scoring content): the
       slope of gain on the forced baseline and the slope on the external index, cross-draw moments over the 24
       discovery draws (19 configurations). Reversed when the slope reaches zero or below.
    B  BIG-Bench Hard (28 models, 25 tasks of 54 items), content scores, requested - forced direct: the mean over the
       24 discovery draws (18 configurations) and the models. Reversed when it reaches zero or below. Beside it, the t
       interval over the 24 per-draw means (23 df), on content and on strict scores.
    C  The exemplar tercile reading (scripts/run_exemplar.py): terciles ranked by strict zero-shot accuracy over the
       discovery draws, benefits over the 24 validation draws (21 configurations). Under the strict scorer the claim
       holds; the count is the smallest of three routes to make it fail (the weak tercile's 95% lower bound to zero or
       below, the strong tercile's 90% upper bound to +2 pp or above, its 90% lower bound to -2 pp or below), with the
       ranking recomputed from the flipped zero-shot cells after each batch; of two flips with equal effect the
       eight-shot one comes first, since it leaves the ranking alone. The same searches with the terciles held at their
       starting members are a check. Under Cheng et al.'s corrected extractor the claim fails; the count is to make it
       hold (weak 95% lower bound above zero, strong 90% interval still inside +-2 pp), the ranking fixed at its strict
       values.

Each result is rebuilt from per-(model, arm, configuration) cell counts of the rows its source analysis loads (the same
loaders), and the run refuses unless the rebuilt draw arrays and values equal the source analysis's before any flip.
Two counts per result: unrestricted (any output may flip, within the correct and incorrect outputs of its cell) and
restricted (only outputs where the second scorer disagrees, each moved to that scorer's reading: strict against content
for A and B, strict against Cheng et al.'s extractor for C). Beside each count: the outputs that enter the result (A and
B: both arms of every cell of the discovery draws; C: both arms of the validation draws and the zero-shot outputs of the
discovery draws, which set the ranking) and those among them where the two scorers disagree. Search, count and edge
cases: src/tipping_points.py.

Outputs: results/tipping_points/tipping_points.json (with "macros", prefix tp), summary.md, summary_clean.md,
tables/tipping_points.csv (one row per search) and tables/flipped_cells.csv (every cell a solution flips).
"""
import argparse
import json
import os
import platform
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
from src import dataset2 as d2, tipping_points as tp  # noqa: E402
from run_forced_direct import dump, fmt  # noqa: E402

OUT = os.path.join(ROOT, "results", "tipping_points")
CONFIG = os.path.join(ROOT, "config", "instability.yaml")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
EXEMPLAR = os.path.join(ROOT, "data", "processed", "exemplar")
TOL = 1e-9
KEY = ["model_key", "arm", "config_id", "item_id"]
ROUTE_WORDS = {"weak_lower": "weak tercile's 95% lower bound to zero or below",
               "strong_upper": "strong tercile's 90% upper bound to +2 pp or above",
               "strong_lower": "strong tercile's 90% lower bound to −2 pp or below",
               "hold": "weak tercile's 95% lower bound above zero"}
ROUTE_TAG = {"weak_lower": "Weak", "strong_upper": "StrongUp", "strong_lower": "StrongDown", "hold": "Weak"}
ROUTE_TEX = {"weak_lower": r"weak tercile's 95\% lower bound", "strong_upper": r"strong tercile's 90\% upper bound",
             "strong_lower": r"strong tercile's 90\% lower bound", "hold": r"weak tercile's 95\% lower bound"}
FLIPS = {"unrestricted": "", "restricted": "Restr"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def stored(path):
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def check(label, got, ref, tol=TOL):
    dev = float(np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float))))
    if not dev <= tol:
        raise SystemExit(f"refused: {label} does not reproduce its source ({got} against {ref}; deviation {dev:.3g})")
    return dev


# ============================================================================ cells


def cell_counts(rows, read, other, arm_names, keys, n_items):
    """Per cell of `keys` (model, arm, configuration): items, outputs correct under `read`, and the outputs where `other`
    disagrees (here_only: correct under `read` and incorrect under `other`; there_only: the reverse). Refused unless
    every cell has every item once."""
    r = rows[rows["arm"].isin(arm_names)][KEY + [read, other]].copy()
    r["config_id"] = r["config_id"].astype(str)
    r["arm"] = r["arm"].map(arm_names)
    if r.duplicated(KEY).any():
        raise SystemExit("refused: duplicated (model, arm, configuration, item) rows")
    a, b = r[read].astype(int), r[other].astype(int)
    if not (a.isin([0, 1]).all() and b.isin([0, 1]).all()):
        raise SystemExit(f"refused: {read} or {other} is not 0/1")
    r = r.assign(_c=a, _h=((a == 1) & (b == 0)).astype(int), _t=((a == 0) & (b == 1)).astype(int))
    g = r.groupby(["model_key", "arm", "config_id"]).agg(n=("item_id", "size"), correct=("_c", "sum"),
                                                         here_only=("_h", "sum"), there_only=("_t", "sum"))
    g = g.reindex(pd.MultiIndex.from_tuples(keys, names=["model_key", "arm", "config_id"]))
    if g["n"].isna().any() or (g["n"] != n_items).any():
        raise SystemExit(f"refused: {int((g['n'].isna() | (g['n'] != n_items)).sum())} cells lack the item set of "
                         f"{n_items}")
    return {k: g[k].to_numpy(np.int64) for k in ("correct", "here_only", "there_only")}


def entering(cells, n):
    return {"outputs": int(n * len(cells["correct"])),
            "disagreeing": int(cells["here_only"].sum() + cells["there_only"].sum()),
            "correct_here_only": int(cells["here_only"].sum()), "correct_there_only": int(cells["there_only"].sum())}


SHARE = tp.BATCH_SHARE


def search(make, cells, n, keys, configs_rows, verify):
    """Both searches (unrestricted, restricted) of one result; `make(down, up)` builds its problem. `verify(v)` reads the
    end state's cell scores with the source analysis's own estimator."""
    out = {}
    for mode in ("unrestricted", "restricted"):
        down, up = (tp.unrestricted(cells["correct"], n) if mode == "unrestricted"
                    else tp.restricted(cells["here_only"], cells["there_only"]))
        t0 = time.time()
        r = tp.greedy(make(down, up), share=SHARE)
        k = r.pop("k")
        r["source_check"] = verify(tp.scores(cells["correct"] + k, n))
        if r["status"] == "reversed" and not r["source_check"]["reversed"]:
            raise SystemExit(f"refused: the source estimator does not confirm the reversal ({r['source_check']})")
        r["composition"] = tp.composition(k, keys, cells["here_only"], cells["there_only"])
        r["flipped_cells"] = [{"model_key": keys[i][0], "arm": keys[i][1], "config_id": keys[i][2],
                               "rows": configs_rows.get(keys[i][2], 0), "net_flips": int(k[i]),
                               "correct_before": int(cells["correct"][i]), "n_items": int(n),
                               "here_only": int(cells["here_only"][i]), "there_only": int(cells["there_only"][i])}
                              for i in np.flatnonzero(k)]
        r["seconds"] = round(time.time() - t0, 1)
        out[mode] = r
    return out


def rows_per_config(draws):
    s = pd.Series(list(map(str, draws))).value_counts()
    return {str(c): int(v) for c, v in s.items()}


# ============================================================================ A and B: content scores, discovery draws


def content_setting(task):
    """Panel, items, discovery draws, the source's content and strict three-arm arrays and the rows behind them."""
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    from run_outcomes import with_content
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    fd_load, _, _ = forced_inputs(task, panel, s, items_info, inp["load"], False)
    col_load, fd_load = with_content(inp["load"], task), with_content(fd_load, task)
    disc = list(s["discovery_config_ids"])
    D = three_arm_arrays(col_load, fd_load, disc, panel, items, value="content_correct")
    rows = pd.concat([col_load(disc), fd_load(disc)], ignore_index=True)
    return {"panel": panel, "items": items, "n": len(items), "disc": disc, "D": D, "rows": rows,
            "sampler_sha256": s["sha256"]}


def run_a(x, alpha, method):
    """A: GSM8K-29 content scores, chain of thought - forced direct, slopes on the forced baseline and on the index."""
    from run_index_checks import slope_block
    from run_transfer_tasks import eci_for
    panel, n, disc = x["panel"], x["n"], x["disc"]
    _, Bfd, Pc = x["D"]
    e = eci_for(panel)
    src = slope_block(Bfd, Pc, n, e, method, alpha)["scales"]["points"]
    ref = stored("results/outcomes_content/outcomes_content.json")["tasks"]["gsm8k"]["outcomes"]["CotMinusForced"]
    ref = ref["slopes"]["scales"]["points"]
    configs = sorted(set(map(str, disc)))
    keys = [(m, a, c) for m in panel for a in ("forced", "cot") for c in configs]
    cells = cell_counts(x["rows"], "content_correct", "strict_correct",
                        {"forced_direct": "forced", "zero_shot_cot": "cot"}, keys, n)
    iB, iP = tp.cell_index(keys, panel, "forced", disc), tp.cell_index(keys, panel, "cot", disc)
    v = tp.scores(cells["correct"], n)
    chk = {"arrays_forced": check("A: forced-direct array", v[iB], Bfd, 0.0),
           "arrays_cot": check("A: chain-of-thought array", v[iP], Pc, 0.0)}
    stats = {"baseline": tp.baseline_slope, "index": lambda B, P: tp.index_slope(B, P, e)}
    out = {"task": "gsm8k", "name": "GSM8K-29", "scoring": "content", "outcome": "CotMinusForced",
           "baseline_arm": "forced_direct", "intervention_arm": "zero_shot_cot", "M": len(panel), "n_items": n,
           "K": len(disc), "configurations": len(configs), "entering": entering(cells, n), "statistics": {}}
    rows = rows_per_config(disc)
    for name, f in stats.items():
        s0 = f(v[iB], v[iP])[0]
        chk[f"{name}_slope_source"] = check(f"A: {name} slope", s0, src[name]["slope"])
        chk[f"{name}_slope_stored"] = check(f"A: {name} slope (stored)", s0, ref[name]["slope"])

        def verify(vk, name=name):
            sl = slope_block(vk[iB], vk[iP], n, e, method, alpha)["scales"]["points"][name]["slope"]
            return {"estimator": "run_index_checks.slope_block", "slope": sl, "reversed": bool(sl <= tp.TIE_TOL)}
        res = search(lambda dn, up, f=f: tp.at_or_below_zero_problem(cells["correct"], n, dn, up, iB, iP, f, "slope"),
                     cells, n, keys, rows, verify)
        out["statistics"][name] = {"value": s0, "set": src[name]["slope_set"], "reversal": "slope at zero or below",
                                   "searches": res}
        print(f"A {name}: slope {s0:.5f}; counts " + ", ".join(f"{m} {r['count']} ({r['status']})"
                                                               for m, r in res.items()), flush=True)
    out["checks"] = chk
    return out


def run_b(x):
    """B: BIG-Bench Hard content scores, requested - forced direct, the mean; and its t interval."""
    panel, n, disc = x["panel"], x["n"], x["disc"]
    Breq, Bfd, _ = x["D"]
    configs = sorted(set(map(str, disc)))
    keys = [(m, a, c) for m in panel for a in ("forced", "requested") for c in configs]
    cells = cell_counts(x["rows"], "content_correct", "strict_correct",
                        {"forced_direct": "forced", "direct_answer": "requested"}, keys, n)
    strict = cell_counts(x["rows"], "strict_correct", "content_correct",
                         {"forced_direct": "forced", "direct_answer": "requested"}, keys, n)
    iB, iP = tp.cell_index(keys, panel, "forced", disc), tp.cell_index(keys, panel, "requested", disc)
    v, vs = tp.scores(cells["correct"], n), tp.scores(strict["correct"], n)
    m0 = tp.mean_gain(v[iB], v[iP])[0]
    oc = stored("results/outcomes_content/outcomes_content.json")["tasks"]["bbh"]["outcomes"]["ReqMinusForced"]["mean"]
    os_ = stored("results/outcomes/outcomes.json")["tasks"]["bbh"]["outcomes"]["ReqMinusForced"]["mean"]
    chk = {"arrays_forced": check("B: forced-direct array", v[iB], Bfd, 0.0),
           "arrays_requested": check("B: requested-direct array", v[iP], Breq, 0.0),
           "mean_source": check("B: mean", m0, float((Breq - Bfd).mean())),
           "mean_stored": check("B: mean (stored)", m0, oc),
           "strict_mean_stored": check("B: strict mean (stored)", tp.mean_gain(vs[iB], vs[iP])[0], os_)}
    intervals = {}
    for lab, vv in (("content", v), ("strict", vs)):
        d = (vv[iP] - vv[iB]).mean(1)                      # per-draw mean over the models
        intervals[lab] = {"mean": float(d.mean()), "ci95": tp.t_interval(d, 0.95), "df": len(d) - 1,
                          "per_draw": d.tolist()}
    def verify(vk):
        mk = float((vk[iP] - vk[iB]).mean())               # run_outcomes.outcome_block's r["mean"]
        return {"estimator": "mean of P - B", "mean": mk, "reversed": bool(mk <= tp.TIE_TOL)}
    res = search(lambda dn, up: tp.at_or_below_zero_problem(cells["correct"], n, dn, up, iB, iP, tp.mean_gain, "mean"),
                 cells, n, keys, rows_per_config(disc), verify)
    print(f"B: mean {m0:.4f} [{intervals['content']['ci95'][0]:.3f}, {intervals['content']['ci95'][1]:.3f}]; counts "
          + ", ".join(f"{m} {r['count']} ({r['status']})" for m, r in res.items()), flush=True)
    return {"task": "bbh", "name": "BIG-Bench Hard", "scoring": "content", "outcome": "ReqMinusForced",
            "baseline_arm": "forced_direct", "intervention_arm": "direct_answer", "M": len(panel), "n_items": n,
            "K": len(disc), "configurations": len(configs), "entering": entering(cells, n),
            "statistics": {"mean": {"value": m0, "reversal": "mean at zero or below", "interval": intervals,
                                    "searches": res}}, "checks": chk}


# ============================================================================ C: the exemplar terciles


def run_c():
    from run_exemplar import (EQ_MARGIN, arrays, collection_inputs, combined_loader, cut_table, exemplar_inputs,
                              long_arrays, tercile_block)
    inp, s, items, items_info = collection_inputs("gsm8k")
    ex_load, _, panel = exemplar_inputs(list(inp["panel"]), s, items_info, inp["load"], False, False)
    load = combined_loader(inp["load"], ex_load, cut_table())
    disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
    n = len(items)
    Bd, _ = arrays(load, disc, panel, items)
    Bv, Pv = arrays(load, val, panel, items)
    cheng = pd.read_parquet(os.path.join(EXEMPLAR, "cheng_scored.parquet"))
    cheng = cheng[cheng["model_key"].isin(panel)]
    Bcv, Pcv = long_arrays(cheng, val, panel)
    rank = Bd.mean(0)
    src = {"strict": tercile_block(panel, rank, Bv, Pv), "cheng": tercile_block(panel, rank, Bcv, Pcv)}
    ref = stored("results/exemplar/exemplar.json")["terciles"]

    # rows of both arms under the reproduction's arm names, with both scorers' readings of each output
    zc, vc = sorted(set(map(str, disc)) | set(map(str, val))), sorted(set(map(str, val)))
    rows = load(zc)
    rows = rows.assign(arm=rows["arm"].map({"direct_answer": "zero_shot_cot", "zero_shot_cot": "eight_shot_cot"}),
                       config_id=rows["config_id"].astype(str))
    rows = rows[(rows["arm"] == "zero_shot_cot") | rows["config_id"].isin(vc)]
    ch = cheng[KEY + ["correct"]].assign(config_id=cheng["config_id"].astype(str))
    ch["item_id"] = ch["item_id"].astype(rows["item_id"].dtype)
    j = rows[KEY + ["strict_correct"]].merge(ch.rename(columns={"correct": "cheng_correct"}), on=KEY, how="left",
                                             validate="one_to_one")
    if j["cheng_correct"].isna().any():
        raise SystemExit(f"refused: {int(j['cheng_correct'].isna().sum())} outputs have no Cheng et al. reading")
    arm_names = {"zero_shot_cot": "zero_shot", "eight_shot_cot": "eight_shot"}
    # A zero-shot flip and an eight-shot flip of the same model and configuration have equal effects on the benefits;
    # the eight-shot cells come first, so a tie goes to the flip that leaves the ranking alone (the first-order effect
    # does not see the ranking).
    zs, es = [(m, "zero_shot", c) for m in panel for c in zc], [(m, "eight_shot", c) for m in panel for c in vc]
    keys = es + zs
    cells = {"strict": cell_counts(j, "strict_correct", "cheng_correct", arm_names, keys, n),
             "cheng": cell_counts(j, "cheng_correct", "strict_correct", arm_names, keys, n)}
    iBd = tp.cell_index(keys, panel, "zero_shot", disc)
    iBv, iPv = tp.cell_index(keys, panel, "zero_shot", val), tp.cell_index(keys, panel, "eight_shot", val)
    vs, vch = tp.scores(cells["strict"]["correct"], n), tp.scores(cells["cheng"]["correct"], n)
    chk = {"arrays_zero_shot_discovery": check("C: zero-shot discovery array", vs[iBd], Bd, 0.0),
           "arrays_strict_validation": max(check("C: strict validation arrays", vs[iBv], Bv, 0.0),
                                           check("C: strict validation arrays", vs[iPv], Pv, 0.0)),
           "arrays_cheng_validation": max(check("C: Cheng validation arrays", vch[iBv], Bcv),
                                          check("C: Cheng validation arrays", vch[iPv], Pcv))}
    rank_fixed = lambda v: rank                                # noqa: E731  (Cheng: ranked by strict, unchanged)
    rank_strict = lambda v: v[iBd].mean(0)                     # noqa: E731  (strict: recomputed from the flipped cells)
    out = {"name": "GSM8K-29, exemplar terciles", "M": len(panel), "n_items": n, "K_discovery": len(disc), "K_validation": len(val),
           "configurations_validation": len(vc), "configurations_zero_shot": len(zc), "margin_pp": EQ_MARGIN,
           "k": src["strict"]["k"], "readings": {}}
    rows_val = rows_per_config(val)
    for sc, vv, rank_of, routes in (("strict", vs, rank_strict, ("weak_lower", "strong_upper", "strong_lower")),
                                    ("cheng", vch, rank_fixed, ("hold",))):
        t0 = tp.tercile_reading(rank_of(vv), vv[iBv], vv[iPv], EQ_MARGIN)
        for g in ("weak", "strong"):
            if [panel[i] for i in t0["members"][g]] != src[sc][g]["models"]:
                raise SystemExit(f"refused: C {sc}: the {g} tercile differs from run_exemplar.tercile_block")
            for q in ("mean", "ci95", "ci90"):
                chk[f"{sc}_{g}_{q}"] = max(check(f"C {sc} {g} {q}", t0[g][q], src[sc][g][q]),
                                           check(f"C {sc} {g} {q} (stored)", t0[g][q], ref[sc][g][q]))
        if t0["claim_holds"] != src[sc]["claim_holds"] or t0["claim_holds"] != ref[sc]["claim_holds"]:
            raise SystemExit(f"refused: C {sc}: the claim's reading differs from its source")
        c = cells[sc]

        def routes_for(rank_of, c=c, sc=sc):
            def verify(vk):
                tb = tercile_block(panel, rank_of(vk), vk[iBv], vk[iPv])
                return {"estimator": "run_exemplar.tercile_block", "claim_holds": tb["claim_holds"],
                        "weak_lo95": tb["weak"]["ci95"][0], "strong_ci90": tb["strong"]["ci90"],
                        "reversed": tb["claim_holds"] != src[sc]["claim_holds"]}
            res = {}
            for route in routes:
                res[route] = search(lambda dn, up, route=route: tp.tercile_problem(
                    c["correct"], n, dn, up, iBv, iPv, rank_of, route, EQ_MARGIN), c, n, keys, rows_val, verify)
                for mode, r in res[route].items():
                    ve = r["value_end"]
                    r["members_changed"] = any([panel[i] for i in ve["members"][g]] != src[sc][g]["models"]
                                               for g in ("weak", "strong"))
            return res

        def smallest(res):
            best = {}
            for mode in ("unrestricted", "restricted"):
                done = [(res[r][mode]["count"], r) for r in routes if res[r][mode]["status"] == "reversed"]
                best[mode] = {"route": min(done)[1], "count": min(done)[0]} if done else {"route": None, "count": None}
            return best
        res = routes_for(rank_of)
        for route in routes:
            print(f"C {sc} {route}: counts " + ", ".join(f"{m} {r['count']} ({r['status']})"
                                                         for m, r in res[route].items()), flush=True)
        out["readings"][sc] = {"claim_holds": t0["claim_holds"], "start": {g: t0[g] for g in ("weak", "strong")},
                               "goal": "make the claim fail" if sc == "strict" else "make the claim hold",
                               "ranking": "recomputed from the flipped zero-shot cells after each batch"
                               if rank_of is not rank_fixed else "fixed at its strict values",
                               "entering": entering(c, n), "routes": res, "smallest": smallest(res)}
        if rank_of is not rank_fixed:
            # check: the same searches with the terciles held at their starting members
            fx = routes_for(rank_fixed)
            out["readings"][sc]["fixed_terciles_check"] = {
                "routes": {route: {m: {"status": r["status"], "count": r["count"]} for m, r in v.items()}
                           for route, v in fx.items()}, "smallest": smallest(fx)}
    out["checks"] = chk
    return out


# ============================================================================ macros, tables, summary


def count_txt(r):
    return f"{r['count']:,}".replace(",", "{,}") if r["status"] == "reversed" else "not reached"


def share_txt(r, total):
    return fmt(100 * r["count"] / total, 3) if r["status"] == "reversed" else "--"


def batch_txt(share):
    v = 100 * share
    return fmt(v, 0) if v == round(v) else fmt(v, 2)


def macros(R):
    m = {}

    def put(k, v):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
        m[k] = str(v).replace("−", "$-$")

    def big(x):
        return f"{int(x):,}".replace(",", "{,}")

    def counts(stem, res, total, end_key):
        for mode, t in FLIPS.items():
            r = res[mode]
            put(stem + t + "Count", count_txt(r))
            put(stem + t + "Share", share_txt(r, total))
            put(stem + t + "Cells", r["composition"]["cells"])
            put(stem + t + "CellsDisagree", big(r["composition"]["disagreeing_in_touched_cells"]))
            # where the flips run out first, the value they leave
            put(stem + t + "End", fmt(r["value_end"][end_key], 3) if r["status"] != "reversed" else "--")

    def ent(stem, e):
        put(stem + "Outputs", big(e["outputs"]))
        put(stem + "Disagree", big(e["disagreeing"]))
        put(stem + "DisagreeShare", fmt(100 * e["disagreeing"] / e["outputs"], 2))
    put("tpBatchShare", batch_txt(R["batch_share"]))
    a = R["A"]
    ent("tpGsm", a["entering"])
    put("tpGsmM", a["M"])
    put("tpGsmK", a["K"])
    put("tpGsmConfigs", a["configurations"])
    for name, t in (("baseline", "Base"), ("index", "Index")):
        st = a["statistics"][name]
        put(f"tpGsm{t}Slope", fmt(st["value"], 3))
        if st["set"]["kind"] == "bounded":
            put(f"tpGsm{t}SlopeLo", fmt(st["set"]["a"], 3))
            put(f"tpGsm{t}SlopeHi", fmt(st["set"]["b"], 3))
        counts(f"tpGsm{t}", st["searches"], a["entering"]["outputs"], "slope")
    b = R["B"]
    ent("tpBbh", b["entering"])
    put("tpBbhM", b["M"])
    put("tpBbhK", b["K"])
    put("tpBbhConfigs", b["configurations"])
    st = b["statistics"]["mean"]
    for lab, t in (("content", ""), ("strict", "Strict")):
        iv = st["interval"][lab]
        put(f"tpBbhMean{t}", fmt(iv["mean"], 2))
        put(f"tpBbhMean{t}Lo", fmt(iv["ci95"][0], 2))
        put(f"tpBbhMean{t}Hi", fmt(iv["ci95"][1], 2))
    counts("tpBbh", st["searches"], b["entering"]["outputs"], "mean")
    c = R["C"]
    ent("tpTer", c["readings"]["strict"]["entering"])          # the same outputs under either reading
    put("tpTerK", c["k"])
    for sc, s in (("strict", "Strict"), ("cheng", "Cheng")):
        x = c["readings"][sc]
        total = x["entering"]["outputs"]
        for mode, t in FLIPS.items():
            b_ = x["smallest"][mode]
            put(f"tpTer{s}{t}Count", "not reached" if b_["count"] is None else big(b_["count"]))
            put(f"tpTer{s}{t}Share", "--" if b_["count"] is None else fmt(100 * b_["count"] / total, 3))
            if b_["route"] is not None:
                co = x["routes"][b_["route"]][mode]["composition"]
                put(f"tpTer{s}{t}Cells", co["cells"])
                put(f"tpTer{s}{t}CellsDisagree", big(co["disagreeing_in_touched_cells"]))
            if sc == "strict":
                put(f"tpTer{s}{t}Route", "--" if b_["route"] is None else ROUTE_TEX[b_["route"]])
                for route, r in x["routes"].items():
                    put(f"tpTer{s}{t}{ROUTE_TAG[route]}Count", count_txt(r[mode]))
                fx = x["fixed_terciles_check"]["smallest"][mode]["count"]
                put(f"tpTer{s}{t}FixedCount", "not reached" if fx is None else big(fx))
    return m


def table_rows(R):
    rows, cells = [], []

    def add(result, statistic, reading, route, start, res, ent):
        for mode, r in res.items():
            co = r["composition"]
            row = {"result": result, "statistic": statistic, "reading": reading, "route": route, "flips": mode,
                   "start": start, "status": r["status"], "count": r["count"] if r["status"] == "reversed" else None,
                   "outputs_entering": ent["outputs"],
                   "share_pct": 100 * r["count"] / ent["outputs"] if r["status"] == "reversed" else None,
                   "disagreeing_entering": ent["disagreeing"], "cells": co["cells"], "models": co["models"],
                   "configurations": co["configurations"],
                   "disagreeing_in_touched_cells": co["disagreeing_in_touched_cells"], "batches": r["batches"],
                   "distance_start": r["distance_start"], "distance_end": r["distance_end"]}
            for arm, d in co["by_arm"].items():
                row[f"{arm}_to_correct"], row[f"{arm}_to_incorrect"] = d["to_correct"], d["to_incorrect"]
            rows.append(row)
            cells.extend({"result": result, "statistic": statistic, "reading": reading, "route": route, "flips": mode,
                          **c} for c in r["flipped_cells"])
    a = R["A"]
    for name, st in a["statistics"].items():
        add("A", f"slope on {name}", "content", "slope to zero", st["value"], st["searches"], a["entering"])
    b = R["B"]
    st = b["statistics"]["mean"]
    add("B", "mean", "content", "mean to zero", st["value"], st["searches"], b["entering"])
    for sc, x in R["C"]["readings"].items():
        for route, res in x["routes"].items():
            add("C", "tercile reading", sc, route, None, res, x["entering"])
    return rows, cells


def summary(R):
    a, b, c = R["A"], R["B"], R["C"]

    def cnt(r):
        return f"{r['count']:,}" if r["status"] == "reversed" else f"not reached ({r['status'].replace('_', ' ')})"

    def shr(r, tot):
        return f"{100 * r['count'] / tot:.3f}%" if r["status"] == "reversed" else "--"
    L = ["# Tipping points of the headline results", "",
         "For each result, the smallest number of individual scored outputs whose score "
         "would have to be wrong, placed where they do the most harm, to reverse it. One flipped output changes its "
         "(model, arm, configuration) cell by 100/n pp in every draw that uses the configuration, so a configuration "
         "drawn twice counts twice. Greedy search: the first-order effect of one flip in each cell, flips taken in order "
         f"of effect up to {batch_txt(R['batch_share'])}% of the remaining distance per batch (at least one), the "
         "statistic recomputed exactly after each batch, and the shortest reversing prefix of the last batch found by "
         "bisection. For a statistic linear in the cells (the mean of B, the slope on the index) this count is the "
         "minimum; for the others it is the greedy count. A flip is limited to the outputs of its cell (a correct output "
         "can only become incorrect, and the reverse). The restricted count flips only outputs where the two scorers "
         "disagree, each to the other scorer's reading; \"not reached\" means that no available flip shortens the "
         "distance any further, and the value at the end is what the flips leave. Each result was rebuilt from per-cell "
         "counts of the rows its source analysis loads, its draw arrays and values equal the source's before any flip, "
         "and every end state was read again with the source's own estimator (`checks` and `source_check` in the "
         "JSON).", ""]
    L += [f"## A. GSM8K-29, content scores, chain of thought − forced direct ({a['M']} models, {a['K']} discovery "
          f"draws, {a['configurations']} configurations)", "",
          f"Outputs entering the result: {a['entering']['outputs']:,} (both arms of every cell); the strict and content "
          f"scorers disagree on {a['entering']['disagreeing']:,} of them "
          f"({100 * a['entering']['disagreeing'] / a['entering']['outputs']:.2f}%; content correct and strict incorrect "
          f"{a['entering']['correct_here_only']:,}, the reverse {a['entering']['correct_there_only']:,}). Reversal: the "
          "slope reaches zero or below.", "",
          "| Slope | Value | Set | Flips | Count | Share | Value at the end | Cells | Disagreeing in those cells | "
          "To correct (forced / cot) | To incorrect (forced / cot) |", "|---|---:|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name, st in a["statistics"].items():
        s_ = st["set"]
        for mode, r in st["searches"].items():
            co = r["composition"]
            ba = co["by_arm"]
            end = f"{r['value_end']['slope']:.2e}".replace("-", "−")
            L.append(f"| {name} | {fmt(st['value'], 4)} | [{fmt(s_['a'], 3)}, {fmt(s_['b'], 3)}] | {mode} | {cnt(r)} | "
                     f"{shr(r, a['entering']['outputs'])} | {end} | {co['cells']} | {co['disagreeing_in_touched_cells']:,} | "
                     f"{ba['forced']['to_correct']:,} / {ba['cot']['to_correct']:,} | "
                     f"{ba['forced']['to_incorrect']:,} / {ba['cot']['to_incorrect']:,} |")
    st = b["statistics"]["mean"]
    iv = st["interval"]
    L += ["", f"## B. BIG-Bench Hard, content scores, requested − forced direct ({b['M']} models, {b['K']} discovery "
          f"draws, {b['configurations']} configurations)", "",
          f"Mean {fmt(iv['content']['mean'], 2)} pp, t interval over the {b['K']} per-draw means ({iv['content']['df']} "
          f"df) [{fmt(iv['content']['ci95'][0], 2)}, {fmt(iv['content']['ci95'][1], 2)}]; on strict scores "
          f"{fmt(iv['strict']['mean'], 2)} [{fmt(iv['strict']['ci95'][0], 2)}, {fmt(iv['strict']['ci95'][1], 2)}]. "
          "A configuration drawn twice gives two equal per-draw means, as in the source analysis.", "",
          f"Outputs entering the result: {b['entering']['outputs']:,}; the scorers disagree on "
          f"{b['entering']['disagreeing']:,} ({100 * b['entering']['disagreeing'] / b['entering']['outputs']:.2f}%; "
          f"content correct and strict incorrect {b['entering']['correct_here_only']:,}, the reverse "
          f"{b['entering']['correct_there_only']:,}). Reversal: the mean reaches zero or below.", "",
          "| Flips | Count | Share | Cells | Disagreeing in those cells | To correct (forced / requested) | "
          "To incorrect (forced / requested) |", "|---|---:|---:|---:|---:|---|---|"]
    for mode, r in st["searches"].items():
        co = r["composition"]
        ba = co["by_arm"]
        L.append(f"| {mode} | {cnt(r)} | {shr(r, b['entering']['outputs'])} | {co['cells']} | "
                 f"{co['disagreeing_in_touched_cells']:,} | {ba['forced']['to_correct']:,} / "
                 f"{ba['requested']['to_correct']:,} | {ba['forced']['to_incorrect']:,} / {ba['requested']['to_incorrect']:,} |")
    e = c["readings"]["strict"]["entering"]
    L += ["", f"## C. The exemplar tercile reading ({c['M']} models, terciles of {c['k']}, {c['K_validation']} validation "
          f"draws over {c['configurations_validation']} configurations)", "",
          f"Outputs entering the result: {e['outputs']:,} (both arms of the validation draws and the zero-shot outputs "
          f"of the discovery draws, {c['configurations_zero_shot']} zero-shot configurations in all); the strict scorer "
          f"and Cheng et al.'s extractor disagree on {e['disagreeing']:,} ({100 * e['disagreeing'] / e['outputs']:.2f}%; "
          f"strict correct and Cheng incorrect {e['correct_here_only']:,}, the reverse {e['correct_there_only']:,}). "
          "Under the strict scorer the ranking is recomputed from the flipped zero-shot cells after each batch, and of two "
          "flips with equal effect the eight-shot one is taken first, since it leaves the ranking alone; under Cheng et "
          "al.'s extractor the ranking stays at its strict values.", "",
          "| Reading | Start | Route | Flips | Count | Share | Cells | Disagreeing in those cells | Terciles changed | "
          "To correct (zero-shot / eight-shot) | To incorrect (zero-shot / eight-shot) |",
          "|---|---|---|---|---:|---:|---:|---:|---|---|---|"]
    for sc, x in c["readings"].items():
        s0 = x["start"]
        start = (f"weak {fmt(s0['weak']['mean'], 2)} [{fmt(s0['weak']['ci95'][0], 2)}, {fmt(s0['weak']['ci95'][1], 2)}]"
                 f"; strong 90% [{fmt(s0['strong']['ci90'][0], 2)}, {fmt(s0['strong']['ci90'][1], 2)}]; claim "
                 f"{'holds' if x['claim_holds'] else 'fails'}")
        for route, res in x["routes"].items():
            for mode, r in res.items():
                co = r["composition"]
                ba = co["by_arm"]
                L.append(f"| {sc} | {start} | {ROUTE_WORDS[route]} | {mode} | {cnt(r)} | {shr(r, x['entering']['outputs'])} "
                         f"| {co['cells']} | {co['disagreeing_in_touched_cells']:,} | "
                         f"{'yes' if r['members_changed'] else 'no'} | {ba['zero_shot']['to_correct']:,} / "
                         f"{ba['eight_shot']['to_correct']:,} | {ba['zero_shot']['to_incorrect']:,} / "
                         f"{ba['eight_shot']['to_incorrect']:,} |")
    L.append("")
    for sc, x in c["readings"].items():
        for mode, bst in x["smallest"].items():
            L.append(f"- {sc}, {mode} flips ({x['goal']}): " + (
                "no route reverses the reading." if bst["count"] is None else
                f"smallest count {bst['count']:,} ({100 * bst['count'] / x['entering']['outputs']:.3f}% of the outputs "
                f"entering), through the {ROUTE_WORDS[bst['route']]}."))
    fx = c["readings"]["strict"]["fixed_terciles_check"]
    L += ["", "Check, strict reading with the terciles held at their starting members: " + "; ".join(
        f"{ROUTE_WORDS[route]}: " + ", ".join(f"{m} {v['count']:,}" if v["status"] == "reversed" else f"{m} not reached"
                                             for m, v in res.items()) for route, res in fx["routes"].items()) + "."]
    L += ["", "Every flipped cell of every solution is in `tables/flipped_cells.csv` (net flips: positive from incorrect "
          "to correct); one row per search in `tables/tipping_points.csv`.", ""]
    return L


# ============================================================================ run


def run(a):
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    alpha, method = float(cfg["alpha"]), d2.load_selection(SELECTION)["H"]
    R = {"analysis": "tipping points of the headline results",
         "created": time.strftime("%Y-%m-%d %H:%M"), "units": "outputs; pp", "batch_share": SHARE,
         "tie_tolerance": tp.TIE_TOL, "slope_method": method}
    x = content_setting("gsm8k")
    print(f"GSM8K content arrays ({time.time() - t0:.0f} s)", flush=True)
    R["A"] = run_a(x, alpha, method)
    del x
    x = content_setting("bbh")
    print(f"BIG-Bench Hard content arrays ({time.time() - t0:.0f} s)", flush=True)
    R["B"] = run_b(x)
    del x
    R["C"] = run_c()
    print(f"exemplar terciles done ({time.time() - t0:.0f} s)", flush=True)
    R["macros"] = macros(R)
    rows, cells = table_rows(R)
    for part in ("A", "B"):
        for st in R[part]["statistics"].values():
            for r in st["searches"].values():
                r.pop("flipped_cells")
    for x_ in R["C"]["readings"].values():
        for res in x_["routes"].values():
            for r in res.values():
                r.pop("flipped_cells")
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "tables", "tipping_points.csv"), index=False)
    pd.DataFrame(cells).to_csv(os.path.join(a.out, "tables", "flipped_cells.csv"), index=False)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    dump(R, os.path.join(a.out, "tipping_points.json"))
    txt = "\n".join(summary(R)) + "\n"
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(f"done in {R['runtime_seconds']} s -> {rel(a.out)}", flush=True)
    return R


def main(argv=None):
    global SHARE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batch-share", type=float, default=tp.BATCH_SHARE,
                    help="the largest share of the remaining distance per batch (0.01); another value is a "
                         "check of the search and needs --out outside results/tipping_points")
    a = ap.parse_args(argv)
    if a.batch_share != tp.BATCH_SHARE and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("--batch-share other than 0.01 needs --out outside results/tipping_points")
    SHARE = a.batch_share
    run(a)


if __name__ == "__main__":
    main()
