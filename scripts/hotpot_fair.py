r"""The fair comparison on HotpotQA. Three choice sets, each saved before any HotpotQA validation draw is
scored, with the code that produced them for GSM8K, BIG-Bench Hard and MATH-500:

1. Strict decision design (scripts/run_decision_strict.py, src/decision_strict.py): one-draw pilots of 100 items,
   200 per design. Shared: items from all items, stream sampling.rng_for('rule-budgets', 'hotpot', 1, 100) with
   src/rule_budgets.draw_pilots, valued on all items. Disjoint: items split in halves by sha256('decision-strict/hotpot/
   <item_id>'), pilots from the pilot half (stream sampling.rng_for('decision-strict', 'hotpot', 1, 100)), valued on
   the other half only. Rules: linear in baseline, quadratic in baseline plus the index, own gain; random third.
   Per contrast; the pilots are shared by the three contrasts. File manifests/predictions_hotpot_decision_strict.*
2. Held-out family (scripts/run_transfer_regret.family_choices): within each family the top round(m / 3); rules fitted
   on the other families' 24-draw discovery means (linear in baseline, quadratic plus index, linear in index), applied
   at the family's baseline from one pilot draw (each distinct discovery configuration once, all items), against the
   family's own one-draw gain and a random third. File manifests/predictions_hotpot_family.*
3. Another benchmark (scripts/run_transfer_regret.task_choices), source GSM8K-29, BIG-Bench Hard-28 or MATH-500,
   on the models the source panel shares with HotpotQA's, top round(m / 3). Source arms by contrast: chain of thought
   against requested direct (cot_req), against forced direct (cot_forced), five-shot forced against zero-shot forced
   (fewshot). Source rows: discovery draws only.
   a. Saved with no HotpotQA data (manifests/predictions_hotpot_cross_benchmark.*, `run_hotpot.py --stage cross`):
      the source's own gain over its 24 discovery draws; its own gain from one discovery draw (each distinct source
      discovery configuration once); the source's capability rules (linear; quadratic plus index) at each model's
      source baseline (24-draw mean); the source's rule linear in the index (no call); a random third. The file also
      keeps the source means the rules of b need.
   b. Saved at the discovery stage from HotpotQA discovery rows (manifests/predictions_hotpot_cross_discovery.*): the
      source's capability rules applied at the HotpotQA one-draw pilot baselines (transfer_regret's design B), and the
      HotpotQA own gain from the same one draw, the comparator of every paired difference.
Evaluation (`run_hotpot.py` full run) reads the saved choices only (src/rule_budgets.load_frozen checks each npz
against its record): HotpotQA validation configurations absent from every discovery draw, each once (main set), and all
distinct validation configurations; regret in pp per chosen model, strict exact match.
"""
from __future__ import annotations

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

TASK = "hotpot"
N_PILOTS = 200
N_ITEMS = 100
SPLIT_LABEL = "decision-strict"
SOURCES = ("gsm8k", "bbh", "math")
SOURCE_TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
SOURCE_SETTING = {"cot_req": "{t}_req", "cot_forced": "{t}_fd", "fewshot": "fewshot_{t}"}
ARMS = ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced")
CROSS = "predictions_hotpot_cross_benchmark"
CROSS_DISC = "predictions_hotpot_cross_discovery"
STRICT = "predictions_hotpot_decision_strict"
FAMILY = "predictions_hotpot_family"
DISCOVERY_FILES = (STRICT, FAMILY, CROSS_DISC)
CROSS_RULES = ("source_own_all", "source_own_1", "source_linear_at_source", "source_quadratic_index_at_source",
               "source_index", "random")
CROSS_DISC_RULES = ("source_linear", "source_quadratic_index", "target_own_1")
CROSS_EVAL_ORDER = ("source_own_all", "source_own_1", "source_linear_at_source", "source_quadratic_index_at_source",
                    "source_index", "source_linear", "source_quadratic_index", "target_own_1", "random")
CROSS_NAMES = {"source_own_all": "source's own gain, 24 draws", "source_own_1": "source's own gain, one draw",
               "source_linear_at_source": "source rule linear in baseline, at source baselines",
               "source_quadratic_index_at_source": "source rule quadratic + index, at source baselines",
               "source_index": "source rule linear in index (no call)",
               "source_linear": "source rule linear in baseline, at HotpotQA pilot baselines",
               "source_quadratic_index": "source rule quadratic + index, at HotpotQA pilot baselines",
               "target_own_1": "HotpotQA own gain, one pilot draw", "random": "random third"}
NOTE_DISCOVERY = ("fitted on HotpotQA discovery rows only and saved before any HotpotQA validation draw is scored")


def files(stem: str, pred_dir: str) -> tuple[str, str]:
    jp = os.path.join(pred_dir, f"{stem}.json")
    return jp, jp[:-5] + ".npz"


def all_files(pred_dir: str) -> list[str]:
    """Every file of the fair comparison, the no-HotpotQA-data manifest first."""
    out = []
    for stem in (CROSS,) + DISCOVERY_FILES:
        out += list(files(stem, pred_dir))
    return out


def panel_eci(panel) -> list[float]:
    from run_transfer_tasks import eci_for
    return eci_for(list(panel))


def third(m: int) -> int:
    import run_transfer_regret as rtr
    return rtr.third(m)


# ============================================================================ item 3a: no HotpotQA data


def source_arrays(contrasts) -> dict:
    """{(source, contrast): (units, panel, B, P)} on the source's discovery draws (strict scoring, pp)."""
    import run_decision as rd
    return {(t, c): rd.setting_arrays(SOURCE_SETTING[c].format(t=t), "discovery") for t in SOURCES for c in contrasts}


def source_sampler_sha(task: str) -> str | None:
    p = os.path.join(ROOT, "manifests", f"sampler_{task}.json")
    return json.load(open(p, encoding="utf-8")).get("sha256") if os.path.exists(p) else None


def cross_record(panel, eci, src: dict) -> tuple[dict, dict]:
    """Record and arrays of item 3a from the sources' discovery arrays alone (src as source_arrays)."""
    from src import decision as dc, rule_budgets as rbud, transfer_regret as tr
    e_all = np.asarray(eci, float)
    arrays, meta = {}, {}
    for (s, c), (units, ps, B, P) in src.items():
        ps = list(map(str, ps))
        shared = [k for k in panel if k in ps]
        jt = [list(panel).index(k) for k in shared]
        js = [ps.index(k) for k in shared]
        n = third(len(shared))
        Bs, Gs = np.asarray(B, float)[:, js], (np.asarray(P, float) - np.asarray(B, float))[:, js]
        bm, gm, e = Bs.mean(0), Gs.mean(0), e_all[jt]
        rows = dc.distinct_rows(units)
        preds = {"source_own_all": gm[None], "source_own_1": Gs[rows],
                 "source_linear_at_source": tr.fit_apply(bm, gm, e, bm, e, "baseline_linear")[None],
                 "source_quadratic_index_at_source": tr.fit_apply(bm, gm, e, bm, e, "baseline_quadratic_index")[None],
                 "source_index": tr.fit_apply(bm, gm, e, bm, e, "index_linear")[None]}
        k = f"cross__{s}__{c}"
        for r, x in preds.items():
            arrays[f"{k}__{r}__pred"] = np.asarray(x, float)
            arrays[f"{k}__{r}__weights"] = tr.third_weights(x, n)
        arrays[f"{k}__random__weights"] = np.full((1, len(shared)), n / len(shared))
        arrays[f"{k}__source_baseline_mean"] = bm
        arrays[f"{k}__source_gain_mean"] = gm
        arrays[f"{k}__eci"] = e
        meta[f"{s}/{c}"] = {"source_setting": SOURCE_SETTING[c].format(t=s), "models": shared, "n_chosen": n,
                            "source_discovery_units": list(map(str, units)), "source_pilot_rows": rows,
                            "source_linear_slope": float(np.polyfit(bm, gm, 1)[0]),
                            "source_sampler_sha256": source_sampler_sha(s),
                            "source_arrays_sha256": rbud.arrays_sha256({"B": np.asarray(B, float),
                                                                        "P": np.asarray(P, float)})}
    rec = {"label": "hotpot_cross_benchmark", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": list(panel),
           "units": "pp; predictions uncentred; weights per model (ties share places)",
           "design": {"rules": list(CROSS_RULES), "pairs": meta, "third": "round(m / 3), at least 1"},
           "note": "fitted on GSM8K, BIG-Bench Hard and MATH-500 discovery draws only; reads no HotpotQA data; saved "
                   "before any HotpotQA output was scored"}
    return rec, arrays


def save_cross(panel, eci, src: dict, pred_dir: str, overwrite: bool = False) -> dict:
    from src import rule_budgets as rbud
    rec, arrays = cross_record(panel, eci, src)
    jp, npz = files(CROSS, pred_dir)
    return rbud.freeze(rec, arrays, jp, npz, overwrite=overwrite)


def load_cross(pred_dir: str, panel=None):
    """The saved item 3a record and arrays (hashes checked); refuses when absent or saved for another panel."""
    from src import rule_budgets as rbud
    jp, npz = files(CROSS, pred_dir)
    if not (os.path.exists(jp) and os.path.exists(npz)):
        raise SystemExit(f"{jp} is absent; run run_hotpot.py --stage cross before any HotpotQA scoring")
    try:
        rec, arrays = rbud.load_frozen(jp, npz)
    except ValueError as err:
        raise SystemExit(f"{jp}: {err}")
    if panel is not None and list(rec["models"]) != list(panel):
        raise SystemExit(f"{jp} was saved for another panel")
    return rec, arrays


# ============================================================================ HotpotQA item-level arrays


def dense(path: str, panel, configs, items) -> tuple[list[str], dict]:
    """(configurations, {arm: (C, M, N) int8 strict exact match}) for the distinct configurations, every cell once."""
    import run_decision_strict as rds
    configs = sorted(set(map(str, configs)))
    df = pd.read_parquet(path, columns=["model_key", "arm", "config_id", "item_id", "strict_correct"],
                         filters=[("config_id", "in", configs)])
    df["item_id"] = df["item_id"].astype(str)
    df = df[df["item_id"].isin(set(map(str, items)))]
    S = rds.load_script("run_noise_anatomy").dense_items(df, list(panel), configs, list(map(str, items)), arms=ARMS)
    return configs, {a: S[i] for i, a in enumerate(ARMS)}


# ============================================================================ items 1, 2, 3b: HotpotQA discovery rows


def strict_record(items, disc, panel, eci, Sd, cfg_index, contrasts, n_pilots=None, n_items=None):
    """Item 1: every pilot's draw, items, predictions and choices, per design and contrast (decision_strict.pilots with
    HotpotQA's contrasts in place of the two baselines)."""
    from src import decision as dc, decision_strict as ds, rule_budgets as rbud, sampling
    n_pilots, n_items = n_pilots or N_PILOTS, n_items or N_ITEMS
    items = [str(i) for i in items]
    N, M = len(items), len(panel)
    n = dc.third_size(M)
    mask = ds.split_items(items, TASK, SPLIT_LABEL)
    rng = sampling.rng_for("rule-budgets", TASK, 1, n_items)
    pos, its = rbud.draw_pilots(n_pilots, len(disc), 1, rng, N=N, n=n_items)
    designs = {"shared": (pos[:, 0], its, np.ones(N, bool))}
    rng = sampling.rng_for(SPLIT_LABEL, TASK, 1, n_items)
    pos2, its2 = ds.draw_pilots(n_pilots, len(disc), rng, np.flatnonzero(mask), n_items)
    if not mask[its2].all():
        raise SystemExit("a disjoint pilot drew an item outside the pilot pool")
    designs["disjoint"] = (pos2, its2, ~mask)
    arrays, rec = {"item_split_pilot_pool": mask}, {}
    e = np.asarray(eci, float)
    for d, (p, it, emask) in designs.items():
        U = ds.pilot_weights(N, it)
        rows = np.array([cfg_index[str(disc[x])] for x in p])
        sc = {a: ds.pilot_scores(Sd[a], rows, U) for a in ARMS}
        arrays[f"{TASK}__{d}__positions"] = p.astype(np.int16)
        arrays[f"{TASK}__{d}__items"] = it.astype(np.int16)
        rec[d] = {"evaluation_items": int(emask.sum()), "pilot_pool_items": int((~emask).sum()) if d == "disjoint" else N,
                  "ties": {}}
        for c, (b, q) in contrasts.items():
            for r, Pr in ds.pilot_predictions(sc[b], sc[q], e).items():
                W, T = ds.choice_weights(Pr, n)
                arrays[f"{TASK}__{d}__{c}__{r}__pred"] = Pr
                arrays[f"{TASK}__{d}__{c}__{r}__weights"] = W
                rec[d]["ties"][f"{c}/{r}"] = T
    record = {"label": "hotpot_decision_strict", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": list(panel),
              "n_chosen": n, "units": "pp; predictions uncentred; weights per model (ties share places)",
              "design": {"n_pilots": n_pilots, "n_items": n_items, "split_label": SPLIT_LABEL, "n_target_items": N,
                         "streams": {"shared": f"sampling.rng_for('rule-budgets', '{TASK}', 1, {n_items})",
                                     "disjoint": f"sampling.rng_for('{SPLIT_LABEL}', '{TASK}', 1, {n_items})"},
                         "contrasts": {c: list(v) for c, v in contrasts.items()}, "discovery_units": list(map(str, disc)),
                         "item_split_sha256": sampling.sha256_json([items[i] for i in np.flatnonzero(mask)]),
                         "designs": rec},
              "note": NOTE_DISCOVERY}
    return record, arrays


def family_record(disc, panel, fams, eci, D: dict):
    """Item 2: transfer_regret's held-out-family choices per contrast (D: {contrast: (Bd, Pd)} on the discovery draws)."""
    import run_transfer_regret as rtr
    arrays, meta = {}, {}
    for c, (B, P) in D.items():
        x = {"facts": {"families": list(fams), "eci": list(map(float, eci)), "discovery_config_ids": list(map(str, disc))},
             "discovery": {"base": np.asarray(B, float), "cot": np.asarray(P, float)}}
        W, groups, ns = rtr.family_choices(x, "base")
        meta[c] = {"groups": {g: [panel[j] for j in ix] for g, ix in groups.items()}, "n_chosen": ns}
        for g, w in W.items():
            for r, a in w.items():
                arrays[f"family__{c}__{g}__{r}"] = a
    record = {"label": "hotpot_family", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": list(panel),
              "units": "weights per model", "design": {"rules": list(rtr.FAMILY_RULES), "contrasts": meta,
                                                       "discovery_units": list(map(str, disc))},
              "note": NOTE_DISCOVERY}
    return record, arrays


def cross_disc_record(disc, panel, D: dict, cross_rec: dict, cross_arr: dict):
    """Item 3b: the saved source rules at the HotpotQA one-draw pilot baselines, and HotpotQA's own one-draw gain on the
    same models; reads the source means from the item 3a manifest only."""
    from src import decision as dc, transfer_regret as tr
    rows = dc.distinct_rows(disc)
    arrays, meta = {}, {}
    for key, m in cross_rec["design"]["pairs"].items():
        s, c = key.split("/")
        if c not in D:
            continue
        jt = [list(panel).index(k) for k in m["models"]]
        n = m["n_chosen"]
        B, P = (np.asarray(x, float)[:, jt] for x in D[c])
        G = P - B
        k = f"cross__{s}__{c}"
        bm, gm, e = cross_arr[f"{k}__source_baseline_mean"], cross_arr[f"{k}__source_gain_mean"], cross_arr[f"{k}__eci"]
        preds = {"source_linear": np.array([tr.fit_apply(bm, gm, e, B[r], e, "baseline_linear") for r in rows]),
                 "source_quadratic_index": np.array([tr.fit_apply(bm, gm, e, B[r], e, "baseline_quadratic_index")
                                                     for r in rows]),
                 "target_own_1": G[rows]}
        for r, x in preds.items():
            arrays[f"{k}__{r}__pred"] = x
            arrays[f"{k}__{r}__weights"] = tr.third_weights(x, n)
        meta[key] = {"models": m["models"], "n_chosen": n, "target_pilot_rows": rows}
    record = {"label": "hotpot_cross_discovery", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": list(panel),
              "units": "pp; predictions uncentred; weights per model", "design": {
                  "rules": list(CROSS_DISC_RULES), "pairs": meta, "discovery_units": list(map(str, disc)),
                  "cross_benchmark_manifest_sha256": cross_rec.get("sha256")},
              "note": NOTE_DISCOVERY}
    return record, arrays


def save_discovery(path, disc, panel, fams, eci, items, contrasts, pred_dir, overwrite=False, D=None) -> dict:
    """Items 1, 2 and 3b from the HotpotQA discovery rows of the scored table at path; refuses before item 3a exists.
    D: {contrast: (Bd, Pd)} cell means on the discovery draws (strict exact match), as run_hotpot.arrays builds them."""
    from src import rule_budgets as rbud
    cross_rec, cross_arr = load_cross(pred_dir, panel)
    configs, Sd = dense(path, panel, disc, items)
    cfg_index = {c: i for i, c in enumerate(configs)}
    if D is None:   # cell means from the item-level arrays (equal to run_hotpot.arrays on complete cells)
        rows = [cfg_index[str(u)] for u in disc]
        D = {c: (Sd[b][rows].mean(2) * 100.0, Sd[q][rows].mean(2) * 100.0) for c, (b, q) in contrasts.items()}
    out = {}
    for stem, (rec, arr) in ((STRICT, strict_record(items, disc, panel, eci, Sd, cfg_index, contrasts)),
                             (FAMILY, family_record(disc, panel, fams, eci, D)),
                             (CROSS_DISC, cross_disc_record(disc, panel, D, cross_rec, cross_arr))):
        jp, npz = files(stem, pred_dir)
        out[stem] = rbud.freeze(rec, arr, jp, npz, overwrite=overwrite)
    return out


# ============================================================================ evaluation (reads the saved choices only)


def load_saved(pred_dir: str, panel) -> dict:
    """{stem: (record, arrays)} of every saved file, hashes checked; refuses on any mismatch."""
    from src import rule_budgets as rbud
    out = {}
    for stem in (CROSS,) + DISCOVERY_FILES:
        jp, npz = files(stem, pred_dir)
        try:
            rec, arr = rbud.load_frozen(jp, npz)
        except (OSError, ValueError) as err:
            raise SystemExit(f"{jp}: {err}")
        if list(rec["models"]) != list(panel):
            raise SystemExit(f"{jp} was saved for another panel")
        out[stem] = (rec, arr)
    return out


def eval_sets(disc, val) -> dict:
    import run_decision as rd
    ev = rd.evaluation_rows(list(map(str, disc)), list(map(str, val)))
    return {"fresh": ev["independent"], "all_distinct": ev["all_distinct"]}


def evaluate_strict(rec, arr, val, Sv, configs, items_n, contrasts) -> dict:
    """Item 1 on the validation configurations: Sv {arm: (C, M, N)} over `configs` (distinct validation configurations)."""
    from src import decision_strict as ds
    M, n = len(rec["models"]), rec["n_chosen"]
    disc = rec["design"]["discovery_units"]
    mask = np.asarray(arr["item_split_pilot_pool"], bool)
    if len(mask) != items_n:
        raise SystemExit("the item set differs from the one the strict pilots were drawn on")
    ci = {c: i for i, c in enumerate(configs)}
    sets = eval_sets(disc, val)
    out = {}
    for d in ("shared", "disjoint"):
        w = ds.eval_weights(items_n, np.ones(items_n, bool) if d == "shared" else ~mask)
        out[d] = {}
        for c, (b, q) in contrasts.items():
            Wd = {r: arr[f"{TASK}__{d}__{c}__{r}__weights"] for r in ds.RULES}
            Wd["random"] = np.full((1, M), n / M)
            out[d][c] = {}
            for ev, rows in sets.items():
                cs = [str(val[r]) for r in rows]
                if len(cs) < 2:
                    out[d][c][ev] = {"n_configurations": len(cs), "configurations": cs,
                                     "note": "fewer than two evaluation configurations"}
                    continue
                ix = [ci[x] for x in cs]
                G = (Sv[q][ix].astype(float) - Sv[b][ix].astype(float)) @ w
                out[d][c][ev] = {**ds.evaluate(Wd, G, n), "configurations": cs}
    return out


def evaluate_family(rec, arr, disc, val, V: dict) -> dict:
    """Item 2: V {contrast: (Bv, Pv)} on the validation draws (rows as the sampler's validation units)."""
    import run_transfer_regret as rtr
    from src import transfer_regret as tr
    out = {}
    for c, (Bv, Pv) in V.items():
        meta = rec["design"]["contrasts"][c]
        panel = rec["models"]
        groups = {g: [panel.index(k) for k in ks] for g, ks in meta["groups"].items()}
        ns = meta["n_chosen"]
        W = {g: {r: arr[f"family__{c}__{g}__{r}"] for r in rtr.FAMILY_RULES} for g in groups}
        out[c] = {}
        for ev, rows in eval_sets(disc, val).items():
            G = (np.asarray(Pv, float) - np.asarray(Bv, float))[rows]
            res = {"n_configurations": len(rows), "rules": {}, "paired_minus_target_own_1": {}}
            if len(rows) < 2:
                res["note"] = "fewer than two evaluation configurations"
                out[c][ev] = res
                continue
            for r in rtr.FAMILY_RULES:
                Wg = {g: W[g][r] for g in groups}
                res["rules"][r] = tr.grouped_regret(Wg, G, groups, ns)
                if r != "target_own_1":
                    res["paired_minus_target_own_1"][r] = tr.paired_grouped(
                        Wg, {g: W[g]["target_own_1"] for g in groups}, G, groups)
            out[c][ev] = res
    return out


def evaluate_cross(rec_a, arr_a, rec_b, arr_b, disc, val, V: dict) -> dict:
    """Item 3: every saved source choice and HotpotQA's own one-draw gain on the shared models."""
    from src import decision as dc
    out = {}
    panel = rec_a["models"]
    for key, m in rec_a["design"]["pairs"].items():
        s, c = key.split("/")
        if c not in V:
            continue
        jt = [panel.index(k) for k in m["models"]]
        n = m["n_chosen"]
        k = f"cross__{s}__{c}"
        W = {r: arr_a[f"{k}__{r}__weights"] for r in CROSS_RULES}
        W.update({r: arr_b[f"{k}__{r}__weights"] for r in CROSS_DISC_RULES})
        Bv, Pv = V[c]
        res_c = {"models": m["models"], "M": len(jt), "n_chosen": n, "source_linear_slope": m["source_linear_slope"]}
        for ev, rows in eval_sets(disc, val).items():
            G = (np.asarray(Pv, float) - np.asarray(Bv, float))[rows][:, jt]
            res = {"n_configurations": len(rows), "rules": {}, "paired_minus_target_own_1": {}}
            if len(rows) < 2:
                res["note"] = "fewer than two evaluation configurations"
                res_c[ev] = res
                continue
            for r in CROSS_EVAL_ORDER:
                res["rules"][r] = dc.summarize(W[r], G, n)
                if r != "target_own_1":
                    res["paired_minus_target_own_1"][r] = dc.paired_difference(W[r], W["target_own_1"], G, n)
            res_c[ev] = res
        out[key] = res_c
    return out


def evaluate_all(saved: dict, disc, val, V: dict, Sv, configs, items_n, contrasts) -> dict:
    rs, ra = saved[STRICT]
    rf, af = saved[FAMILY]
    rc, ac = saved[CROSS]
    rd_, ad = saved[CROSS_DISC]
    return {"saved": {stem: {"sha256": saved[stem][0].get("sha256"), "npz_sha256": saved[stem][0].get("npz_sha256"),
                             "created": saved[stem][0].get("created")} for stem in saved},
            "strict": evaluate_strict(rs, ra, val, Sv, configs, items_n, contrasts),
            "family": evaluate_family(rf, af, disc, val, V),
            "cross_benchmark": evaluate_cross(rc, ac, rd_, ad, disc, val, V)}


# ============================================================================ report lines


def _f2(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _ci(d):
    c = d.get("ci95")
    return f"{_f2(d['estimate'])} [{_f2(c[0])}, {_f2(c[1])}]" if c else _f2(d["estimate"])


def summary_lines(F: dict, titles: dict) -> list[str]:
    L = ["", "## Fair comparison (saved before validation was scored)", "",
         "Plug-in regret in pp per chosen model on the validation configurations in no discovery draw (each once); "
         "cross-fitted regret in parentheses. Paired differences in brackets are 95% t intervals.", "",
         "Strict decision design, disjoint items (100-item one-draw pilots, 200 per design):", "",
         "| Contrast | Configs | Linear | Quadratic + index | Own gain | Random | Own - quadratic + index | Own - linear |",
         "|---|---:|---|---|---|---|---|---|"]
    for c, e in F["strict"]["disjoint"].items():
        e = e["fresh"]
        if "rules" not in e:
            L.append(f"| {titles[c]} | {e['n_configurations']} | -- | -- | -- | -- | -- | -- |")
            continue
        r = e["rules"]
        L.append(f"| {titles[c]} | {e['n_configurations']} | " + " | ".join(
            _ci(r[k]["plugin_interval"]) for k in ("linear", "quadratic_index", "own_gain", "random"))
            + f" | {_ci(e['paired']['own_gain_minus_quadratic_index'])} | {_ci(e['paired']['own_gain_minus_linear'])} |")
    L += ["", "Held-out family (pooled over families; paired: rule minus the family's own one-draw gain):", ""]
    for c, e in F["family"].items():
        e = e["fresh"]
        if "rules" not in e:
            continue
        L.append(f"- {titles[c]}: " + ", ".join(
            f"{r} {_f2(v['regret_plugin'])} ({_f2(v['regret_crossfit'])})" for r, v in e["rules"].items())
            + "; paired " + ", ".join(f"{r} {_ci(v)}" for r, v in e["paired_minus_target_own_1"].items()) + ".")
    L += ["", "Another benchmark (models both panels share; paired: rule minus HotpotQA's own one-draw gain):", ""]
    for key, e in F["cross_benchmark"].items():
        s, c = key.split("/")
        f = e["fresh"]
        if "rules" not in f:
            continue
        L.append(f"- {SOURCE_TITLE[s]} -> HotpotQA, {titles[c]} ({e['M']} models, {e['n_chosen']} chosen): " + ", ".join(
            f"{r} {_f2(v['regret_plugin'])}" for r, v in f["rules"].items())
            + "; paired " + ", ".join(f"{r} {_ci(v)}" for r, v in f["paired_minus_target_own_1"].items()) + ".")
    return L
