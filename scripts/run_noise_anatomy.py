r"""Analysis N, noise anatomy (src/noise_anatomy.py for the definitions).

For each setting: (1) the gain noise W_gain split into Var(e_P) + Var(e_B) - 2 Cov(e_P, e_B) with the moment
definition of the existing analysis, checked against its W_gain and W_baseline; (2) with the configuration draws
held fixed, item subsamples at each n of config/noise_anatomy.yaml, the fit W(n) = W_shift + W_item / n, its exact
design expectation, and the planning rule R(n) = V_delta / (V_delta + W(n)) -> prompts K for 80% and 90% agreement.

Settings (--settings):
    dove_mmlu  pooled DOVE MMLU block, 5 models, five-shot against zero-shot. Item scores from
               data/processed/dove/mmlu_items.parquet (scripts/extract_dove_items.py). Variants: "bank", the exact
               moments over the 2,716 common configurations (the definition of results/dataset1/dove_mmlu.json,
               primary), and "discovery", the Appendix A moments of the 24 discovery draws.
    gsm8k      GSM8K, the 19-model panel of results/dataset2/gsm8k.json, chain of thought against direct, strict
               extraction. Variants: "discovery", the 24 discovery units (the definition of results/dataset2/gsm8k.json,
               primary), and "all_units", the 24 discovery and 24 validation units together.
    gsm8k30    GSM8K, the 30 checkpoints of the extension (src/d2_panels.inputs); same variants as gsm8k.
    bbh        BIG-Bench Hard, the 25 tasks pooled (src/d2_panels.bbh_inputs); same variants as gsm8k.
    math       MATH-500, the 398 numeric target items and the 29 models of results/dataset2/math.json
               (src/d2_panels.math_inputs); same variants as gsm8k.
A run for some settings keeps the other settings already in the JSON, then rewrites the summary.

Usage:
    python scripts/run_noise_anatomy.py                    # dove_mmlu and gsm8k
    python scripts/run_noise_anatomy.py --settings bbh     # once BBH is scored
    python scripts/run_noise_anatomy.py --settings gsm8k30
    python scripts/run_noise_anatomy.py --settings math

Outputs
    results/noise_anatomy/noise_anatomy.json      everything, with a top-level "macros" object for the paper
    results/noise_anatomy/summary.md, summary_clean.md
    results/noise_anatomy/tables/subsamples.csv   every subsample's noise parts, V_delta and R
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (SCRIPT_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import controls, dataset2 as d2, noise_anatomy as na, sampling  # noqa: E402
from src import dove_io as io  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "noise_anatomy.yaml")
OUT_DIR = os.path.join(ROOT, "results", "noise_anatomy")
MAN = os.path.join(ROOT, "manifests")
DOVE_ITEMS = os.path.join(io.PROCESSED, "mmlu_items.parquet")
DOVE_RESULT = os.path.join(ROOT, "results", "dataset1", "dove_mmlu.json")
DOVE_TASKS = os.path.join(ROOT, "results", "dataset1", "dove_tasks.json")
D2_RESULTS = os.path.join(ROOT, "results", "dataset2")
SETTINGS = ("dove_mmlu", "gsm8k", "gsm8k30", "bbh", "math")
PRIMARY = {"dove_mmlu": "bank", "gsm8k": "discovery", "gsm8k30": "discovery", "bbh": "discovery", "math": "discovery"}
TITLE = {"dove_mmlu": "DOVE, pooled MMLU block (five-shot against zero-shot)",
         "gsm8k": "GSM8K, 19 models (chain of thought against direct)",
         "gsm8k30": "GSM8K, 30 models (chain of thought against direct)",
         "bbh": "BIG-Bench Hard, 25 tasks pooled (chain of thought against direct)",
         "math": "MATH-500, 398 numeric items (chain of thought against direct)"}
SHORT = {"dove_mmlu": "MMLU block", "gsm8k": "GSM8K", "gsm8k30": "GSM8K-30", "bbh": "BIG-Bench Hard",
         "math": "MATH-500"}
MACRO_SUFFIX = {"dove_mmlu": "Dove", "gsm8k": "Gsm", "gsm8k30": "GsmThirty", "bbh": "Bbh", "math": "Math"}
TOL = 1e-8


def added_settings(path=None) -> dict:
    """The contrasts of config/added_contrasts.yaml as noise-anatomy settings, by name. Each reuses
    the item grid and the item subsamples of its `noise_base` setting, so the two compare on the same items. None when
    the file is absent."""
    sys.path.insert(0, SCRIPT_DIR)
    import added_contrasts as ac
    p = path or ac.CONFIG
    return ac.load_config(p) if os.path.exists(p) else {}


ADDED = added_settings()
for _name, _c in ADDED.items():
    TITLE[_name] = _c["title"]
    SHORT[_name] = _c["short"]
    MACRO_SUFFIX[_name] = _c["tag"]
    PRIMARY[_name] = "discovery"
ALL_SETTINGS = SETTINGS + tuple(ADDED)


class Refusal(RuntimeError):
    """An input does not match the analysis it is meant to decompose."""


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else o.item() if isinstance(o, np.generic) else str(o))


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# --------------------------------------------------------------------------- inputs: DOVE MMLU block


def load_dove():
    """Dense item scores (configurations, models, items) per arm, with the item weights of the pooled score."""
    spec = json.load(open(os.path.join(MAN, "dove_mmlu_block.json")))
    sampler = json.load(open(os.path.join(MAN, "sampler_dove_mmlu.json")))
    common = [int(c) for c in spec["common_config_ids"]]
    if [int(c) for c in sampler["bank_config_ids"]] != common:
        raise Refusal("the DOVE sampler's bank is not the block's common configuration list")
    if not os.path.exists(DOVE_ITEMS):
        raise Refusal(f"{rel(DOVE_ITEMS)} is missing; run scripts/extract_dove_items.py")
    rec_path = os.path.join(OUT_DIR, "dove_item_extraction.json")
    rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    if rec.get("status") != "ok" or rec.get("common_config_ids_sha256") != spec["common_config_ids_sha256"] \
            or rec.get("item_intersection_sha256") != spec["item_intersection_sha256"]:
        raise Refusal(f"{rel(rec_path)} does not record a successful extraction of this block")
    panel, arms, subjects = spec["panel"], spec["arms"], spec["subjects"]
    items = {s: np.asarray(spec["item_intersection"][s], dtype=np.int64) for s in subjects}
    sizes = np.array([len(items[s]) for s in subjects])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    C, M, N = len(common), len(panel), int(sizes.sum())
    S = {a: np.full((C, M, N), -1, dtype=np.int8) for a in arms}
    f = pq.ParquetFile(DOVE_ITEMS)
    seen = set()
    common_arr = np.asarray(common, dtype=np.int64)
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g)
        sub, mod, arm = (t.column(c).unique().to_pylist() for c in ("subject", "model", "arm"))
        if len(sub) != 1 or len(mod) != 1 or len(arm) != 1:
            raise Refusal(f"row group {g} of {rel(DOVE_ITEMS)} mixes cells")
        s, m, a = sub[0], mod[0], arm[0]
        k = subjects.index(s)
        n = sizes[k]
        cfg = t.column("config_id").to_numpy()
        it = t.column("item").to_numpy()
        if not (np.array_equal(cfg, np.repeat(common_arr, n)) and np.array_equal(it, np.tile(items[s], C))):
            raise Refusal(f"row group {g} ({s}, {m}, {a}) is not in the manifest's configuration and item order")
        S[a][:, panel.index(m), offsets[k]:offsets[k] + n] = t.column("score").to_numpy().reshape(C, n)
        seen.add((s, m, a))
    if len(seen) != len(subjects) * M * len(arms) or any((x < 0).any() for x in S.values()):
        raise Refusal("the item table does not cover every (subject, model, arm) cell of the block")
    v = np.concatenate([np.full(n, 1.0 / (len(subjects) * n)) for n in sizes])
    # Check: the full-set scores equal the pooled scores of scripts/run_dataset1.py (from mmlu_block.parquet).
    block = pd.read_parquet(os.path.join(io.PROCESSED, "mmlu_block.parquet"))
    idx = {c: j for j, c in enumerate(common)}
    for a in arms:
        nc = np.full((len(subjects), C, M), -1, dtype=np.int64)
        x = block[block["arm"] == a]
        nc[x["subject"].map({s: j for j, s in enumerate(subjects)}).to_numpy(), x["config_id"].map(idx).to_numpy(),
           x["model"].map({m: j for j, m in enumerate(panel)}).to_numpy()] = x["n_correct"].to_numpy()
        ref = controls.pooled(nc, sizes)
        got = na.full_scores(S[a], v)
        if not np.allclose(got, ref, rtol=0, atol=1e-9):
            raise Refusal(f"pooled {a} scores from the item table differ from mmlu_block.parquet "
                          f"(max {np.abs(got - ref).max():.3g} pp)")
    disc = np.asarray(sampler["discovery_bank_index"], dtype=np.int64)
    res = json.load(open(DOVE_RESULT))["step2"]["exact"]
    reference = {"source": rel(DOVE_RESULT) + " (step2.exact)",
                 "W_gain": res["noise_W_gain"], "W_baseline": res["noise_W_baseline"],
                 "cov_e": res["coupling"]["E_cov_eP_eB"], "V_delta": res["V_delta"]}
    return {"setting": "dove_mmlu", "B": S["0_shot"], "P": S["5_shot"], "weights": v, "panel": panel,
            "arms": {"baseline": "0_shot", "intervention": "5_shot"},
            "variants": {"bank": {"draws": np.arange(C), "kind": "bank",
                                  "label": f"exact over the {C:,} common configurations, uniform sampler"},
                         "discovery": {"draws": disc, "kind": "draws",
                                       "label": f"Appendix A moments of the {len(disc)} discovery draws"}},
            "reference": {"bank": reference},
            "subjects": {"names": subjects, "offsets": offsets.tolist(), "sizes": sizes.tolist()},
            "provenance": {"items": rel(DOVE_ITEMS), "items_sha256": sha256_file(DOVE_ITEMS),
                           "extraction_record": rel(rec_path), "block_manifest": "manifests/dove_mmlu_block.json",
                           "sampler": "manifests/sampler_dove_mmlu.json", "dove_revision": io.REVISION,
                           "item_weights": "the pooled score's weights, 1 / (56 x subject size)"},
            "checks": {"full_set_scores_equal_mmlu_block": True}}


# --------------------------------------------------------------------------- inputs: Dataset 2


def dense_items(df, panel, configs, items, arms=d2.ARMS, value="strict_correct"):
    """(arms, configs, models, items) int8 from scored rows; every cell exactly once or the input is refused."""
    configs = [str(c) for c in configs]
    m = pd.Index(list(panel)).get_indexer(df["model_key"])
    c = pd.Index(configs).get_indexer(df["config_id"].astype(str))
    a = pd.Index(list(arms)).get_indexer(df["arm"])
    keep = (m >= 0) & (c >= 0) & (a >= 0)
    i = pd.Index(list(items)).get_indexer(df["item_id"][keep])
    if (i < 0).any():
        raise Refusal(f"{int((i < 0).sum())} rows have item ids outside the fixed item set")
    A, C, M, N = len(arms), len(configs), len(panel), len(items)
    flat = ((a[keep].astype(np.int64) * C + c[keep]) * M + m[keep]) * N + i
    cnt = np.bincount(flat, minlength=A * C * M * N)
    if (cnt != 1).any():
        raise Refusal(f"{int((cnt == 0).sum())} (arm, configuration, model, item) cells missing and "
                      f"{int((cnt > 1).sum())} duplicated")
    S = np.zeros(A * C * M * N, dtype=np.int8)
    S[flat] = df[value].to_numpy()[keep].astype(np.int8)
    return S.reshape(A, C, M, N)


def load_d2(setting):
    from src import d2_io, d2_panels
    label = setting
    task = setting if setting in ("bbh", "math") else "gsm8k"
    sampler_path = os.path.join(MAN, f"sampler_{task}.json")
    scorer_path = os.path.join(MAN, f"{task}_scorer.json")
    sampler = json.load(open(sampler_path))
    analysis_path = os.path.join(D2_RESULTS, f"{label}.json")
    analysis = json.load(open(analysis_path)) if os.path.exists(analysis_path) else None
    try:
        if setting == "gsm8k":
            scored = os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet")
            items_df, items_info = d2_io.load_items(split="test")
            md = d2_io.scored_metadata(scored)
            problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info, scorer_path=scorer_path)
            if problems:
                raise Refusal(f"{rel(scored)}: " + "; ".join(problems))
            if analysis is None:
                raise Refusal(f"{rel(analysis_path)} is missing; the 19-model panel is read from it")
            panel = list(analysis["models"])
            items = items_df["item_id"].tolist()

            def load(ids):
                return pd.read_parquet(scored, columns=["model_key", "arm", "config_id", "item_id", "strict_correct"],
                                       filters=[("config_id", "in", sorted(set(map(str, ids))))])
            provenance = {"tables": [{"path": rel(scored), "sha256": sha256_file(scored)}]}
        elif setting == "gsm8k30":
            items_df, items_info = d2_io.load_items(split="test")
            got = d2_panels.inputs(None, None, sampler, items_info, scorer_path)
            panel, load, items = got["panel"], got["load"], items_df["item_id"].tolist()
            provenance = got["provenance"]
        else:
            loader = d2_panels.math_inputs if setting == "math" else d2_panels.bbh_inputs
            got = loader(None, sampler, scorer_path=scorer_path)
            panel, items, provenance = got["panel"], got["items"], got["provenance"]

            def load(ids, _load=got["load"]):
                return _load(ids)
    except (OSError, ValueError, d2_io.D2IOError, d2_panels.PanelError) as exc:
        raise Refusal(f"{setting}: {exc}") from exc
    return dense_d2(setting, sampler, panel, items, load, provenance, sampler_path, scorer_path, analysis, analysis_path)


def dense_d2(setting, sampler, panel, items, load, provenance, sampler_path, scorer_path, analysis, analysis_path,
             arms=None):
    """Dense item scores of a Dataset 2 setting over its discovery and validation units, checked against the cell
    table of the main analysis, with the reference moments of its existing analysis when there is one."""
    if analysis is not None:
        if analysis.get("sampler_sha256") != sampler["sha256"]:
            raise Refusal(f"{rel(analysis_path)} was computed with another sampler")
        if list(analysis["models"]) != list(panel):
            raise Refusal(f"the panel differs from {rel(analysis_path)}")
    disc, val = list(sampler["discovery_config_ids"]), list(sampler["validation_config_ids"])
    configs = sorted(set(map(str, disc + val)))
    df = load(configs)
    S = dense_items(df, panel, configs, items)
    pos = {c: j for j, c in enumerate(configs)}
    # Check: the full-set scores equal the analysis's cell table (src/dataset2.cell_table, unit_arrays).
    cells = d2.cell_table(df, panel, configs, items, values=("strict_correct",))
    for units in (disc, disc + val):
        Bu, Pu = d2.unit_arrays(cells, units, panel, "strict_correct")
        j = [pos[str(u)] for u in units]
        for X, ref in ((S[0][j], Bu), (S[1][j], Pu)):
            if not np.allclose(na.full_scores(X), ref, rtol=0, atol=1e-9):
                raise Refusal(f"{setting}: full-set scores differ from src/dataset2.cell_table")
    reference = None
    if analysis is not None:
        reference = {"source": rel(analysis_path) + " (step2)", "W_gain": analysis["step2"]["noise"]["W_gain"],
                     "W_baseline": analysis["step2"]["noise"]["W_baseline"],
                     "V_delta": analysis["step2"]["V_delta"]["estimate"]}
    return {"setting": setting, "B": S[0], "P": S[1], "weights": None, "panel": panel,
            "arms": arms or {"baseline": d2.ARMS[0], "intervention": d2.ARMS[1]},
            "variants": {"discovery": {"draws": np.array([pos[str(u)] for u in disc]), "kind": "draws",
                                       "label": f"Appendix A moments of the {len(disc)} discovery units"},
                         "all_units": {"draws": np.array([pos[str(u)] for u in disc + val]), "kind": "draws",
                                       "label": f"Appendix A moments of all {len(disc) + len(val)} units "
                                                "(discovery and validation)"}},
            "reference": {"discovery": reference} if reference else {},
            "provenance": {**provenance, "sampler": rel(sampler_path), "scorer": rel(scorer_path),
                           "value": "strict_correct", "item_weights": "uniform"},
            "checks": {"full_set_scores_equal_cell_table": True}}


def arm_name(spec) -> str:
    """An added contrast's arm in words: the collection's arm or the arm of an added task's table, or the folder of its
    scored table."""
    return spec["arm"] if spec["kind"] in ("collection", "table") else \
        f"{os.path.basename(spec['dir'])} ({spec['dir']})"


def load_added(name, contrasts=None):
    """An added contrast (config/added_contrasts.yaml): its two arms' rows through scripts/added_contrasts.py, the
    same dense table and checks as the Dataset 2 settings, and its own Steps 1-3 file as the reference."""
    import added_contrasts as ac
    con = contrasts if contrasts is not None else ADDED
    try:
        x = ac.inputs(name, con)
    except (OSError, ValueError, SystemExit) as exc:
        raise Refusal(f"{name}: {exc}") from exc
    task = x["task"]
    tables = list((x["provenance"].get("collection") or {}).get("tables", [])) \
        if "collection" in (x["baseline"]["kind"], x["intervention"]["kind"]) else []
    for side in ("baseline", "intervention"):
        p = x["provenance"][side]
        if "path" in p:
            tables.append({"path": p["path"], "sha256": p.get("sha256")})
    provenance = {"tables": tables, "contrast": name, "config": "config/added_contrasts.yaml",
                  "arms": x["provenance"]["arms"]}
    analysis_path = os.path.join(ROOT, x["steps"])
    analysis = json.load(open(analysis_path)) if os.path.exists(analysis_path) else None
    return dense_d2(name, x["sampler"], x["panel"], x["items"], x["load"], provenance,
                    os.path.join(MAN, f"sampler_{task}.json"), os.path.join(MAN, f"{task}_scorer.json"), analysis,
                    analysis_path, arms={"baseline": arm_name(x["baseline"]),
                                         "intervention": arm_name(x["intervention"])})


# --------------------------------------------------------------------------- analysis


def subsample_sets(setting, N, ns, reps, seed):
    return {n: na.draw_subsamples(N, n, reps, sampling.rng_for("noise_anatomy", setting, n, seed=seed)) for n in ns}


def analyse_variant(SB, SP, v, kind, ns, sets, targets, planning_ns):
    K, M, N = SB.shape
    B, P = na.full_scores(SB, v), na.full_scores(SP, v)
    split = na.arm_split(B, P, kind)
    design = na.design_expectation(SB, SP, v, kind)
    per_n = {}
    for n in ns:
        U = na.subsample_matrix(N, sets[n], v)
        per_n[n] = na.split_batch(na.subsample_scores(SB, U), na.subsample_scores(SP, U), kind)
    fits = {p: na.fit_inverse_n(ns, [per_n[n][p] for n in ns], N) for p in na.PARTS}
    if not close(fits["W_gain"]["W_full"], split["W_gain"], 1e-9):
        raise Refusal("the full-set subsample does not reproduce W_gain")
    signal = []
    for n in ns:
        x, r = per_n[n]["V_delta"], per_n[n]["R_benefit"]
        signal.append({"n": int(n), "V_delta_mean": float(x.mean()), "V_delta_sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
                       "V_delta_design": na.expected_at(design["parts"]["V_delta"], n),
                       "R_mean": float(np.nanmean(r)), "R_undefined": int(np.isnan(r).sum()),
                       "R_q05": float(np.nanquantile(r, 0.05)), "R_q95": float(np.nanquantile(r, 0.95))})
    grid = sorted(set(int(n) for n in list(planning_ns) + list(ns) if int(n) <= N))   # no extrapolation past N
    fw = fits["W_gain"]
    dw = design["parts"]["W_gain"]
    out = {"kind": kind, "K": int(K), "M": int(M), "N": int(N), "split": split, "design": design, "fits": fits,
           "signal_by_n": signal,
           "planning": {"V_delta": split["V_delta"], "targets": list(targets),
                        "from_fit": na.planning_rule(split["V_delta"], fw["W_shift"], fw["W_item"], grid, targets),
                        "from_design": na.planning_rule(split["V_delta"], dw["W_shift"], dw["W_item"], grid, targets),
                        "note": "V_delta is the full-set value of the same draws, held fixed across n"}}
    return out, per_n


def dove_subjects(data, fit):
    """Each MMLU subject of the block as a benchmark of its own items (exact over the bank), beside the fit's W(n)."""
    s = data["subjects"]
    rows = []
    for name, off, n in zip(s["names"], s["offsets"], s["sizes"]):
        sl = slice(off, off + n)
        B, P = na.full_scores(data["B"][:, :, sl]), na.full_scores(data["P"][:, :, sl])
        sp = na.arm_split(B, P, "bank")
        rows.append({"subject": name, "n_items": int(n), "W_gain": sp["W_gain"], "var_e_intervention":
                     sp["var_e_intervention"], "var_e_baseline": sp["var_e_baseline"], "cov_e": sp["cov_e"],
                     "V_delta": sp["V_delta"], "R_benefit": sp["R_benefit"],
                     "W_fit_at_n": fit["W_shift"] + fit["W_item"] / n})
    W = np.array([r["W_gain"] for r in rows])
    R = np.array([r["R_benefit"] for r in rows])
    out = {"per_subject": rows, "n_subjects": len(rows),
           "W_gain_median": float(np.median(W)), "W_gain_q25": float(np.quantile(W, 0.25)),
           "W_gain_q75": float(np.quantile(W, 0.75)), "W_gain_mean": float(W.mean()),
           "R_median": float(np.nanmedian(R)), "R_mean": float(np.nanmean(R)),
           "V_delta_median": float(np.median([r["V_delta"] for r in rows])),
           "W_fit_at_100": fit["W_shift"] + fit["W_item"] / 100,
           "frac_subject_W_above_fit": float(np.mean(W > np.array([r["W_fit_at_n"] for r in rows])))}
    # The 71 DOVE benchmarks of the secondary analysis (results/dataset1/dove_tasks.json): W from their V_delta and R.
    if os.path.exists(DOVE_TASKS):
        tasks = json.load(open(DOVE_TASKS))["tasks"]
        tr = []
        for t in tasks:
            V, Rt = t["V_delta"], t["R_benefit"]
            Wt = V * (1 - Rt) / Rt if Rt not in (0, None) and math.isfinite(Rt) else float("nan")
            Wn = fit["W_shift"] + fit["W_item"] / t["n_items"]
            tr.append({"task": t["task"], "family": t["family"], "n_items": t["n_items"], "V_delta": V, "R_benefit": Rt,
                       "W_gain": Wt, "W_fit_at_n": Wn,
                       "R_predicted_from_fit": V / (V + Wn) if V + Wn > 0 else float("nan")})
        Wt = np.array([r["W_gain"] for r in tr])
        Rt = np.array([r["R_benefit"] for r in tr])
        Rp = np.array([r["R_predicted_from_fit"] for r in tr])
        ok = np.isfinite(Rt) & np.isfinite(Rp)
        out["dove_tasks"] = {"source": rel(DOVE_TASKS), "n_tasks": len(tr),
                             "n_items": sorted(set(r["n_items"] for r in tr)),
                             "W_gain_definition": "V_delta (1 - R) / R from each benchmark's exact V_delta and R",
                             "W_gain_median": float(np.nanmedian(Wt)), "W_gain_q25": float(np.nanquantile(Wt, 0.25)),
                             "W_gain_q75": float(np.nanquantile(Wt, 0.75)),
                             "R_mean": float(np.nanmean(Rt)), "R_median": float(np.nanmedian(Rt)),
                             "V_delta_median": float(np.median([r["V_delta"] for r in tr])),
                             "V_delta_mean": float(np.mean([r["V_delta"] for r in tr])),
                             "R_predicted_mean": float(np.nanmean(Rp)),
                             "R_predicted_mean_abs_error": float(np.mean(np.abs(Rp[ok] - Rt[ok]))),
                             "R_predicted_corr": float(np.corrcoef(Rp[ok], Rt[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
                             "per_task": tr}
    return out


def run_setting(data, cfg):
    setting = data["setting"]
    base = ADDED[setting]["noise_base"] if setting in ADDED else setting      # an added contrast: its base's items
    ns = [int(n) for n in cfg["n_grid"][base]]
    N = data["B"].shape[2]
    if max(ns) != N:
        raise Refusal(f"{setting}: the largest n of the grid ({max(ns)}) is not the item count ({N})")
    sets = subsample_sets(base, N, ns, int(cfg["subsamples_per_n"]), int(cfg["seed"]))
    out = {"title": TITLE[setting], "panel": data["panel"], "arms": data["arms"], "N": int(N), "n_grid": ns,
           "primary_variant": PRIMARY[setting], "provenance": data["provenance"], "checks": dict(data["checks"]),
           "variants": {}}
    if base != setting:
        out["subsamples_of"] = base
    rows = []
    for name, var in data["variants"].items():
        t0 = time.time()
        d = var["draws"]
        SB = data["B"] if len(d) == data["B"].shape[0] and np.array_equal(d, np.arange(len(d))) else data["B"][d]
        SP = data["P"] if SB is data["B"] else data["P"][d]
        res, per_n = analyse_variant(SB, SP, data["weights"], var["kind"], ns, sets, cfg["agreement_targets"],
                                     cfg["planning_n"])
        res["label"] = var["label"]
        ref = data["reference"].get(name)
        if ref is not None:
            sp = res["split"]
            chk = {"reference": ref["source"],
                   "W_gain_matches": close(sp["W_gain"], ref["W_gain"]),
                   "W_baseline_matches": close(sp["var_e_baseline"], ref["W_baseline"]),
                   "V_delta_matches": close(sp["V_delta"], ref["V_delta"]),
                   "parts_add_up": abs(sp["residual"]) <= 1e-9 * max(1.0, abs(sp["W_gain"])),
                   "reference_W_gain": ref["W_gain"], "reference_W_baseline": ref["W_baseline"],
                   "reference_V_delta": ref["V_delta"]}
            if "cov_e" in ref:
                chk["cov_e_matches"] = close(sp["cov_e"], ref["cov_e"])
                chk["reference_cov_e"] = ref["cov_e"]
            if not all(v for k, v in chk.items() if k.endswith("_matches") or k == "parts_add_up"):
                raise Refusal(f"{setting}/{name}: the split does not reproduce {ref['source']}: {chk}")
            res["check_against_analysis"] = chk
        else:
            res["check_against_analysis"] = None
        res["seconds"] = round(time.time() - t0, 1)
        out["variants"][name] = res
        for n in ns:
            for r in range(len(per_n[n]["W_gain"])):
                rows.append({"setting": setting, "variant": name, "n": n, "subsample": r,
                             **{k: float(per_n[n][k][r]) for k in (*na.PARTS, "V_delta", "R_benefit")}})
        print(f"  {setting}/{name}: W_gain {res['split']['W_gain']:.3f} = {res['split']['var_e_intervention']:.3f} + "
              f"{res['split']['var_e_baseline']:.3f} - 2 x {res['split']['cov_e']:.3f}; W_shift "
              f"{res['fits']['W_gain']['W_shift']:.2f}, W_item {res['fits']['W_gain']['W_item']:.1f} "
              f"({res['seconds']} s)", flush=True)
    if setting == "dove_mmlu":
        out["subjects_and_benchmarks"] = dove_subjects(data, out["variants"]["bank"]["fits"]["W_gain"])
    return out, rows


# --------------------------------------------------------------------------- macros and summary


def fmt(x, nd=1):
    if x is None:
        return "undefined"
    if isinstance(x, str):
        return x
    v = float(x)
    if not math.isfinite(v):
        return "undefined"
    if round(v, nd) == 0:
        v = 0.0
    return f"{v:,.{nd}f}".replace("-", "−")


def term(x, nd=0):
    """' + x/n' or ' − |x|/n' for a fitted formula."""
    return f" {'−' if x < 0 else '+'} {fmt(abs(x), nd)}/n"


def fmt_k(k):
    if k is None:
        return "undefined"
    if k == "unbounded":
        return "unbounded"
    return f"{int(k):,}"


def plan_row(variant, n, source="from_fit"):
    for r in variant["planning"][source]:
        if r["n"] == n:
            return r
    return None


def macros(R):
    m = {}
    for setting, s in R["settings"].items():
        suf = MACRO_SUFFIX[setting]
        v = s["variants"][s["primary_variant"]]
        sp, fw, dw = v["split"], v["fits"]["W_gain"], v["design"]["parts"]["W_gain"]
        m[f"noiseVarP{suf}"] = fmt(sp["var_e_intervention"], 2)
        m[f"noiseVarB{suf}"] = fmt(sp["var_e_baseline"], 2)
        m[f"noiseCov{suf}"] = fmt(sp["cov_e"], 2)
        m[f"noiseMinusTwoCov{suf}"] = fmt(sp["minus_2cov_e"], 2)
        m[f"noiseWgain{suf}"] = fmt(sp["W_gain"], 2)
        m[f"noiseCorr{suf}"] = fmt(sp["corr_e"], 2)
        m[f"noiseRatioGainBase{suf}"] = fmt(sp["ratio_W_gain_to_W_baseline"], 2)
        m[f"noiseRatioPB{suf}"] = fmt(sp["ratio_intervention_to_baseline"], 2)
        m[f"noiseVdelta{suf}"] = fmt(sp["V_delta"], 2 if sp["V_delta"] < 100 else 0)
        m[f"noiseWshift{suf}"] = fmt(fw["W_shift"], 2)
        m[f"noiseWshiftSe{suf}"] = fmt(fw["W_shift_se"], 3)
        m[f"noiseWitem{suf}"] = fmt(fw["W_item"], 0)
        m[f"noiseWitemSe{suf}"] = fmt(fw["W_item_se"], 0)
        m[f"noiseWshiftDesign{suf}"] = fmt(dw["W_shift"], 2)
        m[f"noiseWitemDesign{suf}"] = fmt(dw["W_item"], 0)
        for part, tag in (("var_e_intervention", "VarP"), ("var_e_baseline", "VarB"), ("cov_e", "Cov")):
            m[f"noiseWshift{tag}{suf}"] = fmt(v["fits"][part]["W_shift"], 2)
            m[f"noiseWitem{tag}{suf}"] = fmt(v["fits"][part]["W_item"], 0)
        m[f"noiseShareShiftFull{suf}"] = fmt(100 * fw["W_shift"] / sp["W_gain"], 0) if sp["W_gain"] > 0 else "undefined"
        for n, tag in ((100, "Hundred"), (1319, "GsmSize"), (s["N"], "Full")):
            r = plan_row(v, n)
            if r is None:
                continue
            m[f"noiseWAt{tag}{suf}"] = fmt(r["W"], 1)
            m[f"noiseRAt{tag}{suf}"] = fmt(r["R"], 2)
            m[f"noiseCOneAt{tag}{suf}"] = fmt(r["C_1"], 2)
            m[f"noiseKEightyAt{tag}{suf}"] = fmt_k(r["K_80"])
            m[f"noiseKNinetyAt{tag}{suf}"] = fmt_k(r["K_90"])
        m[f"noiseN{suf}"] = f"{s['N']:,}"
        m[f"noiseM{suf}"] = str(len(s["panel"]))
        m[f"noiseK{suf}"] = f"{v['K']:,}"
        if setting == "dove_mmlu":
            sb = s["subjects_and_benchmarks"]
            m["noiseSubjectWMedianDove"] = fmt(sb["W_gain_median"], 1)
            m["noiseSubjectWLowDove"] = fmt(sb["W_gain_q25"], 1)
            m["noiseSubjectWHighDove"] = fmt(sb["W_gain_q75"], 1)
            m["noiseSubjectRMedianDove"] = fmt(sb["R_median"], 2)
            m["noiseSubjectAboveFitPctDove"] = fmt(100 * sb["frac_subject_W_above_fit"], 0)
            if "dove_tasks" in sb:
                t = sb["dove_tasks"]
                m["noiseTasksN"] = str(t["n_tasks"])
                m["noiseTasksWMedian"] = fmt(t["W_gain_median"], 1)
                m["noiseTasksWLow"] = fmt(t["W_gain_q25"], 1)
                m["noiseTasksWHigh"] = fmt(t["W_gain_q75"], 1)
                m["noiseTasksRMean"] = fmt(t["R_mean"], 2)
                m["noiseTasksRPredMean"] = fmt(t["R_predicted_mean"], 2)
                m["noiseTasksRPredMAE"] = fmt(t["R_predicted_mean_abs_error"], 2)
                m["noiseTasksRPredCorr"] = fmt(t["R_predicted_corr"], 2)
    ex = R.get("dove_extraction")
    if ex:
        m["noiseExtractCells"] = f"{ex['cells_compared_with_mmlu_block']:,}"
        m["noiseExtractScores"] = f"{ex['rows_written']:,}"
        m["noiseExtractFiles"] = f"{ex['files']:,}"
    m["noiseSubsamplesPerN"] = str(R["config"]["subsamples_per_n"])
    ch = R.get("split_only", {}).get("cheng")
    if ch:
        for lab, suf in (("bf16", "Cheng"), ("all", "ChengAll")):
            sp = ch["variants"][lab]["split"]
            m[f"noiseVarP{suf}"] = fmt(sp["var_e_intervention"], 2)
            m[f"noiseVarB{suf}"] = fmt(sp["var_e_baseline"], 2)
            m[f"noiseCov{suf}"] = fmt(sp["cov_e"], 2)
            m[f"noiseCorr{suf}"] = fmt(sp["corr_e"], 2)
            m[f"noiseWgain{suf}"] = fmt(sp["W_gain"], 2)
            m[f"noiseRatioGainBase{suf}"] = fmt(sp["ratio_W_gain_to_W_baseline"], 2)
            m[f"noiseVdelta{suf}"] = fmt(sp["V_delta"], 2)
            m[f"noiseM{suf}"] = str(ch["variants"][lab]["M"])
        m["noiseKCheng"] = str(ch["K"])
    # LaTeX-ready: true minus signs become $-$ as in paper/make_numbers.py.
    return {k: v.replace("−", "$-$") for k, v in m.items()}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def summary(R):
    S = R["settings"]
    L = ["# Noise anatomy (analysis N)", ""]
    cfg = R["config"]
    L.append(f"Computed {R['created']} from `config/noise_anatomy.yaml` (sha256 `{R['config_sha256'][:12]}`), written "
             "before any subsample was drawn. Scores are in percentage points and variances in pp². Panel moments "
             "use denominator M − 1. The definitions are in `src/noise_anatomy.py`.")
    L.append("")
    L.append("## Choices")
    L.append("")
    L.append("- Split: W_gain = Var(e_P) + Var(e_B) − 2 Cov(e_P, e_B), where e is a draw's deviation from the model's "
             "expected score, P the intervention arm and B the baseline arm. Each setting uses the moment definition "
             "of its existing analysis, so that the parts add up to the reported W_gain: on the MMLU block the exact "
             "moments over the 2,716 common configurations under the uniform sampler "
             "(`results/dataset1/dove_mmlu.json`), on GSM8K the Appendix A cross-draw moments of the 24 discovery "
             "units (`results/dataset2/gsm8k.json`), strict extraction. Both definitions are bilinear, so the split "
             "is exact. Sensitivities: the 24 DOVE discovery draws with the Appendix A moments, and all 48 GSM8K "
             "units.")
    L.append(f"- Item subsamples: with the configuration draws held fixed, {cfg['subsamples_per_n']} simple random "
             "samples of n items without replacement per n, one stream per setting and n from seed "
             f"{cfg['seed']}, the same subsamples for every draw variant of a setting. At the full item count the "
             "one subsample is the full set. A subsample's score is 100 (N/n) Σ v_i s_i over its items, with v the "
             "item weights of the full score: uniform on GSM8K, 1/(56 × subject size) on the MMLU block, so the full "
             "set reproduces the pooled score of the existing analysis.")
    L.append("- Fit of W(n) = W_shift + W_item / n: the line passes through the full-set value W(N), which has no "
             "subsampling variance, and W_item is the weighted least-squares slope of (mean W at n − W(N)) on "
             "(1/n − 1/N) over the smaller n, with weights 1 / (Monte Carlo standard error of the mean)². Under "
             "simple random sampling of items this form is exact: E W(n) = W(N) + (1/n − 1/N) Q, where Q is the "
             "finite-population item variance of the noise moment. The 'design' columns compute W_shift and W_item "
             "from Q directly, without subsamples. An unweighted two-coefficient OLS of the per-n means on 1/n is in "
             "the JSON as a sensitivity. The standard errors are Monte Carlo errors of the subsampling, conditional "
             "on the setting's items and configuration draws.")
    L.append("- Planning rule: R(n) = V_delta / (V_delta + W(n)) with the fitted W(n) and V_delta held at its "
             "full-set value from the same draws; R_K = K R / (1 + (K − 1) R); K is the smallest number of prompt "
             "draws per evaluation for which C_K = 1/2 + arcsin(R_K)/π reaches 80% or 90% "
             "(`prompts_needed_gaussian`, Appendix A). The reference is conditional and not a fixed-panel law "
             "(a fixed panel can give a different R). The rule is tabulated only up to the setting's own item count.")
    L.append("- 100-item DOVE benchmarks: each MMLU subject of the block taken as a benchmark of its own items (subject "
             "accuracy, exact over the same configurations); for the 71 benchmarks of `results/dataset1/dove_tasks.json`, "
             "W_gain recovered from their exact V_delta and R as V_delta (1 − R)/R.")
    added = [x for x in S if x in ADDED and ADDED[x].get("pool", "base") == "base"]
    later = [x for x in S if x in ADDED and ADDED[x].get("pool", "base") != "base"]
    if added:
        L.append("- Added contrasts (`config/added_contrasts.yaml`): "
                 + "; ".join(f"{SHORT[x]} ({TITLE[x]}), with the item subsamples of {SHORT[S[x].get('subsamples_of', x)]}"
                             for x in added)
                 + ". Each uses the Appendix A moments of its 24 discovery units, checked against its own Steps 1–3 "
                   "file, and the item grid and item subsamples of the setting named, so the two compare on the same "
                   "items. They add rows and macros (suffix the contrast's tag) and change no other entry.")
    if later:
        L.append("- Added contrasts of a later pool (HotpotQA; `config/added_contrasts.yaml`): "
                 + "; ".join(f"{SHORT[x]} ({TITLE[x]})" for x in later)
                 + ". Each uses the Appendix A moments of its 24 discovery units, checked against its own Steps 1–3 "
                   "file; the three share one item grid and one set of item subsamples (`n_grid` entry "
                   + ", ".join(sorted({f"`{S[x].get('subsamples_of', x)}`" for x in later}))
                   + " of the configuration, streams named after it). They add rows and macros (suffix the contrast's "
                   "tag) and change no other entry.")
    L.append("")
    L.append("## Arm-wise split of the gain noise")
    L.append("")
    rows = []
    for setting, s in S.items():
        for name, v in s["variants"].items():
            sp = v["split"]
            tag = "" if name == s["primary_variant"] else " (sensitivity)"
            rows.append([f"{SHORT[setting]}, {name}{tag}", f"{v['K']:,}", v["M"], fmt(sp["var_e_intervention"], 2),
                         fmt(sp["var_e_baseline"], 2), fmt(sp["cov_e"], 2), fmt(sp["corr_e"], 2),
                         fmt(sp["W_gain"], 2), fmt(sp["ratio_W_gain_to_W_baseline"], 2), fmt(sp["V_delta"], 2)])
    ch = R.get("split_only", {}).get("cheng")
    if ch:
        for lab, title in (("bf16", "Exemplars (Cheng et al.), 8 configurations"),
                           ("all", "Exemplars, with the two FP8 models (sensitivity)")):
            v = ch["variants"][lab]
            sp = v["split"]
            rows.append([title, f"{ch['K']:,}", v["M"], fmt(sp["var_e_intervention"], 2), fmt(sp["var_e_baseline"], 2),
                         fmt(sp["cov_e"], 2), fmt(sp["corr_e"], 2), fmt(sp["W_gain"], 2),
                         fmt(sp["ratio_W_gain_to_W_baseline"], 2), fmt(sp["V_delta"], 2)])
    L.append(md_table(["Setting, draws", "K", "M", "Var(e_P)", "Var(e_B)", "Cov(e_P, e_B)", "Corr", "W_gain",
                       "W_gain / Var(e_B)", "V_delta"], rows))
    L.append("")
    if ch:
        sp = ch["variants"]["bf16"]["split"]
        effect = (f"which removes {fmt(2 * sp['cov_e'], 2)} pp² from their sum" if sp["cov_e"] >= 0
                  else f"which adds {fmt(-2 * sp['cov_e'], 2)} pp² to their sum")
        L.append(f"- Exemplars: Cheng et al. Part B, {ch['arms']['intervention']} against "
                 f"{ch['arms']['baseline']} on GSM8K ({ch['N']:,} items), the {ch['K']} transported configurations as "
                 f"distinct draws with the Appendix A cross-draw moments; split only, no item subsamples. Both arms "
                 f"are chain of thought, and each arm's noise is small ({fmt(sp['var_e_intervention'], 2)} and "
                 f"{fmt(sp['var_e_baseline'], 2)}); the errors correlate at {fmt(sp['corr_e'], 2)}, {effect}. W_gain "
                 f"is {fmt(sp['W_gain'], 2)} pp², against 7–12 pp² for chain of thought and five-shot above. The "
                 "configurations come from the frozen GSM8K bank, so this row is comparable with the GSM8K rows in "
                 "prompt design but not in panel.")
    L.append("")
    for setting, s in S.items():
        v = s["variants"][s["primary_variant"]]
        c = v["check_against_analysis"]
        if c:
            L.append(f"- {SHORT[setting]}: W_gain, W_baseline and V_delta reproduce `{c['reference']}` to within "
                     f"{TOL:g} relative" + (", and so does Cov(e_P, e_B) (its `coupling.E_cov_eP_eB`)" if
                                           "cov_e_matches" in c else "")
                     + "; the parts add up to W_gain (residual "
                     f"{v['split']['residual']:.1e}).")
        sp = v["split"]
        arms = s["arms"]
        effect = (f"which removes {fmt(2 * sp['cov_e'], 2)} pp² from their sum" if sp["cov_e"] >= 0
                  else f"which adds {fmt(-2 * sp['cov_e'], 2)} pp² to their sum")
        L.append(f"- {SHORT[setting]}: the intervention arm ({arms['intervention']}) has noise "
                 f"{fmt(sp['var_e_intervention'], 2)} against {fmt(sp['var_e_baseline'], 2)} for the baseline arm "
                 f"({arms['baseline']}), so the baseline arm carries {fmt(100 * sp['var_e_baseline'] / sp['W_gain'], 0)}% "
                 f"of W_gain by itself. The arms' errors correlate at {fmt(sp['corr_e'], 2)}, {effect}.")
    L.append("")
    L.append("## Noise against item count")
    L.append("")
    rows = []
    for setting, s in S.items():
        for name, v in s["variants"].items():
            fw, dw = v["fits"]["W_gain"], v["design"]["parts"]["W_gain"]
            tag = "" if name == s["primary_variant"] else " (sensitivity)"
            rows.append([f"{SHORT[setting]}, {name}{tag}", f"{v['N']:,}", fmt(fw["W_full"], 2),
                         f"{fmt(fw['W_shift'], 2)} ± {fmt(fw['W_shift_se'], 3)}",
                         f"{fmt(fw['W_item'], 0)} ± {fmt(fw['W_item_se'], 0)}",
                         fmt(dw["W_shift"], 2), fmt(dw["W_item"], 0), f"{fmt(fw['chi2'], 1)} on {fw['df']}"])
    L.append(md_table(["Setting, draws", "N", "W(N)", "W_shift ± se", "W_item ± se", "W_shift (design)",
                       "W_item (design)", "χ² of fit"], rows))
    L.append("")
    for setting, s in S.items():
        v = s["variants"][s["primary_variant"]]
        L.append(f"Per-n W_gain, {SHORT[setting]} ({v['label']}):")
        L.append("")
        fr = v["fits"]
        sig = {r["n"]: r for r in v["signal_by_n"]}
        rows = []
        for r in fr["W_gain"]["per_n"]:
            n = r["n"]
            parts = {p: next(x for x in fr[p]["per_n"] if x["n"] == n)["mean"] for p in na.PARTS}
            rows.append([f"{n:,}", r["n_subsamples"], fmt(r["mean"], 2), f"{fmt(r['q05'], 1)}–{fmt(r['q95'], 1)}",
                         fmt(r["fitted"], 2), fmt(parts["var_e_intervention"], 2), fmt(parts["var_e_baseline"], 2),
                         fmt(parts["cov_e"], 2), fmt(sig[n]["V_delta_mean"], 2), fmt(sig[n]["R_mean"], 3)])
        L.append(md_table(["n", "subsamples", "mean W_gain", "5–95%", "fit", "Var(e_P)", "Var(e_B)", "Cov",
                           "mean V_delta(n)", "mean R(n)"], rows))
        L.append("")
        parts = {p: v["fits"][p] for p in na.PARTS}
        fw = parts["W_gain"]
        L.append(f"Fitted: W_gain = {fmt(fw['W_shift'], 2)}{term(fw['W_item'])}, so the shift part is "
                 f"{fmt(100 * fw['W_shift'] / fw['W_full'], 0)}% of W_gain on all {v['N']:,} items and "
                 f"{fmt(100 * fw['W_shift'] / (fw['W_shift'] + fw['W_item'] / 100), 0)}% at 100 items. By arm: "
                 f"Var(e_P) = {fmt(parts['var_e_intervention']['W_shift'], 2)}{term(parts['var_e_intervention']['W_item'])}, "
                 f"Var(e_B) = {fmt(parts['var_e_baseline']['W_shift'], 2)}{term(parts['var_e_baseline']['W_item'])}, "
                 f"Cov(e_P, e_B) = {fmt(parts['cov_e']['W_shift'], 2)}{term(parts['cov_e']['W_item'])}.")
        L.append("")
    arm_rows = []
    for setting, s in S.items():
        f = s["variants"][s["primary_variant"]]["fits"]
        arm_rows.append(f"{SHORT[setting]}: {fmt(f['var_e_intervention']['W_shift'], 2)} against "
                        f"{fmt(f['var_e_baseline']['W_shift'], 2)} for the shift part, "
                        f"{fmt(f['var_e_intervention']['W_item'], 0)} against {fmt(f['var_e_baseline']['W_item'], 0)} "
                        "for the item part")
    L.append("Intervention arm against baseline arm, " + "; ".join(arm_rows) + ".")
    L.append("")
    L.append("V_delta(n) is the subsample's own true-gain variance. It includes the item sample's model-by-item "
             "differences, so it rises as n falls. The planning rule holds V_delta at its full-set value.")
    L.append("")
    L.append("## Planning rule")
    L.append("")
    for setting, s in S.items():
        v = s["variants"][s["primary_variant"]]
        rows = [[f"{r['n']:,}", fmt(r["W"], 1), fmt(r["R"], 3), fmt(r["C_1"], 3), fmt_k(r["K_80"]), fmt_k(r["K_90"])]
                for r in v["planning"]["from_fit"]]
        L.append(f"{SHORT[setting]}, V_delta = {fmt(v['planning']['V_delta'], 2)} pp²:")
        L.append("")
        L.append(md_table(["n items", "W(n)", "R(n)", "C_1", "K for 80%", "K for 90%"], rows))
        L.append("")
    if "dove_mmlu" in S:
        sb = S["dove_mmlu"]["subjects_and_benchmarks"]
        v = S["dove_mmlu"]["variants"]["bank"]
        L.append("## The 100-item DOVE benchmarks")
        L.append("")
        r100 = plan_row(v, 100)
        L.append(f"A random 100-item sample of the MMLU block has expected gain noise W(100) = "
                 f"{fmt(sb['W_fit_at_100'], 1)} pp², against {fmt(v['split']['W_gain'], 2)} on all "
                 f"{v['N']:,} items; with the block's V_delta that lowers R from {fmt(v['split']['R_benefit'], 3)} to "
                 f"{fmt(r100['R'], 3)}. With each subsample's own V_delta(100), the mean R of the 100-item subsamples "
                 f"is {fmt(next(r for r in v['signal_by_n'] if r['n'] == 100)['R_mean'], 3)}.")
        L.append("")
        L.append(f"The block's {sb['n_subjects']} subjects, each taken as a benchmark of its own 99 or 100 items, have "
                 f"W_gain median {fmt(sb['W_gain_median'], 1)} (quartiles {fmt(sb['W_gain_q25'], 1)}–"
                 f"{fmt(sb['W_gain_q75'], 1)}), V_delta median {fmt(sb['V_delta_median'], 2)} and R median "
                 f"{fmt(sb['R_median'], 3)}. {fmt(100 * sb['frac_subject_W_above_fit'], 0)}% of them lie above the "
                 "fitted W at their size: 100 items of one subject are noisier than 100 items drawn across subjects.")
        t = sb.get("dove_tasks")
        if t:
            L.append("")
            L.append(f"Across the {t['n_tasks']} benchmarks of the secondary analysis ({' or '.join(str(x) for x in t['n_items'])} "
                     f"items each), W_gain has median {fmt(t['W_gain_median'], 1)} (quartiles "
                     f"{fmt(t['W_gain_q25'], 1)}–{fmt(t['W_gain_q75'], 1)}), about "
                     f"{fmt(t['W_gain_median'] / v['split']['W_gain'], 1)} times the pooled block's, while V_delta has "
                     f"median {fmt(t['V_delta_median'], 2)} (mean {fmt(t['V_delta_mean'], 2)}) against the block's "
                     f"{fmt(v['split']['V_delta'], 2)}. Their R has mean {fmt(t['R_mean'], 3)} against "
                     f"{fmt(v['split']['R_benefit'], 3)} for the block. Using the block's fitted W(100) as each "
                     f"benchmark's noise, R = V_delta/(V_delta + W(100)) has mean {fmt(t['R_predicted_mean'], 3)}, mean "
                     f"absolute error {fmt(t['R_predicted_mean_abs_error'], 3)} and correlation "
                     f"{fmt(t['R_predicted_corr'], 2)} against the observed R. The item count accounts for the higher "
                     "noise; the larger true variation of single benchmarks offsets part of it.")
        L.append("")
    L.append("## Inputs and checks")
    L.append("")
    ex = R.get("dove_extraction")
    if ex:
        L.append(f"- MMLU block item scores: `scripts/extract_dove_items.py` read {ex['files']} raw DOVE_Lite files "
                 f"(revision `{ex['dove_revision'][:12]}`; columns {', '.join('`' + c + '`' for c in ex['columns_read'])}), "
                 f"kept the block's common configurations and item intersection, and wrote {ex['rows_written']:,} "
                 f"item scores to `data/processed/dove/mmlu_items.parquet`. The item sums reproduce `n_correct` in "
                 f"`mmlu_block.parquet` for all {ex['cells_compared_with_mmlu_block']:,} (subject, model, arm, "
                 f"configuration) cells, and the item scores equal those of `scripts/build_dove_tables.py` in "
                 f"{ex['files_compared_with_processed_items']} of {ex['files']} files "
                 f"({ex['item_score_differences_with_processed_items']} differences). Record: "
                 "`results/noise_anatomy/dove_item_extraction.json`.")
    for setting, s in S.items():
        prov = s["provenance"]
        if setting == "dove_mmlu":
            L.append("- MMLU block: the full-set scores from the item table equal the pooled scores of "
                     "`scripts/run_dataset1.py` (from `mmlu_block.parquet`) for every configuration, model and arm.")
        else:
            L.append(f"- {SHORT[setting]}: {', '.join('`' + t['path'] + '`' for t in prov['tables'])}, value "
                     f"`{prov['value']}`; the full-set scores equal `src/dataset2.cell_table` for the discovery and "
                     "all units; the table passes the input checks of `scripts/run_dataset2.py` (labels, generation "
                     "folders, sampler, items and scorer).")
    missing = [x for x in SETTINGS if x not in S]
    if missing:
        L.append("- Not yet run: " + ", ".join(SHORT[x] for x in missing) + " (`--settings " + " ".join(missing) +
                 "` once their scored tables and analyses exist).")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("- `results/noise_anatomy/noise_anatomy.json` (with `macros`), `results/noise_anatomy/tables/subsamples.csv`")
    L.append("- Code: `src/noise_anatomy.py`, `scripts/extract_dove_items.py`, `scripts/run_noise_anatomy.py`, "
             "`tests/test_noise_anatomy.py`")
    L.append("")
    return "\n".join(L)



# --------------------------------------------------------------------------- main


def cheng_split():
    """Arm-wise split of the exemplar contrast: Cheng et al. Part B, eight-exemplar against
    zero-shot chain of thought, the 8 transported configurations as distinct draws, Appendix A cross-draw moments
    (src/noise_anatomy.arm_split, "draws"). Split only: no item subsamples. The loader is scripts/run_map_panels.py's."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("run_map_panels_for_noise", os.path.join(here, "run_map_panels.py"))
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    ch = mp.cheng()
    out = {"arms": {"intervention": ch["intervention_arm"], "baseline": ch["baseline_arm"]},
           "K": len(ch["configurations"]), "N": ch["n_items"], "primary_variant": "bf16",
           "note": "the two FP8 70B-class models enter only the 'all' sensitivity, as in results/reproduction/"}
    out["variants"] = {lab: {"M": ch[lab]["M"], "split": ch[lab]["noise_split"],
                             "models": ch["models_bf16"] if lab == "bf16" else ch["models_all"]}
                       for lab in ("bf16", "all")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", choices=ALL_SETTINGS, default=["dove_mmlu", "gsm8k"])
    ap.add_argument("--split-only", action="store_true",
                    help="recompute only the split-only setting (the exemplar contrast) and rewrite the outputs")
    a = ap.parse_args()
    path = os.path.join(OUT_DIR, "noise_anatomy.json")
    if a.split_only:
        t0 = time.time()
        R = json.load(open(path))
        R["split_only"] = {"cheng": cheng_split()}
        R["macros"] = macros(R)
        R["split_only_created"] = time.strftime("%Y-%m-%d %H:%M")
        dump(R, path)
        text = summary(R)
        for name in ("summary.md", "summary_clean.md"):
            with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        print(f"split-only setting done in {time.time() - t0:.1f} s -> {rel(path)}")
        return
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG))
    R = json.load(open(path)) if os.path.exists(path) else {"settings": {}}
    R["settings"] = {k: v for k, v in R.get("settings", {}).items() if k not in a.settings}
    R.update({"analysis": "N, noise anatomy", "units": "percentage points; pp^2",
              "created": time.strftime("%Y-%m-%d %H:%M"), "config": cfg, "config_path": rel(CONFIG),
              "config_sha256": sha256_file(CONFIG)})
    rows = []
    for setting in a.settings:
        print(f"{setting}: loading", flush=True)
        try:
            data = (load_dove() if setting == "dove_mmlu" else load_added(setting) if setting in ADDED
                    else load_d2(setting))
            out, r = run_setting(data, cfg)
        except Refusal as exc:
            sys.exit(f"refused: {exc}")
        R["settings"][setting] = out
        rows += r
        if setting == "dove_mmlu":
            ex = json.load(open(os.path.join(OUT_DIR, "dove_item_extraction.json")))
            R["dove_extraction"] = {k: ex[k] for k in ("dove_revision", "files", "columns_read", "rows_written",
                                                       "runtime_seconds")}
            R["dove_extraction"].update({k: ex["checks"][k] for k in (
                "cells_compared_with_mmlu_block", "files_with_problems", "files_compared_with_processed_items",
                "item_score_differences_with_processed_items")})
        del data
    R["settings"] = {k: R["settings"][k] for k in ALL_SETTINGS if k in R["settings"]}
    R["split_only"] = {"cheng": cheng_split()}
    tab = os.path.join(OUT_DIR, "tables", "subsamples.csv")
    os.makedirs(os.path.dirname(tab), exist_ok=True)
    new = pd.DataFrame(rows)
    if os.path.exists(tab):
        old = pd.read_csv(tab, float_precision="round_trip")      # keeps the other settings' rows exact
        new = pd.concat([old[~old["setting"].isin(a.settings)], new], ignore_index=True)
    new.to_csv(tab, index=False)
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    dump(R, path)
    text = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    print(f"done in {R['runtime_seconds']} s -> {rel(path)}")


if __name__ == "__main__":
    main()
