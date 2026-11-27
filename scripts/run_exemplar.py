r"""Cheng et al.'s eight exemplars against zero-shot chain of thought on GSM8K-29,
as a full setting with a validation split.

PowerShell (from the repository root), after scripts\score_exemplar.py --score:
    cmd /c ".venv\Scripts\python.exe scripts\run_exemplar.py 2>&1"
A diagnostic on the models scored so far (written only under --out, never to manifests/):
    cmd /c ".venv\Scripts\python.exe scripts\run_exemplar.py --partial --quick --out <scratch dir> 2>&1"
A dry run before the data exist (the eight-shot arm replaced by the collection's direct-answer arm):
    cmd /c ".venv\Scripts\python.exe scripts\run_exemplar.py --dry-run --quick --out <scratch dir> 2>&1"
One draw's prompt noise per arm, added to an existing results/exemplar/exemplar.json from the discovery draws alone
(macros exhNoiseVarB and exhNoiseVarP; nothing else is recomputed):
    cmd /c ".venv\Scripts\python.exe scripts\run_exemplar.py --noise-split 2>&1"

Arms per model: zero-shot chain of thought, the collection's outputs (the baseline), and eight-shot chain of thought,
the same frames preceded by the authors' eight exemplar turns. The panel, items and frozen sampler are the
collection's (24 discovery and 24 validation draws). Scores: the GSM8K scorer on both arms; the four
checkpoints run at 2,048 tokens have their longer zero-shot outputs cut to 2,048 tokens and scored again
(data/processed/exemplar/zero_shot_4k.parquet), so both arms are read at one cap.

Order:
1. discovery draws of the two arms;
2. the map's pilot predictions (R from the first 2 and 4 draws, config/reliability_map.yaml) saved to
   manifests/predictions_exemplar_map.json;
3. Steps 1-3 by scripts/run_dataset2.py's run() with zero-shot chain of thought as the baseline arm: its prediction
   rules are saved to manifests/predictions_exemplar_gsm8k.json (and _lenient) before it builds the validation
   draws; its output is results/exemplar/exemplar_gsm8k.json;
4. validation draws; then the slope on three scales, the map point, the per-model table and Cheng et al.'s claims.

Cheng et al.'s claims (exemplars help weak models and not strong ones), on the panel's models that are among the
reproduction's 13 BF16 models, with the reproduction's groups (config/reproduction.yaml), per distinct configuration of
the sampler: the published claim (no strong model's paired benefit interval above zero), the capability slope (benefit
on zero-shot accuracy) on those models and on the whole panel, and the sign of each model's mean benefit against the
published one. Read with the authors' corrected extractor (data/processed/exemplar/cheng_scored.parquet) and with the
strict scorer.

Outputs: results/exemplar/exemplar.json (with "macros", prefix exh), exemplar_gsm8k.json, summary.md,
summary_clean.md, tables/per_model.csv and tables/cheng_per_configuration.csv.
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
from src import dataset2 as d2, instability as ins, noise_anatomy as na, reliability_map as rm, sampling  # noqa: E402
from run_forced_direct import ci, collection_inputs, dump, fmt, hset, jsonable  # noqa: E402

OUT = os.path.join(ROOT, "results", "exemplar")
MANIFESTS = os.path.join(ROOT, "manifests")
PROCESSED = os.path.join(ROOT, "data", "processed", "exemplar")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
MAP_CONFIG = os.path.join(ROOT, "config", "reliability_map.yaml")
INST_CONFIG = os.path.join(ROOT, "config", "instability.yaml")
ARM = "eight_shot_cot"
TITLE = "GSM8K (29 models): eight-shot against zero-shot chain of thought"


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ============================================================================ inputs


def exemplar_inputs(panel, sampler, items_info, col_load, dry_run, partial):
    """Loader of the eight-shot rows after the checks of the scored table; in a dry run the collection's direct-answer
    rows relabelled. With `partial` the panel shrinks to the models scored so far."""
    if dry_run:
        def load(ids):
            df = col_load(ids)
            return df[df["arm"] == "direct_answer"].assign(arm=ARM)
        return load, {"dry_run": "eight-shot arm replaced by the collection's direct-answer arm"}, panel
    from src import d2_io
    path = os.path.join(PROCESSED, "gsm8k_scored.parquet")
    md = d2_io.scored_metadata(path)
    problems = d2_io.main_problems(md, items_info=items_info, sampler=sampler,
                                           scorer_path=os.path.join(MANIFESTS, "gsm8k_scorer.json"))
    models = sorted({i["model_key"] for i in json.loads(md["d2_inputs"])})
    if partial:
        panel = [k for k in panel if k in models]
    elif sorted(panel) != models:
        problems.append(f"models {sorted(set(panel) ^ set(models))} are in only one of the panel and the table")
    if problems:
        raise SystemExit(f"refused: {rel(path)}: " + "; ".join(problems))

    def load(ids):
        df = pd.read_parquet(path, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        if set(df["arm"].unique()) - {ARM}:
            raise SystemExit(f"refused: {rel(path)} holds arms other than {ARM}")
        return df[df["model_key"].isin(panel)]
    prov = {"path": rel(path), "sha256": d2_io.sha256_path(path), "scorer_version": md.get("d2_scorer_version"),
            "gpu_types": {i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])}}
    return load, prov, panel


def cut_table():
    p = os.path.join(PROCESSED, "zero_shot_4k.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"refused: {rel(p)} is missing; run scripts/score_exemplar.py --score")
    df = pd.read_parquet(p)
    return df if len(df) else None


def zero_shot(col, cut):
    """The collection's zero-shot rows, with the cut outputs of the four 2,048-token checkpoints scored at that cap."""
    z = col[col["arm"] == "zero_shot_cot"].copy()
    if cut is None:
        return z, 0
    key = ["model_key", "config_id", "item_id"]
    c = cut[key + ["strict_correct", "lenient_correct", "compliant", "finish_reason"]].copy()
    c["item_id"] = c["item_id"].astype(z["item_id"].dtype)
    z = z.merge(c, on=key, how="left", suffixes=("", "_cut"))
    hit = z["strict_correct_cut"].notna()
    for v in ("strict_correct", "lenient_correct", "compliant", "finish_reason"):
        z.loc[hit, v] = z.loc[hit, v + "_cut"].astype(z[v].dtype) if v != "finish_reason" else z.loc[hit, v + "_cut"]
    return z.drop(columns=[v + "_cut" for v in ("strict_correct", "lenient_correct", "compliant", "finish_reason")]), int(hit.sum())


def combined_loader(col_load, ex_load, cut):
    """Rows of both arms under the analysis's arm names: zero-shot as the baseline (direct_answer), eight-shot as the
    intervention (zero_shot_cot)."""
    def load(ids):
        z, _ = zero_shot(col_load(ids), cut)
        return pd.concat([z.assign(arm="direct_answer"), ex_load(ids).assign(arm="zero_shot_cot")], ignore_index=True)
    return load


def arrays(load, units, panel, items, value="strict_correct"):
    cells = d2.cell_table(load(units), panel, units, items, values=tuple(dict.fromkeys(tuple(d2.VALUES) + (value,))))
    return d2.unit_arrays(cells, units, panel, value)


def gpu_check(col_prov, ex_prov):
    from src import d2_io
    col = {}
    for t in col_prov.get("tables", []):
        md = d2_io.scored_metadata(os.path.join(ROOT, t["path"]))
        col.update({i["model_key"]: i.get("gpu_type") for i in json.loads(md["d2_inputs"])})
    ex = ex_prov.get("gpu_types") or {}
    return {"mismatched": sorted(k for k in ex if col.get(k) != ex[k]),
            "by_model": {k: {"collection": col.get(k), "exemplar": ex[k]} for k in ex}}


# ============================================================================ blocks


def per_model(panel, B, P, ex_rows):
    rows = []
    for j, k in enumerate(panel):
        g = ex_rows[ex_rows["model_key"] == k]
        rows.append({"model_key": k, "zero_shot": B[:, j].mean(), "eight_shot": P[:, j].mean(),
                     "benefit": (P[:, j] - B[:, j]).mean(), "benefit_sd_over_draws": float((P[:, j] - B[:, j]).std(ddof=1)),
                     "eight_shot_compliant_pct": 100 * float(g["compliant"].mean()),
                     "eight_shot_finish_length_pct": 100 * float((g["finish_reason"] == "length").mean())})
    return rows


def map_point(Bd, Pd, Bv, Pv, frozen, cfg, n_splits, M, n_items):
    obs = rm.validation_agreement(Bv, Pv, cfg["group_sizes"], n_splits,
                                  sampling.rng_for("exemplar_map", "splits", "gsm8k", seed=cfg["seed"]))
    point = rm.setting_point(Bd, Pd)
    rows = []
    for q in ("benefit", "accuracy"):
        p = point[q]
        err = {}
        for n in cfg["pilot_sizes"]:
            pr = frozen["settings"]["gsm8k"][str(n)][q]
            err[n] = pr["C_one_draw"] - obs["one_draw"][q]["C"] if pr["defined"] else None
        rows.append({"setting": "exemplar:gsm8k", "quantity": q, "M": M, "n_items": n_items, "R": p["R"], "C": p["C"],
                     "C_ref": p.get("C_ref"), "departure": p.get("departure"), "tie": p.get("tie"),
                     "pilot_error_2": err.get(2), "pilot_error_4": err.get(4),
                     "validation_C_one_draw": obs["one_draw"][q]["C"]})
    return rows, point, obs


def cheng_block(panel, configs, strict_rows, cheng_rows):
    """Cheng et al.'s claims per distinct configuration (module doc), under each scorer."""
    from src import reproduction as rp
    pub = rp.published_table(rp.load_config())
    pub = pub[(pub["precision"] == "bf16") & pub["model_key"].isin(panel)]
    overlap = sorted(pub["model_key"])
    strong = sorted(pub.loc[pub["group"] == "strong", "model_key"])
    out = {"models": overlap, "strong": strong, "weak": sorted(set(overlap) - set(strong)),
           "rule": "the claim holds in a configuration when no strong model's paired benefit interval (Newcombe, 95%) "
                   "lies above zero", "scorers": {}}
    per_cfg_rows = []
    for name, sc in (("cheng", cheng_rows), ("strict", strict_rows)):
        sc = sc[sc["config_id"].isin(configs)].assign(part="exemplar_h")
        ben = rp.benefit_table(sc)
        per = {}
        for cid, g in ben.groupby("config_id", sort=True):
            o = g[g["model_key"].isin(overlap)]
            per[cid] = {"claim": rp.claim_holds(o, strong),
                        "slope_overlap": rp.capability_slope(o["benefit"], o["zero_shot"])["slope"],
                        "slope_panel": rp.capability_slope(g["benefit"], g["zero_shot"])["slope"]}
            per_cfg_rows += [{"scorer": name, "config_id": cid, **{k: r[k] for k in ("model_key", "zero_shot", "eight_shot",
                                                                                     "benefit", "benefit_lo", "benefit_hi")}}
                             for _, r in g.iterrows()]
        mean_b = ben.groupby("model_key")["benefit"].mean()
        signs = {k: {"ours": rp.sign(float(mean_b[k])), "published": rp.sign(float(v))}
                 for k, v in zip(pub["model_key"], pub["published_benefit"]) if k in mean_b.index}
        so = [v["slope_overlap"] for v in per.values()]
        sp = [v["slope_panel"] for v in per.values()]
        out["scorers"][name] = {
            "per_configuration": per, "n_configurations": len(per),
            "claim_holds_in": int(sum(v["claim"]["holds"] for v in per.values())),
            "slope_overlap": {"mean": float(np.mean(so)), "positive": int(sum(x > 0 for x in so)),
                              "negative": int(sum(x < 0 for x in so))},
            "slope_panel": {"mean": float(np.mean(sp)), "positive": int(sum(x > 0 for x in sp)),
                            "negative": int(sum(x < 0 for x in sp))},
            "mean_benefit": {k: float(v) for k, v in mean_b.items() if k in overlap},
            "sign_agreement": {"agree": int(sum(v["ours"] == v["published"] for v in signs.values() if v["ours"] is not None)),
                               "of": len(signs), "by_model": signs}}
    return out, per_cfg_rows


def noise_block(Bd, Pd, step2_noise):
    """One draw's prompt noise per arm on the discovery draws, W_gain = Var(e_P) + Var(e_B) - 2 Cov(e_P, e_B), as
    scripts/run_noise_anatomy.py splits it for the other settings; refused unless it reproduces Step 2's W_gain and
    W_baseline."""
    sp = na.arm_split(Bd, Pd, "draws")
    chk = {"W_gain_matches": bool(np.isclose(sp["W_gain"], step2_noise["W_gain"], rtol=1e-9, atol=1e-12)),
           "W_baseline_matches": bool(np.isclose(sp["var_e_baseline"], step2_noise["W_baseline"], rtol=1e-9, atol=1e-12)),
           "parts_add_up": bool(abs(sp["residual"]) <= 1e-9 * max(1.0, abs(sp["W_gain"])))}
    if not all(chk.values()):
        raise SystemExit(f"the noise split does not reproduce Step 2: {chk} (split {sp}, Step 2 {step2_noise})")
    return {"split": sp, "check_against_step2": chk}


def draw_slopes(Bd, Pd, n):
    """Per discovery draw, the least-squares slope of benefit on zero-shot accuracy across the panel, on each scale
    (descriptive; the patterns table of scripts/run_patterns.py reads it): counts by sign and the share of draws with
    the sign of the cross-draw slope. A draw whose baseline does not vary has no slope and is counted apart."""
    out = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(Bd, Pd, sc, n)
        X, Y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        sl = []
        for r in range(X.shape[0]):
            xc = X[r] - X[r].mean()
            d = float(xc @ xc)
            sl.append(float(xc @ (Y[r] - Y[r].mean())) / d if d > 0 else None)
        ok = [v for v in sl if v is not None]
        out[sc] = {"per_draw": sl, "positive": int(sum(v > 0 for v in ok)), "negative": int(sum(v < 0 for v in ok)),
                   "zero": int(sum(v == 0 for v in ok)), "undefined": int(len(sl) - len(ok))}
    return out


EQ_MARGIN = 2.0  # pp: about the SE of one model's two-arm accuracy difference on 1,319 items at 50%, rounded up


def tercile_block(panel, rank, Bv, Pv, margin=EQ_MARGIN):
    """The weak and strong terciles, the lowest and highest ceil(M/3) models by `rank` (mean strict zero-shot
    accuracy over the discovery draws); each tercile's mean benefit over the validation draws with t intervals over
    those draws. The strong tercile has no benefit when its 90% interval lies inside (-margin, margin) (two one-sided
    tests at 5%); the weak tercile benefits when its 95% interval lies above zero; the claim holds when both do."""
    from scipy import stats
    k = -(-len(panel) // 3)
    order = np.argsort(np.asarray(rank, float), kind="stable")
    groups = {"weak": order[:k], "strong": order[-k:]}
    G = np.asarray(Pv, float) - np.asarray(Bv, float)

    def tint(d, level):
        q = stats.t.ppf(0.5 + level / 2, len(d) - 1)
        h = q * d.std(ddof=1) / np.sqrt(len(d))
        return [float(d.mean() - h), float(d.mean() + h)]
    out = {"k": int(k), "margin_pp": margin, "K_validation": int(G.shape[0])}
    per = {}
    for name, ix in groups.items():
        d = G[:, ix].mean(1)
        per[name] = d
        out[name] = {"models": [panel[i] for i in ix], "rank_mean": float(np.mean(np.asarray(rank)[ix])),
                     "mean": float(d.mean()), "ci95": tint(d, 0.95), "ci90": tint(d, 0.90)}
    dd = per["weak"] - per["strong"]
    out["weak_minus_strong"] = {"mean": float(dd.mean()), "ci95": tint(dd, 0.95)}
    out["weak_benefits"] = bool(out["weak"]["ci95"][0] > 0)
    out["strong_equivalent"] = bool(-margin < out["strong"]["ci90"][0] and out["strong"]["ci90"][1] < margin)
    out["claim_holds"] = out["weak_benefits"] and out["strong_equivalent"]
    return out


def long_arrays(df, units, panel, base="zero_shot_cot", interv=ARM):
    """(B, P), each (K, M) pp, from item rows (model_key, config_id, arm, item_id, correct) with one row per item."""
    n = df["item_id"].nunique()
    acc = df.assign(config_id=df["config_id"].astype(str)).groupby(["config_id", "model_key", "arm"])["correct"].agg(
        ["mean", "size"])
    B, P = np.empty((len(units), len(panel))), np.empty((len(units), len(panel)))
    for r, u in enumerate(units):
        for j, k in enumerate(panel):
            for A, arm in ((B, base), (P, interv)):
                a = acc.loc[(str(u), k, arm)]
                if a["size"] != n:
                    raise SystemExit(f"{k} {arm} {u}: {a['size']} of {n} items")
                A[r, j] = 100.0 * a["mean"]
    return B, P


def strict_long(load, units, cut):
    """Item-level strict scores of both arms in the reproduction's layout (model_key, config_id, arm, item_id, correct)."""
    df = load(units)
    df = df.assign(arm=df["arm"].map({"direct_answer": "zero_shot_cot", "zero_shot_cot": ARM}))
    return df[["model_key", "config_id", "arm", "item_id"]].assign(correct=df["strict_correct"].astype(int))


# ============================================================================ run


def run(a):
    t0 = time.time()
    if (a.dry_run or a.partial) and os.path.abspath(a.pred_dir) == os.path.abspath(MANIFESTS):
        raise SystemExit("a dry run or a partial run writes its frozen predictions under --out, never to manifests/")
    import run_dataset2
    from run_instability import scale_block
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    icfg = yaml.safe_load(open(INST_CONFIG, encoding="utf-8"))
    sel = d2.load_selection(SELECTION)
    method, alpha = sel["H"], float(icfg["alpha"])
    n_splits = 200 if a.quick else int(mcfg["validation_splits"])
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)

    inp, s, items, items_info = collection_inputs("gsm8k")
    ex_load, ex_prov, panel = exemplar_inputs(list(inp["panel"]), s, items_info, inp["load"], a.dry_run, a.partial)
    keep = [i for i, k in enumerate(inp["panel"]) if k in panel]
    families = [inp["families"][i] for i in keep]
    tiers = {k: v for k, v in inp["tiers"].items() if k in panel} if inp["tiers"] else inp["tiers"]
    cut = None if a.dry_run else cut_table()
    load = combined_loader(inp["load"], ex_load, cut)
    disc, val = list(s["discovery_config_ids"]), list(s["validation_config_ids"])
    Bd, Pd = arrays(load, disc, panel, items)
    print(f"discovery arrays ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ map pilot predictions, discovery only
    from src.sampling import sha256_json
    preds = {"gsm8k": {str(n): rm.pilot_prediction(Bd[:n], Pd[:n], mcfg["group_sizes"], mcfg["agreement_targets"])
                       for n in mcfg["pilot_sizes"]}}
    fr = {"label": "exemplar_map", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {"gsm8k": panel},
          "samplers": {"gsm8k": s["sha256"]},
          "note": "pilot predictions of the eight-exemplar map point (R from the first 2 and 4 discovery draws), "
                  "written before the validation draws are built", "settings": jsonable(preds)}
    fr["sha256"] = sha256_json(fr["settings"])
    frozen_map = d2.write_frozen(fr, os.path.join(a.pred_dir, "predictions_exemplar_map.json"),
                                 overwrite=a.overwrite_frozen)

    # ------------------------------------------------------------ Steps 1-3 (freeze inside, then validation)
    prov = {"collection": inp["provenance"], "exemplar": ex_prov,
            "baseline_arm": "zero_shot_cot (relabelled direct_answer inside the analysis)",
            "intervention_arm": "eight_shot_cot (relabelled zero_shot_cot inside the analysis)",
            "zero_shot_cut_4k": None if cut is None else {"rows": int(len(cut)), "path": rel(os.path.join(PROCESSED, "zero_shot_4k.parquet"))}}
    steps = run_dataset2.run(load, s, panel, families, tiers, items, a.out,
                             os.path.join(a.pred_dir, "predictions_exemplar_gsm8k.json"), "exemplar_gsm8k", n_boot,
                             n_ctrl, overwrite_frozen=a.overwrite_frozen, provenance=prov,
                             tables_dir=os.path.join(a.out, "tables_exemplar_gsm8k"))

    # ------------------------------------------------------------ validation draws and the rest
    Bv, Pv = arrays(load, val, panel, items)
    n = len(items)
    R = {"analysis": "eight-shot against zero-shot chain of thought",
         "dry_run": bool(a.dry_run), "partial": bool(a.partial), "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": "percentage points; pp^2", "title": TITLE, "models": panel, "M": len(panel), "n_items": n,
         "K_discovery": Bd.shape[0], "K_validation": Bv.shape[0],
         "steps_file": rel(os.path.join(a.out, "exemplar_gsm8k.json")), "step2": steps["step2"],
         "step3_registered": steps["step3"]["registered"],
         "frozen_map": {"path": rel(os.path.join(a.pred_dir, "predictions_exemplar_map.json")),
                        "sha256": frozen_map["sha256"], "created": frozen_map["created"]},
         "config": {"reliability_map": rel(MAP_CONFIG), "instability": rel(INST_CONFIG)},
         "zero_shot_cut_4k": prov["zero_shot_cut_4k"]}
    R["noise_split"] = noise_block(Bd, Pd, steps["step2"]["noise"])
    ex_all = ex_load(disc + val)
    R["per_model"] = per_model(panel, Bd, Pd, ex_all)
    R["means"] = {k: float(np.mean([r[k] for r in R["per_model"]])) for k in ("zero_shot", "eight_shot", "benefit")}
    R["scale"] = scale_block(Bd, Pd, Bv, Pv, n, method, alpha)
    R["draw_slopes"] = draw_slopes(Bd, Pd, n)
    for sc, v in R["draw_slopes"].items():
        sg = np.sign(R["scale"][sc]["discovery"]["slope"])
        ok = [x for x in v["per_draw"] if x is not None]
        v["repeats"] = float(np.mean([np.sign(x) == sg for x in ok])) if ok else None
    pts, point, obs = map_point(Bd, Pd, Bv, Pv, frozen_map, mcfg, n_splits, len(panel), n)
    R["map"] = {"point": point, "validation_one_draw": obs["one_draw"]}
    R["map_points"] = pts
    configs = sorted(set(map(str, disc)) | set(map(str, val)))
    R["terciles"] = {"strict": tercile_block(panel, Bd.mean(0), Bv, Pv)}
    if not a.dry_run:
        cheng = pd.read_parquet(os.path.join(PROCESSED, "cheng_scored.parquet"))
        cheng = cheng[cheng["model_key"].isin(panel)]
        Bcv, Pcv = long_arrays(cheng, val, panel)
        R["terciles"]["cheng"] = tercile_block(panel, Bd.mean(0), Bcv, Pcv)    # the same terciles, Cheng's extractor
        R["cheng"], cfg_rows = cheng_block(panel, configs, strict_long(load, configs, cut), cheng)
        R["gpu_check"] = gpu_check(inp["provenance"], ex_prov)
    else:
        cfg_rows = []
    R["macros"] = macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    pd.DataFrame(R["per_model"]).to_csv(os.path.join(a.out, "tables", "per_model.csv"), index=False)
    if cfg_rows:
        pd.DataFrame(cfg_rows).to_csv(os.path.join(a.out, "tables", "cheng_per_configuration.csv"), index=False)
    dump(R, os.path.join(a.out, "exemplar.json"))
    write_summary(R, a.out)
    print(f"done in {R['runtime_seconds']} s -> {rel(a.out)}")
    return R


# ============================================================================ macros and summary


RULE = {"constant": "Const", "own_gain": "Own", "averaged_ols": "Ols", "replicated": "Rep", "single_prompt": "Single"}


def kfmt(k):
    """Draws for an agreement target as the map prints them (scripts/run_reliability_map.py)."""
    if k is None:
        return "undefined"
    return r"$\infty$" if np.isinf(k) else f"{int(k):,}".replace(",", "{,}")


def macros(R):
    m = {}

    def put(k, v):
        if not k.isalpha():
            raise ValueError(f"macro name must be letters only: {k}")
        m[k] = str(v).replace("−", "$-$")
    t = "exh"
    put(t + "M", R["M"])
    put(t + "KDisc", R["K_discovery"])
    put(t + "KVal", R["K_validation"])
    for k, s in (("zero_shot", "MeanZero"), ("eight_shot", "MeanEight"), ("benefit", "MeanBenefit")):
        put(t + s, fmt(R["means"][k], 1))
    st = R["step2"]
    put(t + "Vdelta", fmt(st["V_delta"]["estimate"], 1))
    if st["V_delta"].get("ci95"):
        put(t + "VdeltaLo", fmt(st["V_delta"]["ci95"][0], 1))
        put(t + "VdeltaHi", fmt(st["V_delta"]["ci95"][1], 1))
    put(t + "H", fmt(st["H"]["estimate"], 2))
    put(t + "HSet", hset(st["H"]))
    put(t + "Wgain", fmt(st["noise"]["W_gain"], 2))
    if R.get("noise_split"):            # one draw's noise per arm, as noiseVarB and noiseVarP for the other settings
        sp = R["noise_split"]["split"]
        put(t + "NoiseVarB", fmt(sp["var_e_baseline"], 2))
        put(t + "NoiseVarP", fmt(sp["var_e_intervention"], 2))
        put(t + "NoiseCov", fmt(sp["cov_e"], 2))
        put(t + "NoiseCorr", fmt(sp["corr_e"], 2))
    put(t + "Rben", fmt(st["reliability"]["R_benefit"], 2))
    put(t + "Racc", fmt(st["reliability"]["R_accuracy"], 2))
    # Step 3 on the validation draws: risk of the first rule minus the second (positive: the second predicts better)
    for c in R["step3_registered"]:
        stem = t + "RD" + RULE[c["f"]] + RULE[c["g"]]
        put(stem, fmt(c["estimate"], 2))
        ci = c.get("ci95") or c.get("provisional_ci")
        if ci:
            put(stem + "Lo", fmt(ci[0], 2))
            put(stem + "Hi", fmt(ci[1], 2))
    for sc, s in (("points", "Points"), ("logit", "Logit"), ("error_reduction", "ErrRed")):
        d = R["scale"][sc]["discovery"]
        put(t + "Slope" + s, fmt(d["slope"], 3))
        ss = d["slope_set"]
        bounded = ss["kind"] == "bounded"
        put(t + "Slope" + s + "Lo", fmt(ss["a"], 3) if bounded else ss["kind"])
        put(t + "Slope" + s + "Hi", fmt(ss["b"], 3) if bounded else ss["kind"])
    mp = R["map"]["point"]
    for q, s in (("benefit", "Ben"), ("accuracy", "Acc")):
        put(t + "MapR" + s, fmt(mp[q]["R"], 2))
        put(t + "MapC" + s, fmt(mp[q]["C"], 3))
        put(t + "MapCref" + s, fmt(mp[q]["C_ref"], 3))
        for tg, ts in ((0.8, "Eighty"), (0.9, "Ninety")):
            put(t + "MapK" + ts + s, kfmt(rm.k_needed(mp[q]["R"], tg)))
    if R.get("cheng"):
        c = R["cheng"]
        put(t + "ChengModels", len(c["models"]))
        put(t + "ChengStrong", len(c["strong"]))
        for name, s in (("cheng", "Cheng"), ("strict", "Strict")):
            x = c["scorers"][name]
            put(t + "Claim" + s, x["claim_holds_in"])
            put(t + "ClaimOf" + s, x["n_configurations"])
            put(t + "SlopeOverlap" + s, fmt(x["slope_overlap"]["mean"], 3))
            put(t + "SlopeOverlapNeg" + s, x["slope_overlap"]["negative"])
            put(t + "SlopePanel" + s, fmt(x["slope_panel"]["mean"], 3))
            put(t + "SlopePanelNeg" + s, x["slope_panel"]["negative"])
            put(t + "SignAgree" + s, x["sign_agreement"]["agree"])
    for sc, s in (("strict", "Strict"), ("cheng", "Cheng")):
        x = R.get("terciles", {}).get(sc)
        if x is None:
            continue
        put(t + "TerK", x["k"])
        put(t + "TerMargin", fmt(x["margin_pp"], 0))
        for g, gt in (("weak", "Weak"), ("strong", "Strong"), ("weak_minus_strong", "Diff")):
            put(t + "Ter" + gt + s, fmt(x[g]["mean"], 2))
            put(t + "Ter" + gt + s + "Lo", fmt(x[g]["ci95"][0], 2))
            put(t + "Ter" + gt + s + "Hi", fmt(x[g]["ci95"][1], 2))
            if "ci90" in x[g]:
                put(t + "Ter" + gt + s + "LoNinety", fmt(x[g]["ci90"][0], 2))
                put(t + "Ter" + gt + s + "HiNinety", fmt(x[g]["ci90"][1], 2))
        for k, kt in (("weak_benefits", "WeakBenefits"), ("strong_equivalent", "StrongNone"), ("claim_holds", "Claim")):
            put(t + "Ter" + kt + s, "yes" if x[k] else "no")
    if R.get("gpu_check") is not None:
        put(t + "GpuSame", R["M"] - len(R["gpu_check"]["mismatched"]))
    return m


def write_summary(R, out_dir):
    st = R["step2"]
    L = ["# Eight exemplars against zero-shot chain of thought", "",
         "Cheng et al.'s eight exemplar turns before each frame of the collection's chain-of-thought "
         "requests, against the collection's zero-shot outputs, on the GSM8K-29 panel with the collection's frozen "
         "sampler. Decoding equals the collection's chain-of-thought arm (greedy, 3,072 tokens, each model's kernels, "
         "the card type of its collection); the four checkpoints with a 4,096-token context ran at 2,048 tokens, and "
         "their longer zero-shot outputs are cut to 2,048 tokens and scored again. Scores in percentage points, "
         "variances in pp². Steps 1–3 use zero-shot chain of thought as the baseline arm with the methods and margins "
         "of the main analysis; the rules were saved to `manifests/predictions_exemplar_gsm8k.json` before the "
         "validation draws were built (`exemplar_gsm8k.json`). H = −10 × the slope of benefit on zero-shot accuracy: "
         "positive means weaker models gain more.", ""]
    if R["dry_run"]:
        L += ["**Dry run: the eight-shot arm is the direct-answer arm relabelled. No number here is a result.**", ""]
    if R["partial"]:
        L += [f"**Partial run on the {R['M']} models scored so far. No number here is a result.**", ""]
    L += [f"{R['M']} models, {R['n_items']:,} items, {R['K_discovery']} discovery and {R['K_validation']} validation "
          f"draws. Mean accuracy over models: zero-shot {fmt(R['means']['zero_shot'], 1)}, eight-shot "
          f"{fmt(R['means']['eight_shot'], 1)}, benefit {fmt(R['means']['benefit'], 2)}.", "",
          f"V_delta {fmt(st['V_delta']['estimate'], 2)} {ci(st['V_delta'], 2)} pp²; H {fmt(st['H']['estimate'])} "
          f"(set {hset(st['H'])}).", ""]
    if R.get("noise_split"):
        sp = R["noise_split"]["split"]
        L += [f"One draw's prompt noise on the discovery draws: W_gain {fmt(sp['W_gain'], 2)} pp² = Var(e_P) "
              f"{fmt(sp['var_e_intervention'], 2)} (eight-shot) + Var(e_B) {fmt(sp['var_e_baseline'], 2)} (zero-shot) "
              f"− 2 × Cov {fmt(sp['cov_e'], 2)}; the two arms' errors correlate at {fmt(sp['corr_e'], 2)}.", ""]
    if R.get("zero_shot_cut_4k"):
        L += [f"Zero-shot outputs cut to 2,048 tokens: {R['zero_shot_cut_4k']['rows']} "
              "(`data/processed/exemplar/zero_shot_4k.json` has the changes by model).", ""]
    L += ["Slope of benefit on zero-shot accuracy (discovery draws):", "",
          "| Scale | Slope | Set |", "|---|---:|---|"]
    for sc in ins.SCALES:
        d = R["scale"][sc]["discovery"]
        ss = d["slope_set"]
        L.append(f"| {sc.replace('_', ' ')} | {fmt(d['slope'], 3)} | "
                 + (f"[{fmt(ss['a'], 3)}, {fmt(ss['b'], 3)}]" if ss["kind"] == "bounded" else ss["kind"].replace("_", " "))
                 + " |")
    if R.get("terciles"):
        x0 = R["terciles"]["strict"]
        L += ["", f"Terciles: the {x0['k']} models with the lowest and the {x0['k']} with the highest mean zero-shot "
              f"accuracy on the discovery draws; mean benefit over the {x0['K_validation']} validation draws with t "
              f"intervals over those draws. Margin ±{fmt(x0['margin_pp'], 0)} pp: the strong tercile has no benefit "
              "when its 90% interval lies inside the margin, the weak tercile benefits when its 95% interval lies "
              "above zero.", "",
              "| Scorer | Weak: mean [95%] | Strong: mean [95%] | Strong [90%] | Weak − strong [95%] | Weak benefits | "
              "Strong has none | Claim holds |", "|---|---|---|---|---|---|---|---|"]
        for sc, lab in (("strict", "strict (the study's reading)"), ("cheng", "Cheng et al.'s extractor")):
            x = R["terciles"].get(sc)
            if x is None:
                continue
            iv_ = lambda v, k="ci95": f"[{fmt(v[k][0], 2)}, {fmt(v[k][1], 2)}]"  # noqa: E731
            L.append(f"| {lab} | {fmt(x['weak']['mean'], 2)} {iv_(x['weak'])} | {fmt(x['strong']['mean'], 2)} "
                     f"{iv_(x['strong'])} | {iv_(x['strong'], 'ci90')} | {fmt(x['weak_minus_strong']['mean'], 2)} "
                     f"{iv_(x['weak_minus_strong'])} | {'yes' if x['weak_benefits'] else 'no'} | "
                     f"{'yes' if x['strong_equivalent'] else 'no'} | {'yes' if x['claim_holds'] else 'no'} |")
        L += ["", f"Weak tercile: {', '.join(x0['weak']['models'])}. Strong tercile: {', '.join(x0['strong']['models'])}."]
    mp = R["map"]["point"]
    L += ["", f"Map point: R benefit {fmt(mp['benefit']['R'])}, C benefit {fmt(mp['benefit']['C'], 3)} against "
              f"reference {fmt(mp['benefit']['C_ref'], 3)}; R accuracy {fmt(mp['accuracy']['R'])}, C accuracy "
              f"{fmt(mp['accuracy']['C'], 3)}.", ""]
    if R.get("cheng"):
        c = R["cheng"]
        L += ["## Cheng et al.'s claims", "",
              f"On the {len(c['models'])} panel models among the reproduction's BF16 models ({', '.join(c['models'])}; "
              f"strong: {', '.join(c['strong'])}). {c['rule'][0].upper() + c['rule'][1:]}.", "",
              "| Scorer | Claim holds | Slope, these models (mean; negative) | Slope, whole panel (mean; negative) | "
              "Mean benefit sign as published |", "|---|---:|---:|---:|---:|"]
        for name, lab in (("cheng", "corrected extractor"), ("strict", "strict scorer")):
            x = c["scorers"][name]
            L.append(f"| {lab} | {x['claim_holds_in']} of {x['n_configurations']} | {fmt(x['slope_overlap']['mean'], 3)}; "
                     f"{x['slope_overlap']['negative']} | {fmt(x['slope_panel']['mean'], 3)}; {x['slope_panel']['negative']} | "
                     f"{x['sign_agreement']['agree']} of {x['sign_agreement']['of']} |")
        L.append("")
    if R.get("gpu_check"):
        mm = R["gpu_check"]["mismatched"]
        L += [f"GPU type as in the collection: {R['M'] - len(mm)} of {R['M']} models"
              + (f" (differs: {', '.join(mm)})." if mm else "."), ""]
    L += ["| Model | Zero-shot | Eight-shot | Benefit | Eight-shot cut at the cap (%) |", "|---|---:|---:|---:|---:|"]
    for r in sorted(R["per_model"], key=lambda r: r["zero_shot"]):
        L.append(f"| {r['model_key']} | {fmt(r['zero_shot'], 1)} | {fmt(r['eight_shot'], 1)} | {fmt(r['benefit'], 2)} | "
                 f"{fmt(r['eight_shot_finish_length_pct'], 2)} |")
    L.append("")
    txt = "\n".join(L)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=None, help="default manifests/ (a dry or partial run: <out>/frozen)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partial", action="store_true", help="the models scored so far (a diagnostic)")
    ap.add_argument("--quick", action="store_true", help="999 bootstrap draws, 100 control replicates, 200 splits")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    ap.add_argument("--noise-split", action="store_true",
                    help="add one draw's prompt noise per arm to an existing <out>/exemplar.json (discovery draws only)")
    a = ap.parse_args(argv)
    if (a.dry_run or a.partial) and os.path.abspath(a.out) == os.path.abspath(OUT):
        ap.error("a dry or partial run needs --out outside results/exemplar")
    a.pred_dir = a.pred_dir or (os.path.join(a.out, "frozen") if (a.dry_run or a.partial) else MANIFESTS)
    if a.noise_split:
        add_noise_split(a)
        return
    run(a)


def add_noise_split(a):
    """Adds the noise split to <out>/exemplar.json from the discovery draws it was computed on; the macros and the
    summary are written again from the stored results, and nothing else changes."""
    path = os.path.join(a.out, "exemplar.json")
    R = json.load(open(path, encoding="utf-8"))
    inp, s, items, items_info = collection_inputs("gsm8k")
    ex_load, _, panel = exemplar_inputs(list(inp["panel"]), s, items_info, inp["load"], R["dry_run"], R["partial"])
    if list(panel) != list(R["models"]):
        raise SystemExit(f"the panel differs from {rel(path)}'s: {panel} against {R['models']}")
    load = combined_loader(inp["load"], ex_load, None if R["dry_run"] else cut_table())
    Bd, Pd = arrays(load, list(s["discovery_config_ids"]), panel, items)
    if Bd.shape[0] != R["K_discovery"]:
        raise SystemExit(f"{Bd.shape[0]} discovery draws against {R['K_discovery']} in {rel(path)}")
    R["noise_split"] = noise_block(Bd, Pd, R["step2"]["noise"])
    R["noise_split_added"] = time.strftime("%Y-%m-%d %H:%M")
    R["macros"] = macros(R)
    dump(R, path)
    write_summary(R, a.out)
    sp = R["noise_split"]["split"]
    print(f"noise split: Var(e_B) {sp['var_e_baseline']:.4f}, Var(e_P) {sp['var_e_intervention']:.4f}, "
          f"Cov {sp['cov_e']:.4f}, W_gain {sp['W_gain']:.4f} -> {rel(path)}")


if __name__ == "__main__":
    main()
