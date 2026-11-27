r"""Agreement of benefit orderings at low reliability, made from Dataset 2 by item and model subsamples.

Run from the repository root:
    python scripts/run_lowr_map.py

Every map point with R below 0.6 comes from DOVE's five models. Here lower R is made from four Dataset 2 settings by
drawing items (25 to 400) and models (5, 10, all), and each replicate's R from the 24 discovery draws is set against
the agreement observed on the 24 validation draws: one draw (C) and the means of K = 2, 4 and 12 draws in disjoint
groups, beside 1/2 + arcsin(R_K)/pi. Outputs: results/lowr_map/lowr_map.json, summary.md, summary_clean.md,
tables/replicates.csv, tables/bands.csv.

The contrasts of config/added_contrasts.yaml are subsampled the same way after the four
settings, from the same generator, so the four settings' draws are those they had. They enter neither the pooled
bands ("all"), nor the counts, nor the null's pooled bands; their per-setting bands follow the others' and the summary
gives them a section of their own.
"""
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
from src import estimators as d1, robustness as rb  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "lowr_map")
SEED = 20261117
ITEM_SIZES = (25, 50, 100, 200, 400)
PANEL_SIZES = (5, 10, None)            # None: the whole panel
REPLICATES = 40
K_LIST = (2, 4, 12)
N_SPLITS = 100
BANDS = [(-np.inf, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, np.inf)]
SETTINGS = {"exemplar": "eight-shot against zero-shot chain of thought, GSM8K-29",
            "gsm8k": "GSM8K-29: chain of thought against requested direct",
            "bbh": "BIG-Bench Hard-28 pooled: chain of thought against requested direct",
            "math": "MATH-500 (29 models): chain of thought against requested direct"}


def added_settings() -> dict:
    """{pool: {name: title}} of config/added_contrasts.yaml, pool base first; empty when the file is absent."""
    import added_contrasts as ac
    if not os.path.exists(ac.CONFIG):
        return {}
    return {pool: {k: c["title"] for k, c in members.items()} for pool, members in ac.pools(ac.load_config()).items()}


def pk(key, pool):
    """Results key of an added pool: pool base keeps the plain keys, a later pool adds _<pool>."""
    return key if pool == "base" else f"{key}_{pool}"


def all_added(R) -> dict:
    """{name: title} of every added pool in the results."""
    out = {}
    for pool in R.get("added_pools", ["base"]):
        out.update(R.get(pk("added_settings", pool)) or {})
    return out


def tensor(rows, units, panel, items):
    """(draws, models, items) strict scores of one arm's rows; refuses a missing cell."""
    r = rows[["config_id", "model_key", "item_id", "strict_correct"]].copy()
    uniq = list(dict.fromkeys(str(u) for u in units))      # a configuration can be drawn more than once
    r["u"] = pd.Categorical(r["config_id"].astype(str), categories=uniq).codes
    r["m"] = pd.Categorical(r["model_key"], categories=panel).codes
    r["i"] = pd.Categorical(r["item_id"], categories=list(items)).codes
    r = r[(r["u"] >= 0) & (r["m"] >= 0) & (r["i"] >= 0)]
    r = r.drop_duplicates(["u", "m", "i"])
    T = np.full((len(uniq), len(panel), len(items)), np.nan, dtype=np.float32)
    T[r["u"].to_numpy(), r["m"].to_numpy(), r["i"].to_numpy()] = r["strict_correct"].to_numpy(np.float32)
    if np.isnan(T).any():
        raise SystemExit(f"{int(np.isnan(T).sum())} missing cells")
    return T[[uniq.index(str(u)) for u in units]]


def setting_tensors(name):
    """{'discovery': (B, P), 'validation': (B, P)} item-level tensors and the panel."""
    from run_forced_direct import collection_inputs
    if name not in SETTINGS:                     # an added contrast: its loader relabels the two arms
        import added_contrasts as ac
        x = ac.inputs(name)
        s, panel, items, load = x["sampler"], x["panel"], x["items"], x["load"]
        out = {"panel": panel, "n_items": len(items)}
        for split in ("discovery", "validation"):
            units = [str(u) for u in s[f"{split}_config_ids"]]
            rows = load(units)
            out[split] = (tensor(rows[rows["arm"] == "direct_answer"], units, panel, items),
                          tensor(rows[rows["arm"] == "zero_shot_cot"], units, panel, items))
        return out
    task = "gsm8k" if name == "exemplar" else name
    inp, s, items, items_info = collection_inputs(task)
    panel = list(inp["panel"])
    load = inp["load"]
    if name == "exemplar":
        from run_exemplar import combined_loader, cut_table, exemplar_inputs
        ex_load, _, panel = exemplar_inputs(panel, s, items_info, inp["load"], False, False)
        load = combined_loader(inp["load"], ex_load, cut_table())
    out = {"panel": panel, "n_items": len(items)}
    for split in ("discovery", "validation"):
        units = [str(u) for u in s[f"{split}_config_ids"]]
        rows = load(units)
        out[split] = (tensor(rows[rows["arm"] == "direct_answer"], units, panel, items),
                      tensor(rows[rows["arm"] == "zero_shot_cot"], units, panel, items))
    return out


def replicate(Td, Tv, models, items, rng):
    Bd, Pd = (100 * t[:, models][:, :, items].mean(2) for t in Td)
    Bv, Pv = (100 * t[:, models][:, :, items].mean(2) for t in Tv)
    Gd, Gv = Pd - Bd, Pv - Bv
    rel = d1.reliability(Gd)
    R = rel["R"]
    row = {"R_discovery": R, "C_one_draw": rb.one_draw_c(Gv), "C_ref_one_draw": rb.gaussian_reference(R, 1),
           "V_delta": rel["signal"], "W": rel["total"] - rel["signal"]}
    for K in K_LIST:
        row[f"C_K{K}"] = rb.k_draw_agreement(Gv, K, rng, N_SPLITS)
        row[f"C_ref_K{K}"] = rb.gaussian_reference(R, K)
    # the Gaussian null with this replicate's design and moments
    nl = rb.gaussian_null(rel["signal"], rel["total"] - rel["signal"], Gd.shape[0], Gv.shape[0], Gd.shape[1], rng, K_LIST)
    row.update({f"null_{k}": v for k, v in nl.items()})
    return row


def run():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    pools = added_settings()
    added = {k: v for members in pools.values() for k, v in members.items()}
    for name in list(SETTINGS) + list(added):           # the four settings first: their draws stay as they were
        x = setting_tensors(name)
        M, N = len(x["panel"]), x["n_items"]
        print(f"{name}: tensors ({time.time() - t0:.0f} s)", flush=True)
        for n in [s for s in ITEM_SIZES if s < N] + [N]:
            for m in PANEL_SIZES:
                mm = M if m is None else m
                reps = 1 if (m is None and n == N) else REPLICATES
                for r in range(reps):
                    models = np.sort(rng.choice(M, mm, replace=False))
                    items = np.sort(rng.choice(N, n, replace=False))
                    row = replicate(x["discovery"], x["validation"], models, items, rng)
                    rows.append({"setting": name, "n_items": n, "n_models": mm, "replicate": r, **row})
        print(f"{name}: replicates ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    bands = band_table(df, added)
    null = df[[c for c in df.columns if c.startswith("null_")] + ["setting"]].copy()
    null.columns = [c[5:] if c.startswith("null_") else c for c in null.columns]
    null_bands = band_table(null, added)
    main = df[df["setting"].isin(list(SETTINGS))]
    R = {"analysis": "agreement at low reliability from Dataset 2 subsamples",
         "created": time.strftime("%Y-%m-%d %H:%M"), "seed": SEED, "item_sizes": ITEM_SIZES,
         "panel_sizes": [p or "all" for p in PANEL_SIZES], "replicates": REPLICATES, "k_list": K_LIST,
         "n_splits": N_SPLITS, "settings": SETTINGS, "bands": bands.to_dict(orient="records"),
         "null_bands": null_bands.to_dict(orient="records"),
         "n_replicates": len(main), "R_outside_unit_interval": outside(main),
         "runtime_seconds": round(time.time() - t0, 1)}
    if added:
        # one block of keys per pool: pool base under the plain keys, a later pool (HotpotQA, hotpot)
        # under <key>_<pool>
        R.update({"added_config": "config/added_contrasts.yaml", "added_pools": list(pools),
                  "added_note": "kept apart: not in the pooled bands, the counts or the null's pooled bands"})
        for pool, members in pools.items():
            ad = df[df["setting"].isin(list(members))]
            R.update({pk("added_settings", pool): members, pk("n_replicates_added", pool): len(ad),
                      pk("R_outside_unit_interval_added", pool): outside(ad)})
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "tables", "replicates.csv"), index=False)
    bands.to_csv(os.path.join(OUT, "tables", "bands.csv"), index=False)
    null_bands.to_csv(os.path.join(OUT, "tables", "null_bands.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R, df, bands, null_bands))
    print(f"done in {R['runtime_seconds']} s -> results/lowr_map", flush=True)


def band_label(lo, hi):
    if lo == -np.inf:
        return "R < 0"
    if hi == np.inf:
        return f"R >= {lo:g}"
    return f"{lo:g} <= R < {hi:g}"


def outside(df) -> int:
    return int(((df["R_discovery"] < 0) | (df["R_discovery"] > 1)).sum())


def band_table(df, added=()):
    """Bands pooled over the four settings ("all"), then per setting, the added contrasts' last; "all" never holds an
    added contrast."""
    out = []
    for setting in ["all"] + list(SETTINGS) + [a for a in added if a not in SETTINGS]:
        d = df[df["setting"].isin(list(SETTINGS))] if setting == "all" else df[df["setting"] == setting]
        for lo, hi in BANDS:
            b = d[(d["R_discovery"] >= lo) & (d["R_discovery"] < hi)]
            if not len(b):
                continue
            row = {"setting": setting, "band": band_label(lo, hi), "n": len(b),
                   "median_R": float(b["R_discovery"].median()), "C_one_draw": float(b["C_one_draw"].mean())}
            ref = b["C_ref_one_draw"].dropna()
            row["C_ref_one_draw"] = float(ref.mean()) if len(ref) else None
            dep = (b["C_one_draw"] - b["C_ref_one_draw"]).dropna()
            row["departure_one_draw"] = float(dep.mean()) if len(dep) else None
            row["mean_abs_departure_one_draw"] = float(dep.abs().mean()) if len(dep) else None
            for K in K_LIST:
                o, rf = b[f"C_K{K}"].dropna(), b[f"C_ref_K{K}"].dropna()
                row[f"C_K{K}"] = float(o.mean()) if len(o) else None
                row[f"C_ref_K{K}"] = float(rf.mean()) if len(rf) else None
                both = b[[f"C_K{K}", f"C_ref_K{K}"]].dropna()
                row[f"departure_K{K}"] = float((both[f"C_K{K}"] - both[f"C_ref_K{K}"]).mean()) if len(both) else None
            out.append(row)
    return pd.DataFrame(out)


def summary(R, df, bands, null_bands):
    L = ["# Agreement at low reliability, from Dataset 2 subsamples", "",
         "Each replicate draws items (25 to 400, and all) and models (5, 10, "
         f"all) from one setting, {R['replicates']} replicates per cell. R of benefit comes from the 24 discovery draws; "
         "agreement is observed on the 24 validation draws: C over all pairs of single draws, and the agreement of the "
         f"means of K disjoint draws ({R['n_splits']} random splits each). The reference is 1/2 + arcsin(R_K)/π with "
         "R_K = K R / (1 + (K − 1) R); replicates with R outside [0, 1] have no reference and are reported as they are "
         f"({R['R_outside_unit_interval']} of {R['n_replicates']}).", "",
         "Settings: " + "; ".join(f"`{k}` {v}" for k, v in R["settings"].items()) + ".", "",
         "## By R band, all settings pooled", "",
         "| Band | Replicates | Median R | C (1 draw) | Reference | C (K = 2) | Ref. | C (K = 4) | Ref. | C (K = 12) | Ref. |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in bands[bands["setting"] == "all"].iterrows():
        L.append(f"| {r['band']} | {r['n']} | {rb.fmt(r['median_R'], 2)} | {rb.fmt(r['C_one_draw'])} | "
                 f"{rb.fmt(r['C_ref_one_draw'])} | {rb.fmt(r['C_K2'])} | {rb.fmt(r['C_ref_K2'])} | {rb.fmt(r['C_K4'])} | "
                 f"{rb.fmt(r['C_ref_K4'])} | {rb.fmt(r['C_K12'])} | {rb.fmt(r['C_ref_K12'])} |")
    nb = null_bands[null_bands["setting"] == "all"].set_index("band")
    L += ["", "## Against a Gaussian null with the same design", "",
          "For each replicate, a Gaussian null with that replicate's V_delta and W (discovery draws), its numbers of "
          "models and draws, banded by its own discovery R. Banding on a noisy R moves the low bands above the curve by "
          "itself, so the data's departure is read against the null's.", "",
          "| Band | Data: replicates | Data: C − reference | Data: mean abs. departure | Null: replicates | "
          "Null: C − reference | Data − null (K = 4) |", "|---|---|---|---|---|---|---|"]
    for _, r in bands[bands["setting"] == "all"].iterrows():
        n = nb.loc[r["band"]] if r["band"] in nb.index else None
        nd = None if n is None else n["departure_one_draw"]
        d4 = None if n is None or r["departure_K4"] is None or n["departure_K4"] is None else r["departure_K4"] - n["departure_K4"]
        L.append(f"| {r['band']} | {r['n']} | {rb.fmt(r['departure_one_draw'])} | {rb.fmt(r['mean_abs_departure_one_draw'])} | "
                 f"{'0' if n is None else int(n['n'])} | {rb.fmt(nd)} | {rb.fmt(d4)} |")
    L += ["", "## By setting and band (one draw and K = 4)", "",
          "| Setting | Band | Replicates | Median R | C (1 draw) − reference | C (K = 4) − reference |", "|---|---|---|---|---|---|"]
    added = all_added(R)
    for _, r in bands[(bands["setting"] != "all") & ~bands["setting"].isin(list(added))].iterrows():
        L.append(f"| {r['setting']} | {r['band']} | {r['n']} | {rb.fmt(r['median_R'], 2)} | "
                 f"{rb.fmt(r['departure_one_draw'])} | {rb.fmt(r['departure_K4'])} |")
    main = df[df["setting"].isin(list(SETTINGS))]
    full = main[(main["n_models"] == main.groupby("setting")["n_models"].transform("max"))]
    L += ["", "Smallest median R reached per setting: " + "; ".join(
        f"{s}: {rb.fmt(g.groupby(['n_items', 'n_models'])['R_discovery'].median().min(), 2)}" for s, g in main.groupby("setting")) + ".",
          f"Full-panel cells: {len(full)} replicates.", "",
          "Tables: `results/lowr_map/tables/replicates.csv` (every replicate), `tables/bands.csv`."]
    for pool in R.get("added_pools", ["base"]):
        L += added_lines(R, df, bands, null_bands, pool)
    return L


def added_lines(R, df, bands, null_bands, pool="base") -> list:
    """The section on the added contrasts of a pool (config/added_contrasts.yaml), kept apart from everything above."""
    import added_contrasts as ac
    added = R.get(pk("added_settings", pool)) or {}
    if not added:
        return []
    after = "the four settings" + ("" if pool == "base" else " and the contrasts of the earlier pools")
    L = ["", f"## Added contrasts ({ac.pool_info(pool)['title']}), kept apart", "",
         f"The contrasts of `{R['added_config']}`, subsampled in the same way after {after} and from the same "
         f"generator: {R[pk('n_replicates_added', pool)]} replicates, {R[pk('R_outside_unit_interval_added', pool)]} "
         "with R outside [0, 1]. They are in no pooled band or count above.", "",
         "Settings: " + "; ".join(f"`{k}` {v}" for k, v in added.items()) + ".", "",
         "| Setting | Band | Replicates | Median R | C (1 draw) − reference | C (K = 4) − reference | "
         "Null: C (1 draw) − reference |", "|---|---|---|---|---|---|---|"]
    nb = null_bands.set_index(["setting", "band"])
    for _, r in bands[bands["setting"].isin(list(added))].iterrows():
        key = (r["setting"], r["band"])
        nd = nb.loc[key, "departure_one_draw"] if key in nb.index else None
        L.append(f"| {r['setting']} | {r['band']} | {r['n']} | {rb.fmt(r['median_R'], 2)} | "
                 f"{rb.fmt(r['departure_one_draw'])} | {rb.fmt(r['departure_K4'])} | {rb.fmt(nd)} |")
    ad = df[df["setting"].isin(list(added))]
    L += ["", "Smallest median R reached per contrast: " + "; ".join(
        f"{s}: {rb.fmt(g.groupby(['n_items', 'n_models'])['R_discovery'].median().min(), 2)}"
        for s, g in ad.groupby("setting", sort=False)) + "."]
    return L


if __name__ == "__main__":
    run()
