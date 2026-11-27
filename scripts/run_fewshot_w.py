r"""w under five-shot answering (src/fewshot_w.py).

    .\.venv\Scripts\python.exe scripts\run_fewshot_w.py --content     # content reading of the five-shot requested replies
    .\.venv\Scripts\python.exe scripts\run_fewshot_w.py               # the analysis

Saved outputs only; nothing is generated. The five-shot requested arm (results/fewshot_requested/design.md)
covers GSM8K-29 and BIG-Bench Hard-28 on eight configurations (manifests/fewshot_requested_configs.json: four discovery,
four validation); HotpotQA is not in its design, so the 16-token cap applies to both benchmarks
(config/decoding_forced.yaml, scripts/run_baseline_mixture.cap_of).

--content reads every five-shot requested reply (data/d2_out/fewshot_requested/collection/<key>) with the registered
content reader exactly as scripts/score_content_added.py reads the five-shot forced replies (score_content.gold_table,
score_content_added.classify_rows, src/content_scoring.py unchanged), keeps the scored table's rows one to one and writes
data/processed/fewshot_w/fs_requested_content.parquet (refuses to overwrite without --replace).

The analysis reads, on the eight configurations and every target item:
  zero-shot requested  the main collection's requested-direct rows (run_forced_direct.collection_inputs: strict,
                       compliant, n_output_tokens) and data/processed/content/<task>_content.parquet (content)
  zero-shot forced     run_fewshot_requested.existing_tables (strict) and <task>_content.parquet (content)
  five-shot forced     run_fewshot_requested.existing_tables (strict) and fewshot_<task>_content.parquet (content)
  five-shot requested  data/processed/fewshot_requested/all_scored.parquet (strict, compliant, n_output_tokens) and the
                       --content file
It first recomputes the published per-model accuracies and mean gains of results/fewshot_requested/fewshot_requested.json
(strict, discovery and validation) and stops if any differs by more than 1e-9 pp. Then, per benchmark and for the eight
configurations together (the headline) and each half: w and strict-format compliance of each requested arm, the
per-model change, the five-shot effect within each baseline, the sign-change variable and its rank association with the
fall in w (src/fewshot_w.association), strict and content scoring.
Outputs: results/fewshot_w/fewshot_w.json, summary.md, summary_clean.md, tables/per_model_<task>.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from src import content_scoring as cs, d2_io, fewshot_requested as fr, fewshot_w as fw  # noqa: E402

KEY = ["model_key", "config_id", "item_id"]
TASKS = fr.TASKS
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28"}
CONTENT_DIR = ROOT / "data" / "processed" / "content"
OUT_DATA = ROOT / "data" / "processed" / "fewshot_w"
CONTENT_FILE = OUT_DATA / "fs_requested_content.parquet"
RES = ROOT / "results" / "fewshot_w"
PUBLISHED = ROOT / "results" / "fewshot_requested" / "fewshot_requested.json"
SPLITS = ("all", "discovery", "validation")
SCORES = {"strict": "strict_correct", "content": "content_correct"}
REPRO_TOL = 1e-9


def rel(p) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def _std(df, cols):
    out = df[KEY + list(cols)].copy()
    for c in KEY:
        out[c] = out[c].astype(str)
    return out


# ============================================================================ content reading of the new arm


def score_content(replace: bool = False) -> dict:
    """Content classes of the five-shot requested replies, one row per scored row."""
    import score_content as sc_mod
    import score_content_added as sca
    if CONTENT_FILE.exists() and not replace:
        raise SystemExit(f"{rel(CONTENT_FILE)} exists; pass --replace to rewrite it")
    t0 = time.time()
    paths = fr.Paths(None)
    scored = pd.read_parquet(paths.processed / "all_scored.parquet")
    scored["item_id"] = scored["item_id"].astype(str)
    gold = {t: sc_mod.gold_table(t) for t in TASKS}
    parts, counts = [], {}
    for k in sorted(scored["model_key"].unique()):
        wd = paths.outputs / "collection" / k
        out, info = d2_io.inspect_work_dir(wd, string_item_ids=True)
        if any(v for v in (info.get("prefill") or {}).values()):
            raise SystemExit(f"{wd}: ran with a prefill")
        parsed = [fr.parse_job_config_id(c) for c in out["config_id"].astype(str)]
        out = out.assign(task=[p[0] for p in parsed], config_id=[p[1] for p in parsed])
        out["item_id"] = out["item_id"].astype(str)
        mine = scored[scored["model_key"] == k]
        j = mine.merge(out[["model_key", "task", "config_id", "item_id", "output_text", "finish_reason"]],
                       on=["model_key", "task", "config_id", "item_id"], how="left", suffixes=("", "_out"),
                       validate="one_to_one")
        if j["output_text"].isna().any():
            raise SystemExit(f"{wd}: {int(j['output_text'].isna().sum())} scored rows have no output")
        if (j["finish_reason"] != j["finish_reason_out"]).any():
            raise SystemExit(f"{wd}: finish reasons differ from the scored table")
        cls = np.empty(len(j), dtype=object)
        for t in TASKS:
            m = (j["task"] == t).to_numpy()
            if not m.any():
                continue
            ids = j.loc[m, "item_id"].astype(int if t == "gsm8k" else str).tolist()
            cls[m] = sca.classify_rows(j.loc[m, "output_text"].astype(str).tolist(),
                                       j.loc[m, "finish_reason"].astype(str).tolist(), ids, gold[t], t)
        j["content_class"] = cls
        j["content_correct"] = (j["content_class"] == "correct").astype("int8")
        parts.append(j.drop(columns=["output_text", "finish_reason_out"]))
        print(f"content {k}: {len(j)} replies ({time.time() - t0:.0f} s)", flush=True)
    tab = pd.concat(parts, ignore_index=True)
    if len(tab) != len(scored):
        raise SystemExit(f"{len(tab)} classified rows against {len(scored)} scored rows")
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(CONTENT_FILE, index=False)
    for t in TASKS:
        x = tab[tab["task"] == t]
        by = x.groupby("content_class").size().reindex(list(cs.CLASSES), fill_value=0)
        counts[t] = {"n_replies": int(len(x)), **{c: int(v) for c, v in by.items()},
                     "strict_correct": int(x["strict_correct"].sum()), "content_correct": int(x["content_correct"].sum()),
                     "strict_correct_not_content_correct": int(((x["strict_correct"] == 1)
                                                                & (x["content_correct"] == 0)).sum())}
    rec = {"reader": sca.reader_version(), "content_scorer_version": cs.CONTENT_SCORER_VERSION,
           "scored_table": {"path": rel(paths.processed / "all_scored.parquet"),
                            "sha256": fr.sha256_file(paths.processed / "all_scored.parquet")},
           "out": {"path": rel(CONTENT_FILE), "sha256": fr.sha256_file(CONTENT_FILE)}, "counts": counts,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0, 1)}
    RES.mkdir(parents=True, exist_ok=True)
    fr.write_json(RES / "content_record.json", rec)
    return rec


# ============================================================================ loading


def load_task(task: str, configs: list[str], panel: list[str]) -> dict:
    """Long tables of the four arms on the configurations: KEY + strict_correct, content_correct (+ compliant and
    n_output_tokens for the requested arms). Every strict column read from the content tables is checked against the
    published path's strict column row by row."""
    import run_fewshot_requested as rfr
    from run_forced_direct import collection_inputs
    paths = fr.Paths(None)
    ex = rfr.existing_tables(task, configs)                      # the published path (strict, lenient)
    inp, _, _, _ = collection_inputs(task)
    col = inp["load"](configs)
    zs_req = _std(col[col["arm"] == "direct_answer"], ["strict_correct", "compliant", "n_output_tokens"])
    con = pd.read_parquet(CONTENT_DIR / f"{task}_content.parquet",
                          columns=KEY + ["arm", "strict_correct", "compliant", "content_correct"],
                          filters=[("config_id", "in", list(configs))])
    fsc = pd.read_parquet(CONTENT_DIR / f"fewshot_{task}_content.parquet",
                          columns=KEY + ["arm", "strict_correct", "content_correct"],
                          filters=[("config_id", "in", list(configs))])
    new = pd.read_parquet(paths.processed / "all_scored.parquet")
    new = new[new["task"] == task]
    newc = pd.read_parquet(CONTENT_FILE)
    newc = newc[newc["task"] == task]
    checks = {}

    def attach(base, content, name, extra_check=()):
        c = _std(content, ["strict_correct", "content_correct"] + list(extra_check)).rename(
            columns={"strict_correct": "strict_c", **{x: x + "_c" for x in extra_check}})
        b = _std(base, [x for x in base.columns if x not in KEY])
        b = b[b["config_id"].isin(configs) & b["model_key"].isin(panel)]
        j = b.merge(c, on=KEY, how="left", validate="one_to_one")
        if j["content_correct"].isna().any():
            raise SystemExit(f"{task} {name}: {int(j['content_correct'].isna().sum())} rows have no content reading")
        mism = int((j["strict_correct"].astype(int) != j["strict_c"].astype(int)).sum())
        checks[name] = {"rows": int(len(j)), "strict_mismatches": mism}
        for x in extra_check:
            checks[name][f"{x}_mismatches"] = int((j[x].astype(bool) != j[x + "_c"].astype(bool)).sum())
        if mism:
            raise SystemExit(f"{task} {name}: {mism} rows where the content table's strict column differs")
        return j.drop(columns=["strict_c"] + [x + "_c" for x in extra_check])

    zs_req_j = attach(zs_req, con[con["arm"] == "direct_answer"], "zs_requested", ("compliant",))
    exr = _std(ex["zs_requested"], ["strict_correct"])
    chk = zs_req_j[KEY + ["strict_correct"]].merge(exr, on=KEY, how="outer", suffixes=("", "_ex"), indicator=True)
    if (chk["_merge"] != "both").any() or (chk["strict_correct"] != chk["strict_correct_ex"]).any():
        raise SystemExit(f"{task}: the zero-shot requested rows differ from run_fewshot_requested.existing_tables")
    tabs = {"zs_requested": zs_req_j,
            "zs_forced": attach(ex["zs_forced"][KEY + ["strict_correct"]], con[con["arm"] == "forced_direct"], "zs_forced"),
            "fs_forced": attach(ex["fs_forced"][KEY + ["strict_correct"]], fsc[fsc["arm"] == "forced_direct"], "fs_forced"),
            "fs_requested": attach(new[KEY + ["strict_correct", "compliant", "n_output_tokens"]], newc, "fs_requested",
                                   ("compliant",))}
    items = set(tabs["fs_requested"]["item_id"])
    for a, t in tabs.items():
        if set(t["item_id"]) != items:
            raise SystemExit(f"{task}: the {a} arm holds another item set")
    return {"tabs": tabs, "n_items": len(items), "checks": checks}


# ============================================================================ quantities


def per_model(tabs, configs, panel, n_items, cap) -> dict:
    """Accuracy arrays (K, M) per arm and score; w and compliance (M,) of the two requested arms."""
    acc = {s: {a: fr.accuracy_array(t, configs, panel, n_items, v) for a, t in tabs.items()} for s, v in SCORES.items()}
    out = {"acc": acc}
    for a in ("zs_requested", "fs_requested"):
        t = tabs[a]
        t = t[t["config_id"].isin(configs)]
        mi = pd.Index(panel).get_indexer(t["model_key"])
        ci = pd.Index(configs).get_indexer(t["config_id"])
        if (mi < 0).any() or (ci < 0).any():
            raise SystemExit("a row outside the panel or the configurations")
        out[f"w_{a}"] = fw.long_share(t["n_output_tokens"].to_numpy(), cap, mi, ci, len(panel), len(configs))
        out[f"comp_{a}"] = fw.config_mean_share(t["compliant"].to_numpy().astype(bool), mi, ci, len(panel), len(configs))
    return out


def reproduce(task, D, V, panel, n_items, tabs) -> dict:
    """The published per-model strict accuracies (discovery, validation) and validation mean gains, recomputed."""
    R = json.loads(PUBLISHED.read_text(encoding="utf-8"))["tasks"][task]["strict"]
    pub = {r["model_key"]: r for r in R["per_model"]}
    worst, n = 0.0, 0
    for split, cfg in (("discovery", D), ("validation", V)):
        for a in fr.ARMS4:
            mine = fr.accuracy_array(tabs[a], cfg, panel, n_items, "strict_correct").mean(0)
            for m, v in zip(panel, mine):
                worst = max(worst, abs(v - pub[m][f"{a}_{split}"]))
                n += 1
    acc_v = {a: fr.accuracy_array(tabs[a], V, panel, n_items, "strict_correct") for a in fr.ARMS4}
    g = fr.gains(acc_v)
    mg = {"requested": float(g["requested"].mean(0).mean()), "forced": float(g["forced"].mean(0).mean())}
    d = max(abs(mg[k] - R["mean_gain_validation"][k]) for k in mg)
    ok = worst <= REPRO_TOL and d <= REPRO_TOL and set(pub) == set(panel)
    res = {"values_compared": n + 2, "max_abs_diff_pp": max(worst, d), "mean_gain_validation": mg,
           "published_mean_gain_validation": R["mean_gain_validation"], "reproduced": bool(ok)}
    if not ok:
        raise SystemExit(f"{task}: the published five-shot numbers do not reproduce: {res}")
    return res


def split_block(pm, panel, n_boot, rng) -> dict:
    wz, wf = pm["w_zs_requested"], pm["w_fs_requested"]
    cz, cf = pm["comp_zs_requested"], pm["comp_fs_requested"]
    for nm, v in (("w zero-shot", wz), ("w five-shot", wf), ("compliance zero-shot", cz), ("compliance five-shot", cf)):
        if not np.isfinite(v).all():
            raise SystemExit(f"{nm} is not finite for every model")
    change = wf - wz
    fall = wz - wf
    blk = {"w": {"zero_shot_mean": float(wz.mean()), "five_shot_mean": float(wf.mean()),
                 "change_mean": fw.mean_ci(change, n_boot, rng), "change_distribution": fw.distribution(change)},
           "compliance": {"zero_shot_mean": float(cz.mean()), "five_shot_mean": float(cf.mean()),
                          "change_mean": fw.mean_ci(cf - cz, n_boot, rng),
                          "change_distribution": fw.distribution(cf - cz)},
           "scores": {}}
    rows = {"model_key": panel, "w_zero_shot": wz, "w_five_shot": wf, "w_change": change,
            "compliance_zero_shot": cz, "compliance_five_shot": cf}
    for s in SCORES:
        acc = pm["acc"][s]
        g = fr.gains(acc)
        g_r, g_f = g["requested"].mean(0), g["forced"].mean(0)
        S = fw.sign_change(g_r, g_f)
        assoc = fw.association(fall, S, n_boot, rng, cont=g_r - g_f)
        blk["scores"][s] = {"mean_gain_requested": float(g_r.mean()), "mean_gain_forced": float(g_f.mean()),
                            "directions": fw.direction_counts(g_r, g_f), "association": assoc,
                            "sign_change_models": [m for m, x in zip(panel, S) if x]}
        rows.update({f"g_requested_{s}": g_r, f"g_forced_{s}": g_f, f"sign_change_{s}": S})
    blk["_rows"] = pd.DataFrame(rows)
    return blk


def run(n_boot: int) -> dict:
    import run_baseline_mixture as rbm
    import run_fewshot_requested as rfr
    t0 = time.time()
    if not CONTENT_FILE.exists():
        raise SystemExit(f"{rel(CONTENT_FILE)} is absent; run with --content first")
    paths = fr.Paths(None)
    rec, man, panels = rfr.context(paths)
    rng = np.random.default_rng(fw.SEED)
    R = {"analysis": "w under five-shot answering",
         "created": time.strftime("%Y-%m-%d %H:%M"), "units": {"w": "share", "compliance": "share", "gain": "pp"},
         "n_boot": n_boot, "seed": fw.SEED, "content_record": json.loads((RES / "content_record.json").read_text(
             encoding="utf-8")), "tasks": {}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tables").mkdir(exist_ok=True)
    for task in TASKS:
        D, V = list(man["tasks"][task]["discovery"]), list(man["tasks"][task]["validation"])
        configs = {"all": D + V, "discovery": D, "validation": V}
        panel = list(panels[task])
        cap = rbm.cap_of(task)
        L = load_task(task, D + V, panel)
        blk = {"title": TITLE[task], "cap_tokens": cap, "M": len(panel), "n_items": L["n_items"],
               "configurations": configs, "row_checks": L["checks"],
               "reproduction": reproduce(task, D, V, panel, L["n_items"], L["tabs"]), "splits": {}}
        frames = []
        for sp in SPLITS:
            pm = per_model(L["tabs"], configs[sp], panel, L["n_items"], cap)
            b = split_block(pm, panel, n_boot, rng)
            frames.append(b.pop("_rows").assign(split=sp))
            blk["splits"][sp] = b
        pd.concat(frames, ignore_index=True).to_csv(RES / "tables" / f"per_model_{task}.csv", index=False)
        blk["per_model_all"] = frames[0].drop(columns="split").to_dict(orient="records")
        R["tasks"][task] = blk
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_s"] = round(time.time() - t0, 1)
    fr.write_json(RES / "fewshot_w.json", R)
    lines = summary(R)
    for name in ("summary.md", "summary_clean.md"):
        (RES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": rel(RES / "fewshot_w.json"), "runtime_s": R["runtime_s"]}


# ============================================================================ summary


def pc(x, nd=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def f2(x, nd=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def iv(c, scale=1.0, nd=2):
    return "(undefined)" if not c else f"[{c[0] * scale:.{nd}f}, {c[1] * scale:.{nd}f}]"


def summary(R) -> list[str]:
    L = ["# w under five-shot answering", "",
         "The five-shot requested arm against the zero-shot requested arm on the same "
         "models, configurations and items (GSM8K-29 and BIG-Bench Hard-28, eight configurations, four discovery and "
         "four validation). HotpotQA is not in the five-shot design.", "",
         "Definitions. w is the share of requested replies longer than 16 generated tokens, the share within each "
         "configuration averaged over the configurations, as in the main analyses. Strict-format compliance is the "
         "share of replies the strict reader extracts an answer from. The five-shot effect within a baseline is "
         "five-shot minus zero-shot accuracy under that baseline (g_R requested, g_F forced), the model mean over the "
         "configurations. The sign-change variable S is binary: 1 when g_R and g_F have opposite signs, 0 otherwise. "
         "The fall in w is zero-shot w minus five-shot w. Rank statistics are Spearman's rho and Kendall's tau-b of "
         "the fall in w against S; intervals are percentile model-bootstrap intervals (models resampled, "
         f"{R['n_boot']:,} resamples).", "",
         "Three choices were not fixed by the request. None of them is a refit. The headline uses all eight "
         "configurations, and each half is given beside it. The five-shot requested replies had no content reading "
         "before this run; they are read here with the registered content reader "
         f"({R['content_record']['content_scorer_version']}). Beside S, the rank association of the fall in w with "
         "the continuous change g_R - g_F is given.", ""]
    for task, b in R["tasks"].items():
        rp = b["reproduction"]
        L += [f"## {b['title']}", "",
              f"Check: the published per-model strict accuracies and validation mean gains reproduce "
              f"({rp['values_compared']} values, largest difference {rp['max_abs_diff_pp']:.1e} pp; mean gain "
              f"requested {rp['mean_gain_validation']['requested']:.2f}, forced "
              f"{rp['mean_gain_validation']['forced']:.2f}).", "",
              "| Configurations | w zero-shot | w five-shot | change in w [95% CI] | models with w down | compliance "
              "zero-shot | compliance five-shot |", "|---|---|---|---|---|---|---|"]
        for sp, s in b["splits"].items():
            w, c = s["w"], s["compliance"]
            L.append(f"| {sp} | {pc(w['zero_shot_mean'])}% | {pc(w['five_shot_mean'])}% | "
                     f"{pc(w['change_mean']['mean'])} {iv(w['change_mean']['ci95'], 100, 1)} pts | "
                     f"{w['change_distribution']['n_below_zero']} of {b['M']} | {pc(c['zero_shot_mean'])}% | "
                     f"{pc(c['five_shot_mean'])}% |")
        a = b["splits"]["all"]["w"]["change_distribution"]
        L += ["", f"Per-model change in w, all eight configurations: median {pc(a['median'])} points, range "
              f"{pc(a['min'])} to {pc(a['max'])}.", "",
              "| Configurations | Scoring | mean g_R | mean g_F | S = 1 | Spearman [95% CI] | Kendall tau-b [95% CI] | "
              "mean fall in w, S = 1 minus S = 0 [95% CI] | Spearman with g_R - g_F |", "|---|---|---|---|---|---|---|---|---|"]
        for sp, s in b["splits"].items():
            for sc, x in s["scores"].items():
                A = x["association"]
                L.append(f"| {sp} | {sc} | {f2(x['mean_gain_requested'])} | {f2(x['mean_gain_forced'])} | "
                         f"{A['n_sign_change']} of {A['n_models']} | {f2(A['spearman'])} {iv(A.get('spearman_ci95'))} | "
                         f"{f2(A['kendall_tau_b'])} {iv(A.get('kendall_tau_b_ci95'))} | "
                         f"{pc(A['mean_fall_difference'])} {iv(A.get('mean_fall_difference_ci95'), 100, 1)} pts | "
                         f"{f2(A['continuous']['spearman'])} {iv(A['continuous'].get('spearman_ci95'))} |")
        und = [(sp, sc, x["association"]["boot_undefined_rank"]) for sp, s in b["splits"].items()
               for sc, x in s["scores"].items() if x["association"].get("boot_undefined_rank")]
        if und:
            L += ["", "Resamples with S constant, where the rank statistics are undefined and left out of the "
                  "interval: " + "; ".join(f"{sp} {sc} {n}" for sp, sc, n in und) + "."]
        for sc in SCORES:
            d = b["splits"]["all"]["scores"][sc]["directions"]
            L += ["", f"Directions, all eight configurations, {sc}: g_F up and g_R down in "
                  f"{d['forced_up_requested_down']} models; g_F down and g_R up in {d['forced_down_requested_up']}; "
                  f"both up in {d['both_up']}; both down in {d['both_down']}; a zero gain in {d['zero_under_either']}."]
        L += ["", "Per model, all eight configurations (w in %, gains in pp):", "",
              "| Model | w zero-shot | w five-shot | change | g_R strict | g_F strict | S strict | g_R content | "
              "g_F content | S content |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in b["per_model_all"]:
            L.append(f"| {r['model_key']} | {pc(r['w_zero_shot'])} | {pc(r['w_five_shot'])} | {pc(r['w_change'])} | "
                     f"{f2(r['g_requested_strict'])} | {f2(r['g_forced_strict'])} | {int(r['sign_change_strict'])} | "
                     f"{f2(r['g_requested_content'])} | {f2(r['g_forced_content'])} | {int(r['sign_change_content'])} |")
        L.append("")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--content", action="store_true", help="content reading of the five-shot requested replies")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--n-boot", type=int, default=fw.N_BOOT)
    a = ap.parse_args(argv)
    res = score_content(a.replace) if a.content else run(a.n_boot)
    print(json.dumps({k: v for k, v in res.items() if k != "counts"}, indent=1, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
