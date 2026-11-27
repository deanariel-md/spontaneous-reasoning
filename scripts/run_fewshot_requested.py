r"""Analysis of the five-shot requested-direct arm (design in
results/fewshot_requested/design.md): five-shot minus zero-shot within each answer procedure, on GSM8K-29 and BIG-Bench
Hard-28.

    .\.venv\Scripts\python.exe scripts\run_fewshot_requested.py --stage discovery   # after score --stage discovery
    .\.venv\Scripts\python.exe scripts\run_fewshot_requested.py                     # after score --stage all

Arms, on the eight drawn configurations (manifests/fewshot_requested_configs.json) and every target item:
  zs_requested  the main collection's requested-direct arm (run_fewshot.inputs: data/processed/gsm8k, gsm8k_ext, bbh)
  zs_forced     the forced-direct arm (data/processed/forced/<task>_scored.parquet)
  fs_forced     the five-shot forced arm (data/processed/fewshot/<task>_scored.parquet)
  fs_requested  this run (data/processed/fewshot_requested/<stage>_scored.parquet)
Gains g_R = fs_requested - zs_requested and g_F = fs_forced - zs_forced, per configuration and model, in points.

--stage discovery reads the discovery configurations alone and writes, once, manifests/predictions_fewshot_requested.json
and its .sha256: the one-draw choices of the top third under each procedure, the mean choices and the discovery reversal
share with its predicted range. The full run refuses unless the scored table carries that file's hash and the file is
unchanged, refits the discovery quantities from the full table's discovery rows (stops when they differ), then reports on
the validation configurations: reversed pairs between procedures with model and family bootstrap intervals, template-only
reversals within each procedure, both directions of the decision cost (src/decision_directions.both_directions, t
interval over the four validation configurations), per-model accuracies and gains; strict primary, lenient reversals
beside. Outputs: results/fewshot_requested/fewshot_requested.json, summary.md, summary_clean.md, tables/.

Dry runs: --base DIR (outside the repository) with --existing DIR holding <task>_<arm>.parquet for the three existing
arms (columns model_key, config_id, item_id, strict_correct, lenient_correct), and --n-boot.
Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import fewshot_requested as fr  # noqa: E402

TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28"}
EXISTING = ("zs_requested", "zs_forced", "fs_forced")
SEED = 20261121
VALUES = ("strict_correct", "lenient_correct")


def in_repo(base: Path) -> bool:
    try:
        Path(base).resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def families(panel) -> list[str]:
    import yaml
    fam = {}
    for name in ("models.yaml", "models_extension.yaml"):
        for k, v in yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))["dataset2"]["models"].items():
            fam.setdefault(k, v.get("family"))
    return [fam.get(k) or k.split("-")[0] for k in panel]


def _std(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["model_key", "config_id", "item_id", "strict_correct", "lenient_correct"]].copy()
    out["model_key"] = out["model_key"].astype(str)
    out["config_id"] = out["config_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    return out


def existing_tables(task: str, configs: list[str], existing_dir=None) -> dict:
    """{arm: long table} of the three existing arms on the given configurations."""
    if existing_dir:
        out = {}
        for a in EXISTING:
            df = _std(pd.read_parquet(Path(existing_dir) / f"{task}_{a}.parquet"))
            out[a] = df[df["config_id"].isin(list(configs))]
        return out
    import run_fewshot
    x = run_fewshot.inputs(task, False)
    col = x["inp"]["load"](configs)
    return {"zs_requested": _std(col[col["arm"] == "direct_answer"]), "zs_forced": _std(x["fd_load"](configs)),
            "fs_forced": _std(x["fs_load"](configs))}


def task_arrays(task, configs, panel, new: pd.DataFrame, ex: dict, value: str) -> dict:
    """{arm: (K, M) pp} of the four arms on the configurations, after checking that every arm covers one item set."""
    nt = _std(new[new["task"] == task])
    nt = nt[nt["config_id"].isin(configs)]
    items = set(nt["item_id"])
    for a, df in ex.items():
        d = df[df["config_id"].isin(configs) & df["model_key"].isin(panel)]
        if set(d["item_id"]) != items:
            raise fr.FewshotRequestedError(f"{task}: the {a} arm holds another item set than the new arm")
    n_items = len(items)
    tabs = {**ex, "fs_requested": nt}
    return {a: fr.accuracy_array(tabs[a], configs, panel, n_items, value) for a in fr.ARMS4}


def load_new(paths: fr.Paths, stage: str) -> tuple[pd.DataFrame, dict]:
    p = paths.processed / f"{stage}_scored.parquet"
    if not p.is_file():
        raise fr.FewshotRequestedError(f"{p} is absent; run score_fewshot_requested.py --stage {stage} first")
    md = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(p), md


def context(paths: fr.Paths):
    rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    man_raw = paths.configs.read_bytes()
    man = json.loads(man_raw.decode("utf-8"))
    if fr.sha256_bytes(man_raw) != rec["configs_manifest_sha256"]:
        raise fr.FewshotRequestedError("the configurations manifest changed after the requests were built")
    panels = {t: [k for k in rec["models"] if t in rec["models"][k]["tasks"]] for t in fr.TASKS}
    return rec, man, panels


def discovery_arrays(paths, new, man, panels, existing_dir, value="strict_correct"):
    acc = {}
    for t in fr.TASKS:
        D = man["tasks"][t]["discovery"]
        ex = existing_tables(t, D, existing_dir)
        acc[t] = task_arrays(t, D, panels[t], new, ex, value)
    return acc


def stage_discovery(a, paths: fr.Paths) -> dict:
    rec, man, panels = context(paths)
    new, md = load_new(paths, "discovery")
    if md.get("stage") != "discovery":
        raise fr.FewshotRequestedError("the discovery table is not a discovery-stage table")
    acc = discovery_arrays(paths, new, man, panels, a.existing)
    disc = fr.discovery_quantities(acc, panels)
    tables = {"discovery_scored": {"path": "data/processed/fewshot_requested/discovery_scored.parquet",
                                   "sha256": fr.sha256_file(paths.processed / "discovery_scored.parquet")},
              "existing": "dry run: " + str(a.existing) if a.existing else
              "run_fewshot.inputs(task): data/processed/{gsm8k,gsm8k_ext,bbh}, forced, fewshot"}
    obj = fr.predictions_record(disc, fr.sha256_file(paths.configs), tables)
    obj["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sha = fr.write_once(paths.predictions, paths.predictions_sha, obj)
    print(f"saved {paths.predictions.name} (sha256 {sha[:12]}...): discovery reversal share "
          + ", ".join(f"{t} {b['reversal']['reversed_share']:.3f}" for t, b in disc.items()))
    return {"predictions": str(paths.predictions), "sha256": sha}


def run_full(a, paths: fr.Paths) -> dict:
    t0 = time.time()
    sha = fr.check_predictions(paths)
    new, md = load_new(paths, "all")
    if md.get("predictions_sha256") != sha:
        raise fr.FewshotRequestedError("the scored table was not scored after the saved predictions (its metadata "
                                       f"carries {md.get('predictions_sha256')}, the file is {sha})")
    rec, man, panels = context(paths)
    saved = json.loads(paths.predictions.read_text(encoding="utf-8"))
    acc_d = discovery_arrays(paths, new, man, panels, a.existing)
    refit = json.loads(fr.dumps(fr.discovery_quantities(acc_d, panels)))
    if not fr.same_quantities(refit, saved["tasks"]):
        raise fr.FewshotRequestedError("the discovery quantities refitted from the full table differ from the saved file")
    rng = np.random.default_rng(SEED)
    R = {"analysis": "five-shot minus zero-shot within each answer procedure",
         "created": time.strftime("%Y-%m-%d %H:%M"), "predictions_sha256": sha, "units": "pp", "n_boot": a.n_boot,
         "configurations": {t: {"discovery": v["discovery"], "validation": v["validation"]} for t, v in man["tasks"].items()},
         "tasks": {}}
    for t in fr.TASKS:
        V = man["tasks"][t]["validation"]
        D = man["tasks"][t]["discovery"]
        ex_v = existing_tables(t, V, a.existing)
        ex_d = existing_tables(t, D, a.existing)
        fam = families(panels[t])
        blk = {"title": TITLE[t], "families": fam}
        for value in VALUES:
            av = task_arrays(t, V, panels[t], new, ex_v, value)
            ad = task_arrays(t, D, panels[t], new, ex_d, value)
            ev = fr.evaluate_task(saved["tasks"][t], ad, av, fam, a.n_boot, rng)
            if value == "lenient_correct":       # choices were made on strict scores; lenient reversals only
                ev = {k: ev[k] for k in ("reversal_discovery", "reversal_validation", "template_reversal_discovery",
                                         "template_reversal_validation", "mean_gain_validation")}
            blk["strict" if value == "strict_correct" else "lenient"] = ev
        R["tasks"][t] = blk
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_s"] = round(time.time() - t0, 1)
    write_outputs(paths, R)
    return {"out": str(paths.results / "fewshot_requested.json")}


def f2(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


def ci(d, nd=2):
    c = (d or {}).get("ci95")
    return f"[{f2(c[0], nd)}, {f2(c[1], nd)}]" if c else "(no interval)"


def summary(R) -> list[str]:
    L = ["# Five-shot gain within each answer procedure", "",
         "Gain from five demonstrations measured twice: with the answer requested (five-shot requested minus zero-shot "
         "requested) and with the answer forced (five-shot forced minus zero-shot forced). Same configurations, items and "
         "models. Choices were saved from the four discovery configurations before the validation configurations were "
         f"scored (sha256 {R['predictions_sha256'][:12]}...). Strict scoring; points.", ""]
    for t, b in R["tasks"].items():
        s = b["strict"]
        rv, rd = s["reversal_validation"], s["reversal_discovery"]
        rs = s["reversal_validation_resampling"]
        L += [f"## {b['title']}", "",
              f"- Mean gain, validation: requested {f2(s['mean_gain_validation']['requested'])}, forced "
              f"{f2(s['mean_gain_validation']['forced'])}.",
              f"- Reversed pairs between procedures, validation: {rv['reversed']} of {rv['pairs']} "
              f"({f2(100 * rv['reversed_share'], 1)}%; ties {rv['ties']}; 1 − C {f2(rv['one_minus_C'], 3)}); model "
              f"bootstrap {ci(rs['model_bootstrap'], 3)}, family bootstrap {ci(rs['family_bootstrap'], 3)}, leaving one "
              f"family out {f2(rs['leave_one_family_out_range'][0], 3)}–{f2(rs['leave_one_family_out_range'][1], 3)}. "
              f"Discovery: {rd['reversed']} of {rd['pairs']}.",
              f"- Saved statement: validation share within {f2(s['statement_share']['predicted_within'][0], 3)}–"
              f"{f2(s['statement_share']['predicted_within'][1], 3)}; observed {f2(s['statement_share']['observed'], 3)}: "
              f"{'holds' if s['statement_share']['holds'] else 'does not hold'}.",
              "- Template-only reversals (1 − C, one configuration against another), validation: requested "
              f"{f2(s['template_reversal_validation']['requested']['one_minus_C'], 3)}, forced "
              f"{f2(s['template_reversal_validation']['forced']['one_minus_C'], 3)}."]
        for name, lab in (("decision_one_draw", "one-draw choice"), ("decision_mean_choice", "four-configuration choice")):
            d = s[name]
            L.append(f"- Decision cost, {lab} (pp per chosen model, t interval over {s['K_validation']} validation "
                     f"configurations): requested → forced {f2(d['requested_to_forced']['estimate'])} "
                     f"{ci(d['requested_to_forced'])}; forced → requested {f2(d['forced_to_requested']['estimate'])} "
                     f"{ci(d['forced_to_requested'])}.")
        lv = b["lenient"]["reversal_validation"]
        L += [f"- Lenient scoring, validation reversals: {lv['reversed']} of {lv['pairs']}.", ""]
    return L


def write_outputs(paths: fr.Paths, R) -> None:
    paths.results.mkdir(parents=True, exist_ok=True)
    fr.write_json(paths.results / "fewshot_requested.json", R)
    txt = "\n".join(summary(R)) + "\n"
    (paths.results / "summary.md").write_text(txt, encoding="utf-8")
    (paths.results / "summary_clean.md").write_text(txt, encoding="utf-8")
    tdir = paths.results / "tables"
    tdir.mkdir(exist_ok=True)
    rows_c, rows_r = [], []
    for t, b in R["tasks"].items():
        s = b["strict"]
        pd.DataFrame(s["per_model"]).to_csv(tdir / f"per_model_{t}.csv", index=False)
        for name in ("decision_one_draw", "decision_mean_choice"):
            for k, d in s[name].items():
                c = d.get("ci95") or [None, None]
                rows_c.append({"task": t, "design": name, "direction": k, "estimate": d["estimate"], "ci_lo": c[0],
                               "ci_hi": c[1], "regret_mismatched": d["regret_mismatched"],
                               "regret_matched": d["regret_matched"]})
        for sc in ("strict", "lenient"):
            for split in ("discovery", "validation"):
                r = b[sc][f"reversal_{split}"]
                rows_r.append({"task": t, "scoring": sc, "split": split, **r})
    pd.DataFrame(rows_c).to_csv(tdir / "costs.csv", index=False)
    pd.DataFrame(rows_r).to_csv(tdir / "reversals.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("discovery", "all"), default="all")
    ap.add_argument("--base")
    ap.add_argument("--existing")
    ap.add_argument("--n-boot", type=int, default=9999)
    a = ap.parse_args(argv)
    paths = fr.Paths(a.base)
    try:
        if a.existing and in_repo(paths.base):
            raise fr.FewshotRequestedError("--existing is for dry runs; pass --base outside the repository")
        res = stage_discovery(a, paths) if a.stage == "discovery" else run_full(a, paths)
    except fr.FewshotRequestedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
