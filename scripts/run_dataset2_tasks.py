r"""Secondary Dataset 2 analyses across tasks.

PowerShell:
    .\.venv\Scripts\python.exe scripts\run_dataset2_tasks.py

Two tasks, GSM8K and BIG-Bench Hard: the observed-task decomposition with weights of one half each, and transfer
from GSM8K to BIG-Bench Hard and back. Within BIG-Bench Hard: the decomposition over its 25 tasks (equal weights,
as its primary score pools them) and transfer to each held-out family. The estimators are in src/tasks_d2.py.

Inputs: the GSM8K-30 tables and exclusion logs (src/d2_panels.inputs), the BIG-Bench Hard table and log
(src/d2_panels.bbh_inputs) and the two frozen samplers. The panel is the models both tasks retained. Order (hard
rule 6): the discovery units of both tasks; the decompositions; every transfer prediction frozen to
manifests/predictions_tasks.json; only then the validation units. Writes results/dataset2/tasks.json and the tables
in results/dataset2/tables_tasks/. Each BIG-Bench Hard task has 54 items, stated with every task-level value.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import dataset2 as d2, tasks_d2  # noqa: E402
from src.sampling import sha256_json  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LABEL = "tasks"


def gains(df: pd.DataFrame, panel, units, items, value="strict_correct") -> np.ndarray:
    """(K, M) gains in pp of the listed units over a fixed item set."""
    cells = d2.cell_table(df[df["item_id"].isin(items)], panel, units, items)
    B, P = d2.unit_arrays(cells, units, panel, value)
    return P - B


def task_gains(df: pd.DataFrame, panel, units, items_by_task, value="strict_correct") -> dict[str, np.ndarray]:
    return {t: gains(df, panel, units, items, value) for t, items in items_by_task.items()}


def run(gsm: dict, bbh: dict, sampler_gsm: dict, sampler_bbh: dict, out_dir, pred_path, overwrite_frozen=False):
    t0 = time.time()
    panel = [k for k in gsm["panel"] if k in bbh["panel"]]
    if len(panel) < 2:
        raise ValueError(f"the two tasks share {len(panel)} retained models; the analyses need two or more")
    disc_g, val_g = sampler_gsm["discovery_config_ids"], sampler_gsm["validation_config_ids"]
    disc_b, val_b = sampler_bbh["discovery_config_ids"], sampler_bbh["validation_config_ids"]
    if len(disc_g) != len(disc_b):
        raise ValueError(f"the samplers hold {len(disc_g)} and {len(disc_b)} discovery units; bundles pair them by "
                         "position and need equal counts")
    n_items = {t: len(v) for t, v in bbh["items_by_task"].items()}
    tasks = list(bbh["items_by_task"])
    out = {"label": LABEL, "units": "percentage points", "models": panel,
           "left_out": {"gsm8k_only": [k for k in gsm["panel"] if k not in panel],
                        "bbh_only": [k for k in bbh["panel"] if k not in panel]},
           "samplers": {"gsm8k": sampler_gsm["sha256"], "bbh": sampler_bbh["sha256"]},
           "bbh_items_per_task": n_items, "inputs": {"gsm8k": gsm["provenance"], "bbh": bbh["provenance"]}}

    # ------------------------------------------------------------ discovery units only
    Gg = gains(gsm["load"](disc_g), panel, disc_g, gsm["items"])
    Gt = task_gains(bbh["load"](disc_b), panel, disc_b, bbh["items_by_task"])
    Gb = np.mean(np.stack([Gt[t] for t in tasks]), axis=0)              # the primary pooled score: equal weights
    out["two_tasks"] = {"decomposition": tasks_d2.decomposition({"gsm8k": Gg, "bbh": Gb}, tasks_d2.TWO_TASK_WEIGHTS)}
    out["two_tasks"]["decomposition"]["bundles"] = ("bundle r pairs the r-th GSM8K and the r-th BIG-Bench Hard "
                                                    "discovery unit, independent draws from two banks")
    out["within_bbh"] = {"decomposition": tasks_d2.decomposition(Gt, {t: 1 / len(tasks) for t in tasks})}
    out["within_bbh"]["decomposition"]["bundles"] = "one discovery unit (configuration) applied to all 25 tasks"
    out["within_bbh"]["families"] = {t: tasks_d2.bbh_family(t) for t in tasks}

    preds = {"gsm8k_to_bbh": tasks_d2.transfer_prediction(Gg).tolist(),
             "bbh_to_gsm8k": tasks_d2.transfer_prediction(Gb).tolist(),
             "within_bbh": tasks_d2.within_bbh_transfer_predictions(Gt)}
    frozen = {"label": LABEL, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": panel, "units": "pp",
              "samplers": out["samplers"],
              "note": "model-centred transfer predictions from discovery units only; written before any validation "
                      "outcome is read",
              "predictions": preds}
    frozen["sha256"] = sha256_json(json.loads(json.dumps(preds)))
    frozen = d2.write_frozen(frozen, pred_path, overwrite=overwrite_frozen)
    out["frozen_predictions"] = {"path": os.path.relpath(pred_path, ROOT), "sha256": frozen["sha256"],
                                 "created": frozen["created"]}
    print(f"transfer predictions frozen ({len(disc_g)} discovery units per task)")

    # ------------------------------------------------------------ validation units (read only now)
    Zg = gains(gsm["load"](val_g), panel, val_g, gsm["items"])
    Zt = task_gains(bbh["load"](val_b), panel, val_b, bbh["items_by_task"])
    Zb = np.mean(np.stack([Zt[t] for t in tasks]), axis=0)
    p = frozen["predictions"]
    out["two_tasks"]["transfer"] = {
        "gsm8k_to_bbh": tasks_d2.transfer_score(np.asarray(p["gsm8k_to_bbh"]), Zb),
        "bbh_to_gsm8k": tasks_d2.transfer_score(np.asarray(p["bbh_to_gsm8k"]), Zg),
        "note": "observed cases (two tasks); the target's absolute mean benefit is not predicted"}
    rows = []
    for fam, f in p["within_bbh"].items():
        r = tasks_d2.transfer_score(np.asarray(f["prediction"]), tasks_d2.held_out_target(Zt, f["held_out_tasks"]))
        rows.append({"held_out_family": fam, "held_out_tasks": f["held_out_tasks"],
                     "items_per_task": [n_items[t] for t in f["held_out_tasks"]],
                     "n_train_tasks": len(f["train_tasks"]), **r})
    out["within_bbh"]["transfer"] = rows
    out["within_bbh"]["transfer_summary"] = {
        "n_families": len(rows), "positive_difference": sum(r["difference"]["estimate"] > 0 for r in rows),
        "negative_difference": sum(r["difference"]["estimate"] < 0 for r in rows),
        "tie": sum(r["difference"]["estimate"] == 0 for r in rows),
        "note": "held-out families are observed cases, not a sample of future tasks"}
    out["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{LABEL}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else o.item() if isinstance(o, np.generic) else str(o))
    write_tables(out, os.path.join(out_dir, f"tables_{LABEL}"))
    print(f"done in {out['runtime_seconds']} s -> {os.path.relpath(out_dir, ROOT)}")
    return out


def write_tables(out: dict, tdir: str) -> None:
    os.makedirs(tdir, exist_ok=True)
    rows = []
    for scope in ("two_tasks", "within_bbh"):
        dec = out[scope]["decomposition"]
        for k in ("average_task_profile", "model_by_task", "true_within_task", "prompt_noise", "total"):
            x = dec[k]
            rows.append({"scope": scope, "component": k, "estimate": x["estimate"],
                         "jackknife_se": x["jackknife_se"], "ci95_lo": None if x["ci95"] is None else x["ci95"][0],
                         "ci95_hi": None if x["ci95"] is None else x["ci95"][1], "K_bundles": dec["K_bundles"]})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "decomposition.csv"), index=False)
    tr = []
    for name in ("gsm8k_to_bbh", "bbh_to_gsm8k"):
        r = out["two_tasks"]["transfer"][name]
        tr.append({"scope": "two_tasks", "case": name, **_row(r)})
    for r in out["within_bbh"]["transfer"]:
        tr.append({"scope": "within_bbh", "case": r["held_out_family"], "n_tasks": len(r["held_out_tasks"]),
                   "items_per_task": r["items_per_task"][0], **_row(r)})
    pd.DataFrame(tr).to_csv(os.path.join(tdir, "transfer.csv"), index=False)
    pd.DataFrame(out["within_bbh"]["decomposition"]["per_task"]).assign(
        items=lambda d: d["task"].map(out["bbh_items_per_task"])).to_csv(os.path.join(tdir, "bbh_per_task.csv"),
                                                                          index=False)


def _row(r: dict) -> dict:
    d = r["difference"]
    return {"difference": d["estimate"], "jackknife_se": d["jackknife_se"],
            "ci95_lo": None if d["ci95"] is None else d["ci95"][0], "ci95_hi": None if d["ci95"] is None else d["ci95"][1],
            "risk_zero": r["risk_zero"], "risk_transfer": r["risk_transfer"], "K_validation": r["K_validation"]}


def main(argv=None):
    from src import d2_io, d2_panels
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gsm8k-scored", help="default data/processed/gsm8k/scored.parquet")
    ap.add_argument("--gsm8k-extension", nargs="+", help="default data/processed/gsm8k_ext/scored_*.parquet")
    ap.add_argument("--bbh-scored", help="default data/processed/bbh/scored.parquet")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "dataset2"))
    a = ap.parse_args(argv)
    try:
        s_g = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json"), encoding="utf-8"))
        s_b = json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
        items_g, items_info = d2_io.load_items(split="test")
        gsm = d2_panels.inputs(a.gsm8k_scored, a.gsm8k_extension, s_g, items_info,
                               os.path.join(ROOT, "manifests", "gsm8k_scorer.json"))
        gsm["items"] = items_g["item_id"].tolist()
        bbh = d2_panels.bbh_inputs(a.bbh_scored, s_b)
    except (OSError, ValueError, d2_io.D2IOError, d2_panels.PanelError) as exc:
        sys.exit(f"refused: {exc}")
    run(gsm, bbh, s_g, s_b, a.out, os.path.join(ROOT, "manifests", f"predictions_{LABEL}.json"))


if __name__ == "__main__":
    main()
