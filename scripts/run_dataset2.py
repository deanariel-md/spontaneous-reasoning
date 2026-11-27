r"""Dataset 2 (GSM8K, prospective): Steps 1-3, known-effect controls, sensitivities.

PowerShell:
    # pipeline check on synthetic item-level outputs (writes results\dataset2_sim only)
    .\.venv\Scripts\python.exe scripts\run_dataset2.py --simulate
    .\.venv\Scripts\python.exe scripts\run_dataset2.py --simulate --quick
    # analysis, once the bank, sampler, panel and scored outputs exist
    .\.venv\Scripts\python.exe scripts\run_dataset2.py --scored data\processed\gsm8k\scored.parquet
    # BIG-Bench Hard: results\dataset2\bbh.json, tables in results\dataset2\tables_bbh,
    # predictions frozen to manifests\predictions_bbh.json and _lenient.json (main_bbh below)
    .\.venv\Scripts\python.exe scripts\run_dataset2.py --task bbh --scored data\processed\bbh\scored.parquet
    # GSM8K for the 30 checkpoints of the extension: results\dataset2\gsm8k30.json, tables in
    # results\dataset2\tables_gsm8k30, predictions frozen to manifests\predictions_gsm8k30*.json (main_gsm8k30)
    .\.venv\Scripts\python.exe scripts\run_dataset2.py --panel extension

Order of operations: read the sampler; load the discovery units only; fit and freeze
every prediction rule (strict primary and lenient sensitivity) to manifests/predictions_<label>*.json;
only then load the validation units. Steps 1 and 2 use the discovery units and the calibrated
interval methods of the Dataset 2 suite.

The run refuses a table that src/d2_io.main_problems rejects (labels, test
backend, unmatched or incomplete runs, unconfirmed decoding, another sampler, other items, a changed
scorer manifest). Models that the admissibility rules excluded (manifests/gsm8k_admissibility.json)
stay in the registry and leave the panel; a model that was never screened is refused.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import controls, dataset1 as d1, dataset2 as d2, intervals as iv, sampling  # noqa: E402
from src.simulate import GSM8KSim  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else o.item() if isinstance(o, np.generic) else str(o))


def control_scenarios(theta, label):
    """The known-effect scenarios, as used for Dataset 1."""
    rng_u = sampling.rng_for(label, "residual_direction")
    return [{"name": "null (baseline population)", "kind": "null", "population": "baseline"},
            {"name": "null (intervention population)", "kind": "null", "population": "intervention"},
            {"name": "constant gain +5 pp", "kind": "constant", "gain": 5.0},
            {"name": "weaker benefit more (beta -0.3)", "kind": "capability", "gain": 5.0, "beta": -0.3},
            {"name": "stronger benefit more (beta +0.3)", "kind": "capability", "gain": 5.0, "beta": 0.3},
            {"name": "residual heterogeneity (SD 3 pp)", "kind": "residual", "gain": 5.0,
             "u": controls.residual_direction(theta, 3.0, rng_u).tolist()},
            {"name": "sparse outlier (+6 pp on one model)", "kind": "sparse_outlier", "gain": 2.0,
             "index": 0, "extra": 6.0}]


def freeze_rules(load, sampler, panel, items, pred_path, label, overwrite_frozen=False):
    """Load the discovery units only, fit the prediction rules (strict primary, lenient sensitivity) and write them to
    pred_path and its _lenient twin. Returns (frozen by value column, discovery arrays by value column, discovery
    cells). HotpotQA (scripts/run_hotpot.py --stage discovery) calls this before its validation units are scored."""
    disc = sampler["discovery_config_ids"]
    cells_d = d2.cell_table(load(disc), panel, disc, items)
    arrays_d = {v: d2.unit_arrays(cells_d, disc, panel, v) for v in ("strict_correct", "lenient_correct")}
    frozen = {}
    for v, suffix in (("strict_correct", ""), ("lenient_correct", "_lenient")):
        Bd, Pd = arrays_d[v]
        f = d2.freeze_predictions(Bd, Pd, label + suffix, panel, sampler["sha256"])
        frozen[v] = d2.write_frozen(f, pred_path.replace(".json", f"{suffix}.json"), overwrite=overwrite_frozen)
    return frozen, arrays_d, cells_d


def run(load, sampler, panel, families, tiers, items, out_dir, pred_path, label, n_boot, n_ctrl, truth=None,
        overwrite_frozen=False, provenance=None, tables_dir=None):
    t0 = time.time()
    # Checked before anything is frozen: the family sensitivities run after the validation units are read.
    if len(families) != len(panel) or not all(isinstance(f, str) and f.strip() and
                                              not f.strip().upper().startswith("PLACEHOLDER") for f in families):
        raise ValueError(f"every panel model needs a family (config/models.yaml); got {dict(zip(panel, families))}")
    sel = d2.load_selection(SELECTION)
    disc, val = sampler["discovery_config_ids"], sampler["validation_config_ids"]
    kd, kv = len(disc), len(val)
    M = len(panel)
    out = {"label": label, "units": "percentage points", "models": list(panel), "families": list(families),
           "allocation": {"K_discovery": kd, "K_validation": kv},
           "sampler_sha256": sampler["sha256"], "selection": {"path": sel["_path"], "sha256": sel["_sha256"]},
           "practical_margins": d2.PRACTICAL, "n_boot": n_boot, "scored_table": provenance}

    # ------------------------------------------------------------ discovery units only
    frozen, arrays_d, cells_d = freeze_rules(load, sampler, panel, items, pred_path, label, overwrite_frozen)
    out["frozen_predictions"] = {v: {"path": os.path.relpath(pred_path.replace(".json", s), ROOT),
                                     "sha256": frozen[v]["sha256"], "created": frozen[v]["created"]}
                                 for v, s in (("strict_correct", ".json"), ("lenient_correct", "_lenient.json"))}
    print(f"frozen predictions written ({kd} discovery units)")

    Bd, Pd = arrays_d["strict_correct"]
    m = iv.multiplicities(sampling.rng_for(label, "bootstrap"), kd, n_boot)
    out["step1"] = d2.step1(Bd, Pd, sel, m, panel)
    out["step2"] = d2.step2(Bd, Pd, sel, m)
    out["step1"]["per_model_draws"] = d2.per_model_draw_agreement(Pd - Bd) if kd >= 4 else None
    out["step1"]["per_model_draws_accuracy"] = d2.per_model_draw_agreement(Bd) if kd >= 4 else None

    # ------------------------------------------------------------ validation units (read only now)
    allu = list(disc) + list(val)
    cells = d2.cell_table(load(allu), panel, allu, items)
    Bv, Pv = d2.unit_arrays(cells, val, panel, "strict_correct")
    out["step3"] = d2.evaluate_frozen(frozen["strict_correct"], Pv - Bv, sel)
    out["step3"]["validation_agreement_curve"] = d2.validation_agreement_curve(Bv, Pv)
    out["step1"]["capability_claims"] = d2.capability_claims(Bd, Pd, Bv, Pv, sel)
    out["supplement_gaussian_reference"] = d2.gaussian_reference(Bd, Pd, sampling.budgets(kv // 2))

    # ------------------------------------------------------------ known-effect controls
    nc0, nc1 = (x[None] for x in d2.unit_arrays(cells, allu, panel, "strict_correct", counts=True))
    n_items = np.array([len(items)])
    theta = controls.pooled(nc0, n_items).mean(0)
    ctrl = []
    for sc in control_scenarios(theta, label):
        try:
            r = controls.run_control(nc0, nc1, n_items, theta, sc, kd, n_ctrl,
                                     sampling.rng_for(label, "control", sc["name"]))
        except ValueError as exc:
            r = {"scenario": sc["name"], "status": "infeasible for this panel", "detail": str(exc)}
        ctrl.append(r)
    out["controls"] = {"K": kd, "population": f"the {len(allu)} collected units (empirical population)",
                       "scenarios": ctrl, "theta_used": theta.tolist()}
    print("controls done")

    # ------------------------------------------------------------ sensitivities
    Bl, Pl = arrays_d["lenient_correct"]
    Bvl, Pvl = d2.unit_arrays(cells, val, panel, "lenient_correct")
    out["sensitivities"] = {
        "lenient_extraction": {"steps_1_2": d2.panel_summary(Bl, Pl, sel, m, "lenient extraction"),
                               "step3": d2.evaluate_frozen(frozen["lenient_correct"], Pvl - Bvl, sel)},
        "compliance": d2.compliance_summary(cells, allu, panel, tiers),
        "families": d2.family_sensitivities(Bd, Pd, families, sel, m),
        "manual_audit": "pending: sample size and protocol are investigator choices"}
    d1_sel = d2.load_selection(SELECTION, suite="dataset1")
    out["primary_family"] = {
        "dataset2_gap_test": out["step1"]["gap"]["test_zero"],
        "dataset1_gap_validated_test": d1_sel.get("gap") is not None,
        "holm": ("not applicable: the Dataset 1 gap has no validated test (manifests/calibration_selection_v2.json); "
                 "a positive difference across both dataset primaries cannot be claimed")
        if d1_sel.get("gap") is None else "apply Holm at 0.05 to the two validated gap tests"}
    if truth is not None:
        out["simulation_truth"] = truth
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    dump(out, os.path.join(out_dir, f"{label}.json"))
    write_tables(out, out_dir, tables_dir)
    print(f"done in {out['runtime_seconds']} s -> {os.path.relpath(out_dir, ROOT)}")
    return out


def write_tables(out, out_dir, tables_dir=None):
    tdir = tables_dir or os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    s1, s2 = out["step1"], out["step2"]
    rows = []
    for name in ("C_benefit", "C_accuracy"):
        x = s1[name]
        rows.append({"target": name, "estimate": x["C"], "method": x["method"],
                     "ci95_lo": None if x["ci95"] is None else x["ci95"][0],
                     "ci95_hi": None if x["ci95"] is None else x["ci95"][1],
                     "same": x["same"], "opposite": x["opposite"], "tie": x["tie"],
                     "conditional_agreement": x["conditional_agreement"]})
    g = s1["gap"]
    rows.append({"target": "gap", "estimate": g["estimate"], "method": g["method"],
                 "ci95_lo": None if g["ci95"] is None else g["ci95"][0],
                 "ci95_hi": None if g["ci95"] is None else g["ci95"][1]})
    v = s2["V_delta"]
    rows.append({"target": "V_delta", "estimate": v["estimate"], "method": v["method"],
                 "ci95_lo": None if v["ci95"] is None else v["ci95"][0],
                 "ci95_hi": None if v["ci95"] is None else v["ci95"][1],
                 "upper95": v["upper95"], "classification": v["classification"]})
    h = s2["H"]
    hs = h.get("set") or {}
    rows.append({"target": "H", "estimate": h["estimate"], "method": h["method"],
                 "ci95_lo": hs.get("a"), "ci95_hi": hs.get("b"), "set_kind": hs.get("kind"),
                 "classification": h["classification"]})
    for k in ("A", "V_theta"):
        x = s2[k]
        rows.append({"target": k, "estimate": x["estimate"], "method": x["method"],
                     "ci95_lo": None if x["ci95"] is None else x["ci95"][0],
                     "ci95_hi": None if x["ci95"] is None else x["ci95"][1]})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "steps12_registered.csv"), index=False)
    pd.DataFrame([{**r, "ci95_lo": None if r["ci95"] is None else r["ci95"][0],
                   "ci95_hi": None if r["ci95"] is None else r["ci95"][1]}
                  for r in out["step3"]["registered"] + out["step3"]["budget_curve"]]
                 ).drop(columns="ci95").to_csv(os.path.join(tdir, "step3_risk_differences.csv"), index=False)
    pd.DataFrame(out["step3"]["absolute_risk"]).to_csv(os.path.join(tdir, "step3_absolute_risk.csv"), index=False)
    ctrl = []
    for c in out["controls"]["scenarios"]:
        if "truth" not in c:
            ctrl.append({"scenario": c["scenario"], "status": c.get("status")})
            continue
        for x in ("V_delta", "A", "V_theta"):
            ctrl.append({"scenario": c["scenario"], "quantity": x, "truth": c["truth"][x], "mean": c[x]["mean"],
                         "mc_se": c[x]["mc_se"], "bias_z": c[x]["bias_z"]})
        ctrl.append({"scenario": c["scenario"], "quantity": "H_replicated", "truth": c["truth"]["H"],
                     "mean": c["H_replicated"]["mean"], "mc_se": c["H_replicated"]["mc_se"]})
    pd.DataFrame(ctrl).to_csv(os.path.join(tdir, "controls.csv"), index=False)
    pd.DataFrame(out["sensitivities"]["families"] + [out["sensitivities"]["lenient_extraction"]["steps_1_2"]]
                 ).to_csv(os.path.join(tdir, "sensitivities.csv"), index=False)
    pd.DataFrame(out["sensitivities"]["compliance"]["by_model_arm"]).to_csv(
        os.path.join(tdir, "compliance_by_model_arm.csv"), index=False)
    if out["sensitivities"]["compliance"]["by_tier_arm"]:
        pd.DataFrame(out["sensitivities"]["compliance"]["by_tier_arm"]).to_csv(
            os.path.join(tdir, "compliance_by_tier_arm.csv"), index=False)


def simulate(quick):
    label = "gsm8k_sim"
    sim = GSM8KSim(sampling.rng_for(label, "bank"), C=500 if quick else 2000, n_items=200 if quick else 1319)
    probs = np.full(sim.C, 1 / sim.C)
    k = 12
    sampler = sampling.registered_split_bank(sim.config_ids, probs, k, k, label)

    def load(ids):
        return sim.scored(ids)

    tr = sim.truth("strict_correct")
    truth = {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in tr.items()}
    out_dir = os.path.join(ROOT, "results", "dataset2_sim" + ("_quick" if quick else ""))
    out = run(load, sampler, sim.panel, sim.families, sim.tiers, list(range(sim.N)), out_dir,
              os.path.join(out_dir, "predictions_sim.json"), label,
              n_boot=999 if quick else 9999, n_ctrl=100 if quick else 1000, truth=truth,
              overwrite_frozen=True)
    check = sim_check(out, tr)
    pd.DataFrame(check).to_csv(os.path.join(out_dir, "tables", "sim_check.csv"), index=False)
    for r in check:
        print(f"  {r['target']:<12} truth {r['truth']:9.4f}  estimate {r['estimate']:9.4f}  "
              f"interval {r['interval']}  covers {r['covers']}")


def sim_check(out, tr):
    s1, s2 = out["step1"], out["step2"]
    rows = []

    def add(name, est, lo, hi, truth):
        rows.append({"target": name, "truth": float(truth), "estimate": float(est),
                     "interval": None if lo is None else f"[{lo:.4f}, {hi:.4f}]",
                     "covers": None if lo is None else bool(lo <= truth <= hi)})

    for name in ("C_benefit", "C_accuracy"):
        ci = s1[name]["ci95"] or (None, None)
        add(name, s1[name]["C"], ci[0], ci[1], tr[name])
    ci = s1["gap"]["ci95"] or (None, None)
    add("gap", s1["gap"]["estimate"], ci[0], ci[1], tr["gap"])
    ci = s2["V_delta"]["ci95"]
    add("V_delta", s2["V_delta"]["estimate"], ci[0], ci[1], tr["V_delta"])
    hs = s2["H"]["set"]
    add("H", s2["H"]["estimate"], hs["a"] if hs["kind"] == "bounded" else None, hs["b"], tr["H"])
    add("A", s2["A"]["estimate"], *(s2["A"]["ci95"] or (None, None)), tr["A"])
    rows.append({"target": "V_theta", "truth": float(tr["V_theta"]), "estimate": s2["V_theta"]["estimate"],
                 "interval": None, "covers": None})
    return rows


DEFAULTS = {"gsm8k": {"label": "gsm8k", "sampler": os.path.join(ROOT, "manifests", "sampler_gsm8k.json"),
                      "admissibility": os.path.join(ROOT, "manifests", "gsm8k_admissibility.json"),
                      "scorer": os.path.join(ROOT, "manifests", "gsm8k_scorer.json")},
            "bbh": {"label": "bbh", "sampler": os.path.join(ROOT, "manifests", "sampler_bbh.json"),
                    "admissibility": os.path.join(ROOT, "manifests", "bbh_admissibility.json"),
                    "scorer": os.path.join(ROOT, "manifests", "bbh_scorer.json")},
            "math": {"label": "math", "sampler": os.path.join(ROOT, "manifests", "sampler_math.json"),
                     "admissibility": os.path.join(ROOT, "manifests", "math_admissibility.json"),
                     "scorer": os.path.join(ROOT, "manifests", "math_scorer.json")}}


def main_bbh(a):
    """The BIG-Bench Hard analysis (the same analysis, the 25 tasks pooled
    with equal item counts). The table must be a complete BIG-Bench Hard table (d2_task bbh) on the target
    items; the panel is the 30 checkpoints of config/models.yaml and config/models_extension.yaml less the models
    the BIG-Bench Hard screening excluded; tiers are the ECI terciles of the 30 (manifests/d2_panel_extension.json).
    Predictions are frozen to manifests/predictions_<label>*.json before the validation units load; tables go to
    results/dataset2/tables_<label>/ so the GSM8K tables are never overwritten."""
    from src import bbh_d2, d2_io, string_tasks
    tm = string_tasks.module(getattr(a, "task", "bbh")) or bbh_d2    # MATH-500: the same route
    refusal = (f"refused: the scored table is not a complete {tm.TITLE} target table scored with the "
               f"scorer (scripts/score_d2.py --task {tm.TASK}): ")
    try:
        md = d2_io.scored_metadata(a.scored)
    except (OSError, ValueError) as exc:
        sys.exit(f"refused: cannot read {a.scored}: {exc}")
    problems = d2_io.main_problems(md)
    if md.get("d2_task") != tm.TASK:
        problems.append(f"d2_task is {md.get('d2_task')!r}, expected {tm.TASK!r}")
    if problems:
        sys.exit(refusal + "; ".join(problems))
    try:
        sampler = json.load(open(a.sampler))
        items_df, items_info = tm.load_items("target")
        log, log_info = d2_io.load_admissibility(a.admissibility)
        reg = tm.load_registries()
    except (OSError, ValueError, d2_io.D2IOError) as exc:
        sys.exit(f"refused: {exc}")
    if log.get("task") != tm.TASK:
        sys.exit(f"refused: {log_info['path']} is not the {tm.TITLE} exclusion log")
    if sampler.get("label") != tm.TASK:
        sys.exit(f"refused: {a.sampler} is not the {tm.TITLE} sampler (label {sampler.get('label')!r})")
    problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info, scorer_path=a.scorer,
                                           items_label=tm.TARGET_LABEL)
    if problems:
        sys.exit(refusal + "; ".join(problems))
    registered = list(reg) if not hasattr(tm, "panel") else tm.panel()   # MATH: the GSM8K-29 panel
    if set(registered) - set(reg):
        sys.exit(f"refused: models {sorted(set(registered) - set(reg))} are in no registry")
    excluded = {m["model_key"] for m in log["excluded_models"]}
    unscreened = [p for p in d2_io.panel_admissibility_problems(registered, log) if "never screened" in p]
    if unscreened:
        sys.exit("refused: " + "; ".join(unscreened))
    panel = [k for k in registered if k not in excluded]
    if not panel:
        sys.exit("refused: every model was excluded by the admissibility rules")
    families = [reg[k].get("family") for k in panel]
    tiers = tm.extension_tiers(reg, panel)
    items = items_df["item_id"].tolist()

    def load(ids):
        df = pd.read_parquet(a.scored, filters=[("config_id", "in", sorted(set(map(str, ids))))])
        df["item_id"] = df["item_id"].astype(str)
        return df

    run(load, sampler, panel, families, tiers, items, a.out,
        os.path.join(ROOT, "manifests", f"predictions_{a.label}.json"), a.label, a.n_boot, a.n_control,
        provenance={"path": os.path.relpath(a.scored, ROOT), **md,
                    "admissibility": {**log_info, "excluded_models": sorted(excluded & set(registered))},
                    "tiers": "ECI terciles of the 30-model extension panel (manifests/d2_panel_extension.json)"
                             if tiers else None},
        tables_dir=os.path.join(a.out, f"tables_{a.label}"))


def main_gsm8k30(a):
    """GSM8K for the 30 checkpoints of the extension: the same analysis as the
    GSM8K-19 run, on the 19 original checkpoints and the 11 added ones (src/d2_panels.py reads and checks
    the tables and exclusion logs). Predictions are fitted on the discovery units of all 30 and frozen to
    manifests/predictions_gsm8k30*.json before the validation units of the 11 added checkpoints load; tiers are the
    ECI terciles of the 30; the tables go to results/dataset2/tables_gsm8k30/."""
    from src import d2_io, d2_panels
    try:
        sampler = json.load(open(a.sampler))
        items_df, items_info = d2_io.load_items(split="test")
        got = d2_panels.inputs(a.scored, a.scored_extension, sampler, items_info, a.scorer,
                               main_log=a.admissibility)
    except (OSError, ValueError, d2_io.D2IOError, d2_panels.PanelError) as exc:
        sys.exit(f"refused: {exc}")
    run(got["load"], sampler, got["panel"], got["families"], got["tiers"], items_df["item_id"].tolist(), a.out,
        os.path.join(ROOT, "manifests", f"predictions_{a.label}.json"), a.label, a.n_boot, a.n_control,
        provenance=got["provenance"], tables_dir=os.path.join(a.out, f"tables_{a.label}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scored")
    ap.add_argument("--task", choices=("gsm8k", "bbh", "math"), default="gsm8k",
                    help="gsm8k (default), bbh (BIG-Bench Hard) or math (MATH-500)")
    ap.add_argument("--label", help="default gsm8k (bbh: bbh)")
    ap.add_argument("--sampler", help="default manifests/sampler_gsm8k.json (bbh: manifests/sampler_bbh.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "dataset2"))
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--admissibility",
                    help="default manifests/gsm8k_admissibility.json (bbh: manifests/bbh_admissibility.json)")
    ap.add_argument("--scorer", help="default manifests/gsm8k_scorer.json (bbh: manifests/bbh_scorer.json)")
    ap.add_argument("--panel", choices=("main", "extension"), default="main",
                    help="GSM8K only: the 19 original checkpoints (default) or the 30 of the extension "
                         "(label gsm8k30)")
    ap.add_argument("--scored-extension", nargs="+",
                    help="with --panel extension: the added checkpoints' scored tables "
                         "(default data/processed/gsm8k_ext/scored_*.parquet)")
    a = ap.parse_args()
    if a.panel == "extension" and a.task != "gsm8k":
        ap.error("--panel extension is for --task gsm8k; the BIG-Bench Hard panel is the 30 already")
    if a.panel == "extension" and a.label is None:
        a.label = "gsm8k30"
    for k, v in DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.simulate:
        simulate(a.quick)
        return
    if a.panel == "extension":
        a.scored = a.scored or os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet")
        main_gsm8k30(a)
        return
    if not a.scored:
        ap.error("--scored is required unless --simulate")
    if a.task in ("bbh", "math"):
        main_bbh(a)
        return
    from src import d2_io
    import yaml
    refusal = ("refused: the scored table is not a complete GSM8K test table scored with the "
               "scorer (scripts/score_d2.py): ")
    try:
        md = d2_io.scored_metadata(a.scored)
    except (OSError, ValueError) as exc:
        sys.exit(f"refused: cannot read {a.scored}: {exc}")
    problems = d2_io.main_problems(md)                      # the labels and folders first
    if md.get("d2_task", "gsm8k") != "gsm8k":
        problems.append(f"d2_task is {md.get('d2_task')!r}; analyse it with --task {md.get('d2_task')}")
    if problems:
        sys.exit(refusal + "; ".join(problems))
    try:
        sampler = json.load(open(a.sampler))
        items_df, items_info = d2_io.load_items(split="test")
        log, log_info = d2_io.load_admissibility(a.admissibility)
    except (OSError, ValueError, d2_io.D2IOError) as exc:
        sys.exit(f"refused: {exc}")
    problems = d2_io.main_problems(md, sampler=sampler, items_info=items_info, scorer_path=a.scorer)
    if problems:
        sys.exit(refusal + "; ".join(problems))
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models.yaml")))["dataset2"]["models"]
    registered = [k for k in reg if k != "_template"]
    if not registered:
        sys.exit("refused: no Dataset 2 model is listed in config/models.yaml")
    # Models the admissibility rules excluded stay in the registry as the record and leave the panel.
    excluded = {m["model_key"] for m in log["excluded_models"]}
    unscreened = [p for p in d2_io.panel_admissibility_problems(registered, log) if "never screened" in p]
    if unscreened:
        sys.exit("refused: " + "; ".join(unscreened))
    panel = [k for k in registered if k not in excluded]
    if not panel:
        sys.exit("refused: every model was excluded by the admissibility rules")
    families = [reg[k].get("family") for k in panel]
    tiers = {k: reg[k]["tier"] for k in panel} if all("tier" in reg[k] for k in panel) else None
    items = items_df["item_id"].tolist()

    def load(ids):
        return pd.read_parquet(a.scored, filters=[("config_id", "in", sorted(set(map(str, ids))))])

    run(load, sampler, panel, families, tiers, items, a.out,
        os.path.join(ROOT, "manifests", f"predictions_{a.label}.json"), a.label, a.n_boot, a.n_control,
        provenance={"path": os.path.relpath(a.scored, ROOT), **md,
                    "admissibility": {**log_info, "excluded_models": sorted(excluded & set(registered))}})


if __name__ == "__main__":
    main()
