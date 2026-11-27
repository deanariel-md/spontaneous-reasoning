r"""Dataset 1 (DOVE, Route A, pooled MMLU block): Steps 1-3, known-effect controls, sensitivities.

PowerShell:
    .\.venv\Scripts\python.exe scripts\run_dataset1.py            # full run
    .\.venv\Scripts\python.exe scripts\run_dataset1.py --quick    # small Monte Carlo sizes, smoke test

Order of operations: build the frozen bank and the sampler, fit and
freeze every prediction rule from discovery units only to manifests/predictions_dove_mmlu.json,
and only then compute anything from validation units or the full bank.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import controls, dataset1 as d1, sampling  # noqa: E402
from src import dove_io as io  # noqa: E402
from src.estimators import gaussian_agreement, pair_agreement, reliability, unit_moments  # noqa: E402

ROOT = io.ROOT
MAN = os.path.join(ROOT, "manifests")
RES = os.path.join(ROOT, "results", "dataset1")
QUICK = "--quick" in sys.argv
N_MC = 300 if QUICK else 4000
N_CTRL = 200 if QUICK else 2000
K_DISC = K_VAL = 24
LABEL = "dove_mmlu"

PANEL = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3-8B-Instruct",
         "Mistral-7B-Instruct-v0.3", "OLMoE-1B-7B-0924-Instruct"]
SHORT = {"Llama-3.2-1B-Instruct": "Llama-3.2-1B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
         "Meta-Llama-3-8B-Instruct": "Llama-3-8B", "Mistral-7B-Instruct-v0.3": "Mistral-7B-v0.3",
         "OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B"}
FAMILY = {"Llama-3.2-1B-Instruct": "Llama", "Llama-3.2-3B-Instruct": "Llama",
          "Meta-Llama-3-8B-Instruct": "Llama", "Mistral-7B-Instruct-v0.3": "Mistral",
          "OLMoE-1B-7B-0924-Instruct": "OLMoE"}
ARMS = ("0_shot", "5_shot")


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


def load_block():
    # Common configurations and the fixed item intersection come from scripts/build_mmlu_block.py.
    bank = pd.read_csv(os.path.join(MAN, "dove_config_bank.csv"), keep_default_na=False)
    spec = json.load(open(os.path.join(MAN, "dove_mmlu_block.json")))
    block = pd.read_parquet(os.path.join(io.PROCESSED, "mmlu_block.parquet"))
    subjects, common = spec["subjects"], spec["common_config_ids"]
    assert spec["panel"] == PANEL
    excluded_cfg = sorted(set(bank["config_id"]) - set(common))
    idx = {c: k for k, c in enumerate(common)}
    S, C, M = len(subjects), len(common), len(PANEL)
    nc = {a: np.full((S, C, M), -1, dtype=np.int64) for a in ARMS}
    sub_i = {s: k for k, s in enumerate(subjects)}
    mod_i = {m: k for k, m in enumerate(PANEL)}
    for a in ARMS:
        x = block[block["arm"] == a]
        nc[a][x["subject"].map(sub_i).to_numpy(), x["config_id"].map(idx).to_numpy(),
              x["model"].map(mod_i).to_numpy()] = x["n_correct"].to_numpy()
        assert (nc[a] >= 0).all(), "missing cell in the MMLU block"
    n_items = np.array([spec["items_per_subject"][s] for s in subjects])
    return bank, subjects, common, excluded_cfg, nc, n_items


def main():
    t_start = time.time()
    bank, subjects, common, excluded_cfg, nc, n_items = load_block()
    nc0, nc1 = nc["0_shot"], nc["5_shot"]
    S, C, M = nc0.shape
    names = [SHORT[m] for m in PANEL]
    print(f"MMLU block: {S} subjects, {C} common admissible configurations, {M} models")

    # ---------------------------------------------------------------- frozen sampler
    split = sampling.registered_split(C, K_DISC, K_VAL, LABEL)
    split["bank_config_ids"] = common
    split["bank_sha256"] = sampling.sha256_json(common)
    dump(split, os.path.join(MAN, f"sampler_{LABEL}.json"))
    disc = np.array(split["discovery_bank_index"])
    val = np.array(split["validation_bank_index"])

    # ---------------------------------------------------------------- Step 3 freeze (discovery only)
    nc0_d, nc1_d = nc0[:, disc, :], nc1[:, disc, :]
    Bd, Pd = controls.pooled(nc0_d, n_items), controls.pooled(nc1_d, n_items)
    frozen = {"label": LABEL, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "sampler_sha256": sampling.sha256_json(split), "models": PANEL, "units": "pp",
              "note": "fitted on discovery units only; written before any validation outcome is read",
              "budgets": {}}
    for k in sampling.budgets(K_DISC):
        fp = d1.fit_predictors(Bd[:k], Pd[:k])
        frozen["budgets"][str(k)] = {n: {"prediction": v[0].tolist(), "info": v[1]} for n, v in fp.items()}
    frozen["sha256"] = sampling.sha256_json(frozen["budgets"])
    pred_path = os.path.join(MAN, f"predictions_{LABEL}.json")
    # An existing file with the same predictions is kept, so its creation time stays the record of the freeze;
    # different predictions are refused (as src/dataset2.write_frozen does).
    old = json.load(open(pred_path, encoding="utf-8")) if os.path.exists(pred_path) else None
    if old is not None and old.get("sha256") == frozen["sha256"] and old.get("models") == frozen["models"]:
        frozen = old
        print(f"frozen predictions unchanged since {old['created']}: {pred_path}")
    elif old is not None:
        raise SystemExit(f"{pred_path} holds different frozen predictions; refusing to overwrite")
    else:
        dump(frozen, pred_path)
        print(f"frozen predictions written: {pred_path}")
    del nc0_d, nc1_d

    # ---------------------------------------------------------------- full bank (exact conditional)
    B, P = controls.pooled(nc0, n_items), controls.pooled(nc1, n_items)
    G = P - B
    out = {"label": LABEL, "route": "A", "units": "percentage points", "models": PANEL,
           "model_short": names, "families": [FAMILY[m] for m in PANEL],
           "n_subjects": S, "n_configurations": C, "subject_weights": "equal",
           "practical_margins": d1.PRACTICAL, "monte_carlo": {"n_rep": N_MC, "n_control_rep": N_CTRL},
           "allocation": {"K_discovery": K_DISC, "K_validation": K_VAL}}

    # Step 1
    rng = sampling.rng_for(LABEL, "step1")
    s1 = {"benefit_exact": d1.exact_agreement(G), "accuracy_exact": d1.exact_agreement(B),
          "benefit_distinct_configs": pair_agreement(G), "accuracy_distinct_configs": pair_agreement(B)}
    s1["primary_gap_C_accuracy_minus_C_benefit"] = s1["accuracy_exact"]["C"] - s1["benefit_exact"]["C"]
    s1["pairwise_differences_benefit"] = d1.pairwise_differences(G, names)
    s1["pairwise_differences_accuracy"] = d1.pairwise_differences(B, names)
    for e in s1["benefit_exact"]["per_pair"] + s1["accuracy_exact"]["per_pair"]:
        e["pair"] = f"{names[e['i']]} - {names[e['j']]}"
    # Secondary: subject-level (100-item evidence), equal subject weights.
    subj = []
    for s in range(S):
        Bs, Ps = 100 * nc0[s] / n_items[s], 100 * nc1[s] / n_items[s]
        gb, ga = d1.exact_agreement(Ps - Bs), d1.exact_agreement(Bs)
        subj.append({"subject": subjects[s], "C_benefit": gb["C"], "C_accuracy": ga["C"],
                     "tie_benefit": gb["tie"], "tie_accuracy": ga["tie"]})
    s1["subject_level"] = {"per_subject": subj,
                           "mean_C_benefit": float(np.mean([x["C_benefit"] for x in subj])),
                           "mean_C_accuracy": float(np.mean([x["C_accuracy"] for x in subj])),
                           "mean_tie_benefit": float(np.mean([x["tie_benefit"] for x in subj])),
                           "mean_tie_accuracy": float(np.mean([x["tie_accuracy"] for x in subj])),
                           "label": "secondary; 100-item evidence per subject"}
    s1["agreement_curve_mc"] = d1.mc_agreement_curve(B, P, sampling.budgets(K_VAL // 2), N_MC, rng)
    out["step1"] = s1

    # Step 2
    tg = d1.exact_targets(B, P)
    tg["classification_point"] = {
        "H": d1.classify_point(tg["H"], d1.PRACTICAL["H_pp_per_10pp"]),
        "gain_sd": d1.classify_point(tg["gain_sd"], d1.PRACTICAL["gain_sd_pp"]),
        "note": "exact conditional values compared with margins; intervals for a broader prompt "
                "population await the calibration gate"}
    rng = sampling.rng_for(LABEL, "step2")
    out["step2"] = {"exact": tg,
                    "replicated_K_discovery": d1.mc_moment_distribution(B, P, K_DISC, N_MC, rng),
                    "replicated_K12": d1.mc_moment_distribution(B, P, 12, N_MC, rng)}
    # Capability-claim comparison (Step 1 capability claims): independent evaluation blocks.
    rng = sampling.rng_for(LABEL, "capability_blocks")
    out["step1"]["capability_claims"] = {str(k): d1.mc_capability_blocks(B, P, k, N_MC, rng, tg["H"])
                                         for k in (12, K_DISC)}
    # Gaussian reference (supplementary): R from the discovery units, compared with
    # the Monte Carlo curve above (held-out by construction: the curve uses the whole bank).
    Gd = Pd - Bd
    R_disc = reliability(Gd)["R"]
    R_disc_acc = reliability(Bd)["R"]
    out["supplement_gaussian_reference"] = {
        "R_benefit_discovery": R_disc, "R_accuracy_discovery": R_disc_acc,
        "R_benefit_exact": tg["R_benefit"], "R_accuracy_exact": tg["R_accuracy"],
        "curve": [{"K": k,
                   "benefit": gaussian_agreement(R_disc, k) if 0 <= R_disc <= 1 else None,
                   "accuracy": gaussian_agreement(R_disc_acc, k) if 0 <= R_disc_acc <= 1 else None}
                  for k in sampling.budgets(K_VAL // 2)],
        "note": "conditional Gaussian reference; R outside [0,1] is reported and no curve is drawn"}

    # Step 3: validation (read only now)
    Bv, Pv = B[val], P[val]
    Zv = Pv - Bv
    delta = G.mean(0)
    s3 = {"frozen_predictions_sha256": frozen["sha256"], "registered": {}, "budget_curve_registered": []}
    preds24 = {n: np.array(v["prediction"]) for n, v in frozen["budgets"][str(K_DISC)].items()}
    s3["registered"] = d1.evaluate_predictions(Zv, preds24, delta=delta)
    for k in sampling.budgets(K_DISC):
        pk = {n: np.array(v["prediction"]) for n, v in frozen["budgets"][str(k)].items()}
        ev = d1.evaluate_predictions(Zv, pk, delta=delta)
        for c in ev["comparisons"]:
            s3["budget_curve_registered"].append({"K": k, **c})
    # Agreement between two disjoint groups of K validation units (one comparison per K).
    s3["validation_agreement_registered"] = []
    for k in sampling.budgets(K_VAL // 2):
        a, b = val[:k], val[k:2 * k]
        s3["validation_agreement_registered"].append(
            {"K": k, "benefit": d1.agreement_between(G[a].mean(0), G[b].mean(0)),
             "accuracy": d1.agreement_between(B[a].mean(0), B[b].mean(0))})
    rng = sampling.rng_for(LABEL, "step3_mc")
    s3["procedure_mc"] = d1.mc_prediction(B, P, K_DISC, N_MC, rng, ks=sampling.budgets(K_DISC))
    out["step3"] = s3

    # Known-effect controls on empirical scored outcomes
    theta = B.mean(0)
    rng_u = sampling.rng_for(LABEL, "residual_direction")
    scen = [{"name": "null (baseline population)", "kind": "null", "population": "baseline"},
            {"name": "null (intervention population)", "kind": "null", "population": "intervention"},
            {"name": "constant gain +5 pp", "kind": "constant", "gain": 5.0},
            {"name": "weaker benefit more (beta -0.3)", "kind": "capability", "gain": 5.0, "beta": -0.3},
            {"name": "stronger benefit more (beta +0.3)", "kind": "capability", "gain": 5.0, "beta": 0.3},
            {"name": "residual heterogeneity (SD 3 pp)", "kind": "residual", "gain": 5.0,
             "u": controls.residual_direction(theta, 3.0, rng_u).tolist()},
            {"name": "sparse outlier (+6 pp on one model)", "kind": "sparse_outlier", "gain": 2.0,
             "index": 0, "extra": 6.0}]
    ctrl = []
    for sc in scen:
        rng = sampling.rng_for(LABEL, "control", sc["name"])
        ctrl.append(controls.run_control(nc0, nc1, n_items, theta, sc, K_DISC, N_CTRL, rng))
        print(f"control done: {sc['name']}")
    out["controls"] = {"K": K_DISC, "scenarios": ctrl, "theta_used": theta.tolist()}

    # Sensitivities
    sens = []

    def summarize(Bx, Px, label):
        t = d1.exact_targets(Bx, Px)
        ab, aa = d1.exact_agreement(Px - Bx), d1.exact_agreement(Bx)
        return {"label": label, "M": Bx.shape[1], "C_benefit": ab["C"], "C_accuracy": aa["C"],
                "gap": aa["C"] - ab["C"], "tie_benefit": ab["tie"], "V_delta": t["V_delta"],
                "gain_sd": t["gain_sd"], "H": t["H"], "mean_gain": t["mean_gain"],
                "R_benefit": t["R_benefit"]}

    for fam in sorted(set(FAMILY.values())):
        keep = [k for k, m in enumerate(PANEL) if FAMILY[m] != fam]
        sens.append(summarize(B[:, keep], P[:, keep], f"leave out family {fam}"))
    fams = sorted(set(FAMILY.values()))
    Bf = np.stack([B[:, [k for k, m in enumerate(PANEL) if FAMILY[m] == f]].mean(1) for f in fams], 1)
    Pf = np.stack([P[:, [k for k, m in enumerate(PANEL) if FAMILY[m] == f]].mean(1) for f in fams], 1)
    sens.append(summarize(Bf, Pf, "family-balanced (family-mean units)"))
    cfg = bank.set_index("config_id").loc[common]
    ws = ((cfg["enumerator"] == "capitals") & (cfg["choices_order"] == "none")).to_numpy()
    sens.append(summarize(B[ws], P[ws], f"wording/separator changes only ({int(ws.sum())} configurations)"))
    out["sensitivities"] = sens

    out["runtime_seconds"] = round(time.time() - t_start, 1)
    out["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "dove_revision": io.REVISION}
    dump(out, os.path.join(RES, f"{LABEL}{'_quick' if QUICK else ''}.json"))
    print(f"done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
