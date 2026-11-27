r"""Capability rules for a model outside the panel.

Run from the repository root:
    python scripts/run_heldout_capability.py

Each rule is fitted on the other models' discovery means and applied to the held-out model's own discovery baseline
accuracy and external index; hold-outs are single models and whole families. The predictions are saved to
manifests/predictions_heldout_capability.json before the validation draws are read here (an identical file is kept, a
different one refused). Each held-out model's cross-product risk on its 24 validation gains is averaged over models.
Outputs: results/heldout_capability/heldout_capability.json, summary.md, summary_clean.md, tables/risks.csv.
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
from src import dataset2 as d2, robustness as rb  # noqa: E402
from src.sampling import sha256_json  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "heldout_capability")
PRED = os.path.join(ROOT, "manifests", "predictions_heldout_capability.json")
OUTCOMES = {"CotMinusReq": "requested", "CotMinusForced": "forced"}
N_BOOT = 10000
SEED = 20261117


def predictions(X):
    """{task: {outcome: {holdout: {rule: [prediction per model]}, 'own_gain': [...]}}} from discovery draws only."""
    P = {}
    for task, x in X.items():
        f, d = x["facts"], x["discovery"]
        panel, fams, e = f["panel"], f["families"], np.array(f["eci"])
        groups = {"model": {k: [j] for j, k in enumerate(panel)},
                  "family": {fam: [j for j, y in enumerate(fams) if y == fam] for fam in sorted(set(fams))}}
        P[task] = {}
        for oc, arm in OUTCOMES.items():
            b = d[arm].mean(0)
            g = (d["cot"] - d[arm]).mean(0)
            P[task][oc] = {"own_gain": g.tolist()}
            for h, gr in groups.items():
                P[task][oc][h] = {r: rb.heldout_predictions(b, g, e, gr, r).tolist() for r in rb.RULES}
    return P


def risk_block(Zv, preds, rng, clusters=None):
    """Mean per-model cross-product risk of each rule, held-out R^2 against the constant, and the model bootstrap of
    each rule's risk minus the constant's (positive: the rule predicts better)."""
    per = {r: rb.per_model_cross_risk(Zv, np.asarray(p, float)) for r, p in preds.items()}
    out = {}
    base = per["constant"]
    M = len(base)
    for r, v in per.items():
        diff = base - v
        boot = rb.bootstrap_statistic(lambda ix, d_=diff: float(d_[ix].mean()), M, clusters, N_BOOT, rng)
        out[r] = {"risk": float(v.mean()), "risk_constant_minus_rule": float(diff.mean()),
                  "heldout_r2": float(1 - v.mean() / base.mean()) if base.mean() > 0 else None,
                  "bootstrap_constant_minus_rule": boot, "per_model": v.tolist()}
    return out


def run():
    t0 = time.time()
    X = {t: ri.load(t) for t in ri.TASKS}
    P = predictions(X)
    frozen = {"label": "heldout_capability", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "note": "fitted on discovery draws only; written before this analysis "
                      "reads the validation draws", "models": {t: X[t]["facts"]["panel"] for t in X}, "units": "pp",
              "predictions": P}
    frozen["sha256"] = sha256_json(P)
    frozen = d2.write_frozen(frozen, PRED)          # refuses a different existing file
    P = frozen["predictions"]
    rng = np.random.default_rng(SEED)
    R = {"analysis": "capability rules for a held-out model",
         "created": time.strftime("%Y-%m-%d %H:%M"), "frozen": os.path.relpath(PRED, ROOT).replace("\\", "/"),
         "frozen_sha256": frozen["sha256"], "n_boot": N_BOOT, "seed": SEED, "tasks": {}}
    rows = []
    for task, x in X.items():
        v = x["validation"]
        R["tasks"][task] = {}
        for oc, arm in OUTCOMES.items():
            Zv = v["cot"] - v[arm]
            T = {"own_gain_risk": float(rb.per_model_cross_risk(Zv, np.asarray(P[task][oc]["own_gain"])).mean()),
                 "constant_spread_V": float(np.var(Zv.mean(0), ddof=1))}
            for h in ("model", "family"):
                T[h] = risk_block(Zv, P[task][oc][h], rng, None if h == "model" else x["facts"]["families"])
                for r, b in T[h].items():
                    rows.append({"task": task, "outcome": oc, "holdout": h, "rule": r, "risk": b["risk"],
                                 "heldout_r2": b["heldout_r2"], "constant_minus_rule": b["risk_constant_minus_rule"],
                                 "boot_lo": (b["bootstrap_constant_minus_rule"]["ci95"] or [None, None])[0],
                                 "boot_hi": (b["bootstrap_constant_minus_rule"]["ci95"] or [None, None])[1],
                                 "own_gain_risk": T["own_gain_risk"]})
            R["tasks"][task][oc] = T
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "risks.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R, rows))
    print(f"done in {R['runtime_seconds']} s -> results/heldout_capability", flush=True)


def summary(R, rows):
    L = ["# Capability rules for a model outside the panel", "",
         "Rules are fitted on the other models' discovery means and "
         "predict the held-out model's mean gain from its own discovery baseline accuracy and external index. Risk: "
         "each held-out model's cross-product risk on its 24 validation gains (pp², can be negative), averaged over "
         "models. Held-out R² = 1 − risk(rule) / risk(constant). The interval is the bootstrap 95% interval (models "
         "resampled for single-model hold-outs, whole families for family hold-outs) of "
         "risk(constant) − risk(rule): above zero, the rule predicts better than the other models' mean gain. Own gain "
         "(the model's own discovery mean) is the reference for a measured model. The baseline rules need the held-out "
         "model's own direct-arm accuracy; only the index rule needs nothing from it. Predictions saved before the "
         f"validation draws were read: `{R['frozen']}`.", "",
         "| Task | Outcome | Hold-out | Rule | Risk | Held-out R² | Constant − rule | Interval | Own-gain risk |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["rule"] == "constant":
            itv = ""
        else:
            itv = f"[{rb.fmt(r['boot_lo'], 1)}, {rb.fmt(r['boot_hi'], 1)}]"
        L.append(f"| {ri.NAMES[r['task']]} | {r['outcome']} | {r['holdout']} | {r['rule']} | {rb.fmt(r['risk'], 1)} | "
                 f"{rb.fmt(r['heldout_r2'], 2)} | {rb.fmt(r['constant_minus_rule'], 1)} | {itv} | {rb.fmt(r['own_gain_risk'], 2)} |")
    L += ["", "Table: `results/heldout_capability/tables/risks.csv`."]
    return L


if __name__ == "__main__":
    run()
