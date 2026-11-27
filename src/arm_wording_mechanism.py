"""The arm-wording mechanism forecast.

Inputs, read before any arm-wording output is scored: the arm-wording screen's output-token counts (data/d2_out/arm_wording/screen/;
100 requested-direct outputs per model, clause and task; no correctness read) and the main collection's scored GSM8K
and MATH-500 discovery draws (gains and token counts; results/baseline_mixture/baseline_mixture.json, key
decomposition, set discovery, strict scores). w(m, d, t) is the share of model m's requested-direct screen outputs under
direct clause d on task t longer than 16 generated tokens (the forced arm's cap in the arm-wording run and in the main collection).

Predicted gain of chain of thought over requested direct under clause d, per model and task, three forms:
    pooled         (1 - w) x g_F                    g_F: the main collection's gain over the forced arm
    short stratum  (1 - w) x g_S                    g_S: its gain over the forced arm on items whose requested output
                                                    was short (an empty short stratum counts 0)
    partial        (1 - rho x w) x g_F              rho: the task's median within-model recovery on long answers
                                                    (results/baseline_mixture/, discovery draws, strict, models with
                                                    w >= 0.1), since long answers recover only part of the
                                                    chain-of-thought gain
The predicted shares of the statements are computed under each form; their range is the prediction.

Statements (the arm-wording run's six configurations, strict scores; tolerances fixed in advance):
    S1 clause_reversal   per task, the clause-only reversal share of the requested arm (strict reversed share, mean over
                         configuration x trigger cells and the six clause pairs; src/arm_wording.reversal_table) lies
                         within TOL of the predicted range (mean over the six clause pairs of the reversed share between
                         the predicted gains under the two clauses).
    S2 clause_vs_template per task, the clause-only reversal share exceeds the configuration-only reversal share of the
                         requested arm in the same run.
    S3 arm_switch        per task and clause, the arm-switch share (strict reversed share between the gains over the
                         requested and the forced arm, mean over configuration x trigger cells) lies within TOL of the
                         predicted range (reversed share between the predicted gain over the requested arm under the
                         clause and the main collection's gain over the forced arm).
    S4 requested_rises   per task, among models whose screen w ranges over at least W_RANGE across the four clauses:
                         (a) the pooled within-model slope of requested accuracy (mean over the six configurations) on
                         the screen w, model means removed, is positive; (b) more than half of those models have a
                         positive slope.
    S5 forced_steadier   per task: (a) the median over models of the across-clause standard deviation of forced accuracy
                         (ddof 1, the four clauses, means over the six configurations) is smaller than that of requested
                         accuracy; (b) forced varies less than requested in more than half of the models.
Every statement is reported, misses included. The reference ranges (clause-only reversal 11-15% on GSM8K and 8-10% on
MATH-500; arm-switch share 0.18-0.33 and 0.15-0.24) are references, not targets.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from . import estimators as est

LONG_TOKENS = 16
TOL = 0.05
W_RANGE = 0.2
FORMS = ("pooled", "short_stratum", "partial")
TASKS = ("gsm8k", "math")
UNHASHED = ("created", "sha256", "source")
CROSS_REFERENCE = {"gsm8k": {"clause_reversal": [0.11, 0.15], "arm_switch_by_clause": [0.18, 0.33]},
                   "math": {"clause_reversal": [0.08, 0.10], "arm_switch_by_clause": [0.15, 0.24]}}
RULE_TEXT = (
    "Arm-wording mechanism forecast, written from token counts and the main collection's scored discovery draws "
    "before any arm-wording output is scored. w(m, d, t): share of model m's requested-direct outputs in the arm-wording screen under "
    "direct clause d on task t longer than 16 generated tokens (n_output_tokens > 16; 100 outputs per cell: 2 "
    "configurations x 50 screening items). Predicted gain over the requested arm under clause d: pooled (1 - w) x g_F, "
    "short-stratum (1 - w) x g_S, partial (1 - rho x w) x g_F; g_F the main collection's chain-of-thought minus "
    "forced-direct accuracy (strict, mean over its discovery draws), g_S the same on rows whose requested output had at "
    "most 16 tokens (an empty stratum counts 0), rho the task's median within-model recovery on long answers (discovery "
    "draws, strict, models with at least a tenth of their requested answers long). Reversal shares: strict (pairs tied within estimators.TIE_TOL on either side are not reversed), "
    "as src/arm_wording.reversal. Statements S1-S5 of src/arm_wording_mechanism.py with TOL = 0.05 and W_RANGE = 0.2; "
    "observed quantities on the arm-wording run's six configurations, strict scores, every trigger.")


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def reversed_share(a, b) -> float:
    """Strict reversed share between two vectors over models (ties within TIE_TOL on either side are not reversed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    sa, sb = est.tie_sign(a[i] - a[j]), est.tie_sign(b[i] - b[j])
    return float(((sa * sb) < 0).mean())


def predicted_gains(w, g_fd, g_short, rho) -> dict:
    """{form: (D, M)} predicted gain over the requested arm per clause (w: (D, M))."""
    w = np.asarray(w, float)
    g_fd = np.asarray(g_fd, float)[None]
    gs = np.nan_to_num(np.asarray(g_short, float), nan=0.0)[None]
    return {"pooled": (1 - w) * g_fd, "short_stratum": (1 - w) * gs, "partial": (1 - rho * w) * g_fd}


def predicted_shares(pred: dict, g_fd) -> dict:
    """Per form: the clause-only reversal share (mean over clause pairs) and the arm-switch share per clause."""
    out = {}
    for f, P in pred.items():
        D = P.shape[0]
        cl = [reversed_share(P[d], P[e]) for d in range(D) for e in range(D) if d < e]
        out[f] = {"clause_reversal": float(np.mean(cl)), "clause_reversal_by_pair": cl,
                  "arm_switch_by_clause": [reversed_share(P[d], g_fd) for d in range(D)]}
    return out


def _range(vals):
    v = [float(x) for x in vals]
    return [min(v), max(v)]


def forecast_record(models, clauses, inputs: dict, source: dict) -> dict:
    """inputs: {task: {'w': (D, M), 'g_fd': (M,), 'g_short': (M,), 'rho': float, 'w_collection_current': (M,)}}."""
    models, clauses = list(models), list(clauses)
    per_task = {}
    for t, x in inputs.items():
        w = np.asarray(x["w"], float)
        if w.shape != (len(clauses), len(models)):
            raise ValueError(f"{t}: w has shape {w.shape}, expected {(len(clauses), len(models))}")
        pred = predicted_gains(w, x["g_fd"], x["g_short"], float(x["rho"]))
        sh = predicted_shares(pred, x["g_fd"])
        wc = np.asarray(x["w_collection_current"], float)
        ok = np.isfinite(wc)
        corr = float(np.corrcoef(w[0][ok], wc[ok])[0, 1]) if ok.sum() > 2 and np.std(w[0][ok]) > 0 else None
        wr = w.max(0) - w.min(0)
        per_task[t] = {
            "rho": float(x["rho"]),
            "per_model": {m: {"g_forced": float(x["g_fd"][j]),
                              "g_forced_short": None if not np.isfinite(x["g_short"][j]) else float(x["g_short"][j]),
                              "w_collection_current_clause": None if not ok[j] else float(wc[j]),
                              "by_clause": {d: {"w": float(w[k, j]),
                                                **{f"predicted_gain_vs_requested_{f}": float(pred[f][k, j]) for f in FORMS}}
                                            for k, d in enumerate(clauses)}}
                          for j, m in enumerate(models)},
            "predicted": {"by_form": sh,
                          "clause_reversal_range": _range(sh[f]["clause_reversal"] for f in FORMS),
                          "arm_switch_by_clause_range": [_range(sh[f]["arm_switch_by_clause"][k] for f in FORMS)
                                                         for k in range(len(clauses))]},
            "w_summary": {"median_range_over_clauses": float(np.median(wr)),
                          "models_w_range_at_least": int((wr >= W_RANGE).sum()),
                          "models_moved_from_current_more_than": int((np.abs(w - w[0]).max(0) > W_RANGE).sum()),
                          "screen_vs_collection_current_clause_pearson": corr},
            "cross_reference": CROSS_REFERENCE.get(t)}
    body = {"label": "arm_wording_mechanism", "rule": RULE_TEXT, "units": "pp; w and shares in [0, 1]",
            "long_tokens": LONG_TOKENS, "tolerance": TOL, "w_range": W_RANGE, "forms": list(FORMS),
            "models": models, "clauses": clauses, "per_task": per_task,
            "statements": {
                "S1_clause_reversal": f"per task, the observed clause-only reversal share of the requested arm lies within "
                                      f"{TOL} of the predicted range",
                "S2_clause_vs_template": "per task, the clause-only reversal share exceeds the configuration-only "
                                         "reversal share of the requested arm",
                "S3_arm_switch": f"per task and clause, the observed arm-switch share lies within {TOL} of the "
                                 f"predicted range",
                "S4_requested_rises": f"per task, among models whose screen w ranges over at least {W_RANGE}: (a) the "
                                      "pooled within-model slope of requested accuracy on w is positive; (b) more than "
                                      "half of those models have a positive slope",
                "S5_forced_steadier": "per task: (a) the median over models of the across-clause SD of forced accuracy "
                                      "is smaller than that of requested accuracy; (b) forced varies less than "
                                      "requested in more than half of the models"},
            "source": source,
            "note": "read the arm-wording screen's token counts (no correctness) and the main collection's scored discovery "
                    "draws; written before any arm-wording output was scored."}
    body["sha256"] = sha256_json({k: v for k, v in body.items() if k not in UNHASHED})
    return body


def write_forecast(rec: dict, path, sha_path) -> dict:
    """Write-once (an identical forecast is kept, a different one refused) with the file's sha256 beside it."""
    path, sha_path = str(path), str(sha_path)
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("sha256") != rec["sha256"]:
            raise FileExistsError(f"{path} holds a different forecast; refusing to overwrite")
        rec = old
    else:
        rec = dict(rec)
        rec["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(sha_path, "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  {os.path.basename(path)}\n")
    return rec


def check_forecast(path, sha_path) -> str:
    """The file's sha256 when it exists, matches its sidecar and its own content hash; raises ValueError otherwise."""
    path, sha_path = str(path), str(sha_path)
    if not os.path.isfile(path) or not os.path.isfile(sha_path):
        raise ValueError(f"the mechanism forecast {path} (with {os.path.basename(sha_path)}) is not saved")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    want = open(sha_path, encoding="utf-8").read().split()[0]
    if got != want:
        raise ValueError(f"{path} changed after it was saved (sha256 {got} != {want})")
    rec = json.load(open(path, encoding="utf-8"))
    if sha256_json({k: v for k, v in rec.items() if k not in UNHASHED}) != rec.get("sha256"):
        raise ValueError(f"{path} does not match its recorded content hash")
    return got


# --------------------------------------------------------------------------- evaluation (after the arm-wording run is scored)


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    d = float((xc ** 2).sum())
    return None if d <= 0 else float((xc * (y - y.mean())).sum() / d)


def evaluate(rec: dict, task: str, acc: dict, reversals: dict, models, clauses, w_observed=None) -> dict:
    """One task's statements against the arm-wording run. acc: {'zero_shot_cot'|'direct_answer'|'forced_direct': (C, W, M)} accuracy in
    pp on the six configurations (src/arm_wording.TaskArrays.acc, strict); reversals: src/arm_wording.reversal_table on
    the same gains; w_observed: optional (D, M) long shares of the arm-wording run's own requested outputs (no correctness), reported
    beside the screen's."""
    if list(rec["models"]) != list(models) or list(rec["clauses"]) != list(clauses):
        raise ValueError("the forecast was saved for another panel or other clauses")
    T = rec["per_task"][task]
    tol = float(rec["tolerance"])
    D = len(clauses)
    w = np.array([[T["per_model"][m]["by_clause"][d]["w"] for m in models] for d in clauses])
    R = np.asarray(acc["direct_answer"], float).mean(0)      # (D, M)
    F = np.asarray(acc["forced_direct"], float).mean(0)
    C = np.asarray(acc["zero_shot_cot"], float).mean(0).mean(0)   # (M,): every trigger
    out = {"task": task}
    # S1, S2
    obs_cl = float(reversals["clause_switch_requested"]["reversed"])
    lo, hi = T["predicted"]["clause_reversal_range"]
    out["S1_clause_reversal"] = {"observed": obs_cl, "predicted_range": [lo, hi], "tolerance": tol,
                                 "observed_one_minus_C": float(reversals["clause_switch_requested"]["one_minus_C"]),
                                 "holds": bool(lo - tol <= obs_cl <= hi + tol)}
    obs_cfg = float(reversals["configuration_switch_requested"]["reversed"])
    out["S2_clause_vs_template"] = {"clause_only": obs_cl, "configuration_only": obs_cfg,
                                    "holds": bool(obs_cl > obs_cfg)}
    # S3
    rows = []
    for k, d in enumerate(clauses):
        o = float(reversals["arm_switch_by_clause"][k]["reversed"])
        a, b = T["predicted"]["arm_switch_by_clause_range"][k]
        rows.append({"clause": d, "observed": o, "predicted_range": [a, b],
                     "holds": bool(a - tol <= o <= b + tol)})
    out["S3_arm_switch"] = {"by_clause": rows, "tolerance": tol, "holds_all": bool(all(r["holds"] for r in rows)),
                            "holds_count": int(sum(r["holds"] for r in rows)), "clauses": D}
    # S4
    wr = w.max(0) - w.min(0)
    sel = np.flatnonzero(wr >= float(rec["w_range"]))
    slopes = {models[j]: _slope(w[:, j], R[:, j]) for j in sel}
    if len(sel):
        xc = (w[:, sel] - w[:, sel].mean(0)).ravel()
        yc = (R[:, sel] - R[:, sel].mean(0)).ravel()
        pooled = None if (xc ** 2).sum() <= 0 else float((xc * yc).sum() / (xc ** 2).sum())
    else:
        pooled = None
    pos = [s for s in slopes.values() if s is not None]
    out["S4_requested_rises"] = {
        "models": len(sel), "pooled_slope_pp_per_unit_w": pooled, "positive": int(sum(s > 0 for s in pos)),
        "per_model_slope": slopes,
        "holds_a": None if pooled is None else bool(pooled > 0),
        "holds_b": None if not pos else bool(sum(s > 0 for s in pos) > len(pos) / 2),
        "note": None if len(sel) else "no model's screen w ranges over the threshold: not testable"}
    # S5
    sdR, sdF = R.std(0, ddof=1), F.std(0, ddof=1)
    out["S5_forced_steadier"] = {"median_sd_requested_pp": float(np.median(sdR)), "median_sd_forced_pp": float(np.median(sdF)),
                                 "models_forced_less": int((sdF < sdR).sum()), "models": len(models),
                                 "holds_a": bool(np.median(sdF) < np.median(sdR)),
                                 "holds_b": bool((sdF < sdR).sum() > len(models) / 2)}
    # diagnostics: the predicted gains against the observed gains over the requested arm (every trigger, six configurations)
    g_obs = C[None] - R
    diag = {}
    for f in rec["forms"]:
        P = np.array([[T["per_model"][m]["by_clause"][d][f"predicted_gain_vs_requested_{f}"] for m in models]
                      for d in clauses])
        dP, dO = P - P.mean(0), g_obs - g_obs.mean(0)
        diag[f] = {"rms_pp": float(np.sqrt(np.mean((P - g_obs) ** 2))), "mean_error_pp": float(np.mean(P - g_obs)),
                   "pearson": float(np.corrcoef(P.ravel(), g_obs.ravel())[0, 1]),
                   "within_model_pearson": (float(np.corrcoef(dP.ravel(), dO.ravel())[0, 1])
                                            if np.std(dP) > 0 and np.std(dO) > 0 else None)}
    out["diagnostics"] = {"gain_vs_requested": diag, "observed_gain_vs_requested_by_clause_mean_pp":
                          [float(x) for x in g_obs.mean(1)]}
    if w_observed is not None:
        wo = np.asarray(w_observed, float)
        ok = np.isfinite(wo) & np.isfinite(w)
        out["diagnostics"]["w_screen_vs_d24_pearson"] = float(np.corrcoef(w[ok], wo[ok])[0, 1]) if ok.sum() > 2 else None
    return out


def summary_lines(E: dict) -> list:
    def f3(v):
        return "--" if v is None else f"{v:.3f}"
    L = ["", "## Mechanism forecast (saved before the arm-wording run was scored)", "",
         f"Saved `manifests/predictions_arm_wording_mechanism.json` (sha256 `{E['sha256'][:16]}…`, {E['created']})."]
    for t, x in E["tasks"].items():
        s1, s2, s3, s4, s5 = (x[k] for k in ("S1_clause_reversal", "S2_clause_vs_template", "S3_arm_switch",
                                             "S4_requested_rises", "S5_forced_steadier"))
        L += [f"- {t}: S1 clause-only reversal {f3(s1['observed'])}, predicted {f3(s1['predicted_range'][0])}–"
              f"{f3(s1['predicted_range'][1])} ± {s1['tolerance']}: {s1['holds']}. S2 clause {f3(s2['clause_only'])} > "
              f"configuration {f3(s2['configuration_only'])}: {s2['holds']}. S3 arm switch within range for "
              f"{s3['holds_count']} of {s3['clauses']} clauses ("
              + "; ".join(f"{r['clause']} {f3(r['observed'])} vs {f3(r['predicted_range'][0])}–{f3(r['predicted_range'][1])}"
                          for r in s3["by_clause"]) + f"). S4 ({s4['models']} models) pooled slope "
              f"{'--' if s4['pooled_slope_pp_per_unit_w'] is None else round(s4['pooled_slope_pp_per_unit_w'], 2)} pp "
              f"per unit w: {s4['holds_a']}; positive in {s4['positive']}: {s4['holds_b']}. S5 median SD forced "
              f"{s5['median_sd_forced_pp']:.2f} vs requested {s5['median_sd_requested_pp']:.2f} pp: {s5['holds_a']}; "
              f"forced steadier in {s5['models_forced_less']} of {s5['models']}: {s5['holds_b']}."]
    return L
