r"""Default and firm request clauses on the six configurations of the arm-wording run.

Two direct clauses are added to the base run's four, on the same six configurations (manifests/arm_wording_configs.json), the
same items (GSM8K's 1,319 test items, MATH-500's 398 numeric target items), the same decoding and cards, for the same
29 models. Each clause gives a requested-direct arm and its forced arm (the requested user turn byte for byte, then the
assistant turn opened with "\boxed{", 16 new tokens). No chain-of-thought request is generated: the analysis takes
the base run's chain-of-thought outputs (four triggers) from its scored table, read only once the base run has been
scored.

    dirdef   the default request: no instruction about working. The current clause and the single space after it are
             removed from the requested-direct messages; the answer-format sentence that follows stays.
    dirfirm  the firm request: "Give only the final answer. Do not show any working under any circumstances."

Rule for the two texts (fixed before any request of this run was built and before any output under
either clause existed): the default clause is the absence of the clause, which is what a practitioner's prompt and
Meincke et al.'s default condition carry. The firm clause is the current clause's request for the answer, "Give only
the final answer" (its text up to the comma, closed with a period), followed by the firmer sentence that the base run fixed
before any of its outputs existed, as its example of a firmer instruction: "Do not show any working under
any circumstances" (closed with a period). The firm clause differs from the current clause only in how firmly working
is forbidden; the base run's sentence alone would also drop the request for the answer alone.

Order:
  1. a pilot shaped like the base run's screen (the screen's 50 screening items per task and its two configurations, requested
     and forced arms under the two new clauses: 800 requests per model) runs on every model's card; it is each model's
     run check and gives w for the two clauses from output-token counts alone;
  2. manifests/predictions_arm_wording_extra.json and .sha256 are written from those counts, the mechanism forecast
     (w under the base run's four clauses, the forced gains g_F and g_S and rho) and nothing else (write-once);
  3. only then are the collection jobs released; scoring refuses both stages unless the forecast exists and matches;
  4. the discovery stage saves manifests/predictions_arm_wording_extra_discovery.json (the base run's discovery quantities
     with six clauses) before any validation configuration of this run is scored;
  5. the full run evaluates every statement, misses included, and repeats the base run's analyses 1 and 2 with six clauses.

Statements (strict scores, the six configurations of the base run; fixed in advance):
  A1 forced_meets_requested  per task, among models whose pilot w under the firm clause is at most W_NEAR0: (a) the mean
                             over them of requested minus forced accuracy under the firm clause lies within GAP_TOL_PP
                             of 0; (b) the gap lies within GAP_TOL_PP of 0 in more than half of them.
  A2 default_more_working    per task: (a) the median over models of w(default) - w(current), both measured on the
                             collections' own requested outputs, is positive; (b) positive in more than half of models.
  A3 firm_less_working       per task: (a) the median of w(firm) - w(current) is negative; (b) negative in more than
                             half of models.
  A4 pilot_predicts_w        per task: the pilot's w and the collection's w under the two new clauses correlate at
                             least W_CORR over the models x 2 clauses.
  A5 extrapolates_to_forced  per task, among models whose w ranges over at least W_RANGE across the six clauses: with
                             the pooled within-model slope b of requested minus forced accuracy on w, each model's
                             intercept a_m = mean(R - F) - b mean(w); the median a_m lies within GAP_TOL_PP of 0.
  S1-S5                      the mechanism statements (src/arm_wording_mechanism.py) over the six clauses: clause
                             reversal within TOL of the predicted range (S1), clause above configuration reversal (S2),
                             each clause's arm-switch share within TOL of its predicted range (S3), requested accuracy
                             rising with w within model (S4), forced accuracy steadier than requested (S5).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import arm_wording as aw, arm_wording_mechanism as awm, d2_io

ROOT = Path(__file__).resolve().parent.parent
CURRENT = "Give only the final answer, without any working."
FIRM_SENTENCE_D24 = "Do not show any working under any circumstances"
FIRM = "Give only the final answer. " + FIRM_SENTENCE_D24 + "."
NEW = ("dirdef", "dirfirm")
D24_CLAUSES = ("dir0", "dir1", "dir2", "dir3")
CLAUSES6 = D24_CLAUSES + NEW
ARMS = (aw.REQ, aw.FD)
PILOT_PREFIX = "awxp|"
W_NEAR0 = 0.05
GAP_TOL_PP = 2.0
W_CORR = 0.9
TOL = awm.TOL
W_RANGE = awm.W_RANGE
TEXTS = {"dirdef": "", "dirfirm": FIRM}
RULE_TEXT = (
    "Default and firm clauses, forecast written from token counts before any output of the run is scored. "
    "w(m, d, t): share of model m's requested-direct outputs longer than 16 generated tokens; for the base run's four clauses "
    "from the mechanism forecast (base-run screen), for the default and firm clauses from this run's pilot (the base run's "
    "screen's items and two configurations, 100 requested outputs per model, clause and task). Predicted gain over the "
    "requested arm in the three forms of the mechanism forecast (pooled (1 - w) g_F, short-stratum (1 - w) g_S, partial (1 - rho w) g_F), "
    "with g_F, g_S and rho taken unchanged from that forecast. Statements A1-A5 of src/arm_wording_extra.py with "
    f"W_NEAR0 = {W_NEAR0}, GAP_TOL_PP = {GAP_TOL_PP}, W_CORR = {W_CORR}, W_RANGE = {W_RANGE}; S1-S5 of "
    f"src/arm_wording_mechanism.py over the six clauses with TOL = {TOL}. Strict scores on the base run's six configurations; "
    "chain of thought from the base run's scored table, every trigger.")


class ExtraError(Exception):
    """An input or an order of steps that the run refuses."""


class Paths:
    """Every file of the run under one base folder; the attribute names follow src.arm_wording.Paths so that the
    run-check helpers of scripts/build_arm_wording.py work unchanged."""

    def __init__(self, base=None):
        b = Path(base) if base else ROOT
        self.base = b
        self.manifests = b / "manifests"
        self.configs = self.manifests / "arm_wording_configs.json"           # the base run's draw, read only
        self.clauses = self.manifests / "arm_wording_extra_clauses.json"
        self.forecast = self.manifests / "predictions_arm_wording_extra.json"
        self.forecast_sha = self.manifests / "predictions_arm_wording_extra.sha256"
        self.predictions = self.manifests / "predictions_arm_wording_extra_discovery.json"
        self.predictions_sha = self.manifests / "predictions_arm_wording_extra_discovery.sha256"
        self.mechanism = self.manifests / "predictions_arm_wording_mechanism.json"   # the mechanism forecast, read only
        self.mechanism_sha = self.manifests / "predictions_arm_wording_mechanism.sha256"
        self.requests = b / "data" / "requests" / "arm_wording_extra"
        self.outputs = b / "data" / "d2_out" / "arm_wording_extra"
        self.processed = b / "data" / "processed" / "arm_wording_extra"
        self.results = b / "results" / "arm_wording_extra"
        # the base run's files, read only (requests record for the panel and items; its scored table)
        self.d24_requests = b / "data" / "requests" / "arm_wording"
        self.d24_processed = b / "data" / "processed" / "arm_wording"
        self.d24_results = b / "results" / "arm_wording"


# --------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------

def clause_messages(bank: Mapping, config_id: str, clause_id: str, current: str = CURRENT) -> list[dict]:
    """The bank's requested-direct messages with the current clause replaced by the clause. The default clause removes
    the current clause together with the single space after it; it is refused unless, in this configuration, the clause
    stands at the start of a line and is followed by one space and further text on the same line (the answer-format
    sentence), so that no empty line or double space is left."""
    msgs = bank["configurations"][config_id]["arms"]["direct_answer"]["messages"]
    if clause_id == "dirdef":
        joined = "\n".join(m["content"] for m in msgs)
        i = joined.find(current)
        after = joined[i + len(current):i + len(current) + 2]
        if i < 0 or joined.count(current) != 1 or not (i == 0 or joined[i - 1] == "\n") or len(after) < 2 \
                or after[0] != " " or after[1] in " \n":
            raise ExtraError(f"{config_id}: the current clause is not followed by one space and the format sentence")
        return aw.replace_clause(msgs, current + " ", "")
    if clause_id == "dirfirm":
        return aw.replace_clause(msgs, current, FIRM)
    if clause_id == "dir0":
        return aw.replace_clause(msgs, current, current)
    raise ExtraError(f"unknown clause {clause_id}")


def render(model_key: str, task: str, bank: Mapping, config_ids: Sequence[str], items: pd.DataFrame,
           prefix: str = "", clauses: Sequence[str] = NEW) -> pd.DataFrame:
    """One row per configuration x arm (requested, forced) x clause x item; request_id "<prefix><task>/<config>/<clause>
    |<arm>|<item>", as the base run's layout."""
    rows = {c: [] for c in ("request_id", "model_key", "arm", "config_id", "item_id", "messages_json")}
    ids = [str(i) for i in items["item_id"]]
    qs = list(items["question"])
    for cid in config_ids:
        for arm in ARMS:
            for wid in clauses:
                msgs = clause_messages(bank, cid, wid)
                conf = f"{task}/{cid}/{wid}"
                for iid, q in zip(ids, qs):
                    rows["request_id"].append(f"{prefix}{conf}|{arm}|{iid}")
                    rows["model_key"].append(model_key)
                    rows["arm"].append(arm)
                    rows["config_id"].append(conf)
                    rows["item_id"].append(iid)
                    rows["messages_json"].append(json.dumps(d2_io.substitute_question(msgs, q), ensure_ascii=False))
    df = pd.DataFrame(rows)
    if df["request_id"].duplicated().any():
        raise ExtraError("request ids are not unique")
    return df


def clauses_manifest(configs_sha: str, banks_sha: Mapping[str, str]) -> dict:
    return {"kind": "arm_wording_extra_clauses", "rule": "src/arm_wording_extra.py",
            "clauses": [{"id": "dirdef", "text": "", "how": "the current clause and the single space after it are "
                         "removed from the requested-direct messages; the answer-format sentence stays"},
                        {"id": "dirfirm", "text": FIRM, "how": "str.replace of the current clause, once"}],
            "current_clause": CURRENT, "firm_sentence_from_d24": FIRM_SENTENCE_D24,
            "configs_manifest_sha256": configs_sha, "banks_sha256": dict(banks_sha)}


# --------------------------------------------------------------------------------------------
# Forecast (token counts only)
# --------------------------------------------------------------------------------------------

def forecast_inputs(mech: Mapping, models: Sequence[str], w_new: Mapping[str, np.ndarray]) -> dict:
    """{task: w (6, M), g_fd, g_short, rho, w_collection_current} from the mechanism forecast (the base run's four clauses, g_F, g_S,
    rho, unchanged) and the pilot's w under the two new clauses (w_new[task]: (2, M) in NEW order)."""
    if list(mech["models"]) != list(models):
        raise ExtraError("the mechanism forecast was saved for another panel order")
    if list(mech["clauses"]) != list(D24_CLAUSES):
        raise ExtraError(f"the mechanism forecast's clauses are {mech['clauses']}, not {list(D24_CLAUSES)}")
    out = {}
    for t, T in mech["per_task"].items():
        pm = T["per_model"]
        w4 = np.array([[pm[m]["by_clause"][d]["w"] for m in models] for d in D24_CLAUSES], float)
        wn = np.asarray(w_new[t], float)
        if wn.shape != (len(NEW), len(models)):
            raise ExtraError(f"{t}: pilot w has shape {wn.shape}")
        if np.isnan(wn).any():
            raise ExtraError(f"{t}: a model has no pilot output under a new clause")
        out[t] = {"w": np.vstack([w4, wn]),
                  "g_fd": np.array([pm[m]["g_forced"] for m in models], float),
                  "g_short": np.array([np.nan if pm[m]["g_forced_short"] is None else pm[m]["g_forced_short"]
                                       for m in models], float),
                  "rho": float(T["rho"]),
                  "w_collection_current": np.array([np.nan if pm[m]["w_collection_current_clause"] is None
                                                    else pm[m]["w_collection_current_clause"] for m in models], float)}
    return out


def forecast_record(models: Sequence[str], inputs: Mapping, source: Mapping) -> dict:
    """The mechanism record over the six clauses (so src/arm_wording_mechanism.evaluate reads it), with this run's label,
    rule, the near-zero sets and statements A1-A5, re-hashed."""
    body = awm.forecast_record(models, CLAUSES6, inputs, dict(source))
    body = {k: v for k, v in body.items() if k not in ("sha256",)}
    body["label"] = "arm_wording_extra"
    body["rule"] = RULE_TEXT
    body["new_clauses"] = list(NEW)
    body["clause_texts"] = {"dirdef": "", "dirfirm": FIRM}
    body["w_near0"], body["gap_tol_pp"], body["w_corr"] = W_NEAR0, GAP_TOL_PP, W_CORR
    for t, T in body["per_task"].items():
        T["cross_reference"] = None
        w = np.asarray(inputs[t]["w"], float)
        gf = np.asarray(inputs[t]["g_fd"], float)
        for k, d in enumerate(CLAUSES6):
            for j, m in enumerate(models):
                bc = T["per_model"][m]["by_clause"][d]
                for f in awm.FORMS:
                    bc[f"predicted_requested_minus_forced_{f}"] = float(gf[j] - bc[f"predicted_gain_vs_requested_{f}"])
        T["near0"] = {d: [m for j, m in enumerate(models) if w[CLAUSES6.index(d), j] <= W_NEAR0] for d in NEW}
        T["w_median_by_clause"] = {d: float(np.median(w[k])) for k, d in enumerate(CLAUSES6)}
        T["w_new_minus_current_median"] = {d: float(np.median(w[CLAUSES6.index(d)] - w[0])) for d in NEW}
    body["statements"] = {
        **{f"{k}_six_clauses": v for k, v in body["statements"].items()},
        "A1_forced_meets_requested": f"per task, among models whose pilot w under the firm clause is at most {W_NEAR0}: "
                                     f"(a) the mean requested minus forced accuracy under the firm clause lies within "
                                     f"{GAP_TOL_PP} pp of 0; (b) within {GAP_TOL_PP} pp in more than half of them",
        "A2_default_more_working": "per task, on the collections' own requested outputs: (a) the median over models of "
                                   "w(default) - w(current) is positive; (b) positive in more than half of models",
        "A3_firm_less_working": "per task: (a) the median of w(firm) - w(current) is negative; (b) negative in more "
                                "than half of models",
        "A4_pilot_predicts_w": f"per task, the pilot's and the collection's w under the two new clauses correlate at "
                               f"least {W_CORR} over models x clauses",
        "A5_extrapolates_to_forced": f"per task, among models whose w ranges over at least {W_RANGE} across the six "
                                     f"clauses, the median within-model intercept of requested minus forced accuracy "
                                     f"on w (pooled slope) lies within {GAP_TOL_PP} pp of 0"}
    body["note"] = ("read the pilot's token counts (no correctness) and the mechanism forecast; written before any output of "
                    "this run was scored.")
    body["sha256"] = awm.sha256_json({k: v for k, v in body.items() if k not in awm.UNHASHED})
    return body


def write_forecast(rec: dict, paths: Paths) -> dict:
    return awm.write_forecast(rec, paths.forecast, paths.forecast_sha)


def check_forecast(paths: Paths) -> str:
    try:
        return awm.check_forecast(paths.forecast, paths.forecast_sha)
    except ValueError as exc:
        raise ExtraError(f"{exc}; build it with scripts/build_arm_wording_extra.py forecast first")


def check_discovery_predictions(paths: Paths) -> str:
    if not paths.predictions.is_file() or not paths.predictions_sha.is_file():
        raise ExtraError("the discovery predictions are not saved; run score_arm_wording_extra.py --stage discovery "
                         "and run_arm_wording_extra.py --stage discovery first")
    want = paths.predictions_sha.read_text(encoding="utf-8").split()[0]
    got = aw.sha256_file(paths.predictions)
    if got != want:
        raise ExtraError(f"{paths.predictions} changed after it was saved (sha256 {got} != {want})")
    return got


def pilot_w(out_dir, models: Sequence[str], long_tokens: int = awm.LONG_TOKENS) -> tuple[dict, dict]:
    """{task: (2, M) share of requested-direct pilot outputs longer than long_tokens under each new clause} and the
    counts per cell. Reads model_key, arm, config_id and n_output_tokens only."""
    import glob
    parts = []
    for m in models:
        files = sorted(glob.glob(os.path.join(str(out_dir), m, "shard-*.parquet")))
        if not files:
            raise ExtraError(f"no pilot output for {m}")
        parts += [pd.read_parquet(f, columns=["model_key", "arm", "config_id", "n_output_tokens"]) for f in files]
    s = pd.concat(parts, ignore_index=True)
    s = s[s["arm"] == aw.REQ]
    p = s["config_id"].astype(str).map(aw.parse_config_id)
    s = s.assign(task=[x[0] for x in p], wid=[x[2] for x in p])
    if s["n_output_tokens"].isna().any():
        raise ExtraError("a pilot output has no token count")
    W, N = {}, {}
    for t in aw.TASKS:
        x = s[s["task"] == t]
        g = (x["n_output_tokens"] > long_tokens).groupby([x["wid"], x["model_key"]])
        sh, n = g.mean(), g.size()
        W[t] = np.array([[float(sh.get((d, m), np.nan)) for m in models] for d in NEW])
        N[t] = np.array([[int(n.get((d, m), 0)) for m in models] for d in NEW])
    return W, N


# --------------------------------------------------------------------------------------------
# Evaluation (after this run and the base run are scored)
# --------------------------------------------------------------------------------------------

def observed_w(table: pd.DataFrame, task: str, clauses: Sequence[str], models: Sequence[str],
               long_tokens: int = awm.LONG_TOKENS) -> np.ndarray:
    r = table[(table["task"] == task) & (table["arm"] == aw.REQ)]
    sh = (r["n_output_tokens"] > long_tokens).groupby([r["wording"], r["model_key"]]).mean()
    return np.array([[float(sh.get((d, m), np.nan)) for m in models] for d in clauses])


def _half(n_pos: int, n: int):
    return None if n == 0 else bool(n_pos > n / 2)


def evaluate(rec: Mapping, task: str, acc: Mapping[str, np.ndarray], reversals: Mapping, models: Sequence[str],
             w_obs: np.ndarray) -> dict:
    """Every statement for one task. acc: TaskArrays.acc() over the six clauses (strict); reversals:
    src/arm_wording.reversal_table on the same gains; w_obs: (6, M) long shares of the collections' requested outputs."""
    T = rec["per_task"][task]
    models = list(models)
    out = {"task": task, "mechanism_six_clauses": awm.evaluate(dict(rec), task, acc, reversals, models, list(CLAUSES6),
                                                                w_obs)}
    R = np.asarray(acc[aw.REQ], float).mean(0)      # (6, M)
    F = np.asarray(acc[aw.FD], float).mean(0)
    gap = R - F
    tol = float(rec["gap_tol_pp"])
    kf, kd, k0 = CLAUSES6.index("dirfirm"), CLAUSES6.index("dirdef"), 0
    # A1, and the same on the default clause beside it (diagnostic)
    a1 = {}
    for d in NEW:
        Z = [models.index(m) for m in T["near0"][d]]
        g = gap[CLAUSES6.index(d), Z] if Z else np.array([])
        a1[d] = {"models": [models[j] for j in Z], "gap_pp": [float(x) for x in g],
                 "mean_gap_pp": float(g.mean()) if len(g) else None,
                 "within_tol": int((np.abs(g) <= tol).sum()), "n": len(Z),
                 "holds_a": None if not len(g) else bool(abs(g.mean()) <= tol),
                 "holds_b": _half(int((np.abs(g) <= tol).sum()), len(Z)),
                 "note": None if len(Z) else "no model's pilot w is at or below the threshold: not testable"}
    out["A1_forced_meets_requested"] = {**a1["dirfirm"], "tolerance_pp": tol, "w_near0": rec["w_near0"],
                                        "default_clause_beside": a1["dirdef"]}
    # A2, A3
    for name, k, sign in (("A2_default_more_working", kd, 1), ("A3_firm_less_working", kf, -1)):
        diff = w_obs[k] - w_obs[k0]
        ok = np.isfinite(diff)
        d = diff[ok]
        med = float(np.median(d)) if len(d) else None
        n_dir = int((sign * d > 0).sum())
        out[name] = {"median_diff": med, "models_in_direction": n_dir, "n": int(ok.sum()),
                     "holds_a": None if med is None else bool(sign * med > 0), "holds_b": _half(n_dir, int(ok.sum())),
                     "per_model": {m: (None if not ok[j] else float(diff[j])) for j, m in enumerate(models)}}
    # A4
    wp = np.array([[T["per_model"][m]["by_clause"][d]["w"] for m in models] for d in NEW])
    wo = w_obs[[kd, kf]]
    ok = np.isfinite(wo) & np.isfinite(wp)
    r = float(np.corrcoef(wp[ok], wo[ok])[0, 1]) if ok.sum() > 2 and np.std(wp[ok]) > 0 and np.std(wo[ok]) > 0 else None
    out["A4_pilot_predicts_w"] = {"pearson": r, "threshold": rec["w_corr"], "holds": None if r is None else bool(r >= rec["w_corr"])}
    # A5
    w6 = np.array([[T["per_model"][m]["by_clause"][d]["w"] for m in models] for d in CLAUSES6])
    wr = w6.max(0) - w6.min(0)
    sel = np.flatnonzero(wr >= float(rec["w_range"]))
    if len(sel):
        xc = (w6[:, sel] - w6[:, sel].mean(0)).ravel()
        yc = (gap[:, sel] - gap[:, sel].mean(0)).ravel()
        b = None if (xc ** 2).sum() <= 0 else float((xc * yc).sum() / (xc ** 2).sum())
    else:
        b = None
    if b is not None:
        a = gap[:, sel].mean(0) - b * w6[:, sel].mean(0)
        med_a = float(np.median(a))
    else:
        a, med_a = np.array([]), None
    out["A5_extrapolates_to_forced"] = {
        "models": len(sel), "pooled_slope_pp_per_unit_w": b, "median_intercept_pp": med_a,
        "intercepts_pp": {models[j]: float(x) for j, x in zip(sel, a)}, "tolerance_pp": tol,
        "holds": None if med_a is None else bool(abs(med_a) <= tol),
        "note": None if len(sel) else "no model's w ranges over the threshold: not testable"}
    out["accuracy_by_clause_mean_pp"] = {"requested": {d: float(R[k].mean()) for k, d in enumerate(CLAUSES6)},
                                         "forced": {d: float(F[k].mean()) for k, d in enumerate(CLAUSES6)},
                                         "requested_minus_forced": {d: float(gap[k].mean()) for k, d in enumerate(CLAUSES6)}}
    out["w_observed_median_by_clause"] = {d: (None if not np.isfinite(w_obs[k]).any() else float(np.nanmedian(w_obs[k])))
                                          for k, d in enumerate(CLAUSES6)}
    return out


def holds_summary(E: Mapping) -> dict:
    out = {}
    for t, x in E["tasks"].items():
        m = x["mechanism_six_clauses"]
        out[t] = {"A1a": x["A1_forced_meets_requested"]["holds_a"], "A1b": x["A1_forced_meets_requested"]["holds_b"],
                  "A2a": x["A2_default_more_working"]["holds_a"], "A2b": x["A2_default_more_working"]["holds_b"],
                  "A3a": x["A3_firm_less_working"]["holds_a"], "A3b": x["A3_firm_less_working"]["holds_b"],
                  "A4": x["A4_pilot_predicts_w"]["holds"], "A5": x["A5_extrapolates_to_forced"]["holds"],
                  "S1": m["S1_clause_reversal"]["holds"], "S2": m["S2_clause_vs_template"]["holds"],
                  "S3_count": m["S3_arm_switch"]["holds_count"], "S3_all": m["S3_arm_switch"]["holds_all"],
                  "S4a": m["S4_requested_rises"]["holds_a"], "S4b": m["S4_requested_rises"]["holds_b"],
                  "S5a": m["S5_forced_steadier"]["holds_a"], "S5b": m["S5_forced_steadier"]["holds_b"]}
    return out


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
