"""The requested baseline as a mixture of short answers and written-out working.

A requested-direct output longer than the forced arm's output cap (in generated tokens) is one the forced arm could not
have produced: the model wrote more than a bare answer. Let w be a model's share of such outputs. If short requested
answers score like forced ones and long ones like chain of thought, a model's gain over the requested arm is about
(1 - w) times its gain over the forced arm. The correction uses no gold answer: w comes from output lengths alone.

Pure functions; scripts/run_baseline_mixture.py loads the data. Scores in pp; w as a share in [0, 1].
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from scipy import stats

from src import robustness as rb

RULE_TEXT = (
    "For each model, w is the share of its requested-direct outputs longer than the forced arm's output cap (generated "
    "tokens, n_output_tokens of the stored outputs, compared with the cap by '>'), computed per configuration and "
    "averaged over the discovery draws (a configuration drawn twice counts twice). The gain over the forced arm is the "
    "chain-of-thought minus forced-direct accuracy (strict scoring), the model mean over the discovery draws. The "
    "predicted gain over the requested arm is (1 - w) x the gain over the forced arm. Predicted reversals: model pairs "
    "ordered one way by the gain over the forced arm and the other way by the predicted gain over the requested arm "
    "(ties, |difference| <= 1e-9 pp, are neither). The predicted reversal share is their number over all model pairs.")


# --------------------------------------------------------------------------- w and the correction


def draw_weighted_share(long_flags, model_idx, config_idx, n_models: int, n_configs: int, draw_configs) -> np.ndarray:
    """(n_models,) share in [0, 1] of long outputs per model: the share within each configuration, averaged over the
    draws (draw_configs: configuration index of each draw, repeats included). A model with no output in a drawn
    configuration gives nan for that model (reported, not imputed)."""
    long_flags = np.asarray(long_flags, bool)
    model_idx, config_idx = np.asarray(model_idx), np.asarray(config_idx)
    C = int(n_configs)
    n = np.zeros((n_models, C))
    k = np.zeros((n_models, C))
    np.add.at(n, (model_idx, config_idx), 1)
    np.add.at(k, (model_idx, config_idx), long_flags.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = k / n
    share[n == 0] = np.nan
    return share[:, list(draw_configs)].mean(1)


def corrected_gain(g_forced, w) -> np.ndarray:
    """Predicted gain over the requested arm: (1 - w) x gain over the forced arm."""
    return (1.0 - np.asarray(w, float)) * np.asarray(g_forced, float)


def reversed_pairs(g1, g2, names=None) -> list:
    """The model pairs (by name or index) ordered oppositely by g1 and g2; ties under either are not reversed."""
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    i, j = np.triu_indices(len(g1), 1)
    s1 = rb.d1.tie_sign(g1[i] - g1[j])
    s2 = rb.d1.tie_sign(g2[i] - g2[j])
    rev = (s1 * s2) < 0
    nm = list(names) if names is not None else list(range(len(g1)))
    return [[nm[a], nm[b]] for a, b in zip(i[rev], j[rev])]


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]), "spearman": float(stats.spearmanr(x[ok], y[ok]).statistic)}


def mechanism(g_req, g_fd, w) -> dict:
    """Reversed pairs between the observed gain over the requested arm and (a) the gain over the forced arm, (b) the
    corrected gain; correlations of each with the observed gain; root mean square error of (b)."""
    g_req, g_fd, w = (np.asarray(x, float) for x in (g_req, g_fd, w))
    pred = corrected_gain(g_fd, w)
    a, b = rb.pair_reversals(g_req, g_fd), rb.pair_reversals(g_req, pred)
    strip = lambda r: {k: r[k] for k in ("pairs", "reversed", "reversed_share", "ties")}  # noqa: E731
    return {"observed_vs_forced": strip(a), "observed_vs_corrected": strip(b),
            "corr_forced": _corr(g_req, g_fd), "corr_corrected": _corr(g_req, pred),
            "rms_corrected_pp": float(np.sqrt(np.nanmean((pred - g_req) ** 2))),
            "rms_forced_pp": float(np.sqrt(np.nanmean((g_fd - g_req) ** 2)))}


# --------------------------------------------------------------------------- item level


def item_level(model_idx, long_flags, scores: dict, n_models: int, n_boot: int, rng) -> dict:
    """Accuracy (pp) of each arm and score within short and long requested outputs, pooled over the rows of the panel,
    with a percentile model bootstrap (models drawn with replacement, every row of a drawn model kept).
    scores: {(arm, score): (rows,) 0/1}. Returns {'short'|'long': {'share', 'n', 'acc': {arm/score: {...}}}}."""
    model_idx = np.asarray(model_idx)
    lf = np.asarray(long_flags, bool)
    cnt = np.zeros((n_models, 2))
    np.add.at(cnt, (model_idx, lf.astype(int)), 1)
    sums = {}
    for key, v in scores.items():
        s = np.zeros((n_models, 2))
        np.add.at(s, (model_idx, lf.astype(int)), np.asarray(v, float))
        sums[key] = s
    boots = [rng.integers(0, n_models, n_models) for _ in range(n_boot)]
    out = {}
    total = cnt.sum()
    for g, name in ((0, "short"), (1, "long")):
        blk = {"n": int(cnt[:, g].sum()), "share": float(cnt[:, g].sum() / total) if total else None,
               "models_with_rows": int((cnt[:, g] > 0).sum()), "acc": {}}
        sh_b = np.array([cnt[ix, g].sum() / cnt[ix].sum() for ix in boots])
        blk["share_ci95"] = [float(np.quantile(sh_b, 0.025)), float(np.quantile(sh_b, 0.975))]
        for key, s in sums.items():
            n = cnt[:, g].sum()
            est = float(100 * s[:, g].sum() / n) if n else None
            with np.errstate(invalid="ignore", divide="ignore"):
                b = np.array([100 * s[ix, g].sum() / cnt[ix, g].sum() for ix in boots])
            b = b[np.isfinite(b)]
            blk["acc"]["/".join(key)] = {"estimate": est,
                                         "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else None}
        out[name] = blk
    return out


# --------------------------------------------------------------------------- the HotpotQA forecast (saved before validation)


UNHASHED = ("created", "sha256", "source")   # provenance kept beside the content; the sha256 covers the forecast itself


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def forecast_record(panel, w, g_fd, g_req_discovery, cap: int, discovery_units, source: dict) -> dict:
    """The saved forecast: per model w, gain over the forced arm and predicted gain over the requested arm; the
    predicted reversals of the arm switch; the statements the validation draws will test; the rule text."""
    panel = list(panel)
    w, g_fd, g_rq = (np.asarray(x, float) for x in (w, g_fd, g_req_discovery))
    pred = corrected_gain(g_fd, w)
    pairs = reversed_pairs(g_fd, pred, panel)
    n_pairs = len(panel) * (len(panel) - 1) // 2
    body = {
        "label": "hotpot_mechanism", "forced_cap_tokens": int(cap), "units": "pp; w as a share in [0, 1]",
        "rule": RULE_TEXT, "models": panel, "discovery_units": list(map(str, discovery_units)),
        "per_model": {k: {"w": float(w[j]), "gain_vs_forced": float(g_fd[j]), "predicted_gain_vs_requested": float(pred[j])}
                      for j, k in enumerate(panel)},
        "predicted_reversal": {"pairs": n_pairs, "reversed": len(pairs), "share": len(pairs) / n_pairs if n_pairs else None,
                               "reversed_pairs": pairs},
        "statements": {
            "share": "the validation draws' reversal share between the gain over the requested arm and the gain over "
                     "the forced arm (model means over the 24 validation draws) lies within 0.05 of the predicted share",
            "pairs": "among the predicted reversed pairs, the share that reverse on the validation draws exceeds the "
                     "share among the other pairs",
            "gain": "the predicted gain over the requested arm correlates (Pearson, over models) more strongly with the "
                    "validation draws' gain over the requested arm than the discovery gain over the forced arm does"},
        "discovery_context": {"gain_vs_requested_observed": {k: float(g_rq[j]) for j, k in enumerate(panel)},
                              "post_hoc_fit_on_discovery": mechanism(g_rq, g_fd, w)},
        "source": source,
        "note": "read only HotpotQA discovery-draw outputs; written before any validation draw was scored. "
                "The mechanism was found exploratorily on GSM8K, BIG-Bench Hard and "
                "MATH-500; this is its forward test."}
    body["sha256"] = sha256_json({k: v for k, v in body.items() if k not in UNHASHED})
    return body


def write_forecast(rec: dict, path: str) -> dict:
    """Write-once: an existing file with the same content (sha256) is kept as saved; a different one is refused."""
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("sha256") == rec["sha256"]:
            return old
        raise FileExistsError(f"{path} holds a different forecast; refusing to overwrite")
    rec = dict(rec)
    rec["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return rec


def load_forecast(path: str) -> dict:
    """The saved forecast, after checking its sha256 against its content."""
    rec = json.load(open(path, encoding="utf-8"))
    if sha256_json({k: v for k, v in rec.items() if k not in UNHASHED}) != rec.get("sha256"):
        raise ValueError(f"{path} does not match its recorded sha256")
    return rec


def evaluate_forecast(rec: dict, panel, g_req_val, g_fd_val, tol: float = 0.05) -> dict:
    """The saved forecast against the validation draws (model means over the validation draws): the reversal share, the
    predicted pairs (hits, misses, false alarms), the gain correlations, and the correction applied to the validation
    gains with the saved w. Every statement is reported, misses included."""
    panel = list(panel)
    if list(rec["models"]) != panel:
        raise ValueError("the forecast was saved for another panel")
    w = np.array([rec["per_model"][k]["w"] for k in panel])
    pred = np.array([rec["per_model"][k]["predicted_gain_vs_requested"] for k in panel])
    gfd_disc = np.array([rec["per_model"][k]["gain_vs_forced"] for k in panel])
    g_req_val, g_fd_val = np.asarray(g_req_val, float), np.asarray(g_fd_val, float)
    obs_pairs = reversed_pairs(g_req_val, g_fd_val, panel)
    n_pairs = len(panel) * (len(panel) - 1) // 2
    P = {tuple(p) for p in rec["predicted_reversal"]["reversed_pairs"]}
    O = {tuple(p) for p in obs_pairs}
    hits, false_alarms, misses = sorted(P & O), sorted(P - O), sorted(O - P)
    obs_share = len(O) / n_pairs if n_pairs else None
    pred_share = rec["predicted_reversal"]["share"]
    rate_pred = len(hits) / len(P) if P else None
    rate_other = len(misses) / (n_pairs - len(P)) if n_pairs - len(P) else None
    c_pred, c_fd = _corr(g_req_val, pred)["pearson"], _corr(g_req_val, gfd_disc)["pearson"]
    st = {"share": {"predicted": pred_share, "observed": obs_share, "tolerance": tol,
                    "holds": None if obs_share is None or pred_share is None else bool(abs(obs_share - pred_share) <= tol)},
          "pairs": {"rate_among_predicted": rate_pred, "rate_among_other": rate_other,
                    "holds": None if rate_pred is None or rate_other is None else bool(rate_pred > rate_other),
                    "note": None if P else "no pair was predicted to reverse: the statement cannot be tested"},
          "gain": {"pearson_predicted": c_pred, "pearson_discovery_forced_gain": c_fd,
                   "holds": None if c_pred is None or c_fd is None else bool(c_pred > c_fd)}}
    return {"forecast_sha256": rec["sha256"], "forecast_created": rec.get("created"), "pairs": n_pairs,
            "observed_reversed": len(O), "predicted_reversed": len(P), "hits": [list(p) for p in hits],
            "false_alarms": [list(p) for p in false_alarms], "misses": [list(p) for p in misses], "statements": st,
            "correction_on_validation": mechanism(g_req_val, g_fd_val, w)}


# --------------------------------------------------------------------------- the exact decomposition of E[C - F | short]


DECOMP_ROOM_PP = 10.0   # models with at least this much room (chain of thought minus forced, pp) enter the closed fraction


def _w_sum(model_idx, mask, v, wt, n_models):
    s = np.zeros(n_models)
    np.add.at(s, model_idx[mask], (v * wt)[mask])
    return s


def _cov_share(part, total):
    """cov(part, total) / var(total) over models (ddof 1); None when var(total) is not positive (reported as is)."""
    part, total = np.asarray(part, float), np.asarray(total, float)
    v = float(np.var(total, ddof=1)) if len(total) > 1 else float("nan")
    if not np.isfinite(v) or v <= 0:
        return None
    return float(np.cov(part, total, ddof=1)[0, 1] / v)


def _quart(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
            "n": int(len(x))}


def decomposition(model_idx, long_flags, r, f, c, n_models: int, row_weight=None, room_pp: float = DECOMP_ROOM_PP) -> dict:
    """The requested-minus-forced accuracy of each model (its baseline shift, pp) split exactly into a part from rows
    whose requested output is long and a part from short ones: shift_i = long_i + short_i, with
    long_i = 100 x sum over long rows of wt x (r - f) / sum over all rows of wt (short_i alike). r, f, c: (rows,) 0/1
    scores of the requested, forced and chain-of-thought arms on the same (model, configuration, item); row_weight: the
    number of draws of the row's configuration (1 when every configuration counts once).

    Returns per model (w, the three accuracies, both parts, the stratum accuracies), the share of the between-model
    variance of the shift that comes from long rows, cov(long, shift) / var(shift) (ddof 1; the short share is the
    rest; var(long) / var(shift), var(short) / var(shift) and the covariance term beside it), within-model recovery on
    long rows (R_L - F_L) / (C_L - F_L), the short-row gap R_S - F_S, the closed fraction (R - F) / (C - F) against w
    among models with C - F >= room_pp, and both forms of the (1 - w) rule against the observed gain over the requested
    arm C - R: (1 - w) x (C - F) and (1 - w) x (C_S - F_S). Edge cases are reported as they are: a model with no long
    (or no short) row has nan stratum accuracies; a nonpositive recovery denominator is listed, not dropped silently;
    with w = 1 the short-stratum form is (1 - w) x an empty stratum and is reported as 0 with the model listed."""
    m = np.asarray(model_idx)
    lf = np.asarray(long_flags, bool)
    wt = np.ones(len(m)) if row_weight is None else np.asarray(row_weight, float)
    r, f, c = (np.asarray(x, float) for x in (r, f, c))
    allm = np.ones(len(m), bool)
    one = np.ones(len(m))
    N = _w_sum(m, allm, one, wt, n_models)
    NL = _w_sum(m, lf, one, wt, n_models)
    NS = N - NL
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = {a: 100 * _w_sum(m, allm, v, wt, n_models) / N for a, v in (("R", r), ("F", f), ("C", c))}
        accL = {a: 100 * _w_sum(m, lf, v, wt, n_models) / NL for a, v in (("R", r), ("F", f), ("C", c))}
        accS = {a: 100 * _w_sum(m, ~lf, v, wt, n_models) / NS for a, v in (("R", r), ("F", f), ("C", c))}
        w = NL / N
        part_long = 100 * _w_sum(m, lf, r - f, wt, n_models) / N
        part_short = 100 * _w_sum(m, ~lf, r - f, wt, n_models) / N
    shift = acc["R"] - acc["F"]
    if np.nanmax(np.abs(part_long + part_short - shift)) > 1e-8:
        raise AssertionError("the two parts do not add up to the shift")
    vs = float(np.var(shift, ddof=1))
    var_terms = None if not vs > 0 else {"long": float(np.var(part_long, ddof=1) / vs),
                                         "short": float(np.var(part_short, ddof=1) / vs),
                                         "two_cov": float(2 * np.cov(part_long, part_short, ddof=1)[0, 1] / vs)}
    with np.errstate(invalid="ignore", divide="ignore"):
        den = accL["C"] - accL["F"]
        recovery = (accL["R"] - accL["F"]) / den
        room = acc["C"] - acc["F"]
        closed = (acc["R"] - acc["F"]) / room
    rec_ok = np.isfinite(recovery) & (den > 0)
    rec_bad = [{"model": int(j), "n_long_rows_weighted": float(NL[j]),
                "denominator_pp": float(den[j]) if np.isfinite(den[j]) else None} for j in range(n_models) if not rec_ok[j]]
    sel = room >= room_pp
    short_gap = accS["R"] - accS["F"]
    g_obs = acc["C"] - acc["R"]
    pred_pooled = (1 - w) * (acc["C"] - acc["F"])
    gS = accS["C"] - accS["F"]
    no_short = ~np.isfinite(gS)
    pred_short = np.where(no_short & (w >= 1), 0.0, (1 - w) * gS)

    def form(pred):
        ok = np.isfinite(pred)
        rv = rb.pair_reversals(g_obs[ok], pred[ok])
        return {"rms_pp": float(np.sqrt(np.mean((pred[ok] - g_obs[ok]) ** 2))),
                "mean_error_pp": float(np.mean(pred[ok] - g_obs[ok])), "corr": _corr(g_obs, pred),
                "reversed": rv["reversed"], "pairs": rv["pairs"], "reversed_share": rv["reversed_share"],
                "models": int(ok.sum())}
    return {
        "per_model": {"w": w.tolist(), "R": acc["R"].tolist(), "F": acc["F"].tolist(), "C": acc["C"].tolist(),
                      "shift": shift.tolist(), "part_long": part_long.tolist(), "part_short": part_short.tolist(),
                      "R_long": accL["R"].tolist(), "F_long": accL["F"].tolist(), "C_long": accL["C"].tolist(),
                      "R_short": accS["R"].tolist(), "F_short": accS["F"].tolist(), "C_short": accS["C"].tolist(),
                      "recovery_long": recovery.tolist(), "closed_fraction": closed.tolist(),
                      "predicted_pooled": pred_pooled.tolist(), "predicted_short": pred_short.tolist()},
        "variance_share_long": _cov_share(part_long, shift), "variance_share_short": _cov_share(part_short, shift),
        "variance_terms": var_terms, "shift_variance_pp2": vs,
        "recovery_long": _quart(np.where(rec_ok, recovery, np.nan)), "recovery_long_excluded": rec_bad,
        "short_gap_pp": _quart(short_gap), "short_gap_abs_pp": _quart(np.abs(short_gap)),
        "closed_fraction": {"room_pp": room_pp, "models": int(sel.sum()),
                            "corr_with_w": _corr(w[sel], closed[sel]),
                            "corr_shift_with_w": _corr(w[sel], shift[sel]),
                            "corr_shift_with_w_all_models": _corr(w, shift)},
        "forms": {"pooled": form(pred_pooled), "short_stratum": form(pred_short)},
        "short_stratum_empty_models": [int(j) for j in np.flatnonzero(no_short)],
        "note": "variance share: cov(part, shift) / var(shift), ddof 1; the long and short shares add to 1"}


def corrected_choice(g_fd, w, g_req_eval, n: int, g_short=None) -> dict:
    """Top n by the gain over the forced arm and by (1 - w) x that gain (and, given g_short, by (1 - w) x the
    short-stratum gain, an empty stratum counting 0), each valued under the requested arm on g_req_eval ((K, M)
    evaluation gains, or (M,) means): plug-in regret in pp per chosen model (src/decision.regret_plugin)."""
    from src import decision as dc
    G = np.atleast_2d(np.asarray(g_req_eval, float))
    preds = {"forced_gain": np.asarray(g_fd, float), "corrected": corrected_gain(g_fd, w)}
    if g_short is not None:
        preds["corrected_short"] = (1 - np.asarray(w, float)) * np.nan_to_num(np.asarray(g_short, float), nan=0.0)
    out = {}
    for k, p in preds.items():
        W, t = dc.choose(p, n)
        out[k] = {"regret": float(dc.regret_plugin(W, G, n)[0]), "ties_at_cut": int(t), "weights": W.tolist()}
    return out
