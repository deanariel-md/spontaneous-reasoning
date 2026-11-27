"""Dataset 2 analysis: prospective paired evaluation on GSM8K .

Data are scored outputs, one row per (model_key, arm, config_id, item_id) (src/scoring.py). A
unit is one prompt configuration drawn by the sampler; within a unit every model and
both arms use that configuration. The sampler manifest lists the discovery and validation
config ids in the fixed nested order (sampling.registered_split_bank).

Design: Steps 1 and 2 are estimated from the K
discovery units; the prediction rules are fitted on the same units and frozen; the seven risk
differences are estimated on the K validation units. Each target's interval is the method
selected for the Dataset 2 suite in manifests/calibration_selection_v2.json. A target without a
selected method is reported as a point estimate labelled uncalibrated.

Classification against a practical margin reads the calibrated interval:
"outside margin" when the whole interval lies beyond the margin, "within margin" when it lies
inside, "unresolved" otherwise. For V_delta, "within margin" uses the calibrated one-sided 95%
upper bound and "outside margin" the lower end of the calibrated two-sided interval.

Edge cases are reported as they occur: negative variance estimates stay negative, nonpositive
denominators give undefined ratios, Fieller sets may be unbounded, ties are counted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import dataset1 as d1, intervals as iv
from .estimators import (cross_risk, gaussian_agreement, pair_agreement, paired_loss_difference, reliability,
                         tie_sign)
from .sampling import budgets, sha256_json

ARMS = ("direct_answer", "zero_shot_cot")          # baseline, intervention (config/tasks.yaml)
VALUES = ("strict_correct", "lenient_correct", "compliant")
PRACTICAL = d1.PRACTICAL
SUITE = "dataset2"
GATING = ("gap", "C_benefit", "H", "V_delta", "V_delta_upper") + tuple(
    f"risk:{f}-{g}" for f, g in d1.COMPARISONS)


class IncompleteData(ValueError):
    """A required (model, arm, configuration, item) cell is missing or duplicated."""


# --------------------------------------------------------------------------- data adapter


def cell_table(scored: pd.DataFrame, panel, configs, items, arms=ARMS, values=VALUES) -> pd.DataFrame:
    """Per (model_key, arm, config_id): item count, correct counts and percentages.

    Every (model, arm, configuration) in panel x arms x configs must have exactly one row per
    item of the fixed item set; anything else raises IncompleteData (unbalanced
    arrays need a designed estimator, not imputation)."""
    items = pd.Index(pd.unique(pd.Series(list(items))))
    configs = sorted(set(map(str, configs)))
    df = scored[scored["model_key"].isin(panel) & scored["arm"].isin(arms)
                & scored["config_id"].astype(str).isin(configs)].copy()
    df["config_id"] = df["config_id"].astype(str)
    stray = ~df["item_id"].isin(items)
    if stray.any():
        raise IncompleteData(f"{int(stray.sum())} rows have item ids outside the fixed item set, "
                             f"e.g. {df.loc[stray, 'item_id'].head(3).tolist()}")
    key = ["model_key", "arm", "config_id"]
    dup = df.duplicated(key + ["item_id"])
    if dup.any():
        raise IncompleteData(f"{int(dup.sum())} duplicated (model, arm, configuration, item) rows")
    agg = {"n_items": ("item_id", "size")}
    for v in values:
        agg[f"n_{v}"] = (v, "sum")
    g = df.groupby(key, sort=False).agg(**agg)
    want = pd.MultiIndex.from_product([list(panel), list(arms), configs], names=key)
    g = g.reindex(want)
    bad = g["n_items"].isna() | (g["n_items"] != len(items))
    if bad.any():
        ex = g[bad].head(5)
        raise IncompleteData(f"{int(bad.sum())} of {len(g)} (model, arm, configuration) cells lack the "
                             f"full item set of {len(items)}; first: "
                             f"{[(i, None if pd.isna(n) else int(n)) for i, n in zip(ex.index, ex['n_items'])]}")
    for v in values:
        # counts stay integers; a per-item score in [0, 1] (HotpotQA token F1) keeps its fractional sum
        if np.issubdtype(df[v].dtype, np.floating) and not np.allclose(g[f"n_{v}"], np.round(g[f"n_{v}"])):
            g[f"n_{v}"] = g[f"n_{v}"].astype(np.float64)
        else:
            g[f"n_{v}"] = g[f"n_{v}"].astype(np.int64)
        g[f"pct_{v}"] = 100.0 * g[f"n_{v}"] / len(items)
    g["n_items"] = g["n_items"].astype(np.int64)
    return g


def unit_arrays(cells: pd.DataFrame, units, panel, value="strict_correct", arms=ARMS, counts=False):
    """(B, P), each (K, M): baseline and intervention scores of each listed unit, in pp, or the
    correct counts when counts=True. Repeated config ids give repeated rows."""
    col = f"n_{value}" if counts else f"pct_{value}"
    out = []
    for arm in arms:
        idx = pd.MultiIndex.from_tuples([(mk, arm, str(c)) for c in units for mk in panel])
        out.append(cells.loc[idx, col].to_numpy().reshape(len(units), len(panel)))
    return out[0], out[1]


def load_selection(path, suite=SUITE) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    sel = json.loads(raw)["selection"][suite]
    out = {t: v["method"] for t, v in sel.items()}
    out["_sha256"] = hashlib.sha256(raw).hexdigest()
    out["_path"] = os.path.relpath(path)
    return out


# --------------------------------------------------------------------------- intervals


def apply_ci(method, est, loo, bn, bd, bl, m):
    """The calibrated two-sided interval, computed exactly as in src/calibration.py."""
    if method is None:
        return None
    if method == "jack_t":
        lo, hi = iv.jack_t(est, loo)
    elif method == "boot_naive":
        lo, hi, _ = iv.percentile(bn)
    elif method == "boot_distinct":
        lo, hi, _ = iv.percentile(bd)
    elif method == "bca_distinct":
        lo, hi = iv.bca(est, bd, loo)
    elif method == "boot_t":
        lo, hi = iv.boot_t(est, loo, bd, bl, m)
    else:
        raise ValueError(f"unknown interval method {method!r}")
    return [float(lo), float(hi)]


def apply_upper(method, est, loo, bn, bd, bl, m, G):
    if method is None:
        return None
    f = {"jack_t": lambda: iv.jack_t_upper(est, loo),
         "boot_naive": lambda: iv.percentile_upper(bn),
         "boot_distinct": lambda: iv.percentile_upper(bd),
         "bca_distinct": lambda: iv.bca_upper(est, bd, loo),
         "boot_t": lambda: iv.boot_t_upper(est, loo, bd, bl, m),
         "mls": lambda: iv.mls_vdelta_upper(G)}
    if method not in f:
        raise ValueError(f"unknown upper-bound method {method!r}")
    return float(f[method]())


def classify_interval(lo, hi, margin) -> str:
    """Symmetric margin: the interval beyond +margin or -margin, inside, or neither."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if lo > margin:
        return "outside margin (positive)"
    if hi < -margin:
        return "outside margin (negative)"
    if -margin <= lo and hi <= margin:
        return "within margin"
    return "unresolved"


def classify_set(kind, a, b, margin) -> str:
    """Fieller sets: bounded, exclusive (-inf, a] U [b, inf), whole, empty or undefined."""
    if kind == "bounded":
        return classify_interval(a, b, margin)
    if kind == "exclusive":
        return "outside margin (sign unresolved)" if a < -margin and b > margin else "unresolved"
    if kind == "whole":
        return "unresolved"
    return "undefined"


def classify_vdelta(lower_two_sided, upper_one_sided, margin=PRACTICAL["V_delta_pp2"]) -> str:
    lo_ok = lower_two_sided is not None and np.isfinite(lower_two_sided)
    up_ok = upper_one_sided is not None and np.isfinite(upper_one_sided)
    if up_ok and upper_one_sided < margin:
        return "within margin"
    if lo_ok and lower_two_sided > margin:
        return "outside margin (positive)"
    if not (lo_ok or up_ok):
        return "undefined"
    return "unresolved"


def jack_t_test(est, loo) -> dict:
    """Two-sided t test of zero dual to the jackknife-t interval (K - 1 df)."""
    se = iv.jackknife_se(loo)
    if not (np.isfinite(est) and np.isfinite(se) and se > 0):
        return {"t": float("nan"), "df": len(loo) - 1, "p_two_sided": float("nan")}
    t = est / se
    return {"t": float(t), "df": len(loo) - 1, "p_two_sided": float(2 * stats.t.sf(abs(t), len(loo) - 1))}


# --------------------------------------------------------------------------- Step 1


def pairwise_unit_differences(X, names, margin=PRACTICAL["pairwise_pp"]) -> list[dict]:
    """Per model pair: the distribution over units of the one-draw difference."""
    X = np.asarray(X, float)
    out = []
    for a, b in zip(*np.triu_indices(X.shape[1], 1)):
        d = X[:, a] - X[:, b]
        q = np.quantile(d, [0.025, 0.25, 0.5, 0.75, 0.975])
        out.append({"pair": f"{names[a]} - {names[b]}", "mean_difference": float(d.mean()),
                    "sd_one_draw": float(d.std(ddof=1)), "q025": float(q[0]), "q25": float(q[1]),
                    "median": float(q[2]), "q75": float(q[3]), "q975": float(q[4]),
                    "frac_abs_gt_margin": float(np.mean(np.abs(d) > margin))})
    return out


def _agreement_block(ag: iv.Agreement, method, m) -> dict:
    p = ag.parts()
    dec = p["same"] + p["opposite"]
    p["decisive"] = dec
    p["conditional_agreement"] = p["same"] / dec if dec > 0 else float("nan")
    p["method"] = method
    p["ci95"] = apply_ci(method, p["C"], ag.loo(), ag.boot(m, False), ag.boot(m, True), ag.boot_loo(m), m)
    return p


def step1(Bd, Pd, sel, m, names) -> dict:
    """Ordering agreement for benefit and accuracy between distinct discovery units."""
    G = Pd - Bd
    agB, agA = iv.Agreement(G), iv.Agreement(Bd)
    out = {"K": int(Bd.shape[0]),
           "C_benefit": _agreement_block(agB, sel.get("C_benefit"), m),
           "C_accuracy": _agreement_block(agA, sel.get("C_accuracy"), m)}
    est = out["C_accuracy"]["C"] - out["C_benefit"]["C"]
    loo = agA.loo() - agB.loo()
    out["gap"] = {"estimate": est, "method": sel.get("gap"),
                  "ci95": apply_ci(sel.get("gap"), est, loo, agA.boot(m, False) - agB.boot(m, False),
                                   agA.boot(m, True) - agB.boot(m, True),
                                   agA.boot_loo(m) - agB.boot_loo(m), m),
                  "test_zero": jack_t_test(est, loo) if sel.get("gap") == "jack_t" else None}
    out["pairwise_differences_benefit"] = pairwise_unit_differences(G, names)
    out["pairwise_differences_accuracy"] = pairwise_unit_differences(Bd, names)
    return out


def per_model_draw_agreement(X) -> dict:
    """Separately labelled sampling procedure: each model receives its own
    configuration in each evaluation. For a model pair (i, j), evaluation 1 uses units a (for i)
    and b (for j), evaluation 2 uses c and d, with a, b, c, d distinct; averaged over all such
    ordered quadruples and over model pairs."""
    X = np.asarray(X, float)
    K, M = X.shape
    if K < 4:
        raise ValueError("needs at least four units")
    a, b = np.where(~np.eye(K, dtype=bool))
    disjoint = ((a[:, None] != a[None, :]) & (a[:, None] != b[None, :])
                & (b[:, None] != a[None, :]) & (b[:, None] != b[None, :])).astype(float)
    i, j = np.triu_indices(M, 1)
    s = tie_sign(X[a][:, i] - X[b][:, j])         # (ordered unit pairs, model pairs)
    pos, neg = (s > 0).astype(float), (s < 0).astype(float)
    total = K * (K - 1) * (K - 2) * (K - 3)
    same = ((pos * (disjoint @ pos)).sum(0) + (neg * (disjoint @ neg)).sum(0)) / total
    opp = 2 * (pos * (disjoint @ neg)).sum(0) / total
    same, opp = float(same.mean()), float(opp.mean())
    tie = 1 - same - opp
    tie = 0.0 if abs(tie) < 1e-12 else tie
    dec = same + opp
    return {"same": same, "opposite": opp, "tie": tie, "C": same + 0.5 * tie, "decisive": dec,
            "conditional_agreement": same / dec if dec > 0 else float("nan"),
            "label": "per-model prompt draws reconstructed from the collected units; a separate "
                     "sampling procedure from panel-wide shared configurations; no calibrated interval"}


# --------------------------------------------------------------------------- Step 2


def _ratio(a, v):
    with np.errstate(divide="ignore", invalid="ignore"):
        a, v = np.asarray(a, float), np.asarray(v, float)
        return np.where(v > 0, -10 * a / np.where(v > 0, v, 1.0), np.nan)


def step2(Bd, Pd, sel, m) -> dict:
    """Fixed-panel moments from distinct discovery units, H, and their calibrated intervals."""
    G = Pd - Bd
    K = G.shape[0]
    cm = {"V_theta": iv.CrossMoment(Bd, Bd), "A": iv.CrossMoment(G, Bd), "V_delta": iv.CrossMoment(G, G)}
    est = {k: c.estimate() for k, c in cm.items()}
    loo = {k: c.loo() for k, c in cm.items()}
    bn = {k: c.boot(m, distinct=False) for k, c in cm.items()}
    bd = {k: c.boot(m, distinct=True) for k, c in cm.items()}
    bl = {k: c.boot_loo(m) for k, c in cm.items()}
    out = {"K": int(K)}
    for k in ("V_theta", "A"):
        out[k] = {"estimate": est[k], "method": sel.get(k),
                  "ci95": apply_ci(sel.get(k), est[k], loo[k], bn[k], bd[k], bl[k], m)}
    vm = sel.get("V_delta")
    vd_ci = (list(map(float, iv.mls_vdelta(G))) if vm == "mls"
             else apply_ci(vm, est["V_delta"], loo["V_delta"], bn["V_delta"], bd["V_delta"], bl["V_delta"], m))
    um = sel.get("V_delta_upper")
    vd_up = apply_upper(um, est["V_delta"], loo["V_delta"], bn["V_delta"], bd["V_delta"], bl["V_delta"], m, G)
    sd = lambda v: math.sqrt(v) if v is not None and np.isfinite(v) and v >= 0 else None
    out["V_delta"] = {"estimate": est["V_delta"], "negative_estimate": bool(est["V_delta"] < 0),
                      "method": vm, "ci95": vd_ci, "upper_method": um, "upper95": vd_up,
                      "gain_sd": sd(est["V_delta"]),
                      "gain_sd_ci95": None if vd_ci is None else [sd(vd_ci[0]), sd(vd_ci[1])],
                      "gain_sd_upper95": sd(vd_up),
                      "classification": classify_vdelta(None if vd_ci is None else vd_ci[0], vd_up)}
    H = float(_ratio(est["A"], est["V_theta"]))
    hm = sel.get("H")
    h = {"estimate": H, "denominator_nonpositive": bool(est["V_theta"] <= 0), "method": hm}
    if hm == "fieller":
        kind, a, b = iv.scale_fieller(*iv.fieller(est["A"], est["V_theta"], loo["A"], loo["V_theta"]), -10.0)
        h["set"] = {"kind": kind, "a": float(a), "b": float(b)}
        h["classification"] = classify_set(kind, a, b, PRACTICAL["H_pp_per_10pp"])
    else:
        ci = apply_ci(hm, H, _ratio(loo["A"], loo["V_theta"]), _ratio(bn["A"], bn["V_theta"]),
                      _ratio(bd["A"], bd["V_theta"]), _ratio(bl["A"], bl["V_theta"]), m)
        h["ci95"] = ci
        h["classification"] = ("uncalibrated" if ci is None
                               else classify_interval(ci[0], ci[1], PRACTICAL["H_pp_per_10pp"]))
    out["H"] = h
    gbar = G.mean(1)
    out["mean_gain"] = {"estimate": float(gbar.mean()), "se_over_units": float(gbar.std(ddof=1) / math.sqrt(K)),
                        "label": "descriptive; not a calibrated target"}
    out["mean_baseline"] = float(Bd.mean())
    rb, ra = reliability(G), reliability(Bd)
    out["noise"] = {"W_gain": rb["total"] - rb["signal"], "W_baseline": ra["total"] - ra["signal"],
                    "total_gain": rb["total"], "total_baseline": ra["total"]}
    out["reliability"] = {"R_benefit": rb["R"], "R_accuracy": ra["R"]}
    out["model_means"] = {"theta_hat": Bd.mean(0).tolist(), "delta_hat": G.mean(0).tolist()}
    return out


def panel_summary(Bd, Pd, sel, m, label) -> dict:
    """Step 1 and Step 2 headline quantities for one panel (used by the sensitivities)."""
    s1 = step1(Bd, Pd, sel, m, [str(k) for k in range(Bd.shape[1])])
    s2 = step2(Bd, Pd, sel, m)
    return {"label": label, "M": int(Bd.shape[1]),
            "C_benefit": s1["C_benefit"]["C"], "C_benefit_ci95": s1["C_benefit"]["ci95"],
            "C_accuracy": s1["C_accuracy"]["C"], "gap": s1["gap"]["estimate"], "gap_ci95": s1["gap"]["ci95"],
            "tie_benefit": s1["C_benefit"]["tie"],
            "V_delta": s2["V_delta"]["estimate"], "V_delta_ci95": s2["V_delta"]["ci95"],
            "V_delta_upper95": s2["V_delta"]["upper95"], "V_delta_class": s2["V_delta"]["classification"],
            "H": s2["H"]["estimate"], "H_set": s2["H"].get("set"), "H_class": s2["H"]["classification"],
            "mean_gain": s2["mean_gain"]["estimate"]}


# --------------------------------------------------------------------------- capability claims


def capability_claims(Bd, Pd, Bv, Pv, sel) -> dict:
    """Capability-claim comparison on independent evaluation blocks.

    Single-unit naive slopes: every distinct pair of collected units is a pair of independent
    one-unit blocks. Averaged OLS and the replicated estimate: the discovery and the validation
    blocks are the one available pair of independent K-unit blocks."""
    Bu, Pu = np.vstack([Bd, Bv]), np.vstack([Pd, Pv])
    naive = np.array([-10 * d1.slope(g, b) for g, b in zip(Pu - Bu, Bu)])
    ok = np.isfinite(naive) & (naive != 0)
    sg = np.sign(naive[ok])
    n = len(sg)
    pos, neg = int((sg > 0).sum()), int((sg < 0).sum())
    same = (pos * (pos - 1) + neg * (neg - 1)) / (n * (n - 1)) if n > 1 else float("nan")
    # Ties (zero slopes) reported separately, over distinct pairs of units with a defined slope.
    n_fin = int(np.isfinite(naive).sum())
    n_zero = int((naive[np.isfinite(naive)] == 0).sum())
    n_pairs = n_fin * (n_fin - 1) // 2
    p_same, p_opp = pos * (pos - 1) // 2 + neg * (neg - 1) // 2, pos * neg
    pair_fracs = ({"same_sign": p_same / n_pairs, "opposite_sign": p_opp / n_pairs,
                   "at_least_one_tie": (n_pairs - p_same - p_opp) / n_pairs} if n_pairs else None)
    out = {"naive_single_unit": {"n_units": int(len(naive)), "n_undefined_or_zero": int((~ok).sum()),
                                 "n_undefined": int(len(naive) - n_fin), "n_zero_slope": n_zero,
                                 "H_values": naive.tolist(),
                                 "sign_agreement_distinct_unit_pairs": float(same),
                                 "sign_agreement_note": "conditional on both slopes being nonzero and defined; "
                                                        "pairs_with_defined_slopes gives the tie fraction",
                                 "pairs_with_defined_slopes": {"n_pairs": n_pairs, **(pair_fracs or {})},
                                 "frac_abs_above_margin": float(np.mean(np.abs(naive[np.isfinite(naive)])
                                                                        > PRACTICAL["H_pp_per_10pp"]))}}
    blocks = {}
    for name, (B, P) in (("discovery", (Bd, Pd)), ("validation", (Bv, Pv))):
        G = P - B
        avg = -10 * d1.slope(G.mean(0), B.mean(0))
        cA, cV = iv.CrossMoment(G, B), iv.CrossMoment(B, B)
        rep = float(_ratio(cA.estimate(), cV.estimate()))
        kind, a, b = iv.scale_fieller(*iv.fieller(cA.estimate(), cV.estimate(), cA.loo(), cV.loo()), -10.0)
        blocks[name] = {"K": int(B.shape[0]), "averaged_ols_H": avg, "replicated_H": rep,
                        "replicated_fieller": {"kind": kind, "a": float(a), "b": float(b)},
                        "replicated_classification": classify_set(kind, a, b, PRACTICAL["H_pp_per_10pp"])}
    out["blocks"] = blocks
    for est in ("averaged_ols_H", "replicated_H"):
        x, y = blocks["discovery"][est], blocks["validation"][est]
        out[f"{est}_sign_agreement"] = (bool(np.sign(x) == np.sign(y))
                                        if np.isfinite(x) and np.isfinite(y) else None)
    out["note"] = ("The H interval method applies to the replicated estimate; the other estimators "
                   "have no calibrated interval. Sign agreement and interval-based conclusions are "
                   "recorded separately.")
    return out


# --------------------------------------------------------------------------- Step 3


def freeze_predictions(Bd, Pd, label, models, sampler_sha256, budget_list=None) -> dict:
    """Prediction rules fitted on discovery units only, at each nested budget."""
    budget_list = budget_list or budgets(Bd.shape[0])
    frozen = {"label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "sampler_sha256": sampler_sha256, "models": list(models), "units": "pp",
              "note": "fitted on discovery units only; written before any validation outcome is read",
              "budgets": {}}
    for k in budget_list:
        fp = d1.fit_predictors(Bd[:k], Pd[:k])
        frozen["budgets"][str(k)] = {n: {"prediction": v[0].tolist(), "info": v[1]} for n, v in fp.items()}
    frozen["sha256"] = sha256_json(json.loads(json.dumps(frozen["budgets"], default=_jsonable)))
    return frozen


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def write_frozen(frozen: dict, path, overwrite=False) -> dict:
    """Write the frozen predictions. An existing file with the same predictions is kept as is
    (its creation time stands); different predictions are refused unless overwrite is set."""
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("sha256") == frozen["sha256"] and old.get("models") == frozen["models"]:
            return old
        if not overwrite:
            raise FileExistsError(f"{path} holds different frozen predictions; refusing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=1, default=_jsonable)
    return frozen


def _cross_risk_loo(Z, f):
    K = Z.shape[0]
    return np.array([cross_risk(np.delete(Z, u, axis=0), f) for u in range(K)]) if K > 2 else None


def evaluate_frozen(frozen: dict, Zv, sel, k_registered=None) -> dict:
    """Risk differences of the frozen predictions on validation gains Zv (K_v, M)."""
    Zv = np.asarray(Zv, float)
    kv = Zv.shape[0]
    k_registered = str(k_registered or max(map(int, frozen["budgets"])))
    out = {"frozen_sha256": frozen["sha256"], "K_validation": int(kv), "K_discovery": int(k_registered),
           "registered": [], "budget_curve": [], "absolute_risk": []}
    for k, preds in sorted(frozen["budgets"].items(), key=lambda x: int(x[0])):
        P = {n: np.asarray(v["prediction"], float) for n, v in preds.items()}
        for f, g in d1.COMPARISONS:
            if f not in P or g not in P:
                continue
            L = paired_loss_difference(Zv, P[f], P[g])
            e = float(L.mean())
            loo = (L.sum() - L) / (kv - 1)
            meth = sel.get(f"risk:{f}-{g}")
            if meth not in (None, "jack_t"):
                raise NotImplementedError(f"risk-difference method {meth!r}")
            ci = apply_ci(meth, e, loo, None, None, None, None)
            row = {"K": int(k), "f": f, "g": g, "estimate": e, "method": meth, "ci95": ci,
                   "classification": ("uncalibrated" if ci is None
                                      else classify_interval(ci[0], ci[1], PRACTICAL["risk_pp2"]))}
            (out["registered"] if k == k_registered else out["budget_curve"]).append(row)
        if k == k_registered:
            for n, f in P.items():
                r = cross_risk(Zv, f)
                loo = _cross_risk_loo(Zv, f)
                # The square root only with a supported nonnegative interpretation. The absolute risk has
                # no calibrated interval (the calibration covered risk differences), so none is shown.
                out["absolute_risk"].append({"predictor": n, "cross_risk": r,
                                             "jackknife_se": iv.jackknife_se(loo) if loo is not None else None,
                                             "sqrt": None,
                                             "sqrt_note": "not shown: no calibrated interval supports a nonnegative "
                                                          "interpretation (plan C.4)",
                                             "naive_mse_includes_validation_noise": float(np.mean((Zv - f) ** 2)),
                                             "label": "cross-product risk; interval not calibrated. "
                                                      "The naive MSE against noisy validation gains includes "
                                                      "validation noise and is not prediction error"})
    out["note"] = ("Positive differences favour g. Budget-curve points reuse nested discovery units and "
                   "are dependent; no smallest budget is selected without calibrated simultaneous "
                   "inference.")
    return out


def validation_agreement_curve(Bv, Pv, k_list=None) -> list[dict]:
    """Agreement between two disjoint groups of K validation units, one comparison per K (C.6)."""
    kv = Bv.shape[0]
    k_list = k_list or budgets(kv // 2)
    rows = []
    for k in k_list:
        a, b = slice(0, k), slice(k, 2 * k)
        Gv = Pv - Bv
        rows.append({"K": int(k), "benefit": d1.agreement_between(Gv[a].mean(0), Gv[b].mean(0)),
                     "accuracy": d1.agreement_between(Bv[a].mean(0), Bv[b].mean(0))})
    return rows


def gaussian_reference(Bd, Pd, k_list) -> dict:
    """Conditional Gaussian reference (C.1), inputs from discovery units only (supplementary)."""
    Rb, Ra = reliability(Pd - Bd)["R"], reliability(Bd)["R"]
    curve = [{"K": int(k), "benefit": gaussian_agreement(Rb, k) if 0 <= Rb <= 1 else None,
              "accuracy": gaussian_agreement(Ra, k) if 0 <= Ra <= 1 else None} for k in k_list]
    return {"R_benefit_discovery": Rb, "R_accuracy_discovery": Ra, "curve": curve,
            "note": "conditional Gaussian reference; R outside [0, 1] is reported and no curve is drawn"}


# --------------------------------------------------------------------------- Section 9


def family_sensitivities(Bd, Pd, families, sel, m) -> list[dict]:
    """Leave-one-family-out and family-balanced (family-mean units) estimates."""
    fam = np.asarray(families)
    out = []
    for f in sorted(set(families)):
        keep = fam != f
        if keep.sum() >= 3:
            out.append(panel_summary(Bd[:, keep], Pd[:, keep], sel, m, f"leave out family {f}"))
        else:
            out.append({"label": f"leave out family {f}", "M": int(keep.sum()),
                        "note": "fewer than three models remain; not estimated"})
    fs = sorted(set(families))
    if len(fs) >= 3:
        Bf = np.stack([Bd[:, fam == f].mean(1) for f in fs], 1)
        Pf = np.stack([Pd[:, fam == f].mean(1) for f in fs], 1)
        out.append(panel_summary(Bf, Pf, sel, m, f"family-balanced ({len(fs)} family-mean units)"))
    return out


def compliance_summary(cells: pd.DataFrame, units, panel, tiers=None, arms=ARMS) -> dict:
    """Format compliance (share of outputs with a strictly extractable answer in the required
    format) and the strict/lenient gap, by model and arm and, when tiers are given, by tier
    and arm. Means over the listed units with equal weight per unit."""
    rows = []
    for mk in panel:
        for arm in arms:
            x = cells.loc[[(mk, arm, str(c)) for c in units]]
            rows.append({"model_key": mk, "arm": arm, "tier": None if tiers is None else tiers.get(mk),
                         "compliance_pct": float(x["pct_compliant"].mean()),
                         "strict_pct": float(x["pct_strict_correct"].mean()),
                         "lenient_pct": float(x["pct_lenient_correct"].mean())})
    df = pd.DataFrame(rows)
    df["lenient_minus_strict_pp"] = df["lenient_pct"] - df["strict_pct"]
    out = {"by_model_arm": df.to_dict("records")}
    if tiers is not None:
        t = df.groupby(["tier", "arm"], as_index=False)[
            ["compliance_pct", "strict_pct", "lenient_pct", "lenient_minus_strict_pp"]].mean()
        out["by_tier_arm"] = t.to_dict("records")
    else:
        out["by_tier_arm"] = None
        out["note"] = "model tiers are not given; by-tier summary omitted"
    return out
