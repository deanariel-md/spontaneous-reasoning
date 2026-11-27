"""Analysis T across tasks: which rule best predicts a model's benefit on a
task, scored on that task's validation draws.

Every rule predicts the target's model-centred benefit (the ordering and spacing of models, not the task's mean
gain, which no rule from another task can know). Rules, all fitted on discovery draws only:

- constant: zero centred benefit;
- transfer: the model's centred mean gain on the source task (src/tasks_d2.transfer_prediction);
- rescaled transfer: the source's centred mean gains divided by their SD across models and
  multiplied by the SD across models of the target's discovery mean gains (both with denominator M - 1), so the
  source supplies the ordering and spacing and the target's discovery draws supply the scale;
- capability (ECI): the source's OLS slope of centred gain on centred ECI, applied to centred ECI;
- capability (baseline): the source's OLS slope of centred gain on centred direct-arm accuracy, applied to the
  source's centred baseline (the "weaker models benefit more" rule carried to a new task);
- pilot K: the target's own centred mean gain over its first K discovery draws (K = 1, 2, 4), what a short pilot on
  the new task would give;
- own (24): the target's own centred mean gain over all 24 discovery draws, a reference.

Scores use the paired loss differences and cross-product risks on the target's validation draws
(src/estimators.py), with a leave-one-draw-out jackknife; the calibration did not cover these targets, so the
intervals are descriptive.

cross_task_correlation gives the correlation of model gains between two tasks on joint bundles (bundle r pairs the
r-th draw of each task): Pearson and Spearman of the mean gains, and the latent correlation, the cross-draw moment of
the two tasks' gains over the root of their V_delta (all three cross-draw moments, distinct bundles only).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .estimators import cross_moment, cross_risk, paired_loss_difference
from .tasks_d2 import UNCALIBRATED, centred

PILOT_K = (1, 2, 4)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Slope of y on x across models; None when x has no spread (reported, never imputed)."""
    x, y = centred(np.asarray(x, float)), centred(np.asarray(y, float))
    sxx = float(x @ x)
    return None if sxx <= 0 else float(x @ y) / sxx


def rules(source_gain: np.ndarray, source_base: np.ndarray, target_disc_gain: np.ndarray,
          eci: Sequence[float]) -> dict[str, dict]:
    """Frozen prediction vectors (centred, pp) from discovery arrays (K, M)."""
    g = source_gain.mean(0)
    b = source_base.mean(0)
    e = np.asarray(eci, float)
    out = {"constant": {"prediction": np.zeros(g.shape[0])},
           "transfer": {"prediction": centred(g)}}
    sd_s = float(centred(g).std(ddof=1))
    sd_t = float(centred(target_disc_gain.mean(0)).std(ddof=1))
    out["transfer_rescaled"] = {"sd_source_pp": sd_s, "sd_target_discovery_pp": sd_t,
                                "prediction": None if sd_s <= 0 else centred(g) / sd_s * sd_t}
    s_e = ols_slope(e, g)
    out["capability_eci"] = {"slope_pp_per_eci": s_e,
                             "prediction": None if s_e is None else s_e * centred(e)}
    s_b = ols_slope(b, g)
    out["capability_baseline"] = {"slope_pp_per_pp": s_b,
                                  "prediction": None if s_b is None else s_b * centred(b)}
    for k in PILOT_K:
        out[f"pilot_{k}"] = {"prediction": centred(target_disc_gain[:k].mean(0)), "K_pilot": k}
    out["own_24"] = {"prediction": centred(target_disc_gain.mean(0)), "K_pilot": int(target_disc_gain.shape[0])}
    return out


def _jack_from(est: float, loo: np.ndarray) -> dict:
    from . import intervals as iv
    lo, hi = iv.jack_t(est, loo)
    return {"estimate": float(est), "jackknife_se": iv.jackknife_se(loo), "ci95": [float(lo), float(hi)],
            "method": "jack_t", "note": UNCALIBRATED}


def _jack_diff(L: np.ndarray) -> dict:
    kv = L.shape[0]
    est = float(L.mean())
    if kv <= 2:
        return {"estimate": est, "jackknife_se": None, "ci95": None, "method": None}
    return _jack_from(est, (L.sum() - L) / (kv - 1))


def score(preds: Mapping[str, Sequence[float] | None], target_validation: np.ndarray,
          comparisons: Sequence[tuple[str, str]]) -> dict:
    """Risks of every rule and the paired differences of the listed (f, g) pairs (positive favours g)."""
    Z = centred(target_validation)
    risks = {n: (None if p is None else cross_risk(Z, np.asarray(p, float))) for n, p in preds.items()}
    diffs = []
    for f, g in comparisons:
        if preds.get(f) is None or preds.get(g) is None:
            diffs.append({"f": f, "g": g, "estimate": None, "note": "a rule has no prediction"})
            continue
        L = paired_loss_difference(Z, np.asarray(preds[f], float), np.asarray(preds[g], float))
        diffs.append({"f": f, "g": g, **_jack_diff(L)})
    return {"K_validation": int(Z.shape[0]), "risk_pp2": risks, "differences": diffs,
            "note": "risks are cross-product risks in pp^2 (may be negative; never truncated); a positive difference "
                    "favours g"}


COMPARISONS = (("constant", "transfer"), ("constant", "capability_eci"), ("constant", "capability_baseline"),
               ("capability_baseline", "transfer"), ("capability_eci", "transfer"),
               ("transfer", "pilot_1"), ("transfer", "pilot_2"), ("transfer", "pilot_4"), ("transfer", "own_24"),
               ("constant", "pilot_1"), ("pilot_4", "own_24"),
               ("constant", "transfer_rescaled"), ("transfer", "transfer_rescaled"),
               ("capability_baseline", "transfer_rescaled"), ("transfer_rescaled", "pilot_1"),
               ("transfer_rescaled", "pilot_4"))


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties at their average rank."""
    from scipy.stats import rankdata
    return rankdata(x)


def _corr_point(G: np.ndarray, B: np.ndarray) -> dict:
    g, b = G.mean(0), B.mean(0)
    vg, vb, c = cross_moment(G, G), cross_moment(B, B), cross_moment(G, B)
    sg, sb = float(np.std(g, ddof=1)), float(np.std(b, ddof=1))
    rg, rb = _rank(g), _rank(b)
    return {"pearson": float(np.corrcoef(g, b)[0, 1]) if sg > 0 and sb > 0 else None,
            "spearman": float(np.corrcoef(rg, rb)[0, 1]) if np.std(rg) > 0 and np.std(rb) > 0 else None,
            "latent": c / float(np.sqrt(vg * vb)) if vg > 0 and vb > 0 else None,
            "V_delta_a": vg, "V_delta_b": vb, "cross_moment": c, "sd_mean_gain_a": sg, "sd_mean_gain_b": sb}


def cross_task_correlation(G: np.ndarray, B: np.ndarray) -> dict:
    """Correlation of model gains between two tasks, from gains (K, M) of each in bundle order.

    Point values from all K bundles; leave-one-bundle-out jackknife t intervals (descriptive). The latent
    correlation is not bounded to [-1, 1] and is reported as it is; it is None when either V_delta is not
    positive (that fold's jackknife then has no interval)."""
    G, B = np.asarray(G, float), np.asarray(B, float)
    if G.shape != B.shape:
        raise ValueError("both tasks need the same bundles and models")
    K = G.shape[0]
    est = _corr_point(G, B)
    loo = [_corr_point(np.delete(G, u, 0), np.delete(B, u, 0)) for u in range(K)] if K > 2 else None
    out = {"K_bundles": int(K), "M": int(G.shape[1])}
    for k, v in est.items():
        vals = None if loo is None else [r[k] for r in loo]
        if v is None or vals is None or any(x is None for x in vals):
            out[k] = {"estimate": v, "jackknife_se": None, "ci95": None, "method": None}
        else:
            out[k] = _jack_from(v, np.asarray(vals, float))
    return out


def profile_share(v_a: float, v_b: float, rho: float) -> float | None:
    """Share of the two-task average profile in true within-task variation, equal task weights:
    (V_a + V_b + 2 rho sqrt(V_a V_b)) / (2 (V_a + V_b)). At least 1/2 for rho >= 0; at rho = 1 it is
    (s_a + s_b)^2 / (2 (s_a^2 + s_b^2)), below 1 when the tasks' spreads differ."""
    if v_a <= 0 or v_b <= 0:
        return None
    return (v_a + v_b + 2 * rho * np.sqrt(v_a * v_b)) / (2 * (v_a + v_b))
