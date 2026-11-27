"""Secondary Dataset 2 analyses across tasks.

Two tasks: GSM8K and BIG-Bench Hard (its 25 tasks pooled with equal weights, as in its primary score), with
weights of one half each. Within BIG-Bench Hard: its 25 tasks, with families frozen from the task names
(`logical_deduction_*` and `tracking_shuffled_objects_*` are one family each; every other task is its own).

Arrays are (units, models) in pp, from the frozen samplers' units on a common panel. The decomposition uses the
estimator for independent joint draw bundles (src/estimators.fixed_task_components): within BIG-Bench
Hard a unit is one configuration applied to all 25 tasks; across the two tasks the r-th GSM8K unit and the r-th
BIG-Bench Hard unit form bundle r, two independent draws from two banks. It reads discovery units only, as Steps 1
and 2 do. Transfer predicts a target's model-centred benefit from each model's centred gain on the source, fitted on
discovery units and frozen before any validation unit is read, and scores it on the target's validation units
against zero centred benefit with the paired loss differences and cross-product risks. Nothing here
describes future tasks; the targets are observed cases.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import intervals as iv
from .estimators import cross_moment, cross_risk, fixed_task_components, paired_loss_difference
from .tasks import family_weights

FAMILY_PREFIXES = ("logical_deduction_", "tracking_shuffled_objects_")
TWO_TASK_WEIGHTS = {"gsm8k": 0.5, "bbh": 0.5}
UNCALIBRATED = ("jackknife over units; the calibration did not cover this target, so "
                "the interval is descriptive")


def bbh_family(task: str) -> str:
    """The frozen family of a BIG-Bench Hard task: the shared prefix for the two multi-variant tasks."""
    for p in FAMILY_PREFIXES:
        if task.startswith(p):
            return p.rstrip("_")
    return task


def centred(x: np.ndarray) -> np.ndarray:
    """Centred across models (the last axis)."""
    x = np.asarray(x, float)
    return x - x.mean(axis=-1, keepdims=True)


def _jack(est: float, loo: np.ndarray | None) -> dict:
    if loo is None:
        return {"estimate": est, "jackknife_se": None, "ci95": None, "method": None}
    lo, hi = iv.jack_t(est, loo)
    return {"estimate": est, "jackknife_se": iv.jackknife_se(loo), "ci95": [float(lo), float(hi)],
            "method": "jack_t", "note": UNCALIBRATED}


def decomposition(gains: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> dict:
    """C.3 on joint bundles: gains {task: (K, M)} with one row per bundle in the same order for every task.

    The components: the average task profile Var_M(m), model-by-task variation, true within-task
    variation and prompt noise, in pp^2, each with a leave-one-bundle-out jackknife. Negative estimates are reported
    as they are. Per task: its true benefit variance (cross-draw moment) and mean gain."""
    names = list(gains)
    G = np.stack([np.asarray(gains[t], float) for t in names])            # (T, K, M)
    w = np.array([float(weights[t]) for t in names])
    est = fixed_task_components(G, w)
    K = G.shape[1]
    loo = [fixed_task_components(np.delete(G, u, axis=1), w) for u in range(K)] if K > 2 else None
    out = {"tasks": names, "weights": dict(zip(names, (w / w.sum()).tolist())), "K_bundles": int(K),
           "M": int(G.shape[2]), "units": "pp^2"}
    for k in ("average_task_profile", "model_by_task", "true_within_task", "prompt_noise", "total"):
        out[k] = _jack(float(est[k]), None if loo is None else np.array([r[k] for r in loo]))
    within = est["true_within_task"]
    out["share_profile_of_true_within"] = est["average_task_profile"] / within if within > 0 else None
    out["share_note"] = ("not computed: true within-task variation is not positive" if within <= 0 else
                         "point ratio of the two components; no interval")
    out["per_task"] = [{"task": t, "V_delta": cross_moment(G[i], G[i]), "mean_gain": float(G[i].mean())}
                       for i, t in enumerate(names)]
    return out


def transfer_prediction(source: np.ndarray, weights: Sequence[float] | None = None) -> np.ndarray:
    """Each model's centred mean gain on the source: (K, M) for one source, or (T, K, M) with task weights."""
    s = np.asarray(source, float)
    if s.ndim == 2:
        return centred(s.mean(0))
    w = np.asarray(weights, float)
    return centred(np.einsum("t,tm->m", w / w.sum(), s.mean(1)))


def transfer_score(prediction: np.ndarray, target_validation: np.ndarray) -> dict:
    """C.5 on the target's validation units Z (Kv, M), centred per unit: the paired loss difference against zero
    centred benefit (positive favours transfer) with a jackknife over units, and the cross-product risks of both
    rules (may be negative; no square root is shown)."""
    Z = centred(target_validation)
    f = centred(prediction)
    zero = np.zeros_like(f)
    L = paired_loss_difference(Z, zero, f)
    kv = Z.shape[0]
    loo = (L.sum() - L) / (kv - 1) if kv > 2 else None
    return {"K_validation": int(kv), "difference": _jack(float(L.mean()), loo),
            "risk_zero": cross_risk(Z, zero), "risk_transfer": cross_risk(Z, f),
            "target_centred_gain": Z.mean(0).tolist(), "prediction": f.tolist(),
            "note": "positive difference favours transfer over zero centred benefit; risks in pp^2"}


def within_bbh_transfer_predictions(discovery: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """For each held-out family: the prediction from the other families' tasks on discovery units, with task
    weights that macro-average within family and then across families."""
    tasks = list(discovery)
    fams = [bbh_family(t) for t in tasks]
    out = {}
    for fam in sorted(set(fams)):
        train = [t for t, f in zip(tasks, fams) if f != fam]
        w = family_weights([bbh_family(t) for t in train])
        out[fam] = {"held_out_tasks": [t for t, f in zip(tasks, fams) if f == fam], "train_tasks": train,
                    "train_weights": w.tolist(),
                    "prediction": transfer_prediction(np.stack([discovery[t] for t in train]), w).tolist()}
    return out


def held_out_target(validation: Mapping[str, np.ndarray], held_out: Sequence[str]) -> np.ndarray:
    """The held-out family's validation gains, its tasks weighted equally."""
    return np.mean(np.stack([np.asarray(validation[t], float) for t in held_out]), axis=0)
