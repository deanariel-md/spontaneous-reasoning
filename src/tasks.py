"""Secondary Dataset 1 analyses across the observed task collection.

Each task t has its own enumerated configuration bank with arrays B_t, P_t of shape
(configurations_t, models) in pp, for a common fixed panel. Task weights macro-average first
within source family and then across families. Targets are exact conditional values under
uniform with-replacement sampling within each task's bank; nothing here describes future tasks.
"""
from __future__ import annotations

import numpy as np

from .dataset1 import pvar


def family_weights(families: list[str]) -> np.ndarray:
    """Equal weight per family, split equally across that family's tasks."""
    fams = sorted(set(families))
    n = {f: families.count(f) for f in fams}
    w = np.array([1 / (len(fams) * n[f]) for f in families])
    return w / w.sum()


def observed_task_decomposition(banks: list[tuple[np.ndarray, np.ndarray]], weights) -> dict:
    """Exact fixed-task decomposition (C.3):
    sum_t w_t Var_M(d_t) = Var_M(m) + sum_t w_t Var_M(h_t), with d_t = delta_t centered across
    models, m = sum_t w_t d_t, h_t = d_t - m; plus mean within-task prompt-noise variance."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    deltas = np.stack([(P - B).mean(0) for B, P in banks])            # (T, M)
    within = float(sum(wt * pvar(dt) for wt, dt in zip(w, deltas)))
    m = w @ deltas
    profile = pvar(m)
    noise = float(sum(wt * np.mean(np.var((P - B) - (P - B).mean(0), axis=1, ddof=1))
                      for wt, (B, P) in zip(w, banks)))
    return {"average_task_profile": profile, "model_by_task": within - profile,
            "true_within_task": within, "prompt_noise": noise, "total": within + noise,
            "model_profile_m": m.tolist(),
            "share_profile_of_true": profile / within if within > 0 else float("nan")}


def transfer_holdout(banks, families, weights_within=None) -> list[dict]:
    """C.5: for each held-out source family, predict its model-centered benefit from each
    model's weighted average centered gain on the training families, against zero centered
    benefit. Exact conditional risks in pp^2; positive `risk_zero - risk_transfer` favors
    transfer. Requires at least four families."""
    fams = sorted(set(families))
    if len(fams) < 4:
        return []
    deltas = np.stack([(P - B).mean(0) for B, P in banks])
    d = deltas - deltas.mean(1, keepdims=True)
    out = []
    for f in fams:
        tr = [k for k, x in enumerate(families) if x != f]
        te = [k for k, x in enumerate(families) if x == f]
        w_tr = family_weights([families[k] for k in tr])
        w_te = np.ones(len(te)) / len(te)
        pred = w_tr @ d[tr]
        target = w_te @ d[te]
        r_zero = float(np.mean(target ** 2))
        r_tr = float(np.mean((target - pred) ** 2))
        out.append({"held_out_family": f, "n_train_tasks": len(tr), "n_test_tasks": len(te),
                    "risk_zero": r_zero, "risk_transfer": r_tr, "difference": r_zero - r_tr,
                    "target_centered_benefit": target.tolist(), "prediction": pred.tolist()})
    return out
