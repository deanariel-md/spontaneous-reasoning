"""w under five-shot answering.

The five-shot requested arm (results/fewshot_requested/design.md) against the zero-shot requested arm, on
the same models, configurations and items. w is the share of requested replies longer than the forced arm's output cap
(16 generated tokens on GSM8K and BIG-Bench Hard, n_output_tokens compared with the cap by '>'), computed per
configuration and averaged over the configurations, as src/baseline_mixture.draw_weighted_share computes it for the main
analyses. Strict-format compliance is the share of replies the strict reader extracts an answer from (the scored
tables' `compliant` column), averaged the same way.

The five-shot effect of a model within a baseline is its five-shot minus zero-shot accuracy under that baseline, the
model mean over the configurations (g_R within the requested baseline, g_F within the forced one; src/fewshot_requested
.gains). The sign-change variable is binary: S = 1 when g_R and g_F have opposite signs, S = 0 otherwise. A gain within
1e-9 pp of zero has sign 0 (src/estimators.tie_sign, D8) and its model gets S = 0; such models are counted apart.

The fall in w is w_zero-shot - w_five-shot (positive when w fell). Its association with S across models: Spearman's rho
and Kendall's tau-b (scipy), the number of models, the mean fall among models with S = 1 minus the mean among models
with S = 0, and percentile intervals from a model bootstrap (models drawn with replacement; a resample in which either
variable is constant leaves the rank statistics undefined, and such resamples are counted, not dropped silently).

Pure functions; scripts/run_fewshot_w.py loads the data. Shares in [0, 1], gains in pp.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from src import baseline_mixture as bm
from src import estimators as est

N_BOOT = 9999
SEED = 20261127
TIE_TOL = est.TIE_TOL


def config_mean_share(flags, model_idx, config_idx, n_models: int, n_configs: int) -> np.ndarray:
    """(n_models,) share of flagged rows per model: the share within each configuration, averaged over the
    configurations (each once). nan for a model with no row in some configuration (reported, not imputed)."""
    return bm.draw_weighted_share(flags, model_idx, config_idx, n_models, n_configs, list(range(int(n_configs))))


def long_share(n_tokens, cap: int, model_idx, config_idx, n_models: int, n_configs: int) -> np.ndarray:
    """(n_models,) w: share of replies with more than `cap` generated tokens, configuration means averaged."""
    return config_mean_share(np.asarray(n_tokens) > int(cap), model_idx, config_idx, n_models, n_configs)


def signs(g) -> np.ndarray:
    """Sign of each gain with |g| <= 1e-9 pp a tie (0)."""
    return est.tie_sign(np.asarray(g, float))


def sign_change(g_req, g_fd) -> np.ndarray:
    """(M,) int: 1 when the five-shot effect has opposite signs within the two baselines, else 0 (a zero gain is not a
    sign change)."""
    return (signs(g_req) * signs(g_fd) < 0).astype(int)


def _rank_stats(x, y):
    """(spearman, kendall tau-b) or (nan, nan) when either variable is constant or fewer than 3 models."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    rho = stats.spearmanr(x, y).statistic
    tau = stats.kendalltau(x, y, variant="b").statistic
    return float(rho), float(tau)


def _group_diff(x, s):
    """Mean of x among s == 1 minus mean among s == 0; nan when a group is empty."""
    x, s = np.asarray(x, float), np.asarray(s, int)
    if not (s == 1).any() or not (s == 0).any():
        return float("nan")
    return float(x[s == 1].mean() - x[s == 0].mean())


def _pct(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return None
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def association(fall, s, n_boot: int = N_BOOT, rng=None, cont=None) -> dict:
    """Rank association across models between the fall in w and the binary sign-change variable s, with a model
    bootstrap. `cont` (optional): a continuous variable (the change in the five-shot effect between baselines, g_R -
    g_F) whose rank association with the fall in w is reported beside, from the same resamples."""
    fall, s = np.asarray(fall, float), np.asarray(s, int)
    if not np.isfinite(fall).all():
        raise ValueError("the fall in w is not finite for every model")
    M = len(fall)
    rng = np.random.default_rng(SEED) if rng is None else rng
    rho, tau = _rank_stats(fall, s)
    out = {"n_models": int(M), "n_sign_change": int(s.sum()), "n_no_change": int(M - s.sum()),
           "variable": "binary", "spearman": rho, "kendall_tau_b": tau,
           "mean_fall_sign_change": float(fall[s == 1].mean()) if s.any() else None,
           "mean_fall_no_change": float(fall[s == 0].mean()) if (s == 0).any() else None,
           "mean_fall_difference": _group_diff(fall, s), "n_boot": int(n_boot)}
    if cont is not None:
        cont = np.asarray(cont, float)
        c_rho, c_tau = _rank_stats(fall, cont)
        out["continuous"] = {"spearman": c_rho, "kendall_tau_b": c_tau}
    if n_boot <= 0:
        return out
    br, bt, bd, cr, ct = (np.full(n_boot, np.nan) for _ in range(5))
    for b in range(n_boot):
        ix = rng.integers(0, M, M)
        br[b], bt[b] = _rank_stats(fall[ix], s[ix])
        bd[b] = _group_diff(fall[ix], s[ix])
        if cont is not None:
            cr[b], ct[b] = _rank_stats(fall[ix], cont[ix])
    out["spearman_ci95"], out["kendall_tau_b_ci95"] = _pct(br), _pct(bt)
    out["mean_fall_difference_ci95"] = _pct(bd)
    out["boot_undefined_rank"] = int((~np.isfinite(br)).sum())
    out["boot_undefined_difference"] = int((~np.isfinite(bd)).sum())
    if cont is not None:
        out["continuous"]["spearman_ci95"], out["continuous"]["kendall_tau_b_ci95"] = _pct(cr), _pct(ct)
        out["continuous"]["boot_undefined"] = int((~np.isfinite(cr)).sum())
    return out


def mean_ci(x, n_boot: int = N_BOOT, rng=None) -> dict:
    """Mean over models with a percentile model-bootstrap interval."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED) if rng is None else rng
    if n_boot <= 0:
        return {"mean": float(x.mean()), "ci95": None}
    ix = rng.integers(0, len(x), (n_boot, len(x)))
    return {"mean": float(x.mean()), "ci95": _pct(x[ix].mean(1))}


def distribution(x) -> dict:
    """Median, minimum, maximum and counts below, at and above zero (exact zero only) of a per-model quantity."""
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()),
            "n_below_zero": int((x < 0).sum()), "n_zero": int((x == 0).sum()), "n_above_zero": int((x > 0).sum())}


def direction_counts(g_req, g_fd) -> dict:
    """Models by the signs of the two five-shot effects."""
    sr, sf = signs(g_req), signs(g_fd)
    return {"forced_up_requested_down": int(((sf > 0) & (sr < 0)).sum()),
            "forced_down_requested_up": int(((sf < 0) & (sr > 0)).sum()),
            "both_up": int(((sf > 0) & (sr > 0)).sum()), "both_down": int(((sf < 0) & (sr < 0)).sum()),
            "zero_under_either": int(((sf == 0) | (sr == 0)).sum())}
