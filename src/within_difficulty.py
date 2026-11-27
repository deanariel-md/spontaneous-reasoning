"""Does a model write long requested answers more often on harder questions, within itself?

Long: a requested-direct output longer than the forced arm's cap (n_output_tokens > 16), the definition behind w
(src/baseline_mixture.RULE_TEXT). Difficulty of item i for model m: 100 x the share of the other models' forced answers
on item i that are wrong (leave-this-model-out), in percentage points; a larger value is a harder item. The model is
a binomial logit of long on difficulty with one intercept per model (model fixed effects) and one common slope;
optionally one effect per item group (BIG-Bench Hard subtask). The odds ratio per 10 points of difficulty is
exp(slope), the slope taken on difficulty / 10.

Pure functions on (M, I) arrays (models x items) of weighted counts; scripts/run_within_difficulty.py builds them.
Edge cases are returned as they are: a model whose outputs are all long or all short has no finite intercept and is
left out of the pooled fit (listed by the caller); a fit that does not converge is flagged, not dropped.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

MAX_ITER = 200
TOL = 1e-9
SEPARATION_SLOPE = 50.0   # |slope| on the logit scale per 10 points beyond this is read as separation (no finite MLE)


def loo_difficulty(F, N) -> np.ndarray:
    """(M, I) difficulty in pp: 100 x (1 - the other models' forced accuracy on the item).
    F: (M, I) forced correct counts (weighted); N: (M, I) output counts (weighted). A column whose other models have
    no outputs gives nan (reported, not imputed)."""
    F, N = np.asarray(F, float), np.asarray(N, float)
    den = N.sum(0)[None, :] - N
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = (F.sum(0)[None, :] - F) / den
    acc[den <= 0] = np.nan
    return 100.0 * (1.0 - acc)


def variation(K, N) -> np.ndarray:
    """(M,) 'none_long', 'all_long' or 'varies' for each model."""
    k, n = np.asarray(K, float).sum(1), np.asarray(N, float).sum(1)
    return np.where(k <= 0, "none_long", np.where(k >= n, "all_long", "varies"))


def _loglik(k, n, eta):
    return float(np.sum(k * eta - n * np.logaddexp(0.0, eta)))


def fe_logit(K, N, X, group=None, theta0=None, max_iter: int = MAX_ITER, tol: float = TOL) -> dict:
    """Binomial logit of K successes out of N on X with one intercept per row (model) and one common slope; with
    `group` ((I,) integers), also one effect per group of columns, the first group the reference.
    K, N, X: (M, I). Newton steps with step halving; converged when the largest step is below `tol`.
    Returns {'slope', 'alpha' (M,), 'gamma' (G-1,), 'converged', 'iterations', 'loglik', 'separation'}."""
    K, N, X = (np.asarray(a, float) for a in (K, N, X))
    M, I = K.shape
    k, n, x = K.ravel(), N.ravel(), X.ravel()
    a = np.repeat(np.arange(M), I)
    if group is not None:
        g = np.asarray(group, int)
        G = int(g.max()) + 1
        b = np.tile(g, M)
    else:
        G, b = 1, None
    P = M + (G - 1) + 1
    if theta0 is None:
        ks, ns = K.sum(1), N.sum(1)
        p0 = np.clip(ks / ns, 1e-6, 1 - 1e-6)
        theta = np.concatenate([np.log(p0 / (1 - p0)), np.zeros(G - 1), [0.0]])
    else:
        theta = np.asarray(theta0, float).copy()

    def eta_of(th):
        e = th[:M][a] + th[-1] * x
        if b is not None:
            e = e + np.concatenate([[0.0], th[M:M + G - 1]])[b]
        return e

    eta = eta_of(theta)
    ll = _loglik(k, n, eta)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        p = expit(eta)
        Wt = n * p * (1 - p)
        r = k - n * p
        grad = np.empty(P)
        H = np.zeros((P, P))
        grad[:M] = np.bincount(a, r, M)
        grad[-1] = np.sum(r * x)
        H[np.arange(M), np.arange(M)] = np.bincount(a, Wt, M)
        hab = np.bincount(a, Wt * x, M)
        H[:M, -1] = H[-1, :M] = hab
        H[-1, -1] = np.sum(Wt * x * x)
        if b is not None:
            grad[M:M + G - 1] = np.bincount(b, r, G)[1:]
            hgg = np.bincount(b, Wt, G)[1:]
            H[np.arange(M, M + G - 1), np.arange(M, M + G - 1)] = hgg
            hag = np.bincount(a * G + b, Wt, M * G).reshape(M, G)[:, 1:]
            H[:M, M:M + G - 1] = hag
            H[M:M + G - 1, :M] = hag.T
            hgb = np.bincount(b, Wt * x, G)[1:]
            H[M:M + G - 1, -1] = H[-1, M:M + G - 1] = hgb
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(40):
            th_new = theta + t * step
            eta_new = eta_of(th_new)
            ll_new = _loglik(k, n, eta_new)
            if ll_new >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            t *= 0.5
        theta, eta, ll = th_new, eta_new, ll_new
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    sep = bool(abs(theta[-1]) > SEPARATION_SLOPE or not np.all(np.isfinite(theta)))
    return {"slope": float(theta[-1]), "alpha": theta[:M].copy(), "gamma": theta[M:M + G - 1].copy(),
            "theta": theta.copy(), "converged": bool(converged and not sep), "iterations": int(it), "loglik": ll,
            "separation": sep}


def per_model_slopes(K, N, X, group=None) -> list[dict]:
    """One logit of long on X per model (intercept and slope; with `group`, also one effect per group). Models with
    no variation in long get slope None. With `group`, a group in which the model's outputs are all long or all short
    carries no information on the slope once its effect is in the model (its effect has no finite estimate), so it
    is left out of that model's fit; 'groups_dropped' counts them."""
    K, N, X = (np.asarray(a, float) for a in (K, N, X))
    out = []
    var = variation(K, N)
    for m in range(K.shape[0]):
        if var[m] != "varies":
            out.append({"variation": str(var[m]), "slope": None, "converged": None, "separation": None,
                        "iterations": 0, "groups_dropped": None})
            continue
        k, n, x, g, dropped = K[m:m + 1], N[m:m + 1], X[m:m + 1], None, 0
        if group is not None:
            grp = np.asarray(group, int)
            ks, ns = np.bincount(grp, k[0]), np.bincount(grp, n[0])
            keep_g = np.flatnonzero((ks > 0) & (ks < ns))
            dropped = int(((ns > 0) & ~((ks > 0) & (ks < ns))).sum())
            cols = np.isin(grp, keep_g)
            k, n, x = k[:, cols], n[:, cols], x[:, cols]
            g = np.searchsorted(keep_g, grp[cols]) if len(keep_g) > 1 else None
        f = fe_logit(k, n, x, g)
        out.append({"variation": "varies", "slope": f["slope"], "converged": f["converged"],
                    "separation": f["separation"], "iterations": f["iterations"], "groups_dropped": dropped})
    return out


def model_bootstrap(K, N, X, n_boot: int, rng, group=None, fit=None) -> dict:
    """Percentile model bootstrap of the pooled slope: draw the models (rows) with replacement, a model drawn twice
    entering twice with its own intercept; X is held as computed on the full panel. Starts each refit at the
    full-sample estimate. Returns {'slopes' (n_boot,), 'not_converged': int, 'not_converged_group_without_variation':
    int}: the second counts non-converged resamples in which some group has no long output, or only long outputs,
    among the drawn models, so that its effect has no finite estimate (the slope is then at its limit)."""
    K, N, X = (np.asarray(a, float) for a in (K, N, X))
    M = K.shape[0]
    fit = fit or fe_logit(K, N, X, group)
    th = fit["theta"]
    slopes = np.empty(n_boot)
    bad = bad_group = 0
    for j in range(n_boot):
        ix = rng.integers(0, M, M)
        th0 = np.concatenate([th[:M][ix], th[M:]])
        f = fe_logit(K[ix], N[ix], X[ix], group, theta0=th0)
        if not f["converged"]:
            bad += 1
            if group is not None:
                ks = np.bincount(np.asarray(group, int), K[ix].sum(0))
                ns = np.bincount(np.asarray(group, int), N[ix].sum(0))
                bad_group += int(np.any((ns > 0) & ((ks == 0) | (ks == ns))))
        slopes[j] = f["slope"]
    return {"slopes": slopes, "not_converged": int(bad), "not_converged_group_without_variation": int(bad_group)}


def odds_ratio_summary(fit: dict, boot: dict) -> dict:
    """Odds ratio per unit of X (X is difficulty / 10) with its percentile 95% interval."""
    s = boot["slopes"]
    ok = np.isfinite(s)
    lo, hi = (float(np.exp(np.quantile(s[ok], q))) for q in (0.025, 0.975))
    return {"slope": fit["slope"], "odds_ratio": float(np.exp(fit["slope"])), "odds_ratio_ci95": [lo, hi],
            "converged": fit["converged"], "iterations": fit["iterations"], "n_boot": int(len(s)),
            "boot_not_converged": boot["not_converged"],
            "boot_not_converged_group_without_variation": boot.get("not_converged_group_without_variation", 0),
            "boot_nonfinite": int((~ok).sum())}
