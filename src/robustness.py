"""Model-level robustness checks.

Pure functions on arrays; the scripts scripts/run_model_resampling.py, run_readable_sweep.py, run_gsm8k_difficulty.py,
run_heldout_capability.py, run_lowr_map.py and run_arm_effect.py load the data and write results. Scores are in
percentage points; arrays are (draws, models). Cross-draw moments are those of src/intervals.CrossMoment:
distinct draws only, panel denominator M - 1.
"""
import math
import re

import numpy as np

from src import estimators as d1
from src import intervals as iv


# ---------------------------------------------------------------------------- slopes and model resampling


def cross_slope(X, Y):
    """Slope of Y on X from cross-draw moments, A / V_x, and V_x. A nonpositive V_x gives slope nan (reported, never
    replaced); fewer than two models give nan."""
    if np.asarray(X).shape[1] < 2:
        return float("nan"), float("nan")
    vx = iv.CrossMoment(X, X).estimate()
    a = iv.CrossMoment(Y, X).estimate()
    return (a / vx if vx > 0 else float("nan")), vx


def resample_columns(n_models, clusters, n_boot, rng):
    """Index arrays of resampled models: plain model bootstrap (clusters None) or a cluster bootstrap that draws
    clusters with replacement and keeps every model of a drawn cluster."""
    if clusters is None:
        return [rng.integers(0, n_models, n_models) for _ in range(n_boot)]
    labels = list(clusters)
    if len(labels) != n_models:
        raise ValueError("one cluster label per model")
    groups = {}
    for j, c in enumerate(labels):
        groups.setdefault(c, []).append(j)
    keys = sorted(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        out.append(np.concatenate([groups[keys[p]] for p in pick]))
    return out


def bootstrap_statistic(stat, n_models, clusters, n_boot, rng):
    """Percentile summary of stat(index array) over resampled panels. Undefined replicates (nan, a nonpositive V_x)
    are counted and left out of the percentiles; their count is reported."""
    vals = np.array([stat(ix) for ix in resample_columns(n_models, clusters, n_boot, rng)], float)
    ok = np.isfinite(vals)
    v = vals[ok]
    if len(v) == 0:
        return {"n_boot": int(n_boot), "n_undefined": int((~ok).sum()), "ci95": None, "share_below_zero": None}
    return {"n_boot": int(n_boot), "n_undefined": int((~ok).sum()),
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "median": float(np.median(v)), "share_below_zero": float(np.mean(v < 0)),
            "share_above_zero": float(np.mean(v > 0))}


def leave_out(stat, groups):
    """stat with each group of model indices left out: {label: value}."""
    return {g: stat(np.array(ix, dtype=int)) for g, ix in groups.items()}


def interval_sign(ci):
    if ci is None:
        return "undefined"
    lo, hi = ci
    return "above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero"


# ---------------------------------------------------------------------------- pair orderings under two arms

THRESHOLD_TOL = 1e-9    # pp; a gap within this of a threshold is at the threshold (the tie tolerance)


def pair_reversals(g1, g2, thresholds=(0.0, 1.0, 2.0, 5.0, 10.0)):
    """Model pairs ordered one way by gains g1 and the other way by g2 (model means, pp). A pair with a tie (|gap| <=
    1e-9 pp) under either is neither reversed nor kept. For each threshold t, the pairs whose gap exceeds t under
    both and that reverse. A gap within 1e-9 pp of t is at t, not above it: the model
    means are exact ratios of counts, so a gap can equal a threshold exactly, and its floating-point difference then
    falls on either side of t at random; the tolerance is the tie tolerance, and it reads such a gap as exact arithmetic does."""
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    i, j = np.triu_indices(len(g1), 1)
    d1_, d2_ = g1[i] - g1[j], g2[i] - g2[j]
    s1, s2 = d1.tie_sign(d1_), d1.tie_sign(d2_)
    rev = (s1 * s2) < 0
    out = {"pairs": int(len(i)), "reversed": int(rev.sum()), "reversed_share": float(rev.mean()),
           "ties": int(((s1 == 0) | (s2 == 0)).sum()), "by_threshold": []}
    m = np.minimum(np.abs(d1_), np.abs(d2_))
    for t in thresholds:
        big = m > t + THRESHOLD_TOL
        out["by_threshold"].append({"min_gap_pp": float(t), "pairs_both_above": int(big.sum()),
                                    "reversed": int((rev & big).sum()),
                                    "reversed_share_of_all_pairs": float((rev & big).mean())})
    return out


# ---------------------------------------------------------------------------- held-out capability rules


def design_matrix(b, e, rule):
    """Columns of a capability rule: constant; linear in baseline accuracy b; quadratic in b plus the external index e;
    linear in e alone."""
    b, e = np.asarray(b, float), np.asarray(e, float)
    one = np.ones_like(b)
    if rule == "constant":
        return one[:, None]
    if rule == "baseline_linear":
        return np.column_stack([one, b])
    if rule == "baseline_quadratic_index":
        return np.column_stack([one, b, b * b, e])
    if rule == "index_linear":
        return np.column_stack([one, e])
    raise ValueError(f"unknown rule {rule!r}")


RULES = ("constant", "baseline_linear", "baseline_quadratic_index", "index_linear")


def heldout_predictions(b, g, e, groups, rule):
    """Prediction of each held-out model's mean gain from a least-squares fit of the model mean gains g on the rule's
    columns over the other models (groups: {label: indices held out together}). Predictors are centred and scaled on
    the training models only."""
    b, g, e = np.asarray(b, float), np.asarray(g, float), np.asarray(e, float)
    pred = np.full(len(g), np.nan)
    for ix in groups.values():
        ix = np.asarray(ix)
        tr = np.setdiff1d(np.arange(len(g)), ix)
        mb, sb = b[tr].mean(), b[tr].std() or 1.0
        me, se = e[tr].mean(), e[tr].std() or 1.0
        Xtr = design_matrix((b[tr] - mb) / sb, (e[tr] - me) / se, rule)
        if Xtr.shape[1] > len(tr):
            continue
        coef, *_ = np.linalg.lstsq(Xtr, g[tr], rcond=None)
        pred[ix] = design_matrix((b[ix] - mb) / sb, (e[ix] - me) / se, rule) @ coef
    return pred


def per_model_cross_risk(Z, f):
    """Cross-product risk of prediction f_j for each model j from its validation gains Z[:, j] over distinct draw
    pairs (estimators.cross_risk per model). May be negative; never truncated."""
    Z, f = np.asarray(Z, float), np.asarray(f, float)
    E = Z - f[None, :]
    K = E.shape[0]
    return (E.sum(0) ** 2 - (E * E).sum(0)) / (K * (K - 1))


# ---------------------------------------------------------------------------- agreement at low reliability


def one_draw_c(G):
    """Observed one-draw C of benefit (estimators.pair_agreement over all distinct draw pairs)."""
    return d1.pair_agreement(G)["C"]


def k_draw_agreement(G, K, rng, n_splits=200):
    """Agreement between the model-pair orderings of two disjoint groups of K draws (means), averaged over random
    splits of the draws; ties count one half."""
    G = np.asarray(G, float)
    n = G.shape[0]
    if 2 * K > n:
        return None
    i, j = np.triu_indices(G.shape[1], 1)
    vals = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        a, b = G[p[:K]].mean(0), G[p[K:2 * K]].mean(0)
        sa, sb = d1.tie_sign(a[i] - a[j]), d1.tie_sign(b[i] - b[j])
        vals.append(np.mean(np.where((sa == 0) | (sb == 0), 0.5, (sa == sb).astype(float))))
    return float(np.mean(vals))


def gaussian_reference(R, K=1):
    """C = 1/2 + arcsin(R_K)/pi with Spearman-Brown R_K; None for R outside [0, 1] (reported as is)."""
    if R is None or not np.isfinite(R) or R < 0 or R > 1:
        return None
    return d1.gaussian_agreement(float(R), int(K))


# ---------------------------------------------------------------------------- GSM8K solution steps


def gsm8k_steps(answer_text):
    """Number of calculator annotations <<...>> in a GSM8K gold solution, a metadata count of solution steps; a
    solution without any counts the nonempty reasoning lines before the '####' line."""
    calls = re.findall(r"<<[^>]*>>", answer_text)
    if calls:
        return len(calls)
    body = answer_text.split("####")[0]
    return max(1, sum(1 for line in body.splitlines() if line.strip()))


def step_bins(steps, edges=(2, 3, 4, 5)):
    """Bin labels for step counts: '<=2', '3', '4', '5', '>=6' with the default edges."""
    lab = []
    for s in steps:
        if s <= edges[0]:
            lab.append(f"<={edges[0]}")
        elif s > edges[-1]:
            lab.append(f">={edges[-1] + 1}")
        else:
            lab.append(str(int(s)))
    return lab


def least_squares_trend(values):
    """Least-squares slope of values on 1..L (nan entries make it nan)."""
    v = np.asarray(values, float)
    c = np.arange(1, len(v) + 1) - (len(v) + 1) / 2
    return float(c @ v / (c @ c)) if np.isfinite(v).all() else float("nan")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "undefined" if x is None or (isinstance(x, float) and math.isnan(x)) else ("inf" if x > 0 else "-inf")
    return f"{x:.{nd}f}".replace("-", "−")


# ---------------------------------------------------------------------------- same-generation ladders

_GEN = [(r"^gemma-(\d)-", "Gemma-{0}"), (r"^gemma-\d+b", "Gemma-1"), (r"^llama-(\d)-", "Llama-{0}"),
        (r"^meta-llama-(\d)-", "Llama-{0}"), (r"^llama-(\d\.\d)-", "Llama-{0}"), (r"^phi-(\d)", "Phi-{0}"),
        (r"^qwen(\d(?:\.\d)?)-", "Qwen{0}"), (r"^mistral-small-(\d\.\d)", "Mistral-Small-{0}"),
        (r"^mistral-nemo", "Mistral-Nemo"), (r"^mistral-7b-instruct-(v\d\.\d)", "Mistral-7B-{0}")]


def generation(repo):
    """Family generation from a repository name (the version token): Gemma-1, Gemma-2, Gemma-3, Llama-2, Llama-3,
    Llama-3.1, Phi-3, Phi-4, Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, and the Mistral releases one by one."""
    name = repo.split("/")[-1].lower()
    for pat, out in _GEN:
        m = re.match(pat, name)
        if m:
            return out.format(*[g.capitalize() if g and g[0] == "v" else g for g in m.groups()]) if m.groups() else out
    raise ValueError(f"no generation rule for {repo!r}")


def is_mixture(repo):
    """A mixture-of-experts checkpoint (an active-parameter tag such as A3B in the name)."""
    return bool(re.search(r"-a\d+b", repo.lower()))


def ladders(repos, params, keys, include_qwen3_mixtures=False, dense_only=False, keep=None):
    """{generation: [keys]} for generations with at least two distinct parameter counts. Qwen3's two 30B-A3B mixtures
    enter only with include_qwen3_mixtures; the Qwen3.5 and Qwen3.6 ladders hold a mixture unless dense_only drops
    every mixture. keep: an optional set of keys allowed (for example the readable models)."""
    by = {}
    for r, p, k in zip(repos, params, keys):
        g = generation(r)
        if keep is not None and k not in keep:
            continue
        if is_mixture(r) and (dense_only or (g == "Qwen3" and not include_qwen3_mixtures)):
            continue
        by.setdefault(g, []).append((p, k))
    return {g: [k for _, k in sorted(v)] for g, v in sorted(by.items()) if len({p for p, _ in v}) >= 2}


def per_draw_ols_slope(G, x):
    """Least-squares slope of each draw's gains G[r, :] on x; one value per draw."""
    G, x = np.asarray(G, float), np.asarray(x, float)
    xc = x - x.mean()
    return (G - G.mean(1, keepdims=True)) @ xc / (xc @ xc)


def jack_mean(v, alpha=0.05):
    """Mean of per-draw values with the delete-one-draw jackknife t interval and its sign."""
    v = np.asarray(v, float)
    K = len(v)
    est = float(v.mean())
    loo = (v.sum() - v) / (K - 1)
    lo, hi = iv.jack_t(est, loo, alpha)
    return {"estimate": est, "ci95": [float(lo), float(hi)], "sign": interval_sign([lo, hi]),
            "draws_same_sign": float(np.mean(np.sign(v) == np.sign(est)))}


def gaussian_null(V, W, K_d, K_v, M, rng, k_list=(2, 4, 12), n_splits=50):
    """One replicate of a Gaussian null with the design of a map point: model benefits from N(0, V) and independent
    draw noise from N(0, W), K_d discovery and K_v validation draws over M models. Returns the discovery R and the
    validation agreements computed as for the data."""
    V = max(float(V), 1e-6)
    W = max(float(W), 0.0)
    delta = rng.normal(0.0, math.sqrt(V), M)
    Gd = delta[None, :] + rng.normal(0.0, math.sqrt(W), (K_d, M))
    Gv = delta[None, :] + rng.normal(0.0, math.sqrt(W), (K_v, M))
    R = d1.reliability(Gd)["R"]
    out = {"R_discovery": R, "C_one_draw": one_draw_c(Gv), "C_ref_one_draw": gaussian_reference(R, 1)}
    for K in k_list:
        out[f"C_K{K}"] = k_draw_agreement(Gv, K, rng, n_splits)
        out[f"C_ref_K{K}"] = gaussian_reference(R, K)
    return out
