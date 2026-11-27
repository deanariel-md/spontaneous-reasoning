"""Noise anatomy: the arm-wise split of the gain noise and the gain noise
against the number of items.

Arrays are (draws, models) in percentage points, or (draws, models, R) for a batch of R item subsamples. B is the
baseline arm, P the intervention arm, G = P - B, and e is a draw's deviation from the model's expected score.

Two moment definitions, the ones the existing analyses use:
  - "draws": K independent draws; the noise moment of X and Y is the
    within-draw panel covariance minus the cross-draw moment, W(X, Y) = mean_k Cov_M(X_k, Y_k) - cross_moment(X, Y).
    W(G, G) is the W_gain of src/dataset2.step2. Algebraically W(X, Y) is the sum of products of the doubly centred
    arrays over (K - 1)(M - 1).
  - "bank": a fully enumerated bank under the uniform sampler; the expected score is the bank mean and
    W(X, Y) = mean_k Cov_M(e_X,k, e_Y,k), as in src/dataset1.exact_targets (its noise_W_gain, noise_W_baseline and
    coupling E_cov_eP_eB). This is the sum of products of the doubly centred arrays over K (M - 1).
Both are bilinear, so W_gain = Var(e_P) + Var(e_B) - 2 Cov(e_P, e_B) holds exactly; arm_split reports the residual.
Panel moments use denominator M - 1. Negative estimates and undefined ratios are reported as they are.

Noise against item count. A subsample I of n items, drawn by simple random sampling without replacement from the N
items of a setting, gives the scores X_I = 100 (N / n) sum_{i in I} v_i s_i, where s_i are the 0/1 item scores and v
the item weights of the full score (sum 1; uniform for GSM8K, 1 / (subjects x subject size) for the pooled MMLU
block). For any bilinear moment beta, E_I beta(X_I, Y_I) = beta(X, Y) + (1/n - 1/N) Q with
Q = [sum_i beta(y_i^X, y_i^Y) - N beta(X, Y)] / (N - 1) and y_i = 100 N v_i s_i (the finite-population variance of
a sample mean). So E W(n) = W_shift + W_item / n exactly, with W_item = Q and W_shift = W(N) - Q / N:
design_expectation computes these without sampling, and fit_inverse_n estimates them from the subsamples.
"""
from __future__ import annotations

import math

import numpy as np

from .estimators import _array, cross_moment, gaussian_agreement, prompts_needed_gaussian

KINDS = ("draws", "bank")
PARTS = ("var_e_intervention", "var_e_baseline", "cov_e", "W_gain")


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind


# --------------------------------------------------------------------------- single-array moments


def within_moment(X, Y) -> float:
    """Mean over draws of the panel covariance (denominator M - 1) of X_k and Y_k."""
    X, Y = _array(X), _array(Y)
    if X.shape != Y.shape:
        raise ValueError("Arrays must have the same draw and model layout.")
    K, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return float(np.sum(Xc * Yc) / (K * (M - 1)))


def noise_moment(X, Y, kind: str = "draws") -> float:
    """W(X, Y): the noise covariance of two (draws, models) arrays under the named moment definition."""
    X, Y = _array(X), _array(Y)
    if X.shape != Y.shape:
        raise ValueError("Arrays must have the same draw and model layout.")
    if _kind(kind) == "draws":
        return within_moment(X, Y) - cross_moment(X, Y)
    K, M = X.shape
    eX, eY = X - X.mean(axis=0), Y - Y.mean(axis=0)
    eX = eX - eX.mean(axis=1, keepdims=True)
    eY = eY - eY.mean(axis=1, keepdims=True)
    return float(np.sum(eX * eY) / (K * (M - 1)))


def signal_moment(X, Y, kind: str = "draws") -> float:
    """The panel covariance of the expected scores: cross_moment for draws, the bank means' covariance for a bank."""
    X, Y = _array(X), _array(Y)
    if _kind(kind) == "draws":
        return cross_moment(X, Y)
    x, y = X.mean(axis=0), Y.mean(axis=0)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def _ratio(a, b):
    return a / b if b > 0 else float("nan")


def arm_split(B, P, kind: str = "draws") -> dict:
    """W_gain split into Var(e_P) + Var(e_B) - 2 Cov(e_P, e_B), with the signal moments beside it."""
    B, P = _array(B), _array(P)
    if B.shape != P.shape:
        raise ValueError("Arms must have matching shapes.")
    G = P - B
    vP, vB = noise_moment(P, P, kind), noise_moment(B, B, kind)
    c, wG = noise_moment(P, B, kind), noise_moment(G, G, kind)
    parts = vP + vB - 2 * c
    V = signal_moment(G, G, kind)
    total = V + wG
    return {"kind": kind, "K": int(B.shape[0]), "M": int(B.shape[1]),
            "var_e_intervention": vP, "var_e_baseline": vB, "cov_e": c, "minus_2cov_e": -2 * c,
            "W_gain": wG, "parts_sum": parts, "residual": wG - parts,
            "corr_e": c / math.sqrt(vP * vB) if vP > 0 and vB > 0 else float("nan"),
            "ratio_W_gain_to_W_baseline": _ratio(wG, vB),
            "ratio_intervention_to_baseline": _ratio(vP, vB),
            "share_of_W_gain_from_baseline_noise": _ratio(vB, wG),
            "V_delta": V, "V_theta": signal_moment(B, B, kind), "V_pi": signal_moment(P, P, kind),
            "A": signal_moment(G, B, kind),
            "R_benefit": V / total if total > 0 else float("nan")}


# --------------------------------------------------------------------------- batched moments (draws, models, R)


def _batch(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[0] < 2 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("Expected a finite (draws >= 2, models >= 2, batch) array.")
    return X


def _double_centre(X) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    return X - X.mean(axis=1, keepdims=True)


def noise_batch(X, Y, kind: str = "draws") -> np.ndarray:
    """noise_moment for every slice X[:, :, r], Y[:, :, r]; returns shape (R,)."""
    X, Y = _batch(X), _batch(Y)
    K, M, _ = X.shape
    den = (K - 1) * (M - 1) if _kind(kind) == "draws" else K * (M - 1)
    return np.einsum("kmr,kmr->r", _double_centre(X), _double_centre(Y)) / den


def signal_batch(X, Y, kind: str = "draws") -> np.ndarray:
    """signal_moment for every slice; returns shape (R,)."""
    X, Y = _batch(X), _batch(Y)
    K, M, _ = X.shape
    if _kind(kind) == "draws":
        Xc = X - X.mean(axis=1, keepdims=True)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        return (np.einsum("mr,mr->r", Xc.sum(0), Yc.sum(0)) - np.einsum("kmr,kmr->r", Xc, Yc)) / (K * (K - 1) * (M - 1))
    x, y = X.mean(axis=0), Y.mean(axis=0)
    return np.einsum("mr,mr->r", x - x.mean(0), y - y.mean(0)) / (M - 1)


def split_batch(B, P, kind: str = "draws") -> dict:
    """The arm-wise parts, V_delta and R_benefit for every slice of a batch; arrays of shape (R,)."""
    B, P = _batch(B), _batch(P)
    G = P - B
    out = {"var_e_intervention": noise_batch(P, P, kind), "var_e_baseline": noise_batch(B, B, kind),
           "cov_e": noise_batch(P, B, kind), "W_gain": noise_batch(G, G, kind),
           "V_delta": signal_batch(G, G, kind)}
    total = out["V_delta"] + out["W_gain"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["R_benefit"] = np.where(total > 0, out["V_delta"] / np.where(total > 0, total, 1.0), np.nan)
    return out


# --------------------------------------------------------------------------- item subsamples


def item_weights(N: int, v=None) -> np.ndarray:
    """Item weights of the full score (sum one); uniform when v is None."""
    w = np.full(N, 1.0 / N) if v is None else np.asarray(v, dtype=float)
    if w.shape != (N,) or (w <= 0).any() or not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-9:
        raise ValueError("item weights must be positive, finite, one per item and sum to one")
    return w


def draw_subsamples(N: int, n: int, reps: int, rng) -> np.ndarray:
    """(reps, n) item positions, each row a simple random sample without replacement. At n == N: the full set once."""
    if not 1 <= n <= N:
        raise ValueError(f"subsample size {n} outside 1..{N}")
    if n == N:
        return np.arange(N)[None, :]
    return np.stack([np.sort(rng.choice(N, size=n, replace=False)) for _ in range(reps)])


def subsample_matrix(N: int, idx, v=None) -> np.ndarray:
    """(N, R) weights: column r holds 100 (N / n) v_i on the items of subsample r and 0 elsewhere."""
    w = item_weights(N, v)
    idx = np.asarray(idx)
    R, n = idx.shape
    U = np.zeros((N, R))
    U[idx, np.arange(R)[:, None]] = 100.0 * N / n * w[idx]
    return U


def subsample_scores(S, U, chunk: int = 256) -> np.ndarray:
    """Scores (K, M, R) in pp from item scores S (K, M, N) and subsample weights U (N, R), in chunks over K."""
    S = np.asarray(S)
    K, M, N = S.shape
    if U.shape[0] != N:
        raise ValueError("subsample weights do not match the item count")
    out = np.empty((K, M, U.shape[1]))
    for a in range(0, K, chunk):
        out[a:a + chunk] = (S[a:a + chunk].reshape(-1, N).astype(np.float64) @ U).reshape(-1, M, U.shape[1])
    return out


def full_scores(S, v=None) -> np.ndarray:
    """(K, M) scores in pp on the full item set."""
    S = np.asarray(S)
    w = item_weights(S.shape[2], v)
    return 100.0 * np.einsum("kmn,n->km", S.astype(np.float64), w)


def design_expectation(SB, SP, v=None, kind: str = "draws", chunk: int = 256) -> dict:
    """Exact expectation of each noise part (and V_delta) over simple random item subsamples of size n:
    E = full + (1/n - 1/N) W_item, so W_shift = full - W_item / N. No sampling is involved."""
    SB, SP = np.asarray(SB), np.asarray(SP)
    if SB.shape != SP.shape or SB.ndim != 3:
        raise ValueError("item score arrays must be (draws, models, items) and match")
    K, M, N = SB.shape
    if N < 2:
        raise ValueError("need at least two items")
    w = item_weights(N, v)
    pairs = {"var_e_intervention": ("P", "P", noise_batch), "var_e_baseline": ("B", "B", noise_batch),
             "cov_e": ("P", "B", noise_batch), "V_delta": ("G", "G", signal_batch)}
    acc = dict.fromkeys(pairs, 0.0)
    for a in range(0, N, chunk):
        sl = slice(a, a + chunk)
        scale = 100.0 * N * w[sl]
        y = {"B": SB[:, :, sl].astype(np.float64) * scale, "P": SP[:, :, sl].astype(np.float64) * scale}
        y["G"] = y["P"] - y["B"]
        for name, (x1, x2, f) in pairs.items():
            acc[name] += float(f(y[x1], y[x2], kind).sum())
    B, P = full_scores(SB, w), full_scores(SP, w)
    full = {"var_e_intervention": noise_moment(P, P, kind), "var_e_baseline": noise_moment(B, B, kind),
            "cov_e": noise_moment(P, B, kind), "V_delta": signal_moment(P - B, P - B, kind)}
    out = {}
    for name in pairs:
        q = (acc[name] - N * full[name]) / (N - 1)
        out[name] = {"full": full[name], "W_item": q, "W_shift": full[name] - q / N}
    g = {k: out["var_e_intervention"][k] + out["var_e_baseline"][k] - 2 * out["cov_e"][k]
         for k in ("full", "W_item", "W_shift")}
    out["W_gain"] = g
    return {"kind": kind, "K": int(K), "M": int(M), "N": int(N), "parts": out}


def expected_at(design_part: dict, n: float) -> float:
    return design_part["W_shift"] + design_part["W_item"] / n


def config_variance(S, scale: float = 100.0) -> dict:
    """design_expectation for a single model: its spread over configurations that share one item set.

    S is (configurations C, items N): each configuration's item values on the same N items (0/1 scores, or the
    difference of two arms' scores). With y_i = scale * S[:, i] (pp) and the moment beta the variance over
    configurations (denominator C - 1), design_expectation's terms are
        full    = Var_c(mean_i y_ci), the variance of the configuration means;
        W_item  = [sum_i Var_c(y_ci) - N full] / (N - 1), which equals the configuration x item interaction mean
                  square of the two-way layout, sum_ci (y_ci - ybar_c. - ybar_.i + ybar..)^2 / ((C - 1)(N - 1));
        W_shift = full - W_item / N, the spread of the configuration means with the item sampling removed.
    Item difficulty is common to every configuration, so it drops out of the spread and does not enter W_item (the
    variance of an item's value over the configurations is taken about that item's own mean). Under a random
    configuration effect a (variance s2_a) and an interaction with variance s2_ab, E full = s2_a + s2_ab / N and
    E W_item = s2_ab, so W_shift estimates s2_a. Negative values are reported as they are."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] < 2 or S.shape[1] < 2 or not np.isfinite(S).all():
        raise ValueError("Expected a finite (configurations >= 2, items >= 2) array with one value per cell.")
    C, N = S.shape
    y = scale * S
    full = float(np.var(y.mean(axis=1), ddof=1))
    q = float((np.var(y, axis=0, ddof=1).sum() - N * full) / (N - 1))
    return {"C": int(C), "N": int(N), "full": full, "W_item": q, "W_shift": full - q / N}


# --------------------------------------------------------------------------- fit


def fit_inverse_n(ns, values, N: int) -> dict:
    """Fit W(n) = W_shift + W_item / n to subsample values.

    `values[j]` holds the values of the subsamples at size ns[j]; the full item set (n == N) must be present, as a
    single value. Primary fit: the line is constrained through the full-set value W(N), which has no subsampling
    variance, and W_item is estimated by weighted least squares of (mean_n - W(N)) on (1/n - 1/N) over n < N, with
    weights the inverse squared Monte Carlo standard error of each mean (between-subsample variance / count). This
    is the form of the exact design expectation. Sensitivity: ordinary least squares of the per-n means on 1/n over
    the whole grid, equal weights, both coefficients free."""
    ns = [int(n) for n in ns]
    if N not in ns:
        raise ValueError("the grid must include the full item count")
    vals = [np.asarray(v, dtype=float).ravel() for v in values]
    full = vals[ns.index(N)]
    if full.size != 1:
        raise ValueError("the full item set gives exactly one value")
    W_N = float(full[0])
    rows, x, y, se2 = [], [], [], []
    for n, v in zip(ns, vals):
        m = float(v.mean())
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        rows.append({"n": n, "n_subsamples": int(v.size), "mean": m, "sd": sd,
                     "mc_se": sd / math.sqrt(v.size) if v.size > 1 else 0.0,
                     "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
                     "frac_negative": float(np.mean(v < 0))})
        if n < N:
            x.append(1.0 / n - 1.0 / N)
            y.append(m - W_N)
            se2.append((sd ** 2) / v.size if v.size > 1 else float("nan"))
    x, y, se2 = np.array(x), np.array(y), np.array(se2)
    if x.size == 0:
        raise ValueError("need at least one subsample size below the full item count")
    weights_ok = bool(np.isfinite(se2).all() and (se2 > 0).all())
    wts = 1.0 / se2 if weights_ok else np.ones_like(x)
    sxx = float(np.sum(wts * x * x))
    W_item = float(np.sum(wts * x * y) / sxx)
    resid = y - W_item * x
    chi2 = float(np.sum(wts * resid ** 2)) if weights_ok else float("nan")
    df = int(x.size - 1)
    se_item = math.sqrt(1.0 / sxx) if weights_ok else float("nan")
    for r in rows:
        r["fitted"] = (W_N - W_item / N) + W_item / r["n"]
        r["standardized_residual"] = ((r["mean"] - r["fitted"]) / r["mc_se"]) if r["mc_se"] > 0 else None
    # Sensitivity: unconstrained OLS of the means on 1/n over the whole grid.
    xm = np.array([1.0 / r["n"] for r in rows])
    ym = np.array([r["mean"] for r in rows])
    if xm.size >= 2 and np.ptp(xm) > 0:
        b, a = np.polyfit(xm, ym, 1)
        ols = {"W_shift": float(a), "W_item": float(b), "points": int(xm.size)}
    else:
        ols = None
    return {"method": "constrained weighted least squares through W(N); weights 1 / mc_se^2 of each per-n mean"
                      if weights_ok else "constrained least squares through W(N); equal weights (a per-n Monte "
                                         "Carlo variance was not positive)",
            "weights": "inverse squared Monte Carlo standard error" if weights_ok else "equal",
            "N": int(N), "W_full": W_N, "W_item": W_item, "W_item_se": se_item,
            "W_shift": W_N - W_item / N, "W_shift_se": se_item / N if weights_ok else float("nan"),
            "chi2": chi2, "df": df, "per_n": rows, "ols_sensitivity": ols}


# --------------------------------------------------------------------------- planning rule


def planning_rule(V_delta: float, W_shift: float, W_item: float, ns, targets=(0.8, 0.9)) -> list[dict]:
    """K needed for each agreement target at each benchmark size n: R(n) = V_delta / (V_delta + W(n)) with
    W(n) = W_shift + W_item / n, R_K by Spearman-Brown and C_K = 1/2 + arcsin(R_K)/pi (see
    prompts_needed_gaussian). Undefined values are reported with the reason."""
    rows = []
    for n in ns:
        W = W_shift + W_item / n
        den = V_delta + W
        row = {"n": int(n), "W": W, "R": V_delta / den if den > 0 else float("nan")}
        if den <= 0:
            row["undefined"] = "V_delta + W(n) is not positive"
        R = row["R"]
        ok = math.isfinite(R) and 0 <= R <= 1
        if math.isfinite(R) and not ok:
            row["undefined"] = "R outside [0, 1]"
        row["C_1"] = gaussian_agreement(R, 1) if ok else float("nan")
        for t in targets:
            k = prompts_needed_gaussian(R, t) if ok else None
            row[f"K_{int(round(100 * t))}"] = ("unbounded" if k == math.inf else k)
        rows.append(row)
    return rows
