"""Core estimators.

Fixed-panel moments, tie-aware agreement, the supplementary conditional Gaussian
reference, the supplementary fixed-task decomposition, the regularized predictor and
noise-corrected risk. Inputs are percentage-point scores with layout (draws, models).
Not a complete inference package: interval routines live elsewhere and must pass the
calibration checks. tests/test_estimators.py reproduces the verification record of these estimators.
"""
import math
import numpy as np

def _array(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite (draws >= 2, models >= 2) array.")
    return X

def cross_moment(X, Y):
    X, Y = _array(X), _array(Y)
    if X.shape != Y.shape:
        raise ValueError("Arrays must have the same draw and model layout.")
    K, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return float((Xc.sum(0) @ Yc.sum(0) - np.sum(Xc * Yc)) /
                 (K * (K - 1) * (M - 1)))

def unit_moments(B, P):
    B, P = _array(B), _array(P)
    if B.shape != P.shape:
        raise ValueError("Arms must have matching shapes.")
    G = P - B
    return {"V_theta": cross_moment(B, B),
            "A": cross_moment(G, B),
            "V_delta": cross_moment(G, G)}

def reliability(X):
    X = _array(X)
    signal = cross_moment(X, X)
    total = float(np.var(X, axis=1, ddof=1).mean())
    return {"signal": signal, "total": total,
            "R": signal / total if total > 0 else float("nan")}

# A model-pair difference of at most TIE_TOL pp is a tie. Without it,
# np.sign counted differences of floating-point size (two models with equal scores, reached by different sums) as
# decisive.
TIE_TOL = 1e-9


def tie_sign(D, tol=TIE_TOL):
    """Sign of a model-pair difference in pp, with |D| <= tol a tie (0)."""
    D = np.asarray(D, float)
    return np.where(np.abs(D) <= tol, 0.0, np.sign(D))


def pair_agreement(G):
    """All distinct draw pairs; pairs are NOT independent sampling units."""
    G = _array(G)
    K, M = G.shape
    i, j = np.triu_indices(M, 1)
    signs = tie_sign(G[:, i] - G[:, j])
    pos, neg = (signs > 0).sum(0), (signs < 0).sum(0)
    denom = K * (K - 1)
    same = float(np.mean((pos * (pos - 1) + neg * (neg - 1)) / denom))
    opposite = float(np.mean(2 * pos * neg / denom))
    tie = float(1 - same - opposite)
    tie = 0.0 if abs(tie) < 1e-12 else tie   # floating-point residue only
    decisive = same + opposite
    return {"same": same, "opposite": opposite, "tie": tie,
            "C": same + 0.5 * tie,
            "conditional_agreement": same / decisive if decisive > 0 else float("nan")}

def gaussian_agreement(R, K=1):
    """Conditional Gaussian reference; not a general fixed-panel law."""
    if not 0 <= R <= 1 or int(K) != K or K < 1:
        raise ValueError("Require population R in [0,1] and integer K >= 1.")
    RK = K * R / (1 + (K - 1) * R)
    return 0.5 + math.asin(RK) / math.pi

def prompts_needed_gaussian(R, target=0.9):
    if not 0 <= R <= 1 or not 0.5 < target < 1:
        raise ValueError("Require R in [0,1] and target in (0.5,1).")
    if R == 0:
        return math.inf
    if R == 1:
        return 1
    star = math.sin(math.pi * (target - 0.5))
    return max(1, math.ceil(star * (1 - R) / (R * (1 - star))))

def fixed_task_components(G, weights=None):
    """G: (tasks, independent joint draw bundles, common fixed models).
    Components refer to THIS weighted task collection, not future tasks."""
    G = np.asarray(G, dtype=float)
    if G.ndim != 3 or min(G.shape) < 2 or not np.isfinite(G).all():
        raise ValueError("Need >=2 tasks, >=2 joint draws, >=2 models.")
    T = G.shape[0]
    w = np.ones(T) / T if weights is None else np.asarray(weights, float)
    if w.shape != (T,) or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Invalid task weights.")
    w = w / w.sum()
    within = float(sum(w[t] * cross_moment(G[t], G[t]) for t in range(T)))
    pooled = np.einsum("t,tkm->km", w, G)
    profile = cross_moment(pooled, pooled)
    total = float(sum(w[t] * np.var(G[t], axis=1, ddof=1).mean() for t in range(T)))
    return {"average_task_profile": profile,
            "model_by_task": within - profile,
            "prompt_noise": total - within,
            "true_within_task": within, "total": total}

def replicated_linear_predictor(B, P):
    """PSD regularization is for prediction only; raw inference stays unchanged."""
    B, P = _array(B), _array(P)
    q = unit_moments(B, P)
    G = P - B
    K, M = B.shape
    Bb, Gb = B.mean(0), G.mean(0)
    latent_raw = np.array([[q["V_theta"], q["A"]], [q["A"], q["V_delta"]]])
    eig, U = np.linalg.eigh(latent_raw)
    latent = (U * np.maximum(eig, 0)) @ U.T
    Z = np.stack([B, G], axis=1)
    Z = Z - Z.mean(axis=2, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)
    noise = np.einsum("kim,kjm->ij", Z, Z) / (K * (K - 1) * (M - 1))
    covariance = latent + noise
    ridge = max(1e-12, 1e-6 * np.trace(covariance) / 2)
    if latent[1, 1] <= ridge:
        return np.full(M, Gb.mean()), {"fallback": "constant", "psd_projection": bool(np.any(eig < 0))}
    w = np.linalg.solve(covariance + ridge * np.eye(2), latent[:, 1])
    pred = Gb.mean() + w[0] * (Bb - Bb.mean()) + w[1] * (Gb - Gb.mean())
    return pred, {"fallback": None, "psd_projection": bool(np.any(eig < 0)), "weights": w}

def paired_loss_difference(Z, f, g):
    """One value per independent validation draw; positive favors g."""
    Z = _array(Z)
    f, g = np.asarray(f, float), np.asarray(g, float)
    if f.shape != (Z.shape[1],) or g.shape != f.shape:
        raise ValueError("Prediction vectors must match the model panel.")
    return ((Z - f)**2 - (Z - g)**2).mean(axis=1)

def cross_risk(Z, f):
    """Unbiased risk for frozen f under independent mean-zero validation errors.
    May be negative in finite samples; do not silently truncate."""
    Z = _array(Z)
    f = np.asarray(f, float)
    if f.shape != (Z.shape[1],):
        raise ValueError("Prediction vector must match the model panel.")
    E = Z - f
    K = E.shape[0]
    return float(np.mean(E.sum(0)**2 - np.sum(E * E, axis=0)) / (K * (K - 1)))
