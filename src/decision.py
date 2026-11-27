"""Decision analysis: give a strategy to the third of the panel predicted to gain most,
and measure the cost of each way of predicting it in points of gain.

Arrays are draws by models, in percentage points. A choice is a weight per model: 1 for a chosen model, 0 for one left
out, and a shared fraction for models tied at the cut, which is the expected value of breaking the tie at random. The
weights of a choice sum to n, the size of the third.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import stats

from src import dataset1 as d1
from src import robustness as rb
from src.estimators import replicated_linear_predictor

TIE_TOL = 1e-9
PILOT_SIZES = (1, 2, 4)
SHRUNK_SIZES = (2, 4)
N_PILOT_SETS = 200


def third_size(M: int) -> int:
    """n = round(M / 3), at least 1 and at most M - 1: 10 of 29, 9 of 28, 2 of 5."""
    if M < 2:
        raise ValueError("a panel of at least two models is needed")
    return int(min(M - 1, max(1, round(M / 3))))


def choose(pred, n: int, tol: float = TIE_TOL) -> tuple[np.ndarray, int]:
    """Weights of the n highest predictions and the number of models tied at the cut (0 when the cut is clean).
    Models tied at the cut share the remaining places equally."""
    p = np.asarray(pred, float)
    if not np.all(np.isfinite(p)):
        raise ValueError("a prediction is not finite")
    M = len(p)
    if not 1 <= n <= M:
        raise ValueError(f"cannot choose {n} of {M}")
    v = np.sort(p)[::-1][n - 1]
    above = p > v + tol
    tied = np.abs(p - v) <= tol
    w = above.astype(float)
    slots = n - int(above.sum())
    w[tied] = slots / tied.sum()
    n_tied = int(tied.sum()) if tied.sum() > slots else 0
    return w, n_tied


def label_seed(label: str) -> int:
    """The first 8 bytes of sha256(label), for seeding."""
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def distinct_rows(units) -> list[int]:
    """Row index of the first draw of each distinct configuration, in sampler order."""
    seen, rows = set(), []
    for j, u in enumerate(map(str, units)):
        if u not in seen:
            seen.add(u)
            rows.append(j)
    return rows


def pilot_sets(units, task: str, sizes=PILOT_SIZES, n_sets: int = N_PILOT_SETS) -> dict:
    """{K: [[row, ...], ...]} rows of distinct discovery configurations. K = 1: each distinct configuration once. K > 1:
    n_sets sets drawn without replacement within a set, seeded by sha256('decision/<task>/K<k>')."""
    rows = distinct_rows(units)
    out = {}
    for k in sizes:
        if k == 1:
            out[1] = [[r] for r in rows]
            continue
        if k > len(rows):
            raise ValueError(f"{len(rows)} distinct configurations for pilots of {k}")
        rng = np.random.default_rng(label_seed(f"decision/{task}/K{k}"))
        out[k] = [[rows[i] for i in sorted(rng.choice(len(rows), size=k, replace=False))] for _ in range(n_sets)]
    return out


def capability_one_draw(b, g) -> tuple[np.ndarray, float]:
    """Prediction on the least-squares line of gain on baseline accuracy across the panel in one draw, and its slope.
    An undefined slope gives the constant (every model tied)."""
    b, g = np.asarray(b, float), np.asarray(g, float)
    s = d1.slope(g, b)
    if not np.isfinite(s):
        return np.full(len(g), g.mean()), s
    return g.mean() + s * (b - b.mean()), s


def capability_cross(B, G) -> tuple[np.ndarray, float]:
    """The study's cross-draw slope A / V_x of gain on accuracy, applied to the mean accuracy over the draws."""
    s, _ = rb.cross_slope(np.asarray(B, float), np.asarray(G, float))
    gb, bb = G.mean(0), B.mean(0)
    if not np.isfinite(s):
        return np.full(len(gb), gb.mean()), s
    return gb.mean() + s * (bb - bb.mean()), s


def rule_predictions(B, P, sets: dict) -> dict:
    """{rule: [(prediction, info), ...] per pilot set}, plus the 24-draw references, from discovery draws only.
    B, P: (K, M) discovery arrays of baseline and intervention."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    G = P - B
    out = {}
    for k, ss in sets.items():
        own = []
        for rows in ss:
            own.append((G[rows].mean(0), {"rows": rows}))
        out[f"own_{k}"] = own
        if k == 1:
            out["capability_1"] = []
            for rows in ss:
                pred, s = capability_one_draw(B[rows[0]], G[rows[0]])
                out["capability_1"].append((pred, {"rows": rows, "slope": s}))
        if k in SHRUNK_SIZES:
            sh = []
            for rows in ss:
                pred, info = replicated_linear_predictor(B[rows], P[rows])
                info = {key: (v.tolist() if isinstance(v, np.ndarray) else v) for key, v in info.items()}
                sh.append((pred, {"rows": rows, **info}))
            out[f"shrunk_{k}"] = sh
    return out


def reference_predictions(B, P, rows) -> dict:
    """Own gain and the cross-draw capability slope from the given (distinct) discovery rows."""
    B, P = np.asarray(B, float)[rows], np.asarray(P, float)[rows]
    G = P - B
    pred, s = capability_cross(B, G)
    return {"own_all": [(G.mean(0), {"rows": list(rows)})], "capability_all": [(pred, {"rows": list(rows), "slope": s})]}


def choices(preds: dict, n: int) -> dict:
    """{rule: {'weights': (n_sets, M), 'n_tied': [...]}} from rule_predictions."""
    out = {}
    for r, lst in preds.items():
        W, T = [], []
        for pred, _ in lst:
            w, t = choose(pred, n)
            W.append(w)
            T.append(t)
        out[r] = {"weights": np.array(W), "n_tied": T}
    return out


# --------------------------------------------------------------------------- evaluation


def values(W, Gv, n: int) -> np.ndarray:
    """(n_sets, K_v) value of each choice on each evaluation draw: the weighted mean gain of the chosen models."""
    return np.atleast_2d(W) @ np.asarray(Gv, float).T / n


def best_weights(Gv, n: int):
    return choose(np.asarray(Gv, float).mean(0), n)[0]


def regret_plugin(W, Gv, n: int) -> np.ndarray:
    """Per choice: value of the best third on the same draws minus the choice's value (biased upward)."""
    gm = np.asarray(Gv, float).mean(0)
    wb = best_weights(Gv, n)
    return (wb @ gm - np.atleast_2d(W) @ gm) / n


def halves(n_rows: int) -> tuple[list[int], list[int]]:
    h = n_rows // 2
    return list(range(h)), list(range(h, n_rows))


def regret_crossfit(W, Gv, n: int) -> np.ndarray:
    """Per choice: best third chosen on one half of the evaluation draws, valued on the other, both ways, averaged,
    minus the choice's value on the same halves (biased downward; can be negative)."""
    Gv = np.asarray(Gv, float)
    a, b = halves(Gv.shape[0])
    if not a or not b:
        return np.full(np.atleast_2d(W).shape[0], np.nan)
    out = 0.0
    for x, y in ((a, b), (b, a)):
        wb = best_weights(Gv[x], n)
        gy = Gv[y].mean(0)
        out = out + (wb @ gy - np.atleast_2d(W) @ gy) / n
    return out / 2


def regret_against(W, g_true, n: int) -> np.ndarray:
    """Regret against a known gain vector (DOVE's whole common bank)."""
    g = np.asarray(g_true, float)
    wb = choose(g, n)[0]
    return (wb @ g - np.atleast_2d(W) @ g) / n


def same_as_best(W, Gv, n: int, tol: float = 1e-9) -> float:
    wb = best_weights(Gv, n)
    return float(np.mean(np.all(np.abs(np.atleast_2d(W) - wb) <= tol, axis=1)))


def t_interval(d, alpha: float = 0.05) -> dict:
    """Mean of per-draw differences with a t interval (K - 1 degrees of freedom)."""
    d = np.asarray(d, float)
    K = len(d)
    m = float(d.mean())
    if K < 2:
        return {"estimate": m, "ci95": None, "K": K}
    se = float(d.std(ddof=1) / math.sqrt(K))
    q = float(stats.t.ppf(1 - alpha / 2, K - 1))
    return {"estimate": m, "se": se, "ci95": [m - q * se, m + q * se], "K": K}


def paired_difference(Wa, Wb, Gv, n: int) -> dict:
    """Extra regret of choice set a over choice set b (positive: a is worse), averaged over pilot sets, with a t interval
    over the evaluation draws. Linear in the draws, so the best third cancels."""
    da = values(np.atleast_2d(Wa).mean(0), Gv, n)[0]
    db = values(np.atleast_2d(Wb).mean(0), Gv, n)[0]
    return t_interval(db - da)


def summarize(W, Gv, n: int, g_true=None) -> dict:
    rp = regret_plugin(W, Gv, n)
    rc = regret_crossfit(W, Gv, n)
    out = {"n_sets": int(np.atleast_2d(W).shape[0]),
           "regret_plugin": float(rp.mean()), "regret_crossfit": float(np.mean(rc)),
           "regret_plugin_p10": float(np.percentile(rp, 10)), "regret_plugin_p90": float(np.percentile(rp, 90)),
           "share_best_third": same_as_best(W, Gv, n),
           "value": float(values(np.atleast_2d(W), Gv, n).mean())}
    if g_true is not None:
        out["regret_exact"] = float(regret_against(W, g_true, n).mean())
    return out
