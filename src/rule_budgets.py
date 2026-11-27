"""Own gain against capability rules by pilot budget.

A pilot is K prompt draws taken without replacement from a setting's discovery draws and one item subsample shared by
every model, both arms and the K draws. On each pilot the rules are fitted to the subsample's scores: own gain (each
model's mean gain over the pilot), own gain shrunken toward the pilot's panel mean by R_K = K R / (1 + (K - 1) R)
(src/pilot_checks.shrink_factor, R from the same pilot; not defined at K = 1), Step 3's capability rule
(src/dataset1.fit_predictors, "averaged_ols") and the rule quadratic in baseline plus the index
(src/capability_checks.fit_rule). Predictions are uncentred, in percentage points, as in Step 3.

Scoring against validation gains Z (validation draws x models, all items): for each pilot, Step 3's paired loss
difference per validation draw (src/estimators.paired_loss_difference, positive favours the second rule); the
per-draw values are averaged over the pilots, and the interval is Step 3's jack_t over the validation draws
(src/intervals.jack_t, Kv - 1 degrees of freedom). Comparisons are written (capability rule, own gain), so a positive
difference means own gain predicts better.

Item subsamples. Simple random samples score as src/noise_anatomy.subsample_matrix: 100 (N / n) sum v_i s_i over the
sampled items, with v the item weights of the full score. Stratified samples (k items from each stratum) score as the
weighted mean of the strata's subsample means, 100 sum_t w_t mean_{i in I_t} s_i, with w_t the stratum's weight in the
full score, so each stratum keeps its weight whatever its size.

Frozen file. The pilots' predictions go to an npz file whose sha256 is recorded in a JSON manifest; load_frozen
checks both before any scoring. np.savez stores a timestamp, so an identical rerun is recognised by the content hash
of the arrays and keeps the existing files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np

from . import capability_checks as cc, dataset1 as d1, intervals as iv, pilot_checks as pc
from .sampling import sha256_json

OWN = ("own_gain", "own_gain_shrunk")
CAPABILITY = ("averaged_ols", cc.FULL_RULE)
RULES = OWN + CAPABILITY
COMPARISONS = tuple((f, g) for f in CAPABILITY for g in OWN)


# --------------------------------------------------------------------------- pilot design


def srs_items(N: int, n: int, rng) -> np.ndarray:
    """A simple random sample of n of the N item positions, without replacement, sorted."""
    if not 1 <= n <= N:
        raise ValueError(f"subsample size {n} outside 1..{N}")
    return np.sort(rng.choice(N, size=n, replace=False))


def stratified_items(strata, k: int, rng) -> np.ndarray:
    """k item positions from each stratum (a list of position arrays), without replacement, strata in order."""
    out = []
    for s in strata:
        s = np.asarray(s)
        if not 1 <= k <= len(s):
            raise ValueError(f"{k} items per stratum outside 1..{len(s)}")
        out.append(np.sort(rng.choice(s, size=k, replace=False)))
    return np.concatenate(out)


def draw_pilots(n_pilots: int, n_draws: int, K: int, rng, N=None, n=None, strata=None):
    """(positions, items): positions (n_pilots, K), K of the n_draws discovery positions without replacement, sorted;
    items (n_pilots, n) item positions, or None for the full item set. With strata, n / len(strata) items come from
    each stratum. Per pilot the draws are taken first, then the items."""
    if not 1 <= K <= n_draws:
        raise ValueError(f"K = {K} outside 1..{n_draws}")
    if n is not None and strata is not None and n % len(strata):
        raise ValueError(f"{n} items do not split evenly over {len(strata)} strata")
    pos = np.empty((n_pilots, K), dtype=np.int64)
    items = None if n is None else np.empty((n_pilots, n), dtype=np.int64)
    for j in range(n_pilots):
        pos[j] = np.sort(rng.choice(n_draws, size=K, replace=False))
        if n is not None:
            items[j] = (stratified_items(strata, n // len(strata), rng) if strata is not None
                        else srs_items(N, n, rng))
    return pos, items


def full_weights(N: int, v=None, strata=None) -> np.ndarray:
    """Item weights of the full score (sum one): v if given; equal stratum weights spread over each stratum's items
    with strata; uniform otherwise."""
    if v is not None:
        w = np.asarray(v, float)
    elif strata is not None:
        w = np.zeros(N)
        for s in strata:
            w[np.asarray(s)] = 1.0 / (len(strata) * len(s))
    else:
        w = np.full(N, 1.0 / N)
    if w.shape != (N,) or (w <= 0).any() or abs(w.sum() - 1) > 1e-9:
        raise ValueError("item weights must be positive, one per item and sum to one")
    return w


def subsample_weights(N: int, items, v=None, strata=None) -> np.ndarray:
    """(N, P) weights in pp: column p scores pilot p's item subsample (scores = item scores @ weights). items None:
    one column, the full score. Simple random samples: 100 (N / n) v_i on the sampled items. Stratified: 100 w_t / k on
    each of stratum t's k sampled items, w_t the stratum's weight in the full score."""
    w = full_weights(N, v, strata)
    if items is None:
        return 100.0 * w[:, None]
    items = np.asarray(items)
    P, n = items.shape
    U = np.zeros((N, P))
    if strata is None:
        U[items, np.arange(P)[:, None]] = 100.0 * N / n * w[items]
        return U
    k = n // len(strata)
    stratum = np.full(N, -1)
    for t, s in enumerate(strata):
        stratum[np.asarray(s)] = t
    wt = np.array([w[np.asarray(s)].sum() for s in strata])
    if (stratum[items] < 0).any():
        raise ValueError("an item lies outside every stratum")
    counts = np.stack([np.bincount(stratum[r], minlength=len(strata)) for r in items])
    if (counts != k).any():
        raise ValueError("a stratified subsample does not hold k items from every stratum")
    U[items, np.arange(P)[:, None]] = 100.0 * wt[stratum[items]] / k
    return U


def distinct_pilots(configs, items=None) -> int:
    """Number of distinct pilots: the set of configurations drawn (positions mapped to configurations, as repeated
    draws of one configuration give the same scores) and the item subsample."""
    keys = set()
    for j, c in enumerate(np.asarray(configs)):
        it = () if items is None else tuple(np.asarray(items)[j].tolist())
        keys.add((tuple(sorted(c.tolist())), it))
    return len(keys)


# --------------------------------------------------------------------------- rules


def fit_pilot(Bp, Pp, e=None) -> dict:
    """Every rule fitted on one pilot's (K, M) scores. own_gain_shrunk and factor are None when the shrinkage is not
    defined (K = 1, or R undefined); R is the pilot's reliability (nan at K = 1). The quadratic-plus-index rule needs
    the index e."""
    Bp, Pp = np.asarray(Bp, float), np.asarray(Pp, float)
    G = Pp - Bp
    own = G.mean(0)
    f = pc.shrink_factor(G)
    m = own.mean()
    out = {"own_gain": own, "own_gain_shrunk": None if f is None else m + f * (own - m), "factor": f,
           "R": pc.pilot_r(G) if G.shape[0] >= 2 else float("nan"),
           "averaged_ols": d1.fit_predictors(Bp, Pp)["averaged_ols"][0]}
    if e is not None:
        out[cc.FULL_RULE] = cc.fit_rule(Bp, Pp, e, 2, True)["prediction"]
    return out


def unclipped_factor(R, K) -> float:
    """K R / (1 + (K - 1) R) before shrink_factor holds it to [0, 1]; nan when undefined."""
    if K < 2 or not np.isfinite(R):
        return float("nan")
    den = 1 + (K - 1) * R
    return K * R / den if den != 0 else float("inf")


# --------------------------------------------------------------------------- scoring


def loss_differences(Z, F, G) -> np.ndarray:
    """(P, Kv): src/estimators.paired_loss_difference(Z, F[p], G[p]) for each pilot p; positive favours G."""
    Z = np.asarray(Z, float)
    F, G = np.atleast_2d(np.asarray(F, float)), np.atleast_2d(np.asarray(G, float))
    if F.shape != G.shape or F.shape[1] != Z.shape[1]:
        raise ValueError("prediction arrays must be (pilots, models) and match the validation panel")
    return ((Z[None] - F[:, None]) ** 2 - (Z[None] - G[:, None]) ** 2).mean(2)


def cross_risks(Z, F) -> np.ndarray:
    """(P,): src/estimators.cross_risk(Z, F[p]) for each pilot p (may be negative)."""
    Z = np.asarray(Z, float)
    F = np.atleast_2d(np.asarray(F, float))
    E = Z[None] - F[:, None]
    K = Z.shape[0]
    return (E.sum(1) ** 2 - (E * E).sum(1)).mean(1) / (K * (K - 1))


def averaged_difference(L) -> dict:
    """L (P, Kv): the per-draw differences averaged over the P pilots, their mean, and Step 3's jack_t interval over
    the Kv validation draws (leave-one-draw-out means, Kv - 1 degrees of freedom); beside it the spread of the
    per-pilot differences."""
    L = np.atleast_2d(np.asarray(L, float))
    per_draw = L.mean(0)
    kv = len(per_draw)
    est = float(per_draw.mean())
    loo = (per_draw.sum() - per_draw) / (kv - 1)
    lo, hi = iv.jack_t(est, loo)
    per_pilot = L.mean(1)
    return {"estimate": est, "se": iv.jackknife_se(loo),
            "ci95": [float(lo), float(hi)] if np.isfinite(lo) and np.isfinite(hi) else None,
            "df": kv - 1, "per_draw": per_draw.tolist(), "pilots": int(L.shape[0]),
            "share_pilots_positive": float(np.mean(per_pilot > 0)),
            "pilot_q05": float(np.quantile(per_pilot, 0.05)), "pilot_median": float(np.median(per_pilot)),
            "pilot_q95": float(np.quantile(per_pilot, 0.95))}


# --------------------------------------------------------------------------- frozen file


def arrays_sha256(arrays: dict) -> str:
    """Content hash of named arrays: name, dtype, shape and bytes, names sorted."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def freeze(record: dict, arrays: dict, json_path, npz_path, overwrite=False) -> dict:
    """Write the arrays to npz_path and the record, with the npz file's sha256 and the arrays' content hash, to
    json_path. An existing pair holding the same arrays is kept as it is (its creation time stands); different arrays
    are refused unless overwrite is set."""
    content = arrays_sha256(arrays)
    rec = dict(record)
    rec["content_sha256"] = content
    rec["npz"] = os.path.basename(npz_path)
    if os.path.exists(json_path):
        old = json.load(open(json_path, encoding="utf-8"))
        same = (old.get("content_sha256") == content and os.path.exists(npz_path)
                and file_sha256(npz_path) == old.get("npz_sha256"))
        if same:
            return old
        if not overwrite:
            raise FileExistsError(f"{json_path} holds different frozen predictions; refusing to overwrite")
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    tmp = str(npz_path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)
    rec["npz_sha256"] = file_sha256(npz_path)
    rec.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    rec["sha256"] = sha256_json({"design": rec.get("design"), "content_sha256": content,
                                 "npz_sha256": rec["npz_sha256"]})
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def load_frozen(json_path, npz_path):
    """The record and the arrays, after checking the npz file's sha256 and the arrays' content hash against the
    record."""
    rec = json.load(open(json_path, encoding="utf-8"))
    if file_sha256(npz_path) != rec.get("npz_sha256"):
        raise ValueError(f"{npz_path} does not match the sha256 recorded in {json_path}")
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    if arrays_sha256(arrays) != rec.get("content_sha256"):
        raise ValueError(f"the arrays of {npz_path} do not match the content hash recorded in {json_path}")
    return rec, arrays


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None
