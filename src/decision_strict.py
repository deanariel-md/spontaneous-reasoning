"""Equal-call decision comparison with pilot items held out of the evaluation.

A one-draw pilot is one discovery draw scored on a fixed item subsample, for every model and both arms. On it three
rules predict each model's gain: own gain, the capability rule linear in baseline accuracy (src/dataset1.fit_predictors,
"averaged_ols") and the rule quadratic in baseline accuracy plus the external index (src/capability_checks.fit_rule).
All three use the same calls; the index rule adds one public number per model. The top third by each rule is chosen with
src/decision.choose (ties at the cut share the places). A random third is the floor.

Item split. Items are ranked within each stratum (a BIG-Bench Hard task; the whole set otherwise) by
sha256("<label>/<task>/<item_id>"); the first floor(n / 2) of each stratum form the pilot pool, the rest the evaluation
pool. Pilots draw their items from the pilot pool only, and choices are valued on the evaluation pool only, with each
stratum keeping its weight in the full score (BIG-Bench Hard: 25 tasks of equal weight).

Units: pp of accuracy gain per chosen model. Arrays: draws or configurations by models.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import capability_checks as cc, dataset1 as d1, decision as dc, intervals as iv

RULES = ("linear", "quadratic_index", "own_gain")
RULE_NAMES = {"linear": "capability, linear in baseline", "quadratic_index": "capability, quadratic in baseline + index",
              "own_gain": "own measured gain", "random": "random third"}


# --------------------------------------------------------------------------- item split


def item_rank_key(label: str, task: str, item_id) -> str:
    return hashlib.sha256(f"{label}/{task}/{item_id}".encode("utf-8")).hexdigest()


def split_items(item_ids, task: str, label: str, strata=None) -> np.ndarray:
    """Boolean mask over item positions: True for the pilot pool. Within each stratum (positions), the floor(n / 2)
    items with the smallest sha256 key go to the pilot pool. Depends on item ids only."""
    ids = [str(i) for i in item_ids]
    groups = [np.arange(len(ids))] if strata is None else [np.asarray(s) for s in strata]
    mask = np.zeros(len(ids), bool)
    seen = np.zeros(len(ids), int)
    for g in groups:
        seen[g] += 1
        order = sorted(g.tolist(), key=lambda p: item_rank_key(label, task, ids[p]))
        mask[order[:len(order) // 2]] = True
    if (seen != 1).any():
        raise ValueError("the strata must partition the items")
    return mask


def restrict_strata(strata, mask):
    """Each stratum's positions inside the mask (None stays None)."""
    if strata is None:
        return None
    return [np.asarray(s)[mask[np.asarray(s)]] for s in strata]


def eval_weights(N: int, mask, strata=None) -> np.ndarray:
    """(N,) weights in pp summing to 100 over the masked items: each stratum keeps weight 1 / n_strata, spread evenly
    over its masked items; without strata, every masked item weighs the same."""
    mask = np.asarray(mask, bool)
    w = np.zeros(N)
    if strata is None:
        if not mask.any():
            raise ValueError("no items to evaluate on")
        w[mask] = 100.0 / mask.sum()
        return w
    for s in strata:
        s = np.asarray(s)
        k = s[mask[s]]
        if len(k) == 0:
            raise ValueError("a stratum has no evaluation items")
        w[k] = 100.0 / (len(strata) * len(k))
    return w


# --------------------------------------------------------------------------- pilots


def draw_pilots(n_pilots: int, n_draws: int, rng, pool, n: int, strata=None):
    """(positions (n_pilots,), items (n_pilots, n)): one discovery position per pilot, then its items from the pool
    (positions; a simple random sample) or, with strata (positions inside the pool), n / len(strata) from each stratum.
    Per pilot the draw first, then the items, as src/rule_budgets.draw_pilots with K = 1."""
    pool = np.asarray(pool)
    pos = np.empty(n_pilots, dtype=np.int64)
    items = np.empty((n_pilots, n), dtype=np.int64)
    if strata is not None and n % len(strata):
        raise ValueError(f"{n} items do not split evenly over {len(strata)} strata")
    for j in range(n_pilots):
        pos[j] = rng.choice(n_draws, size=1, replace=False)[0]
        if strata is None:
            items[j] = np.sort(pool[rng.choice(len(pool), size=n, replace=False)])
        else:
            k = n // len(strata)
            items[j] = np.concatenate([np.sort(rng.choice(np.asarray(s), size=k, replace=False)) for s in strata])
    return pos, items


def pilot_weights(N: int, items, strata=None) -> np.ndarray:
    """(N, P) weights in pp: each pilot's subsample score. Without strata 100 / n per sampled item; with strata each
    stratum keeps weight 1 / n_strata, spread over its k sampled items."""
    items = np.asarray(items)
    P, n = items.shape
    U = np.zeros((N, P))
    if strata is None:
        U[items, np.arange(P)[:, None]] = 100.0 / n
        return U
    k = n // len(strata)
    stratum = np.full(N, -1)
    for t, s in enumerate(strata):
        stratum[np.asarray(s)] = t
    if (stratum[items] < 0).any():
        raise ValueError("an item lies outside every stratum")
    U[items, np.arange(P)[:, None]] = 100.0 / (len(strata) * k)
    return U


def pilot_scores(S, pos, U) -> np.ndarray:
    """(P, M) scores: S (draws, models, items) at each pilot's draw, weighted by the pilot's column of U."""
    S = np.asarray(S, float)
    return np.stack([S[p] @ U[:, j] for j, p in enumerate(pos)])


def fit_rules(b, p, e) -> dict:
    """Every rule's predicted gain from one pilot: b, p (M,) baseline and intervention scores in pp; e the index."""
    b, p = np.asarray(b, float)[None], np.asarray(p, float)[None]
    return {"own_gain": (p - b)[0],
            "linear": d1.fit_predictors(b, p)["averaged_ols"][0],
            "quadratic_index": cc.fit_rule(b, p, np.asarray(e, float), 2, True)["prediction"]}


def pilot_predictions(Bp, Pp, e) -> dict:
    """{rule: (P, M)} from per-pilot scores Bp, Pp (P, M)."""
    out = {r: [] for r in RULES}
    for b, p in zip(Bp, Pp):
        f = fit_rules(b, p, e)
        for r in RULES:
            out[r].append(f[r])
    return {r: np.array(v) for r, v in out.items()}


def choice_weights(pred, n: int):
    """(P, M) weights and the number of pilots with a tie at the cut."""
    W, T = [], 0
    for x in np.atleast_2d(pred):
        w, t = dc.choose(x, n)
        W.append(w)
        T += int(t > 0)
    return np.array(W), T


# --------------------------------------------------------------------------- evaluation


def jackknife_regret(W, G, n: int) -> dict:
    """Mean plug-in regret over pilots with a leave-one-configuration-out jackknife t interval (C - 1 df). The best
    third is re-chosen on each leave-one-out set, so the interval carries the plug-in's selection."""
    G = np.asarray(G, float)
    est = float(dc.regret_plugin(W, G, n).mean())
    C = G.shape[0]
    if C < 3:
        return {"estimate": est, "ci95": None, "n_configurations": C}
    loo = np.array([dc.regret_plugin(W, np.delete(G, i, axis=0), n).mean() for i in range(C)])
    lo, hi = iv.jack_t(est, loo)
    return {"estimate": est, "ci95": [float(lo), float(hi)] if np.isfinite(lo) else None, "n_configurations": C}


def evaluate(Wd: dict, G, n: int) -> dict:
    """Per rule: plug-in regret with its jackknife interval, cross-fitted regret, spread over pilots, share choosing
    the plug-in best third; paired differences (regret of own gain minus each other rule's, t interval over
    configurations; negative: own gain costs less)."""
    G = np.asarray(G, float)
    out = {"n_configurations": int(G.shape[0]), "rules": {}, "paired": {}}
    gm = G.mean(0)
    out["best_third_value"] = float(dc.best_weights(G, n) @ gm / n)
    out["panel_mean_gain"] = float(gm.mean())
    for r, W in Wd.items():
        s = dc.summarize(W, G, n)
        s["plugin_interval"] = jackknife_regret(W, G, n)
        out["rules"][r] = s
    for r in Wd:
        if r != "own_gain":
            out["paired"][f"own_gain_minus_{r}"] = dc.paired_difference(Wd["own_gain"], Wd[r], G, n)
    return out
