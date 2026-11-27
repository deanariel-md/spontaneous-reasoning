"""Both directions of the wrong-baseline decision cost, and the model-level uncertainty of the arm-switch reversal
share.

A decision gives the strategy to the third of the panel predicted to gain most (src/decision.py). Choices made under
one baseline arm (requested direct or forced direct) are valued under the other. The extra regret of a choice made
under arm a and deployed under arm b is the regret of that choice on b's validation gains minus the regret of the choice
made under b on the same pilot sets, in pp of accuracy gain per chosen model (src/decision.paired_difference: linear in
the draws, so the best third cancels; t interval over evaluation configurations; positive: the wrong baseline costs
points). Nothing here makes a choice from validation data: the choices are the saved ones, and the leave-one-family-out
ranges re-take the top third of the saved discovery predictions on the models that remain.

Arrays: draws or configurations by models, pp.
"""
from __future__ import annotations

import numpy as np

from src import decision as dc
from src.estimators import tie_sign

DIRECTIONS = (("requested_to_forced", "requested", "forced"), ("forced_to_requested", "forced", "requested"))


def both_directions(W: dict, G: dict, n: int) -> dict:
    """W: {'requested', 'forced'} choice weights (sets, M); G: {'requested', 'forced'} evaluation gains (K, M).
    {direction: paired difference} for choosing under one arm and deploying under the other."""
    out = {}
    for name, src, dst in DIRECTIONS:
        d = dc.paired_difference(W[src], W[dst], G[dst], n)
        d["regret_mismatched"] = float(dc.regret_plugin(W[src], G[dst], n).mean())
        d["regret_matched"] = float(dc.regret_plugin(W[dst], G[dst], n).mean())
        out[name] = d
    return out


def rechoose(pred, keep, n: int) -> np.ndarray:
    """(sets, len(keep)) weights: the top n of each saved prediction row restricted to the kept models."""
    P = np.atleast_2d(np.asarray(pred, float))[:, keep]
    return np.array([dc.choose(p, n)[0] for p in P])


def leave_one_family_out(pred: dict, G: dict, families) -> dict:
    """Both directions with each family left out: the top third (decision.third_size of the models that remain) of the
    saved discovery predictions pred {'requested', 'forced'} (sets, M), valued on G {'requested', 'forced'} (K, M)
    restricted to the same models. {'by_family': {family: {direction: estimate}}, 'range': {direction: [min, max]}}."""
    fams = list(families)
    out = {"by_family": {}, "range": {}}
    for f in sorted(set(fams)):
        keep = np.array([j for j, x in enumerate(fams) if x != f])
        if len(keep) < 2:
            out["by_family"][f] = {"note": "fewer than two models remain"}
            continue
        n = dc.third_size(len(keep))
        W = {a: rechoose(pred[a], keep, n) for a in ("requested", "forced")}
        Gk = {a: np.asarray(G[a], float)[:, keep] for a in ("requested", "forced")}
        res = both_directions(W, Gk, n)
        out["by_family"][f] = {"models_left": int(len(keep)), "n_chosen": n,
                               **{k: v["estimate"] for k, v in res.items()}}
    for name, _, _ in DIRECTIONS:
        v = [x[name] for x in out["by_family"].values() if name in x]
        out["range"][name] = [float(min(v)), float(max(v))] if v else None
    return out


# --------------------------------------------------------------------------- reversal share over models


def reversal_matrix(g1, g2) -> np.ndarray:
    """(M, M) boolean: models a and b ordered one way by g1 and the other by g2 (ties under either are not reversed;
    the convention of src/robustness.pair_reversals)."""
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    s1 = tie_sign(g1[:, None] - g1[None, :])
    s2 = tie_sign(g2[:, None] - g2[None, :])
    return (s1 * s2) < 0


def share_on(Rm, ix) -> float:
    """Reversed share among the pairs of positions i < j of the index array ix whose models differ (a model drawn
    twice is not a pair)."""
    ix = np.asarray(ix)
    i, j = np.triu_indices(len(ix), 1)
    ok = ix[i] != ix[j]
    if not ok.any():
        return float("nan")
    return float(Rm[ix[i][ok], ix[j][ok]].mean())


def reversal_uncertainty(g1, g2, families, n_boot: int, rng) -> dict:
    """The reversed share of model pairs between gains g1 and g2 (model means), with a model bootstrap (models drawn
    with replacement, pairs of a model with its own copy left out), a family bootstrap (families drawn whole), each
    family left out, and the count among pairs within one family."""
    Rm = reversal_matrix(g1, g2)
    M = len(Rm)
    fams = list(families)
    obs = share_on(Rm, np.arange(M))
    boot = np.array([share_on(Rm, rng.integers(0, M, M)) for _ in range(n_boot)])
    groups = {}
    for j, f in enumerate(fams):
        groups.setdefault(f, []).append(j)
    keys = sorted(groups)
    fboot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        fboot.append(share_on(Rm, np.concatenate([groups[keys[p]] for p in pick])))
    fboot = np.array(fboot)
    lofo = {f: share_on(Rm, np.array([j for j in range(M) if fams[j] != f])) for f in keys}
    i, j = np.triu_indices(M, 1)
    same = np.array([fams[a] == fams[b] for a, b in zip(i, j)])
    rev = Rm[i, j]

    def pct(v):
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None
    return {"M": M, "pairs": int(len(i)), "reversed": int(rev.sum()), "share": obs,
            "model_bootstrap": {"n_boot": int(n_boot), "ci95": pct(boot), "n_undefined": int((~np.isfinite(boot)).sum())},
            "family_bootstrap": {"n_boot": int(n_boot), "ci95": pct(fboot),
                                 "n_undefined": int((~np.isfinite(fboot)).sum()), "n_families": len(keys)},
            "leave_one_family_out": lofo,
            "leave_one_family_out_range": [float(min(lofo.values())), float(max(lofo.values()))],
            "within_family": {"pairs": int(same.sum()), "reversed": int(rev[same].sum())},
            "between_family": {"pairs": int((~same).sum()), "reversed": int(rev[~same].sum())}}
