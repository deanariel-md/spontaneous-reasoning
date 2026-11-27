"""The split pilot: one pilot's baseline calls divided between the two direct-answer baselines.

Design (the item-disjoint 100-item pilots of src/decision_strict.py, unchanged): each pilot is one discovery draw and
100 pilot-pool items; every model gets 100 chain-of-thought calls and 100 baseline calls. Single-baseline pilots spend
the 100 baseline calls on one baseline (requested direct or forced direct). The split pilot assigns half of the pilot's
items to requested direct and half to forced direct, the same assignment for every model of the pilot (balanced within
each stratum: two of the four items of each BIG-Bench Hard task). A model's predicted gain is its mean paired gain over
the 100 items, chain of thought minus the baseline the item was assigned to. Each policy gives chain of thought to the
top third of the panel by its predicted gain (src/decision.choose: decision.third_size(M) models, 10 of 29; models within
estimators.TIE_TOL of the cut share the remaining places equally).

Valuation: the plug-in regret of src/decision.regret_plugin (best third on the evaluation gains minus the chosen third,
pp of gain per chosen model), averaged over the pilots, under each deployment baseline: the requested deployment values
gains over the requested arm, the forced deployment gains over the forced arm. The worst regret of a policy is the larger
of its two. The comparison: the split pilot's worst regret against the better single-baseline pilot's worst regret (the
single-baseline pilot with the smaller worst regret, chosen with hindsight).

Pure functions; arrays: S (configurations, models, items) 0/1 scores; pilots (P,) configuration rows and (P, n) item
positions.
"""
from __future__ import annotations

import numpy as np

from src import decision as dc

SEED = 20261121          # the assignment seed
CHECK_SEEDS = tuple(range(20))  # the 20 further assignment seeds of the robustness check
POLICIES = ("requested_only", "forced_only", "split")
DEPLOYMENTS = ("requested", "forced")
RULE_TEXT = (
    "Split pilot. Pilots: the item-disjoint 100-item one-draw pilots of the strict decision design (one discovery draw "
    "and 100 items from the pilot half of the items each; 200 pilots), as saved. In each pilot, the 100 items are "
    "assigned half to requested direct and half to forced direct by numpy.random.default_rng(20261121): an array of "
    "uniform draws of shape (pilots, strata, items per stratum) is drawn once, each row of the last axis is argsorted, "
    "and the positions whose argsort value is below half the stratum's items (floor) are requested, the rest forced "
    "(one stratum of 100 items where there are no strata; BIG-Bench Hard: 25 tasks of 4 items in the pilot's task order). "
    "The same assignment holds for every model of a pilot. A model's predicted gain is 100 x the mean over the 100 items "
    "of (chain-of-thought score minus the score of the baseline the item was assigned to), strict scoring. The top "
    "third (src/decision.third_size: round(M / 3), 10 of 29) by predicted gain gets chain of thought; models within "
    "estimators.TIE_TOL (1e-9 pp) of the cut share the remaining places equally (src/decision.choose). The two "
    "single-baseline pilots are the strict design's own-gain choices from the same pilots (100 calls on the requested "
    "or the forced arm). Regret: src/decision.regret_plugin, pp of gain per chosen model, averaged over the pilots, "
    "under each deployment baseline (gains over the requested arm; over the forced arm), on the evaluation half of the "
    "items and the validation configurations absent from every discovery draw, each once. Worst regret: the larger of "
    "the two deployment regrets.")


def assignment(n_pilots: int, n_items: int, seed: int = SEED, n_strata: int = 1) -> np.ndarray:
    """(n_pilots, n_items) boolean: True where the item is assigned to requested direct. Within each stratum (consecutive
    blocks of n_items / n_strata positions) floor(k / 2) of the k items are requested."""
    if n_items % n_strata:
        raise ValueError(f"{n_items} items do not split evenly over {n_strata} strata")
    k = n_items // n_strata
    rng = np.random.default_rng(seed)
    rank = rng.random((n_pilots, n_strata, k)).argsort(axis=-1)
    return (rank < k // 2).reshape(n_pilots, n_items)


def pilot_item_scores(S, rows, items) -> np.ndarray:
    """(P, M, n) scores of each pilot's draw and items: S (configurations, M, N); rows (P,); items (P, n)."""
    S = np.asarray(S)
    return np.stack([S[r][:, it] for r, it in zip(np.asarray(rows), np.asarray(items))]).astype(float)


def single_gains(Sb, Sc, rows, items) -> np.ndarray:
    """(P, M) predicted gain in pp from a single-baseline pilot: 100 x mean over the pilot's items of (cot - baseline)."""
    return 100.0 * (pilot_item_scores(Sc, rows, items) - pilot_item_scores(Sb, rows, items)).mean(2)


def split_gains(Sreq, Sfd, Sc, rows, items, rq) -> np.ndarray:
    """(P, M) predicted gain in pp from the split pilot: rq (P, n) True where the item's baseline is requested."""
    rq = np.asarray(rq, bool)[:, None, :]
    base = np.where(rq, pilot_item_scores(Sreq, rows, items), pilot_item_scores(Sfd, rows, items))
    return 100.0 * (pilot_item_scores(Sc, rows, items) - base).mean(2)


def choices(pred, n: int):
    """(P, M) weights of the top n per pilot (ties at the cut share places) and the number of pilots with such a tie."""
    W, T = [], 0
    for p in np.atleast_2d(np.asarray(pred, float)):
        w, t = dc.choose(p, n)
        W.append(w)
        T += int(t > 0)
    return np.array(W), T


def regrets(W, G: dict, n: int) -> dict:
    """{'requested', 'forced', 'worst'}: mean plug-in regret over pilots under each deployment (G: {deployment: (K, M)
    evaluation gains}) and the larger of the two."""
    out = {d: float(dc.regret_plugin(W, G[d], n).mean()) for d in DEPLOYMENTS}
    out["worst"] = max(out["requested"], out["forced"])
    return out


def compare(Wd: dict, G: dict, n: int) -> dict:
    """Every policy's regrets, the better single-baseline policy (smaller worst regret; ties: requested_only), and the
    split pilot's advantage (better single's worst regret minus the split's; positive: the split does better).
    Component paired differences (split minus each single policy, per deployment) with t intervals over evaluation
    configurations (src/decision.paired_difference; negative: the split costs less)."""
    R = {p: regrets(Wd[p], G, n) for p in POLICIES}
    best = min(("requested_only", "forced_only"), key=lambda p: (R[p]["worst"], p != "requested_only"))
    adv = R[best]["worst"] - R["split"]["worst"]
    paired = {}
    for p in ("requested_only", "forced_only"):
        for d in DEPLOYMENTS:
            paired[f"split_minus_{p}/{d}"] = dc.paired_difference(Wd["split"], Wd[p], G[d], n)
    return {"regret": R, "better_single": best, "better_single_worst": R[best]["worst"],
            "split_worst": R["split"]["worst"], "advantage": float(adv), "split_better": bool(adv > 0),
            "equal": bool(adv == 0), "paired": paired}


def leave_one_family_out(pred: dict, G: dict, families) -> dict:
    """compare() with each family left out: the top third (decision.third_size of the models that remain) of each
    policy's saved predictions pred {policy: (P, M)} on the remaining models, G restricted to them."""
    fams = list(families)
    out = {"by_family": {}}
    for f in sorted(set(fams)):
        keep = np.array([j for j, x in enumerate(fams) if x != f])
        if len(keep) < 2:
            out["by_family"][f] = {"note": "fewer than two models remain"}
            continue
        n = dc.third_size(len(keep))
        Wd = {p: choices(np.atleast_2d(pred[p])[:, keep], n)[0] for p in POLICIES}
        c = compare(Wd, {d: np.asarray(G[d], float)[:, keep] for d in DEPLOYMENTS}, n)
        out["by_family"][f] = {"models_left": int(len(keep)), "n_chosen": n, "regret": c["regret"],
                               "better_single": c["better_single"], "advantage": c["advantage"],
                               "split_better": c["split_better"]}
    adv = [v["advantage"] for v in out["by_family"].values() if "advantage" in v]
    out["advantage_range"] = [float(min(adv)), float(max(adv))] if adv else None
    out["split_better_count"] = int(sum(v.get("split_better", False) for v in out["by_family"].values()))
    out["families"] = len(out["by_family"])
    return out
