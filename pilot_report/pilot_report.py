r"""Pilot report: how far do K independently drawn prompt configurations agree on which models benefit more, and does the
claim of who benefits hold under both direct-answer baselines?

Input: a long CSV with columns model, draw, arm, score. A draw is one prompt configuration applied to every item;
score is the model's accuracy on that draw under that arm, in percentage points (0-100). Other columns are ignored.
Two arms are named on the command line: the baseline and the intervention. Benefit is intervention - baseline, per
model and draw.

Optional input for the comparison-arm check: a CSV with columns model, arm, n_output_tokens, one row per output, or
one row per (model, arm, length) with a `count` column giving the number of outputs of that length. The check reads
the baseline arm's rows.

Output: M, K; for benefit and for accuracy under each arm, the reliability R across draws, the observed one-draw
ordering agreement C between distinct draws, the reference C_ref = 1/2 + arcsin(R)/pi, the draws per evaluation that
80% and 90% agreement need, and an unresolved flag with its reason; each model's mean scores and gain; the
comparison-arm check. A printed report, and a JSON file with --json.

Optional blocks, each added only by its option (with none of them the report is that of version 1.0.0):
- --forced ARM: a second baseline. --baseline is then the requested baseline (the model is asked for the final answer
  only) and ARM the forced baseline (the opening of the answer is written into the reply and the model may add at most
  16 tokens). The report gives R and C for benefit and accuracy under both baselines; per model, the gain under each
  baseline, their difference and, with --tokens, w (the share of requested replies longer than the token budget, 16 by
  default); and the block "who benefits".
- Who benefits, under each baseline: the capability slope of gain on the model's own-baseline accuracy, and with
  --capability on a capability score per model (for example the Epoch Capabilities Index); the top third of models by
  mean gain; with two baselines, how many models the two top thirds share and the share of model pairs the two
  baselines order oppositely by gain.
- --items FILE (needs --forced): the split pilot. Item-level scores of the pilot (columns model, draw, item and either
  arm and score, or one column per arm; scores 1 correct, 0 wrong). Half of each draw's items are assigned to each
  baseline, the same assignment for every model (seed --seed), and the intervention goes to the top third of models by
  mean paired gain. A file with each item scored under one baseline only (a split pilot already run) gives the
  assignment itself.
- --decision: under each baseline, the third chosen by own mean gain and the third chosen by the capability slope, how
  many models they share, and with at least four draws the split-half regret of each (choose on half the draws, value
  on the other half).

Usage:
    python pilot_report.py scores.csv --baseline direct_answer --intervention zero_shot_cot
    python pilot_report.py scores.csv --baseline direct_answer --intervention zero_shot_cot \
        --tokens tokens.csv --token-budget 16 --max-over-budget 0.05 --json report.json
    python pilot_report.py scores.csv --baseline direct_answer --forced forced_direct --intervention zero_shot_cot \
        --tokens tokens.csv --capability capability.csv --items items.csv --decision

From Python:
    from pilot_report import pilot_report, format_report
    res = pilot_report("scores.csv", "direct_answer", "zero_shot_cot", tokens="tokens.csv")
    print(format_report(res))

Estimators (arrays are draws by models, in percentage points):
- R = S / T. S is the cross-draw panel moment over distinct draws: within each draw the scores are centred over
  models, and S averages the products of two different draws' centred scores of the same model over all ordered
  pairs of distinct draws, with the panel denominator M - 1. T is the mean over draws of the variance across models
  (denominator M - 1). R is reported as estimated; a negative S gives a negative R, which is never set to zero.
- C: for every model pair and every ordered pair of distinct draws, 1 when both draws order the pair the same way,
  1/2 when either draw ties it, 0 when they disagree; averaged over draw pairs, then with equal weights over model
  pairs. A difference of at most 1e-9 pp is a tie.
- C_ref = 1/2 + arcsin(R)/pi, the Gaussian reference for the agreement of two independent draws. For the mean of k
  draws, R_k = k R / (1 + (k - 1) R) (Spearman-Brown) and C_ref(k) = 1/2 + arcsin(R_k)/pi.
- Draws for a target agreement a: the smallest k >= 1 with C_ref(k) >= a, that is
  k = max(1, ceil(s (1 - R) / (R (1 - s)))) with s = sin(pi (a - 1/2)).
- Unresolved: when R is not finite (fewer than two draws, or zero within-draw variance) or R <= 0 (no signal beyond
  prompt noise), no draw count is given.
- Capability slope of gain G on x: A / V_x, with A the cross-draw moment of G and x and V_x that of x with itself (the
  moment of R's signal, so x and G come from different draws). x is the baseline's accuracy, or a capability score
  repeated over draws. A V_x of zero or below leaves the slope undefined; V_x and A are reported as they are.
- Top third: round(M / 3) models (at least 1, at most M - 1) with the highest values; models within 1e-9 of the cut
  share the remaining places equally, so a model can be chosen in part. Two thirds share the sum over models of the
  smaller of their two weights.
- Opposite orderings: a model pair is ordered oppositely when the two baselines' mean gains order it in opposite
  directions; a pair with a gap of at most 1e-9 pp under either is tied, neither opposite nor kept. With a threshold
  t, only pairs whose two gaps both exceed t (by more than 1e-9 pp) count.
- Split pilot: in each draw the items are ranked by uniform numbers from numpy.random.default_rng(seed) (one array of
  shape (draws, strata, items per stratum), drawn once and argsorted along its last axis); the floor(k / 2) of each
  stratum's k items with the lowest ranks go to the requested baseline and the rest to the forced baseline. A model's
  split gain is 100 x the mean over the pilot's items of (intervention score - the score under the item's baseline).
- Capability third: the prediction mean(G) + slope x (mean accuracy - its mean over models), the slope from all draws;
  an undefined slope ties every model. Split-half regret: choose a third on one half of the draws, then on the other
  half take the mean gain of the best third minus that of the chosen third, per chosen model (pp). Every split of the
  draws into halves of floor(K/2) and ceil(K/2) draws enters, in both directions, when there are at most 20,000 such
  choices; otherwise 10,000 random splits (seed --seed), both directions.

Requires numpy and pandas only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import textwrap

import numpy as np
import pandas as pd

__version__ = "1.1.0"

TIE_TOL = 1e-9          # a model-pair difference of at most this many pp is a tie
RESIDUE = 1e-12         # floating-point residue at the ends of [0, 1]
TARGETS = (0.80, 0.90)
# A forced direct answer takes at most 16 tokens; a requested direct answer longer than that on more than 5% of items
# is flagged. The study's admissibility screen used a looser rule (more than half above 128 tokens:
# --token-budget 128 --max-over-budget 0.5), which every model of its panels passes by construction.
TOKEN_BUDGET = 16
MAX_OVER_BUDGET = 0.05
MIN_PILOT_DRAWS = 4
FLAG_MESSAGE = "comparison arm does not do what its name says"
SPLIT_SEED = 20261121                       # the study's assignment seed for the split pilot
THRESHOLDS = (0.0, 1.0, 2.0, 5.0, 10.0)     # pp; gap thresholds of the opposite orderings
MAX_SPLIT_CHOICES = 20000                   # split-half regret: every split up to this many choices, else random splits
RANDOM_SPLITS = 10000


# --------------------------------------------------------------------------- estimators


def cross_moment(X, Y):
    """Cross-draw panel moment over distinct draws, denominator M - 1. X, Y: (K, M) with K, M >= 2."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    K, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return float((Xc.sum(0) @ Yc.sum(0) - np.sum(Xc * Yc)) / (K * (K - 1) * (M - 1)))


def reliability(X):
    """signal S, mean within-draw variance T and R = S / T (NaN when T = 0). X: (K, M) with K, M >= 2."""
    X = np.asarray(X, float)
    signal = cross_moment(X, X)
    total = float(np.var(X, axis=1, ddof=1).mean())
    return {"signal": signal, "total": total, "R": signal / total if total > 0 else float("nan")}


def tie_sign(D, tol=TIE_TOL):
    D = np.asarray(D, float)
    return np.where(np.abs(D) <= tol, 0.0, np.sign(D))


def agreement(X, tol=TIE_TOL):
    """One-draw ordering agreement between distinct draws (rows of X), ties counted one half."""
    X = np.asarray(X, float)
    K, M = X.shape
    if M < 2 or K < 2:
        raise ValueError("agreement needs two or more models and two or more draws")
    i, j = np.triu_indices(M, 1)
    s = tie_sign(X[:, i] - X[:, j], tol)
    pos, neg = (s > 0).sum(0).astype(float), (s < 0).sum(0).astype(float)
    d = K * (K - 1)
    same = float(np.mean((pos * (pos - 1) + neg * (neg - 1)) / d))
    opposite = float(np.mean(2 * pos * neg / d))
    tie = 1.0 - same - opposite
    tie = 0.0 if abs(tie) < 1e-12 else tie            # floating-point residue only
    dec = same + opposite
    return {"C": same + 0.5 * tie, "same": same, "opposite": opposite, "tie": tie, "decisive": dec,
            "conditional_agreement": same / dec if dec > 0 else float("nan")}


def _unit(R):
    """R with floating-point residue at the ends of [0, 1] removed; nothing else."""
    if R is not None and math.isfinite(R):
        if 1.0 < R <= 1.0 + RESIDUE:
            return 1.0
        if -RESIDUE <= R < 0.0:
            return 0.0
    return R


def r_defined(R) -> bool:
    R = _unit(R)
    return R is not None and math.isfinite(R) and 0.0 <= R <= 1.0


def r_status(R) -> str:
    if R is None or not math.isfinite(R):
        return "undefined (zero within-draw variance)"
    R = _unit(R)
    if R < 0:
        return "negative estimate"
    if R > 1:
        return "above one"
    return "in [0, 1]"


def spearman_brown(R, k) -> float:
    """R_k = k R / (1 + (k - 1) R); NaN when R is not in [0, 1]."""
    if not r_defined(R):
        return float("nan")
    R = _unit(R)
    return k * R / (1 + (k - 1) * R)


def reference_c(R, k=1) -> float:
    """C_ref = 1/2 + arcsin(R_k)/pi for the mean of k draws; NaN when R is not in [0, 1]."""
    if not r_defined(R):
        return float("nan")
    R = _unit(R)
    RK = k * R / (1 + (k - 1) * R)
    return 0.5 + math.asin(RK) / math.pi


def prompts_needed(R, target):
    """Smallest k >= 1 whose k-draw mean reaches the target agreement under the reference (R in [0, 1])."""
    if not 0 <= R <= 1 or not 0.5 < target < 1:
        raise ValueError("Require R in [0,1] and target in (0.5,1).")
    if R == 0:
        return math.inf
    if R == 1:
        return 1
    star = math.sin(math.pi * (target - 0.5))
    return max(1, math.ceil(star * (1 - R) / (R * (1 - star))))


def draws_needed(R, target):
    """Draws per evaluation for the target agreement; None (unresolved) when R is not finite or R <= 0."""
    if R is None or not np.isfinite(R) or R <= 0:
        return None
    return int(prompts_needed(min(float(R), 1.0), target))


def weighted_median(values, counts):
    """np.median of the values repeated counts times, without expanding them."""
    v, c = np.asarray(values, float), np.asarray(counts, np.int64)
    keep = c > 0
    v, c = v[keep], c[keep]
    if c.sum() == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, cum = v[order], np.cumsum(c[order])
    n = int(cum[-1])

    def nth(k):                                       # k-th smallest, 0-based
        return v[int(np.searchsorted(cum, k + 1))]

    return float(nth(n // 2)) if n % 2 else float((nth(n // 2 - 1) + nth(n // 2)) / 2)


# --------------------------------------------------------------------------- who benefits: estimators


def cross_slope(X, Y):
    """Slope of Y on X from cross-draw moments: A / V_x with A = cross_moment(Y, X) and V_x = cross_moment(X, X).
    X, Y: (K, M) with K, M >= 2. A V_x of zero or below gives slope NaN; A and V_x are returned as they are."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    vx = cross_moment(X, X)
    a = cross_moment(Y, X)
    return {"slope": a / vx if vx > 0 else float("nan"), "A": a, "V_x": vx}


def third_size(M: int) -> int:
    """round(M / 3), at least 1 and at most M - 1: 10 of 29, 9 of 28, 2 of 5."""
    if M < 2:
        raise ValueError("a panel of at least two models is needed")
    return int(min(M - 1, max(1, round(M / 3))))


def choose(values, n: int, tol: float = TIE_TOL):
    """Weights of the n highest values and the number of models tied at the cut (0 when the cut is clean). Models
    within tol of the cut share the remaining places equally."""
    p = np.asarray(values, float)
    if not np.all(np.isfinite(p)):
        raise ValueError("a value to choose by is not finite")
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


def shared(w1, w2) -> float:
    """Models two choices share: the sum over models of the smaller weight (a count when no model is chosen in part)."""
    return float(np.minimum(np.asarray(w1, float), np.asarray(w2, float)).sum())


def pair_reversals(g1, g2, thresholds=THRESHOLDS, tol=TIE_TOL):
    """Model pairs that mean gains g1 order one way and g2 the other way. A pair with a gap of at most tol under either
    is tied, neither opposite nor kept. For each threshold t, the opposite pairs among those whose two gaps both exceed
    t; a gap within tol of t is at t, not above it."""
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    i, j = np.triu_indices(len(g1), 1)
    d1, d2 = g1[i] - g1[j], g2[i] - g2[j]
    s1, s2 = tie_sign(d1, tol), tie_sign(d2, tol)
    rev = (s1 * s2) < 0
    out = {"pairs": int(len(i)), "reversed": int(rev.sum()), "reversed_share": float(rev.mean()),
           "ties": int(((s1 == 0) | (s2 == 0)).sum()), "by_threshold": []}
    m = np.minimum(np.abs(d1), np.abs(d2))
    for t in thresholds:
        big = m > t + tol
        out["by_threshold"].append({"min_gap_pp": float(t), "pairs_both_above": int(big.sum()),
                                    "reversed": int((rev & big).sum()),
                                    "reversed_share_of_all_pairs": float((rev & big).mean())})
    out["reversed_index"] = [(int(a), int(b)) for a, b in zip(i[rev], j[rev])]
    return out


def capability_prediction(B, G):
    """(prediction, slope): mean(G) + slope x (mean accuracy - its mean over models), with the cross-draw slope of G on
    B over all rows; an undefined slope gives every model the same prediction."""
    B, G = np.asarray(B, float), np.asarray(G, float)
    s = cross_slope(B, G)["slope"]
    gb, bb = G.mean(0), B.mean(0)
    if not np.isfinite(s):
        return np.full(len(gb), gb.mean()), s
    return gb.mean() + s * (bb - bb.mean()), s


def regret(W, Gv, n: int):
    """Mean gain of the best third on the evaluation draws Gv minus that of the choice W, per chosen model (pp)."""
    gm = np.asarray(Gv, float).mean(0)
    wb = choose(gm, n)[0]
    return (wb @ gm - np.atleast_2d(W) @ gm) / n


def half_splits(K: int, seed: int = SPLIT_SEED):
    """(choice rows, evaluation rows) pairs: every split into halves of floor(K/2) and ceil(K/2) draws in both
    directions when there are at most MAX_SPLIT_CHOICES, else RANDOM_SPLITS seeded random splits in both directions.
    Returns (pairs, exhaustive)."""
    h = K // 2
    n_exact = math.comb(K, h) * (1 if K % 2 == 0 else 2)
    pairs = []
    if n_exact <= MAX_SPLIT_CHOICES:
        for a in itertools.combinations(range(K), h):
            b = [k for k in range(K) if k not in a]
            pairs.append((list(a), b))
            if K % 2:
                pairs.append((b, list(a)))
        return pairs, True
    rng = np.random.default_rng(seed)
    for _ in range(RANDOM_SPLITS):
        p = rng.permutation(K)
        a, b = sorted(int(x) for x in p[:h]), sorted(int(x) for x in p[h:])
        pairs.extend([(a, b), (b, a)])
    return pairs, False


def split_half_regret(B, G, n: int, seed: int = SPLIT_SEED):
    """Split-half regret of the own-gain third and the capability third (K >= 4): mean, smallest and largest over the
    splits of half_splits, and the number of choices whose capability slope was undefined."""
    B, G = np.asarray(B, float), np.asarray(G, float)
    K = G.shape[0]
    if K < MIN_PILOT_DRAWS:
        return None
    pairs, exhaustive = half_splits(K, seed)
    own, cap, undefined = [], [], 0
    for a, b in pairs:
        w_own = choose(G[a].mean(0), n)[0]
        pred, s = capability_prediction(B[a], G[a])
        undefined += int(not np.isfinite(s))
        w_cap = choose(pred, n)[0]
        own.append(float(regret(w_own, G[b], n)[0]))
        cap.append(float(regret(w_cap, G[b], n)[0]))

    def summ(v):
        return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}

    return {"n_choices": len(pairs), "exhaustive": exhaustive, "half_sizes": [K // 2, K - K // 2],
            "own_gain": summ(own), "capability": summ(cap), "capability_slope_undefined": undefined}


def split_assignment(n_draws: int, n_items: int, seed: int = SPLIT_SEED, n_strata: int = 1):
    """(n_draws, n_items) boolean, True where the item goes to the requested baseline: within each stratum (consecutive
    blocks of n_items / n_strata positions) floor(k / 2) of the k items."""
    if n_items % n_strata:
        raise ValueError(f"{n_items} items do not split evenly over {n_strata} strata")
    k = n_items // n_strata
    rng = np.random.default_rng(seed)
    rank = rng.random((n_draws, n_strata, k)).argsort(axis=-1)
    return (rank < k // 2).reshape(n_draws, n_items)


# --------------------------------------------------------------------------- one quantity


def quantity_block(X, label, targets=TARGETS):
    """R, C, the reference, the draws for each target and the unresolved flag of one (K, M) array."""
    X = np.asarray(X, float)
    K, M = X.shape
    out = {"quantity": label, "K": int(K), "M": int(M)}
    if K < 2:
        out.update({"R": None, "R_status": "undefined (fewer than two draws)", "signal": None, "within": None, "noise": None,
                    "C": None, "same": None, "opposite": None, "tie": None, "C_ref": None,
                    "R_K": None, "C_ref_K": None, "draws_needed": {f"{t:.2f}": None for t in targets},
                    "unresolved": True,
                    "reason": "fewer than two draws: R and C need two or more independent draws"})
        return out
    r = reliability(X)
    R = r["R"]
    ag = agreement(X)
    out.update({"R": R, "R_status": r_status(R), "signal": r["signal"], "within": r["total"],
                "noise": r["total"] - r["signal"], "C": ag["C"], "same": ag["same"], "opposite": ag["opposite"],
                "tie": ag["tie"], "C_ref": reference_c(R), "R_K": spearman_brown(R, K), "C_ref_K": reference_c(R, K),
                "draws_needed": {f"{t:.2f}": draws_needed(R, t) for t in targets}})
    if not math.isfinite(R):
        reason = (f"zero within-draw variance: every draw gives all {M} models the same {label}, so R is "
                  "undefined")
    elif R < 0:
        reason = (f"negative signal (S = {r['signal']:.4g} pp^2, R = {R:.4g}): the draws detect no model "
                  f"differences in {label} beyond prompt noise; the estimate is reported as it is")
    elif R == 0:
        reason = f"zero signal (S = 0): the draws detect no model differences in {label} beyond prompt noise"
    else:
        reason = None
    out["unresolved"] = reason is not None
    out["reason"] = reason
    if reason is None and r["total"] - r["signal"] <= RESIDUE * max(1.0, r["total"]):
        out["note"] = "no prompt noise: the draws agree exactly, so one draw suffices"
    return out


# --------------------------------------------------------------------------- inputs


def read_csv(path):
    """CSV with exact float parsing (pandas' default parser can differ from Python's in the last digit). Lines at the
    top of the file that start with "# " are comments (for example a source credit) and are skipped."""
    skip = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("# "):
                break
            skip += 1
    return pd.read_csv(path, float_precision="round_trip", skiprows=skip)


def arm_arrays(scores, arms):
    """({arm: (K, M) array in pp}, models, draws, notes) from the long score table for the named arms."""
    df = read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
    need = {"model", "draw", "arm", "score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"the score table lacks the columns {sorted(miss)}")
    if len(set(arms)) != len(arms):
        raise ValueError("the baseline and intervention arms must differ")
    df = df[["model", "draw", "arm", "score"]].copy()
    for c in ("model", "draw", "arm"):
        if df[c].isna().any():
            raise ValueError(f"the score table has empty {c} entries")
        df[c] = df[c].astype(str)
    notes = []
    present = list(pd.unique(df["arm"]))
    for a in arms:
        if a not in present:
            raise ValueError(f"arm {a!r} is not in the score table (arms present: {present})")
    other = [a for a in present if a not in arms]
    if other:
        notes.append(f"rows of other arms are ignored: {other}")
    df = df[df["arm"].isin(list(arms))]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    bad = df[~np.isfinite(df["score"].to_numpy(dtype=float))]
    if len(bad):
        raise ValueError(f"{len(bad)} scores are missing or not finite, e.g. {bad.head(3).to_dict('records')}")
    dup = df.duplicated(["model", "draw", "arm"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} rows repeat a (model, draw, arm) cell, e.g. "
                         f"{df[dup].head(3).to_dict('records')}")
    models, draws = list(pd.unique(df["model"])), list(pd.unique(df["draw"]))
    if len(models) < 2:
        raise ValueError("the report needs two or more models")
    out = {}
    for a in arms:
        piv = df[df["arm"] == a].pivot(index="draw", columns="model", values="score").reindex(index=draws,
                                                                                            columns=models)
        hole = piv.isna()
        if hole.to_numpy().any():
            cells = [(d, m) for d in draws for m in models if hole.loc[d, m]]
            raise ValueError(f"arm {a!r} lacks {len(cells)} (draw, model) cells, e.g. {cells[:3]}; every model "
                             f"needs a score on every draw under {'both' if len(arms) == 2 else 'every'} arm"
                             f"{'s' if len(arms) == 2 else ''} (nothing is imputed)")
        out[a] = piv.to_numpy(dtype=float)
    if all(np.all((X >= 0) & (X <= 1)) for X in out.values()):
        notes.append("every score lies in [0, 1]: scores are expected in percentage points (0-100); R and C do "
                     "not depend on the unit, but the moments in pp^2 do")
    return out, models, draws, notes


def score_arrays(scores, baseline, intervention):
    """(B, P, models, draws, notes) from the long score table; B and P are (K, M) in pp."""
    X, models, draws, notes = arm_arrays(scores, (baseline, intervention))
    return X[baseline], X[intervention], models, draws, notes


def token_check(tokens, arm, models, budget=TOKEN_BUDGET, max_share=MAX_OVER_BUDGET):
    """Per model, the share of the arm's outputs longer than `budget` tokens; flagged when above `max_share`."""
    df = read_csv(tokens) if not isinstance(tokens, pd.DataFrame) else tokens.copy()
    need = {"model", "arm", "n_output_tokens"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"the token table lacks the columns {sorted(miss)}")
    df["model"], df["arm"] = df["model"].astype(str), df["arm"].astype(str)
    df = df[df["arm"] == arm].copy()
    if df.empty:
        raise ValueError(f"the token table has no rows for the baseline arm {arm!r}")
    df["n_output_tokens"] = pd.to_numeric(df["n_output_tokens"], errors="coerce")
    if df["n_output_tokens"].isna().any() or (df["n_output_tokens"] < 0).any():
        raise ValueError("n_output_tokens must be a nonnegative number on every row")
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        if df["count"].isna().any() or (df["count"] < 0).any() or (df["count"] % 1 != 0).any():
            raise ValueError("count must be a nonnegative integer on every row")
    else:
        df["count"] = 1
    rows = []
    for m, g in df.groupby("model", sort=False):
        n = int(g["count"].sum())
        over = int(g.loc[g["n_output_tokens"] > budget, "count"].sum())
        share = over / n if n > 0 else float("nan")
        rows.append({"model": m, "n_outputs": n, "n_over_budget": over, "share_over_budget": share,
                     "median_tokens": weighted_median(g["n_output_tokens"], g["count"]),
                     "max_tokens": float(g.loc[g["count"] > 0, "n_output_tokens"].max()) if n > 0 else float("nan"),
                     "flagged": bool(n > 0 and share > max_share)})
    order = {m: i for i, m in enumerate(models)}
    rows.sort(key=lambda r: (order.get(r["model"], len(order)), r["model"]))
    present = {r["model"] for r in rows}
    return {"arm": arm, "token_budget": budget, "max_over_budget": max_share,
            "rule": f"flagged when more than {max_share:.0%} of the {arm} outputs are longer than {budget} tokens",
            "models": rows, "flagged": [r["model"] for r in rows if r["flagged"]],
            "not_checked": [m for m in models if m not in present],
            "not_in_scores": sorted(present - set(models)), "message": FLAG_MESSAGE}


def capability_scores(capability, models):
    """A capability score per model from a CSV (columns model, capability). Returns {'values': (M_kept,), 'keep':
    column indices of the score table's models that have a value, 'missing', 'not_in_scores'}."""
    df = read_csv(capability) if not isinstance(capability, pd.DataFrame) else capability.copy()
    miss = {"model", "capability"} - set(df.columns)
    if miss:
        raise ValueError(f"the capability table lacks the columns {sorted(miss)}")
    if df["model"].isna().any():
        raise ValueError("the capability table has empty model entries")
    df["model"] = df["model"].astype(str)
    if df["model"].duplicated().any():
        raise ValueError(f"the capability table repeats models: {sorted(set(df.loc[df['model'].duplicated(), 'model']))}")
    raw = df["capability"]
    val = pd.to_numeric(raw, errors="coerce")
    bad = raw.notna() & ~np.isfinite(val.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(f"capability values are not finite numbers for {df.loc[bad, 'model'].tolist()[:3]}")
    have = {m: float(v) for m, v in zip(df["model"], val) if np.isfinite(v)}
    keep = [j for j, m in enumerate(models) if m in have]
    return {"values": np.array([have[models[j]] for j in keep], float), "keep": keep,
            "missing": [m for m in models if m not in have],
            "not_in_scores": sorted(set(df["model"]) - set(models))}


def item_scores(items, requested, forced, intervention, models, seed=SPLIT_SEED):
    """Item-level scores of the split pilot and its assignment. Long form (columns model, draw, item, arm, score) or
    one column per arm (model, draw, item, <arms>); an optional `stratum` column balances the seeded assignment within
    strata. Scores in [0, 1]. Returns a dict with the (M, C) arrays over the pilot's (draw, item) cells in file order,
    the boolean requested-assignment per cell and its description."""
    df = read_csv(items) if not isinstance(items, pd.DataFrame) else items.copy()
    keys = ["model", "draw", "item"]
    miss = set(keys) - set(df.columns)
    if miss:
        raise ValueError(f"the item table lacks the columns {sorted(miss)}")
    arms3 = (requested, forced, intervention)
    has_stratum = "stratum" in df.columns
    for c in keys + (["stratum"] if has_stratum else []):
        if df[c].isna().any():
            raise ValueError(f"the item table has empty {c} entries")
        df[c] = df[c].astype(str)
    if {"arm", "score"} <= set(df.columns):
        form = "long"
        if df["arm"].isna().any():
            raise ValueError("the item table has empty arm entries")
        df["arm"] = df["arm"].astype(str)
        df = df[df["arm"].isin(arms3)].copy()
        sc = pd.to_numeric(df["score"], errors="coerce")
        if (~np.isfinite(sc.to_numpy(dtype=float))).any():
            raise ValueError("the item table has missing or non-numeric scores")
        df["score"] = sc
        dup = df.duplicated(keys + ["arm"], keep=False)
        if dup.any():
            raise ValueError(f"{int(dup.sum())} item rows repeat a (model, draw, item, arm) cell, e.g. "
                             f"{df[dup].head(3).to_dict('records')}")
        wide = df.pivot(index=keys, columns="arm", values="score")
        for a in arms3:
            if a not in wide.columns:
                wide[a] = np.nan
    else:
        form = "wide"
        absent = [a for a in arms3 if a not in df.columns]
        if absent:
            raise ValueError(f"the item table has neither the columns arm and score nor a column for each arm "
                             f"(missing {absent})")
        dup = df.duplicated(keys, keep=False)
        if dup.any():
            raise ValueError(f"{int(dup.sum())} item rows repeat a (model, draw, item) cell, e.g. "
                             f"{df[dup][keys].head(3).to_dict('records')}")
        for a in arms3:
            v = pd.to_numeric(df[a], errors="coerce")
            if (df[a].notna() & v.isna()).any():
                raise ValueError(f"the item table's column {a!r} has non-numeric entries")
            df[a] = v
        wide = df.set_index(keys)[list(arms3)]
    got = list(pd.unique(df["model"]))
    if set(got) != set(models):
        raise ValueError(f"the item table's models differ from the score table's: only in the item table "
                         f"{sorted(set(got) - set(models))}, only in the score table {sorted(set(models) - set(got))}")
    cells = df[["draw", "item"]].drop_duplicates()
    cell_list = list(zip(cells["draw"], cells["item"]))
    idx = pd.MultiIndex.from_tuples([(m, d, i) for m in models for d, i in cell_list], names=keys)
    A = {a: wide[a].reindex(idx).to_numpy(dtype=float).reshape(len(models), len(cell_list)) for a in arms3}
    for a, X in A.items():
        ok = X[np.isfinite(X)]
        if ok.size and ((ok < 0) | (ok > 1)).any():
            raise ValueError(f"item scores of arm {a!r} must lie in [0, 1] (1 correct, 0 wrong)")
    hole = ~np.isfinite(A[intervention])
    if hole.any():
        m, c = np.argwhere(hole)[0]
        raise ValueError(f"the item table lacks {int(hole.sum())} intervention scores, e.g. model {models[m]!r}, "
                         f"draw {cell_list[c][0]!r}, item {cell_list[c][1]!r}; every model needs every item")
    pR, pF = np.isfinite(A[requested]), np.isfinite(A[forced])
    draw_of = np.array([d for d, _ in cell_list], dtype=object)
    draws = list(pd.unique(cells["draw"]))
    info = {"form": form, "draws": draws}
    if pR.all() and pF.all():
        # the seeded assignment: draws in order of first appearance, items in file order within a draw (grouped by
        # stratum, strata in order of first appearance, when there is a stratum column)
        strata_of = None
        if has_stratum:
            st = df[["draw", "item", "stratum"]].drop_duplicates()
            if st.duplicated(["draw", "item"]).any():
                raise ValueError("an item has more than one stratum within a draw")
            smap = {(d, i): s for d, i, s in zip(st["draw"], st["item"], st["stratum"])}
            strata_of = [smap[c] for c in cell_list]
        order, n_strata, n_items = [], None, None
        for d in draws:
            pos = [c for c in range(len(cell_list)) if draw_of[c] == d]
            if strata_of is not None:
                labs = list(dict.fromkeys(strata_of[c] for c in pos))
                sizes = [sum(strata_of[c] == s for c in pos) for s in labs]
                if len(set(sizes)) != 1:
                    raise ValueError(f"draw {d!r}: the strata hold different numbers of items {sizes}; the seeded "
                                     "assignment needs equal strata")
                pos = [c for s in labs for c in pos if strata_of[c] == s]
                ns = len(labs)
            else:
                ns = 1
            if n_items is None:
                n_items, n_strata = len(pos), ns
            elif len(pos) != n_items or ns != n_strata:
                raise ValueError(f"the seeded assignment needs the same number of items (and strata) in every draw; "
                                 f"draw {d!r} has {len(pos)} items in {ns} strata, the first draw {n_items} in "
                                 f"{n_strata}")
            order.append(pos)
        if n_items < 2:
            raise ValueError("the split pilot needs at least two items per draw")
        rq_mat = split_assignment(len(draws), n_items, seed, n_strata)
        rq = np.zeros(len(cell_list), bool)
        for k, pos in enumerate(order):
            rq[np.array(pos)] = rq_mat[k]
        info.update({"assignment": "seeded", "seed": int(seed), "n_strata": int(n_strata),
                     "items_per_draw": int(n_items), "order": order})
    else:
        same = (pR.all(0) == pR.any(0)) & (pF.all(0) == pF.any(0))
        one = pR.all(0) ^ pF.all(0)
        if not (same & one).all():
            bad = int((~(same & one)).sum())
            raise ValueError(f"item-level baseline scores must be complete under both baselines (the tool then assigns "
                             f"the items) or present under exactly one baseline per item, the same for every model (a "
                             f"split pilot already run); {bad} (draw, item) cells are neither")
        rq = pR.all(0)
        info.update({"assignment": "observed", "seed": None, "n_strata": None})
    info.update({"cells": cell_list, "requested_mask": rq, "draw_of": draw_of,
                 "R": A[requested], "F": A[forced], "P": A[intervention], "complete": bool(pR.all() and pF.all())})
    return info


# --------------------------------------------------------------------------- who benefits, split pilot, decision


def _baselines(B, F, baseline, forced):
    """[(role, arm, array)] of the baselines in the report."""
    if F is None:
        return [("baseline", baseline, B)]
    return [("requested", baseline, B), ("forced", forced, F)]


def _slope_entry(X, G, K):
    if K < 2:
        return {"slope": None, "A": None, "V_x": None, "defined": False, "reason": "fewer than two draws"}
    s = cross_slope(X, G)
    ok = bool(np.isfinite(s["slope"]))
    return {**s, "defined": ok, "reason": None if ok else
            f"V_x = {s['V_x']:.4g} is not above zero: the slope is undefined"}


def who_benefits(bases, P, models, cap=None):
    """Capability slopes, top thirds and, with two baselines, the shared third and the opposite orderings."""
    P = np.asarray(P, float)
    K, M = P.shape
    n = third_size(M)
    out = {"n_chosen": n, "M": int(M), "baselines": []}
    for role, arm, B in bases:
        G = P - B
        w, tied = choose(G.mean(0), n)
        e = {"role": role, "arm": arm, "slope_on_accuracy": _slope_entry(B, G, K),
             "top_third": {"weights": w.tolist(), "chosen": [models[j] for j in np.flatnonzero(w > 0)],
                           "n_tied": tied}}
        if cap is not None:
            keep = cap["keep"]
            if len(keep) < 2:
                e["slope_on_capability"] = {"slope": None, "A": None, "V_x": None, "defined": False,
                                            "reason": "fewer than two models with a capability score"}
            else:
                X = np.tile(cap["values"], (K, 1))
                e["slope_on_capability"] = _slope_entry(X, G[:, keep], K)
            e["slope_on_capability"]["models"] = len(keep)
        out["baselines"].append(e)
    if len(bases) == 2:
        w1, w2 = (np.array(b["top_third"]["weights"]) for b in out["baselines"])
        out["shared_top_third"] = shared(w1, w2)
        g1, g2 = (P - bases[0][2]).mean(0), (P - bases[1][2]).mean(0)
        rev = pair_reversals(g1, g2)
        rev["reversed_pairs"] = [[models[a], models[b]] for a, b in rev.pop("reversed_index")]
        out["opposite_orderings"] = rev
    if cap is not None:
        out["capability_missing"] = cap["missing"]
        out["capability_not_in_scores"] = cap["not_in_scores"]
    return out


def split_pilot(it, models, thirds):
    """The split pilot's gains and choice from item_scores' output; thirds: {role: weights} of the score table."""
    R, F, P, rq = it["R"], it["F"], it["P"], it["requested_mask"]
    M = len(models)
    n = third_size(M)
    base = np.where(rq[None, :], R, F)
    gain = 100.0 * (P - base).mean(1)
    per_draw = []
    for d in it["draws"]:
        c = it["draw_of"] == d
        per_draw.append((100.0 * (P[:, c] - base[:, c]).mean(1)).tolist())
    w, tied = choose(gain, n)
    out = {"form": it["form"], "assignment": it["assignment"], "seed": it["seed"], "n_strata": it["n_strata"],
           "draws": it["draws"], "n_items": int(len(rq)),
           "items_per_draw": {d: int((it["draw_of"] == d).sum()) for d in it["draws"]},
           "requested_items_per_draw": {d: int(rq[it["draw_of"] == d].sum()) for d in it["draws"]},
           "n_chosen": n, "weights": w.tolist(), "chosen": [models[j] for j in np.flatnonzero(w > 0)],
           "n_tied": tied, "per_draw_split_gain": per_draw,
           "shared_with": {r: shared(w, wt) for r, wt in thirds.items()},
           "per_model": [], "assignment_cells": [{"draw": d, "item": i, "baseline": "requested" if q else "forced"}
                                                 for (d, i), q in zip(it["cells"], rq)]}
    gr = 100.0 * (P - R).mean(1) if it["complete"] else None
    gf = 100.0 * (P - F).mean(1) if it["complete"] else None
    for j, m in enumerate(models):
        out["per_model"].append({"model": m, "split_gain": float(gain[j]), "weight": float(w[j]),
                                 "gain_requested_items": None if gr is None else float(gr[j]),
                                 "gain_forced_items": None if gf is None else float(gf[j])})
    if it["complete"]:
        out["single_baseline_thirds"] = {}
        for role, g in (("requested", gr), ("forced", gf)):
            ws, ts = choose(g, n)
            out["single_baseline_thirds"][role] = {"weights": ws.tolist(), "n_tied": ts,
                                                   "chosen": [models[j] for j in np.flatnonzero(ws > 0)],
                                                   "shared_with_split": shared(w, ws)}
    return out


def decision_block(bases, P, models, seed=SPLIT_SEED):
    """Under each baseline: the own-gain third, the capability third, how many they share, and the split-half regret."""
    P = np.asarray(P, float)
    K, M = P.shape
    n = third_size(M)
    out = {"n_chosen": n, "M": int(M), "baselines": []}
    for role, arm, B in bases:
        G = P - B
        w_own, t_own = choose(G.mean(0), n)
        e = {"role": role, "arm": arm,
             "own_gain": {"weights": w_own.tolist(), "chosen": [models[j] for j in np.flatnonzero(w_own > 0)],
                          "n_tied": t_own}}
        if K >= 2:
            pred, s = capability_prediction(B, G)
            w_cap, t_cap = choose(pred, n)
            e["capability"] = {"slope": float(s), "slope_defined": bool(np.isfinite(s)), "prediction": pred.tolist(),
                               "weights": w_cap.tolist(), "chosen": [models[j] for j in np.flatnonzero(w_cap > 0)],
                               "n_tied": t_cap}
            e["shared"] = shared(w_own, w_cap)
        else:
            e["capability"], e["shared"] = None, None
        e["split_half_regret"] = split_half_regret(B, G, n, seed)
        out["baselines"].append(e)
    return out


# --------------------------------------------------------------------------- report


def report_arrays(B, P, models=None, draws=None, baseline="baseline", intervention="intervention", targets=TARGETS,
                  F=None, forced=None, capability=None, decision=False, seed=SPLIT_SEED):
    """The report from (K, M) arrays of the two arms in pp; F, the forced baseline's array, adds the two-baseline
    blocks; capability (a table or the output of capability_scores) and decision add theirs."""
    B, P = np.asarray(B, float), np.asarray(P, float)
    if B.ndim != 2 or B.shape != P.shape:
        raise ValueError("the arms need matching (draws, models) arrays")
    if not (np.isfinite(B).all() and np.isfinite(P).all()):
        raise ValueError("scores must be finite")
    if F is not None:
        F = np.asarray(F, float)
        if F.shape != B.shape or not np.isfinite(F).all():
            raise ValueError("the forced baseline needs a finite array of the same (draws, models) shape")
        forced = forced or "forced"
    K, M = B.shape
    if M < 2:
        raise ValueError("the report needs two or more models")
    models = list(models) if models is not None else [f"model_{m + 1}" for m in range(M)]
    draws = list(draws) if draws is not None else [str(k + 1) for k in range(K)]
    G = P - B
    blocks = {"benefit": quantity_block(G, "benefit", targets),
              "accuracy_baseline": quantity_block(B, f"accuracy under {baseline}", targets),
              "accuracy_intervention": quantity_block(P, f"accuracy under {intervention}", targets)}
    if F is not None:
        blocks["benefit_forced"] = quantity_block(P - F, f"benefit over {forced}", targets)
        blocks["accuracy_forced"] = quantity_block(F, f"accuracy under {forced}", targets)
    sd = G.std(axis=0, ddof=1) if K >= 2 else np.full(M, np.nan)
    per_model = [{"model": models[m], "baseline_mean": float(B[:, m].mean()),
                  "intervention_mean": float(P[:, m].mean()), "gain_mean": float(G[:, m].mean()),
                  "gain_sd_across_draws": float(sd[m])} for m in range(M)]
    if F is not None:
        GF = P - F
        sdf = GF.std(axis=0, ddof=1) if K >= 2 else np.full(M, np.nan)
        for m, p in enumerate(per_model):
            p.update({"forced_mean": float(F[:, m].mean()), "gain_forced_mean": float(GF[:, m].mean()),
                      "gain_forced_sd_across_draws": float(sdf[m]),
                      "gain_difference": float(G[:, m].mean() - GF[:, m].mean())})
    warnings = []
    if K < MIN_PILOT_DRAWS:
        warnings.append(f"only {K} draw{'s' if K != 1 else ''}: R from fewer than {MIN_PILOT_DRAWS} draws is "
                        f"imprecise and is often negative; plan from a pilot of at least {MIN_PILOT_DRAWS}")
    res = {"tool": "pilot_report", "version": __version__, "baseline": baseline, "intervention": intervention,
           "M": int(M), "K": int(K), "models": models, "draws": draws, "targets": [float(t) for t in targets],
           **blocks, "unresolved": blocks["benefit"]["unresolved"], "unresolved_reason": blocks["benefit"]["reason"],
           "per_model": per_model, "comparison_arm_check": None, "warnings": warnings}
    if F is not None or capability is not None or decision:
        if F is not None:
            res["forced"] = forced
        cap = None
        if capability is not None:
            cap = capability if isinstance(capability, dict) else capability_scores(capability, models)
            if cap["missing"]:
                warnings.append(f"no capability score for {cap['missing']}: left out of the slope on the capability "
                                "score")
        bases = _baselines(B, F, baseline, forced)
        res["who_benefits"] = who_benefits(bases, P, models, cap)
        if decision:
            res["decision"] = decision_block(bases, P, models, seed)
            if K < MIN_PILOT_DRAWS:
                warnings.append(f"the split-half regret needs at least {MIN_PILOT_DRAWS} draws: not computed")
    return res


def pilot_report(scores, baseline, intervention, tokens=None, token_budget=TOKEN_BUDGET,
                 max_over_budget=MAX_OVER_BUDGET, targets=TARGETS, forced=None, capability=None, items=None,
                 seed=SPLIT_SEED, decision=False):
    """The full report. scores, tokens, capability and items are CSV paths or DataFrames."""
    if items is not None and forced is None:
        raise ValueError("the split pilot (items) needs the forced baseline (forced)")
    if forced is None:
        B, P, models, draws, notes = score_arrays(scores, baseline, intervention)
        F = None
    else:
        if len({baseline, forced, intervention}) < 3:
            raise ValueError("the requested baseline, the forced baseline and the intervention must be three arms")
        X, models, draws, notes = arm_arrays(scores, (baseline, forced, intervention))
        B, F, P = X[baseline], X[forced], X[intervention]
    res = report_arrays(B, P, models, draws, baseline, intervention, targets, F=F, forced=forced,
                        capability=capability, decision=decision, seed=seed)
    res["warnings"] = notes + res["warnings"]

    def name(x):
        return None if x is None else ("<DataFrame>" if isinstance(x, pd.DataFrame) else str(x))

    res["inputs"] = {"scores": name(scores), "tokens": name(tokens), "token_budget": token_budget,
                     "max_over_budget": max_over_budget}
    new = forced is not None or capability is not None or items is not None or decision
    if new:
        res["inputs"].update({"forced": forced, "capability": name(capability), "items": name(items), "seed": seed,
                              "decision": bool(decision)})
    if tokens is not None:
        chk = token_check(tokens, baseline, models, token_budget, max_over_budget)
        res["comparison_arm_check"] = chk
        if chk["not_checked"]:
            res["warnings"].append(f"no {baseline} token counts for {chk['not_checked']}: not checked")
        if forced is not None:
            share = {r["model"]: r["share_over_budget"] for r in chk["models"]}
            for p in res["per_model"]:
                p["w"] = share.get(p["model"])
            ws = [p["w"] for p in res["per_model"] if p["w"] is not None and math.isfinite(p["w"])]
            res["w"] = {"token_budget": token_budget, "models": len(ws),
                        "mean": float(np.mean(ws)) if ws else None,
                        "definition": f"share of {baseline} replies longer than {token_budget} tokens"}
    if items is not None:
        it = item_scores(items, baseline, forced, intervention, models, seed)
        thirds = {b["role"]: b["top_third"]["weights"] for b in res["who_benefits"]["baselines"]}
        res["split_pilot"] = split_pilot(it, models, thirds)
    return res


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def _count(x):
    """A shared count: an integer when no model is chosen in part, else two decimals."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def _chosen(models, weights):
    """Names of the chosen models; a model chosen in part carries its weight."""
    out = []
    for m, w in zip(models, weights):
        if w > 0:
            out.append(m if abs(w - 1) < 1e-12 else f"{m} ({w:.2f})")
    return ", ".join(out)


def _wrap(L, text, indent="  ", sub="    "):
    """Wrapped lines of the 1.1.0 blocks; model names are never broken at their hyphens."""
    L.extend(textwrap.wrap(text, width=110, initial_indent=indent, subsequent_indent=sub, break_on_hyphens=False,
                           break_long_words=False))


def format_report(res) -> str:
    K, M, b, i = res["K"], res["M"], res["baseline"], res["intervention"]
    fd = res.get("forced")
    if fd is None:
        cols = [("benefit", "benefit"), ("accuracy_baseline", f"acc. {b}"), ("accuracy_intervention", f"acc. {i}")]
        head = [f"  arms: {b} (baseline), {i} (intervention); benefit = {i} - {b}, in pp"]
    else:
        cols = [("benefit", "benefit/req"), ("benefit_forced", "benefit/forced"), ("accuracy_baseline", "acc. req"),
                ("accuracy_forced", "acc. forced"), ("accuracy_intervention", "acc. interv.")]
        head = [f"  arms: {b} (requested baseline, req), {fd} (forced baseline, forced), {i} (intervention, interv.)",
                f"  benefit = {i} - baseline, in pp, under each baseline"]
    w0 = 36
    w = max(12, max(len(c[1]) for c in cols) + 2)
    L = [f"Pilot report ({res['tool']} {res['version']})", *head, f"  M = {M} models, K = {K} draws", ""]

    def row(label, key, nd=3, fmt=None):
        cells = []
        for q, _ in cols:
            v = res[q][key] if fmt is None else fmt(res[q])
            cells.append((v if isinstance(v, str) else _f(v, nd)).rjust(w))
        L.append(label.ljust(w0) + "".join(cells))

    L.append("".ljust(w0) + "".join(c[1].rjust(w) for c in cols))
    row("R (reliability across draws)", "R")
    row("  signal S (pp^2)", "signal", 2)
    row("  within-draw variance T (pp^2)", "within", 2)
    row("  prompt noise T - S (pp^2)", "noise", 2)
    row("C observed between distinct draws", "C")
    row("  share of tied comparisons", "tie")
    row("C_ref = 1/2 + arcsin(R)/pi", "C_ref")
    row(f"R of the {K}-draw mean", "R_K")
    row(f"C_ref for two {K}-draw means", "C_ref_K")
    for t in res["targets"]:
        key = f"{t:.2f}"
        row(f"draws for {t:.0%} agreement", None,
            fmt=lambda blk, key=key: "-" if blk["draws_needed"][key] is None else str(blk["draws_needed"][key]))
    row("status", None, fmt=lambda blk: "unresolved" if blk["unresolved"] else "resolved")
    for q, name in cols:
        text = (f"{name}: unresolved, {res[q]['reason']}" if res[q]["unresolved"]
                else f"{name}: {res[q]['note']}" if res[q].get("note") else None)
        if text:
            L.extend(textwrap.wrap(text, width=110, initial_indent="  ", subsequent_indent="    "))
    L.append("")
    if fd is None:
        L.append(f"Per model, mean over the {K} draws (pp)")
        wm = max(5, max(len(p["model"]) for p in res["per_model"])) + 2
        wb, wi = max(10, len(b) + 2), max(10, len(i) + 2)
        L.append("model".ljust(wm) + b.rjust(wb) + i.rjust(wi) + "gain".rjust(9) + "gain SD".rjust(10))
        for p in res["per_model"]:
            L.append(p["model"].ljust(wm) + _f(p["baseline_mean"], 2).rjust(wb)
                     + _f(p["intervention_mean"], 2).rjust(wi) + _f(p["gain_mean"], 2).rjust(9)
                     + _f(p["gain_sd_across_draws"], 2).rjust(10))
    else:
        _per_model_two(L, res)
    chk = res.get("comparison_arm_check")
    if chk is not None:
        L.append("")
        L.append(f"Comparison-arm check: {b} outputs longer than {chk['token_budget']} tokens "
                 f"(flagged above {chk['max_over_budget']:.0%})")
        wm2 = max(5, max(len(r["model"]) for r in chk["models"])) + 2
        L.append("model".ljust(wm2) + "outputs".rjust(9) + "over".rjust(8) + "share".rjust(8) + "median".rjust(8)
                 + "max".rjust(7) + "  verdict")
        for r in chk["models"]:
            L.append(r["model"].ljust(wm2) + f"{r['n_outputs']:,}".rjust(9) + f"{r['n_over_budget']:,}".rjust(8)
                     + f"{r['share_over_budget']:.1%}".rjust(8) + _f(r["median_tokens"], 1).rjust(8)
                     + f"{r['max_tokens']:.0f}".rjust(7) + ("  " + FLAG_MESSAGE if r["flagged"] else "  ok"))
        L.extend(textwrap.wrap(f"flagged: {len(chk['flagged'])} of {len(chk['models'])} models"
                               + (f" ({', '.join(chk['flagged'])})" if chk["flagged"] else ""),
                               width=110, initial_indent="  ", subsequent_indent="    "))
        if chk["not_in_scores"]:
            L.append(f"  in the token table only: {chk['not_in_scores']}")
    if res.get("who_benefits") is not None:
        _who_lines(L, res)
    if res.get("split_pilot") is not None:
        _split_lines(L, res)
    if res.get("decision") is not None:
        _decision_lines(L, res)
    if res["warnings"]:
        L.append("")
        for x in res["warnings"]:
            L.extend(textwrap.wrap(f"Warning: {x}", width=110, subsequent_indent="  "))
    return "\n".join(L)


def _per_model_two(L, res):
    K, b, fd, i = res["K"], res["baseline"], res["forced"], res["intervention"]
    has_w = "w" in res
    L.append(f"Per model, mean over the {K} draws (pp)")
    _wrap(L, f"req = {b}, forced = {fd}, interv. = {i}; gain/req = interv. - req, gain/forced = interv. - forced, "
             "SD its standard deviation across draws; difference = gain/req - gain/forced = forced - req"
          + (f"; w = share of {b} replies longer than {res['w']['token_budget']} tokens" if has_w else ""))
    wm = max(5, max(len(p["model"]) for p in res["per_model"])) + 2
    hdr = ("model".ljust(wm) + "req".rjust(8) + "forced".rjust(8) + "interv.".rjust(9) + "gain/req".rjust(10)
           + "SD".rjust(7) + "gain/forced".rjust(13) + "SD".rjust(7) + "difference".rjust(12))
    L.append(hdr + ("w".rjust(8) if has_w else ""))
    for p in res["per_model"]:
        line = (p["model"].ljust(wm) + _f(p["baseline_mean"], 2).rjust(8) + _f(p["forced_mean"], 2).rjust(8)
                + _f(p["intervention_mean"], 2).rjust(9) + _f(p["gain_mean"], 2).rjust(10)
                + _f(p["gain_sd_across_draws"], 2).rjust(7) + _f(p["gain_forced_mean"], 2).rjust(13)
                + _f(p["gain_forced_sd_across_draws"], 2).rjust(7) + _f(p["gain_difference"], 2).rjust(12))
        if has_w:
            line += ("-" if p["w"] is None else f"{p['w']:.1%}").rjust(8)
        L.append(line)
    if has_w and res["w"]["mean"] is not None:
        L.append(f"  w, mean over the {res['w']['models']} models with token counts: {res['w']['mean']:.1%}")


def _slope_text(e, nd=3):
    if e["slope"] is None:
        return "-"
    return _f(e["slope"], nd) if e["defined"] else "undefined"


def _who_lines(L, res):
    wb, K, models = res["who_benefits"], res["K"], res["models"]
    bases = wb["baselines"]
    L.append("")
    L.append(f"Who benefits (gain = {res['intervention']} - baseline; mean over the {K} draws)")
    w0, wc = 52, 13
    L.append("".ljust(w0) + "".join(e["role"].rjust(wc) for e in bases))

    def row(label, fn):
        L.append(label.ljust(w0) + "".join(fn(e).rjust(wc) for e in bases))

    row("capability slope on own-baseline accuracy (pp/pp)", lambda e: _slope_text(e["slope_on_accuracy"]))
    row("  A, cross-draw moment of gain and accuracy (pp^2)", lambda e: _f(e["slope_on_accuracy"]["A"], 2))
    row("  V_x, cross-draw variance of accuracy (pp^2)", lambda e: _f(e["slope_on_accuracy"]["V_x"], 2))
    if "slope_on_capability" in bases[0]:
        n_cap = bases[0]["slope_on_capability"]["models"]
        row(f"slope on the capability score ({n_cap} models)", lambda e: _slope_text(e["slope_on_capability"], 4))
        row("  V_x, variance of the capability score", lambda e: _f(e["slope_on_capability"]["V_x"], 3))
    for e in bases:
        for key in ("slope_on_accuracy", "slope_on_capability"):
            if key in e and e[key]["reason"]:
                _wrap(L, f"{e['role']}, {key.replace('_', ' ')}: {e[key]['reason']}")
    n = wb["n_chosen"]
    L.append(f"top third by mean gain ({n} of {wb['M']} models)")
    for e in bases:
        t = e["top_third"]
        _wrap(L, f"{e['role']}: {_chosen(models, t['weights'])}"
              + (f" ({t['n_tied']} models tied at the cut share the last places)" if t["n_tied"] else ""))
    if "shared_top_third" in wb:
        L.append(f"  shared by the two top thirds: {_count(wb['shared_top_third'])} of {n}")
        r = wb["opposite_orderings"]
        L.append(f"model pairs the two baselines order oppositely by gain: {r['reversed']} of {r['pairs']} "
                 f"({r['reversed_share']:.1%}); pairs tied under either: {r['ties']}")
        parts = [f"above {x['min_gap_pp']:g} pp: {x['reversed']} of {x['pairs_both_above']}"
                 for x in r["by_threshold"] if x["min_gap_pp"] > 0]
        _wrap(L, "with both gaps " + "; ".join(parts))


def _split_lines(L, res):
    sp, models = res["split_pilot"], res["models"]
    L.append("")
    n_d = len(sp["draws"])
    per = sorted(set(sp["items_per_draw"].values()))
    per_txt = f"{per[0]} items per draw" if len(per) == 1 else f"{min(per)} to {max(per)} items per draw"
    L.append(f"Split pilot (item-level scores: {n_d} draw{'s' if n_d != 1 else ''}, {per_txt})")
    rq = sorted(set(sp["requested_items_per_draw"].values()))
    rq_txt = f"{rq[0]}" if len(rq) == 1 else f"{min(rq)} to {max(rq)}"
    if sp["assignment"] == "seeded":
        strata = f", balanced within {sp['n_strata']} strata" if sp["n_strata"] > 1 else ""
        _wrap(L, f"assignment: half of each draw's items to each baseline by numpy.random.default_rng({sp['seed']})"
                 f"{strata}, the same for every model; {rq_txt} items per draw on the requested baseline")
    else:
        _wrap(L, f"assignment: read from the item table (each item scored under one baseline, the same for every "
                 f"model); {rq_txt} items per draw on the requested baseline")
    complete = sp["per_model"][0]["gain_requested_items"] is not None
    wm = max(5, max(len(m) for m in models)) + 2
    L.append("model".ljust(wm) + ("gain/req".rjust(10) + "gain/forced".rjust(13) if complete else "")
             + "split gain".rjust(12) + "chosen".rjust(8))
    for p in sp["per_model"]:
        L.append(p["model"].ljust(wm)
                 + (_f(p["gain_requested_items"], 2).rjust(10) + _f(p["gain_forced_items"], 2).rjust(13)
                    if complete else "")
                 + _f(p["split_gain"], 2).rjust(12)
                 + ("" if p["weight"] == 0 else "yes" if p["weight"] == 1 else f"{p['weight']:.2f}").rjust(8))
    if complete:
        L.append("  gain/req and gain/forced: the same items, all under one baseline (100 x mean paired gain)")
    n = sp["n_chosen"]
    _wrap(L, f"top third by split gain ({n} of {len(models)}): {_chosen(models, sp['weights'])}"
          + (f" ({sp['n_tied']} models tied at the cut share the last places)" if sp["n_tied"] else ""))
    parts = [f"with the {r} third of the draw-level gains: {_count(v)} of {n}" for r, v in sp["shared_with"].items()]
    _wrap(L, "shared " + "; ".join(parts))
    if "single_baseline_thirds" in sp:
        parts = [f"{r}: {_count(v['shared_with_split'])} of {n}" for r, v in sp["single_baseline_thirds"].items()]
        _wrap(L, "shared with the third the same items give under one baseline: " + "; ".join(parts))


def _decision_lines(L, res):
    dc, models, K = res["decision"], res["models"], res["K"]
    bases = dc["baselines"]
    n = dc["n_chosen"]
    L.append("")
    L.append(f"Decision: {res['intervention']} for the top third ({n} of {dc['M']} models)")
    w0, wc = 52, 16
    L.append("".ljust(w0) + "".join(e["role"].rjust(wc) for e in bases))

    def row(label, fn):
        L.append(label.ljust(w0) + "".join(fn(e).rjust(wc) for e in bases))

    row("own-gain third and capability third share",
        lambda e: "-" if e["shared"] is None else f"{_count(e['shared'])} of {n}")
    if bases[0]["split_half_regret"] is not None:
        s0 = bases[0]["split_half_regret"]
        row("split-half regret, own gain (pp per chosen model)",
            lambda e: _f(e["split_half_regret"]["own_gain"]["mean"], 2))
        row("  smallest to largest over the splits", lambda e: f"{_f(e['split_half_regret']['own_gain']['min'], 2)}"
                                                           f" to {_f(e['split_half_regret']['own_gain']['max'], 2)}")
        row("split-half regret, capability slope", lambda e: _f(e["split_half_regret"]["capability"]["mean"], 2))
        row("  smallest to largest over the splits", lambda e: f"{_f(e['split_half_regret']['capability']['min'], 2)}"
                                                           f" to {_f(e['split_half_regret']['capability']['max'], 2)}")
        h1, h2 = s0["half_sizes"]
        how = (f"every split of the {K} draws into halves of {h1} and {h2}, both directions" if s0["exhaustive"]
               else f"{s0['n_choices'] // 2:,} random splits into halves of {h1} and {h2}, both directions")
        _wrap(L, f"splits: {s0['n_choices']:,} choices ({how}); the regret is the best third's mean gain minus the "
                 "chosen third's, on the half not used to choose")
        und = [f"{e['role']} {e['split_half_regret']['capability_slope_undefined']}" for e in bases
               if e["split_half_regret"]["capability_slope_undefined"]]
        if und:
            _wrap(L, "choices with an undefined capability slope (every model tied): " + ", ".join(und))
    else:
        L.append(f"  split-half regret: needs at least {MIN_PILOT_DRAWS} draws")
    for e in bases:
        _wrap(L, f"{e['role']}, third by own mean gain: {_chosen(models, e['own_gain']['weights'])}")
        if e["capability"] is not None:
            c = e["capability"]
            _wrap(L, f"{e['role']}, third by the capability slope: "
                  + (_chosen(models, c["weights"]) if c["slope_defined"]
                     else "slope undefined, every model tied at the cut"))


def jsonable(o):
    """NaN as null (JSON has no NaN), infinity as the string "inf"; every other value as it is."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, float) and math.isinf(o):
        return "inf" if o > 0 else "-inf"
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description="R, C and the draws a claim about who benefits more needs, from K "
                                             "pilot draws of two arms over M models; with --forced, the claim of "
                                             "who benefits under both direct-answer baselines.")
    ap.add_argument("scores", help="CSV with columns model, draw, arm, score (score in pp)")
    ap.add_argument("--baseline", required=True, help="name of the baseline (comparison) arm; with --forced, the "
                                                      "requested baseline")
    ap.add_argument("--intervention", required=True, help="name of the intervention arm")
    ap.add_argument("--tokens", help="CSV with columns model, arm, n_output_tokens [, count] for the "
                                     "comparison-arm check")
    ap.add_argument("--token-budget", type=int, default=TOKEN_BUDGET,
                    help=f"output tokens a direct answer may take (default {TOKEN_BUDGET})")
    ap.add_argument("--max-over-budget", type=float, default=MAX_OVER_BUDGET,
                    help=f"largest share of baseline outputs over the budget that passes (default {MAX_OVER_BUDGET})")
    ap.add_argument("--forced", help="name of the forced baseline's arm; adds the two-baseline report")
    ap.add_argument("--capability", help="CSV with columns model, capability: a capability score per model for the "
                                         "slope of gain on it")
    ap.add_argument("--items", help="CSV of item-level scores for the split pilot (needs --forced)")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED,
                    help=f"seed of the split pilot's assignment and of random splits (default {SPLIT_SEED})")
    ap.add_argument("--decision", action="store_true", help="add the decision block")
    ap.add_argument("--json", help="write the full report to this JSON file")
    a = ap.parse_args(argv)
    try:
        res = pilot_report(a.scores, a.baseline, a.intervention, a.tokens, a.token_budget, a.max_over_budget,
                           forced=a.forced, capability=a.capability, items=a.items, seed=a.seed, decision=a.decision)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(jsonable(res), fh, indent=1, allow_nan=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
