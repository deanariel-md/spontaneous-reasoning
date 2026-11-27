"""Transfer regret.

What does it cost to choose whom to give a strategy from a who-benefits rule fitted where the chooser is not: on other
model families, or on another task? Choices are top thirds (src/decision.choose; ties share the places), regret is in
pp of gain per chosen model against the best third on the evaluation configurations (src/decision).

Transported capability rules are least-squares fits of model mean gains on the rule's columns (src/robustness
.design_matrix: linear in baseline accuracy; quadratic in baseline plus the external index; linear in the index), with
the predictors centred and scaled on the training models, then applied to the target models' columns.
"""
from __future__ import annotations

import numpy as np

from . import decision as dc, robustness as rb

TRANSPORT_RULES = ("baseline_linear", "baseline_quadratic_index", "index_linear")


def fit_apply(b_tr, g_tr, e_tr, b_new, e_new, rule) -> np.ndarray:
    """Fit g_tr on the rule's columns of (b_tr, e_tr), scaled on the training models, and predict at (b_new, e_new).
    A rule with more columns than training models predicts nan (reported, never filled)."""
    b_tr, g_tr, e_tr = (np.asarray(v, float) for v in (b_tr, g_tr, e_tr))
    b_new, e_new = np.asarray(b_new, float), np.asarray(e_new, float)
    mb, sb = b_tr.mean(), b_tr.std() or 1.0
    me, se = e_tr.mean(), e_tr.std() or 1.0
    X = rb.design_matrix((b_tr - mb) / sb, (e_tr - me) / se, rule)
    if X.shape[1] > len(g_tr):
        return np.full(len(b_new), np.nan)
    coef, *_ = np.linalg.lstsq(X, g_tr, rcond=None)
    return rb.design_matrix((b_new - mb) / sb, (e_new - me) / se, rule) @ coef


def third_weights(pred, n):
    """(P, M) top-n weights for each row of predictions; nan rows are refused."""
    return np.array([dc.choose(p, n)[0] for p in np.atleast_2d(pred)])


def grouped_value(Wg: dict, G, groups: dict) -> np.ndarray:
    """Per evaluation configuration, the mean gain per chosen model over all groups: sum over groups of the
    pilot-averaged weights times the group's gains, over the total number chosen. Wg {group: (P, m_g)} weights."""
    G = np.asarray(G, float)
    tot = np.zeros(G.shape[0])
    n_all = 0.0
    for g, ix in groups.items():
        w = np.atleast_2d(Wg[g]).mean(0)
        tot += G[:, ix] @ w
        n_all += w.sum()
    return tot / n_all


def grouped_regret(Wg: dict, G, groups: dict, ns: dict) -> dict:
    """Plug-in and cross-fitted regret per group and pooled (weighted by the number chosen in each group)."""
    per, plug, cross = {}, 0.0, 0.0
    N = sum(ns.values())
    for g, ix in groups.items():
        Gg = np.asarray(G, float)[:, ix]
        rp = float(dc.regret_plugin(Wg[g], Gg, ns[g]).mean())
        rc = float(np.mean(dc.regret_crossfit(Wg[g], Gg, ns[g])))
        per[g] = {"regret_plugin": rp, "regret_crossfit": rc, "n_chosen": ns[g], "m": len(ix)}
        plug += ns[g] * rp
        cross += ns[g] * rc
    return {"regret_plugin": plug / N, "regret_crossfit": cross / N, "per_group": per}


def paired_grouped(Wa: dict, Wb: dict, G, groups: dict) -> dict:
    """Regret of a minus regret of b, pooled over groups (linear in the draws), with a t interval over the evaluation
    configurations; positive: a costs more."""
    return dc.t_interval(grouped_value(Wb, G, groups) - grouped_value(Wa, G, groups))
