"""The arm-switch tests on interventions other than chain of thought
(results/generality/methods.md): pair reversals between two protocols with a model-resampling interval, and the
wrong-protocol decision cost in both directions in the one-draw design.

Arrays are draws by models, pp of accuracy; a benefit is intervention minus baseline.
"""
from __future__ import annotations

import numpy as np

from src import decision as dc
from src import decision_directions as dd
from src import robustness as rb


def evaluation_rows(disc_units, val_units) -> list[int]:
    """Rows of the distinct validation configurations that no discovery draw uses (run_decision.evaluation_rows)."""
    rows = dc.distinct_rows(val_units)
    d = set(map(str, disc_units))
    return [r for r in rows if str(val_units[r]) not in d]


def one_draw_choices(Gd, disc_units, n: int) -> np.ndarray:
    """(pilots, M) weights: the top n by own benefit on one discovery draw, each distinct configuration once (own_1)."""
    Gd = np.asarray(Gd, float)
    return np.array([dc.choose(Gd[r], n)[0] for r in dc.distinct_rows(disc_units)])


def decision_cost(Gd: dict, Gv: dict, disc_units, val_units) -> dict:
    """Both directions between two protocols a and b (the two keys of Gd and Gv, in order). Gd: discovery benefits
    (K_d, M); Gv: validation benefits (K_v, M). {'a_to_b': ..., 'b_to_a': ...}: the extra regret of choosing under
    the first protocol and valuing under the second over choosing under the second (positive: a cost), t interval over
    the fresh validation configurations, with the two plug-in regrets."""
    a, b = list(Gd)
    M = np.asarray(Gd[a]).shape[1]
    n = dc.third_size(M)
    rows = evaluation_rows(disc_units, val_units)
    W = {p: one_draw_choices(Gd[p], disc_units, n) for p in (a, b)}
    out = {"n_chosen": n, "M": int(M), "n_pilots": int(W[a].shape[0]), "n_configurations": len(rows),
           "share_same_choice": float(np.mean(np.all(np.abs(W[a] - W[b]) <= 1e-9, axis=1)))}
    if len(rows) < 2:
        out["note"] = "fewer than two fresh validation configurations"
        return out
    for src, dst in ((a, b), (b, a)):
        G = np.asarray(Gv[dst], float)[rows]
        d = dc.paired_difference(W[src], W[dst], G, n)
        d["regret_mismatched"] = float(dc.regret_plugin(W[src], G, n).mean())
        d["regret_matched"] = float(dc.regret_plugin(W[dst], G, n).mean())
        out[f"{src}_to_{dst}"] = d
    return out


def reversals(g1, g2, families, n_boot: int, rng) -> dict:
    """Pair reversals between two benefit vectors (model means): the counts of src/robustness.pair_reversals (the
    \\ae... counting) and the model and family resampling of src/decision_directions.reversal_uncertainty."""
    out = rb.pair_reversals(g1, g2)
    u = dd.reversal_uncertainty(g1, g2, families, n_boot, rng)
    if u["reversed"] != out["reversed"] or u["pairs"] != out["pairs"]:
        raise RuntimeError("the two reversal counts disagree")
    out.update({k: u[k] for k in ("M", "model_bootstrap", "family_bootstrap", "leave_one_family_out",
                                  "leave_one_family_out_range", "within_family", "between_family")})
    return out
