"""Fig. 2c's reversal counts on validation draws whose templates no discovery draw used.

Fig. 2c counts model pairs ordered one way by the gain over the requested baseline and the other way by the gain over
the forced baseline, then by the corrected gain (1 - w) x gain over the forced baseline (src/baseline_mixture.py). The
published counts use the 24 discovery draws for every quantity. Here the gains come from the validation draws whose
configuration (template) appears in no discovery draw, and w stays the discovery draws' w. The counting itself is
src/baseline_mixture.mechanism, unchanged; this module only selects draws and takes model means.

Pure functions; scripts/run_bmix_val.py loads the data. Gains in pp; w as a share in [0, 1].
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src import baseline_mixture as bm

ARMS = ("requested", "forced", "cot")


def fresh_draws(discovery_ids, validation_ids) -> dict:
    """The validation draws whose configuration no discovery draw used. A configuration drawn twice among them gives
    two draws (as the discovery means count repeats). Returns the draw positions in validation order and the counts."""
    disc = [str(u) for u in discovery_ids]
    val = [str(u) for u in validation_ids]
    used = set(disc)
    idx = [k for k, u in enumerate(val) if u not in used]
    fresh = [val[k] for k in idx]
    cnt = Counter(fresh)
    return {"index": idx, "draws": len(idx), "templates": len(cnt),
            "template_draws": {u: int(cnt[u]) for u in sorted(cnt)},
            "repeated_templates": sorted(u for u in cnt if cnt[u] > 1),
            "overlap_draws": len(val) - len(idx), "overlap_templates": len(set(val) & used),
            "discovery_draws": len(disc), "discovery_templates": len(used),
            "validation_draws": len(val), "validation_templates": len(set(val))}


def gains(arrays: dict, idx=None):
    """Model means (M,) over the selected draws (rows of the (K, M) arrays; all rows when idx is None) of chain of
    thought minus requested and chain of thought minus forced accuracy, pp. A missing cell gives nan for that model
    (reported, not imputed)."""
    c, r, f = (np.asarray(arrays[a], float) for a in ("cot", "requested", "forced"))
    if c.shape != r.shape or c.shape != f.shape or c.ndim != 2:
        raise ValueError("the three arms need (K, M) arrays of one shape")
    if idx is not None:
        idx = list(idx)
        if not idx:
            raise ValueError("no draw selected")
        c, r, f = c[idx], r[idx], f[idx]
    return (c - r).mean(0), (c - f).mean(0)


def recount(arrays: dict, w, idx=None) -> dict:
    """Fig. 2c's counts on the selected draws, with the given w: bm.mechanism(gain over requested, gain over forced, w),
    plus the model means it used."""
    g_req, g_fd = gains(arrays, idx)
    w = np.asarray(w, float)
    if w.shape != g_req.shape:
        raise ValueError(f"w has {w.shape} values for {g_req.shape} models")
    out = bm.mechanism(g_req, g_fd, w)
    out["gain_vs_requested"] = g_req.tolist()
    out["gain_vs_forced"] = g_fd.tolist()
    out["corrected_gain"] = bm.corrected_gain(g_fd, w).tolist()
    out["models_with_nan"] = int((~np.isfinite(g_req) | ~np.isfinite(g_fd)).sum())
    return out


def difference(val: dict, disc: dict) -> dict:
    """Validation minus discovery, in pairs and in shares, for both comparisons; the pair totals must agree."""
    out = {}
    for k in ("observed_vs_forced", "observed_vs_corrected"):
        a, b = val[k], disc[k]
        if a["pairs"] != b["pairs"]:
            raise ValueError(f"{k}: {a['pairs']} pairs on validation, {b['pairs']} on discovery")
        out[k] = {"reversed": int(a["reversed"] - b["reversed"]),
                  "reversed_share": float(a["reversed_share"] - b["reversed_share"])}
    return out
