"""Macros used by paper/make_numbers.py.

hotpot_mech_fresh_macros: results/hotpot/mechanism_forecast_fresh.json (scripts/run_hotpot_mech_fresh.py), prefix
hqMechFresh, written only after run_hotpot.py's full run. Empty until then.
"""
from __future__ import annotations

import json
import os

from macros_added import _check, _set, _yn, num


def hotpot_mech_fresh_results(root):
    p = os.path.join(root, "results", "hotpot", "mechanism_forecast_fresh.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def hotpot_mech_fresh_macros(R):
    """hqMechFreshDraws (validation draws on configurations no discovery draw used), hqMechFreshReuse (draws reusing
    one), hqMechFreshConfigs (distinct new configurations); for each set, <Draws|Distinct>: PredRev, ObsRev, Pairs,
    ObsShare (3 dp), ShareHolds, PairsHolds, GainHolds (yes, no or --)."""
    m = {}
    if not R:
        return m
    _set(m, "hqMechFreshDraws", str(R["fresh_draws"]))
    _set(m, "hqMechFreshReuse", str(R["reusing_discovery"]))
    _set(m, "hqMechFreshConfigs", str(R["fresh_distinct"]))
    for key, tag in (("fresh_draws", "Draws"), ("fresh_distinct", "Distinct")):
        E = R["sets"].get(key) or {}
        if "statements" not in E:
            continue
        s = "hqMechFresh" + tag
        _set(m, s + "Pairs", str(E["pairs"]))
        _set(m, s + "PredRev", str(E["predicted_reversed"]))
        _set(m, s + "ObsRev", str(E["observed_reversed"]))
        _set(m, s + "ObsShare", num(E["statements"]["share"]["observed"], 3))
        for k, t in (("share", "Share"), ("pairs", "Pairs"), ("gain", "Gain")):
            _set(m, s + t + "Holds", _yn(E["statements"][k].get("holds")))
    return _check(m)
