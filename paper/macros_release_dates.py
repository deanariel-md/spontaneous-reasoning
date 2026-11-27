"""Macros of w against release date, and
intervals for the newest-model reversal shares. Written by scripts/release_dates.py into
results/release_dates/release_dates.json, block "macros"; release dates in manifests/panel_release_dates.json, the
within-family pairs in manifests/panel_release_pairs.json (both written before w was read).

release_dates_results(root) reads the JSON, or returns None until it exists; release_dates_macros(R) returns
{name: value}, after checking every name against the list below.
  wDateRho<T>, wDateRho<T>Lo, wDateRho<T>Hi    Spearman rho of w with the release date over the panel (29 models; 28 on
                                               BIG-Bench Hard), 95% percentile interval from 9,999 draws of whole
                                               families; two decimals (T: Gsm, Math, Bbh)
  wRecentMean<T>Pct, wOlderMean<T>Pct          mean w (%) of the checkpoints whose Hugging Face repository was created
                                               in 2025 or later (the 14 of \\aeRecentM; 13 on BIG-Bench Hard) and of
                                               the earlier ones; one decimal
  wNewerHigherGsm, wNewerHigherMath, wNewerPairs
                                               within-family pairs (newer against older release of similar size, the
                                               rule of manifests/panel_release_pairs.json): pairs where the newer has
                                               the higher w, and the number of pairs (the same on both benchmarks)
  aeRecent<T>ReversedPct{Lo,Hi,FamLo,FamHi}    95% percentile intervals of \\aeRecent<T>ReversedPct from 9,999 draws of
                                               models and of whole families (src/decision_directions.
                                               reversal_uncertainty); whole percent, as the point value
w: the share of requested replies longer than 16 tokens, discovery draws (src/baseline_mixture.RULE_TEXT).
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "release_dates", "release_dates.json")
TAGS = ("Gsm", "Math", "Bbh")
NAMES = ([f"wDateRho{t}{s}" for t in TAGS for s in ("", "Lo", "Hi")]
         + [f"w{g}Mean{t}Pct" for g in ("Recent", "Older") for t in TAGS]
         + ["wNewerHigherGsm", "wNewerHigherMath", "wNewerPairs"]
         + [f"aeRecent{t}ReversedPct{s}" for t in TAGS for s in ("Lo", "Hi", "FamLo", "FamHi")])


def release_dates_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def release_dates_macros(R) -> dict:
    if not R:
        return {}
    m = R["macros"]
    if sorted(m) != sorted(NAMES):
        raise ValueError(f"release_dates.json's macros differ from the list: missing {sorted(set(NAMES) - set(m))}, "
                         f"extra {sorted(set(m) - set(NAMES))}")
    out = {}
    for k in NAMES:
        if not re.fullmatch(r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not letters only")
        out[k] = str(m[k])
    return out


if __name__ == "__main__":
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    for k, v in release_dates_macros(release_dates_results(root)).items():
        sys.stdout.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
