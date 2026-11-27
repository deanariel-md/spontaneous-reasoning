"""Macros of Fig. 2c's reversal counts on validation draws whose templates no discovery draw used, prefix bmixVal. Read from results/bmix_val/bmix_val.json
(scripts/run_bmix_val.py). Every value is read from that file; nothing is typed here.

bmix_val_results(root) reads the file, or returns None until it exists; bmix_val_macros(R) returns {name: value}, after
checking every name (letters only, prefix bmixVal).

Fresh draws: the validation draws whose configuration (template) no discovery draw used; a template drawn twice among
them counts twice. Gains are model means over the fresh draws; w is the discovery draws' w (the published one). <T> is
Gsm, Math or Bbh. Strict scoring; the same names with a Content suffix are content scoring.
  bmixVal<T>Draws, Templates        fresh validation draws, and the distinct templates among them
  bmixVal<T>OverlapDraws            validation draws on a template some discovery draw used (24 - Draws)
  bmixVal<T>DiscTemplates, ValTemplates   distinct templates among the 24 discovery and the 24 validation draws
  bmixVal<T>Pairs                   model pairs (every pair is in the denominator)
  bmixVal<T>RevObs, RevObsPct       pairs reversed between the gain over the requested baseline and the gain over the
                                    forced baseline, on the fresh draws; share of all pairs, %, no decimal
  bmixVal<T>RevCorr, RevCorrPct     the same against the corrected gain (1 - w) x gain over the forced baseline
  bmixVal<T>DiffObs, DiffCorr       fresh count minus the published discovery count (bmix<T>RevObs, bmix<T>RevCorr),
                                    pairs, $-$ for a minus
  bmixVal<T>TiesObs, TiesCorr       pairs tied (|gap| <= 1e-9 pp) under either gain, on the fresh draws
  bmixVal<T>AllRevObs, AllRevObsPct, AllRevCorr, AllRevCorrPct   the same counts on all 24 validation draws (reference)
  bmixValMaxChange                  the largest |DiffObs| or |DiffCorr| over the three benchmarks and both scorings
"""
from __future__ import annotations

import json
import math
import os
import re

PREFIX = "bmixVal"
PATH = os.path.join("results", "bmix_val", "bmix_val.json")
TASKS = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"))
SCORES = (("strict", ""), ("content", "Content"))
COMP = (("observed_vs_forced", "Obs"), ("observed_vs_corrected", "Corr"))


def bmix_val_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pct0(share) -> str:
    """A share in [0, 1] as a whole percent (as macros_added.num(100 * share, 0))."""
    v = 100 * float(share)
    if not math.isfinite(v):
        raise ValueError("a share is not finite")
    if round(v) == 0:
        v = 0.0
    return f"{v:.0f}"


def signed_int(n) -> str:
    return str(int(n)).replace("-", "$-$")


def bmix_val_macros(R) -> dict:
    if not R:
        return {}
    out = {}

    def put(k, v):
        if k in out:
            raise ValueError(f"macro {k} set twice")
        out[k] = v

    changes = []
    for task, T in TASKS:
        X = R["tasks"][task]
        D = X["draws"]
        s = f"{PREFIX}{T}"
        put(s + "Draws", str(D["draws"]))
        put(s + "Templates", str(D["templates"]))
        put(s + "OverlapDraws", str(D["overlap_draws"]))
        put(s + "DiscTemplates", str(D["discovery_templates"]))
        put(s + "ValTemplates", str(D["validation_templates"]))
        for sc, S in SCORES:
            B = X["scores"][sc]
            f, a, dif = B["fresh_validation"], B["all_validation"], B["fresh_minus_discovery"]
            if not B["reproduction"]["reproduced"]:
                raise ValueError(f"{task} {sc}: the discovery counts were not reproduced")
            put(s + "Pairs" + S, str(f["observed_vs_forced"]["pairs"]))
            for k, tag in COMP:
                put(s + "Rev" + tag + S, str(f[k]["reversed"]))
                put(s + "Rev" + tag + "Pct" + S, pct0(f[k]["reversed_share"]))
                put(s + "Diff" + tag + S, signed_int(dif[k]["reversed"]))
                put(s + "Ties" + tag + S, str(f[k]["ties"]))
                put(s + "AllRev" + tag + S, str(a[k]["reversed"]))
                put(s + "AllRev" + tag + "Pct" + S, pct0(a[k]["reversed_share"]))
                changes.append(abs(int(dif[k]["reversed"])))
    put(PREFIX + "MaxChange", str(max(changes)))
    for k in out:
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for k, v in sorted(bmix_val_macros(bmix_val_results(root)).items()):
        print(f"\\newcommand{{\\{k}}}{{{v}}}")
