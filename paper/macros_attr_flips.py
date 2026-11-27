"""Macros of which part of the baseline shift makes the reversals, prefix attr. Read from results/attr_flips/attr_flips.json
(scripts/run_attr_flips.py); every value is read from that file, nothing is typed here.

attr_flips_results(root) reads the file, or returns None until it exists; attr_flips_macros(R) returns {name: value},
after checking every name (letters only, prefix attr).

Pairs of models reversed between the requested-baseline gain (C - R) and a forced-baseline gain, model means over the
24 discovery draws. D = R - F splits exactly into a long part (rows whose requested reply is longer than the cut) and a
short part (the other rows). Strict scoring; the same names with Content at the end are content scoring.
<B> is Gsm, Math, Bbh (cut at the forced cap, 16 tokens), HqSixtyFour (HotpotQA cut at 64 tokens, its forced cap) or
HqSixteen (HotpotQA cut at 16 tokens). The observed count does not depend on the cut, so the two HotpotQA names carry
the same value.
  attr<B>ObsRev            observed: reversed between C - R and C - F (the published count)
  attr<B>LongRev           long part alone: reversed between C - R and C - R + long part
  attr<B>ShortRev          short part alone: reversed between C - R and C - R + short part
  attr<B>BothRev           reversed by each part alone (every such pair is also an observed one)
  attr<B>{Obs,Long,Short,Both}RevPct   the same as a % of all pairs, no decimal
  attr<B>LongRevShared     observed pairs that the long part alone also reverses
  attr<B>ShortRevShared    observed pairs that the short part alone also reverses
  attr<B>NeitherRev        observed pairs that neither part alone reverses
  attr<B>Pairs             model pairs (tied pairs included)
  attr<B>Models            models in the panel (strict only; the same under content)
  attr<B>Short{Median,Min,Max}   the short part over models, pp, one decimal, $-$ for a minus (a value that rounds
                           to zero keeps its sign, as make_numbers.f)
"""
from __future__ import annotations

import json
import os
import re

PREFIX = "attr"
PATH = os.path.join("results", "attr_flips", "attr_flips.json")
BENCH = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"), ("hotpot_cap64", "HqSixtyFour"),
         ("hotpot_cap16", "HqSixteen"))
PARTS = (("observed", "Obs"), ("long", "Long"), ("short", "Short"), ("both", "Both"))


def attr_flips_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def f1(x, nd=1):
    """As make_numbers.f: a hyphen prints $-$ (a value that rounds to zero keeps its sign)."""
    return f"{x:.{nd}f}".replace("-", "$-$")


def _num(x, what):
    if x is None:
        raise ValueError(f"{what}: no finite value in the results file")
    return float(x)


def attr_flips_macros(R) -> dict:
    if not R:
        return {}
    out = {}

    def put(k, v):
        if k in out:
            raise ValueError(f"macro {k} set twice")
        out[k] = v

    for unit, B in BENCH:
        T = R["tasks"].get(unit)
        if T is None:
            continue
        put(f"{PREFIX}{B}Models", str(int(T["M"])))
        for score, S in (("strict", ""), ("content", "Content")):
            blk = T["scores"][score]
            c = blk["counts"]
            k = blk["checks"]["observed_equals_published_file"]
            if not k["equal"]:
                raise ValueError(f"{unit} {score}: the observed count differs from the published one")
            n = int(c["pairs"])
            put(f"{PREFIX}{B}Pairs{S}", str(n))
            for key, P in PARTS:
                put(f"{PREFIX}{B}{P}Rev{S}", str(int(c[key])))
                put(f"{PREFIX}{B}{P}RevPct{S}", f"{100 * int(c[key]) / n:.0f}")
            put(f"{PREFIX}{B}LongRevShared{S}", str(int(c["observed_and_long"])))
            put(f"{PREFIX}{B}ShortRevShared{S}", str(int(c["observed_and_short"])))
            put(f"{PREFIX}{B}NeitherRev{S}", str(int(c["observed_neither"])))
            sp = blk["short_part"]
            put(f"{PREFIX}{B}ShortMedian{S}", f1(_num(sp["median"], f"{unit} {score} short median")))
            put(f"{PREFIX}{B}ShortMin{S}", f1(_num(sp["min"], f"{unit} {score} short min")))
            put(f"{PREFIX}{B}ShortMax{S}", f1(_num(sp["max"], f"{unit} {score} short max")))
    for k in out:
        if not re.fullmatch(PREFIX + r"[A-Z][A-Za-z]*", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for k, v in sorted(attr_flips_macros(attr_flips_results(root)).items()):
        print(f"\\newcommand{{\\{k}}}{{{v}}}")
