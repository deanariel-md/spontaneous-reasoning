"""Macros of the size ladder's orderings (written by
scripts/ladder_orderings.py into results/ladder/analysis/ladder_orderings.json, block "macros"), prefixes lad and comb.

ladder_orderings_results(root) reads the JSON, or returns None until it exists; ladder_orderings_macros(R) returns
{name: value}. The values are printed by scripts/ladder_orderings.py from exact ratios of the scored rows' counts
(scripts/ladder_analysis.fnum, half-even on the exact value; a value that rounds to zero without its sign; minus as $-$;
thousands as {,}), so this module passes them through, after checking every name (letters only, prefix lad or comb)
and that the block covers the request's families of names. scripts/verify_ladder_analysis.py recomputes every one of
them from the scored rows with its own code and compares it with paper/numbers.tex.

Suffixes: none for content scoring and outputs cut at the panel's caps (the primary reading); Strict, Uncut,
StrictUncut for the others. <T>: Gsm, Math, Bbh, Hq; Pooled for GSM8K, MATH-500 and BIG-Bench Hard together.
  ladRev<T>, ladPairs<T>, ladRevPct<T>, ladTies<T>       the six ladder models: pairs reversed between CoT - requested
                                                         and CoT - forced (discovery-draw means), pairs, percent, ties
  ladBigRev<T>, ladBigPairs<T>, ladBigTies<T>             the four of 70B and above (count, no share)
  combRev<T>, combPairs<T>, combRevPct<T>, combRevShare<T>, combTies<T>, combM<T>, combRevBootLo<T>, combRevBootHi<T>
                                                         the combined panel (29 + 6; 34 on BIG-Bench Hard) with its
                                                         model bootstrap 95% interval of the share
  combPanel...                                           the same for the 29 panel models (content and strict, cut)
  ladW<Model><T>, ladWDir<Family><T>, ladWPanelMin<T>, ladWPanelMax<T>, ladWPanelMedian<T>
                                                         w on the discovery draws, the family's direction over size,
                                                         the panel's range
  ladShift<Model><T>, ladShiftLong<Model><T>, ladShiftShort<Model><T>, ladShiftLongShare<Model><T>
                                                         requested - forced (pp, discovery draws) and its parts from
                                                         requested outputs longer than 16 tokens and from the others
  ladShiftPanelMin<T>, ladShiftLongPanelMax<T>, ...      the panel's range of each
  ladQwenReqRows<T>, ladQwenReqUnread<T>, ladQwenReqUnreadPct<T>, ladQwenReqBare<T>, ladQwenReqBarePct<T>,
  ladQwenReqBareTokens<T>, ladQwenReqUnreadContentPct<T> Qwen3.5-122B-A10B's zero-shot requested replies the strict
                                                         rule cannot read, and those with no box at all
Models: MinistralEight, MinistralFourteen, MistralLarge, LlamaSeventy, QwenOneTwentyTwo, DeepSeek; the panel rungs
QwenNine, QwenThirtyFive, LlamaEight. Families: Mistral, Qwen, Llama.
"""
from __future__ import annotations

import json
import os

PREFIXES = ("lad", "comb")
REQUIRED = ("ladRevGsm", "ladPairsGsm", "ladRevPctGsm", "ladRevPooled", "ladPairsPooled", "ladRevPctPooled",
            "combRevGsm", "combRevBootLoPooled", "combPanelRevPooled", "ladBigRevGsm", "ladWLlamaSeventyGsm",
            "ladWPanelMinGsm", "ladShiftQwenOneTwentyTwoGsm", "ladShiftLongQwenOneTwentyTwoGsm",
            "ladShiftShortQwenOneTwentyTwoGsm", "ladQwenReqUnreadGsm")


def ladder_orderings_results(root):
    p = os.path.join(root, "results", "ladder", "analysis", "ladder_orderings.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ladder_orderings_macros(R) -> dict:
    m = dict(R.get("macros") or {})
    bad = [k for k in m if not (k.isalpha() and k.startswith(PREFIXES))]
    if bad:
        raise ValueError(f"ladder_orderings: macro names not letters only with prefix lad or comb: {bad[:5]}")
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise ValueError(f"ladder_orderings: macros missing: {missing}")
    for k, v in m.items():
        if not isinstance(v, str) or not v:
            raise ValueError(f"ladder_orderings: macro {k} has no text value")
    return m
