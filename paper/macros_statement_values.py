"""Macros written
by scripts/statement_values.py into results/statement_values/statement_values.json, block "macros". Nothing is computed here.

statement_values_results(root) reads the JSON, or returns None until it exists; statement_values(R) returns {name: value}, after
checking every name (letters only, one of the prefixes below).
  hq...Content          item 1, HotpotQA on content exact match (hotpot-content reader of the run), the strict macros'
                        meaning: hqReqMeanGainContent, hqFdMeanGainContent (mean CoT gain, discovery draws, pp, 1 dp);
                        hqReqRbenContent, hqFdRbenContent (R of benefit, discovery draws, 2 dp); hqReqTrPredCContent,
                        hqFdTrPredCContent (one-draw C predicted from the first four discovery draws, refitted on
                        content scores, exploratory; 3 dp); hqReqTrObsCContent, hqFdTrObsCContent (observed one-draw C,
                        validation draws; 3 dp); hqTrSwitchReversedContent, hqTrNewDrawReversedContent (share of model
                        pairs reversed by the arm switch and by a new draw, validation draws; 3 dp)
  item 4                no macro: \\dec<T>ReqBestThird is valued on the validation configurations, items, panel and
                        third of \\ddir<T>FdToReq (the JSON's best_third block says so); a ddir<T>BestThirdReq is added
                        only if that check fails, which the script reports
  ladBigKnown...        item 6, Llama-3.1-70B, Qwen3.5-122B-A10B and Mistral Large 3: ladBigKnownRev<T>, Pairs<T>,
                        Ties<T>, T in Gsm, Math, Bbh, Hq, Pooled (first three), suffix as the lad macros ("" content
                        cut, Strict, Uncut, StrictUncut). The group of 70B and above without DeepSeek V4.1-Flash-1017,
                        built while its own size was not recorded; it is now in the group on its
                        published size (552B total), kept in the files
  aeRecent...           panel models whose repository was created in 2025 or later: aeRecent<T>Reversed,
                        Pairs, Ties, ReversedPct (0 dp), M (T in Gsm, Math, Bbh strict; Hq content); aeRecentM, the
                        models of the four benchmarks together
  genThiGsmArmQwen      item 8, Qwen checkpoints among the genThiGsmArmM
"""
from __future__ import annotations

import json
import os
import re

PATH = os.path.join("results", "statement_values", "statement_values.json")
PREFIXES = ("hqReq", "hqFd", "hqTr", "ladBigKnown", "aeRecent", "genThiGsmArmQwen", "ddir")


def statement_values_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def statement_values(R) -> dict:
    if not R:
        return {}
    out = {}
    for k, v in R["macros"].items():
        if not re.fullmatch(r"[A-Za-z]+", k) or not k.startswith(PREFIXES):
            raise ValueError(f"macro name {k!r}: letters only, prefix one of {PREFIXES}")
        if k.startswith(("hqReq", "hqFd", "hqTr")) and not k.endswith("Content"):
            raise ValueError(f"macro name {k!r}: the HotpotQA macros of this module end in Content")
        if k.startswith("ddir") and not k.endswith("BestThirdReq"):
            raise ValueError(f"macro name {k!r}: the only ddir macro of this module is ddir<T>BestThirdReq")
        out[k] = str(v)
    return out
