"""Macros of the reading audit of requested answers,
prefix lab, written by scripts/reading_audit.py analyse into results/reading_audit/reading_audit.json, block "macros".
The judge is an LLM run through a message batches API at the model's default sampling (no temperature, top_p or
top_k sent), thinking disabled, effort medium, max_tokens 16. The file exists only once the judge's labels exist
(analyse refuses before), so reading_audit_results returns None until then. The human-check macros appear once a
rater's sheet holds labels, the self-agreement macros once the repeat set is judged.

reading_audit_results(root) reads the JSON, or returns None; reading_audit_macros(R) returns {name: value} after
checking every name (letters only, prefix lab). <T> is Gsm, Math, Bbh or Hq; <S> is empty (strict scoring) or Content.
Shares and accuracies in whole percent with Lo/Hi the 95% interval (bootstrap within strata); differences and
residuals in pp to one decimal; correlations, R^2 and kappa to two decimals.
  labJudge, labJudgeId, labJudgeMaxTokens, labJudgeEffort, labSeed, labCostUsd (billed, USD, two decimals)
  labN, labNLabelled, labNUnread (judged, not read as one letter), labNNotJudged, labNPanel, labNQwen    counts
  labN<T>, labModels<T>                         labelled replies and models per benchmark
  labRLong<T>Pct, labRShort<T>Pct (Lo, Hi)      share labelled R among replies over 16 / at most 16 tokens (item 1)
  labRLongHqSixtyFourPct, labRShortHqSixtyFourPct (Lo, Hi)   the same for HotpotQA at 64 tokens
  labPrec<T>Pct, labRecall<T>Pct (Lo, Hi)       precision and recall of the 16-token cutoff as a marker of R (item 2);
                                                HotpotQA also at 64 (labPrecHqSixtyFourPct, labRecallHqSixtyFourPct)
  labRAll<T>Pct (Lo, Hi)                        share labelled R among all requested replies
  labCorrW<T> (Lo, Hi), labSpearW<T>            Pearson and Spearman correlation across models of the per-model R
                                                share with w (at the forced cap: 16, HotpotQA 64) (item 3)
  labRsqR<T>, labRsqW<T> (and ...Content)       R^2 across models of requested - forced accuracy (strict; content) on
                                                the R share and on w (item 4); labRsqDiff<T>Lo/Hi, the interval of the
                                                difference (R share minus w)
  labQwenRLong<T>Pct, labQwenRLongPct (Lo, Hi), labQwenNLong<T>, labQwenNLong
                                                Qwen4-Max: share of its long replies labelled R, GSM8K, MATH-500 and
                                                pooled (item 5)
  labAcc{Req,Cot,Forced}R<T><S>Pct, labAcc{Req,Cot,Forced}NotR<T><S>Pct
                                                the mixture assumption on the same items: requested, CoT and forced
                                                accuracy (design-weighted) on the replies labelled R and not R
  labGapReqCotR<T><S> (Lo, Hi), labGapReqForcedNotR<T><S> (Lo, Hi)
                                                requested - CoT on R, requested - forced on not R (pp)
  labMixRmsR<T><S>, labMixRmsW<T><S>, labMixRmsDiff<T><S>Lo/Hi, labMixMeanResR<T><S>, labMixMeanResW<T><S>,
  labMixCorrR<T><S>, labMixCorrW<T><S>          across models: requested - forced against r x (CoT - forced) and
                                                w x (CoT - forced): RMS and mean residual (pp), the interval of
                                                RMS r - RMS w, Pearson of observed with predicted
  labHumanN, labUnsureRaterOne, labUnsureRaterTwo      human check: rows per sheet, rows marked unsure
  labN<P>, labAgree<P>Pct, labAgree<P>UnwPct, labKappa<P>, labKappa<P>Unw, labPrecR<P>Pct, labRecallR<P>Pct,
  labKappaR<P>                                  <P> is JudgeRaterOne, JudgeRaterTwo or RaterOneRaterTwo: rows with both
                                                labels, agreement and Cohen's kappa (design-weighted; Unw unweighted),
                                                R against not R (precision of the first against the second, recall)
  labSelfN, labSelfAgreePct, labSelfAgreeUnwPct, labSelfKappa, labSelfKappaUnw
                                                the judge against itself on the repeat set
"""
from __future__ import annotations

import json
import os
import re

PREFIX = "lab"
PATH = os.path.join("results", "reading_audit", "reading_audit.json")


def reading_audit_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        R = json.load(fh)
    return R if R.get("macros") else None


def reading_audit_macros(R) -> dict:
    if not R:
        return {}
    out = {}
    for k, v in R["macros"].items():
        if not re.fullmatch(PREFIX + r"[A-Z][A-Za-z]*", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
        out[k] = str(v)
    return out
