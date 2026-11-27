"""Macros of w under five-shot answering, prefix fsW.
Read from results/fewshot_w/fewshot_w.json (scripts/run_fewshot_w.py). Every value is read from that file; nothing is
typed here.

fewshot_w_results(root) reads the file, or returns None until it exists; fewshot_w_macros(R) returns {name: value},
after checking every name (letters only, prefix fsW) and that no name is set twice.

Five-shot requested against zero-shot requested, same models, configurations and items. <T> is Gsm (GSM8K-29) or Bbh
(BIG-Bench Hard-28). <Sp> is empty for all eight configurations (the headline), Disc for the four discovery
configurations, Val for the four validation configurations. <Sc> is empty for strict scoring, Content for content
scoring. w and compliance in %, one decimal; changes in w and compliance in points of share, one decimal; gains in pp,
two decimals; rank statistics two decimals; $-$ for a minus. Lo and Hi: percentile model-bootstrap interval (9,999
resamples).
  fsW<T>M                                 models in the panel
  fsW<T><Sp>WZsPct, fsW<T><Sp>WFsPct      mean w of the zero-shot and five-shot requested arms
  fsW<T><Sp>WChange{,Lo,Hi}               mean per-model change in w (five-shot minus zero-shot)
  fsW<T><Sp>WChangeMedian, ...Min, ...Max per-model change in w: median, smallest, largest
  fsW<T><Sp>WDownN, WUpN, WSameN          models whose w fell, rose, stayed exactly equal
  fsW<T><Sp>CompZsPct, CompFsPct          strict-format compliance of each requested arm
  fsW<T><Sp>CompChange{,Lo,Hi}, CompChangeMedian, CompChangeMin, CompChangeMax   per-model change in compliance
  fsW<T><Sp>GainReq<Sc>, GainFd<Sc>       mean five-shot effect within the requested and the forced baseline (pp)
  fsW<T><Sp>SignN<Sc>, NoSignN<Sc>        models with S = 1 (the five-shot effect has opposite signs within the two
                                          baselines) and with S = 0
  fsW<T><Sp>FdUpReqDown<Sc>, FdDownReqUp<Sc>, BothUp<Sc>, BothDown<Sc>, ZeroGain<Sc>   models by the two signs
  fsW<T><Sp>Rho<Sc>{,Lo,Hi}, Tau<Sc>{,Lo,Hi}   Spearman's rho and Kendall's tau-b, fall in w against S
  fsW<T><Sp>FallSign<Sc>, FallNoSign<Sc>  mean fall in w (points) among models with S = 1 and S = 0
  fsW<T><Sp>FallDiff<Sc>{,Lo,Hi}          their difference (S = 1 minus S = 0)
  fsW<T><Sp>BootUndef<Sc>                 resamples with S constant (rank statistics undefined, left out of Lo and Hi)
  fsW<T><Sp>RhoCont<Sc>{,Lo,Hi}, TauCont<Sc>{,Lo,Hi}   rank statistics of the fall in w against g_R - g_F
  fsW<T>FsReqReplies, FsReqStrictCorrect, FsReqContentCorrect, FsReqUnreadable   content reading of the five-shot
                                          requested replies (counts over the eight configurations)
  fsW<T>Model<Name>WZsPct, ...WFsPct, ...WChange   per model, all eight configurations (<Name>: the model key in
                                          letters, digits spelled out, a dot as Pt)
  fsWNBoot, fsWCap                        resamples; the token cap of w (16)
"""
from __future__ import annotations

import json
import os
import re

PREFIX = "fsW"
PATH = os.path.join("results", "fewshot_w", "fewshot_w.json")
TASK = {"gsm8k": "Gsm", "bbh": "Bbh"}
SPLIT = {"all": "", "discovery": "Disc", "validation": "Val"}
SCORE = {"strict": "", "content": "Content"}
DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def fewshot_w_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _fin(x, what):
    if x is None:
        raise ValueError(f"{what}: no finite value in the results file")
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{what}: not finite ({v})")
    return v


def fx(x, nd, what):
    """As make_numbers.f: a hyphen prints $-$ (a value that rounds to zero keeps its sign)."""
    return f"{_fin(x, what):.{nd}f}".replace("-", "$-$")


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def model_token(key: str) -> str:
    """Letters only: parts split at non-alphanumerics, each capitalised, digits spelled out, a dot as Pt."""
    out = []
    for part in re.split(r"[^A-Za-z0-9.]+", key):
        for run in re.findall(r"[A-Za-z]+|[0-9]+|\.", part):
            if run == ".":
                out.append("Pt")
            elif run.isdigit():
                out.append("".join(DIGITS[int(c)] for c in run))
            else:
                out.append(run[:1].upper() + run[1:].lower())
    return "".join(out)


def fewshot_w_macros(R) -> dict:
    if not R:
        return {}
    out = {}

    def put(k, v):
        if k in out:
            raise ValueError(f"macro {k} set twice")
        out[k] = v

    def lohi(stem, ci, scale, nd, what):
        if not ci:
            raise ValueError(f"{what}: no interval")
        put(stem + "Lo", fx(scale * ci[0], nd, what + " low"))
        put(stem + "Hi", fx(scale * ci[1], nd, what + " high"))

    caps = set()
    for task, b in R["tasks"].items():
        T = PREFIX + TASK[task]
        caps.add(int(b["cap_tokens"]))
        put(f"{T}M", str(int(b["M"])))
        for sp, s in b["splits"].items():
            P = T + SPLIT[sp]
            w, c = s["w"], s["compliance"]
            put(f"{P}WZsPct", fx(100 * w["zero_shot_mean"], 1, f"{P} w zero-shot"))
            put(f"{P}WFsPct", fx(100 * w["five_shot_mean"], 1, f"{P} w five-shot"))
            put(f"{P}WChange", fx(100 * w["change_mean"]["mean"], 1, f"{P} w change"))
            lohi(f"{P}WChange", w["change_mean"]["ci95"], 100, 1, f"{P} w change")
            d = w["change_distribution"]
            put(f"{P}WChangeMedian", fx(100 * d["median"], 1, f"{P} w change median"))
            put(f"{P}WChangeMin", fx(100 * d["min"], 1, f"{P} w change min"))
            put(f"{P}WChangeMax", fx(100 * d["max"], 1, f"{P} w change max"))
            put(f"{P}WDownN", str(int(d["n_below_zero"])))
            put(f"{P}WUpN", str(int(d["n_above_zero"])))
            put(f"{P}WSameN", str(int(d["n_zero"])))
            put(f"{P}CompZsPct", fx(100 * c["zero_shot_mean"], 1, f"{P} compliance zero-shot"))
            put(f"{P}CompFsPct", fx(100 * c["five_shot_mean"], 1, f"{P} compliance five-shot"))
            put(f"{P}CompChange", fx(100 * c["change_mean"]["mean"], 1, f"{P} compliance change"))
            lohi(f"{P}CompChange", c["change_mean"]["ci95"], 100, 1, f"{P} compliance change")
            dc = c["change_distribution"]
            put(f"{P}CompChangeMedian", fx(100 * dc["median"], 1, f"{P} compliance change median"))
            put(f"{P}CompChangeMin", fx(100 * dc["min"], 1, f"{P} compliance change min"))
            put(f"{P}CompChangeMax", fx(100 * dc["max"], 1, f"{P} compliance change max"))
            for sc, x in s["scores"].items():
                S = SCORE[sc]
                A = x["association"]
                if A["variable"] != "binary":
                    raise ValueError(f"{P} {sc}: the sign-change variable is not binary")
                put(f"{P}GainReq{S}", fx(x["mean_gain_requested"], 2, f"{P} {sc} gain requested"))
                put(f"{P}GainFd{S}", fx(x["mean_gain_forced"], 2, f"{P} {sc} gain forced"))
                put(f"{P}SignN{S}", str(int(A["n_sign_change"])))
                put(f"{P}NoSignN{S}", str(int(A["n_no_change"])))
                dr = x["directions"]
                for key, nm in (("forced_up_requested_down", "FdUpReqDown"), ("forced_down_requested_up", "FdDownReqUp"),
                                ("both_up", "BothUp"), ("both_down", "BothDown"), ("zero_under_either", "ZeroGain")):
                    put(f"{P}{nm}{S}", str(int(dr[key])))
                put(f"{P}Rho{S}", fx(A["spearman"], 2, f"{P} {sc} spearman"))
                lohi(f"{P}Rho{S}", A.get("spearman_ci95"), 1, 2, f"{P} {sc} spearman")
                put(f"{P}Tau{S}", fx(A["kendall_tau_b"], 2, f"{P} {sc} kendall"))
                lohi(f"{P}Tau{S}", A.get("kendall_tau_b_ci95"), 1, 2, f"{P} {sc} kendall")
                put(f"{P}FallSign{S}", fx(100 * A["mean_fall_sign_change"], 1, f"{P} {sc} fall S=1"))
                put(f"{P}FallNoSign{S}", fx(100 * A["mean_fall_no_change"], 1, f"{P} {sc} fall S=0"))
                put(f"{P}FallDiff{S}", fx(100 * A["mean_fall_difference"], 1, f"{P} {sc} fall difference"))
                lohi(f"{P}FallDiff{S}", A.get("mean_fall_difference_ci95"), 100, 1, f"{P} {sc} fall difference")
                put(f"{P}BootUndef{S}", str(int(A["boot_undefined_rank"])))
                C = A["continuous"]
                put(f"{P}RhoCont{S}", fx(C["spearman"], 2, f"{P} {sc} spearman continuous"))
                lohi(f"{P}RhoCont{S}", C.get("spearman_ci95"), 1, 2, f"{P} {sc} spearman continuous")
                put(f"{P}TauCont{S}", fx(C["kendall_tau_b"], 2, f"{P} {sc} kendall continuous"))
                lohi(f"{P}TauCont{S}", C.get("kendall_tau_b_ci95"), 1, 2, f"{P} {sc} kendall continuous")
        cnt = R["content_record"]["counts"][task]
        put(f"{T}FsReqReplies", comma(cnt["n_replies"]))
        put(f"{T}FsReqStrictCorrect", comma(cnt["strict_correct"]))
        put(f"{T}FsReqContentCorrect", comma(cnt["content_correct"]))
        put(f"{T}FsReqUnreadable", comma(cnt["unreadable"]))
        toks = {}
        for r in b["per_model_all"]:
            tok = model_token(r["model_key"])
            if tok in toks:
                raise ValueError(f"model token {tok} for both {toks[tok]} and {r['model_key']}")
            toks[tok] = r["model_key"]
            Q = f"{T}Model{tok}"
            put(f"{Q}WZsPct", fx(100 * r["w_zero_shot"], 1, f"{Q} w zero-shot"))
            put(f"{Q}WFsPct", fx(100 * r["w_five_shot"], 1, f"{Q} w five-shot"))
            put(f"{Q}WChange", fx(100 * r["w_change"], 1, f"{Q} w change"))
    if len(caps) != 1:
        raise ValueError(f"more than one token cap: {caps}")
    put(f"{PREFIX}Cap", str(caps.pop()))
    put(f"{PREFIX}NBoot", comma(R["n_boot"]))
    for k in out:
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
    return out


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for k, v in sorted(fewshot_w_macros(fewshot_w_results(root)).items()):
        print(f"\\newcommand{{\\{k}}}{{{v}}}")
