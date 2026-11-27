r"""HotpotQA under the content reading. The reading itself is src/hotpot_task.content_read (version hotpot_task.CONTENT_SCORER_VERSION); this module holds
what the analyses around it share:

format_flags   where an output puts its answer, on every benchmark: whether it copies the instruction's
               placeholder ("<answer>" on HotpotQA and BIG-Bench Hard, "<number>" on GSM8K and MATH-500, the
               placeholders of the banks' answer formats) anywhere, whether its last complete \boxed{...} is that
               placeholder (with or without text after it), and one exclusive place for the answer:
                   usable_box    a complete \boxed{...} that is not a placeholder box (src/hotpot_task.
                                 is_placeholder_box, the one placeholder rule) and whose content, unwrapped and
                                 cleaned as the content reading cleans it, normalises (HotpotQA normalisation) to
                                 something other than "" and the placeholder's word ("answer", "number");
                   bare_box      no usable box, and a box written without its backslash ("boxed{");
                   unclosed_box  neither, and a \boxed{ that never closes;
                   unboxed       none of these: the answer, if any, is outside every box (placeholder-only boxes
                                 included). A box written with angle brackets or as a tag ("\boxed<Odysseus>",
                                 "<boxed>Liguria</boxed>") is counted here: the content readers read it as a box,
                                 but this is a count of format.
               The brace matching is src/hotpot_task.boxes for every benchmark (a count of format, not a score).
long_flags     the definitions of a long requested output for the decomposition.
decomposition_variant
               the exact decomposition of the baseline shift (src/baseline_mixture.decomposition) and the choice by
               (1 - w) x forced gain for one definition of long and one score, with the choices refitted on the
               discovery draws (exploratory: these definitions were chosen after the saved prediction was checked).
stratified_draw
               a seeded draw of n rows stratified by a column.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import baseline_mixture as bm
from . import decision as dc
from . import hotpot_task as ht

PLACEHOLDERS = {"hotpot": "<answer>", "bbh": "<answer>", "gsm8k": "<number>", "math": "<number>"}
ANSWER_PLACES = ("usable_box", "bare_box", "unclosed_box", "unboxed")
FLAG_COLUMNS = ("placeholder_copy", "last_box_placeholder", "answer_place")
_BARE_BOX = re.compile(r"(?<!\\)boxed\s*\{")


def _clean(u: str, placeholder: str) -> str:
    u = re.sub(r"^\s*" + re.escape(placeholder) + r"\s*[-:]?\s*", "", u, flags=re.IGNORECASE).strip()
    if len(u) > 1 and u[0] == "<" and u[-1] == ">":
        u = u[1:-1].strip()
    return u


def format_flags(text, placeholder: str = ht.PLACEHOLDER) -> tuple[bool, bool, str]:
    """(placeholder_copy, last_box_placeholder, answer_place) of one output (module doc)."""
    t = text if isinstance(text, str) else ""
    ph = placeholder.lower()
    word = ht.normalize_answer(placeholder)
    complete, unclosed = ht.boxes(t)
    copy = ph in t.lower()
    last_ph = bool(complete) and complete[-1].strip().lower().startswith(ph)
    if any(not ht.is_placeholder_box(c) and ht.normalize_answer(_clean(ht.unwrap(c), placeholder)) not in ("", word)
           for c in complete):
        place = "usable_box"
    elif _BARE_BOX.search(t):
        place = "bare_box"
    elif unclosed is not None:
        place = "unclosed_box"
    else:
        place = "unboxed"
    return copy, last_ph, place


def flag_frame(texts, placeholder: str) -> pd.DataFrame:
    """FLAG_COLUMNS for a sequence of outputs (identical texts read once)."""
    cache: dict = {}
    rows = []
    for t in texts:
        k = t if isinstance(t, str) else ""
        if k not in cache:
            cache[k] = format_flags(k, placeholder)
        rows.append(cache[k])
    return pd.DataFrame(rows, columns=list(FLAG_COLUMNS))


def flag_counts(df: pd.DataFrame, by=("model_key", "arm")) -> pd.DataFrame:
    """Counts per group: n, placeholder_copy, last_box_placeholder, each answer place, and placeholder copies with no
    usable box (placeholder_no_usable_box)."""
    g = df.groupby(list(by), observed=True)
    out = pd.DataFrame({"n": g.size(), "placeholder_copy": g["placeholder_copy"].sum(),
                        "last_box_placeholder": g["last_box_placeholder"].sum()})
    for p in ANSWER_PLACES:
        out[p] = (df["answer_place"] == p).groupby([df[c] for c in by], observed=True).sum()
    out["placeholder_no_usable_box"] = (df["placeholder_copy"] & (df["answer_place"] != "usable_box")).groupby(
        [df[c] for c in by], observed=True).sum()
    return out.fillna(0).astype(int).reset_index()


# ---------------------------------------------------------------------------- long answers

LONG_DEFINITIONS = {
    "cap64": "requested output longer than 64 generated tokens (HotpotQA's forced cap; the saved rule)",
    "cap16": "requested output longer than 16 generated tokens (the forced cap of GSM8K, BIG-Bench Hard and MATH-500)",
    "excess8": "requested output at least 8 generated tokens longer than the same row's forced output",
}
CHECK_DEFINITIONS = {
    "excess8_gt": "requested output more than 8 generated tokens longer than the same row's forced output (a "
                  "reproduction check of an earlier diagnostic script)",
}


def long_flags(name: str, ntok_req, ntok_fd=None) -> np.ndarray:
    r = np.asarray(ntok_req, float)
    if name == "cap64":
        return r > 64
    if name == "cap16":
        return r > 16
    if name in ("excess8", "excess8_gt"):
        if ntok_fd is None:
            raise ValueError(f"{name} needs the forced output's token count")
        e = r - np.asarray(ntok_fd, float)
        return e >= 8 if name == "excess8" else e > 8
    raise ValueError(f"unknown definition {name!r}")


def row_weights(config_ids, draws) -> np.ndarray:
    """Per row, the number of draws of its configuration (repeats count; 0 for a configuration in no draw)."""
    cnt = Counter(map(str, draws))
    return np.array([cnt.get(str(c), 0) for c in config_ids], float)


def _block(wide, panel, flag, draws, weighted: bool, score: str) -> dict:
    x = wide[wide["config_id"].isin(set(map(str, draws)))]
    m = pd.Index(panel).get_indexer(x["model_key"])
    if (m < 0).any():
        raise ValueError("a row's model is not in the panel")
    wt = row_weights(x["config_id"], draws) if weighted else None
    return bm.decomposition(m, x[flag].to_numpy(bool), x[f"requested_{score}"].to_numpy(),
                            x[f"forced_{score}"].to_numpy(), x[f"cot_{score}"].to_numpy(), len(panel), wt)


def decomposition_variant(wide: pd.DataFrame, panel, flag: str, disc, val, score: str) -> dict:
    """One definition of long (the boolean column `flag` of `wide`) and one score (columns requested_<score>,
    forced_<score>, cot_<score>; 0/1 per row): the decomposition on all distinct configurations (each once), on the
    discovery and on the validation draws (each configuration weighted by its draws); the mechanism on the discovery
    means (pairs reversed between the gain over the requested arm and the gain over the forced arm, and the corrected
    gain (1 - w) x gain over forced); the choice by (1 - w) x forced gain against the choice by forced gain (and the
    short form), refitted on the discovery draws, valued under the requested arm on the validation means and on the
    validation configurations absent from every discovery draw (each once, with the paired difference). Edge cases are
    reported as they are."""
    panel = list(panel)
    disc, val = list(map(str, disc)), list(map(str, val))
    allc = sorted(set(disc) | set(val))
    sets = {"all_distinct": _block(wide, panel, flag, allc, False, score),
            "discovery": _block(wide, panel, flag, disc, True, score),
            "validation": _block(wide, panel, flag, val, True, score)}
    pd_, pv = sets["discovery"]["per_model"], sets["validation"]["per_model"]
    C, F, R, w = (np.array(pd_[k], float) for k in ("C", "F", "R", "w"))
    gS = np.array(pd_["C_short"], float) - np.array(pd_["F_short"], float)
    n = dc.third_size(len(panel))
    preds = {"forced_gain": C - F, "corrected": bm.corrected_gain(C - F, w),
             "corrected_short": (1 - w) * np.nan_to_num(gS, nan=0.0)}
    W, ties = {}, {}
    for k, p in preds.items():
        W[k], ties[k] = dc.choose(p, n)
    g_req_val = (np.array(pv["C"], float) - np.array(pv["R"], float))[None]
    reg = {k: float(dc.regret_plugin(Wk, g_req_val, n)[0]) for k, Wk in W.items()}
    choice = {"n_chosen": n, "ties_at_cut": {k: int(v) for k, v in ties.items()},
              "weights": {k: v.tolist() for k, v in W.items()},
              "validation_means": {"regret": reg, "holds": bool(reg["corrected"] < reg["forced_gain"]),
                                   "equal": bool(reg["corrected"] == reg["forced_gain"])},
              "chosen_by_both_forms": float(np.minimum(W["corrected"], W["corrected_short"]).sum()),
              "note": "choices refitted on the discovery draws under this definition and score; exploratory"}
    fresh = sorted(set(u for u in val if u not in set(disc)))
    if len(fresh) >= 2:
        x = wide[wide["config_id"].isin(set(fresh))]
        cm = x.groupby(["config_id", "model_key"])[[f"cot_{score}", f"requested_{score}"]].mean() * 100
        G = (cm[f"cot_{score}"] - cm[f"requested_{score}"]).unstack("model_key").loc[fresh, panel].to_numpy(float)
        choice["fresh_configurations"] = {
            "n_configurations": len(fresh), "regret": {k: float(dc.regret_plugin(Wk, G, n)[0]) for k, Wk in W.items()},
            "paired_corrected_minus_forced": dc.paired_difference(W["corrected"], W["forced_gain"], G, n)}
    else:
        choice["fresh_configurations"] = {"n_configurations": len(fresh), "note": "fewer than two fresh configurations"}
    mech = {"discovery": bm.mechanism(C - R, C - F, w)}
    pa = sets["all_distinct"]["per_model"]
    Ca, Fa, Ra, wa = (np.array(pa[k], float) for k in ("C", "F", "R", "w"))
    mech["all_distinct"] = bm.mechanism(Ca - Ra, Ca - Fa, wa)
    return {"flag": flag, "score": score, "sets": sets, "choice": choice, "mechanism": mech,
            "w_discovery_median": float(np.median(w)), "w_all_distinct_median": float(np.median(wa))}


# ---------------------------------------------------------------------------- the draw for rater C


def seed_of(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def allocate(available: dict, n: int, rng) -> dict:
    """Equal allocation of n places over the strata, each capped at what it holds; places a stratum cannot fill go
    one at a time to the strata with rows left, in a seeded order. Every stratum with a row gets at least one place
    when n allows."""
    keys = sorted(k for k, v in available.items() if v > 0)
    if sum(available[k] for k in keys) < n:
        raise ValueError(f"only {sum(available[k] for k in keys)} rows for {n} places")
    alloc = {k: 0 for k in keys}
    left = n
    while left > 0:
        open_ = [k for k in keys if alloc[k] < available[k]]
        share = left // len(open_)
        if share == 0:
            for k in [open_[i] for i in rng.permutation(len(open_))][:left]:
                alloc[k] += 1
            break
        for k in open_:
            take = min(share, available[k] - alloc[k])
            alloc[k] += take
            left -= take
    return alloc


def stratified_draw(df: pd.DataFrame, n: int, by: str, label: str | int) -> pd.DataFrame:
    """n rows of df stratified by column `by` (allocate), drawn without replacement within each stratum from the rows
    in their given order, with numpy.random.default_rng(seed_of(label)), or default_rng(label) for an integer label;
    returned in a seeded random order."""
    rng = np.random.default_rng(label if isinstance(label, int) else seed_of(label))
    groups = {k: g for k, g in df.groupby(by, sort=True, observed=True)}
    alloc = allocate({k: len(g) for k, g in groups.items()}, n, rng)
    parts = []
    for k in sorted(alloc):
        g = groups[k]
        idx = rng.choice(len(g), size=alloc[k], replace=False)
        parts.append(g.iloc[np.sort(idx)])
    out = pd.concat(parts)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)
