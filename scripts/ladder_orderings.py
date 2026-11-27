r"""Orderings on the size ladder: the
reversal share among the six ladder models, in a combined panel of 35 models and among the models of 70B and above;
w by family and size; the baseline shift with its long-answer and short-answer parts; the form of Qwen3.5-122B-A10B's
unreadable requested replies.

Usage (repository root; set PYTHONIOENCODING=utf-8):
    python scripts/ladder_orderings.py
Run it after scripts/ladder_analysis.py, which rewrites ladder_note.md: this script puts its own section back into
ladder_note.md and ladder_note_clean.md (replacing it when present). Then rebuild paper/numbers.tex
(paper/make_numbers.py, which takes this file's macros through paper/macros_ladder_orderings.py) and run
scripts/verify_ladder_analysis.py, which recounts everything here from the scored rows with its own code.

Outputs: results/ladder/analysis/ladder_orderings.json, ladder_orderings.md, ladder_orderings_clean.md, and the section
"Orderings among the ladder models" of ladder_note.md (and its clean copy).

Bases, the panel's own. GSM8K, MATH-500 and BIG-Bench Hard: model means over the 24 discovery draws (a configuration
drawn twice counts twice), as results/arm_effect/ and results/decision_directions/ define them. HotpotQA: its paper
basis for the reversal count, the discovery-draw means of results/arm_effect/ (\aeHqReversedContent; its current count is the panel row of ladder_orderings.json;
results/hotpot_content/hotpot_content.json, arm_effect). The ladder's placements read HotpotQA content on pooled rows
(results/hotpot_content/tables/per_model_arm.csv); the count on that basis is given beside, as a check. Content scoring
primary, strict beside it. Outputs cut at the panel's caps primary; the uncut reading is in its own section (for the
Supplementary Information).

Estimators, the panel's code paths, nothing reimplemented:
  ladder model means  scripts/ladder_analysis.zero_shot_summary on the scored rows as they stand (readers named in the JSON): the
                      per-configuration mean of nebius_panel_report averaged over the draws, held as an exact ratio,
                      on the units with all three zero-shot arms; checked against ladder.json (1e-9);
  panel model means   robustness_inputs.load (strict) and run_decision.task_split (content), as
                      run_decision_directions.reversal_block; HotpotQA through run_hotpot_content._arrays3, as its
                      arm_effect_block; the 29-model counts reproduce the stored ones, and the stored model bootstrap
                      intervals of GSM8K, MATH-500 and BIG-Bench Hard are reproduced from their own seeds;
  reversals           src/robustness.pair_reversals (the \ae... counting): a pair ordered one way by CoT - requested
                      and the other way by CoT - forced; a pair tied under either (|difference| <= 1e-9 pp) is
                      neither reversed nor kept and stays among the pairs; ties are counted and listed;
  model bootstrap     src/decision_directions.reversal_uncertainty (models drawn with replacement, a pair of a model
                      with its own copy left out, 9,999 resamples, percentile 95% interval of the share), as
                      \ddir<T>RevBoot...; pooled over GSM8K, MATH-500 and BIG-Bench Hard, one draw of the union of
                      models serves all three benchmarks (a model absent from a benchmark adds no pair there) and the
                      share is the reversed pairs over the pairs, both summed over the three (pooled_share_on);
  baseline shift      scripts/run_baseline_mixture.decomposition_block (src/baseline_mixture.decomposition): requested
                      minus forced accuracy, split exactly into the part from rows whose requested output is longer than
                      16 generated tokens and the part from the others, each over all the model's rows, on the discovery
                      draws (each configuration weighted by its draws), as for the panel (16 tokens: the forced cap of
                      GSM8K, MATH-500 and BIG-Bench Hard, and the panel's cap16 variant on HotpotQA);
  w                   the ladder's (share of requested outputs longer than the forced cap, discovery draws; 16 tokens,
                      64 on HotpotQA); direction over size by scripts/ladder_analysis.direction_of.
Printing: scripts/ladder_analysis.fnum (half-even on the exact value). Macros (the block "macros" of the JSON, passed
through paper/macros_ladder_orderings.py): a value that rounds to zero prints without its sign, a minus as $-$, and
thousands as {,}.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ladder_analysis as la  # noqa: E402
import nebius_panel_report as npr  # noqa: E402
from src import decision as dc, decision_directions as dd, robustness as rb  # noqa: E402
from src import content_scoring as cs, hotpot_task as ht  # noqa: E402  (their versions are recorded, nothing else)
from src.estimators import tie_sign  # noqa: E402

OUT = ROOT / "results" / "ladder" / "analysis"
STEM = "ladder_orderings"
LADDER_NOTE = "ladder_note.md"
SECTION = "## Orderings among the ladder models"
REQUEST = "scripts/ladder_orderings.py"
KEYS = list(la.LADDER)                         # the six, in the order the ladder reports them
BIG = ["llama-3.1-70b-instruct-fp8", "qwen3.5-122b-a10b-fp8", "mistral-large-2512", "deepseek-v4.1-flash-1017"]
BIG_MIN_B = 70                                 # BIG: total parameters of 70B or more, checked by big_sizes()
QWEN = "qwen3.5-122b-a10b-fp8"
TASKS = la.TASKS                               # gsm8k, math, bbh, hotpot
DISC3 = ("gsm8k", "math", "bbh")               # the three benchmarks pooled
READINGS = la.READINGS                         # cut, uncut
SCORES = ("content", "strict")                 # content primary
SCORE_COL = {"strict": "strict_correct", "content": "content_correct"}
BASES = ("discovery", "all_draws", "pooled")   # discovery primary; the other two for the hand-count check
T_TAG = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh", "hotpot": "Hq"}
TNAME = la.TASK_NAME
MODEL_TAG = {"ministral-3-8b-instruct-2512": "MinistralEight", "ministral-3-14b-instruct-2512": "MinistralFourteen",
             "mistral-large-2512": "MistralLarge", "llama-3.1-70b-instruct-fp8": "LlamaSeventy",
             "qwen3.5-122b-a10b-fp8": "QwenOneTwentyTwo", "deepseek-v4.1-flash-1017": "DeepSeek",
             "qwen3.5-9b": "QwenNine", "qwen3.5-35b-a3b": "QwenThirtyFive", "llama-3.1-8b-instruct": "LlamaEight"}
FAM_TAG = {"Mistral 3": "Mistral", "Qwen3.5": "Qwen", "Llama 3.1": "Llama"}
VARIANT = {("content", "cut"): "", ("strict", "cut"): "Strict", ("content", "uncut"): "Uncut",
           ("strict", "uncut"): "StrictUncut"}
SPLIT_TOKENS = 16
N_BOOT = 9999                                  # as run_decision_directions.N_BOOT (checked at run time)
HAND = {"gsm8k": 2, "math": 4, "bbh": 2, "hotpot": 12}     # hand counts of 15 pairs (content, 48-draw means)
TOL = 1e-9
GAP_THRESHOLDS = (1.0, 2.0, 5.0, 10.0)        # pp; the panel's (src/robustness.pair_reversals, \ae...Gap...)
GAP_WORD = {1.0: "One", 2.0: "Two", 5.0: "Five", 10.0: "Ten"}
SUB_PP = 1.0                                   # the reversed pairs named in the text: a gap of this or less under either
RAW_CHOSEN = "data/d2_out/ladder2/chosen/{key}"
RAW_SHARDS = "data/d2_out/ladder2/{run}/{task}"
SOURCES = {
    "ladder_json": "results/ladder/analysis/ladder.json",
    "decision_directions": "results/decision_directions/decision_directions.json",
    "arm_effect": "results/arm_effect/arm_effect.json",
    "hotpot_content": "results/hotpot_content/hotpot_content.json",
    "compliance": "results/content_scoring/tables/compliance_{task}.csv",
    "baseline_mixture": "results/baseline_mixture/baseline_mixture.json",
    "hotpot_thresholds": "results/hotpot/decomposition_thresholds_post_hoc.json",
    "hotpot_mechanism": "manifests/predictions_hotpot_mechanism.json",
}


def rel(p) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def seed(label: str):
    return np.random.default_rng(dc.label_seed(f"ladder_orderings/{label}"))


def nm(key: str) -> str:
    return la.nm(key) if key in la.LADDER else key


def fx(x) -> float | None:
    return None if x is None else float(x)


# --------------------------------------------------------------------------------------------------------------------
# Reversals
# --------------------------------------------------------------------------------------------------------------------

def reversals(keys: list[str], g1, g2) -> dict:
    """Pairs ordered one way by g1 (CoT - requested) and the other by g2 (CoT - forced): src/robustness.pair_reversals,
    with the reversed and the tied pairs named (src/decision_directions.reversal_matrix, the same convention); its
    by_threshold (the pairs whose two gaps both exceed 1, 2, 5 and 10 pp, and the reversed
    among them, as the panel's \\ae...Gap... macros count them); and each reversed pair's two gaps, |difference| under
    each gain, exact when the gains are (printed half-even to two decimals)."""
    e1, e2 = list(g1), list(g2)
    g1, g2 = np.asarray([float(x) for x in g1]), np.asarray([float(x) for x in g2])
    r = rb.pair_reversals(g1, g2)
    Rm = dd.reversal_matrix(g1, g2)
    i, j = np.triu_indices(len(keys), 1)
    s1 = tie_sign(g1[i] - g1[j])
    s2 = tie_sign(g2[i] - g2[j])
    rev = [[keys[a], keys[b]] for a, b in zip(i, j) if Rm[a, b]]
    tie = [[keys[a], keys[b]] for a, b, x, y in zip(i, j, s1, s2) if x == 0 or y == 0]
    if len(rev) != r["reversed"] or len(tie) != r["ties"]:
        raise RuntimeError("the reversal matrix and pair_reversals disagree")
    gaps = []
    for a, b in zip(i, j):
        if Rm[a, b]:
            d1, d2 = abs(e1[a] - e1[b]), abs(e2[a] - e2[b])
            gaps.append({"pair": [keys[a], keys[b]], "gap_requested": fx(d1), "gap_forced": fx(d2),
                         "printed": [la.fnum(d1, 2), la.fnum(d2, 2)], "min_gap": fx(min(d1, d2))})
    bt = [{"min_gap_pp": x["min_gap_pp"], "pairs_both_above": x["pairs_both_above"], "reversed": x["reversed"]}
          for x in r["by_threshold"] if x["min_gap_pp"] in GAP_THRESHOLDS]
    return {"pairs": r["pairs"], "reversed": r["reversed"], "ties": r["ties"], "share": r["reversed_share"],
            "reversed_pairs": rev, "tied_pairs": tie, "models": len(keys), "by_threshold": bt, "reversed_gaps": gaps}


def pooled_counts(blocks: dict) -> dict:
    """Reversed pairs and pairs summed over benchmarks (each benchmark's pairs among its own models), the threshold
    rows summed the same way."""
    p = sum(b["pairs"] for b in blocks.values())
    r = sum(b["reversed"] for b in blocks.values())
    t = sum(b["ties"] for b in blocks.values())
    bt = [{"min_gap_pp": th, "pairs_both_above": sum(x["pairs_both_above"] for b in blocks.values()
                                                     for x in b["by_threshold"] if x["min_gap_pp"] == th),
           "reversed": sum(x["reversed"] for b in blocks.values() for x in b["by_threshold"] if x["min_gap_pp"] == th)}
          for th in GAP_THRESHOLDS]
    return {"pairs": p, "reversed": r, "ties": t, "share": (r / p) if p else None, "tasks": list(blocks),
            "by_threshold": bt}


def by_gap(x: dict, th: float) -> dict:
    """The threshold row of a block as a block of its own: reversed and pairs among the pairs whose two gaps both
    exceed th pp."""
    b = next(y for y in x["by_threshold"] if y["min_gap_pp"] == th)
    return {"reversed": b["reversed"], "pairs": b["pairs_both_above"]}


def pooled_share_on(parts: list, ix) -> float:
    """The pooled reversed share for one draw of union indices ix: per benchmark, the drawn models present there
    (pos >= 0), every pair of positions i < j whose models differ (a model drawn twice is not a pair, as
    src/decision_directions.share_on), reversed pairs and pairs summed over the benchmarks; nan with no pair."""
    ix = np.asarray(ix)
    rev = n = 0
    for Rm, pos in parts:
        loc = pos[ix]
        loc = loc[loc >= 0]
        i, j = np.triu_indices(len(loc), 1)
        ok = loc[i] != loc[j]
        rev += int(Rm[loc[i][ok], loc[j][ok]].sum())
        n += int(ok.sum())
    return rev / n if n else float("nan")


def percentile_ci(v) -> list | None:
    """The percentile 95% interval of src/decision_directions.reversal_uncertainty (finite values)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None


def pooled_bootstrap(blocks: dict, n_boot: int, rng) -> dict:
    """blocks: {task: (keys, g1, g2)}. The pooled reversed share over the union of models with its model bootstrap:
    one draw of the union (with replacement) serves every benchmark."""
    union: list[str] = []
    for keys, _, _ in blocks.values():
        union += [k for k in keys if k not in union]
    parts = []
    for keys, g1, g2 in blocks.values():
        Rm = dd.reversal_matrix(np.asarray(g1, float), np.asarray(g2, float))
        idx = {k: i for i, k in enumerate(keys)}
        parts.append((Rm, np.array([idx.get(k, -1) for k in union])))
    U = len(union)
    obs = pooled_share_on(parts, np.arange(U))
    boot = np.array([pooled_share_on(parts, rng.integers(0, U, U)) for _ in range(n_boot)])
    return {"union": U, "share": obs, "model_bootstrap": {"n_boot": int(n_boot), "ci95": percentile_ci(boot),
                                                          "n_undefined": int((~np.isfinite(boot)).sum())}}


# --------------------------------------------------------------------------------------------------------------------
# The ladder models: exact means from the scored rows (ladder_analysis.zero_shot_summary)
# --------------------------------------------------------------------------------------------------------------------

def wide_rows(sc: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per (configuration, item) with all three zero-shot arms: the requested output's token count and the
    strict and content scores of each arm, in the layout of scripts/run_baseline_mixture.task_rows."""
    cu = npr.complete_units(sc)
    wide = None
    for a, name in (("direct_answer", "requested"), ("forced_direct", "forced"), ("zero_shot_cot", "cot")):
        x = cu[cu["arm"] == a][["config_id", "item_id", "strict_correct", "content_correct"]
                                + (["n_output_tokens"] if a == "direct_answer" else [])]
        x = x.rename(columns={"strict_correct": f"{name}_strict", "content_correct": f"{name}_content"})
        wide = x if wide is None else wide.merge(x, on=["config_id", "item_id"], how="inner", validate="one_to_one")
    wide["model_key"] = key
    for c in ("requested", "forced", "cot"):
        for s in ("strict", "content"):
            wide[f"{c}_{s}"] = wide[f"{c}_{s}"].astype(float)
    return wide


def ladder_values(J: dict) -> tuple[dict, dict, list[str]]:
    """{reading: {task: {key: summary}}} of the six (exact ratios), {reading: {task: wide rows}}, and the checks
    against ladder.json."""
    V: dict = {r: {t: {} for t in TASKS} for r in READINGS}
    W: dict = {r: {t: [] for t in TASKS} for r in READINGS}
    checks = []
    for key in KEYS:
        for t in TASKS:
            for r in READINGS:
                sc = la.load_scored(la.model_paths(key, ROOT)[r][t])
                s = la.zero_shot_summary(t, sc)
                V[r][t][key] = s
                W[r][t].append(wide_rows(sc, key))
                jb = J["models"][key]["tasks"][t][r]
                for score in SCORE_COL:
                    for basis in BASES:
                        for f in ("gain_over_requested_pp", "gain_over_forced_pp"):
                            a, b = s[score][basis][f], jb[score][basis][f]
                            checks.append(abs(float(a) - float(b)) <= TOL)
                        for arm in ("requested", "forced", "cot"):
                            a, b = s[score][basis]["accuracy_pp"][arm], jb[score][basis]["accuracy_pp"][arm]
                            checks.append(abs(float(a) - float(b)) <= TOL)
                checks.append(abs(float(s["w"]["discovery"]) - float(jb["w"]["discovery"])) <= TOL)
                del sc
    if not all(checks):
        raise SystemExit(f"{sum(not c for c in checks)} ladder values differ from ladder.json: rerun "
                         "scripts/ladder_analysis.py on the current scored rows first")
    return V, {r: {t: pd.concat(W[r][t], ignore_index=True) for t in TASKS} for r in READINGS}, \
        [f"ladder.json: {len(checks)} values recomputed from the scored rows, all equal (1e-9)"]


def gains(V: dict, reading: str, t: str, score: str, basis: str, keys: list[str]) -> tuple[list, list]:
    """CoT - requested and CoT - forced of the given ladder models (exact)."""
    g1 = [V[reading][t][k][score][basis]["gain_over_requested_pp"] for k in keys]
    g2 = [V[reading][t][k][score][basis]["gain_over_forced_pp"] for k in keys]
    return g1, g2


# --------------------------------------------------------------------------------------------------------------------
# The panel: model means through the panel's own loaders
# --------------------------------------------------------------------------------------------------------------------

def panel_means() -> tuple[dict, list[str]]:
    """{task: {score: {"keys", "families", "requested", "forced", "cot"}}} on the discovery draws, through the panel's
    loaders; checked against the stored counts, the files the ladder is placed against, and the ladder's draws."""
    import robustness_inputs as ri
    import run_decision as rd
    import run_hotpot_content as rhc
    P: dict = {}
    notes = []
    for t in DISC3:
        x = ri.load(t)
        f, d = x["facts"], x["discovery"]
        if f["discovery_config_ids"] != npr.sampler(t)[0]:
            raise SystemExit(f"{t}: the panel's discovery draws differ from the ladder's sampler")
        P[t] = {"strict": {"keys": list(f["panel"]), "families": list(f["families"]),
                           **{a: d[a].mean(0) for a in ("requested", "forced", "cot")}}}
        c = rd.task_split(t, "content", "discovery")
        if [str(m) for m in c["panel"]] != f["panel"] or [str(u) for u in c["units"]] != f["discovery_config_ids"]:
            raise SystemExit(f"{t}: content discovery arrays differ from the strict ones in panel or draws")
        P[t]["content"] = {"keys": list(f["panel"]), "families": list(f["families"]),
                           **{a: np.asarray(c[a], float).mean(0) for a in ("requested", "forced", "cot")}}
        comp = pd.read_csv(ROOT / SOURCES["compliance"].format(task=t)).set_index("model_key")
        for s in ("strict", "content"):
            for a in ("requested", "forced", "cot"):
                v = comp.loc[P[t][s]["keys"], f"{a}_{s}_correct"].to_numpy(float)
                if np.abs(v - P[t][s][a]).max() > TOL:
                    raise SystemExit(f"{t} {s} {a}: the panel's means differ from {SOURCES['compliance']}")
        notes.append(f"{TNAME[t]}: panel means equal {SOURCES['compliance'].format(task=t)} (1e-9)")
    ctx = rhc.setup()
    if list(ctx["disc"]) != npr.sampler("hotpot")[0]:
        raise SystemExit("hotpot: the panel's discovery draws differ from the ladder's sampler")
    P["hotpot"] = {}
    for s, path in (("strict", rhc.STRICT_TABLE), ("content", str(Path(rhc.VIEW) / "scored.parquet"))):
        breq, bfd, pc = rhc._arrays3(path, ctx, ctx["disc"])
        P["hotpot"][s] = {"keys": list(ctx["panel"]), "families": list(ctx["fams"]),
                          "requested": breq.mean(0), "forced": bfd.mean(0), "cot": pc.mean(0)}
    H = json.loads((ROOT / SOURCES["hotpot_content"]).read_text(encoding="utf-8"))["arm_effect"]["content"]["per_model"]
    for a in ("requested", "forced", "cot"):
        v = np.array([H[k][a] for k in P["hotpot"]["content"]["keys"]])
        if np.abs(v - P["hotpot"]["content"][a]).max() > TOL:
            raise SystemExit(f"hotpot content {a}: the panel's means differ from {SOURCES['hotpot_content']}")
    notes.append(f"HotpotQA: content means equal {SOURCES['hotpot_content']} (arm_effect, 1e-9)")
    return P, notes


def stored_panel_counts() -> dict:
    """The panel's counts as stored by its analyses: {task: {score: (reversed, pairs)}}."""
    D = json.loads((ROOT / SOURCES["decision_directions"]).read_text(encoding="utf-8"))["tasks"]
    out = {t: {s: (D[t]["reversal"][s]["reversed"], D[t]["reversal"][s]["pairs"]) for s in ("strict", "content")}
           for t in DISC3}
    A = json.loads((ROOT / SOURCES["arm_effect"]).read_text(encoding="utf-8"))["tasks"]["hotpot"]["reversals"]
    H = json.loads((ROOT / SOURCES["hotpot_content"]).read_text(encoding="utf-8"))["arm_effect"]["content"]["reversals"]
    out["hotpot"] = {"strict": (A["reversed"], A["pairs"]), "content": (H["reversed"], H["pairs"])}
    return out


# --------------------------------------------------------------------------------------------------------------------
# Reversals
# --------------------------------------------------------------------------------------------------------------------

def item_reversals(V: dict, P: dict) -> tuple[dict, list[str]]:
    import run_decision_directions as rdd
    if rdd.N_BOOT != N_BOOT:
        raise SystemExit(f"run_decision_directions.N_BOOT is {rdd.N_BOOT}, this script's {N_BOOT}")
    notes = []
    stored = stored_panel_counts()
    DD = json.loads((ROOT / SOURCES["decision_directions"]).read_text(encoding="utf-8"))["tasks"]
    six: dict = {}
    big: dict = {}
    comb: dict = {}
    panel: dict = {}
    for r in READINGS:
        for s in SCORES:
            for basis in BASES:
                blk = {t: reversals(KEYS, *gains(V, r, t, s, basis, KEYS)) for t in TASKS}
                blk["pooled"] = pooled_counts({t: blk[t] for t in DISC3})
                six.setdefault(r, {}).setdefault(s, {})[basis] = blk
            bb = {t: reversals(BIG, *gains(V, r, t, s, "discovery", BIG)) for t in TASKS}
            bb["pooled"] = pooled_counts({t: bb[t] for t in DISC3})
            big.setdefault(r, {})[s] = bb
            cb, pooled_in = {}, {}
            for t in TASKS:
                p = P[t][s]
                keys = p["keys"] + KEYS
                fams = p["families"] + [la.LADDER[k]["family"] for k in KEYS]
                g1l, g2l = gains(V, r, t, s, "discovery", KEYS)
                g1 = np.concatenate([p["cot"] - p["requested"], [float(x) for x in g1l]])
                g2 = np.concatenate([p["cot"] - p["forced"], [float(x) for x in g2l]])
                x = reversals(keys, g1, g2)
                u = dd.reversal_uncertainty(g1, g2, fams, N_BOOT, seed(f"combined/{t}/{s}/{r}"))
                if (u["reversed"], u["pairs"]) != (x["reversed"], x["pairs"]):
                    raise RuntimeError("reversal_uncertainty and pair_reversals disagree")
                x.update({k: u[k] for k in ("model_bootstrap", "family_bootstrap", "within_family", "between_family",
                                            "leave_one_family_out_range")})
                x["with_ladder"] = sum(1 for a, b in x["reversed_pairs"] if a in KEYS or b in KEYS)
                cb[t] = x
                if t in DISC3:
                    pooled_in[t] = (keys, g1, g2)
            pc = pooled_counts({t: cb[t] for t in DISC3})
            pb = pooled_bootstrap(pooled_in, N_BOOT, seed(f"combined/pooled/{s}/{r}"))
            if abs(pb["share"] - pc["share"]) > 1e-12:
                raise RuntimeError("pooled share and pooled counts disagree")
            pc["model_bootstrap"] = pb["model_bootstrap"]
            pc["union"] = pb["union"]
            cb["pooled"] = pc
            comb.setdefault(r, {})[s] = cb
    for s in SCORES:
        pb_in, blk = {}, {}
        for t in TASKS:
            p = P[t][s]
            g1, g2 = p["cot"] - p["requested"], p["cot"] - p["forced"]
            x = reversals(p["keys"], g1, g2)
            if (x["reversed"], x["pairs"]) != tuple(stored[t][s]):
                raise SystemExit(f"{t} {s}: the panel's count {x['reversed']} of {x['pairs']} differs from the stored "
                                 f"{stored[t][s]}")
            if t in DISC3:
                u = dd.reversal_uncertainty(g1, g2, p["families"], N_BOOT, rdd.seed(f"reversal/{t}/{s}"))
                st = DD[t]["reversal"][s]["model_bootstrap"]["ci95"]
                if u["model_bootstrap"]["ci95"] != st:
                    raise SystemExit(f"{t} {s}: the model bootstrap does not reproduce the stored interval")
                x["model_bootstrap_source"] = f"{SOURCES['decision_directions']} (reproduced from its seed)"
                pb_in[t] = (p["keys"], g1, g2)
            else:
                u = dd.reversal_uncertainty(g1, g2, p["families"], N_BOOT, seed(f"panel/{t}/{s}"))
                x["model_bootstrap_source"] = "computed here (seed label ladder_orderings/panel/hotpot/<score>)"
            x.update({k: u[k] for k in ("model_bootstrap", "family_bootstrap")})
            blk[t] = x
        pc = pooled_counts({t: blk[t] for t in DISC3})
        pb = pooled_bootstrap(pb_in, N_BOOT, seed(f"panel/pooled/{s}"))
        pc["model_bootstrap"], pc["union"] = pb["model_bootstrap"], pb["union"]
        blk["pooled"] = pc
        panel[s] = blk
    notes.append("panel counts reproduce the stored ones on all four benchmarks, strict and content "
                 f"({SOURCES['decision_directions']}; {SOURCES['arm_effect']} and {SOURCES['hotpot_content']} for "
                 "HotpotQA), and the stored model bootstrap intervals of GSM8K, MATH-500 and BIG-Bench Hard reproduce "
                 "from their own seeds")
    return {"six": six, "big": big, "combined": comb, "panel": panel}, notes


def big_sizes() -> dict:
    """Each ladder model's recorded total parameters (B, as scripts/ladder_analysis.recorded_size reads them: the
    inventory, the checkpoint's name, or the published figures of manifests/ladder_published_sizes.json), with the
    group of 70B and above checked against them: every model of BIG records a total of BIG_MIN_B or more and every
    other ladder model less; a model with no recorded total stops the run. DeepSeek V4.1-Flash-1017 is grouped by
    its published size."""
    out = {}
    for k in KEYS:
        sz = la.recorded_size(ROOT, k)
        out[k] = {"total": sz.get("total"), "published": bool(sz.get("published")), "source": sz.get("source"),
                  "caveat": sz.get("caveat")}
    unsized = [k for k in KEYS if out[k]["total"] is None]
    if unsized:
        raise SystemExit(f"no total parameters recorded for {unsized}: the group of 70B and above cannot be checked")
    group = {k for k in KEYS if Fraction(out[k]["total"]) >= BIG_MIN_B}
    if group != set(BIG):
        raise SystemExit(f"the recorded sizes give the group {sorted(group)}, not BIG {sorted(BIG)}")
    return out


def big_size_note(sizes: dict) -> str:
    """The group of 70B and above by its members' recorded total parameters, with a caveat where the published figures
    are another checkpoint's."""
    return (f"Each is {BIG_MIN_B}B or more in total parameters by its recorded size (`ladder.md` section 4): "
            + ", ".join(f"{nm(k)} {sizes[k]['total']}B" + (f" ({sizes[k]['caveat']})" if sizes[k]["caveat"] else "")
                        for k in BIG) + ".")


def gap_line(label: str, blk: dict) -> str:
    """'<label>, gaps above 1 pp: <counts>. Above 2 pp: <counts>. Above 5 pp: <counts>.' (reversed of the pairs whose
    two gaps both exceed the threshold)."""
    parts = []
    for n_, th in enumerate((1.0, 2.0, 5.0)):
        b = {t: by_gap(blk[t], th) for t in list(TASKS) + ["pooled"]}
        head = f"{label}, gaps above {th:g} pp: " if n_ == 0 else f"Above {th:g} pp: "
        parts.append(head + counts_line(b, pct=False))
    return ". ".join(parts) + "."


def sub_pp_pairs(blk: dict) -> list[tuple[str, dict]]:
    """The reversed pairs of a block (per benchmark) whose gap under either gain is SUB_PP pp or less."""
    return [(t, g) for t in TASKS for g in blk[t]["reversed_gaps"] if g["min_gap"] <= SUB_PP]


def hand_check(V: dict, six: dict) -> dict:
    """The hand counts (content, 48-draw means, as printed in ladder_note.md) against the counts here: on the
    48 draws exactly, on the 48-draw values rounded to one decimal as the note prints them (a printed tie neither), and
    on the paper's basis; the pairs whose status differs between the 48 draws and the discovery draws."""
    out = {}
    for t in TASKS:
        g1, g2 = gains(V, "cut", t, "content", "all_draws", KEYS)
        r1 = [Fraction(la.fnum(x, 1)) for x in g1]
        r2 = [Fraction(la.fnum(x, 1)) for x in g2]
        printed = reversals(KEYS, r1, r2)
        a = six["cut"]["content"]["all_draws"][t]
        d = six["cut"]["content"]["discovery"][t]
        sa = {tuple(p) for p in a["reversed_pairs"]}
        sd = {tuple(p) for p in d["reversed_pairs"]}
        diff = []
        for p in sorted(sa ^ sd):
            i, j = KEYS.index(p[0]), KEYS.index(p[1])
            row = {"pair": list(p), "reversed_on_48_draws": p in sa, "reversed_on_discovery": p in sd}
            for basis in ("all_draws", "discovery"):
                h1, h2 = gains(V, "cut", t, "content", basis, KEYS)
                row[basis] = {"cot_minus_requested": [fx(h1[i]), fx(h1[j])], "cot_minus_forced": [fx(h2[i]), fx(h2[j])]}
            diff.append(row)
        out[t] = {"hand": HAND[t], "pairs": a["pairs"], "all_draws": a["reversed"],
                  "all_draws_reversed_gaps": a["reversed_gaps"] if a["reversed"] != HAND[t] else [],
                  "all_draws_printed_values": printed["reversed"], "printed_ties": printed["ties"],
                  "discovery": d["reversed"], "agrees_48": a["reversed"] == HAND[t],
                  "pairs_differing_48_vs_discovery": diff}
        if t == "hotpot":
            out[t]["pooled_rows"] = six["cut"]["content"]["pooled"][t]["reversed"]
    return out


# --------------------------------------------------------------------------------------------------------------------
# w by family and size
# --------------------------------------------------------------------------------------------------------------------

def panel_w() -> dict:
    """{task: {model: w}} of the panel on the discovery draws, as exact as its grid allows (the ladder's panel_grid)."""
    B = json.loads((ROOT / SOURCES["baseline_mixture"]).read_text(encoding="utf-8"))["tasks"]
    out = {}
    for t in DISC3:
        G = la.panel_grid(t)["w"]
        out[t] = {k: (la.on_grid(w, G) or w) for k, w in zip(B[t]["models"], B[t]["w_discovery"])}
    M = json.loads((ROOT / SOURCES["hotpot_mechanism"]).read_text(encoding="utf-8"))["per_model"]
    G = la.panel_grid("hotpot")["w"]
    out["hotpot"] = {k: (la.on_grid(v["w"], G) or v["w"]) for k, v in M.items()}
    return out


def item_w(V: dict, J: dict) -> dict:
    pw = panel_w()
    out = {"panel": {}, "families": {}, "ladder": {}, "same_in_both_readings": True}
    for t in TASKS:
        vals = sorted(pw[t].values(), key=float)
        out["panel"][t] = {"n": len(vals), "min": vals[0], "max": vals[-1], "median": la.median_of(vals)}
        for key in KEYS:
            wc, wu = V["cut"][t][key]["w"]["discovery"], V["uncut"][t][key]["w"]["discovery"]
            if wc != wu:
                out["same_in_both_readings"] = False
            p = la.place(wc, pd.Series(list(pw[t].values())))
            out["ladder"].setdefault(key, {})[t] = {"w": wc, "w_uncut": wu, "below": p["below"], "equal": p["equal"],
                                                    "above": p["above"], "outside": p["outside_panel_range"]}
    for fam, spec in la.FAMILY_LADDERS.items():
        rungs = spec["rungs"]
        blk = {"rungs": [{"key": k, "source": src, "label": la.SIZE_LABEL[k]} for k, src in rungs], "tasks": {}}
        for t in TASKS:
            vals = [V["cut"][t][k]["w"]["discovery"] if src == "ladder" else pw[t][k] for k, src in rungs]
            d = la.direction_of(vals, "w")
            jd = J["families"][fam]["direction"][t]["strict"]["w"]["direction"]
            if d != jd:
                raise SystemExit(f"{fam} {t}: w direction {d!r} here, {jd!r} in ladder.json")
            steps = la.step_words(vals, "w")
            blk["tasks"][t] = {"values": vals, "direction": d, "steps": steps}
        out["families"][fam] = blk
    return out


# --------------------------------------------------------------------------------------------------------------------
# The baseline shift and its long and short parts at 16 tokens
# --------------------------------------------------------------------------------------------------------------------

def item_shift(V: dict, W: dict) -> dict:
    import run_baseline_mixture as rbm
    out = {"split_tokens": SPLIT_TOKENS, "ladder": {}, "panel": {}}
    for r in READINGS:
        for t in TASKS:
            disc, _ = npr.sampler(t)
            d = rbm.decomposition_block(W[r][t], KEYS, SPLIT_TOKENS, disc, True, scores=("strict", "content"))
            grid = Fraction(len(disc) * la.panel_grid(t)["items_per_config"], 100)
            for s in ("strict", "content"):
                pm = d[s]["per_model"]
                for j, key in enumerate(KEYS):
                    acc = V[r][t][key][s]["discovery"]["accuracy_pp"]
                    shift = acc["requested"] - acc["forced"]
                    if abs(float(shift) - pm["shift"][j]) > TOL:
                        raise SystemExit(f"{key} {t} {s} {r}: the decomposition's shift differs from the ladder's")
                    long_ = la.on_grid(pm["part_long"][j], grid)
                    short = la.on_grid(pm["part_short"][j], grid)
                    if long_ is None or short is None or long_ + short != shift:
                        raise SystemExit(f"{key} {t} {s} {r}: the parts are not exact on the grid or do not add up")
                    w16 = la.on_grid(pm["w"][j], grid * 100)
                    out["ladder"].setdefault(r, {}).setdefault(s, {}).setdefault(key, {})[t] = {
                        "shift": shift, "part_long": long_, "part_short": short, "long_share": w16,
                        "R_long": pm["R_long"][j], "F_long": pm["F_long"][j], "R_short": pm["R_short"][j],
                        "F_short": pm["F_short"][j]}
    B = json.loads((ROOT / SOURCES["baseline_mixture"]).read_text(encoding="utf-8"))["decomposition"]
    H = json.loads((ROOT / SOURCES["hotpot_thresholds"]).read_text(encoding="utf-8"))["variants"]["cap16"]
    for s in ("strict", "content"):
        for t in TASKS:
            pm = H[s]["sets"]["discovery"]["per_model"] if t == "hotpot" else B[t]["sets"]["discovery"][s]["per_model"]
            # each panel value is a ratio m / G on the discovery draws' grid (G = draws x items / 100): read it as that
            # exact ratio, as the ladder reads the panel's files (la.on_grid), so that a value exactly halfway prints
            # as its even neighbour; a value off the grid keeps its float
            grid = Fraction(len(npr.sampler(t)[0]) * la.panel_grid(t)["items_per_config"], 100)
            blk = {}
            for q in ("shift", "part_long", "part_short"):
                v = [la.on_grid(x, grid) or x for x in pm[q] if x is not None and math.isfinite(x)]
                blk[q] = {"min": min(v, key=float), "max": max(v, key=float), "median": float(np.median(
                    [float(x) for x in v])), "n": len(v)}
            if B.get(t, {}).get("forced_cap_tokens") not in (None, SPLIT_TOKENS) and t != "hotpot":
                raise SystemExit(f"{t}: the panel's decomposition is not at {SPLIT_TOKENS} tokens")
            blk["source"] = (f"{SOURCES['hotpot_thresholds']}, variants.cap16.{s}.sets.discovery" if t == "hotpot"
                             else f"{SOURCES['baseline_mixture']}, decomposition.{t}.sets.discovery.{s}")
            out["panel"].setdefault(s, {})[t] = blk
    return out


# --------------------------------------------------------------------------------------------------------------------
# Qwen3.5-122B-A10B's unreadable requested replies
# --------------------------------------------------------------------------------------------------------------------

OPTION_RE = re.compile(r"^\([A-Z]\)")
LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
EXPR_RE = re.compile(r"[\d\s.,+\-*/^()=%$:{}\[\]\\]+")
FORMS = ("bare number", "bare option", "bare words", "box")


def reply_form(text: str) -> str:
    """The form of a reply the strict rule cannot read: 'box' when it holds \\boxed (a box the strict rule does not
    accept); otherwise, on its stripped text, a bare option letter in parentheses (with or without the option's text
    after it), a bare number or numeric expression (digits and signs once LaTeX commands such as \\frac are set aside),
    or bare words."""
    if "\\boxed" in text:
        return "box"
    s = text.strip()
    if OPTION_RE.match(s):
        return "bare option"
    core = LATEX_CMD_RE.sub("", s)
    if re.search(r"\d", core) and EXPR_RE.fullmatch(core):
        return "bare number"
    return "bare words"


def shown(s: str, n: int = 50) -> str:
    """A reply as printed in a table: line breaks as ⏎, backticks dropped, cut at n characters."""
    s = str(s).replace("\r", "").replace("\n", "⏎").replace("`", "'")
    return s if len(s) <= n else s[:n] + "…"


def item_qwen(V: dict) -> dict:
    run = (ROOT / RAW_CHOSEN.format(key=QWEN)).read_text(encoding="utf-8").strip()
    out = {"model": QWEN, "run": run, "arm": "direct_answer (zero-shot requested)", "reading": "cut", "tasks": {}}
    for t in TASKS:
        sc = la.load_scored(la.model_paths(QWEN, ROOT)["cut"][t])
        sc = sc[sc["arm"] == "direct_answer"]
        parts = []
        for sh in sorted((ROOT / RAW_SHARDS.format(run=run, task=t)).glob("shard-*.parquet")):
            x = pd.read_parquet(sh, columns=["request_id", "arm", "output_text"])
            parts.append(x[x["arm"] == "direct_answer"][["request_id", "output_text"]])
        raw = pd.concat(parts, ignore_index=True)
        df = sc.merge(raw, on="request_id", how="left", validate="one_to_one")
        if df["output_text"].isna().any():
            raise SystemExit(f"{QWEN} {t}: {int(df['output_text'].isna().sum())} requested rows without raw text")
        bad = df[~df["compliant"].astype(bool)].copy()
        bad["form"] = bad["output_text"].map(reply_form)
        forms = {f_: int((bad["form"] == f_).sum()) for f_ in FORMS}
        bare = bad[bad["form"] != "box"]
        ex = {}
        for f_ in FORMS:
            g = bad[bad["form"] == f_]
            if len(g):
                top = g["output_text"].str.strip().value_counts()
                ex[f_] = [shown(s) for s in top.index[:3]]
        out["tasks"][t] = {
            "rows": int(len(df)), "unreadable": int(len(bad)), "forms": forms,
            "bare": int(len(bare)), "bare_single_line": int((~bare["output_text"].str.strip().str.contains("\n")).sum()),
            "bare_at_most_split_tokens": int((bare["n_output_tokens"] <= SPLIT_TOKENS).sum()),
            "bare_content_correct": int(bare["content_correct"].sum()),
            "bare_tokens_median": float(bare["n_output_tokens"].median()) if len(bare) else None,
            "bare_tokens_max": int(bare["n_output_tokens"].max()) if len(bare) else None,
            "unreadable_cut_at_cap": int(bad["cut_at_panel_cap"].astype(bool).sum()) if "cut_at_panel_cap" in bad else 0,
            "unreadable_content_correct": int(bad["content_correct"].sum()),
            "unreadable_strict_correct": int(bad["strict_correct"].sum()),
            "readable_discovery_pct": V["cut"][t][QWEN]["readable_pct"]["requested"]["discovery"],
            "examples": ex}
        del sc, df, raw
    return out


# --------------------------------------------------------------------------------------------------------------------
# Macros
# --------------------------------------------------------------------------------------------------------------------

def mnum(x, d: int = 1) -> str:
    """A macro value: fnum's half-even text on the exact value; a value that rounds to zero without its sign; minus
    as $-$; thousands as {,} for whole numbers."""
    if x is None:
        return "--"
    if d == 0 and isinstance(x, int):
        return f"{x:,}".replace(",", "{,}")
    s = la.fnum(x, d)
    if s.startswith("-") and set(s[1:]) <= set("0."):
        s = s[1:]
    return s.replace("-", "$-$")


def znum(x, d: int = 1) -> str:
    """fnum's text, a value that rounds to zero printed without its sign, as the macros print it (mnum)."""
    s = la.fnum(x, d)
    return s[1:] if s.startswith("-") and set(s[1:]) <= set("0.") else s


def pct0(k: int, n: int) -> str:
    return mnum(Fraction(100 * k, n), 0) if n else "--"


def macros(R: dict) -> dict:
    m: dict = {}

    def put(k, v):
        if not k.isalpha() or not k.startswith(("lad", "comb")):
            raise ValueError(f"macro name {k!r}: letters only, prefix lad or comb")
        if k in m:
            raise ValueError(f"macro {k} set twice")
        m[k] = v

    for (s, r), V_ in VARIANT.items():
        six = R["reversals"]["six"][r][s]["discovery"]
        big = R["reversals"]["big"][r][s]
        cb = R["reversals"]["combined"][r][s]
        for t in list(TASKS) + ["pooled"]:
            tag = "Pooled" if t == "pooled" else T_TAG[t]
            x = six[t]
            put(f"ladRev{tag}{V_}", mnum(x["reversed"], 0))
            put(f"ladPairs{tag}{V_}", mnum(x["pairs"], 0))
            put(f"ladRevPct{tag}{V_}", pct0(x["reversed"], x["pairs"]))
            put(f"ladTies{tag}{V_}", mnum(x["ties"], 0))
            b = big[t]
            put(f"ladBigRev{tag}{V_}", mnum(b["reversed"], 0))
            put(f"ladBigPairs{tag}{V_}", mnum(b["pairs"], 0))
            put(f"ladBigTies{tag}{V_}", mnum(b["ties"], 0))
            c = cb[t]
            put(f"combRev{tag}{V_}", mnum(c["reversed"], 0))
            put(f"combPairs{tag}{V_}", mnum(c["pairs"], 0))
            put(f"combRevPct{tag}{V_}", pct0(c["reversed"], c["pairs"]))
            put(f"combRevShare{tag}{V_}", mnum(Fraction(c["reversed"], c["pairs"]), 3))
            put(f"combTies{tag}{V_}", mnum(c["ties"], 0))
            ci = c["model_bootstrap"]["ci95"]
            put(f"combRevBootLo{tag}{V_}", mnum(ci[0], 2))
            put(f"combRevBootHi{tag}{V_}", mnum(ci[1], 2))
            put(f"combM{tag}{V_}", mnum(c["union"] if t == "pooled" else c["models"], 0))
            # the pairs whose two gaps both exceed 1, 2, 5, 10 pp, and the reversed among
            # them (src/robustness.pair_reversals by_threshold, as \aeHqGapFive...)
            for th in GAP_THRESHOLDS:
                w = GAP_WORD[th]
                for stem, blk in (("ladGap", x), ("ladBigGap", b), ("combGap", c)):
                    y = by_gap(blk, th)
                    put(f"{stem}{w}Pairs{tag}{V_}", mnum(y["pairs"], 0))
                    put(f"{stem}{w}Rev{tag}{V_}", mnum(y["reversed"], 0))
    for s in SCORES:
        V_ = VARIANT[(s, "cut")]
        for t in list(TASKS) + ["pooled"]:
            tag = "Pooled" if t == "pooled" else T_TAG[t]
            p = R["reversals"]["panel"][s][t]
            put(f"combPanelRev{tag}{V_}", mnum(p["reversed"], 0))
            put(f"combPanelPairs{tag}{V_}", mnum(p["pairs"], 0))
            put(f"combPanelRevPct{tag}{V_}", pct0(p["reversed"], p["pairs"]))
            put(f"combPanelRevShare{tag}{V_}", mnum(Fraction(p["reversed"], p["pairs"]), 3))
            ci = p["model_bootstrap"]["ci95"]
            put(f"combPanelRevBootLo{tag}{V_}", mnum(ci[0], 2))
            put(f"combPanelRevBootHi{tag}{V_}", mnum(ci[1], 2))
            put(f"combPanelM{tag}{V_}", mnum(p["union"] if t == "pooled" else p["models"], 0))
    Wd = R["w"]
    for t in TASKS:
        tag = T_TAG[t]
        for key in KEYS:
            put(f"ladW{MODEL_TAG[key]}{tag}", mnum(Wd["ladder"][key][t]["w"], 3))
        for fam, f in Wd["families"].items():
            for rg, v in zip(f["rungs"], f["tasks"][t]["values"]):
                if rg["source"] == "panel":
                    put(f"ladW{MODEL_TAG[rg['key']]}{tag}", mnum(v, 3))
            put(f"ladWDir{FAM_TAG[fam]}{tag}", f["tasks"][t]["direction"])
        p = Wd["panel"][t]
        put(f"ladWPanelMin{tag}", mnum(p["min"], 3))
        put(f"ladWPanelMax{tag}", mnum(p["max"], 3))
        put(f"ladWPanelMedian{tag}", mnum(p["median"], 3))
    S = R["shift"]
    for (s, r), V_ in VARIANT.items():
        for key in KEYS:
            for t in TASKS:
                x = S["ladder"][r][s][key][t]
                stem = f"{MODEL_TAG[key]}{T_TAG[t]}{V_}"
                put(f"ladShift{stem}", mnum(x["shift"], 1))
                put(f"ladShiftLong{stem}", mnum(x["part_long"], 1))
                put(f"ladShiftShort{stem}", mnum(x["part_short"], 1))
                if s == "content":
                    put(f"ladShiftLongShare{stem}", mnum(x["long_share"], 3))
    for s in SCORES:
        V_ = VARIANT[(s, "cut")]
        for t in TASKS:
            p = S["panel"][s][t]
            for q, qn in (("shift", ""), ("part_long", "Long"), ("part_short", "Short")):
                put(f"ladShift{qn}PanelMin{T_TAG[t]}{V_}", mnum(p[q]["min"], 1))
                put(f"ladShift{qn}PanelMax{T_TAG[t]}{V_}", mnum(p[q]["max"], 1))
    Q = R["qwen"]["tasks"]
    for t in TASKS:
        x = Q[t]
        tag = T_TAG[t]
        put(f"ladQwenReqRows{tag}", mnum(x["rows"], 0))
        put(f"ladQwenReqUnread{tag}", mnum(x["unreadable"], 0))
        put(f"ladQwenReqUnreadPct{tag}", mnum(Fraction(100 * x["unreadable"], x["rows"]), 1))
        put(f"ladQwenReqBare{tag}", mnum(x["bare"], 0))
        put(f"ladQwenReqBarePct{tag}", mnum(Fraction(100 * x["bare"], x["unreadable"]), 1) if x["unreadable"] else "--")
        put(f"ladQwenReqBareTokens{tag}", mnum(Fraction(x["bare_tokens_median"]).limit_denominator(2), 1)
            if x["bare_tokens_median"] is not None else "--")
        put(f"ladQwenReqUnreadContentPct{tag}",
            mnum(Fraction(100 * x["unreadable_content_correct"], x["unreadable"]), 1) if x["unreadable"] else "--")
    return m


# --------------------------------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------------------------------

def of(x: dict) -> str:
    return f"{x['reversed']} of {x['pairs']:,}"


def share_ci(x: dict) -> str:
    ci = x["model_bootstrap"]["ci95"]
    return (f"{of(x)} ({la.fnum(Fraction(100 * x['reversed'], x['pairs']), 0)}%, {la.fnum(ci[0], 2)} to "
            f"{la.fnum(ci[1], 2)})")


def pair_names(ps: list) -> str:
    return "; ".join(f"{nm(a)} and {nm(b)}" for a, b in ps) or "none"


def counts_line(blk: dict, tasks=TASKS, with_pooled=True, pct=True) -> str:
    parts = [f"{TNAME[t]} {of(blk[t])}" for t in tasks]
    s = ", ".join(parts)
    if with_pooled:
        p = blk["pooled"]
        s += (f"; pooled over the first three, {of(p)}"
              + (f" ({la.fnum(Fraction(100 * p['reversed'], p['pairs']), 0)}%)" if pct else ""))
    return s


def note_section(R: dict) -> list[str]:
    """The section of ladder_note.md (cut reading; content first, strict beside it). Every number here is recounted by
    scripts/verify_ladder_analysis.py."""
    rv = R["reversals"]
    six_c, six_s = rv["six"]["cut"]["content"]["discovery"], rv["six"]["cut"]["strict"]["discovery"]
    L = [SECTION, "",
         "From `ladder_orderings.md` (script `scripts/ladder_orderings.py`). Cut reading. A "
         "pair is reversed when CoT − requested orders it one way and CoT − forced the other; a pair tied under either "
         "gain is not reversed and stays among the pairs, as the panel counts them. Model means over the discovery "
         "draws, HotpotQA content included (the basis of the paper's HotpotQA count).", ""]
    L.append(f"- Six ladder models, content: {counts_line(six_c)}. Strict: {counts_line(six_s)}.")
    # the pairs whose two gaps both exceed 1, 2 and 5 pp,
    # and the reversed pairs of the six that rest on a gap of SUB_PP pp or less, named
    L.append("- " + gap_line("Six ladder models, content", six_c))
    subs = sub_pp_pairs(six_c)
    L.append(f"- Reversed pairs of the six whose gap under either gain is {SUB_PP:g} pp or less (content, discovery "
             "draws; the gap between the two models under CoT − requested, then under CoT − forced, pp): "
             + ("; ".join(f"{TNAME[t]}, {nm(g['pair'][0])} and {nm(g['pair'][1])} ({g['printed'][0]}, "
                          f"{g['printed'][1]})" for t, g in subs) if subs else "none")
             + ". A count that includes them turns on gaps that small.")
    h = R["hand_check"]
    diffs = []
    for t in TASKS:
        for d in h[t]["pairs_differing_48_vs_discovery"]:
            a, b = d["pair"]
            diffs.append(f"{TNAME[t]}, {nm(a)} and {nm(b)}, " + ("reversed" if d["reversed_on_discovery"] else
                                                                   "not reversed") + " on the discovery draws only "
                         f"(CoT − requested {la.fnum(d['all_draws']['cot_minus_requested'][0], 2)} and "
                         f"{la.fnum(d['all_draws']['cot_minus_requested'][1], 2)} on the 48 draws, "
                         f"{la.fnum(d['discovery']['cot_minus_requested'][0], 2)} and "
                         f"{la.fnum(d['discovery']['cot_minus_requested'][1], 2)} on the discovery draws; CoT − forced "
                         f"{la.fnum(d['discovery']['cot_minus_forced'][0], 2)} and "
                         f"{la.fnum(d['discovery']['cot_minus_forced'][1], 2)})")
    on48 = [f"{TNAME[t]} reverses {len(h[t]['all_draws_reversed_gaps'])}: " + ", ".join(
        f"{nm(g['pair'][0])} and {nm(g['pair'][1])} ({g['printed'][0]}, {g['printed'][1]})"
        for g in h[t]["all_draws_reversed_gaps"]) for t in TASKS if h[t]["all_draws_reversed_gaps"]]
    L.append("- The hand counts (content, 48-draw means): " + ", ".join(
        f"{TNAME[t]} {h[t]['hand']} of {h[t]['pairs']} ({h[t]['all_draws']} here)" for t in TASKS)
        + ". On the discovery draws: " + ", ".join(f"{TNAME[t]} {h[t]['discovery']}" for t in TASKS) + ". "
        + ("The pairs that differ: " + "; ".join(diffs) + "." if diffs else "No pair differs between the two bases.")
        + (" Where the hand count and the 48-draw count differ, the pairs reversed on the 48 draws (gap under CoT "
           "− requested, then under CoT − forced, pp): " + "; ".join(on48) + "." if on48 else ""))
    for s, lab in (("content", "content"), ("strict", "strict")):
        c = rv["combined"]["cut"][s]
        L.append(f"- Combined panel of the 29 panel models and the six ({c['gsm8k']['models']} models; "
                 f"{c['bbh']['models']} on BIG-Bench Hard), {lab}, reversed pairs (share, model bootstrap 95% "
                 "interval): " + ", ".join(f"{TNAME[t]} {share_ci(c[t])}" for t in TASKS)
                 + f"; pooled over the first three, {share_ci(c['pooled'])}.")
        if s == "content":
            L.append("- " + gap_line("Combined panel, content", c))
        p = rv["panel"][s]
        L.append(f"- The 29 panel models ({p['bbh']['models']} on BIG-Bench Hard), {lab}, the same: "
                 + ", ".join(f"{TNAME[t]} {share_ci(p[t])}" for t in TASKS)
                 + f"; pooled over the first three, {share_ci(p['pooled'])}.")
    bc, bs = rv["big"]["cut"]["content"], rv["big"]["cut"]["strict"]
    L.append("- Models of 70B and above (" + ", ".join(nm(k) for k in BIG) + "), content: "
             + counts_line(bc, pct=False) + ". Strict: " + counts_line(bs, pct=False) + ". " + R["big_size_note"])
    L.append("- " + gap_line("Models of 70B and above, content", bc))
    Wd = R["w"]
    fam_parts = []
    for fam, f in Wd["families"].items():
        fam_parts.append(f"{fam} (" + " → ".join(r["label"] for r in f["rungs"]) + "): " + ", ".join(
            f"{TNAME[t]} " + " → ".join(la.fnum(v, 3) for v in f["tasks"][t]["values"])
            + f" ({f['tasks'][t]['direction']})" for t in TASKS))
    L.append("- w by family, discovery draws: " + "; ".join(fam_parts) + ". " + nm(KEYS[-1]) + ": " + ", ".join(
        f"{TNAME[t]} {la.fnum(Wd['ladder'][KEYS[-1]][t]['w'], 3)}" for t in TASKS) + ". The panel's range: "
        + ", ".join(f"{TNAME[t]} {la.fnum(Wd['panel'][t]['min'], 3)} to {la.fnum(Wd['panel'][t]['max'], 3)}"
                    for t in TASKS) + ".")
    S = R["shift"]["ladder"]["cut"]
    for s in ("content", "strict"):
        L.append(f"- Baseline shift (requested − forced), {s}, discovery draws, with its part from rows whose "
                 f"requested output is longer than {SPLIT_TOKENS} tokens and its part from the other rows: " + "; ".join(
                     f"{nm(k)}: " + ", ".join(
                         f"{TNAME[t]} {la.fnum(S[s][k][t]['shift'], 1)} (long {la.fnum(S[s][k][t]['part_long'], 1)}, "
                         f"short {la.fnum(S[s][k][t]['part_short'], 1)})" for t in TASKS) for k in KEYS) + ".")
    Q = R["qwen"]["tasks"]
    L.append(f"- {nm(QWEN)}, zero-shot requested arm, all rows: replies the strict rule cannot read " + ", ".join(
        f"{TNAME[t]} {Q[t]['unreadable']:,} of {Q[t]['rows']:,}" for t in TASKS) + "; of these, the bare answer "
        "with no box " + ", ".join(f"{TNAME[t]} {Q[t]['bare']:,}" for t in TASKS) + " (median length "
        + ", ".join(f"{TNAME[t]} {la.fnum(Fraction(Q[t]['bare_tokens_median']).limit_denominator(2), 0)}"
                    for t in TASKS) + " tokens). Methods sentence: `ladder_orderings.md`, section 7.")
    u6 = rv["six"]["uncut"]["content"]["discovery"]
    uc = rv["combined"]["uncut"]["content"]
    L.append(f"- Uncut reading (Supplementary Information), content: six ladder models {counts_line(u6)}; combined "
             f"panel " + ", ".join(f"{TNAME[t]} {share_ci(uc[t])}" for t in TASKS)
             + f"; pooled over the first three, {share_ci(uc['pooled'])}.")
    return L


def put_section(path: Path, lines: list[str], clean: bool = False) -> None:
    """Replace this script's section of ladder_note.md (from its heading to the next '## ' heading or the end), or put
    it before '## Footnotes to carry' (at the end when that heading is absent)."""
    sec = "\n".join(lines).rstrip("\n")
    if clean:
        sec = la.npl.npt.strip_links(sec)
    new = sec.split("\n")
    eol = "\r\n" if b"\r\n" in path.read_bytes() else "\n"          # keep the file's line endings
    rows = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    start = next((i for i, ln in enumerate(rows) if ln.startswith(SECTION)), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(rows)) if rows[i].startswith("## ")), len(rows))
        rows = rows[:start] + new + ([""] if end < len(rows) else []) + rows[end:]
    else:
        k = next((i for i, ln in enumerate(rows) if ln.startswith("## Footnotes to carry")), None)
        rows = rows[:k] + new + [""] + rows[k:] if k is not None else rows + [""] + new
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline=eol)


def md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


def gap_table(rv: dict, reading: str) -> list[str]:
    """Rows: group, score and threshold; columns: the benchmarks and the pooled value, 'reversed of pairs'."""
    head = ["Group", "Score", "Both gaps above"] + [TNAME[t] for t in TASKS] + ["Pooled, first three"]
    rows = []
    for grp, lab in (("six", "Six ladder models"), ("big", "70B and above"), ("combined", "35 models")):
        for s in SCORES:
            blk = rv[grp][reading][s]["discovery"] if grp == "six" else rv[grp][reading][s]
            for th in GAP_THRESHOLDS:
                rows.append([lab, s, f"{th:g} pp"] + [of(by_gap(blk[t], th)) for t in list(TASKS) + ["pooled"]])
    return md_table(head, rows)


def write_md(R: dict, path: Path) -> None:
    rv = R["reversals"]
    L = [f"# Orderings on the size ladder ({R['created']})", "",
         f"Script `{REQUEST}`; values in "
         f"`{STEM}.json`; independent recount in `scripts/verify_ladder_analysis.py` (`verify.out`). Scored rows read "
         f"by {R['readers']}. Numbers in pp; w and shares as proportions; every number rounded half-even on its exact "
         "value where it is held exactly.", "",
         "Basis: model means over the 24 discovery draws (a configuration drawn twice counts twice), HotpotQA content "
         "included: that is the basis of the paper's HotpotQA count (`\\aeHqReversedContent`, results section: \"with "
         "benefits averaged over the discovery draws\"). The ladder's placements read HotpotQA content on pooled rows; "
         "the count on that basis is in section 2. Content scoring first, strict beside it. Outputs cut at the panel's "
         "caps; the uncut reading is section 8.", "",
         "Reversal: a pair of models ordered one way by CoT − requested and the other by CoT − forced. A pair tied "
         "under either gain (|difference| ≤ 1e-9 pp) is neither reversed nor kept and stays among the pairs "
         "(`src/robustness.pair_reversals`, the panel's counting). Model bootstrap: `src/decision_directions."
         "reversal_uncertainty`, models drawn with replacement, a pair of a model with its own copy left out, "
         f"{N_BOOT:,} resamples, percentile 95% interval of the share; pooled over GSM8K, MATH-500 and BIG-Bench Hard, "
         "one draw of the union of models serves the three benchmarks and the share is reversed pairs over pairs, both "
         "summed.", "", "## Checks", ""]
    L += [f"- {c}" for c in R["checks"]]
    # 1. six
    L += ["", "## 1. Reversals among the six ladder models", ""]
    head = ["Benchmark", "Content: reversed of pairs (%)", "Ties", "Strict: reversed of pairs (%)", "Ties",
            "Reversed pairs, content"]
    rows = []
    for t in list(TASKS) + ["pooled"]:
        c, s = (rv["six"]["cut"][sc]["discovery"][t] for sc in ("content", "strict"))
        rows.append([TNAME.get(t, "Pooled, first three"), f"{of(c)} ({la.fnum(Fraction(100 * c['reversed'], c['pairs']), 0)})",
                     str(c["ties"]), f"{of(s)} ({la.fnum(Fraction(100 * s['reversed'], s['pairs']), 0)})", str(s["ties"]),
                     pair_names(c.get("reversed_pairs", [])) if t != "pooled" else "–"])
    L += md_table(head, rows)
    tied = [(t, rv["six"]["cut"][sc]["discovery"][t]["tied_pairs"], sc) for t in TASKS for sc in SCORES
            if rv["six"]["cut"][sc]["discovery"][t]["tied_pairs"]]
    L += ["", "Tied pairs: " + ("; ".join(f"{TNAME[t]} {sc}: {pair_names(p)}" for t, p, sc in tied) if tied else "none")
          + "."]
    L += ["", "Pairs whose two gaps both exceed a threshold (`by_threshold` of `src/robustness.pair_reversals`, as the "
          "panel's `\\ae<T>GapFive...`): reversed of those pairs. Six ladder models, 70B and above, combined panel of 35; "
          "content and strict, cut reading, discovery draws. The group of 70B and above is that of section 4. "
          + R["big_size_note"], ""]
    L += gap_table(rv, "cut")
    subs = sub_pp_pairs(rv["six"]["cut"]["content"]["discovery"])
    L += ["", f"Reversed pairs of the six whose gap under either gain is {SUB_PP:g} pp or less (content; the gap between "
          "the two models under CoT − requested, then under CoT − forced, pp; exact, rounded half-even):", ""]
    L += [f"- {TNAME[t]}: {nm(g['pair'][0])} and {nm(g['pair'][1])} ({g['printed'][0]}, {g['printed'][1]})"
          for t, g in subs] or ["- none"]
    L += ["", "Every reversed pair of the six, with its two gaps (content):", ""]
    L += md_table(["Benchmark", "Pair", "Gap, CoT − requested", "Gap, CoT − forced"],
                  [[TNAME[t], f"{nm(g['pair'][0])} and {nm(g['pair'][1])}", g["printed"][0], g["printed"][1]]
                   for t in TASKS for g in rv["six"]["cut"]["content"]["discovery"][t]["reversed_gaps"]])
    # 2. hand counts
    h = R["hand_check"]
    L += ["", "## 2. Hand counts", "",
          "Counted by hand from the 48-draw columns of `ladder_note.md` (content). Counted here on the same 48-draw "
          "means (exact), on those means rounded to one decimal as the note prints them (a printed tie counts as not "
          "reversed), and on the discovery draws (the paper's basis).", ""]
    head = ["Benchmark", "By hand", "48 draws, exact", "48 draws, printed values", "Printed ties", "Discovery draws"]
    L += md_table(head, [[TNAME[t], f"{h[t]['hand']} of {h[t]['pairs']}", str(h[t]["all_draws"]),
                          str(h[t]["all_draws_printed_values"]), str(h[t]["printed_ties"]), str(h[t]["discovery"])]
                         for t in TASKS])
    L += [""]
    for t in TASKS:
        x = h[t]
        if not x["pairs_differing_48_vs_discovery"]:
            continue
        for d in x["pairs_differing_48_vs_discovery"]:
            a, b = d["pair"]
            w48 = "reversed" if d["reversed_on_48_draws"] else "not reversed"
            wd = "reversed" if d["reversed_on_discovery"] else "not reversed"
            L.append(f"- {TNAME[t]}, {nm(a)} and {nm(b)}: {w48} on the 48 draws (CoT − requested "
                     f"{la.fnum(d['all_draws']['cot_minus_requested'][0], 2)} and "
                     f"{la.fnum(d['all_draws']['cot_minus_requested'][1], 2)}; CoT − forced "
                     f"{la.fnum(d['all_draws']['cot_minus_forced'][0], 2)} and "
                     f"{la.fnum(d['all_draws']['cot_minus_forced'][1], 2)}), {wd} on the discovery draws "
                     f"({la.fnum(d['discovery']['cot_minus_requested'][0], 2)} and "
                     f"{la.fnum(d['discovery']['cot_minus_requested'][1], 2)}; "
                     f"{la.fnum(d['discovery']['cot_minus_forced'][0], 2)} and "
                     f"{la.fnum(d['discovery']['cot_minus_forced'][1], 2)}).")
    for t in TASKS:
        if h[t]["all_draws_reversed_gaps"]:
            L.append(f"- {TNAME[t]}: the hand count is {h[t]['hand']}, the 48-draw means reverse "
                     f"{h[t]['all_draws']}: " + "; ".join(
                         f"{nm(g['pair'][0])} and {nm(g['pair'][1])} (gaps {g['printed'][0]} and {g['printed'][1]})"
                         for g in h[t]["all_draws_reversed_gaps"]) + ".")
    L.append(f"- HotpotQA content on pooled rows (the ladder's placement basis): {h['hotpot']['pooled_rows']} of 15.")
    # tied pairs of the six on every basis counted: a tie is counted among the pairs and
    # neither reversed nor kept
    basis_word = {"discovery": "discovery draws", "all_draws": "48 draws", "pooled": "pooled rows"}
    tie_parts = [f"{TNAME[t]}, {s_}, {basis_word[bs]}: " + pair_names(blk[t]["tied_pairs"])
                 for s_ in SCORES for bs in BASES for t in TASKS
                 for blk in [rv["six"]["cut"][s_][bs]] if blk[t]["ties"]]
    L.append("- Tied pairs of the six (cut reading; counted among the pairs, neither reversed nor kept): "
             + ("; ".join(tie_parts) if tie_parts else "none") + ".")
    # 3. combined
    L += ["", "## 3. The combined panel of 35 models", "",
          "The 29 panel models and the six ladder models (34 on BIG-Bench Hard, where the panel has 28). Beside it, the "
          "29-model value: for GSM8K, MATH-500 and BIG-Bench Hard the paper's (`\\ddir<T>Rev...`, its interval "
          f"reproduced from its own seed); for HotpotQA and the pooled value computed here with the same resampling.", ""]
    # a tied pair (|difference| <= 1e-9 pp under either gain) is neither reversed nor kept and stays among the pairs;
    # the ties are printed beside the counts
    for s in SCORES:
        head = ["Benchmark", f"35 models, {s}: reversed of pairs", "Ties", "Share", "Model bootstrap 95%",
                "Family bootstrap 95%", "Reversed pairs with a ladder model", f"29 models, {s}", "Ties", "Share",
                "Model bootstrap 95%"]
        rows = []
        for t in list(TASKS) + ["pooled"]:
            c, p = rv["combined"]["cut"][s][t], rv["panel"][s][t]
            fb = c.get("family_bootstrap", {}).get("ci95")
            rows.append([TNAME.get(t, "Pooled, first three"), of(c), str(c["ties"]),
                         la.fnum(Fraction(c["reversed"], c["pairs"]), 3),
                         f"{la.fnum(c['model_bootstrap']['ci95'][0], 2)} to {la.fnum(c['model_bootstrap']['ci95'][1], 2)}",
                         f"{la.fnum(fb[0], 2)} to {la.fnum(fb[1], 2)}" if fb else "–",
                         str(c["with_ladder"]) if "with_ladder" in c else "–", of(p), str(p.get("ties", "–")),
                         la.fnum(Fraction(p["reversed"], p["pairs"]), 3),
                         f"{la.fnum(p['model_bootstrap']['ci95'][0], 2)} to {la.fnum(p['model_bootstrap']['ci95'][1], 2)}"])
        L += md_table(head, rows) + [""]
        tied = [(t, rv["combined"]["cut"][s][t].get("tied_pairs") or []) for t in TASKS]
        tied = [(t, tp) for t, tp in tied if tp]
        L += [f"Tied pairs, 35 models, {s}: " + ("; ".join(f"{TNAME[t]}: {pair_names(tp)}" for t, tp in tied)
                                                  if tied else "none") + ".", ""]
    # 4. big
    L += ["## 4. Models of 70B and above", "",
          ", ".join(nm(k) for k in BIG) + ": 6 pairs per benchmark. Counts only. " + R["big_size_note"], ""]
    head = ["Benchmark", "Content: reversed of pairs", "Ties", "Strict: reversed of pairs", "Ties", "Reversed pairs, content"]
    rows = []
    for t in list(TASKS) + ["pooled"]:
        c, s_ = rv["big"]["cut"]["content"][t], rv["big"]["cut"]["strict"][t]
        rows.append([TNAME.get(t, "Pooled, first three"), of(c), str(c["ties"]), of(s_), str(s_["ties"]),
                     pair_names(c.get("reversed_pairs", [])) if t != "pooled" else "–"])
    L += md_table(head, rows)
    # 5. w
    Wd = R["w"]
    L += ["", "## 5. w by family and size", "",
          "w: share of requested outputs longer than the forced cap (16 tokens; 64 on HotpotQA), per configuration, "
          "averaged over the discovery draws; it reads lengths only, so it is the same for both scores"
          + (" and, here, for both readings." if Wd["same_in_both_readings"] else "; it differs between the readings "
             "(section 8).") + " Direction rung to rung by the ladder's rule (a step of 0.001 or more is up or down).",
          ""]
    head = ["Family", "Rung", "Source"] + [TNAME[t] for t in TASKS]
    rows = []
    for fam, f in Wd["families"].items():
        for i, rg in enumerate(f["rungs"]):
            rows.append([fam if i == 0 else "", rg["label"], rg["source"]]
                        + [la.fnum(f["tasks"][t]["values"][i], 3) for t in TASKS])
        rows.append(["", "Over size", ""] + [f["tasks"][t]["direction"] for t in TASKS])
    k = KEYS[-1]
    rows.append(["none", nm(k), "ladder"] + [la.fnum(Wd["ladder"][k][t]["w"], 3) for t in TASKS])
    rows.append(["Panel", "min to max (median)", f"{Wd['panel']['gsm8k']['n']} models"]
                + [f"{la.fnum(Wd['panel'][t]['min'], 3)} to {la.fnum(Wd['panel'][t]['max'], 3)} "
                   f"({la.fnum(Wd['panel'][t]['median'], 3)})" for t in TASKS])
    L += md_table(head, rows)
    for side, word in (("above", "above the panel's maximum"), ("below", "below the panel's minimum")):
        xs = [f"{nm(k)} on {TNAME[t]} ({la.fnum(Wd['ladder'][k][t]['w'], 3)})" for k in KEYS for t in TASKS
              if Wd["ladder"][k][t]["outside"] == side]
        L += ["", f"Ladder values of w {word}: " + ("; ".join(xs) if xs else "none") + "."]
    # 6. shift
    S = R["shift"]
    L += ["", "## 6. The baseline shift and its two parts", "",
          f"Requested − forced accuracy on the discovery draws, split exactly: the part from rows whose requested "
          f"output is longer than {SPLIT_TOKENS} generated tokens and the part from the others, each the sum of "
          "requested − forced over its rows divided by all the model's rows (`scripts/run_baseline_mixture."
          "decomposition_block`, as for the panel; on HotpotQA the panel's cap16 variant). Long share: the share of "
          f"requested outputs longer than {SPLIT_TOKENS} tokens.", ""]
    L += ["Each cell: shift (long part, short part) [long share]. The parts add to the shift exactly; each is "
          "rounded on its own.", ""]
    for s in SCORES:
        head = ["Model"] + [TNAME[t] for t in TASKS]
        rows = [[nm(k)] + [f"{znum(x['shift'], 1)} ({znum(x['part_long'], 1)}, {znum(x['part_short'], 1)}) "
                           f"[{la.fnum(x['long_share'], 3)}]"
                           for x in (S['ladder']['cut'][s][k][t] for t in TASKS)] for k in KEYS]
        rows.append(["Panel, min to max"] + [
            f"{znum(S['panel'][s][t]['shift']['min'], 1)} to {znum(S['panel'][s][t]['shift']['max'], 1)}; long "
            f"{znum(S['panel'][s][t]['part_long']['min'], 1)} to {znum(S['panel'][s][t]['part_long']['max'], 1)}; "
            f"short {znum(S['panel'][s][t]['part_short']['min'], 1)} to "
            f"{znum(S['panel'][s][t]['part_short']['max'], 1)}" for t in TASKS])
        L += [f"{s.capitalize()}:", ""] + md_table(head, rows) + [""]
    L += ["Panel sources: " + "; ".join(f"{TNAME[t]} `{S['panel']['content'][t]['source']}`" for t in TASKS) + "."]
    # 7. Qwen
    Q = R["qwen"]
    L += ["", f"## 7. {nm(QWEN)}: its unreadable requested replies", "",
          f"Zero-shot requested arm, all rows, cut reading; raw text from `data/d2_out/ladder2/{Q['run']}/`. Unreadable: "
          "the strict scorer's `compliant` flag is false. Forms: a box the strict rule does not accept, or no box at all: "
          "a bare option letter in parentheses (with or without the option's text), a bare number or numeric expression "
          "(digits and signs once LaTeX commands such as `\\frac` are set aside), or bare words.", ""]
    head = ["Benchmark", "Rows", "Unreadable", "No box", "Bare number", "Bare option", "Bare words", "Box",
            "Median tokens, no box", "No box, content-correct", "Readable, discovery draws (%)"]
    rows = []
    for t in TASKS:
        x = Q["tasks"][t]
        fo = x["forms"]
        rows.append([TNAME[t], f"{x['rows']:,}", f"{x['unreadable']:,}", f"{x['bare']:,}",
                     f"{fo['bare number']:,}", f"{fo['bare option']:,}", f"{fo['bare words']:,}", f"{fo['box']:,}",
                     la.fnum(Fraction(x["bare_tokens_median"]).limit_denominator(2), 1),
                     f"{x['bare_content_correct']:,}", la.fnum(x["readable_discovery_pct"], 1)])
    L += md_table(head, rows)
    L += ["", "Most frequent replies of each form (line breaks shown as ⏎):", ""]
    for t, x in Q["tasks"].items():
        L.append(f"- {TNAME[t]}: " + "; ".join(f"{f_}: " + ", ".join(f"`{e}`" for e in ex)
                                               for f_, ex in x["examples"].items()) + ".")
    L += ["", "One sentence for Methods: " + R["qwen_sentence"]]
    # 8. uncut
    L += ["", "## 8. Uncut reading (Supplementary Information)", "",
          "The same counts with the ladder models' outputs read uncut (the raised caps; the panel's values do not "
          "change). The column of 70B and above is the group of section 4. Ties: tied pairs among the 35 models, content and strict, counted among the pairs and neither "
          "reversed nor kept.", ""]
    head = ["Benchmark", "Six, content", "Six, strict", "70B and above, content", "35 models, content (share; model "
            "bootstrap 95%)", "35 models, strict (share; model bootstrap 95%)", "Ties, 35 models (content, strict)"]
    rows = []
    for t in list(TASKS) + ["pooled"]:
        a, b = rv["six"]["uncut"]["content"]["discovery"][t], rv["six"]["uncut"]["strict"]["discovery"][t]
        c = rv["big"]["uncut"]["content"][t]
        d, e = rv["combined"]["uncut"]["content"][t], rv["combined"]["uncut"]["strict"][t]
        rows.append([TNAME.get(t, "Pooled, first three"), of(a), of(b), of(c), share_ci(d), share_ci(e),
                     f"{d['ties']}, {e['ties']}"])
    L += md_table(head, rows)
    ch = []
    for sc in SCORES:
        for t in TASKS:
            a = {tuple(p) for p in rv["six"]["cut"][sc]["discovery"][t]["reversed_pairs"]}
            b = {tuple(p) for p in rv["six"]["uncut"][sc]["discovery"][t]["reversed_pairs"]}
            if a != b:
                ch.append(f"{TNAME[t]} {sc}: reversed only cut {pair_names(sorted(a - b))}; only uncut "
                          f"{pair_names(sorted(b - a))}")
    L += ["", "Pairs of ladder models whose reversal differs between the readings: " + ("; ".join(ch) if ch else "none")
          + "."]
    head = ["Model"] + [TNAME[t] for t in TASKS]
    for s in SCORES:
        L += ["", f"Baseline shift, uncut, {s}: shift (long part, short part):", ""] + md_table(head, [
            [nm(k)] + [f"{znum(x['shift'], 1)} ({znum(x['part_long'], 1)}, {znum(x['part_short'], 1)})"
                       for x in (S["ladder"]["uncut"][s][k][t] for t in TASKS)] for k in KEYS])
    if not Wd["same_in_both_readings"]:
        L += ["", "w uncut: " + "; ".join(f"{nm(k)} " + ", ".join(
            f"{TNAME[t]} {la.fnum(Wd['ladder'][k][t]['w_uncut'], 3)}" for t in TASKS) for k in KEYS) + "."]
    L += ["", "## Macros", "", f"{len(R['macros'])} macros, prefixes `lad` and `comb` (`paper/macros_ladder_orderings.py`). "
          "Content and cut carry no suffix; Strict, Uncut and StrictUncut mark the others. <T>: Gsm, Math, Bbh, Hq, "
          "Pooled (the first three). Names: ladRev<T>, ladPairs<T>, ladRevPct<T>, ladTies<T> (six); ladBigRev<T>, "
          "ladBigPairs<T>, ladBigTies<T> (70B and above); ladGap<G>Pairs<T>, ladGap<G>Rev<T>, ladBigGap<G>Pairs<T>, "
          "ladBigGap<G>Rev<T>, combGap<G>Pairs<T>, combGap<G>Rev<T> (pairs whose two gaps both exceed <G> = One, Two, "
          "Five, Ten pp, and the reversed among them); combRev<T>, combPairs<T>, combRevPct<T>, combRevShare<T>, "
          "combRevBootLo<T>, combRevBootHi<T>, combTies<T>, combM<T> (35 models) and combPanel... (29 models); "
          "ladW<Model><T>, ladWDir<Family><T>, ladWPanelMin/Max/Median<T>; ladShift<Model><T>, ladShiftLong..., "
          "ladShiftShort..., ladShiftLongShare..., ladShift[Long|Short]PanelMin/Max<T>; ladQwenReq...<T>. Models: "
          + ", ".join(f"{v} ({nm(k)})" for k, v in MODEL_TAG.items()) + "."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def qwen_sentence(Q: dict) -> str:
    """One sentence for Methods: the form and the counts of the replies with no box, with two short examples."""
    order = list(TASKS)
    parts = [f"{Q[t]['bare']:,} of {Q[t]['rows']:,} on {TNAME[t]}" for t in order]
    med = sorted({la.fnum(Fraction(Q[t]["bare_tokens_median"]).limit_denominator(2), 0) for t in TASKS}, key=float)
    ex = []
    for t, f_ in (("gsm8k", "bare number"), ("hotpot", "bare words")):
        e = Q[t]["examples"].get(f_)
        if e:
            ex.append(f"`{e[0]}` on {TNAME[t]}")
    return (f"Asked for a direct answer, {nm(QWEN)} often wrote the answer alone, a few tokens long (median "
            f"{' to '.join(med)}), without the box the prompt asks for, so that the strict rule could not read it: "
            + ", ".join(parts[:-1]) + f" and {parts[-1]} zero-shot requested replies (for example "
            + " and ".join(ex) + ").")


# --------------------------------------------------------------------------------------------------------------------

def analyse(out_dir: Path = OUT) -> dict:
    t0 = time.time()
    la.ROUNDING_EDGE.clear()
    SZ = big_sizes()           # the group of 70B and above against the recorded sizes; stops the run if they differ
    J = json.loads((ROOT / SOURCES["ladder_json"]).read_text(encoding="utf-8"))
    V, W, checks = ladder_values(J)
    print(f"ladder values: {time.time() - t0:.0f} s", flush=True)
    P, notes = panel_means()
    checks += notes
    RV, notes = item_reversals(V, P)
    checks += notes
    print(f"reversals: {time.time() - t0:.0f} s", flush=True)
    R = {"created": time.strftime("%Y-%m-%d %H:%M"), "request": REQUEST, "script": "scripts/ladder_orderings.py",
         "ladder_json_created": J.get("created"), "models": {k: nm(k) for k in KEYS}, "big": BIG,
         "readers": f"{cs.CONTENT_SCORER_VERSION} and {ht.CONTENT_SCORER_VERSION} (content), as of this run",
         "big_size_note": big_size_note(SZ), "big_sizes": {k: SZ[k] for k in BIG}, "big_min_b": BIG_MIN_B,
         "families": {k: la.LADDER[k]["family"] for k in KEYS}, "n_boot": N_BOOT, "split_tokens": SPLIT_TOKENS,
         "sources": SOURCES,
         "rules": {"reversal": "pairs ordered one way by CoT - requested and the other by CoT - forced "
                               "(src/robustness.pair_reversals); |difference| <= 1e-9 pp under either is a tie: "
                               "neither reversed nor kept, counted among the pairs",
                   "basis": "model means over the 24 discovery draws (HotpotQA content too: the paper's basis of "
                            "\\aeHqReversedContent); all_draws and pooled rows for the hand-count check",
                   "bootstrap": "src/decision_directions.reversal_uncertainty; pooled: pooled_share_on, one draw of "
                                "the union of models for the three benchmarks",
                   "shift": "scripts/run_baseline_mixture.decomposition_block at 16 tokens, discovery draws weighted "
                            "by their draws",
                   "w": "the ladder's: share of requested outputs longer than the forced cap, discovery draws"},
         "values": {r: {t: {k: {s: {b: {"requested": V[r][t][k][s][b]["accuracy_pp"]["requested"],
                                         "forced": V[r][t][k][s][b]["accuracy_pp"]["forced"],
                                         "cot": V[r][t][k][s][b]["accuracy_pp"]["cot"],
                                         "cot_minus_requested": V[r][t][k][s][b]["gain_over_requested_pp"],
                                         "cot_minus_forced": V[r][t][k][s][b]["gain_over_forced_pp"]}
                                     for b in BASES} for s in SCORE_COL} for k in KEYS} for t in TASKS}
                    for r in READINGS},
         "reversals": RV}
    R["hand_check"] = hand_check(V, RV["six"])
    R["w"] = item_w(V, J)
    R["shift"] = item_shift(V, W)
    print(f"shift: {time.time() - t0:.0f} s", flush=True)
    R["qwen"] = item_qwen(V)
    R["qwen_sentence"] = qwen_sentence(R["qwen"]["tasks"])
    R["checks"] = checks
    R["macros"] = macros(R)
    R["rounding_edge_cases"] = list(la.ROUNDING_EDGE)
    R["runtime_s"] = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{STEM}.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=la.as_json),
                                          encoding="utf-8")
    md = out_dir / f"{STEM}.md"
    write_md(R, md)
    (out_dir / f"{STEM}_clean.md").write_text(la.npl.npt.strip_links(md.read_text(encoding="utf-8")), encoding="utf-8")
    sec = note_section(R)
    for name, clean in ((LADDER_NOTE, False), (LADDER_NOTE.replace(".md", "_clean.md"), True)):
        p = out_dir / name
        if p.exists():
            put_section(p, sec, clean)
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    R = analyse(Path(a.out))
    six = R["reversals"]["six"]["cut"]["content"]["discovery"]
    print("six, content: " + counts_line(six))
    c = R["reversals"]["combined"]["cut"]["content"]
    print("combined, content: " + ", ".join(f"{TNAME[t]} {share_ci(c[t])}" for t in TASKS)
          + f"; pooled {share_ci(c['pooled'])}")
    print(f"{len(R['macros'])} macros; {R['runtime_s']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
