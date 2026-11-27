r"""The cap check: cut native-reasoning outputs continued to twice the cap.

PowerShell (from the repository root), after the continuations are copied back to data/d2_out/capcheck/<key>:
    .\.venv\Scripts\python.exe scripts\run_capcheck.py

Reads manifests/thinking_capcheck_sample.json, data/requests/capcheck/<key>.parquet (the cut texts) and the shards of
scripts/capcheck_generate.py. Each cut text and its continuation are read after the last end-of-reasoning marker and
scored by the task's scorer (scripts/score_thinking.py), strict and lenient. Per checkpoint and task: the
share closed within 61,440 tokens and the share correct (Clopper-Pearson 95%), the estimated native accuracy at 61,440
tokens (the native-reasoning accuracy at 30,720 plus the discovery-draw share of outputs cut before the marker times the share
correct) and its difference from chain of thought, and the share of cut texts that loop. Beside them, the answer the
cut reasoning had reached: the last non-empty \boxed{...} of the cut text that is not one of its prompt's own boxes,
scored the same way, and native accuracy at 30,720 tokens had the run read it. Writes
results/thinking_capcheck/{thinking_capcheck.json, summary.md, summary_clean.md, per_output.csv}.
--partial runs on the continuations present so far and writes under --out (never results/thinking_capcheck).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASKS = {"gsm8k": ("GSM8K", "Gsm"), "math": ("MATH-500, numeric items", "Math")}
TAGS = {"qwen3.5-35b-a3b": "QThreeFiveMoe", "qwen3.6-35b-a3b": "QThreeSixMoe", "qwen3.6-27b": "QThreeSixDense"}
MANIFEST = os.path.join(ROOT, "manifests", "thinking_capcheck_sample.json")
REQ_DIR = os.path.join(ROOT, "data", "requests", "capcheck")
CONT_DIR = os.path.join(ROOT, "data", "d2_out", "capcheck")
OUT = os.path.join(ROOT, "results", "thinking_capcheck")
KEYS = ["model_key", "arm", "config_id", "item_id"]


# ============================================================================ pure helpers (tested)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    from scipy.stats import beta
    if n <= 0:
        raise ValueError("n must be positive")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def looping(text: str, window: int = 100, share: float = 0.9) -> bool:
    """At least `share` of the last `window` non-empty lines repeat a line already seen among them."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-window:]
    if len(lines) < window:
        return False
    seen, rep = set(), 0
    for ln in lines:
        rep += ln in seen
        seen.add(ln)
    return rep >= share * window


def estimate(acc_at_cap: float, cut_open_pp: float, p: float) -> float:
    """Native accuracy (pp) at twice the cap: accuracy at the cap plus the cut-open share (pp) times the share correct."""
    return acc_at_cap + cut_open_pp * p


def prompt_boxes(messages_json: str) -> set[str]:
    """Contents of the \\boxed{...} in the request's own messages (the frames' placeholder, such as <number>)."""
    from src.bbh import boxed_contents
    return {b.strip() for m in json.loads(messages_json) for b in boxed_contents(str(m.get("content", "")))}


def reached_answer(text: str, skip: set[str]) -> str | None:
    """The last complete, non-empty \\boxed{...} of a cut text whose content is not one of the prompt's boxes."""
    from src.bbh import boxed_contents
    for b in reversed(boxed_contents(text)):
        if b.strip() and b.strip() not in skip:
            return b
    return None


def bodies(cut_texts, cont_texts, finish):
    """The text the scorer reads for each cut text and its continuation (score_thinking.answer_bodies) and closure."""
    from score_thinking import answer_bodies
    full = [a + b for a, b in zip(cut_texts, cont_texts)]
    return answer_bodies(full, list(finish))


# ============================================================================ inputs


def continuations(key: str) -> pd.DataFrame:
    d = os.path.join(CONT_DIR, key)
    parts = [pd.read_parquet(os.path.join(d, p)) for p in sorted(os.listdir(d)) if p.startswith("shard-")] \
        if os.path.isdir(d) else []
    if not parts:
        return pd.DataFrame(columns=["request_id", "task", "model_key", "continuation_text", "n_cont_tokens",
                                     "finish_reason", "n_cut_tokens_retok", "n_prompt_tokens"])
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["task", "request_id"], keep="last")
    return df[~dup].reset_index(drop=True)    # a pilot output run again by the full job keeps its latest copy


def score_rows(task: str, rows: pd.DataFrame) -> pd.DataFrame:
    """strict_correct and lenient_correct of rows holding request_id, model_key, arm, config_id, item_id, output_text."""
    from score_thinking import scorer
    fn, _, _, _ = scorer(task)
    out = rows.copy()
    if task == "gsm8k":
        out["item_id"] = out["item_id"].astype(int)
    out["finish_reason"] = out.get("finish_reason", "stop")
    out["n_output_tokens"] = out.get("n_output_tokens", 0)
    sc = fn(out.reset_index(drop=True))
    return sc[["request_id", "strict_correct", "lenient_correct"]]


def scoring_check(task: str, key: str, n: int = 200) -> dict:
    """Rescore n native-reasoning outputs of the checkpoint that closed their reasoning; they must match the scored table."""
    from analyze_thinking_pilot import answer_text
    from src import generation as gen
    out = gen.read_outputs(os.path.join(ROOT, "data", "d2_out", "thinking", task, key))
    out = out[out["output_text"].astype(str).map(lambda t: answer_text(t) is not None)]
    out = out.sample(n=min(n, len(out)), random_state=0).reset_index(drop=True)
    b, _ = bodies(out["output_text"].astype(str), [""] * len(out), out["finish_reason"].astype(str))
    mine = score_rows(task, out.assign(output_text=b, item_id=out["item_id"].astype(str)))
    ref = pd.read_parquet(os.path.join(ROOT, "data", "processed", "thinking", f"{task}_scored.parquet"),
                          filters=[("model_key", "==", key)])
    ref["item_id"] = ref["item_id"].astype(str)
    m = out[KEYS].assign(item_id=out["item_id"].astype(str)).merge(ref[KEYS + ["strict_correct", "lenient_correct"]],
                                                                    on=KEYS, how="left")
    same = (m["strict_correct"].astype(int).to_numpy() == mine["strict_correct"].astype(int).to_numpy()) & \
        (m["lenient_correct"].astype(int).to_numpy() == mine["lenient_correct"].astype(int).to_numpy())
    return {"n": int(len(out)), "matching": int(same.sum())}


def draw_terms(task: str) -> dict:
    """Per model, over the native-reasoning run's discovery draws: native accuracy and the share (pp) of native outputs cut before the
    marker, through the arrays of scripts/run_thinking.py; native accuracy must equal the run's per-model table."""
    import run_thinking as rt
    inp, s, items, items_info = rt.collection_inputs(task)
    full = list(inp["panel"])
    fd_load, _, _ = rt.forced_inputs(task, full, s, items_info, inp["load"], False)
    panel = rt.thinking_panel(task, full, False)
    th_load, _, closure, panel = rt.thinking_inputs(task, panel, s, items_info, inp["load"], False, False)
    disc = list(s["discovery_config_ids"])
    clo = closure.assign(item_id=closure["item_id"].astype(str))

    def cut_load(ids):
        df = th_load(ids)
        m = df[KEYS].assign(item_id=df["item_id"].astype(str)).merge(clo, on=KEYS, how="left")
        if m["closed"].isna().any():
            raise SystemExit(f"{task}: native rows without a closure flag")
        return df.assign(strict_correct=((df["finish_reason"].astype(str) == "length").to_numpy()
                                         & ~m["closed"].astype(bool).to_numpy()).astype(int))
    Pth = rt.four_arm_arrays(inp["load"], fd_load, th_load, disc, panel, items)[3]
    Cut = rt.four_arm_arrays(inp["load"], fd_load, cut_load, disc, panel, items)[3]
    return {k: {"native": float(Pth[:, j].mean()), "cut_open_pp": float(Cut[:, j].mean())} for j, k in enumerate(panel)}


# ============================================================================ run


def run(a):
    t0 = time.time()
    man = json.load(open(MANIFEST, encoding="utf-8"))
    th = json.load(open(os.path.join(ROOT, "results", "thinking", "thinking.json"), encoding="utf-8"))
    res = {"analysis": "thinking_capcheck", "study_part": "cap check", "partial": bool(a.partial),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "manifest": os.path.relpath(MANIFEST, ROOT).replace("\\", "/"),
           "extra_tokens": man["extra_tokens"], "tasks": {}}
    rows_all, macros = [], {}
    put = macros.__setitem__
    for task, (title, tp) in TASKS.items():
        pm = {r["model_key"]: r for r in th["tasks"][task]["per_model"]}
        terms = draw_terms(task)
        for k, v in terms.items():
            if abs(v["native"] - pm[k]["native_reasoning"]) > 1e-9:
                raise SystemExit(f"{task}/{k}: native accuracy {v['native']} differs from the run's {pm[k]['native_reasoning']}")
        T = {"title": title, "models": {}}
        for key in man["models"]:
            smp = man["samples"][f"{task}/{key}"]
            req = pd.read_parquet(os.path.join(REQ_DIR, f"{key}.parquet"))
            req = req[req["task"] == task]
            if sorted(req["request_id"]) != sorted(smp["request_ids"]):
                raise SystemExit(f"{task}/{key}: the request file differs from the manifest's sample")
            cont = continuations(key)
            cont = cont[cont["task"] == task]
            if set(cont["model_key"]) - {key}:
                raise SystemExit(f"{task}/{key}: continuations of another model in its folder")
            df = req.merge(cont.drop(columns=["model_key"]), on=["task", "request_id"], how="inner", validate="one_to_one")
            if len(df) < len(req) and not a.partial:
                raise SystemExit(f"{task}/{key}: {len(req) - len(df)} of {len(req)} continuations missing (--partial)")
            if df.empty:
                continue
            b, closed = bodies(df["cut_text"].astype(str), df["continuation_text"].astype(str), df["finish_reason"].astype(str))
            sc = score_rows(task, df[["request_id", "model_key", "arm", "config_id", "item_id"]].assign(output_text=b))
            df = df.merge(sc, on="request_id", how="left", validate="one_to_one")
            df["closed"] = closed
            df["looping"] = df["cut_text"].astype(str).map(looping)
            ra = [reached_answer(t, prompt_boxes(m)) for t, m in zip(df["cut_text"].astype(str), df["messages_json"].astype(str))]
            df["reached_box"] = [x is not None for x in ra]
            rs = score_rows(task, df[["request_id", "model_key", "arm", "config_id", "item_id"]].assign(
                output_text=["" if x is None else "\\boxed{" + x + "}" for x in ra]))
            df = df.merge(rs.rename(columns={"strict_correct": "reached_strict", "lenient_correct": "reached_lenient"}),
                          on="request_id", how="left", validate="one_to_one")
            df["total_tokens"] = df["cut_tokens"].astype(int) + df["n_cont_tokens"].astype(int)
            n = int(len(df))
            k_closed, k_strict, k_len = int(df["closed"].sum()), int(df["strict_correct"].sum()), int(df["lenient_correct"].sum())
            p, (plo, phi) = k_strict / n, clopper_pearson(k_strict, n)
            pl, (pllo, plhi) = k_len / n, clopper_pearson(k_len, n)
            A, c, cot = terms[key]["native"], terms[key]["cut_open_pp"], pm[key]["chain_of_thought"]
            est = {"estimate": estimate(A, c, p), "ci95": [estimate(A, c, plo), estimate(A, c, phi)]}
            est_l = {"estimate": estimate(A, c, pl), "ci95": [estimate(A, c, pllo), estimate(A, c, plhi)]}
            lp = df["looping"]
            M = {"n_sampled": n, "n_eligible": smp["n_eligible"], "native_at_cap": A, "cut_open_pp": c, "chain_of_thought": cot,
                 "closed": {"k": k_closed, "share": k_closed / n, "ci95": list(clopper_pearson(k_closed, n))},
                 "strict": {"k": k_strict, "share": p, "ci95": [plo, phi]},
                 "lenient": {"k": k_len, "share": pl, "ci95": [pllo, plhi]},
                 "native_at_double_cap": est, "native_at_double_cap_lenient": est_l,
                 "double_cap_minus_cot": {"estimate": est["estimate"] - cot, "ci95": [est["ci95"][0] - cot, est["ci95"][1] - cot]},
                 "cut_again": int((~df["closed"] & (df["finish_reason"] == "length")).sum()),
                 "stopped_unclosed": int((~df["closed"] & (df["finish_reason"] != "length")).sum()),
                 "closing_total_tokens_median": float(df.loc[df["closed"], "total_tokens"].median()) if k_closed else None,
                 "looping": {"k": int(lp.sum()), "share": float(lp.mean()),
                             "closed_if_looping": float(df.loc[lp, "closed"].mean()) if lp.any() else None,
                             "correct_if_looping": float(df.loc[lp, "strict_correct"].mean()) if lp.any() else None,
                             "closed_if_not": float(df.loc[~lp, "closed"].mean()) if (~lp).any() else None,
                             "correct_if_not": float(df.loc[~lp, "strict_correct"].mean()) if (~lp).any() else None},
                 "retokenized_cut_minus_recorded": {"min": int((df["n_cut_tokens_retok"] - df["cut_tokens"]).min()),
                                                    "median": float((df["n_cut_tokens_retok"] - df["cut_tokens"]).median()),
                                                    "max": int((df["n_cut_tokens_retok"] - df["cut_tokens"]).max())}}
            k_box, k_rs, k_rl = int(df["reached_box"].sum()), int(df["reached_strict"].sum()), int(df["reached_lenient"].sum())
            pr, (prlo, prhi) = k_rs / n, clopper_pearson(k_rs, n)
            r_est = {"estimate": estimate(A, c, pr), "ci95": [estimate(A, c, prlo), estimate(A, c, prhi)]}
            M["reached"] = {"box": {"k": k_box, "share": k_box / n, "ci95": list(clopper_pearson(k_box, n))},
                            "strict": {"k": k_rs, "share": pr, "ci95": [prlo, prhi]},
                            "lenient": {"k": k_rl, "share": k_rl / n, "ci95": list(clopper_pearson(k_rl, n))},
                            "strict_if_box": k_rs / k_box if k_box else None,
                            "native_reading_box": r_est,
                            "reading_box_minus_cot": {"estimate": r_est["estimate"] - cot,
                                                      "ci95": [r_est["ci95"][0] - cot, r_est["ci95"][1] - cot]}}
            if not a.skip_scoring_check:
                M["scoring_check"] = scoring_check(task, key)
                if M["scoring_check"]["matching"] != M["scoring_check"]["n"]:
                    raise SystemExit(f"{task}/{key}: rescoring of native-reasoning outputs does not match the scored table")
            T["models"][key] = M
            tg = tp + TAGS[key]
            put(f"thc{tg}N", str(n))
            for nm, blk in (("Closed", M["closed"]), ("Correct", M["strict"]), ("CorrectLenient", M["lenient"])):
                put(f"thc{tg}{nm}", f"{100 * blk['share']:.1f}")
                put(f"thc{tg}{nm}Lo", f"{100 * blk['ci95'][0]:.1f}")
                put(f"thc{tg}{nm}Hi", f"{100 * blk['ci95'][1]:.1f}")
            for nm, blk in (("Est", est), ("EstMinusCot", M["double_cap_minus_cot"])):
                put(f"thc{tg}{nm}", fmt(blk["estimate"]))
                put(f"thc{tg}{nm}Lo", fmt(blk["ci95"][0]))
                put(f"thc{tg}{nm}Hi", fmt(blk["ci95"][1]))
            put(f"thc{tg}Loop", f"{100 * M['looping']['share']:.1f}")
            put(f"thc{tg}CutAgain", f"{100 * M['cut_again'] / n:.1f}")
            R = M["reached"]
            for nm, blk in (("ReachedBox", R["box"]), ("ReachedCorrect", R["strict"]), ("ReachedCorrectLenient", R["lenient"])):
                put(f"thc{tg}{nm}", f"{100 * blk['share']:.1f}")
                put(f"thc{tg}{nm}Lo", f"{100 * blk['ci95'][0]:.1f}")
                put(f"thc{tg}{nm}Hi", f"{100 * blk['ci95'][1]:.1f}")
            for nm, blk in (("ReachedEst", R["native_reading_box"]), ("ReachedEstMinusCot", R["reading_box_minus_cot"])):
                put(f"thc{tg}{nm}", fmt(blk["estimate"]))
                put(f"thc{tg}{nm}Lo", fmt(blk["ci95"][0]))
                put(f"thc{tg}{nm}Hi", fmt(blk["ci95"][1]))
            rows_all.append(df.assign(task=task)[["task", "model_key", "request_id", "config_id", "item_id", "looping",
                                                  "closed", "strict_correct", "lenient_correct", "finish_reason",
                                                  "n_cont_tokens", "total_tokens", "n_cut_tokens_retok", "cut_tokens",
                                                  "reached_box", "reached_strict", "reached_lenient"]])
        # the eight-checkpoint means: the continued checkpoints at their estimate, the others as the run has them
        if len(T["models"]) == len(man["models"]):
            ks = list(pm)
            base = [pm[k]["native_reasoning"] for k in ks]
            cots = [pm[k]["chain_of_thought"] for k in ks]
            pt = [T["models"][k]["native_at_double_cap"]["estimate"] if k in T["models"] else pm[k]["native_reasoning"] for k in ks]
            lo = [T["models"][k]["native_at_double_cap"]["ci95"][0] if k in T["models"] else pm[k]["native_reasoning"] for k in ks]
            hi = [T["models"][k]["native_at_double_cap"]["ci95"][1] if k in T["models"] else pm[k]["native_reasoning"] for k in ks]
            mc = float(np.mean(cots))
            T["mean"] = {"native_at_cap": float(np.mean(base)), "chain_of_thought": mc,
                         "native_at_double_cap": {"estimate": float(np.mean(pt)), "ci95": [float(np.mean(lo)), float(np.mean(hi))],
                                                  "note": "interval from the three checkpoints' bounds together (conservative)"},
                         "double_cap_minus_cot": {"estimate": float(np.mean(pt)) - mc,
                                                  "ci95": [float(np.mean(lo)) - mc, float(np.mean(hi)) - mc]}}
            rb = [T["models"][k]["reached"]["native_reading_box"] if k in T["models"] else None for k in ks]
            rpt = [r["estimate"] if r else b for r, b in zip(rb, base)]
            rlo = [r["ci95"][0] if r else b for r, b in zip(rb, base)]
            rhi = [r["ci95"][1] if r else b for r, b in zip(rb, base)]
            T["mean"]["native_reading_box"] = {"estimate": float(np.mean(rpt)), "ci95": [float(np.mean(rlo)), float(np.mean(rhi))],
                                               "note": "interval from the three checkpoints' bounds together (conservative)"}
            T["mean"]["reading_box_minus_cot"] = {"estimate": float(np.mean(rpt)) - mc,
                                                  "ci95": [float(np.mean(rlo)) - mc, float(np.mean(rhi)) - mc]}
            for nm, blk in (("MeanEst", T["mean"]["native_at_double_cap"]), ("MeanEstMinusCot", T["mean"]["double_cap_minus_cot"]),
                            ("MeanReachedEst", T["mean"]["native_reading_box"]),
                            ("MeanReachedEstMinusCot", T["mean"]["reading_box_minus_cot"])):
                put(f"thc{tp}{nm}", fmt(blk["estimate"]))
                put(f"thc{tp}{nm}Lo", fmt(blk["ci95"][0]))
                put(f"thc{tp}{nm}Hi", fmt(blk["ci95"][1]))
            ns = sum(T["models"][k]["n_sampled"] for k in T["models"])
            for nm, fld in (("Closed", "closed"), ("Correct", "strict")):
                kk = sum(T["models"][k][fld]["k"] for k in T["models"])
                put(f"thc{tp}Pooled{nm}", f"{100 * kk / ns:.1f}")
            for nm, fld in (("ReachedBox", "box"), ("ReachedCorrect", "strict")):
                kk = sum(T["models"][k]["reached"][fld]["k"] for k in T["models"])
                put(f"thc{tp}Pooled{nm}", f"{100 * kk / ns:.1f}")
            put(f"thc{tp}PooledLoop", f"{100 * sum(T['models'][k]['looping']['k'] for k in T['models']) / ns:.1f}")
        res["tasks"][task] = T
    res["macros"] = macros
    res["runtime_seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "thinking_capcheck.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    if rows_all:
        pd.concat(rows_all, ignore_index=True).to_csv(os.path.join(a.out, "per_output.csv"), index=False)
    md = summary(res)
    for nm in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, nm), "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


def fmt(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def pct(blk) -> str:
    return f"{100 * blk['share']:.1f}% [{100 * blk['ci95'][0]:.1f}, {100 * blk['ci95'][1]:.1f}]"


def summary(res: dict) -> str:
    L = ["# Cut native-reasoning outputs continued to twice the cap", "",
         "Three thinking checkpoints often ran past the cap of 30,720 tokens, and an output cut before the end of its "
         "reasoning counts as wrong. For each of them and each task, 300 such outputs drawn at random "
         "(`manifests/thinking_capcheck_sample.json`) were continued by greedy decoding for up to 30,720 more tokens, "
         "61,440 in all, then read and scored as in the main run. Shares carry Clopper-Pearson 95% intervals. Native accuracy "
         "at 61,440 tokens is estimated as the main run's accuracy plus the share of outputs cut before the marker times the "
         "share correct, both terms over the discovery draws. A cut text loops when at least 90% of its last 100 "
         "non-empty lines repeat an earlier one among them." + (" **Partial run.**" if res["partial"] else ""), ""]
    for task, T in res["tasks"].items():
        L += [f"## {T['title']}", "",
              "| Checkpoint | n | Closed | Correct (strict) | Correct (lenient) | Cut again | Looping | Native at 30,720 | "
              "Native at 61,440 | Chain of thought |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, M in T["models"].items():
            e = M["native_at_double_cap"]
            L.append(f"| {k} | {M['n_sampled']} | {pct(M['closed'])} | {pct(M['strict'])} | {pct(M['lenient'])} | "
                     f"{M['cut_again']} | {100 * M['looping']['share']:.1f}% | {M['native_at_cap']:.1f} | "
                     f"{e['estimate']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}] | {M['chain_of_thought']:.1f} |")
        L.append("")
        for k, M in T["models"].items():
            lp = M["looping"]
            f2 = lambda v: "--" if v is None else f"{100 * v:.1f}%"  # noqa: E731
            L.append(f"- {k}: {M['cut_open_pp']:.2f}% of native outputs cut before the marker (discovery draws); "
                     f"looping cut texts closed {f2(lp['closed_if_looping'])} and were correct {f2(lp['correct_if_looping'])}, "
                     f"the others {f2(lp['closed_if_not'])} and {f2(lp['correct_if_not'])}; median length of the closing "
                     f"outputs {M['closing_total_tokens_median'] if M['closing_total_tokens_median'] is not None else '--'} tokens; "
                     f"re-tokenized cut length minus recorded: median {M['retokenized_cut_minus_recorded']['median']:.0f} "
                     f"(range {M['retokenized_cut_minus_recorded']['min']} to {M['retokenized_cut_minus_recorded']['max']})"
                     + (f"; rescoring check {M['scoring_check']['matching']} of {M['scoring_check']['n']}" if "scoring_check" in M else "")
                     + ".")
        if "mean" in T:
            m = T["mean"]
            e, d = m["native_at_double_cap"], m["double_cap_minus_cot"]
            L += ["", f"Mean over the eight checkpoints (the other five as in the main run): native reasoning {m['native_at_cap']:.1f} at "
                  f"30,720 tokens and {e['estimate']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}] at 61,440, against chain of "
                  f"thought {m['chain_of_thought']:.1f}: a difference of {d['estimate']:.1f} [{d['ci95'][0]:.1f}, "
                  f"{d['ci95'][1]:.1f}] (interval from the three checkpoints' bounds together)."]
        L += ["", f"### {T['title']}: the answer the cut reasoning had reached", "",
              "The last non-empty boxed answer of each cut text that is not its prompt's placeholder, scored as above; "
              "native accuracy at 30,720 tokens had the run read it. It credits answers the model never delivered.", "",
              "| Checkpoint | n | Holds a box | Box correct (strict) | Box correct (lenient) | Correct among boxes | "
              "Native at 30,720 | Reading the box | Chain of thought |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, M in T["models"].items():
            R = M["reached"]
            e = R["native_reading_box"]
            sib = "--" if R["strict_if_box"] is None else f"{100 * R['strict_if_box']:.1f}%"
            L.append(f"| {k} | {M['n_sampled']} | {pct(R['box'])} | {pct(R['strict'])} | {pct(R['lenient'])} | {sib} | "
                     f"{M['native_at_cap']:.1f} | {e['estimate']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}] | "
                     f"{M['chain_of_thought']:.1f} |")
        if "mean" in T:
            m = T["mean"]
            e, d = m["native_reading_box"], m["reading_box_minus_cot"]
            L += ["", f"Mean over the eight checkpoints: native reasoning {m['native_at_cap']:.1f} at 30,720 tokens and "
                  f"{e['estimate']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}] reading the box, against chain of thought "
                  f"{m['chain_of_thought']:.1f}: a difference of {d['estimate']:.1f} [{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}]."]
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--skip-scoring-check", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.partial and os.path.abspath(a.out) == os.path.abspath(OUT):
        raise SystemExit("a partial run writes under --out, never to results/thinking_capcheck")
    run(a)


if __name__ == "__main__":
    main()
