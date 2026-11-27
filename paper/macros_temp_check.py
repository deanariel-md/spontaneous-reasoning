"""Macros of the temperature 0.7 spot check on MATH-500, prefix tempCheck, from results/temp_check/temp_check.json (scripts/temp_check.py
report). Every value is read from that file; nothing is typed here.

temp_check_results(root) reads the JSON, or returns None until it exists; temp_check_macros(R) returns {name: value}
after checking every name (letters only, prefix tempCheck).

Models (one per family): Gemma (gemma-3-12b-it), Llama (llama-2-13b-chat-hf), Mistral (mistral-nemo-instruct-2407),
Phi (phi-3-medium-4k-instruct), Qwen (qwen3-14b); Pooled is the mean of the five. Values in percentage points, one
decimal; w in percent. Kind: none = under sampling (temperature 0.7), Greedy = the panel's greedy outputs of the same
requests, Diff = sampled minus greedy; Lo and Hi are the 95% question-bootstrap interval of Diff.
  tempCheckW<Kind>Math<Model>Pct[Lo|Hi]            w, the share of requested outputs longer than 16 tokens (no
                                                   scoring involved, so no Content variant)
  tempCheck<Q><Kind>Math[Content]<Model>[Lo|Hi]    Q: ReqGain (CoT - requested), FdGain (CoT - forced), CorrGain
                                                   ((1 - w)(CoT - forced)), ReqAcc, FdAcc, CotAcc; Content for
                                                   content/1.7.0, else strict
  tempCheckModel<Model>                            the model's name as manifests/temp_check_models.json gives it
  tempCheckWDiffMaxAbsPct, tempCheck<Q>DiffMaxAbs[Content]   the largest absolute per-model difference
  tempCheckWDiffCiZero, tempCheck<Q>DiffCiZero[Content]      of the five models, how many have a Diff interval
                                                   containing 0
  tempCheckSameOut<Arm>Pct                         share of sampled outputs identical to the greedy output of the same
                                                   request, all five models (Arm: Req, Fd, Cot)
  tempCheckModels, tempCheckItems, tempCheckTemplates, tempCheckTemperature, tempCheckGenerations,
  tempCheckBootDraws                               the design
  tempCheckPilotGreedySame, tempCheckPilotGreedyRows   the pilot's greedy replicate: outputs equal to the stored
                                                   greedy ones, of how many (results/temp_check/pilot_check.json)
  tempCheckGpuHours                                GPU-hours of all the check's jobs (temp_check.json, jobs)
"""
from __future__ import annotations

import json
import math
import os
import re

PREFIX = "tempCheck"
PATH = os.path.join("results", "temp_check", "temp_check.json")
PILOT = os.path.join("results", "temp_check", "pilot_check.json")
FAMILY_LABEL = {"Gemma": "Gemma", "Llama": "Llama", "Mistral": "Mistral", "Phi": "Phi", "Qwen": "Qwen"}
Q_NAME = {"req_gain": "ReqGain", "fd_gain": "FdGain", "corrected_gain": "CorrGain", "requested": "ReqAcc",
          "forced": "FdAcc", "cot": "CotAcc"}
ARM_NAME = {"requested": "Req", "forced": "Fd", "cot": "Cot"}


def temp_check_results(root: str):
    p = os.path.join(root, PATH)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        R = json.load(fh)
    q = os.path.join(root, PILOT)
    R["_pilot"] = json.load(open(q, encoding="utf-8")) if os.path.exists(q) else None
    return R


def num(x, nd=1) -> str:
    """As paper/make_numbers.num: a value that rounds to zero prints 0.0, minus as $-$."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "$-$")


def thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def temp_check_macros(R) -> dict:
    if not R:
        return {}
    out = {}

    def put(k, v):
        if not re.fullmatch(PREFIX + r"[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not {PREFIX} followed by letters")
        if k in out:
            raise ValueError(f"macro {k} set twice")
        out[k] = v

    d = R["design"]
    fam = {m["model_key"]: FAMILY_LABEL[m["family"]] for m in d["models"]}
    for m in d["models"]:
        put(f"{PREFIX}Model{fam[m['model_key']]}", str(m.get("name") or m["model_key"]))
    put(f"{PREFIX}Models", str(len(d["models"])))
    put(f"{PREFIX}Items", str(d["n_items"]))
    put(f"{PREFIX}Templates", str(len(d["templates"])))
    put(f"{PREFIX}Temperature", f"{float(d['sampling']['temperature']):.1f}")
    put(f"{PREFIX}Generations", thousands(d["n_generations_sampled"]))
    put(f"{PREFIX}BootDraws", thousands(R["analysis"]["draws"]))
    an = R["analysis"]["scores"]
    for score, S in (("strict_correct", ""), ("content_correct", "Content")):
        r = an[score]
        labels = {k: fam[k] for k in r if k != "pooled"}
        labels["pooled"] = "Pooled"
        for key, L in labels.items():
            x = r[key]
            if S == "":   # w does not depend on the scoring
                put(f"{PREFIX}WMath{L}Pct", num(100 * x["sampled"]["w"]))
                put(f"{PREFIX}WGreedyMath{L}Pct", num(100 * x["greedy"]["w"]))
                put(f"{PREFIX}WDiffMath{L}Pct", num(100 * x["diff"]["w"]))
                lo, hi = x["diff_ci"]["w"]
                put(f"{PREFIX}WDiffMath{L}PctLo", num(None if lo is None else 100 * lo))
                put(f"{PREFIX}WDiffMath{L}PctHi", num(None if hi is None else 100 * hi))
            for q, Q in Q_NAME.items():
                put(f"{PREFIX}{Q}Math{S}{L}", num(100 * x["sampled"][q]))
                put(f"{PREFIX}{Q}GreedyMath{S}{L}", num(100 * x["greedy"][q]))
                put(f"{PREFIX}{Q}DiffMath{S}{L}", num(100 * x["diff"][q]))
                lo, hi = x["diff_ci"][q]
                put(f"{PREFIX}{Q}DiffMath{S}{L}Lo", num(None if lo is None else 100 * lo))
                put(f"{PREFIX}{Q}DiffMath{S}{L}Hi", num(None if hi is None else 100 * hi))
        keys = [k for k in r if k != "pooled"]
        for q, Q in (("w", "W"),) + tuple(Q_NAME.items()):
            if q == "w" and S:
                continue
            mx = max(abs(r[k]["diff"][q]) for k in keys)
            zero = sum(1 for k in keys if r[k]["diff_ci"][q][0] is not None and r[k]["diff_ci"][q][0] <= 0 <= r[k]["diff_ci"][q][1])
            if q == "w":
                put(f"{PREFIX}WDiffMaxAbsPct", num(100 * mx))
                put(f"{PREFIX}WDiffCiZero", str(zero))
            else:
                put(f"{PREFIX}{Q}DiffMaxAbs{S}", num(100 * mx))
                put(f"{PREFIX}{Q}DiffCiZero{S}", str(zero))
    same = {a: [0, 0] for a in ARM_NAME}
    for o in R["outputs"].values():
        for a in ARM_NAME:
            same[a][0] += o[a]["identical_outputs"]
            same[a][1] += o[a]["compared"]
    for a, A in ARM_NAME.items():
        put(f"{PREFIX}SameOut{A}Pct", num(100 * same[a][0] / same[a][1]) if same[a][1] else "n/a")
    P = R.get("_pilot")
    if P:
        g = [m.get("greedy_identical") for m in P["models"].values() if m.get("greedy_identical")]
        put(f"{PREFIX}PilotGreedySame", str(sum(x["identical"] for x in g)))
        put(f"{PREFIX}PilotGreedyRows", str(sum(x["rows"] for x in g)))
    jobs = R.get("jobs") or {}
    if jobs.get("scheduler_gpu_hours") is not None:
        put(f"{PREFIX}GpuHours", num(jobs["scheduler_gpu_hours"], 2))
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for k, v in sorted(temp_check_macros(temp_check_results(root)).items()):
        print(f"\\newcommand{{\\{k}}}{{{v}}}")
