r"""Analysis of native reasoning under the vendors' sampling settings.

PowerShell (from the repository root), after scripts\score_thinking_vendor.py:
    .\.venv\Scripts\python.exe scripts\run_thinking_vendor.py [--no-comparators]

Per checkpoint and task, for the vendor outputs and the greedy outputs of the same 600 requests: the share cut at the
cap before the end-of-reasoning marker, the share looping by the looping rule among the cut outputs, strict accuracy (lenient
beside it) and mean output tokens; each share with a Clopper-Pearson 95% interval. On the same (configuration, item)
cells, the collection's prompted chain of thought and the forced-direct arm (strict), unless --no-comparators. Paired
differences over the cells (vendor - greedy, vendor - chain of thought, greedy - chain of thought, vendor - forced) with
a bootstrap over items: both configurations of an item are resampled together, 10,000 resamples,
numpy.random.default_rng(20261117).

Outputs: results/thinking_vendor/thinking_vendor.json, summary.md and summary_clean.md, per_cell.csv.
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

SCORED = os.path.join(ROOT, "data", "processed", "thinking_vendor", "scored.parquet")
SAMPLE = os.path.join(ROOT, "manifests", "thinking_vendor_sample.json")
OUT = os.path.join(ROOT, "results", "thinking_vendor")
TASKS = {"gsm8k": "GSM8K", "math": "MATH-500, numeric items"}
SEED = 20261117
N_BOOT = 10000
KEYS = ["model_key", "task", "config_id", "item_id"]


# ============================================================================ pure helpers (tested)


def share(k: int, n: int) -> dict:
    from run_capcheck import clopper_pearson
    if n == 0:
        return {"k": 0, "n": 0, "pct": None, "lo": None, "hi": None}
    lo, hi = clopper_pearson(int(k), int(n))
    return {"k": int(k), "n": int(n), "pct": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi}


def item_bootstrap(diff, items, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of the paired differences (pp) and a percentile 95% interval from resampling items with replacement, every
    cell of a drawn item kept together."""
    diff = np.asarray(diff, float)
    items = np.asarray(items).astype(str)
    if diff.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n_cells": 0, "n_items": 0}
    u, inv = np.unique(items, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(u), size=(n_boot, len(u)))
    reps = sums[draw].sum(1) / cnts[draw].sum(1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n_cells": int(diff.size), "n_items": int(len(u))}


def decoding_block(df: pd.DataFrame) -> dict:
    """Shares and accuracy of one decoding's rows (one checkpoint and task)."""
    n = len(df)
    cut = df["cut_open"].astype(bool)
    return {"n": int(n), "cut": share(int(cut.sum()), n),
            "looping_among_cut": share(int((cut & df["looping"].astype(bool)).sum()), int(cut.sum())),
            "looping_all": share(int(df["looping"].astype(bool).sum()), n),
            "strict": share(int(df["strict_correct"].sum()), n), "lenient": share(int(df["lenient_correct"].sum()), n),
            "closed": share(int(df["closed"].astype(bool).sum()), n),
            "mean_output_tokens": float(df["n_output_tokens"].mean()) if n else None}


def paired(wide: pd.DataFrame, a: str, b: str) -> dict:
    m = wide[[a, b, "item_id"]].dropna()
    return item_bootstrap(100 * (m[a].to_numpy(float) - m[b].to_numpy(float)), m["item_id"].to_numpy())


# ============================================================================ inputs


def comparators(sample: dict, keys) -> pd.DataFrame:
    """Strict scores of the collection's chain of thought and the forced-direct arm on the sample's cells."""
    from run_forced_direct import collection_inputs, forced_inputs
    rows = []
    for task, m in sample["tasks"].items():
        inp, s, items, items_info = collection_inputs(task)
        cfgs = list(m["configurations"])
        col = inp["load"](cfgs)
        fd_load, _, _ = forced_inputs(task, list(inp["panel"]), s, items_info, inp["load"], False)
        fd = fd_load(cfgs)
        for name, df in (("cot", col[col["arm"] == "zero_shot_cot"]), ("forced", fd)):
            d = df[df["model_key"].isin(keys)][["model_key", "config_id", "item_id", "strict_correct"]].copy()
            d["item_id"] = d["item_id"].astype(str)
            d = d[d["item_id"].isin(set(m["items"]))]
            rows.append(d.assign(task=task, decoding=name))
    return pd.concat(rows, ignore_index=True)


def wide_table(scored: pd.DataFrame, comp: pd.DataFrame | None) -> pd.DataFrame:
    s = scored[KEYS + ["decoding", "strict_correct"]].copy()
    s["item_id"] = s["item_id"].astype(str)
    if comp is not None and len(comp):
        s = pd.concat([s, comp[KEYS + ["decoding", "strict_correct"]]], ignore_index=True)
    if s.duplicated(KEYS + ["decoding"]).any():
        raise SystemExit("two scores for one cell and decoding")
    return s.pivot_table(index=KEYS, columns="decoding", values="strict_correct", aggfunc="first").reset_index()


# ============================================================================ run


def run(a) -> dict:
    t0 = time.time()
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    scored = pd.read_parquet(a.scored)
    scored["item_id"] = scored["item_id"].astype(str)
    keys = sorted(scored["model_key"].unique())
    comp = None if a.no_comparators else comparators(sample, keys)
    wide = wide_table(scored, comp)
    res = {"label": "thinking_vendor", "study_part": "vendor sampling settings", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "scored": os.path.relpath(a.scored, ROOT).replace("\\", "/"), "bootstrap": {"n": N_BOOT, "seed": SEED},
           "comparators": not a.no_comparators, "cells": {}}
    for k in keys:
        for t in TASKS:
            sub = scored[(scored["model_key"] == k) & (scored["task"] == t)]
            if sub.empty:
                continue
            w = wide[(wide["model_key"] == k) & (wide["task"] == t)]
            blk = {d: decoding_block(sub[sub["decoding"] == d]) for d in ("vendor", "greedy")}
            blk["n_requests_expected"] = 2 * len(sample["tasks"][t]["items"])
            blk["diff_vendor_minus_greedy"] = paired(w, "vendor", "greedy")
            if comp is not None:
                for c in ("cot", "forced"):
                    blk[f"acc_{c}"] = share(int(w[c].sum()), int(w[c].notna().sum())) if c in w else None
                blk["diff_vendor_minus_cot"] = paired(w, "vendor", "cot")
                blk["diff_greedy_minus_cot"] = paired(w, "greedy", "cot")
                blk["diff_vendor_minus_forced"] = paired(w, "vendor", "forced")
            res["cells"][f"{t}/{k}"] = blk
    res["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "thinking_vendor.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    wide.to_csv(os.path.join(a.out, "per_cell.csv"), index=False)
    txt = summary(res)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return res


def fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def pct(b):
    return "--" if not b or b.get("pct") is None else f"{b['pct']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}]"


def dif(b):
    return "--" if not b or b.get("mean") is None else f"{b['mean']:+.1f} [{b['lo']:+.1f}, {b['hi']:+.1f}]"


def summary(res: dict) -> str:
    L = ["# Native reasoning under the vendors' sampling settings", "",
         "Three checkpoints whose greedy native outputs often ran to the 30,720-token cap, on 600 requests per task "
         "(two discovery configurations, 300 items). Vendor: the card's thinking-mode sampling settings "
         "(config/decoding_thinking_vendor.yaml). Greedy: the outputs of the same requests under greedy decoding. Shares in percent with "
         "Clopper-Pearson 95% intervals; differences in points over the paired cells with an item bootstrap.", "",
         "| Task | Checkpoint | Decoding | n | Cut at cap | Looping among cut | Strict accuracy | Mean tokens |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, blk in res["cells"].items():
        t, k = key.split("/", 1)
        for d in ("vendor", "greedy"):
            b = blk[d]
            L.append(f"| {TASKS[t]} | {k} | {d} | {b['n']} | {pct(b['cut'])} | {pct(b['looping_among_cut'])} | "
                     f"{pct(b['strict'])} | {fmt(b['mean_output_tokens'], 0)} |")
    L += ["", "| Task | Checkpoint | Vendor - greedy | Vendor - chain of thought | Greedy - chain of thought | "
              "Chain of thought | Forced direct |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, blk in res["cells"].items():
        t, k = key.split("/", 1)
        L.append(f"| {TASKS[t]} | {k} | {dif(blk['diff_vendor_minus_greedy'])} | {dif(blk.get('diff_vendor_minus_cot'))} | "
                 f"{dif(blk.get('diff_greedy_minus_cot'))} | {pct(blk.get('acc_cot'))} | {pct(blk.get('acc_forced'))} |")
    short = [k for k, b in res["cells"].items() if b["vendor"]["n"] < b["n_requests_expected"]]
    if short:
        L += ["", "Incomplete: " + ", ".join(short) + " (fewer vendor outputs than requests)."]
    L += ["", "The looping rule sees cycles of at most 10 distinct lines, so its shares are lower bounds."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scored", default=SCORED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-comparators", action="store_true",
                    help="leave out the chain-of-thought and forced-direct cells (no collection tables needed)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
