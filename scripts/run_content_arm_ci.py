r"""Interval for the arm-switch limit under content scoring.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_content_arm_ci.py

results/protocol_ceiling/protocol_ceiling.json holds the plug-in limit of the content/requested/points against
content/forced/points pair (`limits`) but bootstraps only the headline pair (strict). This script reads the same
cached benefit matrices as scripts/run_protocol_ceiling.py, recomputes the plug-in (checked against the saved value)
and bootstraps it the way the headline is bootstrapped: configurations resampled with replacement, 2,000 resamples,
percentile interval. It is a new bootstrap with its own seed label ("protocol-ceiling", task, "boot-content"); it does
not touch protocol_ceiling.json or any earlier random stream.

Output: results/protocol_ceiling/content_arm_limit.json, content_arm_limit.md and content_arm_limit_clean.md.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import protocol_ceiling as pcl, sampling  # noqa: E402

OUT = os.path.join(ROOT, "results", "protocol_ceiling")
PAIR = (("content", "requested", "points"), ("content", "forced", "points"))
PAIR_KEY = "content/requested/points | content/forced/points"
N_BOOT = 2000


def pair_result(task, Ya, Yb, saved_plugin, n_boot=N_BOOT):
    plug = pcl.limit_plugin(Ya, Yb)
    if saved_plugin is not None and abs(plug - saved_plugin) > 1e-9:
        raise ValueError(f"{task}: recomputed plug-in {plug} differs from the saved {saved_plugin}")
    b = pcl.bootstrap_limit(Ya, Yb, sampling.rng_for("protocol-ceiling", task, "boot-content"), n_boot)
    return {"plugin": plug, "ci95": b["ci95"], "n_boot": b["n_boot"], "D": int(np.asarray(Ya).shape[0]),
            "saved_plugin": saved_plugin}


def main():
    import run_protocol_ceiling as rpc
    saved = json.load(open(os.path.join(OUT, "protocol_ceiling.json")))
    res = {"analysis": "arm-switch limit under content scoring, with a configuration bootstrap",
           "pair": PAIR_KEY, "tasks": {}}
    for task in rpc.TASKS:
        D, _, _, n = rpc.load(task)
        Y, _ = rpc.benefits(D, n)
        res["tasks"][task] = pair_result(task, Y[PAIR[0]], Y[PAIR[1]], saved["tasks"][task]["limits"][PAIR_KEY]["plugin"])
    with open(os.path.join(OUT, "content_arm_limit.json"), "w") as f:
        json.dump(res, f, indent=1)
    L = ["# Arm-switch limit under content scoring", "",
         "Code `scripts/run_content_arm_ci.py`. The pair is content-scored "
         "requested against content-scored forced, in points, on every distinct configuration of the discovery and "
         "validation draws. The plug-in equals the value in `protocol_ceiling.json`. The interval is a new percentile "
         f"bootstrap over configurations ({N_BOOT} resamples, the headline's recipe, its own seed label); it leaves "
         "`protocol_ceiling.json` unchanged.", "",
         "| Task | D | Plug-in limit | 95% interval |", "|---|---:|---:|---|"]
    for task, r in res["tasks"].items():
        L.append(f"| {task} | {r['D']} | {r['plugin']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    txt = "\n".join(L) + "\n"
    for name in ("content_arm_limit.md", "content_arm_limit_clean.md"):
        with open(os.path.join(OUT, name), "w", newline="\n") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
