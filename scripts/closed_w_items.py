"""Item draws of the Qwen4-Max run (imported by scripts/closed_w.py and scripts/closed_w_full.py).

Design B: every template and a fixed seeded sample of items per benchmark, 50 or 100 (the 50 are the first 50 of the
100). BIG-Bench Hard is drawn per task, 2 per task for 50 and 4 for 100. The run used the 100-item draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BBH_PER_TASK = {50: 2, 100: 4}


def label_seed(label: str) -> int:
    from src.decision import label_seed as ls
    return ls(label)


def strongest_models() -> list[tuple[str, float]]:
    """Panel models by their Epoch Capabilities Index value, highest first (manifests/d2_panel*.json)."""
    eci = {}
    for f in ("d2_panel", "d2_panel_extension"):
        for r in json.loads((ROOT / "manifests" / f"{f}.json").read_text(encoding="utf-8"))["panel"]:
            eci[r["repo"].split("/")[-1].lower()] = float(r["eci"])
    return sorted(eci.items(), key=lambda kv: -kv[1])


def bbh_task(item_id: str) -> str:
    return item_id.split("/", 1)[0]


def sample_items(task: str, items: list[str], n: int) -> list[str]:
    """Nested seeded samples: a seeded permutation of the sorted ids; the first n (for BBH, the first k per task)."""
    items = sorted(set(items))
    if task == "bbh":
        by: dict[str, list[str]] = {}
        for i in items:
            by.setdefault(bbh_task(i), []).append(i)
        out = []
        for k in sorted(by):
            rng = np.random.default_rng(label_seed(f"closed_w/b/bbh/{k}"))
            perm = [by[k][int(j)] for j in rng.permutation(len(by[k]))]
            out += perm[:BBH_PER_TASK[n]]
        return sorted(out)
    rng = np.random.default_rng(label_seed(f"closed_w/b/{task}"))
    perm = [items[int(j)] for j in rng.permutation(len(items))]
    return sorted(perm[:n])
