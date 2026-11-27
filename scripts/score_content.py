r"""Content-level rescoring of the three arms (src/content_scoring.py).

From the repository root:
    python scripts/score_content.py
    python scripts/score_content.py --tasks math      # MATH-500

Reads every output folder of the GSM8K-29 and BIG-Bench Hard-28 collections (requested direct and chain of thought)
and of the forced-direct run (prefill prepended, as src/d2_io.inspect_work_dir does), keeps exactly the rows of the
scored tables the analyses use, classifies each output (correct, wrong, unreadable, malformed box, truncated) and
writes data/processed/content/<task>_content.parquet with the scored table's strict, lenient and compliance columns
beside content_class and content_correct. The class counts by model and arm go to results/content_scoring/; a run on
some tasks keeps the other tasks' entries of classes.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import content_scoring as cs, d2_io  # noqa: E402

OUT = Path(ROOT) / "data" / "processed" / "content"
RES = Path(ROOT) / "results" / "content_scoring"
KEY = ["model_key", "arm", "config_id", "item_id"]
SCORED = ["strict_correct", "lenient_correct", "compliant"]
ARM_NAMES = {"direct_answer": "requested direct", "zero_shot_cot": "chain of thought", "forced_direct": "forced direct"}
TASKS = {"gsm8k": "gsm8k30", "bbh": "bbh", "math": "math"}   # task: its results/dataset2/<label>.json
DEFAULT_TASKS = ("gsm8k", "bbh")
STRING_IDS = ("bbh", "math")


def folders(task, panel):
    d = Path(ROOT) / "data" / "d2_out"
    if task == "gsm8k":
        col = [d / "collection" / m if (d / "collection" / m).is_dir() else d / "collection_ext" / m for m in panel]
    elif task == "math":      # G's rotation copies the collection and the forced arm under data/d2_out/math/
        return [d / "math" / "collection" / m for m in panel] + [d / "math" / "forced" / m for m in panel]
    else:
        col = [d / "bbh" / "collection" / m for m in panel]
    return col + [d / "forced" / task / m for m in panel]


def scored_table(task):
    p = Path(ROOT) / "data" / "processed"
    if task == "gsm8k":
        parts = [pd.read_parquet(p / "gsm8k" / "scored.parquet")] + [pd.read_parquet(f) for f in
                                                                    sorted((p / "gsm8k_ext").glob("scored_*.parquet"))]
    else:
        parts = [pd.read_parquet(p / task / "scored.parquet")]
    parts.append(pd.read_parquet(p / "forced" / f"{task}_scored.parquet"))
    t = pd.concat(parts, ignore_index=True)
    t["config_id"] = t["config_id"].astype(str)
    t["item_id"] = t["item_id"].astype(str if task in STRING_IDS else "int64")
    return t


def gold_table(task):
    if task == "gsm8k":
        from src import gsm8k, scoring
        it = gsm8k.load_gsm8k("test")
        return {int(i): (scoring.normalize_number(g), None, None, None) for i, g in zip(it["item_id"],
                                                                                      it["gold_answer"])}
    if task == "math":
        from src import math_d2
        it, _ = math_d2.load_items("target")
        return {str(i): (g, None, None, None) for i, g in zip(it["item_id"], it["gold_answer"])}
    from src import bbh, bbh_d2
    it, _ = bbh_d2.load_items("target")
    return {str(i): (g, bbh.AnswerType.from_label(a), cs.options(q), k)
            for i, g, a, q, k in zip(it["item_id"], it["gold_answer"], it["answer_type"], it["question"], it["task"])}


def run(tasks=DEFAULT_TASKS):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in tasks:
        res = TASKS[task]
        panel = json.load(open(os.path.join(ROOT, "results", "dataset2", f"{res}.json"), encoding="utf-8"))["models"]
        sc = scored_table(task)
        sc = sc[sc["model_key"].isin(panel)]
        gold = gold_table(task)
        parts = []
        for wd in folders(task, panel):
            out, info = d2_io.inspect_work_dir(wd, string_item_ids=(task in STRING_IDS))
            mine = sc[sc["model_key"] == info["model_key"]]
            mine = mine[mine["arm"].isin(out["arm"].unique())][KEY + SCORED + ["finish_reason"]]
            j = mine.merge(out[KEY + ["output_text", "finish_reason"]], on=KEY, how="left",
                           suffixes=("", "_out"), validate="one_to_one")
            if j["output_text"].isna().any():
                raise SystemExit(f"{wd}: {int(j['output_text'].isna().sum())} scored rows have no output")
            if (j["finish_reason"] != j["finish_reason_out"]).any():
                raise SystemExit(f"{wd}: finish reasons differ from the scored table")
            res_ = [cs.classify(t, f, gold[i][0], task, gold[i][1], opts=gold[i][2], task_name=gold[i][3])
                    for t, f, i in zip(j["output_text"], j["finish_reason"], j["item_id"])]
            j["content_class"] = [r[0] for r in res_]
            j["content_correct"] = (j["content_class"] == "correct").astype("int8")
            parts.append(j.drop(columns=["output_text", "finish_reason_out"]))
            print(f"{task} {wd.parent.name}/{wd.name}: {len(j)} outputs ({time.time() - t0:.0f} s)", flush=True)
        tab = pd.concat(parts, ignore_index=True)
        if len(tab) != len(sc):
            raise SystemExit(f"{task}: {len(tab)} classified rows against {len(sc)} scored rows")
        # strict correct implies content correct: the strict reading is a stricter reading of the same box
        bad = (tab["strict_correct"] == 1) & (tab["content_correct"] == 0)
        tab.to_parquet(OUT / f"{task}_content.parquet", index=False)
        counts = (tab.groupby(["model_key", "arm", "content_class"]).size().unstack(fill_value=0)
                  .reindex(columns=list(cs.CLASSES), fill_value=0))
        counts["compliant"] = tab.groupby(["model_key", "arm"])["compliant"].sum()
        counts["n"] = tab.groupby(["model_key", "arm"]).size()
        counts.reset_index().to_csv(RES / f"classes_{task}.csv", index=False)
        summary[task] = {"n_outputs": int(len(tab)), "strict_correct_not_content_correct": int(bad.sum()),
                         "by_arm": {a: {c: int(v) for c, v in g[list(cs.CLASSES) + ["compliant", "n"]].sum().items()}
                                    for a, g in counts.groupby(level="arm")}}
    path = RES / "classes.json"
    prev = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    created = {**prev.get("created_by_task", {}), **{t: time.strftime("%Y-%m-%d %H:%M") for t in summary}}
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M"), "created_by_task": created, "classes": list(cs.CLASSES),
               "tasks": {**prev.get("tasks", {}), **summary}}, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS))
    run(tuple(dict.fromkeys(ap.parse_args().tasks)))
