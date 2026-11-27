r"""Content scoring of the five-shot, eight-shot and native-thinking outputs, for the generality analysis. The content
reader (src/content_scoring.classify) is applied exactly as scripts/score_content.py applies it to the collection's
arms: the same gold tables, the same folder reader (src/d2_io.inspect_work_dir, which puts the run's prefill,
`\boxed{`, before a forced-style output), the scored table's rows kept one to one, finish reasons checked against it.
Native-thinking outputs are read after the last end-of-reasoning marker, as scripts/score_thinking.py scores them
(scripts/score_thinking.answer_bodies: an output cut before the marker has no answer). The eight-shot setting's
zero-shot baseline uses the collection's content table, except the 101 outputs of the four 4,096-token checkpoints
that the exemplar analysis scores at their cut (data/processed/exemplar/zero_shot_4k.parquet): their cut texts are
read here.

From the repository root:
    python scripts/score_content_added.py

Writes new files only (refuses to overwrite): data/processed/content/fewshot_{gsm8k,math,bbh}_content.parquet,
exemplar_gsm8k_content.parquet, exemplar_zero_shot_4k_content.parquet, thinking_gsm8k_content.parquet; and the record
results/generality/content_added.json (reader version, counts by class, runtime). --replace rewrites them, for
example after the content reader or the forced-reply stop (src/forced_stop.py) changes version; the record then names
the files it replaced.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import content_scoring as cs, d2_io  # noqa: E402

PROC = Path(ROOT) / "data" / "processed"
OUTS = Path(ROOT) / "data" / "d2_out"
CONTENT = PROC / "content"
RECORD = Path(ROOT) / "results" / "generality" / "content_added.json"
KEY = ["model_key", "arm", "config_id", "item_id"]
SCORED = ["strict_correct", "lenient_correct", "compliant"]
STRING_IDS = ("bbh", "math")
# setting: (task, scored table, output folders root, reads after the reasoning marker)
SETTINGS = {
    "fewshot_gsm8k": ("gsm8k", PROC / "fewshot" / "gsm8k_scored.parquet", OUTS / "fewshot" / "gsm8k", False),
    "fewshot_math": ("math", PROC / "fewshot" / "math_scored.parquet", OUTS / "fewshot" / "math", False),
    "fewshot_bbh": ("bbh", PROC / "fewshot" / "bbh_scored.parquet", OUTS / "fewshot" / "bbh", False),
    "exemplar_gsm8k": ("gsm8k", PROC / "exemplar" / "gsm8k_scored.parquet", OUTS / "exemplar" / "gsm8k", False),
    "thinking_gsm8k": ("gsm8k", PROC / "thinking" / "gsm8k_scored.parquet", OUTS / "thinking" / "gsm8k", True),
}


def reader_version():
    p = Path(ROOT) / "src" / "content_scoring.py"
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short", "--", "src/content_scoring.py"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src/content_scoring.py"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    return {"module": "src/content_scoring.py", "last_commit": commit, "uncommitted_changes": dirty,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def classify_rows(texts, finish, items, gold, task):
    res = [cs.classify(t, f, gold[i][0], task, gold[i][1], opts=gold[i][2], task_name=gold[i][3])
           for t, f, i in zip(texts, finish, items)]
    return [r[0] for r in res]


def score_setting(name, gold_cache, t0):
    task, scored_path, root, thinking = SETTINGS[name]
    import score_content
    gold = gold_cache.setdefault(task, score_content.gold_table(task))
    sc = pd.read_parquet(scored_path)
    sc["config_id"] = sc["config_id"].astype(str)
    sc["item_id"] = sc["item_id"].astype(str if task in STRING_IDS else "int64")
    parts = []
    for k in sorted(sc["model_key"].unique()):
        wd = root / k
        out, info = d2_io.inspect_work_dir(wd, string_item_ids=(task in STRING_IDS))
        mine = sc[sc["model_key"] == k][KEY + SCORED + ["finish_reason"]]
        j = mine.merge(out[KEY + ["output_text", "finish_reason"]], on=KEY, how="left", suffixes=("", "_out"),
                       validate="one_to_one")
        if j["output_text"].isna().any():
            raise SystemExit(f"{wd}: {int(j['output_text'].isna().sum())} scored rows have no output")
        if (j["finish_reason"] != j["finish_reason_out"]).any():
            raise SystemExit(f"{wd}: finish reasons differ from the scored table")
        texts = j["output_text"].astype(str).tolist()
        fin = j["finish_reason"].astype(str).tolist()
        if thinking:
            import score_thinking
            texts, _ = score_thinking.answer_bodies(texts, fin)
        j["content_class"] = classify_rows(texts, fin, j["item_id"], gold, task)
        j["content_correct"] = (j["content_class"] == "correct").astype("int8")
        parts.append(j.drop(columns=["output_text", "finish_reason_out"]))
        print(f"{name} {k}: {len(j)} outputs, prefill {info['prefill']} ({time.time() - t0:.0f} s)", flush=True)
    tab = pd.concat(parts, ignore_index=True)
    if len(tab) != len(sc):
        raise SystemExit(f"{name}: {len(tab)} classified rows against {len(sc)} scored rows")
    return tab


def score_cut_zero_shot():
    """The 101 cut zero-shot outputs of the exemplar analysis's four 4,096-token checkpoints, read from their cut texts."""
    import score_content
    gold = score_content.gold_table("gsm8k")
    c = pd.read_parquet(PROC / "exemplar" / "zero_shot_4k.parquet")
    c["config_id"] = c["config_id"].astype(str)
    c["item_id"] = c["item_id"].astype("int64")
    c["content_class"] = classify_rows(c["output_text_cut"].astype(str).tolist(), c["finish_reason"].astype(str).tolist(),
                                       c["item_id"], gold, "gsm8k")
    c["content_correct"] = (c["content_class"] == "correct").astype("int8")
    return c[KEY + SCORED + ["finish_reason", "content_class", "content_correct"]]


def counts(tab):
    by = tab.groupby("content_class").size().reindex(list(cs.CLASSES), fill_value=0)
    return {"n_outputs": int(len(tab)), **{c: int(v) for c, v in by.items()},
            "strict_correct_not_content_correct": int(((tab["strict_correct"] == 1) & (tab["content_correct"] == 0)).sum()),
            "content_correct": int(tab["content_correct"].sum()), "strict_correct": int(tab["strict_correct"].sum())}


def target(name):
    return CONTENT / f"{name}_content.parquet"


def run(names, replace=False):
    t0 = time.time()
    todo = list(names) + ["exemplar_zero_shot_4k"]
    exists = [os.path.relpath(target(n), ROOT) for n in todo if target(n).exists()]
    if exists and not replace:
        raise SystemExit(f"refused: {exists} exist; this script writes new files only (--replace rewrites them)")
    from src import forced_stop
    rec = {"created": time.strftime("%Y-%m-%d %H:%M"), "reader": reader_version(),
           "content_scorer_version": cs.CONTENT_SCORER_VERSION, "forced_stop": forced_stop.VERSION,
           "replaced": [e.replace("\\", "/") for e in exists], "settings": {},
           "note": "content reader applied as scripts/score_content.py applies it to the collection's arms "
                   "(module doc); thinking outputs read after the last end-of-reasoning marker"}
    gold_cache = {}
    for name in names:
        tab = score_setting(name, gold_cache, t0)
        tab.to_parquet(target(name), index=False)
        rec["settings"][name] = {"path": str(target(name).relative_to(ROOT)).replace("\\", "/"),
                                 "scored_table": str(SETTINGS[name][1].relative_to(ROOT)).replace("\\", "/"),
                                 **counts(tab)}
    cut = score_cut_zero_shot()
    cut.to_parquet(target("exemplar_zero_shot_4k"), index=False)
    rec["settings"]["exemplar_zero_shot_4k"] = {"path": str(target("exemplar_zero_shot_4k").relative_to(ROOT)).replace(
        "\\", "/"), "scored_table": "data/processed/exemplar/zero_shot_4k.parquet", **counts(cut)}
    rec["runtime_seconds"] = round(time.time() - t0, 1)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(RECORD, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    ap.add_argument("--replace", action="store_true", help="rewrite existing content tables")
    a = ap.parse_args()
    run(a.settings, replace=a.replace)
