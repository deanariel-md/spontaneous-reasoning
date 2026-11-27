r"""Size ladder: score one ladder model's outputs with the main panel's scorers, strict
and content, twice:
  primary        every output cut at the token level to the panel's cap of its arm (the runner's cut copy:
                 output_text_cut, finish_reason_cut length when the uncut output ran past the cap), the copy that enters
                 the ladder results beside the 29-model panel  -> data/processed/ladder/<task>/<key>.parquet
  supplementary  the uncut outputs at the raised cap (7,000)  -> data/processed/ladder_uncut/<task>/<key>.parquet
The forced arms are never cut (their caps are the panel's), so both tables hold the same rows for them.

Reads data/d2_out/ladder2/<run key>/<task>/ (copied back from $S/out/ladder; <run key> is the base key or its -dk
fallback, as data/d2_out/ladder2/chosen/<key> says) through d2_io.inspect_work_dir, which refuses an incomplete folder
and puts the forced arms' prefill \boxed{ in front of their outputs, as for the panel. Then:
  - strict and lenient: scoring.score_outputs with manifests/gsm8k_scorer.json (GSM8K), bbh.score_outputs,
    math_task.score_outputs, hotpot_task.score_outputs (as scripts/score_d2.py, score_forced_direct.py, score_hotpot.py);
  - content: content_scoring.classify row by row (GSM8K, MATH-500, BIG-Bench Hard, as scripts/score_content.py), and
    hotpot_task.score_content for HotpotQA (as scripts/score_hotpot_content.py).
Writes results/ladder/scored_<key>.json: rows per arm, the share of outputs cut at the panel's cap per task and arm,
outputs that reached the raised cap, and the content classes of both readings. No accuracy is computed here: the
ladder's analysis is written before its scores are read.

PowerShell:  .\.venv\Scripts\python.exe scripts\score_ladder.py --model llama-3.1-70b-instruct-fp8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402

from src import d2_io, generation as gen, ladder as L  # noqa: E402

OUT_ROOT = L.ROOT / "data" / "d2_out" / "ladder2"
PROC = L.ROOT / "data" / "processed" / "ladder"
PROC_UNCUT = L.ROOT / "data" / "processed" / "ladder_uncut"
RES = L.ROOT / "results" / "ladder"


def run_key(key: str, out_root: Path = OUT_ROOT) -> str:
    p = out_root / "chosen" / key
    return p.read_text(encoding="utf-8").strip() if p.exists() else key


def strict(task: str, out: pd.DataFrame) -> pd.DataFrame:
    if task == "gsm8k":
        from src import gsm8k, scoring
        spec, _ = d2_io.load_registered_scorer(str(L.ROOT / "manifests" / "gsm8k_scorer.json"))
        items, _ = d2_io.load_items(None, split="test")
        return scoring.score_outputs(out, gsm8k.gold_lookup(items), spec)
    if task == "bbh":
        from src import bbh, bbh_d2
        items, _ = bbh_d2.load_items("target")
        return bbh.score_outputs(out, items)
    if task == "math":
        from src import math_d2, math_task
        items, _ = math_d2.load_items("target")
        return math_task.score_outputs(out, items)
    from src import hotpot_d2, hotpot_task
    items, _ = hotpot_d2.load_items("target")
    return hotpot_task.score_outputs(out, items)


def content(task: str, sc: pd.DataFrame) -> pd.DataFrame:
    if task == "hotpot":
        from src import hotpot_d2, hotpot_task
        items, _ = hotpot_d2.load_items("target")
        c = hotpot_task.score_content(sc[["item_id", "output_text"]].assign(item_id=sc["item_id"].astype(str)), items)
        for col in ("content_answer", "content_correct", "content_f1", "content_source"):
            sc[col] = c[col].values
        return sc
    sys.path.insert(0, str(L.ROOT / "scripts"))
    import score_content as SC   # gold_table, the panel's reading of each task's gold
    from src import content_scoring as cs
    gold = SC.gold_table(task)
    key = (lambda i: int(i)) if task == "gsm8k" else (lambda i: str(i))
    res = [cs.classify(t, f, gold[key(i)][0], task, gold[key(i)][1], opts=gold[key(i)][2], task_name=gold[key(i)][3])
           for t, f, i in zip(sc["output_text"], sc["finish_reason"], sc["item_id"])]
    sc["content_class"] = [r[0] for r in res]
    sc["content_correct"] = (sc["content_class"] == "correct").astype("int8")
    return sc


def cut_copy(out: pd.DataFrame) -> pd.DataFrame:
    """The primary reading: each output replaced by its copy cut at the panel's cap. Forced arms (prefill already put
    in front by d2_io) are never cut, so their text stays as read."""
    c = out.copy()
    m = c["cut_at_panel_cap"].astype(bool)
    c.loc[m, "output_text"] = c.loc[m, "output_text_cut"]
    c["finish_reason"] = c["finish_reason_cut"]
    c["n_output_tokens"] = c["n_output_tokens_cut"]
    return c


def score_model(key: str, out_root: Path = OUT_ROOT, proc: Path = PROC, proc_uncut: Path = PROC_UNCUT,
                res_dir: Path = RES, tasks=L.TASKS, allow_test_backend: bool = False) -> dict:
    rk = run_key(key, out_root)
    summary = {"model_key": key, "run_key": rk, "created": time.strftime("%Y-%m-%d %H:%M"), "tasks": {}}
    for task in tasks:
        d = out_root / rk / task
        out, info = d2_io.inspect_work_dir(d, string_item_ids=(task != "gsm8k"), allow_test_backend=allow_test_backend)
        missing = [c for c in gen.CUT_COLUMNS if c not in out.columns]
        if missing:
            raise d2_io.ScoreInputError(f"{d}: outputs lack the cut columns {missing}")
        t = {"rows": int(len(out)), "folder": f"{rk}/{task}",
             "by_arm": {a: int(v) for a, v in out["arm"].value_counts().sort_index().items()},
             "share_cut_at_panel_cap": {a: round(float(g["cut_at_panel_cap"].astype(bool).mean()), 5)
                                        for a, g in out.groupby("arm")},
             "cut_at_panel_cap": {a: int(g["cut_at_panel_cap"].astype(bool).sum()) for a, g in out.groupby("arm")},
             "at_raised_cap": {a: int((g["finish_reason"] == "length").sum()) for a, g in out.groupby("arm")},
             "panel_cap": {a: int(g["panel_cap"].iloc[0]) for a, g in out.groupby("arm")}}
        for label, frame, dest in (("cut", cut_copy(out), proc), ("uncut", out, proc_uncut)):
            sc = content(task, strict(task, frame))
            sc["model_key"] = key
            sc["run_key"] = rk
            sc["reading"] = label
            keep = [c for c in sc.columns if c not in ("output_text", "output_text_cut", "output_token_ids_cut")]
            p = dest / task / f"{key}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            sc[keep].to_parquet(p, index=False)
            if "content_class" in sc:
                t[f"content_classes_{label}"] = {a: {k: int(v) for k, v in g["content_class"].value_counts().items()}
                                                 for a, g in sc.groupby("arm")}
            print(f"{key} {task} ({label}): {len(sc)} rows scored -> {p}", flush=True)
        summary["tasks"][task] = t
    L.write_json(res_dir / f"scored_{key}.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    a = ap.parse_args(argv)
    score_model(a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
