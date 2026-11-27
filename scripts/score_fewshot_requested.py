r"""Score the five-shot requested-direct arm in the order the prediction rules need: discovery draws first, validation
after the predictions are saved.

    python scripts/score_fewshot_requested.py --check-runs        # run check of the folders copied back
    python scripts/score_fewshot_requested.py --stage discovery   # the four discovery configurations
    python scripts/run_fewshot_requested.py --stage discovery     # saves the predictions
    python scripts/score_fewshot_requested.py --stage all         # refused until they are saved

Reads data/d2_out/fewshot_requested/collection/<key> (copied back from the GPU machine), checks that each folder ran its
recorded request file to completion (build_arm_wording.pilot_check: one complete run of the whole file, the decoding of
config/decoding.yaml in its log, no prefill, no prompt-render mismatch), and scores GSM8K with the boxed
scorer (manifests/gsm8k_scorer.json, as the main collection's requested arm) and BIG-Bench Hard with bbh-boxed
(src/bbh.score_outputs), strict and lenient, box closure beside them. Writes
data/processed/fewshot_requested/<stage>_scored.parquet and a .meta.json with the folders, run ids, request hashes and,
at stage all, the sha256 of manifests/predictions_fewshot_requested.json. --stage discovery keeps only the rows of the
discovery configurations: no validation row is scored before the predictions are saved.

Dry runs: --base DIR (outside the repository), --allow-test-backend, --items-dir and --synthetic SEED (seeded
correctness in place of the echo backend's empty answers). --synthetic is refused inside the repository.
Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import d2_io, fewshot_requested as fr  # noqa: E402

COLS = ["model_key", "arm", "task", "config_id", "item_id", "strict_correct", "lenient_correct", "compliant",
        "box_closed", "finish_reason", "n_output_tokens", "prompt_sha256"]


def in_repo(base: Path) -> bool:
    try:
        Path(base).resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def items(task: str, items_dir=None) -> pd.DataFrame:
    """Target items with gold answers (and answer_type on BIG-Bench Hard); item_id as a string."""
    if items_dir:
        df = pd.read_parquet(Path(items_dir) / f"{task}_target.parquet")
    elif task == "gsm8k":
        df, _ = d2_io.load_items(split="test")
    else:
        from src import bbh_d2
        df, _ = bbh_d2.load_items("target")
    df = df.copy()
    df["item_id"] = df["item_id"].astype(str)
    return df.reset_index(drop=True)


def score_task(out: pd.DataFrame, task: str, it: pd.DataFrame, spec) -> pd.DataFrame:
    from src import arm_wording as aw, bbh
    if task == "gsm8k":
        s = aw.score(out, "gsm8k", it, spec)        # requested rows: no cut, boxed scorer
    else:
        s = bbh.score_outputs(out.assign(item_id=out["item_id"].astype(str)), it)
        s["box_closed"] = aw.box_closed(s["output_text"].astype(str).tolist())
    s["item_id"] = s["item_id"].astype(str)
    return s


def synthetic(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seeded stand-in correctness for dry runs, a function of the row's key alone (stages agree)."""
    t = table.copy()
    key = t[["model_key", "task", "config_id", "item_id"]].astype(str)
    h = pd.util.hash_pandas_object(key, index=False, hash_key=f"{seed:016d}"[-16:]).to_numpy()
    u = (h >> np.uint64(11)).astype(float) / float(2 ** 53)
    hm = pd.util.hash_pandas_object(t[["model_key"]].astype(str), index=False, hash_key="0000000000000fsr").to_numpy()
    p = 0.2 + 0.6 * ((hm >> np.uint64(11)).astype(float) / float(2 ** 53))
    t["strict_correct"] = (u < p).astype(int)
    t["lenient_correct"] = t["strict_correct"]
    return t


def check_runs(paths: fr.Paths, rec: dict) -> dict:
    import build_arm_wording as baw
    folders = baw.run_folders(paths.outputs / "collection", rec["models"])
    present = {**rec, "models": {k: v for k, v in rec["models"].items() if k in folders}}
    rows = baw.pilot_check(folders, present, None)
    n = sum(r["passes"] for r in rows)
    lines = ["# Five-shot requested arm: run check", "",
             f"{n} of {len(rows)} models copied back pass ({len(rec['models'])} in the build record). A model passes "
             "when one complete run wrote every request of its file, the log carries the decoding of "
             "config/decoding.yaml (no prefill, no stop string, the arm caps) and no prompt-render mismatch. No "
             "correctness was read.", "", "| Model | Passes | GPU | Problems |", "|---|---|---|---|"]
    lines += [f"| {r['model_key']} | {'yes' if r['passes'] else 'no'} | {', '.join(r['gpu'])} | "
              f"{'; '.join(r['problems'])} |" for r in rows]
    paths.results.mkdir(parents=True, exist_ok=True)
    for name in ("run_check.md", "run_check_clean.md"):
        (paths.results / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_check": str(paths.results / "run_check.md"), "passed": n, "of": len(rows),
            "not_yet_copied": sorted(set(rec["models"]) - set(folders))}


def run(args) -> dict:
    paths = fr.Paths(args.base)
    dry = args.synthetic is not None or args.items_dir or args.allow_test_backend
    if dry and in_repo(paths.base):
        raise fr.FewshotRequestedError("--synthetic, --items-dir and --allow-test-backend are for dry runs; pass --base "
                                       "outside the repository")
    rec_p = paths.requests / "collection" / "build_record.json"
    if args.stage == "all":
        pred_sha = fr.check_predictions(paths)          # first: nothing is read before the order holds
    else:
        pred_sha = None
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    if args.check_runs:
        return check_runs(paths, rec)
    keep = {t: set(v["discovery"]) if args.stage == "discovery" else set(v["discovery"] + v["validation"])
            for t, v in rec["configurations"].items()}
    root = paths.outputs / "collection"
    folders = {k: root / k for k in rec["models"] if (root / k).is_dir()}
    missing = sorted(set(rec["models"]) - set(folders))
    if missing:
        raise fr.FewshotRequestedError(f"no output folder for {missing}")
    from src import screening
    spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
    its = {t: items(t, args.items_dir) for t in fr.TASKS}
    parts, meta = [], {}
    for k, d in sorted(folders.items()):
        out, info = d2_io.inspect_work_dir(d, allow_test_backend=args.allow_test_backend, string_item_ids=True)
        shas = {r["requests_sha256"] for r in info["runs"]}
        if shas != {rec["models"][k]["requests_sha256"]}:
            raise fr.FewshotRequestedError(f"{d}: the runs read another request file")
        if len(out) != rec["models"][k]["n_requests"]:
            raise fr.FewshotRequestedError(f"{d}: {len(out)} outputs, the request file has "
                                           f"{rec['models'][k]['n_requests']}")
        if (out["arm"].astype(str) != fr.ARM).any():
            raise fr.FewshotRequestedError(f"{d}: an output of another arm")
        if any(v for v in (info.get("prefill") or {}).values()):
            raise fr.FewshotRequestedError(f"{d}: ran with a prefill")
        parsed = [fr.parse_job_config_id(c) for c in out["config_id"].astype(str)]
        out = out.assign(task=[p[0] for p in parsed], config_id=[p[1] for p in parsed])
        out = out[[c in keep[t] for t, c in zip(out["task"], out["config_id"])]]
        for t in fr.TASKS:
            sub = out[out["task"] == t]
            if len(sub):
                parts.append(score_task(sub, t, its[t], spec))
        meta[k] = {"folder": d2_io.rel_path(d), "run_ids": [r["run_id"] for r in info["runs"]],
                   "requests_sha256": rec["models"][k]["requests_sha256"], "gpu": info.get("gpu_type"),
                   "test_backend": info.get("test_backend")}
    table = pd.concat([p[COLS] for p in parts], ignore_index=True)
    if args.synthetic is not None:
        table = synthetic(table, args.synthetic)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out_path = paths.processed / f"{args.stage}_scored.parquet"
    table.to_parquet(out_path, index=False)
    md = {"stage": args.stage, "configurations": {t: sorted(v) for t, v in keep.items()}, "folders": meta,
          "collection_record_sha256": fr.sha256_file(rec_p), "predictions_sha256": pred_sha,
          "synthetic_seed": args.synthetic,
          "scorers": {"gsm8k": "manifests/gsm8k_scorer.json", "bbh": "src/bbh.py (bbh-boxed)"}}
    fr.write_json(out_path.with_suffix(".meta.json"), md)
    return {"out": str(out_path), "rows": int(len(table)), "models": len(meta), "stage": args.stage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=("discovery", "all"))
    g.add_argument("--check-runs", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--items-dir")
    ap.add_argument("--allow-test-backend", action="store_true")
    ap.add_argument("--synthetic", type=int, metavar="SEED")
    args = ap.parse_args(argv)
    try:
        res = run(args)
    except (fr.FewshotRequestedError, d2_io.D2IOError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
