r"""Score the default- and firm-clause run in the order the prediction rules need: discovery draws first, validation
after the predictions are saved.

    python scripts/build_arm_wording_extra.py forecast        # saved first, from token counts
    python scripts/score_arm_wording_extra.py --check-runs     # run check of the collection folders
    python scripts/score_arm_wording_extra.py --stage discovery
    python scripts/run_arm_wording_extra.py --stage discovery  # saves the discovery predictions
    python scripts/score_arm_wording_extra.py --stage all      # refused until they exist
    python scripts/run_arm_wording_extra.py

Both stages refuse unless manifests/predictions_arm_wording_extra.json exists and matches its .sha256 and its content
hash; --stage all also refuses unless the discovery predictions are saved and unchanged. Scoring as in
scripts/score_arm_wording.py (src/arm_wording.score: GSM8K's boxed scorer with the forced text cut after the first "}", MATH-500 by
src/math_task.py; strict and lenient; box closure). Writes data/processed/arm_wording_extra/<stage>_scored.parquet and
a .meta.json with the forecast's hash.

Dry runs: --base DIR (outside the repository), --allow-test-backend, --items-dir, --synthetic SEED (seeded stand-in
correctness, as scripts/score_arm_wording.py). Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, arm_wording_extra as ax, d2_io  # noqa: E402
import build_arm_wording as baw  # noqa: E402
import score_arm_wording as saw  # noqa: E402


def run(args) -> dict:
    paths = ax.Paths(args.base)
    if args.synthetic is not None and baw.in_repo(paths.base):
        raise ax.ExtraError("--synthetic is for dry runs; pass --base outside the repository")
    rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    if args.check_runs:
        folders = baw.run_folders(paths.outputs / "collection", rec["models"])
        present = {**rec, "models": {k: v for k, v in rec["models"].items() if k in folders}}
        rows = baw.pilot_check(folders, present, aw.load_config())
        n = baw.write_pilot_md(rows, paths, "collection run")
        return {"run_check": str(paths.results / "collection_run_check.md"), "passed": n, "of": len(rows),
                "not_yet_copied": sorted(set(rec["models"]) - set(folders))}
    fc_sha = ax.check_forecast(paths)
    pred_sha = ax.check_discovery_predictions(paths) if args.stage == "all" else None
    keep = {t: set(v["discovery"]) if args.stage == "discovery" else set(v["discovery"] + v["validation"])
            for t, v in rec["configurations"].items()}
    folders = baw.run_folders(paths.outputs / "collection", rec["models"])
    missing = sorted(set(rec["models"]) - set(folders))
    if missing:
        raise ax.ExtraError(f"no output folder for {missing}")
    from src import screening
    spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
    items = {t: baw.items(t, "target", args) for t in aw.TASKS}
    parts, meta = [], {}
    for k, d in sorted(folders.items()):
        out, info = d2_io.inspect_work_dir(d, allow_test_backend=args.allow_test_backend, string_item_ids=True)
        if {r["requests_sha256"] for r in info["runs"]} != {rec["models"][k]["requests_sha256"]}:
            raise ax.ExtraError(f"{d}: the runs read another request file")
        if len(out) != rec["models"][k]["n_requests"]:
            raise ax.ExtraError(f"{d}: {len(out)} outputs, the request file has {rec['models'][k]['n_requests']}")
        parsed = out["config_id"].map(aw.parse_config_id)
        out = out.assign(task=[p[0] for p in parsed], conf=[p[1] for p in parsed])
        out = out[[c in keep[t] for t, c in zip(out["task"], out["conf"])]]
        for t in aw.TASKS:
            parts.append(aw.score(out[out["task"] == t].drop(columns=["task", "conf"]), t, items[t], spec))
        meta[k] = {"folder": d2_io.rel_path(d), "run_ids": [r["run_id"] for r in info["runs"]],
                   "requests_sha256": rec["models"][k]["requests_sha256"], "gpu": info["gpu_type"],
                   "test_backend": info["test_backend"]}
    cols = ["model_key", "arm", "config_id", "item_id", "strict_correct", "lenient_correct", "compliant",
            "box_closed", "finish_reason", "n_output_tokens", "prompt_sha256"]
    table = pd.concat([p[cols] for p in parts], ignore_index=True)
    if args.synthetic is not None:
        table = saw.synthetic(table, args.synthetic)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out_path = paths.processed / f"{args.stage}_scored.parquet"
    table.to_parquet(out_path, index=False)
    md = {"stage": args.stage, "configurations": {t: sorted(v) for t, v in keep.items()}, "folders": meta,
          "collection_record_sha256": aw.sha256_file(paths.requests / "collection" / "build_record.json"),
          "forecast_sha256": fc_sha, "predictions_sha256": pred_sha, "synthetic_seed": args.synthetic}
    aw.write_json(out_path.with_suffix(".meta.json"), md)
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
    except (ax.ExtraError, aw.ArmWordingError, d2_io.D2IOError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
