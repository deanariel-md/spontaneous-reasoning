r"""Score the arm-wording run in the order the prediction rules need: discovery draws first, validation after the
predictions are saved.

    python scripts/score_arm_wording.py --pilots           # only if the separate pilots ran
    python scripts/score_arm_wording.py --check-runs       # run check of the folders copied back
    python scripts/build_arm_wording_mechanism.py       # saved first, from token counts
    python scripts/score_arm_wording.py --stage discovery  # the three discovery configurations
    python scripts/run_arm_wording.py --stage discovery    # saves the predictions
    python scripts/score_arm_wording.py --stage all        # refused until the predictions exist

Reads data/d2_out/arm_wording/collection/<key> (copied back from the GPU machine), checks that each folder ran the
recorded request file to completion, and scores GSM8K with the boxed scorer (manifests/gsm8k_scorer.json)
and MATH-500 with src/math_task.py, strict and lenient, the forced arm on "\boxed{" + output (d2_io prepends the
prefill), box closure beside it. Writes data/processed/arm_wording/<stage>_scored.parquet and a .meta.json with the
folders, run ids and request hashes. --stage discovery keeps only the rows of the discovery configurations; no
validation row is scored before the predictions are saved. Both stages refuse unless the mechanism forecast
(manifests/predictions_arm_wording_mechanism.json) exists and matches its sha256 file and its content hash.

Dry runs: --base DIR (outside the repository), --allow-test-backend, --items-dir and --synthetic SEED, which replaces
correctness by seeded draws from a model of model, arm, wording, configuration and item effects (the echo backend
writes no answers). --synthetic is refused inside the repository.

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, d2_io  # noqa: E402
import build_arm_wording as baw  # noqa: E402


def check_predictions(paths) -> str:
    if not paths.predictions.is_file() or not paths.predictions_sha.is_file():
        raise aw.ArmWordingError("the predictions are not saved; run `score_arm_wording.py --stage discovery` and "
                                 "`run_arm_wording.py --stage discovery` first")
    want = paths.predictions_sha.read_text(encoding="utf-8").split()[0]
    got = aw.sha256_file(paths.predictions)
    if got != want:
        raise aw.ArmWordingError(f"{paths.predictions} changed after it was saved (sha256 {got} != {want})")
    return got


def check_mechanism(paths) -> str:
    """No arm-wording output is scored, at either stage, before the mechanism forecast
    (manifests/predictions_arm_wording_mechanism.json, written by scripts/build_arm_wording_mechanism.py from token
    counts) exists and matches its sha256 file and its content hash."""
    from src import arm_wording_mechanism as awm
    try:
        return awm.check_forecast(paths.mechanism, paths.mechanism_sha)
    except ValueError as exc:
        raise aw.ArmWordingError(f"{exc}; run scripts/build_arm_wording_mechanism.py first")


def synthetic(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seeded stand-in correctness for dry runs: logistic in model ability, arm, wording, configuration and item."""
    t = table.copy()

    def unif(frame, salt):
        h = pd.util.hash_pandas_object(frame.astype(str), index=False, hash_key=f"{seed:08d}{salt}"[-16:].rjust(16, "0"))
        return (h.to_numpy() >> np.uint64(11)).astype(float) / float(2 ** 53)

    def eff(col, scale, salt):
        from scipy import stats
        u = np.clip(unif(t[[col]], salt), 1e-12, 1 - 1e-12)
        return scale * stats.norm.ppf(u)

    # every value is a function of the row's key alone, so the discovery stage and the full table agree
    z = (eff("model_key", 1.2, "m") + eff("item_id", 1.0, "i") + eff("config_id", 0.15, "c")
         + t["arm"].map({aw.COT: 0.8, aw.REQ: 0.0, aw.FD: -0.3}).to_numpy()
         + eff("model_key", 0.5, "g") * (t["arm"] == aw.COT).to_numpy())
    p = 1 / (1 + np.exp(-z))
    key = t[["model_key", "arm", "config_id", "item_id"]]
    t["strict_correct"] = unif(key, "s") < p
    t["lenient_correct"] = t["strict_correct"] | (unif(key, "l") < 0.02)
    return t


def run(args) -> dict:
    paths = aw.Paths(args.base)
    if args.synthetic is not None and baw.in_repo(paths.base):
        raise aw.ArmWordingError("--synthetic is for dry runs; pass --base outside the repository")
    if args.pilots:
        rec = json.loads((paths.requests / "pilot" / "build_record.json").read_text(encoding="utf-8"))
        rows = baw.pilot_check(baw.run_folders(paths.outputs / "pilot", rec["models"]), rec, aw.load_config(args.config))
        n = baw.write_pilot_md(rows, paths, "pilot")
        return {"pilot_check": str(paths.results / "pilot_check.md"), "passed": n, "of": len(rows)}
    if args.check_runs:
        rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
        folders = baw.run_folders(paths.outputs / "collection", rec["models"])
        present = {**rec, "models": {k: v for k, v in rec["models"].items() if k in folders}}
        rows = baw.pilot_check(folders, present, aw.load_config(args.config))
        n = baw.write_pilot_md(rows, paths, "collection run")
        return {"run_check": str(paths.results / "collection_run_check.md"),
                "passed": n, "of": len(rows), "not_yet_copied": sorted(set(rec["models"]) - set(folders))}
    mech_sha = check_mechanism(paths)
    if args.stage == "all":
        pred_sha = check_predictions(paths)
    else:
        pred_sha = None
    rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    keep = {t: set(v["discovery"]) if args.stage == "discovery" else set(v["discovery"] + v["validation"])
            for t, v in rec["configurations"].items()}
    folders = baw.run_folders(paths.outputs / "collection", rec["models"])
    missing = sorted(set(rec["models"]) - set(folders))
    if missing:
        raise aw.ArmWordingError(f"no output folder for {missing}")
    from src import screening
    spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
    items = {t: baw.items(t, "target", args) for t in aw.TASKS}
    parts, meta = [], {}
    for k, d in sorted(folders.items()):
        out, info = d2_io.inspect_work_dir(d, allow_test_backend=args.allow_test_backend, string_item_ids=True)
        shas = {r["requests_sha256"] for r in info["runs"]}
        if shas != {rec["models"][k]["requests_sha256"]}:
            raise aw.ArmWordingError(f"{d}: the runs read another request file")
        if len(out) != rec["models"][k]["n_requests"]:
            raise aw.ArmWordingError(f"{d}: {len(out)} outputs, the request file has {rec['models'][k]['n_requests']}")
        parsed = out["config_id"].map(aw.parse_config_id)
        out = out.assign(task=[p[0] for p in parsed], conf=[p[1] for p in parsed])
        out = out[[c in keep[t] for t, c in zip(out["task"], out["conf"])]]
        for t in aw.TASKS:
            sub = out[out["task"] == t].drop(columns=["task", "conf"])
            parts.append(aw.score(sub, t, items[t], spec))
        meta[k] = {"folder": d2_io.rel_path(d), "run_ids": [r["run_id"] for r in info["runs"]],
                   "requests_sha256": rec["models"][k]["requests_sha256"], "gpu": info["gpu_type"],
                   "test_backend": info["test_backend"]}
    cols = ["model_key", "arm", "config_id", "item_id", "strict_correct", "lenient_correct", "compliant",
            "box_closed", "finish_reason", "n_output_tokens", "prompt_sha256"]
    table = pd.concat([p[cols] for p in parts], ignore_index=True)
    if args.synthetic is not None:
        table = synthetic(table, args.synthetic)
    paths.processed.mkdir(parents=True, exist_ok=True)
    out_path = paths.processed / f"{args.stage}_scored.parquet"
    table.to_parquet(out_path, index=False)
    md = {"stage": args.stage, "configurations": {t: sorted(v) for t, v in keep.items()}, "folders": meta,
          "collection_record_sha256": aw.sha256_file(paths.requests / "collection" / "build_record.json"),
          "predictions_sha256": pred_sha, "mechanism_sha256": mech_sha, "synthetic_seed": args.synthetic,
          "scorers": {"gsm8k": "manifests/gsm8k_scorer.json", "math": "manifests/math_scorer.json"}}
    aw.write_json(out_path.with_suffix(".meta.json"), md)
    return {"out": str(out_path), "rows": int(len(table)), "models": len(meta), "stage": args.stage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=("discovery", "all"))
    g.add_argument("--pilots", action="store_true")
    g.add_argument("--check-runs", action="store_true",
                   help="the run check of the collection folders copied back so far (the canary); no correctness")
    ap.add_argument("--base")
    ap.add_argument("--config")
    ap.add_argument("--items-dir")
    ap.add_argument("--allow-test-backend", action="store_true")
    ap.add_argument("--synthetic", type=int, metavar="SEED")
    args = ap.parse_args(argv)
    try:
        res = run(args)
    except (aw.ArmWordingError, d2_io.D2IOError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
