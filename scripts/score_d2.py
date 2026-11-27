r"""Score Dataset 2 generation outputs into the table that scripts/run_dataset2.py reads.

Input is one generation folder per model (the --out-dir of scripts/generate_d2.py: parquet shards
plus run_log-*.json). The script joins the gold answers of the GSM8K split the requests were built on
(test, or train for screening and tranche requests built with --split train), calls
src.scoring.score_outputs with the scorer manifest, and writes one row per (model_key, arm,
config_id, item_id) with the columns model_key, arm, config_id, item_id, strict_correct,
lenient_correct, compliant, finish_reason, n_output_tokens, cached, prompt_sha256. A screening table
has no strict_correct or lenient_correct column, so items are chosen without looking at outcomes. The scorer record (version, format
spec and its hash), the inputs (run ids, request-file and bank hashes), the allow-flags, the split
and the items hash go into the parquet key-value metadata (keys d2_*) and into a sidecar
<out stem>.meta.json. The sidecar's summary per model and arm gives accuracy for a tranche table
only: the main table would show validation outcomes before the prediction rules are saved.

Usage, scorer manifest (manifests/gsm8k_scorer.json; refused when absent):
    python scripts/score_d2.py --work-dir results/d2/<key1> results/d2/<key2> \
        --scorer manifests/gsm8k_scorer.json --out data/processed/gsm8k/scored.parquet

Usage, dry run with a candidate format spec from src/scoring.py (the output says
"candidate, not registered"):
    python scripts/score_d2.py --work-dir results/d2/<key> --candidate-spec answer_colon

Usage, throughput/calibration tranche (folders whose requests were built with --tranche;
written to data/processed/gsm8k/tranche_scored.parquet, the input of
scripts/precision_matched.py --tranche-scored):
    python scripts/score_d2.py --work-dir results/d2_tranche/<key> --tranche \
        --scorer manifests/gsm8k_scorer.json

Usage, admissibility screening (folders whose requests were built with --screening
--split train; written to data/processed/gsm8k/screening_scored.parquet, the input of
scripts/screen_d2.py):
    python scripts/score_d2.py --work-dir results/d2_screening/<key> --screening \
        --scorer manifests/gsm8k_scorer.json

The split comes from the build record that matches each folder's request file; --split test|train
states it explicitly and is refused when it contradicts the record. Train-split outputs are scored
against the train gold answers and are refused as a main table (a train-split run needs
--screening or --tranche). Each kind has its own default output file, and a score is never written
to the default file of another kind.

Refusals (exit 2, nothing written): a throughput folder (the runner marks it by the folder name
throughput, throughput_log-*.json, shard metadata throughput=true and a throughput-mode run log;
it is excluded from estimation); a folder without a run log; a latest run that
is not complete or used --limit (unless --allow-incomplete); a test (echo) backend (unless
--allow-test-backend); a folder whose request file is not matched by a build record under
--requests-dir (unless --allow-unmatched-requests); a tranche or screening folder scored as another
kind, or the reverse; folders of two splits; a --split that contradicts the build record; one model
in two folders; duplicated (model_key, arm, config_id, item_id) rows; an absent or inconsistent
scorer manifest; output item ids without a gold answer.

BIG-Bench Hard (--task bbh; src/bbh_d2.py). The scorer is src/bbh.py with
manifests/bbh_scorer.json, which must equal its scorer record (--candidate-spec is refused). The kind fixes
the item set whose gold answers are read: the screening set for --screening, the calibration set for
--tranche, the target set otherwise; each folder's build record must name that set. --split is refused.
Defaults: --requests-dir data/requests/bbh, outputs data/processed/bbh/scored.parquet,
tranche_scored.parquet and screening_scored.parquet. The table gains the metadata key d2_task = bbh.
    python scripts/score_d2.py --task bbh --work-dir <one dir per model> --screening

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import bbh_d2, d2_io, generation as gen, gsm8k, string_tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed", "gsm8k")
DEFAULT_OUT = os.path.join(PROCESSED, "scored.parquet")
DEFAULT_TRANCHE_OUT = os.path.join(PROCESSED, "tranche_scored.parquet")
DEFAULT_SCREENING_OUT = os.path.join(PROCESSED, "screening_scored.parquet")
DEFAULT_BY_KIND = {d2_io.KIND_MAIN: DEFAULT_OUT, d2_io.KIND_TRANCHE: DEFAULT_TRANCHE_OUT,
                   d2_io.KIND_SCREENING: DEFAULT_SCREENING_OUT}
DEFAULT_SCORER = os.path.join(ROOT, "manifests", "gsm8k_scorer.json")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k",
                    help="gsm8k (default), bbh (BIG-Bench Hard) or math (MATH-500)")
    ap.add_argument("--work-dir", nargs="+", required=True, metavar="DIR",
                    help="one generation output folder per model (never a throughput folder)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--scorer", metavar="PATH",
                   help="scorer manifest (default manifests/gsm8k_scorer.json; refused when absent)")
    g.add_argument("--candidate-spec", metavar="NAME",
                   help="dry run with a candidate FormatSpec from src/scoring.py CANDIDATE_SPECS; the output is "
                        "labelled 'candidate, not registered'")
    ap.add_argument("--out", help="default data/processed/gsm8k/scored.parquet, tranche_scored.parquet with "
                                  "--tranche, or screening_scored.parquet with --screening")
    ap.add_argument("--tranche", action="store_true",
                    help="the folders hold a tranche (requests built with --tranche); written to a separate table "
                         "and excluded from estimation")
    ap.add_argument("--screening", action="store_true",
                    help="the folders hold admissibility screening outputs (requests built with --screening on the "
                         "train split); written to a separate table and excluded from estimation")
    ap.add_argument("--split", choices=d2_io.SPLITS,
                    help="GSM8K split of the gold answers; default: the split of the matching build record, else test. "
                         "Refused when it contradicts the record")
    ap.add_argument("--requests-dir",
                    help="where scripts/build_requests.py wrote build_record.json (and tranche/build_record.json); "
                         "default data/requests/gsm8k (bbh: data/requests/bbh)")
    ap.add_argument("--items-parquet", metavar="PATH",
                    help="dry run only (needs --candidate-spec): gold answers from a parquet with item_id, question, "
                         "gold_answer instead of the pinned GSM8K test split")
    ap.add_argument("--allow-test-backend", action="store_true", help="accept EchoBackend output (dry runs)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="accept a folder whose latest run is not complete or used --limit (diagnostic tables)")
    ap.add_argument("--allow-unmatched-requests", action="store_true",
                    help="accept a folder whose request file has no build record (diagnostic tables)")
    return ap


def run_bbh(args) -> dict:
    """BIG-Bench Hard or MATH-500 scoring (module doc): the task's scorer, the kind's item set, the same folder
    checks."""
    tm = string_tasks.module(args.task) or bbh_d2
    if args.candidate_spec:
        raise d2_io.ScoreInputError(f"--candidate-spec does not apply to --task {tm.TASK}: {tm.FIXES_SCORER}")
    if args.split is not None:
        raise d2_io.ScoreInputError(f"--split does not apply to --task {tm.TASK}: the kind fixes the item set")
    if args.items_parquet and not args.allow_test_backend:
        raise d2_io.ScoreInputError("--items-parquet is for dry runs and needs --allow-test-backend")
    if args.tranche and args.screening:
        raise d2_io.ScoreInputError("--tranche and --screening are separate kinds of table; pass one of them")
    record = tm.load_scorer(args.scorer)
    kind = d2_io.KIND_SCREENING if args.screening else d2_io.KIND_TRANCHE if args.tranche else d2_io.KIND_MAIN
    out = Path(args.out or tm.SCORED[kind])
    for other_kind, other_path in tm.SCORED.items():
        if other_kind != kind and out.resolve() == Path(other_path).resolve():
            raise d2_io.ScoreInputError(f"{out} is the default output of the {other_kind} table; {kind} scores "
                                        "must not go there")
    requests_dir = args.requests_dir or str(tm.REQUESTS_DIR)
    records = d2_io.load_build_records(requests_dir)
    split = tm.KIND_SPLIT[kind]           # score_work_dirs refuses a folder built on the other split
    items, items_info = tm.load_items(tm.KIND_ITEM_SET[kind], args.items_parquet)
    table, infos = d2_io.score_work_dirs(
        args.work_dir, None, None, records=records, tranche=args.tranche, screening=args.screening, split=split,
        allow_test_backend=args.allow_test_backend, allow_incomplete=args.allow_incomplete,
        allow_unmatched=args.allow_unmatched_requests, score_fn=tm.score_fn(items), string_item_ids=True)
    for i in infos:
        m = i["requests_match"]
        if m["kind"] is not None and m.get("items_canonical_sha256") != items_info["canonical_sha256"]:
            raise d2_io.ScoreInputError(f"{i['path']}: its requests were built on items with canonical sha256 "
                                        f"{m.get('items_canonical_sha256')!r}, not the {items_info['item_set']} set "
                                        f"({items_info['canonical_sha256']!r})")
    return _write(args, table, infos, record, items_info, kind, split, out, requests_dir, task=tm.TASK)


def run(args) -> dict:
    if string_tasks.module(args.task) is not None:
        return run_bbh(args)
    if args.candidate_spec:
        spec, record = d2_io.candidate_scorer(args.candidate_spec)
    else:
        spec, record = d2_io.load_registered_scorer(args.scorer or DEFAULT_SCORER)
    if args.items_parquet and not args.candidate_spec:
        raise d2_io.ScoreInputError("--items-parquet is for dry runs and needs --candidate-spec")
    if args.tranche and args.screening:
        raise d2_io.ScoreInputError("--tranche and --screening are separate kinds of table; pass one of them")
    kind = d2_io.KIND_SCREENING if args.screening else d2_io.KIND_TRANCHE if args.tranche else d2_io.KIND_MAIN
    out = Path(args.out or DEFAULT_BY_KIND[kind])
    for other_kind, other_path in DEFAULT_BY_KIND.items():
        if other_kind != kind and out.resolve() == Path(other_path).resolve():
            raise d2_io.ScoreInputError(f"{out} is the default output of the {other_kind} table; {kind} scores "
                                        "must not go there")
    requests_dir = args.requests_dir or os.path.join(ROOT, "data", "requests", "gsm8k")
    records = d2_io.load_build_records(requests_dir)
    split = d2_io.resolve_split(args.work_dir, records, args.split)
    if split == "train" and kind == d2_io.KIND_MAIN:
        raise d2_io.ScoreInputError("train-split outputs are calibration data and cannot be scored as a main "
                                    "table; add --screening or --tranche")
    items, items_info = d2_io.load_items(args.items_parquet, split=split)
    gold = gsm8k.gold_lookup(items)
    table, infos = d2_io.score_work_dirs(
        args.work_dir, spec, gold, records=records, tranche=args.tranche, screening=args.screening, split=split,
        allow_test_backend=args.allow_test_backend, allow_incomplete=args.allow_incomplete,
        allow_unmatched=args.allow_unmatched_requests)
    return _write(args, table, infos, record, items_info, kind, split, out, requests_dir)


def _write(args, table, infos, record, items_info, kind, split, out, requests_dir, task="gsm8k") -> dict:
    """The scored parquet with its d2_* metadata and the sidecar; a BIG-Bench Hard table adds d2_task."""
    created = d2_io.utc_now()
    summary = d2_io.score_summary(table, kind)
    allow = {"allow_test_backend": bool(args.allow_test_backend), "allow_incomplete": bool(args.allow_incomplete),
             "allow_unmatched_requests": bool(args.allow_unmatched_requests)}
    meta = {
        "d2_schema_version": d2_io.SCHEMA_VERSION, "d2_kind": kind, "d2_split": split,
        "d2_purpose": d2_io.PURPOSES[kind],
        "d2_excluded_from_main": json.dumps(kind != d2_io.KIND_MAIN),
        "d2_scorer_status": record["status"], "d2_scorer_version": record["scorer_version"],
        "d2_scorer_record": json.dumps(record, sort_keys=True), "d2_items": json.dumps(items_info, sort_keys=True),
        "d2_inputs": json.dumps([d2_io.compact_input(i) for i in infos], sort_keys=True),
        "d2_allow_flags": json.dumps(allow, sort_keys=True), "d2_created": created}
    if task != "gsm8k":
        meta["d2_task"] = task
    d2_io.write_scored(table, out, meta)
    sidecar = d2_io.sidecar_path(out)
    units = ("compliant: strict extraction succeeded; a screening table has no correctness column (hard rule 2)"
             if kind == d2_io.KIND_SCREENING else
             "strict_correct, lenient_correct: 0/1 per output; compliant: strict extraction succeeded")
    summary_note = {d2_io.KIND_MAIN: "no accuracy: the table pools discovery and validation units, and "
                                             "scripts/run_dataset2.py reports accuracy after the prediction rules "
                                             "are frozen (hard rule 6)",
                    d2_io.KIND_SCREENING: "no accuracy: screening judges format and behaviour only",
                    d2_io.KIND_TRANCHE: "calibration items; excluded from estimation"}[kind]
    side = {
        "schema_version": d2_io.SCHEMA_VERSION, "created": created, "kind": kind, "split": split,
        "purpose": d2_io.PURPOSES[kind],
        "excluded_from_main": kind != d2_io.KIND_MAIN, "scored_path": d2_io.rel_path(out),
        "scored_sha256": d2_io.sha256_path(out), "n_rows": int(len(table)),
        "columns": list(table.columns), "scorer": record, "items": items_info, "units": units,
        "inputs": infos, "summary_by_model_arm": summary, "summary_note": summary_note,
        "libraries": gen.library_versions(),
        "invocation": {"work_dirs": [str(w) for w in args.work_dir], "requests_dir": d2_io.rel_path(requests_dir),
                       **allow}}
    if task != "gsm8k":
        side = {"task": task, **side}
    d2_io.write_sidecar(sidecar, side)
    result = {"out": str(out), "sidecar": str(sidecar), "kind": kind, "split": split, "n_rows": int(len(table)),
              "models": sorted(table["model_key"].unique()), "scorer_status": record["status"],
              "scorer_version": record["scorer_version"],
              "any_test_backend": any(i["test_backend"] for i in infos),
              "all_requests_matched": all(i["requests_match"]["kind"] is not None for i in infos)}
    return {"task": task, **result} if task != "gsm8k" else result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except d2_io.D2IOError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return gen.EXIT_REFUSED
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return gen.EXIT_ERROR
    if result["scorer_status"] != d2_io.REGISTERED_STATUS:
        print("warning: scorer is a candidate, not registered; the table is not a main table", file=sys.stderr)
    if result["any_test_backend"]:
        print("warning: outputs come from the echo test backend", file=sys.stderr)
    if not result["all_requests_matched"]:
        print("warning: at least one folder has no matching build record (diagnostic table)", file=sys.stderr)
    print(json.dumps(result, indent=1))
    return gen.EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
