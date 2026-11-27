r"""Render the Dataset 2 request files: frozen prompt bank + sampler -> one parquet per model.

Each file has the columns the runner reads (request_id, model_key, arm, config_id, item_id,
messages_json) and one row per distinct configuration x arm x item. Builds of the target items (kind main) render the
GSM8K test items (1,319). Admissibility screening and calibration tranches render a deterministic
sample of GSM8K TRAINING items instead (admissibility is screened on
calibration items from the training split, never on target items). A configuration drawn more than once
by the sampler, in discovery or validation, is rendered once. A build record (build_record.json) next
to the files holds the split, purpose, bank, sampler and items hashes, the request-file hashes and the
counts per model and arm. Bank schema: src/d2_io.py docstring.

The bank (manifests/gsm8k_bank.json) and the sampler (manifests/sampler_gsm8k.json) are frozen files;
this script refuses when they are absent, inconsistent with each other, or
malformed. A build of the target items also needs the admissibility exclusion log
(manifests/gsm8k_admissibility.json): it refuses a model that the rules excluded or that was never
screened, and a bank that is not the retained bank. It makes no model call and needs no GPU.

Usage. Requests of the target items for the frozen panel, test items, frozen sampler:
    .\.venv\Scripts\python.exe scripts\build_requests.py --bank manifests\gsm8k_bank.json `
        --sampler manifests\sampler_gsm8k.json --models <key> [<key> ...] --out-dir data\requests\gsm8k

Usage. Admissibility screening on N training items, every configuration of the bank (or
--configs ID ...), written under <out-dir>\screening and excluded from the main estimation:
    .\.venv\Scripts\python.exe scripts\build_requests.py --bank manifests\gsm8k_bank.json `
        --models <key> [<key> ...] --split train --calibration-items N --screening --out-dir data\requests\gsm8k

Usage. Throughput/noise tranche with an explicit configuration list (written
under <out-dir>\tranche and excluded from the main estimation; the sampler is not used). Use
--split train --calibration-items N so that no target outcome is observed before collection; --split
test (the default) renders the target items and prints a warning:
    .\.venv\Scripts\python.exe scripts\build_requests.py --bank manifests\gsm8k_bank.json `
        --models <key> --configs <id> [<id> ...] --tranche --split train --calibration-items N `
        --out-dir data\requests\gsm8k

The calibration sample (N distinct training item ids) is drawn from item ids only, with
sampling.rng_for("gsm8k-calibration-items", N), sorted, and written once to
manifests/gsm8k_calibration_items.json (split, dataset revision, N, seed label, item ids, sha256).
Later runs reuse it and refuse a different N unless --overwrite. Train request ids are
"train|<config>|<arm>|<item>" so they never coincide with test request ids. --split train is refused
without --screening or --tranche; --screening is refused with --split test; --calibration-items
applies to --split train only.

Model keys must be entries of config/models.yaml dataset2.models (the schema template does not
count) unless --allow-unregistered, which is for tests and dry runs; the record then marks the
model as unregistered. Existing request files with identical content are left alone; different
content needs --overwrite (outputs keyed to the old request ids would no longer match). A dry run with
--items-parquet needs an explicit --calibration-manifest for a train split, so that it cannot write the
repository manifest.

BIG-Bench Hard (--task bbh; src/bbh_d2.py). The request kind fixes
the item set: --screening renders the 200 screening items, --tranche the 1,350 calibration items and a
build of the target items the 1,350 target items, read from data/processed/bbh/items_<set>.parquet and checked
against manifests/bbh_items.json. --split, --calibration-items and --calibration-manifest are refused.
Item ids are "<task>/<index>"; the build record keeps split "train" for screening and tranche requests (items
outside the target set, so the runner's cap-only gate applies) and "test" for target items, and adds the
task and the item set. Defaults: manifests/bbh_bank.json, manifests/sampler_bbh.json,
manifests/bbh_admissibility.json, data/requests/bbh. Without --models-yaml a key may come from
config/models.yaml or config/models_extension.yaml (the 30 checkpoints of the extension):
    .\.venv\Scripts\python.exe scripts\build_requests.py --task bbh --bank data\bank_candidates\bbh_bank_v1.json `
        --models <key> [<key> ...] --screening

MATH-500 (--task math; src/math_d2.py) works the same way: the
kind fixes the item set (200 screening and the calibration items from MATH training problems, the numeric
MATH-500 target items), read from data/processed/math/items_<set>.parquet and checked against
manifests/math_items.json. Defaults: manifests/math_bank.json, manifests/sampler_math.json,
manifests/math_admissibility.json, data/requests/math. The forced-direct arm is derived from these files by
scripts/build_forced_direct.py --task math.

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import bbh_d2, d2_io, generation as gen, hotpot_d2, math_d2, string_tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULTS = {
    "gsm8k": {"bank": os.path.join(ROOT, "manifests", "gsm8k_bank.json"),
              "sampler": os.path.join(ROOT, "manifests", "sampler_gsm8k.json"),
              "out_dir": os.path.join(ROOT, "data", "requests", "gsm8k"),
              "admissibility": os.path.join(ROOT, "manifests", "gsm8k_admissibility.json")},
    "bbh": {"bank": str(bbh_d2.BANK), "sampler": str(bbh_d2.SAMPLER), "out_dir": str(bbh_d2.REQUESTS_DIR),
            "admissibility": str(bbh_d2.ADMISSIBILITY_LOG)},
    "math": {"bank": str(math_d2.BANK), "sampler": str(math_d2.SAMPLER), "out_dir": str(math_d2.REQUESTS_DIR),
             "admissibility": str(math_d2.ADMISSIBILITY_LOG)},
    # HotpotQA: screening and target items only (no tranche); models_hotpot.yaml and
    # decoding_hotpot.yaml are passed with --models-yaml and --decoding-yaml
    "hotpot": {"bank": str(hotpot_d2.BANK), "sampler": str(hotpot_d2.SAMPLER), "out_dir": str(hotpot_d2.REQUESTS_DIR),
               "admissibility": str(hotpot_d2.ADMISSIBILITY_LOG)},
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k",
                    help="gsm8k (default), bbh (BIG-Bench Hard) or math (MATH-500); "
                         "for bbh and math the kind fixes the item set")
    ap.add_argument("--bank", help="frozen prompt bank (default manifests/gsm8k_bank.json; bbh: manifests/bbh_bank.json)")
    ap.add_argument("--sampler",
                    help="sampler manifest (default manifests/sampler_gsm8k.json; bbh: "
                         "manifests/sampler_bbh.json); unused with --tranche and --screening")
    ap.add_argument("--models", nargs="+", required=True, metavar="KEY",
                    help="model keys; one request file per model")
    ap.add_argument("--out-dir",
                    help="default data/requests/gsm8k (bbh: data/requests/bbh); a tranche goes to <out-dir>/tranche "
                         "and screening to <out-dir>/screening")
    ap.add_argument("--split", choices=d2_io.SPLITS,
                    help="GSM8K split to render: test (default; the target items) or train (calibration items; "
                         "needs --screening or --tranche and --calibration-items); refused with --task bbh")
    ap.add_argument("--calibration-items", type=int, metavar="N",
                    help="with --split train: draw N distinct training item ids (deterministic, from ids only) and "
                         "reuse the sample recorded in manifests/gsm8k_calibration_items.json")
    ap.add_argument("--calibration-manifest", metavar="PATH",
                    help="override manifests/gsm8k_calibration_items.json (required with --items-parquet)")
    ap.add_argument("--configs", nargs="+", metavar="ID",
                    help="explicit configuration ids instead of the sampler; requires --tranche or --screening")
    ap.add_argument("--tranche", action="store_true",
                    help="throughput/calibration tranche: render --configs, write under <out-dir>/tranche, "
                         "mark the record excluded_from_main")
    ap.add_argument("--screening", action="store_true",
                    help="admissibility screening: needs --split train; renders --configs or, without it, "
                         "every configuration of the bank; written under <out-dir>/screening and marked "
                         "excluded_from_main")
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="accept model keys that are not in config/models.yaml (tests and dry runs)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace request files, a build record or a calibration sample whose content would change")
    ap.add_argument("--items-parquet", metavar="PATH",
                    help="dry run only (needs --allow-unregistered): items from a parquet with item_id, question, "
                         "gold_answer instead of the pinned GSM8K split")
    ap.add_argument("--admissibility",
                    help="admissibility exclusion log (builds of the target items only; default "
                         "manifests/gsm8k_admissibility.json; bbh: manifests/bbh_admissibility.json)")
    ap.add_argument("--models-yaml", help="override config/models.yaml (bbh: the only registry used, instead of "
                                          "config/models.yaml and config/models_extension.yaml)")
    ap.add_argument("--decoding-yaml", help="override config/decoding.yaml (arms must have max_new_tokens there)")
    return ap


def resolve_defaults(args):
    """Fill the task's default paths (and the GSM8K default split) for the options left unset."""
    for k, v in DEFAULTS[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.task == "gsm8k" and args.split is None:
        args.split = "test"
    return args


def check_mode_bbh(args) -> str:
    """The flag combination of a build with fixed item sets (BIG-Bench Hard, MATH-500); returns the kind."""
    tm = string_tasks.module(args.task) or bbh_d2
    if args.tranche and args.screening:
        raise d2_io.BankError("--tranche and --screening are separate builds; pass one of them")
    for flag, v in (("--split", args.split), ("--calibration-items", args.calibration_items),
                    ("--calibration-manifest", args.calibration_manifest)):
        if v is not None:
            raise d2_io.BankError(f"{flag} does not apply to --task {tm.TASK}: {tm.FIXES_ITEM_SETS}, and the "
                                  "kind chooses one (screening, tranche: calibration, main: target)")
    if args.configs and not (args.tranche or args.screening):
        raise d2_io.BankError("an explicit --configs list is only for a tranche or screening run; add --tranche or "
                              "--screening (their outputs are excluded from the main estimation)")
    if args.tranche and not args.configs:
        raise d2_io.BankError("--tranche needs --configs ID [ID ...]")
    if args.screening:
        return d2_io.KIND_SCREENING
    return d2_io.KIND_TRANCHE if args.tranche else d2_io.KIND_MAIN


def bbh_unregistered(keys, models_yaml) -> list[str]:
    """Keys in none of the BIG-Bench Hard registries (or not in --models-yaml when it is given)."""
    if models_yaml:
        return d2_io.unregistered_models(keys, models_yaml)
    reg = bbh_d2.load_registries()
    return [k for k in keys if k.startswith("_") or k not in reg]


def check_mode(args) -> str:
    """Validate the flag combination; returns the request kind (main, tranche or screening)."""
    if args.tranche and args.screening:
        raise d2_io.BankError("--tranche and --screening are separate builds; pass one of them")
    if args.split == "train" and not (args.tranche or args.screening):
        raise d2_io.BankError("--split train renders calibration items, which can never be main; add "
                              "--screening (admissibility screening) or --tranche")
    if args.screening and args.split != "train":
        raise d2_io.BankError("--screening is admissibility screening on calibration items from the training split "
                              "(never on target items); add --split train")
    if args.split == "train" and args.calibration_items is None:
        raise d2_io.BankError("--split train needs --calibration-items N")
    if args.calibration_items is not None and args.split != "train":
        raise d2_io.BankError("--calibration-items applies to --split train only")
    if args.configs and not (args.tranche or args.screening):
        raise d2_io.BankError("an explicit --configs list is only for a tranche or screening run; add --tranche or "
                              "--screening (their outputs are excluded from the main estimation)")
    if args.tranche and not args.configs:
        raise d2_io.BankError("--tranche needs --configs ID [ID ...]")
    if args.split == "train" and args.items_parquet and not args.calibration_manifest:
        raise d2_io.BankError("a dry run with --items-parquet on the train split needs an explicit "
                              "--calibration-manifest PATH, so that it cannot write manifests/gsm8k_calibration_items.json")
    if args.screening:
        return d2_io.KIND_SCREENING
    return d2_io.KIND_TRANCHE if args.tranche else d2_io.KIND_MAIN


def run(args) -> dict:
    resolve_defaults(args)
    tm = string_tasks.module(args.task)
    bbh = tm is not None                          # a task with fixed item sets (src/string_tasks.py)
    kind = check_mode_bbh(args) if bbh else check_mode(args)
    keys = list(args.models)
    if len(set(keys)) != len(keys):
        raise d2_io.BankError(f"duplicate model keys: {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    for k in keys:
        d2_io.check_model_key(k)
    if bbh:
        unreg = bbh_unregistered(keys, args.models_yaml)
        if unreg and not args.allow_unregistered:
            raise d2_io.BankError(f"model keys {unreg} are not registered in "
                                  f"{args.models_yaml or 'config/models.yaml or config/models_extension.yaml'}; pass "
                                  "--allow-unregistered for a dry run")
    else:
        unreg = d2_io.unregistered_models(keys, args.models_yaml)
        if unreg and not args.allow_unregistered:
            known = sorted(k for k in gen.load_registry(args.models_yaml) if not k.startswith("_")) or ["(none registered)"]
            raise d2_io.BankError(f"model keys {unreg} are not registered in config/models.yaml dataset2.models "
                                  f"(registered: {', '.join(known)}); pass --allow-unregistered for a dry run")

    if args.items_parquet and not args.allow_unregistered:
        raise d2_io.BankError("--items-parquet is for dry runs and needs --allow-unregistered")

    bank, bank_sha = d2_io.load_bank(args.bank)
    warnings = d2_io.validate_bank(bank)
    admissibility = None
    if kind == d2_io.KIND_MAIN:
        log, admissibility = d2_io.load_admissibility(args.admissibility)
        problems = d2_io.panel_admissibility_problems(keys, log) + d2_io.bank_admissibility_problems(bank, log)
        if problems:
            raise d2_io.BankError(f"not admissible under {admissibility['path']}: " + "; ".join(problems))
        try:
            with open(args.sampler, "rb") as fh:
                raw = fh.read()
            sampler = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise d2_io.BankError(f"cannot read the sampler manifest {args.sampler} ({exc}); freeze it with "
                                  "scripts/freeze_sampler_d2.py") from exc
        ids = d2_io.sampler_config_ids(sampler, bank, bank_sha)
        n_draws = len(sampler["discovery_config_ids"]) + len(sampler["validation_config_ids"])
        sampler_info = {"path": d2_io.rel_path(args.sampler), "file_sha256": d2_io.sha256_bytes(raw),
                        "registered_sha256": sampler["sha256"], "label": sampler["label"],
                        "k_discovery": sampler["k_discovery"], "k_validation": sampler["k_validation"]}
    else:
        ids = list(dict.fromkeys(args.configs)) if args.configs else list(bank["configurations"])
        sampler_info, n_draws = None, len(args.configs) if args.configs else len(ids)

    calibration = None
    if bbh:                                       # the kind fixes the item set (src/bbh_d2.py)
        args.split = tm.KIND_SPLIT[kind]
        items, items_info = tm.load_items(tm.KIND_ITEM_SET[kind], args.items_parquet)
    else:
        items, items_info = d2_io.load_items(args.items_parquet, split=args.split)
        if args.split == "train":
            items, items_info, calibration = d2_io.select_calibration_items(
                items, items_info, args.calibration_items, args.calibration_manifest, overwrite=args.overwrite)
        elif kind == d2_io.KIND_TRANCHE:
            warnings = warnings + ["tranche on the GSM8K test items: its outputs are target-item outcomes observed "
                                   "before collection; --split train --calibration-items N keeps the target items unseen"]
    try:
        decoding = gen.load_decoding(args.decoding_yaml)
    except gen.RunRefused as exc:
        raise d2_io.BankError(f"decoding config: {exc}") from exc

    sub = {d2_io.KIND_TRANCHE: d2_io.TRANCHE_DIR, d2_io.KIND_SCREENING: d2_io.SCREENING_DIR}.get(kind)
    out_dir = os.path.join(args.out_dir, sub) if sub else args.out_dir
    excluded = kind != d2_io.KIND_MAIN
    base = d2_io.render_requests(bank, ids, keys[0], items, decoding=decoding, split=args.split, string_item_ids=bbh)
    header = {
        "schema_version": d2_io.SCHEMA_VERSION, "kind": kind, "split": args.split, "purpose": d2_io.PURPOSES[kind],
        "excluded_from_main": excluded, "arms": list(d2_io.ARMS),
        "bank": {"path": d2_io.rel_path(args.bank), "sha256": bank_sha, "generator": bank["generator"],
                 "generator_version": bank["generator_version"], "seed": bank["seed"],
                 "n_configurations": len(bank["configurations"])},
        "sampler": sampler_info, "admissibility": admissibility, "items": items_info,
        "configuration_ids": ids, "n_draws": n_draws, "n_draws_collapsed": n_draws - len(ids),
        "warnings": warnings}
    if bbh:
        header = {"task": tm.TASK, "item_set": items_info["item_set"], **header}
    entries, summary = {}, {}
    for k in keys:
        df = base if k == keys[0] else base.assign(model_key=k)
        if k != keys[0]:
            d2_io.validate_rendered(df, k, decoding)
        path = os.path.join(out_dir, f"{k}.parquet")
        status = d2_io.write_requests_file(df, Path(path), overwrite=args.overwrite)
        entries[k] = {**d2_io.describe_requests(df), "requests_file": f"{k}.parquet",
                      "requests_sha256": d2_io.sha256_path(path), "registered": k not in unreg,
                      "created": d2_io.utc_now()}
        summary[k] = {"file": path, "status": status, "n_requests": entries[k]["n_requests"],
                      "counts_by_arm": entries[k]["counts_by_arm"], "registered": entries[k]["registered"]}
    record_path = Path(out_dir) / d2_io.BUILD_RECORD
    record = d2_io.merge_build_record(record_path, header, entries, overwrite=args.overwrite)
    result = {"kind": kind, "split": args.split, "purpose": d2_io.PURPOSES[kind], "excluded_from_main": excluded,
              "record": str(record_path), "n_configurations": len(ids), "n_items": int(len(items)), "models": summary,
              "contains_unregistered_models": record["contains_unregistered_models"], "warnings": warnings}
    if bbh:
        result = {"task": tm.TASK, "item_set": items_info["item_set"], **result}
    if calibration is not None:
        result["calibration_items"] = {"manifest": items_info["calibration_items"]["path"],
                                       "sha256": calibration["sha256"], "n_items": calibration["n_items"],
                                       "status": calibration["status"]}
    return result


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
    for w in result["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if result["contains_unregistered_models"]:
        print("warning: the record lists models that are not registered in config/models.yaml (dry run)", file=sys.stderr)
    print(json.dumps(result, indent=1))
    return gen.EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
