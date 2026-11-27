r"""Admissibility screening report for the Dataset 2 prompt bank and models. Logic: src/screening.py.

Input is the screening scored table written by scripts/score_d2.py --screening (outputs on GSM8K
TRAINING calibration items, requests built with scripts/build_requests.py --screening --split train) and
the raw generation folders it was scored from (taken from the table's metadata unless --work-dir is
given). Per model x configuration x arm, and aggregated per configuration and per model, the report
gives: n; strict compliance under every candidate spec in src/scoring.py CANDIDATE_SPECS and under the
the scorer's spec when manifests/gsm8k_scorer.json exists; the lenient-extractable share; truncation at the
cap; outputs with think tags; empty outputs; output-token mean, median and p95; and, for the direct arm,
the share of outputs longer than 32, 64 and 128 tokens (the "meaningful direct answer" diagnostic).
It never computes correctness, accuracy or gains, so screening does not look at outcomes.

Rules come from config/admissibility.yaml. While its status is PENDING_REVIEW (every threshold null) only
the metrics report is written: results/screening/screening_report.json, .md and the identical
_clean.md. When the status is `confirmed`, the exclusion log manifests/gsm8k_admissibility.json is
written as well: each excluded model or configuration with the rule and the metric value, and the
retained ids. An existing exclusion log with other content needs --overwrite.

Usage, metrics report from the default screening table:
    python scripts/screen_d2.py

Usage, explicit inputs:
    python scripts/screen_d2.py --scored data/processed/gsm8k/screening_scored.parquet \
        --work-dir results/d2_screening/<key1> results/d2_screening/<key2> --rules config/admissibility.yaml \
        --scorer manifests/gsm8k_scorer.json --out-dir results/screening

Refusals (exit 2, nothing written): a scored table that is not a train-split screening table; generation
folders that do not match the scored table; an invalid rules file; confirmed rules without a
scorer manifest.

BIG-Bench Hard (--task bbh): the same rules file (config/admissibility.yaml, models and
configurations) on the 200 screening items. Compliance is src/bbh.py strict extraction for the
item's answer type (manifests/bbh_scorer.json); the lenient share counts a token of that type; the other
metrics are as above; no correctness is computed. Defaults: data/processed/bbh/screening_scored.parquet,
results/screening_bbh/, manifests/bbh_admissibility.json.
    python scripts/screen_d2.py --task bbh

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import bbh_d2, d2_io, generation as gen, hotpot_d2, math_d2, screening, string_tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULTS = {"gsm8k": {"scored": str(screening.SCORED_DEFAULT), "out_dir": str(screening.REPORT_DIR),
                      "exclusion_log": str(screening.EXCLUSION_LOG)},
            "bbh": {"scored": str(bbh_d2.SCORED[d2_io.KIND_SCREENING]), "out_dir": str(bbh_d2.SCREENING_REPORT_DIR),
                    "exclusion_log": str(bbh_d2.ADMISSIBILITY_LOG)},
            "math": {"scored": str(math_d2.SCORED[d2_io.KIND_SCREENING]), "out_dir": str(math_d2.SCREENING_REPORT_DIR),
                     "exclusion_log": str(math_d2.ADMISSIBILITY_LOG)},
            "hotpot": {"scored": str(hotpot_d2.SCORED[d2_io.KIND_SCREENING]),
                       "out_dir": str(hotpot_d2.SCREENING_REPORT_DIR), "exclusion_log": str(hotpot_d2.ADMISSIBILITY_LOG)}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k",
                    help="gsm8k (default), bbh (BIG-Bench Hard) or math (MATH-500)")
    ap.add_argument("--scored",
                    help="screening scored table (default data/processed/gsm8k/screening_scored.parquet; bbh: "
                         "data/processed/bbh/screening_scored.parquet)")
    ap.add_argument("--work-dir", nargs="+", metavar="DIR",
                    help="generation folders (one per model) holding the raw outputs; default: the folders recorded "
                         "in the scored table's metadata")
    ap.add_argument("--rules", default=str(screening.RULES_PATH),
                    help="admissibility rules (default config/admissibility.yaml)")
    ap.add_argument("--scorer", metavar="PATH",
                    help="scorer manifest; default manifests/gsm8k_scorer.json when it exists (bbh: "
                         "manifests/bbh_scorer.json, required). An explicit path that does not exist is refused")
    ap.add_argument("--out-dir", help="default results/screening (bbh: results/screening_bbh)")
    ap.add_argument("--exclusion-log",
                    help="default manifests/gsm8k_admissibility.json (bbh: manifests/bbh_admissibility.json); written "
                         "only when the rules are confirmed")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing exclusion log whose content would change")
    return ap


def run(args) -> dict:
    for k, v in DEFAULTS[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    report = screening.screen(args.scored, args.work_dir, args.rules, args.scorer, task=args.task)
    if report["exclusions"] is not None:       # refuse a conflicting exclusion log before any file is written
        screening.check_exclusion_log(report, args.exclusion_log, overwrite=args.overwrite)
    paths = screening.write_report(report, args.out_dir)
    log_path = None
    if report["exclusions"] is not None:
        log_path = screening.write_exclusion_log(report, paths["json"], args.exclusion_log, overwrite=args.overwrite)
    ex = report["exclusions"]
    return {"report_json": str(paths["json"]), "report_md": str(paths["md"]), "report_clean_md": str(paths["clean_md"]),
            "rules_status": report["rules"]["status"], "rules_applied": report["rules"]["applied"],
            "exclusion_log": str(log_path) if log_path else None,
            "models": report["inputs"]["models"], "n_configurations": report["inputs"]["n_configurations"],
            "n_rows": report["inputs"]["n_rows"],
            "excluded_models": [m["model_key"] for m in ex["excluded_models"]] if ex else None,
            "excluded_configurations": [c["config_id"] for c in ex["excluded_configurations"]] if ex else None,
            "registered_scorer": report["scorers"]["registered"].get("status")}


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
    if not result["rules_applied"]:
        print("warning: the rules are PENDING_REVIEW; only the metrics report was written, no exclusion log",
              file=sys.stderr)
    print(json.dumps(result, indent=1))
    return gen.EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
