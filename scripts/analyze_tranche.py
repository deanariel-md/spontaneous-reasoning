r"""Measure the Dataset 2 throughput tranche and project cost for 8, 12, 16 and 24 units per arm.

Each --run-dir is the throughput/ folder written by scripts/generate_d2.py --throughput for one model.
The script refuses, and writes nothing, when a folder is not a complete throughput-mode run (main-run
output is never mixed in; tranche data are excluded from the main estimation).

PowerShell:
    .\.venv\Scripts\python.exe scripts\analyze_tranche.py --run-dir results\d2\<key>\throughput `
        [--run-dir results\d2\<key2>\throughput ...] [--models 20] [--items 1319] `
        [--bank manifests\gsm8k_bank.json | --bank-size N] [--usd-per-gpu-hour X] `
        [--budget-yaml config\budget.yaml] [--out-dir results\tranche]

Each --tranche-dir is one model's folder of the noise tranche: an ordinary run on requests
built with scripts\build_requests.py --tranche, matched to tranche\build_record.json under --requests-dir.
A preempted run's folder holds several run logs; its throughput pools them.
    .\.venv\Scripts\python.exe scripts\analyze_tranche.py --tranche-dir data\d2_out\tranche\<key> `
        [--tranche-dir ...] --models 19 --usd-per-gpu-hour 0

Outputs in --out-dir (default results\tranche): cost_projection.json, cost_projection.md and the
identical cost_projection_clean.md.

A folder made with scripts\generate_d2.py --measure-cap carries cap_override in its run log. The projection
then uses the configured output caps: each unique output length enters as min(length, configured cap). The
report names the override (key cap_override, a warning on the console and in the Markdown), and a log that
does not record the configured cap is reported as missing and is not guessed.

Bank: manifests\gsm8k_bank.json (key sampling_probabilities) when the file exists, else --bank-size N
(uniform), else every draw is taken as distinct. --allow-non-vllm accepts a folder made by the Echo test
backend (dry runs only; the output then carries a warning).

Exit codes: 0 written, 2 refused.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import tranche as tr  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", action="append", default=[], metavar="DIR",
                    help="throughput/ folder of one model's run; repeat once per model")
    ap.add_argument("--models", type=int, default=tr.DEFAULT_MODELS, help="panel size M (default %(default)s)")
    ap.add_argument("--items", type=int, default=tr.DEFAULT_ITEMS, help="items per arm (default %(default)s)")
    ap.add_argument("--bank", help="bank manifest with sampling_probabilities (default manifests/gsm8k_bank.json)")
    ap.add_argument("--bank-size", type=int, help="uniform bank of N configurations when no bank file exists")
    ap.add_argument("--usd-per-gpu-hour", type=float, help="price used for the cost column")
    ap.add_argument("--budget-yaml", help="override config/budget.yaml")
    ap.add_argument("--out-dir", help="default results/tranche")
    ap.add_argument("--allow-non-vllm", action="store_true", help="accept Echo test-backend folders (dry runs)")
    ap.add_argument("--tranche-dir", action="append", default=[], metavar="DIR",
                    help="noise-tranche folder of one model (an ordinary run on requests built with --tranche); "
                         "repeat once per model")
    ap.add_argument("--requests-dir", help="where scripts/build_requests.py wrote tranche/build_record.json "
                                           "(default data/requests/gsm8k)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bank = tr.resolve_bank(args.bank, args.bank_size)
        caps = tr.load_caps(args.budget_yaml)
        report = tr.build_report(args.run_dir, models=args.models, items=args.items, bank=bank,
                                 usd_per_gpu_hour=args.usd_per_gpu_hour, caps=caps,
                                 allow_non_vllm=args.allow_non_vllm, tranche_dirs=args.tranche_dir,
                                 requests_dir=args.requests_dir)
        paths = tr.write_outputs(report, args.out_dir)
    except tr.TrancheError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    for key, p in paths.items():
        print(f"{key}: {p}")
    for w in report["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
