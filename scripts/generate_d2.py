r"""Dataset 2 generation runner: one model, one already-rendered request file.

The request file is a parquet with columns request_id, model_key, arm, config_id, item_id,
messages_json. Output shards and the run log go to --out-dir. A rerun with the same --out-dir
resumes: request_ids already present are skipped and identical prompts reuse earlier outputs.

Exit codes: 0 complete, 2 refused (placeholder, gate or invalid input), 75 interrupted after a
SIGTERM or SIGUSR1 (progress saved; requeue), 1 any other error.

PowerShell (EchoBackend smoke test on the development machine, no GPU):
    .\.venv\Scripts\python.exe scripts\generate_d2.py --backend echo `
        --requests data\d2\requests_smoke.parquet --model-key <key in models.yaml> `
        --out-dir results\d2_smoke\<key>

The throughput tranche runs on a train-split tranche request file, never on the test-item
file, so that no target-item output is read when output caps are chosen. Build it with
    .\.venv\Scripts\python.exe scripts\build_requests.py --bank manifests\gsm8k_bank.json `
        --models <key> --configs <id> [<id> ...] --tranche --split train --calibration-items N
which writes data\requests\gsm8k\tranche\<key>.parquet (N distinct GSM8K training items, written to
manifests/gsm8k_calibration_items.json).

PowerShell (throughput tranche on a GPU host; 512 requests, split evenly across arms):
    python scripts/generate_d2.py --backend vllm --throughput --limit 512 `
        --requests data/requests/gsm8k/tranche/<key>.parquet --model-key <key> --out-dir results/d2/<key>

A throughput run with --limit above 2000 is a bulk GPU run: it needs cap.money_usd or cap.gpu_hours in
config/budget.yaml (the Dataset 2 allocation is not required).

Choosing output caps in the throughput tranche: --measure-cap ARM=N (repeatable, throughput only)
replaces max_new_tokens for that arm in that run only, so the length distribution is not hidden by the
provisional cap. The report gives, per arm, the share of generations longer than each of 64, 128, 256,
512, 1024, 2048 and 4096 tokens that lies below the run's cap, plus the share cut at the cap. The
override is recorded in the run log (cap_override) and the run fingerprint; use a separate --out-dir.
max_model_len in config/models.yaml must cover the longest prompt plus N. Use the same train-split
tranche file, so the cap is chosen from calibration items only:
    python scripts/generate_d2.py --backend vllm --throughput --limit 512 `
        --measure-cap direct_answer=1024 --measure-cap zero_shot_cot=4096 `
        --requests data/requests/gsm8k/tranche/<key>.parquet --model-key <key> --out-dir results/d2/<key>_capmeasure
scripts/analyze_tranche.py projects cost from a cap-measurement run at the configured caps (it reads
cap_override from the run log) and warns.

A vllm run (the throughput tranche included) refuses to start when vLLM's FlashInfer sampler will be
used and no CUDA toolkit is discoverable (CUDA_HOME or CUDA_PATH, nvcc on PATH, /usr/local/cuda): run
in the container, set CUDA_HOME, or set VLLM_USE_FLASHINFER_SAMPLER=0.

Bash on a GPU machine (inside the container; see container/README.md):
    python /app/scripts/generate_d2.py --requests /data/requests.parquet --model-key <key> \
        --out-dir /out/<key>

GPUs per job:
    python scripts/generate_d2.py --print-tp --model-key <key>

A full vLLM run needs decoding status "confirmed", a frozen chat template hash and the budget caps
in config/budget.yaml.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import generation as gen  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--requests", help="parquet request file")
    ap.add_argument("--model-key", required=True, help="key in config/models.yaml, dataset2.models")
    ap.add_argument("--out-dir", help="shards and run logs; reuse the same directory to resume")
    ap.add_argument("--backend", choices=["vllm", "echo"], default="vllm")
    ap.add_argument("--limit", type=int, help="first N requests (pilot); with --throughput, the subset size")
    ap.add_argument("--throughput", action="store_true",
                    help="run an arm-stratified subset of --limit requests into <out-dir>/throughput and report tokens/s")
    ap.add_argument("--measure-cap", action="append", default=[], metavar="ARM=N",
                    help="throughput only, repeatable: replace max_new_tokens for ARM by N in this run only, to "
                         "measure output lengths without the configured cap (recorded in the log and fingerprint)")
    ap.add_argument("--shard-size", type=int, default=gen.DEFAULT_SHARD_SIZE,
                    help="distinct generations per chunk (default %(default)s)")
    ap.add_argument("--flush-seconds", type=float, default=gen.DEFAULT_FLUSH_S,
                    help="inside a chunk, write the rows whose outputs have finished at least this often, and soon "
                         "after a stop signal (default %(default)s s)")
    ap.add_argument("--part", metavar="K/N",
                    help="part K of N of the request file, into the same --out-dir as the other parts (one model's "
                         "run on N cards); the folder is complete once all N parts are")
    ap.add_argument("--models-yaml", help="override config/models.yaml")
    ap.add_argument("--decoding-yaml", help="override config/decoding.yaml")
    ap.add_argument("--budget-yaml", help="override config/budget.yaml")
    ap.add_argument("--print-tp", action="store_true", help="print tensor_parallel_size for --model-key and exit")
    ap.add_argument("--print-gpus", action="store_true",
                    help="print the GPUs one job holds (tensor x pipeline parallel size) for --model-key and exit")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.print_tp:
            print(gen.model_tensor_parallel_size(args.model_key, args.models_yaml))
            return gen.EXIT_COMPLETE
        if args.print_gpus:
            print(gen.model_gpu_count(args.model_key, args.models_yaml))
            return gen.EXIT_COMPLETE
        if not args.requests or not args.out_dir:
            raise gen.RunRefused("--requests and --out-dir are required")
        cap_override = gen.parse_cap_overrides(args.measure_cap)
        if cap_override and not args.throughput:
            raise gen.RunRefused("--measure-cap is allowed only together with --throughput")
        part = gen.parse_part(args.part)
        stop = gen.StopFlag().install()
        log = gen.run_generation(
            args.requests, args.model_key, args.out_dir, backend=args.backend, limit=args.limit,
            throughput=args.throughput, shard_size=args.shard_size, flush_seconds=args.flush_seconds,
            models_path=args.models_yaml,
            decoding_path=args.decoding_yaml, budget_path=args.budget_yaml, should_stop=stop,
            cap_override=cap_override or None, part=part)
    except gen.RunRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return gen.EXIT_REFUSED
    except Exception as exc:  # the run log already holds the error; keep the console line short
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return gen.EXIT_ERROR
    t = log.get("totals", {})
    print(json.dumps({"status": log["status"], "run_id": log["run_id"], "unique_generations": t.get("unique_generations"),
                      "cached_calls": t.get("cached_calls"), "runtime_s": log["runtime_s"]}))
    return gen.EXIT_COMPLETE if log["status"] in ("complete", "part_complete") else gen.EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
