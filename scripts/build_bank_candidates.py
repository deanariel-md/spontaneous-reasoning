r"""Build a candidate Dataset 2 prompt bank with PromptSuite 3.0.7.

Runs inside the isolated environment `.venv-promptsuite` (PromptSuite would downgrade packages pinned in the
project environment). Imports are the standard library, pandas, numpy, yaml and
promptsuite, plus src/bank_build.py (standard library only) and src/scoring.py (candidate names only).

No design choice is made here: config/promptsuite.yaml has the status PENDING_REVIEW and this script refuses to
build from it. `--example` builds from the labelled example block so the machinery can be exercised; that bank
carries "EXAMPLE, not a registered bank" in its generator field.

What it does:
  1. generates skeleton variants once with PromptSuite from a frame whose first row holds the placeholders
     {question}, {arm_clause} and {answer_format} as values (no gold column, no few-shot), plus one check row per arm
     that holds the real clause texts;
  2. reads the instruction and prompt-format variants from `field_values` and deduplicates them;
  3. drops every skeleton that loses or duplicates a placeholder (logged in generation_record.dropped);
  4. fills the arm clause per arm and the answer format, leaving {question} as the only item variable, and compares
     the filled messages with PromptSuite's own `conversation` for the check rows;
  5. writes data/bank_candidates/<name>.json in the src/d2_io.py bank schema: generator, generator_version, seed,
     sampling_probabilities, configurations (dimensions plus both arms' messages), and a generation_record.

The script sets PYTHONHASHSEED=0 and PYTHON_DOTENV_DISABLED=1 by re-executing itself: PromptSuite builds some
variant lists from Python sets, so the variants kept can depend on string hashing, and PromptSuite calls
load_dotenv() at import, which would read a .env file from the directories above the package (the environment
sits inside the repository). The dotenv loader is also replaced by a no-op before PromptSuite is imported, and
every API-key variable is removed from the process environment unless an LLM variation type is configured.

Usage, the labelled example:
    .\.venv-promptsuite\Scripts\python.exe scripts\build_bank_candidates.py --example

Usage, the configured bank:
    .\.venv-promptsuite\Scripts\python.exe scripts\build_bank_candidates.py

Usage, with a hosted-LLM paraphrase (paid and unseeded; the key is read from the
environment variable of llm.platform, for example $env:TOGETHER_API_KEY, and never from a file):
    .\.venv-promptsuite\Scripts\python.exe scripts\build_bank_candidates.py --allow-llm-api

With llm.platform local_replay, PromptSuite's paraphrase call is answered from llm.record, the frozen answer of the
pinned paraphraser (config/paraphraser.yaml; scripts/paraphrase_d2.py). The request PromptSuite sends must have the
recorded sha256, no API is called and no key is read. llm.review holds the hand review of the paraphrases; a variant
marked drop is removed with its note, and a variant without a verdict refuses the build.

A candidate is never written to manifests/. Freezing one into manifests/gsm8k_bank.json is a separate step.

Exit codes: 0 written (or already identical), 2 refused, 1 any other error.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import bank_build as bb  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
DEFAULT_CONFIG = ROOT / "config" / "promptsuite.yaml"
DEFAULT_OUT_DIR = ROOT / "data" / "bank_candidates"
FORCED_ENV = {"PYTHONHASHSEED": "0", "PYTHON_DOTENV_DISABLED": "1"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="default config/promptsuite.yaml")
    ap.add_argument("--example", action="store_true",
                    help="build from the `example:` block (allowed while the status is PENDING_REVIEW); the output is "
                         "marked 'EXAMPLE, not a registered bank'")
    ap.add_argument("--allow-llm-api", action="store_true",
                    help="permit a hosted-LLM variation type (needs llm.allowed: true and the platform's API key in "
                         "the environment). Paid and unseeded")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="default data/bank_candidates; never a folder under manifests/")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing candidate file whose content differs")
    return ap


def ensure_environment() -> None:
    """Re-execute with PYTHONHASHSEED=0 and PYTHON_DOTENV_DISABLED=1 unless they are already in effect."""
    if all(os.environ.get(k) == v for k, v in FORCED_ENV.items()) and sys.flags.hash_randomization == 0:
        return
    env = dict(os.environ)
    env.update(FORCED_ENV)
    sys.exit(subprocess.call([sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env=env))


def candidate_specs() -> tuple[dict, str]:
    """Scorer candidate names from src/scoring.py as {name: (template, case_sensitive)}, and the scorer version."""
    from src import scoring
    return ({n: (s.template, s.case_sensitive) for n, s in scoring.CANDIDATE_SPECS.items()}, scoring.SCORER_VERSION)


def load_config(path: str) -> dict:
    import yaml
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise bb.BankBuildError(f"cannot read the configuration {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise bb.BankBuildError(f"{path} must hold a YAML mapping")
    return cfg


def read_llm_key(block: dict, environ) -> str | None:
    if not bb.llm_types_used(block):
        return None
    return environ.get(bb.PLATFORM_KEY_VARS[block["llm"]["platform"]])


def run_promptsuite(block: dict, template: dict, llm_key: str | None, completion=None) -> tuple[list[dict], str, dict]:
    """Generate the skeleton variants. Returns PromptSuite's output rows, its captured console text, and versions.
    The dotenv loader is neutralised before PromptSuite is imported. `completion` replaces the function PromptSuite's
    paraphrase augmenter calls (the replay of a frozen record); PromptSuite then gets a placeholder key, because its
    factory builds the paraphrase augmenter only when a key is set."""
    if sys.flags.hash_randomization != 0:
        raise bb.BankBuildError("PYTHONHASHSEED=0 is not in effect (PromptSuite builds some variant lists from Python "
                                "sets); run the script directly so that it re-executes itself, or set PYTHONHASHSEED=0")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False
    import numpy
    import pandas as pd
    import promptsuite
    from importlib.metadata import version
    if completion is not None:
        import promptsuite.augmentations.text.paraphrase as ps_paraphrase
        ps_paraphrase.get_completion = completion

    installed = version("promptsuite")
    if installed != bb.PROMPTSUITE_VERSION:
        raise bb.BankBuildError(f"promptsuite {installed} is installed; the builder follows version "
                                f"{bb.PROMPTSUITE_VERSION} (requirements-promptsuite.txt)")
    clauses, fmt = block["arm_clauses"], block["answer_format"]["text"]
    rows = [{"question": bb.PH_QUESTION, "arm_clause": bb.PH_ARM, "answer_format": bb.PH_FORMAT}]
    rows += [{"question": bb.PH_QUESTION, "arm_clause": clauses[a], "answer_format": fmt} for a in bb.ARMS]
    frame = pd.DataFrame(rows)
    kwargs = dict(variations_per_field=block["variations_per_field"], max_variations_per_row=block["max_variations_per_row"],
                  random_seed=block["seed"])
    if llm_key is not None:
        kwargs.update(api_platform=block["llm"]["platform"], api_key=llm_key, model_name=block["llm"]["model"])
    elif completion is not None:
        kwargs.update(api_key="local-replay-no-api", model_name=block["llm"]["model"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ps = promptsuite.PromptSuite()
        ps.load_dataframe(frame)
        ps.set_template(template)
        ps.configure(**kwargs)
        out = ps.generate()
    versions = {"promptsuite_version": installed, "python_version": platform.python_version(),
                "pandas_version": pd.__version__, "numpy_version": numpy.__version__}
    return out, buf.getvalue(), versions


def split_rows(out: list[dict]) -> dict[int, list[dict]]:
    by_row: dict[int, list[dict]] = {}
    for o in out:
        by_row.setdefault(int(o["original_row_index"]), []).append(o)
    return by_row


def pair_of(o: dict) -> tuple[str, str]:
    fv = o["field_values"]
    return (fv.get("instruction variations") or "", fv["prompt format variations"])


def fidelity_check(by_row: dict[int, list[dict]], table: dict, block: dict) -> dict:
    """Compare the messages built here with PromptSuite's for the check rows (one per arm): its `conversation`, or
    with the instruction placed in the user turn, its `prompt` string as the one user message."""
    user_turn = bb.placement_of(block) == "user"
    kept = {(k["instruction"], k["prompt_format"]): k for k in table["kept"]}
    ref_pairs = [pair_of(o) for o in by_row[0]]
    compared = 0
    for r, arm in ((1, bb.ARMS[0]), (2, bb.ARMS[1])):
        pairs = [pair_of(o) for o in by_row[r]]
        if pairs != ref_pairs:
            raise bb.BankBuildError(f"PromptSuite produced different skeleton variants for check row {r} than for the "
                                    "placeholder row; the skeleton is not independent of the row")
        for o in by_row[r]:
            k = kept.get(pair_of(o))
            if k is None:
                continue
            theirs = [{"role": "user", "content": o["prompt"]}] if user_turn else o["conversation"]
            if theirs != k["messages"][arm]:
                raise bb.BankBuildError(
                    f"the messages built here differ from PromptSuite's {'prompt' if user_turn else 'conversation'} "
                    f"for arm {arm}: {k['messages'][arm]!r} versus {theirs!r}")
            compared += 1
    return {"rows_compared": ["direct_answer", "zero_shot_cot"], "messages_compared": compared, "result": "identical",
            "compared_with": "prompt" if user_turn else "conversation"}


def read_json_input(rel: str) -> tuple[dict, str, str]:
    """A JSON input named in the configuration (relative to the repository root): content, path, sha256 of the bytes."""
    path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8")), str(rel), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        raise bb.BankBuildError(f"cannot read {rel}: {exc}") from exc


def make_replay(block: dict) -> tuple[object, dict, list]:
    """The completion function that answers PromptSuite's paraphrase request from the frozen record, the record's
    provenance, and the list of request hashes it saw (checked after generation)."""
    record, path, sha = read_json_input(block["llm"]["record"])
    problems = bb.record_problems(record, block)
    if problems:
        raise bb.BankBuildError(f"paraphrase record {path}: " + "; ".join(problems))
    seen: list[str] = []
    want, answer = record["request"]["prompt_sha256"], record["replay"]["text"]

    def completion(prompt, *args, **kwargs):
        seen.append(bb.sha256_text(prompt))
        return answer if seen[-1] == want else "[]"

    return completion, {"record": path, "record_sha256": sha, "request_sha256": want,
                        "response_sha256": record["response"]["sha256"], "replay_sha256": record["replay"]["sha256"],
                        "code_fence_removed": record["replay"]["code_fence_removed"], "paraphraser": record["model"]}, seen


def json_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def run(args) -> dict:
    cfg = load_config(args.config)
    bb.check_status(cfg, args.example)
    candidates, scorer_version = candidate_specs()
    problems = bb.validate_config(cfg, args.example, candidates)
    if problems:
        raise bb.BankBuildError("configuration invalid: " + "; ".join(problems))
    block = bb.active_block(cfg, args.example)
    gate = bb.llm_gate(block, args.allow_llm_api, os.environ)
    if gate:
        raise bb.BankBuildError("hosted LLM not permitted: " + "; ".join(gate))
    out_path = Path(args.out_dir) / f"{block['name']}.json"
    problem = bb.output_path_problem(out_path, ROOT)
    if problem:
        raise bb.BankBuildError(problem)
    existing = out_path.read_bytes() if out_path.exists() else None

    replay = bb.replay_used(block)
    completion, replay_info, seen = make_replay(block) if replay else (None, None, [])
    llm_key = None if replay else read_llm_key(block, os.environ)
    bb.scrub_api_env(os.environ)
    template = bb.make_template(block)
    out, log, versions = run_promptsuite(block, template, llm_key, completion)
    if replay:
        if not seen:
            raise bb.BankBuildError("PromptSuite made no paraphrase request, so the record was not used")
        wrong = sorted(set(s for s in seen if s != replay_info["request_sha256"]))
        if wrong:
            raise bb.BankBuildError(f"PromptSuite sent a paraphrase request with sha256 {wrong[0]}, but the record answers "
                                    f"{replay_info['request_sha256']}; the instruction, variations_per_field or "
                                    "PromptSuite changed since the record was made")
    log_lines = bb.log_problems(log)
    if log_lines:
        raise bb.BankBuildError("PromptSuite reported a warning, fallback or error: " + " | ".join(log_lines[:5]))
    by_row = split_rows(out)
    if set(by_row) != {0, 1, 2}:
        raise bb.BankBuildError(f"PromptSuite returned rows {sorted(by_row)}, expected 0, 1 and 2")
    pairs = [pair_of(o) for o in by_row[0]]
    table = bb.build_skeletons(pairs, block)
    if replay:
        review, review_path, review_sha = read_json_input(block["llm"]["review"])
        rejected, problems = bb.review_rejections(review, replay_info["record_sha256"], table["instruction_variants"],
                                                  block["instruction"]["text"] or None)
        if problems:
            raise bb.BankBuildError(f"paraphrase review {review_path}: " + "; ".join(problems))
        table = bb.build_skeletons(pairs, block, rejected)
        replay_info.update(review=review_path, review_sha256=review_sha, reviewers=list(review["reviewers"]),
                           dropped_in_review=rejected)
    fidelity = fidelity_check(by_row, table, block)

    warnings = []
    for field in ("instruction", "prompt_format"):
        types = block[field]["variation_types"]
        obtained = len(table[f"{field}_variants"])
        if types and block["max_variations_per_row"] is None and obtained < block["variations_per_field"]:
            warnings.append(f"{field}: PromptSuite returned {obtained} of the {block['variations_per_field']} requested variants")
        if len(types) > 1:
            warnings.append(f"{field}: several variation types are listed; PromptSuite 3.0.7 concatenates their variants in "
                            "order and truncates to variations_per_field, so later types can be crowded out and the "
                            "variants cannot be attributed to a type")
    if block["max_variations_per_row"] is not None:
        warnings.append(f"max_variations_per_row is {block['max_variations_per_row']}: PromptSuite kept a random subset of the "
                        "combinations of the two fields, and the variant lists hold only the variants that occur in it "
                        "(the original text can be absent)")
    stray = bb.stray_placeholders([v for v in table["instruction_variants"] + table["prompt_format_variants"]])
    if stray:
        warnings.append(f"brace tokens other than the three placeholders occur in the skeleton text: {stray}")
    if table["dropped"]:
        warnings.append(f"{len(table['dropped'])} skeletons were dropped (generation_record.dropped)")
    if len(table["kept"]) == 1:
        warnings.append("the bank holds a single configuration")

    llm_used = bool(bb.llm_types_used(block))
    spec_template, case_sensitive = candidates[block["answer_format"]["scorer_candidate"]]
    record = {
        "tool": "scripts/build_bank_candidates.py",
        "config_status": "EXAMPLE" if args.example else cfg["status"],
        "config_name": block["name"], "config_sha256": json_sha(block),
        **versions,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "seed": block["seed"], "variations_per_field": block["variations_per_field"],
        "max_variations_per_row": block["max_variations_per_row"],
        "template": template,
        "variation_types": {"instruction": list(block["instruction"]["variation_types"]),
                            "prompt_format": list(block["prompt_format"]["variation_types"])},
        "placeholders": list(bb.PLACEHOLDERS),
        "arm_clauses": dict(block["arm_clauses"]),
        "answer_format": {"scorer_candidate": block["answer_format"]["scorer_candidate"],
                          "text": block["answer_format"]["text"], "scorer_template": spec_template,
                          "scorer_case_sensitive": case_sensitive, "scorer_version": scorer_version},
        "sampling_law": block["sampling_law"], "required_literals": block["required_literals"],
        "llm": {"used": llm_used, "platform": block["llm"]["platform"] if llm_used else None,
                "model": block["llm"]["model"] if llm_used else None, **({"replay": replay_info} if replay else {})},
        "instruction_placement": bb.placement_of(block),
        "variants": {"instruction": table["instruction_variants"], "prompt_format": table["prompt_format_variants"]},
        "variant_sha256": bb.variant_sha256(table["instruction_variants"], table["prompt_format_variants"]),
        "n_skeleton_pairs": table["n_pairs"], "n_configurations": len(table["kept"]),
        "dropped": table["dropped"], "fidelity_check": fidelity, "warnings": warnings,
    }
    bank = bb.assemble_bank(block, table, generator_version=versions["promptsuite_version"], example=args.example,
                            generation_record=record)
    raw = bb.canonical_bank_bytes(bank)
    if existing is not None and existing != raw and not args.overwrite:
        raise bb.BankBuildError(f"{out_path} exists with different content; pass --overwrite to replace it")
    status = "unchanged" if existing == raw else ("replaced" if existing is not None else "written")
    if status != "unchanged":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, out_path)
    return {"status": status, "path": str(out_path), "sha256": hashlib.sha256(raw).hexdigest(),
            "generator": bank["generator"], "n_configurations": len(bank["configurations"]),
            "n_skeleton_pairs": table["n_pairs"], "n_dropped": len(table["dropped"]),
            "variants_instruction": len(table["instruction_variants"]),
            "variants_prompt_format": len(table["prompt_format_variants"]),
            "llm_used": llm_used, "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "warnings": warnings}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")      # config text may hold characters a console cannot print
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except bb.BankBuildError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    for w in result["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps(result, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    ensure_environment()
    sys.exit(main())
