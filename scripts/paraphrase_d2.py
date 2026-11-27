r"""The paraphrase step of the Dataset 2 prompt bank.

PromptSuite 3.0.7's paraphrase augmenter sends one fixed rewriting prompt for the paraphrased field and parses the
answer as a Python list. Its hosted call is unseeded, so here the answer comes from the pinned paraphraser of
config/paraphraser.yaml, generated once on a GPU machine by the Dataset 2 runner (greedy, batch-invariant kernels), and
scripts/build_bank_candidates.py replays it (llm.platform: local_replay).

  request   render PromptSuite's request for the configuration (PromptSuite runs in .venv-promptsuite) and
            write it with the one-row runner request file
  freeze    check the runner's output folder and write the record the builder replays (llm.record). --repeat-dir
            names a second run in a fresh folder, whose answer must be identical
  review    write the review file for the hand check of meaning (llm.review), one empty verdict per paraphrase

Usage:
    .\.venv\Scripts\python.exe scripts\paraphrase_d2.py request
    .\.venv\Scripts\python.exe scripts\paraphrase_d2.py freeze --run-dir <run 1 folder> --repeat-dir <run 2 folder>
    .\.venv\Scripts\python.exe scripts\paraphrase_d2.py review

Exit codes: 0 written (or already identical), 2 refused, 1 any other error.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import bank_build as bb, generation as g  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
CONFIG = ROOT / "config" / "promptsuite.yaml"
PARAPHRASER = ROOT / "config" / "paraphraser.yaml"
REQUEST_DIR = ROOT / "data" / "bank_candidates" / "paraphrase"
VENV_PY = ROOT / ".venv-promptsuite" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ARM = "paraphrase"

# Runs in .venv-promptsuite: PromptSuite's own template and prompt builder, as its factory calls them
# (Paraphrase(n_augments=variations_per_field - 1), then augment(text) formats instruction_template).
_RENDER = r"""
import json, sys
from importlib.metadata import version
from promptsuite.augmentations.text.paraphrase import Paraphrase, instruction_template
text, vpf = sys.argv[1], int(sys.argv[2])
p = Paraphrase(n_augments=vpf - 1, api_key="unused")
print(json.dumps({"prompt": p.build_rephrasing_prompt(instruction_template, p.n_augments, text),
                  "n_augments": p.n_augments, "promptsuite_version": version("promptsuite")}))
"""


class Refused(Exception):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: Path, obj) -> str:
    raw = (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def confirmed_block(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("status") != bb.STATUS_CONFIRMED:
        raise Refused(f"{path} is not confirmed")
    if not bb.replay_used(cfg):
        raise Refused(f"{path} does not replay a paraphrase (llm.platform {bb.REPLAY_PLATFORM!r} with paraphrase_with_llm)")
    return cfg


def paraphrased_field(block: dict) -> str:
    fields = [f for f in ("instruction", "prompt_format") if "paraphrase_with_llm" in block[f]["variation_types"]]
    if len(fields) != 1:
        raise Refused(f"exactly one field may be paraphrased; the configuration paraphrases {fields}")
    return fields[0]


def paraphraser_entry(key: str, path: Path = PARAPHRASER) -> dict:
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset2"]["models"]
    if key not in reg:
        raise Refused(f"llm.model {key!r} is not an entry of {path.name}")
    return reg[key]


def request_rows(req: dict, model_key: str) -> pd.DataFrame:
    return pd.DataFrame([{"request_id": "para-" + req["prompt_sha256"][:16], "model_key": model_key, "arm": ARM,
                          "config_id": f"bank-{req['field']}-paraphrase", "item_id": 0,
                          "messages_json": json.dumps(req["messages"], ensure_ascii=False)}])


def cmd_request(args) -> dict:
    block = confirmed_block(Path(args.config))
    field = paraphrased_field(block)
    paraphraser_entry(block["llm"]["model"])
    text = block[field]["text"]
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHON_DOTENV_DISABLED": "1", "PYTHONIOENCODING": "utf-8"}
    for v in list(bb.PLATFORM_KEY_VARS.values()) + [bb.GENERIC_KEY_VAR]:
        env.pop(v, None)
    res = subprocess.run([str(VENV_PY), "-I", "-c", _RENDER, text, str(block["variations_per_field"])],
                         capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"rendering in .venv-promptsuite failed: {res.stderr.strip()[-800:]}")
    r = json.loads(res.stdout.strip().splitlines()[-1])
    req = {"kind": "promptsuite_paraphrase_request", "promptsuite_version": r["promptsuite_version"], "field": field,
           "text": text, "variations_per_field": block["variations_per_field"], "n_augments": r["n_augments"],
           "messages": [{"role": "user", "content": r["prompt"]}], "prompt_sha256": bb.sha256_text(r["prompt"])}
    out = Path(args.out_dir)
    write_json(out / "request.json", req)
    request_rows(req, block["llm"]["model"]).to_parquet(out / "requests.parquet", index=False)
    return {"request": str(out / "request.json"), "requests": str(out / "requests.parquet"),
            "prompt_sha256": req["prompt_sha256"], "n_augments": req["n_augments"]}


def check_run(run_dir: Path, req: dict, requests_sha: str, model_key: str, entry: dict) -> dict:
    """The one output of a complete vLLM run of the request, with its provenance."""
    out = g.read_outputs(run_dir)
    if len(out) != 1:
        raise Refused(f"{run_dir}: expected one output, found {len(out)}")
    row = out.iloc[0]
    rid = "para-" + req["prompt_sha256"][:16]
    if row["request_id"] != rid or row["model_key"] != model_key or row["arm"] != ARM:
        raise Refused(f"{run_dir}: the output answers {row['request_id']!r} by {row['model_key']!r}, expected {rid!r} "
                      f"by {model_key!r}")
    if row["finish_reason"] != "stop":
        raise Refused(f"{run_dir}: the answer stopped with {row['finish_reason']!r}; PromptSuite could not parse a "
                      "truncated list")
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("run_log-*.json"))]
    done = [lg for lg in logs if lg.get("status") == "complete" and lg.get("shards_written")]
    if len(done) != 1:
        raise Refused(f"{run_dir}: expected one complete run log that wrote the output, found {len(done)}")
    log = done[0]
    model, ct = log.get("model") or {}, log.get("chat_template") or {}
    problems = []
    if (log.get("backend") or {}).get("name") != "vllm":
        problems.append("the backend was not vllm")
    if (model.get("hf_id"), model.get("revision")) != (entry["hf_id"], entry["revision"]):
        problems.append(f"the run used {model.get('hf_id')}@{model.get('revision')}, the registry pins "
                        f"{entry['hf_id']}@{entry['revision']}")
    if ct.get("observed_sha256") != entry["chat_template"]["sha256"] or ct.get("reads_clock"):
        problems.append("the chat template is not the recorded one, or it reads the clock")
    if (log.get("kernels") or {}).get("mode") != entry["kernels"]:
        problems.append(f"kernel mode {(log.get('kernels') or {}).get('mode')!r}, expected {entry['kernels']!r}")
    if (log.get("inputs") or {}).get("requests_sha256") != requests_sha:
        problems.append("the run read another request file")
    if log.get("mode") != "full":
        problems.append(f"run mode {log.get('mode')!r}")
    if problems:
        raise Refused(f"{run_dir}: " + "; ".join(problems))
    log_path = next(p for p in sorted(run_dir.glob("run_log-*.json"))
                    if json.loads(p.read_text(encoding="utf-8")).get("run_id") == log.get("run_id"))
    return {"text": str(row["output_text"]), "finish_reason": "stop", "n_output_tokens": int(row["n_output_tokens"]),
            "prompt_sha256": str(row["prompt_sha256"]), "run_id": log.get("run_id"), "gpu": log.get("gpu"),
            "versions": {k: (log.get("versions") or {}).get(k) for k in ("vllm", "torch", "transformers")},
            "run_log": str(log_path), "run_log_sha256": sha256_bytes(log_path.read_bytes())}


def promptsuite_variants(original: str, parsed: list, vpf: int) -> list[str]:
    """The variants PromptSuite 3.0.7 keeps for a paraphrased field: the parsed list cut to variations_per_field,
    duplicates removed in order, the original inserted first when absent, cut to variations_per_field again."""
    uniq = list(dict.fromkeys(parsed[:vpf]))
    if original not in uniq:
        uniq.insert(0, original)
    return uniq[:vpf]


def cmd_freeze(args) -> dict:
    block = confirmed_block(Path(args.config))
    req_dir = Path(args.request_dir)
    req = json.loads((req_dir / "request.json").read_text(encoding="utf-8"))
    requests_sha = sha256_bytes((req_dir / "requests.parquet").read_bytes())
    key = block["llm"]["model"]
    entry = paraphraser_entry(key)
    first = check_run(Path(args.run_dir), req, requests_sha, key, entry)
    replay_text, fenced = bb.strip_code_fence(first["text"])
    try:
        parsed = ast.literal_eval(replay_text)
    except (ValueError, SyntaxError) as exc:
        raise Refused(f"PromptSuite parses the answer with ast.literal_eval and would fail ({exc}), also after a "
                      "wrapping code fence is removed; it would fall back to the original text") from exc
    if not (isinstance(parsed, list) and all(isinstance(x, str) for x in parsed)):
        raise Refused("the answer is not a Python list of strings")
    repeat = None
    if args.repeat_dir:
        second = check_run(Path(args.repeat_dir), req, requests_sha, key, entry)
        if second["text"] != first["text"]:
            raise Refused("the repeat run gave a different answer; the paraphrase does not regenerate exactly")
        repeat = {k: second[k] for k in ("run_id", "run_log", "run_log_sha256", "gpu")} | {"identical": True}
    record = {"kind": bb.RECORD_KIND, "config_name": block["name"],
              "request": {k: req[k] for k in ("promptsuite_version", "field", "text", "variations_per_field",
                                              "n_augments", "prompt_sha256", "messages")},
              "response": {"text": first["text"], "sha256": bb.sha256_text(first["text"]),
                           "finish_reason": first["finish_reason"], "n_output_tokens": first["n_output_tokens"]},
              "replay": {"text": replay_text, "sha256": bb.sha256_text(replay_text), "code_fence_removed": fenced,
                         "note": ("the Markdown code fence around the list is removed, because PromptSuite's "
                                  "ast.literal_eval cannot parse it" if fenced else "the response unchanged")},
              "parsed": parsed,
              "variants": promptsuite_variants(req["text"], parsed, req["variations_per_field"]),
              "model": {"key": key, "hf_id": entry["hf_id"], "revision": entry["revision"], "kernels": entry["kernels"],
                        "quantization": entry["quantization"], "chat_template_sha256": entry["chat_template"]["sha256"],
                        "rendered_prompt_sha256": first["prompt_sha256"], "run_id": first["run_id"],
                        "run_log": first["run_log"], "run_log_sha256": first["run_log_sha256"], "gpu": first["gpu"],
                        "versions": first["versions"]},
              "repeat": repeat}
    out = ROOT / block["llm"]["record"]
    sha = write_json(out, record)
    return {"record": str(out), "sha256": sha, "n_parsed": len(parsed), "n_variants": len(record["variants"]),
            "repeat_identical": bool(repeat)}


def cmd_review(args) -> dict:
    block = confirmed_block(Path(args.config))
    rec_path, out = ROOT / block["llm"]["record"], ROOT / block["llm"]["review"]
    raw = rec_path.read_bytes()
    record = json.loads(raw.decode("utf-8"))
    if out.exists() and not args.overwrite:
        raise Refused(f"{out} exists; pass --overwrite to start the review again")
    review = {"kind": bb.REVIEW_KIND, "record_sha256": sha256_bytes(raw), "reviewers": [], "date": None,
              "question": ("Does the variant ask for the same task as the original instruction, with nothing added "
                           "or removed? keep or drop; a drop needs a note"),
              "original": record["request"]["text"],
              "verdicts": [{"text": t, "verdict": None, "note": ""} for t in record["variants"]
                           if t != record["request"]["text"]]}
    write_json(out, review)
    return {"review": str(out), "n_verdicts": len(review["verdicts"])}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("request")
    r.add_argument("--out-dir", default=str(REQUEST_DIR))
    f = sub.add_parser("freeze")
    f.add_argument("--run-dir", required=True, help="the runner's output folder for the paraphraser")
    f.add_argument("--repeat-dir", help="a second run in a fresh folder; its answer must be identical")
    f.add_argument("--request-dir", default=str(REQUEST_DIR))
    v = sub.add_parser("review")
    v.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    try:
        res = {"request": cmd_request, "freeze": cmd_freeze, "review": cmd_review}[args.cmd](args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(res, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
