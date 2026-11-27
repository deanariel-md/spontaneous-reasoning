"""The paraphrase step of the Dataset 2 bank: frozen record, repeat check, review (scripts/paraphrase_d2.py and the
record and review checks in src/bank_build.py). Nothing here runs a model."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import bank_build as bb, generation as g

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("paraphrase_d2", ROOT / "scripts" / "paraphrase_d2.py")
pd2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd2)

KEY = "llama-3.3-70b-instruct-fp8"
ENTRY = yaml.safe_load((ROOT / "config" / "paraphraser.yaml").read_text(encoding="utf-8"))["dataset2"]["models"][KEY]
ANSWER = "['Work out the math word problem below.', 'Find the answer to this math word problem.']"


def request():
    prompt = "Can you help me write variations ... Original instruction: '''Solve the following math word problem.'''"
    return {"kind": "promptsuite_paraphrase_request", "promptsuite_version": "3.0.7", "field": "instruction",
            "text": "Solve the following math word problem.", "variations_per_field": 3, "n_augments": 2,
            "messages": [{"role": "user", "content": prompt}], "prompt_sha256": bb.sha256_text(prompt)}


def write_request(tmp_path):
    d = tmp_path / "req"
    d.mkdir()
    req = request()
    (d / "request.json").write_text(json.dumps(req), encoding="utf-8")
    pd2.request_rows(req, KEY).to_parquet(d / "requests.parquet", index=False)
    return d, req


def write_run(path: Path, req, requests_dir: Path, text=ANSWER, finish="stop", **log_over):
    path.mkdir(parents=True)
    rid = "para-" + req["prompt_sha256"][:16]
    pd.DataFrame([{"request_id": rid, "model_key": KEY, "arm": "paraphrase", "config_id": "bank-instruction-paraphrase",
                   "item_id": 0, "prompt_sha256": "p" * 64, "output_text": text, "n_prompt_tokens": 300,
                   "n_output_tokens": 40, "finish_reason": finish, "cached": False}]).to_parquet(path / "shard-r1-00000.parquet")
    log = {"run_id": "r1", "status": "complete", "mode": "full", "shards_written": ["shard-r1-00000.parquet"],
           "backend": {"name": "vllm"}, "model": {"hf_id": ENTRY["hf_id"], "revision": ENTRY["revision"]},
           "chat_template": {"observed_sha256": ENTRY["chat_template"]["sha256"], "reads_clock": False},
           "kernels": {"mode": "batch_invariant"}, "gpu": {"devices": [{"name": "RTX PRO 6000"}]},
           "versions": {"vllm": "0.30.0"},
           "inputs": {"requests_sha256": pd2.sha256_bytes((requests_dir / "requests.parquet").read_bytes())}}
    log.update(log_over)
    (path / "run_log-r1.json").write_text(json.dumps(log), encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    block = yaml.safe_load((ROOT / "config" / "promptsuite.yaml").read_text(encoding="utf-8"))
    block["llm"]["record"] = str(tmp_path / "record.json")
    block["llm"]["review"] = str(tmp_path / "review.json")
    p = tmp_path / "promptsuite.yaml"
    p.write_text(yaml.safe_dump(block), encoding="utf-8")
    return p


def freeze(cfg, req_dir, run, repeat=None):
    args = ["--config", str(cfg), "freeze", "--run-dir", str(run), "--request-dir", str(req_dir)]
    return pd2.main(args + (["--repeat-dir", str(repeat)] if repeat else []))


def test_freeze_writes_a_record_the_builder_accepts_and_a_review_to_fill(cfg, tmp_path):
    req_dir, req = write_request(tmp_path)
    run1 = write_run(tmp_path / "run1", req, req_dir)
    run2 = write_run(tmp_path / "run2", req, req_dir)
    assert freeze(cfg, req_dir, run1, run2) == 0
    rec = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    block = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert bb.record_problems(rec, block) == []
    assert rec["parsed"] == ["Work out the math word problem below.", "Find the answer to this math word problem."]
    assert rec["variants"] == [req["text"]] + rec["parsed"] and rec["repeat"]["identical"] is True
    assert rec["model"]["revision"] == ENTRY["revision"] and rec["request"]["prompt_sha256"] == req["prompt_sha256"]
    assert pd2.main(["--config", str(cfg), "review"]) == 0
    rv = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert [v["text"] for v in rv["verdicts"]] == rec["parsed"] and all(v["verdict"] is None for v in rv["verdicts"])
    rec_sha = pd2.sha256_bytes((tmp_path / "record.json").read_bytes())
    assert rv["record_sha256"] == rec_sha
    # an unfilled review refuses; a filled one drops what it marks, with the note
    _, problems = bb.review_rejections(rv, rec_sha, rec["variants"], req["text"])
    assert problems and any("reviewers" in p for p in problems)
    rv["reviewers"] = ["Reviewer A"]
    rv["verdicts"][0].update(verdict="keep")
    rv["verdicts"][1].update(verdict="drop", note="asks for an answer only")
    rejected, problems = bb.review_rejections(rv, rec_sha, rec["variants"], req["text"])
    assert problems == [] and rejected == {rec["parsed"][1]: "asks for an answer only"}
    assert pd2.main(["--config", str(cfg), "review"]) == 2          # never overwritten without --overwrite


@pytest.mark.parametrize("kw,needle", [
    ({"finish": "length"}, "truncated"),
    ({"text": "```python\n['a'\n```"}, "literal_eval"),
    ({"text": "{'a': 1}"}, "list of strings"),
    ({"model": {"hf_id": "x/y", "revision": "0" * 40}}, "registry pins"),
    ({"kernels": {"mode": "default"}}, "kernel mode"),
    ({"backend": {"name": "echo"}}, "not vllm"),
    ({"chat_template": {"observed_sha256": "0" * 64}}, "chat template"),
])
def test_freeze_refuses_what_cannot_be_replayed(cfg, tmp_path, capsys, kw, needle):
    req_dir, req = write_request(tmp_path)
    log_over = {k: v for k, v in kw.items() if k not in ("finish", "text")}
    run = write_run(tmp_path / "run", req, req_dir, text=kw.get("text", ANSWER), finish=kw.get("finish", "stop"),
                    **log_over)
    assert freeze(cfg, req_dir, run) == 2
    assert needle in capsys.readouterr().err and not (tmp_path / "record.json").exists()


def test_freeze_refuses_a_repeat_that_differs(cfg, tmp_path, capsys):
    req_dir, req = write_request(tmp_path)
    run1 = write_run(tmp_path / "run1", req, req_dir)
    run2 = write_run(tmp_path / "run2", req, req_dir, text="['Another answer.']")
    assert freeze(cfg, req_dir, run1, run2) == 2 and "does not regenerate" in capsys.readouterr().err


def test_a_fenced_list_is_replayed_without_its_fence(cfg, tmp_path):
    req_dir, req = write_request(tmp_path)
    fenced = "```python\n" + ANSWER + "\n```"
    run = write_run(tmp_path / "run", req, req_dir, text=fenced)
    assert freeze(cfg, req_dir, run) == 0
    rec = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert rec["response"]["text"] == fenced and rec["replay"]["text"] == ANSWER and rec["replay"]["code_fence_removed"]
    assert bb.record_problems(rec, yaml.safe_load(cfg.read_text(encoding="utf-8"))) == []
    rec["replay"]["text"] = ANSWER + " "
    rec["replay"]["sha256"] = bb.sha256_text(rec["replay"]["text"])
    assert any("only a wrapping" in p for p in bb.record_problems(rec, {"llm": {"model": KEY}}))


def test_strip_code_fence():
    assert bb.strip_code_fence("```python\n['a']\n```") == ("['a']", True)
    assert bb.strip_code_fence("  ```\n['a']\n```\n") == ("['a']", True)
    assert bb.strip_code_fence("['a']") == ("['a']", False)
    assert bb.strip_code_fence("text ```python\n['a']\n```") == ("text ```python\n['a']\n```", False)


def test_promptsuite_variant_rule():
    assert pd2.promptsuite_variants("o", ["a", "b", "a", "c"], 3) == ["o", "a", "b"]
    assert pd2.promptsuite_variants("o", ["a", "o", "b"], 3) == ["a", "o", "b"]


def test_record_and_review_problems_are_reported_at_once():
    block = {"llm": {"model": KEY}}
    assert bb.record_problems({"kind": "other"}, block)
    bad = {"kind": bb.RECORD_KIND, "request": {"prompt_sha256": "x"},
           "response": {"text": "[]", "sha256": "0" * 64, "finish_reason": "length"}, "model": {"key": "other"}}
    probs = bb.record_problems(bad, block)
    assert len(probs) == 5
    review = {"kind": bb.REVIEW_KIND, "record_sha256": "r", "reviewers": ["A"],
              "verdicts": [{"text": "x", "verdict": "drop", "note": ""}, {"text": "x", "verdict": "maybe"},
                           {"text": "not a variant", "verdict": "keep"}]}
    _, probs = bb.review_rejections(review, "other", ["o", "x", "y"], "o")
    for needle in ("another paraphrase record", "without a note", "repeats a text", "must be one of",
                   "not a paraphrased instruction variant", "no verdict for instruction variant 'y'"):
        assert any(needle in p for p in probs), (needle, probs)
