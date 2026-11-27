"""scripts/restore_layout.py: request keys, the temperature-check seeds, the cached column, the copy of the analysis
inputs (with its sha256 check) and the rebuilt API reply file, on a small synthetic deposit."""
import hashlib
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import restore_layout as rl  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["arm", "config_id", "item_id"])


def test_request_ids_follow_the_builders_suffixes():
    df = frame([("direct_answer", "ps-i05-f04", 269), ("forced_direct", "ps-i05-f04", 269),
                ("zero_shot_cot", "ps-i05-f04", 269)])
    assert rl.request_ids(df, "gsm8k").tolist() == ["ps-i05-f04|direct_answer|269", "ps-i05-f04|forced_direct|269",
                                                    "ps-i05-f04|zero_shot_cot|269"]
    assert rl.request_ids(df, "forced").tolist()[1] == "ps-i05-f04|direct_answer|269|fd"
    lad = frame([("fewshot_forced", "c", "boolean_expressions/1"), ("fewshot_requested", "bbh/c", "x/2")])
    assert rl.request_ids(lad, "ladder2").tolist() == ["c|direct_answer|boolean_expressions/1|fd|fs",
                                                       "bbh/c|direct_answer|x/2|fsr"]
    hq = frame([("fewshot_forced", "ps-i00-f00", "5ae0")])
    assert rl.request_ids(hq, "ladder2/hotpot").tolist() == ["ps-i00-f00|direct_answer|5ae0|fs"]
    assert rl.request_ids(df.iloc[:1], "gsm8k", split="train").tolist() == ["train|ps-i05-f04|direct_answer|269"]


def test_thinking_pilot_keys_drop_the_task_from_the_item():
    df = frame([("thinking_gsm8k", "ps-i01-f02", "gsm8k:4363")])
    assert rl.request_ids(df, "thinking_pilot").tolist() == ["train|ps-i01-f02|direct_answer|4363|th"]


def test_temperature_check_seed_rule():
    rid = "ps-i00-f05|direct_answer|test/algebra/1004.json"
    want = int(hashlib.sha256(("temp_check/t0.7|" + rid).encode()).hexdigest()[:8], 16) % 2 ** 31
    assert rl._seed(rid) == want


def test_cached_column_is_added_once_after_its_neighbour():
    df = pd.DataFrame({"a": [1], "finish_reason": ["stop"], "b": [2]})
    out = rl.with_cached(df, "finish_reason")
    assert list(out.columns) == ["a", "finish_reason", "cached", "b"] and not out["cached"].any()
    assert rl.with_cached(out, "a") is out


def deposit(tmp_path, files):
    dep = tmp_path / "dep"
    (dep / "models").mkdir(parents=True)
    (dep / "analysis_inputs").mkdir()
    (dep / "analysis_inputs" / "build_records.jsonl").write_text("", encoding="utf-8")
    (dep / "SOURCES.csv").write_text("file,built_from\n", encoding="utf-8")
    pd.DataFrame(columns=["collection", "folder", "model_key", "gpu", "requests_sha256"]).to_csv(
        dep / "models" / "runs.csv", index=False)
    index = []
    for name, (restore_to, data, how) in files.items():
        (dep / "analysis_inputs" / name).write_bytes(data)
        index.append({"file": name, "restore_to": restore_to, "how": how,
                      "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    (dep / "analysis_inputs" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return dep


def test_inputs_are_copied_and_checked(tmp_path):
    dep = deposit(tmp_path, {"a.json": ("results/x/a.json", b'{"v": 1}\n', "copy"),
                             "b.parquet": ("data/y/raw.jsonl", b"not copied", "build")})
    root = tmp_path / "root"
    assert rl.main([str(dep), "--root", str(root), "--only", "inputs"]) == 0
    assert (root / "results/x/a.json").read_bytes() == b'{"v": 1}\n'
    assert not (root / "data/y/raw.jsonl").exists()
    rec = json.loads((root / "data/restore_record.json").read_text(encoding="utf-8"))
    assert [f["file"] for f in rec["files"]] == ["results/x/a.json"]
    (dep / "analysis_inputs" / "a.json").write_bytes(b"changed")
    with pytest.raises(rl.RestoreError):
        rl.main([str(dep), "--root", str(root), "--only", "inputs", "--force"])


def test_raw_reply_file_is_rebuilt_from_outputs_and_records(tmp_path):
    dep = deposit(tmp_path, {})
    (dep / "ladder").mkdir()
    for b in rl.LADDER_TASKS:
        rows = [] if b != "gsm8k" else [{"arm": "direct_answer", "config_id": "c1", "item_id": "7",
                                          "output_text": "\\boxed{3}"}]
        pd.DataFrame(rows, columns=["arm", "config_id", "item_id", "output_text"]).to_parquet(
            dep / "ladder" / f"{b}_outputs_m.parquet")
    meta = pd.DataFrame([
        {"key": "gsm8k|c1|direct_answer|7", "task": "gsm8k", "request_id": "c1|direct_answer|7",
         "arm": "direct_answer", "config_id": "c1", "item_id": "7", "phase": "pilot", "status": 200,
         "cost_usd": 1e-5, "absent": "variant,extra_params,prefix_echoed,stop_reason,panel_cap,n_prompt_tokens_local,"
         "prompt_sha256,tries,seconds,t,error,design,cap", "usage_prompt_tokens": 5, "usage_completion_tokens": 2,
         "usage_cached_tokens": 0, "extra_params": None, "text_from": "outputs", "text": None,
         "finish_reason": "stop"},
        {"key": "gsm8k|c1|direct_answer|7", "task": "gsm8k", "request_id": "c1|direct_answer|7",
         "arm": "direct_answer", "config_id": "c1", "item_id": "7", "phase": "complete", "status": 200,
         "cost_usd": 2e-5, "absent": "variant,extra_params,prefix_echoed,stop_reason,panel_cap,"
         "n_prompt_tokens_local,prompt_sha256,tries,seconds,t,error,design,cap", "usage_prompt_tokens": 5,
         "usage_completion_tokens": 3, "usage_cached_tokens": 0, "extra_params": None, "text_from": "own",
         "text": "4\n\\boxed{4}", "finish_reason": "stop"}])
    meta.to_parquet(dep / "analysis_inputs" / "meta.parquet")
    ctx = rl.Ctx(dep, tmp_path / "root", True)
    rl.write_raw(ctx, "meta.parquet", "m", "data/raw.jsonl")
    lines = [json.loads(x) for x in (tmp_path / "root" / "data" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["text"] for x in lines] == ["\\boxed{3}", "4\n\\boxed{4}"]
    assert lines[0]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "cached_tokens": 0}
    assert lines[1]["cost_usd"] == 2e-5 and "t" not in lines[0] and lines[0]["status"] == 200
