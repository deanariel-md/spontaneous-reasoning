"""Five-shot forced-direct requests on MATH-500 (pool rule, saved demonstrations, request files, context
check, scoring and analysis paths)."""
import hashlib
import json
import os
import sys

import pandas as pd
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_fewshot_requests as bf  # noqa: E402
import run_fewshot as rf  # noqa: E402
import score_fewshot as sf  # noqa: E402

DEMOS = os.path.join(ROOT, "manifests", "fewshot_demos_math.json")
REC = os.path.join(ROOT, "data", "requests", "fewshot", "build_record.json")


def test_seed_is_the_hash_prefix():
    want = int.from_bytes(hashlib.sha256(b"fewshot/math/ps-i00-f05").digest()[:8], "big")
    assert bf.demo_seed("math", "ps-i00-f05") == want


def test_pool_rule_on_a_toy_table():
    train = pd.DataFrame({"item_id": ["train/a/0", "train/a/1", "train/a/2", "train/b/0", "train/b/1"],
                          "question": ["q0", "q1", "same as target ", "q3", "q4"],
                          "gold_answer": ["1", None, "2", "3/2", "-4"]})
    pool, info = bf.math_pool_ids(train, {"train/b/1"}, {"same as target"})
    assert pool == ["train/a/0", "train/b/0"]             # training order kept
    assert info["n_numeric"] == 4 and info["n_excluded_sets"] == 1 and info["same_text_ids"] == ["train/a/2"]


def test_math_uses_the_configuration_as_prefix_key():
    assert bf.prefix_key("math", "ps-i00-f00", "test/algebra/1.json") == "ps-i00-f00"


def test_run1_defaults_are_unchanged():
    assert bf.TASKS == ("gsm8k", "bbh") and rf.RUN1_TASKS == ("gsm8k", "bbh")
    assert bf.DECODINGS["math"] == "config/decoding_forced_math.yaml"
    assert bf.CONTEXT_FILES["run1"] == "context_check.json"


def _saved():
    if not os.path.exists(DEMOS):
        pytest.skip("demonstrations not drawn")
    return json.load(open(DEMOS, encoding="utf-8"))


def test_saved_demonstrations_follow_the_rule():
    saved = _saved()
    try:
        again = bf.demo_manifest("math")
    except (FileNotFoundError, OSError, SystemExit):
        pytest.skip("raw data not present")
    assert again["demos"] == saved["demos"] and saved["demos_sha256"] == bf.list_sha(saved["demos"])
    s = json.load(open(os.path.join(ROOT, "manifests", "sampler_math.json"), encoding="utf-8"))
    assert saved["configurations"] == bf.configurations(s) and len(saved["configurations"]) == 31
    excl = set()
    for name in ("calibration", "screening"):
        excl |= set(pd.read_parquet(os.path.join(ROOT, "data", "processed", "math", f"items_{name}.parquet"))["item_id"])
    from src import math_task as mt
    for ds in saved["demos"].values():
        ids = [d["item_id"] for d in ds]
        assert len(set(ids)) == 5 and not set(ids) & excl and all(i.startswith("train/") for i in ids)
        for d in ds:
            assert "{" not in d["gold"] and "}" not in d["gold"]        # the box closes on its own brace
            assert mt.normalize_answer(d["gold"]) == d["gold"]           # the scorer's canonical form
    assert saved["pool"]["n_pool"] == saved["pool"]["n_numeric"] - saved["pool"]["n_excluded_sets"] \
        - saved["pool"]["n_same_text_as_target"]


def test_demonstration_answers_score_as_correct():
    """Each demonstration's answer turn, read by the MATH-500 strict scorer, gives its own gold."""
    saved = _saved()
    from src import math_task as mt
    for ds in list(saved["demos"].values())[:5]:
        for d in ds:
            assert mt.strict_extract(f"\\boxed{{{d['gold']}}}") == d["gold"]


def test_context_check_covers_the_panel_and_fits():
    p = os.path.join(ROOT, "results", "fewshot_context", "context_check_math.json")
    if not os.path.exists(p):
        pytest.skip("context check not run")
    res = json.load(open(p, encoding="utf-8"))
    panel = json.load(open(os.path.join(ROOT, "results", "dataset2", "math.json"), encoding="utf-8"))["models"]
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models_math.yaml"), encoding="utf-8"))["dataset2"]["models"]
    assert res["registry"] == "config/models_math.yaml" and set(res["models"]) == set(panel)
    for k, r in res["models"].items():
        assert r["max_model_len"] == reg[k]["max_model_len"]
        t = r["tasks"]["math"]
        assert t["fits"] and t["longest_prompt"] + bf.CAP <= r["max_model_len"] and t["n_checked"] == 93
    # the run-1 record is a separate file and still covers GSM8K and BIG-Bench Hard
    run1 = json.load(open(os.path.join(ROOT, "results", "fewshot_context", "context_check.json"), encoding="utf-8"))
    assert all("gsm8k" in r["tasks"] for r in run1["models"].values())


def test_built_requests_match_the_forced_arm():
    if not os.path.exists(REC):
        pytest.skip("requests not built")
    rec = json.load(open(REC, encoding="utf-8"))
    if "math" not in rec["tasks"]:
        pytest.skip("MATH-500 requests not built")
    panel = json.load(open(os.path.join(ROOT, "results", "dataset2", "math.json"), encoding="utf-8"))["models"]
    assert set(rec["tasks"]["math"]) == set(panel) and len(panel) == 29
    assert rec["per_task"]["math"]["decoding"] == "config/decoding_forced_math.yaml"
    assert rec["demos"]["math"]["sha256"] == bf.file_sha(DEMOS)
    assert rec["decoding"] == "config/decoding_forced.yaml"        # run 1's entry unchanged
    saved = _saved()
    bank = json.load(open(os.path.join(ROOT, "manifests", "math_bank.json"), encoding="utf-8"))
    for k in ("llama-2-7b-chat-hf", "qwen3-8b"):
        ent = rec["tasks"]["math"][k]
        p = os.path.join(ROOT, "data", "requests", "fewshot", "math", ent["requests_file"])
        assert bf.file_sha(p) == ent["requests_sha256"]
        fs = pd.read_parquet(p)
        fd = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced", "math", f"{k}.parquet"))
        assert len(fs) == len(fd) == ent["n"] == 12338 and ent["configs"] == 31
        assert (fs["request_id"] == fd["request_id"] + "|fs").all() and (fs["arm"] == "forced_direct").all()
        for (_, r), (_, f) in zip(fs.sample(150, random_state=1).iterrows(),
                                  fd.loc[fs.sample(150, random_state=1).index].iterrows()):
            m, t = json.loads(r["messages_json"]), json.loads(f["messages_json"])
            assert len(m) == 11 and m[-1] == t[0]
            frame = bank["configurations"][r["config_id"]]["arms"]["direct_answer"]["messages"]
            assert [x["content"] for x in m[1:10:2]] == [f"\\boxed{{{d['gold']}}}" for d in saved["demos"][r["config_id"]]]
            assert all(x["content"].startswith(frame[0]["content"].split("{question}")[0]) for x in m[0:10:2])


def test_scorer_points_math_at_the_fewshot_folders_and_restores():
    import score_forced_direct as sfd
    before = sfd.forced_dir
    assert before("math", "k").replace("\\", "/").endswith("data/d2_out/math/forced/k")
    with sf.pointed() as s:
        for t in ("gsm8k", "bbh", "math"):
            assert s.forced_dir(t, "k") == os.path.join(sf.OUT, t, "k")
    assert sfd.forced_dir is before


def test_build_record_feeds_the_forced_scoring_path(tmp_path, monkeypatch):
    if not os.path.exists(REC):
        pytest.skip("requests not built")
    rec = json.load(open(REC, encoding="utf-8"))
    if "math" not in rec["tasks"]:
        pytest.skip("MATH-500 requests not built")
    import shutil
    req = tmp_path / "req"
    (req / "math").mkdir(parents=True)
    shutil.copy(REC, req / "build_record.json")
    monkeypatch.setattr(sf, "REQ", str(req))
    with sf.pointed() as sfd:
        path, d = sfd.derived_record("math")
        assert len(d["models"]) == 29 and d["arms"] == ["forced_direct"]
        assert all(v["n_requests"] == 12338 for v in d["models"].values())


def test_pilot_verdict():
    want = {"forced_direct": "\\boxed{"}
    ok = {"status": "complete", "inputs": {"limit": 16}, "prompt_render_mismatches": 0,
          "decoding": {"prefill": want, "stop": []}}
    assert sf.pilot_verdict([ok], want, [])[0]
    assert sf.pilot_verdict([{**ok, "inputs": {"limit": None}}], want, [])[1] == "no complete 16-request run"
    assert not sf.pilot_verdict([{**ok, "decoding": {"prefill": want, "stop": ["}"]}}], want, [])[0]
    assert not sf.pilot_verdict([{**ok, "decoding": {"prefill": {}, "stop": []}}], want, [])[0]
    assert not sf.pilot_verdict([{**ok, "prompt_render_mismatches": 2}], want, [])[0]


def test_decoding_is_the_forced_math_arm():
    d = yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_forced_math.yaml"), encoding="utf-8"))
    assert d["prefill"]["forced_direct"] == "\\boxed{" and d["stop"] == [] and d["max_new_tokens"]["forced_direct"] == 16


def test_run_fewshot_math_alone_and_arrays_helper():
    with pytest.raises(SystemExit):
        rf.main(["--tasks", "math", "gsm8k"])
    with pytest.raises(ValueError):
        rf.fewshot_arrays("math", "test")
    assert rf.fewshot_available("math") == os.path.exists(os.path.join(rf.FS_PROCESSED, "math_scored.parquet"))
