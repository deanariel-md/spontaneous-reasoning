"""Five-shot forced-direct requests (demonstration rule, messages, request files, analysis helpers)."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_fewshot_requests as bf  # noqa: E402
import run_fewshot as rf  # noqa: E402
import score_fewshot as sf  # noqa: E402

FRAME = [{"role": "user", "content": "Solve.\n\nQuestion: {question}\nGive only the final answer, \\boxed{<number>}."}]


def test_seed_is_the_hash_prefix():
    want = int.from_bytes(hashlib.sha256(b"fewshot/gsm8k/ps-i00-f00").digest()[:8], "big")
    assert bf.demo_seed("gsm8k", "ps-i00-f00") == want
    want_b = int.from_bytes(hashlib.sha256(b"fewshot/bbh/snarks/ps-i00-f00").digest()[:8], "big")
    assert bf.demo_seed("bbh", "ps-i00-f00", "snarks") == want_b


def test_draw_is_fixed_without_replacement_in_draw_order():
    pool = list(range(100, 200))
    a = bf.draw_demos(pool, 7)
    assert a == bf.draw_demos(pool, 7) and len(set(a)) == 5 and set(a) <= set(pool)
    rng = np.random.default_rng(7)
    assert a == [pool[i] for i in rng.choice(100, size=5, replace=False)]
    with pytest.raises(ValueError):
        bf.draw_demos([1, 2, 3], 0)


def test_demo_turns_alternate_and_close_the_box():
    t = bf.demo_turns(FRAME, [("q1", "18"), ("q2", "(B)")])
    assert [m["role"] for m in t] == ["user", "assistant", "user", "assistant"]
    assert t[0]["content"] == FRAME[0]["content"].replace("{question}", "q1")
    assert t[1]["content"] == "\\boxed{18}" and t[3]["content"] == "\\boxed{(B)}"
    with pytest.raises(ValueError):
        bf.fill([{"role": "user", "content": "no slot"}], "q")
    with pytest.raises(ValueError):
        bf.demo_turns(FRAME * 2, [("q", "1")])


def test_configurations_are_the_distinct_draws():
    assert bf.configurations({"discovery_config_ids": ["b", "a", "b"], "validation_config_ids": ["c", "a"]}) == ["a", "b", "c"]


def test_fewshot_frame_keeps_the_forced_turn():
    forced = pd.DataFrame({"request_id": ["c1|direct_answer|0|fd", "c1|direct_answer|1|fd"], "model_key": "m",
                           "arm": "forced_direct", "config_id": "c1", "item_id": [0, 1],
                           "messages_json": [json.dumps([{"role": "user", "content": f"target {i}"}]) for i in (0, 1)]})
    pre = {"c1": bf.demo_turns(FRAME, [(f"q{i}", str(i)) for i in range(5)])}
    out = bf.fewshot_frame(forced, "gsm8k", pre)
    assert (out["request_id"] == forced["request_id"] + "|fs").all()
    for (_, r), (_, f) in zip(out.iterrows(), forced.iterrows()):
        m = json.loads(r["messages_json"])
        assert len(m) == 11 and m[-1] == json.loads(f["messages_json"])[0] and m[:10] == pre["c1"]
    assert bf.prefix_key("bbh", "c1", "snarks/12") == ("c1", "snarks")


@pytest.mark.parametrize("task", ["gsm8k", "bbh"])
def test_saved_demonstrations_follow_the_rule(task):
    p = os.path.join(ROOT, "manifests", f"fewshot_demos_{task}.json")
    if not os.path.exists(p):
        pytest.skip("demonstrations not drawn")
    saved = json.load(open(p, encoding="utf-8"))
    try:
        again = bf.demo_manifest(task)
    except (FileNotFoundError, OSError):
        pytest.skip("raw data not present")
    assert again["demos"] == saved["demos"] and saved["demos_sha256"] == bf.list_sha(saved["demos"])
    s = json.load(open(os.path.join(ROOT, "manifests", f"sampler_{task}.json"), encoding="utf-8"))
    assert saved["configurations"] == bf.configurations(s)
    if task == "gsm8k":
        excl = set()
        for n in ("gsm8k_calibration_items.json", "gsm8k_screening_items.json"):
            excl |= set(json.load(open(os.path.join(ROOT, "manifests", n), encoding="utf-8"))["item_ids"])
        for ds in saved["demos"].values():
            assert len(ds) == 5 and not {d["item_id"] for d in ds} & excl
    else:
        man = json.load(open(os.path.join(ROOT, "manifests", "bbh_items.json"), encoding="utf-8"))
        for by in saved["demos"].values():
            assert set(by) == set(man["tasks"])
            for t, ds in by.items():
                pt = man["per_task"][t]
                used = set(pt["target"]) | set(pt["calibration"]) | set(pt["screening"])
                idx = [int(d["item_id"].split("/")[1]) for d in ds]
                assert len(set(idx)) == 5 and not set(idx) & used


def test_built_requests_match_the_forced_arm():
    rec_p = os.path.join(ROOT, "data", "requests", "fewshot", "build_record.json")
    if not os.path.exists(rec_p):
        pytest.skip("requests not built")
    rec = json.load(open(rec_p, encoding="utf-8"))
    for task, k in (("gsm8k", "qwen3-8b"), ("bbh", "llama-2-7b-chat-hf")):
        ent = rec["tasks"][task][k]
        fs = pd.read_parquet(os.path.join(ROOT, "data", "requests", "fewshot", task, ent["requests_file"]))
        fd = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced", task, f"{k}.parquet"))
        assert len(fs) == len(fd) == ent["n"]
        assert (fs["request_id"] == fd["request_id"] + "|fs").all() and (fs["arm"] == "forced_direct").all()
        sub = fs.sample(200, random_state=0)
        for (_, r) in sub.iterrows():
            m = json.loads(r["messages_json"])
            f = json.loads(fd.loc[fd["request_id"] == r["request_id"][:-3], "messages_json"].iloc[0])
            assert len(m) == 11 and m[-1] == f[0]
            assert all(x["role"] == ("user" if i % 2 == 0 else "assistant") for i, x in enumerate(m))


def test_pair_arrays_and_rows():
    from src import dataset2 as d2
    rows = []
    for arm, shift in (("fd", 0), ("fs", 1)):
        for u in ("u1", "u2"):
            for k in ("a", "b"):
                for i in range(4):
                    rows.append({"model_key": k, "arm": "forced_direct", "config_id": u, "item_id": i, "src": arm,
                                 "strict_correct": int((i + shift + (k == "b")) % 2 == 0)})
    df = pd.DataFrame(rows)
    for c in d2.VALUES:
        if c not in df:
            df[c] = df["strict_correct"]
    fd = lambda ids: df[(df["src"] == "fd") & df["config_id"].isin(ids)].drop(columns="src")  # noqa: E731
    fs = lambda ids: df[(df["src"] == "fs") & df["config_id"].isin(ids)].drop(columns="src")  # noqa: E731
    B, P = rf.pair_arrays(fd, fs, ["u1", "u2"], ["a", "b"], list(range(4)))
    assert B.shape == P.shape == (2, 2)
    assert np.allclose(B, 50.0) and np.allclose(P, 50.0)
    r = rf.per_model_rows(["a", "b"], B, P)
    assert r[0]["gain"] == 0.0 and r[1]["model_key"] == "b"


def test_scorer_points_at_the_fewshot_folders_and_restores():
    import score_forced_direct as sfd
    before = (sfd.REQ, sfd.FORCED_OUT, sfd.PROCESSED, sfd.RESULTS)
    with sf.pointed() as s:
        assert s.REQ == sf.REQ and s.FORCED_OUT == sf.OUT and s.PROCESSED == sf.PROCESSED and s.RESULTS == sf.RESULTS
    assert (sfd.REQ, sfd.FORCED_OUT, sfd.PROCESSED, sfd.RESULTS) == before


def test_build_record_feeds_the_forced_scoring_path(tmp_path, monkeypatch):
    """tasks[task] holds model keys only, so score_forced_direct.derived_record reads the five-shot record."""
    rec_p = os.path.join(ROOT, "data", "requests", "fewshot", "build_record.json")
    if not os.path.exists(rec_p):
        pytest.skip("requests not built")
    rec = json.load(open(rec_p, encoding="utf-8"))
    assert {"gsm8k", "bbh"} <= set(rec["demos"]) <= {"gsm8k", "bbh", "math"}     # math: the MATH-500 arm
    for task, n in (("gsm8k", 29), ("bbh", 28)):
        assert len(rec["tasks"][task]) == n and all("source" in e for e in rec["tasks"][task].values())
    req = tmp_path / "req"
    for task in ("gsm8k", "bbh"):
        (req / task).mkdir(parents=True)
    import shutil
    shutil.copy(rec_p, req / "build_record.json")
    monkeypatch.setattr(sf, "REQ", str(req))
    with sf.pointed() as sfd:
        for task, n in (("gsm8k", 29), ("bbh", 28)):
            path, d = sfd.derived_record(task)
            assert len(d["models"]) == n and d["arms"] == ["forced_direct"]
        assert len(sfd.forced_record()["tasks"]["gsm8k"]) == 29     # the pilot check's model list
