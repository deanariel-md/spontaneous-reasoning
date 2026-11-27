"""The cap check: sample rule, loop rule, intervals and the text the scorer reads."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_capcheck_requests as bc  # noqa: E402
import run_capcheck as rc  # noqa: E402
import capcheck_generate as cg  # noqa: E402


def test_draw_is_fixed_sorted_and_a_subset():
    ids = [f"r{i:05d}" for i in range(5000)]
    a = bc.draw(list(reversed(ids)), "gsm8k", "qwen3.6-27b", 300)
    b = bc.draw(ids, "gsm8k", "qwen3.6-27b", 300)
    assert a == b                      # input order does not matter: the ids are sorted first
    assert len(a) == 300 and len(set(a)) == 300 and set(a) <= set(ids) and a == sorted(a)
    assert bc.draw(ids, "math", "qwen3.6-27b", 300) != a      # each task and checkpoint has its own seed
    assert bc.draw(ids[:10], "gsm8k", "qwen3.6-27b", 300) == ids[:10]


def test_seed_is_the_hash_prefix():
    import hashlib
    want = int.from_bytes(hashlib.sha256(b"d18/math/qwen3.5-35b-a3b").digest()[:8], "big")
    assert bc.seed_for("math", "qwen3.5-35b-a3b") == want


def test_cut_open_needs_the_cap_and_no_marker():
    out = pd.DataFrame({"output_text": ["a</think>\n\n5", "thinking on", "thinking on", "x</think>"],
                        "finish_reason": ["length", "length", "stop", "stop"]})
    assert bc.cut_open(out).tolist() == [False, True, False, False]


def test_looping():
    loop = "\n".join(["start", "more"] + ["I'll write:", "500", "\\boxed{500}", ""] * 60)
    assert rc.looping(loop)
    varied = "\n".join(f"step {i}: add {i} and {i + 1}" for i in range(300))
    assert not rc.looping(varied)
    assert not rc.looping("\n".join(["same"] * 50))           # fewer than 100 non-empty lines
    nine = "\n".join([f"line {i % 10}" for i in range(100)])  # 10 distinct lines: 90 repeats of 100
    assert rc.looping(nine)
    eleven = "\n".join([f"line {i % 11}" for i in range(100)])
    assert not rc.looping(eleven)


def test_clopper_pearson_known_values():
    lo, hi = rc.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.30850, abs=1e-4)
    lo, hi = rc.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.69150, abs=1e-4)
    lo, hi = rc.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4) and hi == pytest.approx(0.81291, abs=1e-4)
    with pytest.raises(ValueError):
        rc.clopper_pearson(0, 0)


def test_estimate_adds_the_recovered_share():
    # 74.5 pp at the cap, 20 pp of outputs cut before the marker, 40% of them right with twice the budget
    assert rc.estimate(74.5, 20.0, 0.4) == pytest.approx(82.5)
    assert rc.estimate(74.5, 20.0, 0.0) == 74.5


def test_bodies_read_after_the_marker_of_the_joined_text():
    b, closed = rc.bodies(["long thinking", "long thinking", "long thinking"],
                          [" more</think>\n\nThe answer is \\boxed{5}.", " still going", " gave up"],
                          ["stop", "length", "stop"])
    assert closed.tolist() == [True, False, False]
    assert b[0] == "\n\nThe answer is \\boxed{5}."
    assert b[1] == ""                                     # cut again before the marker: no answer
    assert b[2] == "long thinking gave up"               # stopped without a marker: scored whole


def test_parts_cover_every_row_once():
    req = pd.DataFrame({"task": ["gsm8k"] * 7 + ["math"] * 6, "request_id": [f"r{i:02d}" for i in range(13)]})
    parts = [cg.select_part(req, f"{i}/4") for i in range(4)]
    got = pd.concat(parts)["request_id"].tolist()
    assert sorted(got) == req["request_id"].tolist() and len(got) == len(set(got))
    assert parts[1]["request_id"].tolist() == ["r01", "r05", "r09"]
    assert cg.select_part(req, None) is req
    with pytest.raises(SystemExit):
        cg.select_part(req, "4/4")


def test_continuations_of_a_model_with_none_yet_keep_the_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "CONT_DIR", str(tmp_path))
    df = rc.continuations("no-such-model")
    assert df.empty and {"task", "request_id", "model_key", "continuation_text"} <= set(df.columns)
    assert df.drop(columns=["model_key"]).empty


def test_prompt_boxes_reads_the_messages():
    import json
    msgs = json.dumps([{"role": "user", "content": "Q?\nEnd your reply with a final line that contains \\boxed{<number>}."}])
    assert rc.prompt_boxes(msgs) == {"<number>"}
    assert rc.prompt_boxes(json.dumps([{"role": "user", "content": "no box"}])) == set()


def test_reached_answer_is_the_last_box_that_is_not_the_placeholder():
    skip = {"<number>"}
    t = "So 7.\n\\boxed{7}\nWait.\nThe prompt says \\boxed{<number>}.\nSo \\boxed{\\frac{1}{2}}\nThe prompt says \\boxed{<number>}"
    assert rc.reached_answer(t, skip) == "\\frac{1}{2}"
    assert rc.reached_answer("just thinking \\boxed{<number>}", skip) is None
    assert rc.reached_answer("empty \\boxed{ } then nothing", skip) is None
    assert rc.reached_answer("an open box at the cut \\boxed{4", skip) is None
    assert rc.reached_answer("\\boxed{3} and an open one \\boxed{4", skip) == "3"
