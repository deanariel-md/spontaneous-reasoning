"""The constrained forced-direct arm on BIG-Bench Hard (constraints, request files, pilot check, analysis
helpers)."""
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

import build_constrained_forced as bc  # noqa: E402
import run_forced_constrained as rc  # noqa: E402
import score_forced_constrained as sc  # noqa: E402
from src import bbh  # noqa: E402
from src import generation as g  # noqa: E402

ITEMS = os.path.join(ROOT, "data", "processed", "bbh", "items_target.parquet")
REC = os.path.join(ROOT, "data", "requests", "forced_constrained", "build_record.json")


def test_option_letters():
    q = "Which?\nOptions:\n(A) red\n(B) blue\n(C) green"
    assert bc.option_letters(q) == ["A", "B", "C"]
    for bad in ("Options:\n(A) x", "Options:\n(A) x\n(C) y", "no options"):
        with pytest.raises(ValueError):
            bc.option_letters(bad)


def test_constraints_by_answer_type():
    assert bc.constraint("letter", "Options:\n(A) x\n(B) y") == {"choice": ["(A)}", "A}", "(B)}", "B}"]}
    assert bc.constraint("words:True/False", "q") == {"choice": ["True}", "true}", "TRUE}", "False}", "false}", "FALSE}"]}
    assert bc.constraint("words:yes/no", "q") == {"choice": ["yes}", "Yes}", "YES}", "no}", "No}", "NO}"]}
    assert bc.constraint("integer", "q") == {"regex": bc.INTEGER_REGEX}


def test_allowed():
    assert sc.allowed("(A)}", {"choice": ["(A)}", "A}"]}) and not sc.allowed("(A)", {"choice": ["(A)}", "A}"]})
    r = {"regex": bc.INTEGER_REGEX}
    assert sc.allowed("15}", r) and sc.allowed("-1584}", r)
    assert not sc.allowed("1.5}", r) and not sc.allowed("12345678}", r) and not sc.allowed("15", r)


@pytest.mark.skipif(not os.path.exists(ITEMS), reason="BIG-Bench Hard item table not built")
def test_every_allowed_string_is_read_by_the_registered_scorer():
    items = pd.read_parquet(ITEMS)
    for at, q, gold in zip(items["answer_type"], items["question"], items["gold_answer"]):
        c = bc.constraint(at, q)
        t = bbh.AnswerType.from_label(at)
        strings = c["choice"] if "choice" in c else ["0}", "15}", "-1584}", "9999999}"]
        for s in strings:
            a = bbh.strict_extract("\\boxed{" + s, t)
            assert a is not None, (at, s)
        if "choice" in c:          # the gold answer is among the allowed answers
            assert gold in {bbh.strict_extract("\\boxed{" + s, t) for s in c["choice"]}


@pytest.mark.skipif(not os.path.exists(REC), reason="constrained requests not built")
def test_built_files_keep_the_forced_messages():
    rec = json.load(open(REC, encoding="utf-8"))
    assert len(rec["tasks"]["bbh"]) == 28
    for k in ("qwen3-8b", "gemma-2b-it"):
        ent = rec["tasks"]["bbh"][k]
        gc = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced_constrained", "bbh", ent["requests_file"]))
        fd = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced", "bbh", f"{k}.parquet"))
        assert len(gc) == len(fd) == ent["n"] == 47250
        assert (gc["request_id"] == fd["request_id"] + "|gc").all()
        assert (gc["messages_json"] == fd["messages_json"]).all() and (gc["arm"] == "forced_direct").all()
        assert gc["guided_json"].map(lambda s: bool(g.parse_guided(s))).all()
        pilot = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced_constrained", ent["pilot_file"]))
        assert len(pilot) == 25 and pilot["config_id"].nunique() == 1
        assert pilot["item_id"].str.split("/").str[0].nunique() == 25


@pytest.mark.skipif(not os.path.exists(REC), reason="constrained requests not built")
def test_pilot_check_passes_an_echo_run(tmp_path, monkeypatch):
    from tests.test_generation import Cfg, KEY, decoding_dict
    monkeypatch.setattr(g, "gpu_info", lambda timeout=10.0: {
        "count": 1, "devices": [{"name": "Fake A100", "memory_mib": 81920}], "driver_version": "580.1",
        "cuda_version_driver": "13.0", "nvidia_smi": True})
    dec = decoding_dict(max_new_tokens={"forced_direct": 16}, prefill={"forced_direct": "\\boxed{"}, stop=["}"],
                        include_stop_str_in_output=True, guided={"forced_direct": "request"})
    req = pd.read_parquet(os.path.join(ROOT, "data", "requests", "forced_constrained", "pilot", "qwen3-8b.parquet"))
    req["model_key"] = KEY
    req.to_parquet(tmp_path / "pilot.parquet")
    c = Cfg(tmp_path, decoding=dec)
    assert c.run(tmp_path / "pilot.parquet", tmp_path / "out")["status"] == "complete"
    r = sc.check_pilot_folder(str(tmp_path / "out"), req, None)
    assert r["passes"], r["status"]
    assert r["n_outputs"] == r["allowed"] == r["readable"] == 25
    bad = req.copy()
    bad["guided_json"] = json.dumps({"choice": ["X}", "Y}"]})
    assert not sc.check_pilot_folder(str(tmp_path / "out"), bad, None)["passes"]


def test_readable_models_keeps_panel_order():
    rows = [{"model_key": "b", "closed_and_readable_pct": 99.0}, {"model_key": "a", "closed_and_readable_pct": 95.0},
            {"model_key": "c", "closed_and_readable_pct": 94.9}, {"model_key": "d"}]
    assert rc.readable_models(["a", "b", "c", "d"], rows) == ["a", "b"]


def test_per_model_table_and_slope_blocks():
    rng = np.random.default_rng(0)
    K, M = 24, 8
    base = rng.uniform(20, 80, M)
    D = {a: base + rng.normal(0, 1, (K, M)) + s for a, s in (("requested", 0), ("forced", -3), ("constrained", 2))}
    D["cot"] = 60 + 0.3 * base + rng.normal(0, 1, (K, M))
    panel = [f"m{j}" for j in range(M)]
    rows = rc.per_model_table(panel, D, [{"model_key": "m0", "closed_and_readable_pct": 50.0}],
                              [{"model_key": k, "closed_and_readable_pct": 100.0} for k in panel])
    assert rows[0]["constrained_minus_forced"] == pytest.approx(D["constrained"][:, 0].mean() - D["forced"][:, 0].mean())
    assert rows[0]["forced_closed_and_readable_pct"] == 50.0 and rows[1]["forced_closed_and_readable_pct"] is None
    fams = ["F1", "F1", "F2", "F2", "F3", "F3", "F4", "F4"]
    panels = {"requested": {"all": panel}, "forced": {"all": panel, "readable": panel[:2]},
              "constrained": {"all": panel}}
    from src import dataset2 as d2
    from run_outcomes import SELECTION
    S = rc.slope_blocks(D, panel, fams, d2.load_selection(SELECTION)["H"], 0.05, panels)
    assert S["cot_minus_forced"]["readable"]["note"] == "fewer than three models"
    b = S["cot_minus_constrained"]["all"]
    assert b["M"] == M and b["slope"] < 0                    # gain falls with baseline accuracy here
    assert set(b["leave_one_family_out"]) == {"F1", "F2", "F3", "F4"}
