"""The answer-format screening rule (scripts/select_answer_format_d2.py)."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from src import d2_io, screening

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("select_answer_format_d2", ROOT / "scripts" / "select_answer_format_d2.py")
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


def test_the_repository_candidates_are_valid_and_share_one_frame():
    f = sf.load_formats()
    names = [c["scorer_candidate"] for c in f["candidates"]]
    assert names == ["answer_colon_markup", "the_answer_is_markup", "hash_n", "boxed"]
    assert all(c["text"].startswith("End your reply with a final line") for c in f["candidates"])
    assert f["screening"] == {"configurations": 2, "seed": 20261030}


def test_configurations_are_drawn_from_ids_only_and_reproducibly():
    ids = [f"ps-i{i:02d}-f{j:02d}" for i in range(8) for j in range(8)]
    a = sf.draw_configurations(ids, 2, 20261030)
    assert a == sf.draw_configurations(list(reversed(ids)), 2, 20261030) and len(set(a)) == 2 and a == sorted(a)
    assert a != sf.draw_configurations(ids, 2, 1)
    with pytest.raises(sf.Refused):
        sf.draw_configurations(ids[:1], 2, 0)


def test_ties_go_to_the_earlier_candidate():
    assert sf.choose([("a", 80.0), ("b", 91.5), ("c", 91.5)]) == ("b", True)
    assert sf.choose([("a", 80.0), ("b", 70.0)]) == ("a", False)


def test_the_metrics_only_rules_are_pending_and_valid():
    r = sf.metrics_only_rules()
    assert r["status"] == screening.STATUS_PENDING and screening.validate_rules(r) == []


def _report(bank_sha, values, configs=("c1", "c2"), models=("m1", "m2"), **inp):
    by_model = [{"model_key": m, "arm": a, "n": 400, "strict_compliance_pct": {n: v[(m, a)] for n, v in values.items()}}
                for m in models for a in ("direct_answer", "zero_shot_cot")]
    return {"kind": "admissibility_screening", "rules": {"applied": False},
            "inputs": {"split": "train", "test_backend": False, "bank_sha256": bank_sha, "models": list(models),
                       "arms": ["direct_answer", "zero_shot_cot"], "item_set_identical_across_cells": True, **inp},
            "by_model": by_model, "by_configuration": [{"config_id": c, "arm": a} for c in configs
                                                       for a in ("direct_answer", "zero_shot_cot")]}


def test_select_registers_the_highest_minimum_and_records_every_value(tmp_path, monkeypatch):
    for name in ("SCREEN_MANIFEST", "SCORER", "SELECTION"):
        monkeypatch.setattr(sf, name, tmp_path / f"{name.lower()}.json")
    monkeypatch.setattr(sf, "ROOT", tmp_path)
    cands = [{"scorer_candidate": n, "text": f"text {n}", "bank_sha256": n * 3} for n in ("answer_colon_markup", "hash_n")]
    (tmp_path / "screen_manifest.json").write_text(json.dumps({"candidates": cands, "configurations": ["c1", "c2"]}))
    cells = [(m, a) for m in ("m1", "m2") for a in ("direct_answer", "zero_shot_cot")]
    # answer_colon: one weak cell (60); hash_n: uniformly 85, so hash_n has the higher minimum
    va = {"answer_colon_markup": {c: (60.0 if c == ("m2", "zero_shot_cot") else 99.0) for c in cells}}
    vh = {"hash_n": {c: 85.0 for c in cells}}
    pa, ph = tmp_path / "a.json", tmp_path / "h.json"
    pa.write_text(json.dumps(_report("answer_colon_markup" * 3, va)))
    ph.write_text(json.dumps(_report("hash_n" * 3, vh)))
    res = sf.cmd_select(type("A", (), {"report": [f"answer_colon_markup={pa}", f"hash_n={ph}"]})())
    assert res == {"registered": "hash_n", "tie": False, "min_compliance": {"answer_colon_markup": 60.0, "hash_n": 85.0}}
    spec_, record = d2_io.load_registered_scorer(tmp_path / "scorer.json")
    assert spec_.name == "hash_n" and record["status"] == d2_io.REGISTERED_STATUS
    assert spec_.description.startswith("Registered by the format screening rule")
    sel = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert sel["registered"] == "hash_n" and len(sel["candidates"][0]["by_model_and_arm"]) == 4
    # a second selection from other reports cannot replace the selected scorer
    ph.write_text(json.dumps(_report("hash_n" * 3, {"hash_n": {c: 50.0 for c in cells}})))
    with pytest.raises(sf.Refused, match="not overwritten"):
        sf.cmd_select(type("A", (), {"report": [f"answer_colon_markup={pa}", f"hash_n={ph}"]})())


@pytest.mark.parametrize("edit,needle", [
    (lambda r: r["rules"].update(applied=True), "metrics only"),
    (lambda r: r["inputs"].update(bank_sha256="x"), "another bank"),
    (lambda r: r.update(by_configuration=[{"config_id": "c9", "arm": "direct_answer"}]), "draws"),
    (lambda r: r["inputs"].update(split="test"), "split"),
    (lambda r: r["inputs"].update(item_set_identical_across_cells=False), "item set"),
])
def test_a_report_that_does_not_match_the_screen_is_refused(edit, needle):
    cells = [(m, a) for m in ("m1", "m2") for a in ("direct_answer", "zero_shot_cot")]
    r = _report("s", {"hash_n": {c: 90.0 for c in cells}})
    edit(r)
    problems = sf.check_report(r, {"bank_sha256": "s"}, {"configurations": ["c1", "c2"]})
    assert any(needle in p for p in problems), problems
