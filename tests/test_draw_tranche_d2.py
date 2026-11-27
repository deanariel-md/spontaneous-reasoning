"""scripts/draw_tranche_d2.py: the tranche's configurations drawn by seed from the frozen bank."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src import d2_io

ROOT = Path(__file__).resolve().parent.parent


def load_drawer():
    spec = importlib.util.spec_from_file_location("draw_tranche_d2", ROOT / "scripts" / "draw_tranche_d2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate(n=12):
    arm = lambda clause: {"messages": [{"role": "system", "content": f"Solve. {clause}"},
                                       {"role": "user", "content": "Question: {question}\nResponse:"}]}
    ids = [f"ps-{i:03d}" for i in range(n)]
    return {"generator": "PromptSuite", "generator_version": "3.0.7", "seed": 1,
            "sampling_probabilities": {c: 1 / n for c in ids},
            "configurations": {c: {"dimensions": {"v": i}, "arms": {"direct_answer": arm(f"Answer only {i}."),
                                                                    "zero_shot_cot": arm(f"Think {i}.")}}
                               for i, c in enumerate(ids)},
            "generation_record": {"config_status": "confirmed", "sampling_law": "uniform"}}


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_for(ids, sha, excluded=()):
    return {"inputs": {"bank_sha256": sha}, "models_screened": ["m1", "m2", "m3"],
            "excluded_models": [{"model_key": "m3"}], "retained_models": ["m1", "m2"],
            "configuration_rules_evaluated": True, "configurations_screened": list(ids),
            "excluded_configurations": [{"config_id": c, "violations": [{"rule": "max_truncation_pct"}]}
                                        for c in excluded],
            "retained_configuration_ids": [c for c in ids if c not in excluded]}


@pytest.fixture
def setup(tmp_path):
    cand = candidate()
    cp = tmp_path / "cand.json"
    sha = write(cp, cand)
    ids = list(cand["configurations"])
    excluded = [ids[2], ids[7]]
    lp = tmp_path / "adm.json"
    write(lp, log_for(ids, sha, excluded))
    log, info = d2_io.load_admissibility(lp)
    bp = tmp_path / "bank.json"
    write(bp, d2_io.frozen_bank(cand, sha, log, candidate_path="cand.json", log_info=info))
    tp = tmp_path / "tranche.yaml"
    tp.write_text(yaml.safe_dump({"status": "confirmed", "configurations": 4, "seed": 5, "calibration_items": 50}),
                  encoding="utf-8")
    return {"ids": ids, "excluded": excluded, "lp": lp, "bp": bp, "tp": tp, "out": tmp_path / "tranche.json",
            "tmp": tmp_path}


def test_the_repository_rule_is_confirmed():
    t = load_drawer().load_tranche()
    assert (t["configurations"], t["calibration_items"]) == (8, 1319)


def test_configurations_are_drawn_from_ids_only_reproducibly_and_without_replacement():
    dt = load_drawer()
    ids = [f"c{i:02d}" for i in range(56)]
    a = dt.draw_configurations(ids, 8, 20261030)
    assert a == dt.draw_configurations(list(reversed(ids)), 8, 20261030)
    assert len(set(a)) == 8 and a == sorted(a) and set(a) <= set(ids)
    assert a != dt.draw_configurations(ids, 8, 1)
    with pytest.raises(dt.Refused, match="cannot draw"):
        dt.draw_configurations(ids[:3], 4, 0)


def test_the_draw_comes_from_the_retained_bank_and_is_written_once(setup):
    dt = load_drawer()
    res = dt.draw(setup["tp"], setup["bp"], setup["lp"], setup["out"])
    rec = json.loads(setup["out"].read_text(encoding="utf-8"))
    retained = [c for c in setup["ids"] if c not in setup["excluded"]]
    assert rec["configurations"] == res["configurations"] == dt.draw_configurations(retained, 4, 5)
    assert not set(rec["configurations"]) & set(setup["excluded"])
    assert rec["models"] == ["m1", "m2"] and rec["n_bank_configurations"] == 10
    assert rec["bank"]["sha256"] == hashlib.sha256(setup["bp"].read_bytes()).hexdigest()
    assert rec["admissibility"]["sha256"] == hashlib.sha256(setup["lp"].read_bytes()).hexdigest()
    assert rec["excluded_from_main"] is True
    assert "--tranche --split train --calibration-items 50" in res["build_requests"]
    assert " ".join(res["configurations"]) in res["build_requests"]
    dt.draw(setup["tp"], setup["bp"], setup["lp"], setup["out"])            # identical: left alone
    setup["tp"].write_text(yaml.safe_dump({"status": "confirmed", "configurations": 4, "seed": 6,
                                           "calibration_items": 50}), encoding="utf-8")
    with pytest.raises(dt.Refused, match="not overwritten"):
        dt.draw(setup["tp"], setup["bp"], setup["lp"], setup["out"])


def test_a_bank_frozen_from_another_log_is_refused(setup):
    dt = load_drawer()
    log = json.loads(setup["lp"].read_text(encoding="utf-8"))
    log["note"] = "another screening"
    other = setup["tmp"] / "other.json"
    write(other, log)
    with pytest.raises(dt.Refused, match="was not frozen from"):
        dt.draw(setup["tp"], setup["bp"], other, setup["out"])
    assert not setup["out"].exists()


@pytest.mark.parametrize("edit, words", [
    ({"status": "pending"}, "not confirmed"),
    ({"configurations": 1}, "at least two"),
    ({"configurations": 11}, "cannot draw 11"),
    ({"seed": None}, "seed must be an integer"),
])
def test_an_unusable_rule_is_refused(setup, edit, words):
    dt = load_drawer()
    t = {"status": "confirmed", "configurations": 4, "seed": 5, "calibration_items": 50, **edit}
    setup["tp"].write_text(yaml.safe_dump(t), encoding="utf-8")
    with pytest.raises(dt.Refused, match=words):
        dt.draw(setup["tp"], setup["bp"], setup["lp"], setup["out"])
    assert not setup["out"].exists()


def test_main_reports_a_missing_bank_as_refused(tmp_path, capsys):
    dt = load_drawer()
    assert dt.main(["--bank", str(tmp_path / "none.json"), "--out", str(tmp_path / "t.json")]) == dt.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
