"""scripts/freeze_bank_d2.py and src/d2_io.frozen_bank: candidate bank + exclusion log -> frozen bank."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src import d2_io

ROOT = Path(__file__).resolve().parent.parent


def load_freezer():
    spec = importlib.util.spec_from_file_location("freeze_bank_d2", ROOT / "scripts" / "freeze_bank_d2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate(n=6, status="confirmed", law="uniform", generator="PromptSuite"):
    arm = lambda clause: {"messages": [{"role": "system", "content": f"Solve. {clause}"},
                                       {"role": "user", "content": "Question: {question}\nResponse:"}]}
    ids = [f"ps-{i:03d}" for i in range(n)]
    return {"generator": generator, "generator_version": "3.0.7", "seed": 1,
            "sampling_probabilities": {c: 1 / n for c in ids},
            "configurations": {c: {"dimensions": {"v": i}, "arms": {"direct_answer": arm(f"Answer only {i}."),
                                                                    "zero_shot_cot": arm(f"Think {i}.")}}
                               for i, c in enumerate(ids)},
            "generation_record": {"config_status": status, "sampling_law": law}}


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_for(ids, sha, excluded=(), screened=None, evaluated=True):
    screened = list(ids) if screened is None else screened
    return {"inputs": {"bank_sha256": sha}, "models_screened": ["m1"], "excluded_models": [],
            "retained_models": ["m1"], "configuration_rules_evaluated": evaluated,
            "configurations_screened": screened,
            "excluded_configurations": [{"config_id": c, "violations": [{"rule": "max_truncation_pct"}]}
                                        for c in excluded],
            "retained_configuration_ids": [c for c in screened if c not in excluded]}


@pytest.fixture
def setup(tmp_path):
    cand = candidate()
    cp = tmp_path / "cand.json"
    sha = write(cp, cand)
    ids = list(cand["configurations"])
    lp = tmp_path / "adm.json"
    write(lp, log_for(ids, sha, excluded=[ids[1], ids[4]]))
    return {"cand": cand, "cp": cp, "sha": sha, "ids": ids, "lp": lp, "out": tmp_path / "bank.json", "tmp": tmp_path}


def test_frozen_bank_keeps_exactly_the_retained_configurations(setup):
    fb = load_freezer()
    res = fb.freeze(setup["cp"], setup["out"], setup["lp"])
    bank, _ = d2_io.load_bank(setup["out"])
    ids = setup["ids"]
    keep = [c for c in ids if c not in (ids[1], ids[4])]
    assert list(bank["configurations"]) == keep                         # candidate order, nothing else
    assert bank["sampling_probabilities"] == {c: 0.25 for c in keep}
    assert bank["configurations"][keep[0]] == setup["cand"]["configurations"][keep[0]]
    ff = bank["frozen_from"]
    assert ff["candidate_sha256"] == setup["sha"] and ff["excluded_configuration_ids"] == [ids[1], ids[4]]
    assert ff["admissibility"]["sha256"] == hashlib.sha256(setup["lp"].read_bytes()).hexdigest()
    assert res["n_retained"] == 4 and res["sampling_law"] == "uniform"
    # the frozen bank satisfies the sampler freeze and the main-panel build checks
    log, _ = d2_io.load_admissibility(setup["lp"])
    assert d2_io.bank_admissibility_problems(bank, log) == []
    with pytest.raises(d2_io.BankError, match="never overwritten"):
        fb.freeze(setup["cp"], setup["out"], setup["lp"])


@pytest.mark.parametrize("change, words", [
    (lambda c, l: c.update(generator="PromptSuite (EXAMPLE, not a registered bank)"), "example bank"),
    (lambda c, l: c["generation_record"].update(config_status="PENDING_REVIEW"), "not 'confirmed'"),
    (lambda c, l: c["generation_record"].update(sampling_law="weighted"), "sampling law"),
    (lambda c, l: l["inputs"].update(bank_sha256="0" * 64), "screened the bank"),
    (lambda c, l: l.update(configurations_screened=l["configurations_screened"][:3]), "whole candidate"),
    (lambda c, l: l.update(configuration_rules_evaluated=False), "no configuration rule"),
])
def test_freeze_refusals(setup, change, words):
    cand, log = copy.deepcopy(setup["cand"]), log_for(setup["ids"], None)
    change(cand, log)
    cp = setup["tmp"] / "c2.json"
    sha = write(cp, cand)
    if log["inputs"]["bank_sha256"] is None:
        log["inputs"]["bank_sha256"] = sha
    lp = setup["tmp"] / "l2.json"
    write(lp, log)
    with pytest.raises(d2_io.BankError, match=words):
        load_freezer().freeze(cp, setup["out"], lp)
    assert not setup["out"].exists()


def test_cli_refuses_without_a_log_and_writes_nothing(setup, capsys):
    rc = load_freezer().main(["--candidate", str(setup["cp"]), "--admissibility", str(setup["tmp"] / "none.json"),
                              "--out", str(setup["out"])])
    assert rc == 2 and "does not exist" in capsys.readouterr().err and not setup["out"].exists()
    rc = load_freezer().main(["--candidate", str(setup["cp"]), "--admissibility", str(setup["lp"]),
                              "--out", str(setup["out"])])
    assert rc == 0 and json.loads(capsys.readouterr().out)["n_retained"] == 4
