"""GSM8K-30 inputs (src/d2_panels.py): one exclusion log per added checkpoint, every table through
the main-table guards, every model in exactly one table, and a loader that reads only the units asked for."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src import d2_io, d2_panels

ROOT = Path(__file__).resolve().parents[1]
SAMPLER = {"sha256": "s1"}
ITEMS = {"canonical_sha256": "i1"}


def _registries(tmp_path, main, extension):
    paths = []
    for name, keys in (("models.yaml", main), ("models_extension.yaml", extension)):
        p = tmp_path / name
        p.write_text(yaml.safe_dump({"dataset2": {"models": {k: {"family": f"F{i % 2}", "hf_id": f"org/{k}"}
                                                              for i, k in enumerate(keys)}}}), encoding="utf-8")
        paths.append(p)
    return paths


def _log(path, screened, excluded=()):
    log = {"models_screened": list(screened), "excluded_models": [{"model_key": k} for k in excluded],
           "retained_models": [k for k in screened if k not in excluded], "configurations_screened": [],
           "excluded_configurations": [], "retained_configuration_ids": [], "configuration_rules_evaluated": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log), encoding="utf-8")
    return path


def _table(path, models, scorer_sha, configs=("c1", "c2", "c3"), echo=False):
    inputs = [{"model_key": k, "work_dir": k, "test_backend": echo, "request_kind": "main", "split": "test",
               "incomplete_allowed": False, "decoding_confirmed": True, "sampler_registered_sha256": "s1"}
              for k in models]
    rows = [{"model_key": k, "arm": a, "config_id": c, "item_id": "0", "strict_correct": 1, "lenient_correct": 1,
             "compliant": True} for k in models for a in ("direct_answer", "zero_shot_cot") for c in configs]
    md = {"d2_kind": "main", "d2_split": "test", "d2_scorer_status": "registered",
          "d2_inputs": json.dumps(inputs), "d2_items": json.dumps(ITEMS),
          "d2_scorer_record": json.dumps({"manifest_sha256": scorer_sha}), "d2_task": "gsm8k"}
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False).replace_schema_metadata(md), path)
    return path


@pytest.fixture
def env(tmp_path):
    scorer = tmp_path / "scorer.json"
    scorer.write_text("{}", encoding="utf-8")
    sha = d2_io.sha256_path(scorer)
    regs = _registries(tmp_path, ["a", "b", "c"], ["x", "y"])
    conf_log = _log(tmp_path / "logs" / "gsm8k_admissibility.json", ["a", "b", "c"], excluded=["c"])
    ext_dir = tmp_path / "logs" / "ext"
    _log(ext_dir / "x.json", ["x"])
    _log(ext_dir / "y.json", ["y"])
    conf = _table(tmp_path / "scored.parquet", ["a", "b", "c"], sha)
    ext = [_table(tmp_path / "scored_x.parquet", ["x"], sha), _table(tmp_path / "scored_y.parquet", ["y"], sha)]
    return {"scorer": scorer, "sha": sha, "regs": regs, "conf_log": conf_log, "ext_dir": ext_dir, "conf": conf,
            "ext": ext, "tmp": tmp_path}


def _inputs(env, **kw):
    args = {"main_scored": env["conf"], "extension_scored": env["ext"], "sampler": SAMPLER,
            "items_info": ITEMS, "scorer_path": env["scorer"], "main_log": env["conf_log"],
            "extension_dir": env["ext_dir"], "registries": env["regs"]}
    return d2_panels.inputs(**{**args, **kw})


def test_panel_is_both_registries_less_the_excluded_in_registry_order(env):
    got = _inputs(env)
    assert got["panel"] == ["a", "b", "x", "y"] and got["excluded"] == ["c"]
    assert got["families"] == ["F0", "F1", "F0", "F1"]
    prov = got["provenance"]
    assert [t["models"] for t in prov["tables"]] == [["a", "b"], ["x"], ["y"]]
    assert len(prov["admissibility"]) == 3 and prov["excluded_models"] == ["c"]


def test_the_loader_reads_only_the_units_asked_for_and_drops_excluded_models(env):
    df = _inputs(env)["load"](["c1", "c3"])
    assert set(df["config_id"]) == {"c1", "c3"}
    assert set(df["model_key"]) == {"a", "b", "x", "y"}
    assert len(df) == 4 * 2 * 2


def test_an_added_model_without_its_own_log_is_refused(env):
    (env["ext_dir"] / "y.json").unlink()
    with pytest.raises(d2_panels.PanelError, match="y: never screened"):
        _inputs(env)
    _log(env["ext_dir"] / "y.json", ["x", "y"])
    with pytest.raises(d2_panels.PanelError, match="not y alone"):
        _inputs(env)


def test_an_excluded_added_model_leaves_the_panel(env):
    _log(env["ext_dir"] / "y.json", ["y"], excluded=["y"])
    got = _inputs(env, extension_scored=env["ext"][:1])
    assert got["panel"] == ["a", "b", "x"] and got["excluded"] == ["c", "y"]


def test_a_model_in_two_tables_or_in_none_is_refused(env):
    twice = _table(env["tmp"] / "scored_x2.parquet", ["x"], env["sha"])
    with pytest.raises(d2_panels.PanelError, match="x is in two tables"):
        _inputs(env, extension_scored=env["ext"] + [twice])
    with pytest.raises(d2_panels.PanelError, match=r"no scored table holds \['y'\]"):
        _inputs(env, extension_scored=env["ext"][:1])


def test_every_table_passes_the_main_guards(env):
    echo = _table(env["tmp"] / "scored_y.parquet", ["y"], env["sha"], echo=True)
    with pytest.raises(d2_panels.PanelError, match="test backend"):
        _inputs(env, extension_scored=[env["ext"][0], echo])
    with pytest.raises(d2_panels.PanelError, match="sampler"):
        _inputs(env, sampler={"sha256": "other"})
    with pytest.raises(d2_panels.PanelError, match="scorer manifest"):
        _inputs(env, scorer_path=env["regs"][0])


def test_the_extension_tables_are_found_by_name(tmp_path):
    for k in ("b", "a"):
        (tmp_path / f"scored_{k}.parquet").write_bytes(b"")
    (tmp_path / "screening_a.parquet").write_bytes(b"")
    assert [p.name for p in d2_panels.extension_tables(tmp_path)] == ["scored_a.parquet", "scored_b.parquet"]


def test_the_bbh_inputs_refuse_a_gsm8k_or_echo_table(env):
    with pytest.raises(d2_panels.PanelError, match="expected 'bbh'"):
        d2_panels.bbh_inputs(env["conf"], {"label": "bbh", "sha256": "s1"})
    echo = _table(env["tmp"] / "scored_bbh.parquet", ["a"], env["sha"], echo=True)
    md = d2_io.scored_metadata(echo)
    pq.write_table(pq.read_table(echo).replace_schema_metadata({**md, "d2_task": "bbh"}), echo)
    with pytest.raises(d2_panels.PanelError, match="test backend"):
        d2_panels.bbh_inputs(echo, {"label": "bbh", "sha256": "s1"})


def test_the_extension_panel_is_for_gsm8k_only():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_dataset2.py"), "--task", "bbh", "--panel",
                        "extension"], capture_output=True, text=True)
    assert r.returncode != 0 and "--panel extension is for --task gsm8k" in r.stderr
