"""The MATH-500 entry of scripts/run_forced_direct.py, on simulated data: three
arms, the frozen map and rule files named after the task, and the GSM8K and BIG-Bench Hard defaults unchanged."""
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import d2_panels, dataset2 as d2, intervals as iv, sampling
from src.simulate import GSM8KSim

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_math_fd", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rfd = load_script("run_forced_direct")


def test_task_options_and_default_names(monkeypatch):
    seen = {}
    monkeypatch.setattr(rfd, "run", lambda a: seen.setdefault("a", a))
    rfd.main([])
    a = seen.pop("a")
    assert a.tasks == ["gsm8k", "bbh"] and a.suffix == "" and a.out == rfd.OUT and a.pred_dir == rfd.MANIFESTS
    rfd.main(["--tasks", "math"])
    a = seen.pop("a")
    assert a.suffix == "_math" and a.out == os.path.join(rfd.OUT, "math") and a.pred_dir == rfd.MANIFESTS
    assert rfd.TASKS["math"]["collection"] == "math" and rfd.TASKS["math"]["tag"].isalpha()


def test_math_inputs_refuse_a_bbh_or_echo_table(tmp_path):
    p = tmp_path / "scored.parquet"
    t = pa.table({"model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"], "item_id": ["test/algebra/1.json"],
                  "strict_correct": [1], "lenient_correct": [1], "compliant": [True]})
    inputs = [{"model_key": "m", "test_backend": True, "request_kind": "main", "split": "test",
               "incomplete_allowed": False, "decoding_confirmed": True}]
    md = {"d2_kind": "main", "d2_split": "test", "d2_scorer_status": "registered",
          "d2_inputs": json.dumps(inputs), "d2_task": "bbh"}
    pq.write_table(t.replace_schema_metadata(md), p)
    with pytest.raises(d2_panels.PanelError, match="expected 'math'") as e:
        d2_panels.math_inputs(p, {"label": "math"})
    assert "test backend" in str(e.value)


def simulated(tmp_path):
    sim = GSM8KSim(np.random.default_rng(7), M=8, C=40, n_items=40)
    sampler = sampling.registered_split_bank(sim.config_ids, np.full(sim.C, 1 / sim.C), 24, 24, "math")
    names = [f"test/algebra/{i}.json" for i in range(sim.N)]

    def col_load(ids):
        df = sim.scored(ids)
        df["item_id"] = df["item_id"].map(lambda i: names[int(i)]).astype(str)
        return df

    rng = np.random.default_rng(11)
    lift = {mk: rng.uniform(-0.3, 0.3) for mk in sim.panel}      # forced minus requested, per model

    def fd_load(ids):
        d = col_load(ids)
        d = d[d["arm"] == "direct_answer"].copy()
        u = sampling.rng_for("test-forced-math", *sorted(map(str, ids))).uniform(size=len(d))
        p = d["model_key"].map(lift).to_numpy()
        flip_up = (d["strict_correct"].to_numpy() == 0) & (u < np.clip(p, 0, None))
        flip_down = (d["strict_correct"].to_numpy() == 1) & (u < np.clip(-p, 0, None))
        d.loc[flip_up, "strict_correct"] = 1
        d.loc[flip_down, "strict_correct"] = 0
        return d.assign(arm=rfd.ARM, finish_reason="stop")

    inp = {"panel": list(sim.panel), "families": list(sim.families), "tiers": None, "load": col_load,
           "items": names, "provenance": {"tables": [], "simulated": True}}
    allu = list(sampler["discovery_config_ids"]) + list(sampler["validation_config_ids"])
    f = fd_load(allu)
    box = pd.DataFrame({"model_key": f["model_key"], "config_id": f["config_id"], "item_id": f["item_id"],
                        "box_closed": True, "prefill": "\\boxed{"})
    return sim, sampler, names, inp, fd_load, box


def test_three_arm_analysis_runs_for_math_on_simulated_data(tmp_path, monkeypatch):
    sim, sampler, names, inp, fd_load, box = simulated(tmp_path)
    calls = []
    monkeypatch.setattr(rfd, "collection_inputs", lambda task: calls.append(task) or (inp, sampler, names, None))
    monkeypatch.setattr(rfd, "forced_inputs",
                        lambda task, panel, s, info, col_load, dry_run: (fd_load, {"gpu_types": {}}, box))
    # The collection's own Steps 1-2 for the reproduction check, as results/dataset2/math.json would hold them.
    disc = list(sampler["discovery_config_ids"])
    Breq, _, Pc = rfd.three_arm_arrays(inp["load"], fd_load, disc, inp["panel"], names)
    sel = d2.load_selection(rfd.SELECTION)
    ref = d2.step2(Breq, Pc, sel, iv.multiplicities(sampling.rng_for("test", "ref"), len(disc), 199))
    fake = tmp_path / "root"
    (fake / "results" / "dataset2").mkdir(parents=True)
    (fake / "results" / "dataset2" / "math.json").write_text(json.dumps(rfd.jsonable({"step2": ref})), encoding="utf-8")
    monkeypatch.setattr(rfd, "ROOT", str(fake))
    a = Namespace(dry_run=False, quick=False, n_boot=199, n_control=20, overwrite_frozen=False, tasks=["math"],
                  suffix="_math", out=str(tmp_path / "out"), pred_dir=str(tmp_path / "frozen"))
    R = rfd.run(a)
    assert calls == ["math"] and list(R["tasks"]) == ["math"]
    T = R["tasks"]["math"]
    assert T["M"] == 8 and T["n_items"] == sim.N and T["K_discovery"] == 24 and T["K_validation"] == 24
    assert T["collection_check"]["abs_dev"]["H"] <= 1e-9
    pm = pd.DataFrame(T["per_model"])
    assert np.allclose(pm["collection_gain"], pm["reasoning_benefit"] - pm["adherence_cost"])
    assert abs(T["parts"]["identity_residual"]) < 1e-8
    assert (tmp_path / "frozen" / "predictions_forced_map_math.json").exists()
    assert (tmp_path / "frozen" / "predictions_forced_math.json").exists()
    assert not (tmp_path / "frozen" / "predictions_forced_map.json").exists()
    fr = json.loads((tmp_path / "frozen" / "predictions_forced_map_math.json").read_text(encoding="utf-8"))
    assert fr["label"] == "forced_map_math" and list(fr["settings"]) == ["math"]
    assert R["decoding"] == {"math": rfd.DECODING_NOTE["math"]}
    assert all(k.startswith("fdMath") for k in R["macros"] if k.startswith("fd") and "Pilot" not in k)
    assert [p["setting"] for p in R["map_points"]] == ["forced:math", "forced:math"]
    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "decoding_forced_math.yaml" in summary and "`forced_math.json` here" in summary
    assert (tmp_path / "out" / "tables" / "per_model_math.csv").exists()
