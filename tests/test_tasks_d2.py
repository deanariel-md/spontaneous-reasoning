"""Secondary Dataset 2 analyses across tasks (src/tasks_d2.py, scripts/run_dataset2_tasks.py; registration E2)."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import bbh, tasks_d2
from src.estimators import cross_moment, fixed_task_components

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_dataset2_tasks", ROOT / "scripts" / "run_dataset2_tasks.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_bbh_families_are_frozen_from_task_names():
    fams = {t: tasks_d2.bbh_family(t) for t in bbh.TASKS}
    assert len(set(fams.values())) == 21
    assert {fams[t] for t in bbh.TASKS if t.startswith("logical_deduction_")} == {"logical_deduction"}
    assert {fams[t] for t in bbh.TASKS if t.startswith("tracking_shuffled_objects_")} == {"tracking_shuffled_objects"}
    assert fams["web_of_lies"] == "web_of_lies"


def test_decomposition_is_the_appendix_a_estimator_with_a_jackknife():
    rng = np.random.default_rng(0)
    G = {t: rng.normal(size=(10, 6)) + rng.normal(size=6) for t in ("a", "b", "c")}
    w = {"a": 0.5, "b": 0.25, "c": 0.25}
    d = tasks_d2.decomposition(G, w)
    ref = fixed_task_components(np.stack(list(G.values())), np.array(list(w.values())))
    for k in ("average_task_profile", "model_by_task", "true_within_task", "prompt_noise", "total"):
        assert d[k]["estimate"] == pytest.approx(ref[k])
        assert d[k]["jackknife_se"] > 0 and d[k]["method"] == "jack_t"
    assert d["average_task_profile"]["estimate"] + d["model_by_task"]["estimate"] == pytest.approx(
        d["true_within_task"]["estimate"])
    assert [r["V_delta"] for r in d["per_task"]] == pytest.approx([cross_moment(g, g) for g in G.values()])
    assert d["K_bundles"] == 10 and d["M"] == 6


def test_a_negative_component_is_reported_as_it_is():
    rng = np.random.default_rng(1)
    G = {t: rng.normal(size=(4, 5)) for t in ("a", "b")}                     # pure noise: estimates scatter around 0
    est = [tasks_d2.decomposition(G, {"a": 1, "b": 1})["model_by_task"]["estimate"]]
    for s in range(2, 30):
        r = np.random.default_rng(s)
        est.append(tasks_d2.decomposition({t: r.normal(size=(4, 5)) for t in "ab"}, {"a": 1, "b": 1})
                   ["model_by_task"]["estimate"])
    assert min(est) < 0


def test_transfer_score_is_zero_risk_for_a_perfect_prediction():
    target = np.array([3.0, -1.0, -2.0, 0.0])
    Z = np.tile(target, (5, 1)) + 10.0                                       # a shift: centring removes it
    r = tasks_d2.transfer_score(target, Z)
    assert r["risk_transfer"] == pytest.approx(0.0)
    assert r["risk_zero"] == pytest.approx(np.mean(target ** 2))
    assert r["difference"]["estimate"] == pytest.approx(np.mean(target ** 2))
    assert r["difference"]["jackknife_se"] == pytest.approx(0.0)


def test_within_bbh_predictions_hold_each_family_out():
    rng = np.random.default_rng(2)
    D = {t: rng.normal(size=(6, 4)) for t in bbh.TASKS}
    p = tasks_d2.within_bbh_transfer_predictions(D)
    assert len(p) == 21
    ld = p["logical_deduction"]
    assert len(ld["held_out_tasks"]) == 3 and not any(t.startswith("logical_deduction") for t in ld["train_tasks"])
    assert sum(ld["train_weights"]) == pytest.approx(1.0) and len(ld["train_tasks"]) == 22
    assert np.mean(ld["prediction"]) == pytest.approx(0.0)


# ---------------------------------------------------------------- the script on simulated scored outputs

def _scored(rng, models, configs, items, profile, noise=4.0):
    """Item-level outputs: model i's direct accuracy and gain fixed, each configuration shifting both arms."""
    rows = []
    for c in configs:
        shift = rng.normal(0, noise, size=len(models)) / 100
        for i, m in enumerate(models):
            for arm, add in (("direct_answer", 0.0), ("zero_shot_cot", profile[i])):
                p = np.clip(0.4 + add + shift[i], 0.02, 0.98)
                ok = rng.random(len(items)) < p
                rows += [{"model_key": m, "arm": arm, "config_id": c, "item_id": it, "strict_correct": int(o),
                          "lenient_correct": int(o), "compliant": True} for it, o in zip(items, ok)]
    return pd.DataFrame(rows)


def _sampler(prefix, k):
    ids = [f"{prefix}{j}" for j in range(2 * k)]
    return {"sha256": prefix, "discovery_config_ids": ids[:k], "validation_config_ids": ids[k:]}


def test_the_script_freezes_transfer_before_reading_validation_units(tmp_path):
    rng = np.random.default_rng(5)
    models = [f"m{i}" for i in range(6)]
    profile = np.array([0.30, 0.20, 0.10, 0.00, -0.05, 0.15])
    s_g, s_b = _sampler("g", 6), _sampler("b", 6)
    g_items = [str(i) for i in range(60)]
    by_task = {t: [f"{t}/{i}" for i in range(6)] for t in bbh.TASKS}
    gdf = _scored(rng, models, s_g["discovery_config_ids"] + s_g["validation_config_ids"], g_items, profile)
    bdf = _scored(rng, models, s_b["discovery_config_ids"] + s_b["validation_config_ids"],
                  [i for v in by_task.values() for i in v], profile)
    pred = tmp_path / "predictions_tasks.json"
    seen = []

    def loader(df, s):
        def load(ids):
            seen.append((s["sha256"], set(ids) <= set(s["discovery_config_ids"]), pred.exists()))
            return df[df["config_id"].isin(set(ids))]
        return load

    gsm = {"panel": models, "load": loader(gdf, s_g), "items": g_items, "provenance": {}}
    bb = {"panel": models[:5] + ["other"], "load": loader(bdf, s_b), "items_by_task": by_task, "provenance": {}}
    out = rt.run(gsm, bb, s_g, s_b, str(tmp_path / "res"), str(pred))
    # discovery reads come first and before the freeze; every validation read after it
    disc = [x for x in seen if x[1]]
    val = [x for x in seen if not x[1]]
    assert len(disc) == 2 and not any(x[2] for x in disc) and len(val) == 2 and all(x[2] for x in val)
    assert out["models"] == models[:5] and out["left_out"] == {"gsm8k_only": ["m5"], "bbh_only": ["other"]}
    frozen = json.loads(pred.read_text(encoding="utf-8"))
    assert frozen["sha256"] == out["frozen_predictions"]["sha256"] and len(frozen["predictions"]["within_bbh"]) == 21
    # the same profile in both tasks: transfer beats zero centred benefit in both directions
    for case in ("gsm8k_to_bbh", "bbh_to_gsm8k"):
        assert out["two_tasks"]["transfer"][case]["difference"]["estimate"] > 0
    dec = out["two_tasks"]["decomposition"]
    assert dec["weights"] == {"gsm8k": 0.5, "bbh": 0.5} and dec["K_bundles"] == 6
    assert out["within_bbh"]["transfer_summary"]["n_families"] == 21
    assert out["within_bbh"]["transfer"][0]["items_per_task"][0] == 6
    for f in ("decomposition.csv", "transfer.csv", "bbh_per_task.csv"):
        assert (tmp_path / "res" / "tables_tasks" / f).exists()
    # a second run with the same predictions keeps the frozen file; different predictions are refused
    created = frozen["created"]
    rt.run(gsm, bb, s_g, s_b, str(tmp_path / "res"), str(pred))
    assert json.loads(pred.read_text(encoding="utf-8"))["created"] == created
    gsm2 = {**gsm, "load": loader(gdf.assign(strict_correct=1 - gdf["strict_correct"]), s_g)}
    with pytest.raises(FileExistsError):
        rt.run(gsm2, bb, s_g, s_b, str(tmp_path / "res"), str(pred))


def test_the_script_refuses_unequal_discovery_counts(tmp_path):
    gsm = {"panel": ["a", "b"], "load": None, "items": [], "provenance": {}}
    bb = {"panel": ["a", "b"], "load": None, "items_by_task": {}, "provenance": {}}
    with pytest.raises(ValueError, match="pair them by position"):
        rt.run(gsm, bb, _sampler("g", 3), _sampler("b", 4), str(tmp_path), str(tmp_path / "p.json"))
