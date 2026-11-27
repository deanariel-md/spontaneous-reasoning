"""The five-shot requested-direct arm (src/fewshot_requested.py, scripts/build_fewshot_requested_requests.py,
scripts/score_fewshot_requested.py, scripts/run_fewshot_requested.py, paper/macros_fsr.py). Synthetic data only; no model
output of the run is read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "paper")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import fewshot_requested as fr  # noqa: E402
import run_fewshot_requested as rfr  # noqa: E402
import score_fewshot_requested as sfr  # noqa: E402
import macros_fsr  # noqa: E402


# ------------------------------------------------------------------------------------------ rule and saved draw


def test_draw_is_the_written_rule_and_saved():
    man = json.loads((ROOT / "manifests" / "fewshot_requested_configs.json").read_text(encoding="utf-8"))
    for t in fr.TASKS:
        s = json.loads((ROOT / "manifests" / f"sampler_{t}.json").read_text(encoding="utf-8"))
        d = fr.draw_configurations(t, s)
        assert man["tasks"][t]["discovery"] == d["discovery"] and man["tasks"][t]["validation"] == d["validation"]
        assert len(set(d["discovery"])) == 4 and len(set(d["validation"])) == 4
        assert set(d["discovery"]) <= set(s["discovery_config_ids"])
        assert set(d["validation"]) <= set(s["validation_config_ids"])
        assert not set(d["validation"]) & set(s["discovery_config_ids"])     # fresh: no discovery draw uses them
        assert d["discovery_seed_label"] == f"d28/fsr/{t}/discovery"


def test_draw_refuses_short_pools():
    with pytest.raises(fr.FewshotRequestedError):
        fr.draw_configurations("gsm8k", {"discovery_config_ids": ["a", "b"], "validation_config_ids": ["c"]})


def test_build_record_counts():
    p = ROOT / "data" / "requests" / "fewshot_requested" / "collection" / "build_record.json"
    if not p.exists():
        pytest.skip("request files not built on this machine")
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["n_requests"] == 29 * 8 * 1319 + 28 * 8 * 1350 == 608408
    assert len(rec["models"]) == 29
    assert rec["decoding"] == "config/decoding.yaml" and rec["arm"] == "direct_answer"
    for k, v in rec["models"].items():
        for t, i in v["tasks"].items():
            assert i["final_turn_matches"] == i["n"] and i["configs"] == 8
    assert set(rec["models"]["gemma-4-31b-it"]["tasks"]) == {"gsm8k"}


def test_context_record_fits():
    p = ROOT / "results" / "fewshot_requested" / "context_check.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["all_fit"] and r["cap"] == 1024
    assert max(v["plus_cap"] for m in r["models"].values() for v in m["tasks"].values()) <= 4096


# ------------------------------------------------------------------------------------------ arithmetic


def test_pair_count_ties_and_reversals():
    g1 = np.array([1.0, 2.0, 3.0, 3.0])
    g2 = np.array([2.0, 1.0, 3.0, 4.0])
    r = fr.pair_count(g1, g2)
    assert r["pairs"] == 6 and r["reversed"] == 1 and r["ties"] == 1
    assert r["one_minus_C"] == pytest.approx(1.5 / 6)
    z = fr.pair_count(g1, g1 + 5)
    assert z["reversed"] == 0 and z["reversed_share"] == 0.0     # zero reversals are zero


def test_template_reversal_and_choices():
    G = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    tr = fr.template_reversal(G)
    assert tr["configuration_pairs"] == 1 and tr["reversed_share"] == 1.0
    W, ties = fr.one_draw_choices(G, 1)
    assert W.tolist() == [[1, 0, 0], [0, 0, 1]] and ties == [0, 0]


def test_accuracy_array_refuses_missing_items():
    df = pd.DataFrame({"model_key": ["a", "a", "b"], "config_id": ["c"] * 3, "item_id": ["1", "2", "1"],
                       "strict_correct": [1, 0, 1]})
    with pytest.raises(fr.FewshotRequestedError):
        fr.accuracy_array(df, ["c"], ["a", "b"], 2)
    ok = fr.accuracy_array(df[df.model_key == "a"], ["c"], ["a"], 2)
    assert ok.tolist() == [[50.0]]


# ------------------------------------------------------------------------------------------ synthetic pipeline


PANEL = {"gsm8k": ["fa-1", "fa-2", "fb-1", "fb-2", "fc-1", "fc-2", "fd-1"],
         "bbh": ["fa-1", "fa-2", "fb-1", "fb-2", "fc-1", "fc-2"]}
CFG = {t: {"discovery": [f"d{i}" for i in range(4)], "validation": [f"v{i}" for i in range(4)]} for t in fr.TASKS}
ITEMS = [str(i) for i in range(12)]


def _long(task, arm, seed):
    rng = np.random.default_rng(seed)
    rows = []
    shift = {"zs_requested": 0.0, "zs_forced": -0.1, "fs_forced": 0.05, "fs_requested": 0.1}[arm]
    for j, k in enumerate(PANEL[task]):
        p = np.clip(0.25 + 0.07 * j + shift + (0.03 * j if arm == "fs_requested" else 0), 0.02, 0.98)
        for c in CFG[task]["discovery"] + CFG[task]["validation"]:
            for i in ITEMS:
                v = int(rng.random() < p)
                rows.append({"model_key": k, "config_id": c, "item_id": i, "strict_correct": v, "lenient_correct": v})
    return pd.DataFrame(rows)


def _setup(base: Path):
    paths = fr.Paths(base)
    man = {"tasks": CFG}
    fr.write_json(paths.configs, man)
    rec = {"configs_manifest_sha256": fr.sha256_file(paths.configs), "configurations": CFG,
           "models": {k: {"tasks": {t: {} for t in fr.TASKS if k in PANEL[t]}} for k in PANEL["gsm8k"]}}
    fr.write_json(paths.requests / "collection" / "build_record.json", rec)
    ex = base / "existing"
    ex.mkdir()
    new = []
    for s, t in enumerate(fr.TASKS):
        for a in rfr.EXISTING:
            _long(t, a, 10 * s + rfr.EXISTING.index(a)).to_parquet(ex / f"{t}_{a}.parquet")
        new.append(_long(t, "fs_requested", 100 + s).assign(task=t, arm="direct_answer"))
    new = pd.concat(new, ignore_index=True)
    return paths, ex, new


def _write_stage(paths, new, stage, pred_sha):
    paths.processed.mkdir(parents=True, exist_ok=True)
    keep = new if stage == "all" else new[new["config_id"].str.startswith("d")]
    keep.to_parquet(paths.processed / f"{stage}_scored.parquet", index=False)
    fr.write_json(paths.processed / f"{stage}_scored.meta.json", {"stage": stage, "predictions_sha256": pred_sha})


def test_order_and_refusals(tmp_path):
    paths, ex, new = _setup(tmp_path)
    args = ["--base", str(tmp_path), "--existing", str(ex), "--n-boot", "99"]
    # nothing scored at stage all, and no full run, before the predictions are saved
    assert sfr.main(["--stage", "all", "--base", str(tmp_path)]) == 2
    _write_stage(paths, new, "all", None)
    assert rfr.main(args) == 2
    # discovery stage saves the predictions once
    _write_stage(paths, new, "discovery", None)
    assert rfr.main(args + ["--stage", "discovery"]) == 0
    sha = fr.check_predictions(paths)
    assert rfr.main(args + ["--stage", "discovery"]) == 2            # written once
    saved = json.loads(paths.predictions.read_text(encoding="utf-8"))
    for t in fr.TASKS:
        b = saved["tasks"][t]
        assert len(b["choices"]["requested"]) == 4 and b["n_chosen"] == round(len(PANEL[t]) / 3)
        assert np.allclose(np.sum(b["choices"]["forced"], axis=1), b["n_chosen"])
    # a table scored without the saved hash is refused
    _write_stage(paths, new, "all", None)
    assert rfr.main(args) == 2
    _write_stage(paths, new, "all", "0" * 64)
    assert rfr.main(args) == 2
    # a changed predictions file is refused by both scripts
    raw = paths.predictions.read_bytes()
    paths.predictions.write_bytes(raw.replace(b'"tolerance": 0.05', b'"tolerance": 0.06'))
    assert sfr.main(["--stage", "all", "--base", str(tmp_path)]) == 2
    _write_stage(paths, new, "all", sha)
    assert rfr.main(args) == 2
    paths.predictions.write_bytes(raw)
    # discovery rows that differ from those the predictions were fitted on are refused
    alt = new.copy()
    d0 = (alt["config_id"] == "d0") & (alt["task"] == "gsm8k")
    alt.loc[d0, "strict_correct"] = 1 - alt.loc[d0, "strict_correct"]
    _write_stage(paths, alt, "all", sha)
    assert rfr.main(args) == 2
    # the full run
    _write_stage(paths, new, "all", sha)
    assert rfr.main(args) == 0
    R = json.loads((paths.results / "fewshot_requested.json").read_text(encoding="utf-8"))
    for t in fr.TASKS:
        s = R["tasks"][t]["strict"]
        M = len(PANEL[t])
        assert s["reversal_validation"]["pairs"] == M * (M - 1) // 2
        assert s["K_validation"] == 4
        for k in ("requested_to_forced", "forced_to_requested"):
            d = s["decision_one_draw"][k]
            assert d["K"] == 4 and d["ci95"] is not None
            # extra regret equals the mismatched minus the matched plug-in regret (the best third cancels)
            assert d["estimate"] == pytest.approx(d["regret_mismatched"] - d["regret_matched"], abs=1e-9)
        assert len(s["per_model"]) == M
    for name in ("summary.md", "summary_clean.md", "tables/costs.csv", "tables/reversals.csv",
                 "tables/per_model_gsm8k.csv"):
        assert (paths.results / name).exists()
    m = macros_fsr.fsr_macros(R)
    assert all(k.isalpha() and k.startswith("fsr") for k in m)
    assert m["fsrGsmRev"] == str(R["tasks"]["gsm8k"]["strict"]["reversal_validation"]["reversed"])
    assert m["fsrValConfigs"] == "4"


def test_score_refuses_dry_options_inside_repository():
    assert sfr.main(["--stage", "discovery", "--synthetic", "1"]) == 2


def test_fsr_results_absent_is_none(tmp_path):
    assert macros_fsr.fsr_results(str(tmp_path)) is None


def test_scorers_read_requested_outputs():
    """The two scorers on hand-written requested replies (gold answers only; no model output)."""
    from src import d2_io, screening
    spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
    g = sfr.items("gsm8k").head(1)
    b = sfr.items("bbh").head(1)
    out_g = pd.DataFrame({"request_id": ["x"], "model_key": ["m"], "arm": ["direct_answer"], "config_id": ["c"],
                          "item_id": [g["item_id"].iloc[0]],
                          "output_text": [f"\\boxed{{{g['gold_answer'].iloc[0]}}}"]})
    s = sfr.score_task(out_g, "gsm8k", g, spec)
    assert int(s["strict_correct"].iloc[0]) == 1
    out_b = out_g.assign(item_id=[b["item_id"].iloc[0]], output_text=[f"\\boxed{{{b['gold_answer'].iloc[0]}}}"])
    s = sfr.score_task(out_b, "bbh", b, spec)
    assert int(s["strict_correct"].iloc[0]) == 1 and bool(s["box_closed"].iloc[0])
