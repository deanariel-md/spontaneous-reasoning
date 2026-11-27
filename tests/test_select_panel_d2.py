"""The Dataset 2 panel rule (scripts/select_panel_d2.py) and its frozen manifest."""
import importlib.util
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("select_panel_d2", os.path.join(ROOT, "scripts", "select_panel_d2.py"))
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def fam(ecis, params=None):
    params = params or [1.0] * len(ecis)
    return pd.DataFrame({"eci": ecis, "params_b": params, "repo": [f"v/m{i}" for i in range(len(ecis))]})


def test_a_family_at_or_below_k_keeps_every_checkpoint():
    out = sp.select_family(fam([120.0, 100.0, 110.0]), k=4)
    assert list(out["eci"]) == [100.0, 110.0, 120.0]
    assert set(out["selected_for"]) == {"all eligible"}


def test_a_larger_family_keeps_the_ends_and_the_nearest_to_each_interior_target():
    # range 100-130, targets 110 and 120
    out = sp.select_family(fam([100, 104, 109, 113, 119, 126, 130]), k=4)
    assert list(out["eci"]) == [100, 109, 119, 130]
    assert out["selected_for"].iloc[0] == "lowest ECI" and out["selected_for"].iloc[-1] == "highest ECI"


def test_ties_go_to_the_lower_eci_then_fewer_parameters():
    out = sp.select_family(fam([100, 108, 112, 130], params=[1, 1, 1, 1]), k=3)  # target 115: 112 nearest
    assert list(out["eci"]) == [100, 112, 130]
    out = sp.select_family(fam([100, 110, 120, 140], params=[1, 1, 1, 1]), k=3)  # target 120 exact
    assert list(out["eci"]) == [100, 120, 140]
    out = sp.select_family(fam([100, 115, 125, 140]), k=3)  # target 120: 115 and 125 tie, lower ECI wins
    assert list(out["eci"]) == [100, 115, 140]
    out = sp.select_family(fam([100, 120, 120, 140], params=[1, 9, 2, 1]), k=3)  # same ECI: fewer parameters
    assert out["params_b"].iloc[1] == 2


def _eci(rows):
    return pd.DataFrame([{"Model": n, "eci": e, "eci_ci_low": e - 3, "eci_ci_high": e + 3, "date": "2025-01-01",
                          "Accessibility group": g} for n, e, g in rows])


def test_eligibility_reasons(monkeypatch):
    monkeypatch.setattr(sp, "MAPPING", {"A": ("F", "v/a"), "B": ("F", "v/b"), "C": ("F", "v/c"), "D": ("F", "v/d"),
                                        "E": ("F", None, sp.BASE)})
    monkeypatch.setattr(sp, "EXTRA_HF", {})
    hf = {"A": {"sha": "1", "created": "2025-01-01", "gated": False, "params_b": 8, "weights_gb": 16, "arch": ["X"]},
          "B": {"sha": "2", "created": "2026-07-01", "gated": False, "params_b": 8, "weights_gb": 16, "arch": ["X"]},
          "C": {"sha": "3", "created": "2025-01-01", "gated": False, "params_b": 40, "weights_gb": 80, "arch": ["X"]},
          "D": {"sha": "4", "created": "2025-01-01", "gated": False, "params_b": 8, "weights_gb": 16, "arch": ["Y"]}}
    archs = {"X": {"supported": True, "is_hybrid": False}, "Y": {"supported": False}}
    eci = _eci([(n, 120, "Open weights") for n in "ABCDE"] + [("Z", 150, "Closed weights")])
    c = sp.candidates(eci, hf, archs).set_index("eci_name")
    assert "Z" not in c.index
    assert c.loc["A", "eligible"] and c.loc["A", "reason"] == ""
    assert c.loc["B", "reason"] == sp.AFTER
    assert c.loc["C", "reason"] == sp.SIZE
    assert "not supported by vLLM" in c.loc["D", "reason"]
    assert c.loc["E", "reason"] == sp.BASE and not c.loc["E", "eligible"]


def test_an_unmapped_index_row_stops_the_build(monkeypatch):
    monkeypatch.setattr(sp, "MAPPING", {})
    with pytest.raises(SystemExit, match="no mapping decision"):
        sp.candidates(_eci([("New model", 130, "Open weights")]), {}, {})


def test_small_families_are_dropped():
    cand = pd.DataFrame({"eligible": True, "family": ["F"] * 3 + ["G"] * 2, "eci": [100, 110, 120, 105, 115],
                         "params_b": 1.0, "repo": [f"v/{i}" for i in range(5)]})
    panel, fams = sp.select_panel(cand)
    assert fams["kept"] == ["F"] and fams["dropped"] == ["G"]
    assert len(panel) == 3 and set(panel["tier"]) == {"low", "middle", "high"}


def test_the_frozen_panel_reproduces_from_the_frozen_inputs():
    man = json.load(open(os.path.join(ROOT, "manifests", "d2_panel.json"), encoding="utf-8"))
    for rel, digest in man["inputs"].items():
        assert sp.sha256(os.path.join(ROOT, rel)) == digest, rel
    eci = pd.read_csv(sp.ECI_FILE)
    hf = json.load(open(sp.HF_FILE, encoding="utf-8"))
    archs = json.load(open(sp.VLLM_FILE, encoding="utf-8"))["archs"]
    panel, fams = sp.select_panel(sp.candidates(eci, hf, archs))
    assert list(panel["repo"]) == [p["repo"] for p in man["panel"]]
    assert fams["kept"] == man["families"]["kept"]
    assert 4 <= len(fams["kept"]) <= 6 and 16 <= len(panel) <= 24  # about 20 models from 4-6 families


def test_the_extension_keeps_every_eligible_checkpoint_of_the_kept_families():
    out = sp.select_family(fam([100, 104, 109, 113, 119, 126, 130]), k=None)
    assert list(out["eci"]) == [100, 104, 109, 113, 119, 126, 130]
    assert set(out["selected_for"]) == {"all eligible"}


def test_the_frozen_extension_manifest_contains_the_main_panel():
    base = json.load(open(os.path.join(ROOT, "manifests", "d2_panel.json"), encoding="utf-8"))
    ext = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
    assert ext["parameters"]["k_per_family"] is None and ext["inputs"] == base["inputs"]
    assert {p["repo"] for p in base["panel"]} <= {p["repo"] for p in ext["panel"]}
    assert sorted(set(p["repo"] for p in ext["panel"]) - set(p["repo"] for p in base["panel"])) == ext["added_to_main"]
    assert ext["counts"]["panel"] == 30 and len(ext["added_to_main"]) == 11
    assert ext["families"]["kept"] == base["families"]["kept"]
