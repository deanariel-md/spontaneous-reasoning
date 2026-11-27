"""The full-scale exemplar row of the patterns table (scripts/run_patterns.py, cheng_full) on a fabricated exemplar-run file."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))
import run_patterns as rp  # noqa: E402


def fake_exemplar(dry_run=False, kind="bounded"):
    scale = {sc: {"discovery": {"slope": s, "slope_set": {"kind": kind, "a": s - 0.1, "b": s + 0.1},
                                "slope_set_sign": "below zero" if s < 0 else "above zero"}}
             for sc, s in (("points", -0.2), ("logit", -0.05), ("error_reduction", 0.3))}
    draws = {sc: {"per_draw": [], "positive": p, "negative": 24 - p, "zero": 0, "undefined": 0, "repeats": 0.75}
             for sc, p in (("points", 3), ("logit", 6), ("error_reduction", 18))}
    return {"dry_run": dry_run, "partial": False, "M": 29, "K_discovery": 24, "scale": scale, "draw_slopes": draws,
            "cheng": {"models": ["a", "b", "c", "d", "e"],
                      "scorers": {"strict": {"claim_holds_in": 20, "n_configurations": 29}}}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "results" / "exemplar")
    monkeypatch.setattr(rp, "ROOT", str(tmp_path))
    return tmp_path


def write(root, x):
    (root / "results" / "exemplar" / "exemplar.json").write_text(json.dumps(x), encoding="utf-8")


def test_cheng_full_reads_the_full_run(root):
    write(root, fake_exemplar())
    h = rp.cheng_full()
    assert (h["M"], h["K"], h["claim_models"], h["claim_holds_in"], h["of"]) == (29, 24, 5, 20, 29)
    assert h["scales"]["points"]["slope"] == -0.2 and h["scales"]["points"]["negative"] == 21
    assert h["scales"]["error_reduction"]["positive"] == 18


def test_cheng_full_ignores_a_dry_run_and_a_missing_file(root):
    assert rp.cheng_full() is None and rp.cheng_full_rows(None) == [] and rp.cheng_full_macros(None) == {}
    write(root, fake_exemplar(dry_run=True))
    assert rp.cheng_full() is None


def test_cheng_full_rows_and_macros(root):
    write(root, fake_exemplar())
    h = rp.cheng_full()
    rows = rp.cheng_full_rows(h)
    assert [r["specification"] for r in rows] == ["exemplar/points/strict", "exemplar/logit/strict",
                                                 "exemplar/error_reduction/strict"]
    assert rows[0]["estimate"] == -0.2 and rows[0]["below"] == 21 and rows[0]["claim_holds_in"] == 20
    assert rows[1]["lo"] == pytest.approx(-0.15)
    m = rp.cheng_full_macros(h)
    assert m["patChengFullClaim"] == "20" and m["patChengFullClaimOf"] == "29" and m["patChengFullClaimModels"] == "5"
    assert m["patChengFullPointsNeg"] == "21" and m["patChengFullErrRedPos"] == "18"
    assert m["patChengFullPointsRepeats"] == "75"
    assert all(k.isalpha() for k in m)


def test_an_unbounded_set_prints_its_kind(root):
    write(root, fake_exemplar(kind="exclusive"))
    h = rp.cheng_full()
    assert rp.cheng_full_rows(h)[0]["lo"] is None
    assert rp.cheng_full_macros(h)["patChengFullPointsSlopeLo"] == "exclusive"
