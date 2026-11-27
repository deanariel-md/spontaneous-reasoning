"""scripts/write_curve_tables.py: the curve and regret tables and the line
each writes into its folder's summary."""
import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import write_curve_tables as wfc  # noqa: E402


def test_figure_csvs_and_readme_lines(tmp_path):
    dec = {"settings": {"gsm8k_req": {"title": "t", "result": {"M": 29, "n_chosen": 10, "evaluation": {"independent": {
        "n_configurations": 10, "best_third_value": 68.9, "panel_mean_gain": 43.8,
        "rules": {"capability_1": {"n_sets": 19, "regret_plugin": 29.43, "regret_crossfit": 29.16,
                                   "regret_plugin_p10": 20.9, "regret_plugin_p90": 33.3, "share_best_third": 0.0},
                  "random": {"n_sets": 200, "regret_plugin": 25.1, "regret_crossfit": 25.0}}}}}}}}
    pc = {"tasks": {"gsm8k": {"headline": {"protocols": ["a", "b"], "within_requested": {"2": 0.97, "1": 0.956},
                                           "between_same_draws": {"1": 0.80}, "plugin": 0.79,
                                           "plugin_bootstrap": {"ci95": [0.788, 0.805]}, "posterior": 0.797},
                              "axes": {"arm": {"pairs": 9, "same_draws": {"1": 0.81}, "limit_plugin": 0.809,
                                               "limit_posterior": 0.81},
                                       "within": {"protocols": 18, "empirical": {"1": 0.957}, "model": {"24": 0.99}}}}}}
    for name, obj in (("decision", dec), ("protocol_ceiling", pc)):
        d = tmp_path / "results" / name
        d.mkdir(parents=True)
        (d / f"{name}.json").write_text(json.dumps(obj))
        with open(d / "summary.md", "w", newline="") as fh:
            fh.write("# S\r\n\r\nText.\r\n")
        (d / "summary_clean.md").write_text("# S\n", newline="\n")
    wfc.main(["--root", str(tmp_path)])
    wfc.main(["--root", str(tmp_path)])                    # a second run replaces its line, adds none
    rows = list(csv.DictReader(open(tmp_path / "results" / "decision" / "tables" / "figure_regret.csv")))
    assert [r["rule"] for r in rows] == ["capability_1", "random"]
    assert rows[0]["task"] == "GSM8K-29" and rows[0]["regret_plugin"] == "29.43" and rows[0]["best_third_gain"] == "68.9"
    assert rows[1]["regret_plugin_p10"] == "" and rows[1]["rule_label"] == "random third"
    cur = list(csv.DictReader(open(tmp_path / "results" / "protocol_ceiling" / "tables" / "figure_curves.csv")))
    head = [r for r in cur if r["curve"] == "headline_within_requested"]
    assert [r["K"] for r in head] == ["1", "2"] and head[0]["limit_plugin"] == "1"
    bt = [r for r in cur if r["curve"] == "headline_between_same_draws"][0]
    assert bt["limit_plugin_lo"] == "0.788" and bt["limit_posterior"] == "0.797"
    ax = [r for r in cur if r["curve"] == "axis_arm_same_draws"][0]
    assert ax["pairs"] == "9" and ax["limit_plugin_lo"] == "" and ax["limit_plugin"] == "0.809"
    assert {r["curve"] for r in cur} >= {"within_all_empirical", "within_all_model"}
    raw = open(tmp_path / "results" / "decision" / "summary.md", newline="").read()
    assert raw.count("Figure table: `tables/figure_regret.csv`") == 1 and raw.endswith(".\r\n")
    assert raw.startswith("# S\r\n\r\nText.\r\n\r\nFigure table:") and "\n" not in raw.replace("\r\n", "")
    clean = open(tmp_path / "results" / "protocol_ceiling" / "summary_clean.md", newline="").read()
    assert clean.count("Figure table: `tables/figure_curves.csv`") == 1 and "\r" not in clean


def test_figure_regret_matches_the_regret_table():
    src = os.path.join(ROOT, "results", "decision", "decision.json")
    reg = os.path.join(ROOT, "results", "decision", "tables", "regret.csv")
    if not (os.path.exists(src) and os.path.exists(reg)):
        pytest.skip("decision results absent")
    rows = wfc.decision_rows(json.load(open(src, encoding="utf-8")))
    ref = [r for r in csv.DictReader(open(reg, encoding="utf-8")) if r["evaluation"] == "independent"]
    assert len(rows) == len(ref)
    by = {(r["setting"], r["rule"]): r for r in ref}
    for r in rows:
        x = by[(r["setting"], r["rule"])]
        assert float(x["regret_plugin"]) == pytest.approx(r["regret_plugin"], rel=1e-12)
        assert float(x["regret_crossfit"]) == pytest.approx(r["regret_crossfit"], rel=1e-12)
