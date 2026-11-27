"""Tests of scripts/ladder_analysis.py (size ladder).

1. DeepSeek V4.1-Flash: every value the analysis computes from the scored rows equals results/frontier_panel/
   nebius_panel.json (accuracy, gains, w, corrected gain, template SD, five-shot arms; both readings).
2. A ladder model without files is reported as missing, with its benchmarks listed, never dropped.
3. A fake model written in scripts/score_ladder.py's layout (data/processed/ladder{,_uncut}/<task>/<key>.parquet)
   is picked up on the next run, and its numbers equal those of the rows it was made from.
The data-dependent tests skip when the git-ignored scored rows are not on this machine.
"""
from __future__ import annotations

import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ladder_analysis as la  # noqa: E402

DS = "deepseek-v4.1-flash-1017"
FAKE = "ministral-3-8b-instruct-2512"
DS_FILES = [ROOT / f"data/frontier_panel/scored/{t}_scored{s}.parquet" for t in la.TASKS for s in ("", "_uncut")]
NEBIUS = ROOT / la.NEBIUS_JSON


def panel_root(tmp: Path) -> Path:
    """A root holding only the panel's result files (small CSV and JSON), no ladder rows."""
    for rel in la.PANEL_FILES.values():
        if "{task}" in rel:
            for t in ("gsm8k", "math", "bbh"):
                src = ROOT / rel.format(task=t)
                (tmp / rel.format(task=t)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, tmp / rel.format(task=t))
        else:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, tmp / rel)
    return tmp


needs_ds = pytest.mark.skipif(not (all(p.exists() for p in DS_FILES) and NEBIUS.exists()),
                              reason="DeepSeek's scored rows or its report are not on this machine")


@needs_ds
def test_deepseek_reproduces_nebius_report(tmp_path):
    R = la.analyse(ROOT, tmp_path / "out", keys=[DS])
    rep = R["reproduction"]
    assert rep["values_compared"] > 500
    assert rep["mismatches"] == 0, rep["first_mismatches"]
    assert R["models"][DS]["status"] == "complete"
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    assert f"{rep['values_compared']} values compared" in md
    assert (tmp_path / "out" / "ladder_clean.md").exists()
    # DeepSeek's HotpotQA values that sit exactly halfway print as their even neighbour
    h = R["models"][DS]["tasks"]["hotpot"]["cut"]
    assert h["content"]["pooled"]["gain_over_forced_pp"] == Fraction(611, 72)          # 8.486...
    assert h["strict"]["discovery"]["gain_over_forced_pp"] == Fraction(177, 20)        # 8.85
    # the strict 8.85 above sits exactly halfway; the content accuracy below does not
    assert h["content"]["pooled"]["accuracy_pp"]["cot"] == Fraction(22811, 360)        # 63.364...
    row = next(x for x in md.splitlines() if x.startswith("| DeepSeek V4.1-Flash-1017 | 8.8 ("))
    assert row.split(" | ")[2].startswith("8.5 (")
    assert "HotpotQA, CoT − forced (content), pooled rows: 8.5 pp" in md
    assert la.fnum(h["content"]["pooled"]["accuracy_pp"]["cot"]) == "63.4"
    assert R["rounding_edge_cases"] == []


def test_reproduction_flags_a_changed_value():
    ds = {"tasks": {"gsm8k": {"cut": None}}}
    rep = la.reproduce_nebius(ds, {"tasks": {}, "tasks_uncut": {}, "fewshot": {}})
    assert rep["mismatches"] > 0          # absent on one side is a mismatch, not a pass


def test_missing_models_are_reported_not_dropped(tmp_path):
    root = panel_root(tmp_path)
    R = la.analyse(root, tmp_path / "out")
    assert set(R["models"]) == set(la.LADDER)
    for k, m in R["models"].items():
        assert m["status"] == "missing"
        assert m["tasks_missing"] == list(la.TASKS)
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    for k in la.LADDER:
        row = next(line for line in md.splitlines() if line.startswith(f"| {la.LADDER[k]['name']} |"))
        assert "missing: waits for its scored rows" in row
    assert "0 of 6 ladder models are in" in md
    for name in ("Llama-3.1-70B-Instruct", "Ministral 3 8B Instruct 2512", "Ministral 3 14B Instruct 2512"):
        assert f"waits for {name}" in md
    j = json.loads((tmp_path / "out" / "ladder.json").read_text(encoding="utf-8"))
    assert all(v == "missing" for v in j["status"].values())


def test_partial_model_lists_its_missing_benchmarks(tmp_path):
    root = panel_root(tmp_path)
    src = ROOT / "data/frontier_panel/scored/math_scored.parquet"
    if not src.exists():
        pytest.skip("DeepSeek's MATH-500 rows are not on this machine")
    d = root / "data/processed/ladder/math"
    d.mkdir(parents=True)
    pd.read_parquet(src).assign(model_key=FAKE).to_parquet(d / f"{FAKE}.parquet", index=False)
    m = la.model_block(FAKE, root)
    assert m["status"] == "partial"
    assert m["tasks_missing"] == ["gsm8k", "bbh", "hotpot"]
    assert m["tasks"]["math"]["readings_missing"] == ["uncut"]
    assert any("uncut reading is absent" in e for e in m["edge_cases"])


@needs_ds
def test_fake_cluster_model_in_score_ladder_layout_is_added(tmp_path):
    root = panel_root(tmp_path)
    before = la.analyse(root, tmp_path / "out0", keys=[FAKE, "ministral-3-14b-instruct-2512"])
    assert before["models"][FAKE]["status"] == "missing"
    # score_ladder.py writes data/processed/ladder/<task>/<key>.parquet (cut) and ladder_uncut (uncut)
    for reading, sub, suffix in (("cut", "ladder", ""), ("uncut", "ladder_uncut", "_uncut")):
        d = root / "data/processed" / sub / "math"
        d.mkdir(parents=True, exist_ok=True)
        x = pd.read_parquet(ROOT / f"data/frontier_panel/scored/math_scored{suffix}.parquet")
        x = x.assign(model_key=FAKE, run_key=FAKE, reading=reading)
        x["item_id"] = x["item_id"].astype(str)
        x.to_parquet(d / f"{FAKE}.parquet", index=False)
    R = la.analyse(root, tmp_path / "out1", keys=[FAKE, "ministral-3-14b-instruct-2512"])
    m = R["models"][FAKE]
    assert m["status"] == "partial" and list(m["tasks"]) == ["math"]
    assert R["models"]["ministral-3-14b-instruct-2512"]["status"] == "missing"
    # the fake model's rows are DeepSeek's MATH-500 rows, so its numbers are the Nebius report's
    neb = json.loads(NEBIUS.read_text(encoding="utf-8"))
    for reading, src in (("cut", "tasks"), ("uncut", "tasks_uncut")):
        mine, ref = m["tasks"]["math"][reading], neb[src]["math"]
        for sc in ("strict", "content"):
            for lab in ("all_draws", "discovery"):
                for a in ("requested", "forced", "cot"):
                    assert mine[sc][lab]["accuracy_pp"][a] == pytest.approx(ref[sc][lab]["accuracy_pp"][a], abs=1e-9)
                assert mine[sc][lab]["corrected_gain_pp"] == pytest.approx(ref[sc][lab]["corrected_gain_pp"], abs=1e-9)
        assert mine["w"]["discovery"] == pytest.approx(ref["w"]["discovery"], abs=1e-9)
    # it is placed against the panel and enters the Mistral 3 family table; the 14B rung still waits
    assert R["placement"][FAKE]["math"]["cut"]["strict"]["cot_minus_forced"]["panel_n"] == 29
    rung = next(r for r in R["families"]["Mistral 3"]["rungs"] if r["key"] == FAKE)
    assert rung["present"] and "math" in rung["values"]
    md = (tmp_path / "out1" / "ladder.md").read_text(encoding="utf-8")
    assert "| Ministral 3 8B Instruct 2512 | MATH-500 | strict |" in md
    assert "waits for Ministral 3 14B Instruct 2512" in md
    assert any(line.startswith("| Ministral 3 8B Instruct 2512 |") and "partial" in line for line in md.splitlines())


def test_place_counts_and_range():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    p = la.place(2.0, s)
    assert (p["below"], p["equal"], p["above"]) == (1, 1, 2)
    assert p["outside_panel_range"] is None
    assert la.place(5.0, s)["outside_panel_range"] == "above"
    assert la.place(None, s)["below"] is None


# ----------------------------------------------------------------------------------------------------------------
# Six models in (synthetic rows), completeness, and the independent checker (scripts/verify_ladder_analysis.py)
# ----------------------------------------------------------------------------------------------------------------

import numpy as np  # noqa: E402

import verify_ladder_analysis as vla  # noqa: E402

FS_REQ_TASKS = ("gsm8k", "bbh")
LLAMA = "llama-3.1-70b-instruct-fp8"


def fake_rows(task: str, seed: int, items: int = 3) -> pd.DataFrame:
    """A small scored table in the ladder's layout: every sampler configuration, `items` items, the three zero-shot
    arms and five-shot forced; five-shot requested on GSM8K and BIG-Bench Hard (two configurations, '<task>/' prefix)."""
    rng = np.random.default_rng(seed)
    s = json.loads((ROOT / f"manifests/sampler_{task}.json").read_text(encoding="utf-8"))
    configs = sorted(set(map(str, s["discovery_config_ids"] + s["validation_config_ids"])))
    rows = []
    arms = ["direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced"]
    for c in configs:
        for i in range(items):
            for a in arms:
                rows.append((f"{c}|{a}|item{i}", a, c, f"item{i}"))
    if task in FS_REQ_TASKS:
        for c in configs[:2]:
            for i in range(items):
                rows.append((f"{task}/{c}|fewshot_requested|item{i}", "fewshot_requested", f"{task}/{c}", f"item{i}"))
    df = pd.DataFrame(rows, columns=["request_id", "arm", "config_id", "item_id"])
    n = len(df)
    df["strict_correct"] = rng.integers(0, 2, n)
    df["content_correct"] = np.maximum(df["strict_correct"], rng.integers(0, 2, n))
    df["compliant"] = rng.integers(0, 2, n).astype(bool)
    df["n_output_tokens"] = rng.integers(1, 40, n)
    df["finish_reason"] = "stop"
    df["cut_at_panel_cap"] = rng.random(n) < 0.05
    return df


# The records the text cites: sizes, spend, the panel's precision, the forced tests, the cut-equivalence
# check, the GPU runs' kernels and the Qwen pilot check. The F1 source checks are left out: their numbers are the
# real runs', which the synthetic rows do not reproduce (test_footnote_source_is_checked_against_the_rows covers them).
RECORDS = ("manifests/dataset2_inventory.md", "manifests/ladder_published_sizes.json",
           "config/models.yaml", "config/models_extension.yaml", "config/models_ladder.yaml",
           "results/ladder/mistral_pilot.md", "results/ladder/mistral_forced_test.md",
           "results/frontier_panel/nebius_panel.md", "results/frontier_panel/nebius_prefill_test.md",
           "results/frontier_panel/nebius_realtime_spend.md", "results/ladder/equiv_*.json", "results/ladder/scored_*.json",
           "results/ladder/pilot_check_*.json", "results/ladder/pilot_check_*.md")


def six_model_root(tmp: Path) -> Path:
    """Panel files, the samplers and the ladder decoding files from the repository, plus synthetic scored rows for all
    six ladder models in their layouts and request files with the same ids."""
    root = panel_root(tmp)
    for pat in ("manifests/sampler_*.json", "config/decoding_ladder2_*.yaml") + RECORDS:
        for p in ROOT.glob(pat):
            (root / p.parent.relative_to(ROOT)).mkdir(parents=True, exist_ok=True)
            shutil.copy(p, root / p.relative_to(ROOT))
    design = {t: fake_rows(t, seed=100 + la.TASKS.index(t))[la.ID_COLS] for t in la.TASKS}
    for j, key in enumerate(la.LADDER):
        for t in la.TASKS:
            df = fake_rows(t, seed=17 * j + la.TASKS.index(t))
            df[la.ID_COLS] = design[t].to_numpy()
            uncut = df.copy()
            uncut["strict_correct"] = np.where(uncut["cut_at_panel_cap"], 1, uncut["strict_correct"])
            uncut["content_correct"] = np.maximum(uncut["content_correct"], uncut["strict_correct"])
            for reading, frame in (("cut", df), ("uncut", uncut)):
                p = la.model_paths(key, root)[reading][t]
                p.parent.mkdir(parents=True, exist_ok=True)
                frame.assign(model_key=key, reading=reading).to_parquet(p, index=False)
            src = key if la.LADDER[key]["where"] == "cluster" else la.API_DESIGN_KEY
            rq = root / la.REQ_PATH.format(task=t, key=src)
            rq.parent.mkdir(parents=True, exist_ok=True)
            design[t].assign(model_key=src, messages_json="[]").to_parquet(rq, index=False)
    write_fewshot_panel(root)
    return root


def fs_gap(root: Path, key: str) -> Fraction:
    """A ladder model's BIG-Bench Hard five-shot forced gap, content - strict over all rows of the arm, exact."""
    x = pd.read_parquet(la.model_paths(key, root)["cut"]["bbh"])
    g = x[x["arm"] == "fewshot_forced"]
    return Fraction(100 * (int(g["content_correct"].sum()) - int(g["strict_correct"].sum())), len(g))


def write_fewshot_panel(root: Path, caps: dict | None = None, scorer_shift: Fraction = Fraction(0)) -> dict:
    """A synthetic results/ladder/mistral_bbh_forced_check.json in the layout of scripts/check_mistral_bbh_forced.py:
    28 panel models' five-shot forced strict and content over all 47,250 rows, to 0.01 pp; one panel gap 0.01 pp or
    less from Ministral 3 8B's (so its count is not settled by the file); the file's own scoring of Mistral Large 3's
    arm (its synthetic rows' strict and content to 0.01 pp, moved by scorer_shift); and one panel run log of the arm
    with the given decoding (the ladder's by default)."""
    near = round(fs_gap(root, FAKE), 2)                           # half-even at 0.01 pp: within 0.005 of the gap
    gaps = [Fraction(45 * i, 100) for i in range(27)] + [near]
    scan = {}
    for i, gp in enumerate(gaps):
        strict = Fraction(4000 + i, 100)
        scan[f"fewshot_forced|p{i:02d}"] = {"arm": "fewshot_forced", "model": f"p{i:02d}", "n": 47250,
                                            "strict": float(strict), "content": float(strict + gp)}
        scan[f"forced_direct|p{i:02d}"] = {"arm": "forced_direct", "model": f"p{i:02d}", "n": 47250,
                                           "strict": 10.0, "content": 20.0}
    p = root / la.FEWSHOT_PANEL
    p.parent.mkdir(parents=True, exist_ok=True)
    ml = pd.read_parquet(la.model_paths("mistral-large-2512", root)["cut"]["bbh"])
    ml = ml[ml["arm"] == "fewshot_forced"]
    own = {f"{s_}_pipeline": float(round(Fraction(100 * int(ml[f"{s_}_correct"].sum()), len(ml)) + scorer_shift, 2))
           for s_ in ("strict", "content")}
    p.write_text(json.dumps({"tasks": {"bbh": {"pipeline_matches_saved": True,
                                               "fewshot_forced": dict(n=len(ml), **own)}},
                             "panel_scan": scan}), encoding="utf-8")
    log = root / "data/d2_out/fewshot/bbh/p00/run_log-x.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"decoding": caps or {"max_new_tokens": {"forced_direct": 16}, "stop": ["}"],
                                                    "prefill": {"forced_direct": "\\boxed{"}}}), encoding="utf-8")
    return {f"p{i:02d}": gp for i, gp in enumerate(gaps)}


@pytest.fixture(scope="module")
def six(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("six")
    root = six_model_root(tmp)
    R = la.analyse(root, tmp / "out")
    return root, tmp / "out", R


def test_six_models_in(six):
    root, out, R = six
    assert all(m["status"] == "complete" for m in R["models"].values())
    assert R["findings"][0].startswith("6 of 6 ladder models are in")
    for name, f in R["families"].items():
        assert f["rungs_present"] == f["rungs_total"], name
    assert R["families"]["Llama 3.1"]["rungs_total"] == 2
    assert all(c["exact"] for c in R["completeness"].values())
    assert all(v.get("design_equals_api_design", True) for c in R["completeness"].values() for v in c["tasks"].values())
    md = (out / "ladder.md").read_text(encoding="utf-8")
    assert "### Llama 3.1: 8B → 70B (2 of 2 rungs in)" in md
    assert "| Quantity | 8B (panel) | 70B (ladder) | Over size (strict / content) |" in md
    assert "Shortfalls among the models with scored rows (arm × configuration cells, missing arms): none." in md
    assert "Llama 3.1 ladder: 1 of 2 rungs" not in md
    for k in la.LADDER:
        assert any(line.startswith(f"| {la.LADDER[k]['name']} |") and "| complete |" in line for line in md.splitlines())
        assert f"waits for {la.LADDER[k]['name']}" not in md
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    assert "Llama 3.1 ladder (8B → 70B; 2 of 2 rungs" in wn
    assert (out / "ladder_note_clean.md").exists() and (out / "ladder_clean.md").exists()


def test_checker_agrees_on_the_fake_set(six):
    """Every value in ladder.json and every printed number, word and mark agree, the 48-draw corrected gain included:
    both scripts define it as rules.corrected_gain_fields states, (1 - w over the 48 draws) x the 48-draw gain."""
    root, out, R = six
    lines = []
    n, m = vla.run(root, out / "ladder.json", None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    assert m == 0, mm[:10]
    assert n > 5000
    txt = next(x for x in lines if x.startswith("Printed text"))
    assert int(txt.split(": ")[1].split()[0]) > 1500          # the printed text was parsed and checked
    # the line also says whether the ground-truth stage ran (no raw replies under a synthetic root)
    assert lines[-1] == f"VERIFY: {n} values checked, {m} mismatches; ground truth not run (no raw replies)"
    assert any(x.startswith("GROUND TRUTH: not run") for x in lines)
    assert any(x.startswith("Corrected gain, as src/baseline_mixture.RULE_TEXT states it") for x in lines)
    J = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
    assert "all_draws.corrected_gain_pp" in J["rules"]["corrected_gain_fields"]
    for k, mo in R["models"].items():
        for t, tb in mo["tasks"].items():
            for r in la.READINGS:
                for sc in la.SCORES:
                    b, w = tb[r][sc], tb[r]["w"]
                    assert b["all_draws"]["corrected_gain_pp"] == pytest.approx(
                        (1 - w["all_draws"]) * b["all_draws"]["gain_over_forced_pp"], abs=1e-12)
                    assert b["discovery"]["corrected_gain_pp"] == pytest.approx(
                        (1 - w["discovery"]) * b["discovery"]["gain_over_forced_pp"], abs=1e-12)
                    assert b["pooled"]["corrected_gain_pp"] == pytest.approx(
                        (1 - w["discovery"]) * b["pooled"]["gain_over_forced_pp"], abs=1e-12)


def test_checker_flags_changed_values(six, tmp_path):
    root, out, _ = six
    J = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
    J["models"]["mistral-large-2512"]["tasks"]["math"]["cut"]["content"]["all_draws"]["accuracy_pp"]["forced"] += 1e-6
    J["models"][LLAMA]["tasks"]["bbh"]["uncut"]["w"]["discovery"] += 0.01
    J["families"]["Llama 3.1"]["rungs"][1]["values"]["hotpot"]["content"]["cot_minus_forced"] -= 0.5
    bad = tmp_path / "ladder.json"
    bad.write_text(json.dumps(J), encoding="utf-8")
    lines = []
    _, m = vla.run(root, bad, None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    assert m == len(mm) >= 3
    assert any("mistral-large-2512/math/cut/content/all_draws/forced" in x for x in mm)
    assert any(f"{LLAMA}/bbh/uncut/w/discovery" in x for x in mm)
    assert any(f"family/Llama 3.1/{LLAMA}/hotpot/content/cot_minus_forced" in x for x in mm)


def test_checker_is_written_apart_from_the_analysis():
    src = (ROOT / "scripts/verify_ladder_analysis.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if line.startswith(("import ", "from ")))
    for name in ("ladder_analysis", "nebius_panel_report", "nebius_panel", "src"):
        assert name not in code, name


def test_checker_reads_nebius_md_tables():
    text = ("## Main numbers\n\n| Benchmark | Score | Requested | Forced |\n|---|---|---|---|\n| GSM8K | strict | 1.0 | 2.0 |\n"
            "\nUncut\n\n| Benchmark | Score | Requested | Forced |\n|---|---|---|---|\n| GSM8K | strict | 1.5 | 2.0 |\n")
    tabs = vla.md_tables(text)
    assert [t[1][:3] for t in tabs] == [["Benchmark", "Score", "Requested"]] * 2
    assert tabs[1][2] == [["GSM8K", "strict", "1.5", "2.0"]]
    assert vla.fmt(-0.04, 1) == "-0.0" and vla.fmt(None, 1) == "–"


def test_completeness_lists_shortfalls_and_duplicates(tmp_path):
    root = six_model_root(tmp_path)
    key = "ministral-3-14b-instruct-2512"
    p = la.model_paths(key, root)["cut"]["math"]
    x = pd.read_parquet(p)
    dropped, twice = x.iloc[:2], x.iloc[[-1]]
    x = pd.concat([x.iloc[2:], twice], ignore_index=True)       # two rows missing, one id twice (another cell)
    x.to_parquet(p, index=False)
    c = la.completeness([key], root)[key]
    m = c["tasks"]["math"]["cut"]
    assert m["missing_ids"] == 2 and m["duplicate_ids"] == 1 and m["extra_ids"] == 0
    assert set(m["missing_id_list"]) == set(dropped["request_id"])
    assert not c["exact"] and c["state"].startswith("short:")
    cells = {(s["arm"], s["config_id"]): (s["scored"], s["design"]) for s in m["cells_short"]}
    for a, cf in zip(dropped["arm"], dropped["config_id"]):
        assert cells[(a, cf)][0] == cells[(a, cf)][1] - 1
    assert cells[(twice["arm"].iloc[0], twice["config_id"].iloc[0])][0] == cells[(twice["arm"].iloc[0],
                                                                                   twice["config_id"].iloc[0])][1] + 1
    assert c["tasks"]["math"]["uncut"]["missing_ids"] == 0
    R = la.analyse(root, tmp_path / "out", keys=[key])
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    assert "Shortfalls among the models with scored rows (arm × configuration cells, missing arms): none." not in md
    assert "Completeness: 0 of 1 models in have every one" in md
    assert R["models"][key]["status"] == "complete"          # every benchmark is there; the shortfall is listed


def test_missing_cluster_model_shows_the_loop_state(tmp_path):
    root = panel_root(tmp_path)
    log = root / la.LOOP_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("== pass 2026-11-24 00:46:28\nDL x complete;M llama-3.1-70b-instruct-fp8 "
                   "chosen=llama-3.1-70b-instruct-fp8 jobs_done=12/15;SLOTS 0 5\n== pass 2026-11-24 01:01:52\n"
                   "M llama-3.1-70b-instruct-fp8 chosen=llama-3.1-70b-instruct-fp8 jobs_done=13/15;\n", encoding="utf-8")
    st = la.loop_state(LLAMA, root)
    assert st["jobs_done"] == "13/15" and st["pass"] == "2026-11-24 01:01:52"
    R = la.analyse(root, tmp_path / "out", keys=[LLAMA])
    assert R["models"][LLAMA]["status"] == "missing"
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    assert "0 of 1 ladder models are in." in md
    assert "jobs done 13/15" in md


# ----------------------------------------------------------------------------------------------------------------
# Bases, directions, marks, footnotes, and the checker's text part
# ----------------------------------------------------------------------------------------------------------------

def plant(path: Path, old: str, new: str) -> None:
    s = path.read_text(encoding="utf-8")
    assert old in s, old
    path.write_text(s.replace(old, new, 1), encoding="utf-8")


def test_checker_flags_printed_errors(six, tmp_path):
    """Planted errors in the printed text, one of each kind, are each flagged: a number in a finding, a glance-table
    mark, a direction word of a family table, a number in ladder_note.md, a number nothing accounts for."""
    import re
    root, out, R = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    md = (d / "ladder.md").read_text(encoding="utf-8")
    # a finding number: the count of ladder models in
    plant(d / "ladder.md", "- 6 of 6 ladder models are in", "- 5 of 6 ladder models are in")
    # a glance-table cell: drop the first mark of section 1, or add one where none is due
    sec1 = md.split("## 2.")[0]
    cells = re.findall(r"\| (-?\d+\.\d+ \(\d+/\d+(?:, \d+ tied)?\)( ↑| ↓)?) \|", sec1)
    marked = [c for c, mk in cells if mk]
    if marked:
        plant(d / "ladder.md", f"| {marked[0]} |", f"| {marked[0][:-2]} |")
    else:
        plant(d / "ladder.md", f"| {cells[0][0]} |", f"| {cells[0][0]} ↑ |")
    # a direction word in a section 5 table
    m = re.search(r"\| (rises|falls|flat|(?:up|down|flat)(?:, (?:up|down|flat))+) / ", md.split("## 5.")[1])
    word = m.group(1)
    new = "falls" if word != "falls" else "rises"
    plant(d / "ladder.md", f"| {word} / ", f"| {new} / ")
    # a stray number in a finding with no number of its own
    plant(d / "ladder.md", "DeepSeek V4.1-Flash-1017 is a single model outside the three families,",
          "DeepSeek V4.1-Flash-1017 is a single model outside the three families (7 pp),")
    # a number in ladder_note.md: the first headline w
    wn = (d / "ladder_note.md").read_text(encoding="utf-8")
    row = next(x for x in wn.splitlines() if x.startswith("| Ministral 3 8B Instruct 2512 | GSM8K |"))
    w = row.split(" | ")[8]                       # after the discovery-draw CoT − forced column
    plant(d / "ladder_note.md", row, row.replace(f" | {w} | ", f" | {float(w) + 0.01:.3f} | ", 1))
    lines = []
    _, n_bad = vla.run(root, d / "ladder.json", None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    assert any("models in" in x and "printed 5" in x for x in mm), mm
    assert any(": mark: " in x for x in mm), mm
    assert any(" direction: " in x and f"printed '{new} / " in x for x in mm), mm
    assert any("number not checked: '7'" in x for x in mm), mm
    assert any(x.startswith("MISMATCH ladder_note.md") and " w: " in x for x in mm), mm
    assert n_bad == len(mm) >= 5


def test_direction_rule_flat_steps():
    """A step under 0.1 pp (w: 0.001) is flat; the analysis and the checker read the rule alike."""
    cases = [([89.598765, 92.808642, 92.827160], "pp", "up, flat"), ([1.0, 2.0], "pp", "rises"),
             ([1.0, 1.05], "pp", "flat"), ([5.0, 3.0, 1.0], "pp", "falls"), ([2.0, 3.0, 1.0], "pp", "up, down"),
             ([0.335, 0.015, 0.0151], "w", "down, flat"), ([0.001, 0.0025], "w", "rises"), ([1.0], "pp", None)]
    for vals, kind, want in cases:
        assert la.direction_of(vals, kind) == want, vals
        assert vla.direction(vals, kind) == want, vals


def test_readable_rows_carry_marks():
    """The readable rows of section 6 carry the same mark as every other row."""
    s = pd.Series([20.0, 30.0, 40.0])
    k, t = "qwen3.5-122b-a10b-fp8", "gsm8k"
    cut = {sc: {q: dict(la.place(25.0, s), basis="discovery") for q in la.QUANTITIES + ("residual",)}
           for sc in la.SCORES}
    cut["readable"] = {"requested": dict(la.place(10.0, s), basis="discovery"),
                       "forced": dict(la.place(45.0, s), basis="discovery"),
                       "cot": dict(la.place(25.0, s), basis="discovery")}
    R = {"placement": {k: {t: {"cut": cut}}}}
    lines = la.placement_table(R, t, [k], ("strict", "content"), True)
    row = {x.split(" | ")[1]: x for x in lines if x.startswith("| – |")}
    assert row["Readable, requested (%)"].endswith("| 10.0 (0/3) ↓ |")
    assert row["Readable, forced (%)"].endswith("| 45.0 (3/3) ↑ |")
    assert row["Readable, cot (%)"].endswith("| 25.0 (1/3) |")
    assert la.cell(cut["readable"]["requested"]) == "10.0 (0/3) ↓"


def test_findings_name_one_basis(six):
    """Each finding sentence with numbers names its basis; HotpotQA content (pooled rows) never
    shares a sentence or a glance-table column with a discovery-draw value."""
    root, out, R = six
    md = (out / "ladder.md").read_text(encoding="utf-8")
    sec1 = md.split("## 1. Findings")[1].split("## 2.")[0]
    for f in R["findings"]:
        if " against the panel, " in f:
            disc, hot = f.split(" HotpotQA, ")
            assert "discovery draws" in disc and "pooled" not in disc
            assert hot.startswith("CoT − forced (content), pooled rows:")
        if f.startswith("Requested replies the strict rule cannot read"):
            assert f.startswith("Requested replies the strict rule cannot read, discovery draws")
        if f.startswith("BIG-Bench Hard zero-shot forced arm"):
            assert "discovery draws" in f and "all rows" not in f
        if f.startswith("BIG-Bench Hard five-shot forced arm"):
            assert "all rows of the arm" in f and "discovery" not in f
        if " ladder (" in f:
            disc, hot = f.split(". HotpotQA, content, pooled rows: ")
            assert "content, discovery draws:" in disc and "HotpotQA" not in disc
    tabs = [(ctx, h, rows) for ctx, h, rows in vla.md_tables(sec1)]
    assert len(tabs) == 2
    (ctx_a, head_a, rows_a), (ctx_b, head_b, rows_b) = tabs
    assert "every column on the discovery draws" in ctx_a and head_a[1] == "Benchmark"
    assert {r[1] for r in rows_a} <= {"GSM8K", "MATH-500", "BIG-Bench Hard"}
    assert all(c.endswith("(discovery draws)") or c.endswith("(pooled rows)") for c in head_b[1:])
    assert all("content" in c for c in head_b[1:] if c.endswith("(pooled rows)"))
    J = json.loads((out / "ladder.json").read_text(encoding="utf-8"))
    for k, mo in J["models"].items():
        g_ = mo["tasks"]["bbh"]["cut"]
        assert g_["strict_content_gap_discovery_pp"]["requested"] == pytest.approx(
            g_["content"]["discovery"]["accuracy_pp"]["requested"] - g_["strict"]["discovery"]["accuracy_pp"]["requested"])


def test_hotpot_content_corrected_gain_is_labelled_pooled(six):
    """The only printed corrected gains not on the discovery draws are HotpotQA content cells, and
    those are labelled with the pooled product."""
    root, out, R = six
    md = (out / "ladder.md").read_text(encoding="utf-8")
    sec5 = md.split("## 5.")[1].split("## 6.")[0]
    sec6 = md.split("## 6.")[1].split("## 7.")[0]
    for sec in (sec5, sec6):
        assert f"| {la.POOLED_CG} |" in sec
        for part in sec.split("HotpotQA, content (pooled rows)")[:1]:
            assert la.POOLED_CG not in part.split("Over size:")[-1] if sec is sec5 else la.POOLED_CG not in part
    assert "`all_draws.corrected_gain_pp`" in md and "printed nowhere" in md


MISTRAL_CHECK_JSON = "results/ladder/mistral_bbh_forced_check.json"
NEBIUS_CHECK_JSON = "results/frontier_panel/nebius_forced_check.json"
NEBIUS_CHECK_MD = "results/frontier_panel/nebius_forced_check_clean.md"


@pytest.mark.skipif(not all((ROOT / rel).exists() for rel in (MISTRAL_CHECK_JSON, NEBIUS_CHECK_JSON, NEBIUS_CHECK_MD)),
                    reason="the source checks of footnotes F1 and F2 are not on this machine")
def test_footnote_source_is_checked_against_the_rows(tmp_path):
    """F1 and F2 quote their source checks; the checker reads the sources and flags a footnote
    whose source no longer describes the rows (here: synthetic rows, so F1's gaps from the source cannot match).
    F1's numbers are computed from the two checks' JSON (gap by form), not copied from their reports."""
    root = six_model_root(tmp_path)
    # the synthetic five-shot panel file is the Mistral check's JSON; give it the real check's gap and forms
    p = root / MISTRAL_CHECK_JSON
    syn, real = json.loads(p.read_text(encoding="utf-8")), json.loads((ROOT / MISTRAL_CHECK_JSON).read_text("utf-8"))
    p.write_text(json.dumps(dict(syn, gap=real["gap"], forms_all=real["forms_all"])), encoding="utf-8")
    for rel in (NEBIUS_CHECK_JSON, NEBIUS_CHECK_MD):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    R = la.analyse(root, tmp_path / "out")
    f1, f2 = R["footnotes"][0], R["footnotes"][1]
    rg, ng = real["gap"], json.loads((ROOT / NEBIUS_CHECK_JSON).read_text("utf-8"))["gap"]
    share = lambda g, form, arm: la.fnum(Fraction(100 * g["by_form"][form][arm], g["n_arm"][arm]))  # noqa: E731
    assert f"in {la.fnum(real['forms_all']['fewshot_forced']['bs_letter'])}% of five-shot forced replies" in f1
    assert f"accounts for {share(rg, 'letter with backslash (' + chr(92) + 'C)', 'fewshot_forced')} of the " in f1
    assert f"or the option's text ({share(ng, 'option text, no letter', 'forced_direct')} pp)" in f1
    assert "a LaTeX text wrapper" in f1 and "src/forced_stop.py" in f1
    assert "5 of 5 pairs in each of three separate tests" in f2 and "those 35 same-cap pairs gave the same answer" in f2
    assert "gaps of 0.4 pp or less" not in f1 and "GSM8K, MATH-500 and HotpotQA, computed here" in f1
    lines = []
    _, m = vla.run(root, tmp_path / "out" / "ladder.json", None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    assert any("F1 Mistral five-shot gap from the rows" in x for x in mm), mm
    assert any("F1 DeepSeek zero-shot gap from the rows" in x for x in mm), mm
    assert not any("F2" in x for x in mm), [x for x in mm if "F2" in x]          # F2 agrees with its source
    from_json = [x for x in mm if "F1" in x and "from the rows" not in x]
    assert not from_json, from_json                                              # F1 quotes its sources exactly


def test_footnote_leaves_out_a_check_without_gap_by_form(tmp_path):
    """A check JSON that holds no gap by form (the synthetic five-shot panel file) gives no Mistral clause in F1, and
    the checker does not ask for one."""
    root = six_model_root(tmp_path)
    R = la.analyse(root, tmp_path / "out")
    assert "Mistral Large 3 (`" not in R["footnotes"][0]
    lines = []
    vla.run(root, tmp_path / "out" / "ladder.json", None, out=lines.append)
    assert not any(x.startswith("MISMATCH") and "F1" in x for x in lines), [x for x in lines if "F1" in x]


# ----------------------------------------------------------------------------------------------------------------
# Half-even on exact values, the checker's strict rule, and
# the BIG-Bench Hard five-shot forced gap placed on the panel's all-rows values
# ----------------------------------------------------------------------------------------------------------------

R2_CELLS = [(Fraction(179, 20), 1, "9.0", "8.9"),       # DeepSeek HotpotQA, pooled content, CoT − forced (8.95)
            (Fraction(177, 20), 1, "8.8", "8.9"),       # DeepSeek HotpotQA, discovery strict, CoT − forced (8.85)
            (Fraction(1283, 20), 1, "64.2", "64.1"),    # Qwen3.5-122B-A10B HotpotQA, pooled content, CoT (64.15)
            (Fraction(1267, 20), 1, "63.4", "63.3"),    # DeepSeek HotpotQA, pooled content, CoT (63.35)
            (Fraction(13, 2000), 3, "0.006", "0.007"),  # Llama-3.1-70B HotpotQA, discovery w (0.0065)
            (Fraction(1, 20), 1, "0.0", "0.1"),         # Ministral 3 8B HotpotQA, pooled content, CoT − requested
            (Fraction(111, 40), 2, "2.78", "2.77")]     # median of the panel's five-shot gaps (2.775)


def test_half_even_on_the_grid():
    """The analysis rounds half-even on the exact value, in its own integer arithmetic; it agrees with Python's
    formatting of the exact fraction on every point of the grids the values live on, and a halfway point prints as its
    even neighbour."""
    for q, d, right, wrong in R2_CELLS:
        assert la.fnum(q, d) == right != wrong
    for den, d in ((240, 1), (360, 1), (24000, 3), (480, 1), (40, 2), (7914, 1), (2388, 2)):
        for m in range(-3 * den, 3 * den + 1):
            q = Fraction(m, den)
            got = la.fnum(q, d)
            assert got == format(q, f".{d}f"), (q, d, got)
            if (q * 10 ** d).denominator == 2:                                    # exactly halfway
                assert int(got.replace("-", "").replace(".", "")) % 2 == 0, (q, got)
    assert la.fnum(Fraction(-1, 25)) == "-0.0" and la.fnum(Fraction(-177, 20)) == "-8.8"
    # a template SD: the exact root of an exact variance (Qwen3.5-122B-A10B HotpotQA forced, 0.35 exactly)
    assert la.fnum(la.Root(Fraction(49, 400))) == "0.4" and la.fnum(la.Root(Fraction(9, 400))) == "0.2"
    assert la.fnum(la.Root(Fraction(2))) == "1.4" and la.fnum(la.Root(Fraction(1225, 100)), 2) == "3.50"
    # a float on a boundary is printed from its binary value and listed, never decided silently
    la.ROUNDING_EDGE.clear()
    assert la.fnum(8.950000000000003) == "9.0" and la.fnum(8.849999999999994) == "8.8"
    assert len(la.ROUNDING_EDGE) == 2
    la.ROUNDING_EDGE.clear()
    assert la.fnum(8.94) == "8.9" and la.ROUNDING_EDGE == []


def test_panel_values_are_read_on_their_grid():
    """A panel file's float is read as the ratio m/G on the design's grid; off the grid it is not an exact ratio."""
    assert la.on_grid(64.15000000000001, Fraction(360)) == Fraction(1283, 20)        # pooled HotpotQA, 1/360 pp
    assert la.on_grid(8.849999999999994, Fraction(240)) == Fraction(177, 20)         # discovery means, 1/240 pp
    assert la.on_grid(0.006500000000000001, Fraction(24000)) == Fraction(13, 2000)    # discovery w, 1/24,000
    assert la.on_grid(8.851, Fraction(240)) is None and la.on_grid(float("nan"), Fraction(240)) is None
    assert vla.exact_of(64.15000000000001, 36000) == Fraction(1283, 20)
    assert vla.exact_of(8.849999999999994, 24 * 1000) == Fraction(177, 20)
    p = la.place(None, pd.Series([Fraction("2.77"), Fraction("2.78"), None], dtype=object))
    assert p["panel_median"] == Fraction(111, 40) and la.fnum(p["panel_median"], 2) == "2.78"
    assert vla.place(None, [Fraction("2.77"), Fraction("2.78")])["median"] == Fraction(111, 40)


def test_checker_fails_the_wrong_halfway_neighbour():
    """The checker compares the printed string with the half-even formatting of the exact value;
    the other neighbour of a halfway value fails, and so does a value it holds only as a float on a boundary."""
    for q, d, right, wrong in R2_CELLS:
        T = vla.Tally()
        T.printed("cell", right, q)
        assert T.bad == [], (q, right)
        T.printed("cell", wrong, q)
        assert len(T.bad) == 1 and f"printed {wrong}" in T.bad[0], T.bad
    T = vla.Tally()
    T.printed("float", "8.9", 8.950000000000003)
    T.printed("float", "9.0", 8.950000000000003)
    assert len(T.bad) == 2 and all("halfway" in x for x in T.bad)
    assert not hasattr(T, "ties")
    X = vla.TextCheck(T, None)
    n0 = len(T.bad)
    X.cell("glance", "8.9 (29/29) ↑", {"value": Fraction(179, 20), "below": 29, "n": 29, "equal": 0, "mark": " ↑"})
    X.cell("glance", "9.0 (29/29) ↑", {"value": Fraction(179, 20), "below": 29, "n": 29, "equal": 0, "mark": " ↑"})
    assert len(T.bad) == n0 + 1 and "printed 8.9" in T.bad[-1]
    src = (ROOT / "scripts/verify_ladder_analysis.py").read_text(encoding="utf-8")
    assert "TIE" not in src and "either neighbour" not in src


def test_fewshot_bbh_placed_on_the_panel(six):
    """The five-shot forced gap over all rows of the arm is placed, with the same cell and marks,
    against the panel's gaps on that basis (the file's 0.01 pp values), the basis checked (rows, scorer, caps); a value
    within the file's rounding of a panel gap is listed as not settled; the checker agrees."""
    root, out, R = six
    gaps = {m: Fraction(str(v["content"])) - Fraction(str(v["strict"]))
            for v in json.loads((root / la.FEWSHOT_PANEL).read_text(encoding="utf-8"))["panel_scan"].values()
            if v["arm"] == "fewshot_forced" for m in [v["model"]]}
    fp = R["bbh_forced_gap"]["fewshot_panel"]
    assert fp["check"]["rows_ok"] and fp["check"]["scorer_ok"] and fp["check"]["caps_ok"]
    assert fp["n"] == 28 and fp["min"] == min(gaps.values()) and fp["max"] == max(gaps.values())
    md = (out / "ladder.md").read_text(encoding="utf-8")
    line = next(x for x in md.splitlines() if x.startswith("- BIG-Bench Hard five-shot forced arm"))
    assert "against the panel's gaps on that basis (the same rows, scorer and caps;" in line
    assert "the panel's files have no value" not in line
    for k in R["models"]:
        v = fs_gap(root, k)
        p = R["bbh_forced_gap"]["models"][k]["fewshot_forced_all_rows_placement"]
        b = sum(1 for g_ in gaps.values() if g_ < v - Fraction(1, 10 ** 9))
        mark = " ↑" if v > max(gaps.values()) else (" ↓" if v < min(gaps.values()) else "")
        assert p["value"] == v and p["below"] == b
        assert f"{la.nm(k)} {la.fnum(v, 2)} ({b}/28){mark}" in line
    uns = R["bbh_forced_gap"]["models"][FAKE]["fewshot_within_panel_rounding"]
    assert uns and "so the count is not settled by the file: Ministral 3 8B Instruct 2512 (" in line
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    assert line[2:] in wn


def test_fewshot_basis_failure_is_stated(tmp_path):
    """A panel run log with other caps is a failed basis check, stated in the sentence and confirmed by the checker."""
    root = six_model_root(tmp_path)
    write_fewshot_panel(root, caps={"max_new_tokens": {"forced_direct": 32}, "stop": ["}"],
                                    "prefill": {"forced_direct": "\\boxed{"}})
    R = la.analyse(root, tmp_path / "out")
    assert R["bbh_forced_gap"]["fewshot_panel"]["check"]["caps_ok"] is False
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    assert "against the panel's gaps on that basis (basis check failed: caps;" in md
    lines = []
    _, m = vla.run(root, tmp_path / "out" / "ladder.json", None, out=lines.append)
    assert m == 0, [x for x in lines if x.startswith("MISMATCH")][:5]


def test_checker_reads_sections_7_to_9(six, tmp_path):
    """The checker parses the per-model tables of sections 7 and 8, the five-shot and readability tables and the cut
    table of section 9: a changed cell in each is flagged."""
    root, out, R = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    L = (d / "ladder.md").read_text(encoding="utf-8").split("\n")
    s7, s8, s9 = (L.index(next(x for x in L if x.startswith(f"## {n}."))) for n in (7, 8, 9))

    def bump(start, prefix, col):
        i = next(j for j in range(start, len(L)) if L[j].startswith(prefix))
        c = L[i].split(" | ")
        c[col] = la.fnum(Fraction(c[col].replace(",", "")) + Fraction(1, 10), 1) if "." in c[col] else \
            str(int(c[col].replace(",", "")) + 1)
        L[i] = " | ".join(c)
    bump(s7, "| Llama-3.1-70B-Instruct | MATH-500 | content |", 4)               # forced (48 draws)
    bump(s8, "| Qwen3.5-122B-A10B | GSM8K | strict |", 10)                       # corrected gain (discovery), uncut
    bump(s7, "| Mistral Large 3 (2512) | BIG-Bench Hard | fs_forced | cut |", 6)  # five-shot strict
    bump(s7 + 1 + next(j for j, x in enumerate(L[s7 + 1:]) if x.startswith("| Model | Benchmark | Readable")),
         "| DeepSeek V4.1-Flash-1017 | HotpotQA |", 3)                           # readable, forced
    bump(s9, "| Ministral 3 14B Instruct 2512 | MATH-500 | cot |", 4)            # cut at the panel's cap
    (d / "ladder.md").write_text("\n".join(L), encoding="utf-8")
    lines = []
    _, m = vla.run(root, d / "ladder.json", None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    for what in (f"{LLAMA} math content cut Forced (48 draws)", "qwen3.5-122b-a10b-fp8 gsm8k strict uncut Corrected gain",
                 "mistral-large-2512 bbh fs_forced cut strict", "deepseek-v4.1-flash-1017 hotpot readable forced",
                 "ministral-3-14b-instruct-2512 math cot cut at the panel's cap"):
        assert any(what in x for x in mm), (what, mm)
    assert m == len(mm) >= 5


def test_checker_flags_a_wrong_fewshot_cell(six, tmp_path):
    root, out, R = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    md = (d / "ladder.md").read_text(encoding="utf-8")
    line = next(x for x in md.splitlines() if x.startswith("- BIG-Bench Hard five-shot forced arm"))
    k = "qwen3.5-122b-a10b-fp8"
    p = R["bbh_forced_gap"]["models"][k]["fewshot_forced_all_rows_placement"]
    old = f"Qwen3.5-122B-A10B {la.cell(p, 2)}"
    assert old in line
    plant(d / "ladder.md", old, f"Qwen3.5-122B-A10B {la.fnum(p['value'], 2)} ({p['below'] + 1}/28)")
    lines = []
    _, m = vla.run(root, d / "ladder.json", None, out=lines.append)
    mm = [x for x in lines if x.startswith("MISMATCH")]
    assert any(f"{k} BIG-Bench Hard five-shot forced gap: below" in x for x in mm), mm


# ---------------------------------------------------------------------------------------------------------------------
# Sizes, spend, sources and the printed text
# ---------------------------------------------------------------------------------------------------------------------

def section(md: str, n: int) -> str:
    return md.split(f"\n## {n}.")[1].split("\n## ")[0]


def run_checker(root: Path, out: Path) -> list[str]:
    lines = []
    vla.run(root, out / "ladder.json", None, out=lines.append)
    return [x for x in lines if x.startswith("MISMATCH")]


def test_r3_item1_sizes_come_from_the_records(six):
    """Each model's size is read from a recorded source and printed with it; DeepSeek is a single model outside
    the families, and "top of the size range" appears only as the requested wording, set against the records."""
    root, out, R = six
    sz = R["sizes"]
    assert (sz["ministral-3-8b-instruct-2512"]["total"], sz["ministral-3-8b-instruct-2512"]["arch"]) == ("8.92", "dense")
    assert (sz["ministral-3-14b-instruct-2512"]["total"], sz["ministral-3-14b-instruct-2512"]["arch"]) == ("13.9", "dense")
    q = sz["qwen3.5-122b-a10b-fp8"]
    assert (q["total"], q["active"], q["arch"]) == ("125.1", "10", "mixture of experts")
    ml = sz["mistral-large-2512"]
    assert (ml["total"], ml["active"], ml["arch"], ml["published"]) == ("675", "41", "mixture of experts", True)
    assert ml["url"] == "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512" and ml["read"]
    d = sz[DS]
    assert (d["total"], d["active"], d["active_input"], d["active_output"]) == ("552", None, "8", "16")
    assert d["url"] == "https://api-docs.deepseek.com/news/news260910" and "1017 checkpoint" in d["caveat"]
    o = d["predecessor"]          # V4-Flash's size, kept for the record, not used
    assert (o["name"], o["total"], o["active"], o["used"]) == ("V4-Flash", "284", "13", False)
    md = (out / "ladder.md").read_text(encoding="utf-8")
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    for text in (md, wn):
        i = text.find("top of the size range")
        while i >= 0:
            assert text[:i].endswith(("is listed as the ", "was listed as the ")), text[i - 80:i + 30]
            i = text.find("top of the size range", i + 1)
    assert ("| DeepSeek V4.1-Flash-1017 | none (a single model outside the three families) | 552B total, 8B active "
            "for input and 16B for output, mixture of experts (DeepSeek's release note of V4.1-Flash") in md
    assert "| Mistral Large 3 (2512) | Mistral 3 | 675B total, 41B active, mixture of experts (Mistral's model card" in md
    for text in (md, wn):
        assert "which it replaces" not in text and "not recorded for this release" not in text
    assert ("| Qwen3.5-122B-A10B | Qwen3.5 | 125.1B total, 10B active, mixture of experts "
            "(`manifests/dataset2_inventory.md`, Table 1) |") in md
    line = next(x for x in md.splitlines() if x.startswith("- DeepSeek V4.1-Flash-1017 is a single model outside"))
    assert "its published size is 552B total, with 8B active for input and 16B for output (" in line
    assert "below Mistral Large 3 in total parameters (675B)" in line
    assert "below Llama-3.1-70B-Instruct in active parameters (70B, dense)" in line
    assert line.endswith("so the records do not support that.")
    assert "Size: 125.1B total, 10B active, mixture of experts." in wn


def test_r3_item1_checker_flags_a_wrong_size(six, tmp_path):
    root, out, _ = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    plant(d / "ladder.md", "| Ministral 3 14B Instruct 2512 | Mistral 3 | 13.9B, dense",
          "| Ministral 3 14B Instruct 2512 | Mistral 3 | 14B, dense")
    plant(d / "ladder_note.md", "Size: 125.1B total, 10B active", "Size: 122B total, 10B active")
    mm = run_checker(root, d)
    assert any(x.startswith("MISMATCH ladder.md") and "ministral-3-14b-instruct-2512 total parameters" in x
               for x in mm), mm
    assert any(x.startswith("MISMATCH ladder_note.md") and "qwen3.5-122b-a10b-fp8 total parameters" in x
               for x in mm), mm


def test_published_size_must_be_written_in_its_quotes(tmp_path):
    """A published number its quoted sentences do not write is refused by the analysis, and the
    checker's own reading leaves it unrecorded."""
    man = json.loads((ROOT / la.PUBLISHED_SIZES).read_text(encoding="utf-8"))
    man["models"][DS]["total"] = "553"
    p = tmp_path / la.PUBLISHED_SIZES
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="553B is not written"):
        la.recorded_size(tmp_path, DS)
    assert vla.size_facts(tmp_path, DS)["total"] is None
    assert la.recorded_size(ROOT, DS)["total"] == "552"
    assert vla.size_facts(ROOT, DS)["active_output"] == "16"


def test_checker_flags_a_wrong_published_size(six, tmp_path):
    root, out, _ = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    plant(d / "ladder.md", "| 552B total, 8B active for input", "| 552B total, 9B active for input")
    plant(d / "ladder_note.md", "its published size is 552B total", "its published size is 562B total")
    mm = run_checker(root, d)
    assert any(x.startswith("MISMATCH ladder.md") and "active_input parameters (published)" in x for x in mm), mm
    assert any(x.startswith("MISMATCH ladder_note.md") and "DeepSeek: its published size" in x for x in mm), mm


def fake_block(v) -> dict:
    """One reading of one benchmark on which la.ladder_value reads v for every quantity in pp and 0 for w."""
    b = {"accuracy_pp": {"requested": v, "forced": v, "cot": v}, "gain_over_requested_pp": v,
         "gain_over_forced_pp": v, "corrected_gain_pp": v}
    return {"w": {"discovery": Fraction(0)}, "strict": {"discovery": b, "pooled": b},
            "content": {"discovery": b, "pooled": b}}


def fake_place(models: dict, readable: bool = False) -> dict:
    """A placement dict with every placed quantity of both scores and both readings (the readable shares if asked),
    each placed alike: cut_vs_uncut takes its list of quantities from it."""
    pl = {"below": 0, "above": 0, "equal": 0, "panel_n": 1, "basis": "discovery", "outside_panel_range": None}
    out = {}
    for k, m in models.items():
        for t in m["tasks"]:
            for r in la.READINGS:
                d = {sc: {q: dict(pl) for q in la.PLACED if not (q == "w" and sc == "content")} for sc in la.SCORES}
                if readable:
                    d["readable"] = {a: dict(pl) for a in la.READ_ARMS}
                out.setdefault(k, {}).setdefault(t, {})[r] = d
    return out


def test_r3_item2_a_tie_is_not_a_reversal():
    """A pair tied exactly in one reading and ordered in the other is listed apart, in both scripts; only pairs
    that both readings order, in opposite ways, count as a change of order."""
    vals = (("a", 1, 2), ("b", 1, 1), ("c", 0, 3))
    models = {k: {"tasks": {"gsm8k": {"cut": fake_block(Fraction(c)), "uncut": fake_block(Fraction(u))}}}
              for k, c, u in vals}
    cvu = la.cut_vs_uncut(models, fake_place(models))
    per_pair = sum(1 for s in la.SCORES for q in la.QUANTITIES if q != "w")   # w is 0 in both readings: tied in both
    assert per_pair > 0
    assert len(cvu["ties_separating"]) == per_pair
    assert all(x["models"] == ["a", "b"] and x["tied_in"] == "cut" for x in cvu["ties_separating"])
    assert len(cvu["ordering_flips"]) == 2 * per_pair
    assert all(x["cut"][0] != x["cut"][1] and x["uncut"][0] != x["uncut"][1] for x in cvu["ordering_flips"])
    assert la.order_words(cvu) == (f"{2 * per_pair} pairs of ladder models change order; {per_pair} pairs tied in "
                                   "the cut reading separate in the uncut")
    rows = [{"model": k, "cut": Fraction(c), "uncut": Fraction(u)} for k, c, u in vals]
    flips, ties = vla.pair_orders({"g": rows})
    assert [(a["model"], b["model"]) for a, b in flips] == [("a", "c"), ("b", "c")]
    assert [(a["model"], b["model"], r) for a, b, r in ties] == [("a", "b", "cut")]
    one = {"ordering_flips": [], "ties_separating": [{"tied_in": "uncut"}]}
    assert la.order_words(one) == "0 pairs of ladder models change order; 1 pair tied in the uncut reading separates " \
                                  "in the cut"


QWEN = "qwen3.5-122b-a10b-fp8"
REAL_TIE = [la.model_paths(QWEN, ROOT)[r]["gsm8k"] for r in la.READINGS] + DS_FILES


@pytest.mark.skipif(not (all(p.exists() for p in REAL_TIE) and NEBIUS.exists()),
                    reason="the scored rows are not on this machine")
def test_r3_item2_the_gsm8k_tie_on_the_real_rows(tmp_path):
    """Qwen3.5-122B-A10B and DeepSeek on GSM8K, strict, CoT, cut, both exactly 255425/2638; the pair
    is a tie that separates in the uncut reading, listed as such, and no reversal involves a tie."""
    R = la.analyse(ROOT, tmp_path / "out", keys=[QWEN, DS])
    cvu = R["cut_vs_uncut"]
    t = [x for x in cvu["ties_separating"] if (x["task"], x["score"], x["quantity"]) == ("gsm8k", "strict", "cot")]
    assert len(t) == 1 and t[0]["cut"] == [Fraction(255425, 2638)] * 2 and t[0]["tied_in"] == "cut"
    assert not [x for x in cvu["ordering_flips"] if x["cut"][0] == x["cut"][1] or x["uncut"][0] == x["uncut"][1]]
    md = (tmp_path / "out" / "ladder.md").read_text(encoding="utf-8")
    assert ("- GSM8K, strict, CoT: Qwen3.5-122B-A10B and DeepSeek V4.1-Flash-1017, cut 96.8 and 96.8, uncut 97.1 and "
            "96.8; tied in the cut reading.") in md


def test_r3_item3_format_shares(six, tmp_path):
    """The readable sentence gives, per cell, the answer format's share (strict minus content, as a share of the
    strict value) of strict CoT − requested and of the strict residual, and counts the cells where it is more than
    half; the claim that the strict gains "mostly measure the answer format" is gone; the checker recomputes the
    shares."""
    import re
    root, out, R = six
    fs = R["format_shares"]
    assert fs
    for r in fs:
        x = R["models"][r["model"]]["tasks"][r["task"]]["cut"]
        gs, gc = x["strict"]["discovery"]["gain_over_requested_pp"], x["content"]["discovery"]["gain_over_requested_pp"]
        assert r["share_gain"] == (None if gs == 0 else 100 * (gs - gc) / gs)
    a = sum(r["share_residual"] is not None and r["share_residual"] > 50 for r in fs)
    b = sum(r["share_gain"] is not None and r["share_gain"] > 50 for r in fs)
    line = next(x for x in R["findings"] if x.startswith("Requested replies the strict rule cannot read"))
    assert (f"more than half of the strict residual in {a} of {len(fs)} cells and more than half of strict CoT − "
            f"requested in {b} of {len(fs)}") in line
    assert "mostly measure" not in line
    d = tmp_path / "out"
    shutil.copytree(out, d)
    md = (d / "ladder.md").read_text(encoding="utf-8")
    m = re.search(r"format (-?\d+\.\d)% of CoT − requested", md)
    plant(d / "ladder.md", m.group(0), f"format {la.fnum(Fraction(m.group(1)) + 1)}% of CoT − requested")
    assert any("format share of CoT − requested" in x for x in run_checker(root, d))


def test_r3_item4_rungs_ordered_by_total_parameters(six):
    """The limits name the architecture and the active parameters among what changes between rungs, and say
    that the rungs are ordered by total parameters, with the Qwen3.5 ladder's active parameters falling, then rising."""
    root, out, R = six
    sec2 = section((out / "ladder.md").read_text(encoding="utf-8"), 2)
    assert "a step between rungs also changes the architecture (dense or mixture of experts), the active parameters" \
        in sec2
    line = next(x for x in sec2.splitlines() if x.startswith("- Rungs are ordered by total parameters"))
    assert ("Qwen3.5: 9.65B dense → 36.0B total, 3B active → 125.1B total, 10B active; the active parameters (9.65B → "
            "3B → 10B) do not rise with the total") in line
    assert "Llama 3.1: 8.03B dense → 70B dense; the active parameters rise with the total" in line
    assert "Mistral 3: 8.92B dense → 13.9B dense → 675B total, 41B active; the active parameters rise with the total" \
        in line
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    assert line in wn


def test_r3_item5_ties_are_printed_and_checked():
    """A value equal to panel values prints its tie count in every placement cell and count; the checker
    compares the count."""
    p = la.place(Fraction(1, 2), pd.Series([Fraction(1, 2), Fraction(1, 2), Fraction(0), Fraction(1)], dtype=object))
    assert (p["below"], p["equal"], p["above"]) == (1, 2, 1)
    assert la.cell(p) == "0.5 (1/4, 2 tied)" and la.above_txt(p) == "above 1 of 4, 2 tied"
    q = la.place(Fraction(1, 3), pd.Series([Fraction(1, 2), Fraction(0)], dtype=object))
    assert la.cell(q) == "0.3 (1/2)" and la.above_txt(q) == "above 1 of 2"
    T = vla.Tally()
    X = vla.TextCheck(T, None)
    pl = vla.place(Fraction(1, 2), [Fraction(1, 2), Fraction(1, 2), Fraction(0), Fraction(1)])
    assert pl["equal"] == 2
    X.cell("cell", "0.5 (1/4, 2 tied)", pl)
    assert T.bad == []
    X.cell("cell", "0.5 (1/4)", pl)
    assert len(T.bad) == 1 and "tied" in T.bad[0]


def test_r3_item5_checker_flags_a_dropped_tie_count(six, tmp_path):
    """On the synthetic set: a printed tie count removed from a placement cell, or the count of tied cells in section
    12 changed, is flagged."""
    import re
    root, out, R = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    md = (d / "ladder.md").read_text(encoding="utf-8")
    m = re.search(r"Ties with panel models: (\d+) placement", md)
    assert m, "the synthetic set has no tie with a panel value"
    plant(d / "ladder.md", m.group(0), f"Ties with panel models: {int(m.group(1)) + 1} placement")
    c = re.search(r"\((\d+)/(\d+), (\d+) tied\)", md)
    plant(d / "ladder.md", c.group(0), f"({c.group(1)}/{c.group(2)})")
    mm = run_checker(root, d)
    assert any("ties with panel models, cut" in x for x in mm), mm
    assert any(": tied: " in x for x in mm), mm


def test_r3_item6_headline_table_puts_like_beside_like(six):
    """The note file's headline table carries the discovery-draw CoT − forced beside w and the corrected
    gain it shrinks, so no row shows a corrected gain larger in size than the gain beside it."""
    import re
    root, out, R = six
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    head = next(x for x in wn.splitlines() if x.startswith("| Model | Benchmark | Requested (48 draws)"))
    assert "| CoT − forced (discovery draws) | w (discovery draws) | Corrected gain (discovery draws) |" in head
    rows = [x for x in wn.splitlines()
            if re.match(r"^\| [^|]+ \| (GSM8K|MATH-500|BIG-Bench Hard|HotpotQA) \| -?\d", x)]
    assert len(rows) == 4 * len(la.LADDER)
    for x in rows:
        c = [v.strip() for v in x.strip("|").split("|")]
        g, cg = [Fraction(v) for v in c[7].split(" / ")], [Fraction(v) for v in c[9].split(" / ")]
        assert all(abs(b) <= abs(a) for a, b in zip(g, cg)), x
    for m in R["models"].values():
        for tb in m["tasks"].values():
            for sc in la.SCORES:
                d = tb["cut"][sc]["discovery"]
                assert abs(d["corrected_gain_pp"]) <= abs(d["gain_over_forced_pp"])


@pytest.mark.skipif(not ((ROOT / la.SPEND_MISTRAL).exists() and (ROOT / la.SPEND_NEBIUS).exists()),
                    reason="the run reports are not here")
def test_r3_item7_spend_read_from_the_reports(six, tmp_path):
    """DeepSeek's $90.05 already holds the $3.20 spent before the
    complete-design run (the report sums every reply, the pilot's included, plus the live test), so the total stays
    $90.05 and the text says what it holds; $93.25 would count the $3.20 twice. The checker flags $93.25."""
    sp = la.read_spend(ROOT)
    assert (sp["deepseek"]["total"], sp["deepseek"]["before"]) == (Fraction("90.05"), Fraction("3.20"))
    assert sp["deepseek"]["meter"] == Fraction("3.1977")
    assert (sp["mistral"]["run"], sp["mistral"]["with_test"]) == (Fraction("202.50"), Fraction("202.51"))
    root, out, R = six
    line = next(x for x in R["asks"] if x.startswith("Model lists:"))
    assert ("DeepSeek $90.05 in total (`results/frontier_panel/nebius_panel.md`), which already holds the $3.20 spent "
            "before the complete-design run: the live test, the pilot (whose replies are among the scored rows) and "
            "the test and determinism calls, $3.1977 on the run's meter (`results/frontier_panel/nebius_realtime_spend.md`)"
            ) in line
    assert "93.25" not in line
    d = tmp_path / "out"
    shutil.copytree(out, d)
    plant(d / "ladder.md", "DeepSeek $90.05 in total", "DeepSeek $93.25 in total")
    assert any("DeepSeek spend, total" in x for x in run_checker(root, d))


NEBIUS_RAW = ROOT / "data/frontier_panel/nebius_panel_raw.jsonl"


@pytest.mark.skipif(not NEBIUS_RAW.exists(), reason="DeepSeek's raw replies are not on this machine")
def test_r3_item7_spend_recounted_from_the_replies():
    """The checker's recount from the replies: every reply's cost plus the live test is the report's $90.05, and the
    phases before the complete-design run plus the live test are the meter's $3.1977, inside that total."""
    sp = vla.spend_facts(ROOT)
    raw = sp["raw"]
    assert {"pilot", "complete"} <= set(raw)
    total = sum(raw.values()) + sp["live"]
    before = total - raw["complete"]
    assert format(total, ".2f") == "90.05"
    assert (format(before, ".4f"), format(before, ".2f")) == ("3.1977", "3.20")
    assert format(total + before, ".2f") == "93.25"            # the figure with the $3.20 counted twice


def test_r3_item8_default_kernel_caveat(six, tmp_path):
    """The cut-equivalence check on the default-kernel model is stated as an untested
    caveat for Qwen3.5-122B-A10B, with the shares of its outputs the cut shortens; the checker recomputes them."""
    import re
    root, out, R = six
    assert la.default_kernel_keys(R["models"]) == [QWEN]
    f8 = next(x for x in R["footnotes"] if x.startswith("F8."))
    assert "Qwen3.5-122B-A10B ran with default kernels" in f8
    j4 = next(x for x in R["asks"] if x.startswith("Equivalence of the token-level cut"))
    if (ROOT / "results/ladder/equiv_qwen3.5-122b-a10b-fp8.json").exists():
        # the default-kernel check's rates and the caveat of its rows
        assert "untested" not in f8 and "rerun rows identical to the cut copy" in f8
        assert "These rows were chosen because the run cut them at the panel's cap" in f8
        assert "the ladder's default-kernel rung, checked on rows the run cut at the panel's cap (footnote F8)" in j4
        d = tmp_path / "out_dk"
        shutil.copytree(out, d)
        m = re.search(r"content verdict the same on (\d+) of", f8)
        plant(d / "ladder.md", m.group(0), f"content verdict the same on {int(m.group(1)) - 1} of")
        assert any("default-kernel equivalence: content verdict same" in x for x in run_checker(root, d))
    else:
        assert "untested" in f8
        assert ("it applies to Qwen3.5-122B-A10B, the ladder's default-kernel rung, where it was not run: an untested "
                "caveat (footnote F8)") in j4
    assert f8 in (out / "ladder_note.md").read_text(encoding="utf-8")
    m = re.search(r"(Qwen3\.5-122B-A10B: GSM8K requested )(\d+\.\d\d)%", f8)
    d = tmp_path / "out"
    shutil.copytree(out, d)
    plant(d / "ladder.md", m.group(0), m.group(1) + la.fnum(Fraction(m.group(2)) + Fraction(1, 100), 2) + "%")
    assert any(f"F8 {QWEN} share cut" in x for x in run_checker(root, d))


def test_r3_item9_checker_reads_every_section(six, tmp_path):
    """Numbers in the lines above section 1 and in sections 2, 4, 7 (template SDs), 12 and 13 can fail, and a
    table or a number no checker reads is a mismatch."""
    import re
    root, out, R = six
    d = tmp_path / "out"
    shutil.copytree(out, d)
    md = (d / "ladder.md").read_text(encoding="utf-8")
    plant(d / "ladder.md", "(1,024 requested;", "(1,025 requested;")
    n = re.search(r"the ladder adds (\d+) single models", md).group(1)
    plant(d / "ladder.md", f"the ladder adds {n} single models", f"the ladder adds {int(n) + 1} single models")
    plant(d / "ladder.md", "| Ministral 3 8B Instruct 2512 | Mistral 3 | 8.92B, dense",
          "| Ministral 3 8B Instruct 2512 | Mistral 3 | 8.93B, dense")
    row = re.search(r"\| Mistral Large 3 \(2512\) \| GSM8K \| (\d{1,3}(?:,\d{3})*) \((\d+)\) \|", md)
    plant(d / "ladder.md", row.group(0),
          row.group(0).replace(f"| {row.group(1)} (", f"| {int(row.group(1).replace(',', '')) + 1:,} ("))
    sd = re.search(r"(\| Llama-3\.1-70B-Instruct \| MATH-500 \| strict \| \d+ \| )(\d+\.\d) ", md)
    plant(d / "ladder.md", sd.group(0), f"{sd.group(1)}{la.fnum(Fraction(sd.group(2)) + Fraction(2, 10))} ")
    neg = re.search(r"(Mistral Large 3 \(2512\): template variance minus the item sampling \(interaction mean square "
                    r"over the items\) is negative in )(\d+)", md)
    plant(d / "ladder.md", neg.group(0), f"{neg.group(1)}{int(neg.group(2)) + 1}")
    chk = re.search(r"over (\d+) checks", md)
    plant(d / "ladder.md", chk.group(0), f"over {int(chk.group(1)) + 1} checks")
    plant(d / "ladder.md", "- Runtime ", "| a | b |\n|---|---|\n| 1 | 2 |\n\n- Checked 3 times. Runtime ")
    mm = run_checker(root, d)
    for what in ("panel cap, requested", "ladder models in the families",
                 "ministral-3-8b-instruct-2512 total parameters", "mistral-large-2512 gsm8k design rows",
                 "llama-3.1-70b-instruct-fp8 math strict requested SD", "mistral-large-2512 negative cells",
                 "panel files: checks", "a table with numbers that no checker reads", "number not checked: '3'"):
        assert any(what in x for x in mm), (what, mm[:20])


def test_r3_item9_root_is_checked_exactly():
    """The template SD is checked by bounds on its exact square: √(49/400) = 0.35 is halfway and passes only as 0.4
    (half-even); √(3008/48195) = 0.2498 passes only as 0.2."""
    T = vla.Tally()
    T.root("sd", "0.4", Fraction(49, 400))
    T.root("sd", "0.2", Fraction(3008, 48195))
    assert T.bad == []
    T.root("sd", "0.3", Fraction(49, 400))
    T.root("sd", "0.5", Fraction(49, 400))
    T.root("sd", "0.3", Fraction(3008, 48195))
    assert len(T.bad) == 3
    assert la.fnum(la.Root(Fraction(49, 400))) == "0.4" and la.fnum(la.Root(Fraction(3008, 48195))) == "0.2"


def test_r3_item10_minor_claims(six):
    """The panel's precision is read from the registries (two Llama-2 models ran in float16); DeepSeek is not
    counted among the family members; no stray "in"."""
    root, out, R = six
    assert R["panel_precision"]["llama-2-7b-chat-hf"]["dtype"] == "float16"
    assert R["panel_precision_phrase"] == "bfloat16, except `llama-2-13b-chat-hf` and `llama-2-7b-chat-hf` in float16"
    f3 = next(x for x in R["footnotes"] if x.startswith("F3."))
    assert f3.endswith("The panel ran in bfloat16, except `llama-2-13b-chat-hf` and `llama-2-7b-chat-hf` in float16 "
                       "(`config/models.yaml`, `config/models_extension.yaml`).")
    md = (out / "ladder.md").read_text(encoding="utf-8")
    wn = (out / "ladder_note.md").read_text(encoding="utf-8")
    for text in (md, wn):
        assert "six single models in three families" not in text and "models in falls" not in text
        assert "The panel ran in bfloat16." not in text and "The panel ran in bfloat16 (" not in text
    assert ("the ladder adds 5 single models to 3 families of 2 or 3 rungs, and 1 outside them (DeepSeek "
            "V4.1-Flash-1017)") in md


def test_r3_item9_fewshot_scorer_recomputed(tmp_path):
    """The checker does not take the five-shot file's "pipeline matches saved" flag on its word: it counts
    Mistral Large 3's saved rows of the arm itself and compares them with the file's own scoring. A file whose scoring
    is 1 pp off fails the basis check in the checker while the flag still says it matches."""
    root = six_model_root(tmp_path)
    fp = vla.fewshot_panel_gaps(root)
    assert fp["scorer_ok"] is True and fp["scorer_own"]["strict"][0] == fp["scorer_own"]["strict"][1]
    write_fewshot_panel(root, scorer_shift=Fraction(1))
    fp = vla.fewshot_panel_gaps(root)
    assert fp["scorer_ok"] is False
    R = la.analyse(root, tmp_path / "out")
    assert R["bbh_forced_gap"]["fewshot_panel"]["check"]["scorer_ok"] is True     # the analysis reads the flag
    mm = run_checker(root, tmp_path / "out")
    assert any("five-shot basis" in x for x in mm), mm
    assert any("the file's scoring against the saved rows" in x for x in mm), mm


# ---------------------------------------------------------------------------------------------------------------------
# Comparison coverage, the F7 footnote and ties
# ---------------------------------------------------------------------------------------------------------------------

def placed_keys(R: dict) -> set:
    """(benchmark, score, quantity) of every placement in the cut reading, readable shares as score "readable"."""
    return {(t, sc, q) for a in R["placement"].values() for t, b in a.items() for sc, d in b["cut"].items() for q in d}


def test_r4_item1_cut_vs_uncut_covers_every_placed_quantity(six):
    """The comparison covers every quantity that is placed and printed: per score the seven quantities and the
    residual, and the readable shares where the panel has them (not HotpotQA); its list is the placements' own, and
    the checker reads the same list from ladder.md's tables."""
    root, out, R = six
    cvu = R["cut_vs_uncut"]
    got = {(d["task"], d["score"], d["quantity"]) for d in cvu["differences"]}
    assert got == placed_keys(R)
    for t in la.TASKS:
        for sc in la.SCORES:
            assert (t, sc, "residual") in got
    assert {(t, "readable", a) for t in ("gsm8k", "math", "bbh") for a in la.READ_ARMS} <= got
    assert not [x for x in got if x[0] == "hotpot" and x[1] == "readable"]
    n_models = len(R["models"])
    assert len(cvu["differences"]) == n_models * (4 * 15 + 3 * 3)        # 15 per benchmark and model, 9 readable
    printed = vla.printed_lists((out / "ladder.md").read_text(encoding="utf-8"))
    assert set(printed["placed"]) == got
    assert ("math", "content", "residual") in printed["placed"] and ("bbh", "readable", "forced") in printed["placed"]
    assert {(t, s, q) for t, s, q in printed["family"]} == {
        (t, s, q) for t in la.TASKS for s in la.SCORES for q in la.QUANTITIES if not (q == "w" and s == "content")}
    rc = {(x["model"], x["task"], x["score"], x["quantity"]) for x in cvu["panel_rank_changes"]}
    want = {(k, t, sc, q) for k, a in R["placement"].items() for t, b in a.items() for sc, d in b["cut"].items()
            for q, p in d.items() if "uncut" in b
            and (p["below"], p["above"]) != (b["uncut"][sc][q]["below"], b["uncut"][sc][q]["above"])}
    assert rc == want


def test_r4_item1_checker_takes_its_list_from_the_printed_tables():
    """The checker's list is what ladder.md prints (section 6's rows, the glance tables' placed columns, section
    5's rows); a row the text prints is in it, a row it drops is not, whatever the analysis's own list."""
    md = "\n".join([
        "# x", "", "## 1. Findings", "",
        "| Model | Benchmark | CoT − forced, strict | Residual, content | Requested readable (%) |",
        "|---|---|---|---|---|", "| A | GSM8K | 1.0 (1/2) | 1.0 (1/2) | 1.0 (1/2) |", "",
        "## 5. Family ladders", "", "### F", "", "MATH-500 (strict / content; discovery draws):", "",
        "| Quantity | 8B (ladder) | Over size (strict / content) |", "|---|---|---|", "| CoT | 1.0 / 1.0 | – / – |",
        "| w | 0.1 | – |", "",
        "## 6. Where", "", "### BIG-Bench Hard (discovery draws)", "",
        "| Score | Quantity | Panel min | Panel median | Panel max | A |", "|---|---|---|---|---|---|",
        "| strict | CoT − requested minus corrected gain | 1 | 2 | 3 | 1.0 (1/2) |",
        "| – | Readable, cot (%) | 1 | 2 | 3 | 1.0 (1/2) |", "",
        "### HotpotQA, content (pooled rows)", "",
        "| Score | Quantity | Panel min | Panel median | Panel max | A |", "|---|---|---|---|---|---|",
        "| content | CoT − requested minus (1 − discovery w) × pooled CoT − forced | 1 | 2 | 3 | 1.0 (1/2) |", ""])
    p = vla.printed_lists(md)
    assert p["placed"] == [("gsm8k", "content", "residual"), ("gsm8k", "readable", "requested"),
                           ("gsm8k", "strict", "cot_minus_forced"), ("bbh", "readable", "cot"),
                           ("bbh", "strict", "residual"), ("hotpot", "content", "residual")]
    assert p["family"] == [("math", "content", "cot"), ("math", "strict", "cot"), ("math", "strict", "w")]
    fewer = vla.printed_lists(md.replace("| strict | CoT − requested minus corrected gain | 1 | 2 | 3 | 1.0 (1/2) |\n",
                                         ""))
    assert ("bbh", "strict", "residual") not in fewer["placed"]


def test_r4_item1_checker_flags_a_comparison_without_the_residual(tmp_path, monkeypatch):
    """An analysis whose comparison leaves out a placed, printed quantity (here the
    residual) is caught, because the checker compares what ladder.md prints, not the analysis's list."""
    root = six_model_root(tmp_path)
    orig = la.cut_vs_uncut

    def no_residual(models, place_):
        p2 = {k: {t: {r: {sc: {q: v for q, v in d.items() if q != "residual"} for sc, d in b.items()}
                      for r, b in a.items()} for t, a in x.items()} for k, x in place_.items()}
        return orig(models, p2)
    monkeypatch.setattr(la, "cut_vs_uncut", no_residual)
    la.analyse(root, tmp_path / "out")
    mm = run_checker(root, tmp_path / "out")
    assert any("cut_vs_uncut/differences: recomputed 414, file 366" in x for x in mm), mm[:10]
    assert any(x.startswith("MISMATCH ladder.md") and "residuals: compared" in x for x in mm), mm[:10]


def test_r4_item1_cap_line_names_what_depends_on_the_cap(six, tmp_path):
    """Section 1 says which findings depend on the cap (the residual band, the marks, the median side, the
    readable list), recomputed with the uncut values; the checker recomputes each part and flags a planted change."""
    import re
    root, out, R = six
    line = next(x for x in R["findings"] if x.startswith("Findings above that depend on the cap"))
    cd = R["cut_vs_uncut"]["cap_dependence"]
    n = cd["band_cells"]["discovery"]
    c, u = len(cd["band"]["discovery"]["cut"]), len(cd["band"]["discovery"]["uncut"])
    assert (f"content, discovery draws: {c} of {n} in both readings" in line if
            {x[:2] for x in cd["band"]["discovery"]["cut"]} == {x[:2] for x in cd["band"]["discovery"]["uncut"]}
            else f"content, discovery draws: {c} of {n} cut, {u} of {n} uncut" in line)
    for x in cd["marks"]:
        assert x["cut"]["outside_panel_range"] != x["uncut"]["outside_panel_range"]
    assert line in (out / "ladder_note.md").read_text(encoding="utf-8")
    d = tmp_path / "out"
    shutil.copytree(out, d)
    m = re.search(r"(Residual within ±2 pp: content, discovery draws: )(\d+)( of )", line)
    plant(d / "ladder.md", m.group(0), f"{m.group(1)}{int(m.group(2)) + 1}{m.group(3)}")
    assert any("cap: band discovery" in x for x in run_checker(root, d))


@pytest.mark.skipif(not (all(p.exists() for p in DS_FILES) and NEBIUS.exists()),
                    reason="DeepSeek's scored rows are not on this machine")
def test_r4_item1_deepseek_math_residual_depends_on_the_cap(tmp_path):
    """DeepSeek's rows: its MATH-500 residual is 1.8 (strict) and 0.6 (content) cut against
    -3.3 and -4.6 uncut (both 5.16 pp lower), it leaves the ±2 pp band and goes below the panel's minimum."""
    R = la.analyse(ROOT, tmp_path / "out", keys=[DS])
    cvu = R["cut_vs_uncut"]
    d = {(x["score"], x["quantity"]): x for x in cvu["differences"] if x["task"] == "math"}
    for sc, c, u in (("strict", "1.8", "-3.3"), ("content", "0.6", "-4.6")):
        x = d[(sc, "residual")]
        assert (la.fnum(x["cut"]), la.fnum(x["uncut"]), la.fnum(x["uncut_minus_cut"], 2)) == (c, u, "-5.16")
    moved = {(x["score"], x["quantity"]): x for x in cvu["panel_rank_changes"] if x["task"] == "math"}
    assert (moved[("strict", "residual")]["cut_below"], moved[("strict", "residual")]["uncut_below"]) == (7, 0)
    assert (moved[("content", "residual")]["cut_below"], moved[("content", "residual")]["uncut_below"]) == (8, 0)
    line = next(x for x in R["findings"] if x.startswith("Findings above that depend on the cap"))
    assert "leaves: DeepSeek V4.1-Flash-1017 MATH-500, 0.6 cut, -4.6 uncut" in line
    assert ("DeepSeek V4.1-Flash-1017 MATH-500 content CoT − requested minus corrected gain, discovery draws: 0.6 (8/29) "
            "cut, -4.6 (0/29) ↓ uncut") in line


def test_r4_item4_f7_says_the_pilot_failed(six, tmp_path):
    """F7 says the Qwen3.5-122B-A10B pilot check failed, quotes its verdict, gives the rule and the counts (53 of
    96 requested-arm rows boxed, 0.552, against 90%), and says the full run went ahead with no recorded release
    decision; the checker compares each against the check's own files."""
    root, out, R = six
    pj = json.loads((root / "results/ladder/pilot_check_qwen3.5-122b-a10b-fp8.json").read_text(encoding="utf-8"))
    f7 = next(x for x in R["footnotes"] if x.startswith("F7."))
    assert f7.startswith("F7. Qwen3.5-122B-A10B failed its pilot check")
    assert f"\"{pj['verdict']}\"" in f7 and "flagged" not in f7
    assert pj["verdict"] == "PILOT FAIL qwen3.5-122b-a10b-fp8: requested arms: boxed answer in 0.552 of rows (< 0.9)"
    assert "at least 90% of the requested arms' rows" in f7 and "the pilot gave one in 53 of 96 (0.552)" in f7
    assert ("zero-shot requested 1 of 16 on GSM8K, 1 of 16 on MATH-500, 16 of 16 on BIG-Bench Hard, 3 of 16 on "
            "HotpotQA; five-shot requested 16 of 16 on GSM8K, 16 of 16 on BIG-Bench Hard") in f7
    assert "with no recorded release decision" in f7
    assert f7 in (out / "ladder_note.md").read_text(encoding="utf-8")
    d = tmp_path / "out"
    shutil.copytree(out, d)
    plant(d / "ladder.md", "the pilot gave one in 53 of 96", "the pilot gave one in 54 of 96")
    plant(d / "ladder.md", "3 of 16 on HotpotQA", "4 of 16 on HotpotQA")
    plant(d / "ladder_note.md", "boxed answer in 0.552 of rows", "boxed answer in 0.652 of rows")
    mm = run_checker(root, d)
    assert any(x.startswith("MISMATCH ladder.md") and "F7: boxed rows of the requested arms" in x for x in mm), mm
    assert any(x.startswith("MISMATCH ladder.md") and "F7: hotpot direct_answer boxed" in x for x in mm), mm
    assert any(x.startswith("MISMATCH ladder_note.md") and "F7 quotes the pilot check's verdict" in x for x in mm), mm


def test_r4_item5_largest_change_names_every_tie():
    """When several quantities share the largest change, both scripts name all of them; a printed phrase that
    names one of them is a mismatch."""
    import re
    v = Fraction(19675, 2388)
    k = "ministral-3-14b-instruct-2512"
    diffs = [{"model": k, "task": "math", "score": sc, "quantity": q, "cut": Fraction(0), "uncut": v,
              "uncut_minus_cut": v} for sc in ("strict", "content") for q in ("cot", "cot_minus_forced")]
    diffs.append({"model": k, "task": "math", "score": "strict", "quantity": "requested", "cut": Fraction(0),
                  "uncut": Fraction(1), "uncut_minus_cut": Fraction(1)})
    diffs.append({"model": k, "task": "math", "score": "readable", "quantity": "cot", "cut": Fraction(0),
                  "uncut": Fraction(11), "uncut_minus_cut": Fraction(11)})
    mx, top = la.largest_changes(diffs)
    assert mx == 11 and len(top) == 1
    mx, top = la.largest_changes(diffs, readable=False)
    assert mx == v and len(top) == 4
    phrase = ("8.24 pp, shared by 4 quantities (Ministral 3 14B Instruct 2512, MATH-500: strict CoT, strict CoT − "
              "forced, content CoT, content CoT − forced)")
    assert la.largest_words(mx, top) == phrase
    assert la.largest_words(*la.largest_changes(diffs[:1])) == ("8.24 pp (Ministral 3 14B Instruct 2512, MATH-500: "
                                                                "strict CoT)")
    F = vla.Facts()
    F.diffs = [dict(d, q=d["quantity"], d=d["uncut_minus_cut"]) for d in diffs]
    assert vla.largest(F, False) == (v, {(k, "math", sc, q) for sc in ("strict", "content")
                                         for q in ("cot", "cot_minus_forced")})
    for printed, ok in ((phrase, True), ("8.24 pp (Ministral 3 14B Instruct 2512, MATH-500: strict CoT)", False),
                        (phrase.replace("shared by 4", "shared by 3"), False)):
        T = vla.Tally()
        X = vla.TextCheck(T, F)
        ln = vla.Line(printed, "x")
        X.largest_part(ln, re.match(X.LARGEST + "$", printed), 1, 2, 3, False, "largest")
        assert (T.bad == []) == ok, (printed, T.bad)
