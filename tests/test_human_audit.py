"""scripts/run_human_audit.py: synthetic exports built from the real
manifests/human_audit/assignment.csv, with a synthetic key and LLM-rater file, so the tests run without the git-ignored
key; plus paper/macros_added.human_audit_macros on the results."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_human_audit_t", ROOT / "scripts" / "run_human_audit.py")
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)
ASSIGN = ROOT / "manifests" / "human_audit" / "assignment.csv"
HEADER = ["time_utc", "rater", "audit_id", "event", "verdict", "flags"]


@pytest.fixture(scope="module")
def assignment():
    return pd.read_csv(ASSIGN, dtype=str)


def build_key(assignment, path):
    """Tasks cycle gsm8k, bbh, exemplar; every third answer of a task has the scorers agreeing (both right), the rest
    disagree with strict wrong and the second scorer right (as in the real sample)."""
    rows = []
    for j, i in enumerate(assignment["audit_id"]):
        task = ("gsm8k", "bbh", "exemplar")[j % 3]
        agree = (j // 3) % 3 == 0
        rows.append({"audit_id": i, "task": task, "agreement": "agree" if agree else "disagree",
                     "score_a": 1 if agree else 0, "score_b": 1})
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def build_llm(assignment, path, verdict="gold"):
    pd.DataFrame({"audit_id": assignment["audit_id"], "verdict": verdict, "flags": "", "final_answer": "x",
                  "reason": "r"}).to_csv(path, index=False)


def export_rows(assignment, verdict_of=lambda r, i: "gold"):
    rows, t = [], 0
    for r in ("A", "B"):
        sub = assignment[assignment[f"rater_{r}"] == "1"]
        for i in sub["audit_id"]:
            t += 1
            rows.append([f"2026-11-20T10:{t // 60:02d}:{t % 60:02d}.000Z", r, i, "first", verdict_of(r, i), ""])
    return rows


def write_export(rows, path):
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)


@pytest.fixture
def env(tmp_path, assignment):
    key = tmp_path / "key.csv"
    llm = tmp_path / "llm.csv"
    K = build_key(assignment, key)
    build_llm(assignment, llm)
    kw = dict(out=str(tmp_path / "out"), assignment=str(ASSIGN), key=str(key), llm=str(llm), manifest=None)
    return tmp_path, kw, K


def load(tmp_path):
    return json.load(open(tmp_path / "out" / "human_audit.json", encoding="utf-8"))


def test_absent_export_exits_2(env, capsys):
    tmp_path, kw, _ = env
    assert ha.run(export=str(tmp_path / "missing.csv"), **kw) == 2
    assert "not there yet" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_valid_export_all_agree(env, assignment):
    tmp_path, kw, K = env
    rows = export_rows(assignment)
    write_export(rows, tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    assert R["counts"]["answers"] == 384 and R["counts"]["shared"] == 100 and R["counts"]["decided"] == 384
    ra = R["raters"]
    assert ra["codes"]["agree"] == 100 and ra["codes"]["share"] == 1.0
    # every rating is "gold": one category throughout, so kappa is undefined and says why
    assert ra["codes"]["kappa"] is None and ra["codes"]["kappa_status"].startswith("undefined")
    ref = R["references"]
    # human says correct everywhere; strict is right only where the scorers agree
    sc = K[K["task"].isin(["gsm8k", "bbh"])]
    o = ref["strict_and_content"]
    assert o["decided"] == len(sc)
    assert o["strict"]["agree"] == int((sc["score_a"] == 1).sum())
    assert o["content"]["share"] == 1.0 and o["llm"]["share"] == 1.0
    lo, hi = o["content"]["ci95_wilson"]
    assert hi == pytest.approx(1.0) and 0.98 < lo < 1.0
    assert ref["strict_content_disagree"]["strict"]["agree"] == 0
    assert ref["task:exemplar"]["second_scorer"]["name"] == "cheng"
    assert "content" not in ref["task:exemplar"] and "content" not in ref["all"]
    assert "changed" not in R                     # changed verdicts are not reported
    # each rater alone: every rating gold, so content agreement is the share content scores correct
    for r in ("A", "B"):
        b = R["per_rater"][r]
        assert b["answers"] == 242 and b["cant_tell"] == 0
        assert b["controls"]["all"]["both_scorers_wrong"]["read_correct"] == b["controls"]["all"]["both_scorers_wrong"]["n"]
        assert b["controls"]["all"]["both_scorers_right"]["read_not_correct"] == 0
    s = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert s == (tmp_path / "out" / "summary_clean.md").read_text(encoding="utf-8")
    assert "undefined" in s and "Wilson" in s and "not population" in s


def test_shared_disagreement_is_excluded_and_counted(env, assignment):
    tmp_path, kw, K = env
    shared = list(assignment.loc[assignment["shared"] == "1", "audit_id"])
    d1, d2, d3 = shared[0], shared[1], shared[2]

    def v(r, i):
        if i == d1:
            return "gold" if r == "A" else "other"        # disagree: correct against not correct
        if i == d2:
            return "other" if r == "A" else "none"        # same code, different verdict: agree
        if i == d3:
            return "unclear"                              # both can't tell: agree, excluded as can't tell
        return "gold" if (int(i[1:]) % 4) else "other"
    rows = export_rows(assignment, v)
    rows.append(["2026-11-20T12:00:00.000Z", "A", d1, "flags", "", "changes;cut_off"])
    rows.append(["2026-11-20T12:00:01.000Z", "A", d1, "change", "other", "changes;cut_off"])
    rows.append(["2026-11-20T12:00:02.000Z", "B", d2, "change", "gold", ""])
    rows.append(["2026-11-20T12:00:03.000Z", "B", d2, "change", "none", ""])   # back to the first verdict
    write_export(rows, tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    c, ra = R["counts"], R["raters"]
    assert c["rater_disagreements_excluded"] == 1 and c["cant_tell_excluded"] == 1 and c["decided"] == 382
    assert R["rater_disagreements"] == [{"audit_id": d1, "A": "gold", "B": "other"}]
    assert ra["codes"]["agree"] == 99 and ra["verdicts"]["agree"] == 98
    assert ra["same_code_different_verdict"] == 1
    assert ra["decided_both"]["n"] == 99 and ra["decided_both"]["agree"] == 98
    # kappa by hand on the three codes
    first = {(r[1], r[2]): r[4] for r in rows if r[3] == "first"}
    a = [ha.CODE[first[("A", i)]] for i in sorted(shared)]
    b = [ha.CODE[first[("B", i)]] for i in sorted(shared)]
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(k) / n * b.count(k) / n for k in set(a) | set(b))
    assert ra["codes"]["kappa"] == pytest.approx((po - pe) / (1 - pe))
    assert ra["table_codes"]["A_correct__B_not_correct"] == 1
    # change rows are validated but not reported; the per-rater figures use first verdicts
    assert "changed" not in R and "changed" not in json.dumps(R["per_rater"])
    assert R["flag_rows"] == 1
    m = R.get("macros")
    if m is not None:                          # paper/ is untracked; a clean checkout writes no macros
        assert m["hraShared"] == "100" and m["hraN"] == "382" and m["hraRaterDisagree"] == "1"
        assert m["hraAgreePct"] == "99.0" and "hraChangedPct" not in m
        assert m["hraKappa"] not in ("--", "")


def test_agreement_with_references_by_hand(env, assignment):
    tmp_path, kw, K = env
    rows = export_rows(assignment, lambda r, i: "gold" if (int(i[1:]) % 4) else "other")
    write_export(rows, tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    H = K.copy()
    H["human"] = H["audit_id"].map(lambda i: bool(int(i[1:]) % 4))
    for scope, mask in (("strict_and_content", H["task"].isin(["gsm8k", "bbh"])),
                        ("strict_content_disagree", H["task"].isin(["gsm8k", "bbh"]) & (H["agreement"] == "disagree"))):
        d = H[mask]
        o = R["references"][scope]
        assert o["content"]["agree"] == int((d["human"] == (d["score_b"] == 1)).sum())
        assert o["strict"]["agree"] == int((d["human"] == (d["score_a"] == 1)).sum())
        assert o["llm"]["agree"] == int(d["human"].sum())          # the LLM rater says gold everywhere
        k, n = o["content"]["agree"], o["content"]["n"]
        lo, hi = o["content"]["ci95_wilson"]
        assert lo < k / n < hi
    assert ha.wilson(0, 0) is None
    lo, hi = ha.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)


def test_llm_cant_tell_is_excluded_and_counted(env, assignment):
    tmp_path, kw, K = env
    cl = pd.read_csv(kw["llm"], dtype=str)
    cl.loc[0, "verdict"] = "unclear"
    cl.to_csv(kw["llm"], index=False)
    write_export(export_rows(assignment), tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    assert R["counts"]["llm_cant_tell"] == 1
    assert R["references"]["all"]["llm"]["n"] == 383 and R["references"]["all"]["llm"]["llm_cant_tell"] == 1


@pytest.mark.parametrize("break_it, message", [
    (lambda rows: rows[:-1], "no `first` row"),
    (lambda rows: rows + [rows[0][:]], "more than one `first` row"),
    (lambda rows: [r if k else r[:4] + ["maybe"] + r[5:] for k, r in enumerate(rows)], "verdict other than"),
    (lambda rows: rows + [["t", "C", rows[0][2], "first", "gold", ""]], "a rater other than"),
    (lambda rows: rows + [["t", "A", "V999", "flags", "", ""]], "not in assignment.csv"),
    (lambda rows: rows + [["t", rows[0][1], rows[0][2], "flags", "", "shiny"]], "a flag other than"),
    (lambda rows: rows + [["t", rows[0][1], rows[0][2], "undo", "gold", ""]], "an event other than"),
    (lambda rows: [["t", rows[0][1], rows[0][2], "change", "other", ""]] + rows, "come before"),
])
def test_malformed_export_is_refused(env, assignment, capsys, break_it, message):
    tmp_path, kw, _ = env
    write_export(break_it(export_rows(assignment)), tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 1
    out = capsys.readouterr().out
    assert "refused" in out and message in out
    assert not (tmp_path / "out").exists()


def test_rating_an_unassigned_answer_is_refused(env, assignment, capsys):
    tmp_path, kw, _ = env
    only_b = assignment.loc[(assignment["rater_A"] == "0"), "audit_id"].iloc[0]
    write_export(export_rows(assignment) + [["t", "A", only_b, "first", "gold", ""]], tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "not assigned to that rater" in capsys.readouterr().out


def test_wrong_columns_are_refused(env, capsys):
    tmp_path, kw, _ = env
    pd.DataFrame({"audit_id": ["V001"], "verdict": ["gold"]}).to_csv(tmp_path / "ex.csv", index=False)
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "lacks columns" in capsys.readouterr().out


def with_set_column(rows, names=None):
    """The sheet's later layout: rater = username, set = A/B."""
    names = names or {"A": "user1", "B": "user2"}
    return pd.DataFrame([[t, names[r], r, i, e, v, f] for t, r, i, e, v, f in rows],
                        columns=["time_utc", "rater", "set", "audit_id", "event", "verdict", "flags"])


def test_export_with_set_column_reads_set_as_rater(env, assignment):
    tmp_path, kw, _ = env
    with_set_column(export_rows(assignment)).to_csv(tmp_path / "ex.csv", index=False)
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    assert R["counts"]["answers"] == 384 and R["raters"]["codes"]["agree"] == 100


def test_export_with_set_column_refuses_a_user_in_two_sets(env, assignment, capsys):
    tmp_path, kw, _ = env
    with_set_column(export_rows(assignment), {"A": "user1", "B": "user1"}).to_csv(tmp_path / "ex.csv", index=False)
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "one to one" in capsys.readouterr().out


def test_blank_trailing_rows_are_ignored(env, assignment):
    tmp_path, kw, _ = env
    write_export(export_rows(assignment) + [["", "", "", "", "", ""]] * 3, tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **kw) == 0


def test_kappa_edge_cases():
    k, po, pe, st = ha.cohen_kappa([], [])
    assert k is None and st.startswith("undefined")
    k, po, pe, st = ha.cohen_kappa(["x"] * 5, ["x"] * 5)
    assert k is None and po == 1.0 and pe == 1.0 and "expected agreement is 1" in st
    k, *_ = ha.cohen_kappa(["x", "y"], ["y", "x"])
    assert k == pytest.approx(-1.0)                                     # negative kappa kept as it is
    k, *_ = ha.cohen_kappa(["x", "x", "y"], ["x", "x", "x"])
    assert k == pytest.approx(0.0)


def test_real_key_and_llm_when_present(tmp_path, assignment):
    """With the real git-ignored key and the LLM rater's ratings, a synthetic all-gold export runs end to end."""
    if not os.path.exists(ha.KEY):
        pytest.skip("data/scoring_audit_v2/key.csv is git-ignored and absent")
    write_export(export_rows(assignment), tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), out=str(tmp_path / "out")) == 0
    R = load(tmp_path)
    assert R["counts"]["decided_strict_and_content"] == 288
    assert R["references"]["strict_content_disagree"]["decided"] == 216
    assert R["references"]["task:exemplar"]["decided"] == 96


# ---------------------------------------------------------------- macros


def test_macros_from_results_and_skip_while_absent(env, assignment, tmp_path):
    sys.path.insert(0, str(ROOT / "paper"))
    ma = pytest.importorskip("macros_added")
    root = tmp_path / "root"
    assert ma.human_audit_results(str(root)) is None
    assert "not rated yet" in ma.pending(str(root))["human audit, two raters"]
    _, kw, _ = env
    write_export(export_rows(assignment), tmp_path / "ex.csv")
    assert ha.run(export=str(tmp_path / "ex.csv"), **{**kw, "out": str(root / "results" / "human_audit")}) == 0
    R = ma.human_audit_results(str(root))
    assert ma.pending(str(root))["human audit, two raters"].startswith("rated")
    m = ma.human_audit_macros(R)
    want = {"hraN", "hraShared", "hraKappa", "hraAgreePct", "hraContentAgreePct", "hraContentAgreePctLo",
            "hraContentAgreePctHi", "hraStrictAgreePct", "hraDisagreeContentAgreePct"}
    assert want <= set(m) and all(k.isalpha() for k in m)
    assert "time_utc_range" not in R                                                # rating times not reported
    assert m["hraKappa"] == "--"                                       # undefined: every rating gold
    assert m["hraContentAgreePct"] == "100.0" and m["hraContentAgreePctHi"] == "100.0"
    assert m["hraDisagreeStrictAgreePct"] == "0.0" and "hraChangedPct" not in m      # changes are not reported
    assert m["hraN"] == "384" and m["hraShared"] == "100" and m["hraAgreePct"] == "100.0"
    assert {k: R["macros"][k] for k in m} == m
    assert set(R["macros"]) - set(m) == set(ma.human_audit_rater_macros(R))      # each rater alone


def test_per_rater_controls_by_hand():
    """Each rater alone, no exclusion; control answers from both scorers of the key."""
    key = pd.DataFrame({"audit_id": ["a1", "a2", "a3", "a4", "a5"], "task": ["gsm8k", "gsm8k", "bbh", "exemplar", "bbh"],
                        "agreement": ["agree", "agree", "disagree", "agree", "agree"],
                        "score_a": [False, True, False, False, True], "score_b": [False, True, True, False, True]})
    first = pd.Series({("A", "a1"): "gold", ("A", "a2"): "gold", ("A", "a3"): "gold", ("A", "a4"): "gold",
                       ("A", "a5"): "unclear", ("B", "a1"): "other", ("B", "a2"): "none", ("B", "a3"): "gold",
                       ("B", "a4"): "other", ("B", "a5"): "gold"})
    first.index = pd.MultiIndex.from_tuples(first.index, names=["rater", "audit_id"])
    out = ha.per_rater(first, key)
    A, B = out["A"], out["B"]
    assert A["answers"] == 5 and A["cant_tell"] == 1
    # A on GSM8K and BIG-Bench Hard (a1, a2, a3 decided): content says F, T, T; A says T, T, T
    assert (A["strict_and_content"]["content"]["agree"], A["strict_and_content"]["content"]["n"]) == (2, 3)
    assert A["controls"]["strict_and_content"]["both_scorers_wrong"] == {
        "n": 1, "read_correct": 1, "share": 1.0, "ci95_wilson": ha.wilson(1, 1)}
    assert A["controls"]["all"]["both_scorers_wrong"]["n"] == 2            # a1 and the exemplar run's a4
    assert B["controls"]["all"]["both_scorers_wrong"]["read_correct"] == 0
    assert B["controls"]["strict_and_content"]["both_scorers_right"]["read_not_correct"] == 1     # a2 read "none"
    assert B["all"]["strict"]["n"] == 5
