"""The third rater of the human audit. scripts/run_human_audit.py
--three-raters on synthetic three-rater exports built from the real manifests/human_audit/assignment.csv with a
synthetic key; paper/macros_third_rater.py on the results; the two-rater path unchanged."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_human_audit_t3", ROOT / "scripts" / "run_human_audit.py")
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)
sys.path.insert(0, str(ROOT / "paper"))
ASSIGN = ROOT / "manifests" / "human_audit" / "assignment.csv"
HEADER = ["time_utc", "rater", "set", "audit_id", "event", "verdict", "flags"]
USER = {"A": ("user1", "A"), "B": ("user2", "B"), "C": ("user3", "B")}


@pytest.fixture(scope="module")
def assignment():
    return pd.read_csv(ASSIGN, dtype=str)


def build_key(assignment, path):
    """Tasks cycle gsm8k, bbh, exemplar; every third answer of a task has both scorers right, the rest strict wrong and
    the second scorer right."""
    rows = []
    for j, i in enumerate(assignment["audit_id"]):
        agree = (j // 3) % 3 == 0
        rows.append({"audit_id": i, "task": ("gsm8k", "bbh", "exemplar")[j % 3],
                     "agreement": "agree" if agree else "disagree", "score_a": int(agree), "score_b": 1})
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def rows_for(assignment, verdict_of, raters=("A", "B", "C")):
    out, t = [], 0
    for r in raters:
        col = "rater_A" if r == "A" else "rater_B"
        for i in assignment.loc[assignment[col] == "1", "audit_id"]:
            t += 1
            name, s = USER[r]
            out.append([f"2026-11-21T10:{t // 60 % 60:02d}:{t % 60:02d}.000Z", name, s, i, "first", verdict_of(r, i), ""])
    return out


def write(rows, path):
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)


@pytest.fixture
def env(tmp_path, assignment):
    key = tmp_path / "key.csv"
    K = build_key(assignment, key)
    kw = dict(out=str(tmp_path / "out"), assignment=str(ASSIGN), key=str(key), manifest=None, two_rater_export=None,
              rater3_csv=None)
    return tmp_path, kw, K


def load(tmp_path):
    return json.load(open(tmp_path / "out" / "three_raters.json", encoding="utf-8"))


# ---------------------------------------------------------------- statistics by hand


def test_gwet_ac1_and_kappa_by_hand():
    a = ["correct", "correct", "correct", "not_correct"]
    b = ["correct", "correct", "not_correct", "not_correct"]
    ac, po, pe, st = ha.gwet_ac1(a, b, ha.BINARY)
    # pi_correct = (3 + 2) / 8 = 0.625; Pe = 2 * 0.625 * 0.375 / (2 - 1) = 0.46875
    assert st == "defined" and po == 0.75 and pe == pytest.approx(0.46875)
    assert ac == pytest.approx((0.75 - 0.46875) / (1 - 0.46875))          # 0.5294
    assert ha.cohen_kappa(a, b)[0] == pytest.approx(0.5)
    # three codes: q = 3, Pe = sum pi (1 - pi) / 2
    a3 = ["correct", "not_correct", "cant_tell", "correct"]
    b3 = ["correct", "correct", "cant_tell", "not_correct"]
    ac3, po3, pe3, _ = ha.gwet_ac1(a3, b3, ha.CODES)
    pi = [4 / 8, 2 / 8, 2 / 8]
    assert pe3 == pytest.approx(sum(p * (1 - p) for p in pi) / 2) and po3 == 0.5
    assert ac3 == pytest.approx((0.5 - pe3) / (1 - pe3))
    # every rating the same: kappa undefined, AC1 defined and 1
    same = ["correct"] * 5
    assert ha.cohen_kappa(same, same)[0] is None
    assert ha.gwet_ac1(same, same, ha.BINARY)[0] == pytest.approx(1.0)
    assert ha.gwet_ac1([], [], ha.BINARY)[3].startswith("undefined")


def test_majority():
    assert ha.majority(["correct", "correct", "not_correct"]) == "correct"
    assert ha.majority(["not_correct", "cant_tell", "not_correct"]) == "not_correct"
    assert ha.majority(["correct", "not_correct", "cant_tell"]) is None


# ---------------------------------------------------------------- the run


def test_absent_export_refuses_with_code_2(env, capsys):
    tmp_path, kw, _ = env
    assert ha.run_three(export=str(tmp_path / "ratings_three_raters.csv"), **kw) == 2
    out = capsys.readouterr().out
    assert "not there yet" in out and "ratings_three_raters.csv" in out and "--three-raters" in out
    assert not (tmp_path / "out").exists()
    assert ha.main(["--three-raters", "--export3", str(tmp_path / "nope.csv"), "--rater3-csv", str(tmp_path / "nope3.csv"),
                    "--out", str(tmp_path / "o2")]) == 2
    assert not (tmp_path / "o2").exists()


def test_real_default_path_absent_or_valid():
    """The real export does not exist yet; the default run must refuse (or, once it exists, it is the real file)."""
    try:
        filled = Path(ha.RATER3_CSV).exists() and ha.read_rater3(ha.RATER3_CSV) is not None
    except ha.ExportError:
        filled = True                                     # partly filled: the third rater is still rating
    if not Path(ha.EXPORT3).exists() and not filled:
        assert ha.run_three(out=str(ROOT / "results" / "scratch" / "never_written")) == 2
        assert not (ROOT / "results" / "scratch" / "never_written").exists()


def test_all_agree(env, assignment):
    tmp_path, kw, K = env
    write(rows_for(assignment, lambda r, i: "gold"), tmp_path / "ex.csv")
    assert ha.run_three(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    assert R["counts"] == {"answers": 384, "shared": 100, "set_B": 242, "rated_by_C": 242}
    assert R["pairwise"]["AB"]["n"] == 100 and R["pairwise"]["AC"]["n"] == 100 and R["pairwise"]["BC"]["n"] == 242
    assert R["pairwise"]["BC"]["codes"]["kappa"] is None and R["pairwise"]["BC"]["codes"]["ac1"] == pytest.approx(1)
    assert R["majority"]["unanimous"] == 100 and R["ab_resolution"]["disagreements"] == 0
    h = R["headline"]["C_for_B"]["strict_wrong_content_correct"]
    gb = K[K["task"].isin(["gsm8k", "bbh"]) & (K["score_a"] == 0)]
    assert h["decided"] == len(gb) and h["read_correct"] == len(gb) and h["share_read_correct"] == 1.0
    for f in ("summary_three_raters.md", "summary_three_raters_clean.md", "tables/three_raters_pairwise.csv",
              "tables/three_raters_ab_resolution.csv"):
        assert (tmp_path / "out" / f).exists()
    assert not (tmp_path / "out" / "human_audit.json").exists()          # the two-rater file is not written
    assert "time" not in json.dumps(R["pairwise"]) and "change" not in json.dumps(R["headline"])


def test_disagreements_majority_resolution_and_headline_by_hand(env, assignment):
    tmp_path, kw, K = env
    a = assignment
    shared = sorted(a.loc[a["shared"] == "1", "audit_id"])
    only_b = sorted(a.loc[(a["rater_B"] == "1") & (a["shared"] == "0"), "audit_id"])
    # shared: A-B disagree on s0..s5; C sides with A on s0-s2, with B on s3-s4, can't tell on s5 (no majority)
    dis = shared[:6]
    tvec = K.set_index("audit_id")
    sw_cc = set(K.loc[K["task"].isin(["gsm8k", "bbh"]) & (K["score_a"] == 0), "audit_id"])
    c_off = [i for i in only_b if i in sw_cc][:7]           # C reads 7 set-B-only strict-wrong outputs as wrong

    def v(r, i):
        if i in dis:
            j = dis.index(i)
            A, B = ("gold", "other") if j % 2 == 0 else ("other", "gold")
            C = A if j < 3 else B if j < 5 else "unclear"
            return {"A": A, "B": B, "C": C}[r]
        if r == "C" and i in c_off:
            return "other"
        return "gold"
    write(rows_for(a, v), tmp_path / "ex.csv")
    assert ha.run_three(export=str(tmp_path / "ex.csv"), **kw) == 0
    R = load(tmp_path)
    rs = R["ab_resolution"]
    assert (rs["disagreements"], rs["sides_with_A"], rs["sides_with_B"], rs["unresolved"]) == (6, 3, 2, 1)
    # s0 (A gold), s1 (A other), s2 (A gold) side with A; s3 (B other), s4 (B gold) with B
    assert rs["resolved_correct"] == 3 and rs["resolved_not_correct"] == 2
    mj = R["majority"]
    assert (mj["unanimous"], mj["two_of_three"], mj["no_majority"]) == (94, 5, 1)
    # A-B on the shared: 94 of 100 the same; the 2x2 over the 100 both decided: 3 A-only, 3 B-only correct
    ab = R["pairwise"]["AB"]
    assert ab["codes"]["agree"] == 94 and ab["table_2x2"]["first_only_correct"] == 3 \
        and ab["table_2x2"]["second_only_correct"] == 3 and ab["decided_both"]["n"] == 100
    # B-C on set B: differ on s0-s5 except where C sides with B (s3, s4) -> 4 differ, plus the 7 c_off
    bc = R["pairwise"]["BC"]
    assert bc["n"] == 242 and bc["codes"]["agree"] == 242 - 4 - 7
    assert bc["decided_both"]["n"] == 241                         # s5: C can't tell
    k_hand, _, _, _ = ha.cohen_kappa(*zip(*[(ha.CODE[v("B", i)], ha.CODE[v("C", i)])
                                            for i in sorted(a.loc[a["rater_B"] == "1", "audit_id"])]))
    assert bc["codes"]["kappa"] == pytest.approx(k_hand)
    # headline: strict-wrong, content-correct GSM8K/BBH outputs
    hC = R["headline"]["C_for_B"]["strict_wrong_content_correct"]
    hB = R["headline"]["B_for_C"]["strict_wrong_content_correct"]
    n_sw = len(sw_cc)
    maj_of = {i: ha.majority([ha.CODE[v(r, i)] for r in "ABC"]) for i in shared}
    excl = sum(1 for i in shared if i in sw_cc and maj_of[i] is None)
    not_shared_maj = sum(1 for i in shared if i in sw_cc and maj_of[i] == "not_correct")
    assert hC["no_majority_excluded"] == excl == hB["no_majority_excluded"]
    assert hC["decided"] == n_sw - excl
    assert hC["read_correct"] == n_sw - excl - not_shared_maj - len(c_off)
    assert hB["read_correct"] == n_sw - excl - not_shared_maj
    assert hC["ci95_wilson"] == ha.wilson(hC["read_correct"], hC["decided"])
    assert hC["by_majority"] == sum(1 for i in shared if i in sw_cc) - excl
    pc = R["per_rater"]["C"]
    assert pc["answers"] == 242 and pc["cant_tell"] == 1
    # macros
    import macros_third_rater as mt
    m = mt.third_rater_macros(R)
    assert all(k.isalpha() and k.startswith("hrt") for k in m)
    assert m["hrtABDisagree"] == "6" and m["hrtABSideA"] == "3" and m["hrtABSideB"] == "2"
    assert m["hrtMajNone"] == "1" and m["hrtNBC"] == "242" and m["hrtAgreeBC"] == str(242 - 11)
    assert m["hrtKappaBC"] == mt.num(k_hand, 2)
    assert m["hrtDisagreeContentAgreePct"] == mt._pct(hC["share_read_correct"])
    assert m["hrtWithBDisagreeContentAgreePct"] == mt._pct(hB["share_read_correct"])
    for s in ("hrtDisagreeContentAgreePctLo", "hrtDisagreeContentAgreePctHi", "hrtAcOneBC", "hrtAcOneAC",
              "hrtKappaAB", "hrtRaterCContentAgreePct", "hrtRaterCCtrlWrongNAll", "hrtBothCorrectAC"):
        assert s in m
    assert not any(k.startswith("hra") for k in m)
    assert R["macros"] == m
    assert mt.third_rater_results(str(tmp_path / "nowhere")) is None


def test_third_rater_in_set_a_is_refused(env, assignment, capsys):
    tmp_path, kw, _ = env
    rows = rows_for(assignment, lambda r, i: "gold")
    for r in rows:
        if r[1] == "user3":
            r[2] = "A"
            break
    write(rows, tmp_path / "ex.csv")
    assert ha.run_three(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "set B only" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_third_rater_missing_answers_is_refused(env, assignment, capsys):
    tmp_path, kw, _ = env
    rows = rows_for(assignment, lambda r, i: "gold")
    drop = [j for j, r in enumerate(rows) if r[1] == "user3"][:3]
    rows = [r for j, r in enumerate(rows) if j not in drop]
    write(rows, tmp_path / "ex.csv")
    assert ha.run_three(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "third rater (user3) has no `first` row for 3 of 242" in capsys.readouterr().out


def test_no_third_rater_rows_is_refused(env, assignment, capsys):
    tmp_path, kw, _ = env
    write(rows_for(assignment, lambda r, i: "gold", raters=("A", "B")), tmp_path / "ex.csv")
    assert ha.run_three(export=str(tmp_path / "ex.csv"), **kw) == 1
    assert "no row of the third rater" in capsys.readouterr().out


def test_check_against_the_two_rater_export(env, assignment):
    tmp_path, kw, _ = env
    rows = rows_for(assignment, lambda r, i: "gold")
    write(rows, tmp_path / "ex3.csv")
    two = [r for r in rows if r[1] != "user3"]
    two[0][5] = "other"                                    # one first verdict of A differs in the two-rater file
    write(two, tmp_path / "ex2.csv")
    assert ha.run_three(export=str(tmp_path / "ex3.csv"), **{**kw, "two_rater_export": str(tmp_path / "ex2.csv")}) == 0
    ck = load(tmp_path)["two_rater_export_check"]
    assert ck == {"first_verdicts_compared": 484, "differ": 1, "absent": 0}


# ---------------------------------------------------------------- the third rater's CSV as the source of rater C


def rater3_csv(assignment, verdict_of, path, blank=()):
    ids = assignment.loc[assignment["rater_B"] == "1", "audit_id"].tolist()[::-1]      # any order
    rows = [{"order": j + 1, "audit_id": i, "question": "q", "reference_answer": "g", "model_output": "o",
             "verdict": "" if i in blank else verdict_of("C", i), "note": "free text, not read"}
            for j, i in enumerate(ids)]
    pd.DataFrame(rows, columns=list(ha.RATER3_COLUMNS)).to_csv(path, index=False)


def test_csv_source_matches_the_export_source(env, assignment):
    """The same verdicts through the CSV (with the two-rater export) and through the three-rater export give the same
    analysis; the CSV run names the third rater."""
    tmp_path, kw, _ = env
    a = assignment
    shared = sorted(a.loc[a["shared"] == "1", "audit_id"])

    def v(r, i):
        if i in shared[:4]:
            return {"A": "gold", "B": "other", "C": ("gold", "other", "unclear", "none")[shared.index(i)]}[r]
        return "gold"
    write(rows_for(a, v), tmp_path / "ex3.csv")
    assert ha.run_three(export=str(tmp_path / "ex3.csv"), **kw) == 0
    R1 = load(tmp_path)
    write(rows_for(a, v, raters=("A", "B")), tmp_path / "ex2.csv")
    rater3_csv(a, v, tmp_path / "r3.csv")
    kw2 = {**kw, "out": str(tmp_path / "out2"), "two_rater_export": str(tmp_path / "ex2.csv"),
           "rater3_csv": str(tmp_path / "r3.csv")}
    assert ha.run_three(export=str(tmp_path / "absent.csv"), **kw2) == 0
    R2 = json.load(open(tmp_path / "out2" / "three_raters.json", encoding="utf-8"))
    assert R2["source"] == "rater3 csv" and R2["third_rater"] == "C" and R1["source"] == "three-rater export"
    for k in ("counts", "pairwise", "majority", "ab_resolution", "headline", "per_rater", "first_verdict_counts"):
        assert R2[k] == R1[k], k
    assert R2["ab_resolution"]["sides_with_A"] == 1 and R2["ab_resolution"]["sides_with_B"] == 2
    assert "one of the authors" in (tmp_path / "out2" / "summary_three_raters.md").read_text(encoding="utf-8")
    assert "free text" not in json.dumps(R2)


def test_csv_blank_partial_and_both_sources(env, assignment, capsys):
    tmp_path, kw, _ = env
    a = assignment
    write(rows_for(a, lambda r, i: "gold", raters=("A", "B")), tmp_path / "ex2.csv")
    kw2 = {**kw, "two_rater_export": str(tmp_path / "ex2.csv"), "rater3_csv": str(tmp_path / "r3.csv")}
    every = set(a.loc[a["rater_B"] == "1", "audit_id"])
    rater3_csv(a, lambda r, i: "gold", tmp_path / "r3.csv", blank=every)            # the template as written
    assert ha.run_three(export=str(tmp_path / "absent.csv"), **kw2) == 2
    assert "not there yet" in capsys.readouterr().out
    rater3_csv(a, lambda r, i: "gold", tmp_path / "r3.csv", blank=sorted(every)[:2])  # two still blank
    assert ha.run_three(export=str(tmp_path / "absent.csv"), **kw2) == 1
    assert "2 answers have no verdict yet" in capsys.readouterr().out
    rater3_csv(a, lambda r, i: "maybe", tmp_path / "r3.csv")
    assert ha.run_three(export=str(tmp_path / "absent.csv"), **kw2) == 1
    assert "verdicts other than" in capsys.readouterr().out
    rater3_csv(a, lambda r, i: "Gold ", tmp_path / "r3.csv")                         # case and spaces ignored
    write(rows_for(a, lambda r, i: "gold"), tmp_path / "ex3.csv")
    assert ha.run_three(export=str(tmp_path / "ex3.csv"), **kw2) == 1                 # both sources present
    assert "Two third-rater sources" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
    assert ha.run_three(export=str(tmp_path / "absent.csv"), **kw2) == 0


def test_write_rater3_template(tmp_path, assignment):
    p = tmp_path / "t.csv"
    assert ha.write_rater3_template(str(p)) == 0
    t = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(t.columns) == list(ha.RATER3_COLUMNS) and len(t) == 242
    assert set(t["audit_id"]) == set(assignment.loc[assignment["rater_B"] == "1", "audit_id"])
    assert (t["verdict"] == "").all() and t["order"].tolist() == [str(j) for j in range(1, 243)]
    assert ha.read_rater3(str(p)) is None
    assert ha.write_rater3_template(str(p)) == 1                                        # never overwrites


def test_two_rater_outputs_unchanged():
    """The two-rater path on the real export (when the git-ignored key is here) reproduces results/human_audit/
    human_audit.json and summary.md, apart from the creation time."""
    if not (Path(ha.KEY).exists() and Path(ha.EXPORT).exists() and (ROOT / "results" / "human_audit" /
                                                                    "human_audit.json").exists()):
        pytest.skip("real key or two-rater results absent")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert ha.run(out=d) == 0
        new = json.load(open(Path(d) / "human_audit.json", encoding="utf-8"))
        old = json.load(open(ROOT / "results" / "human_audit" / "human_audit.json", encoding="utf-8"))
        new.pop("created"), old.pop("created")
        assert new == old
        assert (Path(d) / "summary.md").read_text(encoding="utf-8") == \
            (ROOT / "results" / "human_audit" / "summary.md").read_text(encoding="utf-8")
