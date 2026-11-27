r"""Two-rater human audit of answer reading (rating page in manifests/human_audit/).

Run from the repository root, once the `ratings` tab of the rating sheet "Rating audit: ratings" is exported
(File > Download > Comma-separated values, with the `ratings` tab open) to results/scoring_audit_v2/ratings_two_raters.csv:
    python scripts/run_human_audit.py

While the export is absent the script prints a message and exits with code 2. An export that fails validation is
refused with code 1 and a list of the problems; nothing is written.

The export (written by the rating page script, `record`): one row per action, columns
time_utc, rater, audit_id, event, verdict, flags. rater is A or B; event is first, change or flags; verdict is gold,
other, none or unclear on first and change rows and empty on flags rows; flags is a ';'-joined subset of changes,
in_reasoning, cut_off, odd_format. Rows are appended under a lock, so file order is the order of the actions.

Validation: the six columns are present; every rater, event, verdict and flag is a known value; every audit_id is in
manifests/human_audit/assignment.csv and assigned to the rater on that row; each rater has exactly one `first` row for
each answer assigned to them (none missing, none repeated); a `change` row comes after that rater's `first` row for the
answer.

Methods:
- Human rating. Each rater's `first` verdict, as the code correct (gold), not correct (other, none) or can't tell
  (unclear), the mapping of scripts/score_scoring_audit_v2.py.
- Agreement between raters, on the 100 shared answers: share with the same code and Cohen's kappa over the three codes
  (primary); the same on the four verdicts, and on correct / not correct over the answers both decided (secondary).
  Kappa is reported as undefined when the expected agreement is 1 (both raters use one and the same code throughout)
  or there is no answer to compare.
- Resolved human rating. A shared answer takes the code when both raters give the same code; otherwise it is excluded
  from the agreement figures below and counted. Every other answer takes its one rater's code. Can't tell is excluded
  and counted.
- Agreement of the resolved human rating with each reference: the strict scorer (score_a of
  data/scoring_audit_v2/key.csv, every task), the content scorer (score_b on GSM8K and BIG-Bench Hard), Cheng et al.'s
  extractor (score_b on the exemplar contrast) and an LLM rater's reading (results/scoring_audit_v2/llm_ratings.csv;
  the LLM rater's can't tell excluded and counted). Scopes: the GSM8K and BIG-Bench Hard answers, where both strict and
  content scores exist (the headline scope); every answer (strict, LLM rater); the answers where strict and content scoring disagree, and where they
  agree; each task. Each share has a 95% Wilson score interval, which treats the answers as independent (they share
  models and configurations, so the interval is descriptive).
- These are shares of the audit sample, which holds 12 answers where the two scorers disagree for every 4 where they
  agree in each stratum. They are not population agreement rates.
- Each rater alone: every answer the rater rated, no exclusion
  for disagreement with the other rater, agreement with the strict and content scorers on the GSM8K and BIG-Bench
  Hard answers and with the strict scorer on every answer; and accuracy on control answers, those both scorers of the
  key mark wrong (score_a and score_b both 0; the second scorer is content on GSM8K and BIG-Bench Hard, Cheng et al.'s
  extractor on the exemplar contrast) read as correct, and those both mark right read as not correct, on GSM8K and BIG-Bench Hard and
  on every answer. Wilson intervals as above.
- Rating times and changed verdicts are not reported: time_utc serves only the order
  checks of the validation, and change rows only the check that they follow a first row.

Writes results/human_audit/human_audit.json (with the macros of paper/macros_added.human_audit_macros), summary.md and
summary_clean.md.

Third rater: run with --three-raters once the `ratings` tab, now holding user3's rows as well, is exported to
results/scoring_audit_v2/ratings_three_raters.csv:
    python scripts/run_human_audit.py --three-raters
The two-rater outputs above are not touched. Methods:
- Layout: the two-rater export's columns. The third rater is the username `user3` (THIRD_RATER), rater C, who rates set
  B (the 242 answers of rater_B in assignment.csv, the 100 shared included); every user3 row must say set B. The rows of
  the other two usernames go through the two-rater validation unchanged; user3's rows through the same checks against
  set B. Rater A's and B's first verdicts are compared with ratings_two_raters.csv when it exists (reported, a check).
- Each rater alone for C: per_rater, as for A and B.
- Pairwise, A-B and A-C on the 100 shared answers, B-C on the 242 of set B: same code (share), Cohen's kappa and
  Gwet's AC1 over the three codes (AC1 with q = 3, the categories of the scale: Pe = sum_k pi_k (1 - pi_k) / (q - 1),
  pi_k the mean of the two raters' shares of code k); and, on the answers both decided, the 2x2 table of correct / not
  correct with kappa and AC1 (q = 2).
- Majority verdict on the 100 shared answers: the code at least two of the three raters give; no majority when all
  three differ.
- Two-of-three resolution of the A-B disagreements on shared answers: C's code sides with A or with B; unresolved when
  C's code matches neither.
- Headline with the resolved rating, on GSM8K and BIG-Bench Hard: the outputs the strict scorer marks wrong and the
  content scorer correct, the share read as correct (95% Wilson), each output rated by the majority on shared answers
  (no majority or a can't-tell majority excluded and counted) and by its single rating elsewhere, with C in place of B
  on set B (primary) and B in place of C beside it (the majority on shared answers is the same in both). Agreement with
  the content and strict scorers over all GSM8K and BIG-Bench Hard answers under the same two ratings, beside it.
- Rating times and changed verdicts are not reported.
Writes results/human_audit/three_raters.json (with the macros of paper/macros_third_rater.py, prefix hrt),
summary_three_raters.md, summary_three_raters_clean.md and tables/three_raters_*.csv. While the export is absent it
prints what to export and exits with code 2; a malformed export is refused with code 1.
Second source for rater C: rater C may fill
results/scoring_audit_v2/ratings_rater3.csv (columns order, audit_id, question, reference_answer, model_output,
verdict, note; verdict gold, other, none or unclear; blank template from --write-rater3-template, set B in the page's
order, no scorer verdict and no AI reading). Its verdict is C's first verdict; A and B come from ratings_two_raters.csv;
everything after is the same code. A CSV with no verdict filled counts as absent, a partly filled one is refused, and
the CSV beside a three-rater export is refused (one source at a time). `note` is not read.
"""
import argparse
import json
import math
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXPORT = os.path.join(ROOT, "results", "scoring_audit_v2", "ratings_two_raters.csv")
ASSIGNMENT = os.path.join(ROOT, "manifests", "human_audit", "assignment.csv")
KEY = os.path.join(ROOT, "data", "scoring_audit_v2", "key.csv")
LLM_RATINGS = os.path.join(ROOT, "results", "scoring_audit_v2", "llm_ratings.csv")
MANIFEST = os.path.join(ROOT, "manifests", "scoring_audit_v2.json")
OUT = os.path.join(ROOT, "results", "human_audit")
EXPORT3 = os.path.join(ROOT, "results", "scoring_audit_v2", "ratings_three_raters.csv")
THIRD_RATER = "user3"
# the third rating may come as a CSV filled by rater C in place of the page
RATER3_CSV = os.path.join(ROOT, "results", "scoring_audit_v2", "ratings_rater3.csv")
RATER3_COLUMNS = ("order", "audit_id", "question", "reference_answer", "model_output", "verdict", "note")
RATER3_NAME = "C"
ITEMS = os.path.join(ROOT, "data", "human_audit", "rating_items.json")

COLUMNS = ("time_utc", "rater", "audit_id", "event", "verdict", "flags")
RATERS = ("A", "B")
EVENTS = ("first", "change", "flags")
VERDICTS = ("gold", "other", "none", "unclear")
FLAGS = ("changes", "in_reasoning", "cut_off", "odd_format")
CODE = {"gold": "correct", "other": "not_correct", "none": "not_correct", "unclear": "cant_tell"}
CODES = ("correct", "not_correct", "cant_tell")
CONTENT_TASKS = ("gsm8k", "bbh")
TASK_ORDER = ("gsm8k", "bbh", "exemplar")
Z = 1.959963984540054
MAX_LISTED = 8


class ExportError(ValueError):
    """The export does not match the assignment or the values Code.gs writes."""


# ---------------------------------------------------------------- reading and validation


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _listed(xs):
    xs = list(xs)
    s = ", ".join(str(x) for x in xs[:MAX_LISTED])
    return s + (f" and {len(xs) - MAX_LISTED} more" if len(xs) > MAX_LISTED else "")


def read_assignment(path):
    a = read_csv(path)
    need = ["audit_id"] + [f"rater_{r}" for r in RATERS] + ["shared"]
    missing = [c for c in need if c not in a.columns]
    if missing:
        raise ExportError(f"{path} lacks columns {missing}")
    a = a[need].copy()
    for c in need[1:]:
        a[c] = a[c].str.strip().map({"1": True, "0": False})
        if a[c].isna().any():
            raise ExportError(f"{path}: column {c} holds values other than 0/1")
    if a["audit_id"].duplicated().any():
        raise ExportError(f"{path}: duplicate audit_id")
    if not (a["shared"] == (a["rater_A"] & a["rater_B"])).all():
        raise ExportError(f"{path}: `shared` is not rater_A and rater_B")
    return a


def validate_export(ex, assignment):
    """The export as read (strings) -> a clean frame in file order with column `row` (1 = first data row). Raises
    ExportError listing every problem found."""
    if "set" in ex.columns:
        # The sheet's later layout: `rater` holds the rater's username and `set` the assignment (A or B). The set is
        # the rater identity used here; each username must belong to one set and each set to one username.
        ex = ex.copy()
        ex["rater_name"] = ex["rater"].astype(str).str.strip()
        ex["rater"] = ex["set"].astype(str).str.strip()
        pairs = ex.loc[ex["rater_name"].ne("") | ex["rater"].ne(""), ["rater_name", "rater"]].drop_duplicates()
        if pairs["rater_name"].duplicated().any() or pairs["rater"].duplicated().any():
            raise ExportError("the export is refused:\n  - usernames and sets do not match one to one: "
                              + "; ".join(f"{n}->{s}" for n, s in pairs.itertuples(index=False)))
    missing = [c for c in COLUMNS if c not in ex.columns]
    if missing:
        raise ExportError(f"the export lacks columns {missing} (found {list(ex.columns)}); expected the `ratings` tab "
                          f"of the sheet, header {', '.join(COLUMNS)}")
    x = ex[list(COLUMNS)].copy()
    for c in COLUMNS:
        x[c] = x[c].astype(str).str.strip()
    x["row"] = range(2, len(x) + 2)                      # the sheet's row number (row 1 is the header)
    x = x[~x[list(COLUMNS)].eq("").all(axis=1)].reset_index(drop=True)
    problems = []

    def bad(mask, what):
        if mask.any():
            problems.append(f"{int(mask.sum())} rows with {what} (sheet rows {_listed(x.loc[mask, 'row'])})")

    bad(~x["rater"].isin(RATERS), f"a rater other than {'/'.join(RATERS)}")
    bad(~x["event"].isin(EVENTS), f"an event other than {', '.join(EVENTS)}")
    rated = x["event"].isin(("first", "change"))
    bad(rated & ~x["verdict"].isin(VERDICTS), f"a first/change verdict other than {', '.join(VERDICTS)}")
    bad((x["event"] == "flags") & x["verdict"].ne(""), "a verdict on a flags row")
    unknown_flag = x["flags"].map(lambda s: any(f.strip() not in FLAGS for f in s.split(";") if f.strip()))
    bad(unknown_flag, f"a flag other than {', '.join(FLAGS)}")
    known = set(assignment["audit_id"])
    bad(~x["audit_id"].isin(known), "an audit_id that is not in assignment.csv")
    a = assignment.set_index("audit_id")
    ok = x["rater"].isin(RATERS) & x["audit_id"].isin(known)
    theirs = {(r, i) for r in RATERS for i in a.index[a[f"rater_{r}"]]}
    not_theirs = ok & ~pd.Series([p in theirs for p in zip(x["rater"], x["audit_id"])], index=x.index)
    bad(not_theirs, "an answer not assigned to that rater")
    ok &= ~not_theirs
    first = x[(x["event"] == "first") & ok]
    for r in RATERS:
        f = first[first["rater"] == r]
        dup = f["audit_id"][f["audit_id"].duplicated(keep=False)]
        if len(dup):
            problems.append(f"rater {r} has more than one `first` row for {dup.nunique()} answers ({_listed(sorted(set(dup)))})")
        assigned = set(a.index[a[f"rater_{r}"]])
        absent = sorted(assigned - set(f["audit_id"]))
        if absent:
            problems.append(f"rater {r} has no `first` row for {len(absent)} of {len(assigned)} assigned answers "
                            f"({_listed(absent)})")
    fr = first.drop_duplicates(["rater", "audit_id"]).set_index(["rater", "audit_id"])["row"]
    ch = x[(x["event"] == "change") & ok]
    early = [int(r["row"]) for _, r in ch.iterrows()
             if (r["rater"], r["audit_id"]) not in fr.index or r["row"] < fr[(r["rater"], r["audit_id"])]]
    if early:
        problems.append(f"{len(early)} `change` rows come before that rater's `first` row for the answer "
                        f"(sheet rows {_listed(early)})")
    if problems:
        raise ExportError("the export is refused:\n  - " + "\n  - ".join(problems))
    return x


def read_key(path, manifest):
    k = read_csv(path)
    need = ["audit_id", "task", "agreement", "score_a", "score_b"]
    missing = [c for c in need if c not in k.columns]
    if missing:
        raise ExportError(f"{path} lacks columns {missing}")
    k = k[need].copy()
    for s in ("score_a", "score_b"):
        v = k[s].str.strip().str.lower().map({"1": True, "0": False, "true": True, "false": False})
        if v.isna().any():
            raise ExportError(f"{path}: {s} holds values other than 0/1")
        k[s] = v.astype(bool)
    if manifest is not None:
        for task, sc in manifest["scorers"].items():
            if task in CONTENT_TASKS and sc.get("score_b") != "content":
                raise ExportError(f"manifest: the second scorer of {task} is {sc.get('score_b')}, not content")
    return k


def read_llm(path):
    c = read_csv(path)
    if "audit_id" not in c.columns or "verdict" not in c.columns:
        raise ExportError(f"{path} lacks audit_id or verdict")
    c = c[["audit_id", "verdict"]].copy()
    c["verdict"] = c["verdict"].str.strip().str.lower()
    if not c["verdict"].isin(VERDICTS).all():
        raise ExportError(f"{path}: unknown verdicts {sorted(set(c['verdict']) - set(VERDICTS))}")
    if c["audit_id"].duplicated().any():
        raise ExportError(f"{path}: duplicate audit_id")
    return c


# ---------------------------------------------------------------- statistics


def wilson(k, n, z=Z):
    """95% Wilson score interval for k of n; None when n = 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cohen_kappa(a, b):
    """Cohen's kappa of two equal-length label lists. Returns (kappa, observed, expected, status); kappa is None with
    status 'undefined: ...' when there is nothing to compare or the expected agreement is 1."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None, None, None, "undefined: no answer to compare"
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = sorted(set(a) | set(b))
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return None, po, pe, "undefined: expected agreement is 1 (both raters use one and the same category throughout)"
    return (po - pe) / (1 - pe), po, pe, "defined"


def agreement_block(a, b):
    k, po, pe, status = cohen_kappa(a, b)
    n = len(list(a))
    agree = sum(x == y for x, y in zip(a, b))
    return {"n": n, "agree": agree, "share": None if n == 0 else agree / n, "kappa": k, "kappa_status": status,
            "expected_agreement": pe}


def reference_block(human, ref):
    """human, ref: boolean Series on the same index (human decided; ref decided)."""
    n = int(len(human))
    agree = int((human == ref).sum())
    ci = wilson(agree, n)
    return {"n": n, "agree": agree, "share": None if n == 0 else agree / n, "ci95_wilson": ci,
            "human_correct_ref_correct": int((human & ref).sum()), "human_correct_ref_not": int((human & ~ref).sum()),
            "human_not_ref_correct": int((~human & ref).sum()), "human_not_ref_not": int((~human & ~ref).sum())}


# ---------------------------------------------------------------- analysis


def analyse(x, assignment, key, llm):
    first = x[x["event"] == "first"].drop_duplicates(["rater", "audit_id"])
    first = first.set_index(["rater", "audit_id"])["verdict"]

    # agreement between raters on the shared answers
    shared = sorted(assignment.loc[assignment["shared"], "audit_id"])
    va = [first[("A", i)] for i in shared]
    vb = [first[("B", i)] for i in shared]
    ca, cb = [CODE[v] for v in va], [CODE[v] for v in vb]
    both = [(p, q) for p, q in zip(ca, cb) if p != "cant_tell" and q != "cant_tell"]
    table = {f"A_{p}__B_{q}": sum(1 for s, t in zip(ca, cb) if (s, t) == (p, q)) for p in CODES for q in CODES}
    raters = {"shared": len(shared), "codes": agreement_block(ca, cb), "verdicts": agreement_block(va, vb),
              "decided_both": agreement_block([p for p, _ in both], [q for _, q in both]), "table_codes": table,
              "same_code_different_verdict": sum(1 for p, q, s, t in zip(ca, cb, va, vb) if p == q and s != t)}
    disagreements = [{"audit_id": i, "A": s, "B": t} for i, s, t, p, q in zip(shared, va, vb, ca, cb) if p != q]

    # resolved human rating per answer
    rows = []
    for _, r in assignment.iterrows():
        i = r["audit_id"]
        if r["shared"]:
            p, q = CODE[first[("A", i)]], CODE[first[("B", i)]]
            rows.append({"audit_id": i, "shared": True, "resolved": p if p == q else None,
                         "status": "resolved" if p == q else "raters_disagree"})
        else:
            rr = "A" if r["rater_A"] else "B"
            rows.append({"audit_id": i, "shared": False, "resolved": CODE[first[(rr, i)]], "status": "resolved"})
    H = pd.DataFrame(rows)
    H.loc[H["resolved"] == "cant_tell", "status"] = "cant_tell"
    missing_key = sorted(set(H["audit_id"]) - set(key["audit_id"]))
    if missing_key:
        raise ExportError(f"the key lacks {len(missing_key)} assigned answers ({_listed(missing_key)})")
    H = H.merge(key, on="audit_id", how="left", validate="one_to_one")
    H = H.merge(llm.rename(columns={"verdict": "llm_verdict"}), on="audit_id", how="left", validate="one_to_one")
    if H["llm_verdict"].isna().any():
        raise ExportError(f"The LLM rater's ratings lack {int(H['llm_verdict'].isna().sum())} assigned answers")
    H["human"] = H["resolved"].map({"correct": True, "not_correct": False})
    H["llm"] = H["llm_verdict"].map(CODE).map({"correct": True, "not_correct": False})
    counts = {"answers": int(len(H)), "shared": int(H["shared"].sum()), "rater_disagreements_excluded":
              int((H["status"] == "raters_disagree").sum()), "cant_tell_excluded": int((H["status"] == "cant_tell").sum()),
              "decided": int(H["human"].notna().sum()),
              "decided_strict_and_content": int((H["human"].notna() & H["task"].isin(CONTENT_TASKS)).sum()),
              "llm_cant_tell": int(H["llm"].isna().sum())}

    def scope_mask(scope):
        sc = H["task"].isin(CONTENT_TASKS)
        if scope == "strict_and_content":
            return sc
        if scope == "all":
            return pd.Series(True, index=H.index)
        if scope == "strict_content_disagree":
            return sc & (H["agreement"] == "disagree")
        if scope == "strict_content_agree":
            return sc & (H["agreement"] == "agree")
        if scope.startswith("task:"):
            return H["task"] == scope[5:]
        raise KeyError(scope)

    refs = {}
    scopes = ["strict_and_content", "strict_content_disagree", "strict_content_agree", "all"] + \
        [f"task:{t}" for t in sorted(H["task"].unique(), key=lambda t: (TASK_ORDER.index(t) if t in TASK_ORDER else 9, t))]
    for scope in scopes:
        m = scope_mask(scope) & H["human"].notna()
        d = H[m]
        out = {"decided": int(len(d))}
        h = d["human"].astype(bool)
        out["strict"] = reference_block(h, d["score_a"])
        tasks = set(d["task"])
        if tasks and tasks <= set(CONTENT_TASKS):
            out["content"] = reference_block(h, d["score_b"])
        if scope.startswith("task:") and scope[5:] not in CONTENT_TASKS:
            out["second_scorer"] = {"name": "cheng", **reference_block(h, d["score_b"])}
        cd = d["llm"].notna()
        out["llm"] = {**reference_block(h[cd], d.loc[cd, "llm"].astype(bool)),
                         "llm_cant_tell": int((~cd).sum())}
        refs[scope] = out

    verdict_counts = {r: {v: int(sum(first[(r, i)] == v for i in first.loc[r].index)) for v in VERDICTS}
                      for r in RATERS}
    # rating times and changed verdicts are not reported (time_utc and change rows serve only validate_export's checks)
    return {"counts": counts, "raters": raters, "rater_disagreements": disagreements, "references": refs,
            "per_rater": per_rater(first, key), "first_verdict_counts": verdict_counts,
            "flag_rows": int((x["event"] == "flags").sum())}


def per_rater(first, key, raters=RATERS):
    """Each rater alone: every answer the rater rated (first verdicts), no exclusion for disagreement with
    the other rater; can't tell excluded and counted. Agreement with the scorers, and the control answers."""
    out = {}
    for r in raters:
        f = first.loc[r].rename("verdict").reset_index().merge(key, on="audit_id", how="left", validate="one_to_one")
        f["code"] = f["verdict"].map(CODE)
        dec = f[f["code"] != "cant_tell"]
        h = dec["code"] == "correct"
        gb = dec["task"].isin(CONTENT_TASKS)
        blk = {"answers": int(len(f)), "cant_tell": int((f["code"] == "cant_tell").sum()),
               "strict_and_content": {"decided": int(gb.sum()), "content": reference_block(h[gb], dec.loc[gb, "score_b"]),
                                      "strict": reference_block(h[gb], dec.loc[gb, "score_a"])},
               "all": {"decided": int(len(dec)), "strict": reference_block(h, dec["score_a"])},
               "controls": {}}
        for scope, m in (("strict_and_content", gb), ("all", pd.Series(True, index=dec.index))):
            bw = m & ~dec["score_a"] & ~dec["score_b"]
            br = m & dec["score_a"] & dec["score_b"]
            k1, n1 = int((h & bw).sum()), int(bw.sum())
            k2, n2 = int((~h & br).sum()), int(br.sum())
            blk["controls"][scope] = {
                "both_scorers_wrong": {"n": n1, "read_correct": k1, "share": k1 / n1 if n1 else None,
                                       "ci95_wilson": wilson(k1, n1)},
                "both_scorers_right": {"n": n2, "read_not_correct": k2, "share": k2 / n2 if n2 else None,
                                       "ci95_wilson": wilson(k2, n2)}}
        out[r] = blk
    return out


# ---------------------------------------------------------------- output


def pct(x, nd=1):
    return "--" if x is None else f"{100 * x:.{nd}f}"


def ci_txt(ci):
    return "--" if ci is None else f"[{pct(ci[0])}, {pct(ci[1])}]"


def kappa_txt(b):
    return f"{b['kappa']:.2f}" if b["kappa"] is not None else b["kappa_status"]


SCOPE_LABEL = {"strict_and_content": "GSM8K and BIG-Bench Hard (strict and content scores)",
               "strict_content_disagree": "strict and content scoring disagree",
               "strict_content_agree": "strict and content scoring agree",
               "all": "every answer", "task:gsm8k": "GSM8K", "task:bbh": "BIG-Bench Hard",
               "task:exemplar": "exemplars (GSM8K, zero- and eight-shot chain of thought)"}


def summary(R):
    c, ra = R["counts"], R["raters"]
    L = ["# Human audit, two raters", "",
         f"{c['answers']} answers of the second scoring audit, each rated by one of two raters and {c['shared']} by both. "
         "The rating page showed no AI reading. The human rating is each rater's first "
         "verdict, coded correct (matches the correct answer), not correct (a different answer, no final answer) or "
         "can't tell. Method: `scripts/run_human_audit.py` docstring.", "",
         "## Agreement between raters", "",
         f"On the {ra['shared']} shared answers:", "",
         f"- Same code: {ra['codes']['agree']} of {ra['codes']['n']} ({pct(ra['codes']['share'])}%); Cohen's kappa "
         f"{kappa_txt(ra['codes'])}.",
         f"- Same verdict of the four: {ra['verdicts']['agree']} of {ra['verdicts']['n']} "
         f"({pct(ra['verdicts']['share'])}%); Cohen's kappa {kappa_txt(ra['verdicts'])}.",
         f"- Correct / not correct, on the {ra['decided_both']['n']} answers both decided: {ra['decided_both']['agree']} "
         f"agree ({pct(ra['decided_both']['share'])}%); Cohen's kappa {kappa_txt(ra['decided_both'])}.",
         f"- Same code, different verdict (a different answer against no final answer): "
         f"{ra['same_code_different_verdict']}.", "",
         "| Rater A \\ Rater B | correct | not correct | can't tell |", "|---|---:|---:|---:|"]
    lab = {"correct": "correct", "not_correct": "not correct", "cant_tell": "can't tell"}
    for p in CODES:
        L.append(f"| {lab[p]} | " + " | ".join(str(ra["table_codes"][f"A_{p}__B_{q}"]) for q in CODES) + " |")
    L += ["", f"Shared answers where the raters disagree: {c['rater_disagreements_excluded']}"
          + (" (" + "; ".join(f"{d['audit_id']}: A {d['A']}, B {d['B']}" for d in R["rater_disagreements"]) + ")"
             if R["rater_disagreements"] else "") + ". They are excluded below.", "",
          "## Agreement of the human rating with the scorers and with the LLM rater", "",
          f"Resolved human ratings: {c['decided']} of {c['answers']} answers decided; excluded: "
          f"{c['rater_disagreements_excluded']} where the raters disagree, {c['cant_tell_excluded']} can't tell. "
          f"The LLM rater's can't tell: {c['llm_cant_tell']} (excluded from its row).", "",
          "Shares of the audit sample, which holds 12 answers where the two scorers disagree for every 4 where they "
          "agree in each stratum; they are not population agreement rates. Intervals: 95% Wilson score, treating "
          "answers as independent (descriptive).", "",
          "| Scope | Decided | Strict scorer | Content scorer | LLM rater's reading |", "|---|---:|---|---|---|"]
    for scope, o in R["references"].items():
        cell = []
        for k in ("strict", "content", "llm"):
            b = o.get(k)
            cell.append("--" if b is None else f"{b['agree']}/{b['n']} = {pct(b['share'])}% {ci_txt(b['ci95_wilson'])}")
        L.append(f"| {SCOPE_LABEL.get(scope, scope)} | {o['decided']} | " + " | ".join(cell) + " |")
    for scope, o in R["references"].items():
        b = o.get("second_scorer")
        if b:
            L.append(f"\n{SCOPE_LABEL.get(scope, scope)}, Cheng et al.'s extractor: {b['agree']}/{b['n']} = "
                     f"{pct(b['share'])}% {ci_txt(b['ci95_wilson'])}.")
    d = R["references"]["strict_content_disagree"]
    L += ["", "On the answers where strict and content scoring disagree, the strict scorer marks every one wrong and the "
          "content scorer every one right in this sample, so the content scorer's agreement is the share the humans "
          f"read as correct: {d['content']['human_correct_ref_correct']} of {d['content']['n']}." if d.get("content") and
          d["content"]["human_not_ref_not"] + d["content"]["human_correct_ref_not"] == 0 and           d["strict"]["human_correct_ref_correct"] + d["strict"]["human_not_ref_correct"] == 0 else None, "",
          "## First verdicts", "", "| Rater | gold | other | none | unclear |", "|---|---:|---:|---:|---:|"]
    for r, v in R["first_verdict_counts"].items():
        L.append(f"| {r} | " + " | ".join(str(v[k]) for k in VERDICTS) + " |")
    L += ["", "## Each rater alone", "",
          "Every answer the rater rated, first verdicts, no exclusion for disagreement with the other rater. Control "
          "answers: both scorers of the key mark the answer wrong (it should read as not correct) or both mark it "
          "right; the second scorer is content on GSM8K and BIG-Bench Hard and Cheng et al.'s extractor on the exemplar contrast.", "",
          "| Rater | Answers | Can't tell | Content scorer, GSM8K and BIG-Bench Hard | Strict scorer, GSM8K and "
          "BIG-Bench Hard | Strict scorer, every answer |", "|---|---:|---:|---|---|---|"]
    for r, b in R["per_rater"].items():
        sc, al = b["strict_and_content"], b["all"]
        L.append(f"| {r} | {b['answers']} | {b['cant_tell']} | " + " | ".join(
            f"{x['agree']}/{x['n']} = {pct(x['share'])}% {ci_txt(x['ci95_wilson'])}"
            for x in (sc["content"], sc["strict"], al["strict"])) + " |")
    L += ["", "| Rater | Scope | Both scorers wrong, read as correct | Both scorers right, read as not correct |",
          "|---|---|---|---|"]
    for r, b in R["per_rater"].items():
        for scope, lab in (("strict_and_content", "GSM8K and BIG-Bench Hard"), ("all", "every answer")):
            cw, cr = b["controls"][scope]["both_scorers_wrong"], b["controls"][scope]["both_scorers_right"]
            L.append(f"| {r} | {lab} | {cw['read_correct']} of {cw['n']} ({pct(cw['share'])}%) | "
                     f"{cr['read_not_correct']} of {cr['n']} ({pct(cr['share'])}%) |")
    L += ["", f"Flag rows: {R['flag_rows']}."]
    return "\n".join(x for x in L if x is not None) + "\n"


def run(export=EXPORT, out=OUT, assignment=ASSIGNMENT, key=KEY, llm=LLM_RATINGS, manifest=MANIFEST):
    """Returns the exit code: 0 written, 1 refused, 2 export absent."""
    if not os.path.exists(export):
        print(f"The export is not there yet: {export}\nExport the `ratings` tab of the sheet \"Rating audit: ratings\" "
              "(File > Download > Comma-separated values) to that path, then run this script again.")
        return 2
    if not os.path.exists(key):
        print(f"{key} is missing; it is written by scripts/draw_scoring_audit_v2.py (git-ignored).")
        return 1
    try:
        a = read_assignment(assignment)
        x = validate_export(read_csv(export), a)
        man = json.load(open(manifest, encoding="utf-8")) if manifest and os.path.exists(manifest) else None
        R = analyse(x, a, read_key(key, man), read_llm(llm))
    except ExportError as e:
        print(str(e))
        return 1
    R = {"analysis": "two-rater human audit",
         "created": time.strftime("%Y-%m-%d %H:%M:%S"), "export": os.path.relpath(export, ROOT)
         if os.path.abspath(export).startswith(ROOT) else export, "interval": "95% Wilson score", **R}
    mpath = os.path.join(ROOT, "paper", "macros_added.py")   # paper/ is untracked; without it the JSON has no macros
    if os.path.exists(mpath):
        import importlib.util
        spec = importlib.util.spec_from_file_location("macros_added_hra", mpath)
        ma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ma)
        R["macros"] = ma.human_audit_macros(R)
        if hasattr(ma, "human_audit_rater_macros"):         # each rater alone and the control answers
            R["macros"].update(ma.human_audit_rater_macros(R))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "human_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    txt = summary(R)
    for fn in ("summary.md", "summary_clean.md"):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return 0


# ---------------------------------------------------------------- third rater

PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
BINARY = ("correct", "not_correct")


def gwet_ac1(a, b, categories):
    """Gwet's AC1 of two equal-length label lists over the scale's categories (q = len(categories) >= 2). Returns
    (ac1, observed, expected, status); ac1 is None with status 'undefined: ...' when there is nothing to compare."""
    a, b = list(a), list(b)
    n, q = len(a), len(categories)
    if q < 2:
        raise ValueError("AC1 needs at least two categories")
    if n == 0:
        return None, None, None, "undefined: no answer to compare"
    po = sum(x == y for x, y in zip(a, b)) / n
    pi = [(a.count(c) + b.count(c)) / (2 * n) for c in categories]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    if abs(1.0 - pe) < 1e-12:
        return None, po, pe, "undefined: expected agreement is 1"
    return (po - pe) / (1 - pe), po, pe, "defined"


def pair_block(a, b, categories):
    a, b = list(a), list(b)
    blk = agreement_block(a, b)
    ac, _, pe, st = gwet_ac1(a, b, categories)
    blk.update({"ac1": ac, "ac1_status": st, "ac1_expected_agreement": pe, "ac1_categories": len(categories)})
    return blk


def split_three(ex, third=THIRD_RATER):
    """The three-rater export -> (rows of the other two raters, rows of the third). The third rater is the username
    `third` in column `rater` with every row in set B; in the layout without `set`, the rater written C (returned as B,
    for validation against set B)."""
    ex = ex.copy()
    name = ex["rater"].astype(str).str.strip()
    if "set" in ex.columns:
        is3 = name == third
        if not is3.any():
            raise ExportError(f"the export is refused:\n  - no row of the third rater `{third}`")
        sets = set(ex.loc[is3, "set"].astype(str).str.strip())
        if sets != {"B"}:
            raise ExportError(f"the export is refused:\n  - the third rater `{third}` has rows in sets {sorted(sets)}; "
                              f"{third} rates set B only")
        c = ex[is3].copy()
    else:
        is3 = name == "C"
        if not is3.any():
            raise ExportError("the export is refused:\n  - no row of the third rater C")
        c = ex[is3].copy()
        c["rater"] = "B"
    return ex[~is3].copy(), c


def validate_three(ex, assignment, third=THIRD_RATER):
    """-> (rows of A and B as validate_export returns them, rows of C). C is checked as the one rater of set B."""
    two, c = split_three(ex, third)
    x2 = validate_export(two, assignment)
    a_c = assignment.copy()
    a_c["rater_A"] = False
    a_c["shared"] = False
    try:
        xc = validate_export(c, a_c)
    except ExportError as e:
        raise ExportError(str(e).replace("rater B ", f"the third rater ({third}) "))
    xc["rater"] = "C"
    return x2, xc


def read_rater3(path):
    """The third rater's CSV (RATER3_COLUMNS; verdict gold, other, none or unclear) -> rows in the three-rater export
    layout under the username RATER3_NAME, set B, event `first`; None while no verdict is filled in (the template as
    written). A partly filled file, an unknown verdict or a repeated audit_id is refused. `note` is not read."""
    d = read_csv(path)
    missing = [c for c in RATER3_COLUMNS if c not in d.columns]
    if missing:
        raise ExportError(f"{path} lacks columns {missing} (expected {', '.join(RATER3_COLUMNS)})")
    d = d[list(RATER3_COLUMNS)].copy()
    d["audit_id"] = d["audit_id"].astype(str).str.strip()
    d["verdict"] = d["verdict"].astype(str).str.strip().str.lower()
    d = d[d["audit_id"].ne("")].reset_index(drop=True)
    blank = d["verdict"].eq("")
    if blank.all():
        return None
    problems = []
    if blank.any():
        problems.append(f"{int(blank.sum())} answers have no verdict yet ({_listed(d.loc[blank, 'audit_id'])})")
    unknown = ~blank & ~d["verdict"].isin(VERDICTS)
    if unknown.any():
        problems.append(f"{int(unknown.sum())} verdicts other than {', '.join(VERDICTS)} "
                        f"({_listed(d.loc[unknown, 'audit_id'] + '=' + d.loc[unknown, 'verdict'])})")
    dup = d["audit_id"][d["audit_id"].duplicated(keep=False)]
    if len(dup):
        problems.append(f"{dup.nunique()} answers listed more than once ({_listed(sorted(set(dup)))})")
    if problems:
        raise ExportError(f"the third-rater CSV {path} is refused:\n  - " + "\n  - ".join(problems))
    n = len(d)
    return pd.DataFrame({"time_utc": [""] * n, "rater": [RATER3_NAME] * n, "set": ["B"] * n,
                         "audit_id": d["audit_id"].tolist(), "event": ["first"] * n, "verdict": d["verdict"].tolist(),
                         "flags": [""] * n})


def write_rater3_template(path=RATER3_CSV, items=ITEMS):
    """The blank CSV for the third rater: set B's 242 answers in the page's order for set B (rating_items.json), with the question,
    reference answer and model output the page shows; no scorer verdict and no AI reading. Never overwrites."""
    if os.path.exists(path):
        print(f"{path} exists; not overwritten.")
        return 1
    R = json.load(open(items, encoding="utf-8"))
    rows = [{"order": j + 1, "audit_id": i, "question": R["items"][i]["q"], "reference_answer": R["items"][i]["g"],
             "model_output": R["items"][i]["o"], "verdict": "", "note": ""} for j, i in enumerate(R["B"])]
    pd.DataFrame(rows, columns=list(RATER3_COLUMNS)).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {path}: {len(rows)} answers of set B; fill `verdict` with gold, other, none or unclear")
    return 0


def first_verdicts(x):
    f = x[x["event"] == "first"].drop_duplicates(["rater", "audit_id"])
    return f.set_index(["rater", "audit_id"])["verdict"]


def majority(codes):
    """The code at least two of the three give, else None."""
    for c in CODES:
        if sum(x == c for x in codes) >= 2:
            return c
    return None


def analyse_three(x2, xc, assignment, key, two_rater_first=None):
    first = pd.concat([first_verdicts(x2), first_verdicts(xc)])
    shared = sorted(assignment.loc[assignment["shared"], "audit_id"])
    set_b = sorted(assignment.loc[assignment["rater_B"], "audit_id"])
    scope_ids = {("A", "B"): ("shared", shared), ("A", "C"): ("shared", shared), ("B", "C"): ("set B", set_b)}
    pairwise = {}
    for p, q in PAIRS:
        where, ids = scope_ids[(p, q)]
        cp = [CODE[first[(p, i)]] for i in ids]
        cq = [CODE[first[(q, i)]] for i in ids]
        both = [(s, t) for s, t in zip(cp, cq) if s != "cant_tell" and t != "cant_tell"]
        pairwise[p + q] = {
            "answers": where, "n": len(ids),
            "codes": pair_block(cp, cq, CODES),
            "decided_both": pair_block([s for s, _ in both], [t for _, t in both], BINARY),
            "table_codes": {f"{p}_{s}__{q}_{t}": sum(1 for u, v in zip(cp, cq) if (u, v) == (s, t))
                            for s in CODES for t in CODES},
            "table_2x2": {"both_correct": sum(1 for u, v in both if u == v == "correct"),
                          "first_only_correct": sum(1 for u, v in both if u == "correct" and v != "correct"),
                          "second_only_correct": sum(1 for u, v in both if u != "correct" and v == "correct"),
                          "both_not_correct": sum(1 for u, v in both if u == v == "not_correct")}}

    # majority verdict on the shared answers
    maj = {}
    for i in shared:
        cs = [CODE[first[(r, i)]] for r in ("A", "B", "C")]
        maj[i] = {"codes": cs, "majority": majority(cs), "unanimous": len(set(cs)) == 1}
    mv = [v["majority"] for v in maj.values()]
    majority_blk = {"n": len(shared), "unanimous": sum(v["unanimous"] for v in maj.values()),
                    "two_of_three": sum(1 for v in maj.values() if v["majority"] is not None and not v["unanimous"]),
                    "no_majority": sum(1 for v in mv if v is None),
                    **{f"majority_{c}": sum(1 for v in mv if v == c) for c in CODES}}

    # two-of-three resolution of the A-B disagreements
    res = []
    for i in shared:
        a, b, c = maj[i]["codes"]
        if a == b:
            continue
        side = "A" if c == a else "B" if c == b else None
        res.append({"audit_id": i, "A": first[("A", i)], "B": first[("B", i)], "C": first[("C", i)],
                    "sides_with": side, "resolved": c if side else None})
    resolution = {"disagreements": len(res), "resolved": sum(1 for r in res if r["sides_with"]),
                  "unresolved": sum(1 for r in res if not r["sides_with"]),
                  "sides_with_A": sum(1 for r in res if r["sides_with"] == "A"),
                  "sides_with_B": sum(1 for r in res if r["sides_with"] == "B"),
                  **{f"resolved_{c}": sum(1 for r in res if r["resolved"] == c) for c in CODES},
                  "answers": res}

    # headline with the resolved rating
    missing_key = sorted(set(assignment["audit_id"]) - set(key["audit_id"]))
    if missing_key:
        raise ExportError(f"the key lacks {len(missing_key)} assigned answers ({_listed(missing_key)})")
    K = key.set_index("audit_id")
    headline = {}
    for variant, setb in (("C_for_B", "C"), ("B_for_C", "B")):
        rows = []
        for _, r in assignment.iterrows():
            i = r["audit_id"]
            if r["shared"]:
                code, how = maj[i]["majority"], "majority"
            elif r["rater_B"]:
                code, how = CODE[first[(setb, i)]], setb
            else:
                code, how = CODE[first[("A", i)]], "A"
            rows.append({"audit_id": i, "code": code, "how": how})
        H = pd.DataFrame(rows).join(K, on="audit_id")
        H["status"] = H["code"].map({"correct": "decided", "not_correct": "decided", "cant_tell": "cant_tell"}) \
            .fillna("no_majority")
        H["human"] = H["code"].map({"correct": True, "not_correct": False})
        gb = H["task"].isin(CONTENT_TASKS)
        blk = {}
        for scope, m in (("strict_wrong_content_correct", gb & ~H["score_a"] & H["score_b"]),
                         ("strict_and_content", gb)):
            d = H[m]
            dec = d[d["human"].notna()]
            h = dec["human"].astype(bool)
            k, n = int(h.sum()), int(len(dec))
            o = {"answers": int(len(d)), "decided": n,
                 "no_majority_excluded": int((d["status"] == "no_majority").sum()),
                 "cant_tell_excluded": int((d["status"] == "cant_tell").sum()),
                 "by_majority": int((dec["how"] == "majority").sum()),
                 "content": reference_block(h, dec["score_b"]), "strict": reference_block(h, dec["score_a"])}
            if scope == "strict_wrong_content_correct":
                o.update({"read_correct": k, "share_read_correct": k / n if n else None, "ci95_wilson": wilson(k, n),
                          "by_task": {t: {"decided": int((dec["task"] == t).sum()),
                                          "read_correct": int(h[dec["task"] == t].sum())} for t in CONTENT_TASKS}})
            blk[scope] = o
        headline[variant] = blk

    check = None
    if two_rater_first is not None:
        now = first_verdicts(x2)
        differ = sum(1 for k in two_rater_first.index if k in now.index and two_rater_first[k] != now[k])
        absent = sum(1 for k in two_rater_first.index if k not in now.index)
        check = {"first_verdicts_compared": int(len(two_rater_first)), "differ": differ, "absent": absent}
    fc = first.loc["C"]
    return {"counts": {"answers": int(len(assignment)), "shared": len(shared), "set_B": len(set_b),
                       "rated_by_C": int(len(fc))},
            "pairwise": pairwise, "majority": majority_blk, "ab_resolution": resolution, "headline": headline,
            "per_rater": per_rater(first, key, raters=("C",)),
            "first_verdict_counts": {"C": {v: int((fc == v).sum()) for v in VERDICTS}},
            "two_rater_export_check": check}


def _stat(v, status):
    return f"{v:.2f}" if v is not None else status


def summary_three(R):
    c, pw, mj, rs, hl = R["counts"], R["pairwise"], R["majority"], R["ab_resolution"], R["headline"]
    if R.get("source") == "rater3 csv":
        who = (f"A third rater, C (one of the authors, blind to scorer verdicts; verdicts from `{R['export']}`), rated set "
               f"B: {c['set_B']} answers, the {c['shared']} shared with rater A included. The CSV showed the question, "
               "reference answer and model output, no scorer verdict and no AI reading; its one verdict per answer is "
               "the first verdict. Raters A and B from the two-rater export.")
    else:
        who = (f"A third rater, C (one author, blind to scorer verdicts; username `{R.get('third_rater', THIRD_RATER)}`), "
               f"rated set B: {c['set_B']} answers, the {c['shared']} shared with rater A included. The page showed no "
               "AI reading.")
    L = ["# Human audit, three raters", "",
         who + " Codes and the first-verdict rule as in the two-rater audit. Method: `scripts/run_human_audit.py` "
         "docstring.", "",
         "## Agreement between pairs of raters", "",
         "Three codes (correct, not correct, can't tell): same code, Cohen's kappa and Gwet's AC1 (q = 3). On the "
         "answers both decided: the 2x2 table, kappa and AC1 (q = 2).", "",
         "| Pair | Answers | Same code | Kappa | AC1 | Both decided | Both correct | First only correct | "
         "Second only correct | Both not correct | Kappa (2x2) | AC1 (2x2) |",
         "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for p, b in pw.items():
        co, bi, t = b["codes"], b["decided_both"], b["table_2x2"]
        L.append(f"| {p[0]}-{p[1]} | {b['n']} ({b['answers']}) | {co['agree']}/{co['n']} = {pct(co['share'])}% | "
                 f"{_stat(co['kappa'], co['kappa_status'])} | {_stat(co['ac1'], co['ac1_status'])} | {bi['n']} | "
                 f"{t['both_correct']} | {t['first_only_correct']} | {t['second_only_correct']} | "
                 f"{t['both_not_correct']} | {_stat(bi['kappa'], bi['kappa_status'])} | "
                 f"{_stat(bi['ac1'], bi['ac1_status'])} |")
    L += ["", "## Majority verdict on the shared answers", "",
          f"Of {mj['n']}: all three agree on {mj['unanimous']}, two of three on {mj['two_of_three']}, no majority on "
          f"{mj['no_majority']}. Majority correct {mj['majority_correct']}, not correct {mj['majority_not_correct']}, "
          f"can't tell {mj['majority_cant_tell']}.", "",
          "## The A-B disagreements, two of three", "",
          f"A and B disagree on {rs['disagreements']} shared answers. C sides with A on {rs['sides_with_A']} and with B "
          f"on {rs['sides_with_B']}; {rs['unresolved']} stay unresolved. Resolved as correct {rs['resolved_correct']}, "
          f"not correct {rs['resolved_not_correct']}, can't tell {rs['resolved_cant_tell']}.", ""]
    if rs["answers"]:
        L += ["| Answer | A | B | C | Sides with |", "|---|---|---|---|---|"]
        L += [f"| {r['audit_id']} | {r['A']} | {r['B']} | {r['C']} | {r['sides_with'] or 'neither'} |"
              for r in rs["answers"]]
        L.append("")
    L += ["## Headline with the resolved rating", "",
          "GSM8K and BIG-Bench Hard outputs that the strict scorer marks wrong and the content scorer marks correct: the "
          "share read as correct, by the majority on shared answers and by the single rating elsewhere. Shares of the "
          "audit sample (12 disagreeing for every 4 agreeing answers per stratum), not population rates; 95% Wilson "
          "intervals, descriptive.", "",
          "| Rating on set B | Outputs | Decided | No majority | Can't tell | Read as correct | Content scorer, all "
          "GSM8K and BIG-Bench Hard | Strict scorer, same |", "|---|---:|---:|---:|---:|---|---|---|"]
    for v, name in (("C_for_B", "C in place of B (primary)"), ("B_for_C", "B in place of C")):
        h, a = hl[v]["strict_wrong_content_correct"], hl[v]["strict_and_content"]
        L.append(f"| {name} | {h['answers']} | {h['decided']} | {h['no_majority_excluded']} | "
                 f"{h['cant_tell_excluded']} | {h['read_correct']}/{h['decided']} = {pct(h['share_read_correct'])}% "
                 f"{ci_txt(h['ci95_wilson'])} | {a['content']['agree']}/{a['content']['n']} = "
                 f"{pct(a['content']['share'])}% {ci_txt(a['content']['ci95_wilson'])} | "
                 f"{a['strict']['agree']}/{a['strict']['n']} = {pct(a['strict']['share'])}% |")
    b = R["per_rater"]["C"]
    sc, al = b["strict_and_content"], b["all"]
    L += ["", "## Rater C alone", "",
          "| Answers | Can't tell | Content scorer, GSM8K and BIG-Bench Hard | Strict scorer, GSM8K and BIG-Bench Hard | "
          "Strict scorer, every answer |", "|---:|---:|---|---|---|",
          f"| {b['answers']} | {b['cant_tell']} | " + " | ".join(
              f"{x['agree']}/{x['n']} = {pct(x['share'])}% {ci_txt(x['ci95_wilson'])}"
              for x in (sc["content"], sc["strict"], al["strict"])) + " |",
          "", "| Scope | Both scorers wrong, read as correct | Both scorers right, read as not correct |",
          "|---|---|---|"]
    for scope, lab in (("strict_and_content", "GSM8K and BIG-Bench Hard"), ("all", "every answer")):
        cw, cr = b["controls"][scope]["both_scorers_wrong"], b["controls"][scope]["both_scorers_right"]
        L.append(f"| {lab} | {cw['read_correct']} of {cw['n']} ({pct(cw['share'])}%) | "
                 f"{cr['read_not_correct']} of {cr['n']} ({pct(cr['share'])}%) |")
    fv = R["first_verdict_counts"]["C"]
    L += ["", "First verdicts of C: " + ", ".join(f"{k} {fv[k]}" for k in VERDICTS) + "."]
    ck = R.get("two_rater_export_check")
    if ck is not None:
        L.append(f"Check against the two-rater export: {ck['first_verdicts_compared']} first verdicts of A and B "
                 f"compared, {ck['differ']} differ, {ck['absent']} absent from the three-rater export.")
    return "\n".join(L) + "\n"


def write_tables_three(R, out):
    td = os.path.join(out, "tables")
    os.makedirs(td, exist_ok=True)
    rows = []
    for p, b in R["pairwise"].items():
        rows.append({"pair": f"{p[0]}-{p[1]}", "answers": b["n"], "scope": b["answers"],
                     "same_code": b["codes"]["agree"], "kappa": b["codes"]["kappa"], "ac1": b["codes"]["ac1"],
                     "decided_both": b["decided_both"]["n"], **b["table_2x2"],
                     "kappa_2x2": b["decided_both"]["kappa"], "ac1_2x2": b["decided_both"]["ac1"]})
    pd.DataFrame(rows).to_csv(os.path.join(td, "three_raters_pairwise.csv"), index=False)
    pd.DataFrame(R["ab_resolution"]["answers"], columns=["audit_id", "A", "B", "C", "sides_with", "resolved"]) \
        .to_csv(os.path.join(td, "three_raters_ab_resolution.csv"), index=False)


def run_three(export=EXPORT3, out=OUT, assignment=ASSIGNMENT, key=KEY, manifest=MANIFEST, two_rater_export=EXPORT,
              third=THIRD_RATER, rater3_csv=RATER3_CSV):
    """Returns the exit code: 0 written, 1 refused, 2 no third rating yet. Writes only the three-rater files.
    Two sources for rater C: the sheet's three-rater export, or rater C's CSV
    rater3_csv (read_rater3) beside the two-rater export, rater C = RATER3_NAME. A CSV with no verdict filled in counts as
    absent; both sources present is refused."""
    try:
        c_rows = read_rater3(rater3_csv) if rater3_csv and os.path.exists(rater3_csv) else None
    except ExportError as e:
        print(str(e))
        return 1
    have_export = os.path.exists(export)
    if have_export and c_rows is not None:
        print(f"Two third-rater sources are present: the export {export} and the filled CSV {rater3_csv}. Move one "
              "away and run scripts/run_human_audit.py --three-raters again. Nothing was written.")
        return 1
    if not have_export and c_rows is None:
        print(f"The three-rater export is not there yet: {export}\nWhen the third rating is complete, export "
              "the `ratings` tab of the sheet \"Rating audit: ratings\" (File > Download > Comma-separated values, with "
              f"the `ratings` tab open; it holds the rows of user1, user2 and {third}) to that path, or fill the "
              f"`verdict` column of {rater3_csv} (blank template: --write-rater3-template), then run "
              "scripts/run_human_audit.py --three-raters again. Nothing was written.")
        return 2
    if not os.path.exists(key):
        print(f"{key} is missing; it is written by scripts/draw_scoring_audit_v2.py (git-ignored).")
        return 1
    source = "three-rater export" if have_export else "rater3 csv"
    try:
        a = read_assignment(assignment)
        if have_export:
            x2, xc = validate_three(read_csv(export), a, third)
        else:
            if not (two_rater_export and os.path.exists(two_rater_export)):
                raise ExportError(f"the two-rater export {two_rater_export} is needed beside {rater3_csv}")
            third = RATER3_NAME
            x2, xc = validate_three(pd.concat([read_csv(two_rater_export), c_rows], ignore_index=True), a, third)
        man = json.load(open(manifest, encoding="utf-8")) if manifest and os.path.exists(manifest) else None
        f2 = None
        if have_export and two_rater_export and os.path.exists(two_rater_export):
            f2 = first_verdicts(validate_export(read_csv(two_rater_export), a))
        R = analyse_three(x2, xc, a, read_key(key, man), f2)
    except ExportError as e:
        print(str(e))
        return 1
    used = export if have_export else rater3_csv
    R = {"analysis": "three-rater human audit",
         "created": time.strftime("%Y-%m-%d %H:%M:%S"), "export": os.path.relpath(used, ROOT)
         if os.path.abspath(used).startswith(ROOT) else used, "source": source, "third_rater": third,
         "interval": "95% Wilson score", **R}
    mpath = os.path.join(ROOT, "paper", "macros_third_rater.py")
    if os.path.exists(mpath):
        import importlib.util
        spec = importlib.util.spec_from_file_location("macros_third_rater_hrt", mpath)
        mt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mt)
        R["macros"] = mt.third_rater_macros(R)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "three_raters.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    write_tables_three(R, out)
    txt = summary_three(R)
    for fn in ("summary_three_raters.md", "summary_three_raters_clean.md"):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--export", default=EXPORT)
    p.add_argument("--out", default=OUT)
    p.add_argument("--assignment", default=ASSIGNMENT)
    p.add_argument("--key", default=KEY)
    p.add_argument("--llm", default=LLM_RATINGS)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--three-raters", action="store_true",
                   help="the three-rater analysis of ratings_three_raters.csv; the two-rater files are not touched")
    p.add_argument("--export3", default=EXPORT3)
    p.add_argument("--rater3-csv", default=RATER3_CSV)
    p.add_argument("--write-rater3-template", action="store_true",
                   help="write the blank ratings_rater3.csv for the third rater (set B in the page's order); never overwrites")
    a = p.parse_args(argv)
    if a.write_rater3_template:
        return write_rater3_template(a.rater3_csv)
    if a.three_raters:
        return run_three(a.export3, a.out, a.assignment, a.key, a.manifest, rater3_csv=a.rater3_csv)
    return run(a.export, a.out, a.assignment, a.key, a.llm, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
