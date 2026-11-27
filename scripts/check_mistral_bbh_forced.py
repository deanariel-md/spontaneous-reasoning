r"""Why Mistral Large 3's BIG-Bench Hard forced arms score low under the strict reading (read-only check).

Reads Mistral Large 3's raw replies (data/ladder_api/mistral-large-2512/raw.jsonl) and re-scores the forced arms with the
pipeline's own code (scripts/mistral_ladder.outputs_frame and score_frame, which call src/bbh.py and
src/content_scoring.py), under these variants:
    pipeline   as collected: an echoed "\boxed{" removed, the reply cut after its first "}", the prefill put back,
               and a box the cut left open read as the closed box it was meant to be (src/forced_stop.py);
    no_strip   the echoed prefix left in the reply (the reply as the API returned it), otherwise as pipeline;
    no_cut     echo removed, no cut at the first "}" (the whole reply after the prefill, no completion);
    bs_letter  as pipeline, with "(\C)" read as "(C)" (a backslash between "(" and a capital letter removed);
               a hypothetical scorer change, shown only to size the effect.
Then classifies every BIG-Bench Hard forced reply that is content-correct and strict-wrong, draws a sample of 40
across subtasks, scans the main panel's forced BIG-Bench Hard outputs (data/d2_out/forced/bbh, data/d2_out/fewshot/bbh)
and the ladder's local outputs for the same forms, and checks the other three benchmarks' forced arms.

Writes results/ladder/mistral_bbh_forced_check.json (numbers for the report). No API calls, nothing else written.

PowerShell, repository root:
    $env:PYTHONIOENCODING = "utf-8"
    .\.venv\Scripts\python.exe scripts\check_mistral_bbh_forced.py
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mistral_ladder as ml  # noqa: E402
import nebius_panel_report as npr  # noqa: E402
from src import bbh, content_scoring as cs  # noqa: E402

PREFILL = ml.PREFILL
FORCED = ml.FORCED
OUT = ROOT / "results" / "ladder" / "mistral_bbh_forced_check.json"
BS_LETTER = re.compile(r"\(\\([A-Z])\)")
BS_ANY = r"\(\\[A-Z]\)"  # the same, without a group (for str.contains)
WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*\{")
SEED = 0


def read_forced_raw() -> tuple[list[dict], dict]:
    rows, echo = [], collections.Counter()
    with open(ml.RAW, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["arm"] not in FORCED or r.get("status") != 200 or not isinstance(r.get("text"), str):
                continue
            rows.append(r)
            echo[(r["task"], r["arm"], bool(r.get("prefix_echoed")))] += 1
    return rows, echo


def score(rows: list[dict], task: str, cut: bool = True) -> pd.DataFrame:
    import nebius_panel as npl
    keep, keep_npl = ml.CUT_AT_FIRST_BRACE, npl.CUT_AT_FIRST_BRACE
    if not cut:
        ml.CUT_AT_FIRST_BRACE = set()
        npl.CUT_AT_FIRST_BRACE = set()     # no cut, so no completion of a cut box either
    try:
        o = ml.outputs_frame(rows, task)
        sc = ml.score_frame(o, task, "cut")
    finally:
        ml.CUT_AT_FIRST_BRACE, npl.CUT_AT_FIRST_BRACE = keep, keep_npl
    sc["reply"] = o["output_text_cut"].to_numpy()
    return sc


def variant_rows(rows: list[dict], how: str) -> list[dict]:
    out = []
    for r in rows:
        t = r["text"]
        if how == "no_strip" and r.get("prefix_echoed"):
            t = PREFILL + t
        elif how == "bs_letter":
            t = BS_LETTER.sub(r"(\1)", t)
        out.append({**r, "text": t})
    return out


def box_content(reply: str) -> str | None:
    """The last complete box of the reply as the pipeline reads it: the prefill in front and a box the cut left open
    completed (src/forced_stop.py)."""
    from src import forced_stop
    b = bbh.boxed_contents(forced_stop.close(PREFILL + (reply if isinstance(reply, str) else "")))
    return b[-1] if b else None


def classify_gap(bc: str | None, atype, opts: dict, task_name: str) -> str:
    """Form of a content-correct, strict-wrong box content (first rule that fits)."""
    if not isinstance(bc, str):
        return "no complete box"
    s = bc.strip()
    if BS_LETTER.fullmatch(s):
        return "letter with backslash (\\C)"
    if BS_LETTER.match(s):
        return "letter with backslash plus text"
    if "**" in s:
        return "Markdown bold"
    if WRAPPER.search(s):
        return "LaTeX text wrapper"
    if s != bc and bbh.normalize_answer(s, atype) is not None:
        return "surrounding space"
    if atype.kind == "letter":
        if re.match(r"^\(?[A-Z]\)?[\s:.\-]+\S", s):
            return "letter plus option text"
        if re.fullmatch(r"\(?[a-z]\)?\.?", s):
            return "lower-case letter"
        k = cs.clean(s)
        if k in opts:
            return "option text, no letter"
        if k.isdigit() and int(k) < len(cs.NUMBER_WORDS) and cs.NUMBER_WORDS[int(k)] in opts:
            return "option number as digit"
    if atype.kind == "words":
        k = cs.clean(s)
        if k in ("true", "false"):
            return "True/False for Yes/No"
        if any(rx.search(k) for rx, _ in cs.TASK_WORDS.get(task_name, ())):
            return "task wording for Yes/No"
    return "other"


def pct(x) -> float:
    return round(100 * float(np.mean(x)), 2)


def panel_scores(gold: dict) -> dict:
    """The main panel's forced BIG-Bench Hard outputs on disk (every row of the complete design), scored with the same
    code (nebius_panel_report.score_task: the prefill put in front, strict and content), plus the share of replies in
    each form and the strict score if "(\\C)" were read as "(C)". The collection's forced runs stopped at the first "}"
    (kept), so no cut is needed. data/d2_out/forced/bbh holds the zero-shot forced arm; data/d2_out/fewshot/bbh holds
    the five-shot forced arm (stored under arm "forced_direct", request ids ending "|fs"). The ladder's own models are
    not on this PC (data/d2_out/ladder2 holds no BIG-Bench Hard forced outputs)."""
    tasks = set(bbh.TASKS)
    out = {}
    srcs = [("forced_direct", "data/d2_out/forced/bbh/*/shard-*.parquet"),
            ("fewshot_forced", "data/d2_out/fewshot/bbh/*/shard-*.parquet"),
            ("ladder2", "data/d2_out/ladder2/*/**/*.parquet")]
    for label, pat in srcs:
        files = sorted(glob.glob(str(ROOT / pat), recursive=True))
        by_model: dict = {}
        for f in files:
            try:
                d = pd.read_parquet(f, columns=["request_id", "model_key", "arm", "item_id", "output_text",
                                                "finish_reason"])
            except Exception:
                continue
            d = d[d["arm"].isin(FORCED) & d["item_id"].astype(str).str.split("/").str[0].isin(tasks)]
            if not d.empty:
                by_model.setdefault(d["model_key"].iloc[0], []).append(d)
        for m, gs in sorted(by_model.items()):
            g = pd.concat(gs).drop_duplicates("request_id", keep="last").reset_index(drop=True)
            g["arm"] = "forced_direct"        # the scorer's forced arm name: the prefill goes in front
            g["item_id"] = g["item_id"].astype(str)
            s, _ = npr.score_task(g, "bbh")
            t = g["output_text"].fillna("")
            bs = t.str.contains(BS_ANY, regex=True)
            fixed = s["strict_correct"].to_numpy().copy()
            for j in np.flatnonzero(bs.to_numpy()):
                a = bbh.strict_extract(PREFILL + BS_LETTER.sub(r"(\1)", t.iloc[j]), gold[g["item_id"].iloc[j]][1])
                fixed[j] = int(a is not None and a == gold[g["item_id"].iloc[j]][0])
            out[f"{label}|{m}"] = {
                "arm": label, "model": m, "n": int(len(g)),
                "strict": pct(s["strict_correct"]), "content": pct(s["content_correct"]),
                "strict_bs_letter": pct(fixed),
                "bs_letter": pct(bs), "wrapper": pct(t.str.contains(WRAPPER.pattern, regex=True)),
                "bold": pct(t.str.contains(r"\*\*", regex=True)), "echo": pct(t.str.startswith(PREFILL))}
            print("panel", label, m, out[f"{label}|{m}"])
    return out


def main() -> int:
    rows, echo = read_forced_raw()
    res: dict = {"echo": {f"{t}|{a}|{e}": n for (t, a, e), n in sorted(echo.items())}, "tasks": {}}
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    # --- variants, every benchmark; pipeline checked against the saved scored rows
    frames = {}
    for task, rs in by_task.items():
        saved = pd.read_parquet(ml.DATA / "scored" / f"{task}_cut.parquet",
                                columns=["request_id", "arm", "strict_correct", "content_correct"])
        saved = saved[saved["arm"].isin(FORCED)].set_index("request_id")
        tr = {}
        for how in ("pipeline", "no_strip", "no_cut") + (("bs_letter",) if task == "bbh" else ()):
            sc = score(variant_rows(rs, how) if how != "no_cut" else rs, task, cut=how != "no_cut")
            if how == "pipeline":
                s2 = saved.loc[sc["request_id"]]
                tr["pipeline_matches_saved"] = bool(
                    (s2["strict_correct"].to_numpy() == sc["strict_correct"].to_numpy()).all()
                    and (s2["content_correct"].to_numpy() == sc["content_correct"].to_numpy()).all())
                frames[task] = sc
            for arm, g in sc.groupby("arm"):
                d = tr.setdefault(arm, {"n": int(len(g))})
                d[f"strict_{how}"] = pct(g["strict_correct"])
                d[f"content_{how}"] = pct(g["content_correct"])
        res["tasks"][task] = tr
        print(task, json.dumps(tr))

    # --- BIG-Bench Hard: classify content-correct, strict-wrong replies
    gold = npr.content_gold("bbh")
    sc = frames["bbh"].copy()
    sc["subtask"] = sc["item_id"].str.split("/").str[0]
    gap = sc[(sc["content_correct"] == 1) & (sc["strict_correct"] == 0)].copy()
    gap["box"] = gap["reply"].map(box_content)
    gap["form"] = [classify_gap(b, gold[i][1], gold[i][2] or {}, gold[i][3]) for b, i in zip(gap["box"], gap["item_id"])]
    n_arm = sc.groupby("arm").size()
    by_form = gap.groupby(["form", "arm"]).size().unstack(fill_value=0)
    res["gap"] = {"n": {a: int(v) for a, v in gap.groupby("arm").size().items()},
                  "n_arm": {a: int(v) for a, v in n_arm.items()},
                  "by_form": {f: {a: int(v) for a, v in r.items()} for f, r in by_form.iterrows()},
                  "by_form_pp": {f: {a: round(100 * v / n_arm[a], 2) for a, v in r.items()} for f, r in by_form.iterrows()}}
    sub_n = sc.groupby(["subtask", "arm"]).size().unstack(fill_value=0)
    sub = gap.groupby(["subtask", "arm"]).size().unstack(fill_value=0).reindex(index=sub_n.index, fill_value=0)
    top = gap.groupby("subtask")["form"].agg(lambda x: x.value_counts().index[0] + f" ({x.value_counts().iloc[0]})")
    res["gap"]["by_subtask"] = {s: {"n_items_rows": {a: int(sub_n.loc[s, a]) for a in sub_n.columns},
                                    "gap": {a: int(sub.loc[s, a]) if a in sub.columns else 0 for a in sub_n.columns},
                                    "top_form": top.get(s)} for s in sub_n.index}
    sf = gap.groupby(["subtask", "form"]).size().unstack(fill_value=0)
    res["gap"]["subtask_by_form"] = {s: {f: int(v) for f, v in r.items() if v} for s, r in sf.iterrows()}
    # "other" examples, to check the classifier
    res["gap"]["other_examples"] = gap.loc[gap["form"] == "other", "box"].value_counts().head(25).to_dict()

    # sample of 40 across subtasks: round-robin over subtasks in a seeded order, one random row each pass
    rng = np.random.default_rng(SEED)
    pools = {s: g.sample(frac=1, random_state=SEED) for s, g in gap.groupby("subtask")}
    order = list(pools)
    rng.shuffle(order)
    picks, k = [], 0
    while len(picks) < 40:
        for s in order:
            if k < len(pools[s]) and len(picks) < 40:
                picks.append(pools[s].iloc[k])
        k += 1
    samp = pd.DataFrame(picks)
    res["sample"] = [{"subtask": r.subtask, "item_id": r.item_id, "arm": r.arm, "reply": r.reply[:80],
                      "box": r.box, "form": r.form, "gold": gold[r.item_id][0]} for r in samp.itertuples()]

    # strict-wrong rows that content also misses, for the forms (the first-brace cut breaks \text{...})
    miss = sc[(sc["content_correct"] == 0) & (sc["strict_correct"] == 0)]
    res["both_wrong_forms"] = {a: {"n": int(len(g)),
                                   "bs_letter": int(g["reply"].str.contains(BS_ANY, regex=True).sum()),
                                   "wrapper_cut_open": int((g["reply"].str.contains(WRAPPER)
                                                            & g["reply"].map(box_content).isna()).sum())}
                               for a, g in miss.groupby("arm")}

    # forms among every forced reply (not only gap rows), per arm
    res["forms_all"] = {a: {"bs_letter": pct(g["reply"].str.contains(BS_ANY, regex=True)),
                            "wrapper": pct(g["reply"].str.contains(WRAPPER)),
                            "bold": pct(g["reply"].str.contains(r"\*\*", regex=True))}
                        for a, g in sc.groupby("arm")}

    # --- other benchmarks: gap rows and their most common reply forms
    res["other_tasks"] = {}
    for task in ("gsm8k", "math", "hotpot"):
        f = frames[task]
        g = f[(f["content_correct"] == 1) & (f["strict_correct"] == 0)]
        h = f[(f["content_correct"] == 0) & (f["strict_correct"] == 1)]
        res["other_tasks"][task] = {
            "gap_rows": {a: int(v) for a, v in g.groupby("arm").size().items()},
            "reverse_rows": {a: int(v) for a, v in h.groupby("arm").size().items()},
            "n": {a: int(v) for a, v in f.groupby("arm").size().items()},
            "top_gap_replies": g["reply"].str[:40].value_counts().head(8).to_dict(),
            "bs_letter": int(f["reply"].str.contains(BS_ANY, regex=True).sum()),
            "wrapper": int(f["reply"].str.contains(WRAPPER).sum()),
            "bold": int(f["reply"].str.contains(r"\*\*", regex=True).sum())}

    # --- main panel and ladder outputs on disk
    res["panel_scan"] = panel_scores(gold)
    pc = pd.read_csv(ROOT / "results" / "content_scoring" / "tables" / "compliance_bbh.csv")
    pc["gap"] = pc["forced_content_correct"] - pc["forced_strict_correct"]
    res["panel_forced_direct_gap"] = {
        "source": "results/content_scoring/tables/compliance_bbh.csv (24 discovery draws)",
        "mean_strict": round(pc["forced_strict_correct"].mean(), 2),
        "mean_content": round(pc["forced_content_correct"].mean(), 2),
        "mean_gap": round(pc["gap"].mean(), 2), "min_gap": round(pc["gap"].min(), 2),
        "max_gap": round(pc["gap"].max(), 2),
        "rows": pc[["model_key", "forced_strict_correct", "forced_content_correct", "gap"]].round(2).to_dict("records")}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("gap", "both_wrong_forms", "forms_all", "other_tasks")}, indent=1,
                     ensure_ascii=False))
    for k, v in res["panel_scan"].items():
        if v["bs_letter"] or v["wrapper"] > 1 or v["echo"]:
            print(k, v)
    print("panel scan entries", len(res["panel_scan"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
