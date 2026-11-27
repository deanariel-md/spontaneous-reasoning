r"""What the strict reading misses on DeepSeek V4.1-Flash-1017's forced arms (Nebius run; read-only check).

Reads the Nebius raw replies (data/frontier_panel/nebius_panel_raw.jsonl), keeps the reply the report keeps for each key
(scripts/nebius_panel_report.answered), rebuilds the forced and five-shot forced outputs with the report's own code
(nebius_panel_report.outputs_frame, which cuts GSM8K and BIG-Bench Hard forced replies after their first "}") and
re-scores them with nebius_panel_report.score_task (src/bbh.py, src/scoring.py, src/math_task.py, src/hotpot_task.py,
src/content_scoring.py). The re-score is checked against the saved scored rows (data/frontier_panel/scored/).
Variants, on the forced arms:
    pipeline   as scored: the reply cut after its first "}" (GSM8K, BIG-Bench Hard), the prefill put in front;
    no_cut     the whole reply as the endpoint returned it (16 tokens at most; 64 on HotpotQA), prefill in front;
    tags_read  BIG-Bench Hard only, as pipeline with "<answer>" / "</answer>" tags removed and "<X>" and full-width
               "（X）" read as "(X)": a hypothetical scorer change, shown only to size the effect.
Then: the forms of every BIG-Bench Hard forced reply that is content-correct and strict-wrong (the method and form
classes of scripts/check_mistral_bbh_forced.py, with classes added for this model), by subtask, and a sample of 30;
replies that write a placeholder instead of an answer, on all four benchmarks; three rebuilt prompts of each forced
arm checked against the recorded prompt hash (nothing is sent anywhere); the same forms in the main panel's forced
BIG-Bench Hard outputs (text only, data/d2_out/forced/bbh and data/d2_out/fewshot/bbh; the panel's strict and content
gaps are those of results/ladder/mistral_bbh_forced_check.json, cited, not recomputed); and every key the raw file
answered more than once (the determinism checks and the pilot's reruns), per arm.

Prints one JSON object (the numbers for results/frontier_panel/nebius_forced_check.md) to stdout; writes no file.
No API calls.

PowerShell, repository root:
    $env:PYTHONIOENCODING = "utf-8"
    .\.venv\Scripts\python.exe scripts\check_nebius_forced.py > $env:TEMP\nebius_forced_check.json
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_panel as npl  # noqa: E402
import nebius_panel_report as npr  # noqa: E402
from src import bbh, content_scoring as cs  # noqa: E402

PREFILL = npl.PREFILL
FORCED = npl.FORCED_ARMS
TASKS = ("gsm8k", "math", "bbh", "hotpot")
SEED = 0
N_SAMPLE = 30

BS_LETTER = re.compile(r"\(\\([A-Z])\)")
WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\s*\{")
ANSWER_TAG = re.compile(r"^<answer>\s*(?P<a>.*?)\s*(?:</answer>)?$", re.IGNORECASE | re.DOTALL)
ANGLE = re.compile(r"^<(?P<a>[^<>]+)>$")
FULLWIDTH = re.compile(r"^（\s*([A-Z])\s*）$")
INLINE_MATH = re.compile(r"^\\\(\s*\(?([A-Z])\)?\s*\\\)$")
PLACEHOLDERS = (("<answer>", re.compile(r"^<\s*answer\s*>$", re.I)),
                ("<number>", re.compile(r"^<\s*number\s*>$", re.I)),
                ("answer", re.compile(r"^answer$", re.I)),
                ("...", re.compile(r"^(?:\.\.\.|…)$")),
                ("empty box", re.compile(r"^$")),
                ("empty parentheses ()", re.compile(r"^\(\s*\)$")),
                ("empty angle brackets <>", re.compile(r"^<\s*>$")))


# ----------------------------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------------------------

def read_raw() -> list[dict]:
    rows = []
    with open(npl.RAW, encoding="utf-8") as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def box_content(reply) -> str | None:
    """Content of the box the prefill opens (depth-matched, as src/bbh.boxed_contents reads it), or None if it does
    not close; a box the cut left open is read completed, as the pipeline scores it (src/forced_stop.py)."""
    from src import forced_stop
    b = bbh.boxed_contents(forced_stop.close(PREFILL + (reply if isinstance(reply, str) else "")))
    return b[0] if b else None


def placeholder_kind(bc: str | None, reply=None) -> str | None:
    """The placeholder a forced reply writes in place of an answer, or None. An empty reply (the model ended right
    after the prefill) counts as one."""
    if not isinstance(bc, str):
        return "empty reply" if isinstance(reply, str) and not reply.strip() else None
    s = bc.strip()
    for name, rx in PLACEHOLDERS:
        if rx.match(s):
            return name
    return None


def read_tags(text: str) -> str:
    """The hypothetical change of the tags_read variant, applied to the box content."""
    t = text or ""
    j = t.find("}")
    head, tail = (t[:j], t[j:]) if j >= 0 else (t, "")
    s = head.strip()
    m = ANSWER_TAG.match(s)
    if m and m.group("a"):
        s = m.group("a")
    m = ANGLE.match(s)
    if m and re.fullmatch(r"[A-Z]", m.group("a").strip()):
        s = f"({m.group('a').strip()})"
    m = FULLWIDTH.match(s)
    if m:
        s = f"({m.group(1)})"
    return s + tail if s != head.strip() else t


def score(o: pd.DataFrame, task: str, complete: bool = True) -> pd.DataFrame:
    """npr.score_task; complete=False scores the reply without completing a box left open (the no_cut variant: no cut,
    so nothing to complete)."""
    keep = npl.CUT_AT_FIRST_BRACE
    if not complete:
        npl.CUT_AT_FIRST_BRACE = set()
    try:
        sc, _ = npr.score_task(o, task)
    finally:
        npl.CUT_AT_FIRST_BRACE = keep
    sc["reply"] = o["output_text"].to_numpy()
    return sc


def pct(x) -> float:
    return round(100 * float(np.mean(x)), 2) if len(x) else float("nan")


# ----------------------------------------------------------------------------------------------------------------
# Forms
# ----------------------------------------------------------------------------------------------------------------

def classify_gap(bc: str | None, atype, opts: dict, task_name: str) -> str:
    """Form of a content-correct, strict-wrong box content (first rule that fits); the Mistral check's classes plus
    the answer tag, angle brackets, full-width parentheses and LaTeX inline delimiters."""
    if not isinstance(bc, str):
        return "no complete box"
    s = bc.strip()
    m = ANSWER_TAG.match(s)
    if m and m.group("a"):
        inner = m.group("a")
        return "answer tag around a letter (<answer>(B))" if re.fullmatch(r"\(?[A-Z]\)?", inner) \
            else "answer tag around other text"
    m = ANGLE.match(s)
    if m:
        inner = m.group("a").strip()
        if re.fullmatch(r"[A-Z]", inner):
            return "letter in angle brackets (<B>)"
        return "option text or label in angle brackets (<Facts>)"
    if FULLWIDTH.match(s):
        return "letter in full-width parentheses"
    if INLINE_MATH.match(s):
        return "letter in LaTeX inline delimiters (\\(C\\))"
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


def form_flags(t: pd.Series) -> dict:
    """Shares (%) of replies showing each form anywhere in the box content (cut reply)."""
    bc = t.map(box_content)
    s = bc.fillna("").str.strip()
    return {"answer_tag": pct(s.str.contains(r"<\s*/?\s*answer\s*>", case=False, regex=True)),
            "answer_tag_alone": pct(s.str.fullmatch(r"<\s*answer\s*>", case=False)),
            "angle_letter": pct(s.str.fullmatch(r"<[A-Z]>")),
            "angle_other": pct(s.str.fullmatch(r"<[^<>]+>") & ~s.str.fullmatch(r"<[A-Z]>")
                               & ~s.str.fullmatch(r"<\s*answer\s*>", case=False)),
            "fullwidth": pct(s.str.contains("（")),
            "bs_letter": pct(s.str.contains(r"\(\\[A-Z]\)", regex=True)),
            "wrapper": pct(t.fillna("").str.contains(WRAPPER.pattern, regex=True)),
            "empty_box": pct(s.eq("") & bc.notna()),
            "unclosed": pct(bc.isna())}


# ----------------------------------------------------------------------------------------------------------------
# Main panel, text only
# ----------------------------------------------------------------------------------------------------------------

def panel_forms(gold: dict) -> dict:
    tasks = set(bbh.TASKS)
    out = {}
    for label, pat in (("forced_direct", "data/d2_out/forced/bbh/*/shard-*.parquet"),
                       ("fewshot_forced", "data/d2_out/fewshot/bbh/*/shard-*.parquet")):
        by_model: dict = {}
        for f in sorted(glob.glob(str(ROOT / pat))):
            try:
                d = pd.read_parquet(f, columns=["request_id", "model_key", "arm", "item_id", "output_text"])
            except Exception:
                continue
            d = d[d["arm"].isin(FORCED) & d["item_id"].astype(str).str.split("/").str[0].isin(tasks)]
            if not d.empty:
                by_model.setdefault(d["model_key"].iloc[0], []).append(d)
        for m, gs in sorted(by_model.items()):
            g = pd.concat(gs).drop_duplicates("request_id", keep="last")
            t = g["output_text"].fillna("")
            # strict before and after the tags_read change, on the rows the change touches (others keep their score)
            t2 = t.map(read_tags)
            ch = np.flatnonzero((t2 != t).to_numpy())
            before = after = 0
            for j in ch:
                g_ans, atype = gold[str(g["item_id"].iloc[j])][:2]
                a0 = bbh.strict_extract(PREFILL + t.iloc[j], atype)
                a1 = bbh.strict_extract(PREFILL + t2.iloc[j], atype)
                before += int(a0 is not None and a0 == g_ans)
                after += int(a1 is not None and a1 == g_ans)
            out[f"{label}|{m}"] = {"arm": label, "model": m, "n": int(len(g)), **form_flags(t),
                                   "tags_read_rows_touched": int(len(ch)),
                                   "tags_read_strict_change_pp": round(100 * (after - before) / len(g), 2)}
    return out


# ----------------------------------------------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------------------------------------------

def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n)


def determinism(raw: list[dict]) -> dict:
    """Every key answered more than once. Pairs: (a) the pilot reply and the determinism check's reply (same prompt,
    same cap); (b) the two replies of each determinism-variants setting (temperature 0; top_k 1; seed 1234); (c) a
    pilot reply cut at its cap and its rerun at 7,000 (differ: the rerun does not begin with the pilot reply)."""
    ok = [r for r in raw if npl.answered_ok(r)]
    by_key = collections.defaultdict(list)
    for r in ok:
        by_key[r["key"]].append(r)
    multi = {k: sorted(v, key=lambda r: r["t"]) for k, v in by_key.items() if len(v) > 1}
    pairs = []
    for k, rs in multi.items():
        pilot = [r for r in rs if r["phase"] == "pilot"]
        check = [r for r in rs if r["phase"] == "determinism-check"]
        var = collections.defaultdict(list)
        for r in rs:
            if r["phase"] == "determinism-variants":
                var[r.get("variant")].append(r)
        rerun = [r for r in rs if r["phase"] in ("complete", "complete-test")]
        for a in pilot:
            for b in check:
                pairs.append(("pilot vs check", a, b, a["prompt_sha256"] == b["prompt_sha256"]))
        for v, xs in var.items():
            for i in range(0, len(xs) - 1, 2):
                pairs.append((f"variant {v}", xs[i], xs[i + 1], xs[i]["prompt_sha256"] == xs[i + 1]["prompt_sha256"]))
        for a in pilot:
            for b in rerun:
                pairs.append(("pilot (cut) vs rerun at 7,000", a, b, None))
    rows = []
    for kind, a, b, same_prompt in pairs:
        ta, tb = a["text"], b["text"]
        if kind.startswith("pilot (cut)"):
            differ = not tb.startswith(ta)
        else:
            differ = ta != tb
        rows.append({"task": a["task"], "arm": a["arm"], "kind": kind, "key": a["key"], "differ": differ,
                     "same_prompt_hash": same_prompt, "cap_a": a.get("cap"), "cap_b": b.get("cap"),
                     "first_diff_char": first_diff(ta, tb), "len_a": len(ta), "len_b": len(tb),
                     "a_head": ta[:60], "b_head": tb[:60]})
    d = pd.DataFrame(rows)
    # strict answers of same-cap pairs, scored with the report's code
    ans_diff = {}
    same = d[~d["kind"].str.startswith("pilot (cut)")]
    if len(same):
        recs = []
        for kind, a, b, _ in pairs:
            if kind.startswith("pilot (cut)"):
                continue
            for side, r in (("a", a), ("b", b)):
                recs.append(dict(r, request_id=f"{r['request_id']}#{kind}#{side}#{len(recs)}", at_design_cap=True,
                                 pair=f"{r['key']}|{kind}|{len(recs) // 2}", side=side))
        x = pd.DataFrame(recs)
        for task, g in x.groupby("task"):
            o = npr.outputs_frame(g, task)
            sc, _ = npr.score_task(o, task)
            sc["pair"], sc["side"] = g["pair"].to_numpy(), g["side"].to_numpy()
            w = sc.pivot_table(index="pair", columns="side", values="strict_correct", aggfunc="first")
            ans_diff[task] = {"pairs": int(len(w)), "strict_score_differs": int((w["a"] != w["b"]).sum())}
    summary = {}
    for (t, a, k), g in d.groupby(["task", "arm", "kind"]):
        summary[f"{t}|{a}|{k}"] = {"pairs": int(len(g)), "differ": int(g["differ"].sum()),
                                   "same_prompt_hash": None if g["same_prompt_hash"].isna().all()
                                   else int(g["same_prompt_hash"].sum()),
                                   "median_first_diff_char": float(g.loc[g["differ"], "first_diff_char"].median())
                                   if g["differ"].any() else None,
                                   "caps": sorted({f"{x}/{y}" for x, y in zip(g["cap_a"], g["cap_b"])})}
    forced_multi = sum(1 for k, v in multi.items() if v[0]["arm"] in FORCED)
    # identical prompts under different keys (would be pairs too)
    sha = collections.defaultdict(set)
    for r in ok:
        sha[r["prompt_sha256"]].add(r["key"])
    cross = sum(1 for v in sha.values() if len(v) > 1)
    return {"keys_answered_more_than_once": len(multi), "forced_keys_answered_more_than_once": forced_multi,
            "prompt_hashes_shared_by_two_keys": cross, "by_arm_and_kind": summary, "strict_answer": ans_diff,
            "examples": d[d["differ"]].head(8).to_dict("records")}


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

def main() -> int:
    raw = read_raw()
    ans = npr.answered(raw)
    res: dict = {"answered_keys": int(len(ans)), "tasks": {}}
    frames, outs = {}, {}

    for task in TASKS:
        x = ans[(ans["task"] == task) & ans["arm"].isin(FORCED)]
        o = npr.outputs_frame(x, task)
        o_nc = o.copy()
        o_nc["output_text"] = o["raw_text"]
        o_nc["finish_reason"] = x["finish_reason"].to_numpy()
        sc = score(o, task)
        nc = score(o_nc, task, complete=False)
        saved = pd.read_parquet(npr.SCORED_DIR / f"{task}_scored.parquet",
                                columns=["request_id", "strict_correct", "content_correct"]).set_index("request_id")
        s2 = saved.loc[sc["request_id"]]
        tr = {"pipeline_matches_saved": bool(
            (s2["strict_correct"].to_numpy() == sc["strict_correct"].to_numpy()).all()
            and (s2["content_correct"].to_numpy() == sc["content_correct"].to_numpy()).all())}
        variants = {"pipeline": sc, "no_cut": nc}
        if task == "bbh":
            o_t = o.copy()
            o_t["output_text"] = o["output_text"].map(read_tags)
            variants["tags_read"] = score(o_t, task)
        for how, f in variants.items():
            for arm, g in f.groupby("arm"):
                d = tr.setdefault(arm, {"n": int(len(g))})
                d[f"strict_{how}"] = pct(g["strict_correct"])
                d[f"content_{how}"] = pct(g["content_correct"])
        for arm in FORCED:
            m = (sc["arm"] == arm).to_numpy()
            tr[arm]["rows_changed_by_cut_text"] = int((o.loc[m, "output_text"].to_numpy()
                                                       != o.loc[m, "raw_text"].to_numpy()).sum())
            tr[arm]["strict_changed_by_cut"] = int((sc.loc[m, "strict_correct"].to_numpy()
                                                    != nc.loc[m, "strict_correct"].to_numpy()).sum())
            tr[arm]["content_changed_by_cut"] = int((sc.loc[m, "content_correct"].to_numpy()
                                                     != nc.loc[m, "content_correct"].to_numpy()).sum())
            tr[arm]["content_gained_without_cut"] = int(((nc.loc[m, "content_correct"].to_numpy() == 1)
                                                         & (sc.loc[m, "content_correct"].to_numpy() == 0)).sum())
            tr[arm]["content_lost_without_cut"] = int(((nc.loc[m, "content_correct"].to_numpy() == 0)
                                                       & (sc.loc[m, "content_correct"].to_numpy() == 1)).sum())
            tr[arm]["content_class"] = sc.loc[m, "content_class"].value_counts().to_dict() \
                if "content_class" in sc.columns else None
        sc["raw_text"] = o["raw_text"].to_numpy()
        sc["content_nc"] = nc["content_correct"].to_numpy()
        sc["strict_nc"] = nc["strict_correct"].to_numpy()
        frames[task], outs[task] = sc, o
        res["tasks"][task] = tr
        print(task, json.dumps(tr, ensure_ascii=False), file=sys.stderr)

    # --- placeholders, every benchmark
    ph: dict = {}
    for task, sc in frames.items():
        sc["box"] = sc["reply"].map(box_content)
        sc["ph"] = [placeholder_kind(b, t) for b, t in zip(sc["box"], sc["raw_text"])]
        for arm, g in sc.groupby("arm"):
            h = g[g["ph"].notna()]
            after = h["raw_text"].fillna("").map(lambda t: t[t.find("}") + 1:] if "}" in t else "")
            ph[f"{task}|{arm}"] = {
                "n_arm": int(len(g)), "rows": int(len(h)), "pp": round(100 * len(h) / len(g), 2),
                "by_kind": h["ph"].value_counts().to_dict(),
                "strict_correct": int(h["strict_correct"].sum()), "content_correct": int(h["content_correct"].sum()),
                "text_after_first_brace": int(after.str.strip().ne("").sum()),
                "content_correct_without_cut": int(h["content_nc"].sum()),
                "strict_correct_without_cut": int(h["strict_nc"].sum()),
                "after_examples": after[after.str.strip().ne("")].str[:40].value_counts().head(8).to_dict()}
    res["placeholders"] = ph

    # --- BIG-Bench Hard: forms of the content-correct, strict-wrong replies
    gold = npr.content_gold("bbh")
    sc = frames["bbh"]
    sc["subtask"] = sc["item_id"].str.split("/").str[0]
    gap = sc[(sc["content_correct"] == 1) & (sc["strict_correct"] == 0)].copy()
    gap["form"] = [classify_gap(b, gold[i][1], gold[i][2] or {}, gold[i][3]) for b, i in zip(gap["box"], gap["item_id"])]
    n_arm = sc.groupby("arm").size()
    by_form = gap.groupby(["form", "arm"]).size().unstack(fill_value=0)
    res["gap"] = {"n": {a: int(v) for a, v in gap.groupby("arm").size().items()},
                  "n_arm": {a: int(v) for a, v in n_arm.items()},
                  "by_form": {f: {a: int(v) for a, v in r.items()} for f, r in by_form.iterrows()},
                  "by_form_pp": {f: {a: round(100 * v / n_arm[a], 2) for a, v in r.items()}
                                 for f, r in by_form.iterrows()}}
    sub_n = sc.groupby(["subtask", "arm"]).size().unstack(fill_value=0)
    sub = gap.groupby(["subtask", "arm"]).size().unstack(fill_value=0).reindex(index=sub_n.index, fill_value=0)
    res["gap"]["by_subtask"] = {s: {a: int(sub.loc[s, a]) if a in sub.columns else 0 for a in sub_n.columns}
                                for s in sub_n.index}
    sf = gap.groupby(["subtask", "form"]).size().unstack(fill_value=0)
    res["gap"]["subtask_by_form"] = {s: {f: int(v) for f, v in r.sort_values(ascending=False).items() if v}
                                     for s, r in sf.iterrows()}
    res["gap"]["other_examples"] = gap.loc[gap["form"] == "other", "box"].value_counts().head(30).to_dict()
    res["gap"]["examples_by_form"] = {f: g["box"].value_counts().head(6).to_dict() for f, g in gap.groupby("form")}

    rng = np.random.default_rng(SEED)
    pools = {s: g.sample(frac=1, random_state=SEED) for s, g in gap.groupby("subtask")}
    order = list(pools)
    rng.shuffle(order)
    picks, k = [], 0
    while len(picks) < N_SAMPLE:
        for s in order:
            if k < len(pools[s]) and len(picks) < N_SAMPLE:
                picks.append(pools[s].iloc[k])
        k += 1
    res["sample"] = [{"subtask": r.subtask, "item_id": r.item_id, "arm": r.arm, "reply": r.reply[:80],
                      "raw": r.raw_text[:80], "form": r.form, "gold": gold[r.item_id][0]}
                     for r in pd.DataFrame(picks).itertuples()]

    # both readings wrong: placeholder and form counts
    miss = sc[(sc["content_correct"] == 0) & (sc["strict_correct"] == 0)]
    res["both_wrong"] = {a: {"n": int(len(g)), "placeholder": int(g["ph"].notna().sum()),
                             "unclosed_box": int(g["box"].isna().sum()),
                             "wrapper_cut_open": int((g["reply"].str.contains(WRAPPER) & g["box"].isna()).sum()),
                             "content_class": g["content_class"].value_counts().to_dict()}
                         for a, g in miss.groupby("arm")}
    res["forms_all"] = {a: form_flags(g["reply"]) for a, g in sc.groupby("arm")}
    res["forms_all_raw"] = {a: form_flags(g["raw_text"]) for a, g in sc.groupby("arm")}

    # the same model's other BIG-Bench Hard arms: does it write the placeholder there?
    oth = ans[(ans["task"] == "bbh") & ~ans["arm"].isin(FORCED)]
    res["bbh_other_arms"] = {a: {"n": int(len(g)),
                                 "answer_tag": int(g["text"].str.contains(r"<\s*answer\s*>", case=False).sum()),
                                 "boxed_answer_placeholder": int(g["text"].str.contains(r"\\boxed\{\s*<answer>\s*\}",
                                                                                        case=False).sum()),
                                 "boxed_angle_letter": int(g["text"].str.contains(r"\\boxed\{<[A-Z]>\}").sum()),
                                 "boxed_text_wrapper": int(g["text"].str.contains(r"\\boxed\{\(?\\text\{").sum())}
                             for a, g in oth.groupby("arm")}

    # --- other benchmarks
    res["other_tasks"] = {}
    for task in ("gsm8k", "math", "hotpot"):
        f = frames[task]
        g = f[(f["content_correct"] == 1) & (f["strict_correct"] == 0)]
        h = f[(f["content_correct"] == 0) & (f["strict_correct"] == 1)]
        res["other_tasks"][task] = {
            "gap_rows": {a: int(v) for a, v in g.groupby("arm").size().items()},
            "reverse_rows": {a: int(v) for a, v in h.groupby("arm").size().items()},
            "n": {a: int(v) for a, v in f.groupby("arm").size().items()},
            "top_gap_replies": {a: x["reply"].str[:40].value_counts().head(6).to_dict() for a, x in g.groupby("arm")},
            "top_replies": {a: x["reply"].str[:40].value_counts().head(6).to_dict() for a, x in f.groupby("arm")},
            "unclosed": {a: int(x["box"].isna().sum()) for a, x in f.groupby("arm")}}

    # --- rendered prompts: three per forced arm, rebuilt and checked against the recorded hash (not sent)
    df = npl.load_complete(tasks=("bbh",))
    rend = npl.PanelRenderer()
    rec = ans.set_index("key")
    checks = []
    for arm in FORCED:
        for _, r in df[df["arm"] == arm].sample(3, random_state=SEED).iterrows():
            p = rend.prompt_messages(json.loads(r["messages_json"]), arm)
            sha = hashlib.sha256(p.encode("utf-8")).hexdigest()
            checks.append({"arm": arm, "key": r["key"], "hash_matches": sha == rec.loc[r["key"], "prompt_sha256"],
                           "user_turns": p.count("<｜User｜>"), "ends": p[-60:],
                           "instruction_has_boxed_answer": "\\boxed{<answer>}" in p,
                           "exemplar_answers": re.findall(r"</think>(\\boxed\{[^}]*\})<｜end▁of▁sentence｜>", p),
                           "reply": rec.loc[r["key"], "text"]})
    res["prompt_checks"] = checks
    msgs = df[df["arm"].isin(FORCED)]["messages_json"]
    res["instruction_share"] = {
        "rows": int(len(msgs)),
        "last_user_turn_has_boxed_answer": int(msgs.map(lambda m: "\\boxed{<answer>}" in json.loads(m)[-1]["content"]).sum())}

    # --- main panel forms (text only) and the Mistral check's panel gaps (cited)
    res["panel_forms"] = panel_forms(gold)
    mj = json.loads((ROOT / "results" / "ladder" / "mistral_bbh_forced_check.json").read_text(encoding="utf-8"))
    res["mistral_check_panel_gap"] = mj.get("panel_forced_direct_gap", {}).get("mean_gap")

    # --- determinism
    res["determinism"] = determinism(raw)

    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
