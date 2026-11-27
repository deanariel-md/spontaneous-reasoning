r"""Draw the second scoring audit: outputs
where two scorers disagree, by task, arm and capability tier, read by one human coder who does not see the arm.

From the repository root:
    cmd /c ".venv\Scripts\python.exe scripts\draw_scoring_audit_v2.py 2>&1"

Rule, fixed before any output is read:
- Populations. GSM8K-29 and BIG-Bench Hard (28 models), arms requested (direct_answer), chain of thought
  (zero_shot_cot) and forced (forced_direct): strict against content scoring (data/processed/content/<task>_content.parquet).
  Eight-shot comparison on GSM8K-29, arms zero-shot and eight-shot chain of thought: strict against Cheng et al.'s corrected extractor
  (data/processed/exemplar/cheng_scored.parquet; zero-shot strict scores with the 101 rows cut to 2,048 tokens,
  data/processed/exemplar/zero_shot_4k.parquet, as in scripts/run_exemplar.py). Every scored configuration.
- Strata. Task x arm x tier (the external-index terciles of manifests/d2_panel_extension.json) x agreement of the two
  scorers' correctness. Per (task, arm, tier): 12 outputs where they disagree and 4 where they agree, uniformly without
  replacement with the stream sampling.rng_for("scoring-audit-v2", task, arm, tier, agreement); a cell with fewer
  disagreements gives all of them. 24 cells, at most 384 outputs. Population counts are recorded for reweighting.
- Blinding. The coder sees the question (with its options on BIG-Bench Hard), the gold answer and the output, in one
  shuffled order (stream "scoring-audit-v2", "order"); not the task label, the arm, the model, the configuration or any
  scorer's decision. A forced output is shown with its prefilled "\boxed{", so its short form can hint at the arm.

Outputs:
- results/scoring_audit_v2/coding_sheet.csv, instructions.md and instructions_clean.md;
- manifests/scoring_audit_v2.json: the rule, population counts and the drawn rows, written once;
- data/scoring_audit_v2/key.csv (git-ignored): each scorer's decision for the comparison after coding.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from src import bbh_d2, d2_io, sampling  # noqa: E402

DISAGREE, AGREE = 12, 4
TASKS = {"gsm8k": ("direct_answer", "zero_shot_cot", "forced_direct"),
         "bbh": ("direct_answer", "zero_shot_cot", "forced_direct"),
         "exemplar": ("zero_shot_cot", "eight_shot_cot")}
TIERS = ("low", "middle", "high")
OUT = os.path.join(ROOT, "results", "scoring_audit_v2")
MANIFEST = os.path.join(ROOT, "manifests", "scoring_audit_v2.json")
KEY = os.path.join(ROOT, "data", "scoring_audit_v2", "key.csv")
PROC = os.path.join(ROOT, "data", "processed")
OUTS = os.path.join(ROOT, "data", "d2_out")
CELL = ["model_key", "arm", "config_id", "item_id"]


def tiers_for(panel):
    t = bbh_d2.extension_tiers(bbh_d2.load_registries(), panel)
    missing = [k for k in panel if not t or k not in t]
    if missing:
        raise SystemExit(f"no tier for {missing}")
    return t


def populations():
    """(task, frame with CELL, tier, score_a, score_b, a_name, b_name)."""
    out = {}
    for task in ("gsm8k", "bbh"):
        df = pd.read_parquet(os.path.join(PROC, "content", f"{task}_content.parquet"),
                             columns=CELL + ["strict_correct", "content_correct", "content_class"])
        df = df.rename(columns={"strict_correct": "score_a", "content_correct": "score_b"})
        out[task] = (df, "strict", "content")
    ex = pd.read_parquet(os.path.join(PROC, "exemplar", "gsm8k_scored.parquet"), columns=CELL + ["strict_correct"])
    g = pd.read_parquet(os.path.join(PROC, "content", "gsm8k_content.parquet"), columns=CELL + ["strict_correct"])
    zs = g[g["arm"] == "zero_shot_cot"].copy()
    cut = pd.read_parquet(os.path.join(PROC, "exemplar", "zero_shot_4k.parquet"), columns=CELL + ["strict_correct"])
    zs = zs.merge(cut.rename(columns={"strict_correct": "cut"}), on=CELL, how="left", validate="one_to_one")
    zs["strict_correct"] = zs["cut"].fillna(zs["strict_correct"]).astype(int)
    strict = pd.concat([ex, zs.drop(columns="cut")], ignore_index=True)
    ch = pd.read_parquet(os.path.join(PROC, "exemplar", "cheng_scored.parquet"), columns=CELL + ["correct", "pred"])
    df = strict.merge(ch, on=CELL, how="inner", validate="one_to_one")
    if len(df) != len(strict) or len(df) != len(ch):
        raise SystemExit(f"eight-shot comparison: strict {len(strict)} and Cheng {len(ch)} rows do not match ({len(df)} merged)")
    df = df.rename(columns={"strict_correct": "score_a", "correct": "score_b"})
    out["exemplar"] = (df, "strict", "cheng")
    return out


def folder(task, arm, model):
    if task == "bbh":
        return os.path.join(OUTS, "forced", "bbh", model) if arm == "forced_direct" else os.path.join(OUTS, "bbh", "collection", model)
    if arm == "forced_direct":
        return os.path.join(OUTS, "forced", "gsm8k", model)
    if arm == "eight_shot_cot":
        return os.path.join(OUTS, "exemplar", "gsm8k", model)
    for sub in ("collection", "collection_ext"):
        d = os.path.join(OUTS, sub, model)
        if os.path.isdir(d):
            return d
    raise SystemExit(f"no collection folder for {model}")


def texts(pick):
    prefill = yaml.safe_load(open(os.path.join(ROOT, "config", "decoding_forced.yaml"), encoding="utf-8"))["prefill"]["forced_direct"]
    rows = []
    for (task, arm, mk), g in pick.groupby(["task", "arm", "model_key"]):
        d = folder(task, arm, mk)
        keys = set(zip(g["config_id"].astype(str), g["item_id"].astype(str)))
        flt = [("arm", "==", arm), ("config_id", "in", sorted(set(g["config_id"].astype(str))))]
        for f in sorted(glob.glob(os.path.join(d, "shard-*.parquet"))):
            s = pd.read_parquet(f, columns=["arm", "config_id", "item_id", "request_id", "output_text"], filters=flt)
            s = s[[(str(c), str(i)) in keys for c, i in zip(s["config_id"], s["item_id"])]]
            if len(s):
                rows.append(s.assign(task=task, model_key=mk))
    T = pd.concat(rows, ignore_index=True)
    T["item_id"] = T["item_id"].astype(str)
    T = T.drop_duplicates(["task", "model_key", "arm", "config_id", "item_id"])
    T.loc[T["arm"] == "forced_direct", "output_text"] = prefill + T.loc[T["arm"] == "forced_direct", "output_text"]
    p = pick.assign(item_id=pick["item_id"].astype(str))
    p = p.merge(T, on=["task", "model_key", "arm", "config_id", "item_id"], how="left", validate="one_to_one")
    # the eight-shot comparison's zero-shot outputs of the four 2,048-token checkpoints are read as they were scored: cut
    cut = pd.read_parquet(os.path.join(PROC, "exemplar", "zero_shot_4k.parquet"), columns=CELL + ["output_text_cut"])
    cut["item_id"] = cut["item_id"].astype(str)
    p = p.merge(cut.assign(task="exemplar"), on=["task"] + CELL, how="left", validate="one_to_one")
    p["output_text"] = p["output_text_cut"].fillna(p["output_text"])
    if p["output_text"].isna().any():
        raise SystemExit(f"{int(p['output_text'].isna().sum())} drawn outputs have no text in the shards")
    return p.drop(columns="output_text_cut")


def items():
    g, _ = d2_io.load_items(split="test")
    g = g.assign(item_id=g["item_id"].astype(str))[["item_id", "question", "gold_answer"]]
    b, _ = bbh_d2.load_items("target")
    b = b.assign(item_id=b["item_id"].astype(str))[["item_id", "question", "gold_answer"]]
    return {"gsm8k": g, "exemplar": g, "bbh": b}


def draw_cells(df, task, arms=None, sizes=(("disagree", DISAGREE), ("agree", AGREE))):
    """The stratified draw of one population (columns CELL, tier, score_a, score_b): rows in a fixed order (sorted by
    CELL), then per (arm, tier, agreement) a uniform draw without replacement from that cell's own stream."""
    df = df.assign(agreement=np.where(df["score_a"] == df["score_b"], "agree", "disagree"))
    df = df.sort_values(CELL, kind="mergesort").reset_index(drop=True)
    counts, picks = {}, []
    for arm in arms or TASKS[task]:
        for tier in TIERS:
            for ag, size in sizes:
                grp = df[(df["arm"] == arm) & (df["tier"] == tier) & (df["agreement"] == ag)]
                counts[f"{task}/{arm}/{tier}/{ag}"] = int(len(grp))
                if not len(grp):
                    continue
                rng = sampling.rng_for("scoring-audit-v2", task, arm, tier, ag)
                idx = rng.choice(len(grp), size=min(size, len(grp)), replace=False)
                picks.append(grp.iloc[np.sort(idx)].assign(task=task))
    return (pd.concat(picks, ignore_index=True) if picks else df.iloc[:0].assign(task=task)), counts


def main():
    if os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST} exists; the audit is drawn once")
    pops = populations()
    counts, picks, scorers = {}, [], {}
    for task, (df, a_name, b_name) in pops.items():
        scorers[task] = [a_name, b_name]
        tiers = tiers_for(sorted(df["model_key"].unique()))
        p, c = draw_cells(df.assign(tier=df["model_key"].map(tiers)), task)
        picks.append(p)
        counts.update(c)
    pick = pd.concat(picks, ignore_index=True)
    pick["config_id"] = pick["config_id"].astype(str)
    pick = texts(pick)
    it = items()
    q = pd.concat([v.assign(task=t) for t, v in it.items()], ignore_index=True)
    pick = pick.merge(q, on=["task", "item_id"], how="left", validate="many_to_one")
    if pick["question"].isna().any():
        raise SystemExit("drawn outputs without a question")
    order = sampling.rng_for("scoring-audit-v2", "order").permutation(len(pick))
    pick = pick.iloc[order].reset_index(drop=True)
    pick["audit_id"] = [f"V{i + 1:03d}" for i in range(len(pick))]

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(KEY), exist_ok=True)
    sheet = pick[["audit_id", "question", "gold_answer", "output_text"]].copy()
    for c in ("final_answer_as_read", "correct", "notes"):
        sheet[c] = ""
    sheet.to_csv(os.path.join(OUT, "coding_sheet.csv"), index=False, encoding="utf-8-sig")
    keep = ["audit_id", "task", "request_id", "model_key", "arm", "config_id", "item_id", "tier", "agreement",
            "score_a", "score_b"] + [c for c in ("content_class", "pred") if c in pick]
    pick[keep].to_csv(KEY, index=False)
    man = {"label": "scoring_audit_v2", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": "scripts/draw_scoring_audit_v2.py docstring",
           "per_cell": {"disagree": DISAGREE, "agree": AGREE},
           "streams": "sampling.rng_for('scoring-audit-v2', task, arm, tier, agreement) and ('scoring-audit-v2', 'order')",
           "scorers": {t: {"score_a": a, "score_b": b} for t, (a, b) in scorers.items()},
           "population_counts": counts,
           "rows": pick[["audit_id", "task", "request_id", "model_key", "arm", "config_id", "item_id", "tier",
                         "agreement"]].to_dict(orient="records")}
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    write_instructions(len(pick))
    print(f"drew {len(pick)} outputs: {pick.groupby(['task', 'agreement']).size().to_dict()}")


def write_instructions(n):
    txt = f"""# Rating model answers

Thank you for helping. You will see {n} answers that AI models gave to questions. Most are school math word problems;
the rest are multiple-choice or short reasoning questions.

Each answer comes with the correct answer. Your job is to say whether the model's final answer matches it. You don't
need to solve the problem or check the working.

How to read an answer:

- Start at the end of the model's text. The final answer is usually there. Read further up only if the end doesn't
  make it clear.
- If the model changes its mind, the answer it finally settles on counts.
- Small differences don't matter: 12, 12.00, $12 and "12 apples" are the same answer. For multiple choice, the right
  letter and the right option text both count.
- Some answers begin with `\\boxed{{`. That is part of the model's answer, so read it like the rest.

Then pick one:

- **Matches the correct answer.**
- **A different answer.** The model settles on an answer that is not the correct one.
- **No final answer.** It stops, refuses, or lists several answers without choosing one.
- **Can't tell.** Only when you really cannot decide.

Each answer also shows how an LLM rater read it, and marks the choice the LLM rater made. The LLM rater can be wrong.
Make your own call from the model's answer; when you agree, pick the same choice.

If something looks odd you can also tick a box: the model changes its answer, the answer appears only inside the
working, the text is cut off, or the format is odd. This is optional.

Please rate on your own and don't look anything up. It takes about an hour, and the page keeps your ratings as you go,
so you can stop and come back. When you finish, click **Save ratings** and send back the file `ratings.csv`.
"""
    for name in ("instructions.md", "instructions_clean.md"):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
