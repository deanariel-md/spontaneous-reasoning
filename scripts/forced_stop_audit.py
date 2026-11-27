r"""Where the forced arms' stop at the first "}" left the box open.

Reads every stored forced reply, panel and ladder, from the generation folders and the API records, as generated (the
prefill put in front, no completion), and counts per collection, benchmark, arm and model:
  rows           forced replies;
  ended_on_stop  replies that end on the stop string "}" (the arms that stop at "}", or are cut after it);
  open_box       of these, replies whose \boxed{ the stop left open;
  completed      of these, replies src/forced_stop.close completes (the stop closed an answer wrapper);
  open_inner_box left open, but the stop closed a nested \boxed{...} that every reader reads as a complete inner box;
  open_inner_box_restart  of these, the replies where a closing reasoning tag "</think>" stands between the prefill
                 and the inner box (the model closed a reasoning block and began a new box; read as any
                 inner box is);
  open_range     left open (rule 1.2.0): a clock time and "to" ("8pm \text{ to }"), a time range the stop cut before its
                 end; the content reader reads it on temporal_sequences as the option the range begins (content/1.6.0;
                 open_range_read, and open_range_read_correct of these), the rest is unread (open_range_unread);
  open_unfinished left open (rule 1.1.0): the stop closed an answer wrapper, but the box's text asks for more ("10
                 \text{ blue and }") or holds two options ("({A, D}");
  open_other     left open: the group the stop closed is not an answer wrapper (\frac numerator, "{,}", \ref{...}, an
                 exponent), so part of the answer may be missing;
  completed_v100 the replies rule 1.0.0 completed, for comparison;
and, for the replies the rule completes, the content class before and after (src/content_scoring.classify) and
whether the strict and lenient readings change (they must not). Forms: the command of the group the stop closed.
MATH-500 and HotpotQA ran without the stop and are never cut; their forced replies are counted too (rows, open boxes
by finish reason) to show that no stop cut them.

Sources:
  forced direct                  data/d2_out/forced/{gsm8k,bbh}/<key>, data/d2_out/math/forced/<key>
  five-shot forced               data/d2_out/fewshot/{gsm8k,bbh,math}/<key>
  constrained forced             data/d2_out/forced_constrained/bbh/<key>
  arm wording                    data/d2_out/arm_wording/collection/<key>, data/d2_out/arm_wording_extra/collection/<key>
                                 (GSM8K forced replies cut after the first "}" at scoring, src/arm_wording.score)
  HotpotQA                       data/d2_out/hotpot/block_{a,b}/<key>
  ladder, GPU runs               data/d2_out/ladder2/<run key>/<task>
  ladder, DeepSeek on Nebius     data/frontier_panel/outputs/<task>.parquet (cut after the first "}" as stored)
  ladder, Mistral Large 3        data/ladder_api/mistral-large-2512/raw.jsonl (scripts/mistral_ladder.outputs_frame)

PowerShell:  .\.venv\Scripts\python.exe scripts\forced_stop_audit.py
Writes results/forced_stop/audit.json, audit.md and audit_clean.md, and unread_by_config.csv: per collection,
benchmark, arm, model and configuration, the forced rows and the replies the stop left unread (open_unfinished,
open_other and open_range_unread), the stop's loss that no rule recovers (scripts/ladder_analysis.py bounds its family steps by it).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import bbh, content_scoring as cs, forced_stop as fs, scoring  # noqa: E402

OUT = ROOT / "results" / "forced_stop"
D2 = ROOT / "data" / "d2_out"
PREFILL = fs.BOX_PREFILL
FORCED = ("forced_direct", "fewshot_forced")
TASK_NAME = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
ARM_NAME = {"forced_direct": "zero-shot forced", "fewshot_forced": "five-shot forced"}
KINDS = ("completed", "open_inner_box", "open_range", "open_unfinished", "open_other")
UNREAD = ("open_unfinished", "open_other", "open_range_unread")     # open_range_unread: open_range not read (1.2.0)


def _gold(task):
    import score_content as SC
    g = SC.gold_table(task)
    return {str(k): v for k, v in g.items()}


GOLD = {}


def gold(task):
    if task not in GOLD:
        GOLD[task] = _gold(task)
    return GOLD[task]


def decoding(folder: Path) -> dict:
    logs = sorted(folder.glob("run_log-*.json"))
    decs = {json.dumps((json.load(open(p, encoding="utf-8")).get("decoding") or {}), sort_keys=True) for p in logs}
    if len(decs) != 1:
        raise SystemExit(f"{folder}: {len(decs)} different decodings in its run logs")
    return json.loads(next(iter(decs)))


def shards(folder: Path, arms) -> pd.DataFrame:
    parts = []
    for sh in sorted(folder.glob("shard-*.parquet")):
        t = pq.read_table(sh, columns=["model_key", "arm", "config_id", "item_id", "output_text",
                                       "finish_reason"]).to_pandas()
        parts.append(t[t["arm"].isin(arms)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def task_of_arm(arm: str, default: str) -> tuple[str, str]:
    """('gsm8k', 'forced_direct') from a pilot-style label 'gsm8k.forced_direct', else (default, arm)."""
    return tuple(arm.split(".", 1)) if "." in arm else (default, arm)


class Tally:
    def __init__(self):
        self.cells = defaultdict(Counter)
        self.content = defaultdict(Counter)       # (cell) -> Counter((before, after))
        self.forms = defaultdict(Counter)         # (collection, task) -> Counter(command)
        self.examples = defaultdict(list)
        self.strict_changed = Counter()
        self.lenient_changed = Counter()
        self.by_config = defaultdict(Counter)     # (collection, task, arm, model, config) -> rows, unread, unfinished
        self.endings = defaultdict(Counter)       # (collection, task) -> last word or sign of 1.0.0's completions
        self.unfinished = defaultdict(Counter)    # (collection, task) -> last word or sign of open_unfinished replies

    def add(self, coll, task, arm, model, texts, finish, items, stops: bool, cut_first_brace: bool = False,
            configs=None):
        cell = (coll, task, arm, model)
        c = self.cells[cell]
        g = gold(task) if task in ("gsm8k", "bbh", "math") else None
        configs = configs if configs is not None else [None] * len(texts)
        for t, f, i, cf in zip(texts, finish, items, configs):
            t = t if isinstance(t, str) else ""
            if cut_first_brace:
                j = t.find("}")
                if j >= 0:
                    t, f = t[: j + 1], "stop"
            full = PREFILL + t
            c["rows"] += 1
            bc = self.by_config[(coll, task, arm, model, None if cf is None else str(cf))]
            bc["rows"] += 1
            if not stops:
                if not fs.groups(full)[0] == "closed" and not bbh.boxed_contents(full):
                    c[f"open_box_no_stop_{f}"] += 1
                continue
            k = fs.classify(full)
            if k == "no_stop":
                continue
            c["ended_on_stop"] += 1
            if k == "closed":
                continue
            c["open_box"] += 1
            c[k] += 1
            if fs.classify(full, version=fs.V100) == "completed":
                c["completed_v100"] += 1
                self.endings[(coll, task)][ending(full)] += 1
            if k == "open_range":
                # forced-stop/1.2.0: a time range cut at "to"; unread unless the content reader reads it
                # (content/1.6.0, on temporal_sequences: the option the range begins)
                gi = g[str(i)] if g is not None else None
                cls = cs.classify(full, f, gi[0], task, gi[1], opts=gi[2], task_name=gi[3])[0] if gi else None
                read = cls in ("correct", "wrong")
                c["open_range_read" if read else "open_range_unread"] += 1
                c["open_range_read_correct"] += cls == "correct"
                if not read:
                    bc["unread"] += 1
            if k in UNREAD:
                bc["unread"] += 1
            if k == "open_unfinished":
                bc["unfinished"] += 1
                self.unfinished[(coll, task)][ending(full)] += 1
            if k == "open_inner_box" and "</think>" in full[len(PREFILL):full.find(PREFILL, len(PREFILL))]:
                c["open_inner_box_restart"] += 1     # a new box begun after a closed reasoning block
            self.forms[(coll, task)][fs.closed_group(full)] += 1
            if len(self.examples[(coll, task, k)]) < 6 and t not in self.examples[(coll, task, k)]:
                self.examples[(coll, task, k)].append(t)
            if k != "completed" or g is None:
                continue
            new = fs.close(full)
            gi = g[str(i)]
            before = cs.classify(full, f, gi[0], task, gi[1], opts=gi[2], task_name=gi[3])[0]
            after = cs.classify(new, f, gi[0], task, gi[1], opts=gi[2], task_name=gi[3])[0]
            self.content[cell][(before, after)] += 1
            if task == "bbh":
                a = gi[1]
                if bbh.strict_extract(full, a) != bbh.strict_extract(new, a):
                    self.strict_changed[cell] += 1
                if bbh.lenient_extract(full, a) != bbh.lenient_extract(new, a):
                    self.lenient_changed[cell] += 1
            else:
                if scoring.strict_extract(full, SPEC) != scoring.strict_extract(new, SPEC):
                    self.strict_changed[cell] += 1
                if scoring.lenient_extract(full) != scoring.lenient_extract(new):
                    self.lenient_changed[cell] += 1


SPEC = None


def ending(full: str) -> str:
    """The last word or sign of a box's text so far (src/forced_stop.plain), lower case; a single capital letter is
    'a letter', digits 'a number'."""
    p = fs.plain(full)
    m = re.search(r"([A-Za-z]+|[0-9]+|[^A-Za-z0-9\s])$", p)
    if not m:
        return "(empty)"
    w = m.group(1)
    if len(w) == 1 and w.isupper():
        return "a letter"
    return "a number" if w.isdigit() else w.casefold()


def run_folders(T: Tally, coll: str, pattern: str, task: str | None, cut_gsm8k: bool = False):
    for d in sorted(Path(p) for p in glob.glob(str(D2 / pattern)) if Path(p).is_dir()):
        if not list(d.glob("run_log-*.json")):
            continue
        dec = decoding(d)
        pre = dec.get("prefill") or {}
        arms = [a for a in pre if task_of_arm(a, task or "")[1] in FORCED]
        if not arms:
            continue
        df = shards(d, arms)
        if df.empty:
            continue
        for arm, g in df.groupby("arm"):
            if pre[arm] != PREFILL:
                raise SystemExit(f"{d}: prefill {pre[arm]!r}")
            if task is None:     # arm wording: the task is the first part of config_id
                for tk, gg in g.groupby(g["config_id"].astype(str).str.split("/").str[0]):
                    T.add(coll, tk, arm, str(gg["model_key"].iloc[0]), gg["output_text"].tolist(),
                          gg["finish_reason"].tolist(), gg["item_id"].tolist(),
                          stops=(tk == "gsm8k" and cut_gsm8k) or fs.stops_at_brace(dec, arm),
                          cut_first_brace=(tk == "gsm8k" and cut_gsm8k), configs=gg["config_id"].tolist())
            else:
                tk, a = task_of_arm(arm, task)
                T.add(coll, tk, a, str(g["model_key"].iloc[0]), g["output_text"].tolist(), g["finish_reason"].tolist(),
                      g["item_id"].tolist(), stops=fs.stops_at_brace(dec, arm), configs=g["config_id"].tolist())
        print(f"{coll} {d.relative_to(D2)}: {len(df)} forced rows", flush=True)


def run_ladder(T: Tally):
    chosen = D2 / "ladder2" / "chosen"
    for key_file in sorted(chosen.iterdir()):
        rk = key_file.read_text(encoding="utf-8").strip()
        for task in ("gsm8k", "math", "bbh", "hotpot"):
            d = D2 / "ladder2" / rk / task
            dec = decoding(d)
            df = shards(d, list(FORCED))
            for arm, g in df.groupby("arm"):
                T.add("ladder", task, arm, key_file.name, g["output_text"].tolist(), g["finish_reason"].tolist(),
                      g["item_id"].tolist(), stops=fs.stops_at_brace(dec, arm), configs=g["config_id"].tolist())
            print(f"ladder {rk}/{task}: {len(df)} forced rows", flush=True)


def run_nebius(T: Tally):
    import nebius_panel as npl
    for task in ("gsm8k", "math", "bbh", "hotpot"):
        o = pd.read_parquet(ROOT / "data" / "frontier_panel" / "outputs" / f"{task}.parquet",
                            columns=["model_key", "arm", "config_id", "item_id", "output_text", "finish_reason"])
        o = o[o["arm"].isin(FORCED)]
        for arm, g in o.groupby("arm"):
            T.add("ladder", task, arm, "deepseek-v4.1-flash-1017", g["output_text"].tolist(),
                  g["finish_reason"].tolist(), g["item_id"].tolist(), stops=task in npl.CUT_AT_FIRST_BRACE,
                  configs=g["config_id"].tolist())
        print(f"nebius {task}: {len(o)} forced rows", flush=True)


def run_mistral(T: Tally):
    import mistral_ladder as ml
    raw = ml.read_raw()
    for task in ("gsm8k", "math", "bbh", "hotpot"):
        o = ml.outputs_frame(raw, task)
        o = o[o["arm"].isin(FORCED)]
        for arm, g in o.groupby("arm"):
            T.add("ladder", task, arm, ml.MODEL_KEY, g["output_text"].tolist(), g["finish_reason"].tolist(),
                  g["item_id"].tolist(), stops=task in ml.CUT_AT_FIRST_BRACE, configs=g["config_id"].tolist())
        print(f"mistral {task}: {len(o)} forced rows", flush=True)


COLLECTIONS = {
    "forced": "zero-shot forced direct (main panel)",
    "fewshot": "five-shot forced (main panel)",
    "forced_constrained": "constrained forced direct (main panel)",
    "arm_wording": "arm wording (main panel)",
    "arm_wording_extra": "arm wording, default and firm clauses (main panel)",
    "hotpot": "HotpotQA (main panel)",
    "ladder": "size ladder: four GPU-run models, DeepSeek on Nebius, Mistral Large 3",
}


def summarise(T: Tally) -> dict:
    rows = []
    for (coll, task, arm, model), c in sorted(T.cells.items()):
        cc = T.content.get((coll, task, arm, model), Counter())
        moved = {f"{a}->{b}": n for (a, b), n in sorted(cc.items()) if a != b}
        rows.append({"collection": coll, "task": task, "arm": arm, "model": model, "rows": c["rows"],
                     "ended_on_stop": c["ended_on_stop"], "open_box": c["open_box"],
                     **{k: c[k] for k in KINDS}, "completed_v100": c["completed_v100"],
                     "open_range_read": c["open_range_read"], "open_range_unread": c["open_range_unread"],
                     "open_range_read_correct": c["open_range_read_correct"],
                     "open_inner_box_restart": c["open_inner_box_restart"],
                     "unread": sum(c[k] for k in UNREAD),
                     "open_box_no_stop": {k[len("open_box_no_stop_"):]: v for k, v in c.items()
                                          if k.startswith("open_box_no_stop_")},
                     "content_before_after": {f"{a}->{b}": n for (a, b), n in sorted(cc.items())},
                     "content_moved": moved,
                     "content_correct_gained": sum(n for (a, b), n in cc.items() if b == "correct" and a != "correct"),
                     "strict_changed": T.strict_changed.get((coll, task, arm, model), 0),
                     "lenient_changed": T.lenient_changed.get((coll, task, arm, model), 0)})
    return {"created": time.strftime("%Y-%m-%d %H:%M"), "forced_stop_version": fs.VERSION, "rule": fs.RULE,
            "content_scorer_version": cs.CONTENT_SCORER_VERSION, "collections": COLLECTIONS, "cells": rows,
            "forms": {f"{c}/{t}": dict(v.most_common()) for (c, t), v in sorted(T.forms.items())},
            "endings_completed_v100": {f"{c}/{t}": dict(v.most_common()) for (c, t), v in sorted(T.endings.items())},
            "endings_unfinished": {f"{c}/{t}": dict(v.most_common()) for (c, t), v in sorted(T.unfinished.items())},
            "examples": {f"{c}/{t}/{k}": v for (c, t, k), v in sorted(T.examples.items())}}


def fmt(n):
    return f"{n:,}"


def form_label(c) -> str:
    if c is None:
        return "no group closed"
    if c == fs.LEAD_GROUP:
        return "bare group opening the box"
    if c == "":
        return "bare group after other text"
    return "`" + chr(92) + c + "`"


def arm_label(coll: str, arm: str) -> str:
    """The five-shot collection (data/d2_out/fewshot) labels its forced arm forced_direct."""
    return "five-shot forced" if coll == "fewshot" else ARM_NAME.get(arm, arm)


def write_md(res: dict) -> str:
    cells = pd.DataFrame(res["cells"])
    L = ["# Forced replies the stop at the first \"}\" left open", "",
         f"Rule {res['forced_stop_version']} (`src/forced_stop.py`): {res['rule']}. Content reading "
         f"{res['content_scorer_version']}. Counts from every stored forced reply, as generated (prefill in front, no "
         "completion). Written by `scripts/forced_stop_audit.py`.", "",
         "Columns: rows, the forced replies; on stop, replies that end on the stop string \"}\"; open, of these the "
         "replies whose box the stop left open; completed, read as the closed box by the rule; inner box, left open "
         "but the stop closed a nested `\\boxed{...}` that every reader already reads; range, left open because the "
         "box's text is a clock time and \"to\" (rule 1.2.0), read by the content reader on temporal_sequences as the "
         "option the range begins (in brackets: read, and of these right), unread otherwise; unfinished, left open because "
         "the box's text asks for more or holds two options (rule 1.1.0); other, left open because the group the "
         "stop closed is not an answer wrapper; completed by 1.0.0, the replies rule 1.0.0 "
         "completed; content gained, completed replies whose content class goes to correct; strict and lenient "
         "changed, completed replies whose strict or lenient answer changes (0 by construction).", ""]
    for coll, desc in res["collections"].items():
        c = cells[cells["collection"] == coll]
        if c.empty:
            continue
        L += [f"## {desc}", "", "| Benchmark | Arm | Model | Rows | On stop | Open | Completed | Inner box | "
              "Range (read, right) | Unfinished | Other | Completed by 1.0.0 | Content gained | Strict / lenient "
              "changed |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in c.sort_values(["task", "arm", "model"]).itertuples():
            if r.open_box == 0 and r.ended_on_stop == 0:
                continue
            L.append(f"| {TASK_NAME[r.task]} | {arm_label(coll, r.arm)} | {r.model} | {fmt(r.rows)} | "
                     f"{fmt(r.ended_on_stop)} | {fmt(r.open_box)} | {fmt(r.completed)} | {fmt(r.open_inner_box)} | "
                     f"{fmt(r.open_range)} ({fmt(r.open_range_read)}, {fmt(r.open_range_read_correct)}) | "
                     f"{fmt(r.open_unfinished)} | {fmt(r.open_other)} | {fmt(r.completed_v100)} | "
                     f"{fmt(r.content_correct_gained)} | {r.strict_changed} / {r.lenient_changed} |")
        cols = ["rows", "ended_on_stop", "open_box", "completed", "open_inner_box", "open_range", "open_unfinished",
                "open_other", "completed_v100", "content_correct_gained"]
        tot = c.groupby(["task", "arm"])[cols].sum()
        L += ["", "Totals by benchmark and arm (models with no open box included in the rows):", "",
              "| Benchmark | Arm | Rows | On stop | Open | Completed | Inner box | Range | Unfinished | Other | "
              "Completed by 1.0.0 | Content gained |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (t, a), r in tot.iterrows():
            L.append(f"| {TASK_NAME[t]} | {arm_label(coll, a)} | " + " | ".join(fmt(r[k]) for k in cols) + " |")
        L.append("")
    nostop = cells[cells["task"].isin(["math", "hotpot"])]
    L += ["## MATH-500 and HotpotQA", "",
          f"No forced arm of MATH-500 or HotpotQA ran with the stop or was cut after a \"}}\": {fmt(int(nostop['rows'].sum()))} "
          f"forced replies, {fmt(int(nostop['ended_on_stop'].sum()))} ended on a stop. Their boxes left open come from "
          "the 16-token (MATH-500) or 64-token (HotpotQA) cap or from the model's own end of turn, not from a stop:", "",
          "| Collection | Benchmark | Arm | Rows | Open box, cap reached | Open box, ended by the model |",
          "|---|---|---|---|---|---|"]
    for (coll, t, a), g in nostop.groupby(["collection", "task", "arm"]):
        ob = Counter()
        for d in g["open_box_no_stop"]:
            ob.update(d)
        L.append(f"| {coll} | {TASK_NAME[t]} | {arm_label(coll, a)} | {fmt(int(g['rows'].sum()))} | "
                 f"{fmt(ob.get('length', 0))} | {fmt(sum(v for k, v in ob.items() if k != 'length'))} |")
    L += ["", "## Forms: the group the stop closed, in the replies it left open", ""]
    for k, v in res["forms"].items():
        L.append(f"- {k}: " + ", ".join(f"{form_label(c)} {fmt(n)}" for c, n in v.items()))
    L += ["", "## Endings: the last word or sign of the box's text", "",
          "Rule 1.1.0 leaves a reply open when its text ends in " + ", ".join(f"\"{w}\"" for w in fs.CONTINUATION_WORDS)
          + " or " + ", ".join(f"\"{w}\"" for w in fs.CONTINUATION_SIGNS) + ". The endings of every reply rule 1.0.0 "
          "completed (a single capital letter counted as \"a letter\", digits as \"a number\"), and of the replies 1.1.0 "
          "leaves open as unfinished:", ""]
    for k, v in res["endings_completed_v100"].items():
        L.append(f"- completed by 1.0.0, {k}: " + ", ".join(f"{w} {fmt(n)}" for w, n in v.items()))
    for k, v in res["endings_unfinished"].items():
        L.append(f"- unfinished (1.1.0), {k}: " + ", ".join(f"{w} {fmt(n)}" for w, n in v.items()))
    L += ["", "## Examples (reply after the prefill)", ""]
    for k, v in res["examples"].items():
        L.append(f"- {k}: " + "; ".join(f"`{x}`" for x in v))
    return "\n".join(L) + "\n"


def write_by_config(T: Tally, path: Path) -> int:
    """unread_by_config.csv: rows and unread replies per collection, benchmark, arm, model and configuration (forced
    arms that stop at the first "}"; MATH-500 and HotpotQA, which never stop there, are left out)."""
    rows = [{"collection": c, "task": t, "arm": a, "model": m, "config_id": cf, "rows": v["rows"],
             "unread": v["unread"], "unfinished": v["unfinished"]}
            for (c, t, a, m, cf), v in sorted(T.by_config.items(), key=lambda kv: tuple(map(str, kv[0])))
            if t in ("gsm8k", "bbh")]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def main() -> int:
    global SPEC
    from src import d2_io
    SPEC, _ = d2_io.load_registered_scorer(ROOT / "manifests" / "gsm8k_scorer.json")
    t0 = time.time()
    T = Tally()
    for task in ("gsm8k", "bbh"):
        run_folders(T, "forced", f"forced/{task}/*", task)
    run_folders(T, "forced", "math/forced/*", "math")
    for task in ("gsm8k", "bbh", "math"):
        run_folders(T, "fewshot", f"fewshot/{task}/*", task)
    run_folders(T, "forced_constrained", "forced_constrained/bbh/*", "bbh")
    run_folders(T, "arm_wording", "arm_wording/collection/*", None, cut_gsm8k=True)
    run_folders(T, "arm_wording_extra", "arm_wording_extra/collection/*", None, cut_gsm8k=True)
    for blk in ("block_a", "block_b"):
        run_folders(T, "hotpot", f"hotpot/{blk}/*", "hotpot")
    run_ladder(T)
    run_nebius(T)
    run_mistral(T)
    res = summarise(T)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = write_md(res)
    (OUT / "audit.md").write_text(md, encoding="utf-8")
    (OUT / "audit_clean.md").write_text(md, encoding="utf-8")
    n = write_by_config(T, OUT / "unread_by_config.csv")
    print(f"done in {res['runtime_s']} s -> {OUT} ({n} configuration rows in unread_by_config.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
