r"""The forced-stop completion rule against the API models' full replies.

Mistral Large 3 (data/ladder_api/mistral-large-2512/raw.jsonl) and DeepSeek V4.1-Flash-1017 on Nebius
(data/frontier_panel/nebius_panel_raw.jsonl) ran without the stop at the first "}"; their GSM8K and BIG-Bench Hard
forced replies are cut after the first "}" at scoring, to the text the stop returns, and the raw records keep
the whole reply. For every forced reply that the cut leaves with its box open, this script compares

  the pipeline's reading: src/forced_stop.close on the cut text, read by src/content_scoring.classify, and
  the full reply's reading: the prefill's box as the model closed it in the whole reply, read the same way; a box
  that holds two options ("({A}, {F})") is its own class, never correct;

under each rule version (forced-stop/1.0.0, 1.1.0 and 1.2.0; the content reading is the current one for all). Counts per model, benchmark, arm and the rule's class
(completed, left open): replies; full box never closed (the token cap or the model's end of turn came first; no ground
truth); agree (both correct or both not); false gains (the pipeline reads correct, the full box does not); false losses
(the full box reads correct, the pipeline does not); and the same answer read. Unread: the replies
the pipeline reads as a malformed box (left open and not read: open_unfinished, open_other, and open_range replies
the content reader does not read), with those compared and their false losses.

PowerShell:  .\.venv\Scripts\python.exe scripts\forced_stop_full_replies.py
Writes results/forced_stop/full_replies.json, full_replies.md and full_replies_clean.md.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import content_scoring as cs, forced_stop as fs  # noqa: E402

OUT = ROOT / "results" / "forced_stop"
PRE = fs.BOX_PREFILL
TASKS = ("gsm8k", "bbh")
FORCED = ("forced_direct", "fewshot_forced")
ARM_NAME = {"forced_direct": "zero-shot forced", "fewshot_forced": "five-shot forced"}
TASK_NAME = {"gsm8k": "GSM8K", "bbh": "BIG-Bench Hard"}
MODELS = {"mistral-large-2512": "Mistral Large 3", "deepseek-v4.1-flash-1017": "DeepSeek V4.1-Flash-1017"}
OPEN_KINDS = ("completed", "open_inner_box", "open_range", "open_unfinished", "open_other")


def raw_rows() -> dict:
    import mistral_ladder as ml
    import nebius_panel as npl
    out = {}
    for key, raw in (("mistral-large-2512", ml.read_raw()), ("deepseek-v4.1-flash-1017", npl.read_raw())):
        seen, rows = set(), []
        for r in raw:
            if r.get("status") != 200 or not isinstance(r.get("text"), str):
                continue
            if r["task"] not in TASKS or r["arm"] not in FORCED or r["key"] in seen:
                continue
            seen.add(r["key"])
            rows.append(r)
        out[key] = rows
    return out


def gold_tables() -> dict:
    import score_content as SC
    return {t: {str(k): v for k, v in SC.gold_table(t).items()} for t in TASKS}


def full_box(full: str) -> str | None:
    """The content of the prefill's box as the model closed it in the whole reply (a backslash escapes the next
    character), or None when it never closes."""
    depth, i = 1, len(PRE)
    while i < len(full):
        c = full[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return full[len(PRE):i]
        i += 1
    return None


def read(text: str, g: tuple, task: str) -> tuple[str, str | None]:
    return cs.classify(text, "stop", g[0], task, g[1], opts=g[2], task_name=g[3])


def full_reading(text: str, g: tuple, task: str) -> tuple[str, str | None]:
    b = full_box(PRE + text)
    if b is None:
        return "not closed", None
    if fs._TWO_OPTIONS.search(fs.plain(PRE + b)):
        return "two options", None
    return read(PRE + b + "}", g, task)


def compare(rows: dict, gold: dict) -> dict:
    cells = defaultdict(Counter)
    examples = defaultdict(list)
    for model, rs in rows.items():
        for r in rs:
            t, task = r["text"], r["task"]
            j = t.find("}")
            cut = PRE + (t[: j + 1] if j >= 0 else t)
            g = gold[task][str(r["item_id"])]
            fr = None
            for v in fs.VERSIONS:
                kind = fs.classify(cut, version=v)
                if kind not in OPEN_KINDS:
                    break
                if fr is None:
                    fr = full_reading(t, g, task)
                pr = read(fs.close(cut, version=v), g, task)
                c = cells[(v, model, task, r["arm"], kind)]
                c["replies"] += 1
                unread = pr[0] == "malformed_box"
                c["unread"] += unread
                if fr[0] == "not closed":
                    c["full_not_closed"] += 1
                    continue
                c["compared"] += 1
                pc, fc = pr[0] == "correct", fr[0] == "correct"
                c["agree"] += pc == fc
                c["false_gain"] += pc and not fc
                c["false_loss"] += fc and not pc
                c["same_answer"] += (pr[0] == fr[0]) and (pr[1] == fr[1])
                c["unread_compared"] += unread
                c["unread_false_loss"] += unread and fc
                if pc != fc:
                    lab = "false_gain" if pc else "false_loss"
                    ex = examples[(v, model, lab)]
                    if len(ex) < 8:
                        ex.append({"task": task, "arm": r["arm"], "kind": kind, "cut": cut[len(PRE):],
                                   "full_box": full_box(PRE + t), "pipeline": pr, "full": fr})
    return {"cells": cells, "examples": examples}


def totals(cells: dict, v: str, model: str | None = None, kinds=None) -> Counter:
    tot = Counter()
    for (vv, m, _t, _a, k), c in cells.items():
        if vv == v and (model is None or m == model) and (kinds is None or k in kinds):
            tot.update(c)
    return tot


def summary(res: dict) -> dict:
    cells = res["cells"]
    out = {"created": time.strftime("%Y-%m-%d %H:%M"), "versions": list(fs.VERSIONS), "rule": fs.RULE,
           "content_scorer_version": cs.CONTENT_SCORER_VERSION,
           "cells": [{"version": v, "model": m, "task": t, "arm": a, "kind": k, **dict(c)}
                     for (v, m, t, a, k), c in sorted(cells.items())],
           "totals": {}, "examples": {f"{v}|{m}|{lab}": ex for (v, m, lab), ex in sorted(res["examples"].items())}}
    for v in fs.VERSIONS:
        out["totals"][v] = {
            "all": dict(totals(cells, v)),
            "completed": dict(totals(cells, v, kinds=("completed",))),
            "left_open": dict(totals(cells, v, kinds=("open_inner_box", "open_range", "open_unfinished",
                                                      "open_other"))),
            **{m: {"all": dict(totals(cells, v, m)), "completed": dict(totals(cells, v, m, ("completed",)))}
               for m in MODELS}}
    return out


def fmt(n) -> str:
    return f"{int(n):,}"


def shown(s) -> str:
    """A reply's text for a Markdown code span: a tab written as <TAB>, a line break as \\n."""
    return str(s).replace("\t", "<TAB>").replace("\r", "\\r").replace("\n", "\\n")


def write_md(S: dict) -> str:
    L = ["# The completion rule against the API models' full replies", "",
         "Mistral Large 3 and DeepSeek V4.1-Flash-1017 ran without the stop at the first `}`; their GSM8K and "
         "BIG-Bench Hard forced replies are cut after it at scoring, and their raw records keep the whole reply. For "
         "every forced reply the cut leaves with its box open, the pipeline's reading (`src/forced_stop.close`, then "
         "`src/content_scoring.classify`) is compared with the reading of the box the model closed in the whole reply "
         "(a full box holding two options is its own class, never correct). Written by "
         "`scripts/forced_stop_full_replies.py`.", "",
         "Columns: replies the cut left open; full box not closed (no ground truth: the token cap or the model's end "
         "of turn came first); compared; agree on correct / not correct; false gains (the pipeline reads correct, "
         "the full box does not); false losses (the full box reads correct, the pipeline does not); same answer "
         "(same class and answer).", ""]
    for v in S["versions"]:
        L += [f"## Rule {v}", "", "| Model | Benchmark | Arm | Class | Replies | Full not closed | Compared | Agree | "
              "False gains | False losses | Same answer |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in S["cells"]:
            if c["version"] != v:
                continue
            L.append(f"| {MODELS[c['model']]} | {TASK_NAME[c['task']]} | {ARM_NAME[c['arm']]} | {c['kind']} | "
                     f"{fmt(c.get('replies', 0))} | {fmt(c.get('full_not_closed', 0))} | {fmt(c.get('compared', 0))} | "
                     f"{fmt(c.get('agree', 0))} | {fmt(c.get('false_gain', 0))} | {fmt(c.get('false_loss', 0))} | "
                     f"{fmt(c.get('same_answer', 0))} |")
        L += ["", "Totals:", "", "| Set | Replies | Full not closed | Compared | Agree | False gains | False losses | "
              "Same answer |", "|---|---|---|---|---|---|---|---|"]
        T = S["totals"][v]
        rows = [("all", T["all"]), ("completed", T["completed"]), ("left open", T["left_open"])]
        rows += [(f"{MODELS[m]}, all", T[m]["all"]) for m in MODELS]
        rows += [(f"{MODELS[m]}, completed", T[m]["completed"]) for m in MODELS]
        for name, t in rows:
            L.append(f"| {name} | {fmt(t.get('replies', 0))} | {fmt(t.get('full_not_closed', 0))} | "
                     f"{fmt(t.get('compared', 0))} | {fmt(t.get('agree', 0))} | {fmt(t.get('false_gain', 0))} | "
                     f"{fmt(t.get('false_loss', 0))} | {fmt(t.get('same_answer', 0))} |")
        L.append("")
    L += ["## Examples of disagreement (reply after the prefill, cut; the full box)", "",
          "A tab in a reply is shown as `<TAB>`, a line break as `\\n`.", ""]
    for k, ex in S["examples"].items():
        for e in ex[:4]:
            L.append(f"- {k}: {TASK_NAME[e['task']]}, {ARM_NAME[e['arm']]}, {e['kind']}: cut `{shown(e['cut'])}`, "
                     f"full box `{shown(e['full_box'])}`; pipeline {e['pipeline'][0]} ({e['pipeline'][1]}), full "
                     f"{e['full'][0]} ({e['full'][1]})")
    return "\n".join(L) + "\n"


def main() -> int:
    t0 = time.time()
    S = summary(compare(raw_rows(), gold_tables()))
    S["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_replies.json").write_text(json.dumps(S, indent=1, ensure_ascii=False, default=list) + "\n",
                                           encoding="utf-8")
    md = write_md(S)
    (OUT / "full_replies.md").write_text(md, encoding="utf-8")
    (OUT / "full_replies_clean.md").write_text(md, encoding="utf-8")
    for v in fs.VERSIONS:
        print(v, json.dumps(S["totals"][v]["all"]))
    print(f"done in {S['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
