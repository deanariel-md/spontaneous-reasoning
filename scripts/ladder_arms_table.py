r"""Supplementary table: each larger model's three arms and two gains, strict and content, beside the panel's median.

Rebuild (standalone; paper/make_numbers.py does not call it):

    python scripts/ladder_arms_table.py

Run it again whenever scripts/ladder_analysis.py rewrites results/ladder/analysis/ladder.json. It writes
  paper/tables/ladder_arms.tex                 the tabular (as the other files of paper/tables/; no float, no caption)
  results/ladder/analysis/ladder_arms.md       the same values in Markdown, with sources, basis and edge cases
  results/ladder/analysis/ladder_arms_clean.md a copy without citation markers

What is read, nothing recomputed that is saved:
- Ladder models, every benchmark and scoring: results/ladder/analysis/ladder.json, models.<key>.tasks.<task>.cut.<scoring>
  .discovery (accuracy_pp, gain_over_requested_pp, gain_over_forced_pp): outputs cut at the panel's caps, means over
  the 24 discovery draws. Where the placement record has the same basis (every cell but HotpotQA content), its value
  is checked equal to it (1e-9).
- Panel median: the placement record's panel_median (ladder.json, placement.<key>.<task>.cut.<scoring>.<metric>),
  basis "discovery". HotpotQA content is the exception: the placements read it on pooled rows (ladder_note.md,
  footnote F6), whereas the reversal counts (ladRev..., combRev..., \aeHqReversedContent) read it on the discovery
  draws. This table keeps the discovery draws there too, so its panel median is the median of the 29 panel models'
  discovery means saved in results/hotpot_content/hotpot_content.json (arm_effect.content.per_model), their gains
  being CoT minus each baseline of those saved means.
- Check: every ladder value equals the value the reversal counts use (ladder_orderings.json, values.cut.<task>.<key>
  .<scoring>.discovery, 1e-9).

Numbers in pp, rounded half-even at one decimal on their exact value (the ladder's rule: scripts/ladder_analysis
.half_even), the exact value being the ratio on the benchmark's discovery grid (ladder.json panel_grid, discovery_pp;
twice that for a median of an even count). A value off its grid is rounded on its float and listed as an edge case.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ladder_analysis as la  # noqa: E402  (half_even, on_grid: the ladder's rounding)

LADDER = "results/ladder/analysis/ladder.json"
ORDERINGS = "results/ladder/analysis/ladder_orderings.json"
HOTPOT_CONTENT = "results/hotpot_content/hotpot_content.json"
TEX = "paper/tables/ladder_arms.tex"
MD = "results/ladder/analysis/ladder_arms.md"
MD_CLEAN = "results/ladder/analysis/ladder_arms_clean.md"

TASKS = ("gsm8k", "math", "bbh", "hotpot")
TASK_LABEL = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
SCORINGS = ("strict", "content")
METRICS = ("requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced")
METRIC_TEX = {"requested": "Requested", "forced": "Forced", "cot": "CoT", "cot_minus_requested": "CoT $-$ req.",
              "cot_minus_forced": "CoT $-$ forced"}
METRIC_MD = {"requested": "Requested", "forced": "Forced", "cot": "CoT", "cot_minus_requested": "CoT − requested",
             "cot_minus_forced": "CoT − forced"}
TOL = 1e-9


def _load(rel: str):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return json.load(fh, parse_constant=lambda c: float("nan"))


def _saved(disc: dict, metric: str) -> float:
    """A metric of a ladder model's saved discovery block (ladder.json models.<key>.tasks.<t>.cut.<s>.discovery)."""
    if metric == "cot_minus_requested":
        return disc["gain_over_requested_pp"]
    if metric == "cot_minus_forced":
        return disc["gain_over_forced_pp"]
    return disc["accuracy_pp"][metric]


def exact(x: float, G: Fraction, edges: list, where: str):
    """x as its exact ratio on the grid 1/G (or 1/2G, a median of an even count); the float itself when off both."""
    for g in (G, 2 * G):
        q = la.on_grid(x, g)
        if q is not None:
            return q
    edges.append(f"{where}: {x!r} is off the discovery grid 1/{G}; rounded on its float")
    return float(x)


def _median(vals: list):
    """Median, exact on exact values (the middle value, or the mean of the two middle ones)."""
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def build() -> dict:
    """Every cell of the table with its exact value and source, plus the checks and edge cases."""
    lad, order, hq = _load(LADDER), _load(ORDERINGS), _load(HOTPOT_CONTENT)
    keys = list(lad["models"])
    edges: list[str] = []
    checks: list[str] = []
    rows, medians = [], {}
    for t in TASKS:
        G = Fraction(str(lad["panel_grid"][t]["discovery_pp"]))
        for k in keys:
            row = {"task": t, "key": k, "name": lad["models"][k]["name"], "cells": {}}
            for s in SCORINGS:
                disc = lad["models"][k]["tasks"][t]["cut"][s]["discovery"]
                for m in METRICS:
                    v = _saved(disc, m)
                    rec = lad["placement"][k][t]["cut"][s][m]
                    if rec["basis"] == "discovery":
                        if abs(rec["value"] - v) > TOL:
                            raise SystemExit(f"{k} {t} {s} {m}: placement value {rec['value']} differs from the saved "
                                             f"discovery value {v}")
                    elif not (t == "hotpot" and s == "content" and rec["basis"] == "pooled"):
                        raise SystemExit(f"{k} {t} {s} {m}: unexpected placement basis {rec['basis']!r}")
                    o = order["values"]["cut"][t][k][s]["discovery"][m]
                    if abs(o - v) > TOL:
                        raise SystemExit(f"{k} {t} {s} {m}: {v} differs from the reversal counts' value {o}")
                    row["cells"][(s, m)] = exact(v, G, edges, f"{lad['models'][k]['name']}, {TASK_LABEL[t]}, {s}, {m}")
            rows.append(row)
        # the panel's median of each column
        first = lad["placement"][keys[0]][t]["cut"]
        for s in SCORINGS:
            if t == "hotpot" and s == "content":
                per = hq["arm_effect"]["content"]["per_model"]
                panel_keys = [r["model_key"] for r in lad["panel"][t]]
                if sorted(per) != sorted(panel_keys):
                    raise SystemExit("HotpotQA: the content per-model means and the ladder's panel differ in models")
                ex = {pk: {a: exact(per[pk][a], G, edges, f"panel {pk}, HotpotQA, content, {a}")
                           for a in ("requested", "forced", "cot")} for pk in panel_keys}
                col = {"requested": [ex[pk]["requested"] for pk in panel_keys],
                       "forced": [ex[pk]["forced"] for pk in panel_keys],
                       "cot": [ex[pk]["cot"] for pk in panel_keys],
                       "cot_minus_requested": [ex[pk]["cot"] - ex[pk]["requested"] for pk in panel_keys],
                       "cot_minus_forced": [ex[pk]["cot"] - ex[pk]["forced"] for pk in panel_keys]}
                for m in METRICS:
                    medians[(t, s, m)] = {"value": _median(col[m]), "n": len(panel_keys),
                                          "source": f"{HOTPOT_CONTENT}, arm_effect.content.per_model (median of 29 "
                                                    "discovery means)"}
                checks.append(f"HotpotQA content: panel median over the {len(panel_keys)} models of "
                              f"{HOTPOT_CONTENT} (arm_effect.content.per_model), the same models as {LADDER} panel.hotpot")
            else:
                for m in METRICS:
                    rec = first[s][m]
                    if rec["basis"] != "discovery":
                        raise SystemExit(f"{t} {s} {m}: placement basis {rec['basis']!r}, not discovery")
                    for k in keys[1:]:
                        if lad["placement"][k][t]["cut"][s][m]["panel_median"] != rec["panel_median"]:
                            raise SystemExit(f"{t} {s} {m}: the placement records disagree on the panel median")
                    medians[(t, s, m)] = {"value": exact(rec["panel_median"], G, edges,
                                                         f"panel median, {TASK_LABEL[t]}, {s}, {m}"),
                                          "n": rec["panel_n"], "source": f"{LADDER}, placement ... panel_median"}
    checks.append("every ladder value equals ladder.json's placement value where the bases agree, and the value of "
                  "ladder_orderings.json that the reversal counts use (1e-9)")
    halfway = []
    for r in rows:
        for (s, m), q in r["cells"].items():
            if isinstance(q, Fraction) and (q * 10 - (q * 10).__floor__()) == Fraction(1, 2):
                halfway.append(f"{r['name']}, {TASK_LABEL[r['task']]}, {s}, {METRIC_MD[m]}: {float(q)!r} prints as {fmt(q)}")
    for (t, s, m), d in medians.items():
        q = d["value"]
        if isinstance(q, Fraction) and (q * 10 - (q * 10).__floor__()) == Fraction(1, 2):
            halfway.append(f"panel median, {TASK_LABEL[t]}, {s}, {METRIC_MD[m]}: {float(q)!r} prints as {fmt(q)}")
    negatives = [f"{r['name']}, {TASK_LABEL[r['task']]}, {s}, {METRIC_MD[m]}: {fmt(q)}"
                 for r in rows for (s, m), q in r["cells"].items() if q < 0]
    negatives += [f"panel median, {TASK_LABEL[t]}, {s}, {METRIC_MD[m]}: {fmt(d['value'])}"
                  for (t, s, m), d in medians.items() if d["value"] < 0]
    return {"rows": rows, "medians": medians, "edges": edges, "halfway": halfway, "negatives": negatives,
            "checks": checks, "ladder_created": lad.get("created"), "orderings_created": order.get("created")}


def fmt(q, minus: str = "-") -> str:
    """One decimal, half-even on the exact value; a negative value that rounds to zero keeps its sign."""
    s = la.half_even(q, 1) if isinstance(q, Fraction) else la.half_even(Fraction(q), 1)
    return s.replace("-", minus)


def tex(T: dict) -> str:
    cols = len(METRICS) * len(SCORINGS) + 1
    out = [f"% Generated by scripts/ladder_arms_table.py from {LADDER} and {HOTPOT_CONTENT} (ladder.json of "
           f"{T['ladder_created']}); rebuild: .venv\\Scripts\\python.exe scripts\\ladder_arms_table.py",
           "% Accuracy and gains in pp, outputs cut at the panel's caps, means over the 24 discovery draws (HotpotQA "
           "content too: the reversal counts' basis); last row of each block: the panel's median.",
           "\\begin{tabular}{@{}l" + "r" * (cols - 1) + "@{}}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{{len(METRICS)}}}{{c}}{{{s.capitalize()}}}" for s in SCORINGS) + "\\\\",
           "".join(f"\\cmidrule(lr){{{2 + i * len(METRICS)}-{1 + (i + 1) * len(METRICS)}}}"
                   for i in range(len(SCORINGS))),
           "Model & " + " & ".join(METRIC_TEX[m] for _ in SCORINGS for m in METRICS) + "\\\\"]
    for j, t in enumerate(TASKS):
        out.append("\\midrule")
        out.append(f"\\multicolumn{{{cols}}}{{@{{}}l}}{{\\textit{{{TASK_LABEL[t]}}}}}\\\\")
        for r in (r for r in T["rows"] if r["task"] == t):
            out.append(r["name"] + " & " + " & ".join(fmt(r["cells"][(s, m)], "$-$") for s in SCORINGS
                                                     for m in METRICS) + "\\\\")
        n = T["medians"][(t, SCORINGS[0], METRICS[0])]["n"]
        out.append(f"\\textit{{Panel median ({n} models)}} & "
                   + " & ".join(f"\\textit{{{fmt(T['medians'][(t, s, m)]['value'], '$-$')}}}" for s in SCORINGS
                                for m in METRICS) + "\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def md(T: dict) -> str:
    head = ["Benchmark", "Model"] + [f"{s.capitalize()}: {METRIC_MD[m]}" for s in SCORINGS for m in METRICS]
    lines = ["# The larger models' arms and gains beside the panel's median",
             "",
             f"Written by `scripts/ladder_arms_table.py` from "
             f"`{LADDER}` (created {T['ladder_created']}) and `{HOTPOT_CONTENT}`; the LaTeX copy is `{TEX}` (a tabular, "
             "no caption). Rebuild: `python scripts/ladder_arms_table.py` (standalone; "
             "`paper/make_numbers.py` does not call it; rerun after `scripts/ladder_analysis.py`).",
             "",
             "Accuracy (%) of the requested, forced and chain-of-thought (CoT) arms and the two gains (pp), strict and "
             "content scoring, outputs cut at the panel's caps (the ladder's primary reading), model means over the 24 "
             "discovery draws (a configuration drawn twice counts twice). HotpotQA content is on the discovery draws as "
             "well, the basis of the reversal counts (`\\ladRevHq`, `\\combRevHq`, `\\aeHqReversedContent`); the "
             "ladder's placements and the paper's Fig. 4 read HotpotQA content on pooled rows instead (`ladder_note.md`, "
             "footnote F6). The last row of each benchmark is the panel's median of the same column "
             "(29 models; 28 on BIG-Bench Hard): the placement records' `panel_median` (basis discovery), and for "
             "HotpotQA content the median of the 29 models' discovery means in "
             "`results/hotpot_content/hotpot_content.json` (`arm_effect.content.per_model`), each gain taken per model "
             "before the median. Rounded half-even at one decimal on the exact value.",
             "",
             "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for t in TASKS:
        for r in (r for r in T["rows"] if r["task"] == t):
            lines.append(f"| {TASK_LABEL[t]} | {r['name']} | "
                         + " | ".join(fmt(r["cells"][(s, m)]) for s in SCORINGS for m in METRICS) + " |")
        n = T["medians"][(t, SCORINGS[0], METRICS[0])]["n"]
        lines.append(f"| {TASK_LABEL[t]} | Panel median ({n} models) | "
                     + " | ".join(fmt(T["medians"][(t, s, m)]["value"]) for s in SCORINGS for m in METRICS) + " |")
    lines += ["", "## Checks", ""] + [f"- {c}" for c in T["checks"]]
    lines += ["", "## Edge cases", "",
              "- Values off their discovery grid (rounded on the float): "
              + ("; ".join(T["edges"]) if T["edges"] else "none") + ".",
              "- Values exactly halfway at one decimal (printed as the even neighbour): "
              + ("; ".join(T["halfway"]) if T["halfway"] else "none") + ".",
              "- Negative values (printed as they are): " + ("; ".join(T["negatives"]) if T["negatives"] else "none")
              + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    T = build()
    (ROOT / TEX).write_text(tex(T), encoding="utf-8", newline="\n")
    text = md(T)
    (ROOT / MD).write_text(text, encoding="utf-8", newline="\n")
    (ROOT / MD_CLEAN).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TEX}, {MD}, {MD_CLEAN}: {len(T['rows'])} model rows, {len(T['edges'])} off-grid values, "
          f"{len(T['halfway'])} halfway values, {len(T['negatives'])} negative values")


if __name__ == "__main__":
    main()
