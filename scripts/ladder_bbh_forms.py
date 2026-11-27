r"""Size ladder: the answer forms behind the strict-content gap of the locally run rungs' BIG-Bench Hard forced arms.

Read-only. For each locally run ladder model whose BIG-Bench Hard rows are scored (data/processed/ladder/bbh/<key>.parquet,
the cut reading; the forced arms are never cut, so both readings hold the same forced rows), it reads the model's
outputs (data/d2_out/ladder2/<run key>/bbh through d2_io.inspect_work_dir, which puts the prefill \boxed{ in front, as
scripts/score_ladder.py scored them), keeps the forced and five-shot forced rows that the content reading scores correct
and the strict reading wrong, and assigns each the form of the box the strict rule reads, with the classes of
scripts/check_nebius_forced.py (classify_gap; the Mistral check's classes plus the answer tag, angle brackets, full-width
parentheses and LaTeX inline delimiters). No scorer is changed and nothing is re-scored.

Writes results/ladder/analysis/bbh_forced_forms.json, .md and _clean.md. scripts/ladder_analysis.py cites the json in
its footnote F1 when it exists. The json keeps the counts (rows, strict and content rows, gap rows, rows per form) with
each pp; the md prints every pp from those counts with scripts/ladder_analysis.fnum (half-even on the exact ratio).

PowerShell, repository root:
    $env:PYTHONIOENCODING = "utf-8"
    .\.venv\Scripts\python.exe scripts\ladder_bbh_forms.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import bbh, d2_io  # noqa: E402
import check_nebius_forced as cnf  # noqa: E402
import nebius_panel_report as npr  # noqa: E402
from fractions import Fraction  # noqa: E402
from ladder_analysis import fnum  # noqa: E402

KEYS = ("ministral-3-8b-instruct-2512", "ministral-3-14b-instruct-2512", "llama-3.1-70b-instruct-fp8",
        "qwen3.5-122b-a10b-fp8")
ARMS = ("forced_direct", "fewshot_forced")
OUT_ROOT = ROOT / "data" / "d2_out" / "ladder2"
SCORED = ROOT / "data" / "processed" / "ladder" / "bbh" / "{key}.parquet"
OUT = ROOT / "results" / "ladder" / "analysis" / "bbh_forced_forms"


def run_key(key: str) -> str:
    p = OUT_ROOT / "chosen" / key
    return p.read_text(encoding="utf-8").strip() if p.exists() else key


def forms_of(key: str, gold: dict) -> dict | None:
    sp = Path(str(SCORED).format(key=key))
    d = OUT_ROOT / run_key(key) / "bbh"
    if not sp.exists() or not d.is_dir():
        return None
    out, _ = d2_io.inspect_work_dir(d, string_item_ids=True)
    out = out[out["arm"].isin(ARMS)][["request_id", "output_text"]]
    sc = pd.read_parquet(sp, columns=["request_id", "arm", "item_id", "strict_correct", "content_correct"])
    sc = sc[sc["arm"].isin(ARMS)]
    m = sc.merge(out, on="request_id", how="left", validate="one_to_one")
    res = {"run_key": run_key(key), "rows_without_text": int(m["output_text"].isna().sum()), "arms": {}}
    for arm, g in m.groupby("arm"):
        gap = g[(g["content_correct"] == 1) & (g["strict_correct"] == 0)]
        forms = []
        for text, item in zip(gap["output_text"], gap["item_id"].astype(str)):
            b = bbh.boxed_contents(text or "")
            ans, atype, opts, task_name = gold[item][:4]
            forms.append(cnf.classify_gap(b[0] if b else None, atype, opts or {}, task_name))
        vc = pd.Series(forms, dtype=object).value_counts()
        n = len(g)
        res["arms"][arm] = {
            "rows": int(n), "strict_rows": int(g["strict_correct"].sum()), "content_rows": int(g["content_correct"].sum()),
            "strict_pp": float(100 * g["strict_correct"].mean()),
            "content_pp": float(100 * g["content_correct"].mean()), "gap_rows": int(len(gap)),
            "gap_pp": float(100 * len(gap) / n),
            "content_minus_strict_pp": float(100 * (g["content_correct"].mean() - g["strict_correct"].mean())),
            "forms": [{"form": f, "rows": int(c), "pp": float(100 * c / n)} for f, c in vc.items()]}
    return res


def write_md(R: dict, path: Path) -> None:
    L = [f"# Size ladder: BIG-Bench Hard forced arms, forms of the locally run rungs' strict-content gap ({R['created']})", "",
         "Read-only (`scripts/ladder_bbh_forms.py`). Rows the content reading scores correct and the strict reading "
         "wrong, by the form of the box the strict rule reads; classes of `scripts/check_nebius_forced.py`. pp are of "
         "all rows of the arm. No scorer was changed.", ""]
    for k, v in R["models"].items():
        L += [f"## {k} (run key `{v['run_key']}`)", ""]
        for arm, a in v["arms"].items():
            n = a["rows"]
            L += [f"{arm}: {n:,} rows, strict {fnum(Fraction(100 * a['strict_rows'], n))}, content "
                  f"{fnum(Fraction(100 * a['content_rows'], n))}; content-correct and strict-wrong {a['gap_rows']:,} "
                  f"rows ({fnum(Fraction(100 * a['gap_rows'], n))} pp; content − strict "
                  f"{fnum(Fraction(100 * (a['content_rows'] - a['strict_rows']), n))} pp).", "",
                  "| Form | Rows | pp |", "|---|---:|---:|"]
            L += [f"| {f['form']} | {f['rows']:,} | {fnum(Fraction(100 * f['rows'], n), 2)} |" for f in a["forms"]]
            L.append("")
        if v["rows_without_text"]:
            L += [f"Rows with no output text: {v['rows_without_text']}.", ""]
    if R["absent"]:
        L += ["Not yet here (no scored BIG-Bench Hard rows or no outputs on this PC): " + ", ".join(R["absent"]) + ".",
              ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.time()
    gold = npr.content_gold("bbh")
    R = {"created": time.strftime("%Y-%m-%d %H:%M"), "models": {}, "absent": []}
    for k in KEYS:
        r = forms_of(k, gold)
        if r is None:
            R["absent"].append(k)
        else:
            R["models"][k] = r
    R["runtime_s"] = time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    md = OUT.with_suffix(".md")
    write_md(R, md)
    (OUT.parent / (OUT.name + "_clean.md")).write_text(npr.npl.npt.strip_links(md.read_text(encoding="utf-8")),
                                                       encoding="utf-8")
    print(f"wrote {md}; models {list(R['models'])}; absent {R['absent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
