"""Analysis of the reproduction of Cheng et al. (2025) under the fixed protocol.

Reads the scored tables of score_reproduction.py and writes results/reproduction/summary.json, summary.md and
summary_clean.md, with the accuracy and benefit tables as CSV. Part A (numerical) and Part B (transport) are reported
separately; models or configurations without outputs are listed as not run.

    .\\.venv\\Scripts\\python.exe scripts\\run_reproduction.py --scored data\\processed\\reproduction\\numerical_scored.parquet data\\processed\\reproduction\\transport_scored.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import reproduction as rp  # noqa: E402


def _f(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}" if d and isinstance(x, float) and x < 0 else f"{x:.{d}f}"


def _claim(c: dict) -> str:
    if c["holds"]:
        return "holds"
    return "fails (" + ", ".join(c["strong_models_with_benefit_above_zero"]) + ")"


def render(s: dict) -> str:
    A, B = s["numerical"], s["transport"]
    L = ["# Reproduction of Cheng et al. (2025): results", "",
         f"Scorer {s['scorer']}; intervals at {s['interval_level']:.0%}. "
         "Accuracies and benefits in percentage points; benefit is eight-shot minus zero-shot accuracy.", "",
         "## Part A: numerical reproduction", "",
         f"Models run: {len(A['models_run'])} of {len(s['models_planned'])}"
         + (f"; not run: {', '.join(A['models_not_run'])}." if A["models_not_run"] else "."), "",
         "| Model | Arm | Ours | 95% interval | Published | Difference | Published inside | NULL | Truncated |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in A["accuracy"]:
        L.append(f"| {r['model_key']} | {r['arm']} | {r['accuracy']:.2f} | {r['lo']:.2f}–{r['hi']:.2f} | {r['published']:.2f} | "
                 f"{r['difference']:+.2f} | {'yes' if r['published_inside_interval'] else 'no'} | {r['null_preds']} | {r['truncated']} |")
    L += ["", "| Model | Benefit | 95% paired interval | Published benefit | Signs agree |", "|---|---|---|---|---|"]
    for r in A["benefit"]:
        agree = "tie" if r["sign_agrees"] is None else ("yes" if r["sign_agrees"] else "no")
        L.append(f"| {r['model_key']} | {r['benefit']:+.2f} | {r['benefit_lo']:+.2f} to {r['benefit_hi']:+.2f} | "
                 f"{r['published_benefit']:+.2f} | {agree} |")
    sa = A["sign_agreement"]
    L += ["", f"Published claim (no strong model's benefit interval above zero): BF16 models {_claim(A['claim_bf16'])}; "
          f"with the FP8 models {_claim(A['claim_with_fp8'])}.",
          f"Benefit signs agree with the published signs for {sa['agree']} models and disagree for {sa['disagree']} "
          f"({sa['tie_or_undefined']} ties or undefined).",
          f"Capability slope of benefit on zero-shot accuracy: ours {_f(A['capability_slope']['slope'], 3)} over "
          f"{A['capability_slope']['n_models']} models; published values {_f(A['published_capability_slope']['slope'], 3)}.", "",
          "## Part B: prompt transport", ""]
    if not B["configurations"]:
        L.append("No transport outputs.")
    else:
        L += ["| Configuration | Models | Claim (BF16) | Claim (with FP8) | Signs agree | Signs disagree | Capability slope |",
              "|---|---|---|---|---|---|---|"]
        for cid, v in B["per_configuration"].items():
            L.append(f"| {cid} | {v['models']} | {_claim(v['claim_bf16'])} | {_claim(v['claim_with_fp8'])} | "
                     f"{v['sign_agree']} | {v['sign_disagree']} | {_f(v['capability_slope']['slope'], 3)} |")
        ch = B["claim_holds_in"]
        eb = B["equal_budget_slope"]
        L += ["", f"The claim holds in {ch['bf16']} of {ch['of']} configurations for the BF16 models and in "
              f"{ch['with_fp8']} with the FP8 models added. Equal-budget capability slope (benefits averaged over the "
              f"configurations, {eb['n_models']} models with every configuration): {_f(eb['slope'], 3)}."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--out", default=str(rp.ROOT / "results" / "reproduction"))
    a = ap.parse_args(argv)
    cfg = rp.load_config()
    scored = pd.concat([pd.read_parquet(p) for p in a.scored], ignore_index=True)
    if set(scored["scorer"]) != {rp.SCORER_VERSION}:
        raise SystemExit(f"scored with {sorted(set(scored['scorer']))}, expected {rp.SCORER_VERSION}")
    s = rp.analyze(scored, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(s["accuracy_table"]).to_csv(out / "accuracy.csv", index=False)
    pd.DataFrame(s["numerical"]["benefit"] + s["transport"]["benefit"]).to_csv(out / "benefit.csv", index=False)
    md = render(s)
    (out / "summary.md").write_text(md, encoding="utf-8")
    (out / "summary_clean.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
