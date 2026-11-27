"""HotpotQA mechanism forecast on the validation configurations that no discovery draw uses.

The saved forecast (manifests/predictions_hotpot_mechanism.json) is scored by run_hotpot.py's full run on all 24
validation draws, as written. Nine of those draws reuse a configuration that a discovery draw also used. This script
reports the same three statements on the other draws beside the written result, without changing it:

- ``fresh_draws``: the validation draws whose configuration appears in no discovery draw, repeats kept as the sampler
  drew them (run_hotpot.fresh_rows);
- ``fresh_distinct``: each of those configurations once.

It reads the scored table only after the full run has written results/hotpot/mechanism_forecast_evaluation.json, and
refuses otherwise, so the order of the saved forecast is kept. It changes no saved file. Output:
results/hotpot/mechanism_forecast_fresh.json, mechanism_forecast_fresh.md and its clean copy.

    python scripts/run_hotpot_mech_fresh.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import baseline_mixture as bm  # noqa: E402

MAIN_EVALUATION = "mechanism_forecast_evaluation.json"
OUT_JSON = "mechanism_forecast_fresh.json"
OUT_MD = "mechanism_forecast_fresh.md"


def _gains(V):
    g_req = (np.asarray(V["cot_req"][1], float) - np.asarray(V["cot_req"][0], float)).mean(0)
    g_fd = (np.asarray(V["cot_forced"][1], float) - np.asarray(V["cot_forced"][0], float)).mean(0)
    return g_req, g_fd


def run(processed, sampler_path, pred_dir, out, panel=None, items=None) -> dict:
    import run_hotpot as r
    main = os.path.join(out, MAIN_EVALUATION)
    if not os.path.exists(main):
        raise SystemExit(f"{main} is absent: run run_hotpot.py's full run (hq_after.sh) first; this check reads the "
                         "validation draws only after the saved forecast has been scored as written")
    written = json.load(open(main, encoding="utf-8"))
    rec = bm.load_forecast(os.path.join(pred_dir, "predictions_hotpot_mechanism.json"))
    if rec["sha256"] != written.get("forecast_sha256"):
        raise SystemExit("the saved forecast's sha256 differs from the one the full run evaluated")
    sampler = json.load(open(sampler_path, encoding="utf-8"))
    if panel is None:
        panel, _, _ = r.panel_info(None)
    if items is None:
        items = r.item_ids(processed)
    disc, val = list(map(str, sampler["discovery_config_ids"])), list(map(str, sampler["validation_config_ids"]))
    fresh = r.fresh_rows(disc, val)
    fresh_units = [val[i] for i in fresh]
    distinct = sorted(set(fresh_units))
    path = os.path.join(processed, "scored.parquet")
    R = {"analysis": "HotpotQA mechanism forecast on validation configurations absent from discovery",
         "created": time.strftime("%Y-%m-%d %H:%M"), "forecast_sha256": rec["sha256"],
         "validation_draws": len(val), "reusing_discovery": len(val) - len(fresh), "fresh_draws": len(fresh),
         "fresh_distinct": len(distinct), "fresh_configurations": distinct,
         "written_result": {"observed_share": written["statements"]["share"]["observed"],
                            "holds": {k: written["statements"][k].get("holds") for k in ("share", "pairs", "gain")}},
         "sets": {}}
    for name, units in (("fresh_draws", fresh_units), ("fresh_distinct", distinct)):
        if not units:
            R["sets"][name] = {"note": "no validation draw uses a configuration absent from discovery"}
            continue
        V = {c: r.arrays(r.loader(path, c), units, panel, items) for c in ("cot_req", "cot_forced")}
        g_req, g_fd = _gains(V)
        E = bm.evaluate_forecast(rec, panel, g_req, g_fd)
        E["n_units"] = len(units)
        R["sets"][name] = E
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, OUT_JSON), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    md = summary(R)
    for fn in (OUT_MD, OUT_MD.replace(".md", "_clean.md")):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as fh:
            fh.write(md)
    return R


def summary(R) -> str:
    def f3(v):
        return "--" if v is None else f"{v:.3f}"
    L = ["# HotpotQA mechanism forecast on configurations new to validation", "",
         f"Created {R['created']}. The saved forecast (sha256 `{R['forecast_sha256'][:16]}…`) was scored as written on "
         f"all {R['validation_draws']} validation draws by run_hotpot.py's full run. {R['reusing_discovery']} of those "
         f"draws reuse a configuration that a discovery draw used. The rows below repeat the three statements on the "
         f"other {R['fresh_draws']} draws ({R['fresh_distinct']} distinct configurations). The written result stands "
         "as the forecast's test; these rows are a sensitivity beside it.", "",
         "| Set | Units | Pairs | Predicted reversed | Observed reversed | Share predicted | Share observed | Share holds "
         "| Pairs holds | Gain holds |", "|---|---|---|---|---|---|---|---|---|---|"]
    w = R["written_result"]
    L.append(f"| All validation draws (written) | {R['validation_draws']} | | | | | {f3(w['observed_share'])} | "
             f"{w['holds']['share']} | {w['holds']['pairs']} | {w['holds']['gain']} |")
    for name, E in R["sets"].items():
        if "statements" not in E:
            L.append(f"| {name} | 0 | {E['note']} | | | | | | | |")
            continue
        st = E["statements"]
        L.append(f"| {name.replace('_', ' ')} | {E['n_units']} | {E['pairs']} | {E['predicted_reversed']} | "
                 f"{E['observed_reversed']} | {f3(st['share']['predicted'])} | {f3(st['share']['observed'])} | "
                 f"{st['share']['holds']} | {st['pairs']['holds']} | {st['gain']['holds']} |")
    return "\n".join(L) + "\n"


def main():
    import run_hotpot as r
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--processed", default=r.PROCESSED)
    ap.add_argument("--sampler", default=r.SAMPLER)
    ap.add_argument("--pred-dir", default=r.MANIFESTS)
    ap.add_argument("--out", default=r.OUT)
    a = ap.parse_args()
    R = run(a.processed, a.sampler, a.pred_dir, a.out)
    print(summary(R))


if __name__ == "__main__":
    main()
