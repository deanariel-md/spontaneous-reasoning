r"""Save the arm-wording mechanism forecast (src/arm_wording_mechanism.py) before any arm-wording
output is scored.

From the repository root:
    .\.venv\Scripts\python.exe scripts\build_arm_wording_mechanism.py

Reads, and nothing else:
  - the arm-wording screen's output-token counts, data/d2_out/arm_wording/screen/<key>/shard-*.parquet, columns model_key,
    arm, config_id and n_output_tokens only (no output text, no correctness; the screen has no scores);
  - the collection's design record data/requests/arm_wording/collection/build_record.json (the panel's order and the
    chosen clauses);
  - the main collection's scored discovery draws as summarised in results/baseline_mixture/baseline_mixture.json, key
    decomposition, set discovery, strict scores (per model: chain of thought, forced and requested accuracy and the
    short-stratum accuracies; w under the current clause; the task's median within-model recovery on long answers).
Writes manifests/predictions_arm_wording_mechanism.json and its sha256 in manifests/predictions_arm_wording_mechanism.
sha256 (write-once: an identical forecast is kept, a different one refused). scripts/score_arm_wording.py refuses to
score either stage until both exist and match; scripts/run_arm_wording.py's full run evaluates every statement.

--base DIR reads the design record and screen under DIR and writes under DIR (dry runs); --mixture PATH reads another
baseline_mixture.json.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import arm_wording as aw, arm_wording_mechanism as awm  # noqa: E402

MIXTURE = ROOT / "results" / "baseline_mixture" / "baseline_mixture.json"


def screen_w(screen_dir, models, clauses, long_tokens=awm.LONG_TOKENS) -> tuple[dict, dict]:
    """{task: (clauses, models) share of requested-direct screen outputs longer than long_tokens} and the counts."""
    parts = []
    for m in models:
        files = sorted(glob.glob(os.path.join(str(screen_dir), m, "shard-*.parquet")))
        if not files:
            raise SystemExit(f"no screen output for {m}")
        for f in files:
            parts.append(pd.read_parquet(f, columns=["model_key", "arm", "config_id", "n_output_tokens"]))
    s = pd.concat(parts, ignore_index=True)
    s = s[s["arm"] == "direct_answer"]
    p = s["config_id"].astype(str).map(aw.parse_config_id)
    s = s.assign(task=[x[0] for x in p], wid=[x[2] for x in p])
    if s["n_output_tokens"].isna().any():
        raise SystemExit("a screen output has no token count")
    W, N = {}, {}
    for t in aw.TASKS:
        x = s[s["task"] == t]
        g = (x["n_output_tokens"] > long_tokens).groupby([x["wid"], x["model_key"]])
        sh, n = g.mean(), g.size()
        W[t] = np.array([[float(sh.get((d, m), np.nan)) for m in models] for d in clauses])
        N[t] = np.array([[int(n.get((d, m), 0)) for m in models] for d in clauses])
        if np.isnan(W[t]).any():
            raise SystemExit(f"{t}: a model has no screen output under a chosen clause")
    return W, N


def mixture_inputs(path, models) -> tuple[dict, dict]:
    """{task: g_fd, g_short, rho, w_collection_current} in the panel order of `models`, from the decomposition's
    discovery set (strict)."""
    R = json.load(open(path, encoding="utf-8"))
    D = R.get("decomposition")
    if not D:
        raise SystemExit(f"{path} has no decomposition; run scripts/run_baseline_mixture.py --decomposition first")
    out, info = {}, {}
    for t in aw.TASKS:
        T = D[t]
        pm = T["sets"]["discovery"]["strict"]["per_model"]
        panel = list(T["models"])
        if sorted(panel) != sorted(models):
            raise SystemExit(f"{t}: the main collection's panel differs from the arm-wording run's")
        ix = [panel.index(m) for m in models]
        C, F = np.array(pm["C"], float)[ix], np.array(pm["F"], float)[ix]
        Cs, Fs = np.array(pm["C_short"], float)[ix], np.array(pm["F_short"], float)[ix]
        # rho: the median recovery among models with at least a tenth of their requested answers long (the ratio is
        # noise where a model writes few long answers; the all-model median is kept beside it)
        rec_w = T["sets"]["discovery"]["strict"]["recovery_long_w_at_least"]
        rec_all = T["sets"]["discovery"]["strict"]["recovery_long"]
        out[t] = {"g_fd": C - F, "g_short": Cs - Fs, "rho": float(rec_w["median"]),
                  "w_collection_current": np.array(pm["w"], float)[ix]}
        info[t] = {"rho_from": "median within-model recovery on long answers (R_L - F_L) / (C_L - F_L), discovery "
                               f"draws, strict, models with w >= {rec_w['w_min']} and a positive denominator",
                   "recovery_models": rec_w["n"], "all_model_median": rec_all["median"], "all_models": rec_all["n"]}
    return out, info


def build(base=None, mixture=None) -> dict:
    paths = aw.Paths(base)
    rec = json.loads((paths.requests / "collection" / "build_record.json").read_text(encoding="utf-8"))
    models = list(rec["models"])
    clauses = [w["id"] for w in rec["chosen"]["dir"]]
    if clauses[0] != "dir0":
        raise SystemExit("the first chosen clause is not the current one")
    W, N = screen_w(paths.outputs / "screen", models, clauses)
    mix_path = Path(mixture) if mixture else MIXTURE
    inp, info = mixture_inputs(mix_path, models)
    for t in aw.TASKS:
        inp[t]["w"] = W[t]
    scr = Path(paths.outputs / "screen").resolve()
    source = {"screen": str(scr.relative_to(ROOT)).replace("\\", "/") if str(scr).startswith(str(ROOT)) else str(scr),
              "screen_requested_outputs_per_cell": sorted({int(v) for t in aw.TASKS for v in N[t].ravel()}),
              "design_record_sha256": aw.sha256_file(paths.requests / "collection" / "build_record.json"),
              "mixture": str(mix_path.relative_to(ROOT)).replace("\\", "/") if str(mix_path).startswith(str(ROOT))
              else str(mix_path), "mixture_sha256": aw.sha256_file(mix_path), "rho": info,
              "clause_texts": {w["id"]: w["text"] for w in rec["chosen"]["dir"]}}
    r = awm.forecast_record(models, clauses, inp, source)
    return awm.write_forecast(r, paths.mechanism, paths.mechanism_sha)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base")
    ap.add_argument("--mixture")
    a = ap.parse_args(argv)
    r = build(a.base, a.mixture)
    print(f"predictions_arm_wording_mechanism.json: sha256 {r['sha256']} created {r['created']}")
    for t, x in r["per_task"].items():
        p = x["predicted"]
        print(f"{t}: rho {x['rho']:.3f}; clause-only reversal {p['clause_reversal_range'][0]:.3f}-"
              f"{p['clause_reversal_range'][1]:.3f}; arm switch by clause "
              + ", ".join(f"{a_:.3f}-{b_:.3f}" for a_, b_ in p["arm_switch_by_clause_range"])
              + f"; w summary {x['w_summary']}")
    return r


if __name__ == "__main__":
    main()
