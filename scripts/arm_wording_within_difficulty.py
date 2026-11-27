r"""Rewording the direct request: does it move w and requested accuracy together while forced accuracy stays
put? Existing outputs only.

From the repository root:
    python scripts/arm_wording_within_difficulty.py

Reads results/arm_wording/tables/fig_w_vs_requested.csv (written by scripts/run_arm_wording.py --fig-w-table):
one row per model, task (gsm8k, math) and direct clause (dir0, the current wording, and dir1-dir3, the drawn
paraphrases), with w (share of the requested-direct outputs under that clause longer than 16 generated tokens, over
the six configurations and every item) and the strict requested and forced accuracies (pp, mean over the six
configurations). A model qualifies on a task when its w spans at least 0.2 across the four clauses (max - min >= 0.2).

For each qualifying model and task: the slopes of requested and forced accuracy on w across the four clauses, and
their spans; per task, the counts (read by paper/macros_within_difficulty.py).

Outputs: results/arm_wording/within_difficulty.csv (every model, with a qualifies column) and within_difficulty.json (counts per
task).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SRC = os.path.join(ROOT, "results", "arm_wording", "tables", "fig_w_vs_requested.csv")
OUT = os.path.join(ROOT, "results", "arm_wording")
NAME = "within_difficulty"
TASKS = ("gsm8k", "math")
SPAN_MIN = 0.2
WIDTH = 3.1        # width_in and height_in of the record, kept as it has always carried them
HEIGHT = 2.3


def families(root=ROOT) -> dict:
    with open(os.path.join(root, "results", "ladder", "analysis", "ladder.json"), encoding="utf-8") as fh:
        return json.load(fh)["panel_families"]


def slope(x, y):
    """Least-squares slope of y on x (pp per unit w); None when x does not vary."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def table(src=SRC, fam=None, span_min=SPAN_MIN) -> pd.DataFrame:
    """Every model and task with its four clauses, the w span, whether it qualifies, and the within-model slopes of
    requested and forced accuracy on w across the clauses."""
    d = pd.read_csv(src)
    fam = fam if fam is not None else families()
    rows = []
    for (t, m), g in d.groupby(["task", "model_key"], sort=True):
        if sorted(g["clause"]) != ["dir0", "dir1", "dir2", "dir3"]:
            raise SystemExit(f"{t}/{m}: not exactly the four clauses dir0-dir3")
        g = g.sort_values("clause")
        span = float(g["w"].max() - g["w"].min())
        rs = slope(g["w"], g["requested_accuracy_pp"])
        fs = slope(g["w"], g["forced_accuracy_pp"])
        for _, r in g.iterrows():
            rows.append({"task": t, "model_key": m, "family": fam.get(m), "clause": r["clause"],
                         "clause_text": r["clause_text"], "w": r["w"], "n_requested": r["n_requested"],
                         "n_requested_long": r["n_requested_long"],
                         "requested_accuracy_pp": r["requested_accuracy_pp"],
                         "forced_accuracy_pp": r["forced_accuracy_pp"], "w_span": span,
                         "qualifies": bool(span >= span_min),
                         "requested_accuracy_span_pp": float(g["requested_accuracy_pp"].max()
                                                             - g["requested_accuracy_pp"].min()),
                         "forced_accuracy_span_pp": float(g["forced_accuracy_pp"].max() - g["forced_accuracy_pp"].min()),
                         "requested_slope_pp_per_w": rs, "forced_slope_pp_per_w": fs})
    out = pd.DataFrame(rows)
    if out["family"].isna().any():
        raise SystemExit(f"no family for {sorted(out.loc[out['family'].isna(), 'model_key'].unique())}")
    return out


def counts(tab: pd.DataFrame) -> dict:
    per = tab.drop_duplicates(["task", "model_key"])
    out = {}
    for t in TASKS:
        p = per[per["task"] == t]
        q = p[p["qualifies"]]
        out[t] = {"models": int(len(p)), "qualify": int(len(q)),
                  "qualifying_models": sorted(q["model_key"]),
                  "near_threshold": {m: round(s, 4) for m, s in zip(p["model_key"], p["w_span"])
                                     if abs(s - SPAN_MIN) < 0.03},
                  "requested_slope_positive": int((q["requested_slope_pp_per_w"] > 0).sum()),
                  "requested_slope_median_pp_per_w": float(q["requested_slope_pp_per_w"].median()),
                  "forced_slope_median_pp_per_w": float(q["forced_slope_pp_per_w"].median()),
                  "requested_accuracy_span_median_pp": float(q["requested_accuracy_span_pp"].median()),
                  "forced_accuracy_span_median_pp": float(q["forced_accuracy_span_pp"].median())}
    return out




def main():
    tab = table()
    c = counts(tab)
    tab.to_csv(os.path.join(OUT, NAME + ".csv"), index=False)
    rec = {"request": "Fig. 2d, rewording panel", "source": "results/arm_wording/tables/"
           "fig_w_vs_requested.csv", "span_min": SPAN_MIN, "width_in": WIDTH, "height_in": HEIGHT,
           "scoring": "strict", "counts": c}
    with open(os.path.join(OUT, NAME + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(c, indent=1))
    print("wrote", *[os.path.relpath(os.path.join(OUT, NAME + e), ROOT) for e in (".csv", ".json")])


if __name__ == "__main__":
    main()
