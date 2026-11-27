r"""Write the example inputs of pilot_report.py from this repository's processed data.

From the repository root (needs the repository's environment and data/processed/, which is not part of
the release):
    python release/pilot_report/make_examples.py

Outputs in release/pilot_report/examples/:
- dove_arc_challenge_scores.csv: DOVE (Dataset 1), five models, arms zero_shot and five_shot, the 4 pilot draws of
  one benchmark. The benchmark is chosen by a fixed rule, never by its result: the first benchmark in the frozen order
  of manifests/sampler_reliability_map_dove_tasks.json (alphabetical by DOVE file name), which is
  ai2_arc.arc_challenge. Its 4 pilot configurations are that manifest's pilot draws, and the scores are those of
  scripts/run_reliability_map.py (dove_benchmarks: the benchmark's common admissible configurations, accuracy in pp).
- gsm8k29_scores.csv: GSM8K-29 (Dataset 2), arms direct_answer (requested direct) and zero_shot_cot (chain of
  thought), strict scoring, the first 4 discovery draws of manifests/sampler_gsm8k.json, loaded as in
  scripts/run_forced_direct.py (collection_inputs("gsm8k"), src/dataset2.cell_table and unit_arrays).
- gsm8k29_tokens.csv: the direct_answer outputs of the same 4 draws, 29 models and 1,319 items, summarized as one row
  per (model, arm, n_output_tokens) with the number of outputs of that length in `count` (153,004 outputs would take
  several MB as one row each). pilot_report.py reads either form.

- gsm8k29_two_baselines_scores.csv (Example 3): GSM8K-29, arms direct_answer (requested baseline), forced_direct (forced
  baseline) and zero_shot_cot, strict scoring, the same 4 draws as gsm8k29_scores.csv, loaded as in
  scripts/robustness_inputs.py (run_forced_direct.three_arm_arrays) and checked against its cached discovery arrays and
  against gsm8k29_scores.csv. Example 3 reads its token counts from gsm8k29_tokens.csv.
- gsm8k29_pilot_items.csv (Example 3, split pilot): item-level strict scores (1 correct, 0 wrong) under the three arms,
  one row per (model, draw, item) with one column per arm. Rule for the items: for each of the 4 draws, the 100 items
  of the first saved item-disjoint pilot of the strict decision design that uses that draw
  (manifests/predictions_decision_strict.npz, gsm8k__disjoint__positions and __items, in saved order), in that pilot's
  item order. The single-baseline gains of these items must reproduce that file's saved own-gain predictions.
- gsm8k29_capability.csv (Example 3): the Epoch Capabilities Index of each model (column capability), as the study
  reads it (scripts/run_transfer_tasks.eci_for: manifests/d2_panel_extension.json, from
  manifests/external/eci_scores_20261030.csv).

The score files carry an extra column config_id (the configuration behind each draw), which pilot_report.py ignores.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = os.path.join(HERE, "examples")
N_PILOT = 4


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_for_pilot_report", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def long_table(B, P, models, configs, baseline, intervention):
    rows = []
    for k, cfg in enumerate(configs):
        for arm, X in ((baseline, B), (intervention, P)):
            for m, name in enumerate(models):
                rows.append({"model": name, "draw": k + 1, "arm": arm, "score": float(X[k, m]),
                             "config_id": cfg})
    return pd.DataFrame(rows)


def dove_example():
    rmap = load_script("run_reliability_map")
    rt = rmap.load_script("run_dataset1_tasks")
    draws = json.load(open(os.path.join(ROOT, "manifests", "sampler_reliability_map_dove_tasks.json"),
                           encoding="utf-8"))
    task = next(iter(draws["benchmarks"]))                       # first benchmark in the frozen order
    d = draws["benchmarks"][task]
    tasks, _ = rmap.dove_benchmarks(rt)
    t = next(x for x in tasks if x["task"] == task)
    pi = list(d["pilot_index"])[:N_PILOT]
    configs = [int(t["common"][i]) for i in pi]
    if configs != [int(c) for c in d["pilot_config_ids"]][:N_PILOT]:
        raise SystemExit(f"{task}: the pilot indices do not point at the manifest's pilot configurations")
    B, P = t["B"][pi], t["P"][pi]
    df = long_table(B, P, rt.PANEL, configs, "zero_shot", "five_shot")
    return task, df, {"n_items": t["n_items"], "n_bank": len(t["common"]), "rule": "first benchmark in the "
                      "frozen order of manifests/sampler_reliability_map_dove_tasks.json"}


def gsm8k_example():
    from src import dataset2 as d2
    from run_forced_direct import collection_inputs
    inp, s, items, _ = collection_inputs("gsm8k")
    panel = list(inp["panel"])
    units = list(s["discovery_config_ids"])[:N_PILOT]
    df = inp["load"](units)
    cells = d2.cell_table(df, panel, units, items)
    B, P = d2.unit_arrays(cells, units, panel, "strict_correct")
    scores = long_table(B, P, panel, units, "direct_answer", "zero_shot_cot")
    item_set = pd.Index(items)
    direct = df[(df["arm"] == "direct_answer") & df["model_key"].isin(panel) & df["item_id"].isin(item_set)]
    parts = [direct[direct["config_id"].astype(str) == str(u)] for u in units]     # one block per draw
    tok = pd.concat(parts, ignore_index=True)
    per_model = tok.groupby("model_key").size()
    if not (per_model == N_PILOT * len(items)).all() or set(per_model.index) != set(panel):
        raise SystemExit(f"token rows per model differ from {N_PILOT} x {len(items)}: {per_model.to_dict()}")
    agg = (tok.groupby(["model_key", "arm", "n_output_tokens"]).size().rename("count").reset_index()
           .rename(columns={"model_key": "model"}))
    agg["n_output_tokens"] = agg["n_output_tokens"].astype(np.int64)
    order = {m: i for i, m in enumerate(panel)}
    agg = agg.sort_values(["model", "n_output_tokens"], key=lambda c: c.map(order) if c.name == "model" else c)
    return scores, agg.reset_index(drop=True), {"n_items": len(items), "units": units, "M": len(panel)}


def multi_table(arrays, models, configs):
    """Long table of several arms: [(arm, (K, M) array)]."""
    rows = []
    for k, cfg in enumerate(configs):
        for arm, X in arrays:
            for m, name in enumerate(models):
                rows.append({"model": name, "draw": k + 1, "arm": arm, "score": float(X[k, m]), "config_id": cfg})
    return pd.DataFrame(rows)


def gsm8k_two_baselines_example(example2):
    """Example 3: three arms on the 4 draws of Example 2, item-level scores of the split pilot, the capability index."""
    import robustness_inputs as ri
    from run_forced_direct import collection_inputs, forced_inputs, three_arm_arrays
    inp, s, items, items_info = collection_inputs("gsm8k")
    panel = list(inp["panel"])
    units = list(s["discovery_config_ids"])[:N_PILOT]
    fd_load, _, _ = forced_inputs("gsm8k", panel, s, items_info, inp["load"], False)
    Breq, Bfd, Pc = three_arm_arrays(inp["load"], fd_load, units, panel, items)
    x = ri.load("gsm8k")
    if x["facts"]["panel"] != panel or x["facts"]["discovery_config_ids"][:N_PILOT] != units:
        raise SystemExit("the panel or draws differ from scripts/robustness_inputs.py")
    for arm, A in (("requested", Breq), ("forced", Bfd), ("cot", Pc)):
        if not np.array_equal(A, x["discovery"][arm][:N_PILOT]):
            raise SystemExit(f"{arm}: the 4 draws differ from robustness_inputs' cached discovery arrays")
    for arm, A in (("direct_answer", Breq), ("zero_shot_cot", Pc)):
        sub = example2[example2["arm"] == arm].pivot(index="draw", columns="model", values="score")
        if not np.array_equal(sub.loc[list(range(1, N_PILOT + 1)), panel].to_numpy(float), A):
            raise SystemExit(f"{arm}: differs from Example 2's scores")
    scores = multi_table([("direct_answer", Breq), ("forced_direct", Bfd), ("zero_shot_cot", Pc)], panel, units)

    rds = load_script("run_decision_strict")
    xi = rds.task_inputs("gsm8k")
    if xi["panel"] != panel or xi["disc"][:N_PILOT] != [str(u) for u in units]:
        raise SystemExit("run_decision_strict's panel or discovery draws differ")
    configs, Sd = rds.dense(xi, xi["disc"])
    z = np.load(os.path.join(ROOT, "manifests", "predictions_decision_strict.npz"))
    pre = "gsm8k__disjoint__"
    pos, its = z[pre + "positions"].astype(int), z[pre + "items"].astype(int)
    rows, chosen = [], []
    for k in range(N_PILOT):
        p = int(np.flatnonzero(pos == k)[0])                  # the first saved pilot on draw k
        r = configs.index(xi["disc"][k])
        chosen.append(p)
        for name, gain in (("requested", Sd["requested"]), ("forced", Sd["forced"])):
            g = 100.0 * (Sd["cot"][r][:, its[p]].astype(float) - gain[r][:, its[p]].astype(float)).mean(1)
            if np.max(np.abs(g - z[pre + f"{name}__own_gain__pred"][p])) > 1e-9:
                raise SystemExit(f"draw {k + 1}: {name}-only gains differ from the saved pilot {p}")
        for m, name in enumerate(panel):
            for q in its[p]:
                rows.append({"model": name, "draw": k + 1, "item": xi["items"][q],
                             "direct_answer": int(Sd["requested"][r, m, q]),
                             "forced_direct": int(Sd["forced"][r, m, q]),
                             "zero_shot_cot": int(Sd["cot"][r, m, q])})
    item_df = pd.DataFrame(rows)
    cap = pd.DataFrame({"model": panel, "capability": [float(v) for v in x["facts"]["eci"]]})
    return scores, item_df, cap, {"pilots": chosen, "n_items": int(its.shape[1])}


EPOCH_CREDIT = "# " + "Epoch Capabilities Index, Epoch AI, CC BY 4.0, retrieved 30 October 2026, https://epoch.ai/eci"


def write(df, name, comment=None):
    """Write one example CSV; `comment` becomes a first line starting with "# " (the pilot tool skips it)."""
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if comment:
            fh.write(comment + "\n")
        df.to_csv(fh, index=False, lineterminator="\n")
    size = os.path.getsize(path)
    print(f"wrote {os.path.relpath(path, ROOT)}: {len(df):,} rows, {size / 1024:.1f} KB")
    if size > 1_000_000:
        raise SystemExit(f"{name} is {size:,} bytes, above the 1 MB limit for the examples")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    task, dove, info = dove_example()
    print(f"DOVE: {task} ({info['rule']}), {info['n_items']} items, bank of {info['n_bank']} configurations")
    write(dove, f"dove_{task.split('.')[-1]}_scores.csv")
    scores, tokens, info = gsm8k_example()
    print(f"GSM8K-29: {info['M']} models, {info['n_items']} items, draws {info['units']}")
    write(scores, "gsm8k29_scores.csv")
    write(tokens, "gsm8k29_tokens.csv")
    scores3, item_df, cap, info = gsm8k_two_baselines_example(scores)
    print(f"GSM8K-29, two baselines: split-pilot items from the saved strict pilots {info['pilots']} "
          f"({info['n_items']} items each)")
    write(scores3, "gsm8k29_two_baselines_scores.csv")
    write(item_df, "gsm8k29_pilot_items.csv")
    write(cap, "gsm8k29_capability.csv", comment=EPOCH_CREDIT)


if __name__ == "__main__":
    main()
