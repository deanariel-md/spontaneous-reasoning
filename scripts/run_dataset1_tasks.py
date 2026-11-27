r"""Secondary Dataset 1 analyses over every eligible English DOVE benchmark.

Per task: exact conditional Step 1-2 summaries on its own common admissible bank (descriptive,
secondary). Across tasks: the observed-task decomposition and held-out-family transfer, on
the common panel only. Benchmarks lacking any panel model are listed and excluded from the
pooled analyses (task-specific panels cannot be pooled as if identical).

PowerShell:
    .\.venv\Scripts\python.exe scripts\run_dataset1_tasks.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import dataset1 as d1, tasks  # noqa: E402
from src import dove_io as io  # noqa: E402

ROOT = io.ROOT
MAN = os.path.join(ROOT, "manifests")
RES = os.path.join(ROOT, "results", "dataset1")
ARMS = ("0_shot", "5_shot")
PANEL = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Meta-Llama-3-8B-Instruct",
         "Mistral-7B-Instruct-v0.3", "OLMoE-1B-7B-0924-Instruct"]
# The structural and scorer checks of scripts/build_dove_tables.py. A benchmark with a file that fails one is
# excluded, as a failing file would have been for the MMLU block.
CHECKS = ["unknown_config_keys", "duplicate_item_config_rows", "score_missing", "score_not_binary",
          "score_vs_exact_match_disagree", "gold_not_in_prompt", "items_with_varying_gold_text"]
FILE_CHECKS = os.path.join(ROOT, "results", "audit", "file_checks.csv")


def family_of(task: str) -> str:
    return task.split(".")[0]


def failed_check_tasks(path: str = FILE_CHECKS) -> dict[str, list[str]]:
    """Benchmarks with at least one file that fails a check, with the checks that failed."""
    chk = pd.read_csv(path)
    bad = chk[(chk[CHECKS] > 0).any(axis=1)]
    return {name.removesuffix(".parquet"): [c for c in CHECKS if (g[c] > 0).any()]
            for name, g in bad.groupby("benchmark_file")}


def scores_pp(full: pd.DataFrame, arm: str, common: list, n_items: int) -> np.ndarray:
    """Accuracy in pp, configurations by panel models. counts.parquet stores n_correct as int8, and 100 times an
    int8 array stays int8 under NumPy 2 and wraps, so the counts are widened before any arithmetic."""
    piv = (full[(full["arm"] == arm) & full["config_id"].isin(common)]
           .pivot(index="config_id", columns="model", values="n_correct").loc[common, PANEL])
    return 100.0 * piv.to_numpy(dtype=np.int64) / n_items


def main():
    bank = pd.read_csv(os.path.join(MAN, "dove_config_bank.csv"), keep_default_na=False)
    admissible = set(bank.loc[bank["admissible"].astype(str) == "True", "config_id"])
    counts = pd.read_parquet(os.path.join(io.PROCESSED, "counts.parquet"))
    out = {"tasks": [], "excluded_tasks": []}
    banks, fams, names = [], [], []
    failed = failed_check_tasks()
    for task, x in sorted(counts.groupby("file")):
        if task in failed:
            out["excluded_tasks"].append({"task": task, "reason": f"a file fails the audit checks {failed[task]} "
                                                                  "(results/audit/file_checks.csv)"})
            continue
        present = sorted(x["model"].unique())
        if set(present) != set(PANEL):
            out["excluded_tasks"].append({"task": task, "reason": f"panel models present: {present}"})
            continue
        n_items = x.groupby(["model", "arm"])["n_items"].max()
        if n_items.nunique() != 1:
            out["excluded_tasks"].append({"task": task, "reason": "item count differs across model-arm cells"})
            continue
        full = x[x["n_items"] == n_items.iloc[0]]
        per_cfg = full.groupby("config_id").size()
        common = sorted(c for c, n in per_cfg.items() if n == len(PANEL) * len(ARMS) and c in admissible)
        if len(common) < 2:
            out["excluded_tasks"].append({"task": task, "reason": f"{len(common)} common admissible configurations"})
            continue
        B, P = (scores_pp(full, a, common, int(n_items.iloc[0])) for a in ARMS)
        tg = d1.exact_targets(B, P)
        ab, aa = d1.exact_agreement(P - B), d1.exact_agreement(B)
        out["tasks"].append({"task": task, "family": family_of(task), "n_items": int(n_items.iloc[0]),
                             "n_configurations": len(common), "C_benefit": ab["C"], "C_accuracy": aa["C"],
                             "gap": aa["C"] - ab["C"], "tie_benefit": ab["tie"],
                             "mean_gain": tg["mean_gain"], "V_delta": tg["V_delta"], "gain_sd": tg["gain_sd"],
                             "H": tg["H"], "R_benefit": tg["R_benefit"], "R_accuracy": tg["R_accuracy"],
                             "delta": tg["delta"], "theta": tg["theta"]})
        banks.append((B, P))
        fams.append(family_of(task))
        names.append(task)
    w = tasks.family_weights(fams)
    out["families"] = sorted(set(fams))
    out["task_weights"] = dict(zip(names, w.tolist()))
    out["decomposition"] = tasks.observed_task_decomposition(banks, w)
    out["transfer"] = tasks.transfer_holdout(banks, fams)
    out["transfer_note"] = ("omitted: fewer than four eligible source families" if not out["transfer"]
                            else "held-out source families; exact conditional risks in pp^2")
    # Family-level summaries (macro-average within family) for the supplement
    fam_rows = []
    for f in out["families"]:
        rows = [t for t in out["tasks"] if t["family"] == f]
        fam_rows.append({"family": f, "n_tasks": len(rows),
                         **{k: float(np.mean([r[k] for r in rows]))
                            for k in ("C_benefit", "C_accuracy", "gap", "mean_gain", "V_delta", "H")}})
    out["family_summary"] = fam_rows
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "dove_tasks.json"), "w"), indent=1)
    pd.DataFrame([{k: v for k, v in t.items() if k not in ("delta", "theta")} for t in out["tasks"]]) \
        .to_csv(os.path.join(RES, "tables", "tasks_per_task.csv"), index=False)
    pd.DataFrame(fam_rows).to_csv(os.path.join(RES, "tables", "tasks_per_family.csv"), index=False)
    print(json.dumps({k: out[k] for k in ("families", "decomposition", "transfer_note")}, indent=1))
    print("excluded:", out["excluded_tasks"])


if __name__ == "__main__":
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    main()
