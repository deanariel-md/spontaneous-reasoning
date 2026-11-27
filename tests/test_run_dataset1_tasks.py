"""The per-task secondary analysis script (plan C.3, C.5): score widening and the check-based exclusion."""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("run_dataset1_tasks", os.path.join(ROOT, "scripts", "run_dataset1_tasks.py"))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def test_int8_counts_give_accuracy_in_pp_without_wrapping():
    rows = [{"model": m, "arm": "0_shot", "config_id": c, "n_correct": np.int8(n), "n_items": 100}
            for c, n in ((1, 36), (2, 100)) for m in rt.PANEL]
    full = pd.DataFrame(rows).astype({"n_correct": "int8"})
    got = rt.scores_pp(full, "0_shot", [1, 2], 100)
    assert got.shape == (2, len(rt.PANEL)) and np.allclose(got[0], 36.0) and np.allclose(got[1], 100.0)
    assert (100 * full["n_correct"].to_numpy())[0] != 3600          # the wrap the widening avoids


def test_a_benchmark_with_a_failing_file_is_listed_with_its_checks(tmp_path):
    base = {c: 0 for c in rt.CHECKS}
    chk = pd.DataFrame([{**base, "benchmark_file": "a.parquet"},
                        {**base, "benchmark_file": "b.parquet", "score_vs_exact_match_disagree": 3},
                        {**base, "benchmark_file": "b.parquet"}])
    chk.to_csv(tmp_path / "file_checks.csv", index=False)
    assert rt.failed_check_tasks(str(tmp_path / "file_checks.csv")) == {"b": ["score_vs_exact_match_disagree"]}
