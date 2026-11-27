"""Checks for the secondary task-collection analyses (Appendix C.3, C.5)."""
import numpy as np
import pytest

from src import simulate, tasks
from src.estimators import fixed_task_components


def _banks(seed, T=4, C=300, M=6):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        gain = rng.normal(0.3, 0.2, M)
        nc0, nc1, n = simulate.synthetic_counts(rng, M=M, S=2, C=C, gain=gain)
        out.append(simulate.bank_from_counts(nc0, nc1, n))
    return out


def test_family_weights_macro_average():
    w = tasks.family_weights(["a", "a", "b", "c", "c", "c"])
    assert w.sum() == pytest.approx(1)
    assert w[0] == pytest.approx(1 / 6) and w[2] == pytest.approx(1 / 3) and w[3] == pytest.approx(1 / 9)


def test_decomposition_identity_and_agrees_with_appendix_a():
    banks = _banks(1)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    ex = tasks.observed_task_decomposition(banks, w)
    assert ex["average_task_profile"] + ex["model_by_task"] == pytest.approx(ex["true_within_task"])
    # Appendix A estimator on many independent joint bundles approaches the exact values.
    rng = np.random.default_rng(2)
    K = 3000
    G = np.stack([(P - B)[rng.integers(0, B.shape[0], K)] for B, P in banks])
    est = fixed_task_components(G, w)
    for key in ("average_task_profile", "true_within_task", "prompt_noise"):
        assert est[key] == pytest.approx(ex[key], rel=0.1, abs=0.3)


def test_transfer_requires_four_families_and_is_exact():
    banks = _banks(3, T=4)
    assert tasks.transfer_holdout(banks, ["a", "a", "b", "c"]) == []
    res = tasks.transfer_holdout(banks, ["a", "b", "c", "d"])
    assert len(res) == 4
    d = np.stack([(P - B).mean(0) for B, P in banks])
    d = d - d.mean(1, keepdims=True)
    r = res[0]
    pred = d[1:].mean(0)
    assert r["risk_zero"] == pytest.approx(np.mean(d[0] ** 2))
    assert r["risk_transfer"] == pytest.approx(np.mean((d[0] - pred) ** 2))
