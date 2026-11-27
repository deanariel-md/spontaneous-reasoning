import numpy as np
import pytest

from src import dataset1 as d1
from src import pilot_checks as pc
from src.estimators import gaussian_agreement


def test_between_agreement_matches_dataset1():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(6, 7)), rng.normal(size=(6, 7))
    A[0, 1] = A[0, 2]
    got = pc.between_agreement(A, B)
    want = [d1.agreement_between(a, b)["C"] for a, b in zip(A, B)]
    assert got == pytest.approx(want)


def test_chosen_k():
    assert pc.chosen_k(-0.1) is None and pc.chosen_k(float("nan")) is None and pc.chosen_k(0.0) is None
    assert pc.chosen_k(1.3) == 1
    k = pc.chosen_k(0.3)
    assert gaussian_agreement(0.3, k) >= 0.8 > gaussian_agreement(0.3, k - 1)


def test_achieved_enumerated():
    rng = np.random.default_rng(1)
    delta = np.linspace(-3, 3, 8)
    assert pc.achieved_enumerated(np.tile(delta, (50, 1)), 3, 100, rng) == pytest.approx(1.0)
    G = delta + rng.normal(0, 4, (4000, 8))
    one = pc.achieved_enumerated(G, 1, 4000, np.random.default_rng(2))
    # brute force for K = 1: C between two independent rows of the bank
    a, b = G[rng.integers(0, 4000, 20000)], G[rng.integers(0, 4000, 20000)]
    assert one == pytest.approx(pc.between_agreement(a, b).mean(), abs=0.01)
    k20 = pc.achieved_enumerated(G, 20, 4000, np.random.default_rng(3))
    big = pc.achieved_enumerated(G, 400, 500, np.random.default_rng(4))      # the multinomial path
    assert one < k20 < big <= 1


def test_achieved_heldout():
    rng = np.random.default_rng(4)
    Gv = np.arange(5.0) + rng.normal(0, 0.01, (24, 5))
    assert pc.achieved_heldout(Gv, 13, 10, rng) is None
    assert pc.achieved_heldout(Gv, 3, 50, rng) == pytest.approx(1.0)


def test_classify():
    assert pc.classify(None, None) == "unresolved"
    assert pc.classify(20, None) == "not measurable"
    assert pc.classify(2, 0.85) == "meets" and pc.classify(2, 0.79) == "short"


def test_shrink_factor():
    rng = np.random.default_rng(5)
    delta = rng.normal(0, 3, 10)
    G = delta + rng.normal(0, 1, (4, 10))
    f = pc.shrink_factor(G)
    R = pc.pilot_r(G)
    assert f == pytest.approx(min(max(4 * R / (1 + 3 * R), 0), 1))
    assert 0 <= f <= 1
    assert pc.shrink_factor(G[:1]) is None
    noise = rng.normal(0, 1, (2, 10))
    assert pc.shrink_factor(noise) >= 0


def test_charged_sd():
    rng = np.random.default_rng(6)
    G = rng.normal(0, 4, 9) + rng.normal(0, 1, (4, 9))
    assert pc.charged_sd(G, 1) == pytest.approx(np.std(G[0], ddof=1))
    from src.estimators import cross_moment
    assert pc.charged_sd(G, 4) == pytest.approx(np.sqrt(cross_moment(G, G)))
    assert pc.charged_sd(rng.normal(size=(2, 9)) * 0 + np.arange(2)[:, None], 2) is None
