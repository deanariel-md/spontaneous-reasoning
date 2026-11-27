"""Synthetic configuration banks with known structure, for checking the Dataset 1 pipeline.

`synthetic_counts` draws subject-level correct counts for two arms over a bank of
configurations: logit p = model ability + arm gain + subject difficulty + configuration effect
(shared by all models) + model-by-configuration noise, with a within-configuration arm
correlation. `bank_from_counts` pools subjects with equal weights, in pp.
"""
from __future__ import annotations

import numpy as np


def synthetic_counts(rng, M=5, S=6, C=400, n_items=100, ability=None, gain=None,
                     cfg_sd=0.3, noise_sd=0.25, arm_corr=0.5, subject_sd=0.5):
    ability = np.linspace(-1.0, 0.8, M) if ability is None else np.asarray(ability, float)
    gain = np.linspace(0.4, 0.1, M) if gain is None else np.asarray(gain, float)
    subj = rng.normal(0, subject_sd, S)
    cfg = rng.normal(0, cfg_sd, C)
    z0 = rng.normal(size=(S, C, M))
    z1 = arm_corr * z0 + np.sqrt(1 - arm_corr ** 2) * rng.normal(size=(S, C, M))
    eta0 = ability[None, None, :] + subj[:, None, None] + cfg[None, :, None] + noise_sd * z0
    eta1 = eta0 - noise_sd * z0 + noise_sd * z1 + gain[None, None, :]
    n = np.full(S, n_items)
    nc0 = rng.binomial(n_items, 1 / (1 + np.exp(-eta0)))
    nc1 = rng.binomial(n_items, 1 / (1 + np.exp(-eta1)))
    return nc0, nc1, n


def bank_from_counts(nc0, nc1, n):
    from .controls import pooled
    return pooled(nc0, n), pooled(nc1, n)


class GSM8KSim:
    """Item-level synthetic Dataset 2 outputs for checking the pipeline (never for inference).

    A finite bank of C configurations. For model i, item n, configuration c and arm a, the
    probability of a correct final answer is sigmoid(ability_i + easiness_n + gain_i [a = cot]
    + tau_shared s_ca + tau z_cia), with s and z correlated across arms within a configuration.
    The answer is strictly extractable with probability 1 - q_ia (format compliance); a
    non-compliant correct answer is recovered by lenient extraction with probability
    `recover`. Each configuration's item-level outcomes are realized once, from a stream seeded
    by its bank index, as greedy decoding would fix them; the exact targets are computed from
    those realized bank scores.
    """

    arms = ("direct_answer", "zero_shot_cot")

    def __init__(self, rng, M=20, n_families=5, C=2000, n_items=1319, tau=0.12, tau_shared=0.15,
                 rho_arm=0.3, recover=0.7):
        self.seed = int(rng.integers(0, 2 ** 31))
        self.M, self.C, self.N = M, C, n_items
        self.panel = [f"sim-{k:02d}" for k in range(M)]
        self.families = [f"fam{k % n_families}" for k in range(M)]
        self.tiers = {mk: ("low", "mid", "high")[min(2, 3 * k // M)] for k, mk in enumerate(self.panel)}
        self.ability = np.linspace(-1.6, 1.4, M)
        self.gain = 0.9 - 0.25 * (self.ability - self.ability.mean()) + rng.normal(0, 0.15, M)
        self.easiness = rng.normal(0.4, 1.2, n_items)
        self.q = np.stack([np.linspace(0.12, 0.02, M), np.linspace(0.20, 0.04, M)], 1)
        self.recover = recover
        self.config_ids = [f"cfg{c:05d}" for c in range(C)]
        s0 = rng.normal(size=C)
        s1 = rho_arm * s0 + np.sqrt(1 - rho_arm ** 2) * rng.normal(size=C)
        z0 = rng.normal(size=(C, M))
        z1 = rho_arm * z0 + np.sqrt(1 - rho_arm ** 2) * rng.normal(size=(C, M))
        self.eff = np.stack([tau_shared * s0[:, None] + tau * z0,
                             tau_shared * s1[:, None] + tau * z1], 0)       # (arm, C, M)
        self._counts = None

    def _realize(self, c):
        """Item-level (arm, M, N) outcomes of bank configuration c: correct, strict, lenient."""
        rng = np.random.default_rng([self.seed, c])
        out = []
        for a in (0, 1):
            eta = (self.ability + (self.gain if a == 1 else 0.0))[:, None] + self.eff[a][c][:, None] \
                + self.easiness[None, :]
            correct = rng.random((self.M, self.N)) < 1 / (1 + np.exp(-eta))
            comp = rng.random((self.M, self.N)) >= self.q[:, a, None]
            rec = rng.random((self.M, self.N)) < self.recover
            out.append((correct & comp, correct & (comp | rec), comp))
        return out

    def bank_counts(self):
        """Correct counts (outcome, arm, C, M) over the whole bank: strict, lenient."""
        if self._counts is None:
            n = np.empty((2, 2, self.C, self.M), dtype=np.int64)
            for c in range(self.C):
                for a, (s, l, _) in enumerate(self._realize(c)):
                    n[0, a, c], n[1, a, c] = s.sum(1), l.sum(1)
            self._counts = n
        return self._counts

    def truth(self, outcome="strict_correct") -> dict:
        n = self.bank_counts()[0 if outcome == "strict_correct" else 1]
        from .calibration import bank_truth
        return bank_truth(100 * n[0] / self.N, 100 * n[1] / self.N)

    def scored(self, config_ids) -> "pd.DataFrame":
        """Item-level rows for the distinct listed configurations, every model and both arms."""
        import pandas as pd
        parts = []
        m_i, n_i = np.meshgrid(np.arange(self.M), np.arange(self.N), indexing="ij")
        for cid in sorted(set(map(str, config_ids))):
            for a, (s, l, comp) in enumerate(self._realize(self.config_ids.index(cid))):
                parts.append(pd.DataFrame({
                    "model_key": np.array(self.panel)[m_i.ravel()], "arm": self.arms[a],
                    "config_id": cid, "item_id": n_i.ravel(),
                    "strict_correct": s.ravel().astype(np.int8),
                    "lenient_correct": l.ravel().astype(np.int8),
                    "compliant": comp.ravel().astype(np.int8)}))
        return pd.concat(parts, ignore_index=True)
