"""Samplers over a frozen configuration bank.

Route A: each unit is one configuration drawn independently, with replacement and with
equal probability, from the frozen common bank; the same configuration is used for every
model and both arms within a unit. Discovery and validation units are drawn independently.
The draw order is the fixed nested ordering used for replication-budget displays.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

MASTER_SEED = 20261003
BUDGET_GRID = (1, 2, 4, 8, 12, 16, 24)


def rng_for(*labels, seed: int = MASTER_SEED) -> np.random.Generator:
    """Independent stream per named component, reproducible from the master seed."""
    words = [int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:4], "little") for x in labels]
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


def draw_units(n_bank: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_bank, size=k)


def registered_split(n_bank: int, k_discovery: int, k_validation: int, label: str) -> dict:
    rng = rng_for("registered_split", label)
    disc = draw_units(n_bank, k_discovery, rng)
    val = draw_units(n_bank, k_validation, rng)
    return {"label": label, "master_seed": MASTER_SEED, "n_bank": int(n_bank),
            "probabilities": "uniform over bank, with replacement, independent units",
            "k_discovery": int(k_discovery), "k_validation": int(k_validation),
            "discovery_bank_index": disc.tolist(), "validation_bank_index": val.tolist(),
            "nested_order": "budget K uses the first K discovery units in the order listed"}


def registered_split_bank(config_ids, probabilities, k_discovery: int, k_validation: int,
                          label: str) -> dict:
    """Dataset 2 sampler: units drawn independently, with replacement, with the bank's
    probabilities over the frozen bank. Draws depend on metadata only (ids, probabilities,
    seed), never on outcomes."""
    ids = [str(c) for c in config_ids]
    p = np.asarray(probabilities, float)
    if len(ids) != len(set(ids)) or p.shape != (len(ids),):
        raise ValueError("config ids must be unique, with one probability each")
    if (p < 0).any() or abs(p.sum() - 1) > 1e-9:
        raise ValueError("probabilities must be nonnegative and sum to one")
    rng = rng_for("registered_split_bank", label)
    disc = rng.choice(len(ids), size=k_discovery, replace=True, p=p)
    val = rng.choice(len(ids), size=k_validation, replace=True, p=p)
    out = {"label": label, "master_seed": MASTER_SEED, "n_bank": len(ids),
           "bank_sha256": sha256_json(ids), "probabilities_sha256": sha256_json(p.tolist()),
           "probabilities": "registered bank probabilities, with replacement, independent units",
           "k_discovery": int(k_discovery), "k_validation": int(k_validation),
           "discovery_config_ids": [ids[i] for i in disc],
           "validation_config_ids": [ids[i] for i in val],
           "nested_order": "budget K uses the first K discovery units in the order listed"}
    out["sha256"] = sha256_json({k: v for k, v in out.items()})
    return out


def budgets(k_available: int, start: int = 1) -> list[int]:
    return [k for k in BUDGET_GRID if start <= k <= k_available]


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
