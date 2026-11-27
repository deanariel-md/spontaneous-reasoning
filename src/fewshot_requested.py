"""Five-shot requested-direct arm.

Five-shot minus zero-shot within each answer procedure, on GSM8K-29 and BIG-Bench Hard-28:
  requested: g_R = five-shot requested (new) - zero-shot requested (main collection);
  forced:    g_F = five-shot forced - zero-shot forced.
The analysis compares the two procedures' gains: reversed model pairs, template-only reversals within a procedure, and
both directions of the decision cost (choose the top third under one procedure, value under the other).

Pure functions on arrays (configurations by models, pp) and the write-once file of the discovery stage; the I/O lives in
scripts/build_fewshot_requested_requests.py, scripts/score_fewshot_requested.py and scripts/run_fewshot_requested.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src import decision as dc
from src import decision_directions as dd
from src.estimators import tie_sign

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("gsm8k", "bbh")
N_DISC = 4
N_VAL = 4
SUFFIX = "|fsr"
ARM = "direct_answer"
PROCEDURES = ("requested", "forced")
ARMS4 = ("zs_requested", "zs_forced", "fs_requested", "fs_forced")
SHARE_TOL = 0.05
SEED_LABEL = "d28/fsr/{task}/{split}"


class FewshotRequestedError(Exception):
    """A refusal: an order, hash or completeness condition does not hold."""


class Paths:
    """Where the run reads and writes; `base` replaces the repository root in dry runs and tests."""

    def __init__(self, base: str | os.PathLike | None = None):
        b = Path(base) if base else ROOT
        self.base = b
        self.requests = b / "data" / "requests" / "fewshot_requested"
        self.outputs = b / "data" / "d2_out" / "fewshot_requested"
        self.processed = b / "data" / "processed" / "fewshot_requested"
        self.results = b / "results" / "fewshot_requested"
        self.configs = b / "manifests" / "fewshot_requested_configs.json"
        self.predictions = b / "manifests" / "predictions_fewshot_requested.json"
        self.predictions_sha = b / "manifests" / "predictions_fewshot_requested.sha256"


# --------------------------------------------------------------------------------------------- files


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(p: str | os.PathLike) -> str:
    return sha256_bytes(Path(p).read_bytes())


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=_default, allow_nan=True)


def write_json(path, obj) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = dumps(obj).encode("utf-8")
    p.write_bytes(raw)
    return sha256_bytes(raw)


def write_once(path, sha_path, obj) -> str:
    """Write the predictions file and its sha256 file; refuse when either exists already."""
    p, s = Path(path), Path(sha_path)
    if p.exists() or s.exists():
        raise FewshotRequestedError(f"{p.name} exists already; it is written once")
    sha = write_json(p, obj)
    s.write_text(f"{sha}  {p.name}\n", encoding="utf-8")
    return sha


def check_predictions(paths: Paths) -> str:
    """sha256 of the saved predictions; refused when absent or changed since saved."""
    if not paths.predictions.is_file() or not paths.predictions_sha.is_file():
        raise FewshotRequestedError("the predictions are not saved; run `score_fewshot_requested.py --stage discovery` "
                                    "and `run_fewshot_requested.py --stage discovery` first")
    want = paths.predictions_sha.read_text(encoding="utf-8").split()[0]
    got = sha256_file(paths.predictions)
    if got != want:
        raise FewshotRequestedError(f"{paths.predictions.name} changed after it was saved (sha256 {got} != {want})")
    return got


# --------------------------------------------------------------------------------------------- configurations


def seed_for(task: str, split: str) -> int:
    return dc.label_seed(SEED_LABEL.format(task=task, split=split))


def draw_configurations(task: str, sampler: Mapping, n_disc: int = N_DISC, n_val: int = N_VAL) -> dict:
    """The arm-wording run's rule with its own seed labels: n_disc of the distinct discovery configurations and n_val of the distinct
    validation configurations no discovery draw uses, uniformly without replacement, in the order drawn."""
    D = list(dict.fromkeys(map(str, sampler["discovery_config_ids"])))
    V = [c for c in dict.fromkeys(map(str, sampler["validation_config_ids"])) if c not in set(D)]
    if len(D) < n_disc or len(V) < n_val:
        raise FewshotRequestedError(f"{task}: {len(D)} discovery and {len(V)} fresh validation configurations; "
                                    f"need {n_disc} and {n_val}")
    out = {}
    for name, pool, n in (("discovery", D, n_disc), ("validation", V, n_val)):
        rng = np.random.default_rng(seed_for(task, name))
        ix = rng.choice(len(pool), size=n, replace=False)
        out[name] = [pool[int(i)] for i in ix]
        out[f"{name}_pool"] = pool
        out[f"{name}_seed_label"] = SEED_LABEL.format(task=task, split=name)
    return out


def job_config_id(task: str, config_id: str) -> str:
    return f"{task}/{config_id}"


def parse_job_config_id(s: str) -> tuple[str, str]:
    t, _, c = str(s).partition("/")
    if t not in TASKS or not c:
        raise FewshotRequestedError(f"configuration id {s!r} is not <task>/<config>")
    return t, c


# --------------------------------------------------------------------------------------------- arrays


def accuracy_array(df, configs: Sequence[str], panel: Sequence[str], n_items: int, value: str = "strict_correct"):
    """(K, M) accuracy in pp from a long table (model_key, config_id, item_id, value); every cell must hold exactly
    n_items distinct items (a missing cell or item is refused, never imputed)."""
    import pandas as pd
    d = df[df["config_id"].astype(str).isin(set(configs)) & df["model_key"].astype(str).isin(set(panel))]
    g = d.groupby([d["config_id"].astype(str), d["model_key"].astype(str)])
    n = g["item_id"].nunique()
    rows = g.size()
    if (rows != n).any():
        raise FewshotRequestedError("an item occurs twice in a cell")
    full = pd.MultiIndex.from_product([list(configs), list(panel)])
    n = n.reindex(full)
    bad = n[n.isna() | (n != n_items)]
    if len(bad):
        raise FewshotRequestedError(f"{len(bad)} cells lack items (need {n_items}), e.g. {list(bad.index[:3])}")
    acc = (100.0 * g[value].mean()).reindex(full).to_numpy(float).reshape(len(configs), len(panel))
    return acc


def gains(acc: Mapping[str, np.ndarray]) -> dict:
    """{'requested': g_R, 'forced': g_F}, each (K, M) pp."""
    return {"requested": np.asarray(acc["fs_requested"], float) - np.asarray(acc["zs_requested"], float),
            "forced": np.asarray(acc["fs_forced"], float) - np.asarray(acc["zs_forced"], float)}


def pair_count(g1, g2) -> dict:
    """Model pairs between two gain vectors (model means): reversed (opposite strict order), ties (a gap of at most
    1e-9 pp under either), the reversed share of all pairs, and 1 - C with ties counted one half."""
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    i, j = np.triu_indices(len(g1), 1)
    s1, s2 = tie_sign(g1[i] - g1[j]), tie_sign(g2[i] - g2[j])
    rev = (s1 * s2) < 0
    tie = (s1 == 0) | (s2 == 0)
    n = len(i)
    return {"pairs": int(n), "reversed": int(rev.sum()), "ties": int(tie.sum()),
            "reversed_share": float(rev.mean()) if n else float("nan"),
            "one_minus_C": float((rev.sum() + 0.5 * tie.sum()) / n) if n else float("nan")}


def template_reversal(G) -> dict:
    """Within one procedure, one configuration against another (all pairs of distinct configurations): mean 1 - C
    (ties one half) and mean reversed share over the configuration pairs."""
    G = np.asarray(G, float)
    K = G.shape[0]
    res = [pair_count(G[a], G[b]) for a in range(K) for b in range(a + 1, K)]
    if not res:
        return {"configuration_pairs": 0, "one_minus_C": float("nan"), "reversed_share": float("nan")}
    return {"configuration_pairs": len(res), "one_minus_C": float(np.mean([r["one_minus_C"] for r in res])),
            "reversed_share": float(np.mean([r["reversed_share"] for r in res]))}


def one_draw_choices(G, n: int) -> tuple[np.ndarray, list[int]]:
    """(K, M) weights of the top n by own gain on each configuration (row), and the ties at the cut per row."""
    G = np.asarray(G, float)
    out = [dc.choose(g, n) for g in G]
    return np.array([w for w, _ in out]), [int(t) for _, t in out]


# --------------------------------------------------------------------------------------------- discovery stage


def discovery_quantities(acc_disc: Mapping[str, Mapping[str, np.ndarray]], panels: Mapping[str, Sequence[str]]) -> dict:
    """Per task, from the discovery arrays alone: the one-draw choices under each procedure (row = discovery
    configuration, in the order drawn), the four-configuration mean choice, and the discovery reversal share."""
    out = {}
    for t, acc in acc_disc.items():
        panel = list(panels[t])
        M = len(panel)
        n = dc.third_size(M)
        G = gains(acc)
        blk = {"models": panel, "n_chosen": n, "choices": {}, "mean_choices": {}, "ties_at_cut": {}}
        for p in PROCEDURES:
            W, ties = one_draw_choices(G[p], n)
            blk["choices"][p] = W.tolist()
            blk["ties_at_cut"][p] = ties
            blk["mean_choices"][p] = dc.choose(G[p].mean(0), n)[0].tolist()
        blk["mean_gain"] = {p: G[p].mean(0).tolist() for p in PROCEDURES}
        blk["reversal"] = pair_count(G["requested"].mean(0), G["forced"].mean(0))
        blk["template_reversal"] = {p: template_reversal(G[p]) for p in PROCEDURES}
        out[t] = blk
    return out


def predictions_record(disc: Mapping, configs_manifest_sha: str, tables: Mapping) -> dict:
    """The write-once record of the discovery stage."""
    stmts = {t: {"validation_reversed_share_within": [max(0.0, b["reversal"]["reversed_share"] - SHARE_TOL),
                                                      min(1.0, b["reversal"]["reversed_share"] + SHARE_TOL)],
                 "tolerance": SHARE_TOL} for t, b in disc.items()}
    return {"kind": "predictions_fewshot_requested", "study_part": "five-shot requested-direct arm",
            "design": "src/fewshot_requested.py", "configs_manifest_sha256": configs_manifest_sha,
            "inputs": dict(tables), "statements": stmts,
            "rule": "top third (decision.third_size) by own five-shot gain from one discovery configuration, under "
                    "each answer procedure; ties at the cut share the remaining places (decision.choose)",
            "tasks": dict(disc)}


def same_quantities(a, b, tol: float = 1e-9) -> bool:
    """Equality of two JSON-like structures, floats to within tol."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(same_quantities(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_quantities(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(float(a) - float(b)) <= tol
    return a == b


# --------------------------------------------------------------------------------------------- full run


def evaluate_task(saved: Mapping, acc_disc: Mapping, acc_val: Mapping, families: Sequence[str], n_boot: int,
                  rng) -> dict:
    """Validation evaluation of one task from the saved choices and the validation arrays."""
    panel = list(saved["models"])
    n = int(saved["n_chosen"])
    Gd, Gv = gains(acc_disc), gains(acc_val)
    W = {p: np.asarray(saved["choices"][p], float) for p in PROCEDURES}
    Wm = {p: np.atleast_2d(np.asarray(saved["mean_choices"][p], float)) for p in PROCEDURES}
    out = {"models": panel, "M": len(panel), "n_chosen": n, "K_discovery": int(Gd["requested"].shape[0]),
           "K_validation": int(Gv["requested"].shape[0])}
    for split, G in (("discovery", Gd), ("validation", Gv)):
        out[f"reversal_{split}"] = pair_count(G["requested"].mean(0), G["forced"].mean(0))
        out[f"template_reversal_{split}"] = {p: template_reversal(G[p]) for p in PROCEDURES}
    out["reversal_validation_resampling"] = dd.reversal_uncertainty(Gv["requested"].mean(0), Gv["forced"].mean(0),
                                                                    families, n_boot, rng)
    lo, hi = max(0.0, out["reversal_discovery"]["reversed_share"] - SHARE_TOL), \
        min(1.0, out["reversal_discovery"]["reversed_share"] + SHARE_TOL)
    v = out["reversal_validation"]["reversed_share"]
    out["statement_share"] = {"predicted_within": [lo, hi], "observed": v, "holds": bool(lo <= v <= hi)}
    out["decision_one_draw"] = dd.both_directions(W, Gv, n)
    out["decision_mean_choice"] = dd.both_directions(Wm, Gv, n)
    out["regret"] = {p: {"one_draw": float(dc.regret_plugin(W[p], Gv[p], n).mean()),
                         "mean_choice": float(dc.regret_plugin(Wm[p], Gv[p], n).mean())} for p in PROCEDURES}
    out["per_model"] = [{"model_key": k, **{f"{a}_{s}": float(np.asarray(A[a], float)[:, j].mean())
                                            for s, A in (("discovery", acc_disc), ("validation", acc_val)) for a in ARMS4},
                         "gain_requested_validation": float(Gv["requested"][:, j].mean()),
                         "gain_forced_validation": float(Gv["forced"][:, j].mean())}
                        for j, k in enumerate(panel)]
    out["mean_gain_validation"] = {p: float(Gv[p].mean()) for p in PROCEDURES}
    return out
