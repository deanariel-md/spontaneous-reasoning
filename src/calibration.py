"""Interval calibration suite.

A scenario is a finite population of evaluation units with exact fixed-panel targets: a
synthetic configuration bank (M = 20 models, GSM8K-sized item set), the DOVE MMLU bank itself,
or a DOVE known-effect control (pseudo-arms and injected effects, src/controls.py). One
replicate draws K discovery units and K validation units independently with replacement,
computes every estimate with every candidate interval, and records whether each
interval contains the exact target. Nothing is clipped; undefined intervals count as misses
and are tallied separately.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import controls, dataset1 as d1, intervals as iv
from .estimators import paired_loss_difference
from .sampling import rng_for

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ALPHA = iv.ALPHA
MARGIN_VDELTA = d1.PRACTICAL["V_delta_pp2"]

# ----------------------------------------------------------------------------- scenarios

M20 = 20
THETA20 = np.linspace(20, 75, M20)


def _orthogonal(theta, sd, label):
    """Fixed residual vector orthogonal to capability, panel SD = sd (M - 1 denominator)."""
    return controls.residual_direction(theta, sd, rng_for("calibration-residual", label))


def synthetic_specs() -> dict:
    th = THETA20
    tc = th - th.mean()
    base = {"M": M20, "C": 4000, "N": 1319, "tau": 0.12, "tau_shared": 0.15, "rho_arm": 0.3}
    het = 10 + _orthogonal(th, 2.0, "boundary")
    floor_ceiling = np.concatenate([[1.0, 2.0, 3.0, 4.0], np.linspace(10, 80, 12), [90.0, 93.0, 95.0, 97.0]])
    near = np.linspace(49, 51, M20)
    specs = {
        "zero_heterogeneity": dict(base, theta=th, delta=np.full(M20, 10.0)),
        "boundary_heterogeneity": dict(base, theta=th, delta=het),
        "weaker_benefit_more": dict(base, theta=th, delta=10 - 0.3 * tc),
        "stronger_benefit_more": dict(base, theta=th, delta=10 + 0.3 * tc),
        "near_zero_capability_variance": dict(base, theta=near, delta=10 + _orthogonal(near, 2.0, "near")),
        "sparse_outlier": dict(base, theta=th, delta=np.where(np.arange(M20) == 7, 16.0, 10.0)),
        "nonlinear_benefit": dict(base, theta=th, delta=14 - 0.015 * tc ** 2),
        "ties_40_items": dict(base, theta=th, delta=het, N=40),
        "ceiling_floor": dict(base, theta=floor_ceiling,
                              delta=np.minimum(8.0, 0.75 * (99.5 - floor_ceiling))),
        "heterogeneous_noise": dict(base, theta=th, delta=het,
                                    tau=rng_for("calibration-tau").permutation(np.linspace(0.03, 0.35, M20))),
        "correlated_errors": dict(base, theta=th, delta=het, tau_shared=0.6, rho_arm=0.9),
    }
    for name, s in specs.items():
        s["name"], s["kind"] = name, "synthetic"
    return specs


def dove_control_specs(theta) -> dict:
    """The seven Dataset 1 known-effect scenarios (scripts/run_dataset1.py), on the DOVE bank."""
    th = np.asarray(theta, float)
    scen = [
        {"name": "dove_null_baseline", "kind": "null", "population": "baseline"},
        {"name": "dove_null_intervention", "kind": "null", "population": "intervention"},
        {"name": "dove_constant_5", "kind": "constant", "gain": 5.0},
        {"name": "dove_weaker_more", "kind": "capability", "gain": 5.0, "beta": -0.3},
        {"name": "dove_stronger_more", "kind": "capability", "gain": 5.0, "beta": 0.3},
        {"name": "dove_residual_sd3", "kind": "residual", "gain": 5.0,
         "u": controls.residual_direction(th, 3.0, rng_for("calibration-residual", "dove")).tolist()},
        {"name": "dove_sparse_outlier", "kind": "sparse_outlier", "gain": 5.0, "index": 0, "extra": 6.0},
    ]
    return {s["name"]: s for s in scen}


# ----------------------------------------------------------------------------- populations


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _solve_intercepts(target, Z):
    """eta (M,) with mean_c sigmoid(eta + Z[c]) = target, by bisection."""
    lo, hi = np.full(Z.shape[1], -20.0), np.full(Z.shape[1], 20.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        f = _sigmoid(mid + Z).mean(0) - target
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return (lo + hi) / 2


def synthetic_bank(spec, rng):
    """Bank (C, M) of pooled scores in pp for both arms. Logit = intercept + shared configuration
    effect + model-specific configuration effect, arms correlated within a configuration;
    score = binomial over N items. Intercepts are solved so the expected bank means equal the
    design theta and theta + delta; the realized bank defines the exact targets."""
    M, C, N = spec["M"], spec["C"], spec["N"]
    tau = np.broadcast_to(np.asarray(spec["tau"], float), (M,))
    rho, ts = spec["rho_arm"], spec["tau_shared"]
    s0 = rng.normal(size=C)
    s1 = rho * s0 + math.sqrt(1 - rho ** 2) * rng.normal(size=C)
    z0 = rng.normal(size=(C, M))
    z1 = rho * z0 + math.sqrt(1 - rho ** 2) * rng.normal(size=(C, M))
    Z0, Z1 = ts * s0[:, None] + tau * z0, ts * s1[:, None] + tau * z1
    th, de = np.asarray(spec["theta"], float), np.asarray(spec["delta"], float)
    p0 = _sigmoid(_solve_intercepts(th / 100, Z0) + Z0)
    p1 = _sigmoid(_solve_intercepts((th + de) / 100, Z1) + Z1)
    return 100 * rng.binomial(N, p0) / N, 100 * rng.binomial(N, p1) / N


def scale_noise(spec, s) -> dict:
    """The same scenario with both logit-scale configuration SDs multiplied by s (s = 1 is the
    calibration design); used to match a measured noise level."""
    out = dict(spec)
    out["tau"] = np.asarray(spec["tau"], float) * s
    out["tau_shared"] = spec["tau_shared"] * s
    return out


def bank_truth(B, P) -> dict:
    t = d1.exact_targets(B, P)
    cb, ca = d1.exact_agreement(P - B)["C"], d1.exact_agreement(B)["C"]
    return {"V_delta": t["V_delta"], "A": t["A"], "V_theta": t["V_theta"], "H": t["H"],
            "C_benefit": cb, "C_accuracy": ca, "gap": ca - cb, "delta": np.array(t["delta"]),
            "theta": np.array(t["theta"]), "noise_W_gain": t["noise_W_gain"],
            "noise_W_baseline": t["noise_W_baseline"], "R_benefit": t["R_benefit"]}


def load_dove_block():
    """DOVE MMLU block counts (manifests/dove_mmlu_block.json, data/processed/dove)."""
    import pandas as pd
    from . import dove_io as io
    spec = json.load(open(os.path.join(ROOT, "manifests", "dove_mmlu_block.json")))
    blk = pd.read_parquet(os.path.join(io.PROCESSED, "mmlu_block.parquet"))
    subjects, common, panel = spec["subjects"], spec["common_config_ids"], spec["panel"]
    idx = {c: k for k, c in enumerate(common)}
    si = {s: k for k, s in enumerate(subjects)}
    mi = {m: k for k, m in enumerate(panel)}
    nc = {}
    for a in ("0_shot", "5_shot"):
        x = blk[blk["arm"] == a]
        arr = np.full((len(subjects), len(common), len(panel)), -1, dtype=np.int64)
        arr[x["subject"].map(si).to_numpy(), x["config_id"].map(idx).to_numpy(),
            x["model"].map(mi).to_numpy()] = x["n_correct"].to_numpy()
        assert (arr >= 0).all()
        nc[a] = arr
    n_items = np.array([spec["items_per_subject"][s] for s in subjects])
    return nc["0_shot"], nc["5_shot"], n_items


class Population:
    """Sampler for one scenario: draw(K, rng) -> (B, P) with K units; truth dict."""

    def __init__(self, name, phase, dove=None, noise_scale=1.0):
        self.name = name
        syn = synthetic_specs()
        if name in syn:
            self.kind = "bank"
            spec = syn[name] if noise_scale == 1.0 else scale_noise(syn[name], noise_scale)
            self.B, self.P = synthetic_bank(spec, rng_for("calibration-bank", phase, name))
            self.truth = bank_truth(self.B, self.P)
        elif name == "dove_real":
            nc0, nc1, n = dove if dove is not None else load_dove_block()
            self.kind = "bank"
            self.B, self.P = controls.pooled(nc0, n), controls.pooled(nc1, n)
            self.truth = bank_truth(self.B, self.P)
        else:
            nc0, nc1, n = dove if dove is not None else load_dove_block()
            self.kind = "control"
            self.nc0, self.nc1, self.n = nc0, nc1, n
            self.theta = controls.pooled(nc0, n).mean(0)
            self.scenario = dove_control_specs(self.theta)[name]
            sc = self.scenario
            d = (np.zeros(len(self.theta)) if sc["kind"] == "null"
                 else controls.scenario_effects(self.theta, sc))
            th = (controls.pooled(nc1, n).mean(0) if sc["kind"] == "null"
                  and sc.get("population") == "intervention" else self.theta)
            V_theta = d1.pvar(th)
            A = d1.pcov(d, th)
            self.truth = {"V_delta": d1.pvar(d), "A": A, "V_theta": V_theta,
                          "H": -10 * A / V_theta, "delta": d, "theta": th}
        self.M = len(self.truth["delta"])

    def draw(self, K, rng):
        if self.kind == "bank":
            idx = rng.integers(0, self.B.shape[0], K)
            return self.B[idx], self.P[idx]
        B, P, _ = controls.draw_control_units(self.nc0, self.nc1, self.n, K, self.scenario,
                                              self.theta, rng)
        return B, P


# ----------------------------------------------------------------------------- one replicate

MOMENT_METHODS = ("jack_t", "boot_naive", "boot_distinct")


def _interval_set(est, loo, boot_n, boot_d, boot_loo=None, m=None):
    out = {"jack_t": iv.jack_t(est, loo)}
    if boot_loo is not None:
        out["boot_t"] = iv.boot_t(est, loo, boot_d, boot_loo, m)
    lo, hi, _ = iv.percentile(boot_n)
    out["boot_naive"] = (lo, hi)
    lo, hi, _ = iv.percentile(boot_d)
    out["boot_distinct"] = (lo, hi)
    out["bca_distinct"] = iv.bca(est, boot_d, loo)
    return out


def replicate(pop: Population, K: int, rng, n_boot: int) -> dict:
    """All estimates and candidate intervals for one draw of K discovery and K validation units."""
    B, P = pop.draw(K, rng)
    G = P - B
    m = iv.multiplicities(rng, K, n_boot)
    res = {}
    cm = {"V_theta": iv.CrossMoment(B, B), "A": iv.CrossMoment(G, B), "V_delta": iv.CrossMoment(G, G)}
    est = {k: c.estimate() for k, c in cm.items()}
    loo = {k: c.loo() for k, c in cm.items()}
    bn = {k: c.boot(m, distinct=False) for k, c in cm.items()}
    bd = {k: c.boot(m, distinct=True) for k, c in cm.items()}
    bl = {k: c.boot_loo(m) for k, c in cm.items()}
    for k in cm:
        res[k] = {"est": est[k], "ci": _interval_set(est[k], loo[k], bn[k], bd[k], bl[k], m)}
    res["V_delta"]["ci"]["mls"] = iv.mls_vdelta(G)
    res["V_delta_upper"] = {"est": est["V_delta"], "ub": {
        "jack_t": iv.jack_t_upper(est["V_delta"], loo["V_delta"]),
        "boot_naive": iv.percentile_upper(bn["V_delta"]),
        "boot_distinct": iv.percentile_upper(bd["V_delta"]),
        "bca_distinct": iv.bca_upper(est["V_delta"], bd["V_delta"], loo["V_delta"]),
        "boot_t": iv.boot_t_upper(est["V_delta"], loo["V_delta"], bd["V_delta"], bl["V_delta"], m),
        "mls": iv.mls_vdelta_upper(G)}}
    # H = -10 A / V_theta; undefined (NaN) where a denominator is not positive.
    with np.errstate(divide="ignore", invalid="ignore"):
        h = lambda a, v: np.where(v > 0, -10 * a / np.where(v > 0, v, 1.0), np.nan)
        H = float(h(est["A"], est["V_theta"]))
        res["H"] = {"est": H, "ci": _interval_set(H, h(loo["A"], loo["V_theta"]),
                                                    h(bn["A"], bn["V_theta"]), h(bd["A"], bd["V_theta"]),
                                                    h(bl["A"], bl["V_theta"]), m),
                    "denominator_nonpositive": bool(est["V_theta"] <= 0)}
    kind, a, b = iv.fieller(est["A"], est["V_theta"], loo["A"], loo["V_theta"])
    res["H"]["fieller"] = iv.scale_fieller(kind, a, b, -10.0)
    # Step 1: agreement on the same K units.
    ag_b, ag_a = iv.Agreement(G), iv.Agreement(B)
    for name, ob in (("C_benefit", ag_b), ("C_accuracy", ag_a)):
        e = ob.estimate()
        res[name] = {"est": e, "ci": _interval_set(e, ob.loo(), ob.boot(m, False), ob.boot(m, True),
                                                   ob.boot_loo(m), m)}
    e = res["C_accuracy"]["est"] - res["C_benefit"]["est"]
    res["gap"] = {"est": e, "ci": _interval_set(
        e, ag_a.loo() - ag_b.loo(), ag_a.boot(m, False) - ag_b.boot(m, False),
        ag_a.boot(m, True) - ag_b.boot(m, True), ag_a.boot_loo(m) - ag_b.boot_loo(m), m)}
    # Step 3: predictions fitted on the discovery units, evaluated on K validation units.
    preds = {k: v[0] for k, v in d1.fit_predictors(B, P).items()}
    Bv, Pv = pop.draw(K, rng)
    Z = Pv - Bv
    mv = iv.multiplicities(rng, K, n_boot)
    delta = pop.truth["delta"]
    for f, g in d1.COMPARISONS:
        L = paired_loss_difference(Z, preds[f], preds[g])
        e = float(L.mean())
        loo_l = (L.sum() - L) / (K - 1)
        bm = mv @ L / K
        lo, hi, _ = iv.percentile(bm)
        res[f"risk:{f}-{g}"] = {"est": e, "ci": {"jack_t": iv.jack_t(e, loo_l), "boot": (lo, hi)},
                                "target": float(np.mean((delta - preds[f]) ** 2 - (delta - preds[g]) ** 2))}
    return res


# ----------------------------------------------------------------------------- tallies


class Tally:
    """Coverage, tail misses, undefined intervals and widths for one (target, method)."""

    def __init__(self):
        self.n = self.hit = self.lo_miss = self.hi_miss = self.undef = self.unbounded = 0
        self.widths = []
        self.err = []

    def add(self, lo, hi, truth, est=None):
        self.n += 1
        if est is not None and np.isfinite(est):
            self.err.append(est - truth)
        if np.isnan(lo) or np.isnan(hi):
            self.undef += 1
            return
        if np.isinf(lo) or np.isinf(hi):
            self.unbounded += 1
        else:
            self.widths.append(hi - lo)
        if truth < lo:
            self.lo_miss += 1
        elif truth > hi:
            self.hi_miss += 1
        else:
            self.hit += 1

    def add_set(self, kind, a, b, truth):
        """Fieller sets."""
        self.n += 1
        if kind in ("undefined", "empty"):
            self.undef += 1
            return
        if kind == "bounded":
            self.widths.append(b - a)
        else:
            self.unbounded += 1
        if iv.fieller_covers(kind, a, b, truth):
            self.hit += 1
        elif kind == "bounded" and truth < a:
            self.lo_miss += 1
        else:
            self.hi_miss += 1

    def halfwidth_median_unbounded_inf(self) -> float:
        """Median half-width over all n sets, with unbounded and undefined intervals counted as infinite
        width. inf when half or more of the n sets are infinite; nan only when n = 0. Nothing is clipped."""
        n = self.n
        if n == 0:
            return float("nan")
        # every add() path records exactly one of: a finite width, an unbounded set, an undefined set;
        # the missing n - len(widths) sets are the infinite ones and sort after every finite width
        s = np.sort(np.asarray(self.widths, float))
        at = lambda i: float(s[i]) if i < len(s) else float("inf")  # noqa: E731
        med = at(n // 2) if n % 2 else (at(n // 2 - 1) + at(n // 2)) / 2
        return med / 2

    def summary(self) -> dict:
        w = np.array(self.widths)
        e = np.array(self.err)
        return {"n": self.n, "coverage": self.hit / self.n if self.n else float("nan"),
                "coverage_lower95": iv.coverage_lower_bound(self.hit, self.n),
                "miss_below": self.lo_miss / self.n, "miss_above": self.hi_miss / self.n,
                "undefined": self.undef / self.n, "unbounded": self.unbounded / self.n,
                "width_median": float(np.median(w)) if len(w) else float("nan"),
                "width_mean": float(w.mean()) if len(w) else float("nan"),
                "halfwidth_median": float(np.median(w) / 2) if len(w) else float("nan"),
                # unbounded and undefined intervals count as infinite width (median over all n)
                "halfwidth_median_unbounded_inf": self.halfwidth_median_unbounded_inf(),
                "bias": float(e.mean()) if len(e) else float("nan"),
                "bias_mc_se": float(e.std(ddof=1) / math.sqrt(len(e))) if len(e) > 1 else float("nan")}


class UpperTally:
    def __init__(self):
        self.n = self.hit = self.undef = self.below_margin = 0
        self.values = []

    def add(self, ub, truth):
        self.n += 1
        if not np.isfinite(ub):
            self.undef += 1
            return
        self.values.append(ub)
        self.hit += int(truth <= ub)
        self.below_margin += int(ub < MARGIN_VDELTA)

    def summary(self) -> dict:
        v = np.array(self.values)
        return {"n": self.n, "coverage": self.hit / self.n,
                "coverage_lower95": iv.coverage_lower_bound(self.hit, self.n),
                "undefined": self.undef / self.n,
                "P_upper_below_margin": self.below_margin / self.n,
                "upper_median": float(np.median(v)) if len(v) else float("nan")}


def run_cell(name: str, K: int, n_rep: int, n_boot: int, phase: str, dove=None, noise_scale=1.0) -> dict:
    """Coverage summaries for one scenario and budget K."""
    pop = Population(name, phase, dove, noise_scale)
    rng = rng_for("calibration", phase, name, K)
    tallies, uppers = {}, {}
    nonpos = 0
    truth = pop.truth
    for _ in range(n_rep):
        res = replicate(pop, K, rng, n_boot)
        nonpos += int(res["H"]["denominator_nonpositive"])
        for tgt, r in res.items():
            if tgt == "V_delta_upper":
                for meth, ub in r["ub"].items():
                    uppers.setdefault(meth, UpperTally()).add(ub, truth["V_delta"])
                continue
            if tgt in ("C_benefit", "C_accuracy", "gap") and tgt not in truth:
                continue
            t = r["target"] if tgt.startswith("risk:") else truth[tgt]
            for meth, (lo, hi) in r["ci"].items():
                tallies.setdefault((tgt, meth), Tally()).add(lo, hi, t, r["est"])
            if tgt == "H":
                tallies.setdefault(("H", "fieller"), Tally()).add_set(*r["fieller"], t)
    out = {"scenario": name, "K": K, "n_rep": n_rep, "n_boot": n_boot, "phase": phase,
           "truth": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in truth.items()},
           "frac_V_theta_nonpositive": nonpos / n_rep,
           "intervals": [{"target": t, "method": m, **tl.summary()} for (t, m), tl in sorted(tallies.items())],
           "V_delta_upper": [{"method": m, **u.summary()} for m, u in sorted(uppers.items())]}
    return out
