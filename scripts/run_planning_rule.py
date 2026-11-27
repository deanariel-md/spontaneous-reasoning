r"""A conservative planning rule: the number
of prompt draws K for 80% benefit agreement chosen from the lower 80% bound of a four-draw pilot's R, against the
current rule, which uses R itself.

PowerShell (from the repository root):
    cmd /c ".venv\Scripts\python.exe scripts\run_planning_rule.py 2>&1"

The pilot's R equals ICC(3,1) of its benefits, R = (F - 1)/(F + K - 1), with F the ratio of the model and residual
mean squares of the (draws, models) layout (src/intervals.anova_mean_squares). Its one-sided lower 80% bound is
R_L = (F_L - 1)/(F_L + K - 1) with F_L = F / F_0.80(M - 1, (K - 1)(M - 1)), under Gaussian, equal-variance errors.
Each rule plans K = ceil(4.236 (1 - R)/R) (src/pilot_checks.chosen_k) and is unresolved when its R is not positive.

Settings, pilots and the agreement measurement are those of results/pilot_promise (scripts/run_pilot_checks.py):
the same 100 settings, the same 500 four-draw pilots per setting (same random streams), and the same achieved
agreement for a given K (enumerated bank on DOVE; disjoint groups of validation draws elsewhere, not measurable when
2K exceeds 24). The current rule therefore reproduces results/pilot_promise.

Outputs: results/planning_rule/planning_rule.json (with "macros", prefix pr), summary.md, summary_clean.md and
tables/settings.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import intervals, pilot_checks as pc, sampling  # noqa: E402

LOWER_LEVEL = 0.80
RULES = ("current", "lower")
RULE_WORDS = {"current": "K from the pilot's R", "lower": "K from the lower 80% bound of R"}


def load_checks():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_pilot_checks_for_planning", os.path.join(HERE, "run_pilot_checks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r_lower(G, level=LOWER_LEVEL):
    """One-sided lower `level` bound of the pilot's R from the F distribution of its mean-square ratio; None when R is
    undefined (no residual or no model variation). The bound is reported as computed, negative values included."""
    ms_a, ms_e, df1, df2 = intervals.anova_mean_squares(np.asarray(G, float))
    if not (np.isfinite(ms_a) and np.isfinite(ms_e)) or ms_e <= 0:
        return None
    K = np.asarray(G).shape[0]
    f_l = (ms_a / ms_e) / stats.f.ppf(level, df1, df2)
    return float((f_l - 1) / (f_l + K - 1))


def r_from_f(G):
    """R through the mean squares, (F - 1)/(F + K - 1); equals src/pilot_checks.pilot_r (checked in run())."""
    ms_a, ms_e, _, _ = intervals.anova_mean_squares(np.asarray(G, float))
    if ms_e <= 0:
        return None
    K = np.asarray(G).shape[0]
    F = ms_a / ms_e
    return float((F - 1) / (F + K - 1))


def compare_one(ck, setting, kind, arr):
    """The pilots of run_pilot_checks.promise_one, planned by both rules; achieved agreement per K from the same
    streams, so the current rule reproduces results/pilot_promise."""
    rng = sampling.rng_for(ck.SEED_LABEL, setting)
    if kind == "bank":
        G = arr["G"]
        idx = rng.integers(0, G.shape[0], size=(ck.N_PILOTS, ck.PILOT_DRAWS))
        pilots = [G[i] for i in idx]
    else:
        G = arr["Gd"]
        pilots = [G[np.sort(rng.permutation(G.shape[0])[:ck.PILOT_DRAWS])] for _ in range(ck.N_PILOTS)]
    R = np.array([pc.pilot_r(p) for p in pilots])
    RF = np.array([np.nan if (x := r_from_f(p)) is None else x for p in pilots])
    ok = np.isfinite(R) & np.isfinite(RF)
    if not np.allclose(R[ok], RF[ok], rtol=1e-9, atol=1e-12) or (np.isfinite(R) != np.isfinite(RF)).any():
        raise SystemExit(f"{setting}: R through the mean squares differs from pilot_r")
    RL = np.array([np.nan if (x := r_lower(p)) is None else x for p in pilots])
    ks = {"current": [pc.chosen_k(r) for r in R], "lower": [pc.chosen_k(r) for r in RL]}
    achieved = {}
    for k in sorted({k for v in ks.values() for k in v if k is not None}):
        krng = sampling.rng_for(ck.SEED_LABEL, setting, "achieved", k)
        achieved[k] = (pc.achieved_enumerated(arr["G"], k, ck.N_PAIRS, krng) if kind == "bank"
                       else pc.achieved_heldout(arr["Gv"], k, ck.N_SPLITS, krng))
    out = {"setting": setting, "kind": kind, "M": int(G.shape[1]), "n_pilots": len(pilots),
           "R_median": float(np.nanmedian(R)) if np.isfinite(R).any() else None,
           "R_lower_median": float(np.nanmedian(RL)) if np.isfinite(RL).any() else None,
           "R_lower_not_positive": int(np.sum(RL <= 0)), "achieved_by_K": {str(k): v for k, v in achieved.items()}}
    for rule in RULES:
        cls = [pc.classify(k, None if k is None else achieved[k]) for k in ks[rule]]
        kk = [k for k in ks[rule] if k is not None]
        n = len(cls)
        for c in ck.CLASSES:
            out[f"{rule}_share_{c.replace(' ', '_')}"] = cls.count(c) / n
        out[f"{rule}_K_median"] = float(np.median(kk)) if kk else None
        out[f"{rule}_K_mean"] = float(np.mean(kk)) if kk else None
        out[f"{rule}_K_p90"] = float(np.quantile(kk, 0.9)) if kk else None
        out[f"{rule}_K_max"] = int(max(kk)) if kk else None
        out[f"{rule}_resolved"] = len(kk)
        # pilots planning more draws than the 24 a Dataset 2 collection holds (share of all pilots, as the classes)
        out[f"{rule}_share_K_over_24"] = sum(k > 24 for k in kk) / n
    return out


def by_panel(ck, df):
    rows = []
    groups = list(df.groupby("panel", sort=False)) + [("all", df)]
    for panel, g in groups:
        x = {"panel": panel, "settings": int(len(g))}
        for rule in RULES:
            for c in ck.CLASSES:
                col = f"{rule}_share_{c.replace(' ', '_')}"
                x[f"{rule}_mean_share_{c.replace(' ', '_')}"] = float(g[col].mean())
            x[f"{rule}_settings_half_meet"] = int((g[f"{rule}_share_meets"] >= 0.5).sum())
            x[f"{rule}_mean_share_K_over_24"] = float(g[f"{rule}_share_K_over_24"].mean())
            # among the pilots that plan a K: the panel's mean meets share over its mean resolved share
            res = 1.0 - x[f"{rule}_mean_share_unresolved"]
            x[f"{rule}_meets_among_resolved"] = x[f"{rule}_mean_share_meets"] / res if res > 0 else None
            kp = pd.to_numeric(g[f"{rule}_K_p90"], errors="coerce")
            x[f"{rule}_K_p90_median"] = float(kp.median()) if kp.notna().any() else None
            kmed = pd.to_numeric(g[f"{rule}_K_median"], errors="coerce")
            x[f"{rule}_K_median_of_medians"] = float(kmed.median()) if kmed.notna().any() else None
            kmean = pd.to_numeric(g[f"{rule}_K_mean"], errors="coerce")
            x[f"{rule}_K_mean_of_means"] = float(kmean.mean()) if kmean.notna().any() else None
        rows.append(x)
    return rows


def fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}".replace("-", "−")


def big(x, tex=False):
    """A mean K as computed; above a million in powers of ten (a pilot with R just above zero plans an enormous K)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x) or abs(x) < 1e6:
        return fmt(x, 1)
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    return f"${mant:.1f}\\times 10^{{{e}}}$" if tex else f"{mant:.1f} × 10^{e}"


TAG = {"DOVE MMLU block": "DoveBlock", "DOVE benchmarks": "Dove", "GSM8K (19)": "Gsm", "GSM8K (29)": "GsmThirty",
       "BIG-Bench Hard (28), pooled": "Bbh", "BIG-Bench Hard tasks": "BbhTasks", "all": "All"}
RT = {"current": "Current", "lower": "Lower"}


def macros(by, check):
    m = {}
    for x in by:
        t = TAG[x["panel"]]
        m[f"pr{t}Settings"] = str(x["settings"])
        for rule in RULES:
            r = RT[rule]
            for c, ct in (("meets", "Meets"), ("short", "Short"), ("unresolved", "Unresolved"),
                          ("not_measurable", "NotMeasurable")):
                m[f"pr{t}{ct}{r}"] = fmt(100 * x[f"{rule}_mean_share_{c}"], 0)
            m[f"pr{t}HalfMeet{r}"] = str(x[f"{rule}_settings_half_meet"])
            m[f"pr{t}KMedian{r}"] = fmt(x[f"{rule}_K_median_of_medians"], 0)
            m[f"pr{t}KMean{r}"] = big(x[f"{rule}_K_mean_of_means"], tex=True)
            m[f"pr{t}KNinety{r}"] = fmt(x[f"{rule}_K_p90_median"], 0)
            m[f"pr{t}KOverTwentyFour{r}"] = fmt(100 * x[f"{rule}_mean_share_K_over_24"], 0)
            mr = x[f"{rule}_meets_among_resolved"]
            m[f"pr{t}MeetsResolved{r}"] = fmt(None if mr is None else 100 * mr, 0)
    m["prLevel"] = fmt(100 * LOWER_LEVEL, 0)
    m["prTarget"] = fmt(100 * pc.TARGET, 0)
    m["prReproducesPromise"] = "yes" if check["reproduces"] else "no"
    return {k: v.replace("−", "$-$") for k, v in m.items()}


def summary(out):
    L = ["# A conservative planning rule", "",
         "The same 500 four-draw pilots per setting as "
         "`results/pilot_promise`, planned twice: K for 80% benefit agreement from the pilot's R (the current rule), "
         "and from the lower 80% bound of R, R_L = (F_L − 1)/(F_L + K − 1) with F_L = F / F_0.80(M − 1, (K − 1)(M − 1)) "
         "(Gaussian, equal-variance errors). A pilot is unresolved when its R, or R_L, is not positive. Achieved "
         "agreement is measured as in `results/pilot_promise` (enumerated bank on DOVE, disjoint groups of validation "
         "draws elsewhere, not measurable when 2K exceeds 24). Shares are the mean over a panel's settings of each "
         "setting's share of its 500 pilots; unresolved pilots stay in the denominator.", "",
         f"The current rule reproduces `results/pilot_promise` in every setting: {out['check']['reproduces']}.", "",
         "| Panel | Settings | Rule | Meets | Short | Unresolved | Not measurable | Meets among resolved | "
         "Settings where ≥ half meet | Median K | 90th percentile K | K > 24 | Mean K |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in out["by_panel"]:
        for rule in RULES:
            mr = x[f"{rule}_meets_among_resolved"]
            L.append(f"| {x['panel']} | {x['settings']} | {RULE_WORDS[rule]} | "
                     f"{fmt(100 * x[f'{rule}_mean_share_meets'], 0)}% | {fmt(100 * x[f'{rule}_mean_share_short'], 0)}% | "
                     f"{fmt(100 * x[f'{rule}_mean_share_unresolved'], 0)}% | "
                     f"{fmt(100 * x[f'{rule}_mean_share_not_measurable'], 0)}% | "
                     f"{'--' if mr is None else fmt(100 * mr, 0) + '%'} | {x[f'{rule}_settings_half_meet']} | "
                     f"{fmt(x[f'{rule}_K_median_of_medians'], 0)} | {fmt(x[f'{rule}_K_p90_median'], 0)} | "
                     f"{fmt(100 * x[f'{rule}_mean_share_K_over_24'], 0)}% | {big(x[f'{rule}_K_mean_of_means'])} |")
    L += ["", "Meets among resolved: the panel's mean meets share over its mean share of pilots that plan a K. Median K "
          "is the median over the panel's settings of each setting's median planned K among resolved pilots; the 90th "
          "percentile likewise. K > 24: share of all pilots planning more draws than the 24 a Dataset 2 collection "
          "holds. Mean K is the mean over settings of each setting's mean, as computed; a pilot whose R lies just above "
          "zero plans an enormous K, which dominates it. A larger K costs that many more prompt draws per evaluation.", ""]
    return L


def run():
    t0 = time.time()
    ck = load_checks()
    rows_raw = ck.promise_settings()
    print(f"settings loaded: {len(rows_raw)} ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for setting, group, kind, arr in rows_raw:
        r = compare_one(ck, setting, kind, arr)
        r["group"], r["panel"] = group, ck.GROUP_PANEL[group]
        rows.append(r)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "achieved_by_K"} for r in rows])
    # the current rule must reproduce results/pilot_promise setting by setting
    ref = pd.read_csv(os.path.join(ROOT, "results", "pilot_promise", "tables", "settings.csv"))
    ref = ref.set_index("setting")
    diffs = []
    for _, r in df.iterrows():
        for c in ck.CLASSES:
            col = f"share_{c.replace(' ', '_')}"
            a, b = r[f"current_{col}"], ref.loc[r["setting"], col]
            if not np.isclose(a, b, atol=1e-12):
                diffs.append((r["setting"], c, a, b))
    check = {"reproduces": not diffs, "differences": diffs[:20], "settings_compared": int(len(df))}
    if diffs:
        print(f"WARNING: the current rule differs from results/pilot_promise in {len(diffs)} cells", flush=True)
    by = by_panel(ck, df)
    out = {"analysis": "conservative planning rule", "created": time.strftime("%Y-%m-%d %H:%M"),
           "lower_level": LOWER_LEVEL, "target": pc.TARGET, "n_pilots": ck.N_PILOTS, "pilot_draws": ck.PILOT_DRAWS,
           "seed_label": ck.SEED_LABEL, "check": check, "by_panel": by, "settings": rows}
    out["macros"] = macros(by, check)
    ck.write("planning_rule", out, summary(out), {"settings.csv": df})
    print(f"done in {time.time() - t0:.0f} s -> results/planning_rule", flush=True)
    return out


if __name__ == "__main__":
    run()
