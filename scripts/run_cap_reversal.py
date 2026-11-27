r"""Who gains more from chain of thought under each baseline: own-baseline axes, the shared requested-accuracy axis,
model size, release year, and w (exploratory).

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_cap_reversal.py

This turns the strongest results of the probes in results/cap_reversal_probe/ (README.md there; req_axis_check.py,
axis/run_axis_probe.py, reconcile/run_reconcile.py) into one tested analysis. Everything here was chosen after the
shared-axis result (results/cap_shared_axis) was seen: it is exploratory and labelled so.

Gains, per draw and model, in pp: Req = chain of thought - requested; Fd = chain of thought - forced; Corr = (1 - w) x Fd,
with w the model's share of requested replies over the forced arm's 16-token cap, averaged over the discovery draws
(results/baseline_mixture/tables/per_model.csv, w_discovery; src/baseline_mixture.corrected_gain).

Blocks (macro prefix crv):
  Own     the literature's practice, each gain on its own baseline's accuracy: Req on requested accuracy, Fd on forced
          accuracy, Diff = the first minus the second; Corr on requested accuracy; Share = (Corr - Fd) / Diff, the share
          of the difference that shrinking each forced gain by (1 - w) reproduces.
  ReqAcc  both gains on the same axis, requested accuracy: Req, Fd, Diff; the share of resamples with Req < 0 < Fd.
          The item-split check (x from one half of the questions, the gain from the other) is cited from
          results/cap_reversal_probe/axis/axis_probe.json, not recomputed.
  Size    both gains on log2 total parameters (manifests/panel_release_dates.json params_b): pp per doubling.
  Year    both gains on the release date in years since 1 January 2023 (same manifest): pp per year.
  W       Pearson correlation of w with the release year and with log2 parameters, over models.
Every slope is src/robustness.cross_slope (cross-draw moments of Appendix A, denominator M - 1), as in
scripts/run_cap_shared_axis.py. Intervals: 10,000 model resamples and 10,000 family resamples (whole families), one set
of resamples per benchmark and kind (src/sampling.rng_for), used for every statistic and both scorings, so differences
and shares are paired. Undefined resamples (nonpositive V_x, zero denominator) are counted and left out of the
percentiles, never repaired. Leave-one-family-out: the range over the five panels with one family removed.

Data: the 24 discovery draws and the panels of scripts/run_cap_shared_axis.py (GSM8K-29, MATH-500 29 models,
BIG-Bench Hard-28), strict and content scoring, read through run_cap_shared_axis.task_inputs.

Outputs: results/cap_reversal/cap_reversal.json (with macros), summary.md, summary_clean.md, macros.md,
tables/slopes.csv, tables/w_correlations.csv, tables/per_model.csv. Macros: paper/macros_cap_reversal.py.
"""
import json
import os
import re
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import robustness as rb, sampling  # noqa: E402
import robustness_inputs as ri  # noqa: E402
import run_cap_shared_axis as csa  # noqa: E402

OUT = os.path.join(ROOT, "results", "cap_reversal")
PROBE = os.path.join(ROOT, "results", "cap_reversal_probe")
N_BOOT = 10000
SEED_LABEL = "cap-reversal"
YEAR0 = date(2023, 1, 1)
TASKS = (("gsm8k", "Gsm"), ("math", "Math"), ("bbh", "Bbh"))
SCORINGS = (("strict", ""), ("content", "Content"))
BLOCKS = (("own", "Own", ("Req", "Fd", "Diff", "Corr", "Share")),
          ("reqacc", "ReqAcc", ("Req", "Fd", "Diff")),
          ("size", "Size", ("Req", "Fd", "Diff")),
          ("year", "Year", ("Req", "Fd", "Diff")))
DECIMALS = {"own": 3, "reqacc": 3, "size": 2, "year": 2}
UNITS = {"own": "pp per pp of the baseline's own accuracy", "reqacc": "pp per pp of requested accuracy",
         "size": "pp per doubling of total parameters", "year": "pp per year"}
BENCH = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard"}


# ---------------------------------------------------------------------------- inputs

def model_facts(panel):
    """log2 total parameters and release year (years since 1 Jan 2023) of each panel model."""
    e = json.load(open(os.path.join(ROOT, "manifests", "panel_release_dates.json"), encoding="utf-8"))["entries"]
    params = np.array([e[k]["params_b"] for k in panel], float)
    years = np.array([(date.fromisoformat(e[k]["date"]) - YEAR0).days / 365.25 for k in panel], float)
    return params, years, [e[k]["date"] for k in panel]


def w_discovery(task, panel):
    pm = pd.read_csv(os.path.join(ROOT, "results", "baseline_mixture", "tables", "per_model.csv"))
    return pm[pm["task"] == task].set_index("model_key").loc[list(panel), "w_discovery"].to_numpy(float)


# ---------------------------------------------------------------------------- statistics

def cs(X, Y):
    return rb.cross_slope(X, Y)[0]


def ratio(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")


def slope_stats(arms, w, lp, yrs):
    """stat(ix) -> {(block, gain): value} for the model index array ix (one scoring)."""
    R, F, C = (np.asarray(arms[a], float) for a in ("requested", "forced", "cot"))
    K = C.shape[0]
    Greq, Gfd = C - R, C - F
    Gcorr = (1 - np.asarray(w, float))[None, :] * Gfd
    LP, YR = np.tile(lp, (K, 1)), np.tile(yrs, (K, 1))

    def stat(ix):
        r, f, gq, gf, gc = R[:, ix], F[:, ix], Greq[:, ix], Gfd[:, ix], Gcorr[:, ix]
        s = {}
        s[("own", "Req")] = cs(r, gq)
        s[("own", "Fd")] = cs(f, gf)
        s[("own", "Diff")] = s[("own", "Req")] - s[("own", "Fd")]
        s[("own", "Corr")] = cs(r, gc)
        s[("own", "Share")] = ratio(s[("own", "Corr")] - s[("own", "Fd")], s[("own", "Diff")])
        s[("reqacc", "Req")] = s[("own", "Req")]
        s[("reqacc", "Fd")] = cs(r, gf)
        s[("reqacc", "Diff")] = s[("reqacc", "Req")] - s[("reqacc", "Fd")]
        for b, X in (("size", LP[:, ix]), ("year", YR[:, ix])):
            s[(b, "Req")] = cs(X, gq)
            s[(b, "Fd")] = cs(X, gf)
            s[(b, "Diff")] = s[(b, "Req")] - s[(b, "Fd")]
        return s
    return stat


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def w_stats(w, lp, yrs):
    def stat(ix):
        return {("w", "Year"): pearson(w[ix], yrs[ix]), ("w", "Size"): pearson(w[ix], lp[ix])}
    return stat


def summarise(vals):
    """Percentile summary as src/robustness.bootstrap_statistic, from a vector of resample values."""
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    v = vals[ok]
    if len(v) == 0:
        return {"n_boot": int(len(vals)), "n_undefined": int((~ok).sum()), "ci95": None, "share_below_zero": None}
    return {"n_boot": int(len(vals)), "n_undefined": int((~ok).sum()),
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))], "median": float(np.median(v)),
            "share_below_zero": float(np.mean(v < 0)), "share_above_zero": float(np.mean(v > 0))}


def opposite_share(req, fd):
    """Share of resamples (both slopes defined) with requested < 0 < forced, and the reverse."""
    req, fd = np.asarray(req, float), np.asarray(fd, float)
    ok = np.isfinite(req) & np.isfinite(fd)
    n = int(ok.sum())
    return {"n": n, "req_neg_fd_pos": float(np.mean((req[ok] < 0) & (fd[ok] > 0))) if n else None,
            "req_pos_fd_neg": float(np.mean((req[ok] > 0) & (fd[ok] < 0))) if n else None}


def boot_rng(task, kind):
    """The same label gives the same resamples: every statistic and both scorings of a benchmark share them."""
    return sampling.rng_for(SEED_LABEL, task, kind)


def evaluate(stat, M, families, idx_m, idx_f):
    """Point values, model and family resample summaries, leave-one-family-out, and the raw resample vectors."""
    fams = sorted(set(families))
    est = stat(np.arange(M))
    lofo = {fam: stat(np.array([k for k in range(M) if families[k] != fam], int)) for fam in fams}
    BM = [stat(ix) for ix in idx_m]
    BF = [stat(ix) for ix in idx_f]
    cells, raw = {}, {}
    for key, e in est.items():
        vm = np.array([b[key] for b in BM], float)
        vf = np.array([b[key] for b in BF], float)
        lv = {fam: lofo[fam][key] for fam in fams}
        arr = np.array(list(lv.values()), float)
        fin = arr[np.isfinite(arr)]
        cells[key] = {"estimate": float(e), "model_bootstrap": summarise(vm), "family_bootstrap": summarise(vf),
                      "leave_one_family_out": lv,
                      "lofo_min": float(fin.min()) if len(fin) else float("nan"),
                      "lofo_max": float(fin.max()) if len(fin) else float("nan"),
                      "lofo_undefined": int((~np.isfinite(arr)).sum())}
        cells[key]["model_bootstrap_sign"] = rb.interval_sign(cells[key]["model_bootstrap"]["ci95"])
        cells[key]["family_bootstrap_sign"] = rb.interval_sign(cells[key]["family_bootstrap"]["ci95"])
        raw[key] = (vm, vf)
    return cells, raw


def analyse_task(task, f, S):
    panel, families = list(f["panel"]), list(f["families"])
    M = len(panel)
    params, yrs, dates = model_facts(panel)
    lp = np.log2(params)
    w = w_discovery(task, panel)
    idx_m = rb.resample_columns(M, None, N_BOOT, boot_rng(task, "model"))
    idx_f = rb.resample_columns(M, families, N_BOOT, boot_rng(task, "family"))
    T = {"name": ri.NAMES[task], "M": M, "K": int(S["strict"]["cot"].shape[0]), "models": panel, "families": families,
         "params_b": params, "log2_params": lp, "release_date": dates, "years_since_2023": yrs, "w_discovery": w,
         "scorings": {}}
    for sc, _ in SCORINGS:
        cells, raw = evaluate(slope_stats(S[sc], w, lp, yrs), M, families, idx_m, idx_f)
        blocks = {}
        for (b, g), c in cells.items():
            blocks.setdefault(b, {})[g] = c
        vm_r, vf_r = raw[("reqacc", "Req")]
        vm_f, vf_f = raw[("reqacc", "Fd")]
        blocks["reqacc"]["opposite"] = {"model": opposite_share(vm_r, vm_f), "family": opposite_share(vf_r, vf_f)}
        arms = S[sc]
        per_model = []
        for j, k in enumerate(panel):
            gq = float((arms["cot"] - arms["requested"])[:, j].mean())
            gf = float((arms["cot"] - arms["forced"])[:, j].mean())
            per_model.append({"model": k, "family": families[j], "params_b": float(params[j]), "log2_params": float(lp[j]),
                              "release_date": dates[j], "years_since_2023": float(yrs[j]), "w_discovery": float(w[j]),
                              **{f"{a}_pp": float(arms[a][:, j].mean()) for a in ri.ARMS},
                              "gain_requested_pp": gq, "gain_forced_pp": gf, "gain_corrected_pp": float((1 - w[j]) * gf)})
        T["scorings"][sc] = {"blocks": blocks, "per_model": per_model}
    wc, _ = evaluate(w_stats(w, lp, yrs), M, families, idx_m, idx_f)
    T["w"] = {g: c for (_, g), c in wc.items()}
    T["w"]["mean"] = float(np.mean(w))
    T["w"]["r_size_year"] = pearson(lp, yrs)
    return T


# ---------------------------------------------------------------------------- the item-split check (cited)

def item_split():
    """Largest change of a requested-axis slope when x and the gain come from disjoint halves of the questions
    (results/cap_reversal_probe/axis/axis_probe.json: mean over 200 splits of the disattenuated split slope, paper
    pairing, against the slope on all questions); for the single gains and for their difference."""
    p = os.path.join(PROBE, "axis", "axis_probe.json")
    ap = json.load(open(p, encoding="utf-8"))
    rows = []
    for task, T in ap["tasks"].items():
        for sc, S in T["scorings"].items():
            sp = S["split"]
            d = {}
            for g in ("requested", "forced"):
                full = float(sp[f"paper|requested|{g}|same_full"])
                split = float(sp[f"paper|requested|{g}|split_dis"]["mean"])
                d[g] = (full, split)
                rows.append({"task": task, "scoring": sc, "gain": g, "all_questions": full, "split": split,
                             "abs_change": abs(split - full)})
            df, ds = d["requested"][0] - d["forced"][0], d["requested"][1] - d["forced"][1]
            rows.append({"task": task, "scoring": sc, "gain": "diff", "all_questions": df, "split": ds,
                         "abs_change": abs(ds - df)})
    single = max((r for r in rows if r["gain"] != "diff"), key=lambda r: r["abs_change"])
    diff = max((r for r in rows if r["gain"] == "diff"), key=lambda r: r["abs_change"])
    return {"source": "results/cap_reversal_probe/axis/axis_probe.json (axis/run_axis_probe.py, Q2: 200 random "
                      "splits of the questions, paper pairing, disattenuated split slope)",
            "n_splits": ap.get("n_splits"), "rows": rows, "max_single": single, "max_diff": diff}


# ---------------------------------------------------------------------------- checks against the probes

def checks(R):
    out = []

    def add(what, mine, saved, tol):
        d = abs(mine - saved)
        out.append({"check": what, "value": float(mine), "saved": float(saved), "abs_diff": float(d), "tol": tol,
                    "ok": bool(d <= tol)})

    mr = json.load(open(os.path.join(ROOT, "results", "model_resampling", "model_resampling.json"), encoding="utf-8"))
    for task, _ in TASKS:
        B = R["tasks"][task]["scorings"]["strict"]["blocks"]["own"]
        for g, key in (("Req", "CotMinusReq/full"), ("Fd", "CotMinusForced/full")):
            add(f"{task} strict, own-axis {g} slope vs results/model_resampling {key}", B[g]["estimate"],
                mr["tasks"][task]["blocks"][key]["slope"], 1e-9)
    # req_axis_check.out: the requested-accuracy axis, three decimals as printed
    pat = re.compile(r"^(\w+)\s+(\w+)\s+axis=requested\s+requested=(\S+) .*forced=(\S+) .*diff=(\S+) ")
    for line in open(os.path.join(PROBE, "req_axis_check.out"), encoding="utf-8"):
        m = pat.match(line)
        if not m:
            continue
        task, sc = m.group(1), m.group(2)
        B = R["tasks"][task]["scorings"][sc]["blocks"]["reqacc"]
        for g, txt in (("Req", m.group(3)), ("Fd", m.group(4)), ("Diff", m.group(5))):
            add(f"{task} {sc}, requested-axis {g} vs req_axis_check.out ({txt})", B[g]["estimate"], float(txt), 0.0005)
    # reconcile: pooled estimates of tables/slopes.csv (show.out prints them to three decimals)
    rc = pd.read_csv(os.path.join(PROBE, "reconcile", "tables", "slopes.csv"))
    rc = rc[(rc["grouping"] == "pooled") & (rc["variant"] == "alone")]
    get = {(r.task, r.scoring, r.axis, r.outcome): r.estimate for r in rc.itertuples()}
    for task, _ in TASKS:
        for sc, _ in SCORINGS:
            Bl = R["tasks"][task]["scorings"][sc]["blocks"]
            for ax, blk in (("OWN", "own"), ("LP", "size"), ("DATE", "year")):
                for out_, g in (("req", "Req"), ("fd", "Fd"), ("diff", "Diff")) + ((("corr", "Corr"),) if ax == "OWN" else ()):
                    add(f"{task} {sc}, {blk} {g} vs reconcile pooled {ax} {out_}", Bl[blk][g]["estimate"],
                        get[(task, sc, ax, out_)], 1e-8)
            share = get[(task, sc, "OWN", "corr_minus_fd")] / get[(task, sc, "OWN", "diff")]
            add(f"{task} {sc}, own Share vs reconcile corrected share", Bl["own"]["Share"]["estimate"], share, 1e-8)
    desc = pd.read_csv(os.path.join(PROBE, "reconcile", "tables", "descriptives.csv"))
    for task, _ in TASKS:
        d = desc[(desc["task"] == task) & (desc["scoring"] == "strict")].iloc[0]
        W = R["tasks"][task]["w"]
        add(f"{task}, r(w, year) vs reconcile descriptives", W["Year"]["estimate"], d["r_w_years"], 1e-9)
        add(f"{task}, r(w, log2 params) vs reconcile descriptives", W["Size"]["estimate"], d["r_w_log2params"], 1e-9)
    # cap_shared_axis: the forced gain on forced accuracy is the same cell there
    cs_json = os.path.join(ROOT, "results", "cap_shared_axis", "cap_shared_axis.json")
    if os.path.exists(cs_json):
        C = json.load(open(cs_json, encoding="utf-8"))
        for task, _ in TASKS:
            for sc, _ in SCORINGS:
                add(f"{task} {sc}, own Fd vs cap_shared_axis forced-axis forced gain",
                    R["tasks"][task]["scorings"][sc]["blocks"]["own"]["Fd"]["estimate"],
                    C["tasks"][task]["scorings"][sc]["axes"]["forced"]["gains"]["forced"]["slope"], 1e-9)
    return out


# ---------------------------------------------------------------------------- macros

def fx(x, nd):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}".replace("-", "$-$")


def pct(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{100 * x:.0f}"
    return "0" if s == "-0" else s.replace("-", "$-$")


def cell_macros(m, name, c, fmt):
    m[name] = fmt(c["estimate"])
    mb = c["model_bootstrap"]["ci95"] or [None, None]
    fb = c["family_bootstrap"]["ci95"] or [None, None]
    m[name + "Lo"], m[name + "Hi"] = fmt(mb[0]), fmt(mb[1])
    m[name + "FamLo"], m[name + "FamHi"] = fmt(fb[0]), fmt(fb[1])
    m[name + "LofoLo"], m[name + "LofoHi"] = fmt(c["lofo_min"]), fmt(c["lofo_max"])


def macros(R):
    m = {"crvNBoot": f"{N_BOOT:,}".replace(",", "{,}")}
    shares = []
    for task, tt in TASKS:
        T = R["tasks"][task]
        m[f"crvModels{tt}"] = str(T["M"])
        m[f"crvDraws{tt}"] = str(T["K"])
        for sc, st in SCORINGS:
            Bl = T["scorings"][sc]["blocks"]
            for b, bt, gains in BLOCKS:
                for g in gains:
                    fmt = pct if g == "Share" else (lambda x, nd=DECIMALS[b]: fx(x, nd))
                    cell_macros(m, f"crv{tt}{bt}{g}{st}", Bl[b][g], fmt)
            op = Bl["reqacc"]["opposite"]
            m[f"crv{tt}ReqAccOpp{st}"] = pct(op["model"]["req_neg_fd_pos"])
            m[f"crv{tt}ReqAccOpp{st}Fam"] = pct(op["family"]["req_neg_fd_pos"])
            shares.append((sc, Bl["own"]["Share"]["estimate"]))
        for g in ("Year", "Size"):
            cell_macros(m, f"crv{tt}W{g}", T["w"][g], lambda x: fx(x, 2))
    allsh = [v for _, v in shares]
    strict = [v for sc, v in shares if sc == "strict"]
    m["crvOwnShareMin"], m["crvOwnShareMax"] = pct(min(allsh)), pct(max(allsh))
    m["crvOwnShareStrictMin"], m["crvOwnShareStrictMax"] = pct(min(strict)), pct(max(strict))
    m["crvItemSplitMax"] = fx(R["item_split"]["max_single"]["abs_change"], 3)
    m["crvItemSplitMaxDiff"] = fx(R["item_split"]["max_diff"]["abs_change"], 3)
    for k in m:
        if not re.fullmatch(r"crv[A-Za-z]+", k):
            raise ValueError(f"macro name {k!r} is not crv followed by letters")
    return m


# ---------------------------------------------------------------------------- tables and summary

def s3(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}".replace("-", "−")


def ci(c, nd=3):
    return "undefined" if c is None else f"[{s3(c[0], nd)}, {s3(c[1], nd)}]"


def slope_rows(R):
    rows = []
    for task, _ in TASKS:
        T = R["tasks"][task]
        for sc, _ in SCORINGS:
            for b, _, gains in BLOCKS:
                for g in gains:
                    c = T["scorings"][sc]["blocks"][b][g]
                    mb, fb = c["model_bootstrap"], c["family_bootstrap"]
                    rows.append({"task": task, "scoring": sc, "block": b, "gain": g, "M": T["M"],
                                 "units": "proportion" if g == "Share" else UNITS[b], "estimate": c["estimate"],
                                 "model_lo": (mb["ci95"] or [None, None])[0], "model_hi": (mb["ci95"] or [None, None])[1],
                                 "model_share_below_zero": mb["share_below_zero"], "model_undefined": mb["n_undefined"],
                                 "family_lo": (fb["ci95"] or [None, None])[0], "family_hi": (fb["ci95"] or [None, None])[1],
                                 "family_share_below_zero": fb["share_below_zero"], "family_undefined": fb["n_undefined"],
                                 "lofo_min": c["lofo_min"], "lofo_max": c["lofo_max"],
                                 "lofo_undefined": c["lofo_undefined"]})
    return rows


def w_rows(R):
    rows = []
    for task, _ in TASKS:
        W = R["tasks"][task]["w"]
        for g, name in (("Year", "release year"), ("Size", "log2 parameters")):
            c = W[g]
            rows.append({"task": task, "with": name, "r": c["estimate"],
                         "model_lo": (c["model_bootstrap"]["ci95"] or [None, None])[0],
                         "model_hi": (c["model_bootstrap"]["ci95"] or [None, None])[1],
                         "family_lo": (c["family_bootstrap"]["ci95"] or [None, None])[0],
                         "family_hi": (c["family_bootstrap"]["ci95"] or [None, None])[1],
                         "model_undefined": c["model_bootstrap"]["n_undefined"],
                         "family_undefined": c["family_bootstrap"]["n_undefined"],
                         "lofo_min": c["lofo_min"], "lofo_max": c["lofo_max"], "mean_w": W["mean"]})
    return rows


def pct_ci(c):
    return "undefined" if c is None else f"[{100 * c[0]:.0f}, {100 * c[1]:.0f}]".replace("-", "−")


def excl(c):
    return c is not None and (c[0] > 0 or c[1] < 0)


def both_excl(c):
    return excl(c["model_bootstrap"]["ci95"]) and excl(c["family_bootstrap"]["ci95"])


def full(c, nd=3):
    return (f"{s3(c['estimate'], nd)} (models {ci(c['model_bootstrap']['ci95'], nd)}, families "
            f"{ci(c['family_bootstrap']['ci95'], nd)})")


def lead(R):
    L = []
    g = lambda t, sc, b, k: R["tasks"][t]["scorings"][sc]["blocks"][b][k]  # noqa: E731
    own = [(t, sc) for t, _ in TASKS for sc, _ in SCORINGS]
    neg_own = [x for x in own if g(*x, "own", "Diff")["estimate"] < 0]
    ex_own = [x for x in own if both_excl(g(*x, "own", "Diff")) and g(*x, "own", "Diff")["estimate"] < 0]
    L.append(f"Own-baseline axes (each gain on its own baseline's accuracy): the requested gain's slope is below the "
             f"forced gain's in {len(neg_own)} of 6 cells (3 benchmarks × 2 scorings); both intervals of the difference "
             f"exclude zero in {len(ex_own)}: " + "; ".join(f"{BENCH[t]} {sc} {full(g(t, sc, 'own', 'Diff'))}"
                                                           for t, sc in own) + ".")
    L.append("GSM8K strict on its own axes: requested " + full(g("gsm8k", "strict", "own", "Req")) + ", forced "
             + full(g("gsm8k", "strict", "own", "Fd")) + ".")
    sh = [g(t, sc, "own", "Share")["estimate"] for t, sc in own]
    L.append(f"Shrinking each forced gain by (1 − w) and regressing it on requested accuracy reproduces "
             f"{100 * min(sh):.0f}% to {100 * max(sh):.0f}% of the own-axis difference ({100 * min(sh[::2]):.0f}% to "
             f"{100 * max(sh[::2]):.0f}% under strict scoring): "
             + "; ".join(f"{BENCH[t]} {sc} {100 * g(t, sc, 'own', 'Share')['estimate']:.0f}% (models "
                         f"{pct_ci(g(t, sc, 'own', 'Share')['model_bootstrap']['ci95'])}%)"
                         for t, sc in own) + ".")
    ex_rq = [x for x in own if both_excl(g(*x, "reqacc", "Diff")) and g(*x, "reqacc", "Diff")["estimate"] < 0]
    L.append(f"Shared requested-accuracy axis: the difference (requested − forced) is negative with both intervals "
             f"below zero in {len(ex_rq)} of 6 cells: " + "; ".join(f"{BENCH[t]} {sc} {full(g(t, sc, 'reqacc', 'Diff'))}"
                                                                 for t, sc in own) + ".")
    opp = [(t, sc) for t, sc in own if g(t, sc, "reqacc", "Req")["estimate"] < 0 < g(t, sc, "reqacc", "Fd")["estimate"]]
    L.append("Opposite point signs (requested < 0 < forced) on that axis: " +
             ("; ".join(f"{BENCH[t]} {sc} (requested {full(g(t, sc, 'reqacc', 'Req'))}, forced "
                        f"{full(g(t, sc, 'reqacc', 'Fd'))}; share of resamples with requested < 0 < forced: models "
                        f"{100 * g(t, sc, 'reqacc', 'opposite')['model']['req_neg_fd_pos']:.0f}%, families "
                        f"{100 * g(t, sc, 'reqacc', 'opposite')['family']['req_neg_fd_pos']:.0f}%)" for t, sc in opp)
              if opp else "none") + ".")
    it = R["item_split"]
    L.append(f"Splitting the questions between axis and gain changes a requested-axis slope by at most "
             f"{it['max_single']['abs_change']:.3f} and the difference by at most {it['max_diff']['abs_change']:.4f} "
             f"(cited from the axis probe).")
    for b, nm, nd in (("size", "log2 parameters (pp per doubling)", 2), ("year", "release year (pp per year)", 2)):
        L.append(f"On {nm}: " + "; ".join(
            f"{BENCH[t]} {sc} requested {s3(g(t, sc, b, 'Req')['estimate'], nd)}, forced "
            f"{s3(g(t, sc, b, 'Fd')['estimate'], nd)}, difference {full(g(t, sc, b, 'Diff'), nd)}" for t, sc in own) + ".")
    L.append("w against release year and size (Pearson r over models): " + "; ".join(
        f"{BENCH[t]} year {full(R['tasks'][t]['w']['Year'], 2)}, size {full(R['tasks'][t]['w']['Size'], 2)}"
        for t, _ in TASKS) + ".")
    return L


def summary(R):
    L = ["# Who gains more from chain of thought under each baseline: own axes, the requested-accuracy axis, size, "
         "release year (exploratory)", "",
         "**Exploratory.** Chosen after the shared-axis "
         "result (`results/cap_shared_axis/`) had been seen. It turns the strongest results of the probes in "
         "`results/cap_reversal_probe/` into one analysis. Every number below is exploratory. Code "
         "`scripts/run_cap_reversal.py`; macros `paper/macros_cap_reversal.py` (prefix `crv`, list in `macros.md`); "
         "tests `tests/test_cap_reversal.py`.", "",
         "## Answer (exploratory)", ""] + ["- " + s for s in lead(R)] + [""]
    L += ["## Method", "",
          "- Panels and draws: GSM8K-29, MATH-500 (29 models, 398 numeric items), BIG-Bench Hard-28; the 24 discovery "
          "draws; strict and content scoring; the arrays of `scripts/run_cap_shared_axis.py` (`task_inputs`).",
          "- Gains in pp: Req = CoT − requested, Fd = CoT − forced, Corr = (1 − w) × Fd, with w the model's share of "
          "requested replies over 16 tokens on the discovery draws (`results/baseline_mixture/tables/per_model.csv`).",
          "- Own: Req on requested accuracy, Fd on forced accuracy (the literature's practice: each gain on its own "
          "baseline), Diff = Req − Fd, Corr on requested accuracy, Share = (Corr − Fd) / Diff.",
          "- ReqAcc: Req and Fd both on requested accuracy, Diff = Req − Fd (also the slope of forced − requested "
          "accuracy on requested accuracy); share of resamples with Req < 0 < Fd.",
          "- Size: log2 total parameters; Year: years since 1 January 2023 of the repository creation date "
          "(`manifests/panel_release_dates.json`). Both constant over draws.",
          "- W: Pearson r of w with year and with log2 parameters over the panel's models.",
          f"- Slopes: `src/robustness.cross_slope` (cross-draw moments, Appendix A). Intervals: {N_BOOT:,} model "
          f"resamples and {N_BOOT:,} family resamples (five families), percentile 95%, one set per benchmark and kind "
          "shared by every statistic and both scorings (paired). Undefined resamples are counted and left out. Leave one "
          "family out: the range over the five reduced panels.",
          "- Item split (cited, not recomputed): `results/cap_reversal_probe/axis/axis_probe.json`, 200 random halvings "
          "of the questions, x from one half and the gain from the other, disattenuated, paper pairing.", ""]
    for sc, _ in SCORINGS:
        L += [f"## Slopes, {sc} scoring", "",
              "| Benchmark | Block | Gain | Estimate | Model 95% | Family 95% | Leave one family out | Units |",
              "|---|---|---|---:|---|---|---|---|"]
        for task, _ in TASKS:
            for b, bt, gains in BLOCKS:
                for g in gains:
                    c = R["tasks"][task]["scorings"][sc]["blocks"][b][g]
                    nd = DECIMALS[b]
                    L.append(f"| {BENCH[task]} | {bt} | {g} | {s3(c['estimate'], nd)} | "
                             f"{ci(c['model_bootstrap']['ci95'], nd)} | {ci(c['family_bootstrap']['ci95'], nd)} | "
                             f"{s3(c['lofo_min'], nd)} to {s3(c['lofo_max'], nd)} | "
                             f"{'proportion' if g == 'Share' else UNITS[b]} |")
        L.append("")
        L += [f"Share of resamples with requested < 0 < forced on requested accuracy ({sc}):", ""]
        for task, _ in TASKS:
            op = R["tasks"][task]["scorings"][sc]["blocks"]["reqacc"]["opposite"]
            L.append(f"- {BENCH[task]}: models {op['model']['req_neg_fd_pos']:.3f}, families "
                     f"{op['family']['req_neg_fd_pos']:.3f} (the reverse, requested > 0 > forced: "
                     f"{op['model']['req_pos_fd_neg']:.3f}, {op['family']['req_pos_fd_neg']:.3f}).")
        L.append("")
    L += ["## w against release year and size", "",
          "| Benchmark | With | r | Model 95% | Family 95% | Leave one family out | Mean w |", "|---|---|---:|---|---|---|---:|"]
    for r in w_rows(R):
        L.append(f"| {BENCH[r['task']]} | {r['with']} | {s3(r['r'], 2)} | {ci([r['model_lo'], r['model_hi']], 2)} | "
                 f"{ci([r['family_lo'], r['family_hi']], 2)} | {s3(r['lofo_min'], 2)} to {s3(r['lofo_max'], 2)} | "
                 f"{r['mean_w']:.3f} |")
    it = R["item_split"]
    L += ["", "## Item split (cited)", "",
          f"Source: {it['source']}. Largest change of a single requested-axis slope: {it['max_single']['abs_change']:.4f} "
          f"({BENCH[it['max_single']['task']]} {it['max_single']['scoring']}, {it['max_single']['gain']} gain); of the "
          f"difference: {it['max_diff']['abs_change']:.4f} ({BENCH[it['max_diff']['task']]} {it['max_diff']['scoring']}).",
          "", "## Checks", ""]
    for c in R["checks"]:
        L.append(f"- {c['check']}: {c['value']:.6f} against {c['saved']:.6f} (difference {c['abs_diff']:.1e}; "
                 f"{'ok' if c['ok'] else 'NOT within ' + format(c['tol'], 'g')}).")
    L += ["", "## Undefined resamples", ""]
    und = [r for r in slope_rows(R) if r["model_undefined"] or r["family_undefined"] or r["lofo_undefined"]]
    und += [r for r in w_rows(R) if r["model_undefined"] or r["family_undefined"]]
    if not und:
        L.append("None.")
    for r in und:
        L.append(f"- {r['task']} {r.get('scoring', '')} {r.get('block', 'w')} {r.get('gain', r.get('with'))}: "
                 f"{r['model_undefined']} model, {r['family_undefined']} family"
                 + (f", {r['lofo_undefined']} leave-one-family-out" if r.get("lofo_undefined") else "") + ".")
    L += ["", "## Files", "",
          "- `cap_reversal.json` (every number, resample summaries, leave-one-family-out values, checks, macros); "
          "`tables/slopes.csv`; `tables/w_correlations.csv`; `tables/per_model.csv` (model means per arm, gains, "
          "corrected gain, w, size, date); `macros.md`."]
    return L


# ---------------------------------------------------------------------------- run

def run():
    t0 = time.time()
    R = {"analysis": "who gains more from chain of thought under each baseline: own-baseline axes, the requested-"
                     "accuracy axis, size, release year, w (exploratory)",
         "post_hoc": True, "created": time.strftime("%Y-%m-%d %H:%M"), "n_boot": N_BOOT, "seed_label": SEED_LABEL,
         "estimator": "src/robustness.cross_slope (cross-draw moments, Appendix A)", "units": UNITS, "tasks": {}}
    for task, _ in TASKS:
        f, S = csa.task_inputs(task)
        R["tasks"][task] = analyse_task(task, f, S)
        print(f"{task}: done ({time.time() - t0:.0f} s)", flush=True)
    R["item_split"] = item_split()
    R["checks"] = checks(R)
    R["checks_failed"] = [c["check"] for c in R["checks"] if not c["ok"]]
    R["runtime_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    pd.DataFrame(slope_rows(R)).to_csv(os.path.join(OUT, "tables", "slopes.csv"), index=False)
    pd.DataFrame(w_rows(R)).to_csv(os.path.join(OUT, "tables", "w_correlations.csv"), index=False)
    pd.DataFrame([{"task": t, "scoring": sc, **r} for t, _ in TASKS for sc, _ in SCORINGS
                  for r in R["tasks"][t]["scorings"][sc]["per_model"]]).to_csv(
        os.path.join(OUT, "tables", "per_model.csv"), index=False)
    m = macros(R)
    ri.write_outputs(OUT, R, summary(R), macros=m)
    with open(os.path.join(OUT, "macros.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Macros of results/cap_reversal ({len(m)}), as paper/macros_cap_reversal.py returns them (exploratory)\n\n")
        fh.writelines(f"- `\\{k}` = {v}\n" for k, v in m.items())
    for c in R["checks"]:
        if not c["ok"]:
            print("CHECK FAILED:", c, flush=True)
    print(f"done in {R['runtime_seconds']} s -> results/cap_reversal ({len(m)} macros; "
          f"{len(R['checks'])} checks, {len(R['checks_failed'])} failed)", flush=True)
    return R


if __name__ == "__main__":
    run()
