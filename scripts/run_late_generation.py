r"""Late direct answers against release date.
Run from the repository root, after scripts/run_index_checks.py:
    python scripts/run_late_generation.py
GSM8K-19, 24 discovery draws. Per model, the late share of 7.6 (strictly correct direct answers whose boxed answer
ends after token 64), from the cached token table of scripts/run_instability.py; the release date is the creation
date of the checkpoint's Hugging Face repository (`created` in manifests/d2_panel.json). Statistics: the Spearman
correlation of late share and date across the 19 models, the partial Spearman controlling for baseline accuracy, and
the same Spearman within each family of at least three models. Intervals: leave-one-draw-out jackknife t, the shares
and baselines recomputed from 23 draws, the dates fixed (prompt-draw variation for this panel); for the across-model
statistics also a leave-one-model-out jackknife t interval over the 19 models (which models are in the panel). Within
families only the sign is read. Descriptive; no threshold. Outputs:
results/late_generation/late_generation.json (macros, prefix lateGen), summary.md, summary_clean.md, table.csv.
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import bbh_d2, capability_checks as cc  # noqa: E402
from run_index_checks import CONFIG, arrays, fmt, write  # noqa: E402

RES = os.path.join(ROOT, "results")
PANEL = os.path.join(ROOT, "manifests", "d2_panel.json")
SOURCE = ("Hugging Face repository creation date (createdAt), from manifests/external/hf_metadata_20261030.json "
          "via manifests/d2_panel.json")
MIN_FAMILY = 3


def ranks(v):
    from scipy.stats import rankdata
    return rankdata(v)


def spearman(a, b):
    """Spearman correlation with average ranks for ties; None when either variable is constant."""
    ra, rb = ranks(a), ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(s, d, b):
    """Spearman of s and d controlling for b: (r_sd - r_sb r_db) / sqrt((1 - r_sb^2)(1 - r_db^2)); None when undefined."""
    r_sd, r_sb, r_db = spearman(s, d), spearman(s, b), spearman(d, b)
    if None in (r_sd, r_sb, r_db):
        return None
    den = (1 - r_sb ** 2) * (1 - r_db ** 2)
    if den <= 0:
        return None
    return float((r_sd - r_sb * r_db) / np.sqrt(den))


def late_counts(T, units, panel, cap):
    """(C, L), each (K, M): strictly correct direct answers and those ending after token `cap`, per draw and model."""
    t = T.assign(late=((T["strict_correct"] == 1) & (T["answer_token_position"] > cap)).astype(int))
    agg = t.groupby(["model_key", "config_id"]).agg(correct=("strict_correct", "sum"), late=("late", "sum"))
    C = np.array([[agg.loc[(m, str(u)), "correct"] for m in panel] for u in units], float)
    L = np.array([[agg.loc[(m, str(u)), "late"] for m in panel] for u in units], float)
    return C, L


ACROSS = ("spearman", "partial", "baseline_date", "share_baseline")


def statistics(C, L, B, days, families, keep, models=None):
    """Every statistic on the draws in `keep` and, when given, the models in `models` (then the across-model ones only)."""
    c, lt, b = C[keep].sum(0), L[keep].sum(0), B[keep].mean(0)
    s = lt / c
    if models is not None:
        s, b, d = s[models], b[models], days[models]
        return s, b, {"spearman": spearman(s, d), "partial": partial_spearman(s, d, b),
                      "baseline_date": spearman(b, d), "share_baseline": spearman(s, b)}
    out = {"spearman": spearman(s, days), "partial": partial_spearman(s, days, b),
           "baseline_date": spearman(b, days), "share_baseline": spearman(s, b)}
    for f in sorted(set(families)):
        ix = [i for i, y in enumerate(families) if y == f]
        if len(ix) >= MIN_FAMILY:
            out[f"family:{f}"] = spearman(s[ix], days[ix])
    return s, b, out


def jack(est, loo):
    if est is None or any(v is None for v in loo):
        return {"estimate": est, "jackknife_se": None, "ci95": None, "undefined_replicates": sum(v is None for v in loo)}
    r = cc._jack(est, loo)
    r["sign"] = "above zero" if r["ci95"][0] > 0 else "below zero" if r["ci95"][1] < 0 else "includes zero"
    return r


def run():
    t0 = time.time()
    import run_instability as ri
    from src import scoring
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    G19 = ri.load_gsm8k()
    panel, families = list(G19["panel"]), list(G19["families"])
    s19 = {"panel": panel, "items_by_task": {"gsm8k": G19["items"]["item_id"].tolist()}, "G": G19}
    disc = list(G19["sampler"]["discovery_config_ids"])
    B, _ = arrays(s19, disc)
    tok = ri.local_tokenizers(G19["reg"], panel, cfg["variants"]["tokenizer_files"])
    if tok is None:
        raise SystemExit("tokenizer files missing: the token table of 7.6 cannot be read")
    spec = scoring.spec_from_dict(json.load(open(os.path.join(ROOT, "manifests", "gsm8k_scorer.json")))["format_spec"])
    T = ri.token_table(G19, sorted(set(disc)), tok, spec)
    cap = int(cfg["variants"]["direct_answer_cap_tokens"])
    C, L = late_counts(T, disc, panel, cap)
    if (C <= 0).any():
        raise SystemExit("a model has no strictly correct direct answer in a draw: the share is undefined there")

    # The shares must be those of 7.6.
    ref = pd.read_csv(os.path.join(RES, "late_answers", "scatter.csv")).set_index("model_key")
    s_all = L.sum(0) / C.sum(0)
    if list(ref.index) != panel or np.max(np.abs(ref["share_late"].to_numpy() - s_all)) > 1e-12:
        raise SystemExit("the late shares differ from results/late_answers/scatter.csv")
    if np.max(np.abs(ref["baseline"].to_numpy() - B.mean(0))) > 1e-9:
        raise SystemExit("the baselines differ from results/late_answers/scatter.csv")

    reg = bbh_d2.load_registries()
    man = {e["repo"]: e for e in json.load(open(PANEL, encoding="utf-8"))["panel"]}
    repos = [reg[k]["hf_id"] for k in panel]
    missing = [r for r in repos if r not in man or not man[r].get("created")]
    if missing:
        raise SystemExit(f"no creation date in manifests/d2_panel.json for {missing}")
    dates = [man[r]["created"] for r in repos]
    days = np.array([(dt.date.fromisoformat(d) - dt.date(1970, 1, 1)).days for d in dates], float)
    if [man[r]["family"] for r in repos] != families:
        raise SystemExit("the families differ between the panel manifest and the GSM8K-19 inputs")

    K = len(disc)
    s, b, est = statistics(C, L, B, days, families, np.ones(K, bool))
    loo = [statistics(C, L, B, days, families, np.arange(K) != u)[2] for u in range(K)]
    stats = {k: jack(v, [r[k] for r in loo]) for k, v in est.items()}
    M = len(panel)
    lomo = [statistics(C, L, B, days, families, np.ones(K, bool), np.arange(M) != i)[2] for i in range(M)]
    for k in ACROSS:
        stats[k]["models"] = jack(est[k], [r[k] for r in lomo])
    fam = {k.split(":", 1)[1]: v for k, v in stats.items() if k.startswith("family:")}
    fam_n = {f: sum(y == f for y in families) for f in fam}
    ties_date = int(len(days) - len(set(days)))
    ties_share = int(len(s) - len(set(np.round(s, 12))))

    rows = [{"model_key": k, "family": f, "hf_id": r, "release_date": d, "date_source": SOURCE,
             "share_late": float(x), "baseline": float(y)} for k, f, r, d, x, y in zip(panel, families, repos, dates, s, b)]
    os.makedirs(os.path.join(RES, "late_generation"), exist_ok=True)
    pd.DataFrame(rows).sort_values(["release_date", "model_key"]).to_csv(
        os.path.join(RES, "late_generation", "table.csv"), index=False)

    R = {"analysis": "late direct answers against release date",
         "methods": "late share of strictly correct direct answers, per model, against release date", "created": time.strftime("%Y-%m-%d %H:%M"),
         "setting": "GSM8K-19, discovery draws", "M": len(panel), "K": K, "cap_tokens": cap,
         "date_rule": SOURCE, "ties": {"date": ties_date, "share": ties_share},
         "statistics": stats, "families": {f: {"M": fam_n[f], **fam[f]} for f in fam}, "table": rows}
    sp, pa = stats["spearman"], stats["partial"]
    pos = sum(v["estimate"] is not None and v["estimate"] > 0 for v in fam.values())
    m = {"lateGenM": str(len(panel)), "lateGenSpearman": fmt(sp["estimate"], 2),
         "lateGenFamPos": str(pos), "lateGenFamN": str(len(fam))}
    for key, st in (("lateGenSpearman", sp), ("lateGenPartial", pa)):
        if st["estimate"] is None:
            m[key] = "undefined"
            continue
        m[key] = fmt(st["estimate"], 2)
        m[f"{key}Lo"], m[f"{key}Hi"] = (fmt(st["ci95"][0], 2), fmt(st["ci95"][1], 2)) if st["ci95"] else ("--", "--")
        mo = st["models"]
        m[f"{key}ModelsLo"], m[f"{key}ModelsHi"] = ((fmt(mo["ci95"][0], 2), fmt(mo["ci95"][1], 2)) if mo["ci95"]
                                                    else ("--", "--"))
    m["lateGenBaselineDate"] = fmt(stats["baseline_date"]["estimate"], 2)
    m["lateGenShareBaseline"] = fmt(stats["share_baseline"]["estimate"], 2)
    R["macros"] = m
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write("late_generation", R, summary(R))
    print(f"done in {R['runtime_seconds']} s -> results/late_generation", flush=True)


def ci(st, d=2):
    if st["estimate"] is None:
        return "undefined"
    if not st["ci95"]:
        return f"{fmt(st['estimate'], d)} (no interval: {st['undefined_replicates']} replicates undefined)"
    return f"{fmt(st['estimate'], d)} [{fmt(st['ci95'][0], d)}, {fmt(st['ci95'][1], d)}]"


def iv_txt(st, d=2):
    if st["estimate"] is None:
        return "undefined"
    if not st["ci95"]:
        return f"no interval ({st['undefined_replicates']} replicates undefined)"
    return f"[{fmt(st['ci95'][0], d)}, {fmt(st['ci95'][1], d)}]"


def summary(R):
    S = R["statistics"]
    L = ["# Late direct answers against release date", "",
         f"Setting: {R['setting']}; {R['M']} models, {R['K']} draws. Late share: strictly "
         f"correct direct answers whose boxed answer ends after token {R['cap_tokens']} (7.6). Release date: "
         f"{R['date_rule']}. Descriptive; no threshold.", "",
         "Two 95% jackknife t intervals. Over draws: each replicate drops one of the 24 draws, with the panel and "
         "dates fixed, so it shows prompt-draw variation for these 19 models. Over models: each replicate drops one "
         "of the 19 models, so it shows how much the value rests on which models are in the panel.", "",
         "| Statistic | Estimate | Over draws | Over models |", "|---|---:|---|---|"]
    for k, name in (("spearman", "Spearman, late share and release date"),
                    ("partial", "Partial Spearman, controlling for baseline accuracy"),
                    ("baseline_date", "Spearman, baseline accuracy and release date"),
                    ("share_baseline", "Spearman, late share and baseline accuracy")):
        L.append(f"| {name} | {fmt(S[k]['estimate'], 2)} | {iv_txt(S[k])} | {iv_txt(S[k]['models'])} |")
    L += ["",
         f"Ties: {R['ties']['date']} in release date (average ranks), {R['ties']['share']} in late share.", "",
         f"## Within families of at least {MIN_FAMILY} models", "",
         "The sign of the Spearman correlation of late share and release date within each family. With three or four "
         "models the ranks rarely change between draw replicates, so the draw interval is often a single point; it "
         "is shown for completeness and carries no weight here.", "",
         "| Family | Models | Spearman | Over draws |", "|---|---:|---:|---|"]
    for f, v in R["families"].items():
        L.append(f"| {f} | {v['M']} | {fmt(v['estimate'], 2)} | {iv_txt(v)} |")
    L += ["", f"Positive estimates: {R['macros']['lateGenFamPos']} of {R['macros']['lateGenFamN']} families.", "",
          "## Models, by release date", "", "| Model | Family | Release date | Late share | Baseline |",
          "|---|---|---|---:|---:|"]
    for r in sorted(R["table"], key=lambda r: (r["release_date"], r["model_key"])):
        L.append(f"| {r['model_key']} | {r['family']} | {r['release_date']} | {fmt(100 * r['share_late'], 1)}% | "
                 f"{fmt(r['baseline'], 1)} |")
    return L


if __name__ == "__main__":
    run()
