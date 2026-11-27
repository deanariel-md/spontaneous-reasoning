r"""Beyond chain of thought: the arm-switch tests on five-shot answering, eight exemplars and native thinking, with chain
of thought beside them (methods in results/generality/methods.md).

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_generality.py

Existing scored tables only, through the loaders of the analyses that produced each setting (run_decision.py,
run_fewshot.py, run_exemplar.py, run_thinking.py). Rows whose scores do not exist are kept and marked not computed.

Outputs: results/generality/generality.json, summary.md, summary_clean.md, tables/rows.csv, tables/thinking_vs_cot.csv,
tables/per_model.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src import decision as dc, generality as gen  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "generality")
N_BOOT = 9999
TASK_NAMES = {"gsm8k": "GSM8K-29", "math": "MATH-500", "bbh": "BIG-Bench Hard-28"}
SCORER = {"strict": "strict", "content": "content", "lenient": "lenient", "cheng": "Cheng et al.'s corrected extractor"}

# (row id, intervention, task, baseline, protocol change, protocol a, protocol b); a is the study's reading.
ROWS = []
for _t in ("gsm8k", "math", "bbh"):
    ROWS.append((f"fs_{_t}_content", "five-shot", _t, "zero-shot forced direct", "scorer", "strict", "content"))
    ROWS.append((f"fs_{_t}_lenient", "five-shot", _t, "zero-shot forced direct", "scorer", "strict", "lenient"))
ROWS.append(("ex_gsm8k_cheng", "eight exemplars", "gsm8k", "zero-shot chain of thought", "scorer", "strict", "cheng"))
ROWS.append(("ex_gsm8k_content", "eight exemplars", "gsm8k", "zero-shot chain of thought", "scorer", "strict", "content"))
ROWS.append(("thi_gsm8k_arm", "native thinking", "gsm8k", "requested direct / forced direct", "baseline", "requested",
             "forced"))
ROWS.append(("thi_gsm8k_content", "native thinking", "gsm8k", "requested direct", "scorer", "strict", "content"))
ROWS.append(("thifd_gsm8k_content", "native thinking", "gsm8k", "forced direct", "scorer", "strict", "content"))
for _t in ("gsm8k", "math", "bbh"):
    ROWS.append((f"cot_{_t}_content", "chain of thought", _t, "requested direct", "scorer", "strict", "content"))

CONTENT = os.path.join(ROOT, "data", "processed", "content")
# the content tables of scripts/score_content_added.py that each row's content scores need
CONTENT_FILES = {"five-shot": lambda t: [f"fewshot_{t}_content.parquet"],
                 "eight exemplars": lambda t: ["exemplar_gsm8k_content.parquet", "exemplar_zero_shot_4k_content.parquet"],
                 "native thinking": lambda t: ["thinking_gsm8k_content.parquet"]}
NOT_COMPUTED = {
    "content_fs": "no content scoring of the five-shot outputs exists (data/processed/content/ holds the collection's "
                  "requested-direct, chain-of-thought and forced-direct arms only)",
    "content_ex": "no content scoring of the eight-shot outputs exists (data/processed/content/ holds the collection's "
                  "three arms only; the zero-shot arm has one, the eight-shot arm does not)",
    "content_thi": "no content scoring of the native-thinking outputs exists (data/processed/content/ holds the "
                   "collection's three arms only)"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def rng(label):
    return np.random.default_rng(dc.label_seed(f"generality/{label}"))


def families_of(panel):
    fam = {}
    for t in ri.TASKS:
        f = ri.load(t)["facts"]
        fam.update(dict(zip(f["panel"], f["families"])))
    return [fam[k] for k in panel]


# ============================================================================ inputs: (units, panel, B, P) per split


def collection(task, scoring, split):
    import run_decision as rd
    x = rd.task_split(task, scoring, split)
    return [str(u) for u in x["units"]], [str(m) for m in x["panel"]], {a: np.asarray(x[a], float)
                                                                         for a in ("requested", "forced", "cot")}


KEY = ["model_key", "arm", "config_id", "item_id"]


def content_loader(load, path):
    """A loader of scored rows with the content score of a table beside them; refused unless every row has one and the
    strict scores agree (scripts/run_outcomes.with_content, for a named table)."""
    tab = pd.read_parquet(path, columns=KEY + ["strict_correct", "content_correct"])
    tab["config_id"] = tab["config_id"].astype(str)

    def f(ids):
        df = load(ids)
        c = tab[tab["config_id"].isin(set(map(str, ids)))]
        c = c.assign(item_id=c["item_id"].astype(df["item_id"].dtype))
        j = df.merge(c.rename(columns={"strict_correct": "_strict"}), on=KEY, how="left", validate="one_to_one")
        if j["content_correct"].isna().any() or (j["_strict"] != j["strict_correct"]).any():
            raise SystemExit(f"{rel(path)}: the content table does not match the scored rows")
        return j.drop(columns="_strict")
    return f


def fewshot(task, value):
    import run_fewshot
    x = run_fewshot.inputs(task, False)
    if value == "content_correct":
        from run_outcomes import with_content
        x["fd_load"] = with_content(x["fd_load"], task)
        x["fs_load"] = content_loader(x["fs_load"], os.path.join(CONTENT, f"fewshot_{task}_content.parquet"))
    out = {}
    for split, units in (("discovery", x["disc"]), ("validation", x["val"])):
        units = [str(u) for u in units]
        B, P = run_fewshot.pair_arrays(x["fd_load"], x["fs_load"], units, x["panel"], x["items"], value=value)
        out[split] = (units, list(x["panel"]), B, P)
    return out


def exemplar(scorer):
    import run_decision as rd
    out = {}
    if scorer == "strict":
        for split in ("discovery", "validation"):
            x = rd.exemplar_split(split)
            out[split] = ([str(u) for u in x["units"]], [str(m) for m in x["panel"]],
                          np.asarray(x["zero_shot_cot"], float), np.asarray(x["eight_shot_cot"], float))
        return out
    import run_exemplar as rx
    if scorer == "content":
        return exemplar_content()
    ref = exemplar("strict")
    cheng = pd.read_parquet(os.path.join(rx.PROCESSED, "cheng_scored.parquet"))
    panel = ref["discovery"][1]
    cheng = cheng[cheng["model_key"].isin(panel)]
    for split in ("discovery", "validation"):
        units = ref[split][0]
        B, P = rx.long_arrays(cheng, units, panel)
        out[split] = (units, panel, B, P)
    return out


def exemplar_content():
    """The exemplar run's two arms under content scores: the collection's zero-shot content scores, the 101 outputs that the exemplar run
    scores at their cut read from their cut texts, and the eight-shot content table."""
    import run_exemplar as rx
    from run_forced_direct import collection_inputs
    from run_outcomes import with_content
    inp, s, items, items_info = collection_inputs("gsm8k")
    ex_load, _, panel = rx.exemplar_inputs(list(inp["panel"]), s, items_info, inp["load"], False, False)
    col_c = with_content(inp["load"], "gsm8k")
    ex_c = content_loader(ex_load, os.path.join(CONTENT, "exemplar_gsm8k_content.parquet"))
    cut = rx.cut_table()
    cutc = pd.read_parquet(os.path.join(CONTENT, "exemplar_zero_shot_4k_content.parquet"))
    cutc["config_id"] = cutc["config_id"].astype(str)
    k3 = ["model_key", "config_id", "item_id"]

    def load(ids):
        z, _ = rx.zero_shot(col_c(ids), cut)
        c = cutc[cutc["config_id"].isin(set(map(str, ids)))][k3 + ["content_correct", "strict_correct"]]
        c = c.assign(item_id=c["item_id"].astype(z["item_id"].dtype))
        z = z.merge(c.rename(columns={"content_correct": "_cc", "strict_correct": "_sc"}), on=k3, how="left",
                    validate="one_to_one")
        hit = z["_cc"].notna()
        if (z.loc[hit, "_sc"] != z.loc[hit, "strict_correct"]).any():
            raise SystemExit("the cut zero-shot content table does not match the exemplar run's cut strict scores")
        z.loc[hit, "content_correct"] = z.loc[hit, "_cc"].astype(z["content_correct"].dtype)
        z = z.drop(columns=["_cc", "_sc"])
        return pd.concat([z.assign(arm="direct_answer"), ex_c(ids).assign(arm="zero_shot_cot")], ignore_index=True)
    out = {}
    for split in ("discovery", "validation"):
        units = [str(u) for u in s[f"{split}_config_ids"]]
        B, P = rx.arrays(load, units, panel, items, value="content_correct")
        out[split] = (units, list(panel), B, P)
    return out


def thinking(value="strict_correct"):
    """{split: (units, panel, Bfd, Breq, Pc, Pth)} for GSM8K's thinking checkpoints, as scripts/run_thinking.py; with
    value content_correct, every arm under content scores."""
    import run_thinking as rt
    from run_forced_direct import collection_inputs, forced_inputs
    inp, s, items, items_info = collection_inputs("gsm8k")
    full = list(inp["panel"])
    fd_load, _, _ = forced_inputs("gsm8k", full, s, items_info, inp["load"], False)
    panel = rt.thinking_panel("gsm8k", full, False)
    th_load, _, _, panel = rt.thinking_inputs("gsm8k", panel, s, items_info, inp["load"], False, False)
    col_load = inp["load"]
    if value == "content_correct":
        from run_outcomes import with_content
        col_load, fd_load = with_content(col_load, "gsm8k"), with_content(fd_load, "gsm8k")
        th_load = content_loader(th_load, os.path.join(CONTENT, "thinking_gsm8k_content.parquet"))
    out = {}
    for split in ("discovery", "validation"):
        units = [str(u) for u in s[f"{split}_config_ids"]]
        out[split] = (units, list(panel)) + tuple(rt.four_arm_arrays(col_load, fd_load, th_load, units, panel, items,
                                                                      value=value))
    return out


# ============================================================================ one row


def compute(row_id, Bd: dict, units_d, units_v, panel):
    """Bd: {protocol: (discovery benefit (K_d, M), validation benefit (K_v, M))}, two protocols in order."""
    a, b = list(Bd)
    g1, g2 = Bd[a][0].mean(0), Bd[b][0].mean(0)
    rev = gen.reversals(g1, g2, families_of(panel), N_BOOT, rng(row_id))
    cost = gen.decision_cost({p: Bd[p][0] for p in (a, b)}, {p: Bd[p][1] for p in (a, b)}, units_d, units_v)
    return {"computed": True, "M": len(panel), "panel": list(panel), "reversals": rev, "decision_cost": cost,
            "mean_benefit": {a: float(g1.mean()), b: float(g2.mean())},
            "per_model": {a: dict(zip(panel, map(float, g1))), b: dict(zip(panel, map(float, g2)))}}


def same(units_a, panel_a, units_b, panel_b, what):
    if list(units_a) != list(units_b) or list(panel_a) != list(panel_b):
        raise SystemExit(f"{what}: draws or panel differ between the two protocols")


def run():
    t0 = time.time()
    R = {"analysis": "the arm-switch tests on other interventions ("
                     "results/generality/methods.md)", "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": "pp", "n_boot": N_BOOT, "design": "one-draw pilots (own gain, each distinct discovery configuration "
         "once), valued on the fresh validation configurations; reversals on discovery-draw model means",
         "rows": {}, "thinking_vs_cot": {}, "checks": {}}
    cache = {}
    for row_id, interv, task, base, change, a, b in ROWS:
        name = "GSM8K, thinking checkpoints" if interv == "native thinking" else TASK_NAMES[task]
        meta = {"intervention": interv, "task": task, "task_name": name, "baseline": base,
                "change": change, "protocols": [a, b]}
        if b == "content" and interv != "chain of thought":
            need = CONTENT_FILES[interv](task)
            if not all(os.path.exists(os.path.join(CONTENT, f)) for f in need):
                key = {"five-shot": "content_fs", "eight exemplars": "content_ex", "native thinking": "content_thi"}[interv]
                R["rows"][row_id] = {**meta, "computed": False, "reason": NOT_COMPUTED[key]}
                print(f"{row_id}: not computed", flush=True)
                continue
            meta["content_tables"] = [f"data/processed/content/{f}" for f in need]
        if interv == "five-shot":
            S = {p: fewshot(task, f"{p}_correct") for p in (a, b)}
            for sp in ("discovery", "validation"):
                same(S[a][sp][0], S[a][sp][1], S[b][sp][0], S[b][sp][1], row_id)
            ud, uv, panel = S[a]["discovery"][0], S[a]["validation"][0], S[a]["discovery"][1]
            Bd = {p: (S[p]["discovery"][3] - S[p]["discovery"][2], S[p]["validation"][3] - S[p]["validation"][2])
                  for p in (a, b)}
        elif interv == "eight exemplars":
            S = {p: exemplar(p) for p in (a, b)}
            for sp in ("discovery", "validation"):
                same(S[a][sp][0], S[a][sp][1], S[b][sp][0], S[b][sp][1], row_id)
            ud, uv, panel = S[a]["discovery"][0], S[a]["validation"][0], S[a]["discovery"][1]
            Bd = {p: (S[p]["discovery"][3] - S[p]["discovery"][2], S[p]["validation"][3] - S[p]["validation"][2])
                  for p in (a, b)}
        elif interv == "native thinking" and change == "baseline":
            T = cache.setdefault("thinking", thinking())
            ud, panel, Bfd, Breq, Pc, Pth = T["discovery"]
            uv, _, Bfdv, Breqv, Pcv, Pthv = T["validation"]
            Bd = {"requested": (Pth - Breq, Pthv - Breqv), "forced": (Pth - Bfd, Pthv - Bfdv)}
        elif interv == "native thinking":
            ix = 2 if base == "forced direct" else 3     # position of Bfd / Breq in (units, panel, Bfd, Breq, Pc, Pth)
            Bd = {}
            for p, v in (("strict", "strict_correct"), ("content", "content_correct")):
                T = cache.setdefault("thinking" if p == "strict" else "thinking_content", thinking(v))
                Bd[p] = (T["discovery"][5] - T["discovery"][ix], T["validation"][5] - T["validation"][ix])
            S0, S1 = cache["thinking"], cache["thinking_content"]
            same(S0["discovery"][0], S0["discovery"][1], S1["discovery"][0], S1["discovery"][1], row_id)
            same(S0["validation"][0], S0["validation"][1], S1["validation"][0], S1["validation"][1], row_id)
            ud, panel = S0["discovery"][:2]
            uv = S0["validation"][0]
        else:
            S = {}
            for p in (a, b):
                d, v = collection(task, p, "discovery"), collection(task, p, "validation")
                S[p] = (d, v)
            same(S[a][0][0], S[a][0][1], S[b][0][0], S[b][0][1], row_id)
            same(S[a][1][0], S[a][1][1], S[b][1][0], S[b][1][1], row_id)
            ud, panel, uv = S[a][0][0], S[a][0][1], S[a][1][0]
            Bd = {p: (S[p][0][2]["cot"] - S[p][0][2]["requested"], S[p][1][2]["cot"] - S[p][1][2]["requested"])
                  for p in (a, b)}
        R["rows"][row_id] = {**meta, **compute(row_id, Bd, ud, uv, panel)}
        r = R["rows"][row_id]["reversals"]
        print(f"{row_id}: {r['reversed']} of {r['pairs']} reversed ({time.time() - t0:.0f} s)", flush=True)

    # native thinking against prompted chain of thought over the same baseline (beside \thiGsmCBetween)
    T = cache.setdefault("thinking", thinking())
    ud, panel, Bfd, Breq, Pc, Pth = T["discovery"]
    for base, Bx in (("requested", Breq), ("forced", Bfd)):
        R["thinking_vs_cot"][base] = gen.reversals((Pth - Bx).mean(0), (Pc - Bx).mean(0), families_of(panel), N_BOOT,
                                                   rng(f"thinking_vs_cot/{base}"))
        R["thinking_vs_cot"][base]["mean_benefit"] = {"native": float((Pth - Bx).mean()), "cot": float((Pc - Bx).mean())}
    R["thinking_means"] = {"native": float(Pth.mean()), "requested": float(Breq.mean()), "forced": float(Bfd.mean()),
                           "cot": float(Pc.mean()), "M": len(panel), "panel": list(panel)}

    checks(R)
    summary_minmax(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    write(R)
    print(f"done in {R['runtime_seconds']} s -> results/generality", flush=True)
    return R


# ============================================================================ checks


def checks(R):
    """The reproductions of methods.md, section Checks."""
    import json
    import re
    p = os.path.join(ROOT, "paper", "numbers.tex")
    tex = open(p, encoding="utf-8").read() if os.path.exists(p) else ""   # before the first make_numbers.py run: no printed values

    def macro(name):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", tex)
        return None if m is None else m.group(1).replace("$-$", "-").replace(",", "")
    D = json.load(open(os.path.join(ROOT, "results", "decision", "decision.json"), encoding="utf-8"))
    tag = {"gsm8k": "Gsm", "bbh": "Bbh", "math": "Math"}
    C = {"arm_switch": {}, "content_cost": {}, "thinking_means": {}}
    for task in ("gsm8k", "bbh", "math"):
        d, v = collection(task, "strict", "discovery"), collection(task, "strict", "validation")
        Bd = {"requested": (d[2]["cot"] - d[2]["requested"], v[2]["cot"] - v[2]["requested"]),
              "forced": (d[2]["cot"] - d[2]["forced"], v[2]["cot"] - v[2]["forced"])}
        cost = gen.decision_cost({p: Bd[p][0] for p in Bd}, {p: Bd[p][1] for p in Bd}, d[0], v[0])
        rev = gen.rb.pair_reversals(Bd["requested"][0].mean(0), Bd["forced"][0].mean(0))
        t = tag[task]
        C["arm_switch"][task] = {
            "ReqToFd": {"here": round(cost["requested_to_forced"]["estimate"], 2), "printed": macro(f"ddir{t}ReqToFd")},
            "FdToReq": {"here": round(cost["forced_to_requested"]["estimate"], 2), "printed": macro(f"ddir{t}FdToReq")},
            "Reversed": {"here": rev["reversed"], "printed": macro(f"ae{t}Reversed")}}
        c = next(c for c in D["cross_protocol"] if c["choice"] == f"{task}_req" and c["evaluation_setting"] == f"{task}_con")
        printed = c["evaluation"]["independent"]["rules"]["own_1"]["extra_regret"]["estimate"]
        here = R["rows"][f"cot_{task}_content"]["decision_cost"]["strict_to_content"]["estimate"]
        C["content_cost"][task] = {"here": here, "printed": printed, "abs_difference": abs(here - printed)}
    tm = R["thinking_means"]
    for k, name in (("native", "thiGsmMeanNative"), ("requested", "thiGsmMeanReq"), ("forced", "thiGsmMeanForced"),
                    ("cot", "thiGsmMeanCot")):
        C["thinking_means"][k] = {"here": round(tm[k], 1), "printed": macro(name)}
    ok = []
    for task, x in C["arm_switch"].items():
        for k, v in x.items():
            ok.append(v["printed"] is not None and float(v["printed"]) == float(v["here"]))
            v["reproduced"] = ok[-1]
    for task, v in C["content_cost"].items():
        v["reproduced"] = v["abs_difference"] < 1e-9
        ok.append(v["reproduced"])
    for k, v in C["thinking_means"].items():
        v["reproduced"] = v["printed"] is not None and abs(float(v["printed"]) - v["here"]) < 1e-9
        ok.append(v["reproduced"])
    C["all_reproduced"] = bool(all(ok))
    R["checks"] = C


def summary_minmax(R):
    def shares(rows):
        return [(k, 100 * R["rows"][k]["reversals"]["reversed_share"]) for k in rows
                if R["rows"][k].get("computed") and R["rows"][k]["change"] == "scorer"]
    other = shares([k for k in R["rows"] if not k.startswith("cot_")])
    allr = shares(list(R["rows"]))
    R["scorer_reversal_pct"] = {
        "other_interventions": {"min": min(v for _, v in other), "max": max(v for _, v in other),
                                "argmin": min(other, key=lambda x: x[1])[0], "argmax": max(other, key=lambda x: x[1])[0],
                                "rows": [k for k, _ in other]},
        "with_cot": {"min": min(v for _, v in allr), "max": max(v for _, v in allr), "rows": [k for k, _ in allr]}}


# ============================================================================ outputs


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}".replace("-", "−")


def ci(c, nd=2):
    return "–" if not c else f"[{f(c[0], nd)}, {f(c[1], nd)}]"


def proto(p):
    return SCORER.get(p, p)


def summary(R):
    L = ["# Beyond chain of thought: the same tests on the other interventions", "",
         "Methods: `results/generality/methods.md`. Existing scored tables only. Benefits are discovery-draw model means; a pair "
         "is reversed when the two protocols order it oppositely, ties under either are neither (the `\\ae…` "
         "counting). The interval for the share resamples models (9,999 resamples). The decision cost is the one-draw "
         "design of `\\ddir…`: the top third by own benefit on one discovery draw under one protocol, valued under the "
         "other on the fresh validation configurations, against choosing under that protocol; pp of benefit per chosen "
         "model, t interval over the validation configurations. These validation draws had been read by earlier "
         "analyses, so the costs are exploratory.", "",
         "## Reversed pairs and decision cost", "",
         "| Intervention | Benchmark | Baseline | Protocol change | Reversed / pairs | Share | Model-resampling 95% "
         "interval | Cost, a → b (pp) | Cost, b → a (pp) | Fresh configs |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in R["rows"].items():
        a, b = r["protocols"]
        lab = f"{proto(a)} vs {proto(b)}"
        if not r.get("computed"):
            L.append(f"| {r['intervention']} | {r['task_name']} | {r['baseline']} | {r['change']}: {lab} | not computed "
                     f"| – | – | – | – | – |")
            continue
        v, c = r["reversals"], r["decision_cost"]
        ab, ba = c.get(f"{a}_to_{b}", {}), c.get(f"{b}_to_{a}", {})
        L.append(f"| {r['intervention']} | {r['task_name']} | {r['baseline']} | {r['change']}: {lab} | "
                 f"{v['reversed']} / {v['pairs']} | {f(v['reversed_share'], 3)} | {ci(v['model_bootstrap']['ci95'], 3)} | "
                 f"{f(ab.get('estimate'))} {ci(ab.get('ci95'))} | {f(ba.get('estimate'))} {ci(ba.get('ci95'))} | "
                 f"{c['n_configurations']} |")
    L += ["", "a is the first protocol named (the study's reading: strict scoring; for native thinking, the requested "
          "baseline), b the second. Cost a → b: chosen under a, used under b, against choosing under b.", "",
          "Rows not computed:", ""]
    if all(r.get("computed") for r in R["rows"].values()):
        L.append("- none.")
    for k, r in R["rows"].items():
        if not r.get("computed"):
            L.append(f"- {r['intervention']}, {r['task_name']}, {r['change']} {proto(r['protocols'][0])} vs "
                     f"{proto(r['protocols'][1])}: {r['reason']}.")
    L += ["", "## Reversals among pairs with large gaps", "",
          "| Row | Ties | Gap > 1 pp | > 2 pp | > 5 pp | > 10 pp | Family bootstrap | Leave one family out |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in R["rows"].items():
        if not r.get("computed"):
            continue
        v = r["reversals"]
        bt = {x["min_gap_pp"]: x for x in v["by_threshold"]}
        cells = " | ".join(f"{bt[g]['reversed']} of {bt[g]['pairs_both_above']}" for g in (1.0, 2.0, 5.0, 10.0))
        lo = v["leave_one_family_out_range"]
        L.append(f"| {k} | {v['ties']} | {cells} | {ci(v['family_bootstrap']['ci95'], 3)} | "
                 f"{f(lo[0], 3)} to {f(lo[1], 3)} |")
    L += ["", "The native-thinking panel holds two families (Gemma, one model; Qwen, seven), so its family bootstrap is "
          "undefined in the resamples that draw Gemma alone, and leaving Qwen out leaves one model (no pair); both are "
          "reported as they are.", "", "## Native thinking against prompted chain of thought", "",
          f"GSM8K, the {R['thinking_means']['M']} thinking checkpoints, discovery means. Pairs ordered one way by the "
          "native-thinking benefit and the other by the prompted chain-of-thought benefit over the same baseline "
          "(beside `\\thiGsmCBetween`, the draw-by-draw agreement over forced direct).", "",
          "| Baseline | Reversed / pairs | Share | Model-resampling 95% interval | Mean native benefit | Mean CoT benefit |",
          "|---|---|---|---|---|---|"]
    for base, v in R["thinking_vs_cot"].items():
        L.append(f"| {base} direct | {v['reversed']} / {v['pairs']} | {f(v['reversed_share'], 3)} | "
                 f"{ci(v['model_bootstrap']['ci95'], 3)} | {f(v['mean_benefit']['native'], 1)} | "
                 f"{f(v['mean_benefit']['cot'], 1)} |")
    s = R["scorer_reversal_pct"]
    L += ["", "## Across interventions", "",
          f"Reversed share under a scorer change, five-shot, eight exemplars and native thinking (computed rows): "
          f"{f(s['other_interventions']['min'], 1)}% to {f(s['other_interventions']['max'], 1)}%; with the "
          f"chain-of-thought rows: {f(s['with_cot']['min'], 1)}% to {f(s['with_cot']['max'], 1)}%.", "",
          "## Checks", ""]
    C = R["checks"]
    for task, x in C["arm_switch"].items():
        L.append(f"- {TASK_NAMES[task]}, chain of thought, requested vs forced (same code): cost requested → forced "
                 f"{x['ReqToFd']['here']} (printed {x['ReqToFd']['printed']}), forced → requested {x['FdToReq']['here']} "
                 f"(printed {x['FdToReq']['printed']}), reversed {x['Reversed']['here']} (printed "
                 f"{x['Reversed']['printed']}).")
    for task, v in C["content_cost"].items():
        L.append(f"- {TASK_NAMES[task]}, chain of thought, strict → content cost {f(v['here'], 4)} against "
                 f"results/decision/decision.json {f(v['printed'], 4)}: "
                 f"{'reproduced' if v['reproduced'] else 'differs'}.")
    tm = C["thinking_means"]
    L.append("- Native-thinking panel means (native, requested, forced, chain of thought): "
             + ", ".join(f"{v['here']} (printed {v['printed']})" for v in tm.values()) + ".")
    L.append(f"- All reproduced: {'yes' if C['all_reproduced'] else 'no'}.")
    return L


def write(R):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    rows, per = [], []
    for k, r in R["rows"].items():
        base = {"row": k, "intervention": r["intervention"], "task": r["task"], "baseline": r["baseline"],
                "change": r["change"], "protocol_a": r["protocols"][0], "protocol_b": r["protocols"][1],
                "computed": r.get("computed", False)}
        if r.get("computed"):
            v, c = r["reversals"], r["decision_cost"]
            a, b = r["protocols"]
            ab, ba = c.get(f"{a}_to_{b}", {}), c.get(f"{b}_to_{a}", {})
            ci_m = v["model_bootstrap"]["ci95"] or [None, None]
            base.update({"M": r["M"], "pairs": v["pairs"], "reversed": v["reversed"], "share": v["reversed_share"],
                         "ties": v["ties"], "share_lo": ci_m[0], "share_hi": ci_m[1],
                         "cost_a_to_b": ab.get("estimate"), "cost_a_to_b_lo": (ab.get("ci95") or [None, None])[0],
                         "cost_a_to_b_hi": (ab.get("ci95") or [None, None])[1], "cost_b_to_a": ba.get("estimate"),
                         "cost_b_to_a_lo": (ba.get("ci95") or [None, None])[0],
                         "cost_b_to_a_hi": (ba.get("ci95") or [None, None])[1],
                         "n_configurations": c["n_configurations"], "n_chosen": c["n_chosen"]})
            for p in r["protocols"]:
                for m, g in r["per_model"][p].items():
                    per.append({"row": k, "protocol": p, "model_key": m, "benefit": g})
        else:
            base["reason"] = r["reason"]
        rows.append(base)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "rows.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "tables", "per_model.csv"), index=False)
    pd.DataFrame([{"baseline": b, "pairs": v["pairs"], "reversed": v["reversed"], "share": v["reversed_share"],
                   "share_lo": (v["model_bootstrap"]["ci95"] or [None, None])[0],
                   "share_hi": (v["model_bootstrap"]["ci95"] or [None, None])[1]}
                  for b, v in R["thinking_vs_cot"].items()]).to_csv(os.path.join(OUT, "tables", "thinking_vs_cot.csv"),
                                                                    index=False)
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import macros_generality
    ri.write_outputs(OUT, R, summary(R), macros=macros_generality.generality_macros(ri.jsonable(R)))


if __name__ == "__main__":
    run()
