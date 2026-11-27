r"""Both directions of the wrong-baseline decision cost, with leave-one-family-out ranges, and the model-level
uncertainty of the arm-switch reversal share.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_decision_directions.py

GSM8K-29, BIG-Bench Hard-28 and MATH-500 (29 models, 398 numeric items). Their validation draws had been read by
earlier analyses, so nothing here chooses anew: every choice is one already saved,
  design "one-draw pilots": manifests/predictions_decision_<task>_req and _fd (.json/.npz; scripts/run_decision.py),
      every rule; the paper's printed cost is own gain from one draw (own_1), valued on the validation
      configurations absent from every discovery draw;
  design "100-item pilots": manifests/predictions_decision_strict (scripts/run_decision_strict.py),
      item-disjoint pilots, valued on the other half of the items, fresh validation configurations; three
      rules (own gain is the main comparison).
Each choice made under the requested arm is valued under the forced arm against the forced arm's own choice on the same
pilot sets, and the reverse. Leave-one-family-out: each family dropped, the top third of the models that remain
re-taken from the same saved discovery predictions (no validation data enter a choice), both directions valued.
Content scores: the same saved choices, made on strict scores, valued on content-scored validation
gains (scripts/score_content.py); no content-scored choice is saved for the forced arm, so the matched comparator is the
strict-scored choice of the deployment arm, valued on content scores too.

Reversal share: model pairs whose chain-of-thought gain over the requested arm and over the forced arm order them
oppositely (model means over the 24 discovery draws, results/arm_effect/'s definition), with a model bootstrap, a
family bootstrap, each family left out and the count within families; strict and content scores.

Outputs: results/decision_directions/decision_directions.json, summary.md, summary_clean.md, tables/costs.csv,
tables/leave_one_family_out.csv, tables/reversal_share.csv.
"""
import json
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
from src import decision as dc, decision_directions as dd, decision_strict as ds, rule_budgets as rbud  # noqa: E402
import robustness_inputs as ri  # noqa: E402

OUT = os.path.join(ROOT, "results", "decision_directions")
MAN = os.path.join(ROOT, "manifests")
# the 100-item pilots saved by scripts/run_decision_strict.py, in manifests/
STRICT_FROZEN = os.path.join(ROOT, "manifests", "predictions_decision_strict.json")
CONTENT = os.path.join(ROOT, "data", "processed", "content")
TASKS = ("gsm8k", "bbh", "math")
TITLE = {"gsm8k": "GSM8K-29", "bbh": "BIG-Bench Hard-28", "math": "MATH-500 numeric"}
ARM = {"requested": "req", "forced": "fd"}
N_BOOT = 9999
# Reference numbers, reproduced as checks (pp per chosen model; own gain).
REF_PILOT = {"gsm8k": (2.52, 8.81), "math": (3.00, 13.08), "bbh": (2.42, 2.98)}        # one-draw pilots: req->fd, fd->req
REF_TABLE = {"gsm8k": ((2.06, 1.87, 2.25), (8.43, 7.79, 9.07)), "math": ((2.57, 2.36, 2.77), (12.75, 12.27, 13.22)),
         "bbh": ((2.65, 2.42, 2.88), (2.29, 2.13, 2.44))}                          # 100-item pilots, [95% CI]
REF_LOFO = {"gsm8k": (3.72, 11.37), "math": (8.67, 14.22), "bbh": (0.44, 2.44)}  # forced->requested
REF_REV = {"gsm8k": ((0.10, 0.33), (0.15, 0.28), (23, 88)), "math": ((0.10, 0.34), (0.16, 0.24), (25, 88)),
             "bbh": ((0.11, 0.27), (0.13, 0.21), (17, 81))}                         # bootstrap, LOFO, within family


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def seed(label):
    return np.random.default_rng(dc.label_seed(f"decision_directions/{label}"))


# ============================================================================ design 1: one-draw pilots (run_decision)


def one_draw_design(task, fams):
    import run_decision as rd
    rec, W, val, pred = {}, {}, {}, {}
    for a, s in ARM.items():
        name = f"{task}_{s}"
        rec[a], W[a] = rd.load_choices(name)
        _, arr = rbud.load_frozen(os.path.join(MAN, f"predictions_decision_{name}.json"),
                                  os.path.join(MAN, f"predictions_decision_{name}.npz"))
        pred[a] = arr["own_1__pred"]
        val[a] = rd.setting_arrays(name, "validation")
    if rec["requested"]["models"] != rec["forced"]["models"] or \
            rec["requested"]["discovery_units"] != rec["forced"]["discovery_units"]:
        raise SystemExit(f"{task}: the two arms' saved choices differ in panel or discovery draws")
    units, panel = val["requested"][0], val["requested"][1]
    if val["forced"][0] != units or val["forced"][1] != panel or panel != rec["requested"]["models"]:
        raise SystemExit(f"{task}: validation draws or panel differ between the arms")
    rows = rd.evaluation_rows(rec["requested"]["discovery_units"], units)["independent"]
    n = rec["requested"]["n_chosen"]
    G = {a: (val[a][3] - val[a][2])[rows] for a in ARM}
    rules = [r for r in rd.RULE_ORDER if r in W["requested"] and r in W["forced"] and r != "random"]
    out = {"n_chosen": n, "M": len(panel), "n_configurations": len(rows), "rules": {}}
    for r in rules:
        out["rules"][r] = dd.both_directions({a: W[a][r] for a in ARM}, G, n)
    out["random_regret"] = {a: float(dc.regret_plugin(W[a]["random"], G[a], n).mean()) for a in ARM}
    out["leave_one_family_out_own_1"] = dd.leave_one_family_out(pred, G, fams)
    # content scores: the same saved choices valued on content-scored validation gains
    if os.path.exists(os.path.join(CONTENT, f"{task}_content.parquet")):
        x = rd.task_split(task, "content", "validation")
        if [str(u) for u in x["units"]] != units or [str(m) for m in x["panel"]] != panel:
            raise SystemExit(f"{task}: the content arrays differ in draws or panel")
        Gc = {a: (np.asarray(x["cot"], float) - np.asarray(x[a], float))[rows] for a in ARM}
        out["content"] = {"note": "choices made on strict scores (saved), valued on content-scored validation gains",
                          "rules": {r: dd.both_directions({a: W[a][r] for a in ARM}, Gc, n) for r in rules},
                          "random_regret": {a: float(dc.regret_plugin(W[a]["random"], Gc[a], n).mean()) for a in ARM},
                          "leave_one_family_out_own_1": dd.leave_one_family_out(pred, Gc, fams)}
    # the printed requested->forced value must reproduce from results/decision/decision.json unchanged
    D = json.load(open(os.path.join(ROOT, "results", "decision", "decision.json"), encoding="utf-8"))
    c = next(c for c in D["cross_protocol"] if c["choice"] == f"{task}_req" and c["evaluation_setting"] == f"{task}_fd")
    printed = c["evaluation"]["independent"]["rules"]["own_1"]["extra_regret"]
    mine = out["rules"]["own_1"]["requested_to_forced"]
    out["check_printed"] = {"printed": printed["estimate"], "here": mine["estimate"],
                            "abs_difference": abs(printed["estimate"] - mine["estimate"]),
                            "ci_printed": printed.get("ci95"), "ci_here": mine.get("ci95")}
    out["choices"] = {a: {"path": f"manifests/predictions_decision_{task}_{s}.json", "sha256": rec[a]["sha256"]}
                      for a, s in ARM.items()}
    return out


# ============================================================================ design 2: 100-item pilots (decision_strict)


def content_dense(x, configs):
    """{'requested', 'forced', 'cot'}: (configs, models, items) int8 content scores of the given configurations."""
    import run_decision_strict as rds
    configs = sorted(set(map(str, configs)))
    df = pd.read_parquet(os.path.join(CONTENT, f"{x['task']}_content.parquet"),
                         columns=["model_key", "arm", "config_id", "item_id", "content_correct"],
                         filters=[("config_id", "in", configs)])
    df["item_id"] = df["item_id"].astype(str)
    S = rds.load_script("run_noise_anatomy").dense_items(df, x["panel"], configs, x["items"],
                                                         arms=("direct_answer", "zero_shot_cot", "forced_direct"),
                                                         value="content_correct")
    return configs, {"requested": S[0], "cot": S[1], "forced": S[2]}


def strict_design(task, rec, arr, fams):
    import run_decision_strict as rds
    x = rds.task_inputs(task)
    if x["panel"] != rec["design"]["models"][task] or x["disc"] != rec["design"]["discovery_units"][task]:
        raise SystemExit(f"{task}: the panel or discovery draws differ from the saved 100-item pilots")
    N, M = len(x["items"]), len(x["panel"])
    n = dc.third_size(M)
    mask = ds.split_items(x["items"], task, rds.SPLIT_LABEL, x["strata"])
    configs, Sv = rds.dense(x, x["val"])
    out = {"n_chosen": n, "M": M, "designs": {}}
    sources = {"strict": (configs, Sv)}
    if os.path.exists(os.path.join(CONTENT, f"{task}_content.parquet")):
        sources["content"] = content_dense(x, x["val"])
    for d in ("disjoint", "shared"):
        emask = ~mask if d == "disjoint" else np.ones(N, bool)
        w = ds.eval_weights(N, emask, x["strata"])
        out["designs"][d] = {}
        for sc, (cf, S) in sources.items():
            G = {a: rds.evaluation_gains(S, cf, x["val"], x["disc"], a, w)["fresh"][0] for a in ARM}
            blk = {"n_configurations": int(G["requested"].shape[0]), "rules": {}}
            for r in ds.RULES:
                Wr = {a: arr[f"{task}__{d}__{a}__{r}__weights"] for a in ARM}
                blk["rules"][r] = dd.both_directions(Wr, G, n)
            blk["random_regret"] = {a: float(dc.regret_plugin(np.full((1, M), n / M), G[a], n).mean()) for a in ARM}
            pred = {a: arr[f"{task}__{d}__{a}__own_gain__pred"] for a in ARM}
            blk["leave_one_family_out_own_gain"] = dd.leave_one_family_out(pred, G, fams)
            if sc == "content":
                blk["note"] = "choices made on strict scores (saved), valued on content-scored validation gains"
            out["designs"][d][sc] = blk
    # same-baseline own-gain regrets must equal results/decision_strict/decision_strict.json
    S0 = json.load(open(os.path.join(ROOT, "results", "decision_strict", "decision_strict.json"), encoding="utf-8"))
    chk = {}
    for a in ARM:
        printed = S0["tasks"][task]["results"]["disjoint"][a]["fresh"]["rules"]["own_gain"]["regret_plugin"]
        here = out["designs"]["disjoint"]["strict"]["rules"]["own_gain"][
            "requested_to_forced" if a == "forced" else "forced_to_requested"]["regret_matched"]
        chk[a] = {"printed": printed, "here": here, "abs_difference": abs(printed - here)}
    out["check_same_baseline"] = chk
    return out


# ============================================================================ HotpotQA (called by run_hotpot.py's full run)

HQ_LABEL = {"requested": "hotpot", "forced": "hotpot_forced"}
HQ_CONTRAST = {"requested": "cot_req", "forced": "cot_forced"}


def hotpot_directions(pred_dir, panel, fams, disc, val, V: dict, Sv: dict, vconfigs, items_n: int) -> dict:
    """Both directions on HotpotQA from the choices its discovery stage saved for the two contrasts (chain of thought
    against requested direct, label hotpot; against forced direct, label hotpot_forced): the one-draw design
    (manifests/predictions_decision_hotpot*.json/.npz) and the 100-item design (manifests/predictions_hotpot_decision_
    strict.*, item-disjoint and shared). V {contrast: (Bv, Pv)} on the validation draws; Sv {arm: (C, M, N)} over the
    distinct validation configurations vconfigs (scripts/hotpot_fair.dense). Strict exact match."""
    import hotpot_fair as hf
    import run_decision as rd
    panel = list(panel)
    rec, W, pred = {}, {}, {}
    for a, label in HQ_LABEL.items():
        rec[a], W[a] = rd.load_choices(label, pred_dir)
        _, arr = rbud.load_frozen(os.path.join(pred_dir, f"predictions_decision_{label}.json"),
                                  os.path.join(pred_dir, f"predictions_decision_{label}.npz"))
        pred[a] = arr["own_1__pred"]
        if rec[a]["models"] != panel:
            raise SystemExit(f"predictions_decision_{label} was saved for another panel")
    if rec["requested"]["discovery_units"] != rec["forced"]["discovery_units"]:
        raise SystemExit("the two HotpotQA contrasts' choices differ in discovery draws")
    rows = rd.evaluation_rows(rec["requested"]["discovery_units"], list(map(str, val)))["independent"]
    n = rec["requested"]["n_chosen"]
    out = {"analysis": "both directions of the wrong-baseline decision cost on HotpotQA",
           "units": "pp of accuracy gain per chosen model",
           "choices": {a: {"path": f"manifests/predictions_decision_{label}.json", "sha256": rec[a].get("sha256")}
                       for a, label in HQ_LABEL.items()}}
    G = {a: (np.asarray(V[c][1], float) - np.asarray(V[c][0], float))[rows] for a, c in HQ_CONTRAST.items()}
    o = {"n_chosen": n, "M": len(panel), "n_configurations": len(rows), "rules": {}}
    if len(rows) < 2:
        o["note"] = "fewer than two evaluation configurations"
    else:
        for r in [r for r in rd.RULE_ORDER if r in W["requested"] and r in W["forced"] and r != "random"]:
            o["rules"][r] = dd.both_directions({a: W[a][r] for a in ARM}, G, n)
        o["random_regret"] = {a: float(dc.regret_plugin(W[a]["random"], G[a], n).mean()) for a in ARM}
        o["leave_one_family_out_own_1"] = dd.leave_one_family_out(pred, G, fams)
    out["one_draw"] = o
    srec, sarr = rbud.load_frozen(*hf.files(hf.STRICT, pred_dir))
    if list(srec["models"]) != panel:
        raise SystemExit(f"{hf.STRICT} was saved for another panel")
    mask = np.asarray(sarr["item_split_pilot_pool"], bool)
    fresh = hf.eval_sets(disc, val)["fresh"]
    cs = [str(val[r]) for r in fresh]
    ci_ = {c: i for i, c in enumerate(vconfigs)}
    out["pilot100"] = {"saved": {"path": f"manifests/{hf.STRICT}.json", "sha256": srec.get("sha256")}, "designs": {}}
    for d in ("disjoint", "shared"):
        if len(cs) < 2:
            out["pilot100"]["designs"][d] = {"n_configurations": len(cs), "note": "fewer than two evaluation configurations"}
            continue
        w = ds.eval_weights(items_n, ~mask if d == "disjoint" else np.ones(items_n, bool))
        ix = [ci_[c] for c in cs]
        Gs = {}
        for a, c in HQ_CONTRAST.items():
            b, q = {"cot_req": ("direct_answer", "zero_shot_cot"), "cot_forced": ("forced_direct", "zero_shot_cot")}[c]
            Gs[a] = (Sv[q][ix].astype(float) - Sv[b][ix].astype(float)) @ w
        blk = {"n_configurations": len(cs), "configurations": cs, "rules": {}}
        for r in ds.RULES:
            blk["rules"][r] = dd.both_directions(
                {a: sarr[f"hotpot__{d}__{HQ_CONTRAST[a]}__{r}__weights"] for a in ARM}, Gs, srec["n_chosen"])
        blk["random_regret"] = {a: float(dc.regret_plugin(np.full((1, len(panel)), srec["n_chosen"] / len(panel)),
                                                          Gs[a], srec["n_chosen"]).mean()) for a in ARM}
        blk["leave_one_family_out_own_gain"] = dd.leave_one_family_out(
            {a: sarr[f"hotpot__{d}__{HQ_CONTRAST[a]}__own_gain__pred"] for a in ARM}, Gs, fams)
        out["pilot100"]["designs"][d] = blk
    return out


def hotpot_lines(H: dict) -> list:
    L = ["", "## Both directions of the wrong-baseline decision cost (choices saved at the discovery stage)", ""]
    o = H["one_draw"]
    if o.get("rules"):
        r = o["rules"]["own_1"]
        lo = o["leave_one_family_out_own_1"]["range"]
        L.append(f"- One-draw pilots, own gain, {o['n_configurations']} configurations: requested → forced "
                 f"{ci(r['requested_to_forced'])}; forced → requested {ci(r['forced_to_requested'])}; leaving one family "
                 f"out {rng_txt(lo['requested_to_forced'])} and {rng_txt(lo['forced_to_requested'])}.")
    d = H["pilot100"]["designs"].get("disjoint") or {}
    if d.get("rules"):
        r = d["rules"]["own_gain"]
        L.append(f"- 100-item pilots, item-disjoint, own gain, {d['n_configurations']} fresh configurations: requested → "
                 f"forced {ci(r['requested_to_forced'])}; forced → requested {ci(r['forced_to_requested'])}.")
    return L


# ============================================================================ reversal share


def reversal_block(task):
    import run_decision as rd
    x = ri.load(task)
    f, d = x["facts"], x["discovery"]
    fams = f["families"]
    out = {}
    req, fd, cot = (d[a].mean(0) for a in ("requested", "forced", "cot"))
    out["strict"] = dd.reversal_uncertainty(cot - req, cot - fd, fams, N_BOOT, seed(f"reversal/{task}/strict"))
    if os.path.exists(os.path.join(CONTENT, f"{task}_content.parquet")):
        c = rd.task_split(task, "content", "discovery")
        if [str(m) for m in c["panel"]] != f["panel"] or [str(u) for u in c["units"]] != f["discovery_config_ids"]:
            raise SystemExit(f"{task}: content discovery arrays differ from the strict ones in panel or draws")
        cq, cf, cc = (np.asarray(c[a], float).mean(0) for a in ("requested", "forced", "cot"))
        out["content"] = dd.reversal_uncertainty(cc - cq, cc - cf, fams, N_BOOT, seed(f"reversal/{task}/content"))
    return out


# ============================================================================ report


def f2(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}".replace("-", "−")


def ci(d):
    c = d.get("ci95")
    return f"{f2(d['estimate'])} [{f2(c[0])}, {f2(c[1])}]" if c else f2(d["estimate"])


def rng_txt(r):
    return "–" if not r else f"{f2(r[0])} to {f2(r[1])}"


def summary(R):
    L = ["# Both directions of the wrong-baseline decision cost", "",
         "Code `scripts/run_decision_directions.py`, "
         "`src/decision_directions.py`; numbers in `decision_directions.json` and `tables/`. "
         "A practitioner gives chain of thought to the third of the panel predicted to gain most. Extra regret: "
         "the regret of a choice made under one baseline arm and deployed under the other, minus the regret of the "
         "choice made under the deployment arm on the same pilot sets (pp of accuracy gain per chosen model; 95% t "
         "interval over evaluation configurations; positive: the wrong baseline costs points).", "",
         "Every choice is one already saved before these validation draws were read for this purpose; the validation "
         "draws of these three benchmarks had been read by earlier analyses, so this values saved choices and is not a "
         "fresh test. Leave-one-family-out re-takes the top third of the models that remain from the same saved "
         "discovery predictions.", "",
         "## Headline: own gain, both directions", "",
         "| Benchmark | Design | Configs | Requested → forced | Forced → requested | Random third (requested; forced) |",
         "|---|---|---:|---|---|---|"]
    for t, T in R["tasks"].items():
        o = T["one_draw"]
        r = o["rules"]["own_1"]
        L.append(f"| {TITLE[t]} | one-draw pilots, all items | {o['n_configurations']} | {ci(r['requested_to_forced'])} | "
                 f"{ci(r['forced_to_requested'])} | {f2(o['random_regret']['requested'])}; "
                 f"{f2(o['random_regret']['forced'])} |")
        s = T["strict"]["designs"]["disjoint"]["strict"]
        r = s["rules"]["own_gain"]
        L.append(f"| {TITLE[t]} | 100-item pilots, item-disjoint | {s['n_configurations']} | "
                 f"{ci(r['requested_to_forced'])} | {ci(r['forced_to_requested'])} | "
                 f"{f2(s['random_regret']['requested'])}; {f2(s['random_regret']['forced'])} |")
    L += ["", "## Content scores beside the strict ones (own gain; the same saved choices)", "",
          "| Benchmark | Design | Strict: req → fd | Strict: fd → req | Content: req → fd | Content: fd → req |",
          "|---|---|---|---|---|---|"]
    for t, T in R["tasks"].items():
        o = T["one_draw"]
        if "content" in o:
            a, b = o["rules"]["own_1"], o["content"]["rules"]["own_1"]
            L.append(f"| {TITLE[t]} | one-draw pilots | {ci(a['requested_to_forced'])} | {ci(a['forced_to_requested'])} | "
                     f"{ci(b['requested_to_forced'])} | {ci(b['forced_to_requested'])} |")
        s = T["strict"]["designs"]["disjoint"]
        if "content" in s:
            a, b = s["strict"]["rules"]["own_gain"], s["content"]["rules"]["own_gain"]
            L.append(f"| {TITLE[t]} | 100-item pilots | {ci(a['requested_to_forced'])} | "
                     f"{ci(a['forced_to_requested'])} | {ci(b['requested_to_forced'])} | {ci(b['forced_to_requested'])} |")
    L += ["", "Content rows value the strict-scored choices on content-scored gains; no content-scored choice was saved "
          "for the forced arm, so the matched comparator is the deployment arm's strict-scored choice.", "",
          "## Leave one family out (own gain; range over the five families)", "",
          "| Benchmark | Design | Requested → forced | Forced → requested |", "|---|---|---|---|"]
    for t, T in R["tasks"].items():
        lo = T["one_draw"]["leave_one_family_out_own_1"]["range"]
        L.append(f"| {TITLE[t]} | one-draw pilots | {rng_txt(lo['requested_to_forced'])} | "
                 f"{rng_txt(lo['forced_to_requested'])} |")
        lo = T["strict"]["designs"]["disjoint"]["strict"]["leave_one_family_out_own_gain"]["range"]
        L.append(f"| {TITLE[t]} | 100-item pilots | {rng_txt(lo['requested_to_forced'])} | "
                 f"{rng_txt(lo['forced_to_requested'])} |")
    L += ["", "Per family: `tables/leave_one_family_out.csv`.", "",
          "## Every rule (one-draw pilots design, strict)", "",
          "| Benchmark | Rule | Requested → forced | Forced → requested |", "|---|---|---|---|"]
    for t, T in R["tasks"].items():
        for r, v in T["one_draw"]["rules"].items():
            L.append(f"| {TITLE[t]} | {r} | {ci(v['requested_to_forced'])} | {ci(v['forced_to_requested'])} |")
    L += ["", "## Every rule (100-item pilots, item-disjoint, strict)", "",
          "| Benchmark | Rule | Requested → forced | Forced → requested |", "|---|---|---|---|"]
    for t, T in R["tasks"].items():
        for r, v in T["strict"]["designs"]["disjoint"]["strict"]["rules"].items():
            L.append(f"| {TITLE[t]} | {ds.RULE_NAMES[r]} | {ci(v['requested_to_forced'])} | {ci(v['forced_to_requested'])} |")
    L += ["", "## Reversal share of the arm switch over models", "",
          "Model means over the 24 discovery draws (the definition of `results/arm_effect/`). Model bootstrap: models "
          f"drawn with replacement ({N_BOOT} replicates), pairs of a model with its own copy left out; family "
          "bootstrap: families drawn whole; leave one family out: range over the five families.", "",
          "| Benchmark | Scores | Reversed | Share | Model bootstrap 95% | Family bootstrap 95% | Leave one family out | "
          "Within-family pairs reversed |", "|---|---|---|---:|---|---|---|---|"]
    for t, T in R["tasks"].items():
        for sc, v in T["reversal"].items():
            L.append(f"| {TITLE[t]} | {sc} | {v['reversed']} of {v['pairs']} | {v['share']:.3f} | "
                     f"{rng_txt(v['model_bootstrap']['ci95'])} | {rng_txt(v['family_bootstrap']['ci95'])} | "
                     f"{rng_txt(v['leave_one_family_out_range'])} | {v['within_family']['reversed']} of "
                     f"{v['within_family']['pairs']} |")
    L += ["", "## Checks against the printed numbers and the reference values", ""]
    for c in R["checks"]:
        L.append(f"- {c}")
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def checks(R):
    out = []
    for t, T in R["tasks"].items():
        o = T["one_draw"]
        cp = o["check_printed"]
        out.append(f"{TITLE[t]}, one-draw pilots, requested → forced: printed {cp['printed']:.4f}, here {cp['here']:.4f} "
                   f"(difference {cp['abs_difference']:.2g}); {'reproduced' if cp['abs_difference'] < 1e-9 else 'MISMATCH'}.")
        fr = o["rules"]["own_1"]["forced_to_requested"]["estimate"]
        rf = o["rules"]["own_1"]["requested_to_forced"]["estimate"]
        ok = round(rf, 2) == REF_PILOT[t][0] and round(fr, 2) == REF_PILOT[t][1]
        out.append(f"{TITLE[t]}, one-draw pilots: reference {REF_PILOT[t][0]:.2f} / {REF_PILOT[t][1]:.2f}; here {rf:.2f} / {fr:.2f} "
                   f"(requested → forced / forced → requested); {'reproduced' if ok else 'MISMATCH'}.")
        s = T["strict"]["designs"]["disjoint"]["strict"]["rules"]["own_gain"]
        for k, (name, i) in {"requested_to_forced": ("requested → forced", 0),
                             "forced_to_requested": ("forced → requested", 1)}.items():
            want = REF_TABLE[t][i]
            got = (s[k]["estimate"], *(s[k].get("ci95") or [np.nan, np.nan]))
            ok = all(round(g, 2) == w for g, w in zip(got, want))
            out.append(f"{TITLE[t]}, 100-item pilots, {name}: reference {want[0]:.2f} [{want[1]:.2f}, {want[2]:.2f}]; here "
                       f"{got[0]:.2f} [{got[1]:.2f}, {got[2]:.2f}]; {'reproduced' if ok else 'MISMATCH'}.")
        lo = T["strict"]["designs"]["disjoint"]["strict"]["leave_one_family_out_own_gain"]["range"]["forced_to_requested"]
        ok = round(lo[0], 2) == REF_LOFO[t][0] and round(lo[1], 2) == REF_LOFO[t][1]
        out.append(f"{TITLE[t]}, 100-item pilots, forced → requested leaving one family out: reference "
                   f"{REF_LOFO[t][0]:.2f}–{REF_LOFO[t][1]:.2f}; here {lo[0]:.2f}–{lo[1]:.2f}; "
                   f"{'reproduced' if ok else 'MISMATCH'}.")
        for a, c in T["strict"]["check_same_baseline"].items():
            out.append(f"{TITLE[t]}, 100-item pilots, own-gain regret under {a} (the matched choice): printed "
                       f"{c['printed']:.4f}, here {c['here']:.4f}; "
                       f"{'reproduced' if c['abs_difference'] < 1e-9 else 'MISMATCH'}.")
        v = T["reversal"]["strict"]
        fb, fl, fw = REF_REV[t]
        b, l_ = v["model_bootstrap"]["ci95"], v["leave_one_family_out_range"]
        w = (v["within_family"]["reversed"], v["within_family"]["pairs"])
        out.append(f"{TITLE[t]}, reversal share {v['reversed']} of {v['pairs']}: reference model bootstrap "
                   f"{fb[0]:.2f}–{fb[1]:.2f}, here {b[0]:.2f}–{b[1]:.2f} "
                   f"({'same to two decimals' if (round(b[0], 2), round(b[1], 2)) == fb else 'differs; seeds and the handling of duplicated models are not stated in the reference'}); "
                   f"leave one family out reference {fl[0]:.2f}–{fl[1]:.2f}, here {l_[0]:.2f}–{l_[1]:.2f} "
                   f"({'reproduced' if (round(l_[0], 2), round(l_[1], 2)) == fl else 'MISMATCH'}); within-family reference "
                   f"{fw[0]} of {fw[1]}, here {w[0]} of {w[1]} ({'reproduced' if w == fw else 'MISMATCH'}).")
    return out


def tables(R):
    cost, lofo, rev = [], [], []
    for t, T in R["tasks"].items():
        o = T["one_draw"]
        blocks = [("one_draw", "strict", o["rules"])] + ([("one_draw", "content", o["content"]["rules"])] if "content" in o else [])
        for d, sc_blk in T["strict"]["designs"].items():
            for sc, blk in sc_blk.items():
                blocks.append((f"pilot100_{d}", sc, {ds.RULE_NAMES[r]: v for r, v in blk["rules"].items()}))
        for design, sc, rules in blocks:
            for r, v in rules.items():
                for k, x in v.items():
                    c = x.get("ci95") or [None, None]
                    cost.append({"task": t, "design": design, "scores": sc, "rule": r, "direction": k,
                                 "extra_regret": x["estimate"], "ci_lo": c[0], "ci_hi": c[1], "K": x["K"],
                                 "regret_mismatched": x["regret_mismatched"], "regret_matched": x["regret_matched"]})
        lb = [("one_draw", "strict", o["leave_one_family_out_own_1"])]
        if "content" in o:
            lb.append(("one_draw", "content", o["content"]["leave_one_family_out_own_1"]))
        for d, sc_blk in T["strict"]["designs"].items():
            for sc, blk in sc_blk.items():
                lb.append((f"pilot100_{d}", sc, blk["leave_one_family_out_own_gain"]))
        for design, sc, L in lb:
            for fam, x in L["by_family"].items():
                lofo.append({"task": t, "design": design, "scores": sc, "family_left_out": fam, **x})
        for sc, v in T["reversal"].items():
            rev.append({"task": t, "scores": sc, "pairs": v["pairs"], "reversed": v["reversed"], "share": v["share"],
                        "model_boot_lo": v["model_bootstrap"]["ci95"][0], "model_boot_hi": v["model_bootstrap"]["ci95"][1],
                        "family_boot_lo": v["family_bootstrap"]["ci95"][0],
                        "family_boot_hi": v["family_bootstrap"]["ci95"][1],
                        "lofo_min": v["leave_one_family_out_range"][0], "lofo_max": v["leave_one_family_out_range"][1],
                        "within_family_pairs": v["within_family"]["pairs"],
                        "within_family_reversed": v["within_family"]["reversed"],
                        **{f"lofo_{k}": x for k, x in v["leave_one_family_out"].items()}})
    return pd.DataFrame(cost), pd.DataFrame(lofo), pd.DataFrame(rev)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    srec, sarr = rbud.load_frozen(STRICT_FROZEN, STRICT_FROZEN[:-5] + ".npz")
    R = {"analysis": "both directions of the wrong-baseline decision cost",
         "created": time.strftime("%Y-%m-%d %H:%M"), "units": "pp of accuracy gain per chosen model",
         "saved_choices": {"one_draw": "manifests/predictions_decision_<task>_req|_fd.json",
                           "pilot100": {"path": rel(STRICT_FROZEN), "sha256": srec.get("sha256"),
                                        "created": srec.get("created")}},
         "n_boot": N_BOOT, "tasks": {}}
    for t in TASKS:
        facts = ri.load(t)["facts"]
        fams = facts["families"]
        for name in (f"{t}_req", f"{t}_fd"):
            m = json.load(open(os.path.join(MAN, f"predictions_decision_{name}.json"), encoding="utf-8"))["models"]
            if m != facts["panel"] or srec["design"]["models"][t] != facts["panel"]:
                raise SystemExit(f"{t}: the saved choices' panel differs from the panel the families belong to")
        R["tasks"][t] = {"title": TITLE[t], "families": fams, "one_draw": one_draw_design(t, fams),
                         "strict": strict_design(t, srec, sarr, fams), "reversal": reversal_block(t)}
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["checks"] = checks(R)
    R["runtime_s"] = round(time.time() - t0, 1)
    cost, lofo, rev = tables(R)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    cost.to_csv(os.path.join(OUT, "tables", "costs.csv"), index=False)
    lofo.to_csv(os.path.join(OUT, "tables", "leave_one_family_out.csv"), index=False)
    rev.to_csv(os.path.join(OUT, "tables", "reversal_share.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print("\n".join(summary(R)))
    return R


if __name__ == "__main__":
    main()
