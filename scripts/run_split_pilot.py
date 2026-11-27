r"""The split pilot, exploratory
on GSM8K-29, MATH-500 numeric and BIG-Bench Hard-28, and HotpotQA's saved choices (written at its discovery stage,
valued in its full run). Methods and rule text: src/split_pilot.py.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_split_pilot.py

Three benchmarks (exploratory; these validation draws had been read by earlier analyses): the item-disjoint 100-item pilots saved in manifests/predictions_decision_strict.*
(positions, items and the own-gain choices of the two single-baseline pilots), the split pilot's choices at the
assignment seed 20261121, at the 20 further seeds 0-19, and leaving one model family out at seed 20261121 (the top third
of the remaining models re-taken from the same predictions). Valued on the evaluation half of the items, validation
configurations absent from every discovery draw, each once (results/decision_strict's main set). The reference table below is
checked to two decimals; mismatches are reported as they are.

HotpotQA (called by scripts/run_hotpot.py): save_hotpot at the discovery stage writes manifests/predictions_hotpot_
split.json from the discovery table and the saved strict-design file (write-once: an identical file is kept, a
different one refused); evaluate_hotpot in the full run values the saved choices on the validation draws.

Outputs: results/split_pilot/split_pilot.json, summary.md, summary_clean.md, tables/regret.csv, tables/seeds.csv,
tables/leave_one_family_out.csv.
"""
from __future__ import annotations

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
from src import decision as dc, decision_strict as ds, rule_budgets as rbud, split_pilot as sp  # noqa: E402

OUT = os.path.join(ROOT, "results", "split_pilot")
# the 100-item pilots saved by scripts/run_decision_strict.py, in manifests/ (moved unchanged
# from results/decision_strict/frozen/)
STRICT_FROZEN = os.path.join(ROOT, "manifests", "predictions_decision_strict.json")
TASKS = ("gsm8k", "math", "bbh")
TITLE = {"gsm8k": "GSM8K-29", "math": "MATH-500 numeric", "bbh": "BIG-Bench Hard-28"}
# Reference table (requested-deployment / forced-deployment regret): requested-only, forced-only, split
REF_TABLE = {"gsm8k": ((0.56, 2.78), (8.99, 0.72), (1.21, 1.83)),
         "math": ((0.52, 3.14), (13.27, 0.58), (0.65, 2.46)),
         "bbh": ((0.96, 3.75), (3.25, 1.10), (1.60, 1.89))}
REF_LOFO_MIN = 0.40
HOTPOT_SPLIT = "predictions_hotpot_split.json"
HQ_STATEMENT = (
    "Primary outcome: the split pilot's worst regret (the larger of its requested-deployment and forced-deployment "
    "regrets) against the worst regret of the better single-baseline pilot (whichever of the requested-only and "
    "forced-only pilots has the smaller worst regret, chosen with hindsight), valued on HotpotQA's validation "
    "configurations absent from every discovery draw (each once), on the evaluation half of the items, strict exact "
    "match. The statement: the split pilot's worst regret is lower. Both component regrets of every policy are "
    "reported, with the paired differences and their t intervals over evaluation configurations; all distinct "
    "validation configurations and leave-one-family-out are reported beside it. Equal worst regrets are reported as "
    "equal, and the statement as not holding.")


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ============================================================================ the three benchmarks (exploratory)


def task_block(task, srec, sarr):
    import robustness_inputs as ri
    import run_decision_strict as rds
    x = rds.task_inputs(task)
    if x["panel"] != srec["design"]["models"][task] or x["disc"] != srec["design"]["discovery_units"][task]:
        raise SystemExit(f"{task}: the panel or discovery draws differ from the saved 100-item pilots")
    fams = ri.load(task)["facts"]["families"]
    if ri.load(task)["facts"]["panel"] != x["panel"]:
        raise SystemExit(f"{task}: the family list belongs to another panel")
    N, M = len(x["items"]), len(x["panel"])
    n = dc.third_size(M)
    pre = f"{task}__disjoint__"
    pos, items = sarr[pre + "positions"].astype(int), sarr[pre + "items"].astype(int)
    configs, Sd = rds.dense(x, x["disc"])
    ci = {c: i for i, c in enumerate(configs)}
    rows = np.array([ci[x["disc"][p]] for p in pos])
    # the single-baseline pilots: the saved own-gain predictions must reproduce from the item scores
    single = {"requested_only": sp.single_gains(Sd["requested"], Sd["cot"], rows, items),
              "forced_only": sp.single_gains(Sd["forced"], Sd["cot"], rows, items)}
    worst = 0.0
    for p, b in (("requested_only", "requested"), ("forced_only", "forced")):
        worst = max(worst, float(np.max(np.abs(single[p] - sarr[pre + f"{b}__own_gain__pred"]))))
    if worst > 1e-8:
        raise SystemExit(f"{task}: single-baseline gains differ from the saved predictions by {worst:.3g} pp")
    n_strata = 25 if task == "bbh" else 1
    if task == "bbh" and len(x["strata"]) != 25:
        raise SystemExit("BIG-Bench Hard: expected 25 task strata")
    mask = ds.split_items(x["items"], task, rds.SPLIT_LABEL, x["strata"])
    w = ds.eval_weights(N, ~mask, x["strata"])
    vconfigs, Sv = rds.dense(x, x["val"])
    G = {d: rds.evaluation_gains(Sv, vconfigs, x["val"], x["disc"], d, w)["fresh"][0] for d in sp.DEPLOYMENTS}
    W = {"requested_only": sarr[pre + "requested__own_gain__weights"], "forced_only": sarr[pre + "forced__own_gain__weights"]}

    def run(seed):
        rq = sp.assignment(len(pos), items.shape[1], seed, n_strata)
        pr = sp.split_gains(Sd["requested"], Sd["forced"], Sd["cot"], rows, items, rq)
        Ws, T = sp.choices(pr, n)
        return pr, Ws, T, rq

    pr, Ws, T, rq = run(sp.SEED)
    main = sp.compare({**W, "split": Ws}, G, n)
    main["ties_split"] = T
    seeds = []
    for s in sp.CHECK_SEEDS:
        _, Wx, _, _ = run(s)
        c = sp.compare({**W, "split": Wx}, G, n)
        seeds.append({"seed": s, **{f"{p}/{d}": c["regret"][p][d] for p in sp.POLICIES for d in ("requested", "forced")},
                      "split_worst": c["split_worst"], "better_single_worst": c["better_single_worst"],
                      "advantage": c["advantage"], "split_better": c["split_better"]})
    pred = {"requested_only": single["requested_only"], "forced_only": single["forced_only"], "split": pr}
    lofo = sp.leave_one_family_out(pred, G, fams)
    # the reference table, to two decimals
    got = tuple(tuple(round(main["regret"][p][d], 2) for d in ("requested", "forced")) for p in sp.POLICIES)
    adv = [s["advantage"] for s in seeds]
    return {"title": TITLE[task], "M": M, "n_chosen": n, "n_pilots": int(len(pos)), "n_items_pilot": int(items.shape[1]),
            "evaluation_items": int((~mask).sum()), "n_configurations": int(G["requested"].shape[0]),
            "assignment": {"seed": sp.SEED, "n_strata": n_strata, "requested_items_per_pilot": int(rq.sum(1)[0]),
                           "balanced_every_pilot": bool((rq.sum(1) == items.shape[1] // 2).all())},
            "main": main, "seeds": seeds,
            "seeds_summary": {"n": len(seeds), "split_better": int(sum(s["split_better"] for s in seeds)),
                              "advantage_range": [float(min(adv)), float(max(adv))]},
            "leave_one_family_out": lofo,
            "ref_check": {"reference": [list(v) for v in REF_TABLE[task]], "here": [list(v) for v in got],
                            "reproduced": got == REF_TABLE[task],
                            "mismatches": [f"{p}/{d}: reference {REF_TABLE[task][i][j]}, here {got[i][j]}"
                                           for i, p in enumerate(sp.POLICIES) for j, d in enumerate(("requested", "forced"))
                                           if got[i][j] != REF_TABLE[task][i][j]]}}


# ============================================================================ HotpotQA


def hotpot_record(Sd, disc, panel, cfg_index, srec, sarr, source: dict) -> dict:
    """The saved split-pilot choices for HotpotQA from its discovery item scores Sd {arm: (C, M, N)} and the strict
    design's saved disjoint pilots (srec, sarr: manifests/predictions_hotpot_decision_strict.*)."""
    from src.baseline_mixture import sha256_json
    pre = "hotpot__disjoint__"
    pos, items = sarr[pre + "positions"].astype(int), sarr[pre + "items"].astype(int)
    M = len(panel)
    n = dc.third_size(M)
    if int(srec["n_chosen"]) != n:
        raise SystemExit("the strict design's third differs from decision.third_size")
    rows = np.array([cfg_index[str(disc[p])] for p in pos])
    single = {"requested_only": sp.single_gains(Sd["direct_answer"], Sd["zero_shot_cot"], rows, items),
              "forced_only": sp.single_gains(Sd["forced_direct"], Sd["zero_shot_cot"], rows, items)}
    for p, c in (("requested_only", "cot_req"), ("forced_only", "cot_forced")):
        d = float(np.max(np.abs(single[p] - sarr[pre + f"{c}__own_gain__pred"])))
        if d > 1e-8:
            raise SystemExit(f"HotpotQA {p}: gains differ from the saved strict-design predictions by {d:.3g} pp")
    rq = sp.assignment(len(pos), items.shape[1], sp.SEED, 1)
    pr = sp.split_gains(Sd["direct_answer"], Sd["forced_direct"], Sd["zero_shot_cot"], rows, items, rq)
    Ws, T = sp.choices(pr, n)
    W = {"requested_only": sarr[pre + "cot_req__own_gain__weights"], "forced_only": sarr[pre + "cot_forced__own_gain__weights"],
         "split": Ws}
    body = {"label": "hotpot_split", "units": "pp of gain; weights per model (ties share places)", "rule": sp.RULE_TEXT,
            "statement": HQ_STATEMENT, "assignment_seed": sp.SEED, "n_chosen": n, "models": list(panel),
            "n_pilots": int(len(pos)), "n_items_pilot": int(items.shape[1]),
            "discovery_units": list(map(str, disc)),
            "pilots": {"from": "manifests/predictions_hotpot_decision_strict.json, design 'disjoint'",
                       "strict_sha256": srec.get("sha256"), "positions": pos.tolist(), "items": items.tolist()},
            "assignment_requested": rq.astype(int).tolist(),
            "ties_at_cut": {"split": T},
            "predictions": {p: np.round(v, 10).tolist() for p, v in
                            (("requested_only", single["requested_only"]), ("forced_only", single["forced_only"]),
                             ("split", pr))},
            "weights": {p: np.round(v, 12).tolist() for p, v in W.items()},
            "source": source,
            "note": "read only HotpotQA discovery-draw outputs and the strict design's saved discovery file; written before "
                    "any validation draw was scored. The split pilot was found exploratively on "
                    "GSM8K, MATH-500 and BIG-Bench Hard (results/split_pilot/); this is its forward test."}
    body["sha256"] = sha256_json({k: v for k, v in body.items() if k not in ("created", "sha256", "source")})
    return body


def save_hotpot(path, disc, panel, items, pred_dir) -> dict:
    """Write manifests/predictions_hotpot_split.json at HotpotQA's discovery stage (after hotpot_fair.save_discovery)."""
    import hotpot_fair as hf
    from src import baseline_mixture as bm
    srec, sarr = rbud.load_frozen(*hf.files(hf.STRICT, pred_dir))
    if list(srec["models"]) != list(panel):
        raise SystemExit(f"{hf.STRICT} was saved for another panel")
    configs, Sd = hf.dense(path, panel, disc, items)
    meta = path[:-len(".parquet")] + ".json"
    source = {"scored_discovery": rel(path) if os.path.abspath(path).startswith(ROOT) else path,
              "scored_discovery_sha256": (json.load(open(meta, encoding="utf-8")).get("sha256")
                                          if os.path.exists(meta) else None), "score": "strict exact match"}
    rec = hotpot_record(Sd, disc, panel, {c: i for i, c in enumerate(configs)}, srec, sarr, source)
    return bm.write_forecast(rec, os.path.join(pred_dir, HOTPOT_SPLIT))


def evaluate_hotpot(pred_dir, panel, fams, disc, val, Sv: dict, vconfigs, items_n: int) -> dict:
    """The saved split-pilot choices on HotpotQA's validation configurations (Sv {arm: (C, M, N)} over vconfigs, the
    distinct validation configurations; scripts/hotpot_fair.dense). Every statement reported, misses included."""
    import hotpot_fair as hf
    from src import baseline_mixture as bm
    rec = bm.load_forecast(os.path.join(pred_dir, HOTPOT_SPLIT))
    if list(rec["models"]) != list(panel):
        raise SystemExit(f"{HOTPOT_SPLIT} was saved for another panel")
    srec, sarr = rbud.load_frozen(*hf.files(hf.STRICT, pred_dir))
    if srec.get("sha256") != rec["pilots"]["strict_sha256"]:
        raise SystemExit(f"{HOTPOT_SPLIT} was built on another strict-design file")
    mask = np.asarray(sarr["item_split_pilot_pool"], bool)
    if len(mask) != items_n:
        raise SystemExit("the item set differs from the one the pilots were drawn on")
    w = ds.eval_weights(items_n, ~mask)
    n = int(rec["n_chosen"])
    Wd = {p: np.asarray(rec["weights"][p], float) for p in sp.POLICIES}
    pred = {p: np.asarray(rec["predictions"][p], float) for p in sp.POLICIES}
    ci_ = {c: i for i, c in enumerate(vconfigs)}
    sets = hf.eval_sets(disc, val)
    out = {"saved": {"path": f"manifests/{HOTPOT_SPLIT}", "sha256": rec["sha256"], "created": rec.get("created")},
           "statement": rec["statement"], "units": "pp of gain per chosen model", "sets": {}}
    for ev, rows_ in sets.items():
        cs = [str(val[r]) for r in rows_]
        if len(cs) < 2:
            out["sets"][ev] = {"n_configurations": len(cs), "note": "fewer than two evaluation configurations"}
            continue
        ix = [ci_[c] for c in cs]
        G = {"requested": (Sv["zero_shot_cot"][ix].astype(float) - Sv["direct_answer"][ix].astype(float)) @ w,
             "forced": (Sv["zero_shot_cot"][ix].astype(float) - Sv["forced_direct"][ix].astype(float)) @ w}
        c = sp.compare(Wd, G, n)
        c["n_configurations"] = len(cs)
        c["configurations"] = cs
        c["leave_one_family_out"] = sp.leave_one_family_out(pred, G, fams)
        out["sets"][ev] = c
    m = out["sets"].get("fresh") or {}
    out["primary"] = {"set": "fresh", "split_worst": m.get("split_worst"), "better_single": m.get("better_single"),
                      "better_single_worst": m.get("better_single_worst"), "advantage": m.get("advantage"),
                      "holds": m.get("split_better")}
    return out


def hotpot_lines(E: dict) -> list:
    p = E["primary"]
    L = ["", "## Split pilot (saved before validation was scored)", "",
         f"Saved choices `{E['saved']['path']}` (sha256 `{E['saved']['sha256'][:16]}…`, {E['saved']['created']})."]
    for ev, c in E["sets"].items():
        if "regret" not in c:
            L.append(f"- {ev}: {c.get('note')}.")
            continue
        r = c["regret"]
        L.append(f"- {ev} ({c['n_configurations']} configurations): requested-only {f2(r['requested_only']['requested'])} / "
                 f"{f2(r['requested_only']['forced'])}; forced-only {f2(r['forced_only']['requested'])} / "
                 f"{f2(r['forced_only']['forced'])}; split {f2(r['split']['requested'])} / {f2(r['split']['forced'])} "
                 f"(requested / forced deployment regret, pp). Worst regret: better single ({c['better_single']}) "
                 f"{f2(c['better_single_worst'])}, split {f2(c['split_worst'])}; advantage {f2(c['advantage'])}. "
                 f"Leave one family out: split better in {c['leave_one_family_out']['split_better_count']} of "
                 f"{c['leave_one_family_out']['families']}.")
    L.append(f"- Primary statement (fresh configurations): split worst regret lower; holds: {p['holds']}.")
    return L


# ============================================================================ report


def f2(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}".replace("-", "−")


def summary(R):
    L = ["# The split pilot", "",
         "**Exploratory on these three benchmarks.** The result was found by replaying the saved item-disjoint pilots of "
         "`results/decision_strict/`; these validation "
         "draws had been read by earlier analyses. The forward test is HotpotQA, whose split-pilot choices are saved "
         "at its discovery stage before its validation draws are scored (`manifests/predictions_hotpot_split.json`). "
         "Code `scripts/run_split_pilot.py`, `src/split_pilot.py`.", "",
         "Each model gets 100 chain-of-thought calls and 100 baseline calls per pilot (one discovery draw, 100 items "
         "from the pilot half of the items; 200 pilots). Requested-only and forced-only pilots spend the baseline calls "
         "on one baseline; the split pilot assigns half of the pilot's items to each baseline, the same assignment for "
         "every model (two of four items per BIG-Bench Hard task), assignment seed 20261121. The top third by mean paired "
         "gain gets chain of thought. Regret: best third minus chosen third, pp of gain per chosen model, on the other "
         "half of the items and the validation configurations absent from every discovery draw, averaged over pilots.",
         "", "## Regret under each deployment baseline (requested / forced)", "",
         "| Benchmark | Configs | Requested-only pilot | Forced-only pilot | Split pilot | Worst regret: better single → split |",
         "|---|---:|---|---|---|---|"]
    for t, T in R["tasks"].items():
        r = T["main"]["regret"]
        L.append(f"| {T['title']} | {T['n_configurations']} | "
                 + " | ".join(f"{f2(r[p]['requested'])} / {f2(r[p]['forced'])}" for p in sp.POLICIES)
                 + f" | {f2(T['main']['better_single_worst'])} → {f2(T['main']['split_worst'])} |")
    L += ["", "## Paired differences, split minus each single-baseline pilot (pp; 95% t interval over configurations; "
          "negative: the split costs less)", "",
          "| Benchmark | vs requested-only, requested deployment | vs requested-only, forced deployment | "
          "vs forced-only, requested deployment | vs forced-only, forced deployment |", "|---|---|---|---|---|"]

    def ci(d):
        c = d.get("ci95")
        return f"{f2(d['estimate'])} [{f2(c[0])}, {f2(c[1])}]" if c else f2(d["estimate"])
    for t, T in R["tasks"].items():
        pp = T["main"]["paired"]
        L.append(f"| {T['title']} | " + " | ".join(ci(pp[f"split_minus_{p}/{d}"]) for p in ("requested_only", "forced_only")
                                                   for d in ("requested", "forced")) + " |")
    L += ["", "## Robustness", ""]
    for t, T in R["tasks"].items():
        s, lo = T["seeds_summary"], T["leave_one_family_out"]
        L.append(f"- {T['title']}: 20 further assignment seeds (0–19): split better in {s['split_better']} of {s['n']}, "
                 f"advantage {f2(s['advantage_range'][0])} to {f2(s['advantage_range'][1])} pp. Leave one family out "
                 f"(seed 20261121): split better in {lo['split_better_count']} of {lo['families']}, advantage "
                 f"{f2(lo['advantage_range'][0])} to {f2(lo['advantage_range'][1])} pp. Ties at the cut in the split "
                 f"pilot's choices: {T['main']['ties_split']} of {T['n_pilots']} pilots.")
    allfam = [v["advantage"] for T in R["tasks"].values() for v in T["leave_one_family_out"]["by_family"].values()
              if "advantage" in v]
    L += [f"- Smallest advantage over all leave-one-family-out comparisons: {f2(min(allfam))} pp (reference: "
          f"{REF_LOFO_MIN:.2f}).", "", "## Check against the reference table", ""]
    for t, T in R["tasks"].items():
        a = T["ref_check"]
        L.append(f"- {T['title']}: reference " + "; ".join(f"{x[0]:.2f} / {x[1]:.2f}" for x in a["reference"]) + "; here "
                 + "; ".join(f"{x[0]:.2f} / {x[1]:.2f}" for x in a["here"])
                 + ("; reproduced." if a["reproduced"] else "; differs: " + "; ".join(a["mismatches"]) + "."))
    L += ["", f"Runtime {R['runtime_s']:.0f} s."]
    return L


def tables(R):
    reg, sd, lf = [], [], []
    for t, T in R["tasks"].items():
        for p in sp.POLICIES:
            for d in ("requested", "forced", "worst"):
                reg.append({"task": t, "policy": p, "deployment": d, "regret": T["main"]["regret"][p][d]})
        for s in T["seeds"]:
            sd.append({"task": t, **s})
        for f, v in T["leave_one_family_out"]["by_family"].items():
            if "regret" in v:
                lf.append({"task": t, "family_left_out": f, "models_left": v["models_left"], "n_chosen": v["n_chosen"],
                           **{f"{p}/{d}": v["regret"][p][d] for p in sp.POLICIES for d in ("requested", "forced")},
                           "better_single": v["better_single"], "advantage": v["advantage"],
                           "split_better": v["split_better"]})
    return pd.DataFrame(reg), pd.DataFrame(sd), pd.DataFrame(lf)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import robustness_inputs as ri
    t0 = time.time()
    srec, sarr = rbud.load_frozen(STRICT_FROZEN, STRICT_FROZEN[:-5] + ".npz")
    R = {"analysis": "the split pilot; exploratory on these three benchmarks",
         "created": time.strftime("%Y-%m-%d %H:%M"), "rule": sp.RULE_TEXT, "units": "pp of gain per chosen model",
         "saved_pilots": {"path": rel(STRICT_FROZEN), "sha256": srec.get("sha256"), "created": srec.get("created")},
         "tasks": {}}
    for t in TASKS:
        R["tasks"][t] = task_block(t, srec, sarr)
        print(f"{t}: done ({time.time() - t0:.0f} s)", flush=True)
    R["runtime_s"] = round(time.time() - t0, 1)
    reg, sd, lf = tables(R)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    reg.to_csv(os.path.join(OUT, "tables", "regret.csv"), index=False)
    sd.to_csv(os.path.join(OUT, "tables", "seeds.csv"), index=False)
    lf.to_csv(os.path.join(OUT, "tables", "leave_one_family_out.csv"), index=False)
    ri.write_outputs(OUT, R, summary(R))
    print("\n".join(summary(R)))
    return R


if __name__ == "__main__":
    main()
