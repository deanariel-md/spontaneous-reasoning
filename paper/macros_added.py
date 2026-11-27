"""Macros of the added results, for paper/make_numbers.py.

Each function takes a results file already loaded (a dict) and returns {macro name: formatted value}; nothing is
re-estimated. Prefixes:
    fewshot   five-shot against zero-shot forced direct (results/fewshot/, results/fewshot/math/)
    gc        chain of thought against the constrained forced arm, BIG-Bench Hard (results/forced_constrained/)
    tv        native reasoning under the vendors' sampling settings (results/thinking_vendor/)
    dec       decision analysis, every setting (results/decision/)
    dstrict   decision comparison made strict (results/decision_strict/)
    tregret   transfer regret (results/transfer_regret/)
    pceil     protocol ceiling (results/protocol_ceiling/)
    pubre     published claims recomputed (results/published_recompute/)
The Steps 1-3 files of the five-shot and constrained contrasts get the standard Dataset 2 macros in make_numbers.py
(prefixes fewshotGsm, fewshotBbh, fewshotMath, gcBbh). Undefined values print "--"; negative values keep their sign.
"""
import json
import math
import os
import statistics

TASK = {"gsm8k": "Gsm", "bbh": "Bbh", "math": "Math", "hotpot": "Hq"}
TOKEN = {"gsm8k": "Gsm", "bbh": "Bbh", "math": "Math", "req": "Req", "fd": "Fd", "con": "Con", "exemplar": "Exemplar",
         "dove": "Dove", "mmlu": "Mmlu", "fewshot": "Fs", "requested": "Req", "forced": "Forced", "hotpot": "Hq"}
CHECKPOINT = {"qwen3.5-35b-a3b": "QwenFiveMoe", "qwen3.6-27b": "QwenSixDense", "qwen3.6-35b-a3b": "QwenSixMoe"}


def num(x, nd=2):
    """x to nd decimals with a math-mode minus; "--" for None, NaN or infinity (an undefined value, as it is)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    if round(v, nd) == 0:
        v = 0.0
    return f"{v:.{nd}f}".replace("-", "$-$")


def tag_of(key: str) -> str:
    """A results key (gsm8k_req, fewshot_math, gsm8k->bbh/forced) as letters: GsmReq, FsMath, GsmToBbhForced."""
    out = []
    for part in key.replace("->", "_to_").replace("/", "_").split("_"):
        if part in TOKEN:
            out.append(TOKEN[part])
        elif part:
            out.append("".join(ch for ch in part.capitalize() if ch.isalpha()))
    return "".join(out)


def _check(m):
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    return m


def _ci(m, stem, block, nd=2):
    c = (block or {}).get("ci95")
    m[stem + "Lo"] = num(c[0], nd) if c else "--"
    m[stem + "Hi"] = num(c[1], nd) if c else "--"


# ---------------------------------------------------------------- five-shot


def fewshot_macros(results, steps=None):
    """results: the fewshot.json files (GSM8K and BIG-Bench Hard; MATH-500); steps: {task: Steps 1-3 file}, for the
    one-draw budget comparison of own gain against the averaged capability rule."""
    m = {}
    for R in results:
        if R.get("dry_run"):
            continue
        for task, T in R["tasks"].items():
            t = "fewshot" + TASK[task]          # M, C, H, R, W and V_delta to 0 dp: d2_macros on the Steps 1-3 file
            m[t + "Items"] = f"{T['n_items']:,}".replace(",", "{,}")
            m[t + "MeanZero"] = num(T["means"]["zero_shot_forced"], 1)
            m[t + "MeanFive"] = num(T["means"]["five_shot_forced"], 1)
            m[t + "MeanGainFine"] = num(T["means"]["gain"], 2)
            s2 = T.get("step2") or {}
            vd = s2.get("V_delta") or {}
            m[t + "VdeltaFine"] = num(vd.get("estimate"), 2)
            _ci(m, t + "VdeltaFine", vd, 2)
            pt, ob = T["map"]["point"]["benefit"], T["map"]["validation_one_draw"]["benefit"]
            m[t + "MapRbenFine"] = num(pt.get("R"), 3)
            m[t + "MapCval"] = num(ob.get("C"), 3)
            cs = T["capability_slope"]
            m[t + "Slope"] = num(cs["slope"], 3)
            _ci(m, t + "SlopeModel", cs.get("model_bootstrap"), 2)
            _ci(m, t + "SlopeFamily", cs.get("family_bootstrap"), 2)
            mb, fb = (cs.get(k) or {} for k in ("model_bootstrap", "family_bootstrap"))
            both = [b.get("ci95") for b in (mb, fb)]
            m[t + "SlopeIntervalsHoldZero"] = "yes" if all(c and c[0] <= 0 <= c[1] for c in both) else "no"
            st = (steps or {}).get(task)
            if st:
                one = [c for c in st["step3"].get("budget_curve", [])
                       if c["K"] == 1 and c["f"] == "own_gain" and c["g"] == "averaged_ols"]
                if one:
                    m[t + "BudgetOneOwnOls"] = num(one[0]["estimate"], 2)
                    _ci(m, t + "BudgetOneOwnOls", one[0], 2)
    return _check(m)


# ---------------------------------------------------------------- constrained forced arm


def constrained_macros(R, steps=None):
    """R: results/forced_constrained/forced_constrained.json; steps: its Steps 1-3 file (lenient sensitivity)."""
    m = {}
    if R.get("dry_run"):
        return m
    rows = R["per_model"]
    m["gcM"] = str(len(R["panel"]))
    for a, s in (("requested", "Requested"), ("forced", "Forced"), ("constrained", "Constrained"), ("cot", "Cot")):
        m["gcMean" + s] = num(sum(r[a] for r in rows) / len(rows), 1)
    for arm, s in (("forced", "Forced"), ("constrained", "Constrained")):
        m["gcReadable" + s] = str(len(R["panels"][arm]["readable"]))
        vals = [r[f"{arm}_closed_and_readable_pct"] for r in rows if r.get(f"{arm}_closed_and_readable_pct") is not None]
        m["gcReadablePctMin" + s] = num(min(vals), 1) if vals else "--"
    for outcome, s in (("cot_minus_requested", "Req"), ("cot_minus_forced", "Forced"),
                       ("cot_minus_constrained", "Constrained")):
        for name, b in R["slopes"][outcome].items():
            stem = "gcSlope" + s + name.capitalize()
            m[stem + "M"] = str(b["M"])
            if "slope" not in b:
                m[stem] = "--"
                continue
            m[stem] = num(b["slope"], 2)
            _ci(m, stem + "Model", b.get("model_bootstrap"), 2)
            _ci(m, stem + "Family", b.get("family_bootstrap"), 2)
    if steps:
        m["gcBbhMeanGainFine"] = num(steps["step2"]["mean_gain"]["estimate"], 2)
        le = (steps.get("sensitivities") or {}).get("lenient_extraction", {}).get("steps_1_2")
        if le:
            m["gcBbhLenientMeanGain"] = num(le["mean_gain"], 2)
            m["gcBbhLenientCben"] = num(le["C_benefit"], 3)
            m["gcBbhLenientVdelta"] = num(le["V_delta"], 1)
            m["gcBbhLenientH"] = num(le["H"], 2)
    return _check(m)


# ---------------------------------------------------------------- vendor sampling


def vendor_macros(R):
    """results/thinking_vendor/thinking_vendor.json: per task and checkpoint, the cut and accuracy shares under the
    vendor settings and greedy decoding, the paired differences, and the ranges over checkpoints."""
    m = {}
    rng = {}

    def keep(key, v):
        if v is not None and math.isfinite(v):
            rng.setdefault(key, []).append(v)

    for cell, c in R["cells"].items():
        task, ck = cell.split("/")
        t = "tv" + TASK[task] + CHECKPOINT.get(ck, tag_of(ck))
        for dec, s in (("vendor", "Vendor"), ("greedy", "Greedy")):
            d = c[dec]
            m[t + "Cut" + s] = num(d["cut"]["pct"], 1)
            m[t + "Acc" + s] = num(d["strict"]["pct"], 1)
            m[t + "Tokens" + s] = f"{d['mean_output_tokens']:,.0f}".replace(",", "{,}")
            keep("Cut" + s, d["cut"]["pct"])
        for k, s in (("diff_vendor_minus_greedy", "VendorMinusGreedy"), ("diff_vendor_minus_cot", "VendorMinusCot"),
                     ("diff_greedy_minus_cot", "GreedyMinusCot"), ("diff_vendor_minus_forced", "VendorMinusForced")):
            d = c.get(k)
            if d:
                m[t + s] = num(d["mean"], 1)
                m[t + s + "Lo"], m[t + s + "Hi"] = num(d["lo"], 1), num(d["hi"], 1)
                keep(s, d["mean"])
                keep(TASK[task] + s, d["mean"])
        for k, s in (("acc_cot", "Cot"), ("acc_forced", "Forced")):
            if c.get(k):
                m[t + "Acc" + s] = num(c[k]["pct"], 1)
    for key, v in rng.items():
        m["tv" + key + "Min"], m["tv" + key + "Max"] = num(min(v), 1), num(max(v), 1)
    return _check(m)


# ---------------------------------------------------------------- decision analysis

RULES = (("capability_1", "CapOne"), ("own_1", "OwnOne"), ("own_4", "OwnFour"), ("own_all", "OwnAll"),
         ("capability_all", "CapAll"), ("random", "Rand"))


def decision_macros(R):
    """results/decision/decision.json: per setting the chosen count, the regret of the main rules (plug-in and
    cross-fitted, independent evaluation configurations) and the one-draw capability rule minus own gain."""
    m = {}
    if R.get("dry_run"):
        return m
    for key, s in R["settings"].items():
        t = "dec" + tag_of(key)
        r = s["result"]
        ev = r["evaluation"]["independent"]
        m[t + "M"], m[t + "Chosen"] = str(r["M"]), str(r["n_chosen"])
        m[t + "Configs"] = str(ev["n_configurations"])
        m[t + "BestThird"] = num(ev.get("best_third_value"), 2)
        m[t + "PanelGain"] = num(ev.get("panel_mean_gain"), 2)
        for rule, rt in RULES:
            x = ev["rules"].get(rule)
            if x is None:
                continue
            m[t + rt + "Plug"] = num(x["regret_plugin"], 2)
            m[t + rt + "Cross"] = num(x["regret_crossfit"], 2)
        p = next((x for x in ev.get("paired", []) if x["a"] == "capability_1" and x["b"] == "own_1"), None)
        if p:
            m[t + "CapMinusOwn"] = num(p["estimate"], 2)
            _ci(m, t + "CapMinusOwn", p, 2)
    return _check(m)


# ---------------------------------------------------------------- strict decision comparison

STRICT_RULES = (("linear", "Lin"), ("quadratic_index", "Quad"), ("own_gain", "Own"), ("random", "Rand"))


def strict_macros(R, evaluation="fresh"):
    """results/decision_strict/decision_strict.json: per task, design and baseline arm, the plug-in regret of each
    rule on the fresh validation configurations and own gain minus the other rules, with 95% intervals."""
    m = {}
    for task, T in R["tasks"].items():
        for design, res in T["results"].items():
            for arm, by in res.items():
                t = "dstrict" + TASK[task] + design.capitalize() + TOKEN.get(arm, arm.capitalize())
                x = by[evaluation]
                m[t + "Configs"] = str(x["n_configurations"])
                for rule, rt in STRICT_RULES:
                    if rule in x["rules"]:
                        m[t + rt] = num(x["rules"][rule]["regret_plugin"], 2)
                for k, kt in (("own_gain_minus_quadratic_index", "OwnMinusQuad"), ("own_gain_minus_linear",
                                                                                    "OwnMinusLin"),
                              ("own_gain_minus_random", "OwnMinusRand")):
                    p = (x.get("paired") or {}).get(k)
                    if p:
                        m[t + kt] = num(p["estimate"], 2)
                        _ci(m, t + kt, p, 2)
    ac = R.get("ref_check") or {}
    if ac:
        m["dstrictRefReproduced"] = str(sum(bool(v.get("reproduced")) for v in ac.values()))
        m["dstrictRefTasks"] = str(len(ac))
    return _check(m)


# ---------------------------------------------------------------- transfer regret

FAMILY_RULES = (("transport_linear", "Lin"), ("transport_quadratic_index", "Quad"), ("transport_index", "Index"),
                ("target_own_1", "OwnOne"), ("random", "Rand"))
TASK_RULES = (("source_own_all", "SrcOwnAll"), ("source_own_1", "SrcOwnOne"), ("source_linear", "SrcLin"),
              ("source_quadratic_index", "SrcQuad"), ("target_own_1", "TgtOwnOne"), ("random", "Rand"))


def transfer_macros(R):
    """results/transfer_regret/transfer_regret.json: plug-in regret per case and rule, the ranges over cases, and the
    number of task cases where the source's linear rule regrets more than a random third."""
    m = {}
    for part, rules, stem in (("family", FAMILY_RULES, "tregretFam"), ("task", TASK_RULES, "tregretTask")):
        rng = {}
        for key, x in R[part].items():
            t = stem + tag_of(key)
            for rule, rt in rules:
                r = x["rules"].get(rule)
                if r is None:
                    continue
                m[t + rt] = num(r["regret_plugin"], 2)
                rng.setdefault(rt, []).append(r["regret_plugin"])
        for rt, v in rng.items():
            m[stem + rt + "Min"], m[stem + rt + "Max"] = num(min(v), 2), num(max(v), 2)
        m[stem + "Cases"] = str(len(R[part]))
    worse = [k for k, x in R["task"].items()
             if x["rules"].get("source_linear") and x["rules"].get("random")
             and x["rules"]["source_linear"]["regret_plugin"] > x["rules"]["random"]["regret_plugin"]]
    m["tregretTaskLinWorseThanRandom"] = str(len(worse))
    return _check(m)


TASK_RULES_ADDED = TASK_RULES + (("source_linear_at_source", "SrcLinAtSrc"),
                                 ("source_quadratic_index_at_source", "SrcQuadAtSrc"), ("source_index", "SrcIndex"))


def transfer_added_macros(R):
    """results/transfer_regret/transfer_regret.json, keys added_family and added_task (an added task as target,
    HotpotQA): plug-in regret per case and rule, named as transfer_macros names the main cases
    (tregretFamHqReqOwnOne, tregretTaskGsmToHqReqSrcLin, ...), and the paired difference against the target's own
    one-draw gain with its interval (...MinusOwn, Lo, Hi). The ranges and counts of transfer_macros keep the twelve
    main cases."""
    m = {}
    for part, rules, stem in (("added_family", FAMILY_RULES, "tregretFam"), ("added_task", TASK_RULES_ADDED,
                                                                            "tregretTask")):
        for key, x in (R.get(part) or {}).items():
            t = stem + tag_of(key)
            for rule, rt in rules:
                r = x["rules"].get(rule)
                if r is None:
                    continue
                _set(m, t + rt, num(r["regret_plugin"], 2))
                p = (x.get("paired_minus_target_own_1") or {}).get(rule)
                if p:
                    _set(m, t + rt + "MinusOwn", num(p["estimate"], 2))
                    _lohi(m, t + rt + "MinusOwn", p.get("ci95"), 2)
    if R.get("added_family") or R.get("added_task"):
        _set(m, "tregretAddedCases", str(len(R.get("added_family") or {}) + len(R.get("added_task") or {})))
    return _check(m)


# ---------------------------------------------------------------- protocol ceiling

AXES = (("arm", "Arm"), ("scoring", "Scoring"), ("scale", "Scale"), ("random", "Random"))


def ceiling_macros(R):
    """results/protocol_ceiling/protocol_ceiling.json: per task the headline curve (strict, points, arm switch) and
    each axis's agreement at K = 1 and its two limits."""
    m = {}
    for task, T in R["tasks"].items():
        t = "pceil" + TASK.get(task, tag_of(task))
        h = T["headline"]
        m[t + "D"] = str(T["D"])
        w = h["within_requested"]
        kmax = max(w, key=int)
        m[t + "WithinOne"] = num(w["1"], 3)
        m[t + "WithinMax"], m[t + "WithinMaxK"] = num(w[kmax], 3), str(int(kmax))
        for k, s in (("1", "One"), ("12", "Twelve"), ("24", "TwentyFour")):
            m[t + "Between" + s] = num(h["between_same_draws"].get(k), 3)
        m[t + "LimitPlug"] = num(h["plugin"], 3)
        _ci(m, t + "LimitPlug", h.get("plugin_bootstrap"), 3)
        m[t + "LimitPost"] = num(h["posterior"], 3)
        for axis, at in AXES:
            a = T["axes"].get(axis)
            if a is None:
                continue
            m[t + at + "One"] = num(a["same_draws"].get("1"), 3)
            m[t + at + "LimitPlug"] = num(a.get("limit_plugin"), 3)
            m[t + at + "LimitPost"] = num(a.get("limit_posterior"), 3)
    return _check(m)


# ---------------------------------------------------------------- published claims


def published_macros(R):
    """results/published_recompute/published_recompute.json: record counts, claims that change direction under
    another scale, and model pairs reversed against points."""
    m = {"pubreRecords": str(R["n_records"]), "pubreBothArms": str(len(R["claims"])),
         "pubreUntraced": str(len(R["untraced"])), "pubreNoNumbers": str(R["n_no_numbers"])}
    m["pubreChangeDirection"] = str(sum(any(c["changes_direction_from_points"].values()) for c in R["claims"].values()))
    for key, c in R["claims"].items():
        t = "pubre" + "".join(ch for ch in key.split("20")[0].capitalize() if ch.isalpha())
        for sc, s in (("logits", "Logit"), ("error_reduction", "ErrRed")):
            pf = c["pair_flips"][sc]
            m[t + "Flips" + s], m[t + "Pairs" + s] = str(pf["reversed"]), str(pf["pairs"])
            m[t + "Changes" + s] = "yes" if c["changes_direction_from_points"][sc] else "no"
    return _check(m)


# ================================================================ further macros
# New prefixes:
#     ae      size of the arm effect (results/arm_effect/)
#     mres    model and family bootstraps of the capability slopes (results/model_resampling/)
#     gdiff   the slope by GSM8K difficulty stratum (results/gsm8k_difficulty/)
#     rsw     readability cutoff sweep and same-generation ladders (results/readable_sweep/)
#     hcap    capability rules for a held-out model or family (results/heldout_capability/)
#     lowr    agreement at low reliability from Dataset 2 subsamples (results/lowr_map/)
#     hq      HotpotQA (results/hotpot/), read once its full run has written hotpot_report.json
#     aw      arm wording drawn (results/arm_wording/), read once arm_wording.json holds scored outputs
# Further names under existing prefixes (no existing name is reused; make_numbers.put_new refuses one):
#     dstrict (cross-fitted regret, plug-in intervals, best third and panel gain), pceil (the within-protocol curve
#     over all 18 protocols, its limit, pair counts), fewshot (per-model counts, compliance), gc (draw sets and
#     leave-one-out ranges), tv (ranges per task), and da, db, gt, bh, mt (validation-draw agreement, CbenVal).
# Every value is read from a results file or counted from its rows; nothing is re-estimated.

WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 10: "Ten", 12: "Twelve", 24: "TwentyFour"}
OUTCOME = {"CotMinusForced": "Forced", "CotMinusReq": "Req"}
PANEL = {"full": "Full", "readable": "Readable"}
SETTING_TAG = {"fewshot_gsm8k": "FsGsm", "fewshot_bbh": "FsBbh", "fewshot_math": "FsMath", "constrained_bbh": "GcBbh"}


def _added_config():
    """config/added_contrasts.yaml (contrasts, tasks, pools), or {} when it is absent."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "added_contrasts.yaml")
    if not os.path.exists(p):
        return {}
    import yaml
    return yaml.safe_load(open(p, encoding="utf-8")) or {}


def pool_tag(pool):
    """A pool's macro tag (config/added_contrasts.yaml, section pools): empty for pool base."""
    return str(((_added_config().get("pools") or {}).get(pool) or {}).get("tag") or "")


# the tags of the contrasts added later (HotpotQA) come from the configuration
for _k, _c in (_added_config().get("contrasts") or {}).items():
    SETTING_TAG.setdefault(_k, _c["tag"])


def _set(m, k, v):
    if k in m:
        raise ValueError(f"macro {k} set twice")
    m[k] = v


def _lohi(m, stem, ci, nd):
    """stem + Lo/Hi from a two-element interval; "--" for both when the interval is absent."""
    ok = isinstance(ci, (list, tuple)) and len(ci) == 2
    _set(m, stem + "Lo", num(ci[0], nd) if ok else "--")
    _set(m, stem + "Hi", num(ci[1], nd) if ok else "--")


def _drawset(m, stem, s, nd):
    """A draw set (from the delete-one-draw jackknife) as Lo/Hi when bounded; otherwise "--" and its kind in
    stem + "Kind" (an unbounded set is reported as it is)."""
    s = s or {}
    bounded = s.get("kind") == "bounded"
    _lohi(m, stem, [s.get("a"), s.get("b")] if bounded else None, nd)
    if not bounded:
        _set(m, stem + "Kind", str(s.get("kind", "--")).replace("_", " "))


def _finite(xs):
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _minmax(m, stem, xs, nd):
    v = _finite(xs)
    _set(m, stem + "Min", num(min(v), nd) if v else "--")
    _set(m, stem + "Max", num(max(v), nd) if v else "--")


def _task(t):
    return TASK.get(t, tag_of(t))


# ---------------------------------------------------------------- 1. strict decision comparison, further columns


def strict_more_macros(R, evaluation="fresh"):
    """results/decision_strict/decision_strict.json: per task, design and baseline arm, the cross-fitted regret of each
    rule, the plug-in regret's 95% interval, the best third's gain and the panel mean gain (the plug-in regrets and
    the paired differences are in strict_macros)."""
    m = {}
    for task, T in R["tasks"].items():
        for design, res in T["results"].items():
            for arm, by in res.items():
                t = "dstrict" + _task(task) + design.capitalize() + TOKEN.get(arm, arm.capitalize())
                x = by[evaluation]
                _set(m, t + "BestThird", num(x.get("best_third_value"), 2))
                _set(m, t + "PanelGain", num(x.get("panel_mean_gain"), 2))
                for rule, rt in STRICT_RULES:
                    r = x["rules"].get(rule)
                    if r is None:
                        continue
                    _set(m, t + rt + "Cross", num(r.get("regret_crossfit"), 2))
                    _lohi(m, t + rt, (r.get("plugin_interval") or {}).get("ci95"), 2)
    return _check(m)


# ---------------------------------------------------------------- 2. protocol ceiling, further values


def ceiling_more_macros(R):
    """results/protocol_ceiling/protocol_ceiling.json: the within-protocol curve averaged over the 18 protocols
    (empirical at K = 1, 4 and 12; the Gaussian pair model at K = 24), its limit, which is 1 by construction (a
    protocol agrees with itself once the draws are averaged away; the summary's table prints it so), and the number of
    protocol pairs behind each axis."""
    m = {}
    for task, T in R["tasks"].items():
        t = "pceil" + _task(task)
        w = T["axes"].get("within")
        if w:
            _set(m, t + "WithinAllProtocols", str(w["protocols"]))
            for k in (1, 4, 12):
                _set(m, t + "WithinAll" + WORD[k], num(w["empirical"].get(str(k)), 3))
            _set(m, t + "WithinAllTwentyFourModel", num(w["model"].get("24"), 3))
        _set(m, t + "WithinLimit", "1")
        for axis, at in AXES:
            a = T["axes"].get(axis)
            if a is not None:
                _set(m, t + at + "Pairs", str(a["pairs"]))
    return _check(m)


# ---------------------------------------------------------------- 3. arm effect


def arm_effect_macros(R):
    """results/arm_effect/arm_effect.json: reversed pairs overall and among pairs whose two gaps both exceed 1, 2, 5
    and 10 pp, the same on each task's readable panel, the correlations of requested - forced with the share of long
    requested outputs, and requested - forced among the models that rarely write more than the forced cap."""
    m = {"aeForcedCap": str(R["forced_cap_tokens"]), "aeShortSharePct": num(R["short_share_pct"], 0)}
    for task, T in R["tasks"].items():
        t = "ae" + _task(task)
        _set(m, t + "M", str(T["M"]))
        for rev, s in ((T["reversals"], ""), (T.get("reversals_readable"), "Readable")):
            if not rev:
                continue
            if s:
                _set(m, t + "ReadableM", str(T["readable_M"]))
            _set(m, t + s + "Pairs", str(rev["pairs"]))
            _set(m, t + s + "Reversed", str(rev["reversed"]))
            _set(m, t + s + "ReversedShare", num(rev["reversed_share"], 3))
            _set(m, t + s + "ReversedPct", num(100 * rev["reversed_share"], 0))
            _set(m, t + s + "Ties", str(rev["ties"]))
            for b in rev["by_threshold"]:
                g = int(round(b["min_gap_pp"]))
                if g == 0 or g not in WORD:
                    continue
                _set(m, t + s + "Gap" + WORD[g] + "Pairs", str(b["pairs_both_above"]))
                _set(m, t + s + "Gap" + WORD[g] + "Reversed", str(b["reversed"]))
        _set(m, t + "Pearson", num(T["pearson_long_share_vs_req_minus_forced"], 2))
        _set(m, t + "Spearman", num(T["spearman_long_share_vs_req_minus_forced"], 2))
        _set(m, t + "MeanReqMinusForced", num(T["mean_req_minus_forced"], 1))
        _set(m, t + "ShortM", str(len(T["short_models"])))
        rg = T.get("short_models_req_minus_forced_range") or [None, None]
        _set(m, t + "ShortMin", num(rg[0], 1))
        _set(m, t + "ShortMax", num(rg[1], 1))
        _set(m, t + "ShortMean", num(T.get("short_models_req_minus_forced_mean"), 1))
        # the same quantities under token F1 (an added task with an F1 scorer,
        # HotpotQA): same pairs and tie rule, same short-answer stratum (chosen from lengths only)
        b = T.get("token_f1")
        if b:
            rev = b["reversals"]
            _set(m, t + "ReversedFOne", str(rev["reversed"]))
            _set(m, t + "ReversedShareFOne", num(rev["reversed_share"], 3))
            _set(m, t + "ReversedPctFOne", num(100 * rev["reversed_share"], 0))
            _set(m, t + "TiesFOne", str(rev["ties"]))
            _set(m, t + "MeanReqMinusForcedFOne", num(b["mean_req_minus_forced"], 1))
            _set(m, t + "ShortMFOne", str(b["short_models_n"]))
            rg = b.get("short_models_req_minus_forced_range") or [None, None]
            _set(m, t + "ShortMinFOne", num(rg[0], 1))
            _set(m, t + "ShortMaxFOne", num(rg[1], 1))
            _set(m, t + "ShortMeanFOne", num(b.get("short_models_req_minus_forced_mean"), 1))
            _set(m, t + "ShortBelowZeroFOne", str(b["short_models_req_minus_forced_below_zero"]))
    return _check(m)


# ---------------------------------------------------------------- 4. model resampling


def _slope_block(m, t, b, nd=3):
    """One slope with its draw set, model and family bootstraps (95% interval, share of resamples below zero, undefined
    replicates) and leave-one-model-out and leave-one-family-out ranges. A family bootstrap over one family repeats
    the slope; it is reported as it is, with t + "Families" = 1 beside it."""
    _set(m, t + "M", str(b["M"]))
    _set(m, t, num(b.get("slope"), nd))
    _drawset(m, t + "Draw", b.get("draw_set"), nd)
    for k, s in (("model_bootstrap", "Model"), ("family_bootstrap", "Family")):
        bb = b.get(k) or {}
        _lohi(m, t + s, bb.get("ci95"), nd)
        _set(m, t + s + "ShareBelow", num(bb.get("share_below_zero"), 3))
        _set(m, t + s + "Undefined", str(bb.get("n_undefined", "--")))
    lm = b.get("leave_one_model_out") or {}
    _set(m, t + "LomoMin", num(lm.get("min"), nd))
    _set(m, t + "LomoMax", num(lm.get("max"), nd))
    _minmax(m, t + "Lofo", (b.get("leave_one_family_out") or {}).values(), nd)
    if "families" in b:
        _set(m, t + "Families", str(len(b["families"])))


def resampling_macros(R):
    """results/model_resampling/model_resampling.json: each slope (task, outcome, panel), the change from
    GSM8K to MATH-500 on the 23 models, and the trend of the MATH-500 slope over levels 1-5."""
    m = {}
    for task, T in R["tasks"].items():
        for key, b in T["blocks"].items():
            out, panel = key.split("/")
            _slope_block(m, "mres" + _task(task) + OUTCOME.get(out, tag_of(out)) + PANEL.get(panel, tag_of(panel)), b)
    c = R.get("item_g_change")
    if c:
        _set(m, "mresChange", num(c["estimate"], 3))
        _set(m, "mresChangeGsm", num(c["gsm8k"], 3))
        _set(m, "mresChangeMath", num(c["math"], 3))
        for k, s in (("model_bootstrap", "Model"), ("family_bootstrap", "Family")):
            _lohi(m, "mresChange" + s, (c.get(k) or {}).get("ci95"), 3)
            _set(m, "mresChange" + s + "ShareBelow", num((c.get(k) or {}).get("share_below_zero"), 3))
    lt = R.get("math_level_trend")
    if lt:
        _set(m, "mresLevelTrend", num(lt["estimate"], 3))
        for k, s in (("model_bootstrap", "Model"), ("family_bootstrap", "Family")):
            _lohi(m, "mresLevelTrend" + s, (lt.get(k) or {}).get("ci95"), 3)
            _set(m, "mresLevelTrend" + s + "ShareBelow", num((lt.get(k) or {}).get("share_below_zero"), 3))
        for i, v in enumerate(lt.get("level_slopes", []), start=1):
            _set(m, "mresLevel" + WORD[i], num(v, 3))
        # per-level model bootstrap (absent in files written before it was added: no macros then)
        for i, b in enumerate(lt.get("level_model_bootstrap") or [], start=1):
            _lohi(m, "mresLevel" + WORD[i] + "Model", b.get("ci95"), 3)
            _set(m, "mresLevel" + WORD[i] + "ModelShareBelow", num(b.get("share_below_zero"), 3))
    return _check(m)


# ---------------------------------------------------------------- 5. GSM8K difficulty strata

STRATUM = {"<=2": "UpToTwo", "3": "Three", "4": "Four", "5": "Five", ">=6": "SixUp",
           "Q1": "QOne", "Q2": "QTwo", "Q3": "QThree", "Q4": "QFour", "Q5": "QFive"}


def difficulty_macros(R):
    """results/gsm8k_difficulty/gsm8k_difficulty.json: per scheme (calculator steps; cross-fitted quintiles), panel and
    stratum the three arms' accuracy, the slope of chain of thought - forced on forced accuracy with its draw set and
    model bootstrap, the ceiling slope and the slope against the requested arm; the trend over strata with its
    jackknife (draw), model and family intervals; and the ranges of the requested-arm slope and of chain-of-thought
    accuracy over strata."""
    m = {}
    for scheme, s in (("steps", "Steps"), ("crossfit", "Cross")):
        S = R.get(scheme)
        if not S:
            continue
        for st, n in (S.get("n_items") or {}).items():
            _set(m, "gdiff" + s + STRATUM[st] + "Items", str(n))
        for panel, P in S["panels"].items():
            t = "gdiff" + s + PANEL[panel]
            _set(m, t + "M", str(P["M"]))
            for x in P["strata"]:
                u = t + STRATUM[x["stratum"]]
                _set(m, u + "Req", num(x["requested"], 1))
                _set(m, u + "Forced", num(x["forced"], 1))
                _set(m, u + "Cot", num(x["cot"], 1))
                _set(m, u + "Slope", num(x["slope_cot_minus_forced"], 3))
                _drawset(m, u + "SlopeDraw", x.get("set"), 3)
                mb = x.get("model_bootstrap") or {}
                _lohi(m, u + "SlopeModel", mb.get("ci95"), 3)
                _set(m, u + "SlopeModelShareBelow", num(mb.get("share_below_zero"), 3))
                _set(m, u + "Ceiling", num(x["ceiling_slope"], 3))
                _set(m, u + "SlopeReq", num(x["slope_cot_minus_requested"], 3))
                _drawset(m, u + "SlopeReqDraw", x.get("set_requested"), 3)
            _minmax(m, t + "SlopeReq", [x["slope_cot_minus_requested"] for x in P["strata"]], 3)
            _minmax(m, t + "Cot", [x["cot"] for x in P["strata"]], 1)
            tr = P["trend"]
            _set(m, t + "Trend", num(tr["estimate"], 3))
            _lohi(m, t + "TrendDraw", tr.get("ci95"), 3)
            for k, ks in (("model_bootstrap", "Model"), ("family_bootstrap", "Family")):
                _lohi(m, t + "Trend" + ks, (tr.get(k) or {}).get("ci95"), 3)
                _set(m, t + "Trend" + ks + "ShareBelow", num((tr.get(k) or {}).get("share_below_zero"), 3))
    return _check(m)


# ---------------------------------------------------------------- 6. five-shot, per-model counts and compliance

MODEL_TAG = {"qwen3-8b": "QwenThreeEight", "mistral-small-3.1-24b-instruct-2503": "MistralSmall",
             "mistral-7b-instruct-v0.1": "MistralSevenOne"}


def fewshot_more_macros(results, compliance=None):
    """results: the fewshot.json files; per task the number of models, the number whose five-shot gain is negative,
    and the smallest and largest gain (pp); on BIG-Bench Hard the two arms' accuracy for the models named in
    the text. compliance: rows of tables_fewshot_bbh/compliance_by_model_arm.csv (arm
    direct_answer is zero-shot forced, zero_shot_cot is five-shot) for the same models. The placeholder shares quoted
    in the text were counted from output shards and are in no results file, so no macro carries them."""
    m = {}
    for R in results:
        if R.get("dry_run"):
            continue
        for task, T in R["tasks"].items():
            t = "fewshot" + _task(task)
            g = [r["gain"] for r in T["per_model"]]
            _set(m, t + "Models", str(len(g)))
            _set(m, t + "Losers", str(sum(x < 0 for x in g)))
            _minmax(m, t + "Gain", g, 1)
            for r in T["per_model"]:
                k = MODEL_TAG.get(r["model_key"])
                if k and task == "bbh":
                    _set(m, t + k + "AccZero", num(r["zero_shot_forced"], 1))
                    _set(m, t + k + "AccFive", num(r["five_shot_forced"], 1))
    for r in compliance or []:
        k = MODEL_TAG.get(r["model_key"])
        if k:
            arm = {"direct_answer": "Zero", "zero_shot_cot": "Five"}[r["arm"]]
            _set(m, "fewshotBbh" + k + "Compliance" + arm, num(r["compliance_pct"], 1))
    return _check(m)


# ---------------------------------------------------------------- 7. constrained forced arm, further columns


def constrained_more_macros(R):
    """results/forced_constrained/forced_constrained.json: per outcome and panel the slope's draw set, the
    leave-one-model-out and leave-one-family-out ranges ("--" where no family can be left out) and the number of
    families (a family bootstrap over one family repeats the slope; reported as it is)."""
    m = {}
    if R.get("dry_run"):
        return m
    for outcome, s in (("cot_minus_requested", "Req"), ("cot_minus_forced", "Forced"),
                       ("cot_minus_constrained", "Constrained")):
        for name, b in R["slopes"][outcome].items():
            if "slope" not in b:
                continue
            stem = "gcSlope" + s + name.capitalize()
            _drawset(m, stem + "Draw", b.get("draw_set"), 2)
            lm = b.get("leave_one_model_out") or {}
            _set(m, stem + "LomoMin", num(lm.get("min"), 2))
            _set(m, stem + "LomoMax", num(lm.get("max"), 2))
            _minmax(m, stem + "Lofo", (b.get("leave_one_family_out") or {}).values(), 2)
            _set(m, stem + "Families", str(len(b.get("families") or {})))
    return _check(m)


# ---------------------------------------------------------------- 8. vendor sampling, ranges per task


def vendor_more_macros(R):
    """results/thinking_vendor/thinking_vendor.json: the cut share and mean output length under each decoding, as
    ranges over the checkpoints, per task (and the lengths over both tasks; the pooled cut ranges are in
    vendor_macros)."""
    m = {}
    rng = {}
    for cell, c in R["cells"].items():
        task = cell.split("/")[0]
        for dec, s in (("vendor", "Vendor"), ("greedy", "Greedy")):
            rng.setdefault("tv" + _task(task) + "Cut" + s, []).append(c[dec]["cut"]["pct"])
            rng.setdefault("tv" + _task(task) + "Tokens" + s, []).append(c[dec]["mean_output_tokens"])
            rng.setdefault("tvTokens" + s, []).append(c[dec]["mean_output_tokens"])
    for k, v in rng.items():
        if "Tokens" in k:
            f = _finite(v)
            _set(m, k + "Min", f"{min(f):,.0f}".replace(",", "{,}") if f else "--")
            _set(m, k + "Max", f"{max(f):,.0f}".replace(",", "{,}") if f else "--")
        else:
            _minmax(m, k, v, 1)
    return _check(m)


# ---------------------------------------------------------------- 9. readability sweep and ladders

CUTOFF = {80.0: "Eighty", 85.0: "EightyFive", 90.0: "Ninety", 92.5: "NinetyTwoHalf", 95.0: "NinetyFive",
          97.5: "NinetySevenHalf", 99.0: "NinetyNine", 99.5: "NinetyNineHalf"}
VARIANT = {"primary": "Primary", "dense": "Dense", "dense_readable": "DenseReadable", "qwen3_mixtures": "QwenMix"}
BASIS = {"gsm8k": "GsmBasis", "own": "OwnBasis"}


def sweep_macros(R):
    """results/readable_sweep/readable_sweep.json: the slope by readability cutoff (task, readability basis,
    cutoff: M, slope and draw set; "--" where fewer than 3 models pass), and the ladder counts per task, outcome and
    variant, counted from the ladders table as the summary counts them: ladders, rises (Kojima slope's interval above
    zero), falls (below zero) and ladders whose smallest model loses (Wei)."""
    m = {}
    for c in R["cutoffs"]:
        t = "rsw" + _task(c["task"]) + BASIS.get(c["readability_basis"], tag_of(c["readability_basis"])) + \
            CUTOFF[float(c["cutoff_pct"])]
        _set(m, t + "M", str(c["M"]))
        _set(m, t + "Slope", num(c.get("slope"), 3))
        ok = c.get("set_kind") == "bounded"
        _lohi(m, t + "Slope", [c.get("lo"), c.get("hi")] if ok else None, 3)
    groups = {}
    for x in R["ladders"]:
        groups.setdefault((x["task"], x["variant"], x["outcome"]), []).append(x)
    for (task, var, out), rows in groups.items():
        t = "rsw" + _task(task) + OUTCOME.get(out, tag_of(out)) + VARIANT.get(var, tag_of(var))
        _set(m, t + "Ladders", str(len(rows)))
        _set(m, t + "Rise", str(sum(r["kojima_sign"] == "above zero" for r in rows)))
        _set(m, t + "Fall", str(sum(r["kojima_sign"] == "below zero" for r in rows)))
        _set(m, t + "SmallLoses", str(sum(bool(r["wei_smallest_loses"]) for r in rows)))
    return _check(m)


# ---------------------------------------------------------------- 10. held-out capability and low-R map

HCAP_RULES = (("constant", "Const"), ("baseline_linear", "BaseLin"), ("baseline_quadratic_index", "BaseQuad"),
              ("index_linear", "Index"))


def heldout_macros(R):
    """results/heldout_capability/heldout_capability.json: per task, outcome, hold-out and rule the risk (pp^2), the
    held-out R^2, risk(constant) - risk(rule) with its bootstrap interval; the own-gain risk; and the ranges quoted in
    the text (R^2 over the three rules and both hold-outs per task and
    outcome; the index rule's R^2 per task; the own-gain risk over every task and outcome)."""
    m = {}
    own, idx = [], {}
    for task, T in R["tasks"].items():
        tk = _task(task)
        for out, O in T.items():
            t = "hcap" + tk + OUTCOME.get(out, tag_of(out))
            _set(m, t + "OwnRisk", num(O["own_gain_risk"], 2))
            own.append(O["own_gain_risk"])
            r2 = []
            for hold in ("model", "family"):
                H = O.get(hold)
                if not H:
                    continue
                for rule, rt in HCAP_RULES:
                    x = H.get(rule)
                    if x is None:
                        continue
                    u = t + hold.capitalize() + rt
                    _set(m, u + "Risk", num(x["risk"], 1))
                    if rule == "constant":
                        continue
                    _set(m, u + "RTwo", num(x["heldout_r2"], 2))
                    _set(m, u + "Diff", num(x["risk_constant_minus_rule"], 1))
                    _lohi(m, u + "Diff", (x.get("bootstrap_constant_minus_rule") or {}).get("ci95"), 1)
                    r2.append(x["heldout_r2"])
                    if rule == "index_linear":
                        idx.setdefault(tk, []).append(x["heldout_r2"])
            _minmax(m, t + "RTwo", r2, 2)
    for tk, v in idx.items():
        _minmax(m, "hcap" + tk + "IndexRTwo", v, 2)
    _minmax(m, "hcapOwnRisk", own, 2)
    return _check(m)


BAND = {"R < 0": "BandNeg", "0 <= R < 0.2": "BandZero", "0.2 <= R < 0.4": "BandTwo", "0.4 <= R < 0.6": "BandFour",
        "0.6 <= R < 0.8": "BandSix", "R >= 0.8": "BandEight"}


def lowr_macros(R):
    """results/lowr_map/lowr_map.json: replicate counts (and those with R outside [0, 1], reported as they are), the
    replicates below R = 0.6 (summed from the pooled bands), and per setting and band the replicates, median R, the
    departure of C from the Gaussian reference at one draw and K = 4, the same for the Gaussian null with the same
    design, and data minus null (the summary's last column at K = 4; at one draw the same difference of the two
    reported departures)."""
    m = {"lowrReplicates": str(R["n_replicates"]), "lowrRoutside": str(R["R_outside_unit_interval"])}
    if "n_replicates_added" in R:
        _set(m, "lowrReplicatesAdded", str(R["n_replicates_added"]))
        _set(m, "lowrRoutsideAdded", str(R["R_outside_unit_interval_added"]))
    for pool in R.get("added_pools", [])[1:]:          # a later pool (HotpotQA): its own counts, by its tag
        if f"n_replicates_added_{pool}" in R:
            _set(m, "lowrReplicatesAdded" + pool_tag(pool), str(R[f"n_replicates_added_{pool}"]))
            _set(m, "lowrRoutsideAdded" + pool_tag(pool), str(R[f"R_outside_unit_interval_added_{pool}"]))
    null = {(b["setting"], b["band"]): b for b in R["null_bands"]}
    low = [b["n"] for b in R["bands"] if b["setting"] == "all" and b["band"] in ("R < 0", "0 <= R < 0.2",
                                                                                 "0.2 <= R < 0.4", "0.4 <= R < 0.6")]
    _set(m, "lowrBelowSix", str(sum(low)))
    _set(m, "lowrBelowSixPct", num(100 * sum(low) / R["n_replicates"], 1))
    for b in R["bands"]:
        s = "All" if b["setting"] == "all" else SETTING_TAG.get(b["setting"], tag_of(b["setting"]))
        t = "lowr" + s + BAND[b["band"]]
        _set(m, t + "N", str(b["n"]))
        _set(m, t + "MedianR", num(b["median_R"], 2))
        _set(m, t + "DepOne", num(b["departure_one_draw"], 3))
        _set(m, t + "DepFour", num(b["departure_K4"], 3))
        nb = null.get((b["setting"], b["band"]))
        if nb:
            _set(m, t + "NullN", str(nb["n"]))
            _set(m, t + "NullDepOne", num(nb["departure_one_draw"], 3))
            _set(m, t + "NullDepFour", num(nb["departure_K4"], 3))
            for k, s2 in (("departure_one_draw", "One"), ("departure_K4", "Four")):
                d = _finite([b[k], nb[k]])
                _set(m, t + "DataMinusNull" + s2, num(d[0] - d[1], 3) if len(d) == 2 else "--")
    return _check(m)


# ---------------------------------------------------------------- 12. validation-draw agreement

VALIDATION_PREFIX = {"dove_mmlu_block": "da", "gsm8k19": "db", "gsm8k30": "gt", "bbh:pooled": "bh", "math": "mt"}


def validation_agreement_macros(R):
    """results/reliability_map/reliability_map.json: C of benefit over pairs of single validation draws, for the
    settings whose Steps 1-3 macros quote C on the discovery draws (prefixes da, db, gt, bh, mt; name CbenVal)."""
    m = {}
    for s in R["settings"]:
        p = VALIDATION_PREFIX.get(s["setting"])
        if p:
            _set(m, p + "CbenVal", num(s["validation"]["one_draw"]["benefit"]["C"], 3))
    return _check(m)


# ---------------------------------------------------------------- 14. cost of the wrong protocol


def decision_cross_macros(R):
    """results/decision/decision.json, the "Across protocols" part: for each task the extra regret of one draw's own
    gain chosen under the requested arm and valued under the forced arm (choice <T>_req, evaluation <T>_fd, rule
    own_1, independent evaluation configurations), with its 95% t interval over those configurations: decX<T>OwnOne,
    ...Lo, ...Hi (pp per chosen model). A dry run gives none."""
    m = {}
    if R.get("dry_run"):
        return m
    for c in R.get("cross_protocol", []):
        task, _, arm = c["choice"].partition("_")
        if arm != "req" or c["evaluation_setting"] != task + "_fd":
            continue
        x = c["evaluation"]["independent"]["rules"]["own_1"]["extra_regret"]
        stem = "decX" + _task(task) + "OwnOne"
        _set(m, stem, num(x["estimate"], 2))
        _lohi(m, stem, x.get("ci95"), 2)
    return _check(m)


# ---------------------------------------------------------------- 13. HotpotQA and arm wording, once scored

SPELL = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
         "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty", "TwentyOne",
         "TwentyTwo", "TwentyThree", "TwentyFour")
HQ_CONTRAST = {"cot_req": "Req", "cot_forced": "Fd", "fewshot": "Fs"}
# The Steps 1-3 files HotpotQA's full run writes (scripts/run_hotpot.py), for make_numbers.d2_macros, with prefixes
# that keep the three contrasts apart.
HQ_STEPS = (("hotpot.json", "hqReq"), ("hotpot_forced.json", "hqFd"), ("hotpot_fewshot.json", "hqFs"))
TARGET = {0.8: "Eighty", 0.9: "Ninety"}


def _yn(v):
    return "--" if v is None else ("yes" if v else "no")


def _spell(i):
    return SPELL[i] if 0 <= i < len(SPELL) else tag_of(str(i))


def _decision_block(m, t, ev):
    """The decision analysis's macros for one evaluation block (as decision_macros, under the stem t)."""
    _set(m, t + "Configs", str(ev.get("n_configurations", "--")))
    _set(m, t + "BestThird", num(ev.get("best_third_value"), 2))
    _set(m, t + "PanelGain", num(ev.get("panel_mean_gain"), 2))
    for rule, rt in RULES:
        x = (ev.get("rules") or {}).get(rule)
        if x is None:
            continue
        _set(m, t + rt + "Plug", num(x["regret_plugin"], 2))
        _set(m, t + rt + "Cross", num(x["regret_crossfit"], 2))
    p = next((x for x in ev.get("paired", []) if x["a"] == "capability_1" and x["b"] == "own_1"), None)
    if p:
        _set(m, t + "CapMinusOwn", num(p["estimate"], 2))
        _lohi(m, t + "CapMinusOwn", p.get("ci95"), 2)


def hotpot_report(root):
    """results/hotpot/hotpot_report.json, which scripts/run_hotpot.py writes only at the end of its full run (every
    draw scored); None before that."""
    p = os.path.join(root, "results", "hotpot", "hotpot_report.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def hotpot_steps(root):
    """(path, prefix) of HotpotQA's Steps 1-3 files for make_numbers.d2_macros; empty until the full run has written
    hotpot_report.json, so a file left by an earlier stage is never read."""
    d = os.path.join(root, "results", "hotpot")
    if not os.path.exists(os.path.join(d, "hotpot_report.json")):
        return []
    return [(os.path.join(d, f), p) for f, p in HQ_STEPS if os.path.exists(os.path.join(d, f))]


def hotpot_macros(R):
    """results/hotpot/hotpot_report.json: per contrast the arms' means, the map point (R of benefit and accuracy on the
    discovery draws, C on the validation draws), the capability slope with its draw set, model and family bootstraps
    and leave-one-out ranges, V_delta on token F1, the decision analysis on the saved choices, and the four-draw
    statements against the validation draws. Steps 1-3 of each contrast come from make_numbers.d2_macros (HQ_STEPS)."""
    m = {"hqM": str(R["M"]), "hqItems": f"{R['n_items']:,}".replace(",", "{,}")}
    fc = R.get("fresh_configurations") or {}
    if fc:
        _set(m, "hqFreshDraws", str(fc.get("n_draws", "--")))
        _set(m, "hqFreshConfigs", str(fc.get("n_distinct", "--")))
    for c, T in R["contrasts"].items():
        t = "hq" + HQ_CONTRAST.get(c, tag_of(c))
        mean = T.get("means") or {}
        _set(m, t + "MeanB", num(mean.get("baseline"), 1))
        _set(m, t + "MeanP", num(mean.get("intervention"), 1))
        _set(m, t + "MeanGainFine", num(mean.get("gain"), 2))
        pt, ob = T["map"]["point"], T["map"]["validation_one_draw"]
        _set(m, t + "MapRben", num((pt.get("benefit") or {}).get("R"), 3))
        _set(m, t + "MapRacc", num((pt.get("accuracy") or {}).get("R"), 3))
        _set(m, t + "MapCval", num((ob.get("benefit") or {}).get("C"), 3))
        _set(m, t + "MapCvalAcc", num((ob.get("accuracy") or {}).get("C"), 3))
        if T.get("capability_slope"):
            _slope_block(m, t + "Slope", T["capability_slope"])
        vd = ((T.get("f1") or {}).get("step2") or {}).get("V_delta") or {}
        _set(m, t + "FOneVdelta", num(vd.get("estimate"), 2))
        _lohi(m, t + "FOneVdelta", vd.get("ci95"), 2)
        ev = ((T.get("decision") or {}).get("evaluation") or {}).get("independent")
        if ev:
            _decision_block(m, t + "Dec", ev)
    ck = (R.get("tranche") or {}).get("checks") or {}
    for c, x in ck.items():
        if c not in HQ_CONTRAST:
            continue
        t = "hq" + HQ_CONTRAST[c] + "Tr"
        _set(m, t + "PredC", num(x["a"]["predicted"], 3))
        _set(m, t + "ObsC", num(x["a"]["observed"], 3))
        _set(m, t + "CWithin", _yn(x["a"]["within_0.05"]))
        for target, k in (x.get("b") or {}).items():
            name = TARGET.get(round(float(target), 2), tag_of(str(target)))
            _set(m, t + "K" + name, str(k.get("K_needed", "--")) if k.get("K_needed") is not None else "--")
            _set(m, t + "CAtK" + name, num(k.get("observed_C_at_K"), 3))
            _set(m, t + "Reached" + name, _yn(k.get("reached")))
        d = x["d"]
        _set(m, t + "R", num(d["R_benefit"], 3))
        _set(m, t + "OwnBetterForecast", _yn(d["forecast_own_better"]))
        _set(m, t + "OwnBetterObserved", _yn(d["observed_own_better"]))
        _set(m, t + "OwnBetterHolds", _yn(d.get("holds")))
        _set(m, t + "RegretOwnOne", num((d.get("regret") or {}).get("own_1"), 2))
        _set(m, t + "RegretCapOne", num((d.get("regret") or {}).get("capability_1"), 2))
    sw = ck.get("c_arm_switch")
    if sw:
        o = sw["observed"]
        _set(m, "hqTrSwitchReversed", num((o.get("switch") or {}).get("opposite"), 3))
        _set(m, "hqTrNewDrawReversed", num((o.get("new_draw") or {}).get("opposite"), 3))
        _set(m, "hqTrSwitchMore", _yn(o.get("switch_reorders_more")))
        _set(m, "hqTrSwitchPredicted", _yn(sw.get("predicted")))
        _set(m, "hqTrSwitchHolds", _yn(sw.get("holds")))
    return _check(m)


AW_SCHEME = {"fixed": "Fixed", "trigger_drawn": "TriggerDrawn", "clause_drawn": "ClauseDrawn",
             "both_drawn": "BothDrawn"}
AW_REVERSALS = (("arm_switch", "ArmSwitch"), ("clause_switch_requested", "ClauseSwitchReq"),
                ("clause_switch_forced", "ClauseSwitchFd"), ("trigger_switch_requested", "TriggerSwitchReq"),
                ("configuration_switch_requested", "ConfigSwitchReq"), ("arm_and_clause_switch", "ArmClauseSwitch"))


def arm_wording_results(root):
    """results/arm_wording/arm_wording.json when it holds scored outputs; None while the file is absent or holds a dry
    run on synthetic scores (synthetic_seed set)."""
    p = os.path.join(root, "results", "arm_wording", "arm_wording.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf-8"))
    if R.get("kind") != "arm_wording_results" or R.get("synthetic_seed") is not None:
        return None
    return R


def arm_wording_macros(R):
    """results/arm_wording/arm_wording.json (strict scores): per task R of benefit under each scheme (wording fixed or
    drawn) for both baseline arms, with one-draw C, the Gaussian reference, prompts for C = 0.9, the item bootstrap
    interval and the leave-one-configuration-out range; pair reversals by change; requested - forced per clause; the
    protocol ceiling by axis; the decision cost by axis; the checks of the saved predictions; the repeat check; and
    the four verdicts fixed in the arm-wording design."""
    m = {}
    if R.get("synthetic_seed") is not None:
        return m
    _set(m, "awM", str(len(R["models"])))
    _set(m, "awThird", str(R["n_third"]))
    _set(m, "awTriggers", str(len(R["wordings"]["cot"])))
    _set(m, "awClauses", str(len(R["wordings"]["dir"])))
    for task, T in R["per_task"].items():
        tk = "aw" + _task(task)
        cf = T.get("configurations") or {}
        _set(m, tk + "ConfigsDisc", str(len(cf.get("discovery", []))))
        _set(m, tk + "ConfigsVal", str(len(cf.get("validation", []))))
        S = T["strict"]
        boot = S.get("R_item_bootstrap_ci95") or {}
        for base, B in S["R"].items():
            bt = TOKEN.get(base, tag_of(base))
            for sc, x in B.items():
                t = tk + bt + AW_SCHEME.get(sc, tag_of(sc))
                _set(m, t + "R", num(x.get("R"), 3))
                _set(m, t + "Cone", num(x.get("C_one_draw"), 3))
                _set(m, t + "Gauss", num(x.get("gaussian_C"), 3))
                k = x.get("prompts_needed_0.9")
                _set(m, t + "PromptsNinety", "--" if k is None else str(k))
                _lohi(m, t + "R", (boot.get(base) or {}).get(sc), 3)
                lo = ((S.get("R_leave_one_configuration_out") or {}).get(base) or {}).get(sc) or {}
                _set(m, t + "LocoMin", num(lo.get("min"), 3))
                _set(m, t + "LocoMax", num(lo.get("max"), 3))
            fa = (S.get("R_fixed_any_wording") or {}).get(base)
            if fa:
                for k2, s2 in (("mean", "Mean"), ("min", "Min"), ("max", "Max")):
                    _set(m, tk + bt + "FixedAny" + s2, num(fa[k2], 3))
        rv = S["reversals"]
        for k2, s2 in AW_REVERSALS:
            x = rv.get(k2)
            if x is None:
                continue
            _set(m, tk + s2 + "OneMinusC", num(x["one_minus_C"], 3))
            _set(m, tk + s2 + "Reversed", num(x["reversed"], 3))
            _set(m, tk + s2 + "Tied", num(x["tied"], 3))
        by = [x["one_minus_C"] for x in rv.get("arm_switch_by_clause", [])]
        for i, v in enumerate(by, start=1):
            _set(m, tk + "ArmSwitchClause" + _spell(i), num(v, 3))
        _minmax(m, tk + "ArmSwitchClause", by, 3)
        for i, g in enumerate(S.get("direct_gap", []), start=1):
            t = tk + "GapClause" + _spell(i)
            _set(m, t + "Mean", num(g.get("mean_gap_pp"), 2))
            _set(m, t + "Min", num(g.get("min_model_gap_pp"), 2))
            _set(m, t + "Max", num(g.get("max_model_gap_pp"), 2))
            _set(m, t + "Long", num(g.get("long_requested_share"), 3))
        for ax, c in (S.get("ceiling") or {}).items():
            ks = sorted(c["curve"], key=int)
            if ax == "within":
                t = tk + "CeilWithin"
                _set(m, t + "Protocols", str(c["protocols"]))
            else:
                t = tk + "Ceil" + tag_of(ax)
                _set(m, t + "Pairs", str(c["pairs"]))
                _set(m, t + "LimitPlug", num(c.get("limit_plugin"), 3))
                _set(m, t + "LimitPost", num(c.get("limit_posterior"), 3))
                _lohi(m, t + "LimitPlug", c.get("limit_ci95_mean_bounds"), 3)
            if ks:
                _set(m, t + "One", num(c["curve"][ks[0]], 3))
                _set(m, t + "Last", num(c["curve"][ks[-1]], 3))
                _set(m, t + "LastK", str(ks[-1]))
        for ax, g in ((S.get("decision") or {}).get("grouped") or {}).items():
            t = tk + "Dec" + tag_of(ax)
            _set(m, t + "Protocols", str(g["protocols"]))
            _set(m, t + "Extra", num(g.get("estimate"), 2))
            _lohi(m, t + "Extra", g.get("ci95"), 2)
        pc = (S.get("prediction_checks") or {}).get("C_both_drawn")
        if pc:
            _set(m, tk + "PredC", num(pc.get("predicted"), 3))
            _set(m, tk + "ObsC", num(pc.get("observed_validation"), 3))
            _set(m, tk + "PredCWithin", _yn(pc.get("within_tolerance")))
        rc = (R.get("repeat_check") or {}).get(task) or {}
        for arm, a in (rc.get("per_arm") or {}).items():
            t = tk + "Repeat" + tag_of(arm)
            _set(m, t + "Same", num(a.get("same_strict_correctness"), 3))
            _set(m, t + "AccDiff", num(a.get("accuracy_difference_pp_mean"), 2))
    v = R.get("verdicts") or {}
    if v:
        _set(m, "awOnePromptAbstractChanges", _yn(v["one_prompt"]["abstract_changes"]))
        _set(m, "awArmSwitchAbstractChanges", _yn(v["arm_switch"]["abstract_changes"]))
        _set(m, "awClauseMovesAsMuchAsArm", _yn(v["clause_moves_as_much_as_arm"]["abstract_must_say"]))
        _set(m, "awDecisionGainsWording", _yn(v["decision"]["sentence_gains_wording"]))
    return _check(m)


def human_audit_results(root):
    """results/human_audit/human_audit.json (scripts/run_human_audit.py); None while the file is absent."""
    p = os.path.join(root, "results", "human_audit", "human_audit.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _pct(x, nd=1):
    return "--" if x is None else num(100 * x, nd)


def human_audit_macros(R):
    """The two-rater human audit, prefix hra. Percentages to 1 dp; kappa to 2 dp,
    "--" when undefined (the JSON says why). Agreement with the scorers and the LLM rater on the GSM8K and BIG-Bench Hard
    answers, where strict and content scores both exist (hraContentN decided answers), unless the name says All
    (every answer), Disagree (strict and content scoring disagree) or Cheng (the exemplars collection). Shares of the audit sample, not
    population rates. Intervals: 95% Wilson score."""
    m = {}
    c, ra, ref = R["counts"], R["raters"], R["references"]   # changed verdicts are not reported
    _set(m, "hraAnswers", str(c["answers"]))
    _set(m, "hraN", str(c["decided"]))
    _set(m, "hraShared", str(ra["shared"]))
    _set(m, "hraKappa", num(ra["codes"]["kappa"], 2))
    _set(m, "hraAgreePct", _pct(ra["codes"]["share"]))
    _set(m, "hraRaterDisagree", str(c["rater_disagreements_excluded"]))
    _set(m, "hraUnclear", str(c["cant_tell_excluded"]))
    for stem, scope in (("", "strict_and_content"), ("Disagree", "strict_content_disagree")):
        o = ref.get(scope) or {}
        _set(m, "hra" + (stem or "Content") + "N", str(o.get("decided", 0)))
        for k, name in (("content", "Content"), ("strict", "Strict")):
            b = o.get(k) or {}
            _set(m, f"hra{stem}{name}AgreePct", _pct(b.get("share")))
            if k == "content":
                ci = b.get("ci95_wilson")
                _set(m, f"hra{stem}{name}AgreePctLo", _pct(ci[0]) if ci else "--")
                _set(m, f"hra{stem}{name}AgreePctHi", _pct(ci[1]) if ci else "--")
    al = ref.get("all") or {}
    _set(m, "hraStrictAgreeAllPct", _pct((al.get("strict") or {}).get("share")))
    ex = ((ref.get("task:exemplar") or {}).get("second_scorer")) or {}
    _set(m, "hraChengAgreePct", _pct(ex.get("share")))
    ci = ex.get("ci95_wilson")
    _set(m, "hraChengAgreePctLo", _pct(ci[0]) if ci else "--")
    _set(m, "hraChengAgreePctHi", _pct(ci[1]) if ci else "--")
    return _check(m)


# ---------------------------------------------------------------- headline pair, arm switch, validation maps, slopes
# Every value is read from a results file, counted from its rows, or (the validation-draw departures) a difference of two
# values stored in the same row. The one new random step is the content-scored arm limit's interval
# (scripts/run_content_arm_ci.py, results/protocol_ceiling/content_arm_limit.json).

CONTENT_ARM_PAIR = "content/requested/points | content/forced/points"


def content_arm_results(root):
    """results/protocol_ceiling/content_arm_limit.json, or None while it is absent."""
    p = os.path.join(root, "results", "protocol_ceiling", "content_arm_limit.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def ceiling_disjoint_macros(R, content=None):
    """The headline pair (strict, points; requested against forced) on disjoint draws, agreement at K = 1 and at
    the largest K that 2K <= D allows, per task: pceil<T>DisjOne, DisjMax, DisjMaxK. The arm switch with both arms
    content-scored, its plug-in limit (from `limits`) and, when `content` (content_arm_limit.json) has the task, the
    interval of that file: pceil<T>ArmContentLimitPlug, Lo, Hi ("--" for the interval otherwise). A HotpotQA task comes
    in under Hq."""
    m = {}
    for task, T in R["tasks"].items():
        t = "pceil" + _task(task)
        dj = T["headline"]["between_disjoint"]
        kmax = max(dj, key=int)
        _set(m, t + "DisjOne", num(dj.get("1"), 3))
        _set(m, t + "DisjMax", num(dj[kmax], 3))
        _set(m, t + "DisjMaxK", str(int(kmax)))
        lim = T["limits"].get(CONTENT_ARM_PAIR)
        if lim is not None:
            _set(m, t + "ArmContentLimitPlug", num(lim["plugin"], 3))
            c = ((content or {}).get("tasks") or {}).get(task)
            _lohi(m, t + "ArmContentLimitPlug", c.get("ci95") if c else None, 3)
    return _check(m)


MAP_PANEL_TAG = {"gsm8k19": "Gsm", "gsm8k30": "GsmThirty", "bbh:pooled": "Bbh", "math": "Math"}
MAP_THREE_PANELS = ("gsm8k30", "bbh:pooled", "math")


def _is_hotpot_setting(s):
    return str(s.get("setting", "")).startswith("hotpot") or str(s.get("group", "")).startswith("hotpot")


def _rms(xs):
    v = _finite(xs)
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else None


def map_validation_macros(R):
    """From the settings of results/reliability_map/reliability_map.json (101 today; a HotpotQA setting, when the
    map holds one, is kept out of the counts below and read under Hq):
    - mapDepartValidationRms: root mean square over the settings of the observed one-draw validation agreement of benefit
      minus the Gaussian reference at the setting's own map R (observed minus reference; the file's `prediction_errors`
      for the 'map' row hold the same numbers with the opposite sign), with N in mapDepartValidationN. The same over the
      three full panels (GSM8K-29, BIG-Bench Hard-28 pooled, MATH-500: ...RmsPanels), the 71 DOVE benchmarks
      (...RmsDove) and the 25 BIG-Bench Hard tasks (...RmsBbhTasks), each with its N; and the signed departure of each
      panel, mapDepartValidation<Gsm|GsmThirty|Bbh|Math|Hq>.
    - mapValAboveNinety: settings whose one-draw validation agreement exceeds 0.90, out of mapValSettings.
    - mapDoveValMedian: median one-draw validation agreement over the DOVE benchmarks (group dove_tasks)."""
    m = {}
    S = [s for s in R["settings"] if not _is_hotpot_setting(s)]

    def dep(s):
        return s["validation"]["one_draw"]["benefit"]["C"] - s["point"]["benefit"]["C_ref"]
    _set(m, "mapDepartValidationRms", num(_rms([dep(s) for s in S]), 3))
    _set(m, "mapDepartValidationN", str(len(S)))
    for stem, sel in (("mapDepartValidationRmsPanels", [s for s in S if s["setting"] in MAP_THREE_PANELS]),
                      ("mapDepartValidationRmsDove", [s for s in S if s["group"] == "dove_tasks"]),
                      ("mapDepartValidationRmsBbhTasks", [s for s in S if s["group"] == "bbh_tasks"])):
        _set(m, stem, num(_rms([dep(s) for s in sel]), 3))
        _set(m, stem + "N", str(len(sel)))
    for s in R["settings"]:
        tag = "Hq" if _is_hotpot_setting(s) else MAP_PANEL_TAG.get(s["setting"])
        if tag:
            _set(m, "mapDepartValidation" + tag, num(dep(s), 3))
    # HotpotQA's contrasts in a later added pool (added_settings_<pool>): each by its tag
    # (mapDepartValidationHqReq, HqFd, HqFs), and mapDepartValidationHq for chain of thought against requested direct,
    # the contrast of the three panels above
    for pool in R.get("added_pools", [])[1:]:
        for s in R.get(f"added_settings_{pool}") or []:
            _set(m, "mapDepartValidation" + s["tag"], num(dep(s), 3))
            if s.get("setting") == "hotpot:cot_req" and "mapDepartValidationHq" not in m:
                _set(m, "mapDepartValidationHq", num(dep(s), 3))
    c = [s["validation"]["one_draw"]["benefit"]["C"] for s in S]
    _set(m, "mapValSettings", str(len(S)))
    _set(m, "mapValAboveNinety", str(sum(x > 0.90 for x in c)))
    dove = sorted(s["validation"]["one_draw"]["benefit"]["C"] for s in S if s["group"] == "dove_tasks")
    if dove:
        mid = len(dove) // 2
        _set(m, "mapDoveValMedian", num(dove[mid] if len(dove) % 2 else (dove[mid - 1] + dove[mid]) / 2, 3))
        _set(m, "mapDoveValN", str(len(dove)))
    return _check(m)


def _excludes_zero(ci):
    """True when a 95% interval has two finite bounds on the same side of zero; False for an absent or unbounded one
    (counted as not excluding zero)."""
    v = _finite(ci) if isinstance(ci, (list, tuple)) else []
    return len(v) == 2 and (v[0] > 0 or v[1] < 0)


def resampling_count_macros(R):
    """Counted from the blocks of results/model_resampling/model_resampling.json (task x outcome x panel):
    mresSlopes slopes, of which mresExcludeZero have a model-bootstrap interval that excludes zero; the same without the
    BIG-Bench Hard readable panels (four models, one family): mresSlopesNoBbhReadable, mresExcludeZeroNoBbhReadable.
    The same for the family bootstrap over the blocks with at least two families (mresFamilySlopes,
    mresFamilyExcludeZero; a one-family interval repeats the slope and is not counted). These counts cover GSM8K,
    BIG-Bench Hard and MATH-500; a HotpotQA task is counted apart (mresHqSlopes, mresHqExcludeZero)."""
    m = {}

    def rows(tasks, skip_bbh_readable=False):
        return [b for t, T in tasks.items() for k, b in T["blocks"].items()
                if not (skip_bbh_readable and t == "bbh" and k.endswith("/readable"))]

    def model_count(bs):
        return sum(_excludes_zero((b.get("model_bootstrap") or {}).get("ci95")) for b in bs)
    three = {k: v for k, v in R["tasks"].items() if k in ("gsm8k", "bbh", "math")}
    a, b_ = rows(three), rows(three, True)
    _set(m, "mresSlopes", str(len(a)))
    _set(m, "mresExcludeZero", str(model_count(a)))
    _set(m, "mresSlopesNoBbhReadable", str(len(b_)))
    _set(m, "mresExcludeZeroNoBbhReadable", str(model_count(b_)))
    fam = [b for b in a if len(b.get("families") or {}) >= 2]
    _set(m, "mresFamilySlopes", str(len(fam)))
    _set(m, "mresFamilyExcludeZero",
         str(sum(_excludes_zero((b.get("family_bootstrap") or {}).get("ci95")) for b in fam)))
    if "hotpot" in R["tasks"]:
        h = rows({"hotpot": R["tasks"]["hotpot"]})
        _set(m, "mresHqSlopes", str(len(h)))
        _set(m, "mresHqExcludeZero", str(model_count(h)))
    return _check(m)


def lowr_settings_macros(R):
    """From results/lowr_map/lowr_map.json: the number of settings of the main analysis (lowrSettings), of the
    contrasts kept apart (lowrAddedSettings), and the replicates below R = 0.6 that each main setting contributes
    (lowrBelowSix<Exemplar|Gsm|Bbh|Math>, counted from the bands as lowrBelowSix is)."""
    m = {"lowrSettings": str(len(R["settings"])), "lowrAddedSettings": str(len(R.get("added_settings") or {}))}
    for pool in R.get("added_pools", [])[1:]:          # a later pool: lowrAddedSettings<tag>
        m["lowrAddedSettings" + pool_tag(pool)] = str(len(R.get(f"added_settings_{pool}") or {}))
    low = ("R < 0", "0 <= R < 0.2", "0.2 <= R < 0.4", "0.4 <= R < 0.6")
    for s in R["settings"]:
        tag = SETTING_TAG.get(s, tag_of(s))
        _set(m, "lowrBelowSix" + tag, str(sum(b["n"] for b in R["bands"] if b["setting"] == s and b["band"] in low)))
    return _check(m)


def pending(root):
    """What is still waiting: {name: status} for HotpotQA and the arm wording; and the
    two-rater human audit."""
    hq = "scored, macros written" if hotpot_report(root) is not None else \
        "not scored yet (results/hotpot/hotpot_report.json absent); skipped"
    p = os.path.join(root, "results", "arm_wording", "arm_wording.json")
    if arm_wording_results(root) is not None:
        aw = "scored, macros written"
    elif os.path.exists(p):
        aw = "results/arm_wording/arm_wording.json holds no scored outputs (dry run); skipped"
    else:
        aw = "not scored yet (results/arm_wording/arm_wording.json absent); skipped"
    ha = "rated, macros written" if human_audit_results(root) is not None else \
        "not rated yet (results/human_audit/human_audit.json absent); skipped"
    return {"HotpotQA": hq, "arm wording": aw, "human audit, two raters": ha}


# ---------------------------------------------------------------- 15. decision directions, baseline mixture and per-rater audit
# Both directions of the wrong-baseline decision cost (results/decision_directions/), the requested baseline as a mixture
# (results/baseline_mixture/, exploratory on three benchmarks), HotpotQA's both directions and mechanism forecast (from
# results/hotpot/hotpot_report.json once it exists), and each rater of the human audit alone. Strict scoring is primary;
# a Content suffix marks the same quantity on content scores.

DIRECTION = (("requested_to_forced", "ReqToFd"), ("forced_to_requested", "FdToReq"))


def _directions(m, stem, rules_blk, rule, lofo=None, nd=2):
    r = (rules_blk or {}).get(rule) or {}
    for k, tag in DIRECTION:
        x = r.get(k) or {}
        _set(m, stem + tag, num(x.get("estimate"), nd))
        _lohi(m, stem + tag, x.get("ci95"), nd)
        if lofo is not None:
            rg = (lofo.get("range") or {}).get(k)
            _set(m, stem + tag + "LofoMin", num(rg[0], nd) if rg else "--")
            _set(m, stem + tag + "LofoMax", num(rg[1], nd) if rg else "--")


def decision_directions_macros(R):
    """results/decision_directions/decision_directions.json, prefix ddir<T> (T: Gsm, Bbh, Math). One-draw pilots, own
    gain from one draw (the design of decX<T>OwnOne, whose requested-to-forced value ddir<T>ReqToFd repeats):
    ddir<T>ReqToFd, ddir<T>FdToReq (+Lo, Hi; 95% t interval over evaluation configurations), leaving one family out
    ...LofoMin, ...LofoMax, the random third's regret under each arm ddir<T>RandReq, RandFd. 100-item item-disjoint
    pilots (the design of dstrict): ddir<T>Pilot.... Content suffix: the same saved strict-scored choices valued on
    content-scored validation gains. Reversal share of the arm switch: ddir<T>Rev (count), RevPairs, RevShare,
    RevPct, model bootstrap RevBootLo/Hi, family bootstrap RevFamBootLo/Hi, leaving one family out RevLofoMin/Max,
    within one family RevWithinFam of RevWithinFamPairs; the same with a Content suffix after Rev. Across the three
    tasks: ddirOwnFdToReqMin/Max and ddirOwnReqToFdMin/Max (one-draw pilots), and the same with Pilot."""
    m = {}
    acc = {"one": {k: [] for k, _ in DIRECTION}, "pilot": {k: [] for k, _ in DIRECTION}}
    for task, T in R["tasks"].items():
        t = "ddir" + _task(task)
        o = T["one_draw"]
        _directions(m, t, o["rules"], "own_1", o.get("leave_one_family_out_own_1"))
        _set(m, t + "RandReq", num(o["random_regret"]["requested"], 2))
        _set(m, t + "RandFd", num(o["random_regret"]["forced"], 2))
        _set(m, t + "Configs", str(o["n_configurations"]))
        for k, _ in DIRECTION:
            acc["one"][k].append(o["rules"]["own_1"][k]["estimate"])
        if "content" in o:
            c = o["content"]
            r = c["rules"].get("own_1") or {}
            for k, tag in DIRECTION:
                x = r.get(k) or {}
                _set(m, t + tag + "Content", num(x.get("estimate"), 2))
                _lohi(m, t + tag + "Content", x.get("ci95"), 2)
                rg = ((c.get("leave_one_family_out_own_1") or {}).get("range") or {}).get(k)
                _set(m, t + tag + "ContentLofoMin", num(rg[0], 2) if rg else "--")
                _set(m, t + tag + "ContentLofoMax", num(rg[1], 2) if rg else "--")
        dj = ((T.get("strict") or {}).get("designs") or {}).get("disjoint") or {}
        if "strict" in dj:
            s = dj["strict"]
            _directions(m, t + "Pilot", s["rules"], "own_gain", s.get("leave_one_family_out_own_gain"))
            _set(m, t + "PilotConfigs", str(s["n_configurations"]))
            for k, _ in DIRECTION:
                acc["pilot"][k].append(s["rules"]["own_gain"][k]["estimate"])
        if "content" in dj:
            r = dj["content"]["rules"].get("own_gain") or {}
            for k, tag in DIRECTION:
                x = r.get(k) or {}
                _set(m, t + "Pilot" + tag + "Content", num(x.get("estimate"), 2))
                _lohi(m, t + "Pilot" + tag + "Content", x.get("ci95"), 2)
        for sc, suf in (("strict", ""), ("content", "Content")):
            v = (T.get("reversal") or {}).get(sc)
            if not v:
                continue
            s = t + "Rev" + suf
            _set(m, s, str(v["reversed"]))
            _set(m, s + "Pairs", str(v["pairs"]))
            _set(m, s + "Share", num(v["share"], 3))
            _set(m, s + "Pct", num(100 * v["share"], 0))
            _lohi(m, s + "Boot", v["model_bootstrap"].get("ci95"), 2)
            _lohi(m, s + "FamBoot", v["family_bootstrap"].get("ci95"), 2)
            rg = v.get("leave_one_family_out_range")
            _set(m, s + "LofoMin", num(rg[0], 2) if rg else "--")
            _set(m, s + "LofoMax", num(rg[1], 2) if rg else "--")
            _set(m, s + "WithinFam", str(v["within_family"]["reversed"]))
            _set(m, s + "WithinFamPairs", str(v["within_family"]["pairs"]))
    for d, stem in (("one", "ddirOwn"), ("pilot", "ddirPilot")):
        for k, tag in DIRECTION:
            v = acc[d][k]
            _set(m, stem + tag + "Min", num(min(v), 2) if v else "--")
            _set(m, stem + tag + "Max", num(max(v), 2) if v else "--")
    return _check(m)


ARM_TAG = {"requested": "Req", "forced": "Fd", "cot": "Cot"}


def baseline_mixture_macros(R):
    """results/baseline_mixture/baseline_mixture.json (exploratory, GSM8K, BIG-Bench Hard and MATH-500), prefix bmix<T>.
    Item level (distinct configurations of both splits): bmix<T>Cap (the forced cap, tokens), bmix<T>LongPct and
    ShortPct (shares of requested outputs longer than the cap and not, %), bmix<T><Short|Long><Req|Fd|Cot> accuracy
    (strict, 1 dp) with Lo, Hi (95% model bootstrap), the same with a Content suffix. Correction (1 - w) x gain over
    forced, discovery draws: bmix<T>Pairs, bmix<T>RevObs (reversed pairs, observed gain over requested against gain
    over forced) and RevObsPct, bmix<T>RevCorr and RevCorrPct (against the corrected gain), bmix<T>PearsonFd and
    PearsonCorr (2 dp), each with a Content suffix; bmix<T>WMeanPct (mean w over models, %), WMinPct, WMaxPct and WMedianPct (per-model w over
    the panel, discovery draws, %)."""
    m = {}
    for task, T in R["tasks"].items():
        t = "bmix" + _task(task)
        _set(m, t + "Cap", str(T["forced_cap_tokens"]))
        il = T["item_level"]["all_distinct"]
        _set(m, t + "LongPct", num(100 * il["long"]["share"], 0))
        _set(m, t + "ShortPct", num(100 * il["short"]["share"], 0))
        for g, gt in (("short", "Short"), ("long", "Long")):
            for sc, suf in (("strict", ""), ("content", "Content")):
                for arm, at in ARM_TAG.items():
                    x = il[g]["acc"][f"{arm}/{sc}"]
                    _set(m, t + gt + at + suf, num(x["estimate"], 1))
                    _lohi(m, t + gt + at + suf, x.get("ci95"), 1)
        wd = [100 * float(v) for v in T["w_discovery"]]
        _set(m, t + "WMeanPct", num(sum(wd) / len(wd), 0))
        # the range and median of per-model w over the panel (discovery draws)
        _set(m, t + "WMinPct", num(min(wd), 0))
        _set(m, t + "WMaxPct", num(max(wd), 0))
        _set(m, t + "WMedianPct", num(statistics.median(wd), 0))
        for sc, suf in (("strict", ""), ("content", "Content")):
            x = T["mechanism"][sc]
            _set(m, t + "Pairs" + suf, str(x["observed_vs_forced"]["pairs"]))
            _set(m, t + "RevObs" + suf, str(x["observed_vs_forced"]["reversed"]))
            _set(m, t + "RevObsPct" + suf, num(100 * x["observed_vs_forced"]["reversed_share"], 0))
            _set(m, t + "RevCorr" + suf, str(x["observed_vs_corrected"]["reversed"]))
            _set(m, t + "RevCorrPct" + suf, num(100 * x["observed_vs_corrected"]["reversed_share"], 0))
            _set(m, t + "PearsonFd" + suf, num(x["corr_forced"]["pearson"], 2))
            _set(m, t + "PearsonCorr" + suf, num(x["corr_corrected"]["pearson"], 2))
    return _check(m)


def hotpot_directions_macros(R):
    """results/hotpot/hotpot_report.json, keys decision_directions and mechanism_forecast (run_hotpot.py's full run):
    hqDdirReqToFd, hqDdirFdToReq (+Lo, Hi, LofoMin, LofoMax; one-draw pilots, own gain), hqDdirPilotReqToFd, ...FdToReq
    (100-item item-disjoint pilots); hqMechPredRev (predicted reversed pairs), hqMechObsRev (observed on validation),
    hqMechPairs, hqMechHits, hqMechMisses, hqMechFalseAlarms, hqMechPredShare, hqMechObsShare, and whether each saved
    statement held (hqMechShareHolds, PairsHolds, GainHolds: yes, no or --). Empty before the full run."""
    m = {}
    H = R.get("decision_directions")
    if H:
        o = H.get("one_draw") or {}
        if o.get("rules"):
            _directions(m, "hqDdir", o["rules"], "own_1", o.get("leave_one_family_out_own_1"))
        d = ((H.get("pilot100") or {}).get("designs") or {}).get("disjoint") or {}
        if d.get("rules"):
            _directions(m, "hqDdirPilot", d["rules"], "own_gain", d.get("leave_one_family_out_own_gain"))
    E = R.get("mechanism_forecast")
    if E:
        _set(m, "hqMechPairs", str(E["pairs"]))
        _set(m, "hqMechPredRev", str(E["predicted_reversed"]))
        _set(m, "hqMechObsRev", str(E["observed_reversed"]))
        _set(m, "hqMechHits", str(len(E["hits"])))
        _set(m, "hqMechMisses", str(len(E["misses"])))
        _set(m, "hqMechFalseAlarms", str(len(E["false_alarms"])))
        st = E["statements"]
        _set(m, "hqMechPredShare", num(st["share"]["predicted"], 3))
        _set(m, "hqMechObsShare", num(st["share"]["observed"], 3))
        for k, tag in (("share", "Share"), ("pairs", "Pairs"), ("gain", "Gain")):
            _set(m, "hqMech" + tag + "Holds", _yn(st[k].get("holds")))
    return _check(m)


def human_audit_rater_macros(R):
    """results/human_audit/human_audit.json, key per_rater: each rater alone, every answer the rater rated,
    no exclusion for disagreement. hraRater<A|B>Answers, CantTell; agreement on the GSM8K and BIG-Bench Hard answers
    with the content scorer hraRater<r>ContentAgree of ContentN (ContentAgreePct, Lo, Hi; Wilson) and the strict scorer
    (StrictAgreePct); on every answer with the strict scorer (StrictAgreeAllPct). Control answers: both scorers mark
    wrong, read as correct, hraRater<r>CtrlWrongRead of CtrlWrongN (GSM8K and BIG-Bench Hard) and ...All (every answer;
    the second scorer is Cheng et al.'s extractor on the exemplars collection); both mark right, read as not correct, CtrlRightRead of
    CtrlRightN, and ...All. Rating times and changed verdicts are not reported."""
    m = {}
    for r, b in (R.get("per_rater") or {}).items():
        t = "hraRater" + r
        _set(m, t + "Answers", str(b["answers"]))
        _set(m, t + "CantTell", str(b["cant_tell"]))
        sc = b["strict_and_content"]
        x = sc["content"]
        _set(m, t + "ContentAgree", str(x["agree"]))
        _set(m, t + "ContentN", str(x["n"]))
        _set(m, t + "ContentAgreePct", _pct(x["share"]))
        ci = x.get("ci95_wilson")
        _set(m, t + "ContentAgreePctLo", _pct(ci[0]) if ci else "--")
        _set(m, t + "ContentAgreePctHi", _pct(ci[1]) if ci else "--")
        _set(m, t + "StrictAgreePct", _pct(sc["strict"]["share"]))
        _set(m, t + "StrictAgreeAllPct", _pct(b["all"]["strict"]["share"]))
        for scope, suf in (("strict_and_content", ""), ("all", "All")):
            c = b["controls"][scope]
            _set(m, t + "CtrlWrongRead" + suf, str(c["both_scorers_wrong"]["read_correct"]))
            _set(m, t + "CtrlWrongN" + suf, str(c["both_scorers_wrong"]["n"]))
            _set(m, t + "CtrlRightRead" + suf, str(c["both_scorers_right"]["read_not_correct"]))
            _set(m, t + "CtrlRightN" + suf, str(c["both_scorers_right"]["n"]))
    return _check(m)


# ---------------------------------------------------------------- split pilot, baseline-shift decomposition and mechanism forecast
# splitp   the split pilot, exploratory on three benchmarks (results/split_pilot/)
# bmixDec  the exact decomposition of the baseline shift (results/baseline_mixture/, key decomposition; exploratory)
# hqSplit  HotpotQA's saved split-pilot choices on its validation draws (results/hotpot/hotpot_report.json)
# hqDec    HotpotQA's saved decomposition rule on its validation draws (same file)
# awMech   the arm-wording mechanism forecast on the arm-wording draws (results/arm_wording/mechanism_forecast_evaluation.json)

POLICY_TAG = {"requested_only": "ReqOnly", "forced_only": "FdOnly", "split": "Split"}
DEPLOY_TAG = {"requested": "Req", "forced": "Fd"}


def _split_block(m, t, c, nd=2):
    for p, pt in POLICY_TAG.items():
        for d, dt in DEPLOY_TAG.items():
            _set(m, t + pt + dt, num(c["regret"][p][d], nd))
        _set(m, t + pt + "Worst", num(c["regret"][p]["worst"], nd))
    _set(m, t + "BestSingle", {"requested_only": "requested", "forced_only": "forced"}.get(c["better_single"], "--"))
    _set(m, t + "BestSingleWorst", num(c["better_single_worst"], nd))
    _set(m, t + "Adv", num(c["advantage"], nd))
    _set(m, t + "Holds", _yn(c.get("split_better")))


def split_pilot_macros(R):
    """results/split_pilot/split_pilot.json (exploratory, GSM8K, MATH-500 and BIG-Bench Hard), prefix splitp<T>:
    <ReqOnly|FdOnly|Split><Req|Fd> regret under each deployment (pp per chosen model, 2 dp) and ...Worst; BestSingle
    (requested or forced), BestSingleWorst, Adv (better single's worst regret minus the split's), Holds (split lower);
    Configs; SeedsBetter of splitpSeeds, SeedAdvMin, SeedAdvMax; LofoBetter of LofoFamilies, LofoAdvMin, LofoAdvMax;
    splitpLofoAdvMin over every task; RefOk (yes when the cross-check table is reproduced to two decimals)."""
    m = {}
    allf, n_seeds = [], set()
    for task, T in R["tasks"].items():
        t = "splitp" + _task(task)
        _split_block(m, t, T["main"])
        _set(m, t + "Configs", str(T["n_configurations"]))
        s, lo = T["seeds_summary"], T["leave_one_family_out"]
        n_seeds.add(s["n"])
        _set(m, t + "SeedsBetter", str(s["split_better"]))
        _set(m, t + "SeedAdvMin", num(s["advantage_range"][0], 2))
        _set(m, t + "SeedAdvMax", num(s["advantage_range"][1], 2))
        _set(m, t + "LofoBetter", str(lo["split_better_count"]))
        _set(m, t + "LofoFamilies", str(lo["families"]))
        _set(m, t + "LofoAdvMin", num((lo["advantage_range"] or [None, None])[0], 2))
        _set(m, t + "LofoAdvMax", num((lo["advantage_range"] or [None, None])[1], 2))
        _set(m, t + "RefOk", _yn(T["ref_check"]["reproduced"]))
        allf += [v["advantage"] for v in lo["by_family"].values() if "advantage" in v]
    _set(m, "splitpSeeds", str(n_seeds.pop()) if len(n_seeds) == 1 else "--")
    _set(m, "splitpLofoAdvMin", num(min(allf), 2) if allf else "--")
    return _check(m)


def decomposition_macros(R):
    """results/baseline_mixture/baseline_mixture.json, key decomposition (exploratory), prefix bmixDec<T>, strict
    scores and a Content suffix: VarLongPct (long part's share of the between-model variance of the baseline shift, %,
    all distinct configurations), Recov, RecovLo, RecovHi (median and quartiles of within-model recovery on long rows,
    2 dp), RecovN, ShortGap (median short-row requested minus forced, pp, 1 dp), ClosedCorr (closed fraction against w,
    models with at least 10 pp of room, 2 dp), ClosedN, ShiftCorr (baseline shift against w, all models, 2 dp);
    discovery draws: RmsPooled, RmsShort (pp, 1 dp) of the two forms of the (1 - w) rule; choice under the requested arm
    (regret, pp, 1 dp) by forced gain, by (1 - w) x forced gain and by the short form, in sample (ChoiceFd, ChoiceCorr,
    ChoiceShort) and on validation means (ChoiceValFd, ChoiceValCorr, ChoiceValShort); ChoiceBoth of ChoiceN, the models
    chosen by both the (1 - w) x forced-gain form and the short form. Empty before the decomposition is added."""
    m = {}
    for task, T in (R.get("decomposition") or {}).items():
        t = "bmixDec" + _task(task)
        for sc, suf in (("strict", ""), ("content", "Content")):
            d = T["sets"]["all_distinct"][sc]
            v = d["variance_share_long"]
            _set(m, t + "VarLongPct" + suf, num(None if v is None else 100 * v, 0))
            r = d["recovery_long"] or {}
            _set(m, t + "Recov" + suf, num(r.get("median"), 2))
            _set(m, t + "RecovLo" + suf, num(r.get("q25"), 2))
            _set(m, t + "RecovHi" + suf, num(r.get("q75"), 2))
            _set(m, t + "RecovN" + suf, str(r.get("n", "--")))
            _set(m, t + "ShortGap" + suf, num((d["short_gap_pp"] or {}).get("median"), 1))
            cf = d["closed_fraction"]
            _set(m, t + "ClosedCorr" + suf, num(cf["corr_with_w"]["pearson"], 2))
            _set(m, t + "ClosedN" + suf, str(cf["models"]))
            _set(m, t + "ShiftCorr" + suf, num(cf["corr_shift_with_w_all_models"]["pearson"], 2))
            fo = T["sets"]["discovery"][sc]["forms"]
            _set(m, t + "RmsPooled" + suf, num(fo["pooled"]["rms_pp"], 1))
            _set(m, t + "RmsShort" + suf, num(fo["short_stratum"]["rms_pp"], 1))
            cc = T["corrected_choice"][sc]
            for blk, pre in (("in_sample_discovery", "Choice"), ("validation_means", "ChoiceVal")):
                for k, kt in (("forced_gain", "Fd"), ("corrected", "Corr"), ("corrected_short", "Short")):
                    _set(m, t + pre + kt + suf, num(cc[blk][k]["regret"], 1))
            # models chosen by both forms of the (1 - w) rule (the choices come
            # from discovery means, so the in-sample and validation choices are the same models)
            b = cc["validation_means"]
            if "corrected_short" in b:
                wc, ws = b["corrected"]["weights"], b["corrected_short"]["weights"]
                _set(m, t + "ChoiceBoth" + suf, num(sum(min(x, y) for x, y in zip(wc, ws)), 0))
                _set(m, t + "ChoiceN" + suf, num(sum(wc), 0))
    return _check(m)


def hotpot_d27_macros(R):
    """results/hotpot/hotpot_report.json, keys split_pilot and decomposition (run_hotpot.py's full run). hqSplit:
    the fresh-configuration block as splitp (<ReqOnly|FdOnly|Split><Req|Fd>, Worst, BestSingle, BestSingleWorst, Adv,
    Holds), hqSplitConfigs, hqSplitLofoBetter of hqSplitLofoFamilies; hqSplitAllAdv and hqSplitAllHolds on all distinct
    validation configurations. hqDec: VarLongPct (validation draws, %), VarThresholdPct, VarDiscPct (the discovery
    value saved), VarHolds; RegretCorr, RegretFd, RegretShort (requested arm, pp, 1 dp), ChoiceHolds; FreshRegretCorr,
    FreshRegretFd, FreshDiff with Lo, Hi. Empty before the full run."""
    m = {}
    E = R.get("split_pilot")
    if E:
        c = E["sets"].get("fresh") or {}
        if "regret" in c:
            _split_block(m, "hqSplit", c)
            _set(m, "hqSplitConfigs", str(c["n_configurations"]))
            lo = c["leave_one_family_out"]
            _set(m, "hqSplitLofoBetter", str(lo["split_better_count"]))
            _set(m, "hqSplitLofoFamilies", str(lo["families"]))
        a = E["sets"].get("all_distinct") or {}
        if "regret" in a:
            _set(m, "hqSplitAllAdv", num(a["advantage"], 2))
            _set(m, "hqSplitAllHolds", _yn(a.get("split_better")))
    D = R.get("decomposition")
    if D:
        v, c = D["statements"]["variance_share"], D["statements"]["choice"]

        def pct(x):
            return num(None if x is None else 100 * x, 0)
        _set(m, "hqDecVarLongPct", pct(v["observed"]))
        _set(m, "hqDecVarThresholdPct", pct(v["threshold"]))
        _set(m, "hqDecVarDiscPct", pct(v["discovery_value"]))
        _set(m, "hqDecVarHolds", _yn(v["holds"]))
        _set(m, "hqDecRegretCorr", num(c["regret_corrected"], 1))
        _set(m, "hqDecRegretFd", num(c["regret_forced_gain"], 1))
        _set(m, "hqDecRegretShort", num(c["regret_corrected_short"], 1))
        _set(m, "hqDecChoiceHolds", _yn(c["holds"]))
        f = D.get("fresh_configurations") or {}
        if "regret" in f:
            _set(m, "hqDecFreshRegretCorr", num(f["regret"]["corrected"], 1))
            _set(m, "hqDecFreshRegretFd", num(f["regret"]["forced_gain"], 1))
            _set(m, "hqDecFreshDiff", num(f["paired_corrected_minus_forced"]["estimate"], 2))
            _lohi(m, "hqDecFreshDiff", f["paired_corrected_minus_forced"].get("ci95"), 2)
    return _check(m)


def arm_wording_mechanism_results(root):
    """results/arm_wording/mechanism_forecast_evaluation.json once run_arm_wording.py's full run has written it on
    scored outputs (arm_wording_results not None); None before."""
    if arm_wording_results(root) is None:
        return None
    p = os.path.join(root, "results", "arm_wording", "mechanism_forecast_evaluation.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def arm_wording_mechanism_macros(E):
    """results/arm_wording/mechanism_forecast_evaluation.json, prefix awMech<T> (Gsm, Math): Rho (2 dp), WMedRange
    (median range of the screen's w over the four clauses, 2 dp), WMoved (models whose w moves by more than 0.2 from the
    current clause), ScreenCollCorr (screen w against the collection's under the current clause, 2 dp); S1 ClauseObs,
    ClausePredLo, ClausePredHi (reversed shares, %, 1 dp), ClauseHolds; S2 CfgObs (%, 1 dp), ClauseVsCfgHolds; S3
    Arm<Zero..Three>Obs, Lo, Hi (%, 1 dp) and Holds per clause, ArmHoldsCount of ArmClauses; S4 SlopeModels,
    SlopePooled (pp per unit w, 1 dp), SlopePositive, RisesHoldsA, RisesHoldsB; S5 SdReq, SdFd (pp, 2 dp), FdLess of
    Models, SteadierHoldsA, SteadierHoldsB. Empty before the full run."""
    m = {}
    if not E:
        return m
    pc = lambda x: num(None if x is None else 100 * x, 1)  # noqa: E731
    for task, x in E["tasks"].items():
        t = "awMech" + _task(task)
        fs = (E.get("forecast_summary") or {}).get(task) or {}
        if fs:
            _set(m, t + "Rho", num(fs["rho"], 2))
            _set(m, t + "WMedRange", num(fs["w_summary"]["median_range_over_clauses"], 2))
            _set(m, t + "WMoved", str(fs["w_summary"]["models_moved_from_current_more_than"]))
            _set(m, t + "ScreenCollCorr", num(fs["w_summary"]["screen_vs_collection_current_clause_pearson"], 2))
        s1, s2, s3 = x["S1_clause_reversal"], x["S2_clause_vs_template"], x["S3_arm_switch"]
        _set(m, t + "ClauseObs", pc(s1["observed"]))
        _set(m, t + "ClausePredLo", pc(s1["predicted_range"][0]))
        _set(m, t + "ClausePredHi", pc(s1["predicted_range"][1]))
        _set(m, t + "ClauseHolds", _yn(s1["holds"]))
        _set(m, t + "CfgObs", pc(s2["configuration_only"]))
        _set(m, t + "ClauseVsCfgHolds", _yn(s2["holds"]))
        for k, r in enumerate(s3["by_clause"]):
            c = t + "Arm" + _spell(k)
            _set(m, c + "Obs", pc(r["observed"]))
            _set(m, c + "Lo", pc(r["predicted_range"][0]))
            _set(m, c + "Hi", pc(r["predicted_range"][1]))
            _set(m, c + "Holds", _yn(r["holds"]))
        _set(m, t + "ArmHoldsCount", str(s3["holds_count"]))
        _set(m, t + "ArmClauses", str(s3["clauses"]))
        s4, s5 = x["S4_requested_rises"], x["S5_forced_steadier"]
        _set(m, t + "SlopeModels", str(s4["models"]))
        _set(m, t + "SlopePooled", num(s4["pooled_slope_pp_per_unit_w"], 1))
        _set(m, t + "SlopePositive", str(s4["positive"]))
        _set(m, t + "RisesHoldsA", _yn(s4["holds_a"]))
        _set(m, t + "RisesHoldsB", _yn(s4["holds_b"]))
        _set(m, t + "SdReq", num(s5["median_sd_requested_pp"], 2))
        _set(m, t + "SdFd", num(s5["median_sd_forced_pp"], 2))
        _set(m, t + "FdLess", str(s5["models_forced_less"]))
        _set(m, t + "Models", str(s5["models"]))
        _set(m, t + "SteadierHoldsA", _yn(s5["holds_a"]))
        _set(m, t + "SteadierHoldsB", _yn(s5["holds_b"]))
    return _check(m)
