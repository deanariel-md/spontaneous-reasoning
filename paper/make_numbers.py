"""Write paper/numbers.tex from the analysis outputs.

Dataset 1 values come from results/dataset1/dove_mmlu.json and
results/dataset1/registered_intervals.json. Dataset 2 values come from
results/dataset2/gsm8k.json only; until that file exists every Dataset 2 macro
prints a visible placeholder. The analyses listed in V2 add the "macros" object of
results/<name>/<name>.json for each name in V2 whose file exists. Simulated outputs
(results/dataset2_sim) are never read.

Usage (PowerShell, from the repository root):
    .\\.venv\\Scripts\\python.exe paper\\make_numbers.py
"""
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "numbers.tex")
D1 = os.path.join(ROOT, "results", "dataset1", "dove_mmlu.json")
D1_REG = os.path.join(ROOT, "results", "dataset1", "registered_intervals.json")
D2 = os.path.join(ROOT, "results", "dataset2", "gsm8k.json")

macros = {}


def put(name, value):
    if not name.isalpha():
        raise ValueError(f"macro name must be letters only: {name}")
    macros[name] = value


def f(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def minus(s):
    """A value written by an analysis script with a hyphen or a Unicode minus before a number prints $-$, as f() does;
    "--", ranges and values already in $-$ are left as they are."""
    return re.sub(r"(?<![\w$-])-(?=[\d.])", "$-$", s.replace("−", "$-$"))


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


ROMAN = {1: "i", 2: "ii", 4: "iv", 8: "viii", 12: "xii", 16: "xvi", 24: "xxiv"}

# Tables for the appendix.
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)


def num(x, nd=2):
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "$-$")


# ---------------------------------------------------------------- Dataset 1
# (skipped when results/dataset1/ is absent: the manuscript reports no Dataset 1 value)
if os.path.exists(D1) and os.path.exists(D1_REG):
    d = json.load(open(D1, encoding="utf-8"))
    reg = {r["target"]: r for r in json.load(open(D1_REG, encoding="utf-8"))["rows"]}
    s1, s2, s3 = d["step1"], d["step2"], d["step3"]
    ex = s2["exact"]

    put("daM", str(len(d["models"])))
    put("daFam", str(len(set(d["families"]))))
    put("daSubjects", str(d["n_subjects"]))
    put("daConfigs", comma(d["n_configurations"]))
    put("daKdisc", str(d["allocation"]["K_discovery"]))
    put("daKval", str(d["allocation"]["K_validation"]))

    put("daCben", f(s1["benefit_exact"]["C"], 2))
    put("daCacc", f(s1["accuracy_exact"]["C"], 2))
    put("daGap", f(s1["primary_gap_C_accuracy_minus_C_benefit"], 2))
    for row in s1["agreement_curve_mc"]:
        tag = "Cben" if row["quantity"] == "benefit" else "Cacc"
        put(f"da{tag}K{ROMAN[row['K']]}", f(row["C"], 2))

    pairs = s1["pairwise_differences_benefit"]
    put("daPairMaxDiff", f(max(pairs, key=lambda p: abs(p["expected_difference"]))["expected_difference"], 2))
    put("daPairBeyondLo", pct(min(p["frac_abs_gt_margin"] for p in pairs)))
    put("daPairBeyondHi", pct(max(p["frac_abs_gt_margin"] for p in pairs)))
    put("daPairsBeyondMargin", str(sum(p["expected_beyond_margin"] for p in pairs)))
    put("daPairs", str(len(pairs)))

    cc = s1["capability_claims"]["24"]
    put("daNaiveSignAgree", pct(cc["naive_single_unit"]["sign_agreement_between_blocks"]))
    put("daNaiveSignTarget", pct(cc["naive_single_unit"]["sign_matches_target"]))
    put("daNaiveHsd", f(cc["naive_single_unit"]["sd_H"], 1))
    put("daRepHsd", f(cc["replicated"]["sd_H"], 1))
    put("daRepSignAgree", pct(cc["replicated"]["sign_agreement_between_blocks"], 2))

    put("daMeanGain", f(ex["mean_gain"], 1))
    put("daMeanBase", f(ex["mean_baseline"], 1))
    put("daVdelta", f(ex["V_delta"], 2))
    put("daGainSD", f(ex["gain_sd"], 2))
    put("daA", f(ex["A"], 2))
    put("daVtheta", f(ex["V_theta"], 1))
    put("daH", f(ex["H"], 2))
    put("daRben", f(ex["R_benefit"], 2))
    put("daRacc", f(ex["R_accuracy"], 2))
    put("daWgain", f(ex["noise_W_gain"], 1))
    put("daEcovGB", f(ex["coupling"]["E_cov_G_B_one_draw"], 2))
    put("daCovOneDraw", f(ex["coupling"]["E_cov_G_B_one_draw"], 2))
    put("daHnaive", f(ex["naive_single_unit"]["E_H"], 2))
    # The expected one-draw capability slope, pp of gain per pp of baseline (= -H / 10).
    put("daSlopeNaive", f(-ex["naive_single_unit"]["E_H"] / 10, 3))
    put("daHnaivePpos", pct(ex["naive_single_unit"]["P_H_positive"]))
    gmin = min(zip(ex["theta"], ex["delta"]))
    gmax = max(zip(ex["theta"], ex["delta"]))
    put("daDeltaWeakest", f(gmin[1], 1))
    put("daDeltaStrongest", f(gmax[1], 1))
    put("daDeltaMin", f(min(ex["delta"]), 1))
    put("daDeltaMax", f(max(ex["delta"]), 1))

    rk = s2["replicated_K_discovery"]
    put("daVdeltaRepLo", f(rk["V_delta"]["q025"], 1))
    put("daVdeltaRepHi", f(rk["V_delta"]["q975"], 1))
    put("daVdeltaPabove", pct(rk["V_delta"]["P_above_margin"]))

    h = reg["H"]
    put("daHreg", f(h["estimate"], 2))
    put("daHregLo", f(h["ci95"][0], 2))
    put("daHregHi", f(h["ci95"][1], 2))
    v = reg["V_delta one-sided 95% upper bound"]
    put("daVdeltaReg", f(v["estimate"], 2))
    put("daVdeltaRegUpper", f(v["upper"], 2))

    name = {"constant": "Const", "own_gain": "Own", "averaged_ols": "Ols", "replicated": "Rep", "single_prompt": "Single"}
    RD_NAME = dict(name)   # kept apart: the loop over the analyses below reuse the name `name`
    for c in s3["registered"]["comparisons"]:
        stem = "daRD" + name[c["f"]] + name[c["g"]]
        put(stem, f(c["estimate"], 2))
        put(stem + "Lo", f(c["provisional_ci"][0], 2))
        put(stem + "Hi", f(c["provisional_ci"][1], 2))
    for r in s3["procedure_mc"]["risks"]:
        put("daRisk" + name[r["predictor"]] + "K" + ROMAN[r["K"]], f(r["mean_exact_risk"], 1))

    ctr = {c["scenario"]: c for c in d["controls"]["scenarios"]}
    nulls = ["null (baseline population)", "null (intervention population)", "constant gain +5 pp"]
    eff = [k for k in ctr if k not in nulls]
    put("daCtrlVdeltaFalse", pct(max(ctr[k]["V_delta"]["P_above_margin"] for k in nulls), 2))
    put("daCtrlHFalseBase", pct(ctr[nulls[0]]["H_replicated"]["P_abs_above_margin"], 1))
    put("daCtrlHFalseConst", pct(ctr[nulls[2]]["H_replicated"]["P_abs_above_margin"], 1))
    put("daCtrlHFalseMax", pct(max(c["H_replicated"]["P_abs_above_margin"] for c in ctr.values() if abs(c["truth"]["H"]) < 1e-9), 0))
    het = [k for k in eff if "beta" not in k]
    cap = [k for k in eff if "beta" in k]
    put("daCtrlVdeltaDetLo", pct(min(ctr[k]["V_delta"]["P_above_margin"] for k in eff)))
    put("daCtrlVdeltaDetHi", pct(max(ctr[k]["V_delta"]["P_above_margin"] for k in eff)))
    put("daCtrlHDetLo", pct(min(ctr[k]["H_replicated"]["P_abs_above_margin"] for k in cap), 2))
    put("daCtrlNaiveSDLo", f(min(c["H_naive"]["sd"] for c in ctr.values()), 1))
    put("daCtrlNaiveSDHi", f(max(c["H_naive"]["sd"] for c in ctr.values()), 1))
    put("daCtrlRepSDLo", f(min(c["H_replicated"]["sd"] for c in ctr.values()), 2))
    put("daCtrlRepSDHi", f(max(c["H_replicated"]["sd"] for c in ctr.values()), 2))
    put("daCtrlN", str(len(ctr)))

    sens = {x["label"]: x for x in d["sensitivities"]}
    fb = [x for k, x in sens.items() if k.startswith("family-balanced")]
    ws = [x for k, x in sens.items() if k.startswith("wording/separator")]
    if fb:
        put("daFamBalGainSD", f(fb[0]["gain_sd"], 2))
        put("daFamBalGap", f(fb[0]["gap"], 2))
    if ws:
        put("daWsCben", f(ws[0]["C_benefit"], 2))
        put("daWsCacc", f(ws[0]["C_accuracy"], 2))
        put("daWsConfigs", str(91))
    put("daSensMinGap", f(min(x["gap"] for x in d["sensitivities"]), 2))
    put("daSensMaxH", f(max(x["H"] for x in d["sensitivities"]), 2))

    # Extra benefit of a model 10 pp stronger at baseline (= -H), with the interval flipped accordingly.
    put("daStrongerGain", f(-h["estimate"], 1))
    put("daStrongerGainLo", f(-h["ci95"][1], 1))
    put("daStrongerGainHi", f(-h["ci95"][0], 1))


    # The capability slope (pp of gain per pp of baseline) is -H / 10, so the H columns become slope columns: the true
    # value, the mean and the SDs divide by -10 and 10, and the share beyond the margin (|H| > 1, |slope| > 0.1) is the same.
    rows = []
    for c in d["controls"]["scenarios"]:
        rows.append(" & ".join([
            (c["scenario"][0].upper() + c["scenario"][1:]).replace("beta -0.3", "$\\beta=-0.3$").replace("beta +0.3", "$\\beta=+0.3$").replace("+6", "$+$6").replace("+5", "$+$5"),
            num(c["truth"]["V_delta"], 2), num(c["V_delta"]["mean"], 2), pct(c["V_delta"]["P_above_margin"], 1),
            num(-c["truth"]["H"] / 10, 3), num(-c["H_replicated"]["mean"] / 10, 3),
            pct(c["H_replicated"]["P_abs_above_margin"], 1),
            num(c["H_naive"]["sd"] / 10, 3), num(c["H_replicated"]["sd"] / 10, 3)]) + r" \\")
    with open(os.path.join(TAB, "d1_controls.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/dataset1/dove_mmlu.json\n")
        fh.write("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n")
        fh.write("& \\multicolumn{3}{c}{$V_\\delta$ (pp$^2$)} & \\multicolumn{3}{c}{Capability slope} & \\multicolumn{2}{c}{SD of the slope}\\\\\n")
        fh.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}\n")
        fh.write("Scenario & True & Mean & $>$margin & True & Mean & $>$margin & 1 draw & $K{=}24$\\\\\n\\midrule\n")
        fh.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    rows = []
    prim = ("Primary", len(d["models"]), s1["benefit_exact"]["C"], s1["accuracy_exact"]["C"],
            s1["primary_gap_C_accuracy_minus_C_benefit"], ex["gain_sd"], ex["H"])
    rows.append(prim)
    for x in d["sensitivities"]:
        rows.append((x["label"].replace("leave out family", "Leave out").replace("family-balanced (family-mean units)", "Family means").replace("wording/separator changes only (91 configurations)", "Wording and separators only"),
                     x["M"], x["C_benefit"], x["C_accuracy"], x["gap"], x["gain_sd"], x["H"]))
    with open(os.path.join(TAB, "d1_sensitivity.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/dataset1/dove_mmlu.json\n")
        fh.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n")
        fh.write("Variant & $M$ & $\\Cben$ & $\\Cacc$ & Gap & Gain SD & Slope\\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r[0]} & {r[1]} & {num(r[2])} & {num(r[3])} & {num(r[4])} & {num(r[5])} & {num(-r[6] / 10, 3)}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    rows = []
    for i, m in enumerate(d["models"]):
        rows.append(f"{m} & {d['families'][i]} & {num(ex['theta'][i], 1)} & {num(ex['delta'][i], 2)}\\\\")
    with open(os.path.join(TAB, "d1_panel.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/dataset1/dove_mmlu.json\n")
        fh.write("\\begin{tabular}{@{}llrr@{}}\n\\toprule\n")
        fh.write("Model & Family & $\\theta$ (pp) & $\\delta$ (pp)\\\\\n\\midrule\n")
        fh.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Dataset 2
D2_KEYS = ["dbM", "dbFam", "dbKdisc", "dbKval", "dbCben", "dbCacc", "dbGap", "dbGapLo", "dbGapHi",
           "dbVdelta", "dbVdeltaLo", "dbVdeltaHi", "dbVdeltaUpper", "dbGainSD", "dbH", "dbHset",
           "dbRben", "dbRacc", "dbMeanGain", "dbMeanBase",
           "dbRDConstOwn", "dbRDConstOwnLo", "dbRDConstOwnHi",
           "dbRDOwnRep", "dbRDOwnRepLo", "dbRDOwnRepHi",
           "dbRDOwnOls", "dbRDOwnOlsLo", "dbRDOwnOlsHi",
           "dbRDConstSingle", "dbRDConstSingleLo", "dbRDConstSingleHi"]
def d2_macros(path, p):
    """Steps 1-3 of one results/dataset2/<label>.json under the macro prefix p (db: GSM8K-19; gt: GSM8K-30;
    bh: BIG-Bench Hard)."""
    e = json.load(open(path, encoding="utf-8"))
    t1, t2, t3 = e["step1"], e["step2"], e["step3"]
    put(p + "M", str(len(e["models"])))
    put(p + "Fam", str(len(set(e["families"]))))
    put(p + "Kdisc", str(e["allocation"]["K_discovery"]))
    put(p + "Kval", str(e["allocation"]["K_validation"]))
    # Three decimals: the GSM8K agreements differ in the third (0.955 and 0.945) and the gap's upper end is -0.002.
    put(p + "Cben", f(t1["C_benefit"]["C"], 3))
    put(p + "Cacc", f(t1["C_accuracy"]["C"], 3))
    put(p + "Gap", f(t1["gap"]["estimate"], 3))
    put(p + "GapLo", f(t1["gap"]["ci95"][0], 3))
    put(p + "GapHi", f(t1["gap"]["ci95"][1], 3))
    vd = t2["V_delta"]
    # Whole pp^2: V_delta runs to hundreds of pp^2 on GSM8K.
    put(p + "Vdelta", f(vd["estimate"], 0))
    put(p + "VdeltaLo", f(vd["ci95"][0], 0) if vd.get("ci95") else "--")
    put(p + "VdeltaHi", f(vd["ci95"][1], 0) if vd.get("ci95") else "--")
    put(p + "VdeltaUpper", f(vd["upper95"], 0))
    put(p + "GainSD", f(vd["gain_sd"], 2) if vd.get("gain_sd") is not None else "--")
    put(p + "H", f(t2["H"]["estimate"], 2))
    # The Fieller set as the text prints it: an interval when bounded, otherwise its kind in words.
    hs = t2["H"].get("set") or {}
    if hs.get("kind") == "bounded":
        put(p + "Hset", f"[{f(hs['a'], 2)}, {f(hs['b'], 2)}]")
    else:
        put(p + "Hset", str(hs.get("kind", "--")).replace("_", " "))
    put(p + "Rben", f(t2["reliability"]["R_benefit"], 2))
    put(p + "Racc", f(t2["reliability"]["R_accuracy"], 2))
    put(p + "Wgain", f(t2["noise"]["W_gain"], 2))
    put(p + "MeanGain", f(t2["mean_gain"]["estimate"], 1))
    put(p + "MeanBase", f(t2["mean_baseline"], 1))
    for c in t3["registered"]:
        stem = p + "RD" + RD_NAME[c["f"]] + RD_NAME[c["g"]]
        put(stem, f(c["estimate"], 2))
        ci = c.get("ci95") or c.get("provisional_ci")
        if ci:
            put(stem + "Lo", f(ci[0], 2))
            put(stem + "Hi", f(ci[1], 2))


if os.path.exists(D2):
    d2_macros(D2, "db")
for k in D2_KEYS:
    macros.setdefault(k, "\\tbd{" + k[2:] + "}")
# The 30-model GSM8K panel and BIG-Bench Hard, the same macros under their own prefix.
for label, p in (("gsm8k30", "gt"), ("bbh", "bh")):
    path = os.path.join(ROOT, "results", "dataset2", f"{label}.json")
    if os.path.exists(path):
        d2_macros(path, p)
# The forced-direct arm: Steps 1-3 with forced-direct as the baseline, prefixes fsg and fsb.
for label, p in (("forced_gsm8k", "fsg"), ("forced_bbh", "fsb"), ("math/forced_math", "fsm")):
    path = os.path.join(ROOT, "results", "forced_direct", f"{label}.json")
    if os.path.exists(path):
        d2_macros(path, p)
# MATH-500: Steps 1-3 of the collection, prefix mt.
MATH = os.path.join(ROOT, "results", "dataset2", "math.json")
if os.path.exists(MATH):
    d2_macros(MATH, "mt")
    # the item count: MATH-500's 398 numeric target items
    _mi = json.load(open(os.path.join(ROOT, "manifests", "math_items.json"), encoding="utf-8"))
    put("mtItems", f"{_mi['counts']['target_numeric']:,}")
# Cross-task C.3 and C.5 (results/dataset2/tasks.json), prefix ct.
TASKS = os.path.join(ROOT, "results", "dataset2", "tasks.json")
if os.path.exists(TASKS):
    tk = json.load(open(TASKS, encoding="utf-8"))
    put("ctM", str(len(tk["models"])))
    for scope, sp in (("two_tasks", "Two"), ("within_bbh", "Bbh")):
        dec = tk[scope]["decomposition"]
        for comp, cp in (("average_task_profile", "Profile"), ("model_by_task", "ModelTask"),
                         ("true_within_task", "TrueWithin"), ("prompt_noise", "Noise")):
            put("ct" + sp + cp, f(dec[comp]["estimate"], 1))
        sh = dec.get("share_profile_of_true_within")
        put("ct" + sp + "ProfileShare", "--" if sh is None else pct(sh))
    for case, cp in (("gsm8k_to_bbh", "GsmToBbh"), ("bbh_to_gsm8k", "BbhToGsm")):
        r = tk["two_tasks"]["transfer"][case]
        put("ct" + cp + "Diff", f(r["difference"]["estimate"], 1))
        put("ct" + cp + "DiffLo", f(r["difference"]["ci95"][0], 1))
        put("ct" + cp + "DiffHi", f(r["difference"]["ci95"][1], 1))
    ts = tk["within_bbh"]["transfer_summary"]
    put("ctBbhFamilies", str(ts["n_families"]))
    put("ctBbhFamPositive", str(ts["positive_difference"]))
    put("ctBbhFamNegative", str(ts["negative_difference"]))

# The analyses in V2: each results/<name>/<name>.json carries a "macros" object of
# ready-formatted values under its own prefix. A missing file adds nothing; a name another source already set refuses.
V2 = ["reliability_map", "noise_anatomy", "instability", "nonoverlap_validation", "cheng_scales", "transfer_tasks",
      "instability_tasks", "map_panels", "forced_direct", "vdelta_draws", "capability_index", "capability_rule",
      "late_answers", "pilot_promise", "shrunken_pilots", "scoring_audit", "outcomes", "outcomes_lenient",
      "outcomes_content", "content_scoring", "specification", "pilots_v3", "cheng_panel", "patterns", "exemplar",
      "thinking", "outcomes_math", "outcomes_lenient_math", "outcomes_content_math", "math_prediction", "math_primary",
      "thinking_pilot", "rule_budgets", "planning_rule", "scoring_audit_v2", "tipping_points", "late_generation",
      "thinking_capcheck"]
v2_present = []
# A task run apart writes its own file under its analysis's folder (MATH-500's forced arm: results/forced_direct/math/).
V2_APART = {"forced_direct/math": os.path.join(ROOT, "results", "forced_direct", "math", "forced_direct.json")}
for name in V2 + list(V2_APART):
    path = V2_APART.get(name) or os.path.join(ROOT, "results", name, f"{name}.json")
    if not os.path.exists(path):
        continue
    for k, v in json.load(open(path, encoding="utf-8")).get("macros", {}).items():
        if k in macros:
            raise ValueError(f"{name}: macro {k} is already set")
        put(k, minus(str(v)))
    v2_present.append(name)


def res(name):
    path = os.path.join(ROOT, "results", name, f"{name}.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def put_new(name, value):
    """Values derived here from the results files (the names of paper/pending_numbers.tex); a name an analysis
    already set refuses."""
    if name in macros:
        raise ValueError(f"macro {name} is already set")
    put(name, value)


# Capability slope (pp of gain per pp of baseline, = -H / 10) of the GSM8K-19 scoring variants, with the Fieller set
# flipped accordingly.
ins_r = res("instability")
if ins_r:
    for va, stem in (("variant_a", "instVarASlope"), ("variant_b", "instVarBSlope"), ("lenient", "instLenientSlope")):
        h = ins_r["variants"][va]["H"]
        put_new(stem, f(-h["estimate"] / 10, 3))
        s = h.get("set") or {}
        if s.get("kind") == "bounded":
            put_new(stem + "Lo", f(-s["b"] / 10, 3))
            put_new(stem + "Hi", f(-s["a"] / 10, 3))
# Source slopes of gain on the external index in the cross-task transfer (28 shared models, discovery draws).
tr_r = res("transfer_tasks")
if tr_r:
    for case, stem in (("gsm8k_to_bbh", "trGsmToBbhSlopeEci"), ("bbh_to_gsm8k", "trBbhToGsmSlopeEci")):
        put_new(stem, f(tr_r["cases"][case]["slopes"]["capability_eci_pp_per_eci"], 3))
# Cheng et al. reproduction: Part A cells whose published accuracy lies inside our 95% interval; Part B group means.
rep = os.path.join(ROOT, "results", "reproduction", "summary.json")
if os.path.exists(rep):
    acc = json.load(open(rep, encoding="utf-8"))["numerical"]["accuracy"]
    put_new("chengPartAInside", str(sum(bool(x.get("published_inside_interval")) for x in acc)))
    put_new("chengPartACells", str(sum(x.get("published") is not None for x in acc)))
ch_r = res("cheng_scales")
if ch_r:
    for grp, stem in (("strong", "Strong"), ("weak", "Weak")):
        g = ch_r["by_group"][grp]
        put_new(f"cheng{stem}MeanBenefit", f(g["benefit"], 2))
        put_new(f"cheng{stem}ZeroShot", f(g["zero_shot"], 1))
        put_new(f"cheng{stem}N", str(g["n"]))
# Pilot test of the map: settings whose constant-rule risk is positive (the others cannot be beaten in share terms).
mp_r = res("map_panels")
if mp_r:
    rc = [r["risk_constant"] for r in mp_r["pilot_rows"]]
    put_new("mpPilotNpos", str(sum(x > 0 for x in rc)))
    put_new("mpPilotNonpos", str(sum(x <= 0 for x in rc)))
    # Draws to average for 80% benefit agreement in the exemplar contrast (src/estimators.prompts_needed_gaussian).
    star = math.sin(0.3 * math.pi)
    for sub, stem in (("bf16", "mpChengKeighty"), ("all", "mpChengAllKeighty")):
        r_ = mp_r["cheng"][sub]["point"]["benefit"]["R"]
        put_new(stem, "--" if not 0 < r_ <= 1 else str(max(1, math.ceil(star * (1 - r_) / (r_ * (1 - star))))))
# Non-overlapping validation: Overlap counts validation draws that reuse a discovery configuration, Shared counts the
# distinct configurations they reuse.
for t in ("Gsm", "Bbh"):
    if f"nonov{t}Overlap" in macros:
        put_new(f"nonov{t}ReusedDraws", macros[f"nonov{t}Overlap"])
        put_new(f"nonov{t}ReusedConfigs", macros[f"nonov{t}Shared"])
# Forced-direct baseline: C_benefit, R_benefit and the gap of Steps 1-2 (prefixes fsg, fsb) under the fd names.
for src_p, t in (("fsg", "Gsm"), ("fsb", "Bbh")):
    for k in ("Cben", "Cacc", "Gap", "GapLo", "GapHi", "Rben", "Racc", "Vdelta", "VdeltaLo", "VdeltaHi", "H", "Wgain"):
        if src_p + k in macros:
            put_new(f"fd{t}{k}", macros[src_p + k])
# Forced-direct shape for the variants table: vertex of the descriptive quadratic of gain on forced-direct baseline
# (a peak when concave, a minimum when convex).
fd_r = res("forced_direct")
if fd_r and not fd_r.get("dry_run"):
    for task, t in (("gsm8k", "Gsm"), ("bbh", "Bbh")):
        sh = fd_r["tasks"][task]["shape"]
        put_new(f"fd{t}Vertex", "--" if sh["vertex_baseline"] is None else f(sh["vertex_baseline"], 0))
        put_new(f"fd{t}Curvature", sh["curvature"])
        # share of V_delta the forced-direct baseline accounts for, A^2 / (V_x V_delta), as \eciShareBase...
        d = fd_r["tasks"][task]["scale"]["points"]["discovery"]
        ok = d["V_x"] > 0 and d["V_y"] > 0
        put_new(f"fd{t}ShareBase", f(d["A"] ** 2 / (d["V_x"] * d["V_y"]), 2) if ok else "--")
# MATH-500's quadratic, the curve Figure 2 draws over all 29 models (results/forced_direct/math/): \fdMathVertex, the
# peak of gain on forced accuracy; \mathPredForcedVertex is the GSM8K curve's vertex used by the prediction.
fdm_p = os.path.join(ROOT, "results", "forced_direct", "math", "forced_direct.json")
if os.path.exists(fdm_p):
    sh = json.load(open(fdm_p, encoding="utf-8"))["tasks"]["math"]["shape"]
    put_new("fdMathVertex", "--" if sh["vertex_baseline"] is None else f(sh["vertex_baseline"], 0))
    put_new("fdMathCurvature", sh["curvature"])


# The added results. The Steps 1-3 files of the five-shot and
# constrained contrasts get the standard Dataset 2 macros (prefixes fewshotGsm, fewshotBbh, fewshotMath, gcBbh); the
# headline numbers come from paper/macros_added.py (prefixes fewshot, gc, tv, dec, dstrict, tregret, pceil, pubre). The
# new map points are in the macros objects of reliability_map, map_panels and noise_anatomy read above. A name already
# set refuses, so no earlier macro can change.
import macros_added  # noqa: E402


def load_result(*parts):
    path = os.path.join(ROOT, "results", *parts)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


ADDED_STEPS = ((("fewshot", "fewshot_gsm8k.json"), "fewshotGsm"), (("fewshot", "fewshot_bbh.json"), "fewshotBbh"),
               (("fewshot", "math", "fewshot_math.json"), "fewshotMath"),
               (("forced_constrained", "forced_constrained_bbh.json"), "gcBbh"))
for parts, p in ADDED_STEPS:
    path = os.path.join(ROOT, "results", *parts)
    if os.path.exists(path):
        taken = [k for k in macros if k.startswith(p)]
        if taken:
            raise ValueError(f"prefix {p} is already used: {taken[:5]}")
        d2_macros(path, p)
_fs = [x for x in (load_result("fewshot", "fewshot.json"), load_result("fewshot", "math", "fewshot.json")) if x]
_fs_steps = {t: load_result(*parts) for t, parts in (("gsm8k", ("fewshot", "fewshot_gsm8k.json")),
                                                      ("bbh", ("fewshot", "fewshot_bbh.json")),
                                                      ("math", ("fewshot", "math", "fewshot_math.json")))}
_added = [macros_added.fewshot_macros(_fs, _fs_steps)]
for fn, parts, extra in ((macros_added.constrained_macros, ("forced_constrained", "forced_constrained.json"),
                          ("forced_constrained", "forced_constrained_bbh.json")),
                         (macros_added.vendor_macros, ("thinking_vendor", "thinking_vendor.json"), None),
                         (macros_added.decision_macros, ("decision", "decision.json"), None),
                         (macros_added.strict_macros, ("decision_strict", "decision_strict.json"), None),
                         (macros_added.transfer_macros, ("transfer_regret", "transfer_regret.json"), None),
                         (macros_added.ceiling_macros, ("protocol_ceiling", "protocol_ceiling.json"), None),
                         (macros_added.published_macros, ("published_recompute", "published_recompute.json"), None)):
    r_ = load_result(*parts)
    if r_ is not None:
        _added.append(fn(r_, load_result(*extra)) if extra else fn(r_))
for block in _added:
    for k, v in block.items():
        put_new(k, v)

# The results that had no
# macros (prefixes ae, mres, gdiff, rsw, hcap, lowr), further values under the prefixes above (dstrict, pceil, fewshot,
# gc, tv) and the validation-draw agreement (CbenVal under da, db, gt, bh, mt). All read from results files.
_more = []
_cmp = os.path.join(ROOT, "results", "fewshot", "tables_fewshot_bbh", "compliance_by_model_arm.csv")
if _fs:
    import csv  # noqa: E402
    _rows = list(csv.DictReader(open(_cmp, encoding="utf-8"))) if os.path.exists(_cmp) else None
    _more.append(macros_added.fewshot_more_macros(_fs, _rows))
for fn, parts in ((macros_added.strict_more_macros, ("decision_strict", "decision_strict.json")),
                  (macros_added.ceiling_more_macros, ("protocol_ceiling", "protocol_ceiling.json")),
                  (macros_added.constrained_more_macros, ("forced_constrained", "forced_constrained.json")),
                  (macros_added.vendor_more_macros, ("thinking_vendor", "thinking_vendor.json")),
                  (macros_added.arm_effect_macros, ("arm_effect", "arm_effect.json")),
                  (macros_added.resampling_macros, ("model_resampling", "model_resampling.json")),
                  (macros_added.difficulty_macros, ("gsm8k_difficulty", "gsm8k_difficulty.json")),
                  (macros_added.sweep_macros, ("readable_sweep", "readable_sweep.json")),
                  (macros_added.heldout_macros, ("heldout_capability", "heldout_capability.json")),
                  (macros_added.lowr_macros, ("lowr_map", "lowr_map.json")),
                  (macros_added.validation_agreement_macros, ("reliability_map", "reliability_map.json")),
                  (macros_added.decision_cross_macros, ("decision", "decision.json")),
                  # the disjoint-draw arm switch and the content-scored
                  # arm limit (its interval from content_arm_limit.json, written by scripts/run_content_arm_ci.py; "--"
                  # while that file is absent), the map against validation draws, the model-bootstrap counts, the
                  # settings of the low-R map
                  (lambda r_: macros_added.ceiling_disjoint_macros(r_, macros_added.content_arm_results(ROOT)),
                   ("protocol_ceiling", "protocol_ceiling.json")),
                  (macros_added.map_validation_macros, ("reliability_map", "reliability_map.json")),
                  (macros_added.resampling_count_macros, ("model_resampling", "model_resampling.json")),
                  (macros_added.lowr_settings_macros, ("lowr_map", "lowr_map.json")),
                  # both directions of the wrong-baseline decision cost (prefix ddir) and the
                  # requested baseline as a mixture, exploratory on three benchmarks (prefix bmix)
                  (macros_added.decision_directions_macros, ("decision_directions", "decision_directions.json")),
                  (macros_added.baseline_mixture_macros, ("baseline_mixture", "baseline_mixture.json")),
                  # the split pilot (prefix splitp) and the exact decomposition (prefix bmixDec),
                  # both exploratory on three benchmarks
                  (macros_added.split_pilot_macros, ("split_pilot", "split_pilot.json")),
                  (macros_added.decomposition_macros, ("baseline_mixture", "baseline_mixture.json")),
                  # HotpotQA as target of the transfer cases, read from its fair comparison
                  (macros_added.transfer_added_macros, ("transfer_regret", "transfer_regret.json"))):
    r_ = load_result(*parts)
    if r_ is not None:
        _more.append(fn(r_))
# HotpotQA and the arm wording, taken once they are scored and skipped until then. HotpotQA's
# Steps 1-3 files get the standard Dataset 2 macros (prefixes hqReq, hqFd, hqFs); its report and the arm wording's
# results get prefixes hq and aw. A later analysis that adds HotpotQA as a task (decision, protocol ceiling, map) comes
# in through the readers above, under the task tag Hq.
for path, p in macros_added.hotpot_steps(ROOT):
    taken = [k for k in macros if k.startswith(p)]
    if taken:
        raise ValueError(f"prefix {p} is already used: {taken[:5]}")
    d2_macros(path, p)
_hq = macros_added.hotpot_report(ROOT)
if _hq is not None:
    _more.append(macros_added.hotpot_macros(_hq))
    _more.append(macros_added.hotpot_directions_macros(_hq))   # both directions and the mechanism forecast
    _more.append(macros_added.hotpot_d27_macros(_hq))   # the split pilot and the decomposition rule
_aw = macros_added.arm_wording_results(ROOT)
if _aw is not None:
    _more.append(macros_added.arm_wording_macros(_aw))
    # the mechanism forecast saved before the arm-wording run was scored, evaluated by the full run (prefix awMech)
    _more.append(macros_added.arm_wording_mechanism_macros(macros_added.arm_wording_mechanism_results(ROOT)))
# The two-rater human audit (scripts/run_human_audit.py), prefix hra, taken once
# the ratings are exported and the script has run; skipped with a status line until then.
_hra = macros_added.human_audit_results(ROOT)
if _hra is not None:
    _more.append(macros_added.human_audit_macros(_hra))
    _more.append(macros_added.human_audit_rater_macros(_hra))   # each rater alone, control answers
# Each module reads its own results file and is skipped
# until that file exists: gen (the same tests on the other interventions), wck (correctness checks), hrt (third rater), hqMechFresh (HotpotQA forecast on new configurations), fsr (five-shot
# requested direct), awx (default and firm clauses on the arm-wording design).
import macros_awx  # noqa: E402
import macros_hotpot_mech_fresh  # noqa: E402
import macros_fsr  # noqa: E402
import macros_generality  # noqa: E402
import macros_third_rater  # noqa: E402
import macros_output_checks  # noqa: E402
import macros_hotpot_content  # noqa: E402   # HotpotQA under the content reading
import macros_ladder_orderings  # noqa: E402   # orderings on the size ladder (lad, comb)
import macros_conflict  # noqa: E402   # BIG-Bench Hard letter-text conflict boxes (bbhConflict)
import macros_closed_ladder  # noqa: E402   # closed model (cw), fsCheck, ladder values (lad)
import macros_statement_values  # noqa: E402   # (hq...Content, ladBigKnown, aeRecent, genThi)
import macros_threshold_sweep  # noqa: E402   # the threshold sweep (thr), exploratory
import macros_reading_audit  # noqa: E402   # reading audit (lab); None until labelled
import macros_temp_check  # noqa: E402   # temperature check (tempCheck); None until the results exist
import macros_cw_intervals  # noqa: E402   # flagship intervals (cw)
import macros_threshold_decision  # noqa: E402   # thresholds and the decision (thrDec)
import macros_cap_shared_axis  # noqa: E402   # capability slopes on shared axes (capSlope)
import macros_release_dates  # noqa: E402   # w against release date (wDate)
import macros_cap_reversal  # noqa: E402   # exploratory: own axes, size, year, w (crv)
import macros_cw_full  # noqa: E402   # Qwen4-Max on 398 MATH-500 items (cwFull)
import macros_within_difficulty  # noqa: E402   # difficulty, rewording (diff, reword)
import macros_attr_flips  # noqa: E402   # which part of the baseline difference makes the reversals (attr)
import macros_fewshot_w  # noqa: E402   # w under five-shot answering (fsW)
import macros_bmix_val  # noqa: E402   # Fig. 2c on validation templates (bmixVal)
for _fn, _res in ((macros_generality.generality_macros, load_result("generality", "generality.json")),
                  (macros_output_checks.output_checks_macros, macros_output_checks.output_checks_results(ROOT)),
                  (macros_third_rater.third_rater_macros, macros_third_rater.third_rater_results(ROOT)),
                  (macros_hotpot_mech_fresh.hotpot_mech_fresh_macros, macros_hotpot_mech_fresh.hotpot_mech_fresh_results(ROOT)),
                  (macros_fsr.fsr_macros, macros_fsr.fsr_results(ROOT)),
                  (macros_awx.awx_macros, macros_awx.awx_results(ROOT)),
                  (macros_hotpot_content.hotpot_content_macros, macros_hotpot_content.hotpot_content_results(ROOT)),
                  (macros_ladder_orderings.ladder_orderings_macros,
                   macros_ladder_orderings.ladder_orderings_results(ROOT)),
                  (macros_conflict.conflict_macros, macros_conflict.conflict_results(ROOT)),
                  (macros_closed_ladder.closed_ladder_macros, macros_closed_ladder.closed_ladder_results(ROOT)),
                  (macros_statement_values.statement_values, macros_statement_values.statement_values_results(ROOT)),
                  (macros_threshold_sweep.threshold_sweep_macros, macros_threshold_sweep.threshold_sweep_results(ROOT)),
                  (macros_reading_audit.reading_audit_macros, macros_reading_audit.reading_audit_results(ROOT)),
                  (macros_temp_check.temp_check_macros, macros_temp_check.temp_check_results(ROOT)),
                  (macros_cw_intervals.cw_intervals_macros, macros_cw_intervals.cw_intervals_results(ROOT)),
                  (macros_threshold_decision.threshold_decision_macros,
                   macros_threshold_decision.threshold_decision_results(ROOT)),
                  (macros_cap_shared_axis.cap_shared_axis_macros,
                   macros_cap_shared_axis.cap_shared_axis_results(ROOT)),
                  (macros_cap_shared_axis.cap_count_macros,  # cell counts
                   macros_cap_shared_axis.cap_shared_axis_results(ROOT)),
                  (macros_release_dates.release_dates_macros,
                   macros_release_dates.release_dates_results(ROOT)),
                  (macros_cap_reversal.cap_reversal_macros,
                   macros_cap_reversal.cap_reversal_results(ROOT)),
                  (macros_cw_full.cwfull_macros, macros_cw_full.cwfull_results(ROOT)),
                  (macros_within_difficulty.within_difficulty_macros,
                   macros_within_difficulty.within_difficulty_results(ROOT)),
                  (macros_attr_flips.attr_flips_macros, macros_attr_flips.attr_flips_results(ROOT)),
                  (macros_fewshot_w.fewshot_w_macros, macros_fewshot_w.fewshot_w_results(ROOT)),
                  (macros_bmix_val.bmix_val_macros, macros_bmix_val.bmix_val_results(ROOT))):
    if _res is not None:
        _more.append(_fn(_res))
for k, v in macros_added.pending(ROOT).items():
    print(f"{k}: {v}")
for block in _more:
    for k, v in block.items():
        put_new(k, v)

# Design constants.
ext = json.load(open(os.path.join(ROOT, "manifests", "d2_panel_extension.json"), encoding="utf-8"))
put_new("dtEligible", str(ext["counts"]["panel"]))
for bank, stem in (("gsm8k_bank.json", "gsmBankConfigs"), ("bbh_bank.json", "bbhBankConfigs")):
    put_new(stem, str(len(json.load(open(os.path.join(ROOT, "manifests", bank), encoding="utf-8"))["configurations"])))
import yaml  # noqa: E402
cap = yaml.safe_load(open(os.path.join(ROOT, "config", "decoding.yaml"), encoding="utf-8"))["max_new_tokens"]
put_new("capDirect", comma(cap["direct_answer"]))
put_new("capCot", comma(cap["zero_shot_cot"]))
if tr_r:
    n = set(tr_r["bbh_items_per_task"].values())
    put_new("bbhItemsPerTask", str(n.pop()) if len(n) == 1 else "--")

# Appendix table of the Dataset 2 panel: the 30 checkpoints with family, index,
# membership of the 19-model panel, and the screens' exclusions by task.
def tex(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


reg = {}
for path in ("config/models.yaml", "config/models_extension.yaml"):
    reg.update({k: v for k, v in yaml.safe_load(open(os.path.join(ROOT, path), encoding="utf-8"))["dataset2"]["models"]
                .items() if k != "_template"})
by_repo = {e["repo"]: e for e in ext["panel"]}
excl = {"gsm8k": {}, "bbh": {}}
logs = [os.path.join(ROOT, "manifests", "gsm8k_admissibility.json")] + sorted(
    os.path.join(ROOT, "manifests", "gsm8k_admissibility_extension", p)
    for p in os.listdir(os.path.join(ROOT, "manifests", "gsm8k_admissibility_extension")))
for task, paths in (("gsm8k", logs), ("bbh", [os.path.join(ROOT, "manifests", "bbh_admissibility.json")])):
    for p in paths:
        for x in json.load(open(p, encoding="utf-8"))["excluded_models"]:
            excl[task].setdefault(x["model_key"], []).append(x)
main19 = set(json.load(open(D2, encoding="utf-8"))["models"]) if os.path.exists(D2) else set()
rows = []
for k, v in reg.items():
    e = by_repo.get(v["hf_id"])
    if e is None:
        continue
    rows.append((v["family"], float(e["eci"]), k, e["revision"][:7]))
rule_words = {"direct_answer_meaningful": "direct answers over 128 tokens"}
with open(os.path.join(TAB, "d2_panel.tex"), "w", encoding="utf-8") as fh:
    fh.write("% Generated by paper/make_numbers.py from manifests/d2_panel_extension.json, the registries and the "
             "exclusion logs\n")
    fh.write("\\begin{tabular}{@{}lllrccc@{}}\n\\toprule\n")
    fh.write("Family & Checkpoint & Revision & Index & 19-model panel & GSM8K & BIG-Bench Hard\\\\\n\\midrule\n")
    last = None
    for fam, eci, k, rev in sorted(rows):
        cells = []
        for task in ("gsm8k", "bbh"):
            x = excl[task].get(k)
            cells.append("excluded" if x else "\\checkmark")
        in19 = "\\checkmark" if k in main19 else ""
        fh.write(f"{fam if fam != last else ''} & {tex(k)} & \\texttt{{{rev}}} & {eci:.1f} & {in19} & {cells[0]} & "
                 f"{cells[1]}\\\\\n")
        last = fam
    fh.write("\\bottomrule\n\\end{tabular}\n")
    notes = sorted({f"{tex(k)}: {rule_words.get(x['rule'], tex(x['rule']))} in {x['value']:.1f}\\% of screening "
                    f"outputs ({task_name})" for task, task_name in (("gsm8k", "GSM8K"), ("bbh", "BIG-Bench Hard"))
                    for k, xs in excl[task].items() for x in xs})
    fh.write("% Exclusions: " + "; ".join(notes) + "\n")
put_new("dtExclusionThreshold", "50")
put_new("dtExcludedGsm", str(sum(k in {r[2] for r in rows} for k in excl["gsm8k"])))
put_new("dtExcludedBbh", str(sum(k in {r[2] for r in rows} for k in excl["bbh"])))

# Appendix table of the Cheng et al. reproduction: Part A accuracies (ours, 95% interval, published) and the Part B
# capability slopes per transported configuration on three scales, on the BF16 models (the main panel), with
# the over-all-draws slope of all 15 models as the last row (the sensitivity).
# Retired (the reproduction is in neither the paper nor the SI); the
# last table written is in paper/retired/. Set WRITE_CHENG_TABLE to True to write it again.
WRITE_CHENG_TABLE = False
if WRITE_CHENG_TABLE and os.path.exists(rep) and ch_r:
    with open(os.path.join(TAB, "cheng.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/reproduction/summary.json and "
                 "results/cheng_scales/cheng_scales.json\n")
        fh.write("\\begin{tabular}{@{}llrrrc@{}}\n\\toprule\n")
        fh.write("Model & Arm & Ours & 95\\% interval & Published & Inside\\\\\n\\midrule\n")
        arm = {"zero_shot_cot": "zero-shot", "eight_shot_cot": "8 exemplars"}
        last = None
        for x in sorted(acc, key=lambda r: (r["model_key"], r["arm"] != "zero_shot_cot")):
            fh.write(f"{tex(x['model_key']) if x['model_key'] != last else ''} & {arm.get(x['arm'], tex(x['arm']))} & "
                     f"{x['accuracy']:.1f} & {x['lo']:.1f}--{x['hi']:.1f} & {x['published']:.1f} & "
                     f"{'yes' if x['published_inside_interval'] else 'no'}\\\\\n")
            last = x["model_key"]
        fh.write("\\bottomrule\n\\end{tabular}\n")
        # Part B below Part A in the same file: per-configuration slopes over the BF16 models.
        pb, pall = ch_r["part_b"]["bf16_only"], ch_r["part_b"]["all"]
        fh.write("\n\\par\\medskip\n")
        fh.write("\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n")
        fh.write("Configuration & Points & Logits & Error reduction\\\\\n\\midrule\n")
        for i, cfg in enumerate(ch_r["configurations"]):
            fh.write(f"{tex(cfg)} & " + " & ".join(num(pb[sc]["per_configuration"][i], 3)
                                                     for sc in ("points", "logit", "error_reduction")) + "\\\\\n")
        fh.write("\\midrule\nOver all draws & " + " & ".join(num(pb[sc]["cross_draw"]["slope"], 3)
                                                         for sc in ("points", "logit", "error_reduction")) + "\\\\\n")
        fh.write(f"All {ch_r['n_models']} models, over all draws & " + " & ".join(
            num(pall[sc]["cross_draw"]["slope"], 3) for sc in ("points", "logit", "error_reduction")) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

# Dataset 2 sensitivities (appendix placeholder "format compliance by arm, leave-one-family-out and family-balanced
# estimates"): the rows of results/dataset2/<label>.json in the layout of d1_sensitivity.tex, and compliance by tier
# and arm.
def d2_sens_rows(e):
    t1, t2 = e["step1"], e["step2"]
    rows = [("Primary", len(e["models"]), t1["C_benefit"]["C"], t1["C_accuracy"]["C"], t1["gap"]["estimate"],
             t2["V_delta"]["gain_sd"], t2["H"]["estimate"])]
    s = e["sensitivities"]
    for x in s["families"] + [s["lenient_extraction"]["steps_1_2"]]:
        lab = (x["label"].replace("leave out family ", "Leave out ").replace("family-balanced (5 family-mean units)",
                                                                              "Family means")
               .replace("lenient extraction", "Lenient extraction"))
        sd = x["V_delta"] ** 0.5 if x["V_delta"] >= 0 else float("nan")
        rows.append((lab, x["M"], x["C_benefit"], x["C_accuracy"], x["gap"], sd, x["H"]))
    return rows


d2_res = {lab: json.load(open(os.path.join(ROOT, "results", "dataset2", f"{lab}.json"), encoding="utf-8"))
          for lab in ("gsm8k30", "bbh") if os.path.exists(os.path.join(ROOT, "results", "dataset2", f"{lab}.json"))}
if len(d2_res) == 2:
    with open(os.path.join(TAB, "d2_sensitivity.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/dataset2/gsm8k30.json and bbh.json\n")
        fh.write("\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n")
        fh.write("Task & Variant & $M$ & $\\Cben$ & $\\Cacc$ & Gap & Gain SD & Slope\\\\\n\\midrule\n")
        for k, (lab, title) in enumerate((("gsm8k30", "GSM8K"), ("bbh", "BIG-Bench Hard"))):
            for i, r in enumerate(d2_sens_rows(d2_res[lab])):
                fh.write(f"{title if i == 0 else ''} & {r[0]} & {r[1]} & {num(r[2], 3)} & {num(r[3], 3)} & "
                         f"{num(r[4], 3)} & {num(r[5], 1)} & {num(-r[6] / 10, 3)}\\\\\n")
            if k == 0:
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB, "d2_compliance.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/dataset2/gsm8k30.json and bbh.json "
                 "(sensitivities.compliance); percent of outputs, means over the tier's models\n")
        fh.write("\\begin{tabular}{@{}lllrrr@{}}\n\\toprule\n")
        fh.write("Task & Arm & Index tier & Compliant & Strict correct & Lenient correct\\\\\n\\midrule\n")
        arm_w = {"direct_answer": "Direct", "zero_shot_cot": "Chain of thought"}
        for k, (lab, title) in enumerate((("gsm8k30", "GSM8K"), ("bbh", "BIG-Bench Hard"))):
            rows = d2_res[lab]["sensitivities"]["compliance"]["by_tier_arm"]
            order = {"low": 0, "middle": 1, "high": 2}
            last_arm = None
            for i, r in enumerate(sorted(rows, key=lambda r: (r["arm"] != "direct_answer", order[r["tier"]]))):
                fh.write(f"{title if i == 0 else ''} & {arm_w[r['arm']] if r['arm'] != last_arm else ''} & {r['tier']} & "
                         f"{r['compliance_pct']:.1f} & {r['strict_pct']:.1f} & {r['lenient_pct']:.1f}\\\\\n")
                last_arm = r["arm"]
            if k == 0:
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    for lab, p in (("gsm8k30", "gt"), ("bbh", "bh")):
        rows = d2_sens_rows(d2_res[lab])[1:6]
        sl = [-r[6] / 10 for r in rows]
        put_new(p + "LofoSlopeMin", f(min(sl), 3))
        put_new(p + "LofoSlopeMax", f(max(sl), 3))
        put_new(p + "LofoGapMin", f(min(r[4] for r in rows), 3))
        put_new(p + "LofoGapMax", f(max(r[4] for r in rows), 3))
        fb = d2_sens_rows(d2_res[lab])[6]
        put_new(p + "FamBalSlope", f(-fb[6] / 10, 3))
        put_new(p + "FamBalGap", f(fb[4], 3))

# Leave-one-family-out capability slopes, as the DOVE sensitivity table.
ci_r = res("capability_index")
if ci_r:
    with open(os.path.join(TAB, "d2_lofo.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/capability_index/capability_index.json\n")
        fh.write("\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n")
        fh.write("Task & Left out & $M$ & Slope on baseline & Slope on index\\\\\n\\midrule\n")
        for name, label in (("gsm8k30", "GSM8K"), ("bbh", "BIG-Bench Hard")):
            full = ci_r["slopes"][name]["scales"]["points"]
            rows = [("None", len(ci_r["settings"][name]["models"]), full["baseline"]["slope"], full["index"]["slope"])]
            rows += [(r["left_out"], r["M"], r["baseline"]["slope"], r["index"]["slope"])
                     for r in ci_r["leave_one_family_out"][name]]
            for i, (fam, m, sb, si) in enumerate(rows):
                fh.write(f"{label if i == 0 else ''} & {fam} & {m} & {num(sb, 3)} & {num(si, 3)}\\\\\n")
            if name == "gsm8k30":
                fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

# Native reasoning: each thinking checkpoint's share of pilot outputs at the pilot's cap, the exclusion rule's verdict, and
# for the checkpoints run the share of their outputs cut at the runs' cap.
thp_r = res("thinking_pilot")
if thp_r:
    th_r = res("thinking")
    run_cut = {t: {r["model_key"]: r["native_cut_pct"] for r in th_r["tasks"][t]["per_model"]}
               for t in ("gsm8k", "math") if th_r and t in th_r.get("tasks", {})}
    with open(os.path.join(TAB, "thinking_pilot.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/thinking_pilot/thinking_pilot.json and "
                 "results/thinking/thinking.json\n")
        fh.write("\\begin{tabular}{@{}lrrrlrr@{}}\n\\toprule\n")
        fh.write("& \\multicolumn{3}{c}{Pilot outputs at the cap (\\%)} & & \\multicolumn{2}{c}{Run outputs cut at the "
                 "cap (\\%)}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n")
        fh.write("Checkpoint & GSM8K & MATH & Both & Verdict & GSM8K & MATH\\\\\n\\midrule\n")
        for r in sorted(thp_r["models"], key=lambda r: r["model_key"]):
            k, s = r["model_key"], r["pilot_cap_share"]
            cells = [("--" if s["excluded"] else "pending") if k not in run_cut.get(t, {}) else num(run_cut[t][k], 1)
                     for t in ("gsm8k", "math")]
            fh.write(f"{tex(k)} & {num(100 * r['gsm8k']['share_cut_at_pilot_cap'], 1)} & "
                     f"{num(100 * r['math']['share_cut_at_pilot_cap'], 1)} & {num(100 * s['share'], 1)} & "
                     f"{'excluded' if s['excluded'] else 'run'} & {cells[0]} & {cells[1]}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

# Native reasoning: each checkpoint's mean accuracy in the forced-direct, chain-of-thought and native-
# reasoning arms and its share of native outputs cut at the cap, per task (results/thinking/tables/per_model_<task>.csv).
th_r = res("thinking")
if th_r and all(t in th_r.get("tasks", {}) for t in ("gsm8k", "math")):
    rows = {t: {r["model_key"]: r for r in th_r["tasks"][t]["per_model"]} for t in ("gsm8k", "math")}
    # The latent correlation's jackknife interval (descriptive), beside the analysis's own thi...LatentCorr.
    for t, tp in (("gsm8k", "Gsm"), ("math", "Math")):
        ci = th_r["tasks"][t]["interventions"]["latent_corr"]["ci95"]
        put_new(f"thi{tp}LatentCorrLo", f(ci[0], 2))
        put_new(f"thi{tp}LatentCorrHi", f(ci[1], 2))
    with open(os.path.join(TAB, "thinking_runs.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by paper/make_numbers.py from results/thinking/thinking.json\n")
        fh.write("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n")
        fh.write("& \\multicolumn{4}{c}{GSM8K} & \\multicolumn{4}{c}{MATH-500}\\\\\n"
                 "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n")
        fh.write("Checkpoint & Forced & CoT & Native & Cut (\\%) & Forced & CoT & Native & Cut (\\%)\\\\\n\\midrule\n")
        for k in sorted(set(rows["gsm8k"]) | set(rows["math"])):
            cells = []
            for t in ("gsm8k", "math"):
                r = rows[t].get(k)
                cells += (["--"] * 4 if r is None else
                          [num(r["forced_direct"], 1), num(r["chain_of_thought"], 1), num(r["native_reasoning"], 1),
                           num(r["native_cut_pct"], 1)])
            fh.write(f"{tex(k)} & " + " & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

# "family-balanced" (each family one point) is named "family means". The old names stay as aliases of the new
# ones, and the new names are defined for the Dataset 1 and 2 sensitivities that still carry the old ones, until the
# text uses only the new names.
for old, new in (("FamBal", "FamMeans"), ("WeightingFamilyBalanced", "WeightingFamilyMeans")):
    for k in list(macros):
        twin = k.replace(new, old) if new in k else (k.replace(old, new) if old in k else None)
        if twin and twin not in macros:
            macros[twin] = macros[k]

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% Generated by paper/make_numbers.py. Do not edit by hand.\n")
    fh.write(f"% Dataset 2 source present: {os.path.exists(D2)}\n")
    fh.write(f"% Analyses present: {', '.join(v2_present) or 'none'}\n")
    for k in sorted(macros):
        fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(f"wrote {len(macros)} macros to {OUT}; Dataset 2 present: {os.path.exists(D2)}")
