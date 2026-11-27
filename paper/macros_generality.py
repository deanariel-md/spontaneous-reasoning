"""Macros of the generality analysis (results/generality/), prefix gen.

generality_macros(R) takes results/generality/generality.json loaded (a dict) and returns {macro name: value}; nothing
is re-estimated. Names: gen + intervention (Fs five-shot, Ex eight exemplars, Thi native thinking, Cot chain of thought)
+ task (Gsm, Math, Bbh) + change (ScorerCon strict vs content, ScorerLen strict vs lenient, ScorerCheng strict vs Cheng
et al.'s extractor, Arm requested vs forced baseline), then
    Rev, RevPairs, RevShare, RevPct, RevTies       reversed pairs, pairs, share, share in % (1 decimal), tied pairs
    RevLo, RevHi                                   model-resampling 95% interval of the share
    RevFamLo, RevFamHi                             family-resampling 95% interval of the share
    M, Configs, Chosen                             models, fresh validation configurations, size of the chosen third
    <A>To<B>, Lo, Hi                               decision cost chosen under A, used under B (pp per chosen model,
                                                   t interval over the fresh configurations), both directions
Native thinking under the scorer change, one row per baseline: genThiGsmReqScorerCon... (requested direct) and
genThiGsmFdScorerCon... (forced direct). A row that could not be computed gets "--" for every value. Native thinking against prompted chain of thought over the
same baseline: genThiGsmVsCotReq... and genThiGsmVsCotFd... (Rev, RevPairs, RevShare, RevPct, RevLo, RevHi). Across
interventions: genScorerRevPctMin/Max (five-shot, eight exemplars, native thinking) and genScorerRevPctWithCotMin/Max.
"""
import macros_added as ma

INTERVENTION = {"fs": "Fs", "ex": "Ex", "thi": "Thi", "cot": "Cot"}
CHANGE = {"content": "ScorerCon", "lenient": "ScorerLen", "cheng": "ScorerCheng", "arm": "Arm"}
PROTO = {"strict": "Strict", "content": "Con", "lenient": "Len", "cheng": "Cheng", "requested": "Req", "forced": "Fd"}
TASK = {"gsm8k": "Gsm", "math": "Math", "bbh": "Bbh"}
DASH = "--"


STEM = {"thi_gsm8k_content": "genThiGsmReqScorerCon", "thifd_gsm8k_content": "genThiGsmFdScorerCon"}


def stem_of(row_id):
    if row_id in STEM:
        return STEM[row_id]
    iv, task, change = row_id.split("_")
    return "gen" + INTERVENTION[iv] + TASK[task] + CHANGE[change]


def _rev(m, s, v):
    ma._set(m, s + "Rev", str(v["reversed"]))
    ma._set(m, s + "RevPairs", str(v["pairs"]))
    ma._set(m, s + "RevShare", ma.num(v["reversed_share"], 3))
    ma._set(m, s + "RevPct", ma.num(100 * v["reversed_share"], 1))
    ma._lohi(m, s + "Rev", (v.get("model_bootstrap") or {}).get("ci95"), 3)


def generality_macros(R):
    m = {}
    for row_id, r in R["rows"].items():
        s = stem_of(row_id)
        a, b = r["protocols"]
        dirs = [PROTO[a] + "To" + PROTO[b], PROTO[b] + "To" + PROTO[a]]
        if not r.get("computed"):
            for k in ("Rev", "RevPairs", "RevShare", "RevPct", "RevTies", "RevLo", "RevHi", "RevFamLo", "RevFamHi",
                      "M", "Configs", "Chosen"):
                ma._set(m, s + k, DASH)
            for d in dirs:
                for k in ("", "Lo", "Hi"):
                    ma._set(m, s + d + k, DASH)
            continue
        v, c = r["reversals"], r["decision_cost"]
        _rev(m, s, v)
        ma._set(m, s + "RevTies", str(v["ties"]))
        ma._lohi(m, s + "RevFam", (v.get("family_bootstrap") or {}).get("ci95"), 3)
        ma._set(m, s + "M", str(r["M"]))
        ma._set(m, s + "Configs", str(c["n_configurations"]))
        ma._set(m, s + "Chosen", str(c["n_chosen"]))
        for (src, dst), d in zip(((a, b), (b, a)), dirs):
            blk = c.get(f"{src}_to_{dst}") or {}
            ma._set(m, s + d, ma.num(blk.get("estimate"), 2))
            ma._lohi(m, s + d, blk.get("ci95"), 2)
    for base, tag in (("requested", "Req"), ("forced", "Fd")):
        v = R["thinking_vs_cot"].get(base)
        if v:
            _rev(m, "genThiGsmVsCot" + tag, v)
    s = R["scorer_reversal_pct"]
    ma._set(m, "genScorerRevPctMin", ma.num(s["other_interventions"]["min"], 1))
    ma._set(m, "genScorerRevPctMax", ma.num(s["other_interventions"]["max"], 1))
    ma._set(m, "genScorerRevPctWithCotMin", ma.num(s["with_cot"]["min"], 1))
    ma._set(m, "genScorerRevPctWithCotMax", ma.num(s["with_cot"]["max"], 1))
    return ma._check(m)
