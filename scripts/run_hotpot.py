r"""HotpotQA analyses: the four-draw predictions, then the analyses of the other tasks.

Run from the repository root; the order is enforced (scripts/score_hotpot.py and this script):
    python scripts/run_hotpot.py --stage cross           (reads no HotpotQA data)
    python scripts/score_hotpot.py --stage discovery4
    python scripts/run_hotpot.py --stage predict
    python scripts/score_hotpot.py --stage discovery
    python scripts/run_hotpot.py --stage discovery
    python scripts/score_hotpot.py --stage all
    python scripts/run_hotpot.py

Contrasts (baseline B, intervention P; the arm labels of the scored table):
    cot_req      chain of thought against requested direct      label hotpot          (the collection's contrast)
    cot_forced   chain of thought against forced direct         label hotpot_forced
    fewshot      five-shot forced against zero-shot forced      label hotpot_fewshot
Scores: strict exact match (primary), lenient exact match, strict token F1 (100 x F1 per item, cell = mean over items).

--stage predict (reads data/processed/hotpot/scored_discovery4.parquet, the sampler's first four discovery draws) saves
manifests/predictions_hotpot_tranche.json, four statements per contrast on strict exact match:
  (a) C of benefit predicted from the four-draw R (reliability_map.pilot_prediction), to be within 0.05 of the
      observed one-draw C on the validation draws;
  (b) the draws K each agreement target (0.80, 0.90) needs, from the same R, against the validation draws' agreement
      between two groups of K;
  (c) switching the baseline arm (requested to forced, chain of thought fixed) reverses more model pairs than a new
      draw does. Switch: the share of model pairs whose gain ordering reverses between the two arms on the same draw
      (same configuration), averaged over draws. New draw: the share that reverses between two distinct draw positions
      within an arm, averaged over all pairs of positions and over the two arms. One tie convention for both: a pair is
      reversed when both gain differences exceed estimators.TIE_TOL in opposite directions; a pair tied on either side
      is not reversed and is counted as tied. The statistic is the same on the four discovery draws and on the 24
      validation draws. The original split-half calculation (mean gains over draws 1-2 against 3-4; 12 against 12 on
      validation) is kept as a labelled secondary output, c_arm_switch_split_half;
  (d) an empirical forecast: own gain from one draw chooses a better third than the capability slope from one draw
      (lower regret, as in the decision analysis) when the four-draw R of benefit exceeds 1/2, and a worse one otherwise. No
      theory gives this threshold: R > 1/2 is where a one-draw centred own-gain estimate beats a constant in squared
      error, which says nothing exact about top-third regret against a capability slope (DOVE's pooled MMLU block:
      R 0.29, yet one-draw own gain has the lower whole-bank regret, 0.70 against 1.03 pp, results/decision/summary.md).
--stage cross (reads no HotpotQA data; GSM8K, BIG-Bench Hard and MATH-500 discovery draws only) saves
manifests/predictions_hotpot_cross_benchmark.json and .npz, the top-third choices for HotpotQA from each source
benchmark's own gain and from each rule fitted on it that needs no HotpotQA call (scripts/hotpot_fair.py).
Saved before any HotpotQA output is scored; an identical rerun keeps the file, a different one is
refused.
--stage discovery (reads data/processed/hotpot/scored_discovery.parquet, every discovery draw; refuses before the
tranche file and the cross-benchmark file exist, or when the cross-benchmark file is dated after the four-draw
scoring) saves, before any validation draw is scored:
  1. map pilot predictions from the first 2 and 4 discovery draws per contrast, manifests/predictions_hotpot_map.json;
  2. the Steps 1-3 prediction rules per contrast on strict and lenient exact match, manifests/predictions_<label>.json
     and _lenient (scripts/run_dataset2.freeze_rules), and on strict token F1, manifests/predictions_<label>_f1.json;
  3. the decision analysis's choices (top third, every rule of the decision analysis) per contrast, from strict exact match,
     manifests/predictions_decision_<label>.json and .npz (pilot seeds sha256('decision/hotpot/K<k>'));
  4. the fair comparison (scripts/hotpot_fair.py): the strict decision design (100-item one-draw pilots,
     shared and item-disjoint designs; linear, quadratic plus index, own gain), manifests/predictions_hotpot_decision_
     strict.*; held-out-family choices, manifests/predictions_hotpot_family.*; the source benchmarks' capability rules
     at the HotpotQA pilot baselines with HotpotQA's own one-draw gain, manifests/predictions_hotpot_cross_discovery.*;
  5. the mechanism forecast (scripts/run_baseline_mixture.hotpot_forecast): each model's
     share w of requested outputs longer than the forced cap of 64 tokens and its gain over the forced arm, the
     predicted gain over the requested arm (1 - w) x gain over forced and the predicted reversals of the arm switch,
     manifests/predictions_hotpot_mechanism.json (write-once);
  5b. the split pilot's choices, the strict design's disjoint pilots with half of each pilot's items
     on each baseline (scripts/run_split_pilot.save_hotpot), manifests/predictions_hotpot_split.json; the decomposition
     rule and the choice by (1 - w) x forced gain (scripts/run_baseline_mixture.hotpot_decomposition),
     manifests/predictions_hotpot_decomposition.json (both write-once);
  6. manifests/predictions_hotpot_discovery.json, the sha256 of every file above and of the cross-benchmark file,
     which score_hotpot.py --stage all checks before it scores the validation draws.
Full run (reads data/processed/hotpot/scored.parquet; refuses unless the tranche file and the discovery index exist
and the indexed files are unchanged). The rules are refitted from the discovery rows of the full table and must equal
the saved ones (an identical file is kept; a different one is refused):
  1. map pilot predictions checked against the saved file;
  2. Steps 1-3 of the main analysis per contrast (scripts/run_dataset2.run, with its controls and sensitivities);
  3. the same Steps 1-3 on strict token F1, without the count-based known-effect controls (F1 is not a count);
  4. fresh-configuration sensitivity: the saved Step 3 rules (exact match strict and lenient, F1) evaluated on the
     validation draws whose configuration appears in no discovery draw, beside the main target (all 24 validation
     draws, sampled with replacement from the bank);
  5. the decision analysis on the validation draws from the saved choices (scripts/run_decision.evaluate: distinct
     validation configurations, all and those in no discovery draw);
  6. map points (R and C of benefit and accuracy), capability slopes on the discovery draws with the model-level
     summaries of model_resampling, per-model means;
  7. the tranche statements checked on the validation draws;
  8. the fair comparison from the saved choices only (scripts/hotpot_fair.py; its discovery files refitted from the
     full table's discovery rows must equal the saved ones), results/hotpot/fair_rules_evaluation.json;
  9. both directions of the wrong-baseline decision cost from the saved choices of the two
     contrasts (scripts/run_decision_directions.hotpot_directions), results/hotpot/decision_directions.json; the
     mechanism forecast against the validation draws, misses included, results/hotpot/mechanism_forecast_evaluation.json;
 10. the split pilot's saved choices (scripts/run_split_pilot.evaluate_hotpot), results/hotpot/
     split_pilot_evaluation.json, and the decomposition rule (scripts/run_baseline_mixture.evaluate_hotpot_
     decomposition), results/hotpot/decomposition_evaluation.json, every statement reported, misses included.
Outputs: results/hotpot/hotpot_report.json (the aggregate), <label>.json and <label>_f1.json (per contrast; hotpot.json
is the requested-arm contrast), fair_rules_evaluation.json, summary.md, summary_clean.md, tables/.
"""
from __future__ import annotations

import argparse
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

PROCESSED = os.path.join(ROOT, "data", "processed", "hotpot")
OUT = os.path.join(ROOT, "results", "hotpot")
MANIFESTS = os.path.join(ROOT, "manifests")
SAMPLER = os.path.join(MANIFESTS, "sampler_hotpot.json")
SELECTION = os.path.join(MANIFESTS, "calibration_selection_v2.json")
MAP_CONFIG = os.path.join(ROOT, "config", "reliability_map.yaml")
INST_CONFIG = os.path.join(ROOT, "config", "instability.yaml")
CONTRASTS = {"cot_req": ("direct_answer", "zero_shot_cot", "hotpot"),
             "cot_forced": ("forced_direct", "zero_shot_cot", "hotpot_forced"),
             "fewshot": ("forced_direct", "fewshot_forced", "hotpot_fewshot")}
TITLES = {"cot_req": "chain of thought - requested direct", "cot_forced": "chain of thought - forced direct",
          "fewshot": "five-shot forced - zero-shot forced"}
SEED = 20261117
N_TRANCHE = 4
DISCOVERY_INDEX = "predictions_hotpot_discovery.json"
REPORT = "hotpot_report.json"
FAIR_REPORT = "fair_rules_evaluation.json"
C_DEFINITION = ("switch: share of model pairs whose gain ordering reverses between the requested and forced baseline "
                "arms on the same draw (same configuration), averaged over draws; new draw: share that reverses between "
                "two distinct draw positions within an arm, averaged over all pairs of positions and over the two arms; "
                "a pair is reversed when both gain differences exceed estimators.TIE_TOL in opposite directions, and a "
                "pair tied on either side is not reversed (counted as tied); the same statistic on both splits")
C_SPLIT_HALF = ("original definition, kept as a secondary output: pairs in opposite order between the two arms' mean "
                "gains over all draws, against pairs in opposite order between the requested arm's mean gains over the "
                "first and second half of the draws (draws 1-2 against 3-4 on discovery, 12 against 12 on validation)")
D_NOTE = ("empirical forecast, not a theoretical threshold: R > 1/2 is the squared-error crossover of a one-draw centred "
          "own-gain estimate against a constant, and gives no threshold for top-third regret against a capability "
          "slope (DOVE pooled MMLU block: R 0.29, yet one-draw own gain has the lower whole-bank regret, 0.70 against "
          "1.03 pp)")


# ============================================================================ inputs


def item_ids(processed: str = PROCESSED) -> list[str]:
    from src import hotpot_d2
    rec = json.load(open(os.path.join(processed, "item_set.json"), encoding="utf-8"))
    t, _ = hotpot_d2.load_items("target")
    if rec["items"] == "A":
        t = t[t["block"] == "A"]
    return t["item_id"].astype(str).tolist()


def panel_info(panel=None):
    from src import hotpot_d2
    reg = hotpot_d2.load_registries()
    panel = list(panel or hotpot_d2.panel())
    fams = [reg[k].get("family") for k in panel]
    return panel, fams, hotpot_d2.extension_tiers(reg, panel)


def loader(path: str, contrast: str, score: str = "em"):
    """load(ids) -> rows of the two arms of a contrast, relabelled direct_answer (B) and zero_shot_cot (P); with
    score 'f1' the F1 columns stand in for the exact-match columns."""
    b, p, _ = CONTRASTS[contrast]

    def load(ids):
        df = pd.read_parquet(path, filters=[("config_id", "in", sorted(set(map(str, ids)))), ("arm", "in", [b, p])])
        df["item_id"] = df["item_id"].astype(str)
        df["arm"] = df["arm"].map({b: "direct_answer", p: "zero_shot_cot"})
        if score == "f1":
            df["strict_correct"] = df["strict_f1"].astype(float)
            df["lenient_correct"] = df["lenient_f1"].astype(float)
        return df
    return load


def arrays(load, units, panel, items, value="strict_correct"):
    from src import dataset2 as d2
    cells = d2.cell_table(load(units), panel, units, items)
    return d2.unit_arrays(cells, units, panel, value)


# ============================================================================ the four-draw statements


def reordered(x, y) -> dict:
    """Share of model pairs ordered oppositely by two vectors (ties reported)."""
    from src import reliability_map as rm
    a = rm.agreement_between(np.asarray(x, float)[None], np.asarray(y, float)[None])
    return {"opposite": a["opposite"], "tie": a["tie"], "same": a["same"]}


def arm_switch(Breq, Bfd, P, rows_a, rows_b) -> dict:
    """(c), the original split-half calculation, kept as a secondary output: pairs reordered by switching the baseline
    arm (mean gains over all rows) and by a new draw (mean gains over rows_a against rows_b, requested arm)."""
    g_req, g_fd = (P - Breq), (P - Bfd)
    sw = reordered(g_req.mean(0), g_fd.mean(0))
    dr = reordered(g_req[rows_a].mean(0), g_req[rows_b].mean(0))
    return {"definition": C_SPLIT_HALF, "switch": sw, "new_draw": dr,
            "switch_reorders_more": bool(sw["opposite"] > dr["opposite"])}


def _parts(a: dict, **extra) -> dict:
    return {"opposite": a["opposite"], "tie": a["tie"], "same": a["same"], **extra}


def within_arm_reversal(G, units=None) -> dict:
    """Share of model pairs whose gain ordering reverses between two distinct draw positions of one arm, averaged over
    all pairs of positions (G: draws by models). Two positions that landed on the same configuration count as they
    are (identical gains, nothing reversed), as the with-replacement sampler gives them."""
    from src import reliability_map as rm
    G = np.asarray(G, float)
    K = G.shape[0]
    if K < 2:
        return {"opposite": None, "tie": None, "same": None, "n_position_pairs": 0,
                "note": "fewer than two draws: no pair of distinct positions"}
    i, j = np.triu_indices(K, 1)
    a = rm.agreement_between(G[i], G[j])
    same_cfg = None if units is None else int(sum(str(units[x]) == str(units[y]) for x, y in zip(i, j)))
    return _parts(a, n_position_pairs=int(len(i)), position_pairs_on_one_configuration=same_cfg)


def arm_switch_same_draw(Breq, Bfd, P, units=None) -> dict:
    """(c): the share of model pairs whose gain ordering reverses between the requested and
    the forced baseline arm on the same draw (chain of thought fixed), averaged over draws, against the share that
    reverses between two distinct draw positions within an arm, averaged over position pairs and the two arms. One tie
    convention (reliability_map.agreement_between): reversed needs both differences beyond TIE_TOL in opposite
    directions; a tie on either side is not reversed. Edge cases are reported as they are; equal shares give False."""
    from src import reliability_map as rm
    Breq, Bfd, P = (np.asarray(x, float) for x in (Breq, Bfd, P))
    g_req, g_fd = P - Breq, P - Bfd
    sw = _parts(rm.agreement_between(g_req, g_fd), n_draws=int(g_req.shape[0]))
    w_req, w_fd = within_arm_reversal(g_req, units), within_arm_reversal(g_fd, units)
    if w_req["opposite"] is None:
        nd = {"opposite": None, "tie": None, "same": None, "requested": w_req, "forced": w_fd}
        return {"definition": C_DEFINITION, "switch": sw, "new_draw": nd, "difference": None,
                "switch_reorders_more": None, "equal": None}
    nd = {q: 0.5 * (w_req[q] + w_fd[q]) for q in ("opposite", "tie", "same")}
    nd.update({"requested": w_req, "forced": w_fd})
    diff = sw["opposite"] - nd["opposite"]
    return {"definition": C_DEFINITION, "switch": sw, "new_draw": nd, "difference": float(diff),
            "switch_reorders_more": bool(diff > 0), "equal": bool(diff == 0)}


def one_draw_regrets(B4, P4, Gv) -> dict:
    """(d): mean plug-in regret on the validation gains of the third chosen from each one-draw pilot by own gain and
    by the capability slope."""
    from src import decision as dc
    n = dc.third_size(B4.shape[1])
    G4 = P4 - B4
    own = np.array([dc.choose(G4[r], n)[0] for r in range(B4.shape[0])])
    cap = np.array([dc.choose(dc.capability_one_draw(B4[r], G4[r])[0], n)[0] for r in range(B4.shape[0])])
    return {"n": n, "own_1": float(dc.regret_plugin(own, Gv, n).mean()), "capability_1": float(dc.regret_plugin(cap, Gv, n).mean())}


def tranche_statements(T: dict, mcfg: dict, units=None) -> dict:
    """The statements saved before the rest of the collection is scored. T: {contrast: (B4, P4)} on four draws, plus
    'req4', 'fd4' (the two baselines) and 'cot4' for (c); units: the four draws' configurations."""
    from src import reliability_map as rm
    out = {}
    for c, (B4, P4) in T["pairs"].items():
        pp = rm.pilot_prediction(B4, P4, mcfg["group_sizes"], mcfg["agreement_targets"])
        R = pp["benefit"]["R"]
        out[c] = {"pilot": pp,
                  "a_predicted_C_one_draw": pp["benefit"]["C_one_draw"], "a_tolerance": 0.05,
                  "b_K_needed": pp["benefit"]["K_needed"],
                  "d_R_benefit": R,
                  "d_own_gain_beats_capability": (None if R is None or not np.isfinite(R) else bool(R > 0.5)),
                  "d_note": D_NOTE}
    Breq, Bfd, P = T["req4"], T["fd4"], T["cot4"]
    K = Breq.shape[0]
    h = K // 2
    out["c_arm_switch"] = arm_switch_same_draw(Breq, Bfd, P, units)
    out["c_arm_switch_split_half"] = arm_switch(Breq, Bfd, P, list(range(h)), list(range(h, K)))
    return out


def finite(x):
    """x as a float when it is a finite number, else None (the saved files hold non-finite values as "nan", "inf")."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def evaluate_statements(saved: dict, V: dict, D4: dict, mcfg: dict, n_splits: int, val_units=None) -> dict:
    """The saved statements against the validation draws. A saved prediction that is not a finite number (R of benefit
    outside [0, 1], or K infinite at R = 0) is reported as saved and marked not testable."""
    from src import reliability_map as rm, sampling
    res = {}
    for c in CONTRASTS:
        s = saved["statements"][c]
        Bv, Pv = V[c]
        obs = rm.validation_agreement(Bv, Pv, mcfg["group_sizes"], n_splits,
                                      sampling.rng_for("hotpot_tranche", c, seed=mcfg["seed"]))
        C1 = obs["one_draw"]["benefit"]["C"]
        pred = s["a_predicted_C_one_draw"]
        a_ok = None if finite(pred) is None else bool(abs(finite(pred) - C1) <= s["a_tolerance"])
        kres = {}
        kv = Bv.shape[0]
        for t, k in (s["b_K_needed"] or {}).items():
            if finite(k) is None:
                kres[t] = {"K_needed": k, "observed_C_at_K": None, "reached": None,
                           "note": "not testable: the saved K is not a finite number"}
                continue
            if 2 * int(k) > kv:
                kres[t] = {"K_needed": k, "observed_C_at_K": None, "reached": None,
                           "note": "not testable: two groups of K need more validation draws than exist"}
                continue
            rng = sampling.rng_for("hotpot_tranche", c, "K", int(k), seed=mcfg["seed"])
            perms = np.stack([rng.permutation(kv) for _ in range(n_splits)])
            Ck = rm.split_agreement(Pv - Bv, perms, int(k))["C"] if int(k) > 1 else C1
            kres[t] = {"K_needed": int(k), "observed_C_at_K": Ck, "reached": bool(Ck >= float(t))}
        reg = one_draw_regrets(*D4[c], Pv - Bv)
        d_obs = bool(reg["own_1"] < reg["capability_1"])
        res[c] = {"a": {"predicted": pred, "observed": C1, "within_0.05": a_ok}, "b": kres,
                  "d": {"R_benefit": s["d_R_benefit"], "forecast_own_better": s["d_own_gain_beats_capability"],
                        "note": s.get("d_note", D_NOTE), "regret": reg, "observed_own_better": d_obs,
                        "holds": None if s["d_own_gain_beats_capability"] is None
                        else bool(s["d_own_gain_beats_capability"] == d_obs)}}
    Breq, Bfd, P = V["cot_req"][0], V["cot_forced"][0], V["cot_req"][1]
    kv = Breq.shape[0]
    h = kv // 2
    sw = arm_switch_same_draw(Breq, Bfd, P, val_units)
    pred_c = saved["statements"]["c_arm_switch"]["switch_reorders_more"]
    res["c_arm_switch"] = {"definition": C_DEFINITION, "predicted": pred_c, "observed": sw,
                           "holds": None if pred_c is None or sw["switch_reorders_more"] is None
                           else bool(sw["switch_reorders_more"] == pred_c)}
    sh = arm_switch(Breq, Bfd, P, list(range(h)), list(range(h, kv)))
    pred_h = saved["statements"]["c_arm_switch_split_half"]["switch_reorders_more"]
    res["c_arm_switch_split_half"] = {"definition": C_SPLIT_HALF, "secondary": True, "predicted": pred_h,
                                      "observed": sh, "holds": bool(sh["switch_reorders_more"] == pred_h)}
    return res


def stage_predict(a) -> dict:
    from src import dataset2 as d2
    from src.sampling import sha256_json
    from run_index_checks import jsonable
    path = os.path.join(a.processed, "scored_discovery4.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent; run score_hotpot.py --stage discovery4")
    sampler = json.load(open(a.sampler, encoding="utf-8"))
    units = list(map(str, sampler["discovery_config_ids"][:N_TRANCHE]))
    panel, _, _ = panel_info(a.panel)
    items = item_ids(a.processed)
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    pairs = {c: arrays(loader(path, c), units, panel, items) for c in CONTRASTS}
    T = {"pairs": pairs, "req4": pairs["cot_req"][0], "fd4": pairs["cot_forced"][0], "cot4": pairs["cot_req"][1]}
    st = tranche_statements(T, mcfg, units)
    fr = {"label": "hotpot_tranche", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": panel,
          "sampler_sha256": sampler["sha256"], "units": units, "n_items": len(items), "score": "strict exact match",
          "note": "four statements per contrast, from the first "
                  "four discovery draws, saved before the rest of the collection is scored",
          "statements": jsonable(st)}
    fr["sha256"] = sha256_json(fr["statements"])
    out = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    saved = d2.write_frozen(fr, out, overwrite=a.overwrite_frozen)
    print(f"saved {out}")
    return saved


# ============================================================================ rules saved from the discovery draws


def freeze_f1(load, sampler, panel, items, label, pred_path, overwrite):
    """The Steps 1-3 rules on token F1 from the discovery draws; returns (frozen, Bd, Pd)."""
    from src import dataset2 as d2
    disc = list(sampler["discovery_config_ids"])
    Bd, Pd = arrays(load, disc, panel, items)
    f = d2.freeze_predictions(Bd, Pd, label, panel, sampler["sha256"])
    return d2.write_frozen(f, pred_path, overwrite=overwrite), Bd, Pd


def freeze_map(D, panel, sampler, mcfg, pred_dir, overwrite) -> dict:
    """Map pilot predictions from the first 2 and 4 discovery draws per contrast (D: {contrast: (Bd, Pd)})."""
    from src import dataset2 as d2, reliability_map as rm
    from src.sampling import sha256_json
    from run_index_checks import jsonable
    preds = {c: {str(n): rm.pilot_prediction(D[c][0][:n], D[c][1][:n], mcfg["group_sizes"], mcfg["agreement_targets"])
                 for n in mcfg["pilot_sizes"]} for c in CONTRASTS}
    fr = {"label": "hotpot_map", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": panel,
          "sampler": sampler["sha256"], "note": "pilot predictions of the HotpotQA map points (first 2 and 4 discovery "
          "draws), written before the validation draws are scored", "settings": jsonable(preds)}
    fr["sha256"] = sha256_json(fr["settings"])
    return d2.write_frozen(fr, os.path.join(pred_dir, "predictions_hotpot_map.json"), overwrite=overwrite)


def freeze_decision(c, units, panel, B, P, pred_dir, overwrite) -> dict:
    """The decision analysis's choices for one contrast (the decision analysis's rules, pilot seeds 'decision/hotpot/K<k>'),
    written to <pred_dir>/predictions_decision_<label>.json and .npz."""
    import run_decision as rd
    from src import rule_budgets as rbud
    label = CONTRASTS[c][2]
    rec, arr = rd.choice_record(label, [str(u) for u in units], list(panel), B, P, task="hotpot",
                                title=f"HotpotQA: {TITLES[c]}")
    jp = os.path.join(pred_dir, f"predictions_decision_{label}.json")
    return rbud.freeze(rec, arr, jp, jp[:-5] + ".npz", overwrite=overwrite)


def rule_files(pred_dir) -> list[str]:
    """Every file the discovery stage saves."""
    out = [os.path.join(pred_dir, "predictions_hotpot_map.json")]
    for c, (_, _, label) in CONTRASTS.items():
        out += [os.path.join(pred_dir, f"predictions_{label}{s}.json") for s in ("", "_lenient", "_f1")]
        out += [os.path.join(pred_dir, f"predictions_decision_{label}{s}") for s in (".json", ".npz")]
    import hotpot_fair
    import run_baseline_mixture
    import run_split_pilot
    # the mechanism forecast, saved here so the index covers it; the split
    # pilot's choices and the decomposition rule
    return out + hotpot_fair.all_files(pred_dir) + [os.path.join(pred_dir, run_baseline_mixture.HOTPOT_FORECAST),
                                                     os.path.join(pred_dir, run_split_pilot.HOTPOT_SPLIT),
                                                     os.path.join(pred_dir, run_baseline_mixture.HOTPOT_DECOMPOSITION)]


def _index_key(p: str) -> str:
    """Path relative to the repository root when inside it, else absolute (dry runs in a temporary folder)."""
    p = os.path.abspath(p)
    r = os.path.relpath(p, ROOT) if os.path.splitdrive(p)[0].lower() == os.path.splitdrive(ROOT)[0].lower() else p
    return p if r.startswith("..") else r.replace("\\", "/")


def write_index(pred_dir, sampler, scored_path, overwrite) -> dict:
    """manifests/predictions_hotpot_discovery.json: the sha256 of every rule file of the discovery stage. An existing
    index listing the same files and hashes is kept; a different one is refused unless overwrite is set."""
    from src import hotpot_task
    files = {_index_key(p): hotpot_task.file_sha256(p) for p in rule_files(pred_dir)}
    meta_path = scored_path[:-len(".parquet")] + ".json"
    scored_sha = json.load(open(meta_path, encoding="utf-8")).get("sha256") if os.path.exists(meta_path) else None
    rec = {"label": "hotpot_discovery", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sampler_sha256": sampler["sha256"],
           "scored_discovery": {"path": _index_key(scored_path), "sha256": scored_sha},
           "note": "every rule fitted on the discovery draws alone, saved before any validation draw was scored; "
                   "score_hotpot.py --stage all checks these hashes",
           "files": files}
    path = os.path.join(pred_dir, DISCOVERY_INDEX)
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if old.get("files") == files:
            return old
        if not overwrite:
            raise FileExistsError(f"{path} lists other rule files or hashes; refusing to overwrite")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def stage_cross(a) -> dict:
    """Item 3a of the fair comparison (scripts/hotpot_fair.py): choices for HotpotQA's panel from the GSM8K, BIG-Bench
    Hard and MATH-500 discovery draws alone; reads no HotpotQA data."""
    import hotpot_fair as hf
    from src import hotpot_task
    panel, _, _ = panel_info(a.panel)
    rec = hf.save_cross(panel, hf.panel_eci(panel), hf.source_arrays(CONTRASTS), a.pred_dir, a.overwrite_frozen)
    for p in hf.files(hf.CROSS, a.pred_dir):
        print(f"saved {p} sha256 {hotpot_task.file_sha256(p)}")
    return rec


def contrast_arms() -> dict:
    return {c: (b, p) for c, (b, p, _) in CONTRASTS.items()}


def cross_order_problem(pred_dir, processed) -> str | None:
    """Why the cross-benchmark file does not precede the HotpotQA scoring (None when it does): it must be dated no
    later than the four-draw table, the first HotpotQA scores read."""
    import hotpot_fair as hf
    meta = os.path.join(processed, "scored_discovery4.json")
    if not os.path.exists(meta):
        return None
    scored = json.load(open(meta, encoding="utf-8")).get("created")
    saved = json.load(open(hf.files(hf.CROSS, pred_dir)[0], encoding="utf-8")).get("created")
    if scored and saved and saved > scored:
        return (f"{hf.CROSS}.json is dated {saved}, after the four-draw table ({scored}); it must be saved before any "
                "HotpotQA output is scored")
    return None


def stage_discovery(a) -> dict:
    """Every rule except the four-draw statements, fitted on the discovery draws alone and saved before the validation
    draws are scored."""
    import hotpot_fair as hf
    import run_dataset2
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    if not os.path.exists(tranche):
        raise SystemExit(f"{tranche} is absent; run --stage predict first")
    path = os.path.join(a.processed, "scored_discovery.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is absent; run score_hotpot.py --stage discovery")
    panel, fams, _ = panel_info(a.panel)
    hf.load_cross(a.pred_dir, panel)
    problem = cross_order_problem(a.pred_dir, a.processed)
    if problem:
        raise SystemExit(problem)
    sampler = json.load(open(a.sampler, encoding="utf-8"))
    items = item_ids(a.processed)
    disc = list(map(str, sampler["discovery_config_ids"]))
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    D = {c: arrays(loader(path, c), disc, panel, items) for c in CONTRASTS}
    freeze_map(D, panel, sampler, mcfg, a.pred_dir, a.overwrite_frozen)
    for c, (_, _, label) in CONTRASTS.items():
        run_dataset2.freeze_rules(loader(path, c), sampler, panel, items,
                                  os.path.join(a.pred_dir, f"predictions_{label}.json"), label, a.overwrite_frozen)
        freeze_f1(loader(path, c, "f1"), sampler, panel, items, f"{label}_f1",
                  os.path.join(a.pred_dir, f"predictions_{label}_f1.json"), a.overwrite_frozen)
        freeze_decision(c, disc, panel, D[c][0], D[c][1], a.pred_dir, a.overwrite_frozen)
    hf.save_discovery(path, disc, panel, fams, hf.panel_eci(panel), items, contrast_arms(), a.pred_dir,
                      a.overwrite_frozen, D=D)
    import run_baseline_mixture   # the mechanism forecast from the discovery table alone (write-once)
    run_baseline_mixture.hotpot_forecast(a.processed, a.sampler, a.pred_dir, a.panel)
    import run_split_pilot   # the split pilot's choices from the strict design's saved pilots (write-once)
    run_split_pilot.save_hotpot(path, disc, panel, items, a.pred_dir)
    # the decomposition rule and the choice by (1 - w) x forced gain, from the discovery table alone (write-once)
    run_baseline_mixture.hotpot_decomposition(a.processed, a.sampler, a.pred_dir, a.panel)
    idx = write_index(a.pred_dir, sampler, path, a.overwrite_frozen)
    print(f"saved {len(idx['files'])} rule files and {os.path.join(a.pred_dir, DISCOVERY_INDEX)}")
    return idx


# ============================================================================ full run


def fresh_rows(disc, val) -> list[int]:
    """Positions of the validation draws whose configuration appears in no discovery draw (repeats within the
    validation draws kept, as the sampler drew them)."""
    d = set(map(str, disc))
    return [i for i, u in enumerate(map(str, val)) if u not in d]


def fresh_sensitivity(frozen: dict, Zv, rows, sel) -> dict:
    """The saved Step 3 rules evaluated on the fresh-configuration validation draws only."""
    from src import dataset2 as d2
    Zv = np.asarray(Zv, float)
    out = {"label": "fresh-configuration sensitivity: validation draws whose configuration is in no discovery draw",
           "n_draws": len(rows), "rows": list(map(int, rows))}
    if len(rows) < 2:
        out["note"] = "fewer than two fresh-configuration draws: not evaluated"
        return out
    out["step3"] = d2.evaluate_frozen(frozen, Zv[rows], sel)
    return out


def steps_f1(load, sampler, panel, items, label, pred_path, overwrite, fresh=None) -> dict:
    """Steps 1-3 on token F1 without the count-based controls; rules frozen before the validation draws load."""
    from src import dataset2 as d2, intervals as iv, sampling
    sel = d2.load_selection(SELECTION)
    disc, val = list(sampler["discovery_config_ids"]), list(sampler["validation_config_ids"])
    frozen, Bd, Pd = freeze_f1(load, sampler, panel, items, label, pred_path, overwrite)
    m = iv.multiplicities(sampling.rng_for(label, "bootstrap"), len(disc), 9999)
    out = {"label": label, "score": "strict token F1 (100 x F1 per item; cell = mean over items)",
           "frozen_predictions": {"path": os.path.relpath(pred_path, ROOT).replace("\\", "/"), "sha256": frozen["sha256"]},
           "step1": d2.step1(Bd, Pd, sel, m, panel), "step2": d2.step2(Bd, Pd, sel, m),
           "controls": "not run: the known-effect controls simulate counts, and F1 is not a count"}
    Bv, Pv = arrays(load, val, panel, items)
    out["step3"] = d2.evaluate_frozen(frozen, Pv - Bv, sel)
    out["step3_fresh_configurations"] = fresh_sensitivity(frozen, Pv - Bv, fresh_rows(disc, val) if fresh is None
                                                          else fresh, sel)
    return out


def run_full(a) -> dict:
    t0 = time.time()
    import run_dataset2
    import run_decision as rd
    import run_forced_direct as rfd
    import score_hotpot
    from run_model_resampling import block
    from run_index_checks import jsonable
    from src import dataset2 as d2
    tranche = os.path.join(a.pred_dir, "predictions_hotpot_tranche.json")
    if not os.path.exists(tranche):
        raise SystemExit(f"{tranche} is absent; run --stage predict first")
    problems = score_hotpot.discovery_index_problems(os.path.join(a.pred_dir, DISCOVERY_INDEX))
    if problems:
        raise SystemExit("run --stage discovery before the full run: " + "; ".join(problems))
    path = os.path.join(a.processed, "scored.parquet")
    sampler = json.load(open(a.sampler, encoding="utf-8"))
    panel, fams, tiers = panel_info(a.panel)
    items = item_ids(a.processed)
    disc, val = list(map(str, sampler["discovery_config_ids"])), list(map(str, sampler["validation_config_ids"]))
    fresh = fresh_rows(disc, val)
    sel = d2.load_selection(SELECTION)
    mcfg = yaml.safe_load(open(MAP_CONFIG, encoding="utf-8"))
    icfg = yaml.safe_load(open(INST_CONFIG, encoding="utf-8"))
    method, alpha = sel["H"], float(icfg["alpha"])
    n_splits = 200 if a.quick else int(mcfg["validation_splits"])
    n_boot, n_ctrl = (999, 100) if a.quick else (a.n_boot, a.n_control)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)

    # 1. map pilot predictions: refitted on the discovery rows, must equal the saved file
    D = {c: arrays(loader(path, c), disc, panel, items) for c in CONTRASTS}
    frozen_map = freeze_map(D, panel, sampler, mcfg, a.pred_dir, overwrite=False)

    R = {"analysis": "HotpotQA", "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": "percentage points", "models": panel, "M": len(panel), "n_items": len(items),
         "slope_method": method, "seed": SEED, "contrasts": {}, "map_points": [],
         "fresh_configurations": {"validation_positions": fresh, "n_draws": len(fresh),
                                  "n_distinct": len(set(val[i] for i in fresh)),
                                  "note": "sensitivity beside the main target (all validation draws, sampled with "
                                          "replacement from the bank)"}}
    for c, (b, p, label) in CONTRASTS.items():
        prov = {"scored": os.path.relpath(path, ROOT).replace("\\", "/"), "baseline_arm": b, "intervention_arm": p,
                "relabelled": "baseline as direct_answer, intervention as zero_shot_cot"}
        pred_path = os.path.join(a.pred_dir, f"predictions_{label}.json")
        steps = run_dataset2.run(loader(path, c), sampler, panel, fams, tiers, items, a.out, pred_path, label,
                                 n_boot, n_ctrl, overwrite_frozen=False, provenance=prov,
                                 tables_dir=os.path.join(a.out, f"tables_{label}"))
        f1 = steps_f1(loader(path, c, "f1"), sampler, panel, items, f"{label}_f1",
                      os.path.join(a.pred_dir, f"predictions_{label}_f1.json"), False, fresh)
        B, P = D[c]
        Bv, Pv = arrays(loader(path, c), val, panel, items)
        Bvl, Pvl = arrays(loader(path, c), val, panel, items, "lenient_correct")
        fr_strict = json.load(open(pred_path, encoding="utf-8"))
        fr_len = json.load(open(pred_path.replace(".json", "_lenient.json"), encoding="utf-8"))
        fresh_res = {"strict_em": fresh_sensitivity(fr_strict, Pv - Bv, fresh, sel),
                     "lenient_em": fresh_sensitivity(fr_len, Pvl - Bvl, fresh, sel),
                     "f1": f1["step3_fresh_configurations"]}
        # decision: refit choices from the full table's discovery rows (must equal the saved pair), evaluate the saved
        freeze_decision(c, disc, panel, B, P, a.pred_dir, overwrite=False)
        rec, W = rd.load_choices(label, a.pred_dir)
        dec = rd.evaluate(label, rec, W, (val, panel, Bv, Pv))
        pts, point, obs = rfd.map_points(c, B, P, Bv, Pv, frozen_map, mcfg, n_splits, len(panel), len(items))
        for pt in pts:
            pt["setting"] = f"hotpot:{c}"
        slope = block(B, P, fams, method, alpha, np.random.default_rng(SEED), panel)
        rows = [{"model_key": k, "baseline": float(B[:, j].mean()), "intervention": float(P[:, j].mean()),
                 "gain": float((P - B)[:, j].mean())} for j, k in enumerate(panel)]
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "tables", f"per_model_{c}.csv"), index=False)
        R["contrasts"][c] = {"title": TITLES[c], "label": label, "step2": steps.get("step2"),
                             "step3_registered": (steps.get("step3") or {}).get("registered"),
                             "step3_fresh_configurations": {q: (v.get("step3") or {}).get("registered")
                                                            for q, v in fresh_res.items()},
                             "f1": {"step2": f1["step2"], "step3_registered": f1["step3"].get("registered"),
                                    "frozen": f1["frozen_predictions"]},
                             "decision": {"saved": f"manifests/predictions_decision_{label}.json",
                                          "sha256": rec.get("sha256"), **dec},
                             "map": {"point": point, "validation_one_draw": obs["one_draw"]},
                             "capability_slope": slope, "per_model": rows,
                             "means": {q: float(np.mean([r[q] for r in rows])) for q in ("baseline", "intervention", "gain")}}
        with open(os.path.join(a.out, f"{label}_f1.json"), "w", encoding="utf-8") as fh:
            json.dump(jsonable(f1), fh, indent=1)
        with open(os.path.join(a.out, f"{label}_fresh_configurations.json"), "w", encoding="utf-8") as fh:
            json.dump(jsonable(fresh_res), fh, indent=1)
        R["map_points"] += pts
        print(f"{c}: done ({time.time() - t0:.0f} s)", flush=True)

    # 7. the tranche statements against the validation draws
    saved = json.load(open(tranche, encoding="utf-8"))
    units4 = saved["units"]
    D4 = {c: arrays(loader(path, c), units4, panel, items) for c in CONTRASTS}
    V = {c: arrays(loader(path, c), val, panel, items) for c in CONTRASTS}
    R["tranche"] = {"saved": os.path.relpath(tranche, ROOT).replace("\\", "/"), "sha256": saved["sha256"],
                    "checks": evaluate_statements(saved, V, D4, mcfg, n_splits, val)}

    # 8. the fair comparison: discovery files refitted from the full table's discovery rows (identical kept, different
    # refused), then evaluated from the saved choices alone
    import hotpot_fair as hf
    hf.save_discovery(path, disc, panel, fams, hf.panel_eci(panel), items, contrast_arms(), a.pred_dir, False, D=D)
    fair_saved = hf.load_saved(a.pred_dir, panel)
    vconfigs, Sv = hf.dense(path, panel, val, items)
    F = hf.evaluate_all(fair_saved, disc, val, V, Sv, vconfigs, len(items), contrast_arms())
    with open(os.path.join(a.out, FAIR_REPORT), "w", encoding="utf-8") as fh:
        json.dump(jsonable(F), fh, indent=1)
    R["fair_comparison"] = {"evaluation": f"results/hotpot/{FAIR_REPORT}", "saved": F["saved"]}
    # 9. both directions of the wrong-baseline decision cost from the saved choices of the two
    # contrasts, and the mechanism forecast against the validation draws, misses included
    import run_baseline_mixture as rbm
    import run_decision_directions as rdd
    R["decision_directions"] = rdd.hotpot_directions(a.pred_dir, panel, fams, disc, val, V, Sv, vconfigs, len(items))
    R["mechanism_forecast"] = rbm.evaluate_hotpot_forecast(a.pred_dir, panel, V)
    # 10. the split pilot's saved choices and the decomposition rule against the validation draws,
    # misses included
    import run_split_pilot as rsp
    R["split_pilot"] = rsp.evaluate_hotpot(a.pred_dir, panel, fams, disc, val, Sv, vconfigs, len(items))
    R["decomposition"] = rbm.evaluate_hotpot_decomposition(a.pred_dir, a.processed, panel, disc, val)
    for name, key in (("decision_directions.json", "decision_directions"),
                      ("mechanism_forecast_evaluation.json", "mechanism_forecast"),
                      ("split_pilot_evaluation.json", "split_pilot"),
                      ("decomposition_evaluation.json", "decomposition")):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            json.dump(jsonable(R[key]), fh, indent=1)
    R["frozen_map"] = {"path": "manifests/predictions_hotpot_map.json", "sha256": frozen_map["sha256"]}
    R["discovery_index"] = f"manifests/{DISCOVERY_INDEX}"
    R["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, REPORT), "w", encoding="utf-8") as fh:
        json.dump(jsonable(R), fh, indent=1)
    txt = (summary(R) + "\n".join(hf.summary_lines(F, TITLES)) + "\n"
           + "\n".join(rdd.hotpot_lines(R["decision_directions"]) + rbm.hotpot_forecast_lines(R["mechanism_forecast"])
                       + rsp.hotpot_lines(R["split_pilot"]) + rbm.hotpot_decomposition_lines(R["decomposition"]))
           + "\n")
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return R


def est(b):
    if not isinstance(b, dict) or b.get("estimate") is None:
        return "--"
    c = b.get("ci95")
    return f"{b['estimate']:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")


def summary(R) -> str:
    L = ["# HotpotQA", "",
         f"{R['M']} models, {R['n_items']} target items, strict exact match unless marked F1. Points in percent.", "",
         "| Contrast | B mean | P mean | Gain | V_delta | V_delta (F1) | R of benefit | C (validation, one draw) | "
         "Slope of gain on B | Model bootstrap |", "|---|---:|---:|---:|---|---|---:|---:|---:|---|"]
    for c, T in R["contrasts"].items():
        pb, ob, cs = T["map"]["point"]["benefit"], T["map"]["validation_one_draw"]["benefit"], T["capability_slope"]
        mb = cs["model_bootstrap"]["ci95"]
        L.append(f"| {T['title']} | {T['means']['baseline']:.1f} | {T['means']['intervention']:.1f} | "
                 f"{T['means']['gain']:+.1f} | {est((T.get('step2') or {}).get('V_delta'))} | "
                 f"{est(T['f1']['step2'].get('V_delta'))} | {'--' if pb.get('R') is None else round(pb['R'], 3)} | "
                 f"{'--' if ob.get('C') is None else round(ob['C'], 3)} | {cs['slope']:+.3f} | "
                 f"{'--' if not mb else f'[{mb[0]:+.2f}, {mb[1]:+.2f}]'} |")
    fc = R["fresh_configurations"]
    L += ["", f"Fresh-configuration sensitivity: {fc['n_draws']} of the validation draws ({fc['n_distinct']} distinct "
          "configurations) use a configuration that appears in no discovery draw. The saved Step 3 rules are evaluated "
          f"on them alone in `<label>_fresh_configurations.json`, beside the main target on every validation draw.", ""]
    def f3(v):
        return ("--" if v is None else str(v)) if finite(v) is None else f"{finite(v):.3f}"
    def f2(v):
        return ("--" if v is None else str(v)) if finite(v) is None else f"{finite(v):.2f}"
    L += ["Decision analysis from the choices saved before validation was scored (plug-in regret in pp, distinct "
          "validation configurations in no discovery draw):", ""]
    for c, T in R["contrasts"].items():
        ev = (T["decision"].get("evaluation") or {}).get("independent") or {}
        rr = ev.get("rules") or {}
        parts = [f"{r} {f2(rr[r]['regret_plugin'])}" for r in ("capability_1", "own_1", "own_4", "own_all") if r in rr]
        L.append(f"- {T['title']}: {ev.get('n_configurations', 0)} configurations; " + (", ".join(parts) or "--") + ".")
    ck = R["tranche"]["checks"]
    L += ["", "Four-draw statements (saved before the rest was scored; (d) is an empirical forecast):", ""]
    for c in CONTRASTS:
        x = ck[c]
        L.append(f"- {TITLES[c]}: (a) predicted C {f3(x['a']['predicted'])}, observed {f3(x['a']['observed'])}, within "
                 f"0.05: {x['a']['within_0.05']}; (b) {x['b']}; (d) R {f3(x['d']['R_benefit'])}, own gain better "
                 f"forecast {x['d']['forecast_own_better']}, observed {x['d']['observed_own_better']}.")
    cs, ch = ck["c_arm_switch"], ck["c_arm_switch_split_half"]
    o = cs["observed"]
    L.append(f"- (c) switching the baseline arm reverses more pairs on the same draw than a new draw does within an arm: "
             f"predicted {cs['predicted']}, observed {o['switch_reorders_more']} (switch {f3(o['switch']['opposite'])}, "
             f"new draw {f3(o['new_draw']['opposite'])}), holds {cs['holds']}.")
    L.append(f"- (c), original split-half calculation (secondary): predicted {ch['predicted']}, observed "
             f"{ch['observed']['switch_reorders_more']}, holds {ch['holds']}.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("cross", "predict", "discovery", "full"), default="full")
    ap.add_argument("--processed", default=PROCESSED)
    ap.add_argument("--sampler", default=SAMPLER)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred-dir", default=MANIFESTS)
    ap.add_argument("--panel", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--n-control", type=int, default=2000)
    ap.add_argument("--overwrite-frozen", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "cross":
        stage_cross(a)
    elif a.stage == "predict":
        stage_predict(a)
    elif a.stage == "discovery":
        stage_discovery(a)
    else:
        run_full(a)


if __name__ == "__main__":
    main()
