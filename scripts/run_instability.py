r"""Analysis B: "who benefits more" across scales, panels, prompts and
scoring variants, on GSM8K (19 models) and the DOVE pooled MMLU block (5 models).

Run from the repository root (long):
    python scripts/run_instability.py
    # without the tokenizer step (variant A then reads its cached token table, or is skipped)
    python scripts/run_instability.py --no-download

Inputs: config/instability.yaml, data/processed/gsm8k/scored.parquet, the collection's
output shards (data/d2_out/collection/<model_key>/shard-*.parquet, variant A only), manifests/sampler_gsm8k.json,
manifests/d2_panel.json (ECI terciles), config/models.yaml (tokenizer revisions), the DOVE block of
scripts/run_dataset1.py and manifests/sampler_dove_mmlu.json, manifests/calibration_selection_v2.json.

Outputs: results/instability/instability.json (with a "macros" object for paper/make_numbers.py), summary.md,
summary_clean.md, tables/*.csv and manifests/predictions_instability_gsm8k*.json. Nothing in results/dataset2/ is written, and in manifests/ only the files
named predictions_instability_*.

Order for the scoring variants: each variant's prediction rules are fitted on its discovery draws and
written to manifests/predictions_instability_<label>.json before its validation draws are built. For variant A the 64-token rule is
applied to the discovery configurations first and to the validation-only configurations after the freeze. The strict
validation outcomes were already read by the main analysis (results/dataset2/), so these are sensitivity analyses.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import controls, dataset2 as d2, instability as ins, intervals as iv, sampling, scoring  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "instability.yaml")
SELECTION = os.path.join(ROOT, "manifests", "calibration_selection_v2.json")
SCORED = os.path.join(ROOT, "data", "processed", "gsm8k", "scored.parquet")
SHARDS = os.path.join(ROOT, "data", "d2_out", "collection")
TOK_DIR = os.path.join(ROOT, "data", "tokenizers")
TOK_CACHE = os.path.join(ROOT, "data", "processed", "instability")
OUT = os.path.join(ROOT, "results", "instability")
ARMS = d2.ARMS
SETTING_LABEL = {"gsm8k": "GSM8K", "dove": "DOVE MMLU block"}
STRATUM_LABEL = {"random": "random", "tercile:low": "low tercile", "tercile:middle": "middle tercile",
                 "tercile:high": "high tercile"}
VARIANTS = ("strict", "variant_a", "variant_b", "lenient")
VARIANT_LABEL = {"strict": "Main scoring (strict)", "variant_a": "Variant A: answer within 64 tokens",
                 "variant_b": "Variant B: short direct answers only", "lenient": "Lenient extraction"}


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else o.item() if isinstance(o, np.generic) else str(o))


# ============================================================================ data


def load_gsm8k(values=("strict_correct", "lenient_correct", "compliant")):
    from src import d2_io
    sampler = json.load(open(os.path.join(ROOT, "manifests", "sampler_gsm8k.json")))
    items_df, items_info = d2_io.load_items(split="test")
    main = json.load(open(os.path.join(ROOT, "results", "dataset2", "gsm8k.json")))
    panel = list(main["models"])
    reg = yaml.safe_load(open(os.path.join(ROOT, "config", "models.yaml")))["dataset2"]["models"]
    eci = {e["repo"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "d2_panel.json")))["panel"]}
    tiers = {}
    for mk in panel:
        e = eci[reg[mk]["hf_id"]]
        if e["revision"] != reg[mk]["revision"] or e["tier"] != reg[mk]["tier"]:
            raise SystemExit(f"{mk}: manifests/d2_panel.json and config/models.yaml disagree")
        tiers[mk] = e["tier"]
    cols = ["model_key", "arm", "config_id", "item_id", *values, "n_output_tokens"]
    scored = pd.read_parquet(SCORED, columns=cols)
    return {"sampler": sampler, "items": items_df, "items_info": items_info, "panel": panel, "reg": reg,
            "tiers": tiers, "eci": {mk: eci[reg[mk]["hf_id"]]["eci"] for mk in panel}, "scored": scored,
            "families": [reg[k]["family"] for k in panel]}


def load_dove():
    import run_dataset1 as r1
    bank, subjects, common, _, nc, n_items = r1.load_block()
    samp = json.load(open(os.path.join(ROOT, "manifests", "sampler_dove_mmlu.json")))
    if list(samp["bank_config_ids"]) != list(common):
        raise SystemExit("the DOVE sampler's bank differs from the MMLU block's common configurations")
    B, P = controls.pooled(nc["0_shot"], n_items), controls.pooled(nc["5_shot"], n_items)
    return {"B": B, "P": P, "disc": np.array(samp["discovery_bank_index"]), "val": np.array(samp["validation_bank_index"]),
            "panel": list(r1.PANEL), "short": [r1.SHORT[m] for m in r1.PANEL], "n_items": int(n_items.sum()),
            "n_subjects": len(subjects)}


# ============================================================================ scale, prompt, panel


def scale_block(Bd, Pd, Bv, Pv, n_items, method, alpha, bank=None):
    out = {}
    for sc in ins.SCALES:
        sd, sv = ins.scale_scores(Bd, Pd, sc, n_items), ins.scale_scores(Bv, Pv, sc, n_items)
        r = {"unit": ins.SLOPE_UNITS[sc], "discovery": ins.scale_moments(sd["x"], sd["y"], method, alpha),
             "validation": ins.scale_moments(sv["x"], sv["y"], method, alpha),
             "replaced_cells": {"discovery": sd["n_replaced"], "validation": sv["n_replaced"],
                                "of": sd["n_cells"] + sv["n_cells"]}}
        if sc == "points":
            for k in ("discovery", "validation"):
                x = r[k]
                x["H"] = -10 * x["slope"]
                s = x["slope_set"]
                x["H_set"] = dict(zip(("kind", "a", "b"), iv.scale_fieller(s["kind"], s["a"], s["b"], -10.0)))
        if bank is not None:
            sb = ins.scale_scores(bank[0], bank[1], sc, n_items)
            r["exact_bank"] = ins.bank_exact(sb["x"], sb["y"])
            r["replaced_cells"]["bank"] = sb["n_replaced"]
            if sc == "points":
                r["exact_bank"]["H"] = -10 * r["exact_bank"]["slope"]
        out[sc] = r
    return out


def prompt_block(sets, n_items):
    """sets: {name: (B, P, reference_slopes_by_scale)}."""
    out = {}
    for name, (B, P, ref) in sets.items():
        out[name] = {}
        for sc in ins.SCALES:
            s = ins.scale_scores(B, P, sc, n_items)
            sl = ins.single_draw_slopes(s["x"], s["y"])
            out[name][sc] = ins.opposite_sign_share(sl, ref[sc])
    return out


def panel_block(setting, arrays, n_items, strata, sizes, n_resamples, seed_label, method, alpha, bank=None):
    """arrays: (Bd, Pd, Bv, Pv). strata: {name: member indices}. Returns summaries and per-subpanel rows."""
    Bd, Pd, Bv, Pv = arrays
    sc_d = {sc: ins.scale_scores(Bd, Pd, sc, n_items) for sc in ins.SCALES}
    sc_v = {sc: ins.scale_scores(Bv, Pv, sc, n_items) for sc in ins.SCALES}
    sc_b = {sc: ins.scale_scores(bank[0], bank[1], sc, n_items) for sc in ins.SCALES} if bank is not None else None
    summ, rows = {}, []
    for stratum, members in strata.items():
        summ[stratum] = {"members": [int(m) for m in members], "sizes": {}}
        pooled = {sc: [] for sc in ins.SCALES}
        for k in sizes:
            rng = sampling.rng_for(seed_label, setting, stratum, k)
            subsets, mode = ins.subpanel_draw(members, k, n_resamples, rng)
            if not len(subsets):
                continue
            summ[stratum]["sizes"][str(k)] = {}
            for sc in ins.SCALES:
                st = ins.subpanel_statistics(sc_d[sc]["x"], sc_d[sc]["y"], sc_v[sc]["x"], sc_v[sc]["y"], subsets,
                                             method, alpha)
                s = ins.summarize_subpanels(st, mode)
                if mode == "sampled":
                    s["all_subsets_check"] = ins.all_subset_sign_shares(sc_d[sc]["x"], sc_d[sc]["y"], members, k)
                if sc_b is not None:
                    eb = np.array([ins.bank_exact(sc_b[sc]["x"][:, list(c)], sc_b[sc]["y"][:, list(c)])["slope"]
                                   for c in subsets])
                    st["slope_exact_bank"] = eb
                    s["exact_bank_signs"] = ins._signs(eb)
                summ[stratum]["sizes"][str(k)][sc] = s
                pooled[sc].append(st)
                for i, c in enumerate(subsets):
                    rows.append({"setting": setting, "stratum": stratum, "size": k, "scale": sc, "mode": mode,
                                 "subpanel": " ".join(map(str, c)), "slope_discovery": st["slope_discovery"][i],
                                 "slope_validation": st["slope_validation"][i], "set_sign": st["set_sign"][i],
                                 "C_own_gain_disc_val": st["C_own_gain_disc_val"][i],
                                 "C_benefit_one_draw": st["C_benefit_one_draw"][i],
                                 "slope_exact_bank": st["slope_exact_bank"][i] if "slope_exact_bank" in st else None})
        summ[stratum]["pooled_sizes"] = {}
        for sc in ins.SCALES:
            if not pooled[sc]:
                continue
            cat = {key: np.concatenate([p[key] for p in pooled[sc]])
                   for key in ("slope_discovery", "slope_validation", "set_sign", "C_own_gain_disc_val",
                               "C_benefit_one_draw")}
            cat["subsets"] = np.arange(len(cat["slope_discovery"]))
            modes = sorted({v[sc]["mode"] for v in summ[stratum]["sizes"].values()})
            summ[stratum]["pooled_sizes"][sc] = ins.summarize_subpanels(cat, "+".join(modes))
            summ[stratum]["pooled_sizes"][sc]["note"] = "sizes pooled with equal numbers per size where sampled"
    return summ, rows


# ============================================================================ variant A: token positions


def download_tokenizers(reg, panel, files):
    """Tokenizer files only, at the pinned revisions; the stored Hugging Face login is used and never printed."""
    from huggingface_hub import hf_hub_download
    out = {}
    for mk in panel:
        e = reg[mk]
        d = os.path.join(TOK_DIR, mk)
        os.makedirs(d, exist_ok=True)
        rec = {"hf_id": e["hf_id"], "revision": e["revision"], "files": {}}
        for fn in files:
            p = hf_hub_download(e["hf_id"], fn, revision=e["revision"], local_dir=d)
            rec["files"][fn] = {"path": rel(p), "sha256": sha256_file(p)}
        out[mk] = rec
    return out


def local_tokenizers(reg, panel, files):
    out = {}
    for mk in panel:
        rec = {"hf_id": reg[mk]["hf_id"], "revision": reg[mk]["revision"], "files": {}}
        for fn in files:
            p = os.path.join(TOK_DIR, mk, fn)
            if not os.path.exists(p):
                return None
            rec["files"][fn] = {"path": rel(p), "sha256": sha256_file(p)}
        out[mk] = rec
    return out


def direct_outputs(mk, configs):
    files = sorted(glob.glob(os.path.join(SHARDS, mk, "shard-*.parquet")))
    if not files:
        raise SystemExit(f"no output shards for {mk} under {rel(SHARDS)}")
    cols = ["model_key", "arm", "config_id", "item_id", "output_text", "n_output_tokens", "finish_reason"]
    parts = [pd.read_parquet(f, columns=cols, filters=[("arm", "==", "direct_answer"),
                                                       ("config_id", "in", sorted(configs))]) for f in files]
    df = pd.concat(parts, ignore_index=True)
    dup = df.duplicated(["config_id", "item_id"], keep=False)
    if dup.any():
        g = df[dup].groupby(["config_id", "item_id"])["output_text"].nunique()
        if (g > 1).any():
            raise SystemExit(f"{mk}: {int((g > 1).sum())} direct-arm requests have differing duplicate outputs")
        df = df.drop_duplicates(["config_id", "item_id"])
    return df


def token_rows(mk, tok_path, configs, scored_direct, spec, gold):
    """One row per direct-arm output of the listed configurations: the scorer's correctness, the re-tokenized
    length, the located boxed answer's end and its token position."""
    from tokenizers import Tokenizer
    out = direct_outputs(mk, configs)
    sd = scored_direct[scored_direct["config_id"].isin(configs)]
    m = out.merge(sd, on=["config_id", "item_id"], how="outer", suffixes=("", "_scored"), indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"{mk}: outputs and scored rows do not match one to one ({m['_merge'].value_counts().to_dict()})")
    if (m["n_output_tokens"] != m["n_output_tokens_scored"]).any():
        raise SystemExit(f"{mk}: n_output_tokens differs between the shards and the scored table")
    tok = Tokenizer.from_file(tok_path)
    tok.no_truncation()
    tok.no_padding()
    texts = [t if isinstance(t, str) else "" for t in m["output_text"].tolist()]
    enc = tok.encode_batch(texts, add_special_tokens=False)
    m["retok_total"] = [len(e.ids) for e in enc]
    ends, nbox, pos, rescored = [], [], [], []
    for t, e, c, iid in zip(texts, enc, m["strict_correct"].tolist(), m["item_id"].tolist()):
        end, n, p, ok = None, 0, -1, None
        if c == 1:
            ans = scoring.strict_extract(t, spec)
            ok = ans == gold[iid]
            end, n = ins.boxed_answer_end(t, ans)
            p = ins.token_position(e.offsets, end) if end is not None else -1
        ends.append(-1 if end is None else end)
        nbox.append(n)
        pos.append(p)
        rescored.append(ok)
    m["boxed_end_char"], m["n_matching_boxes"], m["answer_token_position"] = ends, nbox, pos
    m["rescored_matches"] = rescored
    m["model_key"] = mk
    keep = ["model_key", "config_id", "item_id", "strict_correct", "n_output_tokens", "finish_reason", "retok_total",
            "boxed_end_char", "n_matching_boxes", "answer_token_position", "rescored_matches"]
    return m[keep]


def token_table(G, configs, tok_info, spec):
    """Cached per model and tokenizer hash; computes only the configurations not yet in the cache."""
    os.makedirs(TOK_CACHE, exist_ok=True)
    gold = {int(i): scoring.normalize_number(a) for i, a in zip(G["items"]["item_id"], G["items"]["gold_answer"])}
    sc = G["scored"]
    out = []
    for mk in G["panel"]:
        tj = tok_info[mk]["files"]["tokenizer.json"]
        cache = os.path.join(TOK_CACHE, f"tokens_{mk}_{tj['sha256'][:12]}.parquet")
        have = pd.read_parquet(cache) if os.path.exists(cache) else None
        done = set(have["config_id"]) if have is not None else set()
        need = sorted(set(configs) - done)
        if need:
            sdir = sc[(sc["model_key"] == mk) & (sc["arm"] == "direct_answer")][
                ["config_id", "item_id", "strict_correct", "n_output_tokens"]]
            new = token_rows(mk, os.path.join(ROOT, tj["path"]), need, sdir, spec, gold)
            have = new if have is None else pd.concat([have, new], ignore_index=True)
            have.to_parquet(cache, index=False)
            print(f"  tokens: {mk} {len(need)} configurations")
        out.append(have[have["config_id"].isin(configs)])
    return pd.concat(out, ignore_index=True)


def tokenization_record(T, cap):
    """How the boxed answer's end was located and how re-tokenized totals compare with n_output_tokens."""
    per = []
    for mk, g in T.groupby("model_key", sort=False):
        c = g[g["strict_correct"] == 1]
        d = g["retok_total"] - g["n_output_tokens"]
        per.append({"model_key": mk, "n_direct_outputs": int(len(g)), "n_correct": int(len(c)),
                    "n_located": int((c["answer_token_position"] >= 1).sum()),
                    "n_rescored_mismatch": int((c["rescored_matches"] == False).sum()),  # noqa: E712
                    "n_more_than_one_matching_box": int((c["n_matching_boxes"] > 1).sum()),
                    "n_correct_beyond_cap": int((c["answer_token_position"] > cap).sum()),
                    "share_correct_beyond_cap": float((c["answer_token_position"] > cap).mean()) if len(c) else None,
                    "n_correct_within_2_of_cap": int(((c["answer_token_position"] - cap).abs() <= 2).sum()),
                    "median_answer_position": float(c["answer_token_position"].median()) if len(c) else None,
                    "retok_minus_generated": {"share_equal": float((d == 0).mean()), "share_minus_one": float((d == -1).mean()),
                                              "share_within_one": float((d.abs() <= 1).mean()),
                                              "share_within_two": float((d.abs() <= 2).mean()),
                                              "mean": float(d.mean()), "min": int(d.min()), "max": int(d.max())}})
    c = T[T["strict_correct"] == 1]
    d = T["retok_total"] - T["n_output_tokens"]
    total = {"n_direct_outputs": int(len(T)), "n_correct": int(len(c)), "n_located": int((c["answer_token_position"] >= 1).sum()),
             "n_rescored_mismatch": int((c["rescored_matches"] == False).sum()),  # noqa: E712
             "n_more_than_one_matching_box": int((c["n_matching_boxes"] > 1).sum()),
             "n_correct_beyond_cap": int((c["answer_token_position"] > cap).sum()),
             "n_correct_within_2_of_cap": int(((c["answer_token_position"] - cap).abs() <= 2).sum()),
             "retok_minus_generated": {"share_equal": float((d == 0).mean()), "share_minus_one": float((d == -1).mean()),
                                       "share_within_one": float((d.abs() <= 1).mean()),
                                       "share_within_two": float((d.abs() <= 2).mean()),
                                       "mean": float(d.mean()), "q01": float(d.quantile(0.01)),
                                       "q99": float(d.quantile(0.99)), "min": int(d.min()), "max": int(d.max())},
             "share_finish_stop": float((T["finish_reason"] == "stop").mean())}
    return {"total": total, "per_model": per,
            "method": ("The answer line is the one strict extraction reads (src/scoring.strict_extract with strip_latex): "
                       "the last line that is neither empty nor a closing display-math delimiter. Every '\\boxed{' on "
                       "it is brace-matched within the line, and the last box whose content reads as the scored "
                       "answer (with the '\\$' and inline-math tolerance of scorer 1.3.0) gives the end, one "
                       "character past its closing brace. The output text is re-tokenized whole with the model's "
                       "tokenizer.json, without special tokens, truncation or padding; the position is the number of "
                       "tokens whose character span starts before that end.")}


# ============================================================================ variants


def variant_cells(G, variant, units, panel, tokens=None, cap=64):
    sc = G["scored"]
    df = sc[sc["model_key"].isin(panel) & sc["config_id"].isin(set(map(str, units)))]
    if variant == "variant_a":
        t = tokens[["model_key", "config_id", "item_id", "answer_token_position"]]
        df = df.merge(t.assign(arm="direct_answer"), on=["model_key", "arm", "config_id", "item_id"], how="left")
        da = df["arm"] == "direct_answer"
        if df.loc[da, "answer_token_position"].isna().any():
            raise SystemExit("variant A: a direct-arm row has no token record")
        df = df.copy()
        df.loc[da, "strict_correct"] = ins.capped_correct(df.loc[da, "strict_correct"],
                                                          df.loc[da, "answer_token_position"], cap)
    return d2.cell_table(df, panel, units, G["items"]["item_id"].tolist())


def run_variant(G, variant, panel, sel, n_boot, tok_fn, cap, frozen_dir, overwrite):
    s = G["sampler"]
    disc, val = s["discovery_config_ids"], s["validation_config_ids"]
    value = "lenient_correct" if variant == "lenient" else "strict_correct"
    label = "gsm8k" if variant == "strict" else f"gsm8k_{variant}"
    tok_d = tok_fn(sorted(set(disc))) if variant == "variant_a" else None
    cd = variant_cells(G, variant, disc, panel, tok_d, cap)
    Bd, Pd = d2.unit_arrays(cd, disc, panel, value)
    frozen = d2.freeze_predictions(Bd, Pd, f"instability_{label}", panel, s["sha256"])
    frozen = d2.write_frozen(frozen, os.path.join(frozen_dir, f"predictions_instability_{label}.json"),
                             overwrite=overwrite)
    # validation draws are built only now
    allu = list(disc) + list(val)
    tok_all = tok_fn(sorted(set(allu))) if variant == "variant_a" else None
    ca = variant_cells(G, variant, allu, panel, tok_all, cap)
    Bv, Pv = d2.unit_arrays(ca, val, panel, value)
    m = iv.multiplicities(sampling.rng_for(label, "bootstrap"), len(disc), n_boot)
    ps = d2.panel_summary(Bd, Pd, sel, m, VARIANT_LABEL[variant])
    s2 = d2.step2(Bd, Pd, sel, m)
    s3 = d2.evaluate_frozen(frozen, Pv - Bv, sel)
    theta, delta = Bd.mean(0), (Pd - Bd).mean(0)
    return {"label": label, "models": list(panel), "M": len(panel), "value": value,
            "frozen": {"path": rel(os.path.join(frozen_dir, f"predictions_instability_{label}.json")),
                       "sha256": frozen["sha256"],
                       "created": frozen["created"]},
            "curve": {"theta": theta.tolist(), "delta": delta.tolist(), "quadratic": ins.quadratic_curve(theta, delta)},
            "summary": ps, "H": s2["H"], "V_delta": s2["V_delta"], "A": s2["A"], "V_theta": s2["V_theta"],
            "reliability": s2["reliability"], "noise": s2["noise"], "mean_gain": s2["mean_gain"]["estimate"],
            "mean_baseline": s2["mean_baseline"],
            "step3": {"K_discovery": s3["K_discovery"], "K_validation": s3["K_validation"],
                      "contrasts_K24": s3["registered"], "absolute_risk": s3["absolute_risk"]},
            "_arrays": (Bd, Pd, Bv, Pv)}


# ============================================================================ macros


def macro_name(*parts):
    name = "inst" + "".join(p[:1].upper() + p[1:] for p in parts)
    if not name.isalpha():
        raise ValueError(f"macro name must be letters only: {name}")
    return name


SC_M = {"points": "Points", "logit": "Logit", "error_reduction": "Err"}
ST_M = {"random": "Rand", "tercile:low": "Low", "tercile:middle": "Mid", "tercile:high": "High"}
SE_M = {"gsm8k": "Gsm", "dove": "Dove"}
VA_M = {"strict": "Strict", "variant_a": "VarA", "variant_b": "VarB", "lenient": "Lenient"}


def build_macros(R):
    mac = {}
    t, p = ins.tex_num, ins.tex_pct

    def put(value, *parts):
        mac[macro_name(*parts)] = value

    for se in ("gsm8k", "dove"):
        for sc in ins.SCALES:
            x = R["scale"][se][sc]
            d = x["discovery"]
            nd = 3 if sc != "points" else 3
            put(t(d["slope"], nd), SE_M[se], "Slope", SC_M[sc])
            bounded = d["slope_set"]["kind"] == "bounded"
            put(t(d["slope_set"]["a"], nd) if bounded else "unbounded", SE_M[se], "Slope", SC_M[sc], "Lo")
            put(t(d["slope_set"]["b"], nd) if bounded else "unbounded", SE_M[se], "Slope", SC_M[sc], "Hi")
            put(t(d["naive_model_mean_slope"], 2), SE_M[se], "Naive", SC_M[sc])
            put(t(d["R_benefit"], 3), SE_M[se], "R", SC_M[sc])
            put(t(d["R_benefit_ci95"][0], 3), SE_M[se], "R", SC_M[sc], "Lo")
            put(t(d["R_benefit_ci95"][1], 3), SE_M[se], "R", SC_M[sc], "Hi")
            put(t(x["validation"]["slope"], nd), SE_M[se], "SlopeVal", SC_M[sc])
            if "exact_bank" in x:
                put(t(x["exact_bank"]["slope"], nd), SE_M[se], "SlopeExact", SC_M[sc])
                put(t(x["exact_bank"]["R_benefit"], 3), SE_M[se], "RExact", SC_M[sc])
            if sc == "points":
                put(t(d["H"], 2), SE_M[se], "H")
        pr = R["prompt"]["gsm8k_units" if se == "gsm8k" else "dove_bank"]
        for sc in ins.SCALES:
            put(p(pr[sc]["share_opposite"], 1), SE_M[se], "Opp", SC_M[sc])
        for stratum, v in R["panel"][se].items():
            for sc, s in v["pooled_sizes"].items():
                base = (SE_M[se], ST_M[stratum], SC_M[sc])
                put(p(s["share_negative"], 0), *base, "NegShare")
                put(p(s["share_positive"], 0), *base, "PosShare")
                put(f"{s['n_subpanels']:,}".replace(",", "{,}"), *base, "N")
                put(t(s["C_own_gain_disc_val"]["median"], 2), *base, "OwnMedian")
                put(t(s["C_own_gain_disc_val"]["min"], 2), *base, "OwnMin")
                put(t(s["C_benefit_one_draw"]["median"], 2), *base, "OneDrawMedian")
                put(p(s["share_disc_val_same_sign"], 1), *base, "SameSign")
                put(p(s["share_set_above_zero"], 0), *base, "SetAbove")
                put(p(s["share_set_below_zero"], 0), *base, "SetBelow")
    for va, x in R["variants"].items():
        h = x["H"]
        put(t(h["estimate"], 2), VA_M[va], "H")
        hs = h.get("set") or {}
        if hs.get("kind") == "bounded":
            put(t(hs["a"], 2), VA_M[va], "HLo")
            put(t(hs["b"], 2), VA_M[va], "HHi")
        v = x["V_delta"]
        put(f"{v['estimate']:.0f}", VA_M[va], "Vdelta")
        if v["ci95"]:
            put(f"{v['ci95'][0]:.0f}", VA_M[va], "VdeltaLo")
            put(f"{v['ci95'][1]:.0f}", VA_M[va], "VdeltaHi")
        put(str(x["M"]), VA_M[va], "M")
        put(t(x["mean_gain"], 1), VA_M[va], "MeanGain")
        for r in x["step3"]["contrasts_K24"]:
            key = {("constant", "own_gain"): "OwnVsConst", ("constant", "averaged_ols"): "OlsVsConst",
                   ("own_gain", "averaged_ols"): "OlsVsOwn"}.get((r["f"], r["g"]))
            if key:
                put(t(r["estimate"], 1), VA_M[va], key)
        q = x["curve"]["quadratic"]
        if "c2" in q:
            put(t(q["vertex_baseline"], 0), VA_M[va], "Vertex")
            put(t(q["c2"] * 100, 2), VA_M[va], "Curvature")
    if "variant_b" in R["variants"]:
        put(str(len(R["variant_b_rule"]["dropped"])), "VarB", "Dropped")
    tr = R.get("tokenization")
    if tr:
        tt = tr["total"]
        put(p(tt["retok_minus_generated"]["share_equal"], 1), "Tok", "Equal")
        put(p(tt["retok_minus_generated"]["share_within_one"], 1), "Tok", "WithinOne")
        put(f"{tt['n_correct_beyond_cap']:,}".replace(",", "{,}"), "Tok", "Beyond")
        put(f"{tt['n_correct']:,}".replace(",", "{,}"), "Tok", "Correct")
        put(p(tt["n_correct_beyond_cap"] / tt["n_correct"], 1), "Tok", "BeyondShare")
    return mac


# ============================================================================ main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true", help="use tokenizer files already in data/tokenizers")
    ap.add_argument("--overwrite-frozen", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    alpha, sub, var = float(cfg["alpha"]), cfg["subpanels"], cfg["variants"]
    cap = int(var["direct_answer_cap_tokens"])
    sel = d2.load_selection(SELECTION)
    sel1 = d2.load_selection(SELECTION, suite="dataset1")
    m_gsm, m_dove = sel["H"], sel1["H"]
    R = {"analysis": "B", "units": "percentage points unless a scale says otherwise",
         "config": {"path": rel(CONFIG), "sha256": sha256_file(CONFIG), **cfg},
         "selection": {"path": sel["_path"], "sha256": sel["_sha256"], "gsm8k_slope_method": m_gsm,
                       "dove_slope_method": m_dove}}

    # ------------------------------------------------------------------ data
    G = load_gsm8k()
    s = G["sampler"]
    disc, val = s["discovery_config_ids"], s["validation_config_ids"]
    panel = G["panel"]
    n_g = len(G["items"])
    cells = d2.cell_table(G["scored"], panel, list(disc) + list(val), G["items"]["item_id"].tolist())
    Bd, Pd = d2.unit_arrays(cells, disc, panel, "strict_correct")
    Bv, Pv = d2.unit_arrays(cells, val, panel, "strict_correct")
    D = load_dove()
    Dd = (D["B"][D["disc"]], D["P"][D["disc"]], D["B"][D["val"]], D["P"][D["val"]])
    R["settings"] = {"gsm8k": {"models": panel, "families": G["families"], "tiers": G["tiers"], "eci": G["eci"],
                               "K_discovery": len(disc), "K_validation": len(val), "n_items": n_g,
                               "draws": "manifests/sampler_gsm8k.json (24 discovery, 24 validation)"},
                     "dove": {"models": D["panel"], "short": D["short"], "K_discovery": len(D["disc"]),
                              "K_validation": len(D["val"]), "n_bank": int(D["B"].shape[0]), "n_items": D["n_items"],
                              "draws": "manifests/sampler_dove_mmlu.json (24 discovery, 24 validation); exact values "
                                       "over the 2,716-configuration bank"}}
    print(f"data loaded ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ scale
    R["scale"] = {"gsm8k": scale_block(Bd, Pd, Bv, Pv, n_g, m_gsm, alpha),
                  "dove": scale_block(*Dd, D["n_items"], m_dove, alpha, bank=(D["B"], D["P"]))}
    R["orientation_plan_v2"] = {"gsm8k": {"points": -0.23, "logit": -0.10, "error_reduction": 0.27},
                                "dove": "all three positive",
                                "note": "naive model-mean slopes, used as a reference"}

    # ------------------------------------------------------------------ prompt
    ref_g = {sc: R["scale"]["gsm8k"][sc]["discovery"]["slope"] for sc in ins.SCALES}
    ref_d = {sc: R["scale"]["dove"][sc]["exact_bank"]["slope"] for sc in ins.SCALES}
    ref_dd = {sc: R["scale"]["dove"][sc]["discovery"]["slope"] for sc in ins.SCALES}
    R["prompt"] = prompt_block({"gsm8k_units": (np.vstack([Bd, Bv]), np.vstack([Pd, Pv]), ref_g)}, n_g)
    R["prompt"].update(prompt_block({"dove_bank": (D["B"], D["P"], ref_d),
                                     "dove_units": (np.vstack([Dd[0], Dd[2]]), np.vstack([Dd[1], Dd[3]]), ref_dd)},
                                    D["n_items"]))
    R["prompt"]["note"] = ("Single-draw OLS slopes against a reference slope on the same scale: GSM8K, the 48 collected "
                           "draws against the 24-draw discovery estimate; DOVE, every configuration of the bank "
                           "against the exact bank value, and the 48 sampled draws against the discovery estimate.")
    print(f"scale and prompt done ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ panel
    idx = {mk: i for i, mk in enumerate(panel)}
    strata_g = {"random": list(range(len(panel)))}
    for tier in ("low", "middle", "high"):
        strata_g[f"tercile:{tier}"] = [idx[mk] for mk in panel if G["tiers"][mk] == tier]
    pg, rows_g = panel_block("gsm8k", (Bd, Pd, Bv, Pv), n_g, strata_g, sub["sizes"], sub["n_resamples"],
                             sub["seed_label"], m_gsm, alpha)
    pdv, rows_d = panel_block("dove", Dd, D["n_items"], {"random": list(range(len(D["panel"])))}, [4],
                              sub["n_resamples"], sub["seed_label"], m_dove, alpha, bank=(D["B"], D["P"]))
    R["panel"] = {"gsm8k": pg, "dove": pdv}
    R["panel_note"] = ("Slopes are cross-draw estimates on the 24 discovery draws of each subpanel; the 95% set uses "
                       "the dataset's H method. Own-gain agreement compares the subpanel's model orderings by mean "
                       "discovery gain and by mean validation gain (24 draws each); one-draw agreement is C_benefit "
                       "between distinct discovery draws. DOVE uses only its five 4-model subpanels, with the "
                       "exact bank slope beside each.")
    print(f"panel done ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ variants
    os.makedirs(a.out, exist_ok=True)
    frozen_dir = os.path.join(ROOT, "manifests")
    med = (G["scored"][G["scored"]["arm"] == "direct_answer"].groupby("model_key")["n_output_tokens"].median())
    med_units = {}
    for mk in panel:
        x = G["scored"][(G["scored"]["model_key"] == mk) & (G["scored"]["arm"] == "direct_answer")]
        reps = pd.Series(list(disc) + list(val)).value_counts()
        w = x["config_id"].map(reps).to_numpy()
        order = np.argsort(x["n_output_tokens"].to_numpy(), kind="stable")
        cw = np.cumsum(w[order])
        med_units[mk] = float(x["n_output_tokens"].to_numpy()[order][np.searchsorted(cw, cw[-1] / 2)])
    dropped = ins.long_direct_models({mk: float(med[mk]) for mk in panel}, cap)
    R["variant_b_rule"] = {"cap_tokens": cap, "median_direct_tokens": {mk: float(med[mk]) for mk in panel},
                           "median_direct_tokens_unit_weighted": med_units, "dropped": dropped,
                           "dropped_if_unit_weighted": ins.long_direct_models(med_units, cap),
                           "note": "median over the collection's distinct direct-arm outputs (29 configurations x "
                                   "1,319 items); the unit-weighted median counts each configuration as often as "
                                   "the 48 draws use it, as a check"}
    tok_info = None
    if not a.no_download:
        tok_info = download_tokenizers(G["reg"], panel, var["tokenizer_files"])
    else:
        tok_info = local_tokenizers(G["reg"], panel, var["tokenizer_files"])
    spec = scoring.spec_from_dict(json.load(open(os.path.join(ROOT, "manifests", "gsm8k_scorer.json")))["format_spec"])
    try:
        import tokenizers
        tok_version = tokenizers.__version__
    except ImportError:
        tok_version = None
    R["tokenizers"] = {"files": tok_info, "library_version": tok_version,
                       "collection_version": var["tokenizers_version"]}
    tok_store = {}

    def tok_fn(configs):
        T = token_table(G, configs, tok_info, spec)
        tok_store["T"] = T
        return T

    R["variants"] = {}
    arrays = {}
    for va in VARIANTS:
        if va == "variant_a" and tok_info is None:
            print("variant A skipped: tokenizer files missing (run without --no-download)")
            continue
        pv = [mk for mk in panel if mk not in dropped] if va == "variant_b" else panel
        r = run_variant(G, va, pv, sel, int(cfg["n_boot"]), tok_fn, cap, frozen_dir, a.overwrite_frozen)
        arrays[va] = r.pop("_arrays")
        R["variants"][va] = r
        print(f"variant {va} done ({time.time() - t0:.0f} s)")
    if "T" in tok_store:
        R["tokenization"] = tokenization_record(tok_store["T"], cap)
        R["tokenization"]["token_table"] = ("per-output rows cached in " + rel(TOK_CACHE) +
                                            "/tokens_<model_key>_<tokenizer sha256 prefix>.parquet (git-ignored)")

    # ------------------------------------------------------------------ outputs
    R["macros"] = build_macros(R)
    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "tokenizers": tok_version}
    tdir = os.path.join(a.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    pd.DataFrame(rows_g + rows_d).to_csv(os.path.join(tdir, "subpanels.csv"), index=False)
    write_tables(R, tdir)
    dump(R, os.path.join(a.out, "instability.json"))
    md = render(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(md)
    print(f"done in {time.time() - t0:.0f} s -> {rel(a.out)}")
    return 0


# ============================================================================ report


def _n(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def _pc(x, nd=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def _set(s, nd=3):
    if s is None:
        return "none"
    if s["kind"] == "bounded":
        return f"[{_n(s['a'], nd)}, {_n(s['b'], nd)}]"
    if s["kind"] == "exclusive":
        return f"(−∞, {_n(s['a'], nd)}] ∪ [{_n(s['b'], nd)}, ∞)"
    return s["kind"]


def _word(slope):
    if slope is None or not np.isfinite(slope):
        return "undefined"
    return "negative" if slope < 0 else ("positive" if slope > 0 else "zero")


def render(R):
    S, G, Dv = R["scale"], R["settings"]["gsm8k"], R["settings"]["dove"]
    L = ["# Analysis B: who benefits more, across scales, panels, prompts and scoring", "",
         "## Setup", "",
         f"- **Settings.** GSM8K: {len(G['models'])} models, zero-shot chain of thought against direct answer, "
         f"{G['n_items']:,} items, {G['K_discovery']} discovery and {G['K_validation']} validation draws "
         "(`manifests/sampler_gsm8k.json`), strict scoring by scorer 1.3.0. DOVE pooled MMLU block: "
         f"{len(Dv['models'])} models, five-shot against zero-shot, the {Dv['K_discovery']} discovery and "
         f"{Dv['K_validation']} validation draws of `manifests/sampler_dove_mmlu.json`, and exact values over the "
         f"{Dv['n_bank']:,}-configuration bank.",
         "- **Slope.** The capability slope is A / V_x, the cross-draw covariance of benefit with capability over "
         "the cross-draw variance of capability, on the discovery draws (Appendix A moments, panel denominator "
         "M − 1). A negative slope means weaker models gain more; in points H = −10 × slope.",
         "- **Scales.** Points: benefit P − B against B. Logits: logit(P) − logit(B) against logit(B). Error "
         "reduction: 100 (P − B) / (100 − B), the share of the baseline error removed in pp, against B. Each "
         "per-draw score is transformed before any moment is taken.",
         "- **Own-gain reliability.** R_benefit is the cross-draw variance of benefit over its mean within-draw "
         "panel variance (Appendix A `reliability`), on the same scale.",
         f"- **Intervals.** 95%, from the draw-level jackknife: a Fieller set for the GSM8K slope and a jackknife t "
         f"interval for the DOVE slope (each dataset's calibrated H method), and a jackknife t interval for "
         f"R_benefit. The calibration covered H in points only, so the logit and error-reduction intervals and all "
         f"R_benefit intervals are descriptive.",
         f"- **Choices.** Fixed in `config/instability.yaml` (sha256 `{R['config']['sha256'][:12]}`) before any "
         "subpanel, scale or variant result was computed.",
         "- **Files.** `instability.json` (with the `macros` for `paper/make_numbers.py`) and `tables/*.csv`.", "",
         "## Scale", "",
         "| Setting | Scale | Slope [95% set] | Naive model-mean slope | R_benefit [95%] | Slope on validation draws |",
         "|---|---|---|---|---|---|"]
    for se in ("gsm8k", "dove"):
        for sc in ins.SCALES:
            d, v = S[se][sc]["discovery"], S[se][sc]["validation"]
            L.append(f"| {SETTING_LABEL[se]} | {ins.SCALE_LABELS[sc]} | {_n(d['slope'], 3)} {_set(d['slope_set'])} | "
                     f"{_n(d['naive_model_mean_slope'], 3)} | {_n(d['R_benefit'], 3)} "
                     f"[{_n(d['R_benefit_ci95'][0], 3)}, {_n(d['R_benefit_ci95'][1], 3)}] | {_n(v['slope'], 3)} |")
    L += ["", "Units: points, pp of gain per pp of baseline; logits, logits of gain per logit of baseline; error "
          "reduction, pp of error reduction per pp of baseline.", ""]
    ex = {sc: S["dove"][sc]["exact_bank"] for sc in ins.SCALES}
    L.append("Exact DOVE bank values: slopes " + ", ".join(f"{_n(ex[sc]['slope'], 3)} ({ins.SCALE_LABELS[sc].lower()})"
                                                         for sc in ins.SCALES)
             + "; R_benefit " + ", ".join(_n(ex[sc]["R_benefit"], 3) for sc in ins.SCALES) + ".")
    g = {sc: S["gsm8k"][sc]["discovery"] for sc in ins.SCALES}
    words = [_word(g[sc]["slope"]) for sc in ins.SCALES]
    excl = all(g[sc]["slope_set_sign"] in ("above zero", "below zero") for sc in ins.SCALES)
    dw = [_word(S['dove'][sc]['discovery']['slope']) for sc in ins.SCALES]
    L += ["", f"On GSM8K the slope is {words[0]} in points, {words[1]} in logits and {words[2]} as error reduction"
          + (", and each 95% set excludes zero." if excl else ".")
          + (" Weaker models gain more points and more logits, and stronger models remove a larger share of their "
             "remaining error." if words == ["negative", "negative", "positive"] else "")
          + (f" On DOVE the slope is {dw[0]} on all three scales" if len(set(dw)) == 1 else
             f" On DOVE the slope is {dw[0]}, {dw[1]} and {dw[2]} on the three scales")
          + (", so stronger models gain more on every scale there." if dw == ["positive"] * 3 else "."),
          "",
          f"**Check against the orientation numbers** (naive model-mean slopes: −0.23, −0.10 and "
          f"+0.27 on GSM8K; all positive on DOVE). The naive slopes here are {_n(g['points']['naive_model_mean_slope'], 3)}, "
          f"{_n(g['logit']['naive_model_mean_slope'], 3)} and {_n(g['error_reduction']['naive_model_mean_slope'], 3)}, "
          "the same to the rounding. The orientation numbers match when the logit slope uses the logit of the "
          "baseline as capability and the error-reduction slope uses the baseline in pp, which is how the scales "
          f"are defined here. On GSM8K the cross-draw slopes differ from the naive ones by at most "
          f"{max(abs(g[sc]['slope'] - g[sc]['naive_model_mean_slope']) for sc in ins.SCALES):.4f}, because the "
          "within-draw noise that couples gain and baseline is small next to V_x there. On DOVE they differ more: "
          f"{_n(S['dove']['points']['discovery']['slope'], 3)} against "
          f"{_n(S['dove']['points']['discovery']['naive_model_mean_slope'], 3)} in points on the same 24 draws. In "
          f"points the DOVE estimate is H = {_n(S['dove']['points']['discovery']['H'], 2)} "
          f"{_set(S['dove']['points']['discovery']['H_set'], 2)} with exact bank value "
          f"{_n(S['dove']['points']['exact_bank']['H'], 2)}, matching `results/dataset1/registered_intervals.json` "
          "and `results/dataset1/summary.md`; the GSM8K value H = "
          f"{_n(S['gsm8k']['points']['discovery']['H'], 2)} {_set(S['gsm8k']['points']['discovery']['H_set'], 2)} "
          "matches `results/dataset2/summary.md`.",
          "",
          f"**Reliability.** On GSM8K R_benefit is {', '.join(_n(g[sc]['R_benefit'], 3) for sc in ins.SCALES)} in points, "
          f"logits and error reduction. On DOVE it is "
          f"{', '.join(_n(S['dove'][sc]['discovery']['R_benefit'], 2) for sc in ins.SCALES)} on the discovery draws "
          f"(exact: {', '.join(_n(ex[sc]['R_benefit'], 2) for sc in ins.SCALES)}). The scale changes how far apart the "
          "models' true gains lie relative to the prompt noise, so it changes how often a one-prompt claim about who "
          "benefits repeats.",
          "",
          "**Undefined cells.** " + "; ".join(
              f"{SETTING_LABEL[se]}: " + ", ".join(
                  f"{ins.SCALE_LABELS[sc].lower()} {sum(v for k, v in S[se][sc]['replaced_cells'].items() if k != 'of')}"
                  for sc in ins.SCALES) for se in ("gsm8k", "dove"))
          + ". No per-draw score is 0% or 100%, so no empirical-proportion replacement was needed.", "",
          "## Prompt", "",
          "Share of single-draw slopes whose sign is opposite to the reference slope on the same scale.", "",
          "| Draws | Reference | Points | Logits | Error reduction |", "|---|---|---|---|---|"]
    P = R["prompt"]
    for key, lab, ref in (("gsm8k_units", "GSM8K, 48 collected draws", "24-draw discovery slope"),
                          ("dove_bank", "DOVE, every bank configuration (2,716)", "exact bank slope"),
                          ("dove_units", "DOVE, 48 sampled draws", "24-draw discovery slope")):
        L.append(f"| {lab} | {ref} | " + " | ".join(
            f"{_pc(P[key][sc]['share_opposite'], 1)} ({P[key][sc]['n_opposite']:,} of {P[key][sc]['n_defined']:,})"
            for sc in ins.SCALES) + " |")
    L += ["", f"The earlier values are reproduced: {_pc(P['dove_bank']['points']['share_opposite'], 1)} of DOVE "
          f"configurations and {_pc(P['gsm8k_units']['points']['share_opposite'], 0)} of GSM8K draws give a slope of "
          "the opposite sign in points. No single-draw slope was exactly zero or undefined.", "",
          "## Panel", "",
          "Subpanels of 4 to 6 models. GSM8K: 2,000 random subpanels per size (from 3,876, 11,628 and 27,132), and every "
          "subpanel within each ECI tercile (low 7 models, middle 6, high 6: 63, 22 and 22 subpanels). DOVE: its five "
          "4-model subpanels. Each slope is the cross-draw estimate on the subpanel's 24 discovery draws. Own-gain "
          "agreement is the share of model pairs ordered the same way by mean discovery gain and mean validation gain "
          "(ties count one half); one-draw agreement is C_benefit between distinct discovery draws. Sizes are pooled "
          "below; `tables/subpanel_summary.csv` has each size.", "",
          "| Setting, stratum | Scale | Subpanels | Slope < 0 | Slope > 0 | Set below 0 | Set above 0 | Same sign on validation | Own-gain agreement, median (min) | One-draw C_benefit, median |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for se in ("gsm8k", "dove"):
        for stratum, w in R["panel"][se].items():
            for sc in ins.SCALES:
                s = w["pooled_sizes"][sc]
                L.append(f"| {SETTING_LABEL[se]}, {STRATUM_LABEL[stratum]} | {ins.SCALE_LABELS[sc]} | {s['n_subpanels']:,} | "
                         f"{_pc(s['share_negative'])} | {_pc(s['share_positive'])} | {_pc(s['share_set_below_zero'])} | "
                         f"{_pc(s['share_set_above_zero'])} | {_pc(s['share_disc_val_same_sign'], 1)} | "
                         f"{_n(s['C_own_gain_disc_val']['median'], 3)} ({_n(s['C_own_gain_disc_val']['min'], 3)}) | "
                         f"{_n(s['C_benefit_one_draw']['median'], 3)} |")
    pr = R["panel"]["gsm8k"]["random"]["pooled_sizes"]["points"]
    lo, mid, hi = (R["panel"]["gsm8k"][f"tercile:{t}"]["pooled_sizes"]["points"] for t in ("low", "middle", "high"))
    chk = [R["panel"]["gsm8k"]["random"]["sizes"][str(k)]["points"] for k in (4, 5, 6)]
    L += ["", f"On GSM8K in points, {_pc(pr['share_negative'])} of random subpanels give a negative slope (weaker "
          f"models gain more) and {_pc(pr['share_positive'])} a positive one; the share of negative slopes rises with "
          f"size ({', '.join(_pc(c['share_negative']) for c in chk)} at 4, 5 and 6 models). Within terciles the sign "
          f"follows the tercile's place on the curve: the low tercile gives a positive slope in "
          f"{_pc(lo['share_positive'])} of its subpanels, "
          + ("the middle and high terciles a negative one in every subpanel. "
             if mid["share_negative"] == 1 and hi["share_negative"] == 1 else
             f"the middle and high terciles a negative one in {_pc(mid['share_negative'])} and "
             f"{_pc(hi['share_negative'])}. ")
          + "Within a subpanel the slope's sign repeats on "
          f"the validation draws in {_pc(pr['share_disc_val_same_sign'], 1)} of random subpanels, so on GSM8K the "
          "sign is set by the panel and barely by the prompt. Own-gain orderings repeat between discovery and "
          f"validation in nearly every subpanel: median agreement {_n(pr['C_own_gain_disc_val']['median'], 2)}, lowest "
          f"{_n(pr['C_own_gain_disc_val']['min'], 2)}.",
          ""]
    dz = {sc: R["panel"]["dove"]["random"]["sizes"]["4"][sc] for sc in ins.SCALES}
    npos = [dz[sc]["slope_signs"]["positive"] for sc in ins.SCALES]
    epos = [dz[sc]["exact_bank_signs"]["positive"] for sc in ins.SCALES]
    L += [("On DOVE all five 4-model subpanels give a positive slope on every scale, and so do their exact bank "
           "slopes. " if npos == [5, 5, 5] and epos == [5, 5, 5] else
           f"On DOVE the five 4-model subpanels give a positive slope in {npos[0]}, {npos[1]} and {npos[2]} of 5 "
           f"(points, logits, error reduction); the exact bank slopes of the same subpanels are positive in "
           f"{epos[0]}, {epos[1]} and {epos[2]}. ")
          + "Own-gain orderings repeat less there than on GSM8K: median agreement "
          f"{_n(dz['points']['C_own_gain_disc_val']['median'], 2)} in points between the two 24-draw groups, and "
          f"one-draw C_benefit {_n(dz['points']['C_benefit_one_draw']['median'], 2)}.",
          "",
          "**Check on the sampled strata.** Over every subpanel of each size, the share of negative slopes in points is "
          + ", ".join(f"{_pc(c['all_subsets_check']['share_negative'], 1)} (sampled {_pc(c['share_negative'], 1)})"
                      for c in chk) + " at 4, 5 and 6 models.", ""]
    # ------------------------------------------------------------------ shape
    V = R["variants"]
    vb = R["variant_b_rule"]
    L += ["## Shape and scoring variants", "",
          "Model-mean gain against model-mean baseline on GSM8K (discovery draws), with a descriptive quadratic fit, "
          "under four scorings. Steps 1–3 of the main analysis are rerun for each with the Dataset 2 suite's interval "
          "methods (calibrated at M = 20). Each variant's prediction rules were fitted on its discovery draws and "
          "frozen to `results/instability/frozen/` before its validation draws were built.", "",
          f"- **Variant A** (answer within {vb['cap_tokens']} tokens): a direct-arm output stays correct only when its "
          "boxed answer ends within its first 64 output tokens. The chain-of-thought arm, panel and items are unchanged.",
          f"- **Variant B** (short direct answers): models whose median direct-arm output exceeds {vb['cap_tokens']} "
          "tokens leave the panel. Dropped: " + ", ".join(f"{mk} ({vb['median_direct_tokens'][mk]:.0f})" for mk in vb["dropped"])
          + f"; llama-2-7b-chat-hf stays at {vb['median_direct_tokens']['llama-2-7b-chat-hf']:.0f}. The median is over "
          "the collection's distinct direct-arm outputs; weighting each configuration by its use among the 48 draws "
          "drops the same models.",
          "- **Lenient extraction**: the last number in the output, for both arms.", "",
          "| Scoring | M | Mean gain (pp) | H [Fieller set] | V_delta (pp²) [95%] | Quadratic vertex (baseline, pp) | Constant vs own gain | Constant vs averaged regression | Own gain vs averaged regression |",
          "|---|---|---|---|---|---|---|---|---|"]
    for va, x in V.items():
        q = x["curve"]["quadratic"]
        st3 = {(r["f"], r["g"]): r for r in x["step3"]["contrasts_K24"]}

        def rd(f, g):
            r = st3.get((f, g))
            return "n/a" if r is None else f"{_n(r['estimate'], 1)} [{_n(r['ci95'][0], 1)}, {_n(r['ci95'][1], 1)}]"
        vd = x["V_delta"]
        L.append(f"| {VARIANT_LABEL[va]} | {x['M']} | {_n(x['mean_gain'], 1)} | {_n(x['H']['estimate'], 2)} "
                 f"{_set(x['H'].get('set'), 2)} | {_n(vd['estimate'], 0)} [{_n(vd['ci95'][0], 0)}, {_n(vd['ci95'][1], 0)}] | "
                 f"{_n(q.get('vertex_baseline'), 0)} | {rd('constant', 'own_gain')} | {rd('constant', 'averaged_ols')} | "
                 f"{rd('own_gain', 'averaged_ols')} |")
    L += ["", "Risk differences in pp² at K = 24 on the validation draws, positive values favouring the second rule, "
          "with jackknife t intervals. The strict row reproduces `results/dataset2/summary.md`.", ""]
    if "variant_a" in V and R.get("tokenization"):
        T = R["tokenization"]
        tt = T["total"]
        rg = tt["retok_minus_generated"]
        beyond = [r for r in T["per_model"] if r["n_correct_beyond_cap"] > 0]
        beyond.sort(key=lambda r: -r["n_correct_beyond_cap"])
        L += [f"**Variant A, token positions.** {T['method']} The boxed answer was located in {tt['n_located']:,} of "
              f"the {tt['n_correct']:,} correct direct answers; re-scoring their text gave the scorer's answer with "
              f"{tt['n_rescored_mismatch']} mismatches, and {tt['n_more_than_one_matching_box']} had more than one "
              f"matching box on the answer line (the last would be used). Over all {tt['n_direct_outputs']:,} direct-arm "
              f"outputs the re-tokenized length equals n_output_tokens in {_pc(rg['share_equal'], 1)}, is one less in "
              f"{_pc(rg['share_minus_one'], 1)} (consistent with generation counting the end-of-sequence token), and is "
              f"within two tokens in {_pc(rg['share_within_two'], 2)} (range {_n(rg['min'], 0)} to {_n(rg['max'], 0)}). "
              f"{tt['n_correct_beyond_cap']:,} correct direct answers ({_pc(tt['n_correct_beyond_cap'] / tt['n_correct'], 1)}) "
              f"end after token 64 and are scored wrong; {tt['n_correct_within_2_of_cap']:,} end within two tokens of "
              "the cap, the band a one-token tokenization difference could move.", "",
              "| Model | Correct direct answers | Ending after token 64 | Share | Median end position |", "|---|---|---|---|---|"]
        for r in beyond:
            L.append(f"| {r['model_key']} | {r['n_correct']:,} | {r['n_correct_beyond_cap']:,} | "
                     f"{_pc(r['share_correct_beyond_cap'], 1)} | {_n(r['median_answer_position'], 0)} |")
        L += ["", f"Models not listed lose no correct direct answer. Tokenizer files and their sha256 are in "
              f"`instability.json` (`tokenizers`); the library is tokenizers {R['tokenizers']['library_version']}, the "
              f"collection's version.", ""]
    s_ = V.get("strict")
    if s_:
        q0 = s_["curve"]["quadratic"]
        L += [("Under the main scoring the fitted curve is an inverted U" if q0["inverted_u"] else
               "Under the main scoring the fitted quadratic is not an inverted U")
              + f" with its vertex at a baseline of {_n(q0['vertex_baseline'], 0)} pp (quadratic R² "
              f"{_n(q0['r2_quadratic'], 2)}, line {_n(q0['r2_linear'], 2)}). A linear slope through such a curve "
              "depends on how many models sit on each side of the vertex."]
        if "variant_b" in V:
            b = V["variant_b"]
            th0 = dict(zip(s_["models"], s_["curve"]["theta"]))
            gone = ", ".join(f"{mk} {th0[mk]:.1f}" for mk in sorted(vb["dropped"], key=lambda k: -th0[k]))
            hb = b["H"]["estimate"]
            L[-1] += (f" Variant B removes the {len(vb['dropped'])} models with long direct answers (direct-answer "
                      f"accuracy {gone} pp), and H becomes {_n(hb, 2)} {_set(b['H'].get('set'), 2)}"
                      + (": in the remaining panel stronger models gain more." if hb < 0 else "."))
        if "variant_a" in V:
            a_ = V["variant_a"]
            L[-1] += (f" Variant A keeps the panel and lowers the direct-answer accuracy of every model whose correct "
                      f"direct answers come after some working; H is {_n(a_['H']['estimate'], 2)} "
                      f"{_set(a_['H'].get('set'), 2)} and the vertex moves to "
                      f"{_n(a_['curve']['quadratic']['vertex_baseline'], 0)} pp.")
        if "lenient" in V:
            le = V["lenient"]
            L[-1] += f" Lenient extraction gives H {_n(le['H']['estimate'], 2)} {_set(le['H'].get('set'), 2)}."
        own_ok = all({(r["f"], r["g"]): r for r in x["step3"]["contrasts_K24"]}[("own_gain", "averaged_ols")]["ci95"][1] < 0
                     and {(r["f"], r["g"]): r for r in x["step3"]["contrasts_K24"]}[("constant", "own_gain")]["ci95"][0] > 0
                     for x in V.values())
        vmin = min(x["V_delta"]["estimate"] for x in V.values())
        L += ["", ("In every variant own gain predicts the validation gains better than the constant and than the "
                   "averaged capability regression, with intervals far from zero, " if own_ok else
                   "Own gain does not beat both rules in every variant (table above), ")
              + f"and V_delta is at least {vmin:.0f} pp²."]
        # per-model values: where the small gains at the two ends come from
        cols = [va for va in ("strict", "variant_a", "lenient") if va in V]
        per = {va: dict(zip(V[va]["models"], zip(V[va]["curve"]["theta"], V[va]["curve"]["delta"]))) for va in cols}
        order = sorted(s_["models"], key=lambda k: -per["strict"][k][0])
        L += ["", "Per-model direct-answer accuracy θ and gain δ (pp, discovery draws); a dash marks a model variant B drops.",
              "", "| Model | Main θ | Main δ | " + " | ".join(f"{VARIANT_LABEL[va].split(':')[0]} θ | "
                                                     f"{VARIANT_LABEL[va].split(':')[0]} δ" for va in cols[1:])
              + " | In variant B |", "|---|---|---|" + "---|---|" * (len(cols) - 1) + "---|"]
        for mk in order:
            L.append(f"| {mk} | {per['strict'][mk][0]:.1f} | {per['strict'][mk][1]:.1f} | "
                     + " | ".join(f"{per[va][mk][0]:.1f} | {per[va][mk][1]:.1f}" for va in cols[1:])
                     + f" | {'—' if mk in vb['dropped'] else 'yes'} |")
        top, bottom = order[:3], order[-3:]
        ends = []
        if "variant_a" in per:
            ends.append("Under variant A the three strongest models' direct-answer accuracy falls to "
                        + ", ".join(f"{per['variant_a'][mk][0]:.1f}" for mk in top)
                        + " pp and their gains become " + ", ".join(f"{per['variant_a'][mk][1]:.1f}" for mk in top)
                        + " pp (main scoring " + ", ".join(f"{per['strict'][mk][1]:.1f}" for mk in top) + ").")
        if "lenient" in per:
            ends.append("Lenient extraction moves the three weakest models' gains from "
                        + ", ".join(f"{per['strict'][mk][1]:.1f}" for mk in bottom) + " to "
                        + ", ".join(f"{per['lenient'][mk][1]:.1f}" for mk in bottom) + " pp.")
        if "variant_a" in per and "lenient" in per:
            a_top = min(per["variant_a"][mk][1] for mk in top)
            l_bot = max(per["lenient"][mk][1] for mk in bottom)
            if a_top > 50 and l_bot < 50:
                ends.append("At the top of the range, then, the small gains under the main scoring mostly reflect "
                            "direct answers that already contain the working; at the bottom, extraction accounts for "
                            "part of them, and the weakest models stay low in both arms under every scoring.")
        if ends:
            L += ["", " ".join(ends)]
    L += ["", "## Choices made here", "",
          "- Capability is the baseline on the same scale as the gain where the scale has one (points, logits); "
          "error reduction has no baseline transform, so its capability is the baseline in pp. These definitions "
          "reproduce the plan's orientation numbers.",
          "- A cell where a scale is undefined would have both arms replaced by (k + ½)/(n + 1); none occurred. "
          "For DOVE, n is the block's 5,597 items and k the pooled share of them.",
          "- GSM8K uses the 24 discovery draws for every slope and R_benefit, as Steps 1 and 2 do; DOVE uses its 24 "
          "discovery draws for the same quantities and gives the exact bank values beside them.",
          "- Random subpanels are sampled without replacement among distinct subpanels, 2,000 per size, by the "
          "seeded stream of `config/instability.yaml`; strata with at most 2,000 subpanels are enumerated.",
          "- The frozen predictions of the variants are written to `results/instability/frozen/`, not `manifests/`, "
          "so the main analysis's manifests stay untouched.",
          "- Variant B's median is over distinct outputs (29 configurations × 1,319 items), with the unit-weighted "
          "median as a check; both drop the same five models.",
          "- Variant A's token table (`data/processed/instability/`, git-ignored) was built for the discovery "
          "configurations first and for the validation-only configurations after the freeze; later runs read it "
          "from that cache. The strict validation outcomes behind it had already been read by the main analysis.",
          "- Variant A counts a correct direct answer as wrong when its box ends at token 65 or later of the "
          "re-tokenized text; re-tokenized and generated lengths differ by about one token, so the "
          + (f"{R['tokenization']['total']['n_correct_within_2_of_cap']:,} answers " if R.get("tokenization") else "")
          + "within two tokens of the cap are the part this choice can move.", ""]
    return "\n".join(L) + "\n"



def write_tables(R, tdir):
    rows = []
    for se in ("gsm8k", "dove"):
        for sc in ins.SCALES:
            x = R["scale"][se][sc]
            for blk in ("discovery", "validation"):
                d = x[blk]
                rows.append({"setting": se, "scale": sc, "draws": blk, "K": d["K"], "M": d["M"], "slope": d["slope"],
                             "set_kind": d["slope_set"]["kind"], "set_lo": d["slope_set"]["a"],
                             "set_hi": d["slope_set"]["b"], "set_sign": d["slope_set_sign"],
                             "naive_model_mean_slope": d["naive_model_mean_slope"], "R_benefit": d["R_benefit"],
                             "R_lo": d["R_benefit_ci95"][0], "R_hi": d["R_benefit_ci95"][1], "V_y": d["V_y"],
                             "W_y": d["W_y"], "V_x": d["V_x"], "A": d["A"], "unit": x["unit"]})
            if "exact_bank" in x:
                e = x["exact_bank"]
                rows.append({"setting": se, "scale": sc, "draws": "exact bank", "K": e["n_configurations"], "M": e["M"],
                             "slope": e["slope"], "R_benefit": e["R_benefit"], "V_y": e["V_y"], "W_y": e["W_y"],
                             "V_x": e["V_x"], "A": e["A"], "unit": x["unit"]})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "scales.csv"), index=False)
    rows = []
    for name, v in R["prompt"].items():
        if name == "note":
            continue
        for sc, x in v.items():
            rows.append({"set": name, "scale": sc, **x})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "single_draw_signs.csv"), index=False)
    rows = []
    for se, v in R["panel"].items():
        for stratum, w in v.items():
            for size, z in list(w["sizes"].items()) + [("4-6 pooled", w["pooled_sizes"])]:
                for sc, s in z.items():
                    rows.append({"setting": se, "stratum": stratum, "size": size, "scale": sc, "n": s["n_subpanels"],
                                 "mode": s["mode"], "share_negative": s["share_negative"],
                                 "share_positive": s["share_positive"], "share_set_above_zero": s["share_set_above_zero"],
                                 "share_set_below_zero": s["share_set_below_zero"],
                                 "share_disc_val_same_sign": s["share_disc_val_same_sign"],
                                 "C_own_median": s["C_own_gain_disc_val"].get("median"),
                                 "C_own_min": s["C_own_gain_disc_val"].get("min"),
                                 "C_one_draw_median": s["C_benefit_one_draw"].get("median"),
                                 "all_subsets_share_negative": (s.get("all_subsets_check") or {}).get("share_negative")})
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "subpanel_summary.csv"), index=False)
    rows = []
    for va, x in R["variants"].items():
        hs = x["H"].get("set") or {}
        row = {"variant": va, "M": x["M"], "H": x["H"]["estimate"], "H_kind": hs.get("kind"), "H_lo": hs.get("a"),
               "H_hi": hs.get("b"), "V_delta": x["V_delta"]["estimate"],
               "V_delta_lo": (x["V_delta"]["ci95"] or [None, None])[0],
               "V_delta_hi": (x["V_delta"]["ci95"] or [None, None])[1], "V_delta_upper95": x["V_delta"]["upper95"],
               "C_benefit": x["summary"]["C_benefit"], "gap": x["summary"]["gap"], "mean_gain": x["mean_gain"],
               "mean_baseline": x["mean_baseline"], **{f"quad_{k}": v for k, v in x["curve"]["quadratic"].items()
                                                         if not isinstance(v, str)}}
        for r in x["step3"]["contrasts_K24"]:
            row[f"risk_{r['f']}_vs_{r['g']}"] = r["estimate"]
            row[f"risk_{r['f']}_vs_{r['g']}_lo"] = (r["ci95"] or [None, None])[0]
            row[f"risk_{r['f']}_vs_{r['g']}_hi"] = (r["ci95"] or [None, None])[1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "variants.csv"), index=False)
    if R.get("tokenization"):
        pd.DataFrame([{**{k: v for k, v in r.items() if k != "retok_minus_generated"},
                       **{f"retok_{k}": v for k, v in r["retok_minus_generated"].items()}}
                      for r in R["tokenization"]["per_model"]]).to_csv(os.path.join(tdir, "variant_a_by_model.csv"),
                                                                       index=False)


if __name__ == "__main__":
    raise SystemExit(main())
