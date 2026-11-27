r"""Score the Nebius V4-Flash replies with the main panel's scorers and write the reports (scripts/nebius_panel.py).

Reads data/frontier_panel/nebius_panel_raw.jsonl (one line per reply), keeps the answered keys, and per benchmark:
  outputs  data/frontier_panel/outputs/<task>.parquet, the shard columns of data/d2_out (request_id, model_key, arm,
           config_id, item_id, prompt_sha256, output_text, n_prompt_tokens, n_output_tokens, finish_reason, cached),
           output_text without the prefill, as the runner stores it; GSM8K and BIG-Bench Hard forced replies cut after
           their first "}" (the collection's stop string, kept);
  scored   data/frontier_panel/scored/<task>_scored.parquet, the columns of data/processed/<task>/scored.parquet plus
           the content reading (content_class / content_correct; HotpotQA content_correct and content_f1), the
           prefill prepended before scoring as src/d2_io.inspect_work_dir does, and a forced reply the cut left with
           its box open read as the closed box it was meant to be (src/forced_stop.py).
Scorers: GSM8K src/scoring.py with manifests/gsm8k_scorer.json; MATH-500 src/math_task.py; BIG-Bench Hard src/bbh.py;
HotpotQA src/hotpot_task.py (strict, and content as hotpot_task.CONTENT_SCORER_VERSION); content for the other three
src/content_scoring.classify (CONTENT_SCORER_VERSION) with the gold tables of scripts/score_content.py.

Summary per benchmark (results/frontier_panel/nebius_panel.{json,md}, _clean.md), on the units with all three arms
answered. Accuracy per configuration is the mean over its answered items; a benchmark value is the mean over the
sampler's draws (manifests/sampler_<task>.json, a configuration drawn twice counts twice): all 48 draws, and the 24
discovery draws for w and the corrected gain as src/baseline_mixture.RULE_TEXT defines them. Template SD: the standard
deviation over the distinct configurations (denominator C - 1) of the per-configuration value; beside it, the variance
left after the item sampling of a configuration's mean is removed, by the panel's estimator for one model
(src/noise_anatomy.config_variance): every configuration has the same items, so item difficulty is common to all
configuration means and drops out of their spread, and the term removed is the configuration x item interaction mean
square over the number of items. Reported as it comes, negative included.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_panel as npl  # noqa: E402

OUT_DIR = npl.DATA / "outputs"
SCORED_DIR = npl.DATA / "scored"
STEM = "nebius_panel"
TASK_NAME = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
FORCED_CAP = {t: npl.CAPS[t]["forced_direct"] for t in npl.TASKS}


# ----------------------------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------------------------

def answered(rows: list[dict]) -> pd.DataFrame:
    """One answered reply per key: the first that stands for the complete design (run at the design's cap, or ended
    before its own cap; nebius_panel.valid_for_design), else the latest answered one, marked at_design_cap False (a
    pilot reply cut at the panel's cap whose rerun at 7,000 has not come back yet)."""
    ok = [dict(r, at_design_cap=npl.valid_for_design(r)) for r in rows if npl.answered_ok(r)]
    df = pd.DataFrame(ok)
    if df.empty:
        return df
    good = df[df["at_design_cap"]].sort_values("t").drop_duplicates("key", keep="first")
    rest = df[~df["key"].isin(good["key"])].sort_values("t").drop_duplicates("key", keep="last")
    return pd.concat([good, rest]).sort_values("t").reset_index(drop=True)


def superseded(rows: list[dict]) -> pd.DataFrame:
    """Answered replies that ran below the design's cap and may have been cut there (the pilot's reruns)."""
    return pd.DataFrame(npl.rerun_rows(rows) if rows else [])


class TokenCutter:
    """The cut at the panel's cap for a reply that came back as text: the reply re-encoded by
    DeepSeek-V4.1-Flash-1017's tokenizer and kept up to the end of its first N tokens (an exact prefix of the reply).
    The server's own token ids are not returned, so the cut is on the local re-encoding; n_output_tokens_local beside
    the server count shows where the two could differ."""

    def __init__(self, tok=None):
        self._tok = tok

    def tok(self):
        if self._tok is None:
            self._tok = npl.PanelRenderer().r.tok
        return self._tok

    def count(self, text: str) -> int:
        return len(self.tok().encode(text or "", add_special_tokens=False).ids)

    def cut(self, text: str, n: int) -> tuple[str, int]:
        enc = self.tok().encode(text or "", add_special_tokens=False)
        if len(enc.ids) <= n:
            return text, len(enc.ids)
        return text[: enc.offsets[n - 1][1]], len(enc.ids)


def caps_of_row(task: str, arm: str, r_cap, r_pcap) -> tuple[int, int]:
    """(design cap, panel cap) of a reply: the ladder's for the design's arms, else the reply's own cap."""
    try:
        return npl.design_cap(task, arm), npl.panel_cap(task, arm)
    except KeyError:
        c = int(r_cap or 0)
        return c, int(r_pcap or c)


def outputs_frame(df: pd.DataFrame, task: str, cutter: TokenCutter | None = None) -> pd.DataFrame:
    """The shard columns of one benchmark's replies, every arm, with both readings: output_text (uncut) and
    output_text_cut, the copy cut at the panel's cap (finish_reason_cut 'length' and cut_at_panel_cap when the server's
    completion count passed that cap). GSM8K and BIG-Bench Hard forced and five-shot forced replies are cut after their
    first '}' in both readings (the collection's stop string, kept)."""
    x = df[df["task"] == task].copy()
    cutter = cutter or TokenCutter()
    has_pcap = "panel_cap" in x.columns
    has_dc = "at_design_cap" in x.columns
    texts, fins, brace, tcut, fcut, ncut, iscut, pcaps, dcaps, nloc = [], [], [], [], [], [], [], [], [], []
    u = x["usage"].tolist()
    for i, (t, a, f) in enumerate(zip(x["text"], x["arm"], x["finish_reason"])):
        n_out = int(u[i].get("completion_tokens") or 0)
        dc, pc = caps_of_row(task, a, x["cap"].iloc[i] if "cap" in x.columns else None,
                             x["panel_cap"].iloc[i] if has_pcap and pd.notna(x["panel_cap"].iloc[i]) else None)
        if a in npl.FORCED_ARMS and task in npl.CUT_AT_FIRST_BRACE:
            s, c = npl.cut_at_first_brace(t)
            texts.append(s)
            fins.append("stop" if c else f)
            brace.append(c and s != t)
        else:
            s, c = t, False
            texts.append(t)
            fins.append(f)
            brace.append(False)
        if a not in npl.FORCED_ARMS and n_out > pc:
            ct, nl = cutter.cut(s, pc)
            tcut.append(ct)
            fcut.append("length")
            ncut.append(pc)
            iscut.append(True)
            nloc.append(nl)
        else:
            tcut.append(s)
            fcut.append(fins[-1])
            ncut.append(n_out)
            iscut.append(False)
            nloc.append(None)
        pcaps.append(pc)
        dcaps.append(dc)
    return pd.DataFrame({
        "request_id": x["request_id"].to_numpy(), "model_key": npl.MODEL_KEY, "arm": x["arm"].to_numpy(),
        "config_id": x["config_id"].astype(str).to_numpy(), "item_id": x["item_id"].astype(str).to_numpy(),
        "prompt_sha256": x["prompt_sha256"].to_numpy(), "output_text": texts,
        "n_prompt_tokens": [int(v.get("prompt_tokens") or 0) for v in u],
        "n_output_tokens": [int(v.get("completion_tokens") or 0) for v in u],
        "finish_reason": fins, "cached": False,
        "n_cached_prompt_tokens": [int(v.get("cached_tokens") or 0) for v in u],
        "raw_text": x["text"].to_numpy(), "cut_after_brace": brace,
        "cap": [int(c) if pd.notna(c) else None for c in (x["cap"] if "cap" in x.columns else [None] * len(x))],
        "design_cap": dcaps, "panel_cap": pcaps, "output_text_cut": tcut, "finish_reason_cut": fcut,
        "n_output_tokens_cut": ncut, "cut_at_panel_cap": iscut, "n_output_tokens_local": nloc,
        "at_design_cap": (x["at_design_cap"].astype(bool).to_numpy() if has_dc else True)})


def reading_frame(o: pd.DataFrame, reading: str) -> pd.DataFrame:
    """The frame scored for one reading: 'cut' (primary) takes the copy cut at the panel's cap, 'uncut' the reply as
    it came at the design's cap (7,000 on the raised arms)."""
    r = o.copy()
    if reading == "cut":
        r["output_text"] = r["output_text_cut"]
        r["finish_reason"] = r["finish_reason_cut"]
        r["n_output_tokens"] = r["n_output_tokens_cut"]
    return r


def complete_units(o: pd.DataFrame) -> pd.DataFrame:
    """Rows of the panel's three arms on the (configuration, item) units that have all three."""
    o = o[o["arm"].isin(npl.ARMS)]
    n = o.groupby(["config_id", "item_id"])["arm"].nunique()
    keep = n[n == 3].index
    return o.set_index(["config_id", "item_id"]).loc[keep].reset_index()


# ----------------------------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------------------------

def scoring_text(o: pd.DataFrame) -> pd.Series:
    return np.where(o["arm"].isin(npl.FORCED_ARMS), npl.PREFILL + o["output_text"].fillna(""), o["output_text"])


def content_gold(task: str) -> dict:
    """scripts/score_content.gold_table, restated so the import does not run that script's setup."""
    from src import bbh, bbh_d2, content_scoring as cs, gsm8k, math_d2, scoring
    if task == "gsm8k":
        it = gsm8k.load_gsm8k("test")
        return {str(int(i)): (scoring.normalize_number(g), None, None, None)
                for i, g in zip(it["item_id"], it["gold_answer"])}
    if task == "math":
        it, _ = math_d2.load_items("target")
        return {str(i): (g, None, None, None) for i, g in zip(it["item_id"], it["gold_answer"])}
    it, _ = bbh_d2.load_items("target")
    return {str(i): (g, bbh.AnswerType.from_label(a), cs.options(q), k)
            for i, g, a, q, k in zip(it["item_id"], it["gold_answer"], it["answer_type"], it["question"], it["task"])}


def score_task(o: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    from src import bbh, bbh_d2, content_scoring as cs, d2_io, gsm8k, hotpot_d2, hotpot_task, math_d2, math_task
    from src import scoring as sc
    from src import forced_stop
    s = o[["request_id", "item_id"]].copy()
    s["output_text"] = scoring_text(o)
    if task in npl.CUT_AT_FIRST_BRACE:
        # forced replies cut after their first "}" (the collection's stop string), read as the closed box they were
        # meant to be (src/forced_stop.py); "forced_prefilled" is Mistral Large 3's forced arm, prefill in front
        fm = o["arm"].isin(list(npl.FORCED_ARMS) + ["forced_prefilled"]).to_numpy()
        s["output_text"], _ = forced_stop.close_series(s["output_text"].tolist(), fm)
    if task == "gsm8k":
        spec, rec = d2_io.load_registered_scorer(ROOT / "manifests" / "gsm8k_scorer.json")
        g = gsm8k.load_gsm8k("test")
        s["item_id"] = s["item_id"].astype(int)
        r = sc.score_outputs(s, g, spec)
        version = rec.get("scorer_version", sc.SCORER_VERSION)
    elif task == "math":
        items, _ = math_d2.load_items("target")
        r = math_task.score_outputs(s, items)
        version = math_task.SCORER_VERSION
    elif task == "bbh":
        items, _ = bbh_d2.load_items("target")
        r = bbh.score_outputs(s, items)
        version = bbh.SCORER_VERSION
    else:
        items, _ = hotpot_d2.load_items("target")
        r = hotpot_task.score_outputs(s, items)
        version = hotpot_task.SCORER_VERSION
    out = o.drop(columns=["output_text", "raw_text", "output_text_cut"], errors="ignore").copy()
    for c in ("strict_correct", "lenient_correct", "compliant", "strict_f1", "lenient_f1"):
        if c in r.columns:
            out[c] = r[c].to_numpy()
    if task == "hotpot":
        c = hotpot_task.score_content(s.assign(item_id=s["item_id"].astype(str)), items)
        out["content_correct"] = c["content_correct"].to_numpy()
        out["content_f1"] = c["content_f1"].to_numpy()
        out["content_source"] = c["content_source"].to_numpy()
        cver = hotpot_task.CONTENT_SCORER_VERSION
    else:
        gold = content_gold(task)
        res = [cs.classify(t, f, gold[str(i)][0], task, gold[str(i)][1], opts=gold[str(i)][2],
                           task_name=gold[str(i)][3])
               for t, f, i in zip(s["output_text"], o["finish_reason"], o["item_id"])]
        out["content_class"] = [x[0] for x in res]
        out["content_correct"] = (out["content_class"] == "correct").astype("int8")
        cver = f"{cs.CONTENT_SCORER_VERSION} (src/content_scoring.py, classify, as scripts/score_content.py)"
    return out, {"strict": version, "content": cver, "forced_stop": forced_stop.VERSION}


# ----------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------

def sampler(task: str) -> tuple[list, list]:
    s = json.loads(npl.SAMPLERS[task].read_text(encoding="utf-8"))
    return [str(u) for u in s["discovery_config_ids"]], [str(u) for u in s["validation_config_ids"]]


def per_config(sc: pd.DataFrame, score: str) -> pd.DataFrame:
    """config x arm mean of `score` (0-1), plus n per config."""
    return sc.pivot_table(index="config_id", columns="arm", values=score, aggfunc="mean")


def draw_mean(v: pd.Series, draws: list) -> float | None:
    x = v.reindex(draws)
    if x.isna().any():
        return None
    return float(x.mean())


def sd(v) -> float | None:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if len(v) >= 2 else None


TEMPLATE_ESTIMATOR = ("variance over the distinct configurations (denominator C - 1) of the configuration means; the "
                      "item sampling removed by src/noise_anatomy.config_variance, the panel's design_expectation for "
                      "one model: W_shift = variance - W_item / N, W_item the configuration x item interaction mean "
                      "square over the N items every configuration shares")


def template_block(sc: pd.DataFrame, score: str) -> dict:
    """SD over the distinct configurations (pp), and the variance left after the item sampling is removed (pp^2):
    src/noise_anatomy.config_variance on the configuration x item table of each quantity (module doc). The removed
    term is the interaction mean square over N (item_sampling_variance_pp2); without one shared item set (a
    configuration lacks an item another has), the removed term and the difference are None."""
    from src import noise_anatomy as na
    pc = per_config(sc, score) * 100
    out = {}
    series = {"requested": pc["direct_answer"], "forced": pc["forced_direct"], "cot": pc["zero_shot_cot"],
              "gain_over_requested": pc["zero_shot_cot"] - pc["direct_answer"],
              "gain_over_forced": pc["zero_shot_cot"] - pc["forced_direct"]}
    wide = sc.pivot_table(index=["config_id", "item_id"], columns="arm", values=score, aggfunc="first")
    item_vals = {"requested": wide["direct_answer"], "forced": wide["forced_direct"], "cot": wide["zero_shot_cot"],
                 "gain_over_requested": wide["zero_shot_cot"] - wide["direct_answer"],
                 "gain_over_forced": wide["zero_shot_cot"] - wide["forced_direct"]}
    for k, v in series.items():
        s = sd(v)
        table = item_vals[k].unstack("item_id")
        comp = None
        if table.shape[0] >= 2 and table.shape[1] >= 2 and not table.isna().to_numpy().any():
            comp = na.config_variance(table.to_numpy(dtype=float))
        out[k] = {"sd_pp": s, "variance_pp2": None if s is None else s ** 2,
                  "items": None if comp is None else comp["N"],
                  "interaction_mean_square_pp2": None if comp is None else comp["W_item"],
                  "item_sampling_variance_pp2": None if comp is None else comp["W_item"] / comp["N"],
                  "variance_minus_item_sampling_pp2": None if (comp is None or s is None)
                  else s ** 2 - comp["W_item"] / comp["N"]}
    return out


def task_summary(task: str, sc: pd.DataFrame, o_all: pd.DataFrame, raw: list[dict]) -> dict:
    disc, val = sampler(task)
    draws = disc + val
    configs = sorted(set(draws))
    have = sorted(sc["config_id"].unique())
    res = {"task": task, "name": TASK_NAME[task], "configs_design": len(configs), "configs_answered": len(have),
           "configs_not_in_sampler": sorted(set(have) - set(configs)),
           "units_answered": int(sc[["config_id", "item_id"]].drop_duplicates().shape[0]),
           "items_per_config": {"min": int(sc[sc.arm == "direct_answer"].groupby("config_id").size().min()),
                                "max": int(sc[sc.arm == "direct_answer"].groupby("config_id").size().max())},
           "forced_cap": FORCED_CAP[task]}
    scores = {"strict": "strict_correct", "content": "content_correct"}
    for name, col in scores.items():
        pc = per_config(sc, col) * 100
        blk = {}
        for lab, dr in (("all_draws", draws), ("discovery", disc)):
            acc = {npl.ARM_LABEL[a]: draw_mean(pc[a], dr) for a in npl.ARMS}
            g_req = None if None in (acc["cot"], acc["requested"]) else acc["cot"] - acc["requested"]
            g_fd = None if None in (acc["cot"], acc["forced"]) else acc["cot"] - acc["forced"]
            blk[lab] = {"accuracy_pp": acc, "gain_over_requested_pp": g_req, "gain_over_forced_pp": g_fd}
        res[name] = blk
        res[name]["template"] = template_block(sc, col)
    req = sc[sc["arm"] == "direct_answer"]
    long_share = (req["n_output_tokens"] > FORCED_CAP[task]).groupby(req["config_id"]).mean()
    res["w"] = {"discovery": draw_mean(long_share, disc), "all_draws": draw_mean(long_share, draws),
                "rule": "share of requested-direct outputs longer than the forced cap "
                        f"({FORCED_CAP[task]} generated tokens, server count, '>'), per configuration, averaged over "
                        "the draws"}
    for name in scores:
        for lab in ("discovery", "all_draws"):
            w = res["w"][lab]
            g = res[name][lab]["gain_over_forced_pp"]
            res[name][lab]["corrected_gain_pp"] = None if None in (w, g) else (1 - w) * g
    fd = o_all[o_all["arm"] == "forced_direct"]
    passes = [npl.forced_pass(t)["pass"] for t in fd["output_text"]]
    res["forced_pass"] = {"n": len(passes), "pass": int(sum(passes)),
                          "rate": (sum(passes) / len(passes)) if passes else None,
                          "replies_cut_after_first_brace": int(fd["cut_after_brace"].sum()),
                          "rule": "the box opened by the prefill closes (brace depth back to 0) and the reply holds no "
                                  "restart (a new \\boxed, a think tag, a role or turn token); GSM8K and BIG-Bench Hard "
                                  "replies read after the cut at the first '}'"}
    res["truncated"] = {npl.ARM_LABEL[a]: int(((o_all["arm"] == a) & (o_all["finish_reason"] == "length")).sum())
                        for a in npl.ARMS}
    res["output_tokens_mean"] = {npl.ARM_LABEL[a]: float(o_all.loc[o_all["arm"] == a, "n_output_tokens"].mean())
                                 for a in npl.ARMS}
    rr = [r for r in raw if r.get("task") == task]
    ok = [r for r in rr if r.get("status") == 200 and isinstance(r.get("text"), str)]
    ts = sorted(r["t"] for r in rr)
    res["calls"] = {"replies": len(ok), "lines": len(rr), "failed_lines": len(rr) - len(ok),
                    "http_requests": int(sum(r.get("tries") or 1 for r in rr)),
                    "retries": int(sum((r.get("tries") or 1) - 1 for r in rr)),
                    "prompt_tokens": int(sum(r["usage"]["prompt_tokens"] or 0 for r in ok)),
                    "cached_prompt_tokens": int(sum(r["usage"]["cached_tokens"] or 0 for r in ok)),
                    "output_tokens": int(sum(r["usage"]["completion_tokens"] or 0 for r in ok)),
                    "usd": float(sum(r.get("cost_usd") or 0 for r in rr)),
                    "call_seconds": float(sum(r.get("seconds") or 0 for r in rr)),
                    "first": ts[0] if ts else None, "last": ts[-1] if ts else None,
                    "prompt_tokens_server_vs_local_mismatch": int(sum(
                        (r["usage"]["prompt_tokens"] or 0) != r.get("n_prompt_tokens_local") for r in ok))}
    res["calls_by_arm"] = {npl.ARM_LABEL[a]: {
        "replies": sum(1 for r in ok if r["arm"] == a),
        "prompt_tokens": int(sum(r["usage"]["prompt_tokens"] or 0 for r in ok if r["arm"] == a)),
        "cached_prompt_tokens": int(sum(r["usage"]["cached_tokens"] or 0 for r in ok if r["arm"] == a)),
        "output_tokens": int(sum(r["usage"]["completion_tokens"] or 0 for r in ok if r["arm"] == a)),
        "usd": float(sum(r["cost_usd"] for r in ok if r["arm"] == a))}
        for a in npl.COMPLETE_ARMS if a in npl.ARMS or any(r["arm"] == a for r in ok)}
    return res


def fewshot_block(task: str, sc: pd.DataFrame) -> dict:
    """The five-shot arms of one benchmark in one reading: per arm, the answered rows and configurations, accuracy in
    pp (strict and content) as the mean over configurations of the per-configuration accuracy, and for the five-shot
    forced arm (the panel's configurations) also the mean over the sampler's draws; the share of five-shot requested
    outputs longer than the forced cap."""
    disc, val = sampler(task)
    draws = disc + val
    out = {}
    for arm in ("fewshot_forced", "fewshot_requested"):
        g = sc[sc["arm"] == arm]
        if g.empty:
            continue
        d = {"rows": int(len(g)), "configs": int(g["config_id"].nunique())}
        for name, col in (("strict", "strict_correct"), ("content", "content_correct")):
            pc = g.groupby("config_id")[col].mean() * 100
            d[f"{name}_pp_config_mean"] = float(pc.mean())
            if arm == "fewshot_forced":
                d[f"{name}_pp_all_draws"] = draw_mean(pc, draws)
        if arm == "fewshot_requested":
            d["share_longer_than_forced_cap"] = float((g["n_output_tokens"] > FORCED_CAP[task]).mean())
        out[npl.ARM_LABEL[arm]] = d
    return out


def cut_block(o: pd.DataFrame) -> dict:
    """Per arm: replies, those cut at the panel's cap (server count above it), those that reached the design's cap
    (finish 'length' at 7,000), pilot replies still waiting for their rerun, and replies whose local re-encoding is
    shorter than the panel's cap although the server counted more (their cut copy is the whole reply)."""
    out = {}
    for arm, g in o.groupby("arm"):
        at_dc = g["at_design_cap"].astype(bool)
        loc = g["n_output_tokens_local"]
        out[npl.ARM_LABEL.get(arm, arm)] = {
            "replies": int(len(g)), "panel_cap": int(g["panel_cap"].max()), "design_cap": int(g["design_cap"].max()),
            "cut_at_panel_cap": int(g["cut_at_panel_cap"].sum()),
            "share_cut": float(g["cut_at_panel_cap"].mean()),
            "reached_design_cap": int(((g["finish_reason"] == "length") & at_dc
                                       & (g["n_output_tokens"] >= g["design_cap"])).sum()),
            "pending_rerun": int((~at_dc).sum()),
            "local_count_below_panel_cap": int((g["cut_at_panel_cap"] & loc.notna()
                                                & (loc.fillna(0) <= g["panel_cap"])).sum())}
    return out


def rerun_check(rows: list[dict], ans: pd.DataFrame, cutter: TokenCutter | None = None) -> dict:
    """The reruns at 7,000 against the pilot's replies cut at the panel's cap: the
    rerun's copy cut at the panel's cap should equal the pilot's reply (greedy, same prompt)."""
    sup = superseded(rows)
    if sup.empty or ans.empty:
        return {"pilot_replies_cut": int(len(sup)), "rerun_back": 0}
    cutter = cutter or TokenCutter()
    new = ans[ans["at_design_cap"].astype(bool)].set_index("key")
    pairs = []
    for r in sup.itertuples(index=False):
        if r.key not in new.index:
            continue
        b = new.loc[r.key]
        ct, _ = cutter.cut(b["text"], int(r.cap))
        pairs.append({"key": r.key, "same": ct == r.text, "same_prefix": b["text"].startswith(r.text)})
    return {"pilot_replies_cut": int(len(sup)), "rerun_back": len(pairs),
            "cut_copy_equal": int(sum(p["same"] for p in pairs)),
            "rerun_starts_with_pilot_reply": int(sum(p["same_prefix"] for p in pairs)),
            "differing_keys": [p["key"] for p in pairs if not p["same"]][:20]}


# ----------------------------------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------------------------------

def f1(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def usd(x):
    if x is None:
        return "–"
    return f"${x:.4f}" if abs(x) < 0.1 else f"${x:,.2f}"


def write_md(rep: dict, path: Path) -> None:
    pj = rep.get("projection") or {}
    L = [f"# DeepSeek V4-Flash on Nebius at the size ladder's complete design ({rep['date']})", ""]
    L += [f"Model: `{npl.MODEL}`, the id Nebius names as the replacement for V4-Flash. Served as {npl.SERVED_AS}: the "
          "weights are fp4 (NVFP4), so this is a quantized checkpoint. Chat template: `chat_template.jinja` of "
          f"`{npl.MODEL}` at revision `{rep['render']['revision']}` (sha256 `{rep['render']['sha256'][:12]}`), "
          "thinking off (`thinking_mode` 'chat'); every prompt was checked against DeepSeek-V4's chat-mode format.",
          "",
          "Every arm went to `/v1/completions` in real time with the locally rendered prompt; the forced arms' prompts "
          "end with the opened box `\\boxed{` inside the assistant turn, and five-shot prompts carry their five worked "
          "pairs as earlier chat turns. Greedy (temperature 0, top_p 1). Design: the size ladder's (requested, forced, "
          "chain of thought, five-shot forced on all four benchmarks, five-shot requested on GSM8K and BIG-Bench Hard), "
          "from the ladder's request files; its zero-shot rows are the main panel's. Caps: 7,000 on the requested, "
          "chain-of-thought and five-shot requested arms, forced arms 16 (HotpotQA 64). Two readings: " + rep["readings"]
          + ". The 5% pilot ran at the panel's caps; its replies that may have been cut there were sent again at 7,000. "
          "GSM8K and BIG-Bench Hard forced and five-shot forced replies are cut after their first `}` (the "
          "collection's stop string).", ""]
    L += ["## What ran", ""]
    L += [f"- Design: {rep['design']['calls']:,} calls ({rep['design']['units']:,} units of benchmark, configuration "
          "and item, three arms each)."]
    if pj:
        L += [f"- Pilot: {pj['pilot']['calls']:,} calls ({pj['pilot']['share_of_calls']:.2%}), {pj['pilot']['rule']}; "
              f"{pj['pilot']['replies']:,} answered."]
    L += [f"- Answered so far: {rep['answered']:,} calls ({rep['answered'] / rep['design']['calls']:.2%} of the design)."
          f" Spend: {usd(rep['spend']['total_usd'])} in total today (live test {usd(rep['spend']['live_test_usd'])}, "
          f"this run {usd(rep['spend']['panel_usd'])}); limit {usd(rep['spend']['limit_usd'])}.", ""]
    rp = rep.get("realtime_projection") or {}
    if rp:
        L += [f"- Real-time projection for the complete design ({rp['written']}): {rp['to_send']:,} calls to send "
              f"({rp['reruns']} reruns), {usd(rp['projected_usd'])} (high {usd(rp['projected_usd_high'])}, no cache "
              f"{usd(rp['projected_usd_nocache'])}); with {usd(rp['spent_usd'])} spent before it, "
              f"{usd(rp['total_usd'])} against the {usd(rp['limit_usd'])} limit. Rule: {rp['rule']}", "",
              "| Benchmark | Arm | Calls | Prompt tok. | Cached share | Output tok. | Projected | No cache | High |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in rp["cells"]:
            L.append(f"| {TASK_NAME[c['task']]} | {npl.ARM_LABEL[c['arm']]} | {c['calls']:,} | "
                     f"{c['prompt_tokens_mean']:.0f} | {c['cached_share']:.2f} | {c['output_tokens_mean']:.0f} | "
                     f"{usd(c['usd'])} | {usd(c['usd_nocache'])} | {usd(c['usd_high'])} |")
        L += [""]
    if pj and pj.get("projected_total_usd") is not None and not rp:
        pilot_s = sum(r.get("seconds", 0) for r in rep.get("runs", []) if r.get("phase") == "pilot")
        rem = pj["projected_total_usd"] - rep["spend"]["total_usd"]
        hours = pilot_s / max(pj["pilot"]["calls"], 1) * (rep["design"]["calls"] - rep["answered"]) / 3600
        L += [f"- Left to run: {rep['design']['calls'] - rep['answered']:,} calls, about {usd(rem)} at the pilot's "
              f"rates and about {hours:.1f} hours of wall time at the pilot's pace (16 threads, no retries needed).", ""]
    if pj:
        L += ["## The pilot's projection for the panel design ($10 limit)", "",
              "Mean billed cost of a pilot call per benchmark and arm (cached input at $0.006 per M where the server "
              "reported cached tokens), times the design's calls.", "",
              "| Benchmark | Arm | Design calls | Pilot calls | Prompt tok. | Cached tok. | Output tok. | $ per 1k calls "
              "| Projected |", "|---|---|---|---|---|---|---|---|---|"]
        for c in pj["cells"]:
            if c.get("usd_per_call") is None:
                L.append(f"| {TASK_NAME[c['task']]} | {npl.ARM_LABEL[c['arm']]} | {c['calls_design']:,} | 0 | – | – "
                         "| – | – | – |")
                continue
            L.append(f"| {TASK_NAME[c['task']]} | {npl.ARM_LABEL[c['arm']]} | {c['calls_design']:,} | "
                     f"{c['pilot_calls']:,} | {c['prompt_tokens_mean']:.0f} | {c['cached_tokens_mean']:.0f} | "
                     f"{c['completion_tokens_mean']:.0f} | {1000 * c['usd_per_call']:.3f} | "
                     f"{usd(c['projected_usd'])} |")
        by_task = {}
        for c in pj["cells"]:
            by_task[c["task"]] = by_task.get(c["task"], 0) + (c.get("projected_usd") or 0)
        L += ["", "Per benchmark: " + "; ".join(f"{TASK_NAME[t]} {usd(v)}" for t, v in by_task.items()) + ".", "",
              f"Projected total, live test included: **{usd(pj['projected_total_usd'])}** against the limit of "
              f"{usd(pj['limit_usd'])}. Forced pass rate in the pilot: "
              + "; ".join(f"{TASK_NAME[t]} {f1(None if v['rate'] is None else 100 * v['rate'])}% ({v['pass']}/{v['n']})"
                          for t, v in pj["forced_pass"].items())
              + f" (rule: at least {pj['pass_min']:.0%} on every benchmark). Decision: "
              + ("continue to the full run." if pj["go"] else
                 "**stop after the pilot**; the rest needs a new limit" +
                 (" (projected cost above the limit)" if not pj["fits"] else "") +
                 (" (forced pass rate below 0.9)" if not pj["pass_ok"] else "") + "."), ""]
    cov = "the pilot's units" if rep["answered"] < rep["design"]["calls"] else "the whole design"
    L += [f"## Main numbers ({cov})", "",
          "Accuracy in pp, mean over the sampler's 48 draws (a configuration drawn twice counts twice) of the "
          "per-configuration accuracy on the answered items. w and the corrected gain use the 24 discovery draws "
          "(src/baseline_mixture.RULE_TEXT): w is the share of requested outputs longer than the forced cap, and the "
          "corrected gain is (1 − w) × the gain over the forced arm.", ""]
    if rep["answered"] < rep["design"]["calls"]:
        L += [f"These are pilot values: about {100 * rep['design_share_answered']:.0f}% of the items of each "
              "configuration, so each configuration's accuracy carries an item-sampling error of several pp.", ""]
    L += ["| Benchmark | Score | Requested | Forced | CoT | CoT − requested | CoT − forced | w (disc.) | Corrected gain "
          "(disc.) | CoT − requested (disc.) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, s in rep["tasks"].items():
        for sc_ in ("strict", "content"):
            a, d = s[sc_]["all_draws"], s[sc_]["discovery"]
            L.append(f"| {TASK_NAME[t]} | {sc_} | {f1(a['accuracy_pp']['requested'])} | {f1(a['accuracy_pp']['forced'])}"
                     f" | {f1(a['accuracy_pp']['cot'])} | {f1(a['gain_over_requested_pp'])} | "
                     f"{f1(a['gain_over_forced_pp'])} | {f1(s['w']['discovery'], 3)} | {f1(d['corrected_gain_pp'])} | "
                     f"{f1(d['gain_over_requested_pp'])} |")
    if rep.get("tasks_uncut"):
        L += ["", "Uncut reading (the replies as they came at 7,000):", "",
              "| Benchmark | Score | Requested | Forced | CoT | CoT − requested | CoT − forced | w (disc.) | "
              "Corrected gain (disc.) | CoT − requested (disc.) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, s in rep["tasks_uncut"].items():
            for sc_ in ("strict", "content"):
                a, d = s[sc_]["all_draws"], s[sc_]["discovery"]
                L.append(f"| {TASK_NAME[t]} | {sc_} | {f1(a['accuracy_pp']['requested'])} | "
                         f"{f1(a['accuracy_pp']['forced'])} | {f1(a['accuracy_pp']['cot'])} | "
                         f"{f1(a['gain_over_requested_pp'])} | {f1(a['gain_over_forced_pp'])} | "
                         f"{f1(s['w']['discovery'], 3)} | {f1(d['corrected_gain_pp'])} | "
                         f"{f1(d['gain_over_requested_pp'])} |")
    if any(v.get("cut") for v in rep.get("fewshot", {}).values()):
        L += ["", "Five-shot arms (pp; mean over configurations of the per-configuration accuracy; five-shot forced "
              "also as the mean over the sampler's 48 draws):", "",
              "| Benchmark | Arm | Reading | Rows | C | Strict | Content | Strict (draws) | Content (draws) | Share "
              "longer than forced cap |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, v in rep["fewshot"].items():
            for reading in ("cut", "uncut"):
                for arm, d in (v.get(reading) or {}).items():
                    L.append(f"| {TASK_NAME[t]} | {arm} | {reading} | {d['rows']:,} | {d['configs']} | "
                             f"{f1(d['strict_pp_config_mean'])} | {f1(d['content_pp_config_mean'])} | "
                             f"{f1(d.get('strict_pp_all_draws'))} | {f1(d.get('content_pp_all_draws'))} | "
                             f"{f1(None if d.get('share_longer_than_forced_cap') is None else 100 * d['share_longer_than_forced_cap'])} |")
    if rep.get("cuts"):
        L += ["", "Cut at the panel's cap, per arm (replies whose server count passed the panel's cap; reached the "
              "design's cap: finish 'length' at 7,000; pending: pilot replies cut at the panel's cap whose rerun has "
              "not come back):", "",
              "| Benchmark | Arm | Replies | Panel cap | Design cap | Cut at panel cap | Share | Reached design cap | "
              "Pending rerun |", "|---|---|---|---|---|---|---|---|---|"]
        for t, v in rep["cuts"].items():
            for arm, d in v.items():
                L.append(f"| {TASK_NAME[t]} | {arm} | {d['replies']:,} | {d['panel_cap']:,} | {d['design_cap']:,} | "
                         f"{d['cut_at_panel_cap']:,} | {100 * d['share_cut']:.2f}% | {d['reached_design_cap']} | "
                         f"{d['pending_rerun']} |")
    rc = rep.get("rerun_check") or {}
    if rc:
        L += ["", f"Reruns: {rc.get('pilot_replies_cut', 0)} pilot replies may have been cut at the panel's cap; "
              f"{rc.get('rerun_back', 0)} came back at 7,000, of which {rc.get('cut_copy_equal', 0)} give the pilot's "
              f"reply exactly when cut at the panel's cap and {rc.get('rerun_starts_with_pilot_reply', 0)} begin with "
              "it."]
    L += ["", "Template SD (pp; standard deviation over the distinct configurations, denominator C − 1) and, in "
          "brackets, the variance left after the item-sampling variance of a configuration mean (pp², as it comes; "
          "negative means the spread is within item-sampling noise). Every configuration has the same items, so the "
          "item-sampling variance is the configuration × item interaction mean square over the number of items "
          "(`src/noise_anatomy.config_variance`).", "",
          "| Benchmark | Score | C | Requested | Forced | CoT | CoT − requested | CoT − forced |",
          "|---|---|---|---|---|---|---|---|"]
    for t, s in rep["tasks"].items():
        for sc_ in ("strict", "content"):
            tb = s[sc_]["template"]
            cells = [f"{f1(tb[k]['sd_pp'])} [{f1(tb[k]['variance_minus_item_sampling_pp2'])}]"
                     for k in ("requested", "forced", "cot", "gain_over_requested", "gain_over_forced")]
            L.append(f"| {TASK_NAME[t]} | {sc_} | {s['configs_answered']} | " + " | ".join(cells) + " |")
    L += ["", "## Forced arm, truncation, calls and spend", "",
          "| Benchmark | Forced pass | Truncated (req./forced/CoT) | Mean output tok. (req./forced/CoT) | Replies | "
          "Retries | Failed | Prompt tok. | Cached tok. | Output tok. | Call time (s) | Spend |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, s in rep["tasks"].items():
        fp, tr, ot, c = s["forced_pass"], s["truncated"], s["output_tokens_mean"], s["calls"]
        L.append(f"| {TASK_NAME[t]} | {f1(None if fp['rate'] is None else 100 * fp['rate'])}% ({fp['pass']}/{fp['n']}) "
                 f"| {tr['requested']}/{tr['forced']}/{tr['cot']} | {f1(ot['requested'])}/{f1(ot['forced'])}/"
                 f"{f1(ot['cot'])} | {c['replies']:,} | {c['retries']} | {c['failed_lines']} | {c['prompt_tokens']:,} | "
                 f"{c['cached_prompt_tokens']:,} | {c['output_tokens']:,} | {c['call_seconds']:,.0f} | {usd(c['usd'])} |")
    L += ["", f"Wall time of the runs: {rep['runtime']}.", ""]
    if rep.get("edge_cases"):
        L += ["## Edge cases", ""] + [f"- {e}" for e in rep["edge_cases"]] + [""]
    L += ["## Files", "",
          f"- Raw replies: `{npl.rel(npl.RAW)}` (one line per reply; no key, no headers).",
          f"- Outputs (shard columns): `{npl.rel(OUT_DIR)}/<task>.parquet`; scored rows: `{npl.rel(SCORED_DIR)}/"
          "<task>_scored.parquet`.",
          f"- Projections: `{npl.rel(npl.PROJECTION)}` (pilot, panel design), `{npl.rel(npl.REALTIME_PROJECTION)}` "
          "(real time, complete design). Scripts: `scripts/nebius_panel.py`, `scripts/nebius_panel_report.py`; tests "
          "`tests/test_nebius_panel.py`, `tests/test_nebius_realtime.py`.",
          f"- Scored rows, uncut reading: `{npl.rel(SCORED_DIR)}/<task>_scored_uncut.parquet`.",
          f"- Scorers: " + "; ".join(f"{TASK_NAME[t]} {v['strict']} and {v['content']}"
                                     for t, v in rep["scorers"].items()) + "."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    raw = npl.read_raw(npl.RAW)
    df_req = npl.load_complete()
    design = {"calls": int(len(df_req)), "units": int(len(npl.units_of(df_req[df_req["arm"].isin(npl.ARMS)]))),
              "calls_by_arm": {f"{t}/{a}": int(n) for (t, a), n in df_req.groupby(["task", "arm"]).size().items()}}
    ans = answered(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    live = npl.live_test_spend(npl.LIVE_TEST_RAW)
    rep = {"date": time.strftime("%Y-%m-%d"), "model": npl.MODEL, "served_as": npl.SERVED_AS, "model_key": npl.MODEL_KEY,
           "render": npl.PanelRenderer().record(), "price": npl.PRICE, "caps": npl.CAPS, "cap_sources": npl.CAP_SOURCES,
           "design": design, "answered": int(len(ans)), "tasks": {}, "tasks_uncut": {}, "fewshot": {}, "cuts": {},
           "scorers": {}, "edge_cases": [],
           "readings": "cut (primary): each reply of a raised arm cut at the panel's cap of its arm (requested 1,024, "
                       "chain of thought 3,072, HotpotQA chain of thought 1,900, five-shot requested 1,024) on the "
                       "local re-encoding; uncut: the reply as it came at the design's cap (7,000)"}
    rep["design_caps"] = {t: {"design": npl.ladder_caps(t)[0], "panel": npl.ladder_caps(t)[1]} for t in npl.TASKS}
    rep["design_share_answered"] = len(ans) / design["calls"]
    for t in npl.TASKS:
        o = outputs_frame(ans, t)
        if o.empty:
            rep["edge_cases"].append(f"{TASK_NAME[t]}: no answered call")
            continue
        o.drop(columns=["raw_text"]).to_parquet(OUT_DIR / f"{t}.parquet", index=False)
        rep["cuts"][t] = cut_block(o)
        rep["fewshot"][t] = {}
        for reading in ("cut", "uncut"):
            r = reading_frame(o, reading)
            sc, ver = score_task(r, t)
            sc.to_parquet(SCORED_DIR / (f"{t}_scored.parquet" if reading == "cut" else f"{t}_scored_uncut.parquet"),
                          index=False)
            rep["scorers"][t] = ver
            rep["fewshot"][t][reading] = fewshot_block(t, sc)
            cu = complete_units(sc)
            if reading == "uncut":
                rep["tasks_uncut"][t] = task_summary(t, cu, r, raw)
                continue
            n_zero = int(sc["arm"].isin(npl.ARMS).sum())
            dropped = n_zero - len(cu)
            if dropped:
                rep["edge_cases"].append(f"{TASK_NAME[t]}: {dropped} answered zero-shot calls left out of the summary "
                                         "because their unit lacks an arm")
            rep["tasks"][t] = task_summary(t, cu, r, raw)
        s = rep["tasks"][t]
        if s["configs_answered"] < s["configs_design"]:
            rep["edge_cases"].append(f"{TASK_NAME[t]}: {s['configs_answered']} of {s['configs_design']} configurations "
                                     "answered; draw means over a missing configuration are reported as absent")
        if s["calls"]["prompt_tokens_server_vs_local_mismatch"]:
            rep["edge_cases"].append(f"{TASK_NAME[t]}: {s['calls']['prompt_tokens_server_vs_local_mismatch']} replies "
                                     "whose server prompt count differs from the local count")
        for a, n in s["truncated"].items():
            if n:
                rep["edge_cases"].append(f"{TASK_NAME[t]}: {n} {a} outputs reached their cap (scored as they are)")
        if s["forced_pass"]["n"] and s["forced_pass"]["pass"] < s["forced_pass"]["n"]:
            rep["edge_cases"].append(f"{TASK_NAME[t]}: {s['forced_pass']['n'] - s['forced_pass']['pass']} forced "
                                     "replies fail the pass rule (scored as they are)")
    rep["rerun_check"] = rerun_check(raw, ans)
    for t in npl.TASKS:
        for a in npl.COMPLETE_ARMS:
            e = [r for r in raw if r.get("task") == t and r.get("arm") == a and r.get("text") == ""]
            if e:
                fr = pd.Series([str(r.get("finish_reason")) for r in e]).value_counts().to_dict()
                rep["edge_cases"].append(f"{TASK_NAME[t]}: {len(e)} empty {npl.ARM_LABEL[a]} replies (finish reason "
                                         f"{fr}; the model ended its turn at once), scored as they are")
        s = rep["tasks"].get(t)
        if s and all(s["strict"][k] == s["content"][k] for k in ("all_draws", "discovery")):
            rep["edge_cases"].append(f"{TASK_NAME[t]}: strict and content scores are identical on every summary value "
                                     "(the model boxed a clean answer in every scored reply the two readings could "
                                     "differ on)")
    if rep["tasks"]:
        rep["edge_cases"].append("forced pass rate: the projection (and the stop rule) read the uncut reply; the "
                                 "table above reads GSM8K and BIG-Bench Hard replies after the cut at the first '}', "
                                 "where an answer such as '2:00\\text{ pm}}' loses its closing brace exactly as under "
                                 "the collection's stop string")
    fails = [r for r in raw if not (r.get("status") == 200 and isinstance(r.get("text"), str))]
    if fails:
        st = pd.Series([str(r.get("status")) for r in fails]).value_counts().to_dict()
        rep["edge_cases"].append(f"failed lines (no reply after retries): {len(fails)}, by status {st}")
    runs = [json.loads(x) for x in (npl.DATA / "nebius_panel_runs.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()] if (npl.DATA / "nebius_panel_runs.jsonl").exists() else []
    rep["runs"] = runs
    rep["runtime"] = "; ".join(f"{r['phase']} run ending {r['end']}: {r.get('seconds', 0):,.0f} s, "
                               f"{r.get('replies', 0):,} replies, stop: {r.get('stop_reason') or 'none (finished)'}"
                               for r in runs) or "–"
    panel_usd = npl.spent_of(raw)
    rep["spend"] = {"live_test_usd": live["usd"], "panel_usd": panel_usd, "total_usd": live["usd"] + panel_usd,
                    "limit_usd": npl.COMPLETE_LIMIT_USD}
    rep["projection"] = json.loads(npl.PROJECTION.read_text(encoding="utf-8")) if npl.PROJECTION.exists() else None
    rep["realtime_projection"] = (json.loads(npl.REALTIME_PROJECTION.read_text(encoding="utf-8"))
                                  if npl.REALTIME_PROJECTION.exists() else None)
    if rep["realtime_projection"]:
        rep["realtime_projection"].pop("check_against_panel", None)
    npl.RES.mkdir(parents=True, exist_ok=True)
    (npl.RES / f"{STEM}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = npl.RES / f"{STEM}.md"
    write_md(rep, md)
    (npl.RES / f"{STEM}_clean.md").write_text(npl.npt.strip_links(md.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {npl.rel(md)}; answered {len(ans)} of {design['calls']}; spend ${rep['spend']['total_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
