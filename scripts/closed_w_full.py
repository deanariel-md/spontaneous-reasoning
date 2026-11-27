r"""Qwen4-Max on all 398 numeric MATH-500 items under MATH-500's first 8 discovery draws, the panel's three arms
(requested, forced, chain of thought). Hard limit $20.00 for the whole task, pilot included. A sibling of scripts/closed_w.py (the 100-item run),
whose request, HTTP, money, scoring and check code it calls unchanged; the 100-item run's files are read only.

Design (config/closed_w_full_math.yaml; every request setting is config/closed_w.yaml's, its configuration hash with it):
  Items      the numeric MATH-500 target set (a gold answer that normalizes to one rational
             number; 398 of 500): data/processed/math/items_target.parquet, its sha256 and count checked against
             manifests/math_items.json; every item.
  Templates  the first 8 of manifests/sampler_math.json's discovery_config_ids, whose nested order reads "budget K uses
             the first K discovery units in the order listed" (K = 8 of src/sampling.BUDGET_GRID); 8 distinct templates.
  Arms       as scripts/closed_w.py sends them (make_request): requested (the panel's user message alone, cap 1,024),
             forced (the same message and the assistant message "\boxed{" with partial: true, cap 16, no stop on
             MATH-500), chain of thought (the panel's zero-shot request, cap 3,072); Qwen4-Max on OpenRouter, provider
             pinned to Alibaba, reasoning off, temperature 0, top_p 1, presence penalty 0. 3,184 calls per arm, 9,552.
  Order      units (template, item) in sha256 order of ('closed_w_full/run', 'math', template, item), the three arms
             of a unit back to back, so a run stopped early still covers the cells evenly.
  Raw        data/frontier_panel/closed_w_full_math/{pilot,run}_raw.jsonl, keyed model|arm|math|request id (|rep<k> in
             the pilot); a restart skips keys already answered under the same configuration hash (no re-billing).
             Alibaba's content-inspection 403 (data_inspection_failed) is a provider block: no output, counted, not
             retried, left out of w and accuracy (scripts/closed_w.parse_reply).

Money: the pilot and the run count everything this design has spent (pilot and run raw files) against --limit-usd,
which may not exceed limits.total_usd ($20.00); a chunk of calls is sent only when its worst case fits under the limit
(scripts/closed_w.Runner). `plan` projects the cost from the 100-item run's billed cost per call on MATH-500 per arm;
above limits.projection_stop_usd ($18) the paid phases refuse. The key is read inside Python and never
printed, logged or written.

Phases (repository root):
    python scripts/closed_w_full.py plan                       # offline dry run
    python scripts/closed_w_full.py pilot --approved --limit-usd 20
    python scripts/closed_w_full.py run --approved --limit-usd 20
    python scripts/closed_w_full.py report
    python scripts/closed_w_full.py budget                     # one entry in config/budget.yaml

Outputs: manifests/closed_w_full_math_items.json; results/frontier_panel/closed_w_full_math{_plan,_pilot,}.{json,md},
the _clean.md copies and closed_w_full_math_templates.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import closed_w as cw  # noqa: E402
import closed_w_items as cwp  # noqa: E402

FULL_CONFIG = ROOT / "config" / "closed_w_full_math.yaml"
TASK = "math"
PHASES = ("pilot", "run")
QUANTS = tuple(cw.QUANTS) + ("gain_diff",)
Q_LABEL = {**cw.Q_LABEL, "cot_minus_requested": "Requested gain (CoT − requested)",
           "cot_minus_forced": "Forced gain (CoT − forced)", "corrected_gain": "Corrected gain (1 − w)(CoT − forced)",
           "gain_diff": "Requested gain − forced gain", "requested": "Requested accuracy", "forced": "Forced accuracy",
           "cot": "CoT accuracy"}
BASIS = "discovery8"
RUN_TITLE = ("Qwen4-Max three arms on all 398 numeric MATH-500 items, first 8 discovery draws, on OpenRouter "
             "(scripts/closed_w_full.py pilot and run; "
             "limit $20 for the task)")


# ----------------------------------------------------------------------------------------------------------------
# Configuration and paths
# ----------------------------------------------------------------------------------------------------------------

def load_full(path: Path = FULL_CONFIG) -> dict:
    f = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if f.get("task") != TASK:
        raise SystemExit(f"{cw.rel(path)}: task must be {TASK!r}")
    if int(f["templates"]["k"]) < 1:
        raise SystemExit(f"{cw.rel(path)}: templates.k must be positive")
    lim = f["limits"]
    if not 0 < float(lim["projection_stop_usd"]) <= float(lim["total_usd"]):
        raise SystemExit(f"{cw.rel(path)}: projection_stop_usd must lie in (0, total_usd]")
    return f


def base_config(fcfg: dict, root: Path = ROOT) -> dict:
    return cw.load_config(Path(root) / fcfg["base_config"])


class FPaths:
    """Where this design reads and writes; every written path differs from the 100-item run's."""

    def __init__(self, fcfg: dict, root: Path = ROOT, **over):
        p = fcfg["paths"]
        self.root = Path(root)

        def pick(name, default):
            return Path(over[name]) if over.get(name) is not None else self.root / default
        self.manifest = pick("manifest", p["manifest"])
        self.raw_dir = pick("raw_dir", p["raw_dir"])
        self.results = pick("results", p["results_dir"])
        self.prefix = str(over.get("prefix") or p["prefix"])
        self.budget = pick("budget", p["budget"])
        self.env = pick("env", ".env")
        self.earlier_raw = pick("earlier_raw", p["earlier_raw"])
        self.earlier_report = pick("earlier_report", p["earlier_report"])
        self.earlier_manifest = pick("earlier_manifest", p["earlier_manifest"])
        self.items_table = pick("items_table", fcfg["items"]["table"])
        self.items_record = pick("items_record", fcfg["items"]["record"])
        self.sampler = pick("sampler", fcfg["templates"]["sampler"])
        self.prefill_result = pick("prefill_result", "results/frontier_panel/qwen_prefill_test.json")

    def raw(self, phase: str) -> Path:
        return self.raw_dir / f"{phase}_raw.jsonl"

    def out(self, suffix: str) -> Path:
        return self.results / f"{self.prefix}{suffix}"

    @property
    def sessions(self) -> Path:
        return self.raw_dir / "sessions.jsonl"


def check_separate(paths: FPaths, cfg: dict, root: Path = ROOT) -> None:
    """Refuses a layout that would write over the 100-item run's manifest, raw files or reports."""
    p = cfg["paths"]
    old_manifest = (Path(root) / p["manifest"]).resolve()
    old_raw = (Path(root) / p["raw_dir"]).resolve()
    if paths.manifest.resolve() == old_manifest:
        raise SystemExit("the manifest path is the 100-item run's")
    if paths.raw_dir.resolve() == old_raw:
        raise SystemExit("the raw directory is the 100-item run's")
    if paths.prefix in ("closed_w", "closed_w_pilot", "qwen_prefill_test", "closed_w_price") or \
            not paths.prefix.startswith("closed_w_full"):
        raise SystemExit(f"result prefix {paths.prefix!r} would collide with the 100-item run's files")


# ----------------------------------------------------------------------------------------------------------------
# Items, templates, requests, manifest
# ----------------------------------------------------------------------------------------------------------------

def target_items(table: Path, record: Path) -> dict:
    """The numeric MATH-500 target items, in the table's order, the table checked against its record."""
    rec = json.loads(Path(record).read_text(encoding="utf-8"))
    want = rec["tables"]["target"]
    got = cw.sha256_file(table)
    if got != want["sha256"]:
        raise SystemExit(f"{cw.rel(table)}: sha256 {got[:12]} differs from {cw.rel(record)} ({want['sha256'][:12]})")
    ids = pd.read_parquet(table, columns=["item_id"])["item_id"].astype(str).tolist()
    n = int(rec["counts"]["target_numeric"])
    if len(ids) != n or len(set(ids)) != n:
        raise SystemExit(f"{cw.rel(table)}: {len(ids)} rows ({len(set(ids))} distinct), the record says {n}")
    return {"items": ids, "n": n, "table": cw.rel(table), "table_sha256": got, "record": cw.rel(record),
            "numeric_rule": rec.get("numeric_rule"), "target_all": rec["counts"].get("target_all")}


def discovery_templates(sampler: Path, k: int) -> dict:
    """The first k discovery draws of a sampler (the nested order of src/sampling); a repeated template refused."""
    from src.sampling import BUDGET_GRID
    s = json.loads(Path(sampler).read_text(encoding="utf-8"))
    disc = [str(x) for x in s["discovery_config_ids"]]
    if k > len(disc):
        raise SystemExit(f"{cw.rel(sampler)} has {len(disc)} discovery draws, fewer than {k}")
    first = disc[:k]
    if len(set(first)) != k:
        raise SystemExit(f"the first {k} discovery draws of {cw.rel(sampler)} repeat a template: {first}")
    return {"templates": first, "k": k, "in_budget_grid": k in BUDGET_GRID, "budget_grid": list(BUDGET_GRID),
            "nested_order": s.get("nested_order"), "k_discovery": int(s["k_discovery"]),
            "distinct_discovery_templates": len(set(disc)), "sampler": cw.rel(sampler),
            "sampler_sha256_field": s.get("sha256"), "sampler_file_sha256": cw.sha256_file(sampler)}


def select(df: pd.DataFrame, items, templates) -> pd.DataFrame:
    return df[df["item_id"].isin(set(items)) & df["config_id"].isin(set(templates))].reset_index(drop=True)


def grid_check(sel: pd.DataFrame, items, templates, arm: str) -> None:
    n = len(items) * len(templates)
    if len(sel) != n or sel[["config_id", "item_id"]].duplicated().any():
        raise SystemExit(f"{arm}: {len(sel)} rows on {len(templates)} templates x {len(items)} items, expected {n} "
                         "(each pair once)")


def pilot_cells(sel_req: pd.DataFrame, templates, k: int, label: str = "closed_w_full/pilot") -> list[dict]:
    """k cells per template, in sha256 order of (label, 'math', template, item), no item twice."""
    d = sel_req.assign(h=[cw.hkey(label, TASK, c, i) for c, i in zip(sel_req["config_id"], sel_req["item_id"])])
    d = d.sort_values("h", kind="stable")
    taken, out = set(), []
    for c in templates:
        n = 0
        for r in d[d["config_id"] == c].itertuples():
            if r.item_id in taken:
                continue
            out.append({"config_id": c, "item_id": r.item_id})
            taken.add(r.item_id)
            n += 1
            if n == k:
                break
        if n < k:
            raise SystemExit(f"template {c}: only {n} pilot cells with new items")
    return out


def load_rows(cfg: dict, root: Path = ROOT, request_ids: dict | None = None) -> dict:
    """{arm: MATH-500 rows of the arm's request files} (scripts/closed_w.load_arm_requests)."""
    return {a: cw.load_arm_requests(cfg, a, TASK, root, None if request_ids is None else request_ids[a])
            for a in cw.run_arms(cfg)}


def build_manifest(cfg: dict, fcfg: dict, rows_by_arm: dict, items: dict, tinfo: dict, control: dict,
                   root: Path = ROOT, made_on: str | None = None) -> dict:
    templates, ids = tinfo["templates"], items["items"]
    sel = {a: select(df, ids, templates) for a, df in rows_by_arm.items()}
    arms = {}
    for a, s in sel.items():
        grid_check(s, ids, templates, a)
        arms[a] = {"source_arm": cw.ARM_SOURCE[a], "n_requests": int(len(s)), "requests_sha256": cw.list_digest(s),
                   "sources": [{"path": cw.rel(p), "sha256": cw.sha256_file(p)}
                               for p in cw.arm_request_paths(cfg, a, TASK, root)]}
    if "forced" in sel and "requested" in sel:
        j = sel["forced"].merge(sel["requested"][["config_id", "item_id", "text"]], on=["config_id", "item_id"],
                                suffixes=("", "_req"))
        diff = int((j["text"] != j["text_req"]).sum())
        if len(j) != len(sel["forced"]) or diff:
            raise SystemExit(f"forced: {diff} user messages differ from the requested arm's")
        arms["forced"]["user_message_equals_requested"] = True
    k = int(fcfg["pilot"]["cells_per_template"])
    cells = pilot_cells(sel["requested"], templates, k)
    prow = {}
    for a, s in sel.items():
        ix = s.set_index(["config_id", "item_id"])
        prow[a] = [{"request_id": ix.loc[(c["config_id"], c["item_id"]), "request_id"], **c,
                    "n_bytes": len(ix.loc[(c["config_id"], c["item_id"]), "text"].encode("utf-8")),
                    "text_sha256": cw.sha256_text(ix.loc[(c["config_id"], c["item_id"]), "text"])} for c in cells]
    n_arm = len(ids) * len(templates)
    return {
        "name": "closed_w_full_math_items", "made_on": made_on or cw.today(), "script": "scripts/closed_w_full.py plan",
        "request": "all numeric MATH-500 items, first 8 discovery templates",
        "base_config": fcfg["base_config"], "config_sha": cw.config_sha(cfg), "model": fcfg["model"], "task": TASK,
        "item_rule": "every item of the numeric MATH-500 target set (the gold answer normalizes to "
                     "one rational number), in the table's order",
        "item_source": {k_: v for k_, v in items.items() if k_ != "items"}, "items": ids, "n_items": len(ids),
        "template_rule": "the first 8 of manifests/sampler_math.json's discovery_config_ids (nested order: budget K "
                         "uses the first K discovery units in the order listed; K = 8 of src/sampling.BUDGET_GRID)",
        "template_source": {k_: v for k_, v in tinfo.items() if k_ != "templates"}, "templates": templates,
        "n_templates": len(templates), "arms": arms, "n_requests_per_arm": n_arm,
        "n_requests": n_arm * len(arms),
        "run_order": "units (template, item) in sha256 order of ('closed_w_full/run', 'math', template, item), the "
                     "arms of a unit back to back",
        "pilot": {"rule": f"{k} cells per template in sha256 order of ('closed_w_full/pilot', 'math', template, item), "
                          "no item twice; each cell in the three arms, sent "
                          f"{fcfg['pilot']['repeats']} times; and {fcfg['pilot']['controls']} thinking-on control calls",
                  "cells": cells, "rows": prow},
        "control": {"rule": "the MATH Level 5 calibration item (data/processed/math/items_screening.parquet) under a "
                            "run template, first in sha256 order of ('closed_w/control', 'math', template, item) "
                            "(scripts/closed_w.pick_control, the 100-item pilot's rule); its direct_answer prompt is "
                            "sent with thinking on", "row": control},
    }


def manifest_body(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "made_on"}


def load_manifest(paths: FPaths, cfg: dict) -> dict:
    if not paths.manifest.exists():
        raise SystemExit(f"{cw.rel(paths.manifest)} is absent: run `scripts/closed_w_full.py plan` first")
    man = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if man.get("config_sha") != cw.config_sha(cfg):
        raise SystemExit("config/closed_w.yaml changed after `plan` (configuration hash differs from the manifest's)")
    return man


def selected(man: dict, rows_by_arm: dict, verify: bool = True) -> dict:
    out = {}
    for a, df in rows_by_arm.items():
        s = select(df, man["items"], man["templates"])
        if verify and (len(s) != man["arms"][a]["n_requests"] or cw.list_digest(s) != man["arms"][a]["requests_sha256"]):
            raise SystemExit(f"{a}: the request list differs from the manifest ({len(s)} rows)")
        out[a] = s
    return out


def run_requests(cfg: dict, sel: dict, models: list[dict]) -> list[dict]:
    arms = [a for a in cw.run_arms(cfg) if a in sel]
    ix = {a: sel[a].set_index(["config_id", "item_id"]) for a in arms}
    units = []
    for c, i in zip(sel[arms[0]]["config_id"], sel[arms[0]]["item_id"]):
        rows = {}
        for a in arms:
            x = ix[a].loc[(c, i)]
            rows[a] = {"request_id": x["request_id"], "config_id": c, "item_id": i, "text": x["text"]}
        units.append((cw.hkey("closed_w_full/run", TASK, c, i), rows))
    units.sort(key=lambda u: u[0])
    return [cw.make_request(cfg, m, TASK, rows[a], a) for _, rows in units for a in arms for m in models]


def pilot_requests(cfg: dict, fcfg: dict, man: dict, pilot_rows: dict, control_row: dict,
                   models: list[dict]) -> list[dict]:
    """Per repeat, per pilot cell, the arms back to back; then the thinking-on controls."""
    arms = [a for a in cw.run_arms(cfg) if a in pilot_rows]
    out = []
    for rep in range(int(fcfg["pilot"]["repeats"])):
        for j in range(len(man["pilot"]["cells"])):
            for a in arms:
                for m in models:
                    out.append(cw.make_request(cfg, m, TASK, pilot_rows[a][j], a, rep=rep))
    for rep in range(int(fcfg["pilot"]["controls"])):
        for m in models:
            out.append(cw.make_request(cfg, m, TASK, control_row, "control", rep=rep))
    return out


def control_source(cfg: dict, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    scr = cw.load_requests(cw.screening_path(cfg, TASK, root))
    levels = pd.read_parquet(Path(root) / "data" / "processed" / "math" / "items_screening.parquet",
                             columns=["item_id", "level"])
    return scr, levels


# ----------------------------------------------------------------------------------------------------------------
# Projection from the 100-item run (dry run)
# ----------------------------------------------------------------------------------------------------------------

def finals(lines: list[dict]) -> tuple[dict, str | None]:
    shas = {x.get("config_sha") for x in lines if x.get("event") == "result"}
    if len(shas) > 1:
        raise SystemExit(f"a raw file holds results under {len(shas)} configurations")
    csha = shas.pop() if shas else None
    return cw.final_by_key(lines, csha), csha


def earlier_costs(lines: list[dict]) -> pd.DataFrame:
    """The 100-item run's answered MATH-500 calls with their billed cost (usage.cost) and tokens."""
    fin, _ = finals(lines)
    rows = [{"arm": cw.row_arm(x), "config_id": x.get("config_id"), "item_id": str(x.get("item_id")),
             "cost_usd": float(x["cost_usd"]), "prompt_tokens": int(x.get("prompt_tokens") or 0),
             "completion_tokens": int(x.get("completion_tokens") or 0),
             "cached_tokens": int(x.get("cached_tokens") or 0)}
            for x in fin.values() if x.get("task") == TASK and x.get("outcome") in ("reply", "refusal")
            and x.get("cost_usd") is not None and x.get("kind") in cw.ARMS]
    return pd.DataFrame(rows, columns=["arm", "config_id", "item_id", "cost_usd", "prompt_tokens",
                                       "completion_tokens", "cached_tokens"])


def project(costs: pd.DataFrame, templates, n_items: int, price: dict, arms, stop_usd: float, draws: int = 2000,
            level: float = 0.95, seed: int = 0) -> dict:
    """Per arm: the mean billed cost per call of the 100-item run on the run's templates (and on all its templates)
    times the run's calls; an item-bootstrap upper bound of the total (the earlier run's items resampled, the same
    draw for every arm); tokens at list prices with no provider cache."""
    calls = int(n_items * len(templates))
    out = {"calls_per_arm": calls, "calls": calls * len(arms), "per_arm": {}, "stop_usd": float(stop_usd)}
    t8 = costs[costs["config_id"].isin(set(templates))]
    items = sorted(t8["item_id"].unique())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(draws, len(items))) if items else np.zeros((draws, 0), dtype=int)
    boot = np.zeros(draws)
    tot = tot_all = tot_nocache = 0.0
    for a in arms:
        s, sa = t8[t8["arm"] == a], costs[costs["arm"] == a]
        if s.empty:
            raise SystemExit(f"no earlier {a} calls on the run's templates: no projection")
        per_item = s.groupby("item_id")["cost_usd"].mean().reindex(items).to_numpy(dtype=float)
        boot += np.nanmean(per_item[idx], axis=1) * calls
        m8, ma = float(s["cost_usd"].mean()), float(sa["cost_usd"].mean())
        pt, ct = float(s["prompt_tokens"].mean()), float(s["completion_tokens"].mean())
        nocache = pt * float(price["prompt"]) + ct * float(price["completion"])
        out["per_arm"][a] = {"earlier_calls_on_templates": int(len(s)), "earlier_calls_all": int(len(sa)),
                             "mean_cost_on_templates": m8, "mean_cost_all_templates": ma,
                             "mean_prompt_tokens": pt, "mean_completion_tokens": ct,
                             "mean_cached_tokens": float(s["cached_tokens"].mean()),
                             "projected_usd": m8 * calls, "projected_all_templates_usd": ma * calls,
                             "projected_no_cache_usd": nocache * calls}
        tot += m8 * calls
        tot_all += ma * calls
        tot_nocache += nocache * calls
    out.update(projected_usd=tot, projected_all_templates_usd=tot_all, projected_no_cache_usd=tot_nocache,
               high_usd=float(np.quantile(boot, (1 + level) / 2)), boot_draws=draws, level=level,
               go=bool(tot <= stop_usd))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Pilot checks
# ----------------------------------------------------------------------------------------------------------------

def forced_continuation(cfg: dict, plist: list[dict], fin: dict) -> dict:
    """The 100-item pilot's forced check on MATH-500 alone: every forced call answered and at least
    checks.prefill_min_continuation of the replies, read as the panel's, continue the box the prefill opened."""
    ck = cfg["checks"]
    rows = []
    for r in plist:
        if r["kind"] != "forced":
            continue
        x = fin.get(r["key"])
        if x is None or x.get("outcome") != "reply":
            rows.append({"key": r["key"], "pass": False, "reasons": ["no reply"], "reading": None,
                         "starts_with_prefill": False})
            continue
        c = cw.forced_check(x.get("text"), TASK, x.get("finish_reason"), cfg, r["stop_sent"],
                            x.get("reasoning_tokens"), x.get("reasoning_text"), x.get("completion_tokens"))
        rows.append({"key": r["key"], "text": x.get("text"), **c})
    share = float(np.mean([z["pass"] for z in rows])) if rows else 0.0
    ok = bool(rows) and all(z["reasons"] != ["no reply"] for z in rows) and share >= ck["prefill_min_continuation"]
    return cw._check("forced_continuation", ok, {
        "calls": len(rows), "continues": share, "stop_sent": sorted({bool(r["stop_sent"]) for r in plist
                                                                      if r["kind"] == "forced"}),
        "starts_with_prefill": sum(bool(z.get("starts_with_prefill")) for z in rows),
        "readings": {k: sum(z.get("reading") == k for z in rows) for k in cw.FORCED_READINGS},
        "failures": [{"key": z["key"], "reasons": z["reasons"],
                      "text": (z.get("text") or "")[:120] if isinstance(z.get("text"), str) else None}
                     for z in rows if not z["pass"]][:5],
        "rule": f"every forced call answered and at least {ck['prefill_min_continuation']:.0%} of the replies, read "
                "as the panel's, continue the open box (no new \\boxed, think tag or turn token, no reasoning, something "
                "inside the box, no line break before it closes, the prefix once); MATH-500 sends no stop, so there is "
                "no stop control"})


def pilot_projection(cfg: dict, plist: list[dict], fin: dict, calls_per_arm: int) -> dict:
    per, tot = {}, 0.0
    for a in cw.run_arms(cfg):
        cs = [float(fin[r["key"]]["cost_usd"]) for r in plist if r["kind"] == a and fin.get(r["key"]) is not None
              and fin[r["key"]].get("cost_usd") is not None]
        if not cs:
            return {"per_arm": per, "projected_usd": None}
        per[a] = {"pilot_calls": len(cs), "mean": float(np.mean(cs)), "projected_usd": float(np.mean(cs)) * calls_per_arm}
        tot += per[a]["projected_usd"]
    return {"per_arm": per, "projected_usd": tot}


# ----------------------------------------------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------------------------------------------

def frame(fin: dict, task: str = TASK) -> pd.DataFrame:
    acc = [dict(v, arm=cw.row_arm(v)) for v in fin.values() if v.get("outcome") in cw.ACCEPTED
           and v.get("task") == task and v.get("kind") in cw.ARMS]
    df = pd.DataFrame(acc)
    if df.empty:
        return df
    if "stop_sent" not in df:
        df["stop_sent"] = False
    df["item_id"] = df["item_id"].astype(str)
    df["n_api"] = [cw.api_count(x) for x in df.to_dict("records")]
    return df


def score(df: pd.DataFrame, scorer, tok) -> tuple[pd.DataFrame, dict]:
    ans = df[df["outcome"].isin(("reply", "refusal"))].copy()
    if ans.empty:
        return ans, {}
    sc, ver = scorer(ans)
    sc["n_local"] = tok.counts(sc["text"].tolist())
    return sc, ver


def point(mats: dict, wv: np.ndarray) -> dict:
    q = cw.model_quantities(mats, wv)
    a, b = q.get("cot_minus_requested"), q.get("cot_minus_forced")
    q["gain_diff"] = (a - b) if a is not None and b is not None else None
    return q


def boot(mats: dict, strata, bases: dict, draws: int, level: float, seed: int, chunk: int = 250) -> dict:
    """Question bootstrap with the templates fixed: items resampled with replacement (within strata), one draw for
    every arm and matrix, so the gains and their difference are paired. The draws are generated as
    scripts/closed_w.boot_quantities generates them, so a shared seed gives its intervals. Percentile intervals;
    [None, None] where no draw is finite."""
    any_m = next(iter(mats.values()))
    C, N = any_m.shape
    if N == 0:
        return {}
    rng = np.random.default_rng(seed)
    groups = {}
    for j, s in enumerate(strata):
        groups.setdefault(s, []).append(j)
    idx = np.empty((draws, N), dtype=np.int64)
    for cols in groups.values():
        cols = np.asarray(cols)
        idx[:, cols] = rng.choice(cols, size=(draws, len(cols)), replace=True)
    per = {k: np.empty((draws, C)) for k in mats}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s0 in range(0, draws, chunk):
            sl = idx[s0:s0 + chunk]
            for k, M in mats.items():
                per[k][s0:s0 + chunk] = np.nanmean(M[:, sl], axis=2).T
    a = (1 - level) / 2

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(x) for x in np.quantile(v, [a, 1 - a])] if len(v) else [None, None]
    out = {}
    for base, wv in bases.items():
        qb = cw.quantities_from(*(cw._wmean(per[x], wv) for x in ("R", "F", "C", "W")))
        qb["gain_diff"] = qb["cot_minus_requested"] - qb["cot_minus_forced"]
        out[base] = {k: ci(v) for k, v in qb.items()}
    return out


def block(scored: pd.DataFrame, templates, items, wv: np.ndarray, score_name: str, cap: int, draws: int,
          level: float, seed: int, count_col: str = "n_api") -> dict:
    col = cw.SCORE_OF[score_name]
    mats = cw.arm_mats(scored, templates, items, col, count_col, cap)
    q = point(mats, wv)
    ci = boot(mats, ["all"] * len(items), {"b": wv}, draws, level, seed).get("b", {})
    cells = {a: int(np.isfinite(mats[k]).sum()) for a, k in (("requested", "R"), ("forced", "F"), ("cot", "C"))}
    return {**q, "ci": ci, "cells": cells, "n_items": len(items), "n_templates": len(templates)}


def calls_table(df_all: pd.DataFrame, att: list[dict], expected: int, arms) -> dict:
    out = {}
    for a in arms:
        sub = df_all[df_all["arm"] == a] if len(df_all) else df_all
        at = [x for x in att if cw.row_arm(x) == a]
        out[a] = {"expected": expected, "answered": int(sub["outcome"].isin(("reply", "refusal")).sum()),
                  "refusals": int((sub["outcome"] == "refusal").sum()),
                  "blocked": int((sub["outcome"] == "blocked").sum()), "missing": int(expected - len(sub)),
                  "truncated": int((sub["finish_reason"] == "length").sum()), "attempts": len(at),
                  "failed_attempts": sum(x.get("outcome") in ("failed", "error") for x in at),
                  "flagged": sum(x.get("outcome") == "flagged" for x in at),
                  "billed_usd": float(sum(float(x.get("cost_usd") or 0) for x in at)),
                  "prompt_tokens": int(sum(int(x.get("prompt_tokens") or 0) for x in at)),
                  "cached_tokens": int(sum(int(x.get("cached_tokens") or 0) for x in at)),
                  "completion_tokens": int(sum(int(x.get("completion_tokens") or 0) for x in at)),
                  "reasoning_tokens": int(sum(int(x.get("reasoning_tokens") or 0) for x in at)),
                  "providers": sorted({str(x) for x in sub["provider"].dropna()}) if len(sub) else [],
                  "served_models": sorted({str(x) for x in sub["served_model"].dropna()}) if len(sub) else []}
    return out


def overlap(new: pd.DataFrame, old: pd.DataFrame, cfg: dict) -> dict:
    """The cells both runs sent (same request id per arm): agreement of the text, the verdicts and the crossing."""
    cols = ["arm", "config_id", "item_id", "text", "strict_correct", "content_correct", "n_api"]
    j = new[cols].merge(old[cols], on=["arm", "config_id", "item_id"], suffixes=("_new", "_old"))
    out = {}
    cap = int(cfg["caps"]["forced"][TASK])
    for a in cw.ARMS:
        g = j[j["arm"] == a]
        if g.empty:
            continue
        cross = ((pd.to_numeric(g["n_api_new"]) > cap) == (pd.to_numeric(g["n_api_old"]) > cap))
        out[a] = {"cells": int(len(g)), "same_text": float((g["text_new"] == g["text_old"]).mean()),
                  "same_strict": float((g["strict_correct_new"] == g["strict_correct_old"]).mean()),
                  "same_content": float((g["content_correct_new"] == g["content_correct_old"]).mean()),
                  "same_crossing": float(cross.mean()) if a == "requested" else None,
                  "strict_new_minus_old_pp": float(100 * (pd.to_numeric(g["strict_correct_new"]).mean()
                                                          - pd.to_numeric(g["strict_correct_old"]).mean()))}
    return out


def session_log(paths: FPaths, rec: dict) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.sessions, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=cw._json_default) + "\n")


def spend(paths: FPaths) -> dict:
    s = {ph: cw.ledger(cw.read_raw(paths.raw(ph))) for ph in PHASES}
    s["total_billed_usd"] = sum(v["billed_usd"] for v in s.values() if isinstance(v, dict))
    s["total_counted_usd"] = sum(s[ph]["counted_usd"] for ph in PHASES)
    return s


# ----------------------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------------------

def _usd(x) -> str:
    return cw.usd(x)


def _num(x, d=1, sign=False, scale=100.0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{scale * x:{'+' if sign else ''}.{d}f}"


def _ci(c, sign=False) -> str:
    if not c or c[0] is None:
        return ""
    return f" [{_num(c[0], sign=sign)}, {_num(c[1], sign=sign)}]"


def render_plan(res: dict) -> str:
    L = []
    w = L.append
    p, man = res["projection"], res["manifest_summary"]
    w(f"# Qwen4-Max on all numeric MATH-500 items, 8 discovery templates: dry run ({res['made_on']})")
    w("")
    w("Written by `scripts/closed_w_full.py plan` (offline: no call).")
    w("")
    w(f"- Items: {man['n_items']} numeric MATH-500 target items ({man['item_source']['table']}, sha256 checked "
      f"against {man['item_source']['record']}; rule: {man['item_source']['numeric_rule']}).")
    w(f"- Templates: {', '.join(man['templates'])}: the first {man['n_templates']} of "
      f"{man['template_source']['sampler']}'s {man['template_source']['k_discovery']} discovery draws "
      f"({man['template_source']['distinct_discovery_templates']} distinct), nested order \"{man['template_source']['nested_order']}\".")
    w(f"- Calls: {man['n_requests_per_arm']:,} per arm, {man['n_requests']:,} in all; configuration hash "
      f"`{man['config_sha'][:12]}` (config/closed_w.yaml, the 100-item run's).")
    w(f"- Pilot: {res['pilot_calls']} calls ({res['pilot_calls'] / man['n_requests']:.1%} of the run).")
    w(f"- Worst case at the caps (the chunk guard's bound): run ${res['worst_case_run_usd']:.2f}, pilot "
      f"${res['worst_case_pilot_usd']:.2f}.")
    w("")
    w("## Projection from the 100-item run's billed cost (usage.cost) per call on MATH-500")
    w("")
    w("| Arm | Earlier calls (these templates) | Mean cost per call | Mean prompt tok. | Mean output tok. | Projected | "
      "All 31 templates' mean, projected | No provider cache |")
    w("|---|---|---|---|---|---|---|---|")
    for a, v in p["per_arm"].items():
        w(f"| {a} | {v['earlier_calls_on_templates']:,} | ${v['mean_cost_on_templates']:.6f} | "
          f"{v['mean_prompt_tokens']:.1f} | {v['mean_completion_tokens']:.1f} | ${v['projected_usd']:.2f} | "
          f"${v['projected_all_templates_usd']:.2f} | ${v['projected_no_cache_usd']:.2f} |")
    w(f"| all | | | | | **${p['projected_usd']:.2f}** | ${p['projected_all_templates_usd']:.2f} | "
      f"${p['projected_no_cache_usd']:.2f} |")
    w("")
    w(f"Item-bootstrap {p['level']:.0%} upper bound of the projection ({p['boot_draws']:,} draws of the earlier run's "
      f"items): ${p['high_usd']:.2f}. Pilot (about 2%) adds about ${res['pilot_projected_usd']:.2f}.")
    w("")
    verdict = ("**go**: the projection is under the stop" if p["go"] else "**stop**: the projection is above the stop; "
               "nothing is sent")
    w(f"Stop rule: projection above ${p['stop_usd']:.2f} means no run. Verdict: {verdict}. Hard limit for the task "
      f"${res['limit_usd']:.2f} (pilot and run).")
    w("")
    return "\n".join(L)


def render_pilot(res: dict) -> str:
    L = []
    w = L.append
    w(f"# Qwen4-Max, all numeric MATH-500 items: pilot ({res['made_on']})")
    w("")
    w(f"Written by `scripts/closed_w_full.py pilot`; configuration `config/closed_w.yaml` (hash "
      f"`{res['config_sha'][:12]}`, the 100-item run's) and `config/closed_w_full_math.yaml`; raw replies "
      f"`{res['raw']}`. {res['n_calls']} calls: {res['n_cells']} cells x 3 arms x {res['repeats']} sends and "
      f"{res['controls']} thinking-on controls. Verdict: **{'PASS' if res['passed'] else 'FAIL'}**.")
    w("")
    w(f"Spend: pilot billed {_usd(res['spend']['pilot']['billed_usd'])}; counted for the task "
      f"{_usd(res['spend']['total_counted_usd'])} of ${res['limit']:.2f}.")
    w("")
    w("| Check | Result | Detail |")
    w("|---|---|---|")
    for c in res["checks"]:
        d = json.dumps(c["detail"], ensure_ascii=False, default=str)
        if c["name"] == "repeat_agreement":
            d = json.dumps({k: v for k, v in c["detail"].items() if k != "rows"}, ensure_ascii=False, default=str)
        w(f"| {c['name']} | {'pass' if c['pass'] else 'FAIL'} | {d[:900].replace('|', '/')} |")
    w("")
    pj = res["projection"]
    if pj.get("projected_usd") is not None:
        w(f"Projection from the pilot's own billed cost per call: ${pj['projected_usd']:.2f} for the run "
          "(" + ", ".join(f"{a} ${v['projected_usd']:.2f}" for a, v in pj["per_arm"].items()) + ").")
        w("")
    return "\n".join(L)


def render_report(rep: dict) -> str:
    L = []
    w = L.append
    d = rep["design"]
    w(f"# Qwen4-Max on all {d['n_items']} numeric MATH-500 items, {d['n_templates']} discovery templates "
      f"({rep['made_on']})")
    w("")
    w(f"Written by `scripts/closed_w_full.py report` from `{rep['raw']}`; items and templates "
      f"`manifests/closed_w_full_math_items.json`. Qwen4-Max (`qwen/qwen4-max`, "
      "reasoning off, presence penalty 0) on OpenRouter, provider pinned to Alibaba, temperature 0, top_p 1; the "
      "100-item run's requests and settings (`config/closed_w.yaml`, hash "
      f"`{rep['config_sha'][:12]}`). Items: the numeric MATH-500 target set. Templates: "
      f"{', '.join(d['templates'])}, the first {d['n_templates']} of MATH-500's discovery draws "
      "(`manifests/sampler_math.json`, nested order), each once.")
    w("")
    w("Accuracy is the mean over the templates of per-template means, in %. w is the share of requested replies "
      "longer than 16 tokens (the forced cap) by the API's count. Requested gain = CoT − requested; forced gain = CoT − "
      "forced; corrected gain = (1 − w)(CoT − forced), w and the gain each averaged over templates first "
      "(src/baseline_mixture.RULE_TEXT); the paired difference = requested gain − forced gain (= forced − requested "
      f"accuracy). Intervals: question bootstrap, {rep['bootstrap']['draws']:,} draws, items resampled together "
      "across templates and arms, templates fixed, 95% percentile.")
    w("")
    w("## Calls")
    w("")
    w("| Arm | Expected | Answered | Refusals | Provider blocks | Missing | Truncated (length) | Attempts | Billed | "
      "Prompt tok. | Output tok. | Reasoning tok. |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, v in rep["calls"].items():
        w(f"| {a} | {v['expected']:,} | {v['answered']:,} | {v['refusals']} | {v['blocked']} | {v['missing']} | "
          f"{v['truncated']} | {v['attempts']:,} | {_usd(v['billed_usd'])} | {v['prompt_tokens']:,} | "
          f"{v['completion_tokens']:,} | {v['reasoning_tokens']} |")
    s = rep["spend"]
    w("")
    w(f"Billed by OpenRouter (usage.cost): pilot {_usd(s['pilot']['billed_usd'])}, run {_usd(s['run']['billed_usd'])}; "
      f"total {_usd(s['total_billed_usd'])} against the ${rep['limit_usd']:.2f} limit (counted "
      f"{_usd(s['total_counted_usd'])}).")
    fr = rep.get("forced_reading") or {}
    if fr:
        w("")
        w(f"Forced replies as read: {fr['n']:,}; as returned {fr['as_returned']:,}, starting with `\\boxed{{` "
          f"{fr['starts_with_prefill']}.")
    ta = rep.get("token_accounting") or {}
    if ta:
        w("")
        w(f"Requested-arm token counts, API minus recount (Qwen3.6-27B tokenizer): median "
          f"{ta['median_api_minus_recount']:+.0f} (10th and 90th percentiles {ta['p10']:+.0f}, {ta['p90']:+.0f}); "
          f"the two counts fall on the same side of 16 on {100 * ta['crossing_agreement']:.1f}% of {ta['n']:,} replies.")
    w("")
    for sc in ("strict", "content"):
        g = next((x for x in rep["gains"] if x["score"] == sc), None)
        if g is None:
            continue
        w(f"## Results, {sc} scoring ({d['n_items']} items x {d['n_templates']} templates)")
        w("")
        w("| Quantity | Value | 95% CI |")
        w("|---|---|---|")
        for q in ("requested", "forced", "cot", "w", "cot_minus_requested", "cot_minus_forced", "corrected_gain",
                  "gain_diff", "residual"):
            sign = q not in ("requested", "forced", "cot", "w")
            w(f"| {Q_LABEL[q]} | {_num(g[q], sign=sign)} |{_ci(g['ci'].get(q), sign)} |")
        w(f"| w by the recount | {_num(g['w_recount'])} | |")
        w("")
    w("## Beside the 100-item run")
    w("")
    w("Rows: (a) this run; (b) this run on the 100-item run's 100 items, same 8 templates; (c) the 100-item run on "
      "the same 100 items and 8 templates (its raw replies, read only); (d) the 100-item run as published "
      "(`results/frontier_panel/closed_w.json`, discovery draws: 24 draws over 18 of its 31 templates), unchanged; "
      "its paired difference and that interval computed here from its raw replies with its own bootstrap seed. "
      "(a) against (b) is the item set; (b) against (c) is the same cells sent again; (c) against (d) the templates.")
    w("")
    for sc in ("strict", "content"):
        rows = [x for x in rep["comparison"] if x["score"] == sc]
        if not rows:
            continue
        w(f"### {sc.capitalize()} scoring")
        w("")
        w("| Row | Items x templates | Requested gain | Forced gain | w | Corrected gain | Requested − forced gain |")
        w("|---|---|---|---|---|---|---|")
        for x in rows:
            c = x.get("ci") or {}
            w(f"| {x['label']} | {x['n_items']} x {x['n_templates']} | {_num(x['cot_minus_requested'], sign=True)}"
              f"{_ci(c.get('cot_minus_requested'), True)} | {_num(x['cot_minus_forced'], sign=True)}"
              f"{_ci(c.get('cot_minus_forced'), True)} | {_num(x['w'])}{_ci(c.get('w'))} | "
              f"{_num(x['corrected_gain'], sign=True)}{_ci(c.get('corrected_gain'), True)} | "
              f"{_num(x['gain_diff'], sign=True)}{_ci(c.get('gain_diff'), True)} |")
        w("")
    pub = rep.get("published_check") or {}
    if pub:
        w(f"Check: the published 100-item values recomputed from its raw replies agree to within "
          f"{pub['max_abs_diff_points']:.2e} (points) and intervals {pub['max_abs_diff_ci']:.2e} on "
          f"{pub['compared']} values (strict and content).")
        w("")
    ov = rep.get("overlap") or {}
    if ov:
        w("### The 800 cells per arm both runs sent")
        w("")
        w("| Arm | Cells | Same text % | Same strict verdict % | Same content verdict % | Same side of 16 tokens % | "
          "Strict accuracy, this run − 100-item run (pp) |")
        w("|---|---|---|---|---|---|---|")
        for a, v in ov.items():
            cross = "-" if v["same_crossing"] is None else f"{100 * v['same_crossing']:.1f}"
            w(f"| {a} | {v['cells']} | {100 * v['same_text']:.1f} | {100 * v['same_strict']:.1f} | "
              f"{100 * v['same_content']:.1f} | {cross} | {v['strict_new_minus_old_pp']:+.1f} |")
        w("")
    w("## Per template (strict)")
    w("")
    w("| Template | Requested | Forced | CoT | w | Requested gain | Forced gain | Corrected gain |")
    w("|---|---|---|---|---|---|---|---|")
    for x in rep["per_template"]:
        if x["score"] != "strict":
            continue
        w(f"| {x['config_id']} | {_num(x['requested'])} | {_num(x['forced'])} | {_num(x['cot'])} | {_num(x['w'])} | "
          f"{_num(x['cot_minus_requested'], sign=True)} | {_num(x['cot_minus_forced'], sign=True)} | "
          f"{_num(x['corrected_gain'], sign=True)} |")
    w("")
    w("## Reading")
    w("")
    w("Refusals are scored as the text they hold; provider blocks (Alibaba's content inspection refused the call, no "
      "output) and missing calls are left out of w and accuracy and counted above. Truncated replies (finish reason "
      "`length`) are kept and scored as they are. Forced replies on MATH-500 are scored as the panel's: the prefill "
      "once in front of the reply, never cut. Scorers: " + json.dumps(rep["scorers"], ensure_ascii=False) + ".")
    w("")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------

def refuse_paid(a, fcfg: dict, paths: FPaths, phase: str) -> str | None:
    if not getattr(a, "approved", False):
        return f"{phase} is a paid phase: it needs --approved and --limit-usd"
    if getattr(a, "limit_usd", None) is None:
        return f"{phase} needs --limit-usd"
    if not a.limit_usd > 0:
        return "--limit-usd must be positive"
    cap = float(fcfg["limits"]["total_usd"])
    if a.limit_usd > cap + 1e-12:
        return f"--limit-usd {a.limit_usd} is above the task's limit ${cap:.2f}"
    pj = paths.out("_plan.json")
    if not pj.exists():
        return f"no dry run ({cw.rel(pj)} absent): run `plan` first"
    plan = json.loads(pj.read_text(encoding="utf-8"))
    if not plan["projection"]["go"]:
        return (f"the dry run projects ${plan['projection']['projected_usd']:.2f}, above the stop "
                f"${plan['projection']['stop_usd']:.2f}: no paid call")
    return None


def cmd_plan(a, cfg: dict, fcfg: dict, paths: FPaths, root: Path = ROOT, log=print, rows=None, control=None,
             ctl=None) -> int:
    check_separate(paths, cfg, root)
    cw.check_panel_decoding(cfg, root)
    items = target_items(paths.items_table, paths.items_record)
    tinfo = discovery_templates(paths.sampler, int(fcfg["templates"]["k"]))
    rows = rows if rows is not None else load_rows(cfg, root)
    if control is None:
        scr, levels = control_source(cfg, root)
        control = cw.pick_control(scr, levels, tinfo["templates"])
    man = build_manifest(cfg, fcfg, rows, items, tinfo, control, root)
    man = json.loads(cw.dumps(man))
    if paths.manifest.exists():
        old = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if manifest_body(old) != manifest_body(man):
            raise SystemExit(f"{cw.rel(paths.manifest)} exists and differs from a fresh plan; it is not overwritten "
                             "(move it aside only if no call was made under it)")
        man = old
        log(f"{cw.rel(paths.manifest)} unchanged (made {old['made_on']})")
    else:
        paths.manifest.parent.mkdir(parents=True, exist_ok=True)
        paths.manifest.write_text(cw.dumps(man) + "\n", encoding="utf-8")
        log(f"wrote {cw.rel(paths.manifest)}")
    models = [cw.model_of(cfg, fcfg["model"])]
    sel = selected(man, rows)
    rlist = run_requests(cfg, sel, models)
    if len(rlist) != man["n_requests"] or len({r["key"] for r in rlist}) != len(rlist):
        raise SystemExit(f"run list: {len(rlist)} calls ({len({r['key'] for r in rlist})} keys), expected "
                         f"{man['n_requests']}")
    prows = {arm: cw.rows_by_id(sel[arm], man["pilot"]["rows"][arm]) for arm in sel}
    ctl = ctl if ctl is not None else control_row(cfg, man, root)
    plist = pilot_requests(cfg, fcfg, man, prows, ctl, models)
    lines = cw.read_raw(paths.earlier_raw)
    costs = earlier_costs(lines)
    b = fcfg["bootstrap"]
    pj = project(costs, man["templates"], man["n_items"], models[0]["price"], cw.run_arms(cfg),
                 float(fcfg["limits"]["projection_stop_usd"]), int(b["draws"]), float(b["level"]),
                 cwp.label_seed("closed_w_full/projection"))
    pilot_proj = sum(pj["per_arm"][r["kind"]]["mean_cost_on_templates"] for r in plist if r["kind"] in cw.ARMS)
    res = {"made_on": cw.today(), "script": "scripts/closed_w_full.py plan", "config_sha": man["config_sha"],
           "manifest": cw.rel(paths.manifest),
           "manifest_summary": {k: man[k] for k in ("n_items", "item_source", "templates", "n_templates",
                                                     "template_source", "n_requests_per_arm", "n_requests",
                                                     "config_sha")},
           "earlier_raw": cw.rel(paths.earlier_raw), "projection": pj, "pilot_calls": len(plist),
           "pilot_projected_usd": pilot_proj, "limit_usd": float(fcfg["limits"]["total_usd"]),
           "worst_case_run_usd": float(sum(r["reserve_usd"] for r in rlist)),
           "worst_case_pilot_usd": float(sum(r["reserve_usd"] for r in plist)),
           "run_keys_example": rlist[0]["key"]}
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.out("_plan.json").write_text(cw.dumps(res) + "\n", encoding="utf-8")
    md = render_plan(res)
    paths.out("_plan.md").write_text(md, encoding="utf-8")
    paths.out("_plan_clean.md").write_text(cw.clean_md(md), encoding="utf-8")
    log(f"calls: {man['n_requests_per_arm']:,} per arm, {len(rlist):,} in all; pilot {len(plist)}")
    log(f"projection ${pj['projected_usd']:.2f} (all templates' mean ${pj['projected_all_templates_usd']:.2f}, "
        f"upper ${pj['high_usd']:.2f}); stop above ${pj['stop_usd']:.2f}: {'GO' if pj['go'] else 'STOP'}")
    return 0 if pj["go"] else 4


def control_row(cfg: dict, man: dict, root: Path = ROOT) -> dict:
    src = cw.load_requests(cw.screening_path(cfg, TASK, root), request_ids=[man["control"]["row"]["request_id"]])
    return cw.rows_by_id(src, [man["control"]["row"]])[0]


def pilot_rows(cfg: dict, man: dict, root: Path = ROOT) -> dict:
    ids = {a: [r["request_id"] for r in man["pilot"]["rows"][a]] for a in man["pilot"]["rows"]}
    rows = load_rows(cfg, root, ids)
    return {a: cw.rows_by_id(rows[a], man["pilot"]["rows"][a]) for a in rows}


def _http(cfg: dict, paths: FPaths):
    return cw.Http(cfg["base_url"], cw.read_key(paths.env, cfg["key_env"]), cfg["run"]["timeout_s"])


def cmd_pilot(a, cfg: dict, fcfg: dict, paths: FPaths, http=None, scorer=None, tok=None, sleep=time.sleep,
              root: Path = ROOT, log=print, prows=None, ctl=None) -> int:
    check_separate(paths, cfg, root)
    why = refuse_paid(a, fcfg, paths, "pilot")
    if why:
        log("refused: " + why)
        return 2
    cw.check_panel_decoding(cfg, root)
    man = load_manifest(paths, cfg)
    csha = cw.config_sha(cfg)
    model = cw.model_of(cfg, fcfg["model"])
    prows = prows if prows is not None else pilot_rows(cfg, man, root)
    ctl = ctl if ctl is not None else control_row(cfg, man, root)
    plist = pilot_requests(cfg, fcfg, man, prows, ctl, [model])
    if http is None:
        http = _http(cfg, paths)
    raw = paths.raw("pilot")
    lines = cw.read_raw(raw)
    other = {x.get("config_sha") for x in lines if x.get("event") == "result"} - {csha}
    if other:
        log("refused: the pilot's raw file holds replies under another configuration")
        return 2
    done = cw.done_keys(lines, csha)
    pre = cw.preflight(http, cfg, model)
    log(f"preflight {model['key']}: {'pass' if pre['pass'] else 'FAIL'}")
    res = {"made_on": cw.today(), "config_sha": csha, "raw": cw.rel(raw), "limit": a.limit_usd,
           "n_calls": len(plist), "n_cells": len(man["pilot"]["cells"]), "repeats": int(fcfg["pilot"]["repeats"]),
           "controls": int(fcfg["pilot"]["controls"]), "manifest_sha256": cw.sha256_file(paths.manifest)}
    paths.results.mkdir(parents=True, exist_ok=True)
    if not pre["pass"]:
        res.update(passed=False, checks=[pre], projection={}, spend=spend(paths), runner=None)
        _write_pilot(paths, res)
        log("pilot stopped before any paid call: preflight failed")
        return 3
    t_start = cw.now_iso()
    sp = spend(paths)
    runner = cw.Runner(http, cfg, raw, "pilot", a.limit_usd, sp["total_counted_usd"], csha, sleep=sleep, log=log)
    summ = runner.run([r for r in plist if r["key"] not in done])
    t_end = cw.now_iso()
    session_log(paths, {"phase": "pilot", "session": runner.session, "t_start": t_start, "t_end": t_end, **summ})
    lines = cw.read_raw(raw)
    fin = cw.final_by_key(lines, csha)
    tok = tok or cw.Tok(cfg, root)
    for r in plist:
        x = fin.get(r["key"])
        if x is not None and x.get("outcome") in cw.ACCEPTED:
            x["n_local"] = tok.count(x.get("text"))
    off = [fin[r["key"]] for r in plist if r["kind"] in cw.ARMS and cw._answered(fin.get(r["key"]))]
    scorer = scorer or (lambda d: cw.score_rows(d, cfg))
    scored, problem = None, None
    if off:
        try:
            scored, _ = scorer(pd.DataFrame(off))
        except Exception as e:  # noqa: BLE001
            problem = f"{type(e).__name__}: {e}"[:300]
    checks = [pre] + [c for c in cw.pilot_model_checks(cfg, model, plist, fin, scored)
                      if c["name"] not in ("forced_continuation", "forced_stop_control")]
    checks.append(forced_continuation(cfg, plist, fin))
    if problem:
        checks.append(cw._check("scoring", False, problem))
    pj = pilot_projection(cfg, plist, fin, int(man["n_requests_per_arm"]))
    sp = spend(paths)
    stop = float(fcfg["limits"]["projection_stop_usd"])
    checks.append(cw._check("projected_full_cost", pj["projected_usd"] is not None
                            and sp["total_counted_usd"] + pj["projected_usd"] <= stop,
                            {"projected_run_usd": pj["projected_usd"], "counted_so_far_usd": sp["total_counted_usd"],
                             "stop_usd": stop, "rule": "the pilot's own cost per call times the run's calls, plus what "
                                                       "the task has spent, at most the stop"}))
    if summ.get("stop_reason"):
        checks.append(cw._check("pilot_completed", False, summ["stop_reason"]))
    res.update(passed=all(c["pass"] for c in checks), checks=checks, projection=pj, spend=sp, runner=summ,
               calls=[{"key": r["key"], "kind": r["kind"], "rep": r["rep"], "outcome": (fin.get(r["key"]) or {}).get(
                   "outcome"), "finish_reason": (fin.get(r["key"]) or {}).get("finish_reason"),
                       "n_api": cw.api_count(fin[r["key"]]) if fin.get(r["key"]) else None,
                       "n_local": (fin.get(r["key"]) or {}).get("n_local"),
                       "cost_usd": (fin.get(r["key"]) or {}).get("cost_usd"),
                       "provider": (fin.get(r["key"]) or {}).get("provider")} for r in plist])
    _write_pilot(paths, res)
    log(f"pilot {'PASSED' if res['passed'] else 'FAILED'}; billed ${sp['pilot']['billed_usd']:.4f}; "
        f"{cw.rel(paths.out('_pilot.md'))}")
    for c in checks:
        if not c["pass"]:
            log(f"  failed: {c['name']}: {json.dumps(c['detail'], default=str)[:300]}")
    return 0 if res["passed"] else 3


def _write_pilot(paths: FPaths, res: dict) -> None:
    paths.out("_pilot.json").write_text(cw.dumps(res) + "\n", encoding="utf-8")
    md = render_pilot(res)
    paths.out("_pilot.md").write_text(md, encoding="utf-8")
    paths.out("_pilot_clean.md").write_text(cw.clean_md(md), encoding="utf-8")


def cmd_run(a, cfg: dict, fcfg: dict, paths: FPaths, http=None, sleep=time.sleep, root: Path = ROOT, log=print,
            rows=None) -> int:
    check_separate(paths, cfg, root)
    why = refuse_paid(a, fcfg, paths, "run")
    if why:
        log("refused: " + why)
        return 2
    cw.check_panel_decoding(cfg, root)
    man = load_manifest(paths, cfg)
    csha = cw.config_sha(cfg)
    pj = paths.out("_pilot.json")
    pilot = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else None
    if pilot is None:
        log(f"refused: no pilot ({cw.rel(pj)} absent)")
        return 2
    if pilot.get("config_sha") != csha or pilot.get("manifest_sha256") != cw.sha256_file(paths.manifest):
        log("refused: the pilot ran under another configuration or manifest")
        return 2
    if not pilot.get("passed"):
        log(f"refused: the pilot did not pass ({cw.rel(paths.out('_pilot.md'))})")
        return 2
    why = cw.prefill_gate(cfg, paths)
    if why:
        log("refused: " + why)
        return 2
    raw = paths.raw("run")
    lines = cw.read_raw(raw)
    if {x.get("config_sha") for x in lines if x.get("event") == "result"} - {csha}:
        log("refused: the run's raw file holds replies under another configuration")
        return 2
    model = cw.model_of(cfg, fcfg["model"])
    rows = rows if rows is not None else load_rows(cfg, root)
    sel = selected(man, rows)
    rlist = run_requests(cfg, sel, [model])
    done = cw.done_keys(lines, csha)
    todo = [r for r in rlist if r["key"] not in done]
    sp = spend(paths)
    log(f"run: {len(rlist):,} calls in the design, {len(done):,} answered before, {len(todo):,} to send; counted so far "
        f"${sp['total_counted_usd']:.4f} (pilot ${sp['pilot']['counted_usd']:.4f}, run ${sp['run']['counted_usd']:.4f}) "
        f"of ${a.limit_usd:.2f}")
    if http is None:
        http = _http(cfg, paths)
    t_start = cw.now_iso()
    runner = cw.Runner(http, cfg, raw, "run", a.limit_usd, sp["total_counted_usd"], csha, check=cw.run_check,
                       sleep=sleep, log=log, threads=getattr(a, "threads", None), chunk=getattr(a, "chunk", None),
                       monitor=cw.ForcedMonitor(cfg))
    summ = runner.run(todo)
    t_end = cw.now_iso()
    session_log(paths, {"phase": "run", "session": runner.session, "t_start": t_start, "t_end": t_end, **summ})
    done_after = cw.done_keys(cw.read_raw(raw), csha)
    left = sum(r["key"] not in done_after for r in rlist)
    if summ.get("stop_reason"):
        log(f"run stopped: {summ['stop_reason']}; {left:,} calls left (rerun the same command to resume)")
        return 3
    log(f"run done: {len(rlist) - left:,} of {len(rlist):,} answered, {left} left; billed this session "
        f"${summ['billed_usd']:.4f}; task total ${spend(paths)['total_billed_usd']:.4f}")
    return 0 if left == 0 else 3


def earlier_published(paths: FPaths) -> dict | None:
    if not paths.earlier_report.exists():
        return None
    return json.loads(paths.earlier_report.read_text(encoding="utf-8"))


def build_report(cfg: dict, fcfg: dict, paths: FPaths, scorer=None, tok=None, root: Path = ROOT) -> dict:
    man = load_manifest(paths, cfg)
    lines = cw.read_raw(paths.raw("run"))
    if not lines:
        raise SystemExit(f"{cw.rel(paths.raw('run'))} holds no replies: nothing to report")
    fin, csha = finals(lines)
    if csha != cw.config_sha(cfg):
        raise SystemExit("the run's replies were sent under another configuration")
    tok = tok or cw.Tok(cfg, root)
    scorer = scorer or (lambda d: cw.score_rows(d, cfg))
    df = frame(fin)
    scored, versions = score(df, scorer, tok)
    model = cw.model_of(cfg, fcfg["model"])
    mk = model["key"]
    templates, items = man["templates"], man["items"]
    cap = cw.forced_cap_of(cfg, TASK)
    b = fcfg["bootstrap"]
    draws, level = int(b["draws"]), float(b["level"])
    wv = np.ones(len(templates))
    att = [x for x in lines if x.get("event") == "result"]
    rep = {"made_on": cw.today(), "script": "scripts/closed_w_full.py report", "config_sha": csha,
           "raw": cw.rel(paths.raw("run")), "model": mk, "task": TASK, "basis": BASIS,
           "design": {"n_items": len(items), "n_templates": len(templates), "templates": templates,
                      "n_requests_per_arm": man["n_requests_per_arm"], "n_requests": man["n_requests"],
                      "item_source": man["item_source"], "template_source": man["template_source"]},
           "bootstrap": {"draws": draws, "level": level, "unit": "question (MATH-500 item)", "templates": "fixed"},
           "scorers": versions, "tokenizer": tok.name, "limit_usd": float(fcfg["limits"]["total_usd"]),
           "calls": calls_table(df, att, int(man["n_requests_per_arm"]), cw.run_arms(cfg)), "spend": spend(paths),
           "gains": [], "per_template": [], "comparison": [], "overlap": {}, "published_check": {}}
    rep["calls_total"] = {k: sum(v[k] for v in rep["calls"].values()) for k in
                          ("expected", "answered", "refusals", "blocked", "missing", "truncated", "attempts")}
    f = scored[scored["arm"] == "forced"] if len(scored) else scored
    if len(f):
        rep["forced_reading"] = {"n": int(len(f)), **{k: int((f["forced_reading"] == k).sum())
                                                       for k in cw.FORCED_READINGS},
                                 "starts_with_prefill": int(f["text"].map(
                                     lambda s: isinstance(s, str) and s.startswith(cfg["prefill"])).sum())}
    rq = scored[scored["arm"] == "requested"].dropna(subset=["n_api"]) if len(scored) else scored
    if len(rq):
        dd = rq["n_api"].astype(float) - rq["n_local"].astype(float)
        rep["token_accounting"] = {"n": int(len(rq)), "median_api_minus_recount": float(dd.median()),
                                   "p10": float(dd.quantile(0.1)), "p90": float(dd.quantile(0.9)),
                                   "crossing_agreement": float(np.mean([(x > cap) == (y > cap) for x, y in
                                                                        zip(rq["n_api"], rq["n_local"])]))}
    for sc in ("strict", "content"):
        seed = cwp.label_seed(f"closed_w_full/bootstrap/{mk}/{TASK}/{sc}/{BASIS}")
        g = block(scored, templates, items, wv, sc, cap, draws, level, seed)
        g_rc = point(cw.arm_mats(scored, templates, items, cw.SCORE_OF[sc], "n_local", cap), wv)
        rep["gains"].append({"model": mk, "task": TASK, "score": sc, "basis": BASIS, **g, "w_recount": g_rc["w"]})
        mats = cw.arm_mats(scored, templates, items, cw.SCORE_OF[sc], "n_api", cap)
        tm = {k: cw.template_means(v) for k, v in mats.items()}
        qt = cw.quantities_from(tm["R"], tm["F"], tm["C"], tm["W"])
        for j, c in enumerate(templates):
            rep["per_template"].append({"model": mk, "task": TASK, "score": sc, "config_id": c,
                                        **{k: (float(v[j]) if np.isfinite(v[j]) else None) for k, v in qt.items()},
                                        "cells": {a: int(np.isfinite(mats[k][j]).sum()) for a, k in
                                                  (("requested", "R"), ("forced", "F"), ("cot", "C"))}})
    # beside the 100-item run (read only)
    em = json.loads(paths.earlier_manifest.read_text(encoding="utf-8")) if paths.earlier_manifest.exists() else None
    pub = earlier_published(paths)
    old_lines = cw.read_raw(paths.earlier_raw)
    if em is not None and old_lines:
        b100 = [str(i) for i in em["tasks"][TASK]["items"]]
        t31 = list(em["tasks"][TASK]["templates"])
        ofin, _ = finals(old_lines)
        odf = frame(ofin)
        osc, _ = score(odf, scorer, tok)
        checks = []
        for sc in ("strict", "content"):
            main = next(x for x in rep["gains"] if x["score"] == sc)
            rep["comparison"].append({"label": "(a) this run", **{k: main[k] for k in QUANTS}, "ci": main["ci"],
                                      "score": sc, "n_items": len(items), "n_templates": len(templates)})
            gb = block(scored, templates, b100, wv, sc, cap, draws, level,
                       cwp.label_seed(f"closed_w_full/bootstrap/{mk}/{TASK}/{sc}/b100"))
            rep["comparison"].append({"label": "(b) this run, the 100-item run's items", **gb, "score": sc})
            gc = block(osc, templates, b100, wv, sc, cap, draws, level,
                       cwp.label_seed(f"closed_w_full/bootstrap/{mk}/{TASK}/{sc}/earlier_b100"))
            rep["comparison"].append({"label": "(c) the 100-item run, same 100 items and 8 templates", **gc,
                                      "score": sc})
            bw = cw.basis_weights(TASK, t31, root)
            if "discovery" in bw:
                mats = cw.arm_mats(osc, t31, b100, cw.SCORE_OF[sc], "n_api", cap)
                q = point(mats, bw["discovery"])
                ci = boot(mats, ["all"] * len(b100), {"discovery": bw["discovery"]}, draws, level,
                          cwp.label_seed(f"closed_w/bootstrap3/{mk}/{TASK}/{sc}"))["discovery"]
                row = {"label": "(d) the 100-item run as published (discovery draws)", **q, "ci": ci, "score": sc,
                       "n_items": len(b100), "n_templates": len(t31), "recomputed": True}
                if pub is not None:
                    pg = [x for x in pub["gains"] if x["model"] == mk and x["task"] == TASK and x["score"] == sc
                          and x["basis"] == "discovery"]
                    if len(pg) == 1:
                        pg = pg[0]
                        for k in cw.QUANTS:
                            if pg.get(k) is not None and q.get(k) is not None:
                                checks.append(("point", abs(pg[k] - q[k])))
                            pc, mc = (pg.get("ci") or {}).get(k), ci.get(k)
                            if pc and mc and pc[0] is not None and mc[0] is not None:
                                checks.append(("ci", max(abs(pc[0] - mc[0]), abs(pc[1] - mc[1]))))
                        # the published values are shown as published; the paired difference and its interval are new
                        row.update({k: pg[k] for k in cw.QUANTS}, recomputed=False,
                                   ci={**{k: pg["ci"].get(k) for k in cw.QUANTS}, "gain_diff": ci["gain_diff"]})
                        row["gain_diff"] = pg["cot_minus_requested"] - pg["cot_minus_forced"]
                rep["comparison"].append(row)
        if checks:
            pts = [v for k, v in checks if k == "point"]
            cis = [v for k, v in checks if k == "ci"]
            rep["published_check"] = {"compared": len(checks), "max_abs_diff_points": float(max(pts) * 100),
                                      "max_abs_diff_ci": float(max(cis) * 100) if cis else None}
        rep["overlap"] = overlap(scored[scored["item_id"].isin(set(b100)) & scored["config_id"].isin(set(templates))],
                                 osc[osc["item_id"].isin(set(b100)) & osc["config_id"].isin(set(templates))], cfg)
    return rep


def cmd_report(a, cfg: dict, fcfg: dict, paths: FPaths, scorer=None, tok=None, root: Path = ROOT, log=print) -> int:
    check_separate(paths, cfg, root)
    rep = build_report(cfg, fcfg, paths, scorer, tok, root)
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.out(".json").write_text(cw.dumps(rep) + "\n", encoding="utf-8")
    md = render_report(rep)
    paths.out(".md").write_text(md, encoding="utf-8")
    paths.out("_clean.md").write_text(cw.clean_md(md), encoding="utf-8")
    pd.DataFrame(rep["per_template"]).drop(columns=["cells"]).to_csv(paths.out("_templates.csv"), index=False)
    log(f"wrote {cw.rel(paths.out('.md'))}, _clean.md, .json, _templates.csv")
    return 0


def budget_entry(cfg: dict, fcfg: dict, paths: FPaths) -> dict:
    """One entry for the task: the pilot and the run together."""
    model = cw.model_of(cfg, fcfg["model"])
    res = {ph: [x for x in cw.read_raw(paths.raw(ph)) if x.get("event") == "result"] for ph in PHASES}
    allr = res["pilot"] + res["run"]
    ans = [x for x in allr if x.get("outcome") in ("reply", "refusal")]

    def tot(k, xs=allr):
        return sum(int(x.get(k) or 0) for x in xs)
    billed = {ph: sum(float(x.get("cost_usd") or 0) for x in v) for ph, v in res.items()}
    sess = [json.loads(l) for l in paths.sessions.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if paths.sessions.exists() else []
    secs = {ph: sum(float(s.get("seconds") or 0) for s in sess if s.get("phase") == ph) for ph in PHASES}
    t0 = min((s["t_start"] for s in sess), default="-")
    t1 = max((s["t_end"] for s in sess), default="-")
    by_kind = ", ".join(f"{ph} {k} {sum(x.get('kind') == k and x.get('outcome') in ('reply', 'refusal') for x in res[ph])}"
                        for ph in PHASES for k in cw.REQUEST_KINDS if any(x.get("kind") == k for x in res[ph]))
    sp = spend(paths)
    return {
        "date": cw.today(),
        "run": RUN_TITLE,
        "model": f"{model['id']} on OpenRouter, provider pinned to {model['provider']['only']}, reasoning "
                 f"{model['reasoning_off']}, temperature {cfg['decoding']['temperature']}, top_p "
                 f"{cfg['decoding']['top_p']}" + (f", {model['extra']}" if model.get("extra") else ""),
        "unique_generations": f"{len(ans):,} answered calls of {len(allr):,} attempts (refusals "
                              f"{sum(x.get('outcome') == 'refusal' for x in allr)}, provider blocks "
                              f"{sum(x.get('outcome') == 'blocked' for x in allr)}); answered by kind: {by_kind}",
        "cached_calls": 0,
        "tokens": f"prompt {tot('prompt_tokens'):,} (cached by the provider {tot('cached_tokens'):,}), completion "
                  f"{tot('completion_tokens'):,} (reasoning {tot('reasoning_tokens'):,})",
        "cost_usd": round(billed["pilot"] + billed["run"], 4),
        "cost_detail": f"billed by OpenRouter (usage.cost): pilot ${billed['pilot']:.4f}, run ${billed['run']:.4f}; "
                       f"counted against the ${float(fcfg['limits']['total_usd']):.2f} limit "
                       f"${sp['total_counted_usd']:.4f}",
        "runtime": f"pilot {secs['pilot']:.0f} s, run {secs['run']:.0f} s ({len(sess)} sessions, {t0} to {t1}, UTC)",
        "notes": f"results/frontier_panel/{paths.prefix}.md; pilot {paths.prefix}_pilot.md; dry run "
                 f"{paths.prefix}_plan.md; the 100-item run's files "
                 "unchanged",
    }


def cmd_budget(a, cfg: dict, fcfg: dict, paths: FPaths, log=print) -> int:
    check_separate(paths, cfg)
    log_ = yaml.safe_load(paths.budget.read_text(encoding="utf-8")).get("log") or []
    if any(isinstance(e, dict) and e.get("run") == RUN_TITLE for e in log_):
        log(f"{cw.rel(paths.budget)} already holds this task's entry: not added again")
        return 2
    entry = budget_entry(cfg, fcfg, paths)
    if not cw.append_budget(paths.budget, entry):
        log("config/budget.yaml not updated (layout not recognised); add this entry by hand:\n" +
            yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
        return 3
    log(f"added one entry to {cw.rel(paths.budget)}: ${entry['cost_usd']:.4f}")
    return 0


def main(argv=None, *, http=None, scorer=None, tok=None, paths: FPaths | None = None, sleep=None, root: Path = ROOT,
         log=print, cfg=None, fcfg=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=FULL_CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    for name in ("pilot", "run"):
        p = sub.add_parser(name)
        p.add_argument("--approved", action="store_true", help="approval for this paid phase")
        p.add_argument("--limit-usd", type=float, default=None, help="hard spend limit for the task (USD)")
        if name == "run":
            p.add_argument("--threads", type=int, default=None)
            p.add_argument("--chunk", type=int, default=None)
    sub.add_parser("report")
    sub.add_parser("budget")
    a = ap.parse_args(argv)
    fcfg = fcfg or load_full(a.config)
    cfg = cfg or base_config(fcfg, root)
    paths = paths or FPaths(fcfg, root)
    sleep = sleep or time.sleep
    if a.cmd == "plan":
        return cmd_plan(a, cfg, fcfg, paths, root, log)
    if a.cmd == "pilot":
        return cmd_pilot(a, cfg, fcfg, paths, http, scorer, tok, sleep, root, log)
    if a.cmd == "run":
        return cmd_run(a, cfg, fcfg, paths, http, sleep, root, log)
    if a.cmd == "report":
        return cmd_report(a, cfg, fcfg, paths, scorer, tok, root, log)
    return cmd_budget(a, cfg, fcfg, paths, log)


if __name__ == "__main__":
    raise SystemExit(main())
