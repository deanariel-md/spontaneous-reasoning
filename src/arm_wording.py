r"""The arm-wording run (config/arm_wording.yaml): the chain-of-thought trigger and the
direct-answer clause are drawn from four wordings each, on six configurations of the GSM8K and MATH-500 banks,
for the 29 models of GSM8K-29, with three arms (chain of thought, requested direct, forced direct).

Data flow
    wordings: Kojima et al.'s Table 4 (config) + the paraphrase record and review (scripts/paraphrase_d2.py)
    -> configurations drawn from the samplers (manifests/arm_wording_configs.json)
    -> screening requests (screening items, two configurations, current wording + five candidates per arm)
    -> screen report and the chosen four per arm (manifests/arm_wording_screen.json, arm_wording_wordings.json)
    -> collection requests (data/requests/arm_wording/collection/<key>.parquet)
    -> scored table, discovery configurations first (data/processed/arm_wording/)
    -> predictions saved (manifests/predictions_arm_wording.json) -> everything scored -> analyses (results/arm_wording/)

Request layout. config_id is "<task>/<bank configuration>/<wording id>"; the wording id is cot0..cot5 for the chain of
thought (cot0 the current trigger) and dir0..dirN for the direct clause (dir0 the current clause), shared by the
requested and forced arms. item_id is the task's item id as a string. request_id is "<config_id>|<arm>|<item_id>",
with the prefix "screen|" for screening requests. Nothing here reads an outcome to choose a wording, configuration,
item or model; the screen reads format and behaviour only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from . import d2_io, decision, estimators as est, protocol_ceiling as pc

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "arm_wording.yaml"
ARMS = ("zero_shot_cot", "direct_answer", "forced_direct")
COT, REQ, FD = ARMS
BANK_ARM = {COT: "zero_shot_cot", REQ: "direct_answer", FD: "direct_answer"}
FAMILY = {COT: "cot", REQ: "dir", FD: "dir"}
TASKS = ("gsm8k", "math")
PREFILL = "\\boxed{"
SCREEN_PREFIX = "screen|"


class ArmWordingError(Exception):
    """An input or an order of steps that the run refuses; scripts print it as REFUSED and exit 2."""


# --------------------------------------------------------------------------------------------
# Paths (one base folder, so a dry run writes nothing in the repository)
# --------------------------------------------------------------------------------------------

class Paths:
    def __init__(self, base: str | os.PathLike | None = None):
        b = Path(base) if base else ROOT
        self.base = b
        self.manifests = b / "manifests"
        self.configs = self.manifests / "arm_wording_configs.json"
        self.screen = self.manifests / "arm_wording_screen.json"
        self.wordings = self.manifests / "arm_wording_wordings.json"
        self.predictions = self.manifests / "predictions_arm_wording.json"
        self.predictions_sha = self.manifests / "predictions_arm_wording.sha256"
        # the mechanism forecast, saved from token counts before any arm-wording output is scored
        self.mechanism = self.manifests / "predictions_arm_wording_mechanism.json"
        self.mechanism_sha = self.manifests / "predictions_arm_wording_mechanism.sha256"
        self.requests = b / "data" / "requests" / "arm_wording"
        self.outputs = b / "data" / "d2_out" / "arm_wording"
        self.processed = b / "data" / "processed" / "arm_wording"
        self.results = b / "results" / "arm_wording"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path: str | os.PathLike, obj: Any) -> str:
    raw = (json.dumps(obj, indent=1, ensure_ascii=False, default=_default) + "\n").encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return sha256_bytes(raw)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def write_once(path: str | os.PathLike, obj: Any, what: str) -> str:
    """Write a manifest; an existing file with the same content is kept (its time stands), other content is refused."""
    p = Path(path)
    new = json.loads(json.dumps(obj, ensure_ascii=False, default=_default))
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if _substance(old) != _substance(new):
            raise ArmWordingError(f"{p} exists with other content; {what} is written once (delete it only if no request "
                                  "built from it was run)")
        return "unchanged"
    write_json(p, new)
    return "written"


def _substance(obj):
    if isinstance(obj, dict):
        return {k: _substance(v) for k, v in obj.items() if k not in ("created",)}
    if isinstance(obj, list):
        return [_substance(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else CONFIG
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def seed_for(label: str) -> int:
    """The first 8 bytes of sha256(label), as in the other seeded draws."""
    return decision.label_seed(label)


# --------------------------------------------------------------------------------------------
# Wordings
# --------------------------------------------------------------------------------------------

def qualifies_as_instruction(text: str) -> bool:
    """The rule for Kojima's templates: a complete sentence ending with a period (at least two words, an upper-case
    first letter); a template ending with a comma or with no final punctuation is the opening of an answer."""
    t = text.strip()
    return bool(t) and t[0].isupper() and t.endswith(".") and len(t.split()) >= 2


def cot_candidates(cfg: Mapping, *, require_verified: bool = True) -> list[dict]:
    """The current trigger (cot0) and the first n_candidates templates of Kojima's Table 4 that qualify, in table
    order. Refused while the strings are unverified, unless `require_verified` is False (tests and dry runs)."""
    c = cfg["cot"]
    if require_verified and not c.get("verified_on"):
        raise ArmWordingError("the chain-of-thought templates in config/arm_wording.yaml are unverified: check them "
                              "against Kojima et al. 2022 (arXiv 2205.11916), Table 4, and set cot.verified_on")
    current = cfg["current"][COT]
    out = [{"id": "cot0", "text": current, "source": "current trigger (config/promptsuite*.yaml)", "current": True}]
    rows = sorted(c["table4_instructive"], key=lambda r: int(r["row"]))
    if not rows or rows[0]["text"] != current:
        raise ArmWordingError("the first template of Table 4 must be the current trigger")
    for r in rows[1:]:
        if len(out) > int(cfg["screen"]["n_candidates"]):
            break
        if r["text"] != current and qualifies_as_instruction(r["text"]):
            out.append({"id": f"cot{len(out)}", "text": r["text"], "source": f"Kojima et al. 2022, Table 4, row {r['row']}",
                        "current": False})
    return out


def direct_candidates(cfg: Mapping, record_path: str | os.PathLike, review_path: str | os.PathLike) -> list[dict]:
    """The current clause (dir0) and the first n_candidates kept paraphrases, in the saved record's variant order.
    Refused when the review does not match the record or holds an empty verdict."""
    rec_raw = Path(record_path).read_bytes()
    rec = json.loads(rec_raw.decode("utf-8"))
    rev = json.loads(Path(review_path).read_text(encoding="utf-8"))
    current = cfg["current"][REQ]
    if rec["request"]["text"] != current:
        raise ArmWordingError(f"the paraphrase record paraphrases {rec['request']['text']!r}, not the current clause")
    if rev.get("record_sha256") != sha256_bytes(rec_raw):
        raise ArmWordingError("the review was written for another paraphrase record")
    verdicts = {v["text"]: v for v in rev["verdicts"]}
    variants = [t for t in rec["variants"] if t != current]
    missing = [t for t in variants if t not in verdicts or verdicts[t]["verdict"] not in ("keep", "drop")]
    if missing:
        raise ArmWordingError(f"{len(missing)} paraphrases have no keep/drop verdict, e.g. {missing[:1]}")
    drops = [t for t in variants if verdicts[t]["verdict"] == "drop" and not verdicts[t].get("note")]
    if drops:
        raise ArmWordingError(f"a drop needs a note: {drops[:1]}")
    out = [{"id": "dir0", "text": current, "source": "current clause (config/promptsuite*.yaml)", "current": True}]
    for t in variants:
        if len(out) > int(cfg["screen"]["n_candidates"]):
            break
        if verdicts[t]["verdict"] == "keep":
            out.append({"id": f"dir{len(out)}", "text": t, "source": "paraphrase, kept in review", "current": False})
    return out


# --------------------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------------------

def draw_configurations(task: str, sampler: Mapping, n_disc: int, n_val: int) -> dict:
    """The draw from the sampler's ids alone: n_disc of the distinct discovery configurations and n_val of
    the distinct validation configurations that no discovery draw uses, uniformly without replacement, in the order
    drawn."""
    D = list(dict.fromkeys(sampler["discovery_config_ids"]))
    V = [c for c in dict.fromkeys(sampler["validation_config_ids"]) if c not in set(D)]
    if len(D) < n_disc or len(V) < n_val:
        raise ArmWordingError(f"{task}: {len(D)} discovery and {len(V)} fresh validation configurations; need "
                              f"{n_disc} and {n_val}")
    out = {}
    for name, pool, n in (("discovery", D, n_disc), ("validation", V, n_val)):
        label = f"d24/arm-wording/{task}/{name}"
        rng = np.random.default_rng(seed_for(label))
        ix = rng.choice(len(pool), size=n, replace=False)
        out[name] = [pool[int(i)] for i in ix]
        out[f"{name}_pool"] = pool
        out[f"{name}_seed_label"] = label
    return out


def configs_manifest(samplers: Mapping[str, Path], banks: Mapping[str, Path], cfg: Mapping) -> dict:
    tasks = {}
    for t in TASKS:
        raw = Path(samplers[t]).read_bytes()
        s = json.loads(raw.decode("utf-8"))
        d = draw_configurations(t, s, int(cfg["configurations"]["n_discovery"]), int(cfg["configurations"]["n_validation"]))
        tasks[t] = {**d, "sampler": d2_io.rel_path(samplers[t]), "sampler_sha256": sha256_bytes(raw),
                    "bank": d2_io.rel_path(banks[t]), "bank_sha256": sha256_file(banks[t])}
    return {"kind": "arm_wording_configurations", "rule": "arm wording run: configurations", "tasks": tasks}


def task_configs(manifest: Mapping, task: str) -> tuple[list[str], list[str]]:
    t = manifest["tasks"][task]
    return list(t["discovery"]), list(t["validation"])


# --------------------------------------------------------------------------------------------
# Messages and requests
# --------------------------------------------------------------------------------------------

def replace_clause(messages: Sequence[Mapping], old: str, new: str) -> list[dict]:
    """The messages with `old` replaced by `new`; `old` must occur exactly once across the messages."""
    n = sum(m["content"].count(old) for m in messages)
    if n != 1:
        raise ArmWordingError(f"the clause {old!r} occurs {n} times in the messages; it must occur exactly once")
    return [{"role": m["role"], "content": m["content"].replace(old, new)} for m in messages]


def arm_messages(bank: Mapping, config_id: str, arm: str, text: str, current: Mapping) -> list[dict]:
    """The bank's messages of the arm (the requested direct messages for the forced arm, whose prefill comes from the
    decoding) with the current clause replaced by `text`."""
    msgs = bank["configurations"][config_id]["arms"][BANK_ARM[arm]]["messages"]
    return replace_clause(msgs, current[BANK_ARM[arm]], text)


def render(model_key: str, task: str, bank: Mapping, config_ids: Sequence[str], wordings: Mapping[str, Sequence[Mapping]],
           items: pd.DataFrame, current: Mapping, prefix: str = "") -> pd.DataFrame:
    """One row per configuration x arm x wording x item, with the runner's columns."""
    rows = {c: [] for c in ("request_id", "model_key", "arm", "config_id", "item_id", "messages_json")}
    ids = [str(i) for i in items["item_id"]]
    qs = list(items["question"])
    for cid in config_ids:
        for arm in ARMS:
            for w in wordings[FAMILY[arm]]:
                msgs = arm_messages(bank, cid, arm, w["text"], current)
                conf = f"{task}/{cid}/{w['id']}"
                for iid, q in zip(ids, qs):
                    rows["request_id"].append(f"{prefix}{conf}|{arm}|{iid}")
                    rows["model_key"].append(model_key)
                    rows["arm"].append(arm)
                    rows["config_id"].append(conf)
                    rows["item_id"].append(iid)
                    rows["messages_json"].append(json.dumps(d2_io.substitute_question(msgs, q), ensure_ascii=False))
    df = pd.DataFrame(rows)
    if df["request_id"].duplicated().any():
        raise ArmWordingError("request ids are not unique")
    return df


def parse_config_id(config_id: str) -> tuple[str, str, str]:
    task, cid, wid = str(config_id).split("/")
    return task, cid, wid


def screen_item_ids(task: str, pool_ids: Sequence[str], n: int) -> list[str]:
    """n screening items drawn uniformly without replacement from the sorted pool, seed sha256(d24/screen-items/<task>),
    returned sorted."""
    pool = sorted(str(i) for i in pool_ids)
    if n > len(pool):
        raise ArmWordingError(f"{task}: {n} screening items asked of a pool of {len(pool)}")
    rng = np.random.default_rng(seed_for(f"d24/screen-items/{task}"))
    return sorted(pool[int(i)] for i in rng.choice(len(pool), size=n, replace=False))


def check_current_matches(ours: pd.DataFrame, collection: pd.DataFrame, task: str) -> dict:
    """The current-wording rows reproduce the collection's request messages byte for byte (same configuration, arm and
    item). Returns the counts; raises on any mismatch."""
    cur = ours[ours["config_id"].str.endswith(("/cot0", "/dir0")) & (ours["arm"] != FD)].copy()
    cur["bank_config"] = cur["config_id"].map(lambda c: parse_config_id(c)[1])
    col = collection.copy()
    col["item_id"] = col["item_id"].astype(str)
    m = cur.merge(col[["config_id", "arm", "item_id", "messages_json"]], left_on=["bank_config", "arm", "item_id"],
                  right_on=["config_id", "arm", "item_id"], suffixes=("", "_collection"))
    bad = int((m["messages_json"] != m["messages_json_collection"]).sum())
    if bad:
        raise ArmWordingError(f"{task}: {bad} current-wording requests differ from the collection's messages")
    return {"task": task, "compared": int(len(m)), "identical": int(len(m))}


# --------------------------------------------------------------------------------------------
# Screening (format and behaviour only; no correctness)
# --------------------------------------------------------------------------------------------

def screen_row_metrics(out: pd.DataFrame, task: str, gsm8k_spec=None) -> pd.DataFrame:
    """Format flags per screening output (src/screening.row_metrics for GSM8K, src/math_d2.row_metrics for MATH-500),
    with task and wording columns; config_id becomes the wording id so that screening.apply_rules pools over the bank
    configurations."""
    from . import math_d2, screening
    df = out.copy()
    parts = df["config_id"].map(parse_config_id)
    df["task"] = [p[0] for p in parts]
    df["wording"] = [p[2] for p in parts]
    df = df[df["task"] == task]
    if task == "math":
        rows = math_d2.row_metrics(df, None, screening.THINK_TAG_RE)
    else:
        rows = screening.row_metrics(df, {}, registered=gsm8k_spec)
    rows["task"] = task
    rows["wording"] = df["wording"].to_numpy()
    rows["bank_config"] = [parse_config_id(c)[1] for c in df["config_id"]]
    rows["box_closed"] = box_closed(df["output_text"].tolist())
    rows["config_id"] = rows["wording"]
    return rows


def box_closed(texts: Sequence[str]) -> np.ndarray:
    """A complete \\boxed{} in the text (the forced arm runs without a stop string; MATH-500's forced-arm rule)."""
    from . import bbh
    return np.array([bool(bbh.boxed_contents(t if isinstance(t, str) else "")) for t in texts])


def apply_screen(rows_by_task: Mapping[str, pd.DataFrame], rules: Mapping, relative: Mapping[str, bool]) -> dict:
    """screening.apply_rules per task on the requested-direct and chain-of-thought rows (wording in place of
    configuration; both model rules diagnostic). A wording fails when it fails on either task."""
    from . import screening
    per_task, failed = {}, {}
    for task, rows in rows_by_task.items():
        r = rows[rows["arm"].isin([COT, REQ])]
        res = screening.apply_rules(r, rules, has_registered=True,
                                    diagnostic_model_rules=("direct_answer_meaningful", "thinking_off_holds"),
                                    relative_truncation=bool(relative[task]))
        per_task[task] = res
        for ex in res["excluded_configurations"]:
            for v in ex["violations"]:
                failed.setdefault(ex["config_id"], []).append({"task": task, **{k: v[k] for k in v if k != "pooled_over_models"}})
        fd = rows[rows["arm"] == FD]
        per_task[task]["forced_box_closed_pct"] = {w: float(100 * g["box_closed"].mean())
                                                   for w, g in fd.groupby("wording")}
    return {"per_task": per_task, "failed": failed}


def choose_wordings(candidates: Mapping[str, Sequence[Mapping]], failed: Mapping[str, list], n_new: int) -> dict:
    """Per family: the current wording and the first n_new candidates that did not fail, in candidate order."""
    out = {}
    for fam, cands in candidates.items():
        cur = [c for c in cands if c["current"]]
        new = [c for c in cands if not c["current"] and c["id"] not in failed][:n_new]
        out[fam] = [{"id": c["id"], "text": c["text"], "source": c["source"], "current": c["current"],
                     "screen_failed": c["id"] in failed} for c in cur + new]
    return out


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def cut_at_first_brace(text: str) -> str:
    """The GSM8K forced arm's text as the collection's stop at the first "}" (kept in the output) returned it: the
    prefilled text cut after the first "}" that follows the prefill. Greedy decoding writes the same tokens up to that
    point with or without the stop string, so the cut reproduces the collection's text."""
    t = text if isinstance(text, str) else ""
    i = t.find("}", len(PREFILL) if t.startswith(PREFILL) else 0)
    return t if i < 0 else t[:i + 1]


def score(out: pd.DataFrame, task: str, items: pd.DataFrame, gsm8k_spec=None) -> pd.DataFrame:
    """Strict and lenient correctness of one task's outputs (prefill already prepended by d2_io.inspect_work_dir).
    GSM8K's forced outputs are first cut after the first closing brace (cut_at_first_brace), as the collection's stop
    string cut them, and a reply the cut left with its box open is read as the closed box it was meant to be
    (src/forced_stop.py); MATH-500's forced arm ran without a stop string in the collection too, and is scored as
    it is."""
    from . import forced_stop, math_task, scoring
    df = out.copy()
    if task == "gsm8k":
        fd = df["arm"] == FD
        df.loc[fd, "output_text"] = [forced_stop.close(cut_at_first_brace(t)) for t in df.loc[fd, "output_text"]]
        g = df.copy()
        g["item_id"] = g["item_id"].astype("int64")
        gold = items.assign(item_id=items["item_id"].astype("int64"))
        s = scoring.score_outputs(g, gold, gsm8k_spec)
        closed = box_closed(s["output_text"].tolist())
        s["box_closed"] = np.where(s["arm"] == FD, [t.startswith(PREFILL) and "}" in t[len(PREFILL):]
                                                    for t in s["output_text"].astype(str)], closed)
    else:
        it = items.assign(item_id=items["item_id"].astype(str))
        s = math_task.score_outputs(df.assign(item_id=df["item_id"].astype(str)), it)
        s["box_closed"] = box_closed(s["output_text"].tolist())
    s["item_id"] = s["item_id"].astype(str)
    return s


def tidy(scored: pd.DataFrame) -> pd.DataFrame:
    """The columns the analysis reads, with task, bank configuration and wording split out of config_id."""
    parts = scored["config_id"].map(parse_config_id)
    out = pd.DataFrame({"task": [p[0] for p in parts], "config": [p[1] for p in parts], "wording": [p[2] for p in parts],
                        "arm": scored["arm"].astype(str).to_numpy(), "model_key": scored["model_key"].astype(str).to_numpy(),
                        "item_id": scored["item_id"].astype(str).to_numpy(),
                        "strict_correct": scored["strict_correct"].astype(float).to_numpy(),
                        "lenient_correct": scored["lenient_correct"].astype(float).to_numpy(),
                        "n_output_tokens": scored["n_output_tokens"].astype(int).to_numpy(),
                        "box_closed": scored["box_closed"].astype(bool).to_numpy(),
                        "prompt_sha256": scored["prompt_sha256"].astype(str).to_numpy()})
    return out


# --------------------------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------------------------

class TaskArrays:
    """Per-item correctness of one task: corr[arm] has shape (configs, wordings, models, items); acc() gives points."""

    def __init__(self, table: pd.DataFrame, task: str, configs: Sequence[str], cot_ids: Sequence[str],
                 dir_ids: Sequence[str], models: Sequence[str], score: str = "strict_correct"):
        t = table[table["task"] == task]
        self.task, self.configs, self.models = task, list(configs), list(models)
        self.wids = {COT: list(cot_ids), REQ: list(dir_ids), FD: list(dir_ids)}
        self.items = sorted(t["item_id"].unique())
        ii = {v: k for k, v in enumerate(self.items)}
        ci = {v: k for k, v in enumerate(self.configs)}
        mi = {v: k for k, v in enumerate(self.models)}
        self.corr = {}
        for arm in ARMS:
            wi = {v: k for k, v in enumerate(self.wids[arm])}
            a = t[(t["arm"] == arm) & t["config"].isin(ci) & t["wording"].isin(wi) & t["model_key"].isin(mi)]
            shape = (len(self.configs), len(self.wids[arm]), len(self.models), len(self.items))
            n_want = int(np.prod(shape))
            if len(a) != n_want:
                raise ArmWordingError(f"{task}/{arm}: {len(a)} scored rows, the design has {n_want}")
            arr = np.full(shape, np.nan)
            arr[a["config"].map(ci).to_numpy(), a["wording"].map(wi).to_numpy(), a["model_key"].map(mi).to_numpy(),
                a["item_id"].map(ii).to_numpy()] = a[score].to_numpy()
            if np.isnan(arr).any():
                raise ArmWordingError(f"{task}/{arm}: duplicated or missing cells")
            self.corr[arm] = arr

    def acc(self, weights=None) -> dict:
        """{arm: (configs, wordings, models)} accuracy in points; `weights` over items (bootstrap counts)."""
        if weights is None:
            return {a: 100.0 * c.mean(-1) for a, c in self.corr.items()}
        w = np.asarray(weights, float)
        return {a: 100.0 * (c @ w) / w.sum() for a, c in self.corr.items()}


def gains(acc: Mapping[str, np.ndarray]) -> dict:
    """{"req"|"fd": (configs, triggers, clauses, models)}: chain of thought minus the direct arm."""
    cot = acc[COT][:, :, None, :]
    return {"req": cot - acc[REQ][:, None, :, :], "fd": cot - acc[FD][:, None, :, :]}


# --------------------------------------------------------------------------------------------
# 1. R when the wording is drawn
# --------------------------------------------------------------------------------------------

SCHEMES = ("fixed", "trigger_drawn", "clause_drawn", "both_drawn")


def scheme_cells(shape, scheme: str, t0: int = 0, d0: int = 0) -> list[tuple[int, int, int]]:
    C, T, D = shape[:3]
    ts = range(T) if scheme in ("trigger_drawn", "both_drawn") else [t0]
    ds = range(D) if scheme in ("clause_drawn", "both_drawn") else [d0]
    return [(c, t, d) for c in range(C) for t in ts for d in ds]


def scheme_pair_mask(cells: Sequence[tuple], scheme: str) -> np.ndarray:
    """Ordered pairs (k, l), k != l, with distinct configurations, and distinct triggers or clauses where drawn."""
    a = np.array(cells)
    ok = a[:, None, 0] != a[None, :, 0]
    if scheme in ("trigger_drawn", "both_drawn"):
        ok &= a[:, None, 1] != a[None, :, 1]
    if scheme in ("clause_drawn", "both_drawn"):
        ok &= a[:, None, 2] != a[None, :, 2]
    return ok


def scheme_R(G: np.ndarray, scheme: str, t0: int = 0, d0: int = 0, configs: Sequence[int] | None = None) -> dict:
    """Signal over total over the scheme's cells (module doc; analysis 1). G: (configs, triggers, clauses, models)."""
    if configs is not None:
        G = G[list(configs)]
    cells = scheme_cells(G.shape, scheme, t0, d0)
    X = np.array([G[c, t, d] for c, t, d in cells])
    M = X.shape[1]
    Xc = X - X.mean(axis=1, keepdims=True)
    Cm = Xc @ Xc.T / (M - 1)
    mask = scheme_pair_mask(cells, scheme)
    n_pairs = int(mask.sum())
    signal = float(Cm[mask].mean()) if n_pairs else float("nan")
    total = float(np.mean(np.diag(Cm)))
    R = signal / total if total > 0 else float("nan")
    return {"R": R, "signal": signal, "total": total, "n_cells": len(cells), "n_ordered_pairs": n_pairs}


def scheme_C(G: np.ndarray, scheme: str, t0: int = 0, d0: int = 0, configs: Sequence[int] | None = None) -> float:
    """Mean one-draw agreement (ties one half) over the scheme's unordered pairs of cells."""
    if configs is not None:
        G = G[list(configs)]
    cells = scheme_cells(G.shape, scheme, t0, d0)
    mask = np.triu(scheme_pair_mask(cells, scheme), 1)
    k, l = np.nonzero(mask)
    if not len(k):
        return float("nan")
    return float(np.mean([pc.pair_c(G[cells[a]], G[cells[b]]) for a, b in zip(k, l)]))


def gaussian_ref(R: float) -> dict:
    """gaussian_agreement and prompts_needed_gaussian when R lies in [0, 1]; otherwise None, with the reason."""
    if not np.isfinite(R) or not 0 <= R <= 1:
        return {"gaussian_C": None, "prompts_needed_0.9": None, "note": f"R = {R} lies outside [0, 1]; reported as it is"}
    return {"gaussian_C": est.gaussian_agreement(R), "prompts_needed_0.9": est.prompts_needed_gaussian(R, 0.9)}


def r_table(G: Mapping[str, np.ndarray], configs: Sequence[int] | None = None) -> dict:
    out = {}
    for base, g in G.items():
        out[base] = {}
        for s in SCHEMES:
            r = scheme_R(g, s, configs=configs)
            out[base][s] = {**r, "C_one_draw": scheme_C(g, s, configs=configs), **gaussian_ref(r["R"])}
    return out


def fixed_any_wording(G: np.ndarray) -> dict:
    T, D = G.shape[1], G.shape[2]
    v = [scheme_R(G, "fixed", t, d)["R"] for t in range(T) for d in range(D)]
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}


def leave_one_config_out(G: np.ndarray, scheme: str) -> dict:
    C = G.shape[0]
    v = [scheme_R(G, scheme, configs=[c for c in range(C) if c != x])["R"] for x in range(C)]
    return {"min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "values": v}


def item_bootstrap_R(ta: TaskArrays, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(ta.items)
    vals = {b: {s: [] for s in SCHEMES} for b in ("req", "fd")}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        G = gains(ta.acc(w))
        for b in vals:
            for s in SCHEMES:
                vals[b][s].append(scheme_R(G[b], s)["R"])
    return {b: {s: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for s, v in d.items()}
            for b, d in vals.items()}


# --------------------------------------------------------------------------------------------
# 2. Pair reversals
# --------------------------------------------------------------------------------------------

def reversal(a, b) -> dict:
    """Between two evaluations over models: 1 - C (ties one half), the strict reversed share and the tie share."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    sa, sb = est.tie_sign(a[i] - a[j]), est.tie_sign(b[i] - b[j])
    tie = (sa == 0) | (sb == 0)
    rev = (sa == -sb) & ~tie
    same = (sa == sb) & ~tie
    return {"one_minus_C": float(1 - (same.mean() + 0.5 * tie.mean())), "reversed": float(rev.mean()),
            "tied": float(tie.mean())}


def _mean_rev(pairs) -> dict:
    rs = [reversal(a, b) for a, b in pairs]
    if not rs:
        return {"one_minus_C": float("nan"), "reversed": float("nan"), "tied": float("nan"), "n": 0}
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {"n": len(rs)}


def reversal_table(G: Mapping[str, np.ndarray], configs: Sequence[int] | None = None) -> dict:
    Gr, Gf = G["req"], G["fd"]
    if configs is not None:
        Gr, Gf = Gr[list(configs)], Gf[list(configs)]
    C, T, D, _ = Gr.shape
    cs, ts, ds = range(C), range(T), range(D)
    out = {"arm_switch_by_clause": [_mean_rev([(Gr[c, t, d], Gf[c, t, d]) for c in cs for t in ts]) for d in ds]}
    out["arm_switch"] = _mean_rev([(Gr[c, t, d], Gf[c, t, d]) for c in cs for t in ts for d in ds])
    out["clause_switch_requested"] = _mean_rev([(Gr[c, t, d], Gr[c, t, e]) for c in cs for t in ts
                                                for d in ds for e in ds if d < e])
    out["clause_switch_forced"] = _mean_rev([(Gf[c, t, d], Gf[c, t, e]) for c in cs for t in ts
                                             for d in ds for e in ds if d < e])
    out["trigger_switch_requested"] = _mean_rev([(Gr[c, t, d], Gr[c, u, d]) for c in cs for d in ds
                                                 for t in ts for u in ts if t < u])
    out["configuration_switch_requested"] = _mean_rev([(Gr[c, t, d], Gr[e, t, d]) for t in ts for d in ds
                                                       for c in cs for e in cs if c < e])
    out["arm_and_clause_switch"] = _mean_rev([(Gr[c, t, d], Gf[c, t, e]) for c in cs for t in ts
                                              for d in ds for e in ds if d != e])
    return out


def direct_gap(acc: Mapping[str, np.ndarray], dir_ids: Sequence[str], long_share: Mapping[str, float]) -> list[dict]:
    """Per clause: the panel's mean requested - forced accuracy over configurations, its range over models, and the
    share of requested-direct outputs longer than the analysis's length."""
    out = []
    for k, w in enumerate(dir_ids):
        per_model = (acc[REQ][:, k, :] - acc[FD][:, k, :]).mean(0)
        out.append({"clause": w, "mean_gap_pp": float(per_model.mean()), "min_model_gap_pp": float(per_model.min()),
                    "max_model_gap_pp": float(per_model.max()), "long_requested_share": long_share.get(w)})
    return out


# --------------------------------------------------------------------------------------------
# 3. Protocol ceiling with wording as a factor
# --------------------------------------------------------------------------------------------

def protocols(G: Mapping[str, np.ndarray]) -> list[tuple[str, int, int]]:
    _, T, D, _ = G["req"].shape
    return [(b, t, d) for b in ("req", "fd") for t in range(T) for d in range(D)]


def axis_of(p, q) -> str:
    diff = [p[0] != q[0], p[1] != q[1], p[2] != q[2]]
    names = {(False, True, False): "trigger", (False, False, True): "clause", (True, False, False): "arm",
             (True, False, True): "arm_and_clause", (False, True, True): "trigger_and_clause",
             (True, True, False): "arm_and_trigger", (True, True, True): "all_three"}
    return names[tuple(diff)]


def ceiling_table(G: Mapping[str, np.ndarray], seed: int, n_splits: int = 200, n_boot: int = 2000) -> dict:
    P = protocols(G)
    C = G["req"].shape[0]
    rng = np.random.default_rng(seed)
    Y = {p: G[p[0]][:, p[1], p[2], :] for p in P}
    within = {f"{p[0]}/t{p[1]}/d{p[2]}": pc.within_curve(Y[p], range(1, C // 2 + 1), rng, n_splits) for p in P}
    by_axis: dict[str, list] = {}
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            p, q = P[a], P[b]
            rec = {"curve": pc.between_curve(Y[p], Y[q], range(1, C + 1), rng, n_splits),
                   "limit_plugin": pc.limit_plugin(Y[p], Y[q]), "limit_posterior": pc.limit_posterior(Y[p], Y[q])}
            if n_boot:
                rec["limit_ci95"] = pc.bootstrap_limit(Y[p], Y[q], rng, n_boot)["ci95"]
            by_axis.setdefault(axis_of(p, q), []).append(rec)
            by_axis.setdefault("random", []).append(rec)
    summ = {}
    for ax, recs in by_axis.items():
        summ[ax] = {"pairs": len(recs),
                    "curve": {K: float(np.mean([r["curve"][K] for r in recs])) for K in recs[0]["curve"]},
                    "limit_plugin": float(np.mean([r["limit_plugin"] for r in recs])),
                    "limit_posterior": float(np.mean([r["limit_posterior"] for r in recs]))}
        if n_boot:
            summ[ax]["limit_ci95_mean_bounds"] = [float(np.mean([r["limit_ci95"][0] for r in recs])),
                                                  float(np.mean([r["limit_ci95"][1] for r in recs]))]
    wk = sorted({K for v in within.values() for K in v})
    summ["within"] = {"protocols": len(P), "curve": {K: float(np.mean([v[K] for v in within.values()])) for K in wk}}
    return summ


# --------------------------------------------------------------------------------------------
# 4. Decision cost
# --------------------------------------------------------------------------------------------

def decision_choices(acc_disc: Mapping[str, np.ndarray], n: int) -> dict:
    """From the discovery configurations alone: for each pilot configuration, the top-third weights chosen under the
    current protocol (requested arm, current trigger and clause), under every protocol, and by the one-draw
    capability slope under the current protocol."""
    G = gains(acc_disc)
    P = protocols(G)
    C = G["req"].shape[0]
    out = {"n": n, "protocols": [list(p) for p in P], "pilots": []}
    for c in range(C):
        g_cur = G["req"][c, 0, 0]
        w_cur, tie_cur = decision.choose(g_cur, n)
        pred_cap, slope = decision.capability_one_draw(acc_disc[REQ][c, 0], g_cur)
        w_cap, tie_cap = decision.choose(pred_cap, n)
        own = []
        for p in P:
            w, tie = decision.choose(G[p[0]][c, p[1], p[2]], n)
            own.append({"weights": w.tolist(), "tied_at_cut": tie})
        out["pilots"].append({"config_index": c, "current": {"weights": w_cur.tolist(), "tied_at_cut": tie_cur},
                              "capability": {"weights": w_cap.tolist(), "tied_at_cut": tie_cap,
                                             "slope": None if not np.isfinite(slope) else float(slope)},
                              "own": own})
    return out


def decision_evaluate(choices: Mapping, G_val: Mapping[str, np.ndarray]) -> dict:
    """Plug-in regret and extra regret (current wording over the evaluation's own protocol) on the validation
    configurations, per protocol and grouped by what differs from the current protocol."""
    n = int(choices["n"])
    P = [tuple(p) for p in choices["protocols"]]
    cur = ("req", 0, 0)
    per_p, groups = [], {}
    for k, p in enumerate(P):
        Gv = G_val[p[0]][:, p[1], p[2], :]
        Wc = np.array([pl["current"]["weights"] for pl in choices["pilots"]])
        Wo = np.array([pl["own"][k]["weights"] for pl in choices["pilots"]])
        Wk = np.array([pl["capability"]["weights"] for pl in choices["pilots"]])
        vc = decision.values(Wc.mean(0), Gv, n)[0]
        vo = decision.values(Wo.mean(0), Gv, n)[0]
        rec = {"protocol": list(p), "axis": "same" if p == cur else axis_of(cur, p),
               "regret_current": float(decision.regret_plugin(Wc, Gv, n).mean()),
               "regret_own": float(decision.regret_plugin(Wo, Gv, n).mean()),
               "regret_capability": float(decision.regret_plugin(Wk, Gv, n).mean()),
               "extra_regret_current_over_own": decision.t_interval(vo - vc)}
        per_p.append(rec)
        groups.setdefault(rec["axis"], []).append(vo - vc)
    grouped = {ax: {"protocols": len(v), **decision.t_interval(np.mean(v, axis=0))} for ax, v in groups.items()}
    return {"per_protocol": per_p, "grouped": grouped}


# --------------------------------------------------------------------------------------------
# Discovery quantities saved before validation, and the verdicts
# --------------------------------------------------------------------------------------------

def discovery_quantities(acc_disc: Mapping[str, np.ndarray], n_third: int) -> dict:
    G = gains(acc_disc)
    R_both = scheme_R(G["req"], "both_drawn")["R"]
    rt = r_table(G)
    pred = gaussian_ref(R_both)["gaussian_C"]
    switch = reversal_table(G)["arm_switch_by_clause"]
    return {"R_table": rt, "predicted_C_both_drawn_validation": pred,
            "arm_switch_share_by_clause": [s["one_minus_C"] for s in switch],
            "decision": decision_choices(acc_disc, n_third)}


def same_quantities(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_quantities(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_quantities(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def verdicts(per_task: Mapping[str, dict], thr: Mapping) -> dict:
    """The fixed statements on the full design (all six configurations)."""
    one, sw, wording_moves, dec = {}, {}, {}, {}
    lo, hi = thr["switch_share_band"]
    tol = float(thr["switch_share_clause_tolerance"])
    for task, r in per_task.items():
        R = r["R"]["req"]["both_drawn"]["R"]
        need = gaussian_ref(R)["prompts_needed_0.9"]
        one[task] = {"R_both_drawn": R, "prompts_needed_0.9": need, "holds": need == 1}
        by = [x["one_minus_C"] for x in r["reversals"]["arm_switch_by_clause"]]
        mean = float(np.mean(by))
        sw[task] = {"by_clause": by, "mean": mean, "band": [lo, hi], "current": by[0],
                    "holds": bool(lo <= mean <= hi and all(abs(v - by[0]) <= tol for v in by))}
        cl, arm = r["reversals"]["clause_switch_requested"]["one_minus_C"], r["reversals"]["arm_switch"]["one_minus_C"]
        wording_moves[task] = {"clause_switch": cl, "arm_switch": arm, "clause_at_least_arm": bool(cl >= arm)}
        g = r["decision"]["grouped"]
        word = [g[a]["estimate"] for a in ("trigger", "clause", "trigger_and_clause") if a in g]
        m = float(np.mean(word)) if word else float("nan")
        dec[task] = {"mean_extra_regret_wording_only_pp": m, "threshold_pp": thr["decision_extra_regret_pp"],
                     "adds_wording": bool(np.isfinite(m) and m >= float(thr["decision_extra_regret_pp"]))}
    return {"one_prompt": {"per_task": one, "abstract_changes": not all(v["holds"] for v in one.values())},
            "arm_switch": {"per_task": sw, "abstract_changes": not all(v["holds"] for v in sw.values())},
            "clause_moves_as_much_as_arm": {"per_task": wording_moves,
                                            "abstract_must_say": any(v["clause_at_least_arm"] for v in wording_moves.values())},
            "decision": {"per_task": dec, "sentence_gains_wording": any(v["adds_wording"] for v in dec.values())}}
