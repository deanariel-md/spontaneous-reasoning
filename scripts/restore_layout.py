r"""Restore, from an unpacked data deposit, the data files at the paths and in the formats the analyses read.

    python scripts/restore_layout.py DEPOSIT                 # write everything under data/ and results/
    python scripts/restore_layout.py DEPOSIT --list          # the files it would write, by group
    python scripts/restore_layout.py DEPOSIT --only inputs   # one group or several (--only panel forced)

DEPOSIT is the unpacked deposit folder (the one holding README.md, SOURCES.csv and the collection folders). Run it from
the repository root in two steps. First `--only inputs`, which copies the analysis inputs (among them the HotpotQA
token lengths that scripts/fetch_hotpot.py checks). Then, once the benchmark files are in place (scripts/fetch_bbh.py,
fetch_math.py, fetch_hotpot.py and the GSM8K test file; see README.md), the whole run, because the scored tables'
labels carry the hash of the item tables.

The deposit stores one folder per collection: the outputs of all models in one table, the prompts once (they are the
same for every model), and the score tables. The analyses read the layout of the study's own runs, which this script
writes back:

- score tables at the scorers' paths, split per model where the scorer wrote one file per model;
- the labels the analyses check before they read a scored table (Parquet key-value metadata d2_*: kind main, split
  test, the scorer record of manifests/<benchmark>_scorer.json, the item tables' hash, and one entry per model with the
  sampler and bank its requests were built with). They are rebuilt from the manifests of this code release and the
  deposit's models/runs.csv; run ids, fingerprints and creation times are not in the deposit and are left out;
- request keys (request_id, <template>|<arm>|<item> with the suffixes some designs add) where a reader joins on them;
- model outputs as generation folders (one shard per model) where a reader opens the folders;
- the analysis inputs that no script of this release can compute from the deposit (deposit folder analysis_inputs/,
  copied to the paths in its index.json).

Every file written is listed in data/restore_record.json with its source in the deposit and its sha256. Existing files
are kept unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

KEY = ["model_key", "arm", "config_id", "item_id"]
RECORD = "data/restore_record.json"


class RestoreError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ small helpers

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ctx:
    """The deposit, the output root and the record of what was written."""

    def __init__(self, deposit: Path, root: Path, force: bool):
        self.dep = deposit
        self.root = root
        self.force = force
        self.written: list[dict] = []
        self._cache: dict = {}
        runs = pd.read_csv(deposit / "models" / "runs.csv", dtype=str, keep_default_na=False)
        self.runs = runs

    def read(self, rel: str, columns=None) -> pd.DataFrame:
        p = self.dep / rel
        if not p.is_file():
            raise RestoreError(f"the deposit lacks {rel}")
        return pd.read_parquet(p, columns=columns)

    def out(self, rel: str) -> Path:
        return self.root / rel

    def skip(self, rel: str) -> bool:
        return self.out(rel).exists() and not self.force

    def note(self, rel: str, sources: list[str], extra: dict | None = None):
        p = self.out(rel)
        rec = {"file": rel, "from": sources, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        if extra:
            rec.update(extra)
        self.written.append(rec)

    def write_parquet(self, df: pd.DataFrame, rel: str, sources: list[str], meta: dict | None = None):
        p = self.out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        t = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
        if meta:
            md = dict(t.schema.metadata or {})
            md.update({k.encode(): str(v).encode("utf-8") for k, v in meta.items()})
            t = t.replace_schema_metadata(md)
        tmp = p.with_name(p.name + ".tmp")
        pq.write_table(t, tmp, compression="zstd")
        os.replace(tmp, p)
        self.note(rel, sources, {"rows": int(len(df))})

    def write_json(self, obj, rel: str, sources: list[str]):
        p = self.out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        self.note(rel, sources)

    def write_text(self, text: str, rel: str, sources: list[str]):
        p = self.out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        self.note(rel, sources)

    def folder_models(self, collection: str, prefix: str) -> list[str]:
        """Model keys of the generation folders under `prefix` (models/runs.csv, column folder)."""
        r = self.runs[(self.runs["collection"] == collection) & self.runs["folder"].str.startswith(prefix + "/")]
        return sorted(set(r["folder"].str[len(prefix) + 1:]))

    def gpu_of(self, folder: str) -> list[str]:
        r = self.runs[self.runs["folder"] == folder]
        out = []
        for g in r["gpu"]:
            for x in g.split(";"):
                if x and x not in out:
                    out.append(x)
        return out

    def requests_shas(self, folder: str) -> list[str]:
        r = self.runs[self.runs["folder"] == folder]
        return sorted({x for x in r["requests_sha256"] if x})

    def n_runs(self, folder: str) -> int:
        return int((self.runs["folder"] == folder).sum())


# ------------------------------------------------------------------------------------------------ labels (d2_* metadata)

TASK_MODULES = {"bbh": "bbh_d2", "math": "math_d2", "hotpot": "hotpot_d2"}


def scorer_record(task: str) -> dict:
    from src import d2_io
    if task == "gsm8k":
        _, rec = d2_io.load_registered_scorer(ROOT / "manifests" / "gsm8k_scorer.json")
        return rec
    import importlib
    tm = importlib.import_module("src." + TASK_MODULES[task])
    return tm.load_scorer(None)


def items_info(task: str, item_set: str = "target") -> dict:
    from src import d2_io
    if task == "gsm8k":
        split = "test" if item_set == "target" else "train"
        _, info = d2_io.load_items(split=split)
        return info
    import importlib
    tm = importlib.import_module("src." + TASK_MODULES[task])
    _, info = tm.load_items(item_set)
    return info


def sampler(task: str) -> dict:
    return json.loads((ROOT / "manifests" / f"sampler_{task}.json").read_text(encoding="utf-8"))


def d2_meta(ctx: Ctx, task: str, folders: list[tuple[str, str, int]], kind: str = "main",
            item_set: str = "target") -> dict:
    """The d2_* labels of a scored table. folders: (model_key, folder under data/d2_out, n_outputs)."""
    from src import d2_io
    sm = sampler(task) if kind == "main" else None
    rec = scorer_record(task)
    info = items_info(task, item_set if kind == "main" else "screening")
    inputs = []
    for mk, folder, n in folders:
        inputs.append({
            "model_key": mk, "work_dir": "data/d2_out/" + folder, "n_outputs": int(n),
            "requests_sha256": ctx.requests_shas(folder), "request_kind": kind,
            "split": "test" if kind == "main" else "train",
            "bank_sha256": (sm or {}).get("bank_sha256"),
            "sampler_registered_sha256": (sm or {}).get("sha256"),
            "test_backend": False, "decoding_confirmed": True, "incomplete_allowed": False,
            "gpu_type": ctx.gpu_of(folder)})
    meta = {"d2_schema_version": d2_io.SCHEMA_VERSION, "d2_kind": kind,
            "d2_split": "test" if kind == "main" else "train", "d2_purpose": d2_io.PURPOSES[kind],
            "d2_excluded_from_main": json.dumps(kind != "main"), "d2_scorer_status": rec["status"],
            "d2_scorer_version": rec["scorer_version"], "d2_scorer_record": json.dumps(rec, sort_keys=True),
            "d2_items": json.dumps(info, sort_keys=True), "d2_inputs": json.dumps(inputs, sort_keys=True),
            "d2_allow_flags": json.dumps({"allow_incomplete": False, "allow_test_backend": False,
                                          "allow_unmatched_requests": False}, sort_keys=True),
            "d2_restored": "scripts/restore_layout.py"}
    if task != "gsm8k":
        meta["d2_task"] = task
    return meta


def with_prompt_sha(ctx: Ctx, scores: pd.DataFrame, outputs_rel: str) -> pd.DataFrame:
    """Add prompt_sha256 to a score table from the outputs table (it was dropped from the scores as a duplicate)."""
    o = ctx.read(outputs_rel, columns=KEY + ["prompt_sha256"])
    o["item_id"] = o["item_id"].astype(scores["item_id"].dtype)
    m = scores.merge(o, on=KEY, how="left", validate="one_to_one")
    if m["prompt_sha256"].isna().any():
        raise RestoreError(f"{outputs_rel}: {int(m['prompt_sha256'].isna().sum())} score rows have no output row")
    return m


def request_id(config_id: str, arm: str, item_id) -> str:
    """The study's request key: <template>|<arm>|<item>. A forced request reuses its requested twin's prompt with the
    opening written into the reply, and its key is the twin's key with |fd added (scripts/build_forced_direct.py)."""
    if arm == "forced_direct":
        return f"{config_id}|direct_answer|{item_id}|fd"
    return f"{config_id}|{arm}|{item_id}"


GROUPS: dict[str, list] = {}


def job(group: str, *targets: str):
    def deco(fn):
        GROUPS.setdefault(group, []).append((targets, fn))
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ jobs

def per_model_folders(ctx: Ctx, df: pd.DataFrame, collection: str, prefix: str) -> list[tuple[str, str, int]]:
    n = df.groupby("model_key").size()
    models = ctx.folder_models(collection, prefix)
    missing = sorted(set(n.index) - set(models))
    if missing:
        raise RestoreError(f"{collection}/{prefix}: models {missing} have no generation folder in models/runs.csv")
    return [(m, f"{prefix}/{m}", int(n[m])) for m in sorted(n.index)]


def scored_table(ctx: Ctx, scores_rel: str, outputs_rel: str, target: str, task: str, collection: str,
                 prefix: str, models: list[str] | None = None):
    s = ctx.read(scores_rel)
    if models is not None:
        s = s[s["model_key"].isin(models)]
    s = with_prompt_sha(ctx, s, outputs_rel)
    meta = d2_meta(ctx, task, per_model_folders(ctx, s, collection, prefix))
    ctx.write_parquet(s.sort_values(KEY, kind="stable"), target, [scores_rel, outputs_rel], meta)


def copy_table(ctx: Ctx, src: str, target: str, models: list[str] | None = None):
    df = ctx.read(src)
    if models is not None:
        df = df[df["model_key"].isin(models)]
    ctx.write_parquet(df, target, [src])


P = "data/processed/"


@job("panel", P + "gsm8k/scored.parquet")
def j_gsm8k(ctx):
    scored_table(ctx, "gsm8k/scores.parquet", "gsm8k/outputs.parquet", P + "gsm8k/scored.parquet", "gsm8k",
                 "gsm8k", "collection", ctx.folder_models("gsm8k", "collection"))


@job("panel")
def j_gsm8k_ext(ctx):
    for m in ctx.folder_models("gsm8k", "collection_ext"):
        t = P + f"gsm8k_ext/scored_{m}.parquet"
        if ctx.skip(t):
            continue
        scored_table(ctx, "gsm8k/scores.parquet", "gsm8k/outputs.parquet", t, "gsm8k", "gsm8k", "collection_ext", [m])


for _b in ("bbh", "math"):
    def _f(ctx, b=_b):
        scored_table(ctx, f"{b}/scores.parquet", f"{b}/outputs.parquet", P + f"{b}/scored.parquet", b, b,
                     f"{b}/collection")
    job("panel", P + f"{_b}/scored.parquet")(_f)

for _b in ("gsm8k", "math", "bbh"):
    job("panel", P + f"content/{_b}_content.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"{b}/content_scores.parquet", P + f"content/{b}_content.parquet"))
    job("panel", P + f"output_checks/{_b}_first_answer.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"{b}/first_box_tokens.parquet", P + f"output_checks/{b}_first_answer.parquet"))

# forced baselines
_FORCED = {"gsm8k": "forced/gsm8k", "bbh": "forced/bbh", "math": "math/forced"}
for _b, _pre in _FORCED.items():
    job("forced", P + f"forced/{_b}_scored.parquet")(
        lambda ctx, b=_b, pre=_pre: scored_table(ctx, f"forced/{b}_scores.parquet", f"forced/{b}_outputs.parquet",
                                                 P + f"forced/{b}_scored.parquet", b, "forced", pre))
    job("forced", P + f"forced/{_b}_box.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"forced/{b}_box.parquet", P + f"forced/{b}_box.parquet"))
job("forced", P + "forced_constrained/bbh_scored.parquet")(
    lambda ctx: scored_table(ctx, "forced_constrained/bbh_scores.parquet", "forced_constrained/bbh_outputs.parquet",
                             P + "forced_constrained/bbh_scored.parquet", "bbh", "forced_constrained",
                             "forced_constrained/bbh"))
job("forced", P + "forced_constrained/bbh_box.parquet")(
    lambda ctx: copy_table(ctx, "forced_constrained/bbh_box.parquet", P + "forced_constrained/bbh_box.parquet"))

# five-shot and exemplars
for _b in ("gsm8k", "bbh", "math"):
    job("fewshot", P + f"fewshot/{_b}_scored.parquet")(
        lambda ctx, b=_b: scored_table(ctx, f"fewshot/{b}_scores.parquet", f"fewshot/{b}_outputs.parquet",
                                       P + f"fewshot/{b}_scored.parquet", b, "fewshot", f"fewshot/{b}"))
    job("fewshot", P + f"fewshot/{_b}_box.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"fewshot/{b}_box.parquet", P + f"fewshot/{b}_box.parquet"))
    job("fewshot", P + f"content/fewshot_{_b}_content.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"fewshot/{b}_content_scores.parquet", P + f"content/fewshot_{b}_content.parquet"))
job("exemplar", P + "exemplar/gsm8k_scored.parquet")(
    lambda ctx: scored_table(ctx, "exemplar/gsm8k_scores.parquet", "exemplar/gsm8k_outputs.parquet",
                             P + "exemplar/gsm8k_scored.parquet", "gsm8k", "exemplar", "exemplar/gsm8k"))
job("exemplar", P + "content/exemplar_gsm8k_content.parquet")(
    lambda ctx: copy_table(ctx, "exemplar/gsm8k_content_scores.parquet", P + "content/exemplar_gsm8k_content.parquet"))

# native reasoning
for _b in ("gsm8k", "math"):
    job("thinking", P + f"thinking/{_b}_scored.parquet")(
        lambda ctx, b=_b: scored_table(ctx, f"thinking/{b}_scores.parquet", f"thinking/{b}_outputs.parquet",
                                       P + f"thinking/{b}_scored.parquet", b, "thinking", f"thinking/{b}"))
    job("thinking", P + f"thinking/{_b}_closure.parquet")(
        lambda ctx, b=_b: copy_table(ctx, f"thinking/{b}_closure.parquet", P + f"thinking/{b}_closure.parquet"))
job("thinking", P + "content/thinking_gsm8k_content.parquet")(
    lambda ctx: copy_table(ctx, "thinking/gsm8k_content_scores.parquet", P + "content/thinking_gsm8k_content.parquet"))


# ------------------------------------------------------------------------------------------------ generation folders

OUTPUT_COLUMNS = ["request_id", "model_key", "arm", "config_id", "item_id", "prompt_sha256", "output_text",
                  "n_prompt_tokens", "n_output_tokens", "finish_reason", "cached"]


def with_cached(df: pd.DataFrame, after: str) -> pd.DataFrame:
    """The column `cached` (a row served from the generation cache), false on every row of the study's runs; the readers
    require it, the deposit leaves it out."""
    if "cached" in df.columns:
        return df
    df = df.copy()
    df.insert(list(df.columns).index(after) + 1, "cached", False)
    return df
# request-key suffixes per collection and arm (the request builders): a forced or five-shot forced request reuses its
# requested twin's prompt, and its key is the twin's key with the suffix.
SUFFIX = {"forced": {"forced_direct": "|fd"}, "math/forced": {"forced_direct": "|fd"},
          "fewshot": {"forced_direct": "|fd|fs"}, "forced_constrained": {"forced_direct": "|fd|gc"},
          "hotpot": {"forced_direct": "|fd", "fewshot_forced": "|fs"},
          "fewshot_requested": {"direct_answer": "|fsr"}, "thinking": {"*": "|th"},
          "temp_check": {"forced_direct": "|fd"},
          "ladder2": {"forced_direct": "|fd", "fewshot_forced": "|fd|fs", "fewshot_requested": "|fsr"},
          "ladder2/hotpot": {"forced_direct": "|fd", "fewshot_forced": "|fs"}}


def request_ids(df: pd.DataFrame, family: str, split: str = "test", task_prefix: str | None = None) -> pd.Series:
    """The study's request keys for the rows of one generation folder (see SUFFIX)."""
    if family == "thinking_pilot":       # training items of both tasks: the item key without its task, split train
        item = df["item_id"].astype(str).str.split(":", n=1).str[-1]
        return "train|" + df["config_id"].astype(str) + "|direct_answer|" + item + "|th"
    suf = SUFFIX.get(family, {})
    arm_part, tail = [], []
    for arm in df["arm"]:
        s = suf.get(arm, suf.get("*", ""))
        arm_part.append("direct_answer" if s else arm)
        tail.append(s)
    rid = (df["config_id"].astype(str) + "|" + pd.Series(arm_part, index=df.index) + "|" + df["item_id"].astype(str)
           + pd.Series(tail, index=df.index))
    if task_prefix:
        rid = task_prefix + "|" + rid
    if split != "test":
        rid = split + "|" + rid
    return rid


def run_logs(ctx: Ctx) -> dict:
    if "run_logs" not in ctx._cache:
        idx = json.loads((ctx.dep / "analysis_inputs" / "index.json").read_text(encoding="utf-8"))
        if not any(x["file"] == "run_logs.jsonl" for x in idx):
            raise RestoreError("the deposit lacks analysis_inputs/run_logs.jsonl")
        by = {}
        with open(ctx.dep / "analysis_inputs" / "run_logs.jsonl", encoding="utf-8") as fh:
            for line in fh:
                x = json.loads(line)
                by.setdefault(x["folder"], []).append(x)
        ctx._cache["run_logs"] = by
    return ctx._cache["run_logs"]


def write_folder(ctx: Ctx, folder: str, rows: pd.DataFrame, family: str, split: str = "test",
                 task_prefix: str | None = None, source: str = ""):
    """data/d2_out/<folder>: the folder's outputs as one shard, and its run logs (analysis_inputs/run_logs.jsonl)."""
    base = f"data/d2_out/{folder}"
    d = ctx.out(base)
    d.mkdir(parents=True, exist_ok=True)
    for old in list(d.glob("shard-*.parquet")) + list(d.glob("run_log-*.json")):
        old.unlink()
    df = rows.copy()
    df.insert(0, "request_id", request_ids(df, family, split, task_prefix))
    if "cached" not in df.columns:
        df["cached"] = False
    cols = OUTPUT_COLUMNS + [c for c in df.columns if c not in OUTPUT_COLUMNS]
    df = df[[c for c in cols if c in df.columns]]
    mk = str(df["model_key"].iloc[0]) if len(df) else ""
    # the runner's shard metadata (src/generation.py), without run ids and fingerprints, which the deposit leaves out
    ctx.write_parquet(df, f"{base}/shard-restored-00000.parquet", [source],
                      {"model_key": mk, "schema_version": "1", "throughput": "false"})
    logs = run_logs(ctx).get(folder)
    if not logs:
        raise RestoreError(f"analysis_inputs/run_logs.jsonl has no run log for {folder}")
    for x in logs:
        rec = dict(x["record"])
        rec["shards_written"] = ["shard-restored-00000.parquet"] if x["log"] == len(logs) - 1 else []
        ctx.write_json(rec, f"{base}/run_log-{x['log']:03d}.json", ["analysis_inputs/run_logs.jsonl"])


def folders_of(ctx: Ctx, outputs_rel: str, family: str, prefix_of, split: str = "test", task_prefix=None,
               columns=None):
    """Split a deposit outputs table into generation folders. prefix_of(model_key, row frame) -> folder."""
    o = ctx.read(outputs_rel, columns=columns)
    for mk, g in o.groupby("model_key", sort=True):
        for folder, gg in prefix_of(mk, g):
            write_folder(ctx, folder, gg.reset_index(drop=True), family, split, task_prefix, outputs_rel)


def simple(prefix):
    return lambda mk, g: [(f"{prefix}/{mk}", g)]


_FOLDERS = {
    "forced/gsm8k_outputs.parquet": ("forced", simple("forced/gsm8k")),
    "forced/bbh_outputs.parquet": ("forced", simple("forced/bbh")),
    "forced/math_outputs.parquet": ("math/forced", simple("math/forced")),
    "fewshot/gsm8k_outputs.parquet": ("fewshot", simple("fewshot/gsm8k")),
    "fewshot/bbh_outputs.parquet": ("fewshot", simple("fewshot/bbh")),
    "fewshot/math_outputs.parquet": ("fewshot", simple("fewshot/math")),
    "forced_constrained/bbh_outputs.parquet": ("forced_constrained", simple("forced_constrained/bbh")),
    "arm_wording/outputs.parquet": ("arm_wording", simple("arm_wording/collection")),
    "thinking/pilot_outputs.parquet": ("thinking_pilot", simple("thinking/pilot")),
}
def _folders_job(src: str, fam: str, pre):
    def fn(ctx):
        folders_of(ctx, src, fam, pre)
    fn.__name__ = fn.__doc__ = f"generation folders of {src}"
    return fn


for _src, (_fam, _pre) in _FOLDERS.items():
    job("folders")(_folders_job(_src, _fam, _pre))


@job("folders")
def j_hotpot_folders(ctx):
    blocks = ctx.read("hotpot/scores.parquet", columns=KEY + ["block"])

    def by_block(mk, g):
        b = blocks[blocks["model_key"] == mk].drop(columns="model_key")
        m = g.merge(b, on=["arm", "config_id", "item_id"], how="left", validate="one_to_one")
        if m["block"].isna().any():
            raise RestoreError(f"hotpot {mk}: outputs without a block")
        return [(f"hotpot/block_{blk.lower()}/{mk}", x.drop(columns="block")) for blk, x in m.groupby("block")]
    folders_of(ctx, "hotpot/outputs.parquet", "hotpot", by_block)


@job("folders")
def j_ladder_folders(ctx):
    r = ctx.runs[ctx.runs["collection"] == "ladder"]
    run_of = {}
    for f, mk in zip(r["folder"], r["model_key"]):
        parts = f.split("/")                       # ladder2/<run>/<task>
        run_of[mk] = parts[1]
    for b in ("gsm8k", "math", "bbh", "hotpot"):
        fam = "ladder2/hotpot" if b == "hotpot" else "ladder2"
        o = ctx.read(f"ladder/{b}_outputs_local.parquet")
        for mk, g in o.groupby("model_key", sort=True):
            if mk not in run_of:
                raise RestoreError(f"ladder: no run folder for {mk} in models/runs.csv")
            write_folder(ctx, f"ladder2/{run_of[mk]}/{b}", g.reset_index(drop=True), fam, "test", None,
                         f"ladder/{b}_outputs_local.parquet")
        for mk in sorted(set(o["model_key"])):
            p = ctx.out(f"data/d2_out/ladder2/chosen/{_chosen_name(mk)}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(run_of[mk] + "\n", encoding="utf-8")


def _chosen_name(model_key: str) -> str:
    return model_key[:-3] if model_key.endswith("-dk") else model_key


@job("folders")
def j_panel_folders(ctx):
    """The main panel's GSM8K, MATH-500 and BIG-Bench Hard folders (the HotpotQA content analysis reads their replies)."""
    ext = set(ctx.folder_models("gsm8k", "collection_ext"))
    folders_of(ctx, "gsm8k/outputs.parquet", "gsm8k",
               lambda mk, g: [((f"collection_ext/{mk}" if mk in ext else f"collection/{mk}"), g)])
    folders_of(ctx, "math/outputs.parquet", "math", simple("math/collection"))
    folders_of(ctx, "bbh/outputs.parquet", "bbh", simple("bbh/collection"))


# ------------------------------------------------------------------------------------------------ analysis inputs

def analysis_index(ctx: Ctx) -> list[dict]:
    p = ctx.dep / "analysis_inputs" / "index.json"
    if not p.is_file():
        raise RestoreError("the deposit lacks analysis_inputs/index.json")
    return json.loads(p.read_text(encoding="utf-8"))


@job("inputs")
def j_inputs(ctx):
    """Copy each file of analysis_inputs/ whose restore_to is a single path; check its sha256."""
    for x in analysis_index(ctx):
        if "<" in x["restore_to"] or x.get("how", "copy") != "copy":
            continue
        src = ctx.dep / "analysis_inputs" / x["file"]
        if sha256_file(src) != x["sha256"]:
            raise RestoreError(f"analysis_inputs/{x['file']} differs from its sha256 in index.json")
        dst = ctx.out(x["restore_to"])
        if dst.exists() and not ctx.force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        ctx.note(x["restore_to"], ["analysis_inputs/" + x["file"]])


@job("inputs")
def j_build_records(ctx):
    with open(ctx.dep / "analysis_inputs" / "build_records.jsonl", encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            ctx.write_json(x["record"], f"data/requests/{x['folder']}/build_record.json",
                           ["analysis_inputs/build_records.jsonl"])


def build_record(ctx: Ctx, folder: str) -> tuple[dict, str]:
    p = ctx.out(f"data/requests/{folder}/build_record.json")
    if not p.is_file():
        j_build_records(ctx)
    return json.loads(p.read_text(encoding="utf-8")), sha256_file(p)


# ------------------------------------------------------------------------------------------------ request files

def requests_file(ctx: Ctx, prompts_rel: str, model_key: str, target: str, family: str, split: str = "test",
                  arms=None):
    pr = ctx.read(prompts_rel)
    if arms is not None:
        pr = pr[pr["arm"].isin(arms)]
    pr = pr.copy()
    pr.insert(0, "model_key", model_key)
    pr.insert(0, "request_id", request_ids(pr, family, split))
    ctx.write_parquet(pr[["request_id", "model_key", "arm", "config_id", "item_id", "messages_json"]], target,
                      [prompts_rel])


@job("requests")
def j_gsm8k_requests(ctx):
    """GSM8K request files of the panel (the thinking-mode pilot measures the longest prompt from them)."""
    for m in ctx.folder_models("gsm8k", "collection"):
        requests_file(ctx, "gsm8k/prompts.parquet", m, f"data/requests/gsm8k/{m}.parquet", "gsm8k")
    for m in ctx.folder_models("gsm8k", "collection_ext"):
        requests_file(ctx, "gsm8k/prompts.parquet", m, f"data/requests/gsm8k_ext/collection_{m}/{m}.parquet", "gsm8k")


@job("requests")
def j_math_bbh_requests(ctx):
    """MATH-500 and BIG-Bench Hard request files of the panel (the Fig. 1a example reads the MATH-500 prompt)."""
    for task in ("math", "bbh"):
        for m in ctx.folder_models(task, f"{task}/collection"):
            requests_file(ctx, f"{task}/prompts.parquet", m, f"data/requests/{task}/{m}.parquet", task)


# ------------------------------------------------------------------------------------------------ HotpotQA

@job("hotpot", P + "hotpot/scored.parquet", P + "hotpot/scored_content.parquet", P + "hotpot/item_set.json")
def j_hotpot(ctx):
    o = ctx.read("hotpot/outputs.parquet", columns=KEY + ["prompt_sha256"])
    for src, target in (("hotpot/scores.parquet", "scored.parquet"), ("hotpot/content_scores.parquet",
                                                                       "scored_content.parquet")):
        s = ctx.read(src)
        m = s.merge(o, on=KEY, how="left", validate="one_to_one")
        if m["prompt_sha256"].isna().any():
            raise RestoreError(f"{src}: rows without an output")
        rid = request_ids(m, "hotpot")
        cols = list(s.columns)
        at = cols.index("n_output_tokens") + 1 if "n_output_tokens" in cols else len(cols)
        m = m[cols[:at] + ["prompt_sha256"] + cols[at:]]
        m.insert(at + 1, "request_id", rid)
        m = with_cached(m, "n_output_tokens")
        ctx.write_parquet(m, P + "hotpot/" + target, [src, "hotpot/outputs.parquet"])
    s = ctx.read("hotpot/scores.parquet", columns=["model_key", "block"])
    comp = {mk: bool((g["block"] == "B").any()) for mk, g in s.groupby("model_key", sort=False)}
    ctx.write_json({"items": "all", "rule": "all 1,000 target items when every model's block B folder holds a complete "
                    "run, otherwise block A", "block_b_complete": comp}, P + "hotpot/item_set.json",
                   ["hotpot/scores.parquet"])
    hotpot_meta(ctx)


def hotpot_meta(ctx: Ctx):
    """data/processed/hotpot/scored.json (the scored table's record: scorer, items, sampler and the generation folder of
    each model and block) and data/processed/hotpot/content_view/ (scored.parquet with strict_correct and strict_f1
    holding the content reading, as scripts/score_hotpot_content.py writes it, item_set.json and scored.json)."""
    sc = pd.read_parquet(ctx.out(P + "hotpot/scored.parquet"))
    rec, info = scorer_record("hotpot"), items_info("hotpot")
    folders = {}
    for mk, blk in sorted(set(zip(sc["model_key"], sc["block"]))):
        f = f"hotpot/block_{blk.lower()}/{mk}"
        logs = run_logs(ctx).get(f) or []
        dec = (logs[-1]["record"].get("decoding") or {}) if logs else {}
        prefill = dict(sorted((dec.get("prefill") or {}).items()))
        folders[f"{mk}/{blk}"] = {"path": f"data/d2_out/{f}", "gpu_type": ctx.gpu_of(f), "prefill": prefill,
                                  "n_outputs": int(((sc["model_key"] == mk) & (sc["block"] == blk)).sum()),
                                  "test_backend": False}
    base = {"stage": "all", "items": "all", "n_items": int(sc["item_id"].nunique()), "scorer": rec["scorer_version"],
            "scorer_rules_sha256": rec["rules_sha256"], "item_table": info["path"],
            "item_canonical_sha256": info["canonical_sha256"],
            "sampler_sha256": sha256_file(ROOT / "manifests" / "sampler_hotpot.json"), "configs": "all",
            "folders": folders, "n_rows": int(len(sc)),
            "by_arm": {a: int(n) for a, n in sc["arm"].value_counts().sort_index().items()}}
    ctx.write_json({**base, "sha256": sha256_file(ctx.out(P + "hotpot/scored.parquet"))}, P + "hotpot/scored.json",
                   ["hotpot/scores.parquet", "models/runs.csv"])
    c = pd.read_parquet(ctx.out(P + "hotpot/scored_content.parquet"))
    v = c[list(sc.columns)].copy()
    v["strict_correct"] = c["content_correct"].to_numpy()
    v["strict_f1"] = c["content_f1"].to_numpy()
    ctx.write_parquet(v, P + "hotpot/content_view/scored.parquet", ["hotpot/content_scores.parquet"])
    ctx.write_text(ctx.out(P + "hotpot/item_set.json").read_text(encoding="utf-8"),
                   P + "hotpot/content_view/item_set.json",
                   ["hotpot/scores.parquet"])
    from src import hotpot_task as ht
    ctx.write_json({**base, "view": "strict_correct holds content_correct and strict_f1 holds content_f1 "
                                    f"({ht.CONTENT_SCORER_VERSION}); lenient columns as run",
                    "from": "data/processed/hotpot/scored_content.parquet",
                    "sha256": sha256_file(ctx.out(P + "hotpot/content_view/scored.parquet"))},
                   P + "hotpot/content_view/scored.json", ["hotpot/content_scores.parquet"])


# ------------------------------------------------------------------------------------------------ rewording, five-shot requested

def sidecar(ctx: Ctx, collection: str, table: pd.DataFrame, scorers: dict, predictions: str,
            mechanism: str | None = None) -> dict:
    rec, rec_sha = build_record(ctx, f"{collection}/collection")
    confs = {}
    for t, c in rec["configurations"].items():
        confs[t] = sorted(set(c.get("discovery", [])) | set(c.get("validation", []))) if isinstance(c, dict) else c
    folders = {}
    for mk in sorted(set(table["model_key"])):
        f = f"{collection}/collection/{mk}"
        folders[mk] = {"folder": f"data/d2_out/{f}", "requests_sha256": (ctx.requests_shas(f) or [None])[0],
                       "gpu": ctx.gpu_of(f), "test_backend": False}
    md = {"stage": "all", "configurations": confs, "folders": folders, "collection_record_sha256": rec_sha,
          "predictions_sha256": sha256_file(ROOT / "manifests" / predictions)}
    if mechanism:
        md["mechanism_sha256"] = sha256_file(ROOT / "manifests" / mechanism)
    md.update({"synthetic_seed": None, "scorers": scorers, "restored": "scripts/restore_layout.py"})
    return md


@job("rewording", P + "arm_wording/all_scored.parquet")
def j_arm_wording(ctx):
    s = ctx.read("arm_wording/scores.parquet")
    s = with_prompt_sha(ctx, s, "arm_wording/outputs.parquet")
    ctx.write_parquet(s, P + "arm_wording/all_scored.parquet", ["arm_wording/scores.parquet", "arm_wording/outputs.parquet"])
    md = sidecar(ctx, "arm_wording", s, {"gsm8k": "manifests/gsm8k_scorer.json", "math": "manifests/math_scorer.json"},
                 "predictions_arm_wording.json", "predictions_arm_wording_mechanism.json")
    ctx.write_json(md, P + "arm_wording/all_scored.meta.json", ["analysis_inputs/build_records.jsonl", "models/runs.csv"])


@job("rewording", P + "fewshot_requested/all_scored.parquet")
def j_fewshot_requested(ctx):
    s = ctx.read("fewshot_requested/scores.parquet")
    o = ctx.read("fewshot_requested/outputs.parquet", columns=KEY + ["prompt_sha256"])
    o[["task", "config_id"]] = o["config_id"].str.split("/", n=1, expand=True)    # outputs key <benchmark>/<template>
    o["item_id"] = o["item_id"].astype(s["item_id"].dtype)
    s = s.merge(o, on=["model_key", "arm", "task", "config_id", "item_id"], how="left", validate="one_to_one")
    if s["prompt_sha256"].isna().any():
        raise RestoreError("fewshot_requested: score rows without an output row")
    ctx.write_parquet(s, P + "fewshot_requested/all_scored.parquet",
                      ["fewshot_requested/scores.parquet", "fewshot_requested/outputs.parquet"])
    md = sidecar(ctx, "fewshot_requested", s, {"gsm8k": "manifests/gsm8k_scorer.json", "bbh": "src/bbh.py (bbh-boxed)"},
                 "predictions_fewshot_requested.json")
    ctx.write_json(md, P + "fewshot_requested/all_scored.meta.json",
                   ["analysis_inputs/build_records.jsonl", "models/runs.csv"])


# ------------------------------------------------------------------------------------------------ larger models

LADDER_TASKS = ("gsm8k", "math", "bbh", "hotpot")
MISTRAL = "data/ladder_api/mistral-large-2512"
NEBIUS = "data/frontier_panel"


def ladder_family(task: str) -> str:
    return "ladder2/hotpot" if task == "hotpot" else "ladder2"


def _order(df: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    return df[[c for c in first if c in df.columns] + [c for c in df.columns if c not in first]]


@job("ladder")
def j_ladder_local_scores(ctx):
    """data/processed/ladder{,_uncut}/<task>/<model>.parquet: the local models' scores, one file per model. model_key is
    the model's name, run_key the generation folder's (data/d2_out/ladder2/<run_key>)."""
    for b in LADDER_TASKS:
        o = ctx.read(f"ladder/{b}_outputs_local.parquet", columns=KEY + ["prompt_sha256"])
        o["run_key"] = o["model_key"]                   # the outputs carry the run's key, the scores the model's name
        o["model_key"] = o["model_key"].map(_chosen_name)
        for reading, sub in (("cut", "ladder"), ("uncut", "ladder_uncut")):
            src = f"ladder/{b}_scores_local_{reading}.parquet"
            s = ctx.read(src)
            m = s.merge(o.astype({"item_id": s["item_id"].dtype}), on=KEY, how="left", validate="one_to_one")
            if m["prompt_sha256"].isna().any():
                raise RestoreError(f"{src}: rows without an output")
            m.insert(0, "request_id", request_ids(m, ladder_family(b)))
            m = with_cached(_order(m, ["request_id"] + KEY + ["prompt_sha256"]), "finish_reason")
            cols = [c for c in m.columns if c not in ("run_key", "reading")] + ["run_key", "reading"]
            m = m[cols]
            for mk, g in m.groupby("model_key", sort=True):
                ctx.write_parquet(g.reset_index(drop=True), f"{P}{sub}/{b}/{mk}.parquet",
                                  [src, f"ladder/{b}_outputs_local.parquet"])


def _int(v):
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _num(v):
    try:
        return None if v is None or pd.isna(v) else v
    except (TypeError, ValueError):
        return v


RAW_INT = {"status", "cap", "panel_cap", "n_prompt_tokens_local", "tries"}


def write_raw(ctx: Ctx, meta_file: str, tag: str, rel: str):
    """An API raw reply file (one JSON line per call) from analysis_inputs/<meta_file> and the reply texts of the
    deposit's ladder/<task>_outputs_<tag>.parquet."""
    meta = pd.read_parquet(ctx.dep / "analysis_inputs" / meta_file)
    texts = {}
    for b in LADDER_TASKS:
        o = ctx.read(f"ladder/{b}_outputs_{tag}.parquet", columns=["arm", "config_id", "item_id", "output_text"])
        texts.update({(b, c, a, str(i)): t for c, a, i, t in zip(o["config_id"], o["arm"], o["item_id"],
                                                                  o["output_text"])})
    fields = [c for c in meta.columns if not c.startswith("usage_") and c not in ("absent", "text_from")]
    order = ["key", "task", "request_id", "arm", "config_id", "item_id", "phase", "variant", "extra_params", "status",
             "text", "prefix_echoed", "finish_reason", "stop_reason", "usage", "cost_usd", "cap", "panel_cap",
             "n_prompt_tokens_local", "prompt_sha256", "tries", "seconds", "t", "error", "design"]
    p = ctx.out(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in meta.to_dict("records"):
            absent = set(filter(None, (r.get("absent") or "").split(",")))
            line = {}
            for k in order:
                if k in absent:
                    continue
                if k == "text":
                    if r["text_from"] == "outputs":
                        line[k] = texts[(r["task"], r["config_id"], r["arm"], str(r["item_id"]))]
                    else:
                        line[k] = r["text"] if r["text_from"] == "own" else None
                elif k == "usage":
                    line[k] = {u: _int(r[f"usage_{u}"]) for u in ("prompt_tokens", "completion_tokens",
                                                                   "cached_tokens")}
                elif k == "extra_params":
                    line[k] = json.loads(r[k]) if isinstance(r.get(k), str) else None
                elif k in fields:
                    v = r.get(k)
                    line[k] = _int(v) if k in RAW_INT else _num(v)
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
    ctx.note(rel, ["analysis_inputs/" + meta_file, f"ladder/<task>_outputs_{tag}.parquet"], {"lines": n})


def _raw_ids(ctx: Ctx, meta_file: str) -> pd.DataFrame:
    m = pd.read_parquet(ctx.dep / "analysis_inputs" / meta_file, columns=["task", "request_id", "arm", "config_id",
                                                                         "item_id"])
    return m.drop_duplicates(["task", "arm", "config_id", "item_id"])


def _with_raw_ids(df: pd.DataFrame, ids: pd.DataFrame, task: str) -> pd.DataFrame:
    i = ids[ids["task"] == task].drop(columns="task")
    m = df.merge(i.astype({"item_id": df["item_id"].dtype}), on=["arm", "config_id", "item_id"], how="left",
                 validate="many_to_one")
    if m["request_id"].isna().any():
        raise RestoreError(f"{task}: rows without a request in the raw replies")
    return m


@job("ladder")
def j_ladder_mistral(ctx):
    """Mistral Large 3: scored tables (data/ladder_api/mistral-large-2512/scored/<task>_{cut,uncut}.parquet) and the
    raw replies (raw.jsonl)."""
    ids = _raw_ids(ctx, "api_raw_mistral_large_3.parquet")
    for b in LADDER_TASKS:
        for reading in ("cut", "uncut"):
            src = f"ladder/{b}_scores_mistral_large_3_{reading}.parquet"
            s = _with_raw_ids(ctx.read(src), ids, b)
            s["key"] = b + "|" + s["request_id"]
            ctx.write_parquet(_order(s, ["request_id", "key"]), f"{MISTRAL}/scored/{b}_{reading}.parquet", [src])
    write_raw(ctx, "api_raw_mistral_large_3.parquet", "mistral_large_3", f"{MISTRAL}/raw.jsonl")


@job("ladder")
def j_ladder_deepseek(ctx):
    """DeepSeek V4.1-Flash (Nebius): outputs (data/frontier_panel/outputs/<task>.parquet), scored tables
    (scored/<task>_scored{,_uncut}.parquet) and the raw replies (nebius_panel_raw.jsonl)."""
    ids = _raw_ids(ctx, "api_raw_deepseek_v4.1_flash.parquet")
    for b in LADDER_TASKS:
        src = f"ladder/{b}_outputs_deepseek_v4.1_flash.parquet"
        o = with_cached(_order(_with_raw_ids(ctx.read(src), ids, b), ["request_id"]), "finish_reason")
        ctx.write_parquet(o, f"{NEBIUS}/outputs/{b}.parquet", [src])
        ps = o[["arm", "config_id", "item_id", "prompt_sha256"]]
        for reading, suffix in (("cut", ""), ("uncut", "_uncut")):
            ssrc = f"ladder/{b}_scores_deepseek_v4.1_flash_{reading}.parquet"
            s = _with_raw_ids(ctx.read(ssrc), ids, b)
            s = s.merge(ps, on=["arm", "config_id", "item_id"], how="left", validate="one_to_one")
            s = with_cached(_order(s, ["request_id"] + KEY + ["prompt_sha256"]), "finish_reason")
            ctx.write_parquet(s, f"{NEBIUS}/scored/{b}_scored{suffix}.parquet", [ssrc, src])
    write_raw(ctx, "api_raw_deepseek_v4.1_flash.parquet", "deepseek_v4.1_flash", f"{NEBIUS}/nebius_panel_raw.jsonl")


@job("ladder")
def j_ladder_requests(ctx):
    """data/requests/ladder/<task>/<model>.parquet: the ladder's prompts for each local model."""
    r = ctx.runs[ctx.runs["collection"] == "ladder"]
    runs = sorted({f.split("/")[1] for f in r["folder"]})
    for b in LADDER_TASKS:
        for run in runs:
            mk = _chosen_name(run)
            requests_file(ctx, f"ladder/{b}_prompts.parquet", mk, f"data/requests/ladder/{b}/{mk}.parquet",
                          ladder_family(b))


# ------------------------------------------------------------------------------------------------ Qwen4-Max

@job("qwen4_max")
def j_closed_w_raw(ctx):
    """data/frontier_panel/closed_w{,_full_math}/run_raw.jsonl from analysis_inputs/closed_w*_run_raw.parquet and the
    reply texts of qwen4_max/*.parquet."""
    for meta, outputs, rel in (("closed_w_run_raw.parquet", "qwen4_max/outputs_scored.parquet",
                                "data/frontier_panel/closed_w/run_raw.jsonl"),
                               ("closed_w_full_math_run_raw.parquet", "qwen4_max/math_full_outputs_scored.parquet",
                                "data/frontier_panel/closed_w_full_math/run_raw.jsonl")):
        o = ctx.read(outputs, columns=["task", "arm", "config_id", "item_id", "output_text"])
        texts = {(t, a, c, str(i)): x for t, a, c, i, x in zip(o["task"], o["arm"], o["config_id"], o["item_id"],
                                                               o["output_text"])}
        m = pd.read_parquet(ctx.dep / "analysis_inputs" / meta)
        p = ctx.out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for line, tf in zip(m["line"], m["text_from"]):
                x = json.loads(line)
                if tf == "outputs":
                    x["text"] = texts[(x.get("task"), x.get("arm"), x.get("config_id"), str(x.get("item_id")))]
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        ctx.note(rel, ["analysis_inputs/" + meta, outputs], {"lines": int(len(m))})


# ------------------------------------------------------------------------------------------------ temperature check

def _seed(request_id: str) -> int:
    return int(hashlib.sha256(("temp_check/t0.7|" + request_id).encode("utf-8")).hexdigest()[:8], 16) % 2 ** 31


@job("temp_check")
def j_temp_check(ctx):
    """data/d2_out/temp_check/run/<model>: the sampled MATH-500 replies at temperature 0.7, and their request files
    data/requests/temp_check/run/<model>.parquet (the greedy run's prompts with a seed per request)."""
    o = ctx.read("temp_check/outputs_scored.parquet")
    o = o.drop(columns=[c for c in ("strict_correct", "content_correct") if c in o.columns])
    for mk, g in o.groupby("model_key", sort=True):
        write_folder(ctx, f"temp_check/run/{mk}", g.reset_index(drop=True), "temp_check", "test", None,
                     "temp_check/outputs_scored.parquet")
    pr = ctx.read("math/prompts.parquet")
    pf = ctx.read("forced/math_prompts.parquet")
    prompts = pd.concat([pr[pr["arm"] != "forced_direct"], pf[pf["arm"] == "forced_direct"]], ignore_index=True)
    for mk, g in o.groupby("model_key", sort=True):
        req = g[["arm", "config_id", "item_id"]].merge(prompts[["arm", "config_id", "item_id", "messages_json"]],
                                                       on=["arm", "config_id", "item_id"], how="left",
                                                       validate="one_to_one")
        if req["messages_json"].isna().any():
            raise RestoreError(f"temp_check {mk}: requests without a prompt")
        req.insert(0, "model_key", mk)
        req.insert(0, "request_id", request_ids(req, "temp_check"))
        req["sampling_seed"] = req["request_id"].map(_seed).astype("int64")
        ctx.write_parquet(req[["request_id", "model_key", "arm", "config_id", "item_id", "messages_json",
                               "sampling_seed"]], f"data/requests/temp_check/run/{mk}.parquet",
                          ["temp_check/outputs_scored.parquet", "math/prompts.parquet", "forced/math_prompts.parquet"])


# ------------------------------------------------------------------------------------------------ reading audit

BENCH = {"gsm8k": "GSM8K", "math": "MATH-500", "bbh": "BIG-Bench Hard", "hotpot": "HotpotQA"}
KEY_COLS = ["gold", "requested_strict", "requested_content", "forced_strict", "forced_content", "cot_strict",
            "cot_content"]


@job("reading_audit")
def j_reading_audit(ctx):
    """results/reading_audit/: the sample (sample.jsonl, sample_key.jsonl), the two human raters' sheets
    (human_rater1.csv, human_rater2.csv, from reading_audit/human_labels.csv and the key in analysis_inputs) and the
    judge's rubric (rubric.md, the rubric and template between the markers scripts/reading_audit.py reads)."""
    base = "results/reading_audit/"
    s = ctx.read("reading_audit/sample.parquet")
    rid = s["sample_id"].str.split("|", n=3).str[3]
    lead = [c for c in s.columns if c not in KEY_COLS]
    smp = s[lead].copy()
    smp.insert(lead.index("item_id") + 1, "request_id", rid)
    for rel, frame in ((base + "sample.jsonl", smp), (base + "sample_key.jsonl", s[["sample_id"] + KEY_COLS])):
        rows = [{k: _plain(v) for k, v in r.items()} for r in frame.to_dict("records")]
        if rel.endswith("sample_key.jsonl"):     # rows without scores (no panel row to score) carry the gold answer only
            rows = [r if any(r[c] is not None for c in KEY_COLS[1:]) else {"sample_id": r["sample_id"], "gold": r["gold"]}
                    for r in rows]
        write_jsonl(ctx, rows, rel, ["reading_audit/sample.parquet"])
    key = pd.read_csv(ctx.out(base + "human_key.csv"))
    h = pd.read_csv(ctx.dep / "reading_audit" / "human_labels.csv")
    by = s.set_index("sample_id")
    for n, rater, col in ((1, "A", "hid_rater1"), (2, "B", "hid_rater2")):
        lab = h[h["rater"] == rater].set_index("sample_id")
        k = key[[col, "sample_id"]].sort_values(col)
        sheet = pd.DataFrame({"hid": k[col].to_numpy(),
                              "benchmark": [BENCH[by.loc[x, "task"]] for x in k["sample_id"]],
                              "prompt": [by.loc[x, "question"] for x in k["sample_id"]],
                              "reply": [by.loc[x, "reply"] for x in k["sample_id"]],
                              "label": [lab.loc[x, "label"] for x in k["sample_id"]],
                              "unsure": [lab.loc[x, "unsure"] for x in k["sample_id"]],
                              "note": None})
        p = ctx.out(base + f"human_rater{n}.csv")
        sheet.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
        ctx.note(base + f"human_rater{n}.csv", ["reading_audit/human_labels.csv", "reading_audit/sample.parquet",
                                                 "analysis_inputs/reading_audit_human_key.csv"])
    md = (ctx.dep / "reading_audit" / "judge_rubric.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)\n```", md, flags=re.S)
    if len(blocks) != 2:
        raise RestoreError("reading_audit/judge_rubric.md: expected two text blocks (rubric, template)")
    text = ("# Reading audit: the judge's rubric\n\n<!-- RUBRIC START -->\n" + blocks[0] + "\n<!-- RUBRIC END -->\n\n"
            "<!-- TEMPLATE START -->\n" + blocks[1] + "\n<!-- TEMPLATE END -->\n")
    ctx.write_text(text, base + "rubric.md", ["reading_audit/judge_rubric.md"])


def write_jsonl(ctx: Ctx, rows: list[dict], rel: str, sources: list[str]):
    p = ctx.out(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ctx.note(rel, sources, {"lines": len(rows)})


def _plain(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


# human audit of the scorers (two raters, then a third)
HUMAN_OUTPUTS = {"gsm8k": ["gsm8k/outputs.parquet", "forced/gsm8k_outputs.parquet"],
                 "bbh": ["bbh/outputs.parquet", "forced/bbh_outputs.parquet"],
                 "exemplar": ["exemplar/gsm8k_outputs.parquet"], "item_h": ["exemplar/gsm8k_outputs.parquet"]}


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


@job("audits", "data/scoring_audit_v2/key.csv")
def j_human_key(ctx):
    k = _csv(ctx.dep / "human_check" / "outputs_key.csv")
    k.insert(2, "request_id", [request_id(c, a, i) for c, a, i in zip(k["config_id"], k["arm"], k["item_id"])])
    p = ctx.out("data/scoring_audit_v2/key.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    k.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
    ctx.note("data/scoring_audit_v2/key.csv", ["human_check/outputs_key.csv"])


@job("audits", "results/scoring_audit_v2/ratings_two_raters.csv", "results/scoring_audit_v2/ratings_rater3.csv")
def j_human_ratings(ctx):
    r = _csv(ctx.dep / "human_check" / "ratings.csv")
    ab = r[r["rater"].isin(["A", "B"])]
    two = pd.DataFrame({"time_utc": "", "rater": ab["rater"].map({"A": "user1", "B": "user2"}), "set": ab["rater"],
                        "audit_id": ab["audit_id"], "event": ab["event"], "verdict": ab["verdict"],
                        "flags": ab["flags"]})
    p = ctx.out("results/scoring_audit_v2/ratings_two_raters.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    two.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
    ctx.note("results/scoring_audit_v2/ratings_two_raters.csv", ["human_check/ratings.csv"],
             {"note": "time_utc is not in the deposit and is written empty; rows keep the recorded order"})
    c = r[r["rater"] == "C"].reset_index(drop=True)
    texts = audit_texts(ctx, c["audit_id"].tolist())
    three = pd.DataFrame({"order": range(1, len(c) + 1), "audit_id": c["audit_id"]})
    three = three.merge(texts, on="audit_id", how="left")
    three["verdict"] = c["verdict"].values
    three["note"] = ""
    p = ctx.out("results/scoring_audit_v2/ratings_rater3.csv")
    three.to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\n")
    ctx.note("results/scoring_audit_v2/ratings_rater3.csv",
             ["human_check/ratings.csv", "human_check/outputs_key.csv", "items/*_items.csv", "*/outputs.parquet"],
             {"note": "question, reference answer and model output rebuilt from the items and outputs tables; note empty"})


def audit_texts(ctx: Ctx, ids: list[str]) -> pd.DataFrame:
    """question, reference_answer and model_output of audited answers, as the rating page showed them: the target
    item's question, its final answer, and the reply (a forced reply with the opening written into it)."""
    k = _csv(ctx.dep / "human_check" / "outputs_key.csv")
    k = k[k["audit_id"].isin(ids)]
    rows = []
    for task, g in k.groupby("task"):
        bench = "bbh" if task == "bbh" else "gsm8k"
        items = _csv(ctx.dep / "items" / f"{bench}_items.csv")
        items = items[items["set"] == "target"].set_index("item_id")
        o = pd.concat([ctx.read(f, columns=KEY + ["output_text"]) for f in HUMAN_OUTPUTS[task]])
        o = o.astype({"item_id": str})
        box = pd.concat([ctx.read(f"forced/{b}_box.parquet", columns=KEY + ["prefill"]) for b in ("gsm8k", "bbh")])
        box = box.astype({"item_id": str})
        m = g.merge(o, on=KEY, how="left").merge(box, on=KEY, how="left")
        for _, x in m.iterrows():
            it = items.loc[x["item_id"]]
            ref = str(it["gold_answer"]) if "gold_answer" in items.columns else str(it["answer"]).split("####")[-1].strip()
            pre = x["prefill"] if isinstance(x["prefill"], str) else ""
            rows.append({"audit_id": x["audit_id"], "question": str(it["question"]), "reference_answer": ref,
                         "model_output": pre + str(x["output_text"])})
    return pd.DataFrame(rows, columns=["audit_id", "question", "reference_answer", "model_output"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("deposit", type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--only", nargs="+", choices=sorted(GROUPS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    dep = a.deposit.resolve()
    if not (dep / "SOURCES.csv").is_file():
        raise SystemExit(f"{dep} is not an unpacked data deposit (no SOURCES.csv)")
    ctx = Ctx(dep, a.root.resolve(), a.force)
    groups = a.only or list(GROUPS)
    if a.list:
        for g in groups:
            for targets, fn in GROUPS[g]:
                for t in targets or [(fn.__doc__ or fn.__name__).strip().splitlines()[0]]:
                    print(g, t)
        return 0
    t0 = time.time()
    for g in groups:
        for targets, fn in GROUPS[g]:
            if targets and all(ctx.skip(t) for t in targets):
                print(f"kept   {g}: {targets[0]}" + (f" (+{len(targets) - 1})" if len(targets) > 1 else ""))
                continue
            t1 = time.time()
            fn(ctx)
            print(f"wrote  {g}: {targets[0] if targets else fn.__name__}" +
                  (f" (+{len(targets) - 1})" if len(targets) > 1 else "") + f"  {time.time() - t1:.1f}s", flush=True)
    rec_p = ctx.out(RECORD)
    old = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.is_file() else {"files": []}
    keep = {r["file"]: r for r in old.get("files", [])}
    keep.update({r["file"]: r for r in ctx.written})
    rec_p.parent.mkdir(parents=True, exist_ok=True)
    rec_p.write_text(json.dumps({"deposit": str(dep), "files": sorted(keep.values(), key=lambda r: r["file"])},
                                indent=1) + "\n", encoding="utf-8")
    print(f"restored {len(ctx.written)} files in {time.time() - t0:.0f}s; record {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
