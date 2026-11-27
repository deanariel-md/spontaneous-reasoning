r"""Size ladder: the analysis of every ladder model whose scored rows are present, set beside the main panel
(29 models; 28 on BIG-Bench Hard). Rerun it as models arrive; nothing is cached between runs.

Ladder models (six):
  API      mistral-large-2512        data/ladder_api/mistral-large-2512/scored/<task>_{cut,uncut}.parquet
                                     (scripts/mistral_ladder.py report)
           deepseek-v4.1-flash-1017  data/frontier_panel/scored/<task>_scored{,_uncut}.parquet
                                     (scripts/nebius_panel_report.py)
  local    llama-3.1-70b-instruct-fp8, qwen3.5-122b-a10b-fp8, ministral-3-8b-instruct-2512,
           ministral-3-14b-instruct-2512: data/processed/ladder/<task>/<key>.parquet (cut, primary) and
           data/processed/ladder_uncut/<task>/<key>.parquet (uncut), the layout scripts/score_ladder.py writes.
A model with no file is reported as missing ("waits for <model>"); a model with some benchmarks is reported as partial,
benchmark by benchmark. Nothing is dropped silently.

Every model is computed the same way, with the functions scripts/nebius_panel_report.py uses for DeepSeek (no new
estimator): per benchmark and reading, the (configuration, item) units with all three zero-shot arms
(complete_units); accuracy per configuration, then the mean over the sampler's 48 draws (manifests/sampler_<task>.json;
a configuration drawn twice counts twice) and over the 24 discovery draws (draw_mean); w, the share of requested
outputs longer than the forced cap (16 tokens; 64 on HotpotQA), per configuration and averaged over the discovery draws
(src/baseline_mixture.RULE_TEXT); the corrected gain (1 - w) x (CoT - forced) on the discovery draws; the template SD
over the distinct configurations and, beside it, the template variance with the item sampling removed by the panel's
noise-anatomy estimator written for one model (src/noise_anatomy.config_variance: minus the configuration x item
interaction mean square over the N items every configuration shares; template_block, reported as it
comes, negative included); the five-shot arms (fewshot_block). The DeepSeek values are checked against
results/frontier_panel/nebius_panel.json, value by value.

The main panel is read from its own results, never recomputed:
  GSM8K, MATH-500, BIG-Bench Hard  results/content_scoring/tables/compliance_<task>.csv (per model and arm, strict and
                                   content, means over the 24 discovery draws) and results/baseline_mixture/tables/
                                   per_model.csv (w on the discovery draws, gains, corrected gains);
  HotpotQA                         strict: results/hotpot/tables/per_model_cot_{req,forced}.csv (discovery draws) and
                                   manifests/predictions_hotpot_mechanism.json (w, corrected gain); content:
                                   results/hotpot_content/tables/per_model_arm.csv (pooled over every row of the 36
                                   configurations), with the corrected gain (1 - w) x the pooled content gain;
  readable answers                 the same compliance tables (requested, forced, CoT; no HotpotQA table);
  families                         results/hotpot/hotpot_forced.json (models, families).
A ladder model is placed against the panel on the panel file's own basis (discovery draws, or pooled rows for HotpotQA
content).

Completeness: every model's scored rows are counted against the design, per benchmark, arm and configuration: a locally run
model's own request files (data/requests/ladder/<task>/<key>.parquet), and for the two API runs the request files they
read (those of ministral-3-14b-instruct-2512); missing, duplicate and extra request ids, ids whose arm, configuration or
item differ, and unscored rows are listed, never dropped or filled in. A locally run model with no scored rows yet shows the
last count of its jobs (results/ladder/auto_ladder2.log, read only).

The stop's loss: the forced arms of GSM8K and BIG-Bench Hard stop at the first "}", and a reply the
stop left with its box open that src/forced_stop.py does not complete is unread (results/forced_stop/unread_by_config.csv
and audit.json, scripts/forced_stop_audit.py). Each family step of a content quantity that reads the zero-shot forced
arm (forced, CoT - forced, corrected gain) is judged again at both ends of the range those unread replies allow, all
counted correct or none (DIRECTION_RULE): a step whose word differs reads "<word> within the stop's loss", and section
1 lists it. Footnote F1 gives each ladder model's unread replies and the check against the API models' full replies
(results/forced_stop/full_replies.json, scripts/forced_stop_full_replies.py).

Footnote F1 cites results/ladder/analysis/bbh_forced_forms.json (scripts/ladder_bbh_forms.py) when it exists. The
BIG-Bench Hard five-shot forced gap (content - strict over all rows of the arm) is placed against the panel's gaps on
that basis, read from results/ladder/mistral_bbh_forced_check.json (panel_scan, scripts/check_mistral_bbh_forced.py:
every row of the panel's five-shot forced arm, scored with the same code, values given to 0.01 pp).
Records the text cites, read at run time, never typed in: each model's size (total and active parameters, dense or
mixture of experts) from the dataset inventory table (INVENTORY), the checkpoint's name and its pinned config.json, or, for the two
API models (Mistral Large 3, DeepSeek V4.1-Flash-1017), the developers' published figures recorded with their URL, read
date and quoted sentences in manifests/ladder_published_sizes.json; DeepSeek's predecessor record
(V4-Flash) is kept in ladder.json, marked not used; the
API spend from results/ladder/mistral_pilot.md and results/frontier_panel/nebius_panel.md; the panel's precision
from config/models.yaml and config/models_extension.yaml.

An independent recomputation of the numbers in ladder.json and of every printed number: scripts/
verify_ladder_analysis.py (written apart from this script).

Printing (ROUNDING_RULE): every number in ladder.md and ladder_note.md is formatted by one function, fnum, which rounds
half-even at the printed precision on the exact value. Every accuracy, gain, share, w, corrected gain and residual of
a ladder model is held as an exact ratio (fractions.Fraction) from the integer counts of its scored rows, checked
against the float of scripts/nebius_panel_report.py (1e-9); each panel value is read from its file as the ratio m/G on
its grid (G from the design: discovery draws x items per configuration, or the arm's rows for pooled values), after a
check that the file's float lies within 1e-6 grid steps of m/G. So 8.95 prints 9.0 and 8.85 prints 8.8. A template SD
is the square root of an exact variance and is rounded on that root exactly (Root). Any other float met while printing
within 1e-9 of a halfway point is listed as an edge case (section 12).

Writes results/ladder/analysis/ladder.json, ladder.md, ladder_clean.md, ladder_note.md and ladder_note_clean.md.

PowerShell, repository root:
    $env:PYTHONIOENCODING = "utf-8"
    .\.venv\Scripts\python.exe scripts\ladder_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nebius_panel as npl  # noqa: E402
import nebius_panel_report as npr  # noqa: E402

TASKS = ("gsm8k", "math", "bbh", "hotpot")
TASK_NAME = npr.TASK_NAME
READINGS = ("cut", "uncut")
SCORES = {"strict": "strict_correct", "content": "content_correct"}
ARMS3 = npl.ARMS                       # direct_answer, forced_direct, zero_shot_cot
FORCED_ARMS = ("forced_direct", "fewshot_forced")
RAISED_ARMS = ("direct_answer", "zero_shot_cot", "fewshot_requested")
DESIGN_ROWS = {   # per task and arm, the complete design (src/ladder.py REQUESTS; restated so the check is visible)
    "gsm8k": {"direct_answer": 38251, "zero_shot_cot": 38251, "forced_direct": 38251, "fewshot_forced": 38251,
              "fewshot_requested": 10552},
    "math": {"direct_answer": 12338, "zero_shot_cot": 12338, "forced_direct": 12338, "fewshot_forced": 12338},
    "bbh": {"direct_answer": 47250, "zero_shot_cot": 47250, "forced_direct": 47250, "fewshot_forced": 47250,
            "fewshot_requested": 10800},
    "hotpot": {a: 36000 for a in ("direct_answer", "zero_shot_cot", "forced_direct", "fewshot_forced")},
}
QUANTITIES = ("requested", "forced", "cot", "cot_minus_requested", "cot_minus_forced", "w", "corrected_gain")
Q_LABEL = {"requested": "Requested", "forced": "Forced", "cot": "CoT", "cot_minus_requested": "CoT − requested",
           "cot_minus_forced": "CoT − forced", "w": "w", "corrected_gain": "Corrected gain",
           "residual": "CoT − requested minus corrected gain"}
# Every quantity placed against the panel and printed with its placement (the glance tables and section 6): per score,
# the seven quantities and the residual; per zero-shot arm, the readable share (score key "readable", GSM8K, MATH-500
# and BIG-Bench Hard, where the panel has it). The cut-against-uncut comparison takes its list from the placements
# themselves, so a quantity that is placed is compared.
PLACED = QUANTITIES + ("residual",)
READ_ARMS = ("requested", "forced", "cot")

# --------------------------------------------------------------------------------------------------------------------
# The six ladder models, in the order they are reported
# --------------------------------------------------------------------------------------------------------------------
LADDER = {
    "ministral-3-8b-instruct-2512": {
        "name": "Ministral 3 8B Instruct 2512", "family": "Mistral", "ladder": "Mistral 3", "where": "cluster",
        "precision": "bfloat16", "route": "vLLM, 1 RTX PRO 6000"},
    "ministral-3-14b-instruct-2512": {
        "name": "Ministral 3 14B Instruct 2512", "family": "Mistral", "ladder": "Mistral 3", "where": "cluster",
        "precision": "bfloat16", "route": "vLLM, 1 RTX PRO 6000"},
    "mistral-large-2512": {
        "name": "Mistral Large 3 (2512)", "family": "Mistral", "ladder": "Mistral 3", "where": "api",
        "precision": "not stated by the provider (Mistral's API; open weights Mistral-Large-3-675B-Instruct-2512)",
        "route": "Mistral's batch API, prefix mode for the forced arms (scripts/mistral_ladder.py)",
        "paths": {"cut": "data/ladder_api/mistral-large-2512/scored/{task}_cut.parquet",
                  "uncut": "data/ladder_api/mistral-large-2512/scored/{task}_uncut.parquet"}},
    "llama-3.1-70b-instruct-fp8": {
        "name": "Llama-3.1-70B-Instruct", "family": "Llama", "ladder": "Llama 3.1", "where": "cluster",
        "precision": "FP8 (RedHatAI FP8-dynamic of Meta's release; KV cache 16-bit)",
        "route": "vLLM, 1 RTX PRO 6000"},
    "qwen3.5-122b-a10b-fp8": {
        "name": "Qwen3.5-122B-A10B", "family": "Qwen", "ladder": "Qwen3.5", "where": "cluster",
        "precision": "FP8 (Qwen's official block FP8; KV cache 16-bit)",
        "route": "vLLM, 2 RTX PRO 6000 (tensor parallel 2)"},
    "deepseek-v4.1-flash-1017": {
        "name": "DeepSeek V4.1-Flash-1017", "family": "DeepSeek", "ladder": None, "where": "api",
        "precision": "fp4 (NVFP4), provider-stated: " + npl.SERVED_AS,
        "route": "Nebius, real time, /v1/completions with the locally rendered prompt (scripts/nebius_panel.py)",
        "paths": {"cut": "data/frontier_panel/scored/{task}_scored.parquet",
                  "uncut": "data/frontier_panel/scored/{task}_scored_uncut.parquet"}},
}
OUTSIDE = "none (a single model outside the three families)"     # DeepSeek's place in the family ladders

# Sizes: total and active parameters and the architecture of each model, read from a
# recorded source at run time; nothing is filled in where the repository records nothing. Kinds:
#   inventory  the checkpoint's row in the inventory table (INVENTORY: "<total> dense" or "<total> / <active>");
#   name       the total in the checkpoint's name (<n>B), with the architecture from the config.json of the pinned
#              checkpoint in the local Hugging Face cache when it is there (expert layers: mixture of experts);
#   published  the developer's published figures, recorded with the page's URL, the read date and the quoted sentences
#              in PUBLISHED_SIZES; a number its quotes do not write is refused. A "predecessor"
#              entry adds the record of the model it replaces, marked not used.
INVENTORY = "manifests/dataset2_inventory.md"
PUBLISHED_SIZES = "manifests/ladder_published_sizes.json"
SIZE_SPEC = {
    "ministral-3-8b-instruct-2512": {"kind": "inventory", "hf_id": "mistralai/Ministral-3-8B-Instruct-2512"},
    "ministral-3-14b-instruct-2512": {"kind": "inventory", "hf_id": "mistralai/Ministral-3-14B-Instruct-2512"},
    "mistral-large-2512": {"kind": "published"},
    "llama-3.1-70b-instruct-fp8": {"kind": "name", "name": "Meta-Llama-3.1-70B-Instruct",
                                   "source": "the checkpoint's name",
                                   "config": ("RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8-dynamic",
                                              "e5718dbb1f3c4ec839a87d8e434954f6339d9a07")},
    "qwen3.5-122b-a10b-fp8": {"kind": "inventory", "hf_id": "Qwen/Qwen3.5-122B-A10B"},
    "deepseek-v4.1-flash-1017": {"kind": "published",
                                 "predecessor": {"of": "V4-Flash", "hf_id": "deepseek-ai/DeepSeek-V4-Flash",
                                                 "request": "the earlier ladder plan",
                                                 "recorded": ("284", "13")}},   # its words: 284B total, about 13B active
    "qwen3.5-9b": {"kind": "inventory", "hf_id": "Qwen/Qwen3.5-9B"},
    "qwen3.5-35b-a3b": {"kind": "inventory", "hf_id": "Qwen/Qwen3.5-35B-A3B"},
    "llama-3.1-8b-instruct": {"kind": "inventory", "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
}
# Spend, read from the run reports at run time.
SPEND_MISTRAL = "results/ladder/mistral_pilot.md"
SPEND_NEBIUS = "results/frontier_panel/nebius_panel.md"
SPEND_NEBIUS_BEFORE = "results/frontier_panel/nebius_realtime_spend.md"     # what the earlier spend was made of
# The panel's precision: dtype per panel model in the registries.
PANEL_CONFIGS = ("config/models.yaml", "config/models_extension.yaml")
DEFAULT_KERNEL_NOTE = ("the agreement of the cut copy with a rerun at the panel's caps on the "
                       "default-kernel model, flagged")
CLUSTER_PATHS = {"cut": "data/processed/ladder/{task}/{key}.parquet",
                 "uncut": "data/processed/ladder_uncut/{task}/{key}.parquet"}

# Family ladders: (key, source) from the smallest rung up. Panel rungs of another release are shown as context only.
FAMILY_LADDERS = {
    "Mistral 3": {"rungs": [("ministral-3-8b-instruct-2512", "ladder"), ("ministral-3-14b-instruct-2512", "ladder"),
                            ("mistral-large-2512", "ladder")],
                  "context": ["mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3", "mistral-nemo-instruct-2407",
                              "mistral-small-3.1-24b-instruct-2503"],
                  "context_note": "the panel's Mistral models, older releases (they stay in the panel; "
                                  "the ladder is Ministral 3 8B → 14B → Large 3)"},
    "Qwen3.5": {"rungs": [("qwen3.5-9b", "panel"), ("qwen3.5-35b-a3b", "panel"), ("qwen3.5-122b-a10b-fp8", "ladder")],
                "context": [], "context_note": ""},
    "Llama 3.1": {"rungs": [("llama-3.1-8b-instruct", "panel"), ("llama-3.1-70b-instruct-fp8", "ladder")],
                  "context": [], "context_note": ""},
}

# Rung labels by size, as the model names give them (Mistral Large 3: 675B).
SIZE_LABEL = {"ministral-3-8b-instruct-2512": "8B", "ministral-3-14b-instruct-2512": "14B",
              "mistral-large-2512": "Large 3 (675B)", "qwen3.5-9b": "9B", "qwen3.5-35b-a3b": "35B-A3B",
              "qwen3.5-122b-a10b-fp8": "122B-A10B", "llama-3.1-8b-instruct": "8B",
              "llama-3.1-70b-instruct-fp8": "70B", "mistral-7b-instruct-v0.1": "7B v0.1",
              "mistral-7b-instruct-v0.3": "7B v0.3", "mistral-nemo-instruct-2407": "Nemo 12B",
              "mistral-small-3.1-24b-instruct-2503": "Small 3.1 24B"}

PANEL_FILES = {
    "compliance": "results/content_scoring/tables/compliance_{task}.csv",
    "mixture": "results/baseline_mixture/tables/per_model.csv",
    "hotpot_req": "results/hotpot/tables/per_model_cot_req.csv",
    "hotpot_forced": "results/hotpot/tables/per_model_cot_forced.csv",
    "hotpot_mechanism": "manifests/predictions_hotpot_mechanism.json",
    "hotpot_content": "results/hotpot_content/tables/per_model_arm.csv",
    "families": "results/hotpot/hotpot_forced.json",
}
NEBIUS_JSON = "results/frontier_panel/nebius_panel.json"
# The design each model's scored rows are counted against: a locally run model's own request files; the two API runs read
# the ladder's request files of this key (scripts/mistral_ladder.py REQ_KEY, scripts/nebius_panel.py LADDER_REQ_KEY).
REQ_PATH = "data/requests/ladder/{task}/{key}.parquet"
API_DESIGN_KEY = "ministral-3-14b-instruct-2512"
ID_COLS = ["request_id", "arm", "config_id", "item_id"]
LOOP_LOG = "results/ladder/auto_ladder2.log"      # the loop's passes (jobs done per model), read only
FORMS_JSON = "results/ladder/analysis/bbh_forced_forms.json"   # scripts/ladder_bbh_forms.py (locally run models)
EQUIV_GLOB = "results/ladder/equiv_*.json"
PILOT_CHECK = "results/ladder/pilot_check_{key}.json"
SCORED_SUMMARY = "results/ladder/scored_{key}.json"
MISTRAL_CHECK_JSON = "results/ladder/mistral_bbh_forced_check.json"  # source of footnote F1 (Mistral Large 3)
NEBIUS_CHECK_JSON = "results/frontier_panel/nebius_forced_check.json"  # source of footnote F1 (DeepSeek)
NEBIUS_CHECK = "results/frontier_panel/nebius_forced_check_clean.md"  # source of footnote F2
# The panel's BIG-Bench Hard five-shot forced arm over all rows (panel_scan, entries "fewshot_forced|<model>"): strict
# and content in pp, rounded to 0.01 by scripts/check_mistral_bbh_forced.py (pct); its run logs give the panel arm's
# decoding (data/d2_out/fewshot/bbh/<model>/run_log-*.json).
FEWSHOT_PANEL = "results/ladder/mistral_bbh_forced_check.json"
FEWSHOT_PANEL_STEP = Fraction(1, 100)      # the file's rounding of each panel value (pp)
FEWSHOT_PANEL_LOGS = "data/d2_out/fewshot/bbh/*/run_log-*.json"
LADDER_BBH_DECODING = "config/decoding_ladder2_bbh.yaml"
OUT = "results/ladder/analysis"
TOL = 1e-9
# Direction over size (family tables): judged on the unrounded (exact) values; a step smaller in size than this is flat.
FLAT = {"pp": Fraction(1, 10), "w": Fraction(1, 1000)}
CHANGE_LISTED = Fraction(1, 20)  # section 9: changes between the readings listed from 0.05 pp
CUT_SHARE_LISTED = Fraction(1, 100)   # arms with 1% or more of their outputs cut
FORM_MIN = Fraction(1, 20)       # footnote F1: forms of 0.05 pp or more
ROUNDING_RULE = ("every printed number is rounded half-even at its printed precision on its exact value: ladder values "
                 "are exact ratios of the integer counts of the scored rows; panel values are the ratios m/G their "
                 "files' floats stand for on the design's grid (G: discovery draws x items per configuration, or the "
                 "arm's rows for pooled values); a template SD is rounded on the exact root of its exact variance")
DIRECTION_RULE = ("judged on the unrounded values, rung to rung: a step of 0.1 pp or more upwards is up, 0.1 pp or "
                  "more downwards is down, anything smaller is flat (for w, a share: 0.001). rises: every step up; "
                  "falls: every step down; flat: every step flat; otherwise the steps in order (for example "
                  "\"up, flat\"). A content step of the forced arm, CoT − forced or the corrected gain on GSM8K or "
                  "BIG-Bench Hard reads \"<word> within the stop's loss\" when the forced replies the stop at the "
                  "first `}` left unread (footnote F1) could give it another word: judged again at both ends of the "
                  "range those replies allow, each rung's unread share of its zero-shot forced arm (discovery draws) "
                  "counted all correct or none, the word differs.")
STOP_UNREAD = "results/forced_stop/unread_by_config.csv"   # scripts/forced_stop_audit.py, per configuration
STOP_AUDIT = "results/forced_stop/audit.json"               # scripts/forced_stop_audit.py, per model and arm
LATEST_FIX_REPORT = "results/ladder/analysis/verify.out"      # named in the checks paragraph of ladder.md
FULL_REPLIES = "results/forced_stop/full_replies.json"      # scripts/forced_stop_full_replies.py
STOP_TASKS = ("gsm8k", "bbh")                               # the benchmarks whose forced arms stop at the first "}"
# how a content value moves when the unread forced replies are counted correct: (low, high) multiples of the rung's
# unread share, times (1 - w) for the corrected gain and the residual
STOP_Q = {"forced": (0, 1), "cot_minus_forced": (-1, 0), "corrected_gain": (-1, 0), "residual": (0, 1)}
WITHIN_LOSS = " within the stop's loss"
PILOT_RULE_MIN = Fraction(9, 10)   # scripts/check_ladder_pilot.py: a boxed answer in at least 90% of requested rows
RESIDUAL_BAND = 2.0          # findings: content residuals within ±2 pp
READABLE_BELOW = 90.0        # findings: requested arms with fewer readable replies than this (%)
CUT_BIG = 1.0                # findings: changes between the readings of 1 pp or more
# Bases. The panel's files: discovery draws, except HotpotQA content (pooled rows). Every sentence and every column
# names its basis.
BASIS_NAME = {"discovery": "discovery draws", "pooled": "pooled rows", "all_draws": "48 draws", "all_rows": "all rows"}
POOLED_CG = "(1 − discovery w) × pooled CoT − forced"


def rel(p: Path, root: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def hf_config(hf_id: str, revision: str) -> dict | None:
    """config.json of a pinned checkpoint in the local Hugging Face cache; None when it is not there."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
               / "hub")
    p = hub / f"models--{hf_id.replace('/', '--')}" / "snapshots" / revision / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def inventory_row(root: Path, hf_id: str) -> dict | None:
    """A checkpoint's parameter cell in the inventory table (INVENTORY): total and active (B, as written) and the table it
    is in; None when the file or the row is absent."""
    p = root / INVENTORY
    if not p.exists():
        return None
    table = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Table"):
            table = line[3:].split(".")[0]
        m = re.match(r"^\|.*?\| `" + re.escape(hf_id) + r"` \| ([\d.]+)(?: (dense)| / ([\d.]+)) \|", line)
        if m:
            return {"total": m.group(1), "active": m.group(1) if m.group(2) else m.group(3),
                    "arch": "dense" if m.group(2) else "mixture of experts", "table": table}
    return None


def recorded_size(root: Path, key: str) -> dict:
    """Total and active parameters (B, as the source writes them) and the architecture of one model, each from its
    recorded source (SIZE_SPEC), with the source named; None for whatever is not recorded."""
    spec = SIZE_SPEC.get(key)
    out = {"total": None, "active": None, "arch": None, "source": None, "recorded": False}
    if spec is None:
        return out
    if spec["kind"] == "inventory":
        r = inventory_row(root, spec["hf_id"])
        if r:
            out.update(total=r["total"], active=r["active"], arch=r["arch"], recorded=True,
                       source=f"`{INVENTORY}`, {r['table']}")
        return out
    if spec["kind"] == "name":
        m = re.search(r"-(\d+(?:\.\d+)?)B-", spec["name"])
        out.update(total=m.group(1) if m else None, recorded=bool(m), source=spec["source"])
        cfg = hf_config(*spec["config"])
        if cfg is not None:
            tc = cfg.get("text_config", cfg)
            moe = any(tc.get(f) for f in ("num_experts", "num_local_experts", "n_routed_experts"))
            out["arch"] = "mixture of experts" if moe else "dense"
            out["arch_source"] = f"config.json of `{spec['config'][0]}` at its pinned revision"
            if not moe:
                out["active"] = out["total"]
        return out
    if spec["kind"] == "published":
        out.update(published_size(root, key))
        if spec.get("predecessor"):
            out["predecessor"] = predecessor_size(root, spec["predecessor"])
        return out
    raise ValueError(f"{key}: size kind {spec['kind']!r} unknown")


SIZE_FIELDS = ("total", "active", "active_input", "active_output")


def published_size(root: Path, key: str) -> dict:
    """A model's published size from PUBLISHED_SIZES: the numbers (B, as the page writes them), the architecture, the
    page (source, URL, read date) and its quoted sentences. Every number must be written in the quotes as <n>B, and a
    mixture of experts must be named there; otherwise ValueError. Nothing recorded: the fields stay None."""
    p = root / PUBLISHED_SIZES
    e = (json.loads(p.read_text(encoding="utf-8"))["models"].get(key) if p.exists() else None)
    if not e:
        return {}
    quotes = " ".join(e["quotes"])
    for f in SIZE_FIELDS:
        if e.get(f) is not None and not re.search(r"(?<![\d.])" + re.escape(e[f]) + r"B\b", quotes):
            raise ValueError(f"{key}: {f} {e[f]}B is not written in the quoted sentences of {PUBLISHED_SIZES}")
    if e["arch"] == "mixture of experts" and not re.search(r"\bMoE\b|Mixture-of-Experts", quotes, re.I):
        raise ValueError(f"{key}: the quoted sentences of {PUBLISHED_SIZES} do not name a mixture of experts")
    return {**{f: e.get(f) for f in SIZE_FIELDS}, "arch": e["arch"], "recorded": True, "published": True,
            "source": f"{e['source']}, {e['url']}, read {e['read']}", "url": e["url"], "read": e["read"],
            "published_for": e["published_for"], "caveat": e.get("caveat"), "quotes": e["quotes"],
            "manifest": PUBLISHED_SIZES}


def predecessor_size(root: Path, spec: dict) -> dict:
    """The size recorded for the model a release replaces (the file named in spec["request"] and the inventory), kept
    for the record and marked not used: DeepSeek V4.1-Flash-1017 is grouped by its own published size."""
    req = root / spec["request"]
    m = re.search(re.escape(spec["of"]) + r"\*\* \((\d+(?:\.\d+)?)B total, about (\d+(?:\.\d+)?)B active",
                  req.read_text(encoding="utf-8")) if req.exists() else None
    if m is None and spec.get("recorded"):
        m = re.match(r"(\S+) (\S+)", " ".join(spec["recorded"]))
    return {"name": spec["of"], "total": m.group(1) if m else None, "active": m.group(2) if m else None,
            "arch": "mixture of experts" if m and m.group(1) != m.group(2) else None,
            "source": f"`{spec['request']}`", "inventory": inventory_row(root, spec["hf_id"]), "used": False,
            "note": "the size recorded for the model this release replaces; not used, since its own published "
                    "size is recorded"}


def size_text(sz: dict, with_source: bool = True) -> str:
    """A model's recorded size in words: '8.92B, dense', '125.1B total, 10B active, mixture of experts', '552B total,
    8B active for input and 16B for output, mixture of experts', or '675B total by name; active parameters and
    architecture not recorded'; with the source, and for published figures of another checkpoint, the caveat."""
    if sz.get("total") is None:
        return "not recorded"
    if sz["arch"] == "dense":
        txt = f"{sz['total']}B" + (" by name" if sz.get("arch_source") else "") + ", dense"
    elif sz["arch"] == "mixture of experts" and sz.get("active_input") and sz.get("active_output"):
        txt = (f"{sz['total']}B total, {sz['active_input']}B active for input and {sz['active_output']}B for output, "
               "mixture of experts")
    elif sz["arch"] == "mixture of experts" and sz["active"]:
        txt = f"{sz['total']}B total, {sz['active']}B active, mixture of experts"
    else:
        txt = f"{sz['total']}B total by name; active parameters and architecture not recorded"
    if with_source:
        txt += (f" ({sz['source']}" + (f"; architecture: {sz['arch_source']}" if sz.get("arch_source") else "")
                + (f"; {sz['caveat']}" if sz.get("caveat") else "") + ")")
    elif sz.get("caveat"):
        txt += f" ({sz['caveat']})"
    return txt


def read_spend(root: Path) -> dict:
    """The API runs' spend as their reports give it (exact decimals); None for what a report does not say."""
    out = {"mistral": None, "deepseek": None}
    p = root / SPEND_MISTRAL
    if p.exists():
        m = re.search(r"Spend: run \$(\d+(?:\.\d+)?) .*?with the forced test \$(\d+(?:\.\d+)?)", p.read_text(encoding="utf-8"))
        if m:
            out["mistral"] = {"run": Fraction(m.group(1)), "with_test": Fraction(m.group(2)), "source": SPEND_MISTRAL}
    p = root / SPEND_NEBIUS
    if p.exists():
        # The report's total sums the cost of every reply in data/frontier_panel/nebius_panel_raw.jsonl (the pilot's,
        # the test calls' and the complete-design run's) plus the live test (scripts/nebius_panel_report.py, spend);
        # the "$3.20 spent before it" of its projection line is the live test, the pilot and the test calls, the
        # meter the real-time run started from: already inside the total, never added to it.
        t = p.read_text(encoding="utf-8")
        a = re.search(r"Spend: \$(\d+(?:\.\d+)?) in total today", t)
        b = re.search(r"with \$(\d+(?:\.\d+)?) spent before it", t)
        q = root / SPEND_NEBIUS_BEFORE
        c = re.search(r"The meter starts from [^(]*\(\$(\d+(?:\.\d+)?)\)", q.read_text(encoding="utf-8")) \
            if q.exists() else None
        if a and b:
            out["deepseek"] = {"total": Fraction(a.group(1)), "before": Fraction(b.group(1)),
                               "meter": Fraction(c.group(1)) if c else None, "source": SPEND_NEBIUS,
                               "before_source": SPEND_NEBIUS_BEFORE}
    return out


def panel_precision(root: Path, keys: list[str]) -> dict:
    """dtype of each panel model in the registries (config/models.yaml, dataset2.panel; config/models_extension.yaml,
    dataset2.models); None for a model in neither."""
    import yaml
    out = {}
    for f in PANEL_CONFIGS:
        p = root / f
        if not p.exists():
            continue
        d = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("dataset2", {})
        for sec in ("panel", "models"):
            for k, m in (d.get(sec) or {}).items():
                if isinstance(m, dict) and "dtype" in m and k in keys:
                    out[k] = {"dtype": m["dtype"], "source": f}
    return {k: out.get(k) for k in keys}


def precision_phrase(pp: dict) -> str:
    """The panel's precision in a phrase: the common dtype, then the exceptions by key."""
    known = {k: v["dtype"] for k, v in pp.items() if v}
    if not known:
        return "not recorded here"
    counts: dict = {}
    for v in known.values():
        counts[v] = counts.get(v, 0) + 1
    main = max(sorted(counts), key=lambda x: counts[x])
    other: dict = {}
    for k, v in sorted(known.items()):
        if v != main:
            other.setdefault(v, []).append(k)
    miss = sorted(k for k, v in pp.items() if not v)
    return (main + ("" if not other else ", except " + "; ".join(" and ".join(f"`{k}`" for k in ks) + f" in {v}"
                                                               for v, ks in sorted(other.items())))
            + ("" if not miss else "; not recorded for " + ", ".join(f"`{k}`" for k in miss)))


ROUNDING_EDGE: list[str] = []      # floats met on a rounding boundary while printing (section 12); reset per run


def half_even(q: Fraction, d: int) -> str:
    """q rounded half-even at d decimals, as text, in integer arithmetic. A negative value that rounds to zero keeps
    its sign ("-0.0"), as Python prints floats."""
    s = q * 10 ** d
    n, r = divmod(s.numerator, s.denominator)            # n = floor(s), 0 <= r < denominator
    if 2 * r > s.denominator or (2 * r == s.denominator and n % 2):
        n += 1
    whole, frac = divmod(abs(n), 10 ** d)
    return ("-" if q < 0 else "") + str(whole) + (f".{frac:0{d}d}" if d else "")


class Root(float):
    """The square root of an exact ratio (a template SD over configurations): a float wherever it is used, with its
    exact square kept (sq) so that fnum rounds the root itself."""

    def __new__(cls, sq: Fraction):
        o = super().__new__(cls, math.sqrt(sq))
        o.sq = Fraction(sq)
        return o


def half_even_root(sq: Fraction, d: int) -> str:
    """sqrt(sq) rounded half-even at d decimals, exactly: n = floor(sqrt(W)), W = sq x 10^(2d), against n + 1/2."""
    w = sq * 10 ** (2 * d)
    n = math.isqrt(w.numerator // w.denominator)
    half = Fraction(2 * n + 1, 2) ** 2
    if w > half or (w == half and n % 2):
        n += 1
    whole, frac = divmod(n, 10 ** d)
    return str(whole) + (f".{frac:0{d}d}" if d else "")


def fnum(x, d=1) -> str:
    """The one place a number becomes text: x rounded half-even at d decimals on its exact value. Exact values
    (Fraction, int) are rounded as they are; a template SD (Root) is rounded on the exact root of its exact square. Any
    other float (a file value off its grid) is rounded on its binary value, which is what it is; one within 1e-9 of a
    halfway point is listed in ROUNDING_EDGE, since the digit it prints there is not settled by the computation."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, Root):
        return half_even_root(x.sq, d)
    if isinstance(x, float):
        s = abs(x) * 10 ** d
        if abs(s - math.floor(s) - 0.5) <= 1e-9 * 10 ** d:
            ROUNDING_EDGE.append(f"{x!r} printed at {d} decimals is a float within 1e-9 of a halfway point; its digit "
                                 "follows the float")
    return half_even(Fraction(x), d)


def nz(x):
    """A value as it is (exact ratios stay exact); None for a missing or non-finite one."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return x if isinstance(x, Fraction) else float(x)


def same(f, q, what: str):
    """The exact value q in place of the float f that scripts/nebius_panel_report.py computes for it, after checking
    that they agree (1e-9); a disagreement is an error, never smoothed over."""
    f = None if f is None or (isinstance(f, float) and not math.isfinite(f)) else float(f)
    if (f is None) != (q is None):
        raise ValueError(f"{what}: exact value {q} against {f}")
    if q is not None and abs(float(q) - f) > TOL:
        raise ValueError(f"{what}: exact value {q} ({float(q)!r}) against {f!r}")
    return q


def config_counts(g: pd.DataFrame, col: str) -> dict:
    """Per configuration: (rows with 1, rows with a value) of a 0/1 column; rows without a value are not counted, as
    pandas' mean skips them."""
    v = g[col].astype("float64")                         # bool, int or float; a missing value stays NaN
    bad = v[v.notna() & ~v.isin([0.0, 1.0])]
    if len(bad):
        raise ValueError(f"{col}: values other than 0 and 1: {sorted(set(bad))[:5]}")
    ok = v.notna()
    k = v[ok].groupby(g.loc[ok, "config_id"]).sum()
    n = ok.groupby(g["config_id"]).sum()
    return {str(c): (int(k.get(c, 0)), int(n[c])) for c in n.index if int(n[c])}


def exact_mean(cnt: dict, draws: list, scale=100):
    """Mean over a draw list of the per-configuration share k/n (a configuration drawn twice counts twice), times
    scale; None when a drawn configuration has no rows (as npr.draw_mean)."""
    if not draws or any(c not in cnt for c in draws):
        return None
    return Fraction(scale) * sum((Fraction(*cnt[c]) for c in draws), Fraction(0)) / len(draws)


def exact_share(g: pd.DataFrame, col: str, scale=100):
    """Share of 1s over every row with a value, times scale; None with no row."""
    v = g[col].astype("float64")
    v = v[v.notna()]
    return Fraction(scale * int(v.sum()), len(v)) if len(v) else None


def sub_(a, b):
    return None if a is None or b is None else a - b


def as_json(o):
    """json.dumps default: an exact ratio is written as its float; anything else as text."""
    return float(o) if isinstance(o, Fraction) else str(o)


def basis_of(t: str, score: str) -> str:
    """The panel file's basis for one benchmark and score."""
    return "pooled" if (t == "hotpot" and score == "content") else "discovery"


def step_word(d, q: str) -> str:
    """One step of a family ladder, judged on the exact step (a float step is compared as the value it holds)."""
    th = FLAT["w"] if q == "w" else FLAT["pp"]
    return "up" if d >= th else ("down" if d <= -th else "flat")


def step_words(vals: list, q: str = "pp", ranges: list | None = None) -> list[str]:
    """The word of each step (DIRECTION_RULE); with `ranges` (per value, the (low, high) the stop's unread replies
    allow), a step whose word differs at the two ends of its range reads '<word> within the stop's loss'."""
    out = []
    for j, (a, b) in enumerate(zip(vals, vals[1:])):
        w = step_word(b - a, q)
        if ranges is not None:
            (alo, ahi), (blo, bhi) = ranges[j], ranges[j + 1]
            if step_word(blo - ahi, q) != step_word(bhi - alo, q):
                w += WITHIN_LOSS
        out.append(w)
    return out


def value_range(v, unread, q: str, w=None):
    """(low, high) of a content value when the rung's unread forced replies (pp of its zero-shot forced arm,
    discovery draws) are counted none or all correct (STOP_Q); (v, v) for other quantities or with nothing unread."""
    if v is None or q not in STOP_Q or not unread:
        return (v, v)
    lo, hi = STOP_Q[q]
    f = (1 - w) if q in ("corrected_gain", "residual") and w is not None else 1
    return (v + lo * f * unread, v + hi * f * unread)


def stop_ranges(pairs: list, t: str, sc_: str, q: str, stop: dict, key_of=lambda r: r["key"], w_of=None):
    """Per rung (pairs of (rung, value)), the (low, high) of a content value of a forced-arm quantity on GSM8K or
    BIG-Bench Hard when the rung's unread forced replies are counted none or all correct; None where the bound does
    not apply (strict scores, MATH-500, HotpotQA, the other quantities)."""
    if sc_ != "content" or t not in STOP_TASKS or q not in STOP_Q or not stop:
        return None
    return [value_range(v, (stop.get(key_of(r)) or {}).get(t, {}).get("discovery_pp"), q,
                        w_of(r) if w_of else None) for r, v in pairs]


def stop_unread(root: Path) -> dict:
    """Per model and benchmark (GSM8K, BIG-Bench Hard): the zero-shot forced replies the stop at the first "}" left
    unread (forced-stop/1.1.0: box open and not completed; scripts/forced_stop_audit.py), over all rows and as an exact
    share of the arm on the discovery draws (per configuration, averaged over the draws, in pp): the most the forced
    content accuracy can fall short of the answers the replies would have held. Ladder models from the audit's
    collection "ladder", panel models from the forced-direct collection ("forced"). Empty when the audit's table is absent."""
    p = root / STOP_UNREAD
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype={"config_id": str, "model": str})
    out = {}
    for t in STOP_TASKS:
        disc, _ = npr.sampler(t)
        sub = df[(df["task"] == t) & (df["arm"] == "forced_direct") & df["collection"].isin(["ladder", "forced"])]
        for (coll, model), g in sub.groupby(["collection", "model"]):
            per = {str(c): Fraction(int(u), int(n)) for c, u, n in zip(g["config_id"], g["unread"], g["rows"]) if n}
            miss = [c for c in disc if str(c) not in per]
            out.setdefault(model, {})[t] = {
                "collection": coll, "rows": int(g["rows"].sum()), "unread": int(g["unread"].sum()),
                "unfinished": int(g["unfinished"].sum()),
                "discovery_pp": None if miss else 100 * sum(per[str(c)] for c in disc) / len(disc),
                "configs_missing": miss}
    return out


def stop_counts(root: Path) -> dict:
    """Per ladder model, benchmark (GSM8K, BIG-Bench Hard) and forced arm: rows, replies the stop left open, completed,
    inner box, range (a time range cut at "to", forced-stop/1.2.0; read by the content reader on temporal_sequences, or
    not), unfinished (text asks for more or holds two options), other, unread (unfinished + other + ranges not read), from the
    audit (STOP_AUDIT); and the API models' check against their full replies (FULL_REPLIES, the current rule): false
    gains and false losses, all replies the cut left open and the unread ones. Empty parts when the files are absent."""
    out = {"audit": {}, "full_replies": {}}
    pa, pf = root / STOP_AUDIT, root / FULL_REPLIES
    if pa.exists():
        a = json.loads(pa.read_text(encoding="utf-8"))
        out["version"] = a.get("forced_stop_version")
        for c in a["cells"]:
            if c["collection"] != "ladder" or c["task"] not in STOP_TASKS:
                continue
            out["audit"].setdefault(c["model"], {}).setdefault(c["task"], {})[c["arm"]] = {
                k: int(c.get(k, 0)) for k in ("rows", "open_box", "completed", "open_inner_box", "open_range",
                                              "open_range_read", "open_range_unread", "open_unfinished",
                                              "open_other", "unread", "completed_v100", "open_inner_box_restart")}
    if pf.exists():
        f = json.loads(pf.read_text(encoding="utf-8"))
        v = out.get("version") or f["versions"][-1]
        for c in f["cells"]:
            if c["version"] != v:
                continue
            d = out["full_replies"].setdefault(c["model"], {"replies": 0, "full_not_closed": 0, "compared": 0,
                                                            "false_gain": 0, "false_loss": 0, "unread": 0,
                                                            "unread_compared": 0, "unread_false_loss": 0})
            for k in ("replies", "full_not_closed", "compared", "false_gain", "false_loss"):
                d[k] += int(c.get(k, 0))
            # forced-stop/1.2.0: unread is what the pipeline reads as a malformed box (the script's
            # per-cell counts): open_unfinished and open_other, and open_range replies the content reader does not read
            d["unread"] += int(c.get("unread", 0))
            d["unread_compared"] += int(c.get("unread_compared", 0))
            d["unread_false_loss"] += int(c.get("unread_false_loss", 0))
    return out


def direction_of(vals: list, q: str = "pp", ranges: list | None = None) -> str | None:
    """Direction over size (DIRECTION_RULE); None with fewer than two values."""
    if len(vals) < 2:
        return None
    steps = step_words(vals, q, ranges)
    for word, name in (("up", "rises"), ("down", "falls"), ("flat", "flat")):
        if all(s == word for s in steps):
            return name
    return ", ".join(steps)


def dword(d: str | None) -> str:
    return "–" if d is None else d


def tied(p: dict) -> str:
    """', k tied' when the value equals k panel values (within 1e-9); empty otherwise. Ties are printed, never hidden
    """
    return f", {p['equal']} tied" if p.get("equal") else ""


def cell(p: dict | None, d: int = 1) -> str:
    """A placement cell: value (b/M[, k tied]) and, outside the panel's range, ↑ or ↓; the same everywhere it is
    printed."""
    if p is None or p.get("value") is None:
        return "–"
    mark = {"above": " ↑", "below": " ↓"}.get(p["outside_panel_range"], "")
    return f"{fnum(p['value'], d)} ({p['below']}/{p['panel_n']}{tied(p)}){mark}"


def above_txt(p: dict) -> str:
    """'above b of M[, k tied]': the placement count in words."""
    return f"above {p['below']} of {p['panel_n']}{tied(p)}"


# --------------------------------------------------------------------------------------------------------------------
# Ladder rows
# --------------------------------------------------------------------------------------------------------------------

def model_paths(key: str, root: Path) -> dict:
    spec = LADDER[key]
    pats = spec.get("paths", CLUSTER_PATHS)
    return {r: {t: root / pats[r].format(task=t, key=key) for t in TASKS} for r in READINGS}


def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["config_id"] = df["config_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    return df


def zero_shot_summary(task: str, sc: pd.DataFrame) -> dict:
    """The numbers nebius_panel_report.task_summary computes from scored rows (accuracy, gains, w, corrected gain,
    template SD), on the units with all three zero-shot arms; plus the pooled per-arm means (the basis of the panel's
    HotpotQA content table) and the forced arms' readability."""
    cu = npr.complete_units(sc)
    disc, val = npr.sampler(task)
    draws = disc + val
    configs = sorted(set(draws))
    have = sorted(cu["config_id"].unique())
    n_zero = int(sc["arm"].isin(ARMS3).sum())
    res = {"configs_design": len(configs), "configs_answered": len(have),
           "configs_not_in_sampler": sorted(set(have) - set(configs)),
           "units_answered": int(cu[["config_id", "item_id"]].drop_duplicates().shape[0]),
           "zero_shot_rows_left_out": n_zero - int(len(cu)), "forced_cap": npr.FORCED_CAP[task]}
    # Each ratio is computed exactly from the integer counts and checked against npr's float (same()).
    for name, col in SCORES.items():
        pc = npr.per_config(cu, col) * 100
        cnt = {a: config_counts(cu[cu["arm"] == a], col) for a in ARMS3}
        blk = {}
        for lab, dr in (("all_draws", draws), ("discovery", disc)):
            acc = {npl.ARM_LABEL[a]: same(npr.draw_mean(pc[a], dr), exact_mean(cnt[a], dr),
                                          f"{task}/{name}/{lab}/{a}") for a in ARMS3}
            blk[lab] = {"accuracy_pp": acc, "gain_over_requested_pp": sub_(acc["cot"], acc["requested"]),
                        "gain_over_forced_pp": sub_(acc["cot"], acc["forced"])}
        pooled = {npl.ARM_LABEL[a]: same(float(100 * cu.loc[cu["arm"] == a, col].mean()),
                                         exact_share(cu[cu["arm"] == a], col), f"{task}/{name}/pooled/{a}")
                  for a in ARMS3}
        blk["pooled"] = {"accuracy_pp": pooled, "gain_over_requested_pp": sub_(pooled["cot"], pooled["requested"]),
                         "gain_over_forced_pp": sub_(pooled["cot"], pooled["forced"])}
        blk["template"] = template_exact(cu, col, f"{task}/{name}/template")
        res[name] = blk
    req = cu[cu["arm"] == "direct_answer"]
    long_ = req["n_output_tokens"] > npr.FORCED_CAP[task]
    long_share = long_.groupby(req["config_id"]).mean()
    lcnt = config_counts(req.assign(long_=long_), "long_")
    res["w"] = {"discovery": same(npr.draw_mean(long_share, disc), exact_mean(lcnt, disc, 1), f"{task}/w/discovery"),
                "all_draws": same(npr.draw_mean(long_share, draws), exact_mean(lcnt, draws, 1), f"{task}/w/all_draws")}
    for name in SCORES:
        for lab in ("discovery", "all_draws", "pooled"):
            w = res["w"]["discovery" if lab == "pooled" else lab]
            g = res[name][lab]["gain_over_forced_pp"]
            res[name][lab]["corrected_gain_pp"] = None if None in (w, g) else (1 - w) * g
    # readable answers: the strict scorer's compliance flag (a recognised answer form was found), per arm
    rd = {}
    for a in ARMS3 + ("fewshot_forced", "fewshot_requested"):
        g = (cu if a in ARMS3 else sc)
        g = g[g["arm"] == a]
        if g.empty or "compliant" not in g:
            continue
        g = g.assign(flag_=g["compliant"].astype(bool))
        flag = g["flag_"].astype(float) * 100
        d = {"all_rows": same(float(flag.mean()), exact_share(g, "flag_"), f"{task}/readable/{a}/all_rows")}
        if a in ARMS3:
            pcr = flag.groupby(g["config_id"]).mean()
            fc = config_counts(g, "flag_")
            d["discovery"] = same(npr.draw_mean(pcr, disc), exact_mean(fc, disc), f"{task}/readable/{a}/discovery")
            d["all_draws"] = same(npr.draw_mean(pcr, draws), exact_mean(fc, draws), f"{task}/readable/{a}/all_draws")
        rd[npl.ARM_LABEL[a]] = d
    res["readable_pct"] = rd
    # content - strict per zero-shot arm on the discovery draws (the same units and draws as the accuracies)
    res["strict_content_gap_discovery_pp"] = {
        lab: sub_(res["content"]["discovery"]["accuracy_pp"][lab], res["strict"]["discovery"]["accuracy_pp"][lab])
        for lab in ("requested", "forced", "cot")}
    # content - strict per arm over all rows of the arm
    gap = {}
    for a in ("direct_answer", "forced_direct", "zero_shot_cot", "fewshot_forced", "fewshot_requested"):
        g = sc[sc["arm"] == a]
        if g.empty:
            gap[npl.ARM_LABEL[a]] = None
            continue
        gap[npl.ARM_LABEL[a]] = same(float(100 * (g["content_correct"].mean() - g["strict_correct"].mean())),
                                     sub_(exact_share(g, "content_correct"), exact_share(g, "strict_correct")),
                                     f"{task}/strict_content_gap/{a}")
    res["strict_content_gap_pp"] = gap
    return res


def exact_var(xs: list):
    """Sample variance (denominator m - 1) of exact values; None with fewer than two."""
    m = len(xs)
    if m < 2:
        return None
    mean = sum(xs, Fraction(0)) / m
    return sum(((x - mean) ** 2 for x in xs), Fraction(0)) / (m - 1)


def interaction_exact(table: np.ndarray) -> Fraction:
    """The configuration x item interaction mean square of a complete (C, N) table of whole-pp values, exactly, by
    src/noise_anatomy.config_variance's route: W_item = [sum_i Var_c(y_ci) - N Var_c(ybar_c)] / (N - 1), with
    sum_i Var_c(y_ci) = (C sum y^2 - sum_i (sum_c y_ci)^2) / (C (C - 1)) in integers."""
    y = table.astype(object)
    C, N = table.shape
    col = [sum(int(v) for v in y[:, i]) for i in range(N)]
    ss = sum(int(v) * int(v) for v in y.ravel())
    sum_var_c = Fraction(C * ss - sum(s * s for s in col), C * (C - 1))
    rows = [Fraction(sum(int(v) for v in y[c, :]), N) for c in range(C)]
    return (sum_var_c - N * exact_var(rows)) / (N - 1)


def template_exact(cu: pd.DataFrame, col: str, what: str) -> dict:
    """npr.template_block with its values exact: the variance over the distinct configurations of the per-configuration
    values (denominator C - 1); the item sampling removed by the panel's estimator for one model
    (src/noise_anatomy.config_variance): every configuration has the same N items, so the removed term is the
    configuration x item interaction mean square over N, not the item variance within a configuration (which holds
    item difficulty, common to all configurations); the difference; the SD as the root of the exact variance (Root).
    Each checked against npr's float (1e-9)."""
    fl = npr.template_block(cu, col)
    arms = {"requested": "direct_answer", "forced": "forced_direct", "cot": "zero_shot_cot"}
    mean = {a: {c: Fraction(100 * k, m) for c, (k, m) in config_counts(cu[cu["arm"] == arm], col).items()}
            for a, arm in arms.items()}
    for g, base in (("gain_over_requested", "requested"), ("gain_over_forced", "forced")):
        mean[g] = {c: mean["cot"][c] - mean[base][c] for c in mean["cot"] if c in mean[base]}
    wide = cu.pivot_table(index=["config_id", "item_id"], columns="arm", values=col, aggfunc="first") * 100
    items = {a: wide[arm] for a, arm in arms.items()}
    items["gain_over_requested"] = wide["zero_shot_cot"] - wide["direct_answer"]
    items["gain_over_forced"] = wide["zero_shot_cot"] - wide["forced_direct"]
    out = {}
    for k in ("requested", "forced", "cot", "gain_over_requested", "gain_over_forced"):
        V = exact_var(list(mean[k].values()))
        table = items[k].unstack("item_id")
        q = n_items = None
        if table.shape[0] >= 2 and table.shape[1] >= 2 and not table.isna().to_numpy().any():
            a = table.to_numpy(dtype=float)
            if np.any(a != np.round(a)):
                raise ValueError(f"{what}/{k}: item values not whole pp")
            n_items = int(a.shape[1])
            q = interaction_exact(np.round(a).astype(np.int64))
            if V is not None and exact_var([Fraction(int(x)) for x in np.round(a).astype(np.int64).sum(axis=1)]) \
                    / (n_items * n_items) != V:
                raise ValueError(f"{what}/{k}: configuration means differ from the table's row means")
        noise = None if q is None else q / n_items
        f = fl[k]
        out[k] = {"sd_pp": None if V is None else Root(V),
                  "variance_pp2": same(f["variance_pp2"], V, f"{what}/{k}/variance"),
                  "items": same(f["items"], n_items, f"{what}/{k}/items"),
                  "interaction_mean_square_pp2": same(f["interaction_mean_square_pp2"], q, f"{what}/{k}/interaction"),
                  "item_sampling_variance_pp2": same(f["item_sampling_variance_pp2"], noise, f"{what}/{k}/noise"),
                  "variance_minus_item_sampling_pp2": same(f["variance_minus_item_sampling_pp2"], sub_(V, noise),
                                                           f"{what}/{k}/variance minus noise")}
        same(f["sd_pp"], None if V is None else Fraction(float(out[k]["sd_pp"])), f"{what}/{k}/sd")
        if set(out[k]) != set(f):
            raise ValueError(f"{what}/{k}: fields differ from npr.template_block")
    return out


def fewshot_summary(task: str, sc: pd.DataFrame) -> dict:
    """npr.fewshot_block (the five-shot arms), its ratios held exactly and checked against its floats."""
    fl = npr.fewshot_block(task, sc)
    disc, val = npr.sampler(task)
    draws = disc + val
    out = {}
    for arm in ("fewshot_forced", "fewshot_requested"):
        lab = npl.ARM_LABEL[arm]
        if lab not in fl:
            continue
        g = sc[sc["arm"] == arm]
        f, d = fl[lab], {"rows": fl[lab]["rows"], "configs": fl[lab]["configs"]}
        for name, col in SCORES.items():
            cnt = config_counts(g, col)
            d[f"{name}_pp_config_mean"] = same(f[f"{name}_pp_config_mean"],
                                               Fraction(100) * sum((Fraction(*v) for v in cnt.values()), Fraction(0))
                                               / len(cnt), f"{task}/{arm}/{name}/config_mean")
            if f"{name}_pp_all_draws" in f:
                d[f"{name}_pp_all_draws"] = same(f[f"{name}_pp_all_draws"], exact_mean(cnt, draws),
                                                 f"{task}/{arm}/{name}/all_draws")
        if "share_longer_than_forced_cap" in f:
            d["share_longer_than_forced_cap"] = same(
                f["share_longer_than_forced_cap"],
                Fraction(int((g["n_output_tokens"] > npr.FORCED_CAP[task]).sum()), len(g)), f"{task}/{arm}/longer")
        if set(d) != set(f):
            raise ValueError(f"{task}/{arm}: fields {sorted(set(f) ^ set(d))} differ from npr.fewshot_block")
        out[lab] = d
    return out


def cut_counts(sc: pd.DataFrame) -> dict:
    """Per arm: rows, outputs cut at the panel's cap (share exact), and (uncut reading) outputs that ended at the
    raised cap."""
    out = {}
    for arm, g in sc.groupby("arm"):
        cut = g["cut_at_panel_cap"].astype(bool) if "cut_at_panel_cap" in g else pd.Series(False, index=g.index)
        out[npl.ARM_LABEL.get(arm, arm)] = {
            "rows": int(len(g)), "cut_at_panel_cap": int(cut.sum()), "share_cut": Fraction(int(cut.sum()), len(g)),
            "finish_length": int((g["finish_reason"] == "length").sum()) if "finish_reason" in g else None}
    return out


def model_block(key: str, root: Path) -> dict:
    """Every number of one ladder model, per benchmark and reading; status complete, partial or missing."""
    paths = model_paths(key, root)
    blk = {"key": key, **{k: v for k, v in LADDER[key].items() if k != "paths"},
           "files": {r: {t: rel(p, root) for t, p in d.items()} for r, d in paths.items()},
           "tasks": {}, "tasks_missing": [], "edge_cases": []}
    for t in TASKS:
        have = {r: paths[r][t].exists() for r in READINGS}
        if not any(have.values()):
            blk["tasks_missing"].append(t)
            continue
        tb = {"readings_missing": [r for r in READINGS if not have[r]]}
        for r in READINGS:
            if not have[r]:
                blk["edge_cases"].append(f"{TASK_NAME[t]}: the {r} reading is absent ({rel(paths[r][t], root)})")
                continue
            sc = load_scored(paths[r][t])
            rows = {a: int(n) for a, n in sc["arm"].value_counts().sort_index().items()}
            short = {a: (rows.get(a, 0), n) for a, n in DESIGN_ROWS[t].items() if rows.get(a, 0) != n}
            if short:
                blk["edge_cases"].append(f"{TASK_NAME[t]} ({r}): rows per arm differ from the design "
                                         + "; ".join(f"{a} {h:,} of {n:,}" for a, (h, n) in short.items()))
            odd = sc.loc[~sc["finish_reason"].isin(["stop", "length"]), ["arm", "finish_reason"]] \
                if "finish_reason" in sc else sc.iloc[0:0]
            if len(odd):
                blk["edge_cases"].append(
                    f"{TASK_NAME[t]} ({r}): {len(odd)} rows end with a finish reason other than stop or length ("
                    + ", ".join(f"{fr} in {n} {npl.ARM_LABEL.get(a, a)}" for (a, fr), n in
                                odd.value_counts(["arm", "finish_reason"]).sort_index().items())
                    + "); scored as they came")
            s = zero_shot_summary(t, sc)
            s["rows_by_arm"] = rows
            s["fewshot"] = fewshot_summary(t, sc)
            s["cuts"] = cut_counts(sc)
            if s["zero_shot_rows_left_out"]:
                blk["edge_cases"].append(f"{TASK_NAME[t]} ({r}): {s['zero_shot_rows_left_out']} zero-shot rows left "
                                         "out because their unit lacks an arm")
            if s["configs_answered"] < s["configs_design"]:
                blk["edge_cases"].append(f"{TASK_NAME[t]} ({r}): {s['configs_answered']} of {s['configs_design']} "
                                         "configurations; draw means over a missing configuration are absent")
            tb[r] = s
        blk["tasks"][t] = tb
    cells = [(t, n, k) for t, tb in blk["tasks"].items() if "cut" in tb for n in SCORES
             for k, v in tb["cut"][n]["template"].items()]
    neg = [c for c in cells if (blk["tasks"][c[0]]["cut"][c[1]]["template"][c[2]]["variance_minus_item_sampling_pp2"]
                                or 0) < 0]
    if neg:
        blk["edge_cases"].append(f"template variance minus the item sampling (interaction mean square over the items) "
                                 f"is negative in {len(neg)} of {len(cells)} cells (benchmark × score × quantity, cut), "
                                 "reported as they come in section 7: there the spread of the configuration means is "
                                 "no larger than their item sampling")
    n_have = len(blk["tasks"])
    blk["status"] = "complete" if n_have == len(TASKS) and not any(
        v["readings_missing"] for v in blk["tasks"].values()) else ("missing" if n_have == 0 else "partial")
    if LADDER[key]["where"] == "cluster":
        sj = root / SCORED_SUMMARY.format(key=key)
        if sj.exists():
            d = json.loads(sj.read_text(encoding="utf-8"))
            blk["run_key"] = d.get("run_key")
            blk["kernels"] = "default (-dk fallback)" if str(d.get("run_key", "")).endswith("-dk") else "batch-invariant"
            blk["scored"] = d.get("created")
        pj = root / PILOT_CHECK.format(key=key)
        if pj.exists():
            p = json.loads(pj.read_text(encoding="utf-8"))
            verdict = str(p.get("verdict", ""))
            req = {}          # the requested arms the pilot rule pools (zero- and five-shot), per task: (boxed, rows)
            for name, x in (p.get("summary") or {}).items():
                if x.get("arm") in ("direct_answer", "fewshot_requested"):
                    req[name] = (int(x["pass"]), int(x["rows"]))
            blk["pilot"] = {"pass": verdict.startswith("PILOT PASS") if verdict else p.get("pass", p.get("ok")),
                            "verdict": verdict or None, "reasons": p.get("reasons") or p.get("failures"),
                            "requested_arms": req, "source": rel(pj, root)}
    else:
        blk["kernels"] = "the provider's"
    return blk


# --------------------------------------------------------------------------------------------------------------------
# Completeness: scored rows against the design, per model, benchmark, arm and configuration
# --------------------------------------------------------------------------------------------------------------------

def read_ids(path: Path, extra: tuple = ()) -> pd.DataFrame:
    cols = ID_COLS + [c for c in extra]
    have = set(pq.ParquetFile(path).schema_arrow.names)
    df = pd.read_parquet(path, columns=[c for c in cols if c in have])
    for c in ID_COLS:
        df[c] = df[c].astype(str)
    return df


def loop_state(key: str, root: Path) -> dict | None:
    """The loop's last word on a model (jobs done), read from its log; None when the log does not name it."""
    p = root / LOOP_LOG
    if not p.exists():
        return None
    stamp, state = None, None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("== pass "):
            stamp = line[len("== pass "):].strip()
        for part in line.split(";"):
            part = part.strip()
            if part.startswith(f"M {key} "):
                bits = dict(b.split("=", 1) for b in part.split()[2:] if "=" in b)
                state = {"pass": stamp, "jobs_done": bits.get("jobs_done"), "run_key": bits.get("chosen"),
                         "source": LOOP_LOG}
    return state


def completeness(keys: list[str], root: Path) -> dict:
    """Per model and benchmark: the design's rows (request files) against the scored rows of each reading, per arm and
    configuration; missing, duplicate and extra request ids, ids whose arm, configuration or item differ, unscored rows
    (no strict or content score), missing arms. Nothing is dropped or filled in; a shortfall is listed as it is."""
    cache: dict = {}

    def design(src: str, t: str):
        if (src, t) not in cache:
            p = root / REQ_PATH.format(task=t, key=src)
            cache[(src, t)] = (read_ids(p) if p.exists() else None, rel(p, root))
        return cache[(src, t)]

    out = {}
    for key in keys:
        src = key if LADDER[key]["where"] == "cluster" else API_DESIGN_KEY
        paths = model_paths(key, root)
        m = {"design_source_key": src, "tasks": {}}
        for t in TASKS:
            d, dpath = design(src, t)
            rec = {"design_file": dpath}
            if d is None:
                rec["design_absent"] = True
                m["tasks"][t] = rec
                continue
            rec["design_rows"] = int(len(d))
            rec["design_duplicate_ids"] = int(d["request_id"].duplicated().sum())
            rec["design_by_arm"] = {a: int(n) for a, n in d["arm"].value_counts().sort_index().items()}
            rec["design_configs_by_arm"] = {a: int(g["config_id"].nunique()) for a, g in d.groupby("arm")}
            if src != API_DESIGN_KEY:
                r_, _ = design(API_DESIGN_KEY, t)
                if r_ is not None:
                    a = d.set_index("request_id")[["arm", "config_id", "item_id"]].sort_index()
                    b = r_.set_index("request_id")[["arm", "config_id", "item_id"]].sort_index()
                    rec["design_equals_api_design"] = bool(a.index.equals(b.index) and a.equals(b))
            dcell = d.groupby(["arm", "config_id"]).size()
            for r in READINGS:
                p = paths[r][t]
                if not p.exists():
                    rec[r] = {"present": False, "scored_rows": 0, "missing_ids": int(d["request_id"].nunique()),
                              "missing_arms": sorted(d["arm"].unique()), "file": rel(p, root)}
                    continue
                s = read_ids(p, extra=("strict_correct", "content_correct"))
                dup = s["request_id"].duplicated()
                ds, ss = set(d["request_id"]), set(s["request_id"])
                miss, extra = ds - ss, ss - ds
                j = s.drop_duplicates("request_id").merge(d.drop_duplicates("request_id"), on="request_id",
                                                          suffixes=("", "_design"))
                bad = j[(j["arm"] != j["arm_design"]) | (j["config_id"] != j["config_id_design"])
                        | (j["item_id"] != j["item_id_design"])]
                unscored = int(sum(s[c].isna().sum() for c in ("strict_correct", "content_correct") if c in s)) \
                    + sum(len(s) for c in ("strict_correct", "content_correct") if c not in s)
                scell = s.groupby(["arm", "config_id"]).size()
                cells = dcell.to_frame("design").join(scell.to_frame("scored"), how="outer").fillna(0).astype(int)
                short = cells[cells["design"] != cells["scored"]]
                by_arm = {}
                for a in sorted(set(d["arm"]) | set(s["arm"])):
                    by_arm[a] = {"design": rec["design_by_arm"].get(a, 0), "scored": int((s["arm"] == a).sum()),
                                 "missing_ids": int(d.loc[d["request_id"].isin(miss), "arm"].eq(a).sum()),
                                 "extra_ids": int(s.loc[s["request_id"].isin(extra), "arm"].eq(a).sum()),
                                 "duplicate_ids": int(s.loc[dup, "arm"].eq(a).sum()),
                                 "configs_design": rec["design_configs_by_arm"].get(a, 0),
                                 "configs_matching": int(((cells.loc[a]["design"] == cells.loc[a]["scored"])
                                                          & (cells.loc[a]["design"] > 0)).sum())
                                 if a in cells.index.get_level_values(0) else 0}
                rec[r] = {"present": True, "file": rel(p, root), "scored_rows": int(len(s)),
                          "duplicate_ids": int(dup.sum()), "missing_ids": len(miss), "extra_ids": len(extra),
                          "id_fields_differ": int(len(bad)), "unscored": unscored,
                          "missing_arms": sorted(a for a, v in by_arm.items() if v["design"] and not v["scored"]),
                          "by_arm": by_arm,
                          "cells_short": [{"arm": a, "config_id": c, "design": int(v["design"]),
                                           "scored": int(v["scored"])} for (a, c), v in short.iterrows()],
                          "missing_id_list": sorted(miss)[:200], "extra_id_list": sorted(extra)[:200]}
            m["tasks"][t] = rec
        tot = {r: sum(v.get(r, {}).get("scored_rows", 0) for v in m["tasks"].values()) for r in READINGS}
        m["design_rows"] = sum(v.get("design_rows", 0) for v in m["tasks"].values())
        m["scored_rows"] = tot
        probs = sum(v.get(r, {}).get(k, 0) for v in m["tasks"].values() for r in READINGS
                    for k in ("duplicate_ids", "missing_ids", "extra_ids", "id_fields_differ", "unscored"))
        m["exact"] = bool(not probs and all(not v.get("design_absent") for v in m["tasks"].values()))
        m["loop"] = loop_state(key, root) if LADDER[key]["where"] == "cluster" else None
        absent = [t for t, v in m["tasks"].items() if v.get("design_absent")]
        lp = m["loop"]
        loop_txt = (f"; cluster jobs done {lp['jobs_done']} at the loop's pass of {lp['pass']} (`{LOOP_LOG}`); not "
                    "yet copied back and scored" if lp and lp.get("jobs_done") else "")
        if absent:
            m["state"] = ("design file absent for " + ", ".join(TASK_NAME[t] for t in absent)
                          + f" (`{m['tasks'][absent[0]]['design_file']}`); scored rows here: cut {tot['cut']:,}, "
                            f"uncut {tot['uncut']:,}" + (loop_txt if not any(tot.values()) else ""))
        elif m["exact"]:
            m["state"] = f"every design row scored once in both readings ({m['design_rows']:,})"
        elif tot["cut"] == 0 and tot["uncut"] == 0:
            m["state"] = f"no scored rows here (0 of {m['design_rows']:,})" + loop_txt
        else:
            m["state"] = (f"short: cut {tot['cut']:,} and uncut {tot['uncut']:,} of {m['design_rows']:,} rows; "
                          "see the shortfall list")
        out[key] = m
    return out


# --------------------------------------------------------------------------------------------------------------------
# The main panel, read from its results
# --------------------------------------------------------------------------------------------------------------------

def load_panel(root: Path) -> dict:
    fam_src = json.loads((root / PANEL_FILES["families"]).read_text(encoding="utf-8"))
    families = dict(zip(fam_src["models"], fam_src["families"]))
    mix = pd.read_csv(root / PANEL_FILES["mixture"])
    out = {"families": families, "tasks": {}, "checks": {}, "basis": {}, "sources": {}}
    for t in ("gsm8k", "math", "bbh"):
        c = pd.read_csv(root / PANEL_FILES["compliance"].format(task=t)).set_index("model_key")
        m = mix[mix["task"] == t].set_index("model_key")
        df = pd.DataFrame(index=c.index)
        for sc_ in SCORES:
            df[f"requested_{sc_}"] = c[f"requested_{sc_}_correct"]
            df[f"forced_{sc_}"] = c[f"forced_{sc_}_correct"]
            df[f"cot_{sc_}"] = c[f"cot_{sc_}_correct"]
            df[f"cot_minus_requested_{sc_}"] = df[f"cot_{sc_}"] - df[f"requested_{sc_}"]
            df[f"cot_minus_forced_{sc_}"] = df[f"cot_{sc_}"] - df[f"forced_{sc_}"]
            df[f"corrected_gain_{sc_}"] = m[f"{sc_}_predicted_gain_vs_requested"].reindex(df.index)
            # the two files must agree on the gains (both are discovery-draw means)
            d1 = (df[f"cot_minus_forced_{sc_}"] - m[f"{sc_}_gain_vs_forced"].reindex(df.index)).abs().max()
            d2 = (df[f"cot_minus_requested_{sc_}"] - m[f"{sc_}_gain_vs_requested"].reindex(df.index)).abs().max()
            out["checks"][f"{t}/{sc_}"] = {"max_abs_diff_gain_vs_forced": float(d1),
                                           "max_abs_diff_gain_vs_requested": float(d2)}
        df["w"] = m["w_discovery"].reindex(df.index)
        for a in ("requested", "forced", "cot"):
            df[f"readable_{a}"] = c[f"{a}_compliant"]
        for sc_ in SCORES:
            df[f"residual_{sc_}"] = df[f"cot_minus_requested_{sc_}"] - df[f"corrected_gain_{sc_}"]
        out["tasks"][t] = df
        out["basis"][t] = {"strict": "discovery", "content": "discovery"}
        out["sources"][t] = [PANEL_FILES["compliance"].format(task=t), PANEL_FILES["mixture"]]
    r = pd.read_csv(root / PANEL_FILES["hotpot_req"]).set_index("model_key")
    fo = pd.read_csv(root / PANEL_FILES["hotpot_forced"]).set_index("model_key")
    mech = json.loads((root / PANEL_FILES["hotpot_mechanism"]).read_text(encoding="utf-8"))["per_model"]
    hc = pd.read_csv(root / PANEL_FILES["hotpot_content"]).pivot(index="model_key", columns="arm", values="content_correct")
    df = pd.DataFrame(index=fo.index)
    df["requested_strict"] = r["baseline"]
    df["forced_strict"] = fo["baseline"]
    df["cot_strict"] = fo["intervention"]
    df["cot_minus_requested_strict"] = r["gain"].reindex(df.index)
    df["cot_minus_forced_strict"] = fo["gain"]
    df["w"] = pd.Series({k: v["w"] for k, v in mech.items()}).reindex(df.index)
    df["corrected_gain_strict"] = pd.Series({k: v["predicted_gain_vs_requested"] for k, v in mech.items()}).reindex(df.index)
    out["checks"]["hotpot/strict"] = {
        "max_abs_diff_gain_vs_forced": float((df["cot_minus_forced_strict"]
                                              - pd.Series({k: v["gain_vs_forced"] for k, v in mech.items()})
                                              .reindex(df.index)).abs().max()),
        "max_abs_diff_cot_between_files": float((r["intervention"].reindex(df.index) - df["cot_strict"]).abs().max())}
    df["requested_content"] = hc["direct_answer"].reindex(df.index)
    df["forced_content"] = hc["forced_direct"].reindex(df.index)
    df["cot_content"] = hc["zero_shot_cot"].reindex(df.index)
    df["cot_minus_requested_content"] = df["cot_content"] - df["requested_content"]
    df["cot_minus_forced_content"] = df["cot_content"] - df["forced_content"]
    df["corrected_gain_content"] = (1 - df["w"]) * df["cot_minus_forced_content"]
    for a in ("requested", "forced", "cot"):
        df[f"readable_{a}"] = np.nan          # no per-model compliance table for HotpotQA in the panel's results
    for sc_ in SCORES:
        df[f"residual_{sc_}"] = df[f"cot_minus_requested_{sc_}"] - df[f"corrected_gain_{sc_}"]
    out["tasks"]["hotpot"] = df
    out["basis"]["hotpot"] = {"strict": "discovery", "content": "pooled"}
    out["sources"]["hotpot"] = [PANEL_FILES["hotpot_req"], PANEL_FILES["hotpot_forced"], PANEL_FILES["hotpot_mechanism"],
                                PANEL_FILES["hotpot_content"]]
    out["exact"], out["grid"], out["off_grid"] = {}, {}, []
    for t in TASKS:
        out["grid"][t] = panel_grid(t)
        out["exact"][t] = panel_exact(t, out["tasks"][t], out["grid"][t], out["off_grid"])
    return out


def panel_grid(t: str) -> dict:
    """The grid the panel's values of one benchmark lie on, from the design: items per configuration (the design's
    rows of an arm over its configurations), the discovery draws, the configurations. A discovery-draw mean in pp is
    100 m / (draws x items), w is m / (draws x items), a pooled mean in pp (HotpotQA content) is
    100 m / (configurations x items), m an integer."""
    disc, val = npr.sampler(t)
    configs = len(set(disc + val))
    items = Fraction(DESIGN_ROWS[t]["forced_direct"], configs)
    if items.denominator != 1:
        raise ValueError(f"{t}: {DESIGN_ROWS[t]['forced_direct']} design rows over {configs} configurations")
    n = int(items)
    return {"items_per_config": n, "discovery_draws": len(disc), "configs": configs,
            "discovery_pp": Fraction(len(disc) * n, 100), "w": Fraction(len(disc) * n), "pooled_pp": Fraction(configs * n, 100)}


def on_grid(x, G: Fraction):
    """The exact ratio m/G a file's float stands for, m = round(x G), when x G lies within 1e-6 of m; None when the
    float is off the grid (or missing)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    y = Fraction(float(x)) * G
    m = round(y)
    return Fraction(m) / G if abs(y - m) <= Fraction(1, 10 ** 6) else None


def panel_exact(t: str, df: pd.DataFrame, grid: dict, off: list) -> pd.DataFrame:
    """The panel's table of one benchmark with every value as its exact ratio: the arms, readability and the file's own
    gains on their grid, w on its grid, differences and the corrected gain (1 - w) x (CoT - forced) computed exactly and
    checked against the file's float (1e-9). A value off its grid, or a product that does not match its file, keeps the
    file's float and is listed (off)."""
    ex = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for k in df.index:
        def put(col, q):
            f = df.at[k, col]
            if q is None and not (isinstance(f, float) and math.isnan(f)):
                off.append(f"panel {TASK_NAME[t]} {col} {k}: {f!r} is not an exact ratio here (off its grid, or built "
                           "from a value off its grid); printed from the float")
                q = float(f)
            elif q is not None and abs(float(q) - float(f)) > TOL:
                off.append(f"panel {TASK_NAME[t]} {col} {k}: exact {q} against the file's {f!r}; printed from the float")
                q = float(f)
            ex.at[k, col] = q
        for sc_ in SCORES:
            G = grid["pooled_pp"] if (t == "hotpot" and sc_ == "content") else grid["discovery_pp"]
            arm = {a: on_grid(df.at[k, f"{a}_{sc_}"], G) for a in ("requested", "forced", "cot")}
            for a in arm:
                put(f"{a}_{sc_}", arm[a])
            if t == "hotpot" and sc_ == "strict":        # the file's own gains (per_model_cot_{req,forced}.csv)
                g_req, g_fd = (on_grid(df.at[k, f"{q}_strict"], G) for q in ("cot_minus_requested", "cot_minus_forced"))
            else:
                g_req, g_fd = sub_(arm["cot"], arm["requested"]), sub_(arm["cot"], arm["forced"])
            put(f"cot_minus_requested_{sc_}", g_req)
            put(f"cot_minus_forced_{sc_}", g_fd)
        w = on_grid(df.at[k, "w"], grid["w"])
        put("w", w)
        for sc_ in SCORES:
            g_fd, g_req = ex.at[k, f"cot_minus_forced_{sc_}"], ex.at[k, f"cot_minus_requested_{sc_}"]
            exact_ok = isinstance(w, Fraction) and isinstance(g_fd, Fraction)
            cg = (1 - w) * g_fd if exact_ok else None
            put(f"corrected_gain_{sc_}", cg)
            cg = ex.at[k, f"corrected_gain_{sc_}"]
            res = g_req - cg if isinstance(g_req, Fraction) and isinstance(cg, Fraction) else None
            put(f"residual_{sc_}", res)
        for a in ("requested", "forced", "cot"):
            put(f"readable_{a}", on_grid(df.at[k, f"readable_{a}"], grid["discovery_pp"]))
    return ex


def ladder_value(s: dict, q: str, score: str, basis: str) -> float | None:
    """One quantity of a ladder model on a given basis (discovery draws, all 48 draws or pooled rows)."""
    if q == "w":
        return nz(s["w"]["discovery"])
    b = s[score][basis]
    if q in ("requested", "forced", "cot"):
        return nz(b["accuracy_pp"][q])
    if q == "cot_minus_requested":
        return nz(b["gain_over_requested_pp"])
    if q == "cot_minus_forced":
        return nz(b["gain_over_forced_pp"])
    if q == "corrected_gain":
        return nz(b["corrected_gain_pp"])
    if q == "residual":
        return sub_(nz(b["gain_over_requested_pp"]), nz(b["corrected_gain_pp"]))
    raise KeyError(q)


def median_of(vals: list):
    """Median of a list (exact when its values are): the middle value, or the mean of the two middle ones."""
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def place(v, panel) -> dict:
    """Where a value sits among the panel's (a Series of exact values or floats; missing ones skipped): counts below,
    equal and above within 1e-9, the mark, and the panel's min, median and max as exact as the values are."""
    vals = [x for x in (panel.tolist() if hasattr(panel, "tolist") else list(panel))
            if x is not None and not (isinstance(x, float) and math.isnan(x))]
    fl = pd.Series([float(x) for x in vals], dtype=float)
    if not vals:
        return {"panel_n": 0, "panel_min": None, "panel_q25": None, "panel_median": None, "panel_q75": None,
                "panel_max": None, "value": v, "below": None, "equal": None, "above": None, "outside_panel_range": None}
    d = {"panel_n": len(vals), "panel_min": min(vals), "panel_q25": float(fl.quantile(0.25)),
         "panel_median": median_of(vals), "panel_q75": float(fl.quantile(0.75)), "panel_max": max(vals), "value": v}
    if v is None:
        d.update(below=None, equal=None, above=None, outside_panel_range=None)
        return d
    vf = float(v)
    d["below"] = int((fl < vf - TOL).sum())
    d["equal"] = int(((fl - vf).abs() <= TOL).sum())
    d["above"] = int((fl > vf + TOL).sum())
    d["outside_panel_range"] = "above" if vf > float(d["panel_max"]) + TOL else (
        "below" if vf < float(d["panel_min"]) - TOL else None)
    return d


def placements(models: dict, panel: dict) -> dict:
    out = {}
    for key, m in models.items():
        for t, tb in m["tasks"].items():
            if "cut" not in tb:
                continue
            pdf = panel["exact"][t]
            for sc_ in SCORES:
                basis = panel["basis"][t][sc_]
                for q in PLACED:
                    col = "w" if q == "w" else f"{q}_{sc_}"
                    if q == "w" and sc_ == "content":
                        continue
                    for r in READINGS:
                        if r not in tb:
                            continue
                        v = ladder_value(tb[r], q, sc_, basis)
                        out.setdefault(key, {}).setdefault(t, {}).setdefault(r, {}).setdefault(sc_, {})[q] = \
                            dict(place(v, pdf[col]), basis=basis)
            for a in READ_ARMS:
                fr = pdf[f"readable_{a}"].dropna()
                if not len(fr):
                    continue
                for r in READINGS:
                    if r not in tb:
                        continue
                    v = placed_value(tb[r], "readable", a)
                    out[key][t][r].setdefault("readable", {})[a] = dict(place(v, fr), basis="discovery")
    return out


def placed_value(s: dict, score: str, q: str, basis: str = "discovery"):
    """A placed quantity of one reading on a basis: a score's quantity (ladder_value), or for score "readable" the
    readable share of arm q (the panel has it on the discovery draws only)."""
    if score == "readable":
        return nz(s["readable_pct"].get(q, {}).get(basis))
    return ladder_value(s, q, score, basis)


# --------------------------------------------------------------------------------------------------------------------
# DeepSeek against its own report
# --------------------------------------------------------------------------------------------------------------------

def reproduce_nebius(ds: dict, nebius: dict) -> dict:
    """Compare every DeepSeek value computed here with results/frontier_panel/nebius_panel.json."""
    cmp, bad = 0, []

    def check(path, a, b):
        nonlocal cmp
        cmp += 1
        if a is None or b is None:
            if not (a is None and b is None):
                bad.append({"path": path, "here": a, "report": b})
            return
        if isinstance(a, (int, float, Fraction)) and isinstance(b, (int, float, Fraction)):
            if abs(float(a) - float(b)) > TOL:
                bad.append({"path": path, "here": a, "report": b})
        elif a != b:
            bad.append({"path": path, "here": a, "report": b})

    for t in TASKS:
        for r, src in (("cut", "tasks"), ("uncut", "tasks_uncut")):
            rep = nebius.get(src, {}).get(t)
            mine = ds["tasks"].get(t, {}).get(r)
            if rep is None or mine is None:
                bad.append({"path": f"{t}/{r}", "here": mine is not None, "report": rep is not None})
                continue
            for k in ("configs_design", "configs_answered", "units_answered"):
                check(f"{t}/{r}/{k}", mine[k], rep[k])
            for k in ("discovery", "all_draws"):
                check(f"{t}/{r}/w/{k}", mine["w"][k], rep["w"][k])
            for sc_ in SCORES:
                for lab in ("all_draws", "discovery"):
                    for a in ("requested", "forced", "cot"):
                        check(f"{t}/{r}/{sc_}/{lab}/{a}", mine[sc_][lab]["accuracy_pp"][a],
                              rep[sc_][lab]["accuracy_pp"][a])
                    for k in ("gain_over_requested_pp", "gain_over_forced_pp", "corrected_gain_pp"):
                        check(f"{t}/{r}/{sc_}/{lab}/{k}", mine[sc_][lab][k], rep[sc_][lab][k])
                for q, v in rep[sc_]["template"].items():
                    for k, x in v.items():
                        check(f"{t}/{r}/{sc_}/template/{q}/{k}", mine[sc_]["template"][q][k], x)
            fs_rep = nebius.get("fewshot", {}).get(t, {}).get(r, {})
            for arm, d in fs_rep.items():
                for k, x in d.items():
                    check(f"{t}/{r}/fewshot/{arm}/{k}", mine["fewshot"].get(arm, {}).get(k), x)
    return {"values_compared": cmp, "mismatches": len(bad), "first_mismatches": bad[:20], "tolerance": TOL,
            "source": NEBIUS_JSON}


# --------------------------------------------------------------------------------------------------------------------
# Cut against uncut, family ladders, findings
# --------------------------------------------------------------------------------------------------------------------

def cut_vs_uncut(models: dict, place_: dict) -> dict:
    """Per model and benchmark, every quantity placed against the panel (the list is the placements' own: per score
    the seven quantities and the residual, and the readable shares where the panel has them): its value under both
    readings on the discovery draws (so the corrected gain is the rule's); the placement against the panel under each,
    on the panel's basis; pairs of ladder models whose order flips between the readings."""
    diffs, flips, rank_changes = [], [], []
    for key, m in models.items():
        for t, tb in m["tasks"].items():
            if not all(r in tb for r in READINGS):
                continue
            pc = place_.get(key, {}).get(t, {})
            for sc_, qs in pc.get("cut", {}).items():
                for q in qs:
                    basis = "discovery"
                    a, b = placed_value(tb["cut"], sc_, q, basis), placed_value(tb["uncut"], sc_, q, basis)
                    diffs.append({"model": key, "task": t, "score": sc_, "quantity": q, "cut": a, "uncut": b,
                                  "uncut_minus_cut": None if None in (a, b) else b - a})
                    x, y = pc["cut"][sc_][q], pc.get("uncut", {}).get(sc_, {}).get(q)
                    if y is not None and (x["below"], x["above"]) != (y["below"], y["above"]):
                        rank_changes.append({"model": key, "task": t, "score": sc_, "quantity": q,
                                             "basis": x["basis"], "cut_below": x["below"],
                                             "uncut_below": y["below"], "cut_equal": x["equal"],
                                             "uncut_equal": y["equal"], "panel_n": x["panel_n"]})
    def sign(x) -> int:                       # on exact values; a float difference within 1e-9 counts as 0
        x = round(x, 9) if isinstance(x, float) else x
        return (x > 0) - (x < 0)
    groups: dict = {}
    for d in diffs:
        if d["cut"] is not None and d["uncut"] is not None:
            groups.setdefault((d["task"], d["score"], d["quantity"]), []).append(d)
    ties = []          # a pair tied in one reading that separates in the other: reported as such, not as a flip
    for (t, sc_, q), rows in sorted(groups.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                sc, su = sign(a["cut"] - b["cut"]), sign(a["uncut"] - b["uncut"])
                if sc == su:
                    continue
                rec = {"task": t, "score": sc_, "quantity": q, "models": [a["model"], b["model"]],
                       "cut": [a["cut"], b["cut"]], "uncut": [a["uncut"], b["uncut"]]}
                if sc and su:                  # both readings order the pair, in opposite ways
                    flips.append(rec)
                else:
                    ties.append(dict(rec, tied_in="cut" if sc == 0 else "uncut"))
    mx, top = largest_changes(diffs)
    mx_s, top_s = largest_changes(diffs, readable=False)
    return {"differences": diffs, "ordering_flips": flips, "ties_separating": ties, "panel_rank_changes": rank_changes,
            "max_abs_difference": mx, "largest": [{k: d[k] for k in ("model", "task", "score", "quantity")}
                                                  for d in top],
            "max_abs_difference_scores": mx_s,
            "largest_scores": [{k: d[k] for k in ("model", "task", "score", "quantity")} for d in top_s]}


def same_value(a, b) -> bool:
    """Equal as exact ratios; within 1e-9 when either is a float (the script's tie tolerance)."""
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    return abs(float(a) - float(b)) <= TOL


def largest_changes(diffs: list, readable: bool | None = None) -> tuple:
    """The largest |uncut − cut| and every quantity that shares it (ties named, never one picked); readable=False
    leaves out the readable shares, True keeps only them."""
    ds = [d for d in diffs if d["uncut_minus_cut"] is not None
          and (readable is None or (d["score"] == "readable") == readable)]
    if not ds:
        return None, []
    mx = max(abs(d["uncut_minus_cut"]) for d in ds)
    return mx, [d for d in ds if same_value(abs(d["uncut_minus_cut"]), mx)]


def change_names(top: list) -> str:
    """The quantities that share a largest change, grouped by model and benchmark: 'Model, Benchmark: strict CoT,
    content CoT'; groups joined by '; '."""
    groups: dict = {}
    for d in top:
        groups.setdefault((d["model"], d["task"]), []).append(f"{d['score']} {Q_LABEL[d['quantity']]}")
    return "; ".join(f"{nm(k)}, {TASK_NAME[t]}: {', '.join(v)}" for (k, t), v in groups.items())


PLACED_WORDS = ("section 6's rows: per score the three arms, the two gains, w, the corrected gain and the residual; "
                "the readable share of each zero-shot arm on GSM8K, MATH-500 and BIG-Bench Hard")


def top_of(cvu: dict, with_readable: bool = True) -> list:
    """The quantities that share the largest change, with the readable shares or without them."""
    return cvu.get("largest" if with_readable else "largest_scores", [])


def largest_words(mx, top: list) -> str:
    """'8.24 pp (Model, Benchmark: strict CoT)' or, when several quantities share it, '8.24 pp, shared by 4
    quantities (...)'."""
    if mx is None:
        return "–"
    return (f"{fnum(mx, 2)} pp" + (f", shared by {len(top)} quantities" if len(top) > 1 else "")
            + f" ({change_names(top)})")


def without_readable(xs: list) -> str:
    """' (n without the readable shares)' when some of the listed changes are readable shares; empty otherwise."""
    n = sum(1 for x in xs if x["score"] != "readable")
    return f" ({n} without the readable shares)" if n != len(xs) else ""


def order_words(cvu: dict) -> str:
    """'n pairs of ladder models change order' and, when there are any, the pairs tied in one reading that separate
    in the other, counted apart."""
    ts = cvu.get("ties_separating", [])
    txt = f"{len(cvu['ordering_flips'])} pairs of ladder models change order" + without_readable(cvu["ordering_flips"])
    for r, other in (("cut", "uncut"), ("uncut", "cut")):
        n = sum(x["tied_in"] == r for x in ts)
        if n:
            txt += f"; {n} pair{'s' if n > 1 else ''} tied in the {r} reading separate{'' if n > 1 else 's'} in the " \
                   f"{other}"
    return txt


def family_tables(models: dict, panel: dict, stop: dict | None = None) -> dict:
    """Each family ladder, smallest rung first: the panel rungs from the panel's files, the ladder rungs computed here,
    both on the panel file's basis; missing rungs marked; the direction of each quantity over the rungs present, the
    content steps of the forced-arm quantities on GSM8K and BIG-Bench Hard judged against the stop's loss."""
    stop = stop or {}
    out = {}
    for fam, spec in FAMILY_LADDERS.items():
        f = {"rungs": [], "context": [], "context_note": spec["context_note"], "direction": {}}
        for key, src in spec["rungs"]:
            f["rungs"].append(rung_row(key, src, models, panel))
        for key in spec["context"]:
            f["context"].append(rung_row(key, "panel", models, panel))
        present = [r for r in f["rungs"] if r["present"]]
        f["rungs_present"] = len(present)
        f["rungs_total"] = len(f["rungs"])
        for t in TASKS:
            for sc_ in SCORES:
                for q in QUANTITIES:
                    if q == "w" and sc_ == "content":
                        continue
                    rv = [(r, r["values"].get(t, {}).get(sc_, {}).get(q)) for r in present]
                    rv = [(r, v) for r, v in rv if v is not None]
                    vals = [v for _, v in rv]
                    kind = "w" if q == "w" else "pp"
                    rg = stop_ranges(rv, t, sc_, q, stop,
                                     w_of=lambda r, t=t: r["values"].get(t, {}).get("strict", {}).get("w"))
                    f["direction"].setdefault(t, {}).setdefault(sc_, {})[q] = {
                        "values": vals, "keys": [r["key"] for r, _ in rv], "basis": basis_of(t, sc_),
                        "steps": [b - a for a, b in zip(vals, vals[1:])],
                        "ranges": rg, "step_words": step_words(vals, kind, rg),
                        "direction": direction_of(vals, kind, rg)}
        out[fam] = f
    return out


def family_uncut_changes(models: dict, panel: dict, fam: dict, stop: dict | None = None) -> list[dict]:
    """Family directions recomputed with the ladder rungs' uncut values (panel rungs as they are); the ones that
    change against the cut reading. The stop's loss bounds both readings alike (the forced arms
    are never cut at the cap)."""
    stop = stop or {}
    out = []
    for name, spec in FAMILY_LADDERS.items():
        for t in TASKS:
            for sc_ in SCORES:
                basis = panel["basis"][t][sc_]
                for q in QUANTITIES:
                    if q == "w" and sc_ == "content":
                        continue
                    cut_v, unc_v, any_ladder, keys_, w_c, w_u = [], [], False, [], {}, {}
                    sbasis = panel["basis"][t]["strict"]
                    for key, src in spec["rungs"]:
                        if src == "panel":
                            pdf = panel["exact"][t]
                            if key not in pdf.index:
                                continue
                            v = nz(pdf.at[key, "w" if q == "w" else f"{q}_{sc_}"])
                            cut_v.append(v)
                            unc_v.append(v)
                            keys_.append(key)
                            w_c[key] = w_u[key] = nz(pdf.at[key, "w"])
                            continue
                        tb = models.get(key, {}).get("tasks", {}).get(t, {})
                        if "cut" not in tb or "uncut" not in tb:
                            continue
                        any_ladder = True
                        cut_v.append(ladder_value(tb["cut"], q, sc_, basis))
                        unc_v.append(ladder_value(tb["uncut"], q, sc_, basis))
                        keys_.append(key)
                        w_c[key] = ladder_value(tb["cut"], "w", "strict", sbasis)
                        w_u[key] = ladder_value(tb["uncut"], "w", "strict", sbasis)
                    if not any_ladder or None in cut_v or None in unc_v:
                        continue
                    kind = "w" if q == "w" else "pp"
                    rc = stop_ranges(list(zip(keys_, cut_v)), t, sc_, q, stop, key_of=lambda k: k, w_of=w_c.get)
                    ru = stop_ranges(list(zip(keys_, unc_v)), t, sc_, q, stop, key_of=lambda k: k, w_of=w_u.get)
                    dc, du = direction_of(cut_v, kind, rc), direction_of(unc_v, kind, ru)
                    if dc != du:
                        out.append({"family": name, "task": t, "score": sc_, "quantity": q, "basis": basis,
                                    "cut": cut_v, "uncut": unc_v, "cut_direction": dc, "uncut_direction": du})
    return out


def cap_dependence(models: dict, place_: dict) -> dict:
    """Section 1's findings recomputed with the uncut values, each on its own basis (the placements': discovery draws,
    HotpotQA content pooled rows): the residual band (content residual within ±RESIDUAL_BAND), the ↑/↓ mark of every
    placed quantity, the side of the panel median of CoT − forced (content), and the requested arm's readable share
    against READABLE_BELOW (discovery draws). Each part lists what differs between the readings."""
    band = {b: {"cut": [], "uncut": [], "values": []} for b in ("discovery", "pooled")}
    n_band = {"discovery": 0, "pooled": 0}
    marks, median, low = [], [], {r: [] for r in READINGS}
    for k, m in models.items():
        for t in TASKS:
            pc = place_.get(k, {}).get(t, {})
            if "cut" not in pc or "uncut" not in pc:
                continue
            b = basis_of(t, "content")
            n_band[b] += 1
            band[b]["values"].append((k, t, tuple(pc[r]["content"]["residual"]["value"] for r in READINGS)))
            for r in READINGS:
                v = pc[r]["content"]["residual"]["value"]
                if v is not None and abs(v) <= RESIDUAL_BAND:
                    band[b][r].append((k, t, v))
                rd = (m["tasks"][t][r].get("readable_pct", {}).get("requested") or {}).get("discovery")
                if rd is not None and rd < READABLE_BELOW:
                    low[r].append((k, t))
            for sc_, qs in pc["cut"].items():
                for q, x in qs.items():
                    y = pc["uncut"].get(sc_, {}).get(q)
                    if y is not None and x["outside_panel_range"] != y["outside_panel_range"]:
                        marks.append({"model": k, "task": t, "score": sc_, "quantity": q, "basis": x["basis"],
                                      "cut": x, "uncut": y})
            x, y = pc["cut"]["content"]["cot_minus_forced"], pc["uncut"]["content"]["cot_minus_forced"]
            above = [p["value"] is not None and p["value"] > p["panel_median"] + TOL for p in (x, y)]
            if above[0] != above[1]:
                median.append({"model": k, "task": t, "basis": x["basis"], "cut": x, "uncut": y})
    return {"band": band, "band_cells": n_band, "marks": marks, "median": median, "readable_low": low}


def cap_line(R: dict) -> str | None:
    """Section 1: which findings depend on the cap, from cap_dependence, in plain words."""
    cd = R["cut_vs_uncut"].get("cap_dependence")
    if not cd or not any(cd["band_cells"].values()):
        return None

    def band_part(b, words):
        n = cd["band_cells"][b]
        if not n:
            return None
        c, u = cd["band"][b]["cut"], cd["band"][b]["uncut"]
        cu = {(k, t): v for k, t, v in c}
        uu = {(k, t): v for k, t, v in u}
        vals = {(k, t): v for k, t, v in cd["band"][b]["values"]}
        moves = ([f"leaves: {nm(k)} {TASK_NAME[t]}, {fnum(cu[(k, t)])} cut, {fnum(vals[(k, t)][1])} uncut"
                  for k, t in cu if (k, t) not in uu]
                 + [f"enters: {nm(k)} {TASK_NAME[t]}, {fnum(vals[(k, t)][0])} cut, {fnum(uu[(k, t)])} uncut"
                    for k, t in uu if (k, t) not in cu])
        if len(c) == len(u) and not moves:
            return f"{words}: {len(c)} of {n} in both readings"
        return f"{words}: {len(c)} of {n} cut, {len(u)} of {n} uncut" + (f" ({'; '.join(moves)})" if moves else "")

    parts = [x for x in (band_part("discovery", "content, discovery draws"),
                         band_part("pooled", "HotpotQA content, pooled rows")) if x]
    lc, lu = cd["readable_low"]["cut"], cd["readable_low"]["uncut"]
    moved = ({t for b in cd["band"].values() for k, t, v in b["values"]
              if (k, t) in {(a, c) for a, c, _ in b["cut"]} ^ {(a, c) for a, c, _ in b["uncut"]}}
             | {x["task"] for x in cd["marks"] + cd["median"]} | {t for k, t in set(lc) ^ set(lu)})
    where = sorted(moved, key=TASKS.index)
    txt = (f"Findings above that depend on the cap ("
           + (f"all on {TASK_NAME[where[0]]}" if len(where) == 1 else
              "on " + ", ".join(TASK_NAME[t] for t in where) if where else "none")
           + f"), each recomputed with the uncut values on its own basis. Residual within ±{RESIDUAL_BAND:g} pp: "
           + "; ".join(parts) + ".")
    mk = cd["marks"]

    def qname(x):
        return f"{x['score']} {Q_LABEL[x['quantity']]}"
    txt += (" Marks ↑ or ↓ (outside the panel's range) that change: " + (
        "; ".join(f"{nm(x['model'])} {TASK_NAME[x['task']]} {qname(x)}, {BASIS_NAME[x['basis']]}: "
                  f"{cell(x['cut'], 3 if x['quantity'] == 'w' else 1)} cut, "
                  f"{cell(x['uncut'], 3 if x['quantity'] == 'w' else 1)} uncut" for x in mk) if mk else "none") + ".")
    md = cd["median"]
    txt += (" Side of the panel median of CoT − forced (content): " + (
        "; ".join(f"{nm(x['model'])} {TASK_NAME[x['task']]}, {BASIS_NAME[x['basis']]}: {cell(x['cut'])} cut, "
                  f"{cell(x['uncut'])} uncut" for x in md) if md else "the same in both readings") + ".")
    txt += (f" Requested replies under {READABLE_BELOW:g}% readable, discovery draws: "
            + (f"the same {len(lc)} cells in both readings" if sorted(lc) == sorted(lu) else
               f"{len(lc)} cells cut, {len(lu)} uncut ("
               + "; ".join([f"only cut: {nm(k)} {TASK_NAME[t]}" for k, t in lc if (k, t) not in lu]
                           + [f"only uncut: {nm(k)} {TASK_NAME[t]}" for k, t in lu if (k, t) not in lc]) + ")")
            + ". Pairs of ladder models that change order, placements that move and family directions that change: "
              "the two lines above and section 9.")
    return txt


def rung_row(key: str, src: str, models: dict, panel: dict) -> dict:
    row = {"key": key, "source": src, "values": {}}
    if src == "panel":
        row["present"] = True
        for t in TASKS:
            pdf = panel["exact"][t]
            if key not in pdf.index:
                continue
            for sc_ in SCORES:
                row["values"].setdefault(t, {})[sc_] = {
                    q: nz(pdf.at[key, "w" if q == "w" else f"{q}_{sc_}"]) for q in QUANTITIES + ("residual",)
                    if not (q == "w" and sc_ == "content")}
        return row
    m = models.get(key)
    row["present"] = bool(m and m["tasks"])
    row["status"] = m["status"] if m else "missing"
    if m:
        for t, tb in m["tasks"].items():
            if "cut" not in tb:
                continue
            for sc_ in SCORES:
                basis = panel["basis"][t][sc_]
                row["values"].setdefault(t, {})[sc_] = {
                    q: ladder_value(tb["cut"], q, sc_, basis) for q in QUANTITIES + ("residual",)
                    if not (q == "w" and sc_ == "content")}
    return row


def nm(key: str) -> str:
    return LADDER[key]["name"].split(" (")[0] if key in LADDER else key


def equiv_checks(root: Path) -> tuple[list[str], list[str], dict]:
    """The cut-equivalence checks on file (results/ladder/equiv_*.json): words for the batch-invariant ones, words for
    the default-kernel ones, and the default-kernel records by model. A default-kernel check reruns rows that the run
    cut at the panel's cap, so its words say so."""
    bi, dkw, dkr = [], [], {}
    for p in sorted(root.glob(EQUIV_GLOB)):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("kernels") == "default":
            s = d.get("scored") or {}
            f = s.get("content_flips") or {}
            dkr[d["model_key"]] = d
            dkw.append(f"{nm(d['model_key'])} (default kernels): {d['identical']} of {d['rows']} rerun rows identical "
                       f"to the cut copy, content verdict the same on {s.get('content_verdict_same', '–')} of "
                       f"{s.get('rows', '–')} ({f.get('wrong_to_right', '–')} wrong to right, "
                       f"{f.get('right_to_wrong', '–')} right to wrong), on rows the run cut at the panel's cap "
                       f"(`{rel(p, root)}`)")
        else:
            bi.append(f"{d['model_key']}: {d['identical']} of {d['rows']} rerun rows identical to the pilot's cut copy "
                      f"({d.get('identical_among_cut', '–')} of {d.get('cut_in_pilot', '–')} cut ones), `{rel(p, root)}`")
    return bi, dkw, dkr


def equiv_caveat(dkr: dict) -> str:
    """The default-kernel check's caveat: what its rates describe and why its verdict changes run one
    way."""
    right = "; ".join(f"{nm(k)}: {d['selection']['content_cut_right']} of {d['selection']['rows']} right under content"
                      for k, d in dkr.items() if d.get("selection"))
    return ("These rows were chosen because the run cut them at the panel's cap (long, often runaway replies), so the "
            "rates describe those rows, not a random row, and the one-way verdict changes follow from that choice: a "
            "cut copy that runs to the cap is usually wrong (" + (right or "–") + "), so a rerun can gain an answer "
            "and has little to lose. Under default kernels a rerun also differs from the run for reasons that have "
            "nothing to do with the cut, so the check does not separate the two.")


def glance_rows(models: dict, place_: dict) -> list[dict]:
    """One row per (ladder model, benchmark): the headline quantities on the panel's basis, each with its placement;
    the requested arm's readability and content - strict gap on the discovery draws."""
    rows = []
    for k, m in models.items():
        for t in TASKS:
            p = place_.get(k, {}).get(t, {}).get("cut")
            if not p:
                continue
            s = models[k]["tasks"][t]["cut"]
            rows.append({"model": k, "task": t,
                         **{f"{q}_{sc_}": p[sc_][q] for sc_ in SCORES
                            for q in ("cot_minus_forced", "cot_minus_requested", "residual")},
                         "w": p["strict"]["w"],
                         "readable_requested": p.get("readable", {}).get("requested"),
                         "readable_requested_disc": s["readable_pct"].get("requested", {}).get("discovery"),
                         "gap_requested_disc": s["strict_content_gap_discovery_pp"]["requested"]})
    return rows


def short(key: str) -> str:
    return LADDER[key]["name"] if key in LADDER else key


def rung_label(key: str) -> str:
    return SIZE_LABEL.get(key, short(key))


DISC_TASKS = ("gsm8k", "math", "bbh")       # every score on the discovery draws in the panel's files


def arrow(vals: list, d: int = 1) -> str:
    return " → ".join(fnum(v, d) for v in vals)


def present_keys(models: dict) -> list[str]:
    return [k for k, m in models.items() if m["status"] != "missing"]


def family_line(name: str, f: dict) -> str:
    """One finding per family ladder: forced accuracy and the two gains across the rungs present (content); one
    sentence on the discovery draws (GSM8K, MATH-500, BIG-Bench Hard), one on pooled rows (HotpotQA)."""
    labels = [rung_label(r["key"]) for r in f["rungs"] if r["present"]]
    miss = [short(r["key"]) for r in f["rungs"] if not r["present"]]
    if f["rungs_present"] < 2:
        return (f"{name} ladder: {f['rungs_present']} of {f['rungs_total']} rungs in; no within-release direction "
                "can be read yet" + (f" (waits for {', '.join(miss)})." if miss else "."))

    def part(t):
        dd = f["direction"][t]["content"]
        return (f"forced {arrow(dd['forced']['values'])} ({dword(dd['forced']['direction'])}), CoT − forced "
                f"{arrow(dd['cot_minus_forced']['values'])} ({dword(dd['cot_minus_forced']['direction'])}), "
                f"CoT − requested {arrow(dd['cot_minus_requested']['values'])} "
                f"({dword(dd['cot_minus_requested']['direction'])})")
    return (f"{name} ladder ({' → '.join(labels)}; {f['rungs_present']} of {f['rungs_total']} rungs), content, "
            "discovery draws: " + "; ".join(f"{TASK_NAME[t]}: {part(t)}" for t in DISC_TASKS)
            + f". HotpotQA, content, pooled rows: {part('hotpot')}."
            + (f" Waits for {', '.join(miss)}." if miss else ""))


def fewshot_panel(root: Path) -> dict | None:
    """The panel's BIG-Bench Hard five-shot forced gap, content - strict over all rows of the arm, per model, from
    FEWSHOT_PANEL (panel_scan; strict and content as the file gives them, to 0.01 pp, read as those decimals), and the
    check that its basis is the ladder's: every entry has the design's rows of the arm; the file's own scoring code
    reproduces the ladder's saved Mistral Large 3 rows (tasks.bbh.pipeline_matches_saved); the decoding in every panel
    run log of the arm (cap, stop string, prefill) equals the ladder's five-shot forced decoding. None without the file."""
    import yaml
    p = root / FEWSHOT_PANEL
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    ent = {v["model"]: v for v in d.get("panel_scan", {}).values() if v.get("arm") == "fewshot_forced"}

    def dec(x):
        q = Fraction(repr(float(x)))
        if (q / FEWSHOT_PANEL_STEP).denominator != 1:
            raise ValueError(f"{FEWSHOT_PANEL}: {x!r} is not given to 0.01 pp")
        return q
    gaps = {m: dec(v["content"]) - dec(v["strict"]) for m, v in sorted(ent.items())}
    n_design = DESIGN_ROWS["bbh"]["fewshot_forced"]
    rows = sorted({int(v["n"]) for v in ent.values()})
    check = {"rows_per_model": rows, "design_rows": n_design, "rows_ok": rows == [n_design],
             "scorer_ok": bool(d.get("tasks", {}).get("bbh", {}).get("pipeline_matches_saved"))}
    lad = root / LADDER_BBH_DECODING
    logs = sorted(root.glob(FEWSHOT_PANEL_LOGS))
    if lad.exists() and logs:
        y = yaml.safe_load(lad.read_text(encoding="utf-8"))
        want = {"max_new_tokens": y["max_new_tokens"]["fewshot_forced"],
                "stop": (y.get("arm_overrides", {}).get("fewshot_forced", {}) or {}).get("stop", y.get("stop")),
                "prefill": y["prefill"]["fewshot_forced"]}
        seen = []
        for f in logs:
            g = json.loads(f.read_text(encoding="utf-8")).get("decoding", {})
            seen.append({"max_new_tokens": g.get("max_new_tokens", {}).get("forced_direct"), "stop": g.get("stop"),
                         "prefill": g.get("prefill", {}).get("forced_direct")})
        check["caps"] = {"ladder": want, "panel_runs": len(logs),
                         "panel_runs_models": len({f.parent.name for f in logs}),
                         "panel": [dict(s) for s in {json.dumps(s, sort_keys=True): s for s in seen}.values()],
                         "same": all(s == want for s in seen)}
        check["caps_ok"] = check["caps"]["same"]
    else:
        check["caps"] = None
        check["caps_ok"] = None         # not checked here: the run logs or the ladder's decoding file are absent
    return {"source": FEWSHOT_PANEL, "precision_pp": FEWSHOT_PANEL_STEP, "gaps": gaps, "check": check}


def bbh_forced_gap(models: dict, panel: dict, root: Path) -> dict:
    """BIG-Bench Hard forced arms, content - strict, placed against the panel on the same basis: the zero-shot arm on
    the discovery draws (results/content_scoring/tables/compliance_bbh.csv), the five-shot arm over all rows of the arm
    (fewshot_panel). A ladder value within the five-shot panel values' rounding (0.01 pp) of a panel gap is listed
    (its count is not settled by the file)."""
    pdf = panel["exact"]["bbh"]
    gap = pd.Series({k: pdf.at[k, "forced_content"] - pdf.at[k, "forced_strict"] for k in pdf.index
                     if pdf.at[k, "forced_content"] is not None and pdf.at[k, "forced_strict"] is not None},
                    dtype=object)
    p0 = place(None, gap)
    out = {"panel": {"n": p0["panel_n"], "min": p0["panel_min"], "median": p0["panel_median"], "max": p0["panel_max"],
                     "basis": "discovery"}, "models": {}}
    fp = fewshot_panel(root)
    if fp:
        g5 = pd.Series(fp["gaps"], dtype=object)
        q0 = place(None, g5)
        out["fewshot_panel"] = {**{k: v for k, v in fp.items() if k != "gaps"}, "gaps": fp["gaps"],
                                "n": q0["panel_n"], "min": q0["panel_min"], "median": q0["panel_median"],
                                "max": q0["panel_max"], "basis": "all_rows"}
    for k in present_keys(models):
        s = models[k]["tasks"].get("bbh", {}).get("cut")
        if not s:
            continue
        v5 = s["strict_content_gap_pp"].get("fs_forced")
        out["models"][k] = {"zero_shot_forced_discovery": dict(place(s["strict_content_gap_discovery_pp"]["forced"],
                                                                     gap), basis="discovery"),
                            "fewshot_forced_all_rows": v5}
        if fp and v5 is not None:
            out["models"][k]["fewshot_forced_all_rows_placement"] = dict(place(v5, g5), basis="all_rows")
            out["models"][k]["fewshot_within_panel_rounding"] = sorted(
                m for m, g_ in fp["gaps"].items() if abs(v5 - g_) <= FEWSHOT_PANEL_STEP)
    return out


def pct(x) -> str:
    """A share in percent, or the reason there is none (a strict value of 0 has no share)."""
    return "– (strict value 0)" if x is None else f"{fnum(x)}%"


def format_shares(R: dict) -> list[dict]:
    """(model, benchmark) cells whose requested replies are under READABLE_BELOW% readable by the strict rule
    (discovery draws), with the answer format's share of strict CoT - requested and of the strict residual: strict
    minus content as a share of the strict value, both scores on the discovery draws (exact; None when the strict
    value is 0)."""
    out = []
    for r in glance_rows(R["models"], R["placement"]):
        if r["readable_requested_disc"] is None or r["readable_requested_disc"] >= READABLE_BELOW:
            continue
        x = R["models"][r["model"]]["tasks"][r["task"]]["cut"]
        g = {s: x[s]["discovery"]["gain_over_requested_pp"] for s in SCORES}
        res = {s: sub_(g[s], x[s]["discovery"]["corrected_gain_pp"]) for s in SCORES}

        def share(v):
            return None if v["strict"] is None or v["content"] is None or v["strict"] == 0 else \
                100 * (v["strict"] - v["content"]) / v["strict"]
        out.append({"model": r["model"], "task": r["task"], "readable": r["readable_requested_disc"],
                    "gap": r["gap_requested_disc"], "gain": g, "residual": res,
                    "share_gain": share(g), "share_residual": share(res)})
    return out


def size_claim(R: dict) -> str:
    """Whether the records support calling DeepSeek the top of the size range: its published size
    against the largest recorded total and active parameters among the other ladder models."""
    o, bits = ds_size_compare(R)
    if o is None:
        return "It is listed as the top of the size range; no size is recorded for it."
    head = (f"It is listed as the top of the size range; its published size is {active_phrase(o)} "
            f"({o['caveat'] + ', ' if o.get('caveat') else ''}{o['url']})")
    return head + (", " + " and ".join(bits) + ", so the records do not support that." if bits else
                   ", the largest recorded in total and in active parameters.")


def active_phrase(o: dict) -> str:
    """'552B total, with 8B active for input and 16B for output' or '<t>B total and <a>B active'."""
    if o.get("active_input") and o.get("active_output"):
        return (f"{o['total']}B total, with {o['active_input']}B active for input and {o['active_output']}B for "
                "output")
    return f"{o['total']}B total and {o['active']}B active" if o.get("active") else f"{o['total']}B total"


def largest_active(o: dict) -> Fraction | None:
    """The largest active-parameter count a size records (for input and output apart, the larger)."""
    xs = [Fraction(o[f]) for f in ("active", "active_input", "active_output") if o.get(f)]
    return max(xs) if xs else None


def ds_size_compare(R: dict) -> tuple[dict | None, list[str]]:
    """DeepSeek's published size against the largest recorded total and active parameters among the other ladder
    models (its active parameters: the larger of input and output): (its size, the phrases where it is below another
    model)."""
    sz = R["sizes"]
    o = sz.get("deepseek-v4.1-flash-1017") or {}
    if o.get("total") is None:
        return None, []
    others = [(k, v) for k, v in sz.items() if k in LADDER and k != "deepseek-v4.1-flash-1017"]
    tot = max(((Fraction(v["total"]), k) for k, v in others if v.get("total")), default=None)
    act = max(((Fraction(v["active"]), k) for k, v in others if v.get("active")), default=None)
    bits = []
    if tot and tot[0] > Fraction(o["total"]):
        bits.append(f"below {nm(tot[1])} in total parameters ({sz[tot[1]]['total']}B)")
    if act and largest_active(o) is not None and act[0] > largest_active(o):
        bits.append(f"below {nm(act[1])} in active parameters ({sz[act[1]]['active']}B, {sz[act[1]]['arch']})")
    return o, bits


def loss_steps(R: dict) -> list[dict]:
    """The family steps within the stop's loss (DIRECTION_RULE): content, GSM8K and BIG-Bench Hard, the forced-arm
    quantities, in family order."""
    stop = R.get("stop_unread") or {}
    out = []
    for name, f in R["families"].items():
        for t in STOP_TASKS:
            for q in ("forced", "cot_minus_forced", "corrected_gain"):
                d = f["direction"].get(t, {}).get("content", {}).get(q)
                if not d or not d.get("ranges"):
                    continue
                for j, w in enumerate(d["step_words"]):
                    if not w.endswith(WITHIN_LOSS):
                        continue
                    (alo, ahi), (blo, bhi) = d["ranges"][j], d["ranges"][j + 1]
                    a, b = d["keys"][j], d["keys"][j + 1]
                    out.append({"family": name, "task": t, "quantity": q, "from": a, "to": b,
                                "step": d["values"][j + 1] - d["values"][j], "low": blo - ahi, "high": bhi - alo,
                                "unread_from": stop[a][t]["discovery_pp"], "unread_to": stop[b][t]["discovery_pp"],
                                "unread_rows_from": stop[a][t]["unread"], "unread_rows_to": stop[b][t]["unread"]})
    return out


LOSS_Q = {"forced": "forced", "cot_minus_forced": "CoT − forced", "corrected_gain": "corrected gain"}


def replies(n: int) -> str:
    return f"{n:,} repl{'y' if n == 1 else 'ies'}"


def stop_loss_line(R: dict) -> str:
    """Section 1: the family steps the stop's unread replies could turn (DIRECTION_RULE), or that none can."""
    if not R.get("stop_unread"):
        return (f"The stop's loss is not checked: `{STOP_UNREAD}` is absent (run `scripts/forced_stop_audit.py`).")
    xs = loss_steps(R)
    head = ("Family steps within the stop's loss (content, discovery draws; the zero-shot forced replies the stop at "
            "the first `}` left unread, counted none or all correct; footnote F1)")
    if not xs:
        return head + ": none."
    return head + ": " + "; ".join(
        f"{x['family']} {TASK_NAME[x['task']]} {LOSS_Q[x['quantity']]}, {rung_label(x['from'])} → "
        f"{rung_label(x['to'])}: {fnum(x['step'], 2)} pp, between {fnum(x['low'], 2)} and {fnum(x['high'], 2)} pp "
        f"(unread: {replies(x['unread_rows_from'])}, {fnum(x['unread_from'], 2)} pp, of the {rung_label(x['from'])}'s "
        f"arm; {replies(x['unread_rows_to'])}, {fnum(x['unread_to'], 2)} pp, of the {rung_label(x['to'])}'s)"
        for x in xs) + (
        f". The stop cut these answers inside their box and the readers do not read them, so the forced arm does not settle "
        f"the direction of {'this step' if len(xs) == 1 else 'these steps'}.")


def findings(R: dict) -> list[str]:
    """Plain statements generated from the numbers, so a rerun keeps them true. Each sentence carries one basis and
    names it."""
    models, place_, fam, cvu, rep, comp = (R["models"], R["placement"], R["families"], R["cut_vs_uncut"],
                                           R["reproduction"], R["completeness"])
    present = present_keys(models)
    missing = [k for k, m in models.items() if m["status"] == "missing"]
    partial = [k for k, m in models.items() if m["status"] == "partial"]
    L = [f"{len(present)} of {len(models)} ladder models are in" + (": " + ", ".join(nm(k) for k in present)
                                                                       if present else "") + "."
         + (f" Partial: {', '.join(nm(k) for k in partial)}." if partial else "")
         + "".join(f" {nm(k)}: {comp[k]['state']}." for k in missing)]
    exact = [k for k in present if comp[k]["exact"]]
    short_ = [k for k in present if not comp[k]["exact"]]
    n_design = max((comp[k]["design_rows"] for k in comp), default=0)
    L.append(f"Completeness: {len(exact)} of {len(present)} models in have every one of the {n_design:,} design rows "
             "scored once in both readings, with no missing, duplicate or extra request id and no unscored row "
             "(section 4)." + (" Short: " + "; ".join(f"{nm(k)} ({comp[k]['state']})" for k in short_) + "."
                               if short_ else ""))
    for name, f in fam.items():
        L.append(family_line(name, f))
    L.append(stop_loss_line(R))
    rows = glance_rows(models, place_)

    def above_med(p):
        return p["value"] is not None and p["value"] > p["panel_median"] + TOL
    for k in present:
        rr = {r["task"]: r for r in rows if r["model"] == k}
        if not rr:
            continue
        dk = [t for t in DISC_TASKS if t in rr]
        txt = ""
        if dk:
            txt = (f"{nm(k)} against the panel, CoT − forced (content), discovery draws: above the panel median on "
                   f"{sum(above_med(rr[t]['cot_minus_forced_content']) for t in dk)} of {len(dk)} benchmarks; "
                   + "; ".join(f"{TASK_NAME[t]} {fnum(p['value'])} pp, {above_txt(p)}"
                               for t in dk for p in [rr[t]["cot_minus_forced_content"]]) + ".")
        if "hotpot" in rr:
            p = rr["hotpot"]["cot_minus_forced_content"]
            txt += ((" " if txt else f"{nm(k)} against the panel. ")
                    + f"HotpotQA, CoT − forced (content), pooled rows: {fnum(p['value'])} pp, {above_txt(p)}, "
                      f"{'above' if above_med(p) else 'not above'} the panel median.")
        L.append(txt)
    if rows:
        def names(rs):
            return ", ".join(f"{nm(r['model'])} {TASK_NAME[r['task']]}" for r in rs)

        def within(rs):
            return [r for r in rs if r["residual_content"]["value"] is not None
                    and abs(r["residual_content"]["value"]) <= RESIDUAL_BAND]

        def olist(items):
            return "; ".join(f"{nm(r['model'])} {TASK_NAME[r['task']]} {sc_} {fnum(r[f'residual_{sc_}']['value'])} "
                             f"(panel {fnum(r[f'residual_{sc_}']['panel_min'])} to "
                             f"{fnum(r[f'residual_{sc_}']['panel_max'])})" for r, sc_ in items)
        dr = [r for r in rows if r["task"] in DISC_TASKS]
        hr = [r for r in rows if r["task"] == "hotpot"]
        sd, sh = within(dr), within(hr)
        txt = ("Baseline shift (residual: observed CoT − requested minus the corrected gain (1 − w) × (CoT − forced)). "
               f"Content, discovery draws (GSM8K, MATH-500, BIG-Bench Hard): within ±{RESIDUAL_BAND:g} pp in "
               f"{len(sd)} of {len(dr)} (model, benchmark) cells" + (f" ({names(sd)})" if sd else "") + ".")
        if hr:
            txt += (f" HotpotQA content, pooled rows (the corrected gain there is {POOLED_CG}): within "
                    f"±{RESIDUAL_BAND:g} pp in {len(sh)} of {len(hr)} cells" + (f" ({names(sh)})" if sh else "") + ".")
        out_d = [(r, sc_) for r in rows for sc_ in SCORES
                 if basis_of(r["task"], sc_) == "discovery" and r[f"residual_{sc_}"]["outside_panel_range"]]
        out_p = [(r, "content") for r in hr if r["residual_content"]["outside_panel_range"]]
        if out_d:
            txt += " Outside the panel's range, discovery draws: " + olist(out_d) + "."
        if out_p:
            txt += " Outside the panel's range, HotpotQA content, pooled rows: " + olist(out_p) + "."
        L.append(txt)
    low = format_shares(R)
    if low:
        n = len(low)
        a = sum(r["share_residual"] is not None and r["share_residual"] > 50 for r in low)
        b = sum(r["share_gain"] is not None and r["share_gain"] > 50 for r in low)
        L.append(f"Requested replies the strict rule cannot read, discovery draws (under {READABLE_BELOW:g}% readable), "
                 "with the answer format's share of strict CoT − requested and of the strict residual (strict minus "
                 "content, as a share of the strict value, discovery draws): "
                 + "; ".join(f"{nm(r['model'])} {TASK_NAME[r['task']]} {fnum(r['readable'])}% readable, content − "
                             f"strict on the requested arm {fnum(r['gap'])} pp, format {pct(r['share_gain'])} of CoT − "
                             f"requested and {pct(r['share_residual'])} of the residual" for r in low)
                 + f". The format is more than half of the strict residual in {a} of {n} cells and more than half of "
                   f"strict CoT − requested in {b} of {n}; in these cells the content columns carry the comparison.")
    bg = R.get("bbh_forced_gap", {})
    if bg.get("models"):
        pn = bg["panel"]
        L.append("BIG-Bench Hard zero-shot forced arm, content − strict, discovery draws, value (b/M) against the "
                 "panel: " + "; ".join(f"{nm(k)} {cell(v['zero_shot_forced_discovery'], 2)}"
                                       for k, v in bg["models"].items())
                 + f". The panel's runs from {fnum(pn['min'], 2)} to {fnum(pn['max'], 2)} (median "
                   f"{fnum(pn['median'], 2)}, {pn['n']} models).")
        fp = bg.get("fewshot_panel")
        if fp:
            ck = fp["check"]
            fails = [w for w, ok in (("rows", ck["rows_ok"]), ("scorer", ck["scorer_ok"]), ("caps", ck["caps_ok"]))
                     if ok is False]
            basis = ("the same rows, scorer and caps" if not fails and ck["caps_ok"] else
                     "the same rows and scorer; caps not checked here" if not fails else
                     "basis check failed: " + ", ".join(fails))
            uns = [(k, v["fewshot_within_panel_rounding"]) for k, v in bg["models"].items()
                   if v.get("fewshot_within_panel_rounding")]
            L.append("BIG-Bench Hard five-shot forced arm, content − strict, all rows of the arm, value (b/M) against "
                     f"the panel's gaps on that basis ({basis}; `{fp['source']}`, panel values to "
                     f"{fnum(fp['precision_pp'], 2)} pp): "
                     + "; ".join(f"{nm(k)} {cell(v.get('fewshot_forced_all_rows_placement'), 2)}"
                                 for k, v in bg["models"].items())
                     + f". The panel's runs from {fnum(fp['min'], 2)} to {fnum(fp['max'], 2)} (median "
                       f"{fnum(fp['median'], 2)}, {fp['n']} models)."
                     + ("" if not uns else " Within the panel values' rounding of a panel gap, so the count is not "
                        "settled by the file: " + "; ".join(f"{nm(k)} ({', '.join(ms)})" for k, ms in uns) + ".")
                     + " Strict and content are reported side by side (footnote F1).")
        else:
            L.append(f"BIG-Bench Hard five-shot forced arm, content − strict, all rows of the arm (no panel file here, "
                     f"`{FEWSHOT_PANEL}`): " + "; ".join(f"{nm(k)} {fnum(v['fewshot_forced_all_rows'], 2)} pp"
                                                         for k, v in bg["models"].items())
                     + ". Strict and content are reported side by side (footnote F1).")
    if "deepseek-v4.1-flash-1017" in present:
        L.append("DeepSeek V4.1-Flash-1017 is a single model outside the three families, with no smaller DeepSeek of "
                 "its release; it carries no family trend. " + size_claim(R))
    mx = cvu["max_abs_difference"]
    if mx is not None:
        big = sorted({TASK_NAME[d["task"]] for d in cvu["differences"]
                      if d["uncut_minus_cut"] is not None and abs(d["uncut_minus_cut"]) >= CUT_BIG})
        rc = cvu["panel_rank_changes"]
        fdc = cvu.get("family_direction_changes", [])
        L.append(f"Cut against uncut, every placed quantity on the discovery draws ({PLACED_WORDS}): "
                 f"the largest change is {largest_words(mx, top_of(cvu))}"
                 + (f"; leaving out the readable shares, "
                    f"{largest_words(cvu['max_abs_difference_scores'], top_of(cvu, False))}"
                    if any(d["score"] == "readable" for d in top_of(cvu)) else "")
                 + f"; changes of {CUT_BIG:g} pp or more: " + (", ".join(big) if big else "none")
                 + f"; {order_words(cvu)} (section 9).")
        L.append("Placements against the panel that move when the uncut values are used: "
                 f"{sum(x['basis'] == 'discovery' for x in rc)} on the discovery draws"
                 f"{without_readable([x for x in rc if x['basis'] == 'discovery'])}, "
                 f"{sum(x['basis'] == 'pooled' for x in rc)} on pooled rows (HotpotQA content). Family directions "
                 f"(section 5) that change: {sum(x['basis'] == 'discovery' for x in fdc)} on the discovery draws, "
                 f"{sum(x['basis'] == 'pooled' for x in fdc)} on pooled rows (section 9).")
        cl = cap_line(R)
        if cl:
            L.append(cl)
    if rep:
        L.append(f"DeepSeek reproduction: {rep['values_compared']} values compared with `{NEBIUS_JSON}`, "
                 f"{rep['mismatches']} differ (tolerance {rep['tolerance']:g}).")
    return L


def glance_table(R: dict) -> list[str]:
    rows = glance_rows(R["models"], R["placement"])
    if not rows:
        return []
    L = []
    dr = [r for r in rows if r["task"] in DISC_TASKS]
    hr = [r for r in rows if r["task"] == "hotpot"]
    if dr:
        L += ["", "At a glance, GSM8K, MATH-500 and BIG-Bench Hard (cut reading; every column on the discovery draws, "
              "the basis of the panel's files; a cell reads value (b/M): above b of the M panel models, and value "
              "(b/M, k tied) when k panel values equal it; ↑ or ↓: outside the panel's range):", "",
              "| Model | Benchmark | CoT − forced, strict | CoT − forced, content | CoT − requested, strict | "
              "CoT − requested, content | w | Residual, strict | Residual, content | Requested readable (%) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in dr:
            L.append(f"| {nm(r['model'])} | {TASK_NAME[r['task']]} | {cell(r['cot_minus_forced_strict'])} | "
                     f"{cell(r['cot_minus_forced_content'])} | {cell(r['cot_minus_requested_strict'])} | "
                     f"{cell(r['cot_minus_requested_content'])} | {cell(r['w'], 3)} | {cell(r['residual_strict'])} | "
                     f"{cell(r['residual_content'])} | {cell(r['readable_requested'])} |")
    if hr:
        L += ["", "At a glance, HotpotQA (cut reading; strict columns and w on the discovery draws; content columns on "
              "pooled rows, the basis of the panel's HotpotQA content file, with the content residual taken against "
              f"{POOLED_CG}; the panel has no HotpotQA readability table, so readability has no placement):", "",
              "| Model | CoT − forced, strict (discovery draws) | CoT − forced, content (pooled rows) | CoT − requested, "
              "strict (discovery draws) | CoT − requested, content (pooled rows) | w (discovery draws) | Residual, "
              "strict (discovery draws) | Residual, content (pooled rows) | Requested readable, % (discovery draws) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in hr:
            L.append(f"| {nm(r['model'])} | {cell(r['cot_minus_forced_strict'])} | "
                     f"{cell(r['cot_minus_forced_content'])} | {cell(r['cot_minus_requested_strict'])} | "
                     f"{cell(r['cot_minus_requested_content'])} | {cell(r['w'], 3)} | {cell(r['residual_strict'])} | "
                     f"{cell(r['residual_content'])} | {fnum(r['readable_requested_disc'])} |")
    L += ["", "Residual: observed CoT − requested minus the corrected gain (1 − w) × (CoT − forced). It equals forced − "
          "requested + w × (CoT − forced), so with w near 0 a negative residual means the requested arm scored above "
          "the forced arm."]
    return L


def scope_lines(R: dict) -> list[str]:
    models, fam = R["models"], R["families"]
    n_in = len(present_keys(models))
    pn = sorted({int(df["cot_minus_forced_strict"].notna().sum()) for df in R["_panel"]["tasks"].values()})
    L = [f"Supports: a description of each family ladder on the panel's complete design (every template, draw and "
         f"item; strict and content; cut and uncut), rung by rung, and of where each of the {n_in} ladder models "
         f"falls among the {' or '.join(map(str, pn))} panel models on the same items and configurations (`ladder.md` "
         "sections 5 and 6). "
         "Each model's own numbers (`ladder.md` section 7) are computed the same way for every model, with the "
         "functions of `scripts/nebius_panel_report.py`: draw means, the corrected gain, and the template SD; the "
         "template variance has its item sampling removed by the panel's noise-anatomy estimator written for one model "
         "(`src/noise_anatomy.config_variance`: the configuration × item interaction mean square over the number of "
         "items)."]
    for name, f in fam.items():
        lad = [r for r in f["rungs"] if r["source"] == "ladder"]
        pan = [r for r in f["rungs"] if r["source"] == "panel"]
        miss = [short(r["key"]) for r in lad if not r["present"]]
        if f["rungs_present"] >= 2:
            L.append(f"{name}: {f['rungs_present']} of {f['rungs_total']} rungs ({len(pan)} from the panel, "
                     f"{len(lad) - len(miss)} from the ladder). The direction from rung to rung can be described; with "
                     "one checkpoint per size there is no interval across sizes and no test of a trend."
                     + (f" Waits for {', '.join(miss)}." if miss else ""))
        else:
            L.append(f"{name}: {f['rungs_present']} of {f['rungs_total']} rungs; no within-release direction yet"
                     + (f" (waits for {', '.join(miss)})." if miss else "."))
    L += ["Does not support: a claim that size causes a change. Each rung is one checkpoint, and a step between rungs "
          "also changes the architecture (dense or mixture of experts), the active parameters, training within the "
          "release, precision (bfloat16, FP8, fp4 or an API's unstated precision, against the panel's "
          f"{R['panel_precision_phrase']}), kernels and serving route (section 4; footnotes F2, F3)."]
    L.append(size_order_line(R))
    in_fam = [k for k in present_keys(models) if LADDER[k]["ladder"]]
    out_fam = [k for k in present_keys(models) if not LADDER[k]["ladder"]]
    sizes = sorted({len(f["rungs"]) for f in fam.values()})
    L += ["Does not support: a slope, a reliability estimate or a variance component across ladder models. The panel's "
          f"estimators of who benefits (analyses A, N and B) need many models on the same draws; the ladder adds "
          f"{len(in_fam)} single models to {len(fam)} families of {' or '.join(map(str, sizes))} rungs"
          + (f", and {len(out_fam)} outside them ({', '.join(nm(k) for k in out_fam)})" if out_fam else "")
          + ", which is too few to estimate them, so none is computed across the ladder."]
    if "deepseek-v4.1-flash-1017" in present_keys(models):
        L.append("DeepSeek V4.1-Flash-1017 is a single model outside the three families; it is placed "
                 "against the panel and never read as a trend.")
    L.append("Placement against the panel says where a model falls, and nothing about why.")
    return L


def size_order_line(R: dict) -> str:
    """Rungs are ordered by total parameters; per family, the recorded total and active parameters rung by rung, and
    whether the active parameters rise with the total."""
    def compact(s):
        if s.get("total") is None:
            return "size not recorded"
        if s.get("arch") == "dense":
            return f"{s['total']}B dense"
        return f"{s['total']}B total" + (f", {s['active']}B active" if s.get("active") else "")
    parts, unordered = [], []
    for name, f in R["families"].items():
        szs = [R["sizes"].get(r["key"], {}) for r in f["rungs"]]
        tot = [s.get("total") for s in szs]
        if None in tot or not all(Fraction(b) > Fraction(a) for a, b in zip(tot, tot[1:])):
            unordered.append(name)
        act = [s.get("active") for s in szs]
        if any(a is None for a in act):
            word = ("the active parameters are not recorded for "
                    + ", ".join(rung_label(r["key"]) for r, a in zip(f["rungs"], act) if a is None))
        elif all(Fraction(b) > Fraction(a) for a, b in zip(act, act[1:])):
            word = "the active parameters rise with the total"
        else:
            word = ("the active parameters (" + " → ".join(f"{a}B" for a in act) + ") do not rise with the total, "
                    "so a direction over size there is not one over active parameters")
        parts.append(f"{name}: " + " → ".join(compact(s) for s in szs) + f"; {word}")
    head = ("Rungs are ordered by total parameters" if not unordered else
            "Rungs are not ordered by total parameters in " + ", ".join(unordered))
    return head + " (sizes and their sources in section 4). " + ". ".join(parts) + "."


def share_pct(c: int, n: int) -> str:
    """A count's share of n in per cent, at two decimals, or at as many more (up to six) as it takes for a nonzero count
    not to print as zero."""
    dp = 2
    while c and dp < 6 and round(Fraction(100 * c, n), dp) == 0:
        dp += 1
    return fnum(Fraction(100 * c, n), dp)


def cut_share_summary(models: dict) -> tuple[list, float | None]:
    over, mx = [], None
    for k in present_keys(models):
        for t, tb in models[k]["tasks"].items():
            for arm, d in (tb.get("cut") or {}).get("cuts", {}).items():
                if arm in ("forced", "fs_forced"):
                    continue
                mx = d["share_cut"] if mx is None else max(mx, d["share_cut"])
                if d["share_cut"] >= CUT_SHARE_LISTED:
                    over.append((k, t, arm, d["share_cut"]))
    return over, mx


def asks_lines(R: dict, root: Path) -> list[str]:
    models, fam, comp, cvu = R["models"], R["families"], R["completeness"], R["cut_vs_uncut"]

    def st(k):
        return models[k]["status"] if k in models else "missing"

    def rungs(name):
        f = fam[name]
        return f"{name} {f['rungs_present']} of {f['rungs_total']} rungs"
    equiv, equiv_dk, equiv_dkr = equiv_checks(root)
    dk_keys = default_kernel_keys(models)
    llama = "llama-3.1-70b-instruct-fp8"
    llama_state = comp.get(llama, {}).get("state", st(llama))
    over, mx = cut_share_summary(models)
    at_raised = []
    for k in present_keys(models):
        for t, tb in models[k]["tasks"].items():
            for arm, d in (tb.get("uncut") or {}).get("cuts", {}).items():
                if arm in ("requested", "cot", "fs_requested") and d.get("finish_length"):
                    at_raised.append((k, t, arm, d["finish_length"], d["rows"]))
    ds = models.get("deepseek-v4.1-flash-1017")
    ds_line = ""
    if ds and ds["status"] != "missing":
        p = R["placement"].get("deepseek-v4.1-flash-1017", {})
        def ds_part(t, name=True):
            x = p[t]["cut"]["content"]["cot_minus_forced"]
            return (f"{TASK_NAME[t]} " if name else "") + f"{fnum(x['value'])} pp ({above_txt(x)})"
        ds_line = (" DeepSeek's CoT − forced (content), discovery draws: "
                   + "; ".join(ds_part(t) for t in DISC_TASKS if t in p) + "."
                   + (f" HotpotQA, pooled rows: {ds_part('hotpot', False)}." if "hotpot" in p else ""))
    n_in = len(present_keys(models))
    sp = R["spend"]
    spend = []
    if sp.get("mistral"):
        m = sp["mistral"]
        spend.append(f"Mistral Large 3 ${fnum(m['run'], 2)} for the run, ${fnum(m['with_test'], 2)} with the forced test "
                     f"(`{m['source']}`)")
    if sp.get("deepseek"):
        d = sp["deepseek"]
        meter = f", ${fnum(d['meter'], 4)} on the run's meter" if d.get("meter") is not None else ""
        spend.append(f"DeepSeek ${fnum(d['total'], 2)} in total (`{d['source']}`), which already holds the "
                     f"${fnum(d['before'], 2)} spent before the complete-design run: the live test, the pilot (whose "
                     f"replies are among the scored rows) and the test and determinism calls{meter} "
                     f"(`{d['before_source']}`)")
    L = [f"Goal: within-family size ladders on the complete design, strict and content. {rungs('Mistral 3')}, "
         f"{rungs('Qwen3.5')}, {rungs('Llama 3.1')} (Llama-3.1-70B: {llama_state}). The {n_in} models in ran the "
         "complete design and are scored strict and content in both readings (sections 4 and 7); the family tables "
         "are in section 5.",
         "Model lists: earlier lists are superseded by the current ladder. The run reports and the plan are in "
         "`results/ladder/plan.md`. Spend on file: "
         + ("; ".join(spend) if spend else "no run report here") + ".",
         "Forced-baseline test before each API run: Mistral Large 3 passed 9 of 9 forced replies "
         "(`results/ladder/mistral_forced_test.md`); DeepSeek passed 3 of 3 on `/v1/completions` "
         "(`results/frontier_panel/nebius_prefill_test.md`).",
         "Thinking off. DeepSeek ran with `thinking_mode` chat "
         "(`results/frontier_panel/nebius_panel.md`); Mistral Large 3 is an instruct model; the locally run models run "
         "with thinking off as in `config/models_ladder.yaml` (Qwen3.5-122B through `enable_thinking: false`).",
         f"Ladders in the paper: Qwen3.5 9B → 35B-A3B → 122B-A10B (Qwen3.5-122B-A10B "
         f"{st('qwen3.5-122b-a10b-fp8')}), Llama 3.1 8B → 70B (Llama-3.1-70B-Instruct {st(llama)}), Mistral 3 8B → "
         f"14B → Large 3 (Ministral 3 8B Instruct 2512 {st('ministral-3-8b-instruct-2512')}, Ministral 3 14B Instruct "
         f"2512 {st('ministral-3-14b-instruct-2512')}, Mistral Large 3 {st('mistral-large-2512')}); DeepSeek "
         f"V4.1-Flash-1017 as a single model outside the three families, no family trend "
         f"({st('deepseek-v4.1-flash-1017')}). It was listed as the top of the size range; the recorded "
         + ("sizes do not support that" if ds_size_compare(R)[1] else
            "sizes support that" if ds_size_compare(R)[0] else "sizes cannot settle that")
         + " (sections 1 and 4)." + ds_line + " Qwen3-2507 and gpt-oss were dropped from the ladder, so the Qwen3-235B "
           "speed rule no longer applies.",
         "Closed models, w only (one mid-tier model each from three providers): not run; waits for "
         "a key and the price check (`results/ladder/plan.md`).",
         "Raised caps (7,000) on the requested, CoT and five-shot requested arms. Uncut outputs that ended "
         "at 7,000: " + ("; ".join(f"{nm(k)} {TASK_NAME[t]} {arm} {n:,} of {rows:,}" for k, t, arm, n, rows in at_raised)
                         if at_raised else "none") + ".",
         "The cut copy is the primary reading of every table here (sections 1, 5, 6, 7).",
         "Share cut at the panel's cap, largest "
         + (f"{fnum(100 * mx, 2)}%" if mx is not None else "–") + "; arms with 1% or more cut: "
         + ("; ".join(f"{nm(k)} {TASK_NAME[t]} {arm} {fnum(100 * s, 2)}%" for k, t, arm, s in over) if over else "none")
         + f". Largest change of a quantity between the readings, every placed quantity on the discovery draws: "
           f"{largest_words(cvu['max_abs_difference'], top_of(cvu))}"
         + (f"; leaving out the readable shares, "
            f"{largest_words(cvu['max_abs_difference_scores'], top_of(cvu, False))}"
            if any(d["score"] == "readable" for d in top_of(cvu)) else "")
         + f"; {order_words(cvu)}. On the panel's basis of each benchmark and score (named per line in section 9): "
           f"{len(cvu['panel_rank_changes'])} placements against the panel move"
           f"{without_readable(cvu['panel_rank_changes'])} and "
           f"{len(cvu.get('family_direction_changes', []))} family directions change.",
         "Equivalence of the token-level cut: " + ("; ".join(equiv + equiv_dk) if equiv + equiv_dk
                                                                 else "no check on file")
         + ". Not run on the API models; on the Nebius endpoint replies to the same prompt differ (footnote F2). The "
           "default-kernel check named gpt-oss-120b, dropped from the ladder; it applies to "
         + (", ".join(nm(k) for k in dk_keys) or "no model in")
         + (", the ladder's default-kernel rung, checked on rows the run cut at the panel's cap (footnote F8)."
            if dk_keys and all(k in equiv_dkr for k in dk_keys) else
            ", the ladder's default-kernel rung, where it was not run: an untested caveat (footnote F8)."),
         "Restart: Qwen3-235B was dropped from the ladder; nothing to restart.",
         f"Mistral Large 3 ran at the raised caps with both scorings ({st('mistral-large-2512')}; "
         f"{comp.get('mistral-large-2512', {}).get('state', '–')}).",
         f"Families: every family has at least two rungs once Llama-3.1-70B is in: {rungs('Mistral 3')}, {rungs('Qwen3.5')}, "
         f"{rungs('Llama 3.1')}. Corrections: Llama-3.1-70B is the panel 8B's release; Qwen3.5-122B-A10B runs "
         "Qwen's official FP8 on 2 cards with thinking off; Ministral 3 8B and 14B run in bfloat16; the precision of "
         "each model is in section 4 and footnote F3."]
    return L


def forced_other_max(models: dict) -> dict:
    """Per model in: the largest content - strict gap over all rows of a forced arm (zero-shot or five-shot) on GSM8K,
    MATH-500 and HotpotQA, cut reading; footnote F1."""
    out = {}
    for k in present_keys(models):
        best = None
        for t in ("gsm8k", "math", "hotpot"):
            s = models[k]["tasks"].get(t, {}).get("cut")
            if not s:
                continue
            for arm in ("forced", "fs_forced"):
                v = s["strict_content_gap_pp"].get(arm)
                if v is not None and (best is None or abs(v) > abs(best["gap_pp"])):
                    best = {"gap_pp": v, "task": t, "arm": arm}
        if best:
            out[k] = best
    return out


ARM_WORDS = {"forced": "zero-shot forced", "fs_forced": "five-shot forced"}


def default_kernel_keys(models: dict) -> list[str]:
    """Ladder models that ran with default kernels (the -dk fallback)."""
    return [k for k in present_keys(models) if str(models[k].get("kernels", "")).startswith("default")]


def check_json(root: Path, rel: str) -> dict | None:
    """A forced-arm check's JSON (scripts/check_mistral_bbh_forced.py, scripts/check_nebius_forced.py), when it is here
    and holds the gap by form that footnote F1 quotes; None otherwise (F1 then leaves that model's clause out)."""
    p = root / rel
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))
    return g if isinstance(g.get("gap"), dict) and "by_form" in g["gap"] else None


PLACEHOLDER_KIND = {"empty box": ("empty box", "empty boxes"), "empty parentheses ()": ("empty `()`", "empty `()`"),
                    "empty reply": ("empty reply", "empty replies")}


def placeholder_words(ph: dict) -> str:
    """DeepSeek's five-shot forced replies with no answer in their box, by kind (scripts/check_nebius_forced.py), and
    how many hold the right answer after the first '}' (content_correct_without_cut)."""
    bk = ph.get("by_kind") or {}
    ans = bk.get("<answer>", 0)
    empty = {k: v for k, v in bk.items() if k != "<answer>"}
    s = (f"{ans:,} a bare `<answer>` placeholder and {sum(empty.values()):,} empty ("
         + ", ".join(f"{v:,} {PLACEHOLDER_KIND.get(k, (k, k))[0 if v == 1 else 1]}" for k, v in empty.items()) + ")")
    if "content_correct_without_cut" in ph:
        s += (f"; {ph['content_correct_without_cut']:,} of them hold the right answer after the first `}}`, which the "
              "panel's stop removes")
    return s


def inner_box_words(R: dict, have: set) -> str:
    """F1's sentence on inner boxes: a box the model opens and closes inside the prefilled one
    is read as the reply's last complete box, whatever stands before it, for every model; the restarts after a closed
    reasoning block, per ladder model, benchmark and forced arm, from the audit (STOP_AUDIT)."""
    au = (R.get("stop_counts") or {}).get("audit") or {}
    if not au:
        return ""
    cells = [(k, t, arm, x["open_inner_box_restart"]) for k in LADDER if k in have for t in STOP_TASKS
             for arm, x in au.get(k, {}).get(t, {}).items() if x.get("open_inner_box_restart")]
    s = (" A box that the model opens and closes inside the prefilled one is read as the reply's last complete box, "
         "whatever stands before it, for every model, as for the panel.")
    if cells:
        s += (" One form of it is a restart, a new box begun after a closed reasoning block (`\\boxed{()⏎</think>⏎⏎"
              "\\boxed{(C)}`), read the same way: " + "; ".join(
                  f"{nm(k)} {n:,} ({TASK_NAME[t]}, "
                  f"{ {'forced_direct': 'zero-shot forced', 'fewshot_forced': 'five-shot forced'}[arm]})"
                  for k, t, arm, n in cells)
              + f"; no other ladder model writes one (`{STOP_AUDIT}`)."
              " Each restart's inner box is in the strict format, so the strict and lenient readings read and credit "
              "the same restarts as the content reading.")
    return s


def stop_words(R: dict, have: set) -> str:
    """F1's clause on the stop's loss: per ladder model the forced replies the stop left unread (none read by the
    completion), the part cut at a connective or holding two options, and what the API models' full replies show."""
    sc = R.get("stop_counts") or {}
    au, fr = sc.get("audit") or {}, sc.get("full_replies") or {}
    if not au:
        return ""
    cells, unf, rng = [], [], []
    for k in [k for k in LADDER if k in have]:
        a = au.get(k, {})
        parts, uparts, rparts = [], [], []
        for t in STOP_TASKS:
            zs, fs_ = a.get(t, {}).get("forced_direct", {}), a.get(t, {}).get("fewshot_forced", {})
            parts.append(f"{TASK_NAME[t]} {zs.get('unread', 0):,} / {fs_.get('unread', 0):,}")
            uparts.append(f"{TASK_NAME[t]} {zs.get('open_unfinished', 0):,} / {fs_.get('open_unfinished', 0):,}")
            rparts.append(f"{TASK_NAME[t]} {zs.get('open_range_read', 0):,} / {fs_.get('open_range_read', 0):,}")
        cells.append(f"{nm(k)}: " + ", ".join(parts))
        if any(a.get(t, {}).get(arm, {}).get("open_unfinished") for t in STOP_TASKS
               for arm in ("forced_direct", "fewshot_forced")):
            unf.append(f"{nm(k)}: " + ", ".join(uparts))
        if any(a.get(t, {}).get(arm, {}).get("open_range_read") for t in STOP_TASKS
               for arm in ("forced_direct", "fewshot_forced")):
            rng.append(f"{nm(k)}: " + ", ".join(rparts))
    s = (" The stop's loss: replies the forced arms' stop at the first `}` left with the box open that the readers "
         f"do not read ({sc.get('version', 'forced-stop')}; `{STOP_AUDIT}`), zero-shot / five-shot forced: "
         + "; ".join(cells) + ". Of these, the text stops on a connective or holds two options (\"10 \\text{ blue and }\", "
         "\"({A, D}\"; the rest stop inside another group: a fraction, an exponent, a repeating decimal, a colour), "
         "zero-shot / five-shot forced: " + ("; ".join(unf) if unf else "none") + ". Time ranges the stop cut at "
         "\"to\" (\"8pm \\text{ to }\") are not in the loss: the 54 temporal_sequences items' options begin at "
         "different times, and the content reader reads the option the range begins (see the completion rule's open_range class), zero-shot / "
         "five-shot forced: " + ("; ".join(rng) if rng else "none") + ".")
    api = [(k, fr[k]) for k in ("mistral-large-2512", "deepseek-v4.1-flash-1017") if k in fr and k in have]
    if api:
        s += (f" Against the API models' full replies (`{FULL_REPLIES}`), the unread replies that are correct in the "
              "box the model closed: " + "; ".join(f"{nm(k)} {d['unread_false_loss']:,} of {d['unread_compared']:,}"
                                                   for k, d in api)
              + "; replies read as correct where the full box is not: "
              + "; ".join(f"{nm(k)} {d['false_gain']:,} of {d['compared'] - d['unread_compared']:,} read"
                          for k, d in api) + ".")
    return s


def footnotes(R: dict, root: Path) -> list[str]:
    """The footnotes to carry. F1's and F2's clauses on Mistral Large 3 and DeepSeek say what the source checks say,
    with their numbers (and appear only when the source file is here); every other number is computed here."""
    models = R["models"]
    have = set(present_keys(models))
    f1 = ("F1. BIG-Bench Hard forced arms: several ladder models write answer forms the strict rule does not accept, so "
          "strict and content are reported side by side. The strict rule is unchanged; a box that the forced arms' "
          "stop at the first `}` left open is read completed when the stop closed an answer wrapper and the text "
          "neither stops on a connective nor holds two options (`src/forced_stop.py`), as for the panel.")
    f1 += inner_box_words(R, have)
    fs, zs = "fewshot_forced", "forced_direct"
    g = check_json(root, MISTRAL_CHECK_JSON)
    if "mistral-large-2512" in have and g:
        gp, n = g["gap"], g["gap"]["n_arm"]
        bs_form = "letter with backslash (" + chr(92) + "C)"
        pp = lambda form, arm: fnum(Fraction(100 * gp["by_form"][form][arm], n[arm]))  # noqa: E731
        gap = lambda arm: fnum(Fraction(100 * gp["n"][arm], n[arm]))  # noqa: E731
        f1 += (f" Mistral Large 3 (`{MISTRAL_CHECK_JSON}`): in {fnum(g['forms_all'][fs]['bs_letter'])}% of five-shot "
               "forced replies it writes the option letter with a backslash, `\\boxed{(\\C)}`, and this form accounts "
               f"for {pp(bs_form, fs)} of the {gap(fs)} pp gap; in the zero-shot forced arm, "
               f"{pp('option text, no letter', zs)} of the {gap(zs)} pp gap is the option's text written without its "
               f"letter and {pp('LaTeX text wrapper', zs)} a LaTeX text wrapper such as `(\\text{{C}})`.")
    g = check_json(root, NEBIUS_CHECK_JSON)
    if "deepseek-v4.1-flash-1017" in have and g:
        gp, n = g["gap"], g["gap"]["n_arm"]
        pp = lambda form, arm: fnum(Fraction(100 * gp["by_form"][form][arm], n[arm]))  # noqa: E731
        gap = lambda arm: fnum(Fraction(100 * gp["n"][arm], n[arm]))  # noqa: E731
        ph = g["placeholders"]["bbh|fewshot_forced"]
        f1 += (f" DeepSeek (`{NEBIUS_CHECK_JSON}`) copies the prompt's `\\boxed{{<answer>}}` placeholder: zero-shot "
               f"forced writes the letter in angle brackets (`<B>`, {pp('letter in angle brackets (<B>)', zs)} pp of the "
               f"{gap(zs)} pp gap) or the option's text ({pp('option text, no letter', zs)} pp); five-shot forced writes "
               f"`<answer>(B)` ({pp('answer tag around a letter (<answer>(B))', fs)} pp of the {gap(fs)} pp gap), the "
               f"option's text ({pp('option text, no letter', fs)} pp) or a LaTeX text wrapper "
               f"({pp('LaTeX text wrapper', fs)} pp); {ph['rows']:,} of {ph['n_arm']:,} five-shot forced replies "
               f"({fnum(Fraction(100 * ph['rows'], ph['n_arm']))}%) hold no answer in their box: "
               + placeholder_words(ph) + ".")
    f1 += stop_words(R, have)
    fo = R.get("forced_other_max", {})
    if fo:
        f1 += (" GSM8K, MATH-500 and HotpotQA, computed here: the largest content − strict gap over all rows of a forced "
               "arm (zero-shot or five-shot) is " + "; ".join(
                   f"{fnum(v['gap_pp'], 2)} pp for {nm(k)} ({TASK_NAME[v['task']]}, {ARM_WORDS[v['arm']]})"
                   for k, v in fo.items()) + ".")
    fj = root / FORMS_JSON
    if fj.exists():
        d = json.loads(fj.read_text(encoding="utf-8"))
        parts = []
        for k, v in d.get("models", {}).items():
            if k not in have:
                continue
            arms = []
            for arm, a in v.get("arms", {}).items():
                # exact from the file's counts: pp of all rows of the arm
                top = [f for f in a["forms"] if Fraction(100 * f["rows"], a["rows"]) >= FORM_MIN][:3]
                arms.append(f"{'five-shot forced' if arm == 'fewshot_forced' else 'zero-shot forced'} "
                            f"{fnum(Fraction(100 * a['gap_rows'], a['rows']))} pp ["
                            + "; ".join(f"{f['form']} {fnum(Fraction(100 * f['rows'], a['rows']))}" for f in top) + "]")
            parts.append(f"{nm(k)}: " + "; ".join(arms))
        if parts:
            f1 += (" The cluster rungs, same classes (`" + FORMS_JSON.replace("results/ladder/analysis/", "")
                   + "`, from `scripts/ladder_bbh_forms.py`; pp of all rows of the arm; the three largest forms of "
                     "0.05 pp or more): " + "; ".join(parts) + ".")
    F = [f1]
    if "deepseek-v4.1-flash-1017" in have and (root / NEBIUS_CHECK).exists():
        F.append(f"F2. The Nebius endpoint does not repeat its greedy decoding (`{NEBIUS_CHECK}`, Q4). On GSM8K chain of "
                 "thought at the same cap, two replies to the same prompt differed in 19 of 20 pilot-against-check "
                 "pairs, and in 5 of 5 pairs in each of three separate tests: at temperature 0, with `top_k` 1, and "
                 "with seed 1234. Scored strict, those 35 same-cap pairs gave the same answer in all 35. On MATH-500, "
                 "all 42 pilot replies rerun at the raised cap (36 requested, 6 chain of thought) differ, in the sense "
                 "that the rerun does not begin with the pilot's cut reply. No forced reply was sent twice, so their "
                 "repeat rate is unknown. Each DeepSeek reply is one draw from the served fp4 model.")
    else:
        F.append("F2. DeepSeek's nondeterminism check is not here (no DeepSeek rows or no source file).")
    hs = root / "manifests/sampler_hotpot.json"
    n_hc = None
    if hs.exists():
        sh = json.loads(hs.read_text(encoding="utf-8"))
        n_hc = len({str(c) for c in sh["discovery_config_ids"] + sh["validation_config_ids"]})
    F += ["F3. Precision differs across the ladder: FP8 for Llama-3.1-70B (RedHatAI FP8-dynamic) and Qwen3.5-122B-A10B "
          "(Qwen's official block FP8), both with a 16-bit KV cache; fp4 (NVFP4, provider-stated) for DeepSeek; "
          "bfloat16 for Ministral 3 8B and 14B; Mistral Large 3 as served by Mistral's API, precision not stated. The "
          f"panel ran in {R['panel_precision_phrase']} (" + ", ".join(f"`{f}`" for f in PANEL_CONFIGS) + ").",
          "F4. Kernels: " + "; ".join(f"{nm(k)} {m.get('kernels', '–')}" for k, m in models.items()
                                      if m["status"] != "missing")
          + ". The panel's Qwen3.5 models ran with default kernels as well.",
          "F5. Mistral Large 3 requests carry the template's default system message, as the panel's Mistral models and "
          "the Ministral 3 rungs run theirs (`scripts/mistral_ladder.py`).",
          "F6. Corrected gain. The rule's corrected gain is (1 − w) × (CoT − forced) with w and the gain both on the "
          "discovery draws (`src/baseline_mixture.RULE_TEXT`), and every corrected gain printed here is that, except "
          "where a heading or column says pooled rows. That is HotpotQA content in the placements (sections 1 and 6) "
          "and the family tables (section 5): the panel's HotpotQA content values are pooled over every row of the "
          f"{n_hc if n_hc else '–'} configurations (`results/hotpot_content/tables/per_model_arm.csv`), so the ladder "
          f"models are placed on that basis too, and the value printed there is {POOLED_CG}, for panel and ladder "
          "alike. The 48-draw field `all_draws.corrected_gain_pp` of `ladder.json` is printed nowhere (its definition: "
          "`ladder.json`, `rules`)."]
    q = models.get("qwen3.5-122b-a10b-fp8")
    if q and q["status"] != "missing" and q.get("pilot") and q["pilot"].get("pass") is False:
        pl = q["pilot"]
        req = pl.get("requested_arms") or {}
        k_, n_ = sum(v[0] for v in req.values()), sum(v[1] for v in req.values())

        def arm_part(arm, words):
            xs = [(t, req[f"{t}.{arm}"]) for t in TASKS if f"{t}.{arm}" in req]
            return f"{words} " + ", ".join(f"{a} of {b} on {TASK_NAME[t]}" for t, (a, b) in xs) if xs else None
        per = "; ".join(x for x in (arm_part("direct_answer", "zero-shot requested"),
                                    arm_part("fewshot_requested", "five-shot requested")) if x)
        F.append("F7. Qwen3.5-122B-A10B failed its pilot check (`" + pl["source"].replace(".json", ".md") + "`: \""
                 + (pl.get("verdict") or "–") + "\"). The rule asks for a boxed answer in at least "
                 f"{fnum(100 * PILOT_RULE_MIN, 0)}% of the requested arms' rows (zero- and five-shot, pooled over the "
                 f"benchmarks; `scripts/check_ladder_pilot.py`); the pilot gave one in {k_} of {n_} "
                 f"({fnum(Fraction(k_, n_) if n_ else None, 3)}): {per}. The full run went ahead under the "
                 f"`{q.get('run_key', '–')}` key (default kernels, the panel's Qwen3.5 mode) with no recorded release "
                 "decision, and its outputs are scored as they are.")
    dk = default_kernel_keys(models)
    if dk:
        eq = []
        _, eq_dk, eq_dkr = equiv_checks(root)
        for p in sorted(root.glob(EQUIV_GLOB)):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("kernels") == "default":
                continue
            eq.append(f"{nm(d['model_key']) if d['model_key'] in LADDER else d['model_key']} "
                      f"({models.get(d['model_key'], {}).get('kernels', '–')} kernels): {d['identical']} of {d['rows']} "
                      f"rerun rows identical to the pilot's cut copy, {d.get('identical_among_cut', '–')} of "
                      f"{d.get('cut_in_pilot', '–')} cut ones (`{rel(p, root)}`)")
        unchecked = [k for k in dk if k not in eq_dkr]
        parts = []
        for k in dk:
            sh = [(t, arm, c) for t, tb in models[k]["tasks"].items()
                  for arm, c in (tb.get("cut") or {}).get("cuts", {}).items()
                  if arm in ("requested", "cot", "fs_requested") and c["cut_at_panel_cap"]]
            parts.append(f"{nm(k)}: " + ("; ".join(f"{TASK_NAME[t]} {arm} {share_pct(c['cut_at_panel_cap'], c['rows'])}% "
                                                    f"({c['cut_at_panel_cap']:,} of {c['rows']:,})" for t, arm, c in sh)
                                         if sh else "none") + "; no other raised arm has an output cut")
        if unchecked:
            F.append("F8. The token-level cut against a rerun at the panel's caps (" + DEFAULT_KERNEL_NOTE + ") was "
                     "checked only under batch-invariant kernels: " + ("; ".join(eq) if eq else "no check on file") + ". "
                     + ", ".join(nm(k) for k in dk) + " ran with default kernels, under which a rerun need not repeat "
                     "its tokens, and was not checked, so the agreement of its cut copy with a run at the panel's caps "
                     "is untested. The outputs the cut shortens, share of the arm's rows: " + "; ".join(parts) + ".")
        else:
            F.append("F8. The token-level cut against a rerun at the panel's caps (" + DEFAULT_KERNEL_NOTE + "). Under "
                     "batch-invariant kernels: " + ("; ".join(eq) if eq else "no check on file") + ". "
                     + ", ".join(nm(k) for k in dk) + " ran with default kernels, under which a rerun need not repeat "
                     "its tokens: " + "; ".join(eq_dk) + ". " + equiv_caveat(eq_dkr)
                     + " The outputs the cut shortens, share of the arm's rows: " + "; ".join(parts) + ".")
    return F


def completeness_table(R: dict) -> list[str]:
    comp = R["completeness"]
    L = ["| Model | Benchmark | Design rows (arms) | Scored, cut | Scored, uncut | Missing ids | Duplicate ids | "
         "Extra ids | Id fields differ | Unscored | Configurations matching (arm × configuration cells) | Design "
         "file equals the API runs' |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in comp.items():
        for t in TASKS:
            r = m["tasks"].get(t, {})
            if r.get("design_absent"):
                L.append(f"| {short(k)} | {TASK_NAME[t]} | design file absent (`{r['design_file']}`) | | | | | | | | | |")
                continue
            c, u = r.get("cut", {}), r.get("uncut", {})

            def both(key):
                a, b = c.get(key, 0), u.get(key, 0)
                return f"{a:,}" if a == b else f"{a:,} / {b:,}"
            cells_d = sum(r["design_configs_by_arm"].values())
            cells_m = sum(v["configs_matching"] for v in c.get("by_arm", {}).values()) if c.get("present") else 0
            eq = r.get("design_equals_api_design")
            L.append(f"| {short(k)} | {TASK_NAME[t]} | {r['design_rows']:,} ({len(r['design_by_arm'])}) | "
                     f"{c.get('scored_rows', 0):,} | {u.get('scored_rows', 0):,} | {both('missing_ids')} | "
                     f"{both('duplicate_ids')} | {both('extra_ids')} | {both('id_fields_differ')} | {both('unscored')} | "
                     f"{cells_m} of {cells_d} | {'it is that file' if eq is None else ('yes' if eq else 'no')} |")
    L += ["", "Two numbers in a cell are cut / uncut. Configurations matching: (arm, configuration) cells whose scored "
          "row count equals the design's (cut reading). The API runs read the ladder's request files of "
          f"`{API_DESIGN_KEY}` (`scripts/mistral_ladder.py`, `scripts/nebius_panel.py`); each locally run model's own request "
          "files are checked equal to those (request ids, arms, configurations, items).", ""]
    L.append("State per model:")
    L += [f"- {short(k)}: {m['state']}." for k, m in comp.items()]
    shortfalls = []
    for k, m in comp.items():
        for t, r in m["tasks"].items():
            for rd in READINGS:
                x = r.get(rd, {})
                if not x.get("present"):
                    continue
                for cell in x.get("cells_short", []):
                    shortfalls.append(f"{short(k)} {TASK_NAME[t]} {rd}: {cell['arm']} {cell['config_id']} scored "
                                      f"{cell['scored']} of {cell['design']}")
                if x.get("missing_arms"):
                    shortfalls.append(f"{short(k)} {TASK_NAME[t]} {rd}: arms with no scored row "
                                      + ", ".join(x["missing_arms"]))
    L += ["", "Shortfalls among the models with scored rows (arm × configuration cells, missing arms): "
          + (str(len(shortfalls)) + "." if shortfalls else "none.")]
    L += [f"- {s}" for s in shortfalls[:200]]
    return L


def family_size_tables(R: dict) -> list[str]:
    """Section 5: per family and benchmark, the rungs' values on one basis per table (discovery draws; HotpotQA content
    on pooled rows in a table of its own), with the direction over size (DIRECTION_RULE)."""
    L = []

    def table(f, rungs, heads, t, scores, label):
        out = [label, "", "| Quantity | " + " | ".join(heads) + " | Over size" + (" (strict / content)" if
                                                                                    len(scores) == 2 else "") + " |",
               "|---|" + "---|" * len(heads) + "---|"]
        for q in QUANTITIES:
            if q == "w" and "strict" not in scores:
                continue
            cells = []
            for r in rungs:
                v = r["values"].get(t)
                if not r["present"] or not v:
                    cells.append("waits")
                elif q == "w":
                    cells.append(fnum(v["strict"].get("w"), 3))
                else:
                    cells.append(" / ".join(fnum(v[sc_][q]) for sc_ in scores))
            if q == "w":
                dr = dword(f["direction"][t]["strict"]["w"]["direction"])
            else:
                dr = " / ".join(dword(f["direction"][t][sc_][q]["direction"]) for sc_ in scores)
            lab = POOLED_CG if (q == "corrected_gain" and scores == ("content",) and t == "hotpot") else Q_LABEL[q]
            out.append(f"| {lab} | " + " | ".join(cells) + f" | {dr} |")
        return out + [""]

    for name, f in R["families"].items():
        rungs = f["rungs"]
        heads = [f"{rung_label(r['key'])} ({r['source']})" for r in rungs]
        L += [f"### {name}: {' → '.join(rung_label(r['key']) for r in rungs)} ({f['rungs_present']} of "
              f"{f['rungs_total']} rungs in)", ""]
        for r in rungs:
            if not r["present"]:
                L.append(f"- {rung_label(r['key'])}: waits for {short(r['key'])} "
                         f"({R['completeness'].get(r['key'], {}).get('state', 'missing')}).")
        if f["rungs_present"] < f["rungs_total"]:
            L.append("")
        for t in DISC_TASKS:
            L += table(f, rungs, heads, t, ("strict", "content"), f"{TASK_NAME[t]} (strict / content; discovery draws):")
        L += table(f, rungs, heads, "hotpot", ("strict",), "HotpotQA, strict (discovery draws):")
        L += table(f, rungs, heads, "hotpot", ("content",), "HotpotQA, content (pooled rows):")
        if f["context"]:
            L += [f"Context, not a rung: {f['context_note']}. Content scores (strict in brackets); the basis of each row "
                  "is named.", "",
                  "| Panel model | Benchmark | Basis | Forced | CoT | CoT − forced | CoT − requested | w |",
                  "|---|---|---|---|---|---|---|---|"]
            for row in f["context"]:
                for t in TASKS:
                    v = row["values"].get(t)
                    if not v:
                        continue
                    s, c = v["strict"], v["content"]
                    basis = ("discovery draws" if t in DISC_TASKS else
                             "content on pooled rows; strict and w on the discovery draws")
                    L.append(f"| {short(row['key'])} | {TASK_NAME[t]} | {basis} | {fnum(c['forced'])} "
                             f"({fnum(s['forced'])}) | {fnum(c['cot'])} ({fnum(s['cot'])}) | "
                             f"{fnum(c['cot_minus_forced'])} ({fnum(s['cot_minus_forced'])}) | "
                             f"{fnum(c['cot_minus_requested'])} ({fnum(s['cot_minus_requested'])}) | "
                             f"{fnum(s.get('w'), 3)} |")
            L.append("")
    return L


def panel_family_marks(R: dict, t: str) -> str:
    """The panel's own members of the ladder families on one benchmark, CoT − forced (content), with their rank, on the
    panel file's basis (named)."""
    pdf = R["_panel"]["exact"][t]
    col = "cot_minus_forced_content"
    p = pdf[col].dropna()
    bits = []
    for name, f in FAMILY_LADDERS.items():
        keys = [k for k, s in f["rungs"] if s == "panel"] + list(f["context"])
        for k in keys:
            if k in p.index:
                v = p.at[k]
                eq = sum(abs(float(x) - float(v)) <= TOL for j, x in p.items() if j != k)      # ties with the others
                bits.append(f"{short(k)} {fnum(v)} (above {sum(float(x) < float(v) - TOL for x in p)}/{len(p)}"
                            + (f", {eq} tied" if eq else "") + ")")
    return (f"Panel family members, CoT − forced (content), {BASIS_NAME[basis_of(t, 'content')]}: " + "; ".join(bits)
            + ".") if bits else ""


def placement_table(R: dict, t: str, keys: list[str], scores: tuple, with_readable: bool) -> list[str]:
    """Section 6: one benchmark, the given scores (one basis per table), every quantity with its placement cell."""
    place_ = R["placement"]
    L = ["| Score | Quantity | Panel min | Panel median | Panel max | " + " | ".join(short(k) for k in keys) + " |",
         "|---|---|---|---|---|" + "---|" * len(keys)]
    for sc_ in scores:
        for q in QUANTITIES + ("residual",):
            if q == "w" and sc_ == "content":
                continue
            p0 = place_[keys[0]][t]["cut"][sc_][q]
            d = 3 if q == "w" else 1
            lab = Q_LABEL[q]
            if basis_of(t, sc_) == "pooled" and q == "corrected_gain":
                lab = POOLED_CG
            elif basis_of(t, sc_) == "pooled" and q == "residual":
                lab = f"CoT − requested minus {POOLED_CG}"
            L.append(f"| {sc_} | {lab} | {fnum(p0['panel_min'], d)} | {fnum(p0['panel_median'], d)} | "
                     f"{fnum(p0['panel_max'], d)} | " + " | ".join(cell(place_[k][t]["cut"][sc_][q], d) for k in keys)
                     + " |")
    if with_readable:
        for a in ("requested", "forced", "cot"):
            fr = [place_[k][t]["cut"].get("readable", {}).get(a) for k in keys]
            if all(fr):
                L.append(f"| – | Readable, {a} (%) | {fnum(fr[0]['panel_min'])} | {fnum(fr[0]['panel_median'])} | "
                         f"{fnum(fr[0]['panel_max'])} | " + " | ".join(cell(p) for p in fr) + " |")
    return L


def write_md(R: dict, path: Path) -> None:
    models, place_ = R["models"], R["placement"]
    present = present_keys(models)
    L = [f"# Size ladder: analysis ({R['created']}; {len(present)} of {len(models)} models in)", "",
         "Every number is in percentage points (pp) unless it is w, a share in [0, 1]. Primary reading: every output "
         "cut at the token level to the panel's cap of its arm (1,024 requested; 3,072 chain of thought, 1,900 on "
         "HotpotQA). Supplementary: the uncut outputs at the raised cap (7,000). Code: `scripts/ladder_analysis.py`; "
         "rerun it when a locally run model's scored rows arrive. Independent check: `scripts/verify_ladder_analysis.py`.",
         "", "Rounding: every number is rounded half-even at its printed precision on its exact value. A ladder model's "
         "accuracies, gains, shares, w, corrected gains and residuals are exact ratios of the counts in its scored rows; "
         "a panel value is the exact ratio its file's float stands for on the design's grid (discovery draws × items "
         "per configuration, or the arm's rows for pooled values). So a value exactly halfway prints as its even "
         "neighbour (8.95 as 9.0, 8.85 as 8.8). Placement counts, marks and direction words are judged on the "
         "unrounded values. A template SD is rounded on the exact square root of its exact variance.",
         "", "## 1. Findings", ""]
    L += [f"- {x}" for x in R["findings"]]
    L += glance_table(R)
    L += ["", "## 2. What the ladder supports and what it does not", ""]
    L += [f"- {x}" for x in R["scope"]]
    L += ["", "## 3. Run notes", ""]
    L += [f"- {x}" for x in R["asks"]]
    L += ["", "## 4. What is in, and completeness against the design", "",
          "| Model | Ladder | Size (recorded) | Route | Precision | Kernels | Status | Benchmarks in |",
          "|---|---|---|---|---|---|---|---|"]
    for k, m in models.items():
        lad = m["ladder"] or OUTSIDE
        tin = ", ".join(TASK_NAME[t] for t in m["tasks"]) or "none"
        st = m["status"] if m["status"] != "missing" else "missing: waits for its scored rows"
        L.append(f"| {m['name']} | {lad} | {size_text(R['sizes'][k])} | {m['route']} | {m['precision']} | "
                 f"{m.get('kernels', '–')} | {st} | {tin} |")
    L += ["", "Panel rungs of the family ladders, sizes as recorded: " + "; ".join(
        f"{short(r['key'])} {size_text(R['sizes'].get(r['key'], {}))}" for f in R["families"].values()
        for r in f["rungs"] if r["source"] == "panel") + ". Rungs are ordered by total parameters (section 2)."]
    L += ["", "Completeness: each model's scored rows against the design, per benchmark (per arm and configuration in "
          "`ladder.json`, key `completeness`).", ""]
    L += completeness_table(R)

    L += ["", "## 5. Family ladders", "",
          "Smallest rung first. Panel rungs come from the panel's files; ladder rungs are computed here on the same "
          "basis, cut reading. Each table names its basis: GSM8K, MATH-500 and BIG-Bench Hard on the discovery draws "
          "(cells strict / content, w one value); HotpotQA in two tables, strict on the discovery draws and content on "
          f"pooled rows, where the corrected-gain row is {POOLED_CG} (footnote F6). Over size: {DIRECTION_RULE}", ""]
    L += family_size_tables(R)

    L += ["## 6. Where each ladder model sits against the panel", "",
          "Panel values from the panel's own result files (sources in section 13), on their basis, named in each "
          "table's heading: means over the 24 discovery draws, except HotpotQA content, pooled over every row of the "
          "36 configurations (footnote F6). Each ladder model is computed on the same basis as the panel in each "
          "table. A cell reads value (b/M): the ladder model's value is above b of the M panel models with a value, "
          "and value (b/M, k tied) when k of them equal it (within 1e-9); ↑ or ↓ marks a value outside the panel's range, in every row. Residual: CoT − requested minus the "
          "corrected gain.", ""]
    for t in TASKS:
        keys = [k for k in present if t in place_.get(k, {})]
        if not keys:
            continue
        if t in DISC_TASKS:
            L += [f"### {TASK_NAME[t]} (discovery draws)", ""]
            L += placement_table(R, t, keys, ("strict", "content"), True)
        else:
            L += [f"### {TASK_NAME[t]}, strict (discovery draws)", ""]
            L += placement_table(R, t, keys, ("strict",), False)
            L += ["", f"### {TASK_NAME[t]}, content (pooled rows)", ""]
            L += placement_table(R, t, keys, ("content",), False)
        mk = panel_family_marks(R, t)
        L += ["", mk, ""] if mk else [""]

    L += ["## 7. Per model and benchmark (cut, primary)", "",
          "Each column names its basis. Accuracy and gains (48 draws): mean over the sampler's 48 draws of the "
          "per-configuration accuracy, on the units with all three zero-shot arms. w, the corrected gain (1 − w) × "
          "(CoT − forced), the rule's (src/baseline_mixture.RULE_TEXT), and the CoT − forced it shrinks and CoT − "
          "requested in the columns so named: the 24 discovery draws; w counts requested outputs longer than the "
          "forced cap (16 tokens; 64 on HotpotQA). The same computation as `results/frontier_panel/nebius_panel.md`, "
          "Main numbers, with the discovery-draw CoT − forced added beside w.", ""]
    for reading in READINGS:
        if reading == "uncut":
            L += ["", "## 8. Per model and benchmark (uncut, supplementary)", ""]
        L += ["| Model | Benchmark | Score | Requested (48 draws) | Forced (48 draws) | CoT (48 draws) | CoT − requested "
              "(48 draws) | CoT − forced (48 draws) | CoT − forced (discovery draws) | w (discovery draws) | Corrected "
              "gain (discovery draws) | CoT − requested (discovery draws) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in present:
            for t, tb in models[k]["tasks"].items():
                s = tb.get(reading)
                if not s:
                    continue
                for sc_ in SCORES:
                    a, d = s[sc_]["all_draws"], s[sc_]["discovery"]
                    L.append(f"| {short(k)} | {TASK_NAME[t]} | {sc_} | {fnum(a['accuracy_pp']['requested'])} | "
                             f"{fnum(a['accuracy_pp']['forced'])} | {fnum(a['accuracy_pp']['cot'])} | "
                             f"{fnum(a['gain_over_requested_pp'])} | {fnum(a['gain_over_forced_pp'])} | "
                             f"{fnum(d['gain_over_forced_pp'])} | {fnum(s['w']['discovery'], 3)} | "
                             f"{fnum(d['corrected_gain_pp'])} | {fnum(d['gain_over_requested_pp'])} |")
        if reading == "uncut":
            break
        L += ["", "### Template SD (cut)", "",
              "Standard deviation over the distinct configurations (pp, denominator C − 1); in brackets the variance "
              "left after the item sampling of a configuration mean is removed (pp², as it comes). Every configuration "
              "has the same items, so item difficulty is common to all configuration means and drops out of their "
              "spread; the term removed is the configuration × item interaction mean square over the number of items "
              "(`src/noise_anatomy.config_variance`, the panel's noise-anatomy estimator for one model). A "
              "negative value means the spread of the configuration means is no larger than their item sampling.", "",
              "| Model | Benchmark | Score | C | Requested | Forced | CoT | CoT − requested | CoT − forced |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in present:
            for t, tb in models[k]["tasks"].items():
                s = tb.get("cut")
                if not s:
                    continue
                for sc_ in SCORES:
                    tm = s[sc_]["template"]
                    cells = [f"{fnum(tm[q]['sd_pp'])} [{fnum(tm[q]['variance_minus_item_sampling_pp2'])}]"
                             for q in ("requested", "forced", "cot", "gain_over_requested", "gain_over_forced")]
                    L.append(f"| {short(k)} | {TASK_NAME[t]} | {sc_} | {s['configs_answered']} | "
                             + " | ".join(cells) + " |")
        L += ["", "### Five-shot arms", "",
              "Mean over configurations of the per-configuration accuracy; five-shot forced also as the mean over the "
              "sampler's 48 draws (`fewshot_block` of `scripts/nebius_panel_report.py`).", "",
              "| Model | Benchmark | Arm | Reading | Rows | C | Strict | Content | Strict (draws) | Content (draws) | "
              "Share longer than forced cap (%) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in present:
            for t, tb in models[k]["tasks"].items():
                for rd in READINGS:
                    for arm, d in (tb.get(rd) or {}).get("fewshot", {}).items():
                        sh = d.get("share_longer_than_forced_cap")
                        L.append(f"| {short(k)} | {TASK_NAME[t]} | {arm} | {rd} | {d['rows']:,} | {d['configs']} | "
                                 f"{fnum(d['strict_pp_config_mean'])} | {fnum(d['content_pp_config_mean'])} | "
                                 f"{fnum(d.get('strict_pp_all_draws'))} | {fnum(d.get('content_pp_all_draws'))} | "
                                 f"{fnum(None if sh is None else 100 * sh)} |")
        L += ["", "### Readable answers and the strict–content gap (cut, all rows)", "",
              "Readable: the strict scorer found a recognised answer form (its `compliant` flag). Content − strict: "
              "the content reading's accuracy minus the strict reading's, per arm.", "",
              "| Model | Benchmark | Readable (%): requested | forced | CoT | five-shot forced | five-shot requested | "
              "Content − strict: requested | forced | CoT | five-shot forced | five-shot requested |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in present:
            for t, tb in models[k]["tasks"].items():
                s = tb.get("cut")
                if not s:
                    continue
                rd_, g = s["readable_pct"], s["strict_content_gap_pp"]
                L.append(f"| {short(k)} | {TASK_NAME[t]} | "
                         + " | ".join(fnum(rd_.get(a, {}).get("all_rows"))
                                      for a in ("requested", "forced", "cot", "fs_forced", "fs_requested"))
                         + " | " + " | ".join(fnum(g[a]) for a in ("requested", "forced", "cot", "fs_forced",
                                                                    "fs_requested")) + " |")

    cvu = R["cut_vs_uncut"]
    L += ["", "## 9. Cut against uncut", "",
          "Share of outputs cut at the panel's cap per model, benchmark and arm (raised arms; the forced arms are never "
          "cut), and outputs of the uncut reading that ended at the raised cap.", "",
          "| Model | Benchmark | Arm | Rows | Cut at panel cap | Share (%) | Ended at raised cap (uncut) |",
          "|---|---|---|---|---|---|---|"]
    for k in present:
        for t, tb in models[k]["tasks"].items():
            cu_, un_ = (tb.get("cut") or {}).get("cuts", {}), (tb.get("uncut") or {}).get("cuts", {})
            for arm in ("requested", "cot", "fs_requested"):
                if arm not in cu_:
                    continue
                d = cu_[arm]
                fl = un_.get(arm, {}).get("finish_length")
                L.append(f"| {short(k)} | {TASK_NAME[t]} | {arm} | {d['rows']:,} | {d['cut_at_panel_cap']:,} | "
                         f"{fnum(100 * d['share_cut'], 2)} | {'–' if fl is None else f'{fl:,}'} |")
    ch = [d for d in cvu["differences"] if d["uncut_minus_cut"] is not None
          and abs(d["uncut_minus_cut"]) >= CHANGE_LISTED]
    n_any = sum(1 for d in cvu["differences"] if d["uncut_minus_cut"] is not None and abs(d["uncut_minus_cut"]) > TOL)
    def n_of(pred):
        ds = [d for d in cvu["differences"] if pred(d)]
        return (sum(1 for d in ds if d["uncut_minus_cut"] is not None and abs(d["uncut_minus_cut"]) >= CHANGE_LISTED),
                sum(1 for d in ds if d["uncut_minus_cut"] is not None and abs(d["uncut_minus_cut"]) > TOL), len(ds))
    parts = []
    for what, pred in (("residuals", lambda d: d["quantity"] == "residual"),
                       ("readable shares", lambda d: d["score"] == "readable")):
        a_, b_, n_ = n_of(pred)
        parts.append(f"{what} {a_} and {b_} of {n_}")
    L += ["", f"Quantities that change by 0.05 pp or more between the readings, every placed quantity on the discovery "
              f"draws ({len(ch)} of {len(cvu['differences'])}; {n_any} change at all; of these, by 0.05 pp or more and "
              f"at all: {'; '.join(parts)}; every change is in `ladder.json`). The placed quantities are section 6's "
              "rows; Score readable is the share of replies the strict rule can read, per zero-shot arm (%), placed on "
              "GSM8K, MATH-500 and BIG-Bench Hard, where the panel has it:", ""]
    if ch:
        L += ["| Model | Benchmark | Score | Quantity | Cut | Uncut | Uncut − cut |", "|---|---|---|---|---|---|---|"]
        for d in sorted(ch, key=lambda x: -abs(x["uncut_minus_cut"])):
            L.append(f"| {short(d['model'])} | {TASK_NAME[d['task']]} | {d['score']} | {Q_LABEL[d['quantity']]} | "
                     f"{fnum(d['cut'], 2)} | {fnum(d['uncut'], 2)} | {fnum(d['uncut_minus_cut'], 2)} |")
    else:
        L.append("None.")
    def pair(fl):
        return (f"- {TASK_NAME[fl['task']]}, {fl['score']}, {Q_LABEL[fl['quantity']]}: {short(fl['models'][0])} and "
                f"{short(fl['models'][1])}, cut {fnum(fl['cut'][0])} and {fnum(fl['cut'][1])}, uncut "
                f"{fnum(fl['uncut'][0])} and {fnum(fl['uncut'][1])}.")
    L += ["", "Pairs of ladder models whose order reverses between the readings (each reading orders the pair, the "
              f"two in opposite ways), discovery draws: {len(cvu['ordering_flips'])}"
              f"{without_readable(cvu['ordering_flips'])}."]
    L += [pair(fl) for fl in cvu["ordering_flips"]]
    ts = cvu.get("ties_separating", [])
    L += ["", "Pairs of ladder models tied exactly in one reading that separate in the other (a tie, not a reversal), "
              f"discovery draws: {len(ts)}."]
    L += [pair(fl)[:-1] + f"; tied in the {fl['tied_in']} reading." for fl in ts]
    L += ["", f"Placements against the panel that change when the uncut value is used, each on the panel's basis "
              f"(named per line): {len(cvu['panel_rank_changes'])}{without_readable(cvu['panel_rank_changes'])}."]
    for rc in cvu["panel_rank_changes"]:
        def ab(r):
            e = rc[f"{r}_equal"]
            return f"above {rc[f'{r}_below']} of {rc['panel_n']}" + (f", {e} tied" if e else "") + f" ({r})"
        L.append(f"- {short(rc['model'])}, {TASK_NAME[rc['task']]}, {rc['score']}, {Q_LABEL[rc['quantity']]}, "
                 f"{BASIS_NAME[rc['basis']]}: {ab('cut')}; {ab('uncut')}.")
    fdc = cvu.get("family_direction_changes", [])
    L += ["", f"Family directions (section 5) that change when the ladder rungs' uncut values are used, each on the "
              f"basis of its section 5 table (named per line): {len(fdc)}."]
    for x in fdc:
        L.append(f"- {x['family']}, {TASK_NAME[x['task']]}, {x['score']}, {Q_LABEL[x['quantity']]}, "
                 f"{BASIS_NAME[x['basis']]}: cut {arrow(x['cut'])} ({dword(x['cut_direction'])}), uncut "
                 f"{arrow(x['uncut'])} ({dword(x['uncut_direction'])}).")

    L += ["", "## 10. Footnotes to carry", ""]
    L += [f"- {x}" for x in R["footnotes"]]

    L += ["", "## 11. DeepSeek reproduction and the independent check", ""]
    rp = R.get("reproduction")
    if rp:
        L.append(f"{rp['values_compared']} values (accuracy, gains, w, corrected gain, template SD with the item-sampling "
                 f"correction, five-shot arms; both readings) compared with `{rp['source']}`: {rp['mismatches']} differ "
                 f"by more than {rp['tolerance']:g}.")
        for b in rp["first_mismatches"]:
            L.append(f"- {b['path']}: here {b['here']}, report {b['report']}")
    else:
        L.append("DeepSeek's scored rows or its report are absent; no reproduction was run.")
    L += ["", "`scripts/verify_ladder_analysis.py` recomputes from the scored rows, with code written apart from this "
          "script (it imports neither this script nor the helpers it uses), every model's arm means, gains, w, corrected "
          "gains, readability, content − strict gaps, rows per arm, cut counts and five-shot arms in both readings and "
          "both scores, the family tables' values and directions, the placements against the panel and their marks, "
          "and DeepSeek's printed tables in `results/frontier_panel/nebius_panel.md`, and compares them with "
          "`ladder.json`. It also parses what is printed, every line of every section of this file (the template SDs "
          "recomputed from the rows, their roots rounded exactly) and the whole of `ladder_note.md`: every number, every "
          "direction word of the family tables and lines, every ↑ or ↓ mark and every tie count, compared at its printed "
          "precision with its own values or with the source file the text cites (spend, sizes, checks); a number it "
          "cannot account for is a mismatch. Each printed number is compared, as a string, with its own half-even "
          "rounding of the exact value; a value exactly halfway passes only as its even neighbour. Its last line, "
          f"`VERIFY: <n> values checked, <m> mismatches`, is in `verify.out` (`{LATEST_FIX_REPORT}`). Both scripts "
          "state each corrected-gain field the same way (`ladder.json`, `rules`): "
          "`discovery.corrected_gain_pp` is the rule's corrected gain; `pooled.corrected_gain_pp` is "
          f"{POOLED_CG}; `all_draws.corrected_gain_pp` is (1 − w over the 48 draws) × the 48-draw CoT − forced, the "
          "definition of `results/frontier_panel/nebius_panel.json`, which this script reproduces; it is not the rule's "
          "corrected gain and is printed nowhere."]
    L += ["", "## 12. Edge cases", ""]
    nt = [(k, t, r, s, q) for k, a in place_.items() for t, b in a.items() for r, c in b.items()
          for s, d in c.items() for q, p in d.items() if p.get("equal")]
    for k, v in R.get("bbh_forced_gap", {}).get("models", {}).items():
        for f in ("zero_shot_forced_discovery", "fewshot_forced_all_rows_placement"):
            if (v.get(f) or {}).get("equal"):
                nt.append((k, "bbh", "cut", "forced gap", f))
    n_cut = sum(r == 'cut' for _, _, r, _, _ in nt)
    ties = ([f"Ties with panel models: {n_cut} placement cell{'s' if n_cut != 1 else ''} in the cut reading and "
             f"{sum(r == 'uncut' for _, _, r, _, _ in nt)} in the uncut reading equal a panel value (within 1e-9); each "
             "printed cell carries its tie count (\"k tied\"); the cells are in `ladder.json` (`placement`, `equal`)"]
            if nt else [])
    ts = cvu.get("ties_separating", [])
    ties += ([f"Ties between ladder models: {len(ts)} pair{'s' if len(ts) != 1 else ''} tied exactly in one reading "
              f"separate{'' if len(ts) != 1 else 's'} in the other (section 9), counted apart from the reversals"]
             if ts else [])
    ec = ([f"{short(k)}: {e}" for k, m in models.items() for e in m["edge_cases"]] + R["panel_edge_cases"]
          + R.get("panel_off_grid", []) + ties + [f"Printing: {e}" for e in ROUNDING_EDGE])
    L += [f"- {e}" for e in ec] or ["- None."]
    L += ["", "## 13. Sources", ""]
    for k, m in models.items():
        L.append(f"- {m['name']}: " + ("; ".join(f"{r} `{m['files'][r][TASKS[0]].replace(TASKS[0], '<task>', 1)}`"
                                                for r in READINGS))
                 + f"; design `{R['completeness'][k]['tasks'][TASKS[0]]['design_file'].replace(TASKS[0], '<task>', 1)}`")
    for t, s in R["panel_sources"].items():
        L.append(f"- Panel, {TASK_NAME[t]}: " + ", ".join(f"`{x}`" for x in s))
    mxc = max(v for d in R["panel_checks"].values() for v in d.values())
    L.append(f"- Panel families: `{PANEL_FILES['families']}`. The gains in the panel's two files per benchmark agree "
             f"(largest |difference| {mxc:.1g} pp over {len(R['panel_checks'])} checks; `ladder.json`, "
             "`panel_checks`).")
    L.append(f"- Loop state (models not yet scored): `{LOOP_LOG}`.")
    L.append(f"- Runtime {R['runtime_s']:.0f} s.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def rung_precision(r: dict, R: dict) -> str:
    """A rung's precision in a few words: a panel rung's dtype in the registries, or the ladder model's (section 4,
    footnote F3)."""
    if r["source"] == "panel":
        d = R["panel_precision"].get(r["key"])
        return d["dtype"] if d else "precision not recorded"
    p = LADDER[r["key"]]["precision"]
    return "API, precision not stated" if p.startswith("not stated") else p.split(" (")[0].split(":")[0]


def write_ladder_note(R: dict, path: Path) -> None:
    """A summary note: what is in, the headline numbers, the footnotes, the limits, what waits.
    Facts and numbers only; written from the same results as ladder.md, so a rerun keeps it current."""
    models, comp = R["models"], R["completeness"]
    present = present_keys(models)
    waiting = [k for k, m in models.items() if m["status"] != "complete"]
    L = [f"# Size ladder: note ({R['created']})", "",
         "Facts and numbers for the manuscript; no prose to paste. Full analysis: `results/ladder/analysis/ladder.md` "
         "(and `ladder.json`); code `scripts/ladder_analysis.py`, independent check `scripts/verify_ladder_analysis.py` "
         "(result in `verify.out`). Numbers in pp; w is a share. Primary reading: outputs cut at the panel's caps. "
         "Every number is rounded half-even on its exact value, so a value exactly halfway prints as its even "
         "neighbour; counts, marks and direction words are judged on the unrounded values.",
         "", "## What is in", ""]
    for k, m in models.items():
        lad = m["ladder"] or OUTSIDE
        L.append(f"- {m['name']}: {m['status']}; {comp[k]['state']}. Ladder: {lad}. Size: "
                 f"{size_text(R['sizes'][k], with_source=False)}. Precision: "
                 f"{m['precision'].split(' (')[0].split(':')[0]}.")
    L += ["", "Sizes with their sources: `ladder.md` section 4. Rungs are ordered by total parameters; the active "
          "parameters and the architecture change between rungs as well (Limits)."]
    L += ["", "## Family ladders (content; cut reading)", "",
          "On the panel's basis, named in each sentence: GSM8K, MATH-500 and BIG-Bench Hard on the discovery draws, "
          "HotpotQA content on pooled rows; so a value can differ in the first decimal from the 48-draw columns of the "
          "table below. Direction over size: " + DIRECTION_RULE + " Strict values and the w and corrected-gain rows: "
          "`ladder.md` section 5.", "",
          "Rungs, smallest first, and where each comes from: " + "; ".join(
              f"{name}: " + " → ".join(f"{rung_label(r['key'])} ({r['source']}, {rung_precision(r, R)})"
                                       for r in f["rungs"])
              for name, f in R["families"].items()) + f". The panel ran in {R['panel_precision_phrase']} (footnote F3).",
          ""]
    for name, f in R["families"].items():
        L.append(f"- {family_line(name, f)}")
    L += ["", "## Headline numbers per model (cut reading)", "",
          "Each column names its basis, and the columns are grouped by it. Accuracy and gains: mean over the sampler's "
          "48 draws. The last three columns are on the 24 discovery draws (for HotpotQA content too): the CoT − forced "
          "that the rule's corrected gain shrinks, w, and the corrected gain (1 − w) × (CoT − forced); since w lies in "
          "[0, 1], the corrected gain is never larger in size than the CoT − forced beside it. Strict / content.",
          "",
          "| Model | Benchmark | Requested (48 draws) | Forced (48 draws) | CoT (48 draws) | CoT − requested (48 draws) "
          "| CoT − forced (48 draws) | CoT − forced (discovery draws) | w (discovery draws) | Corrected gain (discovery "
          "draws) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in present:
        for t, tb in models[k]["tasks"].items():
            s = tb.get("cut")
            if not s:
                continue

            def pair(f):
                return f"{fnum(f(s['strict']))} / {fnum(f(s['content']))}"
            L.append(f"| {nm(k)} | {TASK_NAME[t]} | "
                     + pair(lambda b: b["all_draws"]["accuracy_pp"]["requested"]) + " | "
                     + pair(lambda b: b["all_draws"]["accuracy_pp"]["forced"]) + " | "
                     + pair(lambda b: b["all_draws"]["accuracy_pp"]["cot"]) + " | "
                     + pair(lambda b: b["all_draws"]["gain_over_requested_pp"]) + " | "
                     + pair(lambda b: b["all_draws"]["gain_over_forced_pp"]) + " | "
                     + pair(lambda b: b["discovery"]["gain_over_forced_pp"]) + " | "
                     + f"{fnum(s['w']['discovery'], 3)} | "
                     + pair(lambda b: b["discovery"]["corrected_gain_pp"]) + " |")
    L += ["", "## Against the panel, the baseline shift, formats, cut against uncut", ""]
    L += [f"- {x}" for x in R["findings"][2 + len(R["families"]):]]
    L += ["", "## Footnotes to carry", ""]
    L += [f"- {x}" for x in R["footnotes"]]
    L += ["", "## Limits", ""]
    L += [f"- {x}" for x in R["scope"]]
    L += ["", "## What waits", ""]
    if waiting:
        L += [f"- {models[k]['name']}: {comp[k]['state']}. Its family ladder ({models[k]['ladder'] or 'none'}), the "
              "placements and this note gain its rows when `scripts/score_ladder.py` has scored it; then rerun "
              "`scripts/ladder_analysis.py` and `scripts/verify_ladder_analysis.py`. No code change is needed."
              for k in waiting]
    L.append("- Closed models, w only: not run; waits for a key and the price check.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def analyse(root: Path = ROOT, out_dir: Path | None = None, keys: list[str] | None = None) -> dict:
    t0 = time.time()
    ROUNDING_EDGE.clear()
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = keys or list(LADDER)
    models = {k: model_block(k, root) for k in keys}
    panel = load_panel(root)
    place_ = placements(models, panel)
    stop = stop_unread(root)
    fam = family_tables(models, panel, stop)
    cvu = cut_vs_uncut(models, place_)
    cvu["family_direction_changes"] = family_uncut_changes(models, panel, fam, stop)
    cvu["cap_dependence"] = cap_dependence(models, place_)
    comp = completeness(keys, root)
    rep = None
    nj = root / NEBIUS_JSON
    ds = models.get("deepseek-v4.1-flash-1017")
    if ds and ds["status"] != "missing" and nj.exists():
        rep = reproduce_nebius(ds, json.loads(nj.read_text(encoding="utf-8")))
    pec = []
    for k, d in panel["checks"].items():
        for kk, v in d.items():
            if v > 1e-9:
                pec.append(f"panel {k}: {kk} = {v:.3g} pp between the panel's files")
    R = {"created": time.strftime("%Y-%m-%d %H:%M"), "units": "pp; w as a share in [0, 1]",
         "readings": {"cut": "primary: every output cut at the panel's cap of its arm (token level)",
                      "uncut": "supplementary: the outputs at the raised cap (7,000)"},
         "rules": {"accuracy": "mean over the sampler's 48 draws of the per-configuration accuracy on complete units",
                   "w": "share of requested outputs longer than the forced cap, per configuration, mean over the 24 "
                        "discovery draws (src/baseline_mixture.RULE_TEXT)",
                   "corrected_gain": "the rule's (src/baseline_mixture.RULE_TEXT): (1 - w) x (CoT - forced), w and the "
                                     "gain both on the 24 discovery draws; field <score>.discovery.corrected_gain_pp. "
                                     "Every corrected gain printed in ladder.md and ladder_note.md is this one, except "
                                     "the HotpotQA content cells of the placements and family tables, which are "
                                     "labelled pooled rows",
                   "corrected_gain_fields": {
                       "discovery.corrected_gain_pp": "(1 - w.discovery) x discovery.gain_over_forced_pp: the rule's",
                       "pooled.corrected_gain_pp": "(1 - w.discovery) x pooled.gain_over_forced_pp: the basis of the "
                                                   "panel's HotpotQA content file; printed only where labelled pooled "
                                                   "rows (HotpotQA content in the placements and family tables)",
                       "all_draws.corrected_gain_pp": "(1 - w.all_draws) x all_draws.gain_over_forced_pp: w and the "
                                                      "gain over the 48 draws, the definition of "
                                                      "results/frontier_panel/nebius_panel.json, which this file "
                                                      "reproduces; not the rule's corrected gain; printed nowhere"},
                   "strict_content_gap": "strict_content_gap_pp: content - strict over all rows of the arm; "
                                         "strict_content_gap_discovery_pp: content - strict of the zero-shot arms' "
                                         "discovery-draw accuracies (the same units and draws)",
                   "readable": "readable_pct: share of replies with the strict scorer's compliant flag, per arm; "
                               "discovery and all_draws: per-configuration share averaged over the draws; all_rows: "
                               "over every row of the arm (zero-shot arms: on the units with all three arms)",
                   "direction": DIRECTION_RULE,
                   "marks": "a placement cell reads value (b/M), b panel models below the value, and ↑ or ↓ when the "
                            "value lies above the panel's maximum or below its minimum (tolerance 1e-9); the same "
                            "in every table and sentence",
                   "ties": "a value equal to k panel values (within 1e-9) prints its cell as value (b/M, k tied) and "
                           "its count as 'above b of M, k tied'; a pair of ladder models tied exactly in one reading "
                           "and ordered in the other is listed in cut_vs_uncut.ties_separating, never as an ordering "
                           "flip; ordering_flips holds the pairs that both readings order, in opposite ways",
                   "format_share": "strict minus content as a share of the strict value (%), both on the discovery "
                                   "draws, for CoT - requested and for the residual; None when the strict value is 0",
                   "sizes": "total and active parameters (B, as the source writes them) and the architecture, each "
                            "from the recorded source named in sizes.<key>.source; None where nothing is recorded. "
                            "Published figures (published: true; Mistral Large 3, DeepSeek V4.1-Flash-1017) carry the "
                            "page's URL, read date and quoted sentences from " + PUBLISHED_SIZES + "; DeepSeek's "
                            "active parameters are published apart for input and output (active_input, active_output; "
                            "active None) and are V4.1-Flash's, taken to hold for the 1017 checkpoint (caveat). "
                            "sizes.deepseek-v4.1-flash-1017.predecessor is V4-Flash's recorded size, "
                            "kept for the record, used: false",
                   "cut_vs_uncut": "every placed quantity (the placements' own list: per score the seven quantities "
                                   "and the residual; the readable shares of the zero-shot arms where the panel has "
                                   "them) on the discovery draws; placements against the panel on the panel's basis; "
                                   "the largest change names every quantity that shares it (largest, and largest_scores "
                                   "without the readable shares); cap_dependence: section 1's findings recomputed with "
                                   "the uncut values",
                   "placement": "on the panel file's basis: discovery draws; HotpotQA content pooled rows",
                   "rounding": ROUNDING_RULE},
         "models": models, "status": {k: m["status"] for k, m in models.items()},
         "placement": place_, "families": fam, "cut_vs_uncut": cvu, "reproduction": rep,
         "panel": {t: df.reset_index().to_dict("records") for t, df in panel["tasks"].items()},
         "panel_basis": panel["basis"], "panel_sources": panel["sources"], "panel_checks": panel["checks"],
         "panel_families": panel["families"], "panel_edge_cases": pec, "completeness": comp, "_panel": panel,
         "stop_unread": stop}
    R["stop_counts"] = stop_counts(root)
    R["bbh_forced_gap"] = bbh_forced_gap(models, panel, root)
    R["forced_other_max"] = forced_other_max(models)
    R["sizes"] = {k: recorded_size(root, k) for k in list(LADDER) + [r for f in FAMILY_LADDERS.values()
                                                                     for r, s in f["rungs"] if s == "panel"]}
    R["spend"] = read_spend(root)
    R["panel_precision"] = panel_precision(root, sorted(panel["families"]))
    R["panel_precision_phrase"] = precision_phrase(R["panel_precision"])
    R["format_shares"] = format_shares(R)
    R["findings"] = findings(R)
    R["scope"] = scope_lines(R)
    R["asks"] = asks_lines(R, root)
    R["footnotes"] = footnotes(R, root)
    R["panel_grid"] = panel["grid"]
    R["panel_off_grid"] = panel["off_grid"]
    R["runtime_s"] = time.time() - t0
    # ladder_note.md first, so that section 12 of ladder.md lists every float met on a rounding boundary
    wn = out_dir / "ladder_note.md"
    write_ladder_note(R, wn)
    (out_dir / "ladder_note_clean.md").write_text(npl.npt.strip_links(wn.read_text(encoding="utf-8")), encoding="utf-8")
    md = out_dir / "ladder.md"
    write_md(R, md)
    (out_dir / "ladder_clean.md").write_text(npl.npt.strip_links(md.read_text(encoding="utf-8")), encoding="utf-8")
    R["rounding_edge_cases"] = list(ROUNDING_EDGE)
    (out_dir / "ladder.json").write_text(json.dumps({k: v for k, v in R.items() if not k.startswith("_")}, indent=1,
                                                    default=as_json), encoding="utf-8")
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output folder (default results/ladder/analysis)")
    a = ap.parse_args(argv)
    R = analyse(ROOT, Path(a.out) if a.out else None)
    print(f"wrote {Path(a.out) if a.out else OUT}/ladder.md; status " + ", ".join(f"{k} {v}" for k, v in R["status"].items())
          + (f"; DeepSeek reproduction: {R['reproduction']['mismatches']} of {R['reproduction']['values_compared']} differ"
             if R["reproduction"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
