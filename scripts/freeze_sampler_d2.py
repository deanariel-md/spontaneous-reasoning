r"""Freeze the Dataset 2 sampler: discovery and validation units drawn from the frozen
bank with its sampling probabilities. Uses metadata only; run after the bank and the allocation
are fixed and before any generation.

PowerShell:
    .\.venv\Scripts\python.exe scripts\freeze_sampler_d2.py
    .\.venv\Scripts\python.exe scripts\freeze_sampler_d2.py --bank manifests\gsm8k_bank.json --k-discovery 12 --k-validation 12

Allocation defaults to config/budget.yaml dataset2_allocation. An existing sampler manifest is
never overwritten. The bank must pass src/d2_io.validate_bank and must be the retained bank of the
admissibility exclusion log (manifests/gsm8k_admissibility.json): every configuration screened and
retained, none excluded, none of the retained ones left out. The log's path and sha256 go into the
sampler manifest.

BIG-Bench Hard (--task bbh: 24 discovery and 24 validation units per arm, drawn as for
GSM8K and frozen before the collection): the frozen bank manifests/bbh_bank.json, the log
manifests/bbh_admissibility.json, the label "bbh" (sampling.registered_split_bank draws from the stream
sampling.rng_for("registered_split_bank", "bbh") under the master seed 20261003, as GSM8K's label "gsm8k")
and the output manifests/sampler_bbh.json. The allocation is the same config/budget.yaml entry.
    .\.venv\Scripts\python.exe scripts\freeze_sampler_d2.py --task bbh
"""
import argparse
import hashlib
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import bbh_d2, d2_io, hotpot_d2, math_d2, sampling, string_tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ALLOWED = (8, 12, 16, 24)
DEFAULTS = {"gsm8k": {"bank": os.path.join(ROOT, "manifests", "gsm8k_bank.json"),
                      "out": os.path.join(ROOT, "manifests", "sampler_gsm8k.json"),
                      "admissibility": os.path.join(ROOT, "manifests", "gsm8k_admissibility.json")},
            "bbh": {"bank": str(bbh_d2.BANK), "out": str(bbh_d2.SAMPLER), "admissibility": str(bbh_d2.ADMISSIBILITY_LOG)},
            "math": {"bank": str(math_d2.BANK), "out": str(math_d2.SAMPLER),
                     "admissibility": str(math_d2.ADMISSIBILITY_LOG)},
            "hotpot": {"bank": str(hotpot_d2.BANK), "out": str(hotpot_d2.SAMPLER),
                       "admissibility": str(hotpot_d2.ADMISSIBILITY_LOG)}}


def freeze(bank_path, out_path, k_d, k_v, label="gsm8k", admissibility_path=None):
    if os.path.exists(out_path):
        raise SystemExit(f"refused: {out_path} exists; a frozen sampler is never overwritten")
    for k in (k_d, k_v):
        if k not in ALLOWED:
            raise SystemExit(f"refused: allocation {k} is not one of {ALLOWED}.")
    raw = open(bank_path, "rb").read()
    try:
        bank, _ = d2_io.load_bank(bank_path)
        d2_io.validate_bank(bank)
        log, log_info = d2_io.load_admissibility(admissibility_path)
    except d2_io.D2IOError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    tm = string_tasks.module(label)
    if tm is not None and log.get("task") != tm.TASK:
        raise SystemExit(f"refused: {log_info['path']} is not a {tm.TITLE} exclusion log")
    problems = d2_io.bank_admissibility_problems(bank, log)
    if problems:
        raise SystemExit(f"refused: {bank_path} is not the retained bank of {log_info['path']}: " + "; ".join(problems))
    ids = list(bank["configurations"])
    probs = bank["sampling_probabilities"]
    p = [probs[c] for c in ids] if isinstance(probs, dict) else list(probs)
    out = sampling.registered_split_bank(ids, p, k_d, k_v, label)
    out["bank_file_sha256"] = hashlib.sha256(raw).hexdigest()
    out["bank_path"] = os.path.relpath(bank_path, ROOT)
    out["admissibility"] = log_info
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k")
    ap.add_argument("--bank", help="default manifests/gsm8k_bank.json (bbh: manifests/bbh_bank.json)")
    ap.add_argument("--out", help="default manifests/sampler_gsm8k.json (bbh: manifests/sampler_bbh.json)")
    ap.add_argument("--admissibility",
                    help="default manifests/gsm8k_admissibility.json (bbh: manifests/bbh_admissibility.json)")
    ap.add_argument("--k-discovery", type=int)
    ap.add_argument("--k-validation", type=int)
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    alloc = yaml.safe_load(open(os.path.join(ROOT, "config", "budget.yaml")))["dataset2_allocation"]
    k_d = a.k_discovery or alloc.get("discovery_units_per_arm")
    k_v = a.k_validation or alloc.get("validation_units_per_arm")
    if not (k_d and k_v):
        raise SystemExit("refused: the allocation is not set in config/budget.yaml (dataset2_allocation)")
    if not os.path.exists(a.bank):
        raise SystemExit(f"refused: {a.bank} does not exist; the prompt bank is not frozen")
    out = freeze(a.bank, a.out, k_d, k_v, label=a.task, admissibility_path=a.admissibility)
    print(f"sampler frozen: {os.path.relpath(a.out, ROOT)} ({k_d} discovery, {k_v} validation units; "
          f"{len(set(out['discovery_config_ids'] + out['validation_config_ids']))} distinct configurations)")


if __name__ == "__main__":
    main()
