r"""Freeze the Dataset 2 prompt bank: the retained configurations of a screened candidate bank.

The candidate comes from scripts/build_bank_candidates.py (data/bank_candidates/<name>.json, built from
config/promptsuite.yaml). The exclusion log comes from scripts/screen_d2.py under the
admissibility rules (manifests/gsm8k_admissibility.json) and must have screened that
candidate, all of it. The frozen bank keeps exactly the retained configurations, sets their sampling
probabilities under the law recorded in the candidate (uniform over the retained bank is the only law
implemented) and records the candidate and log it came from (key frozen_from). Uses metadata only.

PowerShell:
    .\.venv\Scripts\python.exe scripts\freeze_bank_d2.py --candidate data\bank_candidates\<name>.json

Refused (exit 2, nothing written): an existing output (a frozen bank is never overwritten), an example
candidate, a candidate built from an unconfirmed configuration, an unknown sampling law, a log that
screened another bank or part of this one, a log with no evaluated configuration rule.

BIG-Bench Hard (--task bbh; registration E2: "The bank is frozen after the screen"): the candidate is
data/bank_candidates/bbh_bank_v1.json (the default), the log manifests/bbh_admissibility.json and the output
manifests/bbh_bank.json. A log that is not a BIG-Bench Hard log (its task) is refused.
    .\.venv\Scripts\python.exe scripts\freeze_bank_d2.py --task bbh
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import bbh_d2, d2_io, hotpot_d2, math_d2, string_tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULTS = {"gsm8k": {"candidate": None, "admissibility": os.path.join(ROOT, "manifests", "gsm8k_admissibility.json"),
                      "out": os.path.join(ROOT, "manifests", "gsm8k_bank.json")},
            "bbh": {"candidate": str(bbh_d2.CANDIDATE_BANK), "admissibility": str(bbh_d2.ADMISSIBILITY_LOG),
                    "out": str(bbh_d2.BANK)},
            "math": {"candidate": str(math_d2.CANDIDATE_BANK), "admissibility": str(math_d2.ADMISSIBILITY_LOG),
                     "out": str(math_d2.BANK)},
            "hotpot": {"candidate": str(hotpot_d2.CANDIDATE_BANK), "admissibility": str(hotpot_d2.ADMISSIBILITY_LOG),
                       "out": str(hotpot_d2.BANK)}}


def freeze(candidate_path, out_path, admissibility_path=None, task="gsm8k") -> dict:
    out = Path(out_path)
    if out.exists():
        raise d2_io.BankError(f"{d2_io.rel_path(out)} exists; a frozen bank is never overwritten")
    candidate, sha = d2_io.load_bank(candidate_path)
    log, log_info = d2_io.load_admissibility(admissibility_path)
    if task != "gsm8k" and log.get("task") != task:
        raise d2_io.BankError(f"{log_info['path']} is a {log.get('task', 'gsm8k')!r} exclusion log, not a {task!r} one")
    bank = d2_io.frozen_bank(candidate, sha, log, candidate_path=d2_io.rel_path(candidate_path), log_info=log_info)
    d2_io._write_json(out, bank)
    return {"out": d2_io.rel_path(out), "sha256": d2_io.sha256_path(out), **{
        k: bank["frozen_from"][k] for k in ("n_candidate_configurations", "n_retained", "excluded_configuration_ids",
                                            "sampling_law")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k")
    ap.add_argument("--candidate", help="candidate bank from scripts/build_bank_candidates.py (required for gsm8k; "
                                        "bbh: data/bank_candidates/bbh_bank_v1.json; math: data/bank_candidates/math_bank_v1.json)")
    ap.add_argument("--admissibility", help="default manifests/gsm8k_admissibility.json (bbh: manifests/bbh_admissibility.json)")
    ap.add_argument("--out", help="default manifests/gsm8k_bank.json (bbh: manifests/bbh_bank.json)")
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.candidate is None:
        ap.error("the following arguments are required: --candidate")
    try:
        res = freeze(a.candidate, a.out, a.admissibility, task=a.task)
    except d2_io.D2IOError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
