r"""Draw the configurations of the Dataset 2 throughput and noise tranche from the frozen bank (config/tranche.yaml).

The draw uses the ids of the frozen bank (manifests/gsm8k_bank.json) and the seed of config/tranche.yaml only, uniformly
and without replacement. The bank must be the retained bank of the exclusion log (manifests/gsm8k_admissibility.json),
and the tranche covers the models the log retained. Writes manifests/gsm8k_tranche.json once and prints the request
build for it (scripts/build_requests.py --tranche).

Usage (PowerShell):
    .\.venv\Scripts\python.exe scripts\draw_tranche_d2.py

BIG-Bench Hard (--task bbh): config/tranche_bbh.yaml (8 configurations, seed 20261031, fixed before any
BIG-Bench Hard output existed), the frozen bank
manifests/bbh_bank.json, the log manifests/bbh_admissibility.json, the stream
sampling.rng_for("bbh-tranche-configurations", 20261031) and the output manifests/bbh_tranche.json. Its
calibration_items must be the size of the calibration set of manifests/bbh_items.json (1,350).
    .\.venv\Scripts\python.exe scripts\draw_tranche_d2.py --task bbh

Exit codes: 0 written (or already identical), 2 refused, 1 any other error.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import bbh_d2, d2_io, math_d2, sampling, string_tasks  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
TRANCHE = ROOT / "config" / "tranche.yaml"
BANK = ROOT / "manifests" / "gsm8k_bank.json"
OUT = ROOT / "manifests" / "gsm8k_tranche.json"
SEED_LABEL = "gsm8k-tranche-configurations"
BBH_SEED_LABEL = "bbh-tranche-configurations"
MATH_SEED_LABEL = "math-tranche-configurations"
STRING_TASK = {"bbh": {"seed_label": BBH_SEED_LABEL, "kind": "bbh_tranche",
                       "rule": "config/tranche_bbh.yaml"},
               "math": {"seed_label": MATH_SEED_LABEL, "kind": "math_tranche",
                        "rule": "config/tranche_math.yaml"}}
DEFAULTS = {"gsm8k": {"tranche": TRANCHE, "bank": BANK, "admissibility": None, "out": OUT},
            "bbh": {"tranche": bbh_d2.TRANCHE_CONFIG, "bank": bbh_d2.BANK, "admissibility": bbh_d2.ADMISSIBILITY_LOG,
                    "out": bbh_d2.TRANCHE_MANIFEST},
            "math": {"tranche": math_d2.TRANCHE_CONFIG, "bank": math_d2.BANK, "admissibility": math_d2.ADMISSIBILITY_LOG,
                     "out": math_d2.TRANCHE_MANIFEST}}


class Refused(Exception):
    pass


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def write_new(path: Path, raw: bytes) -> str:
    """Write once: identical content is left alone, different content is refused."""
    if path.exists() and path.read_bytes() != raw:
        raise Refused(f"{d2_io.rel_path(path)} exists with different content; it is not overwritten")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_tranche(path: Path = TRANCHE) -> dict:
    t = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if t.get("status") != "confirmed":
        raise Refused(f"{d2_io.rel_path(path)} is not confirmed")
    for k in ("configurations", "seed", "calibration_items"):
        if not isinstance(t.get(k), int) or isinstance(t.get(k), bool):
            raise Refused(f"{d2_io.rel_path(path)}: {k} must be an integer")
    if t["configurations"] < 2:
        raise Refused("the gain noise needs at least two configurations")
    return t


def draw_configurations(ids: list[str], k: int, seed: int, label: str = SEED_LABEL) -> list[str]:
    """k configuration ids drawn without replacement from the sorted ids, from metadata only."""
    ids = sorted(ids)
    if not 1 <= k <= len(ids):
        raise Refused(f"cannot draw {k} configurations from {len(ids)}")
    pick = sampling.rng_for(label, seed).choice(len(ids), size=k, replace=False)
    return sorted(ids[int(i)] for i in pick)


def draw(tranche_path: Path = TRANCHE, bank_path: Path = BANK, admissibility_path: Path | None = None,
         out_path: Path = OUT, task: str = "gsm8k", items_manifest: Path | None = None) -> dict:
    if string_tasks.module(task) is not None:
        return draw_bbh(tranche_path, bank_path, admissibility_path, out_path, items_manifest, task=task)
    t = load_tranche(tranche_path)
    try:
        bank, bank_sha = d2_io.load_bank(bank_path)
        log, log_info = d2_io.load_admissibility(admissibility_path)
    except d2_io.D2IOError as exc:
        raise Refused(str(exc)) from exc
    frozen = (bank.get("frozen_from") or {}).get("admissibility") or {}
    if frozen.get("sha256") != log_info["sha256"]:
        raise Refused(f"{d2_io.rel_path(bank_path)} was not frozen from {log_info['path']} "
                      "(scripts/freeze_bank_d2.py)")
    problems = d2_io.bank_admissibility_problems(bank, log)
    if problems:
        raise Refused("the bank is not the retained bank: " + "; ".join(problems))
    models = list(log["retained_models"])
    if not models:
        raise Refused("the screening retained no model")
    ids = [str(c) for c in bank["configurations"]]
    chosen = draw_configurations(ids, t["configurations"], t["seed"])
    record = {"kind": "gsm8k_tranche", "rule": "config/tranche.yaml",
              "tranche_config_sha256": sha256_file(tranche_path),
              "bank": {"path": d2_io.rel_path(bank_path), "sha256": bank_sha},
              "admissibility": log_info, "seed_label": SEED_LABEL, "seed": t["seed"],
              "n_bank_configurations": len(ids), "configurations": chosen, "models": models,
              "calibration_items": t["calibration_items"], "excluded_from_main": True}
    sha = write_new(Path(out_path), json_bytes(record))
    build = (r".\.venv\Scripts\python.exe scripts\build_requests.py --bank " + str(Path(record["bank"]["path"]))
             + " --models " + " ".join(models) + " --configs " + " ".join(chosen)
             + f" --tranche --split train --calibration-items {t['calibration_items']} --out-dir data\\requests\\gsm8k")
    return {"manifest": d2_io.rel_path(out_path), "sha256": sha, "configurations": chosen, "models": models,
            "build_requests": build}


def draw_bbh(tranche_path: Path, bank_path: Path, admissibility_path: Path | None, out_path: Path,
             items_manifest: Path | None = None, task: str = "bbh") -> dict:
    """The BIG-Bench Hard (or MATH-500) tranche (module doc): the GSM8K rule with its own seed label, bank, log and
    output."""
    tm, st = string_tasks.module(task), STRING_TASK[task]
    t = load_tranche(tranche_path)
    try:
        bank, bank_sha = d2_io.load_bank(bank_path)
        log, log_info = d2_io.load_admissibility(admissibility_path or tm.ADMISSIBILITY_LOG)
        cal_items, cal_info = tm.load_items("calibration", manifest_path=items_manifest)
    except d2_io.D2IOError as exc:
        raise Refused(str(exc)) from exc
    if log.get("task") != tm.TASK:
        raise Refused(f"{log_info['path']} is not a {tm.TITLE} exclusion log (task {log.get('task')!r})")
    if t["calibration_items"] != len(cal_items):
        raise Refused(f"{d2_io.rel_path(tranche_path)}: calibration_items {t['calibration_items']} is not the size of "
                      f"the recorded calibration set ({len(cal_items)})")
    frozen = (bank.get("frozen_from") or {}).get("admissibility") or {}
    if frozen.get("sha256") != log_info["sha256"]:
        raise Refused(f"{d2_io.rel_path(bank_path)} was not frozen from {log_info['path']} "
                      f"(scripts/freeze_bank_d2.py --task {tm.TASK})")
    problems = d2_io.bank_admissibility_problems(bank, log)
    if problems:
        raise Refused("the bank is not the retained bank: " + "; ".join(problems))
    models = list(log["retained_models"])
    if not models:
        raise Refused("the screening retained no model")
    ids = [str(c) for c in bank["configurations"]]
    chosen = draw_configurations(ids, t["configurations"], t["seed"], st["seed_label"])
    record = {"kind": st["kind"], "task": tm.TASK, "rule": st["rule"],
              "tranche_config_sha256": sha256_file(tranche_path),
              "bank": {"path": d2_io.rel_path(bank_path), "sha256": bank_sha},
              "admissibility": log_info, "seed_label": st["seed_label"], "seed": t["seed"],
              "n_bank_configurations": len(ids), "configurations": chosen, "models": models,
              "calibration_items": {"item_set": "calibration", "n_items": len(cal_items),
                                    "canonical_sha256": cal_info["canonical_sha256"]},
              "excluded_from_main": True}
    sha = write_new(Path(out_path), json_bytes(record))
    build = (r".\.venv\Scripts\python.exe scripts\build_requests.py --task " + tm.TASK + " --bank "
             + str(Path(record["bank"]["path"]))
             + " --models " + " ".join(models) + " --configs " + " ".join(chosen) + " --tranche")
    return {"manifest": d2_io.rel_path(out_path), "sha256": sha, "configurations": chosen, "models": models,
            "build_requests": build}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=string_tasks.TASKS, default="gsm8k")
    ap.add_argument("--tranche", type=Path, help="default config/tranche.yaml (bbh: config/tranche_bbh.yaml)")
    ap.add_argument("--bank", type=Path, help="default manifests/gsm8k_bank.json (bbh: manifests/bbh_bank.json)")
    ap.add_argument("--admissibility", type=Path, default=None,
                    help="default manifests/gsm8k_admissibility.json (bbh: manifests/bbh_admissibility.json)")
    ap.add_argument("--out", type=Path, help="default manifests/gsm8k_tranche.json (bbh: manifests/bbh_tranche.json)")
    ap.add_argument("--items-manifest", type=Path, help="bbh: override manifests/bbh_items.json (tests)")
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    try:
        res = draw(a.tranche, a.bank, a.admissibility, a.out, task=a.task, items_manifest=a.items_manifest)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(res, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
