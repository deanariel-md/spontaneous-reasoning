r"""The requested answer format of Dataset 2, chosen by a screening rule (config/answer_formats.yaml).

  banks    write one builder configuration per candidate (config/promptsuite.yaml with that answer format), build each
           candidate bank in .venv-promptsuite, and draw the screening configurations by seed, the same for every
           candidate. Writes manifests/gsm8k_format_screen.json and a metrics-only copy of the admissibility rules
           (status PENDING_REVIEW, every threshold null), because confirmed rules need a scorer manifest.
  select   read one metrics-only screening report per candidate (scripts/screen_d2.py --rules <that copy>) and
           register the candidate whose lowest strict compliance over models and arms is highest; a tie goes to the
           earlier candidate. Writes manifests/gsm8k_scorer.json (src/scoring.py scorer_record) and
           manifests/gsm8k_format_selection.json with every value behind the choice.

Usage:
    python scripts/select_answer_format_d2.py banks
    python scripts/select_answer_format_d2.py select --report answer_colon_markup=<report.json> ...

Exit codes: 0 written (or already identical), 2 refused, 1 any other error.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import bank_build as bb, sampling, scoring  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
FORMATS = ROOT / "config" / "answer_formats.yaml"
BANK_CONFIG = ROOT / "config" / "promptsuite.yaml"
RULES = ROOT / "config" / "admissibility.yaml"
WORK = ROOT / "data" / "bank_candidates" / "format_screen"
SCREEN_MANIFEST = ROOT / "manifests" / "gsm8k_format_screen.json"
SCORER = ROOT / "manifests" / "gsm8k_scorer.json"
SELECTION = ROOT / "manifests" / "gsm8k_format_selection.json"
BUILDER = ROOT / "scripts" / "build_bank_candidates.py"
VENV_PY = ROOT / ".venv-promptsuite" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SEED_LABEL = "gsm8k-format-screen"


class Refused(Exception):
    pass


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def write_new(path: Path, raw: bytes) -> str:
    """Write once: identical content is left alone, different content is refused."""
    if path.exists() and path.read_bytes() != raw:
        raise Refused(f"{rel(path)} exists with different content; it is not overwritten")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_formats(path: Path = FORMATS) -> dict:
    f = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if f.get("status") != "confirmed":
        raise Refused(f"{path} is not confirmed")
    names = [c["scorer_candidate"] for c in f["candidates"]]
    unknown = [n for n in names if n not in scoring.CANDIDATE_SPECS]
    if unknown or len(set(names)) != len(names):
        raise Refused(f"candidates must be distinct names in src/scoring.py CANDIDATE_SPECS: {names}")
    for c in f["candidates"]:
        s = scoring.CANDIDATE_SPECS[c["scorer_candidate"]]
        msg = bb.answer_format_problem(c["text"], s.template, s.case_sensitive)
        if msg:
            raise Refused(f"{c['scorer_candidate']}: {msg}")
    return f


def draw_configurations(ids: list[str], k: int, seed: int) -> list[str]:
    """k configuration ids drawn without replacement from the sorted ids, from metadata only."""
    ids = sorted(ids)
    if not 1 <= k <= len(ids):
        raise Refused(f"cannot draw {k} configurations from {len(ids)}")
    pick = sampling.rng_for(SEED_LABEL, seed).choice(len(ids), size=k, replace=False)
    return sorted(ids[int(i)] for i in pick)


def metrics_only_rules(path: Path = RULES) -> dict:
    """The admissibility rules with status PENDING_REVIEW and every threshold null: screen_d2 then reports metrics only."""
    r = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    r["status"] = "PENDING_REVIEW"
    r["configurations"] = {k: {a: None for a in v} for k, v in r["configurations"].items()}
    r["models"] = {k: {x: None for x in v} for k, v in r["models"].items()}
    return r


def cmd_banks(args) -> dict:
    formats = load_formats()
    base = yaml.safe_load(BANK_CONFIG.read_text(encoding="utf-8"))
    WORK.mkdir(parents=True, exist_ok=True)
    out = []
    for c in formats["candidates"]:
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"{base['name']}-format-{c['scorer_candidate']}"
        cfg["answer_format"] = {"scorer_candidate": c["scorer_candidate"], "text": c["text"]}
        cpath = WORK / f"{cfg['name']}.yaml"
        cpath.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        res = subprocess.run([str(VENV_PY), str(BUILDER), "--config", str(cpath), "--out-dir", str(WORK)],
                             capture_output=True, text=True, encoding="utf-8", timeout=900)
        if res.returncode != 0:
            raise Refused(f"the builder refused {cpath.name}: {res.stderr.strip()[-1200:]}")
        bpath = WORK / f"{cfg['name']}.json"
        bank = json.loads(bpath.read_text(encoding="utf-8"))
        out.append({"scorer_candidate": c["scorer_candidate"], "text": c["text"], "config": rel(cpath),
                    "bank": rel(bpath), "bank_sha256": sha256_file(bpath), "ids": sorted(bank["configurations"])})
    if len({tuple(o["ids"]) for o in out}) != 1:
        raise Refused("the candidate banks hold different configuration ids, so the same configurations cannot be screened")
    ids = out[0]["ids"]
    chosen = draw_configurations(ids, int(formats["screening"]["configurations"]), int(formats["screening"]["seed"]))
    rules_path = WORK / "admissibility_metrics_only.yaml"
    rules_path.write_text(yaml.safe_dump(metrics_only_rules(), sort_keys=False), encoding="utf-8")
    manifest = {"kind": "gsm8k_format_screen", "rule": "config/answer_formats.yaml",
                "formats_sha256": sha256_file(FORMATS), "bank_config_sha256": sha256_file(BANK_CONFIG),
                "seed_label": SEED_LABEL, "seed": int(formats["screening"]["seed"]),
                "n_bank_configurations": len(ids), "configurations": chosen,
                "metrics_only_rules": rel(rules_path), "metrics_only_rules_sha256": sha256_file(rules_path),
                "candidates": [{k: o[k] for k in ("scorer_candidate", "text", "config", "bank", "bank_sha256")}
                               for o in out]}
    sha = write_new(SCREEN_MANIFEST, json_bytes(manifest))
    return {"manifest": rel(SCREEN_MANIFEST), "sha256": sha, "configurations": chosen,
            "banks": [o["bank"] for o in out]}


def min_compliance(report: dict, spec_name: str) -> tuple[float, list[dict]]:
    rows = [{"model_key": r["model_key"], "arm": r["arm"], "n": r.get("n"),
             "strict_compliance_pct": float(r["strict_compliance_pct"][spec_name])} for r in report["by_model"]]
    if not rows:
        raise Refused("a screening report has no model rows")
    return min(r["strict_compliance_pct"] for r in rows), rows


def check_report(report: dict, cand: dict, screen: dict) -> list[str]:
    out = []
    inp = report.get("inputs") or {}
    if report.get("kind") != "admissibility_screening":
        out.append("not an admissibility screening report")
    if (report.get("rules") or {}).get("applied"):
        out.append("the report applied admissibility rules; the format screening is metrics only")
    if inp.get("split") != "train" or inp.get("test_backend"):
        out.append(f"split {inp.get('split')!r}, test backend {inp.get('test_backend')!r}")
    if inp.get("bank_sha256") != cand["bank_sha256"]:
        out.append("the report was built from another bank")
    got = sorted({r["config_id"] for r in report.get("by_configuration") or []})
    if got != screen["configurations"]:
        out.append(f"the report screens configurations {got}, the manifest draws {screen['configurations']}")
    if sorted(inp.get("arms") or []) != sorted(bb.ARMS):
        out.append(f"arms {inp.get('arms')}")
    if not inp.get("item_set_identical_across_cells"):
        out.append("the cells do not share one item set")
    return out


def choose(values: list[tuple[str, float]]) -> tuple[str, bool]:
    """The candidate with the highest minimum compliance; a tie goes to the earlier one. Returns (name, tied)."""
    best = max(v for _, v in values)
    winners = [n for n, v in values if v == best]
    return winners[0], len(winners) > 1


def cmd_select(args) -> dict:
    screen = json.loads(SCREEN_MANIFEST.read_text(encoding="utf-8"))
    given = dict(x.split("=", 1) for x in args.report)
    names = [c["scorer_candidate"] for c in screen["candidates"]]
    if sorted(given) != sorted(names):
        raise Refused(f"one report per candidate is needed: {names}; given {sorted(given)}")
    per, models = [], None
    for cand in screen["candidates"]:
        name = cand["scorer_candidate"]
        path = Path(given[name])
        report = json.loads(path.read_text(encoding="utf-8"))
        problems = check_report(report, cand, screen)
        if problems:
            raise Refused(f"{name} report {path}: " + "; ".join(problems))
        m = sorted(report["inputs"]["models"])
        if models is not None and m != models:
            raise Refused(f"the reports screen different models: {models} and {m}")
        models = m
        lo, rows = min_compliance(report, name)
        per.append({"scorer_candidate": name, "min_strict_compliance_pct": lo, "by_model_and_arm": rows,
                    "report": str(path), "report_sha256": sha256_file(path)})
    winner, tied = choose([(p["scorer_candidate"], p["min_strict_compliance_pct"]) for p in per])
    spec = dataclasses.replace(scoring.CANDIDATE_SPECS[winner], description=(
        f"Registered by the format screening rule (manifests/gsm8k_format_selection.json). "
        + scoring.CANDIDATE_SPECS[winner].description.removeprefix("CANDIDATE, not a decision. ")))
    scorer_sha = write_new(SCORER, json_bytes(scoring.scorer_record(spec)))
    selection = {"kind": "gsm8k_format_selection", "rule": "highest minimum strict compliance over models and arms; "
                 "ties go to the earlier candidate (config/answer_formats.yaml)",
                 "screen_manifest_sha256": sha256_file(SCREEN_MANIFEST), "models": models,
                 "candidates": per, "registered": winner, "tie": tied,
                 "text": next(c["text"] for c in screen["candidates"] if c["scorer_candidate"] == winner),
                 "scorer_manifest": rel(SCORER), "scorer_manifest_sha256": scorer_sha}
    write_new(SELECTION, json_bytes(selection))
    return {"registered": winner, "tie": tied,
            "min_compliance": {p["scorer_candidate"]: p["min_strict_compliance_pct"] for p in per}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("banks")
    s = sub.add_parser("select")
    s.add_argument("--report", action="append", default=[], metavar="CANDIDATE=PATH",
                   help="screening_report.json of one candidate's metrics-only screening")
    args = ap.parse_args(argv)
    try:
        res = {"banks": cmd_banks, "select": cmd_select}[args.cmd](args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(res, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
