"""Scores reproduction outputs with the authors' corrected extractor (src/reproduction.py).

Each --work-dir is one model's generation folder of one part. A folder is scored only when its newest run log is a
complete full-mode run without --limit, its request file is the one listed in the part's build record, and the run
used config/models_reproduction.yaml and the model's decoding file from config/reproduction.yaml. Every request of
the file must have exactly one output.

    python scripts/score_reproduction.py --part numerical --work-dir <one dir per model>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generation as gen, gsm8k, reproduction as rp  # noqa: E402


def check_folder(d: Path, part: str, record: dict, cfg: dict) -> tuple[str, dict]:
    logs = sorted(d.glob("run_log-*.json"))
    if not logs:
        raise SystemExit(f"{d}: no run log")
    log = json.loads(logs[-1].read_text(encoding="utf-8"))
    m = log["model"]["model_key"]
    problems = []
    if log.get("status") != "complete" or log.get("mode") != "full" or log["inputs"].get("limit") is not None:
        problems.append(f"newest run log is status {log.get('status')!r}, mode {log.get('mode')!r}, limit {log['inputs'].get('limit')!r}")
    entry = (record.get("models") or {}).get(m)
    if not entry or entry["requests_sha256"] != log["inputs"]["requests_sha256"]:
        problems.append(f"request file of {m} is not the one in the {part} build record")
    if log["inputs"].get("models_yaml_sha256") != rp.sha256_file(rp.ROOT / cfg["registry"]):
        problems.append("the run did not use config/models_reproduction.yaml as it is now")
    dec = cfg["decoding"]["by_model"].get(m, cfg["decoding"]["default"])
    if log["inputs"].get("decoding_yaml_sha256") != rp.sha256_file(rp.ROOT / dec):
        problems.append(f"the run did not use {dec}")
    gpus = {(json.loads(p.read_text(encoding="utf-8")).get("gpu", {}).get("devices") or [{}])[0].get("name") for p in logs}
    if len(gpus) > 1:
        problems.append(f"runs on more than one GPU type: {sorted(map(str, gpus))}")
    if problems:
        raise SystemExit(f"{d}: " + "; ".join(problems))
    return m, entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=rp.PARTS, required=True)
    ap.add_argument("--work-dir", nargs="+", required=True)
    ap.add_argument("--requests-dir", default=str(rp.REQUESTS_DIR))
    ap.add_argument("--out", help="default data/processed/reproduction/<part>_scored.parquet")
    a = ap.parse_args(argv)
    cfg = rp.load_config()
    record = json.loads((Path(a.requests_dir) / a.part / "build_record.json").read_text(encoding="utf-8"))
    items = gsm8k.load_gsm8k(cfg["split"])
    if gsm8k.canonical_sha256(items) != record["items"]["canonical_sha256"]:
        raise SystemExit("GSM8K test items differ from the build record")
    parts = []
    for wd in a.work_dir:
        d = Path(wd)
        m, entry = check_folder(d, a.part, record, cfg)
        req = pd.read_parquet(Path(a.requests_dir) / a.part / entry["requests_file"])
        out = gen.read_outputs(d)
        if out["request_id"].duplicated().any() or set(out["request_id"]) != set(req["request_id"]):
            raise SystemExit(f"{d}: outputs do not match the request file one to one")
        s = rp.score_outputs(out, items, a.part)
        parts.append(s)
        print(f"{a.part:<10} {m:<28} {len(s):>6} outputs, {s['null_pred'].sum():>5} NULL, {s['truncated'].sum():>5} truncated")
    scored = pd.concat(parts, ignore_index=True)
    dest = Path(a.out) if a.out else rp.ROOT / "data" / "processed" / "reproduction" / f"{a.part}_scored.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(dest, index=False)
    print(f"wrote {dest} ({len(scored)} rows, scorer {rp.SCORER_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
