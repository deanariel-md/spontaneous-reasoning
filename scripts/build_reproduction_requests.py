"""Request files for the reproduction of Cheng et al. (2025).

One parquet file per model and part with the runner's columns (request_id, model_key, arm, config_id, item_id,
messages_json) on all 1,319 GSM8K test items, plus a build record per part:

    data/requests/reproduction/<part>/<model_key>.parquet
    data/requests/reproduction/<part>/build_record.json

Parts: numerical (the authors' frames, 2,638 requests per model) and transport (the eight tranche configurations of
the frozen bank, 21,104 per model). Nothing here reads an outcome.

    .\\.venv\\Scripts\\python.exe scripts\\build_reproduction_requests.py --part all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import gsm8k, reproduction as rp  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=[*rp.PARTS, "all"], default="all")
    ap.add_argument("--models", nargs="*", help="model keys (default: every model in config/reproduction.yaml)")
    ap.add_argument("--config", help="override config/reproduction.yaml")
    ap.add_argument("--out", default=str(rp.REQUESTS_DIR))
    a = ap.parse_args(argv)
    cfg = rp.load_config(a.config)
    cfg_path = Path(a.config) if a.config else rp.CONFIG_PATH
    models = a.models or list(cfg["models"])
    unknown = sorted(set(models) - set(cfg["models"]))
    if unknown:
        raise SystemExit(f"not in the reproduction config: {unknown}")
    prompts = rp.load_json(cfg["prompts"])
    items = gsm8k.load_gsm8k(cfg["split"])
    parts = rp.PARTS if a.part == "all" else (a.part,)
    for part in parts:
        arms = rp.part_arms(part, cfg, prompts)
        d = Path(a.out) / part
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "reproduction", "part": part, "split": cfg["split"], "purpose": "reproduction of Cheng et al. (2025), "
            "never part of Dataset 2 estimation", "excluded_from_main": True,
            "protocol": "reproduction protocol", "config_sha256": rp.sha256_file(cfg_path),
            "prompts": {"path": cfg["prompts"], "sha256": rp.sha256_file(rp.ROOT / cfg["prompts"])},
            "items": {"n": int(len(items)), "canonical_sha256": gsm8k.canonical_sha256(items)},
            "configurations": list(arms), "arms": list(rp.ARMS), "models": {}}
        if part == "transport":
            for key in ("bank", "configurations_from"):
                p = cfg["parts"]["transport"][key]
                record[key] = {"path": p, "sha256": rp.sha256_file(rp.ROOT / p)}
        for m in models:
            df = rp.build_requests(m, arms, items)
            f = d / f"{m}.parquet"
            df.to_parquet(f, index=False)
            record["models"][m] = {"requests_file": f.name, "requests_sha256": rp.sha256_file(f), "n_requests": int(len(df)),
                                   "counts_by_arm": {k: int(v) for k, v in df["arm"].value_counts().sort_index().items()}}
            print(f"{part:<10} {m:<28} {len(df):>6} requests")
        (d / "build_record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {d / 'build_record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
