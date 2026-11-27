r"""Fetch the pinned BIG-Bench Hard files and write the manifests and item tables.

Downloads bbh/<task>.json and cot-prompts/<task>.txt of the 25 tasks from
raw.githubusercontent.com at commit 9ee07bd4 (src/bbh.py) into data/raw/bbh/<commit>/, skipping files
already present. Then writes manifests/bbh_scorer.json (the scorer record; it needs no data),
manifests/bbh_items.json (per-file bytes, sha256 and git blob id, each task's answer type and
description, the malformed targets, the seeded target, calibration and screening split, and the
sha256 of the item tables), and the item tables data/processed/bbh/items_{target,calibration,screening}.parquet,
rebuilt from the manifest and checked against its sha256. The downloaded files are data, so run this
with `python -I`.

Usage (PowerShell):
    .\.venv\Scripts\python.exe -I scripts\fetch_bbh.py
    .\.venv\Scripts\python.exe -I scripts\fetch_bbh.py --no-download     # from the cached files

Exit codes: 0 written, 2 refused (a file or task does not meet the recorded rules), 1 any other error.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import bbh  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-download", action="store_true", help="use the cached files only")
    ap.add_argument("--manifest", default=bbh.MANIFEST_PATH)
    ap.add_argument("--scorer-manifest", default=bbh.SCORER_MANIFEST_PATH)
    ap.add_argument("--tables-dir", default=bbh.TABLE_DIR)
    a = ap.parse_args(argv)
    rec = bbh.write_scorer_manifest(a.scorer_manifest)
    print(f"scorer manifest: {a.scorer_manifest} ({rec['scorer_version']}, rules sha256 {rec['rules_sha256'][:12]})")
    if not a.no_download:
        bbh.download()
    missing = [t for t in bbh.TASKS if not bbh.is_cached(t)]
    if missing:
        print(f"REFUSED: files of {len(missing)} tasks are absent, e.g. {missing[:3]}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        m = bbh.write_manifest(a.manifest)
        written = bbh.write_item_tables(bbh.item_tables(m), a.tables_dir)
    except bbh.BBHError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    mismatch = [n for n, w in written.items() if w["sha256"] != m["tables"][n]["sha256"]]
    if mismatch:
        print(f"ERROR: the written tables {mismatch} differ from the manifest's sha256", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps({"manifest": a.manifest, "revision": m["revision"], "sets": m["sets"],
                      "malformed_targets": m["malformed_targets"], "tables": written,
                      "answer_types": {t: r["answer_type"] for t, r in m["per_task"].items()}}, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
