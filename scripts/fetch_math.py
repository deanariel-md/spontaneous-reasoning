r"""Fetch the pinned MATH files and write the manifests and item tables.

Downloads HuggingFaceH4/MATH-500 test.jsonl and the seven EleutherAI/hendrycks_math training parquet files at the
revisions pinned in src/math_task.py into data/raw/math/, skipping files already present (the Hugging Face CLI
login supplies any token; none is written anywhere). Then writes manifests/math_scorer.json (the scorer record),
manifests/math_items.json (the files' sha256, the numeric rule's counts, the seeded screening and calibration sets
and the sha256 of the item tables) and the item tables data/processed/math/items_{target,calibration,screening}.parquet.
The downloaded files are data, so run this with `python -I`.

Usage (PowerShell):
    .\.venv\Scripts\python.exe -I scripts\fetch_math.py
    .\.venv\Scripts\python.exe -I scripts\fetch_math.py --no-download     # from the cached files

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import math_task  # noqa: E402

EXIT_COMPLETE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-download", action="store_true", help="use the cached files only")
    ap.add_argument("--manifest", default=math_task.MANIFEST_PATH)
    ap.add_argument("--scorer-manifest", default=math_task.SCORER_MANIFEST_PATH)
    ap.add_argument("--tables-dir", default=math_task.TABLE_DIR)
    a = ap.parse_args(argv)
    rec = math_task.write_scorer_manifest(a.scorer_manifest)
    print(f"scorer manifest: {a.scorer_manifest} ({rec['scorer_version']}, rules sha256 {rec['rules_sha256'][:12]})")
    if not a.no_download:
        math_task.download()
    try:
        m = math_task.write_manifest(a.manifest, a.tables_dir)
    except math_task.MathError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"manifest": a.manifest, "counts": m["counts"], "sets": m["sets"], "tables": m["tables"],
                      "target_kept_by_level": m["target_kept_by_level"]}, indent=1))
    return EXIT_COMPLETE


if __name__ == "__main__":
    sys.exit(main())
