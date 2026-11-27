r"""HotpotQA manifests and item tables. Downloads nothing.

Reads the pinned raw file (data/raw/hotpotqa/, fetched once; src/hotpot_task.py) and the token
lengths (scripts/hotpot_lengths.py), checks both hashes, and writes:
  manifests/hotpot_source.json       the source, its revision, size and sha256, and why the fallback was used
  manifests/hotpot_task_choice.json  the written rule that chose the task, with the citation counts it compares
  manifests/hotpot_scorer.json       the scorer record (hotpot-boxed/1.0.0)
  manifests/hotpot_items.json        the item sets (target in blocks A and B, screening, demonstration pool), their
                                     counts and sha256, the forced cap, the length rule and its tokenizers
  data/processed/hotpot/items_{target,screening,demo_pool}.parquet
Nothing here reads a model output. Refuses to overwrite an items manifest whose sets would change.

Usage (PowerShell, repository root):
    .\.venv\Scripts\python.exe -I scripts\fetch_hotpot.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import hashlib  # noqa: E402

import pandas as pd  # noqa: E402

from src import hotpot_task as ht  # noqa: E402

SOURCE = os.path.join(ROOT, "manifests", "hotpot_source.json")
CHOICE = os.path.join(ROOT, "manifests", "hotpot_task_choice.json")

SOURCE_REC = {
    "task": "hotpot", "dataset": "HotpotQA (Yang et al. 2018, EMNLP), distractor setting, development set",
    "plan": ht.PLAN, "license": ht.LICENSE,
    "official_file": {"name": "hotpot_dev_distractor_v1.json", "url": ht.OFFICIAL_URL,
                      "status": "not fetched: the host curtis.ml.cmu.edu did not resolve (DNS) from the machine used"},
    "used": {"repo": ht.REPO, "repo_type": "dataset", "config": "distractor", "split": "validation",
             "revision": ht.REVISION, "file": ht.FILE, "bytes": ht.FILE_BYTES, "sha256": ht.FILE_SHA256,
             "lfs_oid": ht.FILE_SHA256, "local_path": "data/raw/hotpotqa/validation-00000-of-00001.parquet",
             "url": f"https://huggingface.co/datasets/{ht.REPO}/resolve/{ht.REVISION}/{ht.FILE}",
             "downloaded": "2026-11-17", "approval": "the official file, falling back to the hotpotqa/hotpot_qa distractor validation split"},
    "content": "7,405 items, all of level hard (the development set has only hard items); 5,918 bridge and 1,487 "
               "comparison; 458 answers are yes or no; 7,345 items have 10 paragraphs and 60 fewer",
}

CHOICE_REC = {
    "task": "hotpot", "written": "2026-11-17", "plan": ht.PLAN,
    "rule": "The third task is short-answer multi-hop question answering over given passages, the task type of the one "
            "non-math short-answer dataset Sprague et al. ran themselves (MuSiQue; arXiv 2409.12183v3, Table 4). Among "
            "the public datasets of this type with public gold answers and an official exact-match and token-F1 scorer "
            "(HotpotQA distractor, 2WikiMultihopQA, MuSiQue-Ans), the study takes the most cited one.",
    "correction": "HotpotQA is not one of Sprague et al.'s datasets. No model output on any candidate existed when "
                  "this rule was written.",
    "sprague_free_form": {"GSM8K": "math", "GSM8K-Hard": "math", "MATH": "math",
                          "MuSiQue": "soft reasoning, short answer, zero-shot only, scored by a GPT-4o equivalence judge",
                          "BiGGen Bench": "free response, scored 1-5 by an LLM judge"},
    "citation_counts": {
        "source": "Semantic Scholar Graph API, /graph/v1/paper/arXiv:<id>?fields=citationCount, read 2026-11-18",
        "HotpotQA": {"arxiv": "1809.09600", "venue": "EMNLP 2018", "citations": 6143},
        "2WikiMultihopQA": {"arxiv": "2011.01060", "venue": "COLING 2020", "citations": 1962},
        "MuSiQue": {"arxiv": "2108.00573", "venue": "TACL 2022", "citations": 1415}},
    "chosen": "HotpotQA, distractor setting, development set",
}


def main() -> int:
    raw = ht.load_raw()
    lj = json.load(open(os.path.join(ht.TABLE_DIR, "lengths.json"), encoding="utf-8"))
    if ht.file_sha256(ht.LENGTHS_PATH) != lj["sha256"] or lj["raw_sha256"] != ht.FILE_SHA256:
        print("REFUSED: data/processed/hotpot/lengths.parquet differs from lengths.json; rerun scripts/hotpot_lengths.py",
              file=sys.stderr)
        return 2
    lengths = pd.read_parquet(ht.LENGTHS_PATH)
    sets = ht.split_items(raw, lengths)
    tabs = ht.tables(raw, sets)
    cap = ht.forced_cap(sets["target"], lengths)
    L = lengths.set_index("item_id")
    t_len = L.loc[sets["target"], "item_tokens"]
    d_len = L.loc[sets["demo_pool"], "demo_tokens"]
    manifest = {
        "task": "hotpot", "plan": ht.PLAN, "source": {"repo": ht.REPO, "revision": ht.REVISION, "file": ht.FILE,
                                                      "sha256": ht.FILE_SHA256},
        "item_text": "src/hotpot_task.py module doc: passage block (paragraphs in file order, 'Title: <title>', newline, "
                     "sentences joined), blank line, 'Question: <question>'",
        "rules": {"eligible": f"normalised gold non-empty and not yes/no; item text at most {ht.MAX_ITEM_TOKENS} tokens "
                              "under each tokenizer of " + ", ".join(ht.LENGTH_MODELS),
                  "target": f"{ht.N_TARGET} eligible ids, sorted, drawn without replacement with "
                            f"numpy.random.default_rng(first 8 bytes of sha256('{ht.SEED_TARGET}')), in the order "
                            f"drawn; block A the first {ht.N_TARGET // 2}, block B the rest",
                  "pool": "eligible, outside the target, sharing no supporting title with any target item, sorted",
                  "screening": f"{ht.N_SCREENING} from the pool, seed sha256('{ht.SEED_SCREENING}'), in the order drawn",
                  "demo_pool": f"the rest of the pool with demonstration text and gold at most {ht.MAX_DEMO_TOKENS} "
                               f"tokens (same tokenizers) and a gold answer free of {ht.DEMO_EXCLUDED_CHARS}, sorted",
                  "forced_cap": "smallest multiple of 16 holding the longest target gold as '\\text{<gold>}}' under "
                                "every panel tokenizer"},
        "counts": sets["counts"],
        "target_by_type": {k: int(v) for k, v in tabs["target"]["qtype"].value_counts().sort_index().items()},
        "target_by_block_and_type": {b: {k: int(v) for k, v in g["qtype"].value_counts().sort_index().items()}
                                     for b, g in tabs["target"].groupby("block")},
        "target_item_tokens": {"median": float(t_len.median()), "max": int(t_len.max())},
        "demo_pool_tokens": {"median": float(d_len.median()), "max": int(d_len.max())},
        "forced_cap": cap,
        "lengths": {"path": lj["path"], "sha256": lj["sha256"], "tokenizers": lj["tokenizers"],
                    "transformers": lj["transformers"]},
        "sets": {k: {"n_items": int(len(v)), "canonical_sha256": ht.canonical_sha256(v),
                     "ids_sha256": hashlib.sha256(json.dumps(list(v["item_id"])).encode()).hexdigest()}
                 for k, v in tabs.items()},
        "tables": {k: {"path": f"data/processed/hotpot/items_{k}.parquet", "n_rows": int(len(t)),
                       "sha256": hashlib.sha256(ht.table_bytes(t)).hexdigest()} for k, t in tabs.items()},
        "tables_definition": "one parquet per set, columns " + ", ".join(ht.TABLE_COLUMNS) + "; question is the item "
                             "variable (the demonstration text for demo_pool); hotpot_question the dataset's question",
    }
    if os.path.exists(ht.MANIFEST_PATH):
        old = json.load(open(ht.MANIFEST_PATH, encoding="utf-8"))
        if old.get("sets") != manifest["sets"]:
            print("REFUSED: manifests/hotpot_items.json exists with other item sets", file=sys.stderr)
            return 2
    ht.write_json(SOURCE_REC, SOURCE)
    ht.write_json(CHOICE_REC, CHOICE)
    rec = ht.write_scorer_manifest()
    os.makedirs(ht.TABLE_DIR, exist_ok=True)
    for k, t in tabs.items():
        p = os.path.join(ht.TABLE_DIR, f"items_{k}.parquet")
        with open(p, "wb") as fh:
            fh.write(ht.table_bytes(t))
    ht.write_json(manifest, ht.MANIFEST_PATH)
    print(json.dumps({"counts": manifest["counts"], "target_by_type": manifest["target_by_type"],
                      "forced_cap": cap, "target_item_tokens": manifest["target_item_tokens"],
                      "demo_pool_tokens": manifest["demo_pool_tokens"], "scorer": rec["scorer_version"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
