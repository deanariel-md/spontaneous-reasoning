r"""Request files of the constrained forced-direct arm on BIG-Bench Hard.

From the repository root, after scripts\build_forced_direct.py:
    .\.venv\Scripts\python.exe scripts\build_constrained_forced.py

For each model of the BIG-Bench Hard-28 panel, the forced arm's request file (data/requests/forced/bbh/<key>.parquet)
is copied with its messages unchanged, the request id gains the suffix "|gc", and the column guided_json holds the
item's constraint, a rule on the item's metadata:
- letter tasks: the option labels listed in the item's input (lines beginning "(X)"), each as "(X)" and "X";
- word tasks: the task's two words, each in its recorded spelling, lower case, capitalised and upper case;
- integer tasks: the regular expression of an optional minus sign and 1 to 7 digits.
Every allowed string ends with the closing "}". The prefill, cap and stop string are those of
config/decoding_forced_constrained.yaml.

Pilot files, data/requests/forced_constrained/pilot/<key>.parquet: the same requests of the first configuration of
manifests/sampler_bbh.json's discovery_config_ids and the first target item of each task (25 requests), so the pilot
meets every kind of constraint (letters, words, the integer pattern).

Outputs: data/requests/forced_constrained/bbh/<key>.parquet, the pilot files and
data/requests/forced_constrained/build_record.json (each source file with its sha256, each output with its sha256 and
request count, the constraint per task). Nothing here reads a model output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SRC = os.path.join(ROOT, "data", "requests", "forced", "bbh")
OUT = os.path.join(ROOT, "data", "requests", "forced_constrained")
ITEMS = os.path.join(ROOT, "data", "processed", "bbh", "items_target.parquet")
SUFFIX = "|gc"
CLOSE = "}"
INTEGER_REGEX = r"-?[0-9]{1,7}\}"
_OPTION = re.compile(r"^\(([A-Z])\)", re.M)


def option_letters(question: str) -> list[str]:
    """The option labels of an item, in order; refuses a list that is not A, B, C, ... from A."""
    got = _OPTION.findall(question)
    want = [chr(65 + i) for i in range(len(got))]
    if len(got) < 2 or got != want:
        raise ValueError(f"option labels {got} are not two or more consecutive letters from A")
    return got


def word_forms(words) -> list[str]:
    out: list[str] = []
    for w in words:
        for f in (w, w.lower(), w.capitalize(), w.upper()):
            if f not in out:
                out.append(f)
    return out


def constraint(answer_type: str, question: str) -> dict:
    """The guided-decoding constraint of one item (module doc)."""
    from src.bbh import AnswerType
    at = AnswerType.from_label(answer_type)
    if at.kind == "letter":
        return {"choice": [f"{s}{CLOSE}" for x in option_letters(question) for s in (f"({x})", x)]}
    if at.kind == "words":
        return {"choice": [f"{w}{CLOSE}" for w in word_forms(at.words)]}
    if at.kind == "integer":
        return {"regex": INTEGER_REGEX}
    raise ValueError(f"answer type {answer_type!r} has no constraint")


def pilot_rows(df: pd.DataFrame, items: pd.DataFrame, config_id: str) -> pd.DataFrame:
    """The pilot: the requests of config_id for the first target item of each task (module doc)."""
    first = items.groupby("task", sort=False)["item_id"].first().astype(str)
    p = df[(df["config_id"].astype(str) == config_id) & df["item_id"].astype(str).isin(set(first))]
    if len(p) != len(first):
        raise ValueError(f"pilot: {len(p)} requests for {len(first)} tasks")
    return p.reset_index(drop=True)


def file_sha(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    items = pd.read_parquet(ITEMS)
    cons = {iid: json.dumps(constraint(at, q), ensure_ascii=False)
            for iid, at, q in zip(items["item_id"].astype(str), items["answer_type"], items["question"])}
    fr = json.load(open(os.path.join(ROOT, "data", "requests", "forced", "build_record.json"), encoding="utf-8"))
    panel = json.load(open(os.path.join(ROOT, "results", "dataset2", "bbh.json"), encoding="utf-8"))["models"]
    os.makedirs(os.path.join(OUT, "bbh"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)
    pilot_cfg = str(json.load(open(os.path.join(ROOT, "manifests", "sampler_bbh.json"), encoding="utf-8"))
                    ["discovery_config_ids"][0])
    rec = {"label": "forced_constrained", "study_part": "constrained forced arm", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "arm": "forced_direct", "decoding": "config/decoding_forced_constrained.yaml",
           "decoding_sha256": file_sha(os.path.join(ROOT, "config", "decoding_forced_constrained.yaml")),
           "rule": ("the forced arm's BIG-Bench Hard requests, messages unchanged, request id suffix '|gc', and per item "
                    "the guided_json constraint of scripts/build_constrained_forced.constraint"),
           "items": {"path": os.path.relpath(ITEMS, ROOT).replace("\\", "/"), "sha256": file_sha(ITEMS)},
           "constraints_sha256": hashlib.sha256(json.dumps(cons, sort_keys=True).encode("utf-8")).hexdigest(),
           "pilot": {"config_id": pilot_cfg, "rule": "the first discovery configuration, the first target item of each task"},
           "constraint_kinds": {}, "tasks": {"bbh": {}}}
    for t, g in items.groupby("task", sort=False):
        c = json.loads(cons[str(g["item_id"].iloc[0])])
        rec["constraint_kinds"][t] = {"answer_type": g["answer_type"].iloc[0], "kind": next(iter(c)),
                                      "n_choices": sorted({len(json.loads(cons[i]).get("choice", [])) for i in g["item_id"].astype(str)})}
    for k in panel:
        fent = fr["tasks"]["bbh"][k]
        src = os.path.join(SRC, f"{k}.parquet")
        if file_sha(src) != fent["requests_sha256"]:
            raise SystemExit(f"{k}: the forced request file differs from its build record")
        df = pd.read_parquet(src)
        miss = sorted(set(df["item_id"].astype(str)) - set(cons))
        if miss:
            raise SystemExit(f"{k}: items without a constraint, e.g. {miss[:3]}")
        df["request_id"] = df["request_id"].astype(str) + SUFFIX
        df["guided_json"] = df["item_id"].astype(str).map(cons)
        dst = os.path.join(OUT, "bbh", f"{k}.parquet")
        df.to_parquet(dst, index=False)
        pdst = os.path.join(OUT, "pilot", f"{k}.parquet")
        pilot_rows(df, items, pilot_cfg).to_parquet(pdst, index=False)
        rec["tasks"]["bbh"][k] = {"source": fent["source"], "source_sha256": fent["source_sha256"],
                                  "forced_requests_file": f"data/requests/forced/bbh/{k}.parquet",
                                  "forced_requests_sha256": fent["requests_sha256"], "requests_file": f"{k}.parquet",
                                  "requests_sha256": file_sha(dst), "n": int(len(df)),
                                  "configs": int(df["config_id"].nunique()),
                                  "pilot_file": f"pilot/{k}.parquet", "pilot_sha256": file_sha(pdst)}
        print(f"{k:<40} {len(df):>6}")
    with open(os.path.join(OUT, "build_record.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"bbh: {len(panel)} models, {sum(v['n'] for v in rec['tasks']['bbh'].values())} requests")


if __name__ == "__main__":
    main()
