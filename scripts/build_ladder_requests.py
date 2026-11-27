r"""Size ladder: build the ladder's decodings and request files (src/ladder.py).

Writes
    config/decoding_ladder2_<task>.yaml     each arm's panel decoding, the raised cap and the cut cap (task_decoding_dict)
    config/decoding_ladder2_pilot.yaml      one decoding with an arm <task>.<arm> per task and arm
    config/decoding_ladder2_equiv.yaml      the raised arms at the panel's caps, for the equivalence run
    data/requests/ladder/<task>/<key>.parquet    the panel's rows of every arm, model_key replaced
    data/requests/ladder/pilot/<key>.parquet     16 items of one configuration per task and arm
    data/requests/ladder/build_record.json       sources, sha256 of every file, the equality check against a second
                                                 panel model's files, the time estimate and the jobs (parts)
The first build's record (dropped models) is kept as data/requests/ladder/build_record_v1.json. The -dk keys get no
files: they answer their base key's file (requests_model_key in config/models_ladder.yaml).

Usage:  python scripts/build_ladder_requests.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import generation as gen  # noqa: E402
from src import ladder as L  # noqa: E402


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


HEAD = {
    "task": "# Size ladder, {task}: every arm of the panel's design on this benchmark in one run.\n"
            "# Each arm keeps its panel collection's decoding (src/ladder.py, MAIN_DECODING); the requested, chain-of-thought\n"
            "# and five-shot requested arms run at 7,000 new tokens, with cut_caps the panel's cap at which every output is\n"
            "# also kept cut at the token level. Written by scripts/build_ladder_requests.py.\n",
    "pilot": "# Size ladder pilot: one run per model holding all four tasks. Arm <task>.<arm> has the parameters of <arm> in\n"
             "# that task's ladder decoding (config/decoding_ladder2_<task>.yaml). Written by scripts/build_ladder_requests.py.\n",
    "equiv": "# Size ladder equivalence run: the pilot's raised arms at the panel's caps, without a\n"
             "# cut, so that each output can be compared with the pilot's cut copy. Written by scripts/build_ladder_requests.py.\n",
}


def write_yaml(path, head, d) -> None:
    (L.ROOT / path).write_text(head + yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_decodings() -> None:
    for t in L.TASKS:
        write_yaml(L.DECODING[t], HEAD["task"].format(task=t), L.task_decoding_dict(t))
    write_yaml(L.PILOT_DECODING, HEAD["pilot"], L.pilot_decoding_dict())
    write_yaml(L.EQUIV_DECODING, HEAD["equiv"], L.pilot_decoding_dict(equiv=True))
    pil = gen.load_decoding(L.ROOT / L.PILOT_DECODING)
    for t in L.TASKS:
        td = gen.load_decoding(L.ROOT / L.DECODING[t])
        for arm in L.TASK_ARMS[t]:
            assert pil.params_for_arm(f"{t}.{arm}") == td.params_for_arm(arm), (t, arm)
            assert pil.cut_caps.get(f"{t}.{arm}") == td.cut_caps.get(arm), (t, arm)
    gen.load_decoding(L.ROOT / L.EQUIV_DECODING)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/requests/ladder")
    ap.add_argument("--models", nargs="*", default=list(L.MODELS))
    a = ap.parse_args(argv)
    out = L.ROOT / a.out
    reg = gen.load_registry(L.ROOT / L.REGISTRY)
    for k in a.models:
        gen.parse_model_spec(k, reg)
        gen.parse_model_spec(k + L.FALLBACK_SUFFIX, reg)
    write_decodings()
    dec = gen.load_decoding(L.ROOT / L.PILOT_DECODING)

    old = out / "build_record.json"
    if old.exists() and not (out / "build_record_v1.json").exists():
        shutil.copy2(old, out / "build_record_v1.json")
    proxy = L.proxy_lengths()
    record = {"label": "size_ladder_requests_v2", "study_part": "size ladder", "requests": ["ladder 1", "ladder 2", "ladder 3"],
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source_key": L.SOURCE_KEY, "check_key": L.CHECK_KEY,
              "registry": L.REGISTRY, "registry_sha256": sha(L.ROOT / L.REGISTRY),
              "pilot_decoding": L.PILOT_DECODING, "pilot_decoding_sha256": sha(L.ROOT / L.PILOT_DECODING),
              "equiv_decoding": L.EQUIV_DECODING, "equiv_decoding_sha256": sha(L.ROOT / L.EQUIV_DECODING),
              "decoding": {t: {"path": L.DECODING[t], "sha256": sha(L.ROOT / L.DECODING[t])} for t in L.TASKS},
              "rule": "the panel's request rows of each task and arm (requested, chain of thought, forced; five-shot "
                      "forced on all four benchmarks; five-shot requested on GSM8K and BIG-Bench Hard), model_key "
                      "replaced, five-shot arms renamed fewshot_forced and fewshot_requested; pilot = per task and arm "
                      f"the first direct_answer row's configuration (or the arm's own first), its first {L.PILOT_ITEMS} "
                      "items, arms relabelled <task>.<arm>",
              "dropped": L.DROPPED, "tasks": {}, "models": {}}
    base = {}
    for t in L.TASKS:
        df, rec = L.task_requests(t, "__ladder__", check_key=L.CHECK_KEY)
        base[t] = df
        record["tasks"][t] = rec
        print(f"{t}: {rec['rows']} rows {rec['by_arm']}, {rec['configs']} configurations, {rec['items']} items")
    for k in a.models:
        mrec = {"files": {}, "estimate": L.estimate(k, proxy), "jobs": L.jobs(k, proxy)}
        pil = []
        for t in L.TASKS:
            df = base[t].copy()
            df["model_key"] = k
            gen.validate_requests(df, k, gen.load_decoding(L.ROOT / L.DECODING[t]))
            p = out / t / f"{k}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            mrec["files"][t] = {"path": str(p.relative_to(L.ROOT)).replace("\\", "/"), "sha256": sha(p),
                                "rows": int(len(df)), "parts": L.parts(k, t, proxy)}
            pil.append(L.pilot_rows(t, df))
        pdf = pd.concat(pil, ignore_index=True)
        pdf["model_key"] = k
        gen.validate_requests(pdf, k, dec)
        p = out / "pilot" / f"{k}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(p, index=False)
        mrec["files"]["pilot"] = {"path": str(p.relative_to(L.ROOT)).replace("\\", "/"), "sha256": sha(p),
                                  "rows": int(len(pdf)),
                                  "by_arm": {x: int(v) for x, v in pdf["arm"].value_counts().sort_index().items()}}
        record["models"][k] = mrec
        print(f"{k}: pilot {len(pdf)} rows; estimate {mrec['estimate']['wall_hours']} h wall, "
              f"{mrec['estimate']['gpu_hours']} GPU-h; jobs {len(mrec['jobs'])}")
    L.write_json(out / "build_record.json", record)
    print(f"written: {out / 'build_record.json'}")
    cat = write_catalog(out / "ladder_catalog.tsv", record)
    print(f"written: {cat} ({sum(1 for _ in open(cat, encoding='utf-8'))} jobs)")
    return 0


# Slurm options of a job, by the model's pool in src/ladder.py: templates with {cards} and {mem}. Set them for your
# cluster (partition, QOS, GPU type) through LADDER_SLURM_PUBLIC and LADDER_SLURM_RESERVED; the defaults ask only for
# the number of GPUs and the memory.
SLURM_FLAGS = {"public": os.environ.get("LADDER_SLURM_PUBLIC", "--gpus={cards} --mem={mem}"),
               "reserved": os.environ.get("LADDER_SLURM_RESERVED", "--gpus={cards} --mem={mem}")}


def card_flags(key: str) -> str:
    m = L.MODELS[key]
    return SLURM_FLAGS["reserved" if m["pool"] == "reserved" else "public"].format(cards=m["cards"], mem=m["mem"])


def write_catalog(path, record) -> str:
    """$S/ladder_catalog.tsv, read by the job-sharing scripts: one row per full job, in offer order
    (the one-card models' jobs interleaved in proportion to their numbers, so the models advance at one pace and
    finish together; ties go to the longer model):
    key, short, pool, task, part, card flags, time limit."""
    rows = []
    for k, m in record["models"].items():
        for i, j in enumerate(m["jobs"]):
            rows.append((L.MODELS[k]["pool"] != "public", (i + 0.5) / len(m["jobs"]), -m["estimate"]["wall_hours"], k, j))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for _, _, _, k, j in rows:
            fh.write("\t".join([k, L.MODELS[k]["short"], L.MODELS[k]["pool"], j["task"], j["part"], card_flags(k),
                                "0-12:00:00"]) + "\n")
    return str(path)


if __name__ == "__main__":
    sys.exit(main())
