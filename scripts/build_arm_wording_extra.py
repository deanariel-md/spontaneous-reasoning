r"""Build the default- and firm-clause run on the arm-wording run's six configurations
(src/arm_wording_extra.py; results/arm_wording_extra/design.md).

Subcommands, in order (repository root):

    .\.venv\Scripts\python.exe scripts\build_arm_wording_extra.py clauses
        write manifests/arm_wording_extra_clauses.json (the two texts and the rule; write-once)
    .\.venv\Scripts\python.exe scripts\build_arm_wording_extra.py requests
        the pilot (the arm-wording screen's items and two configurations; 800 requests per model) and the collection (six
        configurations, 41,208 requests per model) -> data/requests/arm_wording_extra/{pilot,collection}/<key>.parquet
        and build_record.json; checks that the current clause, rendered by the same code, reproduces the arm-wording run's
        requested-direct messages byte for byte on every configuration
    .\.venv\Scripts\python.exe scripts\build_arm_wording_extra.py pilot-check
        the run check of the pilot folders copied back (results/arm_wording_extra/pilot_check.md); no correctness
    .\.venv\Scripts\python.exe scripts\build_arm_wording_extra.py forecast
        after 29 of 29 pilots pass: manifests/predictions_arm_wording_extra.json and .sha256 from the pilot's token
        counts and the mechanism forecast (write-once). The collection jobs are released only after this exists.

--base DIR reads and writes under DIR (dry runs); --models, --items-dir and --bank-dir for dry runs, as
scripts/build_arm_wording.py. Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import arm_wording as aw, arm_wording_extra as ax, d2_io  # noqa: E402
import build_arm_wording as baw  # noqa: E402

BANKS = baw.BANKS


def _bank(task, args):
    p = Path(args.bank_dir) / f"{task}_bank.json" if args.bank_dir else BANKS[task]
    return json.loads(p.read_text(encoding="utf-8")), p


def _d24_record(paths):
    p = paths.d24_requests / "collection" / "build_record.json"
    if not p.is_file():
        raise ax.ExtraError(f"{p} does not exist (the arm-wording collection record gives the panel order and configurations)")
    return json.loads(p.read_text(encoding="utf-8")), p


def cmd_clauses(args, cfg, paths):
    banks = {t: aw.sha256_file(_bank(t, args)[1]) for t in aw.TASKS}
    man = ax.clauses_manifest(aw.sha256_file(paths.configs), banks)
    status = aw.write_once(paths.clauses, man, "the two clauses")
    return {"manifest": str(paths.clauses), "status": status, "firm": ax.FIRM}


def cmd_requests(args, cfg, paths):
    baw.dry_guard(args)
    if not paths.clauses.is_file():
        raise ax.ExtraError(f"{paths.clauses} does not exist; run `clauses` first")
    cl = json.loads(paths.clauses.read_text(encoding="utf-8"))
    if cl["configs_manifest_sha256"] != aw.sha256_file(paths.configs):
        raise ax.ExtraError("the clauses were recorded under another configuration draw")
    man = json.loads(paths.configs.read_text(encoding="utf-8"))
    d24, d24_path = _d24_record(paths)
    keys = list(args.models) if args.models else list(d24["models"])
    scr_path = paths.d24_requests / "screen" / "build_record.json"
    scr = json.loads(scr_path.read_text(encoding="utf-8"))
    pil, col, checks, counts = {}, {}, [], {}
    for task in aw.TASKS:
        b, bp = _bank(task, args)
        if aw.sha256_file(bp) != cl["banks_sha256"][task]:
            raise ax.ExtraError(f"{task}: the bank differs from the one the clauses were recorded with")
        disc, val = aw.task_configs(man, task)
        # the current clause through the same code reproduces the arm-wording run's (and the collection's) messages
        for cid in disc + val:
            want = b["configurations"][cid]["arms"]["direct_answer"]["messages"]
            if ax.clause_messages(b, cid, "dir0") != [{"role": m["role"], "content": m["content"]} for m in want]:
                raise ax.ExtraError(f"{task}/{cid}: the current clause does not reproduce the bank's messages")
            dm, fm = ax.clause_messages(b, cid, "dirdef"), ax.clause_messages(b, cid, "dirfirm")
            if any(ax.CURRENT in m["content"] for m in dm + fm):
                raise ax.ExtraError(f"{task}/{cid}: the current clause survives in a new message")
        checks.append({"task": task, "configurations": len(disc + val), "current_clause_reproduces_bank": True})
        si = scr["screen_items"][task]
        pool = baw.items(task, "screening", args)
        it = pool[pool["item_id"].isin(set(si["item_ids"]))].sort_values("item_id")
        if len(it) != len(si["item_ids"]):
            raise ax.ExtraError(f"{task}: {len(it)} of the arm-wording screen's {len(si['item_ids'])} items found")
        pil[task] = ax.render(keys[0], task, b, si["configurations"], it, prefix=ax.PILOT_PREFIX)
        tgt = baw.items(task, "target", args)
        col[task] = ax.render(keys[0], task, b, disc + val, tgt)
        d24_file = paths.d24_requests / "collection" / f"{keys[0]}.parquet"
        if d24_file.is_file() and not args.bank_dir:
            # the arm-wording run's requested-direct requests under the current clause, byte for byte from this code (request files,
            # not outputs)
            ref = pd.read_parquet(d24_file, columns=["config_id", "arm", "item_id", "messages_json"])
            ref = ref[ref["config_id"].str.startswith(f"{task}/") & ref["config_id"].str.endswith("/dir0")
                      & (ref["arm"] == aw.REQ)]
            ours = ax.render(keys[0], task, b, disc + val, tgt, clauses=("dir0",))
            ours = ours[ours["arm"] == aw.REQ]
            m = ours.merge(ref, on=["config_id", "arm", "item_id"], suffixes=("", "_d24"))
            bad = int((m["messages_json"] != m["messages_json_d24"]).sum())
            if bad or len(m) != len(ref) or len(m) != len(ours):
                raise ax.ExtraError(f"{task}: the current clause does not reproduce the arm-wording requests ({bad} differ, "
                                    f"{len(m)} matched of {len(ref)})")
            checks[-1]["d24_requested_dir0_identical"] = int(len(m))
        counts[task] = {"pilot": int(len(pil[task])), "collection": int(len(col[task])), "target_items": int(len(tgt))}
    frames_p = {k: pd.concat([pil[t].assign(model_key=k) for t in aw.TASKS], ignore_index=True) for k in keys}
    frames_c = {k: pd.concat([col[t].assign(model_key=k) for t in aw.TASKS], ignore_index=True) for k in keys}
    acfg = aw.load_config()
    baw.validate(frames_p, acfg)
    baw.validate(frames_c, acfg)
    header = {"kind": "arm_wording_extra_collection", "rule": "default and firm clauses on the arm-wording configurations",
              "clauses": cl["clauses"], "clauses_sha256": aw.sha256_file(paths.clauses),
              "configs_manifest_sha256": aw.sha256_file(paths.configs), "d24_record_sha256": aw.sha256_file(d24_path),
              "configurations": {t: {"discovery": aw.task_configs(man, t)[0], "validation": aw.task_configs(man, t)[1]}
                                 for t in aw.TASKS},
              "checks": checks, "counts_per_model": counts, "excluded_from_main": False}
    dec = baw.decoding_of_factory(acfg)
    rc = baw.write_files(frames_c, paths.requests / "collection", header, dec)
    hp = {**header, "kind": "arm_wording_extra_pilot", "screen_items": scr["screen_items"],
          "d24_screen_record_sha256": aw.sha256_file(scr_path), "excluded_from_main": True}
    rp = baw.write_files(frames_p, paths.requests / "pilot", hp, dec)
    return {"models": len(keys), "pilot_per_model": next(iter(rp.values()))["n_requests"],
            "collection_per_model": next(iter(rc.values()))["n_requests"], "counts": counts, "checks": checks}


def cmd_pilot_check(args, cfg, paths):
    rec = json.loads((paths.requests / "pilot" / "build_record.json").read_text(encoding="utf-8"))
    rows = baw.pilot_check(baw.run_folders(paths.outputs / "pilot", rec["models"]), rec, aw.load_config())
    n = baw.write_pilot_md(rows, paths, "pilot")
    return {"pilot_check": str(paths.results / "pilot_check.md"), "passed": n, "of": len(rows)}


def cmd_forecast(args, cfg, paths):
    rec = json.loads((paths.requests / "pilot" / "build_record.json").read_text(encoding="utf-8"))
    models = list(rec["models"])
    rows = baw.pilot_check(baw.run_folders(paths.outputs / "pilot", models), rec, aw.load_config())
    bad = [r["model_key"] for r in rows if not r["passes"]]
    if bad and not args.allow_incomplete:
        raise ax.ExtraError(f"{len(bad)} pilots do not pass the run check ({bad[:5]}); see pilot_check.md")
    mp = Path(args.mechanism) if args.mechanism else paths.mechanism
    mp_sha = mp.with_suffix(".sha256")
    from src import arm_wording_mechanism as awm
    try:
        mech_sha = awm.check_forecast(mp, mp_sha)
    except ValueError as exc:
        raise ax.ExtraError(str(exc))
    mech = json.loads(mp.read_text(encoding="utf-8"))
    W, N = ax.pilot_w(paths.outputs / "pilot", models)
    inp = ax.forecast_inputs(mech, models, W)
    source = {"pilot": d2_io.rel_path(paths.outputs / "pilot"),
              "pilot_requested_outputs_per_cell": sorted({int(v) for t in aw.TASKS for v in N[t].ravel()}),
              "pilot_record_sha256": aw.sha256_file(paths.requests / "pilot" / "build_record.json"),
              "mechanism_forecast": d2_io.rel_path(mp), "mechanism_forecast_file_sha256": mech_sha,
              "mechanism_forecast_content_sha256": mech["sha256"],
              "clauses_sha256": aw.sha256_file(paths.clauses)}
    r = ax.forecast_record(models, inp, source)
    r = ax.write_forecast(r, paths)
    return {"forecast": str(paths.forecast), "sha256": r["sha256"], "created": r["created"],
            "near0": {t: {d: len(v) for d, v in x["near0"].items()} for t, x in r["per_task"].items()},
            "w_median_by_clause": {t: x["w_median_by_clause"] for t, x in r["per_task"].items()},
            "predicted": {t: {k: v for k, v in x["predicted"].items() if k != "by_form"} for t, x in r["per_task"].items()}}


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("clauses", "requests", "pilot-check", "forecast"))
    ap.add_argument("--base")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--items-dir")
    ap.add_argument("--bank-dir")
    ap.add_argument("--mechanism", help="dry runs: another mechanism forecast file (its .sha256 beside it)")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    # build_arm_wording.dry_guard reads these attributes
    args.allow_unverified, args.record, args.review = False, None, None
    paths = ax.Paths(args.base)
    try:
        res = {"clauses": cmd_clauses, "requests": cmd_requests, "pilot-check": cmd_pilot_check,
               "forecast": cmd_forecast}[args.cmd](args, None, paths)
    except (ax.ExtraError, aw.ArmWordingError, d2_io.D2IOError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
