r"""Build the arm-wording run (config/arm_wording.yaml; src/arm_wording.py).

Subcommands, in the order the run needs them (repository root):

    .\.venv\Scripts\python.exe scripts\build_arm_wording.py configs
        draw the six configurations per task from the frozen samplers -> manifests/arm_wording_configs.json
    .\.venv\Scripts\python.exe scripts\build_arm_wording.py screen-requests
        the screening requests (current wording + first five candidates per arm, two configurations, 50 screening
        items per task) -> data/requests/arm_wording/screen/<key>.parquet and build_record.json. Refused until the
        Kojima strings are verified (config/arm_wording.yaml, cot.verified_on) and the paraphrase record and its review
        exist (scripts/paraphrase_d2.py --config config/arm_wording_paraphrase.yaml)
    .\.venv\Scripts\python.exe scripts\build_arm_wording.py screen
        read data/d2_out/arm_wording/screen/<key>, apply the format rules, check each screen run as the model's pilot,
        choose the four wordings per arm -> manifests/arm_wording_screen.json, manifests/arm_wording_wordings.json,
        results/arm_wording/pilot_check.md (and _clean.md)
    .\.venv\Scripts\python.exe scripts\build_arm_wording.py requests [--pilots]
        the collection requests -> data/requests/arm_wording/collection/<key>.parquet and build_record.json
        (with --pilots also 24-row pilot files in data/requests/arm_wording/pilot/)

--base DIR writes and reads every file of the run under DIR, outside the repository (dry runs). Dry runs may pass
--models, --items-dir (parquet files <task>_<set>.parquet with item_id, question, gold_answer), --bank-dir,
--allow-unverified, --record and --review; the repository's manifests are then never written.

Exit codes: 0 written, 2 refused, 1 any other error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import arm_wording as aw, d2_io, generation as gen  # noqa: E402

PARA_CFG = ROOT / "config" / "arm_wording_paraphrase.yaml"
SAMPLERS = {"gsm8k": ROOT / "manifests" / "sampler_gsm8k.json", "math": ROOT / "manifests" / "sampler_math.json"}
BANKS = {"gsm8k": ROOT / "manifests" / "gsm8k_bank.json", "math": ROOT / "manifests" / "math_bank.json"}


def in_repo(base: Path) -> bool:
    try:
        base.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def dry_guard(args):
    """Options that bypass a recorded input are for dry runs outside the repository."""
    used = [f for f, v in (("--allow-unverified", args.allow_unverified), ("--items-dir", args.items_dir),
                           ("--record", args.record), ("--review", args.review), ("--bank-dir", args.bank_dir))
            if v]
    if used and in_repo(aw.Paths(args.base).base):
        raise aw.ArmWordingError(f"{', '.join(used)} are for dry runs; pass --base outside the repository")


def panel(args) -> list[str]:
    if args.models:
        return list(args.models)
    from src import math_d2
    return math_d2.panel()


def bank(task: str, args) -> dict:
    p = Path(args.bank_dir) / f"{task}_bank.json" if args.bank_dir else BANKS[task]
    return json.loads(p.read_text(encoding="utf-8"))


def items(task: str, item_set: str, args) -> pd.DataFrame:
    """item_id (str), question, gold_answer of a recorded item set; --items-dir for dry runs."""
    if args.items_dir:
        df = pd.read_parquet(Path(args.items_dir) / f"{task}_{item_set}.parquet")
    elif task == "gsm8k":
        split = "test" if item_set == "target" else "train"
        df, _ = d2_io.load_items(split=split)
        if item_set == "screening":
            ids = set(json.loads((ROOT / "manifests" / "gsm8k_screening_items.json").read_text(encoding="utf-8"))["item_ids"])
            df = df[df["item_id"].isin(ids)]
    else:
        from src import math_d2
        df, _ = math_d2.load_items(item_set)
    df = df[["item_id", "question", "gold_answer"]].copy()
    df["item_id"] = df["item_id"].astype(str)
    return df.reset_index(drop=True)


def wording_inputs(cfg, args, paths):
    cot = aw.cot_candidates(cfg, require_verified=not args.allow_unverified)
    para = yaml.safe_load(PARA_CFG.read_text(encoding="utf-8"))
    record = Path(args.record) if args.record else ROOT / para["llm"]["record"]
    review = Path(args.review) if args.review else ROOT / para["llm"]["review"]
    for p, what in ((record, "paraphrase record"), (review, "paraphrase review")):
        if not p.is_file():
            raise aw.ArmWordingError(f"the {what} {d2_io.rel_path(p)} does not exist; run the paraphrase stage first")
    direct = aw.direct_candidates(cfg, record, review)
    return {"cot": cot, "dir": direct}, {"record": d2_io.rel_path(record), "record_sha256": aw.sha256_file(record),
                                         "review": d2_io.rel_path(review), "review_sha256": aw.sha256_file(review)}


def load_configs(paths) -> dict:
    if not paths.configs.is_file():
        raise aw.ArmWordingError(f"{paths.configs} does not exist; run `build_arm_wording.py configs` first")
    return json.loads(paths.configs.read_text(encoding="utf-8"))


def write_files(frames: dict, out_dir: Path, header: dict, decoding_of) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for k, df in frames.items():
        p = out_dir / f"{k}.parquet"
        status = d2_io.write_requests_file(df, p, overwrite=False)
        models[k] = {"requests_file": p.name, "requests_sha256": d2_io.sha256_path(p), "n_requests": int(len(df)),
                     "counts_by_arm": {a: int(n) for a, n in sorted(df["arm"].value_counts().items())},
                     "decoding": decoding_of(k), "status": status}
    rec = {**header, "models": models}
    aw.write_json(out_dir / "build_record.json", rec)
    return {k: {kk: v[kk] for kk in ("n_requests", "status", "decoding")} for k, v in models.items()}


def decoding_of_factory(cfg):
    from src import math_d2
    return lambda k: cfg["decoding"]["native_4k"] if k in math_d2.NATIVE_4K else cfg["decoding"]["default"]


def validate(frames: dict, cfg, models_yaml=None):
    for k, df in frames.items():
        dec = gen.load_decoding(str(ROOT / decoding_of_factory(cfg)(k)))
        d2_io.validate_rendered(df, k, dec)


# --------------------------------------------------------------------------------------------

def cmd_configs(args, cfg, paths):
    man = aw.configs_manifest(SAMPLERS, BANKS, cfg)
    status = aw.write_once(paths.configs, man, "the configuration draw")
    return {"manifest": str(paths.configs), "status": status,
            **{t: {"discovery": v["discovery"], "validation": v["validation"]} for t, v in man["tasks"].items()}}


def cmd_screen_requests(args, cfg, paths):
    dry_guard(args)
    man = load_configs(paths)
    cands, provenance = wording_inputs(cfg, args, paths)
    keys = panel(args)
    n_items = int(cfg["screen"]["n_items"])
    frames = {k: [] for k in keys}
    screen_items = {}
    for task in aw.TASKS:
        disc, val = aw.task_configs(man, task)
        confs = [disc[0], val[0]]
        pool = items(task, "screening", args)
        ids = aw.screen_item_ids(task, pool["item_id"], n_items)
        screen_items[task] = {"item_ids": ids, "configurations": confs}
        it = pool[pool["item_id"].isin(set(ids))].sort_values("item_id")
        b = bank(task, args)
        base = aw.render(keys[0], task, b, confs, cands, it, cfg["current"], prefix=aw.SCREEN_PREFIX)
        for k in keys:
            frames[k].append(base.assign(model_key=k))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    validate(frames, cfg)
    header = {"kind": "arm_wording_screen", "rule": "arm-wording screen", "candidates": cands,
              "wording_inputs": provenance, "configs_manifest_sha256": aw.sha256_file(paths.configs),
              "screen_items": screen_items, "excluded_from_main": True}
    res = write_files(frames, paths.requests / "screen", header, decoding_of_factory(cfg))
    return {"out": str(paths.requests / "screen"), "models": len(res),
            "requests_per_model": next(iter(res.values()))["n_requests"]}


def run_folders(root: Path, keys) -> dict:
    return {k: root / k for k in keys if (root / k).is_dir()}


def pilot_check(folders: dict, record: dict, cfg) -> list[dict]:
    """A model passes when one complete run wrote every request of its file, the log carries the run's decoding
    (prefill \\boxed{} on the forced arm, no stop string, the arm caps of its decoding file) and no prompt-render
    mismatch. No correctness is read."""
    rows = []
    for k, d in folders.items():
        want = record["models"].get(k)
        logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(d).glob("run_log-*.json"))]
        done = [x for x in logs if x.get("status") == "complete"]
        problems = []
        if want is None:
            problems.append("no request file for this model in the build record")
        if not done:
            problems.append("no complete run")
        if want and not any((x.get("inputs") or {}).get("requests_sha256") == want["requests_sha256"]
                            and (x.get("inputs") or {}).get("limit") is None for x in done):
            problems.append("no complete run of the whole recorded request file (without --limit)")
        dec = gen.load_decoding(str(ROOT / want["decoding"])) if want else None
        for x in done:
            dd = x.get("decoding") or {}
            if dec and dict(dd.get("prefill") or {}) != dict(dec.prefill):
                problems.append(f"prefill {dd.get('prefill')} differs from {want['decoding']}")
            if dec and list(dd.get("stop") or []) != list(dec.stop):
                problems.append(f"stop {dd.get('stop')} differs from {want['decoding']}")
            if dec and dict(dd.get("max_new_tokens") or {}) != dict(dec.max_new_tokens):
                problems.append(f"caps {dd.get('max_new_tokens')} differ from {want['decoding']}")
            if int(x.get("prompt_render_mismatches") or 0) != 0:
                problems.append(f"{x.get('prompt_render_mismatches')} prompt-render mismatches")
        gpu = sorted({" ".join(gen.gpu_type(x.get("gpu"))) for x in done})
        rows.append({"model_key": k, "passes": not problems, "problems": sorted(set(problems)), "gpu": gpu})
    missing = sorted(set(record["models"]) - set(folders))
    rows += [{"model_key": k, "passes": False, "problems": ["no output folder"], "gpu": []} for k in missing]
    return sorted(rows, key=lambda r: r["model_key"])


def write_pilot_md(rows, paths, what):
    n = sum(r["passes"] for r in rows)
    lines = [f"# Arm wording: {what} check", "",
             f"{n} of {len(rows)} models pass. A model passes when one complete run wrote every request of its file, "
             "the log carries the decoding of its file (prefill `\\boxed{` on the forced arm, no stop string, the arm "
             "caps) and no prompt-render mismatch. No correctness was read.", "",
             "| Model | Passes | GPU | Problems |", "|---|---|---|---|"]
    lines += [f"| {r['model_key']} | {'yes' if r['passes'] else 'no'} | {', '.join(r['gpu'])} | "
              f"{'; '.join(r['problems'])} |" for r in rows]
    text = "\n".join(lines) + "\n"
    paths.results.mkdir(parents=True, exist_ok=True)
    name = {"pilot": "pilot_check", "collection run": "collection_run_check"}.get(what, "screen_pilot_check")
    (paths.results / f"{name}.md").write_text(text, encoding="utf-8")
    (paths.results / f"{name}_clean.md").write_text(text, encoding="utf-8")
    return n


def cmd_screen(args, cfg, paths):
    from src import screening
    rec_path = paths.requests / "screen" / "build_record.json"
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    folders = run_folders(paths.outputs / "screen", record["models"])
    checks = pilot_check(folders, record, cfg)
    n_pass = write_pilot_md(checks, paths, "screen (pilot)")
    missing = [r["model_key"] for r in checks if not r["passes"]]
    if missing and not args.allow_incomplete:
        raise aw.ArmWordingError(f"{len(missing)} screen runs do not pass the pilot check ({missing[:5]}); see "
                                 f"{paths.results / 'screen_pilot_check.md'}")
    spec = None
    rows = {t: [] for t in aw.TASKS}
    for k, d in folders.items():
        out, info = d2_io.inspect_work_dir(d, allow_test_backend=args.allow_test_backend, string_item_ids=True)
        if info["runs"][0]["requests_sha256"] != record["models"][k]["requests_sha256"]:
            raise aw.ArmWordingError(f"{d}: the run read another request file")
        if spec is None:
            spec, _ = d2_io.load_registered_scorer(screening.SCORER_PATH)
        for t in aw.TASKS:
            rows[t].append(aw.screen_row_metrics(out, t, spec))
    rows = {t: pd.concat(v, ignore_index=True) for t, v in rows.items()}
    rules, prov = screening.load_rules(ROOT / cfg["screen"]["rules"])
    res = aw.apply_screen(rows, rules, cfg["screen"]["relative_truncation"])
    chosen = aw.choose_wordings({f: record["candidates"][f] for f in ("cot", "dir")}, res["failed"],
                                int(cfg["choose"]["n_new"]))
    summary = {t: screening.aggregate(rows[t], ["config_id", "arm"], [], True) for t in aw.TASKS}
    report = {"kind": "arm_wording_screen_report", "rule": "arm-wording screen",
              "rules": prov, "relative_truncation": cfg["screen"]["relative_truncation"],
              "screen_record_sha256": aw.sha256_file(rec_path), "candidates": record["candidates"],
              "pilot_check": {"passed": n_pass, "of": len(checks)}, "result": res,
              "metrics_by_wording": summary}
    screening.assert_no_outcome_keys(report)
    aw.write_once(paths.screen, report, "the screen report")
    words = {"kind": "arm_wording_wordings", "rule": "arm-wording screen and choice of the four",
             "chosen": chosen, "screen_report_sha256": aw.sha256_file(paths.screen),
             "configs_manifest_sha256": aw.sha256_file(paths.configs), "wording_inputs": record["wording_inputs"],
             "note": None if all(len(v) == 1 + int(cfg["choose"]["n_new"]) for v in chosen.values()) else
             "fewer than the full number of candidates passed; the run uses those that did"}
    aw.write_once(paths.wordings, words, "the choice of wordings")
    return {"screen": str(paths.screen), "wordings": str(paths.wordings), "failed": sorted(res["failed"]),
            "chosen": {f: [w["id"] for w in v] for f, v in chosen.items()}, "pilot_check_passed": n_pass}


def collection_requests(task_frames_for, keys):
    return {k: pd.concat(task_frames_for(k), ignore_index=True) for k in keys}


def cmd_requests(args, cfg, paths):
    dry_guard(args)
    man = load_configs(paths)
    if not paths.wordings.is_file():
        raise aw.ArmWordingError(f"{paths.wordings} does not exist; run `build_arm_wording.py screen` first")
    words = json.loads(paths.wordings.read_text(encoding="utf-8"))
    if words["configs_manifest_sha256"] != aw.sha256_file(paths.configs):
        raise aw.ArmWordingError("the wordings were chosen under another configuration draw")
    chosen = words["chosen"]
    keys = panel(args)
    base, checks = {}, []
    for task in aw.TASKS:
        disc, val = aw.task_configs(man, task)
        it = items(task, "target", args)
        df = aw.render(keys[0], task, bank(task, args), disc + val, chosen, it, cfg["current"])
        base[task] = df
        col = collection_file(task, keys[0]) if not args.bank_dir else None
        if col is not None:
            checks.append(aw.check_current_matches(df, pd.read_parquet(col), task))
    frames = {k: pd.concat([base[t].assign(model_key=k) for t in aw.TASKS], ignore_index=True) for k in keys}
    validate(frames, cfg)
    header = {"kind": "arm_wording_collection", "rule": "arm-wording collection", "chosen": chosen,
              "wordings_sha256": aw.sha256_file(paths.wordings), "configs_manifest_sha256": aw.sha256_file(paths.configs),
              "configurations": {t: {"discovery": aw.task_configs(man, t)[0], "validation": aw.task_configs(man, t)[1]}
                                 for t in aw.TASKS},
              "current_wording_matches_collection": checks, "excluded_from_main": False}
    res = write_files(frames, paths.requests / "collection", header, decoding_of_factory(cfg))
    out = {"out": str(paths.requests / "collection"), "models": len(res),
           "requests_per_model": next(iter(res.values()))["n_requests"], "current_wording_check": checks}
    if args.pilots:
        pil = {}
        for k, df in frames.items():
            first = df.drop_duplicates(["arm", "config_id"]).copy()
            first["task"] = first["config_id"].map(lambda c: aw.parse_config_id(c)[0])
            first["conf"] = first["config_id"].map(lambda c: aw.parse_config_id(c)[1])
            keep = []
            for t in aw.TASKS:
                c0 = aw.task_configs(man, t)[0][0]
                keep.append(first[(first["task"] == t) & (first["conf"] == c0)])
            pil[k] = pd.concat(keep, ignore_index=True)[list(gen.REQUEST_COLUMNS)]
        validate(pil, cfg)
        r = write_files(pil, paths.requests / "pilot", {**header, "kind": "arm_wording_pilot"}, decoding_of_factory(cfg))
        out["pilots"] = {"out": str(paths.requests / "pilot"), "rows_per_model": next(iter(r.values()))["n_requests"]}
    return out


def collection_file(task: str, key: str):
    if task == "math":
        p = ROOT / "data" / "requests" / "math" / f"{key}.parquet"
    else:
        p = ROOT / "data" / "requests" / "gsm8k" / f"{key}.parquet"
        if not p.exists():
            p = ROOT / "data" / "requests" / "gsm8k_ext" / f"collection_{key}" / f"{key}.parquet"
    return p if p.exists() else None


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("configs", "screen-requests", "screen", "requests"))
    ap.add_argument("--base", help="read and write the run's files under this folder (dry runs)")
    ap.add_argument("--config", help="override config/arm_wording.yaml")
    ap.add_argument("--models", nargs="+", help="model keys (default: the GSM8K-29 panel)")
    ap.add_argument("--items-dir", help="dry runs: <task>_<set>.parquet item tables")
    ap.add_argument("--bank-dir", help="dry runs: <task>_bank.json banks")
    ap.add_argument("--record", help="dry runs: a paraphrase record")
    ap.add_argument("--review", help="dry runs: a paraphrase review")
    ap.add_argument("--allow-unverified", action="store_true", help="dry runs: unverified Kojima strings")
    ap.add_argument("--allow-test-backend", action="store_true", help="dry runs: echo outputs")
    ap.add_argument("--allow-incomplete", action="store_true", help="screen with models whose run did not pass")
    ap.add_argument("--pilots", action="store_true", help="requests: also write 24-row pilot files")
    args = ap.parse_args(argv)
    cfg = aw.load_config(args.config)
    paths = aw.Paths(args.base)
    try:
        res = {"configs": cmd_configs, "screen-requests": cmd_screen_requests, "screen": cmd_screen,
               "requests": cmd_requests}[args.cmd](args, cfg, paths)
    except (aw.ArmWordingError, d2_io.D2IOError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
