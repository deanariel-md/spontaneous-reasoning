r"""Dataset 2 throughput tranche: measurement and cost projection.

Input is the `throughput/` folder that `src/generation.py` writes in throughput mode, one folder per
model: parquet shards plus the run log (`throughput_log-<run_id>.json`; `run_log-<run_id>.json` is
read too, and its `mode` field is checked the same way). Output is a measurement per model and arm
and a projection of generations, tokens, GPU-hours and cost for 8, 12, 16 and 24 discovery and
validation units per arm.

Refusals (TrancheError, nothing is written): a folder whose run log is not marked mode "throughput"
and excluded_from_main, a folder with shards that are not marked throughput, a folder without
a single complete run, a folder holding two complete runs, the same model twice, and a run made with
a test backend (unless allowed). Tranche data are excluded from the main estimation, and the
main shards must never be mixed in.

Noise-tranche folders. The tranche of every panel model is an ordinary (full-mode) run on
training items, whose request file was built with `scripts/build_requests.py --tranche`. Such a folder is
read only when every run log names a request file listed in the tranche build record
(`<requests dir>/tranche/build_record.json`, kind "tranche", split "train") for the same model, and one run
completed without --limit. A preempted run leaves several run logs in the folder: every shard is measured,
and the throughput is the output tokens generated over the generation time, summed over the runs whose logs
record both. Main-run (test-split) folders are never read, so tranche data stay out of the main
estimation.

Missing run-log fields are reported as missing (null plus an entry in `missing_fields`). A projection
that needs a missing field is null. Nothing is imputed.

Cap-measurement runs. A run made with `--measure-cap` (src/generation.py) records `cap_override` in its
log with the configured and the effective caps. Its raised caps are not the configured caps, so the
projection uses each overridden arm's configured cap: every unique output length L enters the token
statistics as min(L, configured cap), and the share truncated is the share with L above the configured
cap. The report names the override (`cap_override`) and warns. The tokens-per-second rate is the run's own
(it generated the longer outputs). An override that lowers a cap below the configured one cannot be
corrected (the cut-off tail was never generated), so the projection is a lower bound and the report says so.

Units. One unit is one prompt configuration used in both arms. With K discovery and K validation units
per arm there are 2K units, drawn independently with replacement from the bank. Identical
configurations are generated once (greedy decoding, cache), so the expected number of distinct
configurations is E = sum_c [1 - (1 - p_c)^(2K)].

The module imports without vLLM or torch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
BUDGET_YAML = ROOT / "config" / "budget.yaml"
BANK_JSON = ROOT / "manifests" / "gsm8k_bank.json"
REQUESTS_DIR = ROOT / "data" / "requests" / "gsm8k"
OUT_DIR = ROOT / "results" / "tranche"

SCHEMA_VERSION = "1"
K_GRID = (8, 12, 16, 24)                       # allocations compared before the cap is fixed
ARMS = ("direct_answer", "zero_shot_cot")      # config/tasks.yaml dataset2.intervention (tested)
DEFAULT_MODELS = 20                            # sizing example
DEFAULT_ITEMS = 1319                           # GSM8K test
PROB_SUM_TOL = 1e-6                            # |sum(p) - 1| allowed in the bank file
LOG_GLOBS = ("throughput_log-*.json", "run_log-*.json")
CAP_KEYS = ("gpu_hours", "money_usd")
TEST_BACKENDS = ("echo",)


class TrancheError(Exception):
    """An input does not qualify for the measurement or projection. Nothing is written."""


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrancheError(f"{path} cannot be read as JSON ({exc})") from exc


def _dig(d: Any, *path: str) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _num(v: Any) -> float | None:
    """A finite real number, or None. Booleans and strings do not count."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _pos_int(v: Any) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return int(v)


# --------------------------------------------------------------------------------------------
# Counts and the bank
# --------------------------------------------------------------------------------------------

def generations_before_caching(models: int, items: int, k: int) -> int:
    """M x N_items x 2 arms x 2K units (12/12 gives 1,266,240 and 24/24 gives 2,532,480)."""
    return int(models) * int(items) * 2 * (2 * int(k))


def expected_distinct(probabilities: Sequence[float], draws: int) -> float:
    """Expected number of distinct configurations among `draws` iid draws: sum_c 1 - (1 - p_c)^draws."""
    if draws < 0:
        raise ValueError("draws must be nonnegative")
    terms = []
    for p in probabilities:
        if p <= 0.0:
            terms.append(0.0)
        elif p >= 1.0:
            terms.append(1.0 if draws > 0 else 0.0)
        else:
            terms.append(-math.expm1(draws * math.log1p(-p)))
    return math.fsum(terms)


@dataclasses.dataclass(frozen=True)
class Bank:
    kind: str                                        # "file", "uniform" or "none"
    probabilities: tuple[float, ...] | None
    source: str
    path: str | None = None
    sha256: str | None = None
    note: str | None = None

    @property
    def n_configs(self) -> int | None:
        return None if self.probabilities is None else len(self.probabilities)

    def expected_distinct(self, draws: int) -> float:
        """Without a bank every draw is taken as distinct, so E = draws."""
        if self.probabilities is None:
            return float(draws)
        return expected_distinct(self.probabilities, draws)

    def to_log(self) -> dict:
        return {"kind": self.kind, "source": self.source, "n_configs": self.n_configs,
                "path": self.path, "sha256": self.sha256, "note": self.note}


def load_bank_probabilities(path: str | os.PathLike) -> dict[str, float]:
    """`sampling_probabilities` (config_id -> probability) from the frozen bank manifest."""
    p = Path(path)
    raw = _read_json(p)
    probs = raw.get("sampling_probabilities") if isinstance(raw, Mapping) else None
    if not isinstance(probs, Mapping) or not probs:
        raise TrancheError(f"{p}: key sampling_probabilities must be a non-empty mapping of config_id to probability")
    out: dict[str, float] = {}
    for cid, v in probs.items():
        f = _num(v)
        if f is None or f < 0.0 or f > 1.0:
            raise TrancheError(f"{p}: probability for {cid!r} is {v!r}; each must be a number in [0, 1]")
        out[str(cid)] = f
    total = math.fsum(out.values())
    if abs(total - 1.0) > PROB_SUM_TOL:
        raise TrancheError(f"{p}: sampling_probabilities sum to {total!r}, not 1 (tolerance {PROB_SUM_TOL}); "
                           "the file is not rescaled")
    return out


def resolve_bank(bank_path: str | os.PathLike | None = None, bank_size: int | None = None) -> Bank:
    """The bank file when it exists, else a uniform bank of `bank_size`, else no duplication model."""
    path = Path(bank_path) if bank_path else BANK_JSON
    if bank_size is not None and (isinstance(bank_size, bool) or not isinstance(bank_size, int) or bank_size < 1):
        raise TrancheError("--bank-size must be a positive integer")
    if bank_path and not path.is_file():
        raise TrancheError(f"bank file {path} does not exist")
    if path.is_file():
        probs = load_bank_probabilities(path)
        note = None
        if bank_size is not None:
            note = f"--bank-size {bank_size} ignored: the bank file {path.name} exists and takes precedence"
        return Bank("file", tuple(probs.values()), f"the sampling probabilities in {path.name} ({len(probs)} configurations)",
                    str(path), _sha256_file(path), note)
    if bank_size is not None:
        return Bank("uniform", tuple([1.0 / bank_size] * bank_size),
                    f"a uniform bank of {bank_size} configurations (--bank-size)")
    return Bank("none", None, "no bank file and no --bank-size, so every draw is taken as distinct (E = 2K)")


def load_caps(path: str | os.PathLike | None = None) -> dict:
    """cap.gpu_hours and cap.money_usd from config/budget.yaml (None when not set)."""
    p = Path(path) if path else BUDGET_YAML
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TrancheError(f"cannot read {p}: {exc}") from exc
    cap = raw.get("cap") or {}
    out: dict[str, Any] = {}
    for key in CAP_KEYS:
        v = cap.get(key)
        if v is None:
            out[key] = None
            continue
        f = _num(v)
        if f is None or f < 0:
            raise TrancheError(f"{p}: cap.{key} must be null or a nonnegative number, got {v!r}")
        out[key] = f
    out["source"] = str(p)
    return out


def compare_cap(gpu_hours: float | None, cost_usd: float | None, caps: Mapping) -> tuple[str, dict]:
    """within / above / cap not set / not evaluable. A value equal to the cap is within it."""
    detail = {}
    for key, value in (("gpu_hours", gpu_hours), ("money_usd", cost_usd)):
        cap = caps.get(key)
        if cap is None:
            detail[key] = "cap not set"
        elif value is None:
            detail[key] = "not evaluable"
        elif value > cap:
            detail[key] = "above"
        else:
            detail[key] = "within"
    states = set(detail.values())
    if states == {"cap not set"}:
        summary = "cap not set"
    elif "above" in states:
        summary = "above"
    elif "not evaluable" in states:
        summary = "not evaluable"
    else:
        summary = "within"
    return summary, detail


# --------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------

def _stats(values: Any) -> dict:
    a = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float) if values is not None \
        else np.array([])
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    mx = float(a.max())
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(mx) if mx.is_integer() else mx}


def _arm_block(df: pd.DataFrame, cap: int | None, have: set[str], run_cap: int | None = None) -> dict:
    """Statistics over the unique generations (cached == False) of one arm, as the runner's own report.

    `cap` is the cap the statistics refer to (the configured cap) and `run_cap` the cap the run used. When
    the run's cap was raised above the configured one (--measure-cap), each output length enters the
    statistics as min(L, cap) and the share truncated is the share with L above `cap`; the lengths as
    generated are kept under output_tokens_as_generated."""
    capped = cap is not None and run_cap is not None and run_cap > cap
    block: dict[str, Any] = {"cap": cap, "run_cap": run_cap, "cap_override_applied": bool(capped),
                             "requests": int(len(df)), "unique_generations": None, "cached_requests": None,
                             "output_tokens": _stats(None), "prompt_tokens_mean": None, "share_truncated": None,
                             "other_finish_reasons": None}
    if "cached" not in have:
        return block
    uniq = df[~df["cached"].astype(bool)]
    block["unique_generations"] = int(len(uniq))
    block["cached_requests"] = int(len(df) - len(uniq))
    if "n_output_tokens" in have:
        lengths = pd.to_numeric(uniq["n_output_tokens"], errors="coerce")
        if capped:
            block["output_tokens_as_generated"] = _stats(lengths)
            lengths = lengths.clip(upper=cap)
        block["output_tokens"] = _stats(lengths)
    if "n_prompt_tokens" in have and len(uniq):
        block["prompt_tokens_mean"] = float(pd.to_numeric(uniq["n_prompt_tokens"]).mean())
    if "finish_reason" in have:
        fr = uniq["finish_reason"]
        block["other_finish_reasons"] = int((~fr.isin(["stop", "length"])).sum())
        if len(uniq):
            if capped and "n_output_tokens" in have:
                n_tok = pd.to_numeric(uniq["n_output_tokens"], errors="coerce")
                block["share_truncated"] = float((n_tok > cap).mean())     # would be cut at the configured cap
                block["share_truncated_at_run_cap"] = float((fr == "length").mean())
            else:
                block["share_truncated"] = float((fr == "length").mean())
    return block


def _shard_meta(path: Path) -> dict[str, str]:
    try:
        md = pq.read_schema(path).metadata or {}
    except Exception as exc:   # unreadable parquet: never guess
        raise TrancheError(f"{path} is unreadable ({exc})") from exc
    out = {}
    for k, v in md.items():
        try:
            out[k.decode()] = v.decode()
        except UnicodeDecodeError:
            continue
    return out


def _find_logs(d: Path) -> list[Path]:
    out: list[Path] = []
    for pat in LOG_GLOBS:
        out.extend(sorted(d.glob(pat)))
    return out


def _cap_override(log: Mapping, effective: Mapping, missing: list[str], warnings: list[str]) -> dict | None:
    """The run's cap override (src/generation.py writes log["cap_override"] for --measure-cap runs), or None
    for a run with the configured caps. Returns {"arms": {arm: run cap}, "configured": {arm: configured cap},
    "effective": {arm: cap in force}, "projection": ...}. A log that names an override it does not describe
    fully is reported (missing field plus a warning); nothing is guessed."""
    co = log.get("cap_override")
    if co is None:
        return None
    arms_raw = co.get("arms") if isinstance(co, Mapping) else None
    arms = {str(a): int(n) for a, n in arms_raw.items() if _pos_int(n)} if isinstance(arms_raw, Mapping) else {}
    if not arms:
        missing.append("cap_override.arms")
        warnings.append("CAP-MEASUREMENT RUN: the run log records a cap_override that this tool cannot read, so the "
                        "configured caps are unknown and the projection uses the run's own caps, which are not the "
                        "registered caps; the projected token counts and GPU-hours are not valid for the registered caps")
        return {"arms": {}, "configured": {}, "effective": dict(effective), "readable": False,
                "projection": "the run's own caps (the override could not be read)"}
    conf_raw = co.get("configured_max_new_tokens")
    configured = {str(a): _pos_int(n) for a, n in conf_raw.items()} if isinstance(conf_raw, Mapping) else {}
    configured = {a: n for a, n in configured.items() if n is not None}
    for a in arms:
        if a not in configured:
            missing.append(f"cap_override.configured_max_new_tokens.{a}")
    desc = ", ".join(f"{a}={n} (configured {configured.get(a, 'not recorded')})" for a, n in sorted(arms.items()))
    warnings.append(f"CAP-MEASUREMENT RUN (--measure-cap): output caps were raised for {desc}. The projection uses the "
                    "configured caps: each unique output length enters the token statistics as min(length, configured "
                    "cap). The tokens-per-second rate is the run's own.")
    for a in sorted(arms):
        if a not in configured:
            warnings.append(f"cap_override: the configured cap of arm {a} is not recorded in the log, so that arm's "
                            f"projection uses the raised cap {arms[a]} and overstates tokens for the configured cap")
    return {"arms": arms, "configured": configured, "effective": dict(effective), "readable": True,
            "projection": "configured caps: output lengths enter as min(length, configured cap)"}


def load_throughput_run(run_dir: str | os.PathLike, *, allow_non_vllm: bool = False) -> dict:
    """Validate one throughput folder and measure it. Raises TrancheError on anything that is not a
    complete throughput-mode run of one model."""
    d = Path(run_dir)
    if not d.is_dir():
        raise TrancheError(f"{d} is not a directory")
    logs = _find_logs(d)
    shard_files = sorted(d.glob("shard-*.parquet"))
    if not logs:
        hint = ""
        if (d / "throughput").is_dir():
            hint = f"; pass {d / 'throughput'}"
        raise TrancheError(f"{d} holds no run log (throughput_log-*.json or run_log-*.json){hint}")

    parsed = []
    for lp in logs:
        log = _read_json(lp)
        if not isinstance(log, Mapping):
            raise TrancheError(f"{lp} is not a JSON object")
        if log.get("mode") != "throughput" or log.get("excluded_from_main") is not True:
            raise TrancheError(
                f"{lp} is not a throughput-mode log (mode={log.get('mode')!r}, "
                f"excluded_from_main={log.get('excluded_from_main')!r}); tranche data are "
                "excluded from the main estimation, and main-run files are never mixed in")
        parsed.append((lp, log))
    for sh in shard_files:
        if _shard_meta(sh).get("throughput") != "true":
            raise TrancheError(f"{sh.name} is not marked as a throughput shard; {d} mixes in main-run output")

    complete = [(lp, lg) for lp, lg in parsed if lg.get("status") == "complete"]
    skipped = [{"log_file": lp.name, "run_id": lg.get("run_id"), "status": lg.get("status")}
               for lp, lg in parsed if lg.get("status") != "complete"]
    if not complete:
        raise TrancheError(f"{d} has no complete throughput run (statuses: {[s['status'] for s in skipped]})")
    if len(complete) > 1:
        raise TrancheError(f"{d} holds {len(complete)} complete runs ({[lg.get('run_id') for _, lg in complete]}); "
                           "keep one run per folder so the tranche has one measurement per model")
    log_path, log = complete[0]
    run_id = log.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TrancheError(f"{log_path} has no run_id")

    listed = log.get("shards_written")
    if isinstance(listed, list) and listed:
        names = [str(n) for n in listed]
    else:
        names = [p.name for p in shard_files if p.name.startswith(f"shard-{run_id}-")]
    paths = [d / n for n in names]
    absent = [p.name for p in paths if not p.is_file()]
    if absent:
        raise TrancheError(f"{d}: shards listed in the run log are missing: {absent}")
    for p in paths:
        md = _shard_meta(p)
        if md.get("run_id") not in (None, run_id):
            raise TrancheError(f"{p.name} belongs to run {md.get('run_id')}, not {run_id}")
        fp, lfp = md.get("run_fingerprint"), log.get("run_fingerprint")
        if fp is not None and lfp is not None and fp != lfp:
            raise TrancheError(f"{p.name} was written under a different run fingerprint than {log_path.name}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()
    expected_rows = _pos_int(log.get("requests_this_run"))
    if expected_rows is not None and len(df) != expected_rows:
        raise TrancheError(f"{d}: the run log expects {expected_rows} requests but the shards hold {len(df)}")
    if "arm" not in df.columns:
        raise TrancheError(f"{d}: shards lack the arm column")
    return _measure(d, df, log, log_path, names, skipped, allow_non_vllm=allow_non_vllm)


def _measure(d: Path, df: pd.DataFrame, log: Mapping, log_path: Path, names: list[str], skipped: list[dict], *,
             allow_non_vllm: bool, pooled: Mapping | None = None) -> dict:
    """The measurement of one validated folder: `log` is its complete run's log, `df` the outputs measured.
    `pooled` (noise-tranche folders) replaces the single run's generation time, throughput and logged counts
    with their sums over the folder's runs."""
    run_id = log.get("run_id")
    missing: list[str] = []
    warnings: list[str] = []
    have = set(df.columns)
    for col in ("cached", "n_output_tokens", "n_prompt_tokens", "finish_reason"):
        if col not in have:
            missing.append(f"shard column {col}")

    model_key = _dig(log, "model", "model_key")
    if not isinstance(model_key, str) or not model_key:
        keys = sorted(df["model_key"].astype(str).unique()) if "model_key" in have else []
        if len(keys) != 1:
            raise TrancheError(f"{d}: the model cannot be identified (no model.model_key in the log, shard model_key {keys})")
        model_key = keys[0]
        missing.append("model.model_key (taken from the shards)")
    backend = _dig(log, "backend", "name")
    if not isinstance(backend, str):
        missing.append("backend.name")
        backend = None
    if backend in TEST_BACKENDS and not allow_non_vllm:
        raise TrancheError(f"{d} was produced by the {backend!r} test backend; its timings are not a measurement "
                           "(pass allow_non_vllm / --allow-non-vllm for a dry run)")
    if backend in TEST_BACKENDS:
        warnings.append(f"backend {backend!r} is a test backend; no figure here is a measurement")

    caps = _dig(log, "decoding", "max_new_tokens")           # the caps in force for this run
    caps = caps if isinstance(caps, Mapping) else {}
    override = _cap_override(log, caps, missing, warnings)
    arms = list(ARMS) + sorted(set(df["arm"].astype(str)) - set(ARMS))
    per_arm: dict[str, dict] = {}
    for a in arms:
        run_cap = _pos_int(caps.get(a))
        if run_cap is None and override:
            run_cap = override["arms"].get(a)
        if override and a in override["arms"]:
            cap = override["configured"].get(a)             # the projection refers to the configured cap
            if cap is None:
                cap = run_cap
        else:
            cap = run_cap
        if cap is None:
            missing.append(f"decoding.max_new_tokens.{a}")
        per_arm[a] = _arm_block(df[df["arm"].astype(str) == a], cap, have, run_cap)
        if cap is not None and run_cap is not None and run_cap < cap:
            warnings.append(f"arm {a}: the run used output cap {run_cap}, below the configured cap {cap}; outputs "
                            f"cut at {run_cap} are longer under the configured cap, so the projected token counts "
                            "for this arm are lower bounds")
        if per_arm[a]["other_finish_reasons"]:
            warnings.append(f"arm {a}: {per_arm[a]['other_finish_reasons']} unique generations have a finish_reason "
                            "other than stop or length")
        logged = (pooled["unique_generations"].get(a) if pooled else _dig(log, "counts", "unique_generations", a))
        if isinstance(logged, int) and per_arm[a]["unique_generations"] not in (None, logged):
            warnings.append(f"arm {a}: shards hold {per_arm[a]['unique_generations']} unique generations, "
                            f"the run log counts {logged}")

    n_requests = int(len(df))
    uniq_total = None if "cached" not in have else int(sum(b["unique_generations"] for b in per_arm.values()))
    out_total = None
    if "cached" in have and "n_output_tokens" in have:
        out_total = float(pd.to_numeric(df.loc[~df["cached"].astype(bool), "n_output_tokens"]).sum())

    gen_s = _num(pooled["generation_s"] if pooled else log.get("generation_s"))
    if gen_s is None:
        missing.append("generation_s")
    if pooled:
        tps = (pooled["output_tokens"] / gen_s) if gen_s and gen_s > 0 else None
        tps_source = (f"pooled over {len(pooled['runs'])} run(s): output tokens generated / generation_s"
                      if tps is not None else None)
    else:
        tps = _num(log.get("output_tokens_per_s"))
        tps_source = "run log" if tps is not None else None
    if tps is None and gen_s is not None and gen_s > 0 and out_total is not None:
        tps, tps_source = out_total / gen_s, "derived: shard output tokens / generation_s"
    if tps is None:
        missing.append("output_tokens_per_s")

    gpu = log.get("gpu") if isinstance(log.get("gpu"), Mapping) else {}
    devices = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
    names = sorted({str(x["name"]) for x in devices if isinstance(x, Mapping) and x.get("name")})
    gpu_name = "; ".join(names) if names else None
    gpu_count = _pos_int(gpu.get("count")) if devices else None
    if gpu_name is None:
        missing.append("gpu.devices[].name")
    if gpu_count is None:
        missing.append("gpu.count")
    tp = _pos_int(_dig(log, "model", "tensor_parallel_size"))
    if tp is None:
        missing.append("model.tensor_parallel_size")
    kernels = _dig(log, "model", "kernels")       # the runner generates in the configured kernel mode
    if not isinstance(kernels, str):
        missing.append("model.kernels")
        kernels = None

    return {
        "model_key": model_key, "hf_id": _dig(log, "model", "hf_id"), "revision": _dig(log, "model", "revision"),
        "run_id": run_id, "run_dir": str(d), "log_file": log_path.name, "log_sha256": _sha256_file(log_path),
        "backend": backend, "started_utc": log.get("started_utc"), "versions_vllm": _dig(log, "versions", "vllm"),
        "decoding_status": _dig(log, "decoding", "status"), "decoding_confirmed": log.get("decoding_confirmed"),
        "gpu_name": gpu_name, "gpu_count": gpu_count, "tensor_parallel_size": tp, "kernels": kernels,
        "generation_s": gen_s, "model_load_s": _num(log.get("model_load_s")),
        "output_tokens_per_s": tps, "output_tokens_per_s_source": tps_source,
        "requests": n_requests, "unique_generations": uniq_total, "output_tokens_generated": out_total,
        "requests_per_s": (n_requests / gen_s) if gen_s and gen_s > 0 else None,
        "unique_generations_per_s": (uniq_total / gen_s) if gen_s and gen_s > 0 and uniq_total is not None else None,
        "per_arm": per_arm, "cap_override": override, "shards": names, "skipped_runs": skipped,
        "source": "noise_tranche" if pooled else "throughput",
        "pooled_runs": list(pooled["runs"]) if pooled else None,
        "missing_fields": sorted(set(missing)), "warnings": warnings,
    }


def load_tranche_records(requests_dir: str | os.PathLike) -> dict[str, dict]:
    """requests_sha256 -> {"model_key", "record_path", "n_requests"} from <requests_dir>/tranche/build_record.json,
    which must be a tranche build record on the training split."""
    p = Path(requests_dir) / "tranche" / "build_record.json"
    if not p.is_file():
        raise TrancheError(f"{p} does not exist; a noise-tranche folder is matched to the build record of its "
                           "request file")
    rec = _read_json(p)
    if not isinstance(rec, Mapping) or rec.get("kind") != "tranche" or rec.get("split") != "train":
        got = (rec.get("kind"), rec.get("split")) if isinstance(rec, Mapping) else None
        raise TrancheError(f"{p} is not a tranche build record on the training split (kind, split = {got})")
    out = {}
    for key, ent in (rec.get("models") or {}).items():
        sha = ent.get("requests_sha256") if isinstance(ent, Mapping) else None
        if isinstance(sha, str) and sha:
            out[sha] = {"model_key": key, "record_path": str(p), "n_requests": _pos_int(ent.get("n_requests"))}
    return out


def load_tranche_run(run_dir: str | os.PathLike, records: Mapping[str, Mapping], *,
                     allow_non_vllm: bool = False) -> dict:
    """Validate one noise-tranche folder (an ordinary run on tranche requests) and measure it. `records` comes
    from load_tranche_records. Raises TrancheError on anything that is not a complete run of one model on a
    request file of the tranche build record."""
    d = Path(run_dir)
    if not d.is_dir():
        raise TrancheError(f"{d} is not a directory")
    if d.name == "throughput" or any(d.glob("throughput_log-*.json")):
        raise TrancheError(f"{d} is a throughput folder; pass it with --run-dir")
    logs = sorted(d.glob("run_log-*.json"))
    if not logs:
        raise TrancheError(f"{d} holds no run_log-*.json")
    parsed = []
    for lp in logs:
        log = _read_json(lp)
        if not isinstance(log, Mapping):
            raise TrancheError(f"{lp} is not a JSON object")
        if log.get("mode") != "full":
            raise TrancheError(f"{lp} is not a full-mode log (mode={log.get('mode')!r})")
        parsed.append((lp, log))
    shas = {_dig(lg, "inputs", "requests_sha256") for _, lg in parsed}
    if len(shas) != 1 or None in shas:
        raise TrancheError(f"{d}: its runs used {len(shas)} request files (or a log lacks the hash); one folder "
                           "belongs to one request file")
    sha = next(iter(shas))
    rec = records.get(sha)
    if rec is None:
        raise TrancheError(f"{d}: its request file (sha256 {sha[:12]}...) is not in the tranche build record, so the "
                           "folder is not a noise-tranche run")
    keys = {_dig(lg, "model", "model_key") for _, lg in parsed}
    if keys != {rec["model_key"]}:
        raise TrancheError(f"{d}: run logs name model(s) {sorted(map(str, keys))}, the build record "
                           f"{rec['model_key']!r}")
    complete = [(lp, lg) for lp, lg in parsed
                if lg.get("status") == "complete" and _dig(lg, "inputs", "limit") is None]
    if not complete:
        raise TrancheError(f"{d} has no complete run without --limit "
                           f"(statuses: {[lg.get('status') for _, lg in parsed]})")
    log_path, log = max(complete, key=lambda c: (str(c[1].get("started_utc") or ""), c[0].name))
    gpus = {tuple(sorted({str(x.get("name")) for x in (_dig(lg, "gpu", "devices") or []) if isinstance(x, Mapping)}))
            for _, lg in parsed if lg.get("shards_written")}
    if len(gpus) > 1:
        raise TrancheError(f"{d}: its outputs were generated on {len(gpus)} GPU models {sorted(gpus)}")

    shard_files = sorted(d.glob("shard-*.parquet"))
    if not shard_files:
        raise TrancheError(f"{d} holds no shard-*.parquet")
    fps = set()
    for sh in shard_files:
        md = _shard_meta(sh)
        if md.get("throughput") == "true":
            raise TrancheError(f"{sh.name} is marked as a throughput shard; pass the folder with --run-dir")
        fps.add(md.get("run_fingerprint"))
    if len(fps) != 1:
        raise TrancheError(f"{d}: shards were written under {len(fps)} run fingerprints")
    df = pd.concat([pd.read_parquet(p) for p in shard_files], ignore_index=True)
    if "arm" not in df.columns or "request_id" not in df.columns:
        raise TrancheError(f"{d}: shards lack the arm or request_id column")
    if df["request_id"].duplicated().any():
        raise TrancheError(f"{d}: request_id is duplicated across shards")
    n_file = _pos_int(_dig(log, "inputs", "requests_in_file")) or rec.get("n_requests")
    if n_file is not None and len(df) != n_file:
        raise TrancheError(f"{d}: the request file holds {n_file} requests but the shards hold {len(df)}")

    runs, gen_s, out_tok, unique = [], 0.0, 0.0, {}
    unpooled = []
    for lp, lg in parsed:
        g_s, o_t = _num(lg.get("generation_s")), _num(_dig(lg, "totals", "output_tokens_generated"))
        if g_s is None or o_t is None:
            if lg.get("shards_written"):
                unpooled.append(lp.name)
            continue
        runs.append({"log_file": lp.name, "run_id": lg.get("run_id"), "status": lg.get("status"),
                     "generation_s": g_s, "output_tokens_generated": o_t})
        gen_s += g_s
        out_tok += o_t
        for a, n in (_dig(lg, "counts", "unique_generations") or {}).items():
            if isinstance(n, int) and not isinstance(n, bool):
                unique[a] = unique.get(a, 0) + n
    skipped = [{"log_file": lp.name, "run_id": lg.get("run_id"), "status": lg.get("status")}
               for lp, lg in parsed if lp != log_path]
    m = _measure(d, df, log, log_path, [p.name for p in shard_files], skipped, allow_non_vllm=allow_non_vllm,
                 pooled={"generation_s": gen_s if runs else None, "output_tokens": out_tok, "runs": runs,
                         "unique_generations": unique})
    if unpooled:
        m["warnings"].append(f"run logs {unpooled} wrote shards but lack generation_s or output tokens; their "
                             "generation is in the token statistics and not in the throughput")
    m["requests_sha256"] = sha
    m["build_record"] = rec["record_path"]
    return m


# --------------------------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------------------------

def _max_or_none(values: Sequence[float | None]) -> float | None:
    vals = list(values)
    return None if not vals or any(v is None for v in vals) else max(vals)


def _model_gpu_hours(m: Mapping, unique_per_arm: float) -> float | None:
    """sum over arms of (unique generations of the arm x mean output tokens) / tokens-per-second / 3600 x GPUs.
    None when the throughput, the GPUs per job or an arm mean is missing."""
    tps, tp = m.get("output_tokens_per_s"), m.get("tensor_parallel_size")
    if tps is None or tps <= 0 or tp is None:
        return None
    hours = 0.0
    for a in ARMS:
        mean = m["per_arm"][a]["output_tokens"]["mean"]
        if mean is None:
            return None
        hours += unique_per_arm * mean / tps / 3600.0 * tp
    return hours


def project_rows(measured: Sequence[Mapping], *, models: int, items: int, bank: Bank,
                 usd_per_gpu_hour: float | None, caps: Mapping, ks: Sequence[int] = K_GRID) -> list[dict]:
    mean_out = {a: _max_or_none([m["per_arm"][a]["output_tokens"]["mean"] for m in measured]) for a in ARMS}
    mean_in = {a: _max_or_none([m["per_arm"][a]["prompt_tokens_mean"] for m in measured]) for a in ARMS}
    rows = []
    for k in ks:
        draws = 2 * k
        e = bank.expected_distinct(draws)
        unique = models * items * 2 * e
        per_arm = unique / 2.0
        out_by_arm = {a: (per_arm * mean_out[a] if mean_out[a] is not None else None) for a in ARMS}
        in_by_arm = {a: (per_arm * mean_in[a] if mean_in[a] is not None else None) for a in ARMS}
        by_model = {m["model_key"]: _model_gpu_hours(m, per_arm) for m in measured}
        gpu_hours = _max_or_none(list(by_model.values()))
        binding = None
        if gpu_hours is not None:
            binding = next(key for key, v in by_model.items() if v == gpu_hours)
        # With the whole panel measured, each model can also be projected at its own figures: by_model[k] is the
        # panel as if every model were k, so a model's own share is by_model[k] / models.
        own_sum = None
        if len(measured) == models and all(v is not None for v in by_model.values()):
            own_sum = math.fsum(by_model.values()) / models
        cost = gpu_hours * usd_per_gpu_hour if gpu_hours is not None and usd_per_gpu_hour is not None else None
        summary, detail = compare_cap(gpu_hours, cost, caps)
        rows.append({
            "K": int(k), "expected_distinct_units": e,
            "generations_before_caching": generations_before_caching(models, items, k),
            "unique_generations": unique,
            "output_tokens": None if any(v is None for v in out_by_arm.values()) else math.fsum(out_by_arm.values()),
            "gpu_hours": gpu_hours, "cost_usd": cost, "within_cap": summary,
            "output_tokens_by_arm": out_by_arm,
            "prompt_tokens": None if any(v is None for v in in_by_arm.values()) else math.fsum(in_by_arm.values()),
            "gpu_hours_by_model": by_model, "gpu_hours_binding_model": binding,
            "gpu_hours_sum_of_own": own_sum, "cap_detail": detail})
    return rows


def _assumptions(measured: Sequence[Mapping], bank: Bank, rate: float | None) -> list[str]:
    n = len(measured)
    out = [
        "Every panel model is assumed to run at the measured model's output lengths and throughput. "
        "The GPU-hour and dollar figures are projections, not established figures.",
        ("One model was measured." if n == 1 else
         f"{n} models were measured. Output and prompt tokens use the per-arm maximum of the measured means. "
         "GPU-hours use each measured model's own means, throughput and GPUs per job as if all panel models "
         "behaved like it, and the table reports the largest of those figures."),
        "GPU-hours = unique generations of the arm x mean output tokens of the arm / measured output tokens per "
        "second / 3600 x tensor_parallel_size, summed over the two arms. The run log records generation wall time "
        "for the mixed batch only, so a per-arm request rate is not available. The measured rate includes prefill "
        "and is applied to both arms.",
        "Output caps and decoding are those of the tranche run. Truncated outputs count at their capped length.",
        "Model loading, queueing, scoring, controls and reproduction runs are not included.",
    ]
    if n > 1:
        out.append("When every panel model was measured, gpu_hours_sum_of_own projects each model at its own means, "
                   "throughput and GPUs per job and sums the models. The cap is compared with gpu_hours, the larger "
                   "figure.")
    pooled = [m["model_key"] for m in measured if m.get("source") == "noise_tranche"]
    if pooled:
        out.append("Noise-tranche folders (ordinary runs on tranche requests): " + ", ".join(pooled) + ". Their "
                   "throughput pools the output tokens and generation time of every run in the folder, including "
                   "runs interrupted by preemption.")
    overridden = [m["model_key"] for m in measured if m.get("cap_override")]
    if overridden:
        out.append("Cap-measurement run(s) (--measure-cap) for " + ", ".join(overridden) + ": the projection uses the "
                   "configured output caps. Each unique output length enters as min(length, configured cap), and the "
                   "tokens-per-second rate is the run's own rate, measured while the longer outputs were generated. "
                   "For these models the cap in the sentence above is the configured cap of each overridden arm, "
                   "not the raised cap of the run.")
    if bank.kind == "none":
        out.append("No bank was available, so every draw is taken as distinct and caching is not credited "
                   "(unique generations equal generations before caching).")
    else:
        out.append(f"Unique generations use the expected number of distinct configurations under {bank.source}. "
                   "Identical rendered prompts from different configurations would lower the figure further and "
                   "are not credited.")
    if rate is None:
        out.append("No price per GPU-hour was given, so cost is not computed.")
    else:
        out.append(f"Cost = GPU-hours x {rate:g} USD per GPU-hour, as given on the command line.")
    return out


def build_report(run_dirs: Sequence[str | os.PathLike], *, models: int = DEFAULT_MODELS, items: int = DEFAULT_ITEMS,
                 bank: Bank | None = None, usd_per_gpu_hour: float | None = None, caps: Mapping | None = None,
                 allow_non_vllm: bool = False, tranche_dirs: Sequence[str | os.PathLike] = (),
                 requests_dir: str | os.PathLike | None = None) -> dict:
    """Validate every folder, then build the report. Nothing is written here. `run_dirs` are throughput folders,
    `tranche_dirs` noise-tranche folders matched to the tranche build record under `requests_dir`."""
    if not run_dirs and not tranche_dirs:
        raise TrancheError("at least one throughput folder (--run-dir) or noise-tranche folder (--tranche-dir) "
                           "is required")
    for name, v in (("models", models), ("items", items)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise TrancheError(f"{name} must be a positive integer")
    if usd_per_gpu_hour is not None and (_num(usd_per_gpu_hour) is None or usd_per_gpu_hour < 0):
        raise TrancheError("usd_per_gpu_hour must be a nonnegative number")
    resolved = [str(Path(p).resolve()) for p in list(run_dirs) + list(tranche_dirs)]
    if len(set(resolved)) != len(resolved):
        raise TrancheError("the same folder was given twice")
    measured = [load_throughput_run(p, allow_non_vllm=allow_non_vllm) for p in run_dirs]
    if tranche_dirs:
        records = load_tranche_records(requests_dir if requests_dir is not None else REQUESTS_DIR)
        measured += [load_tranche_run(p, records, allow_non_vllm=allow_non_vllm) for p in tranche_dirs]
    keys = [m["model_key"] for m in measured]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise TrancheError(f"model(s) {dup} appear in more than one folder; one measurement per model")
    bank = bank or resolve_bank()
    caps = dict(caps) if caps is not None else load_caps()
    rows = project_rows(measured, models=models, items=items, bank=bank, usd_per_gpu_hour=usd_per_gpu_hour, caps=caps)
    warnings = [f"{m['model_key']}: {w}" for m in measured for w in m["warnings"]]
    if bank.note:
        warnings.append(bank.note)
    for m in measured:
        if m["missing_fields"]:
            warnings.append(f"{m['model_key']}: run log lacks {', '.join(m['missing_fields'])}")
    cap_override = {m["model_key"]: m["cap_override"] for m in measured if m.get("cap_override")}
    return {
        "schema_version": SCHEMA_VERSION,
        "cap_override": ({"models": cap_override,
                          "projection_uses": "configured caps: each unique output length enters as min(length, "
                                             "configured cap); a raised cap from --measure-cap is not projected"}
                         if cap_override else None),
        "measured": {
            "models": measured,
            "projection_basis": "tokens_per_second",
            "max_of_means_output_tokens": {a: _max_or_none([m["per_arm"][a]["output_tokens"]["mean"] for m in measured])
                                           for a in ARMS},
            "max_of_means_prompt_tokens": {a: _max_or_none([m["per_arm"][a]["prompt_tokens_mean"] for m in measured])
                                           for a in ARMS},
        },
        "assumptions": _assumptions(measured, bank, usd_per_gpu_hour),
        "inputs": {
            "models": models, "items": items, "arms": list(ARMS), "units_per_arm_grid": list(K_GRID),
            "units_total_per_K": "2K (K discovery plus K validation units per arm; one unit is one configuration in both arms)",
            "bank": bank.to_log(), "usd_per_gpu_hour": usd_per_gpu_hour,
            "caps": {k: caps.get(k) for k in CAP_KEYS} | {"source": caps.get("source")},
            "allow_non_vllm": bool(allow_non_vllm),
            "run_dirs": [{"path": m["run_dir"], "run_id": m["run_id"], "log_file": m["log_file"],
                          "log_sha256": m["log_sha256"]} for m in measured]},
        "rows": rows,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

def _n(x: float | None, nd: int = 0, missing: str = "missing") -> str:
    return missing if x is None else f"{x:,.{nd}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render_markdown(report: Mapping) -> str:
    inp, meas = report["inputs"], report["measured"]
    lines = ["# Dataset 2 throughput tranche: cost projection", "",
             "The tranche is excluded from the main estimation. The GPU-hour and dollar figures are "
             "projections from the measured model(s) under the assumptions listed at the end.", "",
             "## Measured", ""]
    for m in meas["models"]:
        lines += [f"### {m['model_key']}", ""]
        ckpt = m.get("hf_id") or "checkpoint not recorded"
        rev = f" at {m['revision'][:12]}" if isinstance(m.get("revision"), str) else ""
        lines += [f"- Run {m['run_id']}, backend {m.get('backend') or 'missing'}; {ckpt}{rev}.",
                  f"- Hardware: {_n(m['gpu_count'])} x {m['gpu_name'] or 'missing'}; "
                  f"tensor_parallel_size {_n(m['tensor_parallel_size'])}; kernels {m.get('kernels') or 'missing'}.",
                  f"- Generation wall time {_n(m['generation_s'], 1)} s for {_n(m['unique_generations'])} unique "
                  f"generations from {m['requests']:,} requests; model load {_n(m['model_load_s'], 1)} s.",
                  f"- Throughput: {_n(m['output_tokens_per_s'], 1)} output tokens per second, "
                  f"{_n(m['unique_generations_per_s'], 2)} unique generations per second, "
                  f"{_n(m['requests_per_s'], 2)} requests per second."]
        if m.get("source") == "noise_tranche":
            lines.append(f"- Noise-tranche folder (an ordinary run on tranche requests); throughput "
                         f"{m.get('output_tokens_per_s_source') or 'missing'}.")
        if m.get("decoding_status") is not None:
            lines.append(f"- Decoding status in the run: {m['decoding_status']}.")
        co = m.get("cap_override")
        if co:
            ran = ", ".join(f"{a} {n}" for a, n in sorted(co["arms"].items())) or "unreadable"
            conf = ", ".join(f"{a} {n}" for a, n in sorted(co["configured"].items())) or "not recorded"
            lines.append(f"- Cap-measurement run (--measure-cap): output caps in the run were {ran}; configured caps "
                         f"{conf}. The Cap column and the token statistics below refer to the configured caps "
                         "(each output length enters as min(length, configured cap)), and so does the projection.")
        lines.append("")
        rows = []
        for a, b in m["per_arm"].items():
            o = b["output_tokens"]
            rows.append([a, _n(b["cap"]), f"{b['requests']:,}", _n(b["unique_generations"]), _n(o["mean"], 1),
                         _n(o["median"], 1), _n(o["p95"], 1), _n(o["max"]), _n(b["prompt_tokens_mean"], 1),
                         "missing" if b["share_truncated"] is None else f"{b['share_truncated']:.3f}"])
        lines += _table(["Arm", "Cap", "Requests", "Unique", "Mean out", "Median out", "p95 out", "Max out",
                         "Mean prompt", "Truncated"], rows)
        lines += ["", "Token statistics cover unique generations. Truncated is the share with finish_reason length.", ""]
        if m["missing_fields"]:
            lines += [f"Missing from this run: {', '.join(m['missing_fields'])}.", ""]
        for w in m["warnings"]:
            lines += [f"Warning: {w}", ""]
    lines += ["## Projection", "",
              f"M = {inp['models']} models, {inp['items']:,} items, two arms. Bank: {inp['bank']['source']}.", ""]
    rows = []
    for r in report["rows"]:
        gh = r["gpu_hours"]
        cost = "no price given" if inp["usd_per_gpu_hour"] is None else _n(r["cost_usd"], 0)
        rows.append([str(r["K"]), str(2 * r["K"]), _n(r["expected_distinct_units"], 2),
                     _n(r["generations_before_caching"]), _n(r["unique_generations"]), _n(r["output_tokens"]),
                     _n(gh, 1), cost, r["within_cap"]])
    lines += _table(["K per arm", "Units", "Expected distinct", "Generations before caching", "Unique generations",
                     "Output tokens", "GPU-hours", "Cost (USD)", "Cap"], rows)
    caps = inp["caps"]
    lines += ["", "Caps in " + Path(str(caps.get("source") or "config/budget.yaml")).name + ": GPU-hours "
              + ("not set" if caps["gpu_hours"] is None else f"{caps['gpu_hours']:,g}") + ", money "
              + ("not set" if caps["money_usd"] is None else f"{caps['money_usd']:,g} USD")
              + ". A value equal to a cap counts as within it.", ""]
    multi = len(meas["models"]) > 1
    if multi:
        lines += ["GPU-hours by measured model:", ""]
        names = [m["model_key"] for m in meas["models"]]
        lines += _table(["K per arm"] + names, [[str(r["K"])] + [_n(r["gpu_hours_by_model"][k], 1) for k in names]
                                                for r in report["rows"]])
        lines.append("")
        if any(r.get("gpu_hours_sum_of_own") is not None for r in report["rows"]):
            lines += ["Every panel model was measured. Each projected at its own figures, the panel takes "
                      + "; ".join(f"K = {r['K']}: {_n(r.get('gpu_hours_sum_of_own'), 1)}" for r in report["rows"])
                      + " GPU-hours. The cap column compares the larger figure above.", ""]
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    lines += ["## Assumptions", ""] + [f"- {a}" for a in report["assumptions"]] + [""]
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_outputs(report: Mapping, out_dir: str | os.PathLike | None = None) -> dict[str, Path]:
    """cost_projection.json, cost_projection.md and the identical cost_projection_clean.md."""
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report)
    paths = {"json": out / "cost_projection.json", "md": out / "cost_projection.md",
             "clean_md": out / "cost_projection_clean.md"}
    _write_text(paths["json"], json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_text(paths["md"], md)
    _write_text(paths["clean_md"], md)
    return paths
